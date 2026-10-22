"""Deep Space Tracker. Run with: python -m streamlit run app.py"""
import bisect
import csv
import html
import io
import json
import math
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import streamlit as st

MISSIONS = {"Voyager 1": "-31", "Voyager 2": "-32"}
API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
AU_KM = 149_597_870.7
KM_PER_MILE = 1.609344
LIGHT_SPEED = 299_792.458
UTC = timezone.utc


def fetch_trajectory(mission_id, center, start):
    """Download 20 minutes of geometric UTC position samples from JPL."""
    settings = {
        "COMMAND": mission_id, "OBJ_DATA": "NO", "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS", "CENTER": center,
        "START_TIME": start.strftime("%Y-%m-%d %H:%M:%S"),
        "STOP_TIME": (start + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S"),
        "STEP_SIZE": "1 min", "TIME_TYPE": "UT", "OUT_UNITS": "KM-S",
        "VEC_TABLE": "2", "VEC_CORR": "NONE", "CSV_FORMAT": "YES",
    }
    params = {k: "'" + v + "'" for k, v in settings.items()}
    params["format"] = "json"
    request = urllib.request.Request(
        API_URL + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": "DeepSpaceTracker/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    result = payload.get("result", "")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise RuntimeError("JPL returned no trajectory for this date.")
    if f"({mission_id})" not in result:
        raise RuntimeError("JPL returned an unexpected spacecraft.")
    samples = []
    body = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    for row in csv.reader(io.StringIO(body.strip())):
        if not row or not row[0].strip():
            continue
        # Julian-day UTC column avoids locale-dependent calendar parsing.
        timestamp = (float(row[0]) - 2440587.5) * 86400
        position = tuple(float(v) for v in row[2:5])
        if len(position) != 3 or not all(math.isfinite(v) for v in position):
            raise RuntimeError("JPL returned invalid coordinates.")
        samples.append((timestamp, position))
    if len(samples) < 2 or any(b[0] <= a[0] for a, b in zip(samples, samples[1:])):
        raise RuntimeError("JPL returned insufficient or unordered samples.")
    return samples


def distance_at(samples, timestamp):
    """Interpolate coordinates inside downloaded coverage, then find distance."""
    if not samples[0][0] <= timestamp <= samples[-1][0]:
        raise ValueError("Prediction coverage expired.")
    index = min(bisect.bisect_right([s[0] for s in samples], timestamp) - 1,
                len(samples) - 2)
    t0, p0 = samples[index]
    t1, p1 = samples[index + 1]
    fraction = (timestamp - t0) / (t1 - t0)
    return math.sqrt(sum((a + fraction * (b - a)) ** 2 for a, b in zip(p0, p1)))


class Tracker:
    """One cache shared by visitors; serialize all JPL API requests."""
    def __init__(self):
        self.lock = threading.Lock()
        self.records = {name: {"tracks": None, "next_fetch": 0,
                               "downloaded": None, "error": ""}
                        for name in MISSIONS}

    def snapshot(self):
        with self.lock:
            for name, mission_id in MISSIONS.items():
                record = self.records[name]
                if time.monotonic() < record["next_fetch"]:
                    continue
                start = datetime.now(UTC).replace(second=0, microsecond=0)
                try:
                    earth = fetch_trajectory(mission_id, "500@399", start)
                    sun = fetch_trajectory(mission_id, "500@10", start)
                    record.update(tracks=(earth, sun), downloaded=datetime.now(UTC),
                                  error="", next_fetch=time.monotonic() + 600)
                except Exception:
                    # Retain good cached data; never substitute made-up distances.
                    record.update(error="Could not reach JPL or obtain a valid trajectory. Retrying in one minute.",
                                  next_fetch=time.monotonic() + 60)
            return {name: dict(record) for name, record in self.records.items()}


@st.cache_resource
def get_tracker():
    return Tracker()


def formatted_distance(km, unit):
    if unit == "Miles":
        return f"{km / KM_PER_MILE:,.0f}", "mi"
    if unit == "AU":
        return f"{km / AU_KM:,.6f}", "AU"
    return f"{km:,.0f}", "km"


def main():
    st.set_page_config(page_title="Deep Space Tracker", page_icon="🛰️", layout="wide")
    st.markdown('''<style>
    .stApp {background: radial-gradient(ellipse at top left,#162b48 0%,#080e1b 65%);color:#edf3fc;}
    .block-container {max-width:1200px;padding-top:3rem;}
    h1,h2,h3 {color:#edf3fc!important;}
    .eyebrow {letter-spacing:.25em;font-size:.75rem;color:#8fb5de;}
    .mission-card {background:linear-gradient(140deg,#15243b,#0e192b);border:1px solid #30445e;
      border-radius:20px;padding:28px;margin:8px 0 20px;min-height:330px;color:#edf3fc;}
    .distance {font-size:clamp(1.5rem,3vw,2.6rem);font-weight:700;letter-spacing:-.03em;margin:20px 0 4px;}
    .muted {color:#a9bcd3;font-size:.88rem;}
    .accent {color:#a3e7cb;}
    .mission-card h2 {margin:0 0 4px;}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">BEYOND THE PLANETS</div>', unsafe_allow_html=True)
    st.title("Deep Space Tracker")
    st.write("Follow Voyager 1 and Voyager 2 as they travel outward through space.")
    a, b = st.columns(2)
    with a:
        reference = st.radio("Measure distance from", ["Sun", "Earth"], horizontal=True)
    with b:
        unit = st.radio("Distance units", ["Kilometers", "Miles", "AU"], horizontal=True)
    st.caption("CURRENT-TIME TRAJECTORY ESTIMATES · Display refreshes every second while this page is active")

    @st.fragment(run_every="1s")
    def live_cards():
        records = get_tracker().snapshot()
        now = datetime.now(UTC)
        st.caption(now.strftime("%Y-%m-%d · %H:%M:%S UTC"))
        for column, (name, record) in zip(st.columns(2), records.items()):
            with column:
                tracks = record["tracks"]
                if tracks is None:
                    st.subheader(name)
                    st.warning(record["error"] or "Waiting for NASA/JPL data.")
                    continue
                # Freeze at the final covered epoch if no fresh download is available.
                end = min(track[-1][0] for track in tracks)
                start = max(track[0][0] for track in tracks)
                timestamp = now.timestamp()
                stale = not start <= timestamp <= end
                covered_time = max(start, min(timestamp, end))
                earth, sun = [distance_at(track, covered_time) for track in tracks]
                km = sun if reference == "Sun" else earth
                number, suffix = formatted_distance(km, unit)
                status = "STALE · distance frozen" if stale else "● Updating prediction"
                downloaded = record["downloaded"].strftime("%H:%M:%S UTC")
                st.markdown(f'''<div class="mission-card">
                  <h2>{html.escape(name)}</h2><div class="muted">Distance from the {reference}'s center</div>
                  <div class="distance">{number} <span style="font-size:1rem">{suffix}</span></div>
                  <div class="muted">{km / AU_KM:,.4f} astronomical units</div>
                  <hr style="border-color:#30445e;margin:24px 0">
                  <div class="muted">Approximate one-way light time from Earth</div>
                  <div style="font-size:1.4rem;margin:4px 0 20px">{earth / LIGHT_SPEED / 3600:.2f} hours</div>
                  <div class="accent">{status}</div>
                  <div class="muted">Trajectory downloaded {downloaded}</div>
                </div>''', unsafe_allow_html=True)
                if stale:
                    frozen_at = datetime.fromtimestamp(covered_time, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                    st.warning("Predictions expired. Showing the last covered time: " + frozen_at)
                if record["error"]:
                    st.warning(record["error"])
    live_cards()
    with st.expander("Where do these numbers come from?"):
        st.write("NASA/JPL Horizons supplies predicted spacecraft coordinates. This app downloads "
                 "one-minute samples covering 20 minutes and refreshes them every 10 minutes. "
                 "Between samples, it interpolates the coordinates and calculates the distance. "
                 "These are model estimates, not a live telemetry feed. Displayed digits do not "
                 "indicate measurement accuracy. Light time is geometric Earth distance divided "
                 "by the speed of light; it is an approximation. Earth distance may temporarily "
                 "decrease because Earth orbits the Sun. No API key is needed.")
        st.markdown("[NASA/JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) · "
                    "[API documentation](https://ssd-api.jpl.nasa.gov/doc/horizons.html)")
    st.caption("Made with Python · Voyager 1 + Voyager 2 · Distances to centers of Earth and Sun")


if __name__ == "__main__":
    main()
