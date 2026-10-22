"""Deep Space Tracker. Run with: python -m streamlit run app.py"""
import bisect
import csv
import html
import io
import json
import math
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import streamlit as st

MISSIONS = {
    "Voyager 1": "-31",
    "Voyager 2": "-32",
    "Pioneer 10": "-23",
    "Pioneer 11": "-24",
    "New Horizons": "-98",
}
MISSION_NOTES = {
    "Voyager 1": "Outbound · Launched 1977",
    "Voyager 2": "Outbound · Launched 1977",
    "Pioneer 10": "Silent · Launched 1972 · Predicted position",
    "Pioneer 11": "Silent · Launched 1973 · Predicted position",
    "New Horizons": "Outbound · Launched 2006",
}
# ---------- EDIT THE WEBSITE'S APPEARANCE HERE ----------
SITE_TITLE = "The New Pioneers"
HERO_TITLE = "The Frontier Has Moved."
HERO_TEXT = (
    "Once, expeditions like Lewis and Clark's traced rivers and crossed continents. "
    "Today, our explorers travel beyond the planets. Five spacecraft carry that "
    "same impulse into the dark: to go farther, and discover what lies ahead."
)
BACKGROUND_COLOR = "#090f16"
CARD_COLOR = "#111c27"
ACCENT_COLOR = "#d8b779"
TEXT_COLOR = "#f4f0e7"

# Optional: upload images to GitHub and enter their file paths below.
# For example, HERO_IMAGE = "images/space.jpg"
# Leave as None to use the built-in design without any image downloads.
HERO_IMAGE = None
MISSION_IMAGES = {
    "Voyager 1": None,  # Example: "images/voyager.jpg"
    "Voyager 2": None,
    "Pioneer 10": None,
    "Pioneer 11": None,
    "New Horizons": None,
}
# ---------- END OF APPEARANCE SETTINGS ----------


def show_optional_image(source, caption=None):
    """Display a configured image. Relative paths start beside this app.py."""
    if not source:
        return
    if source.startswith(("https://", "http://")):
        st.image(source, caption=caption, width="stretch")
        return
    path = Path(__file__).resolve().parent / source
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")


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
    st.set_page_config(page_title=SITE_TITLE, page_icon="✦", layout="wide")
    # CSS controls color, spacing, typography, and card design.
    css = """<style>
    .stApp {background:radial-gradient(ellipse at 85% 5%,#1b2c3b 0%,transparent 45%),BACKGROUND;color:TEXT;}
    .block-container {max-width:1180px;padding-top:1.4rem;}
    h1,h2,h3 {color:TEXT!important;}
    .eyebrow {letter-spacing:.22em;font-size:.73rem;color:ACCENT;text-transform:uppercase;}
    .masthead {border-bottom:1px solid #34414c;padding-bottom:12px;margin-bottom:22px;
      display:flex;justify-content:space-between;gap:20px;color:TEXT;font-size:.8rem;letter-spacing:.14em;}
    .hero {padding:0 0 16px;max-width:850px;}
    .hero h1 {font-family:Georgia,serif;font-size:clamp(2rem,4vw,3rem);font-weight:400;
      line-height:1.1;letter-spacing:-.035em;margin:10px 0 12px;}
    .hero p {max-width:780px;font-size:.95rem;line-height:1.55;color:#bdc7ce;}
    .trail {width:60px;height:2px;background:ACCENT;margin:14px 0;}
    .section-label {border-top:1px solid #34414c;padding-top:14px;margin:8px 0 14px;
      color:ACCENT;font-size:.78rem;letter-spacing:.15em;}
    .mission-card {background:CARD;border:1px solid #34414c;border-top:2px solid ACCENT;
      border-radius:4px;padding:28px;margin:8px 0 12px;min-height:355px;color:TEXT;}
    .distance {font-size:clamp(1.5rem,3vw,2.6rem);font-weight:500;letter-spacing:-.03em;
      margin:20px 0 4px;font-variant-numeric:tabular-nums;}
    .muted {color:#a9b9c7;font-size:.85rem;line-height:1.65;}
    .accent {color:ACCENT;font-size:.8rem;}
    .mission-card h2 {font-family:Georgia,serif;font-size:2rem;font-weight:400;margin:0 0 8px;}
    .closing {border-top:1px solid #34414c;margin-top:45px;padding:24px 0 8px;
      font-family:Georgia,serif;font-style:italic;font-size:1.3rem;color:ACCENT;}
    @media(max-width:600px){.masthead{font-size:.65rem;}.hero{padding-bottom:12px;}.mission-card{padding:22px;}}
    </style>"""
    for token, value in {"BACKGROUND": BACKGROUND_COLOR, "CARD": CARD_COLOR,
                         "ACCENT": ACCENT_COLOR, "TEXT": TEXT_COLOR}.items():
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(f'<div class="masthead"><span>✦ {html.escape(SITE_TITLE.upper())}</span>'
                '<span>AN ATLAS OF THE OUTWARD JOURNEY</span></div>', unsafe_allow_html=True)
    st.markdown(f'<section class="hero"><div class="eyebrow">The next age of exploration</div>'
                f'<h1>{html.escape(HERO_TITLE)}</h1><div class="trail"></div>'
                f'<p>{html.escape(HERO_TEXT)}</p></section>', unsafe_allow_html=True)
    show_optional_image(HERO_IMAGE)
    st.markdown('<div class="section-label">THE EXPEDITION · FIVE SPACECRAFT, STILL TRAVELING</div>',
                unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        reference = st.radio("Measure distance from", ["Earth", "Sun"], index=0, horizontal=True, key="distance_reference_v2")
    with b:
        unit = st.radio("Distance units", ["Kilometers", "Miles", "AU"], horizontal=True)
    st.caption("NASA/JPL trajectory estimates · Distances update every second while this page is active")

    @st.fragment(run_every="1s")
    def live_cards():
        records = get_tracker().snapshot()
        now = datetime.now(UTC)
        st.caption(now.strftime("%Y-%m-%d · %H:%M:%S UTC"))
        items = list(records.items())
        for offset in range(0, len(items), 2):
            columns = st.columns(2)
            for column, (name, record) in zip(columns, items[offset:offset + 2]):
                with column:
                    show_optional_image(MISSION_IMAGES.get(name))
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
                      <h2>{html.escape(name)}</h2>
                      <div class="muted">{html.escape(MISSION_NOTES[name])}</div>
                      <div class="muted">Distance from the {reference}'s center</div>
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
                    if name in ("Pioneer 10", "Pioneer 11"):
                        st.caption("Historical trajectory estimate: JPL cautions against high-precision use. This spacecraft is silent.")
    live_cards()
    st.markdown('<div class="closing">Five spacecraft. One continuing expedition.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
