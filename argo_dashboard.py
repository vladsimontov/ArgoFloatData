#!/usr/bin/env python3
"""
argo_dashboard.py: BGC-Argo explorer
=======================================
Run:  streamlit run argo_dashboard.py -- --root ./argo_local

Flow:  serial / model  ->  matching floats  ->  float dossier
       (Float SN, WMO, all sensors, measurands, position)  ->  map + plots + download.

Data handling defaults (encode the community's QC guidance):
  * Prefer PARAMETER_ADJUSTED over raw when present & not all-NaN.
  * Apply QC filter (keep flags {1,2,5,8}) by default; both are toggenable.
  * Show the data mode (R/A/D) so you always know what you're looking at.

NOT TESTED against live data (offline build). Verify variable names if a plot is empty.
"""

import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Larger default chart fonts so axis titles, ticks, and legends stay readable.
# Individual figures inherit this base unless they set their own size.
_argo_tpl = pio.templates[pio.templates.default]
_argo_tpl.layout.font.size = 15          # base: ticks, legends, annotations
_argo_tpl.layout.title.font.size = 17
_argo_tpl.layout.xaxis.title.font.size = 16
_argo_tpl.layout.yaxis.title.font.size = 16

try:
    import gsw  # TEOS-10 seawater toolbox (derived physics); optional
except Exception:
    gsw = None

# ---------------- args / config ----------------
def get_root():
    # support: streamlit run argo_dashboard.py -- --root PATH
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ARGO_ROOT", "./argo_local"))
    return os.path.abspath(ap.parse_args(argv).root)

ROOT = get_root()
GOOD_QC = {1, 2, 5, 8}
# A float counts as active if it reported a profile within this many days. Argo
# floats surface about every 10 days, so this is several missed cycles. At 60 days
# it marks ~4,300 of the ~20,300 indexed floats active, in line with the ~4,000
# floats the Argo Program reports as the live array.
ACTIVE_DAYS = 60

# Preset spots for the location search: (lat, lon, radius_km), grouped by ocean.
# Chosen from the index rather than by hand: every one of these currently holds at
# least 8 active floats, so a new user exploring never lands on empty water. Some
# obvious-sounding regions are deliberately absent (the Red and Barents Seas have
# no active floats at all; the Black, Weddell and Ross Seas only a handful).
REGIONS = {
    "Mediterranean (western)": (39.0, 5.0, 500),
    "N Atlantic: Labrador Sea": (57.0, -52.0, 500),
    "N Atlantic: Irminger Sea": (60.0, -38.0, 500),
    "N Atlantic: Norwegian Sea": (67.0, 3.0, 500),
    "N Atlantic: Gulf Stream": (38.0, -70.0, 500),
    "N Atlantic: Sargasso Sea": (32.0, -64.0, 600),
    "N Atlantic: Azores": (36.5, -25.0, 500),
    "S Atlantic: Brazil-Malvinas": (-40.0, -50.0, 600),
    "S Atlantic: Benguela": (-30.0, 5.0, 600),
    "Pacific: subpolar N": (50.0, -145.0, 700),
    "Pacific: Gulf of Alaska": (55.0, -145.0, 600),
    "Pacific: California Current": (35.0, -125.0, 600),
    "Pacific: Station ALOHA": (22.75, -158.0, 600),
    "Pacific: equatorial": (0.0, -140.0, 800),
    "Pacific: Kuroshio": (35.0, 145.0, 600),
    "Pacific: Tasman Sea": (-40.0, 160.0, 600),
    "Indian: Arabian Sea": (15.0, 65.0, 600),
    "Indian: Bay of Bengal": (15.0, 88.0, 500),
    "Indian: Kerguelen": (-48.0, 70.0, 700),
    "Indian: Agulhas": (-35.0, 25.0, 600),
    "Southern Ocean: Drake Passage": (-58.0, -63.0, 700),
}
REGION_DEFAULT = "Mediterranean (western)"

# Map dot color per float type: blue = Core, green = BGC, and the darker shade of
# each is its Deep variant. Deep is orthogonal to Core/BGC, so hue carries the
# sensor payload and lightness carries the depth class, which keeps all four
# readable at once. ColorBrewer Blues/Greens, RGBA.
# One color per float type, everywhere it appears (map dots, legends, charts).
# Both palettes were validated (lightness band, chroma floor, CVD and normal-vision
# separation) against light and dark chart surfaces; the deck.gl map keeps the
# light-mode values since its Carto basemap is always light.
TYPE_HEX = {           # light surfaces
    "Core": "#4FA3DC", "Core · Deep": "#1D5FBF",
    "BGC": "#4DAE54", "BGC · Deep": "#1E7E46"}
TYPE_HEX_DARK = {      # the same hues, re-stepped for dark surfaces
    "Core": "#4399D6", "Core · Deep": "#1D5FBF",
    "BGC": "#4CAC53", "BGC · Deep": "#167339"}
# categorical set for platform-model series (fixed order, never cycled)
MODEL_HEX = ["#4269D0", "#D29A0B", "#239E86", "#E05B44"]
MODEL_HEX_DARK = ["#4269D0", "#C08A0C", "#239E86", "#E05B44"]
TYPE_COLORS = {
    "Core": [79, 163, 220, 190],
    "Core · Deep": [29, 95, 191, 220],
    "BGC": [77, 174, 84, 190],
    "BGC · Deep": [30, 126, 70, 220],
}
TYPE_COLOR_OTHER = [150, 150, 150, 180]


def _chart_colors():
    """Theme-matched chart palettes (st.context.theme knows the client theme)."""
    thm = getattr(getattr(st, "context", None), "theme", None)
    dark = bool(thm is not None and getattr(thm, "type", "") == "dark")
    return ((TYPE_HEX_DARK, MODEL_HEX_DARK) if dark else (TYPE_HEX, MODEL_HEX))


# ---- contact / feedback ------------------------------------------------------
# Set ARGO_ISSUES_URL at deploy time (or edit the default) once the repo exists.
# e.g. export ARGO_ISSUES_URL="https://github.com/<owner>/<repo>/issues"
ISSUES_URL = os.environ.get("ARGO_ISSUES_URL",
                            "https://github.com/vladsimontov/ArgoFloatData/issues")
NEW_ISSUE_URL = ISSUES_URL.rstrip("/") + "/new"

# GDAC endpoint for on-demand data fetch (override with ARGO_GDAC if you mirror it)
GDAC = os.environ.get("ARGO_GDAC", "https://data-argo.ifremer.fr").rstrip("/")
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Argo Float Data Explorer: BGC, Core & Deep Argo profiles",
    page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "assets", "logo.svg")
if os.path.exists(_logo_path):
    st.logo(_logo_path, size="large")

# small style polish (Streamlit constrains most styling)
st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  /* brand accent on the active tab underline; per-scheme so auto dark mode,
     which a config.toml [theme] would pin, keeps working */
  @media (prefers-color-scheme: light) {
    .stTabs [data-baseweb="tab-highlight"] {background-color: #1D5FBF;}
  }
  @media (prefers-color-scheme: dark) {
    .stTabs [data-baseweb="tab-highlight"] {background-color: #4FA3DC;}
  }
  [data-testid="stMetricValue"] {font-size: 1.1rem;}
  .stDataFrame {font-size: 0.9rem;}
  /* larger, bolder tab labels so the active view stays easy to track */
  .stTabs [data-baseweb="tab-list"] {gap: 1.5rem;}
  .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {font-size: 1.15rem;}
  .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] p {
      font-size: 1.15rem; font-weight: 600;}
  /* a bit larger, more readable body text, control labels, and captions */
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li {font-size: 1.06rem;}
  [data-testid="stWidgetLabel"] p {font-size: 1.04rem;}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
      font-size: 1.0rem;}
</style>
""", unsafe_allow_html=True)

# Small UX fixes injected into the parent page:
#  1) disable browser auto-correct/-capitalize/-complete/spellcheck on text inputs
#     so serial numbers (e.g. P41308-22EU002) and WMOs aren't mangled.
#  2) let the browser handle Ctrl/Cmd shortcuts (copy/paste/cut/select-all) instead
#     of Streamlit's single-key hotkeys (c = clear cache, r = rerun), which otherwise
#     hijack Ctrl+C. Capture-phase on window fires before Streamlit's handler.
#  3) poke component iframes on tab switch so the deck.gl trajectory map, which can
#     mount at zero size inside an inactive (display:none) tab and stay blank until
#     a full refresh, re-measures and paints when its tab becomes visible.
st.iframe("""
<script>
const win = window.parent, doc = win.document;
function noAutocorrect(){
  doc.querySelectorAll('input[type="text"], input:not([type]), textarea').forEach(el=>{
    el.setAttribute('autocorrect','off');
    el.setAttribute('autocapitalize','off');
    el.setAttribute('autocomplete','off');
    el.setAttribute('spellcheck','false');
  });
}
noAutocorrect();
new MutationObserver(noAutocorrect).observe(doc.body, {childList:true, subtree:true});

win.addEventListener('keydown', function(e){
  // a modifier is held -> the user wants a BROWSER shortcut (copy/paste/etc),
  // not Streamlit's bare-key hotkey; stop it reaching Streamlit's handler.
  if (e.ctrlKey || e.metaKey) e.stopImmediatePropagation();
}, true);

win.addEventListener('click', function(e){
  // On a tab switch, the just-shown panel's map went from 0x0 (hidden) to real
  // size; deck.gl doesn't always catch that. Fire a resize inside every
  // same-origin component iframe (and the window) so it re-measures and paints.
  if (!e.target.closest || !e.target.closest('button[role="tab"]')) return;
  [120, 400].forEach(function(t){
    setTimeout(function(){
      doc.querySelectorAll('iframe').forEach(function(f){
        try { f.contentWindow.dispatchEvent(new Event('resize')); } catch(_){}
      });
      win.dispatchEvent(new Event('resize'));
    }, t);
  });
}, true);
</script>
""", height=1)


# ---------------- data loading (cached) ----------------
@st.cache_data(show_spinner=False)
def load_tables(root):
    sp = os.path.join(root, "sensors.parquet")
    fp = os.path.join(root, "floats.parquet")
    if not (os.path.exists(sp) and os.path.exists(fp)):
        return None, None
    return pd.read_parquet(sp), pd.read_parquet(fp)


# Derived (TEOS-10) variables added onto every loaded float, when gsw is present.
DERIVED_2D = ["SIGMA0", "PT", "CT", "SA", "AOU"]


def _best(ds, base):
    """Adjusted field if present & finite, else raw; None if absent."""
    adj = f"{base}_ADJUSTED"
    if adj in ds and np.isfinite(ds[adj].values).any():
        return ds[adj].values
    return ds[base].values if base in ds else None


def add_derived(ds):
    """Attach TEOS-10 derived fields (SA, CT, PT, SIGMA0, AOU) + MLD to ds."""
    if gsw is None:
        return ds
    if not all(v in ds for v in ("PRES", "TEMP", "PSAL", "LATITUDE", "LONGITUDE")):
        return ds
    P, T, S = _best(ds, "PRES"), _best(ds, "TEMP"), _best(ds, "PSAL")
    if P is None or T is None or S is None:
        return ds
    nprof, nlev = P.shape
    lat = np.repeat(np.asarray(ds["LATITUDE"].values).reshape(nprof, 1), nlev, 1)
    lon = np.repeat(np.asarray(ds["LONGITUDE"].values).reshape(nprof, 1), nlev, 1)
    with np.errstate(invalid="ignore"):
        SA = gsw.SA_from_SP(S, P, lon, lat)
        CT = gsw.CT_from_t(SA, T, P)
        PT = gsw.pt0_from_t(SA, T, P)
        SIG0 = gsw.sigma0(SA, CT)
    dims = ("N_PROF", "N_LEVELS")

    def put(name, arr, units, long):
        ds[name] = (dims, np.asarray(arr))
        ds[name].attrs.update(units=units, long_name=long)

    put("SA", SA, "g/kg", "Absolute Salinity (TEOS-10)")
    put("CT", CT, "degree_Celsius", "Conservative Temperature (TEOS-10)")
    put("PT", PT, "degree_Celsius", "Potential temperature (ref 0 dbar)")
    put("SIGMA0", SIG0, "kg/m3", "Potential density anomaly sigma-0")

    O2 = _best(ds, "DOXY")
    if O2 is not None:
        with np.errstate(invalid="ignore"):
            put("AOU", gsw.O2sol(SA, CT, P, lon, lat) - O2,
                "micromole/kg", "Apparent Oxygen Utilization")

    # Mixed-layer depth: de Boyer Montegut sigma0 +0.03 kg/m3 from a 10 dbar ref
    mld = np.full(nprof, np.nan)
    for i in range(nprof):
        p, d = P[i], SIG0[i]
        m = np.isfinite(p) & np.isfinite(d)
        if m.sum() < 3:
            continue
        pp, dd = p[m], d[m]
        o = np.argsort(pp)
        pp, dd = pp[o], dd[o]
        if pp[0] > 20:                      # no near-surface sample -> skip
            continue
        thr = np.interp(10.0, pp, dd) + 0.03
        below = np.where(dd >= thr)[0]
        if len(below):
            mld[i] = pp[below[0]]
    ds["MLD"] = (("N_PROF",), mld)
    ds["MLD"].attrs.update(units="dbar",
                           long_name="Mixed layer depth (sigma0 +0.03 from 10 dbar)")
    return ds


def _gdac_get(rel, retries=3, progress=None):
    """Download dac/<...>.nc from the GDAC into ROOT (cached on disk). Returns one of:
    'ok'          - present locally or downloaded now,
    'missing'     - the server answered 4xx (e.g. 404): the file does not exist,
    'unreachable' - network error / timeout / 5xx after retries: could not contact it.
    The missing/unreachable split lets callers tell 'no such file' from 'no connection'.
    `progress`, if given, is called as progress(bytes_done, total_bytes) while
    downloading; total_bytes is 0 when the server sends no Content-Length."""
    dest = os.path.join(ROOT, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "ok"
    import requests
    import time as _time
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{GDAC}/{rel}"
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=180,
                              headers={"User-Agent": "argo-dashboard/0.2"}) as r:
                if 400 <= r.status_code < 500:
                    return "missing"   # server says the file isn't there; don't retry
                r.raise_for_status()   # 5xx -> raise, retry below
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                if progress:
                    progress(0, total)
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
                os.replace(tmp, dest)
            return "ok"
        except Exception:
            if i == retries - 1:
                return "unreachable"
            _time.sleep(1.5 * (i + 1))
    return "unreachable"


def fetch_from_gdac(rel, retries=3):
    """Download rel from the GDAC into ROOT (cached on disk). True on success."""
    return _gdac_get(rel, retries) == "ok"


@st.cache_data(show_spinner=True)
def load_sprof(path):
    return add_derived(xr.open_dataset(path, decode_times=True))


@st.cache_data(show_spinner=False)
def load_raw(path):
    # a single-cycle file, opened as reported (no derived fields, no QC filtering)
    return xr.open_dataset(path, decode_times=True)


@st.cache_data(show_spinner=False)
def load_tech(path):
    """Tidy (cycle, name, value, num) frame from an Argo <wmo>_tech.nc file.
    Values arrive as strings; `num` is the numeric coercion (NaN for text ones).
    Parameter names vary by float model and DAC, so nothing here assumes any."""
    try:
        tds = xr.open_dataset(path, decode_times=False)
    except Exception:
        return None

    def _txt(var):
        if var not in tds:
            return None
        return [x.decode("ascii", "ignore").strip() if isinstance(x, bytes)
                else str(x).strip() for x in np.asarray(tds[var].values).ravel()]

    names = _txt("TECHNICAL_PARAMETER_NAME")
    vals = _txt("TECHNICAL_PARAMETER_VALUE")
    if not names or not vals:
        return pd.DataFrame(columns=["cycle", "name", "value", "num"])
    cyc = (pd.to_numeric(pd.Series(np.asarray(tds["CYCLE_NUMBER"].values).ravel()),
                         errors="coerce")
           if "CYCLE_NUMBER" in tds else pd.Series([np.nan] * len(names)))
    df = pd.DataFrame({"cycle": cyc.values[:len(names)], "name": names,
                       "value": vals})
    df = df[df["name"] != ""]
    df["num"] = pd.to_numeric(df["value"], errors="coerce")
    return df.reset_index(drop=True)


def fetch_raw_profile(dac, wmo, cycle, is_bgc):
    """Fetch one cycle's raw NetCDF: the B-file for BGC floats, else the core profile
    file. Tries delayed-mode then real-time. Returns (rel, status): status is 'ok'
    (rel is the path), 'missing' (the GDAC has no such file for this cycle), or
    'unreachable' (could not contact the GDAC). Not cached, so a transient network
    failure can recover on the next try; the actual download is cached on disk."""
    prefixes = ["BD", "BR"] if is_bgc else ["D", "R"]
    status = "missing"          # all candidates answered 404 -> genuinely not there
    for p in prefixes:
        rel = f"dac/{dac}/{wmo}/profiles/{p}{wmo}_{int(cycle):03d}.nc"
        st_ = _gdac_get(rel)
        if st_ == "ok":
            return rel, "ok"
        if st_ == "unreachable":
            status = "unreachable"   # couldn't verify -> surface as a connection issue
    return None, status


def read_manifest(root):
    p = os.path.join(root, "manifest.json")
    if os.path.exists(p):
        import json
        with open(p) as f:
            return json.load(f)
    return {}


# ---------------- plotting helpers ----------------
def _flag_str(x):
    """Normalize a NetCDF char/byte cell (b'8', numpy bytes, str) to a clean str."""
    if isinstance(x, bytes):
        x = x.decode("ascii", "ignore")
    return str(x).strip()


def qc_summary(ds):
    """Per-parameter QC-flag breakdown from the profile file's <PARAM>_QC arrays.
    Returns (per-parameter DataFrame, dict of per-cycle location/time QC)."""
    import collections
    valid = set("0123458")            # real measurements; excludes 9 (missing/padding)
    good = set("1258")                # flags counted as usable
    grades = set("ABCDEF")            # per-profile grade scale (Table 2a)

    def counts(arr, keep):
        a = np.asarray(arr).ravel()
        if a.dtype.kind == "S":
            a = np.char.strip(np.char.decode(a, "ascii", "ignore"))
        elif a.dtype.kind in ("U", "O"):
            a = np.array([_flag_str(x) for x in a])
        else:
            a = a.astype(str)
        return collections.Counter(x for x in a.tolist() if x in keep)

    rows = []
    for v in ds.data_vars:
        if not v.endswith("_QC") or v.endswith("_ADJUSTED_QC"):
            continue
        base = v[:-3]
        if base.startswith("PROFILE_") or v in ("JULD_QC", "POSITION_QC"):
            continue
        c = counts(ds[v].values, valid)
        n = sum(c.values())
        if not n:
            continue
        n3, n4 = c.get("3", 0), c.get("4", 0)
        gr = ds.get(f"PROFILE_{base}_QC")
        g = counts(gr.values, grades) if gr is not None else {}
        rows.append({
            "parameter": base,
            "measurements": n,
            "good %": round(100 * sum(c.get(k, 0) for k in good) / n, 1),
            "questionable %": round(100 * n3 / n, 1),
            "bad %": round(100 * n4 / n, 1),
            "flagged (3+4)": n3 + n4,
            "cycle grades": " ".join(f"{k}:{g[k]}" for k in "ABCDEF" if g.get(k)),
        })
    df = (pd.DataFrame(rows).sort_values("flagged (3+4)", ascending=False)
          .reset_index(drop=True)) if rows else pd.DataFrame()
    pos = {}
    for k, lbl in (("POSITION_QC", "Position"), ("JULD_QC", "Time")):
        if k in ds:
            cc = counts(ds[k].values, valid)
            t = sum(cc.values())
            if t:
                pos[lbl] = (sum(cc.get(x, 0) for x in good), t)
    return df, pos


def _pick(ds, base, adjusted):
    """Choose <base>_ADJUSTED if requested & usable, else <base>."""
    adj = f"{base}_ADJUSTED"
    if adjusted and adj in ds and np.isfinite(ds[adj].values).any():
        return adj
    return base if base in ds else None


def param_long_frame(ds, param, adjusted=True, apply_qc=True):
    """Flatten (N_PROF, N_LEVELS) into a tidy frame for one measurand vs PRES."""
    pcol = _pick(ds, "PRES", adjusted)
    vcol = _pick(ds, param, adjusted)
    if pcol is None or vcol is None:
        return pd.DataFrame(), None, None

    pres = ds[pcol].values
    val = ds[vcol].values
    nprof, nlev = pres.shape

    # per-profile coords broadcast across levels
    def bcast(name):
        if name in ds:
            v = np.asarray(ds[name].values).reshape(nprof, 1)
            return np.repeat(v, nlev, axis=1).ravel()
        return np.full(nprof * nlev, np.nan)

    df = pd.DataFrame({
        "pres": pres.ravel(),
        "value": val.ravel(),
        "cycle": bcast("CYCLE_NUMBER"),
        "lat": bcast("LATITUDE"),
        "lon": bcast("LONGITUDE"),
    })
    if "JULD" in ds:
        juld = np.repeat(ds["JULD"].values.reshape(nprof, 1), nlev, axis=1).ravel()
        df["time"] = juld

    qc_name = f"{vcol}_QC"
    if apply_qc and qc_name in ds:
        qc = ds[qc_name].values.ravel()
        # QC flags come as bytes (b'8') or str; decode before numeric coercion,
        # otherwise str(b'8') == "b'8'" fails to parse and drops every point.
        qc = pd.to_numeric(pd.Series([_flag_str(x) for x in qc]),
                           errors="coerce")
        df["qc"] = qc.values
        df = df[df["qc"].isin(GOOD_QC)]

    df = df.dropna(subset=["pres", "value"])
    return df, pcol, vcol


def _decode_cells(a):
    """Vectorized bytes/char -> str, handling both numpy S-dtype and object arrays."""
    a = np.asarray(a)
    if a.dtype.kind == "S":
        a = np.char.decode(a, "ascii", "ignore")
    flat = np.array([_flag_str(x) for x in a.ravel()])
    return flat.reshape(a.shape)


def data_mode_for(ds, param):
    """Best-effort per-parameter data mode string across profiles."""
    # PARAMETER_DATA_MODE is (N_PROF, N_PARAM); STATION_PARAMETERS names columns.
    try:
        if "PARAMETER_DATA_MODE" in ds and "STATION_PARAMETERS" in ds:
            names = _decode_cells(ds["STATION_PARAMETERS"].values)
            modes = _decode_cells(ds["PARAMETER_DATA_MODE"].values)
            hit = np.where(names[0] == param)[0] if names.ndim == 2 else []
            if len(hit):
                col = modes[:, hit[0]]
                vals, cnts = np.unique(col, return_counts=True)
                return ", ".join(f"{v}:{c}" for v, c in zip(vals, cnts) if v)
    except Exception:
        pass
    if "DATA_MODE" in ds:
        vals, cnts = np.unique(_decode_cells(ds["DATA_MODE"].values),
                               return_counts=True)
        return "(float) " + ", ".join(f"{v}:{c}" for v, c in zip(vals, cnts))
    return "unknown"


def scientific_calib_rows(ds):
    """Delayed-mode SCIENTIFIC_CALIB per parameter (from prof/Sprof)."""
    rows = []
    if not {"SCIENTIFIC_CALIB_COEFFICIENT", "STATION_PARAMETERS"}.issubset(ds):
        return rows
    try:
        names = _decode_cells(ds["STATION_PARAMETERS"].values)          # (NPROF,NPARAM)
        co = _decode_cells(ds["SCIENTIFIC_CALIB_COEFFICIENT"].values)   # (NPROF,NCAL,NPARAM)
        eq = _decode_cells(ds["SCIENTIFIC_CALIB_EQUATION"].values)
        cm = _decode_cells(ds["SCIENTIFIC_CALIB_COMMENT"].values)
        dt = (_decode_cells(ds["SCIENTIFIC_CALIB_DATE"].values)
              if "SCIENTIFIC_CALIB_DATE" in ds else None)
        if names.ndim != 2 or co.ndim != 3:
            return rows
        nprof, ncal, npar = co.shape
        p = nprof - 1                                   # most-recent profile
        seen = set()
        for j in range(npar):
            pn = names[min(p, names.shape[0] - 1), j].strip() if names.shape[1] > j else ""
            if not pn:
                continue
            for k in range(ncal):
                c, e, m = co[p, k, j].strip(), eq[p, k, j].strip(), cm[p, k, j].strip()
                if not (c or e or m):
                    continue
                key = (pn, c, e)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"measurand": pn, "source": "delayed-mode (DMQC)",
                             "coefficient": c, "equation": e, "comment": m,
                             "date": (dt[p, k, j].strip() if dt is not None else "")})
    except Exception:
        pass
    return rows


def predeployment_calib_rows(meta_ds):
    """Factory/sensor PREDEPLOYMENT_CALIB per parameter (from meta.nc)."""
    rows = []
    try:
        def dec(n):
            return (np.atleast_1d(_decode_cells(meta_ds[n].values)).ravel()
                    if n in meta_ds else None)
        co = dec("PREDEPLOYMENT_CALIB_COEFFICIENT")
        if co is None:
            return rows
        eq = dec("PREDEPLOYMENT_CALIB_EQUATION")
        cm = dec("PREDEPLOYMENT_CALIB_COMMENT")
        pn = dec("PARAMETER")
        n = len(co)
        for i in range(n):
            c = co[i].strip()
            e = eq[i].strip() if eq is not None and i < len(eq) else ""
            m = cm[i].strip() if cm is not None and i < len(cm) else ""
            name = pn[i].strip() if pn is not None and i < len(pn) else ""
            if not (c or e or m):
                continue
            rows.append({"measurand": name, "source": "factory (predeployment)",
                         "coefficient": c, "equation": e, "comment": m, "date": ""})
    except Exception:
        pass
    return rows


def section_grid(ds, param, adjusted, apply_qc, n_p=120):
    """Interpolate each profile onto a common pressure grid -> (pgrid, times, Z)."""
    df, pcol, vcol = param_long_frame(ds, param, adjusted, apply_qc)
    if df.empty or "time" not in df.columns:
        return None
    pgrid = np.linspace(df["pres"].min(), df["pres"].max(), n_p)
    times, cols = [], []
    for _, g in df.groupby("cycle"):
        g = g.sort_values("pres")
        if len(g) < 3:
            continue
        z = np.interp(pgrid, g["pres"].to_numpy(), g["value"].to_numpy(),
                      left=np.nan, right=np.nan)
        cols.append(z)
        times.append(g["time"].iloc[0])
    if not cols:
        return None
    order = np.argsort(np.array(times))
    Z = np.array(cols)[order].T                 # (pressure, time)
    return pgrid, np.array(times)[order], Z, vcol


def ts_diagram_frame(ds):
    """Flatten (salinity, temperature, pressure, time) for a T-S diagram."""
    xn, yn = ("SA", "CT") if ("SA" in ds and "CT" in ds) else ("PSAL", "TEMP")
    P = _best(ds, "PRES")
    if P is None or xn not in ds or yn not in ds:
        return None
    x, y = ds[xn].values, ds[yn].values
    nprof, nlev = P.shape
    juld = (np.repeat(ds["JULD"].values.reshape(nprof, 1), nlev, 1).ravel()
            if "JULD" in ds else np.full(nprof * nlev, np.datetime64("NaT")))
    df = pd.DataFrame({"sal": x.ravel(), "temp": y.ravel(),
                       "pres": P.ravel(), "time": juld})
    df = df.dropna(subset=["sal", "temp", "pres"])
    return df, xn, yn


def parse_argo_date(v):
    """Argo date value ('YYYYMMDDHHMMSS' string or numeric) -> Timestamp or NaT."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return pd.NaT
    s = "".join(ch for ch in str(v) if ch.isdigit())
    if len(s) >= 14:
        return pd.to_datetime(s[:14], format="%Y%m%d%H%M%S", errors="coerce")
    if len(s) >= 8:
        return pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")
    return pd.NaT


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; lat2/lon2 may be arrays (vectorized)."""
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@st.cache_data(show_spinner=False)
def weekly_churn(root, days, today):
    """Floats launched within the last 7 days, and floats whose last report crossed
    the `days` inactivity line within the last 7 days, each as a (wmo, type) frame
    using the same type labels as the matches table (BGC / Core, with ' · Deep'
    appended for deep floats). Both come straight from the current index
    (launch_date / last_date), so no history is needed. Keyed on the calendar day
    so it follows the daily index refresh without a restart."""
    _s, fl = load_tables(root)
    empty = pd.DataFrame(columns=["wmo", "type"])
    if fl is None or "last_date" not in fl.columns:
        return empty, empty
    now = pd.Timestamp(today)

    def _typed(mask):
        sub = fl.loc[mask]
        base = (sub["data_kind"].astype(str).map({"bgc": "BGC", "core": "Core"})
                if "data_kind" in sub.columns
                else pd.Series(index=sub.index, dtype=object)).fillna("-")
        deep = (sub["is_deep"].fillna(False).astype(bool)
                if "is_deep" in sub.columns
                else pd.Series(False, index=sub.index))
        return pd.DataFrame({"wmo": sub["wmo"].astype(str),
                             "type": base.where(~deep, base + " · Deep")}
                            ).sort_values("wmo").reset_index(drop=True)

    new = empty
    if "launch_date" in fl.columns:
        lage = (now - pd.to_datetime(fl["launch_date"].map(parse_argo_date),
                                     errors="coerce")).dt.days
        new = _typed(lage.between(0, 7).fillna(False))
    page = (now - pd.to_datetime(fl["last_date"].map(parse_argo_date),
                                 errors="coerce")).dt.days
    silent = _typed(((page > days) & (page <= days + 7)).fillna(False))
    return new, silent


@st.cache_data(show_spinner=False)
def showcase_picks(root, days, today):
    """A few crowd-pleasers chosen from the live index, so they never go stale:
    the longest-record active BGC float carrying oxygen and chlorophyll sensors,
    the longest-record active deep float, the longest record overall, plus the
    active-float pool for a random pick."""
    _s, fl = load_tables(root)
    if fl is None or not len(fl) or "last_date" not in fl.columns:
        return {}, []
    age = (pd.Timestamp(today)
           - pd.to_datetime(fl["last_date"].map(parse_argo_date),
                            errors="coerce")).dt.days
    active = age.le(days).fillna(False)
    prof = pd.to_numeric(fl.get("n_profiles"), errors="coerce").fillna(0)
    params = (fl["parameters"].fillna("").astype(str) if "parameters" in fl.columns
              else pd.Series("", index=fl.index))
    picks = {}
    if "data_kind" in fl.columns:
        _bgc = fl.index[(fl["data_kind"] == "bgc") & active
                        & params.str.contains("DOXY") & params.str.contains("CHLA")]
        if len(_bgc):
            picks["bgc"] = str(fl.loc[prof.loc[_bgc].idxmax(), "wmo"])
    if "is_deep" in fl.columns:
        _deep = fl.index[fl["is_deep"].fillna(False).astype(bool) & active]
        if len(_deep):
            picks["deep"] = str(fl.loc[prof.loc[_deep].idxmax(), "wmo"])
    if float(prof.max()) > 0:
        picks["long"] = str(fl.loc[prof.idxmax(), "wmo"])
    return picks, sorted(fl.loc[active, "wmo"].astype(str))


def _split_pi_field(raw):
    """One pi_name field -> [(person, maybe_truncated)]. Fields hold lists
    ('A, B', 'A/B', surname-only rosters) and the odd 'n/a'; junk drops out.
    Argo meta strings are fixed width, so the LAST person of a long roster can
    be cut mid-name ('..., KEN JOHNSO'); the flag marks that suspicion."""
    s = str(raw).strip()
    if not s or s.lower() in ("n/a", "na", "unknown", "-", "none"):
        return []
    truncated = len(s) >= 60
    for sep in ("/", ";", " and ", " AND ", " And "):
        s = s.replace(sep, ",")
    out = []
    for part in s.split(","):
        part = " ".join(part.replace(".", " ").split())
        if sum(c.isalpha() for c in part) >= 2:
            out.append(part)
    return [(p, truncated and i == len(out) - 1) for i, p in enumerate(out)]


def _pi_key(person):
    """(SURNAME|first initial) key that folds case, initials and nicknames into
    one person. In a mixed-case name the all-caps token is the surname (Coriolis
    style, 'Xavier ANDRE'); otherwise the last token is. A single-token name
    keys as the bare surname and may fold into a full name later."""
    toks = [t for t in person.replace("-", " ").split()
            if any(c.isalpha() for c in t)]
    if not toks:
        return None
    if len(toks) == 1:
        return toks[0].upper()
    ups = [t for t in toks if t.isupper() and len(t) > 1]
    sur = ups[-1] if (ups and len(ups) < len(toks)) else toks[-1]
    first = next((t for t in toks if t is not sur), toks[0])
    return f"{sur.upper()}|{first[0].upper()}"


@st.cache_data(show_spinner=False)
def pi_directory(root):
    """The PI dropdown's people. Argo pi_name fields mix case, initials, name
    order and multi-PI rosters, so one person appears under many spellings
    ('STEPHEN RISER', 'Steve Riser', 'STEPHEN RISER/KEN JOHNSON'). Split the
    rosters, key each person by (surname, first initial), label the group with
    its most common full spelling, and remember every raw field value a person
    appears in, so filtering on them matches solo and team floats alike.
    Bare-surname mentions fold into the full name when unambiguous.
    Returns (labels sorted by surname, {label: set of raw pi_name values})."""
    s, _fl = load_tables(root)
    if s is None or "pi_name" not in s.columns:
        return [], {}
    raw = s["pi_name"].dropna().astype(str).str.strip()
    counts = raw[raw != ""].value_counts()
    groups, suspects = {}, {}
    for rv, n in counts.items():
        for person, trunc in _split_pi_field(rv):
            k = _pi_key(person)
            if k is None:
                continue
            tgt = suspects if trunc else groups
            g = tgt.setdefault(k, {"sp": {}, "raws": set()})
            g["sp"][person] = g["sp"].get(person, 0) + int(n)
            g["raws"].add(rv)
    # fold truncated tail names into the unique full person they prefix; the cut
    # spelling never becomes a label, only its floats carry over
    for k, g in suspects.items():
        if k in groups:
            groups[k]["raws"] |= g["raws"]
            continue
        if "|" in k:
            _sur, _ini = k.split("|")
            cands = [fk for fk in groups if "|" in fk
                     and fk.split("|")[1] == _ini
                     and fk.split("|")[0].startswith(_sur)]
        else:
            cands = [fk for fk in groups
                     if (fk.split("|")[0] if "|" in fk else fk).startswith(k)]
        if len(cands) == 1:
            groups[cands[0]]["raws"] |= g["raws"]
        else:
            groups[k] = g    # ambiguous or unknown: keep it rather than lose floats
    # 'WIJFFELS, JAYNE, ROBBINS' style rosters: a bare surname joins the one
    # full-name group with that surname; ambiguous surnames stay their own entry
    for k in [x for x in list(groups) if "|" not in x]:
        full = [fk for fk in groups if "|" in fk and fk.split("|")[0] == k]
        if len(full) == 1:
            groups[full[0]]["raws"] |= groups[k]["raws"]
            groups.pop(k)
    label_map, by_surname = {}, {}
    for k, g in groups.items():
        best = max(g["sp"].items(), key=lambda kv: (kv[1], len(kv[0])))[0].title()
        label_map.setdefault(best, set()).update(g["raws"])
        by_surname[best] = k.split("|")[0].title()
    labels = sorted(label_map, key=lambda L: (by_surname.get(L, L), L))
    return labels, label_map


@st.cache_data(show_spinner=False)
def pi_activity(root, days, today):
    """Active floats per person and per program, split by float type: the
    'who runs the fleet' view. People come from pi_directory, so spelling
    variants collapse; a team float credits every named PI, so per-person
    counts overlap by design. Returns (by_person, by_program) frames indexed
    by name with one column per float type."""
    _s, fl = load_tables(root)
    empty = pd.DataFrame()
    if fl is None or not {"pi_name", "last_date"}.issubset(fl.columns):
        return empty, empty
    age = (pd.Timestamp(today)
           - pd.to_datetime(fl["last_date"].map(parse_argo_date),
                            errors="coerce")).dt.days
    sub = fl[age.le(days).fillna(False)].copy()
    if not len(sub):
        return empty, empty
    base = (sub["data_kind"].astype(str).map({"bgc": "BGC", "core": "Core"})
            if "data_kind" in sub.columns
            else pd.Series(index=sub.index, dtype=object)).fillna("-")
    deep = (sub["is_deep"].fillna(False).astype(bool)
            if "is_deep" in sub.columns else pd.Series(False, index=sub.index))
    sub["type"] = base.where(~deep, base + " · Deep")
    _labels, _groups = pi_directory(root)
    raw2people = {}
    for _lbl, _raws in _groups.items():
        for _rv in _raws:
            raw2people.setdefault(_rv, []).append(_lbl)
    ppl = (sub.assign(person=sub["pi_name"].fillna("").astype(str).str.strip()
                      .map(raw2people))
           .explode("person").dropna(subset=["person"]))
    by_pi = (ppl.groupby(["person", "type"])["wmo"].nunique()
             .unstack(fill_value=0))
    if "project_name" in sub.columns:
        proj = sub.assign(project=sub["project_name"].fillna("").astype(str)
                          .str.strip())
        proj = proj[proj["project"] != ""]
        by_proj = (proj.groupby(["project", "type"])["wmo"].nunique()
                   .unstack(fill_value=0))
    else:
        by_proj = empty
    return by_pi, by_proj


# WOA23 reference climatology: slim parquets extracted by build_woa.py into
# docs/api (shipped with the repo, so reading them is a local disk read)
WOA_FILES = {"TEMP": "woa23_temperature.parquet",
             "PSAL": "woa23_salinity.parquet",
             "DOXY": "woa23_oxygen.parquet",
             "AOU": "woa23_aou.parquet",
             "NITRATE": "woa23_nitrate.parquet"}


def _woa_path(param):
    f = WOA_FILES.get(param)
    if not f:
        return None
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "docs", "api", f)
    return p if os.path.exists(p) else None


def woa_cell(lat, lon):
    """Snap a position to its WOA 1-degree cell center."""
    la = float(np.clip(np.floor(lat) + 0.5, -89.5, 89.5))
    lo = ((float(lon) + 180.0) % 360.0) - 180.0
    lo = float(np.clip(np.floor(lo) + 0.5, -179.5, 179.5))
    return la, lo


@st.cache_data(show_spinner=False)
def woa_profile(param, cell_lat, cell_lon):
    """WOA23 annual-mean profile at one 1-degree cell -> (depth, pres, val)
    frame, or None. The parquet is sorted by lat/lon in modest row groups, so
    the equality filters make this a point read, not a 2-million-row load.
    Depth (m) becomes pressure (dbar) via TEOS-10 so the overlay shares the
    float's y-axis."""
    p = _woa_path(param)
    if p is None:
        return None
    try:
        df = pd.read_parquet(p, filters=[("lat", "==", cell_lat),
                                         ("lon", "==", cell_lon)])
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.sort_values("depth").reset_index(drop=True)
    try:
        import gsw as _gsw
        df["pres"] = _gsw.p_from_z(-df["depth"].to_numpy("float64"), cell_lat)
    except Exception:
        df["pres"] = df["depth"]   # dbar is within ~2% of meters anyway
    return df


@st.cache_data(show_spinner=False)
def fleet_history(root, today):
    """Per-float launch/last timestamps plus type and platform model: the raw
    material for the fleet analytics charts. Index-only (no GDAC fetches) and
    keyed on the calendar day, so it follows the daily refresh."""
    _s, fl = load_tables(root)
    cols = ["wmo", "type", "platform", "launch", "last"]
    if fl is None or not {"launch_date", "last_date"}.issubset(fl.columns):
        return pd.DataFrame(columns=cols)
    launch = pd.to_datetime(fl["launch_date"].map(parse_argo_date), errors="coerce")
    last = pd.to_datetime(fl["last_date"].map(parse_argo_date), errors="coerce")
    base = (fl["data_kind"].astype(str).map({"bgc": "BGC", "core": "Core"})
            if "data_kind" in fl.columns
            else pd.Series(index=fl.index, dtype=object)).fillna("-")
    deep = (fl["is_deep"].fillna(False).astype(bool) if "is_deep" in fl.columns
            else pd.Series(False, index=fl.index))
    out = pd.DataFrame({"wmo": fl["wmo"].astype(str),
                        "type": base.where(~deep, base + " · Deep"),
                        "platform": (fl["platform_type"].fillna("?").astype(str)
                                     .str.strip()
                                     if "platform_type" in fl.columns else "?"),
                        "launch": launch, "last": last})
    out = out.dropna(subset=["launch", "last"])
    return out[out["launch"] <= out["last"]].reset_index(drop=True)


def climate_band(lat):
    """Latitude band label (meteorological-ish zones)."""
    a = abs(lat)
    hemi = "N" if lat >= 0 else "S"
    band = ("Tropical" if a <= 23.5 else
            "Subtropical" if a <= 35 else
            "Temperate" if a <= 55 else
            "Subpolar" if a <= 66.5 else "Polar")
    return f"{band} ({hemi})"


def ocean_basin(lat, lon):
    """Coarse basin from last position. Marginal seas first; boxes are approximate."""
    L = ((lon + 180) % 360) - 180
    if 40 <= lat <= 48 and 26.5 <= L <= 42:
        return "Black Sea"
    if 30 <= lat <= 47 and -6 <= L <= 36.5:
        return "Mediterranean Sea"
    if 12 <= lat <= 30 and 32 <= L <= 44:
        return "Red Sea"
    if 53 <= lat <= 66 and 10 <= L <= 30:
        return "Baltic Sea"
    if lat <= -60:
        return "Southern Ocean"
    if lat >= 66.5:
        return "Arctic Ocean"
    # Atlantic west boundary steps along the Americas (very coarse)
    atl_w = -75 if lat < 9 else -85 if lat < 17 else -100
    if atl_w <= L <= 20:
        return "North Atlantic" if lat >= 0 else "South Atlantic"
    if 20 < L <= 146:
        return "Indian Ocean" if lat <= 25 else "North Pacific"
    return "North Pacific" if lat >= 0 else "South Pacific"


def mann_kendall_sen(t_years, y):
    """Non-parametric trend: Sen's slope (per year) + Mann-Kendall p-value."""
    t = np.asarray(t_years, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 4:
        return None
    i, j = np.triu_indices(n, 1)
    dt, dy = t[j] - t[i], y[j] - y[i]
    good = dt != 0
    sen = float(np.median(dy[good] / dt[good])) if good.any() else np.nan
    S = float(np.sum(np.sign(dy)))
    _, counts = np.unique(y, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    varS = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if varS <= 0:
        return {"sen": sen, "p": np.nan, "S": S, "n": n}
    Z = (S - 1) / np.sqrt(varS) if S > 0 else \
        (S + 1) / np.sqrt(varS) if S < 0 else 0.0
    from scipy.stats import norm
    return {"sen": sen, "p": float(2 * norm.sf(abs(Z))), "S": S, "Z": Z, "n": n}


# ---------------- UI ----------------
sensors, floats = load_tables(ROOT)
mani = read_manifest(ROOT)

if sensors is None:
    st.error(
        f"No index found in `{ROOT}`.\n\n"
        "Run the pipeline first:\n"
        "```\npython sync_bgc_subset.py --root ./argo_local --budget-gb 5\n"
        "python build_crosswalk.py --root ./argo_local\n```"
    )
    st.stop()

# ---- deep links: ?wmo= (and ?var=) in the URL restore a float on first load.
# Seeding the search box routes through the direct-lookup path, which ignores
# filters, so a linked float is never hidden by a filter the recipient never set.
if "_dl_seen" not in st.session_state:
    st.session_state["_dl_seen"] = True
    _dl_w = str(st.query_params.get("wmo") or "").strip()
    _dl_v = str(st.query_params.get("var") or "").strip().upper()
    if _dl_v:
        st.session_state["_dl_var"] = _dl_v
    if _dl_w and (floats["wmo"].astype(str) == _dl_w).any():
        st.session_state["q_box"] = _dl_w
        st.session_state["wmo_pick"] = _dl_w
    _dl_watch = str(st.query_params.get("watch") or "").strip()
    if _dl_watch:
        _ws = [w for w in _dl_watch.split(".") if w.isdigit()]
        if _ws:
            st.session_state["watchlist"] = _ws
st.session_state.setdefault("watchlist", [])
st.session_state.setdefault("cmp_list", [])


def _open_float(wmo):
    # on_click callbacks run before any widget is created on the rerun, so seeding
    # both the search box and the picker here is safe
    st.session_state["q_box"] = str(wmo)
    st.session_state["wmo_pick"] = str(wmo)


def _sync_watch_param():
    wl = st.session_state.get("watchlist", [])
    if wl:
        st.query_params["watch"] = ".".join(wl)
    else:
        st.query_params.pop("watch", None)


def _toggle_watch(wmo):
    wl = st.session_state.setdefault("watchlist", [])
    wmo = str(wmo)
    if wmo in wl:
        wl.remove(wmo)
    else:
        wl.append(wmo)
    _sync_watch_param()


def _add_to_compare(wmo):
    # a cart, not a redirect: the nav item's count ticks up and the user goes to
    # the Compare page when ready. Plain session key, NOT the page's widget key,
    # because Streamlit drops widget state while the widget's page is not rendered.
    cl = st.session_state.setdefault("cmp_list", [])
    wmo = str(wmo)
    if wmo in cl:
        st.toast(f"Float {wmo} is already on the compare list "
                 f"({len(cl)} of 3).", icon="⚖️")
    elif len(cl) >= 3:
        st.toast("The compare list is full (3 floats). Edit it on the "
                 "Compare page.", icon="⚖️")
    else:
        cl.append(wmo)
        st.toast(f"Added float {wmo} to the compare list "
                 f"({len(cl)} of 3).", icon="⚖️")


_ACK_MD = """
**Thank you to Argo, and to the people and nations who make it possible.**

Every profile in this tool exists because of the **International Argo Program** and
the ~30 nations: their governments, agencies, engineers, and scientists who fund,
build, deploy, recover, quality-control, and then *freely* share these floats with
the world. Sustained ocean observing on this scale is an act of international
generosity, and this tool is only a small window onto their work. **Thank you.** 🌊

> *"These data were collected and made freely available by the International Argo
> Program and the national programs that contribute to it
> (https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program is part of
> the Global Ocean Observing System (GOOS)."*

**Please cite the data**
Argo (2026). *Argo float data and metadata from Global Data Assembly Centre
(Argo GDAC).* SEANOE. https://doi.org/10.17882/42182

**With gratitude to the national programs & Data Assembly Centres**, including
Australia (CSIRO / BOM / IMOS), Canada (DFO / MEDS), China (SIO / CSIO), France
(Ifremer / Coriolis, CNES), Germany (BSH / GEOMAR), India (INCOIS / MoES),
Italy (OGS), Japan (JAMSTEC / JMA), South Korea (KMA / KIOST), the United Kingdom
(Met Office / BODC), the United States (NOAA / SIO), **Euro-Argo ERIC**, and every other
nation and government contributing to Argo. The full float array is a shared gift.

**License:** Argo data are freely available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This tool applies QC
filtering and computes derived quantities (TEOS-10 density, mixed-layer depth,
apparent oxygen utilization, absolute/conservative properties, trends) that are
**not official Argo products**. Data are retrieved from the Argo GDAC.

*Built with the help of [Claude](https://claude.ai) (Anthropic). An independent,
community tool, not affiliated with or endorsed by the Argo Program.*
"""


# ---- pages: Explore is the fall-through body below pg.run(); the other pages
# render and stop, so the big explore flow stays unindented at module level ----
def page_explore():
    pass          # body = everything after st.navigation(...).run()


def page_fleet():
    st.title("Fleet overview")
    st.caption("The whole array at a glance: today\'s counts, this week\'s churn, "
               "and the array\'s history, all from the daily-refreshed metadata index.")
    # freshness banner: counts across the WHOLE index (not just locally-cached data)
    kind = floats["data_kind"] if "data_kind" in floats else pd.Series(dtype=object)
    n_bgc = int((kind == "bgc").sum())
    n_core = int((kind == "core").sum())
    n_deep = int(floats["is_deep"].sum()) if "is_deep" in floats else 0
    n_sbe61 = (sensors.loc[sensors["sensor_model"].fillna("").str.contains("SBE61"),
                           "wmo"].astype(str).nunique()
               if "sensor_model" in sensors else 0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Floats indexed", f"{floats['wmo'].nunique():,}",
              help="Every float searchable in this tool.")
    c2.metric("BGC floats", f"{n_bgc:,}", help="Carry biogeochemical sensors (Sprof).")
    c3.metric("Core floats", f"{n_core:,}", help="Physical T/S floats (prof.nc).")
    c4.metric("Deep floats", f"{n_deep:,}",
              help=f"PLATFORM_FAMILY = FLOAT_DEEP · {n_sbe61:,} with SBE61 (6000 m).")
    c5.metric("Last synced (UTC)", mani.get("synced_utc", "-"),
              help="When the metadata index was last refreshed.")

    # this week's churn: who joined the array, who went quiet, and the net direction,
    # broken down by float type (BGC / Core, deep floats labeled '· Deep')
    _new_df, _silent_df = weekly_churn(
        ROOT, ACTIVE_DAYS, pd.Timestamp.now().normalize().isoformat())
    _net = len(_new_df) - len(_silent_df)


    def _by_type(df):
        """'8 Core, 3 BGC, 1 Core · Deep', largest group first."""
        return (", ".join(f"{int(n)} {t}" for t, n in df["type"].value_counts().items())
                if len(df) else "none")


    w1, w2, w3 = st.columns(3)
    w1.metric("New floats (7 days)", f"{len(_new_df):,}",
              help="Floats whose launch date (from their meta file) falls within the "
                   "last 7 days. A float joins the index once it starts reporting.")
    w1.caption(_by_type(_new_df))
    w2.metric("Fell silent (7 days)", f"{len(_silent_df):,}",
              help=f"Floats whose last report crossed the {ACTIVE_DAYS}-day inactivity "
                   "line within the last 7 days. Argo floats surface about every 10 "
                   "days, so a longer gap almost always means the float has stopped.")
    w2.caption(_by_type(_silent_df))
    w3.metric("Weekly trend",
              "growing" if _net > 0 else "shrinking" if _net < 0 else "steady",
              delta=(f"{_net:+d} floats" if _net else None),
              help="New floats minus newly-silent floats over the last 7 days: the net "
                   "change in the actively reporting array.")
    _net_by_type = (_new_df["type"].value_counts()
                    .sub(_silent_df["type"].value_counts(), fill_value=0).astype(int))
    _net_by_type = _net_by_type[_net_by_type != 0].sort_values()
    if len(_net_by_type):
        w3.caption(", ".join(f"{t} {int(v):+d}" for t, v in _net_by_type.items()))
    if len(_new_df) or len(_silent_df):
        with st.expander("This week's additions and losses (WMOs)", expanded=False):

            def _grouped(df):
                if not len(df):
                    return "none"
                lines = []
                for t, grp in df.groupby("type", sort=True):
                    ws = grp["wmo"].tolist()
                    lines.append(f"**{t}** ({len(ws)}): " + ", ".join(ws[:30])
                                 + (" …" if len(ws) > 30 else ""))
                return "  \n".join(lines)

            e1, e2 = st.columns(2)
            e1.markdown("**New**  \n" + _grouped(_new_df))
            e2.markdown("**Fell silent**  \n" + _grouped(_silent_df))

    # ---- fleet analytics: the whole array's history, reconstructed from the index ----
    with st.expander("Fleet analytics · deployments, lifespans and survival since 1999",
                     expanded=False):
        _fh = fleet_history(ROOT, pd.Timestamp.now().normalize().isoformat())
        if not len(_fh):
            st.info("The index has no launch/last dates to chart.")
        else:
            _now_ts = pd.Timestamp.now().normalize()
            _th, _mh = _chart_colors()
            _ftypes = [t for t in TYPE_COLORS if (_fh["type"] == t).any()]
            _done = _fh[(_now_ts - _fh["last"]).dt.days > ACTIVE_DAYS]
            _life_yrs = ((_done["last"] - _done["launch"]).dt.days / 365.25).clip(lower=0)
            _dep_year = _fh["launch"].dt.year
            a1, a2, a3 = st.columns(3)
            a1.metric("Reporting now", f"{(len(_fh) - len(_done)):,}",
                      help=f"Floats whose last profile is within {ACTIVE_DAYS} days.")
            a2.metric("Median completed lifespan", f"{_life_yrs.median():.1f} yr",
                      help="Across floats that have stopped reporting: launch to last "
                           "profile. Active floats are not counted, so this is not "
                           "biased by floats still going strong.")
            a3.metric("Busiest deployment year",
                      str(int(_dep_year.value_counts().idxmax())),
                      help="The calendar year with the most float launches.")

            g1, g2 = st.columns(2)
            # 1) active floats over time, one line per type
            _months = pd.date_range("1999-01-01", _now_ts, freq="MS")
            _mv = _months.values.astype("datetime64[ns]")
            _fig_a = go.Figure()
            for _t in _ftypes:
                _sub = _fh[_fh["type"] == _t]
                _ls = np.sort(_sub["launch"].values.astype("datetime64[ns]"))
                _le = np.sort(_sub["last"].values.astype("datetime64[ns]"))
                _n = (np.searchsorted(_ls, _mv, side="right")
                      - np.searchsorted(_le, _mv, side="left"))
                _fig_a.add_trace(go.Scatter(
                    x=_months, y=_n, name=_t, mode="lines",
                    line=dict(color=_th[_t], width=2),
                    hovertemplate="%{x|%Y-%m} · %{y:,} reporting<extra>" + _t + "</extra>"))
            _fig_a.update_layout(
                height=360, margin=dict(l=10, r=10, t=70, b=10),
                yaxis_title="floats", legend=dict(orientation="h", y=1.02,
                                                  yanchor="bottom"))
            g1.markdown("**Reporting floats over time**")
            g1.plotly_chart(_fig_a, width="stretch")
            # 2) deployments per year, stacked by type
            _dep = (_fh.assign(year=_dep_year).groupby(["year", "type"]).size()
                    .unstack(fill_value=0))
            _dep = _dep.loc[(_dep.index >= 1998) & (_dep.index <= _now_ts.year),
                            [t for t in _ftypes if t in _dep.columns]]
            _fig_d = go.Figure()
            for _t in _dep.columns:
                _fig_d.add_trace(go.Bar(
                    x=_dep.index, y=_dep[_t], name=_t, marker_color=_th[_t],
                    hovertemplate="%{x} · %{y:,} launched<extra>" + _t + "</extra>"))
            _fig_d.update_layout(
                barmode="stack", height=360, margin=dict(l=10, r=10, t=70, b=10),
                yaxis_title="floats launched",
                legend=dict(orientation="h", y=1.02, yanchor="bottom"))
            g2.markdown("**Deployments per year**")
            g2.plotly_chart(_fig_d, width="stretch")

            g3, g4 = st.columns(2)
            # 3) lifespan distribution (completed floats only; the title names the
            #    single series, so no legend)
            _fig_l = go.Figure(go.Histogram(
                x=_life_yrs, xbins=dict(start=0, end=12, size=0.5),
                marker_color=_th["Core"],
                hovertemplate="%{x} yr · %{y:,} floats<extra></extra>"))
            _fig_l.update_layout(
                height=330, margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="years", yaxis_title="floats")
            g3.markdown("**Completed lifespans (launch to last profile)**")
            g3.plotly_chart(_fig_l, width="stretch")
            # 4) survival by platform model: among floats launched at least t years
            #    ago, the fraction that were still reporting t years after launch
            _top_models = _fh["platform"].replace("?", np.nan).dropna() \
                                         .value_counts().head(4).index.tolist()
            _tgrid = np.arange(0, 10.01, 0.25)
            _fig_s = go.Figure()
            for _mdl, _mc in zip(_top_models, _mh):
                _sub = _fh[_fh["platform"] == _mdl]
                _lifed = (_sub["last"] - _sub["launch"]).dt.days.to_numpy("float64")
                _aged = (_now_ts - _sub["launch"]).dt.days.to_numpy("float64")
                _fr = []
                for _ty in _tgrid:
                    _td = _ty * 365.25
                    _coh = _aged >= _td
                    _fr.append(float((_lifed[_coh] >= _td).mean())
                               if _coh.sum() >= 50 else np.nan)
                _fig_s.add_trace(go.Scatter(
                    x=_tgrid, y=_fr, name=_mdl, mode="lines",
                    line=dict(color=_mc, width=2),
                    hovertemplate="%{x:.2f} yr · %{y:.0%} still reporting<extra>"
                                  + _mdl + "</extra>"))
            _fig_s.update_layout(
                height=360, margin=dict(l=10, r=10, t=70, b=10),
                xaxis_title="years since launch", yaxis_title="still reporting",
                yaxis_tickformat=".0%",
                legend=dict(orientation="h", y=1.02, yanchor="bottom"))
            g4.markdown("**Survival by platform model (top 4)**")
            g4.plotly_chart(_fig_s, width="stretch")
            st.caption(
                "All four charts come from the metadata index alone (launch date, last "
                "profile date, type, platform model), refreshed daily. A float counts "
                "as reporting between its launch and its last profile. Survival curves "
                "stop where a cohort drops below 50 floats.")
            if st.checkbox("Show the numbers as tables", value=False,
                           help="The same data as the charts, for reading exact values "
                                "or copying out."):
                t1, t2 = st.columns(2)
                t1.markdown("**Deployments per year**")
                t1.dataframe(_dep, width="stretch", height=240)
                _active_tbl = pd.DataFrame(
                    {_t: (np.searchsorted(np.sort(_fh.loc[_fh["type"] == _t, "launch"]
                                                  .values.astype("datetime64[ns]")),
                                          _mv, side="right")
                          - np.searchsorted(np.sort(_fh.loc[_fh["type"] == _t, "last"]
                                                    .values.astype("datetime64[ns]")),
                                            _mv, side="left"))
                     for _t in _ftypes}, index=_months.strftime("%Y-%m"))
                t2.markdown("**Reporting floats by month**")
                t2.dataframe(_active_tbl.iloc[::-1], width="stretch", height=240)

            st.subheader("Who runs the fleet")
            _by_pi, _by_proj = pi_activity(
                ROOT, ACTIVE_DAYS, pd.Timestamp.now().normalize().isoformat())

            def _top_bar(bt, col, title, n=10):
                if bt is None or not len(bt):
                    col.info("No attribution data in the index.")
                    return
                bt = (bt.assign(_tot=bt.sum(axis=1)).sort_values("_tot")
                      .drop(columns="_tot").tail(n))
                _figp = go.Figure()
                for _t in [c for c in TYPE_COLORS if c in bt.columns]:
                    _figp.add_trace(go.Bar(
                        y=bt.index, x=bt[_t], name=_t, orientation="h",
                        marker_color=_th[_t],
                        hovertemplate="%{y} · %{x:,} active<extra>" + _t
                                      + "</extra>"))
                _figp.update_layout(
                    barmode="stack", height=440,
                    margin=dict(l=10, r=10, t=60, b=10),
                    xaxis_title="active floats",
                    legend=dict(orientation="h", y=1.02, yanchor="bottom"))
                col.markdown(f"**{title}**")
                col.plotly_chart(_figp, width="stretch")

            p1, p2 = st.columns(2)
            _top_bar(_by_pi, p1, "Most active principal investigators")
            _top_bar(_by_proj, p2, "Biggest programs right now")
            st.caption(f"Floats reporting within {ACTIVE_DAYS} days, split by "
                       "float type. A float run by a team credits every named "
                       "PI, so PI counts can overlap; spelling variants are "
                       "grouped into people the same way as the search filter.")
    st.stop()


def page_about():
    st.title("About this tool and the data")
    st.markdown(_ACK_MD)
    st.markdown(
        f"**🐛 Found a bug, or have a request?** "
        f"[Open an issue]({NEW_ISSUE_URL}) · [browse issues]({ISSUES_URL}). "
        "All Feedback and Feature requests are welcome.")
    st.subheader("Data API")
    st.markdown(
        "Everything this tool knows is also a free **HTTP data API**: the full "
        "float and sensor index as Parquet/JSON, queryable with DuckDB or pandas, "
        "no key needed, refreshed daily. "
        "[Documentation and examples](https://vladsimontov.github.io/ArgoFloatData/api.html).")
    st.subheader("Data notes")
    st.markdown(
        "Float data is fetched on demand from the Argo GDAC when you open a float, "
        "then cached. The GDAC is mutable (delayed-mode QC rewrites history), so "
        "the index is refreshed daily; pin a DOI snapshot for publications.\n\n"
        "The climatology overlay uses the **World Ocean Atlas 2023** (annual "
        "1-degree objectively analyzed means for temperature, salinity, oxygen, "
        "AOU and nitrate), courtesy of "
        "[NOAA NCEI](https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/); "
        "the extracted reference profiles are also served as parquet files under "
        "this project's data API.")
    st.stop()


_PG_EXPLORE = st.Page(page_explore, title="Explore floats", icon="🌍",
                      default=True)
_PG_FLEET = st.Page(page_fleet, title="Fleet overview", icon="📊",
                    url_path="fleet")
def _plottable_params(ds_x):
    """Parameter names this dataset can plot: real (N_PROF, N_LEVELS) measurands
    plus whichever TEOS-10 derived fields add_derived computed."""
    base = {v for v in ds_x.data_vars
            if getattr(ds_x[v], "dims", ()) == ("N_PROF", "N_LEVELS")
            and not v.endswith(("_QC", "_ADJUSTED", "_ADJUSTED_QC",
                                "_ADJUSTED_ERROR"))}
    base |= {d for d in DERIVED_2D if d in ds_x}
    return base


def page_compare():
    st.title("Compare floats")
    st.caption("Median vertical profiles of two or three floats on a shared "
               "pressure axis: each level is that float's median across its whole "
               "record, so this is a climatological comparison, good for spotting "
               "sensor offsets and water-mass contrasts.")
    if "_cmp_dl_seen" not in st.session_state:
        st.session_state["_cmp_dl_seen"] = True
        _dlw = str(st.query_params.get("wmos") or "").strip()
        if _dlw:
            _dl_list = [w for w in _dlw.split(".") if w.isdigit()][:3]
            if _dl_list:
                st.session_state["cmp_list"] = _dl_list
        _dlv = str(st.query_params.get("var") or "").strip().upper()
        if _dlv:
            st.session_state["_cmp_dl_var"] = _dlv
    # the compare list (filled by 'Add to Compare List' buttons, deep links, or
    # typing here) drives the text box; edits to the box flow back into the list.
    # The box's own widget state is dropped whenever another page renders, so the
    # plain cmp_list key is the durable store and the box reseeds from it.
    _cl = st.session_state.get("cmp_list", [])
    if (st.session_state.get("_cmp_sync") != tuple(_cl)
            or ("cmp_page_wmos" not in st.session_state and _cl)):
        st.session_state["_cmp_sync"] = tuple(_cl)
        st.session_state["cmp_page_wmos"] = ", ".join(_cl)
    _in = st.text_input(
        "Floats to compare (two or three WMOs, comma separated)",
        key="cmp_page_wmos", placeholder="e.g. 6903132, 1902304",
        help="Each float's profile file is fetched from the GDAC on demand and "
             "cached, so the first load of a new float downloads its file "
             "(often ~10 MB).")
    _wmos_in = list(dict.fromkeys(w for w in _in.replace(",", " ").split() if w))[:3]
    st.session_state["cmp_list"] = _wmos_in
    st.session_state["_cmp_sync"] = tuple(_wmos_in)
    if len(_wmos_in) < 2:
        st.info("Name at least two floats to compare. Tip: the float view on the "
                "Explore page has an 'Add to Compare List' button that collects "
                "floats here as you browse.")
        st.stop()
    _loaded, _notes = [], []
    for _w in _wmos_in:
        _m = floats[floats["wmo"].astype(str) == _w]
        if not len(_m):
            _notes.append(f"{_w}: not in the index")
            continue
        _row = _m.iloc[0]
        _rel = _row.get("sprof_path")
        if not (isinstance(_rel, str) and _rel
                and os.path.exists(os.path.join(ROOT, _rel))):
            _rel = _row.get("expected_rel")
            if not (isinstance(_rel, str) and _rel):
                _suf = ("_Sprof.nc" if str(_row.get("data_kind")) == "bgc"
                        else "_prof.nc")
                _rel = f"dac/{_row.get('dac')}/{_w}/{_w}{_suf}"
            _dst = os.path.join(ROOT, _rel)
            if not (os.path.exists(_dst) and os.path.getsize(_dst) > 0):
                with st.spinner(f"Fetching float {_w} from the GDAC…"):
                    if _gdac_get(_rel) != "ok":
                        _notes.append(f"{_w}: could not fetch its profile file")
                        continue
        _base = {"bgc": "BGC", "core": "Core"}.get(str(_row.get("data_kind")), "-")
        _typ = f"{_base} · Deep" if bool(_row.get("is_deep")) else _base
        _loaded.append((_w, load_sprof(os.path.join(ROOT, _rel)), _typ))
    if len(_loaded) < 2:
        for _n in _notes:
            st.info(_n)
        st.stop()
    # who is being compared, straight from the index: identity, deployment and
    # record, then how far apart the floats are in space and time
    _sum_rows = []
    for _w, _d, _typ in _loaded:
        _r = floats[floats["wmo"].astype(str) == _w].iloc[0]
        _la, _lo = _r.get("last_lat"), _r.get("last_lon")
        _sum_rows.append({
            "WMO": _w, "type": _typ,
            "platform": str(_r.get("platform_type") or "-").strip() or "-",
            "project": str(_r.get("project_name") or "-").strip() or "-",
            "launched": parse_argo_date(_r.get("launch_date")),
            "last profile": parse_argo_date(_r.get("last_date")),
            "profiles": pd.to_numeric(_r.get("n_profiles"), errors="coerce"),
            "approx region": (f"{climate_band(float(_la))} "
                              f"{ocean_basin(float(_la), float(_lo))}"
                              if pd.notna(_la) and pd.notna(_lo) else "-"),
            "_lat": _la, "_lon": _lo,
        })
    st.dataframe(
        pd.DataFrame(_sum_rows).drop(columns=["_lat", "_lon"]),
        width="stretch", hide_index=True,
        column_config={
            "launched": st.column_config.DatetimeColumn(
                "launched", format="YYYY-MM-DD",
                help="Deployment date from the float's meta file."),
            "last profile": st.column_config.DatetimeColumn(
                "last profile", format="YYYY-MM-DD"),
            "profiles": st.column_config.NumberColumn("profiles", format="%d"),
        })
    _pair_bits = []
    for _i in range(len(_sum_rows)):
        for _j in range(_i + 1, len(_sum_rows)):
            _a, _b = _sum_rows[_i], _sum_rows[_j]
            _bits = []
            if (pd.notna(_a["_lat"]) and pd.notna(_a["_lon"])
                    and pd.notna(_b["_lat"]) and pd.notna(_b["_lon"])):
                _km = float(haversine_km(float(_a["_lat"]), float(_a["_lon"]),
                                         float(_b["_lat"]), float(_b["_lon"])))
                _bits.append(f"last fixes {_km:,.0f} km "
                             f"({_km * 0.621371:,.0f} mi) apart")
            if pd.notna(_a["launched"]) and pd.notna(_b["launched"]):
                _bits.append(f"launched "
                             f"{abs((_a['launched'] - _b['launched']).days):,} "
                             "days apart")
            if _bits:
                _pair_bits.append(f"**{_a['WMO']} vs {_b['WMO']}**: "
                                  + " · ".join(_bits))
    if _pair_bits:
        st.markdown("  \n".join(_pair_bits))
        st.caption("Distances compare each float's last known fix. Floats drift, "
                   "so this is where they were last heard from, not a "
                   "full-history separation.")
    _common = set.intersection(*[_plottable_params(d) for _w, d, _t in _loaded])
    _common -= {"PRES", "MTIME"}
    _opts = sorted(_common)
    if not _opts:
        st.warning("These floats share no plottable parameters.")
        for _n in _notes:
            st.info(_n)
        st.stop()
    _cc1, _cc2 = st.columns([2, 2])
    _dlv = st.session_state.pop("_cmp_dl_var", None)
    if _dlv in _opts:
        st.session_state["cmp_page_var"] = _dlv
    if st.session_state.get("cmp_page_var") not in _opts:
        st.session_state["cmp_page_var"] = next(
            (p for p in ("TEMP", "PSAL", "CT", "PT", "DOXY") if p in _opts),
            _opts[0])
    _cparam = _cc1.selectbox(
        "Parameter", _opts, key="cmp_page_var",
        help="Only parameters every compared float carries, TEOS-10 derived "
             "fields included.")
    _cview = _cc2.radio(
        "Data view", ["Real-time (R)", "QC-filtered", "Adjusted (A/D)"],
        index=2, horizontal=True, key="cmp_page_view",
        help="Same meaning as in Profile & Trend.")
    _cadj = _cview.startswith("Adjusted")
    _cqc = _cview.startswith(("QC", "Adjusted"))
    _wparam = ".".join(w for w, _d, _t in _loaded)
    if st.query_params.get("wmos") != _wparam:
        st.query_params["wmos"] = _wparam
    if st.query_params.get("var") != str(_cparam):
        st.query_params["var"] = str(_cparam)
    _fig = go.Figure()
    _mh = _chart_colors()[1]
    for _i, (_w, _d, _typ) in enumerate(_loaded):
        _df, _pc, _vc = param_long_frame(_d, _cparam, _cadj, _cqc)
        if _df is None or _df.empty:
            _notes.append(f"{_w}: no {_cparam} data under this view")
            continue
        _mp = (_df.assign(_pb=(_df["pres"] // 10) * 10 + 5)
               .groupby("_pb")["value"].median().reset_index())
        _fig.add_trace(go.Scatter(
            x=_mp["value"], y=_mp["_pb"], mode="lines", name=f"{_w} ({_typ})",
            line=dict(color=_mh[_i % len(_mh)], width=2),
            hovertemplate="%{x} at %{y} dbar<extra>" + _w + "</extra>"))
    if _fig.data:
        _fig.update_layout(height=520, margin=dict(l=10, r=10, t=70, b=10),
                           xaxis_title=_cparam, yaxis_title="pressure (dbar)",
                           yaxis_autorange="reversed",
                           legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(_fig, width="stretch")
        st.caption("Median value per 10 dbar pressure bin across each float's "
                   "full record · the page URL names this comparison, so it is "
                   "shareable. Floats from different regions or seasons will "
                   "differ for perfectly good reasons.")
    # over time: the parameter at a pressure horizon, with the floats' separation
    # on a SHARED x-axis (stacked subplots), so zooming one zooms the other
    st.markdown(f"**Over time · {_cparam} at a pressure horizon, with distance "
                "apart**")
    _tsp = st.number_input(
        "Pressure horizon (dbar)", min_value=0.0, max_value=6000.0,
        value=100.0, step=50.0, key="cmp_ts_pres",
        help="Each profile contributes its sample nearest this pressure (within "
             "25 dbar or 10 percent, whichever is larger), so the floats are "
             "compared at the same depth over time.")
    _ts_traces = []
    for _i, (_w, _d, _typ) in enumerate(_loaded):
        _df, _pc, _vc = param_long_frame(_d, _cparam, _cadj, _cqc)
        if _df is None or _df.empty or "time" not in _df.columns:
            continue
        _df = _df.dropna(subset=["time"])
        if not len(_df):
            continue
        _df = _df.assign(_dp=(_df["pres"] - _tsp).abs())
        _near = (_df.sort_values("_dp").groupby("time", as_index=False).first())
        _near = _near[_near["_dp"] <= max(25.0, 0.1 * _tsp)].sort_values("time")
        if not len(_near):
            _notes.append(f"{_w}: no samples near {_tsp:.0f} dbar")
            continue
        _ts_traces.append(go.Scatter(
            x=_near["time"], y=_near["value"], mode="lines+markers",
            name=f"{_w} ({_typ})", marker=dict(size=4),
            line=dict(color=_mh[_i % len(_mh)], width=2),
            hovertemplate="%{x|%Y-%m-%d} · %{y}<extra>" + _w + "</extra>"))

    def _positions(ds_x):
        if not all(k in ds_x for k in ("LATITUDE", "LONGITUDE", "JULD")):
            return None
        _p = pd.DataFrame({
            "time": pd.Series(np.asarray(ds_x["JULD"].values).ravel()),
            "lat": np.asarray(ds_x["LATITUDE"].values).ravel(),
            "lon": np.asarray(ds_x["LONGITUDE"].values).ravel()}).dropna()
        _p = _p.sort_values("time").reset_index(drop=True)
        return _p if len(_p) else None

    _pos = {_w: _positions(_d) for _w, _d, _t in _loaded}
    _dist_traces = []
    _dashes = ["solid", "dash", "dot"]
    _pk = 0
    for _i in range(len(_loaded)):
        for _j in range(_i + 1, len(_loaded)):
            _wa, _wb = _loaded[_i][0], _loaded[_j][0]
            _pa, _pb = _pos.get(_wa), _pos.get(_wb)
            if _pa is None or _pb is None:
                continue
            _mm = pd.merge_asof(_pa, _pb, on="time", direction="nearest",
                                tolerance=pd.Timedelta("15D"),
                                suffixes=("_a", "_b")).dropna()
            if not len(_mm):
                _notes.append(f"{_wa} vs {_wb}: their records do not overlap in "
                              "time, so there is no separation to plot")
                continue
            _km = haversine_km(_mm["lat_a"].to_numpy("float64"),
                               _mm["lon_a"].to_numpy("float64"),
                               _mm["lat_b"].to_numpy("float64"),
                               _mm["lon_b"].to_numpy("float64"))
            _dist_traces.append(go.Scatter(
                x=_mm["time"], y=_km, mode="lines", name=f"{_wa} vs {_wb}",
                line=dict(color="#808A95", width=2,
                          dash=_dashes[_pk % len(_dashes)]),
                customdata=_km * 0.621371,
                hovertemplate="%{x|%Y-%m-%d} · %{y:,.0f} km "
                              "(%{customdata:,.0f} mi) apart<extra>"
                              + f"{_wa} vs {_wb}" + "</extra>"))
            _pk += 1
    if _ts_traces and _dist_traces:
        _overlay = st.toggle(
            "Overlay distance on the measurand panel", value=False,
            key="cmp_overlay_dist",
            help="On: one panel, the measurand on the left axis and the floats' "
                 "separation on the right axis (gray dashed). Off: two stacked "
                 "panels with a shared time axis. The two scales are unrelated, "
                 "so the overlay is off by default.")
        if _overlay:
            # single panel, separation on a clearly-labeled right axis
            _figc = make_subplots(specs=[[{"secondary_y": True}]])
            for _t in _ts_traces:
                _figc.add_trace(_t, secondary_y=False)
            for _t in _dist_traces:
                _t.update(showlegend=True,
                          name=(_t.name or "separation") + " · right axis")
                _figc.add_trace(_t, secondary_y=True)
            _figc.update_yaxes(title_text=_cparam, secondary_y=False)
            _figc.update_yaxes(title_text="separation (km)", rangemode="tozero",
                               showgrid=False, secondary_y=True)
            _figc.update_xaxes(title_text="date")
            _figc.update_layout(
                height=520, margin=dict(l=10, r=10, t=50, b=70),
                legend=dict(orientation="h", y=-0.18, yanchor="top"))
            st.plotly_chart(_figc, width="stretch")
            st.caption(f"{_cparam} at {_tsp:.0f} dbar on the left axis; the gray "
                       "dashed line is the floats' separation on the right axis "
                       "(km, hover for miles). The two scales are unrelated, so "
                       "read each line against its own axis.")
        else:
            # one figure, two rows, one shared x-axis: zoom or pan either row
            # and the other follows
            _figc = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                row_heights=[0.62, 0.38],
                subplot_titles=(f"{_cparam} at {_tsp:.0f} dbar",
                                "Distance apart"))
            for _t in _ts_traces:
                _figc.add_trace(_t, row=1, col=1)
            for _t in _dist_traces:
                if len(_dist_traces) == 1:
                    _t.update(showlegend=False)   # the subplot title names it
                _figc.add_trace(_t, row=2, col=1)
            _figc.update_yaxes(title_text=_cparam, row=1, col=1)
            _figc.update_yaxes(title_text="separation (km)", rangemode="tozero",
                               row=2, col=1)
            _figc.update_xaxes(title_text="date", row=2, col=1)
            _figc.update_layout(
                height=680, margin=dict(l=10, r=10, t=50, b=70),
                legend=dict(orientation="h", y=-0.12, yanchor="top"))
            st.plotly_chart(_figc, width="stretch")
            st.caption("Shared time axis: zooming either panel zooms both. Top: "
                       "one point per profile, the sample nearest the chosen "
                       "pressure (gaps mean the float skipped that depth or "
                       "failed the data view). Bottom: great-circle distance "
                       "between surfacing positions, each profile paired with "
                       "the other float's nearest-in-time profile within 15 "
                       "days; hover for miles.")
    elif _ts_traces:
        _fig_ts = go.Figure(data=_ts_traces)
        _fig_ts.update_layout(height=380, margin=dict(l=10, r=10, t=70, b=10),
                              xaxis_title="date", yaxis_title=_cparam,
                              legend=dict(orientation="h", y=1.02,
                                          yanchor="bottom"))
        st.plotly_chart(_fig_ts, width="stretch")
        st.caption("One point per profile: the sample nearest the chosen "
                   "pressure. Gaps mean the float skipped that depth or its "
                   "values failed the selected data view.")
    elif _dist_traces:
        _fig_dist = go.Figure(data=_dist_traces)
        _fig_dist.update_layout(
            height=320, margin=dict(l=10, r=10, t=70, b=10),
            xaxis_title="date", yaxis_title="separation (km)",
            yaxis_rangemode="tozero",
            showlegend=len(_dist_traces) > 1,
            legend=dict(orientation="h", y=1.02, yanchor="bottom"))
        st.plotly_chart(_fig_dist, width="stretch")
        st.caption("Great-circle distance between surfacing positions, pairing "
                   "each profile with the other float's nearest-in-time profile "
                   "(within 15 days). Hover for miles.")
    for _n in _notes:
        st.info(_n)
    st.stop()


_PG_ABOUT = st.Page(page_about, title="About & data API", icon="ℹ️",
                    url_path="about")
_cmp_n = len(st.session_state.get("cmp_list", []))
_PG_COMPARE = st.Page(page_compare,
                      title=(f"Compare floats ({_cmp_n})" if _cmp_n
                             else "Compare floats"),
                      icon="⚖️", url_path="compare")
st.navigation([_PG_EXPLORE, _PG_COMPARE, _PG_FLEET, _PG_ABOUT]).run()

# ---------------- Explore (default page) ----------------
st.title("Argo Float Data Explorer")
st.caption("Search the global Argo array by WMO, serial number, sensor, program "
           "or position (sidebar), or click a float on the map below.")
_nw, _sw = weekly_churn(ROOT, ACTIVE_DAYS,
                        pd.Timestamp.now().normalize().isoformat())
_h1, _h2 = st.columns([3, 1])
_h1.page_link(_PG_FLEET, icon="📊",
              label=f"This week: {len(_nw)} new float(s), {len(_sw)} fell "
                    "silent. Charts and history live in the Fleet overview.")
with _h2.popover("✨ Example floats", width="stretch",
                 help="A hand-picked float in one click, chosen live from the "
                      "index. Nothing downloads until you pick one."):
    _picks, _active_pool = showcase_picks(
        ROOT, ACTIVE_DAYS, pd.Timestamp.now().normalize().isoformat())
    for _k, _lbl, _blurb in (
            ("bgc", "🧪 BGC veteran",
             "An active biogeochemical float measuring oxygen and chlorophyll, "
             "with one of the longest records in the array"),
            ("deep", "🌊 Deep diver",
             "An active Deep Argo float, built to profile far below the usual "
             "2,000 m"),
            ("long", "📈 Longest record",
             "The float with the most reported profiles in the whole index")):
        if _k in _picks:
            st.button(_lbl, help=f"{_blurb} · WMO {_picks[_k]}",
                      key=f"chip_{_picks[_k]}", on_click=_open_float,
                      args=(_picks[_k],), width="stretch")
    if _active_pool:
        st.button("🎲 Surprise me", key="chip_rand",
                  help="A random float from the active array, different every time",
                  on_click=_open_float,
                  args=(str(np.random.choice(_active_pool)),), width="stretch")


# ---- sidebar: search ----
# compact float-type label per WMO (BGC / Core, flagged Deep); drives both the
# sidebar type filter and the table's type column, so the two always agree
def _float_type(dk, deep):
    base = "BGC" if dk == "bgc" else "Core" if dk == "core" else "-"
    return f"{base} · Deep" if deep else base
type_by_wmo = {}
if {"data_kind", "is_deep"}.issubset(floats.columns):
    type_by_wmo = {str(w): _float_type(dk, bool(d)) for w, dk, d in
                   zip(floats["wmo"], floats["data_kind"], floats["is_deep"])}
# raw last_date per float; parsed lazily for only the rows actually shown, since
# it is an Argo YYYYMMDDHHMMSS number that needs parse_argo_date, not a timestamp
last_raw_by_wmo = (dict(zip(floats["wmo"].astype(str), floats["last_date"]))
                   if "last_date" in floats.columns else {})
nprof_by_wmo = (dict(zip(floats["wmo"].astype(str), floats["n_profiles"]))
                if "n_profiles" in floats.columns else {})

st.sidebar.header("Find a float")
# Two different jobs, kept apart. A WMO or a serial IDENTIFIES a float, so it is one
# free-text box; float type, sensor model and location FILTER a set, so they narrow
# whatever the box found. A serial and a WMO can look alike (17106 could be either),
# so rather than make anyone choose a field, search both and let the matched on
# column say which one hit.
q = st.sidebar.text_input(
    "WMO or serial number", key="q_box", placeholder="e.g. 6903132 or 17106",
    help="One box for either. Matches a WMO, a SENSOR_SERIAL_NO or a FLOAT_SERIAL_NO "
         "(case-insensitive substring), and the results say which one matched. Argo "
         "has no serial-to-float index, so this one is built from every float's "
         "meta.nc.")

# WHAT you keep. Every one of these is a property of the float itself, so they all
# belong together and all apply the same way.
st.sidebar.markdown("**Filters**")
st.sidebar.caption("Narrow what you browse. They combine.")
_type_opts = sorted({t for t in type_by_wmo.values() if t and t != "-"})
type_q = st.sidebar.selectbox(
    "Float type", ["(any)"] + _type_opts,
    help="Core = temperature/salinity. BGC = biogeochemical sensors on board. "
         "Deep = profiles below 4000 m (SBE61 floats reach 6000 m). Matches the "
         "type column in the results.")
models = ["(any)"] + sorted(sensors["sensor_model"].dropna().unique().tolist())
model_q = st.sidebar.selectbox("Sensor model", models,
                               help="e.g. SBE41, SBE41CP for the CTD.")
_projects = (["(any)"] + sorted(sensors["project_name"].dropna().astype(str)
                                .str.strip().replace("", np.nan).dropna().unique())
             if "project_name" in sensors.columns else ["(any)"])
project_q = st.sidebar.selectbox(
    "Project / program", _projects, key="project_q",
    help="The program that owns the float, from its meta file: national programs "
         "(Argo Australia, J-ARGO), missions (SOCCOM, GO-BGC), and others. Type to "
         "search the list.")
_pi_labels, _pi_groups = pi_directory(ROOT)
if st.session_state.get("pi_q") not in (["(any)"] + _pi_labels):
    st.session_state["pi_q"] = "(any)"
pi_q = st.sidebar.selectbox(
    "Principal investigator", ["(any)"] + _pi_labels, key="pi_q",
    help="One entry per person, sorted by surname: spelling variants, initials "
         "and multi-PI teams are grouped by surname and first initial. Picking "
         "a person matches every float that names them, alone or in a team. "
         "Type to search.")
_dacs = (["(any)"] + sorted(sensors["dac"].dropna().astype(str).unique())
         if "dac" in sensors.columns else ["(any)"])
dac_q = st.sidebar.selectbox(
    "DAC (data centre)", _dacs, key="dac_q",
    help="The national Data Assembly Centre that processes the float: a rough "
         "country/agency proxy (aoml = US, coriolis = France, csiro = Australia, "
         "jma = Japan, bodc = UK, ...).")
# Was buried inside the location block, so you could not filter to active floats
# unless you were also searching by position. It is a float property, not a
# location setting, so it lives here with the others.
active_only = st.sidebar.checkbox(
    "Active floats only", value=True,
    help=f"A float is active if it reported within {ACTIVE_DAYS} days. Argo floats "
         "surface about every 10 days, so a longer gap almost always means it has "
         "stopped. Untick for historical work. Ignored when you look up a float by "
         "WMO or serial, so a dead float is never hidden from a direct search.")

# WHERE you look. A scope, not a filter, which is why it is its own section: the two
# competing checkboxes ("search near a position" / "show every active float") were
# really one either/or question wearing two hats.
st.sidebar.markdown("**Where**")
where = st.sidebar.radio(
    "Where", ["Anywhere", "Near a position"], index=0, label_visibility="collapsed",
    help="Anywhere maps every float matching your filters, worldwide: the whole live "
         "array by default. Near a position narrows to floats whose LAST KNOWN fix is "
         "inside a radius of a point. The index stores each float's last fix, so "
         "either way this is 'what was last seen here', not 'what has ever sampled "
         "here'.")
near_mode = where == "Near a position"
lat_q, lon_q, radius_q = 39.0, 5.0, 500
if near_mode:
    _region = st.sidebar.selectbox(
        "Jump to a region", ["(custom)"] + list(REGIONS),
        index=1 + list(REGIONS).index(REGION_DEFAULT),
        help="Well-sampled spots to explore. Picking one fills in the coordinates "
             "and radius below, which you can then adjust.")
    # Apply a region BEFORE the coordinate widgets are created: Streamlit only lets
    # you seed a widget's session_state ahead of instantiating it.
    if _region != "(custom)" and _region != st.session_state.get("_last_region"):
        st.session_state["_last_region"] = _region
        _rla, _rlo, _rr = REGIONS[_region]
        st.session_state["lat_q"] = _rla
        st.session_state["lon_q"] = _rlo
        st.session_state["radius_q"] = _rr
    _rd = REGIONS[REGION_DEFAULT]
    st.session_state.setdefault("lat_q", _rd[0])
    st.session_state.setdefault("lon_q", _rd[1])
    st.session_state.setdefault("radius_q", _rd[2])
    _lc1, _lc2 = st.sidebar.columns(2)
    lat_q = _lc1.number_input("Latitude", min_value=-90.0, max_value=90.0,
                              step=0.5, format="%.2f", key="lat_q")
    lon_q = _lc2.number_input("Longitude", min_value=-180.0, max_value=180.0,
                              step=0.5, format="%.2f", key="lon_q")
    radius_q = st.sidebar.slider("Within (km)", min_value=25, max_value=3000,
                                 step=25, key="radius_q",
                                 help="Great-circle distance from the point above.")


st.sidebar.markdown("---")
st.sidebar.caption(
    f"🐛 [Report a bug / request a feature]({NEW_ISSUE_URL})")
st.sidebar.caption(
    "🌊🌊🌊 Data: **International Argo Program** & its member nations, "
    "freely shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
    "[doi.org/10.17882/42182](https://doi.org/10.17882/42182). "
    "With thanks. See **Acknowledgements** up top.")

@st.cache_data(show_spinner=False)
def active_wmo_set(root, days):
    """WMOs reporting within `days`. parse_argo_date is a per-row Python call, so
    over 20k floats it costs ~0.6s: fine when a radius has already cut the list to
    a handful, but not on every rerun of the whole-globe view. Cache it."""
    _s, fl = load_tables(root)
    if fl is None or "last_date" not in fl.columns:
        return set()
    age = (pd.Timestamp.now().normalize()
           - pd.to_datetime(fl["last_date"].map(parse_argo_date), errors="coerce")).dt.days
    return set(fl.loc[age.le(days).fillna(False), "wmo"].astype(str))


def floats_near(lat, lon, km):
    """WMOs whose last known fix is within km of (lat, lon) -> {wmo: distance_km}.
    Pure index lookup: last_lat/last_lon are already in the crosswalk, so this costs
    no GDAC fetch. Active filtering lives in search(), not here."""
    if not {"last_lat", "last_lon"}.issubset(floats.columns):
        return {}
    f = floats[["wmo", "last_lat", "last_lon", "last_date"]].dropna(
        subset=["last_lat", "last_lon"])
    d = haversine_km(lat, lon, f["last_lat"].to_numpy("float64"),
                     f["last_lon"].to_numpy("float64"))
    f = f.assign(_km=d)
    f = f[f["_km"] <= km]
    return dict(zip(f["wmo"].astype(str), f["_km"]))


# ---- sidebar: watchlist (carried in the page URL, so a bookmark keeps it) ----
_watchlist = st.session_state.get("watchlist", [])
if _watchlist:
    st.sidebar.markdown("**Watchlist**")
    st.sidebar.caption("Carried in the page URL: bookmark the page to keep the "
                       "list. 🔴 means the float has stopped reporting.")
    for _w in list(_watchlist):
        _wlp = parse_argo_date(last_raw_by_wmo.get(_w))
        _wage = ((pd.Timestamp.now().normalize() - _wlp).days
                 if pd.notna(_wlp) else None)
        _dot = "⚪" if _wage is None else ("🟢" if _wage <= ACTIVE_DAYS else "🔴")
        _lbl = f"{_dot} {_w}" + (f" · {_wlp:%Y-%m-%d}" if pd.notna(_wlp) else "")
        _wc1, _wc2 = st.sidebar.columns([5, 1])
        _wc1.button(_lbl, key=f"watch_open_{_w}", on_click=_open_float, args=(_w,),
                    width="stretch",
                    help="Open this float. The date is its last reported profile.")
        _wc2.button("✕", key=f"watch_del_{_w}", on_click=_toggle_watch, args=(_w,),
                    help="Remove from the watchlist.")

st.sidebar.divider()
st.sidebar.caption(
    "**[Data API](https://vladsimontov.github.io/ArgoFloatData/api.html)** · query "
    "the whole index (floats, sensors, serial numbers) over plain HTTP with DuckDB "
    "or pandas. Free, no key, refreshed daily.")


# resolve search -> candidate floats
def search():
    """Two modes, deliberately not mixed.

    A typed WMO/serial is a JUMP: you named a float, so you get that float and every
    filter is ignored. The subtler rule (bypass only the filters that default to ON)
    was worse in practice: a stale Float type from an earlier browse made a valid WMO
    return nothing, with no clue why.

    With no identifier we are BROWSING, and the filters narrow the set."""
    df = sensors.copy()
    if q.strip():
        # OR across the three things a float can be named by, so nobody has to know
        # whether 17106 is a WMO or a serial. regex=False: real serials contain
        # characters that would otherwise be read as a pattern.
        s = q.strip().lower()
        return df[df["wmo"].astype(str).str.contains(s, regex=False)
                  | df["sensor_serial_no"].fillna("").str.lower().str.contains(s, regex=False)
                  | df["float_serial_no"].fillna("").str.lower().str.contains(s, regex=False)]
    if model_q != "(any)":
        df = df[df["sensor_model"].fillna("") == model_q]
    if project_q != "(any)":
        df = df[df["project_name"].fillna("").astype(str).str.strip() == project_q]
    if pi_q != "(any)":
        df = df[df["pi_name"].fillna("").astype(str).str.strip()
                .isin(_pi_groups.get(pi_q, set()))]
    if dac_q != "(any)":
        df = df[df["dac"].fillna("").astype(str) == dac_q]
    if type_q != "(any)":
        df = df[df["wmo"].astype(str).map(type_by_wmo).eq(type_q)]
    if active_only:
        df = df[df["wmo"].astype(str).isin(active_wmo_set(ROOT, ACTIVE_DAYS))]
    if near_mode:
        df = df[df["wmo"].astype(str).isin(near_km)]
    return df

near_km = floats_near(lat_q, lon_q, radius_q) if near_mode else {}
hits = search()

if True:   # there is always something to show: a lookup, or the browse view
    n_floats = int(hits["wmo"].nunique())
    st.markdown(f"#### Matches · {n_floats:,} float{'s' if n_floats != 1 else ''}")
    if n_floats == 0 and q.strip():
        # A dead end should say what to do next. Typing a sensor model into the
        # identifier box is the obvious mistake, so name it rather than just
        # returning nothing.
        _qs = q.strip()
        _mm = next((m for m in models[1:] if m.lower() == _qs.lower()), None)
        if _mm:
            st.info(f"**{_qs}** is a sensor model, not a WMO or serial number. Set "
                    f"**Sensor model** to `{_mm}` in the sidebar filters to list every "
                    "float carrying one.")
        else:
            st.info(f"No float matches **{_qs}**. This box takes a WMO or a serial "
                    "number; for a sensor model or a float type, use the filters "
                    "below it. Serial formats vary by DAC, so try a shorter fragment.")
    # The box searches WMO and both serials at once, so every hit must say which of
    # them it was: that is the whole reason the user never had to pick a field.
    serial_hit = bool(q.strip())
    if serial_hit:
        _sq = q.strip().lower()
        _ssn = hits["sensor_serial_no"].fillna("").astype(str)
        _fsn = hits["float_serial_no"].fillna("").astype(str)
        _wmo = hits["wmo"].astype(str)
        _snm = hits["sensor_model"].fillna("").astype(str)
        _snm = _snm.where(_snm.ne(""), hits["sensor"].fillna("sensor").astype(str))
        hits = hits.assign(matched_on=np.where(
            _ssn.str.lower().str.contains(_sq, regex=False), _snm + " · " + _ssn,
            np.where(_fsn.str.lower().str.contains(_sq, regex=False),
                     "float s/n " + _fsn,
                     np.where(_wmo.str.contains(_sq, regex=False), "WMO " + _wmo, ""))))

    # always one row per float (WMO), with how many sensors matched and their models
    agg = dict(float_serial_no=("float_serial_no", "first"),
               dac=("dac", "first"),
               n_sensors=("sensor", "nunique"),
               sensor_models=("sensor_model",
                              lambda s: ", ".join(sorted(s.dropna().unique()))))
    if serial_hit:
        agg["matched_on"] = ("matched_on",
                             lambda s: ", ".join(sorted({x for x in s if x})))
    show = hits.groupby("wmo", as_index=False).agg(**agg).reset_index(drop=True)
    show.insert(1, "type", show["wmo"].astype(str).map(type_by_wmo).fillna("-"))
    # Still reporting? Argo floats surface roughly every 10 days, so a gap much
    # longer than that means the float has almost certainly stopped.
    _lp = pd.to_datetime(
        show["wmo"].astype(str).map(last_raw_by_wmo).map(parse_argo_date),
        errors="coerce")
    _age = (pd.Timestamp.now().normalize() - _lp).dt.days
    show.insert(2, "status", np.where(_lp.isna(), "unknown",
                                      np.where(_age <= ACTIVE_DAYS,
                                               "active", "inactive")))
    show.insert(3, "last profile", _lp)
    show.insert(4, "profiles",
                pd.to_numeric(show["wmo"].astype(str).map(nprof_by_wmo),
                              errors="coerce").astype("Int64"))
    browsing = not q.strip()
    if browsing and near_mode:
        # distance from the searched point; nearest first. Meaningless with no point,
        # so "Anywhere" gets no km away column rather than a distance from nowhere.
        show.insert(5, "km away", show["wmo"].astype(str).map(near_km).round(0))
        show = show.sort_values("km away").reset_index(drop=True)
    elif q.strip():
        st.caption(f"Looking up “{q.strip()}” across the whole array. Filters are "
                   "ignored for a direct lookup, so a float you name is never hidden "
                   "by a filter you forgot was on. Clear the box to browse again.")
    if serial_hit:
        show.insert(6 if (browsing and near_mode) else 5, "matched_on",
                    show.pop("matched_on"))
    # The map and table stay visible even while a float is open (Vlad's call:
    # auto-collapsing the map felt wrong); the scroll-on-open below still takes
    # the viewport to the float, and scrolling back up finds the map in place.
    _res_box = st.container()
    with _res_box:
        # The map always shows whatever is currently matched: every active float on open,
        # a narrower set as filters go on, a single float for a lookup. It used to be
        # suppressed for lookups, back when it only meant "near this point".
        _map_wmo = None
        if len(show):
            _mp = show[["wmo", "type"] + (["km away"] if "km away" in show.columns
                                          else [])].copy()
            _mp["lat"] = _mp["wmo"].astype(str).map(
                dict(zip(floats["wmo"].astype(str), floats["last_lat"])))
            _mp["lon"] = _mp["wmo"].astype(str).map(
                dict(zip(floats["wmo"].astype(str), floats["last_lon"])))
            _mp = _mp.dropna(subset=["lat", "lon"])
            # list-valued column, so each dot takes its own type color
            _mp["color"] = [TYPE_COLORS.get(t, TYPE_COLOR_OTHER) for t in _mp["type"]]
            if len(_mp):
                # Fit the view to the hits rather than always showing the globe: a single
                # float would otherwise be one pixel lost in an ocean.
                _la = _mp["lat"].to_numpy("float64"); _lo = _mp["lon"].to_numpy("float64")
                _span = max(float(_la.max() - _la.min()), float(_lo.max() - _lo.min()), 0.5)
                _fit_zoom = next((z for thr, z in [(1, 4), (3, 3.5), (8, 3), (20, 2.5),
                                                   (50, 2), (120, 1.2)] if _span < thr), 0.9)
                _pt = pd.DataFrame({"lat": [lat_q], "lon": [lon_q]})
            # deck.gl for every view. A real projection (plotly Scattergeo) fixed the
            # doubled world and the cropped poles, but gave every float its own SVG node:
            # 4,299 of them for the live array, against ONE WebGL canvas here. The world
            # repeating is the cheaper problem to live with than a map that drags.
            if len(_mp):
                # id= is required for Streamlit to report pydeck selections
                _mev = st.pydeck_chart(pdk.Deck(
                    map_style=None,
                    initial_view_state=pdk.ViewState(
                        latitude=float(lat_q) if near_mode else float(np.median(_la)),
                        longitude=float(lon_q) if near_mode else float(np.median(_lo)),
                        zoom=(next((z for thr, z in [(150, 5), (400, 4), (900, 3),
                                                     (2000, 2)] if radius_q < thr), 1)
                              if near_mode else _fit_zoom)),
                    layers=[
                        pdk.Layer("ScatterplotLayer", id="hits", data=_mp,
                                  get_position="[lon, lat]",
                                  get_fill_color="color", get_radius=18000,
                                  radius_min_pixels=4, radius_max_pixels=9,
                                  pickable=True, auto_highlight=True),
                        *([] if not near_mode else
                          [pdk.Layer("ScatterplotLayer", id="point", data=_pt,
                                     get_position="[lon, lat]",
                                     get_fill_color=[214, 40, 40], get_radius=26000,
                                     radius_min_pixels=7, radius_max_pixels=11,
                                     pickable=False)])],
                    tooltip={"text": "float {wmo} · {type}" + ("\n{km away} km away" if near_mode else "")}),
                    height=420, on_select="rerun", selection_mode="single-object",
                    key="matches_map")
                # clicking a dot opens that float, same as ticking its row
                _msel = getattr(_mev, "selection", None)
                _mobjs = ((_msel.get("objects") if isinstance(_msel, dict)
                           else getattr(_msel, "objects", None)) or {})
                _mhits = _mobjs.get("hits") or []
                if _mhits:
                    _map_wmo = str(_mhits[0].get("wmo"))
            if len(_mp):
                # legend: only the types actually on this map, in the table's order
                def _dot(rgba, label):
                    return (f"<span style='display:inline-block;width:9px;height:9px;"
                            f"border-radius:50%;vertical-align:middle;margin:0 4px 2px 0;"
                            f"background:rgba({rgba[0]},{rgba[1]},{rgba[2]},"
                            f"{rgba[3] / 255:.2f})'></span>{label}")
                _present = [t for t in TYPE_COLORS if (_mp["type"] == t).any()]
                _other = bool((~_mp["type"].isin(TYPE_COLORS)).any())
                _leg = [_dot(TYPE_COLORS[t], f"{t} ({int((_mp['type'] == t).sum())})")
                        for t in _present]
                if _other:
                    _leg.append(_dot(TYPE_COLOR_OTHER, "other"))
                if near_mode:
                    _leg.append(_dot([214, 40, 40, 255], "your search point"))
                st.markdown(
                    "<div style='font-size:0.95rem;opacity:0.85'>"
                    + " &nbsp;·&nbsp; ".join(_leg) + "</div>", unsafe_allow_html=True)
                st.caption(
                    (f"The {len(_mp):,} float(s) matching your search, at their last known "
                     "fix" if not near_mode else
                     f"Each float's last known fix within {radius_q:,} km of "
                     f"({lat_q:.2f}, {lon_q:.2f})")
                    + " · blue = Core, green = BGC, darker = Deep · click a float to open "
                      "it. Floats drift, so a fix is where it was last heard from, not "
                      "where it is now or where it was during its lifetime")
        cap = "Select a row's checkbox (far left) to open that float ↓"
        if serial_hit:
            cap += "  ·  the highlighted **matched on** column shows what your search hit"
        st.caption(cap)
        _cfg = {
            "status": st.column_config.TextColumn(
                "status", help=f"Active = this float reported a profile within the last "
                               f"{ACTIVE_DAYS} days. Argo floats surface about every 10 "
                               "days, so a longer gap almost always means the float has "
                               "stopped reporting. Taken from the index's last profile "
                               "date, which is refreshed daily."),
            "last profile": st.column_config.DatetimeColumn(
                "last profile", format="YYYY-MM-DD",
                help="Date of the most recent profile this float reported."),
            "profiles": st.column_config.NumberColumn(
                "profiles", format="%d",
                help="How many profiles (dives) this float has reported, from the index. "
                     "A long record means more history to plot; the Raw tab pulls these "
                     "one cycle at a time."),
            "km away": st.column_config.NumberColumn(
                "km away", format="%.0f km",
                help="Great-circle distance from your search point to this float's last "
                     "known fix. The float has drifted since, so treat it as approximate."),
        }
        if serial_hit:
            _cfg["matched_on"] = st.column_config.TextColumn(
                "matched on", help="Which identifier your search hit: a sensor serial, "
                                   "the float serial, or the WMO.")
        wmos = sorted(hits["wmo"].astype(str).unique().tolist())
        # A float opens from three places: a map dot, a row checkbox, or the dropdown.
        # Resolve them all here, BEFORE the table renders, so the green highlight tracks
        # whatever is actually open on this same rerun rather than lagging one click.
        # Forget remembered clicks whenever the result set changes, and act only on a
        # *new* selection so the dropdown can still override the other two.
        sig = (q, model_q, type_q, project_q, pi_q, dac_q,
               near_mode, lat_q, lon_q, radius_q, active_only)
        # Bring the map to the top when the search CHANGES, so results land as map above /
        # table below instead of below a screenful of header. Deliberately not on first
        # load (an arriving visitor should see the title) and not on every rerun, or
        # opening a float would yank you back up away from the float you just opened.
        if "_scroll_sig" not in st.session_state:
            st.session_state["_scroll_sig"] = sig          # first load: record, don't move
        elif st.session_state["_scroll_sig"] != sig:
            st.session_state["_scroll_sig"] = sig
            _pick_pending = (st.session_state.get("wmo_pick") is not None
                             and str(st.session_state.get("wmo_pick"))
                             != str(st.session_state.get("_focus_wmo")))
            if not _pick_pending:
                st.iframe(f"""
                <script>
                // the sig is embedded so this re-executes only when the search changes
                const _s = {abs(hash(sig)) % 10**8};
                const d = window.parent.document;
                // the global map is a plotly geo chart and the regional one is deck.gl, so
                // match either; retry briefly because the element may not be mounted yet
                function go(n) {{
                  const el = d.querySelector("[data-testid='stDeckGlJsonChart'], "
                                           + "[data-testid='stPlotlyChart']")
                          || d.querySelector("[data-testid='stDataFrame']");
                  if (el) el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                  else if (n > 0) setTimeout(function() {{ go(n - 1); }}, 200);
                }}
                go(12);
                </script>
                """, height=1)
        if st.session_state.get("_match_sig") != sig:
            st.session_state["_match_sig"] = sig
            st.session_state.pop("_last_row", None)
            st.session_state.pop("_last_map_wmo", None)
            st.session_state.pop("_pin_wmo", None)
            st.session_state["matches_shown"] = []
        # The table's selection is by row INDEX and is read-only, so it is read from the
        # key used for the *previous* render, before any reordering below.
        _mver = st.session_state.setdefault("matches_ver", 0)
        _ms = st.session_state.get(f"matches_table_{_mver}")
        # The wmo order as actually DISPLAYED last render. A row click reports an index
        # into that order, so it must be resolved against this and not against `show`,
        # which is rebuilt unpinned each run: pinning shifts every row down by one, so
        # using show.iloc[] here opened the wrong float.
        _shown = st.session_state.get("matches_shown", [])
        # map click (the pydeck chart above already reported this rerun's selection).
        # Only a map click pins: its row is usually far below the fold. A row click
        # needs no pinning and reordering under the cursor would just be jarring.
        if _map_wmo is not None and _map_wmo != st.session_state.get("_last_map_wmo"):
            st.session_state["_last_map_wmo"] = _map_wmo
            st.session_state["wmo_pick"] = _map_wmo
            st.session_state["_pin_wmo"] = _map_wmo
        # row checkbox: read the widget state recorded by the click that caused this rerun
        _seld = ((_ms.get("selection") if isinstance(_ms, dict)
                  else getattr(_ms, "selection", None)) if _ms is not None else None)
        _selrows = ((_seld.get("rows", []) if isinstance(_seld, dict)
                     else getattr(_seld, "rows", []) or []) if _seld is not None else [])
        _selrow0 = _selrows[0] if _selrows else None
        if (_selrow0 is not None and _selrow0 < len(_shown)
                and _selrow0 != st.session_state.get("_last_row")):
            st.session_state["_last_row"] = _selrow0
            st.session_state["wmo_pick"] = _shown[_selrow0]
        _open = st.session_state.get("wmo_pick")
        # Streamlit can neither scroll a dataframe to a row nor tick its checkbox, so a
        # map-clicked float is pinned to the top instead, otherwise its row is lost far
        # below the fold. The pin survives later row clicks so the table stays put.
        _pin = st.session_state.get("_pin_wmo")
        if _pin is not None and (show["wmo"].astype(str) == str(_pin)).any():
            _ix = show.index[show["wmo"].astype(str) == str(_pin)]
            show = pd.concat([show.loc[_ix], show.drop(index=_ix)]).reset_index(drop=True)
        # Retire the table whenever the displayed order changes: its selection is by row
        # index, so a stale tick would otherwise sit on the wrong float.
        _order = show["wmo"].astype(str).tolist()
        if _order != _shown:
            st.session_state["matches_ver"] = _mver = _mver + 1
            st.session_state.pop("_last_row", None)
        st.session_state["matches_shown"] = _order
        _hl = (_order.index(str(_open)) if _open is not None and str(_open) in _order
               else None)
        # Styler colors are explicit, so the green holds in both light and dark mode.
        _sty = show.style
        if serial_hit:
            _sty = _sty.set_properties(subset=["matched_on"],
                                       **{"background-color": "#fff3cd", "color": "#663c00"})
        if _hl is not None:
            _sty = _sty.apply(lambda r: (["background-color:#c3ecd0; color:#0b6b3a"] * len(r)
                                         if r.name == _hl else [""] * len(r)), axis=1)
        st.dataframe(_sty, width="stretch", height=220,
                     on_select="rerun", selection_mode="single-row",
                     key=f"matches_table_{_mver}", column_config=_cfg)
        if _pin is not None and _hl is not None and str(_pin) == str(_open):
            st.caption(f"Float {_open} was picked on the map, so it is pinned to the top; "
                       "the rest keep their order.")
else:
    st.info("Tick **Search near a position** in the sidebar to browse the array on a "
            "map, or type a WMO or serial number to jump to a float.")
    wmos = []

if not wmos:
    # nothing is open, so the URL should not claim a float
    st.query_params.pop("wmo", None)
    st.query_params.pop("var", None)
    st.session_state.pop("_focus_wmo", None)
    st.stop()

# keep the picker in sync with the table; drop a stale pick from a prior search
if st.session_state.get("wmo_pick") not in wmos:
    st.session_state.pop("wmo_pick", None)
sel_wmo = st.selectbox("Select a float (WMO) to inspect", wmos, key="wmo_pick",
                       index=None, placeholder="Choose a float to load its profile",
                       help="Or click a row's checkbox in the Matches table above. The "
                            "profile is fetched from the GDAC only once you pick a "
                            "float, so search results appear instantly.")
if not sel_wmo:
    st.info("Pick a float to load its profile: click a row's checkbox in the table "
            "above, or use the selector. Nothing is downloaded until you choose one, so "
            "your search results stay instant.")
    st.query_params.pop("wmo", None)
    st.query_params.pop("var", None)
    st.session_state.pop("_focus_wmo", None)
    st.stop()

# the URL always names the open float, so the address bar is a shareable link
if st.query_params.get("wmo") != str(sel_wmo):
    st.query_params["wmo"] = str(sel_wmo)
# focus the float when it (re)opens: a map dot, row tick, chip or link click lands
# the viewport on the float content instead of leaving it below the fold
if st.session_state.get("_focus_wmo") != str(sel_wmo):
    st.session_state["_focus_wmo"] = str(sel_wmo)
    st.iframe(f"""
    <script>
    const _w = "{sel_wmo}";   // embedded so this re-executes per opened float
    const d = window.parent.document;
    function go(n) {{
      const el = d.querySelector("[data-testid='stTabs']");
      if (el) el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
      else if (n > 0) setTimeout(function() {{ go(n - 1); }}, 200);
    }}
    go(15);
    </script>
    """, height=1)
_lc1, _lc2, _lc3 = st.columns([2.4, 1, 1])
_lc1.caption("🔗 This view is linkable: the page URL now names this float. Copy it "
             "from the address bar to share.")
_watching = str(sel_wmo) in st.session_state.get("watchlist", [])
_lc2.button("★ Watching · remove" if _watching else "☆ Watch this float",
            key="watch_toggle", on_click=_toggle_watch, args=(str(sel_wmo),),
            width="stretch",
            help="Watched floats appear in the sidebar with a status light. The "
                 "list rides in the page URL, so a bookmark keeps it.")
_lc3.button("⚖️ Add to Compare List", key="goto_cmp", width="stretch",
            on_click=_add_to_compare, args=(str(sel_wmo),),
            help="Collects up to 3 floats; the Compare floats page in the sidebar "
                 "shows how many are on the list and plots them side by side "
                 "when you open it.")

# ---- resolve the selected float and load its data ----
frow = floats[floats["wmo"].astype(str) == str(sel_wmo)]
frow = frow.iloc[0] if len(frow) else None
fsens = sensors[sensors["wmo"].astype(str) == str(sel_wmo)]

sprof_rel = frow.get("sprof_path") if frow is not None else None
has_local = bool(sprof_rel) and os.path.exists(os.path.join(ROOT, str(sprof_rel)))
ds = load_sprof(os.path.join(ROOT, str(sprof_rel))) if has_local else None
if has_local:
    try:
        _mb = os.path.getsize(os.path.join(ROOT, str(sprof_rel))) / 1e6
        st.caption(f"✓ {os.path.basename(str(sprof_rel))} is already cached here "
                   f"({_mb:.1f} MB), no download needed.")
    except OSError:
        pass

# measurands from the crosswalk / meta


def _param_str(row, key):
    # NaN is truthy in Python, so guard with notna before using as a string.
    v = row.get(key) if row is not None else None
    return str(v).strip() if pd.notna(v) else ""


params_str = _param_str(frow, "parameters") or _param_str(frow, "parameters_meta")
measurands = [p for p in params_str.split() if p]

# fetch the profile file on demand from the GDAC if not cached locally
if not has_local:
    rel = frow.get("expected_rel") if frow is not None else None
    if not (isinstance(rel, str) and rel):
        suffix = "_Sprof.nc" if str(frow.get("data_kind")) == "bgc" else "_prof.nc"
        rel = f"dac/{frow.get('dac')}/{sel_wmo}/{sel_wmo}{suffix}"
    _dest = os.path.join(ROOT, rel)
    if os.path.exists(_dest) and os.path.getsize(_dest) > 0:
        # on disk from an earlier visit: no download, so no fetch banner either
        status = "ok"
        st.caption(f"✓ {os.path.basename(rel)} is already cached here "
                   f"({os.path.getsize(_dest) / 1e6:.1f} MB), no download needed.")
    else:
        st.info(f"Fetching this float's data from the Argo GDAC on demand "
                f"(`{os.path.basename(rel)}`), cached after the first load.")
        _base = os.path.basename(rel)
        _dlbar = st.progress(0.0, text=f"Contacting the GDAC for {_base}…")
        _dllast = [-1.0]

        def _dl_tick(done, total):
            # throttle to whole-percent steps so the bar does not flood the websocket
            if total:
                frac = min(done / total, 1.0)
                if frac - _dllast[0] < 0.01 and frac < 1.0:
                    return
                _dllast[0] = frac
                _dlbar.progress(frac, text=f"Downloading {_base} · "
                                           f"{done / 1e6:.1f} of {total / 1e6:.1f} MB")
            elif done / 1e6 - _dllast[0] >= 0.5:
                _dllast[0] = done / 1e6
                _dlbar.progress(0.0, text=f"Downloading {_base} · "
                                          f"{done / 1e6:.1f} MB so far")

        status = _gdac_get(rel, progress=_dl_tick)
        _dlbar.empty()
    if status == "unreachable":
        st.error("Couldn't reach the Argo GDAC to download this float's data. This looks "
                 "like a connection problem, not a missing file. Please try again in a "
                 "moment.")
        st.stop()
    if status == "missing":
        st.error(f"The GDAC has no data file at `{rel}` for this float. It may not have "
                 "reported profiles yet, or its data lives under a different name.")
        st.stop()
    sprof_rel = rel
    has_local = True
    ds = load_sprof(os.path.join(ROOT, rel))

# refine measurands from the file if the meta list was empty
if not measurands:
    measurands = [v for v in ds.data_vars
                  if ds[v].dims == ("N_PROF", "N_LEVELS")
                  and not v.endswith(("_QC", "_ADJUSTED", "_ADJUSTED_QC",
                                      "_ADJUSTED_ERROR"))]

# WMO tag stamped on every chart (which float + how fresh)
_last_juld = pd.Series(ds["JULD"].values).max() if "JULD" in ds else pd.NaT
_ncyc = int(ds.sizes.get("N_PROF", 0))
float_tag = f"Float {sel_wmo}"
if pd.notna(_last_juld):
    float_tag += f"  ·  latest profile {pd.Timestamp(_last_juld):%Y-%m-%d}"
if _ncyc:
    float_tag += f"  ·  {_ncyc} cycles"


def _titled(fig, text):
    fig.update_layout(title=dict(text=text, x=0.0, xanchor="left",
                                 font=dict(size=16)), margin=dict(t=48))
    return fig


def _axis_label(var):
    units = ds[var].attrs.get("units") if var in ds else None
    return f"{var} [{units}]" if units else var


_OVL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]   # T blue, S red, O2 green, +


def _rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _stat_profile(sub, nbins=140):
    """Mean and standard deviation at each pressure bin, from a tidy frame with
    pres/value columns. Returns (bin_centers, means, stds) or None."""
    pr = sub["pres"].to_numpy("float64")
    vv = sub["value"].to_numpy("float64")
    m = np.isfinite(pr) & np.isfinite(vv)
    pr, vv = pr[m], vv[m]
    if pr.size < 3 or pr.min() == pr.max():
        return None
    edges = np.linspace(pr.min(), pr.max(), nbins + 1)
    idx = np.clip(np.digitize(pr, edges) - 1, 0, nbins - 1)
    g = pd.Series(vv).groupby(idx)
    mean = g.mean()
    std = g.std().fillna(0.0)          # single-sample bins have no spread
    if len(mean) < 3:
        return None
    cen = (0.5 * (edges[:-1] + edges[1:]))[mean.index.to_numpy()]
    return cen, mean.to_numpy(), std.to_numpy()


def _stacked_layout(items):
    """Shared layout for the pressure-vs-measurand plots: a reversed pressure
    y-axis with one color-matched x-axis per measurand, the first on the bottom
    and the rest stacked on top. items is [(name, units)]. Returns (layout, xaxis
    refs). Used by both the mean overlay and the individual-casts view."""
    ntop = len(items) - 1
    # Each stacked top axis needs a fixed vertical band for its title + tick
    # labels. Grow the figure by a band per top axis (rather than cramming them
    # into a fixed height) so titles never smear into the neighbouring ticks.
    plot_px, band_px, head_px = 560, 88, 44   # plot area, per-axis band, top headroom
    height = plot_px + (band_px * ntop + head_px if ntop else 0)
    step = band_px / height if ntop else 0.0
    ytop = plot_px / height if ntop else 1.0
    layout = {"height": height, "margin": dict(l=64, r=24, t=10, b=46),
              "showlegend": False,
              "yaxis": dict(title="PRES [decibar]", autorange="reversed",
                            domain=[0.0, ytop])}
    xas = []
    for i, (name, units) in enumerate(items):
        color = _OVL_COLORS[i % len(_OVL_COLORS)]
        xas.append("x" if i == 0 else f"x{i + 1}")
        title = f"{name} [{units}]" if units else name
        ax = dict(title=dict(text=title, font=dict(color=color, size=14)),
                  tickfont=dict(color=color, size=12), showgrid=(i == 0),
                  zeroline=False)
        if i == 0:
            ax["side"] = "bottom"
            layout["xaxis"] = ax
        else:
            ax.update(overlaying="x", side="top", anchor="free",
                      position=min(1.0, ytop + step * (i - 1) + step * 0.18))
            layout[f"xaxis{i + 1}"] = ax
    return layout, xas


def _casts_fig(series, units_map):
    """Every cast drawn discretely, never averaged, on the shared stacked
    pressure/x-axis layout. series is [(name, [(label, pres, val, cfrac)])], one
    entry per cast, where a cast is one N_PROF of one cycle: a cycle's primary and
    near-surface sampling schemes are separate casts and must not be joined, or the
    line leaps from the deepest point back up to the surface. cfrac (0-1) places
    the cast on the date ramp. One measurand colors by date (the drift view); with
    several, casts take their measurand's color at low opacity."""
    layout, xas = _stacked_layout([(nm, units_map.get(nm)) for nm, _ in series])
    fig = go.Figure()
    single = len(series) == 1
    for i, (nm, casts) in enumerate(series):
        base = _OVL_COLORS[i % len(_OVL_COLORS)]
        cols = (px.colors.sample_colorscale("Viridis", [c[3] for c in casts])
                if single and casts else None)
        for j, (label, pres, val, _cf) in enumerate(casts):
            fig.add_trace(go.Scattergl(
                x=val, y=pres, mode="lines",
                line=dict(color=cols[j] if cols else _rgba(base, 0.45), width=1.3),
                xaxis=xas[i], yaxis="y", showlegend=False,
                hovertemplate=(f"{label}<br>{nm}=%{{x:.4g}}"
                               "<br>PRES=%{y:.1f} dbar<extra></extra>")))
    fig.update_layout(**layout)
    return fig


def _overlay_fig(profiles, show_band=True):
    """Multi-x-axis overlay: a shared reversed pressure y-axis, one color-matched
    x-axis per measurand (first on the bottom, the rest stacked on top). Each entry
    is (name, pres, mean, std, units); with show_band a shaded +/- 1 std ribbon is
    drawn behind each mean line."""
    layout, xas = _stacked_layout([(nm, u) for nm, _, _, _, u in profiles])
    fig = go.Figure()
    for i, (name, pres, mean, std, units) in enumerate(profiles):
        color = _OVL_COLORS[i % len(_OVL_COLORS)]
        xa = xas[i]
        if show_band and np.any(std > 0):
            fig.add_trace(go.Scatter(
                x=np.concatenate([mean + std, (mean - std)[::-1]]),
                y=np.concatenate([pres, pres[::-1]]),
                fill="toself", fillcolor=_rgba(color, 0.13), line=dict(width=0),
                hoverinfo="skip", showlegend=False, xaxis=xa, yaxis="y"))
        fig.add_trace(go.Scatter(
            x=mean, y=pres, mode="lines", line=dict(color=color, width=2), name=name,
            xaxis=xa, yaxis="y", customdata=std,
            hovertemplate=f"{name}=%{{x:.4g}} ± %{{customdata:.3g}}"
                          "<br>PRES=%{y:.0f} dbar<extra></extra>"))
    fig.update_layout(**layout)
    return fig


def fetch_raw_many(dac, wmo, cycles, is_bgc, progress=None):
    """Fetch several per-cycle raw files from the GDAC in parallel. Returns
    {cycle: (rel, status)} with status ok/missing/unreachable. Latency-bound, so a
    small pool is a big win; capped low to stay polite to the GDAC. progress(done,
    total, cycle) is called as each finishes."""
    out, done = {}, 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_raw_profile, dac, wmo, c, is_bgc): c for c in cycles}
        for f in as_completed(futs):
            c = futs[f]
            try:
                out[c] = f.result()
            except Exception:
                out[c] = (None, "unreachable")
            done += 1
            if progress:
                progress(done, len(cycles), c)
    return out


def _human_bytes(n):
    """Byte count as a short human-readable size (1 decimal from KB up)."""
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{int(x)} {unit}" if unit == "B" else f"{x:,.1f} {unit}"
        x /= 1024


def _merge_core_ts(df, core_path, nprof, nlev, units):
    """Splice the CTD's TEMP/PSAL (and their QC) from a core profile file onto a BGC
    B-file frame for the same cycle. A BGC B-file holds PRES + biogeochemistry but
    not TEMP/PSAL; the core file for that cycle carries them on an identical PRES
    axis (same N_PROF/N_LEVELS). When the grids match we drop the columns straight
    on by position (no float-equality risk); if they somehow differ we align on
    pressure and leave unmatched rows blank rather than invent values."""
    try:
        cds = load_raw(core_path)
    except Exception:
        return df
    if "PRES" not in cds:
        return df
    CP = np.asarray(cds["PRES"].values, dtype="float64")
    if CP.ndim == 1:
        CP = CP.reshape(1, -1)
    cdims = cds["PRES"].dims

    want = {}
    for v in ("TEMP", "PSAL"):
        if v not in cds or cds[v].dims != cdims:
            continue
        a = np.asarray(cds[v].values, dtype="float64")
        a = a.reshape(1, -1) if a.ndim == 1 else a
        if a.shape != CP.shape:
            continue
        want[v] = a.ravel()
        units[v] = cds[v].attrs.get("units")
        q = f"{v}_QC"
        if q in cds:
            qa = np.asarray(cds[q].values)
            qa = qa.reshape(1, -1) if qa.ndim == 1 else qa
            if qa.shape == CP.shape:
                f = np.array([_flag_str(x) for x in qa.ravel()], dtype=object)
                want[q] = np.where(np.isin(f, ("", " ", "nan", "None", "--")), "", f)
    if not want:
        return df

    if CP.shape == (nprof, nlev):
        for k, arr in want.items():
            df[k] = arr
    else:
        # grids differ: align on (n_prof, pressure). Drop fill pressures first so the
        # repeated 99999 padding can't fan a left join out into a many-to-many blow-up.
        core = pd.DataFrame({"n_prof": np.repeat(np.arange(CP.shape[0]), CP.shape[1]),
                             "pres": CP.ravel(), **want})
        core = core[np.isfinite(core["pres"]) & (core["pres"] < 90000)]
        core = core.drop_duplicates(subset=["n_prof", "pres"], keep="first")
        df = df.merge(core, on=["n_prof", "pres"], how="left")

    # sit T/S (and their QC) right after the pressure column, not at the far end,
    # so the CTD reading reads next to the depth it was taken at
    added = [c for c in want if c in df.columns]
    rest = [c for c in df.columns if c not in added]
    i = rest.index("pres") + 1
    return df[rest[:i] + added + rest[i:]]


def raw_cycle_frame(path, cycle, core_path=None):
    """One cycle's raw file -> a wide tidy frame: cycle/time/lat/lon/pres + every
    numeric measurand on the PRES grid (intermediate sensor signals included) plus
    each parameter's decoded QC flag, so a CSV export carries everything the file
    holds for offline work. For a BGC float, pass core_path to merge the CTD's
    TEMP/PSAL in from the core file for the same cycle (see _merge_core_ts)."""
    rds = load_raw(path)
    if "PRES" not in rds:
        return None, {}
    P = np.asarray(rds["PRES"].values, dtype="float64")
    if P.ndim == 1:
        P = P.reshape(1, -1)
    nprof, nlev = P.shape
    dims = rds["PRES"].dims
    cols, units = {"pres": P.ravel()}, {}
    for v in rds.data_vars:
        if v == "PRES" or rds[v].dims != dims:
            continue
        if v.endswith("_QC"):
            # flags arrive as bytes (b'1'); decode so they survive to CSV, and blank
            # the missing ones so importers see an empty cell, not the text "nan"
            a = np.asarray(rds[v].values)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if a.shape == (nprof, nlev):
                f = np.array([_flag_str(x) for x in a.ravel()], dtype=object)
                cols[v] = np.where(np.isin(f, ("", " ", "nan", "None", "--")), "", f)
        elif np.issubdtype(rds[v].dtype, np.number):
            a = np.asarray(rds[v].values, dtype="float64")
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if a.shape == (nprof, nlev):
                cols[v] = a.ravel()
                units[v] = rds[v].attrs.get("units")
    df = pd.DataFrame(cols)
    df.insert(0, "cycle", cycle)
    # One cycle file can hold several profiles (N_PROF): typically a primary
    # sampling scheme plus a shallow unpumped near-surface one. They are separate
    # casts, so tag them and never join them into a single line.
    df.insert(1, "n_prof", np.repeat(np.arange(nprof), nlev))
    if "VERTICAL_SAMPLING_SCHEME" in rds:
        _v = [(x.decode(errors="ignore") if isinstance(x, bytes) else str(x)).strip()
              for x in np.asarray(rds["VERTICAL_SAMPLING_SCHEME"].values).ravel()]
        _v = [(s.split(":")[0].strip() or "profile") for s in _v][:nprof]
        if len(_v) == nprof:
            df.insert(2, "sampling",
                      np.repeat(np.array(_v, dtype=object).reshape(nprof, 1),
                                nlev, 1).ravel())
    # per-profile fields broadcast down the levels
    for name, key in (("time", "JULD"), ("lat", "LATITUDE"), ("lon", "LONGITUDE")):
        if key in rds:
            v = np.asarray(rds[key].values).ravel()
            if len(v) >= nprof:
                df[name] = np.repeat(v[:nprof].reshape(nprof, 1), nlev, 1).ravel()
    if core_path:
        df = _merge_core_ts(df, core_path, nprof, nlev, units)
    return df, units


# ---- per-float views as tabs (sidebar search + results table stay global) ----
tab_over, tab_traj, tab_prof, tab_overlay, tab_ts, tab_tech, tab_raw = \
    st.tabs(["Overview", "Trajectory", "Profile & Trend", "Overlay", "T-S",
             "Tech", "Raw"])

# ===================== Overview: metadata + sensors + calibration =====================
with tab_over:
    st.subheader(f"Float {sel_wmo}")
    if frow is not None:
        launch = parse_argo_date(frow.get("launch_date"))
        last = parse_argo_date(frow.get("last_date"))
        llat, llon = frow.get("last_lat"), frow.get("last_lon")
        if pd.isna(last) and "JULD" in ds:
            jmax = pd.Series(ds["JULD"].values).max()
            last = pd.Timestamp(jmax) if pd.notna(jmax) else last
        if (pd.isna(llat) or pd.isna(llon)) and "LATITUDE" in ds:
            la, lo = ds["LATITUDE"].values, ds["LONGITUDE"].values
            ok = np.isfinite(la) & np.isfinite(lo)
            if ok.any():
                llat, llon = float(la[ok][-1]), float(lo[ok][-1])
        deployed = "-"
        if pd.notna(launch) and pd.notna(last) and last >= launch:
            dd = (last - launch).days
            deployed = f"{dd:,} days (~{dd / 365.25:.1f} yr)"
        last_profile = f"{last:%Y-%m-%d}" if pd.notna(last) else "-"
        if pd.notna(llat) and pd.notna(llon):
            lon_n = ((float(llon) + 180) % 360) - 180
            position = (f"{abs(float(llat)):.2f}°{'N' if llat >= 0 else 'S'}, "
                        f"{abs(lon_n):.2f}°{'E' if lon_n >= 0 else 'W'}")
            region = (f"{climate_band(float(llat))} · "
                      f"{ocean_basin(float(llat), float(llon))}")
        else:
            position, region = "-", "-"
        _kind = frow.get("data_kind")
        data_file = ({"core": "core (prof.nc, physical T/S)",
                      "bgc": "BGC (Sprof.nc, synthetic)"}.get(_kind)
                     if pd.notna(_kind) else "-")

        def _kv(pairs):
            def _v(x):
                s = "" if x is None else str(x).strip()
                return s if s and s.lower() != "nan" else "-"
            return "\n".join(f"**{k}:** {_v(v)}  " for k, v in pairs)

        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            st.markdown("**Identity**")
            st.markdown(_kv([
                ("WMO", frow.get("wmo")),
                ("Float serial no.", frow.get("float_serial_no")),
                ("Platform type", frow.get("platform_type")),
                ("Platform maker", frow.get("platform_maker")),
                ("WMO inst type", frow.get("wmo_inst_type")),
                ("Data file", data_file),
                ("DAC", frow.get("dac")),
            ]))
        with oc2:
            st.markdown("**Deployment**")
            st.markdown(_kv([
                ("Launch date", f"{launch:%Y-%m-%d}" if pd.notna(launch)
                 else frow.get("launch_date")),
                ("Days deployed", deployed),
                ("Last profile", last_profile),
                ("Last position", position),
                ("Region (approx.)", region),
            ]))
        with oc3:
            st.markdown("**Project**")
            st.markdown(_kv([
                ("Project", frow.get("project_name")),
                ("PI", frow.get("pi_name")),
            ]))

    st.markdown("---")
    sc1, sc2 = st.columns([3, 2])
    with sc1:
        st.markdown("**Sensors on this float**")
        st.dataframe(
            fsens[["sensor", "sensor_model", "sensor_maker", "sensor_serial_no"]]
            .reset_index(drop=True), width="stretch", height=260)
    with sc2:
        st.markdown("**Measurands on board**")
        if measurands:
            st.write(", ".join(measurands))
        else:
            st.write("_(not listed in index/meta; read from the data file)_")

    st.markdown("---")
    st.markdown("**Data quality (QC flags)**")
    qc_df, qc_pos = qc_summary(ds)
    if qc_df.empty:
        st.caption("No QC flags are reported in this float's file.")
    else:
        st.caption("Argo flags every measurement: 1 good, 2 probably good, 3 probably "
                   "bad, 4 bad, 5 changed, 8 estimated, 0 not assessed. **good %** "
                   "counts flags 1/2/5/8, **questionable %** is flag 3, **bad %** is "
                   "flag 4 (any remainder is flag 0, not assessed). **cycle grades** "
                   "grade each cycle by fraction of good data (A = 100%, B = 75 to "
                   "100%, down to F = 0%). These are the reported flags; delayed-mode "
                   "QC may refine them.")
        st.dataframe(qc_df, width="stretch", hide_index=True)
        if qc_pos:
            bits = [f"{lbl} {g}/{t} cycles good" for lbl, (g, t) in qc_pos.items()]
            st.caption("Per-cycle location and time QC: " + ", ".join(bits) + ".")

    with st.expander("🔬 Calibration coefficients (all measurands)", expanded=False):
        st.caption("How each parameter is calibrated: factory / pre-deployment sensor "
                   "coefficients (from meta.nc) that convert raw counts to physical "
                   "units, and delayed-mode (DMQC) adjustments (from the profile file). "
                   "Blank where a float doesn't report them.")
        calib_rows = list(scientific_calib_rows(ds))
        _dac = frow.get("dac") if frow is not None else None
        if _dac:
            meta_rel = f"dac/{_dac}/{sel_wmo}/{sel_wmo}_meta.nc"
            if fetch_from_gdac(meta_rel):
                try:
                    _mds = xr.open_dataset(os.path.join(ROOT, meta_rel),
                                           mask_and_scale=False, decode_times=False)
                    calib_rows = predeployment_calib_rows(_mds) + calib_rows
                    _mds.close()
                except Exception:
                    pass
        if calib_rows:
            cdf = pd.DataFrame(calib_rows)[
                ["measurand", "source", "coefficient", "equation", "comment", "date"]]
            st.dataframe(cdf, width="stretch", height=min(80 + 28 * len(cdf), 460))
        else:
            st.info("No calibration coefficients are reported in this float's files.")

# ===================== Trajectory: real basemap (pydeck) =====================
with tab_traj:
    lat = ds["LATITUDE"].values if "LATITUDE" in ds else None
    lon = ds["LONGITUDE"].values if "LONGITUDE" in ds else None
    if lat is None or lon is None or not np.isfinite(lat).any():
        st.info("No position data for this float.")
    else:
        traj = pd.DataFrame({"lat": lat, "lon": lon,
                             "cycle": (ds["CYCLE_NUMBER"].values
                                       if "CYCLE_NUMBER" in ds
                                       else np.arange(len(lat)))})
        traj = traj.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        st.markdown(f"**{float_tag} · trajectory**")
        path_layer = pdk.Layer(
            "PathLayer",
            data=pd.DataFrame({"path": [traj[["lon", "lat"]].values.tolist()]}),
            get_path="path", get_color=[10, 110, 189], get_width=4,
            width_min_pixels=2)
        cycle_layer = pdk.Layer(
            "ScatterplotLayer", data=traj, get_position="[lon, lat]",
            get_fill_color=[10, 110, 189, 140], get_radius=500,
            radius_min_pixels=2, pickable=True)
        ends = pd.DataFrame({
            "lon": [float(traj["lon"].iloc[0]), float(traj["lon"].iloc[-1])],
            "lat": [float(traj["lat"].iloc[0]), float(traj["lat"].iloc[-1])],
            "color": [[0, 170, 70], [210, 40, 20]]})
        ends_layer = pdk.Layer(
            "ScatterplotLayer", data=ends, get_position="[lon, lat]",
            get_fill_color="color", get_radius=1600, radius_min_pixels=6,
            pickable=False)
        _span = max(float(traj.lat.max() - traj.lat.min()),
                    float(traj.lon.max() - traj.lon.min()), 0.5)
        _zoom = next((z for thr, z in [(1, 7), (3, 6), (8, 5), (20, 4), (50, 3)]
                      if _span < thr), 2)
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=float(traj.lat.mean()), longitude=float(traj.lon.mean()),
                zoom=_zoom),
            layers=[path_layer, cycle_layer, ends_layer],
            tooltip={"text": "cycle {cycle}"}))
        st.caption("🟢 launch · 🔴 latest · blue line = drift track. "
                   "Basemap © CARTO / OpenStreetMap.")

# ===================== Profile: parameter + data-view controls live here =====================
with tab_prof:
    plottable = [p for p in measurands if p in ds or f"{p}_ADJUSTED" in ds]
    derived_here = [d for d in DERIVED_2D if d in ds and d not in plottable]
    if derived_here:
        plottable = plottable + derived_here     # TEOS-10 fields at the end
    param_opts = plottable or measurands

    def _default_param_index(opts):
        for pref in ("TEMP", "PSAL", "CT", "PT", "DOXY"):
            if pref in opts:
                return opts.index(pref)
        for i, o in enumerate(opts):
            if o not in ("PRES", "MTIME"):
                return i
        return 0

    pc1, pc2 = st.columns([2, 2])
    # deep-linkable parameter: ?var= seeds it once, and the keyed widget keeps the
    # choice across float switches (sanitized against the new float's options, so a
    # BGC pick quietly falls back to the default on a core float)
    _dlv = st.session_state.pop("_dl_var", None)
    if _dlv in param_opts:
        st.session_state["param_pick"] = _dlv
    if st.session_state.get("param_pick") not in param_opts:
        st.session_state["param_pick"] = (param_opts[_default_param_index(param_opts)]
                                          if param_opts else None)
    param = pc1.selectbox("Parameter to plot", param_opts, key="param_pick",
                          help="Includes TEOS-10 derived fields "
                               "(SIGMA0, CT, PT, SA, AOU) when computable.")
    if param and st.query_params.get("var") != str(param):
        st.query_params["var"] = str(param)
    view = pc2.radio(
        "Data view", ["Real-time (R)", "QC-filtered", "Adjusted (A/D)"],
        index=2, horizontal=True,
        help="**Real-time (R)**: the reported value, no adjustment (shown whatever the "
             "parameter's data mode). "
             "**QC-filtered**: the reported value, keeping only good QC flags {1,2,5,8}. "
             "**Adjusted (A/D)**: delayed-mode/adjusted, science-ready values (QC-filtered). "
             "The parameter's actual data mode (R/A/D) is shown just below.")
    adjusted = view.startswith("Adjusted")
    apply_qc = view.startswith(("QC", "Adjusted"))
    _woa_ok = param in WOA_FILES and _woa_path(param) is not None
    woa_on = st.toggle(
        "Overlay WOA23 climatology (annual mean at this float's position)",
        key="woa_overlay", disabled=not _woa_ok,
        help="World Ocean Atlas 2023 (NOAA NCEI): the long-term mean profile "
             "for the 1-degree cell at this float's median position, as a "
             "dashed reference line. A climatology is a decades-long mean, so "
             "an offset is not by itself a sensor problem."
             + ("" if _woa_ok else " Available for TEMP, PSAL, DOXY, AOU and "
                                   "NITRATE.")) and _woa_ok

    if param:
        df, pcol, vcol = param_long_frame(ds, param, adjusted, apply_qc)
    else:
        df, pcol, vcol = pd.DataFrame(), None, None

    if not param:
        st.info("No plottable parameters for this float.")
    else:
        st.caption(f"Data mode for {param}: {data_mode_for(ds, param)}")
        if df.empty:
            st.warning("No finite points after selection. Try a different **Data view**, "
                       "since this parameter may only exist as real-time values (no "
                       "QC-passing or adjusted values) for this float.")
        else:
            vlabel, plabel = _axis_label(vcol), _axis_label(pcol)
            if "time" in df.columns:
                _t = pd.to_datetime(df["time"])
                cvals = _t.astype("int64").astype("float64")
                cvals[_t.isna().to_numpy()] = np.nan
                _lo, _hi = np.nanmin(cvals), np.nanmax(cvals)
                _tv = list(np.linspace(_lo, _hi, 5)) if np.isfinite(_lo) else []
                cbar = dict(title="date", tickvals=_tv,
                            ticktext=[pd.Timestamp(int(t)).strftime("%Y-%m-%d")
                                      for t in _tv])
                cdata = _t.dt.strftime("%Y-%m-%d").to_numpy()
                hover = (f"{vcol}=%{{x:.4g}}<br>{pcol}=%{{y:.1f}} dbar"
                         "<br>%{customdata}<extra></extra>")
            else:
                cvals = df["cycle"].to_numpy()
                cdata = cvals
                cbar = dict(title="cycle")
                hover = (f"{vcol}=%{{x:.4g}}<br>{pcol}=%{{y:.1f}} dbar"
                         "<br>cycle %{customdata}<extra></extra>")
            _woa_note, _wp = None, None
            if woa_on:
                _fla = (np.asarray(ds["LATITUDE"].values).ravel()
                        if "LATITUDE" in ds else np.array([]))
                _flo = (np.asarray(ds["LONGITUDE"].values).ravel()
                        if "LONGITUDE" in ds else np.array([]))
                _fin = np.isfinite(_fla) & np.isfinite(_flo)
                if _fin.any():
                    _wla, _wlo = woa_cell(float(np.median(_fla[_fin])),
                                          float(np.median(_flo[_fin])))
                    _wp = woa_profile(param, _wla, _wlo)
                    if _wp is not None and len(_wp):
                        _woa_note = (
                            f"Dashed gray: WOA23 annual climatology (1° cell at "
                            f"{_wla:.1f}, {_wlo:.1f}), NOAA NCEI. A decades-long "
                            "mean, so offsets include real variability and "
                            "trends, not just sensor error.")
                    else:
                        _wp = None
                        _woa_note = ("No WOA23 climatology at this float's grid "
                                     "cell.")
            _main = go.Scattergl(
                x=df["value"], y=df["pres"], mode="markers",
                marker=dict(size=4, color=cvals, colorscale="Viridis",
                            colorbar=cbar),
                customdata=cdata, hovertemplate=hover)
            if _wp is not None:
                # profile and residual panels share one reversed pressure axis;
                # it stays ONE chart element either way, so the tab's structure
                # is stable and st.tabs keeps its selection
                fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                    column_widths=[0.62, 0.38],
                                    horizontal_spacing=0.05)
                fig.add_trace(_main, row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=_wp["val"], y=_wp["pres"], mode="lines", name="WOA23",
                    showlegend=False,
                    line=dict(color="#808A95", width=2.5, dash="dash"),
                    hovertemplate="%{x:.4g} at %{y:.0f} dbar"
                                  "<extra>WOA23 climatology</extra>"),
                    row=1, col=1)
                _pn = df["pres"].to_numpy("float64")
                _wf = np.interp(_pn, _wp["pres"].to_numpy("float64"),
                                _wp["val"].to_numpy("float64"),
                                left=np.nan, right=np.nan)
                _res = df["value"].to_numpy("float64") - _wf
                _okr = np.isfinite(_res)
                fig.add_trace(go.Scattergl(
                    x=_res[_okr], y=_pn[_okr], mode="markers",
                    marker=dict(size=4, color=np.asarray(cvals)[_okr],
                                colorscale="Viridis", showscale=False),
                    customdata=np.asarray(cdata)[_okr],
                    hovertemplate="Δ %{x:.4g}<br>%{y:.1f} dbar<br>%{customdata}"
                                  "<extra>float minus WOA23</extra>"),
                    row=1, col=2)
                if _okr.any():
                    _rb = (pd.DataFrame({"p": (_pn[_okr] // 50) * 50 + 25,
                                         "r": _res[_okr]})
                           .groupby("p")["r"].median())
                    fig.add_trace(go.Scatter(
                        x=_rb.values, y=_rb.index, mode="lines",
                        showlegend=False,
                        line=dict(color="#808A95", width=2),
                        hovertemplate="median Δ %{x:.4g} at %{y:.0f} dbar"
                                      "<extra></extra>"), row=1, col=2)
                    _med = float(np.nanmedian(_res[_okr]))
                    _q1, _q3 = np.nanpercentile(_res[_okr], [25, 75])
                    _woa_note += (f" Residuals (right panel, float minus WOA23): "
                                  f"median {_med:+.3g}, middle half {_q1:+.3g} "
                                  f"to {_q3:+.3g}, {int(_okr.sum()):,} samples "
                                  "within the WOA depth range.")
                fig.add_vline(x=0, row=1, col=2, line_color="#808A95",
                              line_dash="dot", line_width=1)
                fig.update_xaxes(title_text=vlabel, row=1, col=1)
                fig.update_xaxes(title_text="float minus WOA23", row=1, col=2)
                fig.update_yaxes(autorange="reversed")   # depth downward, both
                fig.update_yaxes(title_text=plabel, row=1, col=1)
            else:
                fig = go.Figure(_main)
                fig.update_xaxes(title=vlabel)
                fig.update_yaxes(autorange="reversed", title=plabel)
            fig.update_layout(height=560)
            _titled(fig, f"{float_tag} · {param} profile")
            st.plotly_chart(fig, width="stretch")
            # a permanent slot, filled or left blank: st.tabs resets to the first
            # tab whenever a tab's element COUNT changes between reruns, so the
            # caption must occupy its node whether or not the overlay is on
            _woa_slot = st.empty()
            if _woa_note:
                _woa_slot.caption(_woa_note)

            d1, d2 = st.columns(2)
            d1.download_button("Download this parameter (CSV)",
                               df.to_csv(index=False).encode(),
                               file_name=f"{sel_wmo}_{param}.csv", mime="text/csv")
            _orig_path = os.path.join(ROOT, str(sprof_rel))
            with open(_orig_path, "rb") as _f:
                _orig_bytes = _f.read()
            d2.download_button("Download full float (NetCDF)", _orig_bytes,
                               file_name=os.path.basename(_orig_path),
                               mime="application/x-netcdf")

# ======== Profile & Trend (cont.): depth-time section (measurand-driven) ========
with tab_prof:          # re-enter to append the section, after the profile, before the trend
    if param and not df.empty:
        st.markdown("---")
        st.subheader(f"{param} depth-time section")
        sec = section_grid(ds, param, adjusted, apply_qc)
        if sec is None:
            st.info("Not enough profiles/time info to build a section.")
        else:
            pgrid, stimes, Z, svcol = sec
            fig_sec = go.Figure(go.Heatmap(
                x=stimes, y=pgrid, z=Z, colorscale="Viridis",
                colorbar=dict(title=_axis_label(svcol)),
                hovertemplate="time=%{x|%Y-%m-%d}<br>pres=%{y:.0f} dbar"
                              "<br>value=%{z:.3g}<extra></extra>"))
            if "MLD" in ds and "JULD" in ds:
                mld_df = pd.DataFrame({"time": ds["JULD"].values,
                                       "mld": ds["MLD"].values}).dropna()
                mld_df = mld_df.sort_values("time")
                if not mld_df.empty:
                    fig_sec.add_trace(go.Scatter(
                        x=mld_df["time"], y=mld_df["mld"], mode="lines",
                        line=dict(color="white", width=1.5, dash="dot"),
                        name="MLD"))
            fig_sec.update_yaxes(autorange="reversed", title="PRES [decibar]")
            fig_sec.update_layout(height=440, xaxis_title="time",
                                  margin=dict(l=0, r=0, t=10, b=0),
                                  legend=dict(orientation="h", y=1.02))
            _titled(fig_sec, f"{float_tag} · {param} depth-time section")
            st.plotly_chart(fig_sec, width="stretch")
            st.caption("Each profile linearly interpolated onto a common pressure "
                       "grid. White dotted line = mixed-layer depth (where computed).")

# ===================== T-S diagram (parameter-independent) =====================
with tab_ts:
    st.subheader("Temperature-Salinity diagram")
    tsf = ts_diagram_frame(ds)
    if tsf is None or tsf[0].empty:
        st.info("Salinity/temperature not available for a T-S diagram.")
    else:
        tsdf, xn, yn = tsf
        if len(tsdf) > 150000:
            tsdf = tsdf.sample(150000, random_state=0)
        fig_ts_d = go.Figure()
        if gsw is not None and xn == "SA":
            sa_lin = np.linspace(tsdf["sal"].min(), tsdf["sal"].max(), 60)
            ct_lin = np.linspace(tsdf["temp"].min(), tsdf["temp"].max(), 60)
            SAg, CTg = np.meshgrid(sa_lin, ct_lin)
            with np.errstate(invalid="ignore"):
                dens = gsw.sigma0(SAg, CTg)
            fig_ts_d.add_trace(go.Contour(
                x=sa_lin, y=ct_lin, z=dens, showscale=False,
                contours=dict(coloring="lines", showlabels=True),
                line=dict(width=1), colorscale="Greys",
                hoverinfo="skip", name="sigma0"))
        fig_ts_d.add_trace(go.Scattergl(
            x=tsdf["sal"], y=tsdf["temp"], mode="markers",
            marker=dict(size=3, color=tsdf["pres"], colorscale="Viridis_r",
                        reversescale=False, showscale=True,
                        colorbar=dict(title="PRES [dbar]")),
            hovertemplate=f"{xn}=%{{x:.2f}}<br>{yn}=%{{y:.2f}}"
                          "<br>pres=%{marker.color:.0f} dbar<extra></extra>",
            name="samples"))
        xu = "g/kg" if xn == "SA" else "psu"
        yu = "degree_Celsius"
        fig_ts_d.update_layout(
            height=520, xaxis_title=f"{xn} [{xu}]", yaxis_title=f"{yn} [{yu}]",
            margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        _titled(fig_ts_d, f"{float_tag} · T-S diagram")
        st.plotly_chart(fig_ts_d, width="stretch")
        st.caption("Grey contours = potential density σ₀ (kg/m³); points colored "
                   "by pressure. Water masses cluster along isopycnals."
                   if xn == "SA" else
                   "Practical salinity vs in-situ temperature (install gsw for "
                   "σ₀ contours & absolute salinity).")

# ============ Profile & Trend (cont.): per-pressure time series + Sen/Mann-Kendall ============
with tab_prof:          # re-enter to append the trend to the Profile & Trend tab
    if param and not df.empty and "time" not in df.columns:
        st.markdown("---")
        st.subheader(f"{param} time series at a pressure level")
        st.info("This float has no JULD/time coordinate; can't build a time series.")
    elif param and not df.empty:
        st.markdown("---")
        st.subheader(f"{param} time series at a pressure level")
        pmin, pmax = float(df["pres"].min()), float(df["pres"].max())
        default_p = max(pmin, round(pmin, 1))
        tcol1, tcol2 = st.columns(2)
        target_p = tcol1.number_input(
            "Target pressure (dbar)", min_value=pmin, max_value=pmax,
            value=default_p, step=1.0,
            help="Defaults to the shallowest available level (closest to the "
                 "surface). For each profile the nearest available level is used.")
        max_gap = tcol2.number_input(
            "Max distance from target (dbar)", min_value=0.0, value=3.0, step=1.0,
            help="Drop profiles whose nearest sample is farther than this "
                 "from the target pressure. Tight by default (3 dbar) so a point "
                 "reflects that pressure; widen it if the record is sparsely "
                 "sampled and profiles drop out.")
        near = (df.assign(dp=(df["pres"] - target_p).abs())
                  .sort_values("dp").groupby("cycle", as_index=False).first())
        near = near[near["dp"] <= max_gap].sort_values("time")
        if near.empty:
            st.warning(f"No profiles have a sample within {max_gap:g} dbar of "
                       f"{target_p:g} dbar. Widen the max distance.")
        else:
            near = near.copy()
            deseason = st.toggle(
                "Deseasonalize (remove monthly climatology)", value=False,
                key="deseason",
                help="Subtract each calendar month's mean so a long-term "
                     "trend isn't masked by the seasonal cycle.")
            lat_med = near["lat"].median() if "lat" in near else 0.0
            cal_month = pd.DatetimeIndex(near["time"]).month
            month = cal_month.to_numpy()
            if pd.notna(lat_med) and lat_med < 0:      # S. hemisphere: +6 months
                month = ((month + 5) % 12) + 1
                hemi = "S"
            else:
                hemi = "N"
            season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                          3: "Spring", 4: "Spring", 5: "Spring",
                          6: "Summer", 7: "Summer", 8: "Summer",
                          9: "Fall", 10: "Fall", 11: "Fall"}
            near["season"] = pd.Series(month, index=near.index).map(season_map)
            near["year"] = pd.DatetimeIndex(near["time"]).year
            near["cal_month"] = cal_month
            near["anom"] = near["value"] - near.groupby("cal_month")["value"] \
                                               .transform("mean")
            ycol = "anom" if deseason else "value"
            season_colors = {"Winter": "#4C72B0", "Spring": "#55A868",
                             "Summer": "#C44E52", "Fall": "#DD8452"}
            units = ds[vcol].attrs.get("units") if vcol in ds else None
            base_lab = f"{param} anomaly" if deseason else param
            ylab = f"{base_lab} [{units}]" if units else base_lab
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=near["time"], y=near[ycol], mode="lines",
                line=dict(color="rgba(150,150,150,0.4)", width=1),
                showlegend=False, hoverinfo="skip"))
            for s in ["Winter", "Spring", "Summer", "Fall"]:
                sub = near[near["season"] == s]
                if sub.empty:
                    continue
                fig_ts.add_trace(go.Scatter(
                    x=sub["time"], y=sub[ycol], mode="markers",
                    marker=dict(size=8, color=season_colors[s]), name=s,
                    customdata=sub["pres"],
                    hovertemplate=(f"{base_lab}=%{{y:.3g}}<br>"
                                   "time=%{x|%Y-%m-%d}<br>"
                                   "pres=%{customdata:.1f} dbar"
                                   f"<br>season={s}<extra></extra>")))
            t_years = (pd.DatetimeIndex(near["time"]).asi8.astype("float64")
                       / (365.25 * 24 * 3600 * 1e9))
            yv = near[ycol].to_numpy(dtype="float64")
            res = mann_kendall_sen(t_years, yv)
            if res and np.isfinite(res["sen"]):
                sen = res["sen"]
                inter = np.median(yv) - sen * np.median(t_years)
                fig_ts.add_trace(go.Scatter(
                    x=near["time"], y=sen * t_years + inter, mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name=f"Sen slope {sen:+.3g}/yr"))
            fig_ts.update_layout(height=460, xaxis_title="time",
                                 yaxis_title=ylab, legend_title="season",
                                 margin=dict(l=0, r=0, t=10, b=0))
            _titled(fig_ts, f"{float_tag} · {param} at {target_p:g} dbar")
            st.plotly_chart(fig_ts, width="stretch")

            if res and np.isfinite(res["sen"]):
                p = res["p"]
                sig = ("not significant (p≥0.05)"
                       if not np.isfinite(p) or p >= 0.05 else
                       "significant (p<0.05)" if p >= 0.01 else
                       "highly significant (p<0.01)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Sen's slope", f"{res['sen']:+.3g} {units or ''}/yr",
                          help="How big the trend is, in units per year. It is the "
                               "**median** of the slopes between every pair of points, "
                               "so a few bad or spiky samples barely move it (unlike "
                               "ordinary linear regression). This is the *how much* "
                               "number. Check the Mann-Kendall p to see if it is real.")
                m2.metric("Mann-Kendall p",
                          "n/a" if not np.isfinite(p) else f"{p:.3g}",
                          help="Whether the series really trends one way or is just "
                               "noise. This non-parametric test compares every pair of "
                               "points and counts how often later values exceed earlier "
                               "ones, assuming nothing about the data's distribution and "
                               "resisting outliers (which suits messy ocean data). "
                               "**p < 0.05** means the trend is unlikely to be chance; "
                               "**p >= 0.05** means it could just be noise. This is the "
                               "*is it real* number.")
                m3.metric("Trend", sig,
                          help="The plain verdict from the p-value: significant "
                               "(p<0.05), highly significant (p<0.01), or not "
                               "significant (p>=0.05). If it says not significant, the "
                               "Sen's slope could easily be noise, so do not over-read "
                               "it. Note: in the Real-time or Raw view, uncorrected "
                               "sensor drift can itself look like a trend; the Adjusted "
                               "view guards against that.")
                st.caption(
                    f"n={res['n']} profiles · Sen's slope = median pairwise rate "
                    "(outlier-robust); Mann-Kendall tests for a monotonic trend. "
                    + ("Series shown is the deseasonalized anomaly."
                       if deseason else
                       "Turn on deseasonalize to remove the annual cycle first."))
            st.caption(
                f"{len(near)} profiles · nearest-level pressures "
                f"{near['pres'].min():.1f}-{near['pres'].max():.1f} dbar "
                f"(target {target_p:g}) · {hemi}-hemisphere meteorological seasons.")
            st.download_button(
                "Download time series (CSV)",
                near[["time", "year", "season", "pres", "value", "anom", "cycle"]]
                    .to_csv(index=False).encode(),
                file_name=f"{sel_wmo}_{param}_ts_{target_p:.0f}dbar.csv",
                mime="text/csv", key="ts_download")

# ===================== Raw: per-cycle diagnostic profile =====================
with tab_raw:
    st.subheader("Raw profiles (per-cycle diagnostics)")
    st.caption("Pulls per-cycle raw NetCDFs straight from the GDAC, with every parameter "
               "they report, including intermediate sensor signals (raw fluorescence, "
               "backscatter, optode phase, and so on) that are not carried into the "
               "synthetic product. Unadjusted and with no QC filtering. Each cycle is a "
               "separate download, so pulling many takes a while.")
    _is_bgc = str(frow.get("data_kind")) == "bgc" if frow is not None else False
    _dac = frow.get("dac") if frow is not None else None
    _cyc_vals = (sorted({int(c) for c in np.asarray(ds["CYCLE_NUMBER"].values).ravel()
                         if np.isfinite(c)}) if "CYCLE_NUMBER" in ds else [])
    if not _dac or not _cyc_vals:
        st.info("No per-cycle files are available for this float.")
    else:
        # per-cycle date/position from the synthetic file, so the picker has context
        _cn = np.asarray(ds["CYCLE_NUMBER"].values).ravel()
        _meta = {}
        for _k, _v in (("time", "JULD"), ("lat", "LATITUDE"), ("lon", "LONGITUDE")):
            _meta[_k] = (np.asarray(ds[_v].values).ravel() if _v in ds
                         else np.full(_cn.shape, np.nan))
        _by_cyc = {}
        for _i, _c in enumerate(_cn):
            if np.isfinite(_c) and int(_c) not in _by_cyc:
                _by_cyc[int(_c)] = (_meta["time"][_i], _meta["lat"][_i], _meta["lon"][_i])

        # Everything below is per-float and lives in session_state, so it MUST be
        # dropped when the float changes: otherwise the previous float's pulled
        # profiles stay on screen and read as if they belonged to this one.
        if st.session_state.get("_raw_wmo") != sel_wmo:
            st.session_state["_raw_wmo"] = sel_wmo
            for _k in ("raw_data", "raw_units", "raw_report", "raw_csv",
                       "raw_csv_sig", "raw_pick"):
                st.session_state.pop(_k, None)
            # never reuse a version, so the cycle table can't inherit stale ticks
            st.session_state["raw_pick_ver"] = (
                st.session_state.get("raw_pick_ver", 0) + 1)

        # ---- selection: checkboxes we can also drive from the quick-select buttons ----
        _sel = st.session_state.setdefault("raw_pick", {_cyc_vals[-1]})   # default: latest
        _ver = st.session_state.setdefault("raw_pick_ver", 0)

        def _set_pick(cycles):
            st.session_state["raw_pick"] = set(cycles)
            st.session_state["raw_pick_ver"] = st.session_state["raw_pick_ver"] + 1

        # one tight row: a horizontal container packs these at their natural width,
        # instead of columns stretching each control across the page
        with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
            if st.button("Latest", help="Select only the most recent cycle."):
                _set_pick([_cyc_vals[-1]]); st.rerun()
            if st.button("All", help=f"Select all {len(_cyc_vals)} cycles."):
                _set_pick(_cyc_vals); st.rerun()
            if st.button("None", help="Clear the selection."):
                _set_pick([]); st.rerun()
            # a young float can have fewer cycles than the default stride, and a
            # value above max_value raises before the widget ever draws
            _nth_max = max(1, len(_cyc_vals))
            _nth = st.number_input("Every Nth cycle", min_value=1,
                                   max_value=_nth_max, value=min(10, _nth_max), step=1,
                                   width=150,
                                   help="Sample the whole record without pulling every "
                                        "cycle. Press Apply to tick those boxes.")
            if st.button(f"Apply every {int(_nth)}th"):
                _set_pick(_cyc_vals[::int(_nth)]); st.rerun()

        _tbl = pd.DataFrame({
            "pull": [c in _sel for c in _cyc_vals],
            "cycle": _cyc_vals,
            "date": [pd.to_datetime(_by_cyc.get(c, (np.nan,) * 3)[0], errors="coerce")
                     for c in _cyc_vals],
            # to_numeric so a missing fix is NaN (blank cell) and not a grey "None"
            "lat": pd.to_numeric([_by_cyc.get(c, (np.nan,) * 3)[1] for c in _cyc_vals],
                                 errors="coerce"),
            "lon": pd.to_numeric([_by_cyc.get(c, (np.nan,) * 3)[2] for c in _cyc_vals],
                                 errors="coerce"),
        })
        _ed = st.data_editor(
            _tbl, hide_index=True, height=260, key=f"raw_tbl_{sel_wmo}_{_ver}",
            disabled=["cycle", "date", "lat", "lon"],
            column_config={
                "pull": st.column_config.CheckboxColumn(
                    "pull", help="Tick the cycles to download."),
                "cycle": st.column_config.NumberColumn("cycle", format="%d"),
                "date": st.column_config.DatetimeColumn("date", format="YYYY-MM-DD"),
                "lat": st.column_config.NumberColumn("lat", format="%.2f"),
                "lon": st.column_config.NumberColumn("lon", format="%.2f")})
        chosen = [int(c) for c in _ed.loc[_ed["pull"], "cycle"].tolist()]

        if not chosen:
            st.info("Tick at least one cycle above, then pull.")
        else:
            if len(chosen) > 25:
                st.warning(f"{len(chosen)} cycles is {len(chosen)} separate GDAC "
                           f"downloads, roughly {max(1, round(len(chosen) / 8 * 1.5))}s "
                           "or more, and it re-runs if you touch another control while "
                           "it works. Consider 'Every Nth' to sample the record instead.")
            if st.button(f"Pull {len(chosen)} profile(s) from the GDAC", type="primary"):
                _kind = "B-file" if _is_bgc else "profile file"
                # A BGC pull is two downloads per cycle: the B-file (biogeochemistry)
                # and the core file (the CTD's TEMP/PSAL), merged on a shared PRES axis.
                _phases = 2 if _is_bgc else 1
                with st.status(f"Downloading {len(chosen)} {_kind}(s) from the GDAC",
                               expanded=True) as _stat:
                    _bar = st.progress(0.0, text="Starting…")

                    def _tick(phase, what):
                        def _t(done, total, cyc):
                            _bar.progress((phase + done / total) / _phases,
                                          text=f"{done}/{total} {what} (cycle {cyc})")
                        return _t

                    _res = fetch_raw_many(_dac, sel_wmo, chosen, _is_bgc,
                                          _tick(0, f"{_kind}s"))
                    _ok = {c: r for c, (r, s) in _res.items() if r}
                    _missing = [c for c, (r, s) in _res.items() if not r and s == "missing"]
                    _unreach = [c for c, (r, s) in _res.items()
                                if not r and s == "unreachable"]
                    # BGC B-files omit the CTD's TEMP/PSAL; pull the matching core
                    # files so T and S ride along in the plots and the CSV export.
                    _core = {}
                    if _is_bgc and _ok:
                        _cres = fetch_raw_many(_dac, sel_wmo, sorted(_ok), False,
                                               _tick(1, "core T/S file"))
                        _core = {c: r for c, (r, s) in _cres.items() if r}
                    _bar.progress(1.0, text=f"{len(_ok)}/{len(chosen)} downloaded")
                    _frames, _units = [], {}
                    for _c in sorted(_ok):
                        _cp = (os.path.join(ROOT, _core[_c]) if _c in _core else None)
                        _f, _u = raw_cycle_frame(os.path.join(ROOT, _ok[_c]), _c,
                                                 core_path=_cp)
                        if _f is not None:
                            _frames.append(_f)
                            _units.update({k: v for k, v in _u.items() if v})
                    _ts_note = (f" · T/S merged for {len(_core)}/{len(_ok)}"
                                if _is_bgc else "")
                    _stat.update(
                        label=(f"{len(_ok)} downloaded · {len(_missing)} missing · "
                               f"{len(_unreach)} unreachable{_ts_note}"),
                        state="complete", expanded=False)
                st.session_state["raw_data"] = (
                    pd.concat(_frames, ignore_index=True) if _frames else None)
                st.session_state["raw_units"] = _units
                st.session_state["raw_report"] = (len(_ok), _missing, _unreach)

            raw_df = st.session_state.get("raw_data")
            if raw_df is None:
                st.caption("Nothing pulled yet.")
            else:
                _n_ok, _missing, _unreach = st.session_state.get(
                    "raw_report", (0, [], []))
                if _unreach:
                    st.error(f"Couldn't reach the GDAC for {len(_unreach)} cycle(s) "
                             f"({', '.join(map(str, _unreach[:8]))}"
                             f"{'…' if len(_unreach) > 8 else ''}). That's a connection "
                             "problem, not a missing file; pull again to retry.")
                if _missing:
                    st.warning(f"No raw file exists at the GDAC for {len(_missing)} "
                               f"cycle(s) ({', '.join(map(str, _missing[:8]))}"
                               f"{'…' if len(_missing) > 8 else ''}).")
                _units = st.session_state.get("raw_units", {})
                _pulled = sorted(raw_df["cycle"].unique().tolist())
                _meta_names = ("cycle", "n_prof", "sampling", "pres", "time",
                               "lat", "lon")
                raw_params = [c for c in raw_df.columns
                              if c not in _meta_names and not c.endswith("_QC")]
                if not raw_params:
                    st.info("These raw files report no plottable measurands.")
                else:
                    # prefer a real measurand: the first raw param is often MTIME,
                    # a time offset whose fill values swamp the axis
                    _plain = [p for p in raw_params if not p.endswith("_ADJUSTED")]
                    _rdef = (next(([p] for p in ("TEMP", "PSAL", "DOXY", "CHLA",
                                                 "BBP700", "PRES")
                                   if p in _plain),
                                  [p for p in _plain if p != "MTIME"][:1] or _plain[:1]))
                    mc1, mc2 = st.columns([2, 1])
                    rmeas = mc1.multiselect(
                        "Measurands (raw, up to 4)", raw_params,
                        default=_rdef or raw_params[:1], max_selections=4,
                        help="Every parameter in the raw files, including intermediate "
                             "sensor signals. Each gets its own color-matched x-axis "
                             "over the shared pressure axis.")
                    # keyed, so the choice sticks: a keyless radio's identity comes
                    # from its params, so a changing index would silently reset it
                    # when the measurand count changed.
                    mode = mc2.radio(
                        "Show", ["Individual casts", "Mean ± 1σ"], key="raw_mode",
                        help="Individual casts draws every profile separately (colored "
                             "by date for a single measurand), best for spotting drift "
                             "or an odd cast. Mean ± 1σ averages across the pulled "
                             "profiles, best for comparing measurands.")
                    st.caption(f"{len(_pulled)} profile(s) pulled · cycles "
                               f"{_pulled[0]} to {_pulled[-1]} · {len(raw_df):,} levels.")
                    if not rmeas:
                        st.info("Pick a measurand to plot.")
                    elif mode == "Individual casts":
                        # One line per (cycle, N_PROF). A cycle's primary and
                        # near-surface schemes are separate casts; joining them
                        # would draw a line from the deepest point to the surface.
                        _rank = {c: i for i, c in enumerate(_pulled)}
                        _grp = (["cycle", "n_prof"] if "n_prof" in raw_df.columns
                                else ["cycle"])
                        series = []
                        for m in rmeas:
                            casts = []
                            for _k, _s in raw_df.groupby(_grp, sort=True):
                                _c = _k[0] if isinstance(_k, tuple) else _k
                                _mk = (np.isfinite(_s["pres"].to_numpy())
                                       & np.isfinite(_s[m].to_numpy()))
                                if not _mk.any():
                                    continue
                                _lab = f"cycle {_c}"
                                if "sampling" in _s.columns:
                                    _sc = str(_s["sampling"].iloc[0])
                                    if _sc and _sc != "nan":
                                        _lab += f" · {_sc}"
                                _d = (pd.to_datetime(_s["time"].iloc[0], errors="coerce")
                                      if "time" in _s.columns else pd.NaT)
                                if pd.notna(_d):
                                    _lab += f" · {_d:%Y-%m-%d}"
                                casts.append((
                                    _lab, _s.loc[_mk, "pres"].to_numpy(),
                                    _s.loc[_mk, m].to_numpy(),
                                    _rank.get(_c, 0) / max(1, len(_pulled) - 1)))
                            if casts:
                                series.append((m, casts))
                        if not series:
                            st.warning("No finite samples for these measurands.")
                        else:
                            _ncast = max(len(c) for _, c in series)
                            _nsch = (raw_df["sampling"].nunique()
                                     if "sampling" in raw_df.columns else 1)
                            st.plotly_chart(_casts_fig(series, _units), width="stretch")
                            st.caption(
                                f"Every cast drawn separately, no averaging · "
                                f"{_ncast} cast(s) per measurand"
                                + (f" ({_nsch} sampling schemes, drawn as separate "
                                   "lines)" if _nsch > 1 else "")
                                + " · "
                                + ("dark to light = oldest to newest cycle."
                                   if len(series) == 1
                                   else "colored by measurand, each on its own x-axis."))
                    else:
                        profiles = []
                        for m in rmeas:
                            _sub = (raw_df[["pres", m]].rename(columns={m: "value"}))
                            _mp = _stat_profile(_sub)
                            if _mp is None:
                                continue
                            profiles.append((m, _mp[0], _mp[1], _mp[2], _units.get(m)))
                        if not profiles:
                            st.warning("No finite samples for these measurands.")
                        else:
                            st.plotly_chart(
                                _overlay_fig(profiles, len(_pulled) > 1),
                                width="stretch")
                            st.caption(
                                f"Mean over {len(_pulled)} pulled profile(s)"
                                + (", shaded to ±1 standard deviation"
                                   if len(_pulled) > 1 else "")
                                + "; each measurand keeps its own color and x-axis.")
                    # everything pulled, not just what's plotted: every parameter in
                    # the raw files plus their QC flags, for offline work.
                    # Built once per pull and held in session_state: download_button
                    # takes its data eagerly, so encoding on every rerun would rebuild
                    # a multi-hundred-MB string each time a widget moved.
                    _meta_cols = [c for c in ("cycle", "n_prof", "sampling", "time",
                                             "lat", "lon", "pres")
                                  if c in raw_df.columns]
                    _data_cols = [c for c in raw_df.columns if c not in _meta_cols]
                    _sig = (sel_wmo, tuple(_pulled), len(raw_df),
                            tuple(_meta_cols + _data_cols))
                    if st.session_state.get("raw_csv_sig") != _sig:
                        _o = raw_df[_meta_cols + _data_cols].copy()
                        _o.insert(0, "wmo", sel_wmo)
                        st.session_state["raw_csv_sig"] = _sig
                        st.session_state["raw_csv"] = (
                            _o.to_csv(index=False).encode(), len(_o), len(_o.columns))
                    _csv_bytes, _nrows, _ncols = st.session_state["raw_csv"]
                    st.download_button(
                        f"Download all {len(_pulled)} raw profile(s) as one CSV "
                        f"({_nrows:,} rows × {_ncols} cols · "
                        f"{_human_bytes(len(_csv_bytes))})",
                        _csv_bytes,
                        file_name=f"{sel_wmo}_raw_{_pulled[0]}-{_pulled[-1]}.csv",
                        mime="text/csv", key="raw_csv_all",
                        help="Everything pulled, not just what's plotted: one row per "
                             "level per cycle, with every parameter the raw files "
                             "report and its QC flag, plus cycle, time and position. "
                             "Ready to load straight into MATLAB, pandas or R.")
                    st.caption(
                        f"CSV carries all {len(raw_params)} measurand(s) the raw files "
                        "report, plus their QC flags, not only the ones plotted above.")

# ===================== Overlay: multi-measurand profile over a cycle range =====================
with tab_overlay:
    st.subheader("Multi-measurand overlay")
    st.caption("Overlay measurands on one pressure axis, each with its own color-matched "
               "x-axis, over a chosen range of profiles. The float drifts, so narrowing "
               "the range isolates a region and time window. Set the range to a single "
               "profile for one cast.")
    if "CYCLE_NUMBER" not in ds:
        st.info("This float has no cycle numbering to select a profile range.")
    else:
        _cyc = sorted({int(c) for c in np.asarray(ds["CYCLE_NUMBER"].values).ravel()
                       if np.isfinite(c)})
        cmin, cmax = _cyc[0], _cyc[-1]
        oc1, oc2 = st.columns([1, 1])
        if cmin < cmax:
            X, Y = oc1.slider("Profiles (cycle range)", cmin, cmax, (cmin, cmax),
                              help="Profile X to Profile Y. Set both to the same number "
                                   "for a single cast.")
        else:
            X, Y = cmin, cmax
            oc1.caption(f"Only cycle {cmin} is available.")
        _ovl_def = ([p for p in ("TEMP", "PSAL", "DOXY") if p in param_opts][:3]
                    or param_opts[:2])
        measur = oc2.multiselect(
            "Measurands to overlay (up to 4)", param_opts, default=_ovl_def,
            max_selections=4,
            help=f"Each gets its own color and x-axis. Takes the adjusted-vs-real-time "
                 f"field from the {view} data view (set on Profile & Trend); QC "
                 "filtering is controlled below.")
        ck1, ck2 = st.columns([1, 1])
        qc_on = ck1.checkbox(
            "Apply QA/QC filtering", value=apply_qc,
            help="On: drop levels flagged questionable or bad (QC 3, 4, 9). Off: plot "
                 "every reported level, flags and all, for this overlay only. Turn it "
                 "off to see what the QC screened out.")
        show_band = ck2.checkbox(
            "Show ±1σ spread", value=True,
            help="Shade each measurand to plus or minus one standard deviation across "
                 "the selected profiles, so you can see how variable the water column "
                 "was over that window.")
        if not measur:
            st.info("Pick one or more measurands to overlay.")
        else:
            profiles = []
            for m in measur:
                dfm, pcol, vcol = param_long_frame(ds, m, adjusted, qc_on)
                if dfm.empty or "cycle" not in dfm.columns:
                    continue
                mp = _stat_profile(dfm[dfm["cycle"].between(X, Y)])
                if mp is None:
                    continue
                units = ds[vcol].attrs.get("units") if vcol in ds else None
                profiles.append((m, mp[0], mp[1], mp[2], units))
            if not profiles:
                st.warning(f"No data for these measurands over profiles {X} to {Y}. "
                           "Try a wider range or different measurands.")
            else:
                st.plotly_chart(_overlay_fig(profiles, show_band), width="stretch")
                cn = np.asarray(ds["CYCLE_NUMBER"].values).ravel()
                msk = np.isfinite(cn) & (cn >= X) & (cn <= Y)
                bits = [f"Profiles {X} to {Y}"]
                if "JULD" in ds:
                    jt = pd.to_datetime(
                        pd.Series(np.asarray(ds["JULD"].values).ravel()[msk]),
                        errors="coerce").dropna()
                    if len(jt):
                        bits.append(f"{jt.min():%Y-%m-%d} to {jt.max():%Y-%m-%d}")
                if "LATITUDE" in ds and "LONGITUDE" in ds:
                    la = np.asarray(ds["LATITUDE"].values).ravel()[msk]
                    lo = np.asarray(ds["LONGITUDE"].values).ravel()[msk]
                    fin = np.isfinite(la) & np.isfinite(lo)
                    if fin.any():
                        mla, mlo = float(np.median(la[fin])), float(np.median(lo[fin]))
                        bits.append(f"near {climate_band(mla)} {ocean_basin(mla, mlo)}")
                st.caption(" · ".join(bits) + ". Lines are the mean over the selected "
                           "profiles, shaded to ±1 standard deviation; each measurand "
                           "keeps its own color and x-axis.")


# ===================== Tech: engineering telemetry from <wmo>_tech.nc =====================
with tab_tech:
    st.caption("Engineering telemetry from the float's tech file: battery voltages, "
               "pump activity and cycle timings, straight from the GDAC. Parameter "
               "names vary by float model and DAC, so the list below is whatever "
               "this float actually reports.")
    _trel = f"dac/{frow.get('dac')}/{sel_wmo}/{sel_wmo}_tech.nc"
    _tdest = os.path.join(ROOT, _trel)
    if os.path.exists(_tdest) and os.path.getsize(_tdest) > 0:
        _tst = "ok"
    else:
        with st.spinner("Fetching the tech file from the GDAC…"):
            _tst = _gdac_get(_trel)
    if _tst == "missing":
        st.info("The GDAC has no tech file for this float.")
    elif _tst == "unreachable":
        st.warning("Couldn't reach the GDAC for the tech file. This looks like a "
                   "connection problem; try again in a moment.")
    else:
        _tech = load_tech(_tdest)
        if _tech is None or not len(_tech):
            st.info("This float's tech file has no readable technical parameters.")
        else:
            _tc1, _tc2 = st.columns(2)
            _tc1.metric("Technical parameters", f"{_tech['name'].nunique():,}")
            _tc2.metric("Cycles covered",
                        f"{int(_tech['cycle'].nunique()):,}")
            _tnum = _tech[pd.notna(_tech["num"])]
            _tnames = sorted(_tnum["name"].unique())
            if not _tnames:
                st.info("No numeric technical parameters to plot; the raw values "
                        "are in the table below.")
            else:
                _bat = [n for n in _tnames
                        if "VOLT" in n.upper() or "BATTER" in n.upper()]
                if st.session_state.get("tech_param") not in _tnames:
                    st.session_state["tech_param"] = _bat[0] if _bat else _tnames[0]
                _tp = st.selectbox(
                    "Technical parameter", _tnames, key="tech_param",
                    help="Numeric parameters only. Battery voltages are the usual "
                         "health check: a steady decline is normal ageing, a sharp "
                         "drop is trouble.")
                _tsub = _tnum[_tnum["name"] == _tp].sort_values("cycle")
                _fig_t = go.Figure(go.Scatter(
                    x=_tsub["cycle"], y=_tsub["num"], mode="lines+markers",
                    line=dict(color=_chart_colors()[0]["Core"], width=2),
                    marker=dict(size=5),
                    hovertemplate="cycle %{x:.0f} · %{y}<extra></extra>"))
                _fig_t.update_layout(height=340,
                                     margin=dict(l=10, r=10, t=30, b=10),
                                     xaxis_title="cycle", yaxis_title=_tp)
                st.plotly_chart(_fig_t, width="stretch")
            with st.expander("All technical parameters at the latest cycle"):
                _lc = _tech["cycle"].max()
                st.dataframe(
                    _tech[_tech["cycle"] == _lc][["name", "value"]]
                    .reset_index(drop=True),
                    width="stretch", height=300)
