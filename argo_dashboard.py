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

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go

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

# small style polish (Streamlit constrains most styling)
st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
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


def _gdac_get(rel, retries=3):
    """Download dac/<...>.nc from the GDAC into ROOT (cached on disk). Returns one of:
    'ok'          - present locally or downloaded now,
    'missing'     - the server answered 4xx (e.g. 404): the file does not exist,
    'unreachable' - network error / timeout / 5xx after retries: could not contact it.
    The missing/unreachable split lets callers tell 'no such file' from 'no connection'."""
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
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
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

st.title("🌊 Argo Float Data Explorer")
st.caption("BGC, Core & Deep Argo float profiles.  Search the "
           "whole array by sensor serial number, model or WMO.")
if sensors is None:
    st.error(
        f"No index found in `{ROOT}`.\n\n"
        "Run the pipeline first:\n"
        "```\npython sync_bgc_subset.py --root ./argo_local --budget-gb 5\n"
        "python build_crosswalk.py --root ./argo_local\n```"
    )
    st.stop()

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
st.caption("Float data is fetched on demand from the Argo GDAC when you open a float"
           " · the GDAC is mutable (delayed-mode QC rewrites history), so the index "
           "is refreshed periodically; pin a DOI snapshot for publications.")

# ---- acknowledgements · data source · license (always visible) ----
with st.expander("Acknowledgements | Data Source | License", expanded=False):
    st.markdown("""
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
""")
    st.markdown(
        f"**🐛 Found a bug, or have a request?** "
        f"[Open an issue]({NEW_ISSUE_URL}) · [browse issues]({ISSUES_URL}). "
        "All Feedback and Feature requests are welcome.")

# ---- sidebar: search ----
st.sidebar.header("Find a float")
serial_q = st.sidebar.text_input("Serial number contains",
                                 help="Matches SENSOR_SERIAL_NO or FLOAT_SERIAL_NO "
                                      "(case-insensitive substring).")
models = ["(any)"] + sorted(sensors["sensor_model"].dropna().unique().tolist())
model_q = st.sidebar.selectbox("Sensor model", models,
                               help="e.g. SBE41, SBE41CP for the CTD.")
wmo_q = st.sidebar.text_input("…or WMO number", help="Jump straight to a float.")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"🐛 [Report a bug / request a feature]({NEW_ISSUE_URL})")
st.sidebar.caption(
    "🌊🌊🌊 Data: **International Argo Program** & its member nations, "
    "freely shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). "
    "[doi.org/10.17882/42182](https://doi.org/10.17882/42182). "
    "With thanks. See **Acknowledgements** up top.")

# resolve search -> candidate floats
def search():
    df = sensors.copy()
    if model_q != "(any)":
        df = df[df["sensor_model"].fillna("") == model_q]
    if serial_q.strip():
        s = serial_q.strip().lower()
        m = (df["sensor_serial_no"].fillna("").str.lower().str.contains(s) |
             df["float_serial_no"].fillna("").str.lower().str.contains(s))
        df = df[m]
    if wmo_q.strip():
        df = df[df["wmo"].astype(str).str.contains(wmo_q.strip())]
    return df

hits = search()

st.subheader("Matches")

# compact float-type label per WMO (BGC / Core, flagged Deep) for the table
def _float_type(dk, deep):
    base = "BGC" if dk == "bgc" else "Core" if dk == "core" else "-"
    return f"{base} · Deep" if deep else base
type_by_wmo = {}
if {"data_kind", "is_deep"}.issubset(floats.columns):
    type_by_wmo = {str(w): _float_type(dk, bool(d)) for w, dk, d in
                   zip(floats["wmo"], floats["data_kind"], floats["is_deep"])}

if serial_q.strip() or wmo_q.strip() or model_q != "(any)":
    n_floats = int(hits["wmo"].nunique())
    st.markdown(f"**{n_floats:,} float{'s' if n_floats != 1 else ''} found**")
    unique_only = st.toggle(
        "Show unique floats only", value=True,
        help="On: one row per float (WMO), with how many sensors matched and their "
             "models. Off: one row per matching sensor.")
    # When searching by serial, label what triggered each match (the sensor and its
    # serial, or the float serial) so the hit is visible even in the grouped view.
    serial_hit = bool(serial_q.strip())
    if serial_hit:
        _sq = serial_q.strip().lower()
        _ssn = hits["sensor_serial_no"].fillna("").astype(str)
        _fsn = hits["float_serial_no"].fillna("").astype(str)
        _snm = hits["sensor_model"].fillna("").astype(str)
        _snm = _snm.where(_snm.ne(""), hits["sensor"].fillna("sensor").astype(str))
        hits = hits.assign(matched_on=(_snm + " · " + _ssn).where(
            _ssn.str.lower().str.contains(_sq, regex=False), "float s/n " + _fsn))

    if unique_only:
        agg = dict(float_serial_no=("float_serial_no", "first"),
                   dac=("dac", "first"),
                   n_sensors=("sensor", "nunique"),
                   sensor_models=("sensor_model",
                                  lambda s: ", ".join(sorted(s.dropna().unique()))))
        if serial_hit:
            agg["matched_on"] = ("matched_on",
                                 lambda s: ", ".join(sorted({x for x in s if x})))
        show = hits.groupby("wmo", as_index=False).agg(**agg).reset_index(drop=True)
    else:
        cols = ["wmo", "float_serial_no", "sensor", "sensor_model",
                "sensor_maker", "sensor_serial_no", "dac"]
        if serial_hit:
            cols.append("matched_on")
        show = hits[cols].reset_index(drop=True)
    show.insert(1, "type", show["wmo"].astype(str).map(type_by_wmo).fillna("-"))
    if serial_hit:
        show.insert(2, "matched_on", show.pop("matched_on"))   # prominent: right after type
    cap = "Select a row's checkbox (far left) to open that float ↓"
    if serial_hit:
        cap += "  ·  the highlighted **matched on** column shows what your serial hit"
    st.caption(cap)
    _cfg = ({"matched_on": st.column_config.TextColumn(
                "matched on", help="Which sensor serial (or the float serial) your "
                                   "search matched.")} if serial_hit else None)
    # Highlight the currently-selected row green. Read the selection recorded before
    # this rerun (from the checkbox click that triggered it); Styler colors are
    # explicit, so the green holds in both light and dark mode.
    _ms = st.session_state.get("matches_table")
    _seld = ((_ms.get("selection") if isinstance(_ms, dict)
              else getattr(_ms, "selection", None)) if _ms is not None else None)
    _selrows = ((_seld.get("rows", []) if isinstance(_seld, dict)
                 else getattr(_seld, "rows", []) or []) if _seld is not None else [])
    _selrow0 = _selrows[0] if _selrows else None
    _sty = show.style
    if serial_hit:
        _sty = _sty.set_properties(subset=["matched_on"],
                                   **{"background-color": "#fff3cd", "color": "#663c00"})
    if _selrow0 is not None and _selrow0 < len(show):
        _sty = _sty.apply(lambda r: (["background-color:#c3ecd0; color:#0b6b3a"] * len(r)
                                     if r.name == _selrow0 else [""] * len(r)), axis=1)
    event = st.dataframe(_sty, width="stretch", height=220,
                         on_select="rerun", selection_mode="single-row",
                         key="matches_table", column_config=_cfg)
    wmos = sorted(hits["wmo"].astype(str).unique().tolist())
    # Clicking a table row opens that float by pre-filling the picker below.
    # Forget the remembered click whenever the result set changes, and act only
    # on a *new* row so the dropdown can still override a row selection.
    sig = (serial_q, wmo_q, model_q, unique_only)
    if st.session_state.get("_match_sig") != sig:
        st.session_state["_match_sig"] = sig
        st.session_state.pop("_last_row", None)
    _sel = getattr(event, "selection", None)
    _rows = list(_sel.rows) if _sel and getattr(_sel, "rows", None) else []
    _cur = _rows[0] if _rows else None
    if _cur is not None and _cur < len(show) and _cur != st.session_state.get("_last_row"):
        st.session_state["_last_row"] = _cur
        st.session_state["wmo_pick"] = str(show.iloc[_cur]["wmo"])
else:
    st.info("Enter a serial number, sensor model, or WMO in the sidebar.")
    wmos = []

if not wmos:
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
    st.stop()

# ---- resolve the selected float and load its data ----
frow = floats[floats["wmo"].astype(str) == str(sel_wmo)]
frow = frow.iloc[0] if len(frow) else None
fsens = sensors[sensors["wmo"].astype(str) == str(sel_wmo)]

sprof_rel = frow.get("sprof_path") if frow is not None else None
has_local = bool(sprof_rel) and os.path.exists(os.path.join(ROOT, str(sprof_rel)))
ds = load_sprof(os.path.join(ROOT, str(sprof_rel))) if has_local else None

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
    st.info(f"Fetching this float's data from the Argo GDAC on demand "
            f"(`{os.path.basename(rel)}`), cached after the first load.")
    with st.spinner(f"Downloading {os.path.basename(rel)} from the GDAC…"):
        status = _gdac_get(rel)
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
                                 font=dict(size=13)), margin=dict(t=48))
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


def _overlay_fig(profiles, show_band=True):
    """Multi-x-axis overlay: a shared reversed pressure y-axis, one color-matched
    x-axis per measurand (first on the bottom, the rest stacked on top). Each entry
    is (name, pres, mean, std, units); with show_band a shaded +/- 1 std ribbon is
    drawn behind each mean line."""
    ntop = len(profiles) - 1
    step = 0.09
    ytop = 1.0 - step * ntop if ntop else 1.0
    fig = go.Figure()
    layout = {"height": 600, "margin": dict(l=64, r=24, t=16, b=46),
              "showlegend": False,
              "yaxis": dict(title="PRES [decibar]", autorange="reversed",
                            domain=[0.0, ytop])}
    for i, (name, pres, mean, std, units) in enumerate(profiles):
        color = _OVL_COLORS[i % len(_OVL_COLORS)]
        xa = "x" if i == 0 else f"x{i + 1}"
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
        title = f"{name} [{units}]" if units else name
        ax = dict(title=dict(text=title, font=dict(color=color, size=12)),
                  tickfont=dict(color=color, size=10), showgrid=(i == 0), zeroline=False)
        if i == 0:
            ax["side"] = "bottom"
            layout["xaxis"] = ax
        else:
            ax.update(overlaying="x", side="top", anchor="free",
                      position=min(1.0, ytop + step * (i - 1) + step * 0.35))
            layout[f"xaxis{i + 1}"] = ax
    fig.update_layout(**layout)
    return fig


# ---- per-float views as tabs (sidebar search + results table stay global) ----
tab_over, tab_traj, tab_prof, tab_overlay, tab_ts, tab_raw = st.tabs(
    ["Overview", "Trajectory", "Profile & Trend", "Overlay", "T-S", "Raw"])

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
    param = pc1.selectbox("Parameter to plot", param_opts,
                          index=_default_param_index(param_opts),
                          help="Includes TEOS-10 derived fields "
                               "(SIGMA0, CT, PT, SA, AOU) when computable.")
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
            fig = go.Figure(go.Scattergl(
                x=df["value"], y=df["pres"], mode="markers",
                marker=dict(size=4, color=cvals, colorscale="Viridis", colorbar=cbar),
                customdata=cdata, hovertemplate=hover))
            fig.update_xaxes(title=vlabel)
            fig.update_yaxes(autorange="reversed", title=plabel)   # depth downward
            fig.update_layout(height=560)
            _titled(fig, f"{float_tag} · {param} profile")
            st.plotly_chart(fig, width="stretch")

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
            "Max distance from target (dbar)", min_value=0.0, value=25.0, step=5.0,
            help="Drop profiles whose nearest sample is farther than this "
                 "from the target pressure.")
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
    st.subheader("Raw profile (per-cycle diagnostics)")
    st.caption("Pulls a single cycle's raw NetCDF straight from the GDAC, with every "
               "parameter it reports, including intermediate sensor signals (raw "
               "fluorescence, backscatter, optode phase, and so on) that are not carried "
               "into the synthetic product. One profile at a time, unadjusted and with "
               "no QC filtering.")
    _is_bgc = str(frow.get("data_kind")) == "bgc" if frow is not None else False
    _dac = frow.get("dac") if frow is not None else None
    _cyc_vals = (sorted({int(c) for c in np.asarray(ds["CYCLE_NUMBER"].values).ravel()
                         if np.isfinite(c)}) if "CYCLE_NUMBER" in ds else [])
    if not _dac or not _cyc_vals:
        st.info("No per-cycle files are available for this float.")
    else:
        rc1, rc2 = st.columns([1, 2])
        sel_cyc = rc1.selectbox("Cycle (profile)", _cyc_vals, index=len(_cyc_vals) - 1,
                                help="Each cycle is one dive. Pulls that cycle's raw file "
                                     "from the GDAC, cached after the first load.")
        _kind = "B-file" if _is_bgc else "profile file"
        with st.spinner(f"Fetching cycle {sel_cyc} {_kind} from the GDAC"):
            raw_rel, raw_status = fetch_raw_profile(_dac, sel_wmo, sel_cyc, _is_bgc)
        if not raw_rel and raw_status == "unreachable":
            st.error(f"Couldn't reach the Argo GDAC to fetch cycle {sel_cyc}. This looks "
                     "like a connection problem, not a missing file. Please try again in "
                     "a moment.")
        elif not raw_rel:
            st.warning(f"No raw file exists at the GDAC for cycle {sel_cyc} of this float. "
                       "Checked both the delayed-mode and real-time file names, and "
                       "neither is there. Try another cycle.")
        else:
            raw_ds = load_raw(os.path.join(ROOT, raw_rel))
            _pdims = raw_ds["PRES"].dims if "PRES" in raw_ds else None
            raw_params = [v for v in raw_ds.data_vars
                          if _pdims is not None and v != "PRES"
                          and raw_ds[v].dims == _pdims
                          and np.issubdtype(raw_ds[v].dtype, np.number)
                          and not v.endswith("_QC")]
            if not raw_params:
                st.info(f"`{os.path.basename(raw_rel)}` reports no plottable measurands.")
            else:
                _def = next((i for i, v in enumerate(raw_params)
                             if not v.endswith("_ADJUSTED")), 0)
                sel_rp = rc2.selectbox("Measurand (raw)", raw_params, index=_def,
                                       help="Every parameter in the raw file, including "
                                            "intermediate sensor signals.")
                st.caption(f"Source: `{os.path.basename(raw_rel)}`, "
                           f"{raw_ds.sizes.get('N_PROF', 1)} profile(s) in this cycle.")
                pres = np.asarray(raw_ds["PRES"].values, dtype="float64").ravel()
                val = np.asarray(raw_ds[sel_rp].values, dtype="float64").ravel()
                m = np.isfinite(pres) & np.isfinite(val)
                if not m.any():
                    st.warning(f"No finite `{sel_rp}` samples in cycle {sel_cyc}.")
                else:
                    units = raw_ds[sel_rp].attrs.get("units")
                    xlab = f"{sel_rp} [{units}]" if units else sel_rp
                    figr = go.Figure(go.Scattergl(
                        x=val[m], y=pres[m], mode="markers+lines",
                        marker=dict(size=5, color="#0b7285"),
                        line=dict(color="rgba(11,114,133,0.35)", width=1),
                        hovertemplate=f"{sel_rp}=%{{x:.4g}}<br>PRES=%{{y:.1f}} dbar"
                                      "<extra></extra>"))
                    figr.update_xaxes(title=xlab)
                    figr.update_yaxes(autorange="reversed", title="PRES [decibar]")
                    figr.update_layout(height=560)
                    _titled(figr, f"{float_tag} · cycle {sel_cyc} raw {sel_rp}")
                    st.plotly_chart(figr, width="stretch")
                    with open(os.path.join(ROOT, raw_rel), "rb") as _f:
                        st.download_button("Download this raw cycle (NetCDF)", _f.read(),
                                           file_name=os.path.basename(raw_rel),
                                           mime="application/x-netcdf", key="raw_dl")

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
