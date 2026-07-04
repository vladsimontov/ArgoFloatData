#!/usr/bin/env python3
"""
argo_dashboard.py  —  BGC-Argo explorer
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

st.set_page_config(page_title="BGC-Argo Explorer", layout="wide",
                   initial_sidebar_state="expanded")

# small style polish (Streamlit constrains most styling)
st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  [data-testid="stMetricValue"] {font-size: 1.1rem;}
  .stDataFrame {font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


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


def fetch_from_gdac(rel, retries=3):
    """Download dac/<...>.nc from the GDAC into ROOT (cached on disk). True on success."""
    dest = os.path.join(ROOT, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    import requests
    import time as _time
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{GDAC}/{rel}"
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=180,
                              headers={"User-Agent": "argo-dashboard/0.2"}) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                os.replace(tmp, dest)
            return True
        except Exception:
            if i == retries - 1:
                return False
            _time.sleep(1.5 * (i + 1))
    return False


@st.cache_data(show_spinner=True)
def load_sprof(path):
    return add_derived(xr.open_dataset(path, decode_times=True))


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

st.title("BGC-Argo Explorer")
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
c5.metric("Last synced (UTC)", mani.get("synced_utc", "—"),
          help="When the metadata index was last refreshed.")
st.caption("Float data is fetched on demand from the Argo GDAC when you open a float"
           " · the GDAC is mutable (delayed-mode QC rewrites history), so the index "
           "is refreshed periodically; pin a DOI snapshot for publications.")

# ---- acknowledgements · data source · license (always visible) ----
with st.expander("🙏 Acknowledgements · data source · license", expanded=False):
    st.markdown("""
**Thank you to Argo — and to the people and nations who make it possible.**

Every profile in this tool exists because of the **International Argo Program** and
the ~30 nations — their governments, agencies, engineers, and scientists — who fund,
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
(Met Office / BODC), the United States (NOAA), **Euro-Argo ERIC**, and every other
nation and government contributing to Argo. The full float array is a shared gift.

**License** — Argo data are freely available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This tool applies QC
filtering and computes derived quantities (TEOS-10 density, mixed-layer depth,
apparent oxygen utilization, absolute/conservative properties, trends) that are
**not official Argo products**. Data are retrieved from the Argo GDAC.

*Built with the help of [Claude](https://claude.ai) (Anthropic). An independent,
community tool — not affiliated with or endorsed by the Argo Program.*
""")
    st.markdown(
        f"**🐛 Found a bug, or have a request?** "
        f"[Open an issue]({NEW_ISSUE_URL}) · [browse issues]({ISSUES_URL}). "
        "Feedback on the science, the QC handling, or a float that won't load is "
        "all welcome.")

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
    "🌊 Data: **International Argo Program** & its member nations — "
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
if serial_q.strip() or wmo_q.strip() or model_q != "(any)":
    unique_only = st.toggle(
        "Show unique floats only", value=False,
        help="Collapse the per-sensor rows into one row per float (WMO), "
             "listing how many sensors matched and their models.")
    if unique_only:
        show = (hits.groupby("wmo", as_index=False)
                .agg(float_serial_no=("float_serial_no", "first"),
                     dac=("dac", "first"),
                     n_sensors=("sensor", "nunique"),
                     sensor_models=("sensor_model",
                                    lambda s: ", ".join(
                                        sorted(s.dropna().unique()))))
                .reset_index(drop=True))
    else:
        show = hits[["wmo", "float_serial_no", "sensor", "sensor_model",
                     "sensor_maker", "sensor_serial_no", "dac"]].reset_index(drop=True)
    st.dataframe(show, width="stretch", height=220)
    wmos = sorted(hits["wmo"].astype(str).unique().tolist())
else:
    st.info("Enter a serial number, sensor model, or WMO in the sidebar.")
    wmos = []

if not wmos:
    st.stop()

sel_wmo = st.selectbox("Select a float (WMO) to inspect", wmos)

# ---- float dossier ----
frow = floats[floats["wmo"].astype(str) == str(sel_wmo)]
frow = frow.iloc[0] if len(frow) else None
fsens = sensors[sensors["wmo"].astype(str) == str(sel_wmo)]

# load profile data early (cached) so the dossier can fall back to it when the
# sync index lacks last_date / last position for this float
sprof_rel = frow.get("sprof_path") if frow is not None else None
has_local = bool(sprof_rel) and os.path.exists(os.path.join(ROOT, str(sprof_rel)))
ds = load_sprof(os.path.join(ROOT, str(sprof_rel))) if has_local else None

st.markdown("---")
left, right = st.columns([1, 1])
with left:
    st.subheader(f"Float {sel_wmo}")
    if frow is not None:
        launch = parse_argo_date(frow.get("launch_date"))
        last = parse_argo_date(frow.get("last_date"))
        llat, llon = frow.get("last_lat"), frow.get("last_lon")
        # fall back to the profile file when the sync index lacks these
        if ds is not None:
            if pd.isna(last) and "JULD" in ds:
                jmax = pd.Series(ds["JULD"].values).max()
                last = pd.Timestamp(jmax) if pd.notna(jmax) else last
            if (pd.isna(llat) or pd.isna(llon)) and "LATITUDE" in ds:
                la, lo = ds["LATITUDE"].values, ds["LONGITUDE"].values
                ok = np.isfinite(la) & np.isfinite(lo)
                if ok.any():
                    llat, llon = float(la[ok][-1]), float(lo[ok][-1])
        if pd.notna(launch) and pd.notna(last) and last >= launch:
            dd = (last - launch).days
            deployed = f"{dd:,} days (~{dd / 365.25:.1f} yr), last profile {last:%Y-%m-%d}"
        else:
            deployed = "—"
        if pd.notna(llat) and pd.notna(llon):
            lon_n = ((float(llon) + 180) % 360) - 180
            position = (f"{abs(float(llat)):.2f}°{'N' if llat >= 0 else 'S'}, "
                        f"{abs(lon_n):.2f}°{'E' if lon_n >= 0 else 'W'}")
            region = f"{climate_band(float(llat))} · {ocean_basin(float(llat), float(llon))}"
        else:
            position, region = "—", "—"
        _kind = frow.get("data_kind")
        data_file = ({"core": "core (prof.nc — physical T/S)",
                      "bgc": "BGC (Sprof.nc — synthetic)"}.get(_kind)
                     if pd.notna(_kind) else "—")
        st.write({
            "Float serial no.": frow.get("float_serial_no"),
            "WMO": frow.get("wmo"),
            "DAC": frow.get("dac"),
            "Data file": data_file,
            "Platform type": frow.get("platform_type"),
            "Platform maker": frow.get("platform_maker"),
            "WMO inst type": frow.get("wmo_inst_type"),
            "Launch date": f"{launch:%Y-%m-%d}" if pd.notna(launch)
                           else frow.get("launch_date"),
            "Days deployed": deployed,
            "Last position": position,
            "Region (approx.)": region,
            "Project / PI": f"{frow.get('project_name')} / {frow.get('pi_name')}",
        })
with right:
    st.subheader("Sensors on this float")
    st.dataframe(
        fsens[["sensor", "sensor_model", "sensor_maker", "sensor_serial_no"]]
        .reset_index(drop=True),
        width="stretch", height=240)

# measurands
st.subheader("Measurands on board")
def _param_str(row, key):
    # NaN is truthy in Python, so guard with notna before using as a string.
    v = row.get(key) if row is not None else None
    return str(v).strip() if pd.notna(v) else ""
params_str = _param_str(frow, "parameters") or _param_str(frow, "parameters_meta")
measurands = [p for p in params_str.split() if p]
if measurands:
    st.write(", ".join(measurands))
else:
    st.write("_(not listed in index/meta; will read from data file if available)_")

# ---- data: local, else fetch from the GDAC on demand (cached) ----
st.markdown("---")
st.subheader("Location & plots")

if not has_local:
    # the expected file for this float, from the metadata index
    rel = frow.get("expected_rel") if frow is not None else None
    if not (isinstance(rel, str) and rel):
        suffix = "_Sprof.nc" if str(frow.get("data_kind")) == "bgc" else "_prof.nc"
        rel = f"dac/{frow.get('dac')}/{sel_wmo}/{sel_wmo}{suffix}"
    st.info(f"Fetching this float's data from the Argo GDAC on demand "
            f"(`{os.path.basename(rel)}`) — cached after the first load.")
    with st.spinner(f"Downloading {os.path.basename(rel)} from the GDAC…"):
        ok = fetch_from_gdac(rel)
    if not ok:
        st.error(f"Couldn't fetch `{rel}` from the GDAC. The file may not exist for "
                 "this float (try the other data type), or the GDAC is unreachable.")
        st.stop()
    sprof_rel = rel
    has_local = True
    ds = load_sprof(os.path.join(ROOT, rel))

# refine measurands from the actual file if meta was empty
if not measurands:
    measurands = [v for v in ds.data_vars
                  if ds[v].dims == ("N_PROF", "N_LEVELS")
                  and not v.endswith(("_QC", "_ADJUSTED", "_ADJUSTED_QC",
                                      "_ADJUSTED_ERROR"))]

# WMO tag stamped on every plot: makes clear which float a chart is for and that
# it isn't stale (latest-profile date + cycle count are freshness signals)
_last_juld = pd.Series(ds["JULD"].values).max() if "JULD" in ds else pd.NaT
_ncyc = int(ds.sizes.get("N_PROF", 0))
float_tag = f"Float {sel_wmo}"
if pd.notna(_last_juld):
    float_tag += f"  ·  latest profile {pd.Timestamp(_last_juld):%Y-%m-%d}"
if _ncyc:
    float_tag += f"  ·  {_ncyc} cycles"


def _titled(fig, text):
    fig.update_layout(title=dict(text=text, x=0.0, xanchor="left",
                                 font=dict(size=13)),
                      margin=dict(t=48))
    return fig


# trajectory map (plotly, no token needed)
lat = ds["LATITUDE"].values if "LATITUDE" in ds else None
lon = ds["LONGITUDE"].values if "LONGITUDE" in ds else None
if lat is not None and lon is not None and np.isfinite(lat).any():
    traj = pd.DataFrame({"lat": lat, "lon": lon,
                         "cycle": ds["CYCLE_NUMBER"].values
                         if "CYCLE_NUMBER" in ds else np.arange(len(lat))})
    traj = traj.dropna(subset=["lat", "lon"])
    fig_map = px.line_geo(traj, lat="lat", lon="lon")
    fig_map.add_trace(go.Scattergeo(
        lat=traj["lat"], lon=traj["lon"], mode="markers",
        marker=dict(size=5), text=traj["cycle"], name="profiles"))
    fig_map.update_geos(fitbounds="locations", showland=True,
                        landcolor="rgb(240,240,235)", showcountries=True)
    fig_map.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                          showlegend=False)
    _titled(fig_map, f"{float_tag} — trajectory")
    st.plotly_chart(fig_map, width="stretch")

# controls
cc1, cc2, cc3 = st.columns([2, 1, 1])
plottable = [p for p in measurands if p in ds or f"{p}_ADJUSTED" in ds]
derived_here = [d for d in DERIVED_2D if d in ds and d not in plottable]
if derived_here:
    plottable = plottable + derived_here     # TEOS-10 fields at the end
param = cc1.selectbox("Parameter to plot", plottable or measurands,
                      help="Includes TEOS-10 derived fields "
                           "(SIGMA0, CT, PT, SA, AOU) when computable.")
adjusted = cc2.toggle("Use ADJUSTED", value=True,
                      help="Delayed-mode/adjusted values are science-ready; "
                           "raw are not calibration-corrected.")
apply_qc = cc3.toggle("QC filter (1,2,5,8)", value=True)

if param:
    st.caption(f"Data mode for {param}: {data_mode_for(ds, param)}")
    df, pcol, vcol = param_long_frame(ds, param, adjusted, apply_qc)
    if df.empty:
        st.warning("No finite points after selection. Try toggling ADJUSTED/QC — "
                   "this parameter may only exist in raw form for this float.")
    else:
        color = "time" if "time" in df else "cycle"
        def _axis_label(var):
            units = ds[var].attrs.get("units") if var in ds else None
            return f"{var} [{units}]" if units else var
        fig = px.scatter(df, x="value", y="pres", color=color,
                         labels={"value": _axis_label(vcol),
                                 "pres": _axis_label(pcol)},
                         height=560)
        fig.update_yaxes(autorange="reversed")  # depth downward
        fig.update_traces(marker=dict(size=4))
        _titled(fig, f"{float_tag} — {param} profile")
        st.plotly_chart(fig, width="stretch")

        # ---- downloads ----
        d1, d2 = st.columns(2)
        d1.download_button("Download this parameter (CSV)",
                           df.to_csv(index=False).encode(),
                           file_name=f"{sel_wmo}_{param}.csv", mime="text/csv")

        # serve the pristine original file straight from disk (robust for both
        # Sprof and core prof.nc; avoids to_netcdf round-trip encoding issues)
        _orig_path = os.path.join(ROOT, str(sprof_rel))
        with open(_orig_path, "rb") as _f:
            _orig_bytes = _f.read()
        d2.download_button("Download full float (NetCDF)",
                           _orig_bytes,
                           file_name=os.path.basename(_orig_path),
                           mime="application/x-netcdf")

        # ---- depth-time section (Hovmoller) ----
        st.markdown("---")
        st.subheader(f"{param} depth–time section")
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
            # overlay MLD if available
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
            _titled(fig_sec, f"{float_tag} — {param} depth–time section")
            st.plotly_chart(fig_sec, width="stretch")
            st.caption("Each profile linearly interpolated onto a common pressure "
                       "grid. White dotted line = mixed-layer depth (where computed).")

        # ---- T-S diagram with density contours ----
        st.markdown("---")
        st.subheader("Temperature–Salinity diagram")
        tsf = ts_diagram_frame(ds)
        if tsf is None or tsf[0].empty:
            st.info("Salinity/temperature not available for a T-S diagram.")
        else:
            tsdf, xn, yn = tsf
            # thin very dense clouds for responsiveness
            if len(tsdf) > 20000:
                tsdf = tsdf.sample(20000, random_state=0)
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
            fig_ts_d.add_trace(go.Scatter(
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
            _titled(fig_ts_d, f"{float_tag} — T–S diagram")
            st.plotly_chart(fig_ts_d, width="stretch")
            st.caption("Grey contours = potential density σ₀ (kg/m³); points colored "
                       "by pressure. Water masses cluster along isopycnals."
                       if xn == "SA" else
                       "Practical salinity vs in-situ temperature (install gsw for "
                       "σ₀ contours & absolute salinity).")

        # ---- time series at a fixed pressure level ----
        st.markdown("---")
        st.subheader(f"{param} time series at a pressure level")
        if "time" not in df.columns:
            st.info("This float has no JULD/time coordinate; can't build a time series.")
        else:
            pmin, pmax = float(df["pres"].min()), float(df["pres"].max())
            default_p = float(min(max(20.0, pmin), pmax))
            tcol1, tcol2 = st.columns(2)
            target_p = tcol1.number_input(
                "Target pressure (dbar)", min_value=pmin, max_value=pmax,
                value=round(default_p, 1), step=1.0,
                help="For each profile the nearest available pressure level is used.")
            max_gap = tcol2.number_input(
                "Max distance from target (dbar)", min_value=0.0,
                value=25.0, step=5.0,
                help="Drop profiles whose nearest sample is farther than this "
                     "from the target pressure.")

            # nearest level per profile -> one point per cycle
            near = (df.assign(dp=(df["pres"] - target_p).abs())
                      .sort_values("dp")
                      .groupby("cycle", as_index=False)
                      .first())
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

                # season, hemisphere-aware (meteorological)
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
                # anomaly = value minus its calendar-month mean
                near["anom"] = near["value"] - near.groupby("cal_month")["value"] \
                                                   .transform("mean")
                ycol = "anom" if deseason else "value"
                season_colors = {"Winter": "#4C72B0", "Spring": "#55A868",
                                 "Summer": "#C44E52", "Fall": "#DD8452"}

                units = ds[vcol].attrs.get("units") if vcol in ds else None
                base_lab = f"{param} anomaly" if deseason else param
                ylab = f"{base_lab} [{units}]" if units else base_lab

                fig_ts = go.Figure()
                # faint time-ordered connector behind the seasonal markers
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
                # robust trend: Sen's slope line + Mann-Kendall significance
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
                _titled(fig_ts, f"{float_tag} — {param} at {target_p:g} dbar")
                st.plotly_chart(fig_ts, width="stretch")

                # trend statistics panel
                if res and np.isfinite(res["sen"]):
                    p = res["p"]
                    sig = ("not significant (p≥0.05)"
                           if not np.isfinite(p) or p >= 0.05 else
                           "significant (p<0.05)" if p >= 0.01 else
                           "highly significant (p<0.01)")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Sen's slope", f"{res['sen']:+.3g} {units or ''}/yr")
                    m2.metric("Mann-Kendall p",
                              "n/a" if not np.isfinite(p) else f"{p:.3g}")
                    m3.metric("Trend", sig)
                    st.caption(
                        f"n={res['n']} profiles · Sen's slope = median pairwise rate "
                        "(outlier-robust); Mann-Kendall tests for a monotonic trend. "
                        + ("Series shown is the deseasonalized anomaly."
                           if deseason else
                           "Turn on deseasonalize to remove the annual cycle first."))
                st.caption(
                    f"{len(near)} profiles · nearest-level pressures "
                    f"{near['pres'].min():.1f}–{near['pres'].max():.1f} dbar "
                    f"(target {target_p:g}) · {hemi}-hemisphere meteorological seasons.")
                st.download_button(
                    "Download time series (CSV)",
                    near[["time", "year", "season", "pres", "value", "anom", "cycle"]]
                        .to_csv(index=False).encode(),
                    file_name=f"{sel_wmo}_{param}_ts_{target_p:.0f}dbar.csv",
                    mime="text/csv", key="ts_download")
