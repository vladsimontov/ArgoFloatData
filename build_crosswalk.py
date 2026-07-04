#!/usr/bin/env python3
"""
build_crosswalk.py
==================
Parse every locally-downloaded meta.nc into two flat tables that the dashboard
loads instantly:

  sensors.parquet : ONE ROW PER (float, sensor)
      wmo, dac, float_serial_no, platform_type, platform_maker, wmo_inst_type,
      launch_date, sensor, sensor_model, sensor_maker, sensor_serial_no
  floats.parquet  : ONE ROW PER float
      wmo, dac, float_serial_no, platform_type, ..., parameters (measurands),
      last_date, last_lat, last_lon, n_profiles, has_local_sprof, sprof_path

This is what makes serial-number lookup possible: argopy has no native
serial -> WMO index, so we build one from SENSOR_SERIAL_NO in the meta files.

Honest caveats
--------------
* Serial-number FORMATS vary by DAC (zero-padding, prefixes, model naming like
  SBE41 vs SBE41CP). The dashboard therefore does case-insensitive SUBSTRING
  matching, not exact equality. Confirm a hit by cross-checking model+maker.
* meta.nc variable names below are the Argo v3.1 conventions and are widely but
  not universally populated; missing fields come back as None rather than error.

Tested? NO. If a variable name below isn't present in your files, it's skipped;
inspect one file with `ncdump -h <wmo>_meta.nc` to confirm names.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    from netCDF4 import Dataset, chartostring
except Exception as e:
    print("netCDF4 is required:", e)
    sys.exit(1)


def _one(nc, name):
    """Return a single stripped string for a scalar/1-D char variable, or None."""
    if name not in nc.variables:
        return None
    try:
        val = chartostring(nc.variables[name][:])
        val = np.atleast_1d(val)
        s = "".join([str(x) for x in val.ravel()]).strip()
        return s or None
    except Exception:
        return None


def _list(nc, name):
    """Return a list of stripped strings for an (N, STRING) char variable."""
    if name not in nc.variables:
        return []
    try:
        arr = chartostring(nc.variables[name][:])
        arr = np.atleast_1d(arr)
        return [str(x).strip() for x in arr.ravel()]
    except Exception:
        return []


def parse_meta(path):
    """Parse one meta.nc -> (float_dict, [sensor_dicts])."""
    wmo_from_dir = os.path.basename(path).split("_")[0]
    dac = os.path.basename(os.path.dirname(os.path.dirname(path)))
    with Dataset(path) as nc:
        wmo = _one(nc, "PLATFORM_NUMBER") or wmo_from_dir
        fdict = {
            "wmo": str(wmo).strip(),
            "dac": dac,
            "float_serial_no": _one(nc, "FLOAT_SERIAL_NO"),
            "platform_type": _one(nc, "PLATFORM_TYPE"),
            "platform_maker": _one(nc, "PLATFORM_MAKER"),
            "platform_family": _one(nc, "PLATFORM_FAMILY"),   # 'FLOAT_DEEP' for deep
            "wmo_inst_type": _one(nc, "WMO_INST_TYPE"),
            "launch_date": _one(nc, "LAUNCH_DATE"),
            "project_name": _one(nc, "PROJECT_NAME"),
            "pi_name": _one(nc, "PI_NAME"),
        }

        sensors = _list(nc, "SENSOR")
        models = _list(nc, "SENSOR_MODEL")
        makers = _list(nc, "SENSOR_MAKER")
        serials = _list(nc, "SENSOR_SERIAL_NO")

        def at(lst, i):
            return lst[i] if i < len(lst) else None

        sensor_rows = []
        for i in range(len(sensors)):
            sensor_rows.append({
                **fdict,
                "sensor": at(sensors, i),
                "sensor_model": at(models, i),
                "sensor_maker": at(makers, i),
                "sensor_serial_no": at(serials, i),
            })

        # measurands actually carried, from PARAMETER (dedup, keep order)
        params = _list(nc, "PARAMETER")
        seen, ordered = set(), []
        for p in params:
            if p and p not in seen:
                seen.add(p)
                ordered.append(p)
        fdict["parameters_meta"] = " ".join(ordered)

    return fdict, sensor_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./argo_local")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    meta_files = glob.glob(os.path.join(root, "dac", "*", "*", "*_meta.nc"))
    print(f"Found {len(meta_files)} meta.nc files under {root}")
    if not meta_files:
        print("Nothing to parse. Run sync_bgc_subset.py first.")
        return 1

    float_rows, sensor_rows = [], []
    for p in meta_files:
        try:
            fd, sr = parse_meta(p)
            float_rows.append(fd)
            sensor_rows.extend(sr)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")

    floats = pd.DataFrame(float_rows).drop_duplicates("wmo")
    sensors = pd.DataFrame(sensor_rows)

    # enrich floats with position/params/recency + which data file to expect
    fidx_path = os.path.join(root, "bgc_float_index.parquet")
    if os.path.exists(fidx_path):
        fidx_all = pd.read_parquet(fidx_path)
        want = ["wmo", "last_date", "last_lat", "last_lon", "parameters",
                "n_profiles", "data_suffix", "data_kind", "profiler_type"]
        fidx = fidx_all[[c for c in want if c in fidx_all.columns]]
        floats = floats.merge(fidx, on="wmo", how="left")

    # local availability FIRST: is a data file already on disk? (prefer Sprof)
    def data_rel(row):
        for suffix in ("_Sprof.nc", "_prof.nc"):
            rel = os.path.join("dac", str(row["dac"]), str(row["wmo"]),
                               f"{row['wmo']}{suffix}")
            if os.path.exists(os.path.join(root, rel)):
                return rel
        return None
    floats["sprof_path"] = floats.apply(data_rel, axis=1)   # None if not local yet
    floats["has_local_sprof"] = floats["sprof_path"].notna()

    if "data_suffix" not in floats.columns:
        floats["data_suffix"] = None

    # expected data file for on-demand GDAC fetch: the LOCAL file if we have it,
    # else the index's data_suffix, else BGC-params heuristic, else core prof.nc
    def _suffix(row):
        p = row["sprof_path"]
        if isinstance(p, str):
            return "_Sprof.nc" if p.endswith("_Sprof.nc") else "_prof.nc"
        s = row.get("data_suffix")
        if s in ("_Sprof.nc", "_prof.nc"):
            return s
        params = f"{row.get('parameters') or ''} {row.get('parameters_meta') or ''}"
        bgc = any(k in params for k in ("DOXY", "CHLA", "BBP", "NITRATE", "PH_IN",
                                        "IRRADIANCE", "CDOM"))
        return "_Sprof.nc" if bgc else "_prof.nc"
    floats["data_suffix"] = floats.apply(_suffix, axis=1)
    floats["data_kind"] = floats["data_suffix"].map(
        lambda s: "bgc" if s == "_Sprof.nc" else "core")
    floats["expected_rel"] = floats.apply(
        lambda r: f"dac/{r['dac']}/{r['wmo']}/{r['wmo']}{r['data_suffix']}", axis=1)

    # deep-float flag: authoritative PLATFORM_FAMILY, with a platform_type backstop
    fam = floats.get("platform_family", pd.Series(index=floats.index, dtype=object))
    pt = floats.get("platform_type", pd.Series(index=floats.index, dtype=object))
    floats["is_deep"] = (fam.fillna("").str.contains("DEEP", case=False) |
                         pt.fillna("").str.contains(r"_D\b|_D_|XUANWU|HM4000",
                                                    case=False, regex=True))

    floats.to_parquet(os.path.join(root, "floats.parquet"))
    sensors.to_parquet(os.path.join(root, "sensors.parquet"))

    print(f"floats.parquet : {len(floats)} floats")
    print(f"sensors.parquet: {len(sensors)} sensor rows")
    n_ctd = sensors["sensor_model"].fillna("").str.contains("SBE41", case=False).sum()
    print(f"  (e.g. {n_ctd} sensor rows whose model contains 'SBE41')")
    print("Next: streamlit run argo_dashboard.py -- --root", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
