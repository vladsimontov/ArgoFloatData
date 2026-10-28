#!/usr/bin/env python3
"""
build_woa.py
============
One-time extraction of World Ocean Atlas 2023 reference climatologies into
small parquets under docs/api/, so the dashboard can overlay a WOA profile
on a float's vertical profile with a pure local read (the Streamlit Cloud
clone ships docs/, so no runtime download is ever needed).

Source: NOAA NCEI WOA23 1-degree annual objectively-analyzed means
(https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/). The source
NetCDFs are 44-81 MB each because they carry every statistic field; only
the `*_an` analyzed mean is kept here, which compresses to a few MB per
variable. WOA23 is a static product, so these outputs never need refresh.

Outputs (lat f4, lon f4, depth f4, val f4; sorted by lat, lon, depth and
written in modest row groups so per-cell predicate reads stay cheap):
  docs/api/woa23_temperature.parquet   (in-situ T, degC)
  docs/api/woa23_salinity.parquet
  docs/api/woa23_oxygen.parquet        (umol/kg)
  docs/api/woa23_aou.parquet           (umol/kg)
  docs/api/woa23_nitrate.parquet       (umol/kg)

Usage: python build_woa.py [--src ./argo_local/woa_src] [--out ./docs/api]
"""

import argparse
import os

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

BASE = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA"
# param key -> (directory, climatology, variable prefix, output name, decimals)
SETS = {
    "temperature": ("temperature", "decav", "t", 3),
    "salinity":    ("salinity",    "decav", "s", 3),
    "oxygen":      ("oxygen",      "all",   "o", 2),
    "aou":         ("AOU",         "all",   "A", 2),
    "nitrate":     ("nitrate",     "all",   "n", 2),
}


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        with open(dest + ".part", "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(dest + ".part", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./argo_local/woa_src")
    ap.add_argument("--out", default="./docs/api")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, (d, clim, code, dec) in SETS.items():
        url = f"{BASE}/{d}/netcdf/{clim}/1.00/woa23_{clim}_{code}00_01.nc"
        src = os.path.join(args.src, os.path.basename(url))
        download(url, src)
        # WOA climatological time axes are not CF-decodable; we drop time anyway
        ds = xr.open_dataset(src, decode_times=False)
        da = ds[f"{code}_an"].squeeze(drop=True)   # (depth, lat, lon)
        df = da.to_dataframe(name="val").reset_index()
        # the per-cell standard deviation of observations: the climatological
        # spread, which the dashboard shades as a +-1 sd envelope
        if f"{code}_sd" in ds:
            sd = (ds[f"{code}_sd"].squeeze(drop=True)
                  .to_dataframe(name="sd").reset_index())
            df = df.merge(sd, on=["depth", "lat", "lon"], how="left")
        else:
            df["sd"] = np.nan
        df = df.dropna(subset=["val"])
        df = df[["lat", "lon", "depth", "val", "sd"]].astype("float32")
        df["val"] = df["val"].round(dec)
        df["sd"] = df["sd"].round(dec)
        df = df.sort_values(["lat", "lon", "depth"]).reset_index(drop=True)
        out = os.path.join(args.out, f"woa23_{name}.parquet")
        df.to_parquet(out, index=False, compression="zstd",
                      row_group_size=200_000)
        print(f"{out}: {len(df):,} cells, "
              f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
