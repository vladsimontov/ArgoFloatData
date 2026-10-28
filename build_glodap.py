#!/usr/bin/env python3
"""
build_glodap.py
===============
One-time extraction of the GLODAPv2.2023 merged bottle-data product into a
slim parquet under docs/api/, so the dashboard can overlay real research
cruise bottle samples near a float with a pure local read (the Streamlit
Cloud clone ships docs/, so no runtime download is ever needed).

Source: NOAA NCEI accession 0283442, the 853 MB merged master CSV
(https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/). GLODAP
marks missing values as -9999 and carries a WOCE-style flag per variable
(2 = good/acceptable); only flag-2 values are kept, per variable, and rows
with no kept variable at all are dropped. Temperature carries no flag in
GLODAP (it is treated as reference quality).

Please cite: Lauvset et al. (2024), ESSD 16, 2047-2072 (GLODAPv2.2023).

Output docs/api/glodap_v2_2023.parquet with columns
  lat f4, lon f4, year i2, month i1, pres f4 (dbar),
  temp f4, psal f4, doxy f4, aou f4, nitrate f4   (umol/kg for the last 3)
sorted by lat, lon, pres and written in modest row groups, so latitude-band
predicate reads stay cheap.

Usage: python build_glodap.py [--src ./argo_local/glodap_src] [--out ./docs/api]
"""

import argparse
import os

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

URL = ("https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/"
       "GLODAPv2.2023_Merged_Master_File.csv")
USECOLS = ["G2year", "G2month", "G2latitude", "G2longitude", "G2pressure",
           "G2temperature", "G2salinity", "G2salinityf", "G2oxygen",
           "G2oxygenf", "G2aou", "G2aouf", "G2nitrate", "G2nitratef"]
# output column -> (value column, flag column or None)
VARS = {"temp": ("G2temperature", None),
        "psal": ("G2salinity", "G2salinityf"),
        "doxy": ("G2oxygen", "G2oxygenf"),
        "aou": ("G2aou", "G2aouf"),
        "nitrate": ("G2nitrate", "G2nitratef")}


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
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
    ap.add_argument("--src", default="./argo_local/glodap_src")
    ap.add_argument("--out", default="./docs/api")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    src = os.path.join(args.src, os.path.basename(URL))
    download(URL, src)
    df = pd.read_csv(src, usecols=USECOLS, na_values=[-9999, "-9999"])
    out = pd.DataFrame({
        "lat": df["G2latitude"].astype("float32"),
        "lon": df["G2longitude"].astype("float32"),
        "year": pd.to_numeric(df["G2year"], errors="coerce")
                .fillna(0).astype("int16"),
        "month": pd.to_numeric(df["G2month"], errors="coerce")
                 .fillna(0).astype("int8"),
        "pres": df["G2pressure"].astype("float32"),
    })
    for name, (vcol, fcol) in VARS.items():
        v = df[vcol].astype("float32")
        if fcol is not None:
            v = v.where(df[fcol] == 2)
        out[name] = v.round(3)
    out = out.dropna(subset=["lat", "lon", "pres"])
    out = out.dropna(subset=list(VARS), how="all")
    out = out.sort_values(["lat", "lon", "pres"]).reset_index(drop=True)
    dest = os.path.join(args.out, "glodap_v2_2023.parquet")
    out.to_parquet(dest, index=False, compression="zstd",
                   row_group_size=150_000)
    print(f"{dest}: {len(out):,} samples from "
          f"{out['year'].min()}-{out['year'].max()}, "
          f"{os.path.getsize(dest) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
