#!/usr/bin/env python3
"""
build_api.py
============
Publish the crosswalk as a small static "data API" under docs/api/ so it's
served from the (CDN-backed) GitHub Pages site and queryable over HTTP with
DuckDB / pandas — including the serial-number / sensor lookups that the GDAC,
ERDDAP, and Argovis don't offer.

Outputs (all deterministic, so the daily refresh only commits real changes):
  docs/api/floats.parquet   full per-float table (copy of the crosswalk)
  docs/api/sensors.parquet  full per-sensor table (serial/model/maker)
  docs/api/floats.json      compact per-float index (quick/browser use)
  docs/api/index.json       self-describing manifest (files, schema, counts)
"""

import argparse
import json
import os
import shutil

import pandas as pd

GDAC = "https://data-argo.ifremer.fr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./argo_local")
    ap.add_argument("--out", default="./docs/api")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    floats = pd.read_parquet(os.path.join(args.root, "floats.parquet"))
    sensors = pd.read_parquet(os.path.join(args.root, "sensors.parquet"))

    # 1) serve the full tables as Parquet (byte-identical copy -> deterministic)
    shutil.copyfile(os.path.join(args.root, "floats.parquet"),
                    os.path.join(args.out, "floats.parquet"))
    shutil.copyfile(os.path.join(args.root, "sensors.parquet"),
                    os.path.join(args.out, "sensors.parquet"))

    # 2) compact, actionable per-float JSON (each row carries its GDAC data URL)
    fj = pd.DataFrame({
        "wmo": floats["wmo"].astype(str),
        "type": floats.get("data_kind"),
        "deep": floats.get("is_deep"),
        "dac": floats.get("dac"),
        "platform_type": floats.get("platform_type"),
        "last_lat": floats.get("last_lat"),
        "last_lon": floats.get("last_lon"),
        "last_date": floats.get("last_date"),
        "n_profiles": floats.get("n_profiles"),
        "data_url": floats.get("expected_rel").map(
            lambda r: f"{GDAC}/{r}" if isinstance(r, str) else None),
    })
    fj.to_json(os.path.join(args.out, "floats.json"), orient="records")

    # 3) self-describing manifest (no volatile timestamp -> deterministic)
    def _max_int(s):
        v = pd.to_numeric(s, errors="coerce").max()
        return int(v) if pd.notna(v) else None

    manifest = {
        "name": "Argo Float Data Explorer — data API",
        "description": "Search the whole Argo array by sensor serial number, "
                       "model, WMO, region, or type. Query the Parquet files "
                       "directly with DuckDB or pandas over HTTP.",
        "base_url": "https://vladsimontov.github.io/ArgoFloatData/api/",
        "license": "Argo data: CC BY 4.0 — cite https://doi.org/10.17882/42182",
        "counts": {
            "floats": int(floats["wmo"].nunique()),
            "sensor_rows": int(len(sensors)),
            "bgc": int((floats.get("data_kind") == "bgc").sum()),
            "core": int((floats.get("data_kind") == "core").sum()),
            "deep": int(floats.get("is_deep").fillna(False).sum())
                    if "is_deep" in floats else 0,
        },
        "data_through": _max_int(floats.get("last_date")),
        "files": {
            "floats.parquet": {"rows": int(len(floats)),
                               "columns": sorted(floats.columns)},
            "sensors.parquet": {"rows": int(len(sensors)),
                                "columns": sorted(sensors.columns)},
            "floats.json": {"rows": int(len(fj)),
                            "columns": list(fj.columns)},
        },
    }
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)

    print(f"API published to {args.out}: {len(floats)} floats, "
          f"{len(sensors)} sensor rows")


if __name__ == "__main__":
    main()
