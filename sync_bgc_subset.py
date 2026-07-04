#!/usr/bin/env python3
"""
sync_bgc_subset.py
==================
Build a LOCAL, GDAC-structured subset of Argo data on your machine, bounded to a
size budget (default ~5 GB), preferring the most-recent data.

Datasets (choose with --dataset)
--------------------------------
  bgc  (default) : synthetic-profile index -> <wmo>_Sprof.nc   (BGC-sensor floats)
  core           : core-profile index      -> <wmo>_prof.nc    (physical T/S; this
                   is how you reach Deep Argo / SBE61 6000 m floats, which are core)
  both           : pull both, merged into one local tree + index

What it downloads
-----------------
1. meta.nc for EVERY selected float  -> small; powers the serial/sensor crosswalk.
2. The aggregated per-float data file (<wmo>_Sprof.nc for bgc, <wmo>_prof.nc for
   core) for the MOST-RECENTLY-ACTIVE floats, in date order, until the budget.

Honest limitations
-------------------
* Sprof (bgc) is the SYNTHETIC profile: QC'd/adjusted, science-ready, no raw
  intermediate signals (those live in B-files, not fetched here).
* core prof.nc is the physical (T/S) profile — no BGC parameters.
* The GDAC is mutable: delayed-mode QC rewrites old files. Re-run to refresh; pin
  a monthly DOI snapshot for reproducible figures.

Tested? The bgc path is exercised; the core path was added later — VERIFY the two
GDAC constants below against the live server if a path 404s.
"""

import argparse
import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

# --- GDAC constants (VERIFY if paths 404) ------------------------------------
GDAC = "https://data-argo.ifremer.fr"                       # Ifremer HTTPS GDAC root
SPROF_INDEX = f"{GDAC}/argo_synthetic-profile_index.txt"    # BGC Sprof-able floats
CORE_INDEX = f"{GDAC}/ar_index_global_prof.txt.gz"          # 58 MB gz (312 MB plain)
# File paths on the GDAC live under /dac/<dac>/<wmo>/...
# -----------------------------------------------------------------------------

DATASETS = {
    "bgc":  {"index": SPROF_INDEX, "suffix": "_Sprof.nc", "kind": "bgc"},
    "core": {"index": CORE_INDEX,  "suffix": "_prof.nc",  "kind": "core"},
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "argo-dashboard/0.2"})


def _get_text(url, retries=3):
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def _download_file(url, dest, retries=3):
    """Stream a file to dest. Returns bytes written (or existing size), -1 on fail."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return os.path.getsize(dest)          # resume: skip existing
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for i in range(retries):
        try:
            with SESSION.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                n = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        n += len(chunk)
                os.replace(tmp, dest)
                return n
        except Exception:
            if i == retries - 1:
                return -1
            time.sleep(2 * (i + 1))


def fetch_index(url, root, refresh=False):
    """Download (and cache) an Argo index; return it as a DataFrame with dac/wmo."""
    local = os.path.join(root, "_" + os.path.basename(url))
    if refresh and os.path.exists(local):
        os.remove(local)
    if not (os.path.exists(local) and os.path.getsize(local) > 0):
        print(f"  fetching index {os.path.basename(url)} ...")
        if _download_file(url, local) < 0:
            raise RuntimeError(f"failed to download index {url}")
    # comment='#' skips the header banner; compression inferred from extension (.gz)
    df = pd.read_csv(local, comment="#", compression="infer",
                     dtype={"profiler_type": "str"}, low_memory=False)
    parts = df["file"].str.split("/", expand=True)
    df["dac"] = parts[0]
    df["wmo"] = parts[1]
    return df


def per_float_table(idx):
    """One row per float: most-recent date/position + params + profiler type."""
    idx = idx.copy()
    idx["date"] = pd.to_numeric(idx["date"], errors="coerce")
    has_params = "parameters" in idx.columns
    has_prof = "profiler_type" in idx.columns
    agg = dict(last_date=("date", "max"),
               last_lat=("latitude", "last"),
               last_lon=("longitude", "last"),
               n_profiles=("file", "count"))
    if has_params:
        agg["parameters"] = ("parameters", "last")
    if has_prof:
        agg["profiler_type"] = ("profiler_type", "last")
    per = (idx.sort_values("date")
              .groupby(["dac", "wmo"])
              .agg(**agg)
              .reset_index()
              .sort_values("last_date", ascending=False))
    if not has_params:
        per["parameters"] = ""
    if not has_prof:
        per["profiler_type"] = ""
    return per


def process_dataset(key, root, args):
    """Sync one dataset (bgc|core). Returns (included_rows, selected_per_float)."""
    cfg = DATASETS[key]
    print(f"\n=== dataset: {key}  ({cfg['suffix']}) ===")
    idx = fetch_index(cfg["index"], root, args.refresh_index)
    per = per_float_table(idx)
    print(f"floats in {key} index: {len(per)}")

    # optional profiler-type filter (WMO instrument-type codes; e.g. Deep floats)
    if args.profiler_type:
        want = {p.strip() for p in args.profiler_type.split(",") if p.strip()}
        ptype = per["profiler_type"].astype(str).str.split(".").str[0]
        per = per[ptype.isin(want)]
        print(f"  after profiler-type {sorted(want)}: {len(per)} floats")

    targeted = bool(args.wmo.strip())
    if targeted:
        wanted = {w.strip() for w in args.wmo.split(",") if w.strip()}
        per = per[per["wmo"].astype(str).isin(wanted)]
        missing = wanted - set(per["wmo"].astype(str))
        if missing:
            print(f"  note: not found in {key} index (skipped): {sorted(missing)}")
    elif args.limit_floats > 0:
        per = per.head(args.limit_floats)

    print(f"selected: {len(per)} floats")
    if per.empty:
        return [], per

    # ---- 1. meta.nc for every selected float ----
    if not args.skip_meta:
        jobs = [(f"{GDAC}/dac/{r.dac}/{r.wmo}/{r.wmo}_meta.nc",
                 os.path.join(root, "dac", r.dac, r.wmo, f"{r.wmo}_meta.nc"))
                for r in per.itertuples()]
        print(f"downloading {len(jobs)} meta.nc files ...")
        ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_download_file, u, d): u for u, d in jobs}
            for f in tqdm(as_completed(futs), total=len(futs), unit="file"):
                if f.result() > 0:
                    ok += 1
        print(f"  meta.nc downloaded/present: {ok}/{len(jobs)}")

    # ---- 2. data files (skip entirely in metadata-only mode) ----
    if args.meta_only:
        print(f"  meta-only: {len(per)} floats indexed for on-demand fetch "
              f"(no {cfg['suffix']} downloaded)")
        return [], per

    #        (targeted --wmo pulls, or --budget-gb<=0, ignore the budget)
    unbounded = targeted or args.budget_gb <= 0
    budget = float("inf") if unbounded else args.budget_gb * (1 << 30)
    used = 0
    included = []
    print(f"downloading {cfg['suffix']} up to "
          f"{'no cap' if unbounded else f'~{args.budget_gb} GB'} ...")
    for r in tqdm(list(per.itertuples()), unit="float"):
        if used >= budget:
            break
        rel = f"dac/{r.dac}/{r.wmo}/{r.wmo}{cfg['suffix']}"
        n = _download_file(f"{GDAC}/{rel}", os.path.join(root, rel))
        if n > 0:
            used += n
            included.append({
                "wmo": r.wmo, "dac": r.dac,
                "last_date": int(getattr(r, "last_date", 0) or 0),
                "last_lat": r.last_lat, "last_lon": r.last_lon,
                "parameters": getattr(r, "parameters", ""),
                "profiler_type": getattr(r, "profiler_type", ""),
                "n_profiles": int(r.n_profiles),
                "data_bytes": n, "data_path": rel,
                "data_kind": cfg["kind"], "data_suffix": cfg["suffix"],
            })
    print(f"  {key}: {used / (1 << 30):.2f} GB across {len(included)} floats")
    return included, per


def main():
    ap = argparse.ArgumentParser(description="Sync a local Argo subset (BGC or core).")
    ap.add_argument("--root", default="./argo_local",
                    help="Local GDAC root (a 'dac/...' tree is created inside).")
    ap.add_argument("--dataset", choices=["bgc", "core", "both"], default="bgc",
                    help="bgc=synthetic Sprof, core=physical prof (incl. Deep Argo).")
    ap.add_argument("--budget-gb", type=float, default=5.0,
                    help="Approx size cap for data files (<=0 means no cap). "
                         "meta files are extra & small.")
    ap.add_argument("--workers", type=int, default=8, help="Parallel downloads.")
    ap.add_argument("--limit-floats", type=int, default=0,
                    help="For a quick test: only the N most-recent floats (0=all).")
    ap.add_argument("--wmo", default="",
                    help="Comma-separated WMO(s) to pull specifically. Overrides "
                         "--limit-floats and ignores the size budget.")
    ap.add_argument("--profiler-type", default="",
                    help="Comma-separated WMO instrument/profiler-type code(s) to "
                         "keep (e.g. Deep-float codes). Applies to the chosen index.")
    ap.add_argument("--skip-meta", action="store_true",
                    help="Don't (re)download meta files.")
    ap.add_argument("--meta-only", action="store_true",
                    help="Download only meta.nc + build the index — no Sprof/prof "
                         "data. Floats are fetched from the GDAC on demand in the "
                         "app. Use with --dataset both --limit-floats 0 for the "
                         "full-array metadata pull.")
    ap.add_argument("--refresh-index", action="store_true",
                    help="Re-download the index file(s) instead of using the cache.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)
    print(f"Local GDAC root: {root}")

    keys = ["bgc", "core"] if args.dataset == "both" else [args.dataset]
    all_included, all_per = [], []
    for k in keys:
        inc, per = process_dataset(k, root, args)
        all_included += inc
        if len(per):
            all_per.append(per.assign(data_suffix=DATASETS[k]["suffix"],
                                      data_kind=DATASETS[k]["kind"]))

    # ---- merge per-float index parquet across datasets & prior runs ----
    if all_per:
        newidx = pd.concat(all_per, ignore_index=True)
        idx_path = os.path.join(root, "bgc_float_index.parquet")
        if os.path.exists(idx_path):
            old = pd.read_parquet(idx_path)
            newidx = pd.concat([old, newidx], ignore_index=True)
        # prefer BGC (Sprof, richer) over core when a float is in both indexes;
        # stable sort keeps old-before-new order so re-runs refresh same-kind rows
        newidx["_pref"] = (newidx.get("data_kind") == "bgc").astype(int)
        newidx = (newidx.sort_values("_pref", kind="stable")
                        .drop_duplicates(subset=["dac", "wmo"], keep="last")
                        .drop(columns="_pref"))
        # deterministic order so the weekly refresh only commits real changes
        newidx = newidx.sort_values(["dac", "wmo"]).reset_index(drop=True)
        newidx.to_parquet(idx_path)

    # ---- merge manifest across prior runs (dedup by dac+wmo) ----
    man_path = os.path.join(root, "manifest.json")
    floats = all_included
    if os.path.exists(man_path):
        try:
            old = json.load(open(man_path)).get("floats", [])
            seen = {(f["dac"], f["wmo"]) for f in all_included}
            floats = all_included + [f for f in old
                                     if (f.get("dac"), f.get("wmo")) not in seen]
        except Exception:
            pass
    manifest = {
        "synced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gdac": GDAC,
        "datasets": keys,
        "budget_gb": args.budget_gb,
        "n_floats_with_local_data": len(floats),
        "floats": floats,
    }
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nDone. Local data for {len(floats)} floats "
          f"({', '.join(keys)}). Manifest -> {man_path}")
    print("Next: python build_crosswalk.py --root", root)


if __name__ == "__main__":
    sys.exit(main())
