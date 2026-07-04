# BGC-Argo Explorer

A local, PI-facing dashboard for BGC-Argo floats: look up a sensor **serial
number** (e.g. an SBE41 CTD), see the float it belongs to, every other sensor on
that float, the measurands on board, where the float is, and interactive plots —
with QC / data-mode handling built in. All served from a bounded (~5 GB) local
copy of the most-recent BGC data.

> **License:** code is [MIT](LICENSE); the bundled Argo metadata and any Argo
> data this tool retrieves are © the International Argo Program under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
> [Acknowledgements, data source & license](#acknowledgements-data-source--license).

## Install
```bash
pip install -r requirements.txt
```

## Run order (three steps)
```bash
# 1. Download: meta.nc (small) + most-recent data files up to a size budget.
python sync_bgc_subset.py --root ./argo_local --budget-gb 5          # BGC (default)
#    quick first test: add --limit-floats 50

# 2. Build the serial/sensor crosswalk from the meta files.
python build_crosswalk.py --root ./argo_local

# 3. Launch the UI.
streamlit run argo_dashboard.py -- --root ./argo_local
```
To refresh later, re-run steps 1–2 (downloads resume; existing files are skipped).

### Core (physical T/S) and Deep Argo floats
BGC is the default. To also pull **core** floats — the physical T/S array, which is
how you reach **Deep Argo / SBE61 6000 m** floats (they carry no BGC sensors) — use
`--dataset core` (or `both`). Core reads the core-profile index (~20k floats) and
downloads `<wmo>_prof.nc`; all the physical plots (profiles, sections, T–S, σ₀, MLD,
trends) work on them, BGC-only plots stay empty.
```bash
python sync_bgc_subset.py --root ./argo_local --dataset core --limit-floats 50
# a specific float (ignores the size budget):
python sync_bgc_subset.py --root ./argo_local --dataset core --wmo 4902911
python build_crosswalk.py --root ./argo_local          # then rebuild the crosswalk
```
The index is cached locally after the first fetch; add `--refresh-index` to re-pull.
Once core floats are ingested, search **sensor model `SBE61`** in the sidebar to find
the 6000 m Deep floats.

### Recommended for a public deployment: metadata for everything + fetch on demand
Rather than mirror the whole ~150–250 GB array, pull **only the metadata for every
float** (small: ~12 MB of parquets, a one-time ~1 GB meta download) and let the app
**fetch each float's data file from the GDAC on demand** the first time it's opened
(cached thereafter). Search/dossier work for the entire array; only the plots trigger
a fetch.
```bash
# one-time metadata pull for the whole array (BGC + core), no data files
python sync_bgc_subset.py --root ./argo_local --dataset both --meta-only --limit-floats 0
python build_crosswalk.py --root ./argo_local
streamlit run argo_dashboard.py -- --root ./argo_local
```
The meta download is resumable (existing files skipped). Set `ARGO_GDAC` to point the
on-demand fetch at a mirror (e.g. the AWS Open Data S3 mirror) instead of Ifremer.
Re-run the two build steps periodically to pick up newly deployed floats.

## How the three requirements are met
- **Serial-number lookup** — no such index exists in Argo, so `build_crosswalk.py`
  parses `SENSOR_SERIAL_NO` / `SENSOR_MODEL` / `FLOAT_SERIAL_NO` out of every
  `meta.nc` into `sensors.parquet`. The sidebar searches it.
- **Float dossier** — Float SN, WMO, DAC, platform type, and **all sensors** on
  the float come straight from that crosswalk.
- **Measurands + location + plots** — read from each float's `Sprof.nc`; map and
  depth profiles render with Plotly.

## Honest limitations (read these)
- **Not yet tested against live data.** It was written offline against the
  ArgoPy 1.4.0 cheatsheet and the documented Argo v3.1 file layout. Spots that may
  need a one-line fix are marked `VERIFY` in the code. If a plot is empty or a path
  404s, that's the first place to look.
- **Sprof, not raw counts.** The 5 GB subset holds *synthetic* profiles
  (QC'd/adjusted, science-ready). Intermediate/raw sensor signals live in the
  **B-files**, which are not in this subset. The dashboard can fetch a single
  float's data on demand; extend that to B-files if you need raw signals for
  diagnostics.
- **Serial matching is fuzzy.** Serial formats vary by DAC (padding, prefixes,
  `SBE41` vs `SBE41CP`). Search is case-insensitive substring; confirm a hit by
  cross-checking model + maker.
- **The dataset is mutable.** Delayed-mode QC and reprocessing rewrite old files.
  Re-sync regularly; for reproducible figures, pin an `ArgoDOI()` monthly snapshot.
- **Defaults encode QC guidance, not a fix.** The UI prefers `*_ADJUSTED` and
  filters QC flags {1,2,5,8} by default and shows the data mode — it makes the data
  state legible; it does not correct sensor drift.

## Files
| file | purpose |
|---|---|
| `sync_bgc_subset.py` | download bounded local BGC/core mini-GDAC + manifest |
| `build_crosswalk.py` | meta.nc → `sensors.parquet`, `floats.parquet` |
| `argo_dashboard.py` | Streamlit UI |

## Acknowledgements, data source & license
Thank you to the **International Argo Program** and the ~30 nations — their
governments, agencies, engineers, and scientists — who fund, build, deploy, quality-
control, and *freely* share these floats with the world. 🌊

> These data were collected and made freely available by the International Argo
> Program and the national programs that contribute to it
> (https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program is part of the
> Global Ocean Observing System (GOOS).

**Cite:** Argo (2026). *Argo float data and metadata from Global Data Assembly Centre
(Argo GDAC).* SEANOE. https://doi.org/10.17882/42182

With gratitude to the national programs & Data Assembly Centres, including Australia
(CSIRO/BOM/IMOS), Canada (DFO/MEDS), China (SIO/CSIO), France (Ifremer/Coriolis,
CNES), Germany (BSH/GEOMAR), India (INCOIS/MoES), Italy (OGS), Japan (JAMSTEC/JMA),
South Korea (KMA/KIOST), the UK (Met Office/BODC), the USA (NOAA), **Euro-Argo ERIC**,
and every other nation contributing to Argo.

**License:** Argo data are freely available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
This tool applies QC filtering and computes derived quantities (TEOS-10 density, MLD,
AOU, absolute/conservative properties, trends) that are **not official Argo
products**. Data are retrieved from the Argo GDAC.

*Built with the help of [Claude](https://claude.ai) (Anthropic). An independent,
community tool — not affiliated with or endorsed by the Argo Program.*
