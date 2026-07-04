# 🌊 Argo Float Data Explorer

Search and visualize the **entire global Argo float array** — BGC, core, and
Deep Argo (SBE61 6000 m) — by sensor **serial number**, model, WMO, or region.
Look up a CTD serial like `SBE41` or `SBE61`, find the float it belongs to, every
other sensor on board, where it's been, and what it's measured — then plot
profiles, depth–time sections, T–S diagrams, and multi-year trends. Float data is
streamed **live from the Argo GDAC** on demand, so the whole ~20,000-float array is
searchable from a repo just a few megabytes in size.

**▶ Live app: https://argofloat.streamlit.app** &nbsp;·&nbsp;
**About / landing: https://vladsimontov.github.io/ArgoFloatData/**

> **License:** code is [MIT](LICENSE); Argo data (the bundled metadata index and any
> profiles the app retrieves) is © the International Argo Program under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
> [Acknowledgements](#acknowledgements-data-source--license).

---

## What it does

- **Find any float** — search **20,321 floats** by CTD/sensor serial number, sensor
  model (SBE41, SBE41CP, **SBE61**, optodes, ECO, SUNA…), WMO number, or region.
  Click a match to open it; a serial search adds a highlighted **matched on** column
  showing exactly which sensor + serial (or float serial) triggered the hit. Argo has
  no native serial→float index, so this builds one from every float's `meta.nc`.
- **Float dossier** — a structured **Identity / Deployment / Project** panel: float
  serial, WMO, DAC, platform type/maker, **all sensors**, measurands on board,
  **days deployed**, last position, an approximate **region** (climate band + ocean
  basin), and which data file backs it (BGC vs core).
- **Interactive plots**, grouped into tabs (Overview · Trajectory · Profile & Trend · T–S):
  - **Trajectory map** — the float's drift track on an interactive **basemap**
    (pydeck + Carto tiles), with launch and latest positions marked.
  - **Vertical profiles** — real-time or `*_ADJUSTED`, QC-flag aware, colored by date.
  - **Depth–time section** (Hovmöller) with a mixed-layer-depth overlay.
  - **Temperature–Salinity diagram** with σ₀ density contours.
  - **Time series at a chosen pressure** — nearest-level, colored by season, with a
    **Sen's-slope + Mann–Kendall** trend test and optional deseasonalizing.
- **TEOS-10 derived fields** (via `gsw`): potential density σ₀, conservative &
  potential temperature, absolute salinity, apparent oxygen utilization, and mixed
  layer depth — all computed on the fly.
- **Download** any parameter as CSV, or the full float as NetCDF.
- Every plot is stamped with the float's **WMO + latest-profile date** so a chart is
  never mistaken for another float or a stale view.

## How it works

The app stores only a **compact metadata index** (a few MB of Parquet) covering the
whole array, so search and the dossier are instant. When you open a float, its full
profile file (`<wmo>_Sprof.nc` for BGC, `<wmo>_prof.nc` for core/deep) is **fetched
live from the Argo GDAC** and cached — the ~150–250 GB of bulk data never has to be
hosted. A weekly [GitHub Action](.github/workflows/refresh-metadata.yml) refreshes
the index so newly deployed floats appear automatically.

```
meta.nc (all floats)  ──build_crosswalk──▶  sensors/floats.parquet  ──▶  search + dossier
                                                                          │  (on float open)
                                            Argo GDAC  ──fetch on demand──▶  profiles + plots
```

## Coverage: BGC, Core & Deep Argo

| kind | file | what | count |
|---|---|---|---|
| **BGC** | `Sprof.nc` | biogeochemical (O₂, chl-a, nitrate, pH, irradiance…) synthetic profiles | ~2,900 |
| **Core** | `prof.nc` | physical temperature/salinity floats | ~17,400 |
| **Deep** | `prof.nc` | Deep Argo (4000–6000 m); SBE61 6000 m = Deep SOLO / APEX / Xuanwu | ~580 (≈350 SBE61) |

All physical plots work on every float; BGC-only parameters simply stay empty for
core/deep floats.

## Run your own instance

The repo ships the metadata index for the whole array, so it runs immediately:

```bash
pip install -r requirements.txt
streamlit run argo_dashboard.py -- --root ./argo_local
```

Search works for all ~20,000 floats out of the box; each float's data is fetched from
the GDAC the first time you open it.

### Rebuild or scope the index yourself (optional)

The pipeline is `sync → build_crosswalk → run`:

```bash
# Full array, metadata only (~1 GB one-time meta download, resumable) — recommended
python sync_bgc_subset.py --root ./argo_local --dataset both --meta-only --limit-floats 0
python build_crosswalk.py  --root ./argo_local

# …or a small scoped set with data downloaded (immediately plottable, no fetch):
python sync_bgc_subset.py --root ./argo_local --dataset bgc  --limit-floats 20
python sync_bgc_subset.py --root ./argo_local --dataset core --profiler-type 849,862,874,882 --limit-floats 20  # SBE61 deep
python build_crosswalk.py  --root ./argo_local
```

`--dataset` = `bgc` | `core` | `both`. `--meta-only` skips the bulky data (fetched on
demand). The GDAC index is cached; `--refresh-index` re-pulls it. `ARGO_GDAC` points
the on-demand fetch at a mirror (e.g. the AWS Open Data S3 mirror) instead of Ifremer.

## Deploy

Deployed free on **Streamlit Community Cloud** from this repo (main file
`argo_dashboard.py`); the [`docs/`](docs/) landing page is served by GitHub Pages. No
secrets required — optional env vars: `ARGO_GDAC` (data mirror) and `ARGO_ISSUES_URL`
(bug-report link). The weekly workflow keeps the index fresh and Streamlit redeploys
on each push.

## Data API

A free, no-key HTTP data API is published under
[`/api/`](https://vladsimontov.github.io/ArgoFloatData/api/) — query the whole array
by **sensor serial number**, model, WMO, region, or type with DuckDB or pandas (the
serial/sensor lookup is something the GDAC, ERDDAP, and Argovis don't offer). Docs:
**https://vladsimontov.github.io/ArgoFloatData/api.html**

```sql
-- every 6000 m Deep float (SBE61 CTD), straight over HTTP
SELECT DISTINCT wmo
FROM 'https://vladsimontov.github.io/ArgoFloatData/api/sensors.parquet'
WHERE sensor_model = 'SBE61';
```

Files: `api/floats.parquet`, `api/sensors.parquet`, `api/floats.json` (each row
carries its GDAC `data_url`), `api/index.json` (manifest). Regenerated weekly by
`build_api.py`. Profile data itself is fetched from the GDAC — this API indexes it,
it doesn't rehost it.

## Good to know

- **The GDAC is mutable** — delayed-mode QC and reprocessing rewrite old files. The
  index refreshes weekly; for reproducible figures, pin a monthly
  [DOI snapshot](https://doi.org/10.17882/42182).
- **Synthetic/physical profiles, not raw counts.** Sprof/prof are QC'd, science-ready
  profiles; intermediate raw sensor signals live in the B-files (not used here).
- **Serial matching is fuzzy** — formats vary by DAC (padding, prefixes, `SBE41` vs
  `SBE41CP`). Search is case-insensitive substring; confirm a hit with model + maker.
- **Defaults encode QC guidance, not a fix.** The UI prefers `*_ADJUSTED`, keeps QC
  flags {1,2,5,8}, and shows the data mode (R/A/D) — it makes the data state legible,
  it doesn't correct sensor drift. Derived quantities are **not official Argo
  products**.

## Files

| path | purpose |
|---|---|
| `argo_dashboard.py` | the Streamlit app |
| `sync_bgc_subset.py` | download meta.nc (and optionally data) for BGC/core floats |
| `build_crosswalk.py` | parse meta.nc → `sensors.parquet`, `floats.parquet` |
| `build_api.py` | publish the crosswalk as the `docs/api/` data API |
| `argo_local/*.parquet` | the committed metadata index (search + dossier) |
| `docs/` | GitHub Pages landing page + `api.html` docs + `api/` data files |
| `.github/workflows/refresh-metadata.yml` | weekly index + API auto-refresh |

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
