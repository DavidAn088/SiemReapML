# Siem Reap 3D Clay/Sand classification: V3-clean

**V3, written simply, with the same results.** Same data rules, folds, seeds, models, tuning,
model choice, final model, external tests and map as V3. Only the code is new: 10 notebooks that
read from top to bottom, one helper file (`helpers.py`) and one settings file (`settings.toml`).

Notebook 10 proves it: it compares every result with V3's saved results and prints **IDENTICAL**
or **DIFFERENT** for each one.

## What is the same as V3
| | |
|---|---|
| Data | Downtown, 0-15 m, 0.5 m cells; Clay (CL, CH) = 1, Sand (SC, SM, SP) = 0; gaps ≤ 3 m filled (V3 rule) |
| Inputs | Easting, Northing, top elevation (no depth, no scaling) |
| Folds | V3's blockCV folds (106 m; 212 and 318 m for sensitivity), read from the V3 folder, not re-made |
| External holdout | V3's 11 boreholes, checked against V3's lock |
| Models and tuning | RF, XGB, MLP, GWRF, GWXGB; nested spatial CV with Optuna; V3's search spaces and seeds |
| Model choice | V3's 1-SE rule |
| Map | 120-borehole map model, 50 m grid, reliability zones |

**Left out:** the depth-only reference (V3's saved results still contain it).

## How to run (school PC)
1. Copy V3's `data\raw` files into `V3clean\data\raw` (see `data/raw/README.md`).
2. Check `settings.toml → [v3] folder` points to your V3 project folder.
3. Open Jupyter in the `V3clean` folder (same Python environment as V3) and run the notebooks in order:

| Notebook | Time | |
|---|---|---|
| `01_data` | < 1 min | cells, checked identical to V3 |
| `02_folds` | seconds | reads V3's folds |
| `03_spatial_cv` | **pilot 5 min, then about 7-15 h** | first run with `PILOT = True` (XGB, fold 1, compared with V3). If it says **IDENTICAL**, set `PILOT = False` and run again. Saves after every fold |
| `04_results` | < 1 min | scores and V3's model choice |
| `05_final_model` | 5-30 min | refits into this folder; checks against V3's frozen model (never changed) |
| `06_xai` | 20-60 min | permutation importance, SHAP |
| `07_external_tests` | ~1 min | V3 holdout and the 10 new boreholes (with and without the SS rule) |
| `08_map` | 10-30 min | 3D map for ArcGIS Pro (CSV + netCDF voxels) |
| `09_figures` | 1-2 min | thesis and publication figures |
| `10_compare_with_v3` | 1-2 min | **the proof**: every result vs V3 |

## Figures (notebook 09)
PDF (vector) + PNG (600 dpi); widths 90 mm (one column) or 190 mm (two columns); 7-8 pt text; panel
labels (a), (b), (c); one colour per model throughout. Borehole names are hidden (BH-A, BH-B, BH-C).
Figures that show borehole locations (Figs. 1, 3, 11) are saved in `private/`.
In Fig. 3b the 106 m blocks are too small to see at the scale of Downtown; the points carry the fold colours.

## Why "identical" is possible, and what could break it
The new code does the same calculations in the same order with the same seeds. Two details
matter and are handled: cells and predictions are saved to CSV and read back exactly as V3 did
(reading a CSV can change the last digit of a number), and the same Python packages must be
installed as when V3 ran. If a package was updated since, notebook 03's pilot and notebook 10
will show it.

## Test run
`settings.toml → [run] quick = true` runs everything in a few minutes (V3's "smoke" settings) and
compares with V3's smoke outputs. Quick results are never thesis results.
