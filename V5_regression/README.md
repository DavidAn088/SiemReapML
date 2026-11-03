# Siem Reap soil parameters: V5 regression

Spatial regression of continuous soil parameters in the upper 15 m beneath Downtown Siem Reap,
under the **same spatial validation as the Clay/Sand classification (V3 / V3-clean)**: the same
Downtown boreholes, V3's blockCV folds (106 m main; 212 and 318 m sensitivity) and V3's 11
hash-locked external-holdout boreholes, read from the V3 folder.

## What V5 does

| | |
|---|---|
| Targets | `SPT_N`, `Water_Content_pct`, `LL_pct`, `PI_pct` (change in `settings.toml → [cv] targets` and `[targets.*]`) |
| Samples | one row = one sample record with a value; record ≤ 3 m long, middle depth 0-15 m, value in the target's min/max. No gap filling (point values) |
| Inputs | Easting, Northing, mid elevation of the sample (= ground elevation − middle depth), as in V3 |
| Transform | SPT-N is learned as log(1 + N); every score is on the original scale |
| Models | RF, XGB (Optuna, 30 trials, inner out-of-fold RMSE) and a **depth-only reference** (training mean per 1 m slice) |
| Scores | RMSE, MAE, R², bias, Spearman per outer fold; skill vs depth-only = 1 − RMSE / RMSE_depth |
| Model choice | per target, on the 106 m block: lowest mean RMSE; within one SE, lowest fold SD, then lowest MAE. Fixed before the results |
| External test | V3's 11 holdout boreholes, scored once (notebook 04, locked) |

## How to run (school PC)

1. Copy V3's `data\raw` borehole CSV into `V5_regression\data\raw` (see `data/raw/README.md`).
2. Check `settings.toml → [v3] folder` points to the V3 project folder.
3. Open Jupyter in the `V5_regression` folder (same Python environment as V3) and run in order:

| Notebook | Time | |
|---|---|---|
| `01_data` | < 1 min | samples per target, depth profiles, values by USCS group |
| `02_spatial_cv` | **pilot a few minutes**, then roughly 1-2 h per target (estimate) | `PILOT = True` first; then `PILOT = False`. Saves after every fold and continues after a crash. `RUN_TARGETS = ["SPT_N"]` runs one target |
| `03_results` | < 1 min | scores, skill vs depth, model choice, figures |
| `04_final_external` | 5-15 min per target | final models (frozen) and the external test, **once** |

The run times are estimates from the V3 timings and the smaller sample count; the pilot gives the real figure.

## Test run

`settings.toml → [run] quick = true` runs everything in a few minutes (1 fold, 2 trials, small models) and writes
to `outputs_quick/`. It only checks that the code works: **quick results are never thesis results.**
`python -m pytest -q tests` checks the rules on synthetic data.

## To disclose in the thesis

* V5 reuses V3's folds and holdout, which were made for the Clay/Sand cells. A borehole with values of a
  parameter but no Clay/Sand cell has no V3 fold and is left out (notebook 01 reports how many).
* SPT refusals and "non-plastic" entries were blanked in the CSV export, so they are missing, not zero; the
  SPT-N model therefore does not see refusal values.
* The 11 external-holdout boreholes were already scored once with the V3 classifier; V5 never uses them for
  training, tuning or model choice.

## Folders

* `outputs/<step>/`: tables and figures without borehole codes or locations
* `outputs/<step>/private/`: predictions per sample, checkpoints and models. Not shared, not in git.
