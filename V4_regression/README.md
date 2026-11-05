# Siem Reap soil parameters: V4 regression

Layer-average soil parameters in the upper 15 m beneath Downtown Siem Reap from **Easting, Northing, mid
elevation and the predicted P(Clay)** of the Clay/Sand classifier, validated with V3's spatial CV, then mapped
in 3D on the classifier's grid.

| | |
|---|---|
| Targets | directly measured only: water content, LL, PL, wet unit weight, Su (clay samples), SPT-N. Derived: PI = LL − PL (notebook 05). Void ratio, dry unit weight etc. are left to the settlement stage |
| Rows | the mean of a borehole's samples in each **1.5 m layer** (`[layers] layer_m`); sample rules: record ≤ 3 m, middle depth 0-15 m, value in range (SPT-N ≤ 60), values beyond mean ± 3 SD dropped |
| Soil input | **P(Clay)** from the V3-clean XGB classifier at each sample's 0.5 m cell, layer mean. Development boreholes: the **out-of-fold** model (fold k model for a borehole in fold k), so the input is as reliable as on the map. Holdout boreholes: the frozen final classifier, which never saw them. Map: the grid's `p_clay`. The logged USCS symbol is not an input |
| Models | RF, XGB, MLP, GWRF, GWXGB (Optuna, nested) and the depth-only reference |
| Validation | V3's Downtown boreholes and 106 m blockCV folds (whole boreholes tested); V3's 11 hash-locked holdout boreholes scored once (notebook 04) |
| Model choice | per target: lowest mean RMSE; within one SE, lowest fold SD, then lowest MAE (the depth reference competes) |
| Map | classifier grid (50 m columns) → 1.5 m layers; each target predicted with its final model, plus its CV RMSE for the depth band (uncertainty) and the grid's reliability zone |

## How to run (school PC)

1. Put the `V4_regression` folder **inside the V3clean folder** (next to V3clean's `settings.toml`). V4 reads
   V3clean's out-of-fold classifier models (`outputs/03_spatial_cv/private/models/`) and final model.
2. Copy the borehole CSV into `V4_regression\data\raw`. Check two paths in `settings.toml`: `[v3] folder`
   (the V3 project: folds and holdout) and `[map] grid_csv` (the classifier grid, set to
   `D:/ML Thesis-Downtown/Soil Classification Downtown/siemreap_3d_points_120bh_50m.csv`).
3. Open Jupyter in `V4_regression` and run in order:

| Notebook | Time | |
|---|---|---|
| `01_data` | ~1 min | layers with P(Clay); prints how well P(Clay) matches the logged class (AUC) |
| `02_spatial_cv` | pilot ~1 min, then 2-3 h (estimate) | `PILOT = True` first, then `False`. Saves after every fold; `RUN_TARGETS` runs one target at a time |
| `03_results` | < 1 min | scores, skill vs depth, model choice, scores by depth band |
| `04_final_external` | 5-15 min per target | final models (frozen) and the external test, **once**: run only after reviewing 03 |
| `05_map` | a few min | 3D layer map with uncertainty (private CSV) and profiles by depth |

`settings.toml → [run] quick = true` runs a tiny check version into `outputs_quick/` (never thesis results).
`python -m pytest -q tests` checks the rules on synthetic data.

## To disclose in the thesis

* The map grid comes from the 120-borehole classifier (trained on all boreholes, including the holdout). It is
  used only for the map, never for training or scoring.
* Beyond about 100 m from a borehole (the indicator variogram range) the map is close to the depth trend; the
  reliability zone is kept in the map file.

## Folders

* `outputs/<step>/`: tables and figures without borehole codes or locations
* `outputs/<step>/private/`: predictions per layer, checkpoints, models and the map. Not shared, not in git.
