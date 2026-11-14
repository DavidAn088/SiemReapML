# Siem Reap soil parameters: V3 regression

Two honest ways to predict soil parameters better than single-sample spatial regression (V2), both under
V3's spatial CV: the same Downtown boreholes, V3's 106 m blockCV folds (whole boreholes are tested), nested
Optuna, a depth-only reference, and V3's 11 hash-locked external-holdout boreholes.

## A. Layer models (notebooks 02-04)

| | |
|---|---|
| Target | the **mean of a borehole's samples in each 1.5 m layer** (`[layers] layer_m`). A single lab test carries noise no model can predict; settlement calculations use layer values |
| Inputs | Easting, Northing, mid elevation of the layer, **Clay share of the layer** from the logged USCS (Clay = CL, CH; Sand = SC, SM, SP) |
| Targets | water content, void ratio, LL, PI, wet unit weight, specific gravity, SPT-N, Su (clay samples) |
| Models | RF, XGB, MLP, GWRF, GWXGB and the depth-only reference |

## B. Property models (notebook 05)

| | |
|---|---|
| Idea | transformation models (Phoon & Kulhawy 1999): a parameter from **other parameters measured on the same sample** |
| Inputs | water content (`[property] inputs`), depth and the logged soil class. **No location** |
| Targets | void ratio, wet unit weight, specific gravity, LL, PI, Su |
| Models | RF, XGB, **Linear** (least squares: the classical correlation) and the depth-only reference |
| Use | fills in parameters that were not tested where water content was |

Sample rules for both: record ≤ 3 m long, middle depth 0-15 m, value in the target's min/max (SPT-N ≤ 60),
values beyond mean ± 3 SD dropped, Clay/Sand symbol required.

**The soil class is the logged one, also at test time** (as in Pintusorachai et al. 2024 and Khamjan et al. 2026,
and as in the student's earlier notebooks).

## How to run (school PC)

1. Put `V3_regression` anywhere and copy the borehole CSV into `V3_regression\data\raw`. `[v3] folder` in
   `settings.toml` is the V3 project folder (folds and holdout).
2. Open Jupyter in `V3_regression` and run in order:

| Notebook | Time | |
|---|---|---|
| `01_data` | < 1 min | layer tables and property tables, counts |
| `05_property_models` | 10-30 min (estimate) | B: CV, scores, model choice. Saves after every fold |
| `02_spatial_cv` | pilot ~1 min, then 2-4 h (estimate) | A: `PILOT = True` first, then `False`. `RUN_TARGETS` runs one target at a time |
| `03_results` | < 1 min | A: scores, skill vs depth, model choice |
| `04_final_external` | 5-15 min per target | A: final models and the external test, **once** (only after reviewing 03) |

`settings.toml → [run] quick = true` runs a tiny check version into `outputs_quick/` (never thesis results).
`python -m pytest -q tests` checks the rules on synthetic data.

## Folders

* `outputs/<step>/`: tables and figures without borehole codes or locations
* `outputs/<step>/private/`: predictions per sample or layer, checkpoints and models. Not shared, not in git.
