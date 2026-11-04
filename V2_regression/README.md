# Siem Reap soil parameters: V2 regression

Soil parameters in the upper 15 m beneath Downtown Siem Reap from **4 inputs**: Easting, Northing, mid elevation
and the **soil class logged in the borehole** (Clay = CL, CH; Sand = SC, SM, SP), as in Pintusorachai et al. (2024)
and Khamjan et al. (2026). Validation is V3's spatial CV: the same Downtown boreholes, V3's 106 m blockCV folds and
V3's 11 hash-locked external-holdout boreholes, read from the V3 folder.

| | |
|---|---|
| Targets | water content, void ratio, LL, PI, wet unit weight, specific gravity, SPT-N, Su (clay samples) |
| Samples | one row = one sample record with a value and a Clay/Sand symbol; record ≤ 3 m long, middle depth 0-15 m, value in the target's min/max (SPT-N ≤ 60), then values beyond mean ± 3 SD dropped |
| Inputs | Easting, Northing, mid elevation (= ground elevation − middle depth), logged soil class |
| Models | RF, XGB (Optuna, 30 trials, inner out-of-fold RMSE) and a depth-only reference (training mean per 1 m slice) |
| Scores | RMSE, MAE, R², bias, Spearman per outer fold and pooled; skill vs depth-only = 1 − RMSE / RMSE_depth |
| Model choice | per target: lowest mean RMSE; within one SE, lowest fold SD, then lowest MAE |
| External test | V3's 11 holdout boreholes, scored once (notebook 04, locked) |

**The soil class is the logged one, also at test time.** The scores therefore assume the soil class at the
predicted point is known. On a map it would come from the Clay/Sand classifier, whose errors are not included
here (V5 measures that case).

## How to run (school PC)

1. Put the `V2_regression` folder anywhere (e.g. next to `V5_regression`).
2. Copy the borehole CSV into `V2_regression\data\raw`. `[v3] folder` in `settings.toml` is the V3 project folder
   (folds and holdout are read from there).
3. Open Jupyter in `V2_regression` and run in order:

| Notebook | Time | |
|---|---|---|
| `01_data` | < 1 min | samples per target, depth profiles, values by soil class |
| `02_spatial_cv` | pilot ~1 min, then about 1 h | `PILOT = True` first, then `PILOT = False`. Saves after every fold |
| `03_results` | < 1 min | scores, skill vs depth, model choice, figures |
| `04_final_external` | 5-15 min per target | final models (frozen) and the external test, **once** |

`settings.toml → [run] quick = true` runs a tiny check version into `outputs_quick/` (never thesis results).
`python -m pytest -q tests` checks the rules on synthetic data.

## Folders

* `outputs/<step>/`: tables and figures without borehole codes or locations
* `outputs/<step>/private/`: predictions per sample, checkpoints and models. Not shared, not in git.
