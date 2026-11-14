# Siem Reap soil parameters: V6 regression

Soil parameters in the upper 15 m beneath Downtown Siem Reap from **4 inputs**: Easting, Northing, sample top
elevation and the **soil type logged in the borehole**, scored with **two tests**:

* **Test A (honest):** V3's spatial CV: 106 m blockCV folds, nested tuning, and V3's 11 hash-locked external-holdout boreholes.
* **Test B (comparison):** one random 80:20 borehole split, as in the Phnom Penh unit-weight notebook. Easier than A,
  because test boreholes usually have a training borehole nearby. Only for comparing R²; model choice uses Test A.

| | |
|---|---|
| Targets | water content, LL, PL, wet unit weight, Su (clay samples), SPT-N. PI = LL − PL is computed, not modelled |
| Samples | one row = one sample record (no resampling): record ≤ 3 m long, middle depth 0-15 m, value in the target's min/max (SPT-N ≤ 60), then values beyond mean ± 3 SD dropped |
| Inputs | Easting, Northing, elevation (ground − top depth of the record; `"mid"` possible), logged soil type |
| Soil type | `[soil_input] mode = "clay_sand"` (default): Clay = CL, CH (1), Sand = SC, SM, SP (0). `"uscs6"`: CH, CL, ML/MH, SC, SM, SP; trees use the group number, MLP one 0/1 column per group. **LL and PL always use Clay/Sand**: USCS CL/CH/ML is defined from LL and PI, so the 6 groups would hand the model part of the answer |
| Models | RF, XGB, MLP, GWRF, GWXGB (Optuna: 30 trials, 20 for the GW models, inner out-of-fold RMSE) |
| References | Depth (training mean per 1 m slice) and SoilDepth (training mean per soil group and 1 m slice). A model that does not beat SoilDepth adds nothing beyond soil type and depth |
| Scores | RMSE, MAE, R², bias, Spearman; skill vs Depth and vs SoilDepth = 1 − RMSE / RMSE_reference |
| Model choice | per target, Test A: lowest mean RMSE; within one SE, lowest fold SD, then lowest MAE. Never picked on test R² |
| External test | V3's 11 holdout boreholes, scored once (notebook 04, locked) |

**The soil type is the logged one, also at test time.** The scores therefore show how well a parameter can be
predicted when the soil type at that point is known. A map has no logged soil type (it would come from a classifier
with its own errors; V4 measures that case).

## How to run (school PC)

1. Put the `V6_regression` folder anywhere (e.g. `E:\DavidAn\V6_regression`).
2. Copy the borehole CSV into `V6_regression\data\raw`. `[v3] folder` in `settings.toml` is the V3 project folder
   (folds and holdout are read from there).
3. Choose the soil input in `settings.toml → [soil_input] mode` (`clay_sand` or `uscs6`). Changing it later means
   a new run: move the `outputs` and `data\processed` folders away first.
4. Open Jupyter in `V6_regression` and run in order:

| Notebook | Time | |
|---|---|---|
| `01_data` | < 1 min | samples per target, all logged USCS symbols, depth profiles, values by soil group |
| `02_spatial_cv` | pilot ~1 min, then about 2-4 h (estimate) | Test A. `PILOT = True` first, then `PILOT = False`. Saves after every fold; `RUN_TARGETS` runs one target at a time |
| `02b_random_split` | pilot ~1 min, then under 1 h (estimate) | Test B, same pilot / full steps |
| `03_results` | < 1 min | scores, skill vs both references, model choice, Test A next to Test B |
| `04_final_external` | 5-15 min per target | final models (frozen) and the external test, **once** |

`settings.toml → [run] quick = true` runs a tiny check version into `outputs_quick/` (never thesis results).
`python -m pytest -q tests` checks the rules on synthetic data.

## Folders

* `outputs/<step>/`: tables and figures without borehole codes or locations
* `outputs/<step>/private/`: predictions per sample, checkpoints and models. Not shared, not in git.
