# Siem Reap Soil Parameters for Ground Movement (V7 regression)

Predicting the soil parameters that control ground movement when groundwater or soil moisture changes (clay
consolidation, sand settlement, shrink–swell), in the upper 15 m beneath Downtown Siem Reap, from location,
elevation and logged soil type, under an honest spatial validation.

## Dataset

- **Source:** `SiemReap_BH_Master_Downtown_DEM_Region_v1_ArcGIS.csv` (borehole sample records; not in git)
- **Samples:** 4,269 records; 120 Downtown boreholes; 2,078 Downtown records at 0–15 m
- **Inputs (4):** Easting, Northing, top elevation of the sample, logged soil type (Clay = CL, CH; Sand = SC, SM, SP)
- **Targets (10):** SPT-N, water content, LL, PL, void ratio e0, wet unit weight, fines, Su (clay), Eu (clay), E′ (sand)

| Mechanism | Parameters predicted here | Calculated later (notebook 05) |
|---|---|---|
| Clay consolidation when groundwater falls | w, LL, e0, γwet, Su, Eu | Cc (correlations), σ′p (from Su), settlement for water-level drops |
| Sand settlement | SPT-N, E′ | immediate settlement |
| Shrink–swell with seasonal moisture | LL, PL, fines, w | PI, liquidity index, swell potential |

Not modelled: Gs (2.60–2.76, use the median), dry/saturated/submerged unit weight, porosity and saturation
(calculated from w, Gs, e0, γ), c′ and φ′ (not needed for movement). No oedometer data (Cc, Cr, σ′p, Cv) exists.

## Project Structure

```
V7_regression/
├── README.md
├── requirements.txt
├── settings.toml          # every rule and choice (change values here, never in a notebook)
├── utils.py               # data, cleaning, models, cross-validation, scores
├── check_packages.py      # lists every package with OK / MISSING
├── data/raw/              # the borehole files (not in git; see data/raw/README.md)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_model_building.ipynb
│   └── 04_evaluation.ipynb
└── tests/test_utils.py    # checks of the rules on synthetic data
```

## Notebooks

| # | Notebook | What You'll Learn | Run time |
|---|---|---|---|
| 01 | `01_eda` | the raw data before any rule: columns, missing values, data availability per depth band and borehole, distributions, depth profiles, Clay vs Sand, plasticity chart, consistency checks, correlations, data status | < 1 min |
| 02 | `02_data_cleaning` | the rules and what each removes: Downtown, 0–15 m, record ≤ 3 m, **SPT-N ≤ 60 (no 3 SD rule)**, **other parameters: mean ± 3 SD**, Su/Eu clay only, E′ sand only; V3's development and holdout sets | < 1 min |
| 03 | `03_model_building` | RF, XGB, MLP, GWRF, GWXGB + Depth and SoilDepth references; **Test A** (V3's 106 m spatial folds, nested Optuna tuning) and **Test B** (random 80:20 borehole split, comparison only) | pilot a few min; full run about 5–8 h |
| 04 | `04_evaluation` | results table, model choice per parameter (rule fixed before the results), skill vs references, 1:1 plots, skill by distance and depth, Test A vs Test B | < 1 min |
| 05 | *later* | settlement and swell from the predicted parameters | |
| 06 | *later* | 3D map and W–E / S–N cross-sections | |

## Results

Filled in from notebook 04 after the full run on the school PC (never from a quick test run).

## Key Findings

To be written from the outputs, not assumed.

## Tech Stack

Python 3.11+, pandas, numpy, scikit-learn, XGBoost, Optuna, SciPy, matplotlib, seaborn, joblib, tqdm.

## How to Run (school PC, Anaconda PowerShell)

1. Unzip so that you have `E:\DavidAn\V7_regression\settings.toml`.
2. Copy the two CSV files into `E:\DavidAn\V7_regression\data\raw` (see `data/raw/README.md`).
3. Open `settings.toml` in Notepad and check `[v3] folder` = your V3 project folder (the one with `data\folds` and `config\external_holdout.lock.json`).
4. In Anaconda PowerShell:
   ```
   conda activate E:\DavidAn\SiemReapML_VSCode\SiemReapML\.conda-env
   cd E:\DavidAn\V7_regression
   python check_packages.py
   jupyter lab
   ```
5. Run the notebooks in order (kernel **SiemReapML env**, Run All): 01 → 02 → 03 (first with `PILOT = True`, then `PILOT = False`) → 04.

**Test run:** `settings.toml → [run] quick = true` runs a tiny version into `outputs_quick/` (never thesis results).
`python -m pytest -q tests` checks the rules.

## Rules of This Project

- `outputs/<step>/`: tables and figures without borehole codes or locations; `outputs/<step>/private/`: predictions per sample, checkpoints, the borehole map. Not shared, not in git.
- Model choice uses Test A only. Test B is only for comparing with random-split studies.
- V3's 11 external-holdout boreholes are never used for training, tuning or choice.
