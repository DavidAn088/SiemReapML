# Siem Reap 3D Clay/Sand classification: V4

A from-scratch rebuild in a simple style: 9 notebooks that read from top to bottom, one helper file
(`helpers.py`), two short R scripts and **one settings file** (`settings.toml`).

## What V4 does

| | V4 |
|---|---|
| Data | **Downtown + 5 km ring** (145 boreholes, 143 with labels, 3,706 cells of 0.5 m). Switch: `settings.toml → [area] area = "downtown"` for Downtown only |
| Target | Clay (CL, CH) = 1, Sand (SC, SM, SP) = 0; 0-15 m |
| Inputs | Easting, Northing, top elevation of the cell (no depth, no scaling) |
| Gap filling | gaps ≤ 3 m between two cells of the same class; an unclassified SS sample does not stop the fill (student's rule). Filled cells are used for training **and** testing; every score is also given on measured cells only; the weight of filled cells is tuned |
| Block size | **blockCV only**: `cv_spatial_autocor()$range` with one column per 3 m slice (blockCV takes the median itself). blockCV 4.0-2 from GitHub, commit `51e6903` |
| Folds | `cv_spatial()`, k = 5 square blocks; sensitivity 2× and 3×; purge 1× and 2× |
| Models | RF, XGB, MLP, GWRF, GWXGB + depth-only reference; nested CV with Optuna (same search spaces as V3) |
| Final model | pre-declared 1-SE rule (same as V3), tuned and trained on all boreholes, then frozen |
| External test | only the 10 new boreholes, scored once (notebook 07) |
| Map | 50 m grid over the same area, with reliability zones |

## How to run (school PC)

1. Copy the contents of the V3 project's `data\raw` folder into `V4\data\raw` (see `data/raw/README.md`).
2. Open Jupyter in the `V4` folder with the same Python environment as V3.
3. Run the notebooks in order: open each one and choose **Run All**.

| Notebook | Measured time* | Notes |
|---|---|---|
| `01_data` | < 1 min | |
| `02_blockcv` | 1-5 min (+ 5-10 min the first time, to install blockCV) | set `RSCRIPT` / `RTOOLS` at the top; needs internet the first time |
| `03_spatial_cv` | **about 7 h on a fast 4-core PC; plan 0.5-1.5 days on the school PC** | saves after every fold; if it stops, run it again and it continues |
| `04_results` | < 1 min | chooses the final model |
| `05_final_model` | 5-30 min | runs only once (the model is frozen) |
| `06_xai` | 20-60 min | |
| `07_new_external_test` | < 1 min | scores the 10 new boreholes **once** |
| `08_map` | 10-60 min | 50 m grid; about 4 million cells for Downtown + 5 km |
| `09_thesis_figures` | < 1 min | |

\*Notebook 03 time comes from timing full-size Optuna trials (per outer fold: RF 7.6 min, XGB 0.6, MLP 24.8, GWRF 18.8,
GWXGB 2.5) × 80 outer folds. Real times depend on the PC and on the hyperparameters Optuna tries.

## Test run

`settings.toml → [run] quick = true` runs everything in a few minutes (1 fold, 2 trials, coarse map) and writes to
`outputs_quick/`. It only checks that the code works: **quick results are never thesis results.**

## Folders

* `outputs/<step>/`: tables and figures without borehole locations
* `outputs/<step>/private/`: anything that shows boreholes (maps, predictions, models, 3D files). Not shared, not in git.

## To disclose in the thesis

* The 10 new boreholes were already scored once with the V3 model; the decision to add the 5 km ring came after that.
  V4 never trains, tunes or chooses anything with them.
* The SS gap rule was decided by the student after seeing the V3 test; in V4 it is used for training and testing alike.
