# Siem Reap 3D Clay/Sand classification: V1_Li

The ideas of **Li, Zhang & He (2026)**, *Algorithms* 19(2):128 (https://doi.org/10.3390/a19020128),
applied to the Siem Reap boreholes, with **our own data, our own tuning and spatial block cross-validation**.
A new project written from scratch in the simple style: 9 notebooks that read from top to bottom, one helper file
(`helpers.py`), two short R scripts and **one settings file** (`settings.toml`). Nothing is reused from V3.

## The question

> Do the VA-HDW features of Li et al. (2026), and their GBARNN model, improve Clay/Sand prediction
> when the test boreholes are spatially separated from the training boreholes?

Li et al. tested with a random split of points, so test points sit next to training points of the same borehole.
Here every test borehole is in a different spatial block than the training boreholes (blockCV), so the scores show
how well a model predicts **away from** the boreholes, which is what a 3D map needs.

## What V1_Li does

| | V1_Li |
|---|---|
| Data | **Downtown + 5 km ring** (145 boreholes, 143 with labels, 3,706 cells of 0.5 m). Switch: `settings.toml → [area] area = "downtown"` |
| Target | Clay (CL, CH) = 1, Sand (SC, SM, SP) = 0; 0-15 m |
| Gap filling | gaps ≤ 3 m between two cells of the same class; an unclassified SS sample does not stop the fill (SS rule). Filled cells are used for training **and** testing; every score is also given on measured cells only; the weight of filled cells is tuned |
| Inputs, set 1 | **XYZ**: Easting, Northing, top elevation of the cell (the baseline used by most ML soil studies) |
| Inputs, set 2 | **XYZ + VA-HDW** (Li et al.): for every reference borehole, the distance **d**, the class at the cell's elevation **Φ** and the influence coefficient **S** |
| Models | RF, XGB, GWRF, GWXGB on **both** sets; **GBARNN** (Li et al.) on XYZ + VA-HDW, with X, Y, Z also fed to its head; Li et al.'s comparison models **KNN** and **SVM** on min-max normalised XYZ, as in the paper |
| Tuning | Optuna (TPE) for RF, XGB, GWRF, GWXGB and GBARNN (layer sizes, learning rate, epochs, batch size, weight decay). KNN and SVM: **grid search over the paper's own grid** (Table 6: KNN n_neighbors 1-15, leaf_size 20/30/40, p 1/2; SVM rbf, C 0.1/1/10/100, gamma scale/auto). The decision threshold and (except KNN, which has no sample weights) the weight of filled cells are tuned too |
| Block size | **106 m** (the thesis block size, fixed in `settings.toml → [blockcv] block_m`). blockCV's own `cv_spatial_autocor()$range` (one column per 3 m slice) is computed and reported for information. blockCV 4.0-2 from GitHub, commit `51e6903` |
| Folds | `cv_spatial()`, k = 5 square blocks, all boreholes in the CV; sensitivity 2× and 3× the block size; purge 1× and 2× (XGB, both sets) |
| VA-HDW gain | rule decided before the results: per model, the fold-by-fold Macro-F1 difference (XYZ + VA-HDW − XYZ); **helps** only if the mean is above one SE |
| Final model | rule decided before the results (recall rule, 1-SE rule; all models on both sets are candidates); tuned and trained on all boreholes, then frozen. A comparison model (best on the other feature set) is frozen too |
| External test | only the **10 new boreholes**, scored once (notebook 07) |
| Map | 50 m grid over the same area, with reliability zones |

### VA-HDW and GBARNN: how they are adapted (state these in the thesis)

* **No leakage.** The reference boreholes are always the training boreholes of that fit: in the outer CV the test
  boreholes are never references, and every inner-CV model builds its features from its own inner-training boreholes.
  A cell's own borehole is masked (d = 20 km, Φ = 0.5, S = 0). A test in `tests/` checks that changing the labels of
  the test boreholes never changes their predictions.
* **Layers.** Li et al. have ordered strata 1-7. Our "layer" is a run of one class (Clay or Sand) in a borehole, and Φ is
  Clay (1) or Sand (0). Above the first or below the last labelled cell of a borehole, Φ is its first or last class
  and S = 0.
* **GBARNN output.** As in the paper's equations (one weight per borehole, softmax), P(Clay) = Σ weight × the
  borehole's clay share. The model sees depth through which boreholes get weight and through X, Y, Z in the head.
* **Not included: GeoPDNN.** Li et al. also compare with GeoPDNN, a semi-supervised model from another paper
  (their ref. [19]) with its own settings and no published code; it is not re-implemented here.
* **No published code.** GBARNN is written in NumPy (no PyTorch needed); its gradients are checked against numerical
  derivatives in `tests/`. Input scaling and the Leaky ReLU slope (0.01) are standard choices, not from the paper.

## How to run (school PC)

1. Copy your raw data files into `V1_Li\data\raw` (see `data/raw/README.md`). Put the **blockCV ZIP** from GitHub
   there too (R has no internet on the school PC).
2. Open Jupyter in the `V1_Li` folder with the same Python environment as before (kernel **SiemReapML env**).
   No new package is needed. To check: `python check_packages.py` lists every Python and R package with OK / MISSING.
3. Run the notebooks in order: open each one and choose **Run All**.

| Notebook | Time | Notes |
|---|---|---|
| `01_data` | < 1 min | |
| `02_blockcv` | 1-5 min | set `RSCRIPT` / `RTOOLS` at the top (school PC paths are already there) |
| `03_spatial_cv` | **about 26 h on a fast 4-core PC; plan 2-3 days on the school PC** | saves after every fold; if it stops, run it again and it continues. `RUN_ONLY = ["_x1_purge0"]` gives the main results first |
| `04_results` | < 1 min | VA-HDW gain test, final model |
| `05_final_model` | 10-60 min | runs only once (the models are frozen) |
| `06_xai` | 10-40 min | |
| `07_new_external_test` | < 1 min | scores the 10 new boreholes **once** |
| `08_map` | 20-90 min | 50 m grid; about 4 million cells |
| `09_figures` | < 1 min | |

Notebook 03 time comes from timing full-size Optuna trials (minutes per outer fold, XYZ / XYZ + VA-HDW):
RF 8 / 18, XGB 1 / 7, GWRF 23 / 59, GWXGB 2 / 5, **GBARNN 151** (VA-HDW only); KNN 1 and SVM 2 (XYZ only, full grid: 90 and 24 trials).
Main runs: about 23 h (GBARNN alone about 13 h); block-size and purge runs (XGB): about 2.5 h.
Real times depend on the PC and on the hyperparameters Optuna tries.
`RUN_ONLY = ["_x1_purge0"]` runs the main results first; `RUN_ONLY = ["GBARNN"]` runs only GBARNN (for example overnight).

## Test run

`settings.toml → [run] quick = true` runs everything in about 15 minutes (1 fold, 2 trials, coarse map) and writes to
`outputs_quick/`. It only checks that the code works: **quick results are never thesis results.**
`python -m pytest -q tests` runs the checks of the main rules (20 tests).

## Folders

* `outputs/<step>/`: tables and figures without borehole locations
* `outputs/<step>/private/`: anything that shows boreholes (maps, predictions, models, 3D files). Not shared, not in git.

## To disclose in the thesis

* The 10 new boreholes were already scored with earlier models of this study; V1_Li never trains, tunes or chooses
  anything with them.
* The SS gap rule was decided by the student before V1_Li; it is used for training and testing alike.
* Scores are **not** comparable with the 92.8 % of Li et al.: their random split puts test points next to training
  points of the same borehole; spatial block CV does not.
