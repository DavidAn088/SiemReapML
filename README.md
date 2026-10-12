# SiemReapML

3D Clay/Sand classification of the upper 15 m beneath Downtown Siem Reap, Cambodia, using machine learning with nested spatial cross-validation. Spatial folds come from the original R package **blockCV**; modelling runs in Python. This is a master's thesis project in Geotechnical Engineering at Chulalongkorn University. The research brief is [`docs/research_brief.md`](docs/research_brief.md), and the source review and design is [`docs/01_source_review_and_design.md`](docs/01_source_review_and_design.md).

## Status

| Stage | State |
|---|---|
| 1–2 Source review, data audit | Done (`scripts/01_audit_raw_data.py`, `scripts/02_verify_sources.py`) |
| 3 Experimental design | Documented; implemented with the brief's settings in `config/project.toml` |
| 4 R spatial autocorrelation + blockCV folds | Scripts ready (`R/`); executed in the smoke test |
| 5 Nested CV, 5 models × 2 training extents | Notebooks ready; **smoke-tested only** |
| 6–7 Results, final model, external test, 3D output | Notebooks ready; **smoke-tested only** |

**No experimental result exists yet.** The smoke test checks that the software runs end to end with reduced budgets. Its outputs live in the git-ignored `outputs/smoke/` and must not be reported.

## Workflow

```
notebooks/01_preprocessing.ipynb            0.5 m cells, external holdout (locked), inputs for R
R/00_install_packages.R                     once: CRAN packages + blockCV 4.0-1 from the supplied repository
R/01_spatial_autocorrelation.R              indicator variograms by depth slice, Moran's I, vertical variogram
R/02_blockcv_folds.R                        blockCV folds 100/200/300 m, cv_distance, fallback inner folds
notebooks/02_spatial_folds_review.ipynb     fold integrity, real test-to-train distances, inner-fold adequacy,
                                            purge effect, primary-block decision (recorded before modelling)
notebooks/03_nested_cv_models.ipynb         RF, XGB, MLP, GWRF, GWXGB (+ depth-prior reference) x scenarios A/B
                                            x block sizes, Optuna, checkpointed, determinism check
notebooks/04_results_analysis.ipynb         all metrics, folds/pooled/borehole/depth/distance, robustness,
                                            pre-declared model selection
notebooks/05_final_model_external_3d.ipynb  final model, frozen; external evaluation once; ArcGIS Pro 3D output
```

### Setup

**Step-by-step for VS Code: [`RUN_IN_VSCODE.md`](RUN_IN_VSCODE.md).** The last cell of notebook 01 runs the three R scripts, so RStudio is optional.

1. **Python:** `conda env create -f environment.yml && conda activate siemreap-ml`, or `pip install -r requirements.txt`.
2. **R** (≥ 4.3; on Windows also Rtools for the same R version): notebook 01 runs `R/00_install_packages.R` for you, or open `SiemReapML.Rproj` in RStudio and run `source("R/00_install_packages.R")`.
3. **Data:** place the files listed in `data/raw/README.md`, including the unzipped `blockCV-master/`.

### Run

* **Software validation first.** Set `SIEMREAP_MODE=smoke` before starting Jupyter (or uncomment the line in each notebook's first cell). This uses one outer fold, the 100 m blocks, two Optuna trials and a 100 m 3D grid.
* **The experiment.** Unset the variable (or set `SIEMREAP_MODE=full`), restart the kernels, and run the notebooks in order. The R scripts run from the last cell of notebook 01 (or in RStudio).
* **Resuming and splitting.** Notebook 03 resumes from its checkpoints. `RUN_BLOCKS`, `RUN_MODELS` and `RUN_SCENARIOS` let you run a subset first, e.g. the primary block size.

### Settings (from the brief, all in `config/project.toml`)

| | |
|---|---|
| Target | CL/CH → Clay = 1; SC/SM/SP → Sand = 0; ML, CL-ML and missing symbols excluded, never converted |
| Cells | 0–15 m below ground, 0.5 m, labelled only from overlapping sample intervals (≥ 50 % coverage) |
| Predictors | `UTM_Easting_m`, `UTM_Northing_m`, `Top_Elevation_m = Ground_Elevation_m − cell top depth` |
| Scenarios | A: Downtown development boreholes. B: A + 5 km ring boreholes (training only). Identical Downtown test folds |
| Outer CV | blockCV `cv_spatial`, square blocks 100 m (planned primary), 200 m and 300 m; k = 5; random balanced; seeded |
| Inner CV | the four remaining outer folds (adequacy-checked; R fallback folds otherwise) |
| Tuning | Optuna TPE; 50 trials (RF, XGB, MLP) and 30 (GWRF, GWXGB) per outer fold; inner-OOF Macro-F1; threshold from inner OOF |
| External holdout | 11 Downtown boreholes in two groups: generalisation (isolated sites ≥ 500 m) and interpolation. Chosen without labels; hash locked in `config/external_holdout.lock.json` |
| Selection | pre-declared rule on the primary block size (recall constraints, then Macro-F1, then 1-SE rule) |
| 3D output | 25 m columns × 30 layers inside the Downtown polygon; ground from the same bilinear DEM sampler as the boreholes; CSV points and a netCDF voxel file |

### Compute (measured in the smoke test; 4-core machine)

See `docs/01_source_review_and_design.md` section 8 for the per-model timings and the estimate for the full run.

## Data confidentiality

This repository is public. Raw borehole records, borehole identifiers and coordinates are never committed: `data/` (except the README), `outputs/*/restricted/`, checkpoints, models and smoke outputs are git-ignored. Notebooks are committed **without outputs**; run them locally to see results.

## Tests

`python -m pytest tests -q` runs unit tests on synthetic data. They check the discretisation rule, deterministic models, borehole integrity of splits and purge, isolation of external boreholes, the GW fallback, and the equivalence of lazy and eager GW fitting.
