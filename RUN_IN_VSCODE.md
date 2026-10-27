# Running the notebooks in VS Code

The notebooks in `notebooks/` run in order, 01 → 05. Notebook 01 also runs the three R scripts (spatial autocorrelation and the blockCV folds), so you do not need RStudio.

## 1. Install once

| What | Notes |
|---|---|
| [VS Code](https://code.visualstudio.com/) with the **Python** and **Jupyter** extensions | When you open the folder, VS Code offers to install both (listed in `.vscode/extensions.json`) |
| **Python 3.13** | [Miniforge](https://github.com/conda-forge/miniforge) or Anaconda (for the conda route), or python.org |
| **R ≥ 4.3** | [cran.r-project.org](https://cran.r-project.org/) |
| **Rtools** (Windows only) | The version that matches your R (e.g. Rtools44 for R 4.4). blockCV is compiled from the supplied source. On macOS install the Xcode command-line tools instead |

The first run of notebook 01 needs internet to download the R packages (sf, terra, gstat, automap, …).

## 2. Open the project folder

Unzip, then in VS Code use **File → Open Folder…** and choose the `SiemReapML` folder (the one that contains `config/`, `notebooks/` and `data/`).

* **From the ZIP:** your data are already in `data/raw/`.
* **From a git clone:** copy the data files there first, as listed in `data/raw/README.md`.

## 3. Create the Python environment

Open a terminal in VS Code (**Terminal → New Terminal**) and use one of the two options below.

**Conda**
```
conda env create -f environment.yml
conda activate siemreap-ml
```

**venv (Windows)**
```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Check the install:
```
python -m pytest tests -q
```
It should report `9 passed`.

## 4. Select the kernel

1. Open `notebooks/01_preprocessing.ipynb`.
2. Click **Select Kernel** (top right) → **Python Environments** → `siemreap-ml` (or `.venv`).

The other notebooks then offer the same kernel.

## 5. Run

### First: a smoke run (software check, not results)

1. In the **first cell of each notebook**, remove the `#` from this line:
   ```python
   # os.environ["SIEMREAP_MODE"] = "smoke"
   ```
2. Run notebooks 01 → 05 with **Run All**.

The smoke run uses one outer fold, the 100 m blocks, two Optuna trials and a 100 m 3D grid. Its outputs go to `outputs/smoke/`; never report them.

### Then: the experiment

1. Put the `#` back on that line in every notebook.
2. **Restart** each kernel (toolbar → **Restart**), because a running kernel keeps the smoke setting.
3. Run notebooks 01 → 05 again.

| Notebook | What to do |
|---|---|
| `01_preprocessing` | **Run All.** The last cell runs `R/00_install_packages.R`, `R/01_spatial_autocorrelation.R` and `R/02_blockcv_folds.R`, showing R's messages as they come. If it says *Rscript not found*, set `RSCRIPT = r"C:\Program Files\R\R-4.x.x\bin\Rscript.exe"` in that cell and run the cell again |
| `02_spatial_folds_review` | Review the folds and the block-size rule. The planned primary block is 100 m (`primary_block_m` in `config/project.toml`). If you decide otherwise, change that value **before notebook 03**, run notebook 02 again so it records the decision, and write down the reason |
| `03_nested_cv_models` | The long run: roughly 1.5–2.5 days on one 4-core PC for everything (it depends on the computer); run `RUN_BLOCKS = [100]` first to get the primary results sooner, or split the block sizes across two computers. Set `n_jobs` in `config/project.toml` to your number of cores. You can run part of it first, e.g. `RUN_BLOCKS = [100]`. It saves a checkpoint after every outer fold, so if VS Code closes or the PC sleeps, run the notebook again and it continues from where it stopped |
| `04_results_analysis` | Metrics, figures and the pre-declared model selection |
| `05_final_model_external_3d` | Final model, the one-time external evaluation, and the 3D output for ArcGIS Pro |
| `06_journal_figures` | Publication figures in the style of Valavi et al. (2019) |
| `07_explainability` | Explainable AI: held-out permutation importance (single inputs and location vs elevation) and SHAP |
| `08_thesis_figures` | The student's figure layout: validation layout with purge circles, model comparison, confusion matrix + mean ROC, SHAP and permutation panels of the final model (reads outputs only) |
| `09_new_external_test` | Put `External_BHs_SR_Elev.csv` in `data\raw\`, then **Run All once**: the frozen model and the depth-only reference on the 10 new boreholes. Do not re-run notebook 05 before it |
| `10_map_model_120bh` | Refit of the final model on all 120 Downtown boreholes, for the 3D map only (50 m grid; not evaluated). Output: `outputs/10_map_model/restricted/siemreap_3d_points_120bh_50m.csv` |

To keep notebook 03 running without the editor open, run it from a terminal (with the environment activated):
```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_nested_cv_models.ipynb
```

Also stop Windows from sleeping during the long run (**Settings → System → Power**).

## Where the outputs are

| Folder | Contents |
|---|---|
| `outputs/<stage>/aggregate/` | Metrics, summary tables and figures without borehole locations |
| `outputs/<stage>/restricted/` | Borehole-level tables, maps, models and the 3D files |
| `outputs/05_final/restricted/siemreap_3d_points.csv` | ArcGIS Pro: *XY Table To Point*, with X, Y and Z, EPSG:32648 |
| `outputs/05_final/restricted/siemreap_3d_voxels.nc` | ArcGIS Pro: *Add Voxel Layer* in a local scene |

## If something stops

| Message | Fix |
|---|---|
| `blockCV installation failed` | Install Rtools for your R version, restart VS Code, and run the last cell of notebook 01 again |
| `Checksum of … is …, expected …` | The file in `data/raw/` differs from the one the project was built on. Put the original back |
| `The external holdout recomputed … differs from the locked selection` | The external holdout must not change. Do not edit `config/external_holdout.lock.json`; check that the CSV is the original |
| `ModuleNotFoundError` | The wrong kernel is selected; choose `siemreap-ml` / `.venv` (step 4) |

## Keep the data private

The `data/` and `outputs/*/restricted/` folders hold borehole records and locations. `.gitignore` keeps them out of git; do not upload them elsewhere.

