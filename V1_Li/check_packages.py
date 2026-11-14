"""Check that every package V1_Li needs is installed.  Run (Anaconda Prompt, in the V1_Li folder):
    python check_packages.py
Python packages: used by helpers.py and the notebooks.  R packages: used by notebook 02 (blockCV)."""

import importlib
import os
import shutil
import subprocess
import sys

PYTHON = {  # import name: what it is used for
    "numpy": "all", "pandas": "all", "scipy": "GW models, distances", "sklearn": "RF, KNN, SVM, metrics",
    "xgboost": "XGB, GWXGB", "optuna": "tuning (TPE and grid search)", "shap": "notebook 06",
    "joblib": "saving models", "matplotlib": "figures", "geopandas": "areas, blockCV maps",
    "shapely": "areas", "pyproj": "coordinates", "pyogrio": "reading shapefiles", "rasterio": "DEM",
    "xarray": "3D voxel file", "netCDF4": "3D voxel file", "tqdm": "progress bars", "nbformat": "notebooks",
    "ipykernel": "Jupyter kernel", "pytest": "tests (optional)",
}
R_PACKAGES = ["jsonlite", "sf", "terra", "ggplot2", "cowplot", "automap", "Rcpp", "blockCV"]   # remotes: only without the ZIP

print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
missing = []
for name, use in PYTHON.items():
    try:
        v = getattr(importlib.import_module(name), "__version__", "?")
        print(f"  OK       {name:<11} {v:<12} {use}")
    except Exception:
        print(f"  MISSING  {name:<11} {'':<12} {use}")
        missing.append(name)
if sys.version_info < (3, 11):
    print("  Python 3.11 or newer is needed (tomllib).")

rscript = r"E:\DavidAn\R\R-4.6.1\bin\Rscript.exe"
rscript = rscript if os.path.exists(rscript) else shutil.which("Rscript")
print("\nR packages" + (f" ({rscript})" if rscript else ""))
if rscript:
    code = ("for (p in c(%s)) cat(sprintf('  %%-8s %%-11s %%s\\n', if (requireNamespace(p, quietly = TRUE)) 'OK' "
            "else 'MISSING', p, if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else ''))"
            % ", ".join(f"'{p}'" for p in R_PACKAGES))
    print(subprocess.run([rscript, "-e", code], capture_output=True, text=True).stdout, end="")
    print("  (blockCV MISSING is fine before the first run: notebook 02 installs it from the ZIP in data\\raw)")
else:
    print("  Rscript not found: set RSCRIPT in notebook 02")

print("\nAll Python packages found." if not missing else
      f"\nInstall the missing ones in the SiemReapML env:  pip install {' '.join('scikit-learn' if m == 'sklearn' else m for m in missing)}")
