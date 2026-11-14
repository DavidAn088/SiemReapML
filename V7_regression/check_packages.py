"""Check that every package V7 needs is installed.  Run in the V7_regression folder:  python check_packages.py"""
import importlib
import sys

PACKAGES = {"numpy": "all", "pandas": "all", "scipy": "GW models, Spearman", "sklearn": "RF, MLP",
            "xgboost": "XGB, GWXGB", "optuna": "tuning", "matplotlib": "figures", "seaborn": "figures",
            "joblib": "saving runs", "tqdm": "progress bars", "ipykernel": "Jupyter kernel", "pytest": "tests (optional)"}
print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
missing = []
for name, use in PACKAGES.items():
    try:
        v = getattr(importlib.import_module(name), "__version__", "?")
        print(f"  OK       {name:<11} {v:<12} {use}")
    except Exception:
        print(f"  MISSING  {name:<11} {'':<12} {use}")
        missing.append("scikit-learn" if name == "sklearn" else name)
if sys.version_info < (3, 11):
    print("  Python 3.11 or newer is needed (tomllib).")
print("\nAll packages found." if not missing else f"\nInstall the missing ones:  pip install {' '.join(missing)}")
