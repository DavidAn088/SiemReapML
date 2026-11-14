"""Shared code for the Siem Reap 3D Clay/Sand classification notebooks.

Modules
-------
config     project configuration, paths, run mode (full / smoke) and seeds
data       raw CSV loading, borehole table, 0.5 m discretisation, external holdout
geo        polygons, DEM sampling and the 3D prediction grid geometry
folds      import and validation of the R blockCV fold files, inner folds, distances
models     the five model families and their Optuna search spaces
gw         geographically weighted random forest / XGBoost classifiers
metrics    evaluation metrics, summaries and the pre-declared selection rule
nested_cv  nested spatial cross-validation runner with checkpointing
plots      figures
"""

__version__ = "0.1.0"
