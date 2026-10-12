"""The five model families, their Optuna search spaces and the depth-band reference.

All models take X = [UTM_Easting_m, UTM_Northing_m, Top_Elevation_m] and expose
fit(X, y) and predict_proba(X). Everything data-dependent (scaling, class weights) is
computed inside fit, i.e. from the training partition only (brief section 5).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .gw import GWClassifier, _balanced


def suggest(trial, space: dict) -> dict:
    """Draw one parameter set from a search space defined in config/project.toml."""
    out = {}
    for name, s in space.items():
        if s["type"] == "int":
            out[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1), log=s.get("log", False))
        elif s["type"] == "float":
            out[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        elif s["type"] == "categorical":
            out[name] = trial.suggest_categorical(name, s["choices"])
        else:
            raise ValueError(f"unknown parameter type {s['type']!r} for {name}")
    return out


class RFModel(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None, seed=0, n_jobs=1):
        self.params, self.seed, self.n_jobs = params, seed, n_jobs

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier

        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        self.model_ = RandomForestClassifier(**p, class_weight=None if cw == "none" else cw, bootstrap=True,
                                             random_state=self.seed, n_jobs=self.n_jobs).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        # single-threaded prediction: multi-threaded tree averaging sums in arrival order and
        # changes the last floating-point bit between runs
        self.model_.n_jobs = 1
        return self.model_.predict_proba(X)


class XGBModel(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None, seed=0, n_jobs=1):
        self.params, self.seed, self.n_jobs = params, seed, n_jobs

    def fit(self, X, y):
        from xgboost import XGBClassifier

        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        w = _balanced(np.asarray(y)) if cw == "balanced" else None
        self.model_ = XGBClassifier(**p, tree_method="hist", random_state=self.seed, n_jobs=self.n_jobs,
                                    eval_metric="logloss").fit(X, y, sample_weight=w)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class MLPModel(ClassifierMixin, BaseEstimator):
    """Standardised inputs (scaler fitted on the training partition) and an MLP without
    sklearn's internal random-row early-stopping split, which would mix cells of one
    borehole between fitting and validation."""

    def __init__(self, params=None, seed=0, n_jobs=1):
        self.params, self.seed, self.n_jobs = params, seed, n_jobs

    def fit(self, X, y):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        layers = (p.pop("n_units"),) * p.pop("n_layers")
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = MLPClassifier(hidden_layer_sizes=layers, early_stopping=False, random_state=self.seed, **p)
        w = _balanced(np.asarray(y)) if cw == "balanced" else None
        self.model_.fit(self.scaler_.transform(X), y, sample_weight=w)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.scaler_.transform(X))


def build(name: str, params: dict, cfg: dict, seed: int, smoke: bool = False):
    """Instantiate a model family with a parameter set from its search space."""
    n_jobs = cfg["experiment"]["n_jobs"]
    p = dict(params)
    if smoke:  # reduced sizes for the software validation run only
        sm = cfg["smoke"]
        if "n_estimators" in p:
            p["n_estimators"] = min(p["n_estimators"], sm["rf_trees"])
        if "max_iter" in p:
            p["max_iter"] = min(p["max_iter"], sm["mlp_max_iter"])
    if name == "RF":
        return RFModel(p, seed, n_jobs)
    if name == "XGB":
        return XGBModel(p, seed, n_jobs)
    if name == "MLP":
        return MLPModel(p, seed, n_jobs)
    if name in ("GWRF", "GWXGB"):
        g = cfg["gw"]
        base = "rf" if name == "GWRF" else "xgb"
        k, alpha, kernel, cw = p.pop("k_neighbours"), p.pop("alpha"), p.pop("kernel"), p.pop("class_weight", "none")
        if base == "rf":
            n_local = g["local_rf_trees"] if not smoke else min(g["local_rf_trees"], 30)
            n_global = 3 * n_local
        else:
            n_local = n_global = min(p.pop("n_estimators"), g["local_xgb_rounds_max"])
        return GWClassifier(base=base, k_neighbours=k, alpha=alpha, kernel=kernel, learner_params=p,
                            class_weight=cw, local_n_estimators=n_local, global_n_estimators=n_global,
                            min_boreholes=g["min_boreholes"], min_cells_per_class=g["min_cells_per_class"],
                            max_neighbours=g["max_neighbours"], random_state=seed, n_jobs=n_jobs)
    raise ValueError(f"unknown model {name!r}")


class DepthPrior(ClassifierMixin, BaseEstimator):
    """Reference baseline (not one of the five families): P(Clay) = training clay share of
    the cell's depth slice. Uses depth below ground, which is not a model predictor, and no
    horizontal information; it shows how much skill comes from location."""

    def __init__(self, slice_m=1.5, bottom_m=15.0):
        self.slice_m, self.bottom_m = slice_m, bottom_m

    def fit(self, depth_top, y):
        s = np.floor(np.asarray(depth_top) / self.slice_m).astype(int)
        y = np.asarray(y)
        self.overall_ = float(y.mean())
        self.share_ = {int(k): float(y[s == k].mean()) for k in np.unique(s)}
        return self

    def predict_proba(self, depth_top):
        s = np.floor(np.asarray(depth_top) / self.slice_m).astype(int)
        p = np.array([self.share_.get(int(k), self.overall_) for k in s])
        return np.column_stack([1 - p, p])
