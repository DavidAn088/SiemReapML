"""The model families, their Optuna search spaces and the depth-band reference.

All models take X = [UTM_Easting_m, UTM_Northing_m, Top_Elevation_m] and expose
fit(X, y) and predict_proba(X). Everything data-dependent (scaling, class weights) is
computed inside fit, i.e. from the training partition only (brief section 5).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .gw import GWClassifier, _balanced, expand_proba


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
    def __init__(self, params=None, seed=0, n_jobs=1, n_classes=2):
        self.params, self.seed, self.n_jobs, self.n_classes = params, seed, n_jobs, n_classes

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestClassifier

        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        self.model_ = RandomForestClassifier(**p, class_weight=None if cw == "none" else cw, bootstrap=True,
                                             random_state=self.seed, n_jobs=self.n_jobs).fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        # single-threaded prediction: multi-threaded tree averaging sums in arrival order and
        # changes the last floating-point bit between runs
        self.model_.n_jobs = 1
        return expand_proba(self.model_.predict_proba(X), self.model_.classes_, self.n_classes)


class XGBModel(ClassifierMixin, BaseEstimator):
    def __init__(self, params=None, seed=0, n_jobs=1, n_classes=2):
        self.params, self.seed, self.n_jobs, self.n_classes = params, seed, n_jobs, n_classes

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier

        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        y = np.asarray(y)
        w = _balanced(y) if cw == "balanced" else None
        if sample_weight is not None:
            w = np.asarray(sample_weight, float) * (w if w is not None else 1.0)
        # XGBoost needs labels 0..m-1: encode the classes present in this training set
        self.present_, yy = np.unique(y, return_inverse=True)
        metric = "logloss" if len(self.present_) <= 2 else "mlogloss"
        self.model_ = XGBClassifier(**p, tree_method="hist", random_state=self.seed, n_jobs=self.n_jobs,
                                    eval_metric=metric).fit(X, yy, sample_weight=w)
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict_proba(self, X):
        return expand_proba(self.model_.predict_proba(X), self.present_, self.n_classes)


class MLPModel(ClassifierMixin, BaseEstimator):
    """Standardised inputs (scaler fitted on the training partition) and an MLP without
    sklearn's internal random-row early-stopping split, which would mix cells of one
    borehole between fitting and validation."""

    def __init__(self, params=None, seed=0, n_jobs=1, n_classes=2):
        self.params, self.seed, self.n_jobs, self.n_classes = params, seed, n_jobs, n_classes

    def fit(self, X, y, sample_weight=None):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        layers = (p.pop("n_units"),) * p.pop("n_layers")
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = MLPClassifier(hidden_layer_sizes=layers, early_stopping=False, random_state=self.seed, **p)
        w = _balanced(np.asarray(y)) if cw == "balanced" else None
        if sample_weight is not None:
            w = np.asarray(sample_weight, float) * (w if w is not None else 1.0)
        self.model_.fit(self.scaler_.transform(X), y, sample_weight=w)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return expand_proba(self.model_.predict_proba(self.scaler_.transform(X)), self.model_.classes_, self.n_classes)


class GPModel(ClassifierMixin, BaseEstimator):
    """Gaussian process classifier (Laplace approximation; Rasmussen & Williams 2006, ch. 3),
    the machine-learning counterpart of indicator kriging. The anisotropic Matern kernel has
    separate correlation lengths for Easting, Northing and elevation; they and the amplitude
    are fitted by maximising the marginal likelihood on the training partition, so the model
    has no Optuna search ([gp] in config/project.toml; an optional [search_space.GP] may
    override its keys). Prediction is chunked to bound memory on the 3D grid."""

    def __init__(self, params=None, seed=0, n_jobs=1, settings=None):
        self.params, self.seed, self.n_jobs, self.settings = params, seed, n_jobs, settings

    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        s = {**self.settings, **(self.params or {})}
        X = np.asarray(X, dtype=float)
        self.center_ = X.mean(axis=0)
        h, v = float(s["horizontal_length_m"]), float(s["vertical_length_m"])
        hb, vb = tuple(s["horizontal_bounds_m"]), tuple(s["vertical_bounds_m"])
        kernel = ConstantKernel(float(s["amplitude"]), tuple(s["amplitude_bounds"])) * Matern(
            length_scale=[h, h, v], length_scale_bounds=[hb, hb, vb], nu=float(s["nu"]))
        self.model_ = GaussianProcessClassifier(kernel, n_restarts_optimizer=0, random_state=self.seed)
        self.model_.fit(X - self.center_, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float) - self.center_
        n = int(self.settings["predict_chunk_rows"])
        return np.vstack([self.model_.predict_proba(X[i:i + n]) for i in range(0, len(X), n)])


def build(name: str, params: dict, cfg: dict, seed: int, smoke: bool = False, n_classes: int = 2):
    """Instantiate a model family with a parameter set from its search space. Every model
    returns probabilities for all `n_classes` classes, also when a training set lacks one."""
    n_jobs = cfg["experiment"]["n_jobs"]
    p = dict(params)
    if smoke:  # reduced sizes for the software validation run only
        sm = cfg["smoke"]
        if "n_estimators" in p:
            p["n_estimators"] = min(p["n_estimators"], sm["rf_trees"])
        if "max_iter" in p:
            p["max_iter"] = min(p["max_iter"], sm["mlp_max_iter"])
    if name == "RF":
        return RFModel(p, seed, n_jobs, n_classes)
    if name == "XGB":
        return XGBModel(p, seed, n_jobs, n_classes)
    if name == "MLP":
        return MLPModel(p, seed, n_jobs, n_classes)
    if name == "GP":
        return GPModel(p, seed, n_jobs, cfg["gp"])
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
                            max_neighbours=g["max_neighbours"], random_state=seed, n_jobs=n_jobs, n_classes=n_classes)
    raise ValueError(f"unknown model {name!r}")


class DepthPrior(ClassifierMixin, BaseEstimator):
    """Reference baseline (not one of the five families): class probabilities = training class
    shares of the sample's depth slice (binary: the clay share). Uses depth below ground, which
    is not a model predictor, and no horizontal information; it shows how much skill comes from
    location."""

    def __init__(self, slice_m=1.5, bottom_m=15.0, n_classes=2):
        self.slice_m, self.bottom_m, self.n_classes = slice_m, bottom_m, n_classes

    def fit(self, depth_top, y, sample_weight=None):
        s = np.floor(np.asarray(depth_top) / self.slice_m).astype(int)
        y = np.asarray(y)
        if self.n_classes == 2:
            w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
            self.overall_ = float(np.average(y, weights=w))
            self.share_ = {int(k): float(np.average(y[s == k], weights=w[s == k])) for k in np.unique(s)}
            return self
        K = self.n_classes
        self.overall_ = np.bincount(y, minlength=K) / len(y)
        self.share_ = {int(k): np.bincount(y[s == k], minlength=K) / (s == k).sum() for k in np.unique(s)}
        return self

    def predict_proba(self, depth_top):
        s = np.floor(np.asarray(depth_top) / self.slice_m).astype(int)
        if self.n_classes == 2:
            p = np.array([self.share_.get(int(k), self.overall_) for k in s])
            return np.column_stack([1 - p, p])
        return np.vstack([self.share_.get(int(k), self.overall_) for k in s]) if len(s) else np.zeros((0, self.n_classes))
