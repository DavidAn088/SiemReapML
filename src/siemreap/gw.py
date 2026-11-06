"""Geographically weighted Random Forest and XGBoost classifiers (GWRF, GWXGB).

Formulation
-----------
Follows the Geographical Random Forest of Georganos et al. (2021, Geocarto Int. 36:121-136)
and the local-global ensemble of Geographical-XGBoost (Grekousis 2025, J. Geogr. Syst.
27:169-195). Both were published for regression; this is an adaptation to classification
of 3D borehole cells (binary Clay/Sand, or K soil classes) and is described as such in the thesis.

* Anchors. Every training borehole (unique XY location) is an anchor with its own local
  model. All cells of a borehole share one XY, so a borehole is never split between models.
* Neighbourhood (adaptive bandwidth). The local model of anchor a is trained on all cells of
  the k nearest training boreholes (k counted in boreholes, so densely sampled boreholes do
  not shrink the neighbourhood). k is tuned in the inner CV.
* Distance weighting. 'uniform': every neighbourhood cell has weight 1. 'bisquare':
  w = (1 - (d / b)^2)^2, d = horizontal distance from the anchor to the cell's borehole and
  b = distance to the (k+1)-th nearest borehole, so all k neighbours keep a positive weight.
* Minimum samples and fallback. A neighbourhood must hold at least `min_boreholes`
  boreholes and at least two classes with `min_cells_per_class` cells each (binary: both
  classes). If not, k grows until the condition holds or `max_neighbours` is reached; then
  the anchor falls back to the global model. The share of fallback anchors is stored in
  `fallback_rate_`. A class absent from a neighbourhood gets local probability 0 and is
  supplied by the global model through the blend.
* Prediction. A location uses the local model of its nearest anchor (as in GRF) and blends
  it with a global model fitted to all training cells:
  p = alpha * p_local + (1 - alpha) * p_global, with alpha tuned in the inner CV.
* Complexity. A complete model has 1 global model + one local model per training borehole
  (~100-145). Local models are fitted lazily: only the anchors that are nearest to the
  points being predicted are fitted (a CV test fold needs ~20, the 3D grid needs all), and
  each is cached. Because each anchor has a fixed seed and its own neighbourhood, the
  predictions are identical to fitting every anchor up front. Local models are fitted in
  parallel threads, each single-threaded, so results do not depend on scheduling.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClassifierMixin


def _balanced(y: np.ndarray) -> np.ndarray:
    """'Balanced' sample weights n / (classes present x class count); binary: n / (2 pos), n / (2 neg)."""
    y = np.asarray(y)
    cls, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    if len(cls) < 2:
        return np.ones(len(y))
    return len(y) / (len(cls) * cnt[inv.ravel()])


def expand_proba(proba: np.ndarray, present, n_classes: int) -> np.ndarray:
    """Probabilities of the classes a model saw in training, placed in the columns of all
    `n_classes` classes (classes absent from training get probability 0)."""
    present = np.asarray(present, int)
    proba = np.asarray(proba, float)
    if len(present) == n_classes and (present == np.arange(n_classes)).all():
        return proba
    out = np.zeros((len(proba), n_classes))
    out[:, present] = proba
    return out


def _learner(base: str, p: dict, n_estimators: int, seed: int, n_present: int = 2):
    if base == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=n_estimators, max_depth=p.get("max_depth"),
                                      min_samples_leaf=p.get("min_samples_leaf", 1),
                                      max_features=p.get("max_features", 1), random_state=seed, n_jobs=1)
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=n_estimators, learning_rate=p.get("learning_rate", 0.1),
                         max_depth=p.get("max_depth", 4), min_child_weight=p.get("min_child_weight", 1.0),
                         subsample=p.get("subsample", 1.0), reg_lambda=p.get("reg_lambda", 1.0),
                         tree_method="hist", n_jobs=1, random_state=seed,
                         eval_metric="logloss" if n_present <= 2 else "mlogloss")


def _fit_encoded(base, p, n_estimators, seed, X, y, w):
    """Fit on labels re-coded to 0..m-1 (needed by XGBoost); returns (model, classes present)."""
    present, yy = np.unique(y, return_inverse=True)
    m = _learner(base, p, n_estimators, seed, len(present))
    m.fit(X, yy.ravel(), sample_weight=w)
    return m, present


class GWClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base="rf", k_neighbours=20, alpha=0.5, kernel="bisquare", learner_params=None,
                 class_weight="none", local_n_estimators=100, global_n_estimators=300,
                 min_boreholes=5, min_cells_per_class=20, max_neighbours=80, random_state=0, n_jobs=1,
                 n_classes=2):
        self.base = base
        self.k_neighbours = k_neighbours
        self.alpha = alpha
        self.kernel = kernel
        self.learner_params = learner_params
        self.class_weight = class_weight
        self.local_n_estimators = local_n_estimators
        self.global_n_estimators = global_n_estimators
        self.min_boreholes = min_boreholes
        self.min_cells_per_class = min_cells_per_class
        self.max_neighbours = max_neighbours
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_classes = n_classes

    def _weights(self, y, w):
        return w * _balanced(y) if self.class_weight == "balanced" else w

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self._sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        xy = np.round(X[:, :2], 2)
        self.sites_, inv = np.unique(xy, axis=0, return_inverse=True)
        self._X, self._y, self._inv = X, y, inv.ravel()
        n_sites = len(self.sites_)
        self.global_, self.global_present_ = _fit_encoded(self.base, dict(self.learner_params or {}),
                                                          self.global_n_estimators, self.random_state, X, y,
                                                          self._weights(y, self._sw))
        self.anchor_tree_ = cKDTree(self.sites_)
        kq = min(self.max_neighbours + 1, n_sites)
        d, i = self.anchor_tree_.query(self.sites_, k=kq)
        self._dists, self._idxs, self._kq = np.atleast_2d(d), np.atleast_2d(i), kq
        # cells of each class at each site (n_sites x n_classes)
        self._cc = np.zeros((n_sites, self.n_classes))
        np.add.at(self._cc, (self._inv, y), 1)
        self.neighbourhood_sizes_ = np.array([self._plan(a) or -1 for a in range(n_sites)])
        self.fallback_rate_ = float(np.mean(self.neighbourhood_sizes_ < 0))
        self.local_ = {}
        self.classes_ = np.arange(self.n_classes)
        return self

    def _plan(self, a):
        """Neighbourhood size for anchor a, or None (fall back to the global model)."""
        n_sites = len(self.sites_)
        k = max(self.min_boreholes, min(self.k_neighbours, n_sites))
        while True:
            members = self._idxs[a, :k]
            # at least two classes with min_cells_per_class cells (binary: both classes)
            enough = (self._cc[members].sum(axis=0) >= self.min_cells_per_class).sum()
            if k >= self.min_boreholes and enough >= 2:
                return int(k)
            if k >= min(self.max_neighbours, n_sites):
                return None
            k += 1

    def _fit_local(self, a):
        k = self.neighbourhood_sizes_[a]
        if k < 0:
            return None
        members = self._idxs[a, :k]
        dists = self._dists[a]
        b = dists[k] if k < self._kq else dists[k - 1] * 1.01 + 1e-6
        w_site = (1 - (dists[:k] / max(b, 1e-9)) ** 2) ** 2 if self.kernel == "bisquare" else np.ones(k)
        wmap = np.zeros(len(self.sites_))
        wmap[members] = w_site
        w = wmap[self._inv]
        mask = w > 0
        return _fit_encoded(self.base, dict(self.learner_params or {}), self.local_n_estimators,
                            self.random_state + 1 + int(a), self._X[mask], self._y[mask],
                            self._weights(self._y[mask], w[mask] * self._sw[mask]))

    def ensure_local(self, anchors):
        todo = [int(a) for a in np.unique(anchors) if int(a) not in self.local_]
        fitted = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._fit_local)(a) for a in todo)
        self.local_.update(dict(zip(todo, fitted)))

    def predict_proba(self, X):
        X = np.asarray(X, float)
        K = self.n_classes
        pg = expand_proba(self.global_.predict_proba(X), self.global_present_, K)
        anchor = self.anchor_tree_.query(np.round(X[:, :2], 2))[1]
        self.ensure_local(anchor)
        pl = pg.copy()
        for a in np.unique(anchor):
            fitted = self.local_[int(a)]
            if fitted is not None:
                m, present = fitted
                sel = anchor == a
                pl[sel] = expand_proba(m.predict_proba(X[sel]), present, K)
        if K == 2:  # binary: blend P(Clay) and return [1 - p, p]
            p = self.alpha * pl[:, 1] + (1 - self.alpha) * pg[:, 1]
            return np.column_stack([1 - p, p])
        return self.alpha * pl + (1 - self.alpha) * pg

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1) if self.n_classes > 2 else (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
