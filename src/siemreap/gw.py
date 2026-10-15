"""Geographically weighted Random Forest and XGBoost classifiers (GWRF, GWXGB).

Formulation
-----------
Follows the Geographical Random Forest of Georganos et al. (2021, Geocarto Int. 36:121-136)
and the local-global ensemble of Geographical-XGBoost (Grekousis 2025, J. Geogr. Syst.
27:169-195). Both were published for regression; this is an adaptation to binary
classification of 3D borehole cells and is described as such in the thesis.

* Anchors. Every training borehole (unique XY location) is an anchor with its own local
  model. All cells of a borehole share one XY, so a borehole is never split between models.
* Neighbourhood (adaptive bandwidth). The local model of anchor a is trained on all cells of
  the k nearest training boreholes (k counted in boreholes, so densely sampled boreholes do
  not shrink the neighbourhood). k is tuned in the inner CV.
* Distance weighting. 'uniform': every neighbourhood cell has weight 1. 'bisquare':
  w = (1 - (d / b)^2)^2, d = horizontal distance from the anchor to the cell's borehole and
  b = distance to the (k+1)-th nearest borehole, so all k neighbours keep a positive weight.
* Minimum samples and fallback. A neighbourhood must hold at least `min_boreholes`
  boreholes and `min_cells_per_class` cells of each class. If not, k grows until the
  condition holds or `max_neighbours` is reached; then the anchor falls back to the global
  model. The share of fallback anchors is stored in `fallback_rate_`.
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
    n, pos = len(y), y.sum()
    if pos == 0 or pos == n:
        return np.ones(n)
    return np.where(y == 1, n / (2 * pos), n / (2 * (n - pos)))


def _learner(base: str, p: dict, n_estimators: int, seed: int):
    if base == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=n_estimators, max_depth=p.get("max_depth"),
                                      min_samples_leaf=p.get("min_samples_leaf", 1),
                                      max_features=p.get("max_features", 1), random_state=seed, n_jobs=1)
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=n_estimators, learning_rate=p.get("learning_rate", 0.1),
                         max_depth=p.get("max_depth", 4), min_child_weight=p.get("min_child_weight", 1.0),
                         subsample=p.get("subsample", 1.0), reg_lambda=p.get("reg_lambda", 1.0),
                         tree_method="hist", n_jobs=1, random_state=seed, eval_metric="logloss")


class GWClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base="rf", k_neighbours=20, alpha=0.5, kernel="bisquare", learner_params=None,
                 class_weight="none", local_n_estimators=100, global_n_estimators=300,
                 min_boreholes=5, min_cells_per_class=20, max_neighbours=80, random_state=0, n_jobs=1):
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

    def _weights(self, y, w):
        return w * _balanced(y) if self.class_weight == "balanced" else w

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        xy = np.round(X[:, :2], 2)
        self.sites_, inv = np.unique(xy, axis=0, return_inverse=True)
        self._X, self._y, self._inv = X, y, inv.ravel()
        n_sites = len(self.sites_)
        self.global_ = _learner(self.base, dict(self.learner_params or {}), self.global_n_estimators, self.random_state)
        self.global_.fit(X, y, sample_weight=self._weights(y, np.ones(len(y))))
        self.anchor_tree_ = cKDTree(self.sites_)
        kq = min(self.max_neighbours + 1, n_sites)
        d, i = self.anchor_tree_.query(self.sites_, k=kq)
        self._dists, self._idxs, self._kq = np.atleast_2d(d), np.atleast_2d(i), kq
        self._pos = np.bincount(self._inv, weights=y, minlength=n_sites)
        self._cnt = np.bincount(self._inv, minlength=n_sites)
        self.neighbourhood_sizes_ = np.array([self._plan(a) or -1 for a in range(n_sites)])
        self.fallback_rate_ = float(np.mean(self.neighbourhood_sizes_ < 0))
        self.local_ = {}
        self.classes_ = np.array([0, 1])
        return self

    def _plan(self, a):
        """Neighbourhood size for anchor a, or None (fall back to the global model)."""
        n_sites = len(self.sites_)
        k = max(self.min_boreholes, min(self.k_neighbours, n_sites))
        while True:
            members = self._idxs[a, :k]
            pos = self._pos[members].sum()
            neg = self._cnt[members].sum() - pos
            if k >= self.min_boreholes and min(pos, neg) >= self.min_cells_per_class:
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
        m = _learner(self.base, dict(self.learner_params or {}), self.local_n_estimators, self.random_state + 1 + int(a))
        m.fit(self._X[mask], self._y[mask], sample_weight=self._weights(self._y[mask], w[mask]))
        return m

    def ensure_local(self, anchors):
        todo = [int(a) for a in np.unique(anchors) if int(a) not in self.local_]
        fitted = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._fit_local)(a) for a in todo)
        self.local_.update(dict(zip(todo, fitted)))

    def predict_proba(self, X):
        X = np.asarray(X, float)
        pg = self.global_.predict_proba(X)[:, 1]
        anchor = self.anchor_tree_.query(np.round(X[:, :2], 2))[1]
        self.ensure_local(anchor)
        pl = pg.copy()
        for a in np.unique(anchor):
            m = self.local_[int(a)]
            if m is not None:
                sel = anchor == a
                pl[sel] = m.predict_proba(X[sel])[:, 1]
        p = self.alpha * pl + (1 - self.alpha) * pg
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
