"""V3-clean helper functions: the V3 experiment, written simply.

Every rule, seed and order of steps is the same as V3 (src/siemreap in the V3 project), so
the results are identical. Notebook 10 checks this against V3's saved results.

Sections
  1. Settings and files
  2. Boreholes -> 0.5 m cells
  3. Models (RF, XGB, MLP, GWRF, GWXGB)
  4. Nested spatial cross-validation
  5. Metrics and model selection
  6. 3D map

Every model has the same three inputs (FEATURES):
  model.fit(X, y, weights)  and  model.predict_proba(X)[:, 1] = probability of Clay
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# harmless notice from tqdm when the Jupyter widget package is missing (text progress bars are used)
warnings.filterwarnings("ignore", message="IProgress not found")

FEATURES = ["UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m"]


# =============================================================== 1. Settings and files ====

def find_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "settings.toml").exists():
            return p
    raise FileNotFoundError("settings.toml not found: open the notebooks from the V3clean folder")


def load_settings(root: Path | None = None) -> dict:
    root = root or find_root()
    S = tomllib.loads((root / "settings.toml").read_text(encoding="utf-8"))
    if os.environ.get("V3CLEAN_QUICK") == "1":            # software tests only
        S["run"]["quick"] = True
    if os.environ.get("V3CLEAN_V3_FOLDER"):               # software tests only
        S["v3"]["folder"] = os.environ["V3CLEAN_V3_FOLDER"]
    S["root"] = root
    S["v3_root"] = Path(S["v3"]["folder"])
    return S


def out_dir(S: dict, step: str, private: bool = False) -> Path:
    """outputs/<step>/ (no borehole locations) or outputs/<step>/private/ (shows boreholes).
    Quick runs write to outputs_quick/."""
    base = "outputs_quick" if S["run"]["quick"] else "outputs"
    d = S["root"] / base / step / ("private" if private else "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir(S: dict) -> Path:
    d = S["root"] / ("data/processed_quick" if S["run"]["quick"] else "data/processed")
    d.mkdir(parents=True, exist_ok=True)
    return d


def v3_outputs(S: dict, stage: str, kind: str = "aggregate") -> Path:
    """A V3 output folder (V3's smoke folder in a quick run)."""
    return S["v3_root"] / "outputs" / ("smoke" if S["run"]["quick"] else "") / stage / kind


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_file(S: dict, key: str) -> Path:
    p = S["root"] / S["files"][key]
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing (see data/raw/README.md)")
    want = S["files"].get(key.replace("_csv", "_sha256"))
    if want and sha256(p) != want:
        raise ValueError(f"{p.name} is not the expected file (SHA-256 differs from settings.toml)")
    return p


def seed_for(S: dict, *parts) -> int:
    """V3's seed rule: a fixed number from the global seed and a name, e.g. seed_for(S, tag, fold)."""
    key = json.dumps([S["run"]["seed"], *map(str, parts)])
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def polygon(S: dict, key: str):
    import geopandas as gpd
    import shapely

    g = gpd.read_file(S["root"] / S["files"][key])
    return shapely.force_2d(g.to_crs(S["files"]["epsg"]).geometry.union_all())


def nearest_distance(from_xy, to_xy) -> np.ndarray:
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(to_xy)).query(np.asarray(from_xy))[0]


# ======================================================= 2. Boreholes -> 0.5 m cells ====

def read_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.insert(0, "csv_line", np.arange(len(df)) + 2)
    return df


def load_downtown(S: dict) -> pd.DataFrame:
    """Sample records of the Downtown boreholes (V3 = Downtown only)."""
    df = read_records(check_file(S, "boreholes_csv"))
    return df[df.Region == S["labels"]["region_downtown"]].reset_index(drop=True)


def _edges(S):
    c = S["cells"]
    e = np.arange(c["top_m"], c["bottom_m"] + 1e-9, c["cell_m"])
    return e[:-1], e[1:]


def _overlap(frm, to, tops, bots):
    """Metres of each record (columns) inside each cell (rows)."""
    return np.clip(np.minimum(to[None, :], bots[:, None]) - np.maximum(frm[None, :], tops[:, None]), 0, None)


def make_cells(records: pd.DataFrame, S: dict) -> tuple[pd.DataFrame, dict]:
    """A cell gets Clay (1) or Sand (0) when records of that class cover at least half of it and
    no record of the other class touches it. Other symbols, missing symbols and records longer
    than 3 m give no label."""
    c, lab = S["cells"], S["labels"]
    clay, sand = set(lab["clay"]), set(lab["sand"])
    tops, bots = _edges(S)
    length = records.Depth_To_m - records.Depth_From_m
    labelled = records[records.USCS.isin(clay | sand) & (length <= c["max_sample_length_m"])]
    meta = records.groupby("BH_Code").agg(E=("UTM_Easting_m", "first"), N=("UTM_Northing_m", "first"),
                                          G=("Ground_Elevation_m", "first"), Region=("Region", "first"))
    rows, conflict, low = [], 0, 0
    for code, b in labelled.groupby("BH_Code", sort=True):
        ov = _overlap(b.Depth_From_m.values, b.Depth_To_m.values, tops, bots)
        is_clay = b.USCS.isin(clay).values
        cov_clay, cov_sand = ov[:, is_clay].sum(axis=1), ov[:, ~is_clay].sum(axis=1)
        m = meta.loc[code]
        for j in np.where(ov.sum(axis=1) > 0)[0]:
            if cov_clay[j] > 0 and cov_sand[j] > 0:
                conflict += 1
                continue
            if max(cov_clay[j], cov_sand[j]) / c["cell_m"] < c["min_coverage"] - 1e-9:
                low += 1
                continue
            rows.append({"BH_Code": code, "cell": int(j), "depth_top_m": tops[j], "UTM_Easting_m": m.E,
                         "UTM_Northing_m": m.N, "Ground_Elevation_m": m.G, "Top_Elevation_m": m.G - tops[j],
                         "Region": m.Region, "y": int(cov_clay[j] > 0), "measured": 1})
    cells = pd.DataFrame(rows)
    return cells, {"measured_cells": len(cells), "cells_dropped_both_classes": conflict, "cells_dropped_low_coverage": low}


def fill_gaps(cells: pd.DataFrame, records: pd.DataFrame, S: dict) -> tuple[pd.DataFrame, dict]:
    """Fill the cells between two labelled cells of the same class (measured = 0) when the gap
    is at most 3 m and no record touches it (V3 rule: also an unclassified record stops it)."""
    c = S["cells"]
    tops, _ = _edges(S)
    bots = tops + c["cell_m"]
    max_cells = int(round(c["fill_max_gap_m"] / c["cell_m"]))
    blockers = records if c["unclassified_blocks_fill"] else records[records.USCS.notna()]
    touched = {code: set(np.where(_overlap(b.Depth_From_m.values, b.Depth_To_m.values, tops, bots).sum(axis=1) > 0)[0])
               for code, b in blockers.groupby("BH_Code")}
    new, n = [], {"gaps": 0, "different_class": 0, "too_long": 0, "record_inside": 0, "filled_gaps": 0}
    for code, g in cells.sort_values(["BH_Code", "cell"]).groupby("BH_Code"):
        rows = g.to_dict("records")
        for a, b in zip(rows[:-1], rows[1:]):
            gap = range(a["cell"] + 1, b["cell"])
            if not len(gap):
                continue
            n["gaps"] += 1
            if a["y"] != b["y"]:
                n["different_class"] += 1
            elif len(gap) > max_cells:
                n["too_long"] += 1
            elif touched.get(code, set()) & set(gap):
                n["record_inside"] += 1
            else:
                n["filled_gaps"] += 1
                new += [{**a, "cell": j, "depth_top_m": tops[j], "Top_Elevation_m": a["Ground_Elevation_m"] - tops[j],
                         "measured": 0} for j in gap]
    out = pd.concat([cells, pd.DataFrame(new)], ignore_index=True) if new else cells.copy()
    return out.sort_values(["BH_Code", "cell"]).reset_index(drop=True), {**n, "filled_cells": len(new)}


def load_v3_model(S: dict, path):
    """Open a model file saved by V3 (read only). V3 saved its models as classes of its own
    package, so V3's src folder must be visible to open them."""
    import sys
    import joblib
    src = str(S["v3_root"] / "src")
    if src not in sys.path:
        sys.path.append(src)
    return joblib.load(path)


def holdout_digest(codes) -> str:
    """V3's fingerprint of the external-holdout list (stored in V3's config/external_holdout.lock.json)."""
    return hashlib.sha256("\n".join(sorted(codes)).encode("utf-8")).hexdigest()


def load_v3_holdout(S: dict) -> pd.DataFrame:
    """The 11 external-holdout boreholes chosen by V3 before any modelling, checked against V3's lock."""
    h = pd.read_csv(S["v3_root"] / "data" / "processed" / "external_holdout.csv").set_index("BH_Code")
    lock = json.loads((S["v3_root"] / "config" / "external_holdout.lock.json").read_text())
    if holdout_digest(h.index) != lock["sha256"]:
        raise RuntimeError("V3's external-holdout list does not match V3's lock")
    return h


def load_new_external(S: dict) -> tuple[pd.DataFrame, dict]:
    """The new external boreholes, minus those that are also in the main file."""
    df = read_records(check_file(S, "new_external_csv"))
    main = read_records(check_file(S, "boreholes_csv"))
    dup = sorted(set(df.BH_Code) & set(main.BH_Code))
    df = df[~df.BH_Code.isin(dup)].copy()
    df["Region"] = S["labels"]["region_downtown"]
    return df.reset_index(drop=True), {"already_in_main_file": len(dup), "new_boreholes": df.BH_Code.nunique()}


# ======================================================================== 3. Models ====

def balanced_weights(y) -> np.ndarray:
    """Weights that give Clay and Sand the same total weight."""
    y = np.asarray(y)
    cls, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    if len(cls) < 2:
        return np.ones(len(y))
    return len(y) / (len(cls) * cnt[inv.ravel()])


def _two_columns(proba, classes):
    """P(Sand), P(Clay) also when a model saw only one class."""
    proba, classes = np.asarray(proba, float), np.asarray(classes, int)
    if len(classes) == 2:
        return proba
    out = np.zeros((len(proba), 2))
    out[:, classes] = proba
    return out


class RF:
    def __init__(self, params, seed, n_jobs):
        self.params, self.seed, self.n_jobs = dict(params), seed, n_jobs

    def fit(self, X, y, w=None):
        from sklearn.ensemble import RandomForestClassifier
        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        self.model_ = RandomForestClassifier(**p, class_weight=None if cw == "none" else cw, bootstrap=True,
                                             random_state=self.seed, n_jobs=self.n_jobs).fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        self.model_.n_jobs = 1        # single-threaded prediction: identical on every run
        return _two_columns(self.model_.predict_proba(X), self.model_.classes_)


class XGB:
    def __init__(self, params, seed, n_jobs):
        self.params, self.seed, self.n_jobs = dict(params), seed, n_jobs

    def fit(self, X, y, w=None):
        from xgboost import XGBClassifier
        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        y = np.asarray(y)
        weight = balanced_weights(y) if cw == "balanced" else None
        if w is not None:
            weight = np.asarray(w, float) * (weight if weight is not None else 1.0)
        self.present_, yy = np.unique(y, return_inverse=True)
        self.model_ = XGBClassifier(**p, tree_method="hist", random_state=self.seed, n_jobs=self.n_jobs,
                                    eval_metric="logloss").fit(X, yy, sample_weight=weight)
        return self

    def predict_proba(self, X):
        return _two_columns(self.model_.predict_proba(X), self.present_)


class MLP:
    """Inputs standardised on the training data; no early stopping (its random validation rows
    would mix cells of one borehole)."""

    def __init__(self, params, seed, n_jobs):
        self.params, self.seed = dict(params), seed

    def fit(self, X, y, w=None):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        p = dict(self.params)
        cw = p.pop("class_weight", "none")
        layers = (p.pop("n_units"),) * p.pop("n_layers")
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = MLPClassifier(hidden_layer_sizes=layers, early_stopping=False, random_state=self.seed, **p)
        weight = balanced_weights(y) if cw == "balanced" else None
        if w is not None:
            weight = np.asarray(w, float) * (weight if weight is not None else 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(self.scaler_.transform(X), y, sample_weight=weight)
        return self

    def predict_proba(self, X):
        return _two_columns(self.model_.predict_proba(self.scaler_.transform(X)), self.model_.classes_)


class GW:
    """Geographically weighted RF / XGBoost (GWRF, GWXGB), exactly as V3.

    * Every training borehole is an anchor with its own local model, trained on the cells of its
      k nearest boreholes (k grows until both classes have `min_cells_per_class` cells, up to
      `max_neighbours`; otherwise the anchor uses the global model).
    * Kernel 'uniform': weight 1; 'bisquare': (1 - (d/b)^2)^2, b = distance to the (k+1)-th borehole.
    * A cell uses the local model of its nearest anchor, blended with a global model:
      P = alpha * P_local + (1 - alpha) * P_global.
    * Local models are trained when a prediction first needs them (each with its own fixed seed)."""

    def __init__(self, base, learner, k, alpha, kernel, class_weight, n_local, n_global, S, seed):
        self.base, self.learner, self.k, self.alpha, self.kernel = base, dict(learner), k, alpha, kernel
        self.class_weight, self.n_local, self.n_global, self.seed = class_weight, n_local, n_global, seed
        self.g, self.n_jobs = S["gw"], S["run"]["n_jobs"]

    def _new(self, n_trees, seed):
        p = self.learner
        if self.base == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=n_trees, max_depth=p.get("max_depth"),
                                          min_samples_leaf=p.get("min_samples_leaf", 1),
                                          max_features=p.get("max_features", 1), random_state=seed, n_jobs=1)
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=n_trees, learning_rate=p.get("learning_rate", 0.1),
                             max_depth=p.get("max_depth", 4), min_child_weight=p.get("min_child_weight", 1.0),
                             subsample=p.get("subsample", 1.0), reg_lambda=p.get("reg_lambda", 1.0),
                             tree_method="hist", n_jobs=1, random_state=seed, eval_metric="logloss")

    def _fit_one(self, n_trees, seed, X, y, w):
        present, yy = np.unique(y, return_inverse=True)
        m = self._new(n_trees, seed)
        m.fit(X, yy.ravel(), sample_weight=w)
        return m, present

    def _w(self, y, w):
        return w * balanced_weights(y) if self.class_weight == "balanced" else w

    def fit(self, X, y, w=None):
        from scipy.spatial import cKDTree
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.sw_ = np.ones(len(y)) if w is None else np.asarray(w, float)
        self.anchors_, inv = np.unique(np.round(X[:, :2], 2), axis=0, return_inverse=True)
        self.X_, self.y_, self.site_ = X, y, inv.ravel()
        n = len(self.anchors_)
        self.global_ = self._fit_one(self.n_global, self.seed, X, y, self._w(y, self.sw_))
        self.tree_ = cKDTree(self.anchors_)
        kq = min(self.g["max_neighbours"] + 1, n)
        d, i = self.tree_.query(self.anchors_, k=kq)
        self.dist_, self.nbr_, self.kq_ = np.atleast_2d(d), np.atleast_2d(i), kq
        counts = np.zeros((n, 2))
        np.add.at(counts, (self.site_, y), 1)
        self.size_ = np.array([self._neighbourhood(a, counts) for a in range(n)])
        self.fallback_rate_ = float(np.mean(self.size_ < 0))
        self.local_ = {}
        return self

    def _neighbourhood(self, a, counts):
        """Number of neighbour boreholes of anchor a, or -1 (use the global model)."""
        g, n = self.g, len(self.anchors_)
        k = max(g["min_boreholes"], min(self.k, n))
        while True:
            enough = (counts[self.nbr_[a, :k]].sum(axis=0) >= g["min_cells_per_class"]).sum()
            if k >= g["min_boreholes"] and enough >= 2:
                return int(k)
            if k >= min(g["max_neighbours"], n):
                return -1
            k += 1

    def _fit_local(self, a):
        k = self.size_[a]
        if k < 0:
            return None
        d = self.dist_[a]
        b = d[k] if k < self.kq_ else d[k - 1] * 1.01 + 1e-6
        w_site = (1 - (d[:k] / max(b, 1e-9)) ** 2) ** 2 if self.kernel == "bisquare" else np.ones(k)
        wmap = np.zeros(len(self.anchors_))
        wmap[self.nbr_[a, :k]] = w_site
        w = wmap[self.site_]
        use = w > 0
        return self._fit_one(self.n_local, self.seed + 1 + int(a), self.X_[use], self.y_[use],
                             self._w(self.y_[use], w[use] * self.sw_[use]))

    def predict_proba(self, X):
        from joblib import Parallel, delayed
        X = np.asarray(X, float)
        m, present = self.global_
        p_global = _two_columns(m.predict_proba(X), present)
        anchor = self.tree_.query(np.round(X[:, :2], 2))[1]
        todo = [int(a) for a in np.unique(anchor) if int(a) not in self.local_]
        fitted = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._fit_local)(a) for a in todo)
        self.local_.update(zip(todo, fitted))
        p_local = p_global.copy()
        for a in np.unique(anchor):
            f = self.local_[int(a)]
            if f is not None:
                sel = anchor == a
                p_local[sel] = _two_columns(f[0].predict_proba(X[sel]), f[1])
        p = self.alpha * p_local[:, 1] + (1 - self.alpha) * p_global[:, 1]
        return np.column_stack([1 - p, p])


def make_model(name: str, params: dict, seed: int, S: dict):
    """A model with one parameter set. Quick runs use smaller models (as V3's smoke mode)."""
    p = {k: v for k, v in params.items() if k != "filled_weight"}
    quick, q = S["run"]["quick"], S["quick"]
    if quick and "n_estimators" in p:
        p["n_estimators"] = min(p["n_estimators"], q["rf_trees"])
    if quick and "max_iter" in p:
        p["max_iter"] = min(p["max_iter"], q["mlp_max_iter"])
    n_jobs = S["run"]["n_jobs"]
    if name == "RF":
        return RF(p, seed, n_jobs)
    if name == "XGB":
        return XGB(p, seed, n_jobs)
    if name == "MLP":
        return MLP(p, seed, n_jobs)
    g = S["gw"]
    k, alpha, kernel, cw = p.pop("k_neighbours"), p.pop("alpha"), p.pop("kernel"), p.pop("class_weight", "none")
    if name == "GWRF":
        n_local = min(g["local_rf_trees"], 30) if quick else g["local_rf_trees"]
        return GW("rf", p, k, alpha, kernel, cw, n_local, 3 * n_local, S, seed)
    n = min(p.pop("n_estimators"), g["local_xgb_rounds_max"])
    return GW("xgb", p, k, alpha, kernel, cw, n, n, S, seed)


def fill_weights(cells: pd.DataFrame, params: dict):
    """Training weights: 1 for measured cells, `filled_weight` for filled cells (None = all 1)."""
    fw = params.get("filled_weight", 1.0)
    meas = cells.measured.values == 1
    return None if fw == 1.0 or meas.all() else np.where(meas, 1.0, float(fw))


def suggest(trial, space: dict) -> dict:
    p = {}
    for name, s in space.items():
        if s["type"] == "int":
            p[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1), log=s.get("log", False))
        elif s["type"] == "float":
            p[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        else:
            p[name] = trial.suggest_categorical(name, s["choices"])
    return p


# ================================================== 4. Nested spatial cross-validation ====

def load_folds(S: dict, block_m: int) -> pd.Series:
    """V3's blockCV outer folds: the fold number of every Downtown development borehole."""
    f = pd.read_csv(S["v3_root"] / "data" / "folds" / f"outer_folds_{int(block_m)}m.csv").set_index("BH_Code")
    return f.fold


def block_sizes(S: dict) -> list[int]:
    """V3's block sizes: blockCV's 106 m, then round(2 x 106) and round(3 x 106)."""
    d = json.loads((S["v3_root"] / "data" / "folds" / "block_size_decision.json").read_text())
    base = int(d["block_size_m"])
    return list(dict.fromkeys([base] + [int(round(base * m)) for m in S["blockcv"]["multiples"][1:]]))


def outer_split(dev: pd.DataFrame, folds: pd.Series, k: int, purge_m: float = 0.0):
    """Test = development boreholes in fold k. Train = the other folds, minus (with a purge)
    the boreholes closer than purge_m to any test borehole."""
    fold = dev.BH_Code.map(folds)
    test = dev[fold == k]
    train = dev[fold.notna() & (fold != k)]
    if purge_m > 0:
        tb = test.drop_duplicates("BH_Code").set_index("BH_Code")[FEATURES[:2]]
        trb = train.drop_duplicates("BH_Code").set_index("BH_Code")[FEATURES[:2]]
        too_close = set(trb.index[nearest_distance(trb.values, tb.values) < purge_m])
        train = train[~train.BH_Code.isin(too_close)]
    return train, test


def inner_groups(train: pd.DataFrame, folds: pd.Series, k: int, block_m: int, S: dict):
    """Inner folds = the other 4 outer folds. If one of them is too small or too one-sided
    (V3's check), V3's backup inner folds made by blockCV are used for this outer fold."""
    ic = S["inner_cv"]

    def ok(groups):
        for g in sorted(set(groups[groups > 0])):
            v = train[groups == g]
            share = v.y.mean()
            if not (v.BH_Code.nunique() >= ic["min_boreholes"] and len(v) >= ic["min_cells"]
                    and ic["min_class_share"] <= share <= 1 - ic["min_class_share"]):
                return False
        return True

    groups = train.BH_Code.map(folds).astype(int).values.copy()
    groups[groups == 0] = -1
    if ok(groups):
        return groups, "outer_folds"
    fb = pd.read_csv(S["v3_root"] / "data" / "folds" / f"inner_fallback_{int(block_m)}m.csv")
    m = fb[fb.outer_fold == k].set_index("BH_Code").inner_fold
    return train.BH_Code.map(m).fillna(-1).astype(int).values, "backup_folds"


def threshold_grid(S):
    a, b, s = S["cv"]["threshold_grid"]
    return np.round(np.arange(a, b + s / 2, s), 4)


def macro_f1(y, y_pred) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y, y_pred, average="macro", labels=[0, 1], zero_division=0)


def best_threshold(y, p, grid) -> tuple[float, float]:
    """Threshold on P(Clay) with the highest Macro-F1 (ties: the one closest to 0.5)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    best = (-1.0, 0.5)
    for t in grid:
        f = macro_f1(y, (p >= t).astype(int))
        if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and abs(t - 0.5) < abs(best[1] - 0.5)):
            best = (f, float(t))
    return best[1], best[0]


def n_trials(name: str, S: dict) -> int:
    return S["quick"]["n_trials"] if S["run"]["quick"] else S["cv"]["n_trials"][name]


def tune(name, train, groups, seed, tpe_seed, S, on_trial=None) -> dict:
    """Optuna (TPE) picks the hyperparameters and the filled-cell weight with the highest inner
    Macro-F1. Each trial: train on 3 inner groups, predict the 4th, for all 4 groups; then the
    best threshold on these out-of-fold predictions."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = train[FEATURES].values, train.y.values
    ids = sorted(set(groups[groups > 0]))
    grid = threshold_grid(S)
    has_filled = not (train.measured.values == 1).all()

    def inner_predictions(params):
        oof = np.full(len(train), np.nan)
        w = fill_weights(train, params)
        for g in ids:
            val = groups == g
            model = make_model(name, params, seed, S).fit(X[~val], y[~val], None if w is None else w[~val])
            oof[val] = model.predict_proba(X[val])[:, 1]
        return oof

    scored = np.isin(groups, ids)

    def objective(trial):
        params = suggest(trial, S["space"][name])
        if has_filled:
            params["filled_weight"] = trial.suggest_categorical("filled_weight", [float(v) for v in S["cv"]["filled_weights"]])
        oof = inner_predictions(params)
        t, f = best_threshold(y[scored], oof[scored], grid)
        trial.set_user_attr("threshold", t)
        return f

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tpe_seed))
    study.optimize(objective, n_trials=n_trials(name, S), n_jobs=1,
                   callbacks=[lambda study, trial: on_trial()] if on_trial else None)
    return {"params": study.best_params, "threshold": study.best_trial.user_attrs["threshold"],
            "inner_macro_f1": study.best_value,
            "trials": study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))}


def run_list(S: dict) -> list[dict]:
    """The V3 runs (without the depth-only reference):
    main block (106 m): RF, XGB, MLP, GWRF, GWXGB;  212 and 318 m: RF, XGB;
    purge at 106 and 212 m (main block): RF, XGB.  Quick: main block + RF with the first purge (as V3 smoke)."""
    cv, sizes = S["cv"], block_sizes(S)
    main = sizes[0]
    if S["run"]["quick"]:
        runs = [(m, main, 0.0) for m in cv["models"]] + [(cv["purge_models"][0], main, float(main * cv["purge_multiples"][0]))]
    else:
        runs = [(m, b, 0.0) for b in sizes for m in (cv["models"] if b == main else cv["sensitivity_models"])]
        runs += [(m, main, float(main * x)) for x in cv["purge_multiples"] for m in cv["purge_models"]]
    return [{"model": m, "block_m": b, "purge_m": p, "tag": f"A_{b}m_p{int(p)}_{m}"} for m, b, p in runs]


def run_fold(run: dict, dev: pd.DataFrame, k: int, S: dict, on_trial=None) -> dict:
    """One outer fold of one run: tune on the training boreholes, refit, predict fold k."""
    t0 = time.time()
    folds = load_folds(S, run["block_m"])
    train, test = outer_split(dev, folds, k, run["purge_m"])
    groups, source = inner_groups(train, folds, k, run["block_m"], S)
    seed = seed_for(S, run["tag"], k)
    best = tune(run["model"], train, groups, seed, seed_for(S, run["tag"], k, "tpe"), S, on_trial)
    model = make_model(run["model"], best["params"], seed, S).fit(train[FEATURES].values, train.y.values,
                                                                    fill_weights(train, best["params"]))
    p = model.predict_proba(test[FEATURES].values)[:, 1]
    pred = test[["BH_Code", "cell", "depth_top_m", *FEATURES, "y", "measured"]].copy()
    pred["p_clay"] = p
    pred["threshold"] = best["threshold"]
    pred["y_pred"] = (p >= best["threshold"]).astype(int)
    tb = test.drop_duplicates("BH_Code").set_index("BH_Code")[FEATURES[:2]]
    trb = train.drop_duplicates("BH_Code").set_index("BH_Code")[FEATURES[:2]]
    dist = pd.Series(nearest_distance(tb.values, trb.values), index=tb.index)
    pred["dist_nearest_train_m"] = pred.BH_Code.map(dist).values
    pred = pred.assign(outer_fold=k, scenario="A", block_m=run["block_m"], model=run["model"], purge_m=run["purge_m"])
    info = {**run, "outer_fold": k, "inner_folds": source, "params": best["params"], "threshold": best["threshold"],
            "inner_macro_f1": best["inner_macro_f1"], "train_boreholes": int(train.BH_Code.nunique()),
            "train_cells": len(train), "test_boreholes": int(test.BH_Code.nunique()), "test_cells": len(test),
            "gw_fallback_rate": getattr(model, "fallback_rate_", None), "seconds": round(time.time() - t0, 1)}
    return {"predictions": pred, "info": info, "trials": best["trials"], "model": model}


def fit_final(name: str, dev: pd.DataFrame, block_m: int, S: dict, on_trial=None) -> dict:
    """Final model, as V3: tune on ALL development boreholes (groups = the 5 outer folds),
    then train on all of them."""
    folds = load_folds(S, block_m)
    groups = dev.BH_Code.map(folds).astype(int).values.copy()
    groups[groups == 0] = -1
    tag = f"final_A_{block_m}m_{name}"
    seed = seed_for(S, tag)
    best = tune(name, dev, groups, seed, seed_for(S, tag, "tpe"), S, on_trial)
    model = make_model(name, best["params"], seed, S).fit(dev[FEATURES].values, dev.y.values,
                                                          fill_weights(dev, best["params"]))
    return {**best, "model": model}


def refit(name: str, params: dict, cells: pd.DataFrame, seed_tag: str, S: dict):
    """Train with fixed parameters, no tuning (the 120-borehole map model of V3 notebook 10)."""
    return make_model(name, params, seed_for(S, seed_tag), S).fit(cells[FEATURES].values, cells.y.values,
                                                                  fill_weights(cells, params))


# ================================================ 5. Metrics and model selection ====

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "recall_clay", "recall_sand", "mcc", "roc_auc",
           "pr_auc_clay", "pr_auc_sand", "brier"]
RUN_KEYS = ["scenario", "block_m", "purge_m", "model"]


def scores(y, p, y_pred) -> dict:
    """V3's metrics for one set of predictions."""
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                                 confusion_matrix, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
    y, p, y_pred = np.asarray(y, int), np.asarray(p, float), np.asarray(y_pred, int)
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "clay_share": float(y.mean()) if len(y) else np.nan,
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred) if both else np.nan,
            "macro_f1": macro_f1(y, y_pred),
            "recall_clay": recall_score(y, y_pred, pos_label=1, zero_division=np.nan),
            "recall_sand": recall_score(y, y_pred, pos_label=0, zero_division=np.nan),
            "precision_clay": precision_score(y, y_pred, pos_label=1, zero_division=np.nan),
            "precision_sand": precision_score(y, y_pred, pos_label=0, zero_division=np.nan),
            "mcc": matthews_corrcoef(y, y_pred) if both else np.nan,
            "roc_auc": roc_auc_score(y, p) if both else np.nan,
            "pr_auc_clay": average_precision_score(y, p) if both else np.nan,
            "pr_auc_sand": average_precision_score(1 - y, 1 - p) if both else np.nan,
            "brier": brier_score_loss(y, p, pos_label=1) if len(y) else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def score_table(preds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in preds.groupby(keys, sort=True, dropna=False):
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **scores(g.y, g.p_clay, g.y_pred)})
    return pd.DataFrame(rows)


def fold_scores(preds: pd.DataFrame) -> pd.DataFrame:
    return score_table(preds, RUN_KEYS + ["outer_fold"])


def summary(fold_table: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and minimum over the outer folds."""
    g = fold_table.groupby(RUN_KEYS)[METRICS]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.min().add_suffix("_min"))
    out["n_folds"] = fold_table.groupby(RUN_KEYS).size()
    return out.reset_index()


def select_model(summ: pd.DataFrame, main_block: int) -> dict:
    """V3's rule, decided before the results, on the main block size without purge:
      1. eligible: mean Clay and Sand recall >= 0.50 and no fold below 0.25;
      2. best = highest mean Macro-F1;
      3. among eligible models within one standard error of the best: lowest fold SD of
         Macro-F1, then lowest mean Brier score."""
    s = summ[(summ.block_m == main_block) & (summ.purge_m == 0)].copy()
    s["eligible"] = ((s.recall_clay_mean >= 0.5) & (s.recall_sand_mean >= 0.5)
                     & (s.recall_clay_min >= 0.25) & (s.recall_sand_min >= 0.25))
    e = s[s.eligible]
    if e.empty:
        return {"selected": None, "table": s}
    best = e.loc[e.macro_f1_mean.idxmax()]
    se = best.macro_f1_sd / np.sqrt(best.n_folds) if best.n_folds > 1 else 0.0
    near = e[e.macro_f1_mean >= best.macro_f1_mean - se].sort_values(["macro_f1_sd", "brier_mean"])
    return {"selected": near.model.iloc[0], "highest_mean": best.model, "best_macro_f1_mean": float(best.macro_f1_mean),
            "one_se": float(se), "within_1se": list(near.model), "table": s}


def distance_table(preds: pd.DataFrame, edges=(0, 50, 100, 200, 300, 500, 1000, np.inf)) -> pd.DataFrame:
    """Scores by distance from a test borehole to the nearest training borehole (V3's bins)."""
    labels = [f"{a:g}-{b:g} m" if np.isfinite(b) else f">{a:g} m" for a, b in zip(edges[:-1], edges[1:])]
    d = preds.assign(distance_bin=pd.cut(preds.dist_nearest_train_m, list(edges), right=False, labels=labels).astype(str))
    t = score_table(d, RUN_KEYS + ["distance_bin"])
    t["distance_bin"] = pd.Categorical(t.distance_bin, labels, ordered=True)
    return t.sort_values(RUN_KEYS + ["distance_bin"])


# ===================================================================== 6. 3D map ====

class DEM:
    """The DEM, sampled like the CSV ground elevations (bilinear between cell centres)."""

    def __init__(self, S: dict):
        import rasterio
        with rasterio.open(S["root"] / S["files"]["dem"]) as r:
            self.z = r.read(1).astype("float64")
            if r.nodata is not None:
                self.z[self.z == r.nodata] = np.nan
            self.left, self.top, self.res = r.bounds.left, r.bounds.top, r.res[0]

    def elevation(self, x, y) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        fx, fy = (x - self.left) / self.res - 0.5, (self.top - y) / self.res - 0.5
        j, i = np.floor(fx).astype(int), np.floor(fy).astype(int)
        a, b = fx - j, fy - i
        ok = (i >= 0) & (j >= 0) & (i + 1 < self.z.shape[0]) & (j + 1 < self.z.shape[1])
        out = np.full(x.shape, np.nan)
        i, j, a, b, z = i[ok], j[ok], a[ok], b[ok], self.z
        out[ok] = (1 - b) * ((1 - a) * z[i, j] + a * z[i, j + 1]) + b * ((1 - a) * z[i + 1, j] + a * z[i + 1, j + 1])
        return out


def map_columns(S: dict, spacing: float) -> pd.DataFrame:
    """Grid points inside Downtown with their DEM ground elevation."""
    import shapely
    poly = polygon(S, "downtown_shp")
    x0, y0, x1, y1 = poly.bounds
    gx = np.arange(np.floor(x0 / spacing) * spacing + spacing / 2, x1, spacing)
    gy = np.arange(np.floor(y0 / spacing) * spacing + spacing / 2, y1, spacing)
    X, Y = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(poly, X.ravel(), Y.ravel())
    cols = pd.DataFrame({"X": X.ravel()[inside], "Y": Y.ravel()[inside]})
    cols["ground_m"] = DEM(S).elevation(cols.X.values, cols.Y.values)
    return cols


def map_cells(cols: pd.DataFrame, S: dict) -> pd.DataFrame:
    tops, _ = _edges(S)
    n = len(tops)
    g = pd.DataFrame({"col": np.repeat(np.arange(len(cols)), n), "layer": np.tile(np.arange(n), len(cols)),
                      "depth_top_m": np.tile(tops, len(cols))})
    g["UTM_Easting_m"] = cols.X.values[g.col]
    g["UTM_Northing_m"] = cols.Y.values[g.col]
    g["ground_m"] = cols.ground_m.values[g.col]
    g["Top_Elevation_m"] = g.ground_m - g.depth_top_m
    g["Z"] = g.Top_Elevation_m - S["cells"]["cell_m"] / 2
    return g


def to_voxel_netcdf(grid: pd.DataFrame, cols: pd.DataFrame, S: dict, path) -> dict:
    """netCDF (CF-1.8) on a regular elevation axis, for an ArcGIS Pro voxel layer."""
    import xarray as xr
    from pyproj import CRS
    h, bottom = S["cells"]["cell_m"], S["cells"]["bottom_m"]
    xs, ys = np.unique(cols.X.values), np.unique(cols.Y.values)
    ix, iy = np.searchsorted(xs, cols.X.values), np.searchsorted(ys, cols.Y.values)
    z0 = np.floor((cols.ground_m.min() - bottom) / h) * h
    zc = np.arange(z0 + h / 2, np.ceil(cols.ground_m.max() / h) * h, h)
    col, layer = grid.col.values, grid.layer.values
    m = np.floor((cols.ground_m.values[col] - h * layer - z0 - h / 2) / h).astype(int)
    ok = (m >= 0) & (m < len(zc))
    data = {}
    for name, fill, dtype in (("p_clay", np.nan, "float32"), ("clay", -1, "int8"), ("reliability_zone", -1, "int8")):
        a = np.full((len(zc), len(ys), len(xs)), fill, dtype=dtype)
        a[m[ok], iy[col[ok]], ix[col[ok]]] = grid[name].values[ok]
        data[name] = (("z", "y", "x"), a)
    crs = CRS.from_epsg(S["files"]["epsg"])
    ds = xr.Dataset(data, coords={"z": zc, "y": ys, "x": xs})
    ds["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": crs.to_wkt(), "spatial_ref": crs.to_wkt()})
    for v in data:
        ds[v].attrs["grid_mapping"] = "spatial_ref"
    ds.x.attrs.update(standard_name="projection_x_coordinate", units="m", axis="X")
    ds.y.attrs.update(standard_name="projection_y_coordinate", units="m", axis="Y")
    ds.z.attrs.update(standard_name="height", units="m", positive="up", axis="Z")
    ds.attrs.update(Conventions="CF-1.8", title="Siem Reap Downtown 3D Clay/Sand prediction (V3)")
    enc = {v: {"zlib": True, "complevel": 4, **({"_FillValue": -1} if data[v][1].dtype == np.int8 else {})} for v in data}
    ds.to_netcdf(path, engine="netcdf4", encoding=enc)
    return {"shape_zyx": [len(zc), len(ys), len(xs)], "voxels_filled": int(ok.sum())}
