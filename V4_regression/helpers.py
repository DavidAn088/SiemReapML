"""V4 regression helper functions: layer-average soil parameters from location and the
classifier's predicted soil type, under V3's spatial CV, then a 3D map on the classifier's grid.

* Target: the mean of a borehole's samples in each 1.5 m layer (settlement uses layer values).
  Only directly measured parameters (w, LL, PL, wet unit weight, Su, SPT-N); derived ones
  (PI, e0, dry unit weight, ...) are calculated from the predictions, not modelled.
* Inputs: Easting, Northing, mid elevation of the layer and P(Clay) = the mean predicted
  probability of Clay of the layer's samples, from the Clay/Sand classifier (V3-clean, XGB):
    - development boreholes: the classifier's OUT-OF-FOLD model (fold k model for a borehole
      in fold k), so the input is as good as on the map, not as good as the log;
    - external-holdout boreholes: the frozen final classifier (trained without them);
    - map: the classifier grid (p_clay) at every grid point.
* V3's Downtown boreholes, 106 m blockCV folds and 11 hash-locked holdout boreholes.

  model.fit(X, y, depth)  and  model.predict(X, depth) = value on the original scale
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

warnings.filterwarnings("ignore", message="IProgress not found")

FEATURES = ["UTM_Easting_m", "UTM_Northing_m", "Mid_Elevation_m", "p_clay"]
COORDS = ["UTM_Easting_m", "UTM_Northing_m"]          # for distances (always kept in the tables)
NO_TUNING = ("Depth",)


def design(df: pd.DataFrame, name: str = "") -> np.ndarray:
    return df[FEATURES].values


def set_features(cols: list[str]) -> None:
    """Inputs of the following runs (e.g. the property inputs in notebook 05)."""
    global FEATURES
    FEATURES = list(cols)


# =============================================================== 1. Settings and files ====

def find_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "settings.toml").exists():
            return p
    raise FileNotFoundError("settings.toml not found: open the notebooks from the V5_regression folder")


def load_settings(root: Path | None = None) -> dict:
    root = root or find_root()
    S = tomllib.loads((root / "settings.toml").read_text(encoding="utf-8"))
    if os.environ.get("V5_QUICK") == "1":                 # software tests only
        S["run"]["quick"] = True
    if os.environ.get("V5_V3_FOLDER"):                    # software tests only
        S["v3"]["folder"] = os.environ["V5_V3_FOLDER"]
    S["root"] = root
    S["v3_root"] = Path(S["v3"]["folder"])
    S["v3clean_root"] = find_v3clean(root, S)
    return S


def find_v3clean(root: Path, S: dict) -> Path:
    """The V3-clean folder: [classifier] v3clean_folder if it holds the classifier models, otherwise
    the first folder near V4 (up to 3 levels below the folder that holds V4) that does."""
    c = S["classifier"]
    want = c["fold_models"].format(k=1)
    first = (root / c["v3clean_folder"]).resolve()
    if (first / want).exists():
        return first
    base = root.parent
    for depth in range(4):
        for d in sorted(base.glob("/".join(["*"] * depth))) if depth else [base]:
            if d.is_dir() and d != root and (d / want).exists():
                return d.resolve()
    return first          # not found: load_classifiers() reports the missing files


def out_dir(S: dict, step: str, private: bool = False) -> Path:
    """outputs/<step>/ (no borehole codes or locations) or outputs/<step>/private/."""
    base = "outputs_quick" if S["run"]["quick"] else "outputs"
    d = S["root"] / base / step / ("private" if private else "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir(S: dict) -> Path:
    d = S["root"] / ("data/processed_quick" if S["run"]["quick"] else "data/processed")
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    key = json.dumps([S["run"]["seed"], *map(str, parts)])
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def nearest_distance(from_xy, to_xy) -> np.ndarray:
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(to_xy)).query(np.asarray(from_xy))[0]


# ============================================== 2. Boreholes -> samples of one target ====

def load_downtown(S: dict) -> pd.DataFrame:
    df = pd.read_csv(check_file(S, "boreholes_csv"), encoding="utf-8-sig", low_memory=False)
    return df[df.Region == S["files"]["region_downtown"]].reset_index(drop=True)


def make_samples(records: pd.DataFrame, target: str, S: dict, extra: list[str] = ()) -> tuple[pd.DataFrame, dict]:
    """Rows with a value of the target. Kept when the record is at most 3 m long, its middle
    lies in 0-15 m and the value is inside the target's [min, max] (and, if the target lists
    `uscs`, the record's USCS symbol is one of them). Then values farther than `outlier_sd`
    standard deviations from the mean of the kept values are dropped. Mid elevation =
    ground elevation - middle depth (the same ground elevation as V3's cells)."""
    t, s = S["targets"][target], S["samples"]
    v = pd.to_numeric(records[t["column"]], errors="coerce")
    r = records.assign(value=v)[v.notna()].copy()
    info = {"records_with_value": len(r)}
    mid = (r.Depth_From_m + r.Depth_To_m) / 2
    long_ = (r.Depth_To_m - r.Depth_From_m) > s["max_sample_length_m"]
    outside = (mid < s["top_m"]) | (mid > s["bottom_m"])
    bad = (r.value < t["min"]) | (r.value > t["max"])
    info |= {"dropped_longer_than_3m": int(long_.sum()), "dropped_outside_depth": int((outside & ~long_).sum()),
             "dropped_out_of_range": int((bad & ~long_ & ~outside).sum())}
    if t.get("uscs"):                                    # e.g. Su on clay samples only
        other = ~r.USCS.isin(t["uscs"]) & ~long_ & ~outside & ~bad
        info["dropped_other_uscs"] = int(other.sum())
        bad = bad | other
    keep = ~long_ & ~outside & ~bad
    k_sd = s.get("outlier_sd")
    if k_sd:                                             # Khamjan et al. (2026): drop values beyond mean +- 3 SD
        v = r.value[keep]
        far = keep & ((r.value - v.mean()).abs() > k_sd * v.std())
        info["dropped_beyond_sd"] = int(far.sum())
        keep = keep & ~far
    r, mid = r[keep], mid[keep]
    out = pd.DataFrame({"BH_Code": r.BH_Code.values, "depth_m": mid.values,
                        "UTM_Easting_m": r.UTM_Easting_m.values, "UTM_Northing_m": r.UTM_Northing_m.values,
                        "Ground_Elevation_m": r.Ground_Elevation_m.values,
                        "Mid_Elevation_m": (r.Ground_Elevation_m - mid).values,
                        "USCS": r.USCS.values if "USCS" in r else None, "y": r.value.values.astype(float),
                        **{c: pd.to_numeric(r[c], errors="coerce").values for c in extra}})
    clay, sand = set(S["soil_class"]["clay"]), set(S["soil_class"]["sand"])
    out["soil_clay"] = np.where(out.USCS.isin(clay), 1.0, np.where(out.USCS.isin(sand), 0.0, np.nan))
    info["samples"] = len(out)
    return out.sort_values(["BH_Code", "depth_m"]).reset_index(drop=True), info


def to_layers(samples: pd.DataFrame, S: dict) -> pd.DataFrame:
    """Mean of the samples of one borehole in each layer of `layer_m` (0-1.5 m, 1.5-3 m, ...):
    target, depth and predicted P(Clay); logged_clay_share = share of logged Clay symbols among
    the layer's Clay/Sand symbols (for information only, not an input)."""
    lm = S["layers"]["layer_m"]
    d = samples.assign(layer=np.floor(samples.depth_m / lm).astype(int))
    g = d.groupby(["BH_Code", "layer"], sort=True)
    out = g.agg(depth_m=("depth_m", "mean"), UTM_Easting_m=("UTM_Easting_m", "first"),
                UTM_Northing_m=("UTM_Northing_m", "first"), Ground_Elevation_m=("Ground_Elevation_m", "first"),
                p_clay=("p_clay", "mean"), logged_clay_share=("soil_clay", "mean"),
                y=("y", "mean"), n_samples=("y", "size")).reset_index()
    out["Mid_Elevation_m"] = out.Ground_Elevation_m - out.depth_m
    return out


# ------------------------------------------------------------ predicted soil type ----

def _v3clean_load(S: dict, paths: list[Path]) -> list:
    """Open model files saved by V3-clean (its classes live in its own helpers.py, which is loaded
    under that name only while the files are opened)."""
    import importlib.util
    import sys

    import joblib
    root = S["v3clean_root"]
    spec = importlib.util.spec_from_file_location("helpers", root / "helpers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = sys.modules.get("helpers")
    sys.modules["helpers"] = mod
    try:
        return [joblib.load(p) for p in paths]
    finally:
        if old is not None:
            sys.modules["helpers"] = old


def load_classifiers(S: dict) -> tuple[dict, object, dict]:
    """The out-of-fold classifiers {k: model} and the frozen final classifier of V3-clean."""
    c, root = S["classifier"], S["v3clean_root"]
    fold_paths = {k: root / c["fold_models"].format(k=k) for k in range(1, S["cv"]["k"] + 1)}
    missing = [str(p) for p in fold_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"V3-clean out-of-fold classifier models missing: {missing}")
    rec = json.loads((root / c["final_json"]).read_text())
    final_path = root / Path(c["final_json"]).parent / "private" / rec["model_file"]
    if sha256(final_path) != rec["model_sha256"]:
        raise RuntimeError("the V3-clean final model file has changed")
    models = _v3clean_load(S, [*fold_paths.values(), final_path])
    return dict(zip(fold_paths, models[:-1])), models[-1], {"final_model": rec["model"], "final_threshold": rec["threshold"]}


def classifier_inputs(df: pd.DataFrame, S: dict) -> np.ndarray:
    """The classifier's inputs at each sample: Easting, Northing, top elevation of its 0.5 m cell."""
    cm = S["classifier"]["cell_m"]
    top = np.floor(np.asarray(df.depth_m) / cm) * cm
    return np.c_[df.UTM_Easting_m.values, df.UTM_Northing_m.values, df.Ground_Elevation_m.values - top]


def add_p_clay(samples: pd.DataFrame, folds: pd.Series, fold_models: dict, final_model, S: dict) -> pd.Series:
    """P(Clay) of each sample: a development borehole in fold k gets the fold-k (out-of-fold)
    classifier; every other borehole (external holdout, no fold) gets the final classifier."""
    p = np.full(len(samples), np.nan)
    fold = samples.BH_Code.map(folds)
    for k, m in fold_models.items():
        sel = (fold == k).values & (samples.set == "development").values
        if sel.any():
            p[sel] = m.predict_proba(classifier_inputs(samples[sel], S))[:, 1]
    rest = np.isnan(p)
    if rest.any():
        p[rest] = final_model.predict_proba(classifier_inputs(samples[rest], S))[:, 1]
    return pd.Series(p, index=samples.index)


# ------------------------------------------------------------ map grid ----

def grid_layers(grid: pd.DataFrame, S: dict) -> pd.DataFrame:
    """The classifier grid (0.5 m cells) -> 1.5 m layers per grid column, with the same inputs as
    the training layers: mid elevation = ground - layer mid depth, p_clay = mean of its cells."""
    lm = S["layers"]["layer_m"]
    g = grid.assign(layer=np.floor(grid.depth_top_m / lm + 1e-9).astype(int))
    out = g.groupby(["X", "Y", "layer"], sort=False).agg(
        ground_m=("ground_m", "first"), p_clay=("p_clay", "mean"),
        dist_nearest_bh_m=("dist_nearest_bh_m", "first"), reliability_zone=("reliability_zone", "first")).reset_index()
    out["depth_m"] = (out.layer + 0.5) * lm
    out = out.rename(columns={"X": "UTM_Easting_m", "Y": "UTM_Northing_m"})
    out["Mid_Elevation_m"] = out.ground_m - out.depth_m
    return out


def holdout_digest(codes) -> str:
    return hashlib.sha256("\n".join(sorted(codes)).encode("utf-8")).hexdigest()


def load_v3_holdout(S: dict) -> pd.DataFrame:
    """V3's 11 external-holdout boreholes, checked against V3's lock."""
    h = pd.read_csv(S["v3_root"] / "data" / "processed" / "external_holdout.csv").set_index("BH_Code")
    lock = json.loads((S["v3_root"] / "config" / "external_holdout.lock.json").read_text())
    if holdout_digest(h.index) != lock["sha256"]:
        raise RuntimeError("V3's external-holdout list does not match V3's lock")
    return h


def assign_sets(samples: pd.DataFrame, holdout_codes, folds: pd.Series) -> pd.Series:
    """external = V3 holdout; development = in V3's folds; no_fold = neither (a borehole without
    Clay/Sand cells got no fold in V3, so it cannot be tested under V3's folds; it is left out)."""
    return pd.Series(np.select([samples.BH_Code.isin(set(holdout_codes)), samples.BH_Code.isin(folds.index)],
                               ["external", "development"], "no_fold"), index=samples.index)


# ======================================================================== 3. Models ====

class Depth:
    """Depth-only reference: the training mean in each depth slice (overall mean where a slice
    has too few samples). Uses depth only, no location."""

    def __init__(self, bin_m, min_n):
        self.bin_m, self.min_n = bin_m, min_n

    def fit(self, X, y, depth):
        b = np.floor(np.asarray(depth) / self.bin_m).astype(int)
        s = pd.DataFrame({"b": b, "y": np.asarray(y, float)}).groupby("b").y.agg(["mean", "size"])
        self.overall_ = float(np.mean(y))
        self.table_ = s["mean"][s["size"] >= self.min_n].to_dict()
        return self

    def predict(self, X, depth):
        b = np.floor(np.asarray(depth) / self.bin_m).astype(int)
        return np.array([self.table_.get(v, self.overall_) for v in b])


class Tree:
    """RF or XGB regressor; with log = true it learns log(1 + y) and predicts exp(.) - 1."""

    def __init__(self, kind, params, seed, n_jobs, log):
        self.kind, self.params, self.seed, self.n_jobs, self.log = kind, dict(params), seed, n_jobs, log

    def fit(self, X, y, depth=None):
        y = np.log1p(y) if self.log else np.asarray(y, float)
        if self.kind == "RF":
            from sklearn.ensemble import RandomForestRegressor
            if isinstance(self.params.get("max_features"), int):         # fewer inputs in property models
                self.params["max_features"] = min(self.params["max_features"], np.asarray(X).shape[1])
            self.model_ = RandomForestRegressor(**self.params, bootstrap=True, random_state=self.seed,
                                                n_jobs=self.n_jobs)
        else:
            from xgboost import XGBRegressor
            self.model_ = XGBRegressor(**self.params, tree_method="hist", random_state=self.seed,
                                       n_jobs=self.n_jobs, objective="reg:squarederror")
        self.model_.fit(X, y)
        return self

    def predict(self, X, depth=None):
        if self.kind == "RF":
            self.model_.n_jobs = 1
        p = self.model_.predict(X)
        return np.expm1(p) if self.log else p


def _fwd(y, log):
    return np.log1p(y) if log else np.asarray(y, float)


def _inv(p, log):
    return np.expm1(p) if log else p


class MLP:
    """MLP regressor; inputs and target standardised on the training data; no early stopping (its
    random validation rows would mix samples of one borehole)."""

    def __init__(self, params, seed, log):
        self.params, self.seed, self.log = dict(params), seed, log

    def fit(self, X, y, depth=None):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        p = dict(self.params)
        layers = (p.pop("n_units"),) * p.pop("n_layers")
        y = _fwd(y, self.log)
        self.xs_ = StandardScaler().fit(X)
        self.ym_, self.ysd_ = float(np.mean(y)), float(np.std(y)) or 1.0
        self.model_ = MLPRegressor(hidden_layer_sizes=layers, early_stopping=False, random_state=self.seed, **p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(self.xs_.transform(X), (y - self.ym_) / self.ysd_)
        return self

    def predict(self, X, depth=None):
        return _inv(self.model_.predict(self.xs_.transform(X)) * self.ysd_ + self.ym_, self.log)


class GW:
    """Geographically weighted RF / XGBoost regression (GWRF, GWXGB), as V3's GW classifiers.

    * Every training borehole is an anchor with a local model trained on the samples of its k nearest
      boreholes (k grows until there are `min_samples` samples, up to `max_neighbours`; otherwise
      the anchor uses the global model).
    * Kernel 'uniform': weight 1; 'bisquare': (1 - (d/b)^2)^2, b = distance to the (k+1)-th borehole.
    * A sample uses the local model of its nearest anchor, blended with a global model:
      y = alpha * y_local + (1 - alpha) * y_global.
    * Local models are trained when a prediction first needs them (each with its own fixed seed)."""

    def __init__(self, base, learner, k, alpha, kernel, n_local, n_global, S, seed, log):
        self.base, self.learner, self.k, self.alpha, self.kernel = base, dict(learner), k, alpha, kernel
        self.n_local, self.n_global, self.seed, self.log = n_local, n_global, seed, log
        self.g, self.n_jobs = S["gw"], S["run"]["n_jobs"]

    def _new(self, n_trees, seed):
        p = self.learner
        if self.base == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=n_trees, max_depth=p.get("max_depth"),
                                         min_samples_leaf=p.get("min_samples_leaf", 1),
                                         max_features=p.get("max_features", 1), random_state=seed, n_jobs=1)
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=n_trees, learning_rate=p.get("learning_rate", 0.1),
                            max_depth=p.get("max_depth", 4), min_child_weight=p.get("min_child_weight", 1.0),
                            subsample=p.get("subsample", 1.0), reg_lambda=p.get("reg_lambda", 1.0),
                            tree_method="hist", n_jobs=1, random_state=seed, objective="reg:squarederror")

    def fit(self, X, y, depth=None):
        from scipy.spatial import cKDTree
        X, y = np.asarray(X, float), _fwd(y, self.log)
        self.anchors_, inv = np.unique(np.round(X[:, :2], 2), axis=0, return_inverse=True)
        self.X_, self.y_, self.site_ = X, y, inv.ravel()
        n = len(self.anchors_)
        self.global_ = self._new(self.n_global, self.seed).fit(X, y)
        self.tree_ = cKDTree(self.anchors_)
        kq = min(self.g["max_neighbours"] + 1, n)
        d, i = self.tree_.query(self.anchors_, k=kq)
        self.dist_, self.nbr_, self.kq_ = np.atleast_2d(d), np.atleast_2d(i), kq
        counts = np.bincount(self.site_, minlength=n)
        self.size_ = np.array([self._neighbourhood(a, counts) for a in range(n)])
        self.fallback_rate_ = float(np.mean(self.size_ < 0))
        self.local_ = {}
        return self

    def _neighbourhood(self, a, counts):
        g, n = self.g, len(self.anchors_)
        k = max(g["min_boreholes"], min(self.k, n))
        while True:
            if k >= g["min_boreholes"] and counts[self.nbr_[a, :k]].sum() >= g["min_samples"]:
                return int(min(k, n))
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
        return self._new(self.n_local, self.seed + 1 + int(a)).fit(self.X_[use], self.y_[use], sample_weight=w[use])

    def predict(self, X, depth=None):
        from joblib import Parallel, delayed
        X = np.asarray(X, float)
        y_global = self.global_.predict(X)
        anchor = self.tree_.query(np.round(X[:, :2], 2))[1]
        todo = [int(a) for a in np.unique(anchor) if int(a) not in self.local_]
        fitted = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._fit_local)(a) for a in todo)
        self.local_.update(zip(todo, fitted))
        y_local = y_global.copy()
        for a in np.unique(anchor):
            f = self.local_[int(a)]
            if f is not None:
                sel = anchor == a
                y_local[sel] = f.predict(X[sel])
        return _inv(self.alpha * y_local + (1 - self.alpha) * y_global, self.log)


class Linear:
    """Ordinary least squares on the inputs (on log(1 + y) for log targets): the classical
    correlation, as a reference for the ML models."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y, depth=None):
        from sklearn.linear_model import LinearRegression
        self.model_ = LinearRegression().fit(X, _fwd(y, self.log))
        return self

    def predict(self, X, depth=None):
        return _inv(self.model_.predict(X), self.log)


def make_model(name: str, params: dict, seed: int, target: str, S: dict):
    if name == "Depth":
        return Depth(S["cv"]["depth_bin_m"], S["cv"]["depth_min_samples"])
    if name == "Linear":
        return Linear(S["targets"][target]["log"])
    p, quick, log = dict(params), S["run"]["quick"], S["targets"][target]["log"]
    if quick and "n_estimators" in p:
        p["n_estimators"] = min(p["n_estimators"], S["quick"]["trees"])
    if quick and "max_iter" in p:
        p["max_iter"] = min(p["max_iter"], 200)
    if name in ("RF", "XGB"):
        return Tree(name, p, seed, S["run"]["n_jobs"], log)
    if name == "MLP":
        return MLP(p, seed, log)
    g = S["gw"]
    k, alpha, kernel = p.pop("k_neighbours"), p.pop("alpha"), p.pop("kernel")
    if name == "GWRF":
        n_local = min(g["local_rf_trees"], 30) if quick else g["local_rf_trees"]
        return GW("rf", p, k, alpha, kernel, n_local, 3 * n_local, S, seed, log)
    n = min(p.pop("n_estimators"), g["local_xgb_rounds_max"])
    return GW("xgb", p, k, alpha, kernel, n, n, S, seed, log)


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
    f = pd.read_csv(S["v3_root"] / "data" / "folds" / f"outer_folds_{int(block_m)}m.csv").set_index("BH_Code")
    return f.fold


def block_sizes(S: dict) -> list[int]:
    d = json.loads((S["v3_root"] / "data" / "folds" / "block_size_decision.json").read_text())
    base = int(d["block_size_m"])
    return list(dict.fromkeys([base] + [int(round(base * m)) for m in S["cv"]["block_multiples"][1:]]))


def outer_split(dev: pd.DataFrame, folds: pd.Series, k: int):
    fold = dev.BH_Code.map(folds)
    return dev[fold.notna() & (fold != k)], dev[fold == k]


def inner_groups(train: pd.DataFrame, folds: pd.Series, k: int, block_m: int, S: dict):
    """Inner folds = the other 4 outer folds; V3's backup inner folds if one is too small."""
    ic = S["inner_cv"]
    groups = train.BH_Code.map(folds).astype(int).values.copy()
    groups[groups == 0] = -1

    def ok(g):
        ids = sorted(set(g[g > 0]))
        return len(ids) >= 2 and all(train[g == i].BH_Code.nunique() >= ic["min_boreholes"]
                                     and (g == i).sum() >= ic["min_samples"] for i in ids)

    if ok(groups):
        return groups, "outer_folds"
    p = S["v3_root"] / "data" / "folds" / f"inner_fallback_{int(block_m)}m.csv"
    if p.exists():
        fb = pd.read_csv(p)
        m = fb[fb.outer_fold == k].set_index("BH_Code").inner_fold
        g = train.BH_Code.map(m).fillna(-1).astype(int).values
        if len(set(g[g > 0])) >= 2:
            return g, "backup_folds"
    return groups, "outer_folds_small"


def n_trials(name: str, S: dict) -> int:
    return 0 if name in NO_TUNING else (S["quick"]["n_trials"] if S["run"]["quick"] else S["cv"]["n_trials"][name])


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def tune(name, target, train, groups, seed, tpe_seed, S, on_trial=None) -> dict:
    """Optuna (TPE) picks the hyperparameters with the lowest inner out-of-fold RMSE (original scale)."""
    if name in NO_TUNING:
        return {"params": {}, "inner_rmse": np.nan, "trials": pd.DataFrame()}
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y, d = design(train), train.y.values, train.depth_m.values
    ids = sorted(set(groups[groups > 0]))
    scored = np.isin(groups, ids)

    def objective(trial):
        params = suggest(trial, S["space"][name])
        oof = np.full(len(train), np.nan)
        for g in ids:
            val = groups == g
            oof[val] = make_model(name, params, seed, target, S).fit(X[~val], y[~val], d[~val]).predict(X[val], d[val])
        return rmse(y[scored], oof[scored])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=tpe_seed))
    study.optimize(objective, n_trials=n_trials(name, S), n_jobs=1,
                   callbacks=[lambda study, trial: on_trial()] if on_trial else None)
    return {"params": study.best_params, "inner_rmse": study.best_value,
            "trials": study.trials_dataframe(attrs=("number", "value", "params", "duration"))}


def run_list(S: dict) -> list[dict]:
    """Every target x (RF, XGB, MLP, GWRF, GWXGB, Depth) on the main block size (106 m)."""
    b = block_sizes(S)[0]
    return [{"target": t, "model": m, "block_m": b, "tag": f"{t}_{b}m_{m}"}
            for t in S["cv"]["targets"] for m in [*S["cv"]["models"], "Depth"]]


def run_fold(run: dict, dev: pd.DataFrame, k: int, S: dict, on_trial=None) -> dict:
    """One outer fold of one run: tune on the training boreholes, refit, predict fold k."""
    t0 = time.time()
    name = run["model"]
    folds = load_folds(S, run["block_m"])
    train, test = outer_split(dev, folds, k)
    groups, source = inner_groups(train, folds, k, run["block_m"], S)
    seed = seed_for(S, run["tag"], k)
    best = tune(name, run["target"], train, groups, seed, seed_for(S, run["tag"], k, "tpe"), S, on_trial)
    model = make_model(name, best["params"], seed, run["target"], S).fit(design(train), train.y.values, train.depth_m.values)
    pred = test[list(dict.fromkeys(["BH_Code", "depth_m", *COORDS, *FEATURES, "y"]))].copy()
    pred["y_pred"] = model.predict(design(test), test.depth_m.values) if len(test) else []
    if len(test):
        tb = test.drop_duplicates("BH_Code").set_index("BH_Code")[COORDS]
        trb = train.drop_duplicates("BH_Code").set_index("BH_Code")[COORDS]
        pred["dist_nearest_train_m"] = pred.BH_Code.map(pd.Series(nearest_distance(tb.values, trb.values), index=tb.index)).values
    pred = pred.assign(outer_fold=k, target=run["target"], block_m=run["block_m"], model=name)
    info = {**run, "outer_fold": k, "inner_folds": source, "params": best["params"], "inner_rmse": best["inner_rmse"],
            "train_boreholes": int(train.BH_Code.nunique()), "train_samples": len(train),
            "test_boreholes": int(test.BH_Code.nunique()), "test_samples": len(test),
            "seconds": round(time.time() - t0, 1)}
    return {"predictions": pred, "info": info, "trials": best["trials"], "model": model}


def fit_final(name: str, target: str, dev: pd.DataFrame, block_m: int, S: dict, on_trial=None) -> dict:
    """Final model: tune on ALL development boreholes (groups = the 5 outer folds), then train on all."""
    folds = load_folds(S, block_m)
    groups = dev.BH_Code.map(folds).astype(int).values.copy()
    groups[groups == 0] = -1
    tag = f"final_{target}_{block_m}m_{name}"
    seed = seed_for(S, tag)
    best = tune(name, target, dev, groups, seed, seed_for(S, tag, "tpe"), S, on_trial)
    model = make_model(name, best["params"], seed, target, S).fit(design(dev), dev.y.values, dev.depth_m.values)
    return {**best, "model": model}


# ================================================ 5. Metrics and model selection ====

METRICS = ["rmse", "mae", "r2", "bias", "spearman"]
RUN_KEYS = ["target", "block_m", "model"]


def scores(y, p) -> dict:
    """RMSE, MAE, R² (1 - SSE/SST around the test mean), bias (mean of prediction - observed),
    Spearman rank correlation."""
    from scipy.stats import spearmanr
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    if n == 0:
        return {"n": 0, **{m: np.nan for m in METRICS}}
    sst = np.sum((y - y.mean()) ** 2)
    return {"n": n, "rmse": rmse(y, p), "mae": float(np.mean(np.abs(y - p))),
            "r2": float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan,
            "bias": float(np.mean(p - y)),
            "spearman": float(spearmanr(y, p).statistic) if n > 2 and np.ptp(y) > 0 and np.ptp(p) > 0 else np.nan}


def score_table(preds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in preds.groupby(keys, sort=True, dropna=False):
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **scores(g.y, g.y_pred)})
    return pd.DataFrame(rows)


def fold_scores(preds: pd.DataFrame) -> pd.DataFrame:
    return score_table(preds, RUN_KEYS + ["outer_fold"])


def summary(fold_table: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and minimum over the outer folds, and the skill against the depth-only reference
    (1 - RMSE / RMSE_depth, from the mean fold RMSE; > 0 = better than depth alone)."""
    g = fold_table.groupby(RUN_KEYS)[METRICS]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.min().add_suffix("_min"))
    out["n_folds"] = fold_table.groupby(RUN_KEYS).size()
    out = out.reset_index()
    ref = out[out.model == "Depth"].set_index(["target", "block_m"]).rmse_mean
    out["skill_vs_depth"] = 1 - out.rmse_mean / out.set_index(["target", "block_m"]).index.map(ref).values
    return out


def select_model(summ: pd.DataFrame, target: str, main_block: int) -> dict:
    """Rule, decided before the results, on the main block size, per target:
      1. best = lowest mean fold RMSE (the depth-only reference competes too);
      2. among models within one standard error of the best: lowest fold SD of RMSE,
         then lowest mean MAE (the same shape as V3's 1-SE rule)."""
    s = summ[(summ.target == target) & (summ.block_m == main_block)].copy()
    best = s.loc[s.rmse_mean.idxmin()]
    se = best.rmse_sd / np.sqrt(best.n_folds) if best.n_folds > 1 else 0.0
    near = s[s.rmse_mean <= best.rmse_mean + se].sort_values(["rmse_sd", "mae_mean"])
    return {"target": target, "selected": near.model.iloc[0], "lowest_mean": best.model,
            "best_rmse_mean": float(best.rmse_mean), "one_se": float(se), "within_1se": list(near.model)}


def distance_table(preds: pd.DataFrame, edges=(0, 50, 100, 200, 300, 500, 1000, np.inf)) -> pd.DataFrame:
    labels = [f"{a:g}-{b:g} m" if np.isfinite(b) else f">{a:g} m" for a, b in zip(edges[:-1], edges[1:])]
    d = preds.assign(distance_bin=pd.cut(preds.dist_nearest_train_m, list(edges), right=False, labels=labels).astype(str))
    return score_table(d, RUN_KEYS + ["distance_bin"])


def depth_table(preds: pd.DataFrame, band_m: float = 3.0) -> pd.DataFrame:
    d = preds.assign(depth_band=(np.floor(preds.depth_m / band_m) * band_m).map(lambda a: f"{a:g}-{a + band_m:g} m"))
    return score_table(d, RUN_KEYS + ["depth_band"])
