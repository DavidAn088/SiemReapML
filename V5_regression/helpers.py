"""V5 regression helper functions: continuous soil parameters under V3's spatial CV.

The same Downtown boreholes, blockCV folds (106 / 212 / 318 m) and 11 external-holdout boreholes
as V3 (read from the V3 folder). The targets are measured values on sample records, not 0.5 m
cells, so there is no labelling or gap filling: one row = one sample with a value.

Sections
  1. Settings and files
  2. Boreholes -> samples of one target
  3. Models (RF, XGB, depth-only reference)
  4. Nested spatial cross-validation
  5. Metrics and model selection

Inputs: FEATURES (Easting, Northing, mid elevation); models named "<RF|XGB>_soil" also get the
soil type (Clay = 1, Sand = 0) as a 4th input, as Khamjan et al. (2026):
  * training: the LOGGED type of the sample (USCS group; 0.5 when the record is neither group);
  * outer test: the type PREDICTED by the classifier for that borehole when it was out of fold
    (V3's out-of-fold XGB predictions, same folds), so classification errors are counted.
    The score with the logged type is also kept (y_pred_logged), to show the difference.
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

FEATURES = ["UTM_Easting_m", "UTM_Northing_m", "Mid_Elevation_m"]
SOIL = "soil_clay"          # 4th input of the *_soil models


def kind(name: str) -> str:
    """RF_soil -> RF"""
    return name.split("_")[0]


def uses_soil(name: str) -> bool:
    return name.endswith("_soil")


def design(df: pd.DataFrame, name: str, soil=None) -> np.ndarray:
    """Model inputs. For *_soil models the soil column is `soil` if given, else the logged type."""
    X = df[FEATURES].values
    if not uses_soil(name):
        return X
    s = df["clay_logged"].fillna(0.5).values if soil is None else np.asarray(soil, float)
    return np.c_[X, s]


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
    return S


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


def make_samples(records: pd.DataFrame, target: str, S: dict) -> tuple[pd.DataFrame, dict]:
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
    if t.get("uscs"):                                    # e.g. Su on clay samples only
        other = ~r.USCS.isin(t["uscs"]) & ~long_ & ~outside & ~bad
        info["dropped_other_uscs"] = int(other.sum())
        bad = bad | other
    info |= {"dropped_longer_than_3m": int(long_.sum()), "dropped_outside_depth": int((outside & ~long_).sum()),
             "dropped_out_of_range": int((bad & ~long_ & ~outside).sum())}
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
                        "USCS": r.USCS.values if "USCS" in r else None, "y": r.value.values.astype(float)})
    lab = S.get("soil_type", {})
    clay, sand = set(lab.get("clay", ["CL", "CH"])), set(lab.get("sand", ["SC", "SM", "SP"]))
    out["clay_logged"] = np.where(out.USCS.isin(clay), 1.0, np.where(out.USCS.isin(sand), 0.0, np.nan))
    info["samples"] = len(out)
    return out.sort_values(["BH_Code", "depth_m"]).reset_index(drop=True), info


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


def make_model(name: str, params: dict, seed: int, target: str, S: dict):
    if name == "Depth":
        return Depth(S["cv"]["depth_bin_m"], S["cv"]["depth_min_samples"])
    p = dict(params)
    if S["run"]["quick"] and "n_estimators" in p:
        p["n_estimators"] = min(p["n_estimators"], S["quick"]["trees"])
    return Tree(kind(name), p, seed, S["run"]["n_jobs"], S["targets"][target]["log"])


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
    return 0 if name == "Depth" else (S["quick"]["n_trials"] if S["run"]["quick"] else S["cv"]["n_trials"][kind(name)])


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def tune(name, target, train, groups, seed, tpe_seed, S, on_trial=None) -> dict:
    """Optuna (TPE) picks the hyperparameters with the lowest inner out-of-fold RMSE (original scale)."""
    if name == "Depth":
        return {"params": {}, "inner_rmse": np.nan, "trials": pd.DataFrame()}
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y, d = design(train, name), train.y.values, train.depth_m.values
    ids = sorted(set(groups[groups > 0]))
    scored = np.isin(groups, ids)

    def objective(trial):
        params = suggest(trial, S["space"][kind(name)])
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
    """Every target x block size x model: RF, XGB, RF_soil, XGB_soil (not for targets limited to
    one soil group, e.g. Su on clay) and the depth reference. Quick: main block only."""
    sizes = block_sizes(S)
    sizes = sizes[:1] if S["run"]["quick"] else sizes
    runs = []
    for t in S["cv"]["targets"]:
        soil = [f"{m}_soil" for m in S["cv"]["models"]] if S["cv"].get("soil_type_models") and not S["targets"][t].get("uscs") else []
        runs += [{"target": t, "model": m, "block_m": b, "tag": f"{t}_{b}m_{m}"}
                 for b in sizes for m in [*S["cv"]["models"], *soil, "Depth"]]
    return runs


# ------------------------------------------------------------ predicted soil type ----

def cell_of(depth_m, S: dict) -> np.ndarray:
    """Index of V3's 0.5 m cell (0-15 m) that holds this depth."""
    c = S["soil_type"]
    return np.clip(np.floor(np.asarray(depth_m) / c["cell_m"]), 0, round(15 / c["cell_m"]) - 1).astype(int)


_CV_SOIL = {}


def v3_cv_soil(S: dict, block_m: int) -> pd.DataFrame:
    """V3's out-of-fold Clay/Sand predictions of the classifier (soil_type.cv_model, no purge) at this
    block size: BH_Code, cell, outer_fold, soil_pred."""
    key = (str(S["v3_root"]), block_m)
    if key not in _CV_SOIL:
        c = S["soil_type"]
        p = pd.read_csv(S["v3_root"] / c["v3_cv_predictions"])
        p = p[(p.model == c["cv_model"]) & (p.block_m == block_m) & (p.purge_m == 0)]
        if "scenario" in p:
            p = p[p.scenario == "A"]
        _CV_SOIL[key] = p[["BH_Code", "cell", "outer_fold", "y_pred"]].rename(columns={"y_pred": "soil_pred"})
    return _CV_SOIL[key]


def lookup_soil(samples: pd.DataFrame, cells: pd.DataFrame, S: dict) -> tuple[np.ndarray, dict]:
    """Predicted type of each sample from cell predictions (BH_Code, cell, soil_pred): the sample's
    own cell; else the nearest predicted cell of the same borehole within nearest_max_m; else 0.5."""
    max_cells = int(round(S["soil_type"]["nearest_max_m"] / S["soil_type"]["cell_m"]))
    by_bh = {code: g.set_index("cell").soil_pred for code, g in cells.groupby("BH_Code")}
    out, n = np.full(len(samples), 0.5), {"own_cell": 0, "nearest_cell": 0, "unknown": 0}
    for i, (code, c) in enumerate(zip(samples.BH_Code.values, cell_of(samples.depth_m.values, S))):
        g = by_bh.get(code)
        if g is not None and c in g.index:
            out[i] = g[c]; n["own_cell"] += 1
            continue
        if g is not None and len(g):
            d = np.abs(g.index.values - c)
            if d.min() <= max_cells:
                out[i] = g.iloc[int(np.argmin(d))]; n["nearest_cell"] += 1
                continue
        n["unknown"] += 1
    return out, n


def load_classifier(S: dict):
    """The thesis's final Clay/Sand classifier (frozen, from V3-clean notebook 05) and its threshold.
    V3-clean saved it with its own helpers.py, which is loaded here under that name only while the
    file is opened."""
    import importlib.util
    import sys

    import joblib
    root = Path(S["soil_type"]["v3clean_folder"])
    rec = json.loads((root / "outputs" / "05_final_model" / "final_model.json").read_text())
    path = root / "outputs" / "05_final_model" / "private" / rec["model_file"]
    if sha256(path) != rec["model_sha256"]:
        raise RuntimeError("the V3-clean final model file has changed")
    spec = importlib.util.spec_from_file_location("helpers", root / "helpers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = sys.modules.get("helpers")
    sys.modules["helpers"] = mod
    try:
        model = joblib.load(path)
    finally:
        if old is not None:
            sys.modules["helpers"] = old
    return model, float(rec["threshold"]), rec["model"]


def classify(samples: pd.DataFrame, model, threshold: float, S: dict) -> np.ndarray:
    """Clay (1) / Sand (0) from the classifier at each sample's 0.5 m cell (its inputs: Easting,
    Northing, top elevation of the cell = ground elevation - cell top depth)."""
    top = cell_of(samples.depth_m.values, S) * S["soil_type"]["cell_m"]
    X = np.c_[samples.UTM_Easting_m.values, samples.UTM_Northing_m.values, samples.Ground_Elevation_m.values - top]
    return (model.predict_proba(X)[:, 1] >= threshold).astype(float)


# ------------------------------------------------------------ one outer fold ----

def run_fold(run: dict, dev: pd.DataFrame, k: int, S: dict, on_trial=None) -> dict:
    """One outer fold of one run: tune on the training boreholes, refit, predict fold k.
    *_soil models: trained with the logged soil type; tested with the classifier's out-of-fold
    type (y_pred) and, for comparison, with the logged type (y_pred_logged)."""
    t0 = time.time()
    name = run["model"]
    folds = load_folds(S, run["block_m"])
    train, test = outer_split(dev, folds, k)
    groups, source = inner_groups(train, folds, k, run["block_m"], S)
    seed = seed_for(S, run["tag"], k)
    best = tune(name, run["target"], train, groups, seed, seed_for(S, run["tag"], k, "tpe"), S, on_trial)
    model = make_model(name, best["params"], seed, run["target"], S).fit(
        design(train, name), train.y.values, train.depth_m.values)
    pred = test[["BH_Code", "depth_m", *FEATURES, "y"]].copy()
    soil_info = {}
    if uses_soil(name) and len(test):
        cv = v3_cv_soil(S, run["block_m"])
        soil, soil_info = lookup_soil(test, cv[cv.outer_fold == k], S)
        pred["soil_pred"], pred["soil_logged"] = soil, test.clay_logged.values
        pred["y_pred"] = model.predict(design(test, name, soil), test.depth_m.values)
        pred["y_pred_logged"] = model.predict(design(test, name), test.depth_m.values)
    else:
        pred["y_pred"] = model.predict(design(test, name), test.depth_m.values) if len(test) else []
    if len(test):
        tb = test.drop_duplicates("BH_Code").set_index("BH_Code")[FEATURES[:2]]
        trb = train.drop_duplicates("BH_Code").set_index("BH_Code")[FEATURES[:2]]
        pred["dist_nearest_train_m"] = pred.BH_Code.map(pd.Series(nearest_distance(tb.values, trb.values), index=tb.index)).values
    pred = pred.assign(outer_fold=k, target=run["target"], block_m=run["block_m"], model=name)
    info = {**run, "outer_fold": k, "inner_folds": source, "params": best["params"], "inner_rmse": best["inner_rmse"],
            "train_boreholes": int(train.BH_Code.nunique()), "train_samples": len(train),
            "test_boreholes": int(test.BH_Code.nunique()), "test_samples": len(test),
            **{f"soil_{a}": v for a, v in soil_info.items()}, "seconds": round(time.time() - t0, 1)}
    return {"predictions": pred, "info": info, "trials": best["trials"], "model": model}


def fit_final(name: str, target: str, dev: pd.DataFrame, block_m: int, S: dict, on_trial=None) -> dict:
    """Final model: tune on ALL development boreholes (groups = the 5 outer folds), then train on all
    (*_soil models with the logged soil type)."""
    folds = load_folds(S, block_m)
    groups = dev.BH_Code.map(folds).astype(int).values.copy()
    groups[groups == 0] = -1
    tag = f"final_{target}_{block_m}m_{name}"
    seed = seed_for(S, tag)
    best = tune(name, target, dev, groups, seed, seed_for(S, tag, "tpe"), S, on_trial)
    model = make_model(name, best["params"], seed, target, S).fit(design(dev, name), dev.y.values, dev.depth_m.values)
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
