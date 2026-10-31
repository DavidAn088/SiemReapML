"""V1_Li helper functions, used by the notebooks.

The ideas of Li, Zhang & He (2026, Algorithms 19:128) - the VA-HDW features and the GBARNN model -
applied to the Siem Reap boreholes, tuned and validated with nested spatial block cross-validation.

Sections
  1. Settings and files
  2. Boreholes -> 0.5 m cells (labels and gap filling)
  3. VA-HDW features
  4. Models (RF, XGB, MLP, GWRF, GWXGB, GBARNN)
  5. Nested spatial cross-validation
  6. Metrics and model selection
  7. 3D map

Every model is used the same way:
  model = make_model(name, features, params, seed, S)    features = "XYZ" or "XYZ_VAHDW"
  model.fit(cells, weights)   cells = a table of 0.5 m cells with the inputs and the label y
  model.p_clay(cells)         probability of Clay for each cell
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import tomllib
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# harmless notice from tqdm when the Jupyter widget package is missing (the notebooks use text progress bars)
warnings.filterwarnings("ignore", message="IProgress not found")

FEATURES = ["UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m"]


# =============================================================== 1. Settings and files ====

def find_root() -> Path:
    """The V1_Li folder: the first folder (here or above) that contains settings.toml."""
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "settings.toml").exists():
            return p
    raise FileNotFoundError("settings.toml not found: open the notebooks from the V1_Li folder")


def load_settings(root: Path | None = None) -> dict:
    root = root or find_root()
    S = tomllib.loads((root / "settings.toml").read_text(encoding="utf-8"))
    if os.environ.get("V1_QUICK") == "1":          # used for software tests only
        S["run"]["quick"] = True
    S["root"] = root
    return S


def out_dir(S: dict, step: str, private: bool = False) -> Path:
    """outputs/<step>/ for tables and figures without borehole locations,
    outputs/<step>/private/ for anything that shows boreholes (never shared or committed).
    Quick runs write to outputs_quick/ so they can never mix with real results."""
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
    """Path of a raw file; stops if its SHA-256 differs from settings.toml."""
    p = S["root"] / S["files"][key]
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing (see data/raw/README.md)")
    want = S["files"].get(key.replace("_csv", "_sha256"))
    if want and sha256(p) != want:
        raise ValueError(f"{p.name} is not the expected file (SHA-256 differs from settings.toml)")
    return p


def folds_dir(S: dict) -> Path:
    d = S["root"] / ("data/folds_quick" if S["run"]["quick"] else "data/folds")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_r_settings(S: dict) -> Path:
    """The settings the R scripts need, as JSON (R reads it with jsonlite)."""
    rel = lambda p: p.relative_to(S["root"]).as_posix()
    r = {"blockcv": {**S["blockcv"], "seed": S["run"]["seed"]}, "epsg": S["files"]["epsg"], "area": S["area"]["area"],
         "processed_dir": rel(processed_dir(S)), "folds_dir": rel(folds_dir(S)), "out_dir": rel(out_dir(S, "02_blockcv"))}
    p = S["root"] / "data" / "r_settings.json"
    p.write_text(json.dumps(r, indent=1))
    return p


def seed_of(S: dict, *parts) -> int:
    """A fixed random seed for one model fit, e.g. seed_of(S, "XGB", 1)."""
    return zlib.crc32("|".join(map(str, (S["run"]["seed"], *parts))).encode()) % (2**31)


def area_regions(S: dict) -> list[str]:
    lab = S["labels"]
    return {"downtown": [lab["region_downtown"]],
            "downtown_5km": [lab["region_downtown"], lab["region_ring"]]}[S["area"]["area"]]


def polygon(S: dict, key: str):
    import geopandas as gpd
    import shapely

    g = gpd.read_file(S["root"] / S["files"][key])
    return shapely.force_2d(g.to_crs(S["files"]["epsg"]).geometry.union_all())


def area_polygon(S: dict):
    """Downtown, or Downtown + the 5 km ring."""
    import shapely

    if S["area"]["area"] == "downtown":
        return polygon(S, "downtown_shp")
    return shapely.union(polygon(S, "downtown_shp"), polygon(S, "ring_shp"))


# =============================================== 2. Boreholes -> 0.5 m cells ====

def read_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.insert(0, "csv_line", np.arange(len(df)) + 2)   # line number in the file (header = line 1)
    return df


def load_boreholes(S: dict) -> pd.DataFrame:
    """Sample records of the boreholes in the chosen area (minus manually excluded ones)."""
    df = read_records(check_file(S, "boreholes_csv"))
    df = df[df.Region.isin(area_regions(S))]
    flags = S["root"] / S["files"]["manual_flags"]
    if flags.exists():
        f = pd.read_csv(flags)
        out = f.loc[f.exclude_from_training.astype(str).str.lower().isin(["1", "true", "yes"]), "BH_Code"]
        df = df[~df.BH_Code.isin(out)]
    return df.reset_index(drop=True)


def load_new_external(S: dict) -> tuple[pd.DataFrame, dict]:
    """The 10 new external boreholes. Boreholes that are also in the main file are removed
    (they are training data). All remaining ones must lie inside Downtown."""
    import shapely

    df = read_records(check_file(S, "new_external_csv"))
    main = read_records(check_file(S, "boreholes_csv"))
    dup = sorted(set(df.BH_Code) & set(main.BH_Code))
    df = df[~df.BH_Code.isin(dup)].copy()
    bh = df.groupby("BH_Code")[["UTM_Easting_m", "UTM_Northing_m"]].first()
    inside = shapely.contains_xy(polygon(S, "downtown_shp"), bh.UTM_Easting_m.values, bh.UTM_Northing_m.values)
    if not inside.all():
        raise ValueError(f"{int((~inside).sum())} new borehole(s) lie outside Downtown")
    df["Region"] = S["labels"]["region_downtown"]
    return df.reset_index(drop=True), {"boreholes_in_file": len(bh) + len(dup), "already_in_main_file": len(dup),
                                       "new_boreholes": len(bh), "records": len(df)}


def _cell_edges(S: dict):
    c = S["cells"]
    edges = np.arange(c["top_m"], c["bottom_m"] + 1e-9, c["cell_m"])
    return edges[:-1], edges[1:]


def _overlap(frm, to, tops, bots) -> np.ndarray:
    """Metres of each record (columns) inside each cell (rows)."""
    return np.clip(np.minimum(to[None, :], bots[:, None]) - np.maximum(frm[None, :], tops[:, None]), 0, None)


def make_cells(records: pd.DataFrame, S: dict) -> tuple[pd.DataFrame, dict]:
    """Label each 0.5 m cell from the sample records that overlap it.

    A cell gets Clay (1) or Sand (0) when records of that class cover at least `min_coverage`
    of it and no record of the other class touches it. Records with another symbol, no symbol,
    or longer than `max_sample_length_m` give no label."""
    c, lab = S["cells"], S["labels"]
    clay, sand = set(lab["clay"]), set(lab["sand"])
    tops, bots = _cell_edges(S)
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
        for j in np.where(cov_clay + cov_sand > 0)[0]:
            if cov_clay[j] > 0 and cov_sand[j] > 0:
                conflict += 1
                continue
            if max(cov_clay[j], cov_sand[j]) / c["cell_m"] < c["min_coverage"] - 1e-9:
                low += 1
                continue
            rows.append({"BH_Code": code, "cell": int(j), "depth_top_m": tops[j], "UTM_Easting_m": m.E,
                         "UTM_Northing_m": m.N, "Ground_Elevation_m": m.G, "Top_Elevation_m": m.G - tops[j],
                         "Region": m.Region, "y": int(cov_clay[j] > 0), "measured": 1})
    cells = pd.DataFrame(rows, columns=["BH_Code", "cell", "depth_top_m", "UTM_Easting_m", "UTM_Northing_m",
                                        "Ground_Elevation_m", "Top_Elevation_m", "Region", "y", "measured"])
    return cells, {"measured_cells": len(cells), "boreholes": cells.BH_Code.nunique(),
                   "cells_dropped_both_classes": conflict, "cells_dropped_low_coverage": low}


def fill_gaps(cells: pd.DataFrame, records: pd.DataFrame, S: dict) -> tuple[pd.DataFrame, dict]:
    """Fill the unsampled cells between two labelled cells of the same class (measured = 0).

    A gap is filled only when it is at most `fill_max_gap_m` thick and no record touches it.
    With unclassified_blocks_fill = false (the student's rule), a record WITHOUT a USCS symbol
    (e.g. an SS sample that was never classified) does not count, so it is filled through.
    Nothing is filled above the first or below the last labelled cell of a borehole."""
    c = S["cells"]
    tops, bots = _cell_edges(S)
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
    out = out.sort_values(["BH_Code", "cell"]).reset_index(drop=True)
    return out, {**n, "filled_cells": len(new)}


def borehole_table(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per borehole: location, region, number of cells and clay share."""
    return cells.groupby("BH_Code").agg(UTM_Easting_m=("UTM_Easting_m", "first"),
                                        UTM_Northing_m=("UTM_Northing_m", "first"),
                                        Region=("Region", "first"), cells=("y", "size"),
                                        measured=("measured", "sum"), clay_share=("y", "mean")).reset_index()


def slice_table(cells: pd.DataFrame, slice_m: float, S: dict) -> pd.DataFrame:
    """Clay share of each borehole in each depth slice (input of blockCV). Only boreholes with
    cells in every slice are kept, because blockCV cannot use empty values."""
    s = (cells.depth_top_m // slice_m).astype(int)
    t = cells.assign(slice=s).pivot_table(index="BH_Code", columns="slice", values="y", aggfunc="mean")
    n_slices = int(np.ceil((S["cells"]["bottom_m"] - S["cells"]["top_m"]) / slice_m))
    t = t.reindex(columns=range(n_slices))
    t.columns = [f"clay_{slice_m * i:g}_{min(slice_m * (i + 1), S['cells']['bottom_m']):g}m" for i in t.columns]
    t = t.dropna()
    xy = cells.groupby("BH_Code")[["UTM_Easting_m", "UTM_Northing_m"]].first()
    return xy.join(t, how="inner").reset_index()


def nearest_distance(from_xy, to_xy) -> np.ndarray:
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(to_xy, float)).query(np.asarray(from_xy, float))[0]


# ============================================================== 3. VA-HDW features ====
# After Li, Zhang & He (2026), Algorithms 19:128 ("Vertical Alignment - Horizontal Distance Weighting").
# For a cell (x, y, z) and a reference borehole i:
#   d   = horizontal distance from the cell to borehole i
#   Phi = class (Clay 1 / Sand 0) of borehole i at the cell's elevation z (vertical alignment);
#         above the first / below the last labelled cell of the borehole: its first / last class
#   S   = influence coefficient 2 * min(z_top - z, z - z_bottom) / (z_top - z_bottom) of the layer
#         (run of one class) of borehole i that contains z: 1 in the middle of a layer, 0 at its
#         boundaries and outside the borehole
# Reference boreholes = the boreholes of the training cells of each fit, so a model never uses the
# labels of a test borehole. A cell's own borehole is masked (d = 20 km, Phi = 0.5, S = 0).

class References:
    """The reference boreholes of one fit, as continuous class columns (a cell without a label
    takes the class of the nearest labelled cell of the borehole: the boring-log reading)."""

    def __init__(self, cells: pd.DataFrame, S: dict):
        c = S["cells"]
        self.cell_m = c["cell_m"]
        n_cells = int(round((c["bottom_m"] - c["top_m"]) / c["cell_m"]))
        bh = cells.groupby("BH_Code", sort=True).agg(E=("UTM_Easting_m", "first"), N=("UTM_Northing_m", "first"),
                                                     G=("Ground_Elevation_m", "first"), clay=("y", "mean"))
        self.codes = np.asarray(bh.index, dtype=object)
        self.xy = bh[["E", "N"]].values.astype(float)
        self.ground = bh.G.values.astype(float)
        self.clay_share = bh.clay.values.astype(float)
        n = len(bh)
        self.cls = np.zeros((n, n_cells))
        self.top = np.zeros((n, n_cells))           # depth of the top of the layer that contains each cell
        self.bot = np.zeros((n, n_cells))           # depth of its bottom
        self.first = np.zeros(n, int)
        self.last = np.zeros(n, int)
        for i, (code, g) in enumerate(cells.groupby("BH_Code", sort=True)):
            lab = np.full(n_cells, np.nan)
            lab[g.cell.values] = g.y.values
            idx = np.where(~np.isnan(lab))[0]
            self.first[i], self.last[i] = idx[0], idx[-1]
            nearest = idx[np.abs(np.arange(n_cells)[:, None] - idx[None, :]).argmin(axis=1)]
            col = lab[nearest]
            self.cls[i] = col
            start, end = idx[0], idx[-1] + 1           # layers = runs of one class, inside the borehole only
            for j in range(start + 1, end + 1):
                if j == end or col[j] != col[start]:
                    self.top[i, start:j], self.bot[i, start:j] = start * self.cell_m, j * self.cell_m
                    start = j

    def features(self, cells: pd.DataFrame, S: dict):
        """d (km), Phi and S of every cell (rows) for every reference borehole (columns)."""
        z = cells.Top_Elevation_m.values - self.cell_m / 2                     # cell-centre elevation
        xy = cells[["UTM_Easting_m", "UTM_Northing_m"]].values.astype(float)
        d = np.sqrt(((xy[:, None, :] - self.xy[None, :, :]) ** 2).sum(axis=2)) / 1000.0
        depth = self.ground[None, :] - z[:, None]                               # depth of z in each borehole
        j = np.floor(depth / self.cell_m).astype(int)
        jc = np.clip(j, self.first[None, :], self.last[None, :])
        rows = np.arange(len(self.codes))[None, :]
        phi = self.cls[rows, jc]
        top, bot = self.top[rows, jc], self.bot[rows, jc]
        inside = (j >= self.first[None, :]) & (j <= self.last[None, :])
        s = np.where(inside, 2 * np.minimum(depth - top, bot - depth) / (bot - top), 0.0)
        s = np.clip(s, 0.0, 1.0)
        if "BH_Code" in cells:                                                 # mask the cell's own borehole
            own = np.asarray(cells.BH_Code, dtype=object)[:, None] == self.codes[None, :]
            d = np.where(own, S["features"]["masked_distance_km"], d)
            phi = np.where(own, 0.5, phi)
            s = np.where(own, 0.0, s)
        return d, phi, s


def feature_names(features: str, refs: References | None) -> list[str]:
    names = list(FEATURES)
    if features == "XYZ_VAHDW":
        for kind in ("d", "phi", "s"):
            names += [f"{kind}_{i}" for i in range(len(refs.codes))]
    return names


# ======================================================================== 4. Models ====

def balanced_weights(y) -> np.ndarray:
    """Weights that give Clay and Sand the same total weight."""
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return np.ones(len(y))
    return np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))


def _weights(y, w, class_weight) -> np.ndarray:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return w * balanced_weights(y) if class_weight == "balanced" else w


class SklearnBase:
    """RF, XGB or MLP on a feature matrix."""

    def __init__(self, name: str, params: dict, seed: int, S: dict):
        self.name, self.params, self.seed, self.S = name, dict(params), seed, S

    def _build(self):
        p, S = dict(self.params), self.S
        p.pop("class_weight", None)
        quick = S["run"]["quick"]
        if quick and "n_estimators" in p:
            p["n_estimators"] = min(p["n_estimators"], S["quick"]["max_trees"])
        if self.name == "RF":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**p, random_state=self.seed, n_jobs=S["run"]["n_jobs"])
        if self.name == "XGB":
            from xgboost import XGBClassifier
            return XGBClassifier(**p, tree_method="hist", eval_metric="logloss", random_state=self.seed,
                                 n_jobs=S["run"]["n_jobs"])
        if self.name == "MLP":
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            layers = (p.pop("n_units"),) * p.pop("n_layers")
            if quick:
                p["max_iter"] = min(p["max_iter"], S["quick"]["mlp_max_iter"])
            # no early stopping: its random validation rows would mix cells of one borehole
            return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=layers, early_stopping=False,
                                                                 random_state=self.seed, **p))
        raise ValueError(self.name)

    def fit(self, X, y, w=None, refs=None):
        from sklearn.exceptions import ConvergenceWarning

        self._n_cols = X.shape[1]
        w = _weights(y, w, self.params.get("class_weight", "none"))
        self.model_ = self._build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if self.name == "MLP":
                self.model_.fit(X, y, mlpclassifier__sample_weight=w)
            else:
                self.model_.fit(X, y, sample_weight=w)
        if self.name == "RF":
            self.model_.n_jobs = 1     # single-threaded prediction gives identical results on every run
        return self

    def predict(self, X) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]


class GWBase:
    """Geographically weighted RF / XGBoost (GWRF, GWXGB) on a feature matrix whose first two
    columns are Easting and Northing (they choose the local neighbourhoods).

    After Georganos et al. (2021) and Grekousis (2025), adapted to Clay/Sand classification:
      * every training borehole is an "anchor" with its own local model, trained on the cells
        of its k nearest boreholes (k is tuned; it grows until both classes have
        `min_cells_per_class` cells, up to `max_neighbours`; otherwise the global model is used);
      * kernel 'uniform' gives every neighbour weight 1, 'bisquare' gives (1 - (d/b)^2)^2;
      * a cell is predicted by the local model of its nearest anchor, blended with a global
        model trained on all cells:  P = alpha * P_local + (1 - alpha) * P_global.
    Local models are trained only when a prediction needs them, then kept."""

    def __init__(self, name: str, params: dict, seed: int, S: dict):
        self.name, self.params, self.seed, self.S = name, dict(params), seed, S

    def _learner(self, n_trees, seed):
        p = self.params
        if self.name == "GWRF":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=n_trees, max_depth=p["max_depth"],
                                          min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                          random_state=seed, n_jobs=1)
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=n_trees, learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                             min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                             reg_lambda=p["reg_lambda"], tree_method="hist", eval_metric="logloss",
                             random_state=seed, n_jobs=1)

    def _n_trees(self):
        g, quick = self.S["gw"], self.S["run"]["quick"]
        if self.name == "GWRF":
            local = min(g["local_rf_trees"], 30) if quick else g["local_rf_trees"]
            return local, 3 * local
        n = min(self.params["n_estimators"], g["local_xgb_rounds_max"])
        n = min(n, self.S["quick"]["max_trees"]) if quick else n
        return n, n

    def fit(self, X, y, w=None, refs=None):
        from scipy.spatial import cKDTree

        g = self.S["gw"]
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y, int)
        self.w_ = np.ones(len(y)) if w is None else np.asarray(w, float)
        self.n_local_, n_global = self._n_trees()
        self.global_ = self._learner(n_global, self.seed)
        self.global_.fit(self.X_, self.y_, sample_weight=_weights(self.y_, self.w_, self.params["class_weight"]))
        self.anchors_, self.site_ = np.unique(np.round(self.X_[:, :2], 2), axis=0, return_inverse=True)
        self.site_ = self.site_.ravel()
        self.tree_ = cKDTree(self.anchors_)
        n_sites = len(self.anchors_)
        kq = min(g["max_neighbours"] + 1, n_sites)
        d, i = self.tree_.query(self.anchors_, k=kq)
        self.dist_, self.nbr_ = np.atleast_2d(d), np.atleast_2d(i)
        counts = np.zeros((n_sites, 2))
        np.add.at(counts, (self.site_, self.y_), 1)
        self.k_ = np.array([self._neighbourhood(a, counts) for a in range(n_sites)])
        self.fallback_rate_ = float(np.mean(self.k_ < 0))
        self.local_ = {}
        return self

    def _neighbourhood(self, a, counts) -> int:
        """Number of neighbour boreholes for anchor a (-1 = use the global model)."""
        g, n_sites = self.S["gw"], len(self.anchors_)
        k = max(g["min_boreholes"], min(self.params["k_neighbours"], n_sites))
        while True:
            if (counts[self.nbr_[a, :k]].sum(axis=0) >= g["min_cells_per_class"]).all():
                return int(k)
            if k >= min(g["max_neighbours"], n_sites):
                return -1
            k += 1

    def _fit_local(self, a):
        k = self.k_[a]
        if k < 0:
            return None
        d = self.dist_[a]
        b = d[k] if k < d.shape[0] else d[k - 1] * 1.01 + 1e-6      # bandwidth: distance to the (k+1)-th borehole
        w_site = np.zeros(len(self.anchors_))
        w_site[self.nbr_[a, :k]] = (1 - (d[:k] / max(b, 1e-9)) ** 2) ** 2 if self.params["kernel"] == "bisquare" else 1.0
        w = w_site[self.site_] * self.w_
        use = w > 0
        m = self._learner(self.n_local_, self.seed + 1 + int(a))
        m.fit(self.X_[use], self.y_[use], sample_weight=_weights(self.y_[use], w[use], self.params["class_weight"]))
        return m

    def predict(self, X) -> np.ndarray:
        from joblib import Parallel, delayed

        X = np.asarray(X, float)
        p_global = self.global_.predict_proba(X)[:, 1]
        anchor = self.tree_.query(np.round(X[:, :2], 2))[1]
        todo = [int(a) for a in np.unique(anchor) if int(a) not in self.local_]
        fitted = Parallel(n_jobs=self.S["run"]["n_jobs"], prefer="threads")(delayed(self._fit_local)(a) for a in todo)
        self.local_.update(zip(todo, fitted))
        p_local = p_global.copy()
        for a in np.unique(anchor):
            m = self.local_[int(a)]
            if m is not None:
                sel = anchor == a
                p_local[sel] = m.predict_proba(X[sel])[:, 1]
        alpha = self.params["alpha"]
        return alpha * p_local + (1 - alpha) * p_global


def _lrelu(x):
    return np.where(x > 0, x, 0.01 * x)


def _dlrelu(x):
    return np.where(x > 0, 1.0, 0.01)


class GBARNN:
    """Generalized Borehole Autoregressive Neural Network (Li et al., 2026), in NumPy.

    * GBDNN unit (shared by all reference boreholes): (d, Phi, S) -> g1 -> g2 -> 1 gives a
      "generalised distance" between the cell and each borehole.
    * Head: [generalised distances of all boreholes, X, Y, Z] -> u1 -> u2 -> one score per borehole;
      softmax over boreholes (the cell's own borehole masked) gives the borehole weights.
    * P(Clay) = sum over boreholes of weight x clay share of that borehole.
    Leaky ReLU, binary cross-entropy (with sample weights), AdamW. Layer sizes, learning rate,
    epochs, batch size and weight decay are tuned like the other models."""

    def __init__(self, name: str, params: dict, seed: int, S: dict):
        self.params, self.seed, self.S = dict(params), seed, S

    @staticmethod
    def _split(X, n):
        return X[:, :3], X[:, 3:3 + n], X[:, 3 + n:3 + 2 * n], X[:, 3 + 2 * n:3 + 3 * n]

    def _inputs(self, X):
        xyz, d, phi, s = self._split(np.asarray(X, float), self.n_)
        mask = d < self.S["features"]["masked_distance_km"] - 1e-9            # usable boreholes
        F = np.stack([(d - self.d_mu_) / self.d_sd_, phi, s], axis=2)
        return F, (xyz - self.x_mu_) / self.x_sd_, mask

    def _forward(self, F, Xs, mask, P):
        H1 = F @ P["W1"] + P["c1"]
        A1 = _lrelu(H1)
        H2 = A1 @ P["W2"] + P["c2"]
        A2 = _lrelu(H2)
        G = A2 @ P["w3"] + P["c3"]
        Z0 = np.concatenate([G, Xs], axis=1)
        H3 = Z0 @ P["U1"] + P["e1"]
        A3 = _lrelu(H3)
        H4 = A3 @ P["U2"] + P["e2"]
        A4 = _lrelu(H4)
        L = A4 @ P["U3"] + P["e3"]
        L = np.where(mask, L, -1e9)
        L = L - L.max(axis=1, keepdims=True)
        E = np.exp(L) * mask
        Wt = E / np.maximum(E.sum(axis=1, keepdims=True), 1e-300)
        p = np.clip(Wt @ self.b_, 1e-6, 1 - 1e-6)
        return p, (F, H1, A1, H2, A2, Z0, H3, A3, H4, A4, Wt)

    def _backward(self, p, y, sw, cache, P):
        F, H1, A1, H2, A2, Z0, H3, A3, H4, A4, Wt = cache
        dp = sw * (-(y / p) + (1 - y) / (1 - p)) / sw.sum()
        dW = dp[:, None] * self.b_[None, :]
        dL = Wt * (dW - (dW * Wt).sum(axis=1, keepdims=True))
        g = {"U3": A4.T @ dL, "e3": dL.sum(0)}
        dH4 = (dL @ P["U3"].T) * _dlrelu(H4)
        g["U2"], g["e2"] = A3.T @ dH4, dH4.sum(0)
        dH3 = (dH4 @ P["U2"].T) * _dlrelu(H3)
        g["U1"], g["e1"] = Z0.T @ dH3, dH3.sum(0)
        dG = (dH3 @ P["U1"].T)[:, :self.n_]
        g["w3"], g["c3"] = np.einsum("bnk,bn->k", A2, dG), np.array(dG.sum())
        dH2 = (dG[:, :, None] * P["w3"][None, None, :]) * _dlrelu(H2)
        g["W2"], g["c2"] = np.einsum("bni,bnj->ij", A1, dH2), dH2.sum((0, 1))
        dH1 = (dH2 @ P["W2"].T) * _dlrelu(H1)
        g["W1"], g["c1"] = np.einsum("bni,bnj->ij", F, dH1), dH1.sum((0, 1))
        return g

    @staticmethod
    def loss(p, y, sw):
        return float(-(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / sw.sum())

    def _init(self, rng):
        p, n = self.params, self.n_
        def he(i, o):
            return rng.normal(0, np.sqrt(2.0 / i), (i, o))
        return {"W1": he(3, p["g1"]), "c1": np.zeros(p["g1"]), "W2": he(p["g1"], p["g2"]), "c2": np.zeros(p["g2"]),
                "w3": he(p["g2"], 1)[:, 0], "c3": np.zeros(()), "U1": he(n + 3, p["u1"]), "e1": np.zeros(p["u1"]),
                "U2": he(p["u1"], p["u2"]), "e2": np.zeros(p["u2"]), "U3": he(p["u2"], n) * 0.1, "e3": np.zeros(n)}

    def fit(self, X, y, w=None, refs=None):
        self.n_ = len(refs.codes)
        self.b_ = refs.clay_share
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        sw = _weights(y, w, self.params.get("class_weight", "none"))
        xyz, d, _, _ = self._split(X, self.n_)
        usable = d < self.S["features"]["masked_distance_km"] - 1e-9
        self.d_mu_, self.d_sd_ = d[usable].mean(), d[usable].std() + 1e-9
        self.x_mu_, self.x_sd_ = xyz.mean(axis=0), xyz.std(axis=0) + 1e-9
        F, Xs, mask = self._inputs(X)
        rng = np.random.default_rng(self.seed)
        P = self._init(rng)
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(v) for k, v in P.items()}
        lr, wd, bs = self.params["learning_rate"], self.params["weight_decay"], int(self.params["batch_size"])
        epochs = self.params["epochs"]
        if self.S["run"]["quick"]:
            epochs = min(epochs, self.S["quick"]["max_epochs"])
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        for _ in range(int(epochs)):
            order = rng.permutation(len(y))
            for s in range(0, len(y), bs):
                i = order[s:s + bs]
                p, cache = self._forward(F[i], Xs[i], mask[i], P)
                grads = self._backward(p, y[i], sw[i], cache, P)
                t += 1
                for k in P:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                    step = (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
                    decay = wd * P[k] if P[k].ndim == 2 else 0.0      # weight decay on weight matrices only
                    P[k] = P[k] - lr * (step + decay)
        self.P_ = P
        return self

    def predict(self, X) -> np.ndarray:
        F, Xs, mask = self._inputs(X)
        out = np.empty(len(F))
        for s in range(0, len(F), 4096):
            out[s:s + 4096], _ = self._forward(F[s:s + 4096], Xs[s:s + 4096], mask[s:s + 4096], self.P_)
        return out


class Model:
    """One model on one feature set:  fit(cells, weights)  and  p_clay(cells).
    With XYZ_VAHDW the reference boreholes are the boreholes of the training cells of this fit,
    so the labels of other (test) boreholes are never used."""

    def __init__(self, name: str, features: str, params: dict, seed: int, S: dict):
        self.name, self.features, self.S = name, features, S
        params = {k: v for k, v in params.items() if k != "filled_weight"}
        base = {"GWRF": GWBase, "GWXGB": GWBase, "GBARNN": GBARNN}.get(name, SklearnBase)
        self.base = base(name, params, seed, S)
        self.refs = None

    def matrix(self, cells: pd.DataFrame) -> np.ndarray:
        X = cells[FEATURES].values.astype(float)
        if self.features == "XYZ":
            return X
        d, phi, s = self.refs.features(cells, self.S)
        return np.hstack([X, d, phi, s])

    def _chunks(self, cells):
        n = self.S["features"]["chunk_rows"]
        for i in range(0, len(cells), n):
            yield cells.iloc[i:i + n]

    def fit(self, cells: pd.DataFrame, w=None):
        if self.features == "XYZ_VAHDW":
            self.refs = References(cells, self.S)
        self.base.fit(self.matrix(cells), cells.y.values, w, self.refs)
        self.fallback_rate_ = getattr(self.base, "fallback_rate_", None)
        return self

    def p_clay(self, cells: pd.DataFrame) -> np.ndarray:
        return np.concatenate([self.base.predict(self.matrix(c)) for c in self._chunks(cells)]) if len(cells) else np.empty(0)

    def p_from_matrix(self, X) -> np.ndarray:
        """Predictions from a feature matrix (used by the permutation importance)."""
        return self.base.predict(X)

    def columns(self) -> list[str]:
        return feature_names(self.features, self.refs)


def make_model(name: str, features: str, params: dict, seed: int, S: dict) -> Model:
    return Model(name, features, params, seed, S)


def fill_weights(cells: pd.DataFrame, params: dict):
    """Training weights: 1 for measured cells, `filled_weight` for filled cells."""
    fw = float(params.get("filled_weight", 1.0))
    return None if fw == 1.0 else np.where(cells.measured.values == 1, 1.0, fw)


def suggest(trial, space: dict) -> dict:
    """One parameter set from a search space in settings.toml."""
    p = {}
    for name, s in space.items():
        if s["type"] == "int":
            p[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1), log=s.get("log", False))
        elif s["type"] == "float":
            p[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        else:
            p[name] = trial.suggest_categorical(name, s["choices"])
    return p


# ================================================== 5. Nested spatial cross-validation ====

def load_folds(S: dict, multiple: int = 1) -> tuple[pd.Series, float]:
    """Outer folds made by blockCV in R: fold number of every borehole, and the block size."""
    f = pd.read_csv(folds_dir(S) / f"folds_x{multiple}.csv")
    return f.set_index("BH_Code").fold, float(f.block_m.iloc[0])


def outer_split(cells, folds, k, purge_m=0.0):
    """Test = boreholes of fold k. Train = all other boreholes, minus (with a purge) those
    closer than purge_m to any test borehole."""
    fold = cells.BH_Code.map(folds)
    test, train = cells[fold == k], cells[fold != k]
    if purge_m > 0:
        bh_tr = train.groupby("BH_Code")[["UTM_Easting_m", "UTM_Northing_m"]].first()
        bh_te = test.groupby("BH_Code")[["UTM_Easting_m", "UTM_Northing_m"]].first()
        too_close = bh_tr.index[nearest_distance(bh_tr.values, bh_te.values) < purge_m]
        train = train[~train.BH_Code.isin(too_close)]
    return train, test


def threshold_grid(S):
    a, b, s = S["cv"]["threshold_grid"]
    return np.round(np.arange(a, b + s / 2, s), 4)


def macro_f1(y, y_pred) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y, y_pred, average="macro", labels=[0, 1], zero_division=0)


def best_threshold(y, p, grid) -> tuple[float, float]:
    """Decision threshold on P(Clay) with the highest Macro-F1."""
    scores = [macro_f1(y, (p >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def inner_predictions(name, features, params, train, groups, seed, S) -> np.ndarray:
    """Out-of-fold P(Clay) of the training cells: each inner group is predicted by a model
    trained on the other groups (the inner groups are the remaining outer folds). With VA-HDW,
    the references of each inner model are its own training boreholes."""
    p = np.full(len(train), np.nan)
    w = fill_weights(train, params)
    for g in np.unique(groups):
        val = groups == g
        m = make_model(name, features, params, seed, S).fit(train[~val], None if w is None else w[~val])
        p[val] = m.p_clay(train[val])
    return p


def n_trials(name: str, S: dict) -> int:
    return S["quick"]["n_trials"] if S["run"]["quick"] else S["cv"]["n_trials"][name]


def tune(name, features, train, groups, seed, S, on_trial=None) -> dict:
    """Optuna (TPE) chooses the hyperparameters and the filled-cell weight with the highest
    inner Macro-F1; for every trial the threshold is the best one on the inner predictions."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y, grid = train.y.values, threshold_grid(S)
    has_filled = (train.measured == 0).any()

    def objective(trial):
        p = suggest(trial, S["space"][name])
        if has_filled:
            p["filled_weight"] = trial.suggest_categorical("filled_weight", [float(v) for v in S["cv"]["filled_weights"]])
        t, f = best_threshold(y, inner_predictions(name, features, p, train, groups, seed, S), grid)
        trial.set_user_attr("threshold", t)
        return f

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials(name, S),
                   callbacks=[lambda study, trial: on_trial()] if on_trial else None)
    return {"params": study.best_params, "threshold": study.best_trial.user_attrs["threshold"],
            "inner_macro_f1": study.best_value,
            "trials": study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))}


def run_fold(run: dict, cells, folds, k, purge_m, S, on_trial=None) -> dict:
    """One outer fold: tune on the training boreholes, refit, predict the test fold."""
    t0 = time.time()
    name, features, tag = run["model"], run["features"], run["tag"]
    train, test = outer_split(cells, folds, k, purge_m)
    groups = train.BH_Code.map(folds).values          # inner groups = the other outer folds
    seed = seed_of(S, tag, k)
    best = tune(name, features, train, groups, seed, S, on_trial)
    model = make_model(name, features, best["params"], seed, S).fit(train, fill_weights(train, best["params"]))
    p = model.p_clay(test)
    bh_tr = train.groupby("BH_Code")[["UTM_Easting_m", "UTM_Northing_m"]].first()
    pred = test[["BH_Code", "cell", "depth_top_m", *FEATURES, "Region", "y", "measured"]].copy()
    pred["p_clay"] = p
    pred["threshold"] = best["threshold"]
    pred["y_pred"] = (p >= best["threshold"]).astype(int)
    pred["dist_nearest_train_m"] = nearest_distance(pred[FEATURES[:2]].values, bh_tr.values)
    pred["outer_fold"] = k
    info = {"tag": tag, "outer_fold": k, "params": best["params"], "threshold": best["threshold"],
            "inner_macro_f1": best["inner_macro_f1"], "train_boreholes": int(train.BH_Code.nunique()),
            "train_cells": len(train), "train_filled_cells": int((train.measured == 0).sum()),
            "test_boreholes": int(test.BH_Code.nunique()), "test_cells": len(test),
            "gw_fallback_rate": model.fallback_rate_, "seconds": round(time.time() - t0, 1)}
    return {"predictions": pred, "info": info, "trials": best["trials"], "model": model}


def load_final(S: dict, comparison: bool = False):
    """The frozen final model (or, with comparison=True, the frozen comparison model on the other
    feature set), after checking that its file is exactly the one recorded in notebook 05."""
    import joblib

    rec = json.loads((out_dir(S, "05_final_model") / "final_model.json").read_text())
    if comparison:
        rec = rec["comparison"]
    path = out_dir(S, "05_final_model", private=True) / rec["file"]
    if sha256(path) != rec["sha256"]:
        raise RuntimeError(f"{path.name} is not the frozen file (SHA-256 differs)")
    return joblib.load(path), rec


def run_list(S: dict) -> list[dict]:
    """Every run of the experiment:
    main block size: RF, XGB, MLP, GWRF, GWXGB on XYZ and on XYZ_VAHDW, and GBARNN on XYZ_VAHDW;
    2x and 3x block size: XGB on both feature sets; purge at 1x and 2x (main block): XGB on both sets."""
    cv, sets = S["cv"], S["features"]["sets"]
    runs = [{"model": m, "features": f, "multiple": 1, "purge_x": 0} for m in cv["models"] for f in sets]
    runs += [{"model": m, "features": "XYZ_VAHDW", "multiple": 1, "purge_x": 0} for m in cv["vahdw_only_models"]]
    runs += [{"model": m, "features": f, "multiple": x, "purge_x": 0} for x in S["blockcv"]["multiples"] if x != 1
             for m in cv["sensitivity_models"] for f in sets]
    runs += [{"model": m, "features": f, "multiple": 1, "purge_x": p} for p in cv["purge_multiples"]
             for m in cv["purge_models"] for f in sets]
    for r in runs:
        r["tag"] = f"{r['model']}_{r['features']}_x{r['multiple']}_purge{r['purge_x']}"
    return runs


def settings_key(S: dict, cells: pd.DataFrame, folds: pd.Series) -> str:
    """Fingerprint of the data, folds and settings; a saved fold is reused only if it matches."""
    parts = {k: S[k] for k in ("run", "area", "cells", "cv", "gw", "space", "quick", "features")}
    parts["cells"] = [parts["cells"], int(pd.util.hash_pandas_object(cells[["BH_Code", "cell", "y", "measured", *FEATURES]],
                                                                      index=False).sum())]
    parts["folds"] = int(pd.util.hash_pandas_object(folds.sort_index()).sum())
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ================================================ 6. Metrics and model selection ====

def scores(y, p, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef,
                                 recall_score, roc_auc_score)
    y, p, y_pred = np.asarray(y, int), np.asarray(p, float), np.asarray(y_pred, int)
    both = len(np.unique(y)) == 2
    return {"n": len(y), "clay_share": float(y.mean()) if len(y) else np.nan,
            "accuracy": accuracy_score(y, y_pred) if len(y) else np.nan,
            "balanced_accuracy": balanced_accuracy_score(y, y_pred) if both else np.nan,
            "macro_f1": macro_f1(y, y_pred) if len(y) else np.nan,
            "recall_clay": recall_score(y, y_pred, pos_label=1, zero_division=np.nan),
            "recall_sand": recall_score(y, y_pred, pos_label=0, zero_division=np.nan),
            "mcc": matthews_corrcoef(y, y_pred) if both else np.nan,
            "roc_auc": roc_auc_score(y, p) if both else np.nan,
            "brier": brier_score_loss(y, p) if len(y) else np.nan}


RUN_KEYS = ["model", "features", "multiple", "block_m", "purge_x", "purge_m"]


def fold_scores(preds: pd.DataFrame) -> pd.DataFrame:
    """Scores of every run and outer fold, on all test cells and on measured cells only."""
    rows = []
    for key, g in preds.groupby(RUN_KEYS + ["outer_fold"]):
        for cells_, d in (("all", g), ("measured", g[g.measured == 1])):
            rows.append({**dict(zip(RUN_KEYS + ["outer_fold"], key)), "cells": cells_,
                         **scores(d.y, d.p_clay, d.y_pred)})
    return pd.DataFrame(rows)


def summary(fold_table: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and minimum over the outer folds."""
    m = ["accuracy", "balanced_accuracy", "macro_f1", "recall_clay", "recall_sand", "mcc", "roc_auc", "brier"]
    g = fold_table.groupby(RUN_KEYS + ["cells"])[m]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.min().add_suffix("_min"))
    return out.join(fold_table.groupby(RUN_KEYS + ["cells"]).size().rename("n_folds")).reset_index()


def select_model(summ: pd.DataFrame, cells: str = "all") -> dict:
    """Rule decided before the results, on the main block size without purge; every model on
    every feature set is a candidate:
      1. eligible: mean Clay and Sand recall >= 0.50 and no fold below 0.25;
      2. best = highest mean Macro-F1;
      3. among eligible candidates within one standard error of the best: lowest fold SD of
         Macro-F1, then lowest mean Brier score."""
    s = summ[(summ.multiple == 1) & (summ.purge_x == 0) & (summ.cells == cells)].copy()
    s["candidate"] = s.model + " | " + s.features
    s["eligible"] = ((s.recall_clay_mean >= 0.5) & (s.recall_sand_mean >= 0.5)
                     & (s.recall_clay_min >= 0.25) & (s.recall_sand_min >= 0.25))
    e = s[s.eligible]
    if e.empty:
        return {"selected": None, "reason": "no model meets the recall rule", "table": s}
    best = e.loc[e.macro_f1_mean.idxmax()]
    se = best.macro_f1_sd / np.sqrt(best.n_folds) if best.n_folds > 1 else 0.0
    near = e[e.macro_f1_mean >= best.macro_f1_mean - se].sort_values(["macro_f1_sd", "brier_mean"])
    pick = near.iloc[0]
    return {"selected": pick.model, "features": pick.features, "highest_mean": best.candidate,
            "best_macro_f1_mean": float(best.macro_f1_mean), "one_se": float(se),
            "within_1se": list(near.candidate), "table": s}


def paired_gain(fold_table: pd.DataFrame, cells: str = "all", multiple: int = 1, purge_x: int = 0) -> pd.DataFrame:
    """Rule decided before the results: for each model, the fold-by-fold difference in Macro-F1
    XYZ_VAHDW - XYZ (same folds, same test boreholes). VA-HDW 'helps' only if the mean difference
    is larger than one standard error of the differences; 'hurts' if below minus one SE."""
    f = fold_table[(fold_table.cells == cells) & (fold_table.multiple == multiple) & (fold_table.purge_x == purge_x)]
    t = f.pivot_table(index=["model", "outer_fold"], columns="features", values="macro_f1").dropna()
    rows = []
    for m, g in t.groupby(level="model"):
        diff = g["XYZ_VAHDW"] - g["XYZ"]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        verdict = "helps" if diff.mean() > se else ("hurts" if diff.mean() < -se else "no clear difference")
        rows.append({"model": m, "folds": len(diff), "macro_f1_XYZ": g["XYZ"].mean(), "macro_f1_XYZ_VAHDW": g["XYZ_VAHDW"].mean(),
                     "mean_gain": diff.mean(), "se_gain": se, "verdict": verdict})
    return pd.DataFrame(rows)


def distance_bands(preds: pd.DataFrame, edges=(0, 100, 300, np.inf)) -> pd.DataFrame:
    """Scores by distance from the test borehole to the nearest training borehole."""
    lab = [f"{a:g}-{b:g} m" if np.isfinite(b) else f"> {a:g} m" for a, b in zip(edges[:-1], edges[1:])]
    d = preds.assign(band=pd.cut(preds.dist_nearest_train_m, list(edges), right=True, labels=lab, include_lowest=True))
    rows = [{"model": m, "features": f, "cells": c, "band": b, **scores(g.y, g.p_clay, g.y_pred)}
            for c, dd in (("all", d), ("measured", d[d.measured == 1]))
            for (m, f, b), g in dd.groupby(["model", "features", "band"], observed=True)]
    return pd.DataFrame(rows)


# ===================================================================== 7. 3D map ====

class DEM:
    """The DEM, sampled the same way as the CSV ground elevations: bilinear interpolation
    between the four surrounding cell centres."""

    def __init__(self, S: dict):
        import rasterio

        with rasterio.open(S["root"] / S["files"]["dem"]) as r:
            self.z = r.read(1).astype(float)
            if r.nodata is not None:
                self.z[self.z == r.nodata] = np.nan
            self.left, self.top, self.res = r.bounds.left, r.bounds.top, r.res[0]

    def elevation(self, x, y) -> np.ndarray:
        fx = (np.asarray(x, float) - self.left) / self.res - 0.5
        fy = (self.top - np.asarray(y, float)) / self.res - 0.5
        j, i = np.floor(fx).astype(int), np.floor(fy).astype(int)
        a, b = fx - j, fy - i
        ok = (i >= 0) & (j >= 0) & (i + 1 < self.z.shape[0]) & (j + 1 < self.z.shape[1])
        out = np.full(len(fx), np.nan)
        i, j, a, b, z = i[ok], j[ok], a[ok], b[ok], self.z
        out[ok] = (1 - b) * ((1 - a) * z[i, j] + a * z[i, j + 1]) + b * ((1 - a) * z[i + 1, j] + a * z[i + 1, j + 1])
        return out


def map_columns(S: dict, spacing: float) -> pd.DataFrame:
    """Grid points (column centres) inside the area, with their DEM ground elevation."""
    import shapely

    poly = area_polygon(S)
    x0, y0, x1, y1 = poly.bounds
    gx = np.arange(np.floor(x0 / spacing) * spacing + spacing / 2, x1, spacing)
    gy = np.arange(np.floor(y0 / spacing) * spacing + spacing / 2, y1, spacing)
    X, Y = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(poly, X.ravel(), Y.ravel())
    cols = pd.DataFrame({"X": X.ravel()[inside], "Y": Y.ravel()[inside]})
    cols["ground_m"] = DEM(S).elevation(cols.X, cols.Y)
    if cols.ground_m.isna().any():
        raise ValueError(f"{int(cols.ground_m.isna().sum())} map columns have no DEM value")
    return cols


def map_cells(cols: pd.DataFrame, S: dict) -> pd.DataFrame:
    """30 cells of 0.5 m (0-15 m) below every grid point."""
    tops, _ = _cell_edges(S)
    n = len(tops)
    g = pd.DataFrame({"col": np.repeat(np.arange(len(cols)), n), "layer": np.tile(np.arange(n), len(cols)),
                      "depth_top_m": np.tile(tops, len(cols))})
    g["UTM_Easting_m"] = cols.X.values[g.col]
    g["UTM_Northing_m"] = cols.Y.values[g.col]
    g["ground_m"] = cols.ground_m.values[g.col]
    g["Top_Elevation_m"] = g.ground_m - g.depth_top_m
    g["Z"] = g.Top_Elevation_m - S["cells"]["cell_m"] / 2           # cell-centre elevation
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
    for name, fill, dtype in (("p_clay", np.nan, "float32"), ("clay", -1, "int8"), ("reliability_zone", -1, "int8"),
                              ("extrap_vertical", -1, "int8")):
        a = np.full((len(zc), len(ys), len(xs)), fill, dtype=dtype)
        a[m[ok], iy[col[ok]], ix[col[ok]]] = grid[name].values[ok]
        data[name] = (("z", "y", "x"), a)
    crs = CRS.from_epsg(S["files"]["epsg"])
    ds = xr.Dataset(data, coords={"z": zc, "y": ys, "x": xs})
    ds["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": crs.to_wkt(), "spatial_ref": crs.to_wkt()})
    for v in data:
        ds[v].attrs["grid_mapping"] = "spatial_ref"
    ds.p_clay.attrs.update(long_name="probability of Clay", units="1")
    ds.clay.attrs.update(long_name="predicted class (1 = Clay, 0 = Sand)")
    ds.x.attrs.update(standard_name="projection_x_coordinate", units="m", axis="X")
    ds.y.attrs.update(standard_name="projection_y_coordinate", units="m", axis="Y")
    ds.z.attrs.update(standard_name="height", units="m", positive="up", axis="Z")
    ds.attrs.update(Conventions="CF-1.8", title="Siem Reap 3D Clay/Sand prediction (V1_Li)")
    enc = {v: {"zlib": True, "complevel": 4, **({"_FillValue": -1} if data[v][1].dtype == np.int8 else {})} for v in data}
    ds.to_netcdf(path, engine="netcdf4", encoding=enc)
    return {"shape_zyx": [len(zc), len(ys), len(xs)], "voxels_filled": int(ok.sum())}
