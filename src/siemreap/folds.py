"""Outer and inner spatial folds.

The outer folds are produced in R by blockCV (R/02_blockcv_folds.R) and only imported here;
nothing in Python regenerates them (brief section 7). File formats:

* data/folds/outer_folds_<b>m.csv : BH_Code, fold, role, block_size_m
    fold 1..k  for Downtown development boreholes (role = "cv")
    fold 0     for 5 km ring development boreholes (role = "train_only")
* data/folds/inner_fallback_<b>m.csv : BH_Code, outer_fold, inner_fold
    R-generated inner folds, used only for an outer fold whose re-used outer groups fail
    the pre-specified adequacy check (brief section 9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import folds_dir
from .geo import nearest_distance

XY = ["UTM_Easting_m", "UTM_Northing_m"]


def load_outer(cfg: dict, block_m: int) -> pd.DataFrame:
    p = folds_dir(cfg) / f"outer_folds_{int(block_m)}m.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run R/02_blockcv_folds.R first.")
    f = pd.read_csv(p).set_index("BH_Code")
    if f.index.duplicated().any():
        raise ValueError(f"{p.name}: a borehole appears more than once")
    return f


def load_inner_fallback(cfg: dict, block_m: int) -> pd.DataFrame | None:
    p = folds_dir(cfg) / f"inner_fallback_{int(block_m)}m.csv"
    return pd.read_csv(p) if p.exists() else None


def validate_outer(folds: pd.DataFrame, cells: pd.DataFrame, cfg: dict) -> dict:
    """Integrity checks; raises on any violation, returns a summary otherwise."""
    k = cfg["blockcv"]["k"]
    bh = cells.drop_duplicates("BH_Code").set_index("BH_Code")
    dev_dt = set(bh.index[bh.in_A])
    dev_ring = set(bh.index[bh.in_B & ~bh.in_A])
    not_dev = set(bh.index[bh.set != "development"])
    cv = folds[folds.role == "cv"]
    problems = []
    if set(cv.index) != dev_dt:
        problems.append(f"cv boreholes differ from Downtown development boreholes "
                        f"(missing {len(dev_dt - set(cv.index))}, extra {len(set(cv.index) - dev_dt)})")
    if not set(cv.fold).issubset(set(range(1, k + 1))) or cv.fold.nunique() != k:
        problems.append(f"cv folds must be exactly 1..{k}")
    if set(folds.index) & not_dev:
        problems.append("external or excluded boreholes present in the fold file")
    ring = folds[folds.role == "train_only"]
    if set(ring.index) - dev_ring or (ring.fold != 0).any():
        problems.append("train_only rows must be 5 km ring development boreholes with fold 0")
    c = cells[cells.in_A].join(cv.fold, on="BH_Code")
    per = c.groupby("fold").agg(boreholes=("BH_Code", "nunique"), cells=("y", "size"), clay_share=("y", "mean"))
    if (per.clay_share <= 0).any() or (per.clay_share >= 1).any():
        problems.append("a test fold contains only one class")
    if problems:
        raise ValueError("; ".join(problems))
    return {"k": k, "downtown_dev_boreholes": len(dev_dt), "ring_train_only": int(len(ring)),
            "per_fold": per.round(3).reset_index().to_dict("records")}


def split(cells: pd.DataFrame, folds: pd.DataFrame, k: int, scenario: str, purge_m: float = 0.0):
    """Outer split. Test: Downtown development boreholes in fold k (same in both scenarios).
    Train: scenario boreholes not in fold k (5 km ring boreholes are train-only in B).
    `purge_m` > 0 also removes training boreholes within that distance of any test borehole."""
    flag = {"A": "in_A", "B": "in_B"}[scenario]
    fold = cells.BH_Code.map(folds.fold)
    test = cells[cells.in_A & (fold == k)]
    train = cells[cells[flag] & fold.notna() & (fold != k)]
    if purge_m > 0:
        tb, trb = _bh_xy(test), _bh_xy(train)
        d = nearest_distance(trb.values, tb.values)
        drop = set(trb.index[d < purge_m])
        train = train[~train.BH_Code.isin(drop)]
    return train, test


def _bh_xy(cells: pd.DataFrame) -> pd.DataFrame:
    return cells.drop_duplicates("BH_Code").set_index("BH_Code")[XY]


def adequacy(train: pd.DataFrame, groups: np.ndarray, cfg: dict) -> pd.DataFrame:
    ic = cfg["inner_cv"]
    rows = []
    for g in sorted(set(groups[groups > 0])):
        v = train[groups == g]
        share = v.y.mean()
        rows.append({"inner_fold": int(g), "boreholes": v.BH_Code.nunique(), "cells": len(v),
                     "clay_share": round(float(share), 3),
                     "ok": bool(v.BH_Code.nunique() >= ic["min_boreholes"] and len(v) >= ic["min_cells"]
                                and ic["min_class_share"] <= share <= 1 - ic["min_class_share"])})
    return pd.DataFrame(rows)


def inner_groups(train: pd.DataFrame, folds: pd.DataFrame, k: int, cfg: dict, block_m: int):
    """Inner-fold label per training cell (-1 = always in inner training: 5 km ring cells).

    Default: re-use the remaining outer folds of the Downtown training boreholes. If any group
    fails the adequacy check, use the R-generated fallback inner folds for this outer fold."""
    groups = train.BH_Code.map(folds.fold).astype(int).values.copy()
    groups[groups == 0] = -1
    adq = adequacy(train, groups, cfg)
    source = "outer_folds"
    if not adq.ok.all():
        fb = load_inner_fallback(cfg, block_m)
        if fb is None:
            raise RuntimeError(f"Inner folds for outer fold {k} fail the adequacy check and no R fallback "
                               f"file exists (R/02_blockcv_folds.R writes inner_fallback_{block_m}m.csv).")
        m = fb[fb.outer_fold == k].set_index("BH_Code").inner_fold
        groups = train.BH_Code.map(m).fillna(-1).astype(int).values
        adq = adequacy(train, groups, cfg)
        source = "r_fallback"
    return groups, adq, source


def test_train_distances(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Horizontal distance from each test borehole to its nearest training borehole."""
    tb, trb = _bh_xy(test), _bh_xy(train)
    return pd.Series(nearest_distance(tb.values, trb.values), index=tb.index, name="dist_nearest_train_m")
