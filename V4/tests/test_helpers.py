"""Checks of the main rules in helpers.py.  Run:  python -m pytest -q tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import helpers as H  # noqa: E402


@pytest.fixture
def S():
    S = H.load_settings(ROOT)
    S["run"]["quick"] = True
    S["run"]["n_jobs"] = 1
    return S


def records(rows):
    """rows: (BH, from, to, USCS) with ground elevation 10 m at (0, 0)."""
    df = pd.DataFrame(rows, columns=["BH_Code", "Depth_From_m", "Depth_To_m", "USCS"])
    return df.assign(UTM_Easting_m=0.0, UTM_Northing_m=0.0, Ground_Elevation_m=10.0, Region="Downtown")


def test_labels_and_the_ss_rule(S):
    r = records([("B", 0.5, 1.0, "CL"),     # cell 1 Clay
                 ("B", 1.0, 1.45, None),    # cell 2: unclassified SS sample
                 ("B", 1.5, 2.0, "CL"),     # cell 3 Clay
                 ("B", 2.0, 2.5, "SC"),     # cell 4 Sand
                 ("B", 3.0, 3.5, "ML")])    # cell 6: a symbol outside the binary map -> no label
    cells, info = H.make_cells(r, S)
    assert list(cells.cell) == [1, 3, 4] and list(cells.y) == [1, 1, 0]
    assert np.allclose(cells.Top_Elevation_m, 10.0 - cells.depth_top_m)
    filled, _ = H.fill_gaps(cells, r, S)                           # student's rule: fill through the SS sample
    assert list(filled[filled.measured == 0].cell) == [2] and filled[filled.cell == 2].y.item() == 1
    S["cells"]["unclassified_blocks_fill"] = True                  # V3 rule: the SS sample blocks the fill
    strict, info = H.fill_gaps(cells, r, S)
    assert (strict.measured == 1).all() and info["record_inside"] == 1


def test_cell_touched_by_both_classes_is_dropped(S):
    cells, info = H.make_cells(records([("B", 0.0, 0.3, "CL"), ("B", 0.3, 0.5, "SM")]), S)
    assert cells.empty and info["cells_dropped_both_classes"] == 1


def synthetic_cells(n_bh=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bh):
        x, y = rng.uniform(0, 1000, 2)
        for j in range(20):
            rows.append({"BH_Code": f"B{b}", "cell": j, "depth_top_m": j * 0.5, "UTM_Easting_m": x,
                         "UTM_Northing_m": y, "Top_Elevation_m": 10 - j * 0.5, "Region": "Downtown",
                         "y": int((x > 500) ^ (j > 10) ^ (rng.random() < 0.1)), "measured": int(j % 4 != 0)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name", ["RF", "XGB", "MLP", "GWRF", "GWXGB", "DEPTH_ONLY"])
def test_every_model_gives_probabilities(S, name):
    c = synthetic_cells()
    space = S["space"].get(name, {})
    params = {k: (v["choices"][0] if v["type"] == "categorical" else v["low"]) for k, v in space.items()}
    if "k_neighbours" in params:
        params["k_neighbours"] = 10
    params["filled_weight"] = 0.5
    p = H.make_model(name, params, 1, S).fit(c, H.fill_weights(c, params)).p_clay(c)
    assert len(p) == len(c) and np.all((p >= 0) & (p <= 1))


def test_purge_removes_close_training_boreholes():
    c = synthetic_cells()
    folds = pd.Series([1 + i % 5 for i in range(30)], index=[f"B{i}" for i in range(30)])
    train0, test = H.outer_split(c, folds, 1)
    train, _ = H.outer_split(c, folds, 1, purge_m=200)
    xy = lambda d: d.groupby("BH_Code")[["UTM_Easting_m", "UTM_Northing_m"]].first().values
    assert H.nearest_distance(xy(train), xy(test)).min() >= 200
    assert train.BH_Code.nunique() < train0.BH_Code.nunique()
    assert not set(train.BH_Code) & set(test.BH_Code)


def test_selection_rule_prefers_lower_sd_within_one_se():
    base = {"multiple": 1, "purge_x": 0, "cells": "all", "n_folds": 5, "recall_clay_mean": 0.6, "recall_sand_mean": 0.6,
            "recall_clay_min": 0.4, "recall_sand_min": 0.4, "brier_mean": 0.2}
    summ = pd.DataFrame([{**base, "model": "A", "macro_f1_mean": 0.65, "macro_f1_sd": 0.10},
                         {**base, "model": "B", "macro_f1_mean": 0.63, "macro_f1_sd": 0.03},
                         {**base, "model": "C", "macro_f1_mean": 0.55, "macro_f1_sd": 0.01}])
    sel = H.select_model(summ)
    assert sel["highest_mean"] == "A" and sel["selected"] == "B" and "C" not in sel["within_1se"]
