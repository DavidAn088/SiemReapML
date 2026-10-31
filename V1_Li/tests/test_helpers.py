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


def synthetic_cells(n_bh=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bh):
        x, y = rng.uniform(0, 1000, 2)
        ground = 10 + rng.uniform(-1, 1)
        for j in range(20):
            rows.append({"BH_Code": f"B{b:02d}", "cell": j, "depth_top_m": j * 0.5, "UTM_Easting_m": x,
                         "UTM_Northing_m": y, "Ground_Elevation_m": ground, "Top_Elevation_m": ground - j * 0.5,
                         "Region": "Downtown", "y": int((x > 500) ^ (j > 10) ^ (rng.random() < 0.1)),
                         "measured": int(j % 4 != 0)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ VA-HDW features ----

def test_vahdw_values_on_a_known_borehole(S):
    # one reference borehole at (0, 0), ground 10 m: clay 0-2 m (cells 0-3), sand 2-5 m (cells 4-9), no label below
    ref = pd.DataFrame({"BH_Code": "R", "cell": range(10), "UTM_Easting_m": 0.0, "UTM_Northing_m": 0.0,
                        "Ground_Elevation_m": 10.0, "y": [1] * 4 + [0] * 6})
    refs = H.References(ref, S)
    q = pd.DataFrame({"UTM_Easting_m": [300.0, 0.0, 0.0, 0.0], "UTM_Northing_m": [400.0, 0.0, 0.0, 0.0],
                      "Top_Elevation_m": [9.5, 8.25, 6.75, 2.0]})            # centres at 9.25, 8.0, 6.5, 1.75 m
    d, phi, s = refs.features(q, S)
    assert d[0, 0] == pytest.approx(0.5)                                     # 500 m = 0.5 km
    assert phi[:, 0].tolist() == [1, 0, 0, 0]                                # last class (sand) below the borehole
    assert s[0, 0] == pytest.approx(2 * 0.75 / 2.0)                          # 0.75 m below the top of a 2 m layer
    assert s[1, 0] == pytest.approx(0.0)                                     # exactly on the clay/sand boundary
    assert s[2, 0] == pytest.approx(2 * 1.5 / 3.0)                           # middle of the 3 m sand layer -> 1
    assert s[3, 0] == 0.0                                                    # outside the borehole


def test_own_borehole_is_masked(S):
    c = synthetic_cells()
    refs = H.References(c, S)
    d, phi, s = refs.features(c, S)
    own = np.asarray(c.BH_Code, dtype=object)[:, None] == refs.codes[None, :]
    assert (d[own] == S["features"]["masked_distance_km"]).all() and (phi[own] == 0.5).all() and (s[own] == 0).all()


def test_no_leakage_test_labels_never_used(S):
    """Predictions for test cells must not change when the test cells' labels change."""
    c = synthetic_cells()
    test_bh = {"B00", "B01", "B02", "B03"}
    train, test = c[~c.BH_Code.isin(test_bh)], c[c.BH_Code.isin(test_bh)]
    params = {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1.0, "subsample": 1.0,
              "colsample_bynode": 1.0, "reg_lambda": 1.0, "gamma": 0.0, "class_weight": "none"}
    m = H.make_model("XGB", "XYZ_VAHDW", params, 1, S).fit(train)
    assert not set(m.refs.codes) & test_bh                                   # test boreholes are never references
    p1 = m.p_clay(test)
    p2 = m.p_clay(test.assign(y=1 - test.y))                                 # flip every test label
    assert np.array_equal(p1, p2)


# ------------------------------------------------------------------------- GBARNN ----

def test_gbarnn_gradients_match_numerical_derivatives(S):
    rng = np.random.default_rng(3)
    n, B = 5, 7
    net = H.GBARNN("GBARNN", {"g1": 4, "g2": 3, "u1": 6, "u2": 5, "class_weight": "none"}, 0, S)
    net.n_, net.b_ = n, rng.uniform(0.1, 0.9, n)
    P = net._init(rng)
    for k in P:                                                              # non-zero biases for a full check
        P[k] = P[k] + rng.normal(0, 0.1, np.shape(P[k]))
    F = rng.normal(size=(B, n, 3))
    Xs = rng.normal(size=(B, 3))
    mask = np.ones((B, n), bool)
    mask[0, 1] = False                                                       # one masked borehole
    y = rng.integers(0, 2, B).astype(float)
    sw = rng.uniform(0.5, 1.5, B)
    p, cache = net._forward(F, Xs, mask, P)
    g = net._backward(p, y, sw, cache, P)
    eps = 1e-6
    for k in P:
        flat = np.asarray(P[k]).reshape(-1)
        for idx in rng.choice(flat.size, min(4, flat.size), replace=False):
            Pp = {kk: np.array(vv, dtype=float) for kk, vv in P.items()}
            Pm = {kk: np.array(vv, dtype=float) for kk, vv in P.items()}
            Pp[k].reshape(-1)[idx] += eps
            Pm[k].reshape(-1)[idx] -= eps
            num = (net.loss(net._forward(F, Xs, mask, Pp)[0], y, sw) - net.loss(net._forward(F, Xs, mask, Pm)[0], y, sw)) / (2 * eps)
            assert np.asarray(g[k]).reshape(-1)[idx] == pytest.approx(num, rel=1e-4, abs=1e-7), k


def test_gbarnn_learns_and_gives_probabilities(S):
    c = synthetic_cells()
    params = {"g1": 6, "g2": 3, "u1": 32, "u2": 16, "learning_rate": 0.005, "epochs": 20, "batch_size": 64,
              "weight_decay": 1e-4, "class_weight": "balanced"}
    m = H.make_model("GBARNN", "XYZ_VAHDW", params, 1, S).fit(c)
    p = m.p_clay(c)
    assert len(p) == len(c) and np.all((p > 0) & (p < 1))


# ------------------------------------------------------------- all models, both sets ----

@pytest.mark.parametrize("name", ["RF", "XGB", "GWRF", "GWXGB", "KNN", "SVM"])
@pytest.mark.parametrize("features", ["XYZ", "XYZ_VAHDW"])
def test_every_model_on_both_feature_sets(S, name, features):
    c = synthetic_cells()
    space = S["space"][name]
    params = {k: (v["choices"][-1] if v["type"] == "categorical" else v["low"]) for k, v in space.items()}
    if "k_neighbours" in params:
        params["k_neighbours"] = 10
    params["filled_weight"] = 0.5
    m = H.make_model(name, features, params, 1, S).fit(c, H.fill_weights(c, params))
    p = m.p_clay(c)
    assert len(p) == len(c) and np.all((p >= 0) & (p <= 1))
    assert len(m.columns()) == (3 if features == "XYZ" else 3 + 3 * c.BH_Code.nunique())


def test_paired_gain_rule():
    rows = []
    for k, (a, b) in enumerate([(0.60, 0.66), (0.62, 0.67), (0.58, 0.65), (0.61, 0.66), (0.59, 0.66)], 1):
        for f, v in (("XYZ", a), ("XYZ_VAHDW", b)):
            rows.append({"model": "XGB", "features": f, "multiple": 1, "purge_x": 0, "cells": "all",
                         "outer_fold": k, "macro_f1": v})
    g = H.paired_gain(pd.DataFrame(rows))
    assert g.verdict.iloc[0] == "helps" and g.mean_gain.iloc[0] == pytest.approx(0.06)


def test_grid_search_covers_li_table6(S):
    S["run"]["quick"] = False
    assert H.n_trials("KNN", S) == 15 * 3 * 2                      # n_neighbors x leaf_size x p (no weights)
    assert H.n_trials("SVM", S) == 4 * 1 * 2 * len(S["cv"]["filled_weights"])
    assert H.n_trials("SVM", S, with_filled_weight=False) == 8


def test_run_list_models(S):
    runs = {(r["model"], r["features"]) for r in H.run_list(S) if r["multiple"] == 1 and r["purge_x"] == 0}
    assert ("KNN", "XYZ") in runs and ("SVM", "XYZ") in runs and ("KNN", "XYZ_VAHDW") not in runs
    assert ("GBARNN", "XYZ_VAHDW") in runs and not any(m == "MLP" for m, _ in runs)
