"""Checks of the V2 regression rules in helpers.py.  Run:  python -m pytest -q tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import helpers as H  # noqa: E402

S = H.load_settings(ROOT)


def records(rows, uscs="CL"):
    df = pd.DataFrame(rows, columns=["BH_Code", "Depth_From_m", "Depth_To_m", "SPT_N"])
    return df.assign(UTM_Easting_m=0.0, UTM_Northing_m=0.0, Ground_Elevation_m=10.0, USCS=uscs, Region="Downtown")


def test_sample_filters():
    r = records([("B", 1.0, 1.45, 5), ("B", 2.0, 6.0, 8), ("B", 15.0, 15.45, 20), ("B", 3.0, 3.45, 500),
                 ("B", 4.0, 4.45, None), ("B", 14.5, 14.95, 30)])
    s, info = H.make_samples(r, "SPT_N", S)
    assert list(s.y) == [5, 30]
    assert info["dropped_longer_than_3m"] == 1 and info["dropped_outside_depth"] == 1 and info["dropped_out_of_range"] == 1
    assert np.allclose(s.Mid_Elevation_m, [10 - 1.225, 10 - 14.725])


def test_soil_class_is_the_fourth_input_and_others_are_dropped():
    r = records([("B", 1.0, 1.45, 5), ("B", 2.0, 2.45, 6), ("B", 3.0, 3.45, 7)])
    r["USCS"] = ["CH", "SM", "ML"]
    s, info = H.make_samples(r, "SPT_N", S)
    assert list(s.soil_clay) == [1.0, 0.0] and info["dropped_no_clay_sand_class"] == 1
    assert H.design(s).shape == (2, 4)


def test_three_sd_outlier_dropped():
    r = records([("B", d, d + 0.45, 10) for d in np.arange(0, 14, 0.5)] + [("B", 14.0, 14.45, 59)])
    s, info = H.make_samples(r, "SPT_N", S)
    assert info["dropped_beyond_sd"] == 1 and s.y.max() == 10


def test_depth_reference_uses_slice_means_and_overall_mean():
    m = H.Depth(1.0, 2).fit(None, [1, 3, 10], [0.2, 0.7, 5.5])
    assert np.allclose(m.predict(None, [0.9, 5.1, 9.0]), [2, 14 / 3, 14 / 3])


def test_selection_rule_within_one_se():
    f = pd.DataFrame({"target": "T", "block_m": 106, "model": ["RF"] * 5 + ["XGB"] * 5 + ["Depth"] * 5,
                      "outer_fold": list(range(1, 6)) * 3,
                      "rmse": [1, 3, 1, 3, 2] + [2.1, 2.1, 2.1, 2.1, 2.1] + [4, 4, 4, 4, 4],
                      "mae": 1.0, "r2": 0.0, "bias": 0.0, "spearman": 0.0})
    summ = H.summary(f)
    c = H.select_model(summ, "T", 106)
    assert c["lowest_mean"] == "RF" and c["selected"] == "XGB"


def test_all_five_models_fit_and_predict():
    rng = np.random.default_rng(0)
    n = 30 * 12
    X = np.c_[np.repeat(rng.uniform(0, 1000, (30, 2)), 12, axis=0), rng.uniform(0, 15, n), rng.integers(0, 2, n)]
    y = 5 + X[:, 2] + rng.normal(0, 1, n)
    params = {"RF": {"n_estimators": 50}, "XGB": {"n_estimators": 50},
              "MLP": {"n_layers": 1, "n_units": 8, "max_iter": 200},
              "GWRF": {"k_neighbours": 10, "alpha": 0.5, "kernel": "bisquare", "max_depth": 5},
              "GWXGB": {"k_neighbours": 10, "alpha": 0.5, "kernel": "uniform", "n_estimators": 50}}
    for name, p in params.items():
        m = H.make_model(name, p, 1, "SPT_N", S).fit(X, y)
        pred = m.predict(X)
        assert pred.shape == (n,) and np.isfinite(pred).all() and H.rmse(y, pred) < 3, name
