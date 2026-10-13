"""Unit tests on synthetic data (no project data needed): python -m pytest tests -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from siemreap import config, data, folds as F, metrics as M  # noqa: E402
from siemreap.gw import GWClassifier  # noqa: E402
from siemreap.models import build  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return config.load_config(ROOT)


def raw_records(rows):
    cols = ["BH_Code", "Project_No", "Region", "UTM_Easting_m", "UTM_Northing_m", "Ground_Elevation_m",
            "Depth_From_m", "Depth_To_m", "USCS", "Sample_Type", "Coordinate"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "csv_line", np.arange(len(df)) + 2)
    return df


def test_discretise_labels_only_sampled_cells(cfg):
    r = [  # one borehole, ground 15 m
        ("B1", "P", "Downtown", 0, 0, 15.0, 1.0, 1.45, "CL", "SS", None),   # cell 2 (1.0-1.5), 90 % covered
        ("B1", "P", "Downtown", 0, 0, 15.0, 2.0, 2.10, "SC", "SS", None),   # cell 4, 20 % covered -> dropped
        ("B1", "P", "Downtown", 0, 0, 15.0, 3.0, 3.50, "ML", "ST", None),   # outside the binary map -> no label
        ("B1", "P", "Downtown", 0, 0, 15.0, 4.0, 4.50, "SM", "ST", None),   # cell 8, Sand
        ("B1", "P", "Downtown", 0, 0, 15.0, 5.0, 9.00, "CL", "D", None),    # 4 m record > max_interval -> ignored
        ("B1", "P", "Downtown", 0, 0, 15.0, 10.0, 10.5, "CL", "ST", None),  # cell 20 Clay ...
        ("B1", "P", "Downtown", 0, 0, 15.0, 10.25, 10.5, "SC", "SS", None),  # ... but overlapped by Sand -> conflict
    ]
    cells, info = data.discretise(raw_records(r), cfg)
    assert list(cells.cell) == [2, 8]
    assert list(cells.y) == [1, 0]
    assert np.allclose(cells.Top_Elevation_m, 15.0 - cells.depth_top_m)
    assert info["conflicting_cells_dropped"] == 1 and info["low_coverage_cells_dropped"] == 1
    assert cells.csv_lines.iloc[0] == "2"  # traceable to the source record


def synthetic_cells(n_bh=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bh):
        x, y = rng.uniform(0, 3000, 2)
        g = 15 + rng.normal(0, 1)
        for j in range(30):
            d = 0.5 * j
            p = 1 / (1 + np.exp(-(0.002 * (x - 1500) + 0.4 * (7 - d))))
            rows.append({"BH_Code": f"B{b:02d}", "cell": j, "depth_top_m": d, "UTM_Easting_m": x, "UTM_Northing_m": y,
                         "Top_Elevation_m": g - d, "y": int(rng.random() < p), "Region": "Downtown"})
    c = pd.DataFrame(rows)
    c["set"], c["is_downtown"], c["in_A"], c["in_B"] = "development", True, True, True
    return c


def test_gw_lazy_fitting_equals_fitting_every_anchor(cfg):
    c = synthetic_cells()
    X, y = c[["UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m"]].values, c.y.values
    tr, te = c.BH_Code < "B30", c.BH_Code >= "B30"
    kw = dict(base="rf", k_neighbours=10, alpha=0.6, kernel="bisquare", learner_params={"max_depth": 6},
              local_n_estimators=20, global_n_estimators=40, random_state=7, n_jobs=2)
    lazy = GWClassifier(**kw).fit(X[tr], y[tr])
    p_lazy = lazy.predict_proba(X[te])[:, 1]
    eager = GWClassifier(**kw).fit(X[tr], y[tr])
    eager.ensure_local(np.arange(len(eager.sites_)))  # fit every anchor first
    assert np.array_equal(p_lazy, eager.predict_proba(X[te])[:, 1])
    assert len(lazy.local_) < len(eager.local_)


def test_gw_falls_back_when_neighbourhood_has_one_class():
    X = np.array([[0, 0, 1], [0, 0, 2], [10, 0, 1], [10, 0, 2], [1000, 0, 1], [1000, 0, 2]], float)
    y = np.array([1, 1, 1, 1, 0, 0])
    m = GWClassifier(base="xgb", k_neighbours=2, min_boreholes=2, min_cells_per_class=3, max_neighbours=3,
                     learner_params={"max_depth": 2}, local_n_estimators=5, global_n_estimators=5).fit(X, y)
    assert m.fallback_rate_ == 1.0  # no neighbourhood holds 3 cells of each class
    assert np.allclose(m.predict_proba(X)[:, 1], m.global_.predict_proba(X)[:, 1])


@pytest.mark.parametrize("name,params", [
    ("RF", {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 2, "max_features": 2, "max_samples": 0.8,
            "class_weight": "balanced"}),
    ("XGB", {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1.0, "subsample": 0.8,
             "colsample_bynode": 1.0, "reg_lambda": 1.0, "gamma": 0.0, "class_weight": "none"}),
    ("MLP", {"n_layers": 1, "n_units": 16, "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3,
             "batch_size": 32, "max_iter": 200, "class_weight": "balanced"}),
])
def test_models_are_deterministic(cfg, name, params):
    c = synthetic_cells(20)
    X, y = c[["UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m"]].values, c.y.values
    p1 = build(name, params, cfg, seed=3).fit(X, y).predict_proba(X)[:, 1]
    p2 = build(name, params, cfg, seed=3).fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(p1, p2)


def test_split_keeps_boreholes_whole_and_purges(cfg):
    c = synthetic_cells(30)
    bh = sorted(c.BH_Code.unique())
    folds = pd.DataFrame({"BH_Code": bh, "fold": [i % 5 + 1 for i in range(len(bh))], "role": "cv",
                          "block_size_m": 100}).set_index("BH_Code")
    for k in range(1, 6):
        tr, te = F.split(c, folds, k, "A")
        assert not set(tr.BH_Code) & set(te.BH_Code)
        assert len(tr) + len(te) == len(c)
        trp, _ = F.split(c, folds, k, "A", purge_m=500)
        d = F.test_train_distances(trp, te)
        assert (d >= 500).all()


def test_external_boreholes_never_enter_development(cfg):
    c = synthetic_cells(10)
    mt = data.model_table(c.drop(columns=["set", "is_downtown", "in_A", "in_B"]), pd.Index(["B03"]), cfg,
                          flags=pd.DataFrame(columns=["BH_Code", "exclude_from_training", "exclude_from_external", "reason"]))
    assert not mt[mt.BH_Code == "B03"].in_A.any() and not mt[mt.BH_Code == "B03"].in_B.any()
    assert (mt[mt.BH_Code == "B03"].set == "external").all()


def test_threshold_and_metrics():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.35, 0.4, 0.8, 0.9])
    t, f = M.best_threshold(y, p, np.round(np.arange(0.05, 0.96, 0.01), 2))
    assert f == 1.0 and t == pytest.approx(0.40)  # separable in (0.35, 0.40]; ties go to the value nearest 0.5
    r = M.metric_row(y, p, (p >= 0.5).astype(int))
    assert r["recall_clay"] == pytest.approx(2 / 3) and r["recall_sand"] == 1.0 and r["tp"] == 2


def test_gp_is_deterministic_and_chunked_prediction_matches(cfg):
    c = synthetic_cells(12)
    X, y = c[["UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m"]].values, c.y.values
    small = {**cfg, "gp": {**cfg["gp"], "predict_chunk_rows": 7}}
    p1 = build("GP", {}, cfg, seed=3).fit(X, y).predict_proba(X)[:, 1]
    p2 = build("GP", {}, cfg, seed=3).fit(X, y).predict_proba(X)[:, 1]
    p3 = build("GP", {}, small, seed=3).fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(p1, p2)
    assert np.allclose(p1, p3, rtol=0, atol=1e-9)  # chunks change only floating-point rounding
    assert ((p1 > 0) & (p1 < 1)).all()


def test_robustness_models_only_off_the_primary_block(cfg):
    from siemreap.nested_cv import BASELINE, specs_from_config
    specs = [s for s in specs_from_config(cfg, "full") if s.purge_m == 0]
    primary = cfg["blockcv"]["primary_block_m"]
    on = {s.model for s in specs if s.block_m == primary}
    off = {s.model for s in specs if s.block_m != primary}
    assert on == set(cfg["experiment"]["models"]) | {BASELINE}
    assert off == set(cfg["experiment"]["robustness_models"]) | {BASELINE}
