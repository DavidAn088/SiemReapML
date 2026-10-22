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
    gp = {"nu": 1.5, "amplitude": 1.0, "amplitude_bounds": [0.01, 1e4], "horizontal_length_m": 200.0,
          "horizontal_bounds_m": [5.0, 2e4], "vertical_length_m": 3.0, "vertical_bounds_m": [0.1, 100.0],
          "predict_chunk_rows": 10000}
    cfg = {**cfg, "gp": gp}                      # the GP is no longer in the configuration (revision 3)
    small = {**cfg, "gp": {**gp, "predict_chunk_rows": 7}}
    p1 = build("GP", {}, cfg, seed=3).fit(X, y).predict_proba(X)[:, 1]
    p2 = build("GP", {}, cfg, seed=3).fit(X, y).predict_proba(X)[:, 1]
    p3 = build("GP", {}, small, seed=3).fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(p1, p2)
    assert np.allclose(p1, p3, rtol=0, atol=1e-9)  # chunks change only floating-point rounding
    assert ((p1 > 0) & (p1 < 1)).all()


def test_every_model_on_every_block_unless_a_robustness_subset_is_set(cfg):
    from siemreap.nested_cv import BASELINE, specs_from_config
    primary = 100
    ex = {k: v for k, v in cfg["experiment"].items() if k != "robustness_models"}
    cfg = {**cfg, "blockcv": {**cfg["blockcv"], "auto_block_size": False, "block_sizes_m": [100, 200, 300],
                              "primary_block_m": 100}, "experiment": ex}

    def models_by_block(c):
        sp = [s for s in specs_from_config(c, "full") if s.purge_m == 0]
        return {s.model for s in sp if s.block_m == primary}, {s.model for s in sp if s.block_m != primary}

    on, off = models_by_block(cfg)
    assert on == off == set(cfg["experiment"]["models"]) | {BASELINE}
    on, off = models_by_block({**cfg, "experiment": {**cfg["experiment"], "robustness_models": ["RF"]}})
    assert on == set(cfg["experiment"]["models"]) | {BASELINE} and off == {"RF", BASELINE}


def test_fill_gaps_only_fills_untouched_same_class_gaps(cfg):
    r = [  # ground 15 m; cell j covers [0.5 j, 0.5 j + 0.5)
        ("B1", "P", "Downtown", 0, 0, 15.0, 1.0, 1.45, "SM", "SS", None),   # cell 2 Sand
        ("B1", "P", "Downtown", 0, 0, 15.0, 2.5, 2.95, "SP", "SS", None),   # cell 5 Sand -> cells 3, 4 filled
        ("B1", "P", "Downtown", 0, 0, 15.0, 4.0, 4.45, "CL", "SS", None),   # cell 8 Clay -> 6, 7 differ: not filled
        ("B1", "P", "Downtown", 0, 0, 15.0, 4.7, 4.90, "ML", "SS", None),   # touches cell 9
        ("B1", "P", "Downtown", 0, 0, 15.0, 5.5, 5.95, "CH", "SS", None),   # cell 11 Clay -> 9, 10 touched: not filled
    ]
    raw = raw_records(r)
    cells, _ = data.discretise(raw, cfg)
    on = {**cfg, "depth": {**cfg["depth"], "fill_gaps": True}}
    out, info = data.fill_gaps(cells, raw, on)
    filled = out[out.measured == 0]
    assert list(filled.cell) == [3, 4] and (filled.y == 0).all()
    assert info["different_class"] == 1 and info["touched_by_a_record"] == 1
    assert np.allclose(out.Top_Elevation_m, 15.0 - out.depth_top_m)
    assert out.cell.min() == 2  # nothing above the first measured cell
    off = {**cfg, "depth": {**cfg["depth"], "fill_gaps": False}}
    assert (data.fill_gaps(cells, raw, off)[0].measured == 1).all()


@pytest.mark.parametrize("score_filled", [False, True])
def test_filled_cells_are_scored_only_with_score_filled(cfg, score_filled):
    from siemreap.nested_cv import RunSpec, run_fold
    cfg = {**cfg, "depth": {**cfg["depth"], "score_filled": score_filled}}
    c = synthetic_cells(40)
    c["measured"] = np.where(c.cell % 3 == 1, 0, 1)  # every third cell plays a filled cell
    bh = sorted(c.BH_Code.unique())
    folds = pd.DataFrame({"BH_Code": bh, "fold": [i % 5 + 1 for i in range(len(bh))], "role": "cv",
                          "block_size_m": 100}).set_index("BH_Code")
    res = run_fold(c, folds, RunSpec("A", 100, "DEPTH_PRIOR"), 1, cfg, "full")
    test_bh = set(folds.index[folds.fold == 1])
    expect = c[c.BH_Code.isin(test_bh) & ((c.measured == 1) | score_filled)]
    assert len(res["predictions"]) == len(expect)
    assert (res["predictions"].measured == 0).any() == score_filled
    assert res["info"]["n_train_filled_cells"] > 0


def test_filled_weight_is_tuned_and_changes_training(cfg):
    from siemreap.nested_cv import RunSpec, run_fold
    cfg = {**cfg, "depth": {**cfg["depth"], "score_filled": True},
           "experiment": {**cfg["experiment"], "filled_weights": [0.25, 1.0]}}
    c = synthetic_cells(40)
    c["measured"] = np.where(c.cell % 3 == 1, 0, 1)
    bh = sorted(c.BH_Code.unique())
    folds = pd.DataFrame({"BH_Code": bh, "fold": [i % 5 + 1 for i in range(len(bh))], "role": "cv",
                          "block_size_m": 100}).set_index("BH_Code")
    res = run_fold(c, folds, RunSpec("A", 100, "XGB"), 1, cfg, "smoke", n_trials=3)
    assert res["info"]["best_params"]["filled_weight"] in (0.25, 1.0)
    p = {"n_estimators": 30, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1.0, "subsample": 1.0,
         "colsample_bynode": 1.0, "reg_lambda": 1.0, "gamma": 0.0, "class_weight": "none"}
    X, y = c[["UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m"]].values, c.y.values
    w = np.where(c.measured == 1, 1.0, 0.25)
    a = build("XGB", p, cfg, seed=1).fit(X, y).predict_proba(X[:20])
    b = build("XGB", p, cfg, seed=1).fit(X, y, sample_weight=w).predict_proba(X[:20])
    assert not np.allclose(a, b)


# ------------------------------------------------------------- 5-class target ----

@pytest.mark.parametrize("name", ["RF", "XGB", "MLP"])
def test_multiclass_models_return_all_classes_even_if_one_is_missing(cfg, name):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 100, (200, 3))
    y = rng.choice([0, 1, 3], 200)            # class 2 absent from training
    p = {"RF": {"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 2, "max_features": 2, "max_samples": 0.8,
                "class_weight": "balanced"},
         "XGB": {"n_estimators": 30, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 1.0, "subsample": 1.0,
                 "colsample_bynode": 1.0, "reg_lambda": 1.0, "gamma": 0.0, "class_weight": "balanced"},
         "MLP": {"n_layers": 1, "n_units": 8, "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3,
                 "batch_size": 32, "max_iter": 50, "class_weight": "none"}}[name]
    P = build(name, p, cfg, seed=1, n_classes=4).fit(X, y).predict_proba(X[:10])
    assert P.shape == (10, 4)
    assert np.allclose(P.sum(axis=1), 1) and np.all(P[:, 2] == 0)


def test_gw_multiclass_probabilities():
    rng = np.random.default_rng(1)
    sites = rng.uniform(0, 1000, (30, 2))
    X = np.vstack([np.column_stack([np.repeat(s[None], 12, 0), rng.uniform(0, 20, 12)]) for s in sites])
    y = (X[:, 2] // 5).astype(int) % 5
    m = GWClassifier(base="xgb", k_neighbours=8, learner_params={"max_depth": 2}, local_n_estimators=20,
                     global_n_estimators=20, min_cells_per_class=5, n_classes=5).fit(X, y)
    P = m.predict_proba(X[:40])
    assert P.shape == (40, 5) and np.allclose(P.sum(axis=1), 1)
    assert m.fallback_rate_ < 1


def test_multiclass_metrics_and_depth_prior():
    from siemreap.models import DepthPrior

    y = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    P = np.eye(5)[y] * 0.8 + 0.04
    r = M.metric_row_mc(y, P, P.argmax(axis=1), y_bin=(y < 3).astype(int), yhat_bin=(y < 3).astype(int))
    assert r["accuracy"] == 1 and r["macro_f1"] == 1 and r["kappa"] == 1 and r["bin_accuracy"] == 1
    dp = DepthPrior(slice_m=1.5, n_classes=5).fit(np.array([0.0, 0.5, 2.0, 2.5]), np.array([0, 0, 3, 4]))
    p = dp.predict_proba(np.array([0.2, 2.2, 30.0]))
    assert np.allclose(p[0], [1, 0, 0, 0, 0]) and np.allclose(p[1], [0, 0, 0, 0.5, 0.5])
    assert np.allclose(p[2], [0.5, 0, 0, 0.25, 0.25])  # unseen slice -> overall shares
