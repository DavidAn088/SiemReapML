"""Nested spatial cross-validation (brief section 9).

For one run specification (scenario, block size, purge distance, model):

  for each outer fold k (R blockCV folds):
      test  = Downtown development boreholes in fold k
      train = scenario boreholes outside fold k (minus purged boreholes, if any)
      inner folds = the remaining outer folds of the Downtown training boreholes
                    (R fallback folds if they fail the adequacy check);
                    5 km ring boreholes are always inner-training
      Optuna (TPE, seeded) maximises inner out-of-fold Macro-F1; for every trial the
      decision threshold is the one that maximises Macro-F1 on the pooled inner OOF
      predictions, so thresholds are chosen without outer labels
      refit the best configuration on the whole outer-training set, predict fold k

With gap filling on (design report section 10), filled cells are training data only: the
outer test cells, the inner validation scores and the thresholds use measured cells, unless
`score_filled` is set (revision 3, section 12): then filled cells are also test and validation
cells, every prediction carries its `measured` flag, and metrics are reported on all cells and
on measured cells. The training weight of filled cells (`filled_weights`) is tuned in the
inner CV like a hyperparameter.

Results are checkpointed per outer fold, so an interrupted run resumes at the next fold.
Each fold's Optuna study runs in memory with a fixed seed; a fold is therefore either
complete and reproducible or re-run from its start.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from . import folds as F
from .config import seed_for, target_kind
from .data import FEATURES, class_info, clay_classes, measured, n_classes
from .metrics import best_threshold, multiclass_macro_f1, threshold_grid
from .models import DepthPrior, build, suggest

optuna.logging.set_verbosity(optuna.logging.WARNING)
BASELINE = "DEPTH_PRIOR"


def _measured_mask(cells: pd.DataFrame) -> np.ndarray:
    return cells.measured.values == 1 if "measured" in cells else np.ones(len(cells), bool)


def _scored_mask(cells: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Cells that are scored in validation: measured cells, or all cells with score_filled."""
    return np.ones(len(cells), bool) if cfg["depth"].get("score_filled") else _measured_mask(cells)


def _weights(meas: np.ndarray, params: dict) -> np.ndarray | None:
    """Training weights: 1 for measured cells, `filled_weight` for filled cells."""
    fw = params.get("filled_weight", 1.0)
    return None if fw == 1.0 or meas.all() else np.where(meas, 1.0, float(fw))


def _suggest_filled_weight(trial, cfg, meas) -> dict:
    fws = cfg["experiment"].get("filled_weights")
    if not fws or meas.all():
        return {}
    return {"filled_weight": trial.suggest_categorical("filled_weight", [float(w) for w in fws])}


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    block_m: int
    model: str
    purge_m: float = 0.0

    @property
    def tag(self) -> str:
        return f"{self.scenario}_{self.block_m}m_p{int(self.purge_m)}_{self.model}"


def _fit_predict(name, params, cfg, seed, smoke, Xtr, ytr, Xte, dtr=None, dte=None, K=2, wtr=None):
    """Fit and predict. Binary: returns P(Clay) as a vector; multiclass: the n x K matrix.
    `wtr`: optional training sample weights (filled-cell weight)."""
    params = {k: v for k, v in params.items() if k != "filled_weight"}
    if name == BASELINE:
        m = DepthPrior(bottom_m=cfg["depth"]["bottom_m"], n_classes=K).fit(dtr, ytr)
        P = m.predict_proba(dte)
    else:
        m = build(name, params, cfg, seed, smoke, n_classes=K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if wtr is None:
                m.fit(Xtr, ytr)
            else:
                m.fit(Xtr, ytr, sample_weight=wtr)
        P = m.predict_proba(Xte)
    return m, (P[:, 1] if K == 2 else P)


def _score(y, P, K, grid):
    """(threshold, Macro-F1): binary picks the threshold on P(Clay) that maximises Macro-F1;
    multiclass predicts the most probable class (no threshold)."""
    if K == 2:
        return best_threshold(y, P, grid)
    return np.nan, multiclass_macro_f1(y, P, K)


def _prediction_table(test, P, thr, K, cfg):
    cols = ["BH_Code", "cell", "depth_top_m", "UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m", "y"]
    cols += ["measured"] if "measured" in test else []
    if K == 2:
        pred = test[cols].copy()
        return pred.assign(p_clay=P, threshold=thr, y_pred=(P >= thr).astype(int))
    clay = clay_classes(cfg)
    pred = test[cols + ["y_bin"]].copy()
    yp = P.argmax(axis=1)
    pred = pred.assign(**{f"p_{c}": P[:, c] for c in range(K)})
    return pred.assign(y_pred=yp, p_clay=P[:, clay].sum(axis=1), y_pred_bin=np.isin(yp, clay).astype(int),
                       threshold=np.nan)


def run_fold(cells: pd.DataFrame, folds: pd.DataFrame, spec: RunSpec, k: int, cfg: dict, mode: str,
             n_trials: int | None = None, keep_model: bool = False) -> dict:
    smoke = mode == "smoke"
    train, test = F.split(cells, folds, k, spec.scenario, spec.purge_m)
    if not cfg["depth"].get("score_filled"):
        test = measured(test)
    groups, adq, source = F.inner_groups(train, folds, k, cfg, spec.block_m)
    Xtr, ytr, dtr = train[FEATURES].values, train.y.values, train.depth_top_m.values
    meas = _measured_mask(train)
    scored = _scored_mask(train, cfg)
    grid = threshold_grid(cfg)
    K = n_classes(cfg)
    seed = seed_for(cfg, spec.tag, k)
    inner_ids = sorted(set(groups[groups > 0]))
    t0 = time.time()

    def inner_oof(params):
        oof = np.full(len(train) if K == 2 else (len(train), K), np.nan)
        w = _weights(meas, params)
        for g in inner_ids:
            va = groups == g
            ev = va & scored  # fit on all cells outside the group, score its measured (or all) cells
            _, oof[ev] = _fit_predict(spec.model, params, cfg, seed, smoke, Xtr[~va], ytr[~va], Xtr[ev],
                                      dtr[~va], dtr[ev], K, None if w is None else w[~va])
        mask = np.isin(groups, inner_ids) & scored
        return oof, mask

    space = cfg["search_space"].get(spec.model, {})
    if spec.model == BASELINE or not space:  # depth-only reference; GP (kernel fitted by marginal likelihood)
        params = {}
        oof, mask = inner_oof(params)
        thr, inner_f1 = _score(ytr[mask], oof[mask], K, grid)
        trials = pd.DataFrame([{"number": 0, "value": inner_f1, "user_attrs_threshold": thr}])
    else:
        n_trials = n_trials or (cfg["smoke"]["n_trials"] if smoke else cfg["experiment"]["n_trials"][spec.model])

        def objective(trial):
            p = {**suggest(trial, space), **_suggest_filled_weight(trial, cfg, meas)}
            oof, mask = inner_oof(p)
            t, f = _score(ytr[mask], oof[mask], K, grid)
            trial.set_user_attr("threshold", t)
            per = [_score(ytr[(groups == g) & scored], oof[(groups == g) & scored], K, [t])[1] for g in inner_ids]
            trial.set_user_attr("inner_fold_macro_f1", [round(x, 4) for x in per])
            return f

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed_for(cfg, spec.tag, k, "tpe")))
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        params, thr, inner_f1 = study.best_params, study.best_trial.user_attrs["threshold"], study.best_value
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))

    w = _weights(meas, params)
    model, p_test = _fit_predict(spec.model, params, cfg, seed, smoke, Xtr, ytr, test[FEATURES].values,
                                 dtr, test.depth_top_m.values, K, w)
    dist = F.test_train_distances(train, test)
    pred = _prediction_table(test, p_test, thr, K, cfg)
    pred = pred.assign(dist_nearest_train_m=pred.BH_Code.map(dist).values, outer_fold=k, **asdict(spec))
    info = {**asdict(spec), "outer_fold": k, "mode": mode, "inner_source": source,
            "inner_adequacy": adq.to_dict("records"), "n_train_boreholes": int(train.BH_Code.nunique()),
            "n_train_filled_cells": int((~meas).sum()),
            "n_train_cells": int(len(train)), "n_test_boreholes": int(test.BH_Code.nunique()),
            "n_test_cells": int(len(test)), "n_test_filled_cells": int((test.measured == 0).sum()) if "measured" in test else 0,
            "best_params": params, "threshold": thr,
            "inner_macro_f1": float(inner_f1), "n_trials": int(len(trials)), "seconds": round(time.time() - t0, 1),
            "gw_fallback_rate": getattr(model, "fallback_rate_", None)}
    out = {"predictions": pred, "info": info, "trials": trials}
    if keep_model:
        out["model"] = model
    return out


def fingerprint(cells: pd.DataFrame, folds: pd.DataFrame, spec: RunSpec, cfg: dict, mode: str) -> str:
    """Hash of everything that determines a fold's result; a checkpoint is reused only if it matches."""
    import hashlib
    import json

    parts = {"spec": asdict(spec), "mode": mode, "seed": cfg["seeds"]["global"],
             "space": cfg["search_space"].get(spec.model), "n_trials": cfg["experiment"]["n_trials"].get(spec.model),
             "smoke": cfg["smoke"], "threshold_grid": cfg["experiment"]["threshold_grid"], "inner_cv": cfg["inner_cv"],
             "gw": cfg["gw"], "folds": pd.util.hash_pandas_object(folds.sort_index(), index=True).sum().item(),
             "cells": pd.util.hash_pandas_object(cells[["BH_Code", "cell", "y", *FEATURES]], index=False).sum().item()}
    if "measured" in cells:  # gap filling: which cells are measured is part of the experiment
        parts["measured"] = pd.util.hash_pandas_object(cells[["BH_Code", "cell", "measured"]], index=False).sum().item()
    if spec.model == "GP":  # added per model, so the fingerprints of the other models are unchanged
        parts["gp"] = cfg["gp"]
    if cfg["depth"].get("score_filled") or cfg["experiment"].get("filled_weights"):  # revision 3 settings
        parts["evaluation"] = {"score_filled": cfg["depth"].get("score_filled", False),
                               "filled_weights": cfg["experiment"].get("filled_weights")}
    if target_kind(cfg) != "binary":  # added for the multiclass target only (binary fingerprints unchanged)
        parts["target"] = cfg["target"]
        parts["classes"] = class_info(cfg)
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run(cells: pd.DataFrame, cfg: dict, spec: RunSpec, outer_folds: list[int], mode: str, out_dir: Path,
        save_models: bool = False, log=print) -> tuple[pd.DataFrame, list[dict]]:
    """Run (or resume) all requested outer folds of one specification."""
    folds = F.load_outer(cfg, spec.block_m)
    fp = fingerprint(cells, folds, spec, cfg, mode)
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    preds, infos = [], []
    for k in outer_folds:
        f = ck / f"{spec.tag}_fold{k}.joblib"
        if f.exists():
            res = joblib.load(f)
            if res["info"].get("fingerprint") != fp:
                raise RuntimeError(f"Checkpoint {f.name} was produced with different data, folds or settings. "
                                   "Delete outputs/.../03_nested_cv/checkpoints/ (and models/) to re-run.")
        else:
            res = run_fold(cells, folds, spec, k, cfg, mode, keep_model=save_models)
            res["info"]["fingerprint"] = fp
            if save_models:
                (out_dir / "models").mkdir(exist_ok=True)
                joblib.dump(res.pop("model"), out_dir / "models" / f"{spec.tag}_fold{k}.joblib", compress=3)
            joblib.dump(res, f, compress=3)
            log(f"  {spec.tag} fold {k}: inner Macro-F1 {res['info']['inner_macro_f1']:.3f}, "
                f"{res['info']['n_trials']} trials, {res['info']['seconds']:.0f} s")
        preds.append(res["predictions"])
        infos.append(res["info"])
        (out_dir / "trials").mkdir(exist_ok=True)
        res["trials"].assign(**asdict(spec), outer_fold=k).to_csv(out_dir / "trials" / f"{spec.tag}_fold{k}.csv",
                                                                   index=False)
    return pd.concat(preds, ignore_index=True), infos


def specs_from_config(cfg: dict, mode: str) -> list[RunSpec]:
    """Every run of the experiment: all models (+ reference) x 2 scenarios on the primary block
    size, the robustness models (+ reference) on the other block sizes, and the separate purge
    analysis on the primary block size."""
    ex = cfg["experiment"]
    blocks = (cfg["smoke"]["block_sizes_m"] or [F.primary_block(cfg)]) if mode == "smoke" else F.block_sizes(cfg)
    primary = int(blocks[0]) if mode == "smoke" else F.primary_block(cfg)
    reference = [BASELINE] if ex.get("reference_baseline") else []
    robust = list(ex.get("robustness_models", ex["models"]))
    specs = [RunSpec(s, int(b), m) for b in blocks for s in ex["scenarios"]
             for m in (list(ex["models"]) if int(b) == primary else robust) + reference]
    purge_d = ex.get("purge_distances_m") or [primary * m for m in ex.get("purge_multiples", [])]
    purge = purge_d[:1] if mode == "smoke" else purge_d
    purge_models = ex["purge_models"][:1] if mode == "smoke" else ex["purge_models"]
    specs += [RunSpec(s, primary, m, float(d)) for d in purge for s in ex["scenarios"] for m in purge_models]
    return specs


def fit_final(cells: pd.DataFrame, cfg: dict, scenario: str, model_name: str, block_m: int, mode: str,
              n_trials: int | None = None) -> dict:
    """Final model (brief section 12): tune on ALL development data of the scenario with the
    same procedure as the inner loop (groups = the k outer folds of the Downtown development
    boreholes; 5 km ring boreholes always in training), then refit on all development data."""
    smoke = mode == "smoke"
    folds = F.load_outer(cfg, block_m)
    flag = {"A": "in_A", "B": "in_B"}[scenario]
    train = cells[cells[flag]]
    groups = train.BH_Code.map(folds.fold).astype(int).values.copy()
    groups[groups == 0] = -1
    Xtr, ytr, dtr = train[FEATURES].values, train.y.values, train.depth_top_m.values
    grid = threshold_grid(cfg)
    K = n_classes(cfg)
    ids = sorted(set(groups[groups > 0]))
    spec_tag = f"final_{scenario}_{block_m}m_{model_name}"
    seed = seed_for(cfg, spec_tag)

    meas = _measured_mask(train)
    sc = _scored_mask(train, cfg)
    scored = (groups > 0) & sc

    def oof_of(params):
        oof = np.full(len(train) if K == 2 else (len(train), K), np.nan)
        w = _weights(meas, params)
        for g in ids:
            va = groups == g
            ev = va & sc
            _, oof[ev] = _fit_predict(model_name, params, cfg, seed, smoke, Xtr[~va], ytr[~va], Xtr[ev],
                                      dtr[~va], dtr[ev], K, None if w is None else w[~va])
        return oof

    if model_name == BASELINE or not cfg["search_space"].get(model_name):
        params = {}
        thr, f1 = _score(ytr[scored], oof_of(params)[scored], K, grid)
        trials = pd.DataFrame([{"number": 0, "value": f1, "user_attrs_threshold": thr}])
    else:
        n_trials = n_trials or (cfg["smoke"]["n_trials"] if smoke else cfg["experiment"]["n_trials"][model_name])

        def objective(trial):
            p = {**suggest(trial, cfg["search_space"][model_name]), **_suggest_filled_weight(trial, cfg, meas)}
            oof = oof_of(p)
            t, f = _score(ytr[scored], oof[scored], K, grid)
            trial.set_user_attr("threshold", t)
            return f

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed_for(cfg, spec_tag, "tpe")))
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        params, thr, f1 = study.best_params, study.best_trial.user_attrs["threshold"], study.best_value
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    model, _ = _fit_predict(model_name, params, cfg, seed, smoke, Xtr, ytr, Xtr[:1], dtr, dtr[:1], K,
                            _weights(meas, params))
    return {"model": model, "params": params, "threshold": thr, "cv_macro_f1": float(f1), "trials": trials,
            "scenario": scenario, "model_name": model_name, "block_m": block_m, "n_classes": K,
            "n_train_boreholes": int(train.BH_Code.nunique()), "n_train_cells": int(len(train)),
            "n_train_filled_cells": int((~meas).sum()),
            "top_elevation_range": (float(train.Top_Elevation_m.min()), float(train.Top_Elevation_m.max())),
            "train_bh_xy": train.drop_duplicates("BH_Code")[["UTM_Easting_m", "UTM_Northing_m"]].values}


def refit(train: pd.DataFrame, cfg: dict, model_name: str, params: dict, seed_tag: str, mode: str = "full"):
    """Refit a model with fixed hyperparameters (no tuning) on `train`, with the filled-cell
    weight in `params`; used for the production model of the 3D map (notebook 10)."""
    K = n_classes(cfg)
    meas = _measured_mask(train)
    model, _ = _fit_predict(model_name, params, cfg, seed_for(cfg, seed_tag), mode == "smoke",
                            train[FEATURES].values, train.y.values, train[FEATURES].values[:1],
                            train.depth_top_m.values, train.depth_top_m.values[:1], K, _weights(meas, params))
    return model
