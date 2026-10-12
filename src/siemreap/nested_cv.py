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
from .config import seed_for
from .data import FEATURES
from .metrics import best_threshold, threshold_grid
from .models import DepthPrior, build, suggest

optuna.logging.set_verbosity(optuna.logging.WARNING)
BASELINE = "DEPTH_PRIOR"


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    block_m: int
    model: str
    purge_m: float = 0.0

    @property
    def tag(self) -> str:
        return f"{self.scenario}_{self.block_m}m_p{int(self.purge_m)}_{self.model}"


def _fit_predict(name, params, cfg, seed, smoke, Xtr, ytr, Xte, dtr=None, dte=None):
    if name == BASELINE:
        m = DepthPrior(bottom_m=cfg["depth"]["bottom_m"]).fit(dtr, ytr)
        return m, m.predict_proba(dte)[:, 1]
    m = build(name, params, cfg, seed, smoke)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(Xtr, ytr)
    return m, m.predict_proba(Xte)[:, 1]


def run_fold(cells: pd.DataFrame, folds: pd.DataFrame, spec: RunSpec, k: int, cfg: dict, mode: str,
             n_trials: int | None = None, keep_model: bool = False) -> dict:
    smoke = mode == "smoke"
    train, test = F.split(cells, folds, k, spec.scenario, spec.purge_m)
    groups, adq, source = F.inner_groups(train, folds, k, cfg, spec.block_m)
    Xtr, ytr, dtr = train[FEATURES].values, train.y.values, train.depth_top_m.values
    grid = threshold_grid(cfg)
    seed = seed_for(cfg, spec.tag, k)
    inner_ids = sorted(set(groups[groups > 0]))
    t0 = time.time()

    def inner_oof(params):
        oof = np.full(len(train), np.nan)
        for g in inner_ids:
            va = groups == g
            _, oof[va] = _fit_predict(spec.model, params, cfg, seed, smoke, Xtr[~va], ytr[~va], Xtr[va],
                                      dtr[~va], dtr[va])
        mask = np.isin(groups, inner_ids)
        return oof, mask

    if spec.model == BASELINE:
        params = {}
        oof, mask = inner_oof(params)
        thr, inner_f1 = best_threshold(ytr[mask], oof[mask], grid)
        trials = pd.DataFrame([{"number": 0, "value": inner_f1, "user_attrs_threshold": thr}])
    else:
        space = cfg["search_space"][spec.model]
        n_trials = n_trials or (cfg["smoke"]["n_trials"] if smoke else cfg["experiment"]["n_trials"][spec.model])

        def objective(trial):
            p = suggest(trial, space)
            oof, mask = inner_oof(p)
            t, f = best_threshold(ytr[mask], oof[mask], grid)
            trial.set_user_attr("threshold", t)
            per = [best_threshold(ytr[groups == g], oof[groups == g], [t])[1] for g in inner_ids]
            trial.set_user_attr("inner_fold_macro_f1", [round(x, 4) for x in per])
            return f

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed_for(cfg, spec.tag, k, "tpe")))
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        params, thr, inner_f1 = study.best_params, study.best_trial.user_attrs["threshold"], study.best_value
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))

    model, p_test = _fit_predict(spec.model, params, cfg, seed, smoke, Xtr, ytr, test[FEATURES].values,
                                 dtr, test.depth_top_m.values)
    dist = F.test_train_distances(train, test)
    pred = test[["BH_Code", "cell", "depth_top_m", "UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m", "y"]].copy()
    pred = pred.assign(p_clay=p_test, threshold=thr, y_pred=(p_test >= thr).astype(int),
                       dist_nearest_train_m=pred.BH_Code.map(dist).values, outer_fold=k, **asdict(spec))
    info = {**asdict(spec), "outer_fold": k, "mode": mode, "inner_source": source,
            "inner_adequacy": adq.to_dict("records"), "n_train_boreholes": int(train.BH_Code.nunique()),
            "n_train_cells": int(len(train)), "n_test_boreholes": int(test.BH_Code.nunique()),
            "n_test_cells": int(len(test)), "best_params": params, "threshold": thr,
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
    """Every run of the experiment: 5 models (+ reference) x 2 scenarios x block sizes, plus
    the separate purge analysis on the primary block size."""
    ex, bc = cfg["experiment"], cfg["blockcv"]
    blocks = cfg["smoke"]["block_sizes_m"] if mode == "smoke" else bc["block_sizes_m"]
    models = list(ex["models"]) + ([BASELINE] if ex.get("reference_baseline") else [])
    specs = [RunSpec(s, int(b), m) for b in blocks for s in ex["scenarios"] for m in models]
    purge = ex["purge_distances_m"][:1] if mode == "smoke" else ex["purge_distances_m"]
    purge_models = ex["purge_models"][:1] if mode == "smoke" else ex["purge_models"]
    primary = int(blocks[0]) if mode == "smoke" else int(bc["primary_block_m"])
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
    ids = sorted(set(groups[groups > 0]))
    spec_tag = f"final_{scenario}_{block_m}m_{model_name}"
    seed = seed_for(cfg, spec_tag)

    def oof_of(params):
        oof = np.full(len(train), np.nan)
        for g in ids:
            va = groups == g
            _, oof[va] = _fit_predict(model_name, params, cfg, seed, smoke, Xtr[~va], ytr[~va], Xtr[va], dtr[~va], dtr[va])
        return oof

    if model_name == BASELINE:
        params = {}
        thr, f1 = best_threshold(ytr[groups > 0], oof_of(params)[groups > 0], grid)
        trials = pd.DataFrame([{"number": 0, "value": f1, "user_attrs_threshold": thr}])
    else:
        n_trials = n_trials or (cfg["smoke"]["n_trials"] if smoke else cfg["experiment"]["n_trials"][model_name])

        def objective(trial):
            p = suggest(trial, cfg["search_space"][model_name])
            oof = oof_of(p)
            t, f = best_threshold(ytr[groups > 0], oof[groups > 0], grid)
            trial.set_user_attr("threshold", t)
            return f

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed_for(cfg, spec_tag, "tpe")))
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        params, thr, f1 = study.best_params, study.best_trial.user_attrs["threshold"], study.best_value
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    model, _ = _fit_predict(model_name, params, cfg, seed, smoke, Xtr, ytr, Xtr[:1], dtr, dtr[:1])
    return {"model": model, "params": params, "threshold": thr, "cv_macro_f1": float(f1), "trials": trials,
            "scenario": scenario, "model_name": model_name, "block_m": block_m,
            "n_train_boreholes": int(train.BH_Code.nunique()), "n_train_cells": int(len(train)),
            "top_elevation_range": (float(train.Top_Elevation_m.min()), float(train.Top_Elevation_m.max())),
            "train_bh_xy": train.drop_duplicates("BH_Code")[["UTM_Easting_m", "UTM_Northing_m"]].values}
