"""Evaluation metrics (brief section 11) and the pre-declared model-selection rule (section 12).

Clay = 1 is the positive class wherever a metric needs one; Sand-specific values are
reported separately (Sand recall, PR-AUC with Sand as positive)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

RUN_KEYS = ["scenario", "block_m", "purge_m", "model"]
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "recall_clay", "recall_sand", "mcc", "roc_auc",
           "pr_auc_clay", "pr_auc_sand", "brier"]


def metric_row(y, p, yhat) -> dict:
    y, p, yhat = np.asarray(y, int), np.asarray(p, float), np.asarray(yhat, int)
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "clay_share": float(y.mean()) if len(y) else np.nan,
        "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat) if both else np.nan,
        "macro_f1": f1_score(y, yhat, average="macro", labels=[0, 1], zero_division=0),
        "recall_clay": recall_score(y, yhat, pos_label=1, zero_division=np.nan),
        "recall_sand": recall_score(y, yhat, pos_label=0, zero_division=np.nan),
        "precision_clay": precision_score(y, yhat, pos_label=1, zero_division=np.nan),
        "precision_sand": precision_score(y, yhat, pos_label=0, zero_division=np.nan),
        "mcc": matthews_corrcoef(y, yhat) if both else np.nan,
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "pr_auc_clay": average_precision_score(y, p) if both else np.nan,
        "pr_auc_sand": average_precision_score(1 - y, 1 - p) if both else np.nan,
        "brier": brier_score_loss(y, p, pos_label=1) if len(y) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y, p, grid) -> tuple[float, float]:
    """Threshold on P(Clay) that maximises Macro-F1 (ties -> closest to 0.5)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    best = (-1.0, 0.5)
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), average="macro", labels=[0, 1], zero_division=0)
        if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and abs(t - 0.5) < abs(best[1] - 0.5)):
            best = (f, float(t))
    return best[1], best[0]


def threshold_grid(cfg: dict) -> np.ndarray:
    a, b, s = cfg["experiment"]["threshold_grid"]
    return np.round(np.arange(a, b + s / 2, s), 4)


def _apply(preds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in preds.groupby(keys, sort=True, dropna=False):
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **metric_row(g.y, g.p_clay, g.y_pred)})
    return pd.DataFrame(rows)


def fold_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics of each outer fold."""
    return _apply(preds, RUN_KEYS + ["outer_fold"])


def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and minimum over outer folds."""
    g = folds.groupby(RUN_KEYS)[METRICS]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.min().add_suffix("_min"))
    out["n_folds"] = folds.groupby(RUN_KEYS).size()
    return out.reset_index()


def pooled_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics of all out-of-fold predictions pooled (reported separately from fold means;
    pooled AUC mixes outer-fold models with different calibration)."""
    return _apply(preds, RUN_KEYS)


def borehole_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Per test borehole. Macro-F1 / balanced accuracy only where the borehole has both classes."""
    rows = []
    for k, g in preds.groupby(RUN_KEYS + ["BH_Code"], sort=True):
        both = g.y.nunique() == 2
        rows.append({**dict(zip(RUN_KEYS + ["BH_Code"], k)), "outer_fold": int(g.outer_fold.iloc[0]),
                     "n": len(g), "clay_share": g.y.mean(), "accuracy": (g.y == g.y_pred).mean(),
                     "macro_f1": f1_score(g.y, g.y_pred, average="macro", labels=[0, 1], zero_division=0) if both else np.nan,
                     "balanced_accuracy": balanced_accuracy_score(g.y, g.y_pred) if both else np.nan,
                     "dist_nearest_train_m": g.dist_nearest_train_m.iloc[0],
                     "mean_entropy": entropy(g.p_clay).mean()})
    return pd.DataFrame(rows)


def depth_band_table(preds: pd.DataFrame, bands) -> pd.DataFrame:
    labels = [f"{a:g}-{b:g} m" for a, b in zip(bands[:-1], bands[1:])]
    d = preds.assign(depth_band=pd.cut(preds.depth_top_m, bands, right=False, labels=labels).astype(str))
    t = _apply(d, RUN_KEYS + ["depth_band"])
    t["depth_band"] = pd.Categorical(t.depth_band, labels, ordered=True)
    return t.sort_values(RUN_KEYS + ["depth_band"])


def distance_table(preds: pd.DataFrame, edges=(0, 50, 100, 200, 300, 500, 1000, np.inf)) -> pd.DataFrame:
    """Sensitivity to the horizontal distance between a test borehole and its nearest
    training borehole in the same outer fold."""
    labels = [f"{a:g}-{b:g} m" if np.isfinite(b) else f">{a:g} m" for a, b in zip(edges[:-1], edges[1:])]
    d = preds.assign(distance_bin=pd.cut(preds.dist_nearest_train_m, edges, right=False, labels=labels).astype(str))
    t = _apply(d, RUN_KEYS + ["distance_bin"])
    t["distance_bin"] = pd.Categorical(t.distance_bin, labels, ordered=True)
    return t.sort_values(RUN_KEYS + ["distance_bin"])


def entropy(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def reliability_table(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    d = preds.assign(bin=pd.cut(preds.p_clay, edges, include_lowest=True))
    return (d.groupby(RUN_KEYS + ["bin"], observed=True)
            .agg(mean_predicted=("p_clay", "mean"), observed_clay=("y", "mean"), n=("y", "size")).reset_index())


def select_model(summary: pd.DataFrame, cfg: dict, exclude=("DEPTH_PRIOR",)) -> dict:
    """Pre-declared rule (docs/01_source_review_and_design.md section 5.7), applied to the
    primary block size without purge:
    1. eligible: mean Clay and Sand recall >= 0.50 and no fold with a class recall < 0.25;
    2. rank by mean outer-fold Macro-F1;
    3. among eligible candidates within one standard error of the best, prefer the lowest
       fold SD of Macro-F1, then the lowest mean Brier score."""
    s = summary[(summary.block_m == cfg["blockcv"]["primary_block_m"]) & (summary.purge_m == 0)
                & ~summary.model.isin(exclude)].copy()
    s["eligible"] = ((s.recall_clay_mean >= 0.5) & (s.recall_sand_mean >= 0.5)
                     & (s.recall_clay_min >= 0.25) & (s.recall_sand_min >= 0.25))
    e = s[s.eligible]
    if e.empty:
        return {"selected": None, "reason": "no candidate meets the recall constraints", "table": s}
    best = e.loc[e.macro_f1_mean.idxmax()]
    se = best.macro_f1_sd / np.sqrt(best.n_folds) if best.n_folds > 1 else 0.0
    near = e[e.macro_f1_mean >= best.macro_f1_mean - se].sort_values(["macro_f1_sd", "brier_mean"])
    pick = near.iloc[0]
    return {"selected": {"scenario": pick.scenario, "model": pick.model, "block_m": int(pick.block_m)},
            "best_macro_f1_mean": float(best.macro_f1_mean), "one_se": float(se),
            "candidates_within_1se": near[RUN_KEYS].to_dict("records"), "table": s}
