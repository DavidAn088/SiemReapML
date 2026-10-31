"""Evaluation metrics (brief section 11) and the pre-declared model-selection rule (section 12).

Binary target: Clay = 1 is the positive class wherever a metric needs one; Sand-specific
values are reported separately (Sand recall, PR-AUC with Sand as positive).

Multiclass target (prediction tables with columns p_0 .. p_{K-1}): the predicted class is the
most probable one. Metrics: accuracy, balanced accuracy, Macro-F1, weighted F1, Cohen's kappa,
MCC, one-vs-rest ROC-AUC, multiclass Brier score and log loss, recall and F1 of every class,
and the same predictions merged back to Clay/Sand (bin_*)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

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


MC_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa", "mcc", "roc_auc_ovr",
              "brier", "log_loss", "bin_accuracy", "bin_macro_f1", "bin_recall_clay", "bin_recall_sand"]


def is_multiclass(preds: pd.DataFrame) -> bool:
    return "p_0" in preds.columns


def n_classes_of(preds: pd.DataFrame) -> int:
    return sum(c.startswith("p_") and c[2:].isdigit() for c in preds.columns)


def proba_matrix(preds: pd.DataFrame) -> np.ndarray:
    return preds[[f"p_{c}" for c in range(n_classes_of(preds))]].values


def multiclass_macro_f1(y, P, K) -> float:
    return f1_score(np.asarray(y, int), np.asarray(P).argmax(axis=1), average="macro", labels=list(range(K)),
                    zero_division=0)


def metric_row_mc(y, P, yhat, y_bin=None, yhat_bin=None) -> dict:
    """Multiclass metrics of one set of predictions (P: n x K probabilities)."""
    y, yhat, P = np.asarray(y, int), np.asarray(yhat, int), np.asarray(P, float)
    K = P.shape[1]
    labels = list(range(K))
    present = np.unique(y)
    row = {"n": int(len(y)), "accuracy": accuracy_score(y, yhat) if len(y) else np.nan,
           "balanced_accuracy": balanced_accuracy_score(y, yhat) if len(present) > 1 else np.nan,
           "macro_f1": f1_score(y, yhat, average="macro", labels=labels, zero_division=0),
           "weighted_f1": f1_score(y, yhat, average="weighted", labels=labels, zero_division=0),
           "kappa": cohen_kappa_score(y, yhat, labels=labels) if len(present) > 1 else np.nan,
           "mcc": matthews_corrcoef(y, yhat) if len(present) > 1 else np.nan,
           "roc_auc_ovr": (roc_auc_score(y, P, multi_class="ovr", average="macro", labels=labels)
                           if len(present) == K else np.nan),
           "brier": float(np.mean(np.sum((P - np.eye(K)[y]) ** 2, axis=1))) if len(y) else np.nan,
           "log_loss": log_loss(y, np.clip(P, 1e-15, 1), labels=labels) if len(y) else np.nan}
    rec = recall_score(y, yhat, average=None, labels=labels, zero_division=np.nan)
    f1c = f1_score(y, yhat, average=None, labels=labels, zero_division=0)
    for c in labels:
        row[f"recall_c{c}"] = rec[c] if (y == c).any() else np.nan
        row[f"f1_c{c}"] = f1c[c] if (y == c).any() or (yhat == c).any() else np.nan
        row[f"n_c{c}"] = int((y == c).sum())
    if y_bin is not None:
        yb, pb = np.asarray(y_bin, int), np.asarray(yhat_bin, int)
        row.update({"bin_accuracy": accuracy_score(yb, pb) if len(yb) else np.nan,
                    "bin_macro_f1": f1_score(yb, pb, average="macro", labels=[0, 1], zero_division=0),
                    "bin_recall_clay": recall_score(yb, pb, pos_label=1, zero_division=np.nan),
                    "bin_recall_sand": recall_score(yb, pb, pos_label=0, zero_division=np.nan)})
    return row


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
    rows, mc = [], is_multiclass(preds)
    pcols = [f"p_{c}" for c in range(n_classes_of(preds))] if mc else None
    for k, g in preds.groupby(keys, sort=True, dropna=False):
        key = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        if mc:
            rows.append({**key, **metric_row_mc(g.y, g[pcols].values, g.y_pred, g.y_bin, g.y_pred_bin)})
        else:
            rows.append({**key, **metric_row(g.y, g.p_clay, g.y_pred)})
    return pd.DataFrame(rows)


def metric_names(table: pd.DataFrame) -> list[str]:
    """Metrics summarised over folds: the binary set, or the multiclass set plus per-class values."""
    if "kappa" not in table.columns:
        return METRICS
    per = sorted([c for c in table.columns if c.startswith(("recall_c", "f1_c"))], key=lambda c: (c[:2], int(c.split("_c")[1])))
    return MC_METRICS + per


def fold_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics of each outer fold."""
    return _apply(preds, RUN_KEYS + ["outer_fold"])


def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and minimum over outer folds."""
    g = folds.groupby(RUN_KEYS)[metric_names(folds)]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.min().add_suffix("_min"))
    out["n_folds"] = folds.groupby(RUN_KEYS).size()
    return out.reset_index()


def pooled_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics of all out-of-fold predictions pooled (reported separately from fold means;
    pooled AUC mixes outer-fold models with different calibration)."""
    return _apply(preds, RUN_KEYS)


def borehole_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Per test borehole. Macro-F1 / balanced accuracy only where the borehole has both classes."""
    if is_multiclass(preds):
        rows = []
        for k, g in preds.groupby(RUN_KEYS + ["BH_Code"], sort=True):
            P = proba_matrix(g)
            rows.append({**dict(zip(RUN_KEYS + ["BH_Code"], k)), "outer_fold": int(g.outer_fold.iloc[0]),
                         "n": len(g), "classes": g.y.nunique(), "accuracy": (g.y == g.y_pred).mean(),
                         "bin_accuracy": (g.y_bin == g.y_pred_bin).mean(),
                         "macro_f1": f1_score(g.y, g.y_pred, average="macro", labels=sorted(g.y.unique()), zero_division=0)
                         if g.y.nunique() > 1 else np.nan,
                         "dist_nearest_train_m": g.dist_nearest_train_m.iloc[0],
                         "mean_entropy": entropy_mc(P).mean()})
        return pd.DataFrame(rows)
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


def entropy_mc(P) -> np.ndarray:
    """Normalised entropy of class probabilities (0 = certain, 1 = uniform over all classes)."""
    P = np.clip(np.asarray(P, float), 1e-12, 1)
    return -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])


def reliability_table(preds: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    if is_multiclass(preds):  # top-label calibration: confidence of the predicted class vs accuracy
        P = proba_matrix(preds)
        d = preds.assign(confidence=P.max(axis=1), correct=(preds.y == preds.y_pred).astype(float))
        d = d.assign(bin=pd.cut(d.confidence, edges, include_lowest=True))
        return (d.groupby(RUN_KEYS + ["bin"], observed=True)
                .agg(mean_predicted=("confidence", "mean"), observed_clay=("correct", "mean"), n=("y", "size"))
                .reset_index())
    d = preds.assign(bin=pd.cut(preds.p_clay, edges, include_lowest=True))
    return (d.groupby(RUN_KEYS + ["bin"], observed=True)
            .agg(mean_predicted=("p_clay", "mean"), observed_clay=("y", "mean"), n=("y", "size")).reset_index())


def confusion_table(preds: pd.DataFrame, K: int, normalise: bool = True) -> pd.DataFrame:
    """Confusion matrix (rows = observed class, columns = predicted class); normalised to the
    share of each observed class (recall, %) as in Khamjan et al. (2026, Fig. 4a)."""
    cm = confusion_matrix(preds.y, preds.y_pred, labels=list(range(K))).astype(float)
    if normalise:
        cm = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in range(K)], columns=[f"pred_{c}" for c in range(K)])


def pr_auc_by_class(preds: pd.DataFrame) -> pd.DataFrame:
    """One-vs-rest PR-AUC of every class with its no-skill baseline (class prevalence)."""
    P, K = proba_matrix(preds), n_classes_of(preds)
    y = preds.y.values
    return pd.DataFrame([{"class": c, "pr_auc": average_precision_score(y == c, P[:, c]) if (y == c).any() else np.nan,
                          "baseline": float((y == c).mean())} for c in range(K)])


def select_model_mc(summary: pd.DataFrame, cfg: dict, primary_block: int, exclude=("DEPTH_PRIOR",)) -> dict:
    """Pre-declared rule for the multiclass target (design report section 11), applied to the
    primary (autocorrelation) block size:
    1. eligible: the mean recall of EVERY class is at least 1/K (no class is ignored);
    2. rank by mean outer-fold Macro-F1;
    3. among eligible candidates within one standard error of the best, prefer the lowest fold
       SD of Macro-F1, then the lowest mean log loss."""
    s = summary[(summary.block_m == primary_block) & (summary.purge_m == 0) & ~summary.model.isin(exclude)].copy()
    rec = sorted([c for c in s.columns if c.startswith("recall_c") and c.endswith("_mean")])
    K = len(rec)
    s["min_class_recall_mean"] = s[rec].min(axis=1)
    s["eligible"] = s.min_class_recall_mean >= 1.0 / K
    e = s[s.eligible]
    if e.empty:
        return {"selected": None, "reason": f"no candidate has every class recall >= 1/{K}", "table": s}
    best = e.loc[e.macro_f1_mean.idxmax()]
    se = best.macro_f1_sd / np.sqrt(best.n_folds) if best.n_folds > 1 else 0.0
    near = e[e.macro_f1_mean >= best.macro_f1_mean - se].sort_values(["macro_f1_sd", "log_loss_mean"])
    pick = near.iloc[0]
    return {"selected": {"scenario": pick.scenario, "model": pick.model, "block_m": int(pick.block_m)},
            "best_macro_f1_mean": float(best.macro_f1_mean), "one_se": float(se),
            "candidates_within_1se": near[RUN_KEYS].to_dict("records"), "table": s}


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
