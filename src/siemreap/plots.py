"""Figures for the notebooks (matplotlib; validated categorical palette, thin marks)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SCENARIO = {"A": BLUE, "B": ORANGE}
SCENARIO_LABEL = {"A": "A: Downtown only", "B": "B: Downtown + 5 km"}
CLASS = {1: BLUE, 0: ORANGE}
MODEL_ORDER = ["RF", "XGB", "MLP", "GWRF", "GWXGB", "GP", "DEPTH_PRIOR"]  # palette slot order; colour follows the model


def style(ax, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def _models(df):
    return [m for m in MODEL_ORDER if m in set(df.model)]


def fold_dotplot(folds: pd.DataFrame, metric="macro_f1", title=None, ax=None):
    """Each outer fold as a dot, the fold mean as a bar, per model and scenario."""
    models = _models(folds)
    ax = ax or plt.subplots(figsize=(8, 3.8), dpi=150)[1]
    for j, sc in enumerate(sorted(folds.scenario.unique())):
        for i, m in enumerate(models):
            v = folds[(folds.model == m) & (folds.scenario == sc)][metric].dropna()
            if v.empty:
                continue
            x = i + (j - 0.5) * 0.3
            ax.scatter(np.full(len(v), x), v, s=18, color=SCENARIO.get(sc, AQUA), alpha=0.7, edgecolor="white",
                       linewidth=0.5, label=SCENARIO_LABEL.get(sc, sc) if i == 0 else None, zorder=3)
            ax.hlines(v.mean(), x - 0.11, x + 0.11, color=SCENARIO.get(sc, AQUA), linewidth=2.2, zorder=4)
    ax.set_xticks(range(len(models)), [m.replace("_", " ") for m in models])
    ax.set_ylabel(metric.replace("_", " "), color=INK)
    ax.set_title(title or f"{metric.replace('_', ' ')} by outer fold (dots) and mean (bar)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    return style(ax, "y")


MODEL_COLOURS = {"RF": "#2a78d6", "XGB": "#eb6834", "MLP": "#1baf7a", "GWRF": "#eda100", "GWXGB": "#e87ba4",
                 "GP": "#008300", "DEPTH_PRIOR": "#52514e"}


def band_plot(tab: pd.DataFrame, key: str, metric="macro_f1", title=None, axes=None):
    """Metric per depth band or distance bin; one panel per scenario, one line per model."""
    scen = sorted(tab.scenario.unique())
    if axes is None:
        fig, axes = plt.subplots(1, len(scen), figsize=(6.2 * len(scen), 3.8), dpi=150, sharey=True, squeeze=False)
        axes = axes.ravel()
    order = list(tab[key].cat.categories) if hasattr(tab[key], "cat") else list(dict.fromkeys(tab[key].astype(str)))
    for ax, sc in zip(axes, scen):
        for m in _models(tab):
            t = tab[(tab.model == m) & (tab.scenario == sc)]
            if t.empty:
                continue
            ax.plot([order.index(str(v)) for v in t[key].astype(str)], t[metric], marker="o", markersize=4,
                    linewidth=2 if m != "DEPTH_PRIOR" else 1.4, linestyle="-" if m != "DEPTH_PRIOR" else "--",
                    color=MODEL_COLOURS.get(m, INK2), label=m.replace("_", " "))
        ax.set_xticks(range(len(order)), order, fontsize=8)
        ax.set_title(f"{title or ''} - {SCENARIO_LABEL.get(sc, sc)}", loc="left", fontsize=10)
        style(ax)
    axes[0].set_ylabel(metric.replace("_", " "), color=INK)
    axes[-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return axes


def reliability_plot(rel: pd.DataFrame, title="Reliability (pooled out-of-fold)", ax=None):
    ax = ax or plt.subplots(figsize=(4.6, 4.4), dpi=150)[1]
    ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1)
    for m in _models(rel):
        for sc in sorted(rel.scenario.unique()):
            t = rel[(rel.model == m) & (rel.scenario == sc)]
            ax.plot(t.mean_predicted, t.observed_clay, marker="o", markersize=3, linewidth=1.2,
                    color=SCENARIO.get(sc, AQUA), alpha=0.8, label=f"{m} ({sc})")
    ax.set_xlabel("Mean predicted P(Clay)")
    ax.set_ylabel("Observed Clay share")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    return style(ax)


def profiles(preds: pd.DataFrame, bh_codes, cell_m=0.5, ncols=6, title=None):
    """Observed vs predicted columns for test boreholes (restricted: borehole codes)."""
    bh_codes = list(bh_codes)
    nrows = int(np.ceil(len(bh_codes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 3.6 * nrows), dpi=150, sharey=True, squeeze=False)
    for ax, code in zip(axes.ravel(), bh_codes):
        g = preds[preds.BH_Code == code].sort_values("depth_top_m")
        for x0, col in ((0.0, "y"), (1.1, "y_pred")):
            for _, r in g.iterrows():
                ax.add_patch(plt.Rectangle((x0, r.depth_top_m), 1.0, cell_m, color=CLASS[int(r[col])], lw=0))
        ax.plot(2.2 + g.p_clay, g.depth_top_m + cell_m / 2, color=INK, linewidth=1)
        ax.axvline(2.2 + g.threshold.iloc[0], color=INK2, linestyle=":", linewidth=0.8)
        ax.set_xlim(-0.1, 3.3)
        ax.set_xticks([0.5, 1.6, 2.7], ["obs", "pred", "P(clay)"], fontsize=7)
        ax.set_title(f"{code[-18:]}\nfold {int(g.outer_fold.iloc[0])}, {g.dist_nearest_train_m.iloc[0]:.0f} m",
                     fontsize=6.5)
    for ax in axes.ravel()[len(bh_codes):]:
        ax.axis("off")
    axes[0, 0].set_ylim(15, 0)
    axes[0, 0].set_ylabel("Depth (m)")
    fig.suptitle(title or "Observed (blue = Clay, orange = Sand) vs predicted", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def error_map(bh_tab: pd.DataFrame, xy: pd.DataFrame, polygon=None, metric="accuracy", title=None, ax=None):
    """Borehole-level metric on a map (restricted: coordinates)."""
    ax = ax or plt.subplots(figsize=(7.5, 5.2), dpi=150)[1]
    if polygon is not None:
        bx, by = polygon.exterior.xy
        ax.plot(bx, by, color=INK2, linewidth=0.8)
    d = bh_tab.join(xy, on="BH_Code")
    sc = ax.scatter(d.UTM_Easting_m, d.UTM_Northing_m, c=d[metric], cmap="viridis", vmin=0, vmax=1, s=22,
                    edgecolor="white", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label=metric.replace("_", " "), shrink=0.8)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_title(title or f"Borehole {metric.replace('_', ' ')}", loc="left", fontsize=10)
    return style(ax)


def clay_share_by_depth(cells: pd.DataFrame, by="Region", slice_m=1.5, ax=None):
    ax = ax or plt.subplots(figsize=(4.8, 5.4), dpi=150)[1]
    c = cells.assign(s=np.floor(cells.depth_top_m / slice_m) * slice_m + slice_m / 2)
    for i, (k, g) in enumerate(c.groupby(by)):
        t = g.groupby("s").y.mean()
        ax.plot(t.values, t.index, marker="o", markersize=4, linewidth=1.8, color=[BLUE, ORANGE, AQUA, YELLOW][i % 4],
                label=f"{k} ({len(g)} cells)")
    ax.axvline(0.5, color=INK2, linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(15, 0)
    ax.set_xlabel("Clay share")
    ax.set_ylabel("Depth below ground (m)")
    ax.legend(frameon=False, fontsize=8)
    return style(ax)


# ------------------------------------------------------------- multiclass target ----
# Clay classes in blues and greens (soft -> hard), sand classes in oranges/yellows (loose -> dense)
CLAY_RAMP = ["#a6cbf2", "#2a78d6", "#0c3f86", "#1baf7a"]
SAND_RAMP = ["#f6c38a", "#eb6834", "#eda100"]


def class_colours(classes: list[dict]) -> list[str]:
    """Colour of each final class (classes.json order), from its group and rank in the group."""
    out, n = [], {"clay": 0, "sand": 0}
    for c in classes:
        ramp = CLAY_RAMP if c["group"] == "clay" else SAND_RAMP
        out.append(ramp[min(n[c["group"]], len(ramp) - 1)])
        n[c["group"]] += 1
    return out


def confusion_plot(cm: pd.DataFrame, names: list[str], title=None, ax=None):
    """Confusion matrix as % of each observed class (recall), as Khamjan et al. (2026, Fig. 4a)."""
    ax = ax or plt.subplots(figsize=(5.6, 4.8), dpi=150)[1]
    v = cm.values
    im = ax.imshow(v, cmap="Blues", vmin=0, vmax=100)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v[i, j] > 55 else INK)
    short = [n.replace(" clay", "\nclay").replace(" sand", "\nsand") for n in names]
    ax.set_xticks(range(len(names)), short, fontsize=7, rotation=0)
    ax.set_yticks(range(len(names)), short, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Observed class")
    plt.colorbar(im, ax=ax, shrink=0.8, label="% of observed class")
    ax.set_title(title or "Confusion matrix (% recall)", loc="left", fontsize=10)
    return ax


def pr_curves(preds: pd.DataFrame, names: list[str], colours: list[str], title=None, ax=None):
    """One-vs-rest precision-recall curve of every class with its no-skill baseline (dashed)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    ax = ax or plt.subplots(figsize=(5.2, 4.6), dpi=150)[1]
    y = preds.y.values
    for c, (name, col) in enumerate(zip(names, colours)):
        if not (y == c).any():
            continue
        pr, rc, _ = precision_recall_curve(y == c, preds[f"p_{c}"].values)
        ap = average_precision_score(y == c, preds[f"p_{c}"].values)
        ax.plot(rc, pr, color=col, linewidth=1.6, label=f"{name} (PR-AUC {ap:.2f}, baseline {(y == c).mean():.2f})")
        ax.axhline((y == c).mean(), color=col, linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title(title or "Precision-recall by class (one vs rest)", loc="left", fontsize=10)
    return style(ax)


def class_share_by_depth(cells: pd.DataFrame, names: list[str], colours: list[str], slice_m=1.5, bottom_m=20.0, ax=None):
    """Stacked class shares per depth slice (measured samples)."""
    ax = ax or plt.subplots(figsize=(5.0, 5.4), dpi=150)[1]
    s = np.floor(cells.depth_top_m / slice_m) * slice_m
    t = pd.crosstab(s, cells.y, normalize="index").reindex(columns=range(len(names)), fill_value=0)
    left = np.zeros(len(t))
    for c, (name, col) in enumerate(zip(names, colours)):
        ax.barh(t.index + slice_m / 2, t[c].values, height=slice_m * 0.92, left=left, color=col, label=name)
        left += t[c].values
    ax.set_ylim(bottom_m, 0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of samples")
    ax.set_ylabel("Depth below ground (m)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    return style(ax, "x")


def profiles_mc(preds: pd.DataFrame, bh_codes, colours: list[str], ncols=6, bottom_m=20.0, title=None):
    """Observed vs predicted classes of test boreholes (restricted: borehole codes)."""
    bh_codes = list(bh_codes)
    nrows = int(np.ceil(len(bh_codes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 3.6 * nrows), dpi=150, sharey=True, squeeze=False)
    for ax, code in zip(axes.ravel(), bh_codes):
        g = preds[preds.BH_Code == code].sort_values("depth_top_m")
        for x0, col in ((0.0, "y"), (1.1, "y_pred")):
            for _, r in g.iterrows():
                ax.add_patch(plt.Rectangle((x0, r.depth_top_m), 1.0, 0.6, color=colours[int(r[col])], lw=0))
        ax.set_xlim(-0.1, 2.2)
        ax.set_xticks([0.5, 1.6], ["obs", "pred"], fontsize=7)
        ax.set_title(f"{code[-18:]}\nfold {int(g.outer_fold.iloc[0])}, {g.dist_nearest_train_m.iloc[0]:.0f} m",
                     fontsize=6.5)
    for ax in axes.ravel()[len(bh_codes):]:
        ax.axis("off")
    axes[0, 0].set_ylim(bottom_m, 0)
    axes[0, 0].set_ylabel("Depth (m)")
    fig.suptitle(title or "Observed vs predicted soil class", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig
