"""Stage 2 data audit of the Siem Reap borehole file.

Reads the received CSV without modifying it, checks it for internal consistency and
writes two kinds of output:

* outputs/01_audit/aggregate/  - counts and summaries with no borehole identifiers
  or coordinates (safe to commit to the public repository).
* outputs/01_audit/restricted/ - borehole-level inventory, issue log with source line
  numbers, and the borehole map (kept out of git; see .gitignore).

Nothing here builds folds, fits variograms or trains models. The 0.5 m discretisation
is a *preview* of one candidate rule so its consequences can be reviewed before the
preprocessing rule is fixed.

Usage:
    python scripts/01_audit_raw_data.py [--config config/project.toml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "01_audit"
AGG = OUT / "aggregate"
RES = OUT / "restricted"

# Chart colours (validated categorical slots 1-3 of the dataviz reference palette).
C_CLAY = "#2a78d6"
C_SAND = "#eb6834"
C_REGION = {"Downtown": "#2a78d6", "Downtown_5km": "#eb6834", "Downtown_10km": "#1baf7a", "Unassigned": "#8a8984"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

ISSUES: list[dict] = []


def issue(code: str, severity: str, bh: str, line: int | None, detail: str) -> None:
    ISSUES.append({"issue": code, "severity": severity, "BH_Code": bh, "csv_line": line, "detail": detail})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def style_axes(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(cfg: dict, allow_mismatch: bool) -> pd.DataFrame:
    path = ROOT / cfg["data"]["raw_csv"]
    digest = sha256(path)
    if digest != cfg["data"]["raw_csv_sha256"] and not allow_mismatch:
        sys.exit(f"Checksum mismatch for {path}: {digest}. Update config only if the new file is intended.")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    n_lines = sum(1 for _ in path.open("rb"))
    # One record per physical line is what makes csv_line traceable; fail loudly otherwise.
    assert n_lines == len(df) + 1, "CSV has embedded newlines; csv_line would not be traceable"
    df.insert(0, "csv_line", np.arange(len(df)) + 2)  # header is line 1
    df["Region"] = df["Region"].fillna("Unassigned")
    return df


def field_completeness(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "csv_line"]
    return pd.DataFrame(
        {"field": cols, "dtype": [str(df[c].dtype) for c in cols], "non_null": [int(df[c].notna().sum()) for c in cols]}
    ).assign(pct_complete=lambda t: (100 * t.non_null / len(df)).round(1))


def parse_coordinate(s, tr: Transformer):
    """Return (E, N, format) from the free-text Coordinate field."""
    if not isinstance(s, str):
        return np.nan, np.nan, "missing"
    m = re.match(r"\s*UTM48N:\s*N\s*([\d.]+),\s*E\s*([\d.]+)", s)
    if m:
        return float(m.group(2)), float(m.group(1)), "utm48n_text"
    lat, lon = (float(v) for v in s.split(","))
    e, n = tr.transform(lon, lat)
    return e, n, "latlon_text"


def borehole_inventory(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = df.groupby("BH_Code", sort=True)
    for col in ["UTM_Easting_m", "UTM_Northing_m", "Ground_Elevation_m", "Region", "Project_No"]:
        bad = g[col].nunique(dropna=False) > 1
        for bh in bad[bad].index:
            issue("I00", "high", bh, None, f"{col} not constant within borehole")

    bh = g.agg(
        Project_No=("Project_No", "first"),
        Region=("Region", "first"),
        E=("UTM_Easting_m", "first"),
        N=("UTM_Northing_m", "first"),
        Ground_Elevation_m=("Ground_Elevation_m", "first"),
        Coordinate=("Coordinate", "first"),
        n_records=("csv_line", "size"),
        first_line=("csv_line", "min"),
        depth_min_m=("Depth_From_m", "min"),
        depth_max_m=("Depth_To_m", "max"),
        n_uscs=("USCS", "count"),
    )
    lab = set(cfg["labels"]["clay"]) | set(cfg["labels"]["sand"])
    in_depth = df[df.Depth_From_m < cfg["depth"]["bottom_m"]]
    bh["n_binary_labels_0_15m"] = in_depth[in_depth.USCS.isin(lab)].groupby("BH_Code").size().reindex(bh.index).fillna(0).astype(int)

    # Ground elevation: float32-like values (> 2 decimals) are characteristic of a DEM extraction.
    z = bh.Ground_Elevation_m
    bh["elev_more_than_2dp"] = (z * 100 - np.round(z * 100)).abs() > 1e-6

    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{cfg['data']['crs_epsg']}", always_xy=True)
    parsed = bh.Coordinate.map(lambda s: parse_coordinate(s, tr))
    bh["coord_text_format"] = [p[2] for p in parsed]
    bh["coord_text_vs_utm_m"] = np.hypot([p[0] for p in parsed] - bh.E, [p[1] for p in parsed] - bh.N).round(1)

    dt = bh[bh.Region == cfg["regions"]["downtown"]]
    tree = cKDTree(dt[["E", "N"]].values)
    d, _ = tree.query(bh[["E", "N"]].values, k=2)
    is_dt = (bh.Region == cfg["regions"]["downtown"]).values
    # For Downtown boreholes the nearest neighbour is the 2nd hit (the 1st is itself).
    bh["dist_nearest_other_downtown_bh_m"] = np.where(is_dt, d[:, 1], d[:, 0]).round(1)

    for code, r in bh.iterrows():
        if r.coord_text_format == "missing":
            issue("I05", "low", code, r.first_line, "Coordinate text field empty; UTM fields used")
        elif r.coord_text_vs_utm_m > 5:
            issue("I04", "high" if r.coord_text_vs_utm_m > 50 else "medium", code, r.first_line,
                  f"Coordinate text and UTM fields disagree by {r.coord_text_vs_utm_m:.0f} m")
        if not r.elev_more_than_2dp:
            issue("I11", "medium", code, r.first_line,
                  f"Ground elevation {r.Ground_Elevation_m} m has <=2 decimals; checked against the DEM in 02_verify_sources.py")
        if r.depth_max_m < cfg["depth"]["bottom_m"]:
            issue("I13", "low", code, r.first_line, f"Borehole ends at {r.depth_max_m} m (< 15 m)")
    return bh


def record_checks(df: pd.DataFrame, cfg: dict) -> None:
    calc_top = df.Ground_Elevation_m - df.Depth_From_m
    for _, r in df[(calc_top - df.Top_Elevation_m).abs() > 1e-6].iterrows():
        issue("I01", "high", r.BH_Code, r.csv_line, "Top_Elevation_m != Ground_Elevation_m - Depth_From_m")
    calc_bot = df.Ground_Elevation_m - df.Depth_To_m
    for _, r in df[(calc_bot - df.Bottom_Elevation_m).abs() > 1e-6].iterrows():
        issue("I01", "high", r.BH_Code, r.csv_line, "Bottom_Elevation_m != Ground_Elevation_m - Depth_To_m")

    length = df.Depth_To_m - df.Depth_From_m
    for _, r in df[length <= 0].iterrows():
        issue("I02", "high", r.BH_Code, r.csv_line, "Depth_To_m <= Depth_From_m")
    for _, r in df[length > cfg["depth"]["max_interval_m"]].iterrows():
        issue("I02", "high", r.BH_Code, r.csv_line,
              f"Single record spans {r.Depth_From_m}-{r.Depth_To_m} m ({r.Sample_Type}, {r.USCS}); not a measured 0.5 m-scale stratum")

    for bh, d in df.sort_values(["BH_Code", "Depth_From_m"]).groupby("BH_Code"):
        f, t, ln = d.Depth_From_m.values, d.Depth_To_m.values, d.csv_line.values
        for i in np.where(f[1:] < t[:-1] - 1e-9)[0]:
            issue("I03", "medium", bh, int(ln[i + 1]),
                  f"Interval {f[i+1]}-{t[i+1]} m overlaps previous {f[i]}-{t[i]} m (line {ln[i]})")

    dup = df.duplicated(["BH_Code", "Depth_From_m", "Depth_To_m"], keep=False)
    for _, r in df[dup].iterrows():
        issue("I12", "high", r.BH_Code, r.csv_line, "Duplicate BH/depth interval")

    for _, r in df[df.USCS.isna()].iterrows():
        issue("I06", "info", r.BH_Code, r.csv_line, f"USCS missing ({r.Sample_Type} {r.Depth_From_m}-{r.Depth_To_m} m)")
    mapped = set(cfg["labels"]["clay"]) | set(cfg["labels"]["sand"])
    for _, r in df[df.USCS.notna() & ~df.USCS.isin(mapped)].iterrows():
        issue("I07", "medium", r.BH_Code, r.csv_line,
              f"USCS '{r.USCS}' outside binary map; fines={r.Silt_Clay_pct}, LL={r.LL_pct}, PI={r.PI_pct}")

    lab = df.dropna(subset=["Silt_Clay_pct", "USCS"])
    coarse_lab = lab.USCS.isin(cfg["labels"]["sand"] + ["SW", "GC", "GM"])
    fine_lab = lab.USCS.isin(cfg["labels"]["clay"] + ["ML", "CL-ML", "MH"])
    bad = (coarse_lab & (lab.Silt_Clay_pct >= 50)) | (fine_lab & (lab.Silt_Clay_pct < 50))
    for _, r in lab[bad].iterrows():
        issue("I08", "medium", r.BH_Code, r.csv_line,
              f"USCS {r.USCS} but lab fines {r.Silt_Clay_pct}% (ASTM D2487 boundary 50%)")

    pl = df.dropna(subset=["LL_pct", "PI_pct"])
    for _, r in pl[(pl.USCS == "CL") & (pl.LL_pct >= 50)].iterrows():
        issue("I09", "info", r.BH_Code, r.csv_line, f"CL with LL={r.LL_pct} (CH by definition; same binary class)")
    pp = df.dropna(subset=["LL_pct", "PL_pct", "PI_pct"])
    for _, r in pp[(pp.LL_pct - pp.PL_pct - pp.PI_pct).abs() > 0.5].iterrows():
        issue("I10", "low", r.BH_Code, r.csv_line, f"PI {r.PI_pct} != LL-PL ({r.LL_pct}-{r.PL_pct})")

    for _, r in df[df.Source_File.isna()].iterrows():
        issue("I15", "low", r.BH_Code, r.csv_line, "Source_File empty; record cannot be traced to a source report")

    std_types = {"SS", "ST", "SPT", "U", "D", "D/U", "Interval"}
    for _, r in df[~df.Sample_Type.isin(std_types)].iterrows():
        issue("I14", "low", r.BH_Code, r.csv_line, f"Non-standard Sample_Type value '{r.Sample_Type}'")


def discretise_preview(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Candidate rule: a 0.5 m cell inherits a class only from labelled sample intervals that
    overlap it, never from interpolation between samples. Long records are skipped."""
    dc = cfg["depth"]
    clay, sand = set(cfg["labels"]["clay"]), set(cfg["labels"]["sand"])
    d = df[df.USCS.isin(clay | sand) & ((df.Depth_To_m - df.Depth_From_m) <= dc["max_interval_m"])].copy()
    d["y"] = d.USCS.isin(clay).astype(int)
    edges = np.arange(dc["top_m"], dc["bottom_m"] + 1e-9, dc["cell_m"])
    rows = []
    for bh, b in d.groupby("BH_Code"):
        f, t, y, ln = b.Depth_From_m.values, b.Depth_To_m.values, b.y.values, b.csv_line.values
        for j in range(len(edges) - 1):
            top, bot = edges[j], edges[j + 1]
            ov = np.clip(np.minimum(t, bot) - np.maximum(f, top), 0, None)
            if not (ov > 0).any():
                continue
            cov_clay, cov_sand = ov[y == 1].sum(), ov[y == 0].sum()
            conflict = cov_clay > 0 and cov_sand > 0
            cov = max(cov_clay, cov_sand) / dc["cell_m"]
            label = np.nan if conflict else int(cov_clay > 0)
            rows.append((bh, b.Region.iloc[0], j, top, round(cov, 3), conflict, label,
                         ";".join(str(v) for v in ln[ov > 0])))
    cells = pd.DataFrame(rows, columns=["BH_Code", "Region", "cell", "depth_top_m", "coverage", "conflict", "y", "csv_lines"])
    cells["labelled"] = (~cells.conflict) & (cells.coverage >= dc["min_cell_coverage"] - 1e-9)
    return cells


def spatial_diagnostics(bh: pd.DataFrame, cfg: dict) -> dict:
    dt = bh[bh.Region == cfg["regions"]["downtown"]]
    X = dt[["E", "N"]].values
    nn = dt.dist_nearest_other_downtown_bh_m.values
    out = {
        "downtown_extent_m": {"easting": round(float(np.ptp(X[:, 0]))), "northing": round(float(np.ptp(X[:, 1])))},
        "downtown_nn_distance_m": {q: round(float(np.percentile(nn, p)), 1) for q, p in
                                   [("min", 0), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("max", 100)]},
        "downtown_nn_below_m": {str(t): int((nn < t).sum()) for t in (10, 25, 50, 100, 200, 300)},
    }
    Z = linkage(X, "single")
    out["single_linkage_site_clusters"] = {}
    for h in (25, 50, 100, 200, 300):
        s = pd.Series(fcluster(Z, h, "distance")).value_counts()
        out["single_linkage_site_clusters"][str(h)] = {"clusters": int(s.size), "largest": int(s.max()), "singletons": int((s == 1).sum())}
    # Geometry-only planning diagnostic on an axis-aligned square grid anchored at the data
    # minimum. This is NOT blockCV and is never used to assign folds.
    out["square_grid_planning_diagnostic"] = {}
    tree = cKDTree(X)
    for size in (100, 200, 300):
        cell = np.floor((X[:, 0] - X[:, 0].min()) / size) * 1e6 + np.floor((X[:, 1] - X[:, 1].min()) / size)
        dist, idx = tree.query(X, k=len(X))
        sep = np.array([dist[i][cell[idx[i]] != cell[i]][0] for i in range(len(X))])
        occ = pd.Series(cell).value_counts()
        out["square_grid_planning_diagnostic"][str(size)] = {
            "occupied_cells": int(occ.size), "max_bh_per_cell": int(occ.max()),
            "median_dist_to_nearest_bh_in_other_cell_m": round(float(np.median(sep))),
            "bh_with_other_cell_neighbour_within_25m": int((sep < 25).sum()),
            "bh_with_other_cell_neighbour_within_100m": int((sep < 100).sum()),
        }
    for r in ("Downtown_5km", "Downtown_10km", "Unassigned"):
        s = bh[bh.Region == r]
        if len(s):
            v = s.dist_nearest_other_downtown_bh_m
            out[f"{r}_dist_to_nearest_downtown_bh_m"] = {"n": int(len(s)), "min": round(v.min()), "median": round(v.median()), "max": round(v.max())}
    return out


def fig_depth_profile(cells: pd.DataFrame, cfg: dict, bin_m: float = 1.5) -> None:
    """Clay share in 1.5 m depth bins (left) beside the number of labelled 0.5 m cells per
    depth (right). The share is binned at 1.5 m because most logs follow a 1.5 m sampling
    cycle that leaves every third 0.5 m cell almost empty."""
    L = cells[cells.labelled].copy()
    half = cfg["depth"]["cell_m"] / 2
    L["bin_top"] = np.floor(L.depth_top_m / bin_m + 1e-9) * bin_m
    fig, (ax, axn) = plt.subplots(1, 2, figsize=(8.0, 6.2), dpi=200, sharey=True,
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    regions = (("Downtown", C_REGION["Downtown"], -0.06), ("Downtown_5km", C_REGION["Downtown_5km"], 0.06))
    for region, colour, shift in regions:
        r = L[L.Region == region]
        b = r.groupby("bin_top").y.agg(["mean", "size"])
        ax.plot(b["mean"], b.index + bin_m / 2, color=colour, linewidth=2, marker="o", markersize=5,
                label=f"{region.replace('_', ' ')} ({len(r)} cells)")
        c = r.groupby("depth_top_m").size()
        axn.barh(c.index + half + shift, c.values, height=0.12, color=colour)
    ax.axvline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(cfg["depth"]["bottom_m"], 0)
    ax.set_xlabel("Share of labelled cells that are Clay", color=INK, fontsize=10)
    ax.set_ylabel("Depth below ground (m)", color=INK, fontsize=10)
    ax.set_title(f"Clay share per {bin_m:g} m depth bin", color=INK, fontsize=10, loc="left")
    axn.set_xlabel("Labelled 0.5 m cells", color=INK, fontsize=10)
    axn.set_title("Labelled cells per 0.5 m depth", color=INK, fontsize=10, loc="left")
    for a in (ax, axn):
        style_axes(a)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Audit preview of the 0.5 m discretisation", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(AGG / "fig_clay_share_by_depth.png")
    plt.close(fig)


def fig_fines(df: pd.DataFrame, cfg: dict) -> None:
    lab = df.dropna(subset=["Silt_Clay_pct"])
    bins = np.arange(0, 101, 2.5)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    ax.hist(lab[lab.USCS.isin(cfg["labels"]["clay"])].Silt_Clay_pct, bins=bins, color=C_CLAY,
            alpha=0.85, label="Logged CL/CH", edgecolor="white", linewidth=0.6)
    ax.hist(lab[lab.USCS.isin(cfg["labels"]["sand"])].Silt_Clay_pct, bins=bins, color=C_SAND,
            alpha=0.85, label="Logged SC/SM/SP", edgecolor="white", linewidth=0.6)
    ax.axvline(50, color=INK, linewidth=1, linestyle="--")
    ax.text(49, ax.get_ylim()[1] * 0.92, "50% fines (USCS boundary)", fontsize=8, color=INK_2, ha="right")
    ax.set_xlabel("Fines content, % passing 0.075 mm", color=INK, fontsize=10)
    ax.set_ylabel("Lab-tested samples", color=INK, fontsize=10)
    ax.set_title("Logged class vs lab fines content", color=INK, fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(AGG / "fig_fines_vs_logged_class.png")
    plt.close(fig)


def fig_nn(bh: pd.DataFrame, cfg: dict) -> None:
    nn = np.sort(bh[bh.Region == cfg["regions"]["downtown"]].dist_nearest_other_downtown_bh_m.values)
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=200)
    ax.step(nn, np.arange(1, len(nn) + 1) / len(nn), where="post", color=C_CLAY, linewidth=2)
    for s in (100, 200, 300):
        ax.axvline(s, color=INK_2, linewidth=0.8, linestyle=":")
        ax.text(s * 1.03, 0.04, f"{s} m", fontsize=8, color=INK_2, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Distance to nearest other Downtown borehole (m, log scale)", color=INK, fontsize=10)
    ax.set_ylabel("Cumulative share of boreholes", color=INK, fontsize=10)
    ax.set_title(f"Downtown nearest-neighbour distances (n = {len(nn)})", color=INK, fontsize=11, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(AGG / "fig_downtown_nn_distance_ecdf.png")
    plt.close(fig)


def fig_map(bh: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=200)
    for ax, sub, title in ((axes[0], bh, "All boreholes"), (axes[1], bh[bh.Region.isin(["Downtown", "Downtown_5km"])], "Downtown and 5 km region")):
        for region, colour in C_REGION.items():
            s = sub[sub.Region == region]
            if len(s):
                ax.scatter(s.E, s.N, s=14, color=colour, edgecolor="white", linewidth=0.5, label=f"{region} ({len(s)})")
        ax.set_aspect("equal")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Easting (m, EPSG:32648)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        style_axes(ax)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "fig_borehole_map.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "config" / "project.toml"))
    ap.add_argument("--allow-checksum-mismatch", action="store_true")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    AGG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    df = load(cfg, args.allow_checksum_mismatch)
    bh = borehole_inventory(df, cfg)
    record_checks(df, cfg)
    cells = discretise_preview(df, cfg)
    spatial = spatial_diagnostics(bh, cfg)

    # ---- aggregate outputs (no identifiers) ----
    field_completeness(df).to_csv(AGG / "field_completeness.csv", index=False)
    pd.crosstab(df.USCS.fillna("(missing)"), df.Region, margins=True).to_csv(AGG / "uscs_by_region_all_depths.csv")
    top = df[df.Depth_From_m < cfg["depth"]["bottom_m"]]
    pd.crosstab(top.USCS.fillna("(missing)"), top.Region, margins=True).to_csv(AGG / "uscs_by_region_0_15m.csv")
    pd.crosstab(df.Sample_Type.fillna("(missing)"), df.Region, margins=True).to_csv(AGG / "sample_type_by_region.csv")
    lab = df.dropna(subset=["Silt_Clay_pct", "USCS"])
    pd.crosstab(lab.USCS, pd.cut(lab.Silt_Clay_pct, [0, 30, 40, 45, 50, 55, 60, 70, 100.01], right=False)).to_csv(
        AGG / "uscs_vs_lab_fines_bins.csv")

    L = cells[cells.labelled].copy()
    L["depth_band"] = pd.cut(L.depth_top_m, [0, 5, 10, 15], right=False, labels=["0-5 m", "5-10 m", "10-15 m"])
    L.groupby(["Region", "depth_band"], observed=True).y.agg(cells="size", clay_share="mean").round(3).to_csv(
        AGG / "preview_cells_by_region_depthband.csv")
    L.groupby(["Region", "depth_top_m"]).y.agg(cells="size", clay_share="mean").round(3).to_csv(
        AGG / "preview_cells_by_region_depth.csv")

    issues = pd.DataFrame(ISSUES)
    issue_summary = (issues.groupby(["issue", "severity"]).agg(records=("issue", "size"), boreholes=("BH_Code", "nunique"))
                     .reset_index())
    issue_summary.to_csv(AGG / "issue_summary.csv", index=False)

    per_bh = L.groupby("BH_Code").agg(Region=("Region", "first"), n=("y", "size"), clay=("y", "mean"))
    summary = {
        "input_sha256": cfg["data"]["raw_csv_sha256"],
        "records": int(len(df)),
        "fields": int(df.shape[1] - 1),
        "boreholes": int(bh.shape[0]),
        "projects": int(df.Project_No.nunique()),
        "boreholes_by_region": bh.Region.value_counts().to_dict(),
        "projects_by_region": bh.groupby("Region").Project_No.nunique().to_dict(),
        "ground_elevation_m_by_region": bh.groupby("Region").Ground_Elevation_m.describe()[["min", "50%", "max"]].round(2).to_dict("index"),
        "borehole_max_depth_m_by_region": bh.groupby("Region").depth_max_m.describe()[["min", "50%", "max"]].round(2).to_dict("index"),
        "boreholes_shallower_than_15m": int((bh.depth_max_m < cfg["depth"]["bottom_m"]).sum()),
        "elevation_values_with_more_than_2dp": int(bh.elev_more_than_2dp.sum()),
        "coordinate_text_formats": bh.coord_text_format.value_counts().to_dict(),
        "coordinate_text_vs_utm_over_5m": int((bh.coord_text_vs_utm_m > 5).sum()),
        "lab_tested_with_uscs": int(len(lab)),
        "lab_uscs_group_inconsistent": int(((issues.issue == "I08")).sum()),
        "lab_fines_within_45_55pct_share": round(float(((lab.Silt_Clay_pct >= 45) & (lab.Silt_Clay_pct < 55)).mean()), 3),
        "preview_rule": {"cell_m": cfg["depth"]["cell_m"], "min_cell_coverage": cfg["depth"]["min_cell_coverage"],
                         "max_interval_m": cfg["depth"]["max_interval_m"]},
        "preview_labelled_cells_by_region": L.Region.value_counts().to_dict(),
        "preview_clay_share_by_region": L.groupby("Region").y.mean().round(3).to_dict(),
        "preview_conflicting_cells": int(cells.conflict.sum()),
        "preview_boreholes_with_labels_by_region": L.groupby("Region").BH_Code.nunique().to_dict(),
        "preview_downtown_cells_per_bh": per_bh[per_bh.Region == "Downtown"].n.describe()[["min", "50%", "max"]].to_dict(),
        "preview_downtown_bh_all_clay": int(((per_bh.Region == "Downtown") & (per_bh.clay == 1)).sum()),
        "preview_downtown_bh_all_sand": int(((per_bh.Region == "Downtown") & (per_bh.clay == 0)).sum()),
        "spatial": spatial,
    }
    (AGG / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    fig_depth_profile(cells, cfg)
    fig_fines(df, cfg)
    fig_nn(bh, cfg)

    # ---- restricted outputs (borehole identifiers / coordinates) ----
    bh.drop(columns=["Coordinate"]).to_csv(RES / "borehole_inventory.csv")
    issues.sort_values(["severity", "issue", "BH_Code", "csv_line"]).to_csv(RES / "issue_log.csv", index=False)
    cells.to_csv(RES / "preview_cells.csv", index=False)
    fig_map(bh)

    print(json.dumps({k: summary[k] for k in ("records", "boreholes", "boreholes_by_region",
                                              "preview_labelled_cells_by_region", "preview_clay_share_by_region")}, indent=1))
    print(issue_summary.to_string(index=False))


if __name__ == "__main__":
    main()
