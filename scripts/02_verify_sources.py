"""Stage 2b: verify the CSV against the raw workbook, the boundary polygons and the DEM.

Re-derives, from the received files only, how the CSV used in Stage 2 relates to its
sources:

* workbook master sheet vs CSV (row alignment, per-column differences, formulas,
  text qualifiers lost on export, the workbook's own Region scheme);
* per-borehole digitised sheets vs the master rows they were appended as;
* Region labels vs point-in-polygon membership in downtown.shp / Ring_5km.shp;
* Ground_Elevation_m vs bilinear and nearest-cell samples of the DEM;
* the prediction domain: distance from a 25 m grid of columns in the Downtown polygon to
  the nearest Downtown borehole (geometry only - recomputed with development boreholes
  once the external holdout is fixed);
* inventories that bear on the labelling rule (long records, SS labels without lab data).

Like 01_audit_raw_data.py it writes aggregate outputs (committable) and restricted
borehole-level outputs (git-ignored). Nothing here uses class labels to build folds,
variograms or models.

Usage:
    python scripts/02_verify_sources.py [--config config/project.toml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import rasterio
import shapely
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "02_source_verification"
AGG = OUT / "aggregate"
RES = OUT / "restricted"

C_PRED = "#2a78d6"
C_BH = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# Region vocabulary of the CSV -> polygon class it should correspond to.
CSV_REGION_TO_CLASS = {"Downtown": "downtown", "Downtown_5km": "ring_5km", "Downtown_10km": "outside", "": "outside"}
NUMERIC_TOL = 1e-6
DIFF_COUNTS = ["numeric_diff_gt_1e-6", "string_diff", "blank_in_workbook_filled_in_csv", "text_in_workbook_blank_in_csv"]


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


def check_checksums(cfg: dict, allow: bool) -> pd.DataFrame:
    src = cfg["sources"]
    rows = [
        ("csv", ROOT / cfg["data"]["raw_csv"], cfg["data"]["raw_csv_sha256"]),
        ("workbook", ROOT / src["workbook"], src["workbook_sha256"]),
        ("downtown_shp", ROOT / src["downtown_shp"], src["downtown_shp_sha256"]),
        ("ring_shp", ROOT / src["ring_shp"], src["ring_shp_sha256"]),
        ("dem_w001001", ROOT / src["dem"] / "w001001.adf", src["dem_data_sha256"]),
    ]
    out = []
    for name, path, expected in rows:
        got = sha256(path)
        out.append({"file": name, "sha256": got, "matches_config": got == expected})
        if got != expected and not allow:
            sys.exit(f"Checksum mismatch for {path}: {got}")
    return pd.DataFrame(out)


# ---------------------------------------------------------------- workbook ----

def read_master(cfg: dict):
    """Return (cached-value DataFrame, formula-cell counts per column, cached-empty counts)."""
    path = ROOT / cfg["sources"]["workbook"]
    sheet = cfg["sources"]["master_sheet"]
    wb_f = openpyxl.load_workbook(path, data_only=False)
    wb_v = openpyxl.load_workbook(path, data_only=True)
    ws_f, ws_v = wb_f[sheet], wb_v[sheet]
    header = [c.value for c in ws_v[1]]
    vals = [list(r) for r in ws_v.iter_rows(min_row=2, values_only=True)]
    df = pd.DataFrame(vals, columns=header)
    formula_rows: dict[str, list[int]] = {}
    for row in ws_f.iter_rows(min_row=2):
        for c in row:
            if c.data_type == "f" or (isinstance(c.value, str) and c.value.startswith("=")):
                formula_rows.setdefault(header[c.column - 1], []).append(c.row)
    formula = {col: {"cells": len(r), "first_row": min(r), "last_row": max(r),
                     "cached_empty": int(df.loc[[x - 2 for x in r], col].isna().sum())}
               for col, r in formula_rows.items()}
    return df, formula, wb_v


def as_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except ValueError:
        return None  # text that is not a number


def compare_master_csv(wb: pd.DataFrame, csv: pd.DataFrame):
    if len(wb) != len(csv):
        sys.exit(f"Row count differs: workbook {len(wb)} vs CSV {len(csv)}")
    key = ["BH_Code", "Depth_From_m", "Depth_To_m"]
    aligned = bool((wb[key[0]].astype(str).values == csv[key[0]].astype(str).values).all()
                   and np.allclose(wb[key[1]].astype(float), csv[key[1]].astype(float))
                   and np.allclose(wb[key[2]].astype(float), csv[key[2]].astype(float)))
    if not aligned:
        sys.exit("Workbook master and CSV are not row-aligned on BH_Code/Depth_From_m/Depth_To_m")
    rows, text_lost = [], []
    for col in csv.columns:
        if col in ("csv_line", "Region_class"):
            continue
        a, b = wb[col], csv[col]
        n_text_lost = n_num = n_str = n_blank_wb = 0
        max_abs = 0.0
        for i, (x, y) in enumerate(zip(a, b)):
            xf, yf = as_float(x), as_float(y)
            x_blank = x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == "")
            y_blank = y is None or (isinstance(y, float) and np.isnan(y)) or (isinstance(y, str) and y.strip() == "")
            if x_blank and y_blank:
                continue
            if xf is None and y_blank:          # text in workbook, blank in CSV
                n_text_lost += 1
                text_lost.append({"csv_line": i + 2, "BH_Code": csv.BH_Code.iat[i], "column": col, "workbook_value": x})
            elif x_blank and not y_blank:
                n_blank_wb += 1
            elif xf is not None and yf is not None and not (np.isnan(xf) or np.isnan(yf)):
                d = abs(xf - yf)
                max_abs = max(max_abs, d)
                n_num += d > NUMERIC_TOL
            elif str(x).strip() != str(y).strip():
                n_str += 1
        rows.append({"column": col, "numeric_diff_gt_1e-6": n_num, "max_abs_numeric_diff": max_abs,
                     "string_diff": n_str, "blank_in_workbook_filled_in_csv": n_blank_wb,
                     "text_in_workbook_blank_in_csv": n_text_lost})
    return aligned, pd.DataFrame(rows), pd.DataFrame(text_lost)


def _norm(v) -> str:
    """Normalise a cell for identity comparison (numbers to 4 dp, text stripped)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    return str(v).strip()


def digitised_sheets(wb_v, master: pd.DataFrame) -> pd.DataFrame:
    """Compare each per-borehole sheet (data from row 5; A=Sample_No, B=type, C/D=From/To,
    Q=fines, T=USCS) with the master borehole it was appended as.

    The borehole is identified from content, not from the sheet name: the master borehole
    with the same number of rows whose (Sample_No, Sample_Type, fines, USCS) sequence matches
    best. Depths are left out of the match on purpose, because they are what is being checked."""
    groups = {code: g for code, g in master.groupby("BH_Code", sort=False)}
    out = []
    for ws in wb_v.worksheets:
        if not ws.title.startswith("SR-"):
            continue
        rows = [r for r in ws.iter_rows(min_row=5, values_only=True) if r[0] is not None or r[2] is not None]
        sig = [(_norm(r[0]), _norm(r[1]), _norm(r[16]), _norm(r[19])) for r in rows]
        scores = []
        for code, g in groups.items():
            if len(g) != len(sig):
                continue
            msig = list(zip(g.Sample_No.map(_norm), g.Sample_Type.map(_norm), g.Silt_Clay_pct.map(_norm), g.USCS.map(_norm)))
            scores.append((float(np.mean([a == b for a, b in zip(sig, msig)])), code))
        scores.sort(reverse=True)
        best, code = scores[0] if scores else (0.0, None)
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        rec = {"sheet": ws.title, "BH_Code": code, "match_score": round(best, 3), "runner_up_score": round(runner_up, 3),
               "mapped": bool(best >= 0.95 and runner_up < 0.8), "sheet_rows": len(rows)}
        if rec["mapped"]:
            m = groups[code]
            sf = np.array([as_float(r[2]) for r in rows]), np.array([as_float(r[3]) for r in rows])
            mf, mt = m.Depth_From_m.astype(float).values, m.Depth_To_m.astype(float).values
            rec["depth_mismatch_rows"] = int(((np.abs(sf[0] - mf) > NUMERIC_TOL) | (np.abs(sf[1] - mt) > NUMERIC_TOL)).sum())
            rec["uscs_mismatch_rows"] = int(sum(_norm(r[19]) != _norm(u) for r, u in zip(rows, m.USCS)))
            rec["sheet_overlapping_intervals"] = int((sf[0][1:] < sf[1][:-1] - 1e-9).sum())
            rec["master_overlapping_intervals"] = int((mf[1:] < mt[:-1] - 1e-9).sum())
        out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------ geometry/DEM ----

def polygon_checks(cfg: dict, bh: pd.DataFrame) -> tuple[dict, pd.DataFrame, object]:
    down = gpd.read_file(ROOT / cfg["sources"]["downtown_shp"])
    ring = gpd.read_file(ROOT / cfg["sources"]["ring_shp"])
    epsg = cfg["data"]["crs_epsg"]
    assert down.crs.to_epsg() == epsg and ring.crs.to_epsg() == epsg, "polygon CRS differs from config"
    D = shapely.force_2d(down.geometry.union_all())
    R = shapely.force_2d(ring.geometry.union_all())
    hole = shapely.Polygon(R.interiors[0]) if R.geom_type == "Polygon" and len(R.interiors) == 1 else None
    expected_ring = D.buffer(5000, quad_segs=64).difference(D)
    x, y = bh.E.values, bh.N.values
    in_d = shapely.contains_xy(D, x, y)
    in_r = shapely.contains_xy(R, x, y)
    cls = np.where(in_d, "downtown", np.where(in_r, "ring_5km", "outside"))
    boundary = D.boundary
    signed = np.array([(-1 if a else 1) * boundary.distance(shapely.Point(px, py)) for a, px, py in zip(in_d, x, y)])
    bh = bh.assign(polygon_class=cls, signed_dist_to_downtown_m=signed.round(1))
    expected = bh.Region_csv.map(lambda r: CSV_REGION_TO_CLASS.get(r, "unknown"))
    checks = {
        "downtown_area_km2": round(D.area / 1e6, 3),
        "downtown_is_valid": bool(D.is_valid),
        "ring_area_km2": round(R.area / 1e6, 3),
        "ring_interior_rings": len(R.interiors) if R.geom_type == "Polygon" else None,
        "ring_hole_vs_downtown_symdiff_m2": round(hole.symmetric_difference(D).area, 3) if hole else None,
        "ring_vs_buffer5000_minus_downtown_symdiff_m2": round(R.symmetric_difference(expected_ring).area, 1),
        "ring_vs_buffer5000_symdiff_share_pct": round(100 * R.symmetric_difference(expected_ring).area / R.area, 4),
        "boreholes_by_polygon_class": pd.Series(cls).value_counts().to_dict(),
        "csv_region_agrees_with_polygons": int((expected.values == cls).sum()),
        "csv_region_disagrees_with_polygons": int((expected.values != cls).sum()),
        "boreholes_within_50m_of_downtown_boundary": int((np.abs(signed) < 50).sum()),
        "nearest_outside_borehole_to_downtown_m": round(float(signed[signed > 0].min()), 1),
        "min_inside_distance_to_downtown_boundary_m": round(float(-signed[signed < 0].max()), 1),
    }
    return checks, bh, D


def sample_dem(cfg: dict, x: np.ndarray, y: np.ndarray):
    """Bilinear (cell-centre) and nearest-cell samples; NaN where any needed cell is nodata."""
    with rasterio.open(ROOT / cfg["sources"]["dem"]) as r:
        z = r.read(1).astype("float64")
        nod = r.nodata
        left, top, res = r.bounds.left, r.bounds.top, r.res[0]
        meta = {"driver": r.driver, "crs_epsg": r.crs.to_epsg(), "pixel_m": r.res[0], "width": r.width,
                "height": r.height, "bounds": [round(v, 3) for v in r.bounds], "dtype": r.dtypes[0]}
    z[z == nod] = np.nan
    meta["valid_share"] = round(float(np.isfinite(z).mean()), 4)
    fx = (x - left) / res - 0.5
    fy = (top - y) / res - 0.5
    j0, i0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    wx, wy = fx - j0, fy - i0
    ok = (i0 >= 0) & (j0 >= 0) & (i0 + 1 < z.shape[0]) & (j0 + 1 < z.shape[1])
    bil = np.full(len(x), np.nan)
    ii, jj = i0[ok], j0[ok]
    bil[ok] = ((1 - wy[ok]) * ((1 - wx[ok]) * z[ii, jj] + wx[ok] * z[ii, jj + 1])
               + wy[ok] * ((1 - wx[ok]) * z[ii + 1, jj] + wx[ok] * z[ii + 1, jj + 1]))
    jn, inn = np.floor((x - left) / res).astype(int), np.floor((top - y) / res).astype(int)
    okn = (inn >= 0) & (jn >= 0) & (inn < z.shape[0]) & (jn < z.shape[1])
    near = np.full(len(x), np.nan)
    near[okn] = z[inn[okn], jn[okn]]
    return bil, near, z, (left, top, res), meta


def dem_checks(cfg: dict, bh: pd.DataFrame, D) -> tuple[dict, pd.DataFrame]:
    bil, near, z, (left, top, res), meta = sample_dem(cfg, bh.E.values, bh.N.values)
    bh = bh.assign(dem_bilinear=bil, dem_nearest=near)
    cov = np.isfinite(bil)
    d_bil = np.abs(bh.Ground_csv.values - bil)
    d_near = np.abs(bh.Ground_csv.values - near)

    def wb_method(r):
        if pd.isna(r.Ground_wb):
            return "blank"
        if not np.isfinite(r.dem_bilinear):
            return "outside_dem"
        if abs(r.Ground_wb - r.dem_bilinear) <= 5e-5:
            return "bilinear"
        if abs(r.Ground_wb - r.dem_nearest) <= 1e-4:
            return "nearest_cell"
        return "other"

    bh["ground_wb_method"] = bh.apply(wb_method, axis=1)
    bh["ground_wb_minus_csv_m"] = (bh.Ground_wb - bh.Ground_csv).round(4)
    rows, cols = np.indices(z.shape)
    cx, cy = left + (cols + 0.5) * res, top - (rows + 0.5) * res
    in_d = shapely.contains_xy(D, cx.ravel(), cy.ravel()).reshape(z.shape)
    zd = z[in_d]
    dt = bh.polygon_class == "downtown"
    meta.update({
        "boreholes_inside_dem": int(cov.sum()),
        "boreholes_outside_dem": int((~cov).sum()),
        "boreholes_outside_dem_by_polygon_class": bh.loc[~cov, "polygon_class"].value_counts().to_dict(),
        "csv_ground_vs_bilinear_max_abs_m": float(np.nanmax(d_bil[cov])),
        "csv_ground_reproduced_by_bilinear_within_1e-4": int((d_bil[cov] <= 1e-4).sum()),
        "csv_ground_reproduced_by_nearest_within_1e-4": int((d_near[cov] <= 1e-4).sum()),
        "downtown_nearest_vs_bilinear_abs_m": {q: round(float(np.percentile(np.abs(near[dt] - bil[dt]), p)), 3)
                                               for q, p in (("median", 50), ("p90", 90), ("max", 100))},
        "downtown_boreholes_nearest_vs_bilinear_gt_0.5m": int((np.abs(near[dt] - bil[dt]) > 0.5).sum()),
        "workbook_ground_method_by_borehole": bh.ground_wb_method.value_counts().to_dict(),
        "workbook_minus_csv_ground_max_abs_m": round(float(np.nanmax(np.abs(bh.ground_wb_minus_csv_m))), 3),
        "downtown_dem_cells": int(in_d.sum()),
        "downtown_dem_nodata_cells": int(np.isnan(zd).sum()),
        "downtown_dem_elevation_m": {q: round(float(np.nanpercentile(zd, p)), 2)
                                     for q, p in (("min", 0), ("p1", 1), ("p50", 50), ("p99", 99), ("max", 100))},
        "downtown_borehole_ground_range_m": [round(float(bh.Ground_csv[dt].min()), 2), round(float(bh.Ground_csv[dt].max()), 2)],
        "dem_cells_exactly_1.5m": int((z == 1.5).sum()),
    })
    return meta, bh


def prediction_domain(bh: pd.DataFrame, D, spacing: float = 25.0) -> tuple[dict, np.ndarray, np.ndarray]:
    """Distances from a regular grid of XY columns inside the Downtown polygon (centres at
    k*spacing + spacing/2) to the nearest Downtown borehole."""
    minx, miny, maxx, maxy = D.bounds
    gx = np.arange(np.floor(minx / spacing) * spacing + spacing / 2, maxx, spacing)
    gy = np.arange(np.floor(miny / spacing) * spacing + spacing / 2, maxy, spacing)
    X, Y = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(D, X.ravel(), Y.ravel())
    pts = np.column_stack([X.ravel()[inside], Y.ravel()[inside]])
    dbh = bh[bh.polygon_class == "downtown"][["E", "N"]].values
    dist, _ = cKDTree(dbh).query(pts)
    nn, _ = cKDTree(dbh).query(dbh, k=2)
    hull = shapely.MultiPoint([tuple(p) for p in dbh]).convex_hull
    out = {
        "grid_spacing_m": spacing,
        "grid_columns_inside_downtown": int(len(pts)),
        "boreholes_used": int(len(dbh)),
        "note": "All Downtown boreholes; recompute with development boreholes after the external holdout is fixed.",
        "column_to_nearest_borehole_m": {f"p{p}": round(float(np.percentile(dist, p)), 1) for p in (5, 10, 25, 50, 75, 90, 95)}
                                        | {"max": round(float(dist.max()), 1)},
        "share_of_columns_within_m": {str(t): round(float((dist <= t).mean()), 4) for t in (50, 100, 200, 300, 500, 1000)},
        "borehole_nn_distance_m": {f"p{p}": round(float(np.percentile(nn[:, 1], p)), 1) for p in (10, 50, 90)},
        "borehole_convex_hull_share_of_downtown": round(hull.area / D.area, 3),
    }
    return out, dist, nn[:, 1]


def fig_distances(dist: np.ndarray, nn: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    for v, colour, label in ((dist, C_PRED, "Downtown 25 m grid column to nearest borehole"),
                             (nn, C_BH, "Borehole to nearest other borehole")):
        s = np.sort(v)
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=colour, linewidth=2, label=label)
    for b in (100, 200, 300):
        ax.axvline(b, color=INK_2, linewidth=0.8, linestyle=":")
        ax.text(b * 1.03, 0.04, f"{b} m", fontsize=8, color=INK_2, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Horizontal distance (m, log scale)", color=INK, fontsize=10)
    ax.set_ylabel("Cumulative share", color=INK, fontsize=10)
    ax.set_title("Prediction distances vs borehole spacing (Downtown)", color=INK, fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(AGG / "fig_prediction_vs_borehole_distances.png")
    plt.close(fig)


# ------------------------------------------------------------- inventories ----

def label_support(csv: pd.DataFrame, bh: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    lab = set(cfg["labels"]["clay"]) | set(cfg["labels"]["sand"])
    c = csv.merge(bh[["BH_Code", "polygon_class"]], on="BH_Code")
    c["length_m"] = c.Depth_To_m - c.Depth_From_m
    long = c[c.length_m > 1.0]
    long_bh = long.groupby("BH_Code").agg(polygon_class=("polygon_class", "first"), long_records=("csv_line", "size"),
                                          longest_m=("length_m", "max"), first_line=("csv_line", "min"))
    lab_cols = ["Silt_Clay_pct", "LL_pct", "PI_pct", "Water_Content_pct"]
    ss = c[(c.Sample_Type == "SS") & c.USCS.isin(lab)]
    ss_no_lab = ss[ss[lab_cols].isna().all(axis=1)]
    top = c[(c.Depth_From_m < cfg["depth"]["bottom_m"]) & (c.polygon_class == "downtown")]
    by_block = (top.assign(block=np.where(top.csv_line >= 3214, "appended_rows_3214_4270", "rows_2_3213"),
                           labelled=top.USCS.isin(lab))
                .groupby(["block", "Sample_Type"]).labelled.agg(["size", "sum"]).reset_index()
                .rename(columns={"size": "records", "sum": "binary_labelled"}))
    summary = {
        "records_longer_than_1m_by_polygon_class": long.polygon_class.value_counts().to_dict(),
        "boreholes_with_records_longer_than_1m_by_polygon_class": long_bh.polygon_class.value_counts().to_dict(),
        "ss_records_with_binary_label": int(len(ss)),
        "ss_records_with_binary_label_and_no_lab_data": int(len(ss_no_lab)),
        "downtown_0_15m_label_density_by_workbook_block_and_sample_type": by_block.to_dict("records"),
    }
    return summary, long_bh.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "config" / "project.toml"))
    ap.add_argument("--allow-checksum-mismatch", action="store_true")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    AGG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    checksums = check_checksums(cfg, args.allow_checksum_mismatch)
    csv = pd.read_csv(ROOT / cfg["data"]["raw_csv"], encoding="utf-8-sig", low_memory=False)
    csv.insert(0, "csv_line", np.arange(len(csv)) + 2)
    wb, formulas, wb_v = read_master(cfg)

    aligned, col_diff, text_lost = compare_master_csv(wb, csv)
    sheets = digitised_sheets(wb_v, wb)
    qc = pd.DataFrame(list(wb_v[cfg["sources"]["qc_sheet"]].iter_rows(values_only=True)))

    bh = (csv.assign(Region_csv=csv.Region.fillna(""), Region_wb=wb.Region.fillna("(blank)").values,
                     Ground_wb=pd.to_numeric(wb.Ground_Elevation_m, errors="coerce").values)
          .groupby("BH_Code", sort=True)
          .agg(E=("UTM_Easting_m", "first"), N=("UTM_Northing_m", "first"), Region_csv=("Region_csv", "first"),
               Region_wb=("Region_wb", "first"), Ground_csv=("Ground_Elevation_m", "first"),
               Ground_wb=("Ground_wb", "first"), first_line=("csv_line", "min"))
          .reset_index())
    poly, bh, D = polygon_checks(cfg, bh)
    dem, bh = dem_checks(cfg, bh, D)
    domain, dist, nn = prediction_domain(bh, D)
    labels, long_bh = label_support(csv, bh, cfg)

    region_xtab = pd.crosstab(bh.Region_wb, bh.Region_csv.replace("", "(blank)"), margins=True)
    summary = {
        "checksums_all_match": bool(checksums.matches_config.all()),
        "workbook_master_rows": int(len(wb)),
        "workbook_master_row_aligned_with_csv": aligned,
        "workbook_formula_columns": formulas,
        "columns_identical_to_csv": int((col_diff[DIFF_COUNTS] == 0).all(axis=1).sum()),
        "columns_compared": int(len(col_diff)),
        "columns_differing": col_diff.loc[(col_diff[DIFF_COUNTS] != 0).any(axis=1), "column"].tolist(),
        # Region is re-derived from polygons in the CSV, so its blanks are not lost text.
        "text_qualifiers_blanked_in_csv": int((text_lost.column != "Region").sum()),
        "digitised_sheets": int(len(sheets)),
        "digitised_sheets_mapped": int(sheets.mapped.sum()),
        "digitised_sheets_matching_master_depths": int((sheets.depth_mismatch_rows == 0).sum()),
        "digitised_sheets_with_depth_conflicts": int((sheets.depth_mismatch_rows > 0).sum()),
        "digitised_depth_conflict_rows": int(sheets.depth_mismatch_rows.sum()),
        "digitised_uscs_mismatch_rows": int(sheets.uscs_mismatch_rows.sum()),
        "qc_log_items": int(len(qc) - 1),
        "polygons": poly,
        "dem": dem,
        "prediction_domain": domain,
        "label_support": labels,
    }
    (AGG / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    checksums.to_csv(AGG / "source_checksums.csv", index=False)
    col_diff.to_csv(AGG / "workbook_vs_csv_column_differences.csv", index=False)
    region_xtab.to_csv(AGG / "region_workbook_vs_csv_boreholes.csv")
    fig_distances(dist, nn)

    bh.to_csv(RES / "borehole_source_checks.csv", index=False)
    sheets.to_csv(RES / "digitised_sheets_vs_master.csv", index=False)
    text_lost[text_lost.column != "Region"].to_csv(RES / "workbook_text_values_blanked_in_csv.csv", index=False)
    qc.to_csv(RES / "qc_digitalization_log.csv", index=False, header=False)
    long_bh.to_csv(RES / "boreholes_with_long_records.csv", index=False)

    print(json.dumps({k: summary[k] for k in summary if k not in ("polygons", "dem", "prediction_domain", "label_support",
                                                                  "workbook_formula_columns")}, indent=1))
    print(json.dumps({"polygons": poly, "prediction_domain": domain}, indent=1, default=str))


if __name__ == "__main__":
    main()
