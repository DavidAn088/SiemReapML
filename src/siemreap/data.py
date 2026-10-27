"""Raw data, borehole table, 0.5 m discretisation and the external holdout.

All rules follow the research brief (sections 4 and 10) and config/project.toml:

* binary target CL/CH -> Clay = 1, SC/SM/SP -> Sand = 0; every other symbol is excluded;
* 0.5 m cells between 0 and 15 m below ground; a cell is labelled from the sample intervals
  that overlap it; optionally (`fill_gaps`, design report section 10) an unsampled gap between
  two measured cells of the same class is filled for training, flagged measured = 0;
* predictors UTM_Easting_m, UTM_Northing_m and Top_Elevation_m = Ground_Elevation_m - cell
  top depth (the brief's Top = Ground - Depth_From applied to the cell);
* every cell keeps its borehole code and the CSV line numbers of the records it came from.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

from .config import sha256_file, seed_for

FEATURES = ["UTM_Easting_m", "UTM_Northing_m", "Top_Elevation_m"]
TARGET = "y"


# --------------------------------------------------------------------------- raw ----

def load_raw(cfg: dict, verify_checksum: bool = True) -> pd.DataFrame:
    p = cfg["_root"] / cfg["data"]["raw_csv"]
    if verify_checksum:
        got = sha256_file(p)
        if got != cfg["data"]["raw_csv_sha256"]:
            raise ValueError(f"Checksum of {p.name} is {got}, expected {cfg['data']['raw_csv_sha256']}")
    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    df.insert(0, "csv_line", np.arange(len(df)) + 2)  # header is line 1; one record per line
    df["Region"] = df["Region"].fillna("Unassigned")
    return df


def _parse_coordinate(s, transformer):
    if not isinstance(s, str):
        return np.nan, np.nan
    m = re.match(r"\s*UTM48N:\s*N\s*([\d.]+),\s*E\s*([\d.]+)", s)
    if m:
        return float(m.group(2)), float(m.group(1))
    lat, lon = (float(v) for v in s.split(","))
    return transformer.transform(lon, lat)


def borehole_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per borehole with location, region and sampling metadata (no class values)."""
    from pyproj import Transformer

    bottom = cfg["external"].get("eligibility_bottom_m", cfg["depth"]["bottom_m"])
    g = df.groupby("BH_Code", sort=True)
    bh = g.agg(Project_No=("Project_No", "first"), Region=("Region", "first"),
               E=("UTM_Easting_m", "first"), N=("UTM_Northing_m", "first"),
               Ground_Elevation_m=("Ground_Elevation_m", "first"), Coordinate=("Coordinate", "first"),
               depth_max_m=("Depth_To_m", "max"), n_records=("csv_line", "size"))
    top = df[df.Depth_From_m < bottom].assign(length=lambda d: d.Depth_To_m - d.Depth_From_m)
    bh["n_uscs_intervals_0_15"] = top[top.USCS.notna()].groupby("BH_Code").size().reindex(bh.index).fillna(0).astype(int)
    bh["max_record_length_0_15_m"] = top.groupby("BH_Code").length.max().reindex(bh.index)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{cfg['data']['crs_epsg']}", always_xy=True)
    txt = [_parse_coordinate(s, tr) for s in bh.Coordinate]
    bh["coordinate_text_mismatch_m"] = np.hypot([t[0] for t in txt] - bh.E, [t[1] for t in txt] - bh.N)
    return bh.drop(columns="Coordinate")


# ------------------------------------------------------------------ discretise ----

def discretise(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Label 0.5 m cells from the labelled sample intervals that overlap them.

    A cell [top, top + cell) receives class c when labelled intervals of class c cover at
    least `min_cell_coverage` of its thickness and no interval of the other class touches it.
    Intervals with a symbol outside the binary map, without a symbol, or longer than
    `max_interval_m` contribute nothing. Returns the cells and a summary of what was dropped."""
    dc, lab = cfg["depth"], cfg["labels"]
    clay, sand = set(lab["clay"]), set(lab["sand"])
    length = df.Depth_To_m - df.Depth_From_m
    labelled = df[df.USCS.isin(clay | sand) & (length <= dc["max_interval_m"])].copy()
    labelled["cls"] = labelled.USCS.isin(clay).astype(int)
    long_limit = cfg["external"]["max_record_length_m"]
    edges = np.arange(dc["top_m"], dc["bottom_m"] + 1e-9, dc["cell_m"])
    tops, bots = edges[:-1], edges[1:]
    rows, n_conflict, n_low_cov = [], 0, 0
    bh_meta = df.groupby("BH_Code").agg(E=("UTM_Easting_m", "first"), N=("UTM_Northing_m", "first"),
                                        G=("Ground_Elevation_m", "first"), Region=("Region", "first"))
    for code, b in labelled.groupby("BH_Code", sort=True):
        f, t = b.Depth_From_m.values, b.Depth_To_m.values
        cls, lines, stype = b.cls.values, b.csv_line.values, b.Sample_Type.fillna("?").astype(str).values
        ov = np.clip(np.minimum(t[None, :], bots[:, None]) - np.maximum(f[None, :], tops[:, None]), 0, None)
        m = bh_meta.loc[code]
        for j in np.where(ov.sum(axis=1) > 0)[0]:
            o = ov[j]
            cov_clay, cov_sand = o[cls == 1].sum(), o[cls == 0].sum()
            if cov_clay > 0 and cov_sand > 0:
                n_conflict += 1
                continue
            cov = max(cov_clay, cov_sand) / dc["cell_m"]
            if cov < dc["min_cell_coverage"] - 1e-9:
                n_low_cov += 1
                continue
            used = o > 0
            rows.append({
                "BH_Code": code, "cell": int(j), "depth_top_m": tops[j], "depth_mid_m": tops[j] + dc["cell_m"] / 2,
                "UTM_Easting_m": m.E, "UTM_Northing_m": m.N, "Ground_Elevation_m": m.G,
                "Top_Elevation_m": m.G - tops[j], "Region": m.Region, TARGET: int(cov_clay > 0),
                "coverage": round(cov, 3), "csv_lines": ";".join(map(str, lines[used])),
                "sample_types": ";".join(sorted(set(stype[used]))),
                "from_long_record": bool(((t - f)[used] > long_limit).any()),
            })
    cells = pd.DataFrame(rows)
    info = {"labelled_cells": int(len(cells)), "conflicting_cells_dropped": n_conflict,
            "low_coverage_cells_dropped": n_low_cov,
            "intervals_longer_than_max_ignored": int(((length > dc["max_interval_m"]) & df.USCS.isin(clay | sand)).sum()),
            "records_outside_binary_map": int((df.USCS.notna() & ~df.USCS.isin(clay | sand)).sum())}
    return cells, info


def fill_gaps(cells: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Fill unsampled cells between two measured cells of the same class (design revision of
    2026-10-13, design report section 10; a deviation from the brief, which forbids labels for
    unknown intervals).

    A gap is filled only when (a) the measured cells directly above and below it have the same
    class, (b) it is at most `fill_max_gap_m` thick, and (c) no record of any kind touches any
    of its cells (another class, a symbol outside the binary map, a missing symbol, an ignored
    long record). Nothing is filled above the first or below the last measured cell. Every
    cell gets `measured` (1 = from records, 0 = filled); filled cells are for training only and
    every evaluation uses measured cells."""
    dc = cfg["depth"]
    cells = cells.assign(measured=1, filled_between="")
    if not dc.get("fill_gaps", False):
        return cells, {"fill_gaps": False, "filled_cells": 0}
    cell_m = dc["cell_m"]
    max_cells = int(round(dc["fill_max_gap_m"] / cell_m))
    edges = np.arange(dc["top_m"], dc["bottom_m"] + 1e-9, cell_m)
    tops, bots = edges[:-1], edges[1:]
    touched = {}
    for code, b in df.groupby("BH_Code", sort=False):
        f, t = b.Depth_From_m.values, b.Depth_To_m.values
        ov = np.minimum(t[None, :], bots[:, None]) - np.maximum(f[None, :], tops[:, None]) > 1e-9
        touched[code] = set(np.where(ov.any(axis=1))[0].tolist())
    new, n = [], {"gaps": 0, "different_class": 0, "too_long": 0, "touched_by_a_record": 0, "filled_gaps": 0}
    for code, g in cells.sort_values(["BH_Code", "cell"]).groupby("BH_Code", sort=True):
        recs = g.to_dict("records")
        for a, b in zip(recs[:-1], recs[1:]):
            gap = range(a["cell"] + 1, b["cell"])
            if not len(gap):
                continue
            n["gaps"] += 1
            if a[TARGET] != b[TARGET]:
                n["different_class"] += 1
            elif len(gap) > max_cells:
                n["too_long"] += 1
            elif touched.get(code, set()) & set(gap):
                n["touched_by_a_record"] += 1
            else:
                n["filled_gaps"] += 1
                for j in gap:
                    new.append({**a, "cell": j, "depth_top_m": tops[j], "depth_mid_m": tops[j] + cell_m / 2,
                                "Top_Elevation_m": a["Ground_Elevation_m"] - tops[j], "coverage": 0.0,
                                "csv_lines": "", "sample_types": "fill", "from_long_record": False,
                                "measured": 0, "filled_between": f"{a['csv_lines']}/{b['csv_lines']}"})
    # concat only when something was filled: an empty frame would turn every column into object dtype
    out = pd.concat([cells, pd.DataFrame(new, columns=cells.columns)], ignore_index=True) if new else cells.copy()
    out = out.sort_values(["BH_Code", "cell"]).reset_index(drop=True)
    info = {"fill_gaps": True, "fill_max_gap_m": dc["fill_max_gap_m"], **n, "filled_cells": len(new),
            "measured_cells": int(len(cells))}
    return out, info


def fill_rule_check(cells: pd.DataFrame) -> dict:
    """Empirical check of the gap-filling assumption on measured cells: for each measured cell
    whose nearest measured cells from other records above and below have the same class, how
    often does the cell itself have that class too?"""
    m = measured(cells).sort_values(["BH_Code", "cell"])
    agree = n = 0
    for _, g in m.groupby("BH_Code"):
        ys = g.y.values
        lines = [set(str(v).split(";")) for v in g.csv_lines.values]
        for i in range(len(ys)):
            up = next((j for j in range(i - 1, -1, -1) if not lines[j] & lines[i]), None)
            dn = next((j for j in range(i + 1, len(ys)) if not lines[j] & lines[i]), None)
            if up is not None and dn is not None and ys[up] == ys[dn]:
                n += 1
                agree += int(ys[i] == ys[up])
    return {"cells_bracketed_by_same_class": n, "share_agreeing": round(agree / n, 4) if n else None}


def measured(cells: pd.DataFrame) -> pd.DataFrame:
    """The measured cells of a table (all of them when gap filling is off)."""
    return cells[cells.measured == 1] if "measured" in cells.columns else cells


# ------------------------------------------------- multiclass (USCS x SPT-N) ----

def fine_classes(cfg: dict) -> list[dict]:
    """The fine USCS x SPT-N classes of [target], in order: clay classes by increasing
    SPT-N, then sand classes. Each has a group, a name and an SPT-N interval (lo, hi]."""
    t, out = cfg["target"], []
    for group in ("clay", "sand"):
        names, upper = t[f"{group}_names"], list(t[f"{group}_n_upper"])
        if len(names) != len(upper) + 1:
            raise ValueError(f"[target] {group}_names needs one name more than {group}_n_upper")
        bounds = [-np.inf, *upper, np.inf]
        for i, name in enumerate(names):
            out.append({"group": group, "name": name, "lo": bounds[i], "hi": bounds[i + 1]})
    return out


def fine_class_of(uscs: pd.Series, spt: pd.Series, cfg: dict) -> pd.Series:
    """Index of the fine class for each record (NaN without a usable USCS symbol or SPT-N)."""
    t = cfg["target"]
    out = pd.Series(np.nan, index=uscs.index)
    for i, c in enumerate(fine_classes(cfg)):
        m = uscs.isin(t[c["group"]]) & (spt > c["lo"]) & (spt <= c["hi"])
        out[m] = i
    return out


def merge_classes(counts: pd.Series, cfg: dict) -> dict:
    """Merge rule fixed before modelling (counts only): within each group, while the smallest
    class has fewer than `min_class_records` records, merge it with its smaller neighbour in
    SPT-N order. Returns the fine-to-final mapping and the final class names."""
    fine = fine_classes(cfg)
    minimum = cfg["target"]["min_class_records"]
    final, mapping = [], {}
    for group in ("clay", "sand"):
        idx = [i for i, c in enumerate(fine) if c["group"] == group]
        runs = [[i] for i in idx]                      # consecutive fine classes per final class
        size = lambda r: int(sum(counts.get(i, 0) for i in r))
        while len(runs) > 1 and min(size(r) for r in runs) < minimum:
            j = min(range(len(runs)), key=lambda j: (size(runs[j]), j))
            nb = [n for n in (j - 1, j + 1) if 0 <= n < len(runs)]
            n = min(nb, key=lambda n: (size(runs[n]), n))
            a, b = sorted((j, n))
            runs[a:b + 1] = [runs[a] + runs[b]]
        for r in runs:
            first, last = fine[r[0]]["name"], fine[r[-1]]["name"]
            name = f"{first} {group}" if len(r) == 1 else f"{first} to {last[0].lower() + last[1:]} {group}"
            for i in r:
                mapping[i] = len(final)
            final.append({"name": name, "group": group, "fine": [fine[i]["name"] for i in r],
                          "spt_n_range": [None if not np.isfinite(v) else float(v) for v in (fine[r[0]]["lo"], fine[r[-1]]["hi"])],
                          "records": size(r)})
    return {"mapping": mapping, "classes": final}


def samples(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """One row per sampled interval with a USCS symbol of either group and an SPT-N value,
    starting above `bottom_m` and no longer than `max_interval_m`. Columns follow the binary
    cell table (BH_Code, cell, depth_top_m, the three predictors, measured = 1) so folds,
    distances and the nested CV treat both targets alike. `fine` holds the fine class; `y`
    is added by `apply_classes` once the merge rule has been applied."""
    dc = cfg["depth"]
    length = df.Depth_To_m - df.Depth_From_m
    fine = fine_class_of(df.USCS, pd.to_numeric(df.SPT_N, errors="coerce"), cfg)
    keep = fine.notna() & (df.Depth_From_m >= dc["top_m"]) & (df.Depth_From_m < dc["bottom_m"]) & (length <= dc["max_interval_m"])
    s = df[keep].copy()
    s["fine"] = fine[keep].astype(int)
    s = s.sort_values(["BH_Code", "Depth_From_m", "csv_line"])
    out = pd.DataFrame({
        "BH_Code": s.BH_Code.values, "cell": s.groupby("BH_Code").cumcount().values,
        "depth_top_m": s.Depth_From_m.values, "depth_mid_m": ((s.Depth_From_m + s.Depth_To_m) / 2).values,
        "UTM_Easting_m": s.UTM_Easting_m.values, "UTM_Northing_m": s.UTM_Northing_m.values,
        "Ground_Elevation_m": s.Ground_Elevation_m.values,
        "Top_Elevation_m": (s.Ground_Elevation_m - s.Depth_From_m).values, "Region": s.Region.values,
        "USCS": s.USCS.values, "SPT_N": pd.to_numeric(s.SPT_N, errors="coerce").values, "fine": s.fine.values,
        "y_bin": s.USCS.isin(cfg["target"]["clay"]).astype(int).values,
        "csv_lines": s.csv_line.astype(str).values, "sample_types": s.Sample_Type.fillna("?").astype(str).values,
        "measured": 1})
    t = cfg["target"]
    with_uscs = df.USCS.isin(t["clay"] + t["sand"]) & (df.Depth_From_m < dc["bottom_m"])
    info = {"samples": int(len(out)), "boreholes": int(out.BH_Code.nunique()),
            "records_with_uscs_in_map": int(with_uscs.sum()),
            "of_which_without_spt_n": int((with_uscs & pd.to_numeric(df.SPT_N, errors="coerce").isna()).sum()),
            "longer_than_max_interval": int((with_uscs & fine.notna() & (length > dc["max_interval_m"])).sum()),
            "same_borehole_same_depth_duplicates": int(out.duplicated(["BH_Code", "depth_top_m"]).sum())}
    return out, info


def apply_classes(cells: pd.DataFrame, merge: dict) -> pd.DataFrame:
    """Final class index `y` and `class_name` from the fine class and the merge mapping."""
    m = {int(k): int(v) for k, v in merge["mapping"].items()}
    y = cells.fine.map(m).astype(int)
    return cells.assign(y=y, class_name=[merge["classes"][i]["name"] for i in y])


def class_info(cfg: dict) -> dict:
    """The final classes written by the preprocessing notebook (processed_dir/classes.json)."""
    from .config import processed_dir

    p = processed_dir(cfg) / "classes.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run the preprocessing notebook first.")
    return json.loads(p.read_text())


def n_classes(cfg: dict) -> int:
    from .config import target_kind

    return 2 if target_kind(cfg) == "binary" else len(class_info(cfg)["classes"])


def clay_classes(cfg: dict) -> list[int]:
    """Class indices that belong to the Clay group (binary: [1])."""
    from .config import target_kind

    if target_kind(cfg) == "binary":
        return [1]
    return [i for i, c in enumerate(class_info(cfg)["classes"]) if c["group"] == "clay"]


# ------------------------------------------------------- second external test ----

def load_new_external(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """The second external borehole file ([new_external]), checked against its SHA-256.
    Boreholes already in the main CSV (same BH_Code) are removed, because they are training
    data; the remaining ones must lie inside the Downtown polygon. Returns the records (same
    columns as `load_raw`) and a summary that contains no class values."""
    import shapely

    from .geo import load_polygon

    ne = cfg["new_external"]
    p = cfg["_root"] / ne["raw_csv"]
    got = sha256_file(p)
    if got != ne["raw_csv_sha256"]:
        raise ValueError(f"Checksum of {p.name} is {got}, expected {ne['raw_csv_sha256']}")
    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    df.insert(0, "csv_line", np.arange(len(df)) + 2)
    main = load_raw(cfg)
    dup = sorted(set(df.BH_Code) & set(main.BH_Code))
    df = df[~df.BH_Code.isin(dup)].copy()
    poly = load_polygon(cfg, "downtown_shp")
    bh = df.groupby("BH_Code").agg(E=("UTM_Easting_m", "first"), N=("UTM_Northing_m", "first"))
    outside = sorted(bh.index[~shapely.contains_xy(poly, bh.E.values, bh.N.values)])
    if outside:
        raise ValueError(f"{len(outside)} new external borehole(s) lie outside the Downtown polygon: {outside}")
    df["Region"] = cfg["regions"]["downtown"]
    info = {"file_sha256": got, "boreholes_in_file": int(len(bh) + len(dup)),
            "already_in_main_csv_removed": len(dup), "new_boreholes": int(len(bh)), "records": int(len(df))}
    return df, info


# --------------------------------------------------------------- manual flags ----

def manual_flags(cfg: dict) -> pd.DataFrame:
    """Optional git-ignored CSV: BH_Code, exclude_from_training, exclude_from_external, reason."""
    p = cfg["_root"] / cfg["paths"]["manual_flags"]
    cols = ["BH_Code", "exclude_from_training", "exclude_from_external", "reason"]
    if not p.exists():
        return pd.DataFrame(columns=cols)
    f = pd.read_csv(p)
    for c in ("exclude_from_training", "exclude_from_external"):
        f[c] = f[c].astype(str).str.lower().isin(["1", "true", "yes"])
    return f[cols]


# ------------------------------------------------------------ external holdout ----

def _sites(xy: np.ndarray, linkage_m: float) -> np.ndarray:
    if len(xy) == 1:
        return np.array([1])
    return fcluster(linkage(xy, "single"), linkage_m, "distance")


def select_external(bh: pd.DataFrame, cfg: dict, flags: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Choose the external Downtown boreholes from coordinates and sampling metadata only.

    Generalisation group: whole sites (single-linkage clusters) whose nearest borehole outside
    the site is at least `generalisation_min_isolation_m` away. Interpolation group: one
    borehole from each of several multi-borehole sites; its site neighbours stay in
    development. Both groups are spread over k-means zones. Deterministic given the config."""
    from sklearn.cluster import KMeans

    ex = cfg["external"]
    dt = bh[bh.Region == cfg["regions"]["downtown"]].sort_index().copy()
    flags = flags if flags is not None else manual_flags(cfg)
    barred = set(flags.loc[flags.exclude_from_external | flags.exclude_from_training, "BH_Code"])
    dt["eligible"] = ((dt.depth_max_m >= ex["min_depth_m"])
                      & (dt.n_uscs_intervals_0_15 >= ex["min_uscs_intervals"])
                      & (dt.coordinate_text_mismatch_m <= ex["max_coordinate_mismatch_m"])  # NaN (no text) -> False
                      & (dt.max_record_length_0_15_m <= ex["max_record_length_m"])
                      & ~dt.index.isin(barred))
    xy = dt[["E", "N"]].values
    lab = _sites(xy, ex["site_linkage_m"])
    # canonical site key: the alphabetically first borehole code in the site
    key = pd.Series(dt.index, index=dt.index).groupby(lab).transform("min").values
    dt["site"] = key
    iso = {}
    for s in np.unique(key):
        inside = key == s
        iso[s] = cKDTree(xy[~inside]).query(xy[inside])[0].min() if (~inside).any() else np.inf
    dt["site_isolation_m"] = dt.site.map(iso)
    km = KMeans(n_clusters=ex["n_zones"], random_state=seed_for(cfg, "external", "zones"), n_init=20).fit(xy)
    dt["zone"] = km.labels_
    site_tab = (dt.groupby("site").agg(n=("eligible", "size"), all_eligible=("eligible", "all"),
                                       any_eligible=("eligible", "any"), isolation_m=("site_isolation_m", "first"),
                                       zone=("zone", lambda z: int(pd.Series(z).mode().min())))
                .sort_index())
    rng = np.random.default_rng(seed_for(cfg, "external", "draw"))

    def round_robin(cands: pd.DataFrame, n: int) -> list[str]:
        chosen, zones = [], list(rng.permutation(sorted(cands.zone.unique())))
        pools = {z: list(cands.index[cands.zone == z]) for z in zones}
        while len(chosen) < n and any(pools.values()):
            for z in zones:
                if pools[z] and len(chosen) < n:
                    chosen.append(pools[z].pop(int(rng.integers(len(pools[z])))))
        return chosen

    gen_c = site_tab[site_tab.all_eligible & (site_tab.isolation_m >= ex["generalisation_min_isolation_m"])]
    gen_sites = round_robin(gen_c, ex["n_generalisation_sites"])
    int_c = site_tab[(site_tab.n >= ex["interpolation_min_site_size"]) & site_tab.any_eligible
                     & ~site_tab.index.isin(gen_sites)]
    int_sites = round_robin(int_c, ex["n_interpolation_boreholes"])
    picks = []
    for s in gen_sites:
        for code in dt.index[dt.site == s]:
            picks.append((code, "generalisation", s))
    for s in int_sites:
        members = sorted(dt.index[(dt.site == s) & dt.eligible])
        picks.append((members[int(rng.integers(len(members)))], "interpolation", s))
    sel = pd.DataFrame(picks, columns=["BH_Code", "group", "site"]).set_index("BH_Code")
    sel["zone"] = dt.loc[sel.index, "zone"]
    sel["site_isolation_m"] = dt.loc[sel.index, "site_isolation_m"].round(1)
    info = {"downtown_boreholes": int(len(dt)), "eligible": int(dt.eligible.sum()),
            "sites": int(len(site_tab)), "generalisation_candidate_sites": int(len(gen_c)),
            "interpolation_candidate_sites": int(len(int_c)),
            "selected_generalisation_boreholes": int((sel.group == "generalisation").sum()),
            "selected_interpolation_boreholes": int((sel.group == "interpolation").sum())}
    if len(gen_sites) < ex["n_generalisation_sites"] or len(int_sites) < ex["n_interpolation_boreholes"]:
        info["warning"] = "fewer candidates than requested; see counts"
    return sel, info


def holdout_digest(codes) -> str:
    return hashlib.sha256("\n".join(sorted(codes)).encode("utf-8")).hexdigest()


def lock_external(cfg: dict, sel: pd.DataFrame, info: dict) -> dict:
    """Write the lock on first use; afterwards refuse any different selection.

    The lock (config/external_holdout.lock.json) holds only the SHA-256 of the sorted
    borehole codes and the selection settings, so it can be committed to a public repository."""
    lock_path = cfg["_root"] / "config" / "external_holdout.lock.json"
    digest = holdout_digest(sel.index)
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["sha256"] != digest:
            raise RuntimeError(
                "The external holdout recomputed from the current inputs differs from the locked "
                "selection. The external set must not change once fixed (brief section 10). If no "
                "model has been trained yet and the change is intended, delete the lock file deliberately.")
        return lock
    lock = {"sha256": digest, "n_boreholes": int(len(sel)),
            "groups": sel.group.value_counts().to_dict(), "selection_info": info,
            "rule": cfg["external"], "seed": cfg["seeds"]["global"], "locked_on": date.today().isoformat(),
            "note": "Selected from coordinates and sampling metadata only, before any modelling."}
    lock_path.write_text(json.dumps(lock, indent=2))
    return lock


# --------------------------------------------------------------- model table ----

def model_table(cells: pd.DataFrame, external: pd.Index, cfg: dict, flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `set` (development / external / excluded_region / excluded_manual) and scenario flags."""
    reg = cfg["regions"]
    flags = flags if flags is not None else manual_flags(cfg)
    no_train = set(flags.loc[flags.exclude_from_training, "BH_Code"])
    c = cells.copy()
    c["set"] = np.select(
        [c.BH_Code.isin(external), c.BH_Code.isin(no_train), c.Region.isin([reg["downtown"], reg["buffer_5km"]])],
        ["external", "excluded_manual", "development"], default="excluded_region")
    dev = c.set == "development"
    c["is_downtown"] = c.Region == reg["downtown"]
    c["in_A"] = dev & c.is_downtown
    c["in_B"] = dev & c.Region.isin([reg["downtown"], reg["buffer_5km"]])
    return c


def boreholes_of(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per borehole of a cell table (coordinates and flags)."""
    keep = [c for c in ["BH_Code", "UTM_Easting_m", "UTM_Northing_m", "Region", "set", "is_downtown", "in_A", "in_B"]
            if c in cells.columns]
    return cells[keep].drop_duplicates("BH_Code").set_index("BH_Code").sort_index()


def write_csv(df: pd.DataFrame, p: Path, **kw) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, **kw)
    return p
