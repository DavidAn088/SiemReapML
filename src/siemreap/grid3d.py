"""3D prediction grid and ArcGIS Pro exports (brief sections 2.7 and 14.8).

Columns sit on a regular XY grid inside the Downtown polygon. Each column has 30 cells of
0.5 m between 0 and 15 m below the DEM surface. Ground elevation is the bilinear DEM sample,
the same sampler that produced the training ground elevations, so the predictor
Top_Elevation_m = ground - cell top depth means the same thing in training and prediction.

Exports
* points CSV: X, Y, Z (cell-centre elevation) with attributes; ArcGIS Pro "XY Table To
  Point" with Z field = Z gives a 3D point layer.
* netCDF (CF-1.8) on a regular elevation axis for an ArcGIS Pro voxel layer; voxels above
  the ground or deeper than 15 m are NoData.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES
from .geo import DEM, grid_columns, load_polygon, nearest_distance
from .metrics import entropy


def columns(cfg: dict) -> pd.DataFrame:
    poly = load_polygon(cfg, "downtown_shp")
    x, y = grid_columns(poly, cfg["grid3d"]["spacing_m"])
    g = DEM(cfg).bilinear(x, y)
    c = pd.DataFrame({"X": x, "Y": y, "ground_m": g})
    if c.ground_m.isna().any():
        raise ValueError(f"{int(c.ground_m.isna().sum())} grid columns have no DEM value")
    return c


def cells(cols: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dc = cfg["depth"]
    tops = np.arange(dc["top_m"], dc["bottom_m"] - 1e-9, dc["cell_m"])
    n = len(tops)
    out = pd.DataFrame({
        "col": np.repeat(np.arange(len(cols)), n),
        "layer": np.tile(np.arange(n), len(cols)),
        "depth_top_m": np.tile(tops, len(cols)),
    })
    out["UTM_Easting_m"] = cols.X.values[out.col]
    out["UTM_Northing_m"] = cols.Y.values[out.col]
    out["ground_m"] = cols.ground_m.values[out.col]
    out["Top_Elevation_m"] = out.ground_m - out.depth_top_m
    out["Z"] = out.Top_Elevation_m - dc["cell_m"] / 2
    return out


def predict(model, threshold: float, grid: pd.DataFrame, chunk_rows: int) -> pd.DataFrame:
    p = np.empty(len(grid), dtype="float32")
    X = grid[FEATURES].values
    for s in range(0, len(grid), chunk_rows):
        p[s:s + chunk_rows] = model.predict_proba(X[s:s + chunk_rows])[:, 1]
    return grid.assign(p_clay=p, clay=(p >= threshold).astype("int8"), entropy=entropy(p).astype("float32"))


def add_flags(grid: pd.DataFrame, cols: pd.DataFrame, train_bh_xy: np.ndarray, validated_distance_m: float,
              top_elev_range: tuple[float, float]) -> pd.DataFrame:
    """Horizontal flag: column farther from the nearest training borehole than the distances
    evaluated in spatial CV. Vertical flag: cell-top elevation outside the training range."""
    d = nearest_distance(cols[["X", "Y"]].values, train_bh_xy)
    g = grid.assign(dist_nearest_bh_m=d[grid.col].astype("float32"))
    g["extrap_horizontal"] = (g.dist_nearest_bh_m > validated_distance_m).astype("int8")
    lo, hi = top_elev_range
    g["extrap_vertical"] = ((g.Top_Elevation_m < lo) | (g.Top_Elevation_m > hi)).astype("int8")
    return g


def to_points_csv(grid: pd.DataFrame, path) -> None:
    cols = ["X", "Y", "Z", "depth_top_m", "Top_Elevation_m", "ground_m", "p_clay", "clay", "entropy",
            "dist_nearest_bh_m", "extrap_horizontal", "extrap_vertical"]
    g = grid.rename(columns={"UTM_Easting_m": "X", "UTM_Northing_m": "Y"})
    g[cols].to_csv(path, index=False, float_format="%.3f")


def to_voxel_netcdf(grid: pd.DataFrame, cols: pd.DataFrame, cfg: dict, path) -> dict:
    import xarray as xr
    from pyproj import CRS

    dc, sp = cfg["depth"], cfg["grid3d"]["spacing_m"]
    h = dc["cell_m"]
    xs, ys = np.unique(cols.X.values), np.unique(cols.Y.values)
    ix = np.searchsorted(xs, cols.X.values)
    iy = np.searchsorted(ys, cols.Y.values)
    z0 = np.floor((cols.ground_m.min() - dc["bottom_m"]) / h) * h
    z1 = np.ceil(cols.ground_m.max() / h) * h
    zc = np.arange(z0 + h / 2, z1, h)
    shape = (len(zc), len(ys), len(xs))
    col, layer = grid.col.values, grid.layer.values
    ground = cols.ground_m.values[col]
    m = np.floor((ground - h * layer - z0 - h / 2) / h).astype(int)  # voxel whose centre lies in the cell
    ok = (m >= 0) & (m < len(zc))
    data = {}
    for name, fill, dtype in (("p_clay", np.nan, "float32"), ("entropy", np.nan, "float32"),
                              ("clay", -1, "int8"), ("extrap_horizontal", -1, "int8"), ("extrap_vertical", -1, "int8")):
        if name not in grid:
            continue
        a = np.full(shape, fill, dtype=dtype)
        a[m[ok], iy[col[ok]], ix[col[ok]]] = grid[name].values[ok]
        data[name] = (("z", "y", "x"), a)
    crs = CRS.from_epsg(cfg["data"]["crs_epsg"])
    ds = xr.Dataset(data, coords={"z": zc, "y": ys, "x": xs})
    ds["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": crs.to_wkt(), "spatial_ref": crs.to_wkt(),
                                               "epsg_code": f"EPSG:{crs.to_epsg()}"})
    for v in data:
        ds[v].attrs["grid_mapping"] = "spatial_ref"
    ds.p_clay.attrs.update(long_name="probability of Clay (CL/CH)", units="1")
    if "clay" in ds:
        ds.clay.attrs.update(long_name="predicted class (1 = Clay, 0 = Sand)")
    ds.x.attrs.update(standard_name="projection_x_coordinate", units="m", axis="X")
    ds.y.attrs.update(standard_name="projection_y_coordinate", units="m", axis="Y")
    ds.z.attrs.update(standard_name="height", long_name="elevation (DEM datum)", units="m", positive="up", axis="Z")
    ds.attrs.update(Conventions="CF-1.8", title="Siem Reap Downtown 3D Clay/Sand prediction",
                    source="siemreap notebooks 05_final_model_external_3d", grid_spacing_m=sp, layer_thickness_m=h)
    enc = {v: {"zlib": True, "complevel": 4, **({"_FillValue": -1} if data[v][1].dtype == np.int8 else {})}
           for v in data}
    ds.to_netcdf(path, engine="netcdf4", encoding=enc)
    return {"shape_zyx": shape, "z_range_m": [float(zc[0]), float(zc[-1])], "voxels_filled": int(ok.sum())}
