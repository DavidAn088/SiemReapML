"""Polygons, DEM sampling and prediction-grid geometry (EPSG:32648 throughout)."""

from __future__ import annotations

import numpy as np


def load_polygon(cfg: dict, key: str):
    """Union of the shapefile's features as a 2D shapely geometry ('downtown_shp' or 'ring_shp')."""
    import geopandas as gpd
    import shapely

    g = gpd.read_file(cfg["_root"] / cfg["sources"][key])
    if g.crs is None or g.crs.to_epsg() != cfg["data"]["crs_epsg"]:
        raise ValueError(f"{key} must be in EPSG:{cfg['data']['crs_epsg']}")
    return shapely.force_2d(g.geometry.union_all())


class DEM:
    """The supplied DEM, sampled exactly as the CSV ground elevations were produced:
    bilinear interpolation between the four surrounding cell centres."""

    def __init__(self, cfg: dict):
        import rasterio

        with rasterio.open(cfg["_root"] / cfg["sources"]["dem"]) as r:
            z = r.read(1).astype("float64")
            nod = r.nodata
            self.left, self.top, self.res = r.bounds.left, r.bounds.top, r.res[0]
            self.epsg = r.crs.to_epsg()
        z[z == nod] = np.nan
        self.z = z

    def bilinear(self, x, y) -> np.ndarray:
        x, y = np.asarray(x, float), np.asarray(y, float)
        fx = (x - self.left) / self.res - 0.5
        fy = (self.top - y) / self.res - 0.5
        j0, i0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        wx, wy = fx - j0, fy - i0
        ok = (i0 >= 0) & (j0 >= 0) & (i0 + 1 < self.z.shape[0]) & (j0 + 1 < self.z.shape[1])
        out = np.full(x.shape, np.nan)
        i, j, a, b = i0[ok], j0[ok], wx[ok], wy[ok]
        z = self.z
        out[ok] = (1 - b) * ((1 - a) * z[i, j] + a * z[i, j + 1]) + b * ((1 - a) * z[i + 1, j] + a * z[i + 1, j + 1])
        return out


def grid_columns(polygon, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Centres of a regular grid (k * spacing + spacing / 2) that fall inside the polygon."""
    import shapely

    minx, miny, maxx, maxy = polygon.bounds
    gx = np.arange(np.floor(minx / spacing) * spacing + spacing / 2, maxx, spacing)
    gy = np.arange(np.floor(miny / spacing) * spacing + spacing / 2, maxy, spacing)
    X, Y = np.meshgrid(gx, gy)
    inside = shapely.contains_xy(polygon, X.ravel(), Y.ravel())
    return X.ravel()[inside], Y.ravel()[inside]


def nearest_distance(from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
    """Horizontal distance from each `from` point to the nearest `to` point."""
    from scipy.spatial import cKDTree

    return cKDTree(np.asarray(to_xy)).query(np.asarray(from_xy))[0]
