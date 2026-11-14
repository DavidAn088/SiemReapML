# V3-clean: summary for the 3D map (hand-over to a new chat)

Paste this file into the new chat. It describes the frozen V3 result (recomputed by V3-clean, proven
identical), what the map step does, how to use a grid made in ArcGIS Pro, and which grid size to use and why.

---

## 1. V3 in one page

| Item | V3 |
|---|---|
| Question | 3D Clay/Sand classification of Siem Reap Downtown, 0-15 m below ground |
| Target | Clay (CL, CH) = 1, Sand (SC, SM, SP) = 0; other symbols give no label |
| Cells | 0.5 m thick; a cell is labelled when samples of one class cover ≥ half of it; gaps ≤ 3 m between two cells of the same class are filled (V3 rule: any record in the gap, also an unclassified SS sample, stops the fill) |
| Boreholes | 120 in Downtown: 109 development (CV + final model), 11 external holdout (chosen before modelling) |
| Inputs | Easting, Northing, top elevation of the cell (X, Y, Z). Z = DEM ground elevation − depth |
| Validation | nested spatial CV: 5 outer folds from blockCV `cv_spatial`, 106 m square blocks (block size = median `cv_spatial_autocor` range of the depth slices); inner CV on the other folds; Optuna tunes hyperparameters, filled-cell weight and decision threshold |
| Models | RF, XGB, MLP, GWRF, GWXGB |
| Final model | **XGB** (documented override; the pre-declared 1-SE rule chose GWXGB). Reason: highest mean outer-fold Macro-F1 (0.644) and lowest Brier score, within one SE of GWXGB, simpler to apply and explain. Threshold **0.58**; 250 trees, learning rate 0.0238, max depth 3 |

**Main CV result (106 m blocks, all test cells):** XGB Macro-F1 **0.644 ± 0.095**, balanced accuracy 0.647,
ROC-AUC 0.701, Brier 0.217 (measured cells only: 0.636). RF 0.636, GWXGB 0.621, GWRF 0.608, MLP 0.567.

**Skill falls with distance from boreholes (XGB, Macro-F1):** 0-50 m 0.88; 50-100 m 0.78; 100-200 m 0.60;
beyond 200 m about 0.4-0.6 (few cells and boreholes; not a smooth trend).

**Larger blocks / purge (XGB):** 212 m 0.569; 318 m 0.533; purge 106 m 0.569; purge 212 m 0.577.

**External tests (XGB):** A, 11 holdout boreholes: 0.589. B, 10 new boreholes (measured cells): 0.580.
C, the same 10 with SS gaps filled: 0.511. XGB and GWXGB are tied externally (differences 0.00-0.02).

**Explainable AI:** location (mainly Northing) and elevation matter about equally; Easting little.

**Known weak point, found 2026-10-31 (state it in the thesis):** the CV score depends strongly on how the
boreholes fall into folds. Re-running RF with the same design but a new blockCV split (blockCV 4.0-2) gave
Macro-F1 about 0.55, with single folds from 0.46 to 0.69. V3's 0.644 is one split; a repeated-split
check (several blockCV seeds, mean ± SD) is the planned remedy.

---

## 2. What the map step does (V3-clean notebook 08)

1. **Map model:** the final model type (XGB), its hyperparameters and threshold 0.58 are **refitted on all
   120 Downtown boreholes** (109 development + 11 holdout). The map model is not evaluated; its expected
   skill is the CV and external result above.
2. **Grid columns:** points every **50 m** inside the Downtown polygon (32,280 columns). Ground elevation
   of each column = DEM, bilinear interpolation (the same way the borehole ground elevations were taken).
3. **Cells:** each column gets 30 cells of 0.5 m (0-15 m): `Top_Elevation_m = ground − depth_top`,
   `Z = Top_Elevation_m − 0.25` (cell centre). 968,400 cells in total.
4. **Prediction per cell:** `p_clay` (probability of Clay) and `clay` (1 if `p_clay ≥ 0.58`).
5. **Reliability per cell:** `dist_nearest_bh_m` (to the nearest of the 120 boreholes) and
   `reliability_zone`: 1 = ≤ 100 m, 2 = 100-300 m, 3 = > 300 m. `extrap_vertical` = 1 when the cell's
   elevation is outside the training elevations (0.4% of cells).
6. **Outputs** (in `outputs/08_map/private/`, never shared publicly):
   * `siemreap_3d_points_120bh_50m.csv`: X, Y, Z, depth_top_m, Top_Elevation_m, ground_m, p_clay, clay,
     dist_nearest_bh_m, reliability_zone, extrap_vertical (EPSG:32648, WGS 84 / UTM 48N);
   * `siemreap_3d_voxels_120bh_50m.nc`: netCDF (CF-1.8) voxel file for ArcGIS Pro.

**Map summary:** zone 1 = 3.1% of the map, zone 2 = 15.3%, zone 3 = **81.6%**. Predicted clay share 0.369.
**Use:** detailed only in zones 1-2; in zone 3 the map shows the general trend with depth and area.

---

## 3. Using your own grid from ArcGIS Pro

Only step 2 above changes: the grid columns come from your ArcGIS layer instead of from the code.
The model, threshold and the rest of notebook 08 stay the same.

**In ArcGIS Pro**
1. *Create Fishnet* (Data Management): template extent = Downtown polygon; cell size width = height = **50**;
   tick *Create Label Points*. Coordinate system **WGS 84 / UTM zone 48N (EPSG:32648)**.
2. Keep only the label points inside Downtown: *Pairwise Clip* (or *Select By Location*) with the Downtown polygon.
3. Ground elevation: *Extract Multi Values to Points* with the Downtown DEM, **tick "Bilinear interpolation"**
   (matches how the training ground elevations were taken). Rename the new field to `ground_m`.
4. *Calculate Geometry Attributes*: point x → `X`, point y → `Y` (in EPSG:32648 metres).
5. *Export Table* to CSV with the fields `X, Y, ground_m`. Delete rows where `ground_m` is empty (outside the DEM).

**In notebook 08 (V3-clean)** replace the one line `cols = H.map_columns(S, SPACING)` by:

```python
cols = pd.read_csv(r"E:\path\to\your_grid.csv")[["X", "Y", "ground_m"]].dropna().reset_index(drop=True)
SPACING = 50.0          # must equal the fishnet cell size (used in file names and the voxel file)
```

Everything after it (`H.map_cells`, prediction, reliability zones, CSV and netCDF export) works unchanged.
For the voxel file, the points must lie on one regular grid (a fishnet does).

**Back in ArcGIS Pro**
* Points: *XY Table To Point*, X = `X`, Y = `Y`, Z = `Z`, EPSG:32648; symbolise `clay` or `p_clay`;
  filter or colour by `reliability_zone`.
* Voxel: *Add Multidimensional Voxel Layer* (local scene) with the `.nc` file; variables `p_clay`, `clay`,
  `reliability_zone`; use vertical exaggeration (for example 20-50×) because the model is 15 m thick and kilometres wide.
* Always show the reliability zones (or the 100 m / 300 m distance rings) with the map.

---

## 4. Grid size: recommendation and reasons

**Recommendation: 50 m × 50 m horizontally, 0.5 m vertically (keep V3's grid).** Optional sensitivity: 100 m.

| Reason | Value | What it implies |
|---|---|---|
| Vertical support of the training data | cells of 0.5 m | map cells must be 0.5 m thick: the model was trained to predict 0.5 m cells, so a different thickness would predict something it never learned |
| Spatial autocorrelation range of the clay share (blockCV) | 106 m | about half the range (≈ 50 m) resolves the variation the data can describe; much finer cells only interpolate the same model surface |
| DEM resolution | 30.5 m | the model's Z depends on DEM ground elevation; cells finer than about 30 m add no new elevation information |
| Borehole spacing in Downtown | nearest neighbour: median 51 m (≈ 95 m between sites); 120 boreholes over 80.7 km² (≈ 820 m per borehole on average) | the model is reliable only within about 100 m of a borehole; finer cells would suggest detail that is not there |
| Prediction skill | 0.88 within 50 m, 0.78 at 50-100 m, ≤ 0.6 beyond | a 50 m cell is about the scale at which skill still holds near boreholes |
| File size / run time | 50 m: 968,400 cells, 10-30 min | 25 m would give about 3.9 million cells for no real gain; 100 m (≈ 242,000 cells) hides detail near boreholes |

**Literature support**
* **Hengl (2006)**, *Finding the right pixel size*, Computers & Geosciences 32(9): 1283-1298,
  doi:10.1016/j.cageo.2005.11.008. Gives rules to choose grid resolution from the input data:
  sampling (inspection) density, and the nugget and range of the variogram (one of its worked examples is
  a soil point dataset). Its point-density rule is usually quoted as about
  0.0791·√(A/N) for the finest legible cell; with A = 80.7 km² and N = 120 that is ≈ 65 m.
  **Check the exact constants in the paper before you cite numbers** (they were not verified from the
  full text here). The range-based argument (≈ half of 106 m) and the DEM argument (≥ 30 m) both point to about 50 m.
* **Stafleu et al. (2011)**, *3D modelling of the shallow subsurface of Zeeland, the Netherlands*,
  Netherlands Journal of Geosciences 90(4): 293-310, doi:10.1017/S0016774600000597. The national GeoTOP
  voxel model uses **100 m × 100 m × 0.5 m** voxels for the upper 30 m, from borehole descriptions, with
  a class probability per voxel. Precedent for thin vertical cells (0.5 m) with much wider horizontal
  cells, and for mapping probabilities, not only classes.
* **Roberts et al. (2017)** and **Valavi et al. (2019)** for the spatial CV (already cited in V3): the map's
  skill is the spatial-CV skill, which is why the reliability zones are shown with it.
* The DEM argument (no finer than the 30 m DEM) is a plain consequence of Z being taken from the DEM; it needs
  no citation, but you can add one on DEM resolution if a reviewer asks.

**Thesis sentence (adapt):** "The 3D model uses 50 m × 50 m × 0.5 m cells. The vertical size equals the 0.5 m
support of the training cells (as in the GeoTOP voxel model, Stafleu et al., 2011); the horizontal size is about
half the 106 m autocorrelation range of the clay share and coarser than the 30 m DEM, following the guidance
of Hengl (2006) that grid resolution should reflect sampling density, spatial autocorrelation and the
resolution of the DEM."
