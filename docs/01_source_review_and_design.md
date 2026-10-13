# Stage 1–3 report: source review, data audit and experimental design

**Project:** 3D Clay/Sand classification of the upper 15 m in Downtown Siem Reap using machine learning and spatial cross-validation
**Date:** 2026-10-12 (updated the same day after the workbook, boundary polygons and DEM arrived)  **Status:** Stages 1–2 complete, including verification of the CSV against its sources; Stage 3 design proposed and awaiting your review.

---

## 0. What has and has not been done

| Stage | Status | Evidence |
|---|---|---|
| 1 Source review | Done for the files received: CSV, raw Excel workbook, Downtown and 5 km ring polygons, DEM, one paper, blockCV source zip. | §1, §2 |
| 2 Data audit | Done. CSV audit, then verification of the CSV against the workbook, polygons and DEM (§1.7). Each finding of the verification was re-derived by an independent check before inclusion. | `scripts/01_audit_raw_data.py`, `scripts/02_verify_sources.py`, `outputs/0*/aggregate/` |
| 3 Methodological design | Implemented with the brief's settings; where the brief left a choice, the recommendation below is the configured default (§8). | §3–§6, §8 |
| 4 R blockCV / variograms | Scripts written and executed in the smoke test, after the external holdout was locked (§8). | `R/` |
| 5–7 Modelling, results, reporting | Notebooks written and **smoke-tested only**. No experimental model score exists yet. | `notebooks/`, §8 |
| Design revision | 2026-10-13 after advisor review: GP added; MLP, GWRF and GWXGB on the primary block size only. | §9 |
| Design revision 2 | 2026-10-13: gap filling for training (scores on measured cells only); all six models on every block size; full restart. | §10 |

There is no earlier experiment, saved prediction or model score in this project. The only label statistics computed so far are the descriptive audit counts in §1. They cover all boreholes, including the ones that will later be reserved for external evaluation. §5.6 explains why the holdout selection does not use them.

The repository is **public**, so the raw files and every borehole-level output (inventory, issue logs with source line numbers, maps, QC log) are kept out of git. They live under `data/raw/` and `outputs/*/restricted/`. Only aggregate statistics are committed, and this document names no borehole, client or project.

---

## 1. Verified summary of the dataset

### 1.1 What was received

| Item | Finding |
|---|---|
| File | `SiemReap_BH_Master_Downtown_DEM_Region_v1_ArcGIS.csv` (UTF-8 with BOM), SHA-256 `caf59d0b…0f72`. It is the workbook's master sheet exported row for row, with `Region` and the four elevation fields recomputed in ArcGIS from the supplied polygons and DEM (§1.7). |
| Workbook | `Final_Men_Ten_Mg_Hah_KGEO_Digitalized_COMPLETE_Depth_Formulas.xlsx`, SHA-256 `78a426c6…aaddc`. 42 sheets: the master `Study_Range_Stacking` (4,269 rows), 40 per-borehole digitised sheets and a `QC_Digitalization` log (42 items). |
| GIS | `downtown.shp` (80.74 km²), `Ring_5km.shp` (266.61 km²), both EPSG:32648; DEM `dem_downtown` (ESRI grid, 30.51 m float32, EPSG:32648, about 10 km around Downtown). |
| Size | 4,269 records × 41 fields; one record per physical line, so every record can be traced to a CSV line number. |
| Boreholes | **189** unique `BH_Code`s from 73 projects (`Project_No`) and 97 source report PDFs (`Source_File`). |
| Record type | Each record is a **sample interval**, not a continuous stratum. Median length is 0.45 m (SS) and 0.50 m (ST). Laboratory and in-situ fields (Atterberg limits, grading, unit weights, SPT N, etc.) are attached to the samples. |
| CRS | UTM fields are consistent with WGS 84 / UTM 48N (EPSG:32648). 65 boreholes carry an explicit "UTM48N" text coordinate, and 64 of these match the numeric fields exactly. 120 carry a lat/lon text coordinate, which converts to within 1 m of the UTM fields in 114 cases. |

Fields named in the brief that exist: `BH_Code`, `UTM_Easting_m`, `UTM_Northing_m`, `Ground_Elevation_m`, `Depth_From_m`, `Depth_To_m`, `Top_Elevation_m`, `USCS`, `Region`. All are 100 % complete except `USCS` (96.2 %) and `Region` (84.7 %; the blank values belong to 40 boreholes 13–20 km from the nearest Downtown borehole).

### 1.2 Regions and geography

| `Region` value | Boreholes | Projects | Distance to nearest Downtown borehole | Ground elevation (m), min / median / max |
|---|---|---|---|---|
| Downtown | **120** | 58 | – | 10.2 / 15.3 / 23.4 |
| Downtown_5km | **25** | 10 | 1.6–7.5 km (median 3.5 km) | 7.6 / 9.4 / 19.5 |
| Downtown_10km | 4 | 3 | 6.2–11.2 km | 1.5 / 29.7 / 33.6 |
| (blank) | 40 | 6 | 13.4–20.0 km | 1.5 / 13.1 / 40.1 |

* Downtown boreholes span 9.3 km (E) × 6.8 km (N) and are **strongly clustered**. The nearest-neighbour distance has a median of 51 m (10th percentile 14 m, 90th percentile 442 m). Half of the boreholes have another borehole within 51 m, 60 % within 100 m and 80 % within 200 m. Single-linkage grouping at 100 m gives 75 "sites", 48 of them single boreholes. One project lays about two dozen boreholes along a linear alignment in the north-west.
* The 5 km boreholes are at least 1.6 km from any Downtown borehole. That is far beyond any candidate block size, which matters for Scenario B (§5.3).
* **Resolved with the polygons (§1.7):** `Region` in the CSV equals point-in-polygon membership for all 189 boreholes. "Downtown_5km" is the ring 0–5 km outside the Downtown polygon edge; "Downtown_10km" boreholes lie 5.3–9.6 km and blank-Region boreholes 10.5–19.6 km from the polygon.

### 1.3 Depths and elevations

* `Top_Elevation_m = Ground_Elevation_m − Depth_From_m` holds exactly for all 4,269 records, and the same check passes for the bottom and mid elevations. The convention is depth below ground, positive downwards.
* **Ground elevations are DEM-derived, and this is now confirmed exactly:** for all 149 boreholes inside the supplied DEM (including all 145 in the two training extents) `Ground_Elevation_m` equals a bilinear sample of that DEM at the UTM coordinates, to float32 precision (§1.7). No borehole in the training extents has a surveyed ground level in the data. One of the two 1.5 m values is the DEM's own value: the DEM contains a 13,295-cell flat at exactly 1.5 m, south of Downtown. The other lies outside the DEM. Both boreholes are outside the study extents. The DEM's product, vertical datum and DSM/DTM nature are still unknown (M3).
* 181 of 189 boreholes reach at least 15 m. The 8 that do not include 5 Downtown boreholes ending at 5.05–9.45 m.

### 1.4 Soil labels

USCS symbols on records that start above 15 m:

| USCS | Downtown | Downtown_5km | Binary target (brief) |
|---|---|---|---|
| CL | 955 | 157 | Clay = 1 |
| CH | 21 | 22 | Clay = 1 |
| SC | 971 | 219 | Sand = 0 |
| SM | 37 | 19 | Sand = 0 |
| SP | 2 | 0 | Sand = 0 |
| ML | 0 | 11 | **excluded pending review** |
| (missing) | 92 | 25 | excluded (no label) |

Findings that bear on the target definition:

1. **The binary target is effectively "fines ≥ 50 %".** SC and CL dominate. Among the 1,889 lab-tested samples with a USCS symbol, **23.5 % have fines between 45 % and 55 %**, so the Clay/Sand boundary sits inside the main mode of the grading distribution (`fig_fines_vs_logged_class.png`). Even a perfect spatial model will face irreducible label ambiguity for borderline sandy-lean-clay / clayey-sand material. This has to be stated when interpreting Macro-F1.
2. **43 lab-tested samples (2.3 %, 17 boreholes) have a logged group that contradicts their own fines content** under ASTM D2487: logged CL with fines < 50 %, or logged SC with fines ≥ 50 %. Of these, 17 are in Downtown and 26 in the 5 km zone, concentrated in two projects (issue I08).
3. **ML / CL-ML occur only outside Downtown**: 25 records in one 5 km project (24 ML, 1 CL-ML) and 1 ML record in the 10 km zone. Of the 21 ML records with Atterberg limits, **14 plot as CL** on the plasticity chart (PI > 7 and above the A-line) and 7 as ML; the CL-ML record plots as CL-ML. Per the brief they are **not** converted automatically; options are in D2.
4. **161 records have no USCS** (19 boreholes, 16 of them in Downtown). They are mostly SS samples in recent projects. They are left unlabelled; nothing is imputed.
5. 11 records logged CL have LL ≥ 50 (CH by definition). Both map to Clay, so the binary target is unaffected.

### 1.5 Interval structure and anomalies

| Code | Issue | Records / boreholes | Proposed handling |
|---|---|---|---|
| I02 | Single records spanning > 3 m. **Two 5 km boreholes have one record each covering 0–20.5 m (CL) and 0–17.5 m (SC)**, followed by normal 0.5 m samples below; the 10 km zone has stratum-type "Interval" records. | 7 / 4 | Not treated as measured strata. The two 5 km boreholes contribute no labels in 0–15 m until the source logs are checked (D4). |
| I03 | Overlapping intervals (19.45–19.95 vs 19.0–19.5 m). | 1 / 1 | Below 15 m; no effect. |
| I04 | Text coordinate vs UTM field mismatch > 5 m: 54, 60 and **600 m** for three Downtown boreholes, 64 m (5 km), 452 m (10 km, two boreholes). None changes region. Two text values are copies of a neighbouring borehole's location (UTM kept); the 10 km pair's offset equals one WGS 84 / Indian 1960 datum conversion; the 600 m and 60 m cases differ in easting only (§1.7). | 6 / 6 | Keep the UTM fields. M4 is narrowed to the two easting-only cases. These boreholes are barred from the external set until resolved. |
| I05 | Coordinate text empty (UTM present). | 4 / 4 | Informational. |
| I06 | USCS missing. | 161 / 19 | Unlabelled. |
| I07 | USCS outside the binary map (ML, CL-ML). | 26 / 4 | D2. |
| I08 | Logged group contradicts lab fines. | 43 / 17 | D5. |
| I11 | Ground elevation exactly 1.5 m. One is the DEM's own value on a 1.5 m flat; the other lies outside the DEM. | 2 / 2 | Outside both scenarios. |
| I13 | Borehole shorter than 15 m. | 8 / 8 | Use the measured part only. |
| I14 | Non-standard `Sample_Type` text (`SS-1`, `-`, blank). | 21 / 5 | Cosmetic. |
| I15 | `Source_File` empty. | 16 / 2 | 10 km zone only; outside both scenarios. |

No duplicate records, zero or negative interval lengths, within-borehole coordinate changes or elevation-formula errors were found. Further issues found by comparing the CSV with its sources are listed in §1.7.

### 1.6 What the 0.5 m discretisation produces (audit preview)

The data are **sample intervals separated by unsampled gaps** (typical gap ≈ 0.55–1.05 m). The brief forbids extending a recorded layer or labelling unknown intervals, so the only compliant rule is to label a 0.5 m cell from the sample intervals that actually overlap it. The preview rule is:

> Cell *j* covers depth [0.5 *j*, 0.5 *j* + 0.5) m, *j* = 0…29. It receives a class if at least 50 % of its thickness is covered by labelled sample intervals of a single class (CL/CH or SC/SM/SP) and by no interval of the other class. Records longer than 3 m are ignored. The cell's predictor elevation is `Ground_Elevation_m − 0.5 j`, the brief's formula applied to the cell top.

| | Downtown | Downtown_5km |
|---|---|---|
| Boreholes with ≥ 1 labelled cell | 120 | 23 |
| Labelled cells | **2,064** | 453 |
| Clay share | 48.2 % | 41.5 % |
| Clay share at 0–5 / 5–10 / 10–15 m | 59.6 / 52.9 / 30.7 % | 61.4 / 32.5 / 30.1 % |
| Conflicting cells (both classes) | 0 | 0 |

* Only **57 %** of the 0–15 m cells in Downtown boreholes receive a label. Most logs follow a **1.5 m sampling cycle** with two samples (typically one SS and one ST) per 1.5 m, so every third 0.5 m cell holds only 4–9 labels across all 120 boreholes against roughly 70–110 for the others (`fig_clay_share_by_depth.png`). Training data are therefore concentrated at recurring depths, and the 3D map will interpolate vertically between them. Depth-specific metrics should use the 0–5, 5–10 and 10–15 m bands, not single cells.
* Clay dominates the upper ~8 m and clayey sand dominates below ~10 m. **The 5 km zone is sandier than Downtown at 5–13 m.** That non-stationarity is the central hypothesis for Scenario A vs B: added 5 km data could bias Downtown predictions towards Sand at mid-depth. This is a hypothesis to test, not a result.
* 102 of 120 Downtown boreholes contain both classes, 6 are all Clay and 12 are all Sand within the labelled cells. Borehole-level Macro-F1 is undefined for the 18 single-class boreholes (§5.7).

* The preview counts include **four Downtown boreholes logged as contiguous 1.5 m disturbed/undisturbed (D/U) records** (54 records; §1.7, V3). Each such record labels three cells, so these four boreholes are labelled almost continuously, unlike the gapped 0.45–0.5 m samples elsewhere. D12 decides how to treat them.

The discretisation rule is **not frozen** (D3). Nothing in the preview is a modelling result.


### 1.7 Verification against the workbook, polygons and DEM

`scripts/02_verify_sources.py` re-derives the following from the received files. Every item was first found by one analysis and then re-derived independently by a second analysis that tried to refute it. Items that the second analysis corrected are given here in their corrected form. Aggregate outputs are in `outputs/02_source_verification/aggregate/`; borehole-level evidence is in the git-ignored `restricted/` folder.

**How the CSV relates to its sources**

* The CSV is the workbook master sheet `Study_Range_Stacking` in the same row order: borehole code and depth interval agree in all 4,269 rows. 33 of 41 columns are identical, including `USCS`, both depth fields, both UTM fields and all laboratory values. **Labels, depths and XY predictors therefore do not depend on which file is used.**
* The 8 differing columns are:
  * `Region`, recomputed from the polygons;
  * `Ground_Elevation_m` and the three derived elevations, recomputed from the DEM;
  * `PL_pct`, `Pocket_Penetration_kPa` and `SPT_N`, where the export blanked **85 text qualifiers**: 45 SPT refusal or partial-penetration entries, 32 "> 300" pocket-penetrometer readings, 7 "Non-Plastic" and 1 "Inadequate Sample". None is a predictor, but the CSV understates SPT refusals and non-plastic fines.
* **The workbook's own `Region` column is an older, different zoning** (City / Buffer_5km / Buffer_10km) and is blank for 39 boreholes. Mapped to the nearest equivalents, it disagrees with the polygons for 57 of 189 boreholes. It must not be used; scenario membership comes from the polygons.
* **Ground elevation:** the CSV values are bilinear DEM samples for all 149 boreholes inside the DEM. The workbook is blank for 39 boreholes (36 Downtown) whose elevation formulas therefore evaluate to empty. For 29 others it holds nearest-cell values that differ from the CSV by up to 0.51 m, and one value is copied from a neighbouring borehole. The CSV, with one documented sampler, is the consistent source. The 40 boreholes outside the DEM take their elevations from a raster that was not supplied; all are outside both training extents.

**Polygons**

* Both polygons are valid single-part polygons in EPSG:32648. The ring's hole is identical to the Downtown polygon (symmetric difference 0 m²). Its outer edge equals a 5,000 m planar buffer of Downtown to within 1.4 m; the symmetric difference is 0.007 % of the ring area, from arcs converted to chords.
* Region labels are robust to small boundary or coordinate errors. The nearest borehole outside Downtown is 556 m out; the nearest inside is 30.7 m from the edge (2 boreholes within 50 m, both at the straight 12.3 km northern edge, which the file metadata records as a clip by a city AOI).

**Further data-quality findings**

| # | Finding | Effect on the training extents | Proposed handling |
|---|---|---|---|
| V1 | **Two boreholes of the 24-borehole power-line project (BH-6, BH-8) have depth conflicts inside the workbook.** Their digitised sheets record overlapping intervals at 15.8–23.5 m and 17.0–23.1 m. The master holds the project's regular 0.5–21.45 m template. The QC log's maximum depths match the sheets, not the master. Their SS "fines" values are depths (Depth_To + 0.05 m). Every labelled row is logged SC, including 10 ST samples whose own grading gives ≥ 50 % fines. | 58 records; 35 Sand labels in Downtown 0–15 m; 10 of the 17 Downtown I08 conflicts | Exclude from the primary analysis until the source logs are checked; sensitivity with them (D11, M16) |
| V2 | **Undocumented coordinate provenance for 5 Downtown boreholes.** For four, the QC log says the source report states no coordinate, yet UTM values are present. For one power-line borehole, the QC log records no coordinate, yet the master has one. | 5 Downtown boreholes | Keep, but bar from the external set (M17) |
| V3 | **Contiguous long records:** 143 records longer than 1 m in 11 boreholes. In Downtown, 4 boreholes from two projects are logged as continuous 1.5 m D/U records (54 records) covering most of 0–15 m. | 4 Downtown boreholes labelled almost continuously | D12 |
| V4 | **SS labels never have their own laboratory data.** Of 1,987 SS records with a USCS symbol, 1,969 have no grading, Atterberg or water-content values; the other 18 are the V1 artefacts. Label density also depends on the digitisation batch: the newest digitised projects left SS rows unclassified (in Downtown 0–15 m, 235 of 320 SS rows labelled in the appended block vs 747 of 750 elsewhere). | About half of all Downtown labels are SS labels | Keep as logged; sensitivity with lab-supported labels only (D13) |
| V5 | **The QC log (42 items, all "PASS") contains no check of USCS labels.** It cannot be cited as validating the classification. | All labels | Rely on the source reports; report as a limitation |
| V6 | **DEM characteristics.** 30.51 m float32 grid with a pixel size consistent with a reprojected ~1 arc-second product; product, version, vertical datum and DSM/DTM status are not recorded. Large constant-value flats (1.5 m, 17.0 m) suggest hydro-flattened water surfaces (an interpretation, not verified). Local anomalies in Downtown are strongly one-sided: 23 cells more than 5 m above their 3 × 3 median, none 5 m below. That fits above-ground objects (DSM) but also embankments. No borehole cell sits on such a spike. | Every `Top_Elevation_m` value | M3 |
| V7 | **`Top_Elevation_m` is sensitive to DEM sampling at the scale of the 0.5 m cell.** At the 120 Downtown boreholes, nearest-cell and bilinear samples differ by a median of 0.15 m (19 boreholes > 0.5 m). A typical 15 m horizontal error changes the bilinear value by about 0.24 m. Boreholes on the same site within 30 m differ by up to 1.0 m purely from the DEM gradient. This is a sensitivity, not a measured error. | Every borehole | Use the same bilinear sampler for boreholes and the 3D grid; report as a limitation |

**The prediction domain (geometry only)**

A 25 m grid of XY columns inside the Downtown polygon has 129,179 columns. Distance from a column to the nearest of the 120 Downtown boreholes:

| Percentile | 5 | 10 | 25 | 50 | 75 | 90 | 95 | max |
|---|---|---|---|---|---|---|---|---|
| Distance (m) | 134 | 203 | 371 | **677** | 1,309 | 2,213 | 2,718 | 3,730 |

Only 3.1 % of the Downtown area lies within 100 m of a borehole, 18.4 % within 300 m and 65.2 % within 1 km. The boreholes' convex hull covers 49.9 % of the polygon. **The typical prediction distance is about 13 times the median borehole spacing (51 m)** (`fig_prediction_vs_borehole_distances.png`). Adding the 5 km boreholes changes the nearest borehole for 8 % of columns but leaves the median unchanged (675 m). These figures use all Downtown boreholes and will be recomputed with development boreholes once the external set is fixed. Their consequences for the validation design are in §4.2–4.3.

---

## 2. Assessment of the published research

### 2.1 Supplied paper: Valavi et al. (2019)

Valavi, Elith, Lahoz-Monfort & Guillera-Arroita (2019), *Methods Ecol. Evol.* 10: 225–232, doi:10.1111/2041-210X.13107. It introduces blockCV for species distribution models, with three strategies: spatial blocks, buffers (spatial LOO) and environmental blocks. It also offers variogram-based guidance on block size and interactive tools. Points that matter here:

* **Block size.** The paper follows Roberts et al. (2017): blocks should relate to the range of spatial autocorrelation, ideally in model residuals. Before any model exists, a variogram of the raw response is a defensible proxy for presence–absence-type (binary) data. That is our situation, because Clay/Sand is binary.
* **Blocks do not guarantee separation.** The paper notes, and the package documentation repeats, that points on either side of a block edge are not spatially separated. Buffering avoids this at a computational cost.
* **The paper is 2D and predictor-raster oriented.** Our problem is 3D, but every borehole is a vertical line, so blocking in the horizontal plane keeps a borehole's samples together by construction. The autocorrelation step is different: the paper's default looks at predictor rasters, which we do not have. Our three predictors are coordinates and elevation, whose "autocorrelation" is trivial. The response variogram is the only meaningful option.

### 2.2 The supplied blockCV source is version 4.0-1, not the version in the paper

The zip contains blockCV **4.0-1 (DESCRIPTION dated 2026-09-22)**, SHA-256 `0d8c8a88…817e`. I read the R sources of `cv_spatial`, `cv_spatial_autocor` and `cv_block_size`, plus the NEWS file. Implications:

* The functions named in the paper (`spatialBlock`, `spatialAutoRange`, `buffering`, `rangeExplorer`) **were removed in 4.0-0.** The R scripts must use `cv_spatial`, `cv_spatial_autocor`, `cv_buffer`, `cv_nndm`/`cv_knndm`, `cv_distance` and `cv_summary`. The thesis should cite both the 2019 paper and the exact package version.
* `cv_spatial` defaults to **hexagonal** blocks. In `sf::st_make_grid`, which blockCV calls, the hexagon `cellsize` is the distance between opposite edges, so a "100 m" hexagon covers 0.866 × 100² m². I recommend **square blocks (`hexagon = FALSE`)** so that "100 m block" means a 100 m square, as in the paper's Fig. 1a. This also makes `offset` easy to use for sensitivity analysis.
* Fold balancing (`selection = "random"`, `balance = TRUE`) was reworked in 4.0-0. It scores test folds per class with a chi-square criterion, so folds for a given seed differ from 3.x. The installed version and seed must be recorded.
* **Borehole-integrity risk found in the code.** When a point touches two blocks, `cv_spatial` resolves the duplicate *per record*, at random. If sample-level rows are passed and a borehole sits exactly on a block edge, its samples could be split across folds. The probability is negligible with millimetre-precision coordinates, but the R script will assert that every borehole has one fold ID and will stop otherwise.
* `cv_spatial_autocor(x, column)` fits `automap::autofitVariogram` and reports `var_model[2, 3]`. That is the fitted model's **range parameter**, not necessarily the practical range (§3.3). The script will convert it explicitly.
* 4.0-0 adds `cv_distance`, which compares the test-to-nearest-train distances of a fold design with the prediction-to-nearest-train distances of the mapping domain (Wasserstein-1). It also adds `cv_knndm` (Linnenbrink et al. 2024). Both answer the brief's point that block size does not equal exclusion distance, and both need the prediction-domain polygon (M2).
* The source contains C++ (`src/`), so installing it from the zip on Windows needs **Rtools**.

### 2.3 Additional literature (independently checked)

Only one paper was uploaded. The brief mentions papers on ML soil classification and 3D subsurface prediction, so please send those you intended (M11). I checked the works below through publisher or indexing records, at **abstract level unless stated otherwise**. I have not read the full text of any of them, and nothing below is quoted beyond what the abstracts state.

| Reference | Relevance | Verification |
|---|---|---|
| Roberts et al. 2017, *Ecography* 40: 913–929 | Blocking for structured data; residual-range guidance | Cited in the supplied paper |
| Le Rest et al. 2014, *Global Ecol. Biogeogr.* 23: 811–820 | Spatial LOO with buffer | Cited in the supplied paper |
| Schratz et al. 2019, *Ecol. Model.* 406: 109–120 | **Nested spatial CV.** Hyperparameter tuning must also be spatial; non-spatial CV was markedly over-optimistic; tuning gains saturated at about 50 iterations (preprint figures) | Web, abstract/preprint |
| Meyer, Reudenbach, Wöllauer & Nauss 2019, *Ecol. Model.* 411: 108815 | **Coordinates as predictors** cause overfitting: models reproduce training data and fail at spatial prediction. Directly relevant, because our only predictors are E, N and elevation | Web, abstract |
| Ploton et al. 2020, *Nat. Commun.* 11: 4540 | Random CV hides near-zero spatial predictive power | Web, abstract |
| Wadoux, Heuvelink, de Bruin & Brus 2021, *Ecol. Model.* 457: 109692 | Spatial CV is **not** a design-based estimate of map accuracy. Our results must be framed as performance under a stated prediction-distance scenario, not as "map accuracy" | Web, abstract |
| Milà, Mateu, Pebesma & Meyer 2022, *Methods Ecol. Evol.* 13: 1304–1316 | NNDM LOO CV: match CV distances to prediction distances | Web, abstract |
| Linnenbrink, Milà, Ludwig & Meyer 2024, *Geosci. Model Dev.* 17: 5897–5912 | kNNDM (k-fold version), implemented in blockCV 4.0 | Web, abstract |
| Brenning 2023, *Int. J. Geogr. Inf. Sci.* 37: 584–606 | Spatial prediction error profiles: error as a function of prediction distance. The template for our distance-sensitivity analysis | Web, abstract |
| Georganos et al. 2021, *Geocarto Int.* 36: 121–136 (online 2019) | **Geographical Random Forest**: local RFs combined with a global RF; R package SpatialML | Web, abstract |
| Sun, Zhou, Kim & Hu 2024, *Trans. GIS* 28: 2476–2491 | PyGRF: principled bandwidth and local-weight selection, local sample expansion | Web, abstract |
| Grekousis 2025, *J. Geogr. Syst.* 27: 169–195 | **Geographical-XGBoost**: local XGBoost models ensembled with a global model via α; bandwidth by CV. **Formulated for regression** | Web, abstract and package docs |
| Hu et al. 2024, *Eng. Geol.* (Singapore, 165 boreholes) | RF and XGBoost outperformed ANN and SVM for 3D stratigraphy; neighbourhood aggregation added about 5 % borehole accuracy | Aggregator listing only; volume and DOI to be confirmed |
| Khamjan, …, Likitlersuang 2026, *Eng. Geol.* 371: 108880 (Bangkok) | Geographically weighted RF/XGBoost variants for soil classification and a 3D model. The closest precedent to this project | Index record only; **full text not checked** |
| Akiba et al. 2019 (Optuna, *KDD*); Chen & Guestrin 2016 (XGBoost, *KDD*); Breiman 2001 (*Mach. Learn.*); Varma & Simon 2006 (*BMC Bioinformatics*); Cawley & Talbot 2010 (*JMLR*); Journel 1983 (*Math. Geol.*, indicator geostatistics); ASTM D2487 | Standard method references | From my own knowledge; confirm DOIs before final citation |

**Critical synthesis for this project**

1. With **E, N and elevation as the only predictors**, every model here is a 3D spatial interpolator. Meyer et al. (2019) show such models can fit training data well while transferring poorly. Spatial CV is what exposes this. If spatial structure is weak at the evaluated distances, performance will fall towards what a depth-only rule can achieve. The design should include that reference (D6), so any gain from horizontal spatial information is measured and not assumed.
2. Under the Wadoux et al. (2021) critique, with non-probability (project-driven) sampling, no CV variant estimates Downtown-wide map accuracy without bias. The honest framing is *"accuracy when predicting at sites X–Y m from the nearest training borehole"*, with the X–Y range reported for each design. That is why the prediction-distance diagnostics in §4 are central.
3. Both published GW tree methods (GRF and G-XGBoost) were built for **regression** and areal or point data. Applying them to **binary classification of 3D samples grouped in boreholes** is an adaptation and must be described as one (§5.4).
4. The geotechnical ML literature found so far reports RF and XGBoost as strong for borehole stratigraphy. I could not confirm whether those studies used borehole-grouped or spatially blocked validation, so their scores are not comparable to ours.

---

## 3. Spatial autocorrelation analysis appropriate for this dataset

None of this has been run yet. It runs in R (gstat, automap, blockCV) on **development boreholes only**, after the external set is fixed.

### 3.1 Variable

The binary indicator *I*(**u**) = 1 if the 0.5 m cell at **u** = (*x*, *y*, *z*) is Clay, else 0. The indicator semivariogram is

  γ_I(**h**) = 1 / (2 N(**h**)) · Σ_{(i,j): **u**_i − **u**_j ≈ **h**} [I(**u**_i) − I(**u**_j)]²

(Journel 1983). Its sill approximates *p*(1 − *p*), with *p* the clay proportion.

### 3.2 Why depth must be handled explicitly

The clay proportion falls from about 0.6 to about 0.3 over 15 m (§1.6). A single horizontal variogram that pools all depths mixes vertical trend into horizontal variance. Proposed analyses:

1. **Horizontal, depth-sliced indicator variograms** for 1.5 m slices aligned with the sampling cycle, and as a coarser check for the 0–5, 5–10 and 10–15 m bands. Each borehole contributes **one value per slice** (the clay fraction of its labelled cells in that slice), so within-borehole pairs never enter as zero-distance pairs.
2. **The same variograms on depth-detrended residuals**, *R* = *I* − *p̂*(*z*), with *p̂*(*z*) the development-set clay share of the slice.
3. **Vertical indicator variogram** from within-borehole pairs. This gives the vertical correlation length, i.e. layer thickness, and the horizontal/vertical anisotropy that matters for interpreting the 3D map.
4. **Join-count or Moran's I tests** per slice on a *k*-nearest-borehole graph, with permutation p-values. These test whether horizontal dependence exists at all at the scales the data can resolve.
5. **Pair counts per lag** reported with every variogram. Clustering means the shortest lags (< 50 m) are dominated by within-project pairs and the 200–1,000 m lags are thin. **Bootstrap over sites** (resampling 100 m site clusters) gives an interval for the range.

`cv_spatial_autocor` is called for each slice for consistency with blockCV. `gstat::variogram` and `fit.variogram` are run alongside with explicit lag widths (first lag ≤ 25 m, cutoff ≈ 3 km), so the fit does not depend on automap's default binning.

### 3.3 Variogram parameters vs effective correlation distance

A fitted model has a nugget *c*₀, a partial sill *c* and a **range parameter** *a*. These are model coefficients, and *a* means different things in different models (gstat parameterisation):

| Model | γ(*h*) − *c*₀ | Practical (95 %) range |
|---|---|---|
| Spherical | *c*[1.5(*h*/*a*) − 0.5(*h*/*a*)³] for *h* ≤ *a* | *a* (reaches the sill exactly) |
| Exponential | *c*[1 − exp(−*h*/*a*)] | ≈ 3*a* |
| Gaussian | *c*[1 − exp(−(*h*/*a*)²)] | ≈ √3·*a* ≈ 1.73*a* |
| Matérn/Stein (κ) | – | no closed form; solved numerically |

blockCV reports *a*. For an exponential fit the **effective correlation distance is three times the reported number**. The report will give *a*, the practical range *r*₉₅ (where γ reaches 95 % of *c*₀ + *c*), and the **nugget-to-sill ratio**. A high nugget ratio means much of the Clay/Sand variation is uncorrelated even at the shortest resolvable lag, through borderline labels (§1.4), logging differences or genuine small-scale heterogeneity. In that case "spatial autocorrelation range" overstates how much a neighbour actually tells you.

These are response variograms, which blockCV documents as **exploratory proxies** for residual autocorrelation. After the outer CV, the residual (*I* − *p̂*) variograms will be reported as a diagnostic. They will **not** be used to change the block size retrospectively (§4.3).

---

## 4. Spatial-CV and nested-CV designs compared

### 4.1 Outer-loop options

| Design | Removes within-borehole leakage | Removes near-neighbour leakage | What it estimates | Verdict |
|---|---|---|---|---|
| Random row k-fold / 80:20 | No | No | Re-prediction of known boreholes | Invalid as principal method (brief). Not used. |
| Leave-borehole-out (group k-fold) | Yes | No: 50 % of boreholes have a neighbour within 51 m | Interpolation within sites | Reference only |
| Leave-site-out (100 m clusters) | Yes | Partly | New borehole at a new site | Possible alternative |
| **blockCV `cv_spatial`, 100 / 200 / 300 m squares, k = 5, random balanced** | Yes | Depends on size (below) | Prediction at block-separated distances | **Required; primary design** |
| `cv_buffer` (spatial LOO with radius) | Yes | Yes, by construction | Prediction at ≥ radius | Robustness / purge analysis |
| `cv_knndm` / `cv_distance` | Yes | Matched to the prediction domain | Map-like prediction distances | **Diagnostic to justify block size**, and proposed additional analysis (D15); the boundary is now available |

### 4.2 What the candidate block sizes imply here

Geometry-only planning diagnostic: a square grid anchored at the data minimum, **not blockCV and not used for folds**.

| Block | Occupied blocks | Boreholes with a borehole in another block within 25 m | … within 100 m | Median distance to nearest borehole in another block |
|---|---|---|---|---|
| 100 m | 88 | 6 | 38 | 149 m |
| 200 m | 75 | 2 | 22 | 182 m |
| 300 m | 68 | 0 | 13 | 319 m |

With five random folds, about 4 in 5 of those other-block neighbours land in training. So under **100 m blocks a substantial share of test boreholes will have a training borehole within tens of metres**: 100 m CV largely measures interpolation inside site clusters. At 300 m, no test borehole can have a training neighbour closer than about 50 m, and the median separation is around 300 m.

**Now quantified with the boundary (§1.7):** the 3D map must predict at a median distance of 677 m from the nearest borehole (P90 2.2 km), and 82 % of the Downtown area is more than 300 m from any borehole. Even the largest candidate block (300 m) therefore tests prediction distances well below those of the mapping task. Two consequences:

1. Whatever block size is primary, CV scores describe **near-borehole prediction**. They are not Downtown-wide map accuracy (Wadoux et al. 2021).
2. The map must carry a **validated-distance mask**: columns farther from the nearest development borehole than the test-to-train distances actually evaluated in CV (for example their 95th percentile, from `cv_distance`) are flagged as unvalidated extrapolation, not reported as if they had the CV accuracy.

To estimate performance at map-like distances, I propose one **additional** blockCV analysis: `cv_knndm` (Linnenbrink et al. 2024) with the 25 m Downtown grid as the prediction domain. It sits beside the 100/200/300 m designs and does not replace them, and needs your approval because it goes beyond the brief's candidate sizes (D15).

### 4.3 Recommended rule for the primary block size (fixed before any model is fitted)

The brief plans 100 m as primary, subject to review, and forbids choosing it by score. I propose writing this rule into the configuration before Stage 4, using only development-set geometry and the response variograms:

1. Compute *r̃*: the median over depth slices of the practical range *r*₉₅ of the detrended indicator variograms (§3.2), with a site-bootstrap interval.
2. Compute the prediction-domain distribution: horizontal distance from every 3D-grid column in the Downtown polygon to the nearest development borehole.
3. For each of 100, 200 and 300 m, generate the blockCV folds and compute `cv_distance`: the test-to-nearest-train distance distribution and its Wasserstein distance *W* to the prediction distribution.
4. **Primary = 100 m** if 100 m ≥ *r̃* and *W*(100) is within 10 % of the smallest *W*. Otherwise primary = the candidate ≥ *r̃* with the smallest *W*. If no candidate reaches *r̃*, primary = 300 m and the limitation is reported. The other two sizes are always reported as robustness analyses.

Given the prediction-distance distribution in §1.7, step 3 will very likely favour the largest candidate. That expectation comes from geometry alone, not from any score. The rule is kept as written so that the choice stays mechanical and is made before modelling.

The rule, the computed quantities and the resulting choice are written to `config/` and committed **before** any model sees the folds.

### 4.4 Inner loop (nested CV)

* **Default, as the brief prefers:** for outer fold *k*, the four remaining R-generated folds are the inner folds (inner 4-fold, spatially blocked by construction).
* **Pre-specified adequacy check** for every outer fold: each inner validation group has ≥ 8 boreholes, ≥ 100 labelled cells, and each class ≥ 20 % of its cells. With about 108 development boreholes, each fold has about 21 boreholes and should pass. Any fold that fails is handled in R: `cv_spatial(k = 4)` with the same block size and seed on that outer-training set, exported to CSV and imported by Python. If that also fails, k = 3. The fallback is logged per fold.
* Inner folds alone control Optuna tuning, class-weight choice, the MLP scaler, the GW bandwidth and local weight, and the decision threshold (chosen to maximise inner out-of-fold Macro-F1). Outer-test labels are never read during tuning.

### 4.5 Spatial purge (separate analysis)

The purge removes from training every borehole within *d* of any test borehole, with *d* ∈ {0 (blocks only), 100 m, 200 m}, using the primary folds. The number of training boreholes lost per fold is reported, because with 108 boreholes the purge also costs training data. It is evaluated separately and never mixed into the primary comparison.

---

## 5. Recommended training and validation architecture

### 5.1 Data flow

```
raw CSV (checksummed)
 └─ audit (done) → resolve D/M items → preprocessing: 0.5 m cells, traceable to csv_line
     └─ external holdout selection (coordinates + sampling metadata only; IDs hashed and committed)
         └─ development set (Downtown dev + 5 km)
             ├─ R: variograms, block-size rule, cv_spatial 100/200/300 m → folds_*.csv (+ maps, cv_distance)
             └─ Python: import R folds (never regenerated) → nested CV × 5 models × 2 scenarios
                 └─ pre-declared selection → final refit on all dev data → freeze
                     └─ single evaluation on external boreholes → 3D prediction for ArcGIS Pro
```

### 5.2 Predictors and preprocessing

* Exactly `UTM_Easting_m`, `UTM_Northing_m`, `Top_Elevation_m`, with `Top_Elevation_m = Ground_Elevation_m − (cell top depth)`. The 3D grid uses the same convention, so training and prediction match.
* Trees (RF, XGB, GW variants) need no scaling. The MLP uses a `StandardScaler` fitted inside each training partition.
* Class weighting (none vs balanced) is an inner-CV hyperparameter. No imputation is needed: the three predictors are complete.
* Two limitations of the fixed predictors go into the thesis without changing them: absolute elevation mixes ground-surface variation (10–23 m in Downtown) with stratigraphic depth, and axis-aligned coordinate splits give blocky maps (Meyer et al. 2019). Adding depth or rotated coordinates (Møller et al. 2020, *SOIL* 6: 269–289) would be a future step that needs your explicit permission.

### 5.3 Training-extent scenarios

* **Test sets are identical:** the Downtown development boreholes in outer fold *k*.
* **Scenario A** trains on the Downtown development boreholes not in fold *k*.
* **Scenario B** trains on the same boreholes plus all eligible 5 km boreholes.
* The 5 km boreholes lie ≥ 1.6 km from any Downtown borehole, so no 100–300 m block can put them in a test block. In R, the block grid is built over the combined extent and folds are assigned from the Downtown boreholes. 5 km boreholes are written to the fold CSV as `train_only`, with their distance to the nearest borehole of each test fold, which documents the brief's exclusion requirement. In inner CV for Scenario B, 5 km boreholes are always inner-training and never inner-validation, so tuning targets Downtown performance.

### 5.4 The five model families

| Model | Formulation | Tuned (Optuna TPE, inner CV) |
|---|---|---|
| RF | scikit-learn `RandomForestClassifier` | n_estimators, max_depth, min_samples_leaf, max_features, class_weight |
| XGB | `xgboost.XGBClassifier`, `tree_method="hist"` | n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, scale_pos_weight |
| MLP | scikit-learn `MLPClassifier` with in-fold standardisation | hidden layers/units, alpha, learning_rate_init, batch_size; early stopping on an in-fold group split |
| **GWRF** | Following Georganos et al. (2021): one **local** RF per training borehole, fitted on the cells of its *k* nearest training **boreholes** (adaptive bandwidth counted in boreholes, so long boreholes do not shrink the neighbourhood). Prediction at a location uses the local model of the nearest anchor, blended with a global RF: *p̂* = α *p*_local + (1 − α) *p*_global | *k*, α, optional bisquare distance weights *w* = (1 − (*d*/*d*_k)²)² (on/off), local RF size |
| **GWXGB** | Same anchors and blending with local XGBoost classifiers, following the local–global ensemble of Grekousis (2025). G-XGBoost's error-linked spatial weights are defined for regression; the classification variant here is **our adaptation** and will be described as such | *k*, α, distance weights, local XGB parameters |

GW safeguards: a local model needs ≥ 5 boreholes and ≥ 20 cells of each class. Otherwise its neighbourhood grows up to *k*_max, and if still short the anchor falls back to the global model. Fallback rates are reported. Cost per fit is roughly (number of anchors ≈ 110–135) local fits on a few hundred cells each.

Determinism: fixed seeds per model, fold and trial; Optuna `TPESampler(seed)` with sequential trials; a fixed thread count for XGBoost and RF. A test reruns one fold twice and compares prediction hashes.

**Budgets:** 50 Optuna trials per model per outer fold for RF, XGB and MLP, where Schratz et al. report tuning gains saturating at about 50 iterations. 30 trials for GWRF and GWXGB, with the shared tree parameters' ranges narrowed around the global optimum. The full search spaces go into `config/` before Stage 5.

### 5.5 Reference baseline (proposed, D6)

A **depth-band prior**: predict P(Clay) as the training-fold clay share of the cell's 1.5 m depth slice, thresholded in inner CV. It uses no horizontal information. It is not a sixth model family, only the yardstick that shows whether E and N add skill. It needs your approval because the brief says "exactly five model families".

### 5.6 External holdout: proposed selection procedure

The selection uses **coordinates and sampling metadata only, never class values**, and runs before any variogram or fold is computed.

1. **Eligible:** Downtown boreholes reaching ≥ 15 m, with ≥ 10 sampled intervals in 0–15 m that carry any USCS symbol (presence, not class), and no unresolved data-quality issue: coordinate mismatch (I04), undocumented coordinate provenance (V2), depth conflict (V1) or contiguous long records (V3). These criteria exclude 12 of 120 boreholes. 101 remain eligible; the remaining 7 fail the depth or record-count criteria. The excluded boreholes stay in development unless D11/D12 say otherwise.
2. **Sites:** single-linkage clusters at 100 m (coordinates only).
3. **Generalisation group (5 sites):** whole sites whose nearest borehole outside the site is **≥ 500 m** away. This is raised from 300 m because the median prediction distance is 677 m. 15 eligible sites (23 boreholes) qualify. Downtown is split into 3 spatial zones by seeded k-means on coordinates, and sites are sampled across zones with a fixed seed. These test **geographically unfamiliar sites**.
4. **Interpolation group (5 boreholes):** one randomly chosen borehole from each of 5 sites with **≥ 2** boreholes, sampled across zones. This is relaxed from ≥ 3 because only 6 eligible sites have three or more. The chosen borehole's site neighbours (< 100 m) stay in development. These test **interpolation near existing boreholes**.
5. The sorted ID list is written to `outputs/…/restricted/`, and its **SHA-256 is committed** to the public repository. That fixes the list permanently without publishing borehole identities, and shows it was never changed after scores existed.

Disclosure for the thesis: the external boreholes were selected after a descriptive audit of all labels (§1) but before any modelling or validation design, and the selection rule used no class information. If any of these boreholes were used in **your own earlier modelling** outside this project, they must be called *retrospective holdouts* (M10). The boundary files' metadata refer to an ArcGIS project named "Siem_Reap_3D_V5", which suggests earlier versions of this work exist.

### 5.7 Metrics and reporting

For every model × scenario × block size: Accuracy, balanced accuracy, Macro-F1, Clay recall, Sand recall, MCC, ROC-AUC, PR-AUC (average precision, reported for each class as positive), Brier score and confusion matrices.

* Each outer fold is reported individually, plus mean ± SD over the 5 folds (with the minimum fold) and **separately labelled pooled out-of-fold** metrics. Pooled AUC mixes five differently calibrated models, which will be noted.
* **Borehole-level:** accuracy for every test borehole; Macro-F1 and balanced accuracy only where the borehole contains both classes (102 of 120 in Downtown).
* **Depth bands** 0–5, 5–10 and 10–15 m; actual vs predicted stratigraphic columns for all test boreholes; maps of borehole-level error.
* **Distance sensitivity:** error against horizontal distance to the nearest training borehole in the same outer fold (Brenning 2023 style).
* **Uncertainty:** predicted probability, entropy, and spread across the five outer-fold refits; reliability diagrams.

**Pre-declared selection rule:** rank candidates by mean outer-fold Macro-F1 on the primary block size. A candidate is eligible only if mean Clay recall and mean Sand recall are both ≥ 0.50 and no fold has a class recall < 0.25. Within one standard error of the best, prefer lower fold SD, then lower Brier score. Scenario A and B winners are reported side by side. Because choosing among 10 candidates by outer score is itself optimistic (Cawley & Talbot 2010), the **external holdout is the unbiased check** of the chosen model. It is evaluated once, after the final refit (dev data, same inner-CV tuning) has been frozen.

### 5.8 3D output for ArcGIS Pro

* XY columns on a regular grid inside the Downtown polygon: 25 m proposed, giving 129,179 columns × 30 layers = 3.9 million cells. The 25 m grid oversamples the 30.5 m DEM, so horizontal ground detail is limited to about 30 m. A 30.51 m grid aligned to the DEM pixels is the alternative.
* `Top_Elevation_m = DEM(x, y) − depth`, sampled **bilinearly** from the supplied DEM exactly as for the boreholes. A different sampler would shift grid elevations by up to about 1 m relative to training.
* Two extrapolation flags: **horizontal**, for distance beyond the CV-validated range (§4.2); and **vertical**, for cell-top elevation outside the training range. About 0.4 % of cells in Scenario A and 0.16 % in Scenario B are vertical extrapolation, mostly in the low-lying south and south-west of Downtown.
* Deliverables:
  * a point table (CSV or file geodatabase) with X, Y, Z (cell centre elevation), depth, P(Clay), class, entropy, inter-model SD, distance to the nearest borehole, and an extrapolation flag (distance > the evaluated CV distance range);
  * a CF-compliant netCDF on a regular elevation grid (cells above ground or below 15 m set to NoData) for an ArcGIS Pro voxel layer.

---

## 6. Practical workflow and computational requirements

### 6.1 Repository layout

```
config/        project.toml (all settings and search spaces), external_holdout.lock.json (hash only)
data/raw/      received files (git-ignored); data/processed/, data/folds/ (git-ignored)
scripts/       01_audit_raw_data.py, 02_verify_sources.py
R/             00_install_packages.R, 01_spatial_autocorrelation.R, 02_blockcv_folds.R
notebooks/     01_preprocessing ... 05_final_model_external_3d (committed without outputs)
src/siemreap/  config, data, geo, folds, models, gw, metrics, nested_cv, grid3d, plots
tests/         unit tests on synthetic data
outputs/       per stage; restricted/, checkpoints, models and smoke/ are git-ignored
```

### 6.2 Software

* **R** ≥ 4.4 with blockCV **4.0-1 installed from the supplied zip** (Rtools on Windows), sf, terra, automap ≥ 1.1-20, gstat, ggplot2, cowplot, Rcpp, spdep, pinned with `renv`.
* **Python** 3.11–3.13 with pandas, numpy, scipy, scikit-learn, xgboost, optuna, joblib, matplotlib, pyproj, xarray/netCDF4, pinned in `environment.yml`.
* Optuna studies are stored in SQLite (resumable). Each outer fold × model × scenario × block size writes a checkpoint (fitted model, OOF predictions, trial table) and is skipped on rerun if present.

### 6.3 Smoke test before the full experiment

One outer fold, two inner folds, three trials per model, small forests. Outputs go to `outputs/smoke/` and are labelled `SMOKE TEST – NOT RESULTS` in file names and metadata. Automated tests:

* no borehole in both train and test;
* no external borehole in any training set;
* Python folds identical to the R CSVs;
* bit-identical predictions on rerun;
* GW fallback triggered on a synthetic case.

### 6.4 Compute

Measured timings and the full-run estimate are in §8.4.

---

## 7. Information and decisions needed before Stage 4

### Information

| # | Item | Status after the second delivery | Still needed |
|---|---|---|---|
| M1 | Raw Excel workbook | **Received and reconciled** (§1.7). It holds only sample intervals: no soil descriptions or continuous layer boundaries. Continuous layers exist only in the source boring-log PDFs. | See M18 |
| M2 | Downtown and 5 km polygons | **Received and verified.** CSV `Region` equals polygon membership for 189/189 boreholes. | Confirm the straight 12.3 km northern edge (a clip by a city AOI) is the intended study limit, and that the workbook's City/Buffer zones are retired. |
| M3 | DEM | **Received.** CSV ground elevations are bilinear samples of it (149/149 within coverage). | Product name and version, vertical datum, DSM or DTM, and how it was reprojected and clipped. Surveyed ground levels from the logs, if any, especially where the DEM sits 2–3.5 m above its surroundings at a borehole. |
| M4 | Coordinates of the 6 I04 boreholes | **Narrowed:** 2 are text-copy errors (UTM kept), 2 are outside the study extents (datum-conversion offset). | Correct easting for the two Downtown boreholes with easting-only differences (600 m and 60 m). |
| M5 | The two single 0–20.5 m and 0–17.5 m records | Unchanged (identical in the workbook). | Source-log check; treated as unlabelled meanwhile. |
| M10 | Prior use of these boreholes in your own modelling | Open. The boundary metadata refer to a project named "Siem_Reap_3D_V5". | Which boreholes, if any, were used before; determines the external-set wording. |
| M11 | Other research papers named in the brief | Open. | The papers. |
| M12 | Public repository | Open; raw data and borehole-level outputs are kept local. | Keep as is, or make the repository private? |
| M15 | Your machine and R version | Open. | Cores, RAM, R version. |
| M16 | **New:** source logs for the two power-line boreholes with depth conflicts (V1) | – | Which depth axis is correct; whether the ST "fines" values are real grading. |
| M17 | **New:** coordinate provenance for 5 Downtown boreholes (V2) | – | Where their UTM values came from. |
| M18 | **New (optional):** source boring-log PDFs generally | – | They would show whether SS symbols are visual log classifications (V4) and could supply continuous layer boundaries; neither is needed to proceed. |

### Decisions to approve

| # | Decision | My recommendation |
|---|---|---|
| D1 | Scenario membership from the polygons: Downtown (120) and the 0–5 km ring (25). The 10 km band and the boreholes beyond it are excluded from both scenarios. | Approve. |
| D2 | ML / CL-ML (5 km zone only, 11 records in 0–15 m). | Exclude in the primary analysis. As sensitivity, reclassify by the plasticity chart using lab data (14 of 21 plot as CL). |
| D3 | 0.5 m discretisation rule (§1.6): labels only from overlapping sample intervals, ≥ 50 % coverage, cell-top elevation. | Approve. |
| D4 | Records > 3 m are not measured strata. | Approve, pending M5. |
| D5 | The 43 logged-label vs lab-fines contradictions. | Keep the logged USCS in the primary analysis, flag them, and run a lab-relabelled sensitivity. The 10 conflicts from the V1 boreholes are covered by D11. |
| D6 | Depth-band prior as a non-competing reference baseline (§5.5). | Approve. |
| D7 | Square blocks (`hexagon = FALSE`), k = 5, random balanced selection, assignment balanced on the cell-level class. | Approve. |
| D8 | Block-size decision rule (§4.3). | Approve before Stage 4. |
| D9 | External holdout procedure (§5.6): 5 generalisation sites (≥ 500 m isolation) and 5 interpolation boreholes. | Approve, or set different numbers. |
| D10 | Selection rule and metric constraints (§5.7). | Approve. |
| D11 | **New:** the two boreholes with depth conflicts (V1). | Exclude from the primary analysis until M16 is answered; sensitivity with them included. |
| D12 | **New:** the four Downtown boreholes logged as contiguous 1.5 m D/U records (V3). | Include them: they are measured intervals, not extensions. Flag them, and run a sensitivity without them, because they are labelled almost continuously. |
| D13 | **New:** SS labels (V4). | Keep as logged in the primary analysis. Sensitivity with lab-supported labels only (samples with their own grading or Atterberg data). |
| D14 | **New:** source of truth. | Workbook master for record content (labels, depths, coordinates, lab data, text qualifiers). `Region` from the polygons, and ground elevation by bilinear sampling of the supplied DEM; both reproduce the CSV exactly. The CSV is kept as a verified cross-check. |
| D15 | **New:** prediction-distance validation (§4.2). | Add `cv_knndm` (blockCV) with the Downtown 25 m grid as the prediction domain, as an analysis beside the 100/200/300 m designs, and a validated-distance mask on the 3D map. |

Once M2–M4 and M16–M17 are resolved (or explicitly waived) and D1–D15 are approved, the next steps are, in order: freeze the preprocessing; select and hash the external boreholes; run the R variogram and blockCV analysis; then the smoke test.

---

## 8. Implementation and smoke test (2026-10-12)

You asked to set the Excel workbook aside and to implement the ML workflow with the settings of the research brief. The CSV is therefore the single data source (D14). The pipeline is `R/` + `notebooks/` + `src/siemreap/`, configured entirely from `config/project.toml`; the README gives the run order.

### 8.1 Settings applied

Where the brief fixes a setting, it is used as stated. Where it leaves a choice open, the recommendation of this report is the configured default and can be changed in one place:

| Decision | Default in `config/project.toml` |
|---|---|
| D1 Training extents | Downtown and the 0–5 km ring by the polygon-verified `Region` (120 and 25 boreholes before the external holdout) |
| D2 ML / CL-ML | Excluded (never converted to Clay) |
| D3 Discretisation | 0.5 m cells labelled only from overlapping sample intervals (≥ 50 % coverage, no conflict); `Top_Elevation_m` = ground − cell top |
| D4 Long records | Records > 3 m ignored |
| D5 Logged vs lab class | Logged USCS kept |
| D6 Reference | Depth-band prior reported beside the five models, excluded from selection |
| D7 Blocks | Square, k = 5, random balanced on the cell class, seed 20261012 |
| D8 Primary block size | **100 m as planned by the brief.** The rule's result is recorded in notebook 02 (§8.3) |
| D9 External holdout | 11 Downtown boreholes: 6 generalisation (5 isolated sites, 555–905 m from other boreholes) and 5 interpolation. Hash locked in `config/external_holdout.lock.json` |
| D10 Selection | Pre-declared rule of §5.7 |
| D11–D13 | Not applied as exclusions: training uses the CSV as received. The three boreholes with workbook depth or coordinate doubts are only barred from the external set, via a git-ignored flags file |
| D15 kNNDM | Available in `R/02_blockcv_folds.R` (`knndm_diagnostic`), off by default |

### 8.2 What the smoke test checked and fixed

The complete chain (notebook 01 → R scripts → notebooks 02–05) ran with reduced budgets: one outer fold, 100 m blocks, two Optuna trials, a 100 m 3D grid. Outputs went to the git-ignored `outputs/smoke/`. **None of its scores is a result.** It exposed and fixed three software issues:

1. **A borehole split by blockCV.** A Downtown borehole whose northing is a whole number lay exactly on a 100 m block edge, and `cv_spatial` assigned its cells to two folds. The R script now detects any split and shifts the blocks by 0.01 % of the block size (1 cm at 100 m) until none remains, recording the offset used (1 cm for all three sizes).
2. **Last-bit nondeterminism in RF predictions** from multi-threaded tree averaging (3 × 10⁻¹⁶). Prediction is now single-threaded, and a re-fitted fold reproduces its stored predictions bit for bit (checked in notebook 03).
3. **GWRF cost.** Local models are now fitted only for the anchors that a prediction actually uses. A unit test shows the predictions are bit-identical to fitting every anchor. This cut the per-trial cost about fivefold.

Further checks that passed:
* Python and R test-to-train distances agree to 0.01 m.
* Re-using the remaining outer folds as inner folds passes the adequacy check for all 30 outer-fold × block × scenario cases, so the R fallback folds were not needed.
* The external boreholes never enter any training set, and in smoke mode their labels are never read; notebook 05 uses a development stand-in.
* 9 unit tests on synthetic data pass.

**Sandbox limitation.** This environment's network policy blocks CRAN, so the R package `automap` could not be installed here. blockCV 4.0-1 was installed unmodified from the supplied source against an empty placeholder, never committed, so that `cv_spatial` and `cv_distance` could run; neither calls automap. `cv_spatial_autocor` therefore did not run here; its failure is recorded in the output table. On your machine `R/00_install_packages.R` installs the real automap from CRAN.

### 8.3 Primary block size: the pre-registered rule disagrees with the planned 100 m

Notebook 02 evaluates the rule of §4.3 from geometry and the response variograms only; no model score is involved. With the development boreholes:

* **r̃.** The median practical range over the ten 1.5 m slices is about 100 m. Five slices show no structure resolvable beyond the first 25 m lag.
* **Wasserstein distance.** *W* between test-to-train and prediction distances is 722 for 100 m blocks, 625 for 200 m and 624 for 300 m.
* **Rule outcome: 300 m.** 100 m blocks reach r̃ but leave *W* more than 10 % above the minimum.

These inputs come from the smoke run; the full run reproduces them except the subsampling interval. **Your decision:** keep 100 m as the brief plans (with this reason documented), or set `primary_block_m = 300` before notebook 03. All three sizes are run either way.

### 8.4 Measured cost and full-run estimate

Seconds per Optuna trial on one outer fold, full model sizes, 4 cores (each trial is four inner fits):

| | RF | XGB | MLP | GWRF | GWXGB |
|---|---|---|---|---|---|
| Scenario A | 8.7 | 0.7 | 9.0 | 20.6 | 1.4 |
| Scenario B | 8.8 | 0.7 | 39.3 | 21.0 | 2.8 |

With 50/50/50/30/30 trials × 5 folds × 2 scenarios, one block size takes about 6.6 h. The full experiment is about 23 h on 4 cores: three block sizes plus the purge runs, about 2.6 h. Most of the time goes to the MLP in Scenario B, RF and GWRF. Notebook 03 is resumable and can run one block size at a time (`RUN_BLOCKS`). Run the primary block size first.


## 9. Design change after advisor review (2026-10-13)

### 9.1 What changed

1. **A Gaussian process classifier (GP) is added** as a sixth ML model family (`GPModel` in `src/siemreap/models.py`, settings in `[gp]`).
2. **MLP, GWRF and GWXGB run on the primary block size only.** The robustness runs on 200 and 300 m use RF, XGB and GP plus the depth-only reference (`robustness_models` in `[experiment]`). The purge runs are unchanged (RF and XGB).

The brief names five model families run on all block sizes. Both points are deviations from it and are to be reported as such.

### 9.2 Why

The advisor questioned whether the chosen models suit the data. The audit supports that for three of them:

* **Data.** There are 109 Downtown development boreholes (1,896 cells, strongly correlated within a borehole), three coordinate predictors only, a horizontal range of about 100 m, and prediction distances well beyond it (median 746 m).
* **MLP.** It needs far more independent samples than this data provides.
* **GWRF and GWXGB.** They need dense neighbourhoods of boreholes, which most of the domain lacks.
* **Cost.** Together these three take about 90 % of the compute.

The GP models the spatial correlation directly. It is the machine-learning counterpart of indicator kriging: a latent Gaussian process with an anisotropic Matern kernel (ν = 1.5) and separate correlation lengths for Easting, Northing and elevation, fitted by maximum marginal likelihood (Laplace approximation; Rasmussen & Williams 2006) inside each training partition. It therefore needs no Optuna search; the inner folds only set its decision threshold. The student chose to stay within ML rather than add a geostatistical model.

### 9.3 What was known when the decision was made

The decision was made during the full run of notebook 03. At that point the only printed scores were the inner tuning scores of RF and XGB (five outer folds) and MLP (two outer folds), Scenario A, 100 m blocks. No outer-fold (test) score of any model had been computed, and the external holdout was untouched. The change rests on suitability and compute, not on scores.

### 9.4 Integrity of the running experiment

* **Existing checkpoints stay valid.** The run fingerprint of every existing model × scenario × block × purge combination is unchanged (all 108 checked), so finished folds are reused, not recomputed.
* **GP has its own fingerprint,** which includes the `[gp]` settings.
* **Tests.** Unit tests check that the GP is deterministic, that chunked prediction changes only floating-point rounding (≤ 10⁻⁹), and that the robustness subset is applied only off the primary block size.

### 9.5 Cost

* **One GP fit.** It takes about 65–70 s at 1,500–1,900 training cells on the 4-core development machine.
* **Per block size.** One GP run (5 outer folds × 5 fits) takes roughly 20–30 min per scenario.
* **3D grid.** If the GP becomes the final model, predicting the 25 m grid takes about 30–60 min.
* **Whole experiment.** It drops from about 23–26 h to roughly 12–13 h on the author's machine. The primary block size remains the largest share, mainly the MLP in Scenario B.

## 10. Design revision 2 (2026-10-13): gap filling, all six models on every block size

### 10.1 What changed

1. **Gap filling.** An unsampled gap between two measured cells of the same class is labelled with that class (`fill_gaps`, `fill_max_gap_m` in `[depth]`; `siemreap.data.fill_gaps`). A gap stays unlabelled when:
   * the classes above and below differ;
   * any record of any kind touches it: the other class, ML or CL-ML, a missing symbol, or an ignored long record;
   * it is longer than 3 m;
   * it lies above the first or below the last measured cell.

   Filled cells carry `measured = 0`. This deviates from the brief ("never … create synthetic labels for unknown intervals").
2. **All six models on every block size.** That is the brief's five (RF, XGB, MLP, GWRF, GWXGB) plus the GP of §9. The block-size subset of §9.1 (point 2) is withdrawn; `robustness_models` remains as an optional setting and is off.
3. **The experiment restarts from scratch.** The training data differ, so no checkpoint of the earlier run is valid, and the run fingerprints now include the measured/filled flag. Use a fresh project folder.

### 10.2 Why

This is the student's decision, on geological grounds: between two samples of the same class 1–1.5 m apart, the soil is very likely the same, and filling uses that information for training. Two checks on the development data support the assumption:

* **Filled cells.** The rule fills 819 cells in the Downtown development boreholes, against 1,896 measured cells (+43 %).
* **Agreement.** A measured cell whose nearest measured cells from other records above and below share a class has that same class in 99.0 % of the 1,546 such cells in the development boreholes, 99.8 % in the Downtown ones and 95 % in the 5 km ring ones (`data.fill_rule_check`, printed in notebook 01). Filled ring cells, used only in Scenario B, are therefore somewhat less certain.

### 10.3 Safeguards: filled cells are never scored

A filled cell is never test data. Only same-class gaps are filled, so scoring filled cells would weight the test toward easy cells inside thick layers and inflate every metric. Therefore:

* **Measured cells only:** the outer test folds, the inner validation scores, the decision thresholds and the external evaluation;
* **Measured cells only:** the variograms and the R blockCV folds, which are therefore identical to those of the design without filling;
* **All cells, measured and filled:** training, in both the inner and outer loops, and the final model.

Two unit tests cover the rule and the measured-only scoring. Because every reported score is on measured soil, the results stay comparable with a run without filling. Running the same notebooks in a separate folder with `fill_gaps = false` gives that comparison, if wanted.

### 10.4 Cost

The training sets grow by roughly 40–50 %, which slows every fit. The GP grows fastest, because its cost rises with the cube of the training size. The whole experiment (50 runs × 5 outer folds) is estimated at roughly 1.5–2.5 days on one 4-core PC. The MLP in Scenario B is the largest and least certain part. Notebook 03 is checkpointed. It can run the 100 m primary block first (`RUN_BLOCKS = [100]`), or be split across two computers by block size.
