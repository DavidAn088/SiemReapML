# Stage 1–3 report: source review, data audit and experimental design

**Project:** 3D Clay/Sand classification of the upper 15 m in Downtown Siem Reap using machine learning and spatial cross-validation
**Date:** 2026-10-12  **Status:** Stages 1–2 complete; Stage 3 design proposed and awaiting your review.

---

## 0. What has and has not been done

| Stage | Status | Evidence |
|---|---|---|
| 1 Source review | Done for the files received (one CSV, one paper, blockCV source zip). | §1, §2 |
| 2 Data audit | Done. Reproducible script, aggregate tables and figures. | `scripts/01_audit_raw_data.py`, `outputs/01_audit/aggregate/` |
| 3 Methodological design | Proposed (this document). **Not frozen**: it needs your decisions in §7. | §3–§6 |
| 4 R blockCV / variograms | **Not started.** I held off on purpose: the external holdout boreholes have to be chosen first (§5.6), and variograms and fold balancing are label-dependent design steps. | – |
| 5–7 Modelling, results, reporting | **Not started.** No model has been fitted, tuned or scored. | – |

There is no earlier experiment, saved prediction or model score in this project. The only label statistics computed so far are the descriptive audit counts in §1. They cover all boreholes, including the ones that will later be reserved for external evaluation. §5.6 explains why the holdout selection does not use them.

The repository is **public**, so the raw CSV and every borehole-level output (inventory, issue log with source line numbers, borehole map) are kept out of git. They live under `data/raw/` and `outputs/01_audit/restricted/`. Only aggregate statistics are committed.

---

## 1. Verified summary of the dataset

### 1.1 What was received

| Item | Finding |
|---|---|
| File | `SiemReap_BH_Master_Downtown_DEM_Region_v1_ArcGIS.csv` (UTF-8 with BOM), SHA-256 `caf59d0b…0f72`. **This is a CSV export, not the raw Excel workbook named in the brief.** The file name and contents show processing in ArcGIS: a `Region` field, DEM-derived ground elevations, and computed elevations. |
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
* **No boundary geometry is included.** `Region` is a label only, so I cannot verify how it was assigned, what "Downtown_5km" means (a 0–5 km ring outside the Downtown boundary?), or where the 3D prediction domain ends. See §7, M2.

### 1.3 Depths and elevations

* `Top_Elevation_m = Ground_Elevation_m − Depth_From_m` holds exactly for all 4,269 records, and the same check passes for the bottom and mid elevations. The convention is depth below ground, positive downwards.
* **Ground elevations are DEM-derived.** 187 of 189 values carry float32-style decimals (e.g. 40.144863128662109), which is typical of raster extraction, not of surveyed levels. Two boreholes outside the study regions have exactly 1.5 m, which is implausible here and looks like a different source or a placeholder. Because the fixed predictor `Top_Elevation_m` inherits any DEM error, the DEM's product, resolution, vertical datum and DSM/DTM nature matter (M3).
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
| I04 | Text coordinate vs UTM field mismatch > 5 m: 54, 60 and **600 m** for three Downtown boreholes, 64 m (5 km), 452 m (10 km, two boreholes). | 6 / 6 | Needs the source report (M4). These boreholes are barred from the external set until resolved. |
| I05 | Coordinate text empty (UTM present). | 4 / 4 | Informational. |
| I06 | USCS missing. | 161 / 19 | Unlabelled. |
| I07 | USCS outside the binary map (ML, CL-ML). | 26 / 4 | D2. |
| I08 | Logged group contradicts lab fines. | 43 / 17 | D5. |
| I11 | Ground elevation lacks DEM precision (1.5 m). | 2 / 2 | Outside both scenarios. |
| I13 | Borehole shorter than 15 m. | 8 / 8 | Use the measured part only. |
| I14 | Non-standard `Sample_Type` text (`SS-1`, `-`, blank). | 21 / 5 | Cosmetic. |
| I15 | `Source_File` empty. | 16 / 2 | 10 km zone only; outside both scenarios. |

No duplicate records, zero or negative interval lengths, within-borehole coordinate changes or elevation-formula errors were found.

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

The discretisation rule is **not frozen** (D3). Nothing in the preview is a modelling result.

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
| `cv_knndm` / `cv_distance` | Yes | Matched to the prediction domain | Map-like prediction distances | **Diagnostic to justify block size**; needs boundary (M2) |

### 4.2 What the candidate block sizes imply here

Geometry-only planning diagnostic: a square grid anchored at the data minimum, **not blockCV and not used for folds**.

| Block | Occupied blocks | Boreholes with a borehole in another block within 25 m | … within 100 m | Median distance to nearest borehole in another block |
|---|---|---|---|---|
| 100 m | 88 | 6 | 38 | 149 m |
| 200 m | 75 | 2 | 22 | 182 m |
| 300 m | 68 | 0 | 13 | 319 m |

With five random folds, about 4 in 5 of those other-block neighbours land in training. So under **100 m blocks a substantial share of test boreholes will have a training borehole within tens of metres**: 100 m CV largely measures interpolation inside site clusters. At 300 m, no test borehole can have a training neighbour closer than about 50 m, and the median separation is around 300 m. Meanwhile the 3D map has to predict across a 9 × 7 km area where most voxels are probably hundreds of metres from any borehole. That last point needs the boundary to quantify.

### 4.3 Recommended rule for the primary block size (fixed before any model is fitted)

The brief plans 100 m as primary, subject to review, and forbids choosing it by score. I propose writing this rule into the configuration before Stage 4, using only development-set geometry and the response variograms:

1. Compute *r̃*: the median over depth slices of the practical range *r*₉₅ of the detrended indicator variograms (§3.2), with a site-bootstrap interval.
2. Compute the prediction-domain distribution: horizontal distance from every 3D-grid column in the Downtown polygon to the nearest development borehole.
3. For each of 100, 200 and 300 m, generate the blockCV folds and compute `cv_distance`: the test-to-nearest-train distance distribution and its Wasserstein distance *W* to the prediction distribution.
4. **Primary = 100 m** if 100 m ≥ *r̃* and *W*(100) is within 10 % of the smallest *W*. Otherwise primary = the candidate ≥ *r̃* with the smallest *W*. If no candidate reaches *r̃*, primary = 300 m and the limitation is reported. The other two sizes are always reported as robustness analyses.

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

1. **Eligible:** Downtown boreholes reaching ≥ 15 m, with ≥ 10 sampled intervals in 0–15 m that carry any USCS symbol (presence, not class), and no unresolved coordinate issue (I04).
2. **Sites:** single-linkage clusters at 100 m (coordinates only).
3. **Generalisation group (5 boreholes):** whole sites whose nearest borehole outside the site is ≥ 300 m away. Downtown is split into 3 spatial zones by seeded k-means on coordinates, and sites are sampled across zones with a fixed seed. These test **geographically unfamiliar sites**.
4. **Interpolation group (5 boreholes):** one randomly chosen borehole from each of 5 sites with ≥ 3 boreholes, sampled across zones. Its site neighbours (< 100 m) stay in development. These test **interpolation near existing boreholes**.
5. The sorted ID list is written to `outputs/…/restricted/`, and its **SHA-256 is committed** to the public repository. That fixes the list permanently without publishing borehole identities, and shows it was never changed after scores existed.

Disclosure for the thesis: the external boreholes were selected after a descriptive audit of all labels (§1) but before any modelling or validation design, and the selection rule used no class information. If any of these boreholes were used in **your own earlier modelling** outside this project, they must be called *retrospective holdouts* (M10).

### 5.7 Metrics and reporting

For every model × scenario × block size: Accuracy, balanced accuracy, Macro-F1, Clay recall, Sand recall, MCC, ROC-AUC, PR-AUC (average precision, reported for each class as positive), Brier score and confusion matrices.

* Each outer fold is reported individually, plus mean ± SD over the 5 folds (with the minimum fold) and **separately labelled pooled out-of-fold** metrics. Pooled AUC mixes five differently calibrated models, which will be noted.
* **Borehole-level:** accuracy for every test borehole; Macro-F1 and balanced accuracy only where the borehole contains both classes (102 of 120 in Downtown).
* **Depth bands** 0–5, 5–10 and 10–15 m; actual vs predicted stratigraphic columns for all test boreholes; maps of borehole-level error.
* **Distance sensitivity:** error against horizontal distance to the nearest training borehole in the same outer fold (Brenning 2023 style).
* **Uncertainty:** predicted probability, entropy, and spread across the five outer-fold refits; reliability diagrams.

**Pre-declared selection rule:** rank candidates by mean outer-fold Macro-F1 on the primary block size. A candidate is eligible only if mean Clay recall and mean Sand recall are both ≥ 0.50 and no fold has a class recall < 0.25. Within one standard error of the best, prefer lower fold SD, then lower Brier score. Scenario A and B winners are reported side by side. Because choosing among 10 candidates by outer score is itself optimistic (Cawley & Talbot 2010), the **external holdout is the unbiased check** of the chosen model. It is evaluated once, after the final refit (dev data, same inner-CV tuning) has been frozen.

### 5.8 3D output for ArcGIS Pro

* XY columns on a regular grid inside the Downtown polygon (25 m proposed), with 0.5 m layers to 15 m below the DEM surface.
* `Top_Elevation_m = DEM(x, y) − depth`, so the **same DEM** used for the boreholes is required (M3).
* Deliverables:
  * a point table (CSV or file geodatabase) with X, Y, Z (cell centre elevation), depth, P(Clay), class, entropy, inter-model SD, distance to the nearest borehole, and an extrapolation flag (distance > the evaluated CV distance range);
  * a CF-compliant netCDF on a regular elevation grid (cells above ground or below 15 m set to NoData) for an ArcGIS Pro voxel layer.

---

## 6. Practical workflow and computational requirements

### 6.1 Repository layout (planned)

```
config/        project.toml (constants), search spaces, block-size decision, holdout hash
data/raw/      received files (git-ignored)
scripts/       01_audit (done), 02_preprocess, 03_select_external
R/             10_variograms.R, 11_blockcv_folds.R (blockCV 4.0-1, renv.lock)
notebooks/     preprocessing, five-model nested CV, results (Jupyter)
src/           shared Python: fold import, models, GW wrappers, metrics
outputs/       per stage; restricted/ subfolders git-ignored
tests/         fold integrity, leakage, determinism, R/Python fold equality
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

### 6.4 Compute estimate (to be replaced by measured smoke-test timings)

Data are small: about 1.9k development cells in Scenario A and 2.3k in B. Assumptions on a typical 8-core laptop: RF/XGB fit ≈ 0.2–0.5 s, MLP ≈ 1–3 s, GW fit ≈ 110–135 local fits ≈ 10–20 s.

| Component | Fits per model, scenario, block size | Estimated wall time (8 cores) |
|---|---|---|
| RF, XGB | 5 × 4 × 50 = 1,000 | 5–15 min each |
| MLP | 1,000 | 0.5–1.5 h |
| GWRF, GWXGB | 5 × 4 × 30 = 600 | 1–3 h each |
| One block size, both scenarios, all 5 models | – | **≈ 6–15 h** |
| Robustness (two more block sizes) + purge | – | ≈ 2× the primary |
| R variograms and blockCV | – | minutes |
| 3D prediction (≈ 10⁵ columns × 30 layers) | – | minutes (global), < 1 h (GW) |

Total: roughly 1–2 days of unattended computing. Memory is not a constraint.

---

## 7. Information and decisions needed before Stage 4

### Information to supply

| # | Item | Why it blocks |
|---|---|---|
| M1 | **The raw Excel workbook** (only the derived CSV `…v1_ArcGIS.csv` was received). Does it contain continuous stratum descriptions (layer boundaries) as well as samples? | Traceability to original records. If continuous logs exist, far more 0.5 m cells could be labelled legitimately. |
| M2 | **Downtown boundary and 5 km buffer polygons** (shapefile/GDB, CRS) and the definition of each `Region` value. | Verifies the `Region` labels, defines the prediction domain, and is needed for the block-size rule (§4.3) and the 3D grid. |
| M3 | **The DEM** used for `Ground_Elevation_m`: product, resolution, vertical datum, DSM or DTM. Do the logs report surveyed ground levels? | `Top_Elevation_m` and the 3D grid depend on it. |
| M4 | Correct coordinates for the 6 boreholes with text/UTM mismatches (3 in Downtown, up to 600 m). | Location errors corrupt blocking and distances. |
| M5 | Source-log check of the two single 0–20.5 m and 0–17.5 m records. | Currently treated as unlabelled. |
| M10 | Have any of these boreholes been used in your own earlier modelling? | Determines whether the external set can be called independent. |
| M11 | The other research papers referred to in the brief. | Only the blockCV paper was received. |
| M12 | The repository is **public**. Keep raw data and borehole-level outputs local (current set-up), or make the repository private and commit them? | Data confidentiality. |
| M15 | Your machine (cores, RAM) and R version. | Compute planning and environment pinning. |

### Decisions to approve

| # | Decision | My recommendation |
|---|---|---|
| D1 | Scenario membership: Downtown (120) and Downtown_5km (25). The 10 km zone and unassigned boreholes are excluded from both scenarios. | Approve. |
| D2 | ML / CL-ML (5 km zone only, 11 records in 0–15 m). | Exclude in the primary analysis. As sensitivity, reclassify by the plasticity chart using lab data (14 of 21 plot as CL). |
| D3 | 0.5 m discretisation rule (§1.6): labels only from overlapping sample intervals, ≥ 50 % coverage, cell-top elevation. | Approve. |
| D4 | Records > 3 m are not measured strata. | Approve, pending M5. |
| D5 | The 43 logged-label vs lab-fines contradictions. | Keep the logged USCS (the source classification) in the primary analysis, flag them, and run a lab-relabelled sensitivity. |
| D6 | Depth-band prior as a non-competing reference baseline (§5.5). | Approve. |
| D7 | Square blocks (`hexagon = FALSE`), k = 5, random balanced selection, assignment balanced on the cell-level class. | Approve. |
| D8 | Block-size decision rule (§4.3). | Approve before Stage 4. |
| D9 | External holdout procedure (§5.6): 5 generalisation + 5 interpolation boreholes. | Approve, or set different numbers. |
| D10 | Selection rule and metric constraints (§5.7). | Approve. |

Once M1–M4 are resolved (or explicitly waived) and D1–D10 are approved, the next steps are, in order: freeze the preprocessing; select and hash the external boreholes; run the R variogram and blockCV analysis; then the smoke test.
