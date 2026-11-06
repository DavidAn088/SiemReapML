# Siem Reap 3D Clay/Sand classification (V3): methods, results and defence notes

This document describes the V3 study from start to end: data, cells, spatial cross-validation, models,
model choice, external tests, explainable AI, the 3D map, reproducibility, figures, limitations and the
questions a reviewer or committee is likely to ask.

All numbers come from the full V3 run, confirmed **identical** by V3-clean (notebook 10, 12 of 12 checks).
Values marked **[fill in]** are produced only after running notebooks 05-09 with XGB as the final model;
copy them from those notebooks. Do not report anything from a quick test run (`quick = true`).

---

## 1. Aim and scope

* **Question:** can machine learning predict whether the soil at any point under Downtown Siem Reap,
  0-15 m deep, is **Clay** or **Sand**, from borehole logs alone?
* **Area:** Downtown Siem Reap (polygon `downtown.shp`, 80.7 km²).
* **Depth:** 0-15 m below the ground surface (DEM).
* **Target:** binary, **Clay = 1** (USCS CL, CH) and **Sand = 0** (USCS SC, SM, SP). Other symbols (ML, CL-ML)
  and missing symbols give no label; they are never converted.
* **Inputs:** Easting, Northing (UTM 48N, EPSG:32648) and the **top elevation of the cell** (ground elevation
  from the DEM minus depth). No depth input, no scaling (tree models do not need it; the MLP standardises
  inside each training set).

---

## 2. Data

| Item | Value |
|---|---|
| Borehole file | `SiemReap_BH_Master_Downtown_DEM_Region_v1_ArcGIS.csv` (SHA-256 checked) |
| Downtown boreholes | 120 |
| Development boreholes (used for CV and the final model) | 109 |
| External holdout boreholes (kept out before any modelling) | 11 |
| New external boreholes (received after V3 was frozen) | 10 (13 in the file; 3 were already in the main file and are removed) |

### 2.1 From sample records to 0.5 m cells
1. Every borehole is cut into **0.5 m cells** from 0 to 15 m (30 cells). The 0.5 m cell matches the median
   sample length (0.45 m).
2. A cell gets **Clay** or **Sand** when samples of that class cover **at least half** of it and **no sample of
   the other class** touches it. Records longer than 3 m are not used as labels.
3. **Gap filling:** an unsampled gap between two cells of the **same class** is filled with that class when the
   gap is **at most 3 m** and **no record lies inside it** (also an unclassified SS sample stops the fill).
   Nothing is filled above the first or below the last labelled cell.

| | Boreholes | Cells | Measured | Filled | Clay share |
|---|---|---|---|---|---|
| Development | 109 | 2,715 | 1,896 | 819 | 0.469 |
| External holdout | 11 | 266 | 168 | 98 | 0.417 |
| **Total** | **120** | **2,981** | **2,064** | **917** | |

Gap-filling counts: 935 gaps, 711 filled, 154 not filled (different class above and below), 70 not filled
(a record inside the gap).

**Filled cells** are used for training **and** testing (as V3 was designed); every score is also reported
on **measured cells only**. The training weight of filled cells (0.25, 0.5 or 1.0) is tuned in the inner CV.

### 2.2 External holdout (chosen before any modelling)
The 11 holdout boreholes were selected from coordinates and sampling metadata only (never class values):
6 whole isolated sites ("generalisation") and 5 boreholes from multi-borehole sites ("interpolation"),
spread over 3 spatial zones. The list is locked by a SHA-256 fingerprint (`config/external_holdout.lock.json`)
and checked every run.

---

## 3. Spatial cross-validation

### 3.1 Why spatial
Cells of one borehole, and boreholes close together, are similar. Random cross-validation would put
neighbours in training and testing and overstate the skill. Spatial blocks keep test boreholes away from
training boreholes (Roberts et al., 2017; Valavi et al., 2019).

### 3.2 Block size from the data (blockCV)
* Package: **blockCV 4.0-1** (Valavi et al., 2019) in R, with automap 1.1.20.
* `cv_spatial_autocor` was run on the Clay indicator in each **1.5 m depth slice** (10 slices); each gives an
  autocorrelation range.
* **Block size = median of the 10 ranges = 106 m.**

| Slice (m) | Range (m) |
|---|---|
| 0.0-1.5 | 104.9 |
| 1.5-3.0 | 107.4 |
| 3.0-4.5 | 116.1 |
| 4.5-6.0 | 88.4 |
| 6.0-7.5 | 102.7 |
| 7.5-9.0 | 67.6 |
| 9.0-10.5 | 83.2 |
| 10.5-12.0 | 113.3 |
| 12.0-13.5 | 224.3 |
| 13.5-15.0 | 181.0 |
| **Median** | **106.2 → 106 m** |

* Folds: `cv_spatial`, **k = 5**, square blocks of 106 m; 20-23 boreholes per fold, clay share 0.43-0.52.
* **Sensitivity:** blocks of 212 m and 318 m (2× and 3×).
* **Purge:** training boreholes closer than 106 m or 212 m to any test borehole removed (separate analysis).

### 3.3 Nested cross-validation
```
for each outer fold k = 1..5:
    test  = development boreholes in fold k
    train = the other 4 folds (minus purged boreholes in the purge analysis)
    inner CV = the 4 training folds (each predicted by a model trained on the other 3)
        Optuna (TPE, seeded) tries hyperparameter sets and the filled-cell weight
        each set: decision threshold = the one with the highest inner Macro-F1
    refit the best set on all training boreholes  ->  predict fold k
```
* Trials per outer fold: RF, XGB, MLP 50; GWRF, GWXGB 30.
* The test fold is never used for any choice (hyperparameters, threshold, filled-cell weight).
* All seeds are fixed, so every result is reproducible.

### 3.4 Runs
| Block | Purge | Models |
|---|---|---|
| 106 m (main) | none | RF, XGB, MLP, GWRF, GWXGB |
| 212 m, 318 m | none | RF, XGB |
| 106 m | 106 m, 212 m | RF, XGB |

65 outer folds in total (13 runs × 5 folds).

---

## 4. Models

| Model | Description |
|---|---|
| **RF** | Random Forest (Breiman, 2001) |
| **XGB** | XGBoost gradient-boosted trees (Chen & Guestrin, 2016) |
| **MLP** | Multilayer perceptron; inputs standardised on the training data; no early stopping (its random validation split would mix cells of one borehole) |
| **GWRF / GWXGB** | Geographically weighted RF / XGBoost, after Georganos et al. (2021) and Grekousis (2025), adapted to classification: every training borehole has a local model trained on its k nearest boreholes (k tuned; bisquare or uniform kernel); a cell uses the local model of its nearest borehole blended with a global model, P = α·P_local + (1 − α)·P_global (α tuned) |

Class imbalance: "balanced" class weights are a tuned option. Hyperparameter search spaces are in
`settings.toml` (`[space.*]`).

---

## 5. Metrics and model choice

### 5.1 Metrics
Macro-F1 (main), balanced accuracy, Clay and Sand recall, MCC, ROC-AUC, PR-AUC, Brier score; mean, SD and
minimum over the 5 outer folds; on all test cells and on measured cells only.

### 5.2 Results, main block size (106 m), all test cells
| Model | Macro-F1 | SD | Balanced acc. | Clay recall | Sand recall | ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|
| **XGB** | **0.644** | 0.095 | 0.647 | 0.594 | 0.700 | 0.701 | **0.217** |
| RF | 0.636 | 0.076 | 0.639 | 0.570 | 0.708 | 0.687 | 0.230 |
| GWXGB | 0.621 | 0.067 | 0.624 | 0.596 | 0.652 | 0.679 | 0.226 |
| GWRF | 0.608 | 0.068 | 0.613 | 0.588 | 0.637 | 0.667 | 0.251 |
| MLP | 0.567 | 0.097 | 0.580 | 0.574 | 0.585 | 0.633 | 0.267 |

Measured cells only: XGB 0.636, RF 0.632, GWXGB 0.622, GWRF 0.595, MLP 0.577 (Macro-F1), almost the same.

### 5.3 Pre-declared selection rule (one-standard-error rule)
1. Eligible: mean Clay and Sand recall ≥ 0.50, no fold below 0.25 (all five eligible).
2. Best = highest mean Macro-F1: **XGB, 0.644**. SE = SD/√5 = 0.095/√5 = **0.042**.
3. Tied = within one SE (≥ 0.602): **XGB, RF, GWXGB, GWRF**. Among them, the lowest fold SD, then the lowest
   Brier: **GWXGB** (SD 0.067).

(On measured cells only, the same rule would pick GWRF.)

### 5.4 Final model: XGB (documented override)
The rule selected **GWXGB**. **XGB** is used as the final model instead. Reason (`settings.toml → [final]`):
> Highest mean outer-fold Macro-F1 (0.644) and lowest Brier score; within one SE of the rule's choice
> (GWXGB), so not worse; a single global model is simpler to apply and to explain.

**Report this openly:** the rule's choice, the override and its reason. Because XGB and GWXGB are tied within
one SE, the choice does not change the conclusions. GWXGB is shown beside XGB in every figure.

Final XGB: tuned on all 109 development boreholes (inner groups = the 5 outer folds; 50 Optuna trials), then
trained on all of them. Inner-CV Macro-F1 0.662. **Decision threshold 0.58.** Hyperparameters: 250 trees,
learning rate 0.0238, max depth 3, min child weight 1.71, subsample 0.69, colsample by node 0.97,
L2 (lambda) 0.113, gamma 3.71, balanced class weights, filled-cell weight 1.0.

---

## 6. Distance, block size and purge

### 6.1 Score by distance to the nearest training borehole (Macro-F1, 106 m blocks)
| Distance | Test cells | XGB | RF | GWXGB | GWRF | MLP |
|---|---|---|---|---|---|---|
| 0-50 m | 246 | 0.879 | 0.950 | 0.902 | 0.943 | 0.673 |
| 50-100 m | 493 | 0.784 | 0.760 | 0.746 | 0.746 | 0.649 |
| 100-200 m | 861 | 0.602 | 0.571 | 0.610 | 0.603 | 0.563 |
| 200-300 m | 83 | 0.382 | 0.399 | 0.392 | 0.323 | 0.324 |
| 300-500 m | 543 | 0.479 | 0.538 | 0.448 | 0.517 | 0.561 |
| 500-1000 m | 330 | 0.634 | 0.533 | 0.497 | 0.376 | 0.491 |
| > 1000 m | 159 | 0.673 | 0.660 | 0.645 | 0.533 | 0.573 |

**Main finding:** the models are good close to boreholes (about 0.9 within 50 m, about 0.75 at 50-100 m) and
much weaker beyond about 200 m. The far bins hold few cells and few boreholes, so their differences are not
meaningful.

### 6.2 Larger blocks and purge (Macro-F1)
| Model | 106 m | 212 m | 318 m | Purge 106 m | Purge 212 m |
|---|---|---|---|---|---|
| XGB | 0.644 | 0.569 | 0.533 | 0.569 | 0.577 |
| RF | 0.636 | 0.558 | 0.500 | 0.534 | 0.559 |

Scores fall when the test boreholes are farther from the training boreholes, consistent with 6.1.

---

## 7. External tests

| Test | Boreholes | Cells |
|---|---|---|
| **A. V3 holdout** | 11 (kept out before modelling) | 266 (168 measured) |
| **B. New boreholes** | 10 (received after V3 was frozen; only ST samples are classified) | 84 (all measured) |
| **C. New boreholes, SS gaps filled** | the same 10; an unclassified SS sample between two samples of the same class takes that class (sensitivity, decided after B was seen) | 244 (84 measured) |

**GWXGB** (the rule's choice; scored once in V3):

| Test | Macro-F1 | Balanced acc. | ROC-AUC | Brier |
|---|---|---|---|---|
| A | 0.607 | 0.608 | 0.680 | 0.243 |
| B | 0.580 | 0.667 | 0.644 | 0.217 |
| C | 0.505 | 0.602 | 0.627 | 0.224 |

**XGB** (final model), obtained **after** GWXGB's external results were known (state this in the thesis):

| Test | Macro-F1 | Balanced acc. | ROC-AUC | Brier |
|---|---|---|---|---|
| A | 0.589 | 0.596 | 0.672 | 0.245 |
| B | 0.580 | 0.667 | 0.643 | 0.224 |
| C | 0.511 | 0.612 | 0.633 | 0.230 |

**Reading:** on the external boreholes XGB and GWXGB are tied (differences of 0.00-0.02 on 84-266 cells),
which agrees with the 1-SE result. XGB is **not** better externally; the override rests on "not worse" and
simplicity, not on better external scores. The two models are close because GWXGB's tuned blend weight is
α = 0.22: its predictions come mostly (78%) from its global XGBoost part.

Per borehole (Fig. 9b), accuracy ranges from 0.14 to 1.00 and does not fall smoothly with distance: with
21 boreholes, local geology at single boreholes dominates. Describe it as large variation between
boreholes, not as a trend.

The new boreholes are 195-1,217 m from the nearest training borehole; per-borehole accuracy falls with
distance (Fig. 9b).

---

## 8. Explainable AI (held-out outer folds, 106 m)

* **Permutation importance:** drop in balanced accuracy when an input is shuffled (10 repeats); single
  inputs and groups (location = Easting + Northing together; elevation).
* **SHAP:** exact Shapley values (Lundberg & Lee, 2017) of P(Clay) for 300 test cells per fold, background of
  100 training cells of that fold.

| Model | Location (E + N) | Elevation | Easting | Northing |
|---|---|---|---|---|
| XGB | 0.107 | 0.069 | 0.012 | 0.080 |
| RF | 0.099 | 0.081 | 0.026 | 0.087 |
| GWXGB | 0.080 | 0.070 | 0.007 | 0.055 |
| GWRF | 0.063 | 0.074 | 0.008 | 0.059 |
| MLP | 0.039 | 0.063 | 0.017 | 0.039 |

(Permutation importance, all test cells.) Mean |SHAP| (XGB): Easting 0.069, Northing 0.104, Elevation 0.111.
**Reading:** location (mainly Northing) and elevation matter about equally; Easting matters little.

---

## 9. 3D map

* Map model: the final model type, hyperparameters and threshold, **refitted on all 120 Downtown boreholes**
  (109 development + 11 holdout). It is not evaluated; its expected skill is that of the CV and the external tests.
* Grid: **50 m** columns inside Downtown (32,280 columns × 30 cells = 968,400 cells, 0-15 m).
* Each cell: P(Clay), class at the final threshold, distance to the nearest of the 120 boreholes, reliability
  zone, vertical-extrapolation flag.

| Reliability zone | Distance to nearest borehole | Share of the map |
|---|---|---|
| 1 | ≤ 100 m | 3.1% |
| 2 | 100-300 m | 15.3% |
| 3 | > 300 m | **81.6%** |

Predicted clay share: **0.369** with XGB (0.407 with GWXGB); **91.4%** of the 968,400 cells get the same class
from both models, so the choice of final model changes the map only slightly. Vertical extrapolation: 0.4% of cells. Output: CSV points for ArcGIS Pro
(*XY Table To Point*, Z field) and a netCDF voxel file.

**Use of the map:** detailed only in zones 1-2; in zone 3 it shows the general depth and area trend.

---

## 10. Reproducibility

* **V3-clean** recomputes V3 with readable code (one `settings.toml`, one `helpers.py`, 10 notebooks) and
  reads V3's blockCV folds and holdout list. Notebook 10 compares every result with V3:
  cells, 65/65 outer folds, all scores, the selection, the final model, 250 permutation values,
  7,500 SHAP values, 3 external tests and 968,400 map cells: **all identical**.
* With the XGB override, notebook 10 reports the final model, its external tests and the map as
  **changed on purpose** and everything else as identical.
* Raw files are pinned by SHA-256; seeds are fixed; the Python environment is the same as V3's.
* Keep: the run notebooks with their outputs, `settings.toml`, and the package versions.

---

## 11. Figures (notebook 09)

PDF (vector) and PNG (600 dpi); 90 mm or 190 mm wide; * = final model (XGB), † = rule's choice (GWXGB).

| Fig. | Content |
|---|---|
| 1 | Study area and boreholes (private: shows locations) |
| 2 | Clay share by depth |
| 3 | blockCV range per slice, folds, purge example (private) |
| 4 | Model comparison: Macro-F1 and balanced accuracy, mean ± SD |
| 5 | Macro-F1 vs distance to the nearest training borehole |
| 6 | Confusion matrix and mean spatial ROC curve |
| 7 | SHAP importance, permutation importance, SHAP values |
| 8 | Elevation: observed clay share vs elevation effect |
| 9 | External tests A, B, C: XGB vs GWXGB |
| 9b | Accuracy of each external borehole vs distance |
| 10 | Soil columns of three test boreholes chosen by rule (near, median, far) |
| 11 | 3D map: P(Clay) at four depths with reliability zones (private) |
| S1 | Block-size sensitivity and purge |
| S2 | Soil columns of all 21 external boreholes |

Borehole names are hidden (BH-A…, EX-01…). Figures showing locations stay private until the data owners agree.

---

## 12. Limitations

1. **Inputs are coordinates and elevation only.** The models can copy nearby boreholes but cannot explain why
   clay or sand is present; skill falls with distance (Section 6.1).
2. **Sparse boreholes:** 82% of the map is more than 300 m from a borehole.
3. **Five folds:** fold-to-fold variation is large (SD 0.07-0.10); models within one SE cannot be ranked.
4. **Final-model override:** XGB replaced the rule's GWXGB after the results were known; XGB's external scores
   are therefore not fully independent.
5. **Gap-filled cells** are inferred, not measured; scores on measured cells only are nearly the same.
6. **New boreholes:** only ST samples are classified in the logs; test C uses an assumption decided after
   test B was seen.
7. **Binary target:** Clay vs Sand only; silts (ML, CL-ML) are excluded.

**Future work:** inputs that carry geology (relative elevation from the DEM, distance to the Siem Reap River
and old channels, geological or landform units; McBratney et al., 2003), and boreholes in the gaps more than
300 m from existing ones.

---

## 13. Questions a reviewer or committee is likely to ask

| Question | Answer |
|---|---|
| Why XGB when your rule chose GWXGB? | Tied within one SE; reason documented (5.4); GWXGB reported beside it |
| Are XGB's external tests independent? | Not fully: scored after GWXGB's results were known; stated in the thesis |
| Macro-F1 0.64 is low. | Strict spatial test; skill depends on distance (about 0.9 within 50 m); random CV would score higher only through leakage |
| Do filled cells inflate the scores? | No: measured-only scores are almost the same (XGB 0.636 vs 0.644) |
| Why only X, Y, Z? | Common in borehole studies; geology inputs proposed as future work |
| Why 106 m blocks? | blockCV autocorrelation range (median of 10 slices); sensitivity at 212 and 318 m |
| Easting and Northing are correlated: is SHAP misleading? | Grouped permutation importance of location (E + N) is reported too |
| Is the map reliable everywhere? | No: reliability zones; 82% in zone 3 |
| Why five models, why not KNN or SVM? | They cover trees, boosting, neural networks and local models and score alike; the limit is the information in the inputs, not the algorithm |
| Can we see the code and data? | Code: public repository; data: confidential, on request with the data owners' permission |

---

## 14. Checklist before submission or defence

- [x] Notebooks 05, 07, 09 run with XGB; values copied into this document
- [x] Notebook 08 map clay share copied (Section 9)
- [x] Notebook 10 ends with "EVERYTHING IDENTICAL TO V3, except the items changed on purpose"
- [ ] Advisor agrees with the override reason
- [ ] Limitations section written (Section 12)
- [ ] A lab mate re-runs the notebooks on another PC from the README
- [ ] Mock defence on the questions in Section 13
- [ ] Journal figure and data-availability rules checked
- [ ] Data owners' permission for figures that show borehole locations

---

## References

* Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32.
* Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. KDD 2016*, 785-794.
* Georganos, S., et al. (2021). Geographical random forests. *Geocarto International*, 36, 121-136.
* Grekousis, G. (2025). Geographical-XGBoost. *Journal of Geographical Systems*, 27, 169-195.
* Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
* Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
* McBratney, A. B., Mendonça Santos, M. L., & Minasny, B. (2003). On digital soil mapping. *Geoderma*, 117, 3-52.
* Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or
  phylogenetic structure. *Ecography*, 40, 913-929.
* Schratz, P., et al. (2019). Hyperparameter tuning and performance assessment of statistical and
  machine-learning algorithms using spatial data. *Ecological Modelling*, 406, 109-120.
* Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019). blockCV: An R package for generating
  spatially or environmentally separated folds for k-fold cross-validation of species distribution models.
  *Methods in Ecology and Evolution*, 10, 225-232.

Check every reference against the original before submission.
