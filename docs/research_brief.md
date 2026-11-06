# MASTER PROMPT — INDEPENDENT RESEARCH PROJECT

## 3D Subsoil Classification Using Machine Learning and Spatial Cross-Validation in Siem Reap, Cambodia

### 1. Role and research independence

Act as an experienced geotechnical engineering researcher, machine-learning scientist, and spatial-statistics specialist with expertise in subsurface geological modelling, spatial cross-validation, R, Python, and scientific publication.

I am conducting a master's thesis in Geotechnical Engineering at Chulalongkorn University, Thailand.

My research focuses on three-dimensional subsurface soil classification using machine learning in Downtown Siem Reap, Cambodia.

**Treat this as a completely new research project.**

Use only the following sources:

- The raw borehole Excel workbook I upload.
- The research papers I provide.
- The GitHub repository I specify.
- Additional peer-reviewed scientific literature and official software documentation that you independently verify.
- Explicit methodological requirements written in this prompt.

Do not assume the existence of earlier experiments, notebooks, saved predictions, previous model scores, or methodological decisions beyond those explicitly stated here.

Do not invent historical results, fill data gaps with assumed values, or infer information from unrelated conversations or projects.

I want an independent, critically reasoned scientific investigation, not a reproduction of another AI's conclusions.

### 2. Research objectives

Develop a robust, reproducible ML framework for classifying the subsurface into Clay and Sand within the upper 15 m of Downtown Siem Reap.

The research must:

1. Compare five machine-learning models under consistent experimental conditions.
2. Evaluate two geographical training extents: Downtown only and Downtown + 5 km.
3. Investigate spatial autocorrelation using the original R `blockCV` package.
4. Implement nested spatial cross-validation.
5. Examine model performance on unseen Downtown boreholes.
6. Investigate uncertainty, depth-dependent prediction quality, and spatial transferability.
7. Generate a 3D Clay/Sand classification and probability dataset suitable for ArcGIS Pro.
8. Produce reproducible results suitable for a master's thesis and a potential peer-reviewed journal article.

Predictive accuracy matters, but scientific validity is essential. Aim for strong performance, including Macro-F1 above 0.60 if achievable, without manipulating validation procedures to achieve a target score.

### 3. Research materials

I will provide:

**Dataset:** Raw borehole Excel workbook, containing geotechnical investigation records.

**GitHub repository:** https://github.com/rvalavi/blockCV

**Research papers:** Uploaded publications discussing machine-learning soil classification, spatial modelling, geographical validation, and three-dimensional subsurface prediction.

Read these resources before implementing models.

Inspect the Excel workbook thoroughly. Determine the actual number of boreholes, geographical distribution, recorded depths, available soil classifications, and data completeness.

Identify relevant fields, which may include:

- `BH_Code`
- `UTM_Easting_m`
- `UTM_Northing_m`
- `Ground_Elevation_m`
- `Depth_From_m`
- `Depth_To_m`
- `Top_Elevation_m`
- `USCS`
- `Region`

Do not assume the field names, dataset size, or classification counts until verified.

Use the WGS 84 / UTM Zone 48N coordinate system (EPSG:32648), subject to confirmation from the data.

If the workbook does not contain a reliable Downtown boundary or the information needed to define the 5 km buffer, request the necessary GIS boundary rather than creating an arbitrary boundary.

### 4. Soil classification and preprocessing

Study depth: **0–15 m below ground surface**.

Vertical resolution: **0.5 m**.

Binary soil classification:

| USCS classification | Target |
|---|---|
| CL | Clay = 1 |
| CH | Clay = 1 |
| SC | Sand = 0 |
| SM | Sand = 0 |
| SP | Sand = 0 |

Review other USCS symbols and ambiguous classifications individually.

Do not automatically convert ML or CL-ML into Clay.

Only generate 0.5 m classification observations within measured borehole strata. Never extend a recorded soil layer beyond the actual borehole log or create synthetic labels for unknown intervals.

Preserve borehole identities throughout preprocessing.

Audit overlapping intervals, missing values, duplicate records, inconsistent coordinates, elevation calculations, and soil classification errors.

Ensure that all processed observations can be traced to the original Excel records.

### 5. Fixed machine-learning inputs

Use exactly three predictor variables:

1. `UTM_Easting_m`
2. `UTM_Northing_m`
3. `Top_Elevation_m`

Where appropriate:

`Top_Elevation_m = Ground_Elevation_m - Depth_From_m`

Verify the actual elevation and depth conventions before calculating.

Do not introduce additional predictors without explicit permission.

Any scaling, imputation, class weighting, feature transformations, or probability calibration must be fitted using training data only.

### 6. Machine-learning models

Evaluate exactly five model families:

1. Random Forest (RF)
2. XGBoost (XGB)
3. Multilayer Perceptron (MLP)
4. Geographically Weighted Random Forest (GWRF)
5. Geographically Weighted XGBoost (GWXGB)

Use credible published formulations for geographically weighted models.

Carefully define geographical neighborhoods, distance weighting, local model fitting, minimum samples, fallback methods, and computational complexity.

Predictions must be deterministic when random seeds and inputs remain unchanged.

Use Optuna for hyperparameter tuning, with clearly documented search spaces and budgets.

All five models must be evaluated against identical outer validation folds and comparable performance metrics.

### 7. Spatial autocorrelation and original R blockCV

**The spatial cross-validation must use the original R `blockCV` package from the supplied GitHub repository.**

Do not silently recreate `blockCV` with an approximate Python implementation.

Use R for spatial autocorrelation and spatial-fold construction.

Use Python for machine-learning training, tuning, prediction, and evaluation.

First, investigate the spatial dependence of the target soil classes, including appropriate depth-specific analyses.

Calculate and interpret variogram models and spatial autocorrelation ranges. Explain the distinction between variogram parameters and effective spatial correlation distances.

Evaluate the following candidate spatial block sizes:

- 100 m
- 200 m
- 300 m

Use 100 m as the planned primary experiment, subject to methodological review before model fitting.

Treat 200 m and 300 m as spatial robustness analyses.

Justify the primary choice using actual data, spatial autocorrelation, borehole distribution, and scientific literature—not predictive scores.

Use `blockCV` to generate spatial fold assignments and export them to CSV.

Retain the R scripts, fold maps, autocorrelation plots, and borehole-to-fold assignments.

**Python must import and use the actual R-generated fold assignments.**

Do not regenerate the outer spatial folds in Python.

Check the actual nearest-training-borehole distances in each fold. A spatial block size does not automatically guarantee an equivalent minimum geographical exclusion distance.

### 8. Training extent comparison

Compare two scenarios.

**Scenario A: Downtown-only training**

Train models using eligible Downtown boreholes.

Test exclusively on held-out Downtown boreholes.

**Scenario B: Downtown + 5 km training**

Train models using eligible boreholes from Downtown and its surrounding 5 km buffer.

Test on exactly the same held-out Downtown boreholes used in Scenario A.

Use the same original R spatial-fold assignments and the same evaluation procedure for both scenarios.

All samples from a borehole must remain in the same geographical validation group.

Exclude any borehole assigned to a held-out fold from the corresponding training partition, including surrounding-buffer boreholes where applicable.

Determine whether expanding the training extent improves or reduces Downtown prediction performance.

### 9. Nested spatial cross-validation

Implement properly nested spatial cross-validation.

**Outer CV**

Five spatial folds generated by the original R `blockCV` package.

The outer folds provide held-out Downtown predictions.

**Inner CV**

Spatially separated inner folds constructed using only the eligible outer-training observations.

Where valid, reuse the four eligible training-fold groups from the original R assignments.

Evaluate whether this grouping provides adequate spatial and class representation. If not, propose and justify a valid nested alternative before training.

The inner folds must control hyperparameter optimization, preprocessing choices, probability thresholds, and model selection.

Do not use outer-held-out labels during tuning.

Never split rows randomly if observations belonging to the same borehole could appear in training and validation simultaneously.

Do not use conventional random-row 80/20 or random five-fold cross-validation as the principal evaluation method.

Evaluate any spatial purge separately, with consideration of the limited number of available boreholes.

### 10. External evaluation

Reserve a small set of Downtown boreholes for external assessment before training and tuning.

Propose a scientifically justified selection procedure based on the data distribution and project requirements, without inspecting model performance.

Record the selected borehole IDs permanently.

Exclude all their labels from modelling and data-dependent validation design.

Never select or change external boreholes in response to model scores.

Evaluate them only after development-based model selection and final model freezing.

If the reserved data have previously been used for modelling or methodological exploration, disclose that limitation and describe them as retrospective holdouts rather than genuinely independent new observations.

Distinguish interpolation near existing training boreholes from generalization to geographically unfamiliar sites.

### 11. Evaluation and reliability analysis

Report:

- Accuracy
- Balanced accuracy
- Macro-F1
- Clay recall
- Sand recall
- Matthews correlation coefficient
- ROC-AUC
- PR-AUC, where appropriate
- Brier score
- Confusion matrices

For every model, provide:

- Individual outer-fold results
- Mean and standard deviation across spatial folds
- Pooled out-of-fold metrics, separately labeled
- Borehole-level accuracy and Macro-F1
- Depth-specific performance for 0–5 m, 5–10 m, and 10–15 m
- Actual-versus-predicted borehole stratigraphic profiles
- Spatial distribution of errors
- Model confidence and uncertainty
- Sensitivity to distance from training boreholes

Compare the two training extents under identical evaluation conditions.

Do not conceal poor-performing folds or soil classes.

A model with higher overall accuracy but extremely poor Sand or Clay recall should not automatically be considered the best.

Interpret the results in both statistical and geotechnical terms.

### 12. Model selection and spatial robustness

Select the preferred model using the predetermined development-validation criteria.

Consider Macro-F1, balanced accuracy, both soil recalls, fold-to-fold variability, borehole-level reliability, and uncertainty.

Do not choose the primary block size based on the highest model score.

Evaluate 200 m and 300 m spatial-block sensitivity separately and interpret how changing validation geography affects the results.

Clearly distinguish exploratory model development from the final reported assessment.

Do not repeatedly adapt the methodology to outer-test failures while claiming the resulting evaluation remains unbiased.

If model performance is insufficient, identify credible causes and possible future improvements without manufacturing higher metrics.

### 13. Reproducibility

Create a clean research workflow with:

- Original Excel data audit
- Preprocessing notebook
- Original R `blockCV` analysis script
- Exported R spatial folds
- Python five-model training notebook
- Optuna tuning records
- Saved outer-fold predictions
- Trained models
- Final comparison tables
- Scientific figures
- Reproducibility and configuration files

The code must run in a reproducible environment, preferably Windows with RStudio and Jupyter Notebook.

Use fixed random seeds, explicit software versions, and consistent model configurations.

Provide checkpointing for computationally expensive experiments.

Test the workflow with a small software validation run before attempting the full experiment.

Never present smoke-test outputs as completed experimental results.

### 14. Final scientific outputs

Produce:

1. Verified, preprocessed soil classification dataset.
2. Spatial autocorrelation report.
3. Original R `blockCV` 100 m, 200 m, and 300 m outputs.
4. Five-model comparison for both geographical training extents.
5. Nested spatial-validation metrics and diagnostic figures.
6. Selected model with documented limitations.
7. External borehole evaluation, where appropriately available.
8. ArcGIS Pro-ready 3D Clay/Sand prediction and probability data.
9. Publication-quality methodology, equations, figures, tables, and discussion.
10. Reproducible research code and documentation.

Use peer-reviewed references and accurate citations.

Do not invent figures, citations, results, or completed computational experiments.

### 15. Required working procedure

Follow these stages in order:

**Stage 1 — Source review**

Read the uploaded Excel workbook, GitHub repository, and supplied research papers.

**Stage 2 — Data audit**

Investigate the boreholes, coordinates, soil labels, depths, elevations, missing values, and geographical coverage.

**Stage 3 — Methodological design**

Review the scientific literature and recommend a defensible spatial-validation and five-model comparison architecture.

**Stage 4 — R blockCV**

Perform the spatial autocorrelation analysis and generate the original R spatial-fold assignments.

**Stage 5 — Python modelling**

Build, verify, and execute the five-model nested spatial-CV workflow under both training extents.

**Stage 6 — Results analysis**

Evaluate predictive performance, class-specific reliability, depth effects, spatial robustness, and uncertainty.

**Stage 7 — Final reporting**

Prepare the reproducible final model, 3D prediction outputs, and scientific reporting materials.

At each stage, explain what has actually been completed and what remains.

Do not run long computational experiments before confirming the data preprocessing and validation architecture.

### 16. Your first task

Begin by independently reviewing my uploaded raw Excel workbook, research papers, and the original R `blockCV` GitHub repository.

Provide:

1. A verified summary of my dataset.
2. An assessment of the relevant published research.
3. An explanation of appropriate spatial autocorrelation analysis for my dataset.
4. A comparison of possible spatial-CV and nested-CV designs.
5. A recommended model training and validation architecture.
6. A practical experimental workflow and estimated computational requirements.
7. Any missing information that must be resolved before modelling.

**Do not assume any earlier model results exist. Do not fabricate an experimental history. Do not begin full model training until the source-data audit and experimental design have been reviewed.**

My goal is an independent, rigorous, reproducible machine-learning investigation of three-dimensional Clay/Sand classification in Downtown Siem Reap, suitable for a geotechnical engineering master's thesis and potential peer-reviewed publication.

**Start from the uploaded primary sources and develop the research independently.**
