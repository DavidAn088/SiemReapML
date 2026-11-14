# Raw data (not in git)

Copy from the V3 project's `data\raw` folder. Fingerprints (SHA-256) are checked against `settings.toml`.

| File | What it is |
|---|---|
| `SiemReap_BH_Master_Downtown_DEM_Region_v1_ArcGIS.csv` | main borehole file |
| `External_BHs_SR_Elev.csv` | new external boreholes (scored once, later) |

V7 also **reads** (never changes) the V3 project folder set in `settings.toml → [v3] folder`: its blockCV folds
(`data/folds`) and its external-holdout list (`data/processed/external_holdout.csv`, `config/external_holdout.lock.json`).
