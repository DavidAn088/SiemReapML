# Raw data (not in git)

Copy these from the V3 project's `data\raw` folder. Their SHA-256 fingerprints are checked against `settings.toml`.

| File or folder | What it is |
|---|---|
| `SiemReap_BH_Master_Downtown_DEM_Region_v1_ArcGIS.csv` | main borehole file |
| `External_BHs_SR_Elev.csv` | the new external boreholes (notebook 07) |
| `gis/downtown/` | Downtown boundary (shapefile) |
| `gis/dem_downtown/` | DEM (ESRI grid folder) |

V3-clean also **reads** (never changes) the V3 project folder set in `settings.toml → [v3] folder`:
its blockCV folds (`data/folds`), its external-holdout list (`data/processed/external_holdout.csv`,
`config/external_holdout.lock.json`) and, for notebook 10, its results (`outputs/`).
