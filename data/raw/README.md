# Raw data (not in git)

Place the received files here. They are git-ignored because the repository is public.
The scripts verify these checksums (configured in `config/project.toml`) before running.

| Path | SHA-256 | Received |
|---|---|---|
| `SiemReap_BH_Master_Downtown_DEM_Region_v1_ArcGIS.csv` | `caf59d0b4bd244303fc48679d28fc62fd572c9175a856744b07554e48f380f72` | 2026-10-12 |
| `Final_Men_Ten_Mg_Hah_KGEO_Digitalized_COMPLETE_Depth_Formulas.xlsx` | `78a426c61a30cd103b1477aa7b52e67a8cc45f1163e00d5f0d0d938032aaaddc` | 2026-10-12 |
| `gis/downtown/downtown.shp` (+ .shx, .dbf, .prj, ...) | `6ab634176e29a9691e8117deb37a5864b63797fb1aa065c48ae9adc9654f76e7` | 2026-10-12 |
| `gis/ring_5km/Ring_5km.shp` (+ .shx, .dbf, .prj, ...) | `70e23e22503b1f401cd9c7845b24a2a6478df3016e57a9ca85b3ea0defc0cfd8` | 2026-10-12 |
| `gis/dem_downtown/` (ESRI grid; checksum of `w001001.adf`) | `9ef9086739ea12cd4a219d6290045d781c07b0ec619af8401be28d59eb492a9f` | 2026-10-12 |

The 5 km ring zip also contained a georeferenced land-use master-plan image, which is not used.
