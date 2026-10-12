# SiemReapML

3D Clay/Sand classification of the upper 15 m beneath Downtown Siem Reap, Cambodia, using machine learning with nested spatial cross-validation. Spatial folds come from the R package blockCV; modelling runs in Python. This is a master's thesis project in Geotechnical Engineering at Chulalongkorn University.

## Status

| Stage | State |
|---|---|
| 1 Source review | Done for the files received |
| 2 Data audit | Done: `scripts/01_audit_raw_data.py` |
| 3 Experimental design | Proposed, awaiting review: [`docs/01_source_review_and_design.md`](docs/01_source_review_and_design.md) |
| 4 R blockCV + variograms | Not started |
| 5 Python nested CV, 5 models × 2 training extents | Not started |
| 6–7 Results, 3D output, reporting | Not started |

No model has been trained and no performance figure exists yet.

## Layout

```
config/project.toml          fixed constants (depth range, 0.5 m cells, label map, CRS, seed)
data/raw/                    received data (git-ignored; see data/raw/README.md)
scripts/01_audit_raw_data.py Stage 2 audit
outputs/01_audit/aggregate/  audit tables and figures without borehole identifiers
outputs/01_audit/restricted/ borehole-level audit outputs (git-ignored)
docs/                        stage reports
```

## Reproducing the audit

```bash
pip install -r requirements-audit.txt
# place the CSV in data/raw/ (checksum is verified)
python scripts/01_audit_raw_data.py
```

## Data confidentiality

This repository is public. Raw borehole records, borehole identifiers and coordinates are never committed. Only aggregate statistics are.
