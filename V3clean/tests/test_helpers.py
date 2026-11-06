"""Checks of V3's rules in helpers.py.  Run:  python -m pytest -q tests"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import helpers as H  # noqa: E402

S = H.load_settings(ROOT)


def records(rows):
    df = pd.DataFrame(rows, columns=["BH_Code", "Depth_From_m", "Depth_To_m", "USCS"])
    return df.assign(UTM_Easting_m=0.0, UTM_Northing_m=0.0, Ground_Elevation_m=10.0, Region="Downtown")


def test_v3_fill_rule_unclassified_record_blocks():
    r = records([("B", 0.5, 1.0, "CL"), ("B", 1.0, 1.45, None), ("B", 1.5, 2.0, "CL"), ("B", 2.0, 2.5, "CL"),
                 ("B", 3.0, 3.5, "CL")])
    cells, _ = H.make_cells(r, S)
    out, info = H.fill_gaps(cells, r, S)
    assert list(out[out.measured == 0].cell) == [5]          # 2.5-3.0 m filled; cell 2 has a record inside
    assert info["record_inside"] == 1


def test_threshold_ties_go_to_the_value_closest_to_half():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    t, f = H.best_threshold(y, p, H.threshold_grid(S))
    assert f == 1.0 and abs(t - 0.5) < 1e-9


def test_v3_seed_rule():
    assert H.seed_for(S, "A_106m_p0_XGB", 1) == H.seed_for(S, "A_106m_p0_XGB", 1)
    assert H.seed_for(S, "A_106m_p0_XGB", 1) != H.seed_for(S, "A_106m_p0_XGB", 2)
