"""An initial-condition source that publishes none of a model input refuses before the model runs.

WOOF owner canary run_b4dfcfa2b3767fb7 (AIFS from the GFS analysis, 2026-10-26): GFS has no skt, stl1, stl2, swvl1,
swvl2 or tcw; the model ran on NaN inputs and the failure surfaced only in preparation."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pytest


def test_all_missing_inputs_are_named_and_masked_fields_stay():
    torch = pytest.importorskip("torch")
    from gpuwm.earth2_runner import absent_initial_fields
    values = torch.ones(1, 1, 3, 4, 5)
    values[0, 0, 1] = float("nan")          # tcw: the source has none of it
    values[0, 0, 2, 0, :] = float("nan")    # sst: missing over land only
    coords = OrderedDict(time=np.zeros(1), lead_time=np.zeros(1), variable=np.array(["t2m", "tcw", "sst"]),
                         lat=np.zeros(4), lon=np.zeros(5))
    assert absent_initial_fields(values, coords) == ["tcw"]


def test_aifs_rows_start_from_the_ifs_analysis():
    table = json.loads((Path(__file__).resolve().parents[1] / "gpuwm/authorities/earth2-models.v1.json").read_text())
    assert table["runtime"]["initial_condition_sources"]["ifs"]["earth2studio_class"] == "IFS"
    for name in ("aifs", "aifs2", "aifs-ens", "aifs2-ens"):
        assert table["models"][name]["initial_condition_source"] == "ifs", name
