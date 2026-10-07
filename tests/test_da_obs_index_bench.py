"""The neighbour-roster bench runs and its two modes agree byte for byte.

tools/da_obs_index_bench.py is how the roster's speed and identity were
measured at the recent case's shape.  The breakage this prevents: a bench
whose stand-in drifted from what analyze() accepts, or whose digest stopped
being comparable between the two neighbour searches.
"""
import json

import pytest

from tools import da_obs_index_bench as bench


def test_bench_modes_share_one_digest_on_a_small_stand_in(capsys):
    rows = []
    for mode in ("forward", "index"):
        bench.main(["--mode", mode, "--namespace", "numpy", "--ny", "31",
                    "--nz", "9", "--budget-mib", "64"])
        rows.append(json.loads(capsys.readouterr().out.strip().splitlines()[-1]))
    assert rows[0]["digest"] == rows[1]["digest"]
    assert rows[0]["active_points"] == rows[1]["active_points"] > 0
    assert rows[1]["neighbor_index"]["neighbours"] > 0
