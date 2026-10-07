"""The analysis capture/replay tool: replay selects the neighbour search, compare is exact.

tools/da_radar_capture.py is how the roster was judged on the real case
with the background held fixed (two driver runs do not reproduce the
forecast leg bit for bit).  The breakages this prevents: a replay whose
--neighbor-search never reaches the filter configuration (it would compare
a route with itself), and a compare that calls rounding identity.
"""
import json
import pickle

import numpy as np

from gpuwm.da import radar_assimilation as ra
from gpuwm.da.letkf import Localization
from tools import da_radar_capture as cap


def test_replay_reaches_the_filter_config_and_compare_is_exact(tmp_path, monkeypatch, capsys):
    slot = tmp_path / "call00"
    slot.mkdir()
    cfg = ra.RadarAssimilationConfig(
        localization=Localization(9000.0, 3000.0), rtps_alpha=0.5,
        analysis_fields=("u",), velocity=True)
    with open(slot / "inputs.pkl", "wb") as stream:
        pickle.dump({"checkpoints": {0: "m0.npz"}, "observations": "radar.nc",
                     "grid": None, "cfg": cfg, "extra_obs": None,
                     "extra_obs_provenance": None, "has_reflectivity": False,
                     "other": []}, stream)
    seen = []

    def fake(checkpoints, observations, grid, cfg, **kwargs):
        seen.append(ra._letkf_config(cfg, 512.0).neighbor_search)
        value = 1.0 if seen[-1] == "index" else 1.0 + 2**-40
        return ({0: {"u": np.full((2, 3, 4), value)}},
                {"filter": {}, "stage_wall_seconds": {}})

    monkeypatch.setattr(ra, "assimilate_radar_grid", fake)
    cap.main(["replay", str(slot), str(tmp_path / "a"), "--neighbor-search", "index"])
    cap.main(["replay", str(slot), str(tmp_path / "b"), "--neighbor-search", "forward"])
    cap.main(["replay", str(slot), str(tmp_path / "c"), "--neighbor-search", "index"])
    assert seen == ["index", "forward", "index"]
    capsys.readouterr()
    rows = cap._compare(tmp_path / "a", tmp_path / "b")
    assert rows["u"]["bitwise"] is False and rows["u"]["max_abs_diff"] > 0
    assert cap._compare(tmp_path / "a", tmp_path / "c")["u"]["bitwise"] is True
    receipt = json.loads((tmp_path / "a" / "receipt.json").read_text())
    assert receipt["mode"] == "index"
    assert receipt["peak_rss_gib"] is None or receipt["peak_rss_gib"] > 0
