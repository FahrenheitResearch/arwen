"""The analysis capture replays the solve it captured, byte for byte.

Breakage prevented: a real-case identity receipt built from a capture that
does not reproduce the captured solve would prove nothing about the engine.
"""
import json

import numpy as np

from gpuwm.da import letkf
from gpuwm.da import radar_assimilation as ra
from test_letkf_host_parallel import geodesic_problem
from tools import da_analysis_capture as cap


def test_capture_replay_and_compare_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_DA_HOST_WORKERS", "1")
    prior, obs, grid, cfg = geodesic_problem(fields=3, batches=3, seed=12)
    direct = letkf.analyze(prior, obs, grid, cfg, solve_namespace=np)

    monkeypatch.setattr(ra, "_execute_analysis", ra._execute_analysis)
    cap.install(tmp_path / "cap")
    increments, *_rest, storage, _attempts = ra._execute_analysis(
        letkf.analyze, prior, obs, grid, cfg, namespace=np, device="host")
    assert storage == "host"
    call = tmp_path / "cap" / "call-000"
    record = json.loads((call / "capture.json").read_text())
    assert record["fields"] == list(prior) and record["batches"] == 3
    # The live solve's own increments are hashed, so a replay can be held
    # against what the run applied, not only against another replay.
    assert record["sha256_in_run"] == {
        name: cap._digest(direct[name]) for name in prior}

    replayed = cap.replay(call, workers=None, out=tmp_path / "a.npz",
                          record_path=tmp_path / "a.json")
    for name in prior:
        assert increments[name].tobytes() == direct[name].tobytes()
        assert replayed["sha256"][name] == cap._digest(direct[name])
    np.savez(tmp_path / "b.npz", **direct)
    assert cap.compare(tmp_path / "a.npz", tmp_path / "b.npz")["all_identical"]

    moved = dict(direct)
    moved["field_0"] = direct["field_0"].copy()
    moved["field_0"].flat[7] += 1e-9
    np.savez(tmp_path / "c.npz", **moved)
    result = cap.compare(tmp_path / "a.npz", tmp_path / "c.npz")
    assert not result["all_identical"]
    assert result["fields"]["field_0"]["differing"] == 1
    assert result["fields"]["field_1"]["identical_bytes"]
