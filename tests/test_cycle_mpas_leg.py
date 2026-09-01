"""The standalone leg runner: anchor N -> analysis -> advance -> anchor N+1.

The dycore lives in another repo that pins this one by SHA, so the leg
runner talks to it through files.  Two things are tested here and they
are the two that a plausible-looking implementation gets wrong: the stub
backend must announce that it did not integrate anything, and a leg whose
carried diagnostics no longer describe its prognostics must refuse to
advance rather than integrate a state that is not in agreement with
itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.cycle.anchor import prognostic_sha256, read_anchor, write_anchor
from gpuwm.cycle.contracts import CycleRefusal
from tools.cycle_mpas_leg import main, run_leg

NZ, NCELLS, NEDGES = 4, 10, 24
P0, RD, CP = 1.0e5, 287.0, 1004.5


def _prognostic(seed: int, time_seconds: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "rho": 1.0 + rng.random((NZ, NCELLS)),
        "rho_theta": 250.0 + 30.0 * rng.random((NZ, NCELLS)),
        "rho_u": rng.random((NZ, NEDGES)),
        "rho_w": rng.random((NZ + 1, NCELLS)),
        "scalars": rng.random((6, NZ, NCELLS)),
        "time_seconds": np.asarray(time_seconds, dtype=np.float64),
    }


def _exner(rho_theta: np.ndarray) -> np.ndarray:
    return (RD * rho_theta / P0) ** (RD / (CP - RD))


def _seed_anchor(root, *, analysis=None, extra_derived=False):
    prognostic = _prognostic(11, 0.0)
    derived = {"exner": _exner(prognostic["rho_theta"])}
    if extra_derived:
        derived["normal_velocity"] = np.zeros((NZ, NEDGES))
    return write_anchor(root, cycle_index=0, anchor_ticks=0,
                        valid_time="2026-08-14T02:00:00Z",
                        parent_kind="replay", prognostic=prognostic,
                        derived=derived, mesh_id="mesh-3km-test",
                        analysis=analysis)


def _seed_history(root):
    frames = root / "history"
    frames.mkdir(parents=True, exist_ok=True)
    for index, time_seconds in enumerate((0.0, 120.0, 240.0), start=0):
        state = _prognostic(11 + index, time_seconds)
        state["exner"] = _exner(state["rho_theta"])
        np.savez(frames / f"frame_{index:03d}.npz", **state)
    return str(frames / "frame_*.npz")


def test_replay_backend_stamps_the_stub_loudly(tmp_path, capsys):
    _seed_anchor(tmp_path)
    history = _seed_history(tmp_path)
    code = main(["--root", str(tmp_path), "--backend", "replay",
                 "--history", history, "--cycle-seconds", "120"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "REPLAY BACKEND: this leg did not integrate a dycore" in \
        captured.out

    doc = read_anchor(tmp_path / "anchors" / sorted(
        p.name for p in (tmp_path / "anchors").iterdir())[-1])
    assert doc.manifest["cycle_index"] == 1
    assert doc.manifest["parent"]["kind"] == "replay"
    assert doc.manifest["anchor_ticks"] == 120_000
    assert doc.manifest["analysis"]["state"] == "NULL_ARM"
    assert doc.manifest["analysis"]["ingestion"]["state"] == "NULL_ARM"
    assert doc.manifest["diagnostics_rebuilt"] is False


def test_stale_derived_refuses_when_rebuild_unavailable(tmp_path, capsys):
    """RED-ON-REVERT ANCHOR: drop the stale-derived branch, this goes green."""
    increment = {"rho_theta": np.zeros((NZ, NCELLS))}
    increment["rho_theta"][2, 4] = 2.5
    _seed_anchor(tmp_path, extra_derived=True,
                 analysis={"state": "APPLIED", "arrays": increment})
    history = _seed_history(tmp_path)
    code = main(["--root", str(tmp_path), "--backend", "replay",
                 "--history", history, "--cycle-seconds", "120"])
    captured = capsys.readouterr()
    assert code == 3
    assert "cannot resume on stale derived diagnostics" in captured.err
    assert "normal_velocity" in captured.err
    # ...and nothing was published on top of the refusal.
    assert sorted(p.name for p in (tmp_path / "anchors").iterdir()) == \
        [p.name for p in (tmp_path / "anchors").iterdir() if "_000_" in p.name]

    with pytest.raises(CycleRefusal) as excinfo:
        run_leg(root=tmp_path, backend_name="replay", history=history,
                cycle_seconds=120.0, port_root=None)
    assert excinfo.value.observed["unrebuildable_fields"] == \
        ["normal_velocity"]


def test_rebuildable_stale_derived_advances_and_receipts_ingestion(tmp_path,
                                                                   capsys):
    increment = {"rho_theta": np.zeros((NZ, NCELLS))}
    increment["rho_theta"][2, 4] = 2.5
    anchor0 = _seed_anchor(tmp_path,
                           analysis={"state": "APPLIED",
                                     "arrays": increment})
    background = read_anchor(anchor0).manifest["parent"]["prognostic_sha256"]
    history = _seed_history(tmp_path)
    code = main(["--root", str(tmp_path), "--backend", "replay",
                 "--history", history, "--cycle-seconds", "120"])
    captured = capsys.readouterr()
    assert code == 0, captured.err

    doc = read_anchor(tmp_path / "anchors" / sorted(
        p.name for p in (tmp_path / "anchors").iterdir())[-1])
    assert doc.manifest["diagnostics_rebuilt"] is True
    ingestion = doc.manifest["analysis"]["ingestion"]
    assert ingestion["state"] == "APPLIED"
    assert ingestion["background_sha256"] == background
    assert ingestion["analysis_sha256"] != background
    assert ingestion["increment_nonzero_cells"] == 1
    assert ingestion["increment_sha256"] == prognostic_sha256(increment)


def test_mpas_cuda_backend_names_the_missing_module(tmp_path, capsys):
    _seed_anchor(tmp_path)
    code = main(["--root", str(tmp_path), "--backend", "mpas-cuda",
                 "--port-root", str(tmp_path / "not-a-port"),
                 "--cycle-seconds", "120"])
    captured = capsys.readouterr()
    assert code == 2
    assert "not-a-port" in captured.err
    assert "mpas_cuda" in captured.err or "module" in captured.err
