"""The anchor: a versioned on-disk cycle boundary, not a pickle.

The restart artifact this replaces is ``pickle.dump`` of live host
objects -- unreadable by a Rust DA engine, undurable across numpy
versions, undiffable.  These tests hold the replacement to the three
properties that made the pickle unacceptable: it round-trips bitwise, it
hashes canonically so an independent implementation can reproduce the
number, and a half-written one is refused rather than read.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gpuwm.cycle.anchor import (AnchorRefusal, COMMIT_MARKER_NAME,
                                anchor_for_cycle, latest_anchor,
                                prognostic_sha256, read_anchor, write_anchor)
from gpuwm.cycle.contracts import ANCHOR_SCHEMA

NZ, NCELLS, NEDGES = 4, 12, 30


def _prognostic(dtype=np.float64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "rho": (1.0 + rng.random((NZ, NCELLS))).astype(dtype),
        "rho_theta": (250.0 + rng.random((NZ, NCELLS))).astype(dtype),
        "rho_u": rng.random((NZ, NEDGES)).astype(dtype),
        "rho_w": rng.random((NZ + 1, NCELLS)).astype(dtype),
        "scalars": rng.random((6, NZ, NCELLS)).astype(dtype),
        "time_seconds": np.asarray(120.0, dtype=np.float64),
    }


def _derived(dtype=np.float64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(8)
    return {
        "exner": rng.random((NZ, NCELLS)).astype(dtype),
        "density_perturbation": rng.random((NZ, NCELLS)).astype(dtype),
        "rho_theta_perturbation": rng.random((NZ, NCELLS)).astype(dtype),
        "pressure_perturbation": rng.random((NZ, NCELLS)).astype(dtype),
        "normal_velocity": rng.random((NZ, NEDGES)).astype(dtype),
        "vertical_velocity": rng.random((NZ + 1, NCELLS)).astype(dtype),
    }


def _write(root, **overrides):
    kwargs = dict(cycle_index=1, anchor_ticks=120_000,
                  valid_time="2026-08-14T02:00:00Z",
                  parent_kind="replay", prognostic=_prognostic(),
                  derived=_derived(), mesh_id="mesh-3km-test")
    kwargs.update(overrides)
    return write_anchor(root, **kwargs)


def test_anchor_roundtrips_bitwise(tmp_path):
    prognostic = _prognostic()
    derived = _derived()
    seam = {"microphysics_accum": np.arange(NCELLS, dtype=np.float32)}
    path = _write(tmp_path, prognostic=prognostic, derived=derived, seam=seam,
                  children=[{"grid_id": "d02", "state": "LIVE",
                             "placement": {"lat": 35.2, "lon": -97.4,
                                           "dx_m": 500.0, "nx": 64, "ny": 64},
                             "arrays": {"u": np.ones((3, 4))}}])

    doc = read_anchor(path)
    assert doc.manifest["schema"] == ANCHOR_SCHEMA
    assert doc.manifest["cycle_index"] == 1
    assert doc.manifest["anchor_ticks"] == 120_000
    assert doc.manifest["parent"]["kind"] == "replay"
    assert doc.manifest["parent"]["mesh_id"] == "mesh-3km-test"
    assert doc.manifest["parent"]["n_cells"] == NCELLS
    assert doc.manifest["parent"]["n_edges"] == NEDGES
    assert doc.manifest["parent"]["n_levels"] == NZ
    assert doc.manifest["array_format"] in ("netcdf4", "npz")

    restored = doc.prognostic()
    assert set(restored) == set(prognostic)
    for name, array in prognostic.items():
        assert np.array_equal(restored[name], array), name
        assert restored[name].dtype == array.dtype, name
    for name, array in _derived().items():
        assert np.array_equal(doc.derived()[name], array), name
    assert np.array_equal(doc.seam()["microphysics_accum"],
                          seam["microphysics_accum"])

    assert doc.manifest["parent"]["prognostic_sha256"] == \
        prognostic_sha256(prognostic)
    assert doc.manifest["derived"]["derived_from_sha256"] == \
        prognostic_sha256(prognostic)
    child = doc.manifest["children"][0]
    assert child["grid_id"] == "d02" and child["state"] == "LIVE"
    assert child["placement"]["dx_m"] == 500.0
    assert np.array_equal(doc.child_state("d02")["u"], np.ones((3, 4)))

    assert latest_anchor(tmp_path) == path
    assert anchor_for_cycle(tmp_path, 1) == path
    assert anchor_for_cycle(tmp_path, 4) is None


def test_sha_is_order_independent_and_dtype_sensitive():
    forward = _prognostic()
    shuffled = {name: forward[name] for name in reversed(list(forward))}
    assert list(shuffled) != list(forward)
    assert prognostic_sha256(shuffled) == prognostic_sha256(forward)

    as32 = {name: array.astype(np.float32) if array.dtype == np.float64
            else array for name, array in forward.items()}
    assert prognostic_sha256(as32) != prognostic_sha256(forward)

    bumped = {name: array.copy() for name, array in forward.items()}
    bumped["rho"][0, 0] = np.nextafter(bumped["rho"][0, 0], 1.0e9)
    assert prognostic_sha256(bumped) != prognostic_sha256(forward)


def test_partial_anchor_refuses(tmp_path):
    path = _write(tmp_path)
    (path / COMMIT_MARKER_NAME).unlink()
    with pytest.raises(AnchorRefusal) as excinfo:
        read_anchor(path)
    message = str(excinfo.value)
    assert "anchor is not committed" in message
    assert COMMIT_MARKER_NAME in message
    assert latest_anchor(tmp_path) is None


def test_commit_marker_detects_a_truncated_member(tmp_path):
    path = _write(tmp_path)
    marker = json.loads((path / COMMIT_MARKER_NAME).read_text())
    assert any(entry["path"].startswith("parent_prognostic")
               for entry in marker["files"])
    (path / "parent_derived.nc").write_bytes(b"truncated")
    with pytest.raises(AnchorRefusal) as excinfo:
        read_anchor(path).derived()
    assert "does not match the COMMIT marker" in str(excinfo.value)


def test_missing_prognostic_field_refuses_by_name(tmp_path):
    prognostic = _prognostic()
    prognostic.pop("rho_w")
    with pytest.raises(AnchorRefusal) as excinfo:
        _write(tmp_path, prognostic=prognostic)
    message = str(excinfo.value)
    assert "prognostic mapping is missing required fields" in message
    assert "rho_w" in message
    assert not list(tmp_path.glob("anchors/anchor_*"))


def test_nonfinite_prognostic_refuses_with_count_and_index(tmp_path):
    prognostic = _prognostic()
    prognostic["rho_theta"][1, 3] = np.nan
    prognostic["rho_theta"][2, 5] = np.inf
    with pytest.raises(AnchorRefusal) as excinfo:
        _write(tmp_path, prognostic=prognostic)
    message = str(excinfo.value)
    assert "non-finite values" in message
    assert "field='rho_theta'" in message
    assert "count=2" in message
    assert f"first_flat_index={1 * NCELLS + 3}" in message


def test_time_seconds_must_agree_with_anchor_ticks(tmp_path):
    prognostic = _prognostic()
    prognostic["time_seconds"] = np.asarray(119.0, dtype=np.float64)
    with pytest.raises(AnchorRefusal) as excinfo:
        _write(tmp_path, prognostic=prognostic, anchor_ticks=120_000)
    message = str(excinfo.value)
    assert "time_seconds disagrees with anchor_ticks" in message
    assert "state_time_seconds=119.0" in message


def test_increment_is_carried_and_reloadable(tmp_path):
    increment = {"rho_theta": np.zeros((NZ, NCELLS))}
    increment["rho_theta"][0, 1] = 0.25
    path = _write(tmp_path, analysis={"state": "APPLIED",
                                      "arrays": increment,
                                      "ingestion": {"state": "APPLIED"}})
    doc = read_anchor(path)
    assert doc.manifest["analysis"]["state"] == "APPLIED"
    assert np.array_equal(doc.increment()["rho_theta"],
                          increment["rho_theta"])
    assert doc.manifest["analysis"]["increment_sha256"] == \
        prognostic_sha256(increment)
