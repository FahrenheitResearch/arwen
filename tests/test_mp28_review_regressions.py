"""Behavioral landing-review gates for the aerosol-aware default route."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import json

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gate_cutter_requires_measurement_copy(tmp_path):
    tool = ROOT / "tools/thompson_aerosol_column_oracle/make_oracle_gate.py"
    result = subprocess.run([sys.executable, str(tool), str(tmp_path),
                             str(tmp_path / "gate.npz")], capture_output=True,
                            text=True)
    assert result.returncode == 2
    assert "--wrfread" in result.stderr
    assert not (tmp_path / "gate.npz").exists()


def test_gate_cutter_refuses_same_labels_with_changed_input_words(tmp_path):
    gate = dict(np.load(ROOT / "tests/data/mp28_column_oracle_wrf461.npz"))
    inputs = ("p", "th", "geop", "w", "qv", "qc", "qr", "qi", "qs", "qg",
              "ni", "nr", "nc", "nwfa", "nifa", "nwfa2d", "nifa2d")
    raw = {k: gate[k] for k in inputs}
    raw.update(labels=gate["labels"], regime=np.asarray(["gate"]*157))
    stock, measured = tmp_path / "stock", tmp_path / "measured"
    stock.mkdir(); measured.mkdir()
    for path in (stock, measured):
        np.savez(path / "columns.npz", **raw)
        for dt in (20, 5):
            expected = {str(k): gate[f"wrf_{k}_dt{dt}"] for k in gate["fields"]}
            expected.update(in_pii=gate[f"in_pii_dt{dt}"], in_dz=gate["p"],
                            in_hgt=gate["p"])
            np.savez(path / f"gpu-strict-dt{dt}.npz", **expected)
            np.savez(path / f"summary-strict-dt{dt}.wrf.npz", **expected)
    raw["qv"] = raw["qv"].copy()
    raw["qv"][0, 0] = np.nextafter(raw["qv"][0, 0], np.float32(np.inf))
    np.savez(measured / "columns.npz", **raw)
    result = subprocess.run([sys.executable,
        str(ROOT / "tools/thompson_aerosol_column_oracle/make_oracle_gate.py"),
        str(stock), str(tmp_path / "gate.npz"), "--wrfread", str(measured)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "different input words: qv" in result.stderr
    assert not (tmp_path / "gate.npz").exists()


def test_retired_powf_probe_is_a_named_tombstone():
    tool = ROOT / "tools/thompson_fork_oracle/plain_powf_probe.py"
    result = subprocess.run([sys.executable, str(tool)], capture_output=True,
                            text=True)
    assert result.returncode == 2
    assert "retired" in result.stderr
    assert "test_no_plain_powf_survives_in_any_mp28_arm" in result.stderr
    assert "Traceback" not in result.stderr
    receipt = json.loads((ROOT / "tests/data/thompson_fork_plain_powf_receipt.json").read_text())
    assert receipt["status"] == "retired"
    assert "site_count" not in receipt


@pytest.mark.gpu
def test_entry_mask_preserves_exact_melting_level():
    cp = pytest.importorskip("cupy")
    from gpuwm.core.thompson_aerosol_state import launch_aa_entry_warm_mask
    t = cp.asarray(np.array([[273.15, 273.15, 273.15],
                             [273.15, 273.15003,
                              np.nextafter(np.float32(273.15), np.float32(0))],
                             [273.15, 273.15, 273.15]], np.float32)[:, None, :])
    mask = cp.empty_like(t)
    launch_aa_entry_warm_mask(t, mask)
    np.testing.assert_array_equal(cp.asnumpy(mask)[:, 0],
                                  [[2, 1, 2], [2, 1, 0], [2, 2, 2]])


@pytest.mark.gpu
def test_mp28_obsop_uses_aerosol_diagnosis_words():
    cp = pytest.importorskip("cupy")
    from test_da_obsop_thompson_gpu import _thompson_state
    from gpuwm.da import obsop
    from gpuwm.core.thompson_aerosol_state import (
        launch_aerosol_exner, launch_aa_graupel_number_init,
        launch_aa_graupel_number_finalize, launch_aa_refl10cm)
    from gpuwm.core.thompson_aerosol_state import launch_tau1_density
    state, _ = _thompson_state(shape=(7, 3, 5))
    state.p[...] = cp.linspace(25000, 100000, state.p.size,
                               dtype=cp.float32).reshape(state.p.shape)
    state.qg[...] = cp.asarray(np.geomspace(2e-12, 0.02, state.qg.size,
                                           dtype=np.float32).reshape(state.qg.shape))
    before = cp.asnumpy(state.qg).tobytes()
    got = obsop.simulated_reflectivity(
        state, SimpleNamespace(mp_physics=28, thompson_version="wrf_461")).copy()
    exner = cp.empty_like(state.p)
    launch_aerosol_exner(state.p, exner)
    t = (state.thb[:, None, None] + state.thp) * exner
    mass = state.qg.copy()
    shadow = cp.empty_like(mass)
    rho = cp.empty_like(mass)
    launch_aa_graupel_number_init(mass, t, state.p, state.qv, shadow)
    launch_tau1_density(t, state.p, state.qv, rho)
    launch_aa_graupel_number_finalize(mass, shadow, rho)
    expected = cp.empty_like(mass)
    launch_aa_refl10cm(state.qv, state.qr, state.nr, state.qs, mass,
                      shadow, t, state.p, expected)
    assert cp.asnumpy(got).tobytes() == cp.asnumpy(expected).tobytes()
    assert cp.asnumpy(state.qg).tobytes() == before
