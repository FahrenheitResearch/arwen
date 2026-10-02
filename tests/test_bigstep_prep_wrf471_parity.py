"""Compiled WRF prep parity and the boundary defect the mirror shared."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.bigstep_prep_oracle import (
    ORACLE_DIR, PREP_CASES, PHY_MAPPING, load_prep_fixture,
    measure_prep, prep_port_outputs,
)


def test_prep_oracle_source_fixture_and_constant_pins():
    receipt = json.loads((ORACLE_DIR / "prep-receipt.json").read_text())
    assert receipt["wrf_commit"] == "f52c197ed39d12e087d02c50f412d90d418f6186"
    assert receipt["source_sha256"] == "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
    assert receipt["constants_sha256"] == "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
    assert receipt["state_sha256"] == hashlib.sha256((ORACLE_DIR / "state-real.npz").read_bytes()).hexdigest()
    assert receipt["omega_fixture_sha256"] == hashlib.sha256((ORACLE_DIR / "coupling.npz").read_bytes()).hexdigest()
    for filename, digest in receipt["files"].items():
        assert hashlib.sha256((ORACLE_DIR / filename).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("case", PREP_CASES)
def test_prep_fixture_captures_all_native_outputs(case):
    fixture = load_prep_fixture(case)
    for name in ("ph_tend", "rho", "th_phy", "th_phy_m_t0", "p_phy", "pi_phy", "u_phy", "v_phy",
                 "p8w", "t_phy", "t8w", "z", "z_at_w", "dz8w", "p_hyd", "p_hyd_w"):
        assert fixture[name].shape == (50, 10, 12)
    assert fixture["input_U"].shape == (49, 10, 13)
    assert fixture["input_V"].shape == (49, 11, 12)
    assert np.isfinite(fixture["ph_tend"]).all()


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("case", PREP_CASES)
def test_bigstep_rhs_ph_compiled_wrf471_every_output_word(case):
    fixture = load_prep_fixture(case)
    pins = json.loads((ORACLE_DIR / "prep-measurements.json").read_text())
    got = prep_port_outputs(fixture)
    measured = measure_prep(fixture,{"ph_tend":got["ph_tend"]})
    assert measured["ph_tend"] == pins[case]["ph_tend"]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("case", PREP_CASES)
def test_bigstep_phy_prep_compiled_wrf471_exposed_outputs(case):
    fixture = load_prep_fixture(case)
    pins = json.loads((ORACLE_DIR / "prep-measurements.json").read_text())
    got = prep_port_outputs(fixture)
    measured = measure_prep(fixture,{name:got[name] for name in PHY_MAPPING})
    assert measured == {name:pins[case][name] for name in PHY_MAPPING}


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("supplied_face_masses", (False, True))
def test_rhs_ph_specified_corner_has_no_horizontal_advection(supplied_face_masses):
    fixture = load_prep_fixture("real-specified2")
    motionless = {key:value.copy() for key,value in fixture.items()}
    motionless["input_W"][...] = 0
    motionless["input_WW"][...] = 0
    got = prep_port_outputs(motionless,supplied_face_masses=supplied_face_masses)["ph_tend"]
    # WRF skips both directional terms at the four outer corners. The
    # former half-face implementation produced a nonzero contribution.
    for j in (0, got.shape[1]-1):
        for i in (0, got.shape[2]-1):
            assert np.array_equal(got[:,j,i].view("u4"),np.zeros(50,"f4").view("u4"))


@pytest.mark.parametrize("case", PREP_CASES)
def test_phy_interface_temperature_rounding_is_in_its_input(case):
    from gpuwm.core.rrtmg_legacy import _t8w_columns
    fixture = load_prep_fixture(case)
    temperature = fixture["t_phy"][:49].reshape(49,-1).T
    height = fixture["z_at_w"].reshape(50,-1).T
    got = _t8w_columns(temperature,height,fixture["input_FNM"],fixture["input_FNP"])
    want = fixture["t8w"].reshape(50,-1).T
    assert np.array_equal(got.view("u4"),want.view("u4"))


def test_prep_word_measurement_detects_one_word_mutation():
    fixture = load_prep_fixture("real-periodic2")
    mutated = fixture["ph_tend"].copy()
    mutated[3,2,2] = np.nextafter(mutated[3,2,2],np.float32(np.inf))
    distance = fp32_ulp_distance(mutated,fixture["ph_tend"])
    assert np.count_nonzero(distance) == 1
    assert int(distance.max()) == 1
