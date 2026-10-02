"""Every native momentum output word against compiled WRF v4.7.1.

The numerical table is an observation, asserted for equality. No tolerance or
``allclose`` hides a change. The same fixture also retains the source WRF
pressure and quantifies the launch's total/base pressure representation loss.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify.bigstep_momentum_oracle import (
    MOMENTUM_CASES, MOMENTUM_ORACLE_DIR, MOMENTUM_ROUTINES,
    load_momentum_fixture, load_momentum_measurement_pins,
    measure_momentum_parity, momentum_port_outputs,
)


def test_momentum_oracle_wrf471_source_and_fixture_pins():
    receipt = json.loads((MOMENTUM_ORACLE_DIR / "momentum-receipt.json").read_text())
    assert receipt["wrf_commit"] == "f52c197ed39d12e087d02c50f412d90d418f6186"
    assert receipt["source_sha256"] == "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
    assert receipt["constants_sha256"] == "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
    assert "-O0" in receipt["flags"] and "-ffp-contract=off" in receipt["flags"]
    assert "-DEM_CORE=1" in receipt["defines"]
    assert receipt["config_record"] == "actual generated WRF module_configure grid_config_rec_type"
    assert set(receipt["extracted"]) == set(MOMENTUM_ROUTINES) - {"combined"}
    for case in MOMENTUM_CASES:
        entry = receipt["cases"][case]
        assert hashlib.sha256((MOMENTUM_ORACLE_DIR / entry["fixture"]).read_bytes()).hexdigest() == entry["sha256"]
    platform_receipt = json.loads((MOMENTUM_ORACLE_DIR / "momentum-device-sm89-nvrtc12.9.86.json").read_text())
    assert platform_receipt["compile_platform"]["device_compute_capability"] == "89"
    assert platform_receipt["compile_platform"]["nvrtc_build"] == "12.9.86"
    assert len(platform_receipt["compile_platform"]["nvrtc_library_sha256"]) == 64
    for name, key in (("momentum-measurements-sm89-nvrtc12.9.86.json", "measurement_sha256"),
                      ("momentum-diagnostics-sm89-nvrtc12.9.86.json", "diagnostic_sha256")):
        assert hashlib.sha256((MOMENTUM_ORACLE_DIR / name).read_bytes()).hexdigest() == platform_receipt[key]
    primary = json.loads((MOMENTUM_ORACLE_DIR / "momentum-measurements.json").read_text())
    alternate = json.loads((MOMENTUM_ORACLE_DIR / "momentum-measurements-sm89-nvrtc12.9.86.json").read_text())
    for case in MOMENTUM_CASES:
        for routine in MOMENTUM_ROUTINES:
            for field in ("ru_t", "rv_t", "rw_t"):
                assert alternate[case][routine][field]["reference_sha256"] == primary[case][routine][field]["reference_sha256"]
                assert alternate[case][routine][field]["words"] == primary[case][routine][field]["words"]


@pytest.mark.parametrize("case", MOMENTUM_CASES)
def test_momentum_oracle_native_staggering_halos_and_pressure_contract(case):
    fixture = load_momentum_fixture(case)
    inputs = fixture.inputs
    nz, ny, nx = inputs["p"].shape
    assert inputs["u"].shape == (nz, ny, nx + 1)
    assert inputs["v"].shape == (nz, ny + 1, nx)
    assert inputs["w"].shape == (nz + 1, ny, nx)
    for routine in (*MOMENTUM_ROUTINES, "original_pressure"):
        for field in ("ru_t", "rv_t", "rw_t"):
            assert fixture.reference[routine][field].shape == inputs[field].shape
            assert np.isfinite(fixture.reference[routine][field]).all()
    canonical = np.asarray(inputs["p"] - inputs["pb"], dtype=np.float32)
    assert int(np.count_nonzero(canonical.view(np.uint32) != inputs["p_raw"].view(np.uint32))) == fixture.metadata["pressure_roundtrip_differing"]
    assert float(np.max(np.abs(canonical.astype(np.float64) - inputs["p_raw"]))) == fixture.metadata["pressure_roundtrip_max_abs_Pa"]
    if fixture.metadata["boundary_x"]:
        for routine in MOMENTUM_ROUTINES:
            assert np.array_equal(fixture.reference[routine]["ru_t"][:, :, (0, -1)].view(np.uint32), inputs["ru_t"][:, :, (0, -1)].view(np.uint32))
    if fixture.metadata["boundary_y"]:
        for routine in MOMENTUM_ROUTINES:
            assert np.array_equal(fixture.reference[routine]["rv_t"][:, (0, -1), :].view(np.uint32), inputs["rv_t"][:, (0, -1), :].view(np.uint32))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("case", MOMENTUM_CASES)
@pytest.mark.parametrize("routine", MOMENTUM_ROUTINES)
def test_bigstep_momentum_compiled_wrf471_every_output_word(case, routine):
    fixture = load_momentum_fixture(case)
    pin, platform = load_momentum_measurement_pins()
    outputs = momentum_port_outputs(fixture, routine)
    assert measure_momentum_parity(fixture.reference[routine], outputs) == pin[case][routine], f"momentum output drift on sm_{platform[0]} / NVRTC {platform[1]}"


def test_momentum_word_gate_detects_a_one_ulp_mutation():
    fixture = load_momentum_fixture("real")
    expected = fixture.reference["coriolis"]
    mutated = {field: value.copy() for field, value in expected.items()}
    mutated["ru_t"][0, 1, 1] = np.nextafter(mutated["ru_t"][0, 1, 1], np.float32(np.inf))
    measured = measure_momentum_parity(expected, mutated)
    assert measured["ru_t"]["max_ulp"] == 1
    assert measured["ru_t"]["differing_words"] == 1


def test_momentum_arithmetic_cause_controls_are_bit_identical():
    for filename in ("momentum-diagnostics.json", "momentum-diagnostics-sm89-nvrtc12.9.86.json"):
        diagnostics = json.loads((MOMENTUM_ORACLE_DIR / filename).read_text())
        for case in MOMENTUM_CASES:
            for control in ("explicit_fp32_operation_tree", "coriolis_wrf_sum_order_without_fma", "curvature_wrf_sum_order_without_fma"):
                for field in diagnostics[case][control].values():
                    assert field["max_ulp"] == 0
                    assert field["differing_words"] == 0
                    assert field["actual_sha256"] == field["reference_sha256"]
