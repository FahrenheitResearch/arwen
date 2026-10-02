"""Big-step coupling gates against unchanged compiled WRF v4.7.1.

The expected arrays are Fortran output words, not a Python transcription.
Nonzero rounding tables pin exact measurements and complete output hashes.
They must not be mistaken for a freely adjustable numerical tolerance.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify.bigstep_coupling_oracle import (ORACLE_DIR, CASE_NAMES,
    coupling_cases, load_coupling_oracle, coupling_port_outputs,
    coupling_wrf_flux_trace, coupling_php_order_trace, word_measurement)


def test_coupling_fixture_words_and_fortran_pin():
    receipt=json.loads((ORACLE_DIR/"coupling-receipt.json").read_text())
    assert receipt["build"]["wrf_commit"]=="f52c197ed39d12e087d02c50f412d90d418f6186"
    assert receipt["build"]["source_sha256"]=="bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
    assert receipt["build"]["constants_sha256"]=="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
    assert "-DEM_CORE=1" in receipt["build"]["flags"]
    assert "-ffp-contract=off" in receipt["build"]["flags"]
    assert "_ZGV" not in receipt["build"]["undefined_symbols"]
    assert tuple(case["name"] for case in receipt["cases"])==CASE_NAMES
    assert hashlib.sha256((ORACLE_DIR/"coupling.npz").read_bytes()).hexdigest()==receipt["fixture_sha256"]
    fixture=load_coupling_oracle()
    assert set(fixture)==set(receipt["arrays"])
    for key,array in fixture.items():
        assert list(array.shape)==receipt["arrays"][key]["shape"]
        assert hashlib.sha256(array.tobytes()).hexdigest()==receipt["arrays"][key]["sha256"]


@pytest.fixture(scope="module")
def results():
    cases=coupling_cases()
    return {case["name"]:coupling_port_outputs(case) for case in cases}


@pytest.fixture(scope="module")
def reference():
    return load_coupling_oracle()


@pytest.fixture(scope="module")
def baseline():
    return json.loads((ORACLE_DIR/"coupling-baseline.json").read_text())


def assert_words_equal(a,b):
    np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",CASE_NAMES)
def test_calc_cq_against_compiled_wrf(name,results,reference,baseline):
    for key in ("cqu","cqv","cqwr"):
        actual,expected=results[name][key],reference[name+"/"+key]
        assert word_measurement(actual,expected)==baseline[name][key]
        if name=="real_boundary" and key=="cqu":
            # acoustic.cu explicitly fills periodic duplicate scratch faces;
            # nonperiodic boundary-normal faces never consume these factors.
            assert_words_equal(actual[:,:,1:-1],expected[:,:,1:-1])
            assert baseline[name][key]["changed"]==672
        elif name=="real_boundary" and key=="cqv":
            assert_words_equal(actual[:,1:-1,:],expected[:,1:-1,:])
            assert baseline[name][key]["changed"]==878
        else:
            assert_words_equal(actual,expected)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",CASE_NAMES)
def test_calc_mu_uv_against_compiled_wrf(name,results,reference,baseline):
    # ieva.py declares the separate-total-mass addition order. The retained
    # perturbation column exposes its single-ULP rounding difference.
    for key in ("muu","muv"):
        actual,expected=results[name][key],reference[name+"/"+key]
        measured=word_measurement(actual,expected)
        assert measured==baseline[name][key]
        assert measured["max_ulp"]==(1 if name=="mass_perturbation" else 0)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",CASE_NAMES)
def test_calc_ww_cp_against_compiled_wrf(name,results,reference,baseline):
    # Direct division in the port versus WRF's stored reciprocal multiply
    # changes a flux's last bit; cancellation amplifies its relative ULPs.
    assert word_measurement(results[name]["ww"],reference[name+"/ww"])==baseline[name]["ww"]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",CASE_NAMES)
def test_calc_ww_cp_scan_with_wrf_flux_words(name,reference,baseline):
    case=next(case for case in coupling_cases() if case["name"]==name)
    actual=coupling_wrf_flux_trace(case)
    expected=reference[name+"/ww"]
    assert_words_equal(actual,expected)
    assert word_measurement(actual,expected)==baseline[name]["ww_wrf_flux_trace"]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",CASE_NAMES)
def test_calc_php_fused_consumer_against_compiled_wrf(name,results,reference,baseline):
    # WRF stores and rounds full half-level geopotential before its face
    # difference. acoustic.cu:199-234 documents a fused consumer that takes
    # separate perturbation/base face differences and then joins them.
    case=next(case for case in coupling_cases() if case["name"]==name)
    trace=coupling_php_order_trace(case)
    for key in ("php_ru","php_rv"):
        actual,expected=results[name][key],reference[name+"/"+key]
        assert word_measurement(actual,expected)==baseline[name][key]
        assert_words_equal(actual,trace[key])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",CASE_NAMES)
def test_w_damp_all_outputs_against_compiled_wrf(name,results,reference):
    # Includes both strict onset branches, nextafter neighbours at 1/2,
    # upper/lower rows, both signs of w and the two returned CFL scalars.
    for key in ("rwd","maxv","maxh"):
        assert_words_equal(results[name][key],reference[name+"/"+key])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name",CASE_NAMES)
def test_rk_rayleigh_damp_declared_missing_path(name,results,reference,baseline):
    # diffusion.py:25-37 and apply_rayleigh_damping document this no-op for
    # damp_opt=2. It is a measured restriction, never numerical parity.
    for key in ("rayleigh_ru","rayleigh_rv","rayleigh_rw","rayleigh_rt"):
        assert not np.any(results[name][key])
        assert word_measurement(results[name][key],reference[name+"/"+key])==baseline[name][key]
    assert any(baseline[name][key]["changed"] for key in ("rayleigh_ru","rayleigh_rv","rayleigh_rt"))
