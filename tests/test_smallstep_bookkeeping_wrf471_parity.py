"""Exact measurement pins for the compiled WRF v4.7.1 small-step routines.

The metrics are measurements, asserted for equality rather than treated as
an allowed tolerance.  Native theta and the separately labelled full-theta
representation witness are both pinned.  Fixtures contain native WRF output
words, not a Python reimplementation's answer.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify.smallstep_bookkeeping_oracle import (
    BOOKKEEPING_DIR, OUTPUT_COVERAGE, PORT_RUNNERS, ROUNDING_TRACES, load_bookkeeping)
from gpuwm.verify.smallstep_oracle import word_metrics


def _manifest():
    return json.loads((BOOKKEEPING_DIR / "manifest.json").read_text())


def _names(routine):
    return tuple(name for name, row in _manifest()["files"].items()
                 if row["routine"] == routine)


def test_bookkeeping_compiled_wrf_fixture_hashes():
    manifest = _manifest()
    assert len(manifest["files"]) == 120
    for name, row in manifest["files"].items():
        assert hashlib.sha256((BOOKKEEPING_DIR / name).read_bytes()).hexdigest() == row["sha256"]


def test_bookkeeping_native_and_representation_witnesses_are_separate():
    for name in _names("prep"):
        fixture = load_bookkeeping(BOOKKEEPING_DIR / name)
        assert "ref_native_th_pp" in fixture
        assert "isolation_th_pp" in fixture
        assert set(key for key in fixture if key.startswith("ref_")) >= {
            "ref_u_pp", "ref_v_pp", "ref_w_pp", "ref_native_th_pp", "ref_ph_pp",
            "ref_mu_pp", "ref_al_pp", "ref_p_pp", "ref_p_pp_old"}
        for key, value in fixture.items():
            if key.startswith("ref_"):
                assert value.dtype == np.dtype("float32")


def test_every_compiled_wrf_bookkeeping_output_array_has_a_comparison():
    schema = json.loads((BOOKKEEPING_DIR.parent / "wrf-schema.json").read_text())
    for routine, coverage in OUTPUT_COVERAGE.items():
        expected = {name for name, decl in schema["routines"][routine]["declarations"].items()
                    if decl["dimensions"] and decl["intent"] in ("out", "inout")}
        assert set(coverage) == expected, routine
        fixture_routine = {"small_step_prep": "prep", "small_step_finish": "finish",
                           "calc_p_rho": "prep", "sumflux": "sumflux"}[routine]
        for filename in _names(fixture_routine):
            fixture = load_bookkeeping(BOOKKEEPING_DIR / filename)
            assert all("ref_" + key in fixture for key in coverage.values()), (routine, filename)


def _assert_measurement(name, routine):
    pins = json.loads((BOOKKEEPING_DIR / "measurements.json").read_text())
    fixture = load_bookkeeping(BOOKKEEPING_DIR / name)
    actual = PORT_RUNNERS[routine](fixture)
    assert all(key[4:] in actual for key in fixture if key.startswith("ref_"))
    row = {"routine": routine, "native": {}, "isolation": {}}
    for key, value in fixture.items():
        if key.startswith("ref_") and key[4:] in actual:
            row["native"][key[4:]] = word_metrics(actual[key[4:]], value)
        if key.startswith("isolation_") and key[10:] in actual:
            row["isolation"][key[10:]] = word_metrics(actual[key[10:]], value)
    row["causal_trace"] = {key: word_metrics(actual[key], value)
                           for key, value in ROUNDING_TRACES[routine](fixture).items()}
    assert all(metric["different_words"] == 0
               for metric in row["causal_trace"].values())
    assert row == pins["files"][name], name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", _names("prep"))
def test_small_step_prep_against_compiled_wrf471(name):
    _assert_measurement(name, "prep")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", _names("finish"))
def test_small_step_finish_against_compiled_wrf471(name):
    _assert_measurement(name, "finish")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", _names("sumflux"))
def test_sumflux_against_compiled_wrf471(name):
    _assert_measurement(name, "sumflux")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", _names("emdiv"))
def test_apply_emdiv_against_compiled_wrf471(name):
    _assert_measurement(name, "emdiv")
