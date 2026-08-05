"""``gpuwm doctor``'s verdict on the eigensolver the radar-DA analysis needs.

This line exists because of a specific field failure, twice: the analysis died
inside cuSOLVER while every other piece of evidence said the GPU was healthy.
It was healthy -- cupy's core, its elementwise kernels and its reductions all
load independently of cuBLAS/cuSOLVER, so an install can multiply arrays all
day and still be unable to factor one.  A check that only proves "cupy
imports" reproduces the masquerade instead of catching it.

Every branch is forced here rather than left to whatever this box happens to
have, because on a working box only one of the four is reachable and the other
three are exactly the ones a reader in trouble will see.
"""

from __future__ import annotations

import pytest

from gpuwm import doctor


def _check(monkeypatch, payload):
    monkeypatch.setattr(doctor, "_eigensolver_probe", lambda: payload)
    return doctor._da_eigensolver_check()


def test_the_probe_runs_the_operation_rather_than_testing_for_a_name():
    """The probe must SOLVE, not import.  Nothing cheaper detects this fault."""
    source = doctor._EIGENSOLVER_PROBE
    assert "batched_eigh" in source
    assert "cupy.linalg.eigh" in source
    # And it must check the answer, not merely that the call returned.
    assert "residual" in source


def test_both_solvers_working_is_verified_and_prints_no_remedy(monkeypatch):
    check = _check(monkeypatch, {"jacobi": "ok", "library": "ok"})
    assert check.status == "verified"
    assert check.remedy is None and check.action is None
    assert not check.blocking


def test_no_cusolver_is_still_verified_and_says_the_default_does_not_need_it(
        monkeypatch):
    """THE branch this check was written for.

    A box with no cuSOLVER runs data assimilation perfectly well now, so the
    honest verdict is "verified", with the absence reported as news and the
    install printed for the cases that still want it.  Calling it a failure
    would train the reader to ignore the line.
    """
    check = _check(monkeypatch, {
        "jacobi": "ok",
        "library": "RuntimeError: CUSOLVER_STATUS_NOT_INITIALIZED"})
    assert check.status == "verified"
    assert not check.blocking
    assert "cuSOLVER unavailable" in check.detail
    assert "the default analysis does not need" in check.detail
    assert check.remedy and "nvidia-cusolver-cu12" in check.remedy


def test_the_bundled_kernel_failing_is_reported_even_when_cusolver_works(
        monkeypatch):
    """The reverse fault must not hide behind a working fallback."""
    check = _check(monkeypatch, {
        "jacobi": "CompileException: nvrtc said no", "library": "ok"})
    assert check.status == "present"
    assert "cuSOLVER works" in check.detail
    assert "eigensolver='library'" in check.remedy
    assert not check.blocking


def test_neither_solver_working_names_both_failures_and_spares_forecasts(
        monkeypatch):
    check = _check(monkeypatch, {"jacobi": "boom", "library": "bang"})
    assert check.status == "missing"
    assert "boom" in check.detail and "bang" in check.detail
    # A forecast-only install must not read this as its problem.
    assert "forecasts are unaffected" in check.detail
    assert not check.blocking, (
        "an install that never assimilates radar must not exit nonzero")


def test_without_cupy_the_line_declines_to_judge(monkeypatch):
    check = _check(monkeypatch, {"cupy": "not installed"})
    assert check.status == "info"
    assert check.remedy is None
    assert "not judged" in check.detail


def test_a_probe_that_cannot_run_is_reported_as_a_gap_not_as_a_pass(
        monkeypatch):
    check = _check(monkeypatch, {"probe": "did not run: timeout"})
    assert check.status == "missing"
    assert "could not be run" in check.detail


def test_the_check_is_in_the_default_estate(monkeypatch):
    """A check behind a flag is a check nobody runs."""
    monkeypatch.setattr(doctor, "_eigensolver_probe",
                        lambda: {"jacobi": "ok", "library": "ok"})
    names = [c.name for c in doctor.collect_checks(sources=())]
    assert "radar-DA eigensolver" in names
    # Immediately after the cupy line, so the two read together.
    assert names.index("radar-DA eigensolver") == names.index(
        "cupy (GPU runtime)") + 1


@pytest.mark.parametrize("payload", [
    {"jacobi": "ok", "library": "nope"},
    {"jacobi": "nope", "library": "ok"},
    {"jacobi": "nope", "library": "nope"},
    {"probe": "did not run"},
])
def test_every_branch_that_prints_a_remedy_prints_a_runnable_one(
        monkeypatch, payload):
    """The closing line of the report claims this of every remedy it prints."""
    check = _check(monkeypatch, payload)
    if not check.remedy:
        return
    for line in check.remedy.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert stripped.split()[0] in {"pip", "python", "gpuwm", "set"}, line
