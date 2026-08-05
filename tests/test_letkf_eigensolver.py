"""The filter's choice of eigensolver: the switch, and the A/B across it.

``gpuwm.da.letkf`` reached exactly one linear-algebra library, at exactly one
call, and it now has its own kernel for that call.  These tests hold the seam:
the mode is validated and recorded, the default needs no CUDA library, and the
ANALYSIS -- not the eigendecomposition, the analysis -- is the same either way.

The CPU tests here import no cupy and run in the ``-m "not gpu"`` tier;
``tests/conftest.py`` marks the device ones from their own imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da.letkf import (
    EIGENSOLVER_MODES, GriddedObs, LetkfConfig, LetkfDiagnostics, LetkfError,
    Localization, analyze,
)
from test_letkf import _tiny_case


def _cfg(fields, **kwargs):
    kwargs.setdefault("rtps_alpha", 0.6)
    return LetkfConfig(
        localization=Localization(horizontal_m=3500.0, vertical_m=1500.0),
        analysis_fields=fields, **kwargs)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_the_modes_are_the_three_documented_ones():
    assert EIGENSOLVER_MODES == ("auto", "jacobi", "library")


def test_an_unknown_mode_is_refused_at_construction():
    with pytest.raises(LetkfError, match="eigensolver must be one of"):
        _cfg(("theta",), eigensolver="cusolver")


def test_the_default_is_auto_and_auto_means_numpy_on_the_host():
    """No CUDA library is involved in a host analysis, and never was.

    Worth pinning: ``--solve-device host`` is the documented fallback for a
    box whose device linear algebra is broken, and it would stop being one if
    ``auto`` ever tried to reach the kernel from numpy.
    """
    fields = ("theta",)
    assert _cfg(fields).eigensolver == "auto"
    grid, prior, obs, members, shape = _tiny_case(fields=fields)
    diag = LetkfDiagnostics()
    analyze(prior, [obs], grid, _cfg(fields), diag)
    assert diag.eigensolver == "library"
    assert diag.max_jacobi_sweeps == 0


def test_requiring_the_kernel_from_numpy_refuses_and_says_why():
    """Not silently satisfied by LAPACK: 'jacobi' names a specific kernel."""
    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(fields=fields)
    with pytest.raises(LetkfError, match="needs the analysis on the device"):
        analyze(prior, [obs], grid, _cfg(fields, eigensolver="jacobi"))


def test_the_library_failure_message_names_cusolver_and_the_way_out():
    """The masquerade, spelled out where it is actually raised.

    A missing cuSOLVER surfaces here as an ordinary exception from ``eigh``
    after every elementwise operation in the analysis has already succeeded.
    The message has to say that the rest working proves nothing, or the next
    reader spends the evening on the GPU rather than on one library.
    """
    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(fields=fields)

    def boom(_a):
        raise RuntimeError("CUSOLVER_STATUS_NOT_INITIALIZED")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(np.linalg, "eigh", boom)
        with pytest.raises(LetkfError) as caught:
            analyze(prior, [obs], grid, _cfg(fields, eigensolver="library"))
    text = str(caught.value)
    assert "cuSOLVER" in text
    assert "gpuwm doctor" in text
    assert "eigensolver='auto'" in text
    assert "do not treat the prior as one" in text


def test_a_requested_size_the_kernel_cannot_take_is_refused_by_number():
    """Above the supported ensemble size, 'jacobi' refuses instead of drifting."""
    import gpuwm.da.letkf as letkf

    # ``object()`` stands in for a cupy namespace without needing a device:
    # the resolver only asks "is this numpy?" and then "does the kernel take
    # R?", and the second question is answered on the host.
    with pytest.raises(LetkfError, match="2 <= R <= 64"):
        letkf._resolve_eigensolver(
            object(), 200, np.dtype(np.float64),
            _cfg(("theta",), eigensolver="jacobi"))


def test_auto_falls_back_rather_than_refusing_when_the_kernel_cannot_take_it():
    import gpuwm.da.letkf as letkf

    which = letkf._resolve_eigensolver(
        object(), 200, np.dtype(np.float64), _cfg(("theta",)))
    assert which == "library"


# ---------------------------------------------------------------------------
# The A/B that matters: the ANALYSIS, not the eigenvectors
# ---------------------------------------------------------------------------

def test_the_two_solvers_produce_the_same_analysis():
    """The end-to-end claim, on the device, through the whole transform.

    Not an eigenvector comparison -- see tests/test_jacobi_eigh_gpu.py for
    why that would be meaningless under the degeneracy this matrix carries.
    This runs the entire analysis twice on identical device inputs, changing
    nothing but which kernel factors ``(R-1)I/rho + C Yb``, and compares the
    increments the filter hands its caller.

    The tolerance is the repo's scale-relative convention -- worst absolute
    difference over the field's own magnitude -- because increments cancel to
    near zero at most gridpoints and an elementwise ``rtol`` there measures
    rounding against rounding.
    """
    cp = pytest.importorskip("cupy")

    fields = ("theta", "qv", "u")
    grid, prior, obs, members, shape = _tiny_case(
        members=10, nz=4, ny=20, nx=20, n_obs=40, fields=fields)
    device_prior = {f: cp.asarray(v) for f, v in prior.items()}
    device_obs = GriddedObs(
        name=obs.name, values=cp.asarray(obs.values), errors=obs.errors,
        simulated=cp.asarray(obs.simulated), mask=cp.asarray(obs.mask))

    out = {}
    diags = {}
    for mode in ("jacobi", "library"):
        diags[mode] = LetkfDiagnostics()
        out[mode] = analyze(
            device_prior, [device_obs], grid,
            _cfg(fields, chunk_points=512, eigensolver=mode), diags[mode])

    assert diags["jacobi"].eigensolver == "jacobi"
    assert diags["library"].eigensolver == "library"
    assert 0 < diags["jacobi"].max_jacobi_sweeps < 20
    assert diags["jacobi"].active_points == diags["library"].active_points

    for f in fields:
        mine = cp.asnumpy(out["jacobi"][f])
        theirs = cp.asnumpy(out["library"][f])
        scale = max(float(np.abs(theirs).max()), 1e-30)
        assert float(np.abs(mine - theirs).max()) / scale < 1e-11, f
        # The localisation guarantee is structural, not numerical: a
        # gridpoint outside every cutoff must be EXACTLY zero under both.
        assert np.array_equal(mine == 0.0, theirs == 0.0), f


def test_auto_prefers_the_kernel_on_the_device():
    """The point of the exercise: a stock DA run reaches no linear-algebra library."""
    cp = pytest.importorskip("cupy")

    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(members=10, fields=fields)
    diag = LetkfDiagnostics()
    analyze({f: cp.asarray(v) for f, v in prior.items()},
            [GriddedObs(name=obs.name, values=cp.asarray(obs.values),
                        errors=obs.errors, simulated=cp.asarray(obs.simulated),
                        mask=cp.asarray(obs.mask))],
            grid, _cfg(fields), diag)
    assert diag.eigensolver == "jacobi"


def test_the_device_analysis_is_bitwise_reproducible_under_the_kernel():
    """Same inputs, same bytes, run after run -- the project's own standard."""
    cp = pytest.importorskip("cupy")

    fields = ("theta", "qv")
    grid, prior, obs, members, shape = _tiny_case(
        members=10, nz=4, ny=16, nx=16, n_obs=30, fields=fields)
    device_prior = {f: cp.asarray(v) for f, v in prior.items()}
    device_obs = GriddedObs(
        name=obs.name, values=cp.asarray(obs.values), errors=obs.errors,
        simulated=cp.asarray(obs.simulated), mask=cp.asarray(obs.mask))
    cfg = _cfg(fields, eigensolver="jacobi")

    first = analyze(device_prior, [device_obs], grid, cfg)
    baseline = {f: cp.asnumpy(v).tobytes() for f, v in first.items()}
    for _ in range(3):
        again = analyze(device_prior, [device_obs], grid, cfg)
        for f in fields:
            assert cp.asnumpy(again[f]).tobytes() == baseline[f], f


def test_the_kernel_adds_no_chunk_sensitivity_of_its_own():
    """Re-chunking moves the device analysis by an ulp -- and it always did.

    Worth pinning precisely, because the obvious claim is wrong in an
    interesting way.  The EIGENSOLVER is chunk-invariant: one block per
    matrix, no cross-matrix communication, and
    ``tests/test_jacobi_eigh_gpu.py`` proves a matrix solves to the same
    bytes wherever it sits in the batch.  The ANALYSIS around it is not,
    because CuPy picks reduction and batched-GEMM kernels by array shape, so
    ``s.mean(axis=0)`` and ``cmat @ yb`` sum in a different order when the
    chunk changes.  That is upstream of any eigensolver: the same measurement
    under ``eigensolver='library'`` moves by the same amount.

    So what is asserted is the thing that would actually be a regression --
    that swapping cuSOLVER for this kernel did not make the chunk knob any
    more of a numerical one than it already was -- and the size of the
    pre-existing wobble, so that a real growth in it is visible.
    """
    cp = pytest.importorskip("cupy")

    fields = ("theta",)
    grid, prior, obs, members, shape = _tiny_case(
        members=10, nz=4, ny=16, nx=16, n_obs=30, fields=fields)
    device_prior = {f: cp.asarray(v) for f, v in prior.items()}
    device_obs = GriddedObs(
        name=obs.name, values=cp.asarray(obs.values), errors=obs.errors,
        simulated=cp.asarray(obs.simulated), mask=cp.asarray(obs.mask))

    spread = {}
    for mode in ("jacobi", "library"):
        runs = [cp.asnumpy(analyze(
            device_prior, [device_obs], grid,
            _cfg(fields, chunk_points=chunk, eigensolver=mode))["theta"])
            for chunk in (37, 256, 4096)]
        scale = max(float(np.abs(runs[0]).max()), 1e-30)
        spread[mode] = max(float(np.abs(r - runs[0]).max()) for r in runs[1:])
        spread[mode] /= scale
        # Whatever the chunking does, it does not reach the structural zeros.
        for run in runs[1:]:
            assert np.array_equal(run == 0.0, runs[0] == 0.0)

    assert spread["jacobi"] < 1e-14, spread
    # The kernel is not allowed to be MORE chunk-sensitive than cuSOLVER was.
    # Equal to within a factor of two is what "the same wobble" means here.
    assert spread["jacobi"] <= 2.0 * max(spread["library"], 1e-18), spread
