"""The WSM6 REFL_10CM melting-particle switch and its MPAS-seam default.

Defect: hex (the MPAS column seam) wrote WSM6's wet-snow bright band into
every REFL_10CM frame.  Measured on conus3km 2026-10-03 12Z f01, the term
added a median +9 dB at the composite maximum and tripled 35 dBZ coverage
against MRMS.  NOAA's own MPAS ships the term off
(``config_tempo_refl10cm_from_melting`` default false), so the seam's state
now defaults to off while an ARW DomainState keeps WRF's routine.  The
rule lives in refl.py; gpuwm/core/mpas_column_batch.py is byte-pinned by
hex's engine manifest and is deliberately unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpuwm.core import mpas_column_batch as mcb
from gpuwm.verify.npref import np_refl10cm_wsm6_column


def _melting_column(nz=30):
    """A stratiform column: snow aloft, a 0 C level near k=12, rain below."""
    z = np.linspace(100.0, 9000.0, nz)
    t = 288.0 - 0.0065 * z                       # 0 C near 2300 m
    p = 100000.0 * np.exp(-z / 8000.0)
    qv = np.full(nz, 5.0e-3)
    qr = np.where(t > 273.15, 2.0e-4, 0.0)
    # snow aloft, thinning through the melting layer down to +4 C
    qs = np.where(t <= 273.15, 4.0e-4,
                  np.clip(4.0e-4 * (277.15 - t) / 4.0, 0.0, None))
    qg = np.where((t > 260.0) & (t <= 273.15), 3.0e-5, 0.0)
    return qv, qr, qs, qg, t, p


def _dry_reference(qv, qr, qs, qg, t, p, hail_opt=0):
    """Rain + dry snow + dry graupel, written from mp_wsm6.F90 directly."""
    from gpuwm.core.wsm6_constants import rimed_ice_constants
    pi = 3.1415926535897932384626434
    r = rimed_ice_constants(hail_opt)
    xam_r, xam_s, xam_g = pi * 1000.0 / 6.0, 100.0 * pi / 6.0, r.deng * pi / 6.0
    out = []
    for k in range(t.size):
        rho = 0.622 * p[k] / (287.0 * t[k] * (max(1e-10, qv[k]) + 0.622))
        ze = [1e-22, 1e-22, 1e-22]
        if qr[k] > 1e-9:
            lam = (xam_r * 6.0 * 8e6 / (qr[k] * rho)) ** 0.25
            ze[0] = 8e6 * 720.0 * lam ** -7
        fac = (0.176 / 0.93) * (6.0 / pi) ** 2
        if qs[k] > 1e-9:
            n0s = min(1e11, 2e6 * math.exp(-0.12 * min(-0.001, t[k] - 273.15)))
            lam = (xam_s * 6.0 * n0s / (qs[k] * rho)) ** 0.25
            ze[1] = fac * (xam_s / 900.0) ** 2 * n0s * 720.0 * lam ** -7
        if qg[k] > 1e-9:
            lam = (xam_g * 6.0 * r.n0g / (qg[k] * rho)) ** 0.25
            ze[2] = fac * (xam_g / 900.0) ** 2 * r.n0g * 720.0 * lam ** -7
        out.append(max(-35.0, 10.0 * math.log10(sum(ze) * 1e18)))
    return np.array(out)


# --------------------------------------------------------------------------
# float64 mirror
# --------------------------------------------------------------------------

def test_mirror_default_keeps_the_wrf_melting_term():
    col = _melting_column()
    assert np.array_equal(np_refl10cm_wsm6_column(*col),
                          np_refl10cm_wsm6_column(*col, melting=True))


def test_mirror_without_melting_is_the_dry_rayleigh_sum():
    col = _melting_column()
    got = np_refl10cm_wsm6_column(*col, melting=False)
    np.testing.assert_allclose(got, _dry_reference(*col), atol=1e-9)


def test_the_melting_term_only_raises_levels_below_the_melting_level():
    qv, qr, qs, qg, t, p = col = _melting_column()
    wet = np_refl10cm_wsm6_column(*col)
    dry = np_refl10cm_wsm6_column(*col, melting=False)
    below = (t > 273.15) & (qs > 1e-9)
    assert below.any()
    # the bright band: several dB at the snow-bearing levels below 0 C
    assert np.max(wet[below] - dry[below]) > 3.0
    # nothing changes at or above the melting level
    np.testing.assert_array_equal(wet[t <= 273.15], dry[t <= 273.15])


def test_a_column_without_rain_never_melts():
    qv, qr, qs, qg, t, p = _melting_column()
    col = (qv, np.zeros_like(qr), qs, qg, t, p)
    assert np.array_equal(np_refl10cm_wsm6_column(*col),
                          np_refl10cm_wsm6_column(*col, melting=False))


# --------------------------------------------------------------------------
# the MPAS seam default
# --------------------------------------------------------------------------

def test_the_seam_state_class_lives_where_the_rule_looks():
    """The default keys on the seam state's defining module; a rename of
    either side must fail here, not silently restore the bright band."""
    from gpuwm.core.refl import MPAS_SEAM_STATE_MODULE
    assert mcb._ColumnBatchState.__module__ == MPAS_SEAM_STATE_MODULE


def _state_of(module, **attrs):
    cls = type("_ColumnBatchState", (), {"__module__": module})
    obj = cls()
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def test_the_mpas_seam_state_defaults_to_no_melting_term():
    from gpuwm.core.refl import wsm6_refl_melting_for
    assert wsm6_refl_melting_for(
        _state_of("gpuwm.core.mpas_column_batch")) is False


def test_every_other_state_keeps_the_wrf_term():
    from gpuwm.core.refl import wsm6_refl_melting_for
    assert wsm6_refl_melting_for(_state_of("gpuwm.core.state")) is True


def test_an_explicit_switch_wins_on_either_side():
    from gpuwm.core.refl import wsm6_refl_melting_for
    seam = "gpuwm.core.mpas_column_batch"
    assert wsm6_refl_melting_for(
        _state_of(seam, refl10cm_from_melting=True)) is True
    assert wsm6_refl_melting_for(
        _state_of("gpuwm.core.state", refl10cm_from_melting=False)) is False


# --------------------------------------------------------------------------
# device path (GPU)
# --------------------------------------------------------------------------

def test_device_no_melt_path_matches_the_float64_mirror():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - CPU-only host
        pytest.skip("no CUDA device")
    from gpuwm.core.refl import launch_refl10cm_wsm6

    rng = np.random.default_rng(3)
    nz, ny, nx = 30, 4, 9
    base = _melting_column(nz)
    fields = []
    for i, a in enumerate(base):
        a3 = np.repeat(np.repeat(a[:, None, None], ny, 1), nx, 2)
        if i in (1, 2, 3):                      # perturb the species
            a3 = a3 * rng.uniform(0.0, 3.0, a3.shape)
        fields.append(np.ascontiguousarray(a3, dtype=np.float32))
    dev = [cp.asarray(f) for f in fields]
    for melting in (False, True):
        refl = cp.zeros((nz, ny, nx), dtype=cp.float32)
        launch_refl10cm_wsm6(*dev, refl, melting=melting)
        got = cp.asnumpy(refl)
        for j in range(ny):
            for i in range(nx):
                ref = np_refl10cm_wsm6_column(
                    *(f[:, j, i] for f in fields), melting=melting)
                np.testing.assert_allclose(got[:, j, i], ref, atol=2e-3)


# --------------------------------------------------------------------------
# who decides: None is not an answer
# --------------------------------------------------------------------------

def test_a_permissive_view_answering_none_keeps_the_wrf_term():
    from gpuwm.core.refl import wsm6_refl_melting_for

    class Permissive:
        """Answers None for every unknown name, like hex's f00 view."""

        def __getattr__(self, name):
            return None

    assert wsm6_refl_melting_for(Permissive()) is True


# --------------------------------------------------------------------------
# Thompson: the same rule, its own no-melt path (GPU)
# --------------------------------------------------------------------------

def _thompson_fields(with_rain_below=True):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - CPU-only host
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(11)
    nz, ny, nx = 30, 3, 7
    qv, qr, qs, qg, t, p = _melting_column(nz)
    if not with_rain_below:
        qr = np.zeros_like(qr)
    nr = np.where(qr > 0, 5.0e3, 0.0)
    ng = np.where(qg > 0, 2.0e3, 0.0)
    out = []
    for i, a in enumerate((qv, qr, nr, qs, qg, ng, t, p)):
        a3 = np.repeat(np.repeat(a[:, None, None], ny, 1), nx, 2)
        if i in (1, 2, 3, 4, 5):
            a3 = a3 * rng.uniform(0.2, 3.0, a3.shape)
        out.append(cp.asarray(np.ascontiguousarray(a3, dtype=np.float32)))
    return cp, out, t


def test_thompson_dry_path_equals_the_frozen_kernel_when_nothing_melts():
    from gpuwm.core.refl import launch_refl10cm_thompson
    cp, f, _ = _thompson_fields(with_rain_below=False)
    wet = cp.zeros_like(f[0]); dry = cp.zeros_like(f[0])
    launch_refl10cm_thompson(*f, wet, melting=True)
    launch_refl10cm_thompson(*f, dry, melting=False)
    np.testing.assert_allclose(cp.asnumpy(dry), cp.asnumpy(wet), atol=1e-4)


def test_thompson_dry_path_only_drops_the_melting_layer_boost():
    from gpuwm.core.refl import launch_refl10cm_thompson
    cp, f, t = _thompson_fields(with_rain_below=True)
    wet = cp.zeros_like(f[0]); dry = cp.zeros_like(f[0])
    launch_refl10cm_thompson(*f, wet, melting=True)
    launch_refl10cm_thompson(*f, dry, melting=False)
    w, d = cp.asnumpy(wet), cp.asnumpy(dry)
    above = t <= 273.15
    np.testing.assert_allclose(d[above], w[above], atol=1e-4)
    assert np.all(d[~above] <= w[~above] + 1e-4)
    assert np.max(w[~above] - d[~above]) > 1.0
