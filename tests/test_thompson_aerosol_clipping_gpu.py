"""The mp=28 conservation limiters and the pair re-enforcement, in WRF's REAL.

WHAT THIS GATES
---------------
module_mp_thompson.F (v4.6.1; the HRRR fork declares the same) holds the
conservation limiters' ``sump``, ``rate_max`` and ``ratio`` in REAL (:1615):
each DOUBLE sum of process rates is rounded to float32, compared in float32
with ``-r*odts``, and the float32 quotient widens to rescale the DOUBLE
rates (:2878-2941).  The cloud tendency is then the sum of the RESCALED
rates (:2987).  Blossey's re-enforcement (:2945-2954) rounds the paired
magnitude to REAL as well.

The breakage these tests prevent, measured on the mp=28 column oracle
(tools/thompson_aerosol_column_oracle) against WRF v4.6.1's own Fortran:

* the warm network rescaled with a DOUBLE ratio and then replaced the cloud
  sink by the limit itself; where a step drains the whole cloud, :3975 forms
  ``qc1d + qcten*DT`` as the difference of two nearly equal numbers, so the
  rain and the cloud left behind moved (7 rain cells of the edge column with
  masses and no numbers at dt = 20 s);
* the re-enforced rain-graupel and rain-snow pairs kept the DOUBLE
  magnitude (and the cold network forced the rain rate negative):
  prr_rcg / prg_rcg differed from WRF at 372 of 372 active cells,
  prr_rcs / prs_rcs at 30 of 113 (host bitwise view, 153 columns).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _f32(values):
    import cupy as cp
    return cp.asarray(np.asarray(values, dtype=np.float32))


def _reenforce_reference(a, b):
    """:2945-2947 in NumPy: ratio = REAL(MIN(ABS(a), ABS(b))),
    a = ratio * SIGN(1.0, SNGL(a)), b = -a."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ratio = np.minimum(np.abs(a), np.abs(b)).astype(np.float32)
    with np.errstate(under="ignore", over="ignore"):
        sign = np.copysign(np.float32(1.0), a.astype(np.float32))
    a_new = (ratio * sign).astype(np.float32).astype(np.float64)
    return a_new, -a_new


def test_reenforce_pair_rounds_the_magnitude_to_real_and_keeps_a_sign():
    import cupy as cp

    from gpuwm.core.thompson_aerosol_launch import probe_reenforce_pair

    rng = np.random.default_rng(20261007)
    n = 200000
    # Normal float32 magnitudes: the default (non-strict) build flushes
    # float32 subnormals, the strict one does not.
    mag = 10.0 ** rng.uniform(-37.0, -2.0, n)
    a = mag * rng.choice([-1.0, 1.0], n)
    b = -a * (1.0 + rng.uniform(-0.3, 0.3, n))
    # Equal magnitudes, zeros of both signs, and magnitudes whose float32
    # rounding underflows to a signed zero.
    a[:8] = [0.0, -0.0, 1.0e-50, -1.0e-50, 3.0e-7, -3.0e-7, 0.0, -2.0e-9]
    b[:8] = [5.0e-9, 5.0e-9, -4.0e-9, 4.0e-9, -3.0e-7, 3.0e-7, -0.0, 0.0]
    got_a, got_b = probe_reenforce_pair(cp.asarray(a), cp.asarray(b))
    want_a, want_b = _reenforce_reference(a, b)
    assert np.array_equal(cp.asnumpy(got_a).view(np.int64),
                          want_a.view(np.int64))
    assert np.array_equal(cp.asnumpy(got_b).view(np.int64),
                          want_b.view(np.int64))
    # The double magnitude this replaces really is a different number.
    keep = np.minimum(np.abs(a), np.abs(b))
    assert np.count_nonzero(np.abs(want_a) != keep) > n // 2


def _drained_warm_column(n=64):
    """Liquid only, warm, heavy rain over a cloud whose entry content steps
    by 1.37 per level, so the cloud limiter fires on part of the column
    and the entry mantissa differs at every level."""
    pressure = np.linspace(95000.0, 60000.0, n).astype(np.float32)
    temperature = np.linspace(292.0, 276.0, n).astype(np.float32)
    qv = np.full(n, 0.010, dtype=np.float32)
    rho = (np.float32(0.622) * pressure
           / (np.float32(287.04) * temperature
              * (qv + np.float32(0.622)))).astype(np.float32)
    qc = np.minimum(1.0e-7 * 1.2 ** np.arange(n), 2.0e-3).astype(np.float32)
    # 20 g/kg of rain at a 0.6 mm volume diameter: accretion reaches its
    # own cap rc*odts and autoconversion on top of it trips the limiter.
    qr = np.full(n, 2.0e-2, dtype=np.float32)
    nr = (qr * (3.672 / 6.0e-4) ** 3 / (np.pi * 1000.0)).astype(np.float32)
    nc = (1.0e8 / rho).astype(np.float32)
    rng = np.random.default_rng(7)
    nwfa = (rng.uniform(1.0e8, 2.0e9, n) / rho).astype(np.float32)
    nifa = (rng.uniform(1.0e4, 1.0e6, n) / rho).astype(np.float32)
    return dict(pressure=pressure, temperature=temperature, qv=qv, rho=rho,
                qc=qc, qr=qr, nr=nr, nc=nc, nwfa=nwfa, nifa=nifa)


def test_warm_cloud_limiter_is_wrfs_real_arithmetic_on_the_rescaled_sum():
    import cupy as cp

    from test_thompson_aerosol_warm_gpu import (
        _t_efrw, _t_efsw, _zero_collision_tables)

    from gpuwm.core.thompson_aerosol_warm import (
        launch_aerosol_warm_source_network, probe_warm_rates)

    dt = 20.0
    col = _drained_warm_column()
    n = col["pressure"].size
    rates = probe_warm_rates(
        _f32(col["pressure"]), _f32(col["temperature"]), _f32(col["qv"]),
        _f32(col["qc"]), _f32(col["nc"]), _f32(col["qr"]), _f32(col["nr"]),
        _f32(col["nwfa"]), _f32(col["nifa"]), _t_efrw(), dt)
    prr_wau = cp.asnumpy(rates["prr_wau"])
    prr_rcw = cp.asnumpy(rates["prr_rcw"])

    qcten = _f32(np.zeros(n))
    qrten = _f32(np.zeros(n))
    nrten = _f32(np.zeros(n))
    rain_snow, rain_graupel = _zero_collision_tables()
    launch_aerosol_warm_source_network(
        _f32(col["qc"]), _f32(col["qr"]), _f32(col["nr"]),
        _f32(np.zeros(n)), _f32(np.zeros(n)), _f32(np.zeros(n)),
        _f32(np.ones(n)), _f32(np.zeros(n)),
        _f32(col["temperature"]), _f32(col["pressure"]), _f32(col["qv"]),
        _f32(col["nc"]), _f32(col["nwfa"]), _f32(col["nifa"]),
        _f32(np.zeros(n)), _f32(np.zeros(n)), _f32(np.zeros(n)),
        _t_efrw(), _t_efsw(), rain_snow, rain_graupel, dt,
        qcten=qcten, qrten=qrten, nrten=nrten)
    got = cp.asnumpy(qcten)

    # :2878-2890 and :2987, transcribed.  At a warm liquid-only level
    # pri_wfz, prs_scw, prg_scw and prg_gcw are zero.
    rho = col["rho"]
    orho = (np.float32(1.0) / rho).astype(np.float32)
    odts = np.float32(1.0) / np.float32(dt)
    rc = (col["qc"] * rho).astype(np.float32)
    sump = (-(prr_wau + prr_rcw)).astype(np.float32)
    rate_max = (-(rc * odts)).astype(np.float32)
    fire = (sump < rate_max) & (col["qc"] > np.float32(1.0e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(fire, rate_max / sump, np.float32(1.0)).astype(
            np.float32)
    wau = np.where(fire, prr_wau * ratio.astype(np.float64), prr_wau)
    rcw = np.where(fire, prr_rcw * ratio.astype(np.float64), prr_rcw)
    want = (0.0 - (wau + rcw) * orho.astype(np.float64)).astype(np.float32)

    assert np.count_nonzero(fire) >= 5, "the column must drain its cloud"
    assert np.count_nonzero(~fire & (prr_rcw > 0.0)) >= 8
    assert np.array_equal(got.view(np.int32), want.view(np.int32)), (
        np.flatnonzero(got.view(np.int32) != want.view(np.int32)))

    # The form this replaced (DOUBLE ratio, the limit substituted for the
    # sink) gives a different cloud tendency on this column.
    limit = (rc * odts).astype(np.float32).astype(np.float64)
    old = np.where(fire, limit, prr_wau + prr_rcw)
    old_qcten = (0.0 - old * orho.astype(np.float64)).astype(np.float32)
    assert np.any(old_qcten.view(np.int32) != want.view(np.int32))
