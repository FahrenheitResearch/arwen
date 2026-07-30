# tests/test_nest_interp.py
"""Phase 5 Task 10: parent->child nest interpolation operators (lane L3).

Acceptance targets are the pre-registered N1 operator-oracle gates in
gpuwm/verify/nest_gates.py: sint_vs_fp64_mirror, bdy_interp1_vs_fp64_mirror
(REAL*8-rdt scheme included), blend_terrain_vs_fp64_mirror,
adjust_tempqv_vs_fp64_mirror, copy_fcn_vs_fp64_mirror (BOTH parity
branches), sint_quadratic_reproduction, sint_parent_coincident_odd_ratio --
all fp64_mirror_floor comparisons at FP64_MIRROR_MAX_ULPS, plus the plan Task-10
fixtures: the REAL*8-rdt discriminating fixture, the F7 ratio-not-1
tendency-divisor fixture, the F6 limiter-activation fixture, the stagger
ioff/joff pins, the blend-weight table pin, and the width-formula sz=5 pin.

Every hand-derived expectation below is computed from the WRF v4.6.1
Fortran itself (bundle share/sint.F, share/interp_fcn.F,
dyn_em/nest_init_utils.F) -- exact rational arithmetic for SINT, literal
step-by-step transcription elsewhere -- never from the npref mirrors.

Kernel-vs-mirror parity tests are ``gpu``-marked (controller-run); the
kernel source additionally gets an offline NVRTC compile check that needs
no device.

Fix-round additions (Fable review NEEDS_FIXES(2) + codex shadow
FINDINGS(2)): the mirror's WRF REAL difference regression
(``test_bdy_mirror_forms_real_difference``), the sign-changing-
cancellation regression pinning where the FP64 oracle stops being the
comparator's mirror and the REAL statement emulation takes over
(``test_sint_fp64_oracle_cancellation_and_real_emulation`` + gpu
companion), fail-loud ``out=`` buffer validation, the explicit
idempotent-checked geometry binding contract (F4), and the mirror-side
bdy-wrapper guard.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.nest_interp import (adjust_tempqv, bdy_width, blend_terrain,
                                    register_nest, sint, sint_offsets,
                                    _wrapper_offset)
from gpuwm.verify import nest_gates as ng
from gpuwm.verify.npref import (np_adjust_tempqv, np_bdy_interp1,
                                np_bdy_width, np_blend_terrain, np_copy_fcn,
                                np_copy_fcnm, np_copy_fcni, np_sint)

MAX_ULPS = ng.FP64_MIRROR_MAX_ULPS


def _lex32(f):
    """Monotone integer key of FP32 bit patterns (for ULP distances)."""
    return monotone_fp32_key(np.asarray(f, np.float32).reshape(-1))


def ulp32(a, b):
    """Max elementwise ULP distance between two arrays, compared as FP32."""
    a32 = np.asarray(a, np.float32)
    b32 = np.asarray(b, np.float32)
    assert np.isfinite(a32).all() and np.isfinite(b32).all(), \
        "NaN/Inf FAILS the split ULP comparators (nest_gates F10)"
    return int(np.max(np.abs(_lex32(a32) - _lex32(b32)), initial=0))


def _mass_reg(nri=3, ipos=4, jpos=3, child=9, parent=12, stagger="",
              wrapper="interp"):
    return register_nest(nri=nri, nrj=nri, i_parent_start=ipos,
                         j_parent_start=jpos, child_nx=child, child_ny=child,
                         parent_nx=parent, parent_ny=parent,
                         stagger=stagger, wrapper=wrapper)


def test_numpy_child_materialization_operators_do_not_launch_cuda(monkeypatch):
    """CPU hierarchy finalization must stay on host through child blending."""
    import gpuwm.core.nest_interp as module

    monkeypatch.setattr(
        module, "_kernel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("NumPy child setup attempted a CUDA kernel")))
    reg = _mass_reg()
    parent = np.linspace(
        0.5, 3.0, 2 * reg.nyp * reg.nxp,
        dtype=np.float32).reshape(2, reg.nyp, reg.nxp)
    got = sint(parent, reg)
    expected = np.asarray(np_sint(parent, reg, dtype=np.float32),
                          dtype=np.float32)
    np.testing.assert_array_equal(got, expected)

    coarse = np.arange(18 * 18, dtype=np.float32).reshape(18, 18)
    fine = np.full((18, 18), np.float32(500.0))
    expected_blend = np.asarray(np_blend_terrain(coarse, fine),
                                dtype=np.float32)
    blend_terrain(coarse, fine)
    np.testing.assert_array_equal(fine, expected_blend)

    nz, ny, nx = 3, 4, 5
    mub = np.full((ny, nx), 90000.0, dtype=np.float32)
    saved = np.full((ny, nx), 89000.0, dtype=np.float32)
    c3 = np.array([0.8, 0.4, 0.1], dtype=np.float32)
    c4 = np.array([1000.0, 5000.0, 8000.0], dtype=np.float32)
    theta = np.full((nz, ny, nx), 2.0, dtype=np.float32)
    pp = np.full((nz, ny, nx), 100.0, dtype=np.float32)
    qv = np.full((nz, ny, nx), 0.008, dtype=np.float32)
    expected_theta, expected_qv = np_adjust_tempqv(
        mub, saved, c3, c4, 10000.0, theta, pp, qv, use_theta_m=0)
    adjust_tempqv(
        mub, saved, c3, c4, 10000.0, theta, pp, qv, use_theta_m=0)
    np.testing.assert_array_equal(
        theta, np.asarray(expected_theta, dtype=np.float32))
    np.testing.assert_array_equal(qv, np.asarray(expected_qv, dtype=np.float32))


def test_numpy_sint_has_no_verification_runtime_dependency(monkeypatch):
    """The standalone RW-WPS wheel intentionally omits ``gpuwm.verify``."""
    import builtins

    reg = _mass_reg()
    parent = np.linspace(
        -2.0, 4.0, 2 * reg.nyp * reg.nxp,
        dtype=np.float32).reshape(2, reg.nyp, reg.nxp)
    expected = np.asarray(
        np_sint(parent, reg, dtype=np.float32), dtype=np.float32)
    real_import = builtins.__import__

    def reject_verify(name, *args, **kwargs):
        if name == "gpuwm.verify" or name.startswith("gpuwm.verify."):
            raise AssertionError(
                "production NumPy SINT imported developer verification code")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_verify)
    np.testing.assert_array_equal(sint(parent, reg), expected)


# ---------------------------------------------------------------------------
# Ledger cross-links: these tests implement the registered N1 comparators.
# ---------------------------------------------------------------------------

def test_n1_gate_records_are_the_acceptance_targets():
    for metric in ("sint_vs_fp64_mirror", "bdy_interp1_vs_fp64_mirror",
                   "blend_terrain_vs_fp64_mirror",
                   "adjust_tempqv_vs_fp64_mirror", "copy_fcn_vs_fp64_mirror"):
        assert ng.gate("N1", metric).kind == "fp64_mirror_floor"
    assert ng.gate("N1", "sint_quadratic_reproduction").kind == "exact"
    assert ng.gate("N1", "sint_parent_coincident_odd_ratio").kind == "exact"
    assert MAX_ULPS == 8


# ---------------------------------------------------------------------------
# Width formula and geometry pins.
# ---------------------------------------------------------------------------

def test_bdy_width_formula_pin():
    # sz = MIN(MAX(spec_zone, relax_zone+1), spec_bdy_width)
    # (interp_fcn.F:2517); bundle spec_zone=1/relax_zone=4/width=5 -> 5.
    for fn in (bdy_width, np_bdy_width):
        assert fn(1, 4, 5) == 5
        assert fn(1, 9, 5) == 5      # clamped by spec_bdy_width
        assert fn(7, 2, 9) == 7      # spec_zone dominates
        assert fn(2, 4, 9) == 5      # relax_zone + 1


def test_sint_offset_tables_fp32_pins():
    # XIG(J+(I-1)*rr) = (rr-1-rioff)/(2rr) - (J-1)/rr (sint.F:56), rioff=1
    # only for staggered even ratios (sint.F:49-52).  FP64-built,
    # FP32-stored (registered deviation vs WRF's REAL build).
    third = np.float32(np.float64(1.0) / 3.0)
    fifth2 = np.float32(np.float64(2.0) / 5.0)
    fifth1 = np.float32(np.float64(1.0) / 5.0)
    cases = {
        (1, False): [0.0], (1, True): [0.0],
        (2, False): [0.25, -0.25], (2, True): [0.0, -0.5],
        (3, False): [third, 0.0, -third], (3, True): [third, 0.0, -third],
        (4, False): [0.375, 0.125, -0.125, -0.375],
        (4, True): [0.25, 0.0, -0.25, -0.5],
        (5, False): [fifth2, fifth1, 0.0, -fifth1, -fifth2],
    }
    for (rr, stag), expect in cases.items():
        got = sint_offsets(rr, stag)
        assert got.dtype == np.float32
        assert got.tolist() == [np.float32(v) for v in expect], (rr, stag)


def test_sint_offsets_vs_wrf_real_build_emulation():
    # Verification-round R1: the PROVENANCE D5 coincidence claim, proven
    # against the WRF PATH rather than gpuwm's own stored values.  WRF
    # builds XIG per-op in REAL (sint.F:49-57):
    #   (float(rr)-1.-rioff)/float(2*rr) - FLOAT(J-1)*1./float(rr)
    # The FP64-build/FP32-store deviation coincides bitwise for
    # rr in {1,2,3,4} (every bundle-chain ratio) and diverges by exactly
    # 1 ULP at rr=5, ip=3: WRF per-op fl32(0.4f - 0.6f) = -0.20000002
    # (0xBE4CCCCE, an EXACT FP32 subtraction of the two rounded
    # operands) vs gpuwm fl32(fl64(-0.2)) = -0.2 (0xBE4CCCCD).
    f32 = np.float32
    for rr in (1, 2, 3, 4, 5):
        for stag in (False, True):
            rioff = f32(1.0) if (stag and rr % 2 == 0) else f32(0.0)
            wrf = np.array(
                [(f32(rr) - f32(1.0) - rioff) / f32(2 * rr)
                 - f32(ip) * f32(1.0) / f32(rr) for ip in range(rr)],
                dtype=np.float32)
            ours = sint_offsets(rr, stag)
            if rr == 5:
                assert wrf[3] == np.float32(-0.20000002)
                assert wrf[3].view(np.uint32) == np.uint32(0xBE4CCCCE)
                assert ours[3] == np.float32(-0.2)
                assert ours[3].view(np.uint32) == np.uint32(0xBE4CCCCD)
                assert ulp32(ours[3], wrf[3]) == 1
                np.testing.assert_array_equal(ours[[0, 1, 2, 4]],
                                              wrf[[0, 1, 2, 4]])
            else:
                np.testing.assert_array_equal(ours, wrf)


def test_stagger_wrapper_offset_pins():
    # interp_fcn_sint: ioff = (nri-1)/2 on stagger (interp_fcn.F:922-925);
    # bdy_interp1: ioff = MAX((nri-1)/2, 1) (:2504-2510).  They agree for
    # every ratio >= 3 and DIVERGE at nri in {1, 2} -- transliterated
    # faithfully, anomaly pinned (flagged for the N2 ratio-1 fixtures).
    for nri, interp_off, bdy_off in ((1, 0, 1), (2, 0, 1), (3, 1, 1),
                                     (4, 1, 1), (5, 2, 2)):
        assert _wrapper_offset("interp", True, nri) == interp_off
        assert _wrapper_offset("bdy", True, nri) == bdy_off
        assert _wrapper_offset("interp", False, nri) == 0
        assert _wrapper_offset("bdy", False, nri) == 0
    with pytest.raises(ValueError):
        _wrapper_offset("nearest", True, 3)


def test_donor_map_pins_mass_and_stagger():
    # Hand-derived from ci = ipos + (n1-1)/nri, ip = mod(n1-1, nri) with
    # n1 = ni + ioff (interp_fcn.F:975-985), 1-based -> 0-based.
    reg = register_nest(nri=3, nrj=3, i_parent_start=5, j_parent_start=5,
                        child_nx=6, child_ny=6, parent_nx=12, parent_ny=12)
    assert reg.ci.tolist() == [4, 4, 4, 5, 5, 5]
    assert reg.ip.tolist() == [0, 1, 2, 0, 1, 2]
    regu = register_nest(nri=3, nrj=3, i_parent_start=5, j_parent_start=5,
                         child_nx=6, child_ny=6, parent_nx=12, parent_ny=12,
                         stagger="x")
    assert regu.ioff == 1 and regu.joff == 0
    assert regu.nxc == 7 and regu.nyc == 6
    assert regu.ci.tolist() == [4, 4, 5, 5, 5, 6, 6]
    assert regu.ip.tolist() == [1, 2, 0, 1, 2, 0, 1]
    # y direction of the u registration is the mass map.
    assert regu.cj.tolist() == reg.cj.tolist()


def test_register_nest_validation():
    with pytest.raises(ValueError, match="square"):
        register_nest(nri=3, nrj=4, i_parent_start=5, j_parent_start=5,
                      child_nx=6, child_ny=6, parent_nx=12, parent_ny=12)
    with pytest.raises(ValueError, match="stagger"):
        _mass_reg(stagger="z")
    with pytest.raises(ValueError, match="SINT stencil"):
        # child origin at parent cell 1: donor 0 lacks the -2 stencil.
        register_nest(nri=3, nrj=3, i_parent_start=1, j_parent_start=5,
                      child_nx=6, child_ny=6, parent_nx=12, parent_ny=12)


# ---------------------------------------------------------------------------
# SINT property gates (N1 exact records).
# ---------------------------------------------------------------------------

def test_sint_parent_coincident_odd_ratio():
    # N1 gate sint_parent_coincident_odd_ratio: on odd ratios the center
    # child point has XIG = XJG = 0 (sint.F:56 with ip = (rr-1)/2), and the
    # whole SINT pipeline collapses to identity: DONOR(..,0) = 0,
    # TR4(..,0) = 0, W = Y0, limited fluxes 0 -> XF = Y0 exactly.
    rng = np.random.default_rng(11)
    parent = (10.0 + 10.0 * rng.random((13, 13))).astype(np.float32)
    reg = _mass_reg(nri=3, child=9, parent=13)
    out = np_sint(parent, reg)
    centers_x = np.flatnonzero(reg.ip == 1)
    centers_y = np.flatnonzero(reg.jp == 1)
    for jc in centers_y:
        for ic in centers_x:
            assert out[jc, ic] == np.float64(
                parent[reg.cj[jc], reg.ci[ic]])   # exact, not approx


@pytest.mark.parametrize("nri,stagger", [(3, ""), (2, ""), (4, "x"),
                                         (3, "y")])
def test_sint_quadratic_reproduction(nri, stagger):
    # N1 gate sint_quadratic_reproduction: SINT is exact for cubics per
    # pass; a monotone quadratic (no limiter clip) must be reproduced AT
    # THE GEOMETRY-DEFINED SAMPLE POINT x* = ci - XIG, y* = cj - XJG (the
    # donor advection displacement, sint.F:36-44), with the FP32-rounded
    # offsets defining the point.
    def f(x, y):
        return 100.0 + 0.5 * x + 0.7 * y + 0.003 * x * x + 0.004 * y * y \
            + 0.002 * x * y

    reg = register_nest(nri=nri, nrj=nri, i_parent_start=4, j_parent_start=4,
                        child_nx=2 * nri, child_ny=2 * nri,
                        parent_nx=14, parent_ny=14, stagger=stagger)
    xp = np.arange(reg.nxp, dtype=np.float64)
    yp = np.arange(reg.nyp, dtype=np.float64)
    parent = f(xp[None, :], yp[:, None])
    out = np_sint(parent, reg)
    xs = reg.ci - np.float64(reg.xig)[reg.ip]
    ys = reg.cj - np.float64(reg.xjg)[reg.jp]
    expect = f(xs[None, :], ys[:, None])
    np.testing.assert_allclose(out, expect, rtol=1e-12, atol=1e-10)


def test_sint_limiter_activation_fixture():
    # Plan Task 10 / F6 amendment: a field where the SINT monotonic
    # limiter CLIPS -- a fixed-weight linear (TR4) stencil fails it, the
    # transliterated operator passes.  Hand derivation from sint.F:286-309
    # in exact rationals, stencil Y(-2..2) = (-2, 4, -2, -2, 4), A = -1/2
    # (ratio-4 x-staggered sub-position ip = 3, XIG = -0.5 exactly):
    #
    #   FL0 = DONOR(4,-2,-1/2)  = -2 * -1/2                    = 1
    #   FL1 = DONOR(-2,-2,-1/2) = -2 * -1/2                    = 1
    #   W   = -2 - (1 - 1)                                     = -2
    #   MXM = max(4,-2,-2,W) = 4      MN = min(...) = -2
    #   TR4(Y-2..Y1, A) = 13/64       TR4(Y-1..Y2, A) = 11/8
    #   F0  = 13/64 - 1 = -51/64      F1 = 11/8 - 1 = 3/8
    #   OV  = (4-(-2)) / (PP(F0) - PN(F1) + EP) = 6/EP  (huge)
    #   UN  = (W - MN) / (PP(F1) - PN(F0) + EP) = 0     (W == MN)
    #   F0' = 0*min(1,OV) + (-51/64)*min(1,UN=0)         = 0
    #   F1' = (3/8)*min(1,UN=0) + 0*min(1,OV)            = 0
    #   OUT = W - (F1' - F0')                            = -2   (CLIPPED)
    #
    # while the UNLIMITED 4th-order value is Y0 - (TR4_1 - TR4_0)
    # = -2 - (11/8 - 13/64) = -203/64 = -3.171875, an undershoot below
    # min(neighbors) that the limiter must remove.
    pattern = np.array([-2.0, 4.0, -2.0, -2.0, 4.0])
    reg = register_nest(nri=4, nrj=4, i_parent_start=4, j_parent_start=4,
                        child_nx=8, child_ny=8, parent_nx=16, parent_ny=16,
                        stagger="x")
    parent = np.zeros((reg.nyp, reg.nxp))
    parent[:, :] = np.arange(reg.nxp)[None, :] * 0.0  # constant in y
    parent[:, 2:7] = pattern[None, :]                 # donor ci0=4 stencil
    out = np_sint(parent, reg)
    picks = [ic for ic in range(reg.nxc)
             if reg.ci[ic] == 4 and reg.ip[ic] == 3]
    assert picks, "fixture must exercise the a=-1/2 sub-position"
    for ic in picks:
        assert reg.xig[reg.ip[ic]] == np.float32(-0.5)
        # The limited (transliterated) operator: exactly the low-order
        # donor value; UN = 0 zeroes both antidiffusive fluxes bitwise.
        assert np.all(out[:, ic] == -2.0)
    # The would-be fixed-weight stencil (unlimited TR4 evaluation of the
    # same geometry -- what a precomputed-weights implementation returns):
    a = -0.5
    one12, one24 = 1.0 / 12.0, 1.0 / 24.0

    def tr4(ym1, y0, yp1, yp2):
        return (a * one12 * (7.0 * (yp1 + y0) - (yp2 + ym1))
                - a * a * one24 * (15.0 * (yp1 - y0) - (yp2 - ym1))
                - a ** 3 * one12 * ((yp1 + y0) - (yp2 + ym1))
                + a ** 4 * one24 * (3.0 * (yp1 - y0) - (yp2 - ym1)))

    unlimited = pattern[2] - (tr4(*pattern[1:5]) - tr4(*pattern[0:4]))
    np.testing.assert_allclose(unlimited, -203.0 / 64.0, rtol=1e-14)
    # The linear-stencil result undershoots MN and FAILS the oracle the
    # transliterated operator passes: the two differ far beyond the floor.
    assert ulp32(unlimited, -2.0) > MAX_ULPS


# ---------------------------------------------------------------------------
# bdy_interp1: REAL*8-rdt scheme, F7 divisor contract, table semantics.
# ---------------------------------------------------------------------------

def _const_bdy_case(parent_value, nri=3, stagger="", child_value=0.0):
    reg = register_nest(nri=nri, nrj=nri, i_parent_start=4, j_parent_start=4,
                        child_nx=12, child_ny=12, parent_nx=16, parent_ny=16,
                        stagger=stagger, wrapper="bdy")
    parent = np.full((reg.nyp, reg.nxp), parent_value, dtype=np.float64)
    child = np.full((reg.nyc, reg.nxc), child_value, dtype=np.float64)
    return reg, parent, child


def test_bdy_real8_rdt_discriminating_fixture():
    # Plan Task 10: a fixture proving the REAL*8-rdt ORDERING.  Hand
    # transcription of interp_fcn.F:2480/:2500/:2583 for one element:
    # cdt (REAL) = 60, difference (REAL) = 0.010299897:
    #   REAL*8 rdt = 1.D0/cdt; tend = REAL(rdt * DBLE(diff))
    # against the WRONG all-FP32 transcription REAL rdt32 = 1./cdt;
    # tend = diff * rdt32.  The two double-rounding paths land on
    # DIFFERENT FP32 values for this diff -- so an all-FP32 kernel fails
    # a bitwise check the mirrored scheme passes.
    cdt = np.float32(60.0)
    diff = np.float32(0.010299897)
    real8 = np.float32(np.float64(1.0) / np.float64(cdt) * np.float64(diff))
    plain = np.float32(diff * np.float32(np.float32(1.0) / cdt))
    assert real8 == np.float32(0.00017166494)   # hand-pinned bit pattern
    assert plain == np.float32(0.00017166496)
    assert real8 != plain                       # the fixture discriminates
    # The mirror realizes the REAL*8 path: constant parent == diff, child
    # == 0 makes SINT exact (constant field), so every tendency element
    # must round to the REAL*8 value, not the plain-FP32 one.
    reg, parent, child = _const_bdy_case(float(diff))
    tables = np_bdy_interp1(parent, child, reg, parent_dt_fp32=60.0)
    for side in ("west", "east", "south", "north"):
        value, tendency = tables[side]
        assert np.all(value == 0.0)
        assert np.all(np.float32(tendency) == real8)
        assert not np.any(np.float32(tendency) == plain)


def test_bdy_tendency_divisor_is_the_parent_step():
    # F7 amendment fixture (ratio 3 != 1): cdt IS the coarse step
    # (interp_fcn.F:2320 "Time step size for CG and FG"; decls :2345,
    # :2472).  Constant parent 7.5, child 0, parent dt 60 (exact dyadic
    # quotients): tendency = 7.5/60 = 0.125 everywhere.  A wrong-child-dt
    # transcription (20 = 60/3) differs by EXACTLY the parent:child time
    # ratio 3 and fails the oracle comparison.
    reg, parent, child = _const_bdy_case(7.5)
    right = np_bdy_interp1(parent, child, reg, parent_dt_fp32=60.0,
                           parent_interval_ticks=180)
    wrong = np_bdy_interp1(parent, child, reg, parent_dt_fp32=20.0)
    for side in ("west", "east", "south", "north"):
        rt = right[side][1]
        wt = wrong[side][1]
        # 7.5/60 = 0.125 and 7.5/20 = 0.375 at the comparator's FP32
        # resolution (the FP64 mirror carries fl64(1/60) internally).
        assert np.all(np.float32(rt) == np.float32(0.125))
        assert np.all(np.float32(wt) == np.float32(0.375))
        np.testing.assert_allclose(wt, 3.0 * rt, rtol=1e-14)
        assert np.all(wt != rt)
        assert ulp32(wt, rt) > MAX_ULPS         # ... and it FAILS the floor


def test_bdy_parent_step_uses_fp32_value_not_exact_rational():
    # Shadow F7 fixture: the parent's chained REAL dt is fl32(5/3), not the
    # exact rational 5/3 and not the child's fl32(5/9).  With a constant
    # parent, SINT is exact and the three interpretations land separately.
    cdt = np.float32(5.0 / 3.0)
    diff = np.float32(-17.30576)
    correct = np.float32(
        np.float64(1.0) / np.float64(cdt) * np.float64(diff))
    exact_rational = np.float32(np.float64(3.0 / 5.0) * np.float64(diff))
    child_divisor = np.float32(
        np.float64(1.0) / np.float64(np.float32(cdt / np.float32(3.0)))
        * np.float64(diff))
    assert correct.view(np.uint32) == np.uint32(0xC12622A3)
    assert exact_rational.view(np.uint32) == np.uint32(0xC12622A2)
    assert child_divisor.view(np.uint32) == np.uint32(0xC1F933F5)

    reg, parent, child = _const_bdy_case(float(diff))
    tables = np_bdy_interp1(
        parent, child, reg, parent_dt_fp32=cdt,
        parent_interval_ticks=5, dtype=np.float32)
    for side in ("west", "east", "south", "north"):
        got = np.float32(tables[side][1])
        assert np.all(got == correct)
        assert not np.any(got == exact_rational)
        assert not np.any(got == child_divisor)


def test_bdy_value_tables_are_the_child_current_state():
    # bdy_xs(nj,k,ni) = nfld(ni,k,nj) (:2584): VALUE is the child's own
    # current state (a REAL store -- the fixture is FP32-representable so
    # the equality is exact); east/north width index 0 sits at the domain
    # edge (:2593-2617), matching the Phase-4 lateral_bc.py orientation.
    reg, parent, _ = _const_bdy_case(5.0)
    nyc, nxc = reg.nyc, reg.nxc
    child = (np.arange(nyc * nxc, dtype=np.float64)
             .reshape(nyc, nxc) / 7.0 + 1.0).astype(np.float32
                                                    ).astype(np.float64)
    tables = np_bdy_interp1(parent, child, reg, parent_dt_fp32=60.0)
    sz = bdy_width(1, 4, 5)
    west_v = tables["west"][0]
    east_v = tables["east"][0]
    south_v = tables["south"][0]
    north_v = tables["north"][0]
    assert west_v.shape == (1, nyc, sz) and south_v.shape == (1, sz, nxc)
    np.testing.assert_array_equal(west_v[0], child[:, :sz])
    np.testing.assert_array_equal(south_v[0], child[:sz, :])
    for w in range(sz):
        np.testing.assert_array_equal(east_v[0, :, w], child[:, nxc - 1 - w])
        np.testing.assert_array_equal(north_v[0, w, :], child[nyc - 1 - w, :])


def test_bdy_interp1_validation():
    reg, parent, child = _const_bdy_case(1.0)
    with pytest.raises(ValueError, match="positive"):
        np_bdy_interp1(parent, child, reg, parent_dt_fp32=-60.0)
    with pytest.raises(ValueError, match="parent_interval_ticks"):
        np_bdy_interp1(parent, child, reg, parent_dt_fp32=60.0,
                       parent_interval_ticks=0)


def test_np_bdy_interp1_requires_bdy_wrapper():
    # Fix round (review finding 4): the mirror must hard-require the bdy
    # wrapper exactly like the kernel driver -- an 'interp' registration
    # mirrors the wrong stagger geometry (ioff formulas diverge,
    # interp_fcn.F:922-925 vs :2504-2510).
    reg = _mass_reg()          # wrapper="interp"
    parent = np.zeros((reg.nyp, reg.nxp))
    child = np.zeros((reg.nyc, reg.nxc))
    with pytest.raises(ValueError, match="wrapper='bdy'"):
        np_bdy_interp1(parent, child, reg, parent_dt_fp32=60.0)


def test_bdy_mirror_forms_real_difference():
    # Shadow-review F2 regression: WRF's :2583 difference is REAL -- ONE
    # FP32 rounding BEFORE the REAL*8 promote (psca and nfld are REAL
    # arrays, :2465-2486).  Hand chain for constant parent P and child N
    # (SINT is exact on constants), transcribed from the Fortran:
    P = np.float32(17.287472)
    N = np.float32(2.9138315)
    rdt = np.float64(1.0) / np.float64(np.float32(60.0))   # :2500
    d32 = P - N                             # the :2583 REAL difference
    assert np.float64(d32) != np.float64(P) - np.float64(N)   # it rounds
    wrf = np.float32(rdt * np.float64(d32))
    fp64_diff = np.float32(rdt * (np.float64(P) - np.float64(N)))
    assert wrf == np.float32(0.23956066)         # hand-pinned bits
    assert fp64_diff == np.float32(0.23956068)
    assert wrf != fp64_diff                      # the fixture discriminates
    # The mirror must land on the REAL-difference value on every side
    # (a former FP64-difference mirror lands on fp64_diff instead).
    reg, parent, child = _const_bdy_case(float(P), child_value=float(N))
    tables = np_bdy_interp1(parent, child, reg, parent_dt_fp32=60.0)
    for side in ("west", "east", "south", "north"):
        value, tendency = tables[side]
        assert np.all(np.float32(tendency) == wrf)
        assert not np.any(np.float32(tendency) == fp64_diff)
        assert np.all(value == np.float64(N))


def test_sint_fp64_oracle_cancellation_and_real_emulation():
    # Shadow-review F1 regression + adjudication: where the SINT output
    # crosses zero, the FP64 oracle and the FP32 pipeline legitimately
    # diverge by orders of magnitude in output ULPs (O(eps32*field)
    # representation spread survives the cancelling subtraction), so the
    # FP64 mirror cannot serve a cancellation-sensitive comparator there.  The
    # comparator's mirror at such points is the WRF REAL statement
    # emulation -- np_sint(dtype=np.float32), the SAME transliteration
    # evaluated at per-op FP32 exactly as the Fortran REAL / the
    # -fmad=false kernel.  Construction: a linear-in-x parent field whose
    # zero crossing sits 3e-8 inside the ip=0 sample point (offset
    # a = fl32(1/3)) of donor column ci0 = 4.
    reg = _mass_reg(nri=3, ipos=4, jpos=3, child=9, parent=13)
    a32 = reg.xig[0]
    assert a32 == np.float32(np.float64(1.0) / 3.0)
    z = -np.float64(a32) + 3.0e-8
    row = np.float32((np.arange(13, dtype=np.float64) - 4.0) - z)
    parent = np.tile(row, (13, 1))
    out64 = np_sint(parent, reg)
    out32 = np_sint(parent, reg, dtype=np.float32)
    assert out32.dtype == np.float32
    ic = 3
    assert reg.ci[ic] == 4 and reg.ip[ic] == 0
    # The REAL emulation pins the FP32 pipeline value (derived by an
    # independent per-op FP32 transcription of sint.F:286-310):
    assert np.all(out32[:, ic] == np.float32(-4.4703484e-08))
    # ... while the FP64 oracle sits ~9.3e6 ULPs away at this point --
    # the shadow's floor breach, now pinned as the boundary of the FP64
    # oracle's soundness (bounded-away-from-zero fixtures only):
    assert ulp32(out64[:, ic], out32[:, ic]) > MAX_ULPS
    # Away from the crossing the two precisions agree at the floor.
    far = [i for i in range(reg.nxc) if i != ic and reg.ip[i] == 0]
    assert ulp32(out64[:, far], out32[:, far]) <= MAX_ULPS
    with pytest.raises(ValueError, match="dtype"):
        np_sint(parent, reg, dtype=np.int32)


def test_sint_out_buffer_validation():
    # Fix round (review finding 2): a wrong out= buffer must fail loud
    # before any launch (the kernels write out[tid] over nz*nyc*nxc
    # threads).  Validation happens before any CuPy work, so the raising
    # path is CPU-testable with NumPy stand-ins.
    from gpuwm.core.nest_interp import sint
    reg = _mass_reg()
    parent = np.zeros((3, reg.nyp, reg.nxp), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        sint(parent, reg, out=np.zeros((3, reg.nyc, reg.nxc + 1),
                                       dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        sint(parent, reg, out=np.zeros((3, reg.nyc, reg.nxc),
                                       dtype=np.float64))
    with pytest.raises(ValueError, match="contiguous"):
        sint(parent, reg,
             out=np.zeros((3, reg.nyc, 2 * reg.nxc),
                          dtype=np.float32)[:, :, ::2])


def test_bdy_out_table_validation():
    from gpuwm.core.nest_interp import bdy_interp1
    reg, _, _ = _const_bdy_case(1.0)
    parent = np.zeros((1, reg.nyp, reg.nxp), dtype=np.float32)
    child = np.zeros((1, reg.nyc, reg.nxc), dtype=np.float32)
    sz = bdy_width(1, 4, 5)

    def good(shape):
        return np.zeros(shape, dtype=np.float32)

    out = {"west": (good((1, reg.nyc, sz)), good((1, reg.nyc, sz))),
           "east": (good((1, reg.nyc, sz)), good((1, reg.nyc, sz))),
           "south": (good((1, sz, reg.nxc)), good((1, sz, reg.nxc))),
           "north": (good((1, sz, reg.nxc)),
                     good((1, sz + 1, reg.nxc)))}      # bad tendency
    with pytest.raises(ValueError, match="north tendency"):
        bdy_interp1(parent, child, reg, parent_dt_fp32=60.0, out=out)


# ---------------------------------------------------------------------------
# blend_terrain (nest_init_utils.F:712-785).
# ---------------------------------------------------------------------------

def test_blend_terrain_weight_table_pin():
    # Plan Task 10 blend-weight table pin: with fine == 6 and coarse == 0,
    # (blend_cell*fine + (bw+1-blend_cell)*coarse)/(bw+1) = blend_cell, so
    # the west-edge interior-row profile reads the weight table directly:
    # spec rows 1..5 pure coarse (0), blend rows 6..10 = 1,2,3,4,5 (i.e.
    # fine weights blend_cell/6 = 1/6..5/6), interior pure fine (6).
    ny = nx = 24
    fine = np.full((ny, nx), 6.0)
    coarse = np.zeros((ny, nx))
    out = np_blend_terrain(coarse, fine, spec_bdy_width=5, blend_width=5)
    mid = ny // 2  # row far from the j edges
    assert out[mid, :12].tolist() == [0., 0., 0., 0., 0.,
                                      1., 2., 3., 4., 5., 6., 6.]
    # East mirror: i == ide - sbw - cell with ide = nx+1 (:761).
    assert out[mid, -12:].tolist() == [6., 6., 5., 4., 3., 2., 1.,
                                       0., 0., 0., 0., 0.]
    # Corner: (i,j) 1-based (6,8) matches i-cell 1 and j-cell 3; the
    # descending blend_cell loop's overwrite order makes the SMALLEST
    # matching cell win (:759-765) -> weight 1.
    assert out[7, 5] == 1.0
    # Interior untouched, spec frame pure coarse everywhere.
    assert np.all(out[11:13, 11:13] == 6.0)
    assert np.all(out[:5, :] == 0.0) and np.all(out[:, :5] == 0.0)


def test_blend_terrain_3d_k_inert():
    ny = nx = 20
    rng = np.random.default_rng(3)
    fine = rng.random((3, ny, nx))
    coarse = rng.random((3, ny, nx))
    out = np_blend_terrain(coarse, fine, spec_bdy_width=5, blend_width=5)
    for k in range(3):
        np.testing.assert_array_equal(
            out[k], np_blend_terrain(coarse[k], fine[k]))


# ---------------------------------------------------------------------------
# adjust_tempqv (nest_init_utils.F:812-890).
# ---------------------------------------------------------------------------

def test_adjust_tempqv_hand_point():
    # Single-point hand transcription (see the derivation constants in the
    # commit): mub 97000 <- save_mub 98000, c3=0.9, c4=5000, p_top=10000,
    # th'=2, pp=250, qv=0.008:
    #   p_old = 5000 + 0.9*98000 + 10000 + 250 = 103450        (:851)
    #   p_new = 5000 + 0.9*97000 + 10000 + 250 = 102550        (:867)
    # use_theta_m=0: tc = 302*(1.0345)**(2/7) - 273.15 = 31.790884607...,
    # rh = e/es = 0.2791934770..., dth = +0.253249532... ->
    # th_new = 2.2532495322568593, qv_new = 0.007840486596997764.
    args = (np.full((1, 1), 97000.0), np.full((1, 1), 98000.0),
            np.array([0.9]), np.array([5000.0]), 10000.0,
            np.full((1, 1, 1), 2.0), np.full((1, 1, 1), 250.0),
            np.full((1, 1, 1), 0.008))
    th0, qv0 = np_adjust_tempqv(*args, use_theta_m=0)
    np.testing.assert_allclose(th0[0, 0, 0], 2.2532495322568593, rtol=1e-13)
    np.testing.assert_allclose(qv0[0, 0, 0], 0.007840486596997764,
                               rtol=1e-13)
    # use_theta_m=1 (:853/:870/:877): the moist branch conserves the same
    # dry-theta correction but converts through theta_m, landing on a
    # different qv: 0.007836691969644838.  This is the only number in the
    # hand transcription that touches rvovrd, and it was re-derived by hand
    # when the constant was respelled -- WRF spells R_v/R_d inline here and
    # gfortran folds it to the float32 word 0x3fcdded2 (1.6083624362945557),
    # where this file previously carried the double quotient
    # 1.6083623693379792.  Same derivation, same eight lines, one constant:
    # 0.00783669196980337 -> 0.007836691969644838, 2.0e-11 relative.  th1 is
    # unchanged to within the rtol here (2.2532495322568593 ->
    # ...568025, 2.5e-14) because thloc*(1+rvovrd*qv) undoes its own divide.
    th1, qv1 = np_adjust_tempqv(*args, use_theta_m=1)
    np.testing.assert_allclose(th1[0, 0, 0], 2.2532495322568593, rtol=1e-13)
    np.testing.assert_allclose(qv1[0, 0, 0], 0.007836691969644838,
                               rtol=1e-13)
    assert qv0[0, 0, 0] != qv1[0, 0, 0]


def test_adjust_tempqv_identity_when_mub_unchanged():
    # save_mub == mub -> p_old == p_new -> dth = 0 and RH round-trips:
    # theta must be exactly unchanged; qv reconstructs through es at the
    # same tc (FP64 roundoff only).
    rng = np.random.default_rng(5)
    mub = 96000.0 + 2000.0 * rng.random((4, 5))
    c3 = np.linspace(1.0, 0.1, 6)
    c4 = np.linspace(0.0, 4000.0, 6)
    # Quantize theta so th+300-300 round-trips exactly in FP64 (the
    # identity is about dth == 0, not about sum rounding).
    th = np.round(300.0 * 0.02 * rng.standard_normal((6, 4, 5))
                  * 2.0 ** 20) / 2.0 ** 20
    pp = 300.0 * rng.standard_normal((6, 4, 5))
    qv = 0.002 + 0.015 * rng.random((6, 4, 5))
    th2, qv2 = np_adjust_tempqv(mub, mub, c3, c4, 10000.0, th, pp, qv,
                                use_theta_m=0)
    np.testing.assert_array_equal(th2, th)
    np.testing.assert_allclose(qv2, qv, rtol=1e-12)


# ---------------------------------------------------------------------------
# copy_fcn family (dormant Phase-5b feedback), BOTH parity branches.
# ---------------------------------------------------------------------------

def _feedback_case(nri, stagger, child_cells=3, parent=10, ipos=4):
    """One-parent-cell-wide feedback fixture geometry."""
    child = nri * child_cells
    reg = register_nest(nri=nri, nrj=nri, i_parent_start=ipos,
                        j_parent_start=ipos, child_nx=child, child_ny=child,
                        parent_nx=parent, parent_ny=parent, stagger=stagger)
    return reg


def test_copy_fcn_odd_mass_cell_average_pin():
    # Odd branch (interp_fcn.F:1463-1517): with nri=nrj=3, ipos=jpos=4,
    # child mass 9x9 (3 parent cells), spec_zone=1 the feedback range is
    # the single parent cell ci=cj=5 (1-based bounds 4+1 .. 4+3-1-1 = 5),
    # centered on child ni=nj=(5-4)*3+2=5; its 3x3 children carry 1..9:
    # cfld = sum 1/REAL(9) * k.  1/REAL(9) is a REAL (FP32) weight; the
    # FP64 sum of fl32(1/9)*k over k=1..9 is exact in FP64 (each product
    # and partial sum representable), so the pin is bitwise:
    w32 = np.float64(np.float32(1.0) / np.float32(9.0))
    expect = w32 * 45.0                      # = 5.000000037252903
    assert expect == 5.000000037252903
    reg = _feedback_case(3, "")
    parent = np.full((reg.nyp, reg.nxp), -7.0)
    child = np.zeros((reg.nyc, reg.nxc))
    child[3:6, 3:6] = np.arange(1.0, 10.0).reshape(3, 3)
    out = np_copy_fcn(parent, child, i_parent_start=4, j_parent_start=4,
                      nri=3, nrj=3)
    assert out[4, 4] == expect
    # Everything outside the spec_zone-clipped feedback range is untouched.
    untouched = out.copy()
    untouched[4, 4] = -7.0
    np.testing.assert_array_equal(untouched, np.full_like(out, -7.0))


def test_copy_fcn_even_mass_cell_average_pin():
    # EVEN branch (interp_fcn.F:1567-1663): nri=4 (the bundle's d02!) is
    # cell-average too -- 1/(nri*nrj) over ipoints/jpoints 0..3 anchored
    # at the SW child ni=(ci-ipos)*nri+istag (:1648) -- NOT point
    # sampling (the stale design-spec text; registered deviations list).
    # Weight fl32(1/16) = 0.0625 is dyadic, so 1..16 average is exactly
    # 8.5.
    reg = _feedback_case(4, "")
    parent = np.zeros((reg.nyp, reg.nxp))
    child = np.zeros((reg.nyc, reg.nxc))
    child[4:8, 4:8] = np.arange(1.0, 17.0).reshape(4, 4)
    out = np_copy_fcn(parent, child, i_parent_start=4, j_parent_start=4,
                      nri=4, nrj=4)
    assert out[4, 4] == 8.5
    assert np.count_nonzero(out) == 1


def test_copy_fcn_stagger_along_face_averages():
    # u odd (:1519-1539): 3 j-points at the coincident face, weight
    # 1/REAL(3); u even (:1695-1713): stride-nri ijpoints -> nrj j-points
    # at the face, weight 1/REAL(4) -- both along-face, never in-cell.
    reg = _feedback_case(3, "x")
    parent = np.zeros((reg.nyp, reg.nxp))
    child = np.zeros((reg.nyc, reg.nxc))
    # feedback cells ci in {5,6}, cj=5 -> for ci=5 the u face is
    # ni=(5-4)*3+1=4 (1-based), nj center 5, j-points nj-1..nj+1
    # (ijpoints 2,5,8 -> jpoints -1,0,1) -> parent (0-based) [4, 4].
    child[3:6, 3] = (3.0, 5.0, 10.0)
    out = np_copy_fcn(parent, child, i_parent_start=4, j_parent_start=4,
                      nri=3, nrj=3, xstag=True)
    w32 = np.float64(np.float32(1.0) / np.float32(3.0))
    assert out[4, 4] == w32 * 3.0 + w32 * 5.0 + w32 * 10.0
    regv = _feedback_case(4, "y")
    parentv = np.zeros((regv.nyp, regv.nxp))
    childv = np.zeros((regv.nyc, regv.nxc))
    # even v (:1717-1737): ijpoints 1..nri -> the 4 i-points along the
    # face at nj=(cj-jpos)*nrj+1 = 5 (0-based 4), ni..ni+3 = 0-based 4..7.
    childv[4, 4:8] = (1.0, 2.0, 3.0, 4.0)
    outv = np_copy_fcn(parentv, childv, i_parent_start=4, j_parent_start=4,
                       nri=4, nrj=4, ystag=True)
    assert outv[4, 4] == 2.5     # 1/4 dyadic: exact
    assert np.count_nonzero(outv) == 1


def test_copy_fcnm_and_fcni_corner_pins():
    # copy_fcnm odd (:1793-1804): center child ni=(ci-ipos)*nri+istag+1;
    # even (:1806-1818): "pick nearest neighbor on SW corner",
    # ni = (ci-ipos)*nri + 1 + (nri/2 - 1).
    reg = _feedback_case(3, "")
    parent = np.zeros((reg.nyp, reg.nxp))
    child = np.arange(float(reg.nyc * reg.nxc)).reshape(reg.nyc, reg.nxc)
    out = np_copy_fcnm(parent, child, i_parent_start=4, j_parent_start=4,
                       nri=3, nrj=3)
    assert out[4, 4] == child[4, 4]          # 1-based (5,5): center child
    reg4 = _feedback_case(4, "")
    parent4 = np.zeros((reg4.nyp, reg4.nxp))
    child4 = np.arange(float(reg4.nyc * reg4.nxc)).reshape(reg4.nyc,
                                                           reg4.nxc)
    out4 = np_copy_fcnm(parent4, child4, i_parent_start=4, j_parent_start=4,
                        nri=4, nrj=4)
    # ni = (5-4)*4 + 1 + (4/2-1) = 6 (1-based) -> 0-based (5,5).
    assert out4[4, 4] == child4[5, 5]
    ints = np_copy_fcni(parent4.astype(np.int32), child4.astype(np.int32),
                        i_parent_start=4, j_parent_start=4, nri=4, nrj=4)
    assert ints.dtype == np.int32
    assert ints[4, 4] == int(child4[5, 5])
    with pytest.raises(TypeError, match="INTEGER"):
        np_copy_fcni(parent4, child4, i_parent_start=4, j_parent_start=4,
                     nri=4, nrj=4)


# ---------------------------------------------------------------------------
# Offline NVRTC compile check (no device; plan: kernels compile-checked
# offline via NVRTC where possible).
# ---------------------------------------------------------------------------

def test_nest_cu_compiles_under_nvrtc():
    pytest.importorskip("cupy")
    try:
        from cupy.cuda.compiler import compile_using_nvrtc
    except ImportError:  # pragma: no cover - cupy API drift
        pytest.skip("cupy.cuda.compiler.compile_using_nvrtc unavailable")
    from gpuwm.core.kernels import _preamble
    src = _preamble() + (Path(__file__).resolve().parents[1] / "gpuwm"
                         / "core" / "kernels" / "nest.cu").read_text()
    # arch pinned explicitly so no device query happens; -fmad=false is
    # the module's contracted compile mode (nest_interp._nest_module).
    result = compile_using_nvrtc(src, options=("-std=c++17", "-fmad=false"),
                                 arch="80", filename="nest.cu")
    code = result[0] if isinstance(result, tuple) else result
    # Depending on the CUDA toolkit, NVRTC hands back PTX text or a binary
    # CUBIN; either way a successful compile plus the extern "C" symbol
    # names present is the offline check.
    blob = code if isinstance(code, bytes) else str(code).encode()
    assert blob
    for name in (b"nest_sint", b"nest_bdy_interp1", b"nest_blend_terrain",
                 b"nest_adjust_tempqv", b"nest_copy_fcn", b"nest_copy_fcnm",
                 b"nest_copy_fcni"):
        assert name in blob


# ---------------------------------------------------------------------------
# Kernel-vs-mirror parity at the pinned FP64-mirror floor (controller-run GPU).
# ---------------------------------------------------------------------------

def _dev_reg_case(nri, stagger, wrapper, seed=0, child_cells=4, parent=18):
    rng = np.random.default_rng(seed)
    reg = register_nest(nri=nri, nrj=nri, i_parent_start=5, j_parent_start=4,
                        child_nx=nri * child_cells, child_ny=nri * child_cells,
                        parent_nx=parent, parent_ny=parent,
                        stagger=stagger, wrapper=wrapper)
    parent_f = (10.0 + 10.0 * rng.random((6, reg.nyp, reg.nxp))
                ).astype(np.float32)
    return reg, parent_f


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("nri,stagger", [(3, ""), (3, "x"), (3, "y"),
                                         (4, ""), (4, "x"), (4, "y")])
def test_gpu_sint_matches_mirror(nri, stagger):
    import cupy as cp
    from gpuwm.core.nest_interp import sint
    reg, parent = _dev_reg_case(nri, stagger, "interp", seed=nri)
    out = sint(cp.asarray(parent), reg)
    mirror = np_sint(parent, reg)
    assert ulp32(cp.asnumpy(out), np.float32(mirror)) <= MAX_ULPS


@requires_gpu
@pytest.mark.gpu
def test_gpu_sint_limiter_fixture_bitwise():
    import cupy as cp
    from gpuwm.core.nest_interp import sint
    reg = register_nest(nri=4, nrj=4, i_parent_start=4, j_parent_start=4,
                        child_nx=8, child_ny=8, parent_nx=16, parent_ny=16,
                        stagger="x")
    parent = np.zeros((3, reg.nyp, reg.nxp), dtype=np.float32)
    parent[:, :, 2:7] = np.array([-2, 4, -2, -2, 4], dtype=np.float32)
    out = cp.asnumpy(sint(cp.asarray(parent), reg))
    picks = [ic for ic in range(reg.nxc)
             if reg.ci[ic] == 4 and reg.ip[ic] == 3]
    for ic in picks:
        assert np.all(out[:, :, ic] == np.float32(-2.0))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("nri,stagger", [(3, ""), (3, "x"), (4, "y")])
def test_gpu_bdy_interp1_matches_mirror(nri, stagger):
    import cupy as cp
    from gpuwm.core.nest_interp import bdy_interp1
    reg, parent = _dev_reg_case(nri, stagger, "bdy", seed=10 + nri)
    child = np.ones((6, reg.nyc, reg.nxc), dtype=np.float32)
    dev = bdy_interp1(cp.asarray(parent), cp.asarray(child), reg,
                      parent_dt_fp32=60.0, parent_interval_ticks=180)
    mirror = np_bdy_interp1(parent, child, reg, parent_dt_fp32=60.0)
    for side in ("west", "east", "south", "north"):
        for got, ref in zip(dev[side], mirror[side]):
            assert ulp32(cp.asnumpy(got), np.float32(ref)) <= MAX_ULPS, side


@requires_gpu
@pytest.mark.gpu
def test_gpu_bdy_real8_rdt_bit_pin():
    # The kernel must realize the REAL*8 scheme bitwise on the
    # discriminating fixture (constant field -> SINT exact).
    import cupy as cp
    from gpuwm.core.nest_interp import bdy_interp1
    reg = register_nest(nri=3, nrj=3, i_parent_start=4, j_parent_start=4,
                        child_nx=12, child_ny=12, parent_nx=16, parent_ny=16,
                        wrapper="bdy")
    parent = cp.full((1, reg.nyp, reg.nxp), cp.float32(0.010299897))
    child = cp.zeros((1, reg.nyc, reg.nxc), dtype=cp.float32)
    dev = bdy_interp1(parent, child, reg, parent_dt_fp32=60.0)
    for side in ("west", "east", "south", "north"):
        tend = cp.asnumpy(dev[side][1])
        assert np.all(tend == np.float32(0.00017166494))


@requires_gpu
@pytest.mark.gpu
def test_gpu_blend_terrain_matches_mirror():
    import cupy as cp
    from gpuwm.core.nest_interp import blend_terrain
    rng = np.random.default_rng(2)
    fine = (200.0 + 300.0 * rng.random((2, 24, 26))).astype(np.float32)
    coarse = (200.0 + 300.0 * rng.random((2, 24, 26))).astype(np.float32)
    dev = cp.asarray(fine)
    blend_terrain(cp.asarray(coarse), dev)
    mirror = np_blend_terrain(coarse, fine)
    assert ulp32(cp.asnumpy(dev), np.float32(mirror)) <= MAX_ULPS


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("use_theta_m", [0, 1])
def test_gpu_adjust_tempqv_matches_mirror(use_theta_m):
    import cupy as cp
    from gpuwm.core.nest_interp import adjust_tempqv
    rng = np.random.default_rng(4)
    ny, nx, nz = 6, 7, 8
    mub = (96000.0 + 2000.0 * rng.random((ny, nx))).astype(np.float32)
    save_mub = (mub + 800.0 * rng.standard_normal((ny, nx))
                ).astype(np.float32)
    c3 = np.linspace(1.0, 0.05, nz).astype(np.float32)
    c4 = np.linspace(0.0, 5000.0, nz).astype(np.float32)
    th = (6.0 * rng.standard_normal((nz, ny, nx))).astype(np.float32)
    pp = (400.0 * rng.standard_normal((nz, ny, nx))).astype(np.float32)
    qv = (0.002 + 0.015 * rng.random((nz, ny, nx))).astype(np.float32)
    th_d, pp_d, qv_d = (cp.asarray(a) for a in (th, pp, qv))
    adjust_tempqv(cp.asarray(mub), cp.asarray(save_mub), cp.asarray(c3),
                  cp.asarray(c4), 10000.0, th_d, pp_d, qv_d,
                  use_theta_m=use_theta_m)
    th_m, qv_m = np_adjust_tempqv(mub, save_mub, c3, c4, 10000.0, th, pp,
                                  qv, use_theta_m=use_theta_m)
    assert ulp32(cp.asnumpy(th_d), np.float32(th_m)) <= MAX_ULPS
    assert ulp32(cp.asnumpy(qv_d), np.float32(qv_m)) <= MAX_ULPS
    np.testing.assert_array_equal(cp.asnumpy(pp_d), pp)   # pp is read-only


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("nri,stagger", [(3, ""), (3, "x"), (3, "y"),
                                         (4, ""), (4, "x"), (4, "y")])
def test_gpu_copy_fcn_matches_mirror(nri, stagger):
    import cupy as cp
    from gpuwm.core.nest_interp import copy_fcn
    rng = np.random.default_rng(20 + nri)
    reg = _feedback_case(nri, stagger, child_cells=5, parent=12)
    parent = (5.0 + rng.random((4, reg.nyp, reg.nxp))).astype(np.float32)
    child = (5.0 + rng.random((4, reg.nyc, reg.nxc))).astype(np.float32)
    dev = cp.asarray(parent)
    copy_fcn(dev, cp.asarray(child), reg)
    mirror = np_copy_fcn(parent, child, i_parent_start=4, j_parent_start=4,
                         nri=nri, nrj=nri, xstag=reg.xstag, ystag=reg.ystag)
    assert ulp32(cp.asnumpy(dev), np.float32(mirror)) <= MAX_ULPS


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("nri", [3, 4])
def test_gpu_copy_fcnm_fcni_match_mirror(nri):
    import cupy as cp
    from gpuwm.core.nest_interp import copy_fcni, copy_fcnm
    rng = np.random.default_rng(30 + nri)
    reg = _feedback_case(nri, "", child_cells=5, parent=12)
    parent = rng.random((2, reg.nyp, reg.nxp)).astype(np.float32)
    child = rng.random((2, reg.nyc, reg.nxc)).astype(np.float32)
    dev = cp.asarray(parent)
    copy_fcnm(dev, cp.asarray(child), reg)
    mirror = np_copy_fcnm(parent, child, i_parent_start=4, j_parent_start=4,
                          nri=nri, nrj=nri)
    np.testing.assert_array_equal(cp.asnumpy(dev), np.float32(mirror))
    pi = (parent * 100).astype(np.int32)
    ci = (child * 100).astype(np.int32)
    devi = cp.asarray(pi)
    copy_fcni(devi, cp.asarray(ci), reg)
    mirror_i = np_copy_fcni(pi, ci, i_parent_start=4, j_parent_start=4,
                            nri=nri, nrj=nri)
    np.testing.assert_array_equal(cp.asnumpy(devi), mirror_i)


@requires_gpu
@pytest.mark.gpu
def test_gpu_sint_cancellation_real_mirror():
    # Shadow-review F1 companion: at the sign-changing cancellation
    # fixture the kernel must sit at the WRF REAL emulation
    # (np_sint dtype=np.float32) within the floor -- expected bitwise,
    # since -fmad=false gives per-op round-to-nearest FP32 on both sides
    # -- while the FP64 oracle is millions of ULPs away there (which is
    # why bounded fixtures feed the FP64-oracle parity tests above).
    import cupy as cp
    from gpuwm.core.nest_interp import sint
    reg = _mass_reg(nri=3, ipos=4, jpos=3, child=9, parent=13)
    z = -np.float64(reg.xig[0]) + 3.0e-8
    row = np.float32((np.arange(13, dtype=np.float64) - 4.0) - z)
    parent = np.tile(row, (13, 1))
    out = cp.asnumpy(sint(cp.asarray(parent), reg))
    real32 = np_sint(parent, reg, dtype=np.float32)
    fp64 = np_sint(parent, reg)
    ic = 3
    assert ulp32(out[:, ic], real32[:, ic]) <= MAX_ULPS
    assert ulp32(out[:, ic], np.float32(fp64[:, ic])) > MAX_ULPS


@requires_gpu
@pytest.mark.gpu
def test_gpu_device_tables_binding_contract():
    # Fix round (review finding 3): geometry binding is explicit and
    # idempotent-checked -- manifest-after-default fails loud (the F4
    # bypass), manifest binding is idempotent, drivers then reuse the
    # bound tables.
    import cupy as cp
    reg = _mass_reg()
    tabs = reg.device_tables()             # driver-style default upload
    assert reg.device_tables() is tabs     # idempotent
    with pytest.raises(RuntimeError, match="BEFORE first use"):
        reg.device_tables(alloc=lambda name, shape, dtype:
                          cp.empty(shape, dtype=dtype))
    reg2 = _mass_reg()
    calls = []

    def manifest_alloc(name, shape, dtype):
        calls.append((name, tuple(shape)))
        return cp.empty(shape, dtype=dtype)

    bound = reg2.device_tables(alloc=manifest_alloc)
    n_alloc = len(calls)
    assert n_alloc == 6                    # ci/ip/cj/jp/xig/xjg
    assert [name for name, _ in calls] == [
        "ci", "ip", "cj", "jp", "xig", "xjg"]
    assert reg2.device_tables(alloc=manifest_alloc) is bound   # idempotent
    assert len(calls) == n_alloc           # no re-allocation
    assert reg2.device_tables() is bound   # drivers get the bound tables
