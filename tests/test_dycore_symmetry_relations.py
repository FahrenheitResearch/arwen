# tests/test_dycore_symmetry_relations.py
"""Symmetry relations of the dycore operator, asserted as tests.

WHY THIS FILE EXISTS.  Every dycore gate in this tree proves one of two
things: that the code has not CHANGED (byte-identical replay against
``tests/data/phase2_step_regression.npz``, the pinned kernel source
SHA-256s under ``gpuwm/certify/kernel_manifest.py``), or that it
CONSERVES (``domain_mass_measure`` closing against the boundary
tendency).  A compensating error is invariant under both.  Measured on
this branch, by injecting faults into scratch copies of the kernel
directory: an ``al_pp -> alt`` substitution in the terrain-following
pressure gradient drives an at-rest atmosphere to 108.5 m/s while the
domain mass drift stays at 2.064e-09 -- IDENTICAL to the clean run in
every digit -- and a one-cell shift of the x advection stencil moves the
drift only from 5.251e-10 to 1.470e-09.  A source pin reports that the
kernel changed; it never reports that it changed wrongly.

The relations below are the first assertions in this tree that constrain
the STRUCTURE of the operator rather than its bytes or its integrals.
They come from the symmetry group of the governing equations (space and
time translation, Galilean boost, rotation, scaling; Oberlack, J. Fluid
Mech. 447, 2001), read through the metamorphic-relation framing that
scientific software uses when no oracle exists (Chen, Cheung & Yiu 1998;
Chen et al., ACM Comput. Surv. 51(1), 2018; Kanewala & Bieman, Inf.
Softw. Technol. 56, 2014).  Grid-orientation independence is a standard
dycore test in its own right (Williamson et al., J. Comput. Phys. 102,
1992, cases 1-2 at rotation angles alpha = 0 and ~pi/2); well-balanced
schemes over terrain are the classic terrain-following problem (LeVeque,
J. Comput. Phys. 146, 1998; Klemp, Skamarock & Dudhia, Mon. Weather Rev.
135, 2007).

WHICH RELATIONS SURVIVE THIS DYCORE, AND WHICH DO NOT.

  R1  x-reflection             KEPT   (test_r1_*)
  R2  x<->y transpose          KEPT, e = 0 only (test_r2_*)
  R3  well-balancedness        KEPT   (test_r3_*)
  R4  Galilean boost           REJECTED, and not implemented
  R5  arbitrary-angle rotation REJECTED, and not implemented
  R6  continuous scaling       REJECTED, and not implemented

R4 is rejected on measurement, not taste.  ``advection.cu``'s ``flux5``
carries ``fabsf(vel)``, so the upwind dissipation coefficient is a
function of the advecting velocity and the operator is not linear in it
(Wicker & Skamarock, Mon. Weather Rev. 130, 2002; ARW Tech Note
NCAR/TN-556+STR).  Boosting u by U and comparing against the rest-frame
solution translated by an EXACT integer cell count -- so that no
interpolation enters -- gives max|diff| = 1.157e-02 K at U = 50 m/s
(1.24% of the signal) and 2.188e-02 K at U = 200 m/s (2.35%), four orders
of magnitude above the floor the relations below sit at, and not
converging.  No tolerance for it is derivable from the scheme, so the
test could only assert a number that came from nowhere.  R5 needs
interpolation off the C-grid for any angle that is not a multiple of 90
degrees, which injects more error than it measures; 90-degree multiples
are exactly R2 composed with R1, so R2 is the whole of the recoverable
content.  R6 is broken in the equations themselves by g and the EOS
constants.

WHAT EACH RELATION ACTUALLY CATCHES.  Six faults, each injected into a
scratch copy of the kernel directory and run in a fresh process, at the
configurations these tests use.  "worst" is the largest residual as a
multiple of this file's tolerance, so > 1.00x is a failing test:

  fault injected                R1        R2      mass drift   caught by
  ---------------------------------------------------------------------
  (clean)                    0.19x     0.18x     5.251e-10    --
  x advection stencil +1     7.05x     6.73x     1.470e-09    R1 and R2
  y advection stencil +1     0.21x     6.68x     0.000e+00    R2 only
  flux5 centre pair 2.7e-3   1.33x     0.19x     0.000e+00    R1 only
    asymmetric
  flux5 37.0 -> 37.3         0.28x     0.40x     1.050e-10    NEITHER
    (symmetric magnitude)
  Coriolis v-equation f      0.19x    23.85x     6.302e-10    R2 only
    sign flipped

Read that table before trusting either relation.  Three things in it
matter.

  * The two relations are COMPLEMENTARY, not redundant.  A y-only stencil
    error is invisible to an x-mirror and a purely x-directed asymmetry is
    invisible to a transpose; each relation catches a fault the other
    misses.
  * R1 is a PARITY grading test, so it is blind to any fault that
    preserves the grading.  Flipping the sign of a whole term does
    preserve it -- ``+f*<rv>`` and ``-f*<rv>`` are both odd under the
    mirror -- which is why the Coriolis sign flip sits exactly at R1's
    floor and is caught by R2 instead.
  * Every one of the six faults leaves the conservation gate green.  Two
    of them give a mass drift SMALLER than the clean run's.

WHAT NONE OF THEM CAN SEE.  All four measured.

  * Reflection and transpose see ASYMMETRY, not MAGNITUDE.  A symmetric
    coefficient error in the flux5 stencil (37.0 -> 37.3) sits at the
    floor in both, and so does a scaling of the Coriolis f.
  * The detection threshold is about 1e-3 relative stencil asymmetry.
    Sweeping the flux5 centre pair: 2.7e-4 sits at the floor at 20, 60 and
    120 steps; 2.7e-3 is caught.  That threshold is set by ``typedef
    float real`` in ``common.cuh`` -- the whole dycore is FP32 -- not by
    the test design.
  * Identity map factors mean the ``vxgm`` / msf-gradient code is NOT
    covered: with msf = 1, ``vxgm`` is identically zero.  Real-data
    Lambert map factors are outside all three relations.
  * Periodic boundaries only.  Nesting, and the open and specified
    boundary branches, are outside all three.

PRIOR ART IN THIS TREE, and why it is not enough.
``gpuwm/verify/cases/straka.py`` already asserts a self-symmetry relation
(``symmetry_err = max|thp - thp[:, :, ::-1]|``, gated at 0.05 K).  That
gate is left exactly as it is by this file, deliberately: it covers one
field, one case, one time level, in 2-D (ny = 1), dry, non-rotating,
flat, and it only runs in the case battery.  It is also a SINGLE-run
relation, which is why it can only be posed for f = e = 0 -- a symmetric
state is not preserved by a rotating operator.  Tightening it toward the
floor measured here is a separate decision with its own evidence.

COST.  Thirteen relation instances, 32^3 x 60 steps, about 8 s of GPU
wall on an RTX 3080 plus the cupy import.  Cheap enough to sit in the
ordinary ``-m gpu`` pytest run rather than the case battery.
"""

import math

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = pytest.mark.gpu

# --- Grid, shared by every relation -------------------------------------
NX = NY = NZ = 32
DX = DY = 1000.0
ZTOP = 10000.0
DT = 3.0

#: 60 rather than 20 because the SEPARATION grows with it.  The residual
#: floor is a random walk in sqrt(n) while a systematic fault accumulates
#: faster, so a longer run buys detection power at linear cost.  Measured
#: on the injected one-cell x-stencil shift: 2.40x the tolerance at 20
#: steps, 7.05x at 60, 8.66x at 120, against a clean floor that stays at
#: 0.19-0.26x of tolerance at all three.
NSTEPS = 60

#: FP32 unit roundoff.  Every tolerance below is expressed in these.
EPS32 = float(np.finfo(np.float32).eps)

#: Per-field scale the residual is pinned to.  This is the part that is
#: easy to get wrong: the reflection residual does NOT scale with the
#: perturbation amplitude, it scales with the BASE STATE.  Measured by
#: shrinking the initial amplitude 1000x, which leaves the absolute
#: theta' asymmetry where it was (4.6e-4 K against 5.2e-4 K).  The floor
#: is rounding of the large intermediates being differenced, so the wind
#: scale is the sound speed -- the acoustic substeps set the magnitude of
#: those intermediates -- and not the wind itself.
_C_SOUND = 340.0
_THETA_SCALE = 300.0

#: Tolerance coefficients, fitted from the step-count sweep and then given
#: roughly 5x headroom.  The floor is a random walk: the clean residual as
#: a fraction of ``K*sqrt(n)`` is 0.24 / 0.19 / 0.26 at n = 20 / 60 / 120,
#: i.e. flat, which is what licenses the sqrt(n) form.  The floor is also
#: independent of domain size (nx = 32 / 64 / 128 all land in the same
#: band), so K carries no resolution term.
_K = {"thp": 15.0, "mup": 10.0, "php": 5.0, "u": 30.0, "v": 30.0, "w": 50.0}


def _tol_ulp(field, n_steps=NSTEPS):
    """Tolerance for ``field``, in FP32 ulp of that field's own scale."""
    return _K[field] * math.sqrt(n_steps)


def _sounding(z):
    from gpuwm.core import constants as c
    return 300.0 * np.exp(1e-4 * np.asarray(z, float) / c.G)


def _cfg(**kw):
    from gpuwm.config import RunConfig
    kw.setdefault("case", "igw")
    return RunConfig(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, ztop=ZTOP, dt=DT,
                     run_seconds=NSTEPS * DT, **kw)


def _bell(h0, shift=0.0, flip=False):
    """A bell hill, off-centre so no accidental symmetry helps the test."""
    x = (np.arange(NX) + 0.5) * DX - 0.5 * NX * DX
    if flip:
        x = -x
    h = h0 * (10000.0 ** 2) / ((x - shift) ** 2 + 10000.0 ** 2)
    return np.broadcast_to(h[None, :], (NY, NX)).copy()


def _scales(base):
    """Per-field residual scales, from the base state actually in use."""
    return {"thp": _THETA_SCALE,
            "mup": float(np.max(np.abs(np.asarray(base.mub)))),
            "php": float(np.max(np.abs(base.phb))),
            "u": _C_SOUND, "v": _C_SOUND, "w": _C_SOUND}


# =======================================================================
# R3 -- WELL-BALANCEDNESS AT REST
# =======================================================================
#
# BREAKAGE THIS PREVENTS: a base-state quantity used where a perturbation
# quantity belongs, inside the terrain-following horizontal pressure
# gradient.  That is the classic terrain-following defect: the large
# horizontal gradient of the BASE pressure across a sloping coordinate
# surface stops cancelling, and the scheme spins motion out of a resting,
# hydrostatically balanced atmosphere.  Measured, by substituting the
# total inverse density ``alt`` for the perturbation ``al_pp`` in
# ``pgrad_face`` (acoustic.cu): the at-rest residual goes from 1.163e-03
# m/s to 108.5 m/s over a 2000 m hill, while the domain mass drift stays
# at 2.064e-09 -- identical to the clean run in every digit.  The
# conservation gate is structurally incapable of seeing this, and a
# byte-identity gate certifies it forever once committed.
#
# THE GATE IS THE HEIGHT-INDEPENDENCE OF THE RESIDUAL, NOT ITS SIZE.
# Clean, across a 200x range of hill height, the residual does not move:
#
#     h0 = 10 m    |u|max = 1.115e-03   |w|max = 1.489e-03 m/s
#     h0 = 500 m   |u|max = 8.774e-04   |w|max = 1.536e-03 m/s
#     h0 = 2000 m  |u|max = 1.163e-03   |w|max = 1.620e-03 m/s
#
# That is FP32 cancellation.  Under the injected metric defect the same
# sweep gives 5.570e-01 / 2.807e+01 / 1.085e+02 m/s -- a ratio of 194.85
# across a 200x change in slope, which is the signature of the defect
# itself.  An absolute threshold would have to sit loose enough to admit
# the clean round-off bath and would therefore admit a real
# slope-proportional term as well; the ratio separates them.
# ``gpuwm/verify/cases/hill2d.py`` records the same round-off-excited
# noise bath independently (w RMS ~ 3e-4 m/s).
#
# WHAT R3 CANNOT SEE, and the reason is worth knowing before anyone
# extends it: in the ARW perturbation-form pressure gradient EVERY term of
# ``pgrad_face`` is a product involving a perturbation quantity, so at
# rest the whole gradient is analytically zero and scaling any one of its
# coefficients multiplies zero.  Scaling the base-geopotential half of
# the metric term by 1e-4, 1e-3 and 1e-2 was injected and none of the
# three moved the residual at all.  Well-balancedness is a STRUCTURAL
# property of this discretization; what R3 catches is a change that
# breaks the structure, not one that perturbs a coefficient.

def _at_rest_residual(h0):
    """Max |u|, |v|, |w|, |thp|, |mup| after NSTEPS from exact rest."""
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    cfg = _cfg(case="hill2d", terrain_opt=(1 if h0 else 0),
               hill_height=(h0 or 10.0))
    coord = make_vertical_coord(cfg.nz)
    tz = _bell(h0) if h0 else None
    base = make_base_state(coord, _sounding, p_surf=cfg.p_surf,
                           ztop=cfg.ztop, terrain_z=tz)
    s = init_at_rest(cfg, coord, base, terrain_z=tz)
    run_steps(s, cfg, n=NSTEPS)
    return {n: float(np.abs(cp.asnumpy(getattr(s, n))).max())
            for n in ("u", "v", "w", "thp", "mup")}


@requires_gpu
def test_r3_flat_at_rest_is_exactly_at_rest():
    """Over flat terrain a resting balanced state stays EXACTLY resting.

    u, v, theta' and mu' are bitwise zero after 60 steps -- measured
    0.000e+00, not "small" -- so these are true equality assertions and
    cost nothing to hold.  Injecting a 1e-3 leak of the base geopotential
    into the perturbation geopotential difference breaks them at 4.584
    m/s, so they are not vacuous.  w is not zero: it carries a round-off
    bath of 9.041e-04 m/s, so it is bounded rather than zeroed, at 5.5x
    headroom.  The exact zeros are what carries the signal here; the w
    bound is a pinned observation present only so a divergence cannot slip
    past unremarked.
    """
    r = _at_rest_residual(0.0)
    assert r["u"] == 0.0, f"flat at-rest u went non-zero: {r['u']:.3e} m/s"
    assert r["v"] == 0.0, f"flat at-rest v went non-zero: {r['v']:.3e} m/s"
    assert r["thp"] == 0.0, f"flat at-rest theta': {r['thp']:.3e} K"
    assert r["mup"] == 0.0, f"flat at-rest mu': {r['mup']:.3e} Pa"
    assert r["w"] < 5.0e-3, (
        f"flat at-rest |w|max = {r['w']:.3e} m/s, above the 5e-3 m/s "
        "round-off bath bound (measured 9.041e-04)")


@requires_gpu
def test_r3_terrain_residual_does_not_scale_with_hill_height():
    """The at-rest residual over terrain must not grow with the slope.

    A base-state quantity standing in for a perturbation quantity in the
    terrain-following pressure gradient leaves a residual proportional to
    the coordinate slope; FP32 cancellation does not.  Measured across a
    200x change in hill height the clean ratio is 1.04 (1.115e-03 m/s at
    h0 = 10 m against 1.163e-03 at h0 = 2000 m) and the ratio under the
    injected ``al_pp -> alt`` defect is 194.85.  A 3x band therefore
    separates them by nearly two orders of magnitude in either direction.
    """
    lo = _at_rest_residual(10.0)
    hi = _at_rest_residual(2000.0)
    for field in ("u", "w"):
        a, b = lo[field], hi[field]
        assert a > 0.0, (
            f"h0 = 10 m gave |{field}|max exactly 0 -- this probe is not "
            "measuring anything, the terrain is not reaching the dycore")
        ratio = b / a
        assert 1.0 / 3.0 < ratio < 3.0, (
            f"|{field}|max scales with hill height: {a:.3e} at h0 = 10 m "
            f"against {b:.3e} at h0 = 2000 m (ratio {ratio:.2f}, over a "
            "200x change in slope).  A slope-proportional at-rest residual "
            "is the terrain-following pressure gradient using a base-state "
            "quantity where a perturbation belongs, not round-off.  The "
            "conservation gate cannot see this: the injected form of it "
            "left the domain mass drift identical to the clean run.")
        # Divergence backstop, not the relation: a blow-up that happened
        # to scale equally at both heights would pass the ratio test.
        assert b < 1.0e-1, (
            f"|{field}|max = {b:.3e} m/s at h0 = 2000 m: the at-rest state "
            "is not at rest at all")


# =======================================================================
# R1 / R2 -- the two-run reflection and transpose relations
# =======================================================================
#
# Both are run in TWO-RUN form: integrate a reference initial condition,
# integrate the transformed initial condition with the transformed
# parameters, and require the second result to be the transform of the
# first.  The two-run form is what keeps Coriolis INSIDE the relation.
#
# R1, x-reflection:  x -> -x,  u -> -u,  f -> -f,  e -> -e,
#                    sina -> -sina,  terrain mirrored.
# Working the kernels line by line is what fixes those parameter flips.
# ``flux5`` is exactly reflection-antisymmetric: reversing its six stencil
# arguments and negating ``vel`` negates the flux term for term with
# identical groupings, and IEEE round-to-nearest and sm_120's FTZ are both
# sign-symmetric, so the relation is exact in exact arithmetic and
# rounding-limited in FP32.  ``coriolis_map.cu`` closes only with all
# three parameter flips: the u equation's ``-<e><cosa><rw>`` needs e to
# flip and cosa not to, the v equation's ``+<e><sina><rw>`` needs sina to
# flip as well, and the RERADIUS curvature terms are mirror-invariant with
# no flip at all.  Mirroring east-west swaps hemispheres; that is what the
# f and e sign flips are.  All three flips do work, measured:
# omitting the f flip alone puts the residual at 24.76x tolerance, the e
# flip at 13.93x, the sina flip at 3.14x.
#
# R2, x<->y transpose:  u <-> v,  f -> -f.  Requires nx == ny, dx == dy.

#: Parity of each field under R1.  -1 = the mirrored field is negated.
_PARITY_X = {"thp": 1, "mup": 1, "php": 1, "u": -1, "v": 1, "w": 1,
             "qv": 1, "qc": 1}

#: The fields R1 and R2 assert on.  Moist species are deliberately absent;
#: see ``test_r1_survives_moisture_limiter_and_microphysics``.
_DYNAMIC = ("thp", "mup", "php", "u", "v", "w")


def _mirror_x(a, parity):
    m = np.ascontiguousarray(a[..., ::-1])
    return -m if parity < 0 else m


def _swap_xy(a):
    return np.ascontiguousarray(np.swapaxes(a, a.ndim - 2, a.ndim - 1))


def _reference_fields(cfg, moist, seed=11):
    """The reference initial condition, as host arrays.

    Deliberately generic: pseudorandom winds, so every stencil sees a
    non-smooth field rather than one a low-order scheme happens to
    reproduce well.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((NZ, NY, NX + 1)).astype(np.float32) * 3.0
    v = rng.standard_normal((NZ, NY + 1, NX)).astype(np.float32) * 3.0
    w = np.zeros((NZ + 1, NY, NX), np.float32)
    w[1:NZ] = rng.standard_normal((NZ - 1, NY, NX)).astype(np.float32) * 0.02
    u[..., NX] = u[..., 0]          # periodic duplicate face
    v[:, NY, :] = v[:, 0, :]        # periodic duplicate row
    out = {"u": u, "v": v, "w": w}
    if moist:
        out["qv"] = np.abs(rng.standard_normal(
            (NZ, NY, NX)).astype(np.float32)) * 2e-3
        out["qc"] = np.abs(rng.standard_normal(
            (NZ, NY, NX)).astype(np.float32)) * 1e-4
    return out


def _thp_func(x, z):
    """Off-centre in x, modulated in y, so neither R1 nor R2 is trivially
    satisfied by an initial condition that already has the symmetry."""
    zz = np.asarray(z)
    zz = zz if zz.ndim == 3 else zz[:, None, None]
    xx = np.asarray(x)[None, None, :]
    yy = (np.arange(NY) + 0.5)[None, :, None]
    g = (np.sin(np.pi * np.clip(zz, 0.0, ZTOP) / ZTOP)
         / (1.0 + ((xx - 7000.0) / 4000.0) ** 2)
         * (1.0 + 0.3 * np.cos(2.0 * np.pi * yy / NY)))
    return np.broadcast_to(g, (NZ, NY, NX)).copy()


def _coriolis_fields(f0, e0, sina0):
    """Spatially varying f/e, so the kernel's own ``<f>_x`` and ``<e>_x``
    face averaging is exercised rather than differencing a constant."""
    i = (np.arange(NX) + 0.5)[None, :]
    shape_x = 1.0 + 0.3 * np.cos(2.0 * np.pi * i / NX)
    f = np.broadcast_to(f0 * shape_x, (NY, NX)).copy()
    e = np.broadcast_to(e0 * shape_x, (NY, NX)).copy()
    sina = np.full((NY, NX), sina0, float)
    cosa = np.full((NY, NX), math.sqrt(max(0.0, 1.0 - sina0 * sina0)), float)
    return f, e, sina, cosa


def _run_pair(transform, *, f0=0.0, e0=0.0, sina0=0.0, h0=0.0,
              moist=False, mp_physics=0, n_steps=NSTEPS):
    """Integrate the reference and the transformed run; return residuals.

    Returns ``(residual_ulp, tolerance_ulp)`` keyed by field, where the
    residual is ``max|transformed_result - transform(reference_result)|``
    divided by that field's scale and by the FP32 unit roundoff.
    """
    import cupy as cp
    from gpuwm.core.dycore import run_steps, set_w_surface
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_theta_perturbation

    if transform == "transpose" and not (NX == NY and DX == DY):
        raise AssertionError("R2 needs a square, isotropic grid")

    kw = dict(moist=moist, mp_physics=mp_physics)
    if h0:
        kw.update(case="hill2d", terrain_opt=1, hill_height=h0)
    cfg = _cfg(**kw)
    coord = make_vertical_coord(cfg.nz)

    def build(tz):
        base = make_base_state(coord, _sounding, p_surf=cfg.p_surf,
                               ztop=cfg.ztop, terrain_z=tz)
        return base, init_theta_perturbation(cfg, coord, base, _thp_func,
                                             terrain_z=tz)

    tz_ref = _bell(h0, shift=6000.0) if h0 else None
    base_ref, ref = build(tz_ref)

    if transform == "reflect":
        tz_img = _bell(h0, shift=6000.0, flip=True) if h0 else None

        def xform(a, name):
            return _mirror_x(a, _PARITY_X[name])
    else:
        tz_img = _swap_xy(tz_ref) if h0 else None

        def xform(a, name):
            return _swap_xy(a)

    base_img, img = build(tz_img)

    # INSTRUMENT CHECK, not a physics claim.  The base-state builder is
    # column-local, so transformed terrain must give the exactly
    # transformed base state.  If it does not, every residual below is
    # measuring the builder rather than the dycore.
    for name in ("mub", "phb"):
        a = np.asarray(getattr(base_ref, name))
        b = np.asarray(getattr(base_img, name))
        if a.ndim >= 2:
            want = _mirror_x(a, 1) if transform == "reflect" else _swap_xy(a)
            assert np.array_equal(b, want), (
                f"base-state {name} is not the exact {transform} image of "
                "itself; the reference for this relation is invalid")

    # The transformed run's initial condition is the transform of the
    # reference's OWN arrays, so the two states are exact images of one
    # another rather than two independent evaluations of a formula.
    # theta' and phi' are read back off the reference state instead of
    # being rebuilt from ``_thp_func``: the hydrostatic rebalancing inside
    # ``init_theta_perturbation`` would otherwise have to be ASSUMED
    # transform-equivariant, and this file is testing the dycore, not the
    # initializer.
    fields = _reference_fields(cfg, moist)
    for name in ("thp", "php"):
        fields[name] = cp.asnumpy(getattr(ref, name)).copy()
    #: Under the transpose the u equation becomes the v equation, so the
    #: transformed run's u is built from the reference's v.
    target = {"u": "v", "v": "u"} if transform == "transpose" else {}
    for name, arr in fields.items():
        getattr(ref, name)[...] = cp.asarray(arr)
        getattr(img, target.get(name, name))[...] = cp.asarray(
            xform(arr, name))

    if f0 or e0:
        f, e, sina, cosa = _coriolis_fields(f0, e0, sina0)
        ref.set_map_coriolis(f=f, e=e, sina=sina, cosa=cosa)
        if transform == "reflect":
            img.set_map_coriolis(f=-_mirror_x(f, 1), e=-_mirror_x(e, 1),
                                 sina=-_mirror_x(sina, 1),
                                 cosa=_mirror_x(cosa, 1))
        else:
            img.set_map_coriolis(f=-_swap_xy(f), e=_swap_xy(e),
                                 sina=_swap_xy(sina), cosa=_swap_xy(cosa))

    if h0:
        set_w_surface(ref, cfg)
        set_w_surface(img, cfg)

    run_steps(ref, cfg, n=n_steps)
    run_steps(img, cfg, n=n_steps)

    scales = _scales(base_ref)
    names = _DYNAMIC + (("qv", "qc") if moist else ())
    residual = {}
    for name in names:
        src = ({"u": "v", "v": "u"}.get(name, name)
               if transform == "transpose" else name)
        a = cp.asnumpy(getattr(ref, src)).astype(np.float64)
        b = cp.asnumpy(getattr(img, name)).astype(np.float64)
        scale = scales.get(name) or float(np.max(np.abs(a))) or 1.0
        residual[name] = float(
            np.abs(b - xform(a, src)).max()) / scale / EPS32
    tol = {n: _tol_ulp(n, n_steps) for n in _DYNAMIC}
    return residual, tol


def _assert_relation(label, residual, tol, breakage):
    bad = [f"{n}: {residual[n]:.1f} ulp against a {tol[n]:.1f} ulp "
           f"tolerance ({residual[n] / tol[n]:.1f}x)"
           for n in _DYNAMIC if residual[n] > tol[n]]
    assert not bad, (
        f"{label}: the relation does not hold.\n  "
        + "\n  ".join(bad)
        + "\nAll residuals (FP32 ulp of each field's scale): "
        + " ".join(f"{n}={residual[n]:.1f}" for n in sorted(residual))
        + f"\nWHAT THIS MEANS: {breakage}")


# --- R1 ----------------------------------------------------------------
#
# BREAKAGE THIS PREVENTS: an index or coefficient error in ONE direction
# of a stencil -- a scheme that stays perfectly conservative and perfectly
# reproducible while advecting wrongly.  Measured: shifting one arm of the
# periodic x flux5 stencil by a single cell puts the residual at 7.05x
# tolerance, and skewing the flux5 centre pair by 2.7e-3 puts it at 1.33x,
# while the domain mass drift under those two faults is 1.470e-09 and
# exactly 0.000e+00 against a clean 5.251e-10.  Conservation is blind to
# both.  R1 is a parity grading test, so what it detects is a fault that
# BREAKS the grading; see the fault table in the module docstring for what
# that excludes.

_R1_BREAKAGE = (
    "an index or coefficient error in the x direction of a stencil that "
    "breaks its reflection parity.  Such a fault leaves mass conservation "
    "and byte-identical replay both green -- measured, twice.")


@pytest.mark.parametrize("f0,e0,sina0", [
    pytest.param(0.0, 0.0, 0.0, id="nonrotating"),
    pytest.param(1e-4, 0.0, 0.0, id="f_only"),
    pytest.param(1e-4, 1e-4, 0.0, id="f_and_e"),
    pytest.param(1e-4, 1e-4, 0.3, id="f_and_e_rotated_grid"),
])
@requires_gpu
def test_r1_x_reflection(f0, e0, sina0):
    """Mirroring x, u, f, e and sina must mirror the whole solution.

    The four configurations walk the Coriolis kernel in: no rotation at
    all, then f, then the ``e = 2*Omega*cos(phi)`` terms, then a rotated
    map frame with sina != 0 so the ``<e><sina><rw>`` term in the v
    equation and the ``-sina*<rv>`` term in w are both live.  Each needs a
    different parameter flip, the relation closes only when every flip is
    right, and omitting any one of the three breaks it (24.76x, 13.93x and
    3.14x of tolerance respectively).
    """
    residual, tol = _run_pair("reflect", f0=f0, e0=e0, sina0=sina0)
    _assert_relation(f"R1 x-reflection (f={f0:g}, e={e0:g}, sina={sina0:g})",
                     residual, tol, _R1_BREAKAGE)


@requires_gpu
def test_r1_survives_terrain():
    """R1 holds over mirrored terrain, so terrain-following does not break
    it.  Measured at h0 = 0 / 500 / 2000 m the floor does not move (theta'
    21.3 / 21.3 / 17.1 ulp against a 116.2 ulp tolerance), which is what
    licenses using the same tolerance over a slope as over flat ground."""
    residual, tol = _run_pair("reflect", f0=1e-4, e0=0.0, h0=500.0)
    _assert_relation("R1 x-reflection over mirrored terrain (h0 = 500 m)",
                     residual, tol, _R1_BREAKAGE)


@requires_gpu
def test_r1_survives_moisture_limiter_and_microphysics():
    """R1 holds through the nonlinear positive-definite limiter and
    Kessler microphysics -- for the DYNAMIC fields only.

    The moist species are deliberately excluded from the assertion.
    Kessler's autoconversion and accretion thresholds are branches on
    floating-point comparisons, so a rounding difference of one ulp
    between the two runs flips a branch and the moist residual stops being
    a rounding quantity at all: measured qv 946 -> 1012 ulp and qc 750 ->
    898 ulp as the limiter and then microphysics come on, against dynamic
    fields that stay at the floor throughout (theta' 17.9 -> 25.6 ulp
    against a 116.2 ulp tolerance).  No tolerance for qv/qc is derivable
    from the scheme, and a fitted one would be a number that came from
    nowhere.  A relation for threshold-branching microphysics is separate
    work.
    """
    residual, tol = _run_pair("reflect", f0=1e-4, moist=True, mp_physics=1)
    _assert_relation("R1 x-reflection with the PD limiter and Kessler",
                     residual, tol, _R1_BREAKAGE)


# --- R2 ----------------------------------------------------------------
#
# BREAKAGE THIS PREVENTS: the x and y discretizations silently diverging,
# and any direction-paired term whose sign is wrong on one side of the
# pair.  ``flux_div_u`` and ``flux_div_v`` are separate CUDA kernels
# required to implement the same operator in two directions, and the
# scalar advection kernel writes its x and y flux5 stencils out
# separately; nothing else in this tree asserts that they agree.
# Measured, both halves: a one-cell shift of the Y advection stencil puts
# the residual at 6.68x tolerance while R1 sees nothing (0.21x), and
# flipping the sign of the Coriolis f term in the v equation alone puts it
# at 23.85x while R1 again sees nothing.  The mass drift under those two
# faults is 0.000e+00 and 6.302e-10, against a clean 5.251e-10.

_R2_BREAKAGE = (
    "the x and y discretizations have diverged, or a direction-paired "
    "term carries the wrong sign on one side.  A change landed on the x "
    "path and not the y path is exactly this, and conservation and byte "
    "identity are both blind to it -- measured, twice.")


@pytest.mark.parametrize("f0", [
    pytest.param(0.0, id="nonrotating"),
    pytest.param(1e-4, id="f_only"),
])
@requires_gpu
def test_r2_xy_transpose(f0):
    """Transposing x and y, swapping u and v and flipping f must transpose
    the solution.  ``e`` is zero in both configurations, and that
    exclusion is deliberate rather than convenient -- see
    ``test_r2_transpose_is_broken_by_the_e_coriolis_term``."""
    residual, tol = _run_pair("transpose", f0=f0, e0=0.0)
    _assert_relation(f"R2 x<->y transpose (f={f0:g}, e=0)",
                     residual, tol, _R2_BREAKAGE)


@requires_gpu
def test_r2_transpose_is_broken_by_the_e_coriolis_term():
    """The e exclusion in R2 is pinned as a deliberate break, so that it
    is not quietly widened later.

    ``e = 2*Omega*cos(phi)`` distinguishes EAST: ``coriolis_map.cu`` gives
    the u equation ``-<e>*<cosa>*<rw>`` and the w equation
    ``+e*(cosa*<ru> - sina*<rv>)``, and there is no plain-e counterpart in
    the v equation at all.  The transpose relation MUST therefore fail
    with e != 0.  That is correct physics, not a defect, and this test
    asserts the failure so a future change cannot make R2 look as though
    it covers the e terms.  Measured: u goes to 1565.1 ulp against 30.5 at
    e = 0, a factor of 51.
    """
    off, _ = _run_pair("transpose", f0=1e-4, e0=0.0)
    on, _ = _run_pair("transpose", f0=1e-4, e0=1e-4)
    ratio = on["u"] / off["u"]
    assert ratio > 8.0, (
        f"the e-Coriolis term no longer breaks the x<->y transpose: u "
        f"residual {on['u']:.1f} ulp with e = 1e-4 against {off['u']:.1f} "
        f"ulp with e = 0 (ratio {ratio:.1f}, expected the order of 51). "
        "Either the e terms have stopped being applied at all, or an "
        "east-distinguishing term has been made symmetric in x and y. "
        "Both are defects; neither is a reason to widen R2 to cover e.")
