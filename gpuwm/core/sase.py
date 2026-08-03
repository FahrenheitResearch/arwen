"""SASE-L1 device launchers (kernels/sase.cu, stage-3 Tasks 2-4 + 6c).

Launchers for the SASE CUDA kernels: the horizontal top-hat test
filters, the structure-function sensor sums, the strain tensor (clamped
uniform/variable dz), the width-parameterized Germano lift, the
deviatoric model stress, the domain-level dynamic partition solve
(device FP64 Gram/projection accumulation, authority 2x2 tail on host),
the sensor-state assembly from device D2 sums, and the fused SPLIT SASE
step :func:`launch_sase_step` (S3-6c device mirror of the authority
``sase_split_step``, S3-6e governed: solve -> vertical channel ->
GOVERNED stress (dynamic eddy viscosity f-blended with the audited 2-D
Smagorinsky deformation diffusivity; smag production share bypasses e
to heat) -> horizontal-explicit tendencies + production split ->
per-column FP64 backward-Euler Thomas momentum solve -> implicit-flux
production -> damp_opt=3 production taper -> e update/clip-to-heat ->
implicit e-transport, in place, FP64 ledger reductions with the split
dKE_expl/dKE_impl channels).  The v0 explicit fused step and the
roll-based periodic vertical it stepped are RETIRED (S3-6b report
section 7); the frozen CPU ``sase_ref_step`` keeps the v0 history but
has no device consumer.  Array layout is gpuwm's standard
``(nz, ny, nx)`` with x fastest; all device fields are C-contiguous
float32 (``DTYPE``).  The FP64 verification authority is
:mod:`gpuwm.verify.sase_ref`; the physics constants used here are
imported from it so the device path cannot drift from the authority's
values.

Model integration (stage-3 Task 6, rewired by S3-6c) lives in
:mod:`gpuwm.core.physics` ``PhysicsDriver._run_sase``: it drives the
split step on the model's per-column ``(nz, ny, nx)`` layer
thicknesses (z_mode-3 device coefficient build), the horizontal K_h
scalar-mix launcher (:func:`launch_scalar_mix`) paired with the
implicit K_v/Pr_t(f) vertical scalar channel (S3-6g blended Prandtl
number; :func:`launch_implicit_vertical_diffusion` on the step's
returned ``kv`` field), the N^2 launcher (:func:`launch_n2`), and the
theta heat deposit.  The per-step temporaries deliberately remain CuPy-pool
allocations (see ``preflight.sase_workspace_phases`` for the bound
pairing contract and the rationale).
"""

from __future__ import annotations

import math

import numpy as np
import cupy as cp

from gpuwm.core.kernels import get_kernel_int_defines
from gpuwm.core.state import DTYPE
from gpuwm.verify.sase_ref import (BL89_MIX_EXP, BLACKADAR_LAMBDA, C_E,
                                   MAX_COLUMN_LEVELS,
                                   C_ED,
                                   C_ES, C_KS, C_KV, C_MOM_BG,
                                   C_S, CKS_BLEND_EXP, CP_AIR,
                                   E_MIN, EP2_RV, G_ACCEL, KARMAN,
                                   LS_COEF,
                                   N2_SCREEN, NU_BLEND_EPS, P0_REF,
                                   RD_AIR, RIB_CRIT, RIB_WSPD2_FLOOR,
                                   SFC_WSPD_FLOOR, SMAG_KM_CAP,
                                   RV_AIR,
                                   SVP1, SVP2, SVP3, SVPT0,
                                   SensorState,
                                   VENT_DEPTH_CAP, VENT_ENT_COEF,
                                   VENT_MASK, VENT_MB_COEF,
                                   VENT_MIN_RUN_CELLS,
                                   VENT_QT_STEP_CAP,
                                   VENT_SAT_ADJUST_ITERS,
                                   VENT_SIGW_SHARE,
                                   VENT_THETA_STEP_CAP,
                                   XLV, _FILTER_WEIGHTS,
                                   _blackadar_length, _solve_tail,
                                   _w_bound_tail, partition_cap,
                                   prandtl_blend)

#: The closure's compile-time integer tier, handed to the kernel loader
#: with every launch so the host constants below and the device
#: ``#define``\ s are one definition (S3-2 review fold-in: no host/device
#: TPB drift).  The lane that wrote this carried the pair in a shared
#: ``CUDA_INT_DEFINES`` dict inside :mod:`gpuwm.core.kernels`; this head
#: gives each translation unit its own tier through
#: :func:`~gpuwm.core.kernels.get_kernel_int_defines`, so the closure's
#: constants live with the closure and no scheme name reaches the
#: generic loader.  Values must stay positive powers of two: SASE_TPB
#: sizes shared-memory tree reductions (``for s = blockDim.x / 2``),
#: which silently drop lanes on a non-power-of-two block.
_INT_DEFINES: tuple[tuple[str, int], ...] = (
    ("SASE_KMAX", MAX_COLUMN_LEVELS),
    ("SASE_TPB", 128),
)
assert all(v > 0 and (v & (v - 1)) == 0 for _, v in _INT_DEFINES), (
    "SASE integer defines must be positive powers of two "
    "(they size tree-reduction blocks)")
_DEFINE_VALUES = dict(_INT_DEFINES)

#: Threads per block, single-sourced with the device ``SASE_TPB``.
_TPB = _DEFINE_VALUES["SASE_TPB"]
#: Max column depth of the in-thread FP64 Thomas sweeps (sase.cu
#: sase_thomas_*); single-sourced with the device ``SASE_KMAX``.
_KMAX = _DEFINE_VALUES["SASE_KMAX"]


def _kern(func: str):
    """One stable wrapper per SASE device symbol, compiled at the tier."""
    return get_kernel_int_defines("sase", func, _INT_DEFINES)


def _check_field(name: str, arr, shape) -> None:
    if (not isinstance(arr, cp.ndarray) or arr.shape != tuple(shape)
            or arr.dtype != DTYPE or not arr.flags.c_contiguous):
        raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                         f"array with shape {tuple(shape)}")


def _tile(shape):
    nz, ny, nx = shape
    return ((nx + _TPB - 1) // _TPB, ny, nz), (_TPB, 1, 1)


def _flat(n: int):
    return ((n + _TPB - 1) // _TPB,), (_TPB,)


def launch_box_filter(field, width: int, out=None):
    """Top-hat test filter of nominal width ``width``*grid, x then y.

    Horizontal directions wrap periodically; the vertical is untouched
    (authority ``box_filter``).  Returns a new array unless ``out`` is
    given.
    """
    if width not in _FILTER_WEIGHTS:
        raise ValueError(f"width must be one of {sorted(_FILTER_WEIGHTS)}, "
                         f"got {width}")
    if not isinstance(field, cp.ndarray) or field.ndim != 3:
        raise ValueError("field must be a 3-D CuPy array (nz, ny, nx)")
    _check_field("field", field, field.shape)
    nz, ny, nx = field.shape
    if out is None:
        out = cp.empty_like(field)
    else:
        _check_field("out", out, field.shape)
    grid, block = _tile(field.shape)
    kern = _kern("sase_box_filter")
    kern(grid, block, (field, out, np.int32(width),
                       np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def launch_structure_functions(u, v, w) -> dict[int, float]:
    """Domain-mean D2(r), r in {1, 2, 4} cells, horizontal directions.

    Device block partial sums with FP64 accumulators; the host finishes
    the reduction (authority ``structure_functions``).  Returns plain
    Python floats keyed by r.
    """
    shape = u.shape
    for name, arr in (("u", u), ("v", v), ("w", w)):
        _check_field(name, arr, shape)
    nz, ny, nx = shape
    ncell = nz * ny * nx
    nblocks = (ncell + _TPB - 1) // _TPB
    kern = _kern("sase_structure_partial")
    totals = np.zeros(3, dtype=np.float64)
    for comp in (u, v, w):
        partials = cp.zeros((3, nblocks), dtype=cp.float64)
        kern((nblocks,), (_TPB,),
             (comp, partials, np.int32(nz), np.int32(ny), np.int32(nx),
              np.int32(nblocks)))
        totals += 0.5 * cp.asnumpy(partials).sum(axis=1) / ncell
    return {1: float(totals[0]), 2: float(totals[1]), 4: float(totals[2])}


def _ddz_coefficients(dz_col, nz: int):
    """FP64 Lagrange coefficients for the variable-dz clamped stencil.

    Mirrors the authority ``_ddz_var`` exactly: layer-center heights from
    the cumsum-half-layer construction, coefficient-form grouping on the
    center spacings ``h_m``/``h_p``, one-sided edge rows over the edge
    center spacings.  All arithmetic here is FP64; only the final cast to
    FP32 meets the device.
    """
    t = np.asarray(dz_col, dtype=np.float64)
    if t.shape != (nz,):
        raise ValueError(f"dz_col must have shape ({nz},) on the shared-"
                         f"column host path (got {t.shape}); per-column "
                         "(nz, ny, nx) thicknesses must arrive as a "
                         "C-contiguous float32 CuPy field matching the "
                         "velocity shape (device coefficient build)")
    z = np.cumsum(t) - 0.5 * t
    h_m = z[1:-1] - z[:-2]
    h_p = z[2:] - z[1:-1]
    cm = np.zeros(nz)
    c0 = np.zeros(nz)
    cp_ = np.zeros(nz)
    cm[1:-1] = -(h_p / (h_m * (h_p + h_m)))
    c0[1:-1] = (h_p - h_m) / (h_p * h_m)
    cp_[1:-1] = h_m / (h_p * (h_p + h_m))
    return cm, c0, cp_, float(z[1] - z[0]), float(z[-1] - z[-2])


def _z_stencil(shape, dz=None, dz_col=None):
    """Kernel-argument pack for the shared vertical-derivative stencil.

    Returns ``(cm, c0, cp device arrays, dz, two_dz, h_lo, h_hi,
    z_mode)`` ready to splice into a kernel launch.  z_mode 0 is the
    uniform clamped vertical (authority uniform expressions), 1 the
    variable-spacing clamped stencil (authority ``_ddz_var`` coefficient
    form, FP64 host precompute),
    3 (Task 6, the model's terrain-following columns) the per-column
    clamped stencil: ``dz_col`` is a C-contiguous float32 device field
    of layer thicknesses matching ``shape``, and the three coefficient
    FIELDS are built on device in FP64 by ``sase_ddz_coefficients``
    (edge rows folded into the coefficient rows), so the per-step model
    ``dz`` never round-trips through the host while keeping the
    FP64-precompute precision contract of the (nz,) path.  The z_mode-2
    roll-based periodic vertical retired with the v0 explicit step
    (S3-6c): every surviving SASE vertical operator is clamped.
    """
    nz, ny, nx = (int(extent) for extent in shape)
    if (dz is None) == (dz_col is None):
        raise ValueError("exactly one of dz and dz_col must be given")
    if (dz_col is not None and isinstance(dz_col, cp.ndarray)
            and dz_col.ndim == 3):
        _check_field("dz_col", dz_col, (nz, ny, nx))
        if nz < 2:
            raise ValueError("per-column dz_col requires nz >= 2")
        coeffs = [cp.empty((nz, ny, nx), dtype=DTYPE) for _ in range(3)]
        grid, block = _flat(ny * nx)
        _kern("sase_ddz_coefficients")(
            grid, block, (dz_col, *coeffs, np.int32(nz), np.int32(ny),
                          np.int32(nx)))
        return (*coeffs, DTYPE(0.0), DTYPE(0.0), DTYPE(0.0), DTYPE(0.0),
                np.int32(3))
    if dz_col is not None:
        cm, c0, cp_, h_lo, h_hi = _ddz_coefficients(dz_col, nz)
        dz_s, two_dz, z_mode = 0.0, 0.0, 1
    else:
        cm = c0 = cp_ = np.zeros(nz)
        h_lo = h_hi = 0.0
        dz_s, two_dz = float(dz), 2.0 * float(dz)
        z_mode = 0
    return (cp.asarray(cm, dtype=DTYPE), cp.asarray(c0, dtype=DTYPE),
            cp.asarray(cp_, dtype=DTYPE), DTYPE(dz_s), DTYPE(two_dz),
            DTYPE(h_lo), DTYPE(h_hi), np.int32(z_mode))


def _thickness_args(shape, dz=None, dz_col=None):
    """Thickness-mode kernel arguments ``(t, dz, t_mode)`` for the
    per-column split-step kernels (``sase_thick`` contract).

    t_mode 0 -- uniform column: scalar ``dz``; ``t`` is a 1-element
    dummy the kernels never dereference (dedicated allocation, so no
    ``__restrict__`` aliasing analysis is needed).  t_mode 1 -- shared
    ``(nz,)`` column: the FP32 cast of the host thicknesses.  t_mode 3
    -- per-column ``(nz, ny, nx)`` float32 device field (the same
    object the z_mode-3 coefficient build consumes).  FP64 geometry is
    rebuilt in-kernel from the FP32 thicknesses (exact promotions;
    face spacings equal the authority's z-center differences exactly
    because the cumsum-half-layer construction telescopes).
    """
    nz, ny, nx = (int(extent) for extent in shape)
    if (dz is None) == (dz_col is None):
        raise ValueError("exactly one of dz and dz_col must be given")
    if dz_col is None:
        return cp.zeros(1, dtype=DTYPE), DTYPE(float(dz)), np.int32(0)
    if isinstance(dz_col, cp.ndarray) and dz_col.ndim == 3:
        _check_field("dz_col", dz_col, (nz, ny, nx))
        return dz_col, DTYPE(0.0), np.int32(3)
    t = np.asarray(dz_col, dtype=np.float64)
    if t.shape != (nz,):
        raise ValueError(f"dz_col must have shape ({nz},) on the shared-"
                         f"column host path (got {t.shape})")
    return cp.asarray(t, dtype=DTYPE), DTYPE(0.0), np.int32(1)


def launch_strain(u, v, w, *, dx: float, dy: float, dz: float | None = None,
                  dz_col=None):
    """Resolved strain tensor [xx, yy, zz, xy, xz, yz].

    Exactly one of ``dz`` (uniform spacing, authority uniform clamped
    expressions) or ``dz_col`` (layer thicknesses: shape ``(nz,)`` host
    array, or a same-shape C-contiguous float32 device FIELD for the
    per-column model columns -- authority ``_ddz_var`` coefficient form
    either way) must be given.  The vertical is clamped in every mode
    (the roll-based ``periodic_z`` operator retired with the v0
    explicit step, S3-6c).  Returns six new float32 device arrays.
    """
    shape = u.shape
    for name, arr in (("u", u), ("v", v), ("w", w)):
        _check_field(name, arr, shape)
    nz, ny, nx = shape
    zcm, zc0, zcp, dz_s, two_dz, h_lo, h_hi, z_mode = _z_stencil(
        shape, dz=dz, dz_col=dz_col)
    outs = [cp.empty(shape, dtype=DTYPE) for _ in range(6)]
    grid, block = _tile(shape)
    kern = _kern("sase_strain")
    kern(grid, block,
         (u, v, w, *outs, zcm, zc0, zcp,
          DTYPE(2.0 * float(dx)), DTYPE(2.0 * float(dy)),
          dz_s, two_dz, h_lo, h_hi, z_mode,
          np.int32(nz), np.int32(ny), np.int32(nx)))
    return outs


def launch_germano_lift(u, v, w, *, width: int):
    """Resolved lift L_ij = filt(vel_i*vel_j) - filt_i*filt_j at ``width``.

    The authority ``germano_lift`` is the width-2 special case; width 4
    is the general `_identity_rows` construction.  Returns six new
    float32 device arrays in the authority's ``_PAIRS`` order
    (uu, vv, ww, uv, uw, vw).
    """
    shape = u.shape
    for name, arr in (("u", u), ("v", v), ("w", w)):
        _check_field(name, arr, shape)
    ncell = int(np.prod(shape))
    filt = [launch_box_filter(a, width) for a in (u, v, w)]
    prods = [cp.empty(shape, dtype=DTYPE) for _ in range(6)]
    grid, block = _flat(ncell)
    _kern("sase_velocity_products")(
        grid, block, (u, v, w, *prods, np.int64(ncell)))
    fprods = [launch_box_filter(p, width) for p in prods]
    lifts = [cp.empty(shape, dtype=DTYPE) for _ in range(6)]
    _kern("sase_lift_combine")(
        grid, block, (*fprods, *filt, *lifts, np.int64(ncell)))
    return lifts


def launch_model_stress(e, s_list, c_nu: float, f: float,
                        delta_eddy: float, delta_mom: float):
    """SASE-L1 modeled SGS stress (authority 6-arg ``model_stress``).

    ``s_list`` is the [xx, yy, zz, xy, xz, yz] strain; the viscous term
    acts on the deviatoric strain and tau_zz is closed from the trace
    identity so ``tau_kk == 2*max(e, E_MIN)`` holds on device to ~1.5 ULP
    by construction (kernel comment has the analysis).  Returns six new
    float32 device arrays.
    """
    shape = e.shape
    _check_field("e", e, shape)
    if len(s_list) != 6:
        raise ValueError(f"s_list must have 6 components, got {len(s_list)}")
    for k, s in enumerate(s_list):
        _check_field(f"s_list[{k}]", s, shape)
    ncell = int(np.prod(shape))
    taus = [cp.empty(shape, dtype=DTYPE) for _ in range(6)]
    grid, block = _flat(ncell)
    kern = _kern("sase_model_stress")
    kern(grid, block,
         (e, *s_list, *taus,
          DTYPE(c_nu), DTYPE(f), DTYPE(delta_eddy), DTYPE(delta_mom),
          # S3-6g: the FIXED momentum-background constant (decision
          # table, authority docstring); FP32-identical to the former
          # DTYPE(C_K / PR_T).
          DTYPE(C_MOM_BG), DTYPE(E_MIN), np.int64(ncell)))
    return taus


def launch_governed_stress(e, s_list, c_nu: float, f: float, delta: float):
    """S3-6e RANS-governed horizontal stress (authority
    ``governed_stress``): the dynamic eddy viscosity f*c_nu*delta*sqrt(e)
    blended with the audited 2-D Smagorinsky deformation diffusivity at
    weight (1-f), constants (C_S, SMAG_KM_CAP, NU_BLEND_EPS)
    single-sourced from :mod:`gpuwm.verify.sase_ref`.  Returns
    ``(taus, km, r)``: six new stress arrays (trace closure as
    :func:`launch_model_stress`), the governed horizontal diffusivity
    FIELD km (serves the stress, the e-transport, and the scalar
    K_h = km/Pr_t(f)), and the smag-share field r in [0, 1] weighting
    the production heat bypass.
    """
    shape = e.shape
    _check_field("e", e, shape)
    if len(s_list) != 6:
        raise ValueError(f"s_list must have 6 components, got {len(s_list)}")
    for k, s in enumerate(s_list):
        _check_field(f"s_list[{k}]", s, shape)
    ncell = int(np.prod(shape))
    taus = [cp.empty(shape, dtype=DTYPE) for _ in range(6)]
    km = cp.empty(shape, dtype=DTYPE)
    r = cp.empty(shape, dtype=DTYPE)
    grid, block = _flat(ncell)
    kern = _kern("sase_model_stress_gov")
    kern(grid, block,
         (e, *s_list, *taus, km, r,
          DTYPE(c_nu), DTYPE(f), DTYPE(delta),
          DTYPE(C_S), DTYPE(SMAG_KM_CAP), DTYPE(NU_BLEND_EPS),
          DTYPE(E_MIN), np.int64(ncell)))
    return taus, km, r


def launch_dynamic_solve(u, v, w, e, *, dx: float, dy: float, dz: float,
                         delta: float, manufactured_lifts=None,
                         exclude_boundary_width: int = 0):
    """Domain-level dynamic partition solve (authority ``dynamic_solve``).

    Device side: for each test width (2, 4) build the basis fields at the
    authority's grouping level -- deviatoric fine strain premultiplied by
    ``-2*delta*sqrt(max(e, E_MIN))`` then box-filtered (the shared
    refiltered term), deviatoric coarse strain from the filtered
    velocities, the width-appropriate resolved lift with its trace
    removed -- and accumulate the five Gram/projection scalars ``a.a,
    a.b, b.b, a.r, b.r`` with FP64 in-kernel accumulators.  The eddy
    basis rides the test filter (``width*delta``); the momentum basis is
    grid-anchored (``delta``, unit weight).  Like the authority, the
    solve keeps the uniform-dz clamped strain.

    Host side: the 2x2 tail is line-identical to the authority's
    ``dynamic_solve`` (same np.linalg.cond gate at 1e12, same
    np.linalg.solve, same clip/recovery order and constants).

    ``manufactured_lifts`` (dict width -> six device arrays in ``_PAIRS``
    order) replaces the resolved lift for inverse-crime verification,
    mirroring the authority's keyword.  ``exclude_boundary_width > 0``
    (Task-6 fix round, registered specified-boundary adjudication)
    excludes the outer that-many rows on all four lateral edges from the
    Gram/projection reductions via the in-kernel mask in
    ``sase_solve_partial`` (see the kernel comment for why a mask
    preserves the golden-pinned deterministic reduction bitwise at
    width 0).  Returns ``(c_nu, f)`` floats.
    """
    shape = u.shape
    for name, arr in (("u", u), ("v", v), ("w", w), ("e", e)):
        _check_field(name, arr, shape)
    bw = int(exclude_boundary_width)
    if bw < 0 or (bw > 0 and (2 * bw >= u.shape[1] or 2 * bw >= u.shape[2])):
        raise ValueError(
            f"exclude_boundary_width={exclude_boundary_width} must be "
            f">= 0 and leave a non-empty interior for {u.shape[1]} x "
            f"{u.shape[2]}")
    if manufactured_lifts is not None:
        if set(manufactured_lifts) != {2, 4}:
            raise ValueError("manufactured_lifts must be keyed by widths "
                             f"{{2, 4}}, got {sorted(manufactured_lifts)}")
        for width, lift in manufactured_lifts.items():
            if len(lift) != 6:
                raise ValueError(f"manufactured_lifts[{width}] must have 6 "
                                 f"components, got {len(lift)}")
            for k, comp in enumerate(lift):
                _check_field(f"manufactured_lifts[{width}][{k}]", comp, shape)
    nz, ny, nx = shape
    ncell = nz * ny * nx
    nblocks = (ncell + _TPB - 1) // _TPB
    grid, block = _flat(ncell)
    # Width-independent pieces: fine strain and the refilter integrand
    # -2*delta*root_e*S_fine^dev (authority s_fine/refilt construction).
    s_fine = launch_strain(u, v, w, dx=dx, dy=dy, dz=dz)
    premul = [cp.empty(shape, dtype=DTYPE) for _ in range(6)]
    _kern("sase_basis_premultiply")(
        grid, block, (e, *s_fine, *premul,
                      DTYPE(delta), DTYPE(E_MIN), np.int64(ncell)))
    kern = _kern("sase_solve_partial")
    moments = np.zeros(5, dtype=np.float64)
    for width in (2, 4):
        filt = [launch_box_filter(a, width) for a in (u, v, w)]
        s_coarse = launch_strain(*filt, dx=dx, dy=dy, dz=dz)
        refilt = [launch_box_filter(p, width) for p in premul]
        lift = (list(manufactured_lifts[width])
                if manufactured_lifts is not None
                else launch_germano_lift(u, v, w, width=width))
        partials = cp.zeros((5, nblocks), dtype=cp.float64)
        kern((nblocks,), (_TPB,),
             (e, *s_coarse, *refilt, *lift, partials,
              DTYPE(width * delta), DTYPE(delta), DTYPE(E_MIN),
              np.int32(ny), np.int32(nx), np.int32(bw),
              np.int32(nblocks), np.int64(ncell)))
        moments += cp.asnumpy(partials).sum(axis=1)
        # Drop the six-field lift binding before the next width's
        # allocations: left bound, the width-2 lifts stay live through
        # the width-4 basis/lift construction and raise the solve's
        # simultaneous peak from 49 to 55 full fields.  Preflight's
        # sase_workspace_phases transcription counts on this drop (the
        # S3-5 pairing contract); the width-2 partials buffer
        # intentionally remains referenced and IS transcribed.  A pure
        # reference drop: no arithmetic is reordered, device results are
        # bitwise unchanged.
        lift = None
    aa, ab, bb, ar, br = (float(x) for x in moments)
    # Host tail: the authority's ``_solve_tail`` on the device-accumulated
    # moments (S3-6b extraction; formerly a verbatim copy of the
    # dynamic_solve tail -- arithmetic is unchanged bitwise).
    gram = np.array([[aa, ab], [ab, bb]])
    proj = np.array([ar, br])
    return _solve_tail(gram, proj)


def launch_sensor_state(u, v, w, *, e_mean: float) -> SensorState:
    """Sensor state from device D2 sums (authority ``sensor_state``).

    The structure functions come from the device FP64 block reduction
    (:func:`launch_structure_functions`); the scalar tail -- resolved
    energy, log-log slope fit, alpha clip, and the degenerate all-subgrid
    branch -- applies the authority's formulas host-side.
    """
    d2 = launch_structure_functions(u, v, w)
    e_res = 0.5 * d2[2]
    if min(d2.values()) <= 0.0:
        # Degenerate resolved field: everything is subgrid.
        return SensorState(alpha=1.0, slope=0.0, e_res=e_res)
    lr = np.log(np.array([1.0, 2.0, 4.0]))
    ld = np.log(np.array([d2[1], d2[2], d2[4]]))
    slope = float(np.polyfit(lr, ld, 1)[0])
    alpha = float(np.clip(e_mean / (e_mean + e_res), 0.0, 1.0))
    return SensorState(alpha=alpha, slope=slope, e_res=e_res)


def launch_bulk_richardson_zi(u, v, theta, *, dz: float | None = None,
                              dz_col=None, out=None):
    """Per-column bulk-Richardson BL height (authority
    ``bulk_richardson_zi``; S3-6f).  One FP64 in-thread column sweep per
    (j, i): the YSU-convention Rib crossing at the registered RIB_CRIT
    with the RIB_WSPD2_FLOOR wind floor, linear interpolation between
    the bracketing layer centers, first-interior-center floor and
    top-center no-crossing fallback (conventions pinned at the
    authority).  ``dz``/``dz_col`` follow the :func:`_thickness_args`
    contract.  Returns a ``(ny, nx)`` float32 device field.
    """
    shape = u.shape
    for name, arr in (("u", u), ("v", v), ("theta", theta)):
        _check_field(name, arr, shape)
    nz, ny, nx = shape
    t_arr, t_dz, t_mode = _thickness_args(shape, dz=dz, dz_col=dz_col)
    if out is None:
        out = cp.empty((ny, nx), dtype=DTYPE)
    elif (not isinstance(out, cp.ndarray) or out.shape != (ny, nx)
          or out.dtype != DTYPE or not out.flags.c_contiguous):
        raise ValueError(f"out must be a C-contiguous float32 CuPy array "
                         f"with shape {(ny, nx)}")
    grid, block = _flat(ny * nx)
    _kern("sase_zi_column")(
        grid, block,
        (u, v, theta, t_arr, t_dz, t_mode, out,
         DTYPE(RIB_CRIT), DTYPE(RIB_WSPD2_FLOOR), DTYPE(G_ACCEL),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def launch_w_sensor_moments(w, e, n2=None, *,
                            exclude_boundary_width: int = 0):
    """S3-6f w-sensor device moments: ``(d2w, count, e_mean)``.

    One deterministic FP64 block reduction (``sase_w_sensor_partial``)
    accumulates the N^2-screened w structure-function sums for r in
    {1, 2, 4}, the screen-passing cell count, and the interior floored-e
    sum; the host finishes the authority's normalizations (D2 =
    0.5*sum/count; e_mean over the interior cell count).  ``n2 is
    None`` treats every cell as neutral = passing (the authority
    convention); ``exclude_boundary_width`` masks anchor cells exactly
    like the solve reductions (specified-boundary adjudication).  The
    caller applies the authority ``_w_bound_tail`` to ``d2w[2]``.
    """
    shape = w.shape
    _check_field("w", w, shape)
    _check_field("e", e, shape)
    if n2 is not None:
        _check_field("n2", n2, shape)
    bw = int(exclude_boundary_width)
    nz, ny, nx = shape
    if bw < 0 or (bw > 0 and (2 * bw >= ny or 2 * bw >= nx)):
        raise ValueError(
            f"exclude_boundary_width={exclude_boundary_width} must be "
            f">= 0 and leave a non-empty interior for {ny} x {nx}")
    ncell = nz * ny * nx
    nblocks = (ncell + _TPB - 1) // _TPB
    # Restrict-alias idiom (see launch_sase_step): with n2 absent the
    # kernel receives ``e`` as a stand-in pointer gated hard off by
    # has_n2 == 0 -- never dereferenced through the n2 slot.
    n2_arg = e if n2 is None else n2
    partials = cp.zeros((5, nblocks), dtype=cp.float64)
    _kern("sase_w_sensor_partial")(
        (nblocks,), (_TPB,),
        (w, e, n2_arg, np.int32(0 if n2 is None else 1),
         DTYPE(N2_SCREEN), DTYPE(E_MIN), partials,
         np.int32(ny), np.int32(nx), np.int32(bw),
         np.int32(nblocks), np.int64(ncell)))
    sums = cp.asnumpy(partials).sum(axis=1)
    count = int(round(sums[3]))
    d2w = {r: (0.5 * float(sums[t]) / count if count else 0.0)
           for t, r in enumerate((1, 2, 4))}
    n_int = nz * (ny - 2 * bw) * (nx - 2 * bw)
    e_mean = float(sums[4]) / n_int
    return d2w, count, e_mean


def launch_implicit_vertical_diffusion(phi, kv, *, dt: float,
                                       kfac: float = 1.0,
                                       dz: float | None = None, dz_col=None,
                                       floor: float | None = None,
                                       sfc_flux=None, sfc_rho1=None,
                                       sfc_fac: float | None = None):
    """Backward-Euler vertical diffusion of ``phi`` in place (authority
    ``implicit_vertical_diffusion``): one FP64 Thomas sweep per column
    (in-thread registers/local, no global workspace), flux-form
    zero-flux operator with face diffusivity ``kfac * avg(kv)`` (the
    arithmetic face mean of the cell-centered ``kv`` field), thickness
    weights from ``dz``/``dz_col`` (the :func:`_thickness_args`
    contract).  ``kfac`` carries the channel convention (2 for the
    e-transport, 1/Pr_t(f) for the driver scalars -- S3-6g); ``floor``
    (if given)
    re-floors the FP32 result post-solve (the e channel's E_MIN fold).
    Unconditionally stable, max principle, conservative -- the
    authority's pinned M-matrix properties.  Returns ``phi``.

    S3-11b surface scalar-flux deposit (authority
    :func:`gpuwm.verify.sase_ref.surface_scalar_flux_deposit`;
    registered SFC_SCALAR_FLUX = "explicit-deposit-v1"): ``sfc_flux``
    -- a C-contiguous (ny, nx) float32 DIMENSIONAL surface-flux field
    (HFX [W m^-2] for the theta row, QFX [kg m^-2 s^-1] for the qv
    row; positive upward), given together with ``sfc_rho1``, the
    (ny, nx) float32 lowest-level MOIST density -- engages the
    explicit lowest-layer deposit

        phi*[0] = phi[0] + dt*sfc_flux/(sfc_rho1*sfc_fac*thick_0)

    fused into the bottom rhs of the SAME kernel launch, BEFORE the
    sweep (the authority's registered deposit-then-solve composition,
    op-order-identical in FP64 with no intermediate FP32 round of the
    deposited value -- kernel comment).  ``sfc_fac`` is the row
    constant: the authority CP_AIR for theta, omitted (= 1.0) for qv
    (FP-exact, so both authority rows ride one kernel bitwise);
    passed to the kernel as a scalar double.  A ``sfc_fac`` WITHOUT
    ``sfc_flux``/``sfc_rho1`` is REJECTED -- a lone row constant has
    nothing to scale and silently ignoring it would mask a dropped
    flux argument at a call site.  ``sfc_rho1`` MUST be
    the same moist rho1 the surface e source computes
    (``physics.sase_surface_rho1`` -- the S3-11a report obligation;
    positivity is the driver's contract, not re-checked here: device
    launchers do no data-dependent validation).  ZERO-FLUX IDENTITY
    (the S3-11a seam-off contract, device-strengthened): cells where
    ``sfc_flux`` is exactly +-0.0 take NO add at all (in-kernel
    guard), so their columns are BITWISE the ``sfc_flux=None`` sweep
    -- the seam is OFF-able only through the fluxes themselves.  The
    one documented FP divergence from the authority: at zero flux the
    authority's ``x + 0.0`` flips a -0.0 bottom value to +0.0 while
    the guarded kernel keeps it (unphysical inputs; bounded by the
    zero-flux identity gate).  qc/qi and the e-transport pass no
    ``sfc_flux`` (no surface source -- authority scope).

    nz == 1 (S3-6c review Minor, documented): the identity solve
    returns ``phi`` UNTOUCHED -- including the optional ``floor``,
    which is deliberately NOT applied on this path (mirroring the
    authority's faceless nz == 1 branch).  Safe for every current
    caller: the only floored channel is the split step's e-transport,
    whose input arrives pre-floored (e_clip >= E_MIN from the e-update
    kernel) and passes through unchanged; a future caller wanting the
    floor on unfloored nz == 1 data must apply it itself.  A
    ``sfc_flux`` on the faceless branch is REJECTED rather than
    silently dropped (surface heating must not vanish into the
    identity path; no model column is nz == 1).
    """
    shape = phi.shape
    _check_field("phi", phi, shape)
    _check_field("kv", kv, shape)
    nz, ny, nx = shape
    if (sfc_flux is None) != (sfc_rho1 is None):
        raise ValueError("sfc_flux and sfc_rho1 must be given together")
    if sfc_fac is not None and sfc_flux is None:
        raise ValueError("sfc_fac requires sfc_flux/sfc_rho1 (a lone row "
                         "constant has nothing to scale)")
    fac = 1.0 if sfc_fac is None else float(sfc_fac)
    if sfc_flux is not None:
        for name, arr in (("sfc_flux", sfc_flux), ("sfc_rho1", sfc_rho1)):
            if (not isinstance(arr, cp.ndarray) or arr.shape != (ny, nx)
                    or arr.dtype != DTYPE or not arr.flags.c_contiguous):
                raise ValueError(f"{name} must be a C-contiguous float32 "
                                 f"CuPy array with shape {(ny, nx)}")
        if not fac > 0.0:
            raise ValueError("sfc_fac must be strictly positive")
        if nz == 1:
            raise ValueError("sfc_flux is not supported on the faceless "
                             "nz == 1 identity path")
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds the Thomas column bound "
                         f"SASE_KMAX={_KMAX}")
    if nz == 1:
        return phi                             # no faces: identity solve
    t_arr, dz_s, t_mode = _thickness_args(shape, dz=dz, dz_col=dz_col)
    grid, block = _flat(ny * nx)
    # Gated-dummy pointers when the seam is off (the module's
    # restrict-alias idiom: kv is a const input of this kernel and the
    # aliased slots are never dereferenced under has_sfc == 0).
    has_sfc = sfc_flux is not None
    flux_arg = sfc_flux if has_sfc else kv
    rho_arg = sfc_rho1 if has_sfc else kv
    _kern("sase_thomas_scalar")(
        grid, block,
        (phi, kv, flux_arg, np.int32(1 if has_sfc else 0), rho_arg,
         np.float64(fac), t_arr, dz_s, t_mode, DTYPE(dt), DTYPE(kfac),
         DTYPE(0.0 if floor is None else floor),
         np.int32(0 if floor is None else 1),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return phi


def launch_vertical_channel(e, theta, *, f: float, n2=None,
                            dz: float | None = None, dz_col=None,
                            z0: float = 0.0, kv=None, leps=None,
                            n2_dry=None):
    """S3-6h/S3-6i vertical channel at e^n (authority
    ``bl89_rans_lengths`` + ``stable_limit_coefficient`` composition
    inside ``sase_split_step`` step 2): one FP64 in-thread column
    sweep per (j, i) builds l_B, the frozen l_les = min(l_B, l_s),
    the BL89 displacement pair l_up/l_down against the LIVE theta
    profile (exact segment quadratures + the stable quadratic
    fractional-segment solve -- kernel comment has the mirror
    derivation), the registered combinations (the -2/3-power mean for
    mixing, the min for dissipation), the kappa-z-matched RANS pair
    l_mix_r = min(l_les, l_mix_BL89) / l_eps_r = min(l_B, l_eps_BL89),
    the S3-6i decoupled stable-limit coefficient C_r = C_KV +
    (C_KS/LS_COEF - C_KV)*min(l_mix_r/l_s, 1)^CKS_BLEND_EXP (C_KV in
    neutral/unstable cells exactly; the C_KS*e/N stable limit where
    l_s binds), and the two-product K blend.  Writes ``kv`` =
    f*(C_KV*l_les*sqrt(e)) + (1-f)*(C_r*l_mix_r*sqrt(e)) (f = 1
    FP-exact LES limb) and ``leps`` = l_eps_r (the RANS limb of the
    e-update kernel's l_d blend).  Constants single-sourced from the
    authority registry (BL89_MIX_EXP, C_KS et al.).  ``dz``/``dz_col``
    follow the :func:`_thickness_args` contract; ``n2`` may be None
    (test boxes; the restrict-alias dummy idiom).  Returns
    ``(kv, leps)``.

    SASE-M1b seam (S4-3c device twin of the authority
    ``bl89_rans_lengths`` ``n2_dry`` seam; sase_ref module docstring,
    SASE-M1b section, MOIST_MASTER_LENGTH =
    "bl89-n2eff-excursion-min-v1"): with ``n2_dry`` given (the DRY
    field, so ``n2`` must be the M1 effective field), the
    M1-substituted cells -- n2 != n2_dry bitwise, the seam's own mask
    -- have BOTH composed lengths additionally min-bounded by the
    moist parcel-excursion length l_m = min(l_up_m, l_down_m) of the
    BL89-family excursion integrals against n2_eff (kernel comment
    has the transcribed discretization), applied BEFORE the C_r
    evaluation exactly as the authority composes it.  Unsubstituted
    cells keep their bits verbatim (the kernel branch never runs);
    ``n2_dry=None`` (every pre-M1b caller) is bitwise the S3-6h
    channel (gated-dummy idiom).  ``n2_dry`` REQUIRES ``n2`` (the
    substitution mask derives from the pair -- the authority
    contract, mirrored).
    """
    shape = e.shape
    _check_field("e", e, shape)
    _check_field("theta", theta, shape)
    if n2 is not None:
        _check_field("n2", n2, shape)
    if n2_dry is not None:
        if n2 is None:
            raise ValueError(
                "n2_dry requires n2: the M1b substitution mask derives "
                "from the (n2_eff, dry) pair")
        _check_field("n2_dry", n2_dry, shape)
    nz, ny, nx = shape
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds the BL89 column bound "
                         f"SASE_KMAX={_KMAX}")
    t_arr, t_dz, t_mode = _thickness_args(shape, dz=dz, dz_col=dz_col)
    if kv is None:
        kv = cp.empty(shape, dtype=DTYPE)
    else:
        _check_field("kv", kv, shape)
    if leps is None:
        leps = cp.empty(shape, dtype=DTYPE)
    else:
        _check_field("leps", leps, shape)
    n2_arg = e if n2 is None else n2           # dummy pointer, gated off
    n2d_arg = e if n2_dry is None else n2_dry  # dummy pointer, gated off
    grid, block = _flat(ny * nx)
    _kern("sase_vertical_channel")(
        grid, block,
        (e, n2_arg, np.int32(0 if n2 is None else 1),
         n2d_arg, np.int32(0 if n2_dry is None else 1), theta,
         t_arr, t_dz, t_mode, kv, leps, DTYPE(z0),
         DTYPE(KARMAN), DTYPE(BLACKADAR_LAMBDA), DTYPE(C_KV),
         DTYPE(LS_COEF), DTYPE(E_MIN), DTYPE(f), DTYPE(G_ACCEL),
         DTYPE(BL89_MIX_EXP), DTYPE(C_KS), DTYPE(CKS_BLEND_EXP),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return kv, leps


def launch_blackadar_length(shape, *, dz: float | None = None, dz_col=None,
                            z0: float = 0.0, out=None):
    """S3-12 state-independent Blackadar length field l_B(z + z0)
    (authority :func:`gpuwm.verify.sase_ref._blackadar_length` on the
    ``_column_geometry`` layer centers): the RANS member of the additive
    dissipation channel's reference length
    ``l_ref = delta**f * l_B(z+z0)**(1-f)``
    (:func:`~gpuwm.verify.sase_ref.neutral_dissipation_length`); the
    e-update kernel forms the blend itself with its l_d endpoint-branch
    idiom.  GEOMETRY ONLY -- no e, no theta, no n2 (the
    state-independence the whole S3-12 amendment rests on), so the
    field is launched once per step and only under the additive switch.
    One FP64 in-thread column sweep per (j, i) in the AUTHORITY'S op
    order (uniform mode ``z_k = (k + 0.5)*dz`` as a product; thickness
    modes ``np.cumsum(t) - 0.5*t``), constants passed as doubles, one
    FP32 rounding at the store -- bitwise the FP32 image of the
    authority's FP64 l_B on shared inputs (pinned max ULP 0 in
    tests/test_sase_gpu.py).  ``dz``/``dz_col`` follow the
    :func:`_thickness_args` contract.  Returns ``out`` (allocated when
    not given).
    """
    nz, ny, nx = (int(extent) for extent in shape)
    t_arr, t_dz, t_mode = _thickness_args(shape, dz=dz, dz_col=dz_col)
    if out is None:
        out = cp.empty((nz, ny, nx), dtype=DTYPE)
    else:
        _check_field("out", out, (nz, ny, nx))
    grid, block = _flat(ny * nx)
    _kern("sase_blackadar_length")(
        grid, block,
        (t_arr, t_dz, t_mode, out,
         np.float64(z0), np.float64(KARMAN), np.float64(BLACKADAR_LAMBDA),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def _taper_weights(shape, dz=None, dz_col=None, zdamp: float = 0.0):
    """S3-6e damping-layer taper field g (authority
    ``damp_taper_weights``): 1 below ztop - zdamp, cos^2 across the
    layer, 0 at the per-column top interface -- the damp_opt=3 KDH
    weight law complemented, at layer centers.  FP64 geometry (host for
    uniform/shared columns, device for the per-column thickness field
    -- the cumsum-half-layer construction either way), one FP32 cast at
    the end.  Returns a C-contiguous (nz, ny, nx) float32 device field.
    """
    nz, ny, nx = (int(extent) for extent in shape)
    zd = float(zdamp)
    if dz_col is not None and isinstance(dz_col, cp.ndarray):
        t64 = dz_col.astype(cp.float64)
        z_if = cp.cumsum(t64, axis=0)
        zc = z_if - 0.5 * t64
        htop = z_if[-1]
        arg = cp.clip((zc - (htop - zd)) / zd, 0.0, 1.0)
        g = 1.0 - cp.sin(0.5 * np.pi * arg) ** 2
        return cp.ascontiguousarray(g.astype(DTYPE))
    if dz_col is not None:
        t = np.asarray(dz_col, dtype=np.float64)
        z_if = np.cumsum(t)
        zc = z_if - 0.5 * t
        htop = float(z_if[-1])
    else:
        zc = (np.arange(nz, dtype=np.float64) + 0.5) * float(dz)
        htop = nz * float(dz)
    arg = np.clip((zc - (htop - zd)) / zd, 0.0, 1.0)
    g = (1.0 - np.sin(0.5 * np.pi * arg) ** 2).astype(np.float32)
    return cp.ascontiguousarray(
        cp.broadcast_to(cp.asarray(g)[:, None, None], (nz, ny, nx)))


def launch_sase_step(u, v, w, theta, e, *, dx: float, dy: float, dz: float,
                     delta: float, dt: float, n2=None, dz_col=None,
                     heat=None, exclude_boundary_width: int = 0,
                     z0: float = 0.0, zdamp: float | None = None,
                     ust=None, wspd_sfc=None, n2_moist=None,
                     stable_dissipation: bool = False,
                     additive_dissipation: bool = False):
    """One fused SPLIT SASE-L1 step on device arrays, in place (S3-6c).

    Device mirror of the authority :func:`gpuwm.verify.sase_ref.
    sase_split_step` -- explicit horizontal channel, implicit vertical
    channel -- superseding the v0 fully explicit step IN PLACE (same
    name, same driver seam).  The v0 formulation's instability
    (horizontal-scale nu stepped explicitly in the vertical, diffusion
    number O(10^2) at d01 parameters) is RESOLVED by S3-6b/6c: the
    vertical channel now rides K_v = C_KV*l_v*sqrt(e) on the Blackadar
    length and advances backward-Euler (unconditionally stable
    M-matrix Thomas columns), so no vertical CFL bound exists; the
    explicit horizontal channel keeps its own 13x CFL margin at d01
    scale (authority test derivations).

    Sequence (authority order of operations, pinned by the SPLIT_*
    trajectory goldens): dynamic partition solve (clamped uniform-dz
    strain, coefficients only, exactly as before) followed by the
    S3-6f PARTITION BOUNDS (authority step 1b, mesoscale sensing
    concession): the bulk-Richardson z_i column kernel -> interior-mean
    z_i -> the authority ``partition_cap`` Delta/z_i cap, and the
    N^2-screened w structure-function reduction -> the authority
    ``_w_bound_tail`` resolved-fraction bound, with
    f = min(f_solved, f_cap, f_w) the value every downstream consumer
    receives (governed stress, l_d blend, momentum-background weight,
    and -- S3-6h -- the vertical-channel length blend);
    then the vertical channel at e^n (S3-6h/S3-6i: the FP64
    per-column BL89/kappa-z composed build of
    :func:`launch_vertical_channel` -- K_v = f*C_KV*min(l_B, l_s)*
    sqrt(e) + (1-f)*C_r*min(l_B, l_s, l_mix_BL89)*sqrt(e) against the
    live theta profile with C_r the S3-6i decoupled stable-limit
    coefficient (K_v -> C_KS*e/N where l_s binds), plus the
    l_eps_rans field the e-update's l_d blend consumes), full clamped
    strain + the
    S3-6e GOVERNED stress (authority ``governed_stress``: dynamic eddy
    viscosity f-blended with the audited 2-D Smagorinsky deformation
    diffusivity; the governed km field also serves the e-transport and
    the driver scalar channel), horizontal-explicit momentum
    tendencies + the P_h,e/P_h,heat production split (the smag share
    bypasses e straight to heat -- WRF km_opt=4 energetics, exactly
    bookkept), per-column FP64 backward-Euler Thomas solve for u, v, w
    (rhs = u + dt*du_h; zero-flux ends; flux-form thickness weights),
    implicit-flux production P_v from the solved fields, explicit e
    sources (K_v/Pr_t(f) buoyancy at the S3-6g blended Prandtl number,
    horizontal 2*K_m transport) tapered by
    the damp_opt=3 weight profile when ``zdamp`` is given (the S3-6e
    damping-layer taper: the withheld production share redirects to
    heat, the buoyancy channel tapers without redirect), the S3-6d
    ANALYTIC dissipation substep e* -> e*/(1 + b*dt)^2 with
    b = C_E*sqrt(max(e*, E_MIN))/(2*l_d) computed FP64 per cell (the
    exact decay solution; l_d is the blended length at e^n), the
    E_MIN clip with clip-to-heat, then the implicit 2*K_v vertical
    e-transport with the E_MIN re-floor.  ``u, v, w, e`` are updated
    in place; ``theta`` and ``n2`` are read-only; ``heat`` (same-shape
    float32, if given) receives the exact decay decrement minus
    clip_gain plus the smag bypass and taper redirect (no longer
    pointwise sign-definite -- authority module docstring).

    ``dz_col=None`` runs the uniform CLAMPED box (box mode assigns
    nominal heights z_k = (k+1/2)*dz) -- the split ledger theorem's
    domain needs only horizontal periodicity, so the parity and
    closure gates run here; there is no periodic vertical anywhere
    anymore.  A ``dz_col`` of layer thicknesses -- ``(nz,)`` shared
    column, or the model's per-column ``(nz, ny, nx)`` float32 device
    field (the driver's integration path) -- selects the variable-dz
    model mode, where the ledger is diagnostic only (unweighted
    telescoping breaks under 1/thick_k, authority docstring).

    S3-6j surface momentum stress (authority module docstring, S3-6j
    section): ``ust`` -- a C-contiguous (ny, nx) float32 friction-
    velocity field, or None -- engages the implicit drag bottom BC in
    the u/v Thomas sweeps: c = ust^2/max(|V1^n|, SFC_WSPD_FLOOR)
    folded into the bottom diagonal in-kernel (FP64; the YSU
    linearization), w/e/scalars unchanged (zero-flux; scope rationale
    at the authority).  The drag applies at ALL f (the lane's one
    intentional cross-limb change).  ``ust=None`` is the pre-S3-6j
    step bitwise (the drag arguments are gated dummies).  S3-9c
    (authority module docstring, S3-9c section): ``wspd_sfc`` -- the
    live sfclay gust-ENHANCED (ny, nx) float32 speed field, or None
    -- multiplies c by the audited YSU gustiness factor
    (spd1/max(wspd_sfc, 1e-9))^2 in-kernel (npref.py:6495-6496);
    ``wspd_sfc=None`` forms no factor (the S3-6j arithmetic bitwise;
    gated dummy pointer), and ``wspd_sfc`` without ``ust`` is
    rejected -- there is no drag row to correct.

    Returns the ledger dict ``{dKE, dE, dHeat, residual, dKE_sfc,
    dE_sfc_src, sfc_conv_resid, c_nu, f,
    dKE_expl, dKE_impl, f_solved, f_cap, f_w, zi, w_coverage, pr_t,
    kv, km_h}`` (the S3-6f scalars diagnose the partition bounds;
    ``f`` is the value the step USED, ``pr_t`` the S3-6g blended
    Prandtl number at that f) from deterministic FP64 in-kernel
    block reductions finished on host, with the split theorem's
    channels: dKE_expl pairs u^n against the horizontal deposit,
    dKE_impl pairs u^{n+1} against the implicit increment (with drag
    it CONTAINS the drag work -- the S3-6j identity), dE =
    sum(e_clip - e^n) - dt*sum(g*buoy) - dt*sum(t_h) (the
    implicit-transport increment excluded per the theorem; the
    buoyancy channel is the TAPERED one), dHeat sums the heat field.
    S3-6j channels: ``dKE_sfc`` is the measured drag work (third FP64
    momentum-kernel reduction, <= 0), ``residual`` the
    BOUNDARY-CONSISTENT closure dKE + dE + dHeat - dKE_sfc,
    ``dE_sfc_src`` the modeled u*^3/(kappa*0.5*thick_0) similarity
    deposit and ``sfc_conv_resid`` their DIAGNOSED mismatch (never
    forced closed) -- all exactly 0.0 with ``ust=None``.
    ``kv`` is the float32 vertical-diffusivity FIELD at e^n -- the
    driver's scalar channel rides it (K_v/Pr_t(f) faces); ``km_h`` is
    the governed HORIZONTAL diffusivity field (S3-6e) the driver's
    scalar channel rides as K_h = km_h/Pr_t(f); the driver pops both
    so the retained ledger stays scalar-only.

    SASE-M1 seam (S4-2 device twin of the authority ``sase_split_step``
    ``n2_moist`` seam; sase_ref module docstring, SASE-M1 section):
    ``n2_moist`` -- the :func:`launch_moist_n2` effective-stability
    field, or None -- substitutes at EXACTLY the authority's three
    spec points: it replaces ``n2`` as the field the vertical channel
    (l_les/l_s, BL89 RANS lengths, the C_r stable-limit coefficient)
    and the e-update's l_d stability min consume (points 1 and 3,
    coefficients only), and the e-update's buoyancy source becomes
    -(K_v/Pr_t)*N^2_m WHERE ``n2_moist`` departs bitwise from ``n2``
    (point 2; elsewhere the literal dry expression stands unchanged --
    the moist-n2 kernel copies unsaturated dry bits, so the FP32
    inequality is exactly the substituted-cell mask).  The step-1b
    w-sensor gravity-wave screen KEEPS the dry ``n2`` (not a
    substitution point -- the authority contract).  ``n2_moist=None``
    is the pre-M1 step bitwise (the dry-n2 slot is a gated dummy);
    ``n2_moist`` bitwise-equal to ``n2`` (a fully unsaturated field)
    is arithmetically inert -- the mask-false branch adds/changes
    nothing.  ``n2_moist`` REQUIRES ``n2`` (the substitution mask and
    the w-sensor screen both need the dry field -- the authority
    ValueError, mirrored).  The model driver chooses between the two
    calls with ``RunConfig.sase_moist_n2`` (default True passes the
    field, False passes None): EVERY M1 and M1b entry point hangs off
    this one argument, so the switch needs no second wire -- and
    ``None`` here disables the M1b limb below with it.

    SASE-M1b (S4-3c device twin of the authority ``sase_split_step``
    step-2b amendment; sase_ref module docstring, SASE-M1b section):
    with the seam engaged the step-2 vertical channel receives the
    dry field beside n2_eff (``n2_dry=n2`` -- exactly the authority's
    ``mlkw`` idiom), so the RANS-limb composed lengths are
    additionally min-bounded by the moist parcel-excursion length
    l_m = min(l_up_m, l_down_m) in the M1-substituted cells
    (MOIST_MASTER_LENGTH = "bl89-n2eff-excursion-min-v1") and the
    bound rides into (a) the kernel's K_v RANS limb (C_r evaluated AT
    the bounded length) and (b) the e-update's l_d blend through the
    exported leps field.  ``n2_moist=None`` remains bitwise the
    pre-M1 step (no n2_dry formed); f = 1 keeps kv bitwise the
    pre-limb LES limb (the FP-exact two-product argument -- the limb
    cannot reach the LES limit); unsaturated cells keep their bits
    verbatim.

    S3-6k seam (device twin of the authority ``sase_split_step``
    ``stable_dissipation`` seam; sase_ref module docstring, S3-6k
    section): ``stable_dissipation`` selects the DISSIPATION
    coefficient of the analytic decay substep -- C_E everywhere with
    the default False (the ``has_ces == 0`` gate leaves the kernel's
    multiplicand the literal c_e, bitwise the pre-S3-6k step), and the
    per-cell blend with True, so where the l_s stability length binds
    the dissipation length the limb decays at C_ES instead of C_E.
    Nothing else moves: no new device field is allocated, no workspace
    phase changes, ``sase_vertical_channel``'s own C_r is untouched,
    and the neutral/unstable cells and the whole f = 1 LES limb are
    bitwise unchanged with the seam ON.  The model driver sets it from
    ``RunConfig.sase_stable_dissipation``.

    S3-12 seam (device twin of the authority ``sase_split_step``
    ``additive_dissipation`` seam; sase_ref module docstring, S3-12
    section): ``additive_dissipation`` ADDS Deardorff's second,
    grid-scale dissipation channel to whichever base coefficient the
    S3-6k switch selected -- the e-update kernel forms
    C_eps += (1-f)*w*C_ED*(l_d/l_ref) on the SAME (l_d, e^n, n2_eff, f)
    it already holds, with l_ref = delta**f * l_B(z+z0)**(1-f) blended
    from the :func:`launch_blackadar_length` field at the l_d blend's
    own endpoint branches.  The two switches COMPOSE exactly as the
    authority's do (S3-6k selects the base, S3-12 adds to it).  With
    the default False the kernel gate ``has_ced == 0`` adds NOTHING --
    bitwise the pre-S3-12 step, no l_B field allocated or launched (the
    lb slot rides the gated-dummy idiom); with it True the one new
    device field is the state-independent l_B (geometry only, computed
    once per step before the e update), and the neutral/unstable cells,
    the ``n2 is None`` test boxes and the whole f = 1 LES limb remain
    bitwise unchanged BY SELECTION (the shared ``ls_v > 0.0f`` gate).
    The model driver sets it from ``RunConfig.sase_additive_
    dissipation``.
    """
    shape = u.shape
    for name, arr in (("u", u), ("v", v), ("w", w), ("theta", theta),
                      ("e", e)):
        _check_field(name, arr, shape)
    if n2 is not None:
        _check_field("n2", n2, shape)
    if n2_moist is not None:
        if n2 is None:
            raise ValueError(
                "n2_moist requires n2: the M1 substitution mask derives "
                "from the dry field and the w-sensor screen keeps it")
        _check_field("n2_moist", n2_moist, shape)
    if heat is None:
        heat = cp.empty(shape, dtype=DTYPE)
    else:
        _check_field("heat", heat, shape)
    nz, ny, nx = shape
    if ust is not None and (
            not isinstance(ust, cp.ndarray) or ust.shape != (ny, nx)
            or ust.dtype != DTYPE or not ust.flags.c_contiguous):
        raise ValueError(f"ust must be a C-contiguous float32 CuPy array "
                         f"with shape {(ny, nx)}")
    # S3-9c: the gustiness factor corrects the drag row, which does not
    # exist without a friction velocity (authority contract).
    if wspd_sfc is not None and ust is None:
        raise ValueError("wspd_sfc requires ust")
    if wspd_sfc is not None and (
            not isinstance(wspd_sfc, cp.ndarray)
            or wspd_sfc.shape != (ny, nx) or wspd_sfc.dtype != DTYPE
            or not wspd_sfc.flags.c_contiguous):
        raise ValueError(f"wspd_sfc must be a C-contiguous float32 CuPy "
                         f"array with shape {(ny, nx)}")
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds the Thomas column bound "
                         f"SASE_KMAX={_KMAX}")
    uniform = dz_col is None
    zcm, zc0, zcp, dz_s, two_dz, h_lo, h_hi, z_mode = _z_stencil(
        shape, dz=dz if uniform else None, dz_col=dz_col)
    t_arr, t_dz, t_mode = _thickness_args(
        shape, dz=dz if uniform else None, dz_col=dz_col)
    # Coefficients only: the solve intentionally keeps the clamped
    # uniform-dz strain (authority contract) in both modes.
    # exclude_boundary_width reaches ONLY the solve reductions (the
    # specified-boundary adjudication); the field updates stay
    # everywhere -- the driver owns the post-step e_sgs boundary floor
    # and the coupled-tendency masks.
    c_nu, f_solved = launch_dynamic_solve(
        u, v, w, e, dx=dx, dy=dy, dz=dz, delta=delta,
        exclude_boundary_width=exclude_boundary_width)
    # 1b. S3-6f partition bounds (authority sase_split_step step 1b /
    # module docstring S3-6f section): the Delta/z_i cap and the
    # N^2-screened w-based resolved-fraction bound cap the SOLVED f
    # before any consumer reads it; both reductions share the solve's
    # interior exclusion.  The zi field and the FP64 partials release
    # before the step's field allocations (preflight covering
    # superset).
    bw = int(exclude_boundary_width)
    zi_col = launch_bulk_richardson_zi(
        u, v, theta, dz=dz if uniform else None, dz_col=dz_col)
    zi_int = zi_col[bw:-bw, bw:-bw] if bw else zi_col
    zi = float(zi_int.mean(dtype=cp.float64))
    f_cap = partition_cap(delta, zi)
    # SASE-M1: the w-sensor screen KEEPS the dry n2 -- deliberately NOT
    # n2_eff (the authority's non-substitution point; docstring).
    d2w, w_count, e_mean = launch_w_sensor_moments(
        w, e, n2=n2, exclude_boundary_width=bw)
    n_int = nz * (ny - 2 * bw) * (nx - 2 * bw)
    wsens = _w_bound_tail(d2w[2], w_count, n_int, e_mean)
    f = min(f_solved, f_cap, wsens.f_w)
    # 1c. S3-6g regime-consistent Prandtl number at the used f
    # (authority sase_split_step step 1c / module docstring S3-6g
    # section): host-FP64 blend, one FP32 cast at the kernel argument;
    # FP-exact PR_LES at f = 1 (the former DTYPE(PR_T) value), PR_RANS
    # at f = 0.  The driver's scalar channels recompute the identical
    # value from the retained ledger f.
    pr_t = prandtl_blend(f)
    zi_col = zi_int = None
    ncol = ny * nx
    dims = (np.int32(nz), np.int32(ny), np.int32(nx))
    # INVARIANT (restrict-alias landmine, Task-6 carry-forward): with
    # ``n2 is None`` the kernels receive ``e`` as a stand-in pointer for
    # the const n2 slot, gated hard off by has_n2 == 0 -- both
    # sase_vertical_channel and sase_split_e_update read n2 ONLY inside
    # their ``if (has_n2)`` branches, so the aliased pointer is never
    # dereferenced and the ``__restrict__`` qualifiers see no
    # overlapping ACCESS.  The model driver's hot loop always passes a
    # real n2 field (physics._run_sase); the alias exists only on the
    # test-box path.  If either kernel ever gains an unconditional n2
    # read, replace this alias with a dedicated dummy allocation FIRST.
    # SASE-M1 (S4-2): n2_eff is what the stability machinery (points 1
    # and 3) consumes -- n2_moist when the seam is engaged, else the
    # SAME object as n2 (the pre-M1 step bitwise).  The dry-n2 slot of
    # the e-update kernel (the point-2 substitution mask) rides the
    # same gated-dummy idiom when the seam is off.
    n2_eff = n2 if n2_moist is None else n2_moist
    n2_arg = e if n2_eff is None else n2_eff   # dummy pointer, gated off
    has_n2 = np.int32(0 if n2_eff is None else 1)
    n2d_arg = e if n2_moist is None else n2    # dummy pointer, gated off
    has_moist = np.int32(0 if n2_moist is None else 1)
    # 2. Vertical channel at e^n (S3-6h/S3-6i): the two-product K
    #    blend f*(C_KV*l_les*sqrt(e)) + (1-f)*(C_r*l_mix_rans*sqrt(e))
    #    with the S3-6i decoupled stable-limit coefficient C_r, plus
    #    the l_eps_rans field the e-update's l_d blend consumes
    #    (:func:`launch_vertical_channel`; the launch happens AFTER
    #    step 1b because f now enters the channel).
    col_grid, col_block = _flat(ncol)
    # SASE-M1b (S4-3c; docstring above): with the seam engaged the
    # channel gets the dry field beside n2_eff and min-bounds the
    # RANS-limb lengths by the moist excursion length in the
    # substituted cells; with n2_moist=None the call is LITERALLY the
    # pre-M1b call (no kwarg formed) -- the authority mlkw idiom.
    mlkw = {} if n2_moist is None else {"n2_dry": n2}
    kv, leps = launch_vertical_channel(
        e, theta, f=f, n2=n2_eff, dz=dz if uniform else None,
        dz_col=dz_col, z0=z0, **mlkw)
    # 3. Full clamped strain + the S3-6e GOVERNED stress: one tau
    # evaluation plus the governed diffusivity field km (stress,
    # e-transport, and driver scalar K_h all ride it) and the smag
    # share r for the production heat bypass.
    if uniform:
        s = launch_strain(u, v, w, dx=dx, dy=dy, dz=dz)
    else:
        s = launch_strain(u, v, w, dx=dx, dy=dy, dz_col=dz_col)
    tau, km, r = launch_governed_stress(e, s, c_nu, f, delta)
    two_dx, two_dy = DTYPE(2.0 * float(dx)), DTYPE(2.0 * float(dy))
    zargs = (zcm, zc0, zcp, two_dx, two_dy, dz_s, two_dz, h_lo, h_hi,
             z_mode)
    grid, block = _tile(shape)
    # 6a. Horizontal e-flux integrands at e^n with the governed K_m
    # FIELD (S3-6e harmonization -- the bare-C_K km_coef prefactor and
    # its registered asymmetry are retired on this path; the vertical
    # leg remains the implicit 2*K_v transport).
    hflux = [cp.empty(shape, dtype=DTYPE) for _ in range(2)]
    _kern("sase_e_hflux")(
        grid, block,
        (e, km, *hflux, two_dx, two_dy, DTYPE(E_MIN), *dims))
    # 4. Horizontal-explicit tendencies + the S3-6e P_h,e/P_h,heat
    # production split (tau_zz never enters the split step -- its
    # vertical flux is remodeled by the K_v channel).
    tend = [cp.empty(shape, dtype=DTYPE)
            for _ in range(5)]                 # du,dv,dw,ph_e,ph_heat
    _kern("sase_split_tendencies")(
        grid, block,
        (tau[0], tau[1], tau[3], tau[4], tau[5], u, v, w, e, r, *tend,
         two_dx, two_dy, DTYPE(E_MIN), *dims))
    # Drop the 13-field strain/stress/smag-share binding before the
    # Thomas/P_v/partials allocations: the apply-phase peak sits at the
    # split-tendencies launch (preflight sase_workspace_phases pairing
    # contract; pure reference drop, no arithmetic reordered).  km
    # stays live: the e-update consumed nothing from it, but the driver
    # scalar channel rides it after return.
    s = None
    tau = None
    r = None
    # 5. Implicit vertical momentum channel: per-column FP64 Thomas
    # sweeps with the dKE_expl/dKE_impl/dKE_sfc ledger channels.
    # S3-6j: with ust given the u/v sweeps carry the implicit surface
    # stress (drag built in-kernel from the PRE-solve level-1 winds at
    # the FP64 SFC_WSPD_FLOOR); with ust=None the kernel receives kv
    # as a never-dereferenced stand-in gated hard off by has_drag == 0
    # (the module's restrict-alias idiom: kv is already a const input
    # of this kernel, so the alias sees only const reads and none
    # through the ust slot).
    nblocks_col = (ncol + _TPB - 1) // _TPB
    partials_m = cp.zeros((3, nblocks_col), dtype=cp.float64)
    ust_arg = kv if ust is None else ust       # dummy pointer, gated off
    # S3-9c: the sfclay enhanced-speed slot rides the same gated-dummy
    # idiom (kv is a const input of this kernel; the alias sees only
    # const reads and none through the wspd slot when has_wspd == 0).
    wspd_arg = kv if wspd_sfc is None else wspd_sfc
    _kern("sase_thomas_momentum")(
        col_grid, col_block,
        (u, v, w, tend[0], tend[1], tend[2], kv,
         ust_arg, np.int32(0 if ust is None else 1),
         wspd_arg, np.int32(0 if wspd_sfc is None else 1),
         np.float64(SFC_WSPD_FLOOR), t_arr, t_dz, t_mode,
         partials_m, DTYPE(dt), np.int32(nblocks_col), *dims))
    ph_e, ph_heat = tend[3], tend[4]
    tend = None                                # du/dv/dw consumed
    # P_v from the implicit-solved (stored) fields -- the theorem's
    # identity-(ii) pairing.
    pv = cp.empty(shape, dtype=DTYPE)
    _kern("sase_vertical_production")(
        grid, block, (u, v, w, kv, t_arr, t_dz, t_mode, pv, *dims))
    # S3-6e damping-layer taper weights (authority damp_taper_weights;
    # the driver passes cfg.zdamp under damp_opt == 3).  With no taper
    # the kernel receives ``e`` as a stand-in pointer gated hard off by
    # has_taper == 0 -- the same restrict-alias idiom as the n2 slot
    # (the pointer is never dereferenced; see the invariant above).
    if zdamp is not None and float(zdamp) > 0.0:
        taper = _taper_weights(shape, dz=dz if uniform else None,
                               dz_col=dz_col, zdamp=zdamp)
        taper_arg, has_taper = taper, np.int32(1)
    else:
        taper_arg, has_taper = e, np.int32(0)  # dummy pointer, gated off
    # S3-12: the additive channel's state-independent l_B reference
    # field (launcher docstring, S3-12 section) -- geometry only,
    # launched exactly when the seam is on; OFF passes leps as a
    # never-dereferenced stand-in gated hard off by has_ced == 0 (the
    # module's restrict-alias idiom: leps is already a const input of
    # this kernel, so the alias sees only const reads and none through
    # the lb slot).  Transcribed in preflight sase_workspace_phases as
    # an apply-phase covering-superset entry (the solve phase dominates
    # by ~20 fields, so the category bound does not move).
    if additive_dissipation:
        lb_ref = launch_blackadar_length(
            shape, dz=dz if uniform else None, dz_col=dz_col, z0=z0)
        lb_arg, has_ced = lb_ref, np.int32(1)
    else:
        lb_arg, has_ced = leps, np.int32(0)    # dummy pointer, gated off
    # 6-8. Explicit e sources (tapered), production split, clip-to-heat,
    # ledger partials.
    ncell = nz * ny * nx
    nblocks = (ncell + _TPB - 1) // _TPB
    partials_e = cp.zeros((4, nblocks), dtype=cp.float64)
    _kern("sase_split_e_update")(
        (nblocks,), (_TPB,),
        (e, heat, theta, n2_arg, has_n2, n2d_arg, has_moist,
         kv, leps, lb_arg, has_ced, ph_e, ph_heat, pv,
         *hflux, taper_arg, has_taper,
         partials_e, *zargs, DTYPE(dt), DTYPE(f), DTYPE(delta),
         DTYPE(pr_t), DTYPE(C_E), DTYPE(LS_COEF),
         # S3-6k: the stable-limb dissipation coefficient and its gate.
         # has_ces == 0 leaves the kernel's decay multiplicand the
         # literal c_e -- bitwise the pre-S3-6k step -- so the two
         # constants below are inert scalars on the default path.
         DTYPE(C_ES), DTYPE(CKS_BLEND_EXP),
         np.int32(1 if stable_dissipation else 0),
         # S3-12: the additive-channel constant; inert at has_ced == 0
         # on the same argument.
         DTYPE(C_ED),
         DTYPE(G_ACCEL),
         DTYPE(E_MIN), np.int32(nblocks), *dims))
    # 7b. Implicit vertical e-transport (2*K_v faces) + E_MIN re-floor.
    launch_implicit_vertical_diffusion(
        e, kv, dt=dt, kfac=2.0, dz=dz if uniform else None,
        dz_col=dz_col, floor=E_MIN)
    sums_m = cp.asnumpy(partials_m).sum(axis=1)
    sums_e = cp.asnumpy(partials_e).sum(axis=1)
    dke_expl = float(sums_m[0])
    dke_impl = float(sums_m[1])
    dke_sfc = float(sums_m[2])
    d_ke = dke_expl + dke_impl
    d_e = float(sums_e[0]) - float(sums_e[1]) - float(sums_e[2])
    d_heat = float(sums_e[3])
    # S3-6j diagnosed conversion channel (authority docstring, S3-6j
    # section): the modeled u*^3 similarity deposit over the same
    # columns, FP64 host reduction; 0.0 with ust=None (and dke_sfc is
    # then exactly the kernel's untouched zero partials).
    if ust is not None:
        ust64 = ust.astype(cp.float64)
        if t_mode == 0:
            t0 = float(t_dz)
        elif t_mode == 3:
            t0 = dz_col[0].astype(cp.float64)
        else:
            t0 = float(t_arr[0])
        de_sfc_src = dt * float(cp.sum(
            ust64 ** 3 / (KARMAN * 0.5 * t0)))
    else:
        de_sfc_src = 0.0
    return {"dKE": d_ke, "dE": d_e, "dHeat": d_heat,
            "residual": d_ke + d_e + d_heat - dke_sfc,
            "dKE_sfc": dke_sfc, "dE_sfc_src": de_sfc_src,
            "sfc_conv_resid": de_sfc_src + dke_sfc,
            "c_nu": c_nu, "f": f,
            "dKE_expl": dke_expl, "dKE_impl": dke_impl,
            # S3-6f partition-bound diagnostics ("f" above is the value
            # the step USED = min of the next three).
            "f_solved": f_solved, "f_cap": f_cap, "f_w": wsens.f_w,
            "zi": zi, "w_coverage": wsens.coverage,
            # S3-6g: the blended Prandtl number the step USED
            # (= prandtl_blend(f); the driver recomputes the same
            # value from the retained f for its scalar channels).
            "pr_t": pr_t,
            "kv": kv, "km_h": km}


def launch_n2(theta, *, dz: float | None = None, dz_col=None, out=None):
    """Brunt-Vaisala N^2 = (g/theta)*d(theta)/dz (authority
    ``brunt_vaisala_n2``): the stability-length input computed from the
    model's own theta profile on the SAME clamped (variable-``dz_col``)
    stencil every other SASE vertical operator uses.  ``dz_col`` follows
    the :func:`launch_strain` contract ((nz,) host column or per-column
    (nz, ny, nx) float32 device field).  Returns ``out`` (allocated when
    not given).
    """
    shape = theta.shape
    _check_field("theta", theta, shape)
    nz, ny, nx = shape
    if out is None:
        out = cp.empty(shape, dtype=DTYPE)
    else:
        _check_field("out", out, shape)
    zargs = _z_stencil(shape, dz=dz, dz_col=dz_col)
    zcm, zc0, zcp, dz_s, two_dz, h_lo, h_hi, z_mode = zargs
    grid, block = _tile(shape)
    _kern("sase_n2")(
        grid, block,
        (theta, out, zcm, zc0, zcp, DTYPE(0.0), DTYPE(0.0), dz_s, two_dz,
         h_lo, h_hi, z_mode, DTYPE(G_ACCEL),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def launch_moist_n2(theta, qv, qc, pressure, n2_dry, *, dz_col=None,
                    out=None):
    """SASE-M1 effective stability N^2_eff (S4-2 device mirror of the
    authority :func:`gpuwm.verify.sase_ref.moist_n2`; sase_ref module
    docstring, SASE-M1 section, has the DK82 Eq.-36 derivation): the
    saturated moist N^2_m with condensate loading where the registered
    binary switch fires (MOIST_STABILITY = "dk82-saturated-v1",
    MOIST_STABILITY_SWITCH = "binary-qc-or-rh100-liquid": qc > 0 OR
    qv >= qs,liq(T, p)), the ``n2_dry`` input BITWISE elsewhere -- the
    kernel copies the literal dry FP32 bits on the switch-false branch,
    so a fully unsaturated field returns ``n2_dry``'s exact bytes (the
    M1 unsaturated-identity contract; the driver feeds the result to
    the existing n2 consumers through ``launch_sase_step``'s
    ``n2_moist`` seam, which is then arithmetically inert).

    One FP64 in-thread column pass per (j, i) in the authority's exact
    op order (Exner T, Tetens liquid es on the model's own saturation
    constants, qs, the a/b moist-lapse factors, the loading term) with
    the vertical derivatives on the authority ``_ddz_var`` clamped
    variable-``dz_col`` stencil rebuilt in FP64 in-thread (sequential
    cumsum-half-layer z centers -- the ``_z_centers`` grouping); one
    FP32 rounding at the saturated store.  All physics constants are
    single-sourced from the authority registry as FP64 kernel
    arguments.  ``pressure`` is the FULL pressure [Pa] at layer
    centers; ``dz_col`` follows the :func:`_thickness_args` contract
    ((nz,) host column or per-column (nz, ny, nx) float32 device
    field) and is REQUIRED -- the authority signature has no uniform-dz
    mode.  ``n2_dry`` must be the ``launch_n2`` field of the SAME
    theta/dz_col (the driver's contract; launchers do no
    data-dependent validation).  Returns ``out`` (allocated when not
    given).
    """
    shape = theta.shape
    for name, arr in (("theta", theta), ("qv", qv), ("qc", qc),
                      ("pressure", pressure), ("n2_dry", n2_dry)):
        _check_field(name, arr, shape)
    nz, ny, nx = shape
    if dz_col is None:
        raise ValueError("dz_col is required: the authority moist_n2 "
                         "has no uniform-dz mode")
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds the moist-n2 column bound "
                         f"SASE_KMAX={_KMAX}")
    t_arr, t_dz, t_mode = _thickness_args(shape, dz_col=dz_col)
    if out is None:
        out = cp.empty(shape, dtype=DTYPE)
    else:
        _check_field("out", out, shape)
    grid, block = _flat(ny * nx)
    _kern("sase_moist_n2")(
        grid, block,
        (theta, qv, qc, pressure, n2_dry, out, t_arr, t_dz, t_mode,
         np.float64(RD_AIR / CP_AIR), np.float64(P0_REF),
         np.float64(SVP1), np.float64(SVP2), np.float64(SVP3),
         np.float64(SVPT0), np.float64(EP2_RV), np.float64(XLV),
         np.float64(RD_AIR), np.float64(CP_AIR), np.float64(G_ACCEL),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def _vent_plane(name: str, arr, shape) -> None:
    ny, nx = shape
    if (not isinstance(arr, cp.ndarray) or arr.shape != (ny, nx)
            or arr.dtype != DTYPE or not arr.flags.c_contiguous):
        raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                         f"array with shape {(ny, nx)}")


def launch_plume_vent_flux(theta, qv, qc, pressure, e_sgs, n2_moist,
                           n2_dry, rho1, *, f_blend: float, dz_col=None,
                           out=None, indices: bool = False):
    """SASE-M2 conditional venting limb (S4-5 device mirror of the
    authority :func:`gpuwm.verify.sase_ref.plume_vent_flux`; sase_ref
    module docstring, SASE-M2 section, has the complete formulation,
    every derivation and the registered constants).

    Returns the three face-registered flux profiles
    ``(F_theta, F_qv, F_qc)``, each a ``(nz + 1, ny, nx)`` float32 device
    field with ``F[0] = F[nz] = +0.0`` written as the literal (the
    interface contract; never a computed-then-zeroed value, never a
    negative zero).  Units are dynamic: ``F_theta`` [K kg m^-2 s^-1],
    ``F_qv``/``F_qc`` [kg m^-2 s^-1]; the S4-5 deposit is
    ``phi_k += (F[k] - F[k+1])*dt/(rho1*thick_k)``, applied through
    :func:`launch_vent_deposit_scale` + :func:`launch_vent_deposit`.

    ONE FP64 in-thread column sweep per (j, i) in the authority's exact
    op order (spec C12's ``sase_vertical_channel`` pattern; six KMAX
    double columns per thread), with a single FP32 rounding at the face
    stores.  The M1 substitution mask the authority takes as
    ``n2m_mask`` arrives here the way every other M-limb consumes it --
    as the BITWISE departure ``n2_moist != n2_dry`` of the two fields
    the driver already holds (the M1b ``bl89_rans_lengths`` seam idiom),
    so the seam is never re-derived and no new driver field appears.
    ``rho1`` is the ``(ny, nx)`` float32 lowest-level moist density
    plane (``physics.sase_surface_rho1`` -- the S3-11a convention, the
    same object the surface deposit rides); positivity is the driver's
    contract, not re-checked here.  ``f_blend`` is the step's USED
    partition fraction: the FP-exact two-product blend
    ``M_used = (1 - f)*M_base`` makes ``f = 1`` a bitwise +0.0 deposit
    (the LES-limit identity).  ``dz_col`` follows the
    :func:`_thickness_args` contract and is REQUIRED -- the authority
    signature has no uniform-dz mode.

    ``indices=True`` additionally returns a ``(7, ny, nx)`` int32 field
    of the seven diagnosed indices
    ``(k_base, k_top, k_r, k_lid, k_lfc, k_nb, kb)``, ``-1`` on a
    stood-down column.  This is the S4-5 parity gate's INDEX channel and
    is off on the driver path: index agreement with the authority is a
    pass/fail gate SEPARATE from the flux tolerance, because a one-cell
    move in any selected level is a median 35% flux change on real
    fields (design doc SASE-M2 amendment, "root / anchor separation") --
    a mirror that agrees to 2e-6 on a column where an index differs has
    not agreed at all.

    ``out`` (three same-shaped float32 buffers) may be supplied for
    reuse across steps.
    """
    shape = theta.shape
    for name, arr in (("theta", theta), ("qv", qv), ("qc", qc),
                      ("pressure", pressure), ("e_sgs", e_sgs),
                      ("n2_moist", n2_moist), ("n2_dry", n2_dry)):
        _check_field(name, arr, shape)
    nz, ny, nx = shape
    _vent_plane("rho1", rho1, (ny, nx))
    f = float(f_blend)
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"f_blend must be in [0, 1], got {f}")
    if dz_col is None:
        raise ValueError("dz_col is required: the authority "
                         "plume_vent_flux has no uniform-dz mode")
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds the vent column bound "
                         f"SASE_KMAX={_KMAX}")
    t_arr, t_dz, t_mode = _thickness_args(shape, dz_col=dz_col)
    fshape = (nz + 1, ny, nx)
    if out is None:
        out = [cp.empty(fshape, dtype=DTYPE) for _ in range(3)]
    else:
        out = list(out)
        if len(out) != 3:
            raise ValueError("out must hold three face-profile buffers")
        for i, arr in enumerate(out):
            _check_field(f"out[{i}]", arr, fshape)
    idx = (cp.empty((7, ny, nx), dtype=np.int32) if indices
           else cp.empty(1, dtype=np.int32))
    grid, block = _flat(ny * nx)
    _kern("sase_plume_vent_flux")(
        grid, block,
        (theta, qv, qc, pressure, e_sgs, n2_moist, n2_dry, rho1,
         out[0], out[1], out[2], idx, np.int32(1 if indices else 0),
         t_arr, t_dz, t_mode,
         np.float64(f), np.float64(RD_AIR / CP_AIR), np.float64(P0_REF),
         np.float64(SVP1), np.float64(SVP2), np.float64(SVP3),
         np.float64(SVPT0), np.float64(EP2_RV),
         np.float64(XLV / CP_AIR), np.float64(XLV), np.float64(CP_AIR),
         np.float64(RV_AIR / RD_AIR - 1.0), np.float64(G_ACCEL),
         np.float64(E_MIN), np.float64(VENT_MB_COEF),
         np.float64(VENT_ENT_COEF), np.float64(VENT_SIGW_SHARE),
         np.float64(VENT_DEPTH_CAP), np.int32(VENT_MIN_RUN_CELLS),
         np.int32(VENT_SAT_ADJUST_ITERS),
         np.int32(1 if VENT_MASK == "per-level-theta-es-v1" else 0),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    if indices:
        return out[0], out[1], out[2], idx
    return out[0], out[1], out[2]


def launch_vent_deposit_scale(f_theta, f_qv, f_qc, rho1, *, dt: float,
                              dz_col=None, out=None):
    """SASE-M2 deposit-seam CAP-FAMILY rescale factor (S4-5 device
    mirror of the authority
    :func:`gpuwm.verify.sase_ref.vent_deposit_rescale`; sase_ref module
    docstring, SASE-M2 RATE CAP).

    Returns the per-column ``(ny, nx)`` FLOAT64 factor

        s = min(1, VENT_THETA_STEP_CAP/|dtheta|max,
                   VENT_QT_STEP_CAP/|dqv|max,
                   VENT_QT_STEP_CAP/|dqc|max),

    each quotient over the column's own max of the UNSCALED deposit.
    The factor is per-COLUMN and UNIFORM across all three rows, which is
    what preserves the telescoping ledger, the exact zero end faces and
    the qv/qc partition -- a per-level clip destroys all three (measured
    at the authority: sum thick*dtheta = -3.74 against 0.0).  FP64 out
    deliberately: an FP32 scale plane would insert a second rounding
    between the registered cap and the deposited value for no memory
    that matters (one (ny, nx) plane).  DIVIDE GUARD: an inactive column
    has every |d|max exactly +0.0 and its quotient is SKIPPED, so ``s``
    stays exactly 1.0 without forming an infinity.
    """
    fshape = f_theta.shape
    for name, arr in (("f_qv", f_qv), ("f_qc", f_qc)):
        _check_field(name, arr, fshape)
    _check_field("f_theta", f_theta, fshape)
    nzp1, ny, nx = fshape
    nz = nzp1 - 1
    if nz < 1:
        raise ValueError("face profiles need at least 2 faces (nz >= 1)")
    _vent_plane("rho1", rho1, (ny, nx))
    if dz_col is None:
        raise ValueError("dz_col is required: the authority "
                         "vent_deposit_rescale has no uniform-dz mode")
    t_arr, t_dz, t_mode = _thickness_args((nz, ny, nx), dz_col=dz_col)
    if out is None:
        out = cp.empty((ny, nx), dtype=np.float64)
    elif (not isinstance(out, cp.ndarray) or out.shape != (ny, nx)
            or out.dtype != np.float64 or not out.flags.c_contiguous):
        raise ValueError("out must be a C-contiguous float64 CuPy array "
                         f"with shape {(ny, nx)}")
    grid, block = _flat(ny * nx)
    _kern("sase_vent_deposit_scale")(
        grid, block,
        (f_theta, f_qv, f_qc, rho1, out, t_arr, t_dz, t_mode,
         np.float64(dt), np.float64(VENT_THETA_STEP_CAP),
         np.float64(VENT_QT_STEP_CAP),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def launch_vent_deposit(phi, f_row, scale, rho1, *, dt: float,
                        dz_col=None):
    """SASE-M2 explicit flux-form RHS deposit of ONE scalar row, in
    place (S4-5 driver seam; spec C1-C3, binding):

        phi[k] += (Fs[k] - Fs[k+1])*dt/(rho1*thick_k),   Fs = scale*F.

    ``phi`` is the PRE-SOLVE state ``s* = s + dt*T_h`` of the driver's
    scalar loop and the deposit lands BEFORE
    :func:`launch_implicit_vertical_diffusion` -- the registered
    deposit-then-solve order generalizing SFC_SCALAR_FLUX =
    "explicit-deposit-v1" (the S3-11a seam).  NOTHING is inserted inside
    :func:`launch_sase_step` (the ledger theorem reads theta read-only)
    and NO Thomas row is touched: the solver's pinned max principle is
    exactly what non-local transport must be free to violate, so this
    deposit cannot ride the implicit sweep.

    ``scale`` is the :func:`launch_vent_deposit_scale` FP64 plane and
    multiplies the FLUXES (the authority's registered ``F_phi *= s``
    wording), not the formed deposit.  LEDGER: with F[0] = F[nz] = +0.0
    the interior faces telescope and ``sum_k thick_k*dphi_k = 0`` -- the
    S3-11a boundary-consistent scalar ledger extends with a ZERO
    net-column term, and the surface flux stays owned by the S3-11a
    deposit (the double-counting ban).  Returns ``phi``.
    """
    shape = phi.shape
    _check_field("phi", phi, shape)
    nz, ny, nx = shape
    _check_field("f_row", f_row, (nz + 1, ny, nx))
    _vent_plane("rho1", rho1, (ny, nx))
    if (not isinstance(scale, cp.ndarray) or scale.shape != (ny, nx)
            or scale.dtype != np.float64 or not scale.flags.c_contiguous):
        raise ValueError("scale must be a C-contiguous float64 CuPy "
                         f"array with shape {(ny, nx)}")
    if dz_col is None:
        raise ValueError("dz_col is required: the authority "
                         "vent_deposit_rescale has no uniform-dz mode")
    t_arr, t_dz, t_mode = _thickness_args(shape, dz_col=dz_col)
    grid, block = _tile(shape)
    _kern("sase_vent_deposit")(
        grid, block,
        (phi, f_row, scale, rho1, t_arr, t_dz, t_mode, np.float64(dt),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return phi


def launch_vent_flux_diag(f_row, scale, out, *, fac: float = 1.0):
    """SPLIT-FLUX DIAGNOSTIC, M2 vent channel (output-only).

    Writes ``out[k] = fac * scale * f_row[k]`` over the whole
    ``(nz + 1, ny, nx)`` face field -- the CAP-SCALED vent flux the
    deposit actually applied.  ``f_row`` alone is the UNSCALED profile
    that :func:`launch_plume_vent_flux` returns, and recording it would
    record a flux the model did not apply: the cap multiplies the
    FLUXES in-kernel inside :func:`launch_vent_deposit`.  The product is
    formed in the deposit's own FP64 op order (``s * (double)F``) with a
    single FP32 rounding at the store, so at ``fac = 1.0`` the recorded
    field is exactly the FP32 image of the FP64 value the deposit
    consumed.

    UNITS are the caller's: ``fac = 1.0`` keeps the row's own units
    (``kg m^-2 s^-1`` on a moisture row); ``fac = CP_AIR`` converts a
    theta row [K kg m^-2 s^-1] to a heat flux [W m^-2].  SIGN is the
    model's own vent convention, POSITIVE UPWARD (the deposit forms
    ``dphi_k = (F[k] - F[k+1])*dt/(rho1*thick_k)``).  Faces 0 and nz
    carry the interface contract's literal +0.0 through the multiply.

    READ-ONLY in every argument but ``out``: this launcher cannot
    perturb the state.  Returns ``out``.
    """
    fshape = f_row.shape
    _check_field("f_row", f_row, fshape)
    _check_field("out", out, fshape)
    if len(fshape) != 3:
        raise ValueError("f_row must be a (nz + 1, ny, nx) face field")
    nzp1, ny, nx = fshape
    if (not isinstance(scale, cp.ndarray) or scale.shape != (ny, nx)
            or scale.dtype != np.float64 or not scale.flags.c_contiguous):
        raise ValueError("scale must be a C-contiguous float64 CuPy "
                         f"array with shape {(ny, nx)}")
    grid, block = _tile(fshape)
    _kern("sase_vent_flux_diag")(
        grid, block,
        (f_row, scale, out, np.float64(fac),
         np.int32(nzp1), np.int32(ny), np.int32(nx)))
    return out


def launch_diff_flux_diag(phi, kv, rho1, out, *, kfac: float = 1.0,
                          fac: float = 1.0, dz: float | None = None,
                          dz_col=None):
    """SPLIT-FLUX DIAGNOSTIC, K_v implicit-diffusion channel
    (output-only).

    :func:`launch_implicit_vertical_diffusion` holds its face
    coefficients in per-thread registers and materializes NO face flux,
    so the flux is RECOVERED here rather than copied.  Backward Euler
    evaluates the operator at the NEW state, so the POST-SOLVE field
    determines it exactly:

        h[k]   = 0.5*(thick[k-1] + thick[k])
        K_f[k] = kfac * 0.5*(kv[k-1] + kv[k])
        F[k]   = -rho1 * K_f[k] * (phi[k] - phi[k-1]) / h[k]

    for the interior faces ``k = 1 .. nz-1``, with ``F[0] = F[nz]``
    the literal +0.0 (the vent seam's interface contract, shared so the
    two channels are summable face by face).  ``h`` and ``K_f`` are
    built in ``sase_thomas_scalar``'s verbatim op order, so this is the
    flux the solver used and not a nearby number.

    ``phi`` is the field AFTER the solve returns and BEFORE the driver
    converts it back to rate form; ``kv``/``dz_col`` and ``kfac``
    (= 1/Pr_t(f) on the driver's scalar rows) are the SAME objects the
    solve consumed.  ``rho1`` is the ``(ny, nx)`` lowest-level moist
    density plane -- the SAME plane the M2 deposit divides by, which is
    what makes the two channels summable and their convergences add up
    to the model's own increment.

    UNITS and SIGN follow :func:`launch_vent_flux_diag`: ``fac = 1.0``
    for a moisture row [kg m^-2 s^-1], ``fac = CP_AIR`` for a theta row
    [W m^-2], POSITIVE UPWARD -- which is why the form above carries the
    leading minus against the gradient.  FACE 0 IS +0.0 even on the
    theta/qv rows, whose true bottom-face flux is the S3-11b surface
    deposit (HFX/CP_AIR, QFX): that deposit is fused into the Thomas
    bottom rhs and touches no interior face, so the interior recovery is
    unaffected and the surface flux stays owned by HFX/QFX on disk (the
    double-counting ban).

    READ-ONLY in every argument but ``out``.  Returns ``out``.
    """
    shape = phi.shape
    _check_field("phi", phi, shape)
    _check_field("kv", kv, shape)
    nz, ny, nx = shape
    _check_field("out", out, (nz + 1, ny, nx))
    _vent_plane("rho1", rho1, (ny, nx))
    if nz < 2:
        raise ValueError("the recovery needs at least one interior face "
                         "(nz >= 2); the solver is an identity at nz == 1")
    t_arr, t_dz, t_mode = _thickness_args(shape, dz=dz, dz_col=dz_col)
    grid, block = _tile((nz + 1, ny, nx))
    _kern("sase_diff_flux_diag")(
        grid, block,
        (phi, kv, rho1, out, t_arr, t_dz, t_mode,
         np.float64(kfac), np.float64(fac),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def launch_scalar_mix(s, e=None, *, kh_coef: float | None = None,
                      kh_field=None, kh_fac: float = 1.0,
                      dx: float, dy: float, out=None, flux=None):
    """HORIZONTAL K_h down-gradient mixing tendency for one scalar:
    ``ddx(K_h ddx s) + ddy(K_h ddy s)``.  Two mutually exclusive K_h
    modes:

    * ``e``/``kh_coef`` (v0 seam): K_h = kh_coef*sqrt(max(e, E_MIN)),
      the pre-governor blend -- retained for its parity fixtures and
      the CPU shim fallback;
    * ``kh_field``/``kh_fac`` (S3-6e governed channel, authority
      ``scalar_hmix``): K_h = kh_fac*kh_field with ``kh_field`` the
      split step's exported governed diffusivity ``km_h`` and
      ``kh_fac`` = 1/Pr_t(f) the scalar convention (S3-6g blended
      Prandtl number at the step's used f) -- the driver's
      integration path.

    S3-6c retirement: the explicit VERTICAL leg of the v0 full-3D
    ``scalar_mix`` mirror is superseded by the implicit K_v/Pr_t(f)
    channel (:func:`launch_implicit_vertical_diffusion` on the split
    step's ``kv`` field, wired at driver level), so this launcher
    computes exactly the horizontal-explicit half of the split scalar
    channel.  Two passes mirroring the e-transport grouping exactly:
    cell-centered periodic horizontal fluxes, then the centered flux
    divergence.  ``flux`` (two same-shape float32 buffers) may be
    supplied for reuse across scalars; ``out`` receives the tendency.
    """
    shape = s.shape
    _check_field("s", s, shape)
    if (kh_field is None) == (e is None and kh_coef is None):
        raise ValueError("exactly one of (e, kh_coef) and kh_field "
                         "must be given")
    if kh_field is None:
        if e is None or kh_coef is None:
            raise ValueError("the coefficient mode needs both e and "
                             "kh_coef")
        _check_field("e", e, shape)
    else:
        _check_field("kh_field", kh_field, shape)
    nz, ny, nx = shape
    if flux is None:
        flux = [cp.empty(shape, dtype=DTYPE) for _ in range(2)]
    else:
        if len(flux) != 2:
            raise ValueError(f"flux must have 2 components, got {len(flux)}")
        for idx, buf in enumerate(flux):
            _check_field(f"flux[{idx}]", buf, shape)
    if out is None:
        out = cp.empty(shape, dtype=DTYPE)
    else:
        _check_field("out", out, shape)
    two_dx, two_dy = DTYPE(2.0 * float(dx)), DTYPE(2.0 * float(dy))
    dims = (np.int32(nz), np.int32(ny), np.int32(nx))
    grid, block = _tile(shape)
    if kh_field is None:
        _kern("sase_scalar_hflux")(
            grid, block,
            (s, e, *flux, two_dx, two_dy, DTYPE(kh_coef), DTYPE(E_MIN),
             *dims))
    else:
        _kern("sase_scalar_hflux_km")(
            grid, block,
            (s, kh_field, *flux, two_dx, two_dy, DTYPE(kh_fac), *dims))
    _kern("sase_hflux_div")(
        grid, block, (*flux, out, two_dx, two_dy, *dims))
    return out


#: u*-bin edges (m/s) of the smoke-gate surface-e diagnostic (stage-3
#: Task 7 adjudication): bins <0.2, 0.2-0.3, 0.3-0.4, 0.4-0.6, >0.6.
SURFACE_E_USTAR_BIN_EDGES = (0.2, 0.3, 0.4, 0.6)


def sase_surface_e_stats(e0, ust, dz1, *, boundary_width: int = 0) -> dict:
    """Lowest-level e statistics vs the analytic surface equilibrium.

    Smoke-gate diagnostic (stage-3 Task 7 adjudication; data collection
    for the queued Strang-splitting decision -- NO pass/fail semantics
    here).  Inputs are (ny, nx) surface planes: ``e0`` the lowest-level
    prognostic subgrid energy, ``ust`` the live SFCLAY friction
    velocity, ``dz1`` the lowest layer thickness; CuPy or NumPy (small
    2-D fields -- the arithmetic runs host-side FP64).

    Per column the analytic equilibrium of the surface-source/decay
    balance is ``e_eq = (S*l_B/C_E)^(2/3)`` with the neutral shear
    source ``S = u*^3/(kappa*0.5*dz1)`` evaluated at the lowest half
    level ``z1 = dz1/2`` (exactly :func:`gpuwm.core.physics.
    sase_surface_e_source`'s shear term -- the buoyancy term is
    deliberately excluded: the adjudication asks for the ratio computed
    from live u* alone) and ``l_B`` the authority Blackadar length at
    z1.  The S3-6d equilibrium fixture characterized the discrete
    driver sequence's ratio e/e_eq at 0.427 for u* = 0.25 (decreasing
    with u*: 0.501 at 0.2 .. 0.205 at 0.5 -- the operator-splitting
    bias) so ratios well below unity are EXPECTED; this diagnostic
    measures that bias in the live run, binned by u* so the bias's
    u*-dependence is visible.

    ``boundary_width`` excludes the outer rows from the ratio/bin
    statistics (the specified-domain e floor holds e = E_MIN there, a
    policy value that would poison the equilibrium comparison); the
    plain e0 min/mean/max is reported for BOTH the full plane and the
    interior.  Returns a JSON-ready dict of plain floats/ints.
    """
    e0 = cp.asnumpy(e0) if isinstance(e0, cp.ndarray) else np.asarray(e0)
    ust = cp.asnumpy(ust) if isinstance(ust, cp.ndarray) else np.asarray(ust)
    dz1 = cp.asnumpy(dz1) if isinstance(dz1, cp.ndarray) else np.asarray(dz1)
    e0 = e0.astype(np.float64)
    ust = ust.astype(np.float64)
    dz1 = dz1.astype(np.float64)
    if not (e0.ndim == 2 and e0.shape == ust.shape == dz1.shape):
        raise ValueError(
            f"e0/ust/dz1 must share one (ny, nx) shape, got "
            f"{e0.shape}/{ust.shape}/{dz1.shape}")
    w = int(boundary_width)
    if w < 0 or (w and (2 * w >= e0.shape[0] or 2 * w >= e0.shape[1])):
        raise ValueError(
            f"boundary_width {w} leaves no interior of {e0.shape}")
    sl = (slice(w, -w) if w else slice(None),) * 2
    e_i, ust_i, dz1_i = e0[sl], ust[sl], dz1[sl]
    z1 = 0.5 * dz1_i
    lb = _blackadar_length(z1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_shear = ust_i ** 3 / (KARMAN * z1)
        e_eq = (s_shear * lb / C_E) ** (2.0 / 3.0)
        ratio = np.where(e_eq > 0.0, e_i / e_eq, np.nan)

    def _plane(values) -> dict:
        return {"min": float(values.min()), "mean": float(values.mean()),
                "max": float(values.max())}

    edges = (-np.inf,) + SURFACE_E_USTAR_BIN_EDGES + (np.inf,)
    labels = ("<0.2", "0.2-0.3", "0.3-0.4", "0.4-0.6", ">0.6")
    bins = []
    for label, lo, hi in zip(labels, edges[:-1], edges[1:]):
        mask = (ust_i >= lo) & (ust_i < hi) & np.isfinite(ratio)
        entry = {"ustar": label, "count": int(mask.sum())}
        if entry["count"]:
            entry.update(
                ust_mean=float(ust_i[mask].mean()),
                e_mean=float(e_i[mask].mean()),
                ratio_min=float(ratio[mask].min()),
                ratio_mean=float(ratio[mask].mean()),
                ratio_max=float(ratio[mask].max()))
        bins.append(entry)
    return {"e0": _plane(e0), "e0_interior": _plane(e_i),
            "ust_interior": _plane(ust_i),
            "boundary_width": w, "bins": bins}


def sase_e_cap_stats(e, u, v, dz_col, ust, *, boundary_width: int = 0,
                     percentile: float = 99.9) -> dict:
    """Live-field equilibrium cap for the re-derived smoke gate 2a.

    DERIVATION (S3-6e adjudication; replaces the guessed e <= 50;
    S3-6f doc fix: this is the f = 0 -- RANS-limit -- fixed point, NOT
    "the closure's own" at live f).  The VERTICAL channel's local shear
    balance -- production K_v*S^2 against dissipation C_E*e^{3/2}/l_v
    with K_v = C_KV*l_v*sqrt(e) -- has the fixed point

        e_eq = (C_KV/C_E) * l_v^2 * S^2.

    At live f > 0 the step's dissipation actually rides the BLENDED
    l_d -- the then-linear f*delta + (1-f)*l_B >= l_v when this was
    derived; GEOMETRIC delta**f * l_eps_rans**(1-f) since S3-9, which
    by weighted AM-GM sits at or below the linear value and can only
    LOWER the closure's fixed point toward this cap -- so the true
    fixed point can sit ABOVE this cap by up to (l_d/l_v)^(2/3),
    unboundedly so at f -> 1 with mesoscale delta (the S3-6e review's
    1280x dissipation throttle).  The S3-6f partition cap drives f -> 0 exactly in that
    regime, which is what makes this formula the cap the gate may hold
    the run to: it is the fixed point of the closure the concession
    REQUIRES to be live at mesoscale Delta.

    Evaluating it with the NEUTRAL Blackadar length l_B >= l_v (i.e.
    dropping the stability limit) can only RAISE the estimate, so

        e_cap = (C_KV/C_E) * l_B(z)^2 * S_v^2

    is an upper envelope of the locally sustainable e, with S_v^2 the
    live resolved vertical shear ((du/dz)^2 + (dv/dz)^2, face
    differences mapped to cells by the max of the adjacent faces --
    again the envelope choice).  The surface row additionally takes the
    named-source balance e_eq = (S_sfc*l_B(z1)/C_E)^(2/3), S_sfc =
    u*^3/(kappa*z1) (``sase_surface_e_stats``' e_eq; buoyancy excluded
    per the G7 adjudication), whichever is larger.  The gate then reads

        domain-max e <= C_GATE * percentile_99.9(e_cap)   [interior]

    S3-6k GATE WATCH (registered, deliberately NOT implemented here).
    The cap above is the NEUTRAL fixed point -- it evaluates l_B with the
    stability limit dropped and reads C_E, so it is unchanged by
    ``RunConfig.sase_stable_dissipation``.  The STABLE fixed point it
    envelopes is not: cutting the stable-limb dissipation coefficient to
    C_ES raises it by up to (C_E/C_ES) = 4.8947x wherever rho = 1.  If a
    switch-on run trips gate 2a, the correct fix is to make THIS formula
    read the same blended coefficient the step used -- never to raise
    C_GATE, which is a registered constant.

    -- the percentile (not pointwise) comparison deliberately grants
    transport-fed cells the headroom of the domain's strong-forcing
    population, and C_GATE = 3 (registry) covers convective additions
    and discrete overshoot.  Inputs are (nz, ny, nx) A-grid winds, the
    per-column layer thicknesses, the prognostic e, and the (ny, nx)
    SFCLAY u*; CuPy or NumPy (host FP64 arithmetic).
    """
    to_np = (lambda a: cp.asnumpy(a) if isinstance(a, cp.ndarray)
             else np.asarray(a))
    e = to_np(e).astype(np.float64)
    u = to_np(u).astype(np.float64)
    v = to_np(v).astype(np.float64)
    t = to_np(dz_col).astype(np.float64)
    ust = to_np(ust).astype(np.float64)
    if not (e.ndim == 3 and e.shape == u.shape == v.shape == t.shape
            and ust.shape == e.shape[1:]):
        raise ValueError(
            f"shape mismatch: e{e.shape} u{u.shape} v{v.shape} "
            f"dz_col{t.shape} ust{ust.shape}")
    nz = e.shape[0]
    w_bd = int(boundary_width)
    if w_bd < 0 or (w_bd and (2 * w_bd >= e.shape[1]
                              or 2 * w_bd >= e.shape[2])):
        raise ValueError(
            f"boundary_width {w_bd} leaves no interior of {e.shape}")
    z = np.cumsum(t, axis=0) - 0.5 * t         # layer centers
    lb = _blackadar_length(z)
    cap = np.zeros_like(e)
    if nz > 1:
        h = z[1:] - z[:-1]
        s2f = (((u[1:] - u[:-1]) / h) ** 2
               + ((v[1:] - v[:-1]) / h) ** 2)  # face shear^2
        s2 = np.zeros_like(e)
        s2[:-1] = s2f
        s2[1:] = np.maximum(s2[1:], s2f)       # cell = max adjacent face
        cap = (C_KV / C_E) * lb ** 2 * s2
    z1 = z[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_sfc = ust ** 3 / (KARMAN * z1)
        cap0 = (s_sfc * _blackadar_length(z1) / C_E) ** (2.0 / 3.0)
    cap[0] = np.maximum(cap[0], np.where(np.isfinite(cap0), cap0, 0.0))
    sl = (slice(None),) + (slice(w_bd, -w_bd) if w_bd
                           else slice(None),) * 2
    e_i, cap_i = e[sl], cap[sl]
    cap_p = float(np.percentile(cap_i, percentile))
    e_max = float(e_i.max())
    return {"e_max": e_max, "e_mean": float(e_i.mean()),
            "cap_percentile": percentile, "cap_p": cap_p,
            "cap_max": float(cap_i.max()),
            "ratio": e_max / max(cap_p, 1.0e-30),
            "boundary_width": w_bd}


# ---------------------------------------------------------------------------
# S3-9e: on-device gate-2a statistics (same instrument, cheaper route).
#
# Forensics on a three-domain production run (recorded with the
# campaign evidence) measured the legacy host path above pulling
# ~393 MB pageable
# D2H per root period and starving the GPU 1.0-1.7 s while single-thread
# host numpy reduced ~24.5M FP64 cells -- ~6.3% of run wall time.  The
# device path below evaluates the IDENTICAL FP64 expression graph with
# CuPy kernels and transfers only the four reduced scalars (one 32-byte
# D2H); the ratio is derived host-side from those scalars exactly as the
# legacy path derives it.  The legacy function above is the RETAINED
# REFERENCE ("never weaken a guard": same stats, same cadence) and the
# dual-path probe compares the two on the same state.
# ---------------------------------------------------------------------------

#: S3-9e dual-path comparison rules: per-stat relative tolerances of the
#: device path against the legacy host reference, DERIVED (not tuned):
#:
#: * ``e_max`` -- 0.0 (bitwise).  Both paths take the max of the same
#:   exact FP32->FP64-converted e values; a max reduction returns one of
#:   its inputs regardless of traversal order, so the paths must agree
#:   bit for bit (inputs are NaN-free under the health gate).
#: * ``e_mean`` -- 1e-12.  The summand set is identical (exact
#:   FP32->FP64 converts); only the accumulation ORDER differs (numpy
#:   pairwise vs CuPy per-thread-sequential + block-tree).  For
#:   nonnegative summands (e >= E_MIN >= 0, the instrument's domain) the
#:   classical bound |sum_A - sum_B| <= (h_A + h_B)*eps*sum(x) holds
#:   with h the summation depth; numpy pairwise has h <= 2*log2(N) + 128
#:   and CuPy's two-stage reduction h <= chunk + tree, both well under
#:   2048 at the largest domain (N = 12.3e6 cells), so
#:   (h_A + h_B)*eps <= 4096 * 2^-52 = 9.1e-13; 1e-12 rounds up.
#: * ``cap_p`` / ``cap_max`` -- 1e-14.  The cap field is elementwise
#:   bitwise identical between the paths EXCEPT through the surface
#:   row's two transcendental pow evaluations (``ust ** 3`` and
#:   ``** (2/3)``), where CUDA pow and host libm each sit within ~2 ulp
#:   of correctly rounded: per-element relative divergence <= ~8 ulp =
#:   1.8e-15.  Order statistics (max, the percentile picks) of
#:   elementwise-(1 +/- d)-bounded arrays differ by <= d relative, and
#:   the percentile interpolation adds <= ~4 ulp (numpy vs CuPy lerp
#:   grouping): <= ~12 ulp total; 1e-14 (= 45 ulp) adds margin.
#: * ``ratio`` -- 1e-13.  ``e_max / max(cap_p, 1e-30)`` is one host
#:   division of the transferred scalars on each path: relative error
#:   <= rel(e_max) + rel(cap_p) + 1 ulp <= ~1.1e-14; 1e-13 adds margin.
#: * ``cap_percentile`` / ``boundary_width`` -- exact (echoed inputs).
E_CAP_STATS_COMPARE_RTOL = {
    "e_max": 0.0,
    "e_mean": 1.0e-12,
    "cap_p": 1.0e-14,
    "cap_max": 1.0e-14,
    "ratio": 1.0e-13,
}


def sase_e_cap_stats_device(e, u, v, dz_col, ust, *,
                            boundary_width: int = 0,
                            percentile: float = 99.9, xp=None) -> dict:
    """:func:`sase_e_cap_stats` computed with device reductions (S3-9e).

    Same numbers, cheaper route: FP64 accumulators throughout, one
    32-byte D2H of the four reduced scalars instead of ~393 MB of field
    pulls.  The expression graph restructures the legacy host path ONLY
    where NumPy's own fast paths make the restructure bitwise-neutral:

    * ``x ** 2`` is written ``x * x`` (NumPy's ``__pow__`` exponent-2
      fast path IS ``square`` = elementwise multiply, while CuPy's
      ``power`` would route through device ``pow``);
    * ``np.cumsum`` is written as the sequential per-column recurrence
      it is defined to be (CuPy's ``cumsum`` is a parallel scan with a
      different rounding order);
    * ``_blackadar_length`` is inlined verbatim (its ``np.asarray``
      coercion cannot accept a device array), ``z0 = 0.0`` preserved.

    Under ``xp=numpy`` this function is therefore BITWISE identical to
    the legacy reference on every stat -- the CPU spine test pins that.
    Under ``xp=cupy`` (default) every elementwise FP64 op is the same
    IEEE round-to-nearest operation in the same order (one CuPy kernel
    per op, so no cross-op FMA contraction), leaving only the
    reduction-order and transcendental divergences bounded by
    :data:`E_CAP_STATS_COMPARE_RTOL`.  Returns the legacy dict plus
    ``stats_path`` ("device", or "device-graph-host" under numpy).
    """
    if xp is None:
        xp = cp
    e = xp.asarray(e).astype(np.float64)
    u = xp.asarray(u).astype(np.float64)
    v = xp.asarray(v).astype(np.float64)
    t = xp.asarray(dz_col).astype(np.float64)
    ust = xp.asarray(ust).astype(np.float64)
    if not (e.ndim == 3 and e.shape == u.shape == v.shape == t.shape
            and ust.shape == e.shape[1:]):
        raise ValueError(
            f"shape mismatch: e{e.shape} u{u.shape} v{v.shape} "
            f"dz_col{t.shape} ust{ust.shape}")
    nz = e.shape[0]
    w_bd = int(boundary_width)
    if w_bd < 0 or (w_bd and (2 * w_bd >= e.shape[1]
                              or 2 * w_bd >= e.shape[2])):
        raise ValueError(
            f"boundary_width {w_bd} leaves no interior of {e.shape}")
    # np.cumsum's defining sequential recurrence, written out.
    csum = xp.empty_like(t)
    csum[0] = t[0]
    for k in range(1, nz):
        csum[k] = csum[k - 1] + t[k]
    z = csum - 0.5 * t                         # layer centers
    del csum
    kz = KARMAN * (z + 0.0)                    # _blackadar_length, inline
    lb = kz / (1.0 + kz / BLACKADAR_LAMBDA)
    del kz
    cap = xp.zeros_like(e)
    if nz > 1:
        h = z[1:] - z[:-1]
        du = (u[1:] - u[:-1]) / h
        dv = (v[1:] - v[:-1]) / h
        del h
        s2f = du * du + dv * dv                # face shear^2 (** 2 == square)
        del du, dv
        s2 = xp.zeros_like(e)
        s2[:-1] = s2f
        s2[1:] = xp.maximum(s2[1:], s2f)       # cell = max adjacent face
        del s2f
        cap = (C_KV / C_E) * (lb * lb) * s2    # lb ** 2 == square
        del s2
    del u, v
    z1 = z[0]
    kz1 = KARMAN * (z1 + 0.0)                  # _blackadar_length(z1), inline
    lb1 = kz1 / (1.0 + kz1 / BLACKADAR_LAMBDA)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_sfc = ust ** 3 / (KARMAN * z1)
        cap0 = (s_sfc * lb1 / C_E) ** (2.0 / 3.0)
    cap[0] = xp.maximum(cap[0], xp.where(xp.isfinite(cap0), cap0, 0.0))
    del z, lb, z1, kz1, lb1, s_sfc, cap0, ust, t
    sl = (slice(None),) + (slice(w_bd, -w_bd) if w_bd
                           else slice(None),) * 2
    e_i, cap_i = e[sl], cap[sl]
    scalars = xp.stack((e_i.max(), e_i.mean(),
                        xp.percentile(cap_i, percentile), cap_i.max()))
    if xp is not np:
        scalars = cp.asnumpy(scalars)          # the ONLY D2H: 32 bytes
    e_max, e_mean, cap_p, cap_max = (float(s) for s in scalars)
    return {"e_max": e_max, "e_mean": e_mean,
            "cap_percentile": percentile, "cap_p": cap_p,
            "cap_max": cap_max,
            "ratio": e_max / max(cap_p, 1.0e-30),
            "boundary_width": w_bd,
            "stats_path": "device" if xp is not np else "device-graph-host"}


def compare_e_cap_stats(host_stats: dict, device_stats: dict) -> dict:
    """Per-stat equivalence record of one dual gate-2a sample (S3-9e §2).

    Applies :data:`E_CAP_STATS_COMPARE_RTOL` statwise (``e_max`` bitwise;
    the rest under the derived relative bounds) plus exact equality on
    the echoed ``cap_percentile``/``boundary_width``.  Returns a
    JSON-ready record; ``record["equivalent"]`` is the roll-up.
    """
    record: dict = {"rtol": dict(E_CAP_STATS_COMPARE_RTOL), "stats": {}}
    ok = True
    for key in ("cap_percentile", "boundary_width"):
        same = host_stats[key] == device_stats[key]
        record["stats"][key] = {"host": host_stats[key],
                                "device": device_stats[key],
                                "pass": bool(same)}
        ok = ok and same
    for key, rtol in E_CAP_STATS_COMPARE_RTOL.items():
        a, b = float(host_stats[key]), float(device_stats[key])
        if a == b:
            passed, rel = True, 0.0
        elif math.isfinite(a) and math.isfinite(b):
            rel = abs(a - b) / max(abs(a), abs(b))
            passed = rel <= rtol
        else:
            passed, rel = False, math.inf
        record["stats"][key] = {"host": a, "device": b, "rel_err": rel,
                                "rtol": rtol, "pass": bool(passed)}
        ok = ok and passed
    record["equivalent"] = bool(ok)
    return record


def sase_e_cap_stats_dual(e, u, v, dz_col, ust, *, boundary_width: int = 0,
                          percentile: float = 99.9, xp=None,
                          raise_on_mismatch: bool = True) -> dict:
    """Receipt-equivalence probe (S3-9e §2): BOTH paths, same state.

    Runs the legacy host-numpy reference AND the device path on the same
    inputs, compares every stat under :func:`compare_e_cap_stats`, and
    returns the DEVICE dict (the production numbers) with the complete
    comparison attached under ``dual_probe``.  Fail-loud by default: an
    out-of-tolerance stat raises so a validation run cannot silently
    record inequivalent receipts.  Enabled by ``GPUWM_GATE2A_DUAL=1``.
    """
    device_stats = sase_e_cap_stats_device(
        e, u, v, dz_col, ust, boundary_width=boundary_width,
        percentile=percentile, xp=xp)
    host_stats = sase_e_cap_stats(
        e, u, v, dz_col, ust, boundary_width=boundary_width,
        percentile=percentile)
    probe = compare_e_cap_stats(host_stats, device_stats)
    out = dict(device_stats)
    out["dual_probe"] = probe
    if raise_on_mismatch and not probe["equivalent"]:
        failing = {key: rec for key, rec in probe["stats"].items()
                   if not rec["pass"]}
        raise ValueError(
            "gate-2a dual-path probe: device stats are NOT equivalent to "
            f"the legacy host reference: {failing}")
    return out
