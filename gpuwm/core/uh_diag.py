"""UP_HELI_MAX -- WRF's updraft-helicity running-max diagnostic.

Transcribed from WRF v4.6.1 ``SUBROUTINE cal_helicity``
(dyn_em/module_diffusion_em.F:7132-7579, pinned tree
d66e442fccc04111067e29274c9f9eaccc3cef28), with the metric prep it consumes
(``compute_diff_metrics``, :6882-7130) evaluated pointwise from the same
expressions.  Driver wiring in WRF:

- called EVERY model step when ``nwp_diagnostics == 1``
  (module_first_rk_step_part2.F:533-556; the namelist gate is
  Registry.EM_COMMON:2210, &time_control, default 0);
- integrates w times zeta over the fixed 2000-5000 m AGL band, only over
  layers ENTIRELY inside the band, aborting a column whose 8-point-average
  w is non-positive anywhere inside the band (:7474-7511);
- 9-point-smooths the column result and folds it into the running max
  ``UP_HELI_MAX`` (:7515-7533; Registry IO "rh02": restart + history,
  units "m2 s-2");
- the accumulator is zeroed at the start of each history interval
  (phys/module_diag_nwp.F:246-269).

This is a DIAGNOSTIC: nothing here writes any model-evolution array, and
tests pin that nothing in the package reads the accumulator back into the
trajectory.

Ratified divergences from WRF, recorded here and in the STEP17 handoff:

1. Sampling instant.  WRF samples the START-of-step state (cal_helicity
   runs inside the first RK step) and resets on the FIRST step AFTER a
   history write -- so each frame's max covers the open interval
   (T_prev, T_frame) EXCLUDING both endpoint states.  gpuwm samples the
   completed step's state in the dycore epilogue and resets immediately
   after the frame is written: the window is (T_prev, T_frame] INCLUDING
   the emitted frame's own state.  The interior samples are identical; the
   difference is one endpoint sample per frame, and gpuwm's frame max is
   always >= the instantaneous UH of the frame it rides in.
2. Substep cadence.  Under a compatibility integrator running internal
   substeps (dt = clock_dt/n), the max is folded EVERY internal step --
   the same self-consistent reading PROVENANCE.md D1 ratified for
   h_diabatic capture; the native-dt path restores WRF cadence unchanged.
3. Top-layer stand-in.  At the top mass layer WRF reads one-past-the-end
   wavg/rvort values its loops never wrote (uninitialised automatic
   arrays), guarded only by the zu <= 5000 m test; gpuwm substitutes 0.0
   (which zeroes the column through the w > 0 test).  Reachable only with
   a model top below 5000 m AGL.
4. Halo edge write.  WRF's boundary copy writes up_heli_max(ide, j) /
   up_heli_max(i, jde) into staggered halo memory that never reaches any
   output; gpuwm's (ny, nx) field has no such slot.  Consequence (shared
   with WRF's own emitted field): the east column and north row of
   UP_HELI_MAX stay exactly zero, and the west column / south row mirror
   their inward neighbours.

The oracle for the transcription is tools/uh_wrf461_oracle (fixtures under
gpuwm/data/uh/oracle); tests/test_uh_wrf461_parity.py holds the NumPy
mirror and the CUDA kernel at the recorded ULP baseline against it.
"""
from __future__ import annotations

import numpy as np

from gpuwm.core.constants import G

DTYPE = np.float32

#: WRF cal_helicity's fixed integration bounds, m AGL (:7472,:7499).
UH_LAYER_BOTTOM_M = np.float32(2000.0)
UH_LAYER_TOP_M = np.float32(5000.0)

#: The serialized accumulator slot (restart-carried, wrfout-emitted).
UP_HELI_MAX_SLOT = "up_heli_max"
#: Per-step work planes, rebuilt every launch (never serialized).
UH_WORK_COLUMN_SLOT = "uh_diag_col"
UH_WORK_USE_SLOT = "uh_diag_use"

#: CONSUMER-OWNED tracking windows (Drew's ruling, 2026-08-07).
#:
#: Same operator as UP_HELI_MAX and the same fold below -- what differs is
#: WHO RESETS.  UP_HELI_MAX is zeroed by the history writer, so its window
#: is the output cadence; a model consumer reading it therefore has its
#: decisions moved by an output knob, which is how ``--history-interval``
#: came to steer a storm-following nest.  Each of these is zeroed by the
#: consumer that reads it, immediately after it reads, so its window is
#: "max since I last looked" and nothing else.
#:
#: TWO of them, not one, because the two consumers evaluate on genuinely
#: different granularities: relocation at cadence boundaries inside one
#: ``execute_experiment`` call, spawn at LEG boundaries through schedule
#: surgery (gpuwm/core/spawn_runner.py's opening note).  A shared window
#: would let whichever consumer read first blind the other at any
#: boundary they happen to share.
#:
#: The names deliberately do NOT contain "up_heli_max": these are not the
#: WRF diagnostic, they are never emitted, and the governance roster in
#: tests/test_uh_lifecycle.py pins that slot's readers by substring.
UH_FOLLOW_WINDOW_SLOT = "uh_follow_window"
UH_SPAWN_WINDOW_SLOT = "uh_spawn_window"
#: Folded every step beside UP_HELI_MAX; reset only by their consumers.
TRACKER_WINDOW_SLOTS = (UH_FOLLOW_WINDOW_SLOT, UH_SPAWN_WINDOW_SLOT)

_THREADS = 256

_F0 = np.float32(0.0)
_F025 = np.float32(0.25)
_F05 = np.float32(0.5)
_F0125 = np.float32(0.125)
_F00625 = np.float32(0.0625)
_F1 = np.float32(1.0)
_G = DTYPE(G)


# ---------------------------------------------------------------------------
# NumPy mirror (vectorized over the corner grid; identical per-element
# operation order to the Fortran scalar chain)
# ---------------------------------------------------------------------------

def _phb3(phb, ny, nx):
    if phb.ndim == 1:
        return np.broadcast_to(
            phb[:, None, None], (phb.shape[0], ny, nx))
    return phb


def cal_helicity_uh_columns_np(u, v, w, ph, phb, msfu, msfv, ht,
                               dn, dnw, fnm, fnp, cf1, cf2, cf3,
                               rdx, rdy):
    """WRF cal_helicity through the column integration (:7206-7511).

    Inputs are gpuwm-shaped float32 arrays -- u (nz,ny,nx+1), v (nz,ny+1,nx),
    w/ph (nz+1,ny,nx), phb 3-D or a flat (nz+1,) column, msfu (ny,nx+1),
    msfv (ny+1,nx), ht (ny,nx); dn/dnw/fnm/fnp are the (nz,) coordinate
    columns with the WRF k=1 endpoint zero at index 0.

    Returns ``(uh, use_column)`` on the mass-indexed (ny, nx) grid; the
    corner values live at [1:, 1:] exactly as WRF stores corner-point uh
    into mass-indexed arrays, and the never-written row/column 0 is 0.
    """
    nzf, ny, nx = w.shape
    nz = nzf - 1
    ph = np.asarray(ph)
    phb = _phb3(np.asarray(phb), ny, nx)
    cf1 = np.float32(cf1)
    cf2 = np.float32(cf2)
    cf3 = np.float32(cf3)
    rdx = np.float32(rdx)
    rdy = np.float32(rdy)
    # cft1/cft2 (:7211-7215); dnw/dn 1-based index kte-1 = nz -> [nz-1].
    cft2 = -(_F05 * np.float32(dnw[nz - 1]) / np.float32(dn[nz - 1]))
    cft1 = np.float32(1.0) - cft2

    # z at w levels and pointwise metric pieces (compute_diff_metrics).
    z_w = (ph + phb) / _G                              # (nz+1, ny, nx)
    rdzw = _F1 / (z_w[1:] - z_w[:-1])                  # (nz, ny, nx)
    # zx at u points (w levels), two rounded passes; interior only -- the
    # corner stencils below never touch the zeroed domain faces.
    zx = (rdx * (phb[:, :, 1:] - phb[:, :, :-1])) / _G \
        + (rdx * (ph[:, :, 1:] - ph[:, :, :-1])) / _G  # (nz+1, ny, nx-1)
    zy = (rdy * (phb[:, 1:, :] - phb[:, :-1, :])) / _G \
        + (rdy * (ph[:, 1:, :] - ph[:, :-1, :])) / _G  # (nz+1, ny-1, nx)

    # hat fields (:7260, :7344).
    hat_v = v / msfv[None, :, :]                       # (nz, ny+1, nx)
    hat_u = u / msfu[None, :, :]                       # (nz, ny, nx+1)

    # Corner grid: cy in [1, ny-1], cx in [1, nx-1].
    cys = slice(1, ny)
    cys_m1 = slice(0, ny - 1)
    cxs = slice(1, nx)
    cxs_m1 = slice(0, nx - 1)

    # mm (:7251).
    mm = (_F025 * (msfu[cys_m1, cxs] + msfu[cys, cxs])) \
        * (msfv[cys, cxs_m1] + msfv[cys, cxs])

    def hatavg_v(kw):
        # v-hat onto (corner, w level); pair order (i-1, i).
        if kw == 0:
            acc = cf1 * hat_v[0, cys, cxs_m1]
            acc = acc + cf2 * hat_v[1, cys, cxs_m1]
            acc = acc + cf3 * hat_v[2, cys, cxs_m1]
            acc = acc + cf1 * hat_v[0, cys, cxs]
            acc = acc + cf2 * hat_v[1, cys, cxs]
            acc = acc + cf3 * hat_v[2, cys, cxs]
            return _F05 * acc
        if kw == nz:
            t1 = cft1 * (hat_v[nz - 1, cys, cxs] + hat_v[nz - 1, cys, cxs_m1])
            t2 = cft2 * (hat_v[nz - 2, cys, cxs] + hat_v[nz - 2, cys, cxs_m1])
            return _F05 * (t1 + t2)
        t1 = np.float32(fnm[kw]) * (hat_v[kw, cys, cxs_m1]
                                    + hat_v[kw, cys, cxs])
        t2 = np.float32(fnp[kw]) * (hat_v[kw - 1, cys, cxs_m1]
                                    + hat_v[kw - 1, cys, cxs])
        return _F05 * (t1 + t2)

    def hatavg_u(kw):
        # u-hat onto (corner, w level); pair order (j-1, j).
        if kw == 0:
            acc = cf1 * hat_u[0, cys_m1, cxs]
            acc = acc + cf2 * hat_u[1, cys_m1, cxs]
            acc = acc + cf3 * hat_u[2, cys_m1, cxs]
            acc = acc + cf1 * hat_u[0, cys, cxs]
            acc = acc + cf2 * hat_u[1, cys, cxs]
            acc = acc + cf3 * hat_u[2, cys, cxs]
            return _F05 * acc
        if kw == nz:
            t1 = cft1 * (hat_u[nz - 1, cys_m1, cxs] + hat_u[nz - 1, cys, cxs])
            t2 = cft2 * (hat_u[nz - 2, cys_m1, cxs] + hat_u[nz - 2, cys, cxs])
            return _F05 * (t1 + t2)
        t1 = np.float32(fnm[kw]) * (hat_u[kw, cys_m1, cxs]
                                    + hat_u[kw, cys, cxs])
        t2 = np.float32(fnp[kw]) * (hat_u[kw - 1, cys_m1, cxs]
                                    + hat_u[kw - 1, cys, cxs])
        return _F05 * (t1 + t2)

    def rvort(km):
        # dv/dx: tmpzx over (j-1, j) x (k, k+1); rdzw order NE,SE,SW,NW
        # (:7305-7310).  zx u-point index cx maps to zx[:, :, cx-1].
        tmpzx = _F025 * (((zx[km, cys_m1, cxs_m1] + zx[km, cys, cxs_m1])
                          + zx[km + 1, cys_m1, cxs_m1])
                         + zx[km + 1, cys, cxs_m1])
        rdzw4 = ((rdzw[km, cys, cxs] + rdzw[km, cys_m1, cxs])
                 + rdzw[km, cys_m1, cxs_m1]) + rdzw[km, cys, cxs_m1]
        tmp1v = ((hatavg_v(km + 1) - hatavg_v(km)) * _F025 * tmpzx) * rdzw4
        rv = mm * (rdx * (hat_v[km, cys, cxs] - hat_v[km, cys, cxs_m1])
                   - tmp1v)
        # du/dy: tmpzy over (i-1, i) x (k, k+1); rdzw order NE,NW,SW,SE
        # (:7383-7388).  zy v-point index cy maps to zy[:, cy-1, :].
        tmpzy = _F025 * (((zy[km, cys_m1, cxs_m1] + zy[km, cys_m1, cxs])
                          + zy[km + 1, cys_m1, cxs_m1])
                         + zy[km + 1, cys_m1, cxs])
        rdzw4 = ((rdzw[km, cys, cxs] + rdzw[km, cys, cxs_m1])
                 + rdzw[km, cys_m1, cxs_m1]) + rdzw[km, cys_m1, cxs]
        tmp1u = ((hatavg_u(km + 1) - hatavg_u(km)) * _F025 * tmpzy) * rdzw4
        return rv - mm * (rdy * (hat_u[km, cys, cxs] - hat_u[km, cys_m1, cxs])
                          - tmp1u)

    def wavg(km):
        # 8-point average (:7458-7462), authority term order.
        acc = w[km, cys, cxs]
        acc = acc + w[km, cys, cxs_m1]
        acc = acc + w[km, cys_m1, cxs]
        acc = acc + w[km, cys_m1, cxs_m1]
        acc = acc + w[km + 1, cys, cxs]
        acc = acc + w[km + 1, cys, cxs_m1]
        acc = acc + w[km + 1, cys_m1, cxs]
        acc = acc + w[km + 1, cys_m1, cxs_m1]
        return _F0125 * acc

    def agl(kw):
        # 4-point corner AGL height (:7487-7497), term order (i,j), (i-1,j),
        # (i,j-1), (i-1,j-1).
        acc = z_w[kw, cys, cxs] - ht[cys, cxs]
        acc = acc + (z_w[kw, cys, cxs_m1] - ht[cys, cxs_m1])
        acc = acc + (z_w[kw, cys_m1, cxs] - ht[cys_m1, cxs])
        acc = acc + (z_w[kw, cys_m1, cxs_m1] - ht[cys_m1, cxs_m1])
        return _F025 * acc

    uh_c = np.zeros((ny - 1, nx - 1), dtype=np.float32)
    use_c = np.ones((ny - 1, nx - 1), dtype=bool)
    zl = agl(0)
    wa_k = None
    rv_k = None
    for km in range(nz):
        zu = agl(km + 1)
        band = (zl >= UH_LAYER_BOTTOM_M) & (zu <= UH_LAYER_TOP_M)
        if band.any():
            if wa_k is None:
                wa_k = wavg(km)
                rv_k = rvort(km)
            if km + 1 < nz:
                wa_k1 = wavg(km + 1)
                rv_k1 = rvort(km + 1)
            else:
                # WRF's never-written ktf+1 values; defined 0.0 stand-in.
                wa_k1 = np.zeros_like(wa_k)
                rv_k1 = np.zeros_like(rv_k)
            wpos = (wa_k > _F0) & (wa_k1 > _F0)
            with np.errstate(all="ignore"):
                contrib = ((wa_k * rv_k + wa_k1 * rv_k1) * _F05) * (zu - zl)
            add = band & wpos
            kill = band & ~wpos
            uh_c = np.where(add, uh_c + contrib, uh_c)
            uh_c = np.where(kill, _F0, uh_c)
            use_c = use_c & ~kill
            wa_k, rv_k = wa_k1, rv_k1
        else:
            wa_k = rv_k = None
        zl = zu
    uh = np.zeros((ny, nx), dtype=np.float32)
    use = np.ones((ny, nx), dtype=bool)
    uh[1:, 1:] = uh_c
    use[1:, 1:] = use_c
    return uh, use


def apply_uh_smoother_max_np(uh, use, up_heli_max):
    """Smoother + running max + boundary copies (:7515-7575), in place."""
    ny, nx = up_heli_max.shape
    c = (slice(1, ny - 1), slice(1, nx - 1))

    def sh(dy, dx):
        return uh[1 + dy:ny - 1 + dy, 1 + dx:nx - 1 + dx]

    edge = ((sh(0, 1) + sh(0, -1)) + sh(1, 0)) + sh(-1, 0)
    corner = ((sh(1, 1) + sh(-1, 1)) + sh(1, -1)) + sh(-1, -1)
    uh_smth = (_F025 * uh[c] + _F0125 * edge) + _F00625 * corner
    grow = use[c] & (uh_smth > up_heli_max[c])
    up_heli_max[c] = np.where(grow, uh_smth, up_heli_max[c])
    # Boundary copies, x then y (:7541-7557); the ide/jde halo writes have
    # no slot in a (ny, nx) field (ratified divergence 4 above).
    up_heli_max[:, 0] = up_heli_max[:, 1]
    up_heli_max[0, :] = up_heli_max[1, :]


def mirror_up_heli_max_step_np(u, v, w, ph, phb, msfu, msfv, ht,
                               dn, dnw, fnm, fnp, cf1, cf2, cf3,
                               rdx, rdy, up_heli_max):
    """One WRF step of the diagnostic on host arrays; updates the max."""
    uh, use = cal_helicity_uh_columns_np(
        u, v, w, ph, phb, msfu, msfv, ht, dn, dnw, fnm, fnp,
        cf1, cf2, cf3, rdx, rdy)
    apply_uh_smoother_max_np(uh, use, up_heli_max)
    return uh, use


# ---------------------------------------------------------------------------
# Model-facing lifecycle
# ---------------------------------------------------------------------------

def _supported_boundary_geometry(cfg) -> bool:
    """cal_helicity's non-periodic bound adjustments must apply on BOTH axes
    (module_diffusion_em.F:7226-7233): specified/nested real cases, or open
    boundaries on both axes.  The periodic idealized path has no
    transcription here and refuses rather than guessing."""
    return bool(cfg.specified or cfg.nested or (cfg.open_x and cfg.open_y))


def update_up_heli_max(state, cfg) -> None:
    """Fold this step's updraft helicity into the running UP_HELI_MAX.

    Reads only completed-step model state; writes only the diagnostic's own
    scratch slots.  Call once per dycore step when ``nwp_diagnostics == 1``.
    """
    if not _supported_boundary_geometry(cfg):
        raise NotImplementedError(
            "nwp_diagnostics=1 requires specified/nested (or open-x AND "
            "open-y) lateral boundaries; WRF's cal_helicity periodic branch "
            "is not transcribed")
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    if nx < 4 or ny < 4 or nz < 3:
        raise ValueError(
            "the updraft-helicity stencil needs nx, ny >= 4 and nz >= 3; "
            f"got nx={nx} ny={ny} nz={nz}")
    # Literal slot names: the preflight completeness scanner classifies
    # every scratch(...) call site statically.
    up_heli_max = state.scratch((ny, nx), "up_heli_max")
    uh = state.scratch((ny, nx), "uh_diag_col")
    use = state.scratch((ny, nx), "uh_diag_use")
    # WRF grid%rdx/rdy: 1/dx in FP32 (map factors are separate operands).
    rdx = DTYPE(1.0) / DTYPE(cfg.dx)
    rdy = DTYPE(1.0) / DTYPE(cfg.dy)

    # One column computation, folded into every window that exists.  The
    # smoother is a pure function of (uh, use) and writes only its target,
    # so folding N targets from one pass is exactly N independent running
    # maxima over the same steps -- no second UH evaluation, no forked
    # arithmetic, and a window whose consumer never resets it is simply
    # the same number UP_HELI_MAX would carry.
    if isinstance(state.u, np.ndarray):
        uh_np, use_np = cal_helicity_uh_columns_np(
            state.u, state.v, state.w, state.php, state.phb,
            state.msfu, state.msfv, state.ht,
            _to_host(state.dn), _to_host(state.dnw),
            _to_host(state.fnm), _to_host(state.fnp),
            state.cf1, state.cf2, state.cf3, rdx, rdy)
        uh[...] = uh_np
        use[...] = use_np.astype(np.float32)
        apply_uh_smoother_max_np(uh_np, use_np, up_heli_max)
        for window in _tracker_windows(state):
            apply_uh_smoother_max_np(uh_np, use_np, window)
        return

    device_uh_step(
        state.u, state.v, state.w, state.php, state.phb,
        state.msfu, state.msfv, state.ht,
        state.dn, state.dnw, state.fnm, state.fnp,
        state.cf1, state.cf2, state.cf3, rdx, rdy,
        uh, use, up_heli_max)
    for window in _tracker_windows(state):
        device_fold_window(uh, use, window)


def _tracker_windows(state):
    """The consumer windows this state carries, in declared order.

    Duck-typed tolerant exactly like :func:`reset_up_heli_max`: reduced
    states and test doubles without a scratch pool simply have none, and
    a state built with ``nwp_diagnostics = 0`` never allocates them.
    """
    existing_scratch = getattr(state, "existing_scratch", None)
    if existing_scratch is None:
        return ()
    return tuple(window for window in
                 (existing_scratch(slot) for slot in TRACKER_WINDOW_SLOTS)
                 if window is not None)


def device_fold_window(uh, use, window) -> None:
    """Fold already-computed columns into one more running max on device.

    The second half of :func:`device_uh_step`, against a different
    target: the same ``uh_smooth_max`` kernel and the same boundary
    copies (:7541-7557), never a re-derivation of the columns.
    """
    from gpuwm.core.kernels import get_kernel

    ny, nx = window.shape
    blocks = ((nx * ny) + _THREADS - 1) // _THREADS
    get_kernel("uh_diag", "uh_smooth_max")(
        (blocks,), (_THREADS,),
        (uh, use, window, np.int32(ny), np.int32(nx)))
    window[:, 0] = window[:, 1]
    window[0, :] = window[1, :]


def reset_tracker_window(state, slot: str) -> None:
    """Zero one consumer's window, immediately after that consumer read it.

    EVERY evaluation resets, accepted or held (Drew's ruling,
    2026-08-07).  Letting the window grow across holds would make a move
    more likely simply because more time passed since the last one, which
    is cadence dependence coming back in through the reset rule instead of
    through the history writer.
    """
    if slot not in TRACKER_WINDOW_SLOTS:
        raise ValueError(
            f"{slot!r} is not a tracker window; the consumer windows are "
            f"{TRACKER_WINDOW_SLOTS} and UP_HELI_MAX is reset by the "
            "history writer alone (reset_up_heli_max)")
    existing_scratch = getattr(state, "existing_scratch", None)
    if existing_scratch is None:
        return
    buf = existing_scratch(slot)
    if buf is not None:
        buf[...] = 0.0


def device_uh_step(u, v, w, ph, phb, msfu, msfv, ht, dn, dnw, fnm, fnp,
                   cf1, cf2, cf3, rdx, rdy, uh, use, up_heli_max) -> None:
    """One WRF step of the diagnostic on device arrays (CUDA kernels)."""
    from gpuwm.core.kernels import get_kernel

    nzf, ny, nx = w.shape
    nz = nzf - 1
    phb3d = np.int32(0 if phb.ndim == 1 else 1)
    for name, arr in (("u", u), ("v", v), ("w", w), ("ph", ph),
                      ("phb", phb)):
        if arr.dtype != np.float32 or not arr.flags.c_contiguous:
            raise ValueError(f"uh_diag requires contiguous FP32 {name}")
    blocks = ((nx * ny) + _THREADS - 1) // _THREADS
    get_kernel("uh_diag", "uh_columns")(
        (blocks,), (_THREADS,),
        (u, v, w, ph, phb, phb3d, msfu, msfv, ht, dn, dnw, fnm, fnp,
         DTYPE(cf1), DTYPE(cf2), DTYPE(cf3), DTYPE(rdx), DTYPE(rdy),
         uh, use, np.int32(nz), np.int32(ny), np.int32(nx)))
    get_kernel("uh_diag", "uh_smooth_max")(
        (blocks,), (_THREADS,),
        (uh, use, up_heli_max, np.int32(ny), np.int32(nx)))
    # Boundary copies, x then y (:7541-7557).
    up_heli_max[:, 0] = up_heli_max[:, 1]
    up_heli_max[0, :] = up_heli_max[1, :]


def _to_host(arr):
    return arr if isinstance(arr, np.ndarray) else arr.get()


def reset_up_heli_max(state) -> None:
    """Zero the accumulator after its history frame is written (the WRF
    history-interval reset, module_diag_nwp.F:246-269, moved to the write
    itself -- ratified divergence 1 in the module docstring)."""
    # Duck-type tolerant like the wrfout builder: history-handler test
    # doubles and reduced states without a scratch pool simply have no
    # accumulator to reset.
    existing_scratch = getattr(state, "existing_scratch", None)
    if existing_scratch is None:
        return
    buf = existing_scratch("up_heli_max")
    if buf is not None:
        buf[...] = 0.0
