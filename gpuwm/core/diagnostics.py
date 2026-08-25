"""Thin Python launcher for the equation-of-state diagnostics kernel."""

from __future__ import annotations

import numpy as np

from gpuwm.core import constants as c
from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DomainState

_THREADS = 256


def update_diagnostics(state: DomainState, hypsometric_opt: int = 1,
                       window: tuple[int, int, int, int] | None = None
                       ) -> None:
    """Recompute ``p``, ``al``, ``alt`` in place from (thp, php, mup[, qv]).

    General hybrid/terrain form: the kernel consumes the 2-D dry mass
    ``mub2d`` plus the hybrid coefficients ``c1h``/``c2h``, and the base
    profiles either as Phase 1 flat columns or per-column 3-D fields.
    Moist states (Task 5) evaluate the EOS on ``theta_m = theta *
    (1 + Rv/Rd*qv)``; dry states pass ``thp`` as a dummy qv pointer that
    the kernel never reads (moist = 0).

    ``hypsometric_opt`` keys the WRF calc_p_rho_phi branch
    (dyn_em/module_big_step_utilities_em.F:1025-1052): 1 is the frozen
    gpuwm d(phi)/d(eta) inversion (bitwise the pre-option kernel); 2 is
    the WRF Registry-default log-pressure form, which additionally reads
    the reference-pressure coefficients ``c3h/c4h/c3f/c4f`` and
    ``state.p_top`` loaded by ``load_base``.  Callers with a
    :class:`~gpuwm.config.RunConfig` pass ``cfg.hypsometric_opt``;
    idealized initializers keep the default 1 exactly as WRF forces for
    ideal cases (share/input_wrf.F:1038).

    ``window`` is an optional ``(j0, i0, nyw, nxw)`` column rectangle.
    The kernel is column-local -- each thread reads and writes only its
    own column -- so a windowed call is bitwise the full call on the
    window and leaves every other column untouched; the default remains
    the whole domain, and the full-domain launch is arithmetic-identical
    to the pre-window kernel.  The consumer is the two-way-feedback
    finalize, whose restriction changes only the columns under the nest.
    """
    if hypsometric_opt not in (1, 2):
        raise ValueError(
            f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    if hypsometric_opt == 2 and state.p_top is None:
        raise RuntimeError(
            "hypsometric_opt=2 needs state.p_top: load_base must run "
            "before the log-pressure EOS diagnostic")
    if isinstance(state.p, np.ndarray):
        _update_diagnostics_numpy(state, hypsometric_opt, window)
        return
    nz, ny, nx = state.p.shape
    j0, i0, nyw, nxw = _validated_window(window, ny, nx)
    moist = state.qv is not None
    kernel = get_kernel("diagnostics", "calc_p_alpha")
    blocks = (nxw * nyw + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (state.thp, state.php, state.mup,
            state.thb, state.phb, state.alb, state.rdnw,
            state.c1h, state.c2h, state.c3h, state.c4h,
            state.c3f, state.c4f, state.mub2d,
            state.qv if moist else state.thp,
            np.float32(0.0 if state.p_top is None else state.p_top),
            np.int32(hypsometric_opt),
            np.int32(moist), np.int32(state.thb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(j0), np.int32(i0), np.int32(nyw), np.int32(nxw),
            state.p, state.al, state.alt))


def _validated_window(window, ny: int, nx: int) -> tuple[int, int, int, int]:
    """Clamp-free window validation: out of bounds is a caller bug."""
    if window is None:
        return 0, 0, ny, nx
    j0, i0, nyw, nxw = (int(v) for v in window)
    if (j0 < 0 or i0 < 0 or nyw < 1 or nxw < 1
            or j0 + nyw > ny or i0 + nxw > nx):
        raise ValueError(
            f"diagnostics window {window!r} does not fit ({ny}, {nx})")
    return j0, i0, nyw, nxw


def _update_diagnostics_numpy(state: DomainState,
                              hypsometric_opt: int,
                              window=None) -> None:
    """NumPy realization of ``calc_p_alpha`` for setup/export states.

    Operations stay FP32, matching the CUDA kernel's ``real`` arithmetic.
    This path is intentionally limited to preprocessing: forecast states
    continue to use the CUDA kernel above.  ``window`` slices the column
    rectangle exactly as the kernel's windowed launch does; the FP32
    arithmetic is per column, so the sliced compute equals the full
    compute on the window.
    """
    ny, nx = state.p.shape[-2:]
    j0, i0, nyw, nxw = _validated_window(window, ny, nx)
    win = (slice(j0, j0 + nyw), slice(i0, i0 + nxw))
    kwin = (slice(None),) + win

    def col(value):
        """Window a per-column input; broadcastable inputs pass through."""
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 3:
            return value[kwin]
        if value.ndim == 2:
            return value[win]
        return value

    def prof(value):
        value = np.asarray(value, dtype=np.float32)
        return value[kwin] if value.ndim == 3 else value[:, None, None]

    f32 = np.float32
    mu = col(np.asarray(state.mub2d + state.mup, dtype=np.float32))
    theta = np.asarray(
        prof(state.thb) + col(state.thp), dtype=np.float32)
    if state.qv is not None:
        # WRF's qvf = 1.+rvovrd*moist(i,k,j,P_QV) (calc_p_rho_phi,
        # module_big_step_utilities_em.F:1064).  ``c.RVOVRD`` is the float32
        # quotient gfortran folds; ``f32(c.RV / c.RD)`` is a *double* divide
        # rounded once and lands one ULP away, which would also put this
        # preprocessing path one ULP off the CUDA kernel that reads the very
        # same constant through the ``#define`` preamble.
        theta = np.asarray(
            theta * (f32(1.0) + f32(c.RVOVRD) * col(state.qv)),
            dtype=np.float32)
    phi = np.asarray(prof(state.phb) + col(state.php), dtype=np.float32)
    dphi = np.asarray(phi[1:] - phi[:-1], dtype=np.float32)
    alb = prof(state.alb)
    if hypsometric_opt == 2:
        p_top = f32(state.p_top)
        c3f = np.asarray(state.c3f, dtype=np.float32)[:, None, None]
        c4f = np.asarray(state.c4f, dtype=np.float32)[:, None, None]
        c3h = np.asarray(state.c3h, dtype=np.float32)[:, None, None]
        c4h = np.asarray(state.c4h, dtype=np.float32)[:, None, None]
        pfu = np.asarray(c3f[1:] * mu + c4f[1:] + p_top,
                         dtype=np.float32)
        pfd = np.asarray(c3f[:-1] * mu + c4f[:-1] + p_top,
                         dtype=np.float32)
        phm = np.asarray(c3h * mu + c4h + p_top, dtype=np.float32)
        al = np.asarray(
            dphi / phm / np.log(np.asarray(pfd / pfu, dtype=np.float32))
            - alb, dtype=np.float32)
        alt = np.asarray(al + alb, dtype=np.float32)
    else:
        rdnw = np.asarray(state.rdnw, dtype=np.float32)[:, None, None]
        c1h = np.asarray(state.c1h, dtype=np.float32)[:, None, None]
        c2h = np.asarray(state.c2h, dtype=np.float32)[:, None, None]
        alt = np.asarray(
            -dphi * rdnw / (c1h * mu + c2h), dtype=np.float32)
        al = np.asarray(alt - alb, dtype=np.float32)
    pressure_base = np.asarray(
        (f32(c.RD) * theta) / (f32(c.P0) * alt), dtype=np.float32)
    pressure = np.asarray(
        f32(c.P0) * np.power(pressure_base, f32(c.GAMMA)),
        dtype=np.float32)
    state.p[kwin] = pressure
    state.al[kwin] = al
    state.alt[kwin] = alt
