"""Kain--Fritsch cumulus column scheme and CUDA launcher.

The numerical authority is :func:`gpuwm.verify.npref.np_kf_column`.  This
module owns the packaged ``KF_LUTAB`` data and the device-facing adapter for
WRF ``cu_physics=1``.  Array layout is ``(nz, ny, nx)`` with x fastest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

import numpy as np


_TPB = 32
#: Unspecialized ``KF_KMAX`` in ``kernels/kf.cu`` and the ceiling on nz.
_KMAX = 128
VERTICAL_LEVEL_BOUNDS = (8, _KMAX)
DTYPE = np.float32


def kernel_local_frame_bytes(nz: int) -> int:
    """Per-thread local frame ``kf_column`` compiles to at ``KF_KMAX = nz``.

    ``kf_column`` declares 47 float/int column arrays of ``KF_KMAX`` entries
    and nothing else that scales with the bound, so the frame the driver
    reports is exactly ``47 * 4 * KF_KMAX`` bytes.  Measured on the RTX 5090
    (driver 610.74, NVRTC 13.x, ``-std=c++17``): 24,064 B at 128 and 9,212 B
    at 49, both exactly 188 B per level.  ``gpuwm.core.preflight`` prices the
    launch-time local-memory reservation from this, and
    ``tests/test_kernel_local_bounds.py`` re-measures it against the driver.
    """
    return 188 * int(nz)


def kernel_capacity(nz: int) -> int:
    """The ``KF_KMAX`` this ``nz`` compiles against.

    Exactly ``nz``: every loop in ``kf_column`` runs to the runtime ``nz``
    and the highest index any of them forms is ``nz - 1``, so the bound is a
    pure allocation size.  Specializing it is worth 3,699 MiB of driver
    local-memory reservation on a 49-level case (see the module comment in
    ``kernels/kf.cu``), which is why this is exact and not tiered.
    """
    nz = int(nz)
    minimum, maximum = VERTICAL_LEVEL_BOUNDS
    if nz < minimum or nz > maximum:
        raise ValueError(
            f"KF requires {minimum} <= nz <= {maximum}, got {nz}")
    return nz


class KFPhaseMode(IntEnum):
    """WRF KF output-category contract derived from ``F_QI/F_QS``.

    The integer values are part of the CUDA ABI.  They distinguish WRF's
    four feedback branches instead of reducing the phase contract to a
    Morrison/non-Morrison boolean: the first two branches also apply the
    latent-fusion temperature adjustment required when frozen condensate is
    returned through liquid prognostics.
    """

    WARM_RAIN = 0
    NO_SEPARATE_SNOW = 1
    SEPARATE_SNOW = 2
    SEPARATE_ICE_SNOW = 3


def kf_phase_mode_for_microphysics(mp_physics: int) -> KFPhaseMode:
    """Resolve the supported WRF microphysics-to-KF phase flags."""
    if int(mp_physics) in (6, 8, 10, 18):
        return KFPhaseMode.SEPARATE_ICE_SNOW
    if int(mp_physics) == 1:
        return KFPhaseMode.WARM_RAIN
    if int(mp_physics) == 0:
        # WRF sets WARM_RAIN only for Kessler.  With no microphysics scheme,
        # F_QS is false and KF uses the melting-level !F_QS closure.
        return KFPhaseMode.NO_SEPARATE_SNOW
    raise ValueError(
        f"KF has no verified phase-output contract for mp_physics="
        f"{mp_physics}")


def _model_clock_dt(cfg) -> float:
    """WRF's model-clock ``dt`` for clock-defined cumulus arithmetic.

    The real74 compatibility integrator advances internal substeps
    (``cfg.clock_dt > cfg.dt``); KF's driver formulas are defined on the
    model clock, the same idiom as ``lateral_boundary_clock_dt`` and the
    clock-scaled sixth-order diffusion factor.
    """
    clock_dt = float(getattr(cfg, "clock_dt", 0.0) or 0.0)
    return clock_dt if clock_dt > 0.0 else float(cfg.dt)


@dataclass(frozen=True)
class KFTable:
    """Host representation of WRF's private ``/KFLUT/`` state."""

    temperature: np.ndarray
    qsat: np.ndarray
    thetae_base: np.ndarray
    log_ratio: np.ndarray
    pressure_top: float
    pressure_reciprocal: float
    thetae_reciprocal: float


@lru_cache(maxsize=1)
def load_kf_table() -> KFTable:
    """Load the shipped FP32 rendering of WRF ``KF_LUTAB``.

    The generating transcription and SHA-256 provenance live beside the
    table in ``gpuwm/data/kf_lutab``.
    """
    path = (Path(__file__).parents[1] / "data" / "kf_lutab" /
            "kf_lutab.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.ascontiguousarray(data[name], dtype=np.float32)
                  for name in ("temperature", "qsat", "thetae_base",
                               "log_ratio")}
        scalars = {name: float(data[name]) for name in
                   ("pressure_top", "pressure_reciprocal",
                    "thetae_reciprocal")}
    if (arrays["temperature"].shape != (250, 220)
            or arrays["qsat"].shape != (250, 220)
            or arrays["thetae_base"].shape != (220,)
            or arrays["log_ratio"].shape != (200,)):
        raise ValueError(f"invalid KF_LUTAB dimensions in {path}")
    return KFTable(**arrays, **scalars)


@lru_cache(maxsize=1)
def _device_table():
    import cupy as cp

    table = load_kf_table()
    return tuple(cp.asarray(value) for value in
                 (table.temperature, table.qsat, table.thetae_base,
                  table.log_ratio))


def launch_kf(u, v, temperature, qv, qc, pressure, exner, dz, w, *,
              dx: float, dt: float, cudt: float,
              phase_mode: KFPhaseMode = KFPhaseMode.SEPARATE_ICE_SNOW):
    """Run KF for a batch of FP32 device columns and return its outputs."""
    import cupy as cp

    columns = {"u": u, "v": v, "temperature": temperature, "qv": qv,
               "qc": qc, "pressure": pressure, "exner": exner,
               "dz": dz, "w": w}
    shape = temperature.shape
    if len(shape) != 3:
        raise ValueError(f"temperature must have shape (nz,ny,nx), got {shape}")
    nz, ny, nx = shape
    minimum, maximum = VERTICAL_LEVEL_BOUNDS
    if nz < minimum or nz > maximum:
        raise ValueError(
            f"KF requires {minimum} <= nz <= {maximum}, got {nz}")
    for name, array in columns.items():
        if (not isinstance(array, cp.ndarray) or array.shape != shape
                or array.dtype != DTYPE or not array.flags.c_contiguous):
            raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                             f"array with shape {shape}")
    for name, value in (("dx", dx), ("dt", dt), ("cudt", cudt)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"KF {name} must be positive and finite, got {value}")
    if isinstance(phase_mode, (bool, np.bool_)):
        raise ValueError("KF phase_mode must be an explicit KFPhaseMode")
    try:
        phase_mode = KFPhaseMode(phase_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid KF phase_mode {phase_mode!r}") from exc

    out = {name: cp.zeros(shape, dtype=DTYPE) for name in
           ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
            "rqrcuten", "rqscuten",
            "updraft_mass_flux", "downdraft_mass_flux")}
    out.update(
        rainc=cp.zeros((ny, nx), dtype=DTYPE),
        triggered=cp.zeros((ny, nx), dtype=cp.int32),
        cape_before=cp.zeros((ny, nx), dtype=DTYPE),
        cape_after=cp.zeros((ny, nx), dtype=DTYPE),
        timec=cp.zeros((ny, nx), dtype=DTYPE),
        nca_seconds=cp.zeros((ny, nx), dtype=DTYPE),
        shallow=cp.zeros((ny, nx), dtype=cp.int32),
        cloud_base=cp.full((ny, nx), -1, dtype=cp.int32),
        cloud_top=cp.full((ny, nx), -1, dtype=cp.int32),
    )
    from gpuwm.core.kernels import get_kernel, get_kernel_int_defines

    table = load_kf_table()
    device_table = _device_table()
    # Compile the column arrays to this configuration's own level count.  At
    # the unspecialized 128 the frame is 24,064 B/thread and the driver
    # reserves 5,738 MiB of local-memory backing store for the process at
    # first launch; at nz = 49 it is 9,212 B and 2,039 MiB.
    capacity = kernel_capacity(nz)
    kernel = (get_kernel("kf", "kf_column") if capacity == _KMAX else
              get_kernel_int_defines("kf", "kf_column",
                                     (("KF_KMAX", capacity),)))
    ncol = ny * nx
    blocks = (ncol + _TPB - 1) // _TPB
    kernel((blocks,), (_TPB,), (
        u, v, temperature, qv, qc, pressure, exner, dz, w,
        *device_table,
        out["rthcuten"], out["rqvcuten"], out["rqccuten"],
        out["rqicuten"], out["rqrcuten"], out["rqscuten"],
        out["rainc"], out["triggered"],
        out["cape_before"], out["cape_after"], out["timec"],
        out["nca_seconds"], out["shallow"],
        out["cloud_base"], out["cloud_top"], out["updraft_mass_flux"],
        out["downdraft_mass_flux"],
        DTYPE(table.pressure_top), DTYPE(table.pressure_reciprocal),
        DTYPE(table.thetae_reciprocal), DTYPE(dx), DTYPE(dt), DTYPE(cudt),
        np.int32(phase_mode),
        np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


class KainFritsch:
    """Stateful ``cu_physics=1`` callable for :class:`PhysicsDriver`.

    The driver remains the sole cadence authority.  Each call returns rates
    to hold through the next STEPCU interval and the matching RAINC increment.
    """

    def __init__(self):
        self.w0avg = None
        self._history_state = None
        self._history_time = None

    def update_trigger_history(self, *, state, cfg):
        """Advance WRF's running-mean trigger velocity for one due call.

        WRF's cumulus driver early-returns on non-due steps
        (``module_cumulus_driver.F:830-864``, ``RETURN`` at :863), so the
        fixed-step W0AVG loop at the top of ``KF_eta_CPS``
        (``module_cu_kfeta.F:232-250``) executes exactly once per STEPCU
        due event, with weight ``1/TST`` and ``TST = 2*STEPCU`` built
        from the model clock -- at cudt=5 that is one sample per 300 s
        and a trigger memory of ~2850 s.  The physics driver invokes this
        once per due call, immediately before the column scheme consumes
        the mean and before the per-column NCA skip test, so held columns
        still refresh their trigger memory.  (Controller re-adjudication
        2026-07-16: the T1 every-step cadence misread the driver's early
        return; v4.6.1 source wins.)
        """
        import cupy as cp

        instantaneous = cp.ascontiguousarray(
            DTYPE(0.5) * (state.w[:-1] + state.w[1:]), dtype=DTYPE)
        if (self.w0avg is None or self._history_state is not state
                or self.w0avg.shape != instantaneous.shape):
            self.w0avg = cp.zeros_like(instantaneous)
            self._history_state = state
            self._history_time = None
        clock_dt = _model_clock_dt(cfg)
        stepcu = (1 if cfg.cudt_minutes <= 0.0 else
                  max(int(np.floor(
                      cfg.cudt_minutes * 60.0 / clock_dt + 0.5)), 1))
        tst = DTYPE(2 * stepcu)
        self.w0avg[...] = (self.w0avg * (tst - DTYPE(1.0))
                           + instantaneous) / tst
        self._history_time = float(state.elapsed_seconds)

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        # Direct callers still receive the first WRF history update.  A driver
        # call at the same elapsed time has already run the hook and must not
        # double-count that sample.
        if (self.w0avg is None or self._history_state is not state
                or self._history_time != float(state.elapsed_seconds)):
            self.update_trigger_history(state=state, cfg=cfg)
        # WRF hands the scheme its model DT (module_cumulus_driver.F:1028),
        # which is the outer clock when the case integrates internal
        # substeps: NIC/TIMEC rounding and the NCA durations stay on WRF's
        # 60 s granularity.
        clock_dt = _model_clock_dt(cfg)
        steps = (1 if cfg.cudt_minutes <= 0.0 else
                 max(int(np.floor(cfg.cudt_minutes * 60.0 / clock_dt + 0.5)),
                     1))
        cudt = (clock_dt if cfg.cudt_minutes <= 0.0
                else steps * clock_dt)
        # Documented deviation (Task 6b audit): the verified scheme derives
        # environmental density internally as p/(Rd*Tv) with
        # Tv = T*(1+0.608*qenv) (kf.cu:291-293; npref.py np_kf_column),
        # where WRF's phy_prep hands the cumulus driver moist
        # rho = 1/alt*(1+qv) (module_big_step_utilities_em.F:4856) --
        # an O(0.1%) layer-mass difference in humid columns.  Adopting
        # WRF's form would change the frozen kernel's input surface, so it
        # is recorded here rather than silently absorbed.
        phase_mode = kf_phase_mode_for_microphysics(cfg.mp_physics)
        result = launch_kf(
            atmosphere["u"], atmosphere["v"], atmosphere["temperature"],
            atmosphere["qv"], atmosphere["qc"], atmosphere["pressure"],
            atmosphere["exner"], atmosphere["dz"], self.w0avg,
            dx=cfg.dx, dt=clock_dt, cudt=cudt,
            phase_mode=phase_mode)
        from gpuwm.core.physics import CumulusResult

        # WRF's persistent rain rate PRATEC = PPTFLX*(1-FBFRC)/DXSQ in
        # mm s-1 (module_cu_kfeta.F:2504).  The kernel reports the rain
        # increment integrated over min(cudt, feedback_time) with
        # nca_seconds = feedback_time for deep and cudt for (rain-free)
        # shallow columns (kf.cu:1140-1149), so the matching division
        # recovers the rate; columns that did not trigger report zero.
        nca = result["nca_seconds"]
        divisor = cp.minimum(DTYPE(cudt), nca)
        safe = cp.where(divisor > DTYPE(0.0), divisor, DTYPE(1.0))
        pratec = cp.where(divisor > DTYPE(0.0), result["rainc"] / safe,
                          DTYPE(0.0))
        separate_ice = phase_mode == KFPhaseMode.SEPARATE_ICE_SNOW
        separate_snow = phase_mode in (
            KFPhaseMode.SEPARATE_SNOW, KFPhaseMode.SEPARATE_ICE_SNOW)
        return CumulusResult(
            rthcuten=result["rthcuten"], rqvcuten=result["rqvcuten"],
            rqccuten=result["rqccuten"],
            rqicuten=(result["rqicuten"] if separate_ice else None),
            rqrcuten=result["rqrcuten"],
            rqscuten=(result["rqscuten"] if separate_snow else None),
            rainc=result["rainc"], nca_seconds=nca, pratec=pratec)


__all__ = ["KFPhaseMode", "KFTable", "KainFritsch",
           "kf_phase_mode_for_microphysics", "launch_kf", "load_kf_table"]
