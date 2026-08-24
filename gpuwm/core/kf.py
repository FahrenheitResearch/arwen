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


#: Launch block, and the tile's granularity.  ``kernels/kf.cu`` indexes the
#: column workspace by the thread's LANE within its block, so this must equal
#: the kernel's ``KFWS_LANES`` and a launch at any other block width would
#: alias lanes.  ``tests/test_kf_workspace.py`` pins the two together.
_TPB = 32
_VALIDATION_TPB = 256
#: ``KF_KMAX`` in ``kernels/kf.cu`` and the ceiling on nz.  It is a REFUSAL
#: ceiling and nothing else since the column workspace landed: no array in
#: that file is sized by it.
_KMAX = 128
VERTICAL_LEVEL_BOUNDS = (8, _KMAX)
DTYPE = np.float32


# ---------------------------------------------------------------------------
# The per-thread column workspace
# ---------------------------------------------------------------------------
# kf.cu used to keep kf_column's 54 column arrays in the per-thread local
# frame, and CUDA prices a local frame at the card's RESIDENT-THREAD
# CAPACITY -- one per-context backing store of
# ``(frame - 1024) * SMs * maxThreadsPerSM``, taken at first launch and never
# returned.  MEASURED on node-1 (RTX 5070 Ti, 70 SMs x 1,536, sm_120, NVRTC
# 13.0.48): the 9,216 B frame at nz = 49 took 840.0 MiB -- exactly the law --
# while the kernel only ever had a fraction of that capacity in flight.
# 52 of the 54 arrays now live in a global workspace this module allocates,
# sized to the threads ACTUALLY in flight, and the columns are launched in
# tiles of that size.  (The other two stay on the stack; kf.cu says why, and
# they leave a 512 B frame that reserves nothing.)
#
# This must match kf.cu's KFWS_SLOTS; tests/test_kf_workspace.py re-derives
# it from the .cu source and fails if either side moves alone.
KFWS_SLOTS = 52

#: Blocks per SM the tile is sized for.  MEASURED, not assumed -- see the
#: sweep recorded in docs/kernel_local_memory_bounds.md.  The workspace is
#: real device memory, so an over-sized tile hands back exactly the
#: over-reservation the cut removed.
KF_TILE_BLOCKS_PER_SM = 8


def kf_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole blocks: kf.cu interleaves the workspace by LANE
    within a block, the way CUDA lays local memory out across a warp, so the
    unit of allocation is one block's region, not one column's.

    The per-slot extent is the RUNTIME ``nz``, not ``KF_KMAX``: every loop in
    ``kf_column`` runs to ``nz`` and the highest index any of them forms is
    ``nz - 1``.
    """
    blocks = (int(columns) + _TPB - 1) // _TPB
    return blocks * KFWS_SLOTS * int(nz) * _TPB


def kf_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The whole point of the workspace is that it is charged per thread IN
    FLIGHT where the local-memory backing store was charged per thread the
    card could ever hold.  Over-sizing the tile hands that back.
    """
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = KF_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            fn.kernel.ptr, _TPB, 0)
        # Never launch more blocks than the card can hold resident: those
        # columns would wait while their workspace slots stayed allocated.
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    return int(min(int(ncol), sms * per_sm * _TPB))


def kernel_capacity(nz: int) -> int:
    """Validate ``nz`` against ``KF_KMAX`` and return it.

    Nothing in ``kernels/kf.cu`` is sized by ``KF_KMAX`` any more -- the
    column arrays live in a runtime-sized global workspace -- so the module
    is compiled ONCE, at the source's own ceiling, instead of once per
    distinct level count.  What this still owns is the contract the ceiling
    names: ``nz`` past it is refused rather than silently truncated, which is
    also the guard ``kf_column`` carries.
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
    """Resolve the supported WRF microphysics-to-KF phase flags.

    ``cu_physics=1`` is WRF's ``kfetascheme``
    (``Registry/Registry.EM_COMMON:3190``), whose driver arm
    (``phys/module_cumulus_driver.F:1015``) hands ``KF_eta_CPS`` the pair
    ``F_QI=f_qi, F_QS=f_qs`` at :1043.  ``KF_eta_CPS`` selects its feedback
    branch from exactly that pair (``phys/module_cu_kfeta.F:2622-2632``):
    with ``F_QS`` true it feeds the hydrometeor tendencies back directly, and
    with ``F_QI`` also true it returns a separate ``DQIDT`` instead of
    folding the ice into snow.

    ``mp_physics=28`` is therefore SEPARATE_ICE_SNOW for the same reason
    ``mp_physics=8`` is: ``Registry/Registry.EM_COMMON:3036`` declares the
    ``thompsonaero`` package as ``moist:qv,qc,qr,qi,qs,qg``, so ``P_QI`` and
    ``P_QS`` are both allocated and ``F_QI``/``F_QS`` are both true.  Before
    28 was admitted here, ``initialize_physics`` could not even construct an
    mp=28 + KF domain -- ``_cumulus_optional_tendency_components`` raised.

    ``mp_physics=9`` (Milbrandt-Yau) resolves the same way and by the same
    single test: ``Registry/Registry.EM_COMMON:3025`` declares ``package
    milbrandt2mom mp_physics==9 - moist:qv,qc,qr,qi,qs,qg,qh``, so ``P_QI``
    and ``P_QS`` are both allocated, ``F_QI``/``F_QS`` are both true and
    ``KF_eta_CPS`` takes the separate-DQIDT branch.  The extra ``qh`` in
    that package changes nothing here: KF's feedback pair is F_QI/F_QS and
    hail is not one of them.
    """
    if int(mp_physics) in (6, 8, 9, 10, 16, 18, 28):
        return KFPhaseMode.SEPARATE_ICE_SNOW
    if int(mp_physics) == 1:
        return KFPhaseMode.WARM_RAIN
    if int(mp_physics) in (0, 50):
        # WRF sets WARM_RAIN only for Kessler.  With no microphysics scheme,
        # F_QS is false and KF uses the melting-level !F_QS closure.
        #
        # mp_physics=50 (P3) reaches the SAME branch, and WRF's own control
        # flow says so without any interpretation on gpuwm's part.  The
        # feedback cascade is
        # ``IF (warm_rain) ... ELSEIF (.NOT. F_QS) ... ELSEIF (F_QS) ...``
        # (phys/module_cu_kfeta.F:2599/:2607/:2622).  P3 does not set
        # ``warm_rain``: ``mp_init`` initialises it .false. at
        # module_physics_init.F:4459 and only the Kessler cases assign it
        # .true. (:4477, :4480), while P3's own case at :4568 does not
        # touch it.  And ``F_QS`` is false because the ``p3_1category``
        # package declares ``moist:qv,qc,qr,qi`` with no qs at all
        # (Registry.EM_COMMON:3038).  warm_rain false + F_QS false selects
        # the melting-level closure at :2607 -- the one this mode names.
        #
        # The tendency contract matches: that branch sets DQIDT = 0 and
        # DQSDT = 0 (:2618, :2620) and folds ice into DQCDT and snow into
        # DQRDT, so KF contributes ``rqr`` only and asks P3's state for no
        # snow or ice tendency array it does not have
        # (_cumulus_optional_tendency_components, physics.py).
        #
        # This is a DERIVED admission, not a measured one: it is exactly as
        # verified as the mp=0 row beside it, and no P3+KF forecast has been
        # compared against WRF.  What IS executed is
        # tests/test_p3_port.py::
        # test_p3_runs_under_the_default_cumulus_scheme, which builds an
        # mp=50 + cu_physics=1 domain through initialize_physics and steps
        # microphysics on it.
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


@lru_cache(maxsize=None)
def _device_table_on(_device: int):
    """The KF_LUTAB on ONE card.  The argument is the cache key and nothing
    else -- it is the CURRENT device, which is what ``cp.asarray`` uploads to.

    Keyed on the device because the alternative was measured to be fatal: a
    process-wide ``lru_cache`` handed every card the pointers of whichever
    one uploaded first, and on a dual-4090 (consumer Ada, no P2P) the second
    card's KF kernel died with CUDA_ERROR_ILLEGAL_ADDRESS -- which destroys
    the context for the whole process, so every later run in it also failed,
    at unrelated allocations.  The table is 250x220x2 + 220 + 200 float32
    = 0.42 MiB, so a copy per card is not a cost worth avoiding.
    """
    import cupy as cp

    table = load_kf_table()
    return tuple(cp.asarray(value) for value in
                 (table.temperature, table.qsat, table.thetae_base,
                  table.log_ratio))


def _device_table():
    import cupy as cp

    return _device_table_on(cp.cuda.runtime.getDevice())


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
    from gpuwm.core.kernels import get_kernel

    table = load_kf_table()
    device_table = _device_table()
    kernel_capacity(nz)          # the refusal, on the same path as the launch
    kernel = get_kernel("kf", "kf_column")
    ncol = ny * nx
    # The column arrays live in a global workspace sized to the threads in
    # flight, not in the per-thread local frame (which CUDA prices at the
    # card's whole resident-thread capacity).  Columns therefore go in tiles
    # of `tile`, each launched with the SAME one-thread-one-column mapping
    # the kernel has always had -- shifted by `col0`, because the state
    # arrays are (nz, ny, nx) and a tile of columns is not a contiguous
    # slice of them.
    tile = kf_tile_columns(kernel, ncol)
    ws = cp.empty(kf_workspace_floats(nz, tile), dtype=DTYPE)
    for col0 in range(0, ncol, tile):
        span = min(tile, ncol - col0)
        blocks = (span + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,), (
            u, v, temperature, qv, qc, pressure, exner, dz, w,
            *device_table,
            out["rthcuten"], out["rqvcuten"], out["rqccuten"],
            out["rqicuten"], out["rqrcuten"], out["rqscuten"],
            out["rainc"], out["triggered"],
            out["cape_before"], out["cape_after"], out["timec"],
            out["nca_seconds"], out["shallow"],
            out["cloud_base"], out["cloud_top"], out["updraft_mass_flux"],
            out["downdraft_mass_flux"], ws,
            DTYPE(table.pressure_top), DTYPE(table.pressure_reciprocal),
            DTYPE(table.thetae_reciprocal), DTYPE(dx), DTYPE(dt),
            DTYPE(cudt), np.int32(phase_mode),
            np.int32(nz), np.int32(ny), np.int32(nx), np.int32(col0)))
    del ws
    return out


def validate_kf_outputs(
        values: tuple, active: int, status, *, describe=None) -> int:
    """Return finite-check flags for native KF feedback arrays.

    ``describe`` opts this site into :mod:`gpuwm.core.health_ledger`; see
    :func:`gpuwm.core.microphysics.validate_surface_diagnostics`.

    ``values`` follows the driver's first-invalid order: the six 3-D rates
    ``rthcuten``, ``rqvcuten``, ``rqccuten``, ``rqicuten``, ``rqrcuten``,
    ``rqscuten``, followed by the 2-D ``nca_seconds`` and ``pratec`` fields.
    Every pointer is valid even when its bit is inactive; the validation
    kernel does not load inactive fields.
    """
    import cupy as cp

    if len(values) != 8:
        raise ValueError(
            f"native KF validation requires 8 arrays, got {len(values)}")
    if (isinstance(active, (bool, np.bool_))
            or not isinstance(active, (int, np.integer))
            or int(active) < 0 or int(active) > 0xff):
        raise ValueError(
            f"KF validation active mask must fit eight bits, got {active!r}")
    if (not isinstance(status, cp.ndarray) or status.shape != (1,)
            or status.dtype != cp.uint32 or not status.flags.c_contiguous):
        raise ValueError(
            "KF validation status must be a C-contiguous uint32 device "
            "array with shape (1,)")

    shape_3d = values[0].shape
    shape_2d = values[6].shape
    if len(shape_3d) != 3 or len(shape_2d) != 2:
        raise ValueError("native KF outputs must retain 3-D/2-D shapes")
    if shape_2d != shape_3d[1:]:
        raise ValueError(
            "native KF surface outputs must match the 3-D horizontal shape")
    for index, value in enumerate(values[:6]):
        if (not isinstance(value, cp.ndarray) or value.shape != shape_3d
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            raise ValueError(
                f"native KF 3-D output {index} must be a C-contiguous "
                f"float32 CuPy array with shape {shape_3d}")
    for index, value in enumerate(values[6:], start=6):
        if (not isinstance(value, cp.ndarray) or value.shape != shape_2d
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            raise ValueError(
                f"native KF 2-D output {index} must be a C-contiguous "
                f"float32 CuPy array with shape {shape_2d}")

    status.fill(cp.uint32(0))
    count_3d = values[0].size
    count_2d = values[6].size
    blocks = (max(count_3d, count_2d) + _VALIDATION_TPB - 1) // _VALIDATION_TPB
    from gpuwm.core.kernels import get_kernel

    get_kernel("kf_validation", "kf_validate_outputs")(
        (blocks,), (_VALIDATION_TPB,),
        values + (np.uint32(active), status,
                  np.int64(count_3d), np.int64(count_2d)))
    from gpuwm.core import health_ledger

    return health_ledger.read_status(
        status, site="kain-fritsch", describe=describe)


class KainFritsch:
    """Stateful ``cu_physics=1`` callable for :class:`PhysicsDriver`.

    The driver remains the sole cadence authority.  Each call returns rates
    to hold through the next STEPCU interval and the matching RAINC increment.
    """

    def __init__(self):
        self.w0avg = None
        self._history_state = None
        self._history_time = None

    def ensure_trigger_history(self, state):
        """Materialise ``w0avg`` NOW rather than on the first due call.

        ``w0avg`` is a CARRIER (``restart.CUMULUS_CALLABLE_ARRAYS``), so it
        belongs to the set a tiled or decomposed run ships between buffers.
        Allocating it lazily means an inventory taken at construction does not
        contain it and an inventory taken after the first cumulus event does:
        the carrier set CHANGES IDENTITY MID-RUN.  A seam plan is built once
        from the inventory and cannot follow that, so
        ``tilestream.multigpu.MultiGPUDomain.refresh_arrays`` aborts the run
        the moment cumulus first fires -- at cudt=5 min and dt=15 s that is
        step 20, twenty steps after anything could still be called a
        start-up failure.  ``tilestream.driver.make_physics_tile_state``
        works around it by throwing a warm-up step away; the array is
        (nz, ny, nx) zeros and there is nothing to warm up.

        Idempotent, and it re-allocates on a state or shape change exactly as
        the lazy path did, so a driver moved to a new state still gets a
        fresh mean rather than the previous domain's memory.
        """
        import cupy as cp

        shape = (int(state.w.shape[0]) - 1,) + tuple(
            int(n) for n in state.w.shape[1:])
        if (self.w0avg is None or self._history_state is not state
                or tuple(self.w0avg.shape) != shape):
            self.w0avg = cp.zeros(shape, dtype=DTYPE)
            self._history_state = state
            self._history_time = None
        return self.w0avg

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
        self.ensure_trigger_history(state)
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
        from gpuwm.core.physics import _NativeKFCumulusResult

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
        return _NativeKFCumulusResult(
            owner=self,
            rthcuten=result["rthcuten"], rqvcuten=result["rqvcuten"],
            rqccuten=result["rqccuten"],
            rqicuten=(result["rqicuten"] if separate_ice else None),
            rqrcuten=result["rqrcuten"],
            rqscuten=(result["rqscuten"] if separate_snow else None),
            rainc=result["rainc"], nca_seconds=nca, pratec=pratec)


__all__ = ["KFPhaseMode", "KFTable", "KainFritsch", "KFWS_SLOTS",
           "KF_TILE_BLOCKS_PER_SM", "kernel_capacity",
           "kf_phase_mode_for_microphysics", "kf_tile_columns",
           "kf_workspace_floats", "launch_kf", "load_kf_table",
           "validate_kf_outputs"]
