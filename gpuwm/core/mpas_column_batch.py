"""Persistent MPAS column-batch physics seam (``run_mpas_column_batch``).

THE COUNTERPARTY CONTRACT (MPAS side, pinned 2026-08-10)
--------------------------------------------------------
An MPAS CUDA dynamical core drives this seam with C-contiguous CuPy
float32 arrays shaped ``[level, column]`` (column fastest, ``k = 0`` the
LOWEST model level; interface fields ``[level + 1, column]``).  Two calls
per model step, strictly alternating:

* **Phase 1** (pre-RK, at time t): run the ARW physics chain -- legacy
  RRTMG radiation, revised-MO surface layer, Noah-MP, YSU, GF or KF
  cumulus -- on the caller's columns and return the complete
  currently-held set of RAW A-grid ``du/dv/dtheta/dq*`` physical rates.
  NO ARW mass coupling, NO A-to-C face interpolation, NO ``cp.roll``, NO
  map factors: where the ARW step converts rates into its coupled slow
  tendencies (``couple_ysu_tendencies``/``couple_column_tendencies`` in
  gpuwm/core/physics.py, consumed by ``PhysicsTendencies.add_to_slow`` at
  gpuwm/core/dycore.py:2394), this seam stops short and hands back the
  rates those functions would have consumed.  Phase 1 NEVER mutates a
  caller array: every input is copied once into seam-owned buffers, and
  the MPAS negative-qv law (their stock init carries small negative qv
  values) is honoured by clamping ``max(qv, 0)`` in the seam's own
  scratch copy, exactly as MPAS bulk physics does -- never by refusing.
* **Phase 2** (post-RK/transport): run WSM6 IN PLACE on the caller's
  arrays -- all six species ``qv/qc/qr/qi/qs/qg`` plus ``theta`` are
  updated in the caller's memory through zero-copy ``[nz, 1, ncol]``
  views.  The precipitation buckets accumulate in the seam and the
  retained microphysics heating ``h_diabatic`` (K/s) is captured with
  WRF's ``mp_tend_lim`` clamp (gpuwm/core/microphysics.py
  ``moist_physics_finish``); ARW feeds that field to every RK step's
  theta tendency (gpuwm/core/dycore.py:2396 ``add_h_diabatic_tendency``),
  so phase 1 republishes it as ``h_diabatic`` for the MPAS side to apply
  or decline explicitly -- it is NOT folded into ``dtheta``.

THE ORCHESTRATION IS THE ARW ORCHESTRATION
------------------------------------------
Phase 1 calls ``PhysicsDriver.compute`` -- the same function object the
ARW step invokes at gpuwm/core/dycore.py:2325 -- on a persistent driver
built by the same ``initialize_physics``.  Radiation runs on its own
cadence and its rates are HELD and returned on non-due calls;
surface/Noah-MP state advances; KF NCA holds and expiry, W0AVG trigger
history, RAINC/RAINBL accumulation and the call counters are all the
driver's own.  Phase 2 calls ``gpuwm.core.microphysics.apply`` followed
by ``PhysicsDriver.accept_microphysics``, the exact post-RK pair from
gpuwm/core/dycore.py:2507-2510.  The class attributes
``_PHASE1_ORCHESTRATION`` / ``_PHASE2_MICROPHYSICS`` hold those function
objects and the contract tests assert identity, so a rewrite of this
seam into thin per-kernel wrappers goes red.

HOW RAW RATES COME OUT OF A COUPLED DRIVER
------------------------------------------
The column-batch state is built so ARW mass coupling is the IDENTITY:
``c1h = 1``, ``c2h = 0``, total dry mass ``mub2d + mup = 1.0`` exactly,
map factors 1 with ``has_msf = False``, and neither ``specified`` nor
``nested`` set.  ``chm = c1h*mu + c2h = 1.0`` makes every mass-coupled
scalar tendency bit-identical to its raw physical rate (an FP32 multiply
by 1.0 is exact), so the driver's held/composed ``rtheta/rq*`` stacks
ARE the raw rates, with the driver's own hold and KF-expiry semantics
intact.  Momentum is different: ``couple_ysu_tendencies``
face-interpolates du/dv onto C-grid staggers (its ``cp.roll``), so the
seam takes du/dv from the driver's retained RAW YSU output
(``last_ysu``, retained because the seam always runs a positive
``bldt``) and holds its own serialized copy across non-due steps and
restarts.

GRID ADAPTATION (data marshalling only)
---------------------------------------
``PhysicsDriver.compute`` derives its per-step atmosphere through
``_prepare_atmosphere``, whose ARW body destaggers C-grid winds and
rebuilds hydrostatic pressure from the ARW mass coordinate.  Both are
grid transforms this batch must not receive: MPAS winds are already
A-grid and MPAS pressures are already the physics pressures.  The state
therefore carries ``prepared_physics_atmosphere``, the supplied-product
seam ``_prepare_atmosphere`` consults first, and this module fills the
SAME dict keys from the caller's fields with the SAME derivation
formulas (exner from ``(p/p0)^(Rd/cp)``, ``T = theta*exner``, physics
density ``(1 + qv)*rho_dry`` for WRF phy_prep's ``(1+qv)/alt``).  No
physics is reimplemented here.

COPIES, STATED
--------------
Per phase-1 call the seam copies each input once into a persistent
internal buffer (11 level fields + 3 interface fields), derives 4 more
(exner when absent, temperature, dz, rho), and copies the harvested
tendencies once into the persistent output buffers.  Phase 2 copies
nothing in or out of theta/species (zero-copy reshape views); it fills
two internal buffers (``alt = 1/rho_dry`` and ``php = z_interface*g``,
one kernel op each) whose round trips cost at most 1 ULP (documented
divergence: WSM6's adapter derives ``rho = 1/alt`` and ``z = php/g``,
gpuwm/core/wsm6.py:109-112).  The returned tendency arrays are
seam-owned and valid until the NEXT phase-1 call; callers that hold
them longer must copy.

RESTART LAW
-----------
``export_state()`` (legal only at a step boundary, i.e. before a phase-1
call) returns a host-side payload covering every persisted item: the
driver manifest gpuwm/io/restart.py builds (fields, held tendency
stacks, raw radiation rates, pending RAINBL, KF ``w0avg``, legacy-RRTMG
``o33d``), every serialized scratch bucket (``mp_*``, ``cu_*``), the
seam's held raw-output buffers, ``h_diabatic``, the WSM6 radii, and the
exact-integer step/cadence bookkeeping.  ``restore_state()`` on a
freshly constructed seam with the identical configuration copies
everything back in place; a restored seam continues bit-identically
(GPU-gated test).  The six transported species and theta live on the
MPAS side and restart with the caller's own state.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from gpuwm.core import constants as _constants
from gpuwm.core.microphysics import apply as _arw_microphysics_apply
from gpuwm.core.physics import PhysicsDriver, initialize_physics

__all__ = [
    "MpasColumnBatchPhysics",
    "MpasColumnPhysicsTendencies",
    "resolve_column_batch_cadences",
    "run_mpas_column_batch",
]

#: Export payload schema version.  Bump on any key-layout change.
RESTART_SCHEMA = "mpas-column-batch-v1"

#: Phase markers for the strict phase1 -> phase2 alternation.
_PHASE1, _PHASE2 = 1, 2

_LEVEL_FIELDS_P1 = ("u", "v", "theta", "pressure", "rho_dry",
                    "qv", "qc", "qr", "qi", "qs", "qg")
_SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg")

#: Seam-held raw output buffers, serialized so a restored seam keeps
#: returning the held du/dv on non-due steps (``last_ysu`` is a
#: rebuild-classified driver attribute and is None after a restore).
_OUTPUT_BUFFERS = ("du", "dv", "dtheta", "dqv", "dqc", "dqr", "dqi",
                   "dqs", "dqg", "h_diabatic")

#: Restart keys whose PRESENCE is legitimately asymmetric between a
#: stepped exporter and a freshly constructed restorer: the KF trigger
#: history exists only after the adapter's first due call, and the
#: legacy-RRTMG o33d grid only after the first radiation call.
_LAZY_RESTART_KEYS = ("cumulus/w0avg", "radiation/o33d_grid")


def _require_positive(name: str, value) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _cadence_steps(name: str, seconds: float, dt: float) -> int:
    """Exact positive step multiple for one scheme cadence.

    The seam OWNS the sub-cadences: every cadence is an exact positive
    integer multiple of the model step, kept as integer step counts and
    float64 scalar seconds -- never downcast into float32 field arrays.
    """
    seconds = _require_positive(name, seconds)
    steps = seconds / dt
    rounded = int(round(steps))
    if rounded < 1 or abs(steps - rounded) > 1.0e-9 * max(steps, 1.0):
        raise ValueError(
            f"{name}={seconds} s is not a positive integer multiple of "
            f"dt={dt} s: the seam owns the sub-cadences and refuses "
            "incommensurate ones")
    return rounded


def resolve_column_batch_cadences(*, dt, radiation_seconds,
                                  surface_pbl_seconds, cumulus_seconds,
                                  cumulus_scheme) -> dict:
    """Validate and resolve the constructor cadences (pure host logic).

    Returns the WRF-minute cadence fields the synthetic ``RunConfig``
    carries plus the exact step counts, so the CPU-hermetic tests can
    pin the cadence semantics without touching a device.

    * Radiation follows WRF's ``MOD(ITIMESTEP, STEPRA) == 1`` calendar
      (first call at step 1, then every ``stepra`` steps).
    * Surface/PBL and cumulus follow ``MOD(ITIMESTEP, STEP*) == 0``
      with the mandatory ITIMESTEP=1 call.
    * ``cumulus_scheme="gf"`` requires ``cumulus_seconds == dt`` and
      maps to WRF's pinned ``cudt = 0`` (GF recomputes every step and
      carries no NCA hold, gpuwm/config.py's own cu_physics=3 law);
      ``"kf"`` takes any commensurate cadence; ``None`` disables the
      slot (``cumulus_seconds`` must then be None too).
    """
    dt = _require_positive("dt", dt)
    if cumulus_scheme not in ("gf", "kf", None):
        raise ValueError(
            f"cumulus_scheme must be 'gf', 'kf' or None, got "
            f"{cumulus_scheme!r}")
    stepra = _cadence_steps("radiation_seconds", radiation_seconds, dt)
    stepbl = _cadence_steps("surface_pbl_seconds", surface_pbl_seconds, dt)
    resolved = {
        "dt": dt,
        "radt_minutes": stepra * dt / 60.0,
        "bldt_minutes": stepbl * dt / 60.0,
        "stepra": stepra,
        "stepbl": stepbl,
        "cu_physics": {"gf": 3, "kf": 1, None: 0}[cumulus_scheme],
    }
    if cumulus_scheme is None:
        if cumulus_seconds is not None:
            raise ValueError("cumulus_seconds requires a cumulus_scheme")
        resolved["cudt_minutes"] = 0.0
        resolved["stepcu"] = 0
        return resolved
    stepcu = _cadence_steps("cumulus_seconds", cumulus_seconds, dt)
    if cumulus_scheme == "gf":
        if stepcu != 1:
            raise ValueError(
                f"cumulus_scheme='gf' requires cumulus_seconds == dt "
                f"({dt} s): WRF pins cudt=0 for Grell-Freitas (STEPCU=1, "
                f"no NCA hold); got {cumulus_seconds} s")
        # GF carries WRF's pinned cudt=0 spelling; the effective cadence
        # is every step.
        resolved["cudt_minutes"] = 0.0
    else:
        resolved["cudt_minutes"] = stepcu * dt / 60.0
    resolved["stepcu"] = stepcu
    return resolved


def _interface_weights(z_interface_nominal) -> tuple[np.ndarray, np.ndarray]:
    """WRF ``fnm``/``fnp`` analogues from nominal interface heights.

    The legacy-RRTMG adapter interpolates layer temperature onto
    interfaces with the 1-D ``state.fnm``/``state.fnp`` weights (WRF
    phy_prep's t8w).  ARW derives them from the eta layer thicknesses;
    the column batch derives the same half-to-full weights from the
    NOMINAL layer thicknesses of the MPAS vertical mesh (a 1-D
    constructor input -- the weights are 1-D by WRF's own contract).
    Float64 setup arithmetic, FP32 storage, exactly like
    ``DomainState.load_base``.
    """
    z = np.asarray(z_interface_nominal, dtype=np.float64)
    if z.ndim != 1 or z.shape[0] < 4:
        raise ValueError(
            "z_interface_nominal_m must be a 1-D array of at least 4 "
            f"interface heights, got shape {z.shape}")
    dz = np.diff(z)
    if not np.all(np.isfinite(dz)) or np.any(dz <= 0.0):
        raise ValueError(
            "z_interface_nominal_m must increase strictly upward")
    nz = dz.shape[0]
    fnm = np.zeros(nz, dtype=np.float64)
    fnp = np.zeros(nz, dtype=np.float64)
    # WRF start_em: dn(k) = 0.5*(dnw(k) + dnw(k-1));
    # fnm(k) = 0.5*dnw(k)/dn(k); fnp(k) = 0.5*dnw(k-1)/dn(k).
    for k in range(1, nz):
        dn = 0.5 * (dz[k] + dz[k - 1])
        fnm[k] = 0.5 * dz[k] / dn
        fnp[k] = 0.5 * dz[k - 1] / dn
    return fnm.astype(np.float32), fnp.astype(np.float32)


@dataclass(frozen=True)
class MpasColumnPhysicsTendencies:
    """The complete currently-held RAW A-grid tendency set of one step.

    Every array is a seam-owned ``[level, column]`` float32 C-contiguous
    CuPy array, valid until the next phase-1 call.  ``du/dv`` (m s-2)
    are the held raw YSU momentum rates; ``dtheta`` (K s-1) is the
    composed PBL + radiation + cumulus rate; ``dq*`` (kg kg-1 s-1) are
    the composed moisture rates (``dqg`` is identically zero: no phase-1
    scheme produces a graupel tendency).  ``h_diabatic`` (K s-1) is the
    retained previous-step microphysics heating ARW adds to every RK
    theta tendency (gpuwm/core/dycore.py:2396); it is NOT folded into
    ``dtheta`` so the MPAS side applies or declines it explicitly.
    """

    du: object
    dv: object
    dtheta: object
    dqv: object
    dqc: object
    dqr: object
    dqi: object
    dqs: object
    dqg: object
    h_diabatic: object
    step_index: int
    elapsed_seconds: float
    radiation_ran: bool
    surface_pbl_ran: bool
    cumulus_ran: bool


class _ColumnBatchState:
    """The minimal ARW-shaped state the physics chain consumes.

    Grid ``(nz, ny=1, nx=ncol)``.  Deliberately NOT a ``DomainState``:
    the dycore's prognostic/acoustic/RK inventory would cost dozens of
    dead ``nz*ncol`` device arrays per batch.  Every attribute here is
    one the phase-1 chain or the phase-2 WSM6 adapter actually reads,
    and the identity-coupling constants (``c1h=1``, ``c2h=0``,
    ``mub2d + mup = 1``, unit map factors) are what make the driver's
    mass-coupled tendencies bit-identical to the raw physical rates.
    """

    def __init__(self, cfg, fnm: np.ndarray, fnp: np.ndarray,
                 p_top: float, terrain_height, xp):
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        shape3 = (nz, ny, nx)

        def zeros(*shape):
            return xp.zeros(shape, dtype=np.float32)

        # Thermodynamic carriers.  thb = 0 makes total theta == thp, so
        # a phase-2 binding of thp to the caller's theta view IS the
        # full theta WSM6 updates in place.
        self.p = zeros(*shape3)
        self.thp = zeros(*shape3)
        self.thb = zeros(nz)
        self.phb = zeros(nz + 1)
        self.php = zeros(nz + 1, ny, nx)
        self.alt = zeros(*shape3)
        self.w = zeros(nz + 1, ny, nx)
        for name in _SPECIES:
            setattr(self, name, zeros(*shape3))
        # WSM6 scheme-native radii (microns), the mp=6 cold start of
        # gpuwm/core/state.py (module_model_constants.F RE_*_BG).
        self.effc = xp.full(shape3, np.float32(2.49), dtype=np.float32)
        self.effi = xp.full(shape3, np.float32(4.99), dtype=np.float32)
        self.effs = xp.full(shape3, np.float32(9.99), dtype=np.float32)
        # Retained microphysics heating (WRF h_diabatic, K/s); zero at
        # init exactly as WRF start_em, one-step lag by construction.
        self.h_diabatic = zeros(*shape3)

        # Identity mass coupling: chm = c1h*(mub2d + mup) + c2h == 1.0.
        self.mup = zeros(ny, nx)
        self.mub2d = xp.ones((ny, nx), dtype=np.float32)
        self.c1h = xp.ones(nz, dtype=np.float32)
        self.c2h = zeros(nz)
        self.msft = xp.ones((ny, nx), dtype=np.float32)
        self.msfu = xp.ones((ny, nx + 1), dtype=np.float32)
        self.msfv = xp.ones((ny + 1, nx), dtype=np.float32)
        self.has_msf = False

        # Static column geometry and the t8w interface weights.
        self.ht = xp.ascontiguousarray(
            xp.asarray(terrain_height, dtype=np.float32).reshape(ny, nx))
        self.fnm = xp.asarray(fnm, dtype=np.float32)
        self.fnp = xp.asarray(fnp, dtype=np.float32)
        self.p_top = float(p_top)

        # Model clock (float64 scalar seconds; the seam keeps the exact
        # integer step count and re-derives this, never accumulates).
        self.elapsed_seconds = 0.0
        self.physics = None
        #: Supplied-atmosphere seam physics._prepare_atmosphere consults.
        self.prepared_physics_atmosphere = None
        self._scratch: dict = {}
        self._xp = xp

    # -- DomainState-compatible surface ---------------------------------
    def scratch(self, shape, slot: str, dtype=None):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        requested = np.dtype(np.float32 if dtype is None else dtype)
        buf = self._scratch.get(slot)
        if buf is None:
            buf = self._xp.zeros(shape, dtype=requested)
            self._scratch[slot] = buf
        elif buf.shape != shape:
            raise ValueError(f"scratch slot {slot!r} has shape {buf.shape}, "
                             f"requested {shape}")
        elif buf.dtype != requested:
            raise ValueError(f"scratch slot {slot!r} has dtype {buf.dtype}, "
                             f"requested {requested}")
        return buf

    def existing_scratch(self, slot: str):
        return self._scratch.get(slot)

    def total_theta(self):
        return self.thb[:, None, None] + self.thp

    def total_mu(self):
        return self.mub2d + self.mup


class MpasColumnBatchPhysics:
    """Persistent two-phase physics seam for one MPAS column batch.

    Construct once (``run_mpas_column_batch``), then per model step call
    ``run_phase1`` (pre-RK tendencies) and ``run_phase2`` (post-RK WSM6
    in place), strictly alternating.  See the module docstring for the
    full contract.
    """

    #: The phase-1 orchestration IS the ARW step's physics entry: the
    #: same function object gpuwm/core/dycore.py:2325 invokes.  The
    #: contract test asserts identity; a thin-wrapper rewrite goes red.
    _PHASE1_ORCHESTRATION = staticmethod(PhysicsDriver.compute)
    #: The phase-2 entry IS the ARW post-RK microphysics dispatch
    #: (gpuwm/core/dycore.py:2507).
    _PHASE2_MICROPHYSICS = staticmethod(_arw_microphysics_apply)

    def __init__(self, *, n_levels, n_columns, dt,
                 radiation_seconds, surface_pbl_seconds,
                 cumulus_seconds=None, cumulus_scheme=None,
                 start_time, latitude_deg, longitude_deg,
                 terrain_height_m, z_interface_nominal_m,
                 p_top_pa, dx_m, gf_ishallow=None, dx_column_m=None,
                 landmask=1.0, xland=None, ivgtyp=10, isltyp=6,
                 vegfra=50.0, tsk=300.0, tmn=285.0, xice=0.0, snow=0.0,
                 snow_depth=0.0, soil_temperature=285.0,
                 soil_moisture=0.30, xice_threshold=0.5,
                 wsm6_hail_opt=0):
        # ---- pure validation FIRST: every refusal below fires before a
        # single device allocation, so the CPU-hermetic contract tests
        # can exercise the whole refusal surface without touching a GPU.
        if not isinstance(n_levels, numbers.Integral) or n_levels < 3:
            raise ValueError(
                f"n_levels must be an integer >= 3, got {n_levels!r}")
        if not isinstance(n_columns, numbers.Integral) or n_columns < 1:
            raise ValueError(
                f"n_columns must be a positive integer, got {n_columns!r}")
        if not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime (UTC)")
        cadences = resolve_column_batch_cadences(
            dt=dt, radiation_seconds=radiation_seconds,
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cumulus_scheme=cumulus_scheme)
        p_top = _require_positive("p_top_pa", p_top_pa)
        dx = _require_positive("dx_m", dx_m)
        if int(wsm6_hail_opt) not in (0, 1):
            raise ValueError(
                f"wsm6_hail_opt must be 0 or 1, got {wsm6_hail_opt!r}")
        threshold = float(xice_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "xice_threshold must be a finite fraction in [0, 1] "
                f"(WRF Registry default 0.5), got {xice_threshold!r}")
        # GF shallow convection.  Native MPAS v8.4.1 HARDWIRES ishallow=1
        # (mpas_atmphys_vars.F:340), so the parity default for this seam is
        # shallow ON -- "fixed means default".  gf_ishallow=0 is the
        # pre-parity behaviour, kept reachable for A/B arms only.
        if gf_ishallow is None:
            gf_ishallow = 1 if cumulus_scheme == "gf" else 0
        if gf_ishallow not in (0, 1):
            raise ValueError(
                f"gf_ishallow must be 0 or 1, got {gf_ishallow!r}")
        if gf_ishallow and cumulus_scheme != "gf":
            raise ValueError(
                "gf_ishallow=1 requires cumulus_scheme='gf'")
        nz, ncol = int(n_levels), int(n_columns)
        fnm, fnp = _interface_weights(z_interface_nominal_m)
        if fnm.shape[0] != nz:
            raise ValueError(
                f"z_interface_nominal_m has {fnm.shape[0] + 1} interfaces "
                f"but n_levels={nz} needs {nz + 1}")
        lat = np.asarray(latitude_deg, dtype=np.float32).reshape(-1)
        lon = np.asarray(longitude_deg, dtype=np.float32).reshape(-1)
        if lat.shape != (ncol,) or lon.shape != (ncol,):
            raise ValueError(
                "latitude_deg/longitude_deg must have one value per "
                f"column ({ncol}), got {lat.shape}/{lon.shape}")
        terrain = np.asarray(terrain_height_m,
                             dtype=np.float32).reshape(-1)
        if terrain.shape != (ncol,):
            raise ValueError(
                f"terrain_height_m must have shape ({ncol},), got "
                f"{terrain.shape}")
        dx_column = None
        if dx_column_m is not None:
            dx_column = np.asarray(dx_column_m,
                                   dtype=np.float32).reshape(-1)
            if dx_column.shape != (ncol,):
                raise ValueError(
                    f"dx_column_m must have one value per column "
                    f"({ncol},), got {dx_column.shape}")
            if not np.all(np.isfinite(dx_column)) or np.any(
                    dx_column <= 0.0):
                raise ValueError(
                    "dx_column_m must be finite and positive")
        top_nominal = float(np.asarray(
            z_interface_nominal_m, dtype=np.float64)[-1])

        from gpuwm.config import RunConfig
        from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY
        cfg = RunConfig(
            nx=ncol, ny=1, nz=nz, dx=dx, dy=dx,
            ztop=top_nominal, dt=float(cadences["dt"]),
            run_seconds=0.0, moist=True, mp_physics=6,
            wsm6_hail_opt=int(wsm6_hail_opt),
            ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
            sf_sfclay_physics=1, sf_surface_physics=4,
            bl_pbl_physics=1,
            cu_physics=cadences["cu_physics"],
            ishallow=int(gf_ishallow),
            radt=cadences["radt_minutes"],
            radt_minutes=cadences["radt_minutes"],
            bldt=cadences["bldt_minutes"],
            cudt_minutes=cadences["cudt_minutes"])
        self._cfg = cfg
        self._cadences = cadences
        self._dt = float(cadences["dt"])
        self._nz, self._ncol = nz, ncol
        self._start_time = start_time
        self._identity = {
            "schema": RESTART_SCHEMA,
            "n_levels": nz, "n_columns": ncol, "dt": self._dt,
            "stepra": cadences["stepra"], "stepbl": cadences["stepbl"],
            "stepcu": cadences["stepcu"],
            "cumulus_scheme": cumulus_scheme,
            "start_time": start_time.isoformat(),
            "p_top_pa": p_top, "dx_m": dx,
            "gf_ishallow": int(gf_ishallow),
            "gf_dx_column": ("per-column" if dx_column is not None
                             else "scalar"),
            "wsm6_hail_opt": int(wsm6_hail_opt),
            # The surface classification identity: the threshold that
            # partitions sea ice from land, and whether XLAND came from
            # the caller (native, wins verbatim) or the landmask
            # derivation.  Both change which columns which surface owns,
            # so a restart across either is a different trajectory.
            "xice_threshold": threshold,
            "xland_source": ("native" if xland is not None
                             else "derived-from-landmask"),
        }

        # ---- device construction from here on --------------------------
        import cupy as cp
        self._cp = cp

        state = _ColumnBatchState(cfg, fnm, fnp, p_top, terrain, cp)
        self._state = state
        #: The species arrays the state was born with; phase 2 rebinds
        #: the state onto caller views and this map is how it comes back.
        self._species_buffers = {
            name: getattr(state, name) for name in _SPECIES}

        def surface(values):
            if np.ndim(values) == 0:
                return values
            if hasattr(values, "__cuda_array_interface__"):
                return values.reshape(1, ncol)
            return np.asarray(values).reshape(1, ncol)

        def soil(values):
            if np.ndim(values) == 0:
                return values
            if hasattr(values, "__cuda_array_interface__"):
                return values.reshape(values.shape[0], 1, ncol)
            values = np.asarray(values)
            return values.reshape(values.shape[0], 1, ncol)

        self._driver = initialize_physics(
            state, cfg,
            landmask=surface(landmask),
            xland=None if xland is None else surface(xland),
            xice_threshold=threshold,
            tsk=surface(tsk),
            soil_temperature=soil(soil_temperature),
            soil_moisture=soil(soil_moisture),
            ivgtyp=surface(ivgtyp), isltyp=surface(isltyp),
            vegfra=surface(vegfra), tmn=surface(tmn),
            xice=surface(xice), snow=surface(snow),
            snow_depth=surface(snow_depth),
            glw=None,
            radiation_start_time=start_time,
            radiation_latitude=lat.reshape(1, ncol),
            radiation_longitude=lon.reshape(1, ncol))

        # Persistent input/atmosphere buffers ((nz[+1], 1, ncol)).
        shape3 = (nz, 1, ncol)
        iface3 = (nz + 1, 1, ncol)
        # GF dynamics forcing lanes: persistent seam-owned buffers the
        # driver's GF adapter reads permanently.  Zero until a caller
        # supplies rates through run_phase1 -- which is also native
        # MPAS's t=0 tend_physics state.  The PBL lanes are retained by
        # the driver's own _run_ysu; per-column dx is a sealed static.
        self._gf_rthdynten = cp.zeros(shape3, dtype=cp.float32)
        self._gf_rqvdynten = cp.zeros(shape3, dtype=cp.float32)
        self._driver.gf_rthdynten = self._gf_rthdynten
        self._driver.gf_rqvdynten = self._gf_rqvdynten
        if dx_column is not None:
            self._driver.gf_dx_column = cp.asarray(
                dx_column.reshape(1, ncol))
        self._in = {name: cp.zeros(shape3, dtype=cp.float32)
                    for name in ("u", "v", "theta", "temperature",
                                 "pressure", "exner", "dz", "rho")}
        self._in.update({name: cp.zeros(iface3, dtype=cp.float32)
                         for name in ("p_interface", "z_interface")})
        self._atmosphere = None
        state.prepared_physics_atmosphere = self._supplied_atmosphere

        # Persistent raw-output buffers ([nz, ncol]); serialized so held
        # du/dv survive a restore (module docstring, RESTART LAW).
        self._out = {name: cp.zeros((nz, ncol), dtype=cp.float32)
                     for name in _OUTPUT_BUFFERS}

        # Exact bookkeeping: integer step count, float64 seconds.
        self._step_index = 0
        self._pending_phase = _PHASE1
        state.elapsed_seconds = 0.0

    # ------------------------------------------------------------------
    def _supplied_atmosphere(self):
        if self._atmosphere is None:
            raise RuntimeError(
                "the column-batch atmosphere is built by run_phase1; the "
                "driver must not run outside it")
        return self._atmosphere

    def _require_input(self, name, value, shape):
        cp = self._cp
        if not isinstance(value, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy ndarray")
        if value.dtype != cp.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape} [level, column], got "
                f"{value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return value

    def _rebind_species_buffers(self):
        """Point the state carriers back at persistent seam memory.

        The next ``run_phase1`` overwrites them; keeping caller views
        bound between calls would let a driver code path read freed or
        repurposed caller memory.
        """
        state = self._state
        for name, buffer in self._species_buffers.items():
            setattr(state, name, buffer)
        state.thp = self._in["theta"]
        state.p = self._in["pressure"]

    # ------------------------------------------------------------------
    def run_phase1(self, *, dt, u, v, theta, pressure, pressure_interface,
                   z_interface, w, rho_dry, qv, qc, qr, qi, qs, qg,
                   exner=None, rthdynten=None,
                   rqvdynten=None) -> MpasColumnPhysicsTendencies:
        """Run the due phase-1 chain; return the held raw tendency set.

        Read-only guarantee: every input array is byte-identical after
        this call (negative qv included -- the clamp happens in seam
        scratch).  ``dt`` must equal the constructor model step.

        ``rthdynten``/``rqvdynten`` ([nz, ncol] float32, K/s dry theta
        and kg/kg/s) are the caller's dynamics forcing for GF's
        RTHFTEN/RQVFTEN lanes -- native MPAS v8.4.1's rthdynten/rqvdynten
        construction, computed at the END of the caller's PREVIOUS
        dynamics step exactly as native consumes them.  ``None`` zeroes
        the lane (native's own tend_physics start state).
        """
        if self._pending_phase != _PHASE1:
            raise RuntimeError(
                "run_phase1 called out of order: the previous step's "
                "run_phase2 has not run (strict phase1 -> phase2 "
                "alternation)")
        if float(dt) != self._dt:
            raise ValueError(
                f"phase-1 dt={dt} does not match the constructor model "
                f"step {self._dt}; the seam owns the sub-cadences")
        cp = self._cp
        nz, ncol = self._nz, self._ncol
        level, iface = (nz, ncol), (nz + 1, ncol)
        inputs = {"u": u, "v": v, "theta": theta, "pressure": pressure,
                  "rho_dry": rho_dry, "qv": qv, "qc": qc, "qr": qr,
                  "qi": qi, "qs": qs, "qg": qg}
        for name in _LEVEL_FIELDS_P1:
            self._require_input(name, inputs[name], level)
        for name, value in (("pressure_interface", pressure_interface),
                            ("z_interface", z_interface), ("w", w)):
            self._require_input(name, value, iface)
        if exner is not None:
            self._require_input("exner", exner, level)
        for name, value in (("rthdynten", rthdynten),
                            ("rqvdynten", rqvdynten)):
            if value is not None:
                self._require_input(name, value, level)

        state, buf = self._state, self._in
        self._rebind_species_buffers()

        def v3(a):
            return a.reshape(a.shape[0], 1, ncol)

        # One copy per input into seam-owned buffers (stated copies).
        buf["u"][...] = v3(u)
        buf["v"][...] = v3(v)
        buf["theta"][...] = v3(theta)
        buf["pressure"][...] = v3(pressure)
        buf["p_interface"][...] = v3(pressure_interface)
        buf["z_interface"][...] = v3(z_interface)
        state.w[...] = v3(w)
        # NEGATIVE-QV LAW: clamp in the seam's scratch copy, exactly as
        # MPAS bulk physics applies max(qv, 0) before its schemes.  The
        # caller's array is never touched.
        cp.maximum(v3(qv), cp.float32(0.0), out=state.qv)
        for name in ("qc", "qr", "qi", "qs", "qg"):
            getattr(state, name)[...] = v3(inputs[name])
        # The same derivations _prepare_atmosphere applies, on the
        # caller's own fields: exner from the EOS pressure,
        # T = theta*exner, physics density (1+qv)*rho_dry for WRF
        # phy_prep's (1+qv)/alt.
        if exner is None:
            cp.power(buf["pressure"] / cp.float32(_constants.P0),
                     cp.float32(_constants.RCP), out=buf["exner"])
        else:
            buf["exner"][...] = v3(exner)
        cp.multiply(buf["theta"], buf["exner"], out=buf["temperature"])
        cp.subtract(buf["z_interface"][1:], buf["z_interface"][:-1],
                    out=buf["dz"])
        cp.multiply(cp.float32(1.0) + state.qv, v3(rho_dry),
                    out=buf["rho"])
        # GF dynamics forcing lanes: refresh the seam-owned persistent
        # buffers the driver's GF adapter reads.  A None lane zeroes --
        # both the pre-upgrade behaviour and native's t=0 state.
        if rthdynten is None:
            self._gf_rthdynten[...] = cp.float32(0.0)
        else:
            self._gf_rthdynten[...] = v3(rthdynten)
        if rqvdynten is None:
            self._gf_rqvdynten[...] = cp.float32(0.0)
        else:
            self._gf_rqvdynten[...] = v3(rqvdynten)

        self._atmosphere = {
            "theta": buf["theta"], "temperature": buf["temperature"],
            "pressure": buf["pressure"],
            "p_interface": buf["p_interface"], "exner": buf["exner"],
            "u": buf["u"], "v": buf["v"],
            "z_interface": buf["z_interface"], "dz": buf["dz"],
            "qv": state.qv, "qc": state.qc, "qi": state.qi,
            "qs": state.qs, "rho": buf["rho"],
        }
        state.elapsed_seconds = float(
            np.float64(self._step_index) * np.float64(self._dt))
        driver, cfg = self._driver, self._cfg
        before = dict(driver.call_counts)
        try:
            # THE ARW ORCHESTRATION, not a wrapper: PhysicsDriver.compute
            # runs radiation/surface/LSM/PBL/cumulus on their own
            # cadences, holds non-due rates, advances the KF clock.
            tendencies = type(self)._PHASE1_ORCHESTRATION(
                driver, state, cfg)
        finally:
            self._atmosphere = None
        out = self._out
        # Identity mass coupling makes the held/composed stacks the raw
        # rates (module docstring); one copy into the stable output set.
        out["dtheta"][...] = tendencies.rtheta.reshape(nz, ncol)
        out["dqv"][...] = tendencies.rqv.reshape(nz, ncol)
        out["dqc"][...] = tendencies.rqc.reshape(nz, ncol)
        for name, value in (("dqr", tendencies.rqr),
                            ("dqi", tendencies.rqi),
                            ("dqs", tendencies.rqs)):
            if value is None:
                out[name][...] = cp.float32(0.0)
            else:
                out[name][...] = value.reshape(nz, ncol)
        # dqg stays zero: no phase-1 scheme produces a graupel rate.
        last_ysu = driver.last_ysu
        if last_ysu is not None:
            out["du"][...] = last_ysu["du"].reshape(nz, ncol)
            out["dv"][...] = last_ysu["dv"].reshape(nz, ncol)
        # else: keep the held buffers (non-due PBL step after a restore).
        out["h_diabatic"][...] = state.h_diabatic.reshape(nz, ncol)

        after = driver.call_counts
        result = MpasColumnPhysicsTendencies(
            du=out["du"], dv=out["dv"], dtheta=out["dtheta"],
            dqv=out["dqv"], dqc=out["dqc"], dqr=out["dqr"],
            dqi=out["dqi"], dqs=out["dqs"], dqg=out["dqg"],
            h_diabatic=out["h_diabatic"],
            step_index=self._step_index,
            elapsed_seconds=state.elapsed_seconds,
            radiation_ran=after["radiation"] > before["radiation"],
            surface_pbl_ran=after["ysu"] > before["ysu"],
            cumulus_ran=after["cumulus"] > before["cumulus"])
        self._pending_phase = _PHASE2
        return result

    # ------------------------------------------------------------------
    def run_phase2(self, *, theta, qv, qc, qr, qi, qs, qg, pressure,
                   rho_dry, z_interface, refl_10cm_due: bool = False) -> dict:
        """Run WSM6 in place on the caller's post-transport arrays.

        ``theta`` and all six species are updated IN the caller's memory
        (zero-copy views); ``pressure``/``rho_dry``/``z_interface`` are
        read-only.  The caller's integrator clamps scalar state before
        this call (pinned contract), so no clamp is applied here.
        Returns per-call surface diagnostics as ``[column]`` copies:
        rainncv/snowncv/graupelncv (mm) and sr.

        ``refl_10cm_due`` is WRF's history-step ``diagflag``: when True the
        scheme adapter computes REFL_10CM inside the same microphysics call
        from its post-call temperature and the unchanged prepared pressure
        (gpuwm/core/refl.py, PROVENANCE.md D2 -- exactly where WRF and native
        MPAS-A compute ``refl10cm``), and the receipt gains ``refl_10cm`` as
        a seam-owned ``(nz, column)`` device copy.  Computing it after this
        call instead would pair post-scheme temperature with re-derived
        post-scheme pressure, which is a different field than WRF defines.
        """
        if self._pending_phase != _PHASE2:
            raise RuntimeError(
                "run_phase2 called out of order: run_phase1 must run "
                "first (strict phase1 -> phase2 alternation)")
        cp = self._cp
        nz, ncol = self._nz, self._ncol
        level, iface = (nz, ncol), (nz + 1, ncol)
        for name, value in (("theta", theta), ("qv", qv), ("qc", qc),
                            ("qr", qr), ("qi", qi), ("qs", qs),
                            ("qg", qg), ("pressure", pressure),
                            ("rho_dry", rho_dry)):
            self._require_input(name, value, level)
        self._require_input("z_interface", z_interface, iface)

        state, cfg, driver = self._state, self._cfg, self._driver

        def v3(a):
            return a.reshape(a.shape[0], 1, ncol)

        # Zero-copy in-place binding: thb == 0, so thp IS full theta.
        state.thp = v3(theta)
        for name, value in (("qv", qv), ("qc", qc), ("qr", qr),
                            ("qi", qi), ("qs", qs), ("qg", qg)):
            setattr(state, name, v3(value))
        state.p = v3(pressure)
        # Documented 1-ULP round trips: WSM6's adapter derives
        # rho = 1/alt and z = (phb + php)/g (gpuwm/core/wsm6.py:109-112),
        # so alt = 1/rho_dry and php = z_interface*g here.
        alt = state.scratch((nz, 1, ncol), "physics_column_alt")
        cp.divide(cp.float32(1.0), v3(rho_dry), out=alt)
        state.alt = alt
        php = state.scratch((nz + 1, 1, ncol), "physics_column_php")
        cp.multiply(v3(z_interface), cp.float32(_constants.G), out=php)
        state.php = php
        try:
            # THE ARW POST-RK PAIR (gpuwm/core/dycore.py:2507-2510):
            # microphysics dispatch, then the driver's bucket/RAINBL
            # acceptance.
            diagnostics = type(self)._PHASE2_MICROPHYSICS(
                state, cfg, self._dt, refl_10cm_due=refl_10cm_due)
            driver.accept_microphysics(diagnostics)
            receipt = {
                "rainncv": diagnostics.rainncv.reshape(ncol).copy(),
                "snowncv": diagnostics.snowncv.reshape(ncol).copy(),
                "graupelncv":
                    diagnostics.graupelncv.reshape(ncol).copy(),
                "sr": diagnostics.sr.reshape(ncol).copy(),
            }
            if refl_10cm_due:
                # Consume the one-frame D2 handoff here, inside the same
                # transaction: an unconsumed stash is a cadence bug the
                # next due call would refuse loudly.
                from gpuwm.core.refl import consume_refl_10cm
                receipt["refl_10cm"] = (
                    consume_refl_10cm(state).reshape(nz, ncol).copy())
        finally:
            # Re-anchor the state carriers on seam-owned memory so no
            # later call can reach the caller's arrays by accident.
            self._rebind_species_buffers()
        # ARW step epilogue (gpuwm/core/dycore.py:2512): the clock
        # advances once per completed model step, exact as steps * dt.
        self._step_index += 1
        state.elapsed_seconds = float(
            np.float64(self._step_index) * np.float64(self._dt))
        self._pending_phase = _PHASE1
        return receipt

    # ------------------------------------------------------------------
    def accumulated_precipitation(self) -> dict:
        """Host-visible bucket copies: RAINNC/SNOWNC/GRAUPELNC/RAINC (mm).

        The buckets accumulate in the seam across phase-2 calls (the
        RAINNC family) and phase-1 cumulus steps (RAINC); a batch with
        no cumulus scheme reports RAINC as zeros, WRF's own convention.
        """
        cp = self._cp
        ncol = self._ncol
        state, driver = self._state, self._driver
        out = {}
        for name, slot in (("RAINNC", "mp_rainnc"),
                           ("SNOWNC", "mp_snownc"),
                           ("GRAUPELNC", "mp_graupelnc")):
            out[name] = state.scratch(
                (1, ncol), slot).reshape(ncol).copy()
        rainc = driver.rainc
        out["RAINC"] = (cp.zeros(ncol, dtype=cp.float32)
                        if rainc is None else rainc.reshape(ncol).copy())
        return out

    @property
    def call_counts(self) -> dict:
        """Copy of the driver's per-scheme call counters."""
        return dict(self._driver.call_counts)

    @property
    def surface_classification(self) -> dict:
        """The construction-time classification receipt: which source
        decided XLAND (``native`` wins verbatim over the landmask
        derivation) and the per-class column counts under the active
        ``xice_threshold`` (sea ice, open water, NOAHMP_SFLX land,
        NOAHMP_GLACIER land)."""
        return dict(self._driver.surface_classification)

    @property
    def last_noahmp_census(self):
        """The most recent Noah-MP step census (or None before one):
        per-class column counts plus ``glacier_path``, the execution
        provenance of the glacier dispatch when glacier columns ran."""
        census = getattr(self._driver, "last_noahmp_census", None)
        return None if census is None else dict(census)

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def elapsed_seconds(self) -> float:
        return self._state.elapsed_seconds

    # ------------------------------------------------------------------
    def _restart_manifest(self):
        """Live name -> device array map of every persisted item.

        Reuses the mainline restart classification: driver arrays via
        ``gpuwm.io.restart._driver_manifest`` (which also ENFORCES that
        every driver attribute is classified), serialized scratch
        buckets via ``classify_scratch_slot``, and the seam's own held
        outputs and carriers on top.
        """
        from gpuwm.io.restart import (_driver_manifest,
                                      classify_scratch_slot)
        driver, state = self._driver, self._state
        manifest = dict(_driver_manifest(driver))
        for slot in sorted(state._scratch):
            if classify_scratch_slot(slot) == "serialize":
                manifest[f"scratch/{slot}"] = state._scratch[slot]
        for name in ("effc", "effi", "effs", "h_diabatic"):
            manifest[f"state/{name}"] = getattr(state, name)
        for name in _OUTPUT_BUFFERS:
            manifest[f"seam/{name}"] = self._out[name]
        return manifest

    def export_state(self) -> dict:
        """Host-side payload of every persisted item (step boundary only)."""
        if self._pending_phase != _PHASE1:
            raise RuntimeError(
                "export_state is legal only at a step boundary (after "
                "run_phase2, before the next run_phase1)")
        cp = self._cp
        driver = self._driver
        arrays = {key: cp.asnumpy(value)
                  for key, value in self._restart_manifest().items()}
        kf_time = getattr(driver.cumulus_callable, "_history_time", None)
        return {
            "identity": dict(self._identity),
            "arrays": arrays,
            "scalars": {
                "step_index": self._step_index,
                "elapsed_seconds": self._state.elapsed_seconds,
                "call_counts": dict(driver.call_counts),
                "microphysics_updates":
                    int(driver.microphysics_updates),
                "ysu_nan_guard_fires": int(driver.ysu_nan_guard_fires),
                "kf_history_time": (None if kf_time is None
                                    else float(kf_time)),
            },
        }

    def restore_state(self, payload: dict) -> None:
        """Restore an ``export_state`` payload in place.

        Refuses identity mismatches, unknown keys and missing keys; a
        restored seam continues bit-identically (GPU-gated test).
        """
        if self._pending_phase != _PHASE1:
            raise RuntimeError(
                "restore_state is legal only at a step boundary")
        expected = {"identity", "arrays", "scalars"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError(
                "restart payload must carry exactly "
                f"{sorted(expected)}, got "
                f"{sorted(payload) if isinstance(payload, dict) else payload!r}")
        if payload["identity"] != self._identity:
            raise ValueError(
                "restart payload identity does not match this seam's "
                f"configuration: {payload['identity']} != "
                f"{self._identity}")
        cp = self._cp
        driver, state = self._driver, self._state
        live = self._restart_manifest()
        stored = dict(payload["arrays"])
        # The two lazily-born carriers may legitimately exist in the
        # payload but not yet on a fresh seam; every OTHER asymmetry is
        # a coverage error.
        lazy = {}
        for key in _LAZY_RESTART_KEYS:
            if key in stored and key not in live:
                lazy[key] = stored.pop(key)
            elif key in live and key not in stored:
                raise ValueError(
                    f"restart payload lacks {key!r} but this seam "
                    "already carries live state for it; restore into a "
                    "freshly constructed seam")
        missing = sorted(set(live) - set(stored))
        extra = sorted(set(stored) - set(live))
        if missing or extra:
            raise ValueError(
                "restart payload arrays do not cover this seam "
                f"(missing {missing}, extra {extra})")
        for key, target in live.items():
            source = np.asarray(stored[key])
            if source.shape != target.shape:
                raise ValueError(
                    f"restart array {key!r} has shape {source.shape}, "
                    f"live {target.shape}")
            target[...] = cp.asarray(source, dtype=target.dtype)
        if "cumulus/w0avg" in lazy:
            adapter = driver.cumulus_callable
            adapter.w0avg = cp.asarray(lazy["cumulus/w0avg"],
                                       dtype=cp.float32)
            adapter._history_state = state
        if "radiation/o33d_grid" in lazy:
            driver.radiation_callable._o33d_grid = np.asarray(
                lazy["radiation/o33d_grid"])
        scalars = payload["scalars"]
        self._step_index = int(scalars["step_index"])
        state.elapsed_seconds = float(scalars["elapsed_seconds"])
        driver.call_counts.clear()
        driver.call_counts.update(
            {key: int(value)
             for key, value in scalars["call_counts"].items()})
        driver.microphysics_updates = int(
            scalars["microphysics_updates"])
        driver.ysu_nan_guard_fires = int(scalars["ysu_nan_guard_fires"])
        adapter = driver.cumulus_callable
        if adapter is not None and hasattr(adapter, "_history_time"):
            adapter._history_time = scalars["kf_history_time"]
        self._pending_phase = _PHASE1


def run_mpas_column_batch(**kwargs) -> MpasColumnBatchPhysics:
    """Construct the persistent MPAS column-batch physics seam.

    The published API entry (also exported as
    ``gpuwm.core.physics.run_mpas_column_batch``).  Keyword-only; an
    unknown or misspelled keyword refuses with its name.  Returns the
    persistent seam whose ``run_phase1``/``run_phase2`` methods drive
    the two per-step hooks; see :class:`MpasColumnBatchPhysics`.
    """
    return MpasColumnBatchPhysics(**kwargs)
