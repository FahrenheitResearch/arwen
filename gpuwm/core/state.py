"""Forecast/setup state container with FP32 arrays on one explicit backend.

In its default CuPy forecast mode, ``DomainState`` owns *all* persistent
device memory. Native stock-WRF setup/export may explicitly use NumPy host
arrays instead. Everything is allocated
once in ``__init__`` (shapes depend only on ``RunConfig``); base-state and
coordinate arrays are filled in place by ``load_base``.  Nothing else in the
model may allocate persistent device arrays — transient work buffers must go
through :meth:`DomainState.scratch`.

Array layout is ``(nz, ny, nx)`` with x fastest.  Staggering (WRF-ARW):
``u (nz,ny,nx+1)``, ``v (nz,ny+1,nx)``, ``w``/``phi (nz+1,ny,nx)``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import threading

import numpy as np

try:  # Native stock-WRF setup/export has a genuine NumPy-only route.
    import cupy as cp
except ImportError:  # pragma: no cover - exercised in an isolated subprocess
    cp = None

from gpuwm.config import SASE_PBL_SCHEME, RunConfig
from gpuwm.core import constants as c
from gpuwm.core.grid import BaseState, VerticalCoord, rebalance_hydrostatic
# Single source for the SASE realizability floor (the e_sgs cold-start
# fill).  From gpuwm.core, not gpuwm.verify: the standalone CPU
# preprocessing distribution omits the verification tree.
from gpuwm.core.sase_limits import E_MIN as SASE_E_MIN

#: Model-field dtype.  FP64 only in setup code and test references.  Keeping
#: the scalar type available without CuPy lets the Rust/NumPy native-input
#: path import on a CPU-only host; CUDA forecast allocation still fails with
#: the explicit optional-dependency error below.
DTYPE = np.float32 if cp is None else cp.float32


def _require_cupy():
    """Return CuPy or fail with the exact missing optional dependency."""

    if cp is None:
        raise RuntimeError(
            "CuPy is required for CUDA forecast state; install gpuwm[gpu] "
            "or select the RW-WPS CPU preprocessing backend"
        )
    return cp


class SharedDycoreStateWorkspace:
    """Per-symbol max-domain storage for restart-rebuilt state arrays.

    The symbol inventory is supplied by preflight's view of the restart
    manifest, so this allocation mechanism owns no second hand-copied list.
    Each :class:`DomainState` receives a C-contiguous shaped prefix of the
    corresponding symbol backing.  The non-blocking ownership token guards
    the executor's one-STEP-or-FORCE-at-a-time correctness contract.
    """

    def __init__(self, symbol_shapes: Mapping[str, tuple[int, ...]]):
        cuda = _require_cupy()
        self._symbol_shapes: dict[str, tuple[int, ...]] = {}
        self._buffers: dict[str, cp.ndarray] = {}
        for symbol in sorted(symbol_shapes):
            shape = tuple(int(extent) for extent in symbol_shapes[symbol])
            if not shape or any(extent < 1 for extent in shape):
                raise ValueError(
                    f"invalid shared dycore-state shape for {symbol!r}: "
                    f"{shape}")
            self._symbol_shapes[symbol] = shape
            self._buffers[symbol] = cuda.zeros(shape, dtype=DTYPE)
        if not self._buffers:
            raise ValueError("shared dycore-state workspace has no symbols")
        self._owner_lock = threading.Lock()
        self._owner = None

    def view(self, symbol: str, shape, dtype=None) -> cp.ndarray:
        """Return one C-contiguous shaped prefix of ``symbol``'s backing."""
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        shape = tuple(int(extent) for extent in shape)
        requested_dtype = np.dtype(DTYPE if dtype is None else dtype)
        if requested_dtype != np.dtype(DTYPE):
            raise TypeError(
                f"shared dycore-state symbol {symbol!r} is float32, "
                f"requested {requested_dtype}")
        try:
            backing = self._buffers[symbol]
        except KeyError as exc:
            raise KeyError(
                f"rebuilt state symbol {symbol!r} is not in this workspace") \
                from exc
        requested = math.prod(shape)
        if requested > backing.size:
            raise ValueError(
                f"shared dycore-state symbol {symbol!r} capacity is "
                f"{backing.size} values ({self._symbol_shapes[symbol]}), "
                f"requested {requested} values ({shape})")
        return backing.reshape(-1)[:requested].reshape(shape)

    def backing(self, symbol: str) -> cp.ndarray:
        """Return a symbol backing for identity/debug inspection only."""
        return self._buffers[symbol]

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(self._buffers)

    @property
    def symbol_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._symbol_shapes)

    @property
    def nbytes(self) -> int:
        return sum(int(buf.nbytes) for buf in self._buffers.values())

    @property
    def owner(self):
        """Current executor ownership token, or ``None`` between turns."""
        return self._owner

    @contextmanager
    def acquire(self, owner):
        """Fail loudly if another STEP/FORCE still owns the workspace."""
        if not self._owner_lock.acquire(blocking=False):
            raise RuntimeError(
                f"shared dycore-state workspace is already owned by "
                f"{self._owner!r}; {owner!r} cannot observe it concurrently")
        self._owner = owner
        try:
            yield self
        finally:
            self._owner = None
            self._owner_lock.release()


class ScratchArena:
    """Shared zero-filled backing for proven step-local scratch slots.

    One backing allocation is retained per independent slot.  Audited slots
    with disjoint lifetimes may alias a larger backing allocation.  A domain
    receives a contiguous prefix reshaped to its requested dimensions, so
    differently sized domains can reuse the same allocation while the flat
    schedule steps exactly one domain at a time.  The lifetime classification
    and alias proof live beside the enforced registry in
    :mod:`gpuwm.core.preflight`; this class is deliberately only an
    allocation/view mechanism.
    """

    def __init__(self, slot_shapes: Mapping[str, tuple[int, ...]], *,
                 slot_aliases: Mapping[str, str] | None = None):
        cuda = _require_cupy()
        self._slot_shapes: dict[str, tuple[int, ...]] = {}
        self._buffers: dict[str, cp.ndarray] = {}
        aliases = dict(slot_aliases or {})
        for slot, raw_shape in slot_shapes.items():
            shape = tuple(int(extent) for extent in raw_shape)
            if not shape or any(extent < 0 for extent in shape):
                raise ValueError(f"invalid scratch-arena shape for {slot!r}: "
                                 f"{shape}")
            self._slot_shapes[slot] = shape
            if slot not in aliases:
                # DomainState.scratch has always zero-allocated every slot.
                # The arena preserves that initialization exactly; the audit
                # admits only slots overwritten before their first read.
                self._buffers[slot] = cuda.zeros(shape, dtype=DTYPE)
        for slot, target in aliases.items():
            if slot not in self._slot_shapes:
                raise KeyError(f"scratch-arena alias {slot!r} has no shape")
            if target not in self._buffers:
                raise KeyError(
                    f"scratch-arena alias target {target!r} is unavailable")
            requested = math.prod(self._slot_shapes[slot])
            if requested > self._buffers[target].size:
                raise ValueError(
                    f"scratch-arena alias {slot!r} needs {requested} values, "
                    f"but target {target!r} has {self._buffers[target].size}")
            self._buffers[slot] = self._buffers[target]

    def has_slot(self, slot: str) -> bool:
        return slot in self._buffers

    def view(self, shape, slot: str, dtype=None) -> cp.ndarray:
        """Return a shaped prefix view of the slot's max-sized backing."""
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        shape = tuple(int(extent) for extent in shape)
        requested_dtype = np.dtype(DTYPE if dtype is None else dtype)
        if requested_dtype != np.dtype(DTYPE):
            raise TypeError(
                f"scratch arena slot {slot!r} is float32, requested "
                f"{requested_dtype}")
        try:
            backing = self._buffers[slot]
        except KeyError as exc:
            raise KeyError(f"scratch slot {slot!r} is not in this arena") \
                from exc
        requested = math.prod(shape)
        if requested > backing.size:
            raise ValueError(
                f"scratch slot {slot!r} arena capacity is {backing.size} "
                f"values ({self._slot_shapes[slot]}), requested {requested} "
                f"values ({shape})")
        return backing.reshape(-1)[:requested].reshape(shape)

    @property
    def slot_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._slot_shapes)

    @property
    def nbytes(self) -> int:
        unique = {id(buf): buf for buf in self._buffers.values()}
        return sum(int(buf.nbytes) for buf in unique.values())

    def poison(self) -> None:
        """Fill every unique arena backing with NaNs.

        This is the Task-14 debug lever for proving the lifetime audit at
        runtime.  The model executor calls it only when explicitly requested
        between complete domain turns; production runs leave it disabled.
        Every arena-admitted slot is write-before-read by construction, so a
        surviving NaN identifies an invalid lifetime classification quickly.
        """
        unique = {id(buf): buf for buf in self._buffers.values()}
        for buf in unique.values():
            buf.fill(DTYPE(np.nan))


def build_shared_scratch_arena(domains: Iterable[object]) -> ScratchArena:
    """Build the deterministic shared arena for a domain configuration set.

    This is the Task-14 handoff: ``build_experiment`` passes its parent-first
    ``DomainConfig`` sequence here, then injects the returned arena into every
    ``DomainState``. Shape selection and lifetime admission share the same
    registry used by preflight, and this function does not mutate the domains.
    """
    from gpuwm.core.preflight import (shared_scratch_arena_aliases,
                                      shared_scratch_arena_shapes)

    domain_tuple = tuple(domains)
    return ScratchArena(
        shared_scratch_arena_shapes(domain_tuple),
        slot_aliases=shared_scratch_arena_aliases(domain_tuple))


def build_shared_dycore_state_workspace(
        domains: Iterable[object]) -> SharedDycoreStateWorkspace:
    """Allocate one maximum backing for each restart-REBUILT state symbol."""
    from gpuwm.core.preflight import shared_dycore_state_workspace_shapes

    domain_tuple = tuple(domains)
    return SharedDycoreStateWorkspace(
        shared_dycore_state_workspace_shapes(domain_tuple))


def refresh_model_time(state, clock, *, kernel_launch: bool = False,
                       after_step: bool = False) -> None:
    """Refresh the legacy state mirror from the integer DomainClock.

    Calendar authority remains ``clock.ticks``.  At solve entry WRF-facing
    consumers receive the REAL ``curr_secs`` image; after solve the public
    compatibility mirror is refreshed from the exact next tick.  Keeping the
    assignment outside ``model.py`` also leaves T9's executor AST audit
    mechanically strict: the schedule walker never assigns an elapsed value.
    """
    if kernel_launch and after_step:
        raise ValueError("kernel_launch and after_step are mutually exclusive")
    if kernel_launch:
        value = float(clock.elapsed_seconds_fp32)
    else:
        ticks = clock.ticks + (clock.spec.step_ticks if after_step else 0)
        value = ticks / clock.tick_den
    state.elapsed_seconds = value


def _height_half_from_phb(phb: np.ndarray) -> np.ndarray:
    """Return the host half-level heights using the established FP64 tree."""
    return 0.5 * (phb[:-1] + phb[1:]) / c.G


def _array_module_for(value):
    """Return NumPy for host setup arrays, otherwise the CUDA array module."""
    if isinstance(value, np.ndarray):
        return np
    return _require_cupy()


def _state_array_module(state):
    """Honor only an explicitly requested host setup state.

    CPU-only unit tests historically monkeypatch this module's ``cp`` symbol
    to NumPy as a CUDA emulator.  Array-type inspection cannot distinguish
    that shim from a real host state, so the constructor records the explicit
    setup choice instead.
    """
    if getattr(state, "_host_setup_state", False):
        return np
    return _require_cupy()


def mu_at_u_faces(mu: cp.ndarray) -> cp.ndarray:
    """Column mass ``(ny, nx)`` averaged to u faces ``(ny, nx+1)``, periodic
    in x; face f lies between cells f-1 and f, face nx duplicates face 0.

    The single sanctioned face-averaging helper (Task 4 consolidated the
    copies that lived in dycore, advection, and diffusion).
    """
    xp = _array_module_for(mu)
    mux = 0.5 * (mu + xp.roll(mu, 1, axis=1))
    return xp.concatenate([mux, mux[:, :1]], axis=1)


def mu_at_v_faces(mu: cp.ndarray) -> cp.ndarray:
    """Column mass ``(ny, nx)`` averaged to v faces ``(ny+1, nx)``, periodic
    in y; face f lies between rows f-1 and f, row ny duplicates row 0."""
    xp = _array_module_for(mu)
    muy = 0.5 * (mu + xp.roll(mu, 1, axis=0))
    return xp.concatenate([muy, muy[:1, :]], axis=0)


class DomainState:
    """All prognostic, diagnostic, and reference arrays for one domain.

    ``array_module`` is an internal setup/export seam. The default remains
    CuPy; NumPy states are not valid inputs to the CUDA forecast integrator.
    """

    def __init__(self, cfg: RunConfig,
                 scratch_arena: ScratchArena | None = None,
                 dycore_state_workspace: SharedDycoreStateWorkspace | None =
                 None, *, array_module=None):
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        xp = _require_cupy() if array_module is None else array_module
        if xp is not np and xp is not cp:
            raise TypeError("array_module must be numpy or cupy")
        self._host_setup_state = array_module is np
        if self._host_setup_state and (scratch_arena is not None
                                       or dycore_state_workspace is not None):
            raise ValueError(
                "NumPy setup states cannot use CUDA scratch/dycore workspaces")

        def zeros(*shape):
            return xp.zeros(shape, dtype=np.float32)

        def rebuilt(symbol, *shape):
            if dycore_state_workspace is None:
                return zeros(*shape)
            return dycore_state_workspace.view(symbol, shape, DTYPE)

        # Prognostic fields (perturbation form).
        self.u = zeros(nz, ny, nx + 1)
        self.v = zeros(nz, ny + 1, nx)
        self.w = zeros(nz + 1, ny, nx)
        self.thp = zeros(nz, ny, nx)        # theta' = theta - thb
        self.php = zeros(nz + 1, ny, nx)    # phi'
        self.mup = zeros(ny, nx)            # mu'

        # Diagnostic fields.
        self.p = zeros(nz, ny, nx)          # full pressure
        self.al = zeros(nz, ny, nx)         # alpha'_d
        self.alt = zeros(nz, ny, nx)        # total alpha_d

        # Moisture mixing ratios (per unit dry mass) plus RK time-t copies.
        # The Phase-2 qv/qc/qr allocation is unchanged.  WSM6's ice mass
        # fields are allocated for mp=6/8; Thompson adds rain/ice number
        # moments for mp=8 and Morrison adds its four transported number
        # moments for mp=10.  Every
        # frozen dry/Kessler state retains its original storage and loops.
        if cfg.moist:
            self.qv = zeros(nz, ny, nx)
            self.qc = zeros(nz, ny, nx)
            self.qr = zeros(nz, ny, nx)
            self.qv0 = rebuilt("qv0", nz, ny, nx)
            self.qc0 = rebuilt("qc0", nz, ny, nx)
            self.qr0 = rebuilt("qr0", nz, ny, nx)
            # WRF h_diabatic (Registry.EM_COMMON:1389, "MICROPHYSICS LATENT
            # HEATING", K s-1): the previous step's microphysics theta
            # increment per second, retained by moist_physics_finish_em
            # (module_big_step_utilities_em.F:5745) and fed to every RK
            # step's theta tendency (rk_addtend_dry, module_em.F:1078-1079).
            # Zero at init exactly as WRF (start_em.F:643-644), so the
            # first step's dynamics see no heating (one-step lag).
            # RESTART ADVISORY: WRF carries h_diabatic in the restart
            # stream (the `r` in the Registry IO string `rdu`,
            # Registry.EM_COMMON:1389); a future gpuwm restart
            # implementation must serialize this field — reconstructing it
            # is impossible and re-zeroing silently drops one step of
            # retained heating on the first resumed trajectory.
            self.h_diabatic = zeros(nz, ny, nx)
            if cfg.mp_physics in (6, 8, 10, 18, 28):
                common = ("qi", "qs", "qg", "effc", "effi", "effs")
                number_and_radius = (
                    ("nr", "ni") if cfg.mp_physics == 8 else
                    (("nc", "nr", "ni", "ns", "ng", "effr")
                     if cfg.mp_physics == 10 else
                    (("qh", "qndrop", "qnr", "qni", "qns", "qng",
                      "qnh", "qnn", "qvolg", "qvolh")
                     if cfg.mp_physics == 18 else
                    # mp=28 (Thompson aerosol-aware, Registry.EM_COMMON:3036)
                    # turns cloud droplet number into a prognostic scalar and
                    # adds the two aerosol number tracers.  The mp=8 tuple
                    # above is deliberately NOT shared: a shared tuple is the
                    # regression risk that would let a future mp=28 field
                    # silently appear on an mp=8 state.
                    (("nc", "nr", "ni", "nwfa", "nifa")
                     if cfg.mp_physics == 28 else ()))))
                for name in common + number_and_radius:
                    setattr(self, name, zeros(nz, ny, nx))
                if cfg.mp_physics == 18:
                    # WRF start_em.F initializes predicted NSSL CCN to
                    # nssl_cccn / 1.225 when no input field is present.
                    # Registry default nssl_cccn=0.5e9 m-3; the resulting
                    # dry-mass mixing ratio is exactly this FP32 value.
                    self.qnn[...] = DTYPE(408163264.0)
                if cfg.mp_physics in (6, 8, 28):
                    # module_model_constants.F WSM6/Thompson background radii,
                    # stored in gpuwm's radiation-facing micron convention.
                    # (RE_QC_BG/RE_QI_BG/RE_QS_BG = 2.49E-6/4.99E-6/9.99E-6
                    # m, module_model_constants.F:62-64.)  mp=28 shares
                    # them: module_mp_thompson.F seeds re_qc1d/re_qi1d/
                    # re_qs1d from those same three parameters and applies
                    # the same MAX(RE_*_BG, MIN(...)) clamp for BOTH
                    # entries (:1466-1479 in mp_gt_driver, the single
                    # driver classic and aerosol-aware Thompson share), so
                    # the background a radiation call sees before the first
                    # microphysics step is identical.
                    self.effc[...] = DTYPE(2.49)
                    self.effi[...] = DTYPE(4.99)
                    self.effs[...] = DTYPE(9.99)
                elif cfg.mp_physics == 10:
                    self.effc[...] = DTYPE(2.5)
                    self.effi[...] = DTYPE(5.0)
                    self.effs[...] = DTYPE(10.0)
                else:
                    # NSSL's native driver bounds are 2.51/10.01/25 um.
                    # State radii use gpuwm's radiation-facing micron
                    # convention; the official-source CUDA diagnostic itself
                    # retains WRF's metre convention at its narrow boundary.
                    self.effc[...] = DTYPE(2.51)
                    self.effi[...] = DTYPE(10.01)
                    self.effs[...] = DTYPE(25.0)
                time_copies = ("qi0", "qs0", "qg0")
                if cfg.mp_physics == 8:
                    time_copies += ("nr0", "ni0")
                elif cfg.mp_physics == 10:
                    time_copies += ("nr0", "ni0", "ns0", "ng0")
                elif cfg.mp_physics == 18:
                    time_copies += (
                        "qh0", "qndrop0", "qnr0", "qni0", "qns0",
                        "qng0", "qnh0", "qnn0", "qvolg0", "qvolh0")
                elif cfg.mp_physics == 28:
                    time_copies += ("nc0", "nr0", "ni0", "nwfa0", "nifa0")
                for name in time_copies:
                    setattr(self, name, rebuilt(name, nz, ny, nx))
                if cfg.mp_physics == 28:
                    # QNWFA2D / QNIFA2D: WRF's surface aerosol emission
                    # TENDENCIES in # kg-1 s-1 (Registry.EM_COMMON; the
                    # field was redefined from a concentration to a
                    # tendency on 13 May 2013, module_mp_thompson.F:
                    # 1313-1315).  They are INTENT(IN) to mp_gt_driver --
                    # microphysics reads them at :1310-1327 and never
                    # writes them -- so they are cross-step CONSTANTS, not
                    # RK-rebuilt copies, and must not go through
                    # ``rebuilt`` (a shared arena backing would let a
                    # sibling domain overwrite them between steps).
                    #
                    # Both start at exactly zero.  thompson_init derives
                    # nwfa2d from the synthetic CCN profile at :510, but
                    # nifa2d is not even a thompson_init dummy argument
                    # (:424-444 take nwfa2d/nbca2d only) and the whole
                    # file never assigns it.  A run with no WIF/dust
                    # ingest therefore keeps nifa2d == 0 for the entire
                    # forecast -- that is WRF's own behaviour, not an
                    # ArWen shortcut.
                    self.nwfa2d = zeros(ny, nx)
                    self.nifa2d = zeros(ny, nx)
        else:
            self.qv = self.qc = self.qr = None
            self.qv0 = self.qc0 = self.qr0 = None
            self.h_diabatic = None

        # WRF two-time-level prognostic TKE (Registry.EM_COMMON:312,
        # ``state real tke ikj dyn_em 2 - r``), the km_opt=2 carrier.
        # Initial value is the allocated zero state exactly as WRF's ideal
        # path (no start_em writer; tke bootstraps from the surface terms
        # or the tke_seed).  RESTART: WRF carries tke in the restart stream
        # (the ``r`` IO flag) and so does gpuwm -- ``tke`` is SERIALIZED and
        # ``tke0`` is REBUILT (written from tke at every dycore.step entry),
        # both classified in gpuwm/io/restart.py.
        if cfg.km_opt == 2:
            self.tke = zeros(nz, ny, nx)
            self.tke0 = rebuilt("tke0", nz, ny, nx)
        else:
            self.tke = self.tke0 = None
        # SASE prognostic subgrid turbulence energy.  The attribute is
        # ``e_sgs`` because ``self.e`` is already the WRF Coriolis cosine
        # parameter 2*Omega*cos(lat); the closure's symbol ``e`` maps to
        # this attribute everywhere (restart key ``state/e_sgs``,
        # preflight item ``e_sgs``).  Cold start fills the realizability
        # floor exactly as the fused step clips.  Allocated ONLY when the
        # closure is active, so every other configuration's object graph
        # stays byte-identical -- the attribute is ABSENT, not None,
        # matching the pattern the restart manifest walk expects.
        if cfg.bl_pbl_physics == SASE_PBL_SCHEME:
            self.e_sgs = zeros(nz, ny, nx)
            self.e_sgs.fill(DTYPE(SASE_E_MIN))

        # RK stage copies (state at the start of the RK3 step).
        self.u0 = rebuilt("u0", nz, ny, nx + 1)
        self.v0 = rebuilt("v0", nz, ny + 1, nx)
        self.w0 = rebuilt("w0", nz + 1, ny, nx)
        self.thp0 = rebuilt("thp0", nz, ny, nx)
        self.php0 = rebuilt("php0", nz + 1, ny, nx)
        self.mup0 = rebuilt("mup0", ny, nx)

        # Slow-physics tendencies (coupled form).
        self.ru_t = rebuilt("ru_t", nz, ny, nx + 1)
        self.rv_t = rebuilt("rv_t", nz, ny + 1, nx)
        self.rw_t = rebuilt("rw_t", nz + 1, ny, nx)
        self.rth_t = rebuilt("rth_t", nz, ny, nx)
        self.rph_t = rebuilt("rph_t", nz + 1, ny, nx)
        self.rmu_t = rebuilt("rmu_t", ny, nx)

        # Acoustic-substep perturbation fields: deviations from the RK stage
        # reference state t* (ARW Tech Note sec. 3.1.2; Tasks 10-12).
        # u_pp/v_pp/w_pp are coupled momenta (mu*u)'' etc., th_pp is coupled
        # (mu*theta)'', ph_pp/mu_pp/p_pp are phi''/mu''/p''; ww_pp is the
        # perturbation eta mass flux Omega'' at w levels; p_pp_old keeps the
        # previous substep's p'' for divergence damping.
        self.u_pp = rebuilt("u_pp", nz, ny, nx + 1)
        self.v_pp = rebuilt("v_pp", nz, ny + 1, nx)
        self.w_pp = rebuilt("w_pp", nz + 1, ny, nx)
        self.th_pp = rebuilt("th_pp", nz, ny, nx)
        self.ph_pp = rebuilt("ph_pp", nz + 1, ny, nx)
        self.mu_pp = rebuilt("mu_pp", ny, nx)
        self.p_pp = rebuilt("p_pp", nz, ny, nx)
        self.p_pp_old = rebuilt("p_pp_old", nz, ny, nx)
        self.ww_pp = rebuilt("ww_pp", nz + 1, ny, nx)
        # Acoustic specific volume alpha'' (Task 4): diagnosed with p'' each
        # substep and consumed by advance_uv's alpha''*d(pb)/dx term, which
        # is nonzero on eta surfaces over terrain.
        self.al_pp = rebuilt("al_pp", nz, ny, nx)

        # Base-state profiles (filled by load_base).  Flat terrain
        # (cfg.terrain_opt == 0, Phase 1) keeps 1-D columns; with terrain the
        # base state is per-column, so the device fields are full 3-D.
        if cfg.terrain_opt == 0:
            self.thb = zeros(nz)
            self.pb = zeros(nz)
            self.alb = zeros(nz)
            self.phb = zeros(nz + 1)
        else:
            self.thb = zeros(nz, ny, nx)
            self.pb = zeros(nz, ny, nx)
            self.alb = zeros(nz, ny, nx)
            self.phb = zeros(nz + 1, ny, nx)
        self.mub = DTYPE(0.0)
        self.p_top = None

        # General-form plumbing (Phase 2 Task 3): kernels that consumed the
        # scalar mub take the (ny, nx) dry-mass field plus the hybrid
        # coefficient arrays c1h/c2h (half levels) and c1f/c2f (full levels)
        # instead — Task 3 wires the diagnostics, Task 4 the dynamics.  ht
        # is the terrain height (WRF HGT; zeros when flat).
        self.mub2d = zeros(ny, nx)
        self.ht = zeros(ny, nx)
        self.c1h = zeros(nz)
        self.c2h = zeros(nz)
        self.c1f = zeros(nz + 1)
        self.c2f = zeros(nz + 1)
        # Hybrid reference-pressure coefficients (WRF c3 = B(eta), c4 =
        # (eta - B)(p0 - pt)): consumed only by the hypsometric_opt=2 EOS
        # diagnostic (calc_p_alpha), which rebuilds the reference dry
        # pressures pfu/pfd/phm = c3*mu + c4 + p_top per column.
        self.c3h = zeros(nz)
        self.c4h = zeros(nz)
        self.c3f = zeros(nz + 1)
        self.c4f = zeros(nz + 1)

        # Map-scale factors at mass/u/v points and Coriolis parameters
        # f = 2*Omega*sin(lat), e = 2*Omega*cos(lat) at mass points (Phase 3
        # Task 3; WRF msftx==msfty etc. — gpuwm carries the single isotropic
        # factor per staggering, exact for Lambert/polar/Mercator).  sina /
        # cosa are the local map-rotation angle (geo_em SINALPHA/COSALPHA,
        # WRF Registry.EM_COMMON:1405-1406) consumed by the coriolis kernel's
        # rotation terms; the identity defaults (sina = 0, cosa = 1) are
        # WRF's unrotated setting (module_big_step_utilities_em.F:3703-3704).
        # The defaults (msf 1, f/e 0, identity rotation) keep every flux form
        # bitwise on the Phase 2 path; use set_map_coriolis() to change them
        # so the has_msf / rotational flags stay consistent.
        self.msft = xp.ones((ny, nx), dtype=np.float32)
        self.msfu = xp.ones((ny, nx + 1), dtype=np.float32)
        self.msfv = xp.ones((ny + 1, nx), dtype=np.float32)
        self.f = zeros(ny, nx)
        self.e = zeros(ny, nx)
        self.sina = zeros(ny, nx)
        self.cosa = xp.ones((ny, nx), dtype=np.float32)
        #: any map factor != 1 (selects the msf-weighted flux forms).
        self.has_msf = False
        #: has_msf or any f/e != 0 (gates the Coriolis+curvature kernel).
        self.rotational = False

        # Phase 3 Task 8: optional setup-time lateral forcing and model
        # clock.  The object is deliberately untyped here to avoid making
        # the core state module import the ingest package.
        self.lateral_boundaries = None
        self.elapsed_seconds = 0.0

        # Phase 3 Task 12: an explicitly initialized PhysicsDriver.  Keeping
        # the default as None makes every pre-physics state allocation and
        # idealized run byte-for-byte unchanged; dycore.step only consults
        # this slot when a physics scheme is enabled in RunConfig.
        self.physics = None

        # Vertical-coordinate arrays (filled by load_base).
        self.dnw = zeros(nz)
        self.rdnw = zeros(nz)
        self.dn = zeros(nz)
        self.rdn = zeros(nz)
        self.fnp = zeros(nz)
        self.fnm = zeros(nz)
        self.znu = zeros(nz)
        self.znw = zeros(nz + 1)

        # Surface extrapolation weights for half-level fields (WRF cf1..cf3,
        # filled by load_base); the acoustic pressure gradient needs p'' at
        # the lowest full level.
        self.cf1 = DTYPE(0.0)
        self.cf2 = DTYPE(0.0)
        self.cf3 = DTYPE(0.0)
        # Model-top linear extrapolation weights for full-level fields
        # (WRF cfn/cfn1, module_initialize_real.F:3754-3755).
        self.cfn = DTYPE(0.0)
        self.cfn1 = DTYPE(0.0)

        # Scratch-buffer pool (see scratch()) and host base geopotential
        # (None until load_base runs — height_half() raises on the sentinel).
        self._scratch: dict[str, cp.ndarray] = {}
        # Keep the default object graph exactly as before: single-domain and
        # frozen real74 paths have no _scratch_arena attribute at all. Only
        # Task-14's multi-domain builder injects this optional infrastructure.
        if scratch_arena is not None:
            self._scratch_arena = scratch_arena
        self._phb_host: np.ndarray | None = None
        self._dz_min: float | None = None

        if cfg.nwp_diagnostics == 1:
            # WRF UP_HELI_MAX (Registry.EM_COMMON:2083, IO "rh02"): allocate
            # the serialized running-max accumulator eagerly so restart
            # manifests and wrfout frame schemas are deterministic from the
            # first step (gpuwm/core/uh_diag.py owns the update/reset).
            self.scratch((ny, nx), "up_heli_max")

    def load_base(self, coord: VerticalCoord, base: BaseState) -> None:
        """Copy the float64 setup-time coordinate/base arrays to device FP32."""
        xp = _state_array_module(self)
        for name in ("dnw", "rdnw", "dn", "rdn", "fnp", "fnm", "znu", "znw",
                     "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f"):
            getattr(self, name)[...] = xp.asarray(getattr(coord, name),
                                                  dtype=np.float32)
        for name in ("thb", "pb", "alb", "phb"):
            dev = getattr(self, name)
            host = np.asarray(getattr(base, name), dtype=np.float64)
            if host.ndim != dev.ndim:
                raise ValueError(
                    f"base state {name} is {host.ndim}-D but the state was "
                    f"allocated for {dev.ndim}-D profiles: cfg.terrain_opt "
                    "must match the terrain_z the base state was built with")
            dev[...] = xp.asarray(host, dtype=np.float32)
        self.p_top = DTYPE(base.p_top)
        if np.ndim(base.mub) == 0:
            self.mub = DTYPE(base.mub)
            self.mub2d[...] = self.mub
        else:
            # Terrain: the (ny, nx) field is the only valid dry mass.  The
            # scalar is retired (None) so any consumer not yet wired for
            # terrain (Task 4) fails loudly instead of computing garbage.
            self.mub = None
            self.mub2d[...] = xp.asarray(base.mub, dtype=np.float32)
        self.ht[...] = (0.0 if base.terrain_z is None
                        else xp.asarray(base.terrain_z, dtype=np.float32))
        # Own the host geopotential backing the invariant spacing cache.
        # BaseState is mutable, and np.asarray would alias an FP64 base.phb;
        # a later caller mutation could then change height_half() without
        # invalidating _dz_min.  The device load already has copy semantics,
        # so retain the same snapshot on host as well.
        self._phb_host = np.array(base.phb, dtype=np.float64, copy=True)
        z_half = _height_half_from_phb(self._phb_host)
        self._dz_min = (float(np.diff(z_half, axis=0).min())
                        if z_half.shape[0] > 1 else None)

        # WRF surface extrapolation weights (dyn_em module_initialize):
        # quadratic-in-eta extrapolation of half-level fields to znw[0].
        if coord.dnw.size >= 3:
            dn, dnw, fnp, fnm = coord.dn, coord.dnw, coord.fnp, coord.fnm
            cof1 = (2.0 * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
            cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
            self.cf1 = DTYPE(fnp[1] + cof1)
            self.cf2 = DTYPE(fnm[1] - cof1 - cof2)
            self.cf3 = DTYPE(cof2)
        if coord.dnw.size >= 1:
            self.cfn = DTYPE(1.0 + coord.fnp[-1])
            self.cfn1 = DTYPE(-coord.fnp[-1])

    def set_map_coriolis(self, msft=None, msfu=None, msfv=None,
                         f=None, e=None, sina=None, cosa=None) -> None:
        """Fill the map factors / Coriolis parameters (float64 host inputs).

        The sanctioned setter: it refreshes the ``has_msf`` (any msf != 1)
        and ``rotational`` (has_msf or any f/e != 0) flags the dycore keys
        its msf-weighted paths and the Coriolis+curvature kernel on.
        Identity values (all-ones msf, all-zero f/e) leave both flags off,
        preserving the bitwise Phase 2 step.  ``sina``/``cosa`` are the
        local map-rotation angle (geo_em SINALPHA/COSALPHA); they only
        scale the e-Coriolis terms inside the kernel, so they do not enter
        the flags — a rotated frame with f = e = 0 exerts no force, exactly
        as in WRF.  Direct assignment to the arrays bypasses the flags —
        don't.
        """
        xp = _state_array_module(self)
        for name, val in (("msft", msft), ("msfu", msfu), ("msfv", msfv),
                          ("f", f), ("e", e), ("sina", sina), ("cosa", cosa)):
            if val is None:
                continue
            dev = getattr(self, name)
            host = np.asarray(val, dtype=np.float64)
            if host.shape != dev.shape:
                raise ValueError(f"{name} must have shape {dev.shape}, "
                                 f"got {host.shape}")
            dev[...] = xp.asarray(host, dtype=np.float32)
        self.has_msf = bool((self.msft != 1.0).any()
                            or (self.msfu != 1.0).any()
                            or (self.msfv != 1.0).any())
        self.rotational = bool(self.has_msf or (self.f != 0.0).any()
                               or (self.e != 0.0).any())

    def scratch(self, shape, slot: str, dtype=None) -> cp.ndarray:
        """Persistent named scratch buffer; the only sanctioned extra device
        allocation.  A slot keeps the shape it was first requested with.

        Without an injected arena this is the original per-state zero-allocation
        path. With an arena, only registry-audited slots present in that arena
        draw a view; carrying/unproven slots still allocate per state.
        """
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        requested_dtype = np.dtype(np.float32 if dtype is None else dtype)
        buf = self._scratch.get(slot)
        if buf is None:
            arena = getattr(self, "_scratch_arena", None)
            if arena is not None and arena.has_slot(slot):
                buf = arena.view(shape, slot, requested_dtype)
            else:
                xp = _state_array_module(self)
                buf = xp.zeros(shape, dtype=requested_dtype)
            self._scratch[slot] = buf
        elif buf.shape != shape:
            raise ValueError(f"scratch slot {slot!r} has shape {buf.shape}, "
                             f"requested {shape}")
        elif buf.dtype != requested_dtype:
            raise ValueError(f"scratch slot {slot!r} has dtype {buf.dtype}, "
                             f"requested {requested_dtype}")
        return buf

    def existing_scratch(self, slot: str) -> cp.ndarray | None:
        """Return the named scratch buffer only if it already exists.

        Never allocates: lets the owner of a persistent slot family (the
        microphysics ring guard snapshotting its own ``mp_*`` accumulators)
        inspect a slot without creating unused buffers for schemes that
        never write it.
        """
        return self._scratch.get(slot)

    def total_theta(self) -> cp.ndarray:
        """Full potential temperature thb + theta' as a device array."""
        thb = self.thb
        return (thb if thb.ndim == 3 else thb[:, None, None]) + self.thp

    def total_mu(self) -> cp.ndarray:
        """Total dry column mass mub + mu' as a ``(ny, nx)`` device array
        (``mub2d`` is the scalar broadcast for flat terrain)."""
        return self.mub2d + self.mup

    def cell_area_weight(self) -> cp.ndarray:
        """Mass-point cell-area weight ``1/msft**2`` as ``(ny, nx)`` FP64.

        ARW carries the column-mass equation on the map plane: the
        acoustic mass update multiplies the layer divergence by
        ``msftx*msfty``, which for gpuwm's isotropic factor is the
        ``m2 = msft*msft`` product formed in
        ``kernels/acoustic.cu advance_mu_th_msf``.  Dividing the column
        mass by that same product restores the physical cell area, so
        ``sum(total_mu * cell_area_weight)`` is the quantity whose
        tendency telescopes to the lateral boundary faces.  The reciprocal
        is taken in FP64 from the FP32 product the kernel itself forms, so
        the weight is the kernel's convention rather than a re-derivation;
        with identity map factors it is exactly 1.0 and the weighted sum
        is bit-identical to the unweighted one.
        """
        xp = _state_array_module(self)
        msft2 = self.msft * self.msft
        return 1.0 / msft2.astype(xp.float64)

    def height_half(self) -> np.ndarray:
        """Base-state half-level heights in metres, on host: ``(nz,)`` for a
        flat base state, per-column ``(nz, ny, nx)`` with terrain.

        Raises ``RuntimeError`` if :meth:`load_base` was never called (it
        used to silently return zeros — final-review carry-over T6).
        """
        if self._phb_host is None:
            raise RuntimeError(
                "height_half() called before load_base(): the base-state "
                "geopotential has not been loaded")
        return _height_half_from_phb(self._phb_host)

    @property
    def dz_min(self) -> float | None:
        """Cached minimum half-level spacing installed with the base state.

        ``None`` denotes a one-layer domain, whose CFL fallback remains the
        configured model-top height.  The pre-load error intentionally
        matches :meth:`height_half`, which formerly supplied this value to
        the integration health check.
        """
        if self._phb_host is None:
            raise RuntimeError(
                "height_half() called before load_base(): the base-state "
                "geopotential has not been loaded")
        return self._dz_min


def _check_terrain_z(base: BaseState, terrain_z) -> None:
    """Cross-check an explicit terrain profile against the base state's.

    The base state (built by ``make_base_state``) is the terrain authority;
    an explicit ``terrain_z`` at init time is a call-site consistency check
    only, and any disagreement raises.
    """
    if terrain_z is None:
        return
    if base.terrain_z is None:
        raise ValueError(
            "terrain_z given but the base state is flat: build the base "
            "state with make_base_state(..., terrain_z=...) so its profiles "
            "carry the terrain")
    if not np.array_equal(np.asarray(terrain_z, dtype=np.float64),
                          np.asarray(base.terrain_z, dtype=np.float64)):
        raise ValueError(
            "terrain_z disagrees with the terrain the base state was built "
            "with (base.terrain_z)")


def init_at_rest(cfg: RunConfig, coord: VerticalCoord, base: BaseState,
                 terrain_z: np.ndarray | None = None) -> DomainState:
    """Allocate a state with zero perturbations on the given base state.

    ``terrain_z`` (optional) must match ``base.terrain_z``; ``ht`` is always
    filled from the base state, so flat call sites are unchanged.
    """
    _check_terrain_z(base, terrain_z)
    s = DomainState(cfg)
    s.load_base(coord, base)
    return s


def _rebalance_hydrostatic_terrain(th_total: np.ndarray, base: BaseState,
                                   coord: VerticalCoord) -> np.ndarray:
    """Per-column discrete hydrostatic recurrence over terrain, float64.

    The hybrid/terrain generalization of ``grid.rebalance_hydrostatic``:
    half-level dry pressure ``pd = c3h*mub + c4h + p_top`` per column,
    column-mass increments ``c1h*mub + c2h``, and the surface pinned at
    ``g*terrain_z``.  Expressions are ordered exactly as in
    ``make_base_state`` so that ``th_total == thb`` reproduces ``phb``
    bitwise and phi' stays identically zero (the base state already
    carries the terrain).
    """
    nz, ny, nx = th_total.shape
    mub = np.asarray(base.mub, dtype=np.float64)
    p = (coord.c3h[:, None, None] * mub[None]
         + coord.c4h[:, None, None] + base.p_top)
    alpha = c.RD * th_total * (p / c.P0) ** c.RCP / p

    ph = np.zeros((nz + 1, ny, nx))
    ph[0] = c.G * base.terrain_z
    for k in range(nz):
        ph[k + 1] = ph[k] - coord.dnw[k] * (coord.c1h[k] * mub
                                            + coord.c2h[k]) * alpha[k]
    return ph


def init_theta_perturbation(cfg: RunConfig, coord: VerticalCoord,
                            base: BaseState, thp_func,
                            terrain_z: np.ndarray | None = None
                            ) -> DomainState:
    """At-rest state plus a theta perturbation, hydrostatically rebalanced.

    ``thp_func(x, z) -> theta' (nz, ny, nx) numpy`` receives the domain-
    centered cell-center x coordinates ``x (nx,)`` and the base-state
    half-level heights ``z`` — ``(nz,)`` for a flat base state (the Phase 1
    contract, bitwise unchanged), per-column ``(nz, ny, nx)`` with terrain.
    ``phi'`` is set so every column is discretely balanced (identically
    zero for theta' = 0); ``mu' = 0``; winds stay zero — cases set them on
    the returned state.
    """
    s = init_at_rest(cfg, coord, base, terrain_z)

    x = (np.arange(cfg.nx) + 0.5) * cfg.dx - 0.5 * cfg.nx * cfg.dx
    z = s.height_half()
    thp = np.asarray(thp_func(x, z), dtype=np.float64)

    if base.terrain_z is None:
        th_total = base.thb[:, None, None] + thp
        ph_total = rebalance_hydrostatic(th_total, base.mub, coord,
                                         cfg.p_surf)
        php = ph_total - base.phb[:, None, None]
    else:
        th_total = base.thb + thp
        ph_total = _rebalance_hydrostatic_terrain(th_total, base, coord)
        php = ph_total - base.phb

    s.thp[...] = cp.asarray(thp, dtype=DTYPE)
    s.php[...] = cp.asarray(php, dtype=DTYPE)
    return s
