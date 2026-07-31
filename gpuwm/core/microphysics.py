"""Microphysics: scheme registry and the Kessler warm-rain launcher.

:func:`apply` is the single public surface, called by ``dycore.step`` after
the RK3 loop (WRF solve_em: the non-timesplit microphysics runs once per
model step on the updated fields) and dispatching on ``cfg.mp_physics``:
0 is a no-op (the state is untouched -- the whole dry/moist-without-mp
suite is bitwise unaffected), 1 is Kessler
(kernels/kessler.cu, transcribed from the bundle's WRF v4.6.1
``phys/module_mp_kessler.F``; float64 mirror
``gpuwm.verify.npref.np_kessler_column``), 6 is WSM6, 8 is the classic
Thompson CUDA path (packaged byte-validated WRF v4.6.1 tables, env-var
table-root override honored), 10 is Morrison two-moment
(``gpuwm.core.morrison`` / ``kernels/morrison.cu``), and 18 is NSSL
two-moment through its persistent production binding.

The scheme inputs follow WRF ``moist_physics_prep_em``
(dyn_em/module_big_step_utilities_em.F) with ``use_theta_m = 0`` (gpuwm's
dry-theta prognostic): full potential temperature ``thb + thp``, DRY
density ``rho = 1/alt``, full-pressure Exner ``pii = (p/p0)^(Rd/cp)``, and
half-level heights / layer depths from the full geopotential -- so
``apply`` must run with up-to-date EOS diagnostics (``dycore.step`` calls
it right after its epilogue ``update_diagnostics``).  The scheme updates
theta (latent heating) and the active water mass/number moments in place;
surface precipitation accumulates in persistent ``mp_*`` scratch slots
(mm, matching WRF's RAINNC/SNOWNC/GRAUPELNC and per-call ``*NCV`` fields).

h_diabatic (audit finding R7): each scheme adapter brackets its column
kernel with WRF's prep/finish pair.  :func:`save_pre_mp_theta` parks the
pre-microphysics full theta in ``state.h_diabatic`` (moist_physics_prep_em,
module_big_step_utilities_em.F:5503/:5523-5526) and
:func:`moist_physics_finish` applies the clamped theta increment ONCE to
the prognostic theta and stores the heating rate ``mpten/dt`` back in
``state.h_diabatic`` (moist_physics_finish_em, :5682-5746) for the NEXT
step's RK tendencies (dycore.add_h_diabatic_tendency).  Float64 mirror:
``gpuwm.verify.npref.np_moist_physics_finish``.

Specified-zone ring exclusion (Wave-1 rank-1 seam fix): on every
specified or nested WRF domain the microphysics tiles are clipped by
``sz = spec_zone`` (solve_em.F:3618-3622; ``specified .or. nested`` per
:3692) -- moist_physics_prep (:3631-3639), the scheme driver
(module_microphysics_driver.F:802-806/:870-879), and
moist_physics_finish including the h_diabatic capture (:4040-4048) all
run ``its = max(i_start, ids+sz) .. ite = min(i_end, ide-1-sz)``,
``jts = max(j_start, jds+sz) .. jte = min(j_end, jde-1-sz)``, so the
outermost ``sz`` mass ring is NEVER touched: ring RAINNC is exactly 0.0
at every lead, ring theta/moisture keep their boundary-installed values,
and ring h_diabatic stays 0 (allocator zero init,
frame/module_domain.F:770-777 / tools/gen_allocs.c:411-415;
set_physical_bc3d writes halo indices only, module_bc.F:867-885).
Scope qualifications (explicit non-goals): WRF's optional
``mp_zero_out`` path zeroes moisture over WHOLE-field mass bounds before
the clipped finish (solve_em.F:4002-4038) -- gpuwm has no mp_zero_out,
so the never-touched statement holds for every supported configuration;
and a specified+periodic_x channel clips j only
(module_microphysics_driver.F:871-873) -- gpuwm has no channel mode and
always excludes the four-sided ring.  An earlier
in-tree justification argued the whole-field call was inert because the
spec-zone theta TENDENCY is replaced and spec-zone theta values are
overwritten each stage; that argument was overbroad -- the schemes'
direct in-place ring theta/moisture updates and the finish's direct
``thp += mpten`` land AFTER dycore.step's end-of-step
``apply_state_boundary_values``, and ring RAINNC accumulated where WRF's
is exactly zero, leaking into interior rows through the next step's
stencils (measured: O(0.1 mm/h) ring RAINNC and a sustained rows-1..4
theta/qv envelope).  :func:`apply` therefore excludes the ring for
specified/nested configs: every scheme here is a per-column kernel with
no horizontal coupling inside the call, so capturing the ring before
dispatch and bit-restoring it afterwards is bitwise identical -- ring
AND interior -- to WRF's clipped tiles (see
:func:`spec_zone_ring_slices`).  Periodic/open configs keep the exact
whole-field behavior (WRF: sz = 0), so every frozen idealized case is
bitwise unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE, DomainState
from gpuwm.physics_compat import thompson_table_root as _thompson_table_root

_TPB = 64          # threads per block (one thread per column)
_KMAX = 256        # matches KESS_KMAX in kernels/kessler.cu
KESSLER_VERTICAL_LEVEL_BOUNDS = (None, _KMAX)

# ---------------------------------------------------------------------------
# WRF specified-zone ring exclusion (solve_em.F:3618-3639)
# ---------------------------------------------------------------------------

#: State attributes a scheme adapter (or the prep/finish bracket) updates in
#: place.  Attributes the configured scheme does not allocate are skipped.
#: h_diabatic is deliberately absent: its ring value is pinned to WRF's
#: exact 0 in :func:`_restore_spec_zone_ring` instead of being restored.
_RING_STATE_FIELDS = (
    "thp", "qv", "qc", "qr", "qi", "qs", "qg",
    "nc", "nr", "ni", "ns", "ng",
    "effc", "effr", "effi", "effs",
)

#: Persistent (ny, nx) surface slots the adapters write: the accumulators
#: RAINNC/SNOWNC/GRAUPELNC plus the per-call *NCV/SR diagnostics (the
#: canonical driver mapping is physics.microphysics_scratch_slots; this is
#: its union across schemes).  WRF never writes any of them in the ring:
#: the accumulators/NCVs by the clipped tiles (zero init), SR by
#: ``grid%sr = 0.`` whole-field each call (solve_em.F:3691) followed by
#: clipped tile writes.
_RING_SURFACE_SLOTS = (
    "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
    "mp_graupelnc", "mp_graupelncv", "mp_sr", "mp_kessler_sr",
)

#: Persistent 3-D scratch slots that survive the call (the REFL_10CM
#: output stash -- WRF's clipped tiles leave ring refl at its allocated 0).
_RING_VOLUME_SLOTS = ("refl_10cm",)


def spec_zone_ring_slices(ny: int, nx: int, sz: int):
    """Non-overlapping index tuples covering exactly the ring WRF's
    clipped microphysics tiles exclude.

    WRF (1-based, ide/jde staggered ends): tiles run
    ``its = ids+sz .. ide-1-sz``, ``jts = jds+sz .. jde-1-sz``
    (solve_em.F:3631-3639, :4040-4048;
    module_microphysics_driver.F:870-879).  On gpuwm's 0-based (ny, nx)
    mass grid the surviving tile is ``sz .. nx-1-sz`` x ``sz .. ny-1-sz``
    and the excluded ring is ``i < sz or i > nx-1-sz or j < sz or
    j > ny-1-sz``.  The leading Ellipsis makes each tuple apply to
    (ny, nx) and (nz, ny, nx) arrays alike.  Degenerate domains
    (``2*sz >= ny`` or ``nx`` -- WRF's clip leaves an empty tile) are
    covered without overlap.
    """
    n_lo = min(sz, ny)
    n_hi = max(ny - sz, n_lo)
    e_lo = min(sz, nx)
    e_hi = max(nx - sz, e_lo)
    return (
        (Ellipsis, slice(0, n_lo), slice(None)),          # south rows
        (Ellipsis, slice(n_hi, ny), slice(None)),         # north rows
        (Ellipsis, slice(n_lo, n_hi), slice(0, e_lo)),    # west columns
        (Ellipsis, slice(n_lo, n_hi), slice(e_hi, nx)),   # east columns
    )


def spec_zone_ring_save_slots(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Preflight-registry helper: every ``mp_ring_save_*`` snapshot slot the
    ring guard creates for this config, with its exact shape.

    Mirrors :func:`_capture_spec_zone_ring` and
    :func:`spec_zone_ring_slices`: one slot per (captured array, non-empty
    ring edge).  Empty when the guard is off (periodic/open, sz = 0, or no
    microphysics).  Consumed by
    ``gpuwm.core.preflight.scratch_slot_registry`` so the completeness and
    allocation gates see the family with true sizes.
    """
    if not (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False)):
        return {}
    sz = int(cfg.spec_zone)
    if sz <= 0 or cfg.mp_physics == 0 or not cfg.moist:
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    fields = ["thp", "qv", "qc", "qr"]
    if cfg.mp_physics in (6, 8, 10):
        fields += ["qi", "qs", "qg"]
    if cfg.mp_physics == 8:
        fields += ["nr", "ni"]
    if cfg.mp_physics == 10:
        fields += ["nc", "nr", "ni", "ns", "ng"]
    if cfg.mp_physics in (6, 8, 10):
        fields += ["effc", "effi", "effs"]
    if cfg.mp_physics == 10:
        fields += ["effr"]
    # Every scheme (Kessler's rain-only fallback included) stashes due
    # reflectivity into the same persistent refl_10cm slot, and the guard
    # captures that slot unconditionally once it exists -- so the family
    # must be enumerated for every scheme or the first due-reflectivity
    # call would allocate unbudgeted mp_ring_save_refl_10cm_* buffers
    # behind the allocation gate.
    volume_slots = ["refl_10cm"]
    if cfg.mp_physics == 1:
        surface_slots = ["mp_rainnc", "mp_rainncv", "mp_kessler_sr"]
    else:
        surface_slots = ["mp_rainnc", "mp_rainncv", "mp_snownc",
                         "mp_snowncv", "mp_graupelnc", "mp_graupelncv",
                         "mp_sr"]
    n_lo = min(sz, ny)
    n_hi = max(ny - sz, n_lo)
    e_lo = min(sz, nx)
    e_hi = max(nx - sz, e_lo)
    edge_dims = ((n_lo, nx), (ny - n_hi, nx),
                 (n_hi - n_lo, e_lo), (n_hi - n_lo, nx - e_hi))
    slots: dict[str, tuple[int, ...]] = {}
    for key in fields + volume_slots:
        for index, (rows, cols) in enumerate(edge_dims):
            if rows and cols:
                slots[f"mp_ring_save_{key}_{index}"] = (nz, rows, cols)
    for key in surface_slots:
        for index, (rows, cols) in enumerate(edge_dims):
            if rows and cols:
                slots[f"mp_ring_save_{key}_{index}"] = (rows, cols)
    return slots


def _ring_guard_slices(state: DomainState, cfg: RunConfig):
    """WRF solve_em.F:3618-3622: ``sz = spec_zone`` iff ``specified .or.
    nested``, else 0 -- so periodic/open configs return None and keep the
    frozen whole-field call bitwise.  gpuwm has no specified+periodic_x
    channel mode, so WRF's channel i-clip exemption
    (module_microphysics_driver.F:871-873) is deliberately not wired.
    """
    if not (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False)):
        return None
    sz = int(cfg.spec_zone)
    if sz <= 0:
        return None
    _, ny, nx = state.p.shape
    return spec_zone_ring_slices(ny, nx, sz)


def _capture_spec_zone_ring(state: DomainState, slices):
    """Snapshot every ring section the microphysics call could touch.

    Every scheme here is a per-column kernel (and prep/finish are
    elementwise), so the call has no horizontal coupling: restoring the
    ring afterwards is bitwise identical -- ring AND interior -- to never
    running the scheme there, which is gpuwm's equivalent of WRF's
    clipped tiles for whole-field kernel launches
    (tests/test_mp_spec_zone_ring.py pins both halves).
    """
    saved = []
    captured_slots = set()

    def snap(arr, key):
        for index, slc in enumerate(slices):
            part = arr[slc]
            if part.size == 0:
                continue
            buf = state.scratch(part.shape, f"mp_ring_save_{key}_{index}")
            buf[...] = part
            saved.append((arr, slc, buf))

    for name in _RING_STATE_FIELDS:
        arr = getattr(state, name, None)
        if arr is not None:
            snap(arr, name)
    for slot in _RING_SURFACE_SLOTS + _RING_VOLUME_SLOTS:
        arr = state.existing_scratch(slot)
        if arr is not None:
            snap(arr, slot)
            captured_slots.add(slot)
    return saved, captured_slots


def _restore_spec_zone_ring(state: DomainState, slices, saved,
                            captured_slots) -> None:
    """Bit-restore the captured ring, zero the ring of any persistent slot
    the call just created (its pre-call value WAS the fresh zero fill),
    and pin ring h_diabatic to WRF's exact value.

    h_diabatic is pinned to 0 rather than restored: WRF's ring h_diabatic
    is identically zero in every specified/nested run (allocator zero
    init, frame/module_domain.F:770-777 / tools/gen_allocs.c:411-415;
    the clipped finish tiles never write it; set_physical_bc3d writes
    halo indices only, module_bc.F:867-885), and the pinned zero -- not a
    restored stale rate -- is what the next step's rk_addtend_dry slot
    must see.
    """
    for arr, slc, buf in saved:
        arr[slc] = buf
    for slot in _RING_SURFACE_SLOTS + _RING_VOLUME_SLOTS:
        if slot in captured_slots:
            continue
        arr = state.existing_scratch(slot)
        if arr is None:
            continue
        for slc in slices:
            arr[slc] = 0
    if state.h_diabatic is not None:
        for slc in slices:
            state.h_diabatic[slc] = 0


@dataclass(frozen=True)
class MicrophysicsDiagnostics:
    """Named post-RK microphysics-to-physics-driver contract.

    Accumulators/increments are millimetres (kg m-2), and ``sr`` is the
    scheme-native frozen fraction used by Noah when ``frpcpn`` is active.  This
    replaces knowledge of private ``DomainState._scratch`` names in the
    pre-RK surface driver.
    """

    rainnc: cp.ndarray
    rainncv: cp.ndarray
    sr: cp.ndarray
    snownc: cp.ndarray | None = None
    snowncv: cp.ndarray | None = None
    graupelnc: cp.ndarray | None = None
    graupelncv: cp.ndarray | None = None
    hailnc: cp.ndarray | None = None
    hailncv: cp.ndarray | None = None


def save_pre_mp_theta(state: DomainState) -> None:
    """WRF ``moist_physics_prep_em``: park the pre-microphysics FULL theta
    in the h_diabatic array (module_big_step_utilities_em.F:5503 "use
    h_diabatic to temporarily save pre-microphysics full theta",
    :5523-5526; ``use_theta_m = 0`` conversion ``th_phy = t_new + t0`` at
    :5513-5521).  :func:`moist_physics_finish` consumes and overwrites it.
    """
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    state.h_diabatic[...] = thb + state.thp


def moist_physics_finish(state: DomainState, cfg: RunConfig, th_phy,
                         dt: float) -> None:
    """WRF ``moist_physics_finish_em`` (module_big_step_utilities_em.F:
    5593-5784), ``use_theta_m = 0`` branch; float64 mirror
    ``gpuwm.verify.npref.np_moist_physics_finish``.

    ``th_phy`` is the scheme's post-microphysics full theta;
    ``state.h_diabatic`` holds the :func:`save_pre_mp_theta` full theta.
    With ``cfg.no_mp_heating == 0`` (:5682; Registry default) the theta
    increment ``mpten = th_phy - saved`` (:5688) is clamped to
    ``+/- cfg.mp_tend_lim*dt`` (:5706-5707), added ONCE to the prognostic
    perturbation theta (:5743), and retained as the heating rate
    ``h_diabatic = mpten/dt`` (:5745) that the NEXT step's RK loop feeds
    to the theta tendency.  With ``no_mp_heating = 1`` theta is left
    untouched (:5775) and ``h_diabatic = 0`` (:5776) -- the scheme's
    moisture updates stand either way.
    """
    if cfg.no_mp_heating == 0:
        lim = DTYPE(cfg.mp_tend_lim * dt)
        mpten = th_phy - state.h_diabatic            # :5688
        mpten = cp.minimum(lim, mpten)               # :5706
        mpten = cp.maximum(-lim, mpten)              # :5707
        state.thp += mpten                           # :5743
        state.h_diabatic[...] = mpten / DTYPE(dt)    # :5745
    else:
        state.h_diabatic[...] = 0.0                  # :5775-5776


def launch_kessler(t, qv, qc, qr, rho, pii, z, dz8w, rainnc, rainncv,
                   dt: float) -> None:
    """Run the ``kessler_column`` kernel on device arrays (in place).

    ``t`` (full theta), ``qv``/``qc``/``qr`` (kg/kg) are updated; ``rho``
    (dry density), ``pii`` (Exner), ``z``/``dz8w`` (m) are read-only, all
    ``(nz, ny, nx)`` float32; ``rainnc``/``rainncv`` are ``(ny, nx)``
    accumulated / per-call surface rain in mm.
    """
    nz, ny, nx = t.shape
    if nz > KESSLER_VERTICAL_LEVEL_BOUNDS[1]:
        raise ValueError(f"nz={nz} exceeds the Kessler kernel's per-thread "
                         "column storage KESS_KMAX="
                         f"{KESSLER_VERTICAL_LEVEL_BOUNDS[1]}")
    kern = get_kernel("kessler", "kessler_column")
    blocks = (ny * nx + _TPB - 1) // _TPB
    kern((blocks,), (_TPB,),
         (t, qv, qc, qr, rho, pii, z, dz8w, rainnc, rainncv,
          DTYPE(dt), np.int32(nz), np.int32(ny), np.int32(nx)))


def _apply_kessler(state: DomainState, cfg: RunConfig, dt: float, *,
                    refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """Build the WRF prep fields from the state and run the column kernel."""
    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]

    th = state.scratch((nz, ny, nx), "mp_th")
    rho = state.scratch((nz, ny, nx), "mp_rho")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    zh = state.scratch((nz, ny, nx), "mp_z")
    dz8w = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")

    th[...] = thb + state.thp
    rho[...] = 1.0 / state.alt                   # dry density 1/(al+alb)
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    zh[...] = 0.5 * (z8w[:nz] + z8w[1:])
    dz8w[...] = z8w[1:] - z8w[:nz]

    rainnc = state.scratch((ny, nx), "mp_rainnc")
    rainncv = state.scratch((ny, nx), "mp_rainncv")
    save_pre_mp_theta(state)                     # WRF moist_physics_prep_em
    launch_kessler(th, state.qv, state.qc, state.qr, rho, pii, zh, dz8w,
                   rainnc, rainncv, dt)
    if refl_10cm_due:
        # Kessler has no native refl10cm routine; keep gpuwm's documented
        # fallback on the same prepared-p/post-scheme-T timing as Morrison.
        from gpuwm.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = th * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, th, dt)     # theta' += mpten; h_diabatic
    sr = state.scratch((ny, nx), "mp_kessler_sr")
    sr[...] = 0.0
    return MicrophysicsDiagnostics(rainnc=rainnc, rainncv=rainncv, sr=sr)


def _apply_thompson(
        state: DomainState, cfg: RunConfig, dt: float, *,
        refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """Forecast adapter for the WRF-ordered classic Thompson path.

    The source, adjustment, same-call melt/fallout, and final phase-cleanup
    order mirrors ``mp_thompson``.  Formerly reachable only behind the
    GPUWM_EXPERIMENTAL_THOMPSON_MP8 process guard; promoted to a first-class
    scheme when the canonical classic tables became package data (product
    decision, product/v1 packaging lane 2026-07-28).  The table root
    resolves to the packaged directory unless GPUWM_THOMPSON_TABLE_ROOT
    overrides it, and ``load_classic_device_tables`` still byte-validates
    every asset before GPU setup -- a wrong or missing root fails closed.
    """
    table_root = _thompson_table_root()
    required = ("qc", "qi", "ni", "qs", "qg", "qr", "nr")
    missing = [name for name in required
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "Thompson mp=8 state lacks " + ", ".join(missing))

    from gpuwm.core.thompson import (
        launch_cloud_sedimentation,
        launch_cloud_saturation_adjust,
        launch_classic_graupel_number_finalize,
        launch_classic_graupel_number_init,
        launch_effective_radius,
        launch_final_phase_cleanup,
        launch_frozen_vapor_network_from_owner,
        launch_graupel_fallout_column_mask,
        launch_graupel_sedimentation,
        launch_hydrometeor_column_mask,
        launch_ice_sedimentation,
        launch_rain_evaporation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
        launch_warm_frozen_source_network_from_owner,
    )
    from gpuwm.core.thompson_runtime import load_classic_device_tables

    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    th = state.scratch((nz, ny, nx), "mp_th")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    temperature = state.scratch((nz, ny, nx), "mp_thompson_temperature")
    dz = state.scratch((nz, ny, nx), "mp_dz8w")
    frozen_reference_density = state.scratch(
        (nz, ny, nx), "mp_thompson_frozen_reference_density")
    frozen_reference_temperature = state.scratch(
        (nz, ny, nx), "mp_thompson_frozen_reference_temperature")
    rain_reference_density = state.scratch(
        (nz, ny, nx), "mp_thompson_rain_reference_density")
    snow_melt_marker = state.scratch(
        (nz, ny, nx), "mp_thompson_snow_melt_marker")
    graupel_melt_marker = state.scratch(
        (nz, ny, nx), "mp_thompson_graupel_melt_marker")
    snow_velocity_boost = state.scratch(
        (nz, ny, nx), "mp_thompson_snow_velocity_boost")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th[...] = thb + state.thp
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    temperature[...] = th * pii
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]

    surface_shape = (ny, nx)
    rainnc = state.scratch(surface_shape, "mp_rainnc")
    rainncv = state.scratch(surface_shape, "mp_rainncv")
    snownc = state.scratch(surface_shape, "mp_snownc")
    snowncv = state.scratch(surface_shape, "mp_snowncv")
    graupelnc = state.scratch(surface_shape, "mp_graupelnc")
    graupelncv = state.scratch(surface_shape, "mp_graupelncv")
    sr = state.scratch(surface_shape, "mp_sr")
    table_owner = load_classic_device_tables(table_root)
    # Classic Thompson's ng1d is transient but it is part of every call's
    # source/melt/fallout trajectory.  Output cadence may decide whether it is
    # consumed by reflectivity, never whether it exists or evolves.
    graupel_number_shadow = state.scratch(
        (nz, ny, nx), "mp_thompson_graupel_number_shadow")

    # Lifetime-alias the zero/one entry marker with the held-temperature
    # buffer: the column mask consumes it before cloud adjustment overwrites
    # every element with the reference temperature needed by snow fallout.
    cp.greater(
        state.qg, DTYPE(1.0e-12), out=frozen_reference_temperature)
    # The cold source writes its latent heating in-place.  Preserve WRF's
    # entry-temperature branch decision so a cold cell heated across 0 C
    # cannot execute the warm source path again in this same call.  The warm
    # source consumes this mask and overwrites the same dedicated buffer with
    # its held prr_gml > 0 marker.  It writes prr_sml > 0 separately: neither
    # marker may alias the later RHOF output.
    cp.greater_equal(
        temperature, DTYPE(273.15), out=graupel_melt_marker)
    # GRAUPELNCV is a current-call diagnostic.  Unlike RAINNCV/SNOWNCV it
    # has no earlier species kernel that resets it before the graupel slice.
    graupelncv.fill(DTYPE(0.0))
    save_pre_mp_theta(state)
    launch_classic_graupel_number_init(
        state.qg, temperature, state.p, state.qv,
        graupel_number_shadow)
    launch_frozen_vapor_network_from_owner(
        state.qi, state.ni, state.qs, state.qg, state.qr, state.nr,
        temperature, state.p, state.qv, table_owner, dt, qc=state.qc,
        graupel_number_shadow=graupel_number_shadow,
        snow_velocity_boost=snow_velocity_boost)
    launch_warm_frozen_source_network_from_owner(
        state.qc, state.qr, state.nr, state.qs, state.qg,
        graupel_number_shadow, graupel_melt_marker, snow_melt_marker,
        temperature, state.p, state.qv,
        table_owner, dt)
    # WRF's rain velocity pass refreshes RHOF for every level only when the
    # post-source column contains rain.  RAINNCV is not populated until the
    # later ice fallout launch, so it safely carries this held column mask.
    launch_hydrometeor_column_mask(state.qr, rainncv)
    # Cloud fallout is guarded by the held post-source ANY(L_qc), so cloud
    # nucleated by the following saturation adjustment in an otherwise empty
    # column waits until the next microphysics call.  SNOWNCV is overwritten
    # by ice fallout before it becomes a public current-call diagnostic.
    launch_hydrometeor_column_mask(state.qc, snowncv)
    # SR is refreshed only after all fallout, so its 2-D buffer safely carries
    # the zero/one column guard until the graupel launch consumes it.
    launch_graupel_fallout_column_mask(
        frozen_reference_temperature, state.qg, sr)
    launch_cloud_saturation_adjust(
        temperature, state.p, state.qv, state.qc,
        reference_density=frozen_reference_density,
        reference_temperature=frozen_reference_temperature)
    launch_rain_evaporation(
        state.qr, state.nr, temperature, state.p, state.qv, dt,
        reference_density=rain_reference_density,
        graupel_melt_marker=graupel_melt_marker)
    # WRF solve_em passes the physical, full-level grid%w_2 field unchanged
    # through microphysics_driver; Thompson copies w(i,k,j) directly into
    # w1d(k).  gpuwm's matching kts:kte view is the lower full-level slice,
    # not a mass-level average (the upper extra interface is outside kte).
    launch_cloud_sedimentation(
        state.qc, temperature, state.p, state.qv, state.w[:-1], dz, dt,
        reference_density=frozen_reference_density,
        rain_active_columns=rainncv, cloud_active_columns=snowncv)
    launch_ice_sedimentation(
        state.qi, state.ni, temperature, state.p, state.qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=frozen_reference_density)
    launch_snow_sedimentation(
        state.qs, temperature, state.p, state.qv, dz,
        rainnc, rainncv, snownc, snowncv, dt,
        reference_density=frozen_reference_density,
        reference_temperature=frozen_reference_temperature,
        snow_melt_marker=snow_melt_marker,
        melt_rain_qr=state.qr,
        melt_rain_nr=state.nr,
        velocity_boost=snow_velocity_boost,
        accumulate_surface=True)
    launch_graupel_sedimentation(
        state.qg, temperature, state.p, state.qv, dz,
        rainnc, rainncv, graupelnc, graupelncv, dt,
        reference_density=frozen_reference_density,
        active_columns=sr,
        graupel_number_shadow=graupel_number_shadow,
        accumulate_surface=True)
    launch_rain_sedimentation(
        state.qr, state.nr, temperature, state.p, state.qv, dz,
        rainnc, rainncv, dt, reference_density=rain_reference_density,
        accumulate_surface=True)
    launch_final_phase_cleanup(
        state.qc, state.qi, state.ni, temperature, state.p, state.qv)
    launch_classic_graupel_number_finalize(
        state.qg, temperature, state.p, state.qv,
        graupel_number_shadow)
    if refl_10cm_due:
        from gpuwm.core.refl import compute_and_stash_refl_10cm
        compute_and_stash_refl_10cm(
            state, cfg, temperature, state.p,
            thompson_graupel_number=graupel_number_shadow)
    # The effective-radius kernel writer already applies WRF's driver-side
    # metre->micron conversion after its clamps (kernels/thompson.cu), so
    # state.effc/effi/effs receive the radiation-facing MICRON contract
    # directly -- no further adapter-side scaling is permitted here.
    launch_effective_radius(
        temperature, state.p, state.qv, state.qc,
        state.qi, state.ni, state.qs,
        state.effc, state.effi, state.effs)
    th[...] = temperature / pii
    moist_physics_finish(state, cfg, th, dt)
    frozen = snowncv + graupelncv
    sr[...] = cp.where(
        rainncv > DTYPE(1.0e-12),
        cp.minimum(DTYPE(1.0), frozen / rainncv), DTYPE(0.0))
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


def normalize_spec_zone_ring_after_restore(state: DomainState,
                                           cfg: RunConfig) -> None:
    """Restart migration normalization: zero the specified-zone ring of
    every microphysics-owned accumulator/diagnostic slot and of
    h_diabatic after a checkpoint restore.

    Checkpoints written before the ring exclusion landed carry the SAME
    restart format version, and whole-field microphysics could leave them
    with nonzero ring RAINNC/*NCV/SR/refl and a stale ring h_diabatic.
    No WRF-valid trajectory can contain such values: the fields are
    allocator-zero-initialized (frame/module_domain.F:770-777,
    tools/gen_allocs.c:411-415) and every MP tile write is clipped
    (solve_em.F:3631-3639/:4040-4048).  Without this step the guard's
    capture/restore would preserve the stale ring FOREVER, and ring
    h_diabatic would feed every RK stage (dycore.add_h_diabatic_tendency)
    before post-RK microphysics re-pins it.  Ring prognostics (theta,
    moisture) are deliberately NOT touched -- the boundary machinery owns
    and overwrites them every step.  Value-level and idempotent (post-fix
    checkpoints already carry zero rings), so the restart format version
    is unchanged: this function IS the documented migration step for
    pre-fix checkpoints.  Periodic/open domains are untouched.
    """
    if cfg.mp_physics == 0:
        return
    slices = _ring_guard_slices(state, cfg)
    if slices is None:
        return
    for slot in _RING_SURFACE_SLOTS + _RING_VOLUME_SLOTS:
        arr = state.existing_scratch(slot)
        if arr is None:
            continue
        for slc in slices:
            arr[slc] = 0
    if state.h_diabatic is not None:
        for slc in slices:
            state.h_diabatic[slc] = 0


def _dispatch_scheme(state: DomainState, cfg: RunConfig, dt: float, *,
                     refl_10cm_due: bool) -> MicrophysicsDiagnostics:
    """Scheme dispatch exactly as before the ring guard existed."""
    if cfg.mp_physics == 1:
        return _apply_kessler(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 6:
        from gpuwm.core.wsm6 import apply as apply_wsm6
        return apply_wsm6(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 8:
        return _apply_thompson(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    elif cfg.mp_physics == 10:
        # Lazy import leaves the frozen Kessler module/import path intact.
        from gpuwm.core.morrison import apply as apply_morrison
        return apply_morrison(
            state, cfg, dt, refl_10cm_due=refl_10cm_due)
    else:
        # Branch-local imports avoid the microphysics -> nssl2_runtime ->
        # microphysics partial-initialization cycle.
        from gpuwm.core.nssl2_default_hooks import NSSL2ProductionBinding
        from gpuwm.core.nssl2_production_coordinator import (
            NSSL2ProductionConfigurationError,
        )
        from gpuwm.core.nssl2_runtime import apply_nssl2_production

        driver = getattr(state, "physics", None)
        if driver is None or getattr(driver, "state", None) is not state:
            raise NSSL2ProductionConfigurationError(
                "mp_physics=18 requires the DomainState PhysicsDriver")
        binding = getattr(driver, "nssl2_binding", None)
        if not isinstance(binding, NSSL2ProductionBinding):
            raise NSSL2ProductionConfigurationError(
                "mp_physics=18 requires its persistent production binding")
        return apply_nssl2_production(
            state,
            cfg,
            dt,
            binding.hooks,
            output_due=refl_10cm_due,
            radiation_due=True,
            validate_values=False,
            binding=binding,
        )


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False) -> MicrophysicsDiagnostics | None:
    """Apply the configured microphysics to ``state`` over ``dt`` seconds.

    ``cfg.mp_physics = 0`` returns immediately (bitwise no-op); ``1`` runs
    Kessler, ``6`` runs WSM6, ``8`` runs classic Thompson, ``10`` runs
    Morrison two-moment, and ``18`` runs NSSL two-moment (all require
    moisture); anything else raises.
    ``refl_10cm_due`` is gpuwm's history-step
    ``diagflag`` equivalent.  The active adapter computes from its unchanged
    prepared pressure and post-scheme temperature before
    ``moist_physics_finish``.

    On specified/nested domains the call excludes the outermost
    ``spec_zone`` mass ring exactly as WRF's clipped tiles do
    (solve_em.F:3618-3639/:4040-4048; module docstring): the ring is
    captured before dispatch and bit-restored afterwards, ring
    h_diabatic is pinned to WRF's 0, and ring RAINNC/SNOWNC/GRAUPELNC,
    the per-call *NCV/SR diagnostics, and the REFL_10CM stash keep their
    never-written values (exactly 0 from a fresh init).  Periodic/open
    configs (WRF: sz = 0) keep the whole-field call bitwise.

    A due diagnostic with ``mp_physics = 0`` is an invalid schedule and
    raises instead of silently producing an output frame without radar data.
    """
    if cfg.mp_physics == 0:
        if refl_10cm_due:
            raise ValueError("REFL_10CM is due without active microphysics")
        return None
    if cfg.mp_physics not in (1, 6, 8, 10, 18):
        raise ValueError(f"unknown mp_physics={cfg.mp_physics} "
                         "(0 = none, 1 = Kessler, 6 = WSM6, "
                         "8 = Thompson, 10 = Morrison, "
                         "18 = NSSL)")
    if state.qv is None:
        raise ValueError(f"mp_physics={cfg.mp_physics} requires "
                         "cfg.moist=True: the state "
                         "carries no qv/qc/qr arrays")
    slices = _ring_guard_slices(state, cfg)
    if slices is None:
        return _dispatch_scheme(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    saved, captured_slots = _capture_spec_zone_ring(state, slices)
    try:
        # The restore must run on EVERY exit path: a post-mutation raise
        # (e.g. the reflectivity handoff refusing a missing driver or an
        # unconsumed stash, refl.py stash_refl_10cm) would otherwise leave
        # ring columns mutated that WRF never dispatches at all.
        return _dispatch_scheme(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    finally:
        _restore_spec_zone_ring(state, slices, saved, captured_slots)
