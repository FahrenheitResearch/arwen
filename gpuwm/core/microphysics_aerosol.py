"""Forecast adapter for WRF v4.6.1 Thompson aerosol-aware (``mp_physics=28``).

:func:`_apply_thompson_aerosol` is a **sibling** of
``gpuwm.core.microphysics._apply_thompson``, not a branch inside it.  That is
deliberate and load-bearing: the classic function body stays textually
diffable against its model-validated form, so ``tests/test_mp8_frozen.py``'s
receipts and any future ``git diff`` can prove mp=8 was not touched by this
port.  The cost is a copied skeleton; the benefit is that "mp=8 is frozen" is
a statement about bytes rather than about control flow.

Numerical authority is ``wrf461-pristine/phys/module_mp_thompson.F``
(WRF v4.6.1, commit ``d66e442``).  Bare ``:NNNN`` line numbers refer to it.

THE FIVE THINGS THIS FILE IS RESPONSIBLE FOR
--------------------------------------------
Every kernel below was unit-gated by its own package against the Fortran
oracle before this adapter existed.  Unit gates cannot see composition
defects, so the five properties this file uniquely owns are stated here and
each is pinned by a named test in ``tests/test_thompson_aerosol_adapter.py``:

1.  **The accumulator contract.**  ``state.nc`` / ``state.nwfa`` /
    ``state.nifa`` are READ-ONLY entry state for the whole call.  The three
    scratch accumulators are ZEROED at entry (they are persistent scratch --
    ``state.scratch`` survives across steps by design, ``state.py:710-735`` --
    so carrying last step's aerosol tendency forward would be a slow,
    plausible-looking drift no single-step column test could catch), written
    by every aerosol kernel, and applied EXACTLY ONCE by
    ``launch_aerosol_state_finalize`` (:3972-4021), which carries WRF's only
    clamps.
    Pinned by ``test_entry_state_is_read_only_until_the_single_terminal_apply``
    and ``test_accumulators_are_zeroed_at_entry_not_carried_between_calls``.

2.  **Warm/cold entry-mask disjointness.**  The cold network returns for
    ``T >= 273.15`` and the warm network returns unless its held mask is set,
    and both kernel headers state that the two gates are exact complements.
    That is only true if the mask is captured on the **entry** temperature,
    BEFORE the cold network writes its latent heating in place.  Capturing it
    afterwards lets a cold cell that the cold network heated across 0 C run
    the warm path as well, double-counting ``pnc_wau``/``pnc_rcw``/
    ``pna_rca``/``pnd_rcd`` at that level -- finite, stable and wrong.
    Pinned by ``test_warm_entry_mask_is_captured_before_the_cold_network``.

3.  **Launch order.**  Surface emission is the LAST aerosol write of the call
    and follows the terminal apply (mp_gt_driver:1310-1327 runs after
    ``mp_thompson`` returns, so it is deliberately unclamped).  The ``ncten``
    balance limiter (:2996-3019) runs ONCE, between the warm network and the
    saturation adjustment -- WRF applies it once per column after every
    ``ncten`` source and before ``pnc_wcd``; calling it from inside both
    networks double-applies it.
    Pinned by ``test_launcher_call_order_is_exactly_the_wrf_driver_order``.

4.  **No Cooper.**  In mp=28 ``iceDeMott`` REPLACES Cooper nucleation
    (:2537-2551 selects one or the other on ``is_aerosol_aware``).  The
    classic ``launch_frozen_vapor_network*`` / ``launch_ice_nucleation``
    launchers carry Cooper and must never appear on this path; using them
    would ADD a second ice source rather than substitute for it.
    Pinned by ``test_no_cooper_bearing_classic_launcher_is_ever_called``.

5.  **Which density feeds which kernel.**  WRF keeps TWO air densities in
    play across the condensation seam and uses both.  :3193 diagnoses the
    TAU+1 density; :3242-3243 builds the working rain mass and number from
    it and :3384-3388 freezes ``ilamr``/``N0_r`` from those; :3490 then
    OVERWRITES ``rho(k)`` inside the condensation loop and :3505-3520's
    ``orho``/``rhof``/``vsc2``/``rvs`` read the newer one.  ``prv_rev`` and
    ``pnr_rev`` therefore scale with the OLDER density while everything
    around them scales with the newer.  Both kernels expose exactly what
    they need -- ``launch_aerosol_saturation_adjust`` writes the :3193
    density into ``reference_density`` unconditionally, and
    ``launch_aerosol_rain_evaporation`` takes it as ``entry_density`` -- so
    the only thing that can get this wrong is the wiring in this file, and
    getting it wrong scales rain evaporation by ``rho_post/rho_pre``: 1.9e-3
    on aero-reduces-to-classic, which was the port's ONLY carved-out
    end-to-end tolerance until WP-12a.
    Pinned by
    ``test_the_adapter_hands_rain_evaporation_wrfs_pre_condensation_density``
    and ``test_suppressing_the_pre_condensation_density_reproduces_the_old_
    defect``.

5b. **And which density SEDIMENTATION reads, which is a third decision.**
    ``launch_rain_sedimentation`` forms its sedimenting rain mass and number
    as ``qr*rho`` and ``nr*rho`` from the ``reference_density`` it is handed;
    those are WRF's ``rr(k)``/``nr(k)`` at :3794-3795, and WRF builds them in
    exactly two places -- :3237-3238 from the :3193 TAU+1 density at every
    level with ``L_qr``, and :3568-3570 from the :3490 post-condensation
    density ONLY where the :3501-3502 rain-evaporation gate passed.  So the
    correct buffer is a LEVEL-WISE MIXTURE, not either density.
    ``launch_aerosol_rain_evaporation`` produces exactly that mixture in its
    ``reference_density`` output: it writes ``entry_density`` by default and
    overwrites with its own post-condensation rho after all three of WRF's
    gates pass.  Before WP-13a it wrote the post-condensation rho
    unconditionally, which gave every level the :3568 answer including the
    ones WRF never rewrote -- worth 5.165e-04 on aero-drop-evap's RAINNC
    (whose column passes the gate at NO level, so the :3237 pair should have
    stood everywhere) and 1.279e-04 on aero-ice-demott-idxin's.  Both
    fixtures now clear the flat G3 gate.
    Pinned by ``test_rain_sedimentation_gets_wrfs_level_wise_working_rain_
    density`` and, at the kernel, by tests/test_thompson_aerosol_sat_gpu.py::
    ``test_rain_evaporation_exports_the_sedimentation_density_wrf_actually_
    used``.

    WHAT IS STILL WRONG THERE AND IS NOT THIS PACKAGE'S TO FIX: the frozen
    mp=8 kernel re-applies WRF's :3240-3250 ``mvd_r`` clamp at sedimentation
    time (thompson.cu:449-453), which :3568-3570 does not carry, and gates
    rain presence on a MIXING RATIO (thompson.cu:438) where :3616 tests a
    MASS CONCENTRATION.  Both are measured on this tree by
    ``test_the_two_residuals_that_live_in_the_frozen_kernel_are_measured_
    here``; both are ArWen-wide (microphysics.py wires mp=8 the same way) and
    correcting them means re-validating the mp=8 trajectory against its 92
    classic fixtures first.

WHAT IS REUSED FROM THE FROZEN mp=8 MODULE, AND WHY THAT IS SOUND
-----------------------------------------------------------------
Eight launchers come from ``gpuwm.core.thompson`` unchanged:

* ``launch_classic_graupel_number_init`` / ``_finalize`` -- ``is_hail_aware``
  is false for mp=8 and mp=28 alike, so the ``idx_bg1 = 5`` / ``rho_g = 400``
  classic graupel-number diagnostic is bit-identical.
* ``launch_hydrometeor_column_mask`` (twice) and
  ``launch_graupel_fallout_column_mask`` -- pure column reductions over mass.
* ``launch_ice_sedimentation`` / ``_snow_`` / ``_graupel_`` / ``_rain_`` --
  :3790-3936 contains no ``is_aerosol_aware`` branch and no nc/nwfa/nifa
  reference.  ``nwfa`` and ``nifa`` have NO sedimentation term anywhere in
  module_mp_thompson.F; any implementation that adds one is wrong.

They are called with the identical ordered argument tuples mp=8 uses, which
``test_reused_classic_launchers_receive_the_mp8_argument_shape`` pins.

ENTRY-STATE ALIASING (why there are no ``nc_entry`` scratch copies)
-------------------------------------------------------------------
``launch_aerosol_entry_cloud_number`` reproduces :1826-1848 including its
in-place side effect: ``qc1d`` and ``nc1d`` are ZEROED wherever
``qc1d <= R1`` (:1844-1845).  After that call ``state.nc`` *is* WRF's
``nc1d``, and nothing writes ``state.nc`` again until the terminal apply at
step 13 -- so ``state.nc`` itself is passed as the ``nc_entry`` argument
everywhere, with no copy.  The same holds for ``state.nwfa``/``state.nifa``,
which no kernel between the entry snapshot and the finalize writes at all.
``gpuwm/core/preflight.py``'s mp=28 slot registry reflects exactly this: it
budgets ``qc_entry`` and ``ni_entry`` copies and no ``nc_entry`` copy.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.microphysics import (
    MicrophysicsDiagnostics,
    moist_physics_finish,
    save_pre_mp_theta,
)
from gpuwm.core.state import DTYPE, DomainState
from gpuwm.physics_compat import thompson_table_root as _thompson_table_root

#: module_mp_thompson.F:183.  The entry threshold that decides whether a
#: level carries a species at all.
R1 = 1.0e-12

#: The state fields an mp=28 call requires.  ``nwfa2d``/``nifa2d`` are checked
#: separately because they are ``(ny, nx)`` and are INTENT(IN) to WRF.
REQUIRED_STATE_FIELDS = (
    "qc", "qi", "ni", "qs", "qg", "qr", "nr", "nc", "nwfa", "nifa",
)

#: The two 2-D surface emission constants (Registry ``QNWFA2D``/``QNIFA2D``,
#: # kg-1 s-1).  microphysics never writes them.
REQUIRED_SURFACE_FIELDS = ("nwfa2d", "nifa2d")

#: Every ``mp_thompson_aero_*`` scratch slot this adapter draws, in the order
#: it draws them.  Must equal ``gpuwm/core/preflight.py``'s mp=28 aerosol
#: working set exactly -- a slot this adapter creates that preflight does not
#: budget is an unpriced allocation behind the arena gate, and a slot
#: preflight budgets that this adapter never draws is dead reserved memory.
#: ``test_scratch_slots_match_the_preflight_registry_exactly`` pins both
#: directions.
AEROSOL_SCRATCH_SLOTS = (
    "mp_thompson_aero_ncten",
    "mp_thompson_aero_nwfaten",
    "mp_thompson_aero_nifaten",
    "mp_thompson_aero_entry_density",
    "mp_thompson_aero_nwfa_entry_m3",
    "mp_thompson_aero_nifa_entry_m3",
    "mp_thompson_aero_tau1_density",
    "mp_thompson_aero_nwfa_work_m3",
    "mp_thompson_aero_qc_entry",
    "mp_thompson_aero_ni_entry",
    "mp_thompson_aero_rc_entry",
    "mp_thompson_aero_nc_entry_m3",
    "mp_thompson_aero_nu_c_entry",
    "mp_thompson_aero_l_qc_entry",
    "mp_thompson_aero_condensation_rate",
)

#: The two int32 members of that set (the entry droplet diagnosis's shape
#: parameter and its ``L_qc`` flag).  Both are 4 bytes per element, so the
#: registry's shape-only accounting is unaffected by the dtype.
AEROSOL_INT_SCRATCH_SLOTS = (
    "mp_thompson_aero_nu_c_entry",
    "mp_thompson_aero_l_qc_entry",
)


def _apply_thompson_aerosol(
        state: DomainState, cfg: RunConfig, dt: float, *,
        refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """One complete ``mp_physics=28`` microphysics call, in WRF driver order.

    The skeleton is ``gpuwm.core.microphysics._apply_thompson`` with the
    aerosol network substituted; every deviation is annotated with the WRF
    line that forces it.  See the module docstring for the four composition
    properties this function is uniquely responsible for.
    """
    table_root = _thompson_table_root()
    missing = [name for name in REQUIRED_STATE_FIELDS
               if getattr(state, name, None) is None]
    missing += [name for name in REQUIRED_SURFACE_FIELDS
                if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "Thompson mp=28 state lacks " + ", ".join(missing))

    # Frozen mp=8 launchers, reused byte-for-byte.  Imported here (not at
    # module scope) for the same reason _apply_thompson does: it keeps the
    # import graph acyclic and lets a call-recording test monkeypatch the
    # owning module's attribute.
    from gpuwm.core.thompson import (
        launch_classic_graupel_number_finalize,
        launch_classic_graupel_number_init,
        launch_graupel_fallout_column_mask,
        launch_graupel_sedimentation,
        launch_hydrometeor_column_mask,
        launch_ice_sedimentation,
        launch_rain_sedimentation,
        launch_snow_sedimentation,
    )
    from gpuwm.core.thompson_aerosol_cold import (
        launch_aa_cold_network_from_owner,
    )
    from gpuwm.core.thompson_aerosol_runtime import (
        device_drop_evaporation_number_table,
        load_aerosol_device_tables,
    )
    from gpuwm.core.thompson_aerosol_sat import (
        launch_aerosol_rain_evaporation,
        launch_aerosol_saturation_adjust,
    )
    from gpuwm.core.thompson_aerosol_sed import (
        launch_aa_cloud_sedimentation,
        launch_aa_final_phase_cleanup,
    )
    from gpuwm.core.thompson_aerosol_state import (
        launch_aerosol_effective_radius,
        launch_aerosol_entry_cloud_number,
        launch_aerosol_entry_snapshot,
        launch_aerosol_state_finalize,
        launch_aerosol_surface_emission,
        launch_aerosol_working_number,
        launch_tau1_density,
        zero_aerosol_accumulators,
    )
    from gpuwm.core.thompson_aerosol_warm import (
        launch_aerosol_warm_source_network_from_owner,
        launch_ncten_balance,
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

    # Both table owners resolve from the SAME root.  A second root would let
    # tnccn_act and the four classic caches come from different WRF builds.
    table_owner = load_classic_device_tables(table_root)
    aerosol_table_owner = load_aerosol_device_tables(table_root)
    drop_evaporation_number = device_drop_evaporation_number_table(
        table_owner)

    graupel_number_shadow = state.scratch(
        (nz, ny, nx), "mp_thompson_graupel_number_shadow")

    # --- the mp=28 aerosol working set (preflight budgets each of these) ---
    ncten = state.scratch((nz, ny, nx), "mp_thompson_aero_ncten")
    nwfaten = state.scratch((nz, ny, nx), "mp_thompson_aero_nwfaten")
    nifaten = state.scratch((nz, ny, nx), "mp_thompson_aero_nifaten")
    entry_density = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_entry_density")
    nwfa_entry_m3 = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_nwfa_entry_m3")
    nifa_entry_m3 = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_nifa_entry_m3")
    tau1_density = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_tau1_density")
    nwfa_work_m3 = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_nwfa_work_m3")
    qc_entry = state.scratch((nz, ny, nx), "mp_thompson_aero_qc_entry")
    ni_entry = state.scratch((nz, ny, nx), "mp_thompson_aero_ni_entry")
    rc_entry = state.scratch((nz, ny, nx), "mp_thompson_aero_rc_entry")
    nc_entry_m3 = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_nc_entry_m3")
    nu_c_entry = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_nu_c_entry", dtype=np.int32)
    l_qc_entry = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_l_qc_entry", dtype=np.int32)
    condensation_rate = state.scratch(
        (nz, ny, nx), "mp_thompson_aero_condensation_rate")

    # (1) ACCUMULATOR ZEROING, :1679-1681.  Explicit and unconditional: these
    # are persistent scratch slots and every one of them is read before it is
    # written somewhere in the call (the balance limiter OVERWRITES ncten, the
    # working-number refresh READS nwfaten, the terminal apply reads all
    # three), so a carried-over value from the previous step is a live
    # physics input, not a harmless stale buffer.
    zero_aerosol_accumulators(ncten, nwfaten, nifaten)

    # Lifetime-alias the zero/one entry-graupel marker with the held-
    # temperature buffer, exactly as mp=8 does: the column mask consumes it
    # before the saturation adjustment overwrites every element with the
    # reference temperature snow fallout needs.
    cp.greater(
        state.qg, DTYPE(1.0e-12), out=frozen_reference_temperature)

    # (2) THE WARM ENTRY MASK, captured on the ENTRY temperature and BEFORE
    # the cold network writes its latent heating in place.  See the module
    # docstring; this single line is the disjointness guarantee both kernel
    # headers depend on.  The warm network consumes it and overwrites the
    # same buffer with WRF's held ``prr_gml > 0`` decision.
    cp.greater_equal(
        temperature, DTYPE(273.15), out=graupel_melt_marker)

    # GRAUPELNCV is a current-call diagnostic with no earlier species kernel
    # to reset it.
    graupelncv.fill(DTYPE(0.0))
    save_pre_mp_theta(state)

    # ---- 1. entry snapshot (:1795-1812) and entry droplet diagnosis -------
    # (:1826-1848).  The second call ZEROES state.qc and state.nc wherever
    # qc <= R1, which is what makes state.nc a legitimate "nc1d" for every
    # later kernel.
    launch_aerosol_entry_snapshot(
        temperature, state.p, state.qv, state.nwfa, state.nifa,
        entry_density, nwfa_entry_m3, nifa_entry_m3)
    launch_aerosol_entry_cloud_number(
        state.qc, state.nc, entry_density,
        rc_entry, nc_entry_m3, nu_c_entry, l_qc_entry)
    # The ncten balance limiter (:2996-3019) needs BOTH the entry and the
    # post-source cloud mass, so the entry value has to be held.
    qc_entry[...] = state.qc
    # :1870-1871 zeroes ni1d wherever qi1d <= R1.  mp=28's only consumer of
    # that zeroing is the final phase cleanup's melt credit
    # ``ncten += ni1d*odt`` (:3949): every other reader of the ice number
    # gates on qi > R1 first, so the zeroing is inert for them.  It is
    # therefore applied to a HELD copy rather than to state.ni, which keeps
    # the four reused classic fallout launchers seeing exactly the field mp=8
    # gives them.
    ni_entry[...] = cp.where(state.qi > DTYPE(R1), state.ni, DTYPE(0.0))

    # ---- 2. classic graupel number (is_hail_aware false for 8 and 28) -----
    launch_classic_graupel_number_init(
        state.qg, temperature, state.p, state.qv,
        graupel_number_shadow)

    # ---- 3. the cold source network --------------------------------------
    launch_aa_cold_network_from_owner(
        state.qi, state.ni, state.qs, state.qg, state.qr, state.nr,
        state.qc, temperature, state.p, state.qv,
        state.nc, state.nwfa, state.nifa,
        ncten, nwfaten, nifaten,
        graupel_number_shadow, snow_velocity_boost, table_owner, dt)

    # ---- 4. the warm source network --------------------------------------
    launch_aerosol_warm_source_network_from_owner(
        state.qc, state.qr, state.nr, state.qs, state.qg,
        graupel_number_shadow, graupel_melt_marker, snow_melt_marker,
        temperature, state.p, state.qv,
        state.nc, state.nwfa, state.nifa,
        ncten, nwfaten, nifaten, table_owner, dt)

    # ---- 5. the ncten balance limiter, ONCE (:2996-3019) ------------------
    # After every ncten source, before the saturation adjustment's pnc_wcd.
    # It OVERWRITES ncten where a clamp fires; running it twice is finite,
    # plausible and wrong.  ``entry_density`` is WRF's :1802 density, not a
    # density rediagnosed from the mutated temperature and vapour.
    launch_ncten_balance(
        qc_entry, state.qc, state.nc, entry_density, ncten, dt)

    # ---- 6. the three column masks, unchanged from mp=8 -------------------
    launch_hydrometeor_column_mask(state.qr, rainncv)
    launch_hydrometeor_column_mask(state.qc, snowncv)
    launch_graupel_fallout_column_mask(
        frozen_reference_temperature, state.qg, sr)

    # ---- 7. the working aerosol number (:3189-3193, :3211) ----------------
    # The TAU+1 density, recomputed from the post-source temperature and
    # vapour -- a genuinely different density from ``entry_density``.
    launch_tau1_density(temperature, state.p, state.qv, tau1_density)
    launch_aerosol_working_number(
        state.nwfa, nwfaten, tau1_density, dt, nwfa_work_m3)

    # ---- 8/9. condensation + CCN activation, then rain evaporation --------
    # WRF passes w1d(k) = w(i,k,j) once at mp_gt_driver:1224 with no
    # averaging, so the lower full-level slice is the exact analogue.
    launch_aerosol_saturation_adjust(
        temperature, state.p, state.qv, state.qc, state.nc, ncten, nwfaten,
        nwfa_work_m3, state.w[:-1],
        aerosol_table_owner.ccn_activation_table, drop_evaporation_number,
        dt,
        reference_density=frozen_reference_density,
        reference_temperature=frozen_reference_temperature,
        condensation_rate=condensation_rate)
    # TWO DENSITIES, AND WRF USES BOTH.  :3242-3243 forms the working rain
    # mass and number from the TAU+1 density diagnosed at :3193 -- BEFORE the
    # condensation block -- and :3384-3388 freezes ilamr/N0_r from them.
    # :3490 then OVERWRITES rho(k) inside the condensation loop, and
    # :3505-3520's orho/rhof/vsc2/rvs all read that post-condensation value.
    # Passing only the post-condensation density (the mp=8 kernel's single
    # density, and this adapter's behaviour before WP-12a) scales prv_rev and
    # pnr_rev by rho_post/rho_pre.  ``launch_aerosol_saturation_adjust``
    # writes exactly WRF's :3193 density into ``reference_density``,
    # unconditionally and at every level, before its own gate -- so the
    # buffer above already holds it and no extra scratch slot is needed.
    launch_aerosol_rain_evaporation(
        state.qr, state.nr, temperature, state.p, state.qv, nwfaten, dt,
        reference_density=rain_reference_density,
        graupel_melt_marker=graupel_melt_marker,
        condensation_rate=condensation_rate,
        entry_density=frozen_reference_density)

    # ---- 10. fallout: cloud from mp=28, the other four REUSED from mp=8 ---
    launch_aa_cloud_sedimentation(
        state.qc, state.nc, ncten, temperature, state.p, state.qv,
        state.w[:-1], dz, dt,
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
    # THE THIRD DENSITY DECISION, and it is not the same as the one above.
    # This kernel builds WRF's rr(k)/nr(k) (:3794-3795) as qr*rho / nr*rho
    # from the buffer below, and WRF builds those at :3237-3238 from the :3193
    # TAU+1 density EVERYWHERE and rebuilds them at :3568-3570 from the :3490
    # post-condensation density ONLY inside the :3501-3502 gate.
    # ``rain_reference_density`` is exactly that level-wise mixture: the rain
    # evaporation kernel writes ``entry_density`` into it by default and
    # overwrites with its own rho once all three of WRF's gates pass.  See
    # note 5b in the module docstring for what handing it a single density
    # costs (5.165e-04 on aero-drop-evap's RAINNC).
    launch_rain_sedimentation(
        state.qr, state.nr, temperature, state.p, state.qv, dz,
        rainnc, rainncv, dt, reference_density=rain_reference_density,
        accumulate_surface=True)

    # ---- 11. number-conserving phase cleanup (:3943-3966) -----------------
    launch_aa_final_phase_cleanup(
        state.qc, state.qi, state.ni, temperature,
        state.nc, ni_entry, ncten, state.p, state.qv, dt)

    # ---- 12. classic graupel-number finalize ------------------------------
    launch_classic_graupel_number_finalize(
        state.qg, temperature, state.p, state.qv,
        graupel_number_shadow)

    # ---- 13. THE single terminal apply and clamp (:3972-4021) -------------
    # The only place in the whole call that writes nc/nwfa/nifa from the
    # accumulators.  Each output aliases its own input, which the kernel
    # supports (every thread reads its element before writing it).
    launch_aerosol_state_finalize(
        state.qc, state.nc, state.nwfa, state.nifa,
        ncten, nwfaten, nifaten, entry_density, dt,
        state.nc, state.nwfa, state.nifa)

    # ---- 14. reflectivity, if due ----------------------------------------
    if refl_10cm_due:
        from gpuwm.core.refl import compute_and_stash_refl_10cm
        # calc_refl10cm (:5710) takes no nc argument and never re-reads rc
        # after :5764, so cloud water and droplet number contribute exactly
        # zero: this is the mp=8 path with the mp=8 arguments.
        compute_and_stash_refl_10cm(
            state, cfg, temperature, state.p,
            thompson_graupel_number=graupel_number_shadow)

    # ---- 15. effective radii, with the PROGNOSTIC droplet number ----------
    # The kernel applies WRF's mp_gt_driver:1476-1478 clamps and then the
    # metre->micron multiply, so state.effc/effi/effs receive gpuwm's
    # radiation-facing MICRON contract directly.
    launch_aerosol_effective_radius(
        temperature, state.p, state.qv, state.qc, state.nc,
        state.qi, state.ni, state.qs,
        state.effc, state.effi, state.effs)

    # ---- 16. surface emission: LAST aerosol write, deliberately unclamped -
    # mp_gt_driver:1310-1327 runs it AFTER mp_thompson has applied its
    # terminal ceiling, so between here and the next call's entry pack
    # (:1805-1806) nwfa/nifa may legitimately exceed 9999.E6.  Adding a
    # ceiling here would silently change the boundary-layer aerosol budget on
    # every step.
    launch_aerosol_surface_emission(
        state.nwfa, state.nifa, state.nwfa2d, state.nifa2d, dt)

    # ---- 17. the unchanged prep/finish bracket and the SR diagnostic ------
    th[...] = temperature / pii
    moist_physics_finish(state, cfg, th, dt)
    # mp_gt_driver:1308, transcribed exactly:
    #     SR(i,j) = (pptsnow + pptgraul + pptice)/(RAINNCV(i,j)+1.e-12)
    # ``snowncv`` is :1301's ``pptsnow + pptice`` and ``graupelncv`` is
    # :1305's ``pptgraul``, so the numerator is WRF's.  The denominator
    # carries WRF's OWN epsilon rather than a guarded division: WRF has no
    # threshold and no MIN here, and the epsilon is not a divide-by-zero
    # guard that a ``where`` could stand in for -- it is part of the value.
    # RAINNCV is pptrain plus the same three frozen terms, so an all-frozen
    # column has SR = f/(f+1e-12) < 1 in WRF, and a guarded f/f returns
    # exactly 1.  On the committed fixtures that is worth 3.701e-04
    # (aero-ice-demott-dep, RAINNCV 2.70e-09 mm) and 3.113e-04
    # (aero-ice-koop, RAINNCV 3.21e-09 mm) -- two orders above the port's
    # 2e-06 gate, on a field the WRF history stream publishes.  The MIN is
    # dropped with it: the numerator is a subset of the denominator's terms,
    # all non-negative, so the ratio cannot exceed 1 and WRF does not clamp
    # it.  Pinned by tests/test_thompson_aerosol_adapter.py::
    # test_the_sr_diagnostic_is_wrfs_epsilon_quotient_not_a_guarded_ratio.
    frozen = snowncv + graupelncv
    sr[...] = frozen / (rainncv + DTYPE(1.0e-12))
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


def thompson_aerosol_init_fill(state: DomainState, cfg: RunConfig) -> dict:
    """``thompson_init``'s synthetic CCN/IN profile, ONCE per domain.

    Invoke from the physics INIT path, never per step.  WRF calls
    ``thompson_init`` from ``phys/module_physics_init.F:4517-4544`` at domain
    construction; nothing in ``mp_gt_driver`` ever refills the profile, and a
    per-step call would overwrite an advected, scavenged aerosol field with
    the synthetic one every step while leaving every bound intact.

    Returns ``{'ccn': bool, 'in': bool}`` -- which of the two independent
    fills ran.  A domain can legitimately get one and not the other:
    ``thompson_init`` makes the CCN decision at :493 and the IN decision at
    :530 from two separate ``MAXVAL`` reductions.

    THE HEIGHT FIELD.  ``hgt`` is ``z8w[:nz]``, the FULL (w) level
    geopotential height above sea level.  WRF's argument is named ``z_at_q``
    but ``dyn_em/start_em.F:870-876`` fills it from the Z-staggered
    ``ph_2 + phb``, so ``hgt(i,1,j)`` is the terrain elevation the ABSOLUTE
    1000 m / 2500 m ``h_01`` thresholds test.  See
    ``gpuwm.core.thompson_aerosol_state``'s module docstring.

    THE TILE-BOUND DIFFERENCE, stated rather than silently absorbed.  WRF's
    presence test reduces over ``nwfa(its:ite-1,:,jts:jte-1)`` (:489/:526)
    while its fill loop covers ``i = its..min(ide-1,ite)`` (:499-500) -- the
    test excludes the last mass column and row that the fill then writes.
    That asymmetry is in WRF, not a transcription slip.  gpuwm's ``(ny, nx)``
    arrays are the mass grid with no staggered extra, so
    :func:`~gpuwm.core.thompson_aerosol_state.aerosol_profile_needs_fill`
    reduces over the WHOLE mass grid and therefore INCLUDES the one column
    and one row WRF's test omits.  The two can only disagree if a domain has
    aerosol in exactly that outermost column/row and nowhere else, in which
    case gpuwm declines the fill and WRF performs it.  gpuwm's behaviour is
    the safer of the two (it never overwrites real aerosol data), and the
    difference is unreachable for the only supported v1 ingest, which is
    all-zero or all-filled.
    """
    if cfg.mp_physics != 28:
        raise ValueError(
            "thompson_aerosol_init_fill is the mp_physics=28 profile fill, "
            f"got mp_physics={cfg.mp_physics}")
    missing = [name for name in ("nwfa", "nifa", "nwfa2d")
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "mp=28 aerosol profile fill needs " + ", ".join(missing))

    from gpuwm.core.thompson_aerosol_state import (
        aerosol_profile_needs_fill,
        launch_aerosol_init_profile,
    )

    nz, ny, nx = state.p.shape
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    # z8w[:nz] is a leading-axis slice of a C-contiguous array, so it is
    # itself C-contiguous and needs no copy.
    hgt = z8w[:nz]

    fill_ccn = aerosol_profile_needs_fill(state.nwfa)
    fill_in = aerosol_profile_needs_fill(state.nifa)
    if fill_ccn or fill_in:
        launch_aerosol_init_profile(
            hgt, state.nwfa, state.nifa, state.nwfa2d,
            fill_ccn=fill_ccn, fill_in=fill_in)
    return {"ccn": bool(fill_ccn), "in": bool(fill_in)}


__all__ = [
    "AEROSOL_INT_SCRATCH_SLOTS",
    "AEROSOL_SCRATCH_SLOTS",
    "REQUIRED_STATE_FIELDS",
    "REQUIRED_SURFACE_FIELDS",
    "_apply_thompson_aerosol",
    "thompson_aerosol_init_fill",
]
