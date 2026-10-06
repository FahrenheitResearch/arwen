"""The RUC coupling layer: WRF's ``LSMRUC`` seam, column by column.

``module_surface_driver.F:3438-3596`` is not a bare ``CALL LSMRUC``.  The
``CASE (RUCLSMSCHEME)`` arm owns four things that live between WRF's grid
arrays and the RUC driver, and every one of them changes an answer:

* the sea-ice snow-free albedo override (:3453-3459): ``ALBBCK =
  SEAICE_ALBEDO_DEFAULT`` wherever ``XICE_THRESHOLD <= XICE <= 1``, applied
  BEFORE the call, unconditionally -- it is not inside the
  ``FRACTIONAL_SEAICE`` block;
* the argument binding itself, which is where WRF's naming traps live.  The
  driver passes ``dz8w`` into an argument named ``z3d`` and ``p_phy`` -- the
  layer-MID pressure -- into one named ``p8w`` (:3506).  Reading ``p8w`` as
  an interface pressure is a silent O(dp/2) bias in every saturation
  humidity RUC computes, because ``qsg = qsn(soilt,tbq)/(p8w*1e-2)``;
* ``GSW``, which RUC takes as the ABSORBED shortwave flux, not the
  downward one.  WRF builds it in the radiation driver as
  ``gsw = swdown*(1-albedo)`` (``module_radiation_driver.F:3660``); and
* the post-call ``CQS``/``CHS`` rebuild (:3580-3585) and
  ``SFCDIAGS_RUCLSM`` (:3587).  RUC does not use WRF's ordinary SFCDIAGS.
  It has its own 2-m diagnostic, ``module_sf_sfcdiags_ruclsm.F``, which
  clamps T2 between TSK and the first level, builds Q2 from a QSFC PROXY
  reconstructed out of QFX rather than from QSFC itself, and saturation-caps
  the result.  Substituting Noah's SFCDIAGS here would be exactly the
  "plausible numbers, wrong scheme" failure that
  :data:`gpuwm.core.physics.LAND_SURFACE_SFCDIAGS_SCHEMES` exists to prevent,
  so RUC is deliberately absent from that set and this module carries its
  own.

So this module is that seam, with the anchors next to the arithmetic.  The
driver below it is :func:`gpuwm.core.ruc.ruc_land_surface_step`, which is
``LSMRUC`` itself (``phys/module_sf_ruclsm.F:84-1175``) and is oracle-matched
against the byte-unmodified module.

Where the column runs
---------------------
On the CARD, in FP32.  :func:`ruc_lsm_step` runs :mod:`gpuwm.core.ruc_fused`:
six full-width kernels per deterministic call (the WRF surface-driver seam and
LSMRUC's
prologue, the three ``sfctmp`` stages, the epilogue with SFCDIAGS, and a
commit that writes the fields only when no check failed), one read of the
flag words at the end, and SFCDIAGS's two power expressions on the host,
through glibc's ``powf`` on every host (:func:`sfcdiags_exner_powers`).
Enabled ``spp_lsm=1`` uses the retained device-resident orchestration
with the historical WRF hydraulic operator between soil-property and moisture
transport calls; its arrays remain on the GPU. The disabled fused path is
unchanged. The ``sfctmp`` stages are generated from the array
orchestration by ``tools/ruc_fused/gen_sfctmp.py`` and call ``ruc.cu``'s
leaves as device functions, so the column arithmetic has one source.

:func:`_ruc_lsm_step_reference` keeps that orchestration (the masked
``sfctmp`` dispatch over :data:`gpuwm.core.ruc_gpu.RUC_SFCTMP_DEVICE_LEAVES`
with the prologue and epilogue as whole-axis array code) as the oracle every
output word of the fused call is tested against, call after call, at both
soil geometries (``tests/test_ruc_lsm_fused.py``,
``tests/test_ruc_sfctmp_fused.py``).  There is no switch that selects the
host leaf set: it exists for the oracle suite, which runs on a machine with
no card, and a forecast that quietly took it would be a thousand times
slower with no signal that it had.  ``tests/test_ruc_runtime.py`` gates the
import edge directly so it cannot rot back.

The published option identity
-----------------------------
:data:`gpuwm.config.RUC_OPTION_IDENTITY` is the enforced part -- one accepted
value per namelist knob, refused by ``validate_run_config`` before a run
starts.  :data:`RUC_RUNTIME_RESTRICTIONS` is the rest: every restriction the
identity schema has no field for.  The first entry is the one a user is most
likely to be bitten by and is not a namelist knob at all -- it is a
compile-time branch of WRF that gpuwm's own core would have taken.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gpuwm.checkpoint_identity import LAND_SURFACE_ALGORITHM_IDENTITIES
from gpuwm.core.noahmp_libm import powf_array
from gpuwm.core.ruc import (RUC_DRIVER_ARW_FORCING,
                            RUC_DRIVER_COLUMN_FORCING,
                            RUC_DRIVER_COLUMN_STATE,
                            RUC_DRIVER_PROFILE_STATE,
                            RUC_SNOW_COVER_OPTION,
                            load_ruc_parameters, ruc_land_surface_step,
                            ruc_initialize_cold_start, ruc_soil_geometry)
from gpuwm.core.ruc_contract import (NUM_SOIL_LAYERS,
                                     RUC_PACKAGE_STATE_FIELDS,
                                     RUC_SOIL_LEVELS_M,
                                     WRF_SUPPORTED_NUM_SOIL_LAYERS)
from gpuwm.core.surface_forcing import SurfacePrecipitationForcing

#: The vegetation dataset identifier gpuwm's land-use ingest produces, and the
#: only one this runner is exercised at.  ``gpuwm.core.ruc`` maps it to
#: VEGPARM's ``MODI-RUC`` section (21 categories, water 17, snow/ice 15).
DEFAULT_VEGETATION_DATASET = "MODIFIED_IGBP_MODIS_NOAH"

#: ``XICE_THRESHOLD`` under ``fractional_seaice = 0``, WRF's Registry
#: default 0.5 (module_surface_driver.F:1365-1366) and the value every RUC
#: run before ``RunConfig.fractional_seaice`` was pinned to.
XICE_THRESHOLD = 0.5
#: The threshold under ``fractional_seaice = 1`` (module_surface_driver.F
#: :1367-1368 of the HRRR v4.1.21 fork; operational HRRR, hrrr_wrf.nl:154).
#: :class:`RucRuntimeParameters` resolves the run's value from the config.
XICE_THRESHOLD_FRACTIONAL = 0.02

#: ``seaice_albedo_default``, Registry.EM_COMMON:2636 default 0.65.  Applied
#: to ALBBCK before the call, ``module_surface_driver.F:3453-3459``.
SEAICE_ALBEDO_DEFAULT = 0.65

#: ``integer, parameter :: isncovr_opt=2``, ``module_sf_ruclsm.F:78``.  A
#: compile-time constant in the pinned object, not a namelist knob.
ISNCOVR_OPT = RUC_SNOW_COVER_OPTION

#: The snow-compaction constants ``c1sn``/``c2sn``, ``module_sf_ruclsm.F``
#: driver defaults.  Also not namelist knobs.
C1SN = 0.026
C2SN = 21.0

#: gpuwm's DEFINED value for WRF's uninitialised ``ilnb``.
#:
#: ``sfctmp`` declares ``ilnb`` as a plain local (``:1385``) and never
#: initialises it; ``snowseaice`` takes it ``intent(inout)`` (``:3896``) and
#: ``snowtemp`` takes it ``intent(out)`` (``:4994``), so under the Fortran
#: standard it is undefined on entry in both.  Both assign it only inside
#: ``if(snhei.ge.snth)`` (``:4038``/``:4059``, ``:5124``/``:5148``) and both
#: then READ it at ``if(ilnb.gt.1)`` under the wider ``if(snhei.gt.0.)``
#: (``:4410``, ``:5716``), which selects the one- or two-layer ``tsnav``.
#:
#: The unassigned window is therefore exactly ``0 < snhei < snth``.  There
#: ``deltsn = 0.05e3/rhosn`` is five times ``snth = 0.01e3/rhosn``
#: (``:3387-3388``, ``:3945-3946``), so ``snhei - deltsn < 0`` and the
#: two-layer form weights a negative thickness.  A pack thinner than the
#: depth at which WRF models one layer cannot have two: ONE is the defined
#: answer, not merely the safe one.  The runner passes ``ilnb_chain=False``
#: so no column inherits another column's value either.
DEFINED_ILNB = 1

#: RUC's persistent per-column state that gpuwm's generic surface field set
#: does not already carry.  Seven of these are the Registry's own RUC package
#: line (``Registry.EM_COMMON:3147``,
#: :data:`gpuwm.core.ruc_contract.RUC_PACKAGE_STATE_FIELDS`); the rest are
#: EM_COMMON state that only RUC writes, so allocating them in a Noah run
#: would change that run's restart inventory and VRAM budget for nothing.
RUC_STATE_2D = (
    # Registry RUC package, 2-D half
    "soilt1", "rhosnf", "snowfallac", "precipfr", "acrunoff",
    # LSMRUC intent(inout) state that no other gpuwm scheme touches
    "tsnav", "sfcexc", "sfcevp", "qvg", "qcg", "qsg", "dew",
)

#: Registry RUC package, 3-D half.  ``(num_soil_layers, ny, nx)`` at
#: whichever admitted geometry the run resolved.
RUC_STATE_3D = ("smfr3d", "keepfr3dflag")

#: The driver locals gpuwm publishes as diagnostics.  WRF keeps all four as
#: automatic arrays of ``LSMRUC`` and returns none of them, so these are
#: gpuwm additions, not Registry fields -- named ``ruc_*`` so a wrfout reader
#: cannot mistake them for WRF output.
RUC_DIAGNOSTICS_2D = ("ruc_infiltr", "ruc_smelt", "ruc_runoff1",
                      "ruc_runoff2")

#: Ice and open-water components carried by WRF's fractional-sea-ice
#: surface-layer wrapper and consumed by the RUC post-call reblend.
RUC_FRACTIONAL_SEAICE_FIELDS = (
    "tsk_save", "tsk_sea", "znt_sea", "ust_sea", "mol_sea", "zol_sea",
    "flhc_sea", "flqc_sea", "cpm_sea",
    "cqs2_sea", "chs2_sea", "chs_sea", "qsfc_sea", "qgh_sea",
    "hfx_sea", "qfx_sea", "lh_sea",
)


def _ruc_fractional_deblend(
        blended, sea_value, xice, ice_component, *, arrays):
    """WRF v4.6.1 ``module_surface_driver.F:3468-3470``."""
    xp = arrays
    one = np.float32(1.0)
    denominator = xp.where(
        ice_component, xice, np.float32(1.0))
    return xp.where(
        ice_component,
        (blended - (one - xice) * np.float32(sea_value)) / denominator,
        blended,
    ).astype(xp.float32)


def _ruc_fractional_reblend(
        ice_value, sea_value, xice, ice_component, *, arrays):
    """WRF v4.6.1 ``module_surface_driver.F:3535-3572``."""
    xp = arrays
    return xp.where(
        ice_component,
        ice_value * xice + (np.float32(1.0) - xice) * sea_value,
        ice_value,
    ).astype(xp.float32)


def _ruc_seaice_albedo_override(
        albbck, xice, seaice_albedo_default, *, arrays,
        xice_threshold: float = XICE_THRESHOLD):
    """WRF v4.6.1 ``module_surface_driver.F:3453-3459``.

    Kept as a pure array operation so the two legal configuration values can
    be distinguished without executing the much larger RUC column.
    ``xice_threshold`` is the run's (0.5 or 0.02 by ``fractional_seaice``).
    """
    xp = arrays
    ice = (xice >= np.float32(xice_threshold)) & (
        xice <= np.float32(1.0))
    return xp.where(
        ice, np.float32(seaice_albedo_default), albbck
    ).astype(xp.float32), ice


#: gpuwm's 2-D field name -> the ``LSMRUC`` argument it binds.  Every name on
#: the right is in :data:`gpuwm.core.ruc.RUC_DRIVER_COLUMN_STATE`.  The four
#: renames are WRF's own: the surface driver passes ``albedo`` as ``alb``,
#: ``tsk`` as ``soilt``, ``smstav``/``smstot`` as ``smavail``/``smmax`` and
#: ``acsnom`` as ``snom`` (``module_surface_driver.F:3517-3522``).
RUC_STATE_BINDING: dict[str, str] = {
    "snow": "snow", "snowh": "snowh", "snowc": "snowc", "canwat": "canwat",
    "snoalb": "snoalb", "albedo": "alb", "emiss": "emiss", "lai": "lai",
    "mavail": "mavail", "sfcexc": "sfcexc", "z0": "z0", "znt": "znt",
    "tsk": "soilt", "hfx": "hfx", "qfx": "qfx", "lh": "lh",
    "sfcevp": "sfcevp", "sfcrunoff": "sfcrunoff", "udrunoff": "udrunoff",
    "acrunoff": "acrunoff", "grdflx": "grdflx", "acsnow": "acsnow",
    "acsnom": "snom", "qvg": "qvg", "qcg": "qcg", "dew": "dew",
    "qsfc": "qsfc", "qsg": "qsg", "chklowq": "chklowq", "soilt1": "soilt1",
    "tsnav": "tsnav", "smstav": "smavail", "smstot": "smmax",
    "rhosnf": "rhosnf", "precipfr": "precipfr", "snowfallac": "snowfallac",
}

#: gpuwm's 3-D field name -> the ``LSMRUC`` argument it binds.  ``tslb`` is
#: WRF's SOIL TEMPERATURE and RUC calls it ``tso``; ``smois`` is ``soilmois``.
RUC_PROFILE_BINDING: dict[str, str] = {
    "smois": "soilmois", "sh2o": "sh2o", "tslb": "tso",
    "smfr3d": "smfr3d", "keepfr3dflag": "keepfr3dflag",
}

#: RUC restart carriers measured INERT over one step, as perturbation targets,
#: on both an unfrozen and a snow-covered grid.  Not a list of fields the
#: restart omits -- every one of them round-trips bit for bit; a list of fields
#: whose restored value provably does not reach the next step, because LSMRUC
#: recomputes each of them before reading it.  Measured, not reasoned: the
#: sweep behind it is ``test_the_inert_restart_carriers_are_the_measured_ones``
#: and it is published because the alternative is a user believing a
#: checkpointed SFCEXC or GRDFLX means something.
#:
#: Three groups.  Pure diagnostics LSMRUC assigns unconditionally every call
#: (``grdflx = -sflx`` at ``:1096``, ``lh``/``hfx``/``sfcexc``, ``smstav``,
#: ``smstot``, ``chklowq``, ``snowc`` at ``:1112``, and T2/TH2/Q2, which
#: SFCDIAGS_RUCLSM rebuilds after the LSM).  Table lookups SOILVEGIN refreshes
#: every call from the vegetation category (``emiss``, ``lai``, ``z0``).  And
#: state the freezing curve re-derives: ``sh2o`` and ``qsg``, which
#: ``soilprop`` and ``:513`` rebuild from TSO/SOILT.
RUC_MEASURED_INERT_CARRIERS: tuple[str, ...] = (
    "chklowq", "dew", "emiss", "grdflx", "lai", "lh", "precipfr", "q2",
    "qcg", "qsg", "sfcexc", "sh2o", "smstav", "smstot", "snowc", "t2",
    "th2", "z0",
)

#: RUC carriers measured LIVE over one step on BOTH grids -- the restart
#: falsification picks from here rather than from a field that looks live.
#: ``tslb`` moved 8 other watched fields on the unfrozen grid and 12 on the
#: snow-covered one; ``smois`` 17 and 24; ``qvg`` 16 and 27; ``tsk`` 16 and 21.
RUC_MEASURED_LIVE_CARRIERS: tuple[str, ...] = (
    "acrunoff", "acsnom", "acsnow", "canwat", "mavail", "qsfc", "qvg",
    "rhosnf", "sfcevp", "sfcrunoff", "smois", "snow", "snowfallac", "snowh",
    "tsk", "tslb", "udrunoff", "znt",
)

#: Restrictions the option-identity schema has no field for.  Each entry is
#: (name, what gpuwm does, why it differs).  Published as data, as registry
#: warnings and in ``docs/wrf_ruc_runtime_admission.md``, because a
#: restriction that lives only in a docstring is a restriction a user meets
#: at hour three of a forecast.
RUC_RUNTIME_RESTRICTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "RUC_soil_ingest_is_wired_but_RUC_is_not_a_registry_profile",
        "The nine-level TSLB/SMOIS a RUC run starts from comes from "
        "gpuwm.ingest.ruc_soil: preprocess_land_surface_soil is the one "
        "soil seam every initializer calls, and it routes "
        "sf_surface_physics=3 to preprocess_ruc_soil (init_soil_depth_3 + "
        "init_soil_3_real, max_ulp 0 against "
        "gpuwm/data/ruc/oracle/soil_ingest.csv, driven over all 861,001 "
        "columns of the four-domain case) while schemes 2/4 stay on "
        "preprocess_noah_soil and anything else refuses.  "
        "gpuwm.core.ruc_runtime.ruc_cold_start then derives SH2O, SMFR3D, "
        "MAVAIL and ZNT from the nine levels through ruclsminit.",
        "Both halves of the former restriction here -- 'the remap has no "
        "wiring' and, before that, 'there is no remap' -- are closed.  What "
        "remains is that RUC_PROFILE_ID is still not a member of "
        "SINGLE_DOMAIN_PHYSICS_PROFILES: every consumer of that tuple "
        "carries per-profile source-absent-state tables with no RUC row "
        "(selectable without them would KeyError on the first source-absent "
        "field), and the column loop runs on the host (see "
        "the_column_loop_runs_on_the_host).  flag_sm_adj stays refused for "
        "a different reason -- it is a real.exe knob, not a runtime one "
        "(see below).",
    ),
    (
        "p8w_is_the_layer_mid_pressure",
        "The p8w argument receives atmosphere['pressure'][0], the lowest "
        "layer's MID pressure.",
        "Not a divergence -- a trap.  module_surface_driver.F:3506 passes "
        "p_phy into an argument LSMRUC names p8w and documents as '3d "
        "pressure (pa)'.  Every saturation humidity in RUC divides by it "
        "(qsg = qsn(soilt,tbq)/(p8w*1e-2)), so binding the interface "
        "pressure instead would be a silent bias, and it would look right.",
    ),
    (
        "myj_arm_unreachable",
        "myj=False only.",
        "ruc_soil_step and ruc_snow_soil_step are fail-closed on myj=True, "
        "so LSMRUC's :681-682 MYJ arm is unreachable and unverified.  gpuwm "
        "has no MYJ PBL, so nothing can select it.",
    ),
    (
        "lai_source_follows_rdlai2d",
        "rdlai2d=False (the default): LAI is SOILVEGIN's table value for "
        "the column's vegetation category, refreshed every call.  "
        "rdlai2d=True: the monthly LAI12M field interpolated to the start "
        "date (real.exe, module_initialize_real.F:1197) stays as the LSM's "
        "LAI and SOILVEGIN leaves it alone (module_sf_ruclsm.F:7028, :7075).",
        "The static catalogue carries LAI12M and the driver seeds the lai "
        "field from it at start; the switch decides whether SOILVEGIN "
        "overwrites that seed.  The receipt records the value that ran.",
    ),
    (
        "snow_cover_option_is_compile_time",
        f"isncovr_opt={ISNCOVR_OPT}.",
        "integer, parameter at module_sf_ruclsm.F:78, not a namelist knob.  "
        "Options 1 and 3 are transcribed in gpuwm.core.ruc for completeness "
        "and are NOT oracle-verified, so the runner pins 2.",
    ),
    (
        "ilnb_is_defined_not_reproduced",
        f"ilnb={DEFINED_ILNB} for every column, with ilnb_chain=False.",
        "WRF reads ilnb uninitialised on 0 < snhei < snth and gets the "
        "previous column's value, which makes tsnav depend on grid "
        "traversal order.  gpuwm implements the defined behaviour instead "
        "and does not reproduce the bug; see DEFINED_ILNB for why 1 is the "
        "defined answer.  oracle/lsmruc.csv still pins WRF's chained "
        "answer, because that is what WRF did.",
    ),
    (
        "five_driver_stale_locals_are_zero",
        "snoh, snflx, s, sublim and evapl enter SFCTMP as zero on every "
        "call.",
        "They are automatic arrays of LSMRUC that SFCTMP reads.  WRF zeroes "
        "them in the ktau==1 block (:531-542) and on every later step reads "
        "whatever is left in that stack region.  oracle/lsmruc_stackfill.csv "
        "is the same driver with a nonzero callee stack and differs from "
        "oracle/lsmruc.csv only in tsnav on the two thin-snow columns, so "
        "on that fixture the five are unobservable and ilnb is not.",
    ),
    (
        "sfcevp_is_double_counted_on_purpose",
        "SFCEVP advances by 2*QFX*dt per call.",
        "module_sf_ruclsm.F accumulates it twice, at :1095 and again at "
        ":1116, with nothing in between changing qfx.  That is a duplicated "
        "statement rather than undefined behaviour, so gpuwm reproduces it "
        "and oracle/lsmruc.csv pins it.  A user integrating SFCEVP as a "
        "water budget must halve it.",
    ),
    (
        "sfcdiags_ruclsm_flux_branch_only",
        "The flux=.true. arms of SFCDIAGS_RUCLSM are transcribed.",
        "flux is a hardcoded local (module_sf_sfcdiags_ruclsm.F:47-48), so "
        "the else arms are dead in the pinned object.  Their T2 form is "
        "CHS/CHS2-ratio based rather than HFX based and would be a "
        "different diagnostic, not a smaller one.",
    ),
    (
        "the_column_loop_runs_on_the_host",
        "Every column goes through gpuwm.core.ruc.ruc_land_surface_step in "
        "host FP32, with one device->host copy of the surface slab before "
        "and one host->device copy after.",
        "gpuwm/core/ruc_gpu.py has no device sfctmp and no device lsmruc -- "
        "the CUDA leaves stop at ruc_snow_soil_step_cuda -- so there is "
        "nothing to launch.  This is the scheme's scaling blocker and the "
        "measured per-column cost is in the registry warnings.",
    ),
    (
        "sr_is_a_BINARY_phase_proxy_off_the_wsm_family",
        "RUC's frozen-precipitation fraction FRZFRAC comes from gpuwm's SR "
        "field.  Under mp_physics in (1, 6, 8, 10, 18) that is the "
        "microphysics scheme's own SR.  Under any other microphysics -- "
        "including mp_physics=0, a dry or warm-rain run -- gpuwm substitutes "
        "the BINARY proxy (T(k=1) <= 273.15), so SR is exactly 0 or exactly "
        "1 and never a fraction.",
        "This is the restriction the sweep for undisclosed ones found, and it "
        "is the RUC analogue of the MYNN FLAG_QS that shipped undeclared.  "
        "WRF's surface driver sets frpcpn=.true. whenever SR is PRESENT "
        "(module_surface_driver.F:3448-3452) and ARW always passes it, so "
        "frpcpn=True is not the divergence; the CONTENT of SR is.  RUC "
        "consumes it as a continuous fraction at "
        "module_sf_ruclsm.F:654-660 (snowrat=rainbl*frzfrac), so a binary "
        "proxy makes every precipitation event all-rain or all-snow with no "
        "mixed phase, which is visible in new-snow density and in the "
        "melt/refreeze budget near 0 C.  The alternative arm of WRF's own "
        "gate is worse and is worth naming: with SR absent WRF sets SR = 1., "
        "i.e. ALL precipitation frozen.",
    ),
    (
        "eighteen_restart_carriers_are_MEASURED_INERT",
        "The 18 names in RUC_MEASURED_INERT_CARRIERS round-trip bit for bit "
        "but their restored values provably do not reach the next step: "
        "LSMRUC recomputes each before reading it.  The 18 live ones are in "
        "RUC_MEASURED_LIVE_CARRIERS.",
        "Not a divergence -- a property of the scheme, measured rather than "
        "assumed, because the Noah-MP lane shipped a restart gate that was "
        "tuned to its own grid and a reader needs to know which checkpointed "
        "fields carry information.  A user treating a checkpointed SFCEXC, "
        "GRDFLX, LH, SMSTAV or T2 as model state is reading a receipt of the "
        "last call, not a state the next call depends on.",
    ),
    (
        "smfr3d_and_keepfr3dflag_are_CONDITIONAL_carriers",
        "SMFR3D reaches the next step only at a cell where "
        "KEEPFR3DFLAG == 1, and only when nudged DOWNWARD.  On an unfrozen "
        "grid no cell has KEEPFR3DFLAG == 1 at all and SMFR3D is never read.",
        "WRF's own structure, measured: soilprop rebuilds soilice from TSO "
        "and SOILMOIS through the freezing curve at "
        "module_sf_ruclsm.F:2695-2703 and consults the restored SMFR3D only "
        "as the cap ``soilice(k)=min(soilice(k),smfrkeep(k))`` at :2704-2707, "
        "inside ``if(keepfr(k).eq.1.)``; and keepfr is assigned 1 at :2744 "
        "only inside ``if (soilice(k).gt.0.)``.  So on a warm column "
        "``tln=log(tso/273.15) >= 0``, soilice is 0, keepfr never reaches 1 "
        "and smfrkeep is unread.  Because the read is a min(), raising "
        "SMFR3D at a keepfr==1 cell also does almost nothing -- an "
        "unbinding cap.  Measured on a 265 K column: SMFR3D scaled to 0.1x "
        "at a keepfr==1 cell moved 21 other carriers; the same nudge at a "
        "keepfr==0 cell moved none.",
    ),
    (
        "there_is_no_urban_coupling_and_WRF_HAS_NONE_HERE_EITHER",
        "sf_urban_physics is not read by gpuwm's RUC seam, and no urban "
        "arrays are carried.",
        "Recorded as a NEGATIVE finding rather than left silent, because the "
        "equivalent assumption shipped undeclared in the Noah-MP lane and "
        "sf_urban_physics is not even a RunConfig field, so no schema check "
        "could have caught it.  Swept and cleared: "
        "module_surface_driver.F's CASE (RUCLSMSCHEME) arm (:3438-3596) "
        "contains no reference to sf_urban_physics, no urban PRESENT() "
        "guard and no urban call, unlike the Noah arm (:2702) and the "
        "Noah-MP arm (:3185, noahmp_urban).  RUC in WRF v4.6.1 has no urban "
        "coupling to omit, so this is a restriction gpuwm does NOT have.",
    ),
    (
        "rhosnf_is_a_SENTINEL_of_-1e3_until_snow_falls",
        "RHOSNF reads -1000 kg m-3 for the whole forecast unless snow "
        "actually falls.",
        "LSMRUC seeds it at :552 in the ktau==1 block and only ever "
        "overwrites it from rhosnfall, the density of NEW snowfall, so a run "
        "with no frozen precipitation reports the seed forever -- measured "
        "over 600 steps with a 15 mm pack present and melting.  Published "
        "because a user meeting a negative density in a wrfout would "
        "reasonably read it as corruption, and because it means RHOSNF "
        "cannot be used to infer anything about an EXISTING pack: it "
        "describes the last snowfall, not the snow on the ground.",
    ),
    (
        "a_snow_pack_over_warm_soil_melts_in_seconds_and_that_is_correct",
        "A pack supplied over a soil column warmer than 273.15 K melts "
        "within a few steps, with ACSNOM accounting for all of it.",
        "Not a divergence, and not a defect -- an inconsistent INITIAL state "
        "resolving.  Measured: 15 mm SWE over a 303 K skin and a 295..303 K "
        "soil column is gone in eight 6 s steps with ACSNOM reaching "
        "15.0020 mm, a budget closure of 0.03%, and TSK dropping 303 -> "
        "292.5 K on the first step.  A warm 5 cm soil layer holds about "
        "3.4 MJ m-2 of excess heat against the 5.0 MJ m-2 the pack needs, so "
        "there is nothing unphysical about the rate.  A 60 mm pack on the "
        "same soil melts 9.5 mm in the first step and then stops, once the "
        "interface has cooled and the remaining pack insulates it.  The "
        "consequence for a caller is that RUC's snow branch is only "
        "meaningfully exercised when TSLB is consistent with the pack: "
        "supply a subfreezing soil column, or the pack is an initialisation "
        "transient.",
    ),
    (
        "sfcdiags_ruclsm_is_handed_snow_and_ignores_it",
        "The SNOW argument of SFCDIAGS_RUCLSM is not passed by gpuwm's seam.",
        "Not a divergence.  module_sf_sfcdiags_ruclsm.F declares SNOW "
        "intent(IN) at :21 and the surface driver passes it at :3591, but "
        "the body never references it, in either the flux=.true. or the "
        "flux=.false. arm.  Named here so a reader diffing gpuwm's call "
        "against WRF's does not go looking for the snow dependence.",
    ),
    (
        "no_wrf_trajectory_comparison",
        "The routines are bitwise against their WRF oracles and the "
        "assembled driver is bitwise against oracle/lsmruc.csv except its "
        "26 pinned upstream-residue cells.  No gpuwm/WRF forecast "
        "trajectory comparison exists.",
        "That is what wrf-matched-run-candidate would require and this scheme "
        "implemented-unverified.",
    ),
)


class RucRuntimeParameters:
    """The RUC parameter bundle, its land-use identity and its soil geometry.

    Deliberately not a dataclass, for the reason
    :class:`gpuwm.core.noahmp_runtime.NoahmpRuntimeParameters` gives: the
    restart layer walks a dataclass field by field, and what identifies a RUC
    run is the table bytes plus the option identity, which
    :meth:`restart_identity` returns as strict JSON.
    """

    def __init__(
            self, bundle=None, *,
            dataset_identifier: str = DEFAULT_VEGETATION_DATASET,
            seaice_albedo_default: float = SEAICE_ALBEDO_DEFAULT,
            num_soil_layers: int = NUM_SOIL_LAYERS,
            rdlai2d: bool = False,
            fractional_seaice: int = 0):
        if int(num_soil_layers) not in WRF_SUPPORTED_NUM_SOIL_LAYERS:
            raise ValueError(
                f"RUC num_soil_layers {num_soil_layers!r} is not one of "
                f"{WRF_SUPPORTED_NUM_SOIL_LAYERS}")
        self.num_soil_layers = int(num_soil_layers)
        if type(rdlai2d) is not bool:
            raise TypeError("rdlai2d must be bool")
        self.rdlai2d = rdlai2d
        if (isinstance(fractional_seaice, bool)
                or fractional_seaice not in (0, 1)):
            raise ValueError(
                f"fractional_seaice must be 0 or 1, got {fractional_seaice!r}")
        self.fractional_seaice = int(fractional_seaice)
        # module_surface_driver.F:1365-1368 (HRRR v4.1.21 fork): the one
        # threshold every sea-ice test in the seam, the fused driver and the
        # CLM lake read.
        self.xice_threshold = (XICE_THRESHOLD_FRACTIONAL
                               if self.fractional_seaice else XICE_THRESHOLD)
        self.bundle = load_ruc_parameters() if bundle is None else bundle
        self.dataset_identifier = str(dataset_identifier)
        value = float(seaice_albedo_default)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "seaice_albedo_default must be finite and in [0, 1], got "
                f"{seaice_albedo_default!r}")
        self.seaice_albedo_default = value
        # Raises on a dataset gpuwm.core.ruc has no RUC VEGPARM section for,
        # rather than silently falling back to another table.
        self.vegetation = self.bundle.vegetation_for(self.dataset_identifier)
        self.zs, self.dzs = ruc_soil_geometry(self.num_soil_layers)
        # module_sf_ruclsm.F has no ISWATER/ISICE table scalar for the RUC
        # sections, so gpuwm.core.ruc resolves them from the section name.
        # Left as None here so exactly one implementation of that rule
        # exists, in the driver.
        self.iswater = None
        self.isice = None

    def restart_identity(self) -> dict:
        """Strict-JSON identity: the table bytes plus the pinned choices."""
        receipt = self.bundle.receipt
        assets = receipt.get("assets", {})
        payload = {}
        if isinstance(assets, Mapping):
            for name, entry in sorted(assets.items()):
                if isinstance(entry, Mapping):
                    payload[str(name)] = {
                        "bytes": int(entry.get("canonical_bytes", 0)),
                        "sha256": str(entry.get("canonical_sha256", "")),
                    }
        identity = {
            # The same string the checkpoint header binds; one spelling.
            "algorithm": LAND_SURFACE_ALGORITHM_IDENTITIES[3],
            "wrf_source": "phys/module_sf_ruclsm.F:LSMRUC + "
                          "phys/module_sf_sfcdiags_ruclsm.F:SFCDIAGS_RUCLSM",
            "dataset_identifier": self.dataset_identifier,
            "vegetation_section": self.vegetation.name,
            # The RESOLVED count, not the module constant.  A
            # nine-level constant in a six-level run's receipt is a false
            # receipt, and a receipt is the thing a later reader trusts.
            "num_soil_layers": int(self.num_soil_layers),
            "soil_level_depths_m": [float(v) for v in self.zs],
            # What the geometry's numbers have actually been judged against.
            # Nine levels is compared field for field to WRF v4.6.1 fixtures;
            # six has a compiled column, host/device agreement and WRF-matched
            # LEVEL DEPTHS, but no forecast oracle at all.  Never let a
            # receipt imply otherwise.
            "soil_geometry_evidence": (
                "wrf-oracle" if self.num_soil_layers == NUM_SOIL_LAYERS
                else "internal-consistency-only"),
            "xice_threshold": float(self.xice_threshold),
            "seaice_albedo_default": float(self.seaice_albedo_default),
            "isncovr_opt": int(ISNCOVR_OPT),
            "c1sn": float(C1SN),
            "c2sn": float(C2SN),
            "defined_ilnb": int(DEFINED_ILNB),
            "ilnb_chain": False,
            "column_solver": "host-fp32",
            "tables": payload,
        }
        # Absent at their defaults, as in a header written before the
        # switches existed, so earlier RUC checkpoints keep their
        # manifest; set, each binds the trajectory it changes.
        if self.fractional_seaice:
            identity["fractional_seaice"] = int(self.fractional_seaice)
        if self.rdlai2d:
            identity["rdlai2d"] = True
        return identity


def ruc_cold_start(fields, *, params: RucRuntimeParameters) -> None:
    """``ruclsminit`` over the whole slab, on the host.

    Runs once, at driver construction, where ``module_physics_init.F`` runs
    it.  ``ruclsminit`` derives SH2O and SMFR3D from TSLB/SMOIS by the
    freezing curve and sets MAVAIL and ZNT; without it SH2O is whatever the
    ingest supplied for SMOIS and SMFR3D is zero everywhere, which claims a
    frozen-soil state of exactly none on a frozen column.

    Every other RUC carrier keeps the value
    :func:`gpuwm.core.physics.initialize_physics` set, which is WRF's
    Registry cold state for it -- with two exceptions LSMRUC itself repairs
    in its own ``ktau==1`` block (``:481-565``): SOILT1 outside 170..400 K is
    rebuilt from SOILT/TSO, and RHOSNF is seeded to -1e3.  Those stay in the
    driver rather than being duplicated here, so there is one implementation
    of each.
    """
    import cupy as cp

    slab = {name: np.ascontiguousarray(cp.asnumpy(fields[name]))
            for name in ("tslb", "smois", "sh2o", "isltyp", "ivgtyp",
                         "xice", "mavail", "znt", "smfr3d")}
    resolved = int(slab["tslb"].shape[0])
    if resolved not in WRF_SUPPORTED_NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC cold start needs a soil column at one of WRF's tabulated "
            f"geometries {WRF_SUPPORTED_NUM_SOIL_LAYERS}, got {resolved}")
    if resolved != params.num_soil_layers:
        raise ValueError(
            f"RUC cold start got a {resolved}-level soil column but the "
            f"runtime parameters resolved {params.num_soil_layers} levels; "
            "the slab and the geometry must be the same run")
    cold = ruc_initialize_cold_start(
        slab["tslb"], slab["smois"], slab["isltyp"], slab["ivgtyp"],
        slab["xice"], mminlu=params.dataset_identifier,
        parameters=params.bundle)
    fields["sh2o"][...] = cp.asarray(np.ascontiguousarray(cold.sh2o))
    fields["smfr3d"][...] = cp.asarray(np.ascontiguousarray(cold.smfr3d))
    fields["mavail"][...] = cp.asarray(np.ascontiguousarray(cold.mavail))
    fields["znt"][...] = cp.asarray(np.ascontiguousarray(cold.znt))


def ruc_device_sfctmp_sets():
    """The leaf set, the stage set and the array namespace a forecast runs.

    ``gpuwm.core.ruc_gpu`` imports cupy at module scope and
    ``tests/conftest.py`` auto-marks any module that does as ``gpu``, so the
    import happens here
    rather than at the top of this module: the RUC oracle suite must keep
    importing :mod:`gpuwm.core.ruc_runtime` on a machine with no card.

    The three go together and that is why they are returned together.  The
    RESIDENT leaf and stage sets return device arrays rather than wrapping
    every returned field in ``cp.asnumpy``, and they are only usable by a
    driver whose masking and recombination are on the same side of the
    boundary -- which is what the namespace does.  Handing a host driver the
    resident leaves, or a device driver the host-facing ones, is a type error
    at the first gather rather than a silent slow path.

    There is deliberately no argument and no environment switch that selects
    the host set instead.  The host leaves are the reference implementation
    the oracle suite compares against, and both sets have been shown
    ``max_ulp 0`` against each other through this driver, warm and
    snow-covered, at 512 / 4,096 / 24,576 columns across all 43 returned
    fields -- so a forecast has nothing to gain from the host set and roughly
    three orders of magnitude of wall clock to lose.
    """

    from gpuwm.core.ruc_gpu import (RUC_DEVICE_ARRAYS,
                                    RUC_SFCTMP_DEVICE_LEAVES_RESIDENT,
                                    RUC_SFCTMP_DEVICE_STAGES_RESIDENT)

    return (RUC_SFCTMP_DEVICE_LEAVES_RESIDENT,
            RUC_SFCTMP_DEVICE_STAGES_RESIDENT,
            RUC_DEVICE_ARRAYS)


def ruc_lsm_step(
    fields,
    atmosphere: Mapping[str, object],
    *,
    params: RucRuntimeParameters,
    precipitation: SurfacePrecipitationForcing,
    dt: float,
    itimestep: int,
    mosaic_lu: int,
    mosaic_soil: int,
    flag_sm_adj: int,
    spp_lsm: int,
    lakemodel: int = 0,
    pattern_spp_lsm=None,
    field_sf=None,
    ruc_irrigation: str = "wrf_461",
    ruc_soilprop: str = "wrf_45",
    ruc_qvg_cold_start: str = "wrf",
    ruc_2m_diagnostic: str = "flux",
    ruc_snow: str = "wrf_461",
    alb_sol: int = 0,
) -> dict[str, int]:
    """One ``CASE (RUCLSMSCHEME)`` arm.  Mutates ``fields`` in place.

    Returns a small census -- land, water, lake and sea-ice column counts --
    so a test can tell "RUC ran" apart from "RUC ran on any land".  The
    counts are reconstructed from the same masks the driver dispatches on
    (``module_sf_ruclsm.F:823-826``, ``:828`` and ``:855``).
    """
    import cupy as cp

    if int(alb_sol) == 1:
        fields = dict(fields, albedo=fields["albsol"],
                      albbck=fields["albbcksol"])

    if int(itimestep) < 1:
        raise ValueError("LSMRUC ktau is one-based and starts at 1")
    if params.rdlai2d and not getattr(params, "_seeded_lai_verified", False):
        # initialize_physics starts LAI not-a-number under rdlai2d; every
        # road that has the monthly LAI12M field seeds it before the first
        # step.  One that did not would integrate RUC on no leaf area at
        # all, so it is refused here, once per run, by name.
        if not bool(cp.isfinite(fields["lai"]).all()):
            raise ValueError(
                "rdlai2d=true keeps the monthly LAI12M leaf area interpolated "
                "to the start date as RUC's LAI (module_sf_ruclsm.F:7075), "
                "but this road seeded no LAI field before the first land-"
                "surface step, so RUC would read no leaf area at all. Run "
                "from a source whose static catalogue carries LAI12M, or set "
                "rdlai2d=false for SOILVEGIN's table LAI")
        params._seeded_lai_verified = True
    # Second line behind validate_run_config, at the seam that consumes each
    # value, so the registry's citation of this file is true for all five.
    from gpuwm.core.ruc_mosaic import irrigation_form, mosaic_option
    mosaic_option(mosaic_lu, "mosaic_lu")
    mosaic_option(mosaic_soil, "mosaic_soil")
    mosaic_option(lakemodel, "lakemodel")
    irrigation_form(ruc_irrigation)
    from gpuwm.core.ruc_tier import (ruc_2m_diagnostic_form,
                                     ruc_qvg_cold_start_form, ruc_snow_form,
                                     ruc_soilprop_form)
    ruc_soilprop_form(ruc_soilprop)
    ruc_qvg_cold_start_form(ruc_qvg_cold_start)
    ruc_2m_diagnostic_form(ruc_2m_diagnostic)
    ruc_snow_form(ruc_snow)
    from gpuwm.core.ruc_spp import validate_spp_mode
    enabled_spp = validate_spp_mode(spp_lsm)
    if int(flag_sm_adj) != 0:
        # Not a runtime knob at all: share/module_soil_pre.F:2063 reads it
        # inside init_soil_3_real, i.e. in real.exe.  It is refused here
        # rather than accepted-and-ignored so a plan that asks for it is told,
        # and it stays refused now that a RUC ingest exists -- LSMRUC is a
        # timestep, and by the time it runs the adjustment has either already
        # happened at setup or never will.  It belongs to
        # gpuwm.ingest.ruc_soil.remap_soil_to_ruc_levels, whose
        # moisture_adjustment argument implements it at max_ulp 0.
        raise ValueError(
            f"flag_sm_adj={flag_sm_adj}: this is a real.exe knob "
            "(share/module_soil_pre.F:2063, RUC soil-moisture adjustment "
            "from a Noah initial state) and not an LSMRUC argument.  Ask for "
            "it at setup, through "
            "gpuwm.ingest.ruc_soil.remap_soil_to_ruc_levels"
            "(moisture_adjustment=True)")

    if enabled_spp:
        # The retained orchestration stays on the GPU. The disabled route
        # retains its original fused kernels and allocation inventory.
        return _ruc_lsm_step_reference(
            fields, atmosphere, params=params, precipitation=precipitation,
            dt=dt, itimestep=itimestep, mosaic_lu=mosaic_lu,
            mosaic_soil=mosaic_soil, flag_sm_adj=flag_sm_adj, spp_lsm=1,
            lakemodel=lakemodel,
            pattern_spp_lsm=pattern_spp_lsm, field_sf=field_sf,
            ruc_irrigation=ruc_irrigation, ruc_soilprop=ruc_soilprop,
            ruc_qvg_cold_start=ruc_qvg_cold_start,
            ruc_2m_diagnostic=ruc_2m_diagnostic, ruc_snow=ruc_snow)

    from gpuwm.core.ruc_fused import step

    return step(fields, atmosphere, params=params, precipitation=precipitation,
                dt=dt, itimestep=itimestep, mosaic_lu=mosaic_lu,
                mosaic_soil=mosaic_soil, lakemodel=lakemodel,
                irrigation=ruc_irrigation, soilprop=ruc_soilprop,
                qvg_cold_start=ruc_qvg_cold_start,
                diagnostic_2m=ruc_2m_diagnostic, snow=ruc_snow)


def _ruc_lsm_step_reference(
    fields,
    atmosphere: Mapping[str, object],
    *,
    params: RucRuntimeParameters,
    precipitation: SurfacePrecipitationForcing,
    dt: float,
    itimestep: int,
    mosaic_lu: int,
    mosaic_soil: int,
    flag_sm_adj: int,
    spp_lsm: int,
    lakemodel: int = 0,
    pattern_spp_lsm=None,
    field_sf=None,
    ruc_irrigation: str = "wrf_461",
    ruc_soilprop: str = "wrf_45",
    ruc_qvg_cold_start: str = "wrf",
    ruc_2m_diagnostic: str = "flux",
    ruc_snow: str = "wrf_461",
) -> dict[str, int]:
    """One ``CASE (RUCLSMSCHEME)`` arm.  Mutates ``fields`` in place.

    Returns a small census -- land, water, lake and sea-ice column counts --
    so a test can tell "RUC ran" apart from "RUC ran on any land".  The
    counts are reconstructed from the same masks the driver dispatches on
    (``module_sf_ruclsm.F:823-826``, ``:828`` and ``:855``).
    """
    import cupy as cp

    if int(itimestep) < 1:
        raise ValueError("LSMRUC ktau is one-based and starts at 1")
    # Second line behind validate_run_config, at the seam that consumes each
    # value, so the registry's citation of this file is true for all five.
    from gpuwm.core.ruc_mosaic import irrigation_form, mosaic_option
    mosaic_option(mosaic_lu, "mosaic_lu")
    mosaic_option(mosaic_soil, "mosaic_soil")
    mosaic_option(lakemodel, "lakemodel")
    irrigation_form(ruc_irrigation)
    from gpuwm.core.ruc_tier import (ruc_2m_diagnostic_form,
                                     ruc_qvg_cold_start_form, ruc_snow_form,
                                     ruc_soilprop_form)
    ruc_soilprop_form(ruc_soilprop)
    ruc_qvg_cold_start_form(ruc_qvg_cold_start)
    ruc_2m_diagnostic_form(ruc_2m_diagnostic)
    ruc_snow_form(ruc_snow)
    from gpuwm.core.ruc_spp import validate_spp_mode
    enabled_spp = validate_spp_mode(spp_lsm)
    if int(flag_sm_adj) != 0:
        # Not a runtime knob at all: share/module_soil_pre.F:2063 reads it
        # inside init_soil_3_real, i.e. in real.exe.  It is refused here
        # rather than accepted-and-ignored so a plan that asks for it is told,
        # and it stays refused now that a RUC ingest exists -- LSMRUC is a
        # timestep, and by the time it runs the adjustment has either already
        # happened at setup or never will.  It belongs to
        # gpuwm.ingest.ruc_soil.remap_soil_to_ruc_levels, whose
        # moisture_adjustment argument implements it at max_ulp 0.
        raise ValueError(
            f"flag_sm_adj={flag_sm_adj}: this is a real.exe knob "
            "(share/module_soil_pre.F:2063, RUC soil-moisture adjustment "
            "from a Noah initial state) and not an LSMRUC argument.  Ask for "
            "it at setup, through "
            "gpuwm.ingest.ruc_soil.remap_soil_to_ruc_levels"
            "(moisture_adjustment=True)")

    names_2d = tuple(RUC_STATE_BINDING) + (
        "swdown", "glw", "chs", "chs2", "cqs2", "flqc", "flhc", "cpm",
        "qgh",
        "albbck", "xland", "xice", "tmn", "shdmin", "shdmax", "vegfra",
        "rainbl", "sr", "ivgtyp", "isltyp", "psfc", "t2", "th2", "q2",
        "lakemask", "gsw", *RUC_FRACTIONAL_SEAICE_FIELDS,
        *RUC_DIAGNOSTICS_2D)
    # The surface slab STAYS ON THE CARD.  It used to be copied down here
    # field by field, driven through a host driver and copied back, which at
    # d04's 360,000 columns is roughly 165 MB in each direction for arrays
    # that were already on the device and go straight back to it.
    device = {name: cp.ascontiguousarray(fields[name]) for name in names_2d}
    device_3d = {name: cp.ascontiguousarray(fields[name])
                 for name in RUC_PROFILE_BINDING}

    # ---- the seam's own arithmetic ---------------------------------------
    # :3453-3459.  Unconditional, before the call, and outside the
    # FRACTIONAL_SEAICE block.  ALBBCK is an input to SOILVEGIN, so this
    # reaches the column's snow-free albedo on the same step.
    device["albbck"], ice = _ruc_seaice_albedo_override(
        device["albbck"], device["xice"], params.seaice_albedo_default,
        arrays=cp, xice_threshold=params.xice_threshold)
    ice_component = ice & (device["xice"] <= np.float32(1.0))
    ice_fraction = device["xice"]
    # module_surface_driver.F:3461-3473.  The static optics are grid-cell
    # blends; LSMRUC must receive the full ice component.
    device["albedo"] = _ruc_fractional_deblend(
        device["albedo"], 0.08, ice_fraction, ice_component, arrays=cp)
    device["emiss"] = _ruc_fractional_deblend(
        device["emiss"], 0.98, ice_fraction, ice_component, arrays=cp)
    device["tsk"] = cp.where(
        ice_component, device["tsk_save"], device["tsk"]).astype(cp.float32)

    temperature = cp.ascontiguousarray(atmosphere["temperature"][0])
    qv = cp.ascontiguousarray(atmosphere["qv"][0])
    qc = cp.ascontiguousarray(atmosphere["qc"][0])
    rho = cp.ascontiguousarray(atmosphere["rho"][0])
    dz1 = cp.ascontiguousarray(atmosphere["dz"][0])
    # :3506 -- p_phy, the layer MID pressure, into an argument named p8w.
    p_mid = cp.ascontiguousarray(atmosphere["pressure"][0])
    values = {
        # forcing
        "z3d": dz1, "p8w": p_mid, "t3d": temperature, "qv3d": qv,
        "qc3d": qc, "rho3d": rho,
        "rainbl": device["rainbl"], "frzfrac": device["sr"],
        "glw": device["glw"], "gsw": device["gsw"], "chs": device["chs"],
        "flqc": device["flqc"], "flhc": device["flhc"],
        "albbck": device["albbck"], "xland": device["xland"],
        "xice": device["xice"], "tbot": device["tmn"],
        "shdmin": device["shdmin"], "shdmax": device["shdmax"],
        "vegfra": device["vegfra"],
        # WRF-ARW/EM_CORE==1 precipitation and lake arguments.
        "rainncv": precipitation.rain_nonconvective,
        "snowncv": precipitation.snow_nonconvective,
        "graupelncv": precipitation.graupel_nonconvective,
        "lakemask": device["lakemask"],
    }
    for name, argument in RUC_STATE_BINDING.items():
        values[argument] = device[name]
    for name, argument in RUC_PROFILE_BINDING.items():
        values[argument] = device_3d[name]
    missing = [name for name in (RUC_DRIVER_PROFILE_STATE
                                 + RUC_DRIVER_COLUMN_STATE
                                 + RUC_DRIVER_COLUMN_FORCING
                                 + RUC_DRIVER_ARW_FORCING)
               if name not in values]
    if missing:
        # A binding table that drifts from the driver's contract is exactly
        # the failure this module exists to prevent, so it is checked rather
        # than trusted.
        raise AssertionError(
            f"RUC seam binding omits LSMRUC arguments: {sorted(missing)}")

    leaves, stages, device_arrays = ruc_device_sfctmp_sets()
    result = ruc_land_surface_step(
        values, dt=float(dt), ktau=int(itimestep), zs=params.zs,
        ivgtyp=device["ivgtyp"], isltyp=device["isltyp"],
        myj=False, em_core=1, lakemodel=lakemodel, frpcpn=True,
        rdlai2d=bool(params.rdlai2d),
        mosaic_lu=int(mosaic_lu), mosaic_soil=int(mosaic_soil),
        landusef=fields.get("landusef"), soilctop=fields.get("soilctop"),
        iswater=params.iswater, isice=params.isice,
        xice_threshold=float(params.xice_threshold),
        ilnb=int(DEFINED_ILNB), ilnb_chain=False,
        c1sn=float(C1SN), c2sn=float(C2SN),
        isncovr_opt=int(ISNCOVR_OPT),
        mminlu=params.dataset_identifier, parameters=params.bundle,
        leaves=leaves, stages=stages, arrays=device_arrays,
        spp_lsm=spp_lsm, pattern_spp_lsm=pattern_spp_lsm, field_sf=field_sf,
        irrigation=ruc_irrigation, soilprop=ruc_soilprop,
        qvg_cold_start=ruc_qvg_cold_start, snow=ruc_snow)

    for name, argument in RUC_STATE_BINDING.items():
        device[name] = cp.ascontiguousarray(
            cp.asarray(getattr(result, argument), dtype=cp.float32))
    for name, argument in RUC_PROFILE_BINDING.items():
        device_3d[name] = cp.ascontiguousarray(
            cp.asarray(getattr(result, argument), dtype=cp.float32))
    for name, local in (("ruc_infiltr", result.infiltr),
                        ("ruc_smelt", result.smelt),
                        ("ruc_runoff1", result.runoff1),
                        ("ruc_runoff2", result.runoff2)):
        device[name] = cp.ascontiguousarray(
            cp.asarray(local, dtype=cp.float32))

    # module_surface_driver.F:3530-3577.  LSMRUC returns full ice values;
    # rebuild the grid-cell blend from the open-water component captured by
    # the fractional surface-layer wrapper.  TSK_SAVE remains ice-only.
    device["albedo"] = _ruc_fractional_reblend(
        device["albedo"], np.float32(0.08), ice_fraction, ice_component,
        arrays=cp)
    device["emiss"] = _ruc_fractional_reblend(
        device["emiss"], np.float32(0.98), ice_fraction, ice_component,
        arrays=cp)
    for name in ("flhc", "flqc", "cpm", "cqs2", "chs2", "chs",
                 "qsfc", "qgh", "hfx", "qfx", "lh"):
        device[name] = _ruc_fractional_reblend(
            device[name], device[f"{name}_sea"], ice_fraction,
            ice_component, arrays=cp)
    device["tsk_save"] = cp.where(
        ice_component, device["tsk"], device["tsk_save"]).astype(cp.float32)
    device["tsk"] = _ruc_fractional_reblend(
        device["tsk"], device["tsk_sea"], ice_fraction, ice_component,
        arrays=cp)

    # :3580-3585.  CQS and CHS are REBUILT from the post-call MAVAIL, and the
    # CHS overwrite persists -- it is the value the next scheme to read CHS
    # sees.  MAVAIL is bounded below by 1e-5 inside SOILMOIST, so the divide
    # cannot be by zero.
    cqs = (device["flqc"] / (device["mavail"] * rho)).astype(cp.float32)
    device["chs"] = (
        device["flhc"] / (device["cpm"] * rho)).astype(cp.float32)

    # :3587.  RUC's own 2-m diagnostic, not SFCDIAGS.
    #
    # This is the ONE thing on this seam that stays on the host, and it stays
    # deliberately.  _sfcdiags_ruclsm raises (1e5/psfc) to R/cp through
    # sfcdiags_exner_powers, glibc's powf on every host, and CUDA's powf is
    # a different function -- the exact divergence class that put 2 ULP
    # into hfx on this lane and cost a session to find.  Moving it to the
    # card is a TRANSCENDENTAL POLICY decision (see
    # _RUC_PROVISIONAL_TRANSCENDENTALS in gpuwm.core.ruc), not a performance
    # one, and the answer must not change to make a call faster.  So exactly
    # the fields it reads come down and the three it writes go back up, which
    # is a bounded and named cost instead of the whole slab.
    diagnostic_inputs = ("psfc", "chs2", "cqs2", "tsk", "hfx", "qfx", "qsfc")
    # Half the lowest layer, LSMRUC's conflx, for the log-profile form.
    half_layer = (dz1 * np.float32(0.5)).astype(np.float32)
    # These fields share the horizontal shape and dtype. Packing preserves
    # their bits and drains the stream once instead of once per field.
    inputs = ([device[name] for name in diagnostic_inputs]
              + [temperature, qv, rho, p_mid, cqs, half_layer])
    if any(value.dtype != inputs[0].dtype or value.shape != inputs[0].shape
           for value in inputs):
        diagnostic_slab = [np.ascontiguousarray(cp.asnumpy(value))
                           for value in inputs]
    else:
        diagnostic_slab = cp.asnumpy(cp.stack(inputs))
    host = {name: diagnostic_slab[index]
            for index, name in enumerate(diagnostic_inputs)}
    t_host, q_host, rho_host, p_host, cqs_host, half_host = (
        diagnostic_slab[len(diagnostic_inputs):])
    _sfcdiags_ruclsm(
        host,
        t3d=t_host, qv3d=q_host, rho3d=rho_host, p3d=p_host,
        cqs=cqs_host, half_layer=half_host, form=ruc_2m_diagnostic)
    for name in ("t2", "th2", "q2"):
        device[name] = cp.asarray(host[name])

    for name in names_2d:
        fields[name][...] = device[name]
    for name, array in device_3d.items():
        fields[name][...] = array

    lake = (device["lakemask"] == np.float32(1.0)) & bool(lakemodel)
    water = ((device["xland"] - np.float32(1.5) >= np.float32(0.0))
             & ~lake)
    seaice = (~water & ~lake) & (
        device["xice"] >= np.float32(params.xice_threshold))
    populations = cp.asnumpy(cp.stack([
        cp.count_nonzero(~water & ~seaice & ~lake),
        cp.count_nonzero(water), cp.count_nonzero(lake),
        cp.count_nonzero(seaice)]))
    return dict(zip(("land", "water", "lake", "sea_ice"),
                    map(int, populations)))


# --------------------------------------------------------------------------
# module_sf_sfcdiags_ruclsm.F, the flux=.true. arms.
# --------------------------------------------------------------------------

#: ``RSLF``'s polynomial coefficients, ``module_sf_sfcdiags_ruclsm.F:155-163``
#: ("saturation functions are from Thompson microphysics scheme").
_RSLF_C = (
    .611583699e03, .444606896e02, .143177157e01, .264224321e-1,
    .299291081e-3, .203154182e-5, .702620698e-8, .379534310e-11,
    -.321582393e-13,
)
#: ``RSIF``'s, ``:190-198``.
_RSIF_C = (
    .609868993e03, .499320233e02, .184672631e01, .402737184e-1,
    .565392987e-3, .521693933e-5, .307839583e-7, .105785160e-9,
    .161444444e-12,
)


def _horner(coefficients: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    """Fortran's nested form, evaluated innermost-first in float32.

    Written as the source's ``C0+X*(C1+X*(C2+...))`` rather than as a
    polynomial sum: the two are not the same float32 number, and the source's
    grouping is the one gfortran emits.
    """
    result = np.full(x.shape, np.float32(coefficients[-1]), dtype=np.float32)
    for coefficient in reversed(coefficients[:-1]):
        result = (np.float32(coefficient) + x * result).astype(np.float32)
    return result


def _rslf(pressure: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """``RSLF(P,T)``, ``:148-168``."""
    x = np.maximum(np.float32(-80.0),
                   (temperature - np.float32(273.16))).astype(np.float32)
    esl = _horner(_RSLF_C, x)
    return (np.float32(.622) * esl / (pressure - esl)).astype(np.float32)


def _rsif(pressure: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """``RSIF(P,T)``, ``:183-203``."""
    x = np.maximum(np.float32(-80.0),
                   (temperature - np.float32(273.16))).astype(np.float32)
    esi = _horner(_RSIF_C, x)
    return (np.float32(.622) * esi / (pressure - esi)).astype(np.float32)


def _saturation_mixing_ratio(pressure, temperature):
    """``:88-94`` / ``:129-136``: ice below 0 C, liquid above."""
    over_ice = (temperature - np.float32(273.15)) <= np.float32(0.0)
    return np.where(over_ice, _rsif(pressure, temperature),
                    _rslf(pressure, temperature)).astype(np.float32)


#: ``ROVCP`` as the surface driver hands it to ``SFCDIAGS_RUCLSM``: R/cp.
SFCDIAGS_ROVCP = np.float32(287.0 / 1004.5)


def sfcdiags_exner_powers(psfc) -> tuple[np.ndarray, np.ndarray]:
    """``(1.E5/PSFC)**ROVCP`` and ``(1.E-5*PSFC)**ROVCP``, ``:61-77``.

    The two host-side transcendentals of RUC's 2-m diagnostic: every T2
    and TH2 word is a float32 product of one of them.  They are REAL
    ``**`` in the source, which gfortran lowers to glibc's ``powf``, and
    they are taken here from :func:`gpuwm.core.noahmp_libm.powf_array`,
    glibc 2.39's ``powf`` as whole-array arithmetic, so every host gets
    that answer.  NumPy's own float32 ``power`` (what these were before)
    is the host's: glibc's on Linux, the MSVC runtime's on Windows, and
    NumPy 2.5's AVX-512 vector loop on an AVX-512 Linux machine such as
    the product boxes, which rounds other words, so T2 and TH2 moved with
    the CPU that ran the forecast.
    """
    pressure = np.asarray(psfc, dtype=np.float32)
    scale = powf_array(np.float32(1.0e5) / pressure, SFCDIAGS_ROVCP)
    inverse = powf_array(np.float32(1.0e-5) * pressure, SFCDIAGS_ROVCP)
    return scale, inverse


def _sfcdiags_ruclsm(host, *, t3d, qv3d, rho3d, p3d, cqs, half_layer=None,
                     form="flux") -> None:
    """``SFCDIAGS_RUCLSM``, ``module_sf_sfcdiags_ruclsm.F:7-146``.

    Only the ``flux = .true.`` arms exist here: ``flux`` is a hardcoded local
    (``:47-48``), so the alternatives are dead in the pinned object.

    Two properties separate this from WRF's ordinary SFCDIAGS and are the
    reason RUC must not borrow it.  T2 is CLAMPED into ``[min(TSK,T1),
    max(TSK,T1)]``, so an unstable column cannot diagnose a 2-m temperature
    outside the pair that bracket it.  Q2 is built from a QSFC PROXY --
    ``qlev1 + QFX/(RHO*CQS)`` -- rather than from QSFC, which the comment at
    ``:97-99`` says is deliberate for densely vegetated columns; it is then
    clamped between QSFCmr and qlev1 and saturation-capped at T2.
    """
    cp_air = np.float32(1004.5)
    rho = rho3d
    # :56.  "Assume that 2-m pressure also equal to PSFC".
    psfc = host["psfc"]
    t1 = t3d
    scale, inverse = sfcdiags_exner_powers(psfc)

    # :61-77.  T2 through TH2, then the bracket clamp, then TH2 again.
    stable = host["chs2"] < np.float32(1.0e-5)
    th2 = np.where(
        stable,
        (t1 * scale).astype(np.float32),
        (host["tsk"] * scale
         - host["hfx"] / (rho * cp_air * host["chs2"])).astype(np.float32),
    ).astype(np.float32)
    t2 = (th2 * inverse).astype(np.float32)
    lower = np.minimum(host["tsk"], t1).astype(np.float32)
    upper = np.maximum(host["tsk"], t1).astype(np.float32)
    t2 = np.minimum(upper, np.maximum(lower, t2)).astype(np.float32)
    th2 = (t2 * scale).astype(np.float32)

    # :83-93.  The first-level saturation trim uses P3D, not PSFC.
    qsat1 = _saturation_mixing_ratio(p3d, t1)
    qlev1 = np.minimum(qsat1, qv3d).astype(np.float32)

    # :100-101.
    qsfcprox = (qlev1 + host["qfx"] / (rho * cqs)).astype(np.float32)
    qsfcmr = (host["qsfc"] / (np.float32(1.0) - host["qsfc"])).astype(
        np.float32)

    # :111-120.
    q2 = np.where(
        host["cqs2"] < np.float32(1.0e-5),
        qlev1,
        (qsfcprox - host["qfx"] / (rho * host["cqs2"])).astype(np.float32),
    ).astype(np.float32)
    # :127-128.
    q2 = np.minimum(np.maximum(qsfcmr, qlev1),
                    np.maximum(np.minimum(qsfcmr, qlev1), q2)).astype(
                        np.float32)
    # :131-141.  The final cap is at PSFC and T2.
    q2 = np.minimum(_saturation_mixing_ratio(psfc, t2), q2).astype(np.float32)

    if form == "log_profile":
        t2, th2, q2 = _sfcdiags_log_profile(
            tsk=host["tsk"], t1=t1, qlev1=qlev1, qsfcmr=qsfcmr,
            half_layer=half_layer, scale=scale, t2=t2, th2=th2, q2=q2)

    host["t2"] = np.ascontiguousarray(t2)
    host["th2"] = np.ascontiguousarray(th2)
    host["q2"] = np.ascontiguousarray(q2)


def _sfcdiags_log_profile(*, tsk, t1, qlev1, qsfcmr, half_layer, scale, t2,
                          th2, q2):
    """The operational RAP/HRRR branch's 2 m block, its
    ``module_sf_sfcdiags_ruclsm.F:150-179``, in the device epilogue's float32
    order (glibc ``logf``, as ``gfk_log``)."""
    from gpuwm.core.noahmp_libm import logf

    f32 = np.float32

    def log(values):
        flat = np.asarray(values, dtype=f32).reshape(-1)
        return np.array([logf(value) for value in flat],
                        dtype=f32).reshape(np.shape(values))

    def factor(fh):
        top = (f32(f32(2.0) + f32(0.05)) / (f32(0.05) + fh).astype(f32)).astype(f32)
        bottom = ((half_layer + f32(0.05)).astype(f32)
                  / (f32(0.05) + fh).astype(f32)).astype(f32)
        return (log(top) / log(bottom)).astype(f32)

    t2 = np.array(t2, dtype=f32, copy=True)
    th2 = np.array(th2, dtype=f32, copy=True)
    q2 = np.array(q2, dtype=f32, copy=True)
    d_t = (t1 - tsk).astype(f32)
    d_q = (qlev1 - qsfcmr).astype(f32)
    warm = d_t > f32(0.0)
    if np.any(warm):
        fh = np.minimum(np.maximum((f32(1.0) - (d_t / f32(10.0)).astype(f32)).astype(f32),
                                   f32(0.01)), f32(1.0)).astype(f32)
        fac = np.where(warm, factor(np.where(warm, fh, f32(1.0))), f32(0.0)).astype(f32)
        t2_alt = (tsk + (fac * d_t).astype(f32)).astype(f32)
        t2 = np.where(warm, t2_alt, t2).astype(f32)
        th2 = np.where(warm, (t2_alt * scale).astype(f32), th2).astype(f32)
    moist = d_q > f32(0.0)
    if np.any(moist):
        fh = np.minimum(np.maximum((f32(1.0) - (d_q / f32(0.003)).astype(f32)).astype(f32),
                                   f32(0.01)), f32(1.0)).astype(f32)
        fac = np.where(moist, factor(np.where(moist, fh, f32(1.0))), f32(0.0)).astype(f32)
        q2 = np.where(moist, (qsfcmr + (fac * d_q).astype(f32)).astype(f32),
                      q2).astype(f32)
    return t2, th2, q2


__all__ = [
    "C1SN",
    "C2SN",
    "DEFAULT_VEGETATION_DATASET",
    "DEFINED_ILNB",
    "ISNCOVR_OPT",
    "RUC_MEASURED_INERT_CARRIERS",
    "RUC_MEASURED_LIVE_CARRIERS",
    "RUC_DIAGNOSTICS_2D",
    "RUC_FRACTIONAL_SEAICE_FIELDS",
    "RUC_PACKAGE_STATE_FIELDS",
    "RUC_PROFILE_BINDING",
    "RUC_RUNTIME_RESTRICTIONS",
    "RUC_STATE_2D",
    "RUC_STATE_3D",
    "RUC_STATE_BINDING",
    "RucRuntimeParameters",
    "SEAICE_ALBEDO_DEFAULT",
    "XICE_THRESHOLD",
    "ruc_cold_start",
    "ruc_device_sfctmp_sets",
    "ruc_lsm_step",
]
