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
On the CARD, in FP32.  :func:`ruc_lsm_step` passes
:data:`gpuwm.core.ruc_gpu.RUC_SFCTMP_DEVICE_LEAVES` and
:data:`gpuwm.core.ruc_gpu.RUC_SFCTMP_DEVICE_STAGES` into the driver, so the
four ``sfctmp`` branches and the snow-preparation prologue -- which together
are the whole of RUC's column arithmetic -- run as CUDA kernels.  There is no
switch that selects the host set: it exists for the oracle suite, which runs
on a machine with no card, and a forecast that quietly took it would be a
thousand times slower with no signal that it had.

That wiring is not a detail.  Until it landed the device leaves had ZERO
importers under ``gpuwm/`` -- the same failure mode the RUC soil ingest had --
so every published RUC forecast cost ran the host loop no matter what the
leaves could do.  ``tests/test_ruc_runtime.py`` gates the import edge
directly so it cannot rot back.

What still runs on the host is ``LSMRUC``'s own prologue and epilogue and the
``sfctmp`` dispatch's masking and mosaic recombination, all of them evaluated
over the whole column axis rather than column by column.  The cost is measured
in :mod:`tests.test_ruc_admission` and published in the registry warnings
rather than estimated.

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

from gpuwm.core.ruc import (RUC_DRIVER_COLUMN_FORCING,
                            RUC_DRIVER_COLUMN_STATE,
                            RUC_DRIVER_PROFILE_STATE,
                            RUC_SNOW_COVER_OPTION,
                            load_ruc_parameters, ruc_land_surface_step,
                            ruc_initialize_cold_start, ruc_soil_geometry)
from gpuwm.core.ruc_contract import (NUM_SOIL_LAYERS,
                                     RUC_PACKAGE_STATE_FIELDS,
                                     RUC_SOIL_LEVELS_M)

#: The vegetation dataset identifier gpuwm's land-use ingest produces, and the
#: only one this runner is exercised at.  ``gpuwm.core.ruc`` maps it to
#: VEGPARM's ``MODI-RUC`` section (21 categories, water 17, snow/ice 15).
DEFAULT_VEGETATION_DATASET = "MODIFIED_IGBP_MODIS_NOAH"

#: ``XICE_THRESHOLD``, WRF Registry default 0.5.  gpuwm has no namelist field
#: for it, so it is pinned here and published in
#: :data:`RUC_RUNTIME_RESTRICTIONS` rather than left implicit.
XICE_THRESHOLD = 0.5

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

#: Registry RUC package, 3-D half.  ``(NUM_SOIL_LAYERS, ny, nx)``.
RUC_STATE_3D = ("smfr3d", "keepfr3dflag")

#: The driver locals gpuwm publishes as diagnostics.  WRF keeps all four as
#: automatic arrays of ``LSMRUC`` and returns none of them, so these are
#: gpuwm additions, not Registry fields -- named ``ruc_*`` so a wrfout reader
#: cannot mistake them for WRF output.
RUC_DIAGNOSTICS_2D = ("ruc_infiltr", "ruc_smelt", "ruc_runoff1",
                      "ruc_runoff2")

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
        "precipitation_partition_is_the_EM_CORE==0_ARM",
        "PRECIPITATION reaches RUC as RAINBL (mm accumulated since the last "
        "surface call) plus the frozen fraction SR, and the phase split is "
        "module_sf_ruclsm.F:654-660: snowrat=rainbl*frzfrac, "
        "prcpms=(rainbl-snowratio)/dt/1000, with grauprat and icerat "
        "identically zero.",
        "This is the single most consequential restriction in the port and "
        "it is not a namelist knob.  LSMRUC has TWO precipitation "
        "partitions and #if (EM_CORE==1) picks the other one (:618-652), "
        "which takes rainncv, snowncv and graupelncv separately and derives "
        "snowrat/grauprat/icerat from the microphysics' own species.  WRF-ARW "
        "IS built with EM_CORE==1, so a stock ARW run does NOT take the arm "
        "gpuwm implements.  Every RUC oracle fixture in "
        "gpuwm/data/ruc/oracle was generated at EM_CORE==0, so the ported "
        "arm is the one with evidence behind it; the other has none.  The "
        "visible consequence is in graupel-bearing convection, where WRF "
        "sends RUC a separate graupel rate that raises new-snow density "
        "through rhonewsn and gpuwm sends none.",
    ),
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
        "moisture_species_supplied",
        "QV3D and QC3D only, from the model's own qv and qc.  A dry state "
        "supplies zeros for both through the physics scratch arrays.",
        "That is WRF's contract for LSMRUC itself -- it declares no other "
        "species -- so it is not a divergence at the call.  It IS a "
        "divergence in combination with the entry above: under EM_CORE==1 "
        "WRF also hands RUC snowncv and graupelncv, so the frozen species "
        "reach the LSM through the precipitation arm gpuwm does not take.",
    ),
    (
        "no_stochastic_perturbations",
        "spp_lsm=0 and no rstochcol / field_sf arrays are constructed.",
        "The SPP perturbation region is EM_CORE==1-only in LSMRUC and the "
        "pinned object does not contain it.  spp_lsm is refused at any "
        "other value rather than accepted and ignored.",
    ),
    (
        "no_mosaic_land_use_or_soil",
        "mosaic_lu=0, mosaic_soil=0, and landusef / soilctop / nlcat / "
        "nscat are not carried at all.",
        "gpuwm.core.ruc.ruc_surface_parameters is fail-closed on SOILVEGIN's "
        "mosaic arms, and LSMRUC's irrigation block (:984-1009) is gated on "
        "the same mosaic_lu==1, so the irrigation block is unreachable "
        "wherever SOILVEGIN is.  Neither is transcribed.",
    ),
    (
        "no_lake_bypass",
        "Lake columns are run as whatever their land-use category says.",
        "LSMRUC's lakemodel/lakemask bypass (:585-590) is EM_CORE==1-only "
        "and absent from the pinned object.  gpuwm carries a lakemask but "
        "RUC does not read it.",
    ),
    (
        "no_fractional_sea_ice",
        "FRACTIONAL_SEAICE==0.  The sea-ice ALBEDO/EMISS/TSK de-blending "
        "before the call (module_surface_driver.F:3461-3499) and the "
        "re-blending after it (:3529-3577) are not transcribed.  The "
        "unconditional ALBBCK = 0.65 sea-ice override at :3453-3459 IS.",
        "gpuwm has no fractional sea-ice state, and the de-blend divides by "
        "XICE.  A column is land, water or fully ice.",
    ),
    (
        "gsw_uses_the_live_albedo",
        "GSW = SWDOWN*(1-ALBEDO) is evaluated at the surface-call time from "
        "the albedo RUC itself last wrote.",
        "WRF builds GSW in the radiation driver "
        "(module_radiation_driver.F:3660) and the surface driver forwards "
        "whatever that left, so WRF's GSW carries the albedo from the last "
        "RADIATION call, not the last LSM call.  RUC updates ALBEDO every "
        "surface call (snow cover), so the two differ by the albedo drift "
        "over at most radt.  gpuwm's choice is the fresher of the two; it "
        "is a divergence either way and is recorded as one.",
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
        "lai_comes_from_the_table",
        "rdlai2d=False; LAI is SOILVEGIN's table value for the column's "
        "vegetation category, refreshed every call.",
        "WRF's rdlai2d=.true. path takes a prescribed 2-D LAI instead.  "
        "gpuwm's ingest has no monthly LAI field for RUC, and inventing one "
        "would be a fabricated forcing.",
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
        "That is what validation-candidate would require and this scheme is "
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

    def __init__(self, bundle=None, *,
                 dataset_identifier: str = DEFAULT_VEGETATION_DATASET):
        self.bundle = load_ruc_parameters() if bundle is None else bundle
        self.dataset_identifier = str(dataset_identifier)
        # Raises on a dataset gpuwm.core.ruc has no RUC VEGPARM section for,
        # rather than silently falling back to another table.
        self.vegetation = self.bundle.vegetation_for(self.dataset_identifier)
        self.zs, self.dzs = ruc_soil_geometry()
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
        return {
            "algorithm": "ruc-lsm-wrf-v4.6.1-v1",
            "wrf_source": "phys/module_sf_ruclsm.F:LSMRUC + "
                          "phys/module_sf_sfcdiags_ruclsm.F:SFCDIAGS_RUCLSM",
            "dataset_identifier": self.dataset_identifier,
            "vegetation_section": self.vegetation.name,
            "num_soil_layers": int(NUM_SOIL_LAYERS),
            "soil_level_depths_m": [float(v) for v in RUC_SOIL_LEVELS_M],
            "xice_threshold": float(XICE_THRESHOLD),
            "seaice_albedo_default": float(SEAICE_ALBEDO_DEFAULT),
            "isncovr_opt": int(ISNCOVR_OPT),
            "c1sn": float(C1SN),
            "c2sn": float(C2SN),
            "defined_ilnb": int(DEFINED_ILNB),
            "ilnb_chain": False,
            "column_solver": "host-fp32",
            "tables": payload,
        }


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
    if slab["tslb"].shape[0] != NUM_SOIL_LAYERS:
        raise ValueError(
            "RUC cold start needs a nine-level soil column, got "
            f"{slab['tslb'].shape[0]}")
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
    dt: float,
    itimestep: int,
    mosaic_lu: int,
    mosaic_soil: int,
    flag_sm_adj: int,
    spp_lsm: int,
) -> dict[str, int]:
    """One ``CASE (RUCLSMSCHEME)`` arm.  Mutates ``fields`` in place.

    Returns a small census -- land, water and sea-ice column counts -- so a
    test can tell "RUC ran" apart from "RUC ran on any land".  The counts are
    reconstructed from the same masks the driver dispatches on
    (``module_sf_ruclsm.F:824`` and ``:851``); RUC skips nothing, so they sum
    to the whole grid.
    """
    import cupy as cp

    if int(itimestep) < 1:
        raise ValueError("LSMRUC ktau is one-based and starts at 1")
    # Second line behind validate_run_config, at the seam that consumes each
    # value, so the registry's citation of this file is true for all four.
    if int(mosaic_lu) != 0 or int(mosaic_soil) != 0:
        raise ValueError(
            f"mosaic_lu={mosaic_lu}, mosaic_soil={mosaic_soil}: SOILVEGIN's "
            "mosaic arms are fail-closed in gpuwm.core.ruc, so LSMRUC's "
            "irrigation block is unreachable and neither is transcribed")
    if int(spp_lsm) != 0:
        # :446-450 assigns rstoch from pattern_spp_lsm, which is an OPTIONAL
        # argument present only under #if (EM_CORE==1).  spp_lsm=1 in the
        # pinned object would dereference an absent optional.
        raise ValueError(
            f"spp_lsm={spp_lsm}: LSMRUC:446-450 reads pattern_spp_lsm, an "
            "optional argument that exists only under EM_CORE==1, so a "
            "perturbed RUC run is not expressible in the pinned object")
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
        "albbck", "xland", "xice", "tmn", "shdmin", "shdmax", "vegfra",
        "rainbl", "sr", "ivgtyp", "isltyp", "psfc", "t2", "th2", "q2",
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
    ice = (device["xice"] >= np.float32(XICE_THRESHOLD)) & (
        device["xice"] <= np.float32(1.0))
    device["albbck"] = cp.where(
        ice, np.float32(SEAICE_ALBEDO_DEFAULT), device["albbck"]
    ).astype(cp.float32)

    temperature = cp.ascontiguousarray(atmosphere["temperature"][0])
    qv = cp.ascontiguousarray(atmosphere["qv"][0])
    qc = cp.ascontiguousarray(atmosphere["qc"][0])
    rho = cp.ascontiguousarray(atmosphere["rho"][0])
    dz1 = cp.ascontiguousarray(atmosphere["dz"][0])
    # :3506 -- p_phy, the layer MID pressure, into an argument named p8w.
    p_mid = cp.ascontiguousarray(atmosphere["pressure"][0])
    # module_radiation_driver.F:3660.  RUC takes ABSORBED shortwave.
    gsw = (device["swdown"] * (np.float32(1.0) - device["albedo"])).astype(
        cp.float32)

    values = {
        # forcing
        "z3d": dz1, "p8w": p_mid, "t3d": temperature, "qv3d": qv,
        "qc3d": qc, "rho3d": rho,
        "rainbl": device["rainbl"], "frzfrac": device["sr"],
        "glw": device["glw"], "gsw": gsw, "chs": device["chs"],
        "flqc": device["flqc"], "flhc": device["flhc"],
        "albbck": device["albbck"], "xland": device["xland"],
        "xice": device["xice"], "tbot": device["tmn"],
        "shdmin": device["shdmin"], "shdmax": device["shdmax"],
        "vegfra": device["vegfra"],
    }
    for name, argument in RUC_STATE_BINDING.items():
        values[argument] = device[name]
    for name, argument in RUC_PROFILE_BINDING.items():
        values[argument] = device_3d[name]
    missing = [name for name in (RUC_DRIVER_PROFILE_STATE
                                 + RUC_DRIVER_COLUMN_STATE
                                 + RUC_DRIVER_COLUMN_FORCING)
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
        myj=False, frpcpn=True, rdlai2d=False,
        mosaic_lu=int(mosaic_lu), mosaic_soil=int(mosaic_soil),
        iswater=params.iswater, isice=params.isice,
        xice_threshold=float(XICE_THRESHOLD),
        ilnb=int(DEFINED_ILNB), ilnb_chain=False,
        c1sn=float(C1SN), c2sn=float(C2SN),
        isncovr_opt=int(ISNCOVR_OPT),
        mminlu=params.dataset_identifier, parameters=params.bundle,
        leaves=leaves, stages=stages, arrays=device_arrays)

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
    # deliberately.  _sfcdiags_ruclsm raises (1e5/psfc) to R/cp with np.power
    # on a float32 array, and numpy's float32 power and CUDA's powf are two
    # different non-glibc functions -- the exact divergence class that put
    # 2 ULP into hfx on this lane and cost a session to find.  Moving it to
    # the card is a TRANSCENDENTAL POLICY decision (see
    # _RUC_PROVISIONAL_TRANSCENDENTALS in gpuwm.core.ruc), not a performance
    # one, and the answer must not change to make a call faster.  So exactly
    # the fields it reads come down and the three it writes go back up, which
    # is a bounded and named cost instead of the whole slab.
    diagnostic_inputs = ("psfc", "chs2", "cqs2", "tsk", "hfx", "qfx", "qsfc")
    host = {name: np.ascontiguousarray(cp.asnumpy(device[name]))
            for name in diagnostic_inputs}
    _sfcdiags_ruclsm(
        host,
        t3d=np.ascontiguousarray(cp.asnumpy(temperature)),
        qv3d=np.ascontiguousarray(cp.asnumpy(qv)),
        rho3d=np.ascontiguousarray(cp.asnumpy(rho)),
        p3d=np.ascontiguousarray(cp.asnumpy(p_mid)),
        cqs=np.ascontiguousarray(cp.asnumpy(cqs)))
    for name in ("t2", "th2", "q2"):
        device[name] = cp.asarray(host[name])

    for name in names_2d:
        fields[name][...] = device[name]
    for name, array in device_3d.items():
        fields[name][...] = array

    water = device["xland"] - np.float32(1.5) >= np.float32(0.0)
    seaice = (~water) & (device["xice"] >= np.float32(XICE_THRESHOLD))
    return {"land": int(cp.count_nonzero(~water & ~seaice)),
            "water": int(cp.count_nonzero(water)),
            "sea_ice": int(cp.count_nonzero(seaice))}


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


def _sfcdiags_ruclsm(host, *, t3d, qv3d, rho3d, p3d, cqs) -> None:
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
    rovcp = np.float32(287.0 / 1004.5)
    cp_air = np.float32(1004.5)
    rho = rho3d
    # :56.  "Assume that 2-m pressure also equal to PSFC".
    psfc = host["psfc"]
    t1 = t3d
    scale = np.power(np.float32(1.0e5) / psfc, rovcp).astype(np.float32)
    inverse = np.power(np.float32(1.0e-5) * psfc, rovcp).astype(np.float32)

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

    host["t2"] = np.ascontiguousarray(t2)
    host["th2"] = np.ascontiguousarray(th2)
    host["q2"] = np.ascontiguousarray(q2)


__all__ = [
    "C1SN",
    "C2SN",
    "DEFAULT_VEGETATION_DATASET",
    "DEFINED_ILNB",
    "ISNCOVR_OPT",
    "RUC_MEASURED_INERT_CARRIERS",
    "RUC_MEASURED_LIVE_CARRIERS",
    "RUC_DIAGNOSTICS_2D",
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
