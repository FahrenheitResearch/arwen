"""The writer's history inventory, reusable without a device or data arrays.

The live writer and disk planner walk the same state and physics mappings.
The planner supplies shape-only carriers from the existing allocation schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np

from gpuwm.config import RunConfig, SASE_PBL_SCHEME, radiation_enabled
from gpuwm.core.physics_inventory import (PHYSICS_SLOT_DISPATCH,
                                          REFL_10CM_MICROPHYSICS,
                                          physics_driver_required)
from gpuwm.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS
from gpuwm.io.sfire_schema import (SFIRE_VOLUME_FIELDS, SFIRE_CLASS_FIELDS,
                                   sfire_history_shapes)

CORE_DIRECT_STATE_FIELDS = {
    "U": "u", "V": "v", "W": "w", "PH": "php", "MU": "mup",
}
MOISTURE_STATE_FIELDS = {
    "QVAPOR": "qv", "QCLOUD": "qc", "QRAIN": "qr",
}

_OUTPUT_FIELDS = {
    "TSK": "tsk", "T2": "t2", "TH2": "th2", "Q2": "q2",
    "U10": "u10", "V10": "v10", "UST": "ust", "HFX": "hfx",
    "QFX": "qfx", "LH": "lh", "PBLH": "pblh",
    "GRDFLX": "grdflx", "PSIM": "psim", "PSIH": "psih",
}

# WRF Registry names for the single-layer canopy's component energy.
# Grid HFX/LH/GRDFLX already include the rural blend; these retain the
# canopy-only terms so its energy split can be diagnosed independently.
_SLUCM_OUTPUT_FIELDS = {
    "FRC_URB2D": "frc_urb2d", "SH_URB": "sh_urb2d",
    "LH_URB": "lh_urb2d", "G_URB": "g_urb2d",
    "RN_URB": "rn_urb2d", "TS_URB": "ts_urb2d",
    "TR_URB": "tr_urb2d", "TB_URB": "tb_urb2d",
    "TG_URB": "tg_urb2d", "TC_URB": "tc_urb2d",
}


#: SPLIT SUBGRID-FLUX DIAGNOSTIC history names -> buffer key.  Both
#: channels are z-FACE fields on the mass column (stagger "Z", like W),
#: POSITIVE UPWARD, and share the deposit's lowest-level moist density
#: rho1 so that
#:   (F_vent[k]-F_vent[k+1] + F_diff[k]-F_diff[k+1])*dt/(rho1*thick_k)
#: reproduces the model's own scalar increment.

_SASE_FLUX_DIAG_OUTPUT = {"SASE_FQV_VENT": "fqv_vent",
                          "SASE_FQV_DIFF": "fqv_diff",
                          "SASE_FTH_VENT": "fth_vent",
                          "SASE_FTH_DIFF": "fth_diff"}


_Z_STAGGERED_MASS_FIELDS = frozenset(
    SCHEME_OUTPUT_FIELDS[key].netcdf_name
    for key in ("el_pbl", "exch_h", "exch_m")) | SFIRE_VOLUME_FIELDS

# Metadata follows the same table-driven inventory as live chemistry output.
CHEM_VAR_META: dict[str, tuple[str, str]] = {}


def live_state_history_fields(state) -> dict[str, object]:
    """Map live model arrays to their WRF Registry history names.

    The mapping is intentionally device/host agnostic.  Both frame builders
    consume it, preventing the asynchronous path from silently carrying a
    smaller scientific inventory than the synchronous writer.
    """
    fields: dict[str, object] = {}
    for output_name, state_name in (
            ("QICE", "qi"), ("QSNOW", "qs"), ("QGRAUP", "qg"),
            ("QNCLOUD", "nc"), ("QNRAIN", "nr"), ("QNICE", "ni"),
            ("QNSNOW", "ns"), ("QNGRAUPEL", "ng"),
            # Aerosol-aware Thompson's two transported aerosol scalars
            # (Registry/registry.new3d_wif:87/:89).  Presence-guarded like
            # every row above, and mp=28 is the only scheme that allocates
            # them, so no other run's inventory changes.  QNCLOUD is NOT a
            # new row here -- it has been mapped to state.nc since Morrison
            # landed; under mp=28 it simply starts carrying a PROGNOSTIC
            # droplet number instead of Morrison's diagnostic one, which is
            # a change in the values WRF also makes, not in the inventory.
            ("QNWFA", "nwfa"), ("QNIFA", "nifa"),
            # P3's rime mass and rime volume (mp_physics=50 only, and
            # presence-guarded like every row above).  WRF gives both the
            # history ``h`` in Registry.EM_COMMON:555-558.  th_old/qv_old
            # are deliberately NOT here: their IO string is ``rusd``
            # (:1598-1599) -- restart, no history -- so WRF does not
            # publish them either, and gpuwm follows.
            ("QIR", "qir"), ("QIB", "qib"),
            # WDM6's CCN reservoir (Registry.EM_COMMON:3031 declares
            # scalar:qnn,qnc,qnr for wdm6scheme).  It publishes under the
            # same QNCCN name NSSL's qnn does further down, and that is
            # safe rather than a collision: mp=16 allocates ``nn`` and mp=18
            # allocates ``qnn``, never both, so at most one row can fire on
            # any state.  QNCLOUD/QNRAIN need no new rows -- WDM6's nc/nr
            # are already mapped above, and under mp=16 they simply carry a
            # double-moment warm-rain pair instead of Morrison's.
            ("QNCCN", "nn"),
            # Milbrandt-Yau's hail NUMBER moment (Registry.EM_COMMON:3025
            # declares scalar:qh,qnc,qnr,qni,qns,qng,qnh for
            # milbrandt2mom).  QHAIL is already published by the NSSL-facing
            # loop below -- it is presence-guarded on state.qh, which mp=9
            # allocates -- but nothing published ``nh``, so an mp=9 parent
            # wrote eleven of its twelve transported species and the
            # offline-child lane's completeness check would fail on a
            # history file ArWen itself had written (audit R-017).  The row
            # is safe by the same never-both argument QNCCN above makes:
            # mp=9 allocates ``nh`` and mp=18 allocates ``qnh``, never both.
            ("QNHAIL", "nh")):
        value = getattr(state, state_name, None)
        if value is not None:
            fields[output_name] = value
    # The two 2-D surface aerosol emission rates.  Separate loop because
    # they are (ny, nx), not (nz, ny, nx): _dims_for routes them by shape,
    # and grouping them with the volume fields above would only obscure
    # that.  WRF's microphysics never writes either -- both are declared
    # OPTIONAL, INTENT(IN) on mp_gt_driver
    # (module_mp_thompson.F:1098) and are only READ, at :1247 and
    # :1320-1321.  So what a wrfout carries is thompson_init's derived
    # nwfa2d (:510) and the exactly-zero nifa2d nothing in
    # module_mp_thompson.F ever fills.
    for output_name, state_name in (
            ("QNWFA2D", "nwfa2d"), ("QNIFA2D", "nifa2d")):
        value = getattr(state, state_name, None)
        if value is not None:
            fields[output_name] = value
    for output_name, state_name in (
            ("QHAIL", "qh"), ("QNDROP", "qndrop"),
            ("QNRAIN", "qnr"), ("QNICE", "qni"),
            ("QNSNOW", "qns"), ("QNGRAUPEL", "qng"),
            ("QNHAIL", "qnh"), ("QNCCN", "qnn"),
            ("QVGRAUPEL", "qvolg"), ("QVHAIL", "qvolh")):
        value = getattr(state, state_name, None)
        if value is not None:
            fields[output_name] = value
    # The published subgrid energy, present only on a state whose PBL
    # closure owns one (SASE's prognostic e, or Shin-Hong's per-step TKE
    # diagnostic).  Scheme-qualified on purpose: WRF's ``TKE_PBL`` is a
    # Z-staggered MYJ/MYNN field on a different stagger, and the frame's
    # 2-D ``E`` is the Coriolis cosine term, not a turbulence quantity.
    # Named for the PRODUCER through the driver's own dispatch receipt,
    # so a Shin-Hong run can never publish its TKE under the SASE name;
    # a state without an attached driver keeps the historical SASE
    # label, which is the only producer such states ever had.
    e_sgs = getattr(state, "e_sgs", None)
    if e_sgs is not None:
        dispatch = getattr(getattr(state, "physics", None),
                           "scheme_dispatch", None)
        runner = (dispatch or {}).get("bl_pbl_physics")
        fields["TKE_SHINHONG" if runner == "_run_shinhong"
               else "TKE_SASE"] = e_sgs

    if getattr(state, "chem", None) is not None:
        from gpuwm.core.chem_history import history_fields as chem_history
        for name, (array, description, units) in chem_history(state).items():
            fields[name] = array
            CHEM_VAR_META[name] = (description, units)

    p_top = getattr(state, "p_top", None)
    if p_top is not None:
        fields["P_TOP"] = np.asarray(p_top, dtype=np.float32)
    for output_name, state_name in (("ZNU", "znu"), ("ZNW", "znw")):
        value = getattr(state, state_name, None)
        if value is not None:
            fields[output_name] = value

    # UP_HELI_MAX rides in every frame of a run that carries the
    # accumulator (allocated eagerly under nwp_diagnostics = 1), keeping
    # the async writer's frame schema constant.  The post-write reset is
    # the call sites' duty (gpuwm.core.uh_diag.reset_up_heli_max), never
    # this read-only builder's.
    existing_scratch = getattr(state, "existing_scratch", None)
    if existing_scratch is not None:
        up_heli_max = existing_scratch("up_heli_max")
        if up_heli_max is not None:
            fields["UP_HELI_MAX"] = up_heli_max

    physics = getattr(state, "physics", None)
    if physics is None:
        return fields
    microphysics = getattr(physics, "microphysics", None)
    if microphysics is not None:
        for output_name, field_name in (
                ("RAINNC", "rainnc"), ("SNOWNC", "snownc"),
                ("GRAUPELNC", "graupelnc"), ("HAILNC", "hailnc")):
            value = getattr(microphysics, field_name, None)
            if value is not None:
                fields[output_name] = value
    # Gate on "a land-surface scheme is routed", not on Noah's parameter
    # bundle: ``noah_params`` is scheme-2 state, so keying the snow/soil
    # history on it would silently drop TSLB/SMOIS/SH2O for any other LSM.
    # ``scheme_dispatch`` is the driver's own resolved routing, and
    # PhysicsDriver refuses to build when a selector value is unrouted.
    dispatch = getattr(physics, "scheme_dispatch", None)
    live_surface = getattr(physics, "fields", {})
    # MYNN's ten carried 3-D arrays and its four plume diagnostics exist only
    # under bl_pbl_physics=5, and wrfout does not auto-walk ``fields`` the way
    # the health collector does, so each emitted field is listed explicitly.
    # The gate is the driver's own resolved routing, for the same reason the
    # land-surface gate below is: a scheme that did not run must not appear to
    # have written state.  The listed keys are the scheme's *runtime* keys and
    # the emitted names come from the output schema, so no name here is
    # spelled twice and a key with no schema row raises rather than shipping
    # an anonymous float32.  ``exch_h``/``exch_m``/``rmol``/``kpbl`` are named
    # individually because they are shared EM_COMMON rows rather than members
    # of MYNN's own runtime inventories.
    if dispatch is not None:
        from gpuwm.core.physics_inventory import PHYSICS_SLOT_DISPATCH

        mynn_runner = PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][5]
        if dispatch.get("bl_pbl_physics") == mynn_runner:
            from gpuwm.core.physics_inventory import (
                MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D,
                MYNN_PBL_STATE_3D,
            )
            for field_name in (*MYNN_PBL_STATE_3D, *MYNN_PBL_DIAGNOSTICS_2D,
                               *MYNN_PBL_DIAGNOSTICS_INT_2D,
                               "exch_h", "exch_m", "rmol", "kpbl"):
                if field_name in live_surface:
                    fields[SCHEME_OUTPUT_FIELDS[field_name].netcdf_name] = \
                        live_surface[field_name]
    # Noah-MP's carried state and published diagnostics, on the same terms:
    # they exist only under sf_surface_physics=4, wrfout does not auto-walk
    # ``fields``, and the gate is the resolved routing.  The output names come
    # from the schema, which carries WRF's *external* names.  They used to be
    # the runtime keys upper-cased, which is not the same thing and was wrong
    # for every Noah-MP field but two: WRF writes ``TV``/``ISNOW``/``ZSNSO``,
    # never ``TVXY``/``ISNOWXY``/``ZSNSOXY``, so no WRF-name consumer could
    # find Noah-MP state in a gpuwm wrfout at all.
    if dispatch is not None:
        from gpuwm.core.physics_inventory import PHYSICS_SLOT_DISPATCH

        noahmp_runner = PHYSICS_SLOT_DISPATCH["sf_surface_physics"][4]
        if dispatch.get("sf_surface_physics") == noahmp_runner:
            from gpuwm.core.noahmp_runtime import (
                NOAHMP_DIAGNOSTICS_2D, NOAHMP_STATE_2D, NOAHMP_STATE_INT_2D,
                NOAHMP_STATE_SNOWSOIL_3D, NOAHMP_STATE_SNOW_3D,
            )
            for field_name in (*NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D,
                               *NOAHMP_STATE_SNOW_3D,
                               *NOAHMP_STATE_SNOWSOIL_3D,
                               *NOAHMP_DIAGNOSTICS_2D):
                if field_name in live_surface:
                    fields[SCHEME_OUTPUT_FIELDS[field_name].netcdf_name] = \
                        live_surface[field_name]
    # RUC's carried state and its four published driver locals, on the same
    # terms: they exist only under sf_surface_physics=3, wrfout does not
    # auto-walk ``fields``, and the gate is the resolved routing.  RUC's
    # external names happen to be its symbols upper-cased, but they are taken
    # from the schema anyway so that the coincidence is not essential; the
    # four ruc_* driver locals have no Registry counterpart and keep their
    # prefix so nothing mistakes them for WRF output.
    if dispatch is not None:
        from gpuwm.core.physics_inventory import PHYSICS_SLOT_DISPATCH

        ruc_runner = PHYSICS_SLOT_DISPATCH["sf_surface_physics"][3]
        if dispatch.get("sf_surface_physics") == ruc_runner:
            from gpuwm.core.ruc_runtime import (
                RUC_DIAGNOSTICS_2D, RUC_STATE_2D, RUC_STATE_3D,
            )
            for field_name in (*RUC_STATE_2D, *RUC_STATE_3D,
                               *RUC_DIAGNOSTICS_2D):
                if field_name in live_surface:
                    fields[SCHEME_OUTPUT_FIELDS[field_name].netcdf_name] = \
                        live_surface[field_name]
    if dispatch is not None:
        land_surface_active = dispatch.get("sf_surface_physics") is not None
    else:
        land_surface_active = getattr(physics, "noah_params", None) is not None
    if not land_surface_active:
        return fields
    for output_name, field_name in (
            ("SNOW", "snow"), ("SNOWH", "snowh"),
            ("SNOWC", "snowc"), ("TSLB", "tslb"),
            ("SMOIS", "smois"), ("SH2O", "sh2o"), ("CANWAT", "canwat"),
            # The land/soil IDENTITY the five rows above are the STATE of.
            # Same gate, same dict, same presence guard -- and the reason
            # they are here rather than left in memory is that a wrfout is
            # this product's boundary: gpuwm's own offline child reads a
            # child-grid history file back as its --child-surface-from
            # source and requires ISLTYP, TMN and VEGFRA among the nine
            # fields it will not fabricate (gpuwm.offline_child
            # ._SURFACE_REQUIRED_FIELDS).  Without these rows gpuwm's
            # history could not seed gpuwm's own child, which is how this
            # was found: on a real 12 km parent, and again on a nested d02.
            #
            # IVGTYP and SEAICE ride the same commit because they are the
            # same class and the same fix -- WRF core `misc` land identity
            # this driver has always carried and never published.  SEAICE
            # in particular closes a silent hole on the reader side: the
            # child's surface reader treats it as optional and substitutes
            # ZEROS when absent, so an ice-covered child was being warm-
            # started ice-free with nothing said.
            ("ISLTYP", "isltyp"), ("IVGTYP", "ivgtyp"),
            ("TMN", "tmn"), ("VEGFRA", "vegfra"), ("SEAICE", "xice")):
        if field_name in live_surface:
            fields[output_name] = live_surface[field_name]
    return fields


def physics_history_fields(physics) -> dict[str, object]:
    """WRF diagnostic name -> live FP32 device surface field.

    The six surface precipitation accumulators are emitted **always**,
    as zeros when the scheme that would fill them is not running.  That
    is WRF's contract, not a convenience: ``RAINC``, ``RAINSH``,
    ``RAINNC``, ``SNOWNC``, ``GRAUPELNC`` and ``HAILNC`` are core
    (``misc``) history rows with no package gate, so stock WRF allocates
    and writes all six in every run -- see
    ``gpuwm.io.wrf_output_schema.PRECIPITATION_OUTPUT_FIELDS`` for the
    rows and for the near neighbours that are excluded.

    gpuwm used to omit each of them whenever its producer was absent,
    which reads as "this run had no cumulus scheme" to a human and as
    "this file is broken" to a reader: every wrf-python/wrf-rust
    precipitation recipe computes ``RAINC + RAINNC`` unconditionally,
    because in WRF output both always exist.  Omitting ``RAINC`` under
    ``cu_physics=0`` therefore failed every downstream QPF product,
    which is exactly how this was found.

    ``RAINSH`` is always zero here, and that is a true statement rather
    than a placeholder: gpuwm implements no shallow-cumulus scheme, and
    zero is what WRF writes for ``shcu_physics=0``.
    """
    output = {name: physics.fields[field]
              for name, field in _OUTPUT_FIELDS.items()}
    if getattr(physics, "surface_energy_diag", False):
        output.update(ALBEDO=physics.fields["albedo"],
                      EMISS=physics.fields["emiss"])
        if "gsw" in physics.fields:
            output["GSW"] = physics.fields["gsw"]
    urban = getattr(physics, "urban", None)
    if urban is not None and urban.option == 1:
        output.update({name: urban.fields[field]
                       for name, field in _SLUCM_OUTPUT_FIELDS.items()})
    # Per-level radiative heating, output-only.  Both are live 3D
    # arrays (allocated :1869-1870, populated :2297-2300) that the
    # step consumes and then discards; the theta budget cannot be
    # closed by level without them.  Presence-guarded so a run with
    # radiation off carries the same inventory it always did.
    for _name, _attr in (("RTHRATLW", "rthratenlw"),
                         ("RTHRATSW", "rthratensw")):
        _value = getattr(physics, _attr, None)
        if _value is not None:
            output[_name] = _value
    if physics.sase_active:
        # Each scheme supplies its own boundary-layer height.  YSU
        # refreshes fields['pblh'] on every due call; SASE has no such
        # field of its own, so it computes its per-column
        # bulk-Richardson z_i here without touching the surface-layer
        # feed (whose convective-velocity term was calibrated against
        # the initialize_physics constant).
        output["PBLH"] = physics._sase_output_pblh()
        if physics.sase_flux_diag is not None:
            # SPLIT SUBGRID-FLUX DIAGNOSTIC (cfg.sase_flux_diag).  The
            # buffers hold the most recent due PBL call flux,
            # retained between calls at a positive PBL cadence.
            # This is the producer's instantaneous flux, not
            # a history-interval mean.  Zeros at the t=0 frame,
            # before any SASE step has run.
            output.update(
                {name: physics.sase_flux_diag[key]
                 for name, key in _SASE_FLUX_DIAG_OUTPUT.items()})
    if physics.hmix_k_diag is not None:
        if not physics.sase_active:
            # The km_opt = 4 producer's own buffers live in the
            # dycore's persistent scratch (prepare_fixed_tendencies
            # fills smag_km/smag_kh once per model step from the
            # time-t fields, which is the state WRF's
            # module_first_rk_step_part2 evaluates them at).  Copied
            # HERE rather than aliased so the published frame keeps a
            # constant schema from frame 0, when no step has run yet
            # and the slots do not exist.  READ-ONLY in the scratch.
            km = physics.state.existing_scratch("smag_km")
            kh = physics.state.existing_scratch("smag_kh")
            if km is not None and kh is not None:
                physics.hmix_k_diag["XKMH"][...] = km
                physics.hmix_k_diag["XKHH"][...] = kh
        output.update(physics.hmix_k_diag)
    if physics.radiation_active:
        output.update(SWDOWN=physics.fields["swdown"],
                      GLW=physics.fields["glw"])
        for name in ("albsol", "albbcksol"):
            if name in physics.fields:
                output[SCHEME_OUTPUT_FIELDS[name].netcdf_name] = physics.fields[name]
        if "albsol" in physics.fields and "gsw" in physics.fields:
            output["GSW"] = physics.fields["gsw"]
        # WRF's SWNORM, written only where slope_rad runs.
        if physics.topo_shortwave is not None:
            output["SWNORM"] = physics.fields["swnorm"]
        # OLR rides the same "only while radiation is running" rule as
        # SWDOWN/GLW, and additionally only while the LONGWAVE half is
        # a scheme that computes a top-of-atmosphere flux.  WRF's own
        # OLR row is core (``misc``, so stock WRF writes it in every
        # run); gpuwm's radiation diagnostics are absent rather than
        # zero when nothing produced them, and this follows that.
        if physics.olr is not None:
            output["OLR"] = physics.olr
        # CLDFRA, the cloud fraction the radiation radiated through, on
        # OLR's terms: present exactly when the attached scheme declares it
        # computes one, held from the most recent radiation call between
        # calls (the writer's attributes say so).
        cldfra = getattr(physics, "cldfra", None)
        if cldfra is not None:
            output["CLDFRA"] = cldfra
    output["RAINC"] = (physics._zero_accumulator() if physics.rainc is None
                       else physics.rainc)
    output["RAINSH"] = physics._zero_accumulator()
    microphysics = physics.microphysics if physics.mp_physics else None
    for name, attribute in (("RAINNC", "rainnc"), ("SNOWNC", "snownc"),
                            ("GRAUPELNC", "graupelnc"),
                            ("HAILNC", "hailnc")):
        value = (None if microphysics is None
                 else getattr(microphysics, attribute, None))
        output[name] = (physics._zero_accumulator() if value is None
                        else value)
    fire = getattr(physics, "fire", None)
    if fire is not None:
        output.update(fire.output_fields())
    return output


def metadata_history_fields(grid, static: dict) -> dict[str, object]:
    lat, lon = grid.latlon_mass()
    lat_u, lon_u = grid.latlon_u()
    lat_v, lon_v = grid.latlon_v()
    f, e = grid.coriolis_m()
    sina, cosa = grid.rotation_m()
    return {
        "XLAT": lat, "XLONG": lon, "XLAT_U": lat_u, "XLONG_U": lon_u,
        "XLAT_V": lat_v, "XLONG_V": lon_v,
        "MAPFAC_M": grid.mapfac_m(), "MAPFAC_U": grid.mapfac_u(),
        "MAPFAC_V": grid.mapfac_v(), "F": f, "E": e,
        "SINALPHA": sina, "COSALPHA": cosa, "HGT": static["HGT_M"],
        "LANDMASK": static["LANDMASK"], "LU_INDEX": static["LU_INDEX"],
    }


@dataclass(frozen=True)
class _HistoryShape:
    """A carrier descriptor, never an allocation of model data."""

    shape: tuple[int, ...]


def _run_config(cfg) -> RunConfig:
    """The resolved run settings the writer's inventory is read from.

    Refused for anything else, such as a grid named only by its size: the
    history a frame writes is set by the physics the writer emits (moisture
    species, scheme diagnostics, soil and radiation fields), so a grid-size
    object filled from the dataclass's dry defaults prices a moist 552x552x49
    child at 0.50 GB a frame where it wrote 1.2 GB, and a disk check built on
    that admits a run that stops partway when the disk fills.
    """
    cfg = getattr(cfg, "run", cfg)
    if isinstance(cfg, RunConfig):
        return cfg
    raise TypeError(
        "history disk pricing needs the domain's resolved RunConfig, got "
        f"{type(cfg).__name__}: the bytes a history frame writes depend on the "
        "physics the writer emits, which a grid size alone does not name")


def produced_history_shapes(cfg, *, include_reflectivity: bool = True
                            ) -> dict[str, tuple[int, ...]]:
    """On-disk inventory of the normal per-domain writer, without arrays.

    Shape carriers exercise the live writer's state, physics and geography
    mappings. Allocation shapes come from the same schema used by preflight.
    The initial analysis frame has no output-due reflectivity stash; callers
    pricing that frame can set ``include_reflectivity=False``.
    """
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.preflight import physics_array_shapes
    from gpuwm.core.topo_radiation import topo_shortwave_active
    cfg = _run_config(cfg)
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    mass, surface = (nz, ny, nx), (ny, nx)
    full = (nz + 1, ny, nx)
    carrier = lambda shape: _HistoryShape(tuple(shape))
    state = SimpleNamespace(**{name: carrier(shape)
                              for name, shape in state_array_shapes(cfg).items()})
    state.p_top = 0.0
    state.physics = None
    uh = carrier(surface) if int(cfg.nwp_diagnostics) == 1 else None
    state.existing_scratch = lambda name: uh if name == "up_heli_max" else None
    if physics_driver_required(cfg):
        allocated = physics_array_shapes(cfg)
        live_fields = {name.removeprefix("fields/"): carrier(shape)
                       for name, shape in allocated.items()
                       if name.startswith("fields/")}
        topography = topo_shortwave_active(cfg)
        if topography:
            # TopoShortwave attaches its output buffer after the ordinary
            # surface allocation pass, so it is not a preflight fields row.
            live_fields["swnorm"] = carrier(surface)
        physics = SimpleNamespace(
            fields=live_fields, state=state, mp_physics=cfg.mp_physics,
            surface_energy_diag=bool(cfg.surface_energy_diag),
            microphysics=SimpleNamespace(),
            scheme_dispatch={name: table.get(int(getattr(cfg, name)))
                             for name, table in PHYSICS_SLOT_DISPATCH.items()},
            sase_active=cfg.bl_pbl_physics == SASE_PBL_SCHEME,
            sase_flux_diag=None, hmix_k_diag=None,
            radiation_active=radiation_enabled(cfg),
            topo_shortwave=object() if topography else None,
            olr=carrier(surface) if "olr" in allocated else None,
            cldfra=carrier(mass) if "cldfra" in allocated else None,
            rainc=carrier(surface) if cfg.cu_physics else None,
            rthratenlw=carrier(mass), rthratensw=carrier(mass),
            _zero_accumulator=lambda: carrier(surface),
            _sase_output_pblh=lambda: carrier(surface),
        )
        fire_shapes = sfire_history_shapes(cfg)
        physics.fire = (SimpleNamespace(output_fields=lambda: {
            name: carrier(shape) for name, shape in fire_shapes.items()}) if fire_shapes else None)
        if int(cfg.sf_urban_physics) == 1:
            physics.urban = SimpleNamespace(option=1, fields={
                field: carrier(surface) for field in _SLUCM_OUTPUT_FIELDS.values()})
        if physics.sase_active and cfg.sase_flux_diag:
            physics.sase_flux_diag = {
                key: carrier(allocated[f"sase_flux_diag/{key}"])
                for key in _SASE_FLUX_DIAG_OUTPUT.values()}
        hmix = {name.removeprefix("hmix_k_diag/"): carrier(shape)
                for name, shape in allocated.items()
                if name.startswith("hmix_k_diag/")}
        physics.hmix_k_diag = hmix or None
        state.physics = physics

    # The base frame broadcasts PHB/PB even for a flat, one-column base
    # state. Staggered winds retain their faces rather than mass-cell counts.
    frame = {"T": carrier(mass),
             **{name: getattr(state, attribute)
                for name, attribute in CORE_DIRECT_STATE_FIELDS.items()},
             "PHB": carrier(full), "MUB": carrier(surface),
             "HGT": carrier(surface), "P": carrier(mass),
             "PB": carrier(mass), "PSFC": carrier(surface)}
    if cfg.moist:
        frame.update({name: getattr(state, attribute)
                      for name, attribute in MOISTURE_STATE_FIELDS.items()})
    frame.update(live_state_history_fields(state))
    if state.physics is not None:
        frame.update(physics_history_fields(state.physics))

    mass2 = carrier(surface)
    u2, v2 = carrier((ny, nx + 1)), carrier((ny + 1, nx))
    grid = SimpleNamespace(
        latlon_mass=lambda: (mass2, mass2), latlon_u=lambda: (u2, u2),
        latlon_v=lambda: (v2, v2), coriolis_m=lambda: (mass2, mass2),
        rotation_m=lambda: (mass2, mass2), mapfac_m=lambda: mass2,
        mapfac_u=lambda: u2, mapfac_v=lambda: v2,
    )
    frame.update(metadata_history_fields(
        grid, {"HGT_M": mass2, "LANDMASK": mass2, "LU_INDEX": mass2}))
    # Normal forecast preparation accepts precisely these microphysics
    # selectors. All produce the output-due radar field when moisture is on.
    if (include_reflectivity and cfg.moist
            and cfg.mp_physics in REFL_10CM_MICROPHYSICS):
        frame["REFL_10CM"] = carrier(mass)
    shapes = {name: tuple(value.shape) for name, value in frame.items()}
    if cfg.chem_sets:
        from gpuwm.core.chem_history import history_schema
        for name, (shape, description, units) in history_schema(cfg).items():
            shapes[name] = shape
            CHEM_VAR_META[name] = (description, units)
    for name in _Z_STAGGERED_MASS_FIELDS:
        if shapes.get(name) == mass:
            shapes[name] = full
    shapes.update(Times=(19,), XTIME=(), ITIMESTEP=())
    return shapes


def history_frame_bytes(cfg, selection=None, *,
                        include_reflectivity: bool = True) -> int:
    """CDF-2 payload plus a bounded allowance for its header metadata.

    Every numerical history field is four bytes; Times is a 19-byte char
    record padded to four bytes by the classic container. The header reserve
    covers projection/provenance attributes, dimension declarations and each
    selected variable's WRF attributes. It never scales with model levels.
    """
    from gpuwm.io.history_selection import HistorySelection

    shapes = produced_history_shapes(
        cfg, include_reflectivity=include_reflectivity)
    selection = HistorySelection() if selection is None else selection
    names = selection.select(shapes)
    payload = sum(4 * math.prod(shapes[name]) if name != "Times" else 20
                  for name in names)
    header = 65536 + 256 * len(names)
    return int(payload + header)
