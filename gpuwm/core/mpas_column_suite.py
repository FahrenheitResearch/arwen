"""Selectable MPAS column-physics suite (``run_mpas_column_suite``).

WHY A SECOND MODULE.  :mod:`gpuwm.core.mpas_column_batch` is the pinned
2026-08-10 counterparty contract: its phase-1 chain is hard-wired to legacy
RRTMG, the revised MM5 surface layer, Noah-MP and YSU
(``mpas_column_batch.py`` RunConfig at its constructor), and the MPAS port
pins that file's bytes by SHA-256.  A dispatch arm inside it would move the
bytes every existing execution anchor is keyed to.  This module is the
same seam with those four slots turned into SELECTORS, and nothing else
changed: the class subclasses :class:`MpasColumnBatchPhysics`, reuses its
species rows, phase-2 microphysics, restart law and bookkeeping, and
overrides only what a selector touches -- the configuration the driver is
built from, the soil column it is handed, the drag statistics, and where
phase 1 reads its raw momentum rates.  The frozen module's bytes do not
move, and a configuration that selects the frozen four is still served by
the frozen module (:func:`run_mpas_column_suite` refuses to duplicate it).

THE SELECTORS (each a table row; adding a scheme is a row, not a branch):

* ``radiation_scheme``: ``"rrtmg_legacy"`` (ra_physics=4, legacy RRTMG).
* ``surface_layer_scheme``: ``"revised_mm5"`` (1) or ``"mynn"`` (5).
* ``land_surface_scheme``: ``"noahmp"`` (4, four soil layers) or ``"ruc"``
  (3, nine soil LEVELS at WRF's own RUC depths,
  :func:`gpuwm.core.ruc.ruc_soil_geometry`).
* ``pbl_scheme``: ``"ysu"`` (1) or ``"mynn"`` (5, MYNN-EDMF at the
  configuration's own defaults).
* ``drag_scheme``: ``"off"`` or ``"gsl"`` (gwd_opt=3, WRF's GSL drag suite,
  :func:`gpuwm.core.terrain_drag.launch_gwdo_gsl`): large-scale gravity-wave
  drag and blocking from the large-scale statistics, small-scale
  gravity-wave drag and form drag from the small-scale ones.  The drag rides
  the PBL slot exactly as WRF's PBL driver adds it, on the PBL cadence.

Pairing is WRF's: YSU needs the revised MM5 surface layer, and the MYNN
surface layer needs the MYNN PBL (both refused by name otherwise).

RAW MOMENTUM RATES.  The frozen seam reads raw du/dv from YSU's retained
output (``driver.last_ysu``), which no other PBL scheme fills.  Every PBL
scheme in the driver ends its due call in ``PhysicsDriver._couple_pbl_slot``,
which copies the slot's raw A-grid rates -- AFTER the gwd_opt drag has been
added -- into ``driver.pbl_raw_rates`` (the seam always runs a positive
``bldt``, so those buffers exist).  Phase 1 here reads du/dv from them, so
YSU, MYNN and the drag are one code path.  Those buffers persist across
non-due steps and are on the driver restart manifest, which is the hold the
frozen seam builds by hand.

GEOMETRY THE DRAG NEEDS.  The column state carries no eta coordinate, so
``znu`` is formed from the nominal interface heights through the
ICAO standard atmosphere (the 6.5 K/km troposphere to 11 km, isothermal
216.65 K above), normalized between the nominal surface and ``p_top``.  The drag reads it
only for ``kpblmax`` (the highest level with znu > 0.6, WRF
module_bl_gwdo_gsl.F:162-164).  MPAS winds are earth-relative, so the drag
sees ``sina = 0, cosa = 1`` exactly as native MPAS hands its own drag
(mpas_atmphys_driver_gwdo.F, gwdo_from_MPAS).
"""

from __future__ import annotations

import dataclasses
import numbers
from datetime import datetime
from types import MappingProxyType

import numpy as np

from gpuwm.core.mpas_column_batch import (
    _GRAUPEL_SCHEMES,
    _MP_PHYSICS_BY_SCHEME,
    _PHASE1,
    _PHASE2,
    _SPECIES_BY_SCHEME,
    _STATE_RESTART_FIELDS_BY_SCHEME,
    _OUTPUT_BUFFERS,
    RESTART_SCHEMA,
    MpasColumnBatchPhysics,
    MpasColumnPhysicsTendencies,
    _ColumnBatchState,
    _interface_weights,
    _require_positive,
    resolve_column_batch_cadences,
)

__all__ = [
    "DRAG_SCHEMES",
    "MpasColumnSuiteTendencies",
    "FROZEN_SELECTION",
    "GENERATION_SELECTORS",
    "LAND_SURFACE_SCHEMES",
    "PBL_SCHEMES",
    "RADIATION_SCHEMES",
    "SURFACE_LAYER_SCHEMES",
    "MpasColumnSuitePhysics",
    "orographic_statistic_names",
    "resolve_generation",
    "resolve_suite_selection",
    "run_mpas_column_suite",
]

#: Selector rows.  Each value is the RunConfig assignment the row makes.
RADIATION_SCHEMES = MappingProxyType({
    "rrtmg_legacy": {"ra_physics": 4, "ra_rrtmg_variant": "legacy"},
})
SURFACE_LAYER_SCHEMES = MappingProxyType({
    "revised_mm5": {"sf_sfclay_physics": 1},
    "mynn": {"sf_sfclay_physics": 5},
})
LAND_SURFACE_SCHEMES = MappingProxyType({
    "noahmp": {"sf_surface_physics": 4, "num_soil_layers": 4},
    "ruc": {"sf_surface_physics": 3, "num_soil_layers": 9},
})
PBL_SCHEMES = MappingProxyType({
    "ysu": {"bl_pbl_physics": 1},
    "mynn": {"bl_pbl_physics": 5},
})
DRAG_SCHEMES = MappingProxyType({
    "off": {"gwd_opt": 0},
    "gsl": {"gwd_opt": 3},
})

#: The four slots the frozen ``mpas_column_batch`` hard-wires.  A selection
#: equal to this is the frozen seam's configuration and is served by the
#: frozen module, never re-served here under a second identity.
FROZEN_SELECTION = MappingProxyType({
    "radiation_scheme": "rrtmg_legacy",
    "surface_layer_scheme": "revised_mm5",
    "land_surface_scheme": "noahmp",
    "pbl_scheme": "ysu",
    "drag_scheme": "off",
})

_SELECTOR_TABLES = MappingProxyType({
    "radiation_scheme": RADIATION_SCHEMES,
    "surface_layer_scheme": SURFACE_LAYER_SCHEMES,
    "land_surface_scheme": LAND_SURFACE_SCHEMES,
    "pbl_scheme": PBL_SCHEMES,
    "drag_scheme": DRAG_SCHEMES,
})

#: Restart schema of THIS seam.  Distinct from the frozen one so a payload
#: of one can never restore into the other.
SUITE_RESTART_SCHEMA = "mpas-column-suite-v1"

#: Generation selectors the seam hands its RunConfig, each with the values
#: the engine's own ``validate_run_config`` admits and the slot it belongs
#: to.  A selector is passed only when its slot runs the scheme it selects
#: a generation OF: a ``bl_mynn_version`` on a YSU seam would be recorded
#: and never read, which is configuration the run does not have.  Adding a
#: selector is a row here; an empty mapping is the RunConfig defaults, so a
#: seam built with no generation is byte-for-byte the seam before this
#: table existed.
GENERATION_SELECTORS = MappingProxyType({
    "aer_opt": {
        "values": (0, 3),
        "requires": ("radiation_scheme", ("rrtmg_legacy",))},
    "tempo_ml_for_bl_nc": {
        "values": (False, True),
        "requires": ("microphysics_scheme", ("tempo",))},
    "tempo_surface_emission_before": {
        "values": (False, True),
        "requires": ("microphysics_scheme", ("tempo",))},
    "thompson_version": {
        "values": ("wrf_461", "wrf_39_noaa"),
        "requires": ("microphysics_scheme", ("thompson_aero",))},
    "thompson_fork_snow_fall": {
        "values": ("blend", "wrf_39_noaa"),
        "requires": ("microphysics_scheme", ("thompson_aero",))},
    "bl_mynn_version": {
        "values": ("wrf_461", "gsd_41"),
        "requires": ("pbl_scheme", ("mynn",))},
    "bl_mynn_mixlength": {
        "values": (1, 2),
        "requires": ("pbl_scheme", ("mynn",))},
    # MYNN number-species mixing by family (gpuwm/config.py
    # MYNN_QN_MIX_KEYS).  The mixed rates leave phase 1 through
    # MpasColumnPhysicsTendencies.scalar_rates; the engine's validator
    # refuses them on a microphysics that does not carry the family.
    "bl_mynn_mixaerosols": {
        "values": (0, 1),
        "requires": ("pbl_scheme", ("mynn",))},
    "bl_mynn_mixnumcon": {
        "values": (0, 1),
        "requires": ("pbl_scheme", ("mynn",))},
    "mynn_sfclay_variant": {
        "values": ("wrf_461", "gsl_wrf39"),
        "requires": ("surface_layer_scheme", ("mynn",))},
    "rrtmg_cloud_optics_form": {
        "values": ("wrf_461", "noaa_wrf39"),
        "requires": ("radiation_scheme", ("rrtmg_legacy",))},
    "rrtmg_aer3_type_source": {
        "values": ("rural", "landmask"),
        "requires": ("radiation_scheme", ("rrtmg_legacy",))},
})


@dataclasses.dataclass(frozen=True)
class MpasColumnSuiteTendencies(MpasColumnPhysicsTendencies):
    """The frozen phase-one result plus rates for non-mass species.

    ``scalar_rates`` maps a transported species' own row name to its raw
    A-grid rate (``[level, column]`` float32, seam-owned, valid until the
    next phase 1): MYNN's ``nwfa``/``nifa`` under bl_mynn_mixaerosols=1 and
    ``nc``/``ni`` under bl_mynn_mixnumcon=1 (gpuwm/config.py
    MYNN_QN_MIX_KEYS).  Empty when nothing is mixed.  A subclass rather
    than a field on the frozen contract, whose module bytes the MPAS port
    pins.
    """

    scalar_rates: object = dataclasses.field(
        default_factory=lambda: MappingProxyType({}))


def resolve_generation(generation, *, microphysics_scheme, selection):
    """Validate a generation mapping against the selected slots.

    Pure host logic.  Returns the RunConfig assignments (a plain dict,
    empty for ``None``).  The engine's ``validate_run_config`` still judges
    the combined configuration afterwards; this table only refuses what it
    would accept silently -- a selector for a scheme the seam does not run.
    """
    if generation is None:
        return {}
    chosen = dict(generation)
    slots = dict(selection, microphysics_scheme=microphysics_scheme)
    out = {}
    for key, value in chosen.items():
        row = GENERATION_SELECTORS.get(key)
        if row is None:
            raise ValueError(
                f"generation selector {key!r} has no row: the seam passes "
                f"only {sorted(GENERATION_SELECTORS)}")
        if value not in row["values"]:
            raise ValueError(
                f"{key} must be one of {list(row['values'])}, got {value!r}")
        slot, schemes = row["requires"]
        if slots.get(slot) not in schemes:
            raise ValueError(
                f"{key}={value!r} selects a generation of {list(schemes)} "
                f"but this seam's {slot} is {slots.get(slot)!r}; it would be "
                "recorded and never read")
        out[key] = value
    return out


def orographic_statistic_names(drag_scheme: str) -> tuple[str, ...]:
    """The engine's static field names the selected drag reads."""
    from gpuwm.static.orographic import GWD_FIELDS
    gwd_opt = DRAG_SCHEMES[drag_scheme]["gwd_opt"]
    return tuple(GWD_FIELDS.get(gwd_opt, ()))


def resolve_suite_selection(*, radiation_scheme, surface_layer_scheme,
                            land_surface_scheme, pbl_scheme,
                            drag_scheme) -> dict:
    """Validate one selection and return its RunConfig assignments.

    Pure host logic, so the CPU contract tests cover every refusal.
    """
    chosen = {"radiation_scheme": radiation_scheme,
              "surface_layer_scheme": surface_layer_scheme,
              "land_surface_scheme": land_surface_scheme,
              "pbl_scheme": pbl_scheme,
              "drag_scheme": drag_scheme}
    assignments: dict = {}
    for key, value in chosen.items():
        table = _SELECTOR_TABLES[key]
        if not isinstance(value, str) or value not in table:
            raise ValueError(
                f"{key} must be one of {sorted(table)}, got {value!r}: a "
                "scheme without a row has no configuration to build")
        assignments.update(table[value])
    if pbl_scheme == "ysu" and surface_layer_scheme != "revised_mm5":
        raise ValueError(
            "pbl_scheme='ysu' needs surface_layer_scheme='revised_mm5': "
            "WRF's YSU reads the revised MM5 layer's exchange terms "
            "(module_pbl_driver.F, bl_pbl_physics=1 with sf_sfclay=1)")
    if surface_layer_scheme == "mynn" and pbl_scheme != "mynn":
        raise ValueError(
            "surface_layer_scheme='mynn' needs pbl_scheme='mynn': the MYNN "
            "surface layer publishes the closure terms only the MYNN PBL "
            "reads")
    if chosen == dict(FROZEN_SELECTION):
        raise ValueError(
            "this selection is the frozen seam's own four "
            "(gpuwm.core.mpas_column_batch.run_mpas_column_batch); serving "
            "it here would give one configuration two identities and two "
            "byte pins.  Construct the frozen seam instead")
    return {"selection": chosen, "config": assignments}


def _nominal_znu(z_interface_nominal_m, p_top_pa: float) -> np.ndarray:
    """An eta-like mass-level coordinate from the nominal interfaces.

    Hydrostatic standard atmosphere over the nominal column (float64 setup
    arithmetic, float32 storage like every other static here), normalized
    so the nominal surface is 1 and ``p_top`` is 0.  Only ``kpblmax``
    reads it.
    """
    z = np.asarray(z_interface_nominal_m, dtype=np.float64)
    zmid = 0.5 * (z[:-1] + z[1:]) - z[0]
    p0 = 101325.0
    p11 = p0 * (216.65 / 288.15) ** 5.25588
    p = np.where(
        zmid <= 11000.0,
        p0 * np.power(1.0 - 2.25577e-5 * np.minimum(zmid, 11000.0), 5.25588),
        p11 * np.exp(-(zmid - 11000.0) / 6341.62))
    znu = (p - float(p_top_pa)) / (p0 - float(p_top_pa))
    return np.clip(znu, 0.0, 1.0).astype(np.float32)


class _SuiteColumnState(_ColumnBatchState):
    """The frozen column state plus what the selectable slots read."""

    def __init__(self, cfg, fnm, fnp, p_top, terrain_height, xp, *,
                 species, znu):
        super().__init__(cfg, fnm, fnp, p_top, terrain_height, xp,
                         species=species)
        ny, nx = cfg.ny, cfg.nx
        # Earth-relative winds: native MPAS hands its drag sina=0, cosa=1.
        self.sina = xp.zeros((ny, nx), dtype=np.float32)
        self.cosa = xp.ones((ny, nx), dtype=np.float32)
        self.znu = xp.asarray(znu, dtype=np.float32)


class MpasColumnSuitePhysics(MpasColumnBatchPhysics):
    """The frozen two-phase seam with selectable phase-1 slots."""

    def __init__(self, *, n_levels, n_columns, dt,
                 radiation_seconds, surface_pbl_seconds,
                 radiation_scheme, surface_layer_scheme,
                 land_surface_scheme, pbl_scheme, drag_scheme,
                 start_time, latitude_deg, longitude_deg,
                 terrain_height_m, z_interface_nominal_m,
                 p_top_pa, dx_m,
                 cumulus_seconds=None, cumulus_scheme=None,
                 microphysics_scheme="wsm6",
                 gf_ishallow=None, dx_column_m=None,
                 landmask=1.0, xland=None, ivgtyp=10, isltyp=6,
                 vegfra=50.0, tsk=300.0, tmn=285.0, xice=0.0, snow=0.0,
                 snow_depth=0.0, soil_temperature=None,
                 soil_moisture=None, xice_threshold=None,
                 fractional_seaice=None, orographic_statistics=None,
                 wsm6_hail_opt=0, generation=None, cell_area_m2=None):
        # ---- pure validation FIRST (no device allocation before it) ----
        if not isinstance(n_levels, numbers.Integral) or n_levels < 3:
            raise ValueError(
                f"n_levels must be an integer >= 3, got {n_levels!r}")
        if not isinstance(n_columns, numbers.Integral) or n_columns < 1:
            raise ValueError(
                f"n_columns must be a positive integer, got {n_columns!r}")
        if not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime (UTC)")
        if microphysics_scheme not in _SPECIES_BY_SCHEME:
            raise ValueError(
                f"microphysics_scheme must be one of "
                f"{sorted(_SPECIES_BY_SCHEME)}, got {microphysics_scheme!r}")
        resolved = resolve_suite_selection(
            radiation_scheme=radiation_scheme,
            surface_layer_scheme=surface_layer_scheme,
            land_surface_scheme=land_surface_scheme,
            pbl_scheme=pbl_scheme, drag_scheme=drag_scheme)
        selection, assignments = resolved["selection"], resolved["config"]
        generation_assignments = resolve_generation(
            generation, microphysics_scheme=microphysics_scheme,
            selection=selection)
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
        if microphysics_scheme != "wsm6" and int(wsm6_hail_opt) != 0:
            raise ValueError(
                "wsm6_hail_opt is a WSM6 hail-mode knob; on a "
                f"microphysics_scheme={microphysics_scheme!r} seam it would "
                "be silently ignored.  Leave it 0")
        # Sea-ice classification is the land surface's own knob in WRF:
        # Noah-MP takes an explicit threshold, RUC resolves its threshold
        # from fractional_seaice (gpuwm/core/ruc_runtime.py).  Each refuses
        # the other's knob instead of reading it and ignoring it.
        if land_surface_scheme == "noahmp":
            if fractional_seaice is not None:
                raise ValueError(
                    "fractional_seaice is RUC's sea-ice switch; Noah-MP "
                    "takes xice_threshold")
            threshold = 0.5 if xice_threshold is None else float(
                xice_threshold)
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "xice_threshold must be a finite fraction in [0, 1], "
                    f"got {xice_threshold!r}")
            seaice = {}
        else:
            if xice_threshold is not None:
                raise ValueError(
                    "xice_threshold is Noah-MP's classification threshold; "
                    "RUC resolves its own from fractional_seaice "
                    "(0: 0.5, 1: 0.02)")
            if fractional_seaice not in (0, 1):
                raise ValueError(
                    "land_surface_scheme='ruc' needs fractional_seaice 0 or "
                    f"1 named explicitly, got {fractional_seaice!r}")
            threshold = None
            seaice = {"fractional_seaice": int(fractional_seaice)}
        if gf_ishallow is None:
            gf_ishallow = 1 if cumulus_scheme == "gf" else 0
        if gf_ishallow not in (0, 1):
            raise ValueError(
                f"gf_ishallow must be 0 or 1, got {gf_ishallow!r}")
        if gf_ishallow and cumulus_scheme != "gf":
            raise ValueError("gf_ishallow=1 requires cumulus_scheme='gf'")
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
        terrain = np.asarray(terrain_height_m, dtype=np.float32).reshape(-1)
        if terrain.shape != (ncol,):
            raise ValueError(
                f"terrain_height_m must have shape ({ncol},), got "
                f"{terrain.shape}")
        dx_column = None
        if dx_column_m is not None:
            dx_column = np.asarray(dx_column_m, dtype=np.float32).reshape(-1)
            if dx_column.shape != (ncol,):
                raise ValueError(
                    f"dx_column_m must have one value per column ({ncol},), "
                    f"got {dx_column.shape}")
            if not np.all(np.isfinite(dx_column)) or np.any(dx_column <= 0.0):
                raise ValueError("dx_column_m must be finite and positive")
        # The soil column: the land surface's own geometry, never a default.
        n_soil = int(assignments["num_soil_layers"])
        soil = {}
        for name, value in (("soil_temperature", soil_temperature),
                            ("soil_moisture", soil_moisture)):
            if value is None:
                raise ValueError(
                    f"{name} is required: land_surface_scheme="
                    f"{land_surface_scheme!r} integrates a {n_soil}-level "
                    "soil column and a defaulted profile is substituted "
                    "state")
            array = (value if hasattr(value, "__cuda_array_interface__")
                     else np.asarray(value, dtype=np.float32))
            if tuple(array.shape) != (n_soil, ncol):
                raise ValueError(
                    f"{name} must have shape ({n_soil}, {ncol}) for "
                    f"land_surface_scheme={land_surface_scheme!r}, got "
                    f"{tuple(array.shape)}")
            soil[name] = array
        # Orographic statistics: exactly the drag's field set, both ways.
        need = orographic_statistic_names(drag_scheme)
        given = {} if orographic_statistics is None else dict(
            orographic_statistics)
        if need and orographic_statistics is None:
            raise ValueError(
                f"drag_scheme={drag_scheme!r} reads {list(need)}; without "
                "them the drag would run on zeros, which turns it off "
                "without saying so")
        missing = [name for name in need if name not in given]
        extra = sorted(name for name in given if name not in need)
        if missing or extra:
            raise ValueError(
                f"orographic_statistics for drag_scheme={drag_scheme!r} must "
                f"carry exactly {list(need)} (missing {missing}, extra "
                f"{extra})")
        oro = {}
        for name in need:
            value = np.asarray(given[name], dtype=np.float32).reshape(-1)
            if value.shape != (ncol,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"orographic statistic {name} must be finite with one "
                    f"value per column ({ncol},), got {value.shape}")
            oro[name] = value.reshape(1, ncol)
        top_nominal = float(np.asarray(
            z_interface_nominal_m, dtype=np.float64)[-1])
        znu = _nominal_znu(z_interface_nominal_m, p_top)

        from gpuwm.config import RunConfig
        from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY
        mp_physics = _MP_PHYSICS_BY_SCHEME[microphysics_scheme]
        aerosol_source = ({"mp28_aerosol_source": "synthetic"}
                          if microphysics_scheme == "thompson_aero" else {})
        radiation = dict(assignments)
        if radiation.pop("ra_rrtmg_variant") != "legacy":
            raise AssertionError("only the legacy RRTMG row exists")
        cfg = RunConfig(
            nx=ncol, ny=1, nz=nz, dx=dx, dy=dx,
            ztop=top_nominal, dt=float(cadences["dt"]),
            run_seconds=0.0, moist=True, mp_physics=mp_physics,
            wsm6_hail_opt=int(wsm6_hail_opt),
            **aerosol_source,
            **seaice,
            **radiation,
            ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
            cu_physics=cadences["cu_physics"],
            ishallow=int(gf_ishallow),
            radt=cadences["radt_minutes"],
            radt_minutes=cadences["radt_minutes"],
            bldt=cadences["bldt_minutes"],
            cudt_minutes=cadences["cudt_minutes"])
        if generation_assignments:
            import dataclasses
            from gpuwm.config import validate_run_config
            cfg = dataclasses.replace(cfg, **generation_assignments)
            validate_run_config(cfg)
        from gpuwm.config import soil_layer_count
        if soil_layer_count(cfg) != n_soil:
            raise ValueError(
                f"the engine resolves {soil_layer_count(cfg)} soil layers "
                f"for land_surface_scheme={land_surface_scheme!r}, the row "
                f"declares {n_soil}")
        self._cfg = cfg
        self._cadences = cadences
        self._dt = float(cadences["dt"])
        self._nz, self._ncol = nz, ncol
        self._start_time = start_time
        self._microphysics_scheme = microphysics_scheme
        self._has_graupel = microphysics_scheme in _GRAUPEL_SCHEMES
        self._species = _SPECIES_BY_SCHEME[microphysics_scheme]
        self._state_restart_fields = (
            _STATE_RESTART_FIELDS_BY_SCHEME[microphysics_scheme])
        self._selection = dict(selection)
        self._identity = {
            "schema": SUITE_RESTART_SCHEMA,
            "batch_schema": RESTART_SCHEMA,
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
            "xice_threshold": threshold,
            "fractional_seaice": seaice.get("fractional_seaice"),
            "xland_source": ("native" if xland is not None
                             else "derived-from-landmask"),
            "microphysics": microphysics_scheme,
            "selection": dict(selection),
            "n_soil_levels": n_soil,
        }
        area = None
        if cell_area_m2 is not None:
            area = np.asarray(cell_area_m2, dtype=np.float64).reshape(-1)
            if area.shape != (ncol,) or not np.all(np.isfinite(area)) or np.any(area <= 0):
                raise ValueError("cell_area_m2 must be finite and positive with one value per column")
        if cfg.tempo_surface_emission_before and microphysics_scheme == "tempo" and area is None:
            raise ValueError("native TEMPO surface emissions require supplied cell_area_m2; using nominal spacing would change the source rate on a variable mesh")
        if generation_assignments:
            # Only a seam that selected a generation carries the key, so a
            # restart payload of a generation-less seam is unchanged.
            self._identity["generation"] = dict(generation_assignments)

        # ---- device construction from here on --------------------------
        import cupy as cp
        self._cp = cp
        self._cell_area_m2 = None if area is None else cp.asarray(area)

        state = _SuiteColumnState(cfg, fnm, fnp, p_top, terrain, cp,
                                  species=self._species, znu=znu)
        self._state = state
        self._species_buffers = {
            name: getattr(state, name) for name in self._species}

        def surface(values):
            if np.ndim(values) == 0:
                return values
            if hasattr(values, "__cuda_array_interface__"):
                return values.reshape(1, ncol)
            return np.asarray(values).reshape(1, ncol)

        def soil_column(values):
            if hasattr(values, "__cuda_array_interface__"):
                return values.reshape(values.shape[0], 1, ncol)
            values = np.asarray(values)
            return values.reshape(values.shape[0], 1, ncol)

        self._momentum_cumulus = (
            _capturing_momentum_cumulus() if cumulus_scheme == "ntiedtke" else None)
        self._driver = _initialize_suite_driver(
            state, cfg,
            landmask=surface(landmask),
            xland=None if xland is None else surface(xland),
            xice_threshold=threshold,
            tsk=surface(tsk),
            soil_temperature=soil_column(soil["soil_temperature"]),
            soil_moisture=soil_column(soil["soil_moisture"]),
            ivgtyp=surface(ivgtyp), isltyp=surface(isltyp),
            vegfra=surface(vegfra), tmn=surface(tmn),
            xice=surface(xice), snow=surface(snow),
            snow_depth=surface(snow_depth),
            glw=None,
            radiation_start_time=start_time,
            radiation_latitude=lat.reshape(1, ncol),
            radiation_longitude=lon.reshape(1, ncol),
            terrain_drag_static=(dict(oro, HGT_M=terrain.reshape(1, ncol))
                                 if oro else None),
            cumulus=self._momentum_cumulus)
        if not {"du", "dv"} <= set(self._driver.pbl_raw_rates):
            raise RuntimeError(
                "the driver holds no raw PBL momentum rates "
                f"({sorted(self._driver.pbl_raw_rates)}); the seam runs a "
                "positive bldt precisely so it does, and without them phase "
                "1 has no A-grid momentum to return")

        self._aerosol_init = None
        if microphysics_scheme == "thompson_aero":
            # Same placeholder reset as the frozen seam (its docstring,
            # FIRST CONTACT): thompson_init runs for real at first contact.
            for name in ("nwfa", "nifa", "nwfa2d"):
                getattr(state, name)[...] = cp.float32(0.0)

        shape3 = (nz, 1, ncol)
        iface3 = (nz + 1, 1, ncol)
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
        self._out = {name: cp.zeros((nz, ncol), dtype=cp.float32)
                     for name in _OUTPUT_BUFFERS}
        # Seam-owned rate buffers for the number species MYNN mixes on
        # this configuration (empty unless a mixing key is on and the
        # microphysics carries the family).  Not on the restart manifest:
        # the engine validator admits these keys only with a PBL call on
        # every model step, so every phase 1 rewrites them before any
        # read, and a restored seam's first phase 1 refills them.
        from gpuwm.config import mynn_mixed_number_species
        mixed = mynn_mixed_number_species(cfg)
        missing = [name for name in mixed if name not in self._species]
        if missing:
            raise ValueError(
                f"MYNN mixes {missing}, which microphysics_scheme="
                f"{microphysics_scheme!r} does not transport; its rates "
                "would have no species to land on")
        self._scalar_out = {name: cp.zeros((nz, ncol), dtype=cp.float32)
                            for name in mixed}
        self.last_scalar_rates = MappingProxyType({})
        self._step_index = 0
        self._pending_phase = _PHASE1
        state.elapsed_seconds = 0.0

    # ------------------------------------------------------------------
    @property
    def mixed_number_species(self) -> tuple[str, ...]:
        """Species whose MYNN mixing rates phase 1 returns (row names)."""
        return tuple(getattr(self, "_scalar_out", None) or ())

    @property
    def selection(self) -> dict:
        """The five selector values this seam was built with."""
        return dict(self._selection)

    @property
    def generation(self) -> dict:
        """The generation selectors this seam handed its RunConfig."""
        return dict(self._identity.get("generation", {}))

    @property
    def n_soil_levels(self) -> int:
        return int(self._identity["n_soil_levels"])

    @property
    def last_noahmp_census(self):
        census = getattr(self._driver, "last_noahmp_census", None)
        return None if census is None else dict(census)

    @property
    def last_ruc_census(self):
        census = getattr(self._driver, "last_ruc_census", None)
        return None if census is None else dict(census)

    def soil_state(self) -> dict:
        """Device copies of the soil column (``[level, column]``)."""
        f = self._driver.fields
        ncol = self._ncol
        return {name: f[name].reshape(f[name].shape[0], ncol).copy()
                for name in ("tslb", "smois") if name in f}

    def run_phase1(self, **kwargs) -> MpasColumnPhysicsTendencies:
        """The frozen phase 1, with du/dv from the PBL slot's raw rates.

        Everything but the momentum source is the frozen implementation:
        the parent returns its result built on the seam-owned output
        buffers, and the parent's ``last_ysu`` read finds None under any
        PBL scheme other than YSU.  Momentum is then taken from
        ``driver.pbl_raw_rates`` -- the raw A-grid slot rates after the
        drag -- for every scheme, YSU included, so the slot has one
        momentum path whichever scheme filled it.
        """
        result = super().run_phase1(**kwargs)
        raw = self._driver.pbl_raw_rates
        nz, ncol = self._nz, self._ncol
        self._out["du"][...] = raw["du"].reshape(nz, ncol)
        self._out["dv"][...] = raw["dv"].reshape(nz, ncol)
        if self._momentum_cumulus is not None:
            momentum = self._momentum_cumulus.last_momentum
            if momentum is None:
                raise RuntimeError("the selected momentum-producing cumulus returned no raw A-grid pair")
            self._cp.add(self._out["du"], momentum[0].reshape(nz, ncol), out=self._out["du"])
            self._cp.add(self._out["dv"], momentum[1].reshape(nz, ncol), out=self._out["dv"])
        scalar_out = getattr(self, "_scalar_out", None) or {}
        if scalar_out:
            # MYNN's mixed number-species rates.  The driver holds them as
            # PhysicsTendencies.extra_scalars (physics.py _run_mynn_pbl),
            # mass-coupled by chm = c1h*mu + c2h, which is exactly 1.0 on
            # this state (module docstring), so they ARE the raw rates.
            # A step without a PBL call would hand back the previous
            # step's rates as this step's; the validator forbids that
            # cadence and this refuses it if it ever arrives.
            if not result.surface_pbl_ran:
                raise RuntimeError(
                    "MYNN number-species mixing needs a PBL call on every "
                    "phase 1; this step had none, so its rates would be "
                    "the previous step's")
            extra = getattr(self._driver.pbl_tendencies,
                            "extra_scalars", None) or {}
            if set(extra) != set(scalar_out):
                raise RuntimeError(
                    f"MYNN returned mixing rates for {sorted(extra)}; this "
                    f"configuration mixes {sorted(scalar_out)}")
            for name, buffer in scalar_out.items():
                buffer[...] = extra[name].reshape(nz, ncol)
            result = MpasColumnSuiteTendencies(
                **{item.name: getattr(result, item.name)
                   for item in dataclasses.fields(result)},
                scalar_rates=MappingProxyType(dict(scalar_out)))
        # The hex backend's frozen phase one keeps only the masses of this
        # result; it reads the extra rates here (hexcore
        # cuda_suite_physics_v841 _with_scalar_rates).
        self.last_scalar_rates = getattr(
            result, "scalar_rates", MappingProxyType({}))
        return result


def _capturing_momentum_cumulus():
    """Retain the native New Tiedtke A-grid pair before ARW face coupling.

    MPAS adds this pair to the PBL rates before edge reconstruction. The
    callable owns no parameterization arithmetic and is called every step.
    """
    from gpuwm.core.ntiedtke import NewTiedtke

    class Capturing(NewTiedtke):
        last_momentum = None

        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if result.rucuten is None or result.rvcuten is None:
                raise RuntimeError("New Tiedtke returned no momentum pair for the MPAS seam")
            self.last_momentum = (result.rucuten, result.rvcuten)
            return result

    return Capturing()


def _initialize_suite_driver(state, cfg, **kwargs):
    """``initialize_physics`` for the selected slots.

    Noah-MP takes the explicit sea-ice threshold; RUC refuses one (it
    resolves its own), so the argument is passed only to Noah-MP.
    """
    from gpuwm.core.physics import initialize_physics
    if int(cfg.sf_surface_physics) != 4:
        kwargs.pop("xice_threshold")
    return initialize_physics(state, cfg, **kwargs)


def run_mpas_column_suite(**kwargs) -> MpasColumnSuitePhysics:
    """Construct the selectable MPAS column-physics seam (keyword-only)."""
    return MpasColumnSuitePhysics(**kwargs)
