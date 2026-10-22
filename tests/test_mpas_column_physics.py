"""CPU contract of the generic MPAS column-physics seam (no device).

The generic entry keeps no scheme list: these hold it to that.  Every
option the registry marks implemented must plan (with the partner WRF's
own pairing rule names, where the engine asks for one), the seam's own
refusals are about the seam (owned fields, unknown fields, missing soil,
inexact statics), and everything else that refuses is the engine's
validator speaking.
"""

import datetime

import numpy as np
import pytest

from gpuwm.core import mpas_column_suite as suite

import mpas_column_physics_cases as cases

NZ, NCOL = cases.NZ, 8
START = datetime.datetime(2021, 6, 1, 18, 0)

def _kwargs(physics, **overrides):
    plan = suite.column_physics_plan(physics)
    kwargs = dict(
        physics=physics, n_levels=NZ, n_columns=NCOL, dt=60.0,
        radiation_seconds=600.0, surface_pbl_seconds=60.0, start_time=START,
        latitude_deg=np.full(NCOL, 35.0), longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, cases.ZTOP, NZ + 1),
        p_top_pa=cases.P_TOP, dx_m=12000.0,
        surface=cases.surface(plan, NCOL), statics=cases.statics(plan, NCOL))
    kwargs.update(overrides)
    return kwargs


def _refuses(match, physics=None, **overrides):
    physics = dict(suite.MPAS_DEFAULT_PHYSICS) if physics is None else physics
    with pytest.raises(ValueError, match=match):
        suite.MpasColumnPhysics(**_kwargs(physics, **overrides))


def test_the_menu_is_the_registry():
    from gpuwm.physics_registry import component_options, registry_view
    menu = suite.physics_selector_menu()
    assert set(menu) == set(registry_view()["components"])
    for component, row in menu.items():
        selectable = {option_id for option_id, option
                      in component_options(component).items()
                      if option.get("selectors")}
        assert set(row["options"]) == selectable, component


@pytest.mark.parametrize('name', [
    'cumulus.grell-3d',
    'cumulus.grell-freitas',
    'cumulus.new-sas',
    'cumulus.new-tiedtke',
    'microphysics.fast-sbm-mp30',
    'microphysics.ferrier-aligo-mp5',
    'microphysics.kessler-mp1',
    'microphysics.lin-mp2',
    'microphysics.milbrandt2mom-mp9',
    'microphysics.morrison-mp10',
    'microphysics.nssl2-mp18',
    'microphysics.off',
    'microphysics.thompson-mp8',
    'microphysics.wdm6-mp16',
    'microphysics.wsm3-mp3',
    'microphysics.wsm5-mp4',
    'microphysics.wsm6r-mp106',
    'microphysics.wsm7-mp24',
    'pbl.ccpp-tke-edmf',
    'pbl.sase',
    'pbl.shinhong',
    'pbl.uw',
    'radiation.wrf-rrtm-dudhia',
    'urban.bep',
])
def test_every_registry_option_plans(name):
    plan = suite.column_physics_plan(cases.selections()[name])
    assert plan.species[:3] == ("qv", "qc", "qr")
    assert set(plan.mass_species) | set(plan.number_species) == set(
        plan.species)
    assert not set(plan.state_fields) & set(plan.species)
    assert plan.cadences is None


def test_the_default_selection_is_the_frozen_seams_species_and_state():
    from gpuwm.core import mpas_column_batch as frozen
    plan = suite.column_physics_plan(suite.MPAS_DEFAULT_PHYSICS)
    assert plan.species == frozen._SPECIES_BY_SCHEME["wsm6"]
    assert set(plan.state_fields) == set(
        frozen._STATE_RESTART_FIELDS_BY_SCHEME["wsm6"])
    assert plan.surface_accumulators == ("RAINNC", "SNOWNC", "GRAUPELNC")
    assert plan.n_soil_levels == 4 and plan.static_fields == ()


@pytest.mark.parametrize("scheme,mp_physics,frozen_only", [
    ("p3", 50, ()), ("thompson_aero", 28, ()),
    # The frozen column state gives TEMPO mp=28's nifa2d; the engine's own
    # allocator (microphysics_schemes.TEMPO.surface_fields) declares
    # nwfa2d alone, and that is what the ARW state TEMPO runs on carries.
    ("tempo", 900, ("nifa2d",))])
def test_the_allocator_agrees_with_the_frozen_species_rows(
        scheme, mp_physics, frozen_only):
    """The hand-kept rows of the frozen seam, recovered from the allocator."""
    from gpuwm.core import mpas_column_batch as frozen
    plan = suite.column_physics_plan(
        dict(suite.MPAS_DEFAULT_PHYSICS, mp_physics=mp_physics))
    assert set(plan.species) == set(frozen._SPECIES_BY_SCHEME[scheme])
    assert set(frozen._STATE_RESTART_FIELDS_BY_SCHEME[scheme]) - set(
        frozen_only) <= set(plan.state_fields)


def test_a_named_scheme_is_selected_by_its_name():
    plan = suite.column_physics_plan(
        dict(suite.MPAS_DEFAULT_PHYSICS, mp_physics="tempo"))
    assert plan.physics["mp_physics"] == 900 and "volg" in plan.species


def test_seam_settings_apply_only_to_their_selection():
    aero = suite.column_physics_plan(
        dict(suite.MPAS_DEFAULT_PHYSICS, mp_physics=28))
    assert aero.config.mp28_aerosol_source == "synthetic"
    assert suite.column_physics_plan(
        suite.MPAS_DEFAULT_PHYSICS).config.mp28_aerosol_source == "auto"
    stated = suite.column_physics_plan(dict(
        suite.MPAS_DEFAULT_PHYSICS, mp_physics=28,
        mp28_aerosol_source="synthetic"))
    assert stated.physics["mp28_aerosol_source"] == "synthetic"


def test_the_soil_geometry_is_the_land_surfaces_own():
    ruc = suite.column_physics_plan(dict(
        suite.MPAS_DEFAULT_PHYSICS, sf_surface_physics=3,
        sf_sfclay_physics=5, bl_pbl_physics=5))
    assert ruc.n_soil_levels == 9
    off = suite.column_physics_plan(dict(
        suite.MPAS_DEFAULT_PHYSICS, sf_surface_physics=0))
    assert off.n_soil_levels == 0


def test_the_drag_statics_are_the_engines_field_table():
    from gpuwm.static.orographic import GWD_FIELDS
    for gwd_opt, names in GWD_FIELDS.items():
        plan = suite.column_physics_plan(
            dict(suite.MPAS_DEFAULT_PHYSICS, gwd_opt=gwd_opt))
        assert plan.static_fields == tuple(names)


def test_seam_owned_and_unknown_fields_refuse_by_name():
    with pytest.raises(ValueError, match="radt_minutes"):
        suite.column_physics_plan(
            dict(suite.MPAS_DEFAULT_PHYSICS, radt_minutes=10.0))
    with pytest.raises(ValueError, match="not RunConfig fields"):
        suite.column_physics_plan(
            dict(suite.MPAS_DEFAULT_PHYSICS, pbl_scheme="ysu"))


def test_the_engine_validator_judges_the_pairing():
    # WRF's own rule, in the engine's own words (no seam pairing table).
    with pytest.raises(ValueError, match="Eta similarity"):
        suite.column_physics_plan(
            dict(suite.MPAS_DEFAULT_PHYSICS, bl_pbl_physics=2))


def test_cadences_are_exact_step_multiples():
    physics = dict(suite.MPAS_DEFAULT_PHYSICS, cu_physics=1)
    plan = suite.column_physics_plan(
        physics, n_levels=NZ, dt=60.0, radiation_seconds=600.0,
        surface_pbl_seconds=120.0, cumulus_seconds=300.0)
    assert (plan.cadences["stepra"], plan.cadences["stepbl"],
            plan.cadences["stepcu"]) == (10, 2, 5)
    assert plan.cadences["cudt_minutes"] == 5.0
    step = suite.column_physics_plan(
        physics, n_levels=NZ, dt=60.0, radiation_seconds=600.0,
        surface_pbl_seconds=60.0)
    assert (step.cadences["stepcu"], step.cadences["cudt_minutes"]) == (1, 0.0)
    with pytest.raises(ValueError, match="not a positive integer multiple"):
        suite.column_physics_plan(
            physics, n_levels=NZ, dt=60.0, radiation_seconds=100.0,
            surface_pbl_seconds=60.0)
    with pytest.raises(ValueError, match="cu_physics is 0"):
        suite.column_physics_plan(
            suite.MPAS_DEFAULT_PHYSICS, n_levels=NZ, dt=60.0,
            radiation_seconds=600.0, surface_pbl_seconds=60.0,
            cumulus_seconds=60.0)


def test_a_hold_on_a_scheme_without_one_is_the_engines_refusal():
    with pytest.raises(ValueError, match="cudt_minutes=0"):
        suite.column_physics_plan(
            dict(suite.MPAS_DEFAULT_PHYSICS, cu_physics=3, ishallow=1),
            n_levels=NZ, dt=60.0, radiation_seconds=600.0,
            surface_pbl_seconds=60.0, cumulus_seconds=300.0)


def test_constructor_refusals_fire_before_any_device_work():
    default = dict(suite.MPAS_DEFAULT_PHYSICS)
    plan = suite.column_physics_plan(default)
    surface = cases.surface(plan, NCOL)
    _refuses("soil_temperature", surface={
        k: v for k, v in surface.items() if k != "soil_temperature"})
    _refuses("shape \\(4, 8\\)", surface=dict(
        surface, soil_moisture=np.full((9, NCOL), 0.3, np.float32)))
    _refuses("does not take", surface=dict(surface, soil_temp=1.0))
    _refuses("extra", statics={"VAR": np.zeros(NCOL, np.float32)})
    drag = dict(default, gwd_opt=3)
    _refuses("missing", physics=drag, statics={})


def test_the_slot_table_entry_is_untouched():
    resolved = suite.resolve_suite_selection(
        radiation_scheme="rrtmg_legacy", surface_layer_scheme="mynn",
        land_surface_scheme="ruc", pbl_scheme="mynn", drag_scheme="gsl")
    assert resolved["config"]["num_soil_layers"] == 9
