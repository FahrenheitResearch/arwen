"""swint_opt and aer_opt on the RunConfig: the implemented values, the
refusals by name, and the route namelist that carries them.

CPU only.  The numerics are graded in tests/test_swint_interpolation.py and
tests/test_rrtmg_aerosol_optics.py; this file holds the doors.
"""
from __future__ import annotations

import dataclasses

import pytest

from gpuwm.config import RunConfig, validate_radiation_driver_options


def _cfg(**overrides):
    values = dict(nx=6, ny=4, nz=5, dx=3000.0, dy=3000.0, ztop=20000.0,
                  dt=20.0, run_seconds=0.0)
    values.update(overrides)
    return RunConfig(**values)


_HRRR = dict(mp_physics=28, ra_physics=4, ra_rrtmg_variant="rrtmg_legacy")


def test_defaults_are_wrf_defaults_and_pass():
    cfg = _cfg()
    assert cfg.swint_opt == 0 and cfg.aer_opt == 0
    validate_radiation_driver_options(cfg)


def test_the_operational_hrrr_values_pass_on_legacy_rrtmg_with_mp28():
    validate_radiation_driver_options(_cfg(swint_opt=1, aer_opt=3, **_HRRR))


@pytest.mark.parametrize(("overrides", "match"), [
    (dict(swint_opt=2), "swint_opt=2"),
    (dict(swint_opt=True), "swint_opt=True"),
    (dict(aer_opt=4), "aer_opt=4"),
    (dict(aer_opt=1, nested=True, **_HRRR), "domain-1 aerod parent forcing"),

    # aer_opt = 3 without the Thompson aerosol numbers
    (dict(aer_opt=3, mp_physics=8, ra_physics=4,
          ra_rrtmg_variant="rrtmg_legacy"), "mp_physics=28"),
    # aer_opt = 3 on the RTE+RRTMGP mapping, which takes no aerosol optics
    (dict(aer_opt=3, mp_physics=28, ra_physics=4,
          ra_rrtmg_variant="rte-rrtmgp"), "legacy RRTMG"),
    # swint_opt = 1 without a shortwave, and on a shortwave that hands the
    # fit no direct beam
    (dict(swint_opt=1, ra_lw_physics=4, ra_sw_physics=0,
          ra_rrtmg_variant="rrtmg_legacy"), "needs a shortwave"),
    (dict(swint_opt=1, ra_physics=1), "Ruiz-Arias"),
])
def test_refusals_name_the_breakage(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_radiation_driver_options(_cfg(**overrides))


def test_the_route_namelist_carries_the_moved_values_and_omits_defaults():
    """The HRRR route runs from the namelist it writes, so swint_opt = 1 and
    aer_opt = 3 must be in it; at 0 the emission keeps its bytes."""
    from types import SimpleNamespace

    from gpuwm import hrrr_route_inputs as route

    def physics_tail(runs):
        lines = []
        for key in ("swint_opt", "aer_opt", "alb_sol"):
            values = {int(getattr(r, key, 0) or 0) for r in runs}
            (value,) = values
            if value:
                lines.append(f" {key:<36s}= {value},")
        return lines

    on = [SimpleNamespace(swint_opt=1, aer_opt=3)] * 2
    off = [SimpleNamespace(swint_opt=0, aer_opt=0)] * 2
    assert physics_tail(on) == [
        " swint_opt                           = 1,",
        " aer_opt                             = 3,"]
    assert physics_tail(off) == []
    # the route module carries exactly that block
    import inspect
    source = inspect.getsource(route.render_namelist_input)
    assert 'for key in ("swint_opt", "aer_opt", "alb_sol"):' in source
    assert 'lines.append(f" {key:<36s}= {value},")' in source


def test_the_checkpoint_echo_drops_both_at_their_default():
    from gpuwm.io import restart
    echo = restart.configuration_echo(_cfg())
    assert "swint_opt" not in echo and "aer_opt" not in echo
    echo = restart.configuration_echo(_cfg(swint_opt=1, aer_opt=3, **_HRRR))
    assert echo["swint_opt"] == 1 and echo["aer_opt"] == 3
    from gpuwm.config import CHEM_RUN_FIELDS, FIRE_RUN_FIELDS
    names = [field.name for field in dataclasses.fields(RunConfig)]
    # The keyword-only fire block follows the chem block.
    names = names[:-len(FIRE_RUN_FIELDS)]
    assert names[-len(CHEM_RUN_FIELDS):] == list(CHEM_RUN_FIELDS)
    names = names[:-len(CHEM_RUN_FIELDS)]
    start = names.index("swint_opt")
    assert names[start - 1:] == [
        "bl_mynn_cloud_tendency_form", "swint_opt", "aer_opt", "alb_sol",
        "thompson_version", "thompson_fork_snow_fall",
        "rrtmg_cloud_optics_form", "rrtmg_smoke_manifest",
        "surface_energy_diag", "cycling", "rrtmg_aer3_type_source",
        "bl_mynn_mixaerosols", "bl_mynn_mixnumcon", "tempo_ml_for_bl_nc",
        "tempo_surface_emission_before", "thompson_table_dir",
        # 2.9.0 (golden/wrf-fixes): scalar advection selector, urban
        # canopy controls and the four WRF 4.7/4.8 fix selectors.
        "scalar_adv_opt", "urban_ucm_generation", "urban_treeoption",
        "urban_ttscheme", "urban_fvg", "urban_table_dir",
        "solar_generation", "ysu_version", "noahmp_snow_age",
        "urban_water_balance",
        # golden/ports-rad: aer_opt = 2 controls and the lightning package.
        "aer_type", "aer_aod550_opt", "aer_angexp_opt",
        "aer_ssa_opt", "aer_asy_opt", "aer_aod550_val", "aer_angexp_val",
        "aer_ssa_val", "aer_asy_val", "aer_auxinput15", "aer_auxinput15_interval",
        "lightning_option", "lightning_dt", "lightning_start_seconds",
        "flashrate_factor", "iccg_method", "iccg_prescribed_num",
        "iccg_prescribed_den", "cellcount_method", "cldtop_adjustment",
        "do_radar_ref",
        # 2.9.0 correct mode (gpuwm/correct_mode.py): the WB5 and WB4 keys.
        "wsm6_conservative_surface", "ntiedtke_dry_mass_coupling",
        "wdm6_conservative_surface", "thompson_cloud_surface_booking",
        # golden/ports-cu: Grell 3D and New SAS controls.
        "cugd_avedx", "maxiens", "maxens", "maxens2", "maxens3",
        "ensdim", "cu_diag", "convtrans_avglen_m",
        "nsas_dx_factor", "sas_pgcon"]
    assert _cfg().cycling is False
    assert "cycling" not in restart.configuration_echo(_cfg())
    for name, default in {"thompson_version": "wrf_461",
            "thompson_fork_snow_fall": "blend", "rrtmg_cloud_optics_form": "wrf_461",
            "rrtmg_smoke_manifest": "", "rrtmg_aer3_type_source": "rural"}.items():
        assert RunConfig.__dataclass_fields__[name].default == default
    # The land/water aerosol type binds only when moved off rural.
    assert "rrtmg_aer3_type_source" not in restart.configuration_echo(
        _cfg(aer_opt=3, **_HRRR))
    echo = restart.configuration_echo(
        _cfg(aer_opt=3, rrtmg_aer3_type_source="landmask", **_HRRR))
    assert echo["rrtmg_aer3_type_source"] == "landmask"


@pytest.mark.parametrize("kind,words", [
    ("relocation", "grid_id = 2 of t.toml moves and sets swint_opt = 1"),
    ("follow", "grid_id = 2 of t.toml moves and sets swint_opt = 1"),
    ("spawn", "grid_id = 2 of t.toml is spawned mid-run and sets swint_opt = 1")])
def test_a_nest_rebuilt_mid_run_refuses_shortwave_interpolation(kind, words):
    """A move or a spawn rebuilds the nest cold without the fit, so its
    daylight shortwave would be zero until the next radiation call; a
    still nest, or swint_opt = 0, is not this refusal's business."""
    from types import SimpleNamespace

    from gpuwm.experiment import _refuse_rebuilt_nest_shortwave_interpolation

    def tree(swint):
        run = SimpleNamespace(swint_opt=swint)
        return [SimpleNamespace(grid_id=1, run=run, follow=None, spawn=None),
                SimpleNamespace(grid_id=2, run=run,
                                follow=object() if kind == "follow" else None,
                                spawn=object() if kind == "spawn" else None)]
    moving = SimpleNamespace(enabled=True, grid_id=2, moves=(), follow=object())
    relocation = moving if kind == "relocation" else None
    with pytest.raises(ValueError, match=words + r".*set swint_opt = 0"):
        _refuse_rebuilt_nest_shortwave_interpolation(tree(1), relocation, "t.toml")
    _refuse_rebuilt_nest_shortwave_interpolation(tree(0), relocation, "t.toml")
    still = [SimpleNamespace(grid_id=1, run=SimpleNamespace(swint_opt=1),
                             follow=None, spawn=None)]
    _refuse_rebuilt_nest_shortwave_interpolation(
        still, SimpleNamespace(enabled=False, grid_id=2, moves=(), follow=None),
        "t.toml")


def test_the_fork_cloud_wrapper_is_explicit_and_legacy_only():
    cfg = _cfg(rrtmg_cloud_optics_form="noaa_wrf39", **_HRRR)
    validate_radiation_driver_options(cfg)
    from gpuwm.io import restart
    assert "rrtmg_cloud_optics_form" not in restart.configuration_echo(_cfg())
    assert restart.configuration_echo(cfg)["rrtmg_cloud_optics_form"] == "noaa_wrf39"
    for overrides in (
            dict(rrtmg_cloud_optics_form="unknown"),
            dict(rrtmg_cloud_optics_form="noaa_wrf39"),
            dict(rrtmg_cloud_optics_form="noaa_wrf39", ra_physics=4,
                 ra_rrtmg_variant="rte-rrtmgp")):
        with pytest.raises(ValueError, match="rrtmg_cloud_optics_form"):
            validate_radiation_driver_options(_cfg(**overrides))


def test_six_type_aerosol_legacy_admission_is_independent_of_microphysics():
    for mp in (1, 6, 8, 10):
        validate_radiation_driver_options(_cfg(aer_opt=1, mp_physics=mp,
            ra_physics=4, ra_rrtmg_variant="rrtmg_legacy"))
