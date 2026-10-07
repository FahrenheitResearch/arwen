"""Admission and native-input identity for the sounding-driven fire door."""
from dataclasses import replace
from datetime import datetime

import pytest

from gpuwm.config import validate_run_config
from gpuwm.fire_ideal import _make_clock, _native_dt, load_native_configuration


NATIVE = """
&time_control
 run_seconds=4, history_interval_s=1,
/
&domains
 max_dom=1, e_we=13, e_sn=11, e_vert=7,
 dx=50., dy=50., ztop=2000., time_step=0,
 time_step_fract_num=1, time_step_fract_den=2,
 sr_x=2, sr_y=2,
/
&dynamics
 hybrid_opt=3, tracer_opt=3,
/
&bdy_control
 open_xs=.true., open_xe=.true.,
 open_ys=.true., open_ye=.true.,
/
&fire
 ifire=2, fire_fuel_read=0, fire_fuel_cat=3,
 fire_lat_init=35., fire_lon_init=-120.,
 fire_mountain_type=2, fire_mountain_height=80.,
 fire_mountain_start_x=50., fire_mountain_end_x=450.,
 fire_mountain_start_y=50., fire_mountain_end_y=350.,
 delt_perturbation=1.5, xrad_perturbation=100.,
 yrad_perturbation=100., zrad_perturbation=200.,
 hght_perturbation=300., sfc_full_init=.true.,
/
"""


def write(tmp_path, text=NATIVE):
    path = tmp_path / "namelist.input"
    path.write_text(text)
    return path


def test_native_ideal_retains_initialization_options_and_bulk_selector(tmp_path):
    cfg = load_native_configuration(write(tmp_path))
    assert (cfg.nx, cfg.ny, cfg.nz) == (12, 10, 6)
    assert cfg.dt == 0.5 and cfg.run_seconds == 4
    assert cfg.fire_mountain_type == 2 and cfg.delt_perturbation == 1.5
    assert cfg.fire_lat_init == 35 and cfg.fire_lon_init == -120
    assert cfg.fire_smoke and "sfire_smoke" in cfg.chem_sets
    assert cfg.terrain_opt == 1 and cfg.hypsometric_opt == 1
    assert cfg.sfc_full_init and cfg.hybrid_opt == 3
    clock = _make_clock(cfg, datetime(1, 1, 1))
    for _ in range(8):
        clock.advance()
    assert clock.at_stop_time and clock.history_due()
    assert clock.elapsed_seconds == 4


def test_generic_vertical_does_not_accept_unproduced_sine_coefficients(tmp_path):
    cfg = load_native_configuration(write(tmp_path))
    with pytest.raises(ValueError, match="generic vertical initializer"):
        validate_run_config(cfg)


@pytest.mark.parametrize("before,after,message", [
    ("ifire=2", "ifire=0", "ifire=2 and map_proj=0"),
    ("max_dom=1", "max_dom=2", "parent boundary"),
    ("tracer_opt=3", "tracer_opt=2", "bulk-smoke row"),
    ("open_xe=.true.", "open_xe=.false.", "matching native boundary sides"),
    ("fire_fuel_cat=3", "fire_fuel_catt=3", "unconsumed &fire keys"),
])
def test_native_ideal_refuses_dropped_or_unproduced_state(tmp_path, before, after, message):
    with pytest.raises(ValueError, match=message):
        load_native_configuration(write(tmp_path, NATIVE.replace(before, after)))


def test_fixed_clock_rejects_alarms_between_model_steps(tmp_path):
    cfg = load_native_configuration(write(tmp_path))
    with pytest.raises(ValueError, match="reach every requested alarm"):
        _make_clock(replace(cfg, output_interval_s=0.7), datetime(1, 1, 1))


def test_fuel_namelist_is_resolved_from_native_input_directory(tmp_path):
    directory = tmp_path / "inputs"
    directory.mkdir()
    fuel = directory / "namelist.fire"
    fuel.write_text("&fuel_categories\n/\n")
    cfg = load_native_configuration(write(tmp_path), input_directory=directory)
    assert cfg.fire_fuel_namelist == str(fuel.resolve())


def test_native_end_date_supplies_zero_run_duration(tmp_path):
    text = NATIVE.replace("run_seconds=4", "run_seconds=0, start_year=2025, start_month=7, start_day=1, "
                          "end_year=2025, end_month=7, end_day=1, end_second=4")
    assert load_native_configuration(write(tmp_path, text)).run_seconds == 4


def test_native_thirds_keep_the_rational_tick_authority(tmp_path):
    from gpuwm.fortran_namelist import parse_namelist
    path = write(tmp_path, NATIVE.replace("time_step_fract_den=2", "time_step_fract_den=3"))
    cfg = load_native_configuration(path)
    clock = _make_clock(cfg, datetime(1, 1, 1), exact_dt=_native_dt(parse_namelist(path)))
    assert clock.tick_den == 3
    for _ in range(12):
        clock.advance()
    assert clock.at_stop_time and clock.elapsed_seconds == 4
    assert clock.history_due()


def test_resolved_native_defaults_are_consumed_without_dropping_explicit_fire_values(tmp_path):
    path = write(tmp_path)
    (tmp_path / "namelist.output").write_text(
        "&dynamics h_sca_adv_order=5, /\n&fire fire_viscosity=.7, fire_mountain_height=999., /\n")
    cfg = load_native_configuration(path)
    assert cfg.h_sca_adv_order == 5 and cfg.fire_viscosity == 0.7
    assert cfg.fire_mountain_height == 80


def test_native_eta_sentinel_uses_procedural_vertical_grid(tmp_path):
    path = write(tmp_path, NATIVE.replace("sr_x=2", "eta_levels=-1., sr_x=2"))
    assert load_native_configuration(path).eta_levels is None
