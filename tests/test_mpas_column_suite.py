"""CPU contract of the selectable MPAS column suite (no device).

Every refusal below fires before a device allocation, so these run on any
machine.  They hold the selector table to its stated rules: one row per
scheme, WRF's PBL / surface-layer pairing, the frozen four refused here
(they are the frozen seam's), the land surface's own soil geometry and
sea-ice knob, and the drag's exact statistics set.
"""

import datetime

import numpy as np
import pytest

from gpuwm.core import mpas_column_suite as suite

NZ, NCOL = 20, 8
START = datetime.datetime(2021, 6, 1, 18, 0)
HRRR_FAMILY = dict(radiation_scheme="rrtmg_legacy",
                   surface_layer_scheme="mynn", land_surface_scheme="ruc",
                   pbl_scheme="mynn", drag_scheme="gsl")


def _kwargs(**overrides):
    names = suite.orographic_statistic_names("gsl")
    kwargs = dict(
        n_levels=NZ, n_columns=NCOL, dt=60.0,
        radiation_seconds=1800.0, surface_pbl_seconds=60.0,
        start_time=START, microphysics_scheme="thompson_aero",
        latitude_deg=np.full(NCOL, 35.0), longitude_deg=np.full(NCOL, -97.0),
        terrain_height_m=np.zeros(NCOL),
        z_interface_nominal_m=np.linspace(0.0, 16000.0, NZ + 1),
        p_top_pa=13000.0, dx_m=12000.0,
        soil_temperature=np.full((9, NCOL), 290.0, np.float32),
        soil_moisture=np.full((9, NCOL), 0.3, np.float32),
        fractional_seaice=1,
        orographic_statistics={name: np.zeros(NCOL, np.float32)
                               for name in names},
        **HRRR_FAMILY)
    kwargs.update(overrides)
    return kwargs


def _refuses(match, **overrides):
    with pytest.raises(ValueError, match=match):
        suite.MpasColumnSuitePhysics(**_kwargs(**overrides))


def test_every_selector_value_maps_to_one_config_row():
    resolved = suite.resolve_suite_selection(**HRRR_FAMILY)
    assert resolved["config"] == {
        "ra_physics": 4, "ra_rrtmg_variant": "legacy",
        "sf_sfclay_physics": 5, "sf_surface_physics": 3,
        "num_soil_layers": 9, "bl_pbl_physics": 5, "gwd_opt": 3}


def test_the_frozen_four_are_refused_here():
    with pytest.raises(ValueError, match="frozen seam"):
        suite.resolve_suite_selection(**suite.FROZEN_SELECTION)


@pytest.mark.parametrize('key', ['drag_scheme'])
def test_an_unknown_scheme_is_refused_by_name(key):
    with pytest.raises(ValueError, match=key):
        suite.resolve_suite_selection(**dict(HRRR_FAMILY, **{key: "nope"}))


def test_wrf_pairing_rules():
    with pytest.raises(ValueError, match="revised_mm5"):
        suite.resolve_suite_selection(**dict(HRRR_FAMILY, pbl_scheme="ysu"))
    # MYNN PBL over the revised MM5 layer is a legal WRF pairing.
    suite.resolve_suite_selection(
        **dict(HRRR_FAMILY, surface_layer_scheme="revised_mm5"))


def test_soil_column_follows_the_land_surface():
    _refuses("shape \\(9, 8\\)",
             soil_temperature=np.full((4, NCOL), 290.0, np.float32))
    _refuses("required", soil_moisture=None)


def test_sea_ice_knobs_belong_to_their_own_land_surface():
    _refuses("Noah-MP's classification threshold", xice_threshold=0.5)
    _refuses("fractional_seaice 0 or 1", fractional_seaice=None)


def test_drag_statistics_are_exact_both_ways():
    names = suite.orographic_statistic_names("gsl")
    assert len(names) == 20
    _refuses("turns it off", orographic_statistics=None)
    partial = {name: np.zeros(NCOL, np.float32) for name in names[:-1]}
    _refuses("missing", orographic_statistics=partial)
    extra = {name: np.zeros(NCOL, np.float32) for name in names}
    extra["VAR"] = np.zeros(NCOL, np.float32)
    _refuses("extra", orographic_statistics=extra)
    _refuses("drag would run on zeros|exactly", drag_scheme="off")


def test_kpblmax_coordinate_is_monotone_and_bounded():
    znu = suite._nominal_znu(np.linspace(0.0, 25000.0, 61), 2000.0)
    assert znu.dtype == np.float32
    assert np.all(np.diff(znu) <= 0) and znu[0] <= 1.0 and znu[-1] >= 0.0
    assert np.all(np.diff(znu)[znu[1:] > 0] < 0)
    from gpuwm.core.terrain_drag import gsl_kpblmax
    # znu > 0.6 ends near 4 km on the standard atmosphere.
    kpblmax = gsl_kpblmax(znu)
    z_mid = 0.5 * (np.linspace(0.0, 25000.0, 61)[:-1]
                   + np.linspace(0.0, 25000.0, 61)[1:])
    assert 3000.0 < z_mid[kpblmax - 2] < 5000.0


# --- generation selectors -------------------------------------------------
def test_no_generation_is_the_runconfig_defaults():
    assert suite.resolve_generation(
        None, microphysics_scheme="thompson_aero", selection=HRRR_FAMILY) == {}
    assert suite.resolve_generation(
        {}, microphysics_scheme="thompson_aero", selection=HRRR_FAMILY) == {}


def test_every_generation_row_names_runconfig_fields_and_their_values():
    import dataclasses
    from gpuwm.config import RunConfig
    names = {field.name: field.default for field in dataclasses.fields(RunConfig)}
    for key, row in suite.GENERATION_SELECTORS.items():
        assert key in names, key
        assert names[key] in row["values"], (key, names[key])


def test_a_generation_for_a_scheme_the_seam_runs_is_admitted():
    chosen = {"thompson_version": "wrf_39_noaa", "bl_mynn_version": "gsd_41",
              "bl_mynn_mixlength": 2, "mynn_sfclay_variant": "gsl_wrf39",
              "rrtmg_cloud_optics_form": "noaa_wrf39"}
    assert suite.resolve_generation(
        chosen, microphysics_scheme="thompson_aero",
        selection=HRRR_FAMILY) == chosen


@pytest.mark.parametrize('key, value, selection, micro, match', [
    pytest.param('bl_mynn_version', 'gsd_41', dict(HRRR_FAMILY, pbl_scheme='ysu', surface_layer_scheme='revised_mm5'), 'thompson_aero', 'never read', id='bl_mynn_version-gsd_41-selection0-thompson_aero-never read'),
    pytest.param('bl_mynn_version', 'gsd_99', HRRR_FAMILY, 'thompson_aero', 'one of', id='bl_mynn_version-gsd_99-selection3-thompson_aero-one of'),
    pytest.param('ysu_topdown_pblmix', 1, HRRR_FAMILY, 'thompson_aero', 'has no row', id='ysu_topdown_pblmix-1-selection4-thompson_aero-has no row'),
])
def test_a_generation_the_seam_would_not_read_is_refused(
        key, value, selection, micro, match):
    with pytest.raises(ValueError, match=match):
        suite.resolve_generation({key: value}, microphysics_scheme=micro,
                                 selection=selection)


def test_the_constructor_refuses_a_bad_generation_before_any_device_work():
    _refuses("never read", microphysics_scheme="wsm6",
             generation={"thompson_version": "wrf_39_noaa"})
