"""Coupled surface stepping, restart and the actual resident-rank road.

Column oracles test WRF arithmetic separately. These runs prevent category
fractions or lake heat storage being lost between physics, restart and ranks.
"""
from dataclasses import replace
import os

import numpy as np
import pytest

from conftest import requires_gpu


def surface_inputs(cfg):
    y, x = np.indices((cfg.ny, cfg.nx))
    lake = ((x >= cfg.nx // 3) & (x < 2 * cfg.nx // 3)
            & (y >= cfg.ny // 3) & (y < 2 * cfg.ny // 3))
    landusef = np.zeros((21, cfg.ny, cfg.nx), np.float32)
    landusef[9] = np.float32(.6)
    landusef[11] = np.float32(.4)
    landusef[:, lake] = 0
    landusef[16, lake] = 1
    soilctop = np.zeros((16, cfg.ny, cfg.nx), np.float32)
    soilctop[2] = np.float32(.25)
    soilctop[5] = np.float32(.75)
    soilctop[:, lake] = 0
    soilctop[13, lake] = 1
    soil_m = np.full((9, cfg.ny, cfg.nx), .28, np.float32)
    soil_m[:, lake] = 1
    return dict(
        landmask=(~lake).astype(np.float32), lakemask=lake.astype(np.float32),
        ivgtyp=np.where(lake, 17, 10), isltyp=np.where(lake, 14, 6),
        tsk=np.where(lake, 289., 300.), sst=np.full(lake.shape, 289.),
        lake_depth=(15. + .1 * x + .2 * y).astype(np.float32),
        lake_depth_flag=1, landusef=landusef, soilctop=soilctop,
        soil_temperature=np.full((9, cfg.ny, cfg.nx), 286., np.float32),
        soil_moisture=soil_m, liquid_moisture=soil_m, vegfra=80.,
        glw=330., swdown=400.)


def _config(nx=48, ny=40):
    from tilestream import ranks_gate
    return replace(ranks_gate.config(nx=nx, ny=ny, nz=20, rung="mynn"),
                   sf_lake_physics=1, mosaic_lu=1, mosaic_soil=1,
                   ra_physics=0, mp_physics=6, use_adaptive_time_step=False,
                   dt=1., run_seconds=20.)


def _build(cfg):
    from tilestream import harness
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    return harness.make_physics_state(cfg, 4242, geography=geo,
                                      **surface_inputs(cfg))


@requires_gpu
def test_coupled_lake_and_mosaic_restart_next_step(tmp_path):
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.io.restart import write_restart, restore_restart
    from tilestream.physics_inventory import carrier_manifest
    from gpuwm.ingest.lateral_bc import build_state_lateral_boundaries, attach_lateral_boundaries

    cfg = _config()
    state, driver = _build(cfg)
    other, _ = _build(cfg)
    boundaries = build_state_lateral_boundaries(
        [state, other], [0., 3600.], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(state, boundaries)
    attach_lateral_boundaries(other, boundaries)
    before = cp.asnumpy(driver.fields["lake_columns"])
    for _ in range(3):
        step(state, cfg)
    assert driver.lake.column_count > 0
    assert np.any(cp.asnumpy(driver.fields["lake_columns"]) != before)
    for name, value in driver.fields.items():
        assert bool(cp.isfinite(value).all()), name
    checkpoint = tmp_path / "surface.npz"
    write_restart(checkpoint, state, cfg)
    restore_restart(checkpoint, other, cfg)
    step(state, cfg)
    step(other, cfg)
    expected, actual = carrier_manifest(state), carrier_manifest(other)
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(cp.asnumpy(actual[name]), cp.asnumpy(expected[name]),
                                      err_msg=name)


@requires_gpu
def test_explicit_zero_lakeflag_uses_elevation_with_initialized_landuse():
    """LAKEFLAG=0 must not be lost just because a land-use object exists."""
    from datetime import datetime, timezone
    import cupy as cp
    from gpuwm.core.landuse import initialize_landuse
    from tilestream import harness

    cfg = _config()
    inputs = surface_inputs(cfg)
    inputs.pop("lakemask")
    # This source has water category 17, with no separate category-21 lake
    # map. WRF uses water elevation when the explicit LAKEFLAG is zero.
    landuse = initialize_landuse(
        inputs["ivgtyp"], soil_type=inputs["isltyp"],
        landmask=inputs["landmask"], snow=0., xice=0.,
        valid_time=datetime(2026, 7, 1, tzinfo=timezone.utc), cen_lat=35.,
        mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21, isice=15,
        soil_temperature=inputs["soil_temperature"])
    assert not landuse.lakemask.any()
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    state, driver = harness.make_physics_state(
        cfg, 4242, geography=geo, landuse=landuse, lake_mask_flag=0, **inputs)
    expected = ((landuse.ivgtyp == 17)
                & (cp.asnumpy(state.ht) >= cfg.lake_min_elev)).astype(np.float32)
    assert expected.any()
    np.testing.assert_array_equal(cp.asnumpy(driver.fields["lakemask"]), expected)
    assert driver.lake.column_count == np.count_nonzero(expected)


@requires_gpu
def test_lake_uses_noahmp_latitude_without_a_radiation_adapter():
    """The caller's Noah-MP latitude is valid lake geometry with fixed sky."""
    import cupy as cp
    from test_noahmp_runtime import _build, _LATITUDE

    _, _, driver = _build(sf_lake_physics=1, use_lakedepth=0,
                          lake_min_elev=-1.)
    assert driver.lake.column_count == 12
    np.testing.assert_array_equal(cp.asnumpy(driver.lake.latitude),
                                  np.full(12, _LATITUDE, np.float32))


@requires_gpu
def test_ruc_lake_two_resident_ranks_match_unsplit_model(monkeypatch):
    """Actual RankedRun constructor, gather, two devices, coupled RK3 and join."""
    import cupy as cp
    from tilestream import harness, ranks_gate

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two allocated cards")
    cfg = _config(nx=128, ny=104)
    real_build = harness.make_physics_state

    def with_surface_state(run, *args, **kwargs):
        # The full-domain oracle starts from the mixed geography. A neutral
        # rank already receives fraction carriers from the production factory;
        # its mandatory gather supplies all true surface state before stepping.
        if "landusef" not in kwargs:
            kwargs.update(surface_inputs(run))
        return real_build(run, *args, **kwargs)

    monkeypatch.setattr(harness, "make_physics_state", with_surface_state)
    count = int(os.environ.get("GPUWM_SURFACE_RANK_STEPS", "4"))
    result = ranks_gate.integrate(cfg, grid=(1, 2), mode="threads",
                                  nsteps=count, devices=(0, 1), change_live=False)
    assert not result["differing"]
    assert result["resident_digest"] == result["ranked_digest"]


def _lake_ice_case(fractional_seaice, ice):
    """The lake fixture with every lake cell at sea-ice fraction ``ice``."""
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)
    from tilestream import harness

    cfg = replace(_config(), fractional_seaice=fractional_seaice)
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    inputs = surface_inputs(cfg)
    lake = inputs["lakemask"] == 1
    xice = np.where(lake, np.float32(ice), np.float32(0)).astype(np.float32)
    built = [harness.make_physics_state(cfg, 4242, geography=geo, xice=xice,
                                        **inputs) for _ in range(2)]
    (state, driver), (other, _) = built
    boundaries = build_state_lateral_boundaries(
        [state, other], [0., 3600.], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(state, boundaries)
    return cfg, state, driver, lake


@requires_gpu
@pytest.mark.parametrize("fractional_seaice,ice,handed_over", [
    (1, 0.3, True), (1, 0.7, True), (1, 0.02, False),
    (0, 0.3, False), (0, 0.7, True)])
def test_lake_ice_above_the_run_threshold_leaves_the_sea_ice_seam(
        fractional_seaice, ice, handed_over):
    """A frozen CLM-lake cell is lake water to the surface, at the run's
    threshold.

    The HRRR v4.1.21 fork moves a lake cell's ice into the lake model and
    hands the surface open lake water (``xice > xice_threshold``:
    module_sf_lake.F lakeini and module_surface_driver.F:1452-1461, the
    same threshold, 0.02 under fractional_seaice = 1).  The host
    write-back used 0.5 while the lake kernel used the run threshold, so
    under fractional_seaice = 1 a cell at XICE 0.3 kept its sea ice and the
    seam overrode its ALBBCK with the sea-ice albedo and ran MYNN's ice and
    water calls over a lake.
    """
    import cupy as cp
    from gpuwm.core.dycore import step

    cfg, state, driver, lake = _lake_ice_case(fractional_seaice, ice)
    threshold = 0.02 if fractional_seaice else 0.5
    assert driver.lake.xice_threshold == threshold
    assert driver.ruc_params.xice_threshold == threshold
    xice = cp.asnumpy(driver.fields["xice"])[lake]
    xland = cp.asnumpy(driver.fields["xland"])[lake]
    ivgtyp = cp.asnumpy(driver.fields["ivgtyp"])[lake]
    if handed_over:
        assert np.all(xice == 0) and np.all(xland == 2)
        assert np.all(ivgtyp == 17)
    else:
        assert np.all(xice == np.float32(ice))
    for _ in range(3):
        step(state, cfg)
    albbck = cp.asnumpy(driver.fields["albbck"])[lake]
    if handed_over or ice < threshold:
        # Lake water: the seam's sea-ice ALBBCK override never touched it.
        assert not np.any(albbck == np.float32(
            driver.ruc_params.seaice_albedo_default))
    for name, value in driver.fields.items():
        assert bool(cp.isfinite(value).all()), name


@requires_gpu
def test_rdlai2d_refuses_a_road_that_seeded_no_leaf_area():
    """rdlai2d keeps the LAI a road seeds from LAI12M.  A road that seeded
    none used to run RUC on the 2.0 allocation default with no word."""
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)
    from tilestream import harness

    cfg = replace(_config(), rdlai2d=True)
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
    built = [harness.make_physics_state(cfg, 4242, geography=geo,
                                        **surface_inputs(cfg))
             for _ in range(2)]
    (state, driver), (other, other_driver) = built
    boundaries = build_state_lateral_boundaries(
        [state, other], [0., 3600.], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(state, boundaries)
    attach_lateral_boundaries(other, boundaries)
    assert not bool(cp.isfinite(driver.fields["lai"]).any())
    with pytest.raises(ValueError, match="seeded no LAI field"):
        step(state, cfg)
    # Seeded, the same configuration runs and SOILVEGIN leaves the seed.
    seeded = cp.full(other_driver.fields["lai"].shape, 1.25, cp.float32)
    other_driver.fields["lai"][...] = seeded
    step(other, cfg)
    np.testing.assert_array_equal(cp.asnumpy(other_driver.fields["lai"]),
                                  cp.asnumpy(seeded))
    # Off, the allocation is the 2.0 default exactly as before.
    off_state, off_driver = harness.make_physics_state(
        _config(), 4242, geography=geo, **surface_inputs(_config()))
    assert bool((off_driver.fields["lai"] == cp.float32(2.0)).all())
