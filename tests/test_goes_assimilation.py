"""GOES CWP through the product, the adapter, the filter and the CLIs.

The test that carries the most weight here is
``test_an_obs_clear_column_removes_model_cloud``: the suppression
direction is the half of satellite cloud assimilation that is easy to
build wrong and impossible to notice, because a system that can only add
cloud still produces increments, still writes receipts, and still looks
like it is working.

``test_builder_cli_writes_a_product_from_two_packs`` runs the real
``tools/obs_goes_grid_build.py`` in a subprocess against a real wrfout, a
real CWP pack and a real cloud-top pack, and reads back what it wrote.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gpuwm.da.letkf import Localization
from gpuwm.da.obs_goes import (GoesObsAdapterError, expand_to_volume,
                               goes_grid_to_gridded_obs, read_document)
from gpuwm.da.obsop_cwp import checkpoint_cwp_provider
from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                         RadarAssimilationError,
                                         assimilate_radar_grid)
from gpuwm.obs.goes_cwp import (CLASS_CLEAR, CLASS_ICE, CwpErrorModel,
                                grid_cwp, join_cloud_top, read_cloudtop_pack,
                                read_cwp_pack)
from gpuwm.obs.goes_grid import (GOES_GRID_SCHEMA, GOES_GRID_STATUS,
                                 GoesGridSchemaError, read_goes_grid,
                                 write_goes_grid)
from gpuwm.obs.target_grid import GridMismatchError, TargetGrid
from gpuwm.static.lambert import LambertGrid

from goes_pack_fixtures import (sibling_block, write_cloudtop_pack,
                                write_cwp_pack)

REPO = Path(__file__).resolve().parent.parent

NX = NY = 9
NZ = 4
DX = 6000.0
TOP_M = 10000.0
MU = 100000.0

ERRORS = CwpErrorModel(clear_g_m2=20.0, rel_liquid=0.3,
                       floor_liquid_g_m2=40.0, rel_ice=0.5,
                       floor_ice_g_m2=80.0)

SETUP = {"c1h": np.ones(NZ), "c2h": np.zeros(NZ),
         "dnw": np.full(NZ, -1.0 / NZ), "mub2d": np.full((NY, NX), MU)}


def _projection(nx=NX, ny=NY, dx=DX):
    return LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)


def _grid(**kwargs) -> TargetGrid:
    return TargetGrid.from_projection(
        _projection(**kwargs), z_w=np.linspace(0.0, TOP_M, NZ + 1),
        name="analytic")


def _product(tmp_path, grid, cells, *, phase, cod, cps, tops=None):
    """One goes-grid document, built the way the builder builds it."""

    lat = np.array([[grid.lat[j, i] for (j, i) in cells]], np.float32)
    lon = np.array([[grid.lon[j, i] for (j, i) in cells]], np.float32)
    scan_x = np.linspace(-0.02, 0.02, len(cells))
    cwp_path = write_cwp_pack(
        tmp_path / "cwp.goespack",
        cod=np.array([cod], np.float32), cps=np.array([cps], np.float32),
        phase=np.array([phase], np.float32), lat=lat, lon=lon,
        x_scan_rad=scan_x, y_scan_rad=[0.08])
    pack = read_cwp_pack(cwp_path)
    joined = join_receipt = None
    if tops is not None:
        ct_path = write_cloudtop_pack(
            tmp_path / "ct.goespack",
            cloud_top_height_m=np.array([tops], np.float32),
            lat=lat, lon=lon, x_scan_rad=scan_x, y_scan_rad=[0.08],
            sibling=sibling_block(cwp_path))
        joined, join_receipt = join_cloud_top(pack, read_cloudtop_pack(ct_path))
    observations = grid_cwp(pack, grid, error_model=ERRORS,
                            cloud_top_m=joined, join_receipt=join_receipt)
    out = tmp_path / "goes.nc"
    write_goes_grid(out, observations, grid, valid_time="2026-08-04T18:01:17Z",
                    overwrite=True)
    return out, observations


def _checkpoints(tmp_path, grid, members, *, qc, qi=1.0e-8, qs=1.0e-8,
                 seed=7):
    """Member npz checkpoints with condensate that has ensemble spread.

    The scale is lognormal so it is strictly positive: a member clipped to
    exactly zero would leave a species constant across the ensemble, and
    the filter refuses a prior field with no spread rather than papering
    over it with a zero increment.
    """

    rng = np.random.default_rng(seed)
    paths = {}
    for index in range(members):
        scale = float(np.exp(0.3 * rng.standard_normal()))
        state = {
            "qc": np.full((NZ, NY, NX), qc * scale),
            "qi": np.full((NZ, NY, NX), qi * scale),
            "qs": np.full((NZ, NY, NX), qs * scale),
            "mup": np.zeros((NY, NX)),
        }
        path = tmp_path / f"member_{index:03d}.npz"
        np.savez(path, **{f"state/{k}": v for k, v in state.items()})
        paths[index] = path
    return paths


def _config(**kwargs):
    base = dict(
        localization=Localization(horizontal_m=24000.0, vertical_m=20000.0),
        rtps_alpha=0.9, analysis_fields=("qc", "qi", "qs"),
        velocity=False, reflectivity=False, cwp=True,
        cwp_localization=Localization(horizontal_m=24000.0,
                                      vertical_m=20000.0),
        positivity_policy="clip", solve_device="host")
    base.update(kwargs)
    return RadarAssimilationConfig(**base)


# ---------------------------------------------------------------------------
# the on-disk product
# ---------------------------------------------------------------------------


def test_the_product_round_trips_and_carries_its_three_receipts(tmp_path):
    grid = _grid()
    path, observations = _product(
        tmp_path, grid, [(4, 4), (4, 5)], phase=[4.0, 0.0], cod=[20.0, 0.0],
        cps=[30.0, 0.0], tops=[9000.0, np.nan])
    document = read_goes_grid(path, expected_grid=grid)
    assert document["schema"] == GOES_GRID_SCHEMA
    assert document["status"] == GOES_GRID_STATUS
    assert document["dims"] == {"south_north": NY, "west_east": NX, "nz": NZ}
    variables = document["variables"]
    assert np.array_equal(variables["cwp_mask"], observations.cwp_mask)
    assert np.allclose(variables["cwp_obs"], observations.cwp_obs, rtol=1e-6)
    assert np.array_equal(variables["obs_level"], observations.obs_level)
    # The three things that must never travel silently.
    assert document["join"]["method"] == "nearest"
    assert document["error_model"]["calibration"] == "UNCALIBRATED"
    assert {row["product"] for row in document["dqf_policy"]} == {
        "COD", "CPS", "ACTP"}


def test_the_product_is_welded_to_one_grid(tmp_path):
    grid = _grid()
    path, _ = _product(tmp_path, grid, [(4, 4)], phase=[4.0], cod=[20.0],
                       cps=[30.0])
    other = TargetGrid.from_projection(
        _projection(), z_w=np.linspace(0.0, 12000.0, NZ + 1), name="other")
    # Same horizontal arrays, a different vertical stretching -- and z_w is
    # the array the file does not store, so only the caller's own grid can
    # catch it.  obs_level indexes exactly that structure.
    with pytest.raises(GridMismatchError):
        read_goes_grid(path, expected_grid=other)


def test_the_product_refuses_a_clear_cell_that_is_not_zero(tmp_path):
    grid = _grid()
    _, observations = _product(tmp_path, grid, [(4, 4)], phase=[0.0],
                               cod=[0.0], cps=[0.0])
    tampered = type(observations)(
        **{**observations.__dict__,
           "cwp_obs": np.where(observations.cwp_mask.astype(bool), 5.0,
                               observations.cwp_obs)})
    write_goes_grid(tmp_path / "bad.nc", tampered, grid,
                    valid_time="2026-08-04T18:01:17Z")
    with pytest.raises(GoesGridSchemaError, match="clear-sky cell"):
        read_goes_grid(tmp_path / "bad.nc", expected_grid=grid)


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------


def test_a_column_integral_is_one_observation_not_nz_of_them(tmp_path):
    """The invariant the whole 2-D-into-3-D expansion exists to protect."""

    grid = _grid()
    path, _ = _product(tmp_path, grid, [(4, 4), (4, 5)], phase=[4.0, 0.0],
                       cod=[20.0, 0.0], cps=[30.0, 0.0], tops=[9000.0, np.nan])
    document = read_goes_grid(path, expected_grid=grid)
    values, errors, mask, simulated, placed = expand_to_volume(
        document, (NZ, NY, NX), np.zeros((3, NY, NX)))
    assert mask.shape == (NZ, NY, NX)
    assert int(mask.sum()) == 2 == int(placed.sum())
    # ...and exactly one level per observed column.
    assert np.all(mask.sum(axis=0)[placed] == 1)
    assert np.all(errors[mask] > 0.0)
    # z_w is 0..10000 in four 2500 m layers: 9000 m is level 3, and the
    # 3000 m fallback for the clear column is level 1.
    assert mask[3, 4, 4] and mask[1, 4, 5]


def test_the_adapter_refuses_a_three_dimensional_hx(tmp_path):
    grid = _grid()
    path, _ = _product(tmp_path, grid, [(4, 4)], phase=[4.0], cod=[20.0],
                       cps=[30.0])
    with pytest.raises(GoesObsAdapterError, match="H\\(x\\) for a column"):
        goes_grid_to_gridded_obs(path, expected_grid=grid,
                                 cwp_simulated=np.zeros((3, NZ, NY, NX)))


def test_the_adapter_requires_the_callers_own_grid(tmp_path):
    grid = _grid()
    path, _ = _product(tmp_path, grid, [(4, 4)], phase=[4.0], cod=[20.0],
                       cps=[30.0])
    with pytest.raises(GoesObsAdapterError, match="expected_grid is required"):
        read_document(path, expected_grid=None)


def test_the_adapter_records_how_far_the_lens_reaches(tmp_path):
    grid = _grid()
    path, _ = _product(tmp_path, grid, [(4, 4)], phase=[4.0], cod=[20.0],
                       cps=[30.0])
    _, provenance = goes_grid_to_gridded_obs(
        path, expected_grid=grid, cwp_simulated=np.zeros((3, NY, NX)),
        localization=Localization(horizontal_m=24000.0, vertical_m=2500.0))
    reach = provenance["vertical_reach"]
    assert reach["median_column_depth_m"] == pytest.approx(TOP_M)
    assert reach["fraction_of_median_column_reached"] == pytest.approx(0.25)
    assert provenance["batches"][0]["observed_points"] == 1
    assert provenance["error_model"]["calibration"] == "UNCALIBRATED"


def test_error_inflation_below_one_is_refused(tmp_path):
    grid = _grid()
    path, _ = _product(tmp_path, grid, [(4, 4)], phase=[4.0], cod=[20.0],
                       cps=[30.0])
    with pytest.raises(GoesObsAdapterError, match="claim of skill"):
        goes_grid_to_gridded_obs(path, expected_grid=grid,
                                 cwp_simulated=np.zeros((3, NY, NX)),
                                 error_inflation=0.5)


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------


def _analyse(tmp_path, grid, product, checkpoints, **cfg_kwargs):
    provider = checkpoint_cwp_provider(None, **SETUP)
    return assimilate_radar_grid(
        checkpoints, None, grid, _config(**cfg_kwargs),
        cwp_observations=product, cwp_provider=provider)


def test_an_obs_clear_column_removes_model_cloud(tmp_path):
    """Suppression: the direction that is easy to build wrong.

    Every member is cloudy; the satellite says clear.  The analysis must
    take condensate OUT.  A system that can only add cloud would pass
    every other test in this file.
    """

    grid = _grid()
    # Clear-sky zeros over a block of columns.
    cells = [(j, i) for j in range(3, 6) for i in range(3, 6)]
    product, observations = _product(
        tmp_path, grid, cells, phase=[0.0] * len(cells),
        cod=[0.0] * len(cells), cps=[0.0] * len(cells))
    assert np.all(observations.cwp_obs[observations.cwp_mask.astype(bool)]
                  == 0.0)
    assert np.all(observations.cwp_class[3:6, 3:6] == CLASS_CLEAR)

    checkpoints = _checkpoints(tmp_path, grid, 12, qc=2.0e-4, qi=1.0e-4)
    increments, provenance = _analyse(tmp_path, grid, product, checkpoints)

    innovation = provenance["innovations"][0]
    assert innovation["name"] == "cwp"
    assert innovation["obs_mean"] == 0.0
    assert innovation["hx_mean"] > 0.0
    # obs - H(x): the model has cloud the satellite did not see.
    assert innovation["innovation_mean"] < 0.0

    # Every member loses condensate in the observed block.
    for index in sorted(increments):
        for field in ("qc", "qi"):
            block = increments[index][field][:, 3:6, 3:6]
            assert block.max() <= 0.0, f"member {index} {field} added cloud"
            assert block.min() < 0.0, f"member {index} {field} did nothing"

    assert provenance["cwp_assimilated"] is True
    assert provenance["cwp_observations"]["batches"][0]["clear_sky_zeros"] == 9


def test_an_obs_cloudy_column_adds_cloud_to_a_dry_model(tmp_path):
    """The other direction, so the previous test is not a sign error."""

    grid = _grid()
    cells = [(j, i) for j in range(3, 6) for i in range(3, 6)]
    n = len(cells)
    product, observations = _product(
        tmp_path, grid, cells, phase=[4.0] * n, cod=[20.0] * n,
        cps=[30.0] * n)
    assert np.all(observations.cwp_class[3:6, 3:6] == CLASS_ICE)

    checkpoints = _checkpoints(tmp_path, grid, 12, qc=1.0e-7, qi=1.0e-7,
                               qs=1.0e-7, seed=11)
    increments, provenance = _analyse(tmp_path, grid, product, checkpoints)
    assert provenance["innovations"][0]["innovation_mean"] > 0.0
    added = np.stack([increments[i]["qi"][:, 3:6, 3:6]
                      for i in sorted(increments)])
    assert added.max() > 0.0


def test_the_analysis_refuses_cwp_without_its_operator(tmp_path):
    grid = _grid()
    product, _ = _product(tmp_path, grid, [(4, 4)], phase=[4.0], cod=[20.0],
                          cps=[30.0])
    checkpoints = _checkpoints(tmp_path, grid, 4, qc=1.0e-4)
    with pytest.raises(RadarAssimilationError, match="no cwp_provider"):
        assimilate_radar_grid(checkpoints, None, grid, _config(),
                              cwp_observations=product)
    with pytest.raises(RadarAssimilationError, match="no cwp_observations"):
        assimilate_radar_grid(
            checkpoints, None, grid, _config(),
            cwp_provider=checkpoint_cwp_provider(None, **SETUP))


def test_the_analysis_refuses_cwp_against_a_wind_only_state_vector():
    with pytest.raises(RadarAssimilationError, match="cwp is enabled"):
        _config(analysis_fields=("u", "v"), positivity_policy=None)


def test_the_analysis_refuses_a_product_from_another_grid(tmp_path):
    grid = _grid()
    product, _ = _product(tmp_path, grid, [(4, 4)], phase=[4.0], cod=[20.0],
                          cps=[30.0])
    other = TargetGrid.from_projection(
        _projection(), z_w=np.linspace(0.0, 12000.0, NZ + 1), name="other")
    checkpoints = _checkpoints(tmp_path, grid, 4, qc=1.0e-4)
    with pytest.raises(GridMismatchError):
        assimilate_radar_grid(
            checkpoints, None, other, _config(),
            cwp_observations=product,
            cwp_provider=checkpoint_cwp_provider(None, **SETUP))


def test_a_missing_product_is_a_refusal_not_an_empty_cycle(tmp_path):
    grid = _grid()
    checkpoints = _checkpoints(tmp_path, grid, 4, qc=1.0e-4)
    with pytest.raises(OSError):
        assimilate_radar_grid(
            checkpoints, None, grid, _config(),
            cwp_observations=tmp_path / "absent.nc",
            cwp_provider=checkpoint_cwp_provider(None, **SETUP))


def test_thinning_keeps_one_observation_per_block(tmp_path):
    grid = _grid()
    cells = [(j, i) for j in range(2, 8) for i in range(2, 8)]
    n = len(cells)
    product, _ = _product(tmp_path, grid, cells, phase=[4.0] * n,
                          cod=[20.0] * n, cps=[30.0] * n)
    checkpoints = _checkpoints(tmp_path, grid, 8, qc=1.0e-4, seed=3)
    _, provenance = _analyse(tmp_path, grid, product, checkpoints,
                             cwp_thinning_cells=3)
    thinning = provenance["cwp_thinning"]
    assert thinning["points_before"] == 36
    assert thinning["points_after"] < thinning["points_before"]
    assert provenance["cwp_observations"]["batches"][0][
        "observed_points"] == thinning["points_after"]


def test_error_inflation_is_applied_exactly_once(tmp_path):
    """Inflating in both the thinner and the adapter would square it."""

    grid = _grid()
    cells = [(j, i) for j in range(3, 6) for i in range(3, 6)]
    n = len(cells)
    product, _ = _product(tmp_path, grid, cells, phase=[4.0] * n,
                          cod=[20.0] * n, cps=[30.0] * n)
    document = read_goes_grid(product, expected_grid=grid)
    stated = document["variables"]["cwp_err"][
        document["variables"]["cwp_mask"].astype(bool)]

    batches, _ = goes_grid_to_gridded_obs(
        product, expected_grid=grid, cwp_simulated=np.zeros((3, NY, NX)),
        error_inflation=2.0)
    used = np.asarray(batches[0].errors)[np.asarray(batches[0].mask)]
    assert np.allclose(np.sort(used), np.sort(stated) * 2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# the command lines
# ---------------------------------------------------------------------------


def _wrfout(path, grid):
    """A minimal wrfout carrying exactly the georeference TargetGrid reads."""

    import netCDF4

    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", grid.ny)
        dataset.createDimension("west_east", grid.nx)
        dataset.createDimension("bottom_top_stag", grid.nz + 1)
        dataset.setncatts({
            "MAP_PROJ": np.int32(grid.projection.wrf_map_proj),
            "TRUELAT1": np.float64(grid.truelat1),
            "TRUELAT2": np.float64(grid.truelat2),
            "STAND_LON": np.float64(grid.stand_lon),
            "CEN_LAT": np.float64(grid.projection.cen_lat),
            "CEN_LON": np.float64(grid.projection.cen_lon),
            "DX": np.float64(grid.dx_m), "DY": np.float64(grid.dy_m)})
        plane = ("Time", "south_north", "west_east")
        for name, values in (("XLAT", grid.lat), ("XLONG", grid.lon),
                             ("HGT", grid.terrain_m)):
            variable = dataset.createVariable(name, "f4", plane)
            variable[:] = np.asarray(values, np.float32)[None]
        volume = ("Time", "bottom_top_stag", "south_north", "west_east")
        phb = dataset.createVariable("PHB", "f4", volume)
        phb[:] = np.asarray(grid.z_w * 9.81, np.float32)[None]
        ph = dataset.createVariable("PH", "f4", volume)
        ph[:] = np.zeros((1, grid.nz + 1, grid.ny, grid.nx), np.float32)
    return path


def test_builder_cli_writes_a_product_from_two_packs(tmp_path):
    """The real tool, in a subprocess, against real packs and a real grid."""

    grid = _grid()
    wrfout = _wrfout(tmp_path / "wrfout_d01", grid)
    cells = [(4, 4), (4, 5)]
    lat = np.array([[grid.lat[j, i] for (j, i) in cells]], np.float32)
    lon = np.array([[grid.lon[j, i] for (j, i) in cells]], np.float32)
    cwp_path = write_cwp_pack(
        tmp_path / "cwp.goespack", cod=np.array([[20.0, 0.0]], np.float32),
        cps=np.array([[30.0, 0.0]], np.float32),
        phase=np.array([[4.0, 0.0]], np.float32), lat=lat, lon=lon,
        x_scan_rad=[0.0, 0.002], y_scan_rad=[0.08])
    ct_path = write_cloudtop_pack(
        tmp_path / "ct.goespack",
        cloud_top_height_m=np.array([[9000.0, np.nan]], np.float32),
        lat=lat, lon=lon, x_scan_rad=[0.0, 0.002], y_scan_rad=[0.08],
        sibling=sibling_block(cwp_path))
    out = tmp_path / "goes_grid.nc"
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "obs_goes_grid_build.py"),
         "--cwp-pack", str(cwp_path), "--cloudtop-pack", str(ct_path),
         "--grid-wrfout", str(wrfout), "--valid-time",
         "2026-08-04T18:01:17Z", "--out", str(out),
         "--err-clear-g-m2", "20", "--err-rel-liquid", "0.3",
         "--err-floor-liquid-g-m2", "40", "--err-rel-ice", "0.5",
         "--err-floor-ice-g-m2", "80"],
        capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["product"]["schema"] == GOES_GRID_SCHEMA
    assert receipt["product"]["observations"] == 2
    assert receipt["join"]["method"] == "nearest"
    assert receipt["error_model"]["calibration"] == "UNCALIBRATED"
    assert receipt["counts"]["observations_at_retrieved_top"] == 1
    assert receipt["counts"]["observations_at_fallback_height"] == 1

    # The product is bound to the grid the TOOL built, which is the
    # wrfout's own float32 round trip of the analytic one -- a genuinely
    # different identity, and the consumer has to bring the same one.
    from_file = TargetGrid.from_wrfout(wrfout)
    with pytest.raises(GridMismatchError):
        read_goes_grid(out, expected_grid=grid)
    document = read_goes_grid(out, expected_grid=from_file)
    assert document["variables"]["cwp_mask"].sum() == 2
    assert document["variables"]["obs_level"][4, 4] == 3


def test_builder_cli_requires_the_uncalibrated_errors_to_be_stated(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "obs_goes_grid_build.py"),
         "--cwp-pack", "x", "--grid-wrfout", "y", "--valid-time", "z",
         "--out", str(tmp_path / "o.nc")],
        capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode != 0
    for flag in ("--err-clear-g-m2", "--err-rel-liquid", "--err-rel-ice"):
        assert flag in result.stderr


def _driver_cli(tmp_path, *extra):
    """The driver's own required arguments, plus whatever the test adds.

    These have to be satisfied before argparse reaches the cross-flag
    refusals, which are the thing under test.
    """

    # PYTHONPATH pins the subprocess to THIS tree, the way
    # tests/test_build_registry.py and tests/test_gpu_marker_discipline.py
    # already do.  Without it the child runs this worktree's SCRIPT while
    # importing gpuwm from wherever the editable install points -- another
    # checkout entirely -- so the driver died on `No module named
    # gpuwm.da` long before argparse reached the cross-flag refusals these
    # tests are about, and the failure looked like a changed refusal
    # message rather than a test exercising the wrong tree.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        [sys.executable, str(REPO / "tools" / "da_cycle_prepared.py"),
         "--prepared-root", str(tmp_path),
         "--proof-sha256", "0" * 64,
         "--source-manifest-sha256", "0" * 64,
         "--prepared-content-sha256", "0" * 64,
         "--physics-profile", "none", "--run-seconds", "600",
         "--history-interval-seconds", "600", "--out", str(tmp_path / "out"),
         *extra],
        capture_output=True, text=True, cwd=str(REPO), env=environment)


@pytest.mark.parametrize("extra,message", [
    ([], "needs --hydrometeors"),
    (["--hydrometeors", "--positivity-policy", "clip"],
     "explicit --cwp-vertical-loc-m"),
])
def test_driver_cli_refuses_an_underspecified_satellite_analysis(
        tmp_path, extra, message):
    result = _driver_cli(
        tmp_path, "--goes-cwp", str(tmp_path / "g.nc"),
        "--obs", str(tmp_path / "o.nc"), *extra)
    assert result.returncode != 0
    assert message in result.stderr


def test_driver_cli_refuses_more_satellite_files_than_legs(tmp_path):
    result = _driver_cli(
        tmp_path, "--goes-cwp", str(tmp_path / "a.nc"),
        "--goes-cwp", str(tmp_path / "b.nc"),
        "--obs", str(tmp_path / "o.nc"), "--hydrometeors",
        "--positivity-policy", "clip", "--cwp-vertical-loc-m", "20000")
    assert result.returncode != 0
    assert "matched to legs by position" in result.stderr
