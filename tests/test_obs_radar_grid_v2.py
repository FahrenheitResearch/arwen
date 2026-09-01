"""radar-grid v2: velocity on each radar's reach window, and v1 still readable.

v1 stored every radar's velocity plane over the whole analysis domain.  A
radar sees a disc ~250 km across and a continental grid is ~5000 km
across, so at 49 bytes per cell per radar that layout is 715 GB for a
160-radar CONUS analysis and ~99% of it zeros.  v2 stores each plane on
that radar's reach window with the window's origin beside it, which is
~8.9 GB for the same network -- the difference between needing a
supercomputer's memory and fitting on rentable hardware.

Nothing else changes.  Every field, unit and meaning is v1's; reflectivity
is not windowed at all, because it merges across radars into one field the
whole network contributed to and was never the problem.

What this file pins:

* v1 files STILL READ.  Receipts recorded the sha256 of v1 files, and a
  receipt whose file can no longer be opened is a dead receipt.  This is
  the property a schema bump is most likely to break and the one it can
  least afford to.
* a v2 file carries bit-identical observations to the v1 file written from
  the same merge -- checked field by field through the whole-domain view,
  with a guard that the windows really were smaller than the domain;
* the adapter produces bit-identical LETKF batches from either;
* fail closed: a window that under-covers its radar, or falls off the
  grid, is refused by the writer and by the reader.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpuwm.obs.superob as superob_mod
from gpuwm.da.letkf import (LetkfConfig, LetkfDiagnostics, Localization,
                            analyze)
from gpuwm.da.obs_radar import letkf_grid_geometry, radar_grid_to_gridded_obs
from gpuwm.obs.radar_grid import (
    RADAR_GRID_SCHEMA_V1,
    RADAR_GRID_SCHEMA_V2,
    RadarGridSchemaError,
    radar_plane,
    read_radar_grid,
    write_radar_grid,
)
from gpuwm.obs.superob import SuperobParams, merge_contributions, superob_volume
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

from test_obs_superob_window import (  # noqa: E402
    PLACED,
    REF_LAT,
    REF_LON,
    _full_domain_window,
    _grid,
    _params,
    _volume_at,
)

VELOCITY_FIELDS = ("vr_obs", "vr_mask", "vr_err", "vr_count", "vr_rejected",
                   "vr_beam_east", "vr_beam_north", "vr_beam_up")


def _merged(grid, params, *, dense: bool, monkeypatch=None):
    """The same three radars, merged into one layout or the other.

    ``dense`` reproduces a v1 producer exactly: whole-domain accumulators,
    and no window metadata at all.  Dropping ``radar_windows`` is the
    second half and not a shortcut -- a set whose windows happen to equal
    the domain is still in v2's layout, and what a v1 producer emitted was
    a set that had no concept of a window.
    """
    if dense:
        monkeypatch.setattr(superob_mod, "horizontal_window",
                            _full_domain_window)
    contributions = [
        superob_volume(_volume_at(grid, site_id=s, j=j, i=i), grid,
                       params=params)
        for s, (j, i) in PLACED.items()]
    observations = merge_contributions(contributions, grid, params=params)
    if dense:
        observations.radar_windows = []
    return observations


def _write(path, observations, grid, params):
    return write_radar_grid(path, observations, grid,
                            valid_time="2026-08-01T11:30:00Z", params=params,
                            provenance={"builder": "test"})


# --------------------------------------------------------------------------
# The schema bump itself.
# --------------------------------------------------------------------------


def test_a_windowed_merge_writes_v2(tmp_path):
    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=False)
    assert observations.windowed
    receipt = _write(tmp_path / "v2.nc", observations, grid, params)
    assert receipt["schema"] == RADAR_GRID_SCHEMA_V2
    document = read_radar_grid(tmp_path / "v2.nc")
    assert document["schema"] == RADAR_GRID_SCHEMA_V2
    assert len(document["radar_windows"]) == len(PLACED)


def test_a_whole_domain_merge_still_writes_v1(tmp_path, monkeypatch):
    """Nothing that produced a v1 file has to change to keep producing one."""
    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=True, monkeypatch=monkeypatch)
    assert not observations.windowed
    receipt = _write(tmp_path / "v1.nc", observations, grid, params)
    assert receipt["schema"] == RADAR_GRID_SCHEMA_V1


def test_v1_files_still_read(tmp_path, monkeypatch):
    """The property a schema bump is most likely to break.

    A receipt recorded the digest of a v1 file.  If v2 stopped reading v1,
    every one of those receipts would become uncheckable -- the file is
    still on disk and still correct, and nothing can open it.
    """
    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=True, monkeypatch=monkeypatch)
    _write(tmp_path / "v1.nc", observations, grid, params)

    document = read_radar_grid(tmp_path / "v1.nc", expected_grid=grid)
    assert document["schema"] == RADAR_GRID_SCHEMA_V1
    # And its planes are reachable through the same accessor as v2's, so a
    # consumer does not branch on schema to read an observation.
    assert radar_plane(document, "vr_obs", 0).shape == (grid.nz, grid.ny,
                                                        grid.nx)
    assert document["radar_windows"][0] == [0, grid.ny - 1, 0, grid.nx - 1]


# --------------------------------------------------------------------------
# The correctness claim.
# --------------------------------------------------------------------------


def test_v2_carries_bit_identical_observations_to_v1(tmp_path, monkeypatch):
    """Same volumes, two layouts, one set of numbers.

    Compared through the whole-domain view so the two are commensurable at
    all, field by field, bitwise.  The assertion on window size is what
    stops this passing vacuously: if v2's windows were the whole domain
    the layouts would be identical and the test would prove nothing.
    """
    grid, params = _grid(), _params()

    windowed = _merged(grid, params, dense=False)
    _write(tmp_path / "v2.nc", windowed, grid, params)
    v2 = read_radar_grid(tmp_path / "v2.nc", expected_grid=grid)

    dense = _merged(grid, params, dense=True, monkeypatch=monkeypatch)
    _write(tmp_path / "v1.nc", dense, grid, params)
    v1 = read_radar_grid(tmp_path / "v1.nc", expected_grid=grid)

    assert v1["schema"] == RADAR_GRID_SCHEMA_V1
    assert v2["schema"] == RADAR_GRID_SCHEMA_V2
    # Non-vacuity: v2's stored planes really are smaller than the domain.
    stored = v2["variables"]["vr_obs"].shape
    assert stored[2] < grid.ny and stored[3] < grid.nx, (
        f"v2 stored {stored} on a {grid.ny}x{grid.nx} grid: not windowed")

    for index in range(len(PLACED)):
        for name in VELOCITY_FIELDS:
            np.testing.assert_array_equal(
                radar_plane(v2, name, index), radar_plane(v1, name, index),
                err_msg=f"radar {index} field {name} differs between layouts")

    # Reflectivity is not windowed and must be untouched by any of this.
    for name in ("z_obs", "z_mask", "z_err", "z_max", "z_mean", "z_count"):
        np.testing.assert_array_equal(v2["variables"][name],
                                      v1["variables"][name],
                                      err_msg=f"{name} differs")


def test_the_fixture_actually_carries_observations(tmp_path):
    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=False)
    _write(tmp_path / "v2.nc", observations, grid, params)
    document = read_radar_grid(tmp_path / "v2.nc")
    assert int(np.sum(document["variables"]["vr_mask"])) > 0
    assert int(np.sum(document["variables"]["z_mask"])) > 0


def test_the_adapter_builds_identical_batches_from_either_layout(
        tmp_path, monkeypatch):
    """The filter must not be able to tell which layout it was fed."""
    grid, params = _grid(), _params()
    members = 3
    rng = np.random.default_rng(7)
    shape = (grid.nz, grid.ny, grid.nx)
    simulated = rng.standard_normal((members,) + shape)

    windowed = _merged(grid, params, dense=False)
    _write(tmp_path / "v2.nc", windowed, grid, params)
    dense = _merged(grid, params, dense=True, monkeypatch=monkeypatch)
    _write(tmp_path / "v1.nc", dense, grid, params)

    def _adapt(path):
        return radar_grid_to_gridded_obs(
            path, expected_grid=grid,
            velocity_simulated=lambda index, radar: simulated)

    from_v2, _ = _adapt(tmp_path / "v2.nc")
    from_v1, _ = _adapt(tmp_path / "v1.nc")

    assert [b.name for b in from_v2] == [b.name for b in from_v1]
    assert len(from_v2) == len(PLACED)

    # The batches are deliberately NOT the same shape: v2's are windowed,
    # which is the entire point -- the filter never sees a whole-domain
    # array for a radar that covers a fraction of the domain.  So the
    # property to prove is not that the arrays match, it is that the
    # ANALYSIS does.  That is what "the filter cannot tell which layout it
    # was fed" has to mean.
    assert all(b.window is not None for b in from_v2)
    assert all(b.window is None for b in from_v1)

    geometry = letkf_grid_geometry(grid)
    prior = {"u": rng.standard_normal((members,) + shape),
             "theta": rng.standard_normal((members,) + shape) + 300.0}
    cfg = LetkfConfig(
        localization=Localization(horizontal_m=3.0 * float(grid.dx_m),
                                  vertical_m=1500.0),
        analysis_fields=("u", "theta"), rtps_alpha=0.0, chunk_points=64)

    v2_diag, v1_diag = LetkfDiagnostics(), LetkfDiagnostics()
    from_v2_analysis = analyze(prior, from_v2, geometry, cfg, v2_diag)
    from_v1_analysis = analyze(prior, from_v1, geometry, cfg, v1_diag)

    for field in from_v1_analysis:
        np.testing.assert_array_equal(
            from_v2_analysis[field], from_v1_analysis[field],
            err_msg=f"the layout changed the analysis of {field}")

    # Non-vacuity at both ends: observations existed, and the analysis
    # actually moved the state.
    assert any(int(b.mask.sum()) > 0 for b in from_v2), (
        "no batch carried an observation; the comparison proved nothing")
    assert v2_diag.active_points > 0
    assert v2_diag.active_points == v1_diag.active_points
    assert np.max(np.abs(from_v2_analysis["u"])) > 0.0


# --------------------------------------------------------------------------
# Fail closed.
# --------------------------------------------------------------------------


def test_the_writer_refuses_a_window_that_under_covers_its_radar(tmp_path):
    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=False)
    j0, j1, i0, i1 = observations.radar_windows[0]
    # Claim a wider window than the stored plane can hold: the plane would
    # be missing observations the window says are there.
    observations.radar_windows[0] = [j0, j0 + grid.ny - 1, i0, i1]
    with pytest.raises(Exception, match="under-covers|does not fit"):
        _write(tmp_path / "bad.nc", observations, grid, params)


def test_the_writer_refuses_a_window_off_the_grid(tmp_path):
    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=False)
    j0, j1, i0, i1 = observations.radar_windows[0]
    observations.radar_windows[0] = [j0, j1, grid.nx - 1, grid.nx + 3]
    with pytest.raises(Exception, match="does not fit"):
        _write(tmp_path / "bad.nc", observations, grid, params)


def test_the_reader_refuses_a_window_that_falls_off_the_grid(tmp_path):
    """A file whose window variables place a plane outside the domain."""
    import netCDF4

    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=False)
    _write(tmp_path / "v2.nc", observations, grid, params)
    with netCDF4.Dataset(tmp_path / "v2.nc", "a") as dataset:
        dataset.variables["radar_j0"][0] = grid.ny - 1
    with pytest.raises(RadarGridSchemaError, match="falls outside"):
        read_radar_grid(tmp_path / "v2.nc")


def test_the_reader_refuses_a_v2_file_missing_its_window_variables(tmp_path):
    """Without them the planes are bytes without a coordinate system."""
    import netCDF4

    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=False)
    _write(tmp_path / "v2.nc", observations, grid, params)

    with netCDF4.Dataset(tmp_path / "v2.nc", "r") as src, \
            netCDF4.Dataset(tmp_path / "stripped.nc", "w",
                            format="NETCDF4_CLASSIC") as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, len(dimension))
        for name, variable in src.variables.items():
            if name == "radar_j0":
                continue
            out = dst.createVariable(name, variable.dtype,
                                     variable.dimensions)
            out.setncatts({k: variable.getncattr(k)
                           for k in variable.ncattrs() if k != "_FillValue"})
            out[:] = variable[:]

    with pytest.raises(RadarGridSchemaError, match="missing variables"):
        read_radar_grid(tmp_path / "stripped.nc")


def test_an_unknown_schema_is_refused_by_name(tmp_path):
    import netCDF4

    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=False)
    _write(tmp_path / "v2.nc", observations, grid, params)
    with netCDF4.Dataset(tmp_path / "v2.nc", "a") as dataset:
        dataset.schema = "gpuwm-obs.radar-grid.v9"
    with pytest.raises(RadarGridSchemaError, match="this module reads"):
        read_radar_grid(tmp_path / "v2.nc")
