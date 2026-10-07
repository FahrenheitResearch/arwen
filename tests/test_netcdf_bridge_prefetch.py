"""One rw_netcdf run for a whole file's variables, the same arrays as one run each.

read_radar_grid decoded each of a radar-grid file's ~28 numeric variables
in its own rw_netcdf process: about 6 s of process starts per analysis on
the recent case.  Dataset.prefetch decodes them in one run.  The breakages
this prevents: a prefetched array that differs from the per-variable read
(policy, masking, dtype, order of records), and prefetch silently falling
back to one process per variable.
"""
import numpy as np

from gpuwm import netcdf_bridge
from gpuwm.obs.radar_grid import read_radar_grid

from test_obs_radar_grid_v2 import _grid, _merged, _params, _write


def test_prefetch_is_one_run_with_the_per_variable_arrays(tmp_path, monkeypatch):
    grid, params = _grid(), _params()
    path = tmp_path / "radar.nc"
    _write(path, _merged(grid, params, dense=False), grid, params)

    single = netcdf_bridge.open_dataset(path)
    single.set_auto_mask(False)
    names = [n for n, v in single.variables.items() if not v.is_character]
    expected = {n: np.asarray(single.variables[n][:]) for n in names}

    runs = []
    real = netcdf_bridge._run

    def counting(arguments, **kwargs):
        runs.append(arguments[1])
        return real(arguments, **kwargs)

    monkeypatch.setattr(netcdf_bridge, "_run", counting)
    batched = netcdf_bridge.open_dataset(path)
    batched.set_auto_mask(False)
    runs.clear()
    assert batched.prefetch(names) == 1
    got = {n: np.asarray(batched.variables[n][:]) for n in names}
    assert runs == ["dump"]
    for n in names:
        assert got[n].dtype == expected[n].dtype and got[n].shape == expected[n].shape
        np.testing.assert_array_equal(got[n], expected[n])


def test_radar_grid_reader_decodes_in_one_run(tmp_path, monkeypatch):
    grid, params = _grid(), _params()
    path = tmp_path / "radar.nc"
    _write(path, _merged(grid, params, dense=False), grid, params)
    reference = read_radar_grid(path, expected_grid=grid)
    runs = []
    real = netcdf_bridge._run

    def counting(arguments, **kwargs):
        runs.append(arguments[1])
        return real(arguments, **kwargs)

    monkeypatch.setattr(netcdf_bridge, "_run", counting)
    again = read_radar_grid(path, expected_grid=grid)
    assert runs.count("dump") == 1
    assert sorted(again["variables"]) == sorted(reference["variables"])
    for name, value in reference["variables"].items():
        np.testing.assert_array_equal(again["variables"][name], value)
