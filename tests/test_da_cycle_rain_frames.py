"""Native rain snapshots are sufficient to difference free-forecast hours."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from tools.da_cycle_prepared import _write_composite_wrfout, rain_snapshot


def test_rain_snapshot_preserves_each_accumulator_and_does_not_reset_it():
    convective = np.array([[7.25, 0.0]], dtype=np.float32)
    grid_scale = np.array([[102.5, 3.0]], dtype=np.float32)
    driver = SimpleNamespace(output_fields=lambda: {
        "RAINC": convective, "RAINNC": grid_scale})
    snapshot = rain_snapshot(driver)
    assert snapshot["RAINC"].tobytes() == convective.tobytes()
    assert snapshot["RAINNC"].tobytes() == grid_scale.tobytes()
    snapshot["RAINNC"][0, 0] = 0.0
    assert grid_scale[0, 0] == 102.5


def test_composite_writer_hands_both_native_rain_arrays_to_rust_owner(
        monkeypatch, tmp_path):
    import gpuwm.io.surface_wrfout as writer
    captured = {}
    monkeypatch.setattr(writer, "write_surface_wrfout", lambda path, data, **kw:
                        captured.update(path=path, data=data, options=kw) or
                        SimpleNamespace(path=path, skipped_report=lambda: {}))
    grid = SimpleNamespace(
        truelat1=30.0, truelat2=60.0, stand_lon=-100.0,
        ref_lat=32.0, ref_lon=-101.0,
        latlon_mass=lambda: (np.full((2, 3), 32.0), np.full((2, 3), -101.0)))
    rain = {"RAINC": np.full((2, 3), 2.0, np.float32),
            "RAINNC": np.full((2, 3), 7.0, np.float32)}
    result = _write_composite_wrfout(
        tmp_path/"score.npz", np.zeros((2, 3), np.float32), grid,
        SimpleNamespace(dx=3000.0, dy=3000.0, dt=15.0), 3600.0,
        SimpleNamespace(start_time=datetime(2026, 6, 14, 19, 10, 30)),
        label="forecast", rain=rain)
    assert result is not None
    assert captured["data"]["RAINC"].tobytes() == rain["RAINC"].tobytes()
    assert captured["data"]["RAINNC"].tobytes() == rain["RAINNC"].tobytes()
    assert captured["options"]["time_str"] == "2026-06-14_20:10:30"
    assert captured["options"]["global_attrs"]["GPUWM_RAIN_RESET_ID"] == 0


def test_saved_wrfout_contains_native_accumulations_and_reset_identity(tmp_path):
    netcdf = pytest.importorskip("netCDF4")
    grid = SimpleNamespace(
        truelat1=30.0, truelat2=60.0, stand_lon=-100.0,
        ref_lat=32.0, ref_lon=-101.0,
        latlon_mass=lambda: (np.full((2, 3), 32.0), np.full((2, 3), -101.0)))
    rain = {"RAINC": np.full((2, 3), 2.5, np.float32),
            "RAINNC": np.full((2, 3), 7.25, np.float32)}
    path = _write_composite_wrfout(
        tmp_path/"score.npz", np.full((2, 3), 35.0, np.float32), grid,
        SimpleNamespace(dx=3000.0, dy=3000.0, dt=15.0), 3600.0,
        SimpleNamespace(start_time=datetime(2026, 6, 14, 19, 10, 30)),
        label="forecast", rain=rain)
    assert path is not None
    with netcdf.Dataset(path) as data:
        assert np.asarray(data.variables["RAINC"][0]).tobytes() == rain["RAINC"].tobytes()
        assert np.asarray(data.variables["RAINNC"][0]).tobytes() == rain["RAINNC"].tobytes()
        assert data.GPUWM_RAIN_RESET_ID == 0


@pytest.mark.parametrize("durations", [["60"], ["60", "0"],
                                       ["60", "nan"], ["-1", "60"]])
def test_invalid_explicit_control_clocks_are_refused_before_gpu_use(
        durations, monkeypatch, tmp_path, capsys):
    import sys
    from tools.da_cycle_prepared import main
    monkeypatch.setattr(sys, "argv", [
        "da_cycle_prepared", "--source", "hrrr-prs",
        "--prepared-root", str(tmp_path/"absent"),
        "--physics-profile", "test", "--proof-sha256", "0"*64,
        "--source-manifest-sha256", "0"*64,
        "--prepared-content-sha256", "0"*64, "--run-seconds", "120",
        "--history-interval-seconds", "120", "--free-legs", "2",
        "--out", str(tmp_path/"out"), "--leg-durations-seconds", *durations])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "one finite positive duration" in capsys.readouterr().err
    assert not (tmp_path/"out").exists()
