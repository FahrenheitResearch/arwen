"""File-backed dry and moist temperature boundary contracts."""

import netCDF4
import numpy as np
import pytest

from gpuwm.ingest import wrfinput as wi
from gpuwm.ingest.lateral_bc import evaluate_boundary_side
from test_analyzed_scalar_boundaries import _cfg, _input, _read, _boundary


@pytest.mark.parametrize("moist", [False, True])
def test_file_temperature_forcing_matches_its_declared_representation(tmp_path, moist):
    cfg = _cfg()
    path = _input(tmp_path / "input", cfg)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.USE_THETA_M = int(moist)
    initial = _read(path, cfg)
    boundary = _boundary(tmp_path / "boundary", initial, cfg)
    ratio = np.float32(461.6) / np.float32(287.0)
    with netCDF4.Dataset(boundary, "a") as dataset:
        for suffix in ("XS", "XE", "YS", "YE"):
            theta = dataset[f"T_B{suffix}"]
            vapour = dataset[f"QVAPOR_B{suffix}"]
            start = np.asarray(theta[0], np.float32)
            if moist:
                # The fixture has total dry mass 13 and unit map factors.
                dry = start / np.float32(13.) + np.float32(300.)
                factor = np.float32(1.) + ratio * (
                    np.asarray(vapour[0], np.float32) / np.float32(13.))
                start = (factor * dry - np.float32(300.)) * np.float32(13.)
            theta[0] = start
            theta[1] = start + np.float32(60.) * np.float32(.125)
            dataset[f"T_BT{suffix}"][:] = np.float32(.125)
            first_q = np.asarray(vapour[0], np.float32)
            vapour[1] = first_q + np.float32(60.) * np.float32(.00002)
            dataset[f"QVAPOR_BT{suffix}"][:] = np.float32(.00002)
    result = wi.read_wrfbdy(boundary, restored=initial, run_seconds=60,
                            forcing_interval_seconds=60, cfg=cfg)
    with netCDF4.Dataset(boundary) as dataset:
        for side, suffix in (("west", "XS"), ("east", "XE"),
                             ("south", "YS"), ("north", "YE")):
            order = (1, 2, 0) if side in ("west", "east") else (1, 0, 2)
            read = lambda name: np.asarray(dataset[name + suffix][0], float)
            A, Ad, Q, Qd = read("T_B"), read("T_BT"), read("QVAPOR_B"), read("QVAPOR_BT")
            for seconds in (0., 12., 30., 60.):
                encoded = A + seconds * Ad
                expected = (13. * ((encoded / 13. + 300.) /
                            (1. + float(ratio) * (Q + seconds * Qd) / 13.) - 300.)
                            if moist else encoded)
                actual, _ = evaluate_boundary_side(
                    getattr(result.intervals[0].fields["theta"], side), seconds)
                np.testing.assert_allclose(actual, expected.transpose(order),
                                           rtol=2e-13, atol=2e-12)


def test_moist_flag_with_dry_temperature_records_is_not_silently_accepted(tmp_path):
    cfg = _cfg()
    path = _input(tmp_path / "input", cfg)
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.USE_THETA_M = 1
    initial = _read(path, cfg)
    boundary = _boundary(tmp_path / "boundary", initial, cfg)
    with pytest.raises(ValueError, match="does not match initial"):
        wi.read_wrfbdy(boundary, restored=initial, run_seconds=60,
                       forcing_interval_seconds=60, cfg=cfg)


def test_single_interval_character_end_time_reaches_the_boundary_reader(tmp_path):
    cfg = _cfg()
    initial = _read(_input(tmp_path / "input", cfg), cfg)
    original = _boundary(tmp_path / "two-records", initial, cfg)
    single = tmp_path / "single-record"
    with netCDF4.Dataset(original) as source, netCDF4.Dataset(single, "w") as target:
        for name, dimension in source.dimensions.items():
            target.createDimension(name, 1 if name == "Time" else len(dimension))
        target.setncatts({name: source.getncattr(name) for name in source.ncattrs()})
        for name, variable in source.variables.items():
            target.createVariable(name, variable.datatype, variable.dimensions)[:] = variable[:1]
        name = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
        target.createVariable(name, "S1", ("Time", "DateStrLen"))[:] = np.frombuffer(
            b"2026-08-25_18:01:00", dtype="S1").reshape(1, 19)
    result = wi.read_wrfbdy(single, restored=initial, run_seconds=60,
                            forcing_interval_seconds=60, cfg=cfg)
    assert len(result.intervals) == 1
    assert result.intervals[0].end_seconds == 60
