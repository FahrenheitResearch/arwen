"""Exact support is compared with the existing full-source arithmetic."""
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ingest.interpolation_support import regular_source_support
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
from gpuwm.ingest.horiz import _RegularGpuPlan


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=requires_gpu)])
def backend(request):
    return request.param


def _plan(backend, shape, y, x):
    if backend == "cpu":
        return CpuPreprocessBackend().indexed_plan(shape, y, x)
    # Unit-spaced axes make the plan's index coordinates independent of a
    # particular weather source or projection. No production API is mocked.
    return _RegularGpuPlan(np.arange(shape[0], dtype=np.float64),
                           np.arange(shape[1], dtype=np.float64), y, x)


def _bytes(value):
    return (value.get() if hasattr(value, "get") else value).tobytes()


@pytest.mark.parametrize("case", ["interior", "lower", "upper", "integer_half"])
@pytest.mark.parametrize("method", ["bilinear", "parabolic"])
def test_support_preserves_each_backends_original_bits(backend, case, method):
    rng = np.random.default_rng(93122)
    shape = (63, 81)
    field = rng.normal(size=(3, *shape)).astype(np.float64)
    field[0, ::3, ::3] = 0
    field[1, ::4, ::2] = 1.e-20
    field[2, ::7, ::2] = -1.e-20
    if case == "interior":
        y, x = rng.uniform(20, 25, (7, 9)), rng.uniform(30, 36, (7, 9))
    elif case == "lower":
        y, x = rng.uniform(0, 2, (7, 9)), rng.uniform(0, 2, (7, 9))
        y[0, 0] = x[0, 0] = 0
    elif case == "upper":
        y, x = rng.uniform(60, 62, (7, 9)), rng.uniform(78, 80, (7, 9))
        y[0, 0], x[0, 0] = 62, 80
    else:
        y = np.array([[21., 21.5, np.nextafter(np.float32(21), np.float32(0)),
                       np.nextafter(np.float32(21), np.float32(99))]])
        x = np.array([[31., 31.5, np.nextafter(np.float32(31), np.float32(0)),
                       np.nextafter(np.float32(31), np.float32(99))]])
    plan = _plan(backend, shape, y, x)
    assert plan._source_support is not None
    assert _bytes(plan.apply(field, method=method, source_support=True)) == _bytes(
        plan.apply(field, method=method))


def test_naive_local_coordinates_change_bytes_but_preserved_coordinates_do_not(backend):
    rng = np.random.default_rng(93122)
    shape = (63, 81)
    field = rng.normal(size=(3, *shape))
    y, x = np.array([[22.123456789]]), np.array([[34.123456789]])
    original = _plan(backend, shape, y, x)
    support = original._source_support
    crop = support.crop(field, shape)
    naive = _plan(backend, crop.shape[-2:],
                  y - support.rows.start, x - support.columns.start)
    expected = _bytes(original.apply(field, method="bilinear"))
    assert _bytes(naive.apply(crop, method="bilinear")) != expected
    assert _bytes(original.apply(field, method="bilinear", source_support=True)) == expected


def test_nearest_retains_global_half_even_donor_ties(backend):
    shape = (12, 14)
    field = np.arange(12 * 14, dtype=np.float64).reshape(shape)
    y, x = np.array([[4.5]]), np.array([[6.5]])
    plan = _plan(backend, shape, y, x)
    assert plan._source_support.rows.start % 2 == 1
    expected = field[4:5, 6:7].astype(np.float32).tobytes()
    assert _bytes(plan.apply(field, method="nearest", source_support=True)) == expected
    assert _bytes(plan.apply(field, method="nearest")) == expected


def test_support_keeps_shape_validation_before_slicing(backend):
    plan = _plan(backend, (12, 14), np.array([[4.5]]), np.array([[6.5]]))
    with pytest.raises(ValueError, match="trailing dimensions"):
        plan.apply(np.zeros((11, 14)), source_support=True)


def test_unproven_or_whole_support_falls_back():
    assert regular_source_support((12, 14), [[-0.1]], [[4.]]) is None
    assert regular_source_support((12, 14), [[11.1]], [[4.]]) is None
    assert regular_source_support((12, 14), [[np.nan]], [[4.]]) is None
    assert regular_source_support((12, 14), [[0., 11.]], [[0., 13.]]) is None
    assert regular_source_support((2**24 + 1, 14), [[2.]], [[4.]]) is None


def test_cpu_conversion_receives_only_proven_payload(monkeypatch):
    import gpuwm.ingest.cpu_backend as module
    plan = _plan("cpu", (63, 81), np.array([[22.25]]), np.array([[34.25]]))
    source = np.ones((3, 63, 81), dtype=np.float64)
    seen = []
    original = module._host_f32
    def observed(value):
        seen.append(value.shape)
        return original(value)
    monkeypatch.setattr(module, "_host_f32", observed)
    a = plan.apply(source)
    b = plan.apply(source, source_support=True)
    assert a.tobytes() == b.tobytes()
    assert seen == [(3, 63, 81), (3, 4, 4)]


@pytest.mark.parametrize("centre", [0., 180., 32.])
def test_support_follows_existing_cyclic_orientation(backend, centre):
    from datetime import datetime
    from gpuwm.ingest.grib import Era5Snapshot
    from gpuwm.ingest.horiz import orient_global_source_longitudes
    latitude = np.arange(-20., 21.)
    longitude = np.arange(360., dtype=np.float64)
    values = np.sin(np.deg2rad(longitude))[None, :] + latitude[:, None] * .01
    snapshot = Era5Snapshot(datetime(2026, 9, 1), np.array([500.]),
                            latitude, longitude, {"T": values})
    target_lat = np.array([[.125, .875], [1.25, 1.75]])
    target_lon = centre + np.array([[-.2, .8], [-.75, 1.125]])
    oriented = orient_global_source_longitudes(snapshot, target_lon)
    if backend == "cpu":
        plan = CpuPreprocessBackend().regular_plan(
            oriented.latitude, oriented.longitude, target_lat, target_lon)
    else:
        plan = _RegularGpuPlan(oriented.latitude, oriented.longitude,
                               target_lat, target_lon)
    for method in ("bilinear", "parabolic", "nearest"):
        assert _bytes(plan.apply(oriented.fields["T"], method=method,
                                 source_support=True)) == _bytes(
            plan.apply(oriented.fields["T"], method=method))



def test_actual_netcdf_source_preserves_every_horizontal_field(tmp_path, monkeypatch):
    import netCDF4
    import test_mapped_source as fixture
    from gpuwm.ingest.cpu_backend import _CpuRegularPlan
    from gpuwm.ingest.horiz import interpolate_era5_to_lambert
    from gpuwm.mapped_source import decode_mapped_source, mapped_frames_to_regular_snapshots
    from gpuwm.static.lambert import LambertGrid
    mapping, source = tmp_path / "mapping.json", tmp_path / "source.nc"
    fixture._write_mapping(mapping, fixture._mapping())
    fixture._write_source(source)
    # The fixture has real NetCDF bytes and independent spatial gradients.
    with netCDF4.Dataset(source, "r+") as dataset:
        gradient = np.arange(30, dtype=np.float64).reshape(5, 6)
        for name in ("air_temperature", "eastward_wind", "northward_wind"):
            dataset.variables[name][:] += gradient * .01
    grid = LambertGrid(32., -99.5, 30., 60., -99.5, 1000., 1000., 6, 6)
    snapshots = mapped_frames_to_regular_snapshots(
        decode_mapped_source(mapping, [source]), initialize_absent_hydrometeors=True)
    for snapshot in snapshots:
        actual = interpolate_era5_to_lambert(snapshot, grid, backend="cpu")
        original = _CpuRegularPlan.apply
        def full(self, field, method="parabolic", *, workers=None, source_support=False):
            return original(self, field, method=method, workers=workers,
                            source_support=False)
        with monkeypatch.context() as patch:
            patch.setattr(_CpuRegularPlan, "apply", full)
            expected = interpolate_era5_to_lambert(snapshot, grid, backend="cpu")
        assert actual.fields.keys() == expected.fields.keys()
        for name in actual.fields:
            assert _bytes(actual.fields[name]) == _bytes(expected.fields[name]), name
