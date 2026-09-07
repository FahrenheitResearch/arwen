"""Original geometry and full donors remain authoritative under local packing."""
from dataclasses import replace
import json

import numpy as np
import pytest

from gpuwm.ingest.atmospheric_window import (
    ATMOSPHERIC_FIELDS, AtmosphericWindow, WindowedAtmosphericField,
    WindowedAtmosphericSnapshot, atmospheric_window_for_grids,
)
from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.static.lambert import LambertGrid
from gpuwm.mapped_source import mapped_frames_to_regular_snapshots
import test_mapped_frameset_streaming as fixture


@pytest.fixture
def source(monkeypatch, tmp_path):
    monkeypatch.setattr(fixture, "_NY", 31)
    monkeypatch.setattr(fixture, "_NX", 37)
    frame = fixture._one_frame()
    fields = dict(frame.fields)
    pressure = fields["air_pressure"].values.copy()
    pressure += np.arange(31 * 37).reshape(31, 37) * .1
    fields["air_pressure"] = replace(fields["air_pressure"], values=pressure)
    fields["unused_diagnostic"] = replace(fields["air_temperature"], name="unused_diagnostic")
    frame = replace(frame, fields=fields)
    directory = fixture.engine_bridge.write_frameset(tmp_path / "frames", (frame,))
    authority = tmp_path / "authority"
    authority.write_text("atmospheric-window witness")
    return frame, fixture._bundle(directory, authority)


def target(lat=36., lon=-96.):
    return LambertGrid(lat, lon, 30., 60., -97., 3000., 3000., 12, 14)


def assert_horizontal_exact(a, b):
    assert a.fields.keys() == b.fields.keys()
    assert a.levels_hpa.tobytes() == b.levels_hpa.tobytes()
    for name in a.fields:
        assert a.fields[name].tobytes() == b.fields[name].tobytes(), name


def test_provider_packs_only_atmosphere_with_original_pressure_ladder(source):
    frame, bundle = source
    full = bundle.regular_snapshots()[0]
    sequence = bundle.regular_snapshots().for_grids((target(),))
    actual = sequence.sorted_by_valid_time()[:][0]
    assert isinstance(actual, WindowedAtmosphericSnapshot)
    assert actual.latitude.tobytes() == full.latitude.tobytes()
    assert actual.longitude.tobytes() == full.longitude.tobytes()
    assert actual.levels_hpa.tobytes() == full.levels_hpa.tobytes()
    assert not np.array_equal(actual.levels_hpa, np.median(actual.fields["PRES"], axis=(1, 2)) / 100.)
    for name, values in full.fields.items():
        expected = actual.window.crop(values) if name in ATMOSPHERIC_FIELDS else values
        assert actual.fields[name].tobytes() == expected.tobytes(), name
        assert not actual.fields[name].flags.writeable
    full_bytes = sum(v.nbytes for v in full.fields.values())
    small_bytes = sum(v.nbytes for v in actual.fields.values())
    nlev = len(full.levels_hpa)
    saved = 8 * nlev * len(ATMOSPHERIC_FIELDS) * (
        np.prod(actual.window.source_shape) - np.prod(actual.window.shape))
    assert full_bytes - small_bytes == saved
    assert small_bytes < full_bytes / 8
    assert bundle.frames._cached_frame is None


def test_actual_cpu_join_all_fields_and_mass_u_v_union(source):
    _, bundle = source
    grids = (target(), target(37., -102.))
    full = bundle.regular_snapshots()[0]
    small = bundle.regular_snapshots().for_grids(grids)[0]
    assert isinstance(small, WindowedAtmosphericSnapshot)
    for grid in grids:
        assert_horizontal_exact(interpolate_era5_to_lambert(full, grid, backend="cpu"),
                                interpolate_era5_to_lambert(small, grid, backend="cpu"))


def test_outside_request_reloads_original_atmosphere_and_preserves_overlay(source):
    _, bundle = source
    full = bundle.regular_snapshots()[0]
    small = bundle.regular_snapshots().for_grids((target(),))[0]
    overlay = full.fields["SKINTEMP"] + 1.25
    small = small.with_fields({"SKINTEMP": overlay})
    full = full.with_fields({"SKINTEMP": overlay})
    outside = target(33., -101.)
    assert_horizontal_exact(interpolate_era5_to_lambert(full, outside, backend="cpu"),
                            interpolate_era5_to_lambert(small, outside, backend="cpu"))


@pytest.mark.parametrize("name", ["air_temperature", "unused_diagnostic"])
def test_corruption_outside_window_and_unused_fields_still_refuse(source, name):
    _, bundle = source
    path = bundle.frames.directory
    document = json.loads((path / "frames.json").read_text())
    entry = next(row for row in document["frames"][0]["fields"] if row["name"] == name)
    with (path / "frames.f64").open("r+b") as stream:
        stream.seek(entry["offset"])
        original = stream.read(1)
        stream.seek(entry["offset"])
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(ValueError, match="hashes to"):
        bundle.regular_snapshots().for_grids((target(),))[0]


@pytest.mark.parametrize("method", ["parabolic", "bilinear"])
@pytest.mark.parametrize("edge", [False, True])
def test_pre_cropped_operand_uses_original_fp32_plan_not_rebased_axes(method, edge):
    shape = (63, 81)
    rng = np.random.default_rng(142)
    values = rng.normal(size=(3, *shape))
    y, x = np.array([[22.123456789]]), np.array([[34.123456789]])
    if edge:
        y, x = np.array([[0., 1.00000001]]), np.array([[79.9999999, 80.]])
    plan = CpuPreprocessBackend().indexed_plan(shape, y, x)
    support = plan._source_support
    window = AtmosphericWindow(shape, (support.rows.start, support.rows.stop),
                                (support.columns.start, support.columns.stop))
    operand = WindowedAtmosphericField(window.crop(values).copy(), window)
    actual = plan.apply(operand, method=method, source_support=True)
    assert actual.tobytes() == plan.apply(values, method=method).tobytes()


def test_full_snapshot_constructor_still_refuses_cropped_arrays(source):
    frame, _ = source
    full = mapped_frames_to_regular_snapshots((frame,), initialize_absent_hydrometeors=True)[0]
    with pytest.raises(ValueError, match="shape"):
        replace(full, fields={"T": full.fields["T"][:, :4, :5]})


def test_cyclic_axis_keeps_existing_full_provider(source):
    from gpuwm.ingest.source_metadata import SourceSnapshotMetadata
    from gpuwm.ingest.grib import Era5Snapshot
    meta = SourceSnapshotMetadata(Era5Snapshot, np.arange(-90., 91.), np.arange(360.))
    assert atmospheric_window_for_grids(meta, (target(),)) is None


def test_window_fallback_provider_does_not_retain_cached_atmosphere_cycle(source):
    import gc
    import weakref
    _, bundle = source
    enabled = gc.isenabled()
    gc.disable()
    try:
        sequence = bundle.regular_snapshots().for_grids((target(),))
        snapshot = sequence[0]
        reference = weakref.ref(snapshot.fields["T"])
        del sequence
        assert reference() is not None
        del snapshot
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


def test_plane_reader_has_no_full_atmospheric_allocation_and_keeps_global_median(source, monkeypatch):
    import weakref
    _, bundle = source
    reader = bundle.frames
    document = next(row for row in reader._entries[0]["fields"] if row["name"] == "air_pressure")
    expected = reader.field(0, "air_pressure").values
    allocations = []
    original = np.empty
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        allocations.append((result.shape, weakref.ref(result)))
        return result
    window = AtmosphericWindow(expected.shape[1:], (14, 18), (20, 25))
    monkeypatch.setattr(np, "empty", observed)
    field, levels = reader._read_window_field(0, document, window)
    assert field.values.tobytes() == window.crop(expected).tobytes()
    assert levels.tobytes() == (np.median(expected, axis=(1, 2)) / 100.).tobytes()
    assert expected.shape not in [shape for shape, _ in allocations]
    assert [shape for shape, _ in allocations] == [(24, 4, 5), (31, 37), (24,)]
    assert allocations[1][1]() is None


@pytest.mark.parametrize("value,missing,match", [
    (np.inf, 0, "infinity"), (np.nan, 0, "missing count"), (-1., 0, "finite and positive"),
])
def test_outside_pressure_value_contract_survives_valid_resealed_hash(source, value, missing, match):
    from gpuwm.mapped_source import _array_sha256
    _, bundle = source
    frames = bundle.frames
    document = next(row for row in frames._entries[0]["fields"] if row["name"] == "air_pressure")
    values = frames.field(0, "air_pressure").values.copy()
    values[0, 0, 0] = value
    with (frames.directory / "frames.f64").open("r+b") as stream:
        stream.seek(document["offset"])
        stream.write(values.tobytes())
    document["sha256"] = _array_sha256(values)
    document["missing_count"] = missing
    with pytest.raises(ValueError, match=match):
        bundle.regular_snapshots().for_grids((target(),))[0]


def test_pressure_coverage_reader_keeps_complete_plane_medians(source, monkeypatch):
    _, bundle = source
    expected = bundle.regular_snapshots().source_pressure_hpa(0)
    original = bundle.frames.field
    def no_full_pressure(index, name):
        assert name != "air_pressure", "coverage allocated a full pressure atmosphere"
        return original(index, name)
    monkeypatch.setattr(bundle.frames, "field", no_full_pressure)
    actual = bundle.regular_snapshots().for_grids((target(),)).source_pressure_hpa(0)
    assert actual.tobytes() == expected.tobytes()


def test_full_reload_rejects_changed_source_clock(source):
    from datetime import timedelta
    _, bundle = source
    full = bundle.regular_snapshots()[0]
    small = bundle.regular_snapshots().for_grids((target(),))[0]
    other = replace(full, valid_time=full.valid_time + timedelta(hours=1))
    small = replace(small, full_factory=lambda: other)
    with pytest.raises(ValueError, match="source geometry, clock, or ladder"):
        small.full_snapshot()


@pytest.mark.parametrize("projected,reverse", [(False, True), (True, False)])
def test_declared_projection_and_scan_order_keep_all_horizontal_bytes(source, tmp_path, projected, reverse):
    from gpuwm import mapped_source as ms
    frame, _ = source
    grid = target()
    if projected:
        parameters = dict(latin1=30., latin2=60., lov=-97., lat1=35., lon1=-100.,
                          dx_m=6000., dy_m=6000., nx=37, ny=31,
                          earth_radius_m=6371229., shape_of_earth=6)
        latitude, longitude = ms._projected_axes(parameters)
        frame = replace(frame, latitude=latitude, longitude=longitude,
            header=replace(frame.header, grid=replace(frame.header.grid,
                projection="lambert_conformal",
                parameters={**parameters, "axis_unit_m": ms.PROJECTED_AXIS_UNIT_M})))
        source_grid = ms.declared_lambert_source_grid(parameters)
        lat, lon = source_grid.ij_to_latlon(18., 15.)
        grid = target(float(lat), float(lon))
    if reverse:
        frame = replace(frame, latitude=frame.latitude[::-1], longitude=frame.longitude[::-1],
                        fields={name: replace(field, values=field.values[..., ::-1, ::-1])
                                for name, field in frame.fields.items()})
    path = fixture.engine_bridge.write_frameset(tmp_path / "geometry", (frame,))
    authority = tmp_path / "authority"
    bundle = fixture._bundle(path, authority)
    full = bundle.regular_snapshots()[0]
    small = bundle.regular_snapshots().for_grids((grid,))[0]
    assert isinstance(small, WindowedAtmosphericSnapshot)
    assert small.projection == full.projection
    assert_horizontal_exact(interpolate_era5_to_lambert(full, grid, backend="cpu"),
                            interpolate_era5_to_lambert(small, grid, backend="cpu"))


@pytest.mark.parametrize("shape,rows,columns", [
    ((20,30), (-1,4), (1,4)), ((20,30), (2,2), (1,4)),
    ((20,30), (1,21), (1,4)), ((20,30), (True,4), (1,4)),
])
def test_window_descriptor_requires_real_original_grid_bounds(shape, rows, columns):
    with pytest.raises(ValueError, match="atmospheric window"):
        AtmosphericWindow(shape, rows, columns)
