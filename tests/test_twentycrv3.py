from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import threading
import time
import weakref

import netCDF4
import numpy as np
import pytest

from gpuwm.twentycrv3 import (
    DISCOVERY_SCHEMA,
    EnsembleRunError,
    FieldBinding,
    build_source_frame_header,
    discover_20crv3,
    estimate_member_bytes,
    plan_member_batches,
    read_member_frame,
    stream_members_atomic,
    validate_bindings,
    validate_target_pressure_levels,
    validate_time_coverage,
)
from gpuwm.source_frame import (
    POLICY_CONTROLLED_FIELDS,
    REQUIRED_3D_FIELDS,
    REQUIRED_SURFACE_FIELDS,
)


def _write_fixture(
    path: Path, *, member_count: int = 2, cadence_hours: int = 1,
    time_values=None, descending_latitude: bool = False,
    descending_longitude: bool = False, ascending_pressure: bool = False,
    permuted: bool = True, bad_temperature_units: bool = False,
    omit_temperature: bool = False, corrupt_member: int | None = None,
) -> Path:
    times = np.asarray(
        list(time_values) if time_values is not None else [0, cadence_hours, 2 * cadence_hours],
        dtype=np.float64,
    )
    latitude = np.asarray([-20.0, 0.0, 20.0], dtype=np.float32)
    longitude = np.asarray([0.0, 90.0, 180.0, 270.0], dtype=np.float32)
    pressure = np.asarray([1000.0, 850.0, 500.0, 100.0], dtype=np.float32)
    if descending_latitude:
        latitude = latitude[::-1]
    if descending_longitude:
        longitude = longitude[::-1]
    if ascending_pressure:
        pressure = pressure[::-1]

    with netCDF4.Dataset(path, "w") as dataset:
        dataset.title = "Synthetic NOAA-CIRES-DOE 20CRv3 ensemble"
        dataset.source = "pytest metadata fixture"
        dataset.createDimension("realization", member_count)
        dataset.createDimension("valid_time", len(times))
        dataset.createDimension("latitude", len(latitude))
        dataset.createDimension("longitude", len(longitude))
        dataset.createDimension("pressure", len(pressure))

        member = dataset.createVariable("realization", "i4", ("realization",))
        member.standard_name = "realization"
        member.axis = "E"
        member[:] = np.arange(1, member_count + 1, dtype=np.int32)

        valid_time = dataset.createVariable("valid_time", "f8", ("valid_time",))
        valid_time.standard_name = "time"
        valid_time.axis = "T"
        valid_time.units = "hours since 2000-01-01 00:00:00 +00:00"
        valid_time.calendar = "standard"
        valid_time[:] = times

        lat = dataset.createVariable("latitude", "f4", ("latitude",))
        lat.standard_name = "latitude"
        lat.axis = "Y"
        lat.units = "degrees_north"
        lat[:] = latitude

        lon = dataset.createVariable("longitude", "f4", ("longitude",))
        lon.standard_name = "longitude"
        lon.axis = "X"
        lon.units = "degrees_east"
        lon[:] = longitude

        level = dataset.createVariable("pressure", "f4", ("pressure",))
        level.standard_name = "air_pressure"
        level.axis = "Z"
        level.units = "hPa"
        level[:] = pressure

        if not omit_temperature:
            dimensions = (
                ("longitude", "realization", "latitude", "valid_time", "pressure")
                if permuted else
                ("realization", "valid_time", "pressure", "latitude", "longitude")
            )
            temperature = dataset.createVariable(
                "temperature", "f4", dimensions, fill_value=np.float32(9.96921e36))
            temperature.standard_name = "air_temperature"
            temperature.units = "bananas" if bad_temperature_units else "K"
            coordinate = {
                "realization": np.arange(1, member_count + 1, dtype=np.float32),
                "valid_time": times.astype(np.float32),
                "pressure": pressure,
                "latitude": latitude,
                "longitude": longitude,
            }
            mesh = np.meshgrid(*(coordinate[name] for name in dimensions), indexing="ij")
            axes = dict(zip(dimensions, mesh))
            values = (
                axes["realization"] * np.float32(1000.0)
                + axes["valid_time"] * np.float32(100.0)
                + axes["pressure"]
                + axes["latitude"]
                + axes["longitude"] / np.float32(100.0)
            ).astype(np.float32)
            if corrupt_member is not None:
                member_axis = dimensions.index("realization")
                selection = [slice(None)] * values.ndim
                selection[member_axis] = corrupt_member
                values[tuple(selection)] = np.float32(9.96921e36)
            temperature[:] = values

        psfc_dimensions = ("valid_time", "longitude", "realization", "latitude")
        psfc = dataset.createVariable("psfc", "f4", psfc_dimensions)
        psfc.standard_name = "surface_air_pressure"
        psfc.units = "hPa"
        mesh = np.meshgrid(
            times.astype(np.float32), longitude,
            np.arange(1, member_count + 1, dtype=np.float32), latitude,
            indexing="ij",
        )
        psfc[:] = np.float32(1000.0) + mesh[2] + mesh[0] * np.float32(0.1)
    return path


def _temperature_binding():
    return FieldBinding("air_temperature", "temperature")


@pytest.mark.parametrize("member_count", (2, 5))
@pytest.mark.parametrize("cadence_hours", (1, 3))
def test_discovery_uses_actual_member_count_and_cadence(
    tmp_path, member_count, cadence_hours,
):
    path = _write_fixture(
        tmp_path / "20cr.nc", member_count=member_count,
        cadence_hours=cadence_hours)
    discovery = discover_20crv3([path])
    assert discovery.schema == DISCOVERY_SCHEMA
    assert discovery.members == tuple(str(value) for value in range(1, member_count + 1))
    assert discovery.cadence_seconds == cadence_hours * 3600
    assert discovery.product_identity["title"].startswith("Synthetic NOAA")
    assert discovery.longitude_coordinate.periodic is True
    identity = discovery.inputs[0]
    assert identity.byte_count == path.stat().st_size
    assert identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "times,match",
    [
        ([0, 1, 1], "duplicate valid time"),
        ([0, 2, 1], "reversed valid-time gap"),
        ([0, 1, 3], r"irregular valid-time gap .* is 7200s; expected 3600s"),
    ],
)
def test_time_axis_fails_loudly_with_exact_bad_gap(tmp_path, times, match):
    path = _write_fixture(tmp_path / "bad-time.nc", time_values=times)
    with pytest.raises(ValueError, match=match):
        discover_20crv3([path])


def test_requested_cadence_and_endpoint_must_exist(tmp_path):
    discovery = discover_20crv3([
        _write_fixture(tmp_path / "three-hour.nc", cadence_hours=3)
    ])
    with pytest.raises(ValueError, match="source cadence is 10800s; requested 3600s"):
        validate_time_coverage(
            discovery, start=discovery.valid_times[0], end=discovery.valid_times[-1],
            cadence_seconds=3600)
    with pytest.raises(ValueError, match="forecast endpoint .* is absent"):
        validate_time_coverage(
            discovery, start=discovery.valid_times[0], end="2000-01-01T12:00:00Z")


def test_permuted_dimensions_and_axis_direction_normalize_identically(tmp_path):
    canonical = discover_20crv3([
        _write_fixture(tmp_path / "canonical.nc", permuted=False)
    ])
    permuted = discover_20crv3([
        _write_fixture(
            tmp_path / "permuted.nc", permuted=True,
            descending_latitude=True, descending_longitude=True,
            ascending_pressure=True)
    ])
    first = read_member_frame(
        canonical, member="2", valid_time=canonical.valid_times[1],
        bindings=[_temperature_binding()])
    second = read_member_frame(
        permuted, member="2", valid_time=permuted.valid_times[1],
        bindings=[_temperature_binding()])
    np.testing.assert_array_equal(first["air_temperature"], second["air_temperature"])
    assert first["air_temperature"].shape == (4, 3, 4)
    assert first["air_temperature"].dtype == np.float32


@pytest.mark.parametrize("target_count", (7, 13, 80))
def test_target_vertical_count_is_generic_and_source_top_is_gated(
    tmp_path, target_count,
):
    discovery = discover_20crv3([_write_fixture(tmp_path / "vertical.nc")])
    target = np.geomspace(100000.0, 15000.0, target_count)
    assert len(validate_target_pressure_levels(discovery, target)) == target_count
    with pytest.raises(ValueError, match="upper-air extrapolation is forbidden"):
        validate_target_pressure_levels(
            discovery, np.geomspace(100000.0, 5000.0, target_count))


def test_bindings_reject_missing_variable_and_bad_units(tmp_path):
    missing = discover_20crv3([
        _write_fixture(tmp_path / "missing.nc", omit_temperature=True)
    ])
    with pytest.raises(ValueError, match="missing 20CRv3 variables: temperature"):
        validate_bindings(
            missing, [_temperature_binding()], required_fields={"air_temperature"})

    bad_units = discover_20crv3([
        _write_fixture(tmp_path / "units.nc", bad_temperature_units=True)
    ])
    with pytest.raises(ValueError, match="unsupported units 'bananas'"):
        validate_bindings(
            bad_units, [_temperature_binding()], required_fields={"air_temperature"})


def test_member_specific_fill_value_is_isolated(tmp_path):
    discovery = discover_20crv3([
        _write_fixture(tmp_path / "corrupt.nc", member_count=2, corrupt_member=1)
    ])
    good = read_member_frame(
        discovery, member="1", valid_time=discovery.valid_times[0],
        bindings=[_temperature_binding()])
    assert np.isfinite(good["air_temperature"]).all()
    with pytest.raises(ValueError, match="member 2 .* contains a fill value"):
        read_member_frame(
            discovery, member="2", valid_time=discovery.valid_times[0],
            bindings=[_temperature_binding()])


def test_normalized_member_produces_valid_canonical_source_frame(tmp_path):
    discovery = discover_20crv3([
        _write_fixture(tmp_path / "frame.nc", member_count=5)
    ])
    fields = {}
    for name in REQUIRED_3D_FIELDS:
        fields[name] = np.ones((4, 3, 4), dtype=np.float32)
    for name in REQUIRED_SURFACE_FIELDS:
        shape = (2, 3, 4) if name in {
            "soil_temperature", "volumetric_soil_moisture",
        } else (3, 4)
        fields[name] = np.ones(shape, dtype=np.float32)
    header = build_source_frame_header(
        discovery,
        member="5",
        valid_time=discovery.valid_times[0],
        fields=fields,
        bindings=[],
        earth_shape="sphere:6371229m",
        soil_depth_m=(0.1, 1.0),
        initialization_policies={
            name: "explicit_validated_cold_start"
            for name in POLICY_CONTROLLED_FIELDS
        },
    )
    assert header.source_id == "20crv3/member/5"
    assert header.grid.scan_order == "+longitude,+latitude"
    assert header.vertical_coordinates["pressure"].level_values == (
        100000.0, 85000.0, 50000.0, 10000.0,
    )
    pressure = next(field for field in header.fields
                    if field.canonical_name == "air_pressure")
    assert pressure.data_reference.startswith("derived:pressure-coordinate:")


def test_estimate_and_batch_plan_enforce_memory_budget(tmp_path):
    discovery = discover_20crv3([
        _write_fixture(tmp_path / "members.nc", member_count=5)
    ])
    estimate = estimate_member_bytes(
        discovery, [_temperature_binding()], valid_times=discovery.valid_times)
    assert estimate == 3 * 4 * 3 * 4 * 4
    plan = plan_member_batches(
        discovery, members=discovery.members, worker_limit=8,
        memory_budget_bytes=estimate * 2, estimated_member_bytes=estimate)
    assert plan.batch_size == 2
    assert tuple(map(len, plan.batches)) == (2, 2, 1)
    with pytest.raises(ValueError, match="cannot admit one estimated member"):
        plan_member_batches(
            discovery, members=discovery.members, worker_limit=8,
            memory_budget_bytes=estimate - 1, estimated_member_bytes=estimate)


def _deterministic_processor(member: str, member_root: Path) -> None:
    time.sleep((6 - int(member)) * 0.001)
    (member_root / "state.bin").write_bytes(f"member={member}\n".encode("ascii"))


def test_worker_counts_and_completion_order_do_not_change_output_bytes(tmp_path):
    discovery = discover_20crv3([
        _write_fixture(tmp_path / "five.nc", member_count=5)
    ])
    generations = []
    manifests = []
    for workers in (1, 4, 8):
        plan = plan_member_batches(
            discovery, members=reversed(discovery.members), worker_limit=workers,
            memory_budget_bytes=10_000, estimated_member_bytes=100)
        root = tmp_path / f"out-{workers}"
        manifest = stream_members_atomic(
            root, run_name="proof", plan=plan,
            process_member=_deterministic_processor,
            manifest_context={"source_sha256": discovery.inputs[0].sha256})
        pointer = json.loads((root / "proof.json").read_text())
        generations.append(pointer["generation"])
        manifests.append(manifest)
    assert len(set(generations)) == 1
    assert manifests[0] == manifests[1] == manifests[2]
    assert [value["member_id"] for value in manifests[0]["members"]] == [
        "1", "2", "3", "4", "5",
    ]


def test_streaming_does_not_retain_released_member_payloads(tmp_path):
    discovery = discover_20crv3([
        _write_fixture(tmp_path / "bounded.nc", member_count=5)
    ])
    plan = plan_member_batches(
        discovery, members=discovery.members, worker_limit=8,
        memory_budget_bytes=200, estimated_member_bytes=100)
    lock = threading.Lock()
    active = 0
    maximum = 0
    references = []

    def processor(member, member_root):
        nonlocal active, maximum
        payload = np.full((1024,), int(member), dtype=np.float32)
        references.append(weakref.ref(payload))
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.005)
            (member_root / "payload.bin").write_bytes(payload.tobytes())
        finally:
            with lock:
                active -= 1

    stream_members_atomic(
        tmp_path / "bounded-out", run_name="bounded", plan=plan,
        process_member=processor, manifest_context={"case": "bounded"})
    gc.collect()
    assert maximum <= plan.batch_size == 2
    assert all(reference() is None for reference in references)


def test_failed_rewrite_preserves_previous_valid_pointer(tmp_path):
    discovery = discover_20crv3([
        _write_fixture(tmp_path / "failure.nc", member_count=2)
    ])
    plan = plan_member_batches(
        discovery, members=discovery.members, worker_limit=2,
        memory_budget_bytes=200, estimated_member_bytes=100)
    root = tmp_path / "failure-out"
    stream_members_atomic(
        root, run_name="same-time", plan=plan,
        process_member=_deterministic_processor,
        manifest_context={"generation": 1})
    previous = (root / "same-time.json").read_bytes()

    def corrupt(member, member_root):
        if member == "2":
            raise ValueError("synthetic member corruption")
        _deterministic_processor(member, member_root)

    with pytest.raises(EnsembleRunError, match="member corruption") as error:
        stream_members_atomic(
            root, run_name="same-time", plan=plan, process_member=corrupt,
            manifest_context={"generation": 2})
    assert list(error.value.failures) == ["2"]
    assert (root / "same-time.json").read_bytes() == previous
    assert not any((root / ".staging").iterdir())
