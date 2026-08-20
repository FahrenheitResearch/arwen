"""A multi-valid-time mapped source costs ONE valid time, not all of them.

Named breakage, measured on real RRFS bytes (3 km CONUS, 45 pressure
levels, seven hourly valid times, one 300x300 target): a bare default
``gpuwm prep`` peaked at 35.2 GiB of host RSS for two valid times and
67.0 GiB for four -- 15.9 GiB per additional forcing time -- so the
seven-time preparation the source's own cadence asks for needed about
114 GiB and was killed by the OOM reaper on every box smaller than that.
Nothing downstream ever holds two valid times: the initialize loop takes
one, keeps its perimeter frames, and drops it.

These tests hold the reader and the packing to that, at a size small
enough to run everywhere.  The fixture is written by the SHIPPED
frameset writer, so the reader is measured against the real writer
rather than against a hand-rolled file.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import weakref
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpuwm import mapped_engine_bridge as engine_bridge  # noqa: E402
from gpuwm.ingest.soil_contract import (  # noqa: E402
    MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE)
from gpuwm.mapped_composition import MappedSourceBundle  # noqa: E402
from gpuwm.source_frame import (  # noqa: E402
    FieldDescriptor, GridDescriptor, SourceFrameHeader, TimeDescriptor,
    VerticalDescriptor)

#: A fixture valid time big enough that holding N of them is visible in
#: RSS and small enough to run on a laptop: six 3-D fields on a
#: 24x160x160 mesh plus soil and surface, about 30 MiB of float64.
_NZ, _NY, _NX, _NSOIL = 24, 160, 160, 4

_THREE_D = (
    "air_temperature", "specific_humidity", "eastward_wind",
    "northward_wind", "geopotential_height", "air_pressure",
)
_SURFACE = (
    "surface_pressure", "terrain_height", "skin_temperature",
    "air_temperature_2m", "specific_humidity_2m", "eastward_wind_10m",
    "northward_wind_10m", "land_fraction",
)
_SOIL = ("soil_temperature", "volumetric_soil_moisture")

_UNITS = {
    "air_temperature": "K", "specific_humidity": "kg kg-1",
    "eastward_wind": "m s-1", "northward_wind": "m s-1",
    "geopotential_height": "m", "air_pressure": "Pa",
    "surface_pressure": "Pa", "terrain_height": "m",
    "skin_temperature": "K", "air_temperature_2m": "K",
    "specific_humidity_2m": "kg kg-1", "eastward_wind_10m": "m s-1",
    "northward_wind_10m": "m s-1", "land_fraction": "1",
    "soil_temperature": "K", "volumetric_soil_moisture": "m3 m-3",
}
_BASE = {
    "air_temperature": 280.0, "specific_humidity": 0.004,
    "eastward_wind": 3.0, "northward_wind": -2.0,
    "geopotential_height": 500.0, "air_pressure": 0.0,
    "surface_pressure": 98000.0, "terrain_height": 300.0,
    "skin_temperature": 288.0, "air_temperature_2m": 287.0,
    "specific_humidity_2m": 0.008, "eastward_wind_10m": 2.0,
    "northward_wind_10m": -1.0, "land_fraction": 1.0,
    "soil_temperature": 284.0, "volumetric_soil_moisture": 0.25,
}
_POLICY_ZERO = (
    "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio", "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio", "vertical_velocity",
    "snow_water_equivalent", "snow_depth", "sea_ice_fraction",
)

#: One valid time's array bytes, the unit every budget below is stated
#: in.  Computed, never guessed, so resizing the fixture moves the
#: budgets with it.
VALID_TIME_BYTES = 8 * (
    len(_THREE_D) * _NZ * _NY * _NX
    + len(_SOIL) * _NSOIL * _NY * _NX
    + len(_SURFACE) * _NY * _NX
)

_LEVELS = tuple(float(100000.0 - 4000.0 * index) for index in range(_NZ))
_CYCLE = datetime(2026, 8, 17, 0, 0, 0)
_UTC_CYCLE = _CYCLE.isoformat() + "+00:00"


def _values(name: str) -> np.ndarray:
    """Deterministic, finite, physically admissible field values."""

    if name == "air_pressure":
        column = np.asarray(_LEVELS, dtype=np.float64)
        return np.repeat(
            column, _NY * _NX).reshape(_NZ, _NY, _NX)
    if name in _THREE_D:
        ramp = np.linspace(0.0, 1.0, _NZ * _NY * _NX, dtype=np.float64)
        return (_BASE[name] + ramp).reshape(_NZ, _NY, _NX)
    if name in _SOIL:
        ramp = np.linspace(0.0, 0.5, _NSOIL * _NY * _NX, dtype=np.float64)
        return (_BASE[name] + ramp).reshape(_NSOIL, _NY, _NX)
    return np.full((_NY, _NX), _BASE[name], dtype=np.float64)


def _axes(name: str) -> tuple[str, ...]:
    if name in _THREE_D:
        return ("vertical", "y", "x")
    if name in _SOIL:
        return ("soil", "y", "x")
    return ("y", "x")


def _one_frame():
    """One canonical frame, built through the shipped dataclasses."""

    from gpuwm.mapped_source import CanonicalField, MappedSourceFrame

    names = _THREE_D + _SURFACE + _SOIL
    fields = {}
    descriptors = []
    time_descriptor = TimeDescriptor(
        reference_time=_UTC_CYCLE,
        valid_time=_UTC_CYCLE,
        lead_seconds=0,
    )
    for name in names:
        values = _values(name)
        axes = _axes(name)
        fields[name] = CanonicalField(
            name=name,
            units=_UNITS[name],
            axes=axes,
            location=("soil" if name in _SOIL
                      else "mass" if name in _THREE_D else "surface"),
            staggering="none",
            values=values,
            missing_count=0,
            source_references=(f"fixture:{name}",),
        )
        descriptors.append(FieldDescriptor(
            canonical_name=name,
            units=_UNITS[name],
            dimensions=axes,
            grid_location=fields[name].location,
            vertical_coordinate=(
                "pressure" if name in _THREE_D
                else "soil_depth" if name in _SOIL else None),
            time=time_descriptor,
            data_reference=f"fixture:{name}",
            dtype="float64",
            shape=tuple(int(size) for size in values.shape),
            source_field=name,
        ))
    header = SourceFrameHeader(
        source_id="fixture-streaming-source",
        source_cycle=_UTC_CYCLE,
        grid=GridDescriptor(
            projection="regular_latitude_longitude",
            nx=_NX, ny=_NY,
            earth_shape="sphere",
            scan_order="+i+j",
            wind_basis="earth_relative",
            parameters={},
        ),
        vertical_coordinates={
            "pressure": VerticalDescriptor(
                coordinate="pressure", level_count=_NZ,
                level_values=_LEVELS),
            "soil_depth": VerticalDescriptor(
                coordinate="soil_depth", level_count=_NSOIL,
                level_values=(0.05, 0.25, 0.7, 1.5), units="m"),
        },
        fields=tuple(descriptors),
        initialization_policies={
            name: "explicit_zero_with_adapter_validation"
            for name in _POLICY_ZERO
        },
    )
    return MappedSourceFrame(
        valid_time=_CYCLE,
        member=None,
        source_cycle=_CYCLE,
        latitude=np.linspace(30.0, 40.0, _NY),
        longitude=np.linspace(-104.0, -94.0, _NX),
        vertical_kind="pressure",
        vertical_units="Pa",
        vertical_values=np.asarray(_LEVELS, dtype=np.float64),
        fields=fields,
        mapping_sha256="a" * 64,
        input_sha256={"fixture.grib2": "b" * 64},
        grid_fingerprint="fixture-grid",
        header=header,
    )


def _write_fixture_frameset(directory: Path, valid_times: int) -> Path:
    """``valid_times`` frames written by the SHIPPED frameset writer.

    The frames share one set of arrays and differ only in valid time, so
    building the fixture costs one valid time however many are written:
    the point of the file is what READING it costs.
    """

    frame = _one_frame()
    engine_bridge.write_frameset(directory, (
        replace(frame, valid_time=_CYCLE + timedelta(hours=index))
        for index in range(valid_times)
    ))
    return directory


_SOIL_CONTRACT = {
    "count": _NSOIL,
    "thickness_m": [0.1, 0.3, 0.6, 1.0],
    "depth_m": [0.05, 0.25, 0.7, 1.5],
    "missing": {"land": "reject", "water": "reject"},
}


def _bundle(directory: Path, authority: Path) -> MappedSourceBundle:
    frames = engine_bridge.open_frameset(directory)
    return MappedSourceBundle(
        frames=frames,
        mapping_path=authority, mapping_sha256="0" * 64,
        composition_path=authority, composition_sha256="1" * 64,
        input_manifest_path=authority, input_manifest_sha256="2" * 64,
        decoder_paths={"grib2_dump": authority},
        decoder_sha256={"grib2_dump": "5" * 64},
        terrain_data_paths=(authority,), terrain_data_sha256=("3" * 64,),
        terrain_provenance_path=authority, terrain_provenance_sha256="4" * 64,
        soil_layer_contract=_SOIL_CONTRACT,
        alignment_receipt={"status": "PASS"},
    )


# --------------------------------------------------------------------
# 1. The reader holds one valid time.
# --------------------------------------------------------------------

def test_a_frameset_holds_one_valid_time_while_it_is_walked(tmp_path):
    """Reading frame N+1 releases frame N.

    Named breakage: a reader that returns every frame at once makes the
    caller's peak scale with the forecast length of the source, which is
    what the RRFS seven-time preparation died of.  This is the invariant
    that fails first if the reader goes back to materializing the set.
    """

    directory = _write_fixture_frameset(tmp_path / "frameset", 5)
    frames = engine_bridge.open_frameset(directory)
    assert len(frames) == 5

    alive = []
    for index in range(len(frames)):
        frame = frames[index]
        assert frame.valid_time == _CYCLE + timedelta(hours=index)
        alive.append(weakref.ref(frame))
        del frame
        gc.collect()
        released = [reference for reference in alive[:-1]
                    if reference() is not None]
        assert not released, (
            f"{len(released)} earlier valid time(s) were still resident "
            f"while frame {index} was being read")


def test_the_frameset_answers_its_scalars_without_reading_an_array(tmp_path):
    """Valid times, members and digests come from the document.

    Named breakage: sealing a preparation's receipt used to walk every
    frame for a header hash and a terrain digest, which pulled the whole
    forcing series back into memory to produce a few hundred bytes of
    JSON.
    """

    directory = _write_fixture_frameset(tmp_path / "frameset", 4)
    frames = engine_bridge.open_frameset(directory)

    assert frames.valid_times == tuple(
        _CYCLE + timedelta(hours=index) for index in range(4))
    assert frames.members == (None, None, None, None)
    assert set(frames.mapping_sha256s) == {"a" * 64}
    assert frames.field_count(0) == len(_THREE_D) + len(_SURFACE) + len(_SOIL)
    assert "terrain_height" in frames.field_names(2)
    # Same terrain in every frame: the fixture shares one array, and the
    # document digest is what the bundle's invariant reads.
    assert len({frames.field_digest(index, "terrain_height")
                for index in range(4)}) == 1
    assert frames._cached_frame is None, \
        "the document questions must not materialize a valid time"


def test_a_streamed_frame_carries_the_bytes_the_eager_read_carries(tmp_path):
    """Streaming changes residency, not a single number.

    Named breakage: a reader rewritten for memory that returned even one
    differently-rounded array would move every downstream product, and
    the whole point of this change is that it moves none.
    """

    directory = _write_fixture_frameset(tmp_path / "frameset", 3)
    eager = engine_bridge.read_frameset(directory)
    streamed = engine_bridge.open_frameset(directory)

    assert len(eager) == len(streamed) == 3
    for index, expected in enumerate(eager):
        actual = streamed[index]
        assert actual.valid_time == expected.valid_time
        assert set(actual.fields) == set(expected.fields)
        for name, field in expected.fields.items():
            np.testing.assert_array_equal(
                actual.fields[name].values, field.values)
            assert actual.fields[name].values.dtype == field.values.dtype


def test_a_perturbed_array_is_refused_when_its_frame_is_read(tmp_path):
    """A flipped bit still refuses, still by field name.

    Named breakage: an array that is not the numbers its manifest
    describes would initialize a run from values no decoder produced.
    Reading one frame at a time must not weaken that -- every frame the
    route consumes is verified when it is read, before any of its
    numbers reach a preparation.
    """

    directory = _write_fixture_frameset(tmp_path / "frameset", 2)
    stream = directory / engine_bridge.FRAMES_STREAM
    raw = bytearray(stream.read_bytes())
    raw[-1] ^= 0xFF
    stream.write_bytes(bytes(raw))

    frames = engine_bridge.open_frameset(directory)
    frames[0]  # the untouched valid time still reads
    with pytest.raises(ValueError, match=r"frame 1 field '.+' hashes to"):
        frames[1]


# --------------------------------------------------------------------
# 2. The packing holds one valid time.
# --------------------------------------------------------------------

def test_regular_snapshots_packs_one_valid_time_at_a_time(tmp_path):
    """Snapshot N+1 releases snapshot N, zero hydrometeors included.

    Named breakage: packing every valid time up front allocated five
    zero-filled hydrometeor arrays PER TIME the size of the source
    pressure field -- 3.2 GiB per valid time on a 3 km CONUS source --
    and held them all from the first time to the last.
    """

    authority = tmp_path / "authority.json"
    authority.write_text("{}", encoding="utf-8")
    directory = _write_fixture_frameset(tmp_path / "frameset", 4)
    snapshots = _bundle(directory, authority).regular_snapshots()

    assert len(snapshots) == 4
    alive = []
    for index in range(len(snapshots)):
        snapshot = snapshots[index]
        assert snapshot.fields["QC"].shape == (_NZ, _NY, _NX)
        np.testing.assert_array_equal(
            snapshot.fields["QG"], np.zeros((_NZ, _NY, _NX)))
        assert MAPPED_SOIL_TEMPERATURE in snapshot.fields
        assert MAPPED_SOIL_MOISTURE in snapshot.fields
        alive.append(weakref.ref(snapshot))
        del snapshot
        gc.collect()
        released = [reference for reference in alive[:-1]
                    if reference() is not None]
        assert not released, (
            f"{len(released)} earlier snapshot(s) were still resident "
            f"while snapshot {index} was being packed")


def test_the_forcing_series_sorts_and_reads_times_without_packing(tmp_path):
    """Valid-time ORDER costs no arrays.

    Named breakage: the mapped route used to sort a materialized tuple
    of snapshots to put them in valid-time order, so ordering N times
    cost N times' arrays before the first was interpolated.
    """

    authority = tmp_path / "authority.json"
    authority.write_text("{}", encoding="utf-8")
    directory = _write_fixture_frameset(tmp_path / "frameset", 3)
    ordered = _bundle(directory, authority).regular_snapshots(
    ).sorted_by_valid_time()

    assert ordered.valid_times == tuple(
        _CYCLE + timedelta(hours=index) for index in range(3))
    assert ordered._cached_snapshot is None
    # The vertical-coverage question is answered from pressure alone.
    np.testing.assert_allclose(
        ordered.source_pressure_hpa(0),
        np.asarray(_LEVELS, dtype=np.float64) / 100.0)
    assert ordered._cached_snapshot is None


# --------------------------------------------------------------------
# 3. The measured budget.
# --------------------------------------------------------------------

def _peak_rss_bytes() -> int:
    """This process's peak resident set, or 0 where the OS will not say."""

    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        # `K32GetProcessMemoryInfo` in kernel32, not the psapi.dll
        # spelling: psapi forwards it and the forwarded entry answers
        # FALSE on this box, which reads as "no peak" and skips the one
        # test in this file that measures the defect.
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not query(kernel32.GetCurrentProcess(),
                     ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.PeakWorkingSetSize)
    return 0


def _measure(valid_times: int, directory: Path, *, mode: str) -> dict:
    """Walk a fixture preparation's forcing series; report the peak.

    ``mode="streamed"`` is the shipped route.  ``mode="all-at-once"``
    is what it replaced -- read every frame, pack every snapshot, then
    walk -- and it is here so the measurement can be checked against a
    known answer in BOTH directions: an instrument that cannot see the
    defect it is meant to gate is not a gate.
    """

    authority = directory / "authority.json"
    authority.write_text("{}", encoding="utf-8")
    frameset = _write_fixture_frameset(directory / "frameset", valid_times)
    # Whatever the fixture cost to write is not what is being measured.
    gc.collect()
    before = _peak_rss_bytes()
    total = 0.0
    if mode == "all-at-once":
        frames = engine_bridge.read_frameset(frameset)
        snapshots = tuple(_bundle_from_frames(frames, authority)
                          .regular_snapshots())
        snapshots = tuple(snapshots[index] for index in range(len(snapshots)))
        for snapshot in snapshots:
            total += float(snapshot.fields["PRES"][0, 0, 0])
            total += float(snapshot.fields["QC"].sum())
    else:
        bundle = _bundle(frameset, authority)
        snapshots = bundle.regular_snapshots().sorted_by_valid_time()
        for index in range(len(snapshots)):
            snapshot = snapshots[index]
            total += float(snapshot.fields["PRES"][0, 0, 0])
            total += float(snapshot.fields["QC"].sum())
        bundle.close()
    return {
        "mode": mode,
        "valid_times": valid_times,
        "peak_before": before,
        "peak_after": _peak_rss_bytes(),
        "checksum": total,
    }


def _bundle_from_frames(frames, authority: Path) -> MappedSourceBundle:
    return MappedSourceBundle(
        frames=frames,
        mapping_path=authority, mapping_sha256="0" * 64,
        composition_path=authority, composition_sha256="1" * 64,
        input_manifest_path=authority, input_manifest_sha256="2" * 64,
        decoder_paths={"grib2_dump": authority},
        decoder_sha256={"grib2_dump": "5" * 64},
        terrain_data_paths=(authority,), terrain_data_sha256=("3" * 64,),
        terrain_provenance_path=authority, terrain_provenance_sha256="4" * 64,
        soil_layer_contract=_SOIL_CONTRACT,
        alignment_receipt={"status": "PASS"},
    )


def _measured_peak(mode: str, valid_times: int, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()),
         "--measure", mode, str(valid_times), str(directory)],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(_peak_rss_bytes() == 0,
                    reason="this OS does not report a peak resident set")
def test_walking_more_valid_times_does_not_raise_the_peak(tmp_path):
    """The measured ceiling: eight valid times cost what two cost.

    Named breakage: this is the RRFS defect at fixture scale.  Reading
    and packing N valid times up front holds N of them, so the peak
    rises by one valid time's arrays for every extra forcing hour --
    15.9 GiB per hour on the real 3 km CONUS source, which is what took
    a seven-time preparation past 100 GiB.

    Both arms are measured with the same instrument in fresh processes.
    The all-at-once arm is the control: it must show the growth, or the
    budget on the streamed arm is measuring nothing.  The streamed
    budget is deliberately loose (two fixture valid times) so it gates
    the SCALING and not the noise.
    """

    one = VALID_TIME_BYTES / 1048576.0
    control = {
        count: _measured_peak("all-at-once", count, tmp_path / f"all{count}")
        for count in (2, 8)
    }
    control_growth = (
        control[8]["peak_after"] - control[2]["peak_after"]) / 1048576.0
    assert control_growth > 3 * one, (
        f"the control arm grew only {control_growth:.1f} MiB across six "
        f"extra valid times ({one:.1f} MiB each), so this measurement "
        "cannot see the defect it is meant to gate")

    streamed = {
        count: _measured_peak("streamed", count, tmp_path / f"stream{count}")
        for count in (2, 8)
    }
    assert streamed[8]["checksum"] == control[8]["checksum"], \
        "the two arms did not read the same numbers"
    growth = (
        streamed[8]["peak_after"] - streamed[2]["peak_after"]) / 1048576.0
    assert growth < 2 * one, (
        f"walking 8 valid times peaked {growth:.1f} MiB above walking 2; "
        f"one fixture valid time is {one:.1f} MiB, so the peak is still "
        f"scaling with the number of forcing times "
        f"(2 times: {streamed[2]['peak_after'] / 1048576.0:.1f} MiB, "
        f"8 times: {streamed[8]['peak_after'] / 1048576.0:.1f} MiB; "
        f"control grew {control_growth:.1f} MiB)")


if __name__ == "__main__":  # the measured arms, each in its own process
    if sys.argv[1] == "--measure":
        target = Path(sys.argv[4])
        target.mkdir(parents=True, exist_ok=True)
        print(json.dumps(
            _measure(int(sys.argv[3]), target, mode=sys.argv[2])))
