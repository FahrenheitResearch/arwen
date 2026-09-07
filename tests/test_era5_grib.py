"""Phase 3 Task 6: ERA5 GRIB1 decode and Vtable-exact field mapping.

Authority is the bundle ``Vtable.ERA5_CDO`` and its combined/per-time GRIB1
files.  The bundled CDO-produced NetCDF files are an independent value and
orientation oracle for the all-Rust bridge.
"""
import gc
import json
import os
import subprocess
import sys
import weakref
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from gpuwm.ingest.grib import (
    Era5Snapshot,
    build_rust_bridge,
    decode_era5_grib,
    parse_vtable,
    _decode_bridge_partials,
    _load_bridge_partials,
    _valid_time,
)


BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
ERA5 = BUNDLE / "era5_grib"
COMBINED = ERA5 / "era5_19740403.grb"
VTABLE = ERA5 / "Vtable.ERA5_CDO"
NC_DIR = ERA5 / "nc"

requires_bundle = pytest.mark.skipif(
    not COMBINED.is_file() or not VTABLE.is_file(),
    reason="WRF_1974_MP55 ERA5 reference bundle not present",
)

# The source contains every parameter-coded Vtable field below except HGT,
# SOILGEO, SOILHGT, PMSL, and SEAICE.  Blank-param rows (derived 2-m RH and
# SNOW) are not decode inventory.  The CDO-only parameter 2 ``utc_date`` is
# deliberately excluded because it is not in the Vtable.
EXPECTED_FIELDS = frozenset(
    {
        "Z", "T", "U", "V", "RH",
        "U10", "V10", "T2", "D2", "LANDSEA", "PSFC",
        "SKINTEMP", "SST", "SNOW_EC",
        "ST000007", "ST007028", "ST028100", "ST100289",
        "SM000007", "SM007028", "SM028100", "SM100289",
    }
)

ORACLE_NAMES = {
    "Z": "Z",
    "T": "T",
    "U": "U",
    "V": "V",
    "RH": "R",
    "PSFC": "SP",
    "SNOW_EC": "SD",
    "U10": "VAR_10U",
    "V10": "VAR_10V",
    "T2": "VAR_2T",
    "D2": "VAR_2D",
    "ST000007": "STL1",
    "ST007028": "STL2",
    "ST028100": "STL3",
    "ST100289": "STL4",
    "SM000007": "SWVL1",
    "SM007028": "SWVL2",
    "SM028100": "SWVL3",
    "SM100289": "SWVL4",
    "SKINTEMP": "SKT",
    "SST": "SSTK",
    "LANDSEA": "LSM",
}

# CDO's NetCDF writer applies a field-wide float32 scale/offset transform, so
# decoding the original GRIB simple packing is slightly more accurate than the
# oracle arrays.  These bounds are at most a few float32 quantization steps and
# many orders tighter than downstream metgrid interpolation gates.
ORACLE_ATOL = {
    "Z": 0.05,
    "PSFC": 0.05,
    "T": 2.5e-4,
    "T2": 2.5e-4,
    "D2": 2.5e-4,
    "ST000007": 2.5e-4,
    "ST007028": 2.5e-4,
    "ST028100": 2.5e-4,
    "ST100289": 2.5e-4,
    "SKINTEMP": 2.5e-4,
    "SST": 2.5e-4,
    "RH": 1.0e-5,
    "SNOW_EC": 1.0e-5,
    "SM000007": 4.0e-8,
    "SM007028": 4.0e-8,
    "SM028100": 4.0e-8,
    "SM100289": 4.0e-8,
    "LANDSEA": 2.0e-5,
    "U": 1.0e-7,
    "V": 1.0e-7,
    "U10": 1.0e-7,
    "V10": 1.0e-7,
}


def test_parse_vtable_preserves_grib1_keys_and_names(tmp_path):
    table = tmp_path / "Vtable"
    table.write_text(
        "GRIB1| Level| From | To | metgrid | metgrid | Description |GRIB2|\n"
        "Param| Type |Level1|Level2| Name   | Units   |             |Discp|\n"
        "-----+------+------+------+--------+---------+-------------+-----+\n"
        " 129 | 100  |   *  |      | GEOPT  | m2 s-2  | geopotential|  0  |\n"
        " 165 |  1   |   0  |      | UU     | m s-1   | 10 m U     |  0  |\n"
        "     |  1   |   0  |      | RH     | %       | derived     |  0  |\n"
        "-----+------+------+------+--------+---------+-------------+-----+\n",
        encoding="utf-8",
    )
    entries = parse_vtable(table)
    assert len(entries) == 3
    assert (entries[0].parameter, entries[0].level_type) == (129, 100)
    assert (entries[0].level1, entries[0].name, entries[0].units) == (
        "*", "GEOPT", "m2 s-2"
    )
    assert entries[1].name == "UU"
    assert entries[2].parameter is None and entries[2].name == "RH"


@requires_bundle
def test_bundle_vtable_contains_required_parameter_mappings():
    by_key = {
        (e.parameter, e.level_type): e.name
        for e in parse_vtable(VTABLE)
        if e.parameter is not None
    }
    expected = {
        (129, 100): "GEOPT", (130, 100): "TT", (131, 100): "UU",
        (132, 100): "VV", (157, 100): "RH", (165, 1): "UU",
        (166, 1): "VV", (167, 1): "TT", (168, 1): "DEWPT",
        (172, 1): "LANDSEA", (134, 1): "PSFC", (235, 1): "SKINTEMP",
        (34, 1): "SST", (141, 1): "SNOW_EC", (139, 1): "ST000007",
        (170, 1): "ST007028", (183, 1): "ST028100",
        (236, 1): "ST100289", (39, 1): "SM000007",
        (40, 1): "SM007028", (41, 1): "SM028100",
        (42, 1): "SM100289",
    }
    assert expected.items() <= by_key.items()


def _synthetic_snapshot():
    levels = np.array([1000.0, 850.0], dtype=np.float64)
    latitude = np.array([25.0, 25.25], dtype=np.float64)
    longitude = np.array([250.0, 250.25, 250.5], dtype=np.float64)
    return Era5Snapshot(
        valid_time=datetime(1974, 4, 3, 12),
        levels_hpa=levels,
        latitude=latitude,
        longitude=longitude,
        fields={
            "T": np.arange(12, dtype=np.float64).reshape(2, 2, 3) + 250.0,
            "PSFC": np.arange(6, dtype=np.float64).reshape(2, 3) + 90000.0,
        },
    )


def test_snapshot_npz_round_trip_without_pickle(tmp_path):
    original = _synthetic_snapshot()
    path = tmp_path / "snapshot.npz"
    original.save_npz(path)
    restored = Era5Snapshot.load_npz(path)
    assert restored.valid_time == original.valid_time
    np.testing.assert_array_equal(restored.levels_hpa, original.levels_hpa)
    np.testing.assert_array_equal(restored.latitude, original.latitude)
    np.testing.assert_array_equal(restored.longitude, original.longitude)
    assert restored.fields.keys() == original.fields.keys()
    for name in original.fields:
        np.testing.assert_array_equal(restored.fields[name], original.fields[name])
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "valid_time", "levels_hpa", "latitude", "longitude",
            "field_names", "field__T", "field__PSFC",
        }


def test_snapshot_rejects_inconsistent_shapes_and_non_float64():
    snap = _synthetic_snapshot()
    with pytest.raises(ValueError, match="T.*shape"):
        Era5Snapshot(
            snap.valid_time, snap.levels_hpa, snap.latitude, snap.longitude,
            {"T": np.zeros((2, 3, 2), dtype=np.float64)},
        )
    with pytest.raises(TypeError, match="float64"):
        Era5Snapshot(
            snap.valid_time, snap.levels_hpa, snap.latitude, snap.longitude,
            {"PSFC": np.zeros((2, 3), dtype=np.float32)},
        )


def _synthetic_partial(source, valid_time):
    """One decoded product's contribution at one valid time.

    The shape ``_load_bridge_partials`` hands the merge: pressure levels
    already stacked, surface fields beside them, coordinates per file.
    """

    from gpuwm.ingest.grib import _PartialSnapshot

    return _PartialSnapshot(
        source=source,
        valid_time=valid_time,
        levels_hpa=(1000, 850),
        latitude=np.array([25.0, 25.25], dtype=np.float64),
        longitude=np.array([250.0, 250.25, 250.5], dtype=np.float64),
        fields=MappingProxyType({
            "T": np.arange(12, dtype=np.float64).reshape(2, 2, 3) + 250.0,
            "PSFC": np.arange(6, dtype=np.float64).reshape(2, 3) + 90000.0,
        }),
    )


def test_one_forcing_product_is_cached_twice_and_clearing_releases_both(
        tmp_path, monkeypatch):
    """The decode's host residency, and the release that ends it.

    A run reaches this module TWICE for one product: the input catalog
    decodes under its own time discovery (``valid_times=None``) and the
    runtime decodes under the catalog's selection.  Those are different
    cache keys, so the merged cache holds two DISJOINT frozen copies of
    the same bytes -- every :class:`Era5Snapshot` field is copied in
    ``__post_init__`` -- on top of the partials both were merged from,
    and until ``clear_forcing_caches`` there was nothing in the package
    that dropped any of it: a global 0.25 deg product stayed resident for
    the life of the worker at three copies of 204 fields per valid time.

    Probed by weak reference rather than by ``cache_info``, because the
    question is what is still ALIVE.  Only the Rust bridge subprocess is
    stubbed; ``cached_era5_forcing`` and the merge are the real ones.
    """
    from gpuwm.ingest import grib

    grib_path = tmp_path / "era5.grb"
    grib_path.write_bytes(b"stub")
    vtable_path = tmp_path / "Vtable"
    vtable_path.write_text("stub\n", encoding="utf-8")
    bridge_path = tmp_path / "bridge"
    bridge_path.write_text("stub\n", encoding="utf-8")
    times = tuple(datetime(1974, 4, 3, 12) + timedelta(hours=6 * step)
                  for step in range(3))
    probes = {}

    def bridge_partials(path, entries, executable):
        partials = tuple(_synthetic_partial(path, value) for value in times)
        probes["partials"] = weakref.ref(partials[0].fields["T"])
        return partials

    monkeypatch.setattr(grib, "parse_vtable", lambda path: ())
    monkeypatch.setattr(grib, "_decode_bridge_partials", bridge_partials)
    try:
        discovered = grib.cached_era5_forcing(
            [grib_path], vtable_path, bridge=bridge_path,
            content_sha256=["ab" * 32])
        selected = grib.cached_era5_forcing(
            [grib_path], vtable_path, bridge=bridge_path,
            content_sha256=["ab" * 32], valid_times=times)
        assert discovered is not selected
        first = discovered.snapshots[0].fields["T"]
        second = selected.snapshots[0].fields["T"]
        assert not np.shares_memory(first, second)
        np.testing.assert_array_equal(first, second)
        probes["discovered"] = weakref.ref(first)
        probes["selected"] = weakref.ref(second)
        del discovered, selected, first, second
        gc.collect()
        assert all(probe() is not None for probe in probes.values()), (
            "the caches are what hold the decode; if this fails the test "
            "no longer measures a release")
        grib.clear_forcing_caches()
        gc.collect()
        assert not any(probe() is not None for probe in probes.values()), (
            "clear_forcing_caches left the decode resident")
    finally:
        grib.clear_forcing_caches()


# --- Bridge-dump gates -------------------------------------------------
#
# These build the ``gpuwm GRIB1 dump format version 1`` the Rust bridge emits
# (``tools/grib1_bridge/src/main.rs``) directly, so the decoder's own gates can
# be fed a definitely-wrong dump without a GRIB fixture.

DUMP_SHAPE = (2, 3)


def _bridge_message(**overrides):
    message = {
        "offset_values": 0, "count": DUMP_SHAPE[0] * DUMP_SHAPE[1],
        "parameter": 167, "level_type": 1, "level": 0,
        "table_version": 128, "center": 98,
        "nx": DUMP_SHAPE[1], "ny": DUMP_SHAPE[0], "scan_mode": 0,
        "year": 1974, "month": 4, "day": 3, "hour": 12, "minute": 0,
        "time_unit": 1, "p1": 0, "p2": 0,
        "time_range_indicator": 0, "has_bitmap": False,
    }
    message.update(overrides)
    return message


def _write_bridge_dump(directory, messages, values):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1, "edition": 1, "dtype": "<f8",
        "shape": list(DUMP_SHAPE),
        "latitude": [25.0 + 0.25 * j for j in range(DUMP_SHAPE[0])],
        "longitude": [250.0 + 0.25 * i for i in range(DUMP_SHAPE[1])],
        "messages": list(messages),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8")
    np.asarray(values, dtype="<f8").tofile(directory / "values.f64")
    return directory


def _bridge_vtable(path):
    path.write_text(
        "GRIB1| Level| From | To | metgrid | metgrid | Description |GRIB2|\n"
        "Param| Type |Level1|Level2| Name   | Units   |             |Discp|\n"
        "-----+------+------+------+--------+---------+-------------+-----+\n"
        " 130 | 100  |   *  |      | TT     | K       | temperature |  0  |\n"
        " 167 |  1   |   0  |      | TT     | K       | 2 m temp    |  0  |\n"
        " 134 |  1   |   0  |      | PSFC   | Pa      | surface p   |  0  |\n"
        "-----+------+------+------+--------+---------+-------------+-----+\n",
        encoding="utf-8",
    )
    return parse_vtable(path)


def test_valid_time_reads_a_16_bit_p1_under_time_range_10():
    # WMO Table 5 indicator 10 spans PDS octets 19-20 as one 16-bit P1, so a
    # 24-hour lead in hours is p1=0x00, p2=0x18.  Reading octet 19 alone
    # timestamps the record at the reference hour, 24 h early.
    message = _bridge_message(time_range_indicator=10, p1=0x00, p2=0x18)
    assert _valid_time(message) == datetime(1974, 4, 4, 12)
    instantaneous = _bridge_message(time_range_indicator=0, p1=6)
    assert _valid_time(instantaneous) == datetime(1974, 4, 3, 18)


def test_valid_time_refuses_an_interval_time_range_indicator():
    # Indicators 2/3/4/5 are valid at reference + P2 and carry interval
    # quantities; binding one to reference + P1 files an accumulation as an
    # instantaneous analysis.
    for indicator in (2, 3, 4, 5):
        with pytest.raises(ValueError, match="time range indicator"):
            _valid_time(_bridge_message(
                time_range_indicator=indicator, p1=0, p2=6))


def test_bridge_dump_with_fewer_messages_than_envelopes_is_refused(
        tmp_path, monkeypatch):
    # Negative control for the silently dropped GRIB1 message: the bridge
    # skips a message whose sections do not parse and still exits 0, so a dump
    # that is short by one message must not read as a complete decode.
    entries = _bridge_vtable(tmp_path / "Vtable")
    prepared = _write_bridge_dump(
        tmp_path / "prepared", [_bridge_message()], np.full(6, 288.0))
    stub = tmp_path / "stub_bridge.py"
    stub.write_text(
        "import shutil, sys\n"
        f"shutil.copytree({str(prepared)!r}, sys.argv[2])\n",
        encoding="utf-8",
    )
    # A real child process still writes the incomplete dump. Launch its
    # Python script explicitly: Windows CreateProcess cannot execute a
    # shebang script, and a WinError 193 would never test the refusal.
    run_process = subprocess.run

    def run_python_stub(command, **kwargs):
        assert command[0] == str(stub)
        return run_process([sys.executable, *command], **kwargs)

    monkeypatch.setattr(subprocess, "run", run_python_stub)
    envelope = b"GRIB" + (12).to_bytes(3, "big") + b"\x01" + b"7777"
    grib = tmp_path / "two_messages.grb"
    grib.write_bytes(envelope * 2)

    with pytest.raises(ValueError, match="dropped by the decoder"):
        _decode_bridge_partials(grib, entries, stub)


def test_bridge_message_grid_must_match_the_primary_axes(tmp_path):
    # A transposed message carries the same point count as the primary grid,
    # so only a shape comparison can catch it before the reshape.
    entries = _bridge_vtable(tmp_path / "Vtable")
    dump = _write_bridge_dump(
        tmp_path / "dump",
        [_bridge_message(nx=DUMP_SHAPE[0], ny=DUMP_SHAPE[1])],
        np.full(6, 288.0),
    )
    with pytest.raises(ValueError, match="not the primary grid"):
        _load_bridge_partials(dump, entries, tmp_path / "source.grb")


def test_bridge_messages_must_agree_on_their_scanning_mode(tmp_path):
    # Two messages of the same shape whose scan modes disagree cannot share
    # one latitude/longitude axis pair: one of them is stored the other way up.
    entries = _bridge_vtable(tmp_path / "Vtable")
    dump = _write_bridge_dump(
        tmp_path / "dump",
        [
            _bridge_message(parameter=167, scan_mode=0),
            _bridge_message(parameter=134, scan_mode=0x40, offset_values=6),
        ],
        np.concatenate([np.full(6, 288.0), np.full(6, 98000.0)]),
    )
    with pytest.raises(ValueError, match="disagree on GRIB1 scanning mode"):
        _load_bridge_partials(dump, entries, tmp_path / "source.grb")


def test_non_finite_values_without_a_bitmap_are_refused(tmp_path):
    # No bitmap means every grid point was coded, so a non-finite cell has no
    # provenance record and must not reach the initial condition.
    entries = _bridge_vtable(tmp_path / "Vtable")
    values = np.full(6, 288.0)
    values[2] = np.nan
    dump = _write_bridge_dump(
        tmp_path / "dump", [_bridge_message(has_bitmap=False)], values)
    with pytest.raises(ValueError, match="non-finite values"):
        _load_bridge_partials(dump, entries, tmp_path / "source.grb")


def test_a_bitmapped_pressure_field_records_its_missing_provenance(tmp_path):
    # The surface branch recorded bitmap provenance and the pressure branch
    # did not, so ``Era5DecodeResult``'s mask-equals-non-finite contract was
    # structurally unreachable for half the inventory.
    entries = _bridge_vtable(tmp_path / "Vtable")
    values = np.full(6, 250.0)
    values[4] = np.nan
    dump = _write_bridge_dump(
        tmp_path / "dump",
        [_bridge_message(parameter=130, level_type=100, level=850,
                         has_bitmap=True)],
        values,
    )
    (partial,) = _load_bridge_partials(
        dump, entries, tmp_path / "source.grb")
    mask = partial.bitmap_missing["T"]
    assert mask.shape == (1, DUMP_SHAPE[0], DUMP_SHAPE[1])
    np.testing.assert_array_equal(mask, ~np.isfinite(partial.fields["T"]))
    assert mask.sum() == 1


@lru_cache(maxsize=1)
def _decode_bundle_cached():
    bridge = build_rust_bridge(release=True)
    return decode_era5_grib(COMBINED, VTABLE, bridge=bridge)


@requires_bundle
def test_rust_bridge_decodes_grib1_and_discovers_all_times():
    snapshots = _decode_bundle_cached()
    assert tuple(s.valid_time for s in snapshots) == (
        datetime(1974, 4, 3, 12),
        datetime(1974, 4, 3, 18),
        datetime(1974, 4, 4, 0),
    )
    for snapshot in snapshots:
        assert snapshot.fields.keys() == EXPECTED_FIELDS


@requires_bundle
def test_bundle_grid_levels_shapes_and_ranges():
    snapshots = _decode_bundle_cached()
    expected_levels = np.array(
        [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175,
         200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
         750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000],
        dtype=np.float64,
    )
    for snapshot in snapshots:
        np.testing.assert_array_equal(snapshot.levels_hpa, expected_levels)
        np.testing.assert_allclose(snapshot.latitude, np.arange(25, 55.01, 0.25),
                                   rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(snapshot.longitude, np.arange(250, 300.01, 0.25),
                                   rtol=0.0, atol=1e-12)
        for name in ("Z", "T", "U", "V", "RH"):
            assert snapshot.fields[name].shape == (37, 121, 201)
        for name in EXPECTED_FIELDS - {"Z", "T", "U", "V", "RH"}:
            assert snapshot.fields[name].shape == (121, 201)
        assert 180.0 <= np.nanmin(snapshot.fields["T"])
        assert np.nanmax(snapshot.fields["T"]) <= 330.0
        assert 60000.0 <= np.nanmin(snapshot.fields["PSFC"])
        assert np.nanmax(snapshot.fields["PSFC"]) <= 110000.0
        assert np.nanmin(snapshot.fields["LANDSEA"]) >= 0.0
        assert np.nanmax(snapshot.fields["LANDSEA"]) <= 1.0


def _oracle(path):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path) as ds:
        out = {}
        for canonical, nc_name in ORACLE_NAMES.items():
            values = np.ma.asarray(ds.variables[nc_name][:]).squeeze(axis=0)
            out[canonical] = np.asarray(
                np.ma.filled(values, np.nan), dtype=np.float32
            )
        return out


@requires_bundle
@pytest.mark.parametrize(
    "index,name",
    [(0, "era5_19740403_12.nc"), (1, "era5_19740403_18.nc"),
     (2, "era5_19740404_00.nc")],
)
def test_rust_decode_matches_cdo_netcdf_oracle_within_quantization(index, name):
    snapshot = _decode_bundle_cached()[index]
    oracle = _oracle(NC_DIR / name)
    for field, expected in oracle.items():
        actual = snapshot.fields[field]
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected),
                                      err_msg=f"{field} missing mask")
        np.testing.assert_allclose(actual, expected, rtol=0.0,
                                   atol=ORACLE_ATOL[field], equal_nan=True,
                                   err_msg=field)


@requires_bundle
def test_real_snapshot_npz_round_trip(tmp_path):
    original = _decode_bundle_cached()[0]
    path = tmp_path / "era5_19740403_12.npz"
    original.save_npz(path)
    restored = Era5Snapshot.load_npz(path)
    assert restored.valid_time == original.valid_time
    assert restored.fields.keys() == original.fields.keys()
    for name in original.fields:
        np.testing.assert_array_equal(restored.fields[name], original.fields[name])


@requires_bundle
@pytest.mark.parametrize(
    "index,name",
    [(0, "era5_19740403_12.grb"), (1, "era5_19740403_18.grb"),
     (2, "era5_19740404_00.grb")],
)
def test_per_time_grib_files_match_combined_inventory_and_values(index, name):
    bridge = build_rust_bridge(release=True)
    (single,) = decode_era5_grib(ERA5 / "grb" / name, VTABLE, bridge=bridge)
    combined = _decode_bundle_cached()[index]
    assert single.valid_time == combined.valid_time
    assert single.fields.keys() == EXPECTED_FIELDS
    for field in EXPECTED_FIELDS:
        np.testing.assert_array_equal(single.fields[field], combined.fields[field])
