from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from gpuwm.mapped_composition import load_composition
from gpuwm.mapped_source import _DecodedCollection, _DirectValue, load_mapping
from gpuwm.source_authorities import (
    twentycrv3_authorities,
    twentycrv3_authority_sha256,
)
from gpuwm.twentycrv3_direct import (
    _bind_implicit_member,
    _expected_inventory,
    _manifest,
    discover_20crv3_grib2,
    write_20crv3_manifest,
)


ROOT = Path(__file__).parents[1]
MAPPING = (
    ROOT / "gpuwm" / "authorities" /
    "rw-wps-20crv3-member-grib2.mapping.json"
)
COMPOSITION = (
    ROOT / "gpuwm" / "authorities" /
    "rw-wps-20crv3-member-grib2.composition.json"
)


def test_exact_authorities_are_resolved_from_the_installable_package() -> None:
    paths = twentycrv3_authorities()
    digests = twentycrv3_authority_sha256()

    assert set(paths) == {"mapping", "composition", "provenance"}
    assert all(path.parent.name == "authorities" for path in paths.values())
    assert {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in paths.items()
    } == dict(digests)


def _write_series(
    root: Path,
    *,
    rows: tuple[tuple[str, str, str], ...] = (
        ("072", "1932032100", "pl"),
        ("072", "1932032100", "sfc"),
        ("072", "1932032103", "pl"),
        ("072", "1932032103", "sfc"),
    ),
) -> dict[tuple[str, str, str], Path]:
    paths = {}
    for member, stamp, role in rows:
        directory = root / ("PRESSURE" if role == "pl" else "SURFACE")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"mem{member}_{stamp}_{role}.grb2"
        path.write_bytes(f"fixture:{member}:{stamp}:{role}".encode())
        paths[(member, stamp, role)] = path
    return paths


def test_mapping_loads_with_exact_private_sample_contract() -> None:
    mapping = load_mapping(MAPPING)

    assert mapping["name"] == "noaa-20crv3-every-member-grib2-native"
    assert mapping["format"] == "grib2"
    assert mapping["coordinates"]["vertical"]["levels"] == [
        5000, 7000, 10000, 15000, 20000, 25000, 30000, 35000,
        40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000,
        80000, 85000, 90000, 92500, 95000, 97500, 100000,
    ]
    assert mapping["target"]["boundary_interval_seconds"] == 10800
    soil = mapping["fields"]["volumetric_soil_moisture"]["selectors"]
    assert len(soil) == 4
    assert all((row["discipline"], row["category"], row["parameter"])
               == (2, 0, 192) for row in soil)


def test_composition_binds_exact_noah_layers_and_in_band_terrain() -> None:
    contract = load_composition(COMPOSITION, MAPPING)

    assert contract["soil_layers"]["remap"]["kind"] \
        == "conservative_layer_means"
    assert [
        (row["top"], row["bottom"])
        for row in contract["soil_layers"]["source_layers"]
    ] == [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)]
    terrain = contract["supplements"]["terrain_height"]
    assert terrain["data_role"] == "twentycrv3_in_band_surface"
    assert terrain["require_invariant_across_time"] is True


def test_exact_archive_inventory_has_115_pressure_and_19_surface_fields() -> None:
    levels = load_mapping(MAPPING)["coordinates"]["vertical"]["levels"]

    assert sum(_expected_inventory("pl", levels).values()) == 115
    assert sum(_expected_inventory("sfc", levels).values()) == 19


def test_discovery_binds_member_pairs_cadence_and_hashes(tmp_path: Path) -> None:
    paths = _write_series(tmp_path)

    result = discover_20crv3_grib2(tmp_path)

    assert result["member"] == "072"
    assert result["file_count"] == 4
    assert result["cadence_seconds"] == 10800
    assert result["valid_times"] == [
        "1932-03-21T00:00:00", "1932-03-21T03:00:00",
    ]
    by_name = {row["filename"]: row for row in result["files"]}
    for path in paths.values():
        assert by_name[path.name]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_discovery_rejects_mixed_filename_members(tmp_path: Path) -> None:
    _write_series(tmp_path, rows=(
        ("072", "1932032100", "pl"),
        ("072", "1932032100", "sfc"),
        ("073", "1932032103", "pl"),
        ("073", "1932032103", "sfc"),
    ))

    with pytest.raises(ValueError, match="mixes member IDs"):
        discover_20crv3_grib2(tmp_path)


def test_discovery_rejects_incomplete_time_pair(tmp_path: Path) -> None:
    _write_series(tmp_path, rows=(
        ("072", "1932032100", "pl"),
        ("072", "1932032100", "sfc"),
        ("072", "1932032103", "pl"),
    ))

    with pytest.raises(ValueError, match="pairing gaps"):
        discover_20crv3_grib2(tmp_path)


def test_manifest_rejects_input_changed_after_hash_binding(tmp_path: Path) -> None:
    paths = _write_series(tmp_path / "source")
    manifest = tmp_path / "manifest.json"
    write_20crv3_manifest(tmp_path / "source", manifest)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    paths[("072", "1932032103", "sfc")].write_bytes(b"changed")

    with pytest.raises(ValueError, match="input identity differs"):
        _manifest(manifest, manifest_sha)


def _collection(member: str | None) -> _DecodedCollection:
    valid = datetime(1932, 3, 21)
    value = _DirectValue(
        name="surface_pressure",
        valid_time=valid,
        member=member,
        source_cycle=valid,
        axes=("y", "x"),
        values=np.full((2, 2), 100000.0),
        missing_count=0,
        references=("fixture",),
    )
    return _DecodedCollection(
        latitude=np.asarray([10.0, 10.5]),
        longitude=np.asarray([20.0, 20.5]),
        vertical_values=np.asarray([100000.0]),
        direct=MappingProxyType({
            (valid, member, "surface_pressure"): value,
        }),
        source_cycles=MappingProxyType({(valid, member): valid}),
        grid_fingerprint="fixture-grid",
    )


def test_filename_member_retags_grib_records_without_pdt_member() -> None:
    bound = _bind_implicit_member(_collection(None), "072")
    key = (datetime(1932, 3, 21), "072", "surface_pressure")

    assert set(bound.direct) == {key}
    assert bound.direct[key].member == "072"
    assert set(bound.source_cycles) == {(datetime(1932, 3, 21), "072")}


def test_filename_member_rejects_unexpected_pdt_member() -> None:
    with pytest.raises(ValueError, match="unexpectedly encodes"):
        _bind_implicit_member(_collection("9"), "072")
