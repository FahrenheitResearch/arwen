from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.mapped_composition import (
    INPUT_MANIFEST_SCHEMA,
    MappedSourceBundle,
    _compose_terrain,
    _decoder_inventory,
    _verify_manifest,
    decode_composed_source,
    load_composition,
)
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)
from gpuwm.mapped_source import _DecodedCollection, _DirectValue, _sha256


ROOT = Path(__file__).parents[1]
MAPPING = ROOT / "configs" / "rw-wps-era5-1974-probe.mapping.json"
COMPOSITION = ROOT / "configs" / "rw-wps-era5-1974-terrain.composition.json"


def _direct(
    name: str,
    valid_time: datetime,
    values: np.ndarray,
    cycle: datetime | None = None,
) -> _DirectValue:
    return _DirectValue(
        name=name,
        valid_time=valid_time,
        member=None,
        source_cycle=cycle or valid_time,
        axes=("y", "x"),
        values=np.asarray(values, dtype=np.float64),
        missing_count=0,
        references=(f"fixture:{name}:{valid_time.isoformat()}",),
    )


def _collection(
    latitude: np.ndarray,
    longitude: np.ndarray,
    fields: dict[tuple[datetime, str | None, str], _DirectValue],
) -> _DecodedCollection:
    cycles = {
        (valid_time, member): value.source_cycle
        for (valid_time, member, _name), value in fields.items()
    }
    return _DecodedCollection(
        latitude=np.asarray(latitude, dtype=np.float64),
        longitude=np.asarray(longitude, dtype=np.float64),
        vertical_values=np.asarray([1000.0, 850.0]),
        direct=MappingProxyType(fields),
        source_cycles=MappingProxyType(cycles),
        grid_fingerprint="fixture-grid",
    )


def _collections(*, changed: bool = False, missing_time: bool = False):
    start = datetime(1974, 4, 3, 12)
    primary_times = (start, start + timedelta(hours=6))
    terrain_times = (
        start - timedelta(hours=12), *primary_times,
        start + timedelta(hours=12),
    )
    primary_latitude = np.asarray([51.5, 51.75, 52.0])
    primary_longitude = np.asarray([260.0, 260.25, 260.5, 260.75])
    terrain_latitude = np.asarray([52.5, 52.25, 52.0, 51.75, 51.5, 51.25])
    terrain_longitude = np.asarray([
        -100.5, -100.25, -100.0, -99.75, -99.5, -99.25, -99.0,
    ])
    primary = _collection(
        primary_latitude,
        primary_longitude,
        {
            # One source CYCLE serves both leads, as a real decode's
            # reference time would; a broadcast belongs to one cycle.
            (time, None, "surface_pressure"): _direct(
                "surface_pressure", time,
                np.full((3, 4), 100000.0 + index),
                cycle=start,
            )
            for index, time in enumerate(primary_times)
        },
    )
    base = np.arange(6 * 7, dtype=np.float64).reshape(6, 7)
    terrain_fields = {}
    for index, time in enumerate(terrain_times):
        if missing_time and time == primary_times[1]:
            continue
        values = base.copy()
        if changed and index == len(terrain_times) - 1:
            values[0, 0] += 1.0
        terrain_fields[(time, None, "terrain_height")] = _direct(
            "terrain_height", time, values,
        )
    terrain = _collection(
        terrain_latitude, terrain_longitude, terrain_fields,
    )
    return primary, terrain, base[[4, 3, 2], 2:6]


def test_checked_in_terrain_composition_binds_mapping_semantics():
    contract = load_composition(COMPOSITION, MAPPING)
    spec = contract["supplements"]["terrain_height"]
    assert spec["selector_authority"] == "mapping_field_exact"
    assert spec["grid_alignment"] == "exact_coordinate_subset"
    assert spec["require_invariant_across_time"] is True
    layers = contract["soil_layers"]["source_layers"]
    assert [(layer["top"], layer["bottom"]) for layer in layers] == [
        (0.0, 0.07),
        (0.07, 0.28),
        (0.28, 1.0),
        (1.0, 2.89),
    ]
    assert [
        layer["selectors"]["soil_temperature"]["parameter"]
        for layer in layers
    ] == [139, 170, 183, 236]
    assert contract["soil_layers"]["depth_units"] == "m"


def test_checked_in_grib2_and_netcdf_compositions_bind_source_soil_contracts():
    gfs = load_composition(
        ROOT / "configs" / "rw-wps-gfs-terrain.composition.json",
        ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json",
    )
    assert gfs["supplements"]["terrain_height"]["format"] == "grib2"
    assert gfs["soil_layers"]["remap"]["kind"] == \
        "conservative_layer_means"
    gfs_last = gfs["soil_layers"]["source_layers"][-1]
    assert (gfs_last["top"], gfs_last["bottom"]) == (1.0, 2.0)
    assert gfs_last["selectors"]["soil_temperature"][
        "second_level_value"
    ] == 2.0

    era5 = load_composition(
        ROOT / "configs" / "rw-wps-era5-netcdf-terrain.composition.json",
        ROOT / "configs" / "rw-wps-era5-netcdf.mapping.json",
    )
    assert era5["supplements"]["terrain_height"]["format"] == "netcdf"
    assert era5["soil_layers"]["remap"]["kind"] == "linear_point_samples"
    # Both spellings per depth: the depth-to-selector pairing survives
    # ECMWF's rename without being rewritten, and both vintages still read.
    assert [
        layer["selectors"]["soil_temperature"]["name"]
        for layer in era5["soil_layers"]["source_layers"]
    ] == [
        ["STL1", "stl1"], ["STL2", "stl2"],
        ["STL3", "stl3"], ["STL4", "stl4"],
    ]


def test_retired_v1_composition_cannot_be_mistaken_for_v2(tmp_path):
    composition = json.loads(COMPOSITION.read_text(encoding="utf-8"))
    composition["schema"] = "gpuwm-mapped-composition-v1"
    path = tmp_path / "retired-v1-composition.json"
    path.write_text(json.dumps(composition), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported composition schema"):
        load_composition(path, MAPPING)


def test_decoder_inventory_is_exact_for_each_source_format(tmp_path):
    bridge = tmp_path / "grib1"
    inventory = tmp_path / "inventory"
    dump = tmp_path / "dump"
    assert _decoder_inventory(
        "grib1", grib1_bridge=bridge,
        grib2_inventory=None, grib2_dump=None,
    ) == {"grib1_bridge": bridge.resolve()}
    assert set(_decoder_inventory(
        "grib2", grib1_bridge=None,
        grib2_inventory=inventory, grib2_dump=dump,
    )) == {"grib2_inventory", "grib2_dump"}
    assert _decoder_inventory(
        "netcdf", grib1_bridge=None,
        grib2_inventory=None, grib2_dump=None,
    ) == {}
    with pytest.raises(ValueError, match="extra=.*grib1_bridge"):
        _decoder_inventory(
            "netcdf", grib1_bridge=bridge,
            grib2_inventory=None, grib2_dump=None,
        )


def test_composition_rejects_unknown_or_weakened_contract(tmp_path):
    payload = json.loads(COMPOSITION.read_text(encoding="utf-8"))
    payload["surprise"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_composition(path, MAPPING)

    payload.pop("surprise")
    payload["supplements"]["terrain_height"][
        "require_invariant_across_time"
    ] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be invariant"):
        load_composition(path, MAPPING)


def test_exact_subset_composition_adds_invariant_terrain_at_primary_times():
    primary, terrain, expected = _collections()
    combined, receipt = _compose_terrain(primary, terrain)

    assert receipt["status"] == "PASS"
    assert receipt["coordinate_match"] == "exact_equivalent_contiguous_subset"
    assert receipt["latitude_index_direction"] == -1
    assert receipt["longitude_equivalence"] == "modulo_360_exact"
    assert receipt["primary_shape"] == [3, 4]
    assert receipt["supplement_shape"] == [6, 7]
    assert len(combined.source_cycles) == 2
    for valid_time, member in combined.source_cycles:
        actual = combined.direct[(valid_time, member, "terrain_height")]
        np.testing.assert_array_equal(actual.values, expected)


def test_composition_rejects_two_terrain_providers():
    primary, terrain, expected = _collections()
    key = next(iter(primary.source_cycles))
    direct = dict(primary.direct)
    direct[(*key, "terrain_height")] = _direct(
        "terrain_height", key[0], expected,
    )
    primary = _collection(primary.latitude, primary.longitude, direct)
    with pytest.raises(ValueError, match="two providers"):
        _compose_terrain(primary, terrain)


def test_composition_rejects_missing_time_changed_or_nonexact_grid():
    primary, terrain, _expected = _collections(missing_time=True)
    with pytest.raises(ValueError, match="lacks exact primary valid time"):
        _compose_terrain(primary, terrain)

    primary, terrain, _expected = _collections(changed=True)
    with pytest.raises(ValueError, match="changes across supplied valid times"):
        _compose_terrain(primary, terrain)

    primary, terrain, _expected = _collections()
    perturbed = _collection(
        terrain.latitude + np.asarray([0, 0, 0, 1.0e-12, 0, 0]),
        terrain.longitude, dict(terrain.direct),
    )
    with pytest.raises(ValueError, match="exact matches"):
        _compose_terrain(primary, perturbed)


def test_composition_manifest_binds_every_authority_before_decode(tmp_path):
    primary = tmp_path / "primary.grb"
    terrain = tmp_path / "terrain.grb"
    provenance = tmp_path / "terrain-provenance.md"
    decoder = tmp_path / "grib1-bridge"
    primary.write_bytes(b"primary")
    terrain.write_bytes(b"terrain")
    provenance.write_text("source provenance\n", encoding="utf-8")
    decoder.write_bytes(b"decoder")
    manifest = tmp_path / "inputs.json"

    def row(path: Path) -> dict[str, object]:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    payload = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "mapping_sha256": _sha256(MAPPING),
        "composition_sha256": _sha256(COMPOSITION),
        "primary_files": [row(primary)],
        "supplements": {"terrain": row(terrain)},
        "provenance": {"terrain_provenance": row(provenance)},
        "decoders": {"grib1_bridge": row(decoder)},
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    digest = _sha256(manifest)
    _verify_manifest(
        manifest, digest,
        mapping_path=MAPPING, composition_path=COMPOSITION,
        primary_files=(primary,), supplement_files={"terrain": terrain},
        provenance_files={"terrain_provenance": provenance},
        decoder_files={"grib1_bridge": decoder},
    )

    terrain.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="byte count differs|SHA differs"):
        _verify_manifest(
            manifest, digest,
            mapping_path=MAPPING, composition_path=COMPOSITION,
            primary_files=(primary,), supplement_files={"terrain": terrain},
            provenance_files={"terrain_provenance": provenance},
            decoder_files={"grib1_bridge": decoder},
        )

    terrain.write_bytes(b"terrain")
    decoder.write_bytes(b"mutated decoder")
    with pytest.raises(ValueError, match="byte count differs|SHA differs"):
        _verify_manifest(
            manifest, digest,
            mapping_path=MAPPING, composition_path=COMPOSITION,
            primary_files=(primary,), supplement_files={"terrain": terrain},
            provenance_files={"terrain_provenance": provenance},
            decoder_files={"grib1_bridge": decoder},
        )


def test_composition_manifest_binds_ordered_multi_file_supplement(tmp_path):
    primary = tmp_path / "primary.grb"
    terrain_a = tmp_path / "terrain-a.grb"
    terrain_b = tmp_path / "terrain-b.grb"
    provenance = tmp_path / "provenance.md"
    decoder = tmp_path / "decoder"
    for path, content in (
        (primary, b"primary"), (terrain_a, b"terrain-a"),
        (terrain_b, b"terrain-b"), (decoder, b"decoder"),
    ):
        path.write_bytes(content)
    provenance.write_text("provenance\n", encoding="utf-8")

    def row(path):
        return {
            "path": path.name, "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({
        "schema": INPUT_MANIFEST_SCHEMA,
        "mapping_sha256": _sha256(MAPPING),
        "composition_sha256": _sha256(COMPOSITION),
        "primary_files": [row(primary)],
        "supplements": {"terrain": [row(terrain_a), row(terrain_b)]},
        "provenance": {"terrain_provenance": row(provenance)},
        "decoders": {"grib1_bridge": row(decoder)},
    }), encoding="utf-8")
    _verify_manifest(
        manifest, _sha256(manifest),
        mapping_path=MAPPING, composition_path=COMPOSITION,
        primary_files=(primary,),
        supplement_files={"terrain": (terrain_a, terrain_b)},
        provenance_files={"terrain_provenance": provenance},
        decoder_files={"grib1_bridge": decoder},
    )
    with pytest.raises(ValueError, match="path differs"):
        _verify_manifest(
            manifest, _sha256(manifest),
            mapping_path=MAPPING, composition_path=COMPOSITION,
            primary_files=(primary,),
            supplement_files={"terrain": (terrain_b, terrain_a)},
            provenance_files={"terrain_provenance": provenance},
            decoder_files={"grib1_bridge": decoder},
        )


def test_composed_decode_rejects_mapping_swap_after_semantic_snapshot(
    tmp_path,
    monkeypatch,
):
    mapping = tmp_path / "mapping.json"
    composition = tmp_path / "composition.json"
    mapping.write_bytes(MAPPING.read_bytes())
    composition.write_bytes(COMPOSITION.read_bytes())
    primary = tmp_path / "primary.grb"
    terrain = tmp_path / "terrain.grb"
    provenance = tmp_path / "terrain.md"
    decoder = tmp_path / "grib1_bridge"
    for path, payload in (
        (primary, b"primary"),
        (terrain, b"terrain"),
        (provenance, b"provenance"),
        (decoder, b"decoder"),
    ):
        path.write_bytes(payload)
    contract = load_composition(composition, mapping)
    spec = contract["supplements"]["terrain_height"]
    data_role = spec["data_role"]
    provenance_role = spec["provenance_role"]

    def row(path):
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": INPUT_MANIFEST_SCHEMA,
                "mapping_sha256": _sha256(mapping),
                "composition_sha256": _sha256(composition),
                "primary_files": [row(primary)],
                "supplements": {data_role: [row(terrain)]},
                "provenance": {provenance_role: row(provenance)},
                "decoders": {"grib1_bridge": row(decoder)},
            }
        ),
        encoding="utf-8",
    )

    def inventory_then_swap(*_args, **_kwargs):
        changed = json.loads(MAPPING.read_text(encoding="utf-8"))
        changed["name"] = "swapped-after-semantic-load"
        mapping.write_text(json.dumps(changed), encoding="utf-8")
        return {"grib1_bridge": decoder.resolve()}

    monkeypatch.setattr(
        "gpuwm.mapped_composition._decoder_inventory",
        inventory_then_swap,
    )
    with pytest.raises(ValueError, match="authority changed after validation"):
        decode_composed_source(
            composition,
            mapping,
            (primary,),
            {data_role: (terrain,)},
            {provenance_role: provenance},
            input_manifest=manifest,
            input_manifest_sha256=_sha256(manifest),
            grib1_bridge=decoder,
        )

def test_bundle_preserves_canonical_soil_arrays_and_explicit_zero_hydrometeors(
    tmp_path, monkeypatch,
):
    valid_time = datetime(1974, 4, 3, 12)
    terrain = np.arange(6, dtype=np.float64).reshape(2, 3)
    policies = {
        name: "explicit_zero_with_adapter_validation"
        for name in (
            "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
            "cloud_ice_mixing_ratio", "snow_mixing_ratio",
            "graupel_or_hail_mixing_ratio",
        )
    }
    frame = SimpleNamespace(
        valid_time=valid_time,
        fields={"terrain_height": SimpleNamespace(values=terrain)},
        header=SimpleNamespace(initialization_policies=policies),
    )
    pressure = np.ones((2, 2, 3), dtype=np.float64)
    soil_temperature = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
    soil_moisture = np.full((4, 2, 3), 0.25, dtype=np.float64)
    fields = {
        "PRES": pressure,
        "SOURCE_OROGRAPHY": terrain,
        MAPPED_SOIL_TEMPERATURE: soil_temperature,
        MAPPED_SOIL_MOISTURE: soil_moisture,
    }
    packed = Era5Snapshot(
        valid_time=valid_time,
        levels_hpa=np.asarray([1000.0, 850.0]),
        latitude=np.asarray([30.0, 31.0]),
        longitude=np.asarray([-100.0, -99.0, -98.0]),
        fields=fields,
    )
    monkeypatch.setattr(
        "gpuwm.mapped_composition.mapped_frames_to_regular_snapshots",
        lambda _frames, **_kwargs: (packed,),
    )
    dummy = tmp_path / "authority"
    dummy.write_bytes(b"authority")
    bundle = MappedSourceBundle(
        frames=(frame,),
        mapping_path=dummy, mapping_sha256="0" * 64,
        composition_path=dummy, composition_sha256="1" * 64,
        input_manifest_path=dummy, input_manifest_sha256="2" * 64,
        decoder_paths={"grib1_bridge": dummy},
        decoder_sha256={"grib1_bridge": "5" * 64},
        terrain_data_paths=(dummy,), terrain_data_sha256=("3" * 64,),
        terrain_provenance_path=dummy, terrain_provenance_sha256="4" * 64,
        soil_layer_contract=json.loads(
            (ROOT / "configs" / "rw-wps-gfs-terrain.composition.json")
            .read_text(encoding="utf-8")
        )["soil_layers"],
        alignment_receipt={"status": "PASS"},
    )

    actual = bundle.regular_snapshots()[0]
    np.testing.assert_array_equal(
        actual.fields[MAPPED_SOIL_TEMPERATURE], soil_temperature,
    )
    np.testing.assert_array_equal(
        actual.fields[MAPPED_SOIL_MOISTURE], soil_moisture,
    )
    assert not any(name.startswith("GFS_ST") for name in actual.fields)
    for name in ("QC", "QR", "QI", "QS", "QG"):
        np.testing.assert_array_equal(actual.fields[name], np.zeros_like(pressure))


def _analysis_only_collections():
    """Terrain supplied at the FIRST primary valid time alone.

    The shape agencies that publish invariants once per cycle actually
    ship: the analysis file carries orography, the forecast-hour files do
    not repeat it.
    """

    primary, terrain, expected = _collections()
    first_time = min(time for time, _member in primary.source_cycles)
    analysis_fields = {
        key: value for key, value in terrain.direct.items()
        if key[0] == first_time
    }
    assert len(analysis_fields) == 1
    analysis_terrain = _collection(
        terrain.latitude, terrain.longitude, analysis_fields,
    )
    return primary, analysis_terrain, expected


def test_analysis_invariant_broadcast_binds_one_analysis_record_to_all_times():
    primary, terrain, expected = _analysis_only_collections()
    combined, receipt = _compose_terrain(
        primary, terrain,
        time_alignment="cycle_invariant_broadcast",
    )

    assert receipt["status"] == "PASS"
    assert receipt["invariant_across_all_supplement_times"] is True
    assert receipt["time_alignment"] == "cycle_invariant_broadcast"
    assert len(receipt["supplement_valid_times"]) == 1
    assert len(receipt["matched_primary_valid_times"]) == 2
    assert len(combined.source_cycles) == 2
    for valid_time, member in combined.source_cycles:
        actual = combined.direct[(valid_time, member, "terrain_height")]
        np.testing.assert_array_equal(actual.values, expected)


def test_default_alignment_still_refuses_an_analysis_only_supplement():
    primary, terrain, _expected = _analysis_only_collections()
    with pytest.raises(ValueError, match="lacks exact primary valid time"):
        _compose_terrain(primary, terrain)


def test_broadcast_alignment_still_refuses_a_changing_supplement():
    primary, terrain, _expected = _collections(changed=True)
    with pytest.raises(ValueError, match="changes across supplied valid times"):
        _compose_terrain(
            primary, terrain,
            time_alignment="cycle_invariant_broadcast",
        )


def test_composition_time_alignment_grammar_is_closed(tmp_path):
    raw = json.loads(COMPOSITION.read_text(encoding="utf-8"))
    raw["supplements"]["terrain_height"]["time_alignment"] = "whenever"
    path = tmp_path / "bad-alignment.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="time alignment"):
        load_composition(path, MAPPING)


def _engine_compose_harness(tmp_path, monkeypatch):
    """Drive ``_compose_through_engine`` with a fake engine seam.

    The engine writes the WHOLE composed frame stream -- f64, tens of GB
    for the largest registered sources -- into the scratch directory
    before read-back, so where that directory is created is a
    correctness question these tests pin.  The bridge calls are faked;
    the ``tempfile.TemporaryDirectory`` call is recorded so each test
    can observe the ``dir=`` the scratch was created with.
    """

    import tempfile

    from gpuwm import mapped_composition, mapped_engine_bridge

    files = {}
    for name in (
        "mapping.json", "composition.json", "manifest.json",
        "primary.grb2", "terrain.grb2", "terrain-provenance.md",
    ):
        path = tmp_path / name
        path.write_text("fixture", encoding="utf-8")
        files[name] = path

    engine_calls: list[dict] = []
    monkeypatch.setattr(
        mapped_engine_bridge, "run_engine",
        lambda *args, **kwargs: engine_calls.append(dict(kwargs)))
    frame = SimpleNamespace(
        mapping_sha256="mapping-hash",
        fields={"terrain_height": SimpleNamespace(
            values=np.zeros((2, 2), dtype=np.float64))},
    )
    monkeypatch.setattr(
        mapped_engine_bridge, "read_frameset", lambda _directory: (frame,))
    monkeypatch.setattr(
        mapped_engine_bridge, "read_composition_evidence",
        lambda _directory: {
            "alignment_receipt": {"status": "PASS"},
            "contributing_sources": [],
        })

    captured: dict[str, object] = {}
    real_factory = tempfile.TemporaryDirectory

    def recording_factory(*args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", recording_factory)

    def compose(destination):
        return mapped_composition._compose_through_engine(
            engine=tmp_path / "engine.exe",
            mapping_path=files["mapping.json"],
            composition_path=files["composition.json"],
            manifest_path=files["manifest.json"],
            manifest_sha256="manifest-hash",
            primary=(files["primary.grb2"],),
            supplements={"terrain": (files["terrain.grb2"],)},
            provenance={"terrain-provenance": files["terrain-provenance.md"]},
            contributing={},
            contract={"soil_layers": {"schema": "fixture"}},
            bindings={},
            terrain_spec={
                "data_role": "terrain",
                "provenance_role": "terrain-provenance",
            },
            decoders={},
            snapshots={},
            before={str(path): f"{name}-hash"
                    for name, path in files.items()}
                   | {str(files["mapping.json"]): "mapping-hash"},
            scratch_destination=destination,
        )

    return SimpleNamespace(
        compose=compose, captured=captured, engine_calls=engine_calls)


def test_engine_compose_scratch_lands_beside_the_destination(
        tmp_path, monkeypatch):
    """The composed frame stream stages on the destination's filesystem.

    Named breakage, measured twice and deterministic: the engine staged
    the whole composed frame stream in the SYSTEM temp, and on a box
    whose system temp is RAM-backed tmpfs (Ubuntu with /tmp on tmpfs and
    a user quota; Fedora mounts /tmp at RAM/2 by default) a bare default
    prep of the biggest registered source died with "cannot write the
    frame stream: Disk quota exceeded (os error 122)" -- at exactly the
    scale the release headlines.  The scratch belongs beside the prep
    output, which is disk-backed by construction because the output
    lands there anyway.
    """

    monkeypatch.delenv("GPUWM_COMPOSE_SCRATCH", raising=False)
    harness = _engine_compose_harness(tmp_path, monkeypatch)
    destination = tmp_path / "case" / "prep-output"

    bundle = harness.compose(destination)

    assert bundle.alignment_receipt["status"] == "PASS"
    assert Path(harness.captured["kwargs"]["dir"]) == destination.parent
    # The parent is created for the scratch (the prep would create it
    # moments later anyway); the destination itself stays untouched so
    # the route's create-only refusal still means something.
    assert destination.parent.is_dir()
    assert not destination.exists()


def test_compose_scratch_env_override_wins_over_the_destination(
        tmp_path, monkeypatch):
    override = tmp_path / "big-disk-scratch"
    override.mkdir()
    monkeypatch.setenv("GPUWM_COMPOSE_SCRATCH", str(override))
    harness = _engine_compose_harness(tmp_path, monkeypatch)

    harness.compose(tmp_path / "case" / "prep-output")

    assert Path(harness.captured["kwargs"]["dir"]) == override


def test_compose_scratch_env_override_refuses_a_missing_directory_by_name(
        tmp_path, monkeypatch):
    """A named override that cannot hold the stream refuses, not falls back.

    Silently falling back to the system temp would put the multi-GB
    stream on exactly the filesystem the caller set the variable to
    steer away from -- the quota failure this placement exists to
    prevent, now with the user believing they had prevented it.
    """

    missing = tmp_path / "no-such-scratch"
    monkeypatch.setenv("GPUWM_COMPOSE_SCRATCH", str(missing))
    harness = _engine_compose_harness(tmp_path, monkeypatch)

    with pytest.raises(NotADirectoryError) as refusal:
        harness.compose(tmp_path / "case" / "prep-output")

    message = str(refusal.value)
    assert "GPUWM_COMPOSE_SCRATCH" in message
    assert str(missing) in message
    assert not harness.engine_calls


def test_compose_scratch_falls_back_to_system_temp_without_a_destination(
        tmp_path, monkeypatch):
    """No destination, no override: the system temp is the documented home.

    Callers with no real output directory (in-memory recompose paths)
    stage streams sized to what they hold in memory, so the system temp
    is proportionate there and demanding a destination would refuse
    work that was never at risk.
    """

    monkeypatch.delenv("GPUWM_COMPOSE_SCRATCH", raising=False)
    harness = _engine_compose_harness(tmp_path, monkeypatch)

    harness.compose(None)

    assert harness.captured["kwargs"].get("dir") is None
