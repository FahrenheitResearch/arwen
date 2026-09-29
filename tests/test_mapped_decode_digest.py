"""Decode goldens compare content at every staging path."""

from __future__ import annotations

import os
import copy
from datetime import datetime
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.mapped_source import CanonicalField, _array_sha256
from gpuwm.source_frame import (
    FieldDescriptor,
    GridDescriptor,
    PORTABLE_HEADER_RULE,
    SourceFrameHeader,
    TimeDescriptor,
)
import test_mapped_engine_parity as parity


@pytest.fixture
def mapping(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({
        "format": "grib2",
        "fields": {
            "air_temperature": {},
            "specific_humidity": {
                "derivation": "specific_humidity_from_relative_humidity",
            },
        },
    }), encoding="utf-8")
    return path


def _frame(root, *, changed_field=None, records=(1, 2), resolve=False):
    root.mkdir(parents=True, exist_ok=True)
    source = root / "sample.grib2"
    source.write_bytes(b"staged bytes")
    recorded = source.resolve() if resolve else source
    references = tuple(f"{recorded}:{record}" for record in records)
    fields = {}
    descriptors = []
    for name, units, value in (
            ("air_temperature", "K", 280.0),
            ("specific_humidity", "kg kg-1", 0.004)):
        values = np.full((2, 2), value, dtype=np.float64)
        if name == changed_field:
            values[0, 0] = np.nextafter(values[0, 0], np.inf)
        field = CanonicalField(
            name=name, units=units, axes=("y", "x"), location="mass",
            staggering="none", values=values, missing_count=0,
            source_references=references,
        )
        fields[name] = field
        descriptors.append(FieldDescriptor(
            canonical_name=name, units=units, dimensions=field.axes,
            grid_location=field.location, vertical_coordinate=None,
            time=TimeDescriptor(
                reference_time="2026-01-01T00:00:00",
                valid_time="2026-01-01T00:00:00", lead_seconds=0,
                statistic="instantaneous",
            ),
            data_reference=f"sha256:{_array_sha256(field.values)}",
            dtype="float64", shape=field.values.shape,
            source_field=";".join(references),
        ))
    header = SourceFrameHeader(
        source_id="sample-source", source_cycle="2026-01-01T00:00:00",
        grid=GridDescriptor(
            projection="regular_latitude_longitude", nx=2, ny=2,
            earth_shape="sphere", scan_order="row_major",
            wind_basis="earth_relative", parameters={},
        ),
        vertical_coordinates={}, fields=tuple(descriptors),
    )
    return source, SimpleNamespace(
        valid_time=datetime(2026, 1, 1), member=None,
        source_cycle=datetime(2026, 1, 1),
        vertical_kind="pressure", vertical_units="Pa",
        grid_fingerprint="grid-content", latitude=np.array([0.0, 1.0]),
        longitude=np.array([0.0, 1.0]), vertical_values=np.array([100000.0]),
        input_sha256={str(recorded): parity._sha256(source)},
        fields=fields, header=header,
    )


def _digest(monkeypatch, mapping, root, **options):
    source, frame = _frame(root, **options)
    monkeypatch.setitem(parity.STAGED_SOURCES, "sample-source", {
        "mapping_path": mapping, "files": (source,),
    })
    return parity.parity_digest("sample-source", mapping, (frame,))


def test_decoded_digest_does_not_move_with_the_staging_root(
        tmp_path, monkeypatch, mapping):
    first = _digest(monkeypatch, mapping, tmp_path / "first")
    second = _digest(monkeypatch, mapping, tmp_path / "second")
    assert first == second
    assert first["schema"] == "gpuwm-mapped-parity-digest-v2"
    assert first["header_rule"] == PORTABLE_HEADER_RULE
    assert "portable_header_sha256" in first["frames"][0]
    assert "header_sha256" not in first["frames"][0]


def test_decoded_digest_does_not_move_through_a_linked_root(
        tmp_path, monkeypatch, mapping):
    expected = _digest(monkeypatch, mapping, tmp_path / "first")
    target = tmp_path / "physical"
    target.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        if os.name != "nt":
            pytest.skip(f"cannot create a directory link: {error}")
        # A symbolic link needs a privilege Windows withholds by default;
        # a junction needs none, and is what a linked staging root is here.
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    observed = _digest(monkeypatch, mapping, link, resolve=True)
    assert observed == expected


@pytest.mark.parametrize("name", ["air_temperature", "specific_humidity"])
def test_decoded_digest_keeps_every_array_bit(name, tmp_path, monkeypatch, mapping):
    original = _digest(monkeypatch, mapping, tmp_path / "staging")
    changed = _digest(
        monkeypatch, mapping, tmp_path / "staging", changed_field=name)
    assert original != changed
    before, after = original["frames"][0], changed["frames"][0]
    assert before["fields"][name]["sha256"] != after["fields"][name]["sha256"]
    if name == "specific_humidity":
        # Header portability cannot remove exact array coverage from the row.
        assert before["portable_header_sha256"] == after["portable_header_sha256"]
    else:
        assert before["portable_header_sha256"] != after["portable_header_sha256"]


def test_decoded_digest_keeps_reference_order(tmp_path, monkeypatch, mapping):
    first = _digest(monkeypatch, mapping, tmp_path / "staging", records=(1, 2))
    second = _digest(monkeypatch, mapping, tmp_path / "staging", records=(2, 1))
    assert first["frames"][0]["fields"] == second["frames"][0]["fields"]
    assert first != second


def _inspection():
    return {
        "mapping": {"path": "mapping.json", "sha256": "mapping-content"},
        "inputs": [{"path": "sample.grib2", "bytes": 12, "sha256": "input-content"}],
        "frames": [{"fields": {
            "air_temperature": {
                "sha256": "array-content",
                "source_references": ["sample.grib2:1", "sample.grib2:2"],
            },
        }}],
        "materialization": {
            "verdict": "PASS", "frame_count": 1,
            "frame_header_sha256": ["a" * 64],
            "frame_header_sha256_portable": ["b" * 64],
            "portable_rule": PORTABLE_HEADER_RULE,
        },
    }


@pytest.fixture
def inspection_row(monkeypatch, mapping):
    monkeypatch.setitem(parity.STAGED_SOURCES, "sample-source", {
        "mapping_path": mapping, "files": (mapping.parent / "sample.grib2",),
    })


def test_inspection_digest_keeps_portable_headers_and_reference_order(inspection_row):
    document = _inspection()
    original = parity.inspection_digest("sample-source", document)
    materialization = original["inspection"]["materialization"]
    assert "frame_header_sha256" not in materialization
    assert materialization["frame_header_sha256_portable"] == ["b" * 64]
    assert materialization["portable_rule"] == PORTABLE_HEADER_RULE
    assert "frame_header_sha256" in document["materialization"]
    changed = copy.deepcopy(document)
    changed["materialization"]["frame_header_sha256"] = ["c" * 64]
    assert parity.inspection_digest("sample-source", changed) == original
    changed["materialization"]["frame_header_sha256_portable"] = ["d" * 64]
    assert parity.inspection_digest("sample-source", changed) != original
    changed["materialization"]["frame_header_sha256_portable"] = ["b" * 64]
    changed["frames"][0]["fields"]["air_temperature"]["source_references"] = [
        "sample.grib2:2", "sample.grib2:1"]
    assert parity.inspection_digest("sample-source", changed) != original


@pytest.mark.parametrize("key,value", [
    ("frame_header_sha256_portable", None),
    ("frame_header_sha256_portable", []),
    ("portable_rule", None),
    ("portable_rule", "unknown-rule"),
])
def test_inspection_digest_requires_complete_portable_headers(key, value, inspection_row):
    document = _inspection()
    if value is None:
        document["materialization"].pop(key)
    else:
        document["materialization"][key] = value
    with pytest.raises(AssertionError, match="(?i)portable|header"):
        parity.inspection_digest("sample-source", document)
