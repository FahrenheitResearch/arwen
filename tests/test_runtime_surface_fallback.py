"""A cycle that publishes no runtime surface record (gate ruling 3).

Archive HRRR cycles (2016 wrfsfc) carry no VEG record.  When the field is
the source table's default, the fetch records its climatology fallback
(id, reason, cycle) and the start reads GREENFRAC; a configuration that
declares the field refuses by name.  Zero matches say the cycle publishes
no such record, not that a choice would be ambiguous.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import runtime_surface_fetch as rsf
from gpuwm.source_adapters import get_source_adapter

CYCLE = datetime(2016, 5, 24, 18)


def _fake_fetch(monkeypatch, *, published: bool):
    from gpuwm import rustwx_fetch

    def run_fetch(binary, *, out, **kwargs):
        name = "hrrr.t18z.wrfsfcf02.grib2"
        (Path(out) / name).write_bytes(b"GRIB-sfc")
        return {"files": [{"name": name, "grib_url": "https://example/" + name,
                           "mode": "full-file", "sha256": "ab" * 32}]}

    def extract(path, output, numeric_selector):
        if not published:
            raise rsf.RuntimeSurfaceRecordAbsent("no match")
        Path(output).write_bytes(b"GRIB-veg")

    monkeypatch.setattr(rustwx_fetch, "write_pattern_file", lambda *a, **k: None)
    monkeypatch.setattr(rustwx_fetch, "run_fetch", run_fetch)
    monkeypatch.setattr(rsf, "extract_runtime_record", extract)
    import gpuwm.fetch as fetch_module
    monkeypatch.setattr(fetch_module, "count_grib2_messages", lambda path: 1)


def _append(tmp_path, declared=frozenset()):
    soil = tmp_path / "hrrr.t18z.soilf02.grib2"
    soil.write_bytes(b"GRIB-soil")
    rows = rsf.append_runtime_surface_records(
        soil, adapter=get_source_adapter("hrrr"), cycle=CYCLE, lead=2,
        host="s3", binary=Path("rw_fetch"), progress=lambda *a: None,
        declared=declared)
    return soil, rows


def test_default_field_absent_from_cycle_records_climatology_fallback(
        tmp_path, monkeypatch):
    _fake_fetch(monkeypatch, published=False)
    soil, rows = _append(tmp_path)
    assert soil.read_bytes() == b"GRIB-soil"  # nothing appended
    (row,) = rows
    assert row["field"] == "VEGFRA"
    assert row["fallback_id"] == "static-greenfrac-monthly-climatology"
    assert row["reason"] == "this cycle publishes no VEG:surface record"
    assert row["cycle"] == "2016-05-24T18Z" and row["lead"] == 2
    assert row["source_file"] == "hrrr.t18z.wrfsfcf02.grib2"


def test_declared_field_absent_from_cycle_refuses_by_name(tmp_path, monkeypatch):
    _fake_fetch(monkeypatch, published=False)
    with pytest.raises(ValueError) as caught:
        _append(tmp_path, declared=frozenset({"VEGFRA"}))
    text = str(caught.value)
    assert "runtime surface VEGFRA" in text
    assert "this cycle publishes no VEG:surface record" in text
    assert "declares runtime_surface VEGFRA" in text
    assert "ambiguous" not in text


def test_published_field_is_still_appended(tmp_path, monkeypatch):
    _fake_fetch(monkeypatch, published=True)
    soil, rows = _append(tmp_path, declared=frozenset({"VEGFRA"}))
    assert soil.read_bytes() == b"GRIB-soil" + b"GRIB-veg"
    (row,) = rows
    assert "fallback_id" not in row and row["field"] == "VEGFRA"


def test_zero_matches_are_absence_not_ambiguity(tmp_path, monkeypatch):
    from gpuwm import bridges
    inventory = tmp_path / "grib2_inventory"
    inventory.write_bytes(b"#!" + rsf.EXTRACT_ABI)
    monkeypatch.setattr(bridges, "find_bridge", lambda name: inventory)
    header = "index\tdiscipline\tcategory\tparameter\tlevel_type\tlevel_value\tpdt"
    calls = {"rows": [header, "1\t0\t0\t0\t1\t0\t0"]}
    monkeypatch.setattr(rsf.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="\n".join(calls["rows"]) + "\n"))
    selector = (("discipline", 2), ("category", 0), ("parameter", 4),
                ("level_type", 1), ("level_value", 0), ("pdt", 0))
    with pytest.raises(rsf.RuntimeSurfaceRecordAbsent, match="matches no GRIB record"):
        rsf.extract_runtime_record(tmp_path / "x.grib2", tmp_path / "y", selector)
    calls["rows"] = [header, "1\t2\t0\t4\t1\t0\t0", "2\t2\t0\t4\t1\t0\t0"]
    with pytest.raises(ValueError, match="found 2 GRIB records") as caught:
        rsf.extract_runtime_record(tmp_path / "x.grib2", tmp_path / "y", selector)
    assert not isinstance(caught.value, rsf.RuntimeSurfaceRecordAbsent)


def _manifest(tmp_path, rows):
    path = tmp_path / "fetch-manifest.json"
    path.write_text(json.dumps({"files": [
        {"name": "hrrr.t18z.soilf02.grib2", "runtime_surface": rows}]}))
    return path


def test_prepare_accepts_a_recorded_fallback_and_returns_it(tmp_path):
    adapter = get_source_adapter("hrrr")
    row = {"field": "VEGFRA", "fallback_id": "static-greenfrac-monthly-climatology",
           "reason": "this cycle publishes no VEG:surface record",
           "cycle": "2016-05-24T18Z", "lead": 2}
    met = SimpleNamespace(fields={})
    recorded = rsf.require_runtime_surface_fields(
        met, adapter, source_manifest=_manifest(tmp_path, [row]))
    assert recorded == {"VEGFRA": [row]}


def test_prepare_still_refuses_a_cache_that_dropped_the_field(tmp_path):
    adapter = get_source_adapter("hrrr")
    published = {"field": "VEGFRA", "sha256": "cd" * 32}
    met = SimpleNamespace(fields={})
    with pytest.raises(ValueError, match="records no fallback"):
        rsf.require_runtime_surface_fields(
            met, adapter, source_manifest=_manifest(tmp_path, [published]))
    with pytest.raises(ValueError, match="records no fallback"):
        rsf.require_runtime_surface_fields(met, adapter)
    assert rsf.require_runtime_surface_fields(
        met, adapter, refuse_dropped=False) == {}
    assert rsf.require_runtime_surface_fields(
        SimpleNamespace(fields={"VEGFRA": object()}), adapter) == {}


def test_declared_fields_must_be_ones_the_source_carries():
    adapter = get_source_adapter("hrrr")
    assert rsf.declared_runtime_surface_fields(None, adapter) == frozenset()
    assert rsf.declared_runtime_surface_fields("VEGFRA", adapter) == {"VEGFRA"}
    with pytest.raises(ValueError, match="does not carry"):
        rsf.declared_runtime_surface_fields("VEGFRA,ALBEDO", adapter)
    with pytest.raises(ValueError, match="comma-separated"):
        rsf.declared_runtime_surface_fields("VEGFRA,", adapter)


def test_fetch_table_declaration_validates_and_reaches_the_fetch_command():
    from gpuwm.fetch import validate_fetch_hints
    validate_fetch_hints({"source": "hrrr", "runtime_surface": "VEGFRA"},
                         source="test")
    with pytest.raises(ValueError, match="does not carry"):
        validate_fetch_hints({"source": "hrrr", "runtime_surface": "SOILW"},
                             source="test")
    from gpuwm.go_cli import fetch_command
    plan = {"source": "hrrr", "cycle": "2016-05-24T18", "hours": 1,
            "area": "33.6,-106.1,41.4,-93.9", "data": Path("data"),
            "runtime_surface": "VEGFRA"}
    command = fetch_command(plan)
    assert command[command.index("--runtime-surface") + 1] == "VEGFRA"
    plan.pop("runtime_surface")
    assert "--runtime-surface" not in fetch_command(plan)


def test_fallback_table_is_keyed_by_field_not_by_source():
    from gpuwm.source_adapters import _ADAPTERS
    declared = {row[0] for adapter in _ADAPTERS
                for row in adapter.runtime_surface_fields}
    assert declared <= set(rsf.RUNTIME_SURFACE_FALLBACKS)
    for adapter in _ADAPTERS:
        assert adapter.source_id not in rsf.RUNTIME_SURFACE_FALLBACKS
