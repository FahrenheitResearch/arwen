"""HRRR cycles before 2020-12-02 publish no vegetation fraction (HRRRv3).

Their wrfsfc files carry no VEG record (GRIB2 2/0/4) and the .idx lists
none.  Both HRRR routes refused every such cycle: the native route's
runtime-surface fetch stopped on rw_fetch's index-subset refusal before
its recorded GREENFRAC fallback could run, and the hrrr-prs composition
refused its vegetation_surface binding ("0 of 148 GRIB message(s) ...
match").  A start now reads the static GREENFRAC monthly climatology
interpolated to the start date, as WPS does, and the receipt records it:
the native fetch receipt's runtime_surface row, and the composition's
contributing source as UNPUBLISHED with the fallback id.  A published
record is still read, a field a configuration declares is still refused,
and a binding with any field that has no fallback is still refused.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import runtime_surface_fetch as rsf
from gpuwm.source_adapters import get_source_adapter

ROOT = Path(__file__).parents[1]
AUTHORITIES = ROOT / "gpuwm" / "authorities"
CYCLE = datetime(2019, 11, 22, 12)
GREENFRAC = "static-greenfrac-monthly-climatology"
INDEX_MISS = ("1 of 1 --var-pattern selectors matched no index record "
              "(first: VEG:surface); refusing to download a silently "
              "smaller subset")


def _fake_rw_fetch(monkeypatch, *, failure):
    from gpuwm import rustwx_fetch
    probes = []

    def run_fetch(binary, *, out, **kwargs):
        raise failure

    def run_probe(binary, **kwargs):
        probes.append(kwargs)
        url = ("https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20191122/"
               "conus/hrrr.t12z.wrfsfcf00.grib2")
        return {"schema": rustwx_fetch.PROBE_REPORT_SCHEMA, "hours": [{
            "forecast_hour": 0, "source": "aws", "grib_url": url,
            "idx_url": url + ".idx", "mode": "idx-subset",
            "probe": {"idx_record_count": 148}}]}

    monkeypatch.setattr(rustwx_fetch, "write_pattern_file", lambda *a, **k: None)
    monkeypatch.setattr(rustwx_fetch, "run_fetch", run_fetch)
    monkeypatch.setattr(rustwx_fetch, "run_probe", run_probe)
    return probes


def _append(tmp_path, declared=frozenset()):
    soil = tmp_path / "hrrr.t12z.soilf00.grib2"
    soil.write_bytes(b"GRIB-soil")
    rows = rsf.append_runtime_surface_records(
        soil, adapter=get_source_adapter("hrrr"), cycle=CYCLE, lead=0,
        host="s3", binary=Path("rw_fetch"), progress=lambda *a: None,
        declared=declared)
    return soil, rows


def _index_miss():
    from gpuwm.rustwx_fetch import EXIT_REFUSED, RwFetchError
    return RwFetchError("fetch", INDEX_MISS, returncode=EXIT_REFUSED)


# -- native route: the fetch records the fallback from the index ---------------

def test_an_index_that_lists_no_veg_records_the_greenfrac_fallback(
        tmp_path, monkeypatch):
    probes = _fake_rw_fetch(monkeypatch, failure=_index_miss())
    soil, rows = _append(tmp_path)
    assert soil.read_bytes() == b"GRIB-soil"  # nothing appended
    (row,) = rows
    assert row["field"] == "VEGFRA" and row["fallback_id"] == GREENFRAC
    assert row["reason"] == "this cycle publishes no VEG:surface record"
    assert row["cycle"] == "2019-11-22T12Z" and row["lead"] == 0
    assert row["source_file"] == "hrrr.t12z.wrfsfcf00.grib2"
    assert row["evidence"] == "index" and row["idx_record_count"] == 148
    assert row["idx_url"].endswith("hrrr.t12z.wrfsfcf00.grib2.idx")
    assert row["source_file_sha256"] is None  # no payload moved
    assert probes and probes[0]["product"] == "sfc"
    # The preparation accepts the recorded row and carries it on.
    manifest = tmp_path / "fetch-manifest.json"
    manifest.write_text(json.dumps({"files": [
        {"name": soil.name, "runtime_surface": rows}]}))
    recorded = rsf.require_runtime_surface_fields(
        SimpleNamespace(fields={}), get_source_adapter("hrrr"),
        source_manifest=manifest)
    assert recorded == {"VEGFRA": rows}


def test_a_declared_field_the_index_does_not_list_still_refuses(
        tmp_path, monkeypatch):
    _fake_rw_fetch(monkeypatch, failure=_index_miss())
    with pytest.raises(ValueError, match="declares runtime_surface VEGFRA"):
        _append(tmp_path, declared=frozenset({"VEGFRA"}))


def test_other_fetch_failures_are_not_mistaken_for_absence(
        tmp_path, monkeypatch):
    from gpuwm.rustwx_fetch import EXIT_REFUSED, EXIT_TRANSFER, RwFetchError
    for failure in (
            RwFetchError("fetch", "connection reset", returncode=EXIT_TRANSFER),
            RwFetchError("fetch", INDEX_MISS.replace("VEG:surface", "SOILW:0 m"),
                         returncode=EXIT_REFUSED),
            RwFetchError("fetch", INDEX_MISS, returncode=EXIT_TRANSFER)):
        _fake_rw_fetch(monkeypatch, failure=failure)
        with pytest.raises(RwFetchError):
            _append(tmp_path)


# -- hrrr-prs route: the composition records the binding as unpublished --------

def _packaged_bindings(name):
    document = json.loads((AUTHORITIES / name).read_text(encoding="utf-8"))
    return document["field_sources"]


def test_both_hrrr_compositions_offer_the_vegetation_fallback():
    for name in ("rw-wps-hrrr-prs-grib2.composition.json",
                 "rw-wps-hrrr-native-grib2.composition.json"):
        bindings = _packaged_bindings(name)
        offered = rsf.composition_unpublished_fallbacks(bindings)
        assert offered == {"vegetation_fraction": GREENFRAC}, name
        # Soil, terrain and every other borrowed field has no fallback, so
        # a binding that carries one of them is still refused when absent.
        others = {str(field) for binding in bindings.values()
                  for field in binding["fields"]} - {"vegetation_fraction"}
        assert not others & set(offered)


def _record(fallback, status="UNPUBLISHED", fields=("vegetation_fraction",)):
    return {"binding": "vegetation_surface",
            "source_id": "noaa-hrrr-conus-surface-vegetation-grib2",
            "data": [{"path": "/x/hrrr.t12z.wrfsfcf00.grib2", "sha256": "ab" * 32}],
            "fields": list(fields),
            "alignment": {"status": status, "reason": "the supplied files "
                          "publish no record for these fields",
                          "fallback": fallback}}


def test_an_unpublished_receipt_names_the_table_fallback_or_refuses():
    rows = rsf.unpublished_binding_fallbacks(
        _record({"vegetation_fraction": GREENFRAC}))
    (row,) = rows["VEGFRA"]
    assert row["fallback_id"] == GREENFRAC
    assert row["source_files"] == ["/x/hrrr.t12z.wrfsfcf00.grib2"]
    assert rsf.unpublished_binding_fallbacks(_record({}, status="PASS")) is None
    for fallback, fields in (({"vegetation_fraction": "zero"}, ("vegetation_fraction",)),
                             ({}, ("vegetation_fraction",)),
                             ({"land_fraction": GREENFRAC}, ("land_fraction",))):
        with pytest.raises(ValueError, match="without a recorded origin"):
            rsf.unpublished_binding_fallbacks(_record(fallback, fields=fields))


def test_the_engine_command_carries_the_fallbacks(tmp_path):
    from gpuwm.mapped_engine_bridge import engine_command
    base = dict(engine=Path("e"), mapping=Path("m"), input_list=Path("i"),
                output=Path("o"))
    plain = engine_command("compose", **base)
    assert "--unpublished-fallback" not in plain
    command = engine_command(
        "compose", **base,
        unpublished_fallbacks={"vegetation_fraction": GREENFRAC})
    assert command[len(plain):] == [
        "--unpublished-fallback", f"vegetation_fraction={GREENFRAC}"]


def test_the_python_twin_records_only_a_fully_covered_binding():
    from gpuwm.mapped_composition import _unpublished_binding_record
    before = {"/m.json": "11" * 32, "/veg.grib2": "22" * 32, "/p.json": "33" * 32}
    binding = {"source_id": "noaa-hrrr-conus-surface-vegetation-grib2",
               "fields": ["vegetation_fraction"]}
    kwargs = dict(mapping_path=Path("/m.json"), donor_files=(Path("/veg.grib2"),),
                  provenance_path=Path("/p.json"), before={
                      str(Path(key)): value for key, value in before.items()})
    record = _unpublished_binding_record(
        "vegetation_surface", binding, {"vegetation_fraction": GREENFRAC}, **kwargs)
    assert record["alignment"] == {
        "status": "UNPUBLISHED",
        "reason": "the supplied files publish no record for these fields",
        "fallback": {"vegetation_fraction": GREENFRAC}}
    assert record["fields"] == ["vegetation_fraction"]
    assert rsf.unpublished_binding_fallbacks(record)["VEGFRA"]
    assert _unpublished_binding_record(
        "vegetation_surface", binding, {}, **kwargs) is None
    soil = dict(binding, fields=["vegetation_fraction", "soil_temperature"])
    assert _unpublished_binding_record(
        "vegetation_surface", soil, {"vegetation_fraction": GREENFRAC},
        **kwargs) is None
    terrain = dict(binding, fields=["terrain_height"])
    assert _unpublished_binding_record(
        "t", terrain, {"terrain_height": GREENFRAC}, **kwargs) is None


def test_a_start_without_vegfra_reads_greenfrac_at_the_valid_date():
    import numpy as np
    from gpuwm.ingest.vegetation import initial_vegetation_fraction
    from gpuwm.static.build import monthly_interp_to_date
    months = np.stack([np.full((2, 3), 0.05 * (month + 1), dtype=np.float32)
                       for month in range(12)])
    static = {"GREENFRAC": months, "LANDMASK": np.ones((2, 3))}
    when = datetime(2019, 11, 22, 12)
    got = initial_vegetation_fraction(SimpleNamespace(fields={}), static, when)
    np.testing.assert_allclose(
        got, 100.0 * monthly_interp_to_date(months, when))
