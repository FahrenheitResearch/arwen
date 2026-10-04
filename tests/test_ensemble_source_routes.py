"""Source identity, indexed range safety, and streaming member addressing."""

from datetime import datetime
import io
import json

import pytest

from gpuwm import fetch_routes, source_adapters, source_readiness
from gpuwm.member_grammar import load_member_grammar, MemberIdentityRefusal
from gpuwm.member_index import select_member_ranges, download_indexed_member
from gpuwm.source_authorities import packaged_member_grammar


INDEX = {"member_key": "number", "offset_key": "_offset", "length_key": "_length",
         "filters": {"type": "pf"}}


def _index(*rows):
    return "\n".join(json.dumps(row) for row in rows).encode()


def test_ensemble_links_and_separate_control_are_registry_facts():
    assert source_adapters.get_source_adapter("gfs").ensemble_source == "gefs"
    assert source_adapters.get_source_adapter("ifs").ensemble_source == "ecmwf-ens"
    assert source_adapters.get_source_adapter("aigfs").ensemble_source == "aigefs"
    assert source_adapters.get_source_adapter("ai-gefs").source_id == "aigefs"
    row = source_adapters.get_source_adapter("ecmwf-ens")
    assert row.ensemble_control_source == "ecmwf-open-data"
    grammar = load_member_grammar(packaged_member_grammar(row.member_set))
    assert [member.ordinal for member in grammar.members()] == list(range(1, 51))
    assert grammar.member_for_ordinal(0) is None
    with pytest.raises(MemberIdentityRefusal):
        grammar.member("p51")


def test_index_selection_never_includes_other_members_or_statistics():
    data = _index(
        {"number": "1", "type": "pf", "_offset": 100, "_length": 40},
        {"number": "2", "type": "pf", "_offset": 0, "_length": 40},
        {"number": "1", "type": "em", "_offset": 40, "_length": 40})
    assert select_member_ranges(data, INDEX, 1) == ((100, 40),)
    with pytest.raises(ValueError, match="no records"):
        select_member_ranges(data, INDEX, 3)


def test_public_regional_members_do_not_claim_missing_initialization_state(tmp_path):
    row = source_adapters.get_source_adapter("rrfs-ens")
    assert source_adapters.get_source_adapter("rrfs").ensemble_source == row.source_id
    assert not row.runnable
    assert "250 hPa" in row.composition_requirement
    assert "soil" in row.composition_requirement
    grammar = load_member_grammar(packaged_member_grammar(row.member_set))
    assert [m.member_id for m in grammar.members()] == [f"m{n:03d}" for n in range(1, 6)]
    assert grammar.member_for_ordinal(0) is None
    plan = fetch_routes.resolve_request("rrfsens", cycle=datetime(2026, 10, 1),
                                        hours=1, member="m005")
    assert len(plan.objects) == 4
    assert all("/m005/" in obj.key for obj in plan.objects)
    assert all(obj.relpath == "upstream/" + obj.key for obj in plan.objects)
    assert plan.route.posting.streams
    for obj in plan.objects:
        path = tmp_path / obj.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"path fixture")
    fetch_routes.write_handoff(plan, tmp_path)
    from pathlib import Path
    handoff = json.loads((tmp_path / fetch_routes.PREP_ARGUMENTS_NAME).read_text())
    assert all((Path(handoff["member_prep"]["inputs"]) / obj.key).is_file()
               for obj in plan.objects)


@pytest.mark.parametrize("changed", [{"member": "2"}, {"pdt": "2"},
                                      {"pdt": "0"}, {"ensemble_size": "50"}])
def test_index_claim_cannot_override_actual_grib_identity(changed):
    from gpuwm.member_prep import verify_member_rows
    row = source_adapters.get_source_adapter("ecmwf-ens")
    grammar = load_member_grammar(packaged_member_grammar(row.member_set))
    good = {"index": "0", "pdt": "1", "member": "1", "ensemble_type": "255",
            "ensemble_size": "51", "generating_process": "4",
            "forecast_generating_process_id": "161", "derived_forecast": "-"}
    with pytest.raises(MemberIdentityRefusal):
        verify_member_rows(grammar, grammar.member("p01"),
                           [good, {**good, "index": "1", **changed}],
                           source_label="indexed member")


@pytest.mark.parametrize("offset,length", [(-1, 40), (0, 0), (True, 40), (0, "40")])
def test_invalid_ranges_refuse_before_http(offset, length):
    with pytest.raises(ValueError, match="invalid"):
        select_member_ranges(_index({"number": "1", "type": "pf", "_offset": offset,
                                     "_length": length}), INDEX, 1)


def test_overlapping_ranges_cannot_duplicate_messages():
    with pytest.raises(ValueError, match="overlap"):
        select_member_ranges(_index(
            {"number": "1", "type": "pf", "_offset": 0, "_length": 40},
            {"number": "2", "type": "pf", "_offset": 30, "_length": 40}), INDEX, 1)


def test_indexed_route_preserves_member_paths_and_per_lead_streaming():
    plan = fetch_routes.resolve_request("ecmwf-ens", cycle=datetime(2026, 10, 1),
                                        hours=6, member="p50")
    assert plan.route.posting.streams
    assert [item.lead for item in plan.objects] == [0, 3, 6]
    assert all(item.member_ordinal == 50 for item in plan.objects)
    assert all(item.relpath.startswith("members/p50/") for item in plan.objects)
    assert all(item.idx_url.endswith("-enfo-ef.index") for item in plan.objects)
    assert all(not item.idx_url.endswith(".grib2.index") for item in plan.objects)
    assert plan.supplement_files == plan.primary_files


def test_posted_readiness_uses_replacement_index_suffix():
    plan = fetch_routes.resolve_request("ecmwf-ens", cycle=datetime(2026, 10, 1),
                                        hours=3, member="p01")
    from types import SimpleNamespace
    window = SimpleNamespace(source="ecmwf-ens", provider=None, plan=plan)
    urls = source_readiness.lead_urls(window, 0, "ecmwf", objects=(plan.objects[0],))
    assert len(urls) == 2
    assert urls[0].endswith("-0h-enfo-ef.grib2")
    assert urls[1].endswith("-0h-enfo-ef.index")


def test_separate_control_layout_starts_at_first_50r1_cycle():
    with pytest.raises(ValueError, match="publication|2026-05-12|2026-05"):
        fetch_routes.resolve_request("ecmwf-ens", cycle=datetime(2026, 5, 12, 0),
                                     hours=3, member="p01")
    plan = fetch_routes.resolve_request("ecmwf-ens", cycle=datetime(2026, 5, 12, 6),
                                        hours=3, member="p01")
    assert plan.objects[0].key.startswith("20260512/06z/")


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


def test_whole_ensemble_http_reply_is_refused_without_reading_body(tmp_path, monkeypatch):
    index = _index({"number": "1", "type": "pf", "_offset": 0, "_length": 40})
    whole = Response(b"must not be read", status=200, headers={"ETag": '"v1"'})
    responses = iter((Response(index), whole))
    monkeypatch.setattr("gpuwm.member_index.paced_urlopen", lambda *a, **k: next(responses))
    reads = []
    whole.read = lambda *args: reads.append(args)
    dest = tmp_path / "member.grib2"
    with pytest.raises(ValueError, match="exact requested bytes"):
        download_indexed_member("https://example.invalid/data", "https://example.invalid/index",
                                dest, declaration=INDEX, member_ordinal=1,
                                source="ecmwf-ens", member="p01")
    assert not reads
    assert not dest.exists()
    assert not dest.with_suffix(".grib2.part").exists()


def test_changed_object_between_ranges_is_not_published(tmp_path, monkeypatch):
    index = _index({"number": "1", "type": "pf", "_offset": 0, "_length": 40},
                   {"number": "1", "type": "pf", "_offset": 80, "_length": 40})
    responses = iter((Response(index),
                      Response(b"x" * 40, 206, {"Content-Range": "bytes 0-39/120", "ETag": '"v1"'}),
                      Response(b"x" * 40, 206, {"Content-Range": "bytes 80-119/120", "ETag": '"v2"'})))
    requests = []
    def open_response(request, **kwargs):
        requests.append(request)
        return next(responses)
    monkeypatch.setattr("gpuwm.member_index.paced_urlopen", open_response)
    dest = tmp_path / "member.grib2"
    with pytest.raises(ValueError, match="changed"):
        download_indexed_member("https://example.invalid/data", "https://example.invalid/index",
                                dest, declaration=INDEX, member_ordinal=1,
                                source="ecmwf-ens", member="p01")
    assert requests[-1].get_header("If-match") == '"v1"'
    assert not dest.exists()


@pytest.mark.parametrize("etag,content_range", [('W/"v1"', "bytes 0-39/40"),
                                                ("v1", "bytes 0-39/40"),
                                                ('"v1"', "bytes 0-39/20")])
def test_unpinned_or_impossible_range_is_rejected_before_body(tmp_path, monkeypatch,
                                                            etag, content_range):
    index = _index({"number": "1", "type": "pf", "_offset": 0, "_length": 40})
    invalid = Response(b"must not be read", 206,
                       {"Content-Range": content_range, "ETag": etag})
    reads = []
    invalid.read = lambda *args: reads.append(args)
    responses = iter((Response(index), invalid))
    monkeypatch.setattr("gpuwm.member_index.paced_urlopen", lambda *a, **k: next(responses))
    with pytest.raises(ValueError, match="strong ETag|object size"):
        download_indexed_member("https://example.invalid/data", "https://example.invalid/index",
                                tmp_path / "member.grib2", declaration=INDEX,
                                member_ordinal=1, source="ecmwf-ens", member="p01")
    assert not reads
