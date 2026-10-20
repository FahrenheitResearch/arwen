"""Track selection, provenance and placement through the real setup door."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from gpuwm import cyclone_setup as tc
from gpuwm import cyclone_tracks as tracks
from gpuwm import domain_wizard as dw
from gpuwm.cyclone_seed import _distance_km
from gpuwm.source_coverage import RegularLatLonWindow

FIXTURE = Path(__file__).parent / "fixtures/cyclone-tracks.json"
START = datetime(2026, 9, 9)
CYCLE = "2026090900"
SOURCE = tc.DEFAULT_SOURCE


def select(path=FIXTURE, **kwargs):
    return tracks.select_track(path, **({"storm": "auto", "member": None,
        "source": SOURCE, "start": START, "hours": 72} | kwargs))


def save(tmp_path, mutate):
    doc = json.loads(FIXTURE.read_text())
    mutate(doc)
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(doc))
    return path


def phases(peak):
    return SimpleNamespace(peak_envelope_bytes=int(peak), binding_phase="forecast",
        verdict=lambda budget: f"fixture peak {peak} exceeds budget {budget}",
        forecast=SimpleNamespace(fixed_envelope_bytes=dw.GIB), tree_road=None)


@pytest.fixture
def admission(monkeypatch):
    budget = dw.SizingBudget(96., 94*dw.GIB, None, "declared fixture", measured=False)
    monkeypatch.setattr(dw, "_domain_target_hardware", lambda args: (budget, None, False))
    monkeypatch.setattr(dw, "_sizing_phases", lambda *a, **k: phases(dw.GIB))
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *a, **k: 90*dw.GIB)
    return budget


def plan(admission, selection=None, **kwargs):
    selection = select(storm="system-a") if selection is None else selection
    f = selection["lineage"]["center_fix"]
    return tc.plan_cyclone(cycle=CYCLE, point=(f["lat"], f["lon"]), hours=72,
                           tiles="off", sizing=admission, track_selection=selection, **kwargs)


def parser():
    result = argparse.ArgumentParser()
    tc.register_cli(result.add_subparsers())
    return result


def test_auto_selects_cycle_strength_and_records_near_tie():
    row = select()["lineage"]
    assert row["track_id"] == "system-b"
    assert row["runner_up"] == {"track_id": "system-a", "cycle_mslp_pa": 99000., "minimum_mslp_pa": 95000.}
    assert row["near_ties"] == []
    assert row["selection_rule"].startswith("lowest-cycle-mslp")
    assert (row["center_fix"]["lat"], row["center_fix"]["lon"]) == (19., -70.)
    assert row["tracks_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert row["tracker"] == json.loads(FIXTURE.read_text())["tracker"]


def test_search_bounds_filter_cycle_position_before_strength_and_not_future_fixes():
    row = select(search_area=[19.5, -87, 20.5, -85])["lineage"]
    assert row["track_id"] == "system-a"
    assert row["rejected"] == [{"track_id": "system-b", "reason": "track-outside-search-area"}]
    assert row["search_area"] == [19.5, -87, 20.5, -85]
    assert "window_start" in row["search_area_basis"]
    with pytest.raises(ValueError, match="track-outside-search-area"):
        select(storm="system-b", search_area=[19.5, -87, 20.5, -85])
    args = parser().parse_args(["cyclone-setup", "--tracks", str(FIXTURE),
        "--storm", "auto", "--search-area=19.5,-87,20.5,-85"])
    assert args.search_area == row["search_area"]


def test_explicit_id_and_window_boundaries():
    assert select(storm="system-b")["lineage"]["track_id"] == "system-b"
    # Interpolated boundary pressure participates in the inclusive window.
    assert select(hours=6)["lineage"]["track_id"] == "system-b"
    later = select(start=datetime(2026, 9, 10), hours=48)
    assert later["lineage"]["center_fix"]["lead_hours"] == 24


def test_ties_are_deterministic_and_visible(tmp_path):
    def tied(doc):
        doc["tracks"][1]["fixes"][0]["mslp_pa"] = doc["tracks"][0]["fixes"][0]["mslp_pa"]
        doc["tracks"].reverse()
    row = select(save(tmp_path, tied))["lineage"]
    assert row["track_id"] == "system-a"
    assert row["near_ties"][0]["cycle_mslp_pa"] == 99000
    assert "tie break" in row["selection_rule"]


@pytest.mark.parametrize("kwargs,reason", [
    ({"member": "absent"}, "track-member-not-found"),
    ({"storm": "absent"}, "track-id-not-found"),
    ({"start": datetime(2026, 9, 8)}, "track-missing-start-fix"),
])
def test_named_refusals(kwargs, reason, monkeypatch):
    from gpuwm import cyclone_catalog
    monkeypatch.setattr(cyclone_catalog, "match_explicit", lambda *args: None)
    with pytest.raises(ValueError, match=reason):
        select(**kwargs)


def test_multi_member_needs_an_explicit_member(tmp_path):
    def another(doc):
        row = deepcopy(doc["members"][0])
        row["member_id"] = "member-02"
        doc["members"].append(row)
    path = save(tmp_path, another)
    with pytest.raises(ValueError, match="track-member-required"):
        select(path)
    assert select(path, member="member-01")["lineage"]["member_id"] == "member-01"


def test_regional_coverage_comes_from_the_selected_source(monkeypatch):
    import gpuwm.cyclone_sources as sources
    original = sources.source_adapter
    def adapter(source):
        row = original(source)
        return SimpleNamespace(coverage_window=RegularLatLonWindow(10., -90., 40., -80.),
                               display_title=row.display_title)
    monkeypatch.setattr(sources, "source_adapter", adapter)
    row = select()["lineage"]
    assert row["track_id"] == "system-a"
    assert row["rejected"] == [{"track_id": "system-b", "reason": "track-outside-source-coverage"}]
    with pytest.raises(ValueError, match="track-outside-source-coverage"):
        select(storm="system-b")


def test_great_circle_interpolation_short_arc_and_missing_fixes():
    fixes = deepcopy(json.loads(FIXTURE.read_text())["tracks"][1]["fixes"])
    for f in fixes:
        f["lat"] = 20.
    fixes[0]["lon"], fixes[1]["lon"] = 179., -179.
    center, interpolation = tracks.fix_at(fixes, datetime(2026, 9, 10, 12))
    assert abs(abs(center["lon"])-180.) < 1e-8
    assert center["lat"] > 20.
    assert interpolation["method"] == "great-circle"
    assert tracks.fix_at([fixes[0], None, fixes[1]], datetime(2026, 9, 10, 12))[0] is None
    assert tracks.fix_at([fixes[0], None, fixes[1]], datetime(2026, 9, 12))[0] == fixes[1]


def test_csv_twin_and_standalone_csv_provenance(tmp_path):
    twin = select(FIXTURE.with_suffix(".csv"))
    assert twin["fixes"] == select()["fixes"]
    assert twin["lineage"]["tracker"] == select()["lineage"]["tracker"]
    assert "metadata_input" in twin["lineage"]
    path = tmp_path / "standalone.csv"
    path.write_bytes(FIXTURE.with_suffix(".csv").read_bytes())
    row = select(path)["lineage"]
    assert row["tracker"] == {}
    assert row["metadata_status"] == "not-supplied-by-standalone-csv"


def test_parent_growth_is_reviewed_and_contains_whole_follower(admission):
    result = plan(admission)
    assert result["kind"] == "proposal" and result["fitting"]["review_required"]
    assert result["domains"][0]["ny"] > tc.ROOT_DIMS[1]
    assert result["track_placement"]["all_fixes_inside_parent"]
    assert result["track_placement"]["all_fixes_with_follower_clearance"]
    assert result["point"] == [20., -86.]
    assert _distance_km(*result["track_placement"]["initial_nest_center"], result["point"]) < 10.
    raw = tomllib.loads(result["config_text"])
    assert raw["lineage"]["cyclone_tracks"] == tracks.toml_lineage(result["lineage"]["cyclone_tracks"])
    ordinary, _ = tc.configuration_text(cycle=CYCLE, point=(20., -86.), hours=72)
    assert raw["domain"][1]["follow"] == tomllib.loads(ordinary)["domain"][1]["follow"]
    assert raw["projection"]["ref_lat"] > 20.


def test_memory_bound_corridor_reports_departing_fixes(admission, monkeypatch):
    monkeypatch.setattr(dw, "_sizing_phases", lambda exp, **k:
                        phases((100 if exp.root.run.ny > 150 else 1)*dw.GIB))
    result = plan(admission)
    assert result["fitting"]["review_required"]
    assert result["track_placement"]["parent_center"] == [20., -86.]
    assert result["track_placement"]["leaves_parent"][-1]["valid"] == "2026-09-12T00:00:00Z"
    assert "start fix" in result["fitting"]["notice"]


def test_two_point_five_km_nest_uses_ten_km_parent(admission):
    result = plan(admission, nest_km=2.5)
    assert [d["dx_m"] for d in result["domains"]] == [10000., 2500.]
    assert result["nest"]["ratio"] == 4
    assert result["track_placement"]["all_fixes_with_follower_clearance"]


def test_cli_review_acceptance_publishes_lineage(admission, tmp_path, capsys):
    out = tmp_path / "setup.toml"
    args = parser().parse_args(["cyclone-setup", "--cycle", CYCLE,
        "--tracks", str(FIXTURE.with_suffix('.csv')), "--storm", "auto", "--member", "member-01",
        "--hours", "72", "--card", "96gb", "--tiles", "off", "--out", str(out)])
    assert tc.main(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert not out.exists() and preview["created"] is False
    args.accept_fit = preview["fitting"]["fit_id"]
    assert tc.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["created"] is True
    assert result["seed"]["method"] == "track"
    receipt = json.loads(out.with_suffix('.cyclone.json').read_text())
    assert receipt["lineage"] == result["lineage"]
    assert receipt["forecast_started"] is False
    assert tomllib.loads(out.read_text())["lineage"]["cyclone_tracks"]["member_id"] == "member-01"


def test_cli_refuses_unknown_member_before_hardware_probe(monkeypatch, capsys):
    def forbidden(*a, **k):
        raise AssertionError("invalid selection must not probe hardware")
    monkeypatch.setattr(dw, "_domain_target_hardware", forbidden)
    args = parser().parse_args(["cyclone-setup", "--cycle", CYCLE, "--tracks", str(FIXTURE),
                               "--storm", "auto", "--member", "absent"])
    assert tc.main(args) == 1
    assert "track-member-not-found" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("flag", ["--point=20,-86", "--advisory-position=20,-86"])
def test_tracks_exclude_manual_positions(flag, capsys):
    args = parser().parse_args(["cyclone-setup", "--cycle", CYCLE,
        "--tracks", str(FIXTURE), "--storm", "auto", flag])
    assert tc.main(args) == 1
    assert "track-position-conflict" in json.loads(capsys.readouterr().out)["error"]
    # Preserve the pre-existing combination of explicit and advisory positions.
    manual = parser().parse_args(["cyclone-setup", "--point=20,-86", "--advisory-position=21,-86"])
    assert manual.point == "20,-86" and manual.advisory_position == "21,-86"


def test_explicit_storm_identity_and_ambiguous_names(tmp_path):
    def labelled(doc):
        doc["tracks"][0].update(atcf_id="AL012026", name="Example", aliases=["Storm One"])
    path = save(tmp_path, labelled)
    for name in ("al012026", "EXAMPLE", "storm one"):
        assert select(path, storm=name)["lineage"]["track_id"] == "system-a"
    doc = json.loads(path.read_text())
    doc["tracks"][1]["name"] = "Example"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="track-storm-ambiguous"):
        select(path, storm="Example")


def test_no_cycle_pressure_is_not_ranked_by_future_strength(tmp_path):
    def missing(doc):
        doc["tracks"][0]["fixes"][0]["mslp_pa"] = None
    selection = select(save(tmp_path, missing))
    assert selection["lineage"]["track_id"] == "system-b"
    assert selection["lineage"]["rejected"] == [
        {"track_id": "system-a", "reason": "track-no-cycle-pressure"}]


def test_explicit_catalogue_identity_keeps_model_fixes(tmp_path, monkeypatch):
    from gpuwm import cyclone_catalog as catalog
    import gzip
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    listing = b"EXAMPLE,AL,L,,,,,01,2024,HU,S,2024092312,2024092518,,,,,,ARCHIVE,,AL012024\n"
    best = b"AL, 01, 2024092412, , BEST, 0, 200N, 840W, 50, 990\n"
    def read(url, cache):
        return (listing if url.endswith('storm_list.txt') else gzip.compress(best)), {"url": url, "sha256": "test"}
    monkeypatch.setattr(catalog, "_read", read)
    candidate = {"track_id": "numeric-1", "center_fix": {"lat": 20., "lon": -84.}, "fixes": ["model-forecast"]}
    for token in ("example", "al012024"):
        selected, identity = catalog.match_explicit(token, datetime(2024, 9, 24, 12), [candidate])
        assert selected is candidate and selected["fixes"] == ["model-forecast"]
        assert identity["storm_id"] == "AL012024" and identity["distance_km"] == 0
    with pytest.raises(ValueError, match="position-ambiguous"):
        catalog.match_explicit("EXAMPLE", datetime(2024, 9, 24, 12), [candidate, dict(candidate)])
