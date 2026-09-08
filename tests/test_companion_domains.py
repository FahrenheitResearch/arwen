"""Candidate domain removal, reference integrity and WPS survivor identity."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import companion_domains as editor
from gpuwm.hrrr_prepared_bundle import render_wps_namelist
from gpuwm.namelist_import import parse_namelist_text
from gpuwm.toml_document import emit_experiment_toml


def topology(parents=(0, 1, 2)):
    return {"domain": [{"grid_id": index + 1, "parent_id": parent}
                       for index, parent in enumerate(parents)]}


def configured_case(tmp_path, parents=(0, 1, 2)):
    raw = topology(parents)
    raw.update(experiment={"name": "domain-removal-proof", "start_time": datetime(2013, 5, 31, 18),
                           "run_seconds": 3600.0, "restart_interval_s": 0.0},
               shared={"nz": 8, "ztop": 16000.0, "p_top": 10000.0, "hybrid_opt": 2,
                       "eta_levels": [1.0, .9, .75, .6, .45, .3, .2, .1, 0.0]},
               projection={"map_proj": "lambert", "ref_lat": 35.0, "ref_lon": -98.0,
                           "truelat1": 30.0, "truelat2": 60.0, "stand_lon": -98.0},
               case_data={"forcing": ["forcing.grib"], "vtable": "Vtable.ERA5",
                          "wps_namelist": "namelist.wps", "geog_root": "GEOG",
                          "forcing_interval_s": 21600.0, "output_domain": 1,
                          "sfcp_to_sfcp": True, "output_title": "Domain removal metadata proof"})
    for row in raw["domain"]:
        root = row["parent_id"] == 0
        row.update(nx=180 if root else 72, ny=160 if root else 72,
                   i_parent_start=1 if root else 20, j_parent_start=1 if root else 20,
                   parent_grid_ratio=1 if root else 3, parent_time_step_ratio=1 if root else 3,
                   specified=root, nested=not root, history_interval_s=60.0)
        if root:
            row.update(dx=12000.0, time_step=60)
    source = tmp_path / "source.toml"
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    exp = editor._build(raw, source)
    wps = render_wps_namelist(exp).replace("geog_data_res     = 'default',", "geog_data_res = " +
        ", ".join(repr(value) for value in ("5m", "default", "modis_lai+default", "5m+modis_lai")[:len(parents)]) + ",")
    (tmp_path / "namelist.wps").write_text(wps, encoding="utf-8")
    return source, raw


def request_for(source, grid_id, include_children=False):
    return {"schema": editor.REQUEST_SCHEMA, "config_path": str(source),
            "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_path": str(source.parent / "candidate.toml"),
            "action": {"kind": "remove_nest", "grid_id": grid_id, "include_children": include_children}}


def edit_action(source, action):
    request = request_for(source, action["grid_id"])
    request["action"] = action
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    return result, candidate


def test_move_root_translates_native_tree_and_preserves_domain_settings(tmp_path):
    source, original = configured_case(tmp_path)
    original_bytes = source.read_bytes()
    old = editor.native_domain_outlines(editor._build(original, source))
    result, candidate = edit_action(source, {"kind": "move_domain", "grid_id": 1,
                                            "latitude": 38.0, "longitude": -94.0})
    assert candidate["domain"] == original["domain"]
    assert candidate["shared"] == original["shared"]
    assert candidate["projection"] == dict(original["projection"], ref_lat=38.0, ref_lon=-94.0)
    outlines = result["configuration"]["domains"]
    assert outlines[0]["center_latlon"] == pytest.approx([38.0, -94.0], abs=1e-8)
    assert all(before["center_latlon"] != after["center_latlon"] for before, after in zip(old, outlines))
    assert source.read_bytes() == original_bytes
    assert result["forecast_started"] is False


@pytest.mark.parametrize("grid_id", [1, 2, 3])
def test_existing_domain_output_edit_round_trips_query_and_preserves_other_settings(tmp_path, grid_id):
    from gpuwm.companion_query import inspect_configuration
    source, original = configured_case(tmp_path)
    before = source.read_bytes()
    result, candidate = edit_action(source, {"kind": "set_output", "grid_id": grid_id,
        "history_interval_s": 300., "restart_interval_s": None})
    expected = deepcopy(original)
    expected["domain"][grid_id - 1]["history_interval_s"] = 300.
    assert candidate["domain"] == expected["domain"]
    assert candidate["shared"] == original["shared"]
    assert candidate["experiment"] == original["experiment"]
    assert candidate["projection"] == original["projection"]
    queried = inspect_configuration(Path(result["config_path"]))
    expected_intervals = [row["history_interval_s"] for row in expected["domain"]]
    assert [row["history_interval_s"] for row in queried["domains"]] == expected_intervals
    assert [row["history_interval_s"] for row in result["configuration"]["domains"]] == expected_intervals
    assert source.read_bytes() == before
    assert result["forecast_started"] is False


@pytest.mark.parametrize("checkpoint", [0., 600.])
def test_output_edit_exposes_global_restart_policy_without_changing_other_domains(tmp_path, checkpoint):
    source, original = configured_case(tmp_path)
    result, candidate = edit_action(source, {"kind": "set_output", "grid_id": 2,
        "history_interval_s": 300., "restart_interval_s": checkpoint})
    assert candidate["experiment"] == dict(original["experiment"], restart_interval_s=checkpoint)
    assert result["configuration"]["experiment"]["restart_interval_s"] == checkpoint
    assert all(row["run"]["restart_interval_s"] == checkpoint for row in result["domains"])
    assert candidate["domain"][0] == original["domain"][0]
    assert candidate["domain"][2] == original["domain"][2]


@pytest.mark.parametrize("grid_id,history,restart,match", [
    (1, 0., None, "history_interval_s"),
    (1, 61., None, "whole number"),
    (2, 21., None, "whole number"),
    (1, 300., 61., "whole number"),
    (1, 300., -1., "non-negative"),
    (1, True, None, "finite number"),
])
def test_invalid_output_cadence_uses_engine_admission_and_publishes_nothing(tmp_path, grid_id, history, restart, match):
    source, _ = configured_case(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError, match=match):
        edit_action(source, {"kind": "set_output", "grid_id": grid_id,
                            "history_interval_s": history, "restart_interval_s": restart})
    assert source.read_bytes() == before
    assert not (source.parent / "candidate.toml").exists()
    assert not (source.parent / "candidate.namelist.wps").exists()


@pytest.mark.parametrize("field", ["history_interval_s", "restart_interval_s"])
def test_output_edit_rejects_subsecond_file_aliasing_even_on_a_legal_model_step(tmp_path, field):
    source, raw = configured_case(tmp_path)
    raw["domain"][0].update(time_step=0, time_step_fract_num=1, time_step_fract_den=2)
    source.write_text(emit_experiment_toml(raw))
    action = {"kind": "set_output", "grid_id": 1, "history_interval_s": 60., field: .5}
    with pytest.raises(ValueError, match="whole number of seconds"):
        edit_action(source, action)
    assert not (source.parent / "candidate.toml").exists()


def test_output_edit_preserves_existing_weather_tracker_cadence_admission(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[1] / "configs/moving_nest_20110427_follow_2km.toml"
    source = tmp_path / "following.toml"
    monkeypatch.setenv("GPUWM_DEMO_20110427_ROOT", str(tmp_path))
    source.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="whole multiple of history_interval_s"):
        edit_action(source, {"kind": "set_output", "grid_id": 1, "history_interval_s": 1800.})
    assert not (source.parent / "candidate.toml").exists()


@pytest.mark.parametrize("mode", ["off", "auto", "on"])
def test_global_tile_mode_preserves_explicit_tile_controls_and_reports_saved_mode(tmp_path, mode):
    from gpuwm.companion_query import inspect_configuration
    source, raw = configured_case(tmp_path)
    raw["tiles"] = {"mode": "on", "vram_budget_bytes": 1024**3, "host_budget_bytes": 4*1024**3,
                    "store": "host", "write_mode": "ring", "pipeline": "prefetch"}
    if mode == "on":
        raw["tiles"].update(tile_nx=32, tile_ny=32, nbuffers=2)
    source.write_text(emit_experiment_toml(raw))
    before = source.read_bytes()
    request = request_for(source, 1)
    request["action"] = {"kind": "set_tiles", "mode": mode}
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    assert candidate["tiles"] == dict(raw["tiles"], mode=mode)
    assert candidate["domain"] == raw["domain"] and candidate["shared"] == raw["shared"]
    assert result["configuration"]["tiles"]["mode"] == mode
    assert inspect_configuration(Path(result["config_path"]))["tiles"]["mode"] == mode
    assert result["validation"]["forecast_or_memory_admission"] == "not_run"
    assert source.read_bytes() == before


def test_tile_query_exposes_engine_off_default_and_invalid_mode_is_not_published(tmp_path):
    from gpuwm.companion_query import inspect_configuration
    source, _ = configured_case(tmp_path)
    assert inspect_configuration(source)["tiles"]["mode"] == "off"
    request = request_for(source, 1)
    request["action"] = {"kind": "set_tiles", "mode": "automatic"}
    with pytest.raises(ValueError, match="off, auto or on"):
        editor.edit_configuration(request)
    assert not (source.parent / "candidate.toml").exists()


@pytest.mark.parametrize("mode", ["off", "auto"])
def test_changing_explicit_tile_pins_keeps_engine_refusal_and_original_settings(tmp_path, mode):
    source, raw = configured_case(tmp_path)
    raw["tiles"] = {"mode": "on", "tile_nx": 32, "tile_ny": 32}
    source.write_text(emit_experiment_toml(raw))
    before = source.read_bytes()
    request = request_for(source, 1)
    request["action"] = {"kind": "set_tiles", "mode": mode}
    with pytest.raises(ValueError, match="surface that is off must be empty|while mode = 'auto'"):
        editor.edit_configuration(request)
    assert source.read_bytes() == before
    assert not (source.parent / "candidate.toml").exists()


def test_move_nested_grid_clamps_to_parent_and_keeps_descendants(tmp_path):
    source, original = configured_case(tmp_path)
    result, candidate = edit_action(source, {"kind": "move_domain", "grid_id": 2,
                                            "latitude": 48.0, "longitude": -70.0})
    exp = editor._build(candidate, Path(result["config_path"]))
    child, parent = exp.domain(2), exp.domain(1)
    assert child.i_parent_start == parent.run.nx - child.run.nx // 3 - 10 + 1
    assert child.j_parent_start == parent.run.ny - child.run.ny // 3 - 10 + 1
    assert candidate["domain"][0] == original["domain"][0]
    assert candidate["domain"][2] == original["domain"][2]
    assert candidate["shared"] == original["shared"]


def test_resize_root_keeps_child_sizes_and_clamps_their_placement(tmp_path):
    source, original = configured_case(tmp_path)
    result, candidate = edit_action(source, {"kind": "resize_domain", "grid_id": 1,
        "bounds": {"south": 34.99, "west": -98.01, "north": 35.01, "east": -97.99}})
    exp = editor._build(candidate, Path(result["config_path"]))
    assert (exp.domain(1).run.nx, exp.domain(1).run.ny) == (44, 44)
    assert (exp.domain(2).i_parent_start, exp.domain(2).j_parent_start) == (11, 11)
    for before, after in zip(original["domain"], candidate["domain"]):
        assert {k: v for k, v in before.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")} == {
            k: v for k, v in after.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")}
    assert [(row["nx"], row["ny"]) for row in candidate["domain"][1:]] == [(72, 72), (72, 72)]
    assert candidate["shared"] == original["shared"]
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())
    assert wps["geogrid"]["e_we"] == [45, 73, 73]
    assert wps["geogrid"]["geog_data_res"] == ["5m", "default", "modis_lai+default"]


def test_resize_nested_grid_snaps_ratio_and_stops_at_parent_boundary(tmp_path):
    source, original = configured_case(tmp_path)
    result, candidate = edit_action(source, {"kind": "resize_domain", "grid_id": 2,
        "bounds": {"south": 20.0, "west": -120.0, "north": 50.0, "east": -70.0}})
    exp = editor._build(candidate, Path(result["config_path"]))
    assert (exp.domain(2).run.nx, exp.domain(2).run.ny) == (480, 420)
    assert (exp.domain(2).i_parent_start, exp.domain(2).j_parent_start) == (11, 11)
    assert candidate["domain"][0] == original["domain"][0]
    assert candidate["domain"][2] == original["domain"][2]
    assert result["configuration"]["geometry_backend"] == "rust-static-fields"


def native_edge_action(raw, path, grid_id, handle, di=0, dj=0):
    bridge, grids = editor._native_grids(editor._build(raw, path))
    grid = grids[grid_id]
    start = (grid.e_we / 2, grid.e_sn / 2)
    points = editor._transform(bridge, grid, 0, [start, (start[0] + di, start[1] + dj)])
    return {"kind": "resize_domain_edges", "grid_id": grid_id, "handle": handle,
            "start": dict(zip(("latitude", "longitude"), points[0])),
            "end": dict(zip(("latitude", "longitude"), points[1]))}


@pytest.mark.parametrize("grid_id", [1, 2, 3])
@pytest.mark.parametrize("handle", ["nw", "n", "ne", "e", "se", "s", "sw", "w"])
def test_zero_pointer_resize_preserves_exact_native_configuration(tmp_path, grid_id, handle):
    source, raw = configured_case(tmp_path)
    original = deepcopy(raw)
    editor._apply(raw, native_edge_action(raw, source, grid_id, handle), source)
    assert raw == original


@pytest.mark.parametrize("grid_id,handle,di,expected_nx,expected_start", [
    (1, "e", 12., 192, 1),
    (2, "e", 8., 81, 20),
    (2, "w", 8., 63, 23),
    (2, "e", 1000., 453, 20),
    (2, "w", -1000., 99, 11),
])
def test_native_edge_resize_snaps_and_clamps_without_moving_the_opposite_anchor(tmp_path, grid_id, handle, di, expected_nx, expected_start):
    source, original = configured_case(tmp_path)
    action = native_edge_action(original, source, grid_id, handle, di=di, dj=7.)
    result, candidate = edit_action(source, action)
    before = editor._build(original, source)
    after = editor._build(candidate, Path(result["config_path"]))
    domain = after.domain(grid_id)
    assert (domain.run.nx, domain.run.ny, domain.i_parent_start) == (expected_nx, before.domain(grid_id).run.ny, expected_start)
    old_bridge, old_grids = editor._native_grids(before)
    new_bridge, new_grids = editor._native_grids(after)
    old_grid, new_grid = old_grids[grid_id], new_grids[grid_id]
    old_x = .5 if handle == "e" else before.domain(grid_id).run.nx + .5
    new_x = .5 if handle == "e" else domain.run.nx + .5
    fixed_before = editor._transform(old_bridge, old_grid, 0, [(old_x, .5), (old_x, domain.run.ny + .5)])
    fixed_after = editor._transform(new_bridge, new_grid, 0, [(new_x, .5), (new_x, domain.run.ny + .5)])
    for old, new in zip(fixed_before, fixed_after):
        assert new == pytest.approx(old, abs=1e-8)
    assert candidate["shared"] == original["shared"]
    for old, new in zip(original["domain"], candidate["domain"]):
        assert {k: v for k, v in old.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")} == {
            k: v for k, v in new.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")}


@pytest.mark.parametrize("action", [
    {"kind": "resize_domain", "grid_id": 1, "bounds": {"south": 40, "north": 30, "west": -100, "east": -90}},
    {"kind": "resize_domain", "grid_id": 1, "bounds": {"south": 30, "north": 40, "west": -100, "east": 100}},
    {"kind": "move_domain", "grid_id": 1, "latitude": 90, "longitude": -100},
])
def test_invalid_drag_leaves_source_and_candidate_untouched(tmp_path, action):
    source, _ = configured_case(tmp_path)
    before = source.read_bytes()
    with pytest.raises(ValueError):
        edit_action(source, action)
    assert source.read_bytes() == before
    assert not (source.parent / "candidate.toml").exists()
    assert not (source.parent / "candidate.namelist.wps").exists()


def test_leaf_removal_publishes_native_candidate_without_changing_original(tmp_path):
    source, original = configured_case(tmp_path)
    before_config = source.read_bytes()
    before_wps = (tmp_path / "namelist.wps").read_bytes()
    result = editor.edit_configuration(request_for(source, 3))
    assert result["created"] is True and result["forecast_started"] is False
    assert [row["grid_id"] for row in result["domains"]] == [1, 2]
    assert [row["parent_id"] for row in result["configuration"]["domains"]] == [0, 1]
    assert all(row["perimeter_latlon"] for row in result["configuration"]["domains"])
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    assert candidate["domain"] == original["domain"][:2]
    assert candidate["case_data"]["wps_namelist"] == result["wps_path"]
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())
    assert wps["share"]["max_dom"] == [2]
    assert wps["geogrid"]["geog_data_res"] == ["5m", "default"]
    assert source.read_bytes() == before_config
    assert (tmp_path / "namelist.wps").read_bytes() == before_wps
    assert json.loads(Path(result["receipt_path"]).read_text())["config_sha256"] == hashlib.sha256(Path(result["config_path"]).read_bytes()).hexdigest()


def test_subtree_requires_explicit_inclusion_and_then_preserves_root(tmp_path):
    source, _ = configured_case(tmp_path)
    with pytest.raises(ValueError, match="explicitly include its children"):
        editor.edit_configuration(request_for(source, 2))
    assert not (tmp_path / "candidate.toml").exists()
    assert not (tmp_path / "candidate.namelist.wps").exists()
    result = editor.edit_configuration(request_for(source, 2, True))
    assert [row["grid_id"] for row in result["domains"]] == [1]
    assert parse_namelist_text(Path(result["wps_path"]).read_text())["geogrid"]["geog_data_res"] == ["5m"]


@pytest.mark.parametrize("grid_id,include_children,match", [(1, True, "root domain"), (9, False, "no domain"), (3, 1, "boolean")])
def test_invalid_removal_preserves_the_tree(grid_id, include_children, match):
    raw = topology()
    before = deepcopy(raw)
    with pytest.raises(ValueError, match=match):
        editor._remove_nest(raw, grid_id, include_children)
    assert raw == before


@pytest.mark.parametrize("reference", ["domain-follow", "relocation-follow", "relocation-containment", "output-domain"])
def test_surviving_references_refuse_with_an_action(reference):
    raw = topology()
    if reference == "domain-follow":
        raw["domain"][1]["follow"] = {"refine_grid_id": 3}
    elif reference == "output-domain":
        raw["case_data"] = {"output_domain": 3}
    else:
        table, key = ("follow", "refine_grid_id") if reference.endswith("follow") else ("containment", "grid_id")
        raw["relocation"] = {"grid_id": 2, table: {key: 3}}
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="Settings first"):
        editor._remove_nest(raw, 3, False)
    assert raw == before


def test_removing_its_owner_removes_relocation_without_touching_survivors():
    raw = topology()
    raw["relocation"] = {"grid_id": 2, "follow": {"refine_grid_id": 3}, "track": {"output": "track.csv"}}
    editor._remove_nest(raw, 2, True)
    assert raw == {"domain": [{"grid_id": 1, "parent_id": 0}]}


def test_middle_removal_preserves_stable_survivor_ids():
    raw = topology((0, 1, 1, 3))
    before = deepcopy(raw)
    editor._remove_nest(raw, 2, False)
    assert raw["domain"] == [row for row in before["domain"] if row["grid_id"] != 2]


def test_middle_removal_round_trips_native_geometry_geography_and_later_edit(tmp_path):
    from gpuwm.native_wrf_contract import validate_native_lambert_contracts
    from gpuwm.static.build import GeogSelection
    from gpuwm.source_hierarchy import _validated_static_one_way_topology
    from gpuwm.wps_domain_ids import domain_ids_from_wps_text
    from types import SimpleNamespace

    source, original = configured_case(tmp_path, (0, 1, 1, 3))
    original_outlines = editor.native_domain_outlines(editor._build(original, source))
    result = editor.edit_configuration(request_for(source, 2))
    candidate_path = Path(result["config_path"])
    candidate = tomllib.loads(candidate_path.read_text())
    exp = editor._build(candidate, candidate_path)
    assert candidate["domain"] == [row for row in original["domain"] if row["grid_id"] != 2]
    assert result["configuration"]["domains"] == [row for row in original_outlines if row["grid_id"] != 2]
    wps_path = Path(result["wps_path"])
    text = wps_path.read_text()
    assert domain_ids_from_wps_text(text, 3) == (1, 3, 4)
    assert parse_namelist_text(text)["geogrid"]["parent_id"] == [1, 1, 2]
    grids = validate_native_lambert_contracts(exp, wps_path, source_name="ERA5")
    topology_receipt = _validated_static_one_way_topology(exp, grids)
    assert [row["grid_id"] for row in topology_receipt["domains"]] == [1, 3, 4]
    data = SimpleNamespace(geog_root=tmp_path / "GEOG", wps_namelist=wps_path)
    assert [GeogSelection.from_case_data(data, domain_id=grid_id).resolution_tokens
            for grid_id in (1, 3, 4)] == [("5m",), ("modis_lai", "default"), ("5m", "modis_lai")]
    with pytest.raises(ValueError, match="absent"):
        GeogSelection.from_case_data(data, domain_id=2)
    # A later edit must interpret the candidate's compact slots through its
    # declared stable IDs, including a custom selection on the final survivor.
    next_request = request_for(candidate_path, 4)
    next_request["output_path"] = str(tmp_path / "next.toml")
    next_result = editor.edit_configuration(next_request)
    next_text = Path(next_result["wps_path"]).read_text()
    assert domain_ids_from_wps_text(next_text, 2) == (1, 3)
    assert parse_namelist_text(next_text)["geogrid"]["geog_data_res"] == ["5m", "modis_lai+default"]


def test_edit_refuses_ambiguous_original_wps_identity_without_publishing(tmp_path):
    source, _ = configured_case(tmp_path, (0, 1, 1, 3))
    path = tmp_path / "namelist.wps"
    path.write_text("! GPUWM_DOMAIN_IDS_V1 = 1,3,2,4\n" + path.read_text())
    with pytest.raises(ValueError, match="original WPS domain identity differs"):
        editor.edit_configuration(request_for(source, 2))
    assert not (tmp_path / "candidate.toml").exists()


@pytest.mark.parametrize("identity", ["", "! GPUWM_DOMAIN_IDS_V1 = 1,4,3\n"])
def test_native_admission_rejects_missing_or_reordered_stable_identity(tmp_path, identity):
    from gpuwm.native_wrf_contract import validate_native_lambert_contracts
    source, _ = configured_case(tmp_path, (0, 1, 1, 3))
    result = editor.edit_configuration(request_for(source, 2))
    path = Path(result["wps_path"])
    text = path.read_text().split("\n", 1)[1]
    path.write_text(identity + text)
    config = Path(result["config_path"])
    exp = editor._build(tomllib.loads(config.read_text()), config)
    with pytest.raises(ValueError, match="domain identity mismatch"):
        validate_native_lambert_contracts(exp, path, source_name="ERA5")


def test_parent_identity_cannot_hide_behind_identical_sibling_geometry(tmp_path):
    from gpuwm.native_wrf_contract import validate_native_lambert_contracts
    source, raw = configured_case(tmp_path, (0, 1, 1, 3))
    exp = editor._build(raw, source)
    path = tmp_path / "namelist.wps"
    original = path.read_text()
    assert "parent_id         = 1, 1, 1, 3," in original
    # d02 and d03 share this fixture's geometry, so moving d04's declared
    # parent between them leaves every numeric grid coordinate identical.
    path.write_text(original.replace("parent_id         = 1, 1, 1, 3,", "parent_id         = 1, 1, 1, 2,"))
    with pytest.raises(ValueError, match="parent-slot identity mismatch"):
        validate_native_lambert_contracts(exp, path, source_name="ERA5")


def test_wps_geography_selection_follows_surviving_ids_not_array_truncation(tmp_path):
    source, raw = configured_case(tmp_path, (0, 1, 1, 3))
    survivors = deepcopy(raw)
    survivors["domain"] = [row for row in survivors["domain"] if row["grid_id"] != 2]
    exp = editor._build(survivors, source)
    wps = editor._wps_text(exp, tmp_path / "namelist.wps", tmp_path / "edited.wps", survivors, 4,
                           original_domain_ids=[1, 2, 3, 4])
    assert [row.grid_id for row in exp.domains] == [1, 3, 4]
    assert parse_namelist_text(wps)["geogrid"]["geog_data_res"] == ["5m", "modis_lai+default", "5m+modis_lai"]


def test_unassigned_wps_geography_uses_fortran_default(tmp_path):
    source, raw = configured_case(tmp_path)
    (tmp_path / "namelist.wps").write_text(render_wps_namelist(editor._build(raw, source)).replace("'default'", "'modis_lai'"))
    wps = editor._wps_text(editor._build(raw, source), tmp_path / "namelist.wps", tmp_path / "edited.wps", raw, 3)
    assert parse_namelist_text(wps)["geogrid"]["geog_data_res"] == ["modis_lai", "default", "default"]


def test_physics_edit_changes_only_selected_domain_and_keeps_native_geometry(tmp_path):
    source, original = configured_case(tmp_path)
    before = source.read_bytes()
    request = request_for(source, 2)
    request["action"] = {"kind": "set_physics", "grid_id": 2, "settings": {"radt_minutes": 15.0}}
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    assert candidate["domain"][1]["radt_minutes"] == 15.0
    assert candidate["domain"][0] == original["domain"][0]
    assert candidate["domain"][2] == original["domain"][2]
    assert result["configuration"]["domains"] == editor.native_domain_outlines(editor._build(original, source))
    assert source.read_bytes() == before
    assert result["forecast_started"] is False


def test_all_domain_physics_edit_and_invalid_setting_do_not_publish_partial_change(tmp_path):
    source, _ = configured_case(tmp_path)
    request = request_for(source, 1)
    request["action"] = {"kind": "set_physics", "grid_id": 0, "settings": {"radt_minutes": 15.0}}
    result = editor.edit_configuration(request)
    assert all(row["run"]["radt_minutes"] == 15.0 for row in result["domains"])
    request["output_path"] = str(tmp_path / "invalid.toml")
    request["action"]["settings"] = {"radt_minutes": -1.0}
    with pytest.raises(ValueError):
        editor.edit_configuration(request)
    assert not (tmp_path / "invalid.toml").exists()


def test_physics_menu_uses_registered_implemented_choices():
    from gpuwm.physics_registry import physics_registry
    registry = physics_registry()
    components = editor.physics_components()
    assert components
    for component in components:
        for option in component["options"]:
            native = registry["components"][component["id"]]["options"][option["registry_option_id"]]
            assert native["implemented"] is True
            assert all(option["selectors"][key] == value for key,value in native["selectors"].items())
            if "ra_rrtmg_variant" in option["selectors"]:
                assert option["settings"]["ra_rrtmg_variant"] in registry["parameters"]["ra_rrtmg_variant"]["enum"]
                assert option["settings"]["wrf_rrtmg_compatibility"] == "none"
            else:
                assert option["label"] == native["label"]


def test_shared_physics_controls_do_not_claim_per_domain_support(tmp_path):
    source, original = configured_case(tmp_path)
    request = request_for(source, 1)
    request["action"] = {"kind":"set_physics", "grid_id":1, "settings":{"icloud":0}}
    with pytest.raises(ValueError, match="Select All domains"):
        editor.edit_configuration(request)
    request["action"]["grid_id"] = 0
    result = editor.edit_configuration(request)
    candidate = tomllib.loads(Path(result["config_path"]).read_text())
    assert candidate["shared"]["icloud"] == 0
    assert candidate["domain"] == original["domain"]
