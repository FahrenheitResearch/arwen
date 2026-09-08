"""CPU proofs: the complete template is priced, not overlaid after fitting."""
from argparse import Namespace
import copy
import hashlib
from datetime import datetime
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import domain_wizard as dw
from gpuwm import starter_template as st
from gpuwm.experiment import load_experiment


def test_card_names_normalize_and_unknown_tier_names_an_exact_capacity(monkeypatch):
    monkeypatch.setattr(dw, "device_memory_probe_subprocess", lambda: pytest.fail("declared card probed GPU"))
    assert dw.resolve_sizing_budget(" 16GB ", None) == dw.resolve_sizing_budget("16gb", None)
    assert dw.resolve_sizing_budget(None, 8).vram_gib == 8
    with pytest.raises(ValueError, match="--vram-gib 8"):
        dw.resolve_sizing_budget("8gb", None)
    with pytest.raises(ValueError, match="not a named GPU tier"):
        dw.resolve_sizing_budget("not-a-card", None)


def starter(tmp_path, *, nested=False):
    ratios = (3,) if nested else ()
    text = dw.render_config(name="custom-science", start_time=datetime(2026, 9, 5),
        hours=6, projection=dw._projection_entries(40, -100, "auto"),
        dims=dw._dims_for_scale(1, ratios), ratios=ratios,
        fetch_hints=dict(source="gfs", cycle="2026-09-05T00", hours=6,
                         out="data/test", cadence=3), case_data=None)
    raw = tomllib.loads(text)
    raw["shared"]["p_top"] = 4200.0
    raw["shared"]["h_sca_adv_order"] = 2
    raw["domain"][0]["epssm"] = 0.65
    raw["domain"][0]["diff_6th_factor"] = 0.11
    raw["domain"][0]["dx"] = 12125.125
    raw["output"] = {"preset": "minimal"}
    if nested:
        raw["domain"][1]["parent_time_step_ratio"] = 5
        raw["domain"][1]["epssm"] = 0.7
        raw["domain"][1]["diff_6th_factor"] = 0.07
        raw["domain"][1]["output"] = {"preset": "full"}
    path = tmp_path / "starter.toml"
    path.write_text(st.render_tables(raw), encoding="utf-8")
    return path, raw


def args(path, out, **kw):
    options = dict(template=path, out=out, point="40,-100", polygon=None,
                   buffer_km=None, source=None, card=None, vram_gib=16,
                   start_time=None, hours=None, write=False)
    options.update(kw)
    return Namespace(**options)


def test_every_priced_candidate_has_full_custom_authority(tmp_path, monkeypatch):
    path, raw = starter(tmp_path, nested=True)
    original = path.read_bytes()
    real = dw._sizing_phases
    seen = []
    def observe(exp, **kw):
        seen.append(exp)
        assert exp.vertical.p_top == 4200
        assert exp.domains[0].run.h_sca_adv_order == 2
        assert exp.domains[0].run.epssm == 0.65
        assert exp.domains[0].run.diff_6th_factor == 0.11
        assert exp.domains[1].run.epssm == 0.7
        assert exp.domains[1].run.diff_6th_factor == 0.07
        assert exp.domains[1].parent_grid_ratio == 3
        assert exp.domains[1].parent_time_step_ratio == 5
        assert exp.domains[1].output.preset == "full"
        assert exp.domains[0].run.dx == 12125.125
        assert exp.domains[0].time_step == raw["domain"][0]["time_step"]
        return real(exp, **kw)
    monkeypatch.setattr(dw, "_sizing_phases", observe)
    out = tmp_path / "preview" / "resolved.toml"
    assert st.fit_main(args(path, out)) == 0
    assert len(seen) > 10
    assert path.read_bytes() == original
    assert not out.parent.exists()


def test_write_preserves_settings_and_outputs_matching_geometry(tmp_path):
    path, raw = starter(tmp_path)
    out = tmp_path / "new" / "fitted.toml"
    assert st.fit_main(args(path, out, write=True, hours=3, start_time="2026-09-05T06Z")) == 0
    resolved = tomllib.loads(out.read_text(encoding="utf-8"))
    exp = load_experiment(out)
    for key in ("shared", "output"):
        assert resolved[key] == raw[key]
    for key, value in raw["domain"][0].items():
        if key not in {"nx", "ny"}:
            assert resolved["domain"][0][key] == value
    assert exp.run_seconds == 10800
    assert resolved["fetch"]["cycle"] == "2026-09-05T06"
    wps = out.with_suffix(".namelist.wps").read_text()
    assert f"e_we              = {exp.domains[0].run.nx + 1}" in wps
    from gpuwm.namelist_import import parse_namelist_text
    assert parse_namelist_text(wps)["geogrid"]["dx"] == [12125.125]
    proof = json.loads(out.with_suffix(".fit.json").read_text())
    assert proof["peak_envelope_bytes"] <= proof["budget_bytes"]
    assert proof["launch_performed"] is False
    assert proof["template_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert proof["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert proof["wps_sha256"] == hashlib.sha256(out.with_suffix(".namelist.wps").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="never overwrites"):
        st.fit_main(args(path, out, write=True))


def test_polygon_prices_custom_settings_without_shrinking_area(tmp_path):
    path, raw = starter(tmp_path)
    poly = tmp_path / "area.geojson"
    poly.write_text(json.dumps({"type":"Polygon", "coordinates":[[
        [-100.2,39.8],[-99.8,39.8],[-99.8,40.2],[-100.2,40.2],[-100.2,39.8]]]}))
    out = tmp_path / "polygon.toml"
    assert st.fit_main(args(path, out, polygon=poly, point=None, write=True)) == 0
    exp = load_experiment(out)
    assert exp.vertical.p_top == 4200
    dw.verify_polygon_containment(exp, dw.load_polygon_footprint(poly), (0.0,))


def test_serializer_roundtrips_nested_settings_and_literal_strings():
    raw = {"experiment":{"name":'quoted " name \\ and newline\n',
                         "start_time": datetime(2026,9,5)},
           "domain":[{"output":{"preset":"full", "history_drop":["X", "Y"]},
                      "eta_levels":[1.,0.5,0.]}]}
    assert tomllib.loads(st.render_tables(raw)) == raw


@pytest.mark.parametrize("topology", ["two_domains", "stable_chain", "branched_tree"])
@pytest.mark.parametrize("target", ["point", "polygon"])
def test_fit_preserves_real_parent_tree_and_science_in_native_wps(tmp_path, topology, target):
    from gpuwm.companion_domains import VORTEX_PRESET
    from gpuwm.namelist_import import parse_namelist_text
    from gpuwm.native_wrf_contract import native_geometry_contract
    from gpuwm.static.projection import grids_from_projection_config, grids_from_wps_namelist
    from gpuwm.wps_domain_ids import domain_ids_from_wps_text

    path, raw = starter(tmp_path, nested=True)
    if topology != "two_domains":
        third = copy.deepcopy(raw["domain"][1])
        third.update(grid_id=3, parent_id=1, nx=60, ny=60, i_parent_start=20, j_parent_start=20)
        third["follow"] = dict(VORTEX_PRESET)
        fourth = copy.deepcopy(third)
        fourth.update(grid_id=4, parent_id=3, i_parent_start=21, j_parent_start=21)
        fourth.pop("follow")
        raw["domain"] = [raw["domain"][0], *([raw["domain"][1]] if topology == "branched_tree" else []), third, fourth]
    path.write_text(st.render_tables(raw), encoding="utf-8")
    before = path.read_bytes()
    out = tmp_path / f"fitted-{topology}-{target}.toml"
    options = dict(point="41,-99", write=True)
    polygon = None
    if target == "polygon":
        polygon = tmp_path / "selected.geojson"
        polygon.write_text(json.dumps({"type": "Polygon", "coordinates": [[
            [-99.2, 40.8], [-98.8, 40.8], [-98.8, 41.2], [-99.2, 41.2], [-99.2, 40.8]]]}))
        options.update(point=None, polygon=polygon)
    assert st.fit_main(args(path, out, **options)) == 0
    actual = tomllib.loads(out.read_text())
    assert actual["shared"] == raw["shared"] and actual["output"] == raw["output"]
    assert actual["experiment"] == raw["experiment"]
    for before_domain, after_domain in zip(raw["domain"], actual["domain"]):
        assert {k: v for k, v in before_domain.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")} == {
            k: v for k, v in after_domain.items() if k not in ("nx", "ny", "i_parent_start", "j_parent_start")}
    exp = load_experiment(out)
    wps = out.with_suffix(".namelist.wps")
    tables = parse_namelist_text(wps.read_text())
    ids = tuple(domain.grid_id for domain in exp.domains)
    assert domain_ids_from_wps_text(wps.read_text(), len(ids)) == ids
    slots = {grid_id: index for index, grid_id in enumerate(ids, 1)}
    assert tables["geogrid"]["parent_id"] == [slots[domain.parent_id or 1] for domain in exp.domains]
    expected_grids = grids_from_projection_config(exp)
    actual_grids = grids_from_wps_namelist(wps)
    for domain, expected_grid, actual_grid in zip(exp.domains, expected_grids, actual_grids):
        assert native_geometry_contract(expected_grid, domain.run) == native_geometry_contract(actual_grid, domain.run)
    if polygon is not None:
        dw.verify_polygon_containment(exp, dw.load_polygon_footprint(polygon), (0.0,) * len(ids))
    proof = json.loads(out.with_suffix(".fit.json").read_text())
    assert proof["domain_order"] == list(ids)
    assert proof["peak_envelope_bytes"] <= proof["budget_bytes"]
    assert proof["launch_performed"] is False and path.read_bytes() == before


def test_relative_companion_paths_use_template_origin(tmp_path):
    path, raw = starter(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sample.grib").write_bytes(b"path-only fixture")
    raw["case_data"] = dict(forcing=["data/*.grib"], geog_root="geog", vtable="Vtable",
               wps_namelist="namelist.wps", water_temperature_overlay="overlay.nc",
               sfcp_to_sfcp=True, output_title="Custom authority")
    original_wps = dw.render_wps_namelist(raw["projection"], [(110,88)], (), source="gfs")
    original_wps = original_wps.replace("'default'", "'custom+default'")
    (tmp_path / "namelist.wps").write_text(original_wps)
    path.write_text(st.render_tables(raw))
    out = tmp_path / "elsewhere" / "fitted.toml"
    fitted = st.Starter(path, out)
    resolved = fitted.raw["case_data"]
    assert resolved["forcing"] == [str((tmp_path / "data/*.grib").resolve())]
    assert resolved["water_temperature_overlay"] == str((tmp_path / "overlay.nc").resolve())
    assert resolved["wps_namelist"] == str(out.with_suffix(".namelist.wps"))
    generated = dw.render_wps_namelist(raw["projection"], [(120,96)], (), source="gfs")
    text, delta = fitted.wps_text(generated, datetime(2026,9,5), 6)
    assert "custom+default" in text
    assert any(key.startswith("wps.geogrid.e_we") for key, _, _ in delta)
    assert (tmp_path / "namelist.wps").read_text() == original_wps


def test_output_cannot_replace_template(tmp_path):
    path, _ = starter(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="never overwrites"):
        st.fit_main(args(path, path, write=True))
    assert path.read_bytes() == before


def test_no_device_flags_use_the_shared_gpu_detector(tmp_path, monkeypatch):
    from gpuwm.cli import build_parser
    path, _ = starter(tmp_path)
    out = tmp_path / "auto.toml"
    seen = []
    def detected():
        seen.append(True)
        return dict(total_bytes=16 * dw.GIB, free_bytes=12 * dw.GIB, profile=None)
    monkeypatch.setattr(dw, "device_memory_probe_subprocess", detected)
    parsed = build_parser().parse_args([
        "domain-fit", str(path), "--point=40,-100", "--out", str(out), "--write"])
    before = path.read_bytes()
    assert st.fit_main(parsed) == 0
    assert seen == [True]
    assert path.read_bytes() == before
    receipt = json.loads(out.with_suffix(".fit.json").read_text())
    assert receipt["peak_envelope_bytes"] <= receipt["budget_bytes"] < 12 * dw.GIB
    from gpuwm.core import preflight
    from gpuwm.go_cli import memory_gate
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", detected)
    gate = memory_gate({"config": out})
    assert not gate["refuse"] and not gate["warn"], gate["verdict"]


def measured_hardware():
    return {"devices": [{"name": "Selected test GPU", "memory_total_bytes": 32 * dw.GIB}],
            "sizing": {"schema": "arwen.target-sizing.v1", "measured_unix_ms": 1788823235782,
                "total_bytes": 32 * dw.GIB, "free_bytes": 12 * dw.GIB,
                "profile": {"name": "Selected test GPU", "multiprocessor_count": 70,
                    "max_threads_per_multiprocessor": 1536, "default_stack_limit_bytes": 1024,
                    "bare_context_bytes": 256 * 1024 ** 2}}}


def test_hardware_snapshot_keeps_capacity_free_and_profile_separate_without_local_probe(tmp_path, monkeypatch):
    from gpuwm.cli import build_parser
    monkeypatch.setattr(dw, "device_memory_probe_subprocess", lambda: pytest.fail("selected hardware must not probe local GPU"))
    hardware = tmp_path / "selected-hardware.json"
    hardware.write_text(json.dumps(measured_hardware()))
    path, _ = starter(tmp_path, nested=True)
    out = tmp_path / "selected-gpu.toml"
    options = build_parser().parse_args(["domain-fit", str(path), "--point=40,-100",
        "--hardware-json", str(hardware), "--out", str(out), "--write"])
    assert st.fit_main(options) == 0
    proof = json.loads(out.with_suffix(".fit.json").read_text())
    assert proof["free_bytes"] == 12 * dw.GIB
    assert proof["selected_hardware"]["total_bytes"] == 32 * dw.GIB
    assert proof["selected_hardware"]["profile"] == measured_hardware()["sizing"]["profile"]
    assert proof["selected_hardware"]["sha256"] == hashlib.sha256(hardware.read_bytes()).hexdigest()
    assert proof["peak_envelope_bytes"] <= proof["budget_bytes"] < proof["free_bytes"]


@pytest.mark.parametrize("section,key,value", [
    ("sizing", "free_bytes", 33 * dw.GIB), ("sizing", "total_bytes", True),
    ("sizing", "free_bytes", -1), ("sizing", "schema", "unknown"),
    ("profile", "multiprocessor_count", "170"), ("profile", "max_threads_per_multiprocessor", 0),
    ("profile", "default_stack_limit_bytes", -1), ("profile", "bare_context_bytes", 33 * dw.GIB),
])
def test_invalid_hardware_snapshots_do_not_substitute_a_local_or_reference_gpu(tmp_path, monkeypatch, section, key, value):
    monkeypatch.setattr(dw, "device_memory_probe_subprocess", lambda: pytest.fail("invalid snapshot must not probe GPU"))
    payload = measured_hardware()
    target = payload["sizing"] if section == "sizing" else payload["sizing"]["profile"]
    target[key] = value
    hardware = tmp_path / "bad-hardware.json"
    hardware.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Selected GPU"):
        st.hardware_sizing(hardware)


@pytest.mark.parametrize("polygon", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_streamed_fit_uses_selected_target_host_memory_for_every_candidate(tmp_path, monkeypatch, polygon, declared):
    from gpuwm.core import streaming
    payload = measured_hardware()
    payload["host_memory"] = {"schema": "arwen.target-host-memory.v1",
        "measured_unix_ms": 1788823235782, "total_bytes": 64 * dw.GIB}
    hardware = tmp_path / "selected-hardware.json"
    hardware.write_text(json.dumps(payload))
    path, raw = starter(tmp_path, nested=True)
    raw["tiles"] = {"mode": "auto", "store": "host"}
    path.write_text(st.render_tables(raw), encoding="utf-8")
    monkeypatch.setattr(dw, "device_memory_probe_subprocess", lambda: pytest.fail("remote Fit probed the local GPU"))
    monkeypatch.setattr(streaming, "_host_total_bytes", lambda: pytest.fail("remote Fit read the desktop host RAM"))
    real = dw._sizing_phases
    observed = []
    declared_free = dw.resolve_sizing_budget(None, 16).free_bytes
    def price(exp, **kwargs):
        machine = kwargs.get("machine")
        assert machine is not None
        assert machine.host_bytes == 64 * dw.GIB
        assert machine.vram_bytes == (declared_free if declared else 12 * dw.GIB)
        observed.append(machine)
        return real(exp, **kwargs)
    monkeypatch.setattr(dw, "_sizing_phases", price)
    options = args(path, tmp_path / "fitted.toml", hardware_json=None if declared else hardware,
                   target_host_memory_json=hardware if declared else None, vram_gib=16 if declared else None, write=True)
    if polygon:
        footprint = tmp_path / "area.geojson"
        footprint.write_text(json.dumps({"type": "Polygon", "coordinates": [[
            [-100.2, 39.8], [-99.8, 39.8], [-99.8, 40.2], [-100.2, 40.2], [-100.2, 39.8]]]}))
        options.polygon, options.point = footprint, None
    assert st.fit_main(options) == 0
    assert len(observed) >= 2
    receipt = json.loads(options.out.with_suffix(".fit.json").read_text())
    assert receipt["selected_host_memory" if declared else "selected_hardware"]["host_memory"] == payload["host_memory"]


def test_streamed_fit_requires_target_host_measurement_without_local_fallback(tmp_path, monkeypatch):
    from gpuwm.core import streaming
    hardware = tmp_path / "selected-hardware.json"
    hardware.write_text(json.dumps(measured_hardware()))
    path, raw = starter(tmp_path)
    raw["tiles"] = {"mode": "on", "store": "host"}
    path.write_text(st.render_tables(raw), encoding="utf-8")
    monkeypatch.setattr(streaming, "_host_total_bytes", lambda: pytest.fail("missing target RAM used desktop RAM"))
    out = tmp_path / "unpublished.toml"
    with pytest.raises(ValueError, match="reconnect it before fitting streamed domains"):
        st.fit_main(args(path, out, hardware_json=hardware, vram_gib=None, write=True))
    assert not out.exists()


@pytest.mark.parametrize("measured", [False, True])
def test_native_memory_gate_refuses_explicit_tree_planner_failure_even_with_small_reference(tmp_path, monkeypatch, measured):
    from types import SimpleNamespace
    from gpuwm import go_cli
    from gpuwm.core import preflight
    path, _ = starter(tmp_path, nested=True)
    phases = SimpleNamespace(tree_road=SimpleNamespace(refusal="native configured tree cannot be planned"),
        peak_envelope_bytes=1024, streamed_forecast=False, ingest_priced=True,
        verdict=lambda budget: "Small resident reference: 1024 bytes")
    monkeypatch.setattr(preflight, "estimate_phases", lambda *args, **kwargs: phases)
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", lambda: measured_hardware()["sizing"] if measured else None)
    monkeypatch.setattr(preflight, "device_memory_probe_reason", lambda: "fixture has no GPU")
    monkeypatch.setattr(go_cli, "_planner_machine", lambda *args: None)
    result = go_cli.memory_gate({"config": path})
    assert result["refuse"] is True
    assert "native configured tree cannot be planned" in result["verdict"]


def test_publication_failure_removes_only_new_owned_companions(tmp_path, monkeypatch):
    wps, receipt, config = [tmp_path / name for name in
                            ("forecast.wps", "forecast.fit.json", "forecast.toml")]
    original_open = Path.open
    def fail_last(path, *args, **kwargs):
        if path == config:
            raise OSError("injected write refusal")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_last)
    with pytest.raises(OSError, match="injected write refusal"):
        st._publish_new_files(((wps, "wps"), (receipt, "receipt"), (config, "toml")))
    assert not wps.exists() and not receipt.exists() and not config.exists()


def test_publication_race_keeps_the_other_files_contents(tmp_path):
    wps, config = tmp_path / "forecast.wps", tmp_path / "forecast.toml"
    config.write_text("another writer owns this", encoding="utf-8")
    with pytest.raises(FileExistsError):
        st._publish_new_files(((wps, "new companion"), (config, "new config")))
    assert config.read_text(encoding="utf-8") == "another writer owns this"
    assert not wps.exists()


def test_printed_path_preserves_apostrophe_spaces_and_shell_metacharacters():
    import os
    path = "Drew's $forecast " + chr(96) + " file.toml"
    quoted = st._command_path(path)
    if os.name == "nt":
        assert quoted == "'Drew''s $forecast " + chr(96) + " file.toml'"
    else:
        import shlex
        assert shlex.split(quoted) == [path]


@pytest.fixture
def tile_machine(monkeypatch):
    """Real planner arithmetic, deterministic observations, no device contact."""
    from gpuwm.core import preflight, streaming
    from tilestream import autoplan
    probe = dict(total_bytes=10 * dw.GIB, free_bytes=int(7.72 * dw.GIB),
                 profile=dict(name="NVIDIA GeForce RTX 3080", multiprocessor_count=68,
                              max_threads_per_multiprocessor=1536,
                              default_stack_limit_bytes=1024, bare_context_bytes=182452224))
    observations = {"probe": probe, "host": 64 * dw.GIB,
                    "available": 48 * dw.GIB, "calls": 0}

    def observed():
        observations["calls"] += 1
        return observations["probe"]

    monkeypatch.setattr(dw, "device_memory_probe_subprocess", observed)
    monkeypatch.setattr(dw, "device_memory_probe_reason", lambda: "synthetic unavailable GPU")
    monkeypatch.setattr(preflight, "device_physical_total_bytes", lambda: probe["total_bytes"])
    monkeypatch.setattr(preflight, "live_device_local_memory_profile",
                        lambda: preflight.profile_from_device_probe(probe))
    monkeypatch.setattr(streaming, "_host_total_bytes", lambda: observations["host"])
    monkeypatch.setattr(preflight, "host_available_bytes", lambda: observations["available"])
    monkeypatch.setattr(autoplan.Machine, "detect", classmethod(
        lambda cls, **kwargs: pytest.fail("Planning must not create a CUDA context")))
    return observations


def tile_starter(tmp_path, *, nested=False):
    path, raw = starter(tmp_path, nested=nested)
    if not nested:
        raw["domain"][0].update(nx=436, ny=348)
    path.write_text(st.render_tables(raw), encoding="utf-8")
    dims = [(domain["nx"], domain["ny"]) for domain in raw["domain"]]
    ratios = (3,) if nested else ()
    path.with_suffix(".namelist.wps").write_text(dw.render_wps_namelist(
        raw["projection"], dims, ratios, source="gfs",
        root_dx_m=raw["domain"][0]["dx"]), encoding="utf-8")
    return path, raw


def test_tiles_copy_uses_real_planner_preserves_science_and_publishes_only_on_write(
        tmp_path, tile_machine, capsys):
    from gpuwm.cli import main
    from gpuwm.core import preflight
    path, raw = tile_starter(tmp_path)
    before = path.read_bytes()
    source_wps = path.with_suffix(".namelist.wps").read_bytes()
    out = tmp_path / "copy" / "tiles.toml"
    profile = preflight.profile_from_device_probe(tile_machine["probe"])
    resident = preflight.estimate_experiment(load_experiment(path),
                                             vram_gib=10, profile=profile)
    assert resident.peak_envelope_bytes > tile_machine["probe"]["free_bytes"] - dw.GIB / 2
    arguments = ["domain-tiles", str(path), "--out", str(out), "--mode", "auto"]

    assert main(arguments) == 0, capsys.readouterr().err
    assert not out.parent.exists()
    assert tile_machine["calls"] == 1
    assert main(arguments + ["--write"]) == 0, capsys.readouterr().err

    assert tile_machine["calls"] == 2
    copied = tomllib.loads(out.read_text(encoding="utf-8"))
    assert copied.pop("tiles") == {"mode": "auto", "store": "host"}
    assert copied == raw
    assert path.read_bytes() == before
    assert path.with_suffix(".namelist.wps").read_bytes() == source_wps
    receipt = json.loads(out.with_suffix(".tiles.json").read_text())
    assert receipt["domains"][0]["road"] == "streamed"
    assert receipt["domains"][0]["tile_nx"] > 0
    assert receipt["peak_envelope_bytes"] <= receipt["budget_bytes"]
    assert receipt["host_store_bytes"] <= receipt["host_budget_bytes"] <= tile_machine["available"]
    assert receipt["launch_performed"] is False and receipt["download_performed"] is False
    assert receipt["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt["wps_sha256"] == hashlib.sha256(out.with_suffix(".namelist.wps").read_bytes()).hexdigest()
    assert "working directory" in receipt["fetch_out_path_basis"]
    assert main(arguments + ["--write"]) == 2
    assert tile_machine["calls"] == 2


def test_tiles_on_forces_streaming_even_when_the_domain_fits_resident(
        tmp_path, tile_machine, capsys):
    from gpuwm.cli import main
    path, raw = tile_starter(tmp_path)
    raw["domain"][0].update(nx=110, ny=88)
    path.write_text(st.render_tables(raw), encoding="utf-8")
    path.with_suffix(".namelist.wps").write_text(dw.render_wps_namelist(
        raw["projection"], [(110, 88)], (), source="gfs",
        root_dx_m=raw["domain"][0]["dx"]), encoding="utf-8")
    tile_machine["probe"].update(total_bytes=16 * dw.GIB, free_bytes=15 * dw.GIB)
    out = tmp_path / "forced.toml"
    assert main(["domain-tiles", str(path), "--out", str(out),
                 "--mode", "on", "--write"]) == 0, capsys.readouterr().err
    receipt = json.loads(out.with_suffix(".tiles.json").read_text())
    assert receipt["domains"][0]["road"] == "streamed"
    assert receipt["peak_envelope_bytes"] <= receipt["budget_bytes"]
    copied = tomllib.loads(out.read_text(encoding="utf-8"))
    assert copied.pop("tiles") == {"mode": "on", "store": "host"}
    assert copied == raw


def test_tiles_on_refuses_when_the_canonical_plan_exceeds_reserved_budget(
        tmp_path, tile_machine, capsys):
    from gpuwm.cli import main
    path, _ = tile_starter(tmp_path)
    out = tmp_path / "forced-busy" / "tiles.toml"
    assert main(["domain-tiles", str(path), "--out", str(out),
                 "--mode", "on", "--write"]) == 2
    assert "Tile streaming does not fit" in capsys.readouterr().err
    assert not out.parent.exists()


def test_tiles_auto_prices_entire_nested_tree_without_changing_any_domain(
        tmp_path, tile_machine, capsys):
    from gpuwm.cli import main
    path, raw = tile_starter(tmp_path, nested=True)
    out = tmp_path / "nested-tiles.toml"
    assert main(["domain-tiles", str(path), "--out", str(out), "--write"]) == 0, capsys.readouterr().err
    copied = tomllib.loads(out.read_text(encoding="utf-8"))
    assert copied.pop("tiles") == {"mode": "auto", "store": "host"}
    assert copied == raw
    receipt = json.loads(out.with_suffix(".tiles.json").read_text())
    assert [row["grid_id"] for row in receipt["domains"]] == [1, 2]
    assert receipt["peak_envelope_bytes"] <= receipt["budget_bytes"]


@pytest.mark.parametrize("resource", ["vram", "host", "memory", "geometry", None])
def test_tile_retry_classifies_only_typed_memory_refusals(tmp_path, tile_machine, monkeypatch, capsys, resource):
    from types import SimpleNamespace
    from gpuwm.cli import main
    from gpuwm.core import preflight
    path, _ = tile_starter(tmp_path, nested=True)
    phases = SimpleNamespace(tree_road=SimpleNamespace(
        refusal="the same planner refusal text", refusal_resource=resource, priced=False))
    monkeypatch.setattr(preflight, "estimate_phases", lambda *args, **kwargs: phases)
    out = tmp_path / "not-published.toml"
    assert main(["domain-tiles", str(path), "--out", str(out), "--write"]) == 2
    output = capsys.readouterr()
    if resource in {"vram", "host", "memory"}:
        error = json.loads(output.out)
        assert error["schema"] == "arwen.configuration-error.v1" and error["kind"] == "memory"
        assert error["memory"]["resource"] == resource
    else:
        assert not output.out
    assert not out.exists()


@pytest.mark.parametrize("resource", ["vram", "host", "geometry", None])
def test_tree_report_retains_resource_type_without_guessing_from_text(tmp_path, tile_machine, monkeypatch, resource):
    from gpuwm.core import streaming
    from tilestream.autoplan import CannotPlan
    path, raw = tile_starter(tmp_path, nested=True)
    raw["tiles"] = {"mode": "auto"}
    path.write_text(st.render_tables(raw), encoding="utf-8")
    def refuse(*args, **kwargs):
        if resource is None:
            raise streaming.StreamingRefused("vram budget words in an unsupported route")
        raise CannotPlan("same planner refusal words", resource)
    monkeypatch.setattr(streaming, "decide_tree", refuse)
    road = streaming.tree_road_plan(load_experiment(path))
    assert not road.priced and road.refusal_resource == resource


@pytest.mark.parametrize("resource", ["vram", "host", "unknown-host", "unknown-gpu"])
def test_tiles_refuses_unavailable_memory_without_publishing(
        tmp_path, tile_machine, capsys, resource):
    from gpuwm.cli import main
    path, _ = tile_starter(tmp_path)
    before = path.read_bytes()
    out = tmp_path / "refused" / "tiles.toml"
    if resource == "vram":
        tile_machine["probe"]["free_bytes"] = dw.GIB // 2
    elif resource == "host":
        tile_machine["available"] = 16 * 1024 ** 2
    elif resource == "unknown-host":
        tile_machine["host"] = None
    else:
        tile_machine["probe"] = None
    assert main(["domain-tiles", str(path), "--out", str(out), "--write"]) == 2
    assert not out.parent.exists()
    assert path.read_bytes() == before
    assert tile_machine["calls"] == 1


@pytest.mark.parametrize("choice", ["pinned", "per-domain", "device-store"])
def test_tiles_recovery_reports_existing_explicit_choices_before_any_probe(
        tmp_path, tile_machine, capsys, choice):
    from gpuwm.cli import main
    path, raw = tile_starter(tmp_path)
    if choice == "pinned":
        raw["tiles"] = dict(mode="on", tile_nx=64, tile_ny=64, nbuffers=2)
    elif choice == "per-domain":
        raw["domain"][0]["tiles"] = dict(mode="off")
    else:
        raw["tiles"] = dict(mode="on", store="device")
    path.write_text(st.render_tables(raw), encoding="utf-8")
    before = path.read_bytes()
    out = tmp_path / "refused" / "tiles.toml"
    assert main(["domain-tiles", str(path), "--out", str(out), "--write"]) == 2
    assert "explicit streaming choices" in capsys.readouterr().err
    assert tile_machine["calls"] == 0
    assert not out.parent.exists()
    assert path.read_bytes() == before


def test_tiles_write_reprices_after_a_successful_preview(tmp_path, tile_machine, capsys):
    from gpuwm.cli import main
    path, _ = tile_starter(tmp_path)
    out = tmp_path / "became-busy" / "tiles.toml"
    arguments = ["domain-tiles", str(path), "--out", str(out)]
    assert main(arguments) == 0, capsys.readouterr().err
    tile_machine["probe"]["free_bytes"] = dw.GIB // 2
    assert main(arguments + ["--write"]) == 2
    assert not out.parent.exists()
    assert tile_machine["calls"] == 2


def test_tiles_copy_rebases_only_schema_owned_paths_in_other_directory(
        tmp_path, tile_machine, capsys):
    from gpuwm.cli import main
    path, raw = tile_starter(tmp_path, nested=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "input.grib").write_bytes(b"path-only fixture")
    raw["case_data"] = dict(forcing=["data/*.grib"], geog_root="geog", vtable="Vtable",
                            wps_namelist="original.namelist.wps",
                            sfcp_to_sfcp=True, output_title="Tile path authority")
    raw["static"] = dict(highres=dict(enabled=False, cache_root="future-cache"))
    path.write_text(st.render_tables(raw), encoding="utf-8")
    out = tmp_path / "new-location" / "tiles.toml"
    assert main(["domain-tiles", str(path), "--out", str(out), "--write"]) == 0, capsys.readouterr().err
    copied = tomllib.loads(out.read_text(encoding="utf-8"))
    assert copied["case_data"]["wps_namelist"] == str((tmp_path / "original.namelist.wps").resolve())
    assert copied["case_data"]["forcing"] == [str((tmp_path / "data/*.grib").resolve())]
    assert copied["static"]["highres"]["cache_root"] == str((tmp_path / "future-cache").resolve())
    for key in ("experiment", "projection", "shared", "domain", "fetch", "output"):
        assert copied[key] == raw[key]
    assert not (tmp_path / "future-cache").exists()


@pytest.mark.parametrize("changed_input", ["config", "wps"])
def test_tiles_refuses_if_an_input_changes_during_planning(
        tmp_path, tile_machine, capsys, monkeypatch, changed_input):
    from gpuwm.cli import main
    path, _ = tile_starter(tmp_path)
    changed = path if changed_input == "config" else path.with_suffix(".namelist.wps")
    before = changed.read_bytes()
    observed = dw.device_memory_probe_subprocess

    def concurrent_edit():
        changed.write_bytes(before + b"\n# edited during preview\n")
        return observed()

    monkeypatch.setattr(dw, "device_memory_probe_subprocess", concurrent_edit)
    out = tmp_path / "concurrent-edit" / "tiles.toml"
    assert main(["domain-tiles", str(path), "--out", str(out), "--write"]) == 2
    assert "changed during planning" in capsys.readouterr().err
    assert changed.read_bytes() == before + b"\n# edited during preview\n"
    assert not out.parent.exists()
