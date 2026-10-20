"""Default-on hydro door decision table, receipts, cost budget and products (CPU)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.optional_routing import activation_receipt, member_settings
from gpuwm.hydro import cost, door, forecast_products, runtime
from gpuwm.hydro.config import configure

ROOT = Path(__file__).resolve().parents[1]
CONUS = ROOT / "configs" / "conus3km.toml"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (runtime.ENV_ENABLED, runtime.ENV_SETTINGS, door.ENV_DECISION,
                 door.ENV_TABLE, door.ENV_CACHE, "GPUWM_RW_HYDRO_WORLD"):
        monkeypatch.delenv(name, raising=False)


def box_geometry(south=30.0, north=31.0, west=-98.0, east=-97.0, lsm=4, center=None):
    points = [(south + (north - south) * j / 4, west + (east - west) * i / 4)
              for j in range(5) for i in range(5)]
    return door.geometry_from_points([p[0] for p in points], [p[1] for p in points],
                                     center_lon=(west + east) / 2 if center is None else center,
                                     parent_lsm=lsm, grid_key="k" * 64)


def prepared_table(tmp_path, *, coverage=((-99.0, 29.0, -96.0, 32.0),), licence="verified",
                   realization="lower-colorado-hyfeatures"):
    manifest = tmp_path / "prepared" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema": "arwen.nextgen.hydrofabric.v1"}))
    shipped = json.loads(door.TABLE.read_text())
    shipped["providers"] = [{
        "id": "test-prepared", "owner": "test", "priority": 1, "kind": "prepared",
        "coverage": [list(box) for box in coverage], "min_covered_fraction": 0.0,
        "realization": realization, "manifest_binding": "fabric_manifest",
        "manifest": str(manifest),
        "data_licence": {"status": licence, "name": "test terms", "reason": "test row",
                         "attribution": "test"}}] + shipped["providers"]
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(shipped))
    return path, manifest


def conus_copy(tmp_path, lsm):
    text = CONUS.read_text().replace("sf_surface_physics = 2", f"sf_surface_physics = {lsm}")
    path = tmp_path / f"conus-lsm{lsm}.toml"
    path.write_text(text)
    return path


# -- decision table -----------------------------------------------------------

def test_fabric_present_turns_hydro_on_with_bound_realization(tmp_path):
    table, manifest = prepared_table(tmp_path)
    decision = door.decide(settings={}, requested=None, table=table, geometry=box_geometry())
    assert decision["enabled"] is True and decision["reason"] == "fabric_covered"
    assert decision["provider"] == "test-prepared"
    assert decision["candidates"][0]["covered_fraction"] == 1.0
    composed, path, build = door.materialize(decision, config_path=tmp_path / "x.toml")
    settings = runtime.read_settings(path)
    assert settings["enabled"] is True and settings["activation_reason"] == "fabric_covered"
    expanded = configure(settings)
    assert expanded["realization_identity"]["id"] == "lower-colorado-hyfeatures"
    assert expanded["stages"][0]["options"]["manifest"] == str(manifest)
    assert "door" not in settings  # never inside the sealed realization identity


def test_fabric_absent_default_conus_suite_names_provider_gap(tmp_path):
    decision = door.decide(settings={}, requested=None, config_path=CONUS)
    assert decision["enabled"] is False
    assert decision["reason"] == "no_covering_fabric"
    assert decision["footprint"]["parent_lsm"] == 2
    assert decision["candidates"]
    assert all(row["gap"] for row in decision["gaps"])


def test_fabric_absent_noahmp_conus_names_every_provider_gap(tmp_path):
    decision = door.decide(settings={}, requested=None, config_path=conus_copy(tmp_path, 4))
    assert decision["enabled"] is False and decision["reason"] == "no_covering_fabric"
    rows = {c["id"]: c for c in decision["candidates"]}
    assert rows["conus-nhdplus-v21"]["covered_fraction"] == 1.0
    assert rows["hydro-world-global"]["covered_fraction"] == 1.0
    text = json.dumps(decision["gaps"])
    assert "data licence pending" in text and "no admitted coupled realization" in text
    assert "rw_hydro_world" in text
    assert {gap["owner"] for gap in decision["gaps"]} == {"astra-c-hydro-data", "hydro-world-fabric"}


def test_partial_cover_falls_through_to_next_row(tmp_path):
    table, _ = prepared_table(tmp_path, coverage=((-97.5, 29.0, -96.0, 32.0),))
    rows = json.loads(table.read_text())
    rows["providers"][0]["min_covered_fraction"] = 1.0
    table.write_text(json.dumps(rows))
    decision = door.decide(settings={}, requested=None, table=table, geometry=box_geometry())
    assert decision["enabled"] is False
    assert decision["candidates"][0]["covered_fraction"] == 0.6
    assert "covers 0.600" in decision["candidates"][0]["reasons"][0]


def test_unverified_licence_or_missing_realization_never_turns_on(tmp_path):
    for kwargs, needle in (({"licence": "pending"}, "data licence pending"),
                           ({"licence": "research-only"}, "data licence research-only (test terms)"),
                           ({"realization": None}, "no admitted coupled realization"),
                           ({"realization": "no-such-row"}, "unknown realization")):
        table, _ = prepared_table(tmp_path, **kwargs)
        decision = door.decide(settings={}, requested=None, table=table, geometry=box_geometry())
        assert decision["enabled"] is False
        assert needle in json.dumps(decision["candidates"][0]["reasons"])


def test_dateline_domain_is_covered_by_dateline_row(tmp_path):
    table, _ = prepared_table(tmp_path, coverage=((170.0, -20.0, -170.0, -10.0),))
    geometry = box_geometry(south=-18, north=-12, west=175.0, east=185.0, center=180.0)
    geometry["points"] = [(lat, door._wrap180(lon)) for lat, lon in geometry["points"]]
    assert geometry["crosses_dateline"] is True
    decision = door.decide(settings={}, requested=None, table=table, geometry=geometry)
    assert decision["enabled"] is True


def test_opt_out_never_reads_the_table_or_the_domain(tmp_path):
    decision = door.decide(settings={"enabled": False, "activation_reason": "cli_off"},
                           requested="off", table=tmp_path / "absent.json",
                           geometry_reader=lambda path: pytest.fail("geometry read"))
    assert decision == {"schema": door.SCHEMA, "requested": "off", "enabled": False,
                        "reason": "cli_off", "gaps": []}


def test_explicit_settings_pass_through_unchanged(tmp_path):
    decision = door.decide(settings={"enabled": True, "realization": "x",
                                     "activation_reason": "cli_enabled"},
                           requested="s.toml", table=tmp_path / "absent.json")
    assert decision["enabled"] is True and decision["fabric"] == "explicit settings"


def test_bare_hydro_flag_without_fabric_refuses_with_gaps(tmp_path):
    with pytest.raises(door.HydroDoorRefusal, match="no hydrofabric covers") as error:
        door.resolve_launch({"enabled": True, "activation_reason": "cli_enabled"},
                            requested="", config_path=conus_copy(tmp_path, 4))
    assert "astra-c-hydro-data" in str(error.value)


def test_unreadable_domain_is_a_named_gap(tmp_path):
    path = tmp_path / "empty.toml"
    path.write_text("")
    decision = door.decide(settings={}, requested=None, config_path=path)
    assert decision["reason"] == "no_domain_geometry"
    assert str(path) in decision["gaps"][0]["gap"]


# -- launcher, runner and ensemble members ------------------------------------

def _launch_args(path, hydro=None):
    return SimpleNamespace(config=path, experiment_config=path, hydro=hydro)


def test_go_scope_receipt_carries_door_decision_and_member_state(tmp_path, monkeypatch):
    table, manifest = prepared_table(tmp_path)
    monkeypatch.setenv(door.ENV_TABLE, str(table))
    monkeypatch.setattr(door, "TABLE", Path(tmp_path / "only-extra.json"))
    shipped = json.loads(table.read_text())
    (tmp_path / "only-extra.json").write_text(json.dumps({**shipped, "providers": []}))
    monkeypatch.setattr(door, "domain_geometry", lambda path: box_geometry())
    monkeypatch.setenv(door.ENV_CACHE, str(tmp_path / "cache"))
    experiment = tmp_path / "experiment.toml"
    experiment.write_text("")
    args = _launch_args(experiment)

    @runtime.go_scope
    def launch(args):
        settings = runtime.current_settings()
        members = {member: member_settings(SimpleNamespace(experiment_config=None))
                   for member in (0, 1)}
        members[0]["settings_bindings"]["fabric_manifest"] = "mutated"
        receipts = {m: activation_receipt(s) for m, s in members.items()}
        return settings, members, receipts, os.environ[runtime.ENV_SETTINGS]
    settings, members, receipts, env_path = launch(args)
    assert settings["enabled"] is True and settings["realization"] == "lower-colorado-hyfeatures"
    assert Path(env_path).name == "hydro-door.toml"
    # One independent settings object per member: no shared mutable state.
    assert members[1]["settings_bindings"]["fabric_manifest"] == str(manifest)
    for receipt in receipts.values():
        assert receipt["enabled"] is True
        assert receipt["door"]["provider"] == "test-prepared"
        assert receipt["door"]["footprint"]["parent_lsm"] == 4
    assert door.ENV_DECISION not in os.environ and runtime.ENV_SETTINGS not in os.environ


def test_go_scope_off_is_off_for_every_member(tmp_path, monkeypatch):
    table, _ = prepared_table(tmp_path)
    monkeypatch.setenv(door.ENV_TABLE, str(table))
    monkeypatch.setattr(door, "domain_geometry", lambda path: pytest.fail("door ran"))
    experiment = tmp_path / "experiment.toml"
    experiment.write_text("")

    @runtime.go_scope
    def launch(args):
        return [activation_receipt(member_settings(SimpleNamespace(experiment_config=None)))
                for _ in range(3)]
    receipts = launch(_launch_args(experiment, hydro="off"))
    assert all(r["enabled"] is False and r["reason"] == "cli_off" for r in receipts)
    assert all(r["door"]["reason"] == "cli_off" for r in receipts)


def test_runner_without_launcher_decides_once_and_receipts_the_gap(tmp_path):
    inputs = _launch_args(conus_copy(tmp_path, 4))

    @runtime.forecast_scope
    def forecast(inputs):
        return activation_receipt(runtime.current_settings())
    receipt = forecast(inputs)
    assert receipt["enabled"] is False
    assert receipt["reason"] == "no_covering_fabric"
    assert receipt["door"]["gaps"]
    assert door.ENV_DECISION not in os.environ


def test_builder_row_builds_once_then_reuses(tmp_path, monkeypatch):
    shipped = json.loads(door.TABLE.read_text())
    row = dict(shipped["providers"][1], data_licence={
        "status": "verified", "name": "CC BY 4.0", "reason": "", "attribution": "x"},
        realization="lower-colorado-hyfeatures")
    row.pop("preparation", None)  # Exercise the generic builder, without physical provider stages.
    fake = tmp_path / "rw_hydro_world"
    fake.write_text("#!/bin/sh\necho 'usage: rw_hydro_world build | admit'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("GPUWM_RW_HYDRO_WORLD", str(fake))
    monkeypatch.setenv("GPUWM_HYDRO_WORLD_TILES", str(tmp_path))
    monkeypatch.setenv(door.ENV_CACHE, str(tmp_path / "cache"))
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        Path(argv[argv.index("--out") + 1]).write_text("gpkg")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def prepare(config):
        out = Path(json.loads(Path(config).read_text())["output"])
        out = out / "content-key"
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text("{}")
        return {"output": str(out)}  # Native rw_hydrofabric prepare's response.
    decision = {"_row": row, "footprint": door._summary(box_geometry()),
                "manifest_binding": "fabric_manifest"}
    _, first, build = door.materialize(decision, config_path=tmp_path / "e.toml",
                                       runner=runner, prepare=prepare)
    assert build["reused"] is False and calls[0][1] == "build"
    assert Path(build["manifest"]).parent.name == "content-key"
    assert "--bbox" in calls[0] and calls[0][calls[0].index("--bbox") + 1] == "-98.0,30.0,-97.0,31.0"
    _, second, build = door.materialize(decision, config_path=tmp_path / "e.toml",
                                        runner=runner, prepare=prepare)
    assert build["reused"] is True and len(calls) == 1 and first == second
    first_bytes = first.read_bytes()
    # A different source root must not reuse a fabric from the previous root.
    other_sources = tmp_path / "other-sources"
    other_sources.mkdir()
    monkeypatch.setenv("GPUWM_HYDRO_WORLD_TILES", str(other_sources))
    _, changed, build = door.materialize(decision, config_path=tmp_path / "e.toml",
                                   runner=runner, prepare=prepare)
    assert build["reused"] is False and len(calls) == 2
    assert changed != first and first.read_bytes() == first_bytes
    _, repeated, build = door.materialize(decision, config_path=tmp_path / "e.toml",
                                          runner=runner, prepare=prepare)
    assert repeated == changed and build["reused"] is True and len(calls) == 2
    monkeypatch.delenv("GPUWM_HYDRO_WORLD_TILES")
    with pytest.raises(RuntimeError, match="source tiles.*not available"):
        door.materialize(decision, config_path=tmp_path / "e.toml", runner=runner,
                         prepare=prepare)
    assert len(calls) == 2


# -- cost budget ----------------------------------------------------------------

class FakeEvent:
    def __init__(self, clock):
        self.clock, self.t, self.done = clock, None, True

    def record(self, stream):
        self.t = self.clock.pop(0)

    def synchronize(self):
        pass


def metered(per_call_ms, calls):
    times = []
    for index in range(calls):
        base = index * 100.0
        times += [base, base + per_call_ms / 2, base + 10, base + 10 + per_call_ms / 2]
    meter = cost.HydroCostMeter(ring=4, events=lambda: FakeEvent(times),
                                elapsed=lambda a, b: b.t - a.t)
    for _ in range(calls):
        meter.before_lsm_start(stream=0)
        meter.before_lsm_end(stream=0)
        meter.advance_start(stream=0)
        meter.advance_end(stream=0)
    stages = [SimpleNamespace(name="routing", parallelism="6723 MC reaches level-ordered")]
    return SimpleNamespace(hydro=SimpleNamespace(interval_seconds=300.0, prepared_stages=stages),
                           hydro_cost=meter)


def test_cost_receipt_names_parallelism_and_passes_budget():
    physics = metered(1.25, 12)
    block = cost.receipt(physics, forecast_seconds=240 * 0.2 + 0.015, atmosphere_steps=240)
    assert block["status"] == "PASS" and block["sampled_calls"] == 12
    assert block["hydro_ms_per_lsm_call_mean"] == pytest.approx(1.25)
    assert block["fraction_of_atmosphere_step"] == pytest.approx(15.0 / 48000.0)
    assert block["parallelism"][0]["parallelism"].startswith("6723 MC")
    cost.refuse_if_over_budget(block)


def test_cost_over_budget_is_a_named_defect_without_failing_forecast(capsys):
    physics = metered(50.0, 12)
    block = cost.receipt(physics, forecast_seconds=12 * 0.01 + 0.6, atmosphere_steps=12)
    assert block["status"] == "DEFECT" and block["defect"]["name"] == "hydro-cost-over-budget"
    cost.refuse_if_over_budget(block)
    assert "DEFECT recorded; forecast continues" in capsys.readouterr().err


def test_cost_receipt_without_hydro_says_not_run():
    assert cost.receipt(SimpleNamespace(), forecast_seconds=1, atmosphere_steps=1)["status"] == "not-run"


# -- products ---------------------------------------------------------------------

def test_matched_gauges_use_exact_roster_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    (tmp_path / "gauges.json").write_text(json.dumps({"rows": [
        {"feature_id": "wb-2420816", "station_id": "08117995"},
        {"feature_id": "wb-999", "station_id": "x"}]}))
    gauges, why = forecast_products.matched_gauges(manifest, ["2420816", "5"])
    assert gauges == ["2420816"] and why is None
    assert forecast_products.matched_gauges(None, [])[1].startswith("no prepared fabric")


def test_products_without_history_stage_name_the_gap(tmp_path):
    physics = SimpleNamespace(hydro=SimpleNamespace(components={"routing": object()}))
    block = forecast_products.publish(physics, {}, outdir=tmp_path, dx_m=3000.0)
    assert block["status"] == "gap" and "prepare_history" in block["gap"]
    assert forecast_products.render_domain(3000.0) == "d01-3km"
    assert forecast_products.render_domain(750.0) == "d01-750m"


def test_successful_renderer_does_not_hide_missing_gauge_binding(tmp_path, monkeypatch, capsys):
    from gpuwm.hydro import output

    history_dir = tmp_path / "written" / "history"
    history_dir.mkdir(parents=True)
    request = history_dir / "renderer-request.json"
    request.write_text(json.dumps({"node_ids": ["5"]}))
    writer = SimpleNamespace(finish=lambda *args, **kwargs:
                             {"output_directory": str(history_dir.parent)})
    monkeypatch.setattr(output, "prepare", lambda **kwargs: writer)
    physics = SimpleNamespace(hydro=SimpleNamespace(
        components={"history": object.__new__(output.History)}))
    block = forecast_products.publish(
        physics, {"settings_bindings": {"fabric_manifest": str(tmp_path / "manifest.json")}},
        outdir=tmp_path / "forecast", dx_m=5000.0,
        renderer_runner=lambda path: tmp_path / "renderer-receipt.json")
    assert block["status"] == "partial"
    assert block["products"][0]["status"] == "partial"
    assert "no gauges.json" in block["products"][0]["gauge_gap"]
    assert "no gauges.json" in capsys.readouterr().err


def test_shipped_global_provider_uses_dem_resolution_auto():
    """The default door delegates resolution choice to the accepted DEM row."""
    from pathlib import Path
    table = json.loads((Path(__file__).resolve().parents[1] /
                        "gpuwm/data/hydro/fabric-providers.v1.json").read_text())
    provider = next(row for row in table["providers"] if row["id"] == "hydro-world-global")
    assert provider["builder"]["source_table"]["row"] == "auto"
    assert provider["realization"] == "parent-noahmp-routed"
    assert provider["parent_lsm"] == [4]
    assert provider["preparation"]["initial_source"] == "era5-arco"
