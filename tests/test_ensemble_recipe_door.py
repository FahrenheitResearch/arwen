"""The time-lagged and multi-model front door: request, plan, door body, boundary.

The door body is run here, on a CPU: the real request, plan, member
configs, chain plans, flag handling and receipts, from the shipped
one-domain configs.  Only the stages that cost something are replaced (a
member's fetch and preparation, the gates that open a card, the forecast
runner), and each test says which.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from gpuwm.ensemble import recipe_door
from gpuwm.ensemble.calibration_admission import UNCALIBRATED_SPREAD_REASON
from gpuwm.ensemble.door import production_run_scope as REAL_RUN_SCOPE, request_for_payload
from gpuwm.ensemble.request import EnsembleRequest

REPO = Path(__file__).resolve().parents[1]
START = datetime(2026, 10, 1, 18)
MULTI = [{"source": "hrrr", "cycle": "2026-10-01T18"}, {"source": "rap", "cycle": "2026-10-01T18"}]
MULTI_TOML = ('[ensemble]\nrecipe="multi-model"\n'
              'trajectories=[{source="hrrr",cycle="2026-10-01T18"},{source="rap",cycle="2026-10-01T18"}]\n')
#: The shipped one-domain configs the door body is run from, each with the
#: WPS namelist `gpuwm domain` wrote beside it.
HOURLY = "hrrr_native_quick_demo"            # hrrr 2026-08-20T00, one hour
MEMBERS = "gefs_member_mesoscale_demo"       # gefs 2026-08-20T00, twelve hours
RECIPE_FLAGS = ["--recipe", "time-lagged", "--members", "2"]


def experiment(domains=1, hours=1):
    return SimpleNamespace(start_time=START, run_seconds=3600.0 * hours, domains=(object(),) * domains,
                           vertical=SimpleNamespace(p_top=5000.0))


def payload(source="hrrr", **fetch):
    return {"experiment": {"start_time": START, "run_seconds": 3600.0},
            "fetch": {"source": source, "cycle": "2026-10-01T18", "hours": 1, **fetch}}


def _case(tmp_path, stem=HOURLY, *, hours=None, fetch="", table=""):
    """A copy of a shipped config and its WPS namelist, optionally longer or with extra tables."""
    text = (REPO / "configs" / f"{stem}.toml").read_text(encoding="utf-8")
    if hours is not None:
        text = text.replace("run_seconds = 3600.0", f"run_seconds = {3600.0 * hours}")
        text = text.replace("\nhours = 1\n", f"\nhours = {hours}\n")
        assert f"hours = {hours}" in text and f"run_seconds = {3600.0 * hours}" in text
    if fetch:
        assert "[fetch]\n" in text
        text = text.replace("[fetch]\n", "[fetch]\n" + fetch)
    folder = tmp_path / "case"
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / "case.toml"
    config.write_text(text + table, encoding="utf-8")
    (folder / "case.namelist.wps").write_bytes(
        (REPO / "configs" / f"{stem}.namelist.wps").read_bytes())
    return config


def _listed(tmp_path, entries, name="members.json"):
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _reviewed(config, request, scratch, **options):
    from gpuwm.experiment import load_experiment
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    recipe = recipe_door.plan_recipe(request, raw, load_experiment(config))
    return recipe, recipe_door.review_members(
        raw, config, config.with_name(config.stem + ".namelist.wps"), recipe,
        scratch=scratch, options=options)


@pytest.fixture
def door(monkeypatch, capsys):
    """The real command line, with the startup banner and install check answered."""
    from gpuwm import capabilities, cli, provenance_gate
    monkeypatch.setattr(provenance_gate, "announce", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)

    def run(*argv):
        code = cli.main([str(item) for item in argv])
        captured = capsys.readouterr()
        return SimpleNamespace(code=code, out=captured.out, err=captured.err,
                               text=captured.out + captured.err)
    return run


@pytest.fixture
def stages(monkeypatch):
    """The door body with only the paid stages replaced.

    Each member's fetch and preparation chain, the gates that open a card
    and the forecast runner are recorders; everything else (the request,
    the plan, the member configs, the receipts, the failure handling) is
    the door's own code.
    """
    from gpuwm import prepared_single_domain_forecast, regional_preparation, stage_cli
    from gpuwm.ensemble import door as door_module

    record = SimpleNamespace(prepared=[], admitted=[], forecast=[], fail={},
                             forecast_result=0, session=None, sim=[])

    def owner(plan, *, config_path, exp, observer, run_dir, prepare_only):
        index = int(Path(run_dir).parent.name.rsplit("-", 1)[1])
        record.prepared.append({
            "member": index, "config": Path(config_path), "prepare_only": prepare_only,
            "run_options": dict(plan.run_options),
            "fetch": tomllib.loads(Path(config_path).read_text(encoding="utf-8"))["fetch"]})
        if index in record.fail:
            raise record.fail[index]
        root = Path(run_dir) / "bundle"
        root.mkdir(parents=True)
        return {"prepared_root": str(root), "experiment_config": str(config_path),
                "wps_namelist": str(Path(config_path).with_name("experiment.namelist.wps"))}

    def runner(argv, observer=None):
        record.forecast.append(list(argv))
        if isinstance(record.forecast_result, BaseException):
            raise record.forecast_result
        return record.forecast_result

    @contextmanager
    def scope(request, *, output_directory, session_factory=None):
        record.session = SimpleNamespace(request=request, input_provider=None, member_roster=None,
                                         completed_products=lambda: {"frames": 1})
        yield record.session

    monkeypatch.setattr(regional_preparation, "preparation_chains", lambda: {
        key: owner for key in ("prepared:go", "prepared:hrrr", "prepared:staged")})
    # ``raising=False``: a door with no gate step still runs these tests, and
    # fails them on what it does rather than on the missing name.
    monkeypatch.setattr(recipe_door, "admit", raising=False, value=(
        lambda plans, **kw: record.admitted.append([plan.line for plan in plans])))
    monkeypatch.setattr(stage_cli, "resolve_bundle", lambda root: {
        "layout": "single", "source": "hrrr", "root": Path(root)})
    monkeypatch.setattr(stage_cli, "sim_command", lambda bundle, **kw: (
        record.sim.append(kw) or ["python", "-m", "runner", "--outdir", str(kw["outdir"])]))
    monkeypatch.setattr(prepared_single_domain_forecast, "main", runner)
    monkeypatch.setattr(door_module, "production_run_scope", scope)
    return record


def _receipt(out):
    (run,) = Path(out).glob("run-*")
    return run, json.loads((run / recipe_door.RECEIPT_NAME).read_text(encoding="utf-8"))


# ---- the request ----------------------------------------------------------

def test_recipe_flag_and_config_table_select_the_same_request():
    flag = request_for_payload(b"[ensemble]\nmembers=2\n", recipe="time-lagged")
    table = request_for_payload(b'[ensemble]\nmembers=2\nrecipe="time-lagged"\n')
    bare = request_for_payload(b"", members=2, recipe="time-lagged")
    assert flag == table == bare
    assert flag.recipe == "time-lagged" and flag.receipt()["recipe"] == "time-lagged"
    assert EnsembleRequest.from_mapping(flag.receipt()) == flag


def test_trajectory_list_selects_multi_model_and_counts_its_members():
    flag = request_for_payload(b"", trajectories=MULTI)
    table = request_for_payload(MULTI_TOML.encode())
    assert flag == table
    assert flag.recipe == "multi-model" and flag.members == 2
    assert [item["source"] for item in flag.trajectories] == ["hrrr", "rap"]
    assert request_for_payload(MULTI_TOML.encode(), members=1).members == 1


def test_request_without_a_recipe_keeps_its_earlier_receipt():
    assert "recipe" not in EnsembleRequest(2).receipt()
    assert "trajectories" not in EnsembleRequest(2).receipt()


@pytest.mark.parametrize("options, words", [
    (dict(members=2, recipe="recentered"), "ensemble recipe must be one of"),
    (dict(members=2, recipe="multi-model"), "needs its trajectory list"),
    (dict(members=2, recipe="time-lagged", trajectories=MULTI), 'belong to recipe = "multi-model"'),
    (dict(members=3, recipe="multi-model", trajectories=MULTI), "would fabricate ensemble size"),
    (dict(members=2, recipe="time-lagged", sources=({"source": "hrrr"},) * 2), "selects every member's source"),
    (dict(members=2, recipe="multi-model", trajectories=[{"source": "hrrr"}] * 2), "needs a source and a cycle"),
])
def test_malformed_recipe_requests_are_refused_by_name(options, words):
    with pytest.raises(ValueError, match=words):
        EnsembleRequest(**options)


def test_recipe_flag_without_a_member_count_names_the_remedy():
    with pytest.raises(ValueError, match="--members N"):
        request_for_payload(b"", recipe="time-lagged")


@pytest.mark.parametrize("suffix", [".json", ".toml"])
def test_trajectory_file_reads_json_and_toml(suffix, tmp_path):
    path = tmp_path / ("members" + suffix)
    path.write_text(json.dumps(MULTI) if suffix == ".json" else
                    "".join('[[trajectories]]\nsource="%s"\ncycle="%s"\n' % (item["source"], item["cycle"])
                            for item in MULTI))
    assert list(recipe_door.load_trajectories(path)) == MULTI
    overrides = recipe_door.flag_overrides(SimpleNamespace(recipe=None, trajectories=path))
    assert request_for_payload(b"", **overrides).recipe == "multi-model"
    with pytest.raises(recipe_door.RecipeRefusal, match="does not exist"):
        recipe_door.load_trajectories(tmp_path / "absent.json")


# ---- the plan --------------------------------------------------------------

def test_time_lagged_plan_is_earlier_cycles_of_the_configs_own_source():
    recipe = recipe_door.plan_recipe(EnsembleRequest(3, recipe="time-lagged"), payload(), experiment())
    assert recipe.kind == "time-lagged"
    assert [(member.trajectory.source, member.trajectory.cycle.hour) for member in recipe.members] == [
        ("hrrr", 18), ("hrrr", 17), ("hrrr", 16)]
    assert [recipe.acquisition_window(member.trajectory)[0] for member in recipe.members] == [0, 1, 2]
    assert len({member.seed for member in recipe.members}) == 3


def test_multi_model_plan_keeps_the_listed_trajectories_in_order():
    recipe = recipe_door.plan_recipe(EnsembleRequest(2, recipe="multi-model", trajectories=MULTI),
                                     payload(), experiment())
    assert recipe.kind == "multi-model"
    assert [member.trajectory.source for member in recipe.members] == ["hrrr", "rap"]


def test_plan_refusals_name_what_would_break():
    request = EnsembleRequest(2, recipe="time-lagged")
    with pytest.raises(recipe_door.RecipeRefusal, match=r"no \[fetch\] source and cycle"):
        recipe_door.plan_recipe(request, {"case_data": {}}, experiment())
    with pytest.raises(recipe_door.RecipeRefusal, match="one domain per member"):
        recipe_door.plan_recipe(request, payload(), experiment(domains=2))
    with pytest.raises(recipe_door.RecipeRefusal, match="forecast_start_hour"):
        recipe_door.plan_recipe(request, payload(forecast_start_hour=3), experiment())
    one_model = [{"source": "hrrr", "cycle": "2026-10-01T18"}, {"source": "hrrr", "cycle": "2026-10-01T17"}]
    with pytest.raises(recipe_door.RecipeRefusal, match="two distinct source models"):
        recipe_door.plan_recipe(EnsembleRequest(2, recipe="multi-model", trajectories=one_model),
                                payload(), experiment())


def test_the_plan_can_be_asked_at_another_cycle_without_a_config_written_for_it():
    """``--cycle`` with ``--readiness``: the same roster, moved with its window."""
    recipe = recipe_door.plan_recipe(EnsembleRequest(2, recipe="time-lagged"), payload(), experiment(),
                                     cycle="2026-10-02T06")
    assert [f"{member.trajectory.cycle:%Y-%m-%dT%H}" for member in recipe.members] == [
        "2026-10-02T06", "2026-10-02T05"]
    assert recipe.start == datetime(2026, 10, 2, 6, tzinfo=timezone.utc)
    assert (recipe.end - recipe.start).total_seconds() == 3600.0


# ---- one model run is one member (multi-model) ------------------------------

@pytest.mark.parametrize("sources, named", [
    (["hrrr", "hrrr-prs", "hrrr-native"], ["hrrr and hrrr-prs and hrrr-native"]),
    (["hrrr", "hrrr-prs"], ["hrrr and hrrr-prs"]),
    # The alias spelling of the same pressure-level product.
    (["hrrr", "hrrr-wrfprs"], ["hrrr and hrrr-prs"]),
    (["hrrr", "rap", "hrrr-prs", "rap-native"], ["hrrr and hrrr-prs", "rap and rap-native"]),
])
def test_multi_model_refuses_one_model_run_listed_under_several_source_ids(sources, named):
    """Breakage it prevents: the file products of ONE model run were planned
    as that many members, so the published spread came only from how each
    product is prepared."""
    listed = [{"source": source, "cycle": "2026-10-01T18"} for source in sources]
    with pytest.raises(recipe_door.RecipeRefusal, match="file products of one") as refused:
        recipe_door.plan_recipe(EnsembleRequest(len(listed), recipe="multi-model", trajectories=listed),
                                payload(), experiment())
    for words in named:
        assert words in str(refused.value)
    assert "would inflate effective ensemble size" in str(refused.value)


def test_model_identity_is_the_source_rows_upstream_model():
    from gpuwm.ensemble.recipes import SourceTrajectory
    cycle = datetime(2026, 10, 1, 18, tzinfo=timezone.utc)
    models = {source: SourceTrajectory(source, cycle).model
              for source in ("hrrr", "hrrr-prs", "hrrr-native", "rap", "rap-native", "gfs")}
    assert models == {"hrrr": "hrrr", "hrrr-prs": "hrrr", "hrrr-native": "hrrr",
                      "rap": "rap", "rap-native": "rap", "gfs": "gfs"}
    # Two different runs of one model are still refused as one model, not as a repeat.
    assert (SourceTrajectory("hrrr", cycle).model_run
            != SourceTrajectory("hrrr", cycle.replace(hour=17)).model_run)


# ---- the config's own source member (time-lagged) ----------------------------

def test_time_lagged_members_keep_the_configs_own_source_member():
    """Breakage it prevents: a config emitted for GEFS member p05 ran the
    control member c00 at every cycle, member 0 included."""
    request = EnsembleRequest(3, recipe="time-lagged")
    described = payload("gefs", member="p05", hours=12, cadence=3)
    described["experiment"]["run_seconds"] = 12 * 3600.0
    recipe = recipe_door.plan_recipe(request, described, experiment(hours=12))
    assert recipe.base.member == "p05"
    assert [member.trajectory.member for member in recipe.members] == ["p05"] * 3
    assert [f"{member.trajectory.cycle:%dT%H}" for member in recipe.members] == ["01T18", "01T12", "01T06"]
    # A config that names no member still takes the route's default.
    default = dict(described, fetch={key: value for key, value in described["fetch"].items()
                                     if key != "member"})
    assert recipe_door.plan_recipe(request, default, experiment(hours=12)).base.member == "c00"


def test_member_configs_carry_the_configs_source_member(tmp_path):
    config = _case(tmp_path, MEMBERS, fetch='member = "p05"\n')
    recipe, plans = _reviewed(config, EnsembleRequest(3, recipe="time-lagged"), tmp_path / "scratch")
    assert [plan.fetch["member"] for plan in plans] == ["p05", "p05", "p05"]
    assert [plan.line for plan in plans] == [
        "member 0: gefs 2026-08-20T00Z member p05", "member 1: gefs 2026-08-19T18Z member p05",
        "member 2: gefs 2026-08-19T12Z member p05"]


def test_planning_command_takes_a_base_member(capsys):
    from gpuwm.ensemble import recipes
    assert recipes.main(["--source", "gefs", "--cycle", "2026-10-01T18:00:00+00:00", "--hours", "12",
                         "--members", "2", "--recipe", "time-lagged", "--member", "p05"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [member["trajectory"]["member"] for member in plan["members"]] == ["p05", "p05"]


# ---- the session guard ------------------------------------------------------

def test_a_recipe_session_without_member_sources_refuses_to_run_copies(tmp_path):
    from gpuwm.ensemble.production import PreparedEnsembleSession
    session = PreparedEnsembleSession({"members": 2, "recipe": "time-lagged"}, output_directory=tmp_path)
    with pytest.raises(ValueError, match="needs each member's own prepared source"):
        session.run_prepared(lambda *a, **k: pytest.fail("a copy of one trajectory was run"), object())
    with pytest.raises(ValueError, match="needs each member's own prepared source"):
        session.run_experiment(lambda *a, **k: pytest.fail("a copy was run"), object(), object(), tmp_path)


# ---- routing ------------------------------------------------------------------

def _shipped_like_config(tmp_path, extra=""):
    config = tmp_path / "case.toml"
    config.write_text("[experiment]\nname='case'\n" + extra)
    return config


@pytest.mark.parametrize("command", ["go", "ensemble"])
@pytest.mark.parametrize("how", ["flag", "table"])
def test_go_and_ensemble_route_a_recipe_to_the_door(command, how, tmp_path, monkeypatch):
    from gpuwm import go_cli
    config = _shipped_like_config(
        tmp_path, "" if how == "flag" else '[ensemble]\nmembers=2\nrecipe="time-lagged"\n')
    reached = []
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble",
                        lambda args, request, observer=None, options=None: reached.append(request) or 0)
    monkeypatch.setattr(go_cli, "_go_launch", lambda *a, **k: pytest.fail("the single-trajectory chain ran"))
    args = SimpleNamespace(config=config, command=command,
                           members=2 if how == "flag" else None,
                           recipe="time-lagged" if how == "flag" else None, trajectories=None)
    assert go_cli.go_main(args) == 0
    assert [request.recipe for request in reached] == ["time-lagged"]
    assert reached[0].members == 2


def test_run_routes_a_recipe_to_the_door_and_refuses_input_directories(tmp_path, monkeypatch):
    from gpuwm import cli, capabilities, provenance_gate
    config = _shipped_like_config(tmp_path)
    reached = []
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble",
                        lambda args, request, observer=None, options=None: reached.append(request) or 0)
    monkeypatch.setattr(provenance_gate, "announce", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)
    assert cli.main(["run", str(config), "--recipe", "time-lagged", "--members", "2",
                     "--outdir", str(tmp_path / "out")]) == 0
    assert [request.recipe for request in reached] == ["time-lagged"]
    assert cli.main(["run", "--wrfinput", str(tmp_path), "--recipe", "time-lagged", "--members", "2",
                     "--outdir", str(tmp_path / "out")]) == 2
    assert len(reached) == 1


def test_member_inputs_reapply_the_shared_preflight_to_the_members_bundle(tmp_path, monkeypatch):
    from gpuwm import prepared_single_domain_forecast, stage_cli
    seen = {}
    monkeypatch.setattr(stage_cli, "resolve_bundle", lambda root: {"source": "hrrr", "root": root})
    monkeypatch.setattr(stage_cli, "single_domain_digests", lambda bundle: {
        "proof": "p" * 64, "source_manifest": "m" * 64, "prepared_content": "c" * 64})
    monkeypatch.setattr(prepared_single_domain_forecast, "preflight_prepared_forecast",
                        lambda **arguments: seen.update(arguments) or "member-inputs")
    shared = SimpleNamespace(preflight_arguments={
        "source": "hrrr", "prepared_root": tmp_path / "base", "source_manifest_sha256": "0" * 64,
        "experiment_config": tmp_path / "base.toml", "wps_namelist": tmp_path / "base.wps",
        "physics_profile": "profile", "run_seconds": 3600.0, "history_interval_seconds": 900.0})
    prepared = {"prepared_root": str(tmp_path / "member"), "experiment_config": str(tmp_path / "m.toml"),
                "wps_namelist": str(tmp_path / "m.wps")}
    assert recipe_door.member_inputs(shared, prepared) == "member-inputs"
    assert seen["prepared_root"] == (tmp_path / "member").resolve()
    assert seen["experiment_config"] == tmp_path / "m.toml" and seen["wps_namelist"] == tmp_path / "m.wps"
    assert (seen["proof_sha256"], seen["source_manifest_sha256"], seen["prepared_content_sha256"]) == (
        "p" * 64, "m" * 64, "c" * 64)
    # The forecast controls are the base member's, unchanged.
    assert (seen["physics_profile"], seen["run_seconds"], seen["history_interval_seconds"]) == (
        "profile", 3600.0, 900.0)


# ---- the door body: dry run, member configs, a whole run ------------------------

@pytest.mark.parametrize("command", ["ensemble", "go"])
def test_dry_run_prints_the_member_plan_and_leaves_the_disk_alone(command, tmp_path, door, stages):
    config = _case(tmp_path)
    out = tmp_path / "out"
    result = door(command, config, *RECIPE_FLAGS, "--outdir", out, "--dry-run")
    assert result.code == 0, result.text
    assert "ensemble: time-lagged recipe, 2 members, valid 2026-08-20T00:00Z to 2026-08-20T01:00Z" in result.out
    assert "ensemble: member 0: hrrr 2026-08-20T00Z" in result.out
    assert "ensemble: member 1: hrrr 2026-08-19T23Z" in result.out
    assert "dry run: nothing was fetched, prepared or run" in result.out
    assert not out.exists()
    assert stages.prepared == [] and stages.admitted == [] and stages.forecast == []


def test_a_lagged_member_config_changes_only_its_fetch_table(tmp_path):
    config = _case(tmp_path)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    recipe, plans = _reviewed(config, EnsembleRequest(2, recipe="time-lagged"), tmp_path / "scratch")
    written = [tomllib.loads((tmp_path / "scratch" / f"member-{index:03d}" / "experiment.toml")
                             .read_text(encoding="utf-8")) for index in (0, 1)]
    assert written[0]["fetch"] == {**raw["fetch"]} == plans[0].fetch
    assert written[1]["fetch"] == {**raw["fetch"], "cycle": "2026-08-19T23", "forecast_start_hour": 1}
    for document in written:
        assert "ensemble" not in document
        assert {key: value for key, value in document.items() if key != "fetch"} == {
            key: value for key, value in raw.items() if key not in ("fetch", "ensemble")}
    # A member of the config's own source is prepared from the namelist the domain door wrote.
    for index in (0, 1):
        assert ((tmp_path / "scratch" / f"member-{index:03d}" / "experiment.namelist.wps").read_bytes()
                == config.with_name("case.namelist.wps").read_bytes())
    assert [plan.chain for plan in plans] == ["prepared:hrrr", "prepared:hrrr"]
    assert [plan.start_lead for plan in plans] == [0, 1]


def test_a_member_from_another_model_gets_that_models_own_fetch_table(tmp_path):
    """Its cadence, its crop box where its fetch takes one, and a WPS
    namelist rewritten for its forcing interval.

    Breakage it prevents: a GFS member of a non-GFS config was written a
    table with no ``area``, which its chain's own planner requires, so the
    run stopped at that member after every earlier one was fetched and
    prepared, and ``--dry-run`` reported the plan as fine.
    """
    from gpuwm import go_cli
    config = _case(tmp_path, hours=6)
    listed = [{"source": "hrrr", "cycle": "2026-08-20T00"}, {"source": "gfs", "cycle": "2026-08-20T00"},
              {"source": "rap", "cycle": "2026-08-20T00"}]
    recipe, plans = _reviewed(config, EnsembleRequest(3, recipe="multi-model", trajectories=listed),
                              tmp_path / "scratch")
    assert [plan.chain for plan in plans] == ["prepared:hrrr", "prepared:go", "prepared:staged"]
    gfs, rap = plans[1].fetch, plans[2].fetch
    assert {key: gfs[key] for key in ("source", "cycle", "hours", "cadence")} == {
        "source": "gfs", "cycle": "2026-08-20T00", "hours": 6, "cadence": 3}
    south, west, north, east = (float(value) for value in gfs["area"].split(","))
    # The domain's own crop box for this source: it holds the domain's centre.
    assert south < 35.3 < north and west < -97.5 < east
    assert gfs["area"] != tomllib.loads(config.read_text(encoding="utf-8"))["fetch"]["area"]
    # A source whose fetch takes whole published objects is written no crop.
    assert rap == {"source": "rap", "cycle": "2026-08-20T00", "hours": 6, "cadence": 1}
    # The chain's own planner accepts the member the door wrote.
    member = tmp_path / "scratch" / "member-001"
    plan = go_cli.plan_from_config(member / "experiment.toml", outdir=tmp_path / "plan",
                                   run_stamp=False, data_dir=tmp_path / "data")
    assert (plan["source"], plan["hours"], plan["cadence"], plan["area"]) == ("gfs", 6, 3, gfs["area"])
    assert "interval_seconds = 10800" in (member / "experiment.namelist.wps").read_text(encoding="utf-8")
    assert "interval_seconds = 3600" in (
        tmp_path / "scratch" / "member-002" / "experiment.namelist.wps").read_text(encoding="utf-8")


def test_a_member_its_chain_cannot_prepare_is_refused_before_any_fetch(tmp_path, door, stages):
    """The dry run and the real run both meet the refusal in the plan."""
    config = _case(tmp_path, hours=6)
    listed = _listed(tmp_path, [{"source": "hrrr", "cycle": "2026-08-20T00"},
                                {"source": "icon-eu", "cycle": "2026-08-20T00"}])
    for extra in (["--dry-run"], []):
        result = door("ensemble", config, "--trajectories", listed, "--outdir", tmp_path / "out", *extra)
        assert result.code == 2, result.text
        assert "member 1 (icon-eu 2026-08-20T00Z) cannot be prepared" in result.err
    assert stages.prepared == [] and not (tmp_path / "out").exists()


def test_a_whole_recipe_run_records_every_member_and_completes(tmp_path, door, stages):
    config = _case(tmp_path)
    out = tmp_path / "out"
    result = door("ensemble", config, *RECIPE_FLAGS, "--outdir", out, "--products", "none")
    assert result.code == 0, result.text
    run, receipt = _receipt(out)
    assert receipt["status"] == "complete" and "failure" not in receipt
    assert receipt["recipe"]["kind"] == "time-lagged" and receipt["request"]["recipe"] == "time-lagged"
    assert [member["member_id"] for member in receipt["members"]] == [0, 1]
    assert [member["start_lead_hours"] for member in receipt["members"]] == [0, 1]
    assert [member["cycle"][:13] for member in receipt["members"]] == ["2026-08-20T00", "2026-08-19T23"]
    assert receipt["products"] == {"frames": 1}
    # Each member was prepared, prepare-only, from its own config in the run folder.
    assert [row["member"] for row in stages.prepared] == [0, 1]
    assert all(row["prepare_only"] for row in stages.prepared)
    # Compared by their parts: on Windows a deep output folder is handed to
    # the chain in its extended spelling.
    assert [row["config"].parts[-5:] for row in stages.prepared] == [
        (run.name, "members", f"member-{index:03d}", "source", "experiment.toml") for index in (0, 1)]
    assert [row["fetch"]["cycle"] for row in stages.prepared] == ["2026-08-20T00", "2026-08-19T23"]
    # One download cache per member's request, under the case folder.
    caches = [Path(row["run_options"]["data_dir"]) for row in stages.prepared]
    assert len(set(caches)) == 2
    assert all(cache.parts[-3:-1] == ("out", "downloads") for cache in caches)
    # The gates were asked once, of both members, before the first preparation.
    assert stages.admitted == [["member 0: hrrr 2026-08-20T00Z", "member 1: hrrr 2026-08-19T23Z"]]
    assert len(stages.forecast) == 1 and stages.session.input_provider is not None
    assert stages.sim[0]["outdir"].parts[-2:] == (run.name, "run")
    assert stages.sim[0]["render_products"] == "none"


# ---- gpuwm go / ensemble flags on the recipe route -------------------------------

@pytest.mark.parametrize("how", ["flag", "table"])
@pytest.mark.parametrize("command", ["ensemble", "go"])
def test_readiness_answers_for_every_member_window_and_runs_nothing(command, how, tmp_path, door, stages):
    """Breakage it prevents: the recipe route returned to the door before
    ``--readiness`` was read, so a scheduler's poll claimed a run folder and
    began the members' downloads, with the card check waived."""
    config = _case(tmp_path, table="" if how == "flag" else '[ensemble]\nmembers = 2\nrecipe = "time-lagged"\n')
    out = tmp_path / "out"
    result = door(command, config, *(RECIPE_FLAGS if how == "flag" else []),
                  "--readiness", "--no-probe", "--outdir", out)
    document = json.loads(result.out)
    assert document["schema"] == "gpuwm.readiness.v1"
    windows = document["recipe"]["member_windows"]
    assert document["recipe"]["kind"] == "time-lagged" and document["recipe"]["members"] == 2
    assert [(row["source"], row["cycle"]) for row in windows] == [
        ("hrrr", "2026-08-20T00"), ("hrrr", "2026-08-19T23")]
    assert [row["readiness"]["window"]["start_lead"] for row in windows] == [0, 1]
    assert result.code == max(row["exit_code"] for row in windows) or result.code == 2
    assert "readiness" in result.err and "recipe members" in result.err
    assert not out.exists()
    assert stages.prepared == [] and stages.admitted == [] and stages.forecast == []


def test_readiness_is_the_worst_member_window(monkeypatch):
    """Ready only when every member's window is; the member ready last decides the retry."""
    from gpuwm import go_cli
    answers = iter([
        ({"state": "ready", "ready": True, "expected_ready_at": "2026-10-01T18:50:00Z",
          "retry_after_seconds": None, "refusal": None, "cycle": "2026-10-01T18"}, 0),
        ({"state": "waiting", "ready": False, "expected_ready_at": "2026-10-01T19:40:00Z",
          "retry_after_seconds": 120.0, "refusal": None, "cycle": "2026-10-01T17"}, 75),
        ({"state": "waiting", "ready": False, "expected_ready_at": "2026-10-01T19:10:00Z",
          "retry_after_seconds": 30.0, "refusal": None, "cycle": "2026-10-01T16"}, 75)])
    asked = []
    monkeypatch.setattr(go_cli, "_readiness_answer", lambda payload, options, pinned, *, no_probe: (
        asked.append((payload["fetch"]["cycle"], payload["fetch"].get("forecast_start_hour", 0),
                      dict(options), no_probe)) or next(answers)))
    document, code = go_cli.recipe_readiness(
        EnsembleRequest(3, recipe="time-lagged"), payload(), experiment(), cycle=None,
        posting={"as_posted": False}, no_probe=True)
    assert code == 75
    assert (document["state"], document["ready"]) == ("waiting", False)
    assert document["retry_after_seconds"] == 120.0
    assert document["expected_ready_at"] == "2026-10-01T19:40:00Z"
    assert document["recipe"]["answered_by_member"] == 1 and document["cycle"] == "2026-10-01T17"
    assert [row["exit_code"] for row in document["recipe"]["member_windows"]] == [0, 75, 75]
    # Every member's own window was asked, under the run's posting rule.
    assert asked == [("2026-10-01T18", 0, {"as_posted": False}, True),
                     ("2026-10-01T17", 1, {"as_posted": False}, True),
                     ("2026-10-01T16", 2, {"as_posted": False}, True)]


def test_a_refused_member_window_refuses_the_readiness_answer(monkeypatch):
    from gpuwm import go_cli
    answers = iter([({"state": "ready", "ready": True, "expected_ready_at": None,
                      "retry_after_seconds": None, "refusal": None}, 0),
                    ({"state": "refused", "ready": False, "expected_ready_at": None,
                      "retry_after_seconds": None, "refusal": "past the archive"}, 2)])
    monkeypatch.setattr(go_cli, "_readiness_answer", lambda *a, **k: next(answers))
    document, code = go_cli.recipe_readiness(
        EnsembleRequest(2, recipe="time-lagged"), payload(), experiment(), cycle=None, posting={})
    assert code == 2 and document["state"] == "refused" and document["ready"] is False
    assert document["refusal"] == "member 1 (hrrr 2026-10-01T17Z): past the archive"


def test_cycle_flag_retimes_the_roster_the_door_plans(tmp_path, door, stages):
    """Breakage it prevents: ``--cycle`` was parsed and dropped on the recipe
    route, so the ensemble was planned and run at the config's own cycle."""
    config = _case(tmp_path)
    out = tmp_path / "out"
    result = door("ensemble", config, *RECIPE_FLAGS, "--cycle", "2026-08-21T06", "--dry-run",
                  "--outdir", out)
    assert result.code == 0, result.text
    assert "re-timed to start 2026-08-21 06:00:00 UTC" in result.out
    assert "valid 2026-08-21T06:00Z to 2026-08-21T07:00Z" in result.out
    assert "member 0: hrrr 2026-08-21T06Z" in result.out and "member 1: hrrr 2026-08-21T05Z" in result.out
    # The real run prepares the re-timed members, not the config's own.
    result = door("ensemble", config, *RECIPE_FLAGS, "--cycle", "2026-08-21T06", "--outdir", out)
    assert result.code == 0, result.text
    assert [row["fetch"]["cycle"] for row in stages.prepared] == ["2026-08-21T06", "2026-08-21T05"]
    refused = door("ensemble", config, *RECIPE_FLAGS, "--cycle", "not-a-cycle", "--dry-run")
    assert refused.code == 2 and "not-a-cycle" in refused.err


def test_posting_and_host_flags_reach_every_members_fetch(tmp_path, door, stages):
    config = _case(tmp_path)
    result = door("go", config, *RECIPE_FLAGS, "--outdir", tmp_path / "out", "--transport", "nomads",
                  "--whole-cycle")
    assert result.code == 0, result.text
    assert len(stages.prepared) == 2
    for row in stages.prepared:
        assert row["run_options"]["transport"] == "nomads"
        assert row["run_options"]["as_posted"] is False
        assert "late_after_minutes" not in row["run_options"]
    stages.prepared.clear()
    result = door("go", config, *RECIPE_FLAGS, "--outdir", tmp_path / "out2", "--late-after-minutes", "7")
    assert result.code == 0, result.text
    assert [row["run_options"].get("late_after_minutes") for row in stages.prepared] == [7.0, 7.0]
    assert all("transport" not in row["run_options"] and "as_posted" not in row["run_options"]
               for row in stages.prepared)


def test_a_host_a_member_source_cannot_pin_is_refused_in_the_plan(tmp_path, door, stages):
    config = _case(tmp_path)
    listed = _listed(tmp_path, [{"source": "hrrr", "cycle": "2026-08-20T00"},
                                {"source": "rap", "cycle": "2026-08-20T00"}])
    result = door("ensemble", config, "--trajectories", listed, "--transport", "s3", "--dry-run")
    assert result.code == 2
    assert "member 1 (rap 2026-08-20T00Z) cannot be prepared" in result.err and "--transport s3" in result.err


@pytest.mark.parametrize("flag, value", [
    ("--prepared-root", "DIR"), ("--restart", "FILE"), ("--data-dir", "DIR"),
    ("--supplement", "PMSL=FILE"), ("--section", "35.0,-98.0,36.0,-97.0"), ("--keep-checkpoints", "0")])
def test_go_flags_the_recipe_route_cannot_use_are_refused_by_name(flag, value, tmp_path, door, stages):
    """Breakage it prevents: each was parsed and read by nothing, so a
    restart became a fresh fetch and forecast at exit 0."""
    config = _case(tmp_path)
    (tmp_path / "DIR").mkdir()
    (tmp_path / "FILE").write_text("x")
    value = value.replace("DIR", str(tmp_path / "DIR")).replace("FILE", str(tmp_path / "FILE"))
    for extra in ([], ["--dry-run"]):
        result = door("go", config, *RECIPE_FLAGS, "--outdir", tmp_path / "out", flag, value, *extra)
        assert result.code == 2, result.text
        assert f"An ensemble recipe does not use {flag}" in result.err
    assert stages.prepared == [] and not (tmp_path / "out").exists()


def test_no_probe_still_belongs_to_readiness_on_the_recipe_route(tmp_path, door, stages):
    result = door("go", _case(tmp_path), *RECIPE_FLAGS, "--no-probe", "--dry-run")
    assert result.code == 2 and "--no-probe belongs to --readiness" in result.err


# ---- gpuwm run: unsupervised, and its supervision flags ---------------------------

@pytest.mark.parametrize("flag, value", [
    ("--restart", "ckpt.npz"), ("--gpu-uuid", "GPU-x"), ("--supervisor-max-restarts", "5"),
    ("--prep-timeout", "30"), ("--allow-shared-gpu", None), ("--health-debug", None),
    ("--preprocess-backend", "cpu"), ("--directory-input-hash", "content")])
def test_run_supervision_flags_a_recipe_cannot_use_are_refused_by_name(flag, value, tmp_path, door, stages):
    """Breakage it prevents: the recipe runs ahead of the supervisor, so each
    flag was parsed and dropped; a ``--gpu-uuid`` pin ran on every card."""
    config = _case(tmp_path)
    result = door("run", config, *RECIPE_FLAGS, "--outdir", tmp_path / "out", flag,
                  *([] if value is None else [value]))
    assert result.code == 2, result.text
    assert "runs in this process, unsupervised" in result.err and flag in result.err
    assert stages.prepared == [] and not (tmp_path / "out").exists()


def test_run_recipe_without_supervision_flags_runs_and_no_supervise_is_true_of_it(tmp_path, door, stages, monkeypatch):
    from gpuwm import supervisor
    monkeypatch.setattr(supervisor, "supervise_from_cli",
                        lambda args: pytest.fail("a recipe run entered the supervisor"))
    result = door("run", _case(tmp_path), *RECIPE_FLAGS, "--no-supervise", "--outdir", tmp_path / "out")
    assert result.code == 0, result.text
    assert _receipt(tmp_path / "out")[1]["status"] == "complete"


# ---- resume and branch continue one trajectory -------------------------------------

@pytest.mark.parametrize("command", ["resume", "branch"])
def test_resume_and_branch_do_not_take_the_recipe_flags(command, tmp_path, capsys):
    """Breakage it prevents: both parsers accepted ``--recipe`` and
    ``--trajectories`` and nothing read them, so the continued run was N
    copies of the checkpointed trajectory."""
    from gpuwm.cli import build_parser
    parser = build_parser()
    for flags in (["--recipe", "time-lagged"], ["--trajectories", str(tmp_path / "members.json")]):
        _args, unparsed = parser.parse_known_args(
            [command, str(tmp_path / "case.toml"), "--outdir", str(tmp_path / "out"), "--members", "2", *flags])
        assert unparsed == flags
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--help"])
    usage = capsys.readouterr().out
    assert "--members" in usage and "--recipe" not in usage and "--trajectories" not in usage
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--help"])
    assert "--recipe" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["resume", "branch"])
def test_resume_and_branch_refuse_a_config_that_selects_a_recipe(command, tmp_path, door, monkeypatch):
    """Before a checkpoint is looked for, a branch folder written or a worker started."""
    from gpuwm import branch, resume, supervisor
    config = _case(tmp_path, table='[ensemble]\nmembers = 2\nrecipe = "time-lagged"\n')
    for module, name in ((supervisor, "supervise_from_cli"), (resume, "resolve_resume_checkpoint"),
                         (branch, "prepare_branch_from_cli")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail(f"{name} was reached"))
    result = door(command, config, "--outdir", tmp_path / "out")
    assert result.code == 2, result.text
    assert f"gpuwm {command} continues one prepared trajectory" in result.err
    assert "selects an ensemble recipe" in result.err
    assert not (tmp_path / "out").exists()


# ---- failures: one line, the stage's exit code, a receipt that says failed --------

def _stage_failures():
    from gpuwm.go_cli import GoInterrupted, GoStageFailed
    from gpuwm.runplan import StageExitError
    return [("source-behind", StageExitError("fetch", 75), 75, "failed"),
            ("stage-exit", GoStageFailed(3, "the preparer's last lines"), 3, "failed"),
            ("interrupted-stage", GoInterrupted("prepare", 4321), 130, "interrupted"),
            ("ctrl-c", KeyboardInterrupt(), 130, "interrupted")]


@pytest.mark.parametrize("name", ["source-behind", "stage-exit", "interrupted-stage", "ctrl-c"])
@pytest.mark.parametrize("command", ["ensemble", "run"])
def test_a_member_stage_failure_exits_with_its_code_and_marks_the_receipt(name, command, tmp_path, door, stages):
    """Breakage it prevents: the chains' own failures left the door as a
    traceback at exit 1 (75 and 130 included) with the receipt still at
    ``preparing``."""
    error, code, status = next(row[1:] for row in _stage_failures() if row[0] == name)
    stages.fail[1] = error
    out = tmp_path / "out"
    result = door(command, _case(tmp_path), *RECIPE_FLAGS, "--outdir", out)
    assert result.code == code
    assert "Traceback" not in result.text
    assert "during member 1's preparation" in result.err
    run, receipt = _receipt(out)
    assert receipt["status"] == status
    assert receipt["failure"] == {"stage": "prepare", "member_id": 1, "exit_code": code,
                                  "error_class": type(error).__name__, "message": str(error)}
    assert [member["member_id"] for member in receipt["members"]] == [0]
    assert stages.forecast == []


def test_a_refusal_inside_a_member_chain_keeps_its_sentence_and_marks_the_receipt(tmp_path, door, stages):
    from gpuwm.runplan import PlanError
    stages.fail[0] = PlanError("the HRRR fetch wrote no SHA256SUMS")
    result = door("ensemble", _case(tmp_path), *RECIPE_FLAGS, "--outdir", tmp_path / "out")
    assert result.code == 2 and "the HRRR fetch wrote no SHA256SUMS" in result.err
    receipt = _receipt(tmp_path / "out")[1]
    assert receipt["status"] == "failed" and receipt["failure"]["member_id"] == 0
    assert receipt["failure"]["exit_code"] == 2 and receipt["members"] == []


@pytest.mark.parametrize("outcome, code", [(RuntimeError("ordinary member returned a failing forecast receipt"), 1),
                                           (75, 75), (2, 2)])
def test_a_forecast_failure_is_one_line_and_a_failed_receipt(outcome, code, tmp_path, door, stages):
    stages.forecast_result = outcome
    out = tmp_path / "out"
    result = door("ensemble", _case(tmp_path), *RECIPE_FLAGS, "--outdir", out)
    assert result.code == code and "Traceback" not in result.text
    run, receipt = _receipt(out)
    assert receipt["status"] == "failed"
    # This runner fails the stage itself and names no member, so the receipt
    # names none; a member's own failure is named (the real-session tests below).
    assert receipt["failure"]["stage"] == "forecast"
    assert receipt["failure"]["member_ids"] == [] and "failed_members" not in receipt["failure"]
    assert receipt["failure"]["exit_code"] == code
    assert [member["member_id"] for member in receipt["members"]] == [0, 1]
    if isinstance(outcome, BaseException):
        assert "the forecast failed: RuntimeError: ordinary member returned a failing forecast receipt" in result.err
        # The traceback is kept beside the receipt, not thrown away.
        assert "RuntimeError" in (run / recipe_door.FAILURE_LOG_NAME).read_text(encoding="utf-8")


# ---- forecast failures over the real ensemble session ------------------------------

class _Collector:
    """The aggregate product owner, without a renderer: these runs never finish."""

    def __init__(self):
        self.rows, self.finished = [], False

    def submit(self, **row):
        self.rows.append(row)

    def finish_run(self):
        self.finished = True
        return {"frames": len(self.rows)}

    def require_complete(self):
        return {}


@pytest.fixture
def real_session(monkeypatch, stages):
    """The door over the real ``PreparedEnsembleSession`` and the real runner hand-off.

    Replaced, beyond ``stages``' member fetch/prepare owner and ``admit``:
    the card reading (one CPU ``CardBudget``), each member's preflight
    (``recipe_door.member_inputs``) and the forecast body a member runs.
    The session, its packing, wave executor, progress adapter, member
    naming and ``run/ensemble-run.json`` are its own; the runner's
    ``run_prepared_forecast`` hands the forecast to the session as it does
    in a real run, and its command line maps a memory refusal to exit 2 as
    the real one does.
    """
    from contextlib import nullcontext
    from gpuwm import prepared_single_domain_forecast as forecast_module
    from gpuwm.ensemble import door as door_module, production
    from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from gpuwm.ensemble.packing import CardBudget
    from gpuwm.ensemble.runtime_context import current_capture, current_session
    from gpuwm.ingest.memory_refusal import InitializationMemoryRefused

    record = SimpleNamespace(fail={}, ran=[], refuse_inputs={}, collector=_Collector())
    real_session_class = production.PreparedEnsembleSession

    def session(request, *, output_directory):
        return real_session_class(request, output_directory=output_directory,
            cards=(CardBudget(0, 1000),), device_scope=lambda _: nullcontext(),
            memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)),
            collector=record.collector)

    def member_inputs(shared_inputs, prepared):
        member = int(Path(prepared["prepared_root"]).parent.parent.name.rsplit("-", 1)[1])
        if member in record.refuse_inputs:
            raise record.refuse_inputs[member]
        return SimpleNamespace(experiment=shared_inputs.experiment,
                               boundary_interval_seconds=shared_inputs.boundary_interval_seconds)

    real_forecast = forecast_module.run_prepared_forecast

    def run_prepared_forecast(inputs, *, output_directory, observer=None, **options):
        if current_session() is not None:
            # The runner's own hand-off to the session.
            return real_forecast(inputs, output_directory=output_directory, observer=observer, **options)
        member = current_capture().member_id
        record.ran.append(member)
        for step in range(1, 4):
            observer(model_elapsed_seconds=20.0 * step, outer_step=step)
        if member in record.fail:
            raise record.fail[member]
        return {"status": "PASS"}

    exp = SimpleNamespace(run_seconds=60.0, start_time=datetime(2026, 8, 20),
                          root=SimpleNamespace(run=SimpleNamespace(dt=3, use_adaptive_time_step=True,
                                                                   mp_physics=16)))
    shared = SimpleNamespace(experiment=exp, boundary_interval_seconds=3600)

    def main(argv, observer=None):
        outdir = Path(argv[argv.index("--outdir") + 1])
        try:
            forecast_module.run_prepared_forecast(shared, output_directory=outdir, observer=observer)
        except InitializationMemoryRefused as error:
            print(f"prepared_single_domain_forecast: {error}", file=sys.stderr)
            return 2
        return 0

    monkeypatch.setattr(door_module, "production_run_scope", REAL_RUN_SCOPE)
    monkeypatch.setattr(production, "PreparedEnsembleSession", session)
    monkeypatch.setattr(recipe_door, "member_inputs", member_inputs)
    monkeypatch.setattr(forecast_module, "run_prepared_forecast", run_prepared_forecast)
    monkeypatch.setattr(forecast_module, "main", main)
    return record


def _run_record(run):
    return json.loads((run / "run" / "ensemble-run.json").read_text(encoding="utf-8"))


def test_a_member_forecast_failure_names_the_member_in_the_receipt_and_the_line(tmp_path, door, real_session):
    """Breakage it prevents: run control named member 1 in ``ensemble-run.json``
    while ``ensemble-recipe.json`` recorded ``member_id: null`` and the door's
    line said only that the forecast failed."""
    real_session.fail[1] = FloatingPointError(
        "full-state health gate failed during post-d01-sync.d01: qv(3, 10, 12): non-finite")
    out = tmp_path / "out"
    result = door("ensemble", _case(tmp_path), *RECIPE_FLAGS, "--outdir", out)
    assert result.code == 1 and real_session.ran == [0, 1]
    run, receipt = _receipt(out)
    failure = receipt["failure"]
    assert receipt["status"] == "failed"
    assert failure["stage"] == "forecast" and failure["member_id"] == 1 and failure["member_ids"] == [1]
    assert failure["error_class"] == "FloatingPointError" and failure["exit_code"] == 1
    assert failure["message"].startswith("member 1: full-state health gate failed")
    (row,) = failure["failed_members"]
    assert row["member_ids"] == [1] and row["device_id"] == 0 and row["error_type"] == "FloatingPointError"
    assert _run_record(run)["failed_members"] == failure["failed_members"]
    assert "ensemble: the forecast of member 1 failed: FloatingPointError" in result.err
    assert not real_session.collector.finished


def test_a_member_error_whose_text_is_not_its_message_is_still_named(tmp_path, door, real_session):
    """Breakage it prevents: a KeyError keeps its own text ("'smois'"), so only
    the traceback log named the member; the receipt and the line did not."""
    real_session.fail[1] = KeyError("smois")
    out = tmp_path / "out"
    result = door("ensemble", _case(tmp_path), *RECIPE_FLAGS, "--outdir", out)
    assert result.code == 1
    failure = _receipt(out)[1]["failure"]
    assert failure["member_id"] == 1 and failure["member_ids"] == [1]
    assert failure["error_class"] == "KeyError" and failure["message"] == "'smois'"
    assert "ensemble: the forecast of member 1 failed: KeyError: 'smois'." in result.err


def test_a_runner_that_returns_a_code_leaves_the_members_own_error_in_the_receipt(tmp_path, door, real_session):
    """Breakage it prevents: a member's memory refusal reached the door as exit 2
    and the receipt said ``RuntimeError('the forecast stage exited 2')`` with no
    member, while ``ensemble-run.json`` held the real class and member."""
    from gpuwm.ingest.memory_refusal import InitializationMemoryRefused
    real_session.fail[1] = InitializationMemoryRefused("the forecast needs 30 GiB and the card has 24")
    out = tmp_path / "out"
    result = door("ensemble", _case(tmp_path), *RECIPE_FLAGS, "--outdir", out)
    assert result.code == 2
    run, receipt = _receipt(out)
    failure = receipt["failure"]
    assert receipt["status"] == "failed" and failure["exit_code"] == 2
    assert failure["error_class"] == "InitializationMemoryRefused"
    assert failure["message"] == "member 1: the forecast needs 30 GiB and the card has 24"
    assert failure["member_id"] == 1 and failure["member_ids"] == [1]
    assert failure["failed_members"] == _run_record(run)["failed_members"]
    assert "ensemble: stopped during the forecast of member 1 (exit 2)" in result.err


def test_a_member_whose_bundle_the_preflight_refuses_is_named(tmp_path, door, real_session):
    """Breakage it prevents: the door's own provider refused member 1's bundle
    and the receipt recorded ``member_id: null``."""
    real_session.refuse_inputs[1] = RuntimeError("member bundle digest differs from its receipt")
    out = tmp_path / "out"
    result = door("ensemble", _case(tmp_path), *RECIPE_FLAGS, "--outdir", out)
    assert result.code == 1 and real_session.ran == []
    failure = _receipt(out)[1]["failure"]
    assert failure["stage"] == "forecast" and failure["member_id"] == 1 and failure["member_ids"] == [1]
    assert "ensemble: the forecast of member 1 failed: RuntimeError" in result.err


def test_a_stop_signal_is_interrupted_with_its_own_code_and_keeps_the_process_status(
        tmp_path, door, real_session, capsys):
    """Breakage it prevents: a SystemExit stop (a SIGTERM handler, gpuwm's own
    ChildStopped) was ``interrupted`` to the session but ``failed``, exit 1, to
    the recipe receipt, and the door said nothing."""
    from gpuwm import cli
    real_session.fail[0] = SystemExit(143)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        cli.main([str(item) for item in ("ensemble", _case(tmp_path), *RECIPE_FLAGS, "--outdir", out)])
    assert caught.value.code == 143
    err = capsys.readouterr().err
    run, receipt = _receipt(out)
    assert receipt["status"] == "interrupted" == _run_record(run)["status"]
    assert receipt["failure"]["exit_code"] == 143 and receipt["failure"]["error_class"] == "SystemExit"
    assert receipt["failure"]["stage"] == "forecast"
    assert "ensemble: interrupted during the forecast (exit 143)." in err
    assert 1 not in real_session.ran


# ---- the gates the ordinary door asks, asked before the first fetch ----------------

def _admit_arguments(tmp_path, monkeypatch, *, members=2, free=None):
    """``recipe_door.admit`` on the shipped config, with every gate passing unless overridden."""
    from gpuwm import capabilities, disk_budget, go_cli, rustwx
    config = _case(tmp_path)
    request = EnsembleRequest(members, recipe="time-lagged")
    recipe, plans = _reviewed(config, request, tmp_path / "scratch")
    asked = []
    monkeypatch.setattr(capabilities, "require", lambda door, *a, **k: asked.append(("install", door)))
    monkeypatch.setattr(go_cli, "_require_forecast_device", lambda: asked.append(("device",)))
    monkeypatch.setattr(go_cli, "memory_gate", lambda plan, experiment=None: asked.append(
        ("memory", plan["source"], Path(plan["config"]).is_file())) or {
            "verdict": "fits", "refuse": False, "warn": False})
    monkeypatch.setattr(go_cli, "geography_refusal", lambda root: asked.append(("geography", root)))
    monkeypatch.setattr(rustwx, "find_renderer", lambda: asked.append(("renderer",)) or Path("rw_wrfbatch"))
    monkeypatch.setattr(go_cli, "admit_render_products",
                        lambda spec, section=None: asked.append(("products", spec)))
    if free is not None:
        monkeypatch.setattr(disk_budget, "free_bytes", lambda path: free)
    arguments = dict(config=config, geog_root=tmp_path / "GEOG", case_root=tmp_path / "out",
                     request=request, options={"render_products": "refl"}, command="ensemble")
    return plans, arguments, asked


def test_the_gates_are_asked_in_the_ordinary_doors_order(tmp_path, monkeypatch):
    plans, arguments, asked = _admit_arguments(tmp_path, monkeypatch)
    recipe_door.admit(plans, **arguments)
    assert [row[0] for row in asked] == ["install", "device", "memory", "geography", "renderer", "products"]
    assert asked[0] == ("install", "gpuwm ensemble")
    # The memory gate is asked once per source, of that member's own reviewed config.
    assert asked[2] == ("memory", "hrrr", True)
    assert asked[3] == ("geography", tmp_path / "GEOG") and asked[5] == ("products", "refl")
    assert not (tmp_path / "out").exists()
    asked.clear()
    recipe_door.admit(plans, **{**arguments, "options": {"no_memory_gate": True}})
    assert "memory" not in [row[0] for row in asked]


@pytest.mark.parametrize("gate, words", [
    ("install", "this command needs cupy"), ("device", "GPU readiness is missing"),
    ("memory", "does not fit this card"), ("geography", "WPS_GEOG tree is not usable"),
    ("renderer", "drawn by the Rust renderer"), ("products", "the renderer's catalog does not carry"),
    ("disk", "ensemble's 2 members would write")])
def test_each_gate_refuses_before_the_run_folder_is_claimed(gate, words, tmp_path, door, monkeypatch):
    """Breakage it prevents: the recipe door fetched and prepared every
    member before any of these was asked, where the ordinary door refuses
    in seconds; a cardless box began the members' downloads."""
    from gpuwm import capabilities, disk_budget, go_cli, rustwx
    reached = []
    monkeypatch.setattr(recipe_door, "prepare_member", lambda *a, **k: reached.append(1) or pytest.fail(
        "a member was fetched before the gates were asked"))
    passing = {
        "install": (capabilities, "require", lambda *a, **k: None),
        "device": (go_cli, "_require_forecast_device", lambda: None),
        "memory": (go_cli, "memory_gate", lambda plan, experiment=None: {
            "verdict": "fits", "refuse": False, "warn": False}),
        "geography": (go_cli, "geography_refusal", lambda root: None),
        "renderer": (rustwx, "find_renderer", lambda: Path("rw_wrfbatch")),
        "products": (go_cli, "admit_render_products", lambda spec, section=None: None),
        "disk": (disk_budget, "free_bytes", lambda path: 2 ** 60)}
    failing = {
        "install": lambda door, *a, **k: (_ for _ in ()).throw(capabilities.CapabilityMissing(
            f"{door}: this command needs cupy", requirement=capabilities.GPU_RUNTIME, command=door)),
        "device": lambda: (_ for _ in ()).throw(go_cli.GoRefusal("GPU readiness is missing: no card")),
        "memory": lambda plan, experiment=None: {"verdict": "over", "refuse": True, "warn": False},
        "geography": lambda root: "the staged WPS_GEOG tree is not usable",
        "renderer": lambda: None,
        "products": lambda spec, section=None: (_ for _ in ()).throw(go_cli.GoRefusal(
            "--products names 'bogus', which the renderer's catalog does not carry")),
        "disk": lambda path: 1}
    for name, (module, attribute, stub) in passing.items():
        monkeypatch.setattr(module, attribute, failing[name] if name == gate else stub)
    monkeypatch.setattr(go_cli, "memory_refusal_text", lambda gate: "this configuration does not fit this card")
    monkeypatch.setattr(rustwx, "renderer_remedy", lambda: "remedy: gpuwm setup")
    out = tmp_path / "out"
    result = door("ensemble", _case(tmp_path), *RECIPE_FLAGS, "--outdir", out)
    assert result.code == 2, result.text
    assert words in result.err
    assert reached == [] and not out.exists()


def test_disk_admission_prices_every_members_download_and_bundle(tmp_path, monkeypatch):
    from gpuwm import disk_budget
    gib = 2 ** 30
    rows = []

    def projected(exp, *, fetch, chain, **kw):
        rows.append((fetch["cycle"], chain, kw["download_present_bytes"]))
        return {"download_bytes": 10 * gib, "preparation_bytes": gib, "history_bytes": 2 * gib,
                "compose_scratch_bytes": 0, "compose_scratch_min_bytes": 0, "compose_scratch": {},
                "download": {"basis": "measured"}}

    monkeypatch.setattr(disk_budget, "projected_run_bytes", projected)
    config = _case(tmp_path)
    for members, keep, free, refused in ((2, False, 23, False), (2, False, 21, True),
                                         (3, False, 23, True), (2, True, 23, True), (2, True, 27, False)):
        request = EnsembleRequest(members, recipe="time-lagged", keep_member_files=keep)
        recipe, plans = _reviewed(config, request, tmp_path / f"scratch-{members}-{keep}-{free}")
        monkeypatch.setattr(disk_budget, "free_bytes", lambda path, free=free: free * gib)
        refusal = recipe_door.disk_refusal(plans, case_root=tmp_path / "out", request=request)
        assert (refusal is not None) == refused, (members, keep, free, refusal)
        if refused:
            assert f"This ensemble's {members} members would write" in refusal
            assert f"{free}.0 GiB free" in refusal and "Refused before the first download" in refusal
    # One row per member, each priced for its own cycle's request and chain.
    assert rows[:2] == [("2026-08-20T00", "prepared:hrrr", 0), ("2026-08-19T23", "prepared:hrrr", 0)]


# ---- run-plan: one door for every chain, refused where it cannot run ----------------

def _plan(tmp_path, config, *, route="prepared", **run_options):
    from gpuwm.runplan import PLAN_SCHEMA, load_plan
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": PLAN_SCHEMA, "name": "recipe-plan", "route": route,
                                "config": {"path": str(config)}, "output_root": str(tmp_path / "run"),
                                "run_options": run_options}), encoding="utf-8")
    return load_plan(path)


@pytest.mark.parametrize("stem", [HOURLY, MEMBERS, "gfs_12km_quickstart"])
@pytest.mark.parametrize("how", ["option", "table"])
def test_run_plan_hands_a_recipe_to_the_door_on_every_chain(stem, how, tmp_path, monkeypatch):
    """Breakage it prevents: on the hourly and staged chains a recipe plan
    fetched and prepared the config's ONE trajectory, and was refused only
    at the forecast stage, where the session finds no member sources."""
    from gpuwm import go_cli, runplan
    from gpuwm.ensemble.runtime_context import ensemble_scope
    from gpuwm.experiment import load_experiment
    wanted = {"members": 2, "recipe": "time-lagged"}
    config = _case(tmp_path, stem, table="" if how == "option" else
                   '[ensemble]\nmembers = 2\nrecipe = "time-lagged"\n')
    plan = _plan(tmp_path, config, **({"ensemble": wanted} if how == "option" else {}))
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    assert runplan._plan_recipe(plan, raw) == "time-lagged"
    assert runplan._recipe_plan_refusal(plan, raw, None) is None
    reached = []
    for chain in ("_hrrr_chain", "_staged_chain"):
        monkeypatch.setattr(runplan, chain, lambda *a, **k: pytest.fail(
            "a single-trajectory chain ran for a recipe request"))
    monkeypatch.setattr(go_cli, "go_main", lambda args, observer=None: reached.append(args) or 0)
    observer = SimpleNamespace(last_model_seconds=0.0, events=None)
    with ensemble_scope(SimpleNamespace(request=EnsembleRequest.from_mapping(wanted))):
        runplan._execute_prepared_route(plan, exp=load_experiment(config), data=None,
                                        config_path=config, observer=observer)
    (args,) = reached
    assert (args.command, Path(args.config)) == ("go", config)
    assert Path(args.outdir).parts[-2:] == (plan.run_dir.name, "chain")


def test_run_plan_without_a_recipe_keeps_its_own_chains(tmp_path, monkeypatch):
    from gpuwm import go_cli, runplan
    from gpuwm.ensemble.runtime_context import ensemble_scope
    from gpuwm.experiment import load_experiment
    config = _case(tmp_path)
    plan = _plan(tmp_path, config, ensemble={"members": 2})
    assert runplan._plan_recipe(plan, tomllib.loads(config.read_text(encoding="utf-8"))) is None
    ran = []
    monkeypatch.setattr(runplan, "_hrrr_chain", lambda plan, **kw: ran.append("hourly") or {})
    monkeypatch.setattr(go_cli, "go_main", lambda *a, **k: pytest.fail("the go arm ran"))
    with ensemble_scope(SimpleNamespace(request=EnsembleRequest(2))):
        runplan._execute_prepared_route(plan, exp=load_experiment(config), data=None,
                                        config_path=config, observer=SimpleNamespace())
    assert ran == ["hourly"]


def test_run_plan_refuses_a_recipe_it_cannot_hand_to_the_door(tmp_path):
    """At plan resolution, so ``--resolve`` and a run both answer before any fetch."""
    from gpuwm import runplan
    config = _case(tmp_path, table='[ensemble]\nmembers = 2\nrecipe = "time-lagged"\n')
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    donor = tmp_path / "pmsl.grib2"
    donor.write_text("x")
    experiment_route = _plan(tmp_path, config, route="experiment")
    assert "every member would be a copy of it" in runplan._recipe_plan_refusal(experiment_route, raw, None)
    assert "one prepared trajectory" in runplan._recipe_plan_refusal(
        _plan(tmp_path, config), raw, {"source": "hrrr"})
    for option, value in (("supplement", [f"PMSL={donor}"]), ("data_dir", str(tmp_path)),
                          ("physics_profile", "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"),
                          ("render_section", "35.0,-98.0,36.0,-97.0")):
        plan = _plan(tmp_path, config, **{option: value})
        refusal = runplan._recipe_plan_refusal(plan, raw, None)
        assert f"does not use run_options.{option}" in refusal and "read by nothing" in refusal
        with pytest.raises(runplan.PlanError, match=f"does not use run_options.{option}"):
            runplan.resolve_plan(plan, require_inputs=False)
    # The same plan with no recipe resolves.
    plain = _case(tmp_path / "plain")
    assert runplan._recipe_plan_refusal(
        _plan(tmp_path, plain, data_dir=str(tmp_path)), tomllib.loads(plain.read_text()), None) is None


def test_run_plan_readiness_answers_for_the_recipes_member_windows(tmp_path):
    from gpuwm import runplan
    config = _case(tmp_path)
    document, code = runplan.plan_readiness(
        _plan(tmp_path, config, ensemble={"members": 2, "recipe": "time-lagged"}), no_probe=True)
    assert [(row["cycle"], row["readiness"]["window"]["start_lead"])
            for row in document["recipe"]["member_windows"]] == [("2026-08-20T00", 0), ("2026-08-19T23", 1)]
    assert code == max(row["exit_code"] for row in document["recipe"]["member_windows"]) or code == 2
    plain, _code = runplan.plan_readiness(_plan(tmp_path, config), no_probe=True)
    assert "recipe" not in plain and plain["cycle"] == "2026-08-20T00"


# ---- the calibration boundary ---------------------------------------------------

@pytest.mark.parametrize("table", ['[ensemble]\nmembers=2\nrecipe="time-lagged"\n', MULTI_TOML],
                         ids=["time-lagged", "multi-model"])
def test_recipes_pass_the_calibration_check_on_every_door(table, tmp_path):
    from gpuwm.ensemble.calibration_admission import refuse_public_arguments
    config = tmp_path / "recipe.toml"
    config.write_text(table)
    for command in ("go", "ensemble", "run", "resume"):
        refuse_public_arguments(SimpleNamespace(command=command, config=config))
    refuse_public_arguments(SimpleNamespace(command="sim", experiment_config=config))
    assert request_for_payload(config.read_bytes()).recipe in recipe_door.RECIPES


@pytest.mark.parametrize("table", ['[ensemble]\nmembers=2\nrecipe="time-lagged"\n', MULTI_TOML],
                         ids=["time-lagged", "multi-model"])
def test_a_recipe_does_not_admit_random_perturbations(table):
    with pytest.raises(ValueError) as caught:
        request_for_payload((table + "[ensemble.stochastic]\nsppt=true\n").encode())
    assert str(caught.value) == UNCALIBRATED_SPREAD_REASON


def _guarded(argv):
    from tests.test_ensemble_calibration_admission import _GUARD
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                       PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-c", _GUARD, "gpuwm", *argv], cwd=REPO, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=180)


@pytest.mark.parametrize("command", ["ensemble", "go", "run"])
@pytest.mark.parametrize("recipe", ["time-lagged", "multi-model"])
def test_real_entry_point_admits_a_recipe_past_the_calibration_check(command, recipe, tmp_path):
    """The real command line: a recipe reaches the startup banner, unrefused."""
    config = tmp_path / "case.toml"
    config.write_text("[experiment]\nname='case'\n")
    argv = [command, str(config), "--members", "2"]
    if recipe == "time-lagged":
        argv += ["--recipe", "time-lagged"]
    else:
        listed = tmp_path / "members.json"
        listed.write_text(json.dumps(MULTI))
        argv += ["--trajectories", str(listed)]
    if command == "run":
        argv += ["--outdir", str(tmp_path / "out")]
    result = _guarded(argv)
    assert UNCALIBRATED_SPREAD_REASON not in result.stdout + result.stderr
    assert 'GUARD_EVENTS=["process spawn"]' in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("command", ["ensemble", "go", "run"])
def test_real_entry_point_still_refuses_random_spread_beside_a_recipe(command, tmp_path):
    config = tmp_path / "case.toml"
    config.write_text('[ensemble]\nmembers=2\nrecipe="time-lagged"\n[ensemble.stochastic]\nsppt=true\n')
    argv = [command, str(config)] + (["--outdir", str(tmp_path / "out")] if command == "run" else [])
    result = _guarded(argv)
    assert "GUARD_EVENTS=[]" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 2 and UNCALIBRATED_SPREAD_REASON in result.stderr


def test_planning_command_takes_the_multi_model_list(tmp_path, capsys):
    from gpuwm.ensemble import recipes
    listed = tmp_path / "members.json"
    listed.write_text(json.dumps(MULTI))
    assert recipes.main(["--source", "hrrr", "--cycle", "2026-10-01T18:00:00+00:00", "--hours", "1",
                         "--members", "2", "--recipe", "multi-model", "--trajectories", str(listed)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [member["trajectory"]["source"] for member in plan["members"]] == ["hrrr", "rap"]


def test_native_hrrr_preparation_binds_every_name_it_calls():
    """The merge that split the preparer left a call without its import.

    Breakage it prevents: `_run_configured` raised NameError on
    `supplement_bindings` for every native HRRR preparation, which no CPU
    test reached; the real door run did.
    """
    import ast
    import builtins
    tree = ast.parse((REPO / "tools" / "prepare_hrrr_wrf.py").read_text(encoding="utf-8"))
    module_names = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            module_names |= {item.id for target in targets for item in ast.walk(target)
                             if isinstance(item, ast.Name)}
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_run_configured")
    local = {argument.arg for argument in function.args.args}
    for node in ast.walk(function):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            local |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            local.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
            if isinstance(node, ast.FunctionDef):
                local.add(node.name)
            local |= {argument.arg for argument in (*node.args.args, *node.args.kwonlyargs)}
        elif isinstance(node, ast.ExceptHandler) and node.name:
            local.add(node.name)
        elif isinstance(node, ast.comprehension):
            local |= {item.id for item in ast.walk(node.target) if isinstance(item, ast.Name)}
    called = {node.func.id for node in ast.walk(function)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "supplement_bindings" in called
    assert sorted(called - local - module_names) == []
