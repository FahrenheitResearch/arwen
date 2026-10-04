"""Every member of an N > 1 ensemble runs its own inputs, or the request is refused.

The defect these hold closed: the public ``--members N`` door prepared one
trajectory, handed it to every member, and published spread maps (zero
everywhere) and probability maps (0 or 1) from N copies of one forecast.

A plain member count now takes the automatic choice of the recipe planner
(the operational ensemble the source's adapter row declares) through the
recipe door.  Where no such ensemble is declared, and at every door that
holds one input, the request is refused by name with the remedy.
"""
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpuwm.ensemble import member_inputs, recipe_door
from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.calibration_admission import UNCALIBRATED_SPREAD_REASON
from gpuwm.ensemble.door import request_for_inputs, request_for_payload
from gpuwm.ensemble.packing import CardBudget
from gpuwm.ensemble.production import PreparedEnsembleSession
from gpuwm.ensemble.request import EnsembleRequest
from gpuwm.ensemble_admission import copies_breakage, member_source_remedy

REPO = Path(__file__).resolve().parents[1]
START = datetime(2026, 10, 1, 18)
#: Shipped one-domain configs: one whose source declares an operational
#: ensemble in the adapter table, one whose source declares none.
WITH_ENSEMBLE = REPO / "configs" / "gfs_12km_quickstart.toml"
WITHOUT_ENSEMBLE = REPO / "configs" / "hrrr_native_quick_demo.toml"


class Collector:
    def submit(self, **row):
        pass

    def finish_run(self):
        return {"frames": 0}

    def require_complete(self):
        return {}


def prepared():
    exp = SimpleNamespace(run_seconds=60, start_time=datetime(2024, 1, 1),
                          root=SimpleNamespace(run=SimpleNamespace(dt=3)))
    return SimpleNamespace(experiment=exp, boundary_interval_seconds=3600)


def session(tmp_path, members, **kwargs):
    return PreparedEnsembleSession(
        {"members": members}, output_directory=tmp_path / "run", collector=Collector(),
        cards=(CardBudget(0, 1000),), device_scope=lambda _: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)),
        **kwargs)


def refused_as_copies(message, members):
    assert copies_breakage(members) in message, message
    assert "--recipe time-lagged" in message and "--trajectories FILE" in message, message


# ---- the session ----------------------------------------------------------------

@pytest.mark.parametrize("members", [2, 20])
def test_a_session_with_no_member_source_refuses_to_run_copies(members, tmp_path):
    ran = []
    run = session(tmp_path, members)
    with pytest.raises(ValueError) as refused:
        run.run_prepared(lambda *a, **k: ran.append(1), prepared())
    refused_as_copies(str(refused.value), members)
    with pytest.raises(ValueError) as refused:
        run.run_experiment(lambda *a, **k: ran.append(1), prepared().experiment, object(), tmp_path / "run")
    refused_as_copies(str(refused.value), members)
    assert not ran and not (tmp_path / "run").exists()


def test_one_member_is_unchanged(tmp_path):
    ran = []
    receipt = session(tmp_path, 1).run_prepared(
        lambda inputs, **k: ran.append(inputs) or {"status": "PASS"}, prepared())
    assert receipt["status"] == "PASS" and len(ran) == 1
    assert "identical_members" not in receipt


def test_an_input_provider_admits_n_members(tmp_path):
    shared = prepared()
    def provider(*, shared_inputs, member_id, request):
        return SimpleNamespace(experiment=shared_inputs.experiment, member=member_id,
                               boundary_interval_seconds=3600)
    seen = []
    receipt = session(tmp_path, 3, input_provider=provider).run_prepared(
        lambda inputs, **k: seen.append(inputs.member) or {"status": "PASS"}, shared)
    assert receipt["status"] == "PASS" and sorted(seen) == [0, 1, 2]


def test_a_door_may_bind_its_provider_after_the_session_is_opened(tmp_path):
    """The recipe door does: its session can be the one an outer door opened."""
    run = session(tmp_path, 2)
    run.input_provider = lambda *, shared_inputs, member_id, request: shared_inputs
    assert run.run_prepared(lambda *a, **k: {"status": "PASS"}, prepared())["status"] == "PASS"


def test_stochastic_physics_admits_n_members_only_when_a_process_is_on(tmp_path):
    from gpuwm.ensemble.stochastic_model import StochasticModelProvider
    experiment = prepared().experiment
    session(tmp_path, 4, stochastic_provider=StochasticModelProvider.from_mapping(
        {"sppt": True}))._require_member_inputs(experiment)
    session(tmp_path, 4, stochastic_provider=StochasticModelProvider.from_mapping(
        {"spp": {"pbl": 1}}))._require_member_inputs(experiment)
    off = session(tmp_path, 4, stochastic_provider=StochasticModelProvider.from_mapping(
        {"sppt": False, "skebs": False, "spp": False}))
    with pytest.raises(ValueError) as refused:
        off._require_member_inputs(experiment)
    refused_as_copies(str(refused.value), 4)
    # Native SPP switches in the prepared configuration bind their own provider.
    selected = SimpleNamespace(domains=(SimpleNamespace(run=SimpleNamespace(spp_conv=0, spp_pbl=1, spp_lsm=0)),))
    session(tmp_path, 4)._require_member_inputs(selected)


def test_an_engine_gate_states_why_it_runs_copies_and_the_manifest_records_it(tmp_path):
    purpose = "identity gate: every member must reproduce the ordinary forecast word for word"
    shared, seen = prepared(), []
    receipt = session(tmp_path, 3, identical_members=purpose).run_prepared(
        lambda inputs, **k: seen.append(inputs) or {"status": "PASS"}, shared)
    assert receipt["status"] == "PASS" and receipt["identical_members"] == purpose
    assert seen == [shared] * 3
    assert json.loads((tmp_path / "run" / "ensemble-run.json").read_text())["identical_members"] == purpose
    for value in (True, "", "   "):
        with pytest.raises(ValueError, match="states, in words"):
            session(tmp_path, 3, identical_members=value)


def test_no_front_door_can_declare_identical_members(tmp_path):
    """The doors build their session from the request alone."""
    from gpuwm.ensemble.door import production_run_scope
    assert "identical_members" not in EnsembleRequest.__dataclass_fields__
    with pytest.raises(ValueError, match="unknown ensemble options"):
        request_for_payload(b"[ensemble]\nmembers=2\nidentical_members='x'\n")
    with production_run_scope(EnsembleRequest(2), output_directory=tmp_path) as opened:
        assert opened.identical_members is None
        with pytest.raises(ValueError) as refused:
            opened.run_prepared(lambda *a, **k: pytest.fail("copies ran"), prepared())
    refused_as_copies(str(refused.value), 2)


def test_listed_sources_that_no_door_binds_name_the_breakage_and_the_remedy(tmp_path):
    sources = [{"source": "hrrr", "cycle": "2026-10-01T18:00:00Z"},
               {"source": "hrrr", "cycle": "2026-10-01T17:00:00Z"}]
    with pytest.raises(ValueError) as refused:
        PreparedEnsembleSession({"members": 2, "sources": sources}, output_directory=tmp_path)
    message = str(refused.value)
    assert "require their bound input provider" not in message
    assert "every member would run the one prepared input" in message
    assert "zero spread and probabilities of 0 or 1" in message
    assert "Next: put the same list in a file and run gpuwm ensemble CONFIG --trajectories FILE" in message


# ---- the plan --------------------------------------------------------------------

def experiment(hours=3):
    return SimpleNamespace(start_time=START, run_seconds=3600.0 * hours, domains=(object(),),
                           vertical=SimpleNamespace(p_top=5000.0))


def payload(source, hours=3):
    return {"fetch": {"source": source, "cycle": "2026-10-01T18", "hours": hours}}


def test_a_plain_member_count_plans_the_sources_operational_ensemble():
    from gpuwm.source_adapters import get_source_adapter
    source = "gfs"
    declared = get_source_adapter(source).ensemble_source
    assert declared, "the adapter table no longer declares this source's operational ensemble"
    recipe = recipe_door.plan_recipe(EnsembleRequest(2), payload(source), experiment())
    assert recipe.kind == "input-ensemble"
    assert [member.trajectory.source for member in recipe.members] == [declared, declared]
    assert len({member.trajectory.member for member in recipe.members}) == 2
    assert len({member.trajectory.identity for member in recipe.members}) == 2


def test_a_source_with_no_operational_ensemble_is_refused_with_the_remedy():
    from gpuwm.source_adapters import get_source_adapter
    source = "hrrr"
    adapter = get_source_adapter(source)
    assert not adapter.member_set and not adapter.ensemble_source
    with pytest.raises(recipe_door.RecipeRefusal) as refused:
        recipe_door.plan_recipe(EnsembleRequest(2), payload(source, hours=1), experiment(hours=1))
    message = str(refused.value)
    # The plan itself says what the request would otherwise have been.
    assert message.startswith(copies_breakage(2) + " " + member_inputs.AUTOMATIC_CHOICE)
    assert f"{member_inputs.AUTOMATIC_CHOICE}, and {source} declares none." in message
    assert member_source_remedy(2) in message


def test_an_operational_ensemble_that_cannot_cover_the_window_says_why():
    with pytest.raises(recipe_door.RecipeRefusal) as refused:
        recipe_door.plan_recipe(EnsembleRequest(2), payload("gfs", hours=1), experiment(hours=1))
    message = str(refused.value)
    assert message.startswith(copies_breakage(2))
    assert "that plan cannot be made for this request" in message and "valid-time knots" in message
    assert member_source_remedy(2) in message


def test_a_named_recipe_keeps_the_doors_own_refusal():
    request = EnsembleRequest(2, recipe="time-lagged")
    assert member_inputs.planned_refusal(request, ValueError("the door's sentence")) == "the door's sentence"
    # A plain member count leads with what the request would have been.  The
    # door's sentence follows unchanged: it can open with a file name.
    plain = member_inputs.planned_refusal(EnsembleRequest(2), ValueError("case.namelist.wps is not beside X. Next: Y"))
    assert plain == (copies_breakage(2) + " Each member needs its own source trajectory instead, and "
                     "that plan was refused: case.namelist.wps is not beside X. Next: Y")
    automatic = member_inputs.no_automatic_members("hrrr", 2, "x")
    assert member_inputs.planned_refusal(EnsembleRequest(2), ValueError(automatic)) == (
        copies_breakage(2) + " " + automatic)


def test_only_the_plans_own_refusals_lead_with_the_copies_sentence():
    """A plan that cannot be made says what a plain member count would have been.

    Every refusal of the plan takes that lead and a named recipe takes none.
    The door's later gates (the card, memory, geography, the renderer, the
    disk) raise the same class and are not about where the members come
    from: led with "N copies of one forecast" under a printed member plan,
    a card refusal read as if the plan had been refused.
    """
    no_fetch = {"case_data": {}}
    with pytest.raises(recipe_door.RecipeRefusal) as refused:
        recipe_door.plan_recipe(EnsembleRequest(3), no_fetch, experiment())
    assert str(refused.value) == (
        copies_breakage(3) + " Each member needs its own source trajectory instead, and that "
        "plan was refused: an ensemble recipe fetches and prepares each member's own source "
        "trajectory, and this config has no [fetch] source and cycle. Next: gpuwm domain --help")
    with pytest.raises(recipe_door.RecipeRefusal) as refused:
        recipe_door.plan_recipe(EnsembleRequest(3, recipe="time-lagged"), no_fetch, experiment())
    assert str(refused.value).startswith("an ensemble recipe fetches and prepares")
    # The same plan, asked for another cycle by --readiness, is refused the same way.
    with pytest.raises(recipe_door.RecipeRefusal) as refused:
        recipe_door.plan_recipe(EnsembleRequest(2), payload("hrrr", hours=1), experiment(hours=1),
                                cycle="2026-10-01T12")
    assert str(refused.value).startswith(copies_breakage(2))


# ---- gpuwm go and gpuwm ensemble ---------------------------------------------------

def _case(tmp_path, extra=""):
    config = tmp_path / "case.toml"
    config.write_text("[experiment]\nname='case'\n" + extra)
    return config


def _go_args(config, command, **flags):
    return SimpleNamespace(config=config, command=command,
                           **{"members": None, "recipe": None, "trajectories": None, **flags})


@pytest.mark.parametrize("command", ["go", "ensemble"])
@pytest.mark.parametrize("how", ["flag", "table"])
def test_go_and_ensemble_plan_members_for_a_plain_member_count(command, how, tmp_path, monkeypatch):
    from gpuwm import go_cli
    config = _case(tmp_path, "" if how == "flag" else "[ensemble]\nmembers=2\n")
    reached = []
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble",
                        lambda args, request, observer=None, options=None: reached.append(request) or 0)
    monkeypatch.setattr(go_cli, "_go_launch", lambda *a, **k: pytest.fail("the one-trajectory chain ran"))
    assert go_cli.go_main(_go_args(config, command, members=2 if how == "flag" else None)) == 0
    assert [(request.members, request.recipe) for request in reached] == [(2, None)]


def test_go_runs_one_member_through_its_own_chain(tmp_path, monkeypatch):
    from gpuwm import go_cli
    from gpuwm.ensemble.runtime_context import current_session
    seen = []
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble", lambda *a, **k: pytest.fail("planned members"))
    monkeypatch.setattr(go_cli, "_go_launch",
                        lambda args, observer=None: seen.append(current_session().request.members) or 0)
    assert go_cli.go_main(_go_args(_case(tmp_path), "ensemble", members=1)) == 0
    assert seen == [1]


def test_go_readiness_is_answered_for_the_members_and_runs_nothing(tmp_path, monkeypatch):
    """The run fetches the members' windows, so those are the windows readiness answers for.

    It was answered for the config's own source: ready at one time for a
    run whose members post at another.
    """
    from gpuwm import go_cli
    asked = []
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble", lambda *a, **k: pytest.fail("planned members"))
    monkeypatch.setattr(go_cli, "_go_launch",
                        lambda *a, **k: pytest.fail("answered for the config's own window"))
    monkeypatch.setattr(go_cli, "_recipe_readiness",
                        lambda args, request, config, payload, cycle, posting:
                        asked.append((request.members, request.recipe, cycle)) or 75)
    assert go_cli.go_main(_go_args(_case(tmp_path), "go", members=2, readiness=True)) == 75
    assert asked == [(2, None, None)]


@pytest.mark.parametrize("flag", ["prepared_root", "restart", "data_dir"])
def test_go_refuses_members_of_one_trajectorys_input(flag, tmp_path, monkeypatch):
    """One prepared bundle, one checkpoint or one download is one input: refused by name."""
    from gpuwm import go_cli
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble", lambda *a, **k: pytest.fail("planned members"))
    monkeypatch.setattr(go_cli, "_go_launch", lambda *a, **k: pytest.fail("the one-trajectory chain ran"))
    with pytest.raises(go_cli.GoRefusal) as refused:
        go_cli.go_main(_go_args(_case(tmp_path), "go", members=3, **{flag: tmp_path / "one"}))
    message = str(refused.value)
    named = "--" + flag.replace("_", "-")
    assert message.startswith(f"An ensemble recipe does not use {named}: {named}: ")
    assert message.endswith("Next: omit it.")


def test_go_hands_a_cycle_flag_to_the_route_that_retimes_the_member_plan(tmp_path, monkeypatch):
    """--cycle is honoured for a plain member count as for a named recipe.

    It was refused here as a flag the member plan would drop, which
    stopped being true when the recipe route learned to re-time the config
    the members are planned from.
    """
    from gpuwm import go_cli
    seen = []
    monkeypatch.setattr(go_cli, "_go_launch", lambda *a, **k: pytest.fail("the one-trajectory chain ran"))
    monkeypatch.setattr(go_cli, "_go_recipe",
                        lambda args, request, observer=None: seen.append((request.members, args.cycle)) or 0)
    assert go_cli.go_main(_go_args(_case(tmp_path), "ensemble", members=2, cycle="2026-10-03T00")) == 0
    assert seen == [(2, "2026-10-03T00")]


def test_go_keeps_the_doors_sentence_word_for_word(tmp_path, monkeypatch):
    """What the plan says about a plain count is the plan's to say; go adds nothing.

    go used to lead every refusal of the door with "N copies of one
    forecast", so a card or memory gate read as a refused member plan.
    """
    from gpuwm import go_cli
    sentence = "GPUWM_NO_LOCAL_GPU is set in this environment. Next: unset it"
    def door(args, request, observer=None, options=None):
        raise recipe_door.RecipeRefusal(sentence)
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble", door)
    with pytest.raises(go_cli.GoRefusal) as refused:
        go_cli.go_main(_go_args(_case(tmp_path), "ensemble", members=2))
    assert str(refused.value) == sentence


# ---- gpuwm run, resume and branch ----------------------------------------------------

@pytest.fixture
def quiet_cli(monkeypatch):
    from gpuwm import capabilities, provenance_gate
    monkeypatch.setattr(provenance_gate, "announce", lambda *a, **k: None)
    monkeypatch.setattr(capabilities, "require_for_command", lambda *a, **k: None)


def test_run_plans_members_for_a_plain_member_count(tmp_path, monkeypatch, quiet_cli):
    from gpuwm import cli, supervisor
    reached = []
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble",
                        lambda args, request, observer=None: reached.append(request) or 0)
    monkeypatch.setattr(supervisor, "supervise_from_cli", lambda *a, **k: pytest.fail("one-input worker started"))
    assert cli.main(["run", str(_case(tmp_path)), "--members", "2", "--outdir", str(tmp_path / "out")]) == 0
    assert cli.main(["run", str(_case(tmp_path, "[ensemble]\nmembers=3\n")),
                     "--outdir", str(tmp_path / "out")]) == 0
    assert [(request.members, request.recipe) for request in reached] == [(2, None), (3, None)]


def test_run_leaves_a_missing_config_to_its_own_refusal(tmp_path, monkeypatch, quiet_cli, capsys):
    from gpuwm import cli
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble", lambda *a, **k: pytest.fail("planned members"))
    assert cli.main(["run", str(tmp_path / "absent.toml"), "--members", "2",
                     "--outdir", str(tmp_path / "out")]) == 2
    message = capsys.readouterr().err
    assert "absent.toml" in message and "copies of one forecast" not in message


@pytest.mark.parametrize("flag, value", [
    ("--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000000"), ("--allow-shared-gpu", None),
    ("--restart", "gpuwmrst.npz"), ("--prep-timeout", "600"),
    ("--supervisor-max-restarts", "1"), ("--health-debug", None)])
@pytest.mark.parametrize("how", ["flag", "table"])
def test_run_refuses_the_supervision_flags_a_member_run_does_not_read(
        flag, value, how, tmp_path, monkeypatch, quiet_cli, capsys):
    """The members run in the calling process, ahead of the supervisor.

    Breakage it prevents: a plain member count took the member route with
    --gpu-uuid, --restart and the other supervision flags accepted and read
    by nothing.  A pinned run put its members on the cards the pin
    excludes, and a card that does not exist was not even refused.
    """
    from gpuwm import cli, supervisor
    monkeypatch.setattr(recipe_door, "run_recipe_ensemble", lambda *a, **k: pytest.fail("planned members"))
    monkeypatch.setattr(supervisor, "supervise_from_cli", lambda *a, **k: pytest.fail("one-input worker started"))
    config = _case(tmp_path, "" if how == "flag" else "[ensemble]\nmembers=2\n")
    argv = ["run", str(config), "--outdir", str(tmp_path / "out"), flag]
    assert cli.main(argv + ([] if value is None else [value])
                    + (["--members", "2"] if how == "flag" else [])) == 2
    message = capsys.readouterr().err
    assert f"does not use {flag}: {flag}: " in message
    assert "runs in this process, unsupervised" in message and "Next: omit it." in message
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flag", ["--wrfinput", "--met-em"])
def test_run_refuses_members_of_one_input_directory(flag, tmp_path, monkeypatch, quiet_cli, capsys):
    from gpuwm import cli, metem_forecast, wrfinput_forecast
    for module, name in ((wrfinput_forecast, "run_wrf_forecast"), (metem_forecast, "run_metem_forecast")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("the input directory was read"))
    assert cli.main(["run", flag, str(tmp_path), "--outdir", str(tmp_path / "out"), "--members", "4"]) == 2
    message = capsys.readouterr().err
    refused_as_copies(message, 4)
    assert "--wrfinput and --met-em name one trajectory's files" in message
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["wrfinput", "met_em"])
def test_input_directory_modules_refuse_n_members_before_native_loading(kind, tmp_path, monkeypatch, capsys):
    from gpuwm import metem_door, metem_forecast, provenance_gate, wrfinput_door, wrfinput_forecast
    module = wrfinput_forecast if kind == "wrfinput" else metem_forecast
    resolver, name = ((wrfinput_door, "resolve_wrfinput_run") if kind == "wrfinput"
                      else (metem_door, "resolve_metem_run"))
    monkeypatch.setattr(resolver, name, lambda *a, **k: pytest.fail("native headers were read"))
    monkeypatch.setattr(provenance_gate, "announce", lambda *a, **k: None)
    flag = "--wrfinput" if kind == "wrfinput" else "--met-em"
    assert module.main([flag, str(tmp_path), "--outdir", str(tmp_path / "out"), "--members", "2"]) == 2
    refused_as_copies(capsys.readouterr().err, 2)
    with pytest.raises(ValueError) as refused:
        request_for_inputs(override={"members": 5})
    refused_as_copies(str(refused.value), 5)
    assert request_for_inputs(members=1).members == 1 and request_for_inputs() is None


@pytest.mark.parametrize("how", ["flag", "table"])
def test_resume_and_branch_refuse_members_of_one_checkpoint(how, tmp_path, monkeypatch, quiet_cli, capsys):
    from gpuwm import branch, cli, resume
    config = _case(tmp_path, "" if how == "flag" else "[ensemble]\nmembers=2\n")
    members = ["--members", "2"] if how == "flag" else []
    monkeypatch.setattr(resume, "resolve_resume_checkpoint",
                        lambda *a, **k: pytest.fail("a checkpoint was located"))
    monkeypatch.setattr(branch, "prepare_branch", lambda *a, **k: pytest.fail("the branch folder was written"))
    for argv in (["resume", str(config), "--outdir", str(tmp_path / "run"), *members],
                 ["branch", str(config), "--from-run", str(tmp_path / "run"),
                  "--outdir", str(tmp_path / "what-if"), *members]):
        assert cli.main(argv) == 2
        message = capsys.readouterr().err
        refused_as_copies(message, 2)
        assert f"gpuwm {argv[0]} continues one forecast from one checkpoint" in message
    assert not (tmp_path / "what-if").exists()


def test_resume_and_branch_refuse_a_recipe_instead_of_dropping_it(tmp_path, monkeypatch, quiet_cli, capsys):
    """The recipe door owns this refusal; the member-count check adds nothing to it.

    Neither command registers --recipe or --trajectories, so the config's
    table is the only way a recipe reaches them.
    """
    from gpuwm import branch, cli, resume
    config = _case(tmp_path, '[ensemble]\nmembers=2\nrecipe="time-lagged"\n')
    monkeypatch.setattr(resume, "resolve_resume_checkpoint",
                        lambda *a, **k: pytest.fail("a checkpoint was located"))
    monkeypatch.setattr(branch, "prepare_branch", lambda *a, **k: pytest.fail("the branch folder was written"))
    for argv in (["resume", str(config), "--outdir", str(tmp_path / "run")],
                 ["branch", str(config), "--from-run", str(tmp_path / "run"),
                  "--outdir", str(tmp_path / "what-if")]):
        assert cli.main(argv) == 2
        message = capsys.readouterr().err
        assert (f"gpuwm {argv[0]} continues one prepared trajectory from its checkpoint, and "
                "case.toml selects an ensemble recipe") in message
        assert "Next: gpuwm ensemble CONFIG --recipe time-lagged" in message
        assert "copies of one forecast" not in message
        with pytest.raises(SystemExit) as unknown:
            cli.main(argv + ["--recipe", "time-lagged"])
        assert unknown.value.code == 2
        assert "unrecognized arguments: --recipe time-lagged" in capsys.readouterr().err
    assert not (tmp_path / "what-if").exists()


def test_a_run_config_refuses_the_member_flags_instead_of_dropping_them(tmp_path, quiet_cli, capsys):
    from gpuwm import cli
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[run]\ncase = "none"\n')
    assert cli.main(["run", str(legacy), "--members", "2", "--outdir", str(tmp_path / "out")]) == 2
    message = capsys.readouterr().err
    assert "opens no ensemble session" in message and "refusing to drop them" in message
    assert "Next: gpuwm ensemble CONFIG --members N" in message


def test_listed_sources_are_refused_before_run_starts_a_worker(tmp_path, monkeypatch, quiet_cli, capsys):
    from gpuwm import cli, supervisor
    monkeypatch.setattr(supervisor, "supervise_from_cli", lambda *a, **k: pytest.fail("a worker was started"))
    config = _case(tmp_path, '[ensemble]\nmembers=2\n[[ensemble.sources]]\nsource="hrrr"\ncycle="2026-10-01T18"\n'
                             '[[ensemble.sources]]\nsource="hrrr"\ncycle="2026-10-01T17"\n')
    assert cli.main(["run", str(config), "--outdir", str(tmp_path / "out")]) == 2
    message = capsys.readouterr().err
    assert "this door binds none of them" in message and "--trajectories FILE" in message


# ---- gpuwm run-plan --------------------------------------------------------------------

@pytest.mark.parametrize("route, source, refused", [
    ("experiment", "gfs", True), ("prepared", "hrrr", True), ("prepared", "gfs", False)])
def test_run_plan_refuses_members_of_one_input_before_its_fetch(route, source, refused):
    from gpuwm import runplan
    chain = runplan._chain_key(route, source)
    assert (chain == "prepared:go") is not refused
    plan = SimpleNamespace(route=route, run_options={"ensemble": {"members": 2}})
    raw = {"fetch": {"source": source}}
    if not refused:
        runplan._refuse_one_input_ensemble(plan, raw)
        return
    with pytest.raises(runplan.PlanError) as error:
        runplan._refuse_one_input_ensemble(plan, raw)
    refused_as_copies(str(error.value), 2)
    # The config's own table is read the same way, and one member is unchanged.
    with pytest.raises(runplan.PlanError):
        runplan._refuse_one_input_ensemble(SimpleNamespace(route=route, run_options={}),
                                           {**raw, "ensemble": {"members": 2}})
    runplan._refuse_one_input_ensemble(SimpleNamespace(route=route, run_options={}),
                                       {**raw, "ensemble": {"members": 1}})
    runplan._refuse_one_input_ensemble(SimpleNamespace(route=route, run_options={}), raw)


def test_run_plan_readiness_answers_for_the_members_a_plain_count_fetches(tmp_path):
    """On the go chain a plain member count runs the source's operational ensemble.

    Its readiness was answered for the config's own source, whose files
    post at another time than the members the run fetches.
    """
    from gpuwm import runplan
    import tomllib

    source = tomllib.loads(WITH_ENSEMBLE.read_text(encoding="utf-8"))["fetch"]["source"]
    from gpuwm.source_adapters import get_source_adapter
    declared = get_source_adapter(source).ensemble_source
    assert declared and declared != source

    def plan(name, **run_options):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "schema": runplan.PLAN_SCHEMA, "name": name, "route": "prepared",
            "config": {"path": str(WITH_ENSEMBLE)}, "output_root": str(tmp_path / "run"),
            "run_options": run_options}), encoding="utf-8")
        return runplan.load_plan(path)

    members = plan("members", ensemble={"members": 2})
    raw = tomllib.loads(WITH_ENSEMBLE.read_text(encoding="utf-8"))
    assert runplan._go_plans_members(members, raw)
    document, _code = runplan.plan_readiness(members, no_probe=True)
    assert document["source"] == declared and document["recipe"]["kind"] == "input-ensemble"
    assert [row["source"] for row in document["recipe"]["member_windows"]] == [declared, declared]
    # One member, and no member count, are the config's own window as before.
    for name, options in (("one", {"ensemble": {"members": 1}}), ("none", {})):
        single = plan(name, **options)
        assert not runplan._go_plans_members(single, raw)
        own, _code = runplan.plan_readiness(single, no_probe=True)
        assert "recipe" not in own and own["source"] == source
    # A chain that prepares one trajectory refuses the run, so nothing here is its members'.
    one_input = SimpleNamespace(route="prepared", run_options={"ensemble": {"members": 2}})
    assert not runplan._go_plans_members(one_input, {"fetch": {"source": "hrrr"}})
    assert not (tmp_path / "run").exists()


# ---- an [ensemble] table no door would honour -------------------------------------------

@pytest.mark.parametrize("table", ["[ensemble]\nmembers=2\n", "[ensemble]\nmembers=1\n"])
def test_sim_refuses_the_table_it_cannot_honour(table, tmp_path, monkeypatch, capsys):
    from gpuwm import stage_cli
    monkeypatch.setattr(stage_cli, "resolve_bundle", lambda *a, **k: pytest.fail("the prepared bundle was opened"))
    config = _case(tmp_path, table)
    assert stage_cli.sim_main(SimpleNamespace(experiment_config=config)) == 2
    message = capsys.readouterr().err
    assert "carries an [ensemble] table, and gpuwm sim runs one forecast" in message
    # A prepared bundle binds its config, so one forecast means preparing again.
    assert "Next: gpuwm ensemble CONFIG" in message
    assert "take the [ensemble] table out of the config and prepare it again" in message


@pytest.mark.parametrize("tree", [False, True])
def test_the_prepared_runners_refuse_the_table_unless_a_door_opened_a_session(tree, tmp_path, monkeypatch, capsys):
    from gpuwm import prepared_domain_tree_forecast, prepared_single_domain_forecast, provenance_gate
    from gpuwm.ensemble.calibration_admission import refuse_explicit_config_argv
    from gpuwm.ensemble.runtime_context import ensemble_scope
    module = prepared_domain_tree_forecast if tree else prepared_single_domain_forecast
    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *a, **k: pytest.fail("startup probe entered"))
    config = _case(tmp_path, "[ensemble]\nmembers=2\n")
    assert module.main(["--experiment-config", str(config)]) == 2
    assert "the prepared forecast runner runs one forecast and opens no ensemble session" in capsys.readouterr().err
    # Hosted by a door that opened the session, the runner inherits it.
    with ensemble_scope(SimpleNamespace(request=EnsembleRequest(2, recipe="time-lagged"))):
        refuse_explicit_config_argv(["--experiment-config", str(config)])
    refuse_explicit_config_argv(["--experiment-config", str(_case(tmp_path))])


def test_a_run_config_refuses_the_table_by_name(tmp_path):
    from gpuwm.config import load_config, _KNOWN_TABLES
    assert "ensemble" not in _KNOWN_TABLES
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[run]\ncase = "none"\n[ensemble]\nmembers = 2\n')
    with pytest.raises(ValueError) as refused:
        load_config(legacy)
    message = str(refused.value)
    assert "carries an [ensemble] table" in message and "opens no ensemble session" in message
    assert "unknown table" not in message and "Next: gpuwm ensemble CONFIG" in message
    assert "remove the [ensemble] table to run this one forecast" in message


# ---- the 2.8.4 overlay table and provider names -----------------------------------------

SHIPPED_OVERLAY = REPO / "configs" / "ensemble" / "may1999_tiny_2member.toml"


def _names_the_overlay_command(message):
    assert "tools.ensemble_forecast" in message
    assert "Next: python -m tools.ensemble_forecast run --ensemble-config FILE" in message
    assert "members, keep_member_files, thresholds" in message


def test_the_shipped_overlay_is_named_as_the_overlay_with_its_command(tmp_path):
    from gpuwm.config import load_config
    from gpuwm.experiment import build_experiment
    import tomllib
    for refuse in (lambda: request_for_payload(SHIPPED_OVERLAY.read_bytes()),
                   lambda: load_config(SHIPPED_OVERLAY),
                   lambda: build_experiment(tomllib.loads(SHIPPED_OVERLAY.read_text(encoding="utf-8")),
                                            source="overlay")):
        with pytest.raises(ValueError) as refused:
            refuse()
        message = str(refused.value)
        assert "unknown ensemble options" not in message
        assert "base_config, n_members" in message and "the overlay file" in message
        _names_the_overlay_command(message)


@pytest.mark.parametrize("name", ["gpuwm.da.perturb", "experimental-stub"])
def test_a_284_provider_name_is_refused_by_name_with_its_command(name):
    with pytest.raises(ValueError) as refused:
        request_for_payload(('[ensemble]\nmembers=2\nperturbation="%s"\n' % name).encode())
    message = str(refused.value)
    assert message != "ensemble perturbation must be an object"
    assert f'perturbation = "{name}" is a provider name' in message
    assert "would run every member unperturbed under that name" in message
    _names_the_overlay_command(message)
    assert UNCALIBRATED_SPREAD_REASON not in message


def test_the_overlay_command_refuses_a_base_config_with_its_own_table(tmp_path):
    """2.8.4 refused it (an unknown table then); read and dropped, it ran the overlay's count."""
    from gpuwm.ensemble.config import load_ensemble_config
    shipped_base = SHIPPED_OVERLAY.with_name("may1999_tiny_ensemble_base.toml")
    (tmp_path / "base.toml").write_text(
        shipped_base.read_text(encoding="utf-8") + "\n[ensemble]\nmembers = 5\n", encoding="utf-8")
    overlay = tmp_path / "overlay.toml"
    text = SHIPPED_OVERLAY.read_text(encoding="utf-8")
    assert 'base_config = "may1999_tiny_ensemble_base.toml"' in text
    overlay.write_text(text.replace('base_config = "may1999_tiny_ensemble_base.toml"',
                                    'base_config = "base.toml"'), encoding="utf-8")
    with pytest.raises(ValueError) as refused:
        load_ensemble_config(overlay)
    message = str(refused.value)
    assert "carries its own [ensemble] table" in message and "would be dropped" in message
    assert "Next: remove [ensemble] from the base config" in message
    # The shipped overlay and its base load as they did.
    shipped = load_ensemble_config(SHIPPED_OVERLAY)
    assert shipped.n_members == 2 and shipped.perturbation == "gpuwm.da.perturb"


def test_none_and_a_mistyped_key_keep_their_own_answers():
    assert request_for_payload(b'[ensemble]\nmembers=1\nperturbation="none"\n').perturbation is None
    with pytest.raises(ValueError, match="unknown ensemble options") as refused:
        request_for_payload(b"[ensemble]\nmembers=2\nmember=3\n")
    assert "the [ensemble] table takes members, keep_member_files, thresholds" in str(refused.value)


# ---- the calibration refusal at the loaders ----------------------------------------------

def test_the_experiment_loader_refuses_random_physics_on_every_route(tmp_path):
    """The overlay command's base config loads through the same builder."""
    from gpuwm.case_data import load_experiment_case
    from gpuwm.experiment import load_experiment
    base = REPO / "configs" / "ensemble" / "may1999_tiny_ensemble_base.toml"
    text = base.read_text(encoding="utf-8")
    assert "[shared]\n" in text
    config = tmp_path / "base.toml"
    for switch in ("spp_pbl = 1", "spp_conv = 1", "spp_lsm = 1"):
        config.write_text(text.replace("[shared]\n", "[shared]\n" + switch + "\n", 1), encoding="utf-8")
        for load in (load_experiment, load_experiment_case):
            with pytest.raises(ValueError) as refused:
                load(config)
            assert str(refused.value) == UNCALIBRATED_SPREAD_REASON, switch


def test_the_run_config_loader_refuses_random_physics(tmp_path):
    from gpuwm.config import load_config
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[run]\ncase = "none"\n[physics]\nspp_pbl = 1\n')
    with pytest.raises(ValueError) as refused:
        load_config(legacy)
    assert str(refused.value) == UNCALIBRATED_SPREAD_REASON


def test_branch_is_checked_before_it_writes_its_run_folder(tmp_path):
    from gpuwm.ensemble.calibration_admission import refuse_public_arguments
    config = _case(tmp_path, "[ensemble]\nmembers=1\n[ensemble.stochastic]\nsppt=true\n")
    with pytest.raises(ValueError) as refused:
        refuse_public_arguments(SimpleNamespace(command="branch", config=config))
    assert str(refused.value) == UNCALIBRATED_SPREAD_REASON


# ---- the calibration campaign --------------------------------------------------------------

@pytest.mark.parametrize("amplitude", [0, 0.5, 1.0])
def test_the_calibration_campaign_binds_its_stochastic_arms(amplitude, tmp_path):
    """Its own arms were refused, so no measurement could ever retire the refusal."""
    import importlib.util
    path = REPO / "tools" / "ensemble_campaign_run.py"
    spec = importlib.util.spec_from_file_location("ensemble_campaign_run_member_inputs", path)
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    roster = SimpleNamespace(members=(SimpleNamespace(member_id=0), SimpleNamespace(member_id=1)))
    request = {"thresholds": {}, "base_seed": 5, "stochastic_amplitude": amplitude}
    run = campaign.campaign_session(request, roster, tmp_path / "forecast")
    assert run.member_roster is roster and run.request.members == 2
    assert run.request.stochastic is None
    if amplitude == 0:
        assert run.stochastic_provider is None
    else:
        reference = campaign.stochastic_controls(amplitude)
        assert run.stochastic_provider.sppt.stddev == reference["sppt"]["stddev"]
        assert run.stochastic_provider.spp == {"conv": 0, "pbl": 1, "lsm": 1}
    # A public request still may not carry the same controls.
    with pytest.raises(ValueError) as refused:
        EnsembleRequest(2, stochastic=campaign.stochastic_controls(0.5))
    assert str(refused.value) == UNCALIBRATED_SPREAD_REASON


# ---- the real command line -------------------------------------------------------------------

_NO_NETWORK = r'''
import json, runpy, socket, sys
events = []
def blocked(*args, **kwargs):
    events.append("network connect")
    raise OSError("GUARD: network connect")
socket.socket.connect = blocked
socket.create_connection = blocked
sys.argv = ["gpuwm"] + sys.argv[1:]
try:
    runpy.run_module("gpuwm", run_name="__main__", alter_sys=True)
except SystemExit as exit:
    code = exit.code
else:
    code = 0
print("GUARD_EVENTS=" + json.dumps(events))
sys.exit(code if isinstance(code, int) else 3)
'''


def _gpuwm(*argv, cwd):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_NO_LOCAL_GPU="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
                       PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-c", _NO_NETWORK, *argv], cwd=cwd, env=environment,
                          capture_output=True, text=True, encoding="utf-8", timeout=300)


@pytest.mark.parametrize("command", ["ensemble", "go"])
def test_real_entry_point_refuses_a_plain_member_count_with_no_ensemble_to_draw_from(command, tmp_path):
    """The real command line, on a shipped config: exit 2, the remedy, nothing fetched."""
    result = _gpuwm(command, str(WITHOUT_ENSEMBLE), "--members", "2", "--dry-run",
                    "--outdir", str(tmp_path / "out"), cwd=tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "GUARD_EVENTS=[]" in result.stdout, result.stdout + result.stderr
    refused_as_copies(result.stderr, 2)
    assert "declares none" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", ["ensemble", "go"])
def test_real_entry_point_plans_the_operational_ensemble_for_a_plain_member_count(command, tmp_path):
    """The same line on a source whose adapter row declares an ensemble: a member plan."""
    result = _gpuwm(command, str(WITH_ENSEMBLE), "--members", "2", "--dry-run",
                    "--outdir", str(tmp_path / "out"), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARD_EVENTS=[]" in result.stdout, result.stdout + result.stderr
    assert "ensemble: input-ensemble recipe, 2 members" in result.stdout
    plan = [line for line in result.stdout.splitlines() if line.startswith("ensemble: member ")]
    assert len(plan) == 2 and len(set(plan)) == 2, result.stdout
    assert "dry run: nothing was fetched, prepared or run" in result.stdout
    assert not (tmp_path / "out").exists()


def _operational_ensemble():
    from gpuwm.source_adapters import get_source_adapter
    import tomllib
    source = tomllib.loads(WITH_ENSEMBLE.read_text(encoding="utf-8"))["fetch"]["source"]
    declared = get_source_adapter(source).ensemble_source
    assert declared and declared != source
    return source, declared


def test_real_entry_point_answers_readiness_for_the_members_that_will_run(tmp_path):
    """--readiness on a plain member count: the members' windows, not the config's own source."""
    _own, declared = _operational_ensemble()
    result = _gpuwm("go", str(WITH_ENSEMBLE), "--members", "2", "--readiness", "--no-probe",
                    "--outdir", str(tmp_path / "out"), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    document = json.loads(result.stdout.split("GUARD_EVENTS=")[0])
    assert "GUARD_EVENTS=[]" in result.stdout, result.stdout + result.stderr
    assert document["source"] == declared
    recipe = document["recipe"]
    assert recipe["kind"] == "input-ensemble" and recipe["members"] == 2
    assert [window["source"] for window in recipe["member_windows"]] == [declared, declared]
    assert len({window["source_member"] for window in recipe["member_windows"]}) == 2
    assert "readiness ready for 2 recipe members" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", ["ensemble", "go"])
def test_real_entry_point_plans_the_members_at_the_cycle_the_flag_names(command, tmp_path):
    """--cycle re-times the config and the members are planned for that cycle."""
    _own, declared = _operational_ensemble()
    result = _gpuwm(command, str(WITH_ENSEMBLE), "--members", "2", "--cycle", "2026-08-19T12",
                    "--dry-run", "--outdir", str(tmp_path / "out"), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARD_EVENTS=[]" in result.stdout, result.stdout + result.stderr
    assert "re-timed to start 2026-08-19 12:00:00 UTC" in result.stdout
    plan = [line for line in result.stdout.splitlines() if line.startswith("ensemble: member ")]
    assert len(plan) == 2 and len(set(plan)) == 2, result.stdout
    assert all(f"{declared} 2026-08-19T12Z member " in line for line in plan), result.stdout
    assert "dry run: nothing was fetched, prepared or run" in result.stdout


@pytest.mark.parametrize("command", ["ensemble", "go", "run"])
def test_a_card_gate_is_reported_in_its_own_words_under_the_member_plan(
        command, tmp_path, monkeypatch, quiet_cli, capsys):
    """A refused card is not a refused member plan.

    The whole door runs on a shipped config: the plan is made and printed,
    then the card gate refuses.  That sentence used to be led with "2
    members from one input are 2 copies of one forecast ... and that plan
    was refused", right under the plan the door had just printed.
    """
    from gpuwm import capabilities, cli, go_cli

    def no_card():
        raise go_cli.GoRefusal("GPU readiness is missing: no card")
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "_require_forecast_device", no_card)
    monkeypatch.setattr(recipe_door, "prepare_member",
                        lambda *a, **k: pytest.fail("a member was fetched"))
    out = tmp_path / "out"
    assert cli.main([command, str(WITH_ENSEMBLE), "--members", "2", "--outdir", str(out)]) == 2
    captured = capsys.readouterr()
    assert "ensemble: input-ensemble recipe, 2 members" in captured.out
    assert "GPU readiness is missing: no card" in captured.err
    assert "copies of one forecast" not in captured.err
    assert "that plan was refused" not in captured.err
    assert not out.exists()
