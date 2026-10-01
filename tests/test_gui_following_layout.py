"""A storm-following Customise runs the event's own layout, on this computer or on a Machines node.

The defects this file holds shut:

- The page carried the cyclone setup's options itself and a start ran
  whatever it was sent. The layout is now read from the storm wiki by its
  event and card size, as the event page's button runs it, and a box or
  grid of the page's own is refused rather than drawn and never run.
- A storm-following start could not run on a Machines node: the
  configuration and the Vtable and WPS namelist it names were written on
  this computer only, and the node's plan named files it never had.
- A storm-following plan named no pictures, and a configuration file's run
  draws none unless its plan says so, so the forecast finished with nothing
  to show. The event page's own start of the same layout drew none either.
- A layout whose nest has a fixed size had its nest grown to the card.
- Two tabs starting one name both answered 200, and the second plan
  replaced the first under a run already started.
- A queued Machines start could write into any folder it was pointed at;
  only the folder the queue hands over as starting is taken.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from gpuwm.first_products import DEFAULT_RENDER_PRODUCTS
from gpuwm.gui import runs
from gpuwm.gui.files import SERVER_DIR
from gpuwm.gui.wiki import recipe_row
from test_gui_machines import FakeMachine, make_api, post
from test_gui_server import DRAFT, gui, request  # noqa: F401 - gui is a fixture

EVENT = "tc-2004086s29318"
CONFIG = '[experiment]\nname = "following"\n'


def _era5_offered(api, monkeypatch):
    monkeypatch.setattr(api, "_offered", lambda *a, **k: {"sources": [{"id": "era5", "route": "prepared"}]})
    monkeypatch.setattr(api, "availability_of", lambda *a, **k: {"state": "yes"})


def _compiler(runner, monkeypatch):
    """The cyclone setup as a stand-in: writes its configuration, two companions and its receipt beside --out."""

    calls = []
    original = runner.query

    def query(argv, **kwargs):
        if any("cyclone-setup" == part for part in argv):
            calls.append(list(argv))
            if "--out" in argv:
                out = Path(argv[argv.index("--out") + 1])
                out.write_text(CONFIG, encoding="utf-8")
                out.with_suffix(".Vtable").write_text("vtable\n", encoding="utf-8")
                out.with_suffix(".namelist.wps").write_text("&share\n/\n", encoding="utf-8")
                out.with_name(out.stem + ".cyclone.json").write_text(json.dumps({"out": str(out)}),
                                                                     encoding="utf-8")
                return {"created": True, "fitting": {"review_required": False}}
            return {"domains": [], "fitting": {"review_required": False}}
        if "--resolve" in argv and "run-plan" in argv:
            return {**original(argv, **kwargs), "disk": {"total_bytes": 2**30}}
        return original(argv, **kwargs)

    monkeypatch.setattr(runner, "query", query)
    return calls


def _payload(api, **extra):
    row = recipe_row(api.wiki.store.data()["recipes"][EVENT], 8)
    recipe = dict(row["recipe"])
    recipe.pop("cyclone_setup", None)
    return {**recipe, "following": True, "event": EVENT, "recipe_card_gb": 8, "name": "following", **extra}


@pytest.fixture()
def following(gui, monkeypatch):  # noqa: F811 - the fixture above
    server, runner = gui
    _era5_offered(server.api, monkeypatch)
    monkeypatch.setattr("gpuwm.gui.api.disk_free_gib", lambda path: 4096.0)
    return server, runner, _compiler(runner, monkeypatch)


def test_a_following_customise_runs_the_events_own_setup_not_the_pages(following):
    server, runner, calls = following
    # Options the page sends of its own are not what runs: the layout is the storm wiki's.
    body = {**_payload(server.api), "cyclone_setup": {"advisory_position": "0.00,0.00", "hours": 1},
            "dry_run": True}
    response, answer = request(server, "POST", "/api/create/start", body=body)
    assert response.status == 200, answer
    assert answer["plan"]["route"] == "experiment"
    prepare = answer["prepare"]
    for option in ("--advisory-position=-28.90,-44.30", "--card=8gb", "--nest-budget-gib=8",
                   "--history-interval=3600", "--nest-history-interval=3600", "--isftcflx=1"):
        assert option in prepare
    assert not (server.root / "following").exists() and not calls


def test_a_gfs_following_plan_names_the_route_its_configuration_runs_on(following, monkeypatch):
    """A152: a storm-following plan on GFS names the prepared route.

    Every storm-following plan named the config-driven route, and `gpuwm
    run-plan` refuses a GFS cyclone configuration there as belonging to the
    prepared route.  The plan and the page's posting question now name the
    route the configuration runs on, and Start's plan asks nothing the
    route does not take.
    """

    server, runner, calls = following
    monkeypatch.setattr(server.api, "_offered",
                        lambda *a, **k: {"sources": [{"id": "gfs", "route": "prepared"}]})
    body = {**_payload(server.api, source="gfs", cycle="2026-09-30T12", hours=6), "dry_run": True}
    response, answer = request(server, "POST", "/api/create/start", body=body)
    assert response.status == 200, answer
    assert answer["plan"]["route"] == "prepared"
    assert "--source=gfs" in answer["prepare"]
    # The event page's own following layout on an ERA5 event keeps the config-driven route, which its
    # [case_data] configuration runs on (the test below).


def test_a_following_start_publishes_its_configuration_and_companions_and_draws_the_standard_set(following):
    server, runner, calls = following
    response, answer = request(server, "POST", "/api/create/start",
                               body=_payload(server.api, hours=24, start_hour=3))
    assert response.status == 200, answer
    rundir = server.root / "following"
    plan = json.loads((rundir / "plan.json").read_text(encoding="utf-8"))
    assert plan["route"] == "experiment" and plan["config"] == {"path": str(rundir / "cyclone.toml")}
    assert plan["run_options"]["render_products"] == DEFAULT_RENDER_PRODUCTS
    assert (rundir / "cyclone.toml").read_text(encoding="utf-8") == CONFIG
    assert (rundir / "cyclone.Vtable").read_text(encoding="utf-8") == "vtable\n"
    assert (rundir / "cyclone.namelist.wps").read_text(encoding="utf-8") == "&share\n/\n"
    # The setup's receipt names its scratch folder and is not an input of the run.
    assert not (rundir / "cyclone.cyclone.json").exists()
    assert "--hours=24" in calls[0] and "--start-hour=3" in calls[0]
    assert not list((server.root / SERVER_DIR / "drafts").glob("following-*"))
    assert len(runner.launched) == 1
    # Pictures the page was told to leave out stay out.
    response, answer = request(server, "POST", "/api/create/start",
                               body=_payload(server.api, name="following-quiet", products="none"))
    assert response.status == 200, answer
    quiet = json.loads((server.root / "following-quiet" / "plan.json").read_text(encoding="utf-8"))
    assert quiet["run_options"]["render_products"] == "none"


@pytest.mark.parametrize("change", [{"lat": 0.0}, {"width_km": 600}, {"dx_km": 3}])
def test_a_following_layout_keeps_the_events_box_and_grid(following, change):
    server, runner, calls = following
    response, answer = request(server, "POST", "/api/create/start",
                               body={**_payload(server.api), **change, "dry_run": True})
    assert response.status == 422, answer
    assert "box and grid" in answer["message"]
    assert not runner.launched and not calls


def test_a_following_layout_is_set_up_again_for_another_card(following):
    server, runner, calls = following
    response, answer = request(server, "POST", "/api/create/start",
                               body={**_payload(server.api, card="16gb"), "dry_run": True})
    assert response.status == 200, answer
    assert "--card=16gb" in answer["prepare"] and "--nest-budget-gib=16" in answer["prepare"]


def test_a_fixed_size_following_layout_keeps_its_size_on_another_card(following, monkeypatch):
    server, runner, calls = following
    store = server.api.wiki.store.data()
    record = json.loads(json.dumps(store["recipes"][EVENT]))
    for row in record["cards"]:
        (row.get("args") or {}).pop("nest_budget_gib", None)
    monkeypatch.setitem(store["recipes"], EVENT, record)
    response, answer = request(server, "POST", "/api/create/start",
                               body={**_payload(server.api, card="16gb"), "dry_run": True})
    assert response.status == 200, answer
    assert "--card=16gb" in answer["prepare"]
    assert not any(part.startswith("--nest-budget-gib") for part in answer["prepare"])


def test_a_following_draft_names_its_event(following):
    server, runner, calls = following
    body = {**_payload(server.api), "dry_run": True}
    body.pop("event")
    response, answer = request(server, "POST", "/api/create/start", body=body)
    assert response.status == 400 and "event" in answer["message"]
    response, answer = request(server, "POST", "/api/create/start",
                               body={**_payload(server.api), "recipe_card_gb": 12, "dry_run": True})
    assert response.status == 422 and "no storm-following layout" in answer["message"]


def test_the_event_pages_following_layout_draws_the_standard_set(following):
    server, runner, calls = following
    response, answer = request(server, "POST", "/api/wiki/simulate",
                               body={"event": EVENT, "card_gb": 8, "name": "catarina", "dry_run": True})
    assert response.status == 200, answer
    assert answer["plan"]["route"] == "experiment"
    assert answer["plan"]["run_options"] == {"render_products": DEFAULT_RENDER_PRODUCTS}


def test_two_tabs_cannot_start_one_name_twice(gui, monkeypatch):  # noqa: F811
    server, runner = gui
    gate = threading.Barrier(2)
    writes = threading.Lock()
    write_plan = server.api._write_plan
    observed = []
    launch = runner.launch

    def together(draft):
        gate.wait(timeout=5)

    def serialized_write(draft, folder):
        # Both requests have passed the name check; only the publication is serialised, so the assertion is
        # about who owns the folder, not about two writers of one temporary file.
        with writes:
            return write_plan(draft, folder)

    def capture(folder, argv, **kwargs):
        observed.append(json.loads((folder / "plan.json").read_text(encoding="utf-8")))
        return launch(folder, argv, **kwargs)

    monkeypatch.setattr(server.api, "night_refusal", together)
    monkeypatch.setattr(server.api, "_write_plan", serialized_write)
    monkeypatch.setattr(runner, "launch", capture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request, server, "POST", "/api/create/start", body={**DRAFT, "hours": hours})
                   for hours in (1, 2)]
        responses = [future.result(timeout=10) for future in futures]
    assert sorted(response.status for response, _ in responses) == [200, 409]
    assert len(runner.launched) == len(observed) == 1
    assert json.loads((server.root / DRAFT["name"] / "plan.json").read_text(encoding="utf-8")) == observed[0]


def _machine_api(tmp_path, monkeypatch):
    from gpuwm.gui.wiki import ensure_seed

    ensure_seed(tmp_path / "runs")  # as `gpuwm gui` does when it starts
    machine = FakeMachine({"name": "box", "host": "me@box", "workspace": "/w"})
    api, _registry = make_api(tmp_path, machine)
    _era5_offered(api, monkeypatch)
    calls = _compiler(api.runner, monkeypatch)
    return api, machine, calls


def test_a_following_start_on_a_machine_sends_its_configuration_and_companions(tmp_path, monkeypatch):
    api, machine, calls = _machine_api(tmp_path, monkeypatch)
    status, body = post(api, "/api/create/start", {**_payload(api, name="f1"), "machine": "box",
                                                   "render_on": "none", "dry_run": True})
    assert status == 200, body
    assert body["plan"]["config"] == {"path": "/w/runs/f1/cyclone.toml"}
    assert "--advisory-position=-28.90,-44.30" in body["prepare"] and not calls
    status, body = post(api, "/api/create/start", {**_payload(api, name="f1"), "machine": "box",
                                                   "render_on": "none"})
    assert status == 200, body
    verb, _args, payload = machine.calls[-1]
    assert verb == "launch"
    assert payload["files"]["plan.json"]["config"] == {"path": "/w/runs/f1/cyclone.toml"}
    assert payload["text_files"] == {"cyclone.toml": CONFIG, "cyclone.Vtable": "vtable\n",
                                     "cyclone.namelist.wps": "&share\n/\n"}
    # The mirror here keeps the same files, so the run's folder says what ran.
    assert (tmp_path / "runs" / "f1" / "cyclone.toml").read_text(encoding="utf-8") == CONFIG


def test_a_queued_machine_start_takes_only_the_folder_the_queue_hands_over(tmp_path, monkeypatch):
    from gpuwm.gui.api import ApiError
    from gpuwm.gui.machines import MachineError
    from gpuwm.gui.queue import STARTING

    api, machine, calls = _machine_api(tmp_path, monkeypatch)
    rundir = tmp_path / "runs" / "f2"
    rundir.mkdir(parents=True)
    (rundir / runs.QUEUED).write_text("{}", encoding="utf-8")
    payload = {**_payload(api, name="f2"), "machine": "box", "render_on": "none"}
    # Not marked starting by the queue: the start does not take the folder.
    with pytest.raises(ApiError) as refused:
        api.create_start_remote("box", dict(payload), False, queued=rundir)
    assert refused.value.status == 409 and not machine.calls
    assert sorted(p.name for p in rundir.iterdir()) == [runs.QUEUED]
    # Handed over: a start that fails leaves only what the queue wrote, companions included, and tries again.
    (rundir / runs.QUEUED).rename(rundir / STARTING)
    real = machine.call

    def dropped(verb, *args, **kwargs):
        if verb == "launch":
            raise MachineError("box dropped the connection.", "Check that box is on.")
        return real(verb, *args, **kwargs)

    machine.call = dropped
    with pytest.raises(ApiError):
        api.create_start_remote("box", dict(payload), False, queued=rundir)
    assert sorted(p.name for p in rundir.iterdir()) == [STARTING]
    machine.call = real
    reply = api.create_start_remote("box", dict(payload), False, queued=rundir)
    assert reply.status == 200
    assert (rundir / "cyclone.namelist.wps").is_file() and machine.calls[-1][2]["text_files"]


def _static_bridge_missing() -> str | None:
    from gpuwm.static import rust_bridge

    return rust_bridge.unavailable_reason()


#: The real cyclone setup publishes its layout through gpuwm.companion_domains,
#: which refuses with "Domain editing requires the native static-fields bridge"
#: where that library is not built, as tests/test_companion_setups.py states.
@pytest.mark.skipif(_static_bridge_missing() is not None,
                    reason="the native static-fields bridge is not built, and the real "
                           "cyclone setup publishes its layout through it")
def test_the_real_cyclone_setup_writes_a_portable_layout_that_resolves(tmp_path, monkeypatch):
    """The page's own compile and publication with the real engine: CPU only, nothing downloaded or launched."""

    from gpuwm.gui.api import CreateMixin
    from gpuwm.gui.jobs import Runner, engine_argv
    from gpuwm.gui.machines import Machine
    from gpuwm.gui.remote_runs import plan_for
    import tomllib

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    root = Path(__file__).resolve().parents[1]
    seed = json.loads((root / "gpuwm/gui/seed/recipes" / f"{EVENT}.json").read_text(encoding="utf-8"))
    record = seed["recipes"][0]
    row = recipe_row(record, 8)

    class PlanningRunner(Runner):
        def query(self, argv, **kwargs):
            assert "cyclone-setup" in argv or ("run-plan" in argv and "--resolve" in argv)
            return super().query(argv, **kwargs)

        def launch(self, *args, **kwargs):
            raise AssertionError("a planning check never launches a forecast")

    class Subject(CreateMixin):
        pass

    subject = Subject()
    subject.root = tmp_path
    subject.runner = PlanningRunner()
    subject.wiki = SimpleNamespace(store=SimpleNamespace(data=lambda: {"recipes": {record["event"]: record}}))
    recipe = dict(row["recipe"])
    recipe.pop("cyclone_setup", None)
    draft = {**recipe, "name": "following-compiler", "hours": 42, "start_hour": 0, "products": None,
             "profile": None, "nz": None, "ladder": None, "chain": None, "buffer_km": None, "following": True}
    subject._bind_following({"event": record["event"], "recipe_card_gb": 8}, draft)
    subject._compile_following(draft)
    folder = tmp_path / draft["name"]
    folder.mkdir()
    subject._prepare_following(draft, folder)
    plan_path = subject._write_plan(draft, folder)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["route"] == "experiment"
    config = Path(plan["config"]["path"])
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    # The card row's own setup: the 240-square nest and hourly history the event page's button runs.
    assert [(d["nx"], d["ny"]) for d in raw["domain"]] == [(200, 160), (240, 240)]
    assert [d["history_interval_s"] for d in raw["domain"]] == [3600, 3600]
    assert raw["domain"][1]["follow"]["field"] == "pressure"
    assert raw["shared"]["nz"] == 49 and raw["shared"]["isftcflx"] == 1
    assert plan["run_options"]["render_products"] == DEFAULT_RENDER_PRODUCTS
    # --resolve accepts a plan whose forecast inputs are absent, so the companions are checked on their own.
    for role in ("vtable", "wps_namelist"):
        declared = Path(raw["case_data"][role])
        assert not declared.is_absolute()
        companion = config.parent / declared
        assert companion.is_file() and companion.stat().st_size > 0
    assert set(draft["_following_files"]) >= {config.name, *(Path(raw["case_data"][r]).name
                                                             for r in ("vtable", "wps_namelist"))}
    resolved = subject.runner.query(engine_argv("run-plan", str(plan_path), "--resolve"), cwd=plan_path.parent)
    domains = resolved["configuration"]["experiment"]["domains"]
    assert [(d["run"]["nx"], d["run"]["ny"], d["run"]["nz"]) for d in domains] == [(200, 160, 49), (240, 240, 49)]
    assert resolved["inputs_present"] is False  # no weather data was fetched
    remote = plan_for(Machine({"name": "target", "kind": "ssh", "host": "fixture.invalid",
                               "workspace": "~/forecast-work"}), plan, draft["name"])
    assert remote["config"]["path"] == "~/forecast-work/runs/following-compiler/cyclone.toml"
    assert str(tmp_path) not in json.dumps(remote)
