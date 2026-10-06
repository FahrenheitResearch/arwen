"""A failed or refused forecast says, on the page, why and what to do.

The breakages this file names, from a user walk on a computer with no
geography tree (the state the Linux install left):

- the event page's button and New forecast's Start each wrote a run
  folder and launched a run that failed at once;
- the failed run's page then read "declared input(s) ... geog_root (a
  folder named in the page server's log)" above "fix the plan document
  and re-run": the log held no such folder, the plan was not what was
  wrong, and the one hint (set the tree up once) was dropped;
- the forecast's article page showed "Failed" and no reason at all; once
  its facts gave the reason and what to do, a note under the title gave
  the same two sentences again.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from gpuwm import runplan
from gpuwm.gui import files, runs
from gpuwm.gui.api import Api
from gpuwm.gui.server import build_server, serve_in_thread

from test_gui_server import DRAFT, FakeRunner, NoGeography, dead_pid, make_run, request
from test_gui_wiki import Era5Runner, _event_with

FETCH = "gpuwm fetch-geog --datasets wrf"
JS = Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static" / "js"


def stage_geography(root: Path) -> None:
    """A tree the WRF static builder's check accepts: each dataset it opens, each with its WPS index."""

    from gpuwm.geog_assets import GEOG_CONSUMER_WRF, datasets_required_by

    for name in datasets_required_by(GEOG_CONSUMER_WRF):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "index").write_text("type = continuous\n", encoding="utf-8")


def _serve(tmp_path, runner):
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    return server


def _made(server) -> list[str]:
    return [p.name for p in server.api.root.iterdir() if p.is_dir() and p.name != "wiki" and not p.name.startswith(".")]


@pytest.fixture()
def no_geography(tmp_path, monkeypatch):
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    (tmp_path / "case-data").mkdir()
    return tmp_path / "case-data" / "WPS_GEOG"


# ------------------------------------------------------------------ the start buttons

def test_new_forecast_start_without_geography_is_refused_and_writes_nothing(tmp_path, no_geography):
    runner = NoGeography()
    server = _serve(tmp_path, runner)
    try:
        for body in (DRAFT, {**DRAFT, "dry_run": True}):
            response, refused = request(server, "POST", "/api/create/start", body=body)
            assert response.status == 409, refused
            assert refused["message"].startswith("This computer has no geography data yet.")
            assert FETCH in refused["fix"] and "GB" in refused["fix"]
            # The page never shows a machine path, the tree's included.
            assert str(tmp_path) not in json.dumps(refused)
        assert runner.launched == [] and _made(server) == []
        # Half a tree is named as that, with how much is missing.
        stage_geography(no_geography)
        shutil.rmtree(next(p for p in no_geography.iterdir()))
        response, refused = request(server, "POST", "/api/create/start", body=DRAFT)
        assert response.status == 409 and "incomplete" in refused["message"] and FETCH in refused["fix"]
        assert _made(server) == []
        # Set up, the same start goes.
        stage_geography(no_geography)
        response, started = request(server, "POST", "/api/create/start", body=DRAFT)
        assert response.status == 200, started
        assert runner.launched and _made(server) == [DRAFT["name"]]
    finally:
        server.shutdown()
        server.server_close()


def test_the_event_button_without_geography_is_refused_and_writes_nothing(tmp_path, no_geography, monkeypatch):
    from gpuwm.gui import api as api_module

    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)

    class Runner(Era5Runner):
        missing_geography = NoGeography.missing_geography

    runner = Runner()
    server = _serve(tmp_path, runner)
    try:
        event = _event_with("era5")
        for dry in (False, True):
            response, refused = request(server, "POST", "/api/wiki/simulate",
                                        body={"event": event["id"], "card_gb": 16, "dry_run": dry})
            assert response.status == 409, refused
            assert FETCH in refused["fix"] and "geography" in refused["message"]
            assert str(tmp_path) not in json.dumps(refused)
        assert runner.launched == [] and _made(server) == []
        stage_geography(no_geography)
        response, started = request(server, "POST", "/api/wiki/simulate", body={"event": event["id"], "card_gb": 16})
        assert response.status == 200, started
        assert runner.launched and _made(server) == [started["run"]]
    finally:
        server.shutdown()
        server.server_close()


def test_a_ready_run_without_geography_stays_ready(tmp_path, no_geography):
    runner = NoGeography()
    server = _serve(tmp_path, runner)
    try:
        run = make_run(server.root, "later", plan=True)
        response, refused = request(server, "POST", "/api/runs/later/start", body={})
        assert response.status == 409 and FETCH in refused["fix"]
        assert runs.status(run)["state"] == "ready" and runner.launched == []
        # A plan that names its own tree is checked there.
        elsewhere = tmp_path / "elsewhere"
        stage_geography(elsewhere)
        plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
        plan["run_options"] = {"geog_root": str(elsewhere)}
        (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        response, started = request(server, "POST", "/api/runs/later/start", body={})
        assert response.status == 200, started
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------ the failed run's words

def _failed_event(error: BaseException) -> dict:
    return {"event": "failed", "stage": "prepare", "error_class": type(error).__name__, "message": str(error),
            "remedy": runplan._remedy(error), "interrupted": False, "exit_code": None}


def test_a_missing_geography_run_reads_what_is_missing_and_the_command(tmp_path):
    tree = "/home/" + "someone/.local/share/gpuwm/WPS_GEOG"
    error = runplan.missing_inputs_refusal(
        [{"role": "geog_root", "path": tree, "kind": "directory", "present": False}], before_download=True)
    event = _failed_event(error)
    assert event["remedy"] == f"Set up the geography data once on this computer with {FETCH}."
    # The terminal reader gets the same next step without --explain, and the reason stays in the explain half.
    from gpuwm.explain import split

    action, why = split(event["message"])
    assert f"remedy: {event['remedy']}" in action and "before the download" in why
    run = make_run(tmp_path, "failed", pid=dead_pid(), end=event)
    end = runs.status(run)["end"]
    assert end["remedy"] == event["remedy"]
    assert end["message"] == ("declared input(s) this run needs are not on disk, and the download does not "
                              f"supply them: geog_root {files.PATH_WORDS}")


def test_a_refusal_that_names_its_remedy_is_carried_and_a_plan_defect_keeps_the_plan_line():
    carried = runplan.PlanError("the thing is missing", remedy="Get the thing.")
    assert runplan._remedy(carried) == "Get the thing."
    assert runplan._remedy(runplan.PlanError("run plan 'name' must be a string")) == \
        "fix the plan document and re-run; nothing was started"


def test_a_refusal_that_states_its_remedy_in_its_message_carries_it(tmp_path, monkeypatch):
    # A missing gfs_grib2_bridge is refused with "  remedy: gpuwm setup" and a "  # or ..." line under it; its
    # failed event said remedy: null, so the page showed the refusal's first line and nothing to do.
    from gpuwm import bridges, go_cli

    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    with pytest.raises(go_cli.GoRefusal) as refused:
        go_cli.resolve_bridge()
    event = _failed_event(refused.value)
    assert event["remedy"].startswith("gpuwm setup\n# or `gpuwm doctor --explain`")
    run = make_run(tmp_path, "no-bridge", pid=dead_pid(), end=event)
    end = runs.status(run)["end"]
    assert end["message"] == "no built gfs_grib2_bridge, which every stage of this chain decodes through."
    assert end["remedy"] == ("gpuwm setup or `gpuwm doctor --explain` for the build route on a platform with no "
                             "published bundle")
    # The next label ends the remedy, and a message that states none still falls to its class.
    assert runplan.stated_remedy("it failed.\n  remedy: build it\n    in the crate folder\n  cargo said:\n    e") == \
        "build it\nin the crate folder"
    assert runplan._remedy(runplan.PlanError("run plan 'name' must be a string")) == \
        "fix the plan document and re-run; nothing was started"


def test_gpuwm_go_prints_a_remedy_its_refusal_states_once(monkeypatch):
    # gpuwm go prints a failed run's text and then "Next:" and the event's remedy. A refusal that states its
    # remedy gives its event that same line, so the terminal read "remedy: gpuwm setup ... Next: gpuwm setup".
    from gpuwm import bridges, go_cli

    monkeypatch.setattr(bridges, "find_bridge", lambda name: None)
    with pytest.raises(go_cli.GoRefusal) as refused:
        go_cli.resolve_bridge()
    for explain in (False, True):
        line = go_cli.failed_line(_failed_event(refused.value), explain=explain)
        assert line.count("remedy: gpuwm setup") == 1 and line.count("gpuwm doctor --explain") == 1
        assert "Next:" not in line
    missing = runplan.missing_inputs_refusal(
        [{"role": "geog_root", "path": "/x/WPS_GEOG", "kind": "directory", "present": False}], before_download=True)
    line = go_cli.failed_line(_failed_event(missing), explain=False)
    assert line.count(FETCH) == 1 and "Next:" not in line
    # A remedy the text does not state is still given after it.
    line = go_cli.failed_line(_failed_event(runplan.PlanError("run plan 'name' must be a string")), explain=False)
    assert line == "run plan 'name' must be a string Next: fix the plan document and re-run; nothing was started"
    carried = runplan.PlanError("the thing is missing\n  remedy: get the thing", remedy="Get the other thing.")
    assert go_cli.failed_line(_failed_event(carried), explain=False).endswith("Next: Get the other thing.")


def test_the_other_missing_inputs_name_their_own_step():
    error = runplan.missing_inputs_refusal([
        {"role": "vtable", "path": "/x/Vtable", "kind": "file", "present": False},
        {"role": "forcing", "path": "/x/f.grib", "kind": "file", "present": False}])
    assert "fetch-geog" not in error.remedy
    assert "vtable" in error.remedy and "[fetch] block" in error.remedy


def test_a_path_the_page_leaves_out_is_in_the_page_servers_log(tmp_path, capsys):
    folder = "/home/" + "someone/runs/q/input-" + tmp_path.name
    record = {"event": "failed", "stage": "prepare", "message": f"The file {folder} is gone.\nmore\n[[explain]]\nwhy",
              "remedy": None}
    run = tmp_path / "gone"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    shown = runs.status(run)["end"]["message"]
    assert shown == f"The file {files.PATH_WORDS} is gone." and "/home/" not in shown
    logged = capsys.readouterr().err
    assert folder in logged and "more" in logged and "[[explain]]" not in logged
    # Read again (the page asks every few seconds), it is not logged again.
    runs._CACHE.clear()
    runs.status(run)
    assert folder not in capsys.readouterr().err


def test_a_remedy_keeps_its_whole_first_paragraph():
    wrapped = "Set up the geography data once\non this computer with gpuwm fetch-geog\n  --datasets wrf.\n\nWhy: a mechanism."
    assert files.plain_message(wrapped, paragraph=True) == \
        "Set up the geography data once on this computer with gpuwm fetch-geog --datasets wrf."
    assert files.plain_message(wrapped) == "Set up the geography data once"


def test_a_remedy_written_as_terminal_comments_keeps_its_words(tmp_path):
    # The GPU runtime's remedy is a commented block around one labelled command. The page cut at its first "#",
    # which was all of it, so a run the engine refused for want of CuPy showed its reason and no remedy.
    from gpuwm.capabilities import GPU_RUNTIME_REMEDY

    record = {"event": "failed", "stage": "preflight", "error_class": "CapabilityMissing",
              "message": "gpuwm run-plan: this command needs cupy (cupy-cuda12x / cupy-cuda13x), which this "
                         "install does not have.\n  What needs it: every path that integrates the model on a card.",
              "remedy": GPU_RUNTIME_REMEDY, "interrupted": False, "exit_code": None}
    run = tmp_path / "no-cupy"
    run.mkdir()
    (run / "events.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    remedy = runs.status(run)["end"]["remedy"]
    # Neither extra leads, as in the terminal: the page names both and what tells them apart.
    assert "pip install 'gpuwm[gpu-cu12]'" in remedy and "pip install 'gpuwm[gpu-cu13]'" in remedy
    assert "13-only" in remedy and "gpuwm doctor" in remedy
    assert "#" not in remedy and "remedy:" not in remedy
    # A command's own trailing comment is still left out.
    assert files.plain_message("gpuwm fetch-tables  # about 40 MB", paragraph=True) == "gpuwm fetch-tables"


def test_the_linux_steps_give_the_geography_size_the_tool_gives():
    # The Linux note said fetch-geog "downloads several gigabytes". The page's refusal quotes the tool's own figures,
    # and the disk the tree takes once unpacked is the one a user has to find room for.
    from gpuwm.geog_assets import GEOG_CONSUMER_WRF, datasets_required_by, size_phrase

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    linux = readme[readme.index("### Linux"):]
    note = " ".join(linux[linux.index("`fetch-geog` sets up"):].split("\n\n", 1)[0].split())
    compressed, unpacked = re.findall(r"~([\d.]+) GB", size_phrase(datasets_required_by(GEOG_CONSUMER_WRF)))
    assert f"about {compressed} GB" in note and f"about {unpacked} GB of disk once unpacked" in note
    assert "several gigabytes" not in note


# ------------------------------------------------------------------ the article page (run in Node)

DOM = r"""
class FakeNode {
  constructor() { this.childNodes = []; this.parentNode = null; }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) { this.childNodes = []; this.append(String(v)); }
  append(...kids) {
    for (const k of kids) {
      const n = k instanceof FakeNode ? k : new FakeText(String(k));
      n.parentNode = this;
      this.childNodes.push(n);
    }
  }
  prepend(...kids) { const old = this.childNodes; this.childNodes = []; this.append(...kids); this.childNodes.push(...old); }
  replaceChildren(...kids) { this.childNodes = []; this.append(...kids); }
  remove() { if (this.parentNode) this.parentNode.childNodes = this.parentNode.childNodes.filter((c) => c !== this); }
}
class FakeText extends FakeNode { constructor(t) { super(); this.data = t; } get textContent() { return this.data; } }
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase(); this.attributes = {}; this.dataset = {}; this.style = {}; this.className = "";
    const names = () => this.className.split(/\s+/).filter(Boolean);
    this.classList = {
      add: (...c) => { this.className = [...new Set([...names(), ...c])].join(" "); },
      remove: (...c) => { this.className = names().filter((x) => !c.includes(x)).join(" "); },
      toggle: (c, on) => { (on ?? !names().includes(c)) ? this.classList.add(c) : this.classList.remove(c); },
      contains: (c) => names().includes(c),
    };
  }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return k in this.attributes ? this.attributes[k] : null; }
  addEventListener() {}
  removeEventListener() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
  // A 2D context that draws nothing: the article's small map runs its real code over it.
  getContext() { return new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => ({ width: 0 })),
    set: (t, k, v) => { t[k] = v; return true; } }); }
  getBoundingClientRect() { return { left: 0, top: 0, width: 360, height: 220 }; }
  focus() {}
  find(cls) {
    if (this.className.split(/\s+/).includes(cls)) return this;
    for (const c of this.childNodes) { if (c instanceof FakeElement) { const f = c.find(cls); if (f) return f; } }
    return null;
  }
}
globalThis.Node = FakeNode;
globalThis.document = { createElement: (t) => new FakeElement(t), createTextNode: (t) => new FakeText(t), title: "",
  addEventListener() {}, removeEventListener() {}, querySelector: () => null, documentElement: new FakeElement("html") };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
globalThis.window = { addEventListener() {}, removeEventListener() {}, devicePixelRatio: 1 };
globalThis.location = { hash: "" };
const input = JSON.parse(await (await import("node:fs/promises")).readFile("input.json", "utf8"));
globalThis.fetch = async (url) => ({ ok: true, status: 200,
  json: async () => (String(url).startsWith("/api/library/run/") ? input.article : {}) });
const { SCREENS } = await import("./router.js");
await import("./wikipages.js");
const body = new FakeElement("div");
const page = { words: input.words, setCrumbs() {}, setContext() {}, setTitle() {}, setData() {}, right: new FakeElement("div") };
await SCREENS.get("run").render(body, [input.run], page);
const note = body.find("endnote");
console.log(JSON.stringify({ text: body.textContent, note: note ? note.textContent : null }));
"""


@pytest.fixture(scope="module")
def node():
    found = shutil.which("node")
    if found is None:
        pytest.skip("Node is not installed")
    return found


def _article(tmp_path, node, end) -> dict:
    """The run article the server answers for a run that ended with ``end``, drawn by wikipages.js's runPage."""

    server = _serve(tmp_path, FakeRunner())
    try:
        run = make_run(server.root, "ended", pid=dead_pid(), end=end, plan=True)
        # A box, so the article's small map is drawn as well, as it is for every run the page starts.
        ring = [[-100.9, 32.6], [-94.3, 32.6], [-94.3, 38.0], [-100.9, 38.0], [-100.9, 32.6]]
        (run / "region.geojson").write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}), encoding="utf-8")
        response, article = request(server, "GET", "/api/wiki/run/ended")
        assert response.status == 200, article
    finally:
        server.shutdown()
        server.server_close()
    folder = tmp_path / "page"
    shutil.copytree(JS, folder)
    (folder / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    (folder / "input.json").write_text(json.dumps({"run": "ended", "article": article, "words": Api.copy()}),
                                       encoding="utf-8")
    (folder / "t.mjs").write_text(DOM, encoding="utf-8")
    done = subprocess.run([node, "t.mjs"], cwd=folder, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_failed_forecasts_article_says_why_and_what_to_do_once(tmp_path, node):
    error = runplan.missing_inputs_refusal(
        [{"role": "geog_root", "path": "/home/" + "someone/WPS_GEOG", "kind": "directory", "present": False}],
        before_download=True)
    drawn = _article(tmp_path, node, _failed_event(error))
    text = drawn["text"]
    reason = "declared input(s) this run needs are not on disk"
    remedy = f"Set up the geography data once on this computer with {FETCH}."
    # The facts give the engine's reason and then what to do, each once: a note under the title drew both again.
    assert drawn["note"] is None
    assert text.count(reason) == 1 and text.count(remedy) == 1
    assert text.index("Why it stopped") < text.index(reason) < text.index("What to do") < text.index(remedy)
    assert "null" not in text and "/home/" not in text


def test_a_stopped_forecasts_article_says_why_it_stopped(tmp_path, node):
    # A stopped forecast has no reason among its facts, so the note under its title is the one place it is shown.
    stopped = runplan.ChainInterrupted("forecast")
    drawn = _article(tmp_path, node, {"event": "failed", "stage": "forecast", "error_class": "ChainInterrupted",
                                      "message": str(stopped), "remedy": None, "interrupted": True,
                                      "exit_code": runplan.INTERRUPT_EXIT_CODE})
    assert drawn["note"] == "interrupted during forecast"
    assert drawn["text"].count("interrupted during forecast") == 1 and "Why it stopped" not in drawn["text"]
    assert "null" not in drawn["text"]


def test_a_finished_forecasts_article_has_no_end_note(tmp_path, node):
    drawn = _article(tmp_path, node, {"event": "completed", "stage": "finalize", "dry_run": False})
    assert drawn["note"] is None and "null" not in drawn["text"]
