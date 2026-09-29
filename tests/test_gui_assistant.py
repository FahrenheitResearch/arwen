"""The assistant against a scripted chat-completions model and a fake engine runner: no GPU, no network.

The scripted model answers the way llama-server does (tool calls, output
held to a JSON schema, log probabilities of the chosen tokens).  The rest
is the real gui server: the assistant's routes, the planner's typed
decisions, the page's own Check the fit and Start.
"""

from __future__ import annotations

import http.server
import json
import math
import threading

import pytest

from gpuwm.gui.assistant import catalog
from gpuwm.gui.assistant.decide import DecisionError, Question, answer_record, label_distribution
from gpuwm.gui.server import build_server, serve_in_thread

from test_gui_server import FakeRunner as BaseRunner, request


@pytest.fixture(autouse=True)
def _no_publication_probe(monkeypatch):
    # New forecast puts a recent start to the fetch's object probe before it accepts it; no server is asked in a
    # test, or a start near the publication frontier waits on the network and the page's request times out.
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    # A page asks through its own short probe; no server is asked in a test either.
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)

COVERED = {"west": -130.0, "east": -60.0, "south": 20.0, "north": 55.0}


class FakeRunner(BaseRunner):
    def query(self, argv, *, cwd=None, timeout=0, log=None):
        document = super().query(argv, cwd=cwd, timeout=timeout, log=log)
        if "--sources" in argv:
            regional = {"source_id": "regional", "display_name": "Regional", "max_forecast_hour": 48,
                        "forcing_interval_seconds": 3600, "coverage": {**COVERED, "describe": "a regional grid"},
                        "run_plan": {"intent_supported": True, "intent_routes": ["prepared"],
                                     "requires_source_root": False}}
            document = {"sources": [*document["sources"], regional]}
        if "--physics-profiles" in argv:
            document = json.loads(json.dumps(document))
            document["sources"].append({"source_id": "regional", "default_profile_id": "p1",
                                        "profiles": [{"profile_id": "p1", "admissible": True, "is_default": True},
                                                     {"profile_id": "p2", "admissible": True}]})
            document["profiles"].append({"profile_id": "p2", "summary": "p2: other words"})
        return document


def _logprobs(label: str, labels: str) -> list[dict]:
    """Tokens of '{"choice": "B", ...}' with the label's alternatives, as llama-server returns them."""

    head = ['{"', 'choice', '":', ' "']
    top = [{"token": label, "logprob": math.log(0.7)}]
    rest = [item for item in labels if item != label]
    for item in rest:
        top.append({"token": item, "logprob": math.log(0.3 / len(rest))})
    tokens = [{"token": text, "logprob": -0.01, "top_logprobs": []} for text in head]
    tokens.append({"token": label, "logprob": math.log(0.7), "top_logprobs": top})
    tokens += [{"token": text, "logprob": -0.01, "top_logprobs": []} for text in ('",', ' "why', '":', ' "ok"}')]
    return tokens


class ScriptedModel(http.server.ThreadingHTTPServer):
    """Answers /v1/chat/completions from the request's shape, and counts what it was asked."""

    def __init__(self, place=(38.5, -98.0)):
        super().__init__(("127.0.0.1", 0), _ModelHandler)
        self.place = place
        self.bodies: list[dict] = []
        self.pick = {}
        self.pick_id = {}
        self.finest_km = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class _ModelHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        message, logprobs = self._answer(body)
        data = json.dumps({"choices": [{"message": message, "finish_reason": "stop",
                                        "logprobs": {"content": logprobs} if logprobs else None}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 10}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _answer(self, body):
        last = body["messages"][-1]["content"]
        schema = ((body.get("response_format") or {}).get("json_schema") or {}).get("schema")
        if schema and "choice" in schema["properties"]:
            labels = "".join(schema["properties"]["choice"]["enum"])
            question = last.split("Question: ", 1)[1].split("\n", 1)[0]
            label = self.server.pick.get(question, "A")
            wanted = self.server.pick_id.get(question)
            if wanted:
                label = next(line.split(")", 1)[0] for line in last.splitlines() if f") {wanted}:" in line)
            label = label if label in labels else "A"
            text = json.dumps({"choice": label, "why": "fits the request"})
            return {"role": "assistant", "content": text}, _logprobs(label, labels)
        if schema:
            lat, lon = self.server.place if "Denver" not in last else (39.7, -105.0)
            if "The place:" not in last and "somewhere" in last:
                lat = lon = None
            return {"role": "assistant", "content": json.dumps(
                {"what": "storms", "place": "a place", "lat": lat, "lon": lon,
                 "finest_km": self.server.finest_km})}, None
        if body.get("tools"):
            asked = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
            if "what happened" in asked.lower():
                # search the wiki, read the event it finds, answer from its summary
                if body["messages"][-1]["role"] == "user":
                    call = {"id": "w1", "type": "function",
                            "function": {"name": "search_wiki", "arguments": json.dumps({"words": "tornado Oklahoma 2013"})}}
                    return {"role": "assistant", "content": "", "tool_calls": [call]}, None
                data = json.loads(last.split("DATA\n", 1)[1].rsplit("\nEND DATA", 1)[0])
                if isinstance(data, list):
                    call = {"id": "w2", "type": "function",
                            "function": {"name": "read_event", "arguments": json.dumps({"event": data[0]["id"]})}}
                    return {"role": "assistant", "content": "", "tool_calls": [call]}, None
                return {"role": "assistant", "content": data["summary"]}, None
            if "which forecasts" in last.lower():
                if body["messages"][-1]["role"] == "tool":
                    return {"role": "assistant", "content": "You have none."}, None
                call = {"id": "c1", "type": "function", "function": {"name": "list_runs", "arguments": "{}"}}
            else:
                call = {"id": "c1", "type": "function",
                        "function": {"name": "plan_forecast", "arguments": json.dumps({"request": last})}}
            if body["messages"][-1]["role"] == "tool":
                return {"role": "assistant", "content": "You have none."}, None
            return {"role": "assistant", "content": "", "tool_calls": [call]}, None
        return {"role": "assistant", "content": "ok"}, None


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_ASSISTANT_HOME", str(tmp_path / "home"))
    model = ScriptedModel()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    runner = FakeRunner()
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    response, _ = request(server, "POST", "/api/assistant/enable", body={"on": True})
    assert response.status == 200
    response, _ = request(server, "POST", "/api/assistant/settings",
                          body={"backend": "endpoint", "endpoint_url": model.url, "endpoint_model": "scripted"})
    assert response.status == 200
    yield server, runner, model
    server.shutdown()
    server.server_close()
    model.shutdown()


def decided_profile(turn):
    return next(d["choice"] for d in turn["decisions"] if d["question"] == "physics")


def test_an_intention_becomes_a_checked_plan_of_listed_ids_and_nothing_starts(rig):
    server, runner, model = rig
    model.pick = {"Which data source should start and drive this forecast?": "B",
                  "Which day does the person want to see?": "C", "Which part of that day do they want to see?": "B"}
    response, turn = request(server, "POST", "/api/assistant/say",
                             body={"text": "I want to see storms over the plains tomorrow afternoon at 1 km"})
    assert response.status == 200, turn
    fill = next(a for a in turn["actions"] if a["type"] == "fill_create")
    fields = fill["fields"]
    assert fields["source"] in {"gfs", "regional"}
    # p1 is the source's default: it goes in as the form's "source's default", p2 by name
    assert decided_profile(turn) in {"p1", "p2"}
    assert fields["profile"] == (None if decided_profile(turn) == "p1" else "p2")
    assert fields["ladder"] in {"12", "12-3", "12-3-1", "12-3-1-0.5", "auto"}
    assert fields["lat"] == 38.5 and fields["width_km"] in (300.0, 600.0, 1000.0, 1600.0, 2500.0)
    decided = {d["question"]: d for d in turn["decisions"]}
    assert {"day", "part_of_day", "box_size", "ladder", "source", "physics", "machine"} <= set(decided)
    for answer in turn["decisions"]:
        assert answer["choice"] in answer["options"]
        if len(answer["options"]) > 1:
            assert answer["probability"] == pytest.approx(0.7, abs=1e-3)
    assert any(c["name"] == "check_fit" and c["ok"] for c in turn["tool_calls"])
    start = next(a for a in turn["actions"] if a["type"] == "confirm")
    assert start["request"]["path"] == "/api/create/start"
    assert runner.launched == []
    # the person's click is the page's own request
    body = {**start["request"]["body"], "name": "clicked"}
    response, reply = request(server, "POST", start["request"]["path"], body=body)
    assert response.status == 200, reply
    assert len(runner.launched) == 1
    plan = json.loads((server.root / "clicked" / "plan.json").read_text())
    assert plan["config"]["intent"]["ladder"] == fields["ladder"]


def test_a_missing_place_is_one_question_then_a_plan(rig):
    server, runner, model = rig
    response, turn = request(server, "POST", "/api/assistant/say",
                             body={"text": "I want to see a thunderstorm somewhere"})
    assert turn["question"] and turn["actions"][0]["type"] == "question"
    response, turn = request(server, "POST", "/api/assistant/say",
                             body={"text": "near Denver", "conversation": turn["conversation"]})
    fill = next(a for a in turn["actions"] if a["type"] == "fill_create")
    assert fill["fields"]["lat"] == 39.7
    response, saved = request(server, "GET", f"/api/assistant/conversations/{turn['conversation']}")
    assert len(saved["turns"]) == 2 and saved["turns"][1]["decisions"]


def test_questions_use_the_pages_read_tools(rig):
    server, runner, model = rig
    response, turn = request(server, "POST", "/api/assistant/say", body={"text": "Which forecasts do I have?"})
    assert turn["reply"] == "You have none."
    assert [c["name"] for c in turn["tool_calls"]] == ["list_runs"]
    tool_message = model.bodies[-1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["content"].startswith("DATA\n")


def test_assistant_keeps_a_section_line_through_fit_edit_and_start_review(rig):
    from gpuwm.gui.api import ApiError
    from gpuwm.gui.assistant.agent import Agent, Turn

    server, runner, _ = rig
    agent = Agent(server.api, server.api.assistant.chat())
    line = "35,-100,36,-99"
    form = agent._form({"name": "section-run", "source": "gfs", "lat": 35, "lon": -100,
                        "card": "8gb", "products": "xsec:wa", "render_section": line})
    assert form["render_section"] == line
    turn = Turn(user="Check this section")
    assert agent._execute(turn, "check_fit", {}, form)["words"]
    changed = "35,-100,36,-98"
    assert agent._execute(turn, "change_plan", {"field": "render_section", "value": changed}, form)["ok"]
    assert agent._execute(turn, "propose_start", {}, form)["ok"]
    start = next(action for action in turn.actions if action["type"] == "confirm")
    assert "problem" not in start
    assert start["request"]["body"]["products"] == "xsec:wa"
    assert start["request"]["body"]["render_section"] == changed
    assert runner.launched == []
    with pytest.raises(ApiError, match="line"):
        agent._execute(Turn(user="Check this section"), "propose_start", {},
                       {key: value for key, value in form.items() if key != "render_section"})


def test_downloading_needs_the_confirm_click(rig, monkeypatch):
    server, _, _ = rig
    request(server, "POST", "/api/assistant/settings", body={"backend": "bundled", "model": "qwen3.5-4b-q4km"})
    response, plan = request(server, "POST", "/api/assistant/install", body={"dry_run": True})
    assert response.status == 200 and plan["items"]
    assert all(item["licence"]["id"] and item["sha256"] for item in plan["items"])
    response, refused = request(server, "POST", "/api/assistant/install", body={})
    assert response.status == 409 and refused["install"]["items"]


def test_an_answer_outside_the_options_is_refused_and_probabilities_come_from_the_label_token():
    question = Question("q", "pick", {"a": "one", "b": "two"})
    with pytest.raises(DecisionError):
        answer_record(question, "c", None, reason="", method="x", backend="y", seconds=0)
    spread = label_distribution(_logprobs("B", "AB"), "AB")
    assert spread == pytest.approx({"A": 0.3, "B": 0.7})
    # a label written with its quote in one token
    quoted = [{"token": '{"choice": "', "top_logprobs": []},
              {"token": 'A"', "top_logprobs": [{"token": 'A"', "logprob": math.log(0.6)},
                                                {"token": 'B"', "logprob": math.log(0.4)}]}]
    assert label_distribution(quoted, "AB") == pytest.approx({"A": 0.6, "B": 0.4})


def test_the_model_is_picked_by_card_memory():
    # A card reports a little under its label (a 16 GB card is 15.9 GiB), so each tier is reached from just below.
    assert catalog.pick_model(6)["id"] == "qwen3.5-4b-q4km"
    assert catalog.pick_model(7.9)["id"] == "qwen3.5-9b-q4km"
    assert catalog.pick_model(10)["id"] == "qwen3.5-9b-q4km"
    assert catalog.pick_model(11.9)["id"] == "qwen3.5-9b-q6k"
    assert catalog.pick_model(15.9)["id"] == "qwen3.8-27b-iq4xs"
    assert catalog.pick_model(23.6)["id"] == "qwen3.8-27b-q5km"
    assert catalog.pick_model(31.4)["id"] == "qwen3.8-27b-q6k"
    assert catalog.pick_model(80)["id"] == "qwen3.8-27b-q6k"
    assert catalog.pick_model(None)["id"] == "qwen3.5-9b-q6k"


def test_every_bundled_row_is_pinned_licensed_and_fits_its_tier():
    rows = catalog.models()
    assert sorted({row["min_card_gib"] for row in rows}) == [6, 8, 12, 16, 24, 32]
    for row in rows:
        assert row["licence"]["id"] in {"Apache-2.0", "MIT"}, row["id"]
        assert len(row["sha256"]) == 64 and int(row["sha256"], 16) >= 0 and row["bytes"] > 0
        assert row["url"] == f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{row['file']}"
        assert len(row["revision"]) == 40 and row["quant"] and row["context"] >= 8192
        # the file must leave room on its own tier for the context and the server's buffers
        assert row["bytes"] / 2**30 < row["min_card_gib"] - 1.4, row["id"]
    for row in catalog.brought():
        # listed with its licence and why it is not bundled; never downloadable by the assistant
        assert row["licence"]["id"] and row["licence"]["url"].startswith("https://") and row["why_not_bundled"]
        assert "sha256" not in row and "url" not in row and catalog.model(row["id"]) is None
        # Settings links the row to where its file is and lists it under a card it can fit
        assert row["where"].startswith("https://") and row.get("file_gib", 0) < row["min_card_gib"], row["id"]


def test_a_forecast_start_takes_the_model_off_the_card(rig):
    server, runner, _ = rig
    stopped = []
    request(server, "POST", "/api/assistant/settings", body={"backend": "bundled"})
    server.api.assistant.local.stop = lambda: stopped.append(True) or True
    response, reply = request(server, "POST", "/api/create/start",
                              body={"name": "r1", "source": "gfs", "lat": 40, "lon": -97, "card": "16gb"})
    assert response.status == 200 and stopped == [True]
    assert reply["assistant"]["unloaded"] is True


def test_the_system_one_wire_answers_and_a_system_one_decider_can_read_it(rig):
    server, _, model = rig
    questions = {"size": {"type": "choice", "instructions": "How big?", "criteria": {"small": "300 km", "large": "2500 km"}}}
    response, body = request(server, "POST", "/api/assistant/v1/systemone", body={"state": {"x": 1}, "questions": questions})
    assert response.status == 200
    answer = body["answers"]["size"]
    assert answer["choice"] == "small" and answer["probabilities"]["small"] == pytest.approx(0.7, abs=1e-3)
    # the same wire, spoken by the decider the assistant uses for Jev and Kev
    from gpuwm.gui.assistant.decide import SystemOneDecider

    decider = SystemOneDecider(f"http://127.0.0.1:{server.port}/api/assistant", backend="kev",
                               headers={"X-ArWen-Token": server.token})
    record = decider.decide(Question("size", "How big?", {"small": "300 km", "large": "2500 km"}), {"x": 1})
    assert record["choice"] == "small" and record["method"] == "systemone" and record["confidence"] is not None


REFUSAL = ("gpuwm run-plan: run plan 'config.intent' does not fit: the forecast needs 48.03 GiB peak envelope, "
           "that EXCEEDS the 22.06 GiB budget by 25.97 GiB; reduce the buffer")


class RefusingRunner(FakeRunner):
    """Check the fit refuses the 1 km ladder, the way the wizard does on a card too small for it."""

    refusal = REFUSAL
    refuse_all = False

    def query(self, argv, *, cwd=None, timeout=0, log=None):
        if "--resolve" in argv and cwd is not None:
            intent = json.loads((cwd / "plan.json").read_text())["config"]["intent"]
            if self.refuse_all or intent.get("ladder") == "12-3-1":
                from gpuwm.gui.jobs import Refused
                raise Refused(self.refusal)
        return super().query(argv, cwd=cwd, timeout=timeout, log=log)


def test_a_refused_fit_says_what_was_given_up_beside_the_field_and_in_the_reply(rig):
    server, _, model = rig
    server.api.runner.__class__ = RefusingRunner
    model.finest_km = 1
    model.pick_id = {"Which grid ladder gives the detail they asked for at the least cost?": "12-3-1",
                     "The check refused the plan (see refusal). Which change fixes it and keeps most of what "
                     "the person asked for?": "coarser-ladder"}
    response, turn = request(server, "POST", "/api/assistant/say",
                             body={"text": "storms over the plains tomorrow afternoon at 1 km"})
    assert response.status == 200, turn
    fill = next(a for a in turn["actions"] if a["type"] == "fill_create")
    assert fill["fields"]["ladder"] == "12-3"
    assert fill["fixes"] == turn["plan"]["fixes"]
    [fix] = fill["fixes"]
    assert (fix["id"], fix["field"], fix["before"], fix["after"]) == ("coarser-ladder", "ladder", "12-3-1", "12-3")
    reason = fill["reasons"]["ladder"]
    assert "You asked for 1 km" in reason and "stops at 3 km" in reason
    assert "48.0 GiB" in reason and "22.1 GiB" in reason and "1 km grid" not in reason
    assert "It is not everything you asked for." in turn["reply"] and fix["words"] in turn["reply"]


def test_a_refused_start_keeps_the_model_and_a_local_server_that_cannot_be_unloaded_is_named(rig):
    server, runner, model = rig
    stopped = []
    request(server, "POST", "/api/assistant/settings", body={"backend": "bundled"})
    server.api.assistant.local.stop = lambda: stopped.append(True) or True

    def busy(rundir, argv):
        from gpuwm.gui.jobs import Refused
        raise Refused("another forecast is running")

    runner.launch = busy
    response, _ = request(server, "POST", "/api/create/start",
                          body={"name": "r2", "source": "gfs", "lat": 40, "lon": -97, "card": "16gb"})
    assert response.status == 409 and stopped == []
    del runner.launch
    # an LM Studio style server on this computer: nothing can take its model off the card, so it is said
    request(server, "POST", "/api/assistant/settings",
            body={"backend": "endpoint", "endpoint_url": model.url, "endpoint_model": "scripted"})
    response, fit = request(server, "POST", "/api/create/fit",
                            body={"source": "gfs", "lat": 40, "lon": -97, "card": "16gb"})
    assert response.status == 200 and "may still hold card memory" in fit["fit"]["warning"]
    response, reply = request(server, "POST", "/api/create/start",
                              body={"name": "r3", "source": "gfs", "lat": 40, "lon": -97, "card": "16gb"})
    assert response.status == 200 and reply["assistant"]["unloaded"] is False
    assert "may still hold card memory" in reply["assistant"]["warning"]


# The home-directory path is assembled at run time so the release snapshot's
# machine-path scan does not read these fixtures as real paths.
_WINDOWS_HOME = "C:" + "\\" + "Users" + "\\someone"


def _engine_refusal(verdict: str) -> str:
    """The refusal run-plan prints, built by the engine's own code: the wizard's sentence, then the --explain tail."""

    from gpuwm.explain import layered, render

    draft = _WINDOWS_HOME + r"\runs\.arwen-gui\drafts\00cb040d918c\plan.json"
    message = layered(f"run plan 'config.intent' does not fit: {verdict}", "the floor")
    return "gpuwm run-plan: " + render(message, explain=False, command=f"gpuwm run-plan {draft} --resolve")


def _ingest_priced_verdict() -> str:
    from gpuwm.core.preflight import GIB, PhaseMemoryEstimate

    phases = PhaseMemoryEstimate(forecast=None, ingest=None, forecast_envelope_bytes=int(48.0 * GIB),
                                 ingest_envelope_bytes=int(26.06 * GIB), source="gfs")
    assert phases.ingest_priced
    return phases.verdict(int(10.75 * GIB))


def test_a_refused_fit_on_the_ingest_priced_wording_gives_its_numbers_and_no_path(rig):
    server, _, model = rig
    refusal = _engine_refusal(_ingest_priced_verdict())
    assert "memory-binding phase" in refusal and "needs" not in refusal and "(run " in refusal

    class Runner(RefusingRunner):
        pass

    Runner.refusal = refusal
    server.api.runner.__class__ = Runner
    model.finest_km = 1
    model.pick_id = {"Which grid ladder gives the detail they asked for at the least cost?": "12-3-1",
                     "The check refused the plan (see refusal). Which change fixes it and keeps most of what "
                     "the person asked for?": "coarser-ladder"}
    response, turn = request(server, "POST", "/api/assistant/say",
                             body={"text": "storms over the plains tomorrow afternoon at 1 km"})
    assert response.status == 200, turn
    fill = next(a for a in turn["actions"] if a["type"] == "fill_create")
    reason = fill["reasons"]["ladder"]
    assert "48.0 GiB" in reason and "10.8 GiB" in reason, reason
    for text in (reason, turn["reply"]):
        assert "(run " not in text and "drafts" not in text and "plan.json" not in text and "\\" not in text, text


def test_cost_words_reads_every_verdict_wording_and_never_hands_back_an_engine_line():
    from gpuwm.gui.assistant.plan import cost_words

    assert cost_words(_engine_refusal(_ingest_priced_verdict())) == (
        "it needs 48.0 GiB of card memory and this card has 10.8 GiB for it")
    assert "48.0 GiB" in cost_words(REFUSAL)
    words = cost_words(_engine_refusal("the layout bottomed out at 120 x 120"))
    assert words == "it does not fit this card"


def test_a_refusal_about_the_install_is_said_at_once_without_plan_changes(rig):
    server, _, model = rig
    install = ("Traceback (most recent call last):\n  File \"data_assets.py\", line 349, in _check_version\n"
               "    raise ImportError(\n    ^^^^^^^^^^^^\n"
               "ImportError:the physics tables need gpuwm-data 0.4 or newer and 0.3.1 is installed at "
               + _WINDOWS_HOME + r"\venv\Lib\site-packages" + "\n"
               "Install it with: pip install -U gpuwm-data\n"
               "  (run gpuwm run-plan " + _WINDOWS_HOME + r"\runs\.arwen-gui\drafts\ab12\plan.json --resolve "
               "--explain for the reason)")

    class Runner(RefusingRunner):
        pass

    Runner.refusal = install
    Runner.refuse_all = True
    server.api.runner.__class__ = Runner
    response, turn = request(server, "POST", "/api/assistant/say",
                             body={"text": "storms over the plains tomorrow afternoon"})
    assert response.status == 200, turn
    asked = [body["messages"][-1]["content"] for body in model.bodies]
    assert not any("The check refused the plan (see refusal)" in text for text in asked)
    assert not any(a["type"] == "fill_create" for a in turn["actions"])
    reply = turn["reply"]
    assert "gpuwm-data 0.4" in reply and "pip install -U gpuwm-data" in reply, reply
    assert "(run " not in reply and "drafts" not in reply and "site-packages" not in reply, reply
    assert "Traceback" not in reply and "ImportError" not in reply and "File " not in reply, reply


def test_a_question_about_a_storm_is_answered_from_the_wiki(rig):
    server, runner, model = rig
    response, turn = request(server, "POST", "/api/assistant/say",
                             body={"text": "What happened in the 2013 Oklahoma EF5 tornado?"}, timeout=30)
    assert response.status == 200, turn
    assert [c["name"] for c in turn["tool_calls"]] == ["search_wiki", "read_event"]
    assert all(c["ok"] for c in turn["tool_calls"])
    assert "EF5" in turn["reply"] and "Deaths in the record: 24" in turn["reply"]
    read = model.bodies[-1]["messages"][-1]["content"]
    # the model reads words and facts, never the page's map geometry
    assert '"facts"' in read and "geometry" not in read and "path_cite" not in read


def test_a_question_about_a_run_is_answered_from_its_folder(rig):
    from gpuwm.gui.assistant.agent import Agent, Turn

    server, runner, model = rig
    rundir = server.api.root / "r1"
    rundir.mkdir(parents=True)
    (rundir / "plan.json").write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "config": {"intent": {
        "source": "gfs", "cycle": "2026-09-25T18", "hours": 3, "physics_profile": "some-suite-v1"}}}))
    agent = Agent(server.api, server.api.assistant.chat())
    words = agent._execute(Turn(user="x"), "read_run", {"run": "r1"}, {})
    facts = {f["label"]: f["text"] for f in words["facts"]}
    assert facts["Physics"] == "some-suite-v1" and facts["Length"] == "3 h"


def test_with_no_model_the_assistant_says_how_to_get_one_at_once(tmp_path, monkeypatch):
    import time

    monkeypatch.setenv("GPUWM_ASSISTANT_HOME", str(tmp_path / "empty-home"))
    server = build_server(tmp_path / "runs", port=0, runner=FakeRunner(), token="t" * 43)
    serve_in_thread(server)
    try:
        response, turned = request(server, "POST", "/api/assistant/enable", body={"on": True})
        # Turning it on downloads nothing: it hands back what Set up would download, to be asked about first.
        assert response.status == 200 and turned["enabled"] is True and turned["install"]["items"]
        assert not (tmp_path / "empty-home").exists()
        response, status = request(server, "GET", "/api/assistant")
        assert response.status == 200 and status["downloaded"] is False and status["model"]["bytes"] > 0
        began = time.monotonic()
        response, answer = request(server, "POST", "/api/assistant/say", body={"text": "Hello?"}, timeout=10)
        assert time.monotonic() - began < 5
        assert response.status == 409
        assert "not downloaded" in answer["message"] and "Set up" in answer["fix"]
        assert answer["install"]["items"] and "Traceback" not in json.dumps(answer)
        # The folder is named in words, in the status and in the refusal, and never as this computer's path.
        assert status["home"] == answer["install"]["home"] == "the folder GPUWM_ASSISTANT_HOME names"
        assert str(tmp_path) not in json.dumps(status) + json.dumps(answer)
    finally:
        server.shutdown()
        server.server_close()



def test_the_model_folder_is_named_from_the_home_folder(monkeypatch):
    from pathlib import Path as _Path

    from gpuwm.gui.assistant.local import home, home_words

    monkeypatch.delenv("GPUWM_ASSISTANT_HOME", raising=False)
    assert home_words(home()) == "the .arwen/assistant folder in your home folder"
    assert str(_Path.home()) not in home_words(home())


@pytest.fixture()
def off_rig(tmp_path, monkeypatch):
    """A fresh page with the assistant as it ships: off. Anything that would download, start or ask a model fails loudly."""

    from gpuwm.gui.assistant import local as local_module

    monkeypatch.setenv("GPUWM_ASSISTANT_HOME", str(tmp_path / "home"))
    touched: list[str] = []

    def forbidden(name):
        def call(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"{name} while the assistant is off")
        return call

    monkeypatch.setattr(local_module.Local, "install", forbidden("install"))
    monkeypatch.setattr(local_module.Local, "start", forbidden("start"))
    monkeypatch.setattr(local_module.Local, "running", forbidden("running"))
    monkeypatch.setattr(local_module.urllib.request, "urlopen", forbidden("urlopen"))
    monkeypatch.setattr("gpuwm.gui.assistant.llm.Chat.__init__", forbidden("chat"))
    server = build_server(tmp_path / "runs", port=0, runner=FakeRunner(), token="t" * 43)
    serve_in_thread(server)
    yield server, touched, tmp_path / "home"
    server.shutdown()
    server.server_close()


def test_the_assistant_ships_off_and_says_how_to_turn_it_on(off_rig):
    server, touched, home = off_rig
    response, session = request(server, "GET", "/api/session")
    assert response.status == 200 and session["assistant"] == {"enabled": False}
    response, status = request(server, "GET", "/api/assistant")
    assert response.status == 200 and status["enabled"] is False and status["loaded"] is False
    assert "Turn it on in Settings" in status["off"]
    # what turning it on would mean for this card: the model, its size and its licence
    assert status["model"]["name"] and status["model"]["bytes"] > 0 and status["model"]["licence"]["id"]
    assert status["downloaded_bytes"] == 0
    response, table = request(server, "GET", "/api/assistant/catalog")
    assert response.status == 200 and table["brought"]
    assert touched == [] and not home.exists()


def test_off_nothing_downloads_starts_or_asks_a_model(off_rig):
    server, touched, home = off_rig
    for path, body in [("/api/assistant/install", {"confirm": True}), ("/api/assistant/install", {"dry_run": True}),
                       ("/api/assistant/load", {}), ("/api/assistant/say", {"text": "Hello?"}),
                       ("/api/assistant/decide", {"question": {"id": "q", "instructions": "x", "options": {"a": "b"}}}),
                       ("/api/assistant/v1/systemone",
                        {"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "b"}}}})]:
        response, answer = request(server, "POST", path, body=body)
        assert response.status == 409, (path, answer)
        assert answer["message"] == "The assistant is off." and "Turn it on" in answer["fix"] and answer["off"] is True
    assert touched == [] and not home.exists()


def test_off_the_pages_and_a_forecast_are_unaffected(off_rig, tmp_path):
    server, touched, home = off_rig
    # an endpoint set while off is remembered, but nothing is said about it and nothing is asked of it
    request(server, "POST", "/api/assistant/settings",
            body={"backend": "endpoint", "endpoint_url": "http://127.0.0.1:1234/v1", "endpoint_model": "m"})
    for path in ("/api/session", "/api/runs", "/api/wiki", "/api/sources", "/api/system"):
        response, _ = request(server, "GET", path)
        assert response.status == 200, path
    response, fit = request(server, "POST", "/api/create/fit",
                            body={"source": "gfs", "lat": 40, "lon": -97, "card": "16gb"})
    assert response.status == 200 and "warning" not in fit["fit"]
    assert "card memory" not in json.dumps(fit)
    response, reply = request(server, "POST", "/api/create/start",
                              body={"name": "r1", "source": "gfs", "lat": 40, "lon": -97, "card": "16gb"})
    assert response.status == 200 and reply["assistant"] is None
    assert touched == [] and not home.exists()


def test_turned_on_then_off_the_server_stops_and_the_model_can_be_removed(rig, tmp_path):
    server, _, _ = rig
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True)
    (home / "models" / "m.gguf").write_bytes(b"x" * 1000)
    (home / "server" / "b").mkdir(parents=True)
    (home / "server" / "b" / "llama-server").write_bytes(b"y" * 24)
    (home / "keep.txt").write_text("not the assistant's download")
    stopped = []
    server.api.assistant.local.stop = lambda: stopped.append(True) or True
    response, off = request(server, "POST", "/api/assistant/enable", body={"on": False})
    assert response.status == 200 and off["enabled"] is False and off["stopped"] is True and stopped
    assert off["downloaded_bytes"] == 1024
    response, _ = request(server, "POST", "/api/assistant/say", body={"text": "Hello?"})
    assert response.status == 409
    response, plan = request(server, "POST", "/api/assistant/remove", body={"dry_run": True})
    assert response.status == 200 and plan["bytes"] == 1024 and (home / "models" / "m.gguf").is_file()
    response, gone = request(server, "POST", "/api/assistant/remove", body={})
    assert response.status == 200 and gone["freed_bytes"] == 1024 and gone["downloaded_bytes"] == 0
    assert not (home / "models").exists() and not (home / "server").exists() and (home / "keep.txt").is_file()
    response, bad = request(server, "POST", "/api/assistant/enable", body={"on": "yes"})
    assert response.status == 400


def test_the_terminal_says_how_to_turn_it_on_and_asks_before_any_download(off_rig, monkeypatch, capsys):
    import argparse

    from gpuwm.gui.assistant import cli
    from gpuwm.gui.assistant import local as local_module

    server, touched, home = off_rig
    monkeypatch.setattr(cli, "_api", lambda root: server.api)
    parser = argparse.ArgumentParser()
    cli.register_cli(parser.add_subparsers())

    def run(*words):
        arguments = parser.parse_args(["assistant", *words])
        return arguments.func(arguments)

    for words in (("setup",), ("say", "hello")):
        assert run(*words) == 1
        assert "The assistant is off. Turn it on with: gpuwm assistant on" in capsys.readouterr().err
    # on: the model, its size and licence, then each file, and a no downloads nothing
    monkeypatch.setattr(local_module.Local, "running", lambda self: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert run("on") == 0
    said = capsys.readouterr().out
    assert "The assistant is on." in said and "licence Apache-2.0" in said and "Nothing downloaded." in said
    assert run("off") == 0
    assert "The assistant is off." in capsys.readouterr().out
    assert touched == [] and not home.exists()
