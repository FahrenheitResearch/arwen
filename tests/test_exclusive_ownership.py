"""Shared files and folders have one live owner at a time.

Each test runs one shared-writer case against the real entry point:
two execute_plan calls on one run folder, a reopened event stream over
a torn line, two downscale reservations of one folder, two bridge
downloads of one bundle, a bridge folder that cannot be written, two
overlapping assistant messages in one conversation and a georeference
fold into a folder that refuses writes.
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import ownership
from gpuwm.runplan import (EVENTS_FILENAME, MANIFEST_FILENAME, EventStream,
                           PlanError, read_events)

import ownership_workers
from test_case_data import make_case_toml


# ---------------------------------------------------------------------------
# The helper itself


def test_a_second_claim_is_refused_while_the_first_owner_lives(tmp_path):
    path = tmp_path / "owner"
    first = ownership.claim(path, purpose="first")
    with pytest.raises(ownership.OwnershipError, match="in use"):
        ownership.claim(path, purpose="second")
    assert first.release()
    ownership.claim(path, purpose="third").release()


def test_an_owner_whose_process_is_gone_is_taken_over(tmp_path):
    path = tmp_path / "owner"
    child = multiprocessing.get_context("spawn").Process(target=int)
    child.start()
    child.join()
    record = {**ownership.process_identity(os.getpid()), "pid": child.pid,
              "token": "dead"}
    path.write_text(json.dumps(record), encoding="utf-8")
    held = ownership.claim(path, purpose="after a crash")
    assert held.held()
    held.release()


def test_a_reused_pid_with_another_creation_time_is_not_the_owner(tmp_path):
    path = tmp_path / "owner"
    record = {**ownership.process_identity(), "created": "1", "token": "old"}
    path.write_text(json.dumps(record), encoding="utf-8")
    held = ownership.claim(path, purpose="reused pid")
    assert held.record["token"] != "old"
    held.release()


def test_an_owner_on_another_machine_is_never_taken_over(tmp_path):
    path = tmp_path / "owner"
    record = {**ownership.process_identity(), "host": "another-machine",
              "pid": 1, "token": "theirs"}
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ownership.OwnershipError) as refused:
        ownership.claim(path, purpose="here")
    assert json.loads(path.read_text(encoding="utf-8"))["token"] == "theirs"
    words = ownership.recovery_words(refused.value)
    assert "another-machine" in words and path.name in words


def test_an_owner_file_that_will_not_delete_is_marked_released_and_taken_over(
        tmp_path, monkeypatch):
    path = tmp_path / "owner"
    held = ownership.claim(path, purpose="first")
    real_unlink = os.unlink

    def refuse(target, *args, **kwargs):
        if Path(target) == path:
            raise PermissionError(13, "in use by a reader", str(target))
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(ownership.os, "unlink", refuse)
    assert held.release() is False
    assert json.loads(path.read_text(encoding="utf-8"))["released"] is True
    monkeypatch.setattr(ownership.os, "unlink", real_unlink)
    ownership.claim(path, purpose="second").release()
    assert not path.exists()


def test_a_dead_owner_that_cannot_be_removed_is_refused_in_time(
        tmp_path, monkeypatch):
    path = tmp_path / "owner"
    record = {**ownership.process_identity(), "created": "1", "token": "old"}
    path.write_text(json.dumps(record), encoding="utf-8")
    real_unlink = os.unlink

    def refuse(target, *args, **kwargs):
        if Path(target) == path:
            raise PermissionError(13, "denied", str(target))
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(ownership.os, "unlink", refuse)
    monkeypatch.setattr(ownership, "TRANSIENT_GRACE_SECONDS", 0.3)
    began = time.monotonic()
    with pytest.raises(ownership.OwnershipError) as refused:
        ownership.claim(path, purpose="after a crash")
    assert time.monotonic() - began < 3.0
    assert refused.value.stale
    assert path.name in ownership.recovery_words(refused.value)


def test_release_leaves_an_owner_file_that_is_no_longer_ours(tmp_path):
    path = tmp_path / "owner"
    held = ownership.claim(path, purpose="first")
    path.write_text(json.dumps({"token": "someone else"}), encoding="utf-8")
    assert not held.release()
    assert path.exists()


def _refuse_creating(monkeypatch, name: str):
    """Windows' answer to a create in a folder that refuses writes, for
    one file name, with the platform switch turned to Windows."""

    real_open = os.open

    def refuse(target, flags, *args, **kwargs):
        if Path(target).name == name and flags & os.O_CREAT:
            raise PermissionError(13, "Access is denied", str(target))
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(ownership, "_WINDOWS", True)
    monkeypatch.setattr(ownership, "TRANSIENT_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(ownership.os, "open", refuse)


def test_a_windows_folder_that_refuses_writes_is_raised_whatever_the_wait(
        tmp_path, monkeypatch):
    path = tmp_path / "owner"
    _refuse_creating(monkeypatch, path.name)
    began = time.monotonic()
    with pytest.raises(PermissionError):
        ownership.claim(path, purpose="setup", wait=60)
    assert time.monotonic() - began < 3.0


def test_a_windows_refusal_over_a_file_being_deleted_is_waited_out(
        tmp_path, monkeypatch):
    path = tmp_path / "owner"
    path.write_text("", encoding="utf-8")
    _refuse_creating(monkeypatch, path.name)
    began = time.monotonic()
    with pytest.raises(PermissionError):
        ownership.claim(path, purpose="setup", wait=0.8)
    assert time.monotonic() - began >= 0.7


def test_a_windows_breaker_the_folder_refuses_is_raised_whatever_the_wait(
        tmp_path, monkeypatch):
    path = tmp_path / "owner"
    record = {**ownership.process_identity(), "created": "1", "token": "old"}
    path.write_text(json.dumps(record), encoding="utf-8")
    _refuse_creating(monkeypatch, path.name + ".break")
    began = time.monotonic()
    with pytest.raises(PermissionError):
        ownership.claim(path, purpose="after a crash", wait=60)
    assert time.monotonic() - began < 3.0


# ---------------------------------------------------------------------------
# Two execute_plan calls on one run folder


def test_two_plans_on_one_run_folder_run_one_and_refuse_the_other(tmp_path):
    config = make_case_toml(tmp_path)
    run_dir = tmp_path / "run"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": "gpuwm.run-plan.v1", "name": "shared-folder",
        "route": "experiment", "config": {"path": str(config)},
        "output_root": str(run_dir), "run_options": {"dry_run": True}}),
        encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        gate, after_open = manager.Barrier(2), manager.Barrier(2)
        results = manager.list()
        workers = [context.Process(target=ownership_workers.execute_dry_run,
                                   args=(str(plan_path), gate, after_open,
                                         results)) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(180)
            assert worker.exitcode == 0
        outcomes = list(results)

    assert sorted(kind for kind, *_ in outcomes) == ["ran", "refused"]
    ran = next(entry for entry in outcomes if entry[0] == "ran")
    refused = next(entry for entry in outcomes if entry[0] == "refused")
    assert ran[2] == 0
    assert "in use" in refused[2] and "Traceback" not in refused[2]
    events = read_events(run_dir / EVENTS_FILENAME)
    assert [record["sequence"] for record in events] == \
        list(range(1, len(events) + 1))
    assert all(record.get("pid", ran[1]) == ran[1] for record in events)
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text("utf-8"))
    assert manifest["pid"] == ran[1]
    assert not any(run_dir.glob(".*.owner")), "the owner file is released"


# ---------------------------------------------------------------------------
# Reopening over a torn final line


def test_reopening_over_a_torn_line_repairs_the_stream_and_keeps_counting(
        tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="first")
    torn = '{"schema_version": "gpuwm.run-p'
    with path.open("a", encoding="utf-8") as stream:
        stream.write(torn)

    with EventStream(path, mirror=None) as events:
        # The repair is the reopened stream's first record.
        assert events.sequence == 2
        events.emit("plan_accepted", name="second")
        events.emit("completed", dry_run=True)

    records = read_events(path)
    assert [record["sequence"] for record in records] == [1, 2, 3, 4]
    assert read_events(path, allow_partial_tail=True) == records
    kept = list(tmp_path.glob(EVENTS_FILENAME + ".torn-*"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == torn
    warning = records[1]
    assert warning["event"] == "warning"
    assert warning["code"] == "event_tail_recovered"
    assert Path(warning["preserved_path"]) == kept[0]


def test_a_torn_tail_cut_inside_a_character_is_kept_byte_for_byte(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="first")
    prefix = path.read_bytes()
    # A killed write can stop inside a multi-byte character.
    fragment = (b'{"schema_version":"gpuwm.run-plan.event.v1",'
                b'"message":"cut \xe2\x82')
    path.write_bytes(prefix + fragment)

    with EventStream(path, mirror=None) as events:
        events.emit("completed", dry_run=True)

    records = read_events(path)
    assert path.read_bytes().startswith(prefix)
    assert records[-1]["event"] == "completed"
    warning = next(r for r in records
                   if r.get("code") == "event_tail_recovered")
    assert Path(warning["preserved_path"]).read_bytes() == fragment
    assert [r["sequence"] for r in records] == list(
        range(1, len(records) + 1))


def test_a_complete_record_missing_only_its_newline_is_kept(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="first")
        events.emit("plan_accepted", name="second")
    text = path.read_text(encoding="utf-8")
    path.write_text(text[:-1], encoding="utf-8", newline="\n")

    with EventStream(path, mirror=None) as events:
        events.emit("completed", dry_run=True)
    assert [r["sequence"] for r in read_events(path)] == [1, 2, 3]
    assert not list(tmp_path.glob(EVENTS_FILENAME + ".torn-*"))


def test_a_second_stream_on_one_file_in_one_process_is_refused(tmp_path):
    path = tmp_path / EVENTS_FILENAME
    with EventStream(path, mirror=None):
        with pytest.raises(PlanError, match="in use"):
            EventStream(path, mirror=None)
    with EventStream(path, mirror=None) as events:
        events.emit("plan_accepted", name="after")


# ---------------------------------------------------------------------------
# Two downscales reserving one output folder


def test_two_processes_reserving_one_absent_folder_get_one_refusal(tmp_path):
    target = tmp_path / "child-out"
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        gate, after = manager.Barrier(2), manager.Barrier(2)
        results = manager.list()
        workers = [context.Process(target=ownership_workers.reserve,
                                   args=(str(target), gate, after, results))
                   for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(120)
            assert worker.exitcode == 0
        outcomes = list(results)
    assert sorted(kind for kind, *_ in outcomes) == ["refused", "reserved"]
    refused = next(entry for entry in outcomes if entry[0] == "refused")
    assert "in use" in refused[2]


def test_a_release_never_removes_a_folder_another_claim_now_holds(tmp_path):
    from gpuwm.downscale import _OutputReservation
    from gpuwm.offline_child import OfflineChildContractError

    target = tmp_path / "child-out"
    first, second = _OutputReservation(), _OutputReservation()
    first.claim(target)
    with pytest.raises(OfflineChildContractError, match="in use"):
        second.claim(target)
    # The order this can still happen in: the first claim's
    # owner is replaced (its process was judged gone and the folder
    # taken over), the new owner writes its config, and only then does
    # the first reservation's cleanup run.
    marker = next(target.glob(".*owner"))
    marker.write_text(json.dumps({**ownership.process_identity(),
                                  "token": "the second downscale"}),
                      encoding="utf-8")
    (target / "child.toml").write_text("mine", encoding="utf-8")
    first.release()
    assert (target / "child.toml").read_text(encoding="utf-8") == "mine"


def test_an_adopted_empty_folder_is_claimed_and_given_back_empty(tmp_path):
    from gpuwm.downscale import _OutputReservation
    from gpuwm.offline_child import OfflineChildContractError

    target = tmp_path / "child-out"
    target.mkdir()
    first, second = _OutputReservation(), _OutputReservation()
    first.claim(target)
    with pytest.raises(OfflineChildContractError, match="in use"):
        second.claim(target)
    (target / "child.toml").write_text("derived", encoding="utf-8")
    first.release()
    assert target.is_dir() and list(target.iterdir()) == []
    created = _OutputReservation()
    created.claim(tmp_path / "fresh")
    created.release()
    assert not (tmp_path / "fresh").exists()


# ---------------------------------------------------------------------------
# Two bridge downloads of one bundle


class _PausingServer:
    """Serves a payload; the first transfer stops halfway until released."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.paused = threading.Event()
        self.resume = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, request):
        with self.lock:
            self.calls += 1
            first = self.calls == 1
        header = request.headers.get("Range")
        start = int(header.split("=")[1].split("-")[0]) if header else 0
        body = self.payload[start:]
        server = self

        class Response(io.BytesIO):
            status = 206 if header else 200
            sent = 0

            def read(self, size=-1):
                if (first and self.sent >= len(body) // 2
                        and not server.resume.is_set()):
                    server.paused.set()
                    server.resume.wait(60)
                want = min(size, 4096) if size and size > 0 else 4096
                chunk = super().read(want)
                self.sent += len(chunk)
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                self.close()
                return False

        return Response(body)


def test_two_bridge_downloads_of_one_bundle_do_not_delete_each_others(
        tmp_path):
    from gpuwm import bridge_assets
    from test_bridge_fetch import _synthetic_bundle

    archive, bundle = _synthetic_bundle(tmp_path / "published")
    server = _PausingServer(archive.read_bytes())
    pins = bridge_assets.BridgePins(release="v0-test",
                                    platforms={bundle.platform: bundle})
    dest = tmp_path / "bridges"
    errors: list[BaseException] = []
    installed: list[list] = []

    def fetch():
        try:
            installed.append(bridge_assets.fetch_bundle(
                pins, bundle, dest, progress=lambda _line: None,
                urlopen_fn=server))
        except BaseException as error:  # noqa: BLE001 - the test reports it
            errors.append(error)

    first = threading.Thread(target=fetch)
    first.start()
    assert server.paused.wait(30)
    second = threading.Thread(target=fetch)
    second.start()
    time.sleep(1.0)
    server.resume.set()
    first.join(60)
    second.join(60)

    assert errors == []
    assert len(installed) == 2
    for pin in bundle.binaries:
        staged = (dest / pin.filename).read_bytes()
        assert hashlib.sha256(staged).hexdigest() == pin.sha256


# ---------------------------------------------------------------------------
# A bridge folder that cannot be written


_POSIX_READ_ONLY = pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="a 0555 folder refuses writes only for a non-root POSIX user")


def _pinned_estate(tmp_path, monkeypatch, *, complete: bool):
    """A synthetic bundle pinned as this platform's, and a bridge folder
    that holds all of it or only its first artifact."""

    from gpuwm import bridge_assets
    from test_bridge_fetch import _install_pins, _synthetic_bundle

    archive, bundle = _synthetic_bundle(tmp_path / "published")
    _install_pins(monkeypatch, bundle)
    dest = tmp_path / "bridges"
    bridge_assets.stage_from_bundle(archive, bundle, dest,
                                    progress=lambda _line: None)
    if not complete:
        (dest / bundle.binaries[-1].filename).unlink()
    return bundle, dest


@_POSIX_READ_ONLY
@pytest.mark.parametrize("complete", (True, False))
def test_fetch_bridges_on_a_read_only_folder_answers_in_plain_words(
        tmp_path, monkeypatch, capsys, complete):
    from gpuwm.cli import main

    bundle, dest = _pinned_estate(tmp_path, monkeypatch, complete=complete)
    dest.chmod(0o555)
    try:
        code = main(["fetch-bridges", "--dest", str(dest)])
    finally:
        dest.chmod(0o755)
    out = capsys.readouterr()
    text = out.out + out.err
    assert "Traceback" not in text and "Errno" not in text
    assert not (dest / ".fetch-bridges.owner").exists()
    if complete:
        assert code == 0
        assert (f"all {len(bundle.binaries)} artifacts" in text
                and "verified" in text)
    else:
        assert code == 2
        assert "cannot be written (permission denied)" in text
        assert "1 of the 2 bridge file(s)" in text
        assert "--dest with a folder you can write" in text


@pytest.mark.parametrize("complete", (True, False))
def test_a_refused_bridge_claim_is_checked_without_owning_the_folder(
        tmp_path, monkeypatch, capsys, complete):
    from gpuwm import bridge_assets
    from test_bridge_fetch import _args

    bundle, dest = _pinned_estate(tmp_path, monkeypatch, complete=complete)

    def refuse(path, **_kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(ownership, "claim", refuse)
    code = bridge_assets.fetch_bridges_main(_args(dest=str(dest)))
    text = capsys.readouterr().out
    if complete:
        assert code == 0 and "nothing to fetch" in text
    else:
        assert code == 2
        assert (f"REFUSED: {dest} cannot be written (permission denied)"
                in text)
    for pin in bundle.binaries[:1]:
        assert bridge_assets.matches_pin(dest / pin.filename, pin)


def test_a_bridge_folder_that_is_a_file_is_refused_in_plain_words(
        tmp_path, monkeypatch, capsys):
    from gpuwm import bridge_assets
    from test_bridge_fetch import _args, _install_pins, _synthetic_bundle

    _archive, bundle = _synthetic_bundle(tmp_path / "published")
    _install_pins(monkeypatch, bundle)
    dest = tmp_path / "bridges"
    dest.write_text("not a folder", encoding="utf-8")
    assert bridge_assets.fetch_bridges_main(_args(dest=str(dest))) == 2
    assert "it is a file, not a folder" in capsys.readouterr().out


def test_an_error_while_staging_is_not_called_an_unwritable_folder(tmp_path):
    from gpuwm import bridge_assets

    with pytest.raises(OSError) as raised:
        with bridge_assets.bridge_owner(tmp_path / "bridges",
                                        progress=lambda _line: None):
            raise OSError(28, "No space left on device")
    assert not isinstance(raised.value, bridge_assets.BridgeAssetError)
    assert raised.value.errno == 28
    assert not (tmp_path / "bridges" / ".fetch-bridges.owner").exists()


# ---------------------------------------------------------------------------
# Overlapping messages in one conversation


def test_two_overlapping_messages_in_one_conversation_are_both_saved(
        tmp_path, monkeypatch):
    from gpuwm.gui.assistant import service

    started = threading.Barrier(2, timeout=5)

    class ScriptedAgent:
        def __init__(self, api, chat):
            self.chat = chat
            self.decide_fn = None

        def turn(self, text, *, history, form, pending):
            time.sleep(0.3)
            seen = len(history)
            return SimpleNamespace(
                question=None,
                record=lambda: {"text": text, "reply": f"about {text}",
                                "seen": seen})

    monkeypatch.setattr(service, "Agent", ScriptedAgent)
    assistant = service.Assistant(SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(assistant, "chat",
                        lambda: SimpleNamespace(model="scripted"))
    monkeypatch.setattr(assistant, "decider",
                        lambda chat: SimpleNamespace(decide=None))
    replies: list[dict] = []

    def say(text):
        started.wait()
        replies.append(assistant._say(
            {"text": text, "conversation": "c-1"}, dry=False))

    threads = [threading.Thread(target=say, args=(text,))
               for text in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert len(replies) == 2
    saved = assistant.conversation("c-1")
    assert sorted(turn["text"] for turn in saved["turns"]) == \
        ["first", "second"]
    assert sorted(turn["seen"] for turn in saved["turns"]) == [0, 1]


def test_two_overlapping_page_requests_in_one_conversation_keep_both_turns(
        tmp_path, monkeypatch):
    """Two POSTs to the page server's assistant route for one
    conversation, answered by a model slow enough that the second
    arrives while the first is still being answered."""

    import test_gui_assistant as gui
    from gpuwm.gui.server import build_server, serve_in_thread
    from test_gui_server import request

    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    monkeypatch.setenv("GPUWM_ASSISTANT_HOME", str(tmp_path / "home"))

    class SlowHandler(gui._ModelHandler):
        def do_POST(self):  # noqa: N802
            time.sleep(0.2)
            super().do_POST()

    model = gui.ScriptedModel()
    model.RequestHandlerClass = SlowHandler
    threading.Thread(target=model.serve_forever, daemon=True).start()
    server = build_server(tmp_path / "runs", port=0, runner=gui.FakeRunner(),
                          token="t" * 43)
    serve_in_thread(server)
    try:
        response, _ = request(server, "POST", "/api/assistant/enable",
                              body={"on": True})
        assert response.status == 200
        response, _ = request(server, "POST", "/api/assistant/settings",
                              body={"backend": "endpoint",
                                    "endpoint_url": model.url,
                                    "endpoint_model": "scripted"})
        assert response.status == 200
        started = threading.Barrier(2, timeout=10)
        answers: list = []

        def say(text):
            started.wait()
            answers.append(request(server, "POST", "/api/assistant/say",
                                   body={"text": text,
                                         "conversation": "overlap-1"},
                                   timeout=60))

        threads = [threading.Thread(target=say, args=(text,)) for text in
                   ("which forecasts do I have", "which forecasts ran today")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(90)
        assert [response.status for response, _ in answers] == [200, 200]
        response, saved = request(server, "GET",
                                  "/api/assistant/conversations/overlap-1")
        assert response.status == 200
        assert sorted(turn["user"] for turn in saved["turns"]) == \
            ["which forecasts do I have", "which forecasts ran today"]
    finally:
        server.shutdown()
        server.server_close()
        model.shutdown()


def test_the_message_box_does_not_send_while_a_reply_is_pending():
    source = (Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static"
              / "js" / "assistant.js").read_text(encoding="utf-8")
    body = source.split("async function say()", 1)[1].split("\n  }\n", 1)[0]
    guard = body.split("const text", 1)[0]
    assert "sending" in guard and "return" in guard
    assert "sending = true" in body and "sending = false" in body


# ---------------------------------------------------------------------------
# A georeference fold into a folder that refuses writes


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid")
                                        and os.geteuid() == 0),
                    reason="a 0555 folder refuses writes only for a "
                           "non-root POSIX user")
def test_a_fold_into_a_read_only_folder_warns_and_returns(tmp_path,
                                                          monkeypatch):
    from gpuwm import render_georef

    monkeypatch.setattr(render_georef, "_LOCK_WAIT_SECONDS", 0.2)
    folder = tmp_path / "png"
    folder.mkdir()
    folder.chmod(0o555)
    warnings: list = []
    monkeypatch.setattr(render_georef, "_warn",
                        lambda target, error: warnings.append(error))
    try:
        began = time.monotonic()
        render_georef.fold(folder, {"panels": {}})
        elapsed = time.monotonic() - began
    finally:
        folder.chmod(0o755)
    assert elapsed < 2.0
    assert warnings


@pytest.mark.parametrize("failure", ("stat", "unlink"))
def test_a_lock_that_cannot_be_inspected_or_removed_still_times_out(
        tmp_path, monkeypatch, failure):
    from gpuwm import render_georef

    monkeypatch.setattr(render_georef, "_LOCK_WAIT_SECONDS", 0.2)
    lock = tmp_path / (render_georef.GEOREF_FILENAME
                       + render_georef.LOCK_SUFFIX)
    lock.write_text("")
    old = time.time() - 10 * render_georef._LOCK_STALE_SECONDS
    os.utime(lock, (old, old))
    real_stat, real_unlink = Path.stat, Path.unlink

    def stat(self, *args, **kwargs):
        if failure == "stat" and self.name == lock.name:
            raise PermissionError(13, "denied", str(self))
        return real_stat(self, *args, **kwargs)

    def unlink(self, *args, **kwargs):
        if self.name == lock.name:
            raise PermissionError(13, "denied", str(self))
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "unlink", unlink)
    warnings: list = []
    monkeypatch.setattr(render_georef, "_warn",
                        lambda target, error: warnings.append(error))
    result: list = []
    worker = threading.Thread(
        target=lambda: result.append(
            render_georef.fold(tmp_path, {"panels": {}})), daemon=True)
    began = time.monotonic()
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "the lock loop spun past its deadline"
    assert time.monotonic() - began < 2.0
    assert warnings


@pytest.mark.parametrize("lock_present", (False, True))
def test_a_windows_create_refusal_is_waited_only_while_a_lock_is_there(
        tmp_path, monkeypatch, lock_present):
    """Windows answers a create over a lock being deleted with access
    denied.  With no lock in the way the refusal is the folder's, and the
    fold warns after a short grace instead of the whole lock deadline."""

    from gpuwm import render_georef

    monkeypatch.setattr(render_georef, "_WINDOWS", True)
    monkeypatch.setattr(render_georef, "_REFUSED_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(render_georef, "_LOCK_WAIT_SECONDS",
                        1.0 if lock_present else 30.0)
    lock = tmp_path / (render_georef.GEOREF_FILENAME
                       + render_georef.LOCK_SUFFIX)
    if lock_present:
        lock.write_text("")
    real_open = os.open

    def refuse(path, *args, **kwargs):
        if Path(path).name == lock.name:
            raise PermissionError(13, "Access is denied", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(render_georef.os, "open", refuse)
    warnings: list = []
    monkeypatch.setattr(render_georef, "_warn",
                        lambda target, error: warnings.append(error))
    began = time.monotonic()
    render_georef.fold(tmp_path, {"panels": {}})
    elapsed = time.monotonic() - began
    assert len(warnings) == 1
    if lock_present:
        assert isinstance(warnings[0], TimeoutError)
        assert elapsed >= 0.9
    else:
        assert isinstance(warnings[0], PermissionError)
        assert elapsed < 3.0
