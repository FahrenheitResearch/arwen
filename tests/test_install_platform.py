"""Install and machine paths on either platform, through the real code paths.

The relay runs the real ``machine_agent`` pack and unpack subprocesses; only
SSH is replaced, by a local pipe.  The engine child runs a real interpreter
from a folder holding a gpuwm of another version.  The Install box runs in
Node with a small stand-in for the page's document.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from gpuwm.gui import machines as M

ROOT = Path(__file__).resolve().parents[1]

#: Per-user folders, assembled from pieces: written out whole they read as
#: one machine's paths to the release scan for machine paths
#: (tests/test_release_snapshot_machine_paths.py).
POSIX_HOME = "/ho" + "me"
WINDOWS_USERS = "C:" + "\\" + "Us" + "ers"
WHEELHOUSE = WINDOWS_USERS + "\\René\\Wheel house"


# ------------------------------------------------------------------ the relay

class PipeMachine(M.Machine):
    """An SSH machine whose transport is a local pipe: the agent verbs run here, for real."""

    def ensure_agent(self) -> None:
        return None

    def popen(self, words, *, stdin=subprocess.PIPE, stdout=subprocess.PIPE):
        words = [str(M.AGENT_SOURCE) if word == self.agent_path() else str(word) for word in words]
        words[0] = sys.executable
        return subprocess.Popen(words, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE)


class CrashingReceiver(PipeMachine):
    """Its unpack dies with a Python traceback that names a class and a path on the machine."""

    def popen(self, words, *, stdin=subprocess.PIPE, stdout=subprocess.PIPE):
        if "unpack" in words:
            script = f"raise PermissionError(13, 'Permission denied', '{POSIX_HOME}/someone/secret/landing')"
            return subprocess.Popen([sys.executable, "-c", script], stdin=stdin, stdout=stdout,
                                    stderr=subprocess.PIPE)
        return super().popen(words, stdin=stdin, stdout=stdout)


class RefusedKey(PipeMachine):
    """ssh itself refuses: exit 255 and ssh's own words."""

    def popen(self, words, *, stdin=subprocess.PIPE, stdout=subprocess.PIPE):
        if "unpack" in words:
            script = ("import sys; sys.stdin.buffer.read(); "
                      "sys.stderr.write('drew@box: Permission denied (publickey).\\n'); sys.exit(255)")
            return subprocess.Popen([sys.executable, "-c", script], stdin=stdin, stdout=stdout,
                                    stderr=subprocess.PIPE)
        return super().popen(words, stdin=stdin, stdout=stdout)


class SmallDisk(PipeMachine):
    """The real unpack, on a machine where a file may hold only four bytes (EFBIG, as a full disk fails a write)."""

    def popen(self, words, *, stdin=subprocess.PIPE, stdout=subprocess.PIPE):
        if "unpack" not in words:
            return super().popen(words, stdin=stdin, stdout=stdout)
        import resource
        import signal

        def limit() -> None:
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (4, 4))

        words = [str(M.AGENT_SOURCE) if word == self.agent_path() else str(word) for word in words]
        words[0] = sys.executable
        return subprocess.Popen(words, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, preexec_fn=limit)


def _ssh(cls, name, tmp_path):
    return cls({"name": name, "kind": "ssh", "host": name, "workspace": str(tmp_path / f"ws-{name}")})


@pytest.fixture
def payload(tmp_path):
    source = tmp_path / "wheel house" / "gpuwm_data-2.8.0-py3-none-any.whl"
    source.parent.mkdir()
    source.write_bytes(b"fourteen bytes")
    return source


def test_a_relay_from_this_computer_lands_the_file_and_returns(tmp_path, payload):
    here = M.Machine(M.local_row())
    box = _ssh(PipeMachine, "box", tmp_path)
    dest = tmp_path / "landed"
    moved = M.relay(here, box, [str(payload)], str(dest))
    assert (dest / payload.name).read_bytes() == b"fourteen bytes"
    assert moved >= len(b"fourteen bytes")


def test_a_relay_between_two_machines_lands_the_file_and_returns(tmp_path, payload):
    one, two = _ssh(PipeMachine, "one", tmp_path), _ssh(PipeMachine, "two", tmp_path)
    dest = tmp_path / "landed"
    moved = M.relay(one, two, [str(payload)], str(dest))
    assert (dest / payload.name).read_bytes() == b"fourteen bytes"
    assert moved >= len(b"fourteen bytes")


def test_a_relay_from_a_machine_to_this_computer_lands_the_file(tmp_path, payload):
    here = M.Machine(M.local_row())
    one = _ssh(PipeMachine, "one", tmp_path)
    dest = tmp_path / "landed"
    assert M.relay(one, here, [str(payload)], str(dest)) == len(b"fourteen bytes")
    assert (dest / payload.name).read_bytes() == b"fourteen bytes"


def _plain(error, *paths):
    """The page's words for ``error``: no traceback, no class name, no path, and not an SSH diagnosis."""

    words = f"{error.message} {error.fix}"
    assert "Traceback" not in words and "Error:" not in words and "[Errno" not in words, words
    assert "SSH key" not in words and "ssh-copy-id" not in words, words
    for path in paths:
        assert str(path) not in words, words
    return words


@pytest.mark.parametrize("from_here", [True, False])
def test_a_landing_folder_under_a_plain_file_is_said_in_plain_words(tmp_path, payload, from_here):
    # The real unpack, asked to save under a plain file: before, its traceback was the page's message.
    source = M.Machine(M.local_row()) if from_here else _ssh(PipeMachine, "one", tmp_path)
    box = _ssh(WatchedMachine, "box", tmp_path)
    box.started = []
    blocker = tmp_path / "workspace is a file"
    blocker.write_bytes(b"x")
    with pytest.raises(M.MachineError) as caught:
        M.relay(source, box, [str(payload)], str(blocker / "install" / "wheels"))
    words = _plain(caught.value, tmp_path)
    assert caught.value.message.startswith("box could not save the files: ")
    assert "Free some space on box or pick a folder it can write" in words
    assert all(proc.returncode is not None for proc in box.started)


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="a folder's mode refuses writes only on POSIX, and not to root")
@pytest.mark.parametrize("from_here", [True, False])
def test_a_folder_the_machine_may_not_write_is_not_called_a_refused_key(tmp_path, payload, from_here):
    source = M.Machine(M.local_row()) if from_here else _ssh(PipeMachine, "one", tmp_path)
    box = _ssh(PipeMachine, "box", tmp_path)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(M.MachineError) as caught:
            M.relay(source, box, [str(payload)], str(locked / "wheels"))
    finally:
        locked.chmod(0o700)
    _plain(caught.value, tmp_path)
    assert caught.value.message == "box could not save the files: Permission denied."


@pytest.mark.skipif(os.name == "nt", reason="RLIMIT_FSIZE is POSIX")
@pytest.mark.parametrize("from_here", [True, False])
def test_a_write_that_fails_part_way_removes_its_part_and_says_why(tmp_path, payload, from_here):
    source = M.Machine(M.local_row()) if from_here else _ssh(PipeMachine, "one", tmp_path)
    box = _ssh(SmallDisk, "box", tmp_path)
    landing = tmp_path / "landing"
    with pytest.raises(M.MachineError) as caught:
        M.relay(source, box, [str(payload)], str(landing))
    _plain(caught.value, tmp_path)
    assert caught.value.message == "box could not save the files: File too large."
    assert list(landing.iterdir()) == []


@pytest.mark.parametrize("from_here", [True, False])
def test_a_receiver_that_crashes_shows_no_traceback(tmp_path, payload, from_here, capsys):
    source = M.Machine(M.local_row()) if from_here else _ssh(PipeMachine, "one", tmp_path)
    broken = _ssh(CrashingReceiver, "box", tmp_path)
    with pytest.raises(M.MachineError) as caught:
        M.relay(source, broken, [str(payload)], str(tmp_path / "never"))
    _plain(caught.value, f"{POSIX_HOME}/someone/secret")
    assert caught.value.message == "box stopped before the files were saved."
    # The raw words go to the server's terminal, for whoever needs them.
    assert "PermissionError" in capsys.readouterr().err


def test_ssh_refusing_the_key_is_still_called_that(tmp_path, payload):
    refused = _ssh(RefusedKey, "box", tmp_path)
    with pytest.raises(M.MachineError) as caught:
        M.relay(M.Machine(M.local_row()), refused, [str(payload)], str(tmp_path / "never"))
    assert "refused this computer's SSH key" in caught.value.message


def test_a_big_copy_into_a_receiver_that_stops_ends_the_sender(tmp_path):
    # Between two machines, once the receiver stops, the rest of the sender's stream was read into memory
    # for nothing; the sender is ended instead, and the receiver's reason is the one given.
    import time

    big = tmp_path / "big.whl"
    big.write_bytes(os.urandom(16 * 1024 * 1024))
    one = _ssh(WatchedMachine, "one", tmp_path)
    one.started = []
    blocker = tmp_path / "plain file"
    blocker.write_bytes(b"x")
    began = time.monotonic()
    with pytest.raises(M.MachineError) as caught:
        M.relay(one, _ssh(PipeMachine, "two", tmp_path), [str(big)], str(blocker / "wheels"))
    assert time.monotonic() - began < 60
    assert caught.value.message.startswith("two could not save the files: ")
    assert [proc.returncode not in (None, 0) for proc in one.started] == [True]


@pytest.mark.parametrize("to_here", [True, False])
def test_a_file_a_machine_cannot_send_is_named_not_skipped(tmp_path, payload, to_here):
    # A machine sending used to skip a missing file and exit 0: a copy that looked finished without it.
    one = _ssh(PipeMachine, "one", tmp_path)
    dest = M.Machine(M.local_row()) if to_here else _ssh(PipeMachine, "two", tmp_path)
    gone = payload.parent / "gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    with pytest.raises(M.MachineError) as caught:
        M.relay(one, dest, [str(payload), str(gone)], str(tmp_path / "landed"))
    _plain(caught.value, tmp_path)
    where = "this computer" if to_here else "two"
    assert caught.value.message == f"{gone.name} could not be read on one, so the copy to {where} stopped."
    assert caught.value.missing == (gone.name,)
    # Nothing was sent: the check comes before the stream.
    assert not (tmp_path / "landed" / payload.name).exists()


class RenderWorker(PipeMachine):
    """Its copies are real agent unpacks; its render verbs answer as a worker that is drawing."""

    def call(self, verb, *args, payload=None, timeout=M.CALL_TIMEOUT_S):
        self.calls.append((verb, payload))
        if verb == "render-list":
            return {"files": [], "manifests": [], "job": {"state": "rendering"}, "alive": True}
        return {"ok": True}


class Machines:
    def __init__(self, *machines):
        self.machines = {machine.name: machine for machine in machines}

    def get(self, name):
        return self.machines[name]


@pytest.mark.parametrize("source_here", [False, True])
def test_a_frame_gone_from_the_source_is_set_aside_and_the_rest_are_drawn(tmp_path, source_here):
    from gpuwm.gui import remote_runs, runs

    frames_at = tmp_path / "forecast there"
    frames_at.mkdir()
    frames = [frames_at / f"wrfout_d01_hour0{hour}" for hour in (1, 2, 3)]
    for frame in frames[0::2]:
        frame.write_bytes(b"a frame")
    # frames[1] was committed and then removed from the source before its turn came.
    source = M.Machine(M.local_row()) if source_here else _ssh(PipeMachine, "one", tmp_path)
    worker = _ssh(RenderWorker, "two", tmp_path)
    worker.calls = []
    rundir = tmp_path / "runs" / "r1"
    rundir.mkdir(parents=True)
    if not source_here:
        (rundir / runs.REMOTE).write_text(json.dumps({"machine": "one", "remote_rundir": str(frames_at),
                                                      "ended": True}), encoding="utf-8")
    else:
        (rundir / runs.PLAN).write_text("{}", encoding="utf-8")
    (rundir / runs.EVENTS).write_text("".join(
        json.dumps({"event": "output_committed", "sequence": n, "path": str(frame)}) + "\n"
        for n, frame in enumerate(frames, 1)), encoding="utf-8")
    remote_runs.request_render(rundir, "two")
    follower = remote_runs.Follower(tmp_path / "runs", "r1", Machines(source, worker))
    follower.tick()
    render = json.loads((rundir / runs.RENDER).read_text(encoding="utf-8"))
    assert render["gone"] == [str(frames[1])]
    assert render["fed"] == [str(frames[0]), str(frames[2])]
    where = "this computer" if source_here else "one"
    assert render["message"] == f"1 frame was gone from {where} before it could be drawn."
    inbox = tmp_path / "ws-two" / "renders" / render["job"] / "inbox"
    assert sorted(p.name for p in inbox.iterdir()) == [frames[0].name, frames[2].name]
    fed = [payload for verb, payload in worker.calls if verb == "render-feed"]
    assert fed == [[frames[0].name, frames[2].name]]
    # The next pass does not stop on the same frame again.
    follower.tick()
    assert [verb for verb, _ in worker.calls].count("render-feed") == 1


def test_every_file_a_machine_does_not_have_is_named_in_one_answer(tmp_path, payload):
    one = _ssh(PipeMachine, "one", tmp_path)
    first, second = (payload.parent / f"wrfout_d01_hour0{hour}" for hour in (1, 2))
    with pytest.raises(M.MachineError) as caught:
        M.relay(one, M.Machine(M.local_row()), [str(first), str(payload), str(second)], str(tmp_path / "landed"))
    assert caught.value.missing == (first.name, second.name)
    assert caught.value.message.startswith(f"{first.name} could not be read on one")


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="a file's mode refuses reads only on POSIX, and not to root")
def test_a_file_a_machine_may_not_read_is_named(tmp_path, payload):
    payload.chmod(0)
    try:
        with pytest.raises(M.MachineError) as caught:
            M.relay(_ssh(PipeMachine, "one", tmp_path), M.Machine(M.local_row()), [str(payload)],
                    str(tmp_path / "landed"))
    finally:
        payload.chmod(0o600)
    assert caught.value.message == f"{payload.name} could not be read on one, so the copy to this computer stopped."


def test_an_answer_from_ssh_is_ssh_and_an_answer_from_the_machine_is_not():
    box = M.Machine({"name": "box", "kind": "ssh", "host": "me@box"})
    traceback = (f"Traceback (most recent call last):\n  File \"{POSIX_HOME}/me/.agent/a.py\", line 9\n"
                 f"PermissionError: [Errno 13] Permission denied: '{POSIX_HOME}/me/ws/runs'\n").encode()
    assert "SSH key" in M.ssh_failure(box, b"me@box: Permission denied (publickey).\n", 255).message
    assert "SSH key" in M.ssh_failure(box, b"me@box: Permission denied (publickey).\n").message
    answer = M.ssh_failure(box, traceback, 1)
    assert answer.message == "box: Permission denied"
    here = M.Machine(M.local_row())
    assert M.ssh_failure(here, traceback, 1).message == f"{M.LOCAL}: Permission denied"


def test_plain_reason_keeps_the_reason_and_drops_the_path():
    assert M.plain_reason(b"mkdir: cannot create directory '/srv/ws/.agent': Permission denied\n") == \
        "Permission denied"
    assert M.plain_reason("mkdir: cannot create directory ‘/srv/ws’: Not a directory") == "Not a directory"
    assert M.plain_reason(b"KeyError: 'files'\n") == "gpuwm's helper stopped with an error"
    assert M.plain_reason(b"mkdir: Not a directory\n") == "Not a directory"
    assert M.plain_reason(b"sh: 1: cannot create /srv/ws/.agent/a.py.part: Permission denied\n") == \
        "Permission denied"
    assert M.plain_reason(b"bash: python3: command not found\n") == "python3: command not found"
    assert M.plain_reason(b"") == "no answer"


class LocalShell(M.Machine):
    """An SSH machine whose ssh runs the remote command in this computer's sh."""

    def ssh_argv(self, remote_command):
        return ["sh", "-c", remote_command]


@pytest.mark.skipif(shutil.which("sh") is None or os.name == "nt", reason="needs a POSIX sh")
def test_a_workspace_the_helper_cannot_be_kept_in_is_said_plainly(tmp_path):
    blocker = tmp_path / "plain file"
    blocker.write_bytes(b"x")
    machine = LocalShell({"name": "box", "kind": "ssh", "host": "box", "workspace": str(blocker / "ws")})
    with pytest.raises(M.MachineError) as caught:
        machine.ensure_agent()
    _plain(caught.value, tmp_path)
    assert caught.value.message == "box could not keep gpuwm's helper in its workspace (Not a directory)."


class WatchedMachine(PipeMachine):
    """Keeps the receivers it starts, so a test can see that each one was reaped."""

    started: list = []

    def popen(self, words, *, stdin=subprocess.PIPE, stdout=subprocess.PIPE):
        proc = super().popen(words, stdin=stdin, stdout=stdout)
        self.started.append(proc)
        return proc


def test_a_file_this_computer_cannot_read_is_named_not_passed_off_as_a_finished_copy(tmp_path, payload):
    # A tar that stops at a file boundary reads as complete to the receiver, so a vanished second file
    # must not come back as a copy that succeeded with one file.
    here = M.Machine(M.local_row())
    box = _ssh(WatchedMachine, "box", tmp_path)
    box.started = []
    gone = payload.parent / "gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    with pytest.raises(M.MachineError) as caught:
        M.relay(here, box, [str(payload), str(gone)], str(tmp_path / "landed"))
    assert gone.name in caught.value.message and "could not be read on this computer" in caught.value.message
    assert str(payload.parent) not in caught.value.message
    assert caught.value.missing == (gone.name,)
    # Nothing was sent: the check comes before the stream, as it does on a machine.
    assert box.started == []


def test_a_file_that_goes_while_this_computer_sends_ends_the_receiver_and_is_named(tmp_path, payload, monkeypatch):
    here = M.Machine(M.local_row())
    box = _ssh(WatchedMachine, "box", tmp_path)
    box.started = []
    second = payload.parent / "gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    second.write_bytes(b"there at the check")
    add = M.tarfile.TarFile.add

    def vanishing(archive, name, *args, **kwargs):
        if Path(name).name == second.name:
            raise FileNotFoundError(2, "No such file or directory", name)
        return add(archive, name, *args, **kwargs)

    monkeypatch.setattr(M.tarfile.TarFile, "add", vanishing)
    with pytest.raises(M.MachineError) as caught:
        M.relay(here, box, [str(payload), str(second)], str(tmp_path / "landed"))
    assert caught.value.message == f"{second.name} could not be read on this computer, so the copy to box stopped."
    assert caught.value.missing == (second.name,)
    assert [proc.returncode is not None for proc in box.started] == [True]


def test_files_this_computer_cannot_save_are_a_plain_error_and_the_sender_is_reaped(tmp_path, payload):
    # The landing folder sits under a plain file, so nothing can be written there.
    here = M.Machine(M.local_row())
    one = _ssh(WatchedMachine, "one", tmp_path)
    one.started = []
    blocker = tmp_path / "not a folder"
    blocker.write_bytes(b"x")
    with pytest.raises(M.MachineError) as caught:
        M.relay(one, here, [str(payload)], str(blocker / "landed"))
    assert caught.value.message.startswith("This computer could not save the files from one:")
    assert str(tmp_path) not in caught.value.message
    assert [proc.returncode is not None for proc in one.started] == [True]


# ------------------------------------------------------------------ wheel names

@pytest.mark.parametrize("folder", [
    WHEELHOUSE,
    r"D:\wheels",
    POSIX_HOME + "/rené/wheel house",
    "C:/Us" + "ers/someone/Downloads/wheels",
])
def test_the_wheels_are_found_by_file_name_whichever_separator_the_folder_uses(folder):
    sep = "\\" if "\\" in folder else "/"
    listing = [f"{folder}{sep}{name}" for name in (
        "gpuwm-2.7.9-cp312-cp312-manylinux_2_28_x86_64.whl",
        "gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "gpuwm-2.8.0-cp312-cp312-win_amd64.whl",
        "gpuwm_data-2.8.0-py3-none-any.whl",
        "gpuwm_data-2.7.9-py3-none-any.whl")]
    found = M.find_wheels(listing, "2.8.0")
    # The full paths come back: the copy reads them.
    assert found == [f"{folder}{sep}gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                     f"{folder}{sep}gpuwm_data-2.8.0-py3-none-any.whl"]
    assert [M.wheel_name(path) for path in found] == ["gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
                                                       "gpuwm_data-2.8.0-py3-none-any.whl"]


def test_a_windows_folder_without_the_linux_wheel_is_still_refused():
    with pytest.raises(M.MachineError, match="No gpuwm 2.8.0 Linux wheel"):
        M.find_wheels([r"C:\w\gpuwm-2.8.0-cp312-cp312-win_amd64.whl",
                       r"C:\w\gpuwm_data-2.8.0-py3-none-any.whl"], "2.8.0")


def test_wheels_found_in_a_real_folder_with_spaces_and_accents(tmp_path):
    folder = tmp_path / "Wheel house René"
    folder.mkdir()
    for name in ("gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl", "gpuwm_data-2.8.0-py3-none-any.whl"):
        (folder / name).write_bytes(b"w")
    listing = [str(path) for path in folder.glob("*.whl")]
    found = M.find_wheels(listing, "2.8.0")
    assert all(Path(path).is_file() for path in found)


class ShellMachine(PipeMachine):
    """An SSH machine whose commands run in this computer's sh, as its login shell would run them."""

    def run(self, words, *, data=b"", timeout=M.CALL_TIMEOUT_S):
        return subprocess.run(["sh", "-c", self.command(words)], input=data, capture_output=True, timeout=timeout)


class TwoMachines(M.Registry):
    def __init__(self, tmp_path, machines):
        super().__init__(tmp_path / "machines.toml")
        self.named = machines

    def get(self, name):
        return self.named.get(name) or super().get(name)

    def probe(self, name, *, fresh=False):
        return {"install_extra": "gpu-cu13"}


@pytest.mark.skipif(shutil.which("sh") is None or os.name == "nt", reason="needs a POSIX sh")
def test_a_wheel_folder_on_another_machine_with_a_space_in_its_name_is_found(tmp_path, monkeypatch):
    # The folder is listed over the machine's shell; splitting that listing on spaces broke the path in two.
    monkeypatch.setattr(M, "local_version", lambda: "2.8.0")
    folder = tmp_path / "wheel house"
    folder.mkdir()
    names = ("gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl", "gpuwm_data-2.8.0-py3-none-any.whl")
    for name in names:
        (folder / name).write_bytes(b"w")
    registry = TwoMachines(tmp_path, {"store": _ssh(ShellMachine, "store", tmp_path),
                                      "box": _ssh(ShellMachine, "box", tmp_path)})
    plan = M.install(registry, "box", wheelhouse=str(folder), from_machine="store", dry=True)
    assert plan["wheels"] == [str(folder / name) for name in names]
    assert plan["from"] == "store"


# ------------------------------------------------------------------ the upgrade line

def test_the_windows_upgrade_line_runs_the_interpreter_with_the_call_operator(monkeypatch):
    from gpuwm import update_cli

    monkeypatch.setattr(update_cli.os, "name", "nt")
    monkeypatch.setattr(update_cli.sys, "executable",
                        WINDOWS_USERS + r"\O'Neil\space venv\Scripts\python.exe")
    assert update_cli.upgrade_command("gpuwm") == \
        "& '" + WINDOWS_USERS + r"\O''Neil\space venv\Scripts\python.exe' -m pip install --upgrade gpuwm"


def test_the_posix_upgrade_line_is_unchanged(monkeypatch):
    from gpuwm import update_cli

    monkeypatch.setattr(update_cli.os, "name", "posix")
    monkeypatch.setattr(update_cli.sys, "executable", POSIX_HOME + "/me/space venv/bin/python")
    assert update_cli.upgrade_command("gpuwm") == \
        "'" + POSIX_HOME + "/me/space venv/bin/python' -m pip install --upgrade gpuwm"


@pytest.mark.skipif(os.name != "nt", reason="PowerShell runs the Windows line")
def test_the_windows_upgrade_line_runs_in_powershell(tmp_path, monkeypatch):
    from gpuwm import update_cli

    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is not installed")
    # A real interpreter at a path with a space and a quote; `pip --version` stands in for the upgrade.
    folder = tmp_path / "O'Neil space"
    folder.mkdir()
    python = folder / "python.exe"
    shutil.copyfile(sys.executable, python)
    for dll in Path(sys.executable).parent.glob("python*.dll"):
        shutil.copyfile(dll, folder / dll.name)
    monkeypatch.setattr(update_cli.sys, "executable", str(python))
    line = update_cli.upgrade_command("gpuwm").replace("pip install --upgrade gpuwm", "pip --version")
    env = {**os.environ, "PYTHONHOME": str(Path(sys.base_prefix))}
    done = subprocess.run([shell, "-NoProfile", "-Command", line], capture_output=True, text=True, timeout=120,
                          env=env)
    assert "Unexpected token" not in done.stderr
    assert done.returncode == 0 and done.stdout.startswith("pip "), done.stdout + done.stderr


# ------------------------------------------------------------------ engine children

@pytest.fixture
def shadow(tmp_path):
    """A folder holding a gpuwm of another version, as an older checkout in the terminal's folder would."""

    folder = tmp_path / "old checkout"
    (folder / "gpuwm").mkdir(parents=True)
    (folder / "gpuwm" / "__init__.py").write_text("__version__ = '2.7.7-shadow'\n", encoding="utf-8")
    (folder / "gpuwm" / "__main__.py").write_text(
        "import json, sys\nprint(json.dumps({'shadow': True}))\n", encoding="utf-8")
    return folder


def test_an_engine_child_started_from_an_old_checkout_imports_the_serving_gpuwm(shadow):
    import gpuwm
    from gpuwm.gui.jobs import engine_argv, engine_env

    argv = engine_argv()
    assert argv[:4] == [sys.executable, "-P", "-m", "gpuwm"]
    probe = [*argv[:2], "-c", "import gpuwm, json; print(json.dumps(gpuwm.__file__))"]
    done = subprocess.run(probe, cwd=shadow, capture_output=True, text=True, timeout=120,
                          env={**os.environ, **engine_env()})
    assert done.returncode == 0, done.stderr
    assert Path(json.loads(done.stdout)).resolve() == Path(gpuwm.__file__).resolve()
    # Without -P and the serving root, the same folder answers with the old checkout.
    old = subprocess.run([sys.executable, "-c", "import gpuwm; print(gpuwm.__version__)"], cwd=shadow,
                         capture_output=True, text=True, timeout=120)
    assert old.stdout.strip() == "2.7.7-shadow"


def test_the_runner_query_reaches_the_serving_gpuwm_from_an_old_checkout(shadow, monkeypatch):
    from gpuwm.gui.jobs import Runner, engine_argv

    import gpuwm

    monkeypatch.chdir(shadow)
    argv = engine_argv()
    answer = Runner().query([*argv[:2], "-c", "import gpuwm, json; print(json.dumps({'file': gpuwm.__file__}))"])
    assert Path(answer["file"]).resolve() == Path(gpuwm.__file__).resolve()


def test_a_wheel_install_needs_no_path_and_a_source_tree_names_its_root(monkeypatch):
    from gpuwm import runtime_manifest
    from gpuwm.gui.jobs import engine_env

    monkeypatch.setattr(runtime_manifest, "installed_distribution", lambda package="gpuwm": object())
    assert engine_env() == {"PYTHONSAFEPATH": "1"}
    monkeypatch.setattr(runtime_manifest, "installed_distribution", lambda package="gpuwm": None)
    monkeypatch.setenv("PYTHONPATH", "elsewhere")
    root = str(Path(runtime_manifest.__file__).resolve().parent.parent)
    assert engine_env() == {"PYTHONSAFEPATH": "1", "PYTHONPATH": os.pathsep.join([root, "elsewhere"])}


def test_a_python_the_query_child_starts_in_turn_also_imports_the_serving_gpuwm(shadow, monkeypatch):
    # A query runs in the server's own folder; a grandchild started without -P used to import the checkout there.
    from gpuwm.gui.jobs import Runner, engine_argv

    import gpuwm

    monkeypatch.chdir(shadow)
    grandchild = ("import json, subprocess, sys; "
                  "out = subprocess.run([sys.executable, '-c', 'import gpuwm; print(gpuwm.__file__)'], "
                  "capture_output=True, text=True).stdout.strip(); print(json.dumps({'file': out}))")
    answer = Runner().query([*engine_argv()[:2], "-c", grandchild])
    assert answer["file"] and Path(answer["file"]).resolve() == Path(gpuwm.__file__).resolve()


def test_the_page_still_shows_the_plain_command():
    from gpuwm.gui.jobs import engine_argv, plain_command

    assert plain_command(engine_argv("run-plan", "--sources")) == "gpuwm run-plan --sources"


def test_the_copied_line_keeps_minus_p_for_a_wheel_and_drops_it_for_a_source_tree(monkeypatch):
    from gpuwm import runtime_manifest
    from gpuwm.gui.jobs import display, engine_argv

    monkeypatch.setattr(runtime_manifest, "installed_distribution", lambda package="gpuwm": object())
    assert " -P -m gpuwm run-plan " in display(engine_argv("run-plan", "p.json")) + " "
    monkeypatch.setattr(runtime_manifest, "installed_distribution", lambda package="gpuwm": None)
    line = display(engine_argv("run-plan", "p.json"))
    assert " -P " not in line and " -m gpuwm run-plan p.json" in line
    # Only engine lines change.
    assert display(["aws", "-P", "-m", "gpuwm"]) == "aws -P -m gpuwm"


@pytest.mark.skipif(os.name == "nt", reason="the line is split as a POSIX shell splits it")
def test_the_copied_line_runs_in_a_source_tree_without_pythonpath(tmp_path, monkeypatch):
    # A source checkout started from its own folder: the copied line, pasted there, reaches that gpuwm.
    #
    # Which gpuwm answered is read from the module's own path, never from whether some gpuwm answered: in a
    # venv with gpuwm installed (the editable install the release battery makes) every line imports a gpuwm,
    # and a line that reached the installed copy instead of the tree it was pasted in looked the same.
    import shlex

    from gpuwm import runtime_manifest
    from gpuwm.gui.jobs import display, engine_argv

    monkeypatch.setattr(runtime_manifest, "installed_distribution", lambda package="gpuwm": None)
    # A terminal the line is pasted into: neither the page's PYTHONPATH nor its safe-path setting.
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    argv = shlex.split(display(engine_argv("--help")))
    # This checkout: the line runs its command.
    done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=300, env=env)
    assert "No module named gpuwm" not in done.stderr
    assert done.returncode == 0, done.stderr[-2000:]
    # A checkout no gpuwm install points at: the line imports that checkout's gpuwm and hands it its arguments.
    tree = tmp_path / "checkout"
    (tree / "gpuwm").mkdir(parents=True)
    (tree / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")
    (tree / "gpuwm" / "__main__.py").write_text(
        "import json, sys\nimport gpuwm\nprint(json.dumps({'file': gpuwm.__file__, 'args': sys.argv[1:]}))\n",
        encoding="utf-8")
    here = subprocess.run(argv, cwd=tree, capture_output=True, text=True, timeout=300, env=env)
    assert here.returncode == 0, here.stderr[-2000:]
    answer = json.loads(here.stdout)
    assert Path(answer["file"]).resolve() == (tree / "gpuwm" / "__init__.py").resolve()
    assert answer["args"] == ["--help"]
    # The same line with -P, as the page copied it before, never reaches that checkout: with no gpuwm installed
    # it finds none, and with one installed it runs that one.
    exact = subprocess.run([argv[0], "-P", *argv[1:]], cwd=tree, capture_output=True, text=True, timeout=300,
                           env=env)
    assert all(str(folder / "gpuwm") not in exact.stdout + exact.stderr for folder in (tree, tree.resolve()))
    assert "No module named gpuwm" in exact.stderr or exact.returncode == 0, exact.stderr[-2000:]


# ------------------------------------------------------------------ an install that stopped

def _agent():
    import importlib.util

    spec = importlib.util.spec_from_file_location("machine_agent_under_test", M.AGENT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(workspace, **fields):
    folder = workspace / "install"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "install.json").write_text(json.dumps(fields), encoding="utf-8")


def test_an_install_whose_process_is_gone_reads_failed_not_installing_for_ever(tmp_path):
    agent = _agent()
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    wheels = ["/w/gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl"]
    _record(tmp_path, state="installing", pid=ended.pid, started_utc=agent.utc(), wheels=wheels)
    record = agent.install_record(tmp_path)
    assert record["state"] == "failed" and record["stopped"] is True
    assert M.install_words(record) == "the install of gpuwm 2.8.0 did not finish"
    _record(tmp_path, state="installing", pid=os.getpid(), started_utc=agent.utc(), wheels=wheels)
    assert agent.install_record(tmp_path)["state"] == "installing"
    assert M.install_words(agent.install_record(tmp_path)) == "installing gpuwm 2.8.0"
    # Between the record and its pid, a moment; a record that never got one reads failed after that.
    _record(tmp_path, state="installing", started_utc=agent.utc())
    assert agent.install_record(tmp_path)["state"] == "installing"
    _record(tmp_path, state="installing", started_utc="2026-01-01T00:00:00Z")
    assert agent.install_record(tmp_path)["state"] == "failed"
    _record(tmp_path, state="installed", pid=ended.pid, wheels=wheels)
    assert agent.install_record(tmp_path)["state"] == "installed"
    assert M.install_words(agent.install_record(tmp_path)) == ""
    assert agent.install_record(tmp_path / "none") is None


def _wheel(folder: Path, name: str, files: dict[str, str], *, version: str = "2.8.0",
           extras: dict[str, str] | None = None) -> Path:
    """A small pure-Python wheel pip installs for real; ``extras`` maps an extra to the one package it adds."""

    import zipfile

    info = f"{name}-{version}.dist-info"
    lines = ["Metadata-Version: 2.1", f"Name: {name.replace('_', '-')}", f"Version: {version}"]
    for extra, requirement in (extras or {}).items():
        lines += [f"Provides-Extra: {extra}", f'Requires-Dist: {requirement}; extra == "{extra}"']
    members = {**files, f"{info}/METADATA": "\n".join(lines) + "\n",
               f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    members[f"{info}/RECORD"] = "".join(f"{member},,\n" for member in [*members, f"{info}/RECORD"])
    path = folder / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for member, text in members.items():
            archive.writestr(member, text)
    return path


def _agent_install(workspace: Path, wheels: list[Path], extra: str, env: dict) -> dict:
    """The real agent install verb, then its detached install to the end; returns the final record."""

    import time

    done = subprocess.run([sys.executable, str(M.AGENT_SOURCE), "install", "--workspace", str(workspace)],
                          input=json.dumps({"wheels": [str(w) for w in wheels], "extra": extra}),
                          capture_output=True, text=True, timeout=120, env=env)
    assert done.returncode == 0 and json.loads(done.stdout)["ok"], done.stdout + done.stderr
    record = workspace / "install" / "install.json"
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        document = json.loads(record.read_text(encoding="utf-8"))
        if document.get("state") != "installing":
            return document
        time.sleep(1)
    raise AssertionError("the install did not end within ten minutes")


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None or shutil.which("python3") is None,
                    reason="the agent's install runs on a Linux machine, with bash and python3")
@pytest.mark.parametrize("first, second", [("gpu-cu12", "gpu-cu13"), ("gpu-cu13", "gpu-cu12")])
def test_an_install_after_the_driver_changed_cuda_major_leaves_one_cupy(tmp_path, first, second):
    house = tmp_path / "wheel house"
    house.mkdir()
    main = _wheel(house, "gpuwm", {"gpuwm_standin.py": "VERSION = '2.8.0'\n"},
                  extras={"gpu-cu12": "cupy-cuda12x", "gpu-cu13": "cupy-cuda13x"})
    data = _wheel(house, "gpuwm_data", {"gpuwm_data_standin.py": ""})
    # The index the extras resolve from; PIP_FIND_LINKS splits on spaces, so its folder has none.
    index = tmp_path / "index"
    index.mkdir()
    for major in ("12", "13"):
        # Both builds install the same cupy package files, as the real ones do.
        _wheel(index, f"cupy_cuda{major}x", {"cupy/__init__.py": f"MAJOR = {major}\n"}, version="14.2.0")
    # The host's python3 and no uv or python3.14t of its own, which the agent
    # would make the venv with (and install a 3.14t through).
    only = tmp_path / "python3-only"
    only.mkdir()
    (only / "python3").symlink_to(shutil.which("python3"))
    env = {**os.environ, "PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(index), "PIP_DISABLE_PIP_VERSION_CHECK": "1",
           "PATH": f"{only}:/usr/bin:/bin", "HOME": str(tmp_path / "home")}
    workspace = tmp_path / "workspace"
    assert _agent_install(workspace, [main, data], first, env)["state"] == "installed"
    # The driver moved to the other major; the Install button sends that major's extra into the same venv.
    ended = _agent_install(workspace, [main, data], second, env)
    log = (workspace / "install" / "install.log").read_text(encoding="utf-8", errors="replace")
    assert ended["state"] == "installed", log[-3000:]
    python = str(workspace / "venv" / "bin" / "python")
    listed = subprocess.run([python, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True,
                            timeout=120, env=env).stdout.lower().splitlines()
    wanted = f"cupy-cuda{second[-2:]}x"
    assert [line.split("==")[0] for line in listed if line.startswith("cupy")] == [wanted]
    answered = subprocess.run([python, "-c", "import cupy; print(cupy.MAJOR)"], capture_output=True, text=True,
                              timeout=120, env=env)
    assert answered.stdout.strip() == second[-2:], answered.stderr
    assert "removing every CuPy build" in log
    # The same major again removes nothing.
    again = _agent_install(workspace, [main, data], second, env)
    assert again["state"] == "installed"
    assert (workspace / "install" / "install.log").read_text(encoding="utf-8", errors="replace").count(
        "removing every CuPy build") == 1


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                    reason="the agent's install runs on a Linux machine, with bash")
def test_an_install_that_cannot_list_the_venv_stops_before_installing(tmp_path):
    workspace = tmp_path / "workspace"
    stub = workspace / "venv" / "bin" / "python"
    stub.parent.mkdir(parents=True)
    calls = tmp_path / "calls.txt"
    stub.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n'
                    '[ "$1 $2 $3" = "-m pip list" ] && { echo "pip: the environment could not be read" >&2; exit 2; }\n'
                    'exit 0\n')
    stub.chmod(0o755)
    wheel = tmp_path / "gpuwm-2.8.0-py3-none-any.whl"
    wheel.write_bytes(b"stand-in")
    ended = _agent_install(workspace, [wheel], "gpu-cu13", dict(os.environ))
    assert ended["state"] == "failed" and ended["exit_code"] == 2
    lines = calls.read_text().splitlines()
    assert lines[-1].startswith("-m pip list")
    assert not any("[gpu-cu13]" in line for line in lines)


# THE BREAKAGE THESE GUARD: a machine installed from the Machines page made its
# venv with plain python3, a Python with the interpreter lock, so its multi-card
# [devices] forecasts ran about 2x slower than every published benchmark (all on
# free-threaded 3.14t).  The agent now makes the venv on 3.14t when the machine
# has one (or uv can install one), and says by name when it cannot.

def _stub_interpreters(tmp_path, *, ft_install_fails=False, ft_venv_fails=False):
    """python3.14t and python3 stand-ins whose venvs log every call."""

    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    venv_python = (
        "#!/bin/sh\n"
        f'echo "$KIND $*" >> {calls}\n'
        'case "$1" in -c) [ "$KIND" = ft ] && exit 0 || exit 1 ;; esac\n'
        'case "$*" in *"[gpu-cu13]"*) [ "$KIND" = ft ] && [ "$FAIL_FT" = 1 ] && exit 5 ;; esac\n'
        "exit 0\n")
    for name, kind in (("python3.14t", "ft"), ("python3", "locked")):
        program = bin_dir / name
        program.write_text(
            "#!/bin/sh\n"
            f'echo "base-{kind} $*" >> {calls}\n'
            f'case "$1" in -c) [ {kind} = ft ] && exit 0 || exit 1 ;; esac\n'
            '[ "$1 $2" = "-m venv" ] || exit 9\n'
            'mkdir -p "$3/bin"\n'
            f"cat > \"$3/bin/python\" <<'EOF'\n#!/bin/sh\nKIND={kind}\n{venv_python}EOF\n"
            'chmod 755 "$3/bin/python"\n'
            # As a distro python3.14t without its venv package: bin/python is
            # written, then ensurepip is missing and venv stops.
            f'if [ {kind} = ft ] && [ "$FAIL_FT_VENV" = 1 ]; then echo "no ensurepip" >&2; exit 1; fi\n'
            'exit 0\n')
        program.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path / "home"),
           "FAIL_FT": "1" if ft_install_fails else "", "FAIL_FT_VENV": "1" if ft_venv_fails else ""}
    return env, calls


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                    reason="the agent's install runs on a Linux machine, with bash")
def test_the_agent_makes_its_venv_on_a_python314t_the_machine_has(tmp_path):
    env, calls = _stub_interpreters(tmp_path)
    wheel = tmp_path / "gpuwm-2.8.8-py3-none-any.whl"
    wheel.write_bytes(b"stand-in")
    ended = _agent_install(tmp_path / "workspace", [wheel], "gpu-cu13", env)
    log = (tmp_path / "workspace" / "install" / "install.log").read_text(encoding="utf-8", errors="replace")
    assert ended["state"] == "installed", log[-3000:]
    lines = calls.read_text().splitlines()
    assert [line for line in lines if " -m venv " in line] == [
        f"base-ft -m venv {tmp_path / 'workspace' / 'venv'}"]
    assert "making the venv on free-threaded python3.14t" in log
    assert "WARNING" not in log
    # cftime 1.6.6 has no cp314t wheel; --prefer-binary takes 1.6.5's.
    main = [line for line in lines if "[gpu-cu13]" in line]
    assert main and all("--prefer-binary" in line for line in main)


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                    reason="the agent's install runs on a Linux machine, with bash")
def test_the_agent_falls_back_to_python3_by_name(tmp_path):
    env, calls = _stub_interpreters(tmp_path)
    (tmp_path / "stub-bin" / "python3.14t").unlink()
    wheel = tmp_path / "gpuwm-2.8.8-py3-none-any.whl"
    wheel.write_bytes(b"stand-in")
    ended = _agent_install(tmp_path / "workspace", [wheel], "gpu-cu13", env)
    log = (tmp_path / "workspace" / "install" / "install.log").read_text(encoding="utf-8", errors="replace")
    assert ended["state"] == "installed", log[-3000:]
    assert "about 2x slower" in log and "python3.14t" in log
    main = [line for line in calls.read_text().splitlines() if "[gpu-cu13]" in line]
    assert main and not any("--prefer-binary" in line for line in main)


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                    reason="the agent's install runs on a Linux machine, with bash")
def test_a_dependency_failing_under_314t_remakes_the_agent_venv_on_python3(tmp_path):
    env, calls = _stub_interpreters(tmp_path, ft_install_fails=True)
    wheel = tmp_path / "gpuwm-2.8.8-py3-none-any.whl"
    wheel.write_bytes(b"stand-in")
    ended = _agent_install(tmp_path / "workspace", [wheel], "gpu-cu13", env)
    log = (tmp_path / "workspace" / "install" / "install.log").read_text(encoding="utf-8", errors="replace")
    assert ended["state"] == "installed", log[-3000:]
    lines = calls.read_text().splitlines()
    assert [line.split()[0] for line in lines if " -m venv " in line] == ["base-ft", "base-locked"]
    assert [line.split()[0] for line in lines if "[gpu-cu13]" in line] == ["ft", "locked"]
    assert "did not install under 3.14t" in log and "about 2x slower" in log


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                    reason="the agent's install runs on a Linux machine, with bash")
def test_a_python314t_that_cannot_make_a_venv_gives_way_to_python3_and_leaves_no_half_venv(tmp_path):
    # Under `set -e` the failed venv step ended the install, and the bin/python
    # it left made the next install take the venv as already made.
    env, calls = _stub_interpreters(tmp_path, ft_venv_fails=True)
    wheel = tmp_path / "gpuwm-2.8.8-py3-none-any.whl"
    wheel.write_bytes(b"stand-in")
    ended = _agent_install(tmp_path / "workspace", [wheel], "gpu-cu13", env)
    log = (tmp_path / "workspace" / "install" / "install.log").read_text(encoding="utf-8", errors="replace")
    assert ended["state"] == "installed", log[-3000:]
    lines = calls.read_text().splitlines()
    assert [line.split()[0] for line in lines if " -m venv " in line] == ["base-ft", "base-locked"]
    assert [line.split()[0] for line in lines if "[gpu-cu13]" in line] == ["locked"]
    assert "python3.14t could not make the venv" in log and "about 2x slower" in log


def test_an_install_without_a_cuda_extra_touches_no_cupy():
    agent = _agent()
    assert agent.one_cupy_lines("py", "") == []
    assert agent.one_cupy_lines("py", "render") == []
    assert any("cupy-cuda13x" in line for line in agent.one_cupy_lines("py", "gpu-cu13"))


def test_the_row_says_an_install_is_going_or_did_not_finish(monkeypatch):
    class Probed(M.Machine):
        def call(self, verb, *args, payload=None, timeout=M.CALL_TIMEOUT_S):
            return {"hostname": "box", "state": "idle", "detail": "idle", "cards": {"devices": []},
                    "gpuwm": {"version": "2.7.7"},
                    "install": {"state": "failed", "wheels": ["/w/gpuwm-2.8.0-cp312-cp312-linux_x86_64.whl"]}}

    monkeypatch.setattr(M, "local_version", lambda: "2.8.0")
    row = M.check(Probed({"name": "box", "kind": "ssh", "host": "box"}))
    assert row["detail"] == "idle; the install of gpuwm 2.8.0 did not finish"
    assert row["offer"]["action"] == "install"


# ------------------------------------------------------------------ doctor: two CuPy builds

def test_doctor_blocks_on_two_cupy_builds_and_removes_both(monkeypatch):
    from gpuwm import doctor

    monkeypatch.setattr(doctor, "_installed_cupy_wheels", lambda: [("cupy-cuda12x", 12), ("cupy-cuda13x", 13)])
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)

    def never(*_args, **_kwargs):
        raise AssertionError("an import cannot judge two builds over one set of files")

    monkeypatch.setattr(doctor, "_import_probe", never)
    check = doctor._cupy_check()
    assert check.status == "missing" and check.blocking
    assert check.severity == doctor.SEVERITY_BROKEN
    assert "cupy-cuda12x" in check.detail and "cupy-cuda13x" in check.detail
    assert check.action == "pip uninstall -y cupy-cuda12x cupy-cuda13x"
    assert "pip install 'gpuwm[gpu-cu13]'" in check.remedy.splitlines()


def test_doctor_does_not_count_what_pip_left_of_a_half_removed_build(tmp_path, monkeypatch):
    import importlib.metadata

    from gpuwm import doctor

    site = tmp_path / "site-packages"
    for folder, name in (("~upy_cuda12x-14.2.0.dist-info", "cupy-cuda12x"),
                         ("cupy_cuda13x-14.2.0.dist-info", "cupy-cuda13x")):
        (site / folder).mkdir(parents=True)
        (site / folder / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 14.2.0\n",
                                                encoding="utf-8")
    everything = importlib.metadata.distributions
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: everything(path=[str(site)]))
    # The leftover's METADATA still names cupy-cuda12x; pip ignores it and cannot uninstall it by that name.
    assert sorted(d.metadata["Name"] for d in everything(path=[str(site)])) == ["cupy-cuda12x", "cupy-cuda13x"]
    assert doctor._installed_cupy_wheels() == [("cupy-cuda13x", 13)]


def test_doctor_names_both_extras_when_the_driver_major_is_unknown(monkeypatch):
    from gpuwm import doctor

    monkeypatch.setattr(doctor, "_installed_cupy_wheels", lambda: [("cupy", None), ("cupy-cuda12x", 12)])
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: None)
    check = doctor._cupy_check()
    assert check.status == "missing" and check.blocking
    assert "pip install 'gpuwm[gpu-cu12]'" in check.remedy and "pip install 'gpuwm[gpu-cu13]'" in check.remedy


# ------------------------------------------------------------------ the Install box

INSTALL_SCRIPT = r"""
// A small stand-in for the page's document: elements with children, attributes, values and click listeners.
class Node {}
class El extends Node {
  constructor(tag) { super(); this.tag = tag; this.children = []; this.attrs = {}; this.listeners = {};
    this.dataset = {}; this.value = ""; this.hidden = false; this.className = ""; }
  setAttribute(k, v) { this.attrs[k] = v; if (k === "value") this.value = v; }
  getAttribute(k) { return this.attrs[k]; }
  addEventListener(kind, fn) { (this.listeners[kind] ||= []).push(fn); }
  append(...kids) { this.children.push(...kids); }
  replaceChildren(...kids) { this.children = kids; }
  focus() {}
  scrollIntoView() {}
  get textContent() { return this.children.map((c) => c.textContent).join(""); }
  set textContent(t) { this.children = [new Text(t)]; }
  click() { return Promise.all((this.listeners.click || []).map((fn) => fn())); }
}
class Text extends Node { constructor(t) { super(); this.textContent = t; } }
globalThis.Node = Node;
globalThis.document = { createElement: (tag) => new El(tag), createTextNode: (t) => new Text(t) };
// A per-user folder, assembled from pieces for the release scan for machine paths.
const FOLDER = "C:\\Us" + "ers\\René\\Wheel house";
const posts = [];
globalThis.fetch = async (path, opts = {}) => {
  if (path === "/api/session") return { ok: true, json: async () => ({ token_header: "X-T", token: "t" }) };
  const body = JSON.parse(opts.body || "{}");
  posts.push({ path, body });
  const reply = body.dry_run
    ? { ok: true, dry_run: true, wheels: [FOLDER + "\\gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        FOLDER + "\\gpuwm_data-2.8.0-py3-none-any.whl"] }
    : { ok: true, message: "Installing gpuwm 2.8.0 on gpu-box." };
  return { ok: true, json: async () => reply };
};
const all = (el) => [el, ...(el.children || []).flatMap((c) => (c instanceof El ? all(c) : []))];
const words = JSON.parse(process.argv[2]);
const { installBox, installRequest } = await import(process.argv[3]);
const out = {};
const rows = [{ name: "this-computer", kind: "local" }, { name: "gpu-box", kind: "ssh" }, { name: "store", kind: "ssh" }];
const box = installBox(rows[1], rows, words, { folder: "" });
const fields = all(box).filter((e) => e.attrs && e.attrs["data-field"]).map((e) => e.attrs["data-field"]);
out.fields = fields;
out.sources = all(box.from).filter((e) => e.tag === "option").map((e) => e.textContent);
const buttons = all(box).filter((e) => e.tag === "button");
const install = buttons.find((b) => b.textContent === words.install);
const check = buttons.find((b) => b.textContent === words.install_check);
await install.click();
out.empty_posts = posts.length;
out.empty_words = box.textContent;
box.folder.value = FOLDER;
await check.click();
out.check_body = posts.at(-1).body;
out.check_words = box.textContent;
box.from.value = "store";
await install.click();
out.install_path = posts.at(-1).path;
out.install_body = posts.at(-1).body;
out.request_blank = installRequest("gpu-box", "  ", "", words);
console.log(JSON.stringify(out));
"""


def test_install_asks_for_the_wheel_folder_and_sends_it(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    words = json.loads((ROOT / "gpuwm/gui/copy/screens.json").read_text(encoding="utf-8"))["machines"]
    script = tmp_path / "t.mjs"
    script.write_text(INSTALL_SCRIPT, encoding="utf-8")
    module = (ROOT / "gpuwm/gui/static/js/machines.js").resolve().as_uri()
    done = subprocess.run([node, str(script), json.dumps(words), module], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["fields"] == ["wheelhouse", "from_machine"]
    assert out["sources"] == [words["this_computer"], "store"]
    # Empty: nothing is sent, and the page says what it needs.
    assert out["empty_posts"] == 0
    assert words["install_need_folder"] in out["empty_words"]
    # Find the wheels: a dry run with the folder; the page names the wheels, never the folder's path.
    assert out["check_body"] == {"wheelhouse": WHEELHOUSE, "dry_run": True}
    assert "gpuwm-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl" in out["check_words"]
    assert "Wheel house" not in out["check_words"]
    assert out["install_path"] == "/api/machines/gpu-box/install"
    assert out["install_body"] == {"wheelhouse": WHEELHOUSE, "from_machine": "store"}
    assert out["request_blank"] == {"error": words["install_need_folder"]}


def test_the_rows_install_button_opens_the_box_rather_than_posting_an_empty_body():
    source = (ROOT / "gpuwm/gui/static/js/machines.js").read_text(encoding="utf-8")
    assert 'act(r.name, "install")' not in source
    assert "openInstall(r)" in source
