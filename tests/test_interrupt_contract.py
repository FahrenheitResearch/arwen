"""Ctrl-C has one bounded contract, and it is the same everywhere.

The fleet's hostile-input node filed ``gpuwm go`` **ignores SIGINT**: a
run signalled mid-stage went on to write three frames, 159 product
images and a certification capsule, while SIGTERM and SIGKILL both
stopped it.  Measuring the signal disposition on a 5080 node reproduced
the mechanism exactly and it is not in this package: a shell that starts
a job in the BACKGROUND masks SIGINT for it (POSIX job control), and
CPython declines to install its own handler over an inherited
``SIG_IGN``, so no ``KeyboardInterrupt`` is ever raised.

    foreground        SIGINT handler at startup: default_int_handler
    `... &`           SIGINT handler at startup: 1   (SIG_IGN)
    `nohup ... &`     SIGINT handler at startup: 1   (SIG_IGN)

That leaves two real gaps, and this file is the bar under both:

* the masked case was *silent* -- so it is named, once, in a warning;
* the unmasked case had no contract at all -- an interrupt escaped as a
  ``KeyboardInterrupt`` traceback -- so it now returns 130, names the
  stage, and signals nothing.

The contract matches the multi-GPU orchestration lane verbatim: Ctrl-C
returns 130 and does not kill children or unrelated processes;
unobserved child pids are only printed.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading

import pytest

import gpuwm.cli as cli
from gpuwm import go_cli
from test_go_chain import _PINNED_FREE_BYTES, _emit, _staged_geog_tree


@pytest.fixture(scope="module")
def gfs_config(tmp_path_factory):
    """The same wizard-authored single-domain GFS config the chain
    tests use, emitted by the real wizard."""

    return _emit(tmp_path_factory.mktemp("interrupt"), "myarea")


@pytest.fixture(scope="module")
def staged_geog(tmp_path_factory):
    """A staged WPS_GEOG tree, because this file tests what `go` does
    once it is RUNNING.

    `go` refuses before the fetch stage when the geography tree its
    prepare stage needs is absent -- the check `gpuwm doctor` performs
    and `go` used to skip, which is why an unstaged tree could cost a
    download and then die on a bare truncated path.  That refusal is
    correct and it is not what this file is about: reaching an
    interrupt means reaching a stage, and reaching a stage means having
    the inputs a run needs.  So these tests bring one, exactly as the
    chain tests do.

    The negative control lives beside them
    (test_the_geography_refusal_still_fires_when_it_should): without
    this tree the refusal fires and no stage runs, which is what stops
    this fixture from being a way to switch the check off.
    """

    return _staged_geog_tree(tmp_path_factory.mktemp("geog"))


@pytest.fixture(autouse=True)
def _a_card_whose_free_vram_this_file_decides(monkeypatch):
    """Pin free VRAM so the pre-fetch memory gate cannot decide this
    file's verdicts (same reasoning as tests/test_go_chain.py).

    THE SEAM IS ``device_memory_probe_subprocess``, and it has to be.
    This pin used to sit on ``cupy.cuda.runtime.memGetInfo``, which the
    gate stopped reading when it moved both device questions into a
    short-lived subprocess (an in-process ``memGetInfo`` stands up a CUDA
    primary context, and the `go` process outlives its gate as the
    chain's stage runner -- 0.486 GiB held for the whole run, measured).
    A pin on a seam nobody reads is not a pin: with a real card in the
    box the gate went on measuring it, and this file's negative control
    -- which brings no ``subprocess.Popen`` stand-in of its own -- died
    on a genuine memory refusal instead of reaching the geography
    refusal it exists to prove.  tests/test_go_chain.py's fixture, whose
    reasoning this one cites, already moved; this is that move.

    The two interrupt tests above never saw it because they replace
    ``subprocess.Popen`` wholesale, which incidentally blinds the gate's
    probe too.  Pinned here, all three read the same card on every box.
    """

    from gpuwm.core import preflight

    monkeypatch.setattr(
        preflight, "device_memory_probe_subprocess",
        lambda **_kwargs: {"free_bytes": _PINNED_FREE_BYTES,
                           "total_bytes": 32 * 1024 ** 3,
                           "profile": None})


def test_the_pinned_card_is_what_the_gate_reads(gfs_config, tmp_path):
    """Non-vacuity for the fixture above, and the guard against a repeat.

    The fixture is what keeps every verdict in this file the file's own.
    When its pin drifted off the seam the gate reads, nothing said so --
    the two tests that patch ``subprocess.Popen`` stayed green and only
    the negative control turned red, on boxes whose card is smaller than
    this config's envelope.  This asserts the pin arrives.
    """

    plan = go_cli.plan_from_config(gfs_config, outdir=tmp_path / "gate")
    gate = go_cli.memory_gate(plan)
    assert gate["free_bytes"] == _PINNED_FREE_BYTES
    assert not gate["refuse"]


# --------------------------------------------------------------------
# The masked case: say so, do not fight the shell for it
# --------------------------------------------------------------------

def test_an_ignored_sigint_is_named_once_on_a_long_command(monkeypatch,
                                                           capsys):
    monkeypatch.setattr(signal, "getsignal", lambda num: signal.SIG_IGN)
    cli._warn_if_interrupt_is_ignored("go")
    err = capsys.readouterr().err
    assert err.count("warning:") == 1
    assert "Ctrl-C" in err and "SIGTERM" in err


def test_nothing_is_said_when_sigint_is_live(monkeypatch, capsys):
    monkeypatch.setattr(signal, "getsignal",
                        lambda num: signal.default_int_handler)
    cli._warn_if_interrupt_is_ignored("go")
    assert capsys.readouterr().err == ""


def test_a_short_command_is_not_lectured_about_signals(monkeypatch, capsys):
    monkeypatch.setattr(signal, "getsignal", lambda num: signal.SIG_IGN)
    cli._warn_if_interrupt_is_ignored("check")
    assert capsys.readouterr().err == ""


def test_the_disposition_is_left_exactly_as_the_shell_set_it(monkeypatch):
    """Re-installing a handler would defeat POSIX job control."""

    installed: list = []
    monkeypatch.setattr(signal, "getsignal", lambda num: signal.SIG_IGN)
    monkeypatch.setattr(signal, "signal",
                        lambda num, handler: installed.append((num, handler)))
    cli._warn_if_interrupt_is_ignored("go")
    assert installed == []


# --------------------------------------------------------------------
# The unmasked case: 130, one sentence, and nothing signalled
# --------------------------------------------------------------------

def test_every_command_returns_130_rather_than_a_traceback(monkeypatch,
                                                           capsys):
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_dispatch", interrupted)
    rc = cli.main(["doctor"])
    assert rc == 130
    err = capsys.readouterr().err
    assert "interrupted (Ctrl-C)" in err
    assert "Traceback" not in err


def test_a_stage_interrupt_carries_the_stage_and_the_pid(monkeypatch):
    """``_run_stage``'s wait loop is where a real Ctrl-C lands."""

    real_join = threading.Thread.join
    seen: list = []

    def join_then_interrupt(self, timeout=None):
        # Only the MAIN thread's join on the stage worker: on Windows
        # Popen.communicate joins its own pipe-reader threads, and a
        # blanket patch would fire there instead of in the wait loop.
        if (self.name.startswith("go-stage-") and not seen
                and threading.current_thread() is threading.main_thread()):
            real_join(self, 0.5)      # let the child start and publish a pid
            seen.append(True)
            raise KeyboardInterrupt
        return real_join(self, timeout)

    monkeypatch.setattr(threading.Thread, "join", join_then_interrupt)

    with pytest.raises(go_cli.GoInterrupted) as caught:
        go_cli._run_stage(
            "prepare",
            [sys.executable, "-c", "import time; time.sleep(3)"],
            explain=False)
    assert caught.value.label == "prepare"
    assert isinstance(caught.value.pid, int) and caught.value.pid > 0


def test_the_interrupt_path_signals_no_pid_at_all(monkeypatch):
    """The whole point of naming the pid is that we do not act on it."""

    killed: list = []
    monkeypatch.setattr(go_cli.subprocess.Popen, "kill",
                        lambda self: killed.append(self.pid), raising=False)
    monkeypatch.setattr(go_cli.subprocess.Popen, "terminate",
                        lambda self: killed.append(self.pid), raising=False)

    real_join = threading.Thread.join
    fired: list = []

    def join_then_interrupt(self, timeout=None):
        if (self.name.startswith("go-stage-") and not fired
                and threading.current_thread() is threading.main_thread()):
            real_join(self, 0.5)
            fired.append(True)
            raise KeyboardInterrupt
        return real_join(self, timeout)

    monkeypatch.setattr(threading.Thread, "join", join_then_interrupt)
    with pytest.raises(go_cli.GoInterrupted):
        go_cli._run_stage(
            "forecast",
            [sys.executable, "-c", "import time; time.sleep(3)"],
            explain=False)
    assert killed == []


def test_go_reports_the_interrupt_in_one_sentence_and_exits_130(
        tmp_path, capsys, monkeypatch, gfs_config, staged_geog):
    """End to end through the CLI, with the stage interrupted for real."""

    def interrupt_at_fetch(command, **kwargs):
        if "gpuwm.fetch" in " ".join(command) or "fetch" in command:
            raise go_cli.GoInterrupted("fetch", 4242)

        class _Ok:
            returncode = 0
            pid = 99

            def communicate(self, *args, **kwargs):
                return "", ""

            # subprocess.run context-manages its Popen (the memory gate
            # asks the card through subprocess.run since the first-run
            # staging work), and the real Popen is a context manager, so
            # the stand-in must be one too.
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def wait(self, *args, **kwargs):
                return 0

            def poll(self):
                return 0

            stdout = None
            stderr = None
            stdin = None
            args = ()

            def kill(self):
                pass

        return _Ok()

    monkeypatch.setattr(subprocess, "Popen", interrupt_at_fetch)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    rc = cli.main(["go", str(gfs_config), "--outdir", str(tmp_path / "go"),
                   "--geog-root", str(staged_geog)])
    captured = capsys.readouterr()
    message = captured.out + captured.err
    assert rc == go_cli.INTERRUPT_EXIT_CODE == 130
    assert "interrupted during fetch" in message
    assert "4242" in message                    # the pid is named
    assert "killed nothing" in message          # and only named
    assert "certification capsule" in message   # what is on disk
    assert "Traceback" not in message


def test_the_mechanism_half_of_the_interrupt_message_is_behind_explain(
        tmp_path, capsys, monkeypatch, gfs_config, staged_geog):
    def interrupt_now(command, **kwargs):
        raise go_cli.GoInterrupted("authority", 7)

    monkeypatch.setattr(subprocess, "Popen", interrupt_now)
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    cli.main(["go", str(gfs_config), "--outdir", str(tmp_path / "plain"),
              "--geog-root", str(staged_geog)])
    plain = capsys.readouterr()
    plain_text = plain.out + plain.err

    cli.main(["go", str(gfs_config), "--explain",
              "--geog-root", str(staged_geog),
              "--outdir", str(tmp_path / "full")])
    full = capsys.readouterr()
    full_text = full.out + full.err

    assert "foreground process group" not in plain_text
    assert "foreground process group" in full_text
    assert "[[explain]]" not in plain_text and "[[explain]]" not in full_text


def test_the_geography_refusal_still_fires_when_it_should(
        tmp_path, capsys, monkeypatch, gfs_config):
    """The negative control for this file's `staged_geog` fixture.

    Those two tests above pass a staged geography tree so they can reach
    the stage an interrupt lands in.  A fixture that exists to get past
    a refusal is one edit away from being a way to switch the refusal
    off, and the only thing that keeps the two apart is a test that
    removes the fixture and checks the refusal is still there.

    So: same config, same everything, no tree.  `go` must refuse before
    any stage runs -- because the point of that check is to refuse on
    THIS side of the download rather than after it.
    """

    ran: list = []
    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda label, command, **kw: ran.append(label))
    monkeypatch.setattr(go_cli, "resolve_bridge",
                        lambda: tmp_path / "gfs_grib2_bridge")

    absent = tmp_path / "never-fetched" / "WPS_GEOG"
    rc = cli.main(["go", str(gfs_config), "--outdir", str(tmp_path / "nogeog"),
                   "--geog-root", str(absent)])
    assert rc == 2
    assert ran == [], "a stage ran despite the geography tree being absent"
    message = capsys.readouterr().err
    assert "gpuwm fetch-geog" in message
    assert str(absent) in message
