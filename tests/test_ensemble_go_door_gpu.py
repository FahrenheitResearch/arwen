"""The real ``gpuwm go`` door with an ensemble, end to end on a card.

``gpuwm go CONFIG --members N`` runs the whole chain a user types: fetch,
prepare, then the members inside this process under the ensemble session,
with go's own stage observer attached. That is the one route where a stage
observer and the member runners meet, and it had no test that ran it: every
door test mocked the session. The run ended at its first forecast step with
``TypeError: 'GoChainEvents' object is not callable``, after the fetch and
the preparation, and nothing caught it.

The same route is where Ctrl-C used to be held until every member in flight
had finished: the members run in this process, so no supervisor can end
them, and the stop has to reach them at a step boundary.

Nothing is mocked here. The tests need a CUDA card, the provider network
(a small GFS window, cached under ``GPUWM_GO_DOOR_OUTDIR`` when that is set)
and the WPS geography tree, and they are gated on ``GPUWM_NETWORK_TESTS=1``
like the recipe door's own real-door file.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import pytest

from conftest import requires_gpu

REPO = Path(__file__).resolve().parents[1]
#: A fixed archived cycle, so every run asks for the same bytes.
CYCLE = os.environ.get("GPUWM_GO_DOOR_CYCLE", "2026-10-02T18")

pytestmark = [pytest.mark.gpu, pytest.mark.network, pytest.mark.slow, requires_gpu,
              pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                                 reason="live source fetch; set GPUWM_NETWORK_TESTS=1")]


def _environment():
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))


def _gpuwm(*argv, timeout):
    return subprocess.run([sys.executable, "-m", "gpuwm", *argv], cwd=REPO, env=_environment(),
                          capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def _tiny_config(tmp_path):
    from gpuwm.geog_assets import default_geog_root
    if not Path(default_geog_root()).is_dir():
        pytest.skip(f"no WPS geography tree at {default_geog_root()}")
    config = tmp_path / "godoor.toml"
    made = _gpuwm("domain", "--point", "35.3,-97.5", "--point-extent-km", "150", "--root-dx", "3",
                  "--card", "32gb", "--hours", "1", "--source", "gfs", "--cycle", CYCLE,
                  "--name", "godoor", "--out", str(config), timeout=600)
    assert made.returncode == 0, made.stdout + made.stderr
    return config, Path(os.environ.get("GPUWM_GO_DOOR_OUTDIR") or tmp_path / "out")


def _run_folders(case_root):
    return set(case_root.glob("run-*")) if case_root.is_dir() else set()


def _go(config, case_root, *flags):
    before = _run_folders(case_root)
    ran = _gpuwm("go", str(config), *flags, "--outdir", str(case_root), timeout=3000)
    return ran, _run_folders(case_root) - before


def _hosted_by_go(ran):
    """Did go's own chain host the members (and not a member-source door)?"""
    return "  .. forecast (in process)" in ran.stdout


def _check_completed(ran, run, members):
    text = ran.stdout + ran.stderr
    assert ran.returncode == 0, text[-6000:]
    assert "Traceback" not in text and "is not callable" not in text, text[-6000:]
    manifest = json.loads((run / "run" / "ensemble-run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["request"]["members"] == members
    assert manifest["members_completed"] == list(range(members))
    # The forecast stage ran in this process, under go's own stage stream.
    assert _hosted_by_go(ran) and "  ok      forecast " in ran.stdout
    from gpuwm.chain_events import CHAIN_EVENTS_FILENAME, read_chain_events
    events = read_chain_events(run / CHAIN_EVENTS_FILENAME)
    assert events[-1]["event"] == "completed"
    assert "forecast" in [row["stage"] for row in events[-1]["stages"]]
    # Each member's progress reached the terminal although no host was attached.
    for member in range(members):
        assert f"ensemble: member {member} started" in ran.stderr, ran.stderr[-3000:]
        assert f"ensemble: member {member} finished 3600 model seconds" in ran.stderr, ran.stderr[-3000:]
    # The closing line names where the pictures are, and that folder exists.
    maps = run / "run" / "maps"
    assert maps.is_dir() and any(maps.glob("d01/ens_*/*/*.png"))
    rendered = [line for line in ran.stdout.splitlines() if line.startswith("go: rendered ")]
    assert rendered == [f"go: rendered {maps}"], rendered
    assert not (run / "png").exists()
    # No map lost part of its subtitle row: a postage stamp used to drop its valid time.
    assert "does not fit the plot width" not in text, text[-3000:]
    assert any(maps.glob("d01/ens_postage_*/*/*.png"))
    return manifest


def test_go_with_one_ensemble_member_runs_from_the_front_door(tmp_path):
    """The crash class: the forecast stage hosted under a session with go's stage observer."""
    config, case_root = _tiny_config(tmp_path)
    ran, runs = _go(config, case_root, "--members", "1", "--keep-member-files")
    assert len(runs) == 1, (ran.stdout + ran.stderr)[-6000:]
    (run,) = runs
    manifest = _check_completed(ran, run, 1)
    assert len(manifest["member_history_files"]) >= 2
    assert all((run / "run" / relative).is_file() for relative in manifest["member_history_files"])


def test_go_with_plain_members_completes_or_is_refused_at_the_door(tmp_path):
    """``--members 2`` with no member sources never dies in the middle of the chain.

    What N members of one input should mean is the admission line's
    decision: the door either runs them, takes a recipe's real members, or
    refuses in one sentence. Each of those is a result. A traceback after
    the fetch and the preparation is not.
    """
    config, case_root = _tiny_config(tmp_path)
    ran, runs = _go(config, case_root, "--members", "2", "--keep-member-files")
    text = ran.stdout + ran.stderr
    assert "Traceback" not in text and "is not callable" not in text, text[-6000:]
    if ran.returncode != 0:
        assert ran.returncode == 2, text[-6000:]
        assert [line for line in ran.stderr.splitlines() if line.startswith("gpuwm go: ")], text[-6000:]
        return
    (run,) = runs
    if _hosted_by_go(ran):
        _check_completed(ran, run, 2)
        return
    # A member-source door ran them: the roster is that door's, and each of
    # its members still said where it was on the terminal.
    manifest = json.loads((run / "run" / "ensemble-run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["request"]["members"] == 2
    roster = manifest["member_order"]
    assert len(roster) == 2 and sorted(manifest["members_completed"]) == sorted(roster)
    for member in roster:
        assert f"ensemble: member {member} started" in ran.stderr, ran.stderr[-3000:]
        assert f"ensemble: member {member} finished" in ran.stderr, ran.stderr[-3000:]
    assert any((run / "run" / "maps").glob("d01/ens_*/*/*.png"))


_MEMBER_STARTED = re.compile(rb"ensemble: members? [0-9][0-9, ]* started \(")


def _interrupt_once_a_member_is_stepping(config, case_root, log, *flags):
    """Start the door, send it SIGINT after its first member started, and time the exit.

    Returns (exit code, seconds from the signal to the exit or None when no
    member started, everything the run printed, the run folders it made).
    """
    before = _run_folders(case_root)
    with open(log, "wb") as stream:
        child = subprocess.Popen(
            [sys.executable, "-m", "gpuwm", "go", str(config), *flags, "--outdir", str(case_root)],
            cwd=REPO, env=_environment(), stdout=stream, stderr=subprocess.STDOUT,
            # Ctrl-C as a terminal delivers it: a test runner started in the
            # background hands its children SIGINT ignored.
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        seconds = None
        try:
            deadline = time.monotonic() + 3000
            while child.poll() is None and time.monotonic() < deadline:
                if _MEMBER_STARTED.search(log.read_bytes()):
                    # Half a second of stepping, then the stop.
                    time.sleep(0.5)
                    sent = time.monotonic()
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=300)
                    seconds = time.monotonic() - sent
                    break
                time.sleep(0.05)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    return child.returncode, seconds, log.read_text(encoding="utf-8", errors="replace"), \
        _run_folders(case_root) - before


@pytest.mark.skipif(sys.platform == "win32", reason="the test delivers Ctrl-C as a POSIX signal")
@pytest.mark.parametrize("members", [1, 2])
def test_interrupt_stops_a_running_ensemble_at_the_door(tmp_path, members):
    """Ctrl-C ends the run at the next model step: one sentence, exit 130.

    The stop used to be held until every member in flight had finished its
    whole forecast. Here the running member has not finished, a member not
    yet started never starts, and the run folder says the run was interrupted.
    """
    config, case_root = _tiny_config(tmp_path)
    code, seconds, text, runs = _interrupt_once_a_member_is_stepping(
        config, case_root, tmp_path / "door.log", "--members", str(members), "--keep-member-files")
    assert "Traceback" not in text, text[-6000:]
    if seconds is None and code == 2 and members > 1:
        assert [line for line in text.splitlines() if line.startswith("gpuwm go: ")], text[-6000:]
        pytest.skip("this door refuses a plain member count before any member starts; "
                    "the one-member case is the interrupt gate")
    assert seconds is not None, f"no member started before the run ended (exit {code}):\n{text[-6000:]}"
    assert code == 130, f"exit {code}, {seconds:.2f} s after the interrupt:\n{text[-6000:]}"
    said = [line for line in text.splitlines() if "interrupted" in line]
    assert len(said) == 1, said
    assert "stopping: 1 running member batch ends at the next model step" in text, text[-3000:]
    # The member in flight did not run on to its end, and no other started.
    assert not re.search(r"ensemble: members? [0-9, ]+ finished", text), text[-3000:]
    assert len(re.findall(_MEMBER_STARTED.pattern.decode(), text)) == 1, text[-3000:]
    (run,) = runs
    manifest = json.loads((run / "run" / "ensemble-run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted" and manifest["error_type"] == "KeyboardInterrupt"
    assert manifest["members_completed"] == [] and manifest["failed_members"] == []
    assert manifest["members_not_completed"] == manifest["member_order"]
    assert len(manifest["member_order"]) == members
    # Nothing in the run folder says the run completed. The session writes
    # its completion report only for a finished roster; a report the runner
    # leaves behind is its own record of the stop.
    report = run / "run" / "report.json"
    if report.exists():
        left = json.loads(report.read_text(encoding="utf-8"))
        assert left["status"] != "PASS" and left["error_type"] == "KeyboardInterrupt", left
    if "  .. forecast (in process)" in text:
        # go hosted the members: its own stage stream closes as interrupted.
        from gpuwm.chain_events import CHAIN_EVENTS_FILENAME, read_chain_events
        last = read_chain_events(run / CHAIN_EVENTS_FILENAME)[-1]
        assert (last["event"], last["status"], last["exit_code"]) == ("failed", "INTERRUPTED", 130), last
        assert text.count("go: interrupted during forecast; no later stage ran") == 1
    print(f"interrupt at the door, {members} member(s): exit {code} in {seconds:.2f} s")
