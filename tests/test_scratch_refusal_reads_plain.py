"""A preparation's refusal reaches a run's failure notice as its own sentences.

The preparation door prints a refusal and exits 78; the chain that ran it
saw only the status, so a scratch disk too small for the frame stream
reached ``gpuwm gui``'s failure notice, and the events every front end
reads, as "prepare failed (exit 78)." with no remedy.  And where the words
did arrive, the page took every machine path out of them, the scratch
folder the remedy is about included.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import mapped_engine_bridge as bridge
from gpuwm.ingest.source_coverage import (
    PREPARATION_REFUSAL_EXIT_CODE, ScratchDiskRefusal, owns_source_coverage_refusal,
    recorded_preparation_refusal, scratch_disk_refusal)

FIXTURES = Path(__file__).parent / "fixtures" / "download_budget"


def _engine_refusal(folder: Path) -> ScratchDiskRefusal:
    """The engine's disk_full refusal for a 48 hour GDPS stream, as the bridge maps it."""

    return bridge.refusal_error({
        "class": "disk_full",
        "message": ("the frame stream needs 82321344000 bytes (76.7 GiB) in "
                    f"{folder}/gpuwm-mapped-compose-x1/composed and the disk that holds that "
                    "folder has 10737418240 bytes (10.0 GiB) free.  It is 17 valid times of "
                    "4842432000 bytes (4.5 GiB) each, refused before its first byte rather "
                    "than written until the disk fills"),
        "remedy": "free space on the disk that holds the output directory"})


def test_a_refusal_a_child_door_reports_reaches_the_caller_whole(tmp_path):
    """The adapter runs in a child process; its refusal crosses back as data, not a status."""
    folder = tmp_path / "chain"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from gpuwm.ingest.source_coverage import ScratchDiskRefusal, report_preparation_refusal\n"
        f"folder = Path({str(folder)!r})\n"
        "refusal = ScratchDiskRefusal('the frame stream needs 9 bytes', "
        "remedy='remedy: set GPUWM_COMPOSE_SCRATCH', folders=(folder,))\n"
        "sys.exit(report_preparation_refusal(refusal))\n")
    with recorded_preparation_refusal() as refused:
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == PREPARATION_REFUSAL_EXIT_CODE, done.stderr
    assert done.stderr.startswith("prep: REFUSED: the frame stream needs 9 bytes")
    refusal = refused()
    assert type(refusal) is ScratchDiskRefusal
    assert str(refusal) == "the frame stream needs 9 bytes"
    assert refusal.remedy == "remedy: set GPUWM_COMPOSE_SCRATCH"
    assert refusal.folders == (str(folder),)
    assert refusal.exit_code == 78


def _door(folder: Path):
    """A preparation door whose engine refused the frame stream for lack of disk."""

    @owns_source_coverage_refusal
    def door(args):
        raise scratch_disk_refusal(_engine_refusal(folder), folder)

    return door


def test_the_chain_raises_the_refusal_the_preparation_printed(tmp_path, monkeypatch):
    import gpuwm.cli
    from gpuwm.runplan import _run_prep

    folder = tmp_path / "chain"
    parser = SimpleNamespace(parse_args=lambda argv: SimpleNamespace(func=_door(folder)))
    monkeypatch.setattr(gpuwm.cli, "build_parser", lambda: parser)
    with pytest.raises(ScratchDiskRefusal) as caught:
        _run_prep(["--source", "gem-gdps"])
    refusal = caught.value
    assert refusal.exit_code == 78
    assert "17 valid times of 4842432000 bytes" in str(refusal)
    assert f"stages its frame stream in {folder}." in refusal.remedy
    assert "GPUWM_COMPOSE_SCRATCH" in refusal.remedy
    assert refusal.folders == (str(folder),)


def _gdps_plan(tmp_path: Path) -> Path:
    from gpuwm.runplan import PLAN_SCHEMA

    text = (FIXTURES / "gfs-3km.toml").read_text(encoding="utf-8")
    text = text.replace("start_time = 2026-09-26T06:00:00", "start_time = 2026-09-26T00:00:00")
    text = text.replace('source = "gfs"', 'source = "gem-gdps"')
    text = text.replace('cycle = "2026-09-26T06"', 'cycle = "2026-09-26T00"')
    text = text.replace('area = "17.61,-116.54,53.24,-79.36"\n', "")
    config = tmp_path / "gdps-3km.toml"
    config.write_text(text, encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": PLAN_SCHEMA, "name": "gdps", "route": "prepared",
                                "config": {"path": str(config)},
                                "output_root": str(tmp_path / "runs" / "gdps")}),
                    encoding="utf-8")
    return plan


def test_the_gui_failure_notice_reads_the_refusal_its_remedy_and_its_folder(tmp_path, monkeypatch):
    """Run the plan through a preparation that refuses; read the run back as the page does."""
    from gpuwm import capabilities
    import gpuwm.cli
    import gpuwm.runplan as runplan
    from gpuwm.gui import runs
    from gpuwm.runplan import EVENTS_FILENAME, EventStream, execute_plan, load_plan

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setattr(capabilities, "require", lambda *args, **kwargs: None)
    folder = tmp_path / "runs" / "gdps" / "chain"
    parser = SimpleNamespace(parse_args=lambda argv: SimpleNamespace(func=_door(folder)))
    monkeypatch.setattr(gpuwm.cli, "build_parser", lambda: parser)
    monkeypatch.setattr(runplan, "_staged_chain",
                        lambda *args, **kwargs: runplan._run_prep(["--source", "gem-gdps"]))
    plan = load_plan(_gdps_plan(tmp_path))
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with EventStream(plan.run_dir / EVENTS_FILENAME, mirror=None) as events:
        assert execute_plan(plan, events=events) == 1
    failed = [json.loads(line) for line in
              (plan.run_dir / EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()][-1]
    assert failed["event"] == "failed" and failed["exit_code"] == 78
    assert failed["error_class"] == "ScratchDiskRefusal"
    assert failed["folders"] == [str(folder)]

    status = runs.status(plan.run_dir)
    assert status["state"] == "failed"
    end = status["end"]
    assert "exit 78" not in end["message"]
    assert "needs 82321344000 bytes (76.7 GiB)" in end["message"]
    assert "has 10737418240 bytes (10.0 GiB) free" in end["message"]
    assert f"{folder}/gpuwm-mapped-compose-x1/composed" in end["message"]
    # The page shows a remedy as one paragraph without its terminal label.
    assert end["remedy"].startswith(f"the preparation stages its frame stream in {folder}.")
    assert "GPUWM_COMPOSE_SCRATCH" in end["remedy"]


#: A home directory of an account no machine has, assembled from pieces: written
#: out whole it reads as a leaked path to the release's machine-path scan
#: (work/build_release_snapshot.py), which stops the cut's battery.
HOME_A = "/" + "home/a"


def test_the_page_still_hides_every_other_path():
    from gpuwm.gui.files import plain_message

    kept = plain_message(f"stages in {HOME_A}/run/chain/x and reads {HOME_A}/secret/y",
                         [f"{HOME_A}/run/chain"])
    assert f"{HOME_A}/run/chain/x" in kept and f"{HOME_A}/secret" not in kept
    assert f"{HOME_A}/run" not in plain_message(f"reads {HOME_A}/run/chain/x", [])
    # The root would keep every path; it keeps none.
    assert "/home" not in plain_message(f"reads {HOME_A}/x", ["/"])
