"""A failed stage whose event stream its owner already closed does not mask the failure.

Breakage it prevents (golden 2.9.0 verify round 1, the ensemble showcase's
defect (b), reproduced on test host B with the candidate wheel): an ensemble
member's preparation refused (GEFS has no analysed aerosol), the door said so
and returned exit 1, and then the command's ledger closed the member's
observer.  Its preparation stage was still open, so the observer wrote
``stage_finished`` to the member's ``prepare-events.jsonl``, which the door's
``finally`` had already closed.  "I/O operation on closed file" replaced the
refusal: the run ended ``gpuwm ensemble: I/O operation on closed file.`` at
exit 2.  The same shape exists wherever a door closes the stream it handed a
RunObserver (gpuwm/regional_preparation.py does it twice).
"""

from __future__ import annotations

import json

from gpuwm import runplan
from gpuwm.stage_timing import ledger_main


def test_closing_an_observer_after_its_stream_closed_ends_the_stage_quietly(tmp_path):
    events = runplan.EventStream(tmp_path / "prepare-events.jsonl", mirror=None)
    observer = runplan.RunObserver(events)
    observer.enter_stage("prepare")
    events.close()
    observer.close()                       # raised ValueError before the fix
    assert observer.stage is None
    rows = [json.loads(line) for line in
            (tmp_path / "prepare-events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["stage_started"]


def test_the_command_keeps_its_own_exit_code_when_a_preparation_failed(tmp_path):
    @ledger_main
    def door():
        events = runplan.EventStream(tmp_path / "prepare-events.jsonl", mirror=None)
        try:
            observer = runplan.RunObserver(events)
            observer.enter_stage("prepare")
            raise ValueError("analyzed aerosol IC/BC requires QNWFA and QNIFA")
        except ValueError:
            return 1                       # the door's own refusal code
        finally:
            events.close()

    assert door() == 1


def test_an_open_stream_still_records_the_failed_stage(tmp_path):
    events = runplan.EventStream(tmp_path / "events.jsonl", mirror=None)
    observer = runplan.RunObserver(events)
    observer.enter_stage("prepare")
    observer.close()
    events.close()
    rows = [json.loads(line) for line in
            (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["stage_started", "stage_finished"]
