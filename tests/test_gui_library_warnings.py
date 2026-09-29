"""The engine's library warnings reach the page, while the run goes and after it ends.

The defect: a library the run drives can warn and let the run go on as
configured (a warm bubble of 15 K, above 10 K, replaces the analysis near its
centre rather than nudging it). The run's events.jsonl carried the warning,
but the page kept only the missing-basemap warning, so neither the live map
nor the map reopened after the run said anything about it.
"""

from __future__ import annotations

import json
import threading

from gpuwm.gui import runs, server as gui_server
from gpuwm.runplan import WARNING_CODES

WARM = ("perturbation bubble amplitude_k = 15 K is above 10 K and 5.0 times WRF's idealized warm bubble "
        "(3 K, em_quarter_ss): it replaces the analysis near its center rather than nudging it. It runs as "
        "configured, and this warning is recorded in the perturbation receipt.")

#: A home directory of an account no machine has, for the path the page must not show. It is
#: assembled, as tests/test_report_bundle.py does, because the release's machine-path scan
#: (work/build_release_snapshot.py) reads a literal one in a shipped file as a leaked path and
#: stops the cut's battery before the freeze.
SOMEONE_HOME = "/" + "home/someone"


def write_events(path, records):
    path.write_text("".join(json.dumps({"sequence": index + 1, **record}) + "\n"
                            for index, record in enumerate(records)), encoding="utf-8")


def test_the_code_is_the_engines():
    assert runs.LIBRARY_WARNING in WARNING_CODES


def test_a_library_warning_stays_after_the_run_ends(tmp_path):
    write_events(tmp_path / runs.EVENTS, [
        {"event": "plan_accepted"},
        {"event": "warning", "code": "library_warning", "message": WARM, "detail": ""},
        {"event": "warning", "code": "library_warning", "message": WARM, "detail": ""},
        {"event": "warning", "code": "library_warning", "message": "Another setting needs review.",
         "detail": "It runs as configured."},
        {"event": "stage_started", "stage": "forecast"},
        {"event": "completed"},
    ])
    info = runs.status(tmp_path)
    assert info["state"] == "finished"
    # Each once, in the order the engine said them.
    assert info["library_warnings"] == [
        {"message": WARM, "detail": None},
        {"message": "Another setting needs review.", "detail": "It runs as configured."}]


def test_a_new_attempt_starts_with_its_own_warnings(tmp_path):
    records = [{"event": "plan_accepted"},
               {"event": "warning", "code": "library_warning", "message": WARM, "detail": ""},
               {"event": "failed", "message": "stopped"}]
    write_events(tmp_path / runs.EVENTS, records)
    assert len(runs.status(tmp_path)["library_warnings"]) == 1
    write_events(tmp_path / runs.EVENTS, [*records, {"event": "plan_accepted"}])
    assert runs.status(tmp_path)["library_warnings"] == []


def test_a_warning_shows_words_not_the_machines_paths(tmp_path):
    write_events(tmp_path / runs.EVENTS, [
        {"event": "plan_accepted"},
        {"event": "warning", "code": "library_warning",
         "message": f"The table at {SOMEONE_HOME}/tables/x.bin is older than this engine.\nsecond line",
         "detail": "It runs as configured."},
    ])
    [warning] = runs.status(tmp_path)["library_warnings"]
    assert SOMEONE_HOME not in warning["message"] and "second line" not in warning["message"]


def test_the_event_stream_sends_a_warning_that_lands_while_the_page_is_open(tmp_path):
    events = tmp_path / runs.EVENTS
    write_events(events, [{"event": "plan_accepted"}, {"event": "stage_started", "stage": "forecast"}])
    gone, stopping = threading.Event(), threading.Event()
    timer = threading.Timer(10.0, stopping.set)
    timer.start()
    seen = []
    try:
        for event, _, data in gui_server.follow(tmp_path, None, gone, stopping):
            if event != "status":
                continue
            seen.append(data["library_warnings"])
            if len(seen) == 1:
                with events.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"sequence": 3, "event": "warning", "code": "library_warning",
                                             "message": WARM, "detail": ""}) + "\n")
            if len(seen) == 2:
                break
    finally:
        timer.cancel()
    assert seen == [[], [{"message": WARM, "detail": None}]]
