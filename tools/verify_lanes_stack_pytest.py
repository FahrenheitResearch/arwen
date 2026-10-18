"""Record real comparator returns from the stack's pytest subprocesses."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import json
from numbers import Integral
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EVENTS = []
REPORTS = []
CURRENT = None
ARRAY_EQUAL = None
CALLS = {}
SERIAL = 0
NAMES = ("compare", "summar", "grade", "measure", "ulp", "bit_stats")


def pytest_addoption(parser):
    parser.addoption("--stack-receipt", default=None)
    parser.addoption("--stack-collection", default=None)


def pytest_collection_finish(session):
    target = session.config.getoption("--stack-collection")
    if target:
        rows = [{"nodeid": item.nodeid,
                 "gpu": item.get_closest_marker("gpu") is not None}
                for item in session.items]
        Path(target).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _counts(value):
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, dict):
        return []
    count_keys = ("compared_values", "compared_words", "words", "values", "n",
                  "count", "total", "compared_slots")
    bad_keys = ("mismatched_values", "mismatched_words", "words_differ",
                "mismatched", "mismatch", "bit_mismatch", "bit_mismatches",
                "words_different", "bits", "differing")
    count = next((value[key] for key in count_keys
                  if isinstance(value.get(key), Integral)), None)
    bad = next((value[key] for key in bad_keys
                if isinstance(value.get(key), Integral)), None)
    if count is not None and bad is not None:
        return [{"words": int(count), "differing": int(bad),
                 "max_ulp": value.get("max_ulp", value.get("ulp"))}]
    rows = []
    for child in value.values():
        if isinstance(child, dict) or is_dataclass(child):
            rows.extend(_counts(child))
    return rows


def _comparison(frame):
    return (str(ROOT) in frame.f_code.co_filename
            and not frame.f_code.co_name.startswith("test_")
            and any(name in frame.f_code.co_name.lower() for name in NAMES))


def _ancestors(frame):
    found = []
    while frame:
        if id(frame) in CALLS:
            found.append(CALLS[id(frame)])
        frame = frame.f_back
    return found


def _profile(frame, event, value):
    global SERIAL
    if not _comparison(frame):
        return
    if event == "call":
        SERIAL += 1
        CALLS[id(frame)] = SERIAL
        return
    if event != "return":
        return
    rows = _counts(value)
    if not rows and "ulp" in frame.f_code.co_name.lower():
        import numpy as np
        if isinstance(value, np.ndarray) and value.dtype.kind in "iu":
            rows = [{"words": int(value.size), "differing": int(np.count_nonzero(value)),
                     "max_ulp": int(value.max(initial=0)),
                     "basis": "oracle ULP distance with its zero and NaN policy"}]
    if rows:
        EVENTS.append({"test": CURRENT, "file": frame.f_code.co_filename,
                       "line": frame.f_code.co_firstlineno,
                       "function": frame.f_code.co_name,
                       "call": CALLS.get(id(frame)), "ancestors": _ancestors(frame.f_back),
                       "rows": rows})
    CALLS.pop(id(frame), None)


def pytest_runtest_setup(item):
    global CURRENT
    CURRENT = item.nodeid


def pytest_runtest_logreport(report):
    REPORTS.append({"nodeid": report.nodeid, "when": report.when,
                    "outcome": report.outcome})


def pytest_sessionstart(session):
    global ARRAY_EQUAL
    if session.config.getoption("--stack-receipt"):
        import numpy as np
        ARRAY_EQUAL = np.testing.assert_array_equal

        def observed_array_equal(actual, expected, *args, **kwargs):
            left, right = np.asarray(actual), np.asarray(expected)
            if (left.shape == right.shape and left.dtype.itemsize in (4, 8)
                    and right.dtype.itemsize == left.dtype.itemsize):
                integer = np.uint32 if left.dtype.itemsize == 4 else np.uint64
                differing = int((left.view(integer) != right.view(integer)).sum())
                EVENTS.append({"test": CURRENT, "function": "numpy.testing.assert_array_equal",
                               "ancestors": _ancestors(sys._getframe().f_back),
                               "rows": [{"words": int(left.size), "differing": differing,
                                         "word_bits": left.dtype.itemsize * 8}]})
            return ARRAY_EQUAL(actual, expected, *args, **kwargs)

        np.testing.assert_array_equal = observed_array_equal
        sys.setprofile(_profile)


def pytest_sessionfinish(session, exitstatus):
    sys.setprofile(None)
    if ARRAY_EQUAL is not None:
        import numpy as np
        np.testing.assert_array_equal = ARRAY_EQUAL
    target = session.config.getoption("--stack-receipt")
    if not target:
        return
    # Counts come from executed comparison functions, including repeated
    # calls. They are not unique grid cells or copied historical counts.
    measured = {event["call"] for event in EVENTS if event.get("call") is not None}
    # Prefer an aggregate's real returned count. If it returns only maxima,
    # retain its executed ULP-array comparisons instead. Never count both.
    events = [event for event in EVENTS
              if not measured.intersection(event.get("ancestors", []))]
    document = {"schema": "gpuwm.stack-comparator-events/v1",
                "exit_code": int(exitstatus), "events": events,
                "reports": REPORTS,
                "counting": "executed numeric-word comparisons; 32 or 64 bits; repeated calls counted",
                "compared_words": sum(row["words"] for event in events
                                      for row in event["rows"]),
                "differing_words": sum(row["differing"] for event in events
                                       for row in event["rows"])}
    Path(target).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("STACK_WORDS " + json.dumps({key: document[key] for key in
                                      ("compared_words", "differing_words")}))
