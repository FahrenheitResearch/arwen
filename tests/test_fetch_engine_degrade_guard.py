"""Degrading to the Python transport is a reportable event, not a default.

The Python transport is correct and stays the always-available fallback,
so nothing here refuses anything.  What it does is make the degrade
impossible to inherit silently: one warning line at SELECTION time, so
every caller of the front door says it, and one field in the receipt so
a slow run can be recognised afterwards instead of guessed at.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from gpuwm import doctor, explain, fetch, rustwx_fetch


def _records(monkeypatch, requested="auto"):
    said: list[dict] = []
    explain.add_warning_observer(said.append)
    try:
        choice = fetch.select_fetch_engine(requested, progress=lambda _: None)
    finally:
        explain.remove_warning_observer(said.append)
    return choice, said


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_a_present_backbone_is_selected_without_a_word(monkeypatch, tmp_path):
    binary = tmp_path / "rw_fetch"
    binary.write_bytes(b"")
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: binary)
    monkeypatch.setattr(rustwx_fetch, "probe_fetch_bin",
                        lambda path: (True, "fetch-record v1"))
    choice, said = _records(monkeypatch)
    assert (choice.engine, choice.binary) == ("rust", binary)
    assert choice.selection == "rust"
    assert not choice.degraded
    assert said == []


def test_a_missing_backbone_degrades_loudly_and_names_the_tax(monkeypatch):
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    choice, said = _records(monkeypatch)
    assert (choice.engine, choice.binary) == ("python", None)
    assert choice.selection == "python-fallback"
    assert choice.degraded
    assert len(said) == 1
    action = said[0]["action"]
    assert action == (
        "gpuwm fetch is using the Python transport (the vendored rw_fetch "
        "backbone is not installed). It has no whole-file branch: every "
        "object is pulled as hundreds of serial .idx range GETs, measured "
        "at 560 s for one 419 MB HRRR file against 27-35 s for the same "
        "file taken whole through the rust backbone -- roughly a 16x tax. "
        "Install the bridges bundle (`gpuwm setup`, or `gpuwm "
        "fetch-bridges`) to get the fast path.")


def test_the_warning_reaches_stderr_as_one_line(monkeypatch, capsys):
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    fetch.select_fetch_engine("auto", progress=lambda _: None)
    lines = [line for line in capsys.readouterr().err.splitlines() if line]
    assert len(lines) == 1
    assert lines[0].startswith("warning: gpuwm fetch is using the Python "
                               "transport")


def test_an_unusable_backbone_degrades_loudly_too(monkeypatch, tmp_path):
    stale = tmp_path / "rw_fetch"
    stale.write_bytes(b"")
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: stale)
    monkeypatch.setattr(rustwx_fetch, "probe_fetch_bin",
                        lambda path: (False, "different fetch-record ABI"))
    choice, said = _records(monkeypatch)
    assert choice.selection == "python-fallback"
    assert len(said) == 1
    assert "16x tax" in said[0]["action"]
    assert "unusable" in said[0]["action"]


def test_asking_for_python_is_not_a_degrade_and_says_nothing(monkeypatch):
    def explode():  # pragma: no cover - must not be reached
        raise AssertionError("--engine python must not probe for a binary")

    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", explode)
    choice, said = _records(monkeypatch, requested="python")
    assert choice.selection == "python-requested"
    assert not choice.degraded
    assert said == []


def test_asking_for_rust_still_refuses_rather_than_degrades(monkeypatch):
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    said: list[dict] = []
    explain.add_warning_observer(said.append)
    try:
        with pytest.raises(ValueError, match="cargo build"):
            fetch.select_fetch_engine("rust")
    finally:
        explain.remove_warning_observer(said.append)
    assert said == []


def test_the_pair_returning_wrapper_keeps_its_arity(monkeypatch):
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    assert fetch.resolve_fetch_engine("auto") == ("python", None)


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------


def _manifest(**overrides) -> dict:
    payload = fetch._manifest_payload(
        source="hrrr", cycle=datetime(2026, 8, 12, tzinfo=timezone.utc),
        hours=(0,), area=None,
        files=[{"name": "a.grib2", "role": "hrrr", "forecast_hour": 0,
                "bytes": 1, "sha256": "0" * 64, "url": None}])
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("engine,passed,expected", [
    ("rust", "rust", "rust"),
    ("python", "python-fallback", "python-fallback"),
    ("python", "python-requested", "python-requested"),
    # A caller that resolved the engine some other way never has a
    # degrade invented for it.
    ("rust", None, "rust"),
    ("python", None, "python-requested"),
])
def test_the_receipt_records_how_the_engine_was_chosen(
        engine, passed, expected):
    assert fetch._engine_selection(engine, passed) == expected


def test_an_unknown_selection_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown engine selection"):
        fetch._engine_selection("python", "urllib")


def test_a_degraded_receipt_is_readable_as_json(tmp_path):
    payload = _manifest(engine="python",
                        engine_selection=fetch.PYTHON_FALLBACK_SELECTION,
                        mode="auto")
    path = fetch.write_fetch_manifest(tmp_path, payload)
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert recorded["engine"] == "python"
    assert recorded["engine_selection"] == "python-fallback"
    # The host axis keeps its own word; the two must not be confusable.
    assert "transport" not in recorded or recorded["transport"] != \
        "python-fallback"


def test_every_selection_name_is_declared():
    assert fetch.PYTHON_FALLBACK_SELECTION in fetch.FETCH_ENGINE_SELECTIONS
    assert set(fetch.FETCH_ENGINE_SELECTIONS) == {
        "rust", "python-requested", "python-fallback"}


# ---------------------------------------------------------------------------
# The doctor line
# ---------------------------------------------------------------------------


def test_doctor_prices_the_missing_backbone_instead_of_only_naming_it(
        monkeypatch):
    monkeypatch.setattr(rustwx_fetch, "find_fetch_bin", lambda: None)
    check = doctor._fetch_backbone_check()
    assert check.status == "info"
    assert "16x" in check.detail
    assert "560 s" in check.detail
    assert "engine_selection='python-fallback'" in check.detail
    assert "16x" in check.brief
