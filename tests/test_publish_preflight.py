"""Refusals stop public writes, including when storage grows during the CI wait."""
import json
from types import SimpleNamespace

import pytest

from tools.release import publish_preflight as preflight


def test_missing_rehearsal_stops_before_pypi_access(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.pypi_capacity, "fetch_project", lambda p: pytest.fail("network attempted"))
    with pytest.raises(OSError):
        preflight.check(tmp_path, tmp_path, tmp_path, "a" * 40)


def test_preflight_writes_capacity_receipt_only_after_rehearsal_validation(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(preflight, "require_receipt", lambda *a: events.append("rehearsal"))
    monkeypatch.setattr(preflight.pypi_capacity, "check_packet", lambda *a, **kw: events.append("capacity"))
    preflight.check(tmp_path, tmp_path, tmp_path, "a" * 40)
    assert events == ["rehearsal", "capacity"]
