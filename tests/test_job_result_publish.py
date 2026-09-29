"""A cancel and the job's own wrapper both publish result.json at once.

Seen on a Linux node: the manager's cancel and the wrapper it had just
signalled wrote the same "result.tmp"; the first replace took the other's
file and the cancel raised FileNotFoundError, so Stop failed.  The
interleaving is made here on purpose: the wrapper's write lands between the
manager's write and its replace.
"""

from __future__ import annotations

import json
import os

from gpuwm.mcp import _jobwrap, jobs


def test_a_cancel_and_the_wrapper_publishing_together_both_land(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    real_replace = os.replace
    between: list[str] = []

    def replace(src, dst):
        if not between:
            between.append(str(src))
            _jobwrap._publish(target, {"exit_code": -15, "cancelled": True, "by": "wrapper"})
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    jobs._publish(target, {"exit_code": None, "cancelled": True, "by": "manager"})
    assert between, "the wrapper's write was not placed between the manager's write and replace"
    assert json.loads(target.read_text(encoding="utf-8"))["cancelled"] is True
    assert not list(tmp_path.glob("*.tmp"))


def test_the_other_order_lands_too(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    real_replace = os.replace
    between: list[str] = []

    def replace(src, dst):
        if not between:
            between.append(str(src))
            jobs._publish(target, {"exit_code": None, "cancelled": True, "by": "manager"})
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    _jobwrap._publish(target, {"exit_code": -15, "cancelled": True, "by": "wrapper"})
    assert json.loads(target.read_text(encoding="utf-8"))["by"] == "wrapper"
    assert not list(tmp_path.glob("*.tmp"))
