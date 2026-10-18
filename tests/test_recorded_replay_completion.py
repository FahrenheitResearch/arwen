"""Completion receipts must contain no post-forecast engine error."""
import json
import os
from pathlib import Path
import pytest


def test_actual_replay_finishes_its_checkpoint_capture():
    path = os.environ.get("GPUWM_EXACT_VERDICT")
    if not path:
        pytest.skip("set the real CLI's verdict JSONL")
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    assert rows
    for row in rows:
        assert not row["woof"].get("error"), row["woof"].get("error")
