"""Use the real CLI receipt to reject a crash with complete field records."""
import json
import os
from pathlib import Path
import pytest


def test_actual_score_only_crash_is_not_an_exact_pass():
    path = os.environ.get("GPUWM_EXACT_VERDICT")
    if not path:
        pytest.skip("set a real score-only verdict for an errored complete replay")
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    assert rows
    for row in rows:
        assert row["woof"]["status"] == "CRASH"
        assert row["woof"]["error"], "the regression receipt must contain a real engine error"
        assert row["outcome"] == "WOOF-CRASH", "matching output fields cannot hide an engine error"
