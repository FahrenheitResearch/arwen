from tools.combo_round4_summary import measurement, report


def test_missing_runs_are_not_reported_as_fail_or_pass():
    row = dict(combo="fixture", case="fixture", outcome="FAIL",
               mp="6", pbl="1", sfclay="91", lsm="2", cu="1", rad="off")
    text = report([row], [])
    assert "Replayed 0 of 1" in text
    assert "| FAIL | 1 | 0 |" in text
    assert "| Missing round-4 verdict | 1 |" in text
    assert "unmeasured" in text


def test_digest_miss_has_no_invented_magnitude():
    assert measurement({"referee": "scalar", "comparisons": {"scalar": {"first_step": 1}}}) is None
    assert measurement({"referee": "scalar", "comparisons": {"scalar": {"details": [
        {"field": "U", "step": 5, "max_abs_diff": 0.5},
        {"field": "U", "step": 1, "max_abs_diff": 0.125}]}}}) == 0.125
