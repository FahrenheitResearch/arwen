import pytest
from tools.coupled_exact_chain import summarize


def run(build="scalar"):
    return dict(case="case", combo="combo", build=build, status="OK", last_good_step="20")


def verdict(comparison):
    return dict(case="case", combo="combo", outcome="FAIL", comparisons={"scalar": comparison})


def test_every_recorded_build_keeps_missing_comparisons_unqualified():
    rows, totals, _ = summarize([run(), run("strict")], [verdict(dict(matched=True, steps_compared=list(range(21))))])
    assert rows[0]["exact_all_fields"]
    assert not rows[1]["exact_all_fields"]
    assert totals["all_field_exact_rows"] == 1


@pytest.mark.parametrize("comparison", [
    dict(matched=True, steps_compared=list(range(20))),
    dict(matched=False, steps_compared=list(range(21)), absent_unexcused=["RTHRATEN"]),
    dict(matched=True, steps_compared=list(range(21)), missing_steps=[4]),
])
def test_short_or_incomplete_replay_never_counts_as_full_length_exact(comparison):
    assert not summarize([run()], [verdict(comparison)])[0][0]["exact_all_fields"]


def test_absent_verdict_is_counted_as_missing():
    _, totals, _ = summarize([run()], [])
    assert totals["missing_rows"] == 1
    assert totals["all_field_exact_rows"] == 0


def test_duplicate_verdict_refused():
    v = verdict({})
    with pytest.raises(ValueError, match="duplicate verdict"):
        summarize([run()], [v, v])
