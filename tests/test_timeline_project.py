import pytest

from gpuwm.timeline.project import project


def reference(mode="rental"):
    return {"schema": "gpuwm.timeline.reference.v1", "product": "regional", "mode": mode, "cards": 1,
            "measured_wall_seconds": 120, "measured_ff": 100 / 120, "nodes": [
        {"id": "prep", "stage": "prep.head", "duration_seconds": 20, "evidence": "reference:1"},
        {"id": "side", "stage": "fetch.side", "duration_seconds": 10, "evidence": "reference:2"},
        {"id": "integrate", "stage": "forecast.integrate", "duration_seconds": 100,
         "depends_on": ["prep", "side"], "card": 0, "evidence": "reference:3"}]}


def delta(lane, seconds_saved, **fields):
    return {"lane": lane, "stage": "prep.head", "seconds_saved": seconds_saved,
            "measured": True, "evidence": lane + ":1", **fields}


def test_independent_overlapping_deltas_count_overlap_once():
    result = project(reference(), [delta("first", 8), delta("second", 5)])
    assert result["seconds_saved"] == 8
    assert result["stage_seconds_removed"]["prep"] == 8
    assert result["projected_ff"] == 100 / 112


def test_explicit_partial_overlap_is_unioned():
    result = project(reference(), [delta("first", 8, removed_spans=[[0, 8]]),
                                   delta("second", 8, removed_spans=[[4, 12]])])
    assert result["seconds_saved"] == 10  # side branch now gates the forecast
    assert result["stage_seconds_removed"]["prep"] == 12
    assert "side" in result["projected"]["critical_path_nodes"]


def test_noncritical_delta_does_not_move_ff():
    result = project(reference(), [delta("side", 8, stage="fetch.side")])
    assert result["seconds_saved"] == 0
    assert result["projected_ff"] == result["measured_ff"]


def test_marginal_delta_after_dependency_uses_remaining_work():
    result = project(reference(), [delta("second", 5, depends_on=["first"]), delta("first", 8)])
    assert result["stage_seconds_removed"]["prep"] == 13
    assert result["seconds_saved"] == 10  # limited by parallel branch


def test_integration_delta_changes_numerator_and_denominator():
    result = project(reference(), [delta("integrate", 50, stage="forecast.integrate")])
    assert result["seconds_saved"] == 50
    assert result["projected_ff"] == 50 / 70


def test_projected_union_across_cards_and_proxy_labels():
    ref = reference("proxy")
    ref["cards"] = 2
    ref["nodes"].append({"id": "other", "stage": "forecast.integrate", "duration_seconds": 90,
                        "depends_on": ["prep", "side"], "card": 1, "evidence": "reference:4"})
    result = project(ref, [])
    assert result["projected_ff"] is None
    assert result["measured_ff"] is None
    assert result["projected_proxy_ff"] == 100 / 120
    assert result["projected"]["integrate_union_seconds"] == 100


@pytest.mark.parametrize("changes,pattern", [
    ({"depends_on": ["absent"]}, "missing or cyclic"),
    ({"measured": False}, "unmeasured"),
    ({"stage": "prep.absent"}, "absent or ambiguous"),
    ({"seconds_saved": -1}, "nonnegative"),
    ({"seconds_saved": 21}, "exceeds work"),
    ({"seconds_saved": float("nan")}, "finite"),
])
def test_invalid_deltas_do_not_produce_optimistic_estimates(changes, pattern):
    with pytest.raises(ValueError, match=pattern):
        project(reference(), [delta("bad", 5, **changes)] if "seconds_saved" not in changes
                else [{**delta("bad", 5), **changes}])


def test_marginal_overlap_reference_mismatch_and_duplicate_lane_refuse():
    with pytest.raises(ValueError, match="overlap depends_on"):
        project(reference(), [delta("first", 8), delta("second", 5, depends_on=["first"], removed_spans=[[15, 20]])])
    with pytest.raises(ValueError, match="duplicate"):
        project(reference(), [delta("same", 8), delta("same", 5)])
    ref = reference()
    ref["measured_wall_seconds"] = 150
    with pytest.raises(ValueError, match="measured wall"):
        project(ref, [])
    with pytest.raises(ValueError, match="dependency edges"):
        project({"schema": "gpuwm.timeline.report.v1"}, [])
