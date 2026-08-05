"""The obs scorer's statistics: masked FSS, contingency scores, remapping.

The load-bearing pin here is the first one.  The battery claims to score
against observations with *the existing FSS engine*, so the masked
implementation must reduce, exactly, to ``1 - field_metrics.fss_distance``
when nothing is masked and the boundary treatment is the engine's own.  If
that identity ever breaks, the battery is running arithmetic nobody reviewed
under a name somebody trusts.

The rest are attempts to make the statistics lie: mutate data under the mask
and demand the score not move, displace a forecast and demand it fall, feed a
uniformly-biased forecast and demand frequency matching remove the bias term.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify import field_metrics
from gpuwm.verify.obs import contingency, fss, regrid


def _blobs(shape=(60, 70), centers=((20, 25), (40, 50)), radius=6.0,
           peak=55.0, background=-10.0):
    ny, nx = shape
    j, i = np.indices((ny, nx)).astype(np.float64)
    field = np.full(shape, background, dtype=np.float64)
    for cj, ci in centers:
        distance = np.hypot(j - cj, i - ci)
        field = np.maximum(
            field, background + (peak - background)
            * np.exp(-0.5 * (distance / radius) ** 2))
    return field


# --------------------------------------------------------------------------
# the identity that lets the battery claim it uses the existing engine
# --------------------------------------------------------------------------


@pytest.mark.parametrize("half_width", [0, 1, 4, 8])
@pytest.mark.parametrize("threshold", [20.0, 30.0, 40.0])
def test_unmasked_edge_fss_is_the_existing_engine(half_width, threshold):
    forecast = _blobs(centers=((20, 27), (41, 52)))
    observed = _blobs()
    valid = np.ones(forecast.shape, dtype=bool)

    result = fss.masked_fss(
        forecast, observed, valid=valid, threshold=threshold,
        half_width=half_width, boundary=fss.EDGE_BOUNDARY)
    engine = 1.0 - field_metrics.fss_distance(
        forecast, observed, threshold=threshold, half_width=half_width)

    assert result.fss == pytest.approx(engine, abs=0.0, rel=0.0)


def test_unmasked_fraction_is_the_existing_neighborhood_fraction():
    events = _blobs() >= 30.0
    valid = np.ones(events.shape, dtype=bool)
    fraction, counts = fss.masked_neighborhood_fraction(
        events, valid, 3, boundary=fss.EDGE_BOUNDARY)
    assert np.array_equal(
        fraction, field_metrics.neighborhood_fraction(events, 3))
    assert np.all(counts == 1.0)


# --------------------------------------------------------------------------
# the mask actually masks
# --------------------------------------------------------------------------


def test_data_under_the_mask_cannot_move_the_score():
    observed = _blobs()
    forecast = _blobs(centers=((20, 27), (41, 52)))
    valid = np.ones(observed.shape, dtype=bool)
    valid[10:20, 10:20] = False

    clean = fss.masked_fss(forecast, observed, valid=valid, threshold=30.0,
                           half_width=4)

    poisoned_obs = observed.copy()
    poisoned_forecast = forecast.copy()
    poisoned_obs[10:20, 10:20] = 75.0
    poisoned_forecast[10:20, 10:20] = -35.0
    poisoned = fss.masked_fss(poisoned_forecast, poisoned_obs, valid=valid,
                              threshold=30.0, half_width=4)

    assert poisoned.fss == clean.fss
    assert poisoned.observed_base_rate == clean.observed_base_rate


def test_masking_changes_the_score_when_it_covers_real_signal():
    observed = _blobs()
    forecast = _blobs(centers=((20, 33), (41, 58)))
    everywhere = np.ones(observed.shape, dtype=bool)
    partial = everywhere.copy()
    partial[:, 40:] = False

    assert (fss.masked_fss(forecast, observed, valid=partial, threshold=30.0,
                           half_width=4).fss
            != fss.masked_fss(forecast, observed, valid=everywhere,
                              threshold=30.0, half_width=4).fss)


def test_a_cell_with_no_valid_neighbourhood_is_dropped_not_scored():
    observed = np.full((21, 21), 40.0)
    forecast = np.full((21, 21), 40.0)
    valid = np.zeros((21, 21), dtype=bool)
    valid[10, 10] = True
    result = fss.masked_fss(forecast, observed, valid=valid, threshold=30.0,
                            half_width=1)
    # Only the 3x3 around the single valid cell has a populated neighborhood.
    assert result.scored_cells == 9
    assert result.fss == 1.0


def test_zero_boundary_does_not_extend_the_field_past_the_domain():
    # One row of events along the array edge, half-width 2.  Under edge
    # extension the box at (0, 10) holds 25 cells, all counted valid, and the
    # two replicated rows are counted as ten observed events that were never
    # observed: 15/25 = 3/5.  Under zero padding the out-of-domain cells are
    # simply not valid, so the box holds 15 real cells of which 5 are events:
    # 1/3.  The difference is entirely fabricated data.
    observed = np.full((21, 21), 0.0)
    forecast = np.full((21, 21), 0.0)
    observed[0, :] = 50.0
    forecast[0, :] = 50.0
    valid = np.ones((21, 21), dtype=bool)

    edge, edge_counts = fss.masked_neighborhood_fraction(
        observed >= 30.0, valid, 2, boundary=fss.EDGE_BOUNDARY)
    zero, zero_counts = fss.masked_neighborhood_fraction(
        observed >= 30.0, valid, 2, boundary=fss.ZERO_BOUNDARY)

    assert edge[0, 10] == pytest.approx(3.0 / 5.0)
    assert zero[0, 10] == pytest.approx(1.0 / 3.0)
    assert edge_counts[0, 10] == pytest.approx(1.0)
    assert zero_counts[0, 10] == pytest.approx(15.0 / 25.0)
    # Well inside the domain the two treatments agree exactly.
    assert edge[10, 10] == zero[10, 10]
    assert fss.masked_fss(forecast, observed, valid=valid, threshold=30.0,
                          half_width=2).fss == 1.0


# --------------------------------------------------------------------------
# the score means what it says
# --------------------------------------------------------------------------


def test_identical_fields_score_one_and_displacement_lowers_the_score():
    observed = _blobs()
    valid = np.ones(observed.shape, dtype=bool)
    perfect = fss.masked_fss(observed, observed, valid=valid, threshold=30.0,
                             half_width=2)
    assert perfect.fss == 1.0

    scores = []
    for shift in (2, 6, 14):
        displaced = np.roll(observed, shift, axis=1)
        scores.append(fss.masked_fss(displaced, observed, valid=valid,
                                     threshold=30.0, half_width=2).fss)
    assert scores == sorted(scores, reverse=True)
    assert scores[0] < 1.0


def test_larger_neighbourhoods_never_reduce_the_score_for_a_displacement():
    observed = _blobs()
    forecast = np.roll(observed, 7, axis=1)
    valid = np.ones(observed.shape, dtype=bool)
    series = [fss.masked_fss(forecast, observed, valid=valid, threshold=30.0,
                             half_width=width).fss
              for width in (1, 4, 8, 16)]
    assert series == sorted(series)


def test_useful_skill_line_tracks_the_observed_base_rate():
    observed = _blobs()
    valid = np.ones(observed.shape, dtype=bool)
    result = fss.masked_fss(observed, observed, valid=valid, threshold=30.0,
                            half_width=2)
    assert result.fss_useful == pytest.approx(
        0.5 + result.observed_base_rate / 2.0)
    assert 0.0 < result.observed_base_rate < 1.0


def test_frequency_matching_removes_a_uniform_amplitude_bias():
    observed = _blobs()
    cold = observed - 8.0
    valid = np.ones(observed.shape, dtype=bool)

    fixed = fss.masked_fss(cold, observed, valid=valid, threshold=30.0,
                           half_width=4)
    matched = fss.masked_fss(cold, observed, valid=valid, threshold=30.0,
                             half_width=4, frequency_matched=True)

    assert fixed.model_base_rate < fixed.observed_base_rate
    assert matched.model_base_rate == pytest.approx(
        matched.observed_base_rate, abs=1.0e-3)
    assert matched.fss > fixed.fss
    assert matched.threshold_obs == 30.0
    assert matched.threshold_model < 30.0
    assert matched.frequency_matched is True


def test_the_matrix_publishes_physical_box_lengths():
    observed = _blobs()
    valid = np.ones(observed.shape, dtype=bool)
    matrix = fss.fss_matrix(observed, observed, valid=valid,
                            thresholds=[20.0, 30.0], half_widths=[1, 4])
    rows = fss.matrix_records(matrix, dx_m=3000.0)
    assert [row["box_length_m"] for row in rows] == [9000.0, 27000.0] * 2
    assert len(matrix) == 4


def test_mean_fss_refuses_an_empty_series():
    with pytest.raises(ValueError, match="at least one scored time"):
        fss.mean_fss([])


def test_masked_fss_refuses_an_unknown_boundary():
    field = np.zeros((8, 8))
    with pytest.raises(ValueError, match="unknown boundary"):
        fss.masked_fss(field, field, valid=np.ones((8, 8), dtype=bool),
                       threshold=1.0, half_width=1, boundary="reflect")


# --------------------------------------------------------------------------
# contingency
# --------------------------------------------------------------------------


def test_contingency_counts_and_scores_by_hand():
    observed = np.array([[10.0, 40.0], [40.0, 5.0]])
    forecast = np.array([[40.0, 40.0], [5.0, 5.0]])
    table = contingency.contingency_table(observed, forecast, threshold=30.0)
    assert table.record() == {
        "hits": 1, "misses": 1, "false_alarms": 1, "correct_negatives": 1,
        "total": 4}
    scores = contingency.contingency_scores(table)
    assert scores["probability_of_detection"] == pytest.approx(0.5)
    assert scores["false_alarm_ratio"] == pytest.approx(0.5)
    assert scores["critical_success_index"] == pytest.approx(1.0 / 3.0)
    assert scores["frequency_bias"] == pytest.approx(1.0)
    assert scores["heidke_skill_score"] == pytest.approx(0.0)


def test_an_undefined_score_is_none_not_zero():
    quiet = np.zeros((4, 4))
    scores = contingency.score_field(quiet, quiet, threshold=30.0)
    assert scores["probability_of_detection"] is None
    assert scores["false_alarm_ratio"] is None
    assert scores["correct_negatives"] == 16


def test_contingency_honours_the_validity_mask():
    observed = np.zeros((4, 4))
    forecast = np.zeros((4, 4))
    observed[0, 0] = 50.0
    forecast[0, 0] = 50.0
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    table = contingency.contingency_table(observed, forecast, threshold=30.0,
                                          valid=mask)
    assert table.hits == 0
    assert table.correct_negatives == 15


# --------------------------------------------------------------------------
# remapping
# --------------------------------------------------------------------------


def _mesh(center_lat, center_lon, n, spacing):
    axis = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * spacing
    lon, lat = np.meshgrid(center_lon + axis, center_lat + axis)
    return lat, lon


def test_nearest_remap_onto_the_same_grid_is_the_identity():
    lat, lon = _mesh(37.0, -97.0, 8, 0.03)
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    valid = np.ones((8, 8), dtype=bool)
    valid[3, 3] = False
    remapped, remapped_valid, plan = regrid.regrid_field(
        source_latitude=lat, source_longitude=lon,
        destination_latitude=lat, destination_longitude=lon,
        values=values, valid=valid, method=regrid.NEAREST,
        max_distance_m=1000.0)
    assert np.array_equal(remapped_valid, valid)
    assert np.array_equal(remapped[valid], values[valid])
    assert plan.max_used_distance_m == pytest.approx(0.0, abs=1.0e-6)


def test_cell_average_remap_averages_the_contained_sources():
    source_lat, source_lon = _mesh(37.0, -97.0, 6, 0.01)
    dest_lat, dest_lon = _mesh(37.0, -97.0, 2, 0.03)
    values = np.arange(36, dtype=np.float64).reshape(6, 6)
    valid = np.ones((6, 6), dtype=bool)
    remapped, remapped_valid, _plan = regrid.regrid_field(
        source_latitude=source_lat, source_longitude=source_lon,
        destination_latitude=dest_lat, destination_longitude=dest_lon,
        values=values, valid=valid, method=regrid.CELL_AVERAGE,
        max_distance_m=10000.0)
    assert remapped_valid.all()
    expected = np.array([
        [values[0:3, 0:3].mean(), values[0:3, 3:6].mean()],
        [values[3:6, 0:3].mean(), values[3:6, 3:6].mean()]])
    assert np.allclose(remapped, expected)


def test_cell_average_ignores_invalid_sources_and_marks_empty_targets():
    source_lat, source_lon = _mesh(37.0, -97.0, 6, 0.01)
    dest_lat, dest_lon = _mesh(37.0, -97.0, 2, 0.03)
    values = np.ones((6, 6), dtype=np.float64)
    values[0:3, 0:3] = 5.0
    valid = np.ones((6, 6), dtype=bool)
    valid[0:3, 0:3] = False
    remapped, remapped_valid, _plan = regrid.regrid_field(
        source_latitude=source_lat, source_longitude=source_lon,
        destination_latitude=dest_lat, destination_longitude=dest_lon,
        values=values, valid=valid, method=regrid.CELL_AVERAGE,
        max_distance_m=10000.0)
    assert not remapped_valid[0, 0]
    assert remapped_valid[0, 1] and remapped_valid[1, 0] and remapped_valid[1, 1]
    assert remapped[0, 0] == 0.0


def test_the_distance_bound_refuses_to_borrow_a_distant_observation():
    source_lat, source_lon = _mesh(37.0, -97.0, 4, 0.02)
    dest_lat, dest_lon = _mesh(37.0, -90.0, 4, 0.02)
    values = np.ones((4, 4), dtype=np.float64)
    valid = np.ones((4, 4), dtype=bool)
    _remapped, remapped_valid, plan = regrid.regrid_field(
        source_latitude=source_lat, source_longitude=source_lon,
        destination_latitude=dest_lat, destination_longitude=dest_lon,
        values=values, valid=valid, method=regrid.NEAREST,
        max_distance_m=5000.0)
    assert not remapped_valid.any()
    assert plan.record()["unreachable_destination_cells"] == 16


def test_a_plan_refuses_a_field_of_the_wrong_shape():
    lat, lon = _mesh(37.0, -97.0, 4, 0.02)
    plan = regrid.build_plan(
        source_latitude=lat, source_longitude=lon,
        destination_latitude=lat, destination_longitude=lon,
        method=regrid.NEAREST, max_distance_m=1000.0)
    with pytest.raises(ValueError, match="does not match the plan"):
        regrid.apply_plan(plan, np.zeros((5, 5)), np.ones((5, 5), dtype=bool))
