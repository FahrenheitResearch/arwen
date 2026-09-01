"""The structure diagnostics, and the one claim they are there to settle.

A neighborhood score can be won by a field that has diffused its storms
into one smooth patch of above-threshold area, so the scorer of record now
reports object statistics, a radial power spectrum and an intensity
distribution beside FSS.  Those numbers immediately invite a wrong reading:
an ensemble MEAN is smooth because averaging is a spatial filter, not
because its members are smooth, and reporting the mean's object count as
the model's object count would be a measurement of the averaging.

The load-bearing test here is therefore not a recovery test.  It is a
matched pair:

* members that disagree about where their cells are must produce a mean
  with strictly less small-scale variance than any of them -- the claim the
  gallery figure makes; and
* members that agree exactly must produce a mean whose every structure
  number is identical to theirs.

The second is the falsification control.  Without it the first could be
satisfied by any function that returns a smaller number for a field called
"mean", including a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.da_sweep_score import (ENSEMBLE_MEAN_STRUCTURE_WARNING,
                                  HISTOGRAM_EDGES_DBZ, OBJECT_MIN_AREA_CELLS,
                                  STRUCTURE_CITATIONS, intensity_distribution,
                                  label_objects, member_spread,
                                  object_statistics, spectral_profile,
                                  spectral_report, structure_block,
                                  structure_means)

DX_KM = 3.0


# --------------------------------------------------------------------------
# connected components
# --------------------------------------------------------------------------


def test_two_separated_blocks_are_two_objects():
    mask = np.zeros((20, 20), bool)
    mask[2:6, 2:6] = True
    mask[12:16, 12:16] = True
    labels = label_objects(mask)
    assert labels.max() == 2
    assert sorted(np.bincount(labels.ravel())[1:]) == [16, 16]


def test_diagonal_touch_is_one_object_at_eight_connectivity():
    mask = np.zeros((10, 10), bool)
    mask[2:4, 2:4] = True
    mask[4:6, 4:6] = True          # touches the first only at a corner
    assert label_objects(mask, connectivity=8).max() == 1
    assert label_objects(mask, connectivity=4).max() == 2


def test_a_u_shape_is_one_object_not_three():
    """The union-find has to merge two arms through their base."""

    mask = np.zeros((12, 12), bool)
    mask[2:10, 2:4] = True
    mask[2:10, 8:10] = True
    mask[8:10, 2:10] = True
    assert label_objects(mask).max() == 1


def test_an_empty_mask_has_no_objects():
    assert label_objects(np.zeros((8, 8), bool)).max() == 0


@pytest.mark.parametrize("connectivity", [0, 5, 9])
def test_only_four_and_eight_connectivity_are_accepted(connectivity):
    with pytest.raises(ValueError):
        label_objects(np.zeros((8, 8), bool), connectivity=connectivity)


# --------------------------------------------------------------------------
# object statistics
# --------------------------------------------------------------------------


def _cells(shape, boxes, value=50.0, background=0.0):
    field = np.full(shape, background, float)
    for j0, i0, size in boxes:
        field[j0:j0 + size, i0:i0 + size] = value
    return field


def test_specks_below_the_area_floor_are_dropped():
    field = _cells((40, 40), [(4, 4, 6)])
    field[30, 30] = 50.0                      # a one-cell speck
    stats = object_statistics(field, dx_km=DX_KM)
    assert stats["count_before_area_filter"] == 2
    assert stats["count"] == 1, "the speck is under the area floor"
    assert stats["min_area_cells"] == OBJECT_MIN_AREA_CELLS


def test_nearest_neighbour_separation_is_the_centroid_distance():
    """Two blocks whose centres are 20 cells apart, at 3 km, are 60 km."""

    field = _cells((60, 60), [(5, 5, 4), (25, 5, 4)])
    stats = object_statistics(field, dx_km=DX_KM)
    assert stats["count"] == 2
    assert stats["mean_nearest_neighbor_km"] == pytest.approx(60.0, abs=1e-6)


def test_one_object_has_no_separation_and_all_the_area():
    field = _cells((30, 30), [(5, 5, 6)])
    stats = object_statistics(field, dx_km=DX_KM)
    assert stats["count"] == 1
    assert stats["mean_nearest_neighbor_km"] is None
    assert stats["largest_object_area_fraction"] == 1.0


def test_a_field_below_the_threshold_has_no_objects():
    stats = object_statistics(np.full((30, 30), 30.0), dx_km=DX_KM)
    assert stats["count"] == 0
    assert stats["total_area_cells"] == 0
    assert stats["largest_object_area_fraction"] is None


# --------------------------------------------------------------------------
# spectrum
# --------------------------------------------------------------------------


def _grid(n=64):
    y, x = np.mgrid[0:n, 0:n]
    return y.astype(float), x.astype(float)


#: Wavelength of the texture added by ``ripple``, in grid intervals.  It
#: sits inside the 2-4dx band, which is the band a blurred field empties
#: first and therefore the band the spectral tests have to be able to see.
RIPPLE_DX = 3.0


def _storm_field(centres, n=64, radius=3.0, peak=70.0, floor=10.0,
                 ripple=0.0):
    """Gaussian cells on a background that is already above the echo floor.

    The background matters.  Every field is clipped from below at
    ``ECHO_FLOOR_DBZ`` before its spectrum is taken, so that two sources
    with different no-echo encodings cannot differ on that alone; a
    synthetic field whose background sits BELOW the floor would have its
    small-scale power dominated by the clip edge rather than by its cells,
    and the test would be measuring the clip.

    ``ripple`` adds a ``RIPPLE_DX``-wavelength texture.  Smooth Gaussian
    bumps have almost no 2-4dx variance to begin with, so a ratio taken
    over that band on a bump-only field is a ratio of two numbers near
    zero and means nothing; the spectral tests need a field that actually
    holds variance where they are looking.
    """

    y, x = _grid(n)
    field = np.full((n, n), floor, float)
    for cy, cx in centres:
        bump = floor + (peak - floor) * np.exp(
            -(((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * radius ** 2)))
        field = np.maximum(field, bump)
    if ripple:
        field = field + ripple * (np.sin(2 * np.pi * y / RIPPLE_DX)
                                  * np.sin(2 * np.pi * x / RIPPLE_DX))
    return field


def test_a_field_against_itself_resolves_every_scale():
    field = _storm_field([(16, 16), (16, 48), (48, 16), (48, 48)],
                         ripple=4.0)
    profile = spectral_profile(field, dx_km=DX_KM)
    report = spectral_report(profile, profile)
    assert report["power_ratio_2_4dx"] == pytest.approx(1.0, abs=1e-9)
    assert report["power_ratio_ge_10dx"] == pytest.approx(1.0, abs=1e-9)
    # Nothing ever falls below the reference, so the finest retained bin is
    # still resolved.
    assert report["effective_resolution_dx"] < 2.5


def test_a_smoothed_field_loses_the_small_scales_and_says_so():
    field = _storm_field([(16, 16), (16, 48), (48, 16), (48, 48)],
                         ripple=4.0)
    smoothed = _smooth(field, passes=2)
    reference = spectral_profile(field, dx_km=DX_KM)
    report = spectral_report(spectral_profile(smoothed, dx_km=DX_KM),
                             reference)
    assert report["power_ratio_2_4dx"] < 0.1
    # and the loss is worse at the smallest scales than at the largest,
    # which is what makes this a blurriness diagnostic rather than an
    # amplitude one
    assert report["power_ratio_2_4dx"] < report["power_ratio_4_10dx"]
    assert report["power_ratio_4_10dx"] < report["power_ratio_ge_10dx"]
    assert report["effective_resolution_dx"] > 4.0


def _smooth(field, passes=1):
    out = np.asarray(field, float)
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = sum(padded[1 + dy:1 + dy + out.shape[0],
                         1 + dx:1 + dx + out.shape[1]]
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    return out


def test_band_variance_fractions_sum_to_one():
    field = _storm_field([(20, 20), (44, 44)], ripple=4.0)
    report = spectral_report(spectral_profile(field, dx_km=DX_KM))
    total = sum(report[f"variance_fraction_{name}"]
                for name in ("2_4dx", "4_10dx", "ge_10dx"))
    assert total == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------
# intensity distribution
# --------------------------------------------------------------------------


def test_a_field_overlaps_its_own_histogram_completely():
    field = _storm_field([(20, 20), (44, 44)])
    stats = intensity_distribution(field, field)
    assert stats["histogram_overlap"] == pytest.approx(1.0)
    for key in ("p50_dbz_bias", "p90_dbz_bias", "p99_dbz_bias"):
        assert stats[key] == pytest.approx(0.0)


def test_a_weaker_field_is_caught_by_the_quantiles():
    field = _storm_field([(20, 20), (44, 44)], peak=60.0)
    weak = _storm_field([(20, 20), (44, 44)], peak=45.0)
    stats = intensity_distribution(weak, field)
    assert stats["p99_dbz_bias"] < -5.0
    assert stats["histogram_overlap"] < 1.0
    assert stats["max_dbz"] < 60.0


def test_the_histogram_bins_are_the_published_ones():
    field = _storm_field([(20, 20)])
    stats = intensity_distribution(field, field)
    assert stats["histogram_edges_dbz"] == list(HISTOGRAM_EDGES_DBZ)


# --------------------------------------------------------------------------
# THE CLAIM: averaging is a filter, and the control that proves it
# --------------------------------------------------------------------------


SHARED_CELLS = ((32, 20), (32, 32), (32, 44), (20, 32), (44, 32))


def _displaced_ensemble(count=8, shift=3):
    """One storm field, translated by a member-specific offset.

    Every member holds exactly the same structure; only its position
    differs.  So any structure difference between a member and the mean of
    the members is produced by the averaging and by nothing else, which is
    what makes this the right construction for the claim.
    """

    members = {}
    for k in range(count):
        angle = 2.0 * np.pi * k / count
        dy = shift * np.cos(angle)
        dx = shift * np.sin(angle)
        members[str(k)] = _storm_field([(cy + dy, cx + dx)
                                        for cy, cx in SHARED_CELLS],
                                       ripple=4.0)
    return members


def _disagreeing_ensemble(count=8, extra=3):
    """Members that agree on a core system and disagree on the fringe.

    Each member carries the same five cells plus ``extra`` of its own.  The
    mean keeps the five every member has and drops the rest below the
    threshold, because a cell one member in eight forecasts arrives in the
    average at an eighth of its amplitude.  That is the real mechanism
    behind an ensemble mean holding fewer objects than its members, and it
    is a property of averaging rather than of the model.
    """

    members = {}
    for k in range(count):
        own = [(12 + 8 * ((k + j) % 5), 8 + 11 * ((k * 2 + j) % 5))
               for j in range(extra)]
        members[str(k)] = _storm_field(list(SHARED_CELLS) + own)
    return members


def test_the_mean_of_disagreeing_members_holds_less_small_scale_variance():
    members = _displaced_ensemble()
    mean = np.mean(list(members.values()), axis=0)
    observed = members["0"]

    reference = spectral_profile(observed, dx_km=DX_KM)
    member_ratios = [
        spectral_report(spectral_profile(field, dx_km=DX_KM),
                        reference)["power_ratio_2_4dx"]
        for field in members.values()]
    mean_ratio = spectral_report(spectral_profile(mean, dx_km=DX_KM),
                                 reference)["power_ratio_2_4dx"]

    assert mean_ratio < min(member_ratios), (
        "the ensemble mean must hold strictly less 2-4dx variance than any "
        "member it was averaged from; if it does not, the spectral "
        "diagnostic is not measuring what the figure says it measures")


def test_the_mean_keeps_only_the_objects_the_members_agree_on():
    members = _disagreeing_ensemble()
    mean = np.mean(list(members.values()), axis=0)
    member_counts = [object_statistics(f, dx_km=DX_KM)["count"]
                     for f in members.values()]
    mean_stats = object_statistics(mean, dx_km=DX_KM)

    assert min(member_counts) > len(SHARED_CELLS), (
        "the construction is wrong if a member does not carry its own cells")
    assert mean_stats["count"] < min(member_counts), (
        "the ensemble mean must hold FEWER objects than any member: the "
        "cells only some members forecast arrive in the average at a "
        "fraction of their amplitude and fall under the threshold")
    assert mean_stats["count"] == len(SHARED_CELLS), (
        "and the ones it keeps are exactly the ones every member had")
    assert mean_stats["total_area_cells"] < min(
        object_statistics(f, dx_km=DX_KM)["total_area_cells"]
        for f in members.values()), (
        "and it holds less above-threshold area than any member, so the "
        "loss is objects and area together rather than a relabelling")


def test_identical_members_leave_the_mean_identical_to_them():
    """The falsification control for the two tests above.

    If members do not disagree there is nothing for the averaging to
    filter, so every structure number of the mean must equal a member's
    exactly.  A diagnostic that reported the mean as smoother here would be
    reporting a property of the word "mean" rather than of the field.
    """

    field = _storm_field([(16, 16), (16, 44), (44, 16), (44, 44)])
    members = {str(k): field.copy() for k in range(6)}
    block = structure_block(observed=field, dx_km=DX_KM,
                            ensemble_mean=np.mean(list(members.values()),
                                                  axis=0),
                            members=members)
    mean_block = block["ensemble_mean"]
    one_member = block["members"]["0"]
    assert mean_block["objects"] == one_member["objects"]
    assert mean_block["spectrum"]["power_ratio_2_4dx"] == pytest.approx(
        one_member["spectrum"]["power_ratio_2_4dx"], abs=1e-9)
    assert mean_block["distribution"]["p99_dbz"] == pytest.approx(
        one_member["distribution"]["p99_dbz"], abs=1e-9)
    assert mean_block["spectrum"]["power_ratio_2_4dx"] == pytest.approx(
        1.0, abs=1e-9), "and both equal the observation they came from"


# --------------------------------------------------------------------------
# the block, and the labelling that keeps the mean apart from the members
# --------------------------------------------------------------------------


def test_the_block_labels_the_mean_and_keeps_the_members_separate():
    members = _displaced_ensemble(count=4)
    block = structure_block(
        observed=members["0"], dx_km=DX_KM,
        ensemble_mean=np.mean(list(members.values()), axis=0),
        members=members, extra={"control": members["1"]})

    assert block["ensemble_mean"]["is_ensemble_average"] is True
    assert block["ensemble_mean"]["warning"] == \
        ENSEMBLE_MEAN_STRUCTURE_WARNING
    assert set(block["members"]) == set(members)
    assert "is_ensemble_average" not in block["members"]["0"]
    assert "control" in block
    assert block["settings"]["dx_km"] == DX_KM
    assert block["settings"]["spectral_pins_sha256"]
    spread = block["member_spread"]["objects.count"]
    assert spread["min"] <= spread["median"] <= spread["max"]
    assert spread["n"] == len(members)


def test_the_observed_block_carries_no_ratio_against_itself():
    field = _storm_field([(20, 20), (44, 44)])
    block = structure_block(observed=field, dx_km=DX_KM)
    assert "power_ratio_2_4dx" not in block["observed"]["spectrum"]
    assert "histogram_overlap" not in block["observed"]["distribution"]


def test_member_spread_ignores_a_statistic_no_member_defines():
    """A single-object member has no nearest-neighbour distance."""

    field = _storm_field([(20, 20)])
    members = {str(k): field.copy() for k in range(3)}
    block = structure_block(observed=field, dx_km=DX_KM, members=members)
    spread = member_spread(block["members"])
    assert "objects.count" in spread
    assert "objects.mean_nearest_neighbor_km" not in spread


def test_structure_means_averages_frames_and_keeps_the_warning():
    members = _displaced_ensemble(count=4)
    frames = [{"structure": structure_block(
        observed=members["0"], dx_km=DX_KM,
        ensemble_mean=np.mean(list(members.values()), axis=0),
        members=members)} for _ in range(3)]
    means = structure_means(frames)
    assert means["warning"] == ENSEMBLE_MEAN_STRUCTURE_WARNING
    assert means["observed"]["objects.count"] == \
        frames[0]["structure"]["observed"]["objects"]["count"]
    assert "members" in means
    assert set(means["members"]["objects.count"]) == {"min", "median", "max"}


def test_every_diagnostic_carries_a_citation():
    for key in ("objects", "nearest_neighbor", "spectrum", "distribution",
                "ensemble_mean_caveat"):
        assert key in STRUCTURE_CITATIONS
        assert len(STRUCTURE_CITATIONS[key]) > 40


def test_structure_is_json_serialisable():
    import json

    members = _displaced_ensemble(count=3)
    block = structure_block(
        observed=members["0"], dx_km=DX_KM,
        ensemble_mean=np.mean(list(members.values()), axis=0),
        members=members)
    json.dumps(block)          # raises if a numpy scalar leaked through
