"""The sweep's two essential claims, checked mechanically.

1. The neighborhood is a square SIDE LENGTH, not a radius.  Every
   comparison in ``docs/da-vs-wofs.md`` against published WoFS skill
   turns on this, because the WoFS literature uses both conventions and
   they differ by a factor of two in linear scale.

2. The scorer uses the gallery's constants, not its own.  A sweep that
   scored itself on a different threshold or a different box than the
   figures would produce numbers that look comparable and are not.

The ensemble campaign scorer and its cross-scorer checks live in the
private campaign export. The public scoring, FSS and nest-member checks
remain here.
"""

from __future__ import annotations


import numpy as np
import pytest

# Four campaign-plan gates live in work/test_da_sweep_plan.py and run in the
# required private pretag leg. Their evidence/ plan is excluded from public source;
# the scoring, FSS, and nest-member assertions below remain public Stage 1 coverage.


def test_neighborhood_is_a_side_length_not_a_radius():
    """27 km at 3 km spacing must be 9 cells across, not 19."""

    from tools.da_sweep_score import metric_constants

    const, _ = metric_constants()
    half_width = max(1, round(const["FSS_BOX_KM"] / 2.0 / 3.0))
    assert half_width == 4
    assert 2 * half_width + 1 == 9, "the box is 9 cells across"
    assert (2 * half_width + 1) * 3.0 == 27.0, "which is 27 km ACROSS"
    # If it were a radius, the box would span 2*27/3+1 = 19 cells.
    assert 2 * half_width + 1 != 19


def test_scorer_takes_its_constants_from_the_renderer():
    from tools import da_nowcast_render as render
    from tools.da_sweep_score import metric_constants

    const, source = metric_constants()
    assert source == "tools.da_nowcast_render"
    assert const["FSS_BOX_KM"] == render.FSS_BOX_KM
    assert const["FSS_THRESHOLD_DBZ"] == render.FSS_THRESHOLD_DBZ
    assert const["MISSING_OBS_FILL_DBZ"] == render.MISSING_OBS_FILL_DBZ
    assert const["COLUMN_THRESHOLD_DBZ"] == render.COLUMN_THRESHOLD_DBZ


def test_fss_of_a_field_against_itself_is_one():
    from gpuwm.verify.field_metrics import fss_distance

    rng = np.random.default_rng(20260805)
    field = rng.normal(20.0, 12.0, size=(40, 40))
    assert fss_distance(field, field, threshold=30.0, half_width=4) == 0.0


def test_fss_is_symmetric_in_its_two_arguments():
    """Both fields are smoothed; this is the fractional-truth family."""

    from gpuwm.verify.field_metrics import fss_distance

    rng = np.random.default_rng(7)
    left = rng.normal(20.0, 12.0, size=(48, 48))
    right = rng.normal(20.0, 12.0, size=(48, 48))
    a = fss_distance(left, right, threshold=30.0, half_width=4)
    b = fss_distance(right, left, threshold=30.0, half_width=4)
    assert a == pytest.approx(b)




def test_a_nest_composite_is_not_scored_as_an_extra_member(tmp_path):
    r"""The nest is a view of a member, never an extra member.

    tools/da_cycle_prepared.py writes nest composites into the SAME
    composites/ directory as the parent members, named
    leg{NN}_{name}_d{GG}.npz.  The scorer's name pattern is
    ^leg(\d+)_(.+)\.npz$, which happily reads "leg00_3_d02" as a member
    called "3_d02".  Averaging a fine-grid field into the ensemble mean
    moves every FSS number in the sweep and raises nothing.

    This only became reachable when the nested-forecast lane and the sweep
    landed in the same tree, and the committed live-fire-3 fixtures predate
    the nest, so nothing else in this file would catch it.
    """

    from tools.da_sweep_score import member_names

    composites = tmp_path / "composites"
    composites.mkdir()
    for name in ("0", "1", "2", "control"):
        (composites / f"leg00_{name}.npz").write_bytes(b"")
    # The same three members, as seen by the nest.
    for name in ("0", "1", "2"):
        (composites / f"leg00_{name}_d02.npz").write_bytes(b"")

    assert member_names(composites, 0) == ["0", "1", "2"]


def test_the_nest_filter_does_not_eat_ordinary_member_names(tmp_path):
    """Guard the guard: a member may legitimately end in a digit."""

    from tools.da_sweep_score import member_names

    composites = tmp_path / "composites"
    composites.mkdir()
    for name in ("0", "10", "d2", "memberd02x"):
        (composites / f"leg03_{name}.npz").write_bytes(b"")

    assert member_names(composites, 3) == ["0", "10", "d2", "memberd02x"]
