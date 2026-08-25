"""Every accepted control on this branch, refused with a focused test.

CONTRIBUTING asks for "a focused failure test for every new accepted
control".  This file is the audit that found the ones that did not have
one, and it exists because the audit was MEASURED rather than eyeballed.
The four relocation modules carry 87 refusal sites; tracing which of
them the suite actually executes gave:

    before this file   87 sites, 59 exercised, 28 never reached
    after  this file   87 sites, 83 exercised,  4 never reached

Most of the 28 were cheap scalar guards.  One was not, and it is first
below: the earth-fixed compensation bound in ``relocate_child``, which
is the containment leg's own safety clamp -- the check that stops a
parent slide from pushing its earth-fixed descendant out of the frame it
lives in.  It shipped in e1a9f4b7 with nothing exercising it.

THE REMAINING FOUR ARE DELIBERATE, and they are the same four either
way: ``relocate_child``'s SINT-overlap mismatch (:1135) and
parent-changed (:1199) checks, and ``donor_alignment_check``'s two
(:1507, :1514).  Every one is an internal consistency assertion that can
fire only on a defect in this code, never on an input a config can
express.  Provoking one would mean corrupting the very state the gate
protects, and the test would then assert that a deliberately broken tree
stays broken -- which is not evidence about anything.  They are covered
instead by their PASSING side, which every relocation test in this tree
asserts on every move (``donor_alignment_pass``,
``parent_bitwise_unchanged``), and by the 492-check ledger audit over
the real 27.6-hour run.

One test was written for this file and deleted rather than kept: a
mechanical check that every refusal message is "long enough to act on".
Its word-count proxy flagged twenty perfectly good parameter guards
("parent_grid_ratio must be >= 1, got 0" is actionable and is four
words), so it measured prose length rather than usefulness and would
have fought every future contributor.  The audit is the deliverable; a
lint rule with a bad metric is not.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.core.storm_tracking as st
from gpuwm.core.nest_relocation import (Placement, RelocationRefusal,
                                        plan_relocation, relocate_child)
from gpuwm.core.relocation_runner import RelocationRunner
from gpuwm.core.storm_tracking import (FollowConfig, NestFootprint,
                                       StormTracker, TrackerRefusal,
                                       build_follow_config)

from test_nest_relocation_staging import _cpu_tree, _initializer
from test_relocation_runner import _tree3


# ---------------------------------------------------------------------------
# The containment leg's own bound (nest_relocation.py, e1a9f4b7)
# ---------------------------------------------------------------------------

def test_an_earth_fixed_compensation_off_the_frame_refuses_by_name():
    """The slide is admissible for the PARENT and inadmissible for the
    descendant it carries.

    ``[relocation.containment]``'s whole promise is that the mover does
    not move relative to the earth while its parent slides under it, and
    the compensation that delivers that is ``-shift x ratio``.  Push the
    parent far enough and the compensation walks the descendant off the
    1-based frame its placement is expressed in.  The runner clamps for
    exactly this (``_containment_opportunity``'s admissible() checks the
    compensated placement, not just the slid parent's), so reaching this
    refusal means the clamp was bypassed -- and it must say so rather
    than write a placement of 0 or -3.
    """
    parent_plane, _parent, child, d03 = _tree3()
    # d03 sits at i_parent_start = 70; a +24 d02-cell slide compensates
    # it by -24 x 3 = -72, i.e. to -2.
    with pytest.raises(RelocationRefusal,
                       match="leaves its parent's 1-based frame"):
        relocate_child(
            child,
            i_parent_start=int(child.cfg.i_parent_start) + 24,
            j_parent_start=int(child.cfg.j_parent_start),
            initializer=_initializer(parent_plane),
            static_provenance="synthetic (test)",
            state_digest=lambda _s: "digest", staging="device",
            earth_fixed_descendants=frozenset({3}))


def test_the_refusal_names_the_descendant_and_the_remedy():
    parent_plane, _parent, child, d03 = _tree3()
    with pytest.raises(RelocationRefusal) as caught:
        relocate_child(
            child,
            i_parent_start=int(child.cfg.i_parent_start) + 24,
            j_parent_start=int(child.cfg.j_parent_start),
            initializer=_initializer(parent_plane),
            static_provenance="synthetic (test)",
            state_digest=lambda _s: "digest", staging="device",
            earth_fixed_descendants=frozenset({3}))
    text = str(caught.value)
    assert "d03" in text                       # WHAT it refused
    assert "clamped" in text                   # and what should have
    assert "earth-fixed" in text               # prevented it


# ---------------------------------------------------------------------------
# The relocation primitive's input guards
# ---------------------------------------------------------------------------

def test_a_negative_placement_generation_refuses():
    with pytest.raises(RelocationRefusal, match="generation must be >= 0"):
        Placement(grid_id=2, i_parent_start=1, j_parent_start=1,
                  generation=-1)


def test_a_placement_below_one_refuses_with_the_namelist_semantics():
    with pytest.raises(RelocationRefusal, match="1-based WRF namelist"):
        Placement(grid_id=2, i_parent_start=0, j_parent_start=1)


def test_a_ratio_below_one_refuses_in_the_plan():
    with pytest.raises(RelocationRefusal, match="parent_grid_ratio must"):
        plan_relocation(placement_from=Placement(2, 1, 1),
                        placement_to=Placement(2, 2, 1),
                        parent_grid_ratio=0, child_nx=10, child_ny=10)


def test_a_plan_across_two_domains_refuses():
    with pytest.raises(RelocationRefusal, match="moves ONE domain"):
        plan_relocation(placement_from=Placement(2, 1, 1),
                        placement_to=Placement(3, 2, 1),
                        parent_grid_ratio=3, child_nx=10, child_ny=10)


def test_a_field_with_too_few_dimensions_refuses():
    """A relocation shifts (ny, nx); a 1-D field has no such window."""
    plan = plan_relocation(placement_from=Placement(2, 1, 1),
                           placement_to=Placement(2, 2, 1),
                           parent_grid_ratio=3, child_nx=10, child_ny=10)
    with pytest.raises(RelocationRefusal, match=r"at least \(ny, nx\)"):
        plan.window((7,))


# ---------------------------------------------------------------------------
# The runner's cadence arithmetic
# ---------------------------------------------------------------------------

def _schedule(period_ticks=60, tick_den=1):
    return SimpleNamespace(period_ticks=period_ticks,
                           clock=SimpleNamespace(tick_den=tick_den))


def _runner_kwargs(parent_plane):
    return dict(on_child_built=lambda *a, **k: None,
                initializer=_initializer(parent_plane),
                static_provenance="synthetic (test)",
                provider=lambda *a, **k: None)


def test_a_cadence_off_the_cycle_boundary_refuses_in_the_runner():
    """The config gate catches this first; the runner keeps its own,
    because a mismatch between the two is a DEFECT and says so."""
    from gpuwm.experiment import RelocationConfig
    parent_plane, _p, _c = _cpu_tree()
    cfg = RelocationConfig(enabled=True, grid_id=2, cadence_seconds=90.0,
                           follow=SimpleNamespace(refine_grid_id=None))
    with pytest.raises(RelocationRefusal, match="which is a defect"):
        RelocationRunner(config=cfg, schedule=_schedule(),
                         **_runner_kwargs(parent_plane))


def test_a_cadence_shorter_than_one_cycle_refuses():
    """Reachable only from a HAND-BUILT config: the loader refuses a
    non-positive cadence_seconds, so this branch guards the runner
    against an experiment object assembled in code.  Worth keeping and
    worth saying -- an unreachable-from-TOML refusal is not dead code,
    it is the second net."""
    parent_plane, _p, _c = _cpu_tree()
    cfg = SimpleNamespace(enabled=True, grid_id=2, cadence_seconds=0.0,
                          moves=(), containment=None,
                          follow=SimpleNamespace(refine_grid_id=None),
                          max_move_parent_cells=4)
    with pytest.raises(RelocationRefusal,
                       match="shorter than one complete cycle"):
        RelocationRunner(config=cfg, schedule=_schedule(),
                         **_runner_kwargs(parent_plane))


def test_a_second_programmatic_provider_refuses():
    """A config that already names its follow source must not acquire a
    second one that would silently shadow it."""
    from gpuwm.experiment import RelocationConfig
    parent_plane, _p, _c = _cpu_tree()
    exp = SimpleNamespace(relocation=RelocationConfig(
        enabled=True, grid_id=2,
        follow=FollowConfig(field="pressure", threshold=1004.0, level_hpa=0,
                            search_margin_cells=10, min_shift_cells=1,
                            max_shift_cells=4, cooldown_seconds=0.0)))
    with pytest.raises(RelocationRefusal,
                       match="already names its follow source"):
        RelocationRunner.from_experiment(
            exp, schedule=_schedule(), on_child_built=lambda *a: None,
            provider=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# [relocation.follow] scalar guards
# ---------------------------------------------------------------------------

# level_hpa = 0 is the sea-level tracker, which is what an hPa threshold
# means; omitting it now takes the 850 hPa default, where the threshold
# would be metres.
_BASE = dict(field="pressure", threshold=1004.0, level_hpa=0,
             search_margin_cells=10, min_shift_cells=1, max_shift_cells=4,
             cooldown_seconds=900.0)


@pytest.mark.parametrize("over,match", [
    ({"threshold": float("nan")}, "threshold must be finite"),
    ({"search_margin_cells": -1}, "search_margin_cells must be >= 0"),
    ({"cooldown_seconds": float("inf")}, "cooldown_seconds must be"),
    ({"cooldown_seconds": -1.0}, "cooldown_seconds must be"),
])
def test_a_follow_scalar_out_of_range_refuses(over, match):
    with pytest.raises(ValueError, match=match):
        FollowConfig(**{**_BASE, **over})


def test_a_non_finite_fallback_threshold_refuses():
    with pytest.raises(ValueError, match="fallback_threshold must be finite"):
        FollowConfig(**{**_BASE, "field": "uh", "threshold": 25.0,
                        "fallback_threshold": float("nan")})


def test_a_non_string_field_refuses_by_name():
    with pytest.raises(ValueError, match="field in \\[relocation.follow\\]"):
        build_follow_config({**_BASE, "field": 3}, "case.toml")


def test_a_refine_grid_id_below_one_refuses():
    """refine_grid_id names the DESCENDANT whose own field locates the
    vortex, so 0 is not a grid it could name."""
    with pytest.raises(ValueError, match="is not a grid id"):
        build_follow_config({**_BASE, "refine_grid_id": 0}, "case.toml")


def test_a_tracker_built_on_something_other_than_a_FollowConfig_refuses():
    with pytest.raises(TypeError, match="must be a FollowConfig"):
        StormTracker({"field": "pressure"})


def test_a_follow_block_on_a_disabled_relocation_refuses_at_the_tracker():
    """The config loader refuses this first; make_plan_provider keeps its
    own check, because reaching it means an experiment object was built
    by hand inconsistently."""
    exp = SimpleNamespace(relocation=SimpleNamespace(
        enabled=False,
        follow=FollowConfig(**_BASE)))
    with pytest.raises(TrackerRefusal, match="disabled \\[relocation\\]"):
        st.make_plan_provider(exp)


# ---------------------------------------------------------------------------
# NestFootprint geometry guards
# ---------------------------------------------------------------------------

_FP = dict(grid_id=2, i_parent_start=10, j_parent_start=10,
           child_nx=30, child_ny=30, parent_grid_ratio=3)


@pytest.mark.parametrize("over,match", [
    ({"parent_grid_ratio": 0}, "parent_grid_ratio must be >= 1"),
    ({"i_parent_start": 0}, "1-based WRF namelist semantics"),
    ({"j_parent_start": -2}, "1-based WRF namelist semantics"),
    ({"child_nx": 1}, "child extents must be >= 2"),
    ({"child_ny": 0}, "child extents must be >= 2"),
])
def test_a_bad_footprint_refuses(over, match):
    with pytest.raises(ValueError, match=match):
        NestFootprint(**{**_FP, **over})


def test_a_footprint_off_the_parent_plane_refuses():
    """The footprint and the parent state disagree about the tree
    geometry, which is a wrong answer waiting to happen rather than a
    small search box."""
    fp = NestFootprint(grid_id=2, i_parent_start=400, j_parent_start=400,
                       child_nx=30, child_ny=30, parent_grid_ratio=3)
    with pytest.raises(TrackerRefusal, match="lies outside the parent plane"):
        fp.search_box((50, 50), 4)


# ---------------------------------------------------------------------------
# The signal readers
# ---------------------------------------------------------------------------

def _column_state(nz=8, ny=12, nx=12, **drop):
    z = np.linspace(20.0, 18000.0, nz)
    p = 101000.0 * np.exp(-z / 8000.0)[:, None, None] * np.ones((nz, ny, nx))
    phi = (np.linspace(0.0, 18000.0, nz + 1)[:, None, None]
           * st.GRAVITY_M_S2 * np.ones((nz + 1, ny, nx)))
    state = SimpleNamespace(p=p, php=phi, phb=np.zeros_like(phi),
                            qv=np.zeros_like(p))
    state.total_theta = lambda: np.full_like(p, 300.0)
    for name in drop:
        setattr(state, name, None)
    return state


@pytest.mark.parametrize("missing", ["p", "php", "phb"])
def test_the_isobaric_reduction_names_the_field_it_lacks(missing):
    with pytest.raises(TrackerRefusal, match=f"carries no '{missing}'"):
        st.level_height_m_from_state(_column_state(**{missing: None}), 850.0)


@pytest.mark.parametrize("missing", ["p", "php", "phb"])
def test_the_mslp_reduction_names_the_field_it_lacks(missing):
    with pytest.raises(TrackerRefusal, match=f"carries no '{missing}'"):
        st.mslp_hpa_from_state(_column_state(**{missing: None}))


def test_a_geopotential_that_is_not_the_staggered_pair_refuses():
    """The isobaric tracker averages a staggered pair to mass points; a
    geopotential with the wrong number of levels is a different field."""
    state = _column_state()
    state.php = state.php[:-2]
    state.phb = state.phb[:-2]
    with pytest.raises(TrackerRefusal, match="levels against"):
        st.level_height_m_from_state(state, 850.0)


def test_a_level_outside_the_column_everywhere_refuses():
    """Nothing on that surface can be tracked, so it refuses rather than
    extrapolating into a height that would out-vote the real centre."""
    with pytest.raises(TrackerRefusal, match="outside the parent column"):
        # 1050 hPa is below the surface at every point of this column.
        st.level_height_m_from_state(_column_state(), 1050.0)


def test_a_column_dcomputeseaprs_cannot_reduce_refuses():
    """DCOMPUTESEAPRS needs a level 100 hPa above every surface; a
    shallow column has none, and the refusal names the reduction."""
    nz, ny, nx = 4, 8, 8
    p = np.linspace(101000.0, 99000.0, nz)[:, None, None] * np.ones(
        (nz, ny, nx))
    phi = np.linspace(0.0, 400.0, nz + 1)[:, None, None] * np.ones(
        (nz + 1, ny, nx))
    state = SimpleNamespace(p=p, php=phi, phb=np.zeros_like(phi),
                            qv=np.zeros_like(p))
    state.total_theta = lambda: np.full_like(p, 300.0)
    with pytest.raises(TrackerRefusal, match="could not be reduced"):
        st.mslp_hpa_from_state(state)


def test_reading_someone_elses_uh_window_refuses():
    """WRF's UP_HELI_MAX is zeroed by the HISTORY writer, so reading it
    would make the nest's placement a function of an output knob -- the
    defect the consumer-owned windows closed (2026-08-07)."""
    with pytest.raises(ValueError, match="consumer-owned tracking window"):
        st._plane_from_state(SimpleNamespace(), "uh", uh_slot="up_heli_max")


def test_a_signal_slot_of_the_wrong_rank_refuses():
    state = SimpleNamespace()
    setattr(state, st.UH_SLOT, np.zeros((2, 2, 2, 2), dtype=np.float32))
    with pytest.raises(TrackerRefusal, match="must be a \\(ny, nx\\) plane"):
        st._plane_from_state(state, "uh")


# ---------------------------------------------------------------------------
# The TWO-DOMAIN tree: one parent, one moving nest -- WRF's own shape
# ---------------------------------------------------------------------------
# This is the configuration WRF's moving nest is usually run in, and it
# is supported: the tracker searches d01, `refine_grid_id = 2` refines on
# the mover itself (the degenerate chain, pinned in
# test_storm_tracking_refinement.py), and PARENT_EDGE_KEEPOUT_CELLS keeps
# the nest clear of d01's edge.
#
# What it CANNOT have is [relocation.containment], because containment
# slides a strict ancestor of the mover and the mover's only ancestor
# here is the root.  Both halves of that refuse by name, and neither had
# a test.

def test_containment_on_the_root_refuses_by_name():
    """A two-domain tree has nothing to slide: d02's only ancestor is
    d01, and the root has no placement of its own."""
    from gpuwm.experiment import ContainmentConfig
    with pytest.raises(ValueError, match="names the root"):
        ContainmentConfig(grid_id=1)


def test_containment_naming_the_mover_itself_refuses_by_name():
    """The other way to write the same mistake in a two-domain tree."""
    from gpuwm.experiment import ContainmentConfig, RelocationConfig
    with pytest.raises(ValueError, match="STRICT ANCESTOR"):
        RelocationConfig(
            enabled=True, grid_id=2,
            follow=build_follow_config(_FOLLOW_2D, "x"),
            containment=ContainmentConfig(grid_id=2))


#: A follow block for a two-domain tree: stage one on d01, stage two on
#: the mover itself.
_FOLLOW_2D = {
    "field": "pressure", "level_hpa": 850.0, "threshold": 30.0,
    "search_margin_cells": 8, "refine_grid_id": 2,
    "min_shift_cells": 1, "max_shift_cells": 2, "cooldown_seconds": 300.0,
}


def test_a_two_domain_tracker_validates_and_refines_on_the_mover():
    """The whole two-domain shape, accepted: no containment table, and
    refine_grid_id naming the mover is legal rather than a special case."""
    from gpuwm.experiment import RelocationConfig
    cfg = RelocationConfig(enabled=True, grid_id=2,
                           follow=build_follow_config(_FOLLOW_2D, "x"))
    assert cfg.containment is None
    assert cfg.follow.refine_grid_id == 2
    assert cfg.receipt()["containment"] is None
