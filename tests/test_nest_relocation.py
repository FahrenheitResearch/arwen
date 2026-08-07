"""CPU contracts for discrete nest relocation (placement, plan, transplant).

The instrument rule applies throughout: every property is checked in BOTH
directions.  A shift test that only asserts "the overlap matches" passes
just as happily when the shift is zero and nothing moved, so each one is
paired with a control that must FAIL under a wrong shift.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.core.nest_relocation import (Placement, RelocationRefusal,
                                        RelocationSegment,
                                        check_admissible,
                                        donor_alignment_check,
                                        plan_relocation, relocatable_attrs,
                                        transplant_overlap)
from gpuwm.experiment import DISCRETE_RELOCATION_MODE, RelocationConfig


RATIO = 3
NX = NY = 24


def _placement(i, j, generation=0):
    return Placement(grid_id=2, i_parent_start=i, j_parent_start=j,
                     generation=generation)


def _plan(di=0, dj=0, nx=NX, ny=NY, ratio=RATIO):
    return plan_relocation(
        placement_from=_placement(10, 10),
        placement_to=_placement(10 + di, 10 + dj, 1),
        parent_grid_ratio=ratio, child_nx=nx, child_ny=ny)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def test_placement_rejects_zero_based_positions():
    """1-based WRF namelist semantics, refused at the boundary not silently."""
    with pytest.raises(RelocationRefusal, match="1-based"):
        Placement(grid_id=2, i_parent_start=0, j_parent_start=5)


def test_placement_position_equality_ignores_generation():
    a = _placement(10, 10, generation=0)
    b = _placement(10, 10, generation=7)
    assert a.same_position(b)
    assert a != b            # the generation is still part of identity
    assert not a.same_position(_placement(11, 10, generation=0))


# ---------------------------------------------------------------------------
# The plan: index-space geometry
# ---------------------------------------------------------------------------

def test_shift_is_whole_parent_cells_times_the_ratio():
    plan = _plan(di=4, dj=-2)
    assert (plan.shift_i, plan.shift_j) == (12, -6)
    assert not plan.null_move


def test_null_move_covers_the_whole_child():
    plan = _plan(di=0, dj=0)
    assert plan.null_move
    assert plan.overlap_cells == NX * NY
    assert plan.overlap_fraction == 1.0
    (dst_j, src_j), (dst_i, src_i) = plan.window((NY, NX))
    assert dst_j == src_j == slice(0, NY)
    assert dst_i == src_i == slice(0, NX)


def test_overlap_shrinks_by_exactly_the_shift():
    plan = _plan(di=2)                       # 6 child cells east
    assert plan.overlap_cells == (NX - 6) * NY
    plan = _plan(di=2, dj=1)                 # 6 east, 3 north
    assert plan.overlap_cells == (NX - 6) * (NY - 3)


def test_window_tracks_the_staggered_extent_not_the_mass_extent():
    """A +1 face extent shifts by the same whole number of child cells."""
    plan = _plan(di=2)
    (_, _), (dst_i, src_i) = plan.window((NY, NX))
    (_, _), (xdst_i, xsrc_i) = plan.window((7, NY, NX + 1))
    assert (dst_i.start, dst_i.stop) == (0, NX - 6)
    assert (xdst_i.start, xdst_i.stop) == (0, NX + 1 - 6)
    assert src_i.start == xsrc_i.start == 6


def test_westward_and_eastward_moves_are_mirror_images():
    east = _plan(di=2)
    west = _plan(di=-2)
    assert east.shift_i == -west.shift_i
    assert east.overlap_cells == west.overlap_cells
    (_, _), (edst, esrc) = east.window((NY, NX))
    (_, _), (wdst, wsrc) = west.window((NY, NX))
    # East reads from high source indices into low destination ones; west
    # is the reverse.  If the sign were dropped both would look identical.
    assert (edst.start, esrc.start) == (0, 6)
    assert (wdst.start, wsrc.start) == (6, 0)


def test_a_move_past_the_whole_child_is_disjoint():
    plan = _plan(di=NX)          # NX parent cells = NX*3 child cells
    assert plan.disjoint
    assert plan.overlap_cells == 0
    assert plan.window((NY, NX)) is None


def test_plan_refuses_to_move_two_different_domains():
    with pytest.raises(RelocationRefusal, match="ONE domain"):
        plan_relocation(
            placement_from=Placement(grid_id=2, i_parent_start=1,
                                     j_parent_start=1),
            placement_to=Placement(grid_id=3, i_parent_start=2,
                                   j_parent_start=1),
            parent_grid_ratio=RATIO, child_nx=NX, child_ny=NY)


# ---------------------------------------------------------------------------
# The transplant, against a synthetic state with a known answer
# ---------------------------------------------------------------------------

def _ramp_state(nx=NX, ny=NY, nz=4, offset=0.0):
    """A state whose every cell is distinguishable from every other."""
    def ramp(shape):
        return (np.arange(int(np.prod(shape)), dtype=np.float32)
                .reshape(shape) + np.float32(offset))
    return SimpleNamespace(
        u=ramp((nz, ny, nx + 1)), v=ramp((nz, ny + 1, nx)),
        w=ramp((nz + 1, ny, nx)), thp=ramp((nz, ny, nx)),
        php=ramp((nz + 1, ny, nx)), mup=ramp((ny, nx)),
        qv=ramp((nz, ny, nx)))


_ATTRS = ("u", "v", "w", "thp", "php", "mup", "qv")


def test_null_move_transplant_is_the_identity():
    source = _ramp_state()
    target = _ramp_state(offset=1000.0)
    plan = _plan(di=0, dj=0)
    receipt = transplant_overlap(source_state=source, target_state=target,
                                 plan=plan, attrs=_ATTRS)
    assert receipt["stamped_field_count"] == len(_ATTRS)
    for name in _ATTRS:
        actual = getattr(target, name).view(np.uint32)
        expected = getattr(source, name).view(np.uint32)
        assert np.array_equal(actual, expected), name


def test_moved_transplant_is_bitwise_on_the_overlap_and_untouched_outside():
    source = _ramp_state()
    target = _ramp_state(offset=1000.0)
    cold = {name: getattr(target, name).copy() for name in _ATTRS}
    plan = _plan(di=2)                        # 6 child cells east
    transplant_overlap(source_state=source, target_state=target, plan=plan,
                       attrs=_ATTRS)
    for name in _ATTRS:
        moved = getattr(target, name)
        original = getattr(source, name)
        extent = moved.shape[-1]
        keep = extent - 6
        # Overlap: bit-identical to the shifted source.
        assert np.array_equal(moved[..., :keep].view(np.uint32),
                              original[..., 6:].view(np.uint32)), name
        # Strip: still exactly what the cold start produced.
        assert np.array_equal(moved[..., keep:].view(np.uint32),
                              cold[name][..., keep:].view(np.uint32)), name


def test_the_overlap_check_fails_under_a_wrong_shift():
    """The instrument's negative control: a shift of the wrong sign must
    NOT satisfy the same equality the correct one does."""
    source = _ramp_state()
    target = _ramp_state(offset=1000.0)
    transplant_overlap(source_state=source, target_state=target,
                       plan=_plan(di=-2), attrs=_ATTRS)
    thp = target.thp
    keep = thp.shape[-1] - 6
    assert not np.array_equal(thp[..., :keep], source.thp[..., 6:])


def test_transplant_refuses_a_shape_change():
    source = _ramp_state()
    target = _ramp_state(nx=NX + 3)
    with pytest.raises(RelocationRefusal, match="never extent"):
        transplant_overlap(source_state=source, target_state=target,
                           plan=_plan(di=1), attrs=("thp",))


def test_transplant_records_a_field_absent_on_one_side():
    source = _ramp_state()
    target = _ramp_state(offset=1.0)
    target.qv = None
    receipt = transplant_overlap(source_state=source, target_state=target,
                                 plan=_plan(di=1), attrs=_ATTRS)
    assert "qv" in receipt["skipped"]
    assert "thp" in receipt["stamped"]


def test_relocatable_inventory_is_the_restart_contract():
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    assert relocatable_attrs() == tuple(STATE_SERIALIZED_ATTRS)


# ---------------------------------------------------------------------------
# The donor-alignment instrument, proved against a known answer both ways
# ---------------------------------------------------------------------------

def _sint_like(parent, placement, ratio, nx, ny):
    """Nearest-donor pickup, which is SINT's ci/ip map without weights.

    Enough to reproduce the property under test: a child cell's value is a
    function of its donor parent cell alone.
    """
    ci = (placement.i_parent_start - 1) + np.arange(nx) // ratio
    cj = (placement.j_parent_start - 1) + np.arange(ny) // ratio
    return parent[np.ix_(cj, ci)].astype(np.float32)


def test_donor_alignment_passes_for_a_whole_parent_cell_move():
    parent = np.arange(60 * 60, dtype=np.float32).reshape(60, 60)
    p0, p1 = _placement(10, 10), _placement(14, 10, 1)
    old = SimpleNamespace(pb=_sint_like(parent, p0, RATIO, NX, NY))
    new = SimpleNamespace(pb=_sint_like(parent, p1, RATIO, NX, NY))
    plan = plan_relocation(placement_from=p0, placement_to=p1,
                           parent_grid_ratio=RATIO, child_nx=NX, child_ny=NY)
    report = donor_alignment_check(source_state=old, target_state=new,
                                   plan=plan)
    assert report["pass"]
    assert report["fields"]["pb"]["bit_mismatches"] == 0


def test_donor_alignment_fails_when_the_shift_is_wrong():
    """Prove the instrument can fail: mis-state the move by one parent cell
    and the SINT-derived overlap must stop agreeing."""
    parent = np.arange(60 * 60, dtype=np.float32).reshape(60, 60)
    p0, p1 = _placement(10, 10), _placement(14, 10, 1)
    old = SimpleNamespace(pb=_sint_like(parent, p0, RATIO, NX, NY))
    new = SimpleNamespace(pb=_sint_like(parent, p1, RATIO, NX, NY))
    wrong = plan_relocation(
        placement_from=p0, placement_to=_placement(15, 10, 1),
        parent_grid_ratio=RATIO, child_nx=NX, child_ny=NY)
    report = donor_alignment_check(source_state=old, target_state=new,
                                   plan=wrong)
    assert not report["pass"]
    assert report["fields"]["pb"]["bit_mismatches"] > 0


# ---------------------------------------------------------------------------
# The append-only segment -- receipt bookkeeping, not a restart contract.
# A restart across a move promises nothing (Drew, 2026-08-06), so these
# pin that the history READS correctly; none of them licenses a resume.
# ---------------------------------------------------------------------------

def _record(di):
    from gpuwm.core.nest_relocation import RelocationRecord

    plan = _plan(di=di)
    return RelocationRecord(
        placement_from=plan.placement_from, placement_to=plan.placement_to,
        parent_grid_ratio=RATIO, shift_i=plan.shift_i, shift_j=plan.shift_j,
        overlap_cells=plan.overlap_cells, child_cells=plan.child_cells,
        null_move=plan.null_move)


def test_segment_id_changes_with_every_move_but_the_base_does_not():
    base = RelocationSegment(base_identity_sha256="a" * 64)
    one = base.append(_record(1))
    two = one.append(_record(2))
    assert base.base_identity_sha256 == two.base_identity_sha256
    assert len({base.segment_id, one.segment_id, two.segment_id}) == 3
    assert (base.generation, one.generation, two.generation) == (0, 1, 2)


def test_records_chain_to_their_predecessor():
    base = RelocationSegment(base_identity_sha256="a" * 64)
    one = base.append(_record(1))
    two = one.append(_record(2))
    assert one.records[0].predecessor_sha256 == base.base_identity_sha256
    assert two.records[1].predecessor_sha256 == one.records[0].sha256


def test_reordering_two_moves_produces_a_different_segment():
    """An append-only chain must not be commutative, or it records a set
    rather than a history."""
    base = RelocationSegment(base_identity_sha256="a" * 64)
    forward = base.append(_record(1)).append(_record(2))
    backward = base.append(_record(2)).append(_record(1))
    assert forward.segment_id != backward.segment_id


def test_record_reports_the_spin_up_strip():
    record = _record(2)
    assert record.overlap_cells == (NX - 6) * NY
    assert record.spin_up_cells == NX * NY - record.overlap_cells
    assert 0.0 < record.overlap_fraction < 1.0


# ---------------------------------------------------------------------------
# Admissibility bounds
# ---------------------------------------------------------------------------

def test_a_disabled_config_refuses_a_real_move_but_allows_the_null_one():
    off = RelocationConfig()
    assert check_admissible(_plan(di=0), off)["null_move"] is True
    with pytest.raises(RelocationRefusal, match="opt-in"):
        check_admissible(_plan(di=1), off)


def test_bounds_refuse_an_over_long_move_and_admit_one_at_the_limit():
    bounds = RelocationConfig(enabled=True, grid_id=2,
                              max_move_parent_cells=4)
    assert check_admissible(_plan(di=4), bounds)["admissible"]
    with pytest.raises(RelocationRefusal, match="over the configured"):
        check_admissible(_plan(di=5), bounds)


def test_bounds_refuse_a_move_that_keeps_too_little_overlap():
    bounds = RelocationConfig(enabled=True, grid_id=2,
                              min_overlap_fraction=0.75)
    assert check_admissible(_plan(di=1), bounds)["admissible"]
    with pytest.raises(RelocationRefusal, match="under the configured floor"):
        check_admissible(_plan(di=3), bounds)


def test_bounds_refuse_a_move_targeting_an_unauthorised_domain():
    bounds = RelocationConfig(enabled=True, grid_id=3)
    with pytest.raises(RelocationRefusal, match="authorises grid_id 3"):
        check_admissible(_plan(di=1), bounds)


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_relocation_config_defaults_to_off():
    cfg = RelocationConfig()
    assert cfg.enabled is False
    assert cfg.mode == DISCRETE_RELOCATION_MODE


def test_enabled_relocation_must_name_a_child():
    with pytest.raises(ValueError, match="must name the grid_id"):
        RelocationConfig(enabled=True)
    with pytest.raises(ValueError, match="only a child can be relocated"):
        RelocationConfig(enabled=True, grid_id=1)


def test_only_the_discrete_mode_is_implemented():
    with pytest.raises(ValueError, match="not implemented"):
        RelocationConfig(enabled=True, grid_id=2, mode="continuous")


# ---------------------------------------------------------------------------
# The new config field must not move any existing fingerprint
# ---------------------------------------------------------------------------

def _scaffold():
    from gpuwm.verify.cases.nest_ideal_r3 import load_scaffold

    return load_scaffold()


def test_relocation_bounds_are_outside_the_restart_identity():
    """Adding a field to ExperimentConfig would otherwise re-key every
    checkpoint ever written, including for runs with no nest at all."""
    from gpuwm.core.model import restart_identity_payload

    exp = _scaffold()
    moving = replace(exp, relocation=RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=4))
    assert "relocation" not in restart_identity_payload(exp)
    assert restart_identity_payload(exp) == restart_identity_payload(moving)


def test_relocation_bounds_do_not_move_the_experiment_fingerprint():
    from gpuwm.core.model import experiment_fingerprint

    exp = _scaffold()
    moving = replace(exp, relocation=RelocationConfig(
        enabled=True, grid_id=2, min_overlap_fraction=0.5))
    catalog = SimpleNamespace(run_provenance={})
    assert (experiment_fingerprint(exp, catalog)
            == experiment_fingerprint(moving, catalog))


def test_the_fingerprint_still_moves_for_a_real_geometry_change():
    """The control: prove the fingerprint is not simply insensitive."""
    from gpuwm.core.model import experiment_fingerprint

    exp = _scaffold()
    catalog = SimpleNamespace(run_provenance={})
    child = exp.domains[1]
    moved = replace(exp, domains=(
        exp.domains[0], replace(child, i_parent_start=child.i_parent_start + 1)))
    assert (experiment_fingerprint(exp, catalog)
            != experiment_fingerprint(moved, catalog))


# ---------------------------------------------------------------------------
# Layer 1 identity: position factored out
# ---------------------------------------------------------------------------

def _two_placements_of_one_domain():
    from gpuwm.verify.cases.nest_ideal_r3 import load_scaffold

    exp = load_scaffold()
    child = exp.domains[1]
    return child, replace(child, i_parent_start=child.i_parent_start + 7)


def test_placement_independent_identity_is_equal_across_a_move():
    from gpuwm.core.nest_relocation import placement_independent_identity

    here, there = _two_placements_of_one_domain()
    assert (placement_independent_identity(here)["identity"]
            == placement_independent_identity(there)["identity"])


def test_the_full_prepared_identity_still_differs_across_a_move():
    """The RULED behaviour, pinned: a move invalidates the cache key.

    A restart across a move promises nothing (Drew, 2026-08-06), so this
    is the intended outcome rather than a blocker to be lifted later.  If
    it ever passes as equal, something has started claiming a resume
    across a placement boundary is meaningful.
    """
    from gpuwm.ingest.prepared_cache import prepared_domain_config_identity

    here, there = _two_placements_of_one_domain()
    assert (prepared_domain_config_identity(here)
            != prepared_domain_config_identity(there))


def test_a_move_invalidates_the_tree_restart_fingerprint():
    """The other half of refuse-across-move, at the restart layer.

    Together with the prepared-cache pin above and the bounds test in the
    fingerprint section, this fixes the whole promise: relocation bounds
    are byte-inert on the fingerprint, and an actual relocation is not.
    """
    from gpuwm.core.model import experiment_fingerprint, restart_identity_payload

    exp = _scaffold()
    child = exp.domains[1]
    moved = replace(exp, domains=(
        exp.domains[0],
        replace(child, i_parent_start=child.i_parent_start + 4)))
    catalog = SimpleNamespace(run_provenance={})
    assert restart_identity_payload(exp) != restart_identity_payload(moved)
    assert (experiment_fingerprint(exp, catalog)
            != experiment_fingerprint(moved, catalog))


def test_placement_independent_identity_still_separates_real_differences():
    from gpuwm.core.nest_relocation import placement_independent_identity

    here, _ = _two_placements_of_one_domain()
    coarser = replace(here, run=replace(here.run, nx=here.run.nx - 3))
    assert (placement_independent_identity(here)["identity"]
            != placement_independent_identity(coarser)["identity"])


def test_base_segment_hangs_off_the_placement_independent_identity():
    from gpuwm.core.nest_relocation import base_segment

    here, there = _two_placements_of_one_domain()
    assert (base_segment(here).base_identity_sha256
            == base_segment(there).base_identity_sha256)
    assert base_segment(here).generation == 0


def test_relocation_record_json_round_trips_through_a_digest():
    record = _record(2)
    document = record.to_json()
    assert document["shift_child_cells"] == [6, 0]
    assert document["null_move"] is False
    assert dataclasses.is_dataclass(record)
    assert len(record.sha256) == 64


# ---------------------------------------------------------------------------
# NestCoupler.relocate: the SINT table rebuild, on CPU
# ---------------------------------------------------------------------------

def _coupler_tree(i_start=3, j_start=3):
    """The smallest tree NestCoupler will accept, built on host arrays.

    A 30-cell child at ratio 3 spans 10 cells of a 16-cell parent, and the
    +-2 SINT donor stencil leaves exactly three admissible placements
    (3, 4, 5).  That narrowness is useful here: it makes the off-grid
    refusal easy to reach deliberately.
    """
    from gpuwm.config import RunConfig
    from gpuwm.core.clock import DomainClock, DomainTicks

    def run(nx, ny, *, nested, grid_id):
        return RunConfig(nx=nx, ny=ny, nz=2, dx=1000.0, dy=1000.0,
                         ztop=10000.0, dt=3.0, run_seconds=9.0,
                         nested=nested, specified=not nested,
                         grid_id=grid_id, spec_bdy_width=5, spec_zone=1,
                         relax_zone=4)

    def clock(grid_id, parent_id, step_ticks, dt):
        spec = DomainTicks(
            grid_id=grid_id, parent_id=parent_id, parent_time_step_ratio=3,
            step_ticks=step_ticks, dt_fp32=np.float32(dt), history_ticks=100,
            restart_ticks=None, radt_ticks=None, stepra=None,
            cudt_ticks=None, stepcu=None, bldt_ticks=None, stepbl=None)
        return DomainClock(spec, tick_den=1, run_ticks=1000)

    prun, crun = run(16, 16, nested=False, grid_id=1), run(
        30, 30, nested=True, grid_id=2)
    parent = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, parent_id=0, parent_grid_ratio=1,
                            i_parent_start=1, j_parent_start=1, run=prun),
        state=SimpleNamespace(), clock=clock(1, 0, 3, 9.0))
    child = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=3,
                            i_parent_start=i_start, j_parent_start=j_start,
                            run=crun),
        state=SimpleNamespace(), parent=parent, clock=clock(2, 1, 1, 3.0))
    return parent, child


def test_coupler_relocate_rebuilds_every_stagger_at_the_new_placement():
    from gpuwm.core.nest import NestCoupler

    _parent, child = _coupler_tree()
    coupler = NestCoupler(child)
    before = {s: r.ci.copy() for s, r in coupler.registrations.items()}
    assert all(r.i_parent_start == 3 for r in coupler.registrations.values())

    child.cfg = replace_ns(child.cfg, i_parent_start=5)
    receipt = coupler.relocate()

    assert receipt["placement_generation"] == 1
    assert receipt["rolling_tables"] == "INVALID"
    assert set(coupler.registrations) == {"m", "x", "y"}
    for stagger, reg in coupler.registrations.items():
        assert reg.i_parent_start == 5
        # A move of 2 parent cells shifts every donor index by exactly 2.
        assert np.array_equal(reg.ci, before[stagger] + 2), stagger


def test_coupler_relocate_keeps_the_manifest_slot_shapes_identical():
    """A relocation must not reallocate: extents and ratio are unchanged,
    so the audited nest_* slot inventory has to come out the same."""
    from gpuwm.core.nest import NestCoupler

    _parent, child = _coupler_tree()
    coupler = NestCoupler(child)
    shapes, dtypes = dict(coupler.slot_shapes), dict(coupler.slot_dtypes)
    child.cfg = replace_ns(child.cfg, i_parent_start=5, j_parent_start=4)
    coupler.relocate()
    assert coupler.slot_shapes == shapes
    assert coupler.slot_dtypes == dtypes


def test_coupler_relocate_invalidates_the_rolling_boundary_tables():
    from gpuwm.core.nest import NestCoupler

    _parent, child = _coupler_tree()
    coupler = NestCoupler(child)
    coupler._valid = True
    coupler._geometry_bound = True
    child.cfg = replace_ns(child.cfg, i_parent_start=5)
    coupler.relocate()
    assert coupler.valid is False
    assert coupler._geometry_bound is False


def test_coupler_relocate_refuses_a_placement_off_the_parent():
    """register_nest owns the +-2 SINT stencil rule; relocation inherits it
    rather than restating it, so an off-grid move fails at the tables."""
    from gpuwm.core.nest import NestCoupler

    _parent, child = _coupler_tree()
    coupler = NestCoupler(child)
    child.cfg = replace_ns(child.cfg, i_parent_start=40)
    with pytest.raises(ValueError, match="outside the parent extent"):
        coupler.relocate()


def replace_ns(namespace, **changes):
    fields = dict(vars(namespace))
    fields.update(changes)
    return SimpleNamespace(**fields)


# ---------------------------------------------------------------------------
# relocate_child's refusals, reached before anything is touched
# ---------------------------------------------------------------------------

def _fake_child(parent_ticks=0, child_ticks=0, i_start=10, j_start=10):
    """A node stub good enough to reach the guards, and no further.

    ``initializer`` is deliberately a function that raises: every test
    below must refuse BEFORE the child is rebuilt, so reaching the
    initializer at all is itself the failure.
    """
    from gpuwm.verify.cases.nest_ideal_r3 import load_scaffold

    exp = load_scaffold()
    child_cfg = replace(exp.domains[1], i_parent_start=i_start,
                        j_parent_start=j_start)
    parent = SimpleNamespace(
        cfg=exp.root, state=SimpleNamespace(),
        clock=SimpleNamespace(ticks=parent_ticks))
    return SimpleNamespace(
        cfg=child_cfg, state=SimpleNamespace(), grid=None, parent=parent,
        coupler=SimpleNamespace(), clock=SimpleNamespace(ticks=child_ticks))


def _explode(*_args, **_kwargs):
    raise AssertionError("the child was rebuilt before the guard refused")


def test_relocate_refuses_while_the_parent_leads_the_child():
    """Relocation is a cycle-boundary operation; mid-step the rebuilt
    tables would describe an instant the child has not reached."""
    from gpuwm.core.nest_relocation import relocate_child

    node = _fake_child(parent_ticks=3, child_ticks=0)
    with pytest.raises(RelocationRefusal, match="synchronized clocks"):
        relocate_child(node, i_parent_start=11, j_parent_start=10,
                       initializer=_explode, state_digest=lambda _s: "x")


def test_relocate_refuses_a_jump_with_no_shared_ground():
    from gpuwm.core.nest_relocation import relocate_child

    node = _fake_child()
    far = node.cfg.i_parent_start + node.cfg.run.nx  # > the child's span
    with pytest.raises(RelocationRefusal, match="shares no cell"):
        relocate_child(node, i_parent_start=far, j_parent_start=10,
                       initializer=_explode, state_digest=lambda _s: "x")


def test_relocate_refuses_the_root_domain():
    from gpuwm.core.nest_relocation import relocate_child

    node = _fake_child()
    node.parent = None
    with pytest.raises(RelocationRefusal, match="no parent"):
        relocate_child(node, i_parent_start=11, j_parent_start=10,
                       initializer=_explode, state_digest=lambda _s: "x")


def test_relocate_refuses_a_child_with_no_coupler():
    from gpuwm.core.nest_relocation import relocate_child

    node = _fake_child()
    node.coupler = None
    with pytest.raises(RelocationRefusal, match="no coupler"):
        relocate_child(node, i_parent_start=11, j_parent_start=10,
                       initializer=_explode, state_digest=lambda _s: "x")


def test_relocate_refuses_a_move_the_bounds_disallow_before_rebuilding():
    from gpuwm.core.nest_relocation import relocate_child

    node = _fake_child()
    bounds = RelocationConfig(enabled=True, grid_id=node.cfg.grid_id,
                              max_move_parent_cells=2)
    with pytest.raises(RelocationRefusal, match="over the configured"):
        relocate_child(node, i_parent_start=node.cfg.i_parent_start + 9,
                       j_parent_start=10, bounds=bounds,
                       initializer=_explode, state_digest=lambda _s: "x")


# ---------------------------------------------------------------------------
# The whole mechanism, on the card
# ---------------------------------------------------------------------------

@requires_gpu
def test_relocation_verify_case_passes_every_component(tmp_path):
    """Run the artifact, not a mock: the idealized ratio-3 tree integrates,
    relocates twice, and integrates again.  ~5 s on a 5090."""
    from gpuwm.verify.cases.nest_relocate import COMPONENTS, run

    report = run(tmp_path / "relocate")
    failing = [name for name in COMPONENTS
               if not report["components"][name]["pass"]]
    assert not failing, failing
    assert report["pass"]

    # The claims, restated as assertions so a silently-emptied component
    # cannot pass by having nothing in it.
    overlap = report["components"]["overlap_bit_identity"]
    assert len(overlap["compared_fields"]) >= 20
    assert sum(item["overlap_bit_mismatches"]
               for item in overlap["fields"].values()) == 0
    assert sum(item["overlap_cells"]
               for item in overlap["fields"].values()) > 1_000_000

    treatment = report["components"]["transplant_treatment_proof"]
    assert treatment["overlap_cells_changed"] > 0
    assert treatment["fraction_changed"] >= treatment["minimum_fraction"]

    null = report["components"]["null_move_bit_identity"]
    assert null["child_state_sha256_before"] == null["child_state_sha256_after"]
    # The non-circular half: state the move REBUILDS rather than copies.
    rebuilt = null["rebuilt_not_copied"]
    assert rebuilt["compared_fields"] >= 5 and rebuilt["pass"]
    assert all(item["bit_mismatches"] == 0
               for item in rebuilt["fields"].values())
    seeds = null["rk_seed_consistency"]
    assert seeds["compared_fields"] >= 6 and seeds["pass"]

    parent = report["components"]["parent_bitwise_unchanged"]
    for arm in ("null_move", "real_move"):
        assert parent[arm]["before"] == parent[arm]["after"]

    assert report["components"]["post_move_integration"]["steps"] > 0
    assert report["real_move_segment"]["generation"] == 2
