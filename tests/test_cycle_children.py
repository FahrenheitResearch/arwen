"""Child lifecycle ACROSS cycle boundaries: re-place, retire, reclaim.

A child cannot be allocated mid-run on this line -- ``active_experiment``
refuses a grid_id that was not declared dormant, and VRAM is reserved at
t=0 so a spawn is an activation rather than an allocation that can OOM
after hours.  The cycle boundary is already a process boundary, so it is
the allocation point, and these tests hold the three things that follow:
a live child is RE-PLACED rather than respawned, a dead storm's child is
RETIRED and its slot comes back, and a child that blows up does not take
the cycle with it.

Pure tests: no GPU, no network, no case names (standing owner rule).
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.cycle.children import (ChildSlot, SlotPool, child_tick_alignment,
                                  emit_dormant_domains, plan_children,
                                  run_child_leg)
from gpuwm.cycle.contracts import CHILD_STATES, CycleRefusal, seconds_to_ticks
from gpuwm.cycle.placement import (PlacementRequest,
                                   parent_geometry_from_fields,
                                   resolve_placement)


def geometry(*, ny: int = 60, nx: int = 80):
    lat = 30.0 + 0.05 * np.arange(ny, dtype=np.float64)[:, None]
    lon = -100.0 + 0.05 * np.arange(nx, dtype=np.float64)[None, :]
    return parent_geometry_from_fields(
        lat_field=np.broadcast_to(lat, (ny, nx)).copy(),
        lon_field=np.broadcast_to(lon, (ny, nx)).copy(), dx_m=3000.0)


def slot(grid_id: int) -> ChildSlot:
    return ChildSlot(grid_id=grid_id, nx=30, ny=30, dx_m=1000.0,
                     dt_seconds=2.5, parent_grid_ratio=3)


def request(lat: float, lon: float, *, strength: float = 55.0
            ) -> PlacementRequest:
    return PlacementRequest(lat=lat, lon=lon, dx_m=1000.0, nx=30, ny=30,
                            source="tracker", strength=strength,
                            evidence={"max_value": strength, "cells": 9})


# ---------------------------------------------------------------------------
# the slot pool
# ---------------------------------------------------------------------------
class TestSlotPool:
    def test_a_claim_removes_a_slot_and_a_release_returns_it(self):
        pool = SlotPool([slot(2), slot(3)])
        assert pool.free() == (2, 3)
        placement = resolve_placement(request(31.5, -98.0),
                                      parent_geometry=geometry(), grid_id=2)
        pool.claim(2, placement)
        assert pool.free() == (3,)
        assert set(pool.live()) == {2}
        pool.release(2, reason="storm decayed")
        assert pool.free() == (2, 3)
        assert pool.live() == {}

    def test_claiming_a_busy_slot_refuses_by_name(self):
        pool = SlotPool([slot(2)])
        placement = resolve_placement(request(31.5, -98.0),
                                      parent_geometry=geometry(), grid_id=2)
        pool.claim(2, placement)
        with pytest.raises(CycleRefusal) as caught:
            pool.claim(2, placement)
        assert "grid_id=2" in str(caught.value)


# ---------------------------------------------------------------------------
# planning one boundary
# ---------------------------------------------------------------------------
class TestPlanChildren:
    def test_live_child_is_replaced_not_respawned(self):
        pool = SlotPool([slot(2), slot(3)])
        parent = geometry()
        first = plan_children(cycle_index=0, pool=pool,
                              requests=[request(31.5, -98.0, strength=60.0)],
                              previous_children=[], retire_below_strength=40.0,
                              min_separation_km=20.0, parent_geometry=parent,
                              allow_clamp=False)
        assert [item["state"] for item in first] == ["PLANNED"]
        assert pool.free() == (3,)

        # The storm moved one parent cell north and is still strong.
        second = plan_children(
            cycle_index=1, pool=pool,
            requests=[request(31.55, -98.0, strength=58.0)],
            previous_children=first, retire_below_strength=40.0,
            min_separation_km=20.0, parent_geometry=parent, allow_clamp=False)
        assert [item["state"] for item in second] == ["LIVE"]
        assert second[0]["grid_id"] == 2, "the SAME slot follows the storm"
        assert second[0]["moved_parent_cells"] == 1
        assert pool.free() == (3,), "no second slot was consumed"

    def test_weak_child_is_retired_and_its_slot_returns_to_the_pool(self):
        pool = SlotPool([slot(2), slot(3)])
        parent = geometry()
        first = plan_children(cycle_index=0, pool=pool,
                              requests=[request(31.5, -98.0, strength=60.0)],
                              previous_children=[], retire_below_strength=40.0,
                              min_separation_km=20.0, parent_geometry=parent,
                              allow_clamp=False)
        assert pool.free() == (3,)

        second = plan_children(
            cycle_index=1, pool=pool,
            requests=[request(31.5, -98.0, strength=12.0)],
            previous_children=first, retire_below_strength=40.0,
            min_separation_km=20.0, parent_geometry=parent, allow_clamp=False)
        retired = [item for item in second if item["state"] == "RETIRED"]
        assert len(retired) == 1
        assert retired[0]["grid_id"] == 2
        assert retired[0]["evidence"]["observed_strength"] == 12.0
        # The despawn: the slot is back, and a new storm can claim it.
        assert pool.free() == (2, 3)

    def test_a_child_with_no_signal_at_all_is_retired(self):
        pool = SlotPool([slot(2)])
        parent = geometry()
        first = plan_children(cycle_index=0, pool=pool,
                              requests=[request(31.5, -98.0, strength=60.0)],
                              previous_children=[], retire_below_strength=40.0,
                              min_separation_km=20.0, parent_geometry=parent,
                              allow_clamp=False)
        second = plan_children(
            cycle_index=1, pool=pool, requests=[],
            previous_children=first, retire_below_strength=40.0,
            min_separation_km=20.0, parent_geometry=parent, allow_clamp=False)
        assert [item["state"] for item in second] == ["RETIRED"]
        assert pool.free() == (2,)

    def test_a_request_with_no_slot_is_refused_not_dropped(self):
        pool = SlotPool([slot(2)])
        parent = geometry()
        got = plan_children(
            cycle_index=0, pool=pool,
            requests=[request(31.5, -98.0, strength=60.0),
                      request(32.0, -97.0, strength=50.0)],
            previous_children=[], retire_below_strength=40.0,
            min_separation_km=5.0, parent_geometry=parent, allow_clamp=False)
        assert len(got) == 2
        refused = [item for item in got if item["state"] == "REFUSED"]
        assert len(refused) == 1
        assert refused[0]["refusal"] == "slot pool exhausted"

    def test_every_record_uses_the_shared_state_vocabulary(self):
        pool = SlotPool([slot(2)])
        got = plan_children(
            cycle_index=0, pool=pool, requests=[request(31.5, -98.0)],
            previous_children=[], retire_below_strength=40.0,
            min_separation_km=5.0, parent_geometry=geometry(),
            allow_clamp=False)
        assert all(item["state"] in CHILD_STATES for item in got)


# ---------------------------------------------------------------------------
# the tick contract a child has to satisfy before anything is allocated
# ---------------------------------------------------------------------------
class TestTickAlignment:
    def test_a_dividing_child_dt_gives_the_ratio(self):
        parent_ticks = seconds_to_ticks(15.0, label="parent dt")
        assert child_tick_alignment(parent_step_ticks=parent_ticks,
                                    child_dt_seconds=2.5).ratio == 6

    def test_non_dividing_child_dt_refuses_with_the_remainder(self):
        parent_ticks = seconds_to_ticks(15.0, label="parent dt")
        with pytest.raises(CycleRefusal) as caught:
            child_tick_alignment(parent_step_ticks=parent_ticks,
                                 child_dt_seconds=4.0)
        text = str(caught.value)
        assert "does not divide the parent step exactly" in text
        assert "remainder=3000" in text

    def test_a_sub_millisecond_child_dt_refuses_rather_than_rounding(self):
        with pytest.raises(CycleRefusal) as caught:
            child_tick_alignment(parent_step_ticks=15000,
                                 child_dt_seconds=2.0005)
        assert "whole number of model ticks" in str(caught.value)


# ---------------------------------------------------------------------------
# the plan becoming a declared experiment
# ---------------------------------------------------------------------------
class TestDormantDomains:
    def test_the_pool_renders_as_dormant_domains_with_a_spawn_table(self):
        pool = SlotPool([slot(2), slot(3)])
        blocks = emit_dormant_domains(
            pool, template_domain={"parent_id": 1, "nz": 50,
                                   "i_parent_start": 1, "j_parent_start": 1})
        assert [block["grid_id"] for block in blocks] == [2, 3]
        for block in blocks:
            assert block["nz"] == 50, "the template's physics is carried"
            assert block["nx"] == 30 and block["ny"] == 30
            assert block["parent_grid_ratio"] == 3
            assert block["spawn"]["trigger"] in ("uh", "reflectivity",
                                                 "time")
        # Every slot is priced at t=0 whether it fills or not, which is
        # the honest cost of deterministic VRAM.
        assert len(blocks) == len(pool.free())


# ---------------------------------------------------------------------------
# a child may die; the cycle may not
# ---------------------------------------------------------------------------
class TestChildLeg:
    def _placement(self):
        return resolve_placement(request(31.5, -98.0),
                                 parent_geometry=geometry(), grid_id=2)

    def test_a_healthy_leg_reports_live(self):
        got = run_child_leg(
            grid_id=2, placement=self._placement(), cycle_index=0,
            advance=lambda **kw: {"u": np.zeros(4), "w": np.ones(4)})
        assert got["state"] == "LIVE"
        assert got["refusal"] is None

    def test_diverged_child_does_not_raise(self):
        """A NaN in a child costs one slot, not the run."""

        state = {"u": np.array([1.0, 2.0, np.nan, 4.0]),
                 "w": np.zeros(4)}
        got = run_child_leg(grid_id=2, placement=self._placement(),
                            cycle_index=3, advance=lambda **kw: state)
        assert got["state"] == "DIVERGED"
        assert got["evidence"]["field"] == "u"
        assert got["evidence"]["cell"] == [2]
        assert str(got["evidence"]["value"]) == "nan"
        assert "d02" in got["refusal"] and "cycle 3" in got["refusal"]

    def test_a_runner_refusal_is_caught_and_named(self):
        def advance(**kwargs):
            raise CycleRefusal("vertical velocity over the ceiling",
                               grid_id=2, w_max=182.4)

        got = run_child_leg(grid_id=2, placement=self._placement(),
                            cycle_index=4, advance=advance)
        assert got["state"] == "DIVERGED"
        assert "w_max=182.4" in got["refusal"]

    def test_a_parent_refusal_is_not_swallowed(self):
        """The asymmetry, asserted: only the CHILD's death is absorbed."""

        def advance(**kwargs):
            raise MemoryError("the allocator, not this child")

        with pytest.raises(MemoryError):
            run_child_leg(grid_id=2, placement=self._placement(),
                          cycle_index=5, advance=advance)
