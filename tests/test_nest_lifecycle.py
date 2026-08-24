from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.nest_lifecycle import (
    RearmConfig, RetireConfig, RetirementWatch, build_domain_follow_config,
)
from gpuwm.core.nest_spawn import SpawnConfig, SpawnWatch
from gpuwm.core.storm_tracking import NestFootprint
from gpuwm.core import uh_diag


class FakeState:
    def __init__(self, **slots):
        self._scratch = {name: np.asarray(value, dtype=np.float32)
                         for name, value in slots.items()}

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


def child_cfg(grid_id=4):
    return SimpleNamespace(
        grid_id=grid_id, i_parent_start=2, j_parent_start=2,
        parent_grid_ratio=1, run=SimpleNamespace(nx=3, ny=3))


def test_retire_requires_continuous_quiet_window():
    state = FakeState(uh_spawn_window=np.full((8, 8), 20.0))
    watch = RetirementWatch(RetireConfig(
        trigger="uh", threshold=60.0, sustained_s=900.0,
        min_lifetime_s=1800.0))
    assert not watch.evaluate(state, child_cfg(), t=1700, born_t=0).get("retire", False)
    assert not watch.evaluate(state, child_cfg(), t=1800, born_t=0)["retire"]
    assert not watch.evaluate(state, child_cfg(), t=2600, born_t=0)["retire"]
    assert watch.evaluate(state, child_cfg(), t=2700, born_t=0)["retire"]
    state._scratch["uh_spawn_window"][...] = 100.0
    assert not watch.evaluate(state, child_cfg(), t=3000, born_t=0)["retire"]
    assert watch.quiet_since is None


def test_rearm_config_is_bounded_and_deterministic():
    cfg = RearmConfig(max_firings=4, cooldown_s=1800)
    assert cfg.to_json()["max_firings"] == 4
    with pytest.raises(ValueError):
        RearmConfig(max_firings=0, cooldown_s=0)


def test_spawn_watch_can_be_rearmed_without_reconstruction():
    watch = SpawnWatch(
        SpawnConfig(trigger="time", at_s=60),
        NestFootprint(grid_id=4, i_parent_start=2, j_parent_start=2,
                      child_nx=10, child_ny=10, parent_grid_ratio=3),
        keepout_cells=1)
    watch.fired = True
    watch.closed = True
    watch.rearm(t=120, episode=2)
    assert not watch.fired
    assert not watch.closed
    assert any(r["decision"] == "rearmed" for r in watch.receipts)


def test_per_domain_follow_requires_own_cadence():
    base = {
        "field": "uh", "threshold": 100.0, "fallback_threshold": 35.0,
        "search_margin_cells": 10, "min_shift_cells": 2,
        "max_shift_cells": 10, "cooldown_seconds": 600.0,
    }
    with pytest.raises(ValueError, match="cadence_seconds"):
        build_domain_follow_config(base, "case.toml", grid_id=4)
    cfg = build_domain_follow_config(
        {**base, "cadence_seconds": 300.0,
         "max_move_parent_cells": 8, "min_overlap_fraction": 0.7},
        "case.toml", grid_id=4)
    assert cfg.cadence_seconds == 300.0
    assert cfg.tracker.field == "uh"


def test_dynamic_follow_windows_are_folded_and_resettable():
    slot = uh_diag.follow_window_slot(4)
    state = FakeState(
        uh_follow_window=np.zeros((3, 3)),
        uh_spawn_window=np.zeros((3, 3)),
        **{slot: np.ones((3, 3))})
    windows = uh_diag._tracker_windows(state)
    assert len(windows) == 3
    uh_diag.reset_tracker_window(state, slot)
    assert not state._scratch[slot].any()
