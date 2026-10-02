"""The opt-in clock reduces work without raising a sanctioned step."""
from dataclasses import dataclass, replace
from fractions import Fraction

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.adaptive_clock import AdaptiveClockDriver
from gpuwm.io.restart import (
    _configuration_digest_values, _require_config_match, configuration_echo)
from test_adaptive_clock_driver import (
    FakeRun, FakeCfg, FakeNode, FakeClock, FakeSpec, FakeModel)


@dataclass
class LatticeRun(FakeRun):
    nx: int = 288
    ny: int = 288
    nz: int = 59
    adaptive_nest_lattice: bool = True


def tree():
    nodes = []
    for gid, nx, ny, step in ((1, 288, 288, 15), (2, 216, 216, 10),
                              (3, 144, 240, Fraction(10, 3))):
        run = LatticeRun(gid, float(step), nx=nx, ny=ny, min_time_step=1,
                         max_time_step=int(step), max_time_step_den=0)
        node = FakeNode(FakeCfg(gid, run, 3 if gid > 1 else 1),
                        FakeClock(FakeSpec(gid, int(step * 300))))
        if nodes:
            node.parent = nodes[-1]
            nodes[-1].children.append(node)
        nodes.append(node)
    nodes[-1].cfg.run = replace(nodes[-1].cfg.run, max_time_step=10,
                                max_time_step_den=3)
    return FakeModel(nodes[0])


def driver(model, calls=None):
    def cfl(gid):
        if calls is not None:
            calls.append(gid)
        return 0., 0.
    return AdaptiveClockDriver(model, cfl_source=cfl, tick_den=300,
                               map_factor_source=lambda gid: 1.)


def test_work_minimum_and_memory():
    model = tree()
    calls = []
    drv = driver(model, calls)
    clocks = {gid: model.node(gid).clock for gid in drv.order}
    drv(0, clocks)
    assert [clocks[gid].step_ticks for gid in drv.order] == [3000, 3000, 1000]
    assert drv.controllers[1].last_dt == 15
    assert calls == [1, 2, 3]


def test_default_bypasses_selection():
    model = tree()
    model.root.cfg.run = replace(model.root.cfg.run, adaptive_nest_lattice=False)
    drv = driver(model)
    drv._nest_lattice_step = lambda *args: pytest.fail("default ran optimizer")
    clocks = {gid: model.node(gid).clock for gid in drv.order}
    drv(0, clocks)
    assert [clocks[gid].step_ticks for gid in drv.order] == [4500, 2250, 750]


def test_alarm_and_end_landings_bypass_selection():
    for alarm in (True, False):
        model = tree()
        if alarm:
            model.root.clock.spec.history_ticks = 3600
        else:
            model.root.clock.run_ticks = 3600
        drv = driver(model)
        drv._nest_lattice_step = lambda *args: pytest.fail("landing optimized")
        clocks = {gid: model.node(gid).clock for gid in drv.order}
        drv(0, clocks)
        assert clocks[1].step_ticks == 3600


def test_candidates_never_raise_proposal_and_ties_keep_longer():
    model = tree()
    drv = driver(model)
    for ticks in range(12, 6000, 12):
        proposal = Fraction(ticks, 300)
        got = drv._nest_lattice_step(proposal, {
            1: proposal, 2: Fraction(10), 3: Fraction(10, 3)})
        assert 0 < got <= proposal
    assert drv._nest_lattice_step(Fraction(10), {
        1: Fraction(10), 2: Fraction(10), 3: Fraction(10, 3)}) == 10


def test_restart_mode_flip_refuses_and_old_default_is_compatible():
    cfg = RunConfig(12, 12, 8, 1000., 1000., 10000., 6., 60.,
                    use_adaptive_time_step=True)
    import dataclasses
    stored = dataclasses.asdict(cfg)
    stored.pop("adaptive_nest_lattice")
    _require_config_match(stored, cfg, "checkpoint")
    with pytest.raises(ValueError, match="adaptive_nest_lattice"):
        _require_config_match(stored, replace(cfg, adaptive_nest_lattice=True),
                              "checkpoint")
    with pytest.raises(ValueError, match="adaptive_nest_lattice"):
        _require_config_match(dict(stored, adaptive_nest_lattice=True), cfg,
                              "checkpoint")
    assert _configuration_digest_values(stored) == _configuration_digest_values(
        dataclasses.asdict(cfg))


def test_restart_echo_omits_only_compatible_defaults():
    cfg = RunConfig(12, 12, 8, 1000., 1000., 10000., 6., 60.,
                    use_adaptive_time_step=True)
    echo = configuration_echo(cfg)
    assert not {"adaptive_nest_lattice", "zadvect_implicit", "w_crit_cfl"} & echo.keys()
    changed = replace(cfg, adaptive_nest_lattice=True, zadvect_implicit=1,
                      w_crit_cfl=0.5)
    changed_echo = configuration_echo(changed)
    assert changed_echo["adaptive_nest_lattice"] is True
    assert changed_echo["zadvect_implicit"] == 1
    assert changed_echo["w_crit_cfl"] == 0.5
    _require_config_match(echo, cfg, "checkpoint")
    with pytest.raises(ValueError, match="adaptive_nest_lattice"):
        _require_config_match(echo, changed, "checkpoint")


def test_mode_requires_adaptive_clock():
    with pytest.raises(ValueError, match="needs use_adaptive_time_step"):
        validate_run_config(RunConfig(12, 12, 8, 1000., 1000., 10000., 6., 60.,
                                      adaptive_nest_lattice=True))


def test_invalid_proposal_does_not_hide_a_legal_cheaper_candidate():
    model = tree()
    model.node(2).cfg.parent_time_step_ratio = 5
    model.node(3).cfg.parent_time_step_ratio = 5
    drv = driver(model)
    drv.tick_den = 100
    assert drv._nest_lattice_step(Fraction(199, 5), {
        1: Fraction(199, 5), 2: Fraction(6), 3: Fraction(1)}) == 36


def test_exact_work_tie_keeps_the_longer_root_step():
    model = tree()
    for gid, nx, ny in ((1, 24, 16), (2, 16, 12), (3, 8, 8)):
        node = model.node(gid)
        node.cfg.run = replace(node.cfg.run, nx=nx, ny=ny)
    drv = driver(model)
    # Cell counts 384 / 192 / 64 make 15 s and 10 s cost exactly equal.
    assert drv._nest_lattice_step(Fraction(15), {
        1: Fraction(15), 2: Fraction(10), 3: Fraction(10, 3)}) == 15


def test_shared_toml_door_identity_and_receipt():
    from datetime import datetime
    from gpuwm.experiment import build_experiment, _DOMAIN_RUN_OVERRIDES
    from gpuwm.core.model import restart_identity_payload
    from gpuwm.terrain_clock import derive_clock
    document = {
        "experiment": {"name": "lattice", "start_time": datetime(2020, 1, 1),
                       "run_seconds": 60., "restart_interval_s": 0.},
        "shared": {"nz": 8, "ztop": 10000., "use_adaptive_time_step": True},
        "domain": [{"grid_id": 1, "parent_id": 0, "nx": 24, "ny": 24,
                    "i_parent_start": 1, "j_parent_start": 1,
                    "parent_grid_ratio": 1, "parent_time_step_ratio": 1,
                    "dx": 1000., "time_step": 6, "history_interval_s": 60.}],
    }
    off = build_experiment(document, source="lattice test")
    assert "adaptive_nest_lattice" not in _DOMAIN_RUN_OVERRIDES
    assert "adaptive_nest_lattice" not in str(restart_identity_payload(off))
    document["shared"]["adaptive_nest_lattice"] = True
    on = build_experiment(document, source="lattice test")
    assert on.root.run.adaptive_nest_lattice
    assert restart_identity_payload(on) != restart_identity_payload(off)
    for exp in (off, on):
        row = derive_clock(1, exp.root.run, Fraction(6), 0., None).receipt()
        assert (row.get("adaptive_step_mode") == "nest-lattice-cell-steps-v1"
                if exp is on else "adaptive_step_mode" not in row)
