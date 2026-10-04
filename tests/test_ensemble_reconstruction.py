from types import SimpleNamespace

import pytest

from gpuwm.ensemble.runtime_context import (MemberOutputCapture, member_output_scope,
    current_member_reconstruction_owner, bind_reconstructed_member_node)


def node():
    return SimpleNamespace(state=SimpleNamespace(), cfg=SimpleNamespace(run=object()),
                           clock=object(), grid=object())


def test_ordinary_and_disabled_rebuilds_are_inert_before_any_attribute_read():
    class Unbound:
        def __getattr__(self, name):
            pytest.fail(f"inactive reconstruction read {name}")
    bind_reconstructed_member_node(Unbound())
    with member_output_scope(MemberOutputCapture(None, 7)):
        bind_reconstructed_member_node(Unbound())
        assert current_member_reconstruction_owner(Unbound()) is None


def test_domain_birth_binds_actual_new_state_grid_and_original_clock():
    newborn, events = node(), []
    prepared = object()
    capture = MemberOutputCapture(None, 7, initialize_callback=lambda **event: events.append(event))
    with member_output_scope(capture):
        bind_reconstructed_member_node(newborn, prepared_case=prepared)
    assert events == [{"prepared_case": prepared, "state": newborn.state,
        "cfg": newborn.cfg.run, "grid": newborn.grid, "clock": newborn.clock}]
    assert not hasattr(newborn.state, "_ensemble_stochastic")


def test_domain_move_retains_same_spectra_owner_without_reseeding():
    owner = SimpleNamespace(enabled=True, member_id=7, hook=SimpleNamespace(pending_step=None))
    outgoing, incoming, seen = node(), node(), []
    outgoing.state._ensemble_stochastic = owner
    capture = MemberOutputCapture(None, 7, initialize_callback=lambda **event:
        seen.append(event["state"]._ensemble_stochastic))
    with member_output_scope(capture):
        retained = current_member_reconstruction_owner(outgoing.state)
        bind_reconstructed_member_node(incoming, previous_owner=retained)
    assert seen == [owner] and incoming.state._ensemble_stochastic is owner


def test_reconstruction_rejects_wrong_member_and_inflight_spectra():
    state = SimpleNamespace(_ensemble_stochastic=SimpleNamespace(enabled=True,
        member_id=9, hook=SimpleNamespace(pending_step=None)))
    with member_output_scope(MemberOutputCapture(None, 7, initialize_callback=lambda **kwargs: None)):
        with pytest.raises(ValueError, match="another ensemble member"):
            current_member_reconstruction_owner(state)
        state._ensemble_stochastic.member_id = 7
        state._ensemble_stochastic.hook.pending_step = 4
        with pytest.raises(ValueError, match="unfinished timestep"):
            current_member_reconstruction_owner(state)


def test_actual_schedule_retarget_rebinds_clock_and_keeps_absolute_spectral_update_index():
    from datetime import datetime
    from gpuwm.config import RunConfig
    from gpuwm.experiment import experiment_from_run_config
    from gpuwm.core.clock import resolve_clock
    from gpuwm.runtime import _retarget_tree_schedule
    from gpuwm.ensemble.stochastic_execution import StochasticPhysicsBinding
    cfg = RunConfig(nx=16, ny=14, nz=8, dx=3000., dy=3000., ztop=20000.,
                    dt=12., run_seconds=120., use_adaptive_time_step=True)
    exp = experiment_from_run_config(cfg, datetime(2024, 1, 1))
    clock = resolve_clock(exp, lbc_interval_s=60.).clocks()[1]
    clock.ticks, clock.step_count = 60 * clock.tick_den, 7
    boundary_ticks = clock.ticks
    calls = []
    hook = SimpleNamespace(enabled=True, completed_step=6, pending_step=None,
        parameter_patterns={}, set_time_step=lambda dt: calls.append(("dt", dt)),
        before_timestep=lambda step: calls.append(("update", step)))
    binding = StochasticPhysicsBinding(hook, member_id=7, clock=clock)
    domain = SimpleNamespace(cfg=exp.root, state=SimpleNamespace(_ensemble_stochastic=binding),
                             clock=clock, grid=object())
    model = SimpleNamespace(root=domain, walk_parent_first=lambda: iter((domain,)))
    def rebind(**event):
        assert event["state"]._ensemble_stochastic is binding
        binding.clock = event["clock"]
    with member_output_scope(MemberOutputCapture(None, 7, initialize_callback=rebind)):
        _retarget_tree_schedule(model, exp, 120., lbc_interval_s=60.)
        assert binding.clock is domain.clock and domain.clock is not clock
        assert domain.clock.step_count == 5 and domain.clock.ticks == boundary_ticks
        binding.before_physics(domain.state, cfg)
        assert calls == [("dt", 12.), ("update", 7)]
        hook.completed_step = 7
        _retarget_tree_schedule(model, exp, 120., lbc_interval_s=60.)
        binding.before_physics(domain.state, cfg)
        assert calls[-2:] == [("dt", 12.), ("update", 8)]
    assert clock.step_count == 7 and clock.ticks == boundary_ticks
