"""Full-domain pattern lifecycle and exact staggered window coordinates."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.ensemble.stochastic import StochasticTimestepHook
from gpuwm.ensemble.stochastic_execution import StochasticPhysicsBinding
from gpuwm.ensemble.stochastic_streaming import (
    StochasticSweepLease, stochastic_window_indices, window_memory_plan,
)


def spec(*, x=2, y=1, nx=4, ny=3, cnx=3, cny=2, periodic=True):
    return SimpleNamespace(ci0=x, cj0=y, nx=nx, ny=ny, cnx=cnx, cny=cny,
                           periodic_x=periodic, periodic_y=periodic)


def test_periodic_component_windows_alias_closing_face_zero_and_wrap_mass_halos():
    window = spec()
    my, mx = stochastic_window_indices(window, "mass")
    uy, ux = stochastic_window_indices(window, "u")
    vy, vx = stochastic_window_indices(window, "v")
    assert mx.tolist() == [2, 3, 0]
    assert ux.tolist() == [2, 3, 0, 1]
    assert my.tolist() == [1, 2] and uy.tolist() == [1, 2]
    assert vy.tolist() == [1, 2, 0] and vx.tolist() == [2, 3, 0]
    _, outside = stochastic_window_indices(spec(x=-1), "u")
    assert outside.tolist() == [3, 0, 1, 2]


def test_nonperiodic_windows_refuse_missing_pattern_coordinates():
    assert stochastic_window_indices(spec(x=1, y=0, periodic=False), "u")[1].tolist() == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="leaves"):
        stochastic_window_indices(spec(periodic=False), "mass")


def test_disabled_provider_transform_and_complete_are_pure_identity_without_cuda():
    hook = StochasticTimestepHook((1, 1), dx=1, dy=1, dt=1, member_seed=0, enabled=False)
    original = object()
    assert hook.transform_nonmicrophysics(original, tendency_scope="nonmicrophysics") is original
    hook.complete_timestep()
    assert hook.after_nonmicrophysics(original, tendency_scope="nonmicrophysics") is original


def test_provider_after_preserves_transform_then_single_complete_order():
    hook = StochasticTimestepHook.__new__(StochasticTimestepHook)
    hook.enabled, hook.pending_step, hook.completed_step = True, 3, 2
    hook._pattern, hook._forcing = object(), object()
    calls = []
    original, transformed = object(), object()
    def transform(rates, **kwargs):
        calls.append((rates, hook.pending_step))
        return transformed
    hook.transform_nonmicrophysics = transform
    assert hook.after_nonmicrophysics(original, tendency_scope="nonmicrophysics") is transformed
    assert calls == [(original, 3)] and hook.completed_step == 3 and hook.pending_step is None
    with pytest.raises(ValueError, match="pending"):
        hook.complete_timestep()


@dataclass
class Rates:
    ru: object
    rv: object
    rtheta: object
    rqv: object
    rqc: object


class Hook:
    enabled = True
    def __init__(self):
        self.sppt = SimpleNamespace(device=0)
        self.skebs = None
        self.spp, self.spp_levels, self.parameter_patterns = {}, {}, {}
        self.completed_step, self.pending_step = -1, None
        self._pattern, self._forcing = None, None
        self.before_calls, self.after_calls, self.commits, self.dt = [], [], [], []
    def set_time_step(self, dt):
        self.dt.append(dt)
    def before_timestep(self, step):
        assert self.pending_step is None and step == self.completed_step + 1
        self.pending_step = step
        self.before_calls.append(step)
        self._pattern = np.arange(20, dtype=np.float32).reshape(4, 5) * np.float32(.01)
    def transform_nonmicrophysics(self, rates, **kwargs):
        assert self.pending_step is not None
        self.after_calls.append(kwargs)
        return {name: values * (np.float32(1) + kwargs["sppt_patterns"][name])
                for name, values in rates.items()}
    def complete_timestep(self):
        self.commits.append(self.pending_step)
        self.completed_step, self.pending_step = self.pending_step, None
        self._pattern = self._forcing = None


def test_one_pattern_update_is_shared_by_all_windows_and_microphysics_is_never_transformed():
    hook = Hook()
    binding = StochasticPhysicsBinding(hook, member_id=19, clock=SimpleNamespace(step_count=0))
    lease = StochasticSweepLease(binding, array_module=np)
    cfg = SimpleNamespace(ny=2, nx=3, nz=1, dt=7.5)
    lease.begin(cfg, windows=2)
    owner_words = hook._pattern.tobytes()
    for window_id, window in enumerate((spec(), spec(x=-1, y=0))):
        state = SimpleNamespace()
        lease.bind_window(state, cfg, window, window_id)
        rate = Rates(np.ones((1, 2, 4), np.float32), np.ones((1, 3, 3), np.float32),
                     np.ones((1, 2, 3), np.float32), np.ones((1, 2, 3), np.float32), object())
        state._ensemble_stochastic.before_physics(state, cfg)
        result = state._ensemble_stochastic.after_nonmicrophysics(state, cfg, rate)
        assert result.rqc is rate.rqc
        for name, values in (("u", result.ru), ("v", result.rv), ("theta", result.rtheta), ("qv", result.rqv)):
            ys, xs = stochastic_window_indices(window, {"u": "u", "v": "v", "theta": "mass", "qv": "mass"}[name])
            expected = np.float32(1) + hook._pattern[np.ix_(ys, xs)]
            assert values[0].tobytes() == expected.tobytes()
        assert hook._pattern.tobytes() == owner_words and hook.pending_step == 0
    assert hook.before_calls == [0] and len(hook.after_calls) == 2 and hook.commits == []
    lease.finish()
    assert hook.commits == [0] and binding.applied_steps == 1
    binding.clock.step_count = 1
    lease.begin(SimpleNamespace(dt=4.5), windows=2)
    assert hook.before_calls == [0, 1] and hook.dt == [7.5, 4.5]
    assert lease.receipt()["full_domain_3d_rate_banks"] == 0


def test_missing_or_duplicate_window_cannot_complete_or_double_apply_owner():
    hook = Hook()
    lease = StochasticSweepLease(StochasticPhysicsBinding(hook, member_id=3), array_module=np)
    cfg = SimpleNamespace(ny=2, nx=3, nz=1, dt=12)
    lease.begin(cfg, windows=2)
    with pytest.raises(RuntimeError, match="complete window"):
        lease.finish()
    assert hook.pending_step == 0 and not hook.commits
    state = SimpleNamespace()
    lease.bind_window(state, cfg, spec(), 0)
    rates = Rates(np.ones((1, 2, 4), np.float32), np.ones((1, 3, 3), np.float32),
                  np.ones((1, 2, 3), np.float32), np.ones((1, 2, 3), np.float32), object())
    view = state._ensemble_stochastic
    view.before_physics(state, cfg)
    view.after_nonmicrophysics(state, cfg, rates)
    with pytest.raises(RuntimeError, match="already applied"):
        view.after_nonmicrophysics(state, cfg, rates)
    assert hook.pending_step == 0 and binding_never_committed(lease)


def binding_never_committed(lease):
    return lease.binding.applied_steps == 0


def test_window_plan_contains_2d_patterns_and_only_local_3d_rate_work():
    plan = window_memory_plan((4, 5), nz=7, sppt=True, skebs=True, spp_levels=("pbl", "lsm"))
    fields = {row["name"]: row for row in plan.inventory(1)}
    assert all(len(fields[name]["shape"]) == 2 for name in fields if ":sppt:" in name or ":skebs:" in name or ":spp:" in name)
    assert fields["stochastic:result:u"]["shape"] == (7, 4, 6)
    assert fields["stochastic:result:v"]["shape"] == (7, 5, 5)
    assert window_memory_plan((4, 5), nz=7) is None


def test_sweep_update_index_continues_restored_owner_when_new_clock_counter_differs():
    hook = Hook()
    hook.completed_step = 12
    binding = StochasticPhysicsBinding(hook, member_id=19, clock=SimpleNamespace(step_count=3))
    lease = StochasticSweepLease(binding, array_module=np)
    cfg = SimpleNamespace(ny=2, nx=3, nz=1, dt=4.5)
    lease.begin(cfg, windows=1)
    assert hook.before_calls == [13] and hook.dt == [4.5] and lease.pending == 13
    assert binding.clock.step_count == 3
