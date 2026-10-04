from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.ensemble.stochastic_execution import (
    StochasticPhysicsBinding, checkpoint_payload, restore_checkpoint_payload, close_stochastic_periodic_faces,
)


@dataclass
class Rates:
    ru: object
    rv: object
    rtheta: object
    rqv: object
    rqc: object
    rw: object = None


def test_off_binding_preserves_rates_and_performs_no_hook_or_state_access():
    class Hook:
        enabled = False
        def before_timestep(self, *args):
            pytest.fail("off hook called")
        def after_nonmicrophysics(self, *args, **kw):
            pytest.fail("off hook called")
    class State:
        def __getattr__(self, key):
            pytest.fail("off hook read state")
    binding, rates = StochasticPhysicsBinding(Hook(), member_id=0), object()
    binding.before_physics(State(), object())
    assert binding.after_nonmicrophysics(State(), object(), rates) is rates


def test_clock_index_and_nonmicrophysics_scope_are_held_without_mutating_driver_rates():
    calls = []
    class Hook:
        enabled = True
        skebs = object()
        completed_step = 6
        def before_timestep(self, step):
            calls.append(("before", step))
        def after_nonmicrophysics(self, rates, **kwargs):
            calls.append(("after", kwargs))
            return {name: value + np.float32(1) for name, value in rates.items()}
    clock = SimpleNamespace(step_count=7)
    factor = object()
    binding = StochasticPhysicsBinding(Hook(), member_id=19, clock=clock,
        mass_factor_provider=lambda state, cfg: factor)
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    before = values.tobytes()
    source = Rates(values, values, values, values, values, values)
    binding.before_physics(object(), object())
    result = binding.after_nonmicrophysics(object(), SimpleNamespace(open_x=True, open_y=True), source)
    assert calls == [("before", 7), ("after", {"tendency_scope": "nonmicrophysics", "mass_factors": factor})]
    assert values.tobytes() == before
    assert result.rqc is source.rqc and result.rw is source.rw
    np.testing.assert_array_equal(result.rtheta, values + np.float32(1))
    assert binding.applied_steps == 1


@pytest.mark.parametrize("periodic_x,periodic_y", [(True, True), (False, True), (True, False), (False, False)])
def test_forced_periodic_faces_copy_exact_zero_face_words_only(periodic_x, periodic_y):
    cfg = SimpleNamespace(open_x=not periodic_x, open_y=not periodic_y, specified=False, nested=False)
    ru_words = np.asarray([0x80000000, 0x00000001, 0x7fc00019, 0x3f800001,
                           0x40000001, 0x41000001, 0x42000001, 0x43000001], np.uint32).reshape(1, 2, 4)
    rv_words = np.asarray([0x80000000, 0x00000001, 0x7fc00019, 0x3f800001,
                           0x40000001, 0x41000001, 0x42000001, 0x43000001,
                           0x44000001, 0x45000001, 0x46000001, 0x47000001], np.uint32).reshape(1, 4, 3)
    ru, rv = ru_words.view(np.float32).copy(), rv_words.view(np.float32).copy()
    original_u, original_v = ru.tobytes(), rv.tobytes()
    scalar = np.asarray([0x80000000, 0x7fc00019], np.uint32).view(np.float32)
    rates = {"u": ru, "v": rv, "theta": scalar, "qv": scalar}
    close_stochastic_periodic_faces(rates, cfg)
    assert ru[..., :-1].tobytes() == ru_words.view(np.float32)[..., :-1].tobytes()
    assert rv[..., :-1, :].tobytes() == rv_words.view(np.float32)[..., :-1, :].tobytes()
    assert ru[..., -1].tobytes() == (ru[..., 0].tobytes() if periodic_x else ru_words.view(np.float32)[..., -1].tobytes())
    assert rv[..., -1, :].tobytes() == (rv[..., 0, :].tobytes() if periodic_y else rv_words.view(np.float32)[..., -1, :].tobytes())
    assert rates["theta"] is scalar and rates["qv"] is scalar
    if not periodic_x:
        assert ru.tobytes() == original_u
    if not periodic_y:
        assert rv.tobytes() == original_v


def test_spp_binding_precedes_compute_and_requires_actual_parameter_consumers():
    patterns, calls = {"pbl": object()}, []
    hook = SimpleNamespace(enabled=True, completed_step=-1, parameter_patterns=patterns,
                           before_timestep=lambda step: calls.append(("pattern", step)))
    binding = StochasticPhysicsBinding(hook, member_id=3)
    state = SimpleNamespace(physics=SimpleNamespace(bind_spp_patterns=lambda p: calls.append(("bind", p))))
    binding.before_physics(state, object())
    assert calls == [("pattern", 0), ("bind", patterns)]
    with pytest.raises(RuntimeError, match="parameter consumers"):
        binding.before_physics(SimpleNamespace(physics=object()), object())


def test_binding_uses_accepted_adaptive_dt_before_the_absolute_pattern_update():
    calls = []
    hook = SimpleNamespace(enabled=True, completed_step=3, parameter_patterns={},
        set_time_step=lambda dt: calls.append(("dt", dt)),
        before_timestep=lambda step: calls.append(("pattern", step)))
    binding = StochasticPhysicsBinding(hook, member_id=19,
                                      clock=SimpleNamespace(step_count=4))
    binding.before_physics(SimpleNamespace(), SimpleNamespace(dt=7.5))
    assert calls == [("dt", 7.5), ("pattern", 4)]


def test_complex_checkpoint_words_are_exact_and_require_complete_array_inventory():
    words = np.array([0, 0x80000000, 0x3f800001, 0xbf800001, 0x7fc00111, 0x00000001,
                      0x01000001, 0x81000001], dtype=np.uint32).reshape(2, 2, 2)
    spectrum = words.view(np.complex64).reshape(2, 2)
    class Hook:
        enabled = True
        def snapshot(self):
            return {"completed_step": 4, "spectrum": spectrum.copy(), "metadata": {"member_seed": 2 ** 64 - 1}}
        def restore(self, state):
            self.restored = state
    hook = Hook()
    binding = StochasticPhysicsBinding(hook, member_id=19, recipe_sha256="a" * 64)
    binding.applied_steps = 5
    metadata, arrays = checkpoint_payload(binding, array_module=np)
    assert len(arrays) == 1
    backing = next(iter(arrays.values()))
    assert backing.dtype == np.dtype("float32")
    assert backing.tobytes() == spectrum.tobytes()
    restore_checkpoint_payload(binding, metadata, arrays, array_module=np)
    assert hook.restored["spectrum"].tobytes() == spectrum.tobytes()
    assert hook.restored["metadata"]["member_seed"] == 2 ** 64 - 1
    with pytest.raises(ValueError, match="inventory"):
        restore_checkpoint_payload(binding, metadata, {}, array_module=np)
    with pytest.raises(ValueError, match="unbound"):
        restore_checkpoint_payload(binding, metadata, {**arrays, "unowned": backing}, array_module=np)
    wrong_member = StochasticPhysicsBinding(Hook(), member_id=0, recipe_sha256="a" * 64)
    with pytest.raises(ValueError, match="another member"):
        restore_checkpoint_payload(wrong_member, metadata, arrays, array_module=np)
