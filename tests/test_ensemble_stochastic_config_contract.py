"""Constructor plumbing without CUDA arrays, transforms or surrogate physics."""
from dataclasses import replace

import pytest

from gpuwm.ensemble.stochastic import StochasticConfig, StochasticTimestepHook


def test_explicit_reference_and_half_amplitude_retain_original_pattern_constructor_identity(monkeypatch):
    import gpuwm.ensemble.stochastic as implementation
    calls = []
    monkeypatch.setattr(implementation, "WrfStochasticPattern", lambda config, shape, **kwargs:
        calls.append((config, shape, kwargs)) or object())
    kwargs = dict(dx=3000., dy=3000., dt=45., member_seed=319,
                  spp=True, spp_levels={"pbl": 49, "lsm": 9})
    StochasticTimestepHook((32, 40), **kwargs)
    defaults = calls[:]
    refs = {name: StochasticConfig.wrf_reference("spp_" + name) for name in ("pbl", "lsm")}
    calls.clear()
    StochasticTimestepHook((32, 40), spp_configs=refs, **kwargs)
    assert calls == defaults
    calls.clear()
    half = {name: replace(config, stddev=config.stddev * .5) for name, config in refs.items()}
    StochasticTimestepHook((32, 40), spp_configs=half, **kwargs)
    assert [(shape, arguments) for _, shape, arguments in calls] == [(shape, arguments) for _, shape, arguments in defaults]
    assert [config.stddev for config, _, _ in calls] == [config.stddev * .5 for config, _, _ in defaults]


@pytest.mark.parametrize("configs", [{}, {"pbl": StochasticConfig.wrf_reference("spp_lsm")},
                                    {"pbl": object()}, {"pbl": StochasticConfig.wrf_reference("spp_pbl"),
                                                       "lsm": StochasticConfig.wrf_reference("spp_lsm")}])
def test_invalid_parameter_contract_is_rejected_before_any_pattern_allocation(configs, monkeypatch):
    import gpuwm.ensemble.stochastic as implementation
    monkeypatch.setattr(implementation, "WrfStochasticPattern", lambda *args, **kwargs: pytest.fail("pattern allocation"))
    with pytest.raises(ValueError, match="spp_configs must exactly match"):
        StochasticTimestepHook((32, 40), dx=3000., dy=3000., dt=45., member_seed=319,
                              spp=True, spp_levels={"pbl": 49}, spp_configs=configs)


def test_disabled_low_level_hook_ignores_parameter_map_and_allocates_nothing(monkeypatch):
    import gpuwm.ensemble.stochastic as implementation
    monkeypatch.setattr(implementation, "WrfStochasticPattern", lambda *args, **kwargs: pytest.fail("pattern allocation"))
    hook = StochasticTimestepHook((32, 40), dx=3000., dy=3000., dt=45., member_seed=319,
        enabled=False, spp=True, spp_levels={"pbl": 49}, spp_configs={"invalid": object()})
    assert not hook.enabled and hook.spp == {} and hook.sppt is None and hook.skebs is None
