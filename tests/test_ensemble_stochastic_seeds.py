"""Seed-label authority and constructor/checkpoint plumbing without CUDA."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.stochastic import StochasticConfig, StochasticTimestepHook
from gpuwm.ensemble.stochastic_seeds import (WRF_SEED_DEFAULTS, normalize_wrf_seed_labels,
                                           process_seed, seed_label_receipt)


def test_label_defaults_match_packaged_primary_registry():
    from gpuwm.wrf_namelist_registry import wrf_namelist_keys
    rows = wrf_namelist_keys()
    assert WRF_SEED_DEFAULTS == {key: int(rows[("stoch", key)]["default"])
                                for key in WRF_SEED_DEFAULTS}


def test_original_seed_is_exactly_retained_without_label_authority():
    original = (1 << 64) - 1
    assert process_seed(original, "sppt") == original
    assert normalize_wrf_seed_labels(None) is None and seed_label_receipt(None) is None


def test_labels_preserve_signed_integer_controls_and_process_independence():
    supplied = {"nens": 19, "iseed_sppt": -314}
    labels = normalize_wrf_seed_labels(supplied)
    assert supplied == {"nens": 19, "iseed_sppt": -314}
    assert labels == dict(WRF_SEED_DEFAULTS, **supplied)
    original = process_seed(873, "sppt", labels)
    assert original == 10486848286198949334
    assert process_seed(873, "sppt", dict(reversed(tuple(labels.items())))) == original
    changed = dict(labels, iseed_sppt=315)
    assert process_seed(873, "sppt", changed) != original
    assert process_seed(873, "spp_pbl", changed) == process_seed(873, "spp_pbl", labels)
    assert process_seed(874, "sppt", labels) != original
    assert process_seed(873, "skebs_psi", labels) != process_seed(873, "skebs_theta", labels)
    assert "not WRF" in seed_label_receipt(labels)["rng_policy"]


@pytest.mark.parametrize("labels", [{"members": 10}, {"nens": True}, {"iseed_sppt": 1.0},
    {"nens": 1 << 31}, {"iseed_spp_pbl": -(1 << 31) - 1}, [1]])
def test_invalid_seed_authority_rejected(labels):
    with pytest.raises(ValueError, match="seed_labels|seed labels"):
        normalize_wrf_seed_labels(labels)


class Pattern:
    def __init__(self, config, shape, **kwargs):
        self.config, self.seed = config, kwargs["member_seed"]
        self.completed_step = -1
        self.validations = 0

    def snapshot(self):
        return {"seed": self.seed, "completed_step": -1, "metadata": {"dt": 45.}}

    def _validate_state(self, state):
        self.validations += 1
        assert state["seed"] == self.seed


def hook(monkeypatch, labels=None, **kwargs):
    import gpuwm.ensemble.stochastic as implementation
    monkeypatch.setattr(implementation, "WrfStochasticPattern", Pattern)
    return StochasticTimestepHook((32, 40), dx=3000., dy=3000., dt=45., member_seed=873,
        sppt=StochasticConfig.wrf_reference("sppt"),
        skebs_psi=StochasticConfig.wrf_reference("skebs_psi"),
        skebs_theta=StochasticConfig.wrf_reference("skebs_theta"),
        spp=True, spp_levels={"pbl": 49}, wrf_seed_labels=labels, **kwargs)


def test_labels_reach_every_original_pattern_constructor_and_checkpoint(monkeypatch):
    labels = {"nens": 9, "iseed_sppt": 117}
    bound = hook(monkeypatch, labels)
    processes = {"sppt": bound.sppt, "skebs_psi": bound.skebs.psi,
                 "skebs_theta": bound.skebs.theta, "spp_pbl": bound.spp["pbl"]}
    assert {name: process.seed for name, process in processes.items()} == {
        name: process_seed(873, name, labels) for name in processes}
    state = bound.snapshot()
    assert state["wrf_seed_labels"] == seed_label_receipt(labels)
    bound.validate_snapshot(state)
    altered = deepcopy(state)
    altered["wrf_seed_labels"]["labels"]["iseed_rand_pert"] += 1
    before = [process.validations for process in processes.values()]
    with pytest.raises(ValueError, match="WRF seed labels"):
        bound.validate_snapshot(altered)
    assert before == [process.validations for process in processes.values()]


def test_default_constructor_and_checkpoint_remain_original(monkeypatch):
    bound = hook(monkeypatch)
    assert bound.sppt.seed == bound.skebs.psi.seed == bound.skebs.theta.seed == bound.spp["pbl"].seed == 873
    state = bound.snapshot()
    assert set(state) == {"enabled", "completed_step", "sppt", "skebs", "spp_levels", "spp"}
    bound.validate_snapshot(state)
    labelled = hook(monkeypatch, {})
    with pytest.raises(ValueError, match="WRF seed labels"):
        labelled.validate_snapshot(state)


def test_disabled_hook_ignores_labels_and_constructs_no_process(monkeypatch):
    import gpuwm.ensemble.stochastic as implementation
    monkeypatch.setattr(implementation, "WrfStochasticPattern", lambda *args, **kwargs:
                        pytest.fail("disabled process allocation"))
    bound = StochasticTimestepHook((32, 40), dx=3000., dy=3000., dt=45., member_seed=873,
        enabled=False, sppt=StochasticConfig.wrf_reference("sppt"), wrf_seed_labels={"invalid": object()})
    assert bound.wrf_seed_labels is None and "wrf_seed_labels" not in bound.snapshot()


def test_effectively_off_hook_ignores_labels_without_any_seed_resolution(monkeypatch):
    import gpuwm.ensemble.stochastic_seeds as seeds
    monkeypatch.setattr(seeds, "normalize_wrf_seed_labels", lambda *args: pytest.fail("off label resolution"))
    monkeypatch.setattr(seeds, "process_seed", lambda *args: pytest.fail("off process key construction"))
    bound = StochasticTimestepHook((32, 40), dx=3000., dy=3000., dt=45., member_seed=873,
        enabled=True, wrf_seed_labels={"invalid": object()})
    assert not bound.enabled and bound.wrf_seed_labels is None
    assert set(bound.snapshot()) == {"enabled", "completed_step", "sppt", "skebs", "spp_levels", "spp"}
    bound.validate_snapshot(bound.snapshot())


def test_provider_retains_global_member_seed_and_optional_label_constructor(monkeypatch):
    import gpuwm.ensemble.stochastic_model as implementation
    from test_ensemble_stochastic_model import config
    from gpuwm.ensemble.request import EnsembleRequest
    controls = {"sppt": True, "wrf_seed_labels": {"nens": 19, "iseed_sppt": 117}}
    # Public requests refuse uncalibrated amplitudes; the provider keeps
    # its seed contract for the calibration campaign.
    with pytest.raises(ValueError, match="calibrated against observations"):
        EnsembleRequest(4, stochastic=controls)
    provider = implementation.StochasticModelProvider.from_mapping(controls)
    calls = []
    monkeypatch.setattr(implementation, "StochasticTimestepHook", lambda *args, **kwargs:
        calls.append(kwargs) or SimpleNamespace(enabled=True, parameter_patterns={}))
    provider.bind_state(state=SimpleNamespace(), cfg=config(), member_id=19, seed=873)
    assert calls[0]["member_seed"] == 873
    assert calls[0]["wrf_seed_labels"] == normalize_wrf_seed_labels(controls["wrf_seed_labels"])


def test_reconstruction_reuses_labelled_owner_and_rejects_changed_labels_before_mutation():
    from test_ensemble_stochastic_model import existing_owner
    from gpuwm.ensemble.stochastic_model import StochasticModelProvider
    cfg, owner = existing_owner()
    labels = normalize_wrf_seed_labels({"nens": 19})
    owner.hook.wrf_seed_labels = labels
    owner.hook.spp["pbl"].seed = process_seed(273, "spp_pbl", labels)
    calls = []
    state = SimpleNamespace(_ensemble_stochastic=owner,
        physics=SimpleNamespace(bind_spp_patterns=lambda value: calls.append(value)))
    provider = StochasticModelProvider(spp=True, wrf_seed_labels=labels)
    new_clock = object()
    assert provider.bind_state(state=state, cfg=cfg, member_id=19, seed=273, clock=new_clock) is owner
    assert owner.clock is new_clock and calls == [owner.hook.parameter_patterns]
    changed = StochasticModelProvider(spp=True, wrf_seed_labels=dict(labels, iseed_rand_pert=4))
    with pytest.raises(ValueError, match="WRF seed labels"):
        changed.bind_state(state=state, cfg=cfg, member_id=19, seed=273, clock=object())
    assert owner.clock is new_clock and len(calls) == 1


def test_runtime_authority_receipt_retains_labels_without_prepared_header_mutation(tmp_path):
    from test_ensemble_stochastic_authority import prepared
    from gpuwm.ensemble.stochastic_model import StochasticModelProvider
    from gpuwm.ensemble.stochastic_authority import configure_member_inputs
    original = prepared(tmp_path)
    provider = StochasticModelProvider.from_mapping({"sppt": True, "wrf_seed_labels": {"nens": 19}})
    bound, receipt = configure_member_inputs(provider, original)
    assert bound is original and receipt["wrf_seed_labels"] == seed_label_receipt({"nens": 19})
    assert bound.domains[0].cache_identity is original.domains[0].cache_identity
