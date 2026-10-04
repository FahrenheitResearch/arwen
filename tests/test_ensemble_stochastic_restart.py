from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.io import restart
from gpuwm.ensemble import stochastic_execution


def test_off_checkpoint_validation_has_no_device_or_payload_operations(monkeypatch):
    monkeypatch.setattr(stochastic_execution, "decode_checkpoint_payload",
                        lambda *args, **kw: pytest.fail("off decoder touched"))
    restart._validate_ensemble_stochastic_checkpoint({}, {}, SimpleNamespace())
    restart._validate_ensemble_stochastic_checkpoint({}, {},
        SimpleNamespace(_ensemble_stochastic=SimpleNamespace(enabled=False)))
    assert restart.classify_state_attr("_ensemble_stochastic") == "infra"


def test_missing_or_unbound_stochastic_payload_cannot_splice_restart():
    for header, arrays, state in (
        ({"ensemble_stochastic": {}}, {}, SimpleNamespace()),
        ({}, {}, SimpleNamespace(_ensemble_stochastic=SimpleNamespace(enabled=True))),
        ({}, {"stochastic/unknown": np.zeros((2,), np.float32)}, SimpleNamespace())):
        with pytest.raises(restart.RestartMismatchError, match="enabling"):
            restart._validate_ensemble_stochastic_checkpoint(header, arrays, state)


def test_complete_stochastic_validation_precedes_any_spectrum_restore(monkeypatch):
    events = []
    binding = SimpleNamespace(enabled=True,
        validate_identity=lambda snapshot: events.append(("identity", snapshot)),
        hook=SimpleNamespace(validate_snapshot=lambda snapshot: events.append(("validate", snapshot)),
                             restore=lambda snapshot: pytest.fail("validation mutated spectrum")))
    snapshot = {"member_id": 19, "hook": {"completed_step": 7}}
    def decode(metadata, arrays):
        assert metadata == {"test": "metadata"}
        assert set(arrays) == {"stochastic/spectrum"}
        return snapshot
    monkeypatch.setattr(stochastic_execution, "decode_checkpoint_payload", decode)
    restart._validate_ensemble_stochastic_checkpoint({"ensemble_stochastic": {"test": "metadata"}},
        {"state/u": object(), "stochastic/spectrum": object()},
        SimpleNamespace(_ensemble_stochastic=binding))
    assert events == [("identity", snapshot), ("validate", snapshot["hook"])]


def test_provider_rejection_is_a_restart_mismatch_without_mutation(monkeypatch):
    monkeypatch.setattr(stochastic_execution, "decode_checkpoint_payload",
                        lambda *args, **kw: {"hook": {}})
    def invalid(snapshot):
        raise ValueError("wrong seed")
    binding = SimpleNamespace(enabled=True, validate_identity=invalid)
    with pytest.raises(restart.RestartMismatchError, match="wrong seed"):
        restart._validate_ensemble_stochastic_checkpoint({"ensemble_stochastic": {}}, {},
            SimpleNamespace(_ensemble_stochastic=binding))


class _SpectrumHook:
    enabled = True
    def __init__(self, words):
        self.spectrum = words.copy().view(np.complex64)
        self.completed_step = 2

    def snapshot(self):
        return {"completed_step": self.completed_step,
                "spp": {"conv": {"spectrum": self.spectrum.copy()}}}

    def validate_snapshot(self, snapshot):
        array = snapshot["spp"]["conv"]["spectrum"]
        if array.dtype != self.spectrum.dtype or array.shape != self.spectrum.shape:
            raise ValueError("spectrum shape or dtype changed")

    def restore(self, snapshot):
        self.validate_snapshot(snapshot)
        self.spectrum[...] = snapshot["spp"]["conv"]["spectrum"]
        self.completed_step = snapshot["completed_step"]


def _host_spectrum_codec(monkeypatch):
    encode, decode = stochastic_execution.checkpoint_payload, stochastic_execution.decode_checkpoint_payload
    monkeypatch.setattr(stochastic_execution, "checkpoint_payload",
        lambda binding, **kwargs: encode(binding, array_module=np))
    monkeypatch.setattr(stochastic_execution, "decode_checkpoint_payload",
        lambda metadata, arrays, **kwargs: decode(metadata, arrays, array_module=np))


def test_managed_stochastic_spectra_roundtrip_through_the_actual_restart_reader(monkeypatch, tmp_path):
    from test_restart import _cfg, _shim_state, _fill_setup, _fill_serialized
    _host_spectrum_codec(monkeypatch)
    cfg = _cfg()
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    _fill_serialized(source, seed=723)
    words = np.array([0, 0x80000000, 0x00000001, 0x3f800001], np.uint32).view(np.float32)
    binding = stochastic_execution.StochasticPhysicsBinding(_SpectrumHook(words), member_id=19,
                                                          recipe_sha256="a" * 64)
    binding.applied_steps = 3
    source._ensemble_stochastic = binding
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)
    header, stored = restart._load_restart(path, with_arrays=True)
    assert "stochastic/binding/hook/spp/conv/spectrum" in stored
    target = _shim_state(cfg, monkeypatch)
    _fill_setup(target)
    _fill_serialized(target, seed=724)
    restored = stochastic_execution.StochasticPhysicsBinding(_SpectrumHook(np.ones_like(words)),
        member_id=19, recipe_sha256="a" * 64)
    target._ensemble_stochastic = restored
    restart.restore_restart(path, target, cfg)
    assert restored.hook.spectrum.tobytes() == binding.hook.spectrum.tobytes()
    assert restored.applied_steps == 3 and restored.hook.completed_step == 2
    assert target.thp.tobytes() == source.thp.tobytes()


def test_unbound_stochastic_namespace_still_refuses_before_any_model_write(monkeypatch, tmp_path):
    from test_restart import (_cfg, _shim_state, _fill_setup, _fill_serialized,
                              _rewrite_restart_archive)
    _host_spectrum_codec(monkeypatch)
    cfg = _cfg()
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    _fill_serialized(source, seed=723)
    words = np.ones(4, np.float32)
    source._ensemble_stochastic = stochastic_execution.StochasticPhysicsBinding(
        _SpectrumHook(words), member_id=19, recipe_sha256="a" * 64)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)
    def inject(payload, header):
        payload["stochastic/unbound"] = words.copy()
    tampered = _rewrite_restart_archive(path, tmp_path / "tampered.npz", inject)
    target = _shim_state(cfg, monkeypatch)
    _fill_setup(target)
    _fill_serialized(target, seed=724)
    hook = _SpectrumHook(np.zeros_like(words))
    target._ensemble_stochastic = stochastic_execution.StochasticPhysicsBinding(
        hook, member_id=19, recipe_sha256="a" * 64)
    before = (target.thp.tobytes(), hook.spectrum.tobytes())
    with pytest.raises(restart.RestartMismatchError, match="unbound spectrum"):
        restart.restore_restart(tampered, target, cfg)
    assert (target.thp.tobytes(), hook.spectrum.tobytes()) == before
