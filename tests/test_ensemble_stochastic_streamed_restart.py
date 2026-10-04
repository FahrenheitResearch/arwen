"""Full-owner spectra remain separate from store carriers and stage atomically."""
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.ensemble import stochastic_execution as codec
from gpuwm.ensemble.stochastic_execution import StochasticPhysicsBinding
from gpuwm.io import restart
from tilestream import restart_stream, physics_inventory
from test_streamed_tree_restart import _member


class Hook:
    enabled = True
    def __init__(self, words, seed=42):
        self.words, self.seed, self.restores = words, seed, 0
    def snapshot(self):
        return {"seed": self.seed, "completed_step": 1, "spectrum": self.words.view(np.complex64).reshape(2, 2)}
    def validate_snapshot(self, snapshot):
        if snapshot["seed"] != self.seed or snapshot["spectrum"].shape != (2, 2):
            raise ValueError("wrong seed or spectrum shape")
    def restore(self, snapshot):
        self.validate_snapshot(snapshot)
        self.words = snapshot["spectrum"].view(np.float32).view(np.uint32).copy().reshape(2, 2, 2)
        self.restores += 1


def numpy_codec(monkeypatch):
    write, decode, restore = codec.checkpoint_payload, codec.decode_checkpoint_payload, codec.restore_checkpoint_payload
    monkeypatch.setattr(codec, "checkpoint_payload", lambda binding: write(binding, array_module=np))
    monkeypatch.setattr(codec, "decode_checkpoint_payload", lambda metadata, arrays, **kwargs: decode(metadata, arrays, array_module=np))
    monkeypatch.setattr(codec, "restore_checkpoint_payload", lambda binding, metadata, arrays: restore(binding, metadata, arrays, array_module=np))


@pytest.mark.parametrize("wrong", [False, True])
def test_streamed_spectra_validate_before_store_mutation_and_restore_once(tmp_path, monkeypatch, wrong):
    numpy_codec(monkeypatch)
    cfg, state, _, _ = _member(tmp_path, monkeypatch, 1)
    words = np.array([0, 0x80000000, 0x3f800001, 0xbf800001, 1, 0x01000001, 0x7f7fffff, 0x807fffff],
                     np.uint32).reshape(2, 2, 2)
    source = StochasticPhysicsBinding(Hook(words), member_id=19, recipe_sha256="fixture")
    source.applied_steps = 2
    setup = restart_stream.capture_domain_setup(state)
    original = {key: value.copy() for key, value in physics_inventory.carrier_manifest(state).items()}
    info = restart_stream.write_streamed_restart(tmp_path / "stochastic.npz", original, cfg,
        setup=setup, template_state=state, scalars=physics_inventory.carrier_scalars(state),
        check_pinned=False, stochastic_binding=source)
    header, payload = restart._load_restart(info.path, with_arrays=True)
    spectral = {key: value for key, value in payload.items() if key.startswith("stochastic/")}
    assert len(spectral) == 1 and header["ensemble_stochastic"]["member_id"] == 19
    assert next(iter(spectral.values())).view(np.uint32).tobytes() == words.tobytes()
    target = {key: np.zeros_like(value) for key, value in original.items()}
    resumed = StochasticPhysicsBinding(Hook(np.zeros_like(words), seed=99 if wrong else 42),
        member_id=19, recipe_sha256="fixture")
    if wrong:
        with pytest.raises(restart_stream.RestartRefused, match="wrong seed"):
            restart_stream.validate_streamed_restart(info.path, target, cfg, setup=setup,
                template_state=state, stochastic_binding=resumed)
        assert all(not value.any() for value in target.values()) and resumed.hook.restores == 0
    else:
        validated = restart_stream.validate_streamed_restart(info.path, target, cfg, setup=setup,
            template_state=state, stochastic_binding=resumed)
        assert all(not value.any() for value in target.values()) and resumed.hook.restores == 0
        validated.apply()
        assert resumed.hook.words.tobytes() == words.tobytes() and resumed.hook.restores == 1
        assert resumed.applied_steps == 2
        assert all(target[key].tobytes() == value.tobytes() for key, value in original.items())
        with pytest.raises(restart_stream.RestartRefused, match="already applied"):
            validated.apply()
        assert resumed.hook.restores == 1
