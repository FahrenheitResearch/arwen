"""Fire checkpoint payloads survive actual disk archives and refuse corruption."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import (_cfg, _identity_bound_physics_state,
                          _fill_serialized, _rewrite_restart_archive)


class _HostFire:
    """A host carrier exercising the production restart archive path."""
    def __init__(self, seed, setup="static-v1"):
        rng = np.random.default_rng(seed)
        self.data = {name: rng.standard_normal(shape).astype(np.float32)
                     for name, shape in (("grid.lfn", (14, 18)),
                                         ("grid.tign", (14, 18)),
                                         ("grid.lfn_s3", (14, 18)),
                                         ("grid.moisture.fmc", (5, 4, 6)),
                                         ("rthfrten", (5, 4, 6)),
                                         ("uah", (4, 7)))}
        self.data["grid.lfn"].view(np.uint32)[0, :3] = (0x80000000, 1, 0x7fc12345)
        self.setup = setup
        self.clock = dict(version=1, time_seconds=37.25, step_count=9,
                          moisture_nexttime=600.0)

    def arrays(self):
        return self.data

    def metadata(self):
        return dict(self.clock)

    def setup_identity(self):
        return dict(schema=1, static=self.setup, refinement=[3, 2])

    def validate_restart(self, arrays, metadata):
        if metadata["version"] != 1 or not np.isfinite(metadata["time_seconds"]):
            raise ValueError("invalid fire clock")

    def restore(self, arrays, metadata):
        self.data = {name: array.copy() for name, array in arrays.items()}
        self.clock = dict(metadata)


def _state(monkeypatch, seed):
    cfg = _cfg()
    state, driver = _identity_bound_physics_state(cfg, monkeypatch)
    _fill_serialized(state, seed)
    driver.fire = _HostFire(seed)
    from gpuwm.core.physics import PhysicsTendencies
    driver.fire_tendencies = PhysicsTendencies(
        ru=np.full_like(state.u, 0.125), rv=np.full_like(state.v, -0.25),
        rtheta=np.full_like(state.p, 2.5), rqv=np.full_like(state.p, 0.03125),
        rqc=np.zeros_like(state.p))
    return cfg, state, driver


def test_fire_disk_checkpoint_preserves_all_words_clocks_and_held_tendencies(monkeypatch, tmp_path):
    cfg, state, source = _state(monkeypatch, 13)
    state.elapsed_seconds = 37.25
    path = restart.write_restart(tmp_path / "fire.npz", state, cfg)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(archive[restart._HEADER_KEY].tobytes())
        assert header["driver"]["fire"] == source.fire.metadata()
        assert {key for key in archive.files if key.startswith("fire/")} == {
            "fire/" + key for key in source.fire.arrays()}
    _, live, target = _state(monkeypatch, 19)
    restart.restore_restart(path, live, cfg)
    assert target.fire.metadata() == source.fire.metadata()
    for key, array in source.fire.arrays().items():
        assert target.fire.arrays()[key].tobytes() == array.tobytes()
    for component in restart.TENDENCY_COMPONENTS:
        expected = getattr(source.fire_tendencies, component)
        got = getattr(target.fire_tendencies, component)
        assert got is None if expected is None else got.tobytes() == expected.tobytes()
    assert live.elapsed_seconds == state.elapsed_seconds


def test_restored_fire_composition_does_not_mutate_or_replay_held_pbl(monkeypatch, tmp_path):
    from gpuwm.core.physics import PhysicsTendencies
    cfg, state, source = _state(monkeypatch, 41)
    assert not source.radiation_active and not source.cu_physics
    # A real fire constructor creates this independent composition target.
    source.tendencies = PhysicsTendencies.zeros(state)
    path = restart.write_restart(tmp_path / "fire-composition.npz", state, cfg)
    _, live, target = _state(monkeypatch, 47)
    restart.restore_restart(path, live, cfg)
    assert target.tendencies is not target.pbl_tendencies
    held = {name: value.tobytes() for name, value in vars(target.pbl_tendencies).items()
            if value is not None}
    target._compose_tendencies(cfg)
    first = {name: value.tobytes() for name, value in vars(target.tendencies).items()
             if value is not None}
    target._compose_tendencies(cfg)
    assert held == {name: value.tobytes() for name, value in vars(target.pbl_tendencies).items()
                    if value is not None}
    assert first == {name: value.tobytes() for name, value in vars(target.tendencies).items()
                     if value is not None}


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "identity", "clock", "held"])
def test_fire_corruption_refuses_before_any_model_mutation(monkeypatch, tmp_path, fault):
    cfg, state, source = _state(monkeypatch, 23)
    path = restart.write_restart(tmp_path / "source.npz", state, cfg)

    def corrupt(payload, header):
        key = "fire/grid.lfn"
        if fault == "missing":
            del payload[key]
        elif fault == "shape":
            payload[key] = payload[key].reshape(1, *payload[key].shape)
        elif fault == "dtype":
            payload[key] = payload[key].astype(np.float64)
        elif fault == "identity":
            header["driver"]["fire_setup_identity"]["static"] = "changed"
        elif fault == "clock":
            header["driver"]["fire"]["time_seconds"] = "broken"
        else:
            del payload["driver/fire_tendencies/rtheta"]

    broken = _rewrite_restart_archive(path, tmp_path / "broken.npz", corrupt)
    _, live, target = _state(monkeypatch, 29)
    before_state = {name: getattr(live, name).tobytes()
                    for name in restart.STATE_SERIALIZED_ATTRS if getattr(live, name, None) is not None}
    before_fire = {key: value.tobytes() for key, value in target.fire.arrays().items()}
    with pytest.raises(restart.RestartMismatchError):
        restart.restore_restart(broken, live, cfg)
    assert {name: getattr(live, name).tobytes() for name in before_state} == before_state
    assert {key: value.tobytes() for key, value in target.fire.arrays().items()} == before_fire


def test_fire_without_attached_driver_has_no_silent_restore_route():
    with pytest.raises(restart.RestartMismatchError, match="no PhysicsDriver"):
        restart._validate_member_namespaces({"fire/grid.lfn": np.zeros((3, 3), np.float32)},
                                            SimpleNamespace(), None, "archive.npz", 6)
