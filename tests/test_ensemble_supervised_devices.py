import pytest

from gpuwm.ensemble.supervised_devices import plan_ensemble_devices
from gpuwm.supervisor import GPUIdentity


def cards():
    return tuple(GPUIdentity(f"GPU-{index}abc", "driver", f"card-{index}", index) for index in range(4))


def test_every_visible_card_can_share_one_supervised_member_worker():
    gpus = cards()
    lease = plan_ensemble_devices(20, gpus)
    assert lease.visible == lease.locked == gpus
    assert lease.cuda_mask == "GPU-0abc,GPU-1abc,GPU-2abc,GPU-3abc"


def test_inherited_mask_keeps_authored_logical_member_indices():
    gpus = cards()
    lease = plan_ensemble_devices({"members": 20, "member_device_ids": [1]}, gpus,
                                 visibility="3,1")
    assert lease.visible == (gpus[3], gpus[1])
    assert lease.locked == (gpus[1],)
    assert lease.cuda_mask == "GPU-3abc,GPU-1abc"


def test_explicit_uuid_retains_the_single_card_door():
    gpus = cards()
    lease = plan_ensemble_devices(20, gpus, requested_uuid=gpus[2].uuid)
    assert lease.locked == lease.visible == (gpus[2],)
    assert lease.cuda_mask == gpus[2].uuid


@pytest.mark.parametrize("mask", ["", "-1", "0,0", "GPU-missing", "5"])
def test_no_hidden_card_can_escape_the_inherited_visibility(mask):
    with pytest.raises(ValueError, match="visibility"):
        plan_ensemble_devices(20, cards(), visibility=mask)


def test_explicit_logical_card_cannot_be_reinterpreted_after_a_pin():
    with pytest.raises(ValueError, match="inherited visible"):
        plan_ensemble_devices({"members": 20, "member_device_ids": [1]}, cards(),
                             requested_uuid="GPU-2abc")


def test_supervisor_locks_all_member_cards_before_one_worker_and_releases_them(tmp_path, monkeypatch):
    from gpuwm import supervisor
    from test_supervisor import _install_scripted_supervisor
    config, _, processes = _install_scripted_supervisor(monkeypatch, tmp_path,
        [[{"status": "complete", "step": 1, "exit": 0}]])
    config.write_text("[experiment]\nname='fake'\n[ensemble]\nmembers=20\n")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpus, locked, events = cards()[:2], set(), []
    monkeypatch.setattr(supervisor, "query_gpus", lambda: gpus)
    class Lock:
        def __init__(self, uuid, **kwargs):
            self.uuid = uuid
        def __enter__(self):
            locked.add(self.uuid)
            events.append(("lock", self.uuid))
        def __exit__(self, *error):
            locked.remove(self.uuid)
            events.append(("release", self.uuid))
    monkeypatch.setattr(supervisor, "GPUFileLock", Lock)
    def check(uuid, **kwargs):
        assert locked == {gpu.uuid for gpu in gpus}
        events.append(("preflight", uuid))
    monkeypatch.setattr(supervisor, "preflight_exclusive_gpu", check)
    popen = supervisor.subprocess.Popen
    def launch(*args, **kwargs):
        assert locked == {gpu.uuid for gpu in gpus}
        events.append(("worker", kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return popen(*args, **kwargs)
    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    result = supervisor.supervise_experiment(config, tmp_path / "out", poll_seconds=.05)
    assert result.heartbeat.status == "complete"
    assert not locked and len(processes) == 1
    assert events == [("lock", "GPU-0abc"), ("lock", "GPU-1abc"),
        ("preflight", "GPU-0abc"), ("preflight", "GPU-1abc"),
        ("worker", "GPU-0abc,GPU-1abc"), ("release", "GPU-1abc"), ("release", "GPU-0abc")]
