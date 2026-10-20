import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gpuwm.io.restart import _load_restart, RestartMismatchError
from gpuwm.io.restart_prefetch import prefetch_restart, restart_payload_scope
from gpuwm.supervisor import validate_manifest_checkpoint, CheckpointValidationError


def checkpoint(path, *, shape=(2, 2)):
    header = {'array_manifest': {'state/u': {'shape': list(shape), 'dtype': 'float32'}}}
    np.savez(path, __gpuwm_restart_header__=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
             **{'state/u': np.ones((2, 2), dtype=np.float32)})


def test_prefetched_manifest_is_still_validated_and_does_not_read_again(tmp_path, monkeypatch):
    path = tmp_path/'parent.npz'
    checkpoint(path)
    payload = prefetch_restart(path)
    path.chmod(0o444)
    # A read-only checkpoint is valid on POSIX. Scope notices metadata changes,
    # including chmod; prefetch the now archived input before using it.
    payload = prefetch_restart(path)
    with restart_payload_scope((payload,)):
        monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('archive read on forecast path'))
        assert validate_manifest_checkpoint(path) == path.resolve()
        header, arrays = _load_restart(path, with_arrays=True)
        assert not arrays['state/u'].flags.writeable
        header['array_manifest'].clear()
        assert _load_restart(path, with_arrays=False)[0]['array_manifest']


def test_prefetched_payload_does_not_bypass_manifest_shape_refusal(tmp_path):
    path = tmp_path/'parent.npz'
    checkpoint(path, shape=(3, 3))
    with restart_payload_scope((prefetch_restart(path),)):
        with pytest.raises(CheckpointValidationError, match='shape/dtype'):
            validate_manifest_checkpoint(path)


def test_replaced_parent_cannot_silently_change_ensemble(tmp_path):
    path = tmp_path/'parent.npz'
    checkpoint(path)
    payload = prefetch_restart(path)
    checkpoint(path)
    with restart_payload_scope((payload,)):
        with pytest.raises(RestartMismatchError, match='changed after prefetch'):
            _load_restart(path, with_arrays=True)


def test_payload_scope_is_not_shared_with_unrelated_thread(tmp_path, monkeypatch):
    path = tmp_path/'parent.npz'
    checkpoint(path)
    payload = prefetch_restart(path)
    original = np.load
    opened = []
    def read(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'load', read)
    with restart_payload_scope((payload,)):
        _load_restart(path, with_arrays=True)
        with ThreadPoolExecutor(1) as pool:
            pool.submit(_load_restart, path, with_arrays=True).result()
    assert opened == [path]
