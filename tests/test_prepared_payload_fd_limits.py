"""Retained sealed payloads remain usable beyond the process fd limit."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gpuwm.ingest import prepared_cache, prepared_mmap, prepared_store


_SUBPROCESS = r'''
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import numpy as np
from gpuwm.ingest import prepared_cache, prepared_mmap, prepared_store

directory = Path(sys.argv[1])
limit, count = map(int, sys.argv[2:4])
backend = sys.argv[4]
directory.mkdir()
arrays = {}
for index in range(count):
    value = (np.arange(32, dtype=np.float32) + index).reshape(4, 8)
    filename = f'a{index:05d}.npy'
    np.save(directory / filename, value, allow_pickle=False)
    key = f'lbc/{index}/qv/south/tendency'
    arrays[key] = {'file': filename, 'shape': list(value.shape),
                   'dtype': str(value.dtype), 'nbytes': value.nbytes,
                   'sha256': prepared_cache._array_sha256(value)}
basis = {'schema': prepared_cache.PREPARED_CACHE_SCHEMA, 'identity': {},
         'metadata': {}, 'arrays': arrays,
         'payload_bytes': sum(row['nbytes'] for row in arrays.values())}
header = dict(basis, status='READY', content_sha256=hashlib.sha256(
    prepared_cache._canonical(basis).encode()).hexdigest())
(directory / 'header.json').write_text(json.dumps(header))
if backend == 'native':
    prepared_mmap._readonly_file_buffer = prepared_mmap._native_unix_buffer
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
reader = prepared_cache.PreparedCacheReader(directory, expected_identity={})
before = len(os.listdir('/proc/self/fd'))
payload = prepared_store._CachePayload(reader, log=lambda _: None)
after = len(os.listdir('/proc/self/fd'))
assert len(payload._maps) == count
views = [payload[key][1:3, 2:7] for key in arrays]
for index, (key, spec) in enumerate(arrays.items()):
    mapped = payload.verified_array(key)
    assert prepared_cache._array_sha256(mapped) == spec['sha256']
    expected = (np.arange(32, dtype=np.float32) + index).reshape(4, 8)
    assert mapped.tobytes() == expected.tobytes()
    assert np.shares_memory(views[index], mapped)
    assert not mapped.flags.writeable
    try:
        mapped.setflags(write=True)
    except ValueError:
        pass
    else:
        raise AssertionError('a retained mapping became writable')
payload.close()
gc.collect()
after_close = len(os.listdir('/proc/self/fd'))
for index, view in enumerate(views):
    expected = (np.arange(32, dtype=np.float32) + index).reshape(4, 8)[1:3, 2:7]
    assert view.tobytes() == expected.tobytes()
    assert not view.flags.writeable
assert after <= before + 2 and after_close <= before + 2, (before, after, after_close)
assert resource.getrlimit(resource.RLIMIT_NOFILE) == (limit, limit)
print(json.dumps({'array_count': count, 'limit_soft': limit, 'limit_hard': limit,
                  'backend': backend, 'fd_before': before, 'fd_retained': after,
                  'fd_after_close_with_views': after_close, 'all_bytes_equal': True}))
'''


@pytest.mark.skipif(sys.platform != 'linux', reason='requires Linux RLIMIT_NOFILE and fd inventory')
@pytest.mark.parametrize('limit,count', [(64, 160), (1024, 1093), (1024, 2118)])
@pytest.mark.parametrize('backend', ['default', 'native'])
def test_sealed_arrays_exceed_hard_descriptor_limit(tmp_path, limit, count, backend):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', GPUWM_NO_LOCAL_GPU='1')
    result = subprocess.run(
        [sys.executable, '-c', _SUBPROCESS, str(tmp_path / 'cache'),
         str(limit), str(count), backend],
        env=environment, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt['limit_soft'] == receipt['limit_hard'] == limit
    assert receipt['array_count'] == count > limit
    assert receipt['all_bytes_equal']
    assert receipt['fd_retained'] <= receipt['fd_before'] + 2


@pytest.mark.parametrize('number', [errno.EMFILE, errno.ENFILE])
@pytest.mark.parametrize('operation', ['stat', 'map'])
def test_retained_payload_resource_failure_is_not_corruption(
        tmp_path, monkeypatch, number, operation):
    from test_prepared_payload import _reader
    reader = _reader(tmp_path / 'cache', [np.arange(8, dtype=np.float32)])
    payload = prepared_store._CachePayload(reader, verify=False)
    error = OSError(number, 'file handles exhausted', str(reader.path))
    if operation == 'stat':
        original = Path.stat
        def fail(path, *args, **kwargs):
            if path.name == 'a00000.npy':
                raise error
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'stat', fail)
    else:
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(prepared_mmap, 'map_npy_readonly', fail)
    with pytest.raises(prepared_cache.PreparedCacheResourceLimitError) as caught:
        payload['field/0']
    assert caught.value.errno == number
    assert caught.value.__cause__ is error
    assert 'ulimit -n' in str(caught.value)
    assert not isinstance(caught.value, prepared_cache.PreparedCacheCorruptError)
