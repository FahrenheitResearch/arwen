"""Descriptor exhaustion is a resource failure, not corrupted input."""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from gpuwm.ingest import prepared_cache, prepared_store, prepared_writer


@pytest.fixture
def cache(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    array = np.arange(32, dtype=np.float32).reshape(4, 8)
    filename = "a00000.npy"
    np.save(directory / filename, array, allow_pickle=False)
    spec = {
        "file": filename, "shape": list(array.shape),
        "dtype": str(array.dtype), "nbytes": array.nbytes,
        "sha256": prepared_cache._array_sha256(array),
    }
    basis = {
        "schema": prepared_cache.PREPARED_CACHE_SCHEMA,
        "identity": {}, "metadata": {}, "arrays": {"state/qv": spec},
        "payload_bytes": array.nbytes,
    }
    header = dict(basis, status="READY", content_sha256=hashlib.sha256(
        prepared_cache._canonical(basis).encode("utf-8")).hexdigest())
    (directory / "header.json").write_text(json.dumps(header), encoding="utf-8")
    return directory, spec


def _assert_resource_failure(call, error):
    with pytest.raises(prepared_cache.PreparedCacheResourceLimitError) as caught:
        call()
    observed = caught.value
    assert observed.errno == error.errno
    assert observed.filename == error.filename
    assert observed.__cause__ is error
    assert isinstance(observed, OSError)
    assert not isinstance(observed, prepared_cache.PreparedCacheCorruptError)
    assert not isinstance(observed, ValueError)
    assert "resource limit" in str(observed)
    assert "ulimit -n" in str(observed)
    assert errno.errorcode[error.errno] in str(observed)
    assert "corrupt" not in str(observed).lower()
    if error.errno == errno.ENFILE:
        assert "system open-file limit" in str(observed)


@pytest.mark.parametrize("number", [errno.EMFILE, errno.ENFILE])
def test_payload_open_preserves_resource_limit(cache, monkeypatch, number):
    directory, spec = cache
    source = directory / spec["file"]
    error = OSError(number, "open failed", str(source))
    original = Path.open

    def fail(path, *args, **kwargs):
        if path == source:
            raise error
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    _assert_resource_failure(
        lambda: prepared_cache.read_manifest_array(directory, "state/qv", spec),
        error)


@pytest.mark.parametrize("number", [errno.EMFILE, errno.ENFILE])
def test_payload_load_preserves_resource_limit(cache, monkeypatch, number):
    directory, spec = cache
    error = OSError(number, "mapping failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(np, "load", fail)
    _assert_resource_failure(
        lambda: prepared_cache.read_manifest_array(directory, "state/qv", spec),
        error)


@pytest.mark.parametrize("number", [errno.EMFILE, errno.ENFILE])
def test_header_read_preserves_resource_limit(cache, monkeypatch, number):
    directory, _spec = cache
    error = OSError(number, "header open failed", str(directory / "header.json"))

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "read_text", fail)
    _assert_resource_failure(
        lambda: prepared_cache.PreparedCacheReader(directory, expected_identity={}),
        error)


@pytest.mark.parametrize("number", [errno.EMFILE, errno.ENFILE])
def test_inventory_read_preserves_resource_limit(cache, monkeypatch, number):
    directory, _spec = cache
    error = OSError(number, "directory open failed", str(directory))

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "iterdir", fail)
    _assert_resource_failure(
        lambda: prepared_cache.PreparedCacheReader(directory, expected_identity={}),
        error)


def test_existing_resource_failure_is_not_wrapped_again():
    error = prepared_cache.PreparedCacheResourceLimitError(errno.EMFILE, "resource limit")
    with pytest.raises(prepared_cache.PreparedCacheResourceLimitError) as caught:
        prepared_cache._raise_if_resource_limit(error, context="prepared cache")
    assert caught.value is error
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("number", [errno.ENOENT, errno.EACCES, errno.EIO])
def test_other_io_failure_is_not_reclassified_as_a_descriptor_limit(number):
    assert prepared_cache._raise_if_resource_limit(
        OSError(number, "read failed"), context="prepared cache") is None


@pytest.mark.parametrize("field,value", [
    ("shape", [8, 4]), ("dtype", "float64"),
    ("nbytes", 64), ("sha256", "0" * 64),
])
def test_real_payload_manifest_failure_remains_corruption(cache, field, value):
    directory, spec = cache
    wrong = dict(spec)
    wrong[field] = value
    with pytest.raises(prepared_cache.PreparedCacheCorruptError,
                       match="fails its manifest"):
        prepared_cache.read_manifest_array(directory, "state/qv", wrong)


def test_truncated_payload_remains_corruption(cache):
    directory, spec = cache
    (directory / spec["file"]).write_bytes(b"\x93NUMPY")
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="unreadable"):
        prepared_cache.read_manifest_array(directory, "state/qv", spec)


def test_invalid_header_digest_remains_corruption(cache):
    directory, _spec = cache
    path = directory / "header.json"
    header = json.loads(path.read_text(encoding="utf-8"))
    header["content_sha256"] = "0" * 64
    path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises(prepared_cache.PreparedCacheCorruptError,
                       match="header content digest mismatch"):
        prepared_cache.PreparedCacheReader(directory, expected_identity={})


_EXHAUSTED_SUBPROCESS = r'''
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import numpy as np
from gpuwm.ingest import prepared_cache

directory = Path(sys.argv[1])
reader = prepared_cache.PreparedCacheReader(directory, expected_identity={})
expected = reader.read_array('state/qv').tobytes()
path = directory / reader.arrays['state/qv']['file']
before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
owned = []
try:
    while True:
        try:
            owned.append(os.open(os.devnull, os.O_RDONLY))
        except OSError as error:
            assert error.errno == errno.EMFILE
            break
    try:
        reader.read_array('state/qv')
    except prepared_cache.PreparedCacheResourceLimitError as error:
        assert error.errno == errno.EMFILE
        assert isinstance(error.__cause__, OSError)
        assert error.__cause__.errno == errno.EMFILE
        assert not isinstance(error, prepared_cache.PreparedCacheCorruptError)
        assert 'ulimit -n' in str(error)
        observed_errno = error.errno
        cause_errno = error.__cause__.errno
        message = str(error)
    else:
        raise AssertionError('an exhausted descriptor budget was not reported')
finally:
    for descriptor in owned:
        os.close(descriptor)
assert reader.read_array('state/qv').tobytes() == expected
assert hashlib.sha256(path.read_bytes()).hexdigest() == before_sha
assert resource.getrlimit(resource.RLIMIT_NOFILE) == (64, 64)
print(json.dumps({'limit_soft': 64, 'limit_hard': 64,
                  'errno': observed_errno, 'cause_errno': cause_errno,
                  'message': message, 'payload_unchanged': True,
                  'owned_descriptors_closed': len(owned)}))
'''


@pytest.mark.skipif(sys.platform != 'linux', reason='requires Linux RLIMIT_NOFILE')
def test_real_descriptor_exhaustion_names_resource_and_remedy(cache):
    directory, _spec = cache
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', GPUWM_NO_LOCAL_GPU='1')
    result = subprocess.run(
        [sys.executable, '-c', _EXHAUSTED_SUBPROCESS, str(directory)],
        env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt['limit_soft'] == receipt['limit_hard'] == 64
    assert receipt['errno'] == receipt['cause_errno'] == errno.EMFILE
    assert 'ulimit -n' in receipt['message']
    assert receipt['payload_unchanged']
    assert receipt['owned_descriptors_closed'] > 0


@pytest.mark.parametrize('number', [errno.EMFILE, errno.ENFILE])
@pytest.mark.parametrize('operation', ['native_hasher', 'hash_arrays'])
def test_native_verification_preserves_resource_limit(cache, monkeypatch,
                                                     number, operation):
    directory, _spec = cache
    reader = prepared_cache.PreparedCacheReader(directory, expected_identity={})
    error = OSError(number, 'native verification cannot open a file')

    def fail(*args, **kwargs):
        raise error

    if operation == 'hash_arrays':
        monkeypatch.setattr(prepared_writer, 'native_hasher', lambda: object())
    monkeypatch.setattr(prepared_writer, operation, fail)
    _assert_resource_failure(
        lambda: prepared_store._CachePayload(reader, log=lambda _: None), error)
