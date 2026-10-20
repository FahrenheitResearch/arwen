"""Resident restoration reuses verified maps without exposing aliases."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
from test_preflight_payload import sealed
from gpuwm import prepared_single_domain_forecast as forecast
from gpuwm.ingest import prepared_cache, prepared_store, prepared_writer


def _inputs(sealed):
    path, identity, reader, payload = sealed
    return SimpleNamespace(prepared_cache_path=path, cache_identity=identity,
                           cache_reader=reader, cache_payload=payload)


def test_resident_reader_returns_independent_writable_exact_arrays(sealed, monkeypatch):
    inputs = _inputs(sealed)
    monkeypatch.setattr(inputs.cache_reader, "read_array",
                        lambda _: pytest.fail("duplicate full file read"))
    monkeypatch.setattr(prepared_writer, "hash_arrays",
                        lambda *a, **k: pytest.fail("duplicate payload digest"))
    reader = forecast._resident_prepared_reader(inputs, None)
    for key in reader.arrays:
        expected = inputs.cache_payload.verified_array(key)
        first, second = reader.read_array(key), reader.read_array(key)
        assert first.tobytes() == second.tobytes() == expected.tobytes()
        assert first.dtype == expected.dtype and first.shape == expected.shape
        assert first.flags.writeable
        assert not np.shares_memory(first, expected)
        assert not np.shares_memory(first, second)
        first.fill(0)
        assert second.tobytes() == expected.tobytes()


@pytest.mark.parametrize("change", ["replace", "in_place", "header", "reader", "identity"])
def test_resident_rebinding_refuses_changed_authority(sealed, change):
    inputs = _inputs(sealed)
    path = inputs.prepared_cache_path
    filename = path / inputs.cache_reader.arrays["state/u"]["file"]
    if change == "replace":
        other = path.parent / "replacement.npy"
        np.save(other, inputs.cache_reader.read_array("state/u"), allow_pickle=False)
        other.replace(filename)
    elif change == "in_place":
        with filename.open("r+b") as handle:
            handle.seek(-1, 2); byte = handle.read(1)
            handle.seek(-1, 2); handle.write(bytes([byte[0] ^ 1]))
    elif change == "header":
        import hashlib
        header = json.loads((path / "header.json").read_text())
        header["metadata"]["user"]["changed"] = True
        basis = {key: header[key] for key in
                 ("schema", "identity", "metadata", "arrays", "payload_bytes")}
        header["content_sha256"] = hashlib.sha256(
            prepared_cache._canonical(basis).encode()).hexdigest()
        (path / "header.json").write_text(json.dumps(header))
    elif change == "reader":
        inputs.cache_reader = prepared_cache.PreparedCacheReader(
            path, expected_identity=inputs.cache_identity)
    else:
        inputs.cache_identity = {"source": "different"}
    with pytest.raises((prepared_cache.PreparedCacheCorruptError,
                        prepared_cache.PreparedCacheMismatchError,
                        prepared_store.PreparedStoreError)):
        forecast._resident_prepared_reader(inputs, None)


def test_resident_reader_refuses_change_during_copy(sealed, monkeypatch):
    inputs = _inputs(sealed)
    reader = forecast._resident_prepared_reader(inputs, None)
    original = prepared_store.np.array
    filename = inputs.prepared_cache_path / reader.arrays["state/u"]["file"]
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with filename.open("r+b") as handle:
            handle.seek(-1, 2); byte = handle.read(1)
            handle.seek(-1, 2); handle.write(bytes([byte[0] ^ 1]))
        return result
    monkeypatch.setattr(prepared_store.np, "array", changed)
    with pytest.raises(prepared_cache.PreparedCacheCorruptError,
                       match="changed during mapped use"):
        reader.read_array("state/u")


def test_streamed_head_and_old_install_keep_original_reader_routes(sealed):
    inputs = _inputs(sealed)
    inputs.cache_payload = SimpleNamespace()
    assert forecast._resident_prepared_reader(inputs, object()) is inputs.cache_reader
    inputs.cache_payload = None
    assert forecast._resident_prepared_reader(inputs, None) is None


def test_actual_host_restore_matches_original_with_writable_host_arrays(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from test_prepared_cache import _HOST_RESTORE_WITHOUT_CUPY

    check = r"""
from dataclasses import fields
from gpuwm import prepared_single_domain_forecast as forecast
from gpuwm.ingest.prepared_cache import PreparedCacheReader
from gpuwm.ingest.prepared_store import _CachePayload

reader = PreparedCacheReader(path, expected_identity=identity)
payload = _CachePayload(reader, log=lambda _: None)
def duplicate_read(*args, **kwargs):
    raise AssertionError("restore repeated the full file read and digest")
reader.read_array = duplicate_read
inputs = SimpleNamespace(prepared_cache_path=path, cache_identity=identity,
                         cache_reader=reader, cache_payload=payload)
candidate = restore_prepared_cache(
    path, expected_identity=identity, cfg=cfg, static=static, array_module=np,
    reader=forecast._resident_prepared_reader(inputs, None))
assert candidate.receipt == restored.receipt
for name in STATE_SERIALIZED_ATTRS:
    expected = getattr(restored.initial_result.state, name, None)
    if expected is not None:
        actual = getattr(candidate.initial_result.state, name)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), name
checks = []
for group in ("coord", "base"):
    expected = getattr(restored.initial_result, group)
    actual = getattr(candidate.initial_result, group)
    for field in fields(expected):
        value = getattr(expected, field.name)
        if isinstance(value, np.ndarray):
            checks.append((f"{group}/{field.name}", value,
                           getattr(actual, field.name)))
for name, value in restored.met.fields.items():
    checks.append((f"met/{name}", value, candidate.met.fields[name]))
for name in ("surface_pressure", "surface_qv"):
    checks.append((f"result/{name}", getattr(restored.initial_result, name),
                   getattr(candidate.initial_result, name)))
for key, expected, actual in checks:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, key
    assert actual.tobytes() == expected.tobytes(), key
    assert actual.flags.writeable, key
    assert not np.shares_memory(actual, payload.verified_array(key)), key
    actual.fill(0)
    assert payload.verified_array(key).tobytes() == expected.tobytes(), key
assert "cupy" not in sys.modules
print("retained-host-restore", len(checks))
"""
    completed = subprocess.run(
        [sys.executable, "-c", _HOST_RESTORE_WITHOUT_CUPY + check, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "retained-host-restore" in completed.stdout
