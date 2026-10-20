"""Sealed tree restoration retains checked bytes and independent writes."""
from types import SimpleNamespace
import json
import hashlib
import numpy as np
import pytest
from test_preflight_payload import sealed
from gpuwm import prepared_domain_tree_forecast as tree
from gpuwm.ingest import prepared_cache, prepared_store, prepared_writer


def bundle_for(sealed):
    path, identity, reader, payload = sealed
    return SimpleNamespace(cache=path, cache_identity=identity,
                           cache_reader=reader, cache_payload=payload)


def test_tree_restores_writable_bytes_without_second_read_or_digest(sealed, monkeypatch):
    bundle = bundle_for(sealed)
    monkeypatch.setattr(bundle.cache_reader, "read_array",
                        lambda _: pytest.fail("duplicate full read"))
    monkeypatch.setattr(prepared_writer, "hash_arrays",
                        lambda *a, **k: pytest.fail("duplicate digest"))
    reader = tree._resident_domain_reader(bundle)
    for key in reader.arrays:
        expected = bundle.cache_payload.verified_array(key)
        first, second = reader.read_array(key), reader.read_array(key)
        assert first.tobytes() == second.tobytes() == expected.tobytes()
        assert first.dtype == expected.dtype and first.shape == expected.shape
        assert first.flags.writeable and not np.shares_memory(first, expected)
        assert not np.shares_memory(first, second)
        first.fill(0)
        assert second.tobytes() == expected.tobytes()


@pytest.mark.parametrize("change", ["replace", "in_place", "header", "reader", "identity"])
def test_tree_refuses_changed_authority(sealed, change):
    bundle = bundle_for(sealed)
    filename = bundle.cache / bundle.cache_reader.arrays["state/u"]["file"]
    if change == "replace":
        replacement = bundle.cache.parent / "replacement.npy"
        np.save(replacement, bundle.cache_reader.read_array("state/u"))
        replacement.replace(filename)
    elif change == "in_place":
        with filename.open("r+b") as handle:
            handle.seek(-1, 2); byte = handle.read(1)
            handle.seek(-1, 2); handle.write(bytes([byte[0] ^ 1]))
    elif change == "header":
        path = bundle.cache / "header.json"
        header = json.loads(path.read_text())
        header["metadata"]["user"]["changed"] = True
        basis = {k: header[k] for k in ("schema", "identity", "metadata", "arrays", "payload_bytes")}
        header["content_sha256"] = hashlib.sha256(prepared_cache._canonical(basis).encode()).hexdigest()
        path.write_text(json.dumps(header))
    elif change == "reader":
        bundle.cache_reader = prepared_cache.PreparedCacheReader(bundle.cache, expected_identity=bundle.cache_identity)
    else:
        bundle.cache_identity = {"source": "different"}
    with pytest.raises((prepared_cache.PreparedCacheCorruptError,
                        prepared_cache.PreparedCacheMismatchError,
                        prepared_store.PreparedStoreError)):
        tree._resident_domain_reader(bundle)


def test_tree_stream_and_old_install_preserve_reader_routes(sealed):
    bundle = bundle_for(sealed)
    bundle.cache_payload = object()
    assert tree._resident_domain_reader(bundle, object()) is bundle.cache_reader
    bundle.cache_payload = None
    assert tree._resident_domain_reader(bundle) is None


def test_missing_native_verifier_keeps_original_preflight(sealed):
    bundle = bundle_for(sealed)
    assert tree.prepared_single._sealed_preflight_payload(bundle.cache_reader) is None
