"""A preflight reuses its verified immutable maps without changing readers."""
from dataclasses import fields
import json
import shutil

import numpy as np
import pytest

from gpuwm.ingest import prepared_cache, prepared_store, prepared_writer
from gpuwm import prepared_single_domain_forecast as forecast


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    from test_prepared_cache import _fixture
    initial, met, boundaries = _fixture()
    identity = {"source": "payload-check"}
    path = tmp_path / "cache"
    prepared_cache.write_prepared_cache(
        path, identity=identity, initial_result=initial, met=met,
        boundaries=boundaries)
    reader = prepared_cache.PreparedCacheReader(path, expected_identity=identity)
    # Exercise the same artifact and original full digest fallback on hosts
    # without the additive bridge. Native hashing has separate built tests.
    monkeypatch.setattr(prepared_writer, "native_hasher", lambda: None)
    payload = prepared_store._CachePayload(reader, log=lambda _: None)
    return path, identity, reader, payload


def test_retained_view_is_the_same_verified_mapping_and_public_reads_mutable(sealed):
    path, identity, reader, payload = sealed
    view = payload.read_only_reader()
    mapped = view.read_array("state/u")
    assert mapped is payload.verified_array("state/u")
    assert isinstance(mapped, np.ndarray)
    assert np.shares_memory(mapped, payload["state/u"])
    assert not mapped.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        mapped[...] = 0.0
    mutable = reader.read_array("state/u")
    assert mutable.flags.writeable
    original = mapped.copy()
    mutable.fill(0.0)
    assert np.array_equal(mapped, original)
    assert payload.require_binding(path, identity, reader=reader) is reader
    payload.close()
    # Store construction retains its own coord/base map references. Clearing
    # the verification registry must not close their underlying mappings.
    assert np.array_equal(mapped, original)
    assert not mapped.flags.writeable


def test_retained_binding_checks_same_maps_without_repeating_hash(sealed, monkeypatch):
    path, identity, reader, payload = sealed
    monkeypatch.setattr(reader, "read_array", lambda _: pytest.fail("duplicate full read"))
    monkeypatch.setattr(prepared_writer, "hash_arrays",
                        lambda *a, **k: pytest.fail("duplicate content hash"))
    assert payload.require_binding(path, identity, reader=reader) is reader
    assert payload.read_only_reader().verify_all()["content_sha256"] == reader.content_sha256
    assert payload.read_only_reader().read_array("base/phb") is payload["base/phb"]


def test_retained_payload_refuses_a_different_path_even_with_identical_bytes(sealed, tmp_path):
    path, identity, reader, payload = sealed
    other = tmp_path / "other"
    shutil.copytree(path, other)
    with pytest.raises(prepared_store.PreparedStoreError, match="different reader or path"):
        payload.require_binding(other, identity, reader=reader)


def test_retained_payload_refuses_another_reader_owner(sealed):
    path, identity, reader, payload = sealed
    other = prepared_cache.PreparedCacheReader(path, expected_identity=identity)
    with pytest.raises(prepared_store.PreparedStoreError, match="different reader or path"):
        payload.require_binding(path, identity, reader=other)


def test_retained_payload_keeps_expected_identity_refusal(sealed):
    path, _identity, reader, payload = sealed
    with pytest.raises(prepared_cache.PreparedCacheMismatchError):
        payload.require_binding(path, {"source": "other"}, reader=reader)


def test_retained_payload_refuses_a_new_valid_header(sealed):
    path, identity, reader, payload = sealed
    header_path = path / "header.json"
    header = json.loads(header_path.read_text())
    header["metadata"]["user"]["changed"] = True
    basis = {key: header[key] for key in ("schema", "identity", "metadata", "arrays", "payload_bytes")}
    import hashlib
    header["content_sha256"] = hashlib.sha256(
        prepared_cache._canonical(basis).encode("utf-8")).hexdigest()
    header_path.write_text(json.dumps(header))
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="header changed"):
        payload.require_binding(path, identity, reader=reader)


@pytest.mark.parametrize("change", ["replace", "in_place"])
def test_retained_payload_refuses_changed_array_before_store_use(sealed, change):
    path, identity, reader, payload = sealed
    filename = path / reader.arrays["state/u"]["file"]
    if change == "replace":
        replacement = path.parent / "replacement.npy"
        np.save(replacement, reader.read_array("state/u"), allow_pickle=False)
        try:
            replacement.replace(filename)
        except PermissionError:
            pytest.skip("the host refuses replacement of an open read-only mapping")
    else:
        with filename.open("r+b") as handle:
            handle.seek(-1, 2)
            byte = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="changed during mapped use"):
        payload.require_binding(path, identity, reader=reader)


def test_retained_manifest_cannot_be_changed_in_memory(sealed):
    path, identity, reader, payload = sealed
    reader.arrays["state/u"]["sha256"] = "0" * 64
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="header changed"):
        payload.require_binding(path, identity, reader=reader)


@pytest.mark.parametrize("part", ["coord_scalars", "base_scalars"])
def test_retained_header_scalars_cannot_be_changed_in_memory(sealed, part):
    path, identity, reader, payload = sealed
    scalars = reader.header["metadata"][part]
    key = next(iter(scalars))
    scalars[key] = "changed"
    with pytest.raises(prepared_cache.PreparedCacheCorruptError, match="header changed"):
        payload.require_binding(path, identity, reader=reader)


@pytest.mark.parametrize("layout,head", [("hierarchy-d01-v1", None),
                                        ("mapped-hierarchy-d01-v1", None),
                                        ("portable-single-domain-v2", {}),
                                        ("mapped-direct-d01-v1", {})])
def test_heads_and_hierarchies_keep_the_original_reader_path(monkeypatch, layout, head):
    monkeypatch.setattr(prepared_writer, "native_hasher",
                        lambda: pytest.fail("unsupported retained path"))
    assert forecast._retained_preflight_payload(None, layout=layout, head=head) is None


def test_old_bridge_keeps_original_preflight_path(monkeypatch):
    monkeypatch.setattr(prepared_writer, "native_hasher", lambda: None)
    assert forecast._retained_preflight_payload(
        None, layout="portable-single-domain-v2", head=None) is None


def test_retained_mapping_is_excluded_from_semantic_input_equality_and_repr():
    spec = next(item for item in fields(forecast.PreparedForecastInputs)
                if item.name == "cache_payload")
    assert not spec.compare and not spec.repr


def test_built_hasher_preflight_retains_real_maps_and_exact_receipt(tmp_path, monkeypatch):
    from test_prepared_single_domain_forecast import (
        _prepared_fixture, _preflight_fixture)
    try:
        entry = prepared_writer.native_hasher()
    except (OSError, RuntimeError) as error:
        pytest.skip(str(error))
    if entry is None:
        pytest.skip("built native mapped hashing is unavailable")
    fixture = _prepared_fixture(tmp_path, "gfs")
    from types import SimpleNamespace
    monkeypatch.setattr(forecast, "validate_native_lambert_contract",
                        lambda *a, **k: SimpleNamespace(source="gfs"))
    monkeypatch.setattr(forecast, "verify_native_static_receipt",
                        lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(forecast, "load_native_static_cache",
                        lambda path, grid, ny, nx: {"STATIC": np.ones((ny, nx))})
    inputs = _preflight_fixture(fixture)
    assert inputs.cache_payload is not None
    assert inputs.cache_reader.content_sha256 == fixture.content_sha256
    view = inputs.cache_payload.read_only_reader()
    assert all(not view.read_array(key).flags.writeable for key in view.arrays)
    assert inputs.cache_payload.require_binding(
        inputs.prepared_cache_path, inputs.cache_identity,
        reader=inputs.cache_reader) is inputs.cache_reader
