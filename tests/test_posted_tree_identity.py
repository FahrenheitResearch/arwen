"""Every domain of a posted tree binds each sealed document's own digest."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gpuwm import prepared_domain_tree_forecast as tree
from gpuwm.ingest import boundary_stream
from gpuwm.ingest.prepared_cache import PreparedCacheReader

from test_native_hrrr_posted import DOCUMENT_KEYS, _native_posted


def test_the_tree_authority_accepts_only_its_declared_document_placeholders():
    plan_sha256 = "1" * 64
    placeholder = boundary_stream.as_posted_placeholder(plan_sha256)
    head = {"basis": {"as_posted": {
        "input_plan_sha256": plan_sha256,
        "manifest_bound_identity_keys": ["input_manifest_sha256"],
        "document_bound_identity_keys": DOCUMENT_KEYS}}}
    authority = {"input_manifest_sha256": placeholder,
                 "bridge_manifest_sha256": placeholder,
                 "source_manifest_sha256": placeholder,
                 "static_cache_sha256": "e" * 64}
    assert tree._bound_authority(authority, head) == authority
    with pytest.raises(ValueError, match="static_cache_sha256"):
        tree._bound_authority(
            {**authority, "static_cache_sha256": placeholder}, head)
    with pytest.raises(ValueError, match="bridge_manifest_sha256"):
        tree._bound_authority(
            {**authority, "bridge_manifest_sha256":
             boundary_stream.as_posted_placeholder("2" * 64)}, head)


@pytest.mark.parametrize("changed", [None, *DOCUMENT_KEYS])
def test_the_tree_seal_holds_a_started_domain_to_its_own_documents(
        tmp_path, monkeypatch, changed):
    root, _ = _native_posted(tmp_path)
    head = boundary_stream.read_head(root)
    cache = root / head["basis"]["cache"]["directory"]
    identity = json.loads((cache / "header.json").read_text())["identity"]
    reader = PreparedCacheReader(cache, expected_identity=identity)
    static = root / "test-static"
    geometry = root / "test-geometry"
    static.write_bytes(b"static")
    geometry.write_bytes(b"geometry")

    def domain(cache_identity):
        return SimpleNamespace(
            grid_id=1, cache_reader=reader, cache_identity=cache_identity,
            static_path=static, geometry_receipt_path=geometry)

    sealed_identity = dict(identity)
    if changed is not None:
        sealed_identity[changed] = "f" * 64
    sealed_inputs = SimpleNamespace(
        prepared_head_sha256=head["head_sha256"],
        domains=(domain(sealed_identity),), statics_corridor=None,
        terrain_clock=None)
    started = SimpleNamespace(
        prepared_root=root, stream_head=head, preflight_arguments={},
        experiment=SimpleNamespace(domains=(SimpleNamespace(grid_id=1),)),
        domains=(domain(head["basis"]["cache"]["identity"]),),
        statics_corridor=None, terrain_clock=None)
    bindings = []

    def preflight(**binding):
        bindings.append(binding)
        return sealed_inputs

    monkeypatch.setattr(tree, "preflight_prepared_tree", preflight)
    stream = SimpleNamespace(
        sealed=lambda: True, wait_sealed=lambda: None,
        consumed_markers=lambda: {})
    if changed is None:
        assert tree._seal_tree_inputs(started, stream=stream) is sealed_inputs
        assert "preparation_receipt_sha256" in bindings[0]
    else:
        with pytest.raises(RuntimeError, match="d01.*cache identity"):
            tree._seal_tree_inputs(started, stream=stream)
