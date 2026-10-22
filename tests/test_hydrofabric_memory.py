"""Complete prepare-only resource inventory, before any CUDA/native work."""
import hashlib
import json
import sys

import pytest

from gpuwm.hydro.memory import MemoryContext, plan


@pytest.fixture
def fabric(tmp_path, monkeypatch):
    from gpuwm import hydrofabric
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "numpy", None)
    def forbidden(*args, **kwargs):
        raise AssertionError("memory hook must not load numerical/native/device resources")
    monkeypatch.setattr(hydrofabric, "RawPayloadReader", forbidden)
    monkeypatch.setattr(hydrofabric, "load_cuda", forbidden)
    roster = dict(node_ids=["reach-b", "reach-a"], catchment_ids=["cat-b", "cat-a", "cat-c"],
                  level_offsets=[0, 1, 2], reverse_level_offsets=[0, 1, 2])
    raw = json.dumps(roster).encode()
    (tmp_path / "network.json").write_bytes(raw)
    fields = {name: dict(path=name + ".bin", dtype=dtype, shape=shape, bytes=size, sha256="a"*64)
              for name, dtype, shape, size in (
                  ("node_ids", "<i8", [2], 16), ("catchment_to_node", "<i4", [3], 12),
                  ("unused", "<f8", [5, 2], 80), ("empty", "<f4", [0], 0))}
    manifest = dict(schema=hydrofabric.SCHEMA, node_count=2, catchment_count=3,
        auxiliary={"network.json": dict(sha256=hashlib.sha256(raw).hexdigest())}, files=fields,
        input=dict(sha256="b"*64, data_license="metadata contract fixture; no source data"),
        request_sha256="c"*64, node_order="prepared", upstream_order="native",
        numerics="fast with explicit strict replay", grid=dict(nx=11, ny=7),
        parent_grid_contract=dict(schema="arwen.nextgen.parent-grid.v1", authority="fixture"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, roster


def configuration(path, copies=1):
    return dict(stages=[dict(name=f"network{i}", factory="gpuwm.hydrofabric:prepare_resource",
                            phase="prepare", options=dict(manifest=str(path))) for i in range(copies)])


def test_all_uploads_are_independent_checkpointed_roots_with_sealed_axes(fabric):
    path, manifest, roster = fabric
    context = MemoryContext("coupled", "strict", 300, (49, 7, 11), 86400)
    value = plan(configuration(path, copies=2), context=context)
    owners = dict(value.stages)
    for name in ("network0", "network1"):
        owner = owners[name]
        assert {a.name: a.nbytes for a in owner.arrays} == {
            "node_ids": 16, "catchment_to_node": 12, "unused": 80, "empty": 0}
        assert all(a.checkpointed and a.alias_of is None for a in owner.arrays)
        assert owner.non_pool_bytes == 0 and not owner.gaps  # memcpy/empty only; no kernels
        evidence = json.loads(owner.evidence)
        for axis, ids in roster.items():
            assert evidence[axis] == ids
        assert evidence["manifest"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert evidence["payload_sha256"] == {key: row["sha256"] for key, row in manifest["files"].items()}
        assert evidence["context"]["numerics"] == "strict"
        assert evidence["prepared_numerics"] == manifest["numerics"]
        assert evidence["input"] == manifest["input"]
        assert evidence["parent_grid_contract"] == manifest["parent_grid_contract"]
        assert all(len(digest) == 64 for digest in evidence["sources"].values())
    row = value.inventory(members=3)
    # All three nonzero roots rounded independently for each load and member.
    assert row["resident_bytes"] == 3*512*2*3
    assert len([name for name in row["backings"] if name.startswith("network")]) == 8
    # Prepare-only resources also participate in driver checkpoint pack/restore.
    assert len(owners["_assembly"].arrays) == 16
    assert not row["complete"]
    assert any("generated status/copy kernel" in gap for gap in row["gaps"])
    assert not any("network0:" in gap or "network1:" in gap for gap in row["gaps"])


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m.update(schema="other"), "schema"),
    (lambda m: m["auxiliary"]["network.json"].update(sha256="b"*64), "metadata SHA256"),
    (lambda m: m["files"]["unused"].update(bytes=72), "byte length"),
    (lambda m: m["files"]["unused"].update(path="../escape.bin"), "escapes"),
    (lambda m: m["files"]["unused"].update(sha256="unsealed"), "SHA256"),
    (lambda m: m["files"]["unused"].update(shape=[5, -2]), "shape/dtype"),
    (lambda m: m["files"]["unused"].update(dtype=">f8"), "shape/dtype"),
])
def test_malformed_inventory_cannot_silently_underprice_uploads(fabric, mutation, match):
    path, manifest, _ = fabric
    mutation(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        plan(configuration(path), context=MemoryContext("standalone", "fast", 300))


@pytest.mark.parametrize('field, value', [
    pytest.param('catchment_ids', ['cat-a'], id='catchment_ids-value1'),
    pytest.param('reverse_level_offsets', [0, 1], id='reverse_level_offsets-value3'),
])
def test_resealed_invalid_roster_cannot_publish_false_ordered_axes(fabric, field, value):
    path, manifest, roster = fabric
    roster[field] = value
    raw = json.dumps(roster).encode()
    (path.parent / "network.json").write_bytes(raw)
    manifest["auxiliary"]["network.json"]["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        plan(configuration(path), context=MemoryContext("standalone", "fast", 300))
