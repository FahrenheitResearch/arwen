"""Data-owner preparation contracts; metadata only, no numerical stand-ins."""
import hashlib
import json
import sys

import pytest

from gpuwm.hydro.memory import ArrayMemory, MemoryContext, StageMemory, plan


@pytest.fixture
def packet(tmp_path, monkeypatch):
    # Opening a device or reading a numerical packet would make planning too
    # late. Only the prepared manifest exists for this CPU contract check.
    monkeypatch.setitem(sys.modules, "cupy", None)
    metadata = dict(schema="arwen.nextgen-forcing.v1", shape=[48, 8, 3],
        files={"forcing.f32": "<f4", "forcing.f64": "<f8"},
        data_sha256={"forcing.f32": "a"*64, "forcing.f64": "b"*64},
        catchment_ids=["cat-67", "cat-27", "cat-52"], interval_seconds=3600,
        time_origin="2015-12-01T00:00:00Z", source_profile_sha256="c"*64,
        area_weights_sha256="d"*64)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(metadata))
    return path, metadata


def settings(path, native=False):
    return dict(numerics="strict", interval_seconds=3600, stages=[dict(name="forcing",
        factory="gpuwm.hydro.forcing_inputs:prepare_series",
        options=dict(manifest=str(path), native_precision=native))])


def context(duration=48*3600):
    return MemoryContext("standalone", "strict", 3600, duration_seconds=duration)


@pytest.mark.parametrize("native,dtype,width,digest", [
    (False, "float32", 4, "a"*64), (True, "float64", 8, "b"*64)])
def test_forcing_hook_prices_packet_precision_and_pins_external_manifest(packet, native, dtype, width, digest):
    path, _ = packet
    value = plan(settings(path, native), context=context())
    owner = dict(value.stages)["forcing"]
    assert [(a.name, a.shape, a.dtype) for a in owner.arrays] == [
        ("frames", (48, 8, 3), dtype), ("fields", (8, 3), dtype),
        ("clock_seconds", (3,), "float64"), ("status", (3,), "int32")]
    assert sum(a.nbytes for a in owner.arrays) == (48*8*3 + 8*3)*width + 3*12
    assert all(a.checkpointed for a in owner.arrays)
    evidence = json.loads(owner.evidence)
    assert evidence["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence["payload_sha256"] == digest
    assert evidence["catchment_ids"] == ["cat-67", "cat-27", "cat-52"]
    assert evidence["context"]["duration_seconds"] == 48*3600
    row = value.inventory(members=2)
    assert row["resident_bytes"] == value.inventory()["resident_bytes"]*2
    assert not row["complete"]  # uncompiled kernel resources stay unpriced
    assert any("kernel/local-memory" in gap for gap in row["gaps"])


def history(ids, fields):
    return dict(name="history", factory="gpuwm.hydro.module_history:prepare",
                options=dict(capacity_steps=48, catchment_ids=ids,
                             origin="2015-12-01T00:00:00Z", fields=fields))


def test_history_resolves_forcing_field_views_and_keeps_pointer_tables_out_of_checkpoint(packet):
    path, metadata = packet
    cfg = settings(path, True)
    cfg["stages"].append(history(metadata["catchment_ids"], [
        dict(name="temperature", component="forcing", field="T2D", units="K"),
        dict(name="rain", component="forcing", field="RAINRATE", units="mm/s")]))
    value = plan(cfg, context=context())
    owner = dict(value.stages)["history"]
    arrays = {a.name: a for a in owner.arrays}
    assert arrays["temperature"].shape == (48, 3)
    assert arrays["rain"].dtype == "float64"
    pointers = [a for a in owner.arrays if a.name.startswith("workspace/")]
    assert len(pointers) == 3 and sum(a.nbytes for a in pointers) == 24*2
    assert all(not a.checkpointed and a.lifetime == "resident" for a in pointers)
    # Borrowed field references introduce no additional backing allocation.
    assert len(arrays) == 7
    assert not owner.gaps
    row = value.inventory()
    assert not any("workspace/" in a["name"] for a in row["arrays"] if a["checkpointed"])
    assert len([name for name in row["backings"] if name.startswith("history/workspace/")]) == 3


def test_unknown_source_fields_and_axis_remain_named_partial(packet):
    from gpuwm.hydro.data_memory import module_history_memory
    _, metadata = packet
    row = history(metadata["catchment_ids"], [
        dict(name="known", component="runoff", field="flux", units="m"),
        dict(name="unknown", component="runoff", field="runtime_view", units="m")])
    owner = module_history_memory(settings={}, options=row["options"], context=context(),
        components={"runoff": StageMemory((ArrayMemory("flux", (3,), "float64"),), "opaque-owner-pin")})
    assert any("ordered catchment axis" in gap for gap in owner.gaps)
    assert any("runtime_view" in gap and "shape/dtype" in gap for gap in owner.gaps)
    assert "unknown" not in {a.name for a in owner.arrays}
    assert sum(a.nbytes for a in owner.arrays if a.name.startswith("workspace/")) == 48


def test_reordered_source_axis_cannot_be_published_as_history_labels(packet):
    path, metadata = packet
    cfg = settings(path)
    cfg["stages"].append(history(list(reversed(metadata["catchment_ids"])), [
        dict(name="rain", component="forcing", field="RAINRATE", units="mm/s")]))
    with pytest.raises(ValueError, match="source catchment axis"):
        plan(cfg, context=context())


def test_duration_and_scheduled_advances_never_silently_shrink_capacity(packet):
    path, metadata = packet
    cfg = settings(path)
    cfg["after_schedule"] = {"owner": "resolved at runtime"}
    cfg["stages"].append(history(metadata["catchment_ids"], [
        dict(name="rain", component="forcing", field="RAINRATE", units="mm/s")]))
    row = plan(cfg, context=context(49*3600)).inventory()
    assert not row["complete"]
    assert any("packet does not cover" in gap for gap in row["gaps"])
    assert any("capacity does not cover" in gap for gap in row["gaps"])
    assert any("resolved capture roster" in gap for gap in row["gaps"])
    assert next(a for a in row["arrays"] if a["name"] == "history/rain")["shape"] == (48, 3)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(shape=[48, 7, 3]), "shape"),
    (lambda p: p.update(catchment_ids=["cat-67"]*3), "catchment axis"),
    (lambda p: p["files"].update({"forcing.f32": "<f8"}), "dtype"),
    (lambda p: p["data_sha256"].update({"forcing.f32": "unsealed"}), "SHA256"),
])
def test_invalid_prepared_axis_precision_and_payload_pin_are_refused(packet, mutation, match):
    path, metadata = packet
    mutation(metadata)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        plan(settings(path), context=context())
