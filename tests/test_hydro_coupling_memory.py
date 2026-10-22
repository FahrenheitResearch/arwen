"""Allocation contracts on the coupled forecast shape; no numerical packets or CUDA."""
from dataclasses import replace
import json
import sys

import pytest

from gpuwm.hydro.memory import ArrayMemory, HydroMemoryPlan, MemoryContext, StageMemory
from gpuwm.hydro.noahowp.factory import prepare_exchange_memory as exchange
from gpuwm.hydro.output import prepare_history_memory as history


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    return MemoryContext("coupled", "fast", 30, (44, 136, 124), 86400)


def inventory(context, stages):
    return HydroMemoryPlan(context, tuple(stages.items()), ("resources unpriced",), "metadata").inventory()


def test_exchange_prices_owned_vectors_and_optional_parent_storage_without_copying_csr(context):
    n = 21805
    resource = StageMemory((ArrayMemory("exchange_offsets", (n+1,), "int64"),
        ArrayMemory("exchange_indices", (50000,), "int32"),
        ArrayMemory("exchange_weights", (50000,), "float32")),
        json.dumps({"context": {"numerics": "fast"}}))
    value = exchange(settings={}, options={"network_component": "network"}, context=context,
                     components={"network": resource})
    row = inventory(context, dict(network=resource, exchange=value))
    own = {a.name: a for a in value.arrays if a.alias_of is None}
    vectors = [a for a in own.values() if not a.name.startswith("workspace/")]
    assert len(vectors) == 7 and sum(a.nbytes for a in vectors) == 21805*28
    assert sum(a.nbytes for a in own.values() if a.name.startswith("workspace/")) == 2*136*124*4
    assert all(a.checkpointed for a in vectors)
    assert own["sticky_status"].status_name == "exchange"
    assert next(a for a in value.arrays if a.name == "offsets").alias_of == "network/exchange_offsets"
    assert not any(name.startswith("exchange/offsets") for name in row["backings"])
    assert any("parent Noah-MP" in gap for gap in value.gaps) and value.non_pool_bytes is None
    assert sum(v["allocated_bytes"] for k, v in row["backings"].items() if k.startswith("exchange/")) == 748032


def test_missing_exchange_source_and_parent_shape_never_claim_zero_allocations(context):
    value = exchange(settings={}, options={"network_component": "missing"},
                     context=replace(context, parent_shape=()), components={})
    assert any("destination count" in g for g in value.gaps)
    assert any("parent grid shape" in g for g in value.gaps)
    assert value.non_pool_bytes is None
    with pytest.raises(ValueError, match="coupled"):
        exchange(settings={}, options={}, context=replace(context, mode="standalone"), components={})


def test_exchange_retains_ordered_axis_and_effective_parent_cadence(context):
    resource = StageMemory((ArrayMemory("exchange_offsets", (4,), "int64"),
        ArrayMemory("exchange_indices", (10,), "int32"), ArrayMemory("exchange_weights", (10,), "float32")),
        json.dumps({"context": {"numerics": "fast"}, "catchment_ids": ["wb-8", "wb-2", "wb-5"]}))
    value = exchange(settings={}, options={"network_component": "network"}, context=context,
                     components={"network": resource})
    assert json.loads(value.evidence)["catchment_ids"] == ["wb-8", "wb-2", "wb-5"]
    assert json.loads(value.evidence)["period_seconds"] == 30
    resource = replace(resource, evidence=json.dumps({"context": {"numerics": "fast"},
        "catchment_ids": ["wb-8", "wb-8", "wb-5"]}))
    with pytest.raises(ValueError, match="catchment axis"):
        exchange(settings={}, options={"network_component": "network"}, context=context,
                 components={"network": resource})


def source(n=21805, dtype="float32"):
    width = 4 if dtype == "float32" else 8
    fields = {"reach_id": dict(array="reach_id", shape=[n], dtype="int64"),
              "nudge_m3_s": dict(array="nudge", shape=[n], dtype=dtype)}
    fields.update({field: dict(array="published", shape=[n], dtype=dtype, offset_bytes=i*n*width)
                   for i, field in enumerate(("discharge_m3_s", "velocity_m_s", "depth_m"))})
    return StageMemory((ArrayMemory("reach_id", (n,), "int64"),
        ArrayMemory("published", (3, n), dtype), ArrayMemory("nudge", (n,), dtype)),
        json.dumps({"context": {"numerics": "fast"}, "reach_output": fields}))


@pytest.mark.parametrize('dtype, width', [
    pytest.param('float32', 4, id='float32-4'),
])
def test_hourly_history_prices_full_slabs_and_borrows_source_views(context, dtype, width):
    route = source(dtype=dtype)
    value = history(settings={}, options=dict(source="routing", capacity_steps=2880, cadence_steps=120),
                    context=context, components={"routing": route})
    row = inventory(context, dict(routing=route, history=value))
    own = [a for a in value.arrays if a.alias_of is None]
    assert sum(a.nbytes for a in own) == 4*24*21805*width+16
    assert {a.name for a in value.arrays if a.checkpointed} == {
        "step", "overflow", "reach_id", "flow", "velocity", "depth", "nudge"}
    assert next(a for a in value.arrays if a.name == "inputs/depth_m").offset_bytes == 2*21805*width
    assert "history/reach_id" not in row["backings"]
    assert not value.gaps and value.non_pool_bytes is None


@pytest.mark.parametrize('offset', [
    pytest.param(-4, id='-4'),
    pytest.param(21805 * 12, id='261660'),
])
def test_history_rejects_outside_or_misaligned_views(context, offset):
    route = source()
    evidence = json.loads(route.evidence)
    evidence["reach_output"]["depth_m"]["offset_bytes"] = offset
    route = replace(route, evidence=json.dumps(evidence))
    with pytest.raises(ValueError, match="view"):
        history(settings={}, options=dict(source="routing", capacity_steps=2880),
                context=context, components={"routing": route})


def test_history_unknown_output_and_scheduled_overflow_stay_named(context):
    value = history(settings={"after_schedule": ["extra"]},
                    options=dict(source="unknown", capacity_steps=12), context=context, components={})
    assert any("reach-output" in gap for gap in value.gaps)
    assert any("duration" in gap for gap in value.gaps)
    assert any("capture roster" in gap for gap in value.gaps)
