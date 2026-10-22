"""Metadata contract checks; no routing values, model mocks or GPU execution."""
from dataclasses import replace
import hashlib
import json
import math
import sys

import pytest

from gpuwm.hydro.memory import ArrayMemory, HydroMemoryPlan, MemoryContext, StageMemory
from gpuwm.hydro.reservoirs import CONTROLS, PARAMETERS, SOURCE_PIN, prepare_memory as reservoirs
from gpuwm.hydro.troute_mc import prepare_memory as mc


@pytest.fixture
def packet(tmp_path):
    n, m, r, s = 6742, 6740, 2, 7
    specs = {"node_ids": ((n,), "<i8"), "routing_parameters": ((n, 8), "<f4"),
             "upstream_offsets": ((n + 1,), "<i4"), "upstream_indices": ((n - 1,), "<i4"),
             "mc_ordered_nodes": ((m,), "<i4"), "selected": ((r,), "<i4"),
             "reservoir_nodes": ((r,), "<i4"), "extra_unused": ((131,), "<f8")}
    rspecs = {"parameters": ((9, r), "<f4"), "offsets": ((r + 1,), "<i8"),
              "controls": ((8, r), "<f8"), "carriers": ((3, r), "<f8")}
    rspecs.update({k: ((r,), "<f4") for k in ("previous_inflow", "inflow", "lateral", "height",
                    "initial_height", "levelpool_release", "levelpool_height")})
    rspecs.update({k: ((r,), "<i4") for k in ("types", "indices", "dynamic_type", "active_source", "fault")})
    rspecs.update({k: ((r,), "<f8") for k in ("clock", "release", "assimilated")})
    rspecs.update({k: ((s,), "<f8") for k in ("series", "times")})
    rspecs["extra_mapped"] = ((3,), "<f8")
    specs.update({"r_" + k: v for k, v in rspecs.items()})
    files = {k: {"shape": shape, "dtype": dtype, "path": k + ".bin",
                 "bytes": math.prod(shape) * int(dtype[-1]),
                 "sha256": hashlib.sha256(k.encode()).hexdigest()} for k, (shape, dtype) in specs.items()}
    manifest = {"schema": "arwen.nextgen.hydrofabric.v1", "files": files,
                "auxiliary": {"network.json": {"sha256": "metadata-roster-pin"}},
                "layers": [{"role": "lakes", "row_count": r}],
                "routing": {"validated_reservoir_mapping": True, "mc_level_offsets": [0, m],
                            "reservoir_levels": [{"level": 0, "rows": "selected"}] * 2,
                            "reservoir_node_indices": "reservoir_nodes"},
                "reservoirs": {"arrays": {k: "r_" + k for k in rspecs}, "parameter_columns": PARAMETERS,
                               "control_columns": CONTROLS, "source_pin": SOURCE_PIN, "validated_csr": True}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    context = MemoryContext("standalone", "strict", 3600)
    options = {"fabric_manifest": str(path)}
    roptions = dict(options, lateral_component="runoff", reservoir_component="reservoirs",
                    publish_channels=True, dt_seconds=300)
    settings = {"time_origin": "2023-01-01T00:00:00Z", "stages": [
        {"name": "reservoirs", "factory": "gpuwm.hydro.reservoirs:prepare", "options": options},
        {"name": "routing", "factory": "gpuwm.hydro.troute_mc:prepare", "options": roptions}]}
    runoff = StageMemory((ArrayMemory("routing_lateral", (n,), "float32", checkpointed=True),),
                         json.dumps({"context": {"numerics": "strict"}}))
    return path, manifest, settings, context, options, roptions, runoff


def construct(packet):
    _, _, settings, context, options, roptions, runoff = packet
    resource = reservoirs(settings=settings, options=options, context=context, components={})
    route = mc(settings=settings, options=roptions, context=context,
               components={"runoff": runoff, "reservoirs": resource})
    return resource, route


def test_local_uploads_carriers_and_embedded_roots_match_runtime_contract(packet):
    path, manifest, settings, context, _, _, runoff = packet
    before = set(sys.modules)
    resource, route = construct(packet)
    assert not ({"cupy", "numpy"} & (set(sys.modules) - before))
    assert list(path.parent.iterdir()) == [path]  # No payload files exist or were needed.
    expected = set(manifest["reservoirs"]["arrays"])
    assert {a.name for a in resource.arrays if a.checkpointed} == expected
    assert {a.name for a in resource.arrays if a.status} == {"fault"}
    assert not any(a.status for a in route.arrays)
    assert len([a for a in route.arrays if a.name.startswith("reservoir_selection/")]) == 1
    assert len([a for a in route.arrays if a.name.startswith("reservoir_node_map/")]) == 1
    assert {a.name.removeprefix("reservoir/") for a in route.arrays
            if a.name.startswith("reservoir/")} == expected
    stages = (("runoff", runoff), ("reservoirs", resource), ("routing", route))
    inventory = HydroMemoryPlan(context, stages, ("test resources unknown",), "test").inventory(members=3)
    assert "routing/reservoir/fault" not in inventory["backings"]
    assert next(a for a in route.arrays if a.name == "reservoir/fault").alias_of == "reservoirs/fabric/r_fault"
    assert next(a for a in route.arrays if a.name == "lateral").alias_of == "runoff/routing_lateral"
    assert "routing/fabric/extra_unused" in inventory["backings"]
    own = [a for a in route.arrays if a.alias_of is None and not a.name.startswith("fabric/")]
    assert sum(a.nbytes for a in own) == 60 * 6742 + 20 * 6740
    # Each direct load_cuda invocation allocates its own full upload roster.
    expected_roots = sum((f["bytes"] + 511) // 512 * 512 for f in manifest["files"].values()) * 2
    expected_roots += sum((a.nbytes + 511) // 512 * 512 for a in own) + (6742 * 4 + 511) // 512 * 512
    assert inventory["resident_bytes"] == expected_roots * 3
    assert resource.non_pool_bytes is None and route.non_pool_bytes is None
    assert json.loads(route.evidence)["manifest"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_shared_fabric_preserves_backings_borrowed_state_and_nudging(packet):
    path, manifest, settings, context, _, roptions, runoff = packet
    from gpuwm.hydro.routing_memory import _DTYPES
    fabric = StageMemory(tuple(ArrayMemory(k, tuple(v["shape"]), _DTYPES[v["dtype"]])
                               for k, v in manifest["files"].items()),
                         json.dumps({"context": {"numerics": "strict"}}))
    settings["stages"].insert(0, {"name": "network", "factory": "gpuwm.hydrofabric:prepare_resource",
                                    "options": {"manifest": str(path)}})
    options = dict(roptions, network_component="network", initial_state_component="initial",
                   nudge_component="initial", publish_channels=False)
    del options["fabric_manifest"]
    initial = StageMemory((ArrayMemory("routing_state", (6742, 3), "float32"),
                           ArrayMemory("nudge_m3_s", (6742,), "float32")), fabric.evidence)
    resource, _ = construct(packet)
    route = mc(settings=settings, options=options, context=context,
               components={"network": fabric, "initial": initial, "runoff": runoff, "reservoirs": resource})
    assert sum(a.nbytes for a in route.arrays if a.alias_of is None) == 44 * 6742
    assert next(a for a in route.arrays if a.name == "parameters").alias_of == "network/routing_parameters"
    assert next(a for a in route.arrays if a.name == "state").alias_of == "initial/routing_state"
    assert next(a for a in route.arrays if a.name == "nudge_m3_s").alias_of == "initial/nudge_m3_s"
    HydroMemoryPlan(context, (("network", fabric), ("initial", initial), ("runoff", runoff),
                             ("reservoirs", resource), ("routing", route)), ("unpriced",), "sha").inventory()


def test_missing_fabric_hook_still_prices_own_allocations_and_names_unknown_aliases(packet):
    path, _, settings, context, _, roptions, runoff = packet
    settings["stages"].insert(0, {"name": "network", "factory": "gpuwm.hydrofabric:prepare_resource",
                                    "options": {"manifest": str(path)}})
    options = dict(roptions, network_component="network")
    del options["fabric_manifest"]
    value = mc(settings=settings, options=options, context=context,
               components={"network": None, "runoff": runoff, "reservoirs": None})
    assert sum(a.nbytes for a in value.arrays if a.alias_of is None) == 60 * 6742 + 20 * 6740
    assert any("provider network lacks" in g for g in value.gaps)
    assert any("embedded reservoir" in g for g in value.gaps)
    assert value.non_pool_bytes is None


@pytest.mark.parametrize('defect', ['length', 'path', 'shape', 'source'])
def test_bad_metadata_cannot_be_accepted_as_valid_inventory(packet, defect):
    path, manifest, settings, context, options, _, _ = packet
    if defect == "length":
        manifest["files"]["r_height"]["bytes"] += 4
    elif defect == "path":
        manifest["files"]["r_height"]["path"] = "../escape.bin"
    elif defect == "shape":
        manifest["files"]["r_parameters"]["shape"] = [18]
    elif defect == "source":
        manifest["reservoirs"]["source_pin"] = "unknown"
    else:
        manifest["reservoirs"]["parameter_columns"] = list(reversed(PARAMETERS))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        reservoirs(settings=settings, options=options, context=context, components={})


def test_cadence_nudging_and_unbound_custom_provider_remain_explicit(packet):
    _, _, settings, context, _, roptions, runoff = packet
    for update in ({"dt_seconds": 301}, {"streamflow_nudging": True}, {"streamflow_nudging": "false"}):
        with pytest.raises(ValueError):
            mc(settings=settings, options=dict(roptions, **update), context=context, components={"runoff": runoff})
    options = {"network_component": "custom", "lateral_component": "runoff"}
    row = mc(settings=settings, options=options, context=context, components={"custom": None})
    assert row.arrays == () and "manifest metadata unavailable" in row.gaps[0]


def test_equal_hashes_do_not_identify_independently_uploaded_selection_backings(packet):
    path, manifest, settings, context, _, roptions, runoff = packet
    manifest["files"]["other_selected"] = dict(manifest["files"]["selected"], path="other.bin")
    manifest["routing"]["reservoir_levels"][1] = {"level": 0, "rows": "other_selected"}
    path.write_text(json.dumps(manifest))
    resource = reservoirs(settings=settings, options=settings["stages"][0]["options"], context=context, components={})
    with pytest.raises(ValueError, match="conflicting reservoir checkpoint name"):
        mc(settings=settings, options=roptions, context=context,
           components={"runoff": runoff, "reservoirs": resource})


def test_declared_unavailable_initial_water_state_is_never_priced_as_cold_start(packet):
    path, manifest, settings, context, _, roptions, runoff = packet
    manifest["routing"]["initial_state_array"] = "absent_water"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="initial_state_array"):
        mc(settings=settings, options=roptions, context=context, components={"runoff": runoff})


def test_native_history_consumes_mc_output_view_contract(packet):
    from gpuwm.hydro.output import prepare_history_memory
    _, _, _, context, _, _, runoff = packet
    resource, route = construct(packet)
    capture = prepare_history_memory(settings={}, options=dict(source="routing", capacity_steps=24),
                                     context=context, components={"routing": route})
    row = HydroMemoryPlan(context, (("runoff", runoff), ("reservoirs", resource),
        ("routing", route), ("history", capture)), ("resources unpriced",), "metadata").inventory()
    assert not capture.gaps
    assert next(a for a in capture.arrays if a.name == "reach_id").alias_of == "routing/fabric/node_ids"
    assert next(a for a in capture.arrays if a.name == "inputs/velocity_m_s").offset_bytes == 6742*4
    assert "history/reach_id" not in row["backings"]
