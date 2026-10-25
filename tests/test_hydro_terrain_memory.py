"""Pre-factory terrain memory contracts; metadata only, no physics stand-ins."""
from dataclasses import asdict, replace
import builtins
import json
import sys
from types import ModuleType

import pytest

from gpuwm.hydro.memory import ArrayMemory, HydroMemoryPlan, MemoryContext, StageMemory, admit_plan, plan
from gpuwm.hydro.terrain_memory import terrain_memory
from gpuwm.nextgen.terrain import prepare_lateral_memory
from gpuwm.nextgen.inundation import prepare_memory as inundation_memory


def prepared():
    context = MemoryContext("standalone", "strict", 300, duration_seconds=86400)
    shape = (737, 582)
    fields = ("surface_depth", "groundwater_storage", "elevation", "soil_depth", "specific_yield",
              "manning", "conductivity", "active", "surface_source", "recharge", "outlet_stage", "outlet_width")
    arrays = tuple(ArrayMemory(n, shape, "int32" if n == "active" else "float32", checkpointed=True)
                   for n in fields)
    evidence = json.dumps({"context": asdict(context), "grid_sha256": "a" * 64})
    providers = {"fields": StageMemory(arrays, evidence),
        "map": StageMemory((ArrayMemory("offsets", (6743,), "int64", checkpointed=True),
                            ArrayMemory("indices", (428934,), "int32", checkpointed=True)), evidence)}
    options = dict(dx=1000, dt=5, period_seconds=300, identity={"grid_sha256": "a" * 64},
                   fields={n: {"component": "fields", "array": n} for n in fields})
    return context, providers, options


def test_whole_prepared_metadata_plan_keeps_scratch_aliases_members_and_gaps(monkeypatch):
    context, providers, options = prepared()
    module = ModuleType("gpuwm.hydro._terrain_memory_test_provider")
    module.prepare_memory = lambda **kw: providers[kw["options"]["key"]]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # Importing CuPy or a numerical host-array library here is an error: no
    # allocated model or numerical payload is needed for admission metadata.
    original = builtins.__import__
    def metadata_import(name, *args, **kwargs):
        if name.split(".")[0] in {"cupy", "numpy"}:
            pytest.fail("numerical library imported by terrain memory discovery")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", metadata_import)
    settings = {"stages": [
        {"name": name, "factory": module.__name__ + ":prepare", "options": {"key": name}}
        for name in providers] + [
        {"name": "terrain", "factory": "gpuwm.nextgen.terrain:prepare", "options": options},
        {"name": "lateral", "factory": "gpuwm.nextgen.terrain:prepare_lateral", "options": {
            "terrain_component": "terrain", "mapping_component": "map", "offsets_array": "offsets",
            "indices_array": "indices", "identity": {"mapping_sha256": "b" * 64}}},
        {"name": "depth", "factory": "gpuwm.nextgen.inundation:prepare", "options": {"terrain_component": "terrain"}},
    ]}
    value = plan(settings, context=context)
    row = value.inventory()
    typed = {a["name"]: a for a in row["arrays"]}
    flux = typed["terrain/workspace/flux"]
    assert flux["shape"] == (14, 737, 582) and flux["logical_bytes"] == 24020304
    assert row["backings"][flux["name"]]["allocated_bytes"] == 24020480
    assert flux["lifetime"] == "resident" and not flux["checkpointed"]
    assert typed["depth/inputs/surface_depth"]["alias_of"] == "fields/surface_depth"
    assert typed["lateral/inputs/surface_interval_volume"]["alias_of"] == "terrain/surface_interval_volume"
    assert typed["terrain/depth_m"]["alias_of"] == "fields/surface_depth"
    assert sum(a["checkpointed"] for a in typed.values()) == 12 + 2 + 18 + 4 + 3
    assert {n for n, a in typed.items() if a["status"]} == {"terrain/health", "lateral/health", "depth/health"}
    assert len(row["backings"]) == 12 + 2 + 7 + 2 + 3 + len([
        a for a in row["arrays"] if a["lifetime"] != "resident"])
    members = value.inventory(members=12)
    assert members["resident_bytes"] == 12 * row["resident_bytes"]
    assert members["transient_bytes"] == 12 * row["transient_bytes"]
    assert not row["complete"] and all("kernel" in g or "device bound" in g for g in row["gaps"])
    receipt = admit_plan(value)
    assert receipt["admission"]["fits"] is None and not receipt["pre_factory_admission"]
    for name in ("terrain", "lateral", "depth"):
        pins = json.loads(row["evidence"][name])
        assert pins["context"] == json.loads(json.dumps(asdict(context)))
        assert len(pins["kernel_source_sha256"]) == len(pins["factory_source_sha256"]) == 64


def test_bound_outputs_and_root_offsets_are_preserved_without_duplicate_backings():
    context, providers, options = prepared()
    root = ArrayMemory("slab", (2, 737, 582), "float64")
    view = ArrayMemory("out", (737, 582), "float64", alias_of="map/slab", offset_bytes=3431472)
    providers["map"] = replace(providers["map"], arrays=(*providers["map"].arrays, root, view))
    options["fields"]["surface_interval_volume"] = {"component": "map", "array": "out"}
    value = terrain_memory(settings={}, options=options, context=context, components=providers)
    providers["terrain"] = value
    lateral = prepare_lateral_memory(settings={}, options=dict(terrain_component="terrain", mapping_component="map",
        offsets_array="offsets", indices_array="indices", identity={"mapping_sha256": "b" * 64}),
        context=context, components=providers)
    target = next(a for a in lateral.arrays if a.name == "inputs/surface_interval_volume")
    assert target.alias_of == "map/slab" and target.offset_bytes == 3431472
    row = HydroMemoryPlan(context, (*providers.items(), ("lateral", lateral)), (), "test").inventory()
    assert "terrain/surface_interval_volume" not in row["backings"]


@pytest.mark.parametrize("defect", ["shape", "dtype", "mutable_alias", "numerics", "escaped_view"])
def test_invalid_provider_metadata_cannot_be_priced_as_valid(defect):
    context, providers, options = prepared()
    source = providers["fields"]
    arrays = list(source.arrays)
    if defect == "shape":
        arrays[0] = replace(arrays[0], shape=(738, 582))
    elif defect == "dtype":
        arrays[0] = replace(arrays[0], dtype="float64")
    elif defect == "mutable_alias":
        arrays[1] = replace(arrays[1], alias_of="fields/surface_depth")
    elif defect == "escaped_view":
        arrays[1] = replace(arrays[1], alias_of="fields/surface_depth", offset_bytes=4)
    providers["fields"] = replace(source, arrays=tuple(arrays), evidence=(
        json.dumps({"context": {"numerics": "fast"}}) if defect == "numerics" else source.evidence))
    with pytest.raises(ValueError):
        terrain_memory(settings={}, options=options, context=context, components=providers)


def test_unknown_provider_prices_only_known_storage_with_named_gaps():
    context, providers, options = prepared()
    providers["fields"] = None
    value = terrain_memory(settings={}, options=options, context=context, components=providers)
    assert [(a.name, a.nbytes) for a in value.arrays] == [("health", 4)]
    assert any("flux and unbound outputs are unpriced" in gap for gap in value.gaps)
    assert value.non_pool_bytes is None
    depth = inundation_memory(settings={}, options={"terrain_component": "missing"}, context=context, components={})
    assert depth.gaps and [a.name for a in depth.arrays] == ["health"]


def test_continental_array_bindings_decode_each_provider_evidence_once(monkeypatch):
    from gpuwm.hydro import terrain_memory as module
    from gpuwm.hydro.memory import ArrayMemory, StageMemory, MemoryContext
    source = StageMemory((ArrayMemory('rows', (17,), 'int32'),),
                         json.dumps({'context': {'numerics': 'fast'}, 'ordered_ids': list(range(8193))}))
    context = MemoryContext('coupled', 'fast', 60)
    bindings = module._Bindings({'fabric': source}, context)
    loads = module.json.loads
    calls = []
    def counted(value):
        calls.append(value)
        return loads(value)
    monkeypatch.setattr(module.json, 'loads', counted)
    for i in range(1000):
        array = bindings.get('fabric', 'rows', f'selection/{i}')
        assert array.alias_of == 'fabric/rows' and array.shape == (17,)
    assert calls == [source.evidence]
    assert bindings.sources == {'fabric': source.evidence}
    # A new binding scope must still reject a different numerics context.
    strict = module._Bindings({'fabric': source}, MemoryContext('coupled', 'strict', 60))
    with pytest.raises(ValueError, match='numerics differs'):
        strict.get('fabric', 'rows', 'selection')


def test_borrower_evidence_seals_prior_roster_without_recursive_embedding():
    import hashlib
    from gpuwm.hydro.terrain_memory import _Bindings
    from gpuwm.hydro.memory import ArrayMemory, StageMemory, MemoryContext, HydroMemoryPlan
    evidence = json.dumps({'context': {'numerics': 'fast'},
                           'ordered_ids': ['cat-"\\\u00e9'] * 8193}, ensure_ascii=False)
    source = StageMemory((ArrayMemory('rows', (17,), 'int32'),), evidence)
    context = MemoryContext('coupled', 'fast', 60)
    bindings = _Bindings({'fabric': source}, context)
    view = bindings.get('fabric', 'rows', 'rows')
    references = bindings.evidence_references()
    assert references == {'fabric': {'stage': 'fabric', 'bytes': len(evidence.encode()),
                                    'sha256': hashlib.sha256(evidence.encode()).hexdigest()}}
    borrower = StageMemory((view,), json.dumps({'providers': references}))
    row = HydroMemoryPlan(context, (('fabric', source), ('borrower', borrower)), (), 'sha').inventory()
    assert row['evidence']['fabric'] == evidence
    assert len(row['evidence']['borrower']) < 256
    assert row['resident_bytes'] == 512  # the borrowed view adds no allocation
    # A direct-manifest provider has no earlier stage; retain its full proof.
    bindings.sources['local_fabric'] = evidence
    assert bindings.evidence_references()['local_fabric']['inline'] == evidence
    # A changed byte must change the reference, even with unchanged extents.
    bindings.sources['fabric'] += ' '
    assert bindings.evidence_references()['fabric']['sha256'] != references['fabric']['sha256']
