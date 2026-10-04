"""Native array interpretation and explicit legacy qualification negative controls."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from gpuwm.ensemble.physical_store import (
    NativePhysicalStore, LEGACY_SCHEMA, digest_file, qualify_legacy_store,
    validate_field_contract, validate_physical_input_binding, BINDING_SCHEMA,
)
from physical_field_fixtures import analytic_field_contract


def _authority(tmp_path):
    grid = {"mass_shape": [2, 3], "fixture": "native unit authority"}
    contract = analytic_field_contract(grid)
    evidence = tmp_path / "analytical-source.txt"
    evidence.write_text("Explicit native analytical field units and coordinates\n")
    contract["evidence"] = {"analytical_fixture_definition": digest_file(evidence)}
    return grid, contract, {"analytical_fixture_definition": evidence}


@pytest.fixture
def native():
    from gpuwm.io.nc_writer_bridge import unavailable_reason
    from gpuwm.netcdf_bridge import find_netcdf_bin
    reason = unavailable_reason()
    if reason or find_netcdf_bin() is None:
        pytest.skip("native physical writer/reader required: " + str(reason))


def _write(tmp_path):
    from test_ensemble_physical_preparation import snapshot
    from gpuwm.ensemble.physical_store import physical_static_identity
    grid, contract, evidence = _authority(tmp_path)
    source = {"input_manifest_sha256": "a"*64,
              "static_identity": physical_static_identity({"HGT_M": np.zeros((2, 3))})}
    store = NativePhysicalStore(tmp_path / "store", grid_identity=grid, source_identity=source,
                                field_contract=contract)
    store.write(snapshot())
    store.seal()
    return store, evidence


def _legacy(tmp_path, *, newline=b"\n"):
    """Build genuine v1 native bytes; this does not relabel a v2 file."""
    from test_ensemble_physical_preparation import snapshot
    from gpuwm.ensemble.physical_store import physical_static_identity, canonical_grid_sha256
    from gpuwm.io.nc_writer_bridge import ClassicSchema
    frame = snapshot()
    grid, contract, evidence = _authority(tmp_path)
    root = tmp_path / "legacy"
    root.mkdir()
    arrays = {"levels_hpa": frame.levels_hpa, "meta__soil_no_source_land": frame.soil_no_source_land,
              **{"field__"+key: value for key, value in frame.fields.items()}}
    schema = ClassicSchema("cdf5")
    schema.put_global_attr("physical_store_schema", LEGACY_SCHEMA)
    schema.put_global_attr("valid_time", frame.valid_time.isoformat())
    schema.put_global_attr("grid_sha256", canonical_grid_sha256(grid))
    ids, inventory = {}, {}
    for name, array in arrays.items():
        dims = [schema.def_dim(f"dim{len(ids)}_{index}", length) for index, length in enumerate(array.shape)]
        ids[name] = schema.def_var(name, "u1" if array.dtype.kind == "b" else array.dtype, dims)
        inventory[name] = {"shape": list(array.shape), "dtype": array.dtype.str}
    path = root / "frame-0000.nc"
    with schema.create(path) as writer:
        for name, varid in ids.items():
            array = arrays[name]
            writer.write_var(varid, array.astype("u1") if array.dtype.kind == "b" else array)
    document = {"schema": LEGACY_SCHEMA, "grid": grid,
                "source": {"input_manifest_sha256": "a"*64,
                           "static_identity": physical_static_identity({"HGT_M": np.zeros((2, 3))})},
                "frames": [{"file": path.name, "sha256": digest_file(path), "bytes": path.stat().st_size,
                            "valid_time": frame.valid_time.isoformat(), "arrays": inventory,
                            "metadata": {"specific_humidity_authority": True, "analyzed_species": [],
                                         "specific_humidity_undershoot_floor": 0., "water_temperature_receipt": None,
                                         "horizontal_operators": None, "masked_field_repairs": None}}]}
    manifest = root / "physical-store.json"
    manifest.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()+newline)
    return root, document, contract, evidence


def test_missing_contract_and_ambiguous_units_are_refused_before_creation(tmp_path):
    grid, contract, _ = _authority(tmp_path)
    with pytest.raises(ValueError, match="field contract"):
        NativePhysicalStore(tmp_path/"missing", grid_identity=grid, source_identity={"source": "fixture"})
    assert not (tmp_path/"missing").exists()
    for mutation in ("unit", "vector", "grid", "hybrid"):
        bad = deepcopy(contract)
        if mutation == "unit":
            bad["arrays"]["field__TT"]["units"] = ""
        elif mutation == "vector":
            bad["arrays"]["field__UU"]["dimensions"] = ["level", "y", "x"]
        elif mutation == "grid":
            bad["grid_sha256"] = "9"*64
        else:
            bad["vertical"]["kind"] = "hybrid_model_levels"
        with pytest.raises(ValueError):
            validate_field_contract(bad, grid)


def test_native_units_and_basis_roundtrip_are_not_array_name_inference(tmp_path, native):
    from gpuwm.netcdf_bridge import open_dataset
    store, _ = _write(tmp_path)
    with open_dataset(store.root / store.document["frames"][0]["file"]) as dataset:
        assert dataset.variables["field__TT"].attributes["units"] == "K"
        assert dataset.variables["field__UU"].attributes["physical_vector_basis"] == "grid_x"
        assert dataset.variables["levels_hpa"].attributes["units"] == "hPa"
    assert NativePhysicalStore(store.root).read(0).fields["TT"].tobytes() == store.read(0).fields["TT"].tobytes()
    document = deepcopy(store.document)
    document["field_contract"]["arrays"]["field__TT"]["units"] = "degC"
    store.manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="header"):
        NativePhysicalStore(store.root).read(0)
    with pytest.raises(ValueError, match="manifest changed"):
        store.read(0)


def test_exact_shape_cannot_disguise_wrong_staggering(tmp_path, native):
    from test_ensemble_physical_preparation import snapshot
    grid, contract, _ = _authority(tmp_path)
    contract["arrays"]["field__TT"]["dimensions"] = ["level", "y", "x_stag"]
    store = NativePhysicalStore(tmp_path/"bad", grid_identity=grid, source_identity={"source": "fixture"},
                                field_contract=contract)
    with pytest.raises(ValueError, match="coordinate dimensions"):
        store.write(snapshot())
    assert not tuple(store.root.glob("*.nc"))


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_legacy_qualification_preserves_bytes_and_rechecks_all_authorities(tmp_path, native, newline):
    root, document, contract, evidence = _legacy(tmp_path, newline=newline)
    manifest = root / "physical-store.json"
    old_manifest, old_frame = manifest.read_bytes(), (root/"frame-0000.nc").read_bytes()
    with pytest.raises(ValueError, match="qualify or recapture"):
        NativePhysicalStore(root)
    kwargs = dict(expected_manifest_sha256=digest_file(manifest), field_contract=contract,
                  source_identity=document["source"], evidence_files=evidence)
    with pytest.raises(ValueError, match="expected original manifest"):
        qualify_legacy_store(root, **{**kwargs, "expected_manifest_sha256": "b"*64})
    with pytest.raises(ValueError, match="exact verified capture"):
        qualify_legacy_store(root, **{**kwargs, "source_identity": {"unverified": True}})
    result = qualify_legacy_store(root, **kwargs)
    assert result["frames"] == 1
    assert manifest.read_bytes() == old_manifest and (root/"frame-0000.nc").read_bytes() == old_frame
    store = NativePhysicalStore(root)
    assert store.qualification["native_attributes_present"] is False
    assert store.read(0).fields["TT"].shape == (5, 2, 3)
    binding = {"schema": BINDING_SCHEMA, "manifest_sha256": digest_file(manifest),
               "manifest": document, "qualification": store.qualification}
    validate_physical_input_binding(binding, input_manifest_sha256="a"*64)
    bad = deepcopy(binding)
    bad["qualification"]["frames"][0]["sha256"] = "f"*64
    with pytest.raises(ValueError, match="native frame bytes"):
        validate_physical_input_binding(bad, input_manifest_sha256="a"*64)
    with pytest.raises(FileExistsError):
        qualify_legacy_store(root, **kwargs)
    with (root/"frame-0000.nc").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="differ from the sealed"):
        NativePhysicalStore(root).read(0)


def test_legacy_qualification_refuses_changed_source_evidence(tmp_path, native):
    root, document, contract, evidence = _legacy(tmp_path)
    next(iter(evidence.values())).write_text("changed evidence")
    with pytest.raises(ValueError, match="source evidence changed"):
        qualify_legacy_store(root, expected_manifest_sha256=digest_file(root/"physical-store.json"),
                             field_contract=contract, source_identity=document["source"], evidence_files=evidence)
    assert not (root/"physical-qualification.json").exists()


def test_mapped_contract_uses_validated_canonical_units_and_native_vector_join(tmp_path):
    from pathlib import Path
    import gpuwm
    from gpuwm.ensemble.mapped_physical_contract import mapped_physical_field_contract
    base = Path(gpuwm.__file__).resolve().parent / "authorities"
    mapping = base / "rw-wps-gefs-ensemble-grib2.mapping.json"
    composition = base / "rw-wps-gefs-ensemble-grib2.composition.json"
    source = {"adapter": "rw-wps-mapped-composition-v2", "mapping_sha256": digest_file(mapping),
              "composition_sha256": digest_file(composition)}
    grid, _, _ = _authority(tmp_path)
    contract = mapped_physical_field_contract(grid, mapping_path=mapping, composition_path=composition,
                                               source_identity=source)
    arrays = contract["arrays"]
    assert arrays["field__TT"]["units"] == "K"
    assert arrays["field__PSFC"]["units"] == "Pa"
    assert arrays["field__SPFH"]["units"] == "kg kg-1"
    assert arrays["field__GHT"]["units"] == "m" and "no second conversion" in arrays["field__GHT"]["operation"]
    assert arrays["field__UU"]["dimensions"] == ["level", "y", "x_stag"]
    assert arrays["field__V10"]["basis"] == "grid_y"
    assert arrays["field__RW_SOIL_MOISTURE"]["units"] == "m3 m-3"
    assert contract["vertical"]["pressure_field"] == "field__PRES"
    with pytest.raises(ValueError, match="captured source authority"):
        mapped_physical_field_contract(grid, mapping_path=mapping, composition_path=composition,
                                        source_identity={**source, "mapping_sha256": "f"*64})
