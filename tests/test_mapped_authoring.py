from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from gpuwm.mapped_authoring import (
    _stable_file_snapshot,
    DESCRIPTOR_SCHEMA,
    VtableRow,
    _selector_from_row,
    author_input_manifest,
    author_mapping,
    compile_mapping_descriptor,
    parse_wps_vtable,
)
from gpuwm.mapped_composition import _verify_manifest
from gpuwm.mapped_source import load_mapping


ROOT = Path(__file__).parents[1]
GFS_MAPPING = ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json"
GFS_DESCRIPTOR = ROOT / "configs" / "rw-wps-gfs-pressure-grib2.descriptor.json"
GFS_VTABLE = ROOT / "configs" / "Vtable.GFS.rw-wps"
GFS_COMPOSITION = ROOT / "configs" / "rw-wps-gfs-terrain.composition.json"
ERA5_NETCDF_MAPPING = ROOT / "configs" / "rw-wps-era5-netcdf.mapping.json"
HRRR_VTABLE = ROOT / "tests" / "fixtures" / "Vtable.raphrrr.ambiguous"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stable_snapshot_accepts_executable_authority(tmp_path):
    decoder = tmp_path / "grib2_inventory.exe"
    decoder.write_bytes(b"decoder-authority")
    snapshot = _stable_file_snapshot(decoder)
    assert snapshot.data == b"decoder-authority"
    assert snapshot.sha256 == _sha(decoder)


def _fake_bridge_identity(path: Path, _role: str) -> dict[str, object]:
    payload = Path(path).read_bytes()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _descriptor_and_vtable(mapping_path: Path) -> tuple[dict, str]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    descriptor = copy.deepcopy(mapping)
    descriptor["schema"] = DESCRIPTOR_SCHEMA
    rows = []
    for field_name, field in descriptor["fields"].items():
        selectors = field.get("selectors", [])
        if not selectors:
            continue
        field.pop("selectors")
        references = []
        for index, selector in enumerate(selectors):
            name = f"RW_{field_name}_{index}"
            level_type = selector.get("level_type", 255)
            rows.append(
                " |  |  *  |  | "
                f"{name} | {field['units']['source']} | {field_name} | "
                f"{selector['discipline']} | {selector['category']} | "
                f"{selector['parameter']} | {level_type} |"
            )
            overrides = {
                key: selector[key]
                for key in (
                    "center",
                    "subcenter",
                    "master_table_version",
                    "local_table_version",
                    "level_value",
                    "second_level_type",
                    "second_level_value",
                    "member",
                )
                if key in selector
            }
            reference = {
                "metgrid_name": name,
                "grib2_level_type": level_type,
            }
            if overrides:
                reference["selector"] = overrides
            references.append(reference)
        field["vtable_selectors"] = references
    return descriptor, "\n".join(rows) + "\n"


def test_grib_descriptor_import_reproduces_real_gfs_mapping(tmp_path):
    expected = json.loads(GFS_MAPPING.read_text(encoding="utf-8"))
    for selector in expected["fields"]["volumetric_soil_moisture"]["selectors"]:
        assert selector["parameter"] == 192
        selector["parameter"] = 191
    expected_path = tmp_path / "nonlocal.mapping.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    descriptor, vtable = _descriptor_and_vtable(expected_path)
    descriptor_path = tmp_path / "descriptor.json"
    vtable_path = tmp_path / "Vtable.GFS"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    vtable_path.write_text(vtable, encoding="utf-8")

    output = tmp_path / "mapping.json"
    receipt = author_mapping(descriptor_path, output, vtable_path=vtable_path)

    assert load_mapping(output) == load_mapping(expected_path)
    assert receipt["status"] == "VALIDATED_NOT_STOCK_WRF_CERTIFIED"
    assert receipt["mapping"]["sha256"] == _sha(output)
    assert len(receipt["selected_rows"]) == 21
    assert (tmp_path / "mapping.authoring.json").is_file()


def test_checked_in_gfs_authorities_reproduce_canonical_mapping():
    compiled, evidence = compile_mapping_descriptor(
        GFS_DESCRIPTOR,
        vtable_path=GFS_VTABLE,
    )

    assert compiled == load_mapping(GFS_MAPPING)
    assert len(evidence["selected_rows"]) == 21


def test_grib2_descriptor_rejects_unbound_local_use_identifiers(tmp_path):
    descriptor, vtable = _descriptor_and_vtable(GFS_MAPPING)
    for reference in descriptor["fields"]["volumetric_soil_moisture"][
        "vtable_selectors"
    ]:
        for key in (
            "center", "subcenter", "master_table_version", "local_table_version"
        ):
            reference["selector"].pop(key)
    descriptor_path = tmp_path / "descriptor.json"
    vtable_path = tmp_path / "Vtable.GFS"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    vtable_path.write_text(vtable, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="local-use identifier.*parameter=192.*explicitly bind",
    ):
        compile_mapping_descriptor(descriptor_path, vtable_path=vtable_path)


def test_grib2_descriptor_accepts_bound_local_use_authority(tmp_path):
    descriptor, vtable = _descriptor_and_vtable(GFS_MAPPING)
    descriptor_path = tmp_path / "descriptor.json"
    vtable_path = tmp_path / "Vtable.GFS"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    vtable_path.write_text(vtable, encoding="utf-8")

    mapping, _receipt = compile_mapping_descriptor(
        descriptor_path, vtable_path=vtable_path
    )

    selector = mapping["fields"]["volumetric_soil_moisture"]["selectors"][0]
    assert {
        key: selector[key]
        for key in (
            "center", "subcenter", "master_table_version", "local_table_version"
        )
    } == {
        "center": 7,
        "subcenter": 0,
        "master_table_version": 2,
        "local_table_version": 1,
    }


def test_grib1_vtable_selector_requires_center_and_table_authority():
    row = VtableRow(
        1,
        "130",
        "100",
        "*",
        "",
        "TT",
        "K",
        "temperature",
        "0",
        "0",
        "0",
        "100",
    )
    with pytest.raises(ValueError, match="center and table_version"):
        _selector_from_row(row, "grib1", {"selector": {}}, "field.tt")

    selector = _selector_from_row(
        row,
        "grib1",
        {"selector": {"center": 98, "table_version": 128}},
        "field.tt",
    )
    assert selector["center"] == 98
    assert selector["table_version"] == 128

    numeric_row = VtableRow(
        2,
        "130",
        "103",
        "2",
        "",
        "TT",
        "K",
        "2 m temperature",
        "0",
        "0",
        "0",
        "103",
    )
    with pytest.raises(ValueError, match="differs from numeric Vtable Level1"):
        _selector_from_row(
            numeric_row,
            "grib1",
            {
                "selector": {
                    "center": 98,
                    "table_version": 128,
                    "level_value": 10,
                }
            },
            "field.t2",
        )


def test_descriptor_rejects_missing_grib_identifiers_and_extreme_numbers():
    row = VtableRow(
        1,
        "",
        "",
        "*",
        "",
        "BAD",
        "1",
        "undefined parameter",
        "0",
        "0",
        "255",
        "1",
    )
    with pytest.raises(ValueError, match="missing/undefined identifier code 255"):
        _selector_from_row(row, "grib2", {}, "field.bad")

    row = VtableRow(
        1,
        "",
        "",
        "*",
        "",
        "TT",
        "K",
        "temperature",
        "0",
        "0",
        "0",
        "100",
    )
    with pytest.raises(ValueError, match="finite number"):
        _selector_from_row(
            row,
            "grib2",
            {"selector": {"level_value": 10**10_000}},
            "field.tt",
        )


def test_netcdf_descriptor_requires_explicit_mapping_semantics(tmp_path):
    descriptor = json.loads(ERA5_NETCDF_MAPPING.read_text(encoding="utf-8"))
    descriptor["schema"] = DESCRIPTOR_SCHEMA
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")

    compiled, evidence = compile_mapping_descriptor(path)

    assert compiled == load_mapping(ERA5_NETCDF_MAPPING)
    assert evidence["vtable"] is None
    assert compiled["fields"]["soil_temperature"]["selector_stack_axis"] == "soil"
    with pytest.raises(ValueError, match="cannot use a WPS Vtable"):
        compile_mapping_descriptor(path, vtable_path=HRRR_VTABLE)

    output = tmp_path / "wrong-format.json"
    with pytest.raises(ValueError, match="differs from expected format"):
        author_mapping(path, output, expected_format="grib2")
    assert not output.exists()
    assert not (tmp_path / "wrong-format.authoring.json").exists()


def test_descriptor_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema":"rw-wps.descriptor.v1","name":"first",'
        '"name":"silent-override"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key 'name'"):
        compile_mapping_descriptor(path)


def test_descriptor_compilation_rejects_authority_swap_after_validation(
    tmp_path,
    monkeypatch,
):
    descriptor = json.loads(ERA5_NETCDF_MAPPING.read_text(encoding="utf-8"))
    descriptor["schema"] = DESCRIPTOR_SCHEMA
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    original_loader = load_mapping

    def load_then_swap(candidate):
        result = original_loader(candidate)
        descriptor["name"] = "swapped-after-validation"
        path.write_text(json.dumps(descriptor), encoding="utf-8")
        return result

    monkeypatch.setattr("gpuwm.mapped_authoring.load_mapping", load_then_swap)
    with pytest.raises(ValueError, match="changed after validation"):
        compile_mapping_descriptor(path)


def test_real_vtable_ambiguity_is_rejected_instead_of_guessing_tt_semantics(
    tmp_path,
):
    rows = parse_wps_vtable(HRRR_VTABLE)
    tt = [row for row in rows if row.metgrid_name == "TT"]
    assert {(row.grib2_level_type, row.level1) for row in tt} == {
        ("105", "*"),
        ("103", "2"),
    }

    descriptor, vtable = _descriptor_and_vtable(GFS_MAPPING)
    first = next(
        field for field in descriptor["fields"].values() if "vtable_selectors" in field
    )
    first["vtable_selectors"][0] = {"metgrid_name": "TT"}
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="resolves 2 Vtable rows"):
        compile_mapping_descriptor(descriptor_path, vtable_path=HRRR_VTABLE)


def test_grib_descriptor_rejects_manual_selectors_and_unknown_semantics(tmp_path):
    descriptor, vtable = _descriptor_and_vtable(GFS_MAPPING)
    direct = next(
        field for field in descriptor["fields"].values() if "vtable_selectors" in field
    )
    direct["selectors"] = [{"format": "grib2"}]
    descriptor_path = tmp_path / "manual.json"
    vtable_path = tmp_path / "Vtable"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    vtable_path.write_text(vtable, encoding="utf-8")
    with pytest.raises(ValueError, match="not hand-authored selectors"):
        compile_mapping_descriptor(descriptor_path, vtable_path=vtable_path)

    descriptor, vtable = _descriptor_and_vtable(GFS_MAPPING)
    direct = next(
        field for field in descriptor["fields"].values() if "vtable_selectors" in field
    )
    direct["surprise"] = "not part of descriptor.v1"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        compile_mapping_descriptor(descriptor_path, vtable_path=vtable_path)


def test_grib2_numeric_vtable_level_needs_explicit_physical_value(tmp_path):
    descriptor, vtable = _descriptor_and_vtable(GFS_MAPPING)
    field = next(
        value for value in descriptor["fields"].values() if "vtable_selectors" in value
    )
    reference = field["vtable_selectors"][0]
    name = reference["metgrid_name"]
    vtable = vtable.replace(f"{name} |", f"{name} |", 1).replace(
        " |  |  *  |  | " + name,
        " |  |  2  |  | " + name,
        1,
    )
    reference.pop("selector", None)
    descriptor_path = tmp_path / "descriptor.json"
    vtable_path = tmp_path / "Vtable"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    vtable_path.write_text(vtable, encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly bind selector.level_value"):
        compile_mapping_descriptor(descriptor_path, vtable_path=vtable_path)


def test_descriptor_rejects_distinct_vtable_rows_with_same_direct_selector(
    tmp_path,
):
    descriptor, vtable = _descriptor_and_vtable(GFS_MAPPING)
    rows = vtable.splitlines()
    first = rows[0].split("|")
    second = rows[1].split("|")
    second[7:11] = first[7:11]
    rows[1] = "|".join(second)
    descriptor_path = tmp_path / "descriptor.json"
    vtable_path = tmp_path / "Vtable"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    vtable_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="selector overlaps"):
        compile_mapping_descriptor(descriptor_path, vtable_path=vtable_path)


def test_grib_descriptor_rejects_broad_and_narrow_selector_overlap(tmp_path):
    descriptor, vtable = _descriptor_and_vtable(GFS_MAPPING)
    direct_fields = [
        value
        for value in descriptor["fields"].values()
        if "vtable_selectors" in value
    ]
    first_reference = direct_fields[0]["vtable_selectors"][0]
    second_reference = direct_fields[1]["vtable_selectors"][0]
    first_reference.pop("selector", None)
    second_reference["selector"] = {"level_value": 500.0}
    rows = vtable.splitlines()
    first = rows[0].split("|")
    second = rows[1].split("|")
    second[7:11] = first[7:11]
    rows[1] = "|".join(second)
    descriptor_path = tmp_path / "descriptor.json"
    vtable_path = tmp_path / "Vtable"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    vtable_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="selector overlaps"):
        compile_mapping_descriptor(descriptor_path, vtable_path=vtable_path)


def test_runtime_mapping_validator_rejects_manual_selector_overlap(tmp_path):
    mapping = json.loads(GFS_MAPPING.read_text(encoding="utf-8"))
    direct_fields = [
        value for value in mapping["fields"].values() if value.get("selectors")
    ]
    broad = copy.deepcopy(direct_fields[0]["selectors"][0])
    assert "level_value" not in broad
    narrow = copy.deepcopy(broad)
    narrow["level_value"] = 500.0
    direct_fields[1]["selectors"][0] = narrow
    path = tmp_path / "overlap.mapping.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match=r"selectors\[0\] overlaps"):
        load_mapping(path)


def test_runtime_mapping_validator_rejects_undefined_second_surface(tmp_path):
    mapping = json.loads(GFS_MAPPING.read_text(encoding="utf-8"))
    selector = next(
        selector
        for field in mapping["fields"].values()
        for selector in field.get("selectors", [])
        if "second_level_type" in selector
    )
    selector["second_level_type"] = 255
    path = tmp_path / "undefined-surface.mapping.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="missing/undefined identifier code 255"):
        load_mapping(path)


def _manifest_case(tmp_path: Path):
    primary_a = tmp_path / "primary-a.grib2"
    primary_b = tmp_path / "primary-b.grib2"
    provenance = tmp_path / "terrain.md"
    inventory = tmp_path / "grib2-inventory"
    dump = tmp_path / "grib2-dump"
    for path, value in (
        (primary_a, b"a"),
        (primary_b, b"bb"),
        (inventory, b"inventory"),
        (dump, b"dump"),
    ):
        path.write_bytes(value)
    provenance.write_text("terrain provenance\n", encoding="utf-8")
    return primary_a, primary_b, provenance, inventory, dump


def test_manifest_authoring_binds_exact_order_roles_decoders_and_round_trips(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "gpuwm.mapped_authoring.bridge_identity",
        _fake_bridge_identity,
    )
    primary_a, primary_b, provenance, inventory, dump = _manifest_case(tmp_path)
    manifest = tmp_path / "inputs.json"
    receipt = author_input_manifest(
        manifest,
        mapping_path=GFS_MAPPING,
        composition_path=GFS_COMPOSITION,
        primary_files=(primary_a, primary_b),
        supplement_files={
            "gfs_valid_time_terrain": (primary_a, primary_b),
        },
        provenance_files={
            "gfs_valid_time_terrain_provenance": provenance,
        },
        grib2_inventory=inventory,
        grib2_dump=dump,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert receipt["status"] == "PASS_IDENTITY_BOUND_NOT_STOCK_WRF_CERTIFIED"
    assert receipt["manifest"]["sha256"] == _sha(manifest)
    assert [row["bytes"] for row in payload["primary_files"]] == [1, 2]
    assert isinstance(payload["supplements"]["gfs_valid_time_terrain"], list)
    _verify_manifest(
        manifest,
        _sha(manifest),
        mapping_path=GFS_MAPPING,
        composition_path=GFS_COMPOSITION,
        primary_files=(primary_a, primary_b),
        supplement_files={
            "gfs_valid_time_terrain": (primary_a, primary_b),
        },
        provenance_files={
            "gfs_valid_time_terrain_provenance": provenance,
        },
        decoder_files={
            "grib2_inventory": inventory,
            "grib2_dump": dump,
        },
    )

    dump.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="byte count differs|SHA differs"):
        _verify_manifest(
            manifest,
            _sha(manifest),
            mapping_path=GFS_MAPPING,
            composition_path=GFS_COMPOSITION,
            primary_files=(primary_a, primary_b),
            supplement_files={
                "gfs_valid_time_terrain": (primary_a, primary_b),
            },
            provenance_files={
                "gfs_valid_time_terrain_provenance": provenance,
            },
            decoder_files={
                "grib2_inventory": inventory,
                "grib2_dump": dump,
            },
        )


def test_authoring_refuses_overwrite_and_wrong_role_without_damage(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "gpuwm.mapped_authoring.bridge_identity",
        _fake_bridge_identity,
    )
    primary_a, primary_b, provenance, inventory, dump = _manifest_case(tmp_path)
    manifest = tmp_path / "inputs.json"
    manifest.write_text("old-valid-manifest\n", encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        author_input_manifest(
            manifest,
            mapping_path=GFS_MAPPING,
            composition_path=GFS_COMPOSITION,
            primary_files=(primary_a, primary_b),
            supplement_files={
                "gfs_valid_time_terrain": (primary_a, primary_b),
            },
            provenance_files={
                "gfs_valid_time_terrain_provenance": provenance,
            },
            grib2_inventory=inventory,
            grib2_dump=dump,
        )
    assert manifest.read_bytes() == before

    with pytest.raises(ValueError, match="supplement role inventory"):
        author_input_manifest(
            tmp_path / "new.json",
            mapping_path=GFS_MAPPING,
            composition_path=GFS_COMPOSITION,
            primary_files=(primary_a, primary_b),
            supplement_files={"wrong": primary_a},
            provenance_files={
                "gfs_valid_time_terrain_provenance": provenance,
            },
            grib2_inventory=inventory,
            grib2_dump=dump,
        )
    assert not (tmp_path / "new.json").exists()


def test_manifest_authoring_rejects_hardlink_alias_inside_ordered_series(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "gpuwm.mapped_authoring.bridge_identity",
        _fake_bridge_identity,
    )
    primary_a, _primary_b, provenance, inventory, dump = _manifest_case(tmp_path)
    alias = tmp_path / "alias.grib2"
    os.link(primary_a, alias)

    with pytest.raises(ValueError, match="filesystem aliases"):
        author_input_manifest(
            tmp_path / "new.json",
            mapping_path=GFS_MAPPING,
            composition_path=GFS_COMPOSITION,
            primary_files=(primary_a, alias),
            supplement_files={
                "gfs_valid_time_terrain": (primary_a, alias),
            },
            provenance_files={
                "gfs_valid_time_terrain_provenance": provenance,
            },
            grib2_inventory=inventory,
            grib2_dump=dump,
        )


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows executable mode bits are path-suffix synthesis, not identity",
)
def test_manifest_authoring_rejects_decoder_mode_drift_after_verification(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "gpuwm.mapped_authoring.bridge_identity",
        _fake_bridge_identity,
    )
    primary_a, primary_b, provenance, inventory, dump = _manifest_case(tmp_path)

    def verify_then_change_mode(*args, **kwargs):
        result = _verify_manifest(*args, **kwargs)
        inventory.chmod(0o444)
        return result

    monkeypatch.setattr(
        "gpuwm.mapped_authoring._verify_manifest",
        verify_then_change_mode,
    )
    output = tmp_path / "inputs.json"
    try:
        with pytest.raises(ValueError, match="changed after validation"):
            author_input_manifest(
                output,
                mapping_path=GFS_MAPPING,
                composition_path=GFS_COMPOSITION,
                primary_files=(primary_a, primary_b),
                supplement_files={
                    "gfs_valid_time_terrain": (primary_a, primary_b),
                },
                provenance_files={
                    "gfs_valid_time_terrain_provenance": provenance,
                },
                grib2_inventory=inventory,
                grib2_dump=dump,
            )
    finally:
        inventory.chmod(0o666)
    assert not output.exists()


def test_manifest_authoring_rejects_mapping_swap_after_snapshot_validation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "gpuwm.mapped_authoring.bridge_identity",
        _fake_bridge_identity,
    )
    primary_a, primary_b, provenance, inventory, dump = _manifest_case(tmp_path)
    mapping = tmp_path / "mapping.json"
    mapping.write_bytes(GFS_MAPPING.read_bytes())

    def inventory_then_swap(
        source_format,
        *,
        grib1_bridge,
        grib2_inventory,
        grib2_dump,
    ):
        assert source_format == "grib2"
        assert grib1_bridge is None
        mapping.write_bytes(ERA5_NETCDF_MAPPING.read_bytes())
        return {
            "grib2_inventory": Path(grib2_inventory).resolve(),
            "grib2_dump": Path(grib2_dump).resolve(),
        }

    monkeypatch.setattr(
        "gpuwm.mapped_authoring._decoder_inventory",
        inventory_then_swap,
    )
    output = tmp_path / "inputs.json"
    with pytest.raises(ValueError, match="changed after validation"):
        author_input_manifest(
            output,
            mapping_path=mapping,
            composition_path=GFS_COMPOSITION,
            primary_files=(primary_a, primary_b),
            supplement_files={
                "gfs_valid_time_terrain": (primary_a, primary_b),
            },
            provenance_files={
                "gfs_valid_time_terrain_provenance": provenance,
            },
            grib2_inventory=inventory,
            grib2_dump=dump,
        )
    assert not output.exists()


def test_manifest_expected_format_mismatch_fails_before_output(tmp_path):
    output = tmp_path / "inputs.json"
    with pytest.raises(ValueError, match="differs from expected format"):
        author_input_manifest(
            output,
            mapping_path=GFS_MAPPING,
            composition_path=GFS_COMPOSITION,
            primary_files=(GFS_MAPPING,),
            supplement_files={"gfs_valid_time_terrain": GFS_COMPOSITION},
            provenance_files={
                "gfs_valid_time_terrain_provenance": HRRR_VTABLE,
            },
            expected_format="netcdf",
        )
    assert not output.exists()


def test_manifest_authoring_rejects_duplicate_mapping_json_keys(tmp_path):
    mapping = tmp_path / "duplicate.mapping.json"
    original = GFS_MAPPING.read_text(encoding="utf-8").lstrip()
    assert original.startswith("{")
    mapping.write_text(
        '{"schema":"rw-wps.mapping.v1",' + original[1:],
        encoding="utf-8",
    )
    output = tmp_path / "inputs.json"
    with pytest.raises(ValueError, match="duplicate JSON object key 'schema'"):
        author_input_manifest(
            output,
            mapping_path=mapping,
            composition_path=GFS_COMPOSITION,
            primary_files=(GFS_MAPPING,),
            supplement_files={"gfs_valid_time_terrain": GFS_MAPPING},
            provenance_files={
                "gfs_valid_time_terrain_provenance": GFS_COMPOSITION,
            },
        )
    assert not output.exists()
