"""Acceptance gates for the arbitrary-but-verified GRIB front door."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gpuwm.source_authorities import packaged_gfs_vtable

from gpuwm import adapt
from gpuwm.adapt import author_adapter, build_composition, verify_grib2_inputs
from gpuwm.cli import main
from gpuwm.mapped_authoring import compile_mapping_descriptor


ROOT = Path(__file__).resolve().parents[1]
VTABLE = packaged_gfs_vtable()
DESCRIPTOR = ROOT / "configs" / "rw-wps-gfs-pressure-grib2.descriptor.json"
ACTUAL_GRIB = (
    ROOT
    / "tests"
    / "fixtures"
    / "gfs-scan-order"
    / "nomads-crop-20260729t18z-f000.grib2"
)


def _mapping():
    mapping, _receipt = compile_mapping_descriptor(
        DESCRIPTOR,
        vtable_path=VTABLE,
    )
    return mapping


def _row(index: int, selector: dict, *, level_value=None) -> dict[str, str]:
    values = {
        "index": str(index),
        "discipline": str(selector["discipline"]),
        "category": str(selector["category"]),
        "parameter": str(selector["parameter"]),
        "center": str(selector.get("center", 7)),
        "subcenter": str(selector.get("subcenter", 0)),
        "master_table_version": str(
            selector.get("master_table_version", 2)
        ),
        "local_table_version": str(selector.get("local_table_version", 1)),
        "reference_time": "2026-07-29T18:00:00",
        "forecast_unit": "1",
        "forecast_time": "0",
        "pdt": "0",
        "level_type": str(selector.get("level_type", 255)),
        "level_value": str(
            selector.get("level_value", 0)
            if level_value is None
            else level_value
        ),
        "second_level_type": str(selector.get("second_level_type", 255)),
        "second_level_value": str(selector.get("second_level_value", 0)),
        "member": str(selector.get("member", "-")),
        "generating_process": "2",
        "forecast_generating_process_id": "81",
        "gdt": "0",
        "nx": "41",
        "ny": "41",
        "lat1": "30",
        "lon1": "260",
        "dx": "0.25",
        "dy": "0.25",
        "latin1": "0",
        "latin2": "0",
        "lov": "0",
        "scan_mode": "0x40",
        "shape_of_earth": "6",
        "resolution_flags": "0x30",
        "drt": "0",
        "bitmap": "false",
    }
    return values


def _complete_gfs_inventory(mapping=None):
    mapping = mapping or _mapping()
    rows = []
    index = 0
    levels = mapping["coordinates"]["vertical"]["levels"]
    for field in mapping["fields"].values():
        if field.get("derivation") is not None:
            continue
        selectors = field["selectors"]
        if "vertical" in field["source_axes"]:
            for level in levels:
                rows.append(_row(index, selectors[0], level_value=level))
                index += 1
        else:
            for selector in selectors:
                rows.append(_row(index, selector))
                index += 1
    return rows


def _fake_bridge_identity(path, role):
    path = Path(path)
    data = path.read_bytes()
    return {
        "role": role,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_gfs_vtable_fixture_authors_runnable_uncertified_bundle(
    tmp_path,
    monkeypatch,
):
    """Full public authoring path: compile, battery, triple, and manifest."""

    rows = _complete_gfs_inventory()
    monkeypatch.setattr(adapt, "_grib2_inventory", lambda *_args: rows)
    monkeypatch.setattr(adapt, "_decode_grib", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "gpuwm.mapped_authoring.bridge_identity",
        _fake_bridge_identity,
    )
    inventory = tmp_path / "grib2_inventory"
    dump = tmp_path / "grib2_dump"
    inventory.write_bytes(b"fixture inventory decoder")
    dump.write_bytes(b"fixture dump decoder")

    result = author_adapter(
        vtable_path=VTABLE,
        descriptor_path=DESCRIPTOR,
        input_files=(ACTUAL_GRIB,),
        output_dir=tmp_path / "adapter",
        grib2_inventory=inventory,
        grib2_dump=dump,
    )

    assert result["status"] == \
        "runnable_mapping_not_stock_wrf_certified"
    assert result["runnable"] is True
    assert result["stock_wrf_certified"] is False
    assert result["battery"]["status"] == "PASS"
    assert result["battery"]["record_inventory"]["selected_record_count"] \
        == len(rows)
    assert result["runtime_bindings"]["source"] == "mapped"
    assert result["runtime_bindings"]["supplements"] == {
        "adapt_in_band_terrain": [str(ACTUAL_GRIB.resolve())],
    }
    assert result["runtime_bindings"]["provenance"] == {
        "adapt_authority_provenance": str(
            (tmp_path / "adapter" / "adapter.provenance.json").resolve()
        ),
    }
    outputs = {
        path.name for path in (tmp_path / "adapter").iterdir()
    }
    assert outputs == {
        "adapter.mapping.json",
        "adapter.composition.json",
        "adapter.provenance.json",
        "adapter.inputs.json",
    }
    provenance = json.loads(
        (tmp_path / "adapter" / "adapter.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["stock_wrf_certified"] is False
    assert provenance["mapping"]["sha256"] == result["mapping"]["sha256"]
    assert provenance["composition"]["sha256"] \
        == result["composition"]["sha256"]
    assert provenance["inputs"][0]["sha256"] == hashlib.sha256(
        ACTUAL_GRIB.read_bytes()
    ).hexdigest()


def test_adapt_refuses_one_exact_missing_field(tmp_path, monkeypatch):
    mapping = _mapping()
    rows = _complete_gfs_inventory(mapping)
    selector = mapping["fields"]["surface_pressure"]["selectors"][0]
    rows = [
        row for row in rows if not adapt._selector_matches(selector, row)
    ]
    monkeypatch.setattr(adapt, "_grib2_inventory", lambda *_args: rows)
    decoder = tmp_path / "inventory"
    decoder.write_bytes(b"fixture")

    with pytest.raises(
        ValueError,
        match=(
            r"record-inventory check failed.*field 'surface_pressure' "
            r"missing selector"
        ),
    ):
        verify_grib2_inputs(
            mapping,
            (ACTUAL_GRIB,),
            inventory_executable=decoder,
        )


def test_adapt_refuses_wrong_grid_template_with_named_adapter_remedy(
    tmp_path,
    monkeypatch,
):
    mapping = _mapping()
    rows = _complete_gfs_inventory(mapping)
    rows[0]["gdt"] = "30"
    monkeypatch.setattr(adapt, "_grib2_inventory", lambda *_args: rows)
    decoder = tmp_path / "inventory"
    decoder.write_bytes(b"fixture")

    with pytest.raises(
        ValueError,
        match=(
            r"grid-family check failed:.*uses GDT 30; generic adapt "
            r"supports regular latitude/longitude GDT 0 only.*"
            r"named-adapter path"
        ),
    ):
        verify_grib2_inputs(
            mapping,
            (ACTUAL_GRIB,),
            inventory_executable=decoder,
        )


def test_adapt_refuses_declared_soil_gap():
    mapping = _mapping()
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        mapping["fields"][name]["selectors"][1]["level_value"] = 0.2

    with pytest.raises(
        ValueError,
        match=(
            r"soil-layer check failed: gap between source layers 0 and 1: "
            r"0\.1 m to 0\.2 m"
        ),
    ):
        build_composition(
            mapping,
            {"soil_policy": {"kind": "identity_complete_layers"}},
        )


def test_adapt_refuses_source_top_that_does_not_cover_model_top(
    tmp_path,
):
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    descriptor["adapt"]["model_top_pa"] = 5000
    path = tmp_path / "top-failure.descriptor.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            r"vertical-coverage check failed: source top 10000 Pa does not "
            r"cover model top 5000 Pa"
        ),
    ):
        author_adapter(
            vtable_path=VTABLE,
            descriptor_path=path,
            input_files=(ACTUAL_GRIB,),
            output_dir=tmp_path / "adapter",
        )
    assert not (tmp_path / "adapter").exists()


def test_adapt_decoder_failure_publishes_no_authority(
    tmp_path,
    monkeypatch,
):
    rows = _complete_gfs_inventory()
    monkeypatch.setattr(adapt, "_grib2_inventory", lambda *_args: rows)

    def refuse_decode(*_args, **_kwargs):
        raise RuntimeError("selected GRIB2 record uses unsupported packing")

    monkeypatch.setattr(adapt, "_decode_grib", refuse_decode)
    inventory = tmp_path / "grib2_inventory"
    dump = tmp_path / "grib2_dump"
    inventory.write_bytes(b"fixture inventory decoder")
    dump.write_bytes(b"fixture dump decoder")
    output = tmp_path / "adapter"

    with pytest.raises(RuntimeError, match="unsupported packing"):
        author_adapter(
            vtable_path=VTABLE,
            descriptor_path=DESCRIPTOR,
            input_files=(ACTUAL_GRIB,),
            output_dir=output,
            grib2_inventory=inventory,
            grib2_dump=dump,
        )
    assert not output.exists()


def test_adapt_skeleton_generator_is_create_only_and_review_required(
    tmp_path,
    capsys,
):
    output = tmp_path / "source.descriptor.json"
    assert main(
        [
            "adapt",
            "--vtable",
            str(VTABLE),
            "--skeleton",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "rw-wps.descriptor.v1"
    assert payload["adapt"]["model_top_pa"] \
        == "REPLACE_WITH_MODEL_TOP_PA"
    assert payload["coordinates"]["vertical"]["levels"] == []
    assert "review every selector" in capsys.readouterr().out

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "adapt",
                "--vtable",
                str(VTABLE),
                "--skeleton",
                str(output),
            ]
        )


# ---------------------------------------------------------------------------
# V-9: the skeleton generator emitted a descriptor its own battery rejects
# ---------------------------------------------------------------------------


def test_the_skeleton_never_assigns_one_vtable_line_to_two_fields():
    """`gpuwm adapt: Vtable line 11 is assigned more than once`.

    A node-7 validation run filled in every REPLACE_WITH_* placeholder
    the skeleton emitted, declared the 21 isobaric levels actually in
    the GRIB, and had the battery refuse the result -- because the
    generator had given four 3-D fields their SURFACE counterpart's
    selector as well as their own.  `air_temperature` claimed the 2 m
    row that `air_temperature_2m` also claimed, and the same for
    relative humidity and both wind components.  The data is not
    ambiguous; the prefix rule that collects `soil_temperature_0_0.1m`
    under `soil_temperature` was swallowing `air_temperature_2m` under
    `air_temperature`.

    A generator whose output its own validator rejects is worse than no
    generator: the user cannot tell which of the two is wrong.
    """
    from collections import defaultdict

    from gpuwm.adapt import descriptor_skeleton

    vtable = packaged_gfs_vtable()
    skeleton = descriptor_skeleton(vtable)

    claimed = defaultdict(list)
    for field_name, spec in skeleton["fields"].items():
        for reference in spec.get("vtable_selectors", ()):
            claimed[reference.get("metgrid_name")].append(field_name)
    doubled = {
        name: sorted(fields)
        for name, fields in claimed.items() if len(fields) > 1
    }
    assert doubled == {}, doubled

    # And the separation is right, not merely non-overlapping: the 3-D
    # field takes the isobaric row and the 2 m/10 m field takes its own.
    for three_d, surface in (
        ("air_temperature", "air_temperature_2m"),
        ("relative_humidity", "relative_humidity_2m"),
        ("eastward_wind", "eastward_wind_10m"),
        ("northward_wind", "northward_wind_10m"),
    ):
        names = {
            field: {
                reference["metgrid_name"]
                for reference in skeleton["fields"][field]["vtable_selectors"]
            }
            for field in (three_d, surface)
        }
        assert names[three_d] and names[surface]
        assert not (names[three_d] & names[surface]), names
        assert all(
            not name.endswith(("_2m_0", "_10m_0"))
            for name in names[three_d]), names[three_d]

    # The multi-layer prefix rule the exclusion must not break: four soil
    # layers still collect under one canonical field.
    soil = skeleton["fields"]["soil_temperature"]["vtable_selectors"]
    assert len(soil) == 4, soil


def test_a_descriptor_fault_is_not_reported_as_a_vtable_fault(tmp_path):
    """The message pointed the reader at the wrong file.

    `Vtable line 11 is assigned more than once` names the Vtable, and
    line 11 of the Vtable was fine -- the double assignment was in the
    descriptor, which in the node-7 case the tool itself had written.
    """
    import pytest as _pytest

    from gpuwm.adapt import descriptor_skeleton
    from gpuwm.mapped_authoring import compile_mapping_descriptor

    vtable = packaged_gfs_vtable()
    skeleton = dict(descriptor_skeleton(vtable))
    fields = dict(skeleton["fields"])
    # Re-introduce exactly the defect that shipped.
    fields["air_temperature"] = {
        **fields["air_temperature"],
        "vtable_selectors": list(
            fields["air_temperature"]["vtable_selectors"])
        + list(fields["air_temperature_2m"]["vtable_selectors"]),
    }
    skeleton["fields"] = fields
    written = tmp_path / "descriptor.json"
    written.write_text(json.dumps(skeleton), encoding="utf-8")

    with _pytest.raises(ValueError) as caught:
        compile_mapping_descriptor(written, vtable_path=vtable)
    message = str(caught.value)
    assert "descriptor" in message
    assert "air_temperature" in message and "air_temperature_2m" in message


def test_the_gfs_vtable_ships_in_the_wheel():
    """A pip user must have the file the documented flow tells them to pass.

    It lived in `configs/`, which is not a package, so setuptools never
    carried it: `gpuwm adapt --vtable configs/Vtable.GFS.rw-wps` named a
    path that exists only in a checkout, while `gpuwm adapt` is exactly
    the surface a wheel-only user reaches for.  It ships beside the
    20CRv3 authorities now, under the same recursive package-data glob
    and the same byte contract.
    """
    import hashlib
    import tomllib

    from gpuwm.source_authorities import (
        packaged_gfs_vtable, packaged_gfs_vtable_sha256,
    )

    path = packaged_gfs_vtable()
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() \
        == packaged_gfs_vtable_sha256()

    # Inside the package, and matched by a declared package-data glob --
    # resolving on this machine is a checkout property, not a wheel one.
    package_root = Path(adapt.__file__).resolve().parent
    relative = path.relative_to(package_root)
    assert relative.parts[0] == "authorities"

    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"))
    globs = pyproject["tool"]["setuptools"]["package-data"]["gpuwm"]
    assert any(
        relative.match(pattern) or relative.as_posix().startswith(
            pattern.split("**")[0])
        for pattern in globs), (relative.as_posix(), globs)

    # And it is the real thing, not a stub: the parser reads it and the
    # skeleton generator produces a descriptor from it.
    assert len(adapt.parse_wps_vtable(path)) >= 20
