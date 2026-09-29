from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from gpuwm import mapped_authoring, source_cli, source_normalization


ROOT = Path(__file__).parents[1]
AUTHORITIES = [
    (
        "configs/rw-wps-era5-netcdf.mapping.json",
        "configs/rw-wps-era5-netcdf-terrain.composition.json",
    ),
]


@pytest.fixture(params=AUTHORITIES)
def saved_inputs(tmp_path, monkeypatch, request):
    # Exercise manifest I/O without requiring a native decoder installation.
    monkeypatch.setenv("GPUWM_MAPPED_ENGINE", "python")
    mapping_name, composition_name = request.param
    mapping = tmp_path / "mapping.json"
    mapping.write_bytes((ROOT / mapping_name).read_bytes())
    composition = ROOT / composition_name
    terrain = json.loads(composition.read_text())["supplements"]["terrain_height"]
    source = tmp_path / "source.nc"
    source.write_bytes(b"saved source bytes")
    provenance = tmp_path / "provenance.txt"
    provenance.write_text("saved provenance\n", encoding="utf-8")
    options = dict(
        mapping_path=mapping,
        composition_path=composition,
        primary_files=(source,),
        supplement_files={terrain["data_role"]: source},
        provenance_files={terrain["provenance_role"]: provenance},
    )
    manifest = tmp_path / "inputs.json"
    mapped_authoring.author_input_manifest(manifest, **options)
    original = manifest.read_bytes()
    # A release can change mapping bytes without changing fetched data.
    document = json.loads(mapping.read_text())
    document["name"] += "-updated"
    mapping.write_text(json.dumps(document), encoding="utf-8")
    return manifest, original, options


@pytest.mark.parametrize("replace_different", [False, True])
def test_mapping_upgrade_preserves_original_and_reuses_sidecar(saved_inputs, replace_different):
    manifest, original, options = saved_inputs
    if replace_different:
        options["replace_different"] = True
    receipt = mapped_authoring.author_input_manifest(manifest, **options)
    selected = Path(receipt["manifest"]["path"])
    assert selected != manifest
    assert selected.parent == manifest.parent
    assert manifest.read_bytes() == original
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == receipt["manifest"]["sha256"]
    updated = json.loads(selected.read_bytes())
    old = json.loads(original)
    assert updated.pop("mapping_sha256") != old.pop("mapping_sha256")
    assert updated == old
    before = selected.stat().st_mtime_ns
    repeated = mapped_authoring.author_input_manifest(manifest, **options)
    assert repeated["reauthored"] is False
    assert repeated["manifest"] == receipt["manifest"]
    assert selected.stat().st_mtime_ns == before


@pytest.mark.parametrize("field", [
    "primary_files", "supplements", "schema", "member", "member_identity",
    "unknown_field",
])
def test_upgrade_refuses_other_manifest_changes(saved_inputs, field):
    manifest, original, options = saved_inputs
    document = json.loads(original)
    document[field] = "changed"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    before = manifest.read_bytes()
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **options)
    assert manifest.read_bytes() == before
    assert list(manifest.parent.glob("inputs.*.json")) == []


@pytest.mark.parametrize("field", ["decoders", "provenance", "composition_sha256"])
@pytest.mark.parametrize("mapping_changed", [False, True])
def test_upgrade_reseals_release_owned_rows(saved_inputs, field, mapping_changed):
    manifest, original, options = saved_inputs
    previous = json.loads(original)
    if not mapping_changed:
        previous["mapping_sha256"] = hashlib.sha256(
            options["mapping_path"].read_bytes()).hexdigest()
    if field == "composition_sha256":
        previous[field] = "a" * 64
    else:
        previous[field] = {
            "old_release_authority": {
                "path": "../old-install/authority", "bytes": 10, "sha256": "b" * 64,
            },
        }
    manifest.write_text(json.dumps(previous), encoding="utf-8")
    before = manifest.read_bytes()
    receipt = mapped_authoring.author_input_manifest(manifest, **options)
    selected = Path(receipt["manifest"]["path"])
    assert selected != manifest
    assert manifest.read_bytes() == before
    assert json.loads(selected.read_bytes())[field] != previous[field]
    repeated = mapped_authoring.author_input_manifest(manifest, **options)
    assert repeated["reauthored"] is False
    assert repeated["manifest"] == receipt["manifest"]


@pytest.mark.parametrize("mapping_name,composition_name", [(
    "configs/rw-wps-gfs-pressure-grib2.mapping.json",
    "configs/rw-wps-gfs-terrain.composition.json",
)])
def test_upgrade_reseals_changed_decoder_binaries(
        tmp_path, monkeypatch, mapping_name, composition_name):
    def identity(path, role):
        data = Path(path).read_bytes()
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    monkeypatch.setattr(mapped_authoring, "bridge_identity", identity)
    composition = ROOT / composition_name
    terrain = json.loads(composition.read_text())["supplements"]["terrain_height"]
    source = tmp_path / "source.grib2"
    source.write_bytes(b"saved input")
    provenance = tmp_path / "provenance.txt"
    provenance.write_text("saved provenance\n", encoding="utf-8")
    inventory = tmp_path / "grib2_inventory"
    dump = tmp_path / "grib2_dump"
    inventory.write_bytes(b"previous inventory")
    dump.write_bytes(b"previous dump")
    options = dict(
        mapping_path=ROOT / mapping_name, composition_path=composition,
        primary_files=(source,),
        supplement_files={terrain["data_role"]: source},
        provenance_files={terrain["provenance_role"]: provenance},
        grib2_inventory=inventory, grib2_dump=dump,
    )
    manifest = tmp_path / "inputs.json"
    mapped_authoring.author_input_manifest(manifest, **options)
    original = manifest.read_bytes()
    inventory.write_bytes(b"current inventory binary")
    dump.write_bytes(b"current dump binary")
    receipt = mapped_authoring.author_input_manifest(manifest, **options)
    selected = Path(receipt["manifest"]["path"])
    assert selected != manifest
    assert manifest.read_bytes() == original
    payload = json.loads(selected.read_bytes())
    for role, binary in (("grib2_inventory", inventory), ("grib2_dump", dump)):
        assert payload["decoders"][role]["sha256"] == identity(binary, role)["sha256"]
        assert payload["decoders"][role] != json.loads(original)["decoders"][role]
    before = selected.stat().st_mtime_ns
    repeated = mapped_authoring.author_input_manifest(manifest, **options)
    assert repeated["reauthored"] is False
    assert repeated["manifest"] == receipt["manifest"]
    assert selected.stat().st_mtime_ns == before


@pytest.mark.parametrize("old_hash", [None, "invalid", 123])
def test_upgrade_refuses_invalid_old_mapping_hash(saved_inputs, old_hash):
    manifest, original, options = saved_inputs
    document = json.loads(original)
    document["mapping_sha256"] = old_hash
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **options)


def test_upgrade_refuses_changed_source_bytes_with_same_size_and_mtime(saved_inputs):
    manifest, original, options = saved_inputs
    source = options["primary_files"][0]
    stat = source.stat()
    source.write_bytes(b"x" * stat.st_size)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **options)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("replace_different", [False, True])
def test_upgrade_does_not_overwrite_sidecar_collision(saved_inputs, replace_different):
    manifest, original, options = saved_inputs
    if replace_different:
        options["replace_different"] = True
    receipt = mapped_authoring.author_input_manifest(manifest, **options)
    selected = Path(receipt["manifest"]["path"])
    selected.write_bytes(b"unrelated content")
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **options)
    assert selected.read_bytes() == b"unrelated content"
    assert manifest.read_bytes() == original


def test_upgrade_validation_failure_leaves_original(saved_inputs, monkeypatch):
    manifest, original, options = saved_inputs

    def fail_validation(*args, **kwargs):
        raise ValueError("source changed after validation")

    monkeypatch.setattr(mapped_authoring, "_verify_manifest", fail_validation)
    with pytest.raises(ValueError, match="source changed after validation"):
        mapped_authoring.author_input_manifest(manifest, **options)
    assert manifest.read_bytes() == original
    assert list(manifest.parent.glob("inputs.*.json")) == []
    assert list(manifest.parent.glob(".*.candidate-*")) == []


@pytest.mark.parametrize("managed", [False, True])
def test_cli_upgrade_binds_selected_manifest(saved_inputs, capsys, managed):
    manifest, original, options = saved_inputs
    args = source_cli._parser().parse_args([
        "--source", "mapped", "--source-format", "netcdf",
        "--mapping", str(options["mapping_path"]),
        "--composition", str(options["composition_path"]),
        "--input", str(options["primary_files"][0]),
        "--author-input-manifest", str(manifest), "--author-only",
        *[token for role, path in options["supplement_files"].items()
          for token in ("--supplement", f"{role}={path}")],
        *[token for role, path in options["provenance_files"].items()
          for token in ("--provenance", f"{role}={path}")],
    ])
    args.source_root_manifest = managed
    result = source_cli._author_mapped_contract(args)
    selected = result["input_manifest"]["manifest"]
    assert Path(args.source_sha256s) == Path(selected["path"])
    assert args.source_sha256s_sha256 == selected["sha256"]
    stderr = capsys.readouterr().err
    assert "REPLACED" not in stderr
    assert str(args.source_sha256s) in stderr
    explanation = next((line for line in stderr.splitlines()
                        if "left unchanged; it was sealed with other mapping, "
                        "composition, decoder or provenance files" in line), "")
    assert str(manifest) in explanation
    assert str(args.source_sha256s) in explanation
    assert "this run binds" in explanation
    assert manifest.read_bytes() == original


def _managed_args(options, manifest):
    args = source_cli._parser().parse_args([
        "--source", "mapped", "--source-format", "netcdf",
        "--mapping", str(options["mapping_path"]),
        "--composition", str(options["composition_path"]),
        "--input", str(options["primary_files"][0]),
        "--author-input-manifest", str(manifest), "--author-only",
        *[token for role, path in options["supplement_files"].items()
          for token in ("--supplement", f"{role}={path}")],
        *[token for role, path in options["provenance_files"].items()
          for token in ("--provenance", f"{role}={path}")],
    ])
    args.source_root_manifest = True
    return args


def test_managed_replacement_removes_the_upgrade_it_replaced(saved_inputs, capsys):
    """Named breakage: a version-only rerun wrote inputs.<digest>.json, a
    later rerun after the fetched bytes changed replaced inputs.json, and
    the sibling stayed behind describing data the folder no longer held."""

    manifest, original, options = saved_inputs
    source_cli._author_mapped_contract(_managed_args(options, manifest))
    [sibling] = [path.resolve() for path in manifest.parent.glob("inputs.*.json")]
    assert manifest.read_bytes() == original
    stderr = capsys.readouterr().err
    assert "REPLACED" not in stderr and f"this run binds {sibling}" in stderr

    options["primary_files"][0].write_bytes(b"refetched source bytes")
    args = _managed_args(options, manifest)
    source_cli._author_mapped_contract(args)
    assert Path(args.source_sha256s) == manifest.resolve()
    assert manifest.read_bytes() != original
    assert list(manifest.parent.glob("inputs.*.json")) == []
    stderr = capsys.readouterr().err
    [replaced] = [line for line in stderr.splitlines()
                  if line.startswith("REPLACED input_manifest=")]
    assert hashlib.sha256(original).hexdigest() in replaced
    assert f"removed {sibling}" in replaced
    assert "KEPT" not in stderr

    before = manifest.read_bytes()
    source_cli._author_mapped_contract(_managed_args(options, manifest))
    stderr = capsys.readouterr().err
    assert "REPLACED" not in stderr
    assert f"MATCHED input_manifest={manifest.resolve()}" in stderr
    assert manifest.read_bytes() == before


def test_managed_replacement_names_an_upgrade_it_could_not_remove(
        saved_inputs, capsys, monkeypatch):
    manifest, original, options = saved_inputs
    source_cli._author_mapped_contract(_managed_args(options, manifest))
    [sibling] = [path.resolve() for path in manifest.parent.glob("inputs.*.json")]
    unlink = Path.unlink

    def held(path, *args, **kwargs):
        if Path(path).resolve() == sibling:
            raise PermissionError("held by another process")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", held)
    options["primary_files"][0].write_bytes(b"refetched source bytes")
    capsys.readouterr()
    source_cli._author_mapped_contract(_managed_args(options, manifest))
    assert manifest.read_bytes() != original
    assert sibling.is_file()
    [replaced] = [line for line in capsys.readouterr().err.splitlines()
                  if line.startswith("REPLACED input_manifest=")]
    assert f"could not remove {sibling}" in replaced
    assert "held by another process" in replaced


def test_managed_replacement_removes_every_upgrade_of_the_replaced_data(saved_inputs):
    manifest, original, options = saved_inputs
    options["replace_different"] = True
    for suffix in ("-first", "-second"):
        mapped_authoring.author_input_manifest(manifest, **options)
        document = json.loads(options["mapping_path"].read_text())
        document["name"] += suffix
        options["mapping_path"].write_text(json.dumps(document), encoding="utf-8")
    siblings = sorted(path.resolve() for path in manifest.parent.glob("inputs.*.json"))
    assert len(siblings) == 2
    options["primary_files"][0].write_bytes(b"refetched source bytes")
    receipt = mapped_authoring.author_input_manifest(manifest, **options)
    assert Path(receipt["manifest"]["path"]) == manifest
    assert sorted(Path(row["path"]) for row in receipt["removed_manifests"]) == siblings
    assert list(manifest.parent.glob("inputs.*.json")) == []


def test_managed_replacement_keeps_files_that_are_not_its_upgrades(saved_inputs):
    manifest, original, options = saved_inputs
    options["replace_different"] = True
    upgraded = Path(mapped_authoring.author_input_manifest(
        manifest, **options)["manifest"]["path"])
    other_data = json.loads(upgraded.read_bytes())
    other_data["primary_files"][0]["sha256"] = "c" * 64
    other_bytes = mapped_authoring._canonical_json(other_data)
    other = manifest.with_name(
        f"inputs.{hashlib.sha256(other_bytes).hexdigest()}.json")
    other.write_bytes(other_bytes)
    renamed = manifest.with_name(f"inputs.{'d' * 64}.json")
    renamed.write_bytes(upgraded.read_bytes())
    backup = manifest.with_name("inputs.backup.json")
    backup.write_bytes(upgraded.read_bytes())
    options["primary_files"][0].write_bytes(b"refetched source bytes")
    receipt = mapped_authoring.author_input_manifest(manifest, **options)
    assert [row["path"] for row in receipt["removed_manifests"]] == [str(upgraded)]
    assert not upgraded.exists()
    # A different data set, a name its bytes do not hash to, and a name
    # no upgrade writes are not an upgrade of the replaced manifest.
    assert other.read_bytes() == other_bytes
    assert renamed.is_file() and backup.is_file()


def test_failed_managed_replacement_keeps_the_upgrade(saved_inputs, monkeypatch):
    manifest, original, options = saved_inputs
    options["replace_different"] = True
    upgraded = Path(mapped_authoring.author_input_manifest(
        manifest, **options)["manifest"]["path"])
    kept = upgraded.read_bytes()
    options["primary_files"][0].write_bytes(b"refetched source bytes")

    def fail_validation(*args, **kwargs):
        raise ValueError("source changed after validation")

    monkeypatch.setattr(mapped_authoring, "_verify_manifest", fail_validation)
    with pytest.raises(ValueError, match="source changed after validation"):
        mapped_authoring.author_input_manifest(manifest, **options)
    assert manifest.read_bytes() == original
    assert upgraded.read_bytes() == kept


def test_a_replacement_with_no_upgrade_reports_no_removal(saved_inputs):
    manifest, original, options = saved_inputs
    options["replace_different"] = True
    options["primary_files"][0].write_bytes(b"refetched source bytes")
    receipt = mapped_authoring.author_input_manifest(manifest, **options)
    assert Path(receipt["manifest"]["path"]) == manifest
    assert manifest.read_bytes() != original
    assert "removed_manifests" not in receipt


@pytest.fixture
def normalized_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_MAPPED_ENGINE", "python")
    mapping_name, composition_name = AUTHORITIES[0]
    composition = ROOT / composition_name
    terrain = json.loads(composition.read_text())["supplements"]["terrain_height"]

    def identity(path):
        data = path.read_bytes()
        return {"path": str(path), "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    raw = [tmp_path / "raw-a", tmp_path / "raw-b"]
    for index, path in enumerate(raw):
        path.write_bytes(f"raw object {index}".encode())
    converter = tmp_path / "converter"

    def artifact(version):
        converter.write_bytes(version)
        request = {"inputs": [identity(path) for path in raw],
                   "target": {"nx": 2, "ny": 2, "dx": 0.25, "dy": 0.25},
                   "converter": identity(converter)}
        key = hashlib.sha256(source_normalization._canonical(request)).hexdigest()
        directory = tmp_path / "normalized" / key
        directory.mkdir(parents=True)
        field = directory / "field.nc"
        field.write_bytes(b"normalized field bytes")
        provenance = directory / "provenance.json"
        provenance.write_text(json.dumps({
            source_normalization.NORMALIZATION_RECEIPT_KEY: {"request": request},
        }), encoding="utf-8")
        return dict(
            mapping_path=ROOT / mapping_name, composition_path=composition,
            primary_files=(field,),
            supplement_files={terrain["data_role"]: field},
            provenance_files={terrain["provenance_role"]: provenance},
        )

    previous = artifact(b"old converter")
    manifest = tmp_path / "inputs.json"
    mapped_authoring.author_input_manifest(manifest, **previous)
    original = manifest.read_bytes()
    current = artifact(b"new converter")
    return manifest, original, previous, current


@pytest.mark.parametrize("replace_different", [False, True])
def test_normalization_converter_upgrade_reuses_raw_inputs(normalized_inputs, replace_different):
    manifest, original, previous, current = normalized_inputs
    if replace_different:
        current["replace_different"] = True
    assert previous["primary_files"] != current["primary_files"]
    receipt = mapped_authoring.author_input_manifest(manifest, **current)
    selected = Path(receipt["manifest"]["path"])
    assert selected != manifest
    assert manifest.read_bytes() == original
    payload = json.loads(selected.read_bytes())
    assert payload["primary_files"] != json.loads(original)["primary_files"]
    assert payload["supplements"] != json.loads(original)["supplements"]
    before = selected.stat().st_mtime_ns
    repeated = mapped_authoring.author_input_manifest(manifest, **current)
    assert repeated["reauthored"] is False
    assert repeated["manifest"] == receipt["manifest"]
    assert selected.stat().st_mtime_ns == before


@pytest.mark.parametrize("change", ["sha256", "path", "bytes", "order", "target"])
@pytest.mark.parametrize("same_outputs", [False, True])
def test_normalization_upgrade_refuses_changed_raw_facts(
        normalized_inputs, change, same_outputs):
    manifest, original, previous, current = normalized_inputs
    provenance = next(iter(current["provenance_files"].values()))
    document = json.loads(provenance.read_bytes())
    request = document[source_normalization.NORMALIZATION_RECEIPT_KEY]["request"]
    if change == "target":
        request["target"]["nx"] += 1
    elif change == "order":
        request["inputs"].reverse()
    else:
        request["inputs"][0][change] = {"sha256": "a" * 64,
                                        "path": "other-raw", "bytes": 123}[change]
    provenance.write_text(json.dumps(document), encoding="utf-8")
    if same_outputs:
        current["primary_files"] = previous["primary_files"]
        current["supplement_files"] = previous["supplement_files"]
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **current)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("side", ["previous", "current"])
@pytest.mark.parametrize("change,same_outputs", [
    ("missing", False), ("invalid", False), ("empty", False),
    ("invalid", True), ("empty", True),
])
def test_normalization_upgrade_requires_both_receipts(
        normalized_inputs, side, change, same_outputs):
    manifest, original, previous, current = normalized_inputs
    options = previous if side == "previous" else current
    provenance = next(iter(options["provenance_files"].values()))
    replacement = {"missing": {}, "invalid": {"normalization": {}},
                   "empty": {"normalization": {"request": {"inputs": [], "target": {}}}}}
    provenance.write_text(json.dumps(replacement[change]), encoding="utf-8")
    if side == "previous":
        payload = json.loads(original)
        row = next(iter(payload["provenance"].values()))
        row.update(bytes=provenance.stat().st_size,
                   sha256=hashlib.sha256(provenance.read_bytes()).hexdigest())
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    before = manifest.read_bytes()
    if same_outputs:
        current["primary_files"] = previous["primary_files"]
        current["supplement_files"] = previous["supplement_files"]
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **current)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("change", ["bytes", "sha256"])
def test_normalization_upgrade_checks_kept_provenance_identity(normalized_inputs, change):
    manifest, original, previous, current = normalized_inputs
    provenance = next(iter(previous["provenance_files"].values()))
    contents = provenance.read_bytes()
    if change == "bytes":
        provenance.write_bytes(contents + b" ")
    else:
        document = json.loads(contents)
        request = document["normalization"]["request"]
        request["converter"]["sha256"] = "a" * 64
        changed = json.dumps(document).encode()
        assert len(changed) == len(contents)
        provenance.write_bytes(changed)
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **current)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("field", ["schema", "member", "member_identity", "unknown_field"])
def test_normalization_upgrade_preserves_other_refusals(normalized_inputs, field):
    manifest, original, previous, current = normalized_inputs
    payload = json.loads(original)
    payload[field] = "different"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **current)


def test_normalization_upgrade_refuses_ambiguous_receipts(normalized_inputs):
    manifest, original, previous, current = normalized_inputs
    payload = json.loads(original)
    payload["provenance"]["other"] = next(iter(payload["provenance"].values()))
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different input manifest"):
        mapped_authoring.author_input_manifest(manifest, **current)


@pytest.mark.parametrize("repeat", [False, True])
def test_normalization_upgrade_rechecks_receipts_before_publication(
        normalized_inputs, monkeypatch, repeat):
    manifest, original, previous, current = normalized_inputs
    if repeat:
        mapped_authoring.author_input_manifest(manifest, **current)
    select = mapped_authoring._manifest_destination

    def select_then_mutate(*args, **kwargs):
        destination = select(*args, **kwargs)
        provenance = next(iter(previous["provenance_files"].values()))
        provenance.write_bytes(provenance.read_bytes() + b" ")
        return destination

    monkeypatch.setattr(mapped_authoring, "_manifest_destination", select_then_mutate)
    with pytest.raises(ValueError, match="changed after validation"):
        mapped_authoring.author_input_manifest(manifest, **current)
    assert manifest.read_bytes() == original
    assert len(list(manifest.parent.glob("inputs.*.json"))) == int(repeat)


@pytest.mark.parametrize("mutate_receipt", [False, True])
def test_managed_normalization_replacement_checks_snapshots(
        normalized_inputs, monkeypatch, mutate_receipt):
    manifest, original, previous, current = normalized_inputs
    provenance = next(iter(current["provenance_files"].values()))
    document = json.loads(provenance.read_bytes())
    document["normalization"]["request"]["inputs"][0]["sha256"] = "a" * 64
    provenance.write_text(json.dumps(document), encoding="utf-8")
    current["replace_different"] = True
    select = mapped_authoring._manifest_destination

    def select_then_mutate(*args, **kwargs):
        destination = select(*args, **kwargs)
        assert destination == manifest
        if mutate_receipt:
            kept = next(iter(previous["provenance_files"].values()))
            kept.write_bytes(kept.read_bytes() + b" ")
        return destination

    monkeypatch.setattr(mapped_authoring, "_manifest_destination", select_then_mutate)
    if mutate_receipt:
        with pytest.raises(ValueError, match="changed after validation"):
            mapped_authoring.author_input_manifest(manifest, **current)
        assert manifest.read_bytes() == original
    else:
        receipt = mapped_authoring.author_input_manifest(manifest, **current)
        assert Path(receipt["manifest"]["path"]) == manifest
        assert manifest.read_bytes() != original
        repeated = mapped_authoring.author_input_manifest(manifest, **current)
        assert repeated["reauthored"] is False
    assert list(manifest.parent.glob("inputs.*.json")) == []


def test_managed_raw_input_change_removes_the_converter_upgrade(normalized_inputs):
    manifest, original, previous, current = normalized_inputs
    current["replace_different"] = True
    upgraded = Path(mapped_authoring.author_input_manifest(
        manifest, **current)["manifest"]["path"])
    assert upgraded != manifest and manifest.read_bytes() == original
    # A refetch changes the raw inputs, so the converter writes a new key.
    kept = next(iter(current["provenance_files"].values()))
    document = json.loads(kept.read_bytes())
    request = document[source_normalization.NORMALIZATION_RECEIPT_KEY]["request"]
    request["inputs"][0]["sha256"] = "a" * 64
    key = hashlib.sha256(source_normalization._canonical(request)).hexdigest()
    directory = manifest.parent / "normalized" / key
    directory.mkdir(parents=True)
    field = directory / "field.nc"
    field.write_bytes(b"refetched normalized field bytes")
    provenance = directory / "provenance.json"
    provenance.write_text(json.dumps(document), encoding="utf-8")
    refetched = dict(current,
                     primary_files=(field,),
                     supplement_files={role: field for role in current["supplement_files"]},
                     provenance_files={role: provenance
                                       for role in current["provenance_files"]})
    receipt = mapped_authoring.author_input_manifest(manifest, **refetched)
    assert Path(receipt["manifest"]["path"]) == manifest
    assert manifest.read_bytes() != original
    assert [row["path"] for row in receipt["removed_manifests"]] == [str(upgraded)]
    assert list(manifest.parent.glob("inputs.*.json")) == []
