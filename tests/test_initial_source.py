"""Separate analysis bindings must not silently change a prepared start."""
import hashlib
import json
from pathlib import Path

import pytest

from gpuwm.initial_source import (
    REQUEST_SCHEMA, RECEIPT_SCHEMA, read_initial_inputs,
    validate_initial_evidence, write_initial_evidence,
)
from gpuwm.source_authorities import packaged_authorities, packaged_profile


def _request(tmp_path, **changes):
    (tmp_path / "analysis.grib2").write_bytes(b"test input")
    request = {"schema": REQUEST_SCHEMA, "source": "hrrr-prs",
               "input_files": ["analysis.grib2"], "supplements": {}}
    request.update(changes)
    path = tmp_path / "initial.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def test_initial_paths_follow_the_declaration_not_process_cwd(tmp_path, monkeypatch):
    request = _request(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    parsed = read_initial_inputs(request)
    assert parsed["source"] == "hrrr-prs"
    assert parsed["primary"] == ((tmp_path / "analysis.grib2").resolve(),)


@pytest.mark.parametrize("change,message", [
    ({"input_files": ["absent"]}, "missing"),
    ({"input_files": ["analysis.grib2", "analysis.grib2"]}, "repeats"),
    ({"input_files": []}, "nonempty"),
    ({"source": "hrrr"}, "packaged mapping"),
    ({"unexpected": True}, "must declare"),
    ({"supplements": {"soil": "analysis.grib2"}}, "path lists"),
])
def test_bad_initial_inventory_refuses_before_decode(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        read_initial_inputs(_request(tmp_path, **change))


def _bound_evidence(tmp_path):
    authorities = packaged_authorities("hrrr-prs-grib2-v1")
    evidence = {f"{role}.json": Path(authorities[role]).read_bytes()
                for role in ("mapping", "composition", "provenance")}
    evidence["request.json"] = _request(tmp_path).read_bytes()
    manifest = {f"{role}_sha256": hashlib.sha256(evidence[f"{role}.json"]).hexdigest()
                for role in ("mapping", "composition")}
    from gpuwm.mapped_composition import (
        INPUT_MANIFEST_SCHEMA, RECEIPT_SCHEMA as COMPOSITION_RECEIPT_SCHEMA,
        _canonical_sha256,
    )
    provenance_role = packaged_profile("hrrr-prs-grib2-v1")["provenance_role"]
    manifest.update(schema=INPUT_MANIFEST_SCHEMA,
                    primary_files=[{"path": "analysis.grib2", "sha256": "1" * 64}],
                    supplements={},
                    provenance={provenance_role: {"sha256": hashlib.sha256(evidence["provenance.json"]).hexdigest()}},
                    decoders={"mapped_engine": {"sha256": "2" * 64}})
    evidence["input-manifest.json"] = json.dumps(manifest).encode()
    composed = {role: {"sha256": hashlib.sha256(evidence[name]).hexdigest()}
                for role, name in (("mapping", "mapping.json"),
                                   ("composition", "composition.json"),
                                   ("input_manifest", "input-manifest.json"))}
    composed.update(schema=COMPOSITION_RECEIPT_SCHEMA,
                    valid_times=["2026-01-01T00:00:00"], frame_count=1,
                    frames=[{"header_sha256": "3" * 64, "terrain_sha256": "4" * 64,
                             "field_count": 1}], decoders=manifest["decoders"])
    composed["receipt_content_sha256"] = _canonical_sha256(composed)
    evidence["composition-receipt.json"] = json.dumps(composed).encode()
    root = tmp_path / "prepared"
    (root / "source-evidence").mkdir(parents=True)
    write_initial_evidence(root / "source-evidence", evidence)
    receipt = {"schema": RECEIPT_SCHEMA, "source": "hrrr-prs",
               "valid_time": "2026-01-01T00:00:00", "aerosol_source": "boundary-analysis",
               "evidence": {name: hashlib.sha256(data).hexdigest()
                            for name, data in evidence.items()}}
    return root, receipt


def test_initial_evidence_is_portable_without_raw_inputs(tmp_path):
    root, receipt = _bound_evidence(tmp_path)
    (tmp_path / "analysis.grib2").unlink()
    assert validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"]) == receipt


@pytest.mark.parametrize("name", ["mapping.json", "input-manifest.json", "composition-receipt.json"])
def test_changed_initial_authority_refuses_at_forecast(tmp_path, name):
    root, receipt = _bound_evidence(tmp_path)
    (root / "source-evidence" / "initial" / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


def test_initial_time_cannot_be_relabelled(tmp_path):
    root, receipt = _bound_evidence(tmp_path)
    with pytest.raises(ValueError, match="another valid time"):
        validate_initial_evidence(root, receipt, valid_time="2026-01-01T01:00:00")


@pytest.mark.parametrize("name", ["../outside", "..\\outside"])
def test_initial_evidence_cannot_escape_bundle(tmp_path, name):
    root, receipt = _bound_evidence(tmp_path)
    receipt["evidence"][name] = "0" * 64
    with pytest.raises(ValueError, match="inside its evidence directory"):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


@pytest.mark.parametrize("change,message", [
    ({"input_manifest": {"sha256": "0" * 64}}, "different input_manifest"),
    ({"valid_times": ["2026-01-01T03:00:00"]}, "another valid time"),
    ({"decoders": {"mapped_engine": {"sha256": "3" * 64}}}, "different decoders"),
])
def test_self_consistent_receipt_still_has_to_bind_the_analysis(tmp_path, change, message):
    # A receipt's own hash is insufficient if it names a different input.
    from gpuwm.mapped_composition import _canonical_sha256
    root, receipt = _bound_evidence(tmp_path)
    path = root / "source-evidence" / "initial" / "composition-receipt.json"
    composed = json.loads(path.read_bytes())
    composed.update(change)
    composed.pop("receipt_content_sha256")
    composed["receipt_content_sha256"] = _canonical_sha256(composed)
    path.write_text(json.dumps(composed), encoding="utf-8")
    receipt["evidence"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=message):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


def _replace_evidence(root, receipt, name, content):
    """Keep content hashes coherent so the semantic validator is exercised."""
    from gpuwm.mapped_composition import _canonical_sha256
    path = root / "source-evidence" / "initial" / name
    if name == "composition-receipt.json" and isinstance(content, dict):
        content.pop("receipt_content_sha256", None)
        content["receipt_content_sha256"] = _canonical_sha256(content)
    path.write_text(json.dumps(content), encoding="utf-8")
    receipt["evidence"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if name == "input-manifest.json":
        composed = json.loads((path.parent / "composition-receipt.json").read_bytes())
        composed["input_manifest"]["sha256"] = receipt["evidence"][name]
        _replace_evidence(root, receipt, "composition-receipt.json", composed)


@pytest.mark.parametrize("key,value", [("source", None), ("source", []),
                                       ("aerosol_source", []), ("aerosol_source", {})])
def test_malformed_receipt_selector_is_a_named_refusal(tmp_path, key, value):
    root, receipt = _bound_evidence(tmp_path)
    receipt[key] = value
    with pytest.raises(ValueError, match="initial analysis receipt"):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


@pytest.mark.parametrize("name,value,match", [
    ("input-manifest.json", [], "input manifest must be an object"),
    ("request.json", [], "initial inputs must declare"),
    ("composition-receipt.json", [], "composition receipt must be an object"),
])
def test_nonobject_evidence_cannot_crash_validation(tmp_path, name, value, match):
    root, receipt = _bound_evidence(tmp_path)
    _replace_evidence(root, receipt, name, value)
    with pytest.raises(ValueError, match=match):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


@pytest.mark.parametrize("field,value,match", [
    ("mapping", [], "different mapping"),
    ("input_manifest", None, "different input_manifest"),
    ("decoders", {"mapped_engine": []}, "decoder inventory needs"),
    ("decoders", {"mapped_engine": {}}, "decoder inventory needs"),
    ("frames", [{}], "no bound frame"),
    ("frames", [None], "no bound frame"),
    ("frame_count", True, "frame count"),
])
def test_incomplete_composition_rows_are_rejected_despite_valid_outer_hash(tmp_path, field, value, match):
    root, receipt = _bound_evidence(tmp_path)
    path = root / "source-evidence" / "initial" / "composition-receipt.json"
    document = json.loads(path.read_bytes())
    document[field] = value
    _replace_evidence(root, receipt, path.name, document)
    with pytest.raises(ValueError, match=match):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


@pytest.mark.parametrize("field,value,match", [
    ("decoders", {"mapped_engine": None}, "decoder inventory needs"),
    ("decoders", {"mapped_engine": {}}, "decoder inventory needs"),
    ("primary_files", [{"sha256": ""}], "no bound source inventory"),
    ("supplements", [], "different or unbound supplements"),
    ("provenance", [], "different provenance"),
])
def test_incomplete_manifest_rows_are_rejected_despite_valid_outer_hash(tmp_path, field, value, match):
    root, receipt = _bound_evidence(tmp_path)
    path = root / "source-evidence" / "initial" / "input-manifest.json"
    document = json.loads(path.read_bytes())
    document[field] = value
    _replace_evidence(root, receipt, path.name, document)
    with pytest.raises(ValueError, match=match):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


@pytest.mark.parametrize("field,value", [("source", None), ("input_files", None),
                                       ("input_files", []), ("supplements", [])])
def test_restored_request_has_the_same_shape_checks_as_a_fresh_request(tmp_path, field, value):
    root, receipt = _bound_evidence(tmp_path)
    path = root / "source-evidence" / "initial" / "request.json"
    document = json.loads(path.read_bytes())
    document[field] = value
    _replace_evidence(root, receipt, path.name, document)
    with pytest.raises(ValueError):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


def test_evidence_directory_symlink_cannot_read_outside_the_prepared_bundle(tmp_path):
    root, receipt = _bound_evidence(tmp_path)
    directory = root / "source-evidence" / "initial"
    outside = tmp_path / "elsewhere"
    directory.rename(outside)
    try:
        directory.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this host cannot create a directory symlink")
    with pytest.raises(ValueError, match="inside its prepared directory"):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


@pytest.mark.parametrize("name", ["../outside.json", "..\\outside.json", "C:outside.json"])
def test_evidence_writer_rejects_an_escaping_name_before_creating_files(tmp_path, name):
    with pytest.raises(ValueError, match="inside its evidence directory"):
        write_initial_evidence(tmp_path, {name: b"invalid"})
    assert not (tmp_path / "initial").exists()


def test_original_boundary_head_backend_receipt_is_preserved(tmp_path):
    root, receipt = _bound_evidence(tmp_path)
    phase = {"backend": "cuda", "selection": {"requested": "auto"},
             "contracts": {"vertical": {"policy": "serialized-endpoint"}}}
    receipt["boundary_head_preprocessing"] = phase
    result = validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])
    assert result["boundary_head_preprocessing"] == phase


@pytest.mark.parametrize("phase", [None, [], {}, {"backend": []}, {"backend": "unknown"},
                                    {"backend": "cpu", "measurement": float("nan")}])
def test_invalid_boundary_head_backend_receipt_is_a_named_refusal(tmp_path, phase):
    root, receipt = _bound_evidence(tmp_path)
    receipt["boundary_head_preprocessing"] = phase
    with pytest.raises(ValueError, match="boundary-head preprocessing"):
        validate_initial_evidence(root, receipt, valid_time=receipt["valid_time"])


@pytest.mark.parametrize("failure", ["wrong_time", "consumer", "decode"])
def test_initial_decode_releases_bundle_and_scratch_on_failure(tmp_path, monkeypatch, failure):
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from gpuwm.initial_source import decode_initial_analysis
    import gpuwm.mapped_authoring as authoring
    import gpuwm.mapped_composition as composition

    request = read_initial_inputs(_request(tmp_path))
    valid = datetime(2026, 1, 1)
    records = {"closed": 0}

    def author(path, **_kwargs):
        Path(path).write_text("{}")

    def decode(*_args, **kwargs):
        scratch = Path(kwargs["scratch_destination"])
        (scratch / "mapped-data").write_bytes(b"owned scratch")
        records["scratch"] = scratch
        if failure == "decode":
            raise RuntimeError("decoder failed")
        time = valid + timedelta(hours=1) if failure == "wrong_time" else valid
        snapshot = SimpleNamespace(valid_time=time)

        def close():
            records["closed"] += 1

        return SimpleNamespace(regular_snapshots=lambda: (snapshot,),
                               soil_layer_contract={}, close=close)

    monkeypatch.setattr(authoring, "author_input_manifest", author)
    monkeypatch.setattr(composition, "decode_composed_source", decode)
    monkeypatch.setattr(composition, "mapped_composition_receipt", lambda _bundle: {})
    expected = ValueError if failure == "wrong_time" else RuntimeError
    with pytest.raises(expected):
        with decode_initial_analysis(request, output_parent=tmp_path / "decode",
                                     grids=(), valid_time=valid):
            raise RuntimeError("consumer failed")
    assert records["closed"] == (0 if failure == "decode" else 1)
    assert not records["scratch"].exists()
