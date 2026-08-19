"""The sim stage accepts a cross-source packaged preparation's receipt.

The mapped-evidence certificate pins the exact top-level inventory of the
composition receipt.  A cross-source preparation's receipt carries one
more key -- ``contributing_sources`` -- and its terrain alignment is the
cross-source binding receipt, not the exact-subset supplement receipt, so
a validator written for single-source profiles refuses the bundle its own
prep stage just published.  These tests hold the extended certificate:
the extra key is accepted exactly when the PACKAGED composition declares
``field_sources`` (never by sniffing the receipt), the terrain data role
comes from the terrain-carrying binding, and every contributing source's
mapping hash must equal the packaged contributing-mapping pin.

The fixture proof and input manifest are the REAL artifacts published by
``gpuwm prep --source aigefs`` on staged 2026-08-17 00Z member bytes
(committed under ``tests/data/aigefs_member_hybrid``), so the shapes
asserted here are shapes the prep stage actually writes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from gpuwm import prepared_single_domain_forecast as runner
from gpuwm.mapped_source import _sha256
from gpuwm.source_authorities import packaged_authorities

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "data" / "aigefs_member_hybrid"
PROFILE_SOURCE = "aigefs"
PROFILE_ID = "aigefs-member-hybrid-grib2-v1"
EXPERIMENT_CONFIG = ROOT / "configs" / "aigefs_member_demo.toml"
WPS_NAMELIST = ROOT / "configs" / "aigefs_member_demo.namelist.wps"


def _canonical_hash(payload: dict) -> str:
    content = dict(payload)
    content.pop("proof_content_sha256", None)
    return hashlib.sha256(
        runner._canonical(content).encode("utf-8")).hexdigest()


@pytest.fixture()
def prepared_tree(tmp_path):
    """A prepared root carrying the real evidence and the real proof."""

    prepared = tmp_path / "prepared"
    evidence = prepared / "source-evidence"
    evidence.mkdir(parents=True)
    authorities = packaged_authorities(PROFILE_ID)
    shutil.copy2(authorities["mapping"], evidence / "mapping.json")
    shutil.copy2(authorities["composition"], evidence / "composition.json")
    shutil.copy2(
        authorities["provenance"],
        evidence / "provenance-physical-analysis-surface.json")
    shutil.copy2(
        FIXTURES / "input-manifest.json", evidence / "input-manifest.json")
    proof = json.loads(
        (FIXTURES / "proof.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (FIXTURES / "input-manifest.json").read_text(encoding="utf-8"))
    return prepared, proof, manifest


def _validate(prepared, proof, manifest):
    return runner._validate_packaged_mapped_evidence(
        prepared_root=prepared,
        proof=proof,
        manifest=manifest,
        manifest_sha256=_sha256(
            prepared / "source-evidence" / "input-manifest.json"),
        experiment_config=EXPERIMENT_CONFIG,
        wps_namelist=WPS_NAMELIST,
        source=PROFILE_SOURCE,
    )


def test_the_real_cross_source_proof_passes_the_certificate(prepared_tree):
    prepared, proof, manifest = prepared_tree
    _validate(prepared, proof, manifest)


def test_a_receipt_without_its_contributing_sources_refuses(prepared_tree):
    """The packaged composition declares bindings, so a receipt that
    names no contributing source is a receipt for some other decode."""

    prepared, proof, manifest = prepared_tree
    proof = copy.deepcopy(proof)
    receipt = proof["source_composition"]
    del receipt["contributing_sources"]
    content = dict(receipt)
    content.pop("receipt_content_sha256")
    receipt["receipt_content_sha256"] = hashlib.sha256(
        runner._canonical(content).encode("utf-8")).hexdigest()
    proof["proof_content_sha256"] = _canonical_hash(proof)
    with pytest.raises(ValueError, match="contributing"):
        _validate(prepared, proof, manifest)


def test_a_donor_mapping_off_the_packaged_pin_refuses(prepared_tree):
    prepared, proof, manifest = prepared_tree
    proof = copy.deepcopy(proof)
    receipt = proof["source_composition"]
    entry = receipt["contributing_sources"][0]
    entry["mapping"]["sha256"] = "0" * 64
    content = dict(receipt)
    content.pop("receipt_content_sha256")
    receipt["receipt_content_sha256"] = hashlib.sha256(
        runner._canonical(content).encode("utf-8")).hexdigest()
    proof["proof_content_sha256"] = _canonical_hash(proof)
    with pytest.raises(ValueError, match="contributing"):
        _validate(prepared, proof, manifest)


def test_a_failed_binding_alignment_refuses(prepared_tree):
    prepared, proof, manifest = prepared_tree
    proof = copy.deepcopy(proof)
    receipt = proof["source_composition"]
    receipt["alignment"]["status"] = "FAIL"
    receipt["contributing_sources"][0]["alignment"]["status"] = "FAIL"
    content = dict(receipt)
    content.pop("receipt_content_sha256")
    receipt["receipt_content_sha256"] = hashlib.sha256(
        runner._canonical(content).encode("utf-8")).hexdigest()
    proof["proof_content_sha256"] = _canonical_hash(proof)
    with pytest.raises(ValueError, match="alignment"):
        _validate(prepared, proof, manifest)
