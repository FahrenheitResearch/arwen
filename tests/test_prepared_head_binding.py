"""A forecast bound to a prepared HEAD runs the sealed checks at the seal.

``--prepared-head-sha256`` binds a chained preparation before its boundary
intervals exist.  The preflight checks everything the start time made
exactly as a sealed binding does; the checks that name the sealed cache
(its content digest, the export, the proof's own digest) run at the seal,
where :func:`_seal_streamed_inputs` re-runs the complete preflight on the
sealed tree and holds it to the head the forecast started from.

Built on the runner's own synthetic GFS bundle, turned into the chained
layout: the head is the bundle without its header and proof, the seal puts
them back with the proof naming the head.  CPU only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.ingest.boundary_stream import (
    HEAD_SCHEMA, SEAL_ONLY_PROOF_KEYS, SEGMENT_SCHEMA, BoundaryStreamError,
    head_sha256, segment_marker_path,
)
from test_prepared_single_domain_forecast import (
    _bind_synthetic_preflight_geometry, _canonical, _prepared_fixture,
    _sha256, load_experiment, runner,
)


def _unseal(fixture):
    """The fixture as a chained preparation publishes it at its head."""

    bundle = Path(fixture.domain_bundle)
    header_path = bundle / "prepared-cache" / "header.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    proof = json.loads(Path(fixture.proof).read_text(encoding="utf-8"))
    intervals = header["metadata"]["lbc"]["intervals"]
    cache = {
        "directory": "prepared-cache",
        "identity": header["identity"],
        "metadata": {key: value for key, value in header["metadata"].items()
                     if key not in {"lbc", "setup_fingerprint"}},
        "arrays": header["arrays"],
        "payload_bytes": header["payload_bytes"],
        "lbc": {
            **{key: header["metadata"]["lbc"][key] for key in (
                "spec_bdy_width", "spec_zone", "relax_zone")},
            "schedule": [[row["start_seconds"], row["end_seconds"]]
                         for row in intervals],
            "fields": intervals[0]["fields"],
        },
        "setup_core_fingerprint": "0" * 64,
    }
    head = {
        "schema": HEAD_SCHEMA,
        "basis": {
            "schema": HEAD_SCHEMA,
            "cache": cache,
            "proof_head": {key: value for key, value in proof.items()
                           if key not in SEAL_ONLY_PROOF_KEYS},
            "input_manifest_sha256": _sha256(fixture.source_manifest),
        },
        "created_utc": "2026-09-28T00:00:00+00:00",
        "decision": {"chained": True},
        "reservation": {},
    }
    head["head_sha256"] = head_sha256(head)
    stream = Path(fixture.prepared) / "boundary-stream"
    (stream / "segments").mkdir(parents=True)
    (stream / "head.json").write_text(json.dumps(head), encoding="utf-8")
    (stream / "producer.json").write_text(json.dumps({
        "updated_epoch": 4e9, "times_built": 1}), encoding="utf-8")
    held = header_path.read_bytes()
    header_path.unlink()
    Path(fixture.proof).unlink()
    return SimpleNamespace(head=head, header=held, proof=proof,
                           intervals=intervals)


def _seal(fixture, chain, *, named_head=None):
    bundle = Path(fixture.domain_bundle)
    for index, row in enumerate(chain.intervals):
        segment_marker_path(fixture.prepared, index).write_text(json.dumps({
            "schema": SEGMENT_SCHEMA,
            "head_sha256": chain.head["head_sha256"],
            "index": index,
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "fields": row["fields"],
            "arrays": {},
            "payload_bytes": 0,
            "prefix": {},
        }), encoding="utf-8")
    (bundle / "prepared-cache" / "header.json").write_bytes(chain.header)
    proof = dict(chain.proof)
    proof["boundary_stream"] = {
        "head_sha256": named_head or chain.head["head_sha256"]}
    if "proof_content_sha256" in proof:
        # The writer digests the proof with the seal's keys in it.
        content = {key: value for key, value in proof.items()
                   if key != "proof_content_sha256"}
        proof["proof_content_sha256"] = hashlib.sha256(
            _canonical(content).encode("utf-8")).hexdigest()
    Path(fixture.proof).write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _head_preflight(fixture, head_digest):
    return runner.preflight_prepared_forecast(
        source=fixture.source, prepared_root=fixture.prepared,
        prepared_head_sha256=head_digest,
        source_manifest_sha256=_sha256(fixture.source_manifest),
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds,
        history_interval_seconds=load_experiment(
            fixture.experiment).root.history_interval_s)


@pytest.mark.parametrize("source", ["gfs", "era5", "20crv3"])
def test_a_head_binding_starts_before_the_seal_and_binds_at_it(
        tmp_path, monkeypatch, source):
    fixture = _prepared_fixture(tmp_path, source)
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    chain = _unseal(fixture)

    inputs = _head_preflight(fixture, chain.head["head_sha256"])
    assert inputs.stream_head["head_sha256"] == chain.head["head_sha256"]
    assert inputs.cache_reader.content_sha256 is None
    assert "proof" not in inputs.file_sha256
    assert "prepared_head" in inputs.file_sha256
    assert inputs.export_source_receipt is None

    _seal(fixture, chain)
    sealed = runner._seal_streamed_inputs(inputs)
    assert sealed.cache_reader.content_sha256 == fixture.content_sha256
    assert sealed.file_sha256["proof"] == _sha256(fixture.proof)
    assert sealed.cache_identity == inputs.cache_identity
    assert sealed.export_source_receipt is not None
    runner._verify_inputs_unchanged(sealed)


def test_a_seal_under_another_head_is_refused(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "gfs")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    chain = _unseal(fixture)
    inputs = _head_preflight(fixture, chain.head["head_sha256"])
    _seal(fixture, chain, named_head="e" * 64)
    with pytest.raises(BoundaryStreamError, match="not the pinned head"):
        runner._seal_streamed_inputs(inputs)


def test_a_head_pin_that_is_not_this_head_is_refused(tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "gfs")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    _unseal(fixture)
    with pytest.raises(BoundaryStreamError, match="pinned head"):
        _head_preflight(fixture, "d" * 64)


def test_a_sealed_chained_bundle_still_binds_by_its_proof(
        tmp_path, monkeypatch):
    fixture = _prepared_fixture(tmp_path, "gfs")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    chain = _unseal(fixture)
    _seal(fixture, chain)
    inputs = runner.preflight_prepared_forecast(
        source="gfs", prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)
    assert inputs.stream_head is None
    assert inputs.cache_reader.content_sha256 == fixture.content_sha256


@pytest.mark.parametrize("head, proof, content, refused", [
    ("a" * 64, None, None, False),
    (None, "b" * 64, "c" * 64, False),
    ("a" * 64, "b" * 64, None, True),
    (None, "b" * 64, None, True),
    (None, None, None, True),
])
def test_a_forecast_binds_exactly_one_preparation(head, proof, content,
                                                   refused):
    args = SimpleNamespace(prepared_head_sha256=head, proof_sha256=proof,
                           prepared_content_sha256=content)
    assert (runner._preparation_binding_refusal(args) is not None) is refused


def test_go_binds_a_chained_head_and_a_sealed_proof_differently(tmp_path):
    from gpuwm.go_cli import forecast_command

    plan = {"runner": "gpuwm.prepared_single_domain_forecast",
            "source": "gfs", "prepared": tmp_path / "prepared",
            "authority": tmp_path / "authority", "run": tmp_path / "run",
            "config": tmp_path / "c.toml"}
    head = forecast_command(plan, {"prepared_head": "a" * 64,
                                   "source_manifest": "b" * 64})
    sealed = forecast_command(plan, {"proof": "c" * 64,
                                     "source_manifest": "b" * 64,
                                     "prepared_content": "d" * 64})
    assert head[head.index("--prepared-head-sha256") + 1] == "a" * 64
    assert "--proof-sha256" not in head
    assert head[head.index("--source-manifest-sha256") + 1] == "b" * 64
    assert "--prepared-head-sha256" not in sealed
    assert sealed[sealed.index("--proof-sha256") + 1] == "c" * 64


def test_a_chained_run_checkpoint_identity_is_the_same_under_either_binding(
        tmp_path, monkeypatch):
    """A checkpoint of a chained run resumes whether the resume binds the
    head or the sealed proof that names it; an unchained tree keeps the
    identity it always had."""

    fixture = _prepared_fixture(tmp_path, "gfs")
    _bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    legacy = runner._single_checkpoint_identity(
        runner.preflight_prepared_forecast(
            source="gfs", prepared_root=fixture.prepared,
            proof_sha256=_sha256(fixture.proof),
            source_manifest_sha256=_sha256(fixture.source_manifest),
            prepared_content_sha256=fixture.content_sha256,
            experiment_config=fixture.experiment, wps_namelist=fixture.wps,
            physics_profile=runner.PHYSICS_PROFILE,
            run_seconds=fixture.run_seconds, history_interval_seconds=3600),
        {"runtime": "x"})
    assert legacy["prepared_content_sha256"] == fixture.content_sha256
    chain = _unseal(fixture)
    head_inputs = _head_preflight(fixture, chain.head["head_sha256"])
    _seal(fixture, chain)
    sealed_inputs = runner.preflight_prepared_forecast(
        source="gfs", prepared_root=fixture.prepared,
        proof_sha256=_sha256(fixture.proof),
        source_manifest_sha256=_sha256(fixture.source_manifest),
        prepared_content_sha256=fixture.content_sha256,
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        physics_profile=runner.PHYSICS_PROFILE,
        run_seconds=fixture.run_seconds, history_interval_seconds=3600)
    at_head = runner._single_checkpoint_identity(head_inputs, {"runtime": "x"})
    at_seal = runner._single_checkpoint_identity(sealed_inputs,
                                                 {"runtime": "x"})
    assert at_head == at_seal
    assert at_head["prepared_head_sha256"] == chain.head["head_sha256"]
