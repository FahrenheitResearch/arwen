"""Source bank isolation, failure recovery and C1 decode counts."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import mapped_composition as composition


def _arguments(tmp_path):
    paths = {}
    for name in ("mapping", "composition", "manifest", "primary", "engine"):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = path
    return dict(engine=paths["engine"], mapping_path=paths["mapping"],
        composition_path=paths["composition"], manifest_path=paths["manifest"],
        manifest_sha256="manifest", primary=(paths["primary"],), supplements={},
        provenance={}, contributing={}, contract={}, bindings={}, terrain_spec=None,
        decoders={}, snapshots={}, before={str(path): name for name, path in paths.items()},
        member=None, member_identity=None, scratch_destination=tmp_path,
        atmospheric_grids=(), workers=1, lead_batch=False, memory_budget_bytes=None)


def _writer(monkeypatch):
    calls = []
    def compose(**arguments):
        directory = arguments["_shared_directory"]
        if not arguments["_shared_hit"]:
            calls.append(directory)
            directory.mkdir(parents=True)
            for name in ("frames.json", "composition.json"):
                (directory / name).write_text("{}")
        return SimpleNamespace(native_decode_memory={"provider": "native"}, directory=directory)
    monkeypatch.setattr(composition, "_compose_through_engine", compose)
    return calls


def test_two_member_tails_share_one_native_composition_and_one_decode_event(tmp_path, monkeypatch):
    calls = _writer(monkeypatch)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("GPUWM_TIME_LEDGER", str(ledger))
    args = _arguments(tmp_path)
    first = composition._shared_engine_composition(args, ("source", tmp_path / "bank"))
    second = composition._shared_engine_composition(args, ("source", tmp_path / "bank"))
    assert first.directory == second.directory
    assert len(calls) == 1
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert sum(row["stage"] == "prep.decode" and row["phase"] == "begin" for row in rows) == 1
    assert all(row["v"] == 1 and row["product"] == "ensemble" and
               row["attrs"]["source_key"] == "source" for row in rows)
    assert any(row["phase"] == "mark" and row["attrs"]["cache_hit"] for row in rows)


@pytest.mark.parametrize("mutation", ["source", "primary", "engine", "member", "lead_batch", "budget"])
def test_different_operands_never_collide(tmp_path, monkeypatch, mutation):
    calls = _writer(monkeypatch)
    args = _arguments(tmp_path)
    binding = ("source", tmp_path / "bank")
    first = composition._shared_engine_composition(args, binding)
    changed = deepcopy(args)
    if mutation == "source":
        binding = ("other-source", binding[1])
    elif mutation == "primary":
        changed["before"][str(changed["primary"][0])] = "other-bytes"
    elif mutation == "engine":
        changed["engine"].write_text("different native decoder")
    elif mutation == "member":
        changed["member"], changed["member_identity"] = "p01", "other-member-policy"
    elif mutation == "budget":
        changed["memory_budget_bytes"] = 4096
    else:
        changed["lead_batch"] = True
    second = composition._shared_engine_composition(changed, binding)
    assert second.directory != first.directory
    assert len(calls) == 2


def test_equivalent_verified_manifest_locations_do_not_force_another_decode(tmp_path, monkeypatch):
    calls = _writer(monkeypatch)
    args = _arguments(tmp_path)
    binding = ("source", tmp_path / "bank")
    first = composition._shared_engine_composition(args, binding)
    changed = {**args, "manifest_path": tmp_path / "other" / "inputs.json",
               "manifest_sha256": "different-relative-paths"}
    second = composition._shared_engine_composition(changed, binding)
    assert second.directory == first.directory
    assert len(calls) == 1


def test_parallel_members_have_one_writer_and_independent_readers(tmp_path, monkeypatch):
    calls = _writer(monkeypatch)
    args = _arguments(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(composition._shared_engine_composition, args,
                   ("source", tmp_path / "bank")) for _ in range(4)]
        bundles = [future.result() for future in futures]
    assert len(calls) == 1
    assert len({bundle.directory for bundle in bundles}) == 1
    assert len({id(bundle) for bundle in bundles}) == 4


def test_changed_sealed_metadata_refuses_instead_of_using_unbound_values(tmp_path, monkeypatch):
    calls = _writer(monkeypatch)
    args = _arguments(tmp_path)
    binding = ("source", tmp_path / "bank")
    bundle = composition._shared_engine_composition(args, binding)
    (bundle.directory / "frames.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="metadata changed after seal"):
        composition._shared_engine_composition(args, binding)
    assert len(calls) == 1


def test_partial_writer_is_never_published_and_retry_composes(tmp_path, monkeypatch):
    args = _arguments(tmp_path)
    def fail(**arguments):
        arguments["_shared_directory"].mkdir(parents=True)
        (arguments["_shared_directory"] / "partial").write_text("partial")
        raise RuntimeError("native writer interrupted")
    monkeypatch.setattr(composition, "_compose_through_engine", fail)
    binding = ("source", tmp_path / "bank")
    with pytest.raises(RuntimeError, match="interrupted"):
        composition._shared_engine_composition(args, binding)
    assert not list(binding[1].glob("*/READY.json"))
    calls = _writer(monkeypatch)
    composition._shared_engine_composition(args, binding)
    assert len(calls) == 1
    assert not list(binding[1].glob("*/composed/partial"))


def test_context_binding_is_worker_local_and_restored_after_failure(tmp_path):
    with composition.shared_composition_source("first", tmp_path):
        with pytest.raises(RuntimeError):
            with composition.shared_composition_source("second", tmp_path):
                raise RuntimeError("stop")
        assert composition._SHARED_COMPOSITION.get() == ("first", tmp_path.resolve())
        with ThreadPoolExecutor(max_workers=1) as workers:
            assert workers.submit(composition._SHARED_COMPOSITION.get).result() is None
    assert composition._SHARED_COMPOSITION.get() is None
