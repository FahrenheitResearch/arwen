from contextlib import nullcontext
from datetime import datetime
import json
from types import SimpleNamespace

import pytest

from gpuwm.ensemble import restart_roster
from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.packing import CardBudget
from gpuwm.ensemble.production import PreparedEnsembleSession
from gpuwm.ensemble.runtime_context import current_capture


class Collector:
    def __init__(self, root):
        self.root, self.rows = root, []

    def submit(self, **row):
        self.rows.append(row["member_id"])

    def save_resume(self):
        restart_roster.atomic_json(self.root / ".ensemble-resume" / "collector.json",
            {"rows": self.rows, "files": []})

    def restore_resume(self):
        self.rows = json.loads((self.root / ".ensemble-resume" / "collector.json").read_text())["rows"]

    def finish_run(self):
        return {"members": self.rows}

    def require_complete(self):
        assert self.rows == [0, 1, 2]


def prepared(member):
    experiment = SimpleNamespace(run_seconds=60, restart_interval_s=30,
        start_time=datetime(2024, 1, 1), root=SimpleNamespace(run=SimpleNamespace()))
    return SimpleNamespace(experiment=experiment, member=member, source="fixture",
        prepared_head_sha256=f"member-head-{member}", boundary_interval_seconds=3600)


def session(root, collector, roster=None):
    return PreparedEnsembleSession({"members": 3, "base_seed": 17}, output_directory=root,
        input_provider=lambda **kwargs: prepared(kwargs["member_id"]), restart_roster=roster,
        collector=collector, cards=(CardBudget(0, 1000),),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)),
        device_scope=lambda _: nullcontext())


@pytest.fixture
def identity(monkeypatch):
    from gpuwm.core import model
    monkeypatch.setattr(model, "restart_identity_payload", lambda exp: {"dt": 3, "physics": 16})


def test_finished_members_are_retained_and_fresh_session_restores_the_original_roster(tmp_path, identity):
    first = Collector(tmp_path)
    def interrupted(inputs, **options):
        member = current_capture().member_id
        if member == 1:
            raise KeyboardInterrupt("reclaimed at member 1")
        first.submit(member_id=member)
        return {"status": "PASS", "member": member, "state_sha256": f"member-{member}-state"}
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, first).run_prepared(interrupted, prepared(0))
    document = json.loads((tmp_path / restart_roster.ROSTER).read_text())
    assert [row["status"] for row in document["members"]] == ["completed", "pending", "pending"]
    second, invoked = Collector(tmp_path), []
    def resumed(inputs, **options):
        member = current_capture().member_id
        invoked.append(member)
        second.submit(member_id=member)
        return {"status": "PASS", "member": member, "state_sha256": f"member-{member}-state"}
    report = session(tmp_path, second, tmp_path / restart_roster.ROSTER).run_prepared(resumed, prepared(0))
    assert report["status"] == "PASS"
    assert invoked == [1, 2]
    assert second.rows == [0, 1, 2]
    assert len(report["completed_member_results"]) == 3
    assert report["member_results"][0]["result"]["state_sha256"] == "member-0-state"
    assert report["ensemble_progress"]["members"][0]["elapsed_seconds"] == 60


def test_wrong_member_seed_and_tampered_retained_file_refuse_before_any_runner(tmp_path, identity):
    first = Collector(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, first).run_prepared(lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()), prepared(0))
    roster = tmp_path / restart_roster.ROSTER
    original = json.loads(roster.read_text())
    changed = json.loads(roster.read_text())
    changed["identities"]["members"][0]["seed"] += 1
    restart_roster.atomic_json(roster, changed)
    invoked = []
    with pytest.raises(ValueError, match="different member identities"):
        session(tmp_path, Collector(tmp_path), roster).run_prepared(lambda *args, **kwargs: invoked.append(1), prepared(0))
    assert not invoked
    restart_roster.atomic_json(roster, original)
    (tmp_path / ".ensemble-resume" / "collector.json").write_text("tampered")
    with pytest.raises(ValueError, match="hash-mismatched"):
        session(tmp_path, Collector(tmp_path), roster).run_prepared(lambda *args, **kwargs: invoked.append(1), prepared(0))
    assert not invoked


def test_checkpoint_is_passed_only_to_its_member_and_all_domains_are_hash_verified(tmp_path, identity, monkeypatch):
    from gpuwm.io import restart
    from gpuwm import supervisor
    first, checkpoint = Collector(tmp_path), tmp_path / "members" / "member-0001" / "gpuwmrst_d01_2024-01-01_00_00_30.npz"
    monkeypatch.setattr(restart, "read_restart_header", lambda path: {"elapsed_seconds": 30})
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", lambda path: path)
    def interrupted(inputs, **options):
        if current_capture().member_id == 1:
            checkpoint.write_bytes(b"checkpoint-member-1")
            options["progress_callback"](model_elapsed_seconds=30, outer_step=10, last_checkpoint=checkpoint)
            raise KeyboardInterrupt()
        first.submit(member_id=0)
        return {"status": "PASS"}
    with pytest.raises(KeyboardInterrupt):
        session(tmp_path, first).run_prepared(interrupted, prepared(0))
    roster = tmp_path / restart_roster.ROSTER
    document = json.loads(roster.read_text())
    assert document["members"][1]["status"] == "checkpoint"
    second, paths = Collector(tmp_path), {}
    def resumed(inputs, **options):
        member = current_capture().member_id
        paths[member] = options.get("restart")
        second.submit(member_id=member)
        return {"status": "PASS"}
    session(tmp_path, second, roster).run_prepared(resumed, prepared(0))
    assert paths == {1: checkpoint, 2: None}


def test_scalar_restart_refuses_instead_of_repeating_one_members_state(tmp_path, identity):
    with pytest.raises(ValueError, match="one checkpoint cannot identify every member"):
        session(tmp_path, Collector(tmp_path)).run_prepared(lambda *args, **kwargs: {}, prepared(0), restart=tmp_path / "one.npz")


def test_checkpointed_roster_uses_one_selected_card_to_keep_its_diagnostic_snapshot_coherent(tmp_path, identity):
    collector = Collector(tmp_path)
    run = session(tmp_path, collector)
    run.cards = (CardBudget(2, 1000), CardBudget(5, 1000))
    devices = []
    run.device_scope = lambda device: devices.append(device) or nullcontext()
    def runner(inputs, **options):
        collector.submit(member_id=current_capture().member_id)
        return {"status": "PASS"}
    report = run.run_prepared(runner, prepared(0))
    assert report["packing"]["waves"] == 3
    assert set(devices) == {2}


def test_real_rust_collector_snapshot_retains_endpoint_words_and_skips_delivered_output(tmp_path):
    import hashlib
    import numpy as np
    from gpuwm.ensemble.batch_products import FieldProducts
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector, NativeDiagnosticSpool
    start = datetime(2024, 1, 1)
    request = FieldProducts("wind10", "m s-1", (25.,))
    coords = np.zeros((2, 2), np.float32)
    geometry = hashlib.sha256(coords.tobytes() + coords.tobytes()).hexdigest()
    original = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch",
        start_time=start, requests=(request,), array_module=np)
    spool = NativeDiagnosticSpool(tmp_path, members=2, requests=(request,),
        latitude=coords, longitude=coords, renderer="rw_wrfbatch")
    pack = tmp_path / ".ensemble-diagnostics" / "retained-pack.nc"
    pack.parent.mkdir()
    pack.write_bytes(b"retained member diagnostics")
    valid = "2024-01-01_00:00:00"
    spool.frames[valid] = {"valid_time": valid, "domain": "d01", "status": "pending",
        "members_received": [0], "members_expected": 2, "packs": [{"path": pack, "member_ids": (0,),
            "sha256": hashlib.sha256(pack.read_bytes()).hexdigest(), "bytes": pack.stat().st_size}],
        "products": [], "maps": [], "available_fields": ["wind10"], "unavailable_fields": []}
    spool._manifest()
    original.spools[("d01", geometry)] = spool
    original.cohorts[(1, 0, valid)] = {geometry: {"members": {0}, "spool": spool}}
    key = (1, 0, 0, geometry)
    words = np.array([0, 0x80000000, 0x00000001, 0x3f800001], np.uint32).view(np.float32).reshape(2, 2)
    original.rain_history[key] = {"initial": words.copy(), 0: words.copy()}
    original.rain_initial_ticks[key] = 0
    original.output_ticks.add((key, 0))
    document = original.save_resume()
    assert pack.relative_to(tmp_path).as_posix() in document["files"]
    restored = HeadlineDiagnosticCollector(tmp_path, members=2, renderer="rw_wrfbatch",
        start_time=start, requests=(request,), array_module=np, resume=True)
    restored.restore_resume()
    assert restored.rain_history[key]["initial"].tobytes() == words.tobytes()
    assert restored.rain_history[key][0].tobytes() == words.tobytes()
    assert restored.spools[("d01", geometry)].frames[valid]["members_received"] == [0]
    assert restored.spools[("d01", geometry)].frames[valid]["packs"][0]["path"] == pack
    assert restored.submit(state=SimpleNamespace(physics=None), streamed=None,
        metadata={"XLAT": coords, "XLONG": coords}, refl_field=None,
        valid_time=start, grid_id=1, episode=0, member_id=0) is None
    assert restored._kernels == {}
    assert pack.read_bytes() == b"retained member diagnostics"
