"""CPU authority protocol checks, separate from native forecast qualification."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble.posted_preparation import (NativeSnapshotAuthority,
    NativeSnapshotContext, PostedMemberPreparation, posted_preparation_scope,
    replace_current_native_snapshot, _publish, _snapshot_words,
    bind_current_prepared_identity, bind_current_source_identity,
    validate_posted_member_head, validate_posted_member_segment)

START = datetime(2024, 5, 25, 18, tzinfo=timezone.utc)

class _Trajectory:
    def __init__(self, member, identity):
        self.source, self.cycle, self.member, self.identity = "source", START, member, identity

class _Store:
    instances = {}
    def __init__(self, root):
        self.__dict__.update(self.instances[Path(root).resolve()].__dict__)
    def read(self, index):
        assert index == 0
        return self.snapshot

class _Stream:
    def __init__(self, trajectory, snapshots, grid, static, source, contract, plan):
        self.trajectory, self.snapshots = trajectory, snapshots
        self.head = {"grid": grid, "source": {**source, "static_identity": static},
                     "field_contract": contract, "input_plan_sha256": plan}
        self.head_sha256 = hashlib.sha256(json.dumps(self.head, sort_keys=True).encode()).hexdigest()
        self.marker_suffix = "a"
    def require(self, valid_time):
        if valid_time not in self.snapshots:
            raise RuntimeError("ordinary source is behind")
        snapshot = self.snapshots[valid_time]
        marker = {"valid_time": valid_time.isoformat(), "store_sha256": "b" * 64,
                  "revision": self.marker_suffix}
        return SimpleNamespace(read=lambda index: snapshot), {
            "head_sha256": self.head_sha256,
            "marker_sha256": hashlib.sha256((json.dumps(marker,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest(),
            "marker": marker}
    def bracket_times(self, valid_time):
        return (valid_time,)

class _Provider:
    def __init__(self, recipe, stream, snapshots):
        self.recipe, self.streams, self.snapshots = recipe, {stream.trajectory.identity: stream}, snapshots
        self.calls = []
    def prepare(self, instant, outputs, *, work_root):
        self.calls.append(instant)
        _, evidence = next(iter(self.streams.values())).require(instant)
        result = {}
        for member_id, path in outputs.items():
            path = Path(path)
            path.mkdir(parents=True)
            store = SimpleNamespace(snapshot=self.snapshots[instant], document={"grid": {"shape": [1, 2]},
                "source": deepcopy(next(iter(self.streams.values())).head["source"]),
                "field_contract": {"units": "K"},
                "frames": [{"valid_time": instant.isoformat()}]}, field_contract={"units": "K"},
                times=(instant.replace(tzinfo=None),), manifest_path=path / "physical-store.json")
            store.manifest_sha256 = _publish(store.manifest_path, store.document)
            _Store.instances[path.resolve()] = store
            member = next(item for item in self.recipe.members if item.index == member_id)
            receipt = {"member_index": member_id, "member_seed": member.seed,
                "recipe_sha256": self.recipe.sha256, "trajectory": {"source": member.trajectory.source,
                    "cycle": member.trajectory.cycle.isoformat(), "member": member.trajectory.member,
                    "identity": member.trajectory.identity},
                "valid_time": instant.isoformat(), "store_sha256": store.manifest_sha256,
                "sources": {member.trajectory.identity: [evidence]}}
            _publish(path / "posted-physical-receipt.json", receipt)
            result[member_id] = {"path": str(store.manifest_path), "sha256": store.manifest_sha256,
                                 "receipt": receipt}
        return result

@pytest.fixture
def preparation(tmp_path):
    trajectory = _Trajectory("member", "c" * 64)
    member = SimpleNamespace(index=17, seed=2**63 + 1, trajectory=trajectory)
    recipe = SimpleNamespace(members=(member,), start=START, end=START + timedelta(hours=1),
                             kind="input-ensemble", sha256="d" * 64)
    snapshots = {START: SimpleNamespace(valid_time=START.replace(tzinfo=None),
        fields={"TT": np.array([[275., 280.]], dtype=np.float32)}, levels_hpa=np.array([850.])),
        recipe.end: SimpleNamespace(valid_time=recipe.end.replace(tzinfo=None),
        fields={"TT": np.array([[276., 281.]], dtype=np.float32)}, levels_hpa=np.array([850.]))}
    grid, static, source, contract, plan = {"shape": [1, 2]}, {"words": "e" * 64}, {"manifest": "f" * 64}, {"units": "K"}, "1" * 64
    stream = _Stream(trajectory, snapshots, grid, static, source, contract, plan)
    provider = _Provider(recipe, stream, snapshots)
    authority = NativeSnapshotAuthority(source, contract, plan)
    owner = PostedMemberPreparation(provider, member_id=17, valid_times=(START, recipe.end),
        output_root=tmp_path / "member", authority_resolver=lambda context: authority,
        _reader=_Store, _geometry=lambda grid, cfg: {"shape": [1, 2]},
        _statics=lambda fields, grid, attrs: {"words": "e" * 64})
    def call(instant):
        return owner.replace_native(NativeSnapshotContext(snapshots[instant], None, None,
            None, None, {}, 1))
    return owner, provider, stream, snapshots, authority, call

def test_no_scope_returns_original_without_touching_native_objects():
    token = object()
    assert replace_current_native_snapshot(token, grid=None, cfg=None, static_fields=None) is token

def test_initial_frame_finishes_without_future_ready_or_seal(preparation):
    owner, provider, stream, snapshots, _, call = preparation
    future = snapshots.pop(owner.times[-1])
    assert call(START) is snapshots[START]
    assert owner.receipt()["ready_knots"] == [0]
    assert owner.receipt()["pending_knots"] == [1]
    assert not owner.receipt()["sealed"]
    assert owner.bind_head()["head"]["member_id"] == 17
    assert owner.bind_head()["head"]["seed"] == 2**63 + 1
    with pytest.raises(ValueError, match="both actual"):
        owner.bind_segment(0)
    with pytest.raises(ValueError, match="missing"):
        owner.seal()
    snapshots[owner.times[-1]] = future
    call(owner.times[-1])
    assert len(owner.bind_segment(0)["frames"]) == 2
    owner.seal()
    assert owner.receipt()["sealed"] and not owner.receipt()["pending_knots"]

def test_duplicate_native_read_is_idempotent_and_preserves_original_index(preparation):
    owner, provider, _, _, _, call = preparation
    call(START)
    before = (owner.root / "ready/00000.json").read_bytes()
    call(START)
    assert provider.calls == [START]
    assert (owner.root / "ready/00000.json").read_bytes() == before

@pytest.mark.parametrize("field", ["source", "units", "plan", "grid", "static"])
def test_actual_native_authority_drift_is_rejected_before_provider(preparation, field):
    owner, provider, stream, _, _, call = preparation
    key = {"source": "source", "units": "field_contract", "plan": "input_plan_sha256",
           "grid": "grid", "static": "source"}[field]
    if field == "static":
        stream.head[key]["static_identity"] = {"words": "2" * 64}
    else:
        stream.head[key] = "changed"
    with pytest.raises(ValueError, match="actual native"):
        call(START)
    assert not provider.calls

def test_native_base_byte_drift_rejected_before_recenter_or_initialize(preparation):
    owner, provider, stream, snapshots, _, call = preparation
    changed = deepcopy(snapshots[START])
    changed.fields["TT"][0, 0] = np.nextafter(changed.fields["TT"][0, 0], np.float32(np.inf))
    stream.snapshots = {**snapshots, START: changed}
    with pytest.raises(ValueError, match="physical bytes"):
        call(START)
    assert not provider.calls

def test_consumed_marker_changes_are_rejected(preparation):
    owner, _, stream, _, _, call = preparation
    call(START)
    stream.marker_suffix = "2"
    with pytest.raises(ValueError, match="marker changed"):
        call(START)

def test_member_head_mutation_is_rejected(preparation):
    owner, _, _, _, _, call = preparation
    call(START)
    owner.owner.head["seed"] += 1
    with pytest.raises(ValueError, match="head changed"):
        owner.bind_head()

def test_metadata_branch_words_are_part_of_native_identity():
    a = SimpleNamespace(valid_time=START, levels_hpa=np.array([850.]), fields={"TT": np.array([270.], dtype=np.float32)},
                        specific_humidity_authority=False)
    b = deepcopy(a)
    b.specific_humidity_authority = True
    assert _snapshot_words(a) != _snapshot_words(b)

def test_active_scope_passes_actual_native_context(preparation):
    owner, _, _, snapshots, _, _ = preparation
    with posted_preparation_scope({1: owner}):
        assert replace_current_native_snapshot(snapshots[START], grid=None, cfg=None,
            static_fields=None, domain_id=1) is snapshots[START]


def test_portable_head_and_segment_bind_actual_consumed_records(preparation):
    owner, _, _, _, authority, call = preparation
    call(START)
    with posted_preparation_scope(owner):
        source = bind_current_source_identity(authority.source_identity)
        identity = bind_current_prepared_identity({"source_identity": source})
    head = source["ensemble_posted_member_input"]
    validate_posted_member_head(head, identity=identity)
    call(owner.times[-1])
    validate_posted_member_segment(owner.bind_segment(0), head, index=0)


def test_portable_authority_rejects_changed_member_seed(preparation):
    owner, _, _, _, _, call = preparation
    call(START)
    head = deepcopy(owner.bind_head())
    head["initial"]["provider_receipt"]["member_seed"] += 1
    with pytest.raises(ValueError, match="digest"):
        validate_posted_member_head(head)


def test_segment_requires_exact_head_and_native_time_positions(preparation):
    owner, _, _, _, _, call = preparation
    call(START)
    head = owner.bind_head()
    call(owner.times[-1])
    segment = owner.bind_segment(0)
    segment["frames"].reverse()
    with pytest.raises(ValueError, match="head/time/digest"):
        validate_posted_member_segment(segment, head, index=0)


def test_inactive_prepared_identity_retains_original_object():
    original = {"source_identity": {"adapter": "native"}}
    assert bind_current_prepared_identity(original) is original
    assert bind_current_source_identity(original["source_identity"]) is original["source_identity"]
def test_native_source_consumers_are_exclusive_before_snapshot_replacement():
    from gpuwm.ensemble.posted_preparation import (
        posted_preparation_scope, require_exclusive_native_consumer)
    require_exclusive_native_consumer(physical_input_store="complete-store")
    require_exclusive_native_consumer(physical_input_provider="posted-provider")
    with posted_preparation_scope(object()):
        require_exclusive_native_consumer()
        for option in ("physical_input_store", "physical_input_provider"):
            with pytest.raises(ValueError, match="one replacement authority"):
                require_exclusive_native_consumer(**{option: "selected"})
