"""The cycling ensemble's on-disk generation: format, identity, ring.

Pure tests: no network, no GPU, no model.  The arrays here are tiny
stand-ins for the driver's host snapshots -- what is under test is the
boundary contract (what is written, what is refused, what resumes),
not the physics that fills it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tools.da_ensemble_state import (
    CONTROL, MIN_SLOTS, SCHEMA, EnsembleIdentity, EnsembleStateError,
    latest_generation, manifest_path, pending_path, read_generation,
    read_manifest, slot_dir, snapshot_path, trajectory_key,
    trajectory_names, validate_resume, write_generation)


def identity(members: int = 2, **overrides) -> EnsembleIdentity:
    base = dict(members=members, nx=8, ny=6, nz=4, dt_s=15.0,
                mp_physics=6, physics_profile="profile-under-test-v1",
                prepared_content_sha256="c" * 64)
    base.update(overrides)
    return EnsembleIdentity(**base)


def variant(field: str, value) -> EnsembleIdentity:
    """The reference identity with exactly one field changed."""

    base = dict(members=2, nx=8, ny=6, nz=4, dt_s=15.0, mp_physics=6,
                physics_profile="profile-under-test-v1",
                prepared_content_sha256="c" * 64)
    base[field] = value
    return EnsembleIdentity(**base)


def snapshots_for(members: int, *, fields=("u", "v")) -> dict:
    out = {}
    for index, name in enumerate(trajectory_names(members)):
        out[name] = {f: np.full((4, 6, 8), index + 1, np.float32)
                     for f in fields}
    return out


# ---------------------------------------------------------------------------
# trajectory naming
# ---------------------------------------------------------------------------
class TestTrajectoryNaming:
    def test_control_keeps_its_name(self):
        assert trajectory_key(CONTROL) == "control"

    def test_members_are_zero_padded(self):
        assert trajectory_key(0) == "m000"
        assert trajectory_key(7) == "m007"
        assert trajectory_key(123) == "m123"

    def test_negative_member_refused(self):
        with pytest.raises(EnsembleStateError, match="negative"):
            trajectory_key(-1)

    def test_names_are_control_then_members(self):
        assert trajectory_names(3) == [CONTROL, 0, 1, 2]

    def test_empty_ensemble_refused(self):
        with pytest.raises(EnsembleStateError, match="at least one"):
            trajectory_names(0)


# ---------------------------------------------------------------------------
# the slot ring
# ---------------------------------------------------------------------------
class TestSlotRing:
    def test_alternates_so_a_write_never_touches_the_resume(self, tmp_path):
        assert slot_dir(tmp_path, 0) != slot_dir(tmp_path, 1)
        assert slot_dir(tmp_path, 0) == slot_dir(tmp_path, 2)

    def test_ring_of_one_is_refused(self, tmp_path):
        with pytest.raises(EnsembleStateError, match="at least 2"):
            slot_dir(tmp_path, 0, slots=1)

    def test_wider_rings_are_allowed(self, tmp_path):
        seen = {slot_dir(tmp_path, g, slots=3) for g in range(6)}
        assert len(seen) == 3

    def test_negative_generation_refused(self, tmp_path):
        with pytest.raises(EnsembleStateError, match="start at 0"):
            slot_dir(tmp_path, -1)

    def test_minimum_is_two(self):
        assert MIN_SLOTS == 2


# ---------------------------------------------------------------------------
# writing and reading one generation
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_snapshots_and_pending_survive_exactly(self, tmp_path):
        ident = identity(2)
        snaps = snapshots_for(2)
        pend = {CONTROL: None,
                0: {"u": np.full((4, 6, 8), 0.5, np.float32)},
                1: {"u": np.full((4, 6, 8), -0.5, np.float32)}}
        write_generation(tmp_path, identity=ident, elapsed_seconds=900.0,
                         leg_number=3, snapshots=snaps, pending=pend)
        back_snaps, back_pend, manifest = read_generation(tmp_path, ident)
        assert manifest["schema"] == SCHEMA
        assert manifest["elapsed_seconds"] == 900.0
        assert manifest["leg_number"] == 3
        for name in trajectory_names(2):
            for field, values in snaps[name].items():
                assert np.array_equal(back_snaps[name][field], values)
        assert back_pend[CONTROL] is None
        assert np.array_equal(back_pend[0]["u"], pend[0]["u"])

    def test_control_without_pending_is_recorded_as_absent(self, tmp_path):
        ident = identity(1)
        write_generation(tmp_path, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, snapshots=snapshots_for(1),
                         pending={CONTROL: None, 0: None})
        manifest = read_manifest(tmp_path)
        assert "pending" not in manifest["trajectories"]["control"]
        assert "pending" not in manifest["trajectories"]["m000"]
        assert not pending_path(tmp_path, 0).exists()

    def test_every_trajectory_has_a_snapshot_file(self, tmp_path):
        ident = identity(2)
        write_generation(tmp_path, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, snapshots=snapshots_for(2),
                         pending={})
        for name in trajectory_names(2):
            assert snapshot_path(tmp_path, name).is_file()

    def test_missing_trajectory_is_refused_before_anything_is_written(
            self, tmp_path):
        ident = identity(3)
        snaps = snapshots_for(3)
        del snaps[2]
        with pytest.raises(EnsembleStateError, match="missing"):
            write_generation(tmp_path, identity=ident,
                             elapsed_seconds=0.0, leg_number=0,
                             snapshots=snaps, pending={})
        assert not manifest_path(tmp_path).exists()

    def test_snapshot_field_inventory_is_recorded(self, tmp_path):
        ident = identity(1)
        write_generation(
            tmp_path, identity=ident, elapsed_seconds=0.0, leg_number=0,
            snapshots=snapshots_for(1, fields=("u", "v", "qv")),
            pending={})
        entry = read_manifest(tmp_path)["trajectories"]["m000"]
        assert entry["snapshot_fields"] == ["qv", "u", "v"]

    def test_valid_time_and_note_ride_along(self, tmp_path):
        ident = identity(1)
        write_generation(tmp_path, identity=ident, elapsed_seconds=60.0,
                         leg_number=1, snapshots=snapshots_for(1),
                         pending={}, valid_time="2026-08-05T06:00:00Z",
                         note="carried")
        manifest = read_manifest(tmp_path)
        assert manifest["valid_time"] == "2026-08-05T06:00:00Z"
        assert manifest["note"] == "carried"


# ---------------------------------------------------------------------------
# identity: the whole point of writing it down
# ---------------------------------------------------------------------------
class TestIdentity:
    @pytest.mark.parametrize("field,value", [
        ("members", 3), ("nx", 9), ("ny", 7), ("nz", 5),
        ("dt_s", 10.0), ("mp_physics", 8),
        ("physics_profile", "some-other-profile-v1"),
        ("prepared_content_sha256", "d" * 64)])
    def test_every_field_is_checked(self, tmp_path, field, value):
        ident = identity(2)
        write_generation(tmp_path, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, snapshots=snapshots_for(2),
                         pending={})
        with pytest.raises(EnsembleStateError, match=field):
            read_generation(tmp_path, variant(field, value))

    def test_all_differences_are_reported_not_just_the_first(
            self, tmp_path):
        ident = identity(2)
        write_generation(tmp_path, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, snapshots=snapshots_for(2),
                         pending={})
        other = identity(2, nx=99, nz=99)
        with pytest.raises(EnsembleStateError) as caught:
            read_generation(tmp_path, other)
        assert "nx" in str(caught.value) and "nz" in str(caught.value)

    def test_matching_identity_passes(self):
        ident = identity(2)
        validate_resume({"identity": ident.to_payload()}, ident)

    def test_truncated_identity_is_refused(self):
        with pytest.raises(EnsembleStateError, match="missing"):
            validate_resume({"identity": {"members": 2}}, identity(2))


# ---------------------------------------------------------------------------
# an unfinished generation is not a generation
# ---------------------------------------------------------------------------
class TestCompletionMarker:
    def test_no_manifest_means_no_generation(self, tmp_path):
        (tmp_path / "snap_control.npz").write_bytes(b"not a npz")
        with pytest.raises(EnsembleStateError, match="never finished"):
            read_manifest(tmp_path)

    def test_foreign_schema_refused(self, tmp_path):
        manifest_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        manifest_path(tmp_path).write_text(
            json.dumps({"schema": "something.else.v1"}), encoding="utf-8")
        with pytest.raises(EnsembleStateError, match="schema"):
            read_manifest(tmp_path)

    def test_rewriting_a_slot_clears_the_old_marker_first(self, tmp_path):
        ident = identity(1)
        write_generation(tmp_path, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, snapshots=snapshots_for(1),
                         pending={})
        write_generation(tmp_path, identity=ident, elapsed_seconds=900.0,
                         leg_number=1, snapshots=snapshots_for(1),
                         pending={})
        assert read_manifest(tmp_path)["leg_number"] == 1
        assert (tmp_path / "ensemble-manifest.json.superseded").is_file()


# ---------------------------------------------------------------------------
# picking up where a daemon left off
# ---------------------------------------------------------------------------
class TestLatestGeneration:
    def test_none_when_nothing_written(self, tmp_path):
        assert latest_generation(tmp_path) is None
        assert latest_generation(tmp_path / "absent") is None

    def test_picks_the_furthest_advanced_not_the_last_sorted(
            self, tmp_path):
        ident = identity(1)
        write_generation(slot_dir(tmp_path, 0), identity=ident,
                         elapsed_seconds=1800.0, leg_number=2,
                         snapshots=snapshots_for(1), pending={})
        write_generation(slot_dir(tmp_path, 1), identity=ident,
                         elapsed_seconds=900.0, leg_number=1,
                         snapshots=snapshots_for(1), pending={})
        found = latest_generation(tmp_path)
        assert found is not None
        directory, manifest = found
        assert directory == slot_dir(tmp_path, 0)
        assert manifest["leg_number"] == 2

    def test_a_torn_slot_is_skipped_not_fatal(self, tmp_path):
        ident = identity(1)
        write_generation(slot_dir(tmp_path, 0), identity=ident,
                         elapsed_seconds=900.0, leg_number=1,
                         snapshots=snapshots_for(1), pending={})
        torn = slot_dir(tmp_path, 1)
        torn.mkdir(parents=True, exist_ok=True)
        (torn / "snap_control.npz").write_bytes(b"half a write")
        directory, manifest = latest_generation(tmp_path)
        assert directory == slot_dir(tmp_path, 0)
        assert manifest["leg_number"] == 1
