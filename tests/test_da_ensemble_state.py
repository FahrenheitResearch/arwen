"""The cycling ensemble's on-disk generation: format, identity, ring.

Pure tests: no network, no GPU, no model.  The restart sets here are
tiny stand-ins with a real restart header and the writer's own member
naming -- what is under test is the boundary contract (what is copied,
what is refused, what resumes), not the physics that fills a set.
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pytest

from tools.da_ensemble_state import (
    CONTROL, MIN_SLOTS, RETIRED_SCHEMA, SCHEMA, EnsembleIdentity,
    EnsembleStateError, latest_generation, manifest_path,
    nested_trajectories, pending_path, read_generation, read_manifest,
    restart_dir, slot_dir, trajectory_key, trajectory_name,
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


_HEADER_KEY = "__gpuwm_restart_header__"


def write_fake_set(directory, *, domain_ids=(1,), fill: float = 1.0,
                   instant: str = "2026-01-01_00_15_00"):
    """A stand-in tree checkpoint set: real header, real member naming.

    One member per domain id, all sharing the instant and the set id in
    their names exactly as ``write_tree_restart`` names them, each with
    the header the restart owner reads the domain set from.  Returns the
    root member, which is what the driver hands the generation writer.
    """
    directory.mkdir(parents=True, exist_ok=True)
    set_id = uuid.uuid4().hex
    root = None
    for gid in domain_ids:
        header = {"format_version": 6, "domain_ids": sorted(domain_ids),
                  "grid_id": int(gid), "elapsed_seconds": 900.0}
        path = directory / f"gpuwmrst_d{int(gid):02d}_{instant}__{set_id}.npz"
        np.savez(path,
                 **{_HEADER_KEY: np.frombuffer(
                     json.dumps(header).encode("utf-8"), dtype=np.uint8),
                    "state/u": np.full((4, 6, 8), fill, np.float32)})
        if int(gid) == min(int(d) for d in domain_ids):
            root = path
    return root


def sets_for(tmp_path, members: int, *, nested=()) -> dict:
    out = {}
    for index, name in enumerate(trajectory_names(members)):
        ids = (1, 2) if name in nested else (1,)
        out[name] = write_fake_set(
            tmp_path / "stage" / trajectory_key(name), domain_ids=ids,
            fill=index + 1)
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

    def test_keys_round_trip_to_names(self):
        for name in trajectory_names(3):
            assert trajectory_name(trajectory_key(name)) == name
        with pytest.raises(EnsembleStateError, match="not a trajectory"):
            trajectory_name("member-3")


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
    def test_restart_sets_and_pending_survive_exactly(self, tmp_path):
        ident = identity(2)
        sets = sets_for(tmp_path, 2)
        pend = {CONTROL: None,
                0: {"u": np.full((4, 6, 8), 0.5, np.float32)},
                1: {"u": np.full((4, 6, 8), -0.5, np.float32)}}
        out = tmp_path / "gen"
        write_generation(out, identity=ident, elapsed_seconds=900.0,
                         leg_number=3, restarts=sets, pending=pend)
        back_sets, back_pend, manifest = read_generation(out, ident)
        assert manifest["schema"] == SCHEMA
        assert manifest["elapsed_seconds"] == 900.0
        assert manifest["leg_number"] == 3
        for name in trajectory_names(2):
            copied = back_sets[name]
            assert copied.is_file()
            assert copied.name == sets[name].name
            assert copied.parent == restart_dir(out, name)
            assert copied.read_bytes() == sets[name].read_bytes()
        assert back_pend[CONTROL] is None
        assert np.array_equal(back_pend[0]["u"], pend[0]["u"])

    def test_a_generation_survives_the_stage_consuming_its_sets(
            self, tmp_path):
        """Restart members are linked, not copied, when the filesystem
        allows: the generation must still hold every byte after the stage
        removes its own names, and a set is never shared with a later
        rewrite of the same slot."""
        import os
        import shutil

        ident = identity(3)
        sets = sets_for(tmp_path, 3)
        expected = {name: root.read_bytes() for name, root in sets.items()}
        out = tmp_path / "gen"
        write_generation(out, identity=ident, elapsed_seconds=900.0,
                         leg_number=1, restarts=sets,
                         pending={name: None for name in sets})
        linked = os.stat(sets[0]).st_ino == os.stat(
            restart_dir(out, 0) / sets[0].name).st_ino
        shutil.rmtree(tmp_path / "stage")
        back_sets, _, _ = read_generation(out, ident)
        for name, root in back_sets.items():
            assert root.read_bytes() == expected[name]
        # One filesystem here, so the restart bytes were linked, not copied.
        assert linked

    def test_control_without_pending_is_recorded_as_absent(self, tmp_path):
        ident = identity(1)
        out = tmp_path / "gen"
        write_generation(out, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, restarts=sets_for(tmp_path, 1),
                         pending={CONTROL: None, 0: None})
        manifest = read_manifest(out)
        assert "pending" not in manifest["trajectories"]["control"]
        assert "pending" not in manifest["trajectories"]["m000"]
        assert not pending_path(out, 0).exists()

    def test_every_trajectory_has_its_set_copied_whole(self, tmp_path):
        ident = identity(2)
        out = tmp_path / "gen"
        sets = sets_for(tmp_path, 2, nested=(CONTROL,))
        write_generation(out, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, restarts=sets, pending={})
        manifest = read_manifest(out)
        control = manifest["trajectories"]["control"]
        assert control["domain_ids"] == [1, 2]
        assert set(control["restart_members"]) == {"1", "2"}
        for member in control["restart_members"].values():
            assert (restart_dir(out, CONTROL) / member).is_file()
        assert manifest["trajectories"]["m000"]["domain_ids"] == [1]
        assert nested_trajectories(manifest) == [CONTROL]

    def test_missing_trajectory_is_refused_before_anything_is_written(
            self, tmp_path):
        ident = identity(3)
        sets = sets_for(tmp_path, 3)
        del sets[2]
        out = tmp_path / "gen"
        with pytest.raises(EnsembleStateError, match="missing"):
            write_generation(out, identity=ident,
                             elapsed_seconds=0.0, leg_number=0,
                             restarts=sets, pending={})
        assert not manifest_path(out).exists()

    def test_a_set_with_a_member_missing_is_refused_whole(self, tmp_path):
        ident = identity(1)
        sets = sets_for(tmp_path, 1, nested=(CONTROL,))
        out = tmp_path / "gen"
        write_generation(out, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, restarts=sets, pending={})
        manifest = read_manifest(out)
        gone = manifest["trajectories"]["control"]["restart_members"]["2"]
        (restart_dir(out, CONTROL) / gone).unlink()
        with pytest.raises(EnsembleStateError, match="not there"):
            read_generation(out, ident)

    def test_rewriting_a_slot_replaces_the_old_set(self, tmp_path):
        ident = identity(1)
        out = tmp_path / "gen"
        first = sets_for(tmp_path / "a", 1, nested=(CONTROL,))
        write_generation(out, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, restarts=first, pending={})
        second = sets_for(tmp_path / "b", 1)
        write_generation(out, identity=ident, elapsed_seconds=900.0,
                         leg_number=1, restarts=second,
                         pending={CONTROL: None, 0: None})
        kept = sorted(p.name for p in restart_dir(out, CONTROL).iterdir())
        assert kept == [second[CONTROL].name], kept

    def test_valid_time_and_note_ride_along(self, tmp_path):
        ident = identity(1)
        out = tmp_path / "gen"
        write_generation(out, identity=ident, elapsed_seconds=60.0,
                         leg_number=1, restarts=sets_for(tmp_path, 1),
                         pending={}, valid_time="2026-08-05T06:00:00Z",
                         note="carried")
        manifest = read_manifest(out)
        assert manifest["valid_time"] == "2026-08-05T06:00:00Z"
        assert manifest["note"] == "carried"
        assert "gpuwm.io.restart" in manifest["contract"]


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
        out = tmp_path / "gen"
        write_generation(out, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, restarts=sets_for(tmp_path, 2),
                         pending={})
        with pytest.raises(EnsembleStateError, match=field):
            read_generation(out, variant(field, value))

    def test_all_differences_are_reported_not_just_the_first(
            self, tmp_path):
        ident = identity(2)
        out = tmp_path / "gen"
        write_generation(out, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, restarts=sets_for(tmp_path, 2),
                         pending={})
        other = identity(2, nx=99, nz=99)
        with pytest.raises(EnsembleStateError) as caught:
            read_generation(out, other)
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
        (tmp_path / "pend_control.npz").write_bytes(b"not a npz")
        with pytest.raises(EnsembleStateError, match="never finished"):
            read_manifest(tmp_path)

    def test_foreign_schema_refused(self, tmp_path):
        manifest_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        manifest_path(tmp_path).write_text(
            json.dumps({"schema": "something.else.v1"}), encoding="utf-8")
        with pytest.raises(EnsembleStateError, match="schema"):
            read_manifest(tmp_path)

    def test_an_atmosphere_only_generation_is_refused_by_name(self, tmp_path):
        """A retired generation carried no soil, surface or accumulators.

        Resuming it would restart every one of those from the prepared
        background at that instant, which is the defect the restart-set
        generation closed, so the refusal names the format, what it
        lacks and the way out.
        """
        manifest_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        manifest_path(tmp_path).write_text(
            json.dumps({"schema": RETIRED_SCHEMA, "leg_number": 4}),
            encoding="utf-8")
        with pytest.raises(EnsembleStateError) as refusal:
            read_manifest(tmp_path)
        message = str(refusal.value)
        assert RETIRED_SCHEMA in message and SCHEMA in message
        assert "soil" in message and "prepared background" in message

    def test_rewriting_a_slot_clears_the_old_marker_first(self, tmp_path):
        ident = identity(1)
        out = tmp_path / "gen"
        write_generation(out, identity=ident, elapsed_seconds=0.0,
                         leg_number=0, restarts=sets_for(tmp_path, 1),
                         pending={})
        write_generation(out, identity=ident, elapsed_seconds=900.0,
                         leg_number=1, restarts=sets_for(tmp_path, 1),
                         pending={})
        assert read_manifest(out)["leg_number"] == 1
        assert (out / "ensemble-manifest.json.superseded").is_file()


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
                         restarts=sets_for(tmp_path / "a", 1), pending={})
        write_generation(slot_dir(tmp_path, 1), identity=ident,
                         elapsed_seconds=900.0, leg_number=1,
                         restarts=sets_for(tmp_path / "b", 1), pending={})
        found = latest_generation(tmp_path)
        assert found is not None
        directory, manifest = found
        assert directory == slot_dir(tmp_path, 0)
        assert manifest["leg_number"] == 2

    def test_a_torn_slot_is_skipped_not_fatal(self, tmp_path):
        ident = identity(1)
        write_generation(slot_dir(tmp_path, 0), identity=ident,
                         elapsed_seconds=900.0, leg_number=1,
                         restarts=sets_for(tmp_path / "a", 1), pending={})
        torn = slot_dir(tmp_path, 1)
        torn.mkdir(parents=True, exist_ok=True)
        (torn / "pend_control.npz").write_bytes(b"half a write")
        directory, manifest = latest_generation(tmp_path)
        assert directory == slot_dir(tmp_path, 0)
        assert manifest["leg_number"] == 1

    def test_a_retired_slot_is_looked_past(self, tmp_path):
        ident = identity(1)
        write_generation(slot_dir(tmp_path, 0), identity=ident,
                         elapsed_seconds=900.0, leg_number=1,
                         restarts=sets_for(tmp_path / "a", 1), pending={})
        old = slot_dir(tmp_path, 1)
        old.mkdir(parents=True, exist_ok=True)
        manifest_path(old).write_text(
            json.dumps({"schema": RETIRED_SCHEMA, "leg_number": 9}),
            encoding="utf-8")
        directory, manifest = latest_generation(tmp_path)
        assert directory == slot_dir(tmp_path, 0)
        assert manifest["leg_number"] == 1
