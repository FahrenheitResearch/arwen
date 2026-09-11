"""Checkpoint publication and complete validation from audit 5063fb3c."""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import (_cfg, _shim_state, _fill_setup, _fill_serialized,
                          _rewrite_restart_archive, _identity_bound_physics_state)


@pytest.mark.parametrize(("key", "match"), [
    ("future/held_pbl_forcing", "under no member namespace this build knows"),
    ("acoustic/omega_pp_v2", "does not classify as checkpoint-only state"),
    ("diag/toa_sw_up", "does not classify as a checkpoint-only driver"),
    ("held/gf_rthblten_v2", "does not classify as a held physics forcing"),
    ("radiation/o3clim_grid", "restart-only driver slots"),
    ("driver/new_tendency", "no driver restore route"),
    ("cumulus/new_history", "no cumulus restore route"),
], ids=["unknown-namespace", "unknown-acoustic", "unknown-diag",
        "unknown-held", "unknown-radiation", "unknown-driver", "unknown-cumulus"])
def test_a_member_this_build_cannot_restore_is_refused_not_dropped(
        monkeypatch, tmp_path, key, match):
    """NEGATIVE CONTROL for the reader's key space.

    ``acoustic/`` and ``diag/`` were introduced on the argument that a new
    cross-step carrier can have its own namespace, absent-tolerant on
    restore, so no file already on disk is rejected and
    ``RESTART_FORMAT_VERSION`` need not move.  That argument is only sound
    while PRESENCE of a member the reader cannot restore is REFUSED --
    absence tolerance is half a contract.  It was not: ``_validate_restart``
    closed the key space for ``state/`` and ``scratch/`` alone, and a
    member under any other prefix -- a namespace a LATER build added on
    exactly the same reasoning -- passed the header/manifest agreement
    check (which compares the file only against its OWN manifest) and was
    then never read.  The resumed run integrated without the carrier the
    namespace exists to carry and reported a clean, manifest-valid,
    bit-continuous restart.

    RED before the closure: every one of these four files restores, with
    the tampered member silently discarded and every state array
    overwritten.
    """
    cfg = _cfg()
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    _fill_serialized(source, seed=20260903)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)

    def _smuggle(payload, header):
        payload[key] = payload["state/thp"].copy()

    tampered = _rewrite_restart_archive(
        path, tmp_path / f"{key.replace('/', '-')}.npz", _smuggle)

    live = _shim_state(cfg, monkeypatch)
    _fill_setup(live)
    _fill_serialized(live, seed=20260904)
    before = {name: getattr(live, name).tobytes()
              for name in restart.STATE_SERIALIZED_ATTRS
              if getattr(live, name, None) is not None}
    with pytest.raises(restart.RestartMismatchError, match=match):
        restart.restore_restart(tampered, live, cfg)
    # And nothing was written on the way to the refusal.
    assert {name: getattr(live, name).tobytes()
            for name in before} == before



def test_the_checkpoint_only_namespaces_are_shape_checked_before_mutation(
        monkeypatch, tmp_path):
    """``acoustic/``'s refusal used to fire from inside the apply phase.

    ``restore_tree_restart`` is built around the invariant that
    ``_validate_restart`` performs every refusal for every member before
    ``_apply_validated_restart`` touches any live state, and holds the
    loaded payloads across the two phases so no member can change in
    between.  ``acoustic/`` and ``diag/`` were outside that: their only
    shape/dtype check was the ``_check_array`` inside the apply phase, by
    which time every ``state/`` array of this domain -- and, in a tree, of
    every domain that sorted earlier -- was already overwritten in place,
    with no rollback.

    RED before the fix: the same refusal is raised, but the nine ``state/``
    arrays are gone.
    """
    cfg = _cfg()
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    _fill_serialized(source, seed=20260905)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)
    assert "acoustic/ww_pp" in restart.state_manifest(source)

    def _reshape_the_carrier(payload, header):
        member = payload["acoustic/ww_pp"]
        payload["acoustic/ww_pp"] = member.reshape((1,) + member.shape)

    tampered = _rewrite_restart_archive(
        path, tmp_path / "reshaped.npz", _reshape_the_carrier)

    live = _shim_state(cfg, monkeypatch)
    _fill_setup(live)
    _fill_serialized(live, seed=20260906)
    before = {name: getattr(live, name).tobytes()
              for name in restart.STATE_SERIALIZED_ATTRS
              if getattr(live, name, None) is not None}
    assert before, "the fixture serialized nothing, so this measures nothing"
    with pytest.raises(restart.RestartMismatchError,
                       match=r"acoustic/ww_pp.*shape"):
        restart.restore_restart(tampered, live, cfg)
    assert {name: getattr(live, name).tobytes()
            for name in before} == before



def test_two_writers_of_one_checkpoint_name_do_not_share_a_scratch_file(
        monkeypatch, tmp_path):
    """NEGATIVE CONTROL for the atomic publish, with two live writers.

    The scratch name used to be derived from the DESTINATION alone
    (``path.name + ".tmp"``), so two writers aiming at the same checkpoint
    name -- a supervisor worker that has not exited while its replacement
    starts, two processes sharing an ``--outdir`` on the single-domain
    route, an offline-child helper and a forecast writing the same instant
    -- opened the same file in ``"wb"``.  The comment over that publish
    says "a crash mid-write must not leave a truncated file under the
    valid gpuwmrst name"; the concurrency case defeated exactly that.

    The interleave is made deterministic rather than raced: the peer
    writer runs to completion inside the first writer's ``np.savez``,
    while the first still holds its scratch file open.

    RED before the per-writer name: the peer's ``os.replace`` renames the
    SHARED scratch file out from under the first writer, whose own
    ``fsync_file`` then raises ``FileNotFoundError`` -- after its buffered
    payload has been flushed into the inode the peer just published under
    the valid name.
    """
    cfg = _cfg()
    first = _shim_state(cfg, monkeypatch)
    _fill_setup(first)
    _fill_serialized(first, seed=20260907)
    peer = _shim_state(cfg, monkeypatch)
    _fill_setup(peer)
    _fill_serialized(peer, seed=20260908)

    target = tmp_path / restart.restart_filename(datetime(1974, 4, 3, 12))
    savez = np.savez
    peer_published = []
    interleaved = []

    def _let_the_peer_publish_mid_write(stream, **payload):
        savez(stream, **payload)
        if not interleaved:
            interleaved.append(True)     # the peer's own savez is not this
            peer_published.append(restart.write_restart(target, peer, cfg))

    monkeypatch.setattr(restart.np, "savez", _let_the_peer_publish_mid_write)
    restart.write_restart(target, first, cfg)
    monkeypatch.setattr(restart.np, "savez", savez)

    assert peer_published, "the interleave never ran, so this measures nothing"
    # The last publisher wins, and what it published is ITS OWN complete
    # archive rather than two payloads spliced at one another's offsets.
    # Read off the disk rather than through restore_restart: the property
    # under test is the published BYTES.
    with np.load(target, allow_pickle=False) as data:
        published = {name: data[name] for name in data.files}
    header = json.loads(bytes(bytearray(
        published.pop(restart._HEADER_KEY))).decode("utf-8"))
    assert set(published) == set(header["array_manifest"])
    checked = 0
    for key, host in published.items():
        if not key.startswith("state/"):
            continue
        name = key[len("state/"):]
        assert host.tobytes() == getattr(first, name).tobytes(), key
        checked += 1
    assert checked, "the archive carried no state members to compare"
    assert not list(tmp_path.glob("*.tmp*"))



def test_the_published_checkpoint_name_is_made_durable_by_a_directory_fsync(
        monkeypatch, tmp_path):
    """``os.replace`` is atomic for readers and not durable for the disk.

    Every other durable publication in the estate ends with the parent
    directory's fsync (``supervisor.atomic_write_json``,
    ``quarantine_file``, ``atomic_publish_file``); the checkpoint -- the
    one artifact a crash is supposed to leave behind -- did not, so a
    checkpoint the run reported as written could be absent after a power
    loss.  On the tree route the root member is the set's commit marker,
    so an unsynced directory can surface a root whose children's renames
    were lost: the one torn shape ``write_tree_restart``'s publish order
    exists to prevent.

    RED before the fix: ``restart._fsync_directory`` does not exist, so
    there is nothing to record.
    """
    cfg = _cfg()
    state = _shim_state(cfg, monkeypatch)
    _fill_setup(state)

    synced = []
    real = restart._fsync_directory
    monkeypatch.setattr(
        restart, "_fsync_directory",
        lambda directory: (synced.append(Path(directory)), real(directory))[1])

    target = tmp_path / "deep" / "rst.npz"
    restart.write_restart(target, state, cfg)
    assert synced == [target.parent]



@pytest.mark.parametrize("defect", ["shape", "dtype"])
def test_diagnostic_carrier_is_checked_before_any_state_copy(monkeypatch, tmp_path, defect):
    from test_restart import _shim_driver_state
    cfg = _cfg()
    source, driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(source)
    _fill_serialized(source, seed=100)
    driver.olr = np.ones(source.mup.shape, np.float32)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)
    def change(payload, header):
        value = payload["diag/olr"]
        payload["diag/olr"] = (value[None] if defect == "shape"
                                else value.astype(np.float64))
    bad = _rewrite_restart_archive(path, tmp_path / "bad.npz", change)
    live, target_driver = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(live)
    _fill_serialized(live, seed=101)
    target_driver.olr = np.zeros(live.mup.shape, np.float32)
    before = {key: value.copy() for key, value in restart.state_manifest(live).items()}
    with pytest.raises(restart.RestartMismatchError, match="diag/olr"):
        restart.restore_restart(bad, live, cfg)
    for key, value in restart.state_manifest(live).items():
        np.testing.assert_array_equal(value, before[key])
    assert not target_driver.olr.any()


def _rrtmgp_4_4_checkpoint(monkeypatch, tmp_path, name):
    """One written 4/4 checkpoint, and the config that wrote it."""
    cfg = _cfg(moist=True, mp_physics=10, morr_rimed_ice=1,
               sf_sfclay_physics=1, sf_surface_physics=2, bl_pbl_physics=1,
               ra_physics=4, cu_physics=1)
    state, _ = _identity_bound_physics_state(cfg, monkeypatch)
    monkeypatch.setattr(restart, "_asset_sha256",
                        lambda path: f"test-sha256:{Path(path).name}")
    return cfg, restart.write_restart(tmp_path / name, state, cfg)


def test_a_crossed_4_4_radiation_resume_is_refused_by_its_breakage(
        monkeypatch, tmp_path):
    """AUDIT R-048.  Radiation scheme id 4 is worn by two different codes --
    the WRF v4.6.1 RRTMG port and the RTE+RRTMGP substitution -- and a
    resume must not swap them under a running trajectory.  It never could:
    the configuration walk reported ``ra_rrtmg_variant`` as a changed field
    and the physics identity reported "radiation, algorithms" as differing
    components.  Neither sentence said WHAT breaks, so the reader learned
    that two records differ, not that the two implementations transcribe
    different algorithms, treat the atmosphere above the model top
    differently (WRF's 4 mb buffer layers against RRTMGP's) and pin
    different coefficient tables.

    The refusal now names both implementations, the breakage and both ways
    out, and it is asked BEFORE the configuration walk so it is the
    sentence the user reads -- asserted here by the message, since the
    generic walk would otherwise answer this same file first.
    """
    cfg, path = _rrtmgp_4_4_checkpoint(monkeypatch, tmp_path, "modern.npz")

    def _claim_the_legacy_port_wrote_it(payload, header):
        header["config"]["ra_rrtmg_variant"] = "rrtmg_legacy"

    crossed = _rewrite_restart_archive(
        path, tmp_path / "crossed.npz", _claim_the_legacy_port_wrote_it)

    live = _shim_state(cfg, monkeypatch)
    _fill_setup(live)
    with pytest.raises(restart.RestartMismatchError) as raised:
        restart.restore_restart(crossed, live, cfg)
    message = str(raised.value)
    assert "WRF v4.6.1 RRTMG port" in message
    assert "RTE+RRTMGP substitution" in message
    # The breakage, not a field name in a list of differences.
    assert "heating-rate" in message and "buffer layers" in message
    # Both ways out, each spelled as the setting that takes it.
    assert "ra_rrtmg_variant='rrtmg_legacy'" in message
    assert "start a new run from t = 0" in message
    # And it answered before the generic configuration walk did.
    assert "written under a different configuration" not in message


def test_a_same_variant_4_4_resume_is_untouched_by_that_refusal(
        monkeypatch, tmp_path):
    """NEGATIVE CONTROL for the R-048 gate: it refuses nothing that was
    resumable before it existed.  A 4/4 checkpoint restores into a run of
    the SAME variant, and a header written before the variant field existed
    restores as the RTE+RRTMGP substitution -- the migration rule the
    configuration walk already applies, which never infers the legacy port.
    """
    cfg, path = _rrtmgp_4_4_checkpoint(monkeypatch, tmp_path, "same.npz")
    live, _ = _identity_bound_physics_state(cfg, monkeypatch)
    restart.restore_restart(path, live, cfg)

    def _drop_the_variant_field(payload, header):
        header["config"].pop("ra_rrtmg_variant", None)

    pre_field = _rewrite_restart_archive(
        path, tmp_path / "pre-field.npz", _drop_the_variant_field)
    fresh, _ = _identity_bound_physics_state(cfg, monkeypatch)
    restart.restore_restart(pre_field, fresh, cfg)
