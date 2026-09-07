"""Checkpoint publication and complete validation from audit 5063fb3c."""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import _cfg, _shim_state, _fill_setup, _fill_serialized, _rewrite_restart_archive


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
