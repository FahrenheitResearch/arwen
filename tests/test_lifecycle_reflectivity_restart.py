"""A lifecycle restart carries the last actual microphysics-time volume."""
from datetime import timedelta

import numpy as np
import pytest

from gpuwm.io import restart
from tilestream.restart_stream import RestartRefused
from gpuwm.core.storm_tracking import signal_plane
from test_restart import _rewrite_restart_archive
from test_streamed_lifecycle_restart import attach, fixture


def _with_held_reflectivity(monkeypatch, *, streamed, clear=False):
    model, start, _ = fixture(monkeypatch, streamed=False)
    endpoints = {}
    expected = {}
    for node in model.walk_parent_first():
        node.clock.spec.history_ticks = 300
        gid = node.cfg.grid_id
        shape = node.state.p.shape
        volume = np.linspace(-35.0, 65.0, np.prod(shape), dtype=np.float32).reshape(shape)
        volume += np.float32(gid)
        # Include a valid signed zero to bind the exact payload, not == alone.
        volume[0, shape[1] // 2, shape[2] // 2] = np.float32(-0.0)
        node.state.scratch(shape, "refl_10cm")[...] = volume
        # The existing production ring guard leaves specified boundary
        # diagnostic cells zero; construct a volume the model can emit.
        from gpuwm.core.microphysics import normalize_spec_zone_ring_after_restore
        normalize_spec_zone_ring_after_restore(node.state, node.cfg.run)
        expected[gid] = node.state._scratch["refl_10cm"].copy()
        assert np.count_nonzero(expected[gid]) > 0
        if streamed:
            endpoints[gid] = attach(node, clear=clear)
            from gpuwm.core.streaming import refl_inventory
            from tilestream.physics_inventory import carrier_inventory
            transport = refl_inventory(carrier_inventory)(node.state)
            endpoints[gid].store["scratch/refl_10cm"] = transport[
                "scratch/refl_10cm"].copy()
            if clear:
                endpoints[gid].store["scratch/refl_10cm"].fill(-3.0)
            # The resident mirror is stale; the store is the value authority.
            node.state._scratch["refl_10cm"].fill(-999.0)
        elif clear:
            node.state._scratch["refl_10cm"].fill(-999.0)
    return model, start, endpoints, expected


@pytest.mark.parametrize("source_streamed", [False, True])
@pytest.mark.parametrize("restore_streamed", [False, True])
def test_lifecycle_restart_preserves_microphysics_time_reflectivity(
        tmp_path, monkeypatch, source_streamed, restore_streamed):
    source, start, _, expected = _with_held_reflectivity(
        monkeypatch, streamed=source_streamed)
    path = restart.write_tree_restart(
        tmp_path / "source", source, start + timedelta(seconds=3600))
    for gid, slots in restart.read_tree_lifecycle_header(path).window_slots.items():
        assert slots.count("refl_10cm") == 1
        member = next(path.parent.glob(f"gpuwmrst_d{int(gid):02d}_*.npz"))
        with np.load(member, allow_pickle=False) as archive:
            assert archive.files.count("scratch/refl_10cm") == 1
            assert not any(name.startswith("driver/refl_10cm") for name in archive.files)
            assert archive["scratch/refl_10cm"].tobytes() == expected[int(gid)].tobytes()

    resumed, _, endpoints, _ = _with_held_reflectivity(
        monkeypatch, streamed=restore_streamed, clear=True)
    if not restore_streamed:
        # Production resident preparation does not allocate this lazy volume.
        for node in resumed.walk_parent_first():
            del node.state._scratch["refl_10cm"]
    restart.restore_tree_restart(path, resumed)
    for node in resumed.walk_parent_first():
        gid = node.cfg.grid_id
        actual = (endpoints[gid].store["scratch/refl_10cm"] if restore_streamed
                  else node.state._scratch["refl_10cm"])
        assert actual.shape == expected[gid].shape
        assert actual.dtype == expected[gid].dtype == np.float32
        assert actual.tobytes() == expected[gid].tobytes()
        # The actual tracker reader must see the restored held volume now,
        # before any microphysics step can recompute it from later state.
        np.testing.assert_array_equal(
            signal_plane(node.state, "reflectivity"),
            expected[gid].astype(np.float64).max(axis=0))


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype"])
@pytest.mark.parametrize("restore_streamed", [False, True])
def test_corrupt_lifecycle_reflectivity_refuses_before_any_domain_changes(
        tmp_path, monkeypatch, corruption, restore_streamed):
    source, start, _, _ = _with_held_reflectivity(monkeypatch, streamed=False)
    path = restart.write_tree_restart(
        tmp_path / "source", source, start + timedelta(seconds=3600))
    child = next(path.parent.glob("gpuwmrst_d02_*.npz"))
    def corrupt(payload, header):
        if corruption == "missing":
            del payload["scratch/refl_10cm"]
        elif corruption == "shape":
            payload["scratch/refl_10cm"] = payload["scratch/refl_10cm"][0]
        else:
            payload["scratch/refl_10cm"] = payload["scratch/refl_10cm"].astype(np.float64)
    amended = _rewrite_restart_archive(child, tmp_path / "corrupt.npz", corrupt)
    child.write_bytes(amended.read_bytes())
    resumed, _, endpoints, _ = _with_held_reflectivity(
        monkeypatch, streamed=restore_streamed, clear=True)
    from tilestream.physics_inventory import carrier_manifest
    before = {node.cfg.grid_id: {k: v.tobytes() for k, v in
              (endpoints[node.cfg.grid_id].store if restore_streamed
               else carrier_manifest(node.state)).items()}
              for node in resumed.walk_parent_first()}
    with pytest.raises((restart.RestartMismatchError, RestartRefused), match="refl_10cm"):
        restart.restore_tree_restart(path, resumed)
    for node in resumed.walk_parent_first():
        actual = (endpoints[node.cfg.grid_id].store if restore_streamed
                  else carrier_manifest(node.state))
        assert {k: v.tobytes() for k, v in actual.items()} == before[node.cfg.grid_id]


def test_held_reflectivity_stays_out_of_an_ordinary_checkpoint(tmp_path, monkeypatch):
    source, _, _, _ = _with_held_reflectivity(monkeypatch, streamed=False)
    node = source.root
    path = restart.write_restart(tmp_path / "ordinary.npz", node.state, node.cfg.run)
    with np.load(path, allow_pickle=False) as archive:
        assert "scratch/refl_10cm" not in archive.files
    assert restart.classify_scratch_slot("refl_10cm") == "rebuild"
    assert "refl_10cm" in restart.LIFECYCLE_HELD_SCRATCH_SLOTS
    assert "refl_10cm" not in restart.lifecycle_window_slots(node.state)


@pytest.mark.parametrize("restore_streamed", [False, True])
def test_old_lifecycle_without_declared_reflectivity_refuses_before_mutation(
        tmp_path, monkeypatch, restore_streamed):
    source, start, _, _ = _with_held_reflectivity(monkeypatch, streamed=False)
    path = restart.write_tree_restart(
        tmp_path / "source", source, start + timedelta(seconds=3600))
    for member in path.parent.glob("gpuwmrst_d*.npz"):
        def old_format(payload, header):
            payload.pop("scratch/refl_10cm", None)
            block = header.get(restart.NEST_LIFECYCLE_HEADER_KEY)
            if block is not None:
                for slots in block["window_slots"].values():
                    slots.remove("refl_10cm")
        amended = _rewrite_restart_archive(member, tmp_path / "old.npz", old_format)
        member.write_bytes(amended.read_bytes())
    resumed, _, endpoints, _ = _with_held_reflectivity(
        monkeypatch, streamed=restore_streamed, clear=True)
    from tilestream.physics_inventory import carrier_manifest
    for node in resumed.walk_parent_first():
        if restore_streamed:
            endpoints[node.cfg.grid_id].store["scratch/refl_10cm"].fill(0.0)
        else:
            del node.state._scratch["refl_10cm"]
    before = {node.cfg.grid_id: {k: v.tobytes() for k, v in
              (endpoints[node.cfg.grid_id].store if restore_streamed
               else carrier_manifest(node.state)).items()}
              for node in resumed.walk_parent_first()}
    with pytest.raises(restart.RestartMismatchError,
                       match="predates held lifecycle reflectivity.*d01"):
        restart.restore_tree_restart(path, resumed)
    for node in resumed.walk_parent_first():
        actual = (endpoints[node.cfg.grid_id].store if restore_streamed
                  else carrier_manifest(node.state))
        assert {k: v.tobytes() for k, v in actual.items()} == before[node.cfg.grid_id]


@pytest.mark.parametrize("direction", ["publish", "adopt"])
def test_held_reflectivity_exchange_uses_canonical_host_view(direction):
    from types import SimpleNamespace
    from gpuwm.core.streamed_state import CanonicalStoreState
    from gpuwm.core.streaming import StreamedDomain, StreamingRefused
    slab = np.zeros((4, 2, 8), np.float32)
    volume = np.linspace(-35., 65., 4 * 24 * 8, dtype=np.float32).reshape(4, 24, 8)
    store = {"scratch/refl_10cm": volume}
    state = CanonicalStoreState(
        SimpleNamespace(), SimpleNamespace(nx=8, ny=24, nz=4),
        store=store, geography={}, scalars={"elapsed_seconds": 120.},
        inventory={"scratch/refl_10cm": slab}, geography_inventory={})
    def forbid(*args):
        raise AssertionError("canonical host view entered resident inventory traversal")
    endpoint = StreamedDomain(SimpleNamespace(store=store), None,
                              state=state, host_store=True, inventory_fn=forbid)
    before = volume.tobytes()
    assert getattr(endpoint, direction)(("scratch/refl_10cm",)) == ("scratch/refl_10cm",)
    assert state.existing_scratch("refl_10cm") is volume
    assert volume.tobytes() == before
    with pytest.raises(StreamingRefused, match="scratch/missing"):
        getattr(endpoint, direction)(("scratch/missing",))
