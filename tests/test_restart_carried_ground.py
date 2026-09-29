"""A nest carried by a moving parent keeps its ring rain across a restart.

When a following parent moves, a nest that rides inside it at the same
parent-relative placement is rebuilt on new ground: its accumulators shift
in index space and interior rain lands in its specified-zone ring.  The
checkpoint lists only the grids that moved themselves in
``moved_grid_ids``, so the restore read the carried nest as "never moved"
and the ring migration zeroed that rain (a 9/3/1 km vortex-following run
resumed at hour three with 71 east-edge d03 RAINNC cells at zero where the
unbroken run held 6.6 to 70.4 mm).
"""
from datetime import timedelta
from types import SimpleNamespace

import numpy as np

from gpuwm.io import restart
from test_restart import (_RING_CARRIED_SLOTS, _cfg, _fill_setup,
                          _rewrite_restart_archive, _ring_mask_2d,
                          _ring_seeded_state, _sealed_tree_fixture,
                          _shim_state)


def _carried_restore(monkeypatch, tmp_path, *, mover, **options):
    """Write a ring-seeded grid-1 checkpoint whose parent is grid 2, stamp
    ``mover`` as the only grid that moved itself, and restore it."""
    cfg = _cfg(nx=8, ny=7, moist=True, mp_physics=10, nested=True)
    state = _ring_seeded_state(monkeypatch, cfg, seed=711)
    written = restart.write_restart(tmp_path / "original.npz", state, cfg)

    def stamp(payload, header):
        header.update(grid_id=1, parent_id=2,
                      relocation={"moves": 1, "segment_id": "0" * 64,
                                  "moved_grid_ids": [mover]})

    path = _rewrite_restart_archive(written, tmp_path / "carried.npz", stamp)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    validated = restart._validate_restart(path, fresh, cfg)
    restart._apply_validated_restart(validated, fresh, cfg, **options)
    return cfg, state, fresh


def _assert_ring_rain_kept(cfg, state, fresh):
    ring = _ring_mask_2d(cfg)
    for slot in _RING_CARRIED_SLOTS:
        got, src = fresh._scratch[slot], state._scratch[slot]
        assert (src[ring] != 0.0).all(), f"{slot}: the fixture seeds rain"
        np.testing.assert_array_equal(got, src, err_msg=slot)
    # h_diabatic is not carried by a move and its ring value stays zero.
    assert (fresh.h_diabatic[:, ring] == 0.0).all()


def test_a_nest_riding_in_a_mover_keeps_its_ring_rain(monkeypatch, tmp_path):
    """Grid 1 rides inside grid 2, and only grid 2 moved itself."""
    _assert_ring_rain_kept(*_carried_restore(monkeypatch, tmp_path, mover=2))


def test_a_nest_two_levels_below_the_mover_keeps_its_ring_rain(
        monkeypatch, tmp_path):
    """Grid 1 rides inside grid 2, which rides inside the mover, grid 3."""
    _assert_ring_rain_kept(*_carried_restore(
        monkeypatch, tmp_path, mover=3,
        parent_by_grid={1: 2, 2: 3, 3: 0}))


def test_a_grid_off_the_movers_branch_keeps_the_migration():
    cfg = SimpleNamespace(grid_id=4)
    topology = {1: 0, 2: 1, 3: 2, 4: 1}
    relocation = {"moved_grid_ids": [2]}
    # Sibling of the mover: same parent, not carried.
    assert restart._grid_relocated(
        {"grid_id": 4, "parent_id": 1, "relocation": relocation}, cfg,
        parent_by_grid=topology) is False
    # The root is the mover's ancestor, never its passenger.
    assert restart._grid_relocated(
        {"grid_id": 1, "parent_id": 0, "relocation": relocation}, cfg,
        parent_by_grid=topology) is False
    # The mover's own child and grandchild are carried.
    for gid in (2, 3):
        assert restart._grid_relocated(
            {"grid_id": gid, "parent_id": topology[gid],
             "relocation": relocation}, cfg,
            parent_by_grid=topology) is True


def test_the_member_header_alone_names_a_directly_carried_nest():
    """A single member still knows its own parent without the set's map."""
    assert restart._grid_relocated(
        {"grid_id": 3, "parent_id": 2,
         "relocation": {"moved_grid_ids": [2]}},
        SimpleNamespace(grid_id=3)) is True


def test_a_parent_cycle_in_a_damaged_map_ends_the_walk():
    assert restart._grid_relocated(
        {"grid_id": 2, "parent_id": 3,
         "relocation": {"moved_grid_ids": [9]}},
        SimpleNamespace(grid_id=2), parent_by_grid={2: 3, 3: 2}) is False


def test_tree_restore_hands_every_member_the_sets_topology(
        monkeypatch, tmp_path):
    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=7200.0, payload_seed=31)
    root_path = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600))
    resumed, _ = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=7200.0, payload_seed=91)
    seen = {}
    real = restart._apply_validated_restart

    def spy(validated, state, cfg, **options):
        seen[int(validated.header["grid_id"])] = options.get("parent_by_grid")
        return real(validated, state, cfg, **options)

    monkeypatch.setattr(restart, "_apply_validated_restart", spy)
    restart.restore_tree_restart(root_path, resumed)
    assert seen == {1: {1: 0, 2: 1}, 2: {1: 0, 2: 1}}
