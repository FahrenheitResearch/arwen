"""The consumer side of a chained domain tree (A136 L6).

A mapped domain tree prepared on the CPU publishes a head (its children
complete, the root's start state streamed) and then one segment per root
boundary interval.  These tests hold the doors that bind that head: the
stage seam that relays it to the tree runner, the runner's own binding
rules, the clock guard that keeps a head-bound run on the clock the sealed
tree chooses, the restart identity a chained tree shares across both
bindings, and the start needs a delayed nest adds.

CPU only; no device, no source data.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from gpuwm import stage_cli
from gpuwm import prepared_domain_tree_forecast as tree
from gpuwm.ingest import boundary_stream
from gpuwm.ingest.boundary_stream import (
    HIERARCHY_HEAD_DIRNAME, PreparedTreeWriter, StreamedClockChanged,
    derived_clock, domain_tree_head_fields, keep_interval_check,
)
from gpuwm.prepared_domain_tree_forecast import StreamedClockGuard

from test_stream_tree_producer import (
    TREE_CACHE, _SnapshotFrames, _initial, _met, _snapshots, _times,
)


def _mapped_hierarchy_schema():
    from gpuwm.prepared_single_domain_forecast import _HIERARCHY_PROOF_SCHEMA

    return next(iter(_HIERARCHY_PROOF_SCHEMA.values()))


def _tree_head(tmp_path, *, labels=("d01", "d02"), proof_head=None,
               tree_fields=True):
    """A real tree head under ``tmp_path/tree`` (the producer's own writer)."""

    snapshots = _snapshots()
    times = _times(len(snapshots))
    staging = tmp_path / ".tmp-tree"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "tree",
        identity={"source": "tree-consumer-test"}, cache_name=TREE_CACHE,
        chained=True)
    frames = _SnapshotFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    proof_head = proof_head or {
        "schema": _mapped_hierarchy_schema(),
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "domain_count": len(labels),
        "forcing_times": [t.isoformat() for t in times],
    }
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[seconds[k], seconds[k + 1]]
                         for k in range(len(times) - 1)],
            "fields": frames.inventory},
        proof_head=proof_head, input_manifest_sha256="a" * 64,
        forcing=frames,
        tree=(domain_tree_head_fields(list(labels), root_cache=TREE_CACHE)
              if tree_fields else None))
    return tmp_path / "tree", writer


# ---------------------------------------------------------------------------
# The stage seam relays a tree head to the tree runner
# ---------------------------------------------------------------------------


def test_a_tree_head_resolves_as_a_tree_bundle(tmp_path):
    root, writer = _tree_head(tmp_path)
    head = json.loads((root / "boundary-stream" / "head.json").read_text(
        encoding="utf-8"))
    bundle = stage_cli.resolve_head_bundle(root, head["head_sha256"])
    assert bundle["layout"] == "tree"
    assert bundle["domains"] == 2
    assert bundle["head_sha256"] == head["head_sha256"]
    assert bundle["source_manifest_sha256"] == "a" * 64


def test_a_tree_head_whose_proof_counts_other_domains_is_refused(tmp_path):
    root, _ = _tree_head(tmp_path, proof_head={
        "schema": _mapped_hierarchy_schema(),
        "status": "READY_NOT_YET_STOCK_WRF_GATED", "domain_count": 3})
    head = json.loads((root / "boundary-stream" / "head.json").read_text(
        encoding="utf-8"))
    with pytest.raises(stage_cli.StageRefusal, match="counts 3"):
        stage_cli.resolve_head_bundle(root, head["head_sha256"])


def test_a_hierarchy_proof_under_a_single_domain_head_is_refused(tmp_path):
    root, _ = _tree_head(tmp_path, tree_fields=False)
    head = json.loads((root / "boundary-stream" / "head.json").read_text(
        encoding="utf-8"))
    with pytest.raises(stage_cli.StageRefusal, match="no runner can bind"):
        stage_cli.resolve_head_bundle(root, head["head_sha256"])


def test_the_tree_command_binds_the_head_not_a_receipt(tmp_path):
    root, _ = _tree_head(tmp_path)
    head = json.loads((root / "boundary-stream" / "head.json").read_text(
        encoding="utf-8"))
    bundle = stage_cli.resolve_head_bundle(root, head["head_sha256"])
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    command = stage_cli.sim_command(
        bundle, experiment_config=config, wps_namelist=None,
        outdir=tmp_path / "run")
    assert command[2] == stage_cli.TREE_RUNNER
    assert "--prepared-head-sha256" in command
    assert command[command.index("--prepared-head-sha256") + 1] \
        == head["head_sha256"]
    assert "--preparation-receipt-sha256" not in command
    assert command[command.index("--prepared-root") + 1] == str(root)
    # The runner's own parser takes exactly what the seam composed.
    args = tree.build_parser().parse_args(command[3:])
    assert args.prepared_head_sha256 == head["head_sha256"]
    assert args.preparation_receipt_sha256 is None


def test_the_tree_runner_binds_exactly_one_preparation(tmp_path):
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    for binding in ({}, {"preparation_receipt_sha256": "a" * 64,
                         "prepared_head_sha256": "b" * 64}):
        with pytest.raises(ValueError, match="exactly one"):
            tree.preflight_prepared_tree(
                prepared_root=tmp_path, experiment_config=config,
                experiment_config_sha256="c" * 64, **binding)


def test_the_head_loader_refuses_a_single_domain_head(tmp_path):
    root, _ = _tree_head(tmp_path, tree_fields=False)
    head = json.loads((root / "boundary-stream" / "head.json").read_text(
        encoding="utf-8"))
    with pytest.raises(ValueError, match="single-domain runner binds"):
        tree._load_head_document(root, head["head_sha256"])


def test_the_head_loader_reads_the_proof_without_its_seal_keys(tmp_path):
    root, _ = _tree_head(tmp_path)
    head = json.loads((root / "boundary-stream" / "head.json").read_text(
        encoding="utf-8"))
    path, document, source, bound = tree._load_head_document(
        root, head["head_sha256"])
    assert path == root / "boundary-stream" / "head.json"
    assert document == head["basis"]["proof_head"]
    assert not set(document) & boundary_stream.SEAL_ONLY_PROOF_KEYS
    assert bound["head_sha256"] == head["head_sha256"]
    assert source in {"mapped", *tree.SUPPORTED_SOURCES}


# ---------------------------------------------------------------------------
# The head's hierarchy is held to what the head binds
# ---------------------------------------------------------------------------


def _two_domain_exp():
    return SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, parent_id=0),
        SimpleNamespace(grid_id=2, parent_id=1)))


def _bound_tree_head(tmp_path):
    import hashlib

    receipt = b'{"grid_id": 2}'
    staging = tmp_path / ".tmp-tree"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "tree",
        identity={"source": "tree-consumer-test"}, cache_name=TREE_CACHE,
        chained=True)
    frames = _SnapshotFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    snapshots = _snapshots()
    frames.add_snapshot(snapshots[0], index=0)
    child = staging / HIERARCHY_HEAD_DIRNAME / "domains" / "d02"
    child.mkdir(parents=True)
    (child / "receipt.json").write_bytes(receipt)
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[0.0, 3600.0], [3600.0, 7200.0]],
            "fields": frames.inventory},
        proof_head={"schema": "test"}, input_manifest_sha256="a" * 64,
        forcing=frames,
        tree=domain_tree_head_fields(
            ["d01", "d02"], root_cache=TREE_CACHE,
            children_receipts={"d02": hashlib.sha256(receipt).hexdigest()}))
    root = tmp_path / "tree"
    head = json.loads((root / "boundary-stream" / "head.json").read_text(
        encoding="utf-8"))
    return root, head


def test_the_heads_children_are_held_to_the_receipts_it_binds(tmp_path):
    root, head = _bound_tree_head(tmp_path)
    hierarchy, receipts = tree._head_hierarchy(root, head, _two_domain_exp())
    assert hierarchy == root / HIERARCHY_HEAD_DIRNAME
    assert receipts == [None, {"grid_id": 2}]
    # A child receipt changed after the head was published is refused.
    (hierarchy / "domains" / "d02" / "receipt.json").write_text(
        '{"grid_id": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="d02 receipt differs"):
        tree._head_hierarchy(root, head, _two_domain_exp())


def test_a_head_for_other_domains_is_refused(tmp_path):
    root, head = _bound_tree_head(tmp_path)
    exp = SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, parent_id=0),
        SimpleNamespace(grid_id=2, parent_id=1),
        SimpleNamespace(grid_id=3, parent_id=1)))
    with pytest.raises(ValueError, match="not this experiment's"):
        tree._head_hierarchy(root, head, exp)


# ---------------------------------------------------------------------------
# What starts after the seal, and why
# ---------------------------------------------------------------------------


def test_a_following_nest_starts_after_the_seal_by_name():
    exp = SimpleNamespace(domains=(SimpleNamespace(grid_id=1),), tiles=None)
    reason = tree._head_needs_seal(exp, relocation_follow=True)
    assert "statics corridor" in reason and "seal" in reason


def test_a_following_nest_starts_at_a_head_that_carries_its_corridor():
    """A136 L7d: a moving or cyclone tree's head holds its corridor.

    The breakage this prevents: a storm-following nest's forecast waited
    for every boundary interval to be prepared and sealed, though the only
    thing it read from the seal was the statics corridor, which needs no
    boundary time.  A head without one (prepared before heads carried it)
    still waits, by name.
    """

    exp = SimpleNamespace(domains=(SimpleNamespace(grid_id=1, tiles=None),),
                          tiles=None)
    assert tree._head_needs_seal(
        exp, relocation_follow=True, head_corridor=True) is None
    assert "statics corridor" in tree._head_needs_seal(
        exp, relocation_follow=True, head_corridor=False)
    # A streamed root uses the same head and corridor as a resident root.
    from gpuwm.core.streaming import StreamingOptions

    tiled = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1, tiles=None),),
        tiles=StreamingOptions.from_mapping({"mode": "on"}))
    assert tree._head_needs_seal(
        tiled, relocation_follow=True, head_corridor=True) is None


@pytest.mark.parametrize("mode", ["on", "auto"])
def test_a_tiles_root_starts_at_its_head(mode):
    """The root's store takes start arrays from the head, then seam tables.

    A186 retires the seal requirement for both explicit streaming and the
    automatic mode a cyclone setup writes.  The planner still chooses the
    card's resident or streamed road, but neither road requires the seal.
    """

    from gpuwm.core.streaming import StreamingOptions

    exp = SimpleNamespace(domains=(SimpleNamespace(grid_id=1, tiles=None),),
                          tiles=StreamingOptions.from_mapping({"mode": mode}))
    assert tree._head_needs_seal(exp, relocation_follow=False) is None


@pytest.mark.parametrize('changed', [False, True])
def test_tiles_root_seal_fingerprint_uses_full_domain_flags(monkeypatch, changed):
    """The last slab is uniform while an earlier row rotates the domain."""
    from dataclasses import replace
    import numpy as np
    from gpuwm.core.streamed_relocation import StreamedChildReconstruction
    from gpuwm.core.streaming import StreamedDomain
    from gpuwm.io import restart
    from gpuwm.state_serialization_contract import STATE_SETUP_ARRAYS
    from tilestream import driver, physics_inventory
    from test_restart import _shim_state
    from test_restart_preserved_forcing import _clear_out_state, _specified

    cfg = _specified()
    resident = _clear_out_state(cfg, monkeypatch)
    resident.elapsed_seconds = 3600.
    for name in ('msft', 'msfu', 'msfv'):
        getattr(resident, name).fill(1.)
    for name in ('f', 'e', 'sina'):
        getattr(resident, name).fill(0.)
    resident.cosa.fill(1.)
    resident.msft[0, 0] = 1.25
    resident.has_msf = resident.rotational = True
    slab = _shim_state(replace(cfg, ny=2), monkeypatch)
    for name in STATE_SETUP_ARRAYS:
        source, target = getattr(resident, name), getattr(slab, name)
        target[...] = (source if source.ndim < 2 else
                       source[..., -target.shape[-2]:, :])
    slab.has_msf = slab.rotational = False
    geo = {key: value.copy() for key, value in driver.geography_inventory(resident).items()}
    store = {key: value.copy() for key, value in physics_inventory.carrier_manifest(resident).items()}
    scalars = physics_inventory.carrier_scalars(resident)
    facade = StreamedChildReconstruction._facade(slab, cfg, store, geo, scalars)
    facade.lateral_boundaries = resident.lateral_boundaries
    assert not facade.has_msf and not facade.rotational
    class Endpoint:
        restart_setup = StreamedDomain.restart_setup
    stream = Endpoint()
    stream._setup = None
    stream._template = stream.template_state = slab
    stream._state = facade
    stream._geography = geo
    stream._boundaries = resident.lateral_boundaries
    stream.store = store
    setup = stream.restart_setup()
    assert setup.scalars['has_msf'] and setup.scalars['rotational']
    expected = restart.setup_fingerprint(resident)
    assert restart.setup_fingerprint(facade) != expected
    if changed:
        # DomainSetup borrows the complete geography. A changed array must
        # fail the final sealed fingerprint comparison even with correct flags.
        geo['setup/ht'][0, 0] += np.float32(1.)
    node = SimpleNamespace(state=facade, clock=SimpleNamespace(elapsed_seconds=3600.))
    observed = tree._root_setup_fingerprint(node, stream)
    assert (observed == expected) is (not changed)
    assert tree._root_setup_fingerprint(SimpleNamespace(state=resident)) == expected


def _tiles_head_inputs(*, mode="auto", bound_to_head=True):
    from gpuwm.core.streaming import StreamingOptions

    run = SimpleNamespace(sf_urban_physics=0)
    root = SimpleNamespace(grid_id=1, parent_id=0, tiles=None, run=run)
    child = SimpleNamespace(grid_id=2, parent_id=1, tiles=None, run=run)
    exp = SimpleNamespace(
        domains=(root, child),
        tiles=StreamingOptions.from_mapping({"mode": mode}))
    return SimpleNamespace(
        experiment=exp, source="gfs",
        domains=tuple(SimpleNamespace(grid_id=dc.grid_id) for dc in exp.domains),
        stream_head={"head_sha256": "d" * 64} if bound_to_head else None)


@pytest.mark.parametrize("mode", ["on", "auto"])
def test_the_tiles_tree_door_runs_the_head_without_waiting_for_the_seal(
        tmp_path, monkeypatch, capsys, mode):
    """The entry point preserves a head binding when the root can stream.

    The actual store and boundary behavior is exercised separately.  This
    door test catches the retired fallback being reintroduced around it.
    """

    from gpuwm import capabilities, provenance_gate

    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *_: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    head_inputs = _tiles_head_inputs(mode=mode)
    bindings, runs = [], []

    def preflight(**binding):
        bindings.append(binding)
        return head_inputs

    def sealed_proof(root, head_sha256, *, on_wait=None):
        pytest.fail("a tiles root waited for its seal before forecasting")

    def run_prepared_tree(bound, *, output_directory, restart, **_):
        runs.append((bound, restart))
        return {"status": "PASS", "readiness": "ready",
                "execution_plan": {"plan_id": "plan", "domain_count": 2},
                "wall_seconds": 1.0, "output": {"frame_count": 1}}

    monkeypatch.setattr(tree, "preflight_prepared_tree", preflight)
    monkeypatch.setattr(tree, "_sealed_proof_sha256", sealed_proof)
    monkeypatch.setattr(tree, "run_prepared_tree", run_prepared_tree)
    # The stand-in domains carry no physics to warn about.
    monkeypatch.setattr(tree, "experimental_selection_sentence",
                        lambda runs: None)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = tmp_path / "tree.toml"
    config.write_text('[experiment]\nname = "tree-door"\n', encoding="utf-8")
    outdir = tmp_path / "run"
    checkpoint = tmp_path / "earlier" / "gpuwmrst_d01"
    code = tree.main([
        "--prepared-root", str(prepared),
        "--prepared-head-sha256", "d" * 64,
        "--experiment-config", str(config),
        "--experiment-config-sha256", "b" * 64,
        "--outdir", str(outdir), "--restart", str(checkpoint),
    ])
    err = capsys.readouterr().err

    assert code == 0, err
    assert "forecast starts after" not in err
    assert [("prepared_head_sha256" in b, "preparation_receipt_sha256" in b)
            for b in bindings] == [(True, False)]
    assert runs == [(head_inputs, checkpoint)]
    assert not (outdir / "evidence" / "failed-run-receipt.json").exists()


def test_a_plain_tree_starts_at_its_head():
    exp = SimpleNamespace(domains=(SimpleNamespace(grid_id=1, tiles=None),),
                          tiles=None)
    assert tree._head_needs_seal(exp, relocation_follow=False) is None


def test_the_root_host_store_loads_the_head_and_takes_later_tables_at_seams(
        tmp_path, monkeypatch):
    """Real head/cache payloads and real lazy interval reads, without a GPU.

    Only slab construction and pinned allocation stand in for the device.
    The actual loader must verify/read the head's initial arrays without a
    sealed cache header, then keep the same checked boundary source through
    a later wait.  An eager future-table read fails rather than hanging.
    """

    from dataclasses import dataclass, replace

    import cupy as cp
    import numpy as np

    from gpuwm.core import resident_admission, streaming
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest import prepared_cache, prepared_store
    from tilestream import driver, hoststore, physics_inventory, realdata

    @dataclass(frozen=True)
    class Config:
        nz: int = 11
        ny: int = 13
        nx: int = 17

    cfg = Config()
    initial = _initial()
    initial.state.u = np.arange(
        cfg.nz * cfg.ny * (cfg.nx + 1), dtype=np.float32).reshape(
            cfg.nz, cfg.ny, cfg.nx + 1)
    initial.coord = make_vertical_coord(cfg.nz, hybrid_opt=0)
    mass = np.full((cfg.ny, cfg.nx), 90_000.0)
    initial.base = replace(
        initial.base, mub=mass, pb=np.full((cfg.nz, *mass.shape), 50_000.0),
        alb=np.full((cfg.nz, *mass.shape), 0.8),
        thb=np.full((cfg.nz, *mass.shape), 290.0),
        phb=np.zeros((cfg.nz + 1, *mass.shape)),
        terrain_z=np.zeros_like(mass))
    initial.surface_pressure = np.full(mass.shape, 99_000.0)
    initial.surface_qv = np.full(mass.shape, 0.01)
    met = SimpleNamespace(fields={"T2": np.full(mass.shape, 279.0)})
    staging = tmp_path / ".tmp-store-head"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "store-head",
        identity={"source": "tree-consumer-test"}, cache_name=TREE_CACHE,
        chained=True)
    snapshots = _snapshots(nz=cfg.nz, ny=cfg.ny, nx=cfg.nx)
    times = _times(len(snapshots))
    frames = _SnapshotFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    writer.write_head(
        initial_result=initial, met=met, lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[0.0, 3600.0], [3600.0, 7200.0]],
            "fields": frames.inventory}, proof_head={"schema": "test"},
        forcing=frames,
        tree=domain_tree_head_fields(["d01", "d02"], root_cache=TREE_CACHE))
    head = boundary_stream.read_head(writer.root)
    reader = prepared_cache.PreparedHeadReader(
        writer.root, head, expected_identity={"source": "tree-consumer-test"})
    checked = []
    source = boundary_stream.streamed_boundaries(
        writer.root, head=head,
        validate=lambda interval: checked.append(interval.start_seconds))

    def no_seal_reader(*_args, **_kwargs):
        pytest.fail("the root store opened its sealed cache before the seal")

    def slab(cfg_slab, coord, base_slab, static_slab, cache_rows, names):
        assert names == ["u"]
        return SimpleNamespace(u=np.array(cache_rows("u"), copy=True))

    monkeypatch.setattr(prepared_cache, "PreparedCacheReader", no_seal_reader)
    monkeypatch.setattr(prepared_store, "_boundaries_from_cache", no_seal_reader)
    monkeypatch.setattr(prepared_store, "_slab_state", slab)
    monkeypatch.setattr(resident_admission, "admitted_slab_rows",
                        lambda cfg, rows, **_: rows)
    monkeypatch.setattr(streaming, "prime_lazy_carriers", lambda *_: None)
    monkeypatch.setattr(driver, "geography_inventory", lambda state: {
        "terrain": np.zeros((state.u.shape[-2], cfg.nx), np.float32)})
    monkeypatch.setattr(physics_inventory, "carrier_scalars", lambda _: {})
    monkeypatch.setattr(realdata, "window_grid", lambda grid, *_: grid)
    monkeypatch.setattr(hoststore, "check_allocatable", lambda *_a, **_k: None)
    monkeypatch.setattr(hoststore, "alloc_pinned_array",
                        lambda shape, dtype: np.empty(shape, dtype))
    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: SimpleNamespace(
        free_all_blocks=lambda: None))
    store = prepared_store.store_from_prepared_cache(
        reader.path, expected_identity={"source": "tree-consumer-test"},
        cfg=cfg, static={}, landuse_attrs={},
        grid=SimpleNamespace(ref_lat=0.0), valid_time=times[0],
        rows_per_slab=4, reader=reader, boundary_source=source,
        inventory_fn=lambda state, _: {"state/u": state.u},
        physics_initializer=lambda *_a, **_k: None, log=lambda *_: None)
    assert store.store["state/u"].tobytes() == initial.state.u.tobytes()
    assert store.receipt["slabs"] == 4
    assert store.receipt["content_sha256"] is None
    assert store.boundaries is source
    assert source.intervals._loaded == {}
    assert checked == []
    assert not (reader.path / "header.json").exists()
    assert not (writer.root / "proof.json").exists()

    frames.add_snapshot(snapshots[1], index=1)
    expected_first = frames.interval(0, times)
    writer.write_segment(0, expected_first)
    first = store.boundaries.interval_at(0.0)
    assert list(source.intervals._loaded) == [0]
    assert checked == [0.0]
    assert source.intervals.ready_prefix() == 1

    waits = []
    expected_second = None

    def publish_at_the_seam(report):
        nonlocal expected_second
        waits.append(report)
        if report is not None:
            assert report["interval"] == 1
            frames.add_snapshot(snapshots[2], index=2)
            expected_second = frames.interval(1, times)
            writer.write_segment(1, expected_second)

    source.intervals.on_wait = publish_at_the_seam
    second = store.boundaries.interval_at(3600.0)
    assert [report["interval"] for report in waits if report is not None] == [1]
    assert waits[-1] is None
    assert checked == [0.0, 3600.0]
    assert not (writer.root / "proof.json").exists()
    for actual, expected in ((first, expected_first), (second, expected_second)):
        assert set(actual.fields) == set(expected.fields)
        for name in actual.fields:
            for side in ("west", "east", "south", "north"):
                got = getattr(actual.fields[name], side)
                want = getattr(expected.fields[name], side)
                assert got.value.tobytes() == want.value.tobytes()
                assert got.tendency.tobytes() == want.tendency.tobytes()
    writer.fail(RuntimeError("test over"))


# ---------------------------------------------------------------------------
# The clock guard
# ---------------------------------------------------------------------------


def _clock(dt="15", division=1, sound=4):
    return {"domains": [{"grid_id": 1, "dt_s": dt, "step_division": division,
                         "time_step_sound": sound}]}


def test_the_derived_clock_ignores_the_reading_and_keeps_the_step():
    reading = _clock()
    reading["domains"][0]["crest_level_wind_m_s"] = 40.0
    assert derived_clock(reading) == derived_clock(_clock())
    assert derived_clock(_clock(dt="10")) != derived_clock(_clock())


def test_an_interval_that_moves_the_clock_ends_the_attempt(monkeypatch):
    from gpuwm import terrain_clock

    guard = StreamedClockGuard.__new__(StreamedClockGuard)
    guard.basis = SimpleNamespace(experiment=None, acoustic=(), statics={},
                                  reach={})
    guard.root = 1
    guard.run_seconds = 7200.0
    guard.expected = derived_clock(_clock())
    guard.starts = {}
    guard.geometry = object()
    guard.intervals = {}
    guard.checked = 0
    answers = iter([_clock(), _clock(dt="10", division=2)])
    monkeypatch.setattr(terrain_clock, "BoundaryWinds",
                        lambda *args: SimpleNamespace())
    monkeypatch.setattr(terrain_clock, "clock_for_domains",
                        lambda *a, **k: (None, "clock"))
    monkeypatch.setattr(terrain_clock, "clock_receipt",
                        lambda clock: next(answers))
    interval = SimpleNamespace(start_seconds=0.0, end_seconds=3600.0,
                               fields={})
    guard(interval)
    assert guard.checked == 1
    later = SimpleNamespace(start_seconds=3600.0, end_seconds=7200.0,
                            fields={})
    with pytest.raises(StreamedClockChanged, match="d01"):
        guard(later)


def test_a_root_with_no_boundary_geometry_is_not_read(monkeypatch):
    guard = StreamedClockGuard.__new__(StreamedClockGuard)
    guard.geometry = None
    guard.checked = 0
    guard(SimpleNamespace(start_seconds=0.0, end_seconds=1.0, fields={}))
    assert guard.checked == 0


def test_the_clock_check_survives_a_later_attachment():
    seen = []
    intervals = SimpleNamespace(validate=None)

    def check(interval):
        seen.append(("clock", interval))

    keep_interval_check(intervals, check)
    # An attachment replaces the hook with its own layout check ...
    intervals.validate = lambda interval: seen.append(("layout", interval))
    keep_interval_check(intervals, check)
    intervals.validate("k")
    assert seen == [("layout", "k"), ("clock", "k")]
    # ... and chaining it again is a no-op.
    keep_interval_check(intervals, check)
    seen.clear()
    intervals.validate("j")
    assert seen == [("layout", "j"), ("clock", "j")]


class _Renders:
    """A stand-in for the runner's own every-frame render."""

    def __init__(self, outdir):
        self.outdir = outdir
        self.halted_before_set_aside = None

    def halt(self, timeout=None):
        self.halted_before_set_aside = not (
            self.outdir / tree.STREAMED_ATTEMPT_DIRNAME).exists()

    def render_threads(self):
        return []

    def wait(self):
        return None


@pytest.mark.parametrize("resumed", [False, True])
def test_a_moved_clock_runs_the_forecast_again_on_the_sealed_tree(
        tmp_path, monkeypatch, capsys, resumed):
    """The head's clock and the seal's differ, and the forecast still ends.

    The sealed tree derives its clock over every boundary interval, the
    head over its start states, so when a later interval moves the clock
    the two differ by construction.  The run used to go on through the
    end-of-run seal binding, whose last check requires the two to agree,
    and so failed on exactly the tree it was recovering.
    """

    import sys

    from gpuwm import capabilities, provenance_gate

    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *_: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    experiment = SimpleNamespace(domains=())
    head_inputs = SimpleNamespace(
        experiment=experiment, terrain_clock=_clock(),
        stream_head={"head_sha256": "d" * 64})
    sealed_inputs = SimpleNamespace(
        experiment=experiment, terrain_clock=_clock(dt="10", division=2),
        stream_head=None)
    assert derived_clock(sealed_inputs.terrain_clock) \
        != derived_clock(head_inputs.terrain_clock)
    bindings, seals, renders, runs, waits = [], [], [], [], []

    def preflight(**binding):
        bindings.append(binding)
        return (head_inputs if "prepared_head_sha256" in binding
                else sealed_inputs)

    def sealed_proof(root, head_sha256, *, on_wait=None):
        seals.append((Path(root), head_sha256))
        waits.append(on_wait)
        return "c" * 64

    def route_renders(args, *, outdir, observer, started):
        renders.append((_Renders(outdir), args.restart))
        return renders[-1][0]

    def run_prepared_tree(bound, *, output_directory, restart,
                          first_products=None, **_):
        runs.append((bound, restart, first_products, sys.exc_info()[1]))
        wrfout = output_directory / "wrfout"
        wrfout.mkdir()
        (wrfout / "wrfout_d01_2026-09-29_12_00_00").write_text(
            "attempt" if bound is head_inputs else "sealed")
        if bound is head_inputs:
            raise StreamedClockChanged(
                "boundary interval 3600 s to 7200 s carries a crest-level "
                "wind that moves the terrain-derived clock of d01")
        return {"status": "PASS", "readiness": "ready",
                "execution_plan": {"plan_id": "plan", "domain_count": 2},
                "wall_seconds": 1.0, "output": {"frame_count": 1}}

    monkeypatch.setattr(tree, "preflight_prepared_tree", preflight)
    monkeypatch.setattr(tree, "_sealed_proof_sha256", sealed_proof)
    monkeypatch.setattr(tree.prepared_single, "_route_owned_first_products",
                        route_renders)
    monkeypatch.setattr(tree, "run_prepared_tree", run_prepared_tree)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = tmp_path / "tree.toml"
    config.write_text('[experiment]\nname = "tree-door"\n', encoding="utf-8")
    outdir = tmp_path / "run"
    checkpoint = tmp_path / "earlier" / "gpuwmrst_d01"
    code = tree.main([
        "--prepared-root", str(prepared),
        "--prepared-head-sha256", "d" * 64,
        "--experiment-config", str(config),
        "--experiment-config-sha256", "b" * 64,
        "--outdir", str(outdir),
        "--render-products", "all",
        *(["--restart", str(checkpoint)] if resumed else []),
    ])
    err = capsys.readouterr().err

    assert code == 0, err
    # The sealed tree is bound as a launch on its proof binds it, after
    # the seal was checked against the head this run started from.
    assert seals == [(prepared, "d" * 64)]
    # The seal wait is said as a wait before the first step, as a seam
    # wait is (``_start_seal_waits``).
    assert [type(wait).__name__ for wait in waits] == ["SeamWaits"]
    assert [sorted(binding) for binding in bindings] == [
        ["experiment_config", "experiment_config_sha256",
         "prepared_head_sha256", "prepared_root"],
        ["experiment_config", "experiment_config_sha256",
         "preparation_receipt_sha256", "prepared_root"]]
    assert bindings[1]["preparation_receipt_sha256"] == "c" * 64
    (first, first_restart, first_renders, _), \
        (second, second_restart, second_renders, pending) = runs
    assert first is head_inputs and second is sealed_inputs
    assert second.terrain_clock == _clock(dt="10", division=2)
    # The second run starts after the handler let go of the attempt: no
    # exception (and with it no traceback holding the attempt's tree) is
    # still pending around it.
    assert pending is None
    # A resumed attempt's checkpoint ran on the head's clock, so the
    # sealed forecast runs from its start time and says so.
    assert first_restart == (checkpoint if resumed else None)
    assert second_restart is None
    assert ("the sealed forecast runs from its start time" in err) \
        == resumed
    # The attempt's render stopped before its folder moved, and the sealed
    # run draws its own frames.
    assert first_renders is renders[0][0]
    assert first_renders.halted_before_set_aside is True
    assert second_renders is renders[1][0] and renders[1][1] is None
    # The attempt is kept, named, beside the sealed forecast.
    attempt = outdir / tree.STREAMED_ATTEMPT_DIRNAME
    assert (attempt / "wrfout" / "wrfout_d01_2026-09-29_12_00_00"
            ).read_text() == "attempt"
    assert (outdir / "wrfout" / "wrfout_d01_2026-09-29_12_00_00"
            ).read_text() == "sealed"
    assert f"kept in {attempt}" in err
    assert not (outdir / "evidence" / "failed-run-receipt.json").exists()


def test_each_streamed_attempt_keeps_its_own_folder(tmp_path):
    outdir = tmp_path / "run"
    outdir.mkdir()
    for number in (1, 2):
        (outdir / "wrfout").mkdir()
        (outdir / "wrfout" / "frame").write_text(str(number))
        (outdir / "progress.json").write_text(str(number))
        tree._set_aside_streamed_attempt(outdir)
    assert sorted(entry.name for entry in outdir.iterdir()) == [
        "streamed-attempt", "streamed-attempt-2"]
    for name, number in (("streamed-attempt", "1"),
                         ("streamed-attempt-2", "2")):
        assert (outdir / name / "wrfout" / "frame").read_text() == number
        assert sorted(entry.name for entry in (outdir / name).iterdir()) \
            == ["progress.json", "wrfout"]


def test_the_set_aside_leaves_the_heartbeat_in_the_run_folder(tmp_path):
    from gpuwm.supervisor import HEARTBEAT_NAME

    outdir = tmp_path / "run"
    outdir.mkdir()
    (outdir / "wrfout").mkdir()
    (outdir / HEARTBEAT_NAME).write_text("worker")
    attempt = tree._set_aside_streamed_attempt(outdir)
    assert (outdir / HEARTBEAT_NAME).read_text() == "worker"
    assert sorted(entry.name for entry in attempt.iterdir()) == ["wrfout"]


# ---------------------------------------------------------------------------
# The rerun under `gpuwm go`'s forecast watchdog
# ---------------------------------------------------------------------------


class _WatchdogClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.mark.parametrize("seal_wait_seconds", [0.0, 200.0])
def test_a_moved_clock_rerun_passes_the_forecast_watchdog(
        tmp_path, monkeypatch, capsys, seal_wait_seconds):
    """`gpuwm go`'s own watchdog lets the rerun on the sealed tree finish.

    THE BREAKAGE (the L6 second repair's check): under `gpuwm go` the
    forecast runs in ``gpuwm.forecast_supervisor`` with a
    ``ForecastHeartbeat`` as its observer and a ``ForecastWatchdog``
    reading ``run-progress.json``.  The set-aside moved that file into
    ``streamed-attempt/`` and the seal wait before the rerun published
    nothing, so the watchdog saw the attempt's last ``integrating`` record
    stand still and stopped the worker at its 120 s step bound ("forecast
    stalled in integrating"); with the seal already there, the rerun's
    first beats (a restore phase, step 0) read as "status moved backward
    from integrating to preparing:...".  go exited the stage 124.

    Driven through ``forecast_supervisor.main`` as go launches it, with
    the watchdog checked after every beat and at every poll of a seal wait
    longer than the 120 s bound.
    """

    import os
    import sys
    import time as real_time

    from gpuwm import capabilities, forecast_supervisor, provenance_gate
    from gpuwm import supervisor

    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *_: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    root, _ = _tree_head(tmp_path)
    head_sha256 = json.loads(
        (boundary_stream.stream_dir(root) / boundary_stream.HEAD_NAME)
        .read_text(encoding="utf-8"))["head_sha256"]
    if not seal_wait_seconds:
        (root / boundary_stream.PROOF_NAME).write_text("{}")
    config = tmp_path / "tree.toml"
    config.write_text('[experiment]\nname = "tree-door"\n', encoding="utf-8")
    outdir = tmp_path / "run"
    argv = ["gpuwm.prepared_domain_tree_forecast",
            "--prepared-root", str(root),
            "--prepared-head-sha256", head_sha256,
            "--experiment-config", str(config),
            "--experiment-config-sha256", "b" * 64,
            "--outdir", str(outdir)]
    clock = _WatchdogClock()
    watchdog = forecast_supervisor.ForecastWatchdog(
        [sys.executable, "-m", *argv], clock=clock)
    assert watchdog.command[3:] == argv
    for name, value in watchdog.env.items():
        monkeypatch.setenv(name, value)
    records, verdicts = [], []

    def check():
        verdicts.append(watchdog.check(os.getpid()))

    class Heartbeat(forecast_supervisor.ForecastHeartbeat):
        def _write(self, status, **kwargs):
            super()._write(status, **kwargs)
            records.append(supervisor.read_heartbeat(self.path))
            check()

    monkeypatch.setattr(forecast_supervisor, "ForecastHeartbeat", Heartbeat)

    # The seal wait's clock is the watchdog's: every poll moves both by a
    # second, the watchdog is asked at each, and the producer seals once
    # the wait has lasted ``seal_wait_seconds``.
    seal_wait = {}

    class Time:
        def __getattr__(self, name):
            return getattr(real_time, name)

        def sleep(self, seconds):
            seal_wait.setdefault("started", clock.now)
            clock.now += 1.0
            if clock.now - seal_wait["started"] >= seal_wait_seconds:
                (root / boundary_stream.PROOF_NAME).write_text("{}")
            check()

    class Intervals(boundary_stream.StreamedIntervals):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("clock", clock)
            kwargs.setdefault("poll_seconds", 1.0)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(boundary_stream, "time", Time())
    monkeypatch.setattr(boundary_stream, "StreamedIntervals", Intervals)
    monkeypatch.setattr(boundary_stream, "verify_seal",
                        lambda root, *, head, **_: {"proof_sha256": "c" * 64})
    experiment = SimpleNamespace(domains=())
    head_inputs = SimpleNamespace(experiment=experiment,
                                  stream_head={"head_sha256": head_sha256})
    sealed_inputs = SimpleNamespace(experiment=experiment, stream_head=None)
    bindings = []

    def preflight(**binding):
        bindings.append(binding)
        return (head_inputs if "prepared_head_sha256" in binding
                else sealed_inputs)

    def run_prepared_tree(bound, *, output_directory, observer, **_):
        observer.preparing("restore-prepared-domain-tree")
        for step in range(6):
            clock.now += 1.0
            observer(model_elapsed_seconds=10.0 * step, outer_step=step,
                     last_durable_wrfout=None, last_checkpoint=None)
        (output_directory / "wrfout").mkdir()
        (output_directory / "wrfout" / "wrfout_d01").write_text(
            "attempt" if bound is head_inputs else "sealed")
        if bound is head_inputs:
            raise StreamedClockChanged(
                "boundary interval 3600 s to 7200 s moves the terrain-derived "
                "clock of d01")
        observer.finalizing("write-receipts")
        return {"status": "PASS", "readiness": "ready",
                "execution_plan": {"plan_id": "plan", "domain_count": 2},
                "wall_seconds": 1.0, "output": {"frame_count": 1}}

    monkeypatch.setattr(tree, "preflight_prepared_tree", preflight)
    monkeypatch.setattr(tree, "run_prepared_tree", run_prepared_tree)

    code = forecast_supervisor.main(argv)
    check()
    err = capsys.readouterr().err

    assert code == 0, err
    assert [verdict for verdict in verdicts if verdict is not None] == []
    assert bindings[-1]["preparation_receipt_sha256"] == "c" * 64
    statuses = [(record.status, None if record.restart is None
                 else record.restart["attempt"]) for record in records]
    restart = statuses.index(("preparing:restart", 2))
    # Every record before the restart is the first attempt's, and every
    # one from it on declares the second.
    assert all(attempt is None for _, attempt in statuses[:restart])
    assert all(attempt == 2 for _, attempt in statuses[restart:])
    assert statuses[restart - 1] == ("integrating", None)
    assert records[restart].outer_step == 0
    assert records[restart].model_elapsed_seconds == 0.0
    assert "clock of d01" in records[restart].restart["reason"]
    waited = [status for status, _ in statuses[restart:]
              if status.startswith("waiting:")]
    if seal_wait_seconds:
        # The seal wait is said on the heartbeat, refreshed as it lasts.
        assert set(waited) == {"waiting:preparation"}
        assert len(waited) >= seal_wait_seconds / 5.0 - 1
        assert clock.now - seal_wait["started"] >= seal_wait_seconds
    else:
        assert waited == []
    assert statuses[-1] == ("complete", 2)
    assert watchdog.last.status == "complete"
    # The heartbeat never left the run folder; the attempt's outputs did.
    attempt = outdir / tree.STREAMED_ATTEMPT_DIRNAME
    assert (outdir / supervisor.HEARTBEAT_NAME).is_file()
    assert not (attempt / supervisor.HEARTBEAT_NAME).exists()
    assert (attempt / "wrfout" / "wrfout_d01").read_text() == "attempt"
    assert (outdir / "wrfout" / "wrfout_d01").read_text() == "sealed"


@pytest.mark.parametrize("moved", [False, True])
def test_a_source_behind_during_a_seal_wait_exits_75_by_name(
        tmp_path, monkeypatch, capsys, moved):
    """A lead past its late time while the forecast waits for the seal.

    Before its first step (a head that needs the seal) or before the rerun
    on the sealed tree: the producer's ``source_behind`` ends the run with
    exit 75 naming the lead, as it does at a seam.  THE BREAKAGE: the seal
    wait turned every producer verdict into a refusal (exit 2) or, before
    the rerun, a crash (exit 1), so a site read a late source as a broken
    configuration.
    """

    from gpuwm import capabilities, provenance_gate

    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *_: None)
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    root, _ = _tree_head(tmp_path)
    head_sha256 = json.loads(
        (boundary_stream.stream_dir(root) / boundary_stream.HEAD_NAME)
        .read_text(encoding="utf-8"))["head_sha256"]
    (boundary_stream.stream_dir(root) / boundary_stream.FAILED_NAME
     ).write_text(json.dumps({
         "code": boundary_stream.SOURCE_BEHIND_CODE,
         "reason": "gefs f030 not posted",
         "details": {"source": "gefs", "cycle": "2026-09-30T12", "lead": 30,
                     "late_after_minutes": 60}}), encoding="utf-8")
    experiment = SimpleNamespace(domains=())
    head_inputs = SimpleNamespace(experiment=experiment,
                                  stream_head={"head_sha256": head_sha256})

    def preflight(**binding):
        if not moved and "prepared_head_sha256" in binding:
            raise tree.TreeHeadNeedsSeal("a nest follows a storm")
        return head_inputs

    def run_prepared_tree(bound, *, output_directory, **_):
        raise StreamedClockChanged("the clock of d01 moved")

    heard = []
    observer = SimpleNamespace(source_behind=heard.append,
                               restarting=lambda why: None,
                               preparing=lambda phase: None)
    monkeypatch.setattr(tree, "preflight_prepared_tree", preflight)
    monkeypatch.setattr(tree, "run_prepared_tree", run_prepared_tree)
    config = tmp_path / "tree.toml"
    config.write_text('[experiment]\nname = "tree-door"\n', encoding="utf-8")
    outdir = tmp_path / "run"
    code = tree.main([
        "--prepared-root", str(root), "--prepared-head-sha256", head_sha256,
        "--experiment-config", str(config),
        "--experiment-config-sha256", "b" * 64, "--outdir", str(outdir)],
        observer=observer)
    err = capsys.readouterr().err

    assert code == boundary_stream.SOURCE_BEHIND_EXIT_CODE, err
    assert "gefs f030" in err
    assert [record["lead"] for record in heard] == [30]
    assert (outdir / "evidence" / "failed-run-receipt.json").is_file()


def test_a_render_that_outlives_its_halt_is_killed_and_waited_for():
    """No render of the set-aside attempt still runs when its folder moves.

    A halt waits a bounded time and can return with its render drawing
    (on Windows it ended ``gpuwm render`` alone); a folder with a picture
    open in it does not move there.  Here the halt ends nothing, and
    :func:`gpuwm.first_products.halt_renders_and_wait` still returns only
    once the render process has exited and its thread has returned.
    """

    import sys
    import threading
    import time

    from gpuwm import first_products

    ended = first_products.WorkerEnd()

    def draw():
        first_products._run_render(
            [sys.executable, "-c", "import time; time.sleep(120)"])

    thread = threading.Thread(target=ended.run, args=(draw,), daemon=True)
    thread.start()
    process = None
    deadline = time.monotonic() + 30.0
    while process is None and time.monotonic() < deadline:
        with first_products._RUNNING_LOCK:
            process = first_products._RUNNING.get(thread.ident)
        time.sleep(0.02)
    assert process is not None and process.poll() is None

    class Renders:
        halts = []

        def halt(self, timeout):
            self.halts.append(timeout)

        def render_threads(self):
            return [(thread, ended)]

    started = time.monotonic()
    assert first_products.halt_renders_and_wait(
        Renders(), timeout=0.1, reap_seconds=20.0) is True
    assert Renders.halts == [0.1]
    assert process.poll() is not None and ended.ended
    assert time.monotonic() - started < 20.0


# ---------------------------------------------------------------------------
# The restart identity a chained tree shares across both bindings
# ---------------------------------------------------------------------------


def test_a_chained_tree_is_one_restart_identity_under_both_bindings(
        tmp_path):
    from gpuwm.experiment import build_experiment

    from test_checkpoint_route_contract import _raw, _wizard_config

    exp = build_experiment(_raw(_wizard_config(tmp_path, ladder="12-3")),
                           source="components-test")
    plan = MappingProxyType({"plan_id": "p", "edges": ()})

    def inputs(root_content, preparation):
        return SimpleNamespace(
            experiment=exp, prepared_head_sha256="f" * 64,
            authority_sha256=MappingProxyType(
                {"preparation_receipt": preparation}),
            domains=(
                SimpleNamespace(grid_id=1, cache_reader=SimpleNamespace(
                    content_sha256=root_content)),
                SimpleNamespace(grid_id=2, cache_reader=SimpleNamespace(
                    content_sha256="beef"))),
            execution_plan=plan)

    runtime = MappingProxyType({"runtime": "x"})
    # At the head the root has no content digest and there is no proof;
    # after the seal both exist.  The identity is the same.
    at_head = tree.tree_restart_identity_components(
        inputs(None, "unused"), runtime)
    sealed = tree.tree_restart_identity_components(
        inputs("0" * 64, "1" * 64), runtime)
    assert at_head == sealed
    assert set(at_head) == set(tree.CHAINED_TREE_RESTART_IDENTITY_COMPONENTS)
    assert at_head["prepared_head_sha256"] == "f" * 64
    assert at_head["domain_cache_content_sha256"] == {"d02": "beef"}
    # A tree prepared in one piece keeps the identity it always had.
    one_shot = SimpleNamespace(**{**vars(inputs("0" * 64, "1" * 64)),
                                  "prepared_head_sha256": None})
    components = tree.tree_restart_identity_components(one_shot, runtime)
    assert set(components) == set(tree.TREE_RESTART_IDENTITY_COMPONENTS)


def test_an_as_posted_tree_is_one_restart_identity_under_both_bindings(
        tmp_path):
    """The children an as-posted head prepared are not its sealed children.

    An as-posted GFS tree's head prepares every child under the input plan
    and its seal writes each child again under the manifest digest, so the
    head's and the sealed child's content digests differ (verify_seal holds
    them equal in everything else).  A forecast bound to the head wrote its
    checkpoints naming the head's children; ``gpuwm go --prepared-root P
    --restart CKPT`` binds the seal and must name the same ones
    (``head_child_content_sha256``), or it refuses every checkpoint of a
    crashed default GFS tree run as "written for a different run".
    """

    from gpuwm.experiment import build_experiment

    from test_checkpoint_route_contract import _raw, _wizard_config

    exp = build_experiment(_raw(_wizard_config(tmp_path, ladder="12-3")),
                           source="components-test")
    plan = MappingProxyType({"plan_id": "p", "edges": ()})

    def inputs(root_content, child_content, head_children=None):
        return SimpleNamespace(
            experiment=exp, prepared_head_sha256="f" * 64,
            authority_sha256=MappingProxyType(
                {"preparation_receipt": "1" * 64}),
            domains=(
                SimpleNamespace(grid_id=1, cache_reader=SimpleNamespace(
                    content_sha256=root_content)),
                SimpleNamespace(grid_id=2, cache_reader=SimpleNamespace(
                    content_sha256=child_content))),
            execution_plan=plan,
            head_child_content_sha256=head_children)

    runtime = MappingProxyType({"runtime": "x"})
    at_head = tree.tree_restart_identity_components(
        inputs(None, "beef"), runtime)
    sealed = tree.tree_restart_identity_components(
        inputs("0" * 64, "cafe", MappingProxyType({"d02": "beef"})), runtime)
    assert at_head == sealed
    assert sealed["domain_cache_content_sha256"] == {"d02": "beef"}
    # Without the head's children the sealed binding names its own, which
    # is the refusal this repairs.
    unpaired = tree.tree_restart_identity_components(
        inputs("0" * 64, "cafe"), runtime)
    assert unpaired["domain_cache_content_sha256"] == {"d02": "cafe"}
    assert unpaired != at_head


# ---------------------------------------------------------------------------
# The doors: run-plan and go send trees through run_chained
# ---------------------------------------------------------------------------


def _body(path, name):
    root = Path(__file__).resolve().parents[1]
    text = (root / path).read_text(encoding="utf-8")
    start = text.index(f"def {name}(")
    end = text.find("\ndef ", start + 1)
    return text[start:end if end > 0 else None]


def test_the_staged_chain_starts_a_tree_on_its_head():
    staged = _body("gpuwm/runplan.py", "_staged_chain")
    assert "len(exp.domains) == 1 and not prepare_only" not in staged
    assert "run_chained(" in staged
    assert "domain_tree_forecast" not in staged
    # The reason that said the tree runner binds only a sealed tree is
    # retired with the defect it named.
    assert "domain_tree_forecast" not in boundary_stream.SEALED_REASONS


def test_go_runs_its_tree_arm_through_run_chained():
    # One domain or a tree, go's GFS-series chain is one run_chained call:
    # the tree-only arm that said the gfs_domain_tree sealed reason went
    # when a GFS tree started chaining (A136 L7b).
    body = _body("gpuwm/go_cli.py", "_go_prepared_main")
    assert "say_prepared_sealed(" not in body
    arm = body[body.index(
        "from gpuwm.ingest.boundary_stream import run_chained"):]
    arm = arm[:arm.index("rendered = _render_stage")]
    assert "run_chained(" in arm
    assert "forecast(None)" not in arm


def test_go_binds_a_tree_head_with_its_digest(tmp_path):
    from gpuwm import go_cli

    authority = tmp_path / "authority"
    authority.mkdir()
    (authority / "experiment.toml").write_text("[experiment]\n",
                                               encoding="utf-8")
    plan = {"prepared": tmp_path / "prepared", "authority": authority,
            "runner": tree.__name__, "run": tmp_path / "run",
            "render": tmp_path / "png"}
    command = go_cli.tree_forecast_command(
        plan, prepared_head_sha256="e" * 64)
    assert command[command.index("--prepared-head-sha256") + 1] == "e" * 64
    assert "--preparation-receipt-sha256" not in command


# ---------------------------------------------------------------------------
# A delayed nest's start lead is a start need
# ---------------------------------------------------------------------------


def _delayed_exp(start_times, *, cadence_hours=1):
    from datetime import timedelta

    start = datetime(2026, 9, 29, 12)
    domains = [SimpleNamespace(grid_id=1, parent_id=0)] + [
        SimpleNamespace(grid_id=index + 2, parent_id=1)
        for index in range(len(start_times))]
    offsets = {1: 0.0, **{index + 2: float(value * 3600)
                          for index, value in enumerate(start_times)}}
    return SimpleNamespace(
        start_time=start, domains=tuple(domains),
        domain_start_offset_exact=lambda gid: offsets[int(gid)],
        domain_start_time=lambda gid: start + timedelta(
            seconds=offsets[int(gid)]))


def test_a_delayed_nest_adds_its_start_lead():
    from gpuwm.source_posting import nest_start_needs

    exp = _delayed_exp([0, 1])
    needs = nest_start_needs(exp, source="hrrr-prs",
                             cycle=datetime(2026, 9, 29, 12), start_lead=0,
                             cadence_hours=1)
    assert needs == [{"role": "nest_start:d03", "source": "hrrr-prs",
                      "lead": 1, "valid_time": "2026-09-29T13:00:00Z"}]


def test_nests_that_start_with_the_root_add_nothing():
    from gpuwm.source_posting import nest_start_needs

    exp = _delayed_exp([0, 0])
    assert nest_start_needs(exp, source="gefs",
                            cycle=datetime(2026, 9, 29, 12), start_lead=6,
                            cadence_hours=3) == []


def test_a_delayed_nest_between_two_leads_is_refused_by_name():
    from gpuwm.source_posting import nest_start_needs

    exp = _delayed_exp([4])
    with pytest.raises(ValueError, match="d02 starts 4 h .* has to start on a lead"):
        nest_start_needs(exp, source="gefs", cycle=datetime(2026, 9, 29, 12),
                         start_lead=0, cadence_hours=3)


def test_several_delayed_nests_on_one_parent_each_add_their_lead():
    from gpuwm.source_posting import nest_start_needs

    exp = _delayed_exp([2, 1, 0])
    needs = nest_start_needs(exp, source="hrrr-prs",
                             cycle=datetime(2026, 9, 29, 11), start_lead=1)
    assert [(need["role"], need["lead"]) for need in needs] == [
        ("nest_start:d03", 2), ("nest_start:d02", 3)]
