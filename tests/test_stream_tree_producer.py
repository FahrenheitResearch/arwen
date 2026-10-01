"""The producer side of a chained domain tree.

A domain tree is chained like a single domain: its head carries the
children (they need only the start time) and the root's start state, one
segment per root interval follows, and the seal writes the one-shot tree.

CPU only; no device, no source data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.grid import BaseState, make_vertical_coord
from gpuwm.ingest import boundary_stream
from gpuwm.ingest.boundary_stream import (
    HIERARCHY_HEAD_DIRNAME, LAYOUT_DOMAIN_TREE, SEAL_ONLY_PROOF_KEYS,
    PreparedTreeWriter, domain_tree_head_fields, read_head,
)
from gpuwm.ingest.lateral_bc import StateBoundaryFrames
from gpuwm.io.restart import STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS


FIELDS = ("u", "v", "theta", "phi", "mu")
TREE_CACHE = f"{HIERARCHY_HEAD_DIRNAME}/domains/d01/prepared-cache"
PROOF_HEAD = {"schema": "test-tree-proof", "forcing_hours": [0, 1, 2]}


def _snapshots(count=3, seed=20260929, nz=3, ny=12, nx=14):
    rng = np.random.default_rng(seed)
    return [{name: rng.standard_normal((nz, ny, nx)) for name in FIELDS}
            for _ in range(count)]


def _times(count):
    start = datetime(2026, 9, 29, 18)
    return [start + timedelta(hours=n) for n in range(count)]


def _initial():
    state = SimpleNamespace(u=np.arange(12, dtype=np.float32).reshape(3, 2, 2))
    for index, name in enumerate(STATE_SETUP_ARRAYS):
        setattr(state, name, np.array([index], dtype=np.float32))
    for name, value in {
            "mub": None, "p_top": 10_000.0, "cf1": 1.0, "cf2": 2.0,
            "cf3": 3.0, "cfn": 4.0, "cfn1": 5.0, "has_msf": True,
            "rotational": True}.items():
        setattr(state, name, value)
    assert set(STATE_SETUP_SCALARS) <= {
        "mub", "p_top", "cf1", "cf2", "cf3", "cfn", "cfn1", "has_msf",
        "rotational"}
    state.lateral_boundaries = None
    coord = make_vertical_coord(2, hybrid_opt=0)
    base = BaseState(
        mub=np.full((2, 2), 90_000.0), p_top=10_000.0,
        pb=np.full((2, 2, 2), 50_000.0), alb=np.full((2, 2, 2), 0.8),
        thb=np.full((2, 2, 2), 290.0), phb=np.zeros((3, 2, 2)),
        terrain_z=np.zeros((2, 2)))
    return SimpleNamespace(
        state=state, coord=coord, base=base,
        surface_pressure=np.full((2, 2), 99_000.0),
        surface_qv=np.full((2, 2), 0.01))


def _met():
    surface = np.ones((2, 2), dtype=np.float32)
    return SimpleNamespace(fields={
        "LANDSEA": surface, "SKINTEMP": 280.0 * surface,
        "SOILT": np.ones((9, 2, 2), dtype=np.float32),
        "SOILW": np.full((9, 2, 2), 0.2, dtype=np.float32),
        "T2": 279.0 * surface,
        "U10": np.ones((2, 3), dtype=np.float32),
        "V10": np.ones((3, 2), dtype=np.float32),
    })


class _SnapshotFrames(StateBoundaryFrames):
    """The route's accumulator, fed plain snapshots for a CPU test."""

    def add_state(self, state, *, index=None):
        self.add_snapshot(state, index=index)


def _head(tmp_path, *, name="tree", tree=True, chained=True, snapshots=None,
          proof_head=PROOF_HEAD):
    snapshots = snapshots or _snapshots()
    times = _times(len(snapshots))
    staging = tmp_path / f".tmp-{name}"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / name,
        identity={"source": "tree-producer-test"},
        cache_name=TREE_CACHE if tree else "prepared-cache",
        chained=chained)
    frames = _SnapshotFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[seconds[k], seconds[k + 1]]
                         for k in range(len(times) - 1)],
            "fields": frames.inventory},
        proof_head=proof_head, input_manifest_sha256="a" * 64,
        forcing=frames,
        tree=(domain_tree_head_fields(["d01", "d02"], root_cache=TREE_CACHE)
              if tree else None))
    return writer, frames, snapshots, times


# ---------------------------------------------------------------------------
# The head of a tree
# ---------------------------------------------------------------------------


def test_a_tree_head_names_its_layout_and_the_digest_binds_it(tmp_path):
    writer, *_ = _head(tmp_path)
    head = read_head(writer.root)
    assert head["layout"] == LAYOUT_DOMAIN_TREE
    assert head["domains"] == ["d01", "d02"]
    assert head["children_artifacts"] == HIERARCHY_HEAD_DIRNAME
    assert head["basis"]["tree"]["root"]["prepared_cache"] == TREE_CACHE
    assert head["basis"]["cache"]["directory"] == TREE_CACHE
    assert (writer.root / TREE_CACHE).is_dir()
    # The top-level copy is for readers; the digest binds basis.tree.
    path = writer.root / "boundary-stream" / "head.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["basis"]["tree"]["domains"] = ["d01"]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(boundary_stream.BoundaryStreamError, match="digest"):
        read_head(writer.root)
    writer.fail(RuntimeError("test over"))


def test_a_single_domain_head_basis_is_what_it_was(tmp_path):
    writer, *_ = _head(tmp_path, tree=False)
    head = read_head(writer.root)
    assert set(head["basis"]) == {
        "schema", "cache", "proof_head", "input_manifest_sha256"}
    assert "layout" not in head
    writer.fail(RuntimeError("test over"))


def test_a_tree_head_needs_the_root_and_a_child():
    with pytest.raises(ValueError, match="d01 and at least one child"):
        domain_tree_head_fields(["d01"], root_cache=TREE_CACHE)
    with pytest.raises(ValueError, match="d01 and at least one child"):
        domain_tree_head_fields(["d02", "d01"], root_cache=TREE_CACHE)


def test_a_tree_head_binds_each_childs_receipt_digest():
    fields = domain_tree_head_fields(
        ["d01", "d02", "d03"], root_cache=TREE_CACHE,
        children_receipts={"d03": "c" * 64, "d02": "b" * 64})
    # Sorted, so the head digest does not depend on the build order.
    assert list(fields["children_receipts"]) == ["d02", "d03"]
    assert fields["children_receipts"]["d02"] == "b" * 64
    assert "children_receipts" not in domain_tree_head_fields(
        ["d01", "d02"], root_cache=TREE_CACHE)


def test_the_tree_seal_keys_are_seal_only_for_every_head():
    # A tree's one-shot artifact tree and its WRF hierarchy exist only at
    # the seal; no single-domain proof carries them.
    assert {"artifact_receipt", "wrf_manifest"} <= SEAL_ONLY_PROOF_KEYS
    # The composition receipt binds inputs that all exist at the head.
    assert "source_composition" not in SEAL_ONLY_PROOF_KEYS
    # So does the statics corridor (A136 L7d): it needs no boundary time,
    # so a chained tree builds it into its head, whose proof binds it, and
    # a moving nest's forecast starts there instead of after the seal.
    assert "statics_corridor" not in SEAL_ONLY_PROOF_KEYS


@pytest.mark.parametrize("key", ["artifact_receipt", "wrf_manifest"])
def test_a_tree_head_refuses_a_proof_key_only_its_seal_knows(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        _head(tmp_path, proof_head={**PROOF_HEAD, key: {}})


def test_a_tree_seal_keeps_the_corridor_its_head_bound(tmp_path):
    """A moving tree's head binds its corridor; the seal may not change it.

    The breakage this prevents: a forecast started on the head moves its
    nest over the head's corridor, so a sealed tree whose corridor is
    another set (or one the head never bound) would not be the tree that
    forecast ran on.
    """

    corridor = {"schema": "gpuwm-statics-corridor-set-v1",
                "status": "READY", "domains": {"d02": {"cache": {
                    "path": "d02.npz", "bytes": 1, "sha256": "e" * 64}}}}
    writer, frames, snapshots, times = _head(
        tmp_path, proof_head={**PROOF_HEAD, "statics_corridor": corridor})
    assert read_head(writer.root)["basis"]["proof_head"][
        "statics_corridor"] == corridor
    for index in range(1, len(snapshots)):
        frames.add_snapshot(snapshots[index], index=index)
        writer.write_segment(index - 1, frames.interval(index - 1, times))
    writer.seal_cache()
    sealed = {**PROOF_HEAD, "artifact_receipt": {}, "wrf_manifest": {},
              "boundary_stream": writer.boundary_stream_proof()}
    other = {**corridor, "domains": {"d02": {"cache": {
        "path": "d02.npz", "bytes": 1, "sha256": "f" * 64}}}}
    with pytest.raises(RuntimeError, match="statics_corridor"):
        writer.publish({**sealed, "statics_corridor": other})
    with pytest.raises(RuntimeError, match="statics_corridor"):
        writer.publish(sealed)
    writer.publish({**sealed, "statics_corridor": corridor})
    assert json.loads((writer.root / "proof.json").read_text(
        encoding="utf-8"))["statics_corridor"] == corridor


def test_a_tree_seal_adds_its_artifact_records_to_the_head_proof(tmp_path):
    writer, frames, snapshots, times = _head(tmp_path)
    for index in range(1, len(snapshots)):
        frames.add_snapshot(snapshots[index], index=index)
        writer.write_segment(index - 1, frames.interval(index - 1, times))
    writer.seal_cache()
    # Anything else that differs from the head is refused by name.
    with pytest.raises(RuntimeError, match="forcing_hours"):
        writer.publish({**PROOF_HEAD, "forcing_hours": [0, 1],
                        "artifact_receipt": {}, "wrf_manifest": {},
                        "boundary_stream": writer.boundary_stream_proof()})
    # A statics corridor the head did not bind is refused too: a moving
    # tree binds it at the head (A136 L7d).
    with pytest.raises(RuntimeError, match="statics_corridor"):
        writer.publish({**PROOF_HEAD, "artifact_receipt": {},
                        "wrf_manifest": {}, "statics_corridor": {},
                        "boundary_stream": writer.boundary_stream_proof()})
    writer.publish({**PROOF_HEAD, "artifact_receipt": {},
                    "wrf_manifest": {},
                    "boundary_stream": writer.boundary_stream_proof()})
    assert (writer.root / "proof.json").is_file()


def test_the_children_add_their_caches_to_the_forecast_host_price(
        tmp_path, monkeypatch):
    seen = []

    def price(**kwargs):
        seen.append(kwargs["head_payload_bytes"])
        return {"total_bytes": 0}

    monkeypatch.setattr(boundary_stream, "forecast_installed", lambda: True)
    monkeypatch.setattr(boundary_stream, "forecast_host_bytes", price)
    monkeypatch.setattr(boundary_stream, "host_admission",
                        lambda **_: {"admitted": True})
    staging = tmp_path / ".tmp-priced"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "priced",
        identity={"source": "tree-producer-test"}, cache_name=TREE_CACHE,
        chained=True)
    frames = _SnapshotFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(_snapshots()[0], index=0)
    head = writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[0.0, 3600.0]], "fields": frames.inventory},
        proof_head=PROOF_HEAD, forcing=frames,
        tree=domain_tree_head_fields(["d01", "d02"], root_cache=TREE_CACHE),
        extra_head_payload_bytes=12345)
    assert head
    cache_bytes = int(writer.head["basis"]["cache"]["payload_bytes"])
    assert seen == [cache_bytes + 12345]
    writer.fail(RuntimeError("test over"))


# ---------------------------------------------------------------------------
# The stream loop and the seal's boundary set
# ---------------------------------------------------------------------------


def test_the_sealed_root_cache_reads_back_the_whole_boundary_set(tmp_path):
    # The stream holds no interval after writing it: the tree's seal reads
    # the root's boundary set back from the sealed cache (TreeStartStates),
    # and that read equals the whole-set build the one-shot tree embeds.
    from gpuwm.ingest.prepared_cache import (
        PreparedCacheReader, _reader_boundaries)

    snapshots = _snapshots(4)
    writer, frames, _snapshots_, times = _head(tmp_path, snapshots=snapshots)
    writer.stream_forcing_times(
        count=4, forcing=frames, times=times,
        build_forcing_time=lambda k: (None, SimpleNamespace(
            state=snapshots[k])))
    writer.seal_cache()
    read = _reader_boundaries(PreparedCacheReader(
        writer.cache_path,
        expected_identity={"source": "tree-producer-test"})).intervals
    whole = _SnapshotFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    for index, snapshot in enumerate(snapshots):
        whole.add_snapshot(snapshot, index=index)
    expected = whole.build(times).intervals
    assert len(read) == len(expected) == 3
    for got, want in zip(read, expected):
        assert (got.start_seconds, got.end_seconds) == (
            want.start_seconds, want.end_seconds)
        for name in FIELDS:
            for side in ("west", "east", "south", "north"):
                a = getattr(got.fields[name], side)
                b = getattr(want.fields[name], side)
                assert a.value.dtype == np.asarray(b.value).dtype
                assert np.array_equal(a.value, b.value)
                assert np.array_equal(a.tendency, b.tendency)
    writer.fail(RuntimeError("test over"))


def test_stream_forcing_times_holds_no_interval_for_the_seal():
    import inspect

    assert "keep" not in inspect.signature(
        PreparedTreeWriter.stream_forcing_times).parameters


# ---------------------------------------------------------------------------
# The start states between head and seal (TreeStartStates)
# ---------------------------------------------------------------------------


class _DeviceArray:
    """Stands in for a CuPy array: only the interface a release looks at."""

    __cuda_array_interface__ = {"shape": (2,), "typestr": "<f4",
                                "data": (0, False), "version": 3}


def _host_root(nz=3, ny=12, nx=14):
    """A real host start state with its boundary set, as a tree root has."""

    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    from gpuwm.ingest.lateral_bc import build_state_lateral_boundaries
    from gpuwm.state_serialization_contract import (
        STATE_DERIVED_SETUP_ARRAYS, STATE_SERIALIZED_ATTRS)

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0,
                    ztop=15000.0, dt=10.0, run_seconds=7200.0, moist=True,
                    mp_physics=6, terrain_opt=1)
    coord = make_vertical_coord(nz, hybrid_opt=0)
    base = BaseState(
        mub=np.full((ny, nx), 90_000.0), p_top=10_000.0,
        pb=np.linspace(95_000.0, 20_000.0, nz)[:, None, None]
        * np.ones((nz, ny, nx)),
        alb=np.full((nz, ny, nx), 0.9), thb=np.full((nz, ny, nx), 300.0),
        phb=np.linspace(0.0, 1.4e5, nz + 1)[:, None, None]
        * np.ones((nz + 1, ny, nx)),
        terrain_z=np.zeros((ny, nx)))
    static = {
        "MAPFAC_M": np.full((ny, nx), 1.01),
        "MAPFAC_U": np.full((ny, nx + 1), 1.01),
        "MAPFAC_V": np.full((ny + 1, nx), 1.01),
        "F": np.full((ny, nx), 1.0e-4), "E": np.full((ny, nx), 5.0e-5),
        "SINALPHA": np.full((ny, nx), 0.1),
        "COSALPHA": np.full((ny, nx), np.sqrt(0.99))}
    setup_only = set(STATE_SETUP_ARRAYS) | set(STATE_DERIVED_SETUP_ARRAYS)
    rng = np.random.default_rng(20260930)

    def prepared(offset):
        state = DomainState(cfg, array_module=np)
        state.load_base(coord, base)
        state.set_map_coriolis(
            static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
            static["F"], static["E"], sina=static["SINALPHA"],
            cosa=static["COSALPHA"])
        for name in STATE_SERIALIZED_ATTRS:
            array = getattr(state, name, None)
            if array is None or name in setup_only:
                continue
            array[...] = (offset + rng.standard_normal(array.shape)).astype(
                array.dtype)
        return state

    states = [prepared(float(k)) for k in range(3)]
    boundaries = build_state_lateral_boundaries(states, [0.0, 3600.0, 7200.0])
    surface = np.ones((ny, nx), dtype=np.float32)
    initial = SimpleNamespace(
        state=states[0], coord=coord, base=base,
        surface_pressure=np.full((ny, nx), 99_000.0),
        surface_qv=np.full((ny, nx), 0.01),
        # Arrays the writer never reads (a RealInitResult's dry_mass and
        # the like), on the card on a CUDA tree: released, not re-read.
        dry_mass=_DeviceArray(), total_pressure=np.zeros((ny, nx)),
        hydrometeor_initialization={"qc": "analyzed"},
        aerosol_initialization={"nwfa": "climatology"},
        surface_moisture_floor={"cells": 3, "minimum": -1.0e-6},
        prognostic_moisture_floor={})
    met = SimpleNamespace(
        fields={
            "LANDSEA": surface, "SKINTEMP": 280.0 * surface,
            "SOILT": np.full((9, ny, nx), 281.0, dtype=np.float32),
            "SOILW": np.full((9, ny, nx), 0.2, dtype=np.float32),
            "T2": 279.0 * surface, "U10": surface, "V10": surface,
            # A field the prepared contract does not keep.
            "TT": np.zeros((nz, ny, nx), dtype=np.float32)},
        water_temperature_receipt={"lake_water_mapping": {"cells": 4}},
        masked_field_repairs={"SKINTEMP": {"other_surface": 2,
                                           "no_source_land": 1}},
        water_temperature=_DeviceArray())
    return cfg, static, initial, met, boundaries, states


def test_the_seal_rewrites_the_root_from_the_head_byte_for_byte(
        tmp_path, monkeypatch):
    """The re-read root writes the one-shot root cache the held root writes.

    The chained tree releases its start state at the head and the seal
    writes d01 from the state read back from the head's sealed cache.  The
    cache that write produces must be the one the held state produces:
    same content digest, so the same arrays, metadata (the aerosol and
    water receipts, the surface repairs) and setup fingerprint.  And the
    moisture floors the proof reports survive the release.
    """

    from gpuwm.ingest.lateral_bc import attach_lateral_boundaries
    from gpuwm.ingest.prepared_cache import (
        PreparedCacheReader, write_prepared_cache)
    from gpuwm.moisture_floor_receipt import moisture_floor_field_names
    import gpuwm.native_wrf_contract as native_wrf_contract

    cfg, static, initial, met, boundaries, states = _host_root()
    identity = {"source": "tree-start-states-test"}
    # The one-shot tree: the held start state, its boundaries attached.
    attach_lateral_boundaries(initial.state, boundaries)
    oneshot = write_prepared_cache(
        tmp_path / "oneshot" / "prepared-cache", identity=identity,
        initial_result=initial, met=met, boundaries=boundaries)
    # The chained tree's head: the start state before any boundary exists.
    fresh = _host_root()
    initial = fresh[2]
    staging = tmp_path / ".tmp-tree"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "tree", identity=identity,
        cache_name=TREE_CACHE, chained=False)
    writer.write_head(
        initial_result=initial, met=met, lbc={
            "spec_bdy_width": boundaries.spec_bdy_width,
            "spec_zone": boundaries.spec_zone,
            "relax_zone": boundaries.relax_zone,
            "schedule": [[i.start_seconds, i.end_seconds]
                         for i in boundaries.intervals],
            "fields": sorted(boundaries.intervals[0].fields)},
        proof_head=PROOF_HEAD, input_manifest_sha256="a" * 64,
        tree=domain_tree_head_fields(["d01", "d02"], root_cache=TREE_CACHE))
    floors = moisture_floor_field_names(initial)
    starts = boundary_stream.TreeStartStates.release(
        root_result=initial, root_met=met, child_results=(),
        child_content_sha256={})
    del initial, met, states, fresh
    for index, interval in enumerate(boundaries.intervals):
        writer.write_segment(index, interval)
    sealed = writer.seal_cache()["content_sha256"]
    assert sealed == oneshot["content_sha256"]
    monkeypatch.setattr(
        native_wrf_contract, "load_native_static_cache",
        lambda path, grid, ny, nx: dict(static))
    exp = SimpleNamespace(domains=(SimpleNamespace(grid_id=1, run=cfg),))
    root_result, root_met, root_boundaries, children = starts.reread(
        writer.root, exp=exp, grids=(None,), root_identity=identity,
        root_content_sha256=sealed)
    assert children == ()
    assert isinstance(root_result.state.u, np.ndarray)
    assert root_result.state.lateral_boundaries is root_boundaries
    assert not hasattr(root_result, "dry_mass")
    assert not hasattr(root_met, "water_temperature")
    assert moisture_floor_field_names(root_result) == floors
    assert root_result.surface_moisture_floor == {
        "cells": 3, "minimum": -1.0e-6}
    rewritten = write_prepared_cache(
        tmp_path / "sealed" / "prepared-cache", identity=identity,
        initial_result=root_result, met=root_met, boundaries=root_boundaries)
    assert rewritten["content_sha256"] == oneshot["content_sha256"]
    header = PreparedCacheReader(
        tmp_path / "sealed" / "prepared-cache",
        expected_identity=identity).header["metadata"]
    assert header["aerosol_initialization"] == {"nwfa": "climatology"}
    assert header["water_temperature"] == {
        "lake_water_mapping": {"cells": 4}}
    assert header["surface_from_other_surface"] == {"SKINTEMP": 2}
    assert header["soil_from_skin_and_field_capacity"] == {"SKINTEMP": 1}
    # A head cache that is not the one the head recorded is refused.
    with pytest.raises(RuntimeError, match="not the one the head recorded"):
        starts.reread(writer.root, exp=exp, grids=(None,),
                      root_identity=identity,
                      root_content_sha256="0" * 64)


def test_a_release_keeps_no_array_and_keeps_every_receipt():
    _cfg, _static, initial, met, _boundaries, _states = _host_root()
    child = SimpleNamespace(
        domain=SimpleNamespace(grid_id=2), grid="grid-d02",
        state=initial.state, real=initial, horizontal=met,
        static_fields={"HGT_M": np.zeros((2, 2))}, soil="soil-d02",
        input_preparation_seconds=1.5, preprocess_receipt={"backend": "cuda"})
    starts = boundary_stream.TreeStartStates.release(
        root_result=initial, root_met=met, child_results=(child,),
        child_content_sha256={"d02": "c" * 64})

    def arrays(shell):
        return sorted(name for name, value in shell.items()
                      if isinstance(value, np.ndarray)
                      or hasattr(value, "__cuda_array_interface__"))

    assert arrays(starts._root) == []
    assert arrays(starts._root_met) == []
    assert "state" not in starts._root and "fields" not in starts._root_met
    assert starts._root["aerosol_initialization"] == {
        "nwfa": "climatology"}
    assert starts._root_met["masked_field_repairs"] == {
        "SKINTEMP": {"other_surface": 2, "no_source_land": 1}}
    (kept,) = starts._children
    assert kept["label"] == "d02" and kept["content_sha256"] == "c" * 64
    assert set(kept["child"]) == {
        "domain", "grid", "static_fields", "soil",
        "input_preparation_seconds", "preprocess_receipt"}
    assert arrays(kept["real"]) == [] and arrays(kept["met"]) == []


def test_a_sealed_tree_that_is_not_its_head_is_refused():
    starts = boundary_stream.TreeStartStates(
        root={}, root_met={}, children=[
            {"label": "d02", "content_sha256": "b" * 64}])

    def receipt(d01, d02):
        return {"domains": [
            {"grid_id": 1, "artifacts": {"prepared_cache": {
                "content_sha256": d01}}},
            {"grid_id": 2, "artifacts": {"prepared_cache": {
                "content_sha256": d02}}}]}

    starts.require_sealed_is_head(receipt("a" * 64, "b" * 64),
                                  root_content_sha256="a" * 64)
    with pytest.raises(RuntimeError, match=r"\['d02'\]"):
        starts.require_sealed_is_head(receipt("a" * 64, "e" * 64),
                                      root_content_sha256="a" * 64)


# ---------------------------------------------------------------------------
# What each tree route says
# ---------------------------------------------------------------------------


def test_each_tree_route_says_why_it_is_sealed_or_chains():
    root = Path(__file__).resolve().parents[1]

    def body(path, name):
        text = (root / path).read_text(encoding="utf-8")
        start = text.index(f"def {name}(")
        end = text.find("\ndef ", start + 1)
        return text[start:end if end > 0 else None]

    # The native HRRR tree chains on its root preparation's head (A136
    # L7c), so the last tree row, "domain_tree", is retired.
    assert 'say_prepared_sealed("domain_tree")' not in body(
        "gpuwm/runplan.py", "_hrrr_chain")
    assert "domain_tree" not in boundary_stream.SEALED_REASONS
    # The staged chain starts a tree's forecast on the head a mapped tree
    # publishes on either backend (A136 L6, L7a), so it says no sealed
    # reason of its own.
    staged = body("gpuwm/runplan.py", "_staged_chain")
    assert 'say_prepared_sealed("domain_tree")' not in staged
    assert "domain_tree_forecast" not in staged
    assert "domain_tree_forecast" not in boundary_stream.SEALED_REASONS
    # A mapped tree chains on either backend: its seal re-reads the start
    # states from the head (TreeStartStates), so nothing holds a start
    # state on the card under a later build and the CUDA row is retired.
    mapped = (root / "gpuwm/mapped_direct.py").read_text(encoding="utf-8")
    assert "say_prepared_sealed(" not in mapped
    assert "domain_tree_cuda" not in boundary_stream.SEALED_REASONS
    # go's tree is a GFS-series tree, which chains like a mapped one
    # (gfs_direct._prepare_chained_gfs_tree), so go says no sealed reason
    # and the gfs_domain_tree row is retired (A136 L7b).
    go = (root / "gpuwm/go_cli.py").read_text(encoding="utf-8")
    assert "say_prepared_sealed(" not in go
    assert "gfs_domain_tree" not in boundary_stream.SEALED_REASONS
    gfs = (root / "gpuwm/gfs_direct.py").read_text(encoding="utf-8")
    assert "say_prepared_sealed(" not in gfs
    # An as-posted tree takes the head and seal whether or not its head is
    # published early, because its children bind the input plan at the
    # head (A136 L3 item (v)); the chained default is unchanged.
    assert ("chain_tree = len(exp.domains) > 1 and (\n"
            "            chained_enabled() or posted_series is not None)") in gfs
