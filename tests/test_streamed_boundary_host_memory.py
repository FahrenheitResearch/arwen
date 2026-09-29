"""A streamed domain's host memory does not grow with its tile count.

THE DEFECT.  ``streaming.tile_boundary_tables`` precomputed, for every
tile, a float64 copy of the domain's lateral forcing windowed onto that tile,
for every interval, field and side, and gave every interior side real
arrays of zeros.  Host memory therefore scaled with the NUMBER OF TILES
rather than with the domain: a 206x204x49 GFS forecast streamed with 1,190
tiles held 27.7 GB, and a reproduction reached 125 GB committed on a 96 GB
machine and was still rising -- into the pagefile, until the machine's own
management calls failed with "Out of memory".  The run-plan estimate for the
same plan said 0.93 GiB of host RAM, because it counted the pinned store and
its arena and no boundary table at all.

What is pinned here, on the CPU:

* the windowing keeps host bytes bounded by a small fraction of the
  domain's own table bytes, at 4 tiles and at 400 alike, because a tile is
  windowed when a buffer binds it, as views;
* a true-edge side is a view of the domain's table and an interior side is
  a zero-stride view of one shared zero -- neither allocates;
* the host estimate (``run-plan --estimate``, ``gpuwm check``, the ``gpuwm
  go`` gate) counts the domain's forcing series in ``host_bytes``, on the
  single-domain road and for a streamed root on a nested tree's mixed road.

The GPU half -- that a streamed run on these tables is bit-identical to the
resident run -- is ``tilestream/test_join.py`` and ``tilestream/test_route.py``.
"""

from __future__ import annotations

import textwrap
import tracemalloc

import numpy as np
import pytest

from gpuwm.core import preflight, streaming
from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                     LateralBoundaries, SideBoundary)

WIDTH = 5
HALO = 3
#: Mass-grid staggering per field, as ``lateral_bc`` lays the tables out.
FIELDS = {"u": (0, 1), "v": (1, 0), "theta": (0, 0), "phi": (0, 0),
          "mu": (0, 0), "qv": (0, 0)}


def _domain(*, nz=10, ny=120, nx=120, intervals=3) -> LateralBoundaries:
    """A domain series with distinguishable nonzero tables, several intervals."""
    rng = np.random.default_rng(11)
    out = []
    for k in range(intervals):
        fields = {}
        for name, (ey, ex) in FIELDS.items():
            levels = 1 if name == "mu" else nz + (name == "phi")

            def side(shape):
                value = 1.0 + rng.random(shape)
                return SideBoundary(value, 1.0e-3 * value)

            fields[name] = FieldBoundary(
                west=side((levels, ny + ey, WIDTH)),
                east=side((levels, ny + ey, WIDTH)),
                south=side((levels, WIDTH, nx + ex)),
                north=side((levels, WIDTH, nx + ex)))
        out.append(BoundaryInterval(k * 10800.0, (k + 1) * 10800.0, fields))
    return LateralBoundaries(tuple(out), WIDTH, 1, 4)


def _table_bytes(bnd) -> int:
    return sum(array.nbytes for interval in bnd.intervals
               for boundary in interval.fields.values()
               for side in (boundary.west, boundary.east,
                            boundary.south, boundary.north)
               for _, array in side.array_items())


def _specs(nx, ny, tile):
    from tilestream import spec as tspec

    return tspec.plan_tiles(nx, ny, tile, tile, HALO, False)


def _sweep_host_bytes(bnd, specs, nbuffers=2):
    """Bytes held and peak while every tile binds every interval in turn.

    The streamed run's own access pattern: a buffer binds a tile, the step
    reads that tile's active interval, and the buffer moves on; ``nbuffers``
    tiles are bound at any one time.  Measured from before the tables are
    built, so the build itself is inside the figure.
    """
    tracemalloc.start()
    try:
        tables = streaming.tile_boundary_tables(bnd, specs)
        bound = []
        for interval in bnd.intervals:
            t = interval.start_seconds + 60.0
            for itile in range(len(tables)):
                tile = tables[itile]
                active = tile.interval_at(t)
                assert active.start_seconds == interval.start_seconds
                bound.append(tile)
                if len(bound) > nbuffers:
                    bound.pop(0)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return current, peak


def test_tile_tables_do_not_scale_host_memory_with_tile_count():
    """THE REGRESSION: 400 tiles hold what 4 tiles hold, a sliver of the domain.

    Before the fix a 120 x 120 domain's tables held 2.2x the domain's own
    bytes at 4 tiles and 44x at 400 (MEASURED, 13.4 MB and 264 MB against
    6.0 MB), growing linearly with the tile count.  The domain here is
    140 x 140 so that its 400-tile tiling (7-cell tiles) keeps every seam
    zone + halo = 7 cells from a forced edge; 120 x 120 in 6-cell tiles put
    zone cells in the halo of tiles that do not own that edge, which the
    windowing now refuses (tilestream.spec.edge_band_unowned).
    """
    bnd = _domain(ny=140, nx=140)
    domain_bytes = _table_bytes(bnd)
    few = _specs(140, 140, 70)
    many = _specs(140, 140, 7)
    assert (len(few), len(many)) == (4, 400)

    few_current, few_peak = _sweep_host_bytes(bnd, few)
    many_current, many_peak = _sweep_host_bytes(bnd, many)

    # Bounded by the domain, not by the tiling: what a streamed run keeps on
    # the host for its tiles is a small fraction of the series it cuts them
    # from, at either tile count.
    for label, current, peak in (("4 tiles", few_current, few_peak),
                                 ("400 tiles", many_current, many_peak)):
        assert peak <= domain_bytes // 4, (
            f"{label}: peak {peak:,} B while windowing, against "
            f"{domain_bytes:,} B of domain tables")
        assert current <= domain_bytes // 20, (
            f"{label}: {current:,} B still held after the sweeps")
    # And a hundred times the tiles is not a hundred times the bytes.
    assert many_peak <= few_peak + domain_bytes // 20, (
        f"peak grew from {few_peak:,} B at 4 tiles to {many_peak:,} B at "
        f"400 tiles")


def test_edges_are_views_and_interior_sides_allocate_nothing():
    """Every side of every tile, interval and field, checked for its storage."""
    bnd = _domain(nz=3, ny=48, nx=64, intervals=2)
    specs = _specs(64, 48, 16)
    tables = streaming.tile_boundary_tables(bnd, specs)
    assert len(tables) == len(specs)
    saw_edge = saw_seam = False
    for spec, tile in zip(specs, tables):
        owned = streaming.owned_edges(spec)
        for k, interval in enumerate(tile.intervals):
            domain_interval = bnd.intervals[k]
            for name, boundary in interval.fields.items():
                for side_name in ("west", "east", "south", "north"):
                    side = getattr(boundary, side_name)
                    source = getattr(domain_interval.fields[name], side_name)
                    for key, array in side.array_items():
                        assert not array.flags.writeable
                        assert not array.flags.owndata, (
                            f"tile {spec.index} {name}/{side_name}/{key} "
                            "owns a copy")
                        if owned[side_name]:
                            assert np.shares_memory(
                                array, dict(source.array_items())[key])
                            saw_edge = True
                        else:
                            assert all(s == 0 for s in array.strides)
                            assert not np.any(array)
                            saw_seam = True
    assert saw_edge and saw_seam


def test_a_true_edge_window_carries_the_domains_values():
    """Views, not approximations: element-wise the domain's own slice.

    Ragged both ways (56 = 5 x 11 + 1, 40 = 3 x 11 + 7), with every seam
    zone + halo = 7 cells from a forced edge (12-cell tiles left the north
    zone in the halo of a tile that does not own that edge).
    """
    bnd = _domain(nz=2, ny=40, nx=56, intervals=2)
    specs = _specs(56, 40, 11)
    tables = streaming.tile_boundary_tables(bnd, specs)
    for spec, tile in zip(specs, tables):
        owned = streaming.owned_edges(spec)
        interval = tile.interval_at(10800.0 + 1.0)
        source = bnd.intervals[1].fields["u"]
        if owned["south"]:
            want = np.asarray(source.south.value)[
                :, :, spec.ci0:spec.ci0 + spec.cnx + 1]
            assert np.array_equal(interval.fields["u"].south.value, want)
        if owned["west"]:
            want = np.asarray(source.west.tendency)[
                :, spec.cj0:spec.cj0 + spec.cny, :]
            assert np.array_equal(interval.fields["u"].west.tendency, want)


def test_a_bound_tiles_series_keeps_its_identity_and_windows_only_what_is_read():
    """The device mirror reloads when an interval's id changes, so it must not.

    A tile held by a buffer answers the same series object, and the same
    interval object on every read; finding the active interval windows that
    interval alone, not every interval before it.
    """
    bnd = _domain(nz=2, ny=40, nx=40, intervals=4)
    tables = streaming.tile_boundary_tables(bnd, _specs(40, 40, 10))
    tile = tables[5]
    assert tables[5] is tile
    last = tile.interval_at(3 * 10800.0 + 5.0)
    assert tile.interval_at(3 * 10800.0 + 7.0) is last
    assert tile.intervals[-1] is last
    assert len(tile.intervals._windowed) == 1
    # The whole series is still there for the attachment's validation pass.
    assert [iv.start_seconds for iv in tile.intervals] == [
        iv.start_seconds for iv in bnd.intervals]
    head = tile.intervals[:2]
    assert len(head) == 2
    assert head[0] is tile.intervals[0] and head[1] is tile.intervals[1]


# --------------------------------------------------------------------------
# the host estimate
# --------------------------------------------------------------------------

#: The shape of the forecast that found the defect: a 206x204 GFS domain at
#: 3 km, 49 levels, 24 h with 3-hourly forcing (eight intervals), streamed
#: on a pinned 9x9 tiling (the tile the desktop's door chose for it).
_TILED_GFS = """\
[experiment]
name = "streamed-host"
start_time = 2026-09-26T06:00:00
run_seconds = 86400.0
restart_interval_s = 0.0

[fetch]
source = "gfs"
cycle = "2026-09-26T06"
hours = 24
cadence = 3

[shared]
nz = 49
ztop = 20000.0
moist = true
moist_cq = true
mp_physics = 10
ra_lw_physics = 4
ra_sw_physics = 4
sf_sfclay_physics = 91
sf_surface_physics = 2
bl_pbl_physics = 1
nwp_diagnostics = 1

[tiles]
mode = "on"
tile_nx = 9
tile_ny = 9

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 206
ny = 204
time_step = 15
dx = 3000.0
history_interval_s = 3600.0
"""


@pytest.fixture()
def tiled_gfs(tmp_path):
    path = tmp_path / "streamed-host.toml"
    path.write_text(textwrap.dedent(_TILED_GFS), encoding="utf-8")
    return path


def test_the_host_estimate_counts_the_domains_boundary_series(tiled_gfs):
    """``host_bytes`` is the pinned store and arena PLUS the forcing series."""
    from gpuwm import runplan

    exp = preflight._load_experiment_any(tiled_gfs)
    run = exp.root.run
    assert run.specified and not run.nested
    phases = preflight.estimate_phases(exp, source=None,
                                       forcing_interval_seconds=10800.0)
    env = phases.streamed
    assert env is not None, "the fixture no longer streams"

    # float64 value and tendency, four sides, every field, eight intervals.
    series = 8 * preflight.lbc_interval_values(run) * 8
    assert series > 100 * 1024 ** 2          # it is not a rounding error here
    assert env.boundary_table_bytes == series
    assert env.pinned_bytes == env.store_bytes + env.arena_bytes
    assert env.host_bytes == env.pinned_bytes + series
    terms = dict(env.terms)
    assert terms["host/boundary_table_bytes"] == series
    assert terms["host/pinned_bytes"] + series == env.host_bytes

    # The run-plan estimate document carries it where a front end reads it.
    execution = runplan._execution_estimate(phases, exp, None)
    assert execution["streamed_forecast"] is True
    assert execution["host_bytes"] == env.host_bytes
    vram = runplan._vram_estimate(phases.forecast, env, exp)
    section = vram["streamed"]
    assert section["host_bytes"] == env.host_bytes
    assert section["boundary_table_bytes"] == series
    assert section["pinned_bytes"] + series == env.host_bytes


def test_a_nested_tree_counts_its_streamed_roots_series(tmp_path):
    """The mixed road prices a streamed root's forcing series the same way.

    Only the root carries a tabulated series (a nest is forced from its
    parent's device frame), so the tree's host figure grows by the root's
    series when the root streams, and by nothing when the root is resident.
    """
    from gpuwm import starter_template as st
    from gpuwm.experiment import load_experiment
    from test_starter_template import tile_starter

    path, raw = tile_starter(tmp_path, nested=True)
    raw["tiles"] = {"mode": "on", "tile_nx": 32, "tile_ny": 32}
    path.write_text(st.render_tables(raw), encoding="utf-8")
    exp = load_experiment(path)
    run = exp.root.run
    series = preflight.lbc_host_series_bytes(
        run, preflight.lbc_intervals(float(run.run_seconds), 10800.0))
    assert series > 0

    road = streaming.tree_road_plan(exp, forcing_interval_seconds=10800.0)
    assert road.priced and road.refusal is None
    assert [row["road"] for row in road.rows] == ["streamed", "streamed"]
    claims = sum(int(row.get("host_claim_bytes") or 0) for row in road.rows)
    assert road.boundary_table_bytes == series
    assert road.pinned_bytes == claims
    assert road.host_bytes == claims + series
    # The report every surface reads carries the same figure.
    phases = preflight.estimate_phases(exp, source=None,
                                       forcing_interval_seconds=10800.0)
    assert phases.streamed.host_bytes == road.host_bytes
    assert "lateral-boundary tables" in phases.verdict(None)

    # Control: the root resident, the nest streamed -- no series counted.
    raw["domain"][0]["tiles"] = {"mode": "off"}
    path.write_text(st.render_tables(raw), encoding="utf-8")
    resident_root = streaming.tree_road_plan(
        load_experiment(path), forcing_interval_seconds=10800.0)
    assert [row["road"] for row in resident_root.rows] == [
        "resident", "streamed"]
    assert resident_root.boundary_table_bytes == 0
    assert resident_root.host_bytes == resident_root.pinned_bytes


def test_the_run_plan_estimate_document_counts_the_series(
        tiled_gfs, tmp_path, monkeypatch):
    """``run-plan --estimate`` itself, on a plan built from the config.

    The schedule comes from the config's own ``[fetch] cadence`` through
    the one read every surface shares, so this is the figure a front end
    shows beside "Streaming host memory".
    """
    from gpuwm import runplan

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess",
                        lambda *a, **k: None)
    geog = tmp_path / "GEOG"
    geog.mkdir()
    plan = runplan.build_plan(
        {"schema": runplan.PLAN_SCHEMA, "name": "streamed-host",
         "route": "prepared", "config": {"path": str(tiled_gfs)},
         "run_options": {"geog_root": str(geog)},
         "output_root": str(tmp_path / "run")},
        source="plan.json", base_dir=tmp_path, sha256="0" * 64)
    document = runplan.estimate_plan(plan)

    exp = preflight._load_experiment_any(tiled_gfs)
    series = 8 * preflight.lbc_interval_values(exp.root.run) * 8
    section = document["vram"]["streamed"]
    assert section["boundary_table_bytes"] == series
    assert document["execution"]["host_bytes"] == section["pinned_bytes"] + series


def test_the_gate_refuses_a_series_the_host_budget_cannot_hold(
        tiled_gfs, monkeypatch):
    """Counted in admission, not only printed: over the budget is a refusal.

    The host is sized so its budget falls between the pinned store and the
    whole claim: the store alone would fit, the forecast with its forcing
    series would not, and the gate says which part it counted.  The control
    is the same plan on a host whose budget holds the whole claim.
    """
    from gpuwm import go_cli
    from tilestream import autoplan

    def _probe(*_args, **_kwargs):
        return {"free_bytes": 15 * 1024 ** 3, "total_bytes": 16 * 1024 ** 3,
                "name": "test card", "local_memory_bytes_per_thread": 0}

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", _probe)
    plan = {"config": str(tiled_gfs), "source": "gfs", "cadence": 3}
    exp = preflight._load_experiment_any(tiled_gfs)
    env = preflight.streamed_forecast_envelope(
        exp, forcing_interval_seconds=10800.0)
    assert env.boundary_table_bytes > 0

    def host_with_budget(budget):
        total = int(budget / autoplan.PINNED_FRACTION)
        monkeypatch.setattr(streaming, "_host_total_bytes", lambda: total)

    host_with_budget(env.host_bytes + 64 * 1024 ** 2)
    admitted = go_cli.memory_gate(plan)
    assert admitted["phases"].streamed.host_bytes == env.host_bytes
    assert admitted["refuse"] is False, admitted["verdict"]
    # The verdict states the series beside the store even when it fits.
    assert "lateral-boundary tables" in admitted["verdict"]
    assert "allows a forecast to hold" not in admitted["verdict"]

    host_with_budget(env.pinned_bytes + env.boundary_table_bytes // 2)
    refused = go_cli.memory_gate(plan)
    assert refused["refuse"] is True, refused["verdict"]
    assert "allows a forecast to hold" in refused["verdict"]
    assert "lateral-boundary tables" in refused["verdict"]
