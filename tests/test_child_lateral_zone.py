"""A downscaled child's lateral zone is sized for its parent, not for itself.

The defect: ``gpuwm downscale`` copied the parent's root-domain lateral
settings into every derived child -- spec_bdy_width 5, spec_zone 1,
relax_zone 4 -- and those count CHILD cells.  At ratio 12 or 20 the whole
zone was 1.25 km or 0.75 km, narrower than one 3 km parent cell, and
``w`` was never relaxed, only copied onto the boundary from the first
interior row.  The children made their own storms on their edges.  The
derived zone's time scale is set in seconds (one child cell crossed at
20 m/s), which the 2 h arms chose over the parent's own 150 s.

These are the CPU halves of the fix: the derived configuration, the
coefficient law, the identity bookkeeping and the tile planning rule that
a zone wider than a tile halo needs.  The GPU halves (the kernel's side
mask and the w relaxation) are in tests/test_child_edge_relaxation_gpu.py.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gpuwm import downscale
from gpuwm.config import RunConfig, validate_run_config
from gpuwm.ingest.lateral_bc import (LateralBoundaries, _frame_rings,
                                     _perimeter_count, _relax_side_mask,
                                     _weights, relax_timescale_seconds,
                                     specified_relaxes_w)

#: A 3 km HRRR parent's restart-evidence config, in the shape the
#: derivation reads (dt 15 s, WRF's zone).
_PARENT_3KM = {
    "nx": 162, "ny": 162, "nz": 49, "dx": 3000.0, "dy": 3000.0,
    "ztop": 20000.0, "dt": 15.0, "run_seconds": 43200.0,
    "output_interval_s": 900.0, "hybrid_opt": 2, "etac": 0.2,
    "hypsometric_opt": 2, "moist": True, "mp_physics": 8,
    "specified": True, "nested": False, "terrain_opt": 1, "map_proj": 1,
    "grid_id": 1, "time_step_sound": 4, "spec_bdy_width": 5,
    "spec_zone": 1, "relax_zone": 4,
}


def _derived(ratio: int, size: int, parent=_PARENT_3KM) -> dict:
    return downscale._derive_child_run_config(
        parent, parent={"dx": parent["dx"], "dy": parent["dy"]},
        ratio=ratio, child_nx=size, child_ny=size, run_seconds=7200.0,
        output_interval_s=900.0)


@pytest.mark.parametrize("ratio,size", [(12, 300), (20, 360)])
def test_derived_zone_spans_two_parent_cells(ratio, size):
    """Relaxed rows are spec_zone .. relax_zone - 1 (WRF relax_bdytend_core),
    so the zone reaches ``relax_zone`` child cells in from the edge.  Before
    the fix that was 4 child cells at every ratio -- a third of a parent
    cell at ratio 12, a fifth at ratio 20."""

    merged = _derived(ratio, size)
    assert merged["relax_zone"] >= 2 * ratio
    assert merged["relax_zone"] == (
        downscale.CHILD_RELAX_PARENT_CELLS * ratio)
    assert merged["spec_zone"] == 1
    assert merged["spec_bdy_width"] == merged["spec_zone"] + merged[
        "relax_zone"]
    cfg = validate_run_config(RunConfig(**merged))
    assert cfg.specified and not cfg.nested
    # in metres: at least two 3 km parent cells
    assert cfg.relax_zone * cfg.dx >= 2 * _PARENT_3KM["dx"]


def test_derived_zone_time_scale_is_one_child_cell_crossing_in_seconds():
    """The first relaxed row follows the specified row as fast as a 20 m/s
    flow crosses one child cell: 12.5 s at 250 m, 7.5 s at 150 m.  The
    parent's own time scale (150 s, 10 parent steps) left rows 1 to 3 of
    the 250 m arm at nearly twice the parent's |w|."""

    for ratio, size, seconds in ((12, 300, 12.5), (20, 360, 7.5)):
        merged = _derived(ratio, size)
        assert merged["dt"] == pytest.approx(15.0 / ratio)
        assert merged["relax_timescale_s"] == pytest.approx(seconds)
        assert merged["relax_timescale_s"] == pytest.approx(
            merged["dx"] / downscale.CHILD_RELAX_CROSSING_SPEED)
        assert merged["relax_w"] is True
    # A grandchild takes its own cell, not its parent's time scale.
    child = _derived(12, 300)
    grandchild = downscale._derive_child_run_config(
        child, parent={"dx": child["dx"], "dy": child["dy"]}, ratio=5,
        child_nx=200, child_ny=200, run_seconds=3600.0,
        output_interval_s=900.0)
    assert grandchild["dx"] == pytest.approx(50.0)
    assert grandchild["relax_timescale_s"] == pytest.approx(2.5)
    assert grandchild["relax_zone"] == 10


def test_derived_time_scale_is_seconds_and_never_stiffer_than_wrf():
    """Set in seconds, it does not follow the step; it is never shorter
    than WRF's own 10 child steps, so no child is nudged harder per step
    than WRF nudges its own domains."""

    zone = downscale.child_lateral_zone(
        _PARENT_3KM, ratio=12, child_nx=300, child_ny=300,
        child_dx=250.0, child_dy=250.0, child_dt=0.5)
    assert zone["relax_timescale_s"] == pytest.approx(12.5)
    # a parent on a longer step hands its child a longer one: WRF's law
    slow = dict(_PARENT_3KM, dt=18.0)
    merged = _derived(12, 300, parent=slow)
    assert merged["dt"] == pytest.approx(1.5)
    assert merged["relax_timescale_s"] == pytest.approx(15.0)
    # the shorter side sets the crossing
    oblong = downscale.child_lateral_zone(
        _PARENT_3KM, ratio=12, child_nx=300, child_ny=300,
        child_dx=250.0, child_dy=200.0, child_dt=0.5)
    assert oblong["relax_timescale_s"] == pytest.approx(10.0)


def test_derived_zone_ramps_linearly_whatever_the_parent_ramp():
    """A parent's exponential ramp counts the parent's own rows: WRF's
    0.33 decays the weights by e over three of them.  Inherited, it cut a
    24-row child zone to its outer three or four rows (the 24th row's
    weight is exp(-0.33 * 23), 5e-4 of the first), which is the narrow
    zone the children made their storms in.  The derived zone ramps
    linearly, the ramp the arms measured narrowest at the zone's inner
    edge."""

    ramped = dict(_PARENT_3KM, spec_exp=0.33)
    merged = _derived(12, 300, parent=ramped)
    assert merged["spec_exp"] == 0.0
    cfg = validate_run_config(RunConfig(**merged))
    fcx, _ = _weights(cfg.spec_bdy_width, cfg.spec_zone, cfg.relax_zone,
                      cfg.dt, cfg.spec_exp,
                      timescale_s=cfg.relax_timescale_s)
    # the inner end of the zone still pulls a twenty-third as hard as the
    # outer end, as the linear ramp does
    assert fcx[cfg.relax_zone - 1] == pytest.approx(
        fcx[cfg.spec_zone] / (cfg.relax_zone - 1), rel=1e-5)
    assert downscale.child_lateral_zone(
        ramped, ratio=12, child_nx=300, child_ny=300, child_dx=250.0,
        child_dy=250.0, child_dt=1.25)["spec_exp"] == 0.0


def test_derived_zone_is_capped_by_the_child_and_floored_by_the_parent():
    small = _derived(20, 40)
    # two zones may take half of the shorter side between them
    assert small["relax_zone"] == 10
    assert 2 * small["relax_zone"] <= 40 // 2
    ratio_two = _derived(2, 200)
    # 2 parent cells at ratio 2 is 4 child cells, WRF's own zone
    assert ratio_two["relax_zone"] == 4
    assert _derived(1, 200)["relax_zone"] == 4
    validate_run_config(RunConfig(**small))


def test_the_zone_keys_are_written_into_the_derived_child_toml(tmp_path):
    import tomllib

    merged = _derived(20, 360)
    path = tmp_path / "child.toml"
    path.write_text(downscale._render_child_toml(merged), encoding="utf-8")
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))["run"]
    for key in ("spec_zone", "relax_zone", "spec_bdy_width",
                "relax_timescale_s", "relax_w"):
        assert parsed[key] == merged[key], key
    from gpuwm.config import load_config
    cfg = load_config(path)
    assert cfg.relax_zone == 40 and cfg.relax_w is True


def test_relaxation_coefficients_are_set_in_physical_time():
    """With a time scale in seconds the zone does not tighten as the
    child's step shrinks; WRF's recipe (time scale 0) does."""

    width, spec_zone, relax_zone = 25, 1, 24
    at_250m = _weights(width, spec_zone, relax_zone, 1.25, 0.0,
                       timescale_s=150.0)
    at_150m = _weights(width, spec_zone, relax_zone, 0.75, 0.0,
                       timescale_s=150.0)
    for a, b in zip(at_250m, at_150m):
        np.testing.assert_array_equal(a, b)
    fcx, gcx = at_250m
    assert fcx[spec_zone] == pytest.approx(1.0 / 150.0, rel=1e-6)
    assert gcx[spec_zone] == pytest.approx(1.0 / 750.0, rel=1e-6)
    # the ramp still runs to zero across the zone
    assert fcx[relax_zone] == 0.0 and fcx[relax_zone - 1] > 0.0
    # and 10 dt reproduces WRF's own coefficients
    wrf = _weights(width, spec_zone, relax_zone, 15.0, 0.0)
    np.testing.assert_allclose(
        _weights(width, spec_zone, relax_zone, 15.0, 0.0,
                 timescale_s=150.0), wrf, rtol=1e-6)
    # WRF's recipe at the child's step is ratio times stiffer
    child_wrf = _weights(width, spec_zone, relax_zone, 1.25, 0.0)
    assert child_wrf[0][spec_zone] == pytest.approx(
        12.0 * fcx[spec_zone], rel=1e-6)


def test_the_two_keys_default_to_wrf_and_stay_out_of_identity():
    from gpuwm.core.model import restart_identity_payload
    from gpuwm.experiment import experiment_from_run_config
    from datetime import datetime, timezone

    cfg = RunConfig(nx=20, ny=20, nz=4, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=15.0, run_seconds=60.0, specified=True)
    assert relax_timescale_seconds(cfg) == 0.0
    assert specified_relaxes_w(cfg) is False
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    plain = restart_identity_payload(experiment_from_run_config(cfg, epoch))
    run = plain["domains"][0]["run"]
    assert "relax_timescale_s" not in run and "relax_w" not in run
    tuned = dataclasses.replace(cfg, relax_timescale_s=150.0, relax_w=True)
    bound = restart_identity_payload(
        experiment_from_run_config(tuned, epoch))["domains"][0]["run"]
    assert bound["relax_timescale_s"] == 150.0 and bound["relax_w"] is True
    assert specified_relaxes_w(tuned) is True
    # a nest always relaxes w; the key names the SPECIFIED treatment
    nest = dataclasses.replace(cfg, specified=False, nested=True,
                               relax_w=True)
    assert specified_relaxes_w(nest) is False


def test_a_nest_reads_the_time_scale_in_wrfs_nested_order():
    """No domain kind ignores the key: a nest takes it in the nested FP32
    operation order, and at 0 the nested weights are WRF's to the bit."""

    width, spec_zone, relax_zone = 5, 1, 4
    wrf = _weights(width, spec_zone, relax_zone, 5.0, 0.0, wrf_real=True)
    again = _weights(width, spec_zone, relax_zone, 5.0, 0.0, wrf_real=True,
                     timescale_s=0.0)
    for a, b in zip(wrf, again):
        np.testing.assert_array_equal(a, b)
    tuned = _weights(width, spec_zone, relax_zone, 5.0, 0.0, wrf_real=True,
                     timescale_s=150.0)
    assert tuned[0][spec_zone] == pytest.approx(1.0 / 150.0, rel=1e-6)
    assert tuned[1][spec_zone] == pytest.approx(1.0 / 750.0, rel=1e-6)
    cfg = RunConfig(nx=20, ny=20, nz=4, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=15.0, run_seconds=60.0,
                    specified=True, relax_timescale_s=-1.0)
    with pytest.raises(ValueError, match="relax_timescale_s"):
        validate_run_config(cfg)


def test_seam_sides_drop_out_of_the_relaxation_mask():
    bnd = LateralBoundaries((), 25, 1, 24)
    assert _relax_side_mask(bnd) == 15
    assert _relax_side_mask(None) == 15
    seamed = dataclasses.replace(bnd, seam_sides=("east", "north"))
    assert _relax_side_mask(seamed) == 1 | 4
    with pytest.raises(ValueError, match="no side"):
        LateralBoundaries((), 25, 1, 24, seam_sides=("up",))
    assert _frame_rings(41, 41, 24) == 21
    assert _frame_rings(300, 300, 24) == 24


def _frame_mask(ny: int, nx: int, rings: int) -> np.ndarray:
    """The cells ``rings`` nested perimeter frames cover."""
    mask = np.zeros((ny, nx), dtype=bool)
    for d in range(rings):
        mask[d, d:nx - d] = True
        mask[ny - 1 - d, d:nx - d] = True
        mask[d + 1:ny - d - 1, d] = True
        mask[d + 1:ny - d - 1, nx - 1 - d] = True
    return mask


@pytest.mark.parametrize("ny,nx,width", [
    (77, 120, 40), (120, 77, 40), (41, 41, 24), (43, 90, 24), (76, 120, 40),
    (300, 300, 24), (10, 12, 3)])
def test_the_frames_count_every_cell_of_a_window_once(ny, nx, width):
    """A tile window narrower than two zones along an odd side has a middle
    ring whose two rows (or two columns) are the same line.  The frame
    count listed that line twice -- 9284 entries for the 9240 cells of a
    77 x 120 window under a 40-cell zone -- and two threads of the
    relaxation kernel then read, modified and wrote each of its cells.  The
    kernel lists exactly this many (tests/test_child_edge_relaxation_gpu.py
    runs its frame_point)."""
    rings = _frame_rings(ny, nx, width)
    mask = _frame_mask(ny, nx, rings)
    assert _perimeter_count(ny, nx, rings) == int(mask.sum())
    if rings < width:
        # a window narrower than two zones: the frames cover all of it
        assert mask.all()


def _strands(n: int, tile: int, halo: int, band: int) -> bool:
    """Reference, on plan_tiles' own tiles: does some tile's interior,
    widened by the halo, hold a cell within ``band`` of an x edge its
    compute window does not reach?"""
    from tilestream import spec as tspec

    for s in tspec.plan_tiles(n, tile + 2 * halo, tile, tile, halo,
                              periodic=False):
        if s.ty:
            continue
        cells = np.arange(max(s.i0 - halo, 0), min(s.i1 + halo, n))
        if s.ci0 != 0 and (cells < band).any():
            return True
        if s.ci0 + s.cnx != n and (cells >= n - band).any():
            return True
    return False


def _seams_between(n: int, tile: int, halo: int, band: int) -> bool:
    """The rule in seam terms: some seam lies more than ``halo`` and fewer
    than ``band + halo`` cells from an edge."""
    return any(halo < d < band + halo
               for seam in range(tile, n, tile) for d in (seam, n - seam))


def test_a_tiles_interior_and_halo_reach_only_zones_it_owns():
    """The halo counts as much as the interior, and no further.  At ratio
    20 (zone 40, halo 18) a 40-cell tiling's second tile has its interior
    start at column 40 and its halo at column 22: columns 22 to 39 are
    zone cells its own cells read every step, beside a seam that relaxes
    nothing.  The resident run relaxes them; the tile ran them unrelaxed,
    and after 15 minutes every point of the tiled child differed.

    A zone cell further out than the halo is a different matter.  A tile
    against the far edge has its window clamped inside the domain, so on
    200 cells the second of two 125-cell tiles computes from column 39,
    one cell inside the west zone, while its interior starts at 125 and
    its halo at 107.  Nothing at column 39 reaches column 125 within a
    step, and the tiled run matches the resident one
    (tests/test_child_edge_relaxation_gpu.py).  So the rule tests the
    interior widened by the halo, not the clamped window."""
    from tilestream import spec as tspec

    assert tspec.edge_band_unowned(160, 40, 18, 40)
    assert not tspec.edge_band_unowned(160, 80, 18, 40)
    second = tspec.plan_tiles(200, 200, 125, 125, 18, periodic=False)[1]
    assert (second.i0, second.ci0) == (125, 39)
    assert not tspec.edge_band_unowned(200, 125, 18, 40)
    # seams 58 or more cells from both edges: every such tiling is legal,
    # whatever its clamped windows hold
    for n, first, last in ((160, 85, 102), (200, 125, 142),
                           (220, 145, 162)):
        for tile in range(first, last + 1):
            assert not tspec.edge_band_unowned(n, tile, 18, 40), (n, tile)
        # one cell wider and the first tile's halo reaches the east zone
        assert tspec.edge_band_unowned(n, last + 1, 18, 40)
    # a seam within the halo of an edge: the tile before it has its window
    # clamped out to that edge, owns it, and relaxes the zone itself
    for tile in range(71, 76):
        last_seam = (-(-160 // tile) - 1) * tile
        assert 160 - last_seam <= 18
        assert not tspec.edge_band_unowned(160, tile, 18, 40)
    # a 24-cell zone under a 16-cell halo: a 24-cell tiling's second tile
    # reads the west zone from column 8, and a 38-cell tiling's last seam
    # sits 34 cells from the east edge, so the tile before it reads six
    # zone columns and does not reach that edge
    assert tspec.edge_band_unowned(300, 24, 16, 24)
    assert tspec.edge_band_unowned(300, 38, 16, 24)
    assert not tspec.edge_band_unowned(300, 43, 16, 24)
    assert not tspec.edge_band_unowned(300, 100, 16, 24)
    for n, halo, band in ((160, 18, 40), (200, 18, 40), (220, 18, 40),
                          (300, 16, 24), (300, 16, 4), (97, 5, 12)):
        for tile in range(1, n - 2 * halo + 1):
            got = tspec.edge_band_unowned(n, tile, halo, band)
            assert got == _strands(n, tile, halo, band), (n, tile, halo,
                                                           band)
            if tile + 2 * halo < n:
                assert got == _seams_between(n, tile, halo, band), (
                    n, tile, halo, band)


def test_tiles_whose_halo_reaches_an_unowned_zone_are_never_planned():
    from types import SimpleNamespace

    from tilestream import spec as tspec
    from tilestream.autoplan import (_best_tile, _geometry_admits_no_tile,
                                     _smallest_legal_tile,
                                     _too_small_to_tile_message, edge_band)

    cfg = RunConfig(**_derived(12, 300))
    assert edge_band(cfg) == 24

    def strands(plan):
        return (tspec.edge_band_unowned(300, plan["tile_nx"], 16, 24)
                or tspec.edge_band_unowned(300, plan["tile_ny"], 16, 24))

    window = (38 + 32) * (38 + 32) * cfg.nz
    unbanded = _best_tile(300, 300, cfg.nz, 16, window, False, False, True,
                          band=0)
    # without the rule this budget plans a tile that strands zone cells
    assert strands(unbanded)
    # with it, nothing narrower than 43 cells is legal, so nothing fits
    assert _best_tile(300, 300, cfg.nz, 16, window, False, False, True,
                      band=24) is None
    assert _smallest_legal_tile(300, 16, False, 24) == 43
    assert _smallest_legal_tile(300, 16, False, 0) == 1
    roomier = (43 + 32) * (43 + 32) * cfg.nz
    banded = _best_tile(300, 300, cfg.nz, 16, roomier, False, False, True,
                        band=24)
    assert banded is not None and not strands(banded)
    # 150 cells under a 40-cell zone and an 18-cell halo tile in two
    # 75-cell tiles: the seam is 75 cells from both edges, though the
    # second window, clamped against the east edge, starts at column 39
    assert not _geometry_admits_no_tile(150, 150, 49, 18, False, False,
                                        True, band=40)
    assert _smallest_legal_tile(150, 18, False, 40) == 75
    # 100 cells have no legal tile at all: every tile that fits leaves a
    # seam between 18 and 58 cells from an edge, and the refusal says why
    # in terms of the zone
    assert _geometry_admits_no_tile(100, 100, 49, 18, False, False, True,
                                    band=40)
    assert not _geometry_admits_no_tile(100, 100, 49, 18, False, False,
                                        True, band=0)
    message = _too_small_to_tile_message(
        100, 100, 18, False, False,
        SimpleNamespace(resident_bytes=lambda cells: 1 << 30), 100 * 100 * 49,
        8 << 30, band=40)
    assert "zone + halo = 58" in message and "RESIDENT" in message
    assert "no more than the halo's 18 cells" in message


def test_a_tile_is_told_its_seams_and_a_stranded_zone_is_refused():
    from gpuwm.core.streaming import StreamingRefused, window_boundaries
    from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         SideBoundary)
    from tilestream import spec as tspec

    nz, ny, nx, width = 2, 120, 120, 25

    def side(shape):
        return SideBoundary(np.zeros(shape), np.zeros(shape))

    field = FieldBoundary(west=side((nz, ny, width)),
                          east=side((nz, ny, width)),
                          south=side((nz, width, nx)),
                          north=side((nz, width, nx)))
    bnd = LateralBoundaries(
        (BoundaryInterval(0.0, 60.0, {"theta": field}),), width, 1, 24)
    specs = tspec.plan_tiles(nx, ny, 40, 40, 16, periodic=False)
    tables = [window_boundaries(bnd, s) for s in specs]
    corner = tables[0]
    assert set(corner.seam_sides) == {"east", "north"}
    middle = tables[4]
    assert set(middle.seam_sides) == {"west", "east", "south", "north"}
    thin = tspec.plan_tiles(nx, ny, 20, 20, 16, periodic=False)
    with pytest.raises(StreamingRefused, match="no relaxation"):
        [window_boundaries(bnd, s) for s in thin]


def test_a_tree_reservation_prices_a_tiling_the_planner_admits():
    """A nested tree reserves card memory for a domain it has not decided
    yet, at the smallest window that domain may stream in.  That search
    skipped the relaxation band: for a 160 x 160 child at ratio 20 it
    priced 32-cell tiles (64-cell windows under a 16-cell halo), which the
    planner refuses, so the reservation held back less than the domain's
    smallest legal road, 80-cell tiles.  Past the redundancy limit it
    priced a one-cell tile's window."""
    from types import SimpleNamespace

    from gpuwm.core import streaming
    from tilestream import spec as tspec

    cfg = RunConfig(**_derived(20, 160))
    node = SimpleNamespace(cfg=SimpleNamespace(run=cfg), parent=None)
    halo = streaming._halo_for(cfg)
    assert cfg.relax_zone == 40
    tile = streaming._inbound_stream_tiling(node)
    assert tile == (80, 80)
    assert not tspec.edge_band_unowned(160, tile[0], halo, 40)
    options = streaming.StreamingOptions(mode="auto", max_redundancy=1.0)
    assert streaming._inbound_stream_tiling(node, options) is None
    fp = streaming.radiation_footprint(cfg, options)
    floor = streaming._minimum_claim_bytes(node, 0, options,
                                           force_stream=True)
    window = 80 + 2 * halo
    assert floor == int(fp.marginal_bytes(window * window * cfg.nz, 1))
    assert floor > int(fp.marginal_bytes((2 * halo + 1) ** 2 * cfg.nz, 1))


def _zone_boundaries(n: int, zone: int):
    from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         SideBoundary)

    width = zone + 1

    def side(shape):
        return SideBoundary(np.zeros(shape), np.zeros(shape))

    field = FieldBoundary(west=side((1, n, width)), east=side((1, n, width)),
                          south=side((1, width, n)),
                          north=side((1, width, n)))
    return LateralBoundaries(
        (BoundaryInterval(0.0, 60.0, {"theta": field}),), width, 1, zone)


def test_a_zone_cell_in_a_tiles_halo_is_refused_on_both_routes():
    """The measured tiling: a 160 x 160 child at ratio 20 on 40-cell tiles
    under an 18-cell halo.  Tile (0, 1)'s halo starts at column 22, so 18
    columns of the west zone sit in it while its west side is a seam; a
    rule that looked only at its interior (column 40) let it run.
    Refused on the windowing route and on the streamed runner's tables,
    before any tile steps.  An 80-cell tiling of the same child keeps its
    seam 80 cells from both edges and is admitted.  A tiling whose clamped
    window holds zone cells past the halo is admitted on both routes."""
    from gpuwm.core.streaming import (StreamingRefused, tile_boundary_tables,
                                      window_boundaries)
    from tilestream import spec as tspec

    bnd = _zone_boundaries(160, 40)
    specs = tspec.plan_tiles(160, 160, 40, 40, 18, periodic=False)
    second = next(s for s in specs if (s.ty, s.tx) == (0, 1))
    assert second.i0 == 40 and second.ci0 == 22
    with pytest.raises(StreamingRefused, match="halo") as refused:
        window_boundaries(bnd, second)
    assert "no relaxation" in str(refused.value)
    assert "zone + halo = 58" in str(refused.value)
    with pytest.raises(StreamingRefused, match="no relaxation"):
        tile_boundary_tables(bnd, specs)
    control = tspec.plan_tiles(160, 160, 80, 80, 18, periodic=False)
    tables = tile_boundary_tables(bnd, control)
    assert [set(tables[i].seam_sides) for i in range(len(control))] == [
        {"east", "north"}, {"west", "north"},
        {"east", "south"}, {"west", "south"}]
    wide = _zone_boundaries(200, 40)
    clamped = tspec.plan_tiles(200, 200, 125, 125, 18, periodic=False)
    assert [(s.ci0, s.cj0) for s in clamped] == [
        (0, 0), (39, 0), (0, 39), (39, 39)]
    tables = tile_boundary_tables(wide, clamped)
    assert [set(tables[i].seam_sides) for i in range(len(clamped))] == [
        {"east", "north"}, {"west", "north"},
        {"east", "south"}, {"west", "south"}]
    assert [set(window_boundaries(wide, s).seam_sides) for s in clamped] == [
        set(tables[i].seam_sides) for i in range(len(clamped))]
    # one cell wider and the first tile's halo reads column 160, the
    # first of the east zone, from a window that stops at 179
    with pytest.raises(StreamingRefused, match="east") as refused:
        tile_boundary_tables(
            wide, tspec.plan_tiles(200, 200, 143, 143, 18, periodic=False))
    assert "no more than the halo's 18 cells" in str(refused.value)


def test_the_streamed_runners_tables_carry_the_seams_and_refuse_a_stranded_zone():
    """The streamed runner windows each tile on demand
    (streaming.tile_boundary_tables), not through window_boundaries.  Those
    tables once came out with no seam_sides, so every seam of every tile
    relaxed its owned cells 16 to 23 cells in toward the zero placeholder,
    and a tiling that strands zone cells ran instead of being refused."""
    from gpuwm.core.streaming import (StreamingRefused, tile_boundary_tables,
                                      window_boundaries)
    from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         SideBoundary)
    from tilestream import spec as tspec

    nz, ny, nx, width = 2, 120, 120, 25

    def side(shape):
        return SideBoundary(np.zeros(shape), np.zeros(shape))

    field = FieldBoundary(west=side((nz, ny, width)),
                          east=side((nz, ny, width)),
                          south=side((nz, width, nx)),
                          north=side((nz, width, nx)))
    bnd = LateralBoundaries(
        (BoundaryInterval(0.0, 60.0, {"theta": field}),), width, 1, 24)
    specs = tspec.plan_tiles(nx, ny, 40, 40, 16, periodic=False)
    tables = tile_boundary_tables(bnd, specs)
    for index, spec in enumerate(specs):
        assert tables[index].seam_sides == window_boundaries(
            bnd, spec).seam_sides
    assert set(tables[0].seam_sides) == {"east", "north"}
    assert set(tables[4].seam_sides) == {"west", "east", "south", "north"}
    thin = tspec.plan_tiles(nx, ny, 20, 20, 16, periodic=False)
    # refused when the tables are built, before any tile has stepped
    with pytest.raises(StreamingRefused, match="no relaxation"):
        tile_boundary_tables(bnd, thin)


def test_the_child_reads_only_the_frames_of_its_own_window():
    """A 2 h child off a 12 h, 15-minute archive used to be handed all 49
    frames and built 48 boundary intervals on the host, 25 rows deep each
    at ratio 12; it reads 9 of them."""

    from datetime import datetime, timedelta
    from pathlib import Path
    from types import SimpleNamespace

    start = datetime(2023, 6, 21, 18)
    frames = tuple(SimpleNamespace(
        path=Path(f"wrfout_{n:02d}"), valid_time=start + timedelta(minutes=15 * n))
        for n in range(49))
    contract = SimpleNamespace(start_time=start, frames=frames)
    used = downscale._frames_for_child_window(contract, 7200.0)
    assert [p.name for p in used] == [f"wrfout_{n:02d}" for n in range(9)]
    # an end between two frames keeps the frame after it, which closes
    # the last interval
    assert len(downscale._frames_for_child_window(contract, 7300.0)) == 10
    # the whole window keeps the whole archive
    assert len(downscale._frames_for_child_window(contract, 43200.0)) == 49
