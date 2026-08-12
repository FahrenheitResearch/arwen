"""Specified lateral boundaries through the multi-GPU decomposition: geometry.

CPU-hermetic half of task #144.  Everything here is host index arithmetic and
host table slicing -- ``plan_split``'s non-periodic clamp, ``seam_plan``'s
edge-side suppression, the per-rank boundary-table windowing, the corner
ownership data path, and every named refusal.  The stepped bit-identity
question (N forced ranks vs the resident specified run) is
``tilestream.multigpu.gate_forced``'s, on a card; it is deliberately not
restated here as a mock.

Failure modes covered, one test (or parametrized set) each:

* an edge rank grown past the domain, or a wrap seam on a specified axis;
* asymmetric per-side halos read as symmetric in the seam arithmetic;
* boundary tables handed to a rank unwindowed, or windowed to the wrong
  slice, or real tables fabricated on an interior seam side;
* a domain corner whose owning rank sees different table values than the
  resident kernel would;
* a forced decomposition accepted without forcing, with mismatched Davies
  zones, with a quarantine-defeating halo, or with ranks too narrow for the
  relaxation frame.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.streaming import owned_edges, window_boundaries
from gpuwm.ingest.lateral_bc import build_lateral_boundaries
from tilestream import harness
from tilestream import multigpu as mg
from tilestream import spec as tspec

NX, NY, NZ = 220, 168, 3
WIDTH = 5


def forced_cfg(nx: int = NX, ny: int = NY):
    return mg.forced_config(nx, ny)


def synthetic_boundaries(nx: int = NX, ny: int = NY, nz: int = NZ,
                         **zones):
    """Two-time forcing over ramp fields whose value ENCODES its indices.

    ``field[k, j, i] = base + k*1e6 + j*1e3 + i`` makes every windowing
    mistake a value mismatch rather than a plausible neighbour, and the
    second snapshot differs so the tendencies are nonzero.
    """
    def ramp(base, shape):
        k, j, i = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                              np.arange(shape[2]), indexing="ij")
        return base + k * 1.0e6 + j * 1.0e3 + i * 1.0

    def snapshot(base):
        return {
            "u": ramp(base + 10.0, (nz, ny, nx + 1)),
            "v": ramp(base + 20.0, (nz, ny + 1, nx)),
            "theta": ramp(base + 30.0, (nz, ny, nx)),
            "phi": ramp(base + 40.0, (nz, ny, nx)),
            "mu": ramp(base + 50.0, (1, ny, nx)),
        }

    zones.setdefault("spec_bdy_width", WIDTH)
    zones.setdefault("spec_zone", 1)
    zones.setdefault("relax_zone", 4)
    return build_lateral_boundaries([snapshot(0.0), snapshot(100.0)],
                                    [0.0, 3600.0], **zones)


# --------------------------------------------------------------------------
# the non-periodic plan
# --------------------------------------------------------------------------

def test_edge_ranks_clamp_to_domain():
    """A specified edge rank has NO halo outside the domain.

    Its array edge IS the domain edge, which is what puts the boundary
    kernel's writes on real boundary cells.
    """
    halo = 20
    left, right = mg.plan_split(NX, NY, halo, gx=2, gy=1, periodic=False)
    assert left.ci0 == 0 and left.halo_left == 0
    assert left.halo_right == halo
    assert right.ci0 + right.cnx == NX and right.halo_right == 0
    assert right.halo_left == halo
    assert not left.periodic_x and not left.periodic_y


def test_unsplit_axis_keeps_real_edges():
    """gy=1 leaves every rank spanning full y with both real y edges."""
    for s in mg.plan_split(NX, NY, 20, gx=2, gy=1, periodic=False):
        assert s.cj0 == 0 and s.cny == NY
        edges = owned_edges(s)
        assert edges["south"] and edges["north"]


@pytest.mark.parametrize("gy,gx", [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4)])
def test_coverage_exactly_once_non_periodic(gy, gx):
    """Every point of every stagger variant written exactly once."""
    specs = mg.plan_split(NX, NY, 20, gx=gx, gy=gy, periodic=False)
    tspec.validate_plan(specs, NY, NX)


def test_owned_edges_mark_exactly_the_domain_edge_ranks():
    specs = mg.plan_split(NX, NY, 20, gx=2, gy=2, periodic=False)
    marks = [tuple(k for k, v in owned_edges(s).items() if v) for s in specs]
    assert marks == [("west", "south"), ("east", "south"),
                     ("west", "north"), ("east", "north")]


# --------------------------------------------------------------------------
# seams
# --------------------------------------------------------------------------

def test_no_wrap_seams_on_specified_axes():
    """A non-periodic split has interior seams ONLY: no band wraps.

    The periodic 2-way x split has 4 x-bands (each rank has two); clamped
    it has 2, and every logical range stays inside the domain.
    """
    halo = 20
    specs = mg.plan_split(NX, NY, halo, gx=2, gy=1, periodic=False)
    seams = mg.seam_plan(specs, halo)
    assert len(seams) == 2
    for sc in seams:
        lo, hi = sc.logical
        assert 0 <= lo < hi <= NX
    periodic = mg.plan_split(NX, NY, halo, gx=2, gy=1, periodic=True)
    assert len(mg.seam_plan(periodic, halo)) == 4


def test_two_d_non_periodic_seam_census():
    """2x2 clamped: one x band and one y band per rank, phases intact."""
    halo = 20
    specs = mg.plan_split(NX, NY, halo, gx=2, gy=2, periodic=False)
    seams = mg.seam_plan(specs, halo)
    assert len(seams) == 8
    assert sorted(sc.phase for sc in seams) == [0] * 4 + [1] * 4
    for sc in seams:
        lo, hi = sc.logical
        n = NX if sc.axis == "x" else NY
        assert 0 <= lo < hi <= n


def test_seam_offsets_respect_asymmetric_halos():
    """An edge rank's band arithmetic uses ITS OWN side widths.

    Rank 0 of a clamped 3-way x split has halo_left == 0, so its right
    band starts at column ``interior_nx``, not ``halo + interior_nx`` --
    and its neighbour's source offset is the NEIGHBOUR's left halo.
    """
    halo = 20
    specs = mg.plan_split(NX, NY, halo, gx=3, gy=1, periodic=False)
    seams = {(sc.dst_gpu, sc.side): sc for sc in mg.seam_plan(specs, halo)}
    r0_right = seams[(0, "right")]
    assert r0_right.dst_sel["mass"].start == specs[0].interior_nx
    src = r0_right.src_sel["mass"]
    assert src.start >= specs[1].halo_left
    assert src.stop <= specs[1].halo_left + specs[1].interior_nx
    r2_left = seams[(2, "left")]
    assert r2_left.dst_sel["mass"] == slice(0, halo)
    assert r2_left.src_sel["mass"].stop <= (specs[1].halo_left
                                            + specs[1].interior_nx)


def test_mixed_axis_periodicity_plans_per_axis():
    """x clamped, y wrapped: y seams wrap, x seams never do."""
    halo = 16
    specs = mg.plan_split(256, 192, halo, gx=2, gy=2,
                          periodic_x=False, periodic_y=True)
    tspec.validate_plan(specs, 192, 256)
    seams = mg.seam_plan(specs, halo)
    x_bands = [sc for sc in seams if sc.axis == "x"]
    y_bands = [sc for sc in seams if sc.axis == "y"]
    assert len(x_bands) == 4 and len(y_bands) == 8
    for sc in x_bands:
        lo, hi = sc.logical
        assert 0 <= lo < hi <= 256
    assert any(sc.logical[0] < 0 or sc.logical[1] > 192 for sc in y_bands)


# --------------------------------------------------------------------------
# boundary-input distribution
# --------------------------------------------------------------------------

def test_real_sides_are_the_domain_tables_windowed():
    """Edge ranks read the DOMAIN's own tables, sliced tangentially.

    Bit equality against manual slices of the same host arrays, for the
    mass fields and both staggered ones (u's south table is cnx+1 wide,
    v's west table is cny+1 tall).
    """
    bnd = synthetic_boundaries()
    specs = mg.plan_split(NX, NY, 20, gx=2, gy=1, periodic=False)
    g0 = bnd.intervals[0].fields
    for spec, own_side in ((specs[0], "west"), (specs[1], "east")):
        wb = window_boundaries(bnd, spec, seam="zeros")
        fields = wb.intervals[0].fields
        for name in ("u", "v", "theta", "phi", "mu"):
            got = getattr(fields[name], own_side)
            ref = getattr(g0[name], own_side)
            assert np.array_equal(got.value, ref.value), (name, own_side)
            assert np.array_equal(got.tendency, ref.tendency)
        x0 = spec.ci0
        for name, ex in (("theta", 0), ("u", 1), ("mu", 0)):
            got = fields[name].south
            ref = getattr(g0[name], "south")
            assert got.value.shape[-1] == spec.cnx + ex
            assert np.array_equal(got.value,
                                  ref.value[:, :, x0:x0 + spec.cnx + ex])
        got = fields["v"].west
        assert got.value.shape[-2] == spec.cny + 1


def test_seam_sides_are_inert_not_fabricated():
    """An interior seam side gets INERT tables, never real-looking ones."""
    bnd = synthetic_boundaries()
    specs = mg.plan_split(NX, NY, 20, gx=2, gy=1, periodic=False)
    wb = window_boundaries(bnd, specs[0], seam="zeros")
    east = wb.intervals[0].fields["theta"].east
    assert not east.value.any() and not east.tendency.any()
    poisoned = window_boundaries(bnd, specs[0], seam="poison")
    peast = poisoned.intervals[0].fields["theta"].east
    assert peast.value.any()
    again = window_boundaries(bnd, specs[0], seam="poison")
    assert np.array_equal(
        peast.value, again.intervals[0].fields["theta"].east.value)


def test_corner_rank_sees_the_resident_corner_values():
    """The synthetic-corner check: 2x2, each rank owns ONE domain corner.

    At a real corner the rank's local frame coincides with the global one,
    so its west table rows and south table columns at the corner must be
    bit-equal to the domain tables at the same global indices -- including
    the overlap cells both sides describe, where the kernel's Y-owns-corner
    rule picks the south/north table.  Its two seam sides stay inert.
    """
    bnd = synthetic_boundaries()
    specs = mg.plan_split(NX, NY, 20, gx=2, gy=2, periodic=False)
    g0 = bnd.intervals[0].fields["theta"]
    for spec in specs:
        wb = window_boundaries(bnd, spec, seam="zeros")
        fields = wb.intervals[0].fields["theta"]
        edges = owned_edges(spec)
        x0, y0 = spec.ci0, spec.cj0
        for side in ("west", "east", "south", "north"):
            got = getattr(fields, side)
            if not edges[side]:
                assert not got.value.any(), (spec.index, side)
                continue
            ref = getattr(g0, side)
            if side in ("west", "east"):
                assert np.array_equal(got.value,
                                      ref.value[:, y0:y0 + spec.cny, :])
            else:
                assert np.array_equal(got.value,
                                      ref.value[:, :, x0:x0 + spec.cnx])
        # The corner block itself, in the tables the kernel would read.
        xs = slice(0, WIDTH) if edges["west"] else slice(-WIDTH, None)
        ys = slice(0, WIDTH) if edges["south"] else slice(-WIDTH, None)
        y_side = fields.south if edges["south"] else fields.north
        y_ref = g0.south if edges["south"] else g0.north
        gx0 = x0 if edges["west"] else x0 + spec.cnx - WIDTH
        assert np.array_equal(y_side.value[:, :, xs],
                              y_ref.value[:, :, gx0:gx0 + WIDTH])
        x_side = fields.west if edges["west"] else fields.east
        x_ref = g0.west if edges["west"] else g0.east
        gy0 = y0 if edges["south"] else y0 + spec.cny - WIDTH
        assert np.array_equal(x_side.value[:, ys, :],
                              x_ref.value[:, gy0:gy0 + WIDTH, :])


# --------------------------------------------------------------------------
# the forced halo, and the refusals
# --------------------------------------------------------------------------

def test_forced_halo_is_radius_plus_frame_width():
    cfg = forced_cfg()
    assert mg.forced_halo(cfg) == (harness.halo_radius(cfg)
                                   + max(cfg.spec_zone, cfg.relax_zone))


def test_validate_accepts_a_correct_forced_plan():
    cfg = forced_cfg()
    halo = mg.forced_halo(cfg)
    specs = mg.plan_split(NX, NY, halo, gx=2, gy=2, periodic=False)
    mg.validate_forced_plan(cfg, specs, halo, synthetic_boundaries())


def test_refuses_specified_without_boundaries():
    cfg = forced_cfg()
    halo = mg.forced_halo(cfg)
    specs = mg.plan_split(NX, NY, halo, gx=2, gy=1, periodic=False)
    with pytest.raises(mg.MultiGPUError, match="boundaries"):
        mg.validate_forced_plan(cfg, specs, halo, None)


def test_refuses_boundaries_without_specified():
    cfg = harness.make_config(NX, NY)
    specs = mg.plan_split(NX, NY, 16, gx=2, gy=1)
    with pytest.raises(mg.MultiGPUError, match="specified is False"):
        mg.validate_forced_plan(cfg, specs, 16, synthetic_boundaries())


def test_refuses_nested_forcing():
    cfg = harness.make_config(NX, NY, periodic=False, specified=False,
                              nested=True, open_x=False, open_y=False)
    specs = mg.plan_split(NX, NY, 16, gx=2, gy=1, periodic=False)
    with pytest.raises(mg.MultiGPUError, match="nested"):
        mg.validate_forced_plan(cfg, specs, 16, None)


def test_refuses_quarantine_defeating_halo():
    """The dependency radius alone is NOT enough on a forced plan."""
    cfg = forced_cfg()
    bare = harness.halo_radius(cfg)
    specs = mg.plan_split(NX, NY, bare, gx=2, gy=1, periodic=False)
    with pytest.raises(mg.MultiGPUError, match="forced-decomposition"):
        mg.validate_forced_plan(cfg, specs, bare, synthetic_boundaries())
    mg.validate_forced_plan(cfg, specs, bare, synthetic_boundaries(),
                            enforce_halo=False)


def test_refuses_davies_zone_mismatch():
    cfg = forced_cfg()
    halo = mg.forced_halo(cfg)
    specs = mg.plan_split(NX, NY, halo, gx=2, gy=1, periodic=False)
    skewed = synthetic_boundaries(relax_zone=3)
    with pytest.raises(mg.MultiGPUError, match="relax_zone"):
        mg.validate_forced_plan(cfg, specs, halo, skewed)


def test_refuses_rank_narrower_than_the_relaxation_frame():
    """A rank whose array cannot hold a unique interior frame is refused
    by name, with the remedy in the message."""
    cfg = mg.forced_config(64, 8)
    specs = mg.plan_split(64, 8, 2, gx=2, gy=1, periodic=False)
    with pytest.raises(mg.MultiGPUError, match="fewer ranks or a wider"):
        mg.validate_forced_plan(cfg, specs, 2, synthetic_boundaries(),
                                enforce_halo=False)


def test_refuses_interiors_too_narrow_for_the_halo():
    with pytest.raises(mg.MultiGPUError, match="too narrow"):
        mg.plan_split(64, NY, 20, gx=4, gy=1, periodic=False)


def test_identity_plan_is_the_whole_forced_domain():
    """(1,1) forced: one rank, no seams, every edge real."""
    cfg = forced_cfg()
    halo = mg.forced_halo(cfg)
    specs = mg.plan_split(NX, NY, halo, gx=1, gy=1, periodic=False)
    assert len(specs) == 1
    s = specs[0]
    assert (s.ci0, s.cj0, s.cnx, s.cny) == (0, 0, NX, NY)
    assert all(owned_edges(s).values())
    assert mg.seam_plan(specs, halo) == []
    mg.validate_forced_plan(cfg, specs, halo, synthetic_boundaries())
