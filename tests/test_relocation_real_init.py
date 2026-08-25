"""Leg-3 contracts: real-data relocation (statics rebuild + adjustment).

The load-bearing claim (Drew's design ruling): overlap-region statics
rebuilt from the same source must equal the old ones -- identical source
+ identical cells = identical bytes -- so the bitwise overlap transplant
survives.  This file proves the mechanism that delivers it (the
placement-translated grid's per-cell bitwise stability), proves it
end-to-end against the REAL 30s static build when the case-data GEOG
tree is present, and pins the t=0-lineage terrain adjustment, the
blend-frame rebase, the donor fill's provenance counts, and the
frame-scoped donor-alignment instrument (with its negative control).
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.ingest.nest_init as ni
import gpuwm.ingest.relocation_init as ri
from gpuwm.core.nest_relocation import (Placement, RelocationRefusal,
                                        donor_alignment_check,
                                        plan_relocation)
from gpuwm.static.lambert import LambertGrid
from gpuwm.verify.npref import np_sint

F32 = np.float32


# ---------------------------------------------------------------------------
# 1. The mechanism: placement translation is bitwise per shared cell
# ---------------------------------------------------------------------------

def _demo_like_child_grid():
    """A 2 km, 150-cell Lambert child in the demo's parameter range."""
    return LambertGrid(
        ref_lat=31.2, ref_lon=-89.9, truelat1=21.9, truelat2=41.9,
        stand_lon=-88.8, dx=2000.0, dy=2000.0, e_we=151, e_sn=151)


def test_translated_grid_shares_cell_bytes_with_the_reference():
    g0 = _demo_like_child_grid()
    shifted = g0.translated(24, 12)
    x = np.arange(1.0, 120.0)
    y = np.arange(1.0, 120.0)[:, None]
    lat_s, lon_s = shifted.ij_to_latlon(x, y)
    lat_0, lon_0 = g0.ij_to_latlon(x + 24.0, y + 12.0)
    np.testing.assert_array_equal(lat_s, lat_0)
    np.testing.assert_array_equal(lon_s, lon_0)
    # The float32 WPS sampling twin -- the arithmetic that actually
    # selects source stencils -- shares the property.
    from gpuwm.static.build import _wps32_for

    t_s = _wps32_for(shifted)
    t_0 = _wps32_for(g0)
    lat_s32, lon_s32 = t_s.ij_to_latlon(x, y)
    lat_032, lon_032 = t_0.ij_to_latlon(x + 24.0, y + 12.0)
    np.testing.assert_array_equal(lat_s32, lat_032)
    np.testing.assert_array_equal(lon_s32, lon_032)
    # Negative control: a wrong shift is not the same ground.
    lat_w, _ = g0.ij_to_latlon(x + 23.0, y + 12.0)
    assert not np.array_equal(lat_s, lat_w)


def test_translated_grid_refuses_fractional_cells_and_round_trips():
    g0 = _demo_like_child_grid()
    with pytest.raises(ValueError, match="whole number"):
        g0.translated(1.5, 0)
    back = g0.translated(7, -3).translated(-7, 3)
    lat_a, lon_a = back.ij_to_latlon(5.0, 9.0)
    lat_b, lon_b = g0.ij_to_latlon(5.0, 9.0)
    assert float(lat_a) == float(lat_b) and float(lon_a) == float(lon_b)


# ---------------------------------------------------------------------------
# 2. THE LOAD-BEARING PROOF, against the real 30s static build
# ---------------------------------------------------------------------------

def _geog_root():
    root = os.environ.get("GPUWM_CASE_DATA_ROOT")
    if not root or not Path(root).is_dir():
        return None
    direct = Path(root) / "WPS_GEOG"
    if direct.is_dir():
        return direct
    staged = sorted(Path(root).glob("*/static/WPS_GEOG"))
    return staged[0] if staged else None


@pytest.mark.skipif(_geog_root() is None,
                    reason="GPUWM_CASE_DATA_ROOT/WPS_GEOG not present")
def test_overlap_statics_equality_on_the_real_static_source():
    """Identical source + identical cells = identical bytes, measured on
    the real WPS_GEOG tree: statics built for two placements of the same
    (small) 2 km footprint agree bitwise on every shared cell, for every
    field the build produces."""
    from gpuwm.static.build import GeogSelection, build_static

    root = _geog_root()
    base = LambertGrid(
        ref_lat=31.2, ref_lon=-89.9, truelat1=21.9, truelat2=41.9,
        stand_lon=-88.8, dx=2000.0, dy=2000.0, e_we=25, e_sn=25)
    di, dj = 6, 3  # two parent cells at ratio 3, one event's move
    moved = base.translated(di, dj)
    selection = GeogSelection.fallback(root)
    fields_a = build_static(base, root, selection=selection)
    fields_b = build_static(moved, root, selection=selection)
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=10,
                                 j_parent_start=10),
        placement_to=Placement(grid_id=2, i_parent_start=10 + di // 3,
                               j_parent_start=10 + dj // 3, generation=1),
        parent_grid_ratio=3, child_nx=24, child_ny=24)
    assert (plan.shift_i, plan.shift_j) == (di, dj)
    verdict = ri.overlap_statics_mismatches(
        fields_a, fields_b, plan, names=tuple(sorted(fields_a)))
    assert verdict["compared_cells"] > 0
    assert verdict["mismatched_fields"] == {}
    assert verdict["pass"]


def test_overlap_statics_mismatches_counts_and_fails_on_a_planted_bit():
    ny = nx = 8
    ground = np.arange(40 * 40, dtype=np.float64).reshape(40, 40)

    def footprint(i0, j0):
        return {"HGT_M": ground[j0:j0 + ny, i0:i0 + nx].copy(),
                "LANDMASK": np.ones((ny, nx))}

    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=5,
                                 j_parent_start=5),
        placement_to=Placement(grid_id=2, i_parent_start=7,
                               j_parent_start=5, generation=1),
        parent_grid_ratio=1, child_nx=nx, child_ny=ny)
    old = footprint(5, 5)
    new = footprint(7, 5)
    verdict = ri.overlap_statics_mismatches(old, new, plan)
    assert verdict["pass"] and verdict["mismatched_fields"] == {}
    assert verdict["within_one_ulp"] == {}
    # ONE ULP is physics-identical ground (the climatology resamplers'
    # crop-dependent accumulation order lands there; MEASURED on the
    # 2011-04-27 case, 3 GREENFRAC values at 5.6e-17): counted advisory,
    # not a refusal.
    new["HGT_M"][3, 2] = np.nextafter(new["HGT_M"][3, 2], np.inf)
    verdict = ri.overlap_statics_mismatches(old, new, plan)
    assert verdict["pass"]
    assert verdict["mismatched_fields"] == {}
    assert verdict["within_one_ulp"] == {"HGT_M": 1}
    # Two ULPs is a real drift and still refuses.
    new["HGT_M"][3, 2] = np.nextafter(new["HGT_M"][3, 2], np.inf)
    verdict = ri.overlap_statics_mismatches(old, new, plan)
    assert verdict["mismatched_fields"] == {"HGT_M": 1}
    assert not verdict["pass"]
    # A NaN in the rebuild fails the adjacency compare and refuses.
    new["HGT_M"][3, 2] = np.nan
    verdict = ri.overlap_statics_mismatches(old, new, plan)
    assert verdict["mismatched_fields"] == {"HGT_M": 1}
    assert not verdict["pass"]
    # A category flip on an integer-valued mask refuses exactly as before.
    new = footprint(7, 5)
    new["LANDMASK"] = new["LANDMASK"].astype(np.int32)
    old["LANDMASK"] = old["LANDMASK"].astype(np.int32)
    new["LANDMASK"][2, 2] = 0
    verdict = ri.overlap_statics_mismatches(old, new, plan)
    assert verdict["mismatched_fields"] == {"LANDMASK": 1}
    assert not verdict["pass"]


# ---------------------------------------------------------------------------
# 3. The initializer: t=0-lineage terrain adjustment, CPU-provable
# ---------------------------------------------------------------------------

_NZ, _PNY, _PNX = 3, 14, 14
_CNY = _CNX = 6


def _eta():
    return [1.0, 0.75, 0.45, 0.0][: _NZ + 1]


def _vertical():
    from gpuwm.experiment import VerticalConfig

    return VerticalConfig(p_top=10000.0, hybrid_opt=1, etac=0.2,
                          eta_levels=tuple(_eta()))


def _coord():
    from gpuwm.core.grid import make_vertical_coord

    return make_vertical_coord(
        _NZ, hybrid_opt=1, etac=0.2,
        eta_levels=np.asarray(_eta(), dtype=np.float64))


def _terrain_parent():
    coord = _coord()
    ny, nx, nz = _PNY, _PNX, _NZ
    y, x = np.mgrid[0:ny, 0:nx]
    ht = (120.0 + 35.0 * np.sin(0.7 * x) * np.cos(0.5 * y)).astype(F32)
    mub2d = (90000.0 + 40.0 * (x + 2 * y)).astype(F32)
    pb = (coord.c3h[:, None, None] * mub2d[None]
          + coord.c4h[:, None, None] + 10000.0).astype(F32)
    thb = (300.0 + 10.0 * np.arange(nz)[:, None, None]
           + 0.0 * mub2d[None]).astype(F32)
    alb = (287.0 * thb * (pb / 1.0e5) ** (287.0 / 1004.0) / pb).astype(F32)
    phb = np.concatenate(
        [9.81 * ht[None], 9.81 * ht[None] + np.cumsum(
            np.broadcast_to(800.0 * np.ones((nz, ny, nx)),
                            (nz, ny, nx)), axis=0)]).astype(F32)
    state = SimpleNamespace(
        mub=None, mub2d=mub2d, p_top=10000.0, pb=pb, alb=alb, thb=thb,
        phb=phb, ht=ht,
        u=np.random.default_rng(7).normal(
            0, 1, (nz, ny, nx + 1)).astype(F32),
        v=np.random.default_rng(8).normal(
            0, 1, (nz, ny + 1, nx)).astype(F32),
        w=np.zeros((nz + 1, ny, nx), F32),
        thp=np.random.default_rng(9).normal(0, 0.5, (nz, ny, nx)).astype(F32),
        php=np.zeros((nz + 1, ny, nx), F32),
        mup=np.random.default_rng(10).normal(0, 5, (ny, nx)).astype(F32),
        qv=np.full((nz, ny, nx), 0.008, F32),
        qc=np.zeros((nz, ny, nx), F32), qr=np.zeros((nz, ny, nx), F32),
        h_diabatic=np.zeros((nz, ny, nx), F32))
    for name in ("znw", "znu", "dnw", "rdnw", "dn", "rdn", "fnp", "fnm",
                 "c1f", "c2f", "c3f", "c4f", "c1h", "c2h", "c3h", "c4h"):
        setattr(state, name, np.array(getattr(coord, name), copy=True))
    parent_grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=2000.0, dy=2000.0, e_we=nx + 1, e_sn=ny + 1)
    parent_run = SimpleNamespace(nx=nx, ny=ny, nz=nz)
    return SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, run=parent_run),
        state=state, grid=parent_grid), parent_grid


class _TerrainCpuState:
    """NumPy DomainState twin rich enough for the adjustment sequence."""

    def __init__(self, cfg):
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        self.u = np.zeros((nz, ny, nx + 1), F32)
        self.v = np.zeros((nz, ny + 1, nx), F32)
        self.w = np.zeros((nz + 1, ny, nx), F32)
        self.thp = np.zeros((nz, ny, nx), F32)
        self.php = np.zeros((nz + 1, ny, nx), F32)
        self.mup = np.zeros((ny, nx), F32)
        self.qv = np.zeros((nz, ny, nx), F32)
        self.qc = np.zeros_like(self.qv)
        self.qr = np.zeros_like(self.qv)
        self.h_diabatic = np.zeros_like(self.qv)
        for name in ("u", "v", "w", "thp", "php", "mup", "qv", "qc", "qr"):
            setattr(self, name + "0", np.zeros_like(getattr(self, name)))

    def load_base(self, coord, base):
        for name in ("znw", "znu", "dnw", "rdnw", "dn", "rdn", "fnp",
                     "fnm", "c1f", "c2f", "c3f", "c4f", "c1h", "c2h",
                     "c3h", "c4h"):
            setattr(self, name, np.array(getattr(coord, name), copy=True))
        for name in ("pb", "alb", "thb", "phb"):
            setattr(self, name, np.asarray(getattr(base, name), F32).copy())
        self.mub2d = np.asarray(base.mub, F32).copy()
        self.mub = None
        if base.terrain_z is not None:
            self.ht = np.asarray(base.terrain_z, F32).copy()
        self.p_top = float(base.p_top)

    def total_theta(self):
        return self.thb + self.thp

    def set_map_coriolis(self, *_args, **_kwargs):
        return None


def _fake_diagnostics(state, _hyp):
    """A pure function of the state, identical in every arm."""
    state.p = np.asarray(state.pb, F32).copy()
    state.al = np.asarray(state.alb, F32).copy()
    state.alt = np.asarray(state.alb, F32).copy()


def _footprint_hgt(child_dc, amplitude=90.0):
    """Absolute-ground fine terrain: same ground, same bytes."""
    nx, ny = int(child_dc.run.nx), int(child_dc.run.ny)
    i0 = int(child_dc.i_parent_start)
    j0 = int(child_dc.j_parent_start)
    x = i0 + np.arange(nx)[None, :]
    y = j0 + np.arange(ny)[:, None]
    return (150.0 + amplitude * np.sin(0.9 * x) * np.sin(0.8 * y))


def _child_dc(i0=4, j0=4):
    run = SimpleNamespace(nx=_CNX, ny=_CNY, nz=_NZ, dx=2000.0, dy=2000.0,
                          terrain_opt=1, hypsometric_opt=1,
                          base_temp=290.0, spec_bdy_width=1)
    return SimpleNamespace(grid_id=2, parent_id=1, i_parent_start=i0,
                           j_parent_start=j0, parent_grid_ratio=1,
                           run=run, blend_width=1,
                           start_time=None)


def _cpu_initializer(monkeypatch, parent_node, reference_dc,
                     reference_grid):
    monkeypatch.setattr(ni, "DomainState", _TerrainCpuState)
    monkeypatch.setattr(
        ni, "sint",
        lambda field, reg: np_sint(field, reg, dtype=np.float64).astype(F32))
    monkeypatch.setattr(ni, "update_diagnostics", _fake_diagnostics)

    def fake_statics(grid, _catalog, child_dc):
        assert grid.known_x == pytest.approx(
            1.0 - (child_dc.i_parent_start - reference_dc.i_parent_start))
        fields = {
            "HGT_M": _footprint_hgt(child_dc),
            "LANDMASK": np.ones((child_dc.run.ny, child_dc.run.nx)),
        }
        return fields, SimpleNamespace(root=Path("synthetic")), {}, False

    monkeypatch.setattr(ri, "_build_footprint_statics", fake_statics)
    return ri.real_relocation_initializer(
        catalog=object(), vertical=_vertical(),
        child_config=reference_dc, reference_grid=reference_grid,
        reference_i_parent_start=reference_dc.i_parent_start,
        reference_j_parent_start=reference_dc.j_parent_start)


def test_initializer_blends_terrain_but_takes_no_t0_column_correction(
        monkeypatch):
    """A MOVE IS NOT AN INITIALIZATION, and WRF spells the difference out.

    Both paths blend the same terrain triple.  Only the t = 0 path then
    corrects theta/qv/MU for the base-column-mass change:
    ``adjust_tempqv`` is called at ``mediation_integrate.F:763`` and
    ``press_adj`` set ``.TRUE.`` at :809, while
    ``share/mediation_nest_move.F`` calls the former nowhere and sets the
    latter ``.FALSE.`` for parent (:242) and nest (:261) alike.

    So this pins BOTH halves: the blend and the re-derivation still land
    bitwise, and the column-mass correction demonstrably does NOT fire.
    """
    from gpuwm.core.nest_interp import blend_terrain
    from gpuwm.ingest.real import _make_real_base

    parent_node, parent_grid = _terrain_parent()
    ref_dc = _child_dc(4, 4)
    ref_grid = parent_grid.nest(4, 4, 1, _CNX + 1, _CNY + 1)
    initialize = _cpu_initializer(monkeypatch, parent_node, ref_dc,
                                  ref_grid)
    result = initialize(ref_dc, parent_node)
    state = result.state
    receipt = result.preprocess_receipt
    assert receipt["static_provenance"] == \
        ri.REAL_DATA_FOOTPRINT_REBUILT_STATICS
    assert receipt["placement_translation_child_cells"] == [0, 0]
    assert set(receipt["timings_seconds"]) == {
        "static_rebuild", "atmosphere_sint", "terrain_adjustment"}

    # The t=0 lineage, composed independently: fine analytic base,
    # three-operand blend against the parent SINT captures, then the
    # start_domain re-derivation from the blended MUB.
    coord = _coord()
    hgt = _footprint_hgt(ref_dc)
    fine = _make_real_base(coord, hgt, 10000.0, 290.0, 1)
    ht_e = np.asarray(fine.terrain_z, F32).copy()
    mub_e = np.asarray(fine.mub, F32).copy()
    phb_e = np.asarray(fine.phb, F32).copy()
    reg = ni._mass_registration(ref_dc, parent_node)
    sint32 = lambda f: np_sint(f, reg, dtype=np.float64).astype(F32)
    ht_i = sint32(parent_node.state.ht)
    mub_i = sint32(parent_node.state.mub2d)
    phb_i = sint32(parent_node.state.phb)
    blend_terrain(ht_i, ht_e, spec_bdy_width=1, blend_width=1)
    blend_terrain(mub_i, mub_e, spec_bdy_width=1, blend_width=1)
    blend_terrain(phb_i, phb_e, spec_bdy_width=1, blend_width=1)
    np.testing.assert_array_equal(state.ht, ht_e)
    np.testing.assert_array_equal(state.mub2d, mub_e)
    np.testing.assert_array_equal(state.phb, phb_e)
    stub = SimpleNamespace(ht=ht_e, mub2d=mub_e, phb=phb_e)
    base_e = ni._base_from_blended(stub, ref_dc.run, coord, 10000.0)
    np.testing.assert_array_equal(state.pb, np.asarray(base_e.pb, F32))
    np.testing.assert_array_equal(state.thb, np.asarray(base_e.thb, F32))
    np.testing.assert_array_equal(state.alb, np.asarray(base_e.alb, F32))

    # thp is still REBASED against the newly derived thb, so that total
    # theta survives the base change -- that part a move does need.
    parent_thp = sint32(parent_node.state.thp)
    assert np.count_nonzero(
        state.thp.view(np.uint32) != parent_thp.view(np.uint32)) > 0

    # But press_adj did NOT fire.  It is the only thing in this sequence
    # that writes MU, so MU matching the parent SINT bitwise EVERYWHERE
    # -- inside the blend frame as well as out -- is the whole claim.
    mup_parent = sint32(parent_node.state.mup)
    mup_delta = state.mup.view(np.uint32) != mup_parent.view(np.uint32)
    frame = ni.blend_zone_mask((_CNY, _CNX), spec_bdy_width=1,
                               blend_width=1)
    assert frame.any(), "a blend frame must exist or this proves nothing"
    assert not mup_delta.any(), (
        "press_adj wrote MU on a relocation; WRF sets press_adj = .FALSE. "
        "on a move (share/mediation_nest_move.F:242,261)")

    # And adjust_tempqv did not touch vapour: on a move the child's
    # columns are its own and already consistent with its terrain.
    if parent_node.state.qv is not None:
        np.testing.assert_array_equal(state.qv, sint32(parent_node.state.qv))

    np.testing.assert_array_equal(state.mup0, state.mup)
    # RK seeds re-taken after the re-derivation.
    np.testing.assert_array_equal(state.thp0, state.thp)


def test_initializer_base_channel_is_placement_stable_on_shared_ground(
        monkeypatch):
    """Two placements' rebuilt children agree bitwise on the doubly-
    interior overlap for every base field the donor-alignment instrument
    reads -- the state-level face of the statics-equality ruling."""
    parent_node, parent_grid = _terrain_parent()
    ref_dc = _child_dc(4, 4)
    ref_grid = parent_grid.nest(4, 4, 1, _CNX + 1, _CNY + 1)
    initialize = _cpu_initializer(monkeypatch, parent_node, ref_dc,
                                  ref_grid)
    state_a = initialize(ref_dc, parent_node).state
    moved_dc = _child_dc(5, 4)
    state_b = initialize(moved_dc, parent_node).state
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=4,
                                 j_parent_start=4),
        placement_to=Placement(grid_id=2, i_parent_start=5,
                               j_parent_start=4, generation=1),
        parent_grid_ratio=1, child_nx=_CNX, child_ny=_CNY)
    alignment = donor_alignment_check(
        source_state=state_a, target_state=state_b, plan=plan,
        frame_width=2)
    assert alignment["pass"], alignment
    assert alignment["compared_field_count"] >= 4
    # And the initializer declares exactly that frame to relocate_child.
    assert initialize.donor_alignment_frame_width == 2


def test_initializer_refuses_a_foreign_reference_grid(monkeypatch):
    parent_node, parent_grid = _terrain_parent()
    ref_dc = _child_dc(4, 4)
    wrong = parent_grid.nest(6, 6, 1, _CNX + 1, _CNY + 1)
    initialize = _cpu_initializer(monkeypatch, parent_node, ref_dc, wrong)
    with pytest.raises(RelocationRefusal, match="drift"):
        initialize(ref_dc, parent_node)


def test_interpolation_undershoot_clamp_follows_the_health_rules(
        monkeypatch):
    """The fresh-strip fill floors exactly the fields whose shared health
    rule says >= 0 (non-strict), counts what it floored, and never
    touches a strict-lower field -- found by the first real-data GPU
    move: one Morrison rain-number strip cell at -1.8e-15 out of SINT."""
    state = SimpleNamespace(
        nr=np.array([[0.0, -1.8e-15], [2.0, 3.0]], F32),
        qv=np.array([[1e-3, -1e-9], [1e-3, 1e-3]], F32),
        u=np.array([[-5.0, 5.0]], F32),          # winds may be negative
        mup=np.array([[-4.0, 4.0]], F32),        # strict-lower: untouched
        p=np.array([[-1.0, 1.0]], F32))          # strict-lower: untouched
    counts = ri.clamp_interpolation_undershoot(state)
    assert counts == {"nr": 1, "qv": 1}
    assert float(state.nr.min()) == 0.0 and float(state.qv.min()) == 0.0
    assert float(state.u.min()) == -5.0
    assert float(state.mup.min()) == -4.0 and float(state.p.min()) == -1.0
    # And the initializer records it on every rebuild receipt.
    parent_node, parent_grid = _terrain_parent()
    ref_dc = _child_dc(4, 4)
    ref_grid = parent_grid.nest(4, 4, 1, _CNX + 1, _CNY + 1)
    initialize = _cpu_initializer(monkeypatch, parent_node, ref_dc,
                                  ref_grid)
    receipt = initialize(ref_dc, parent_node).preprocess_receipt
    assert receipt["interpolation_undershoot_clamped"] == {}


# ---------------------------------------------------------------------------
# 4. Donor fill: provenance counts
# ---------------------------------------------------------------------------

def test_donor_fill_counts_and_class_matching():
    ny = nx = 6
    overlap = np.zeros((ny, nx), bool)
    overlap[:, 2:] = True  # a 2-column western strip was exposed
    landmask = np.ones((ny, nx))
    landmask[:3, :] = 0.0  # northern half water
    plan = ri.donor_fill_plan(overlap_mask=overlap, landmask=landmask)
    counts = plan.counts
    assert counts["strip_cells"] == 12
    assert counts["overlap_cells"] == 24
    assert counts["water_donor_filled"] == 6
    assert counts["land_donor_filled"] == 6
    assert counts["class_fallback_filled"] == 0
    field = np.arange(ny * nx, dtype=F32).reshape(ny, nx)
    filled = plan.apply(field)
    # Overlap cells are untouched; each strip cell got its nearest
    # same-class donor (row-wise nearest is the first overlap column).
    np.testing.assert_array_equal(filled[:, 2:], field[:, 2:])
    np.testing.assert_array_equal(filled[:, 1], field[:, 2])
    np.testing.assert_array_equal(filled[:, 0], field[:, 2])
    # Class matching: water strip rows drew water donors (same rows).
    assert (filled[:3, 0] == field[:3, 2]).all()


def test_donor_fill_counts_the_class_fallback_instead_of_hiding_it():
    ny = nx = 4
    overlap = np.zeros((ny, nx), bool)
    overlap[:, 1:] = True
    landmask = np.ones((ny, nx))
    landmask[:, 0] = 0.0  # the exposed strip is water; overlap all land
    plan = ri.donor_fill_plan(overlap_mask=overlap, landmask=landmask)
    assert plan.counts["class_fallback_filled"] == 4
    assert plan.counts["water_donor_filled"] == 0


def test_donor_fill_refuses_an_empty_overlap():
    with pytest.raises(RelocationRefusal, match="non-empty overlap"):
        ri.donor_fill_plan(overlap_mask=np.zeros((3, 3), bool),
                           landmask=np.ones((3, 3)))


def test_donor_fill_apply_handles_soil_stacks():
    overlap = np.zeros((3, 3), bool)
    overlap[:, 1:] = True
    plan = ri.donor_fill_plan(overlap_mask=overlap,
                              landmask=np.ones((3, 3)))
    soil = np.arange(2 * 3 * 3, dtype=F32).reshape(2, 3, 3)
    filled = plan.apply(soil)
    np.testing.assert_array_equal(filled[:, :, 1:], soil[:, :, 1:])
    np.testing.assert_array_equal(filled[:, :, 0], soil[:, :, 1])


# ---------------------------------------------------------------------------
# 5. After the transplant: perturbations carry bitwise, base changes counted
# ---------------------------------------------------------------------------

def test_perturbations_carry_bitwise_and_base_changes_are_counted(monkeypatch):
    """The blend frame resplits; it does NOT preserve the column total.

    This pinned the opposite contract until 2026-08-21.  Preserving the
    total across a blend-frame base change refuses the base-state slab
    between the two effective terrains, and MEASURED on Melissa's eighth
    relocation (d02's southern frame on the Colombian Andes, fine terrain
    2271 m against parent ~1100 m) that manufactured a 4.6 kPa dry-mass
    hole -- ``MU`` at -4590 Pa, the model surface 100 m BELOW its own
    ``HGT`` -- and the column went non-finite six d02 steps later.  WRF's
    ``mediation_nest_move.F`` carries the perturbation arrays and lets
    the totals move; so does this.
    """
    import gpuwm.core.diagnostics as diagnostics

    diagnosed = []
    monkeypatch.setattr(diagnostics, "update_diagnostics",
                        lambda state, hyp: diagnosed.append(hyp))
    ny = nx = 6
    nz = 2
    rng = np.random.default_rng(3)
    thb_src = rng.normal(300, 5, (nz, ny, nx)).astype(F32)
    thb_dst = thb_src.copy()
    thb_dst[:, 0, :] += F32(1.5)  # the incoming child's frame differs
    thp_src = rng.normal(0, 1, (nz, ny, nx)).astype(F32)
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=3,
                                 j_parent_start=3),
        placement_to=Placement(grid_id=2, i_parent_start=3,
                               j_parent_start=3, generation=1),
        parent_grid_ratio=1, child_nx=nx, child_ny=ny)
    source = SimpleNamespace(thp=thp_src, thb=thb_src)
    target = SimpleNamespace(thp=thp_src.copy(), thb=thb_dst)
    cfg = SimpleNamespace(hypsometric_opt=2)
    receipt = ri.rederive_after_transplant(
        source_state=source, target_state=target, plan=plan, cfg=cfg)
    # The base-changed cells are still COUNTED -- that is the receipt's
    # statement of how much ground changed role, and a nonzero count over
    # water is still the wiring defect worth catching.
    assert receipt["base_changed_cells"]["thp"] == nz * nx
    assert receipt["diagnostics_rederived"] and diagnosed == [2]
    assert "bitwise" in receipt["perturbation_carry"]
    # Every cell, frame included, keeps the bitwise stamp.
    np.testing.assert_array_equal(target.thp, thp_src)
    # And the frame's TOTAL therefore moves by exactly the base change,
    # which is the slab of air between the two effective terrains.
    moved = (target.thp[:, 0, :] + thb_dst[:, 0, :]) - (
        thp_src[:, 0, :] + thb_src[:, 0, :])
    np.testing.assert_allclose(moved, F32(1.5), rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Frame-scoped donor alignment, with its negative control
# ---------------------------------------------------------------------------

def _aligned_states(nx=10, ny=10, nz=2, shift=2, frame=3):
    ground = np.arange((ny + shift) * (nx + shift), dtype=np.float64)
    ground = ground.reshape(ny + shift, nx + shift)

    def footprint(i0, j0):
        pb = np.broadcast_to(
            ground[j0:j0 + ny, i0:i0 + nx], (nz, ny, nx)).astype(F32).copy()
        frame_mask = ~ni.blend_zone_mask(
            (ny, nx), spec_bdy_width=frame, blend_width=0)
        # Contaminate the frame with placement-dependent values, exactly
        # as the real blend does.
        pb[:, ~frame_mask] += F32(0.25 * (i0 + 10 * j0))
        return SimpleNamespace(pb=pb)

    return footprint(0, 0), footprint(shift, 0)


def test_alignment_frame_scoping_and_negative_control():
    source, target = _aligned_states()
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=5,
                                 j_parent_start=5),
        placement_to=Placement(grid_id=2, i_parent_start=7,
                               j_parent_start=5, generation=1),
        parent_grid_ratio=1, child_nx=10, child_ny=10)
    scoped = donor_alignment_check(
        source_state=source, target_state=target, plan=plan, frame_width=3)
    assert scoped["pass"], scoped
    unscoped = donor_alignment_check(
        source_state=source, target_state=target, plan=plan)
    assert not unscoped["pass"]
    # NEGATIVE CONTROL: a wrong shift must fail INSIDE the frame too --
    # the scoped instrument still catches a broken plan.
    wrong = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=5,
                                 j_parent_start=5),
        placement_to=Placement(grid_id=2, i_parent_start=6,
                               j_parent_start=5, generation=1),
        parent_grid_ratio=1, child_nx=10, child_ny=10)
    broken = donor_alignment_check(
        source_state=source, target_state=target, plan=wrong,
        frame_width=3)
    assert not broken["pass"]


def test_alignment_refuses_when_the_frame_swallows_the_overlap():
    source, target = _aligned_states()
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=5,
                                 j_parent_start=5),
        placement_to=Placement(grid_id=2, i_parent_start=7,
                               j_parent_start=5, generation=1),
        parent_grid_ratio=1, child_nx=10, child_ny=10)
    blind = donor_alignment_check(
        source_state=source, target_state=target, plan=plan, frame_width=6)
    assert not blind["pass"]
    assert blind["fields"]["pb"].get("interior_empty") is True


# ---------------------------------------------------------------------------
# 7. The preparer: capture, the enforced statics assertion, receipts
# ---------------------------------------------------------------------------

def _preparer_fixture(monkeypatch, *, corrupt_statics=False):
    from gpuwm.runtime import RealRelocationChildPreparer

    ny = nx = 6
    child_dc = _child_dc(4, 4)
    ground = np.arange(20 * 20, dtype=np.float64).reshape(20, 20)

    def statics_for(dc):
        i0, j0 = int(dc.i_parent_start), int(dc.j_parent_start)
        return {
            "HGT_M": ground[j0:j0 + ny, i0:i0 + nx].copy(),
            "LANDMASK": np.ones((ny, nx)),
        }

    old_case = SimpleNamespace(static_fields=statics_for(child_dc))
    model = SimpleNamespace(_prepared_by_grid_id={2: old_case},
                            _activation_context={})
    preparer = RealRelocationChildPreparer(
        exp=SimpleNamespace(), data=SimpleNamespace(), model=model)
    monkeypatch.setattr(preparer, "_rebuild_driver",
                        lambda *args: 0.125)
    driver = SimpleNamespace(fields={
        "tsk": np.arange(ny * nx, dtype=F32).reshape(ny, nx),
        "tslb": np.arange(2 * ny * nx, dtype=F32).reshape(2, ny, nx),
    })
    node = SimpleNamespace(cfg=child_dc,
                           state=SimpleNamespace(physics=driver))
    new_dc = replace_placement(child_dc, 6, 4)
    new_statics = statics_for(new_dc)
    if corrupt_statics:
        new_statics["HGT_M"][2, 1] += 0.5
    initialized = SimpleNamespace(static_fields=new_statics,
                                  grid="new-grid", state=None)
    return preparer, node, new_dc, initialized


def replace_placement(dc, i0, j0):
    return SimpleNamespace(**{**vars(dc), "i_parent_start": i0,
                              "j_parent_start": j0})


def test_preparer_moves_land_state_and_counts_the_fill(monkeypatch):
    preparer, node, new_dc, initialized = _preparer_fixture(monkeypatch)
    preparer.capture_outgoing(node)
    preparer(initialized, new_dc, SimpleNamespace())
    receipt = preparer.last_receipt
    assert receipt["overlap_statics"]["pass"]
    assert receipt["overlap_statics"]["compared_cells"] > 0
    assert receipt["donor_fill"]["strip_cells"] == 2 * 6
    assert receipt["donor_fill"]["overlap_cells"] == 4 * 6
    assert receipt["fields_moved"] == ["tsk", "tslb"]
    assert receipt["accumulators_reinitialized"] is True
    assert receipt["driver_rebuild_seconds"] == 0.125
    # After the runner mutates the node, bookkeeping follows.
    refreshed = []
    preparer.attach_writers(SimpleNamespace(
        refresh_domain=lambda gid, grid, static_fields: refreshed.append(
            (gid, grid))))
    preparer.after_move(node)
    assert refreshed == [(2, "new-grid")]


def test_preparer_refuses_on_a_statics_mismatch(monkeypatch):
    preparer, node, new_dc, initialized = _preparer_fixture(
        monkeypatch, corrupt_statics=True)
    preparer.capture_outgoing(node)
    with pytest.raises(RelocationRefusal, match="identical bytes"):
        preparer(initialized, new_dc, SimpleNamespace())


def test_preparer_refuses_a_rebuild_without_a_capture(monkeypatch):
    preparer, _node, new_dc, initialized = _preparer_fixture(monkeypatch)
    with pytest.raises(RelocationRefusal, match="capture_outgoing"):
        preparer(initialized, new_dc, SimpleNamespace())


# ---------------------------------------------------------------------------
# 6. The corridor arm: the prepared routes' statics_builder is a drop-in
# ---------------------------------------------------------------------------

def _corridor_initializer(monkeypatch, reference_dc, reference_grid):
    """The SAME CPU scaffold as _cpu_initializer, statics from a sealed
    synthetic corridor crop instead of the catalog build."""
    from gpuwm.static.corridor import (ChildStaticsCorridor,
                                       corridor_footprint_statics_builder)

    monkeypatch.setattr(ni, "DomainState", _TerrainCpuState)
    monkeypatch.setattr(
        ni, "sint",
        lambda field, reg: np_sint(field, reg, dtype=np.float64).astype(F32))
    monkeypatch.setattr(ni, "update_diagnostics", _fake_diagnostics)

    # Absolute-ground corridor over the whole 14x14 parent (ratio 1):
    # cell (j, i) carries the same bytes _footprint_hgt evaluates for
    # any footprint covering it, which is the corridor's own contract.
    x = np.arange(1, _PNX + 1)[None, :].astype(np.float64)
    y = np.arange(1, _PNY + 1)[:, None].astype(np.float64)
    corridor = ChildStaticsCorridor(
        geometry={"grid_id": 2, "parent_id": 1, "parent_grid_ratio": 1,
                  "reference_i_parent_start": reference_dc.i_parent_start,
                  "reference_j_parent_start": reference_dc.j_parent_start,
                  "child_nx": _CNX, "child_ny": _CNY,
                  "parent_nx": _PNX, "parent_ny": _PNY,
                  "corridor_nx": _PNX, "corridor_ny": _PNY,
                  "origin_translation_child_cells": [
                      1 - reference_dc.i_parent_start,
                      1 - reference_dc.j_parent_start]},
        fields={"HGT_M": 150.0 + 90.0 * np.sin(0.9 * x) * np.sin(0.8 * y),
                "LANDMASK": np.ones((_PNY, _PNX))},
        cache_sha256="c" * 64)
    return ri.real_relocation_initializer(
        vertical=_vertical(), child_config=reference_dc,
        reference_grid=reference_grid,
        reference_i_parent_start=reference_dc.i_parent_start,
        reference_j_parent_start=reference_dc.j_parent_start,
        statics_builder=corridor_footprint_statics_builder(corridor))


def test_corridor_statics_builder_is_a_drop_in_for_the_catalog_arm(
        monkeypatch):
    """End to end on the CPU scaffold: the corridor-fed initializer
    produces byte-identical rebuilt-child state to the catalog-fed one
    at a MOVED placement, and its receipt states the corridor
    provenance.  This is the prepared tree route's rebuild, minus only
    the GPU physics driver."""
    from gpuwm.static.corridor import (CORRIDOR_REBUILT_STATICS,
                                       CORRIDOR_STRIP_FILL_SOURCE)

    moved = _child_dc(6, 3)

    parent_node, parent_grid = _terrain_parent()
    ref_dc = _child_dc(4, 4)
    ref_grid = parent_grid.nest(4, 4, 1, _CNX + 1, _CNY + 1)
    catalog_arm = _cpu_initializer(monkeypatch, parent_node, ref_dc,
                                   ref_grid)
    catalog_result = catalog_arm(moved, parent_node)

    corridor_arm = _corridor_initializer(monkeypatch, ref_dc, ref_grid)
    assert corridor_arm.static_provenance == CORRIDOR_REBUILT_STATICS
    assert corridor_arm.strip_fill_source == CORRIDOR_STRIP_FILL_SOURCE
    corridor_result = corridor_arm(moved, parent_node)

    np.testing.assert_array_equal(
        corridor_result.static_fields["HGT_M"],
        catalog_result.static_fields["HGT_M"])
    for name in ("ht", "mub2d", "phb", "pb", "thb", "alb", "thp", "mup"):
        np.testing.assert_array_equal(
            getattr(corridor_result.state, name),
            getattr(catalog_result.state, name), err_msg=name)
    receipt = corridor_result.preprocess_receipt
    assert receipt["static_provenance"] == CORRIDOR_REBUILT_STATICS
    assert receipt["static_source"].startswith("statics-corridor d02")
    assert receipt["highres_applied"] is False
    assert receipt["placement_translation_child_cells"] == [2, -1]
