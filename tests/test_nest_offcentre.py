"""Off-centre nest placement through the exchange path (off-centre task).

LES attempt-2 arm-a died at ~04:51 sim with the production gate naming a
NaN in ``nest.scratch.nest_child_field`` on an off-centre d03/d04 tree,
and "off-centre placement" was recorded as the suspect.  These tests are
the from-scratch adjudication of that suspicion, and the standing
regression bed for the storm-following moving-nest program (a relocated
nest is off-centre by definition).

What they establish, by measurement in both directions:

* the parent->child exchange geometry (``register_nest`` donor maps, the
  SINT operator, and the ``bdy_interp1`` force-table build) carries NO
  centred-placement assumption: force tables are exact on linear fields
  at every quadrant and at the minimum legal edge margins, for ratios
  3 and 5, and the residual tables are IDENTICAL between centred and
  off-centre placements;
* the whole force transaction is translation-EQUIVARIANT at the bit
  level: relocating the nest while translating the parent field with it
  reproduces byte-identical boundary tables, so no arithmetic anywhere
  in the table build reads the nest's absolute position;
* the instrument can fail: a seeded one-cell donor shift fires it, and
  the registration guard refuses stencil-overrunning placements
  per-side (asymmetrically), which is what protects the device kernels
  (they carry no bounds checks of their own);
* the one genuine concentric-nest assumption found in the tree -- a
  tornado-LES case module's ``domain_grids`` placement instrument, which
  built every domain centred on the projection reference
  (docs/les/ATTEMPT2-EXPECTATIONS.md section 6) -- is fixed to use the
  runner's own resolver; the case-scoped pin lives in
  tests/test_les_tornado_attempt1_config.py.

The known WRF ``MAX((nri-1)/2,1)`` bdy-wrapper stagger anomaly at even
ratios (nest_interp.py:51-63) is pinned at its exact half-parent-cell
displacement and proven placement-independent, so a future change cannot
silently grow it or misattribute it to placement.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.core.nest_interp import register_nest
from gpuwm.verify.npref import np_bdy_interp1, np_nest_force, np_sint

PARENT = 168
CHILD = 120
CDT = 9.0
STAGGERS = (("m", False, False), ("x", True, False), ("y", False, True))
SIDES = ("west", "east", "south", "north")


def _span(ratio: int) -> int:
    return CHILD // ratio


def _placements(ratio: int) -> dict[str, tuple[int, int]]:
    """Centred plus all four quadrant corners at the minimum legal margin.

    The production containment rule (gpuwm/experiment.py) requires
    ``spec_bdy_width + blend_width`` = 10 parent rows on every side; the
    corner placements sit exactly there.
    """
    span = _span(ratio)
    centred = (PARENT - span) // 2 + 1
    lo, hi = 11, PARENT - span - 9
    return {
        "centred": (centred, centred),
        "corner_sw": (lo, lo),
        "corner_se": (hi, lo),
        "corner_nw": (lo, hi),
        "corner_ne": (hi, hi),
    }


def _register(ratio, ipos, jpos, stagger, wrapper, *, parent=PARENT,
              child=CHILD):
    return register_nest(
        nri=ratio, nrj=ratio, i_parent_start=ipos, j_parent_start=jpos,
        child_nx=child, child_ny=child, parent_nx=parent, parent_ny=parent,
        stagger="" if stagger == "m" else stagger, wrapper=wrapper)


def _positions(n, staggered, *, origin=None, ratio=None):
    """Validated coordinate model: parent mass point m at coordinate m,
    faces at m - 0.5; child points via the WPS cell-edge alignment."""
    idx = np.arange(1, n + 1, dtype=np.float64)
    if origin is None:
        return idx - 0.5 if staggered else idx
    own = (idx - 1.0) if staggered else (idx - 0.5)
    return (origin - 0.5) + own / ratio


def _linear(xpos, ypos, bx=0.25, cy=-0.125):
    return 3.0 + bx * xpos[None, :] + cy * ypos[:, None]


def _force_residuals(ratio, ipos, jpos, *, reg_override=None):
    """Max |SINT(parent) - exact| per (stagger, side), in field units."""
    residuals = {}
    for stagger, sx, sy in STAGGERS:
        reg = (_register(ratio, ipos, jpos, stagger, "bdy")
               if reg_override is None else reg_override(stagger))
        nxp, nyp = PARENT + sx, PARENT + sy
        nxc, nyc = CHILD + sx, CHILD + sy
        cfld = _linear(_positions(nxp, sx), _positions(nyp, sy))
        nfld = _linear(
            _positions(nxc, sx, origin=ipos, ratio=ratio),
            _positions(nyc, sy, origin=jpos, ratio=ratio))
        tables = np_bdy_interp1(
            cfld[None], nfld[None], reg, parent_dt_fp32=CDT,
            parent_interval_ticks=3, spec_zone=1, relax_zone=4,
            spec_bdy_width=5, dtype=np.float64)
        for side in SIDES:
            value, tendency = tables[side]
            assert np.isfinite(value).all(), (stagger, side)
            assert np.isfinite(tendency).all(), (stagger, side)
            residuals[stagger, side] = float(np.max(np.abs(tendency)) * CDT)
    return residuals


@pytest.mark.parametrize("ratio", [3, 5])
@pytest.mark.parametrize("name", ["centred", "corner_sw", "corner_se",
                                  "corner_nw", "corner_ne"])
def test_force_tables_exact_on_linear_fields_at_every_placement(ratio, name):
    """SINT is exact on linear data, so a nonzero tendency against an
    exactly interpolated child is a geometry error.  The bound is the
    FP32 XIG-table rounding (registered deviation), measured 1.5e-8."""
    ipos, jpos = _placements(ratio)[name]
    residuals = _force_residuals(ratio, ipos, jpos)
    worst = max(residuals.values())
    assert worst <= 1.0e-6, (name, residuals)


@pytest.mark.parametrize("ratio", [2, 3, 5])
def test_force_residuals_are_placement_independent(ratio):
    """Centred and every off-centre residual table must be identical: any
    difference means the table build read the nest's absolute position.
    Ratio 2 is included so the even-ratio stagger anomaly is proven
    placement-independent too."""
    placements = _placements(ratio)
    base = _force_residuals(ratio, *placements["centred"])
    for name, (ipos, jpos) in placements.items():
        if name == "centred":
            continue
        other = _force_residuals(ratio, ipos, jpos)
        delta = max(abs(other[key] - base[key]) for key in base)
        assert delta <= 5.0e-8, (name, delta)


@pytest.mark.parametrize("ratio", [2, 3, 5])
def test_force_transaction_is_translation_equivariant_bitwise(ratio):
    """Relocate the nest and translate the parent field with it: every
    boundary VALUE and TENDENCY table must be BYTE-identical.  The donor
    maps translate exactly with i/j_parent_start, so any bit difference
    localizes an absolute-position read inside the table build.  This is
    the direct falsifier for 'off-centre placement changes the exchange
    arithmetic', and the standing invariant a moving nest relies on."""
    placements = _placements(ratio)
    ip0, jp0 = placements["centred"]
    rng = np.random.default_rng(20260806)
    span = _span(ratio)
    # One reusable random parent pattern, indexed relative to the nest
    # origin so it can be translated with the placement.
    pattern = rng.standard_normal(
        (PARENT + 1 + 2 * PARENT, PARENT + 1 + 2 * PARENT))

    def tables_for(ipos, jpos):
        out = {}
        for stagger, sx, sy in STAGGERS:
            reg = _register(ratio, ipos, jpos, stagger, "bdy")
            nxp, nyp = PARENT + sx, PARENT + sy
            nxc, nyc = CHILD + sx, CHILD + sy
            jj, ii = np.meshgrid(
                np.arange(nyp) - (jpos - 1) + PARENT,
                np.arange(nxp) - (ipos - 1) + PARENT, indexing="ij")
            cfld = np.ascontiguousarray(pattern[jj, ii], dtype=np.float64)
            child = rng2.standard_normal((nyc, nxc))
            out[stagger] = np_bdy_interp1(
                cfld[None], child[None], reg, parent_dt_fp32=CDT,
                parent_interval_ticks=3, spec_zone=1, relax_zone=4,
                spec_bdy_width=5, dtype=np.float32)
        return out

    rng2 = np.random.default_rng(7)
    reference = tables_for(ip0, jp0)
    for name, (ipos, jpos) in placements.items():
        if name == "centred":
            continue
        rng2 = np.random.default_rng(7)   # identical child every placement
        moved = tables_for(ipos, jpos)
        for stagger, _sx, _sy in STAGGERS:
            for side in SIDES:
                for slot in (0, 1):
                    want = np.asarray(reference[stagger][side][slot],
                                      dtype=np.float64)
                    got = np.asarray(moved[stagger][side][slot],
                                     dtype=np.float64)
                    assert np.array_equal(want, got), (
                        f"{name} {stagger}/{side} slot {slot} differs: the "
                        "force-table build read the absolute nest position")
    assert span >= 1  # geometry sanity, keeps span used


def test_the_linear_instrument_fires_on_a_seeded_donor_shift():
    """Negative control: shift the east half of the x donor map by one
    parent cell.  The probe must report exactly one cell of slope."""
    ipos = jpos = (PARENT - _span(3)) // 2 + 1

    def seeded(stagger):
        reg = _register(3, ipos, jpos, stagger, "bdy")
        reg.ci[reg.ci > int(np.mean(reg.ci))] += 1
        return reg

    residuals = _force_residuals(3, ipos, jpos, reg_override=seeded)
    assert max(residuals.values()) >= 0.2, residuals


def test_even_ratio_bdy_stagger_anomaly_is_pinned_half_parent_cell():
    """WRF's bdy_interp1 uses ioff = MAX((nri-1)/2, 1), which at nri=2
    displaces the STAGGERED-direction boundary donors by exactly +0.5
    parent cells (one child cell).  Transliterated faithfully and pinned
    (nest_interp.py:51-63).  On a unit-slope field the residual is the
    displacement itself; the mass direction stays exact.  This pin keeps
    the anomaly (a) from growing and (b) from being mistaken for a
    placement effect -- it is identical at every placement."""
    bx, cy = 0.25, -0.125
    for name, (ipos, jpos) in _placements(2).items():
        residuals = _force_residuals(2, ipos, jpos)
        for side in SIDES:
            assert residuals["m", side] <= 1.0e-6, (name, side)
            assert residuals["x", side] == pytest.approx(
                abs(bx) * 0.5, abs=1e-9), (name, side)
            assert residuals["y", side] == pytest.approx(
                abs(cy) * 0.5, abs=1e-9), (name, side)


def test_init_interpolation_exact_at_every_placement_and_ratio():
    """The init-path wrapper='interp' geometry (parent_only_init and the
    real-data blend capture) is exact on linear fields for ALL ratios,
    even ratio 2, at every placement."""
    bx, cy = 0.25, -0.125
    for ratio in (2, 3, 5):
        for name, (ipos, jpos) in _placements(ratio).items():
            for stagger, sx, sy in STAGGERS:
                reg = _register(ratio, ipos, jpos, stagger, "interp")
                nxp, nyp = PARENT + sx, PARENT + sy
                nxc, nyc = CHILD + sx, CHILD + sy
                cfld = _linear(_positions(nxp, sx), _positions(nyp, sy),
                               bx, cy)
                exact = _linear(
                    _positions(nxc, sx, origin=ipos, ratio=ratio),
                    _positions(nyc, sy, origin=jpos, ratio=ratio), bx, cy)
                got = np_sint(cfld, reg, dtype=np.float64)
                worst = float(np.max(np.abs(got - exact)))
                assert worst <= 1.0e-6, (ratio, name, stagger, worst)


def test_registration_guard_refuses_stencil_overrun_per_side():
    """The device kernels have no bounds checks; the ONLY protection is
    the registration-time donor guard, so it must refuse each side
    independently (asymmetrically), not just symmetric violations."""
    ratio = 3
    span = _span(ratio)
    centred = (PARENT - span) // 2 + 1
    # Legal extremes construct fine.
    for ipos in (11, PARENT - span - 9):
        _register(ratio, ipos, centred, "x", "bdy")
        _register(ratio, centred, ipos, "y", "bdy")
    # One cell past the parent's west/south stencil floor refuses.
    with pytest.raises(ValueError, match="SINT stencil"):
        _register(ratio, 1, centred, "m", "bdy")
    with pytest.raises(ValueError, match="SINT stencil"):
        _register(ratio, centred, 1, "m", "bdy")
    # And past the east/north ceiling refuses too.
    with pytest.raises(ValueError, match="SINT stencil"):
        _register(ratio, PARENT - span + 1, centred, "m", "bdy")
    with pytest.raises(ValueError, match="SINT stencil"):
        _register(ratio, centred, PARENT - span + 1, "m", "bdy")


# --- coupler-level fixtures (the test_nest_coupler.py pattern, with the
# --- placement as a parameter and a 24-cell parent so a 10-cell child can
# --- sit at genuinely asymmetric legal extremes) -------------------------

def _run_cfg(nx, ny, *, nested, grid_id):
    from gpuwm.config import RunConfig

    return RunConfig(nx=nx, ny=ny, nz=2, dx=1000.0, dy=1000.0,
                     ztop=10000.0, dt=3.0, run_seconds=9.0,
                     nested=nested, specified=not nested,
                     grid_id=grid_id, spec_bdy_width=5, spec_zone=1,
                     relax_zone=4)


def _clock(grid_id, parent_id, *, step_ticks, dt, advanced=False):
    from gpuwm.core.clock import DomainClock, DomainTicks

    spec = DomainTicks(
        grid_id=grid_id, parent_id=parent_id, parent_time_step_ratio=3,
        step_ticks=step_ticks, dt_fp32=np.float32(dt),
        history_ticks=100, restart_ticks=None, radt_ticks=None,
        stepra=None, cudt_ticks=None, stepcu=None, bldt_ticks=None,
        stepbl=None)
    clock = DomainClock(spec, tick_den=1, run_ticks=1000)
    if advanced:
        clock.advance()
    return clock


class _State:
    def __init__(self, run):
        nz, ny, nx = run.nz, run.ny, run.nx
        self.mub2d = (np.arange(ny * nx, dtype=np.float32)
                      .reshape(ny, nx) / 9 + 5)
        self.mup = np.full((ny, nx), np.float32(0.25))
        self.u = np.full((nz, ny, nx + 1), np.float32(2.0))
        self.v = np.full((nz, ny + 1, nx), np.float32(-1.5))
        self.w = np.full((nz + 1, ny, nx), np.float32(0.75))
        self.thp = np.full((nz, ny, nx), np.float32(1.25))
        self.php = np.full((nz + 1, ny, nx), np.float32(3.0))
        self.thb = np.array([300.0, 302.0], dtype=np.float32)
        self.c1h = np.array([0.8, 0.6], dtype=np.float32)
        self.c2h = np.array([1.0, 2.0], dtype=np.float32)
        self.c1f = np.array([1.0, 0.7, 0.4], dtype=np.float32)
        self.c2f = np.array([0.0, 1.5, 3.0], dtype=np.float32)
        self.msft = np.full((ny, nx), np.float32(1.25))
        self.msfu = np.full((ny, nx + 1), np.float32(1.5))
        self.msfv = np.full((ny + 1, nx), np.float32(1.75))
        self.qv = np.full((nz, ny, nx), np.float32(0.01))
        self.has_msf = True
        self._scratch = {}
        self.lateral_boundaries = None

    def scratch(self, shape, slot, dtype=None):
        dtype = np.dtype(np.float32 if dtype is None else dtype)
        shape = tuple(shape)
        if slot not in self._scratch:
            self._scratch[slot] = np.zeros(shape, dtype=dtype)
        result = self._scratch[slot]
        assert result.shape == shape and result.dtype == dtype
        return result


class _DeviceState:
    def __init__(self, host, cp):
        self._cp = cp
        self._scratch = {}
        for name, value in vars(host).items():
            if name == "_scratch":
                continue
            setattr(self, name, cp.asarray(value) if isinstance(
                value, np.ndarray) else value)

    def scratch(self, shape, slot, dtype=None):
        dtype = self._cp.dtype(self._cp.float32 if dtype is None else dtype)
        shape = tuple(shape)
        if slot not in self._scratch:
            self._scratch[slot] = self._cp.zeros(shape, dtype=dtype)
        result = self._scratch[slot]
        assert result.shape == shape and result.dtype == dtype
        return result


#: A 10-parent-cell child in a 24-cell parent: the SINT guard admits
#: i_parent_start in [3, 13], so these extremes are truly asymmetric.
_COUPLER_PLACEMENTS = {
    "centred": (8, 8),
    "corner_sw": (3, 3),
    "corner_ne": (13, 13),
    "corner_se": (13, 3),
}


def _coupler_nodes(ipos, jpos):
    prun = _run_cfg(24, 24, nested=False, grid_id=1)
    crun = _run_cfg(30, 30, nested=True, grid_id=2)
    pcfg = SimpleNamespace(grid_id=1, parent_id=0, parent_grid_ratio=1,
                           i_parent_start=1, j_parent_start=1, run=prun)
    ccfg = SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=3,
                           i_parent_start=ipos, j_parent_start=jpos,
                           run=crun)
    parent = SimpleNamespace(
        cfg=pcfg, state=_State(prun),
        clock=_clock(1, 0, step_ticks=3, dt=9.0, advanced=True))
    child_clock = _clock(2, 1, step_ticks=1, dt=3.0)
    child_clock.prepare_step()
    child = SimpleNamespace(cfg=ccfg, state=_State(crun), parent=parent,
                            clock=child_clock)
    return parent, child


def test_coupler_registration_manifest_holds_at_offcentre_placements():
    """NestCoupler construction (geometry + F4/F16 manifest shapes) is
    placement-clean: the slot manifest an off-centre child requests is
    identical to the centred child's, and geometry binds against it."""
    from gpuwm.core.nest import NestCoupler

    manifests = {}
    for name, (ipos, jpos) in _COUPLER_PLACEMENTS.items():
        _, child = _coupler_nodes(ipos, jpos)
        coupler = NestCoupler(child)
        manifests[name] = dict(coupler.slot_shapes)
        for stagger, reg in coupler.registrations.items():
            assert reg.i_parent_start == ipos
            assert reg.j_parent_start == jpos
            assert reg.ci.shape == (30 + (1 if stagger == "x" else 0),)
            assert reg.cj.shape == (30 + (1 if stagger == "y" else 0),)
    for name, manifest in manifests.items():
        assert manifest == manifests["centred"], name


@pytest.mark.parametrize("name", ["centred", "corner_sw", "corner_ne",
                                  "corner_se"])
def test_cpu_mirror_force_tables_finite_at_offcentre_placements(name):
    """The complete per-field REAL-emulation force build (coupling +
    SINT + tendency) stays finite at every placement on the coupler
    fixture states."""
    from gpuwm.core.nest import NestCoupler

    parent, child = _coupler_nodes(*_COUPLER_PLACEMENTS[name])
    coupler = NestCoupler(child)
    tables = np_nest_force(
        parent.state, child.state, coupler.registrations,
        field_names=("u", "v", "w", "t", "ph", "mu", "qv"),
        parent_dt_fp32=parent.clock.spec.dt_fp32,
        parent_interval_ticks=parent.clock.spec.step_ticks,
        spec_zone=1, relax_zone=4, spec_bdy_width=5, dtype=np.float32)
    for field, sides in tables.items():
        for side, (value, tendency) in sides.items():
            assert np.isfinite(value).all(), (name, field, side)
            assert np.isfinite(tendency).all(), (name, field, side)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("name", ["corner_sw", "corner_ne", "corner_se"])
def test_gpu_force_tables_match_mirror_at_offcentre_placements(name):
    """The device transaction (couple_nest_field + nest_bdy_interp1 +
    attach) at off-centre and near-edge placements: finite everywhere
    and bit-equal to the independent REAL-emulation mirror -- the same
    parity the centred fixture has always pinned, extended to the
    placements the moving-nest program needs.  The device kernels have
    no bounds checks, so this is the direct artifact-level check that
    off-centre donor tables drive only in-bounds reads."""
    import cupy as cp

    from gpuwm.core.nest import NestCoupler

    host_parent, host_child = _coupler_nodes(*_COUPLER_PLACEMENTS[name])
    parent = SimpleNamespace(
        cfg=host_parent.cfg, state=_DeviceState(host_parent.state, cp),
        clock=host_parent.clock)
    child = SimpleNamespace(
        cfg=host_child.cfg, state=_DeviceState(host_child.state, cp),
        parent=parent, clock=host_child.clock)
    coupler = NestCoupler(child)
    expected = np_nest_force(
        host_parent.state, host_child.state, coupler.registrations,
        field_names=("u", "v", "w", "t", "ph", "mu"),
        parent_dt_fp32=parent.clock.spec.dt_fp32,
        parent_interval_ticks=parent.clock.spec.step_ticks,
        spec_zone=1, relax_zone=4, spec_bdy_width=5, dtype=np.float32)
    coupler.force(child)

    application_name = {"t": "theta", "ph": "phi"}
    for field in ("u", "v", "w", "t", "ph", "mu"):
        got_field = coupler._last_tables[application_name.get(field, field)]
        for side in ("west", "east", "south", "north"):
            for got, want in zip(got_field[side], expected[field][side]):
                got_host = cp.asnumpy(got)
                assert np.isfinite(got_host).all(), (name, field, side)
                np.testing.assert_array_equal(
                    got_host, np.float32(want),
                    err_msg=f"{name}/{field}/{side}")


# The companion pin for the one concentric-nest assumption this task
# actually found and fixed -- the tornado-LES case's ``domain_grids``
# placement instrument -- lives with its case in
# tests/test_les_tornado_attempt1_config.py
# (test_domain_grids_honours_offcentre_placement), keeping this module
# case-name-free.
