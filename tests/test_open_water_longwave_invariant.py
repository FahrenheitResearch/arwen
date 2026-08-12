"""Over open water, one surface call must not be able to see GLW at all.

THE CLAIM UNDER TEST.  The shipped account of the v1.6.2 nocturnal-dewpoint
collapse is a longwave-deficit story: a frozen 300 W/m2 downward longwave
cools the skin, the skin's saturation humidity collapses with it, and the
2 m dewpoint follows.  That mechanism runs through a LAND-SURFACE energy
budget.  Over open water gpuwm has no such budget in the surface call: the
SFCLAY kernel derives the surface endpoint from TSK and PSFC alone
(``kernels/sfclay.cu:255`` -- ``qs = EP2*es/(ps/1000 - es)`` for every
non-land column), and TSK over water is prescribed rather than predicted.

So the marine surface call has NO PATH from GLW to the published moisture,
and the invariant is exact rather than approximate: hold everything fixed,
change ONLY the declared longwave, run ONE surface call, and QSFC, QFX, Q2
and TSK must be bit-identical.  Not close -- identical.

WHY BIT-IDENTITY IS THE RIGHT BAR.  A tolerance-based version of this test
would pass on a tree where GLW leaks into the marine path through a small
term, and that leak is exactly the implementation defect worth finding.  Any
difference at all names a variable that moved and is reported as a finding.

WHAT A FAILURE WOULD MEAN.  If a field does move, the shipped causal story
is not merely incomplete for water columns, it is wrong about the mechanism,
because the moisture would be responding to longwave through a route nobody
has described.

WHY THE CARRIER CONTRACT ADMITS THIS CONFIGURATION.  This run selects
Noah with radiation fully off, declares GLW as a constant and never
writes SWDOWN -- on a land domain the contract would refuse it, and
should.  Here the domain is ALL WATER, so Noah consumes exactly zero
cells and there is no consumption for an unsourced SWDOWN to corrupt.
The contract counts the consumer's cells with the runners' own land/ice
predicate before refusing (consumer-cell awareness,
gpuwm/core/radiation_carriers.py), which is what lets this invariant
EXECUTE rather than skip or die at the seam.  One land cell in this
domain restores the full refusal; the contract suite pins both
directions.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu


#: The two skies.  300 W/m2 is the value 1.8.7 seeded silently and the value
#: the failing run therefore ran under; 420 W/m2 is a realistic Gulf-coast
#: overnight downward longwave.  A 120 W/m2 separation is enormous -- if the
#: marine path can see longwave at all, it will show here.
GLW_SEED = 300.0
GLW_REALISTIC = 420.0

#: The fields the invariant is stated over, and the reason each is in the
#: list.  QSFC is the surface endpoint, QFX the flux that moves it, Q2 the
#: published diagnostic, TSK the skin the whole longwave story runs through.
INVARIANT_FIELDS = ("qsfc", "qfx", "q2", "tsk")

#: Read as well, but as EVIDENCE rather than as part of the invariant: if
#: something does move, these say how far up the chain it started.
WITNESS_FIELDS = ("cqs2", "chs", "hfx", "ust", "t2", "th2", "mavail")


def _open_water_state(**overrides):
    """A single-column all-water state with a prescribed skin temperature."""
    from test_physics_driver import _base_config
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    cfg = _base_config(**overrides)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.014 * np.exp(-np.asarray(z, np.float64) / 2400.0))
    return state, cfg


def _run_one_surface_call(glw_value):
    """Initialise over water, run ONE physics step, return the fields."""
    import cupy as cp

    from gpuwm.core.physics import initialize_physics

    state, cfg = _open_water_state(
        ra_physics=0, ra_lw_physics=0, ra_sw_physics=0,
        sf_surface_physics=2, sf_sfclay_physics=1, bl_pbl_physics=0,
        mp_physics=0, cu_physics=0)
    driver = initialize_physics(
        state, cfg,
        # ALL WATER.  landmask 0 and xland 2 is what every water column
        # carries; SST prescribes the skin so nothing in this run predicts
        # TSK, which is the physical reason the invariant should hold.
        landmask=0.0, tsk=301.0, sst=301.0, xice=0.0, snow=0.0,
        snow_depth=0.0, vegfra=0.0, ivgtyp=16, isltyp=14,
        glw=glw_value)
    driver.compute(state, cfg)
    captured = {}
    for name in INVARIANT_FIELDS + WITNESS_FIELDS:
        array = driver.fields.get(name)
        if array is not None:
            captured[name] = cp.asnumpy(array).copy()
    captured["_glw"] = cp.asnumpy(driver.fields["glw"]).copy()
    captured["_xland"] = cp.asnumpy(driver.fields["xland"]).copy()
    return captured


@pytest.mark.gpu
@requires_gpu
def test_open_water_surface_call_is_bit_identical_across_a_120_wm2_sky():
    """THE INVARIANT.  Change only the sky; the water must not notice."""
    seed = _run_one_surface_call(GLW_SEED)
    realistic = _run_one_surface_call(GLW_REALISTIC)

    # The experiment really ran: the treatment is present in the arm.
    # (An exact-zero delta on the treatment variable itself would mean the
    # two arms were the same run and the whole test is void.)
    assert float(seed["_glw"].max()) == GLW_SEED
    assert float(realistic["_glw"].max()) == GLW_REALISTIC
    assert float(seed["_xland"].min()) == 2.0, "arm is not all-water"

    moved = []
    for name in INVARIANT_FIELDS:
        if name not in seed:
            continue
        if not np.array_equal(
                seed[name].view(np.uint32), realistic[name].view(np.uint32)):
            delta = float(np.abs(
                realistic[name].astype(np.float64)
                - seed[name].astype(np.float64)).max())
            moved.append((name, delta))

    witnesses = []
    for name in WITNESS_FIELDS:
        if name not in seed:
            continue
        if not np.array_equal(
                seed[name].view(np.uint32), realistic[name].view(np.uint32)):
            delta = float(np.abs(
                realistic[name].astype(np.float64)
                - seed[name].astype(np.float64)).max())
            witnesses.append((name, delta))

    assert not moved, (
        "GLW leaked into the open-water surface call. Invariant fields that "
        f"moved: {moved}. Upstream witnesses that moved: {witnesses}. "
        "Each named field is a route from downward longwave to published "
        "marine moisture that no documented mechanism accounts for.")
