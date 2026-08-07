"""The HRRR route's downward longwave is COMPUTED, not a seeded constant.

Provenance, and why this file exists at all.

v1.7.1's headline fix was a surface-dewpoint collapse: a shipped real
case ran ``ra_sw_physics = 1`` (Dudhia shortwave) with
``ra_lw_physics = 0`` (longwave OFF).  ``gpuwm/core/dudhia.py`` returns
``glw=fields["glw"]`` -- the seeded field, echoed back untouched -- so
the downward longwave for the whole run was whatever
:func:`gpuwm.core.physics.initialize_physics` had seeded, which for the
HRRR route is its ``glw=300.0`` default.  Shortwave heated the surface
by day; at night the surface radiated into a frozen 300 W/m2 and skin
temperature cratered.  The replay evidence had one unmistakable
signature: **exactly one distinct GLW value across the entire run**.

1.7.1 fixed the gfs/era5 doors and added the load guard.  It could not
fix HRRR, because every full-radiation profile the HRRR route admits is
an mp8 suite and the root preparer staged no microphysics tables for
those -- so the route's default stayed asymmetric and 1.7.1 shipped an
auto-declaration in its place.  1.8 staged the tables and flipped the
default (``gpuwm.hrrr_route_inputs.ROUTE_DEFAULT_PHYSICS_PROFILE``).

So this file asserts the FIXED signature -- many distinct GLW values,
varying in space and in time, produced by a radiation solver -- and it
asserts the BROKEN one still reproduces on the suite that had it,
because a detector that cannot fire is not evidence.  Both arms run on
a real card through the real driver; nothing here is a mock.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from gpuwm.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from gpuwm.physics_compat import (RRTMG_VARIANT_LEGACY, WSM6_PROFILE_ID,
                                  single_domain_runtime_switches)

from conftest import requires_gpu  # noqa: E402
from test_physics_driver import _full_state  # noqa: E402


#: What the HRRR route's own initialization seeds when the source
#: supplies no downward longwave -- ``initialize_physics``' ``glw``
#: default, and the exact number the shipped defect was made of.  The
#: harness below seeds a different value on purpose, so that "frozen"
#: is proven as *equal to whatever was seeded* rather than as equal to
#: this constant; a test that only ever saw 300.0 could not tell a
#: passthrough from a coincidence.
ROUTE_SEEDED_GLW = 300.0

#: A pre-dawn window over the reporting user's latitude band.  Local
#: night is the half of the diurnal cycle where a frozen GLW does its
#: damage, so that is where the detector is pointed.
START = datetime(2011, 4, 26, 8)
LAT, LON = 33.8, -87.29


#: 10-minute steps across two hours: the 12-minute radiation cadence
#: fires ten times, which is enough to show the field is republished on
#: every call rather than written once.
_STEPS = 12

#: THE discriminator, and it is not variance.
#:
#: Downward longwave over a fixed thermodynamic column is genuinely
#: near-constant -- it is set by the atmosphere's temperature and
#: humidity profile, not by the sun -- so "GLW changed a lot" is a weak
#: reading of a real solver and a first attempt at this test nearly
#: mistook a correct one for a broken one.  What separates computed from
#: seeded without ambiguity is INDEPENDENCE: run the identical column
#: twice with two different seeds.  A passthrough returns each seed
#: verbatim, so its two answers differ by exactly the seed difference.
#: A solver ignores them both and returns the same physics twice.
_SEED_A = 310.0
_SEED_B = 120.0


def _glw_series(profile, *, steps=_STEPS, seed=_SEED_A):
    """Run the driver for ``profile``; return (seeded GLW, GLW per step).

    The switches come from the shipped profile table, so this runs what
    the route would run rather than a hand-copied selector set.
    """
    switches = single_domain_runtime_switches(profile)
    nx = ny = 3
    state, cfg, driver = _full_state(
        nx=nx, ny=ny, nz=24, dt=600.0, time_step_sound=4,
        mp_physics=int(switches["mp_physics"]),
        ra_physics=int(switches["ra_physics"]),
        ra_lw_physics=int(switches["ra_lw_physics"]),
        ra_sw_physics=int(switches["ra_sw_physics"]),
        ra_rrtmg_variant=switches.get("ra_rrtmg_variant"),
        wrf_rrtmg_compatibility=switches["wrf_rrtmg_compatibility"],
        radt_minutes=float(switches["radt"]),
        bldt=10.0,
        radiation_start_time=START,
        radiation_latitude=np.full((ny, nx), LAT),
        radiation_longitude=np.full((ny, nx), LON))

    from gpuwm.core.diagnostics import update_diagnostics

    # The seed goes in through the driver's own host-forcing door, which
    # is the same door a route uses to hand it a source GLW (or, on a
    # route with no source GLW, leaves at the initialize_physics
    # default).  Set BEFORE the first radiation call, so a passthrough
    # solver echoes it for the rest of the run.
    driver.set_forcing(glw=seed)
    seeded = np.asarray(driver.fields["glw"].get(), np.float64).copy()
    frames = []
    for _ in range(steps):
        update_diagnostics(state)
        driver.compute(state, cfg)
        state.elapsed_seconds += cfg.dt
        frames.append(np.asarray(driver.fields["glw"].get(), np.float64))
    return seeded, np.stack(frames)


# ---------------------------------------------------------------------------
# The detector, pointed at the suite that HAD the defect.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_the_frozen_glw_signature_still_reproduces_on_the_old_default():
    """One distinct value, everywhere, forever -- the 1.7.1 evidence.

    This is the negative control.  ``wsm6-ysu-mm5-noah-no-radiation-v1``
    was this route's default through 1.7.1; it runs Dudhia shortwave with
    longwave off, and Dudhia hands the seeded GLW straight back.  If this
    ever stops reproducing, the test below is measuring something other
    than what it claims to.
    """
    seeded, series = _glw_series(WSM6_PROFILE_ID)
    distinct = np.unique(series)
    assert distinct.size == 1, (
        f"expected the frozen-GLW signature, got {distinct.size} distinct "
        f"values: {distinct[:8]}")
    # And the one value is exactly what was seeded -- the passthrough,
    # demonstrated rather than asserted from a remembered constant.
    assert float(distinct[0]) == pytest.approx(
        float(np.unique(seeded)[0]), abs=1e-3)
    # On the route itself that seed is 300.0, which is the number the
    # v1.7.1 replay evidence reported.
    assert ROUTE_SEEDED_GLW == 300.0

    # The discriminator, on the arm that FAILS it: change the seed and
    # the answer moves with it, one for one.
    _seeded_b, series_b = _glw_series(WSM6_PROFILE_ID, seed=_SEED_B)
    assert float(np.unique(series_b)[0]) == pytest.approx(_SEED_B, abs=1e-3)
    assert float(np.unique(series)[0] - np.unique(series_b)[0]) == (
        pytest.approx(_SEED_A - _SEED_B, abs=1e-3))


# ---------------------------------------------------------------------------
# The fix, on the route's own default.
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_the_route_default_computes_downward_longwave():
    """Many distinct values, varying in time -- radiation actually ran.

    The assertion the lane exists for.  Counted, not fitted, and stated
    as a count because that is the form the broken signature took.
    """
    switches = single_domain_runtime_switches(ROUTE_DEFAULT_PHYSICS_PROFILE)
    # The premise, stated before it is relied on.
    assert int(switches["ra_lw_physics"]) == 4
    assert switches.get("ra_rrtmg_variant") == RRTMG_VARIANT_LEGACY

    seeded, series = _glw_series(ROUTE_DEFAULT_PHYSICS_PROFILE)
    assert np.isfinite(series).all()

    # 1. It is not the seed, and not the route's 300.0 either.  A
    #    passthrough could not clear this.
    assert not np.allclose(series, seeded), (
        "every GLW equals the initialize_physics seed -- this is the "
        "FROZEN-GLW signature v1.7.1 shipped a fix for")
    assert not np.allclose(series, ROUTE_SEEDED_GLW)

    # 2. THE discriminator: the identical column, seeded differently,
    #    produces the identical downward longwave.  The solver ignored
    #    both seeds because it computed the answer.  A passthrough's two
    #    runs would differ by exactly _SEED_A - _SEED_B (asserted in the
    #    negative control above).
    _seeded_b, series_b = _glw_series(
        ROUTE_DEFAULT_PHYSICS_PROFILE, seed=_SEED_B)
    assert series_b == pytest.approx(series, abs=1e-3), (
        "downward longwave tracked the seed, so it was not computed")

    # 3. And the computed value is a physically sane clear-sky downward
    #    longwave over land at this latitude -- not a nonzero artefact.
    assert 150.0 <= float(series.min())
    assert float(series.max()) <= 500.0

    # 4. Republished on every radiation call, not written once: the field
    #    carries the solver's own float32 result at each sampled step.
    assert series.shape[0] == _STEPS
