"""A dated observed perimeter must not emit fuel consumed before initialization."""
import numpy as np
import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize("burn_age", [0.0, 1.0e7])
def test_dated_perimeter_releases_only_current_interval(burn_age):
    import cupy as cp
    from gpuwm.core.sfire import FireOptions, FireState
    from gpuwm.core.sfire_core import CoreOptions, IgnitionLine, fuel_left

    shape = (26, 28)
    z = np.zeros(shape, dtype=np.float32)
    lfn = np.full(shape, 100.0, dtype=np.float32)
    lfn[9:17, 9:19] = -50.0
    tign = np.full(shape, -burn_age, dtype=np.float32)
    state = FireState.from_static(np.ones(shape, dtype=np.float32), z, z, z,
        20.0, 20.0, halo=1, lfn_hist=lfn, historical_tign=tign,
        options=FireOptions(core=CoreOptions(boundary_guard=2),
                            fire_is_real_perim=True),
        ignitions=(IgnitionLine(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),))
    # Independently evaluate the two native consumption times on the supplied
    # geometry. Their difference is the only fuel released by this interval.
    expected_lfn = cp.asarray(lfn)
    expected_tign = cp.asarray(tign)
    start, _ = fuel_left(expected_lfn, expected_tign, state.data["fuel_time"],
                         np.float32(0.0), domain=state.domain)
    end, _ = fuel_left(expected_lfn, expected_tign, state.data["fuel_time"],
                       np.float32(0.5), domain=state.domain)
    state.advance(0.5)
    center = (slice(10, 16), slice(10, 18))
    expected = start[center] - end[center]
    actual = state.data["burnt_area_dt"][center]
    assert cp.asnumpy(actual).tobytes() == cp.asnumpy(expected).tobytes()
    if burn_age:
        assert bool(cp.all(state.data["fuel_frac"][center] == 0))
        for name in ("burnt_area_dt", "fgrnhfx", "fgrnqfx", "dry_burnt_mass"):
            assert bool(cp.all(state.data[name][center] == 0))
    else:
        assert bool(cp.all(actual > 0))
        assert bool(cp.all(state.data["fgrnhfx"][center] > 0))
