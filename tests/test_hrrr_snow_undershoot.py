"""The native HRRR decoder puts its own snow undershoot at zero.

It maps snow water and snow depth with the overlapping parabola, whose
negative weights take a patch with little snow inside a snowpack below
zero by up to 9/32 of the snow around it.  A 2x2 patch holding a trace of
snow (a valley floor that has nearly melted out) inside snow at the
field's maximum reaches about -17/64 of it at the patch centre, past the
quarter the snow admission used to refuse beyond.  (A patch of exactly
zero maps to zero at its centre: WPS's ``oned`` answers zero where both
middle points are zero.)  From a non-negative source every negative
result is that undershoot, so the decoder zeroes it where the source is
known and leaves every other value as the parabola made it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gpuwm.ingest.hrrr import interpolate_hrrr_to_lambert
from test_hrrr_island_donor import (_HostBackend, _ISLAND, _nest_grid,
                                    _window_snapshot)


def _bare_patches_in_snow(shape, depth):
    """Snow at ``depth`` with a 2x2 patch of a trace every four cells."""
    snow = np.full(shape, depth, dtype=np.float32)
    for row in range(2, shape[0] - 2, 4):
        for column in range(2, shape[1] - 2, 4):
            snow[row:row + 2, column:column + 2] = 1.0e-4 * depth
    return snow


def _map(snapshot):
    landmask = np.zeros((216, 222))
    for row, column in _ISLAND:
        landmask[row, column] = 1.0
    return interpolate_hrrr_to_lambert(
        snapshot, _nest_grid(), target_landmask=landmask,
        surface_fallback_radius=8, backend=_HostBackend(),
        target_name="domain 2")


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_snow_beside_nearly_bare_patches_maps_to_no_negative(capsys):
    base = _window_snapshot()
    shape = base.fields["SNOW"].shape
    fields = dict(base.fields)
    fields["SNOW"] = _bare_patches_in_snow(shape, 120.0)
    fields["SNOWH"] = _bare_patches_in_snow(shape, 0.6)
    mapped = _map(replace(base, fields=fields))

    # The parabola itself, from the same plan, for comparison.
    from gpuwm.ingest.hrrr import _ProjectedCpuPlan

    lat, lon = _nest_grid().latlon_mass()
    plan = _ProjectedCpuPlan(base, lat, lon, _HostBackend())
    for name, depth in (("SNOW", 120.0), ("SNOWH", 0.6)):
        parabola = plan.apply(fields[name], method="parabolic")
        got = np.asarray(mapped.fields[name])
        below = parabola < 0.0
        assert below.any(), "the fixture must reach the undershoot"
        assert float(parabola.min()) < -0.25 * depth
        assert float(parabola.min()) >= -9.0 / 32.0 * depth * 1.0001
        np.testing.assert_array_equal(got[below], 0.0)
        np.testing.assert_array_equal(got[~below], parabola[~below])
    said = capsys.readouterr().err
    assert "HRRR SNOW on domain 2" in said
    assert "put at 0" in said


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_a_negative_source_is_left_for_the_snow_admission():
    """Not the operator's undershoot: the admission judges it."""
    base = _window_snapshot()
    shape = base.fields["SNOW"].shape
    fields = dict(base.fields)
    snow = _bare_patches_in_snow(shape, 120.0)
    snow[0, 0] = -9999.0
    fields["SNOW"] = snow
    mapped = _map(replace(base, fields=fields))
    assert float(np.min(mapped.fields["SNOW"])) < 0.0
