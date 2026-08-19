"""Row order on regular GDT-0 grids is an octet, not an assumption.

GRIB2 scan mode 0x40 stores rows south-to-north; 0x00 stores them
north-to-south (how ECMWF products and NCEP's GFS/GDAS publish their
global grids).  The generic decode normalizes BOTH into the one canonical
ascending-latitude frame at the record boundary -- the same place the
longitude unwrap already normalizes the x axis -- because a consumer that
trusted 0x00 bytes under 0x40 semantics would interpret the world upside
down.  Every other scan mode (reversed i, column-major, boustrophedon)
still refuses by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.mapped_source import _regular_latlon_frame


def _row(scan_mode: str, *, ny: int = 3, nx: int = 4) -> dict[str, str]:
    return {
        "gdt": "0",
        "scan_mode": scan_mode,
        "nx": str(nx),
        "ny": str(ny),
        "lat1": "90" if scan_mode == "0x00" else "-90",
        "lon1": "0",
        "dx": "1.0",
        "dy": "1.0",
    }


def test_south_to_north_rows_pass_through_unchanged():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    latitude, longitude, oriented = _regular_latlon_frame(
        _row("0x40"), values.ravel().copy())
    assert latitude.tolist() == [-90.0, -89.0, -88.0]
    np.testing.assert_array_equal(oriented, values)
    assert longitude.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_north_to_south_rows_are_normalized_to_ascending_latitude():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    latitude, longitude, oriented = _regular_latlon_frame(
        _row("0x00"), values.ravel().copy())
    # lat1=90 descending by dy becomes the ascending canonical axis...
    assert latitude.tolist() == [88.0, 89.0, 90.0]
    # ...and the rows flip with it, so every (latitude, value) pair is
    # the pair the producer stored.
    np.testing.assert_array_equal(oriented, values[::-1, :])
    assert longitude.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_the_two_orientations_agree_on_every_latitude_value_pair():
    north_south = np.arange(12, dtype=np.float64).reshape(3, 4)
    lat_ns, _, val_ns = _regular_latlon_frame(
        _row("0x00"), north_south.ravel().copy())
    lat_sn, _, val_sn = _regular_latlon_frame(
        dict(_row("0x40"), lat1="88"), north_south[::-1, :].ravel().copy())
    assert lat_ns.tolist() == lat_sn.tolist()
    np.testing.assert_array_equal(val_ns, val_sn)


@pytest.mark.parametrize("scan_mode", ["0x80", "0xc0", "0x20", "0x10"])
def test_every_other_scan_mode_refuses_by_name(scan_mode):
    values = np.zeros(12, dtype=np.float64)
    with pytest.raises(ValueError, match="scan mode"):
        _regular_latlon_frame(_row(scan_mode), values)


def test_a_projected_template_still_refuses_without_a_declaration():
    values = np.zeros(12, dtype=np.float64)
    with pytest.raises(ValueError, match="GDT 0"):
        _regular_latlon_frame(dict(_row("0x40"), gdt="30"), values)
