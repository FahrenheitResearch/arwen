"""make_DropletNumber against a scalar transcription of the Fortran.

WRF v4.7.1 dyn_em/module_initialize_real.F:9119-9158, the function real.exe
uses to give analysed cloud water a droplet number (:4829-4838).  The
transcription below is written statement by statement, one value at a
time, with the REAL/DOUBLE declarations of the Fortran, and the vectorised
port must agree with it bit for bit on stand-in columns covering the no
aerosol land and water branches, both clamps of the aerosol count, and the
diameter interpolation between them.
"""

import numpy as np
import pytest

from gpuwm.core.thompson_entry import (
    DROPLET_G_RATIO, droplet_mean_diameter_m, make_droplet_number,
)

F = np.float32


def _fortran_make_droplet_number(q_cloud, qnwfa, xland):
    """One call of the Fortran function, line for line."""
    q_cloud, qnwfa, xland = F(q_cloud), F(qnwfa), F(xland)
    pi = F(3.1415926536)                                  # :9125
    am_r = F(F(pi * F(1000.0)) / F(6.0))                  # :9126
    g_ratio = (24, 60, 120, 210, 336, 504, 720, 990, 1320, 1716, 2184,
               2730, 3360, 4080, 4896)                    # :9127-9128
    if qnwfa <= F(0.0):                                   # :9135
        if (xland - F(1.5)) > F(0.0):                     # :9137 ocean
            xdc, nu_c = F(17.0e-6), 12
        else:                                             # :9140 land
            xdc, nu_c = F(11.0e-6), 4
    else:
        q_nwfa = max(F(99.0e6), min(qnwfa, F(5.0e10)))    # :9146
        quotient = F(F(2.5e10) / q_nwfa)                  # :9147, REAL
        nearest = int(np.floor(float(quotient) + 0.5))    # NINT, x > 0
        nu_c = max(2, min(nearest, 15))
        # :9149-9150
        x1 = F(max(F(1.0), min(F(q_nwfa * F(1.0e-9)), F(10.0))) - F(1.0))
        xdc = F(F(F(30.0) - F(F(x1 * F(20.0)) / F(9.0))) * F(1.0e-6))
    lam = (4.0 + float(nu_c)) / float(xdc)                # :9153, DOUBLE
    first = float(F(q_cloud / F(g_ratio[nu_c - 1])))      # REAL quotient
    qnc = first * lam * lam * lam / float(am_r)           # :9154, DOUBLE
    return F(qnc)                                         # :9155 SNGL


#: Per-volume cloud water (kg m^-3), from a trace to a thick deck.
_CLOUD = (1.0e-7, 1.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.2e-3)
#: Per-volume aerosol (m^-3): none, below the 99e6 floor, across the
#: 1e9..1e10 diameter ramp, a NINT tie region, and above the 5e10 cap.
_AEROSOL = (0.0, -5.0, 1.0e6, 99.0e6, 3.0e8, 1.0e9, 1.6666666e9, 2.5e9,
            4.0e9, 7.3e9, 1.0e10, 2.0e10, 5.0e10, 9.0e10)


@pytest.mark.parametrize("xland", [1.0, 2.0])
def test_port_matches_the_fortran_bit_for_bit(xland):
    q, a = np.meshgrid(np.asarray(_CLOUD, dtype=np.float32),
                       np.asarray(_AEROSOL, dtype=np.float32),
                       indexing="ij")
    port = make_droplet_number(q, a, np.float32(xland))
    assert port.dtype == np.float32
    reference = np.array(
        [[_fortran_make_droplet_number(qq, aa, xland)
          for aa in _AEROSOL] for qq in _CLOUD], dtype=np.float32)
    np.testing.assert_array_equal(port.view(np.uint32),
                                  reference.view(np.uint32))


def test_table_is_the_gamma_ratio_it_stands_for():
    from math import gamma
    expected = [gamma(n + 4) / gamma(n + 1) for n in range(1, 16)]
    np.testing.assert_array_equal(DROPLET_G_RATIO,
                                  np.asarray(expected, dtype=np.float32))


def test_no_aerosol_drops_are_cloud_sized_over_land_and_water():
    """0.1 g/m3 of cloud water: continental and maritime cloud, not drizzle."""
    rc = np.float32(1.0e-4)
    land = make_droplet_number(rc, np.float32(0.0), np.float32(1.0))
    water = make_droplet_number(rc, np.float32(0.0), np.float32(2.0))
    assert 300.0e6 < float(land) < 400.0e6
    assert 50.0e6 < float(water) < 70.0e6
    assert float(droplet_mean_diameter_m(rc, land)) == pytest.approx(
        8.17e-6, abs=0.05e-6)
    assert float(droplet_mean_diameter_m(rc, water)) == pytest.approx(
        14.85e-6, abs=0.05e-6)


def test_more_aerosol_makes_more_smaller_drops():
    rc = np.float32(2.0e-4)
    aerosol = np.asarray((1.0e9, 3.0e9, 6.0e9, 1.0e10), dtype=np.float32)
    number = make_droplet_number(rc, aerosol, np.float32(1.0))
    assert np.all(np.diff(number.astype(np.float64)) > 0.0)
