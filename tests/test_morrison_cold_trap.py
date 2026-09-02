"""Morrison deposition-freezing nucleation in the polar-night cold trap.

CPU-only mirror tests (no cupy anywhere in this module, so the whole file
runs under GPUWM_NO_LOCAL_GPU=1; the kernel-vs-mirror parity for the same
arithmetic lives in ``test_morrison.py`` on the rented device).

The conviction these pin: hour 49 of a 384 h T255 native-suite run
(2026-09-01) aborted on 'native physics produced negative qv: -1.6e-4'
with the polar-night model top at 159.8 K and falling.  At T <= ~159 K
POLYSVP's extrapolated liquid curve drops under its ice curve,
``EIS = min(EW, ...)`` makes ``qvi == qvs`` exactly, and the Cooper
trigger ``qv/qvs >= 0.999 && T <= 265.15`` fires with qv AT OR BELOW ice
saturation -- a band empty at every warmer temperature because
``qvs/qvi >= 1.03`` there.  The FUDGEF supersaturation rescale tests only
the ``dum > 0 / sum_dep > dum`` and ``dum < 0 / sum_dep < dum`` sign
pairs, so a positive MNUCCD over a subsaturated state was never rescaled
and ``(500e3/rhoa - moments) * MI0`` of embryo mass -- dt-independent --
was withdrawn from ~1e-7 kg/kg of vapor.  The fix bounds the nucleated
mass by the vapor excess over ice saturation at the source
(morrison.cu MNUCCD block; gpuwm.verify.npref mirrors it), scaling the
number moment with it.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core import constants as c

MASS_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")


def _water_mass(column, rho):
    return float(np.sum(sum(np.asarray(column[n], np.float64)
                            for n in MASS_NAMES)
                        * np.asarray(rho, np.float64)
                        * np.asarray(column["dz"], np.float64)))


def _saturations(p, temp):
    from gpuwm.verify.npref import _np_morrison_polysvp

    ew = np.array([min(0.99 * pk, float(_np_morrison_polysvp(tk, False)))
                   for pk, tk in zip(p, temp)])
    ei = np.array([min(ewk, 0.99 * pk,
                       float(_np_morrison_polysvp(tk, True)))
                   for ewk, pk, tk in zip(ew, p, temp)])
    return (c.EP2 * ew / (p - ew), c.EP2 * ei / (p - ei))


def _vapor_only_column(p, temp, qv):
    nz = p.size
    pii = (p / c.P0) ** c.RCP
    zeros = np.zeros(nz)
    return {
        "theta": temp / pii, "qv": np.asarray(qv, np.float64).copy(),
        "qc": zeros.copy(), "qr": zeros.copy(), "qi": zeros.copy(),
        "qs": zeros.copy(), "qg": zeros.copy(),
        "nc": zeros.copy(), "nr": zeros.copy(), "ni": zeros.copy(),
        "ns": zeros.copy(), "ng": zeros.copy(),
        "rho": p / (c.RD * temp), "pii": pii, "pressure": p.copy(),
        "dz": np.full(nz, 3000.0),
    }


def test_cold_trap_deposition_nucleation_cannot_overdraw_vapor():
    """MNUCCD may not consume vapor that does not exist.

    T = 156 K, qv parked at 0.9995 of ice saturation -- the state the
    stratosphere-free native top cools into, re-parked there by FUDGEF
    every call.  Pre-fix the mirror returned qv ~ -1.6e-4 kg/kg here (the
    conviction magnitude); bounded at the source it must come back with qv
    untouched and nothing nucleated: the number moment scales with the
    withheld mass, so no crystal count exists whose seed mass never did.
    """
    from gpuwm.verify.npref import np_morrison_column

    p = np.array([300.0, 250.0])
    temp = np.full(2, 156.0)
    qvs, qvi = _saturations(p, temp)
    np.testing.assert_array_equal(qvs, qvi)   # the trap's arming condition
    src = _vapor_only_column(p, temp, 0.9995 * qvi)
    rho = src["pressure"] / (c.RD * src["theta"] * src["pii"])
    initial = _water_mass(src, rho)

    out = np_morrison_column(**{k: v.copy() for k, v in src.items()},
                             dt=50.0)

    assert float(out["qv"].min()) >= 0.0      # pre-fix: ~ -1.6e-4
    np.testing.assert_allclose(out["qv"], src["qv"], rtol=1e-12)
    # Conservation-neutral: the un-consumed vapor also deposits nothing.
    assert float(np.abs(out["qi"]).max()) == 0.0
    assert float(np.abs(out["ni"]).max()) == 0.0
    assert out["precip_step"] == 0.0
    final = _water_mass(out, rho)
    assert abs(final - initial) <= 64.0 * np.finfo(np.float64).eps \
        * max(initial, 1.0)


def test_supersaturated_deposition_nucleation_keeps_wrf_magnitude():
    """The cold-trap bound stays silent where vapor covers the embryo mass.

    A 240 K column at 10% ice supersaturation carries a vapor excess over
    ice saturation that covers the Cooper embryo mass many times over, so
    the source bound stays silent and full deposition nucleation proceeds
    (in colder states whose excess is under the embryo mass the bound
    engages and trims the number moment only; the FUDGEF dum > 0 branch
    caps the mass identically in WRF, see the morrison.cu MNUCCD note):
    every level's vapor draw is exactly ``target * MI0`` and the nucleated
    number is the Cooper target.  The vapor comparison is per level and
    sedimentation-immune (sedimentation never writes qv); the number
    comparison tolerates the ~5e-4 of the 11 um crystals that sediment
    out of a level during the step.
    """
    from gpuwm.verify.npref import _MORR_MI0, np_morrison_column

    p = np.array([30000.0, 25000.0])
    temp = np.full(2, 240.0)
    _, qvi = _saturations(p, temp)
    src = _vapor_only_column(p, temp, 1.10 * qvi)
    rhoa = p / (c.RD * temp)
    target = np.minimum(0.005 * np.exp(0.304 * (273.15 - temp)) * 1000.0,
                        500.0e3) / rhoa

    out = np_morrison_column(**{k: v.copy() for k, v in src.items()},
                             dt=50.0)

    np.testing.assert_allclose(src["qv"] - out["qv"], target * _MORR_MI0,
                               rtol=1e-9)
    np.testing.assert_allclose(out["ni"], target, rtol=1e-2)
    assert float(out["qi"].min()) > 0.0
