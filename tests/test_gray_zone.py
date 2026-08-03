"""The published gray-zone partition curve: transcription and behaviour.

Pins the values a typo in the coefficients would move, and the two
endpoint facts most likely to be assumed wrong.
"""
from __future__ import annotations

import numpy as np

from gpuwm.verify import gray_zone as GZ


def test_the_published_curve_is_transcribed_correctly():
    """Honnert et al. (2011) eq. (9), evaluated at the points the paper
    and its figures annotate.

    The 50% crossover of the mixed-layer TKE fit sits at 0.27 (the
    paper's Fig. 5b annotates 0.3), the entrainment-zone fit at 0.45
    (Fig. 5a annotates 0.4), and the curve saturates to 1 by ten
    boundary-layer depths.
    """
    np.testing.assert_allclose(GZ.crossover(), 0.2695, rtol=2e-3)
    np.testing.assert_allclose(GZ.crossover(True), 0.4456, rtol=2e-3)
    for x, want in ((0.013, 0.0508), (0.05, 0.1286), (0.1, 0.2235),
                    (0.2, 0.3989), (0.5, 0.7148), (1.0, 0.8812),
                    (2.0, 0.9565), (10.0, 0.9959)):
        np.testing.assert_allclose(
            float(GZ.subgrid_tke_fraction(x)), want, rtol=1e-3)


def test_the_curve_decays_slowly_which_is_the_easy_thing_to_get_wrong():
    """The Kolmogorov tail, and why "LES limit" is finer than expected.

    The x^(2/3) term is derived from the inertial-range spectrum, not
    fitted, so the subgrid fraction decays as a POWER and not as a
    tanh.  At one tenth of the boundary-layer depth a model still owes
    22% of the energy to its closure -- an assumption that it owes
    almost none there is wrong by a factor of ten, and would make a
    scale-aware scheme look correct while it discarded a fifth of the
    turbulence.
    """
    assert GZ.subgrid_tke_fraction(0.1) > 0.2
    # five per cent subgrid is not reached until ~0.013
    assert GZ.subgrid_tke_fraction(0.02) > 0.05
    assert GZ.subgrid_tke_fraction(0.01) < 0.05
    # ... and at Delta = h the flow is already overwhelmingly subgrid
    assert GZ.subgrid_tke_fraction(1.0) > 0.85
    # monotone increasing over the whole plotted range
    x = np.logspace(-2, 1.5, 400)
    p = GZ.subgrid_tke_fraction(x)
    assert np.all(np.diff(p) > 0.0)
    assert p[0] > 0.0 and p[-1] < 1.0


def test_the_envelope_brackets_the_curve_and_stays_a_probability():
    lo, hi = GZ.subgrid_tke_envelope(np.logspace(-2, 1.5, 200))
    mid = GZ.subgrid_tke_fraction(np.logspace(-2, 1.5, 200))
    assert np.all(lo <= mid) and np.all(mid <= hi)
    assert np.all(lo >= 0.0) and np.all(hi <= 1.0)


def test_the_partition_measurement_is_a_plain_variance():
    """No window, no detrend, no spectrum -- so nothing to correct.

    A uniform field has zero resolved energy (fraction 1); a field whose
    resolved variance equals the subgrid energy splits exactly in half.
    Both are checked because the second is the one an off-by-a-factor-
    of-two in the 0.5 prefactor would break silently.
    """
    shape = (4, 8, 8)
    e = np.full(shape, 0.25)
    zero = np.zeros(shape)
    out = GZ.partition_from_profiles(e, zero, zero, zero)
    np.testing.assert_allclose(out["subgrid_fraction"], 1.0)
    # u with plane variance 1.0 (values +/-1) gives e_res = 0.5*1.0
    u = np.where(np.indices(shape)[2] % 2 == 0, 1.0, -1.0).astype(float)
    out = GZ.partition_from_profiles(np.full(shape, 0.5), u, zero, zero)
    np.testing.assert_allclose(out["e_resolved"], 0.5)
    np.testing.assert_allclose(out["subgrid_fraction"], 0.5)
