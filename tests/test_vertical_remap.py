"""Conservation contracts for the offline child's vertical remap.

Every claim here carries a negative control that must FAIL when the operator
is broken.  A conservation test whose control also passes cannot tell a
conservative operator from one that returned its input unchanged.
"""

import numpy as np
import pytest

from gpuwm.core import constants as c
from gpuwm.vertical_remap import (
    ColumnBasis,
    VerticalRemapRefusal,
    column_integral,
    dry_mass_edges,
    geopotential_thickness_per_mass,
    layer_masses,
    rebuild_geopotential,
    remap_interface_values,
    remap_layer_means,
    remap_receipt,
    require_shared_column_basis,
)

P_TOP = 5000.0


def _ladder(nz, stretch=None):
    """The same ladder shapes ``make_vertical_coord`` builds."""
    if stretch is None:
        return np.linspace(1.0, 0.0, nz + 1)
    zeta = np.linspace(0.0, 1.0, nz + 1)
    znw = np.tanh(stretch * (1.0 - zeta)) / np.tanh(stretch)
    znw[0] = 1.0
    znw[-1] = 0.0
    return znw


def _edges(znw, mu, *, hybrid_opt=2, etac=0.2, p_top=P_TOP):
    return dry_mass_edges(znw, hybrid_opt=hybrid_opt, etac=etac,
                          p_top=p_top, mu=mu)


def _tracer(znw):
    """A smooth exponential plus a sharp Gaussian cloud layer."""
    znu = 0.5 * (znw[:-1] + znw[1:])
    return (1.0e-2 * np.exp(-3.0 * (1.0 - znu))
            + 3.0e-3 * np.exp(-((znu - 0.55) ** 2) / (2 * 0.02 ** 2)))


# --------------------------------------------------------------------------
# 1.  The column is the same mass interval on every ladder.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hybrid_opt", [0, 2])
@pytest.mark.parametrize("mu", [70000.0, 85000.0, 92000.0, 98000.0])
def test_dry_mass_column_is_ladder_independent(hybrid_opt, mu):
    """Every ladder partitions exactly ``[0, mu]``: same endpoints, same total.

    This is the premise the whole remap rests on -- with it, rebinning needs
    no extrapolation at either end.
    """
    for nz, stretch in ((49, None), (49, 1.8), (128, 2.5), (160, 3.0)):
        edges = _edges(_ladder(nz, stretch), mu, hybrid_opt=hybrid_opt)
        assert edges[0] == mu, "surface interface must carry the whole column"
        assert edges[-1] == 0.0, "model top must carry zero dry mass"
        dm = layer_masses(edges)
        assert abs(dm.sum() - mu) < 1e-9, (nz, stretch, dm.sum() - mu)
        assert dm.min() > 0.0


def test_layer_mass_matches_the_trees_own_coupling_weight():
    """``dm`` must be the ``-dnw*(c1h*mu + c2h)`` the tree already uses.

    ``_couple_parent`` forms exactly this as ``chm``; if this module invented
    a second weight the remap would conserve something the dycore does not
    integrate.
    """
    from gpuwm.core.grid import compute_hybrid_coeffs

    mu = 92000.0
    znw = _ladder(49, 1.8)
    hy = compute_hybrid_coeffs(znw, 2, 0.2, float(c.P0), P_TOP)
    expected = -np.diff(znw) * (hy["c1h"] * mu + hy["c2h"])
    assert np.abs(layer_masses(_edges(znw, mu)) - expected).max() < 1e-8


# --------------------------------------------------------------------------
# 2.  Rebinning conserves the column integral.
# --------------------------------------------------------------------------

def test_rebinning_conserves_the_column_integral_to_roundoff():
    """49 -> 128 -> 49 must not move the column's water content."""
    mu = 92000.0
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, ed = _edges(src, mu), _edges(dst, mu)
    q = _tracer(src)

    up = remap_layer_means(es, q, ed)
    back = remap_layer_means(ed, up, es)

    i0 = float(column_integral(es, q))
    assert abs(float(column_integral(ed, up)) - i0) / i0 < 1e-13
    assert abs(float(column_integral(es, back)) - i0) / i0 < 1e-13
    assert up.min() >= 0.0 and back.min() >= 0.0


def test_a_truncated_target_column_breaks_conservation():
    """NEGATIVE CONTROL for the test above.

    Stop the target ladder one bin short of the model top and the drift must
    jump to O(1e-3).  Without this, an operator that returned its input
    unchanged would read as conserving.
    """
    mu = 92000.0
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, ed = _edges(src, mu), _edges(dst, mu)
    q = _tracer(src)

    short = ed[:-1]
    drift = abs(float(column_integral(short, remap_layer_means(es, q, short)))
                - float(column_integral(es, q))) / float(column_integral(es, q))
    assert drift > 1e-4, (
        f"the truncation control drifted only {drift:.3e}; the conservation "
        "assertion above cannot distinguish a working operator from a broken "
        "one")


def test_three_dimensional_columns_conserve_independently():
    """Each column carries its own ``mu``, hence its own edges and weights."""
    rng = np.random.default_rng(0)
    mu = 92000.0 + 3000.0 * rng.standard_normal((3, 4))
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, ed = _edges(src, mu), _edges(dst, mu)
    q = np.broadcast_to(_tracer(src)[:, None, None], (49, 3, 4)).copy()

    up = remap_layer_means(es, q, ed)
    assert up.shape == (128, 3, 4)
    i0, i1 = column_integral(es, q), column_integral(ed, up)
    assert np.abs((i1 - i0) / i0).max() < 1e-13


# --------------------------------------------------------------------------
# 3.  The operator's own negative control: identity when the ladders match.
# --------------------------------------------------------------------------

def test_remap_is_the_exact_identity_when_ladders_match():
    """Matched ladders must give back the FP64 bits, not merely close values.

    This is the operator's self-test.  If it is not the exact identity when
    nothing should change, every other number it produces is suspect.
    """
    mu = 92000.0
    znw = _ladder(49, 1.8)
    edges = _edges(znw, mu)
    q = _tracer(znw)
    assert np.array_equal(remap_layer_means(edges, q, edges), q)


def test_a_perturbed_source_edge_breaks_the_identity():
    """NEGATIVE CONTROL: a test that only ever sees identical inputs is vacuous."""
    mu = 92000.0
    znw = _ladder(49, 1.8)
    edges = _edges(znw, mu)
    moved = edges.copy()
    moved[20] += 1.0
    assert not np.array_equal(remap_layer_means(moved, _tracer(znw), edges),
                              _tracer(znw))


def test_interface_remap_is_the_identity_when_ladders_match():
    mu = 92000.0
    znw = _ladder(49, 1.8)
    edges = _edges(znw, mu)
    w = np.sin(np.linspace(0.0, 3.0, znw.size))
    assert np.array_equal(remap_interface_values(edges, w, edges), w)


# --------------------------------------------------------------------------
# 4.  A folded coordinate is refused, by name, instead of returning garbage.
# --------------------------------------------------------------------------

def test_a_folded_hybrid_column_is_refused_by_name():
    """etac=0.5 over terrain that folds the coordinate must refuse.

    At ``hybrid_opt=2`` the Klemp cubic's ``dB/deta`` exceeds 1 near the
    ground, so a column with small ``mu`` gets ``c1h*mu + c2h <= 0`` and the
    reference dry pressure stops decreasing.  ``alt = dphi/dm`` would then
    divide by a non-positive mass and carry an inf into the child.
    """
    with pytest.raises(VerticalRemapRefusal) as excinfo:
        _edges(_ladder(49, 1.8), 60000.0, etac=0.5)
    message = str(excinfo.value)
    assert "compute_vcoord_1d_coeffs" in message
    assert "-39" in message or "-39.0" in message, message


def test_a_healthy_column_at_the_same_etac_is_admitted():
    """NEGATIVE CONTROL: the refusal above must not be a ban on hybrid_opt=2.

    Same etac, slightly lower terrain (larger mu) -- min(dm) = +287.81 Pa.
    """
    dm = layer_masses(_edges(_ladder(49, 1.8), 70000.0, etac=0.5))
    assert dm.min() > 0.0
    assert 287.0 < dm.min() < 288.0


def test_a_folded_column_never_returns_a_nan_or_an_inf():
    """The refusal has to arrive instead of a number, not alongside one."""
    try:
        edges = _edges(_ladder(49, 1.8), 60000.0, etac=0.5)
    except VerticalRemapRefusal:
        return
    pytest.fail(f"folded column returned {edges!r} instead of refusing")


# --------------------------------------------------------------------------
# 5.  The two ladders must be the same column.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("target,expected", [
    (ColumnBasis(p_top=1000.0, hybrid_opt=2, etac=0.2), "p_top"),
    (ColumnBasis(p_top=P_TOP, hybrid_opt=0, etac=0.2), "hybrid_opt"),
    (ColumnBasis(p_top=P_TOP, hybrid_opt=2, etac=0.4), "etac"),
])
def test_a_mismatched_column_basis_is_refused(target, expected):
    """Refused, not silently rescaled: the endpoints would not coincide."""
    source = ColumnBasis(p_top=P_TOP, hybrid_opt=2, etac=0.2)
    with pytest.raises(VerticalRemapRefusal) as excinfo:
        require_shared_column_basis(source, target, context="child ladder")
    assert expected in str(excinfo.value)


def test_a_matched_column_basis_is_admitted():
    """NEGATIVE CONTROL: a check that refuses everything is not a check."""
    basis = ColumnBasis(p_top=P_TOP, hybrid_opt=2, etac=0.2)
    require_shared_column_basis(basis, basis, context="child ladder")


# --------------------------------------------------------------------------
# 6.  Geopotential: remap alt, rebuild PHI.
# --------------------------------------------------------------------------

def _parent_geopotential(znw, mu, hgt):
    edges = _edges(znw, mu)
    pd_h = 0.5 * (edges[:-1] + edges[1:]) + P_TOP
    thv = 300.0 * (1.0 + 0.15 * (1.0 - 0.5 * (znw[:-1] + znw[1:])))
    alt = c.RD * thv * (pd_h / c.P0) ** c.RCP / pd_h
    return edges, alt, rebuild_geopotential(c.G * hgt, alt, edges)


def test_geopotential_depth_is_preserved_and_stays_dycore_consistent():
    """Remap ``alt = dphi/dm`` and rebuild; do not interpolate PHI itself.

    Two things are asserted: the child's model-top height matches the
    parent's, and the child's own ``update_diagnostics`` recovers exactly the
    ``alt`` that was remapped -- the state is consistent with the dycore's
    discrete hydrostatic relation, not merely near it.
    """
    mu, hgt = 92000.0, 350.0
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, alt, phi_src = _parent_geopotential(src, mu, hgt)
    ed = _edges(dst, mu)

    alt_child = remap_layer_means(es, alt, ed)
    phi_child = rebuild_geopotential(c.G * hgt, alt_child, ed)

    assert phi_child[0] == c.G * hgt
    assert abs(phi_child[-1] - phi_src[-1]) < 1e-6
    recovered = geopotential_thickness_per_mass(phi_child, ed)
    assert np.abs(recovered - alt_child).max() < 1e-12


def test_interpolating_geopotential_in_eta_breaks_dycore_consistency():
    """NEGATIVE CONTROL for the test above.

    Interpolating PHI linearly in ETA also preserves the column depth exactly
    (both ladders share the endpoints), so depth alone cannot discriminate.
    What separates the two is dycore consistency: the eta-interpolated state's
    diagnosed ``alt`` is O(1e-3) m3/kg away from its own thermodynamics.
    """
    mu, hgt = 92000.0, 350.0
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, alt, phi_src = _parent_geopotential(src, mu, hgt)
    ed = _edges(dst, mu)
    alt_child = remap_layer_means(es, alt, ed)

    phi_eta = np.interp(dst[::-1], src[::-1], phi_src[::-1])[::-1]
    recovered = geopotential_thickness_per_mass(phi_eta, ed)
    assert np.abs(recovered - alt_child).max() > 1e-4, (
        "eta-interpolated geopotential scored as well as the reconstruction; "
        "the dycore-consistency metric does not discriminate")


def test_a_non_conservative_alt_moves_the_model_top():
    """NEGATIVE CONTROL: nearest-source-layer ``alt`` must move the top height."""
    mu, hgt = 92000.0, 350.0
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, alt, phi_src = _parent_geopotential(src, mu, hgt)
    ed = _edges(dst, mu)

    centres = 0.5 * (ed[:-1] + ed[1:])
    nearest = alt[np.clip(np.searchsorted(-es, -centres) - 1, 0, alt.size - 1)]
    phi_nn = rebuild_geopotential(c.G * hgt, nearest, ed)
    assert abs(phi_nn[-1] - phi_src[-1]) > 1.0


# --------------------------------------------------------------------------
# 7.  Positivity.
# --------------------------------------------------------------------------

def test_positivity_is_preserved_on_a_single_layer_spike():
    mu = 92000.0
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, ed = _edges(src, mu), _edges(dst, mu)
    spike = np.zeros(49)
    spike[20] = 1.0

    out = remap_layer_means(es, spike, ed)
    assert out.min() >= 0.0
    i0 = float(column_integral(es, spike))
    assert abs(float(column_integral(ed, out)) - i0) / i0 < 1e-13


def test_a_cubic_interpolant_undershoots_the_same_spike():
    """NEGATIVE CONTROL: a positivity test that never sees a negative is vacuous."""
    scipy_interpolate = pytest.importorskip("scipy.interpolate")
    mu = 92000.0
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, ed = _edges(src, mu), _edges(dst, mu)
    spike = np.zeros(49)
    spike[20] = 1.0

    spline = scipy_interpolate.CubicSpline(
        (0.5 * (es[:-1] + es[1:]))[::-1], spike[::-1])
    assert spline(0.5 * (ed[:-1] + ed[1:])).min() < -1e-3


# --------------------------------------------------------------------------
# 8.  The receipt reports what happened.
# --------------------------------------------------------------------------

def test_the_receipt_reports_the_conservation_it_achieved():
    """A remap that quietly failed to conserve must not look like one that did."""
    mu = 92000.0
    src, dst = _ladder(49, 1.8), _ladder(128, 2.5)
    es, ed = _edges(src, mu), _edges(dst, mu)
    q = _tracer(src)

    receipt = remap_receipt("qvapor", es, q, ed, remap_layer_means(es, q, ed))
    assert receipt.source_levels == 49 and receipt.target_levels == 128
    assert receipt.max_relative_drift < 1e-13
    assert receipt.min_target_layer_mass_pa > 0.0
    assert receipt.minimum_value >= 0.0
    assert "qvapor" in receipt.summary()
