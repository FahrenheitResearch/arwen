"""H_CWP(x): the cloud water path operator, against known condensate.

The grid here is chosen so the integral is arithmetic anyone can redo:
``c1h = 1``, ``c2h = 0`` makes the layer mass ``mu * (-dnw)``, and four
layers of ``dnw = -0.25`` make ``sum_k (-dnw) = 1`` exactly.  So a uniform
mixing ratio ``q`` over a column of dry mass ``mu`` integrates to
``1000 * q * mu / G`` g m-2 and nothing else.

The second test is the one that matters more: the operator is checked
against a direct transcription of ``gpuwm/verify/cases/moist_bubble.py``'s
``_water_mass``, which is the model's own column water integral and the
authority this operator claims to be using.  If those two ever disagree,
the operator has stopped being the model's own measure.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import constants as c
from gpuwm.da.obsop_cwp import (CwpComposition, CwpOperatorError,
                                checkpoint_cwp_provider, cloud_water_path,
                                column_mass_per_area,
                                simulated_cloud_water_path)
from gpuwm.obs.goes_cwp import CLASS_CLEAR, CLASS_ICE, CLASS_LIQUID, CLASS_NONE

NZ, NY, NX = 4, 3, 3
MU = 100000.0
Q = 1.0e-5

#: 1000 * Q * MU * sum(-dnw) / G, with sum(-dnw) == 1.
ONE_SPECIES = 1000.0 * Q * MU / c.G

CFG = SimpleNamespace(mp_physics=10)


def _coords():
    return {
        "c1h": np.ones(NZ),
        "c2h": np.zeros(NZ),
        "dnw": np.full(NZ, -0.25),
        "mu": np.full((NY, NX), MU),
    }


def _state(**species):
    fields = {name: np.full((NZ, NY, NX), value)
              for name, value in species.items()}
    return SimpleNamespace(**fields)


def test_a_known_column_integrates_to_the_hand_value():
    coords = _coords()
    state = _state(qc=Q, qi=0.0, qs=0.0)
    out = simulated_cloud_water_path(state, CFG, **coords)
    assert out.shape == (NY, NX)
    assert np.allclose(out, ONE_SPECIES)
    # 1000 * 1e-5 * 1e5 / 9.81
    assert out[0, 0] == pytest.approx(1000.0 * 1.0e-5 * 1.0e5 / 9.81)


def test_the_operator_uses_the_models_own_column_measure():
    """Against a transcription of moist_bubble._water_mass, per column.

    That function is this project's only pre-existing column water
    integral and the authority the operator's docstring names.  This is
    the test that would fail if the operator quietly drifted onto a
    rho*dz form or a different gravity.
    """

    rng = np.random.default_rng(20260806)
    c1h = rng.uniform(0.5, 1.0, NZ)
    c2h = rng.uniform(0.0, 0.2, NZ)
    dnw = -np.abs(rng.uniform(0.1, 0.4, NZ))
    mu = rng.uniform(8.0e4, 1.1e5, (NY, NX))
    qc = rng.uniform(0.0, 1.0e-4, (NZ, NY, NX))
    qi = rng.uniform(0.0, 1.0e-4, (NZ, NY, NX))
    qs = rng.uniform(0.0, 1.0e-4, (NZ, NY, NX))

    # moist_bubble.py:125-134, with the domain sum dropped so it is
    # per-column: sum_k q_tot * (c1h*mu + c2h) * (-dnw) / g.
    qtot = qc + qi + qs
    chm = c1h[:, None, None] * mu[None] + c2h[:, None, None]
    reference = np.sum(qtot * chm * (-dnw[:, None, None]), axis=0) / c.G

    out = cloud_water_path([qc, qi, qs], c1h=c1h, c2h=c2h, dnw=dnw, mu=mu)
    # The operator reports g m-2 where _water_mass reports kg m-2.
    assert np.allclose(out, reference * 1000.0, rtol=1e-12)


def test_layer_mass_refuses_a_sign_flipped_coordinate():
    """Eta decreases upward; the (-dnw) is the sign trap in one place."""

    with pytest.raises(CwpOperatorError, match="non-negative entry"):
        column_mass_per_area(np.ones(NZ), np.zeros(NZ), np.full(NZ, 0.25),
                             np.full((NY, NX), MU))


def test_composition_selects_species_by_the_observations_phase():
    coords = _coords()
    state = _state(qc=Q, qi=2.0 * Q, qs=3.0 * Q)
    classes = np.array([[CLASS_LIQUID, CLASS_ICE, CLASS_CLEAR],
                        [CLASS_NONE, CLASS_LIQUID, CLASS_ICE],
                        [CLASS_CLEAR, CLASS_NONE, CLASS_LIQUID]])
    out = simulated_cloud_water_path(state, CFG, obs_class=classes, **coords)

    liquid = ONE_SPECIES                 # qc
    icy = 6.0 * ONE_SPECIES              # qc + qi + qs
    assert out[0, 0] == pytest.approx(liquid)
    assert out[0, 1] == pytest.approx(icy)
    assert out[0, 2] == pytest.approx(icy)      # clear takes the ice set
    # An unobserved column is still integrated, under the clear
    # composition, so the returned field is complete rather than holed.
    assert out[1, 0] == pytest.approx(icy)


def test_a_clear_observation_can_see_ice_the_model_invented():
    """The suppression direction, at the operator level.

    A model column holding only ice must produce a non-zero H(x) under a
    CLEAR observation, or the zero has nothing to push against and the
    invented cloud is invisible to the one observation most confident it
    should not be there.
    """

    coords = _coords()
    state = _state(qc=0.0, qi=Q, qs=0.0)
    classes = np.full((NY, NX), CLASS_CLEAR)
    out = simulated_cloud_water_path(state, CFG, obs_class=classes, **coords)
    assert np.all(out > 0.0)
    assert out[0, 0] == pytest.approx(ONE_SPECIES)

    # ...and under the spec's liquid composition it would not, which is
    # exactly why the clear branch does not use it.
    liquid_only = simulated_cloud_water_path(
        state, CFG, obs_class=classes,
        composition=CwpComposition(clear=("qc",)), **coords)
    assert np.all(liquid_only == 0.0)


def test_a_composition_the_scheme_cannot_satisfy_is_refused():
    coords = _coords()
    warm_rain = _state(qc=Q)             # no qi, no qs: mp_physics=1
    with pytest.raises(CwpOperatorError, match="biasing H\\(x\\) low"):
        simulated_cloud_water_path(warm_rain, SimpleNamespace(mp_physics=1),
                                   **coords)
    # The escape hatch exists and has to be asked for by name.
    out = simulated_cloud_water_path(
        warm_rain, SimpleNamespace(mp_physics=1),
        allow_missing_species=True, **coords)
    assert out[0, 0] == pytest.approx(ONE_SPECIES)


def test_the_spec_variant_is_one_argument_away():
    coords = _coords()
    state = _state(qc=Q, qi=2.0 * Q, qs=3.0 * Q)
    classes = np.full((NY, NX), CLASS_ICE)
    spec = CwpComposition(ice=("qc", "qi"), clear=("qc", "qi"))
    out = simulated_cloud_water_path(state, CFG, obs_class=classes,
                                     composition=spec, **coords)
    assert out[0, 0] == pytest.approx(3.0 * ONE_SPECIES)   # qc + qi
    assert "matches docs/obs-goes-cwp-operator-spec.md" in \
        spec.to_payload()["diverges_from_spec"]
    assert "Deliberate, recorded" in \
        CwpComposition().to_payload()["diverges_from_spec"]


# ---------------------------------------------------------------------------
# the checkpoint provider
# ---------------------------------------------------------------------------


def test_the_provider_integrates_each_member_over_its_own_column_mass():
    """``mu = mub2d + mup``, per member.

    Using the base state's ``mub2d`` alone would evaluate every member's
    H(x) on the same column mass and erase part of the covariance the
    filter reads.
    """

    provider = checkpoint_cwp_provider(
        CFG, c1h=np.ones(NZ), c2h=np.zeros(NZ), dnw=np.full(NZ, -0.25),
        mub2d=np.full((NY, NX), MU))
    base = {"qc": np.full((NZ, NY, NX), Q),
            "qi": np.zeros((NZ, NY, NX)),
            "qs": np.zeros((NZ, NY, NX))}

    flat = provider(0, dict(base, mup=np.zeros((NY, NX))))
    heavy = provider(1, dict(base, mup=np.full((NY, NX), 0.1 * MU)))
    assert flat[0, 0] == pytest.approx(ONE_SPECIES)
    assert heavy[0, 0] == pytest.approx(1.1 * ONE_SPECIES)


def test_the_provider_refuses_a_checkpoint_without_its_column_mass():
    provider = checkpoint_cwp_provider(
        CFG, c1h=np.ones(NZ), c2h=np.zeros(NZ), dnw=np.full(NZ, -0.25),
        mub2d=np.full((NY, NX), MU))
    with pytest.raises(CwpOperatorError, match="erase the very covariance"):
        provider(0, {"qc": np.full((NZ, NY, NX), Q),
                     "qi": np.zeros((NZ, NY, NX)),
                     "qs": np.zeros((NZ, NY, NX))})


def test_the_provider_refuses_a_scalar_base_column_mass():
    with pytest.raises(CwpOperatorError, match="fail loudly"):
        checkpoint_cwp_provider(CFG, c1h=np.ones(NZ), c2h=np.zeros(NZ),
                                dnw=np.full(NZ, -0.25), mub2d=np.float64(MU))


def test_the_provider_passes_the_observed_phase_through():
    provider = checkpoint_cwp_provider(
        CFG, c1h=np.ones(NZ), c2h=np.zeros(NZ), dnw=np.full(NZ, -0.25),
        mub2d=np.full((NY, NX), MU))
    state = {"qc": np.full((NZ, NY, NX), Q),
             "qi": np.full((NZ, NY, NX), 2.0 * Q),
             "qs": np.full((NZ, NY, NX), 3.0 * Q),
             "mup": np.zeros((NY, NX))}
    liquid = provider(0, state, np.full((NY, NX), CLASS_LIQUID))
    icy = provider(0, state, np.full((NY, NX), CLASS_ICE))
    assert liquid[0, 0] == pytest.approx(ONE_SPECIES)
    assert icy[0, 0] == pytest.approx(6.0 * ONE_SPECIES)
