"""A valid full prior must not become an invalid ensemble after spatial cropping."""
import numpy as np
import pytest

from gpuwm.da.letkf import GriddedObs, GridGeometry, LetkfConfig, LetkfError, Localization
from gpuwm.da.static_covariance import static_analysis
from gpuwm.da.velocity_dispersion import DispersionGate, withhold


def case():
    shape = (2, 9, 9)
    samples = np.arange(5, dtype=np.float64)
    prior = dict(thp=np.broadcast_to(samples[:, None, None, None], (5, *shape)).copy(),
                 qc=np.full((5, *shape), 1.8435110648149077e-19))
    # Usable cloud covariance exists outside the gated zone's crop.
    prior['qc'][:, :, 0, 0] = samples[:, None] * .001
    where = np.zeros(shape[1:], bool)
    where[4, 4] = True
    mask = np.broadcast_to(where[None], shape)
    batches = [GriddedObs(name=name, values=np.full(shape, observed), errors=1.,
                         simulated=prior['thp'], mask=mask)
               for name, observed in (('z', 7.), ('temperature', 3.))]
    loc = Localization(horizontal_m=4000., vertical_m=1000.)
    geometry = GridGeometry(dx_m=3000., dy_m=3000., heights_m=np.array([0., 100.]))
    cfg = LetkfConfig(localization=loc, rtps_alpha=0., analysis_fields=('thp', 'qc'))
    def solve(values, observations, fields, grid):
        from dataclasses import replace
        return static_analysis(values, observations, grid, replace(cfg, analysis_fields=fields))
    return prior, batches, where, geometry, loc, solve


def test_withheld_crop_preserves_globally_usable_cloud_covariance():
    prior, batches, where, geometry, loc, solve = case()
    original_words = {name: value.tobytes() for name, value in prior.items()}
    joint = solve(prior, batches, tuple(prior), geometry)
    reference = solve(prior, batches[1:], tuple(prior), geometry)
    result, receipt = withhold(solve, prior, batches, joint,
        [DispersionGate('z', tuple(prior), where)], tuple(prior),
        geometry=geometry, localization=loc)
    for name in prior:
        expected = joint[name].copy()
        expected[..., where] = reference[name][..., where]
        assert result[name].tobytes() == expected.tobytes()
        assert prior[name].tobytes() == original_words[name]
    assert np.any(result['thp'][0, ..., where] != joint['thp'][0, ..., where])
    assert receipt['solves'][0]['box'] == [0, 8, 0, 8]


def test_globally_unsupported_prior_still_refuses():
    prior, batches, where, geometry, loc, solve = case()
    prior['qc'][:] = 1.8435110648149077e-19
    joint = {name: np.zeros_like(value) for name, value in prior.items()}
    with pytest.raises(LetkfError, match='no usable ensemble spread anywhere'):
        withhold(solve, prior, batches, joint,
            [DispersionGate('z', tuple(prior), where)], tuple(prior),
            geometry=geometry, localization=loc)


def test_unrelated_solver_failure_is_not_retried():
    prior, batches, where, geometry, loc, _ = case()
    calls = []
    def fail(*args):
        calls.append(args)
        raise RuntimeError('unrelated solver failure')
    with pytest.raises(RuntimeError, match='unrelated solver failure'):
        withhold(fail, prior, batches,
            {name: np.zeros_like(value) for name, value in prior.items()},
            [DispersionGate('z', tuple(prior), where)], tuple(prior),
            geometry=geometry, localization=loc)
    assert len(calls) == 1
