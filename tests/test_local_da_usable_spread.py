"""A tiny varying cloud field must not abort an otherwise usable analysis."""
import numpy as np
import pytest

from gpuwm.local_da import PlanError
from gpuwm.local_da_runtime import _analysis_fields_for_states
from gpuwm.da.letkf import GridGeometry, LetkfConfig, Localization, analyze
from gpuwm.da.radar_assimilation import _mass_field


def states(values, dtype=np.float32):
    return [dict(qc=np.full((2, 2, 2), q, dtype=dtype),
                 thp=np.full((2, 2, 2), i, dtype=dtype))
            for i, q in enumerate(values)]


def test_varying_cloud_with_no_usable_relative_covariance_is_not_selected():
    # The live D1 failure has qc magnitude 1.8492379980871303e-19.
    q = np.float32(1.8492379980871303e-19)
    tiny = np.float32(1e-40)
    adjacent = np.nextafter(tiny, np.float32(np.inf))
    prior = states([q] * 9)
    # The field maximum is constant; a much smaller cell varies. The
    # filter widens checkpoint words to float64 mass fields, then tests
    # the widest spread relative to the maximum magnitude of the field.
    for i, state in enumerate(prior):
        state['qc'][0, 0, 0] = tiny if i % 2 else adjacent
    values = np.stack([s['qc'] for s in prior[1:]])
    assert np.any(np.ptp(values, axis=0) > 0)
    fields = _analysis_fields_for_states(prior, 6, static=True)
    sample_prior = {name: np.stack([_mass_field(name, s, where='local analysis')
                                   for s in prior[1:]]) for name in fields}
    # Run the real owner, including its support gate. Before the fix this
    # raises the same no-usable-spread refusal as the original D1 door.
    increments = analyze(sample_prior, [],
                         GridGeometry(dx_m=3000., dy_m=3000., heights_m=np.array([0., 100.])),
                         LetkfConfig(localization=Localization(6000., 1000.), rtps_alpha=0.,
                                     analysis_fields=tuple(fields)))
    assert fields == ['thp']
    assert np.all(increments['thp'] == 0)


def test_constant_float32_cloud_is_not_selected_despite_mean_roundoff():
    prior = states([np.float32(.003)] * 10)
    assert _analysis_fields_for_states(prior, 6, static=True) == ['thp']


def test_usable_cloud_covariance_keeps_the_field_and_its_input_words():
    prior = states([.003 + i * .0001 for i in range(9)])
    words = [s['qc'].tobytes() for s in prior]
    assert _analysis_fields_for_states(prior, 6, static=True) == ['thp', 'qc']
    assert [s['qc'].tobytes() for s in prior] == words


def test_static_background_is_not_counted_as_a_covariance_sample():
    prior = states([.1] + [.003] * 8)
    assert _analysis_fields_for_states(prior, 6, static=True) == ['thp']


def test_nonfinite_prior_is_refused_instead_of_omitted():
    prior = states([.003] * 9)
    prior[1]['qc'][0, 0, 0] = np.nan
    with pytest.raises(PlanError, match="nonfinite prior"):
        _analysis_fields_for_states(prior, 6, static=True)
