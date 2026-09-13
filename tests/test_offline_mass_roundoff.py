"""Mass roundoff uses the donor scale; particle-count floors never apply."""
import numpy as np
import pytest
from gpuwm.ingest.nest_init import clamp_sint_undershoot_mapping
from gpuwm.offline_child import _validate_parent_mass_fields, OfflineChildContractError


def test_mass_roundoff_has_no_absolute_floor_and_preserves_real_failure():
    donor = {'qc': np.array([0., 1.36e-5], dtype=np.float32)}
    value = {'qc': np.array([-6.26e-22, -1e-8, 1.36e-5], dtype=np.float32)}
    report = clamp_sint_undershoot_mapping(value, names=('qc',), floor_scale=0., reference_fields=donor)
    assert value['qc'][0] == 0 and value['qc'][1] < 0
    assert report['qc']['cells'] == 1
    assert report['qc']['tolerance'] < 1e-10
    zero = {'qc': np.zeros(2, dtype=np.float32)}
    value = {'qc': np.array([-1e-20, 0.], dtype=np.float32)}
    assert clamp_sint_undershoot_mapping(value, names=('qc',), floor_scale=0., reference_fields=zero) == {}
    assert value['qc'][0] < 0


@pytest.mark.parametrize('bad', [-6.26e-22, float('nan')])
def test_invalid_raw_source_is_refused_before_any_roundoff_repair(bad):
    with pytest.raises(OfflineChildContractError, match='before interpolation'):
        _validate_parent_mass_fields({'qc': np.array([0., bad], dtype=np.float32)})
