"""A future forcing tail must preserve every previous coefficient word."""
from dataclasses import dataclass
import numpy as np
import pytest
from gpuwm.io.wrf_boundary_extension import require_equal

@dataclass(frozen=True)
class Interval:
    start: float
    fields: dict


def test_prefix_rejects_one_changed_coefficient_and_signed_zero():
    old=Interval(0., {'q':np.array([1., 0.],dtype=np.float32)})
    new=Interval(0., {'q':old.fields['q'].copy()})
    require_equal(old,new)
    new.fields['q'][0]=np.nextafter(np.float32(1.),np.float32(2.))
    with pytest.raises(ValueError,match='array words differ'):
        require_equal(old,new)
    new.fields['q'][:]=[1.,-0.]
    with pytest.raises(ValueError,match='array words differ'):
        require_equal(old,new)


def test_prefix_rejects_changed_clock_or_missing_field():
    old=Interval(0.,{'u':np.zeros(2,dtype=np.float32)})
    with pytest.raises(ValueError):require_equal(old,Interval(1.,old.fields))
    with pytest.raises(ValueError):require_equal(old,Interval(0.,{}))
