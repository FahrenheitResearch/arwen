"""Forecast radar initialization matches its scalar build without moving the column oracle."""
import numpy as np
import pytest
from gpuwm.core.wsm6_constants import (
    WRF_RADAR_F64,WRF_SCALAR_RADAR_F64_OVERRIDES,wrf_radar_words)


@pytest.mark.parametrize("hail",[0,1])
def test_both_initialization_profiles_keep_the_complete_measured_inventory(hail):
    a,b=wrf_radar_words(hail)
    c,d=wrf_radar_words(hail,initialization="scalar_build")
    old=np.asarray([v for _,v in WRF_RADAR_F64],np.uint64)
    assert np.array_equal(b.view(np.uint64),old)
    assert np.array_equal(a.view(np.uint32),c.view(np.uint32))
    changed={n for (n,_),x,y in zip(WRF_RADAR_F64,b.view(np.uint64),d.view(np.uint64)) if x!=y}
    assert changed==set(WRF_SCALAR_RADAR_F64_OVERRIDES)
    for i,(n,_) in enumerate(WRF_RADAR_F64):
        assert int(d.view(np.uint64)[i])==WRF_SCALAR_RADAR_F64_OVERRIDES.get(n,int(old[i]))


def test_unmeasured_initialization_is_refused():
    with pytest.raises(ValueError,match="column_oracle or scalar_build"):
        wrf_radar_words(0,initialization="unknown")
