"""Independent word and hydrostatic checks for the Rust host analysis arm."""
from __future__ import annotations
import numpy as np
import pytest
from gpuwm.da import host_analysis as host
from gpuwm.da import hydrostatic as hydro
from gpuwm.io import analysis_bundle as ab
from analysis_bundle_fixtures import write_wrf_state


@pytest.mark.parametrize("length", [1,7,8,9,127,128,129,255,256,257,8193])
@pytest.mark.parametrize("subtract", [False,True])
def test_stored_words_and_pairwise_statistics(length, subtract):
    random = np.random.default_rng(619)
    first, second = random.normal(size=(2,length))
    expected = first-second if subtract else first+second
    actual, stats = host.binary(first,second,subtract=subtract)
    assert actual.tobytes() == expected.tobytes()
    assert stats["rms"] == float(np.sqrt(np.mean(expected*expected)))
    assert stats["max_abs"] == float(np.max(np.abs(expected)))
    clipped, clipstats = host.binary(first,second,subtract=subtract,clip=True)
    assert clipped.tobytes() == np.where(expected<0.,0.,expected).tobytes()
    assert clipstats["values"] == np.count_nonzero(expected<0.)


def test_signed_zero_and_refusals():
    value,_ = host.binary(np.array([-0.]),np.array([0.]),subtract=True,clip=True)
    assert value.tobytes() == np.array([-0.]).tobytes()
    value,_ = host.binary(np.array([-0.]),np.array([0.]),subtract=True,clip=2)
    assert value.tobytes() == np.array([0.]).tobytes()
    with pytest.raises(ValueError,match="not finite"):
        host.binary([np.inf],[0.])
    with pytest.raises(ValueError):
        host.binary(np.zeros((2,3)),np.zeros((3,2)))


@pytest.mark.parametrize("option", [1,2])
def test_fixed_mass_single_warm_layer_has_analytic_thickness(tmp_path,option):
    state=ab.column_from_file(write_wrf_state(tmp_path/"state.nc"))
    before=hydro.capture_column(state)
    after=dict(before)
    after["thp"]=before["thp"].copy()
    after["thp"][1,2,3]+=0.25
    actual,dp=host.hydrostatic(state,before,after,hypsometric_opt=option)
    # Moist loading and dry mass are unchanged, so pressure recurrence cancels.
    assert np.count_nonzero(dp)==0
    assert np.count_nonzero(actual[:2])==0
    assert np.array_equal(np.argwhere(np.any(actual!=0.,axis=0)),[[2,3]])
    from gpuwm.core import constants as c
    p=before["p"][1,2,3]
    theta=state.thb[1]+before["thp"][1,2,3]
    def density(t):
        return c.RD*t*(1.+c.RVOVRD*before["qv"][1,2,3])*(p/c.P0)**c.RCP/p
    mu=state.mub2d[2,3]+state.mup[2,3]
    if option==1:
        operator=-state.dnw[1]*(state.c1h[1]*mu+state.c2h[1])
    else:
        upper=state.c3f[2]*mu+state.c4f[2]+state.p_top
        lower=state.c3f[1]*mu+state.c4f[1]+state.p_top
        mid=state.c3h[1]*mu+state.c4h[1]+state.p_top
        operator=mid*np.log(lower/upper)
    expected=operator*(density(theta+0.25)-density(theta))
    np.testing.assert_allclose(actual[2:,2,3],expected,rtol=2e-12,atol=2e-10)
    zero,_=host.hydrostatic(state,before,before,hypsometric_opt=option)
    assert np.count_nonzero(zero)==0


@pytest.mark.parametrize("option", [1,2])
def test_column_blocks_on_threads_are_the_one_call_words(tmp_path,option,monkeypatch):
    # single-core law (2026-10-27): the kernel integrates every column alone, so column blocks on threads give the
    # one-call words, mass moving or not
    state=ab.column_from_file(write_wrf_state(tmp_path/"state.nc"))
    before=hydro.capture_column(state)
    after=dict(before)
    after["thp"]=before["thp"]+np.linspace(0.,0.5,before["thp"].size).reshape(before["thp"].shape)
    after["qv"]=before["qv"]*1.01
    mup=state.mup+np.linspace(-20.,20.,state.mup.size).reshape(state.mup.shape)
    for mass in (None, mup):
        monkeypatch.setenv("GPUWM_BUNDLE_THREADS","1")
        one=host.hydrostatic(state,before,after,hypsometric_opt=option,mup_after=mass)
        monkeypatch.setenv("GPUWM_BUNDLE_THREADS","3")
        monkeypatch.setattr(host,"HYDRO_COLUMNS_PER_THREAD",1)
        many=host.hydrostatic(state,before,after,hypsometric_opt=option,mup_after=mass)
        monkeypatch.setattr(host,"HYDRO_COLUMNS_PER_THREAD",16384)
        for a,b in zip(one,many):
            assert a.tobytes()==b.tobytes()
