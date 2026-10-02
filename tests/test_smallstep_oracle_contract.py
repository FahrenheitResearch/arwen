"""The compiled oracle cannot silently transpose fields or accept wrong buffers."""
import json
import hashlib
from pathlib import Path
import numpy as np
from gpuwm.verify.smallstep_oracle import ORACLE_DIR, WRFOracle, load_cases, word_metrics


def _layout(periodic=False):
    oracle = WRFOracle.__new__(WRFOracle)
    oracle.nx,oracle.ny,oracle.nz,oracle.periodic=8,7,49,periodic
    return oracle


def test_real_state_fixture_retains_staggering_and_all_levels():
    cases=list(load_cases())
    assert {name for name,_,_ in cases} == {"real_initial","real_evolved","steep_terrain",
                                           "map_extremes","zero_near_zero","southern_hemisphere"}
    for _,raw,meta in cases:
        nx,ny,nz=meta["nx"],meta["ny"],meta["nz"]
        assert raw["T"].shape==(nz,ny,nx)==(49,7,8)
        assert raw["U"].shape==(nz,ny,nx+1)
        assert raw["V"].shape==(nz,ny+1,nx)
        assert raw["PH"].shape==raw["W"].shape==(nz+1,ny,nx)
        assert all(a.dtype==np.float32 for a in raw.values())
    initial=cases[0][1]
    assert np.ptp(initial["PB"])>50000
    assert np.ptp(initial["U"])>1
    assert np.all(initial["ALT"]>0)


def test_wrf_array_layout_round_trip_and_clamped_ghosts():
    oracle=_layout()
    _,raw,_=next(load_cases())
    for key in ("U","V","W","T","MU","C1F"):
        original=raw[key]
        packed=oracle.array(original)
        assert packed.flags.f_contiguous
        assert np.array_equal(oracle.extract(packed,original.shape).view(np.uint32),original.view(np.uint32))
        if original.ndim==3:
            assert np.array_equal(packed[0],packed[1])
            assert np.array_equal(packed[:,:,0],packed[:,:,1])


def test_periodic_staggered_halos_use_the_mass_grid_period():
    oracle=_layout(True)
    u=np.broadcast_to(np.arange(9,dtype=np.float32),(49,7,9)).copy()
    u[:,:,-1]=u[:,:,0]
    packed=oracle.array(u)
    assert np.all(packed[0]==7)
    assert np.all(packed[9]==0)
    assert np.all(packed[10]==1)
    v=np.broadcast_to(np.arange(8,dtype=np.float32)[None,:,None],(49,8,8)).copy()
    v[:,-1]=v[:,0]
    packed=oracle.array(v)
    assert np.all(packed[:,:,0]==6)
    assert np.all(packed[:,:,8]==0)
    assert np.all(packed[:,:,9]==1)


def test_word_comparison_preserves_signed_zero_and_nonfinite_mismatches():
    measured=word_metrics(np.array([0.0,np.inf,np.nan],np.float32),
                          np.array([-0.0,-np.inf,1.0],np.float32))
    assert measured["different_words"]==3
    assert measured["nonfinite_different_words"]==2
    assert measured["max_ulp"]==0


def test_real_wrf_argument_schema_retains_every_output():
    schema=json.loads((ORACLE_DIR/"wrf-schema.json").read_text())
    assert len(schema["routines"] )==8
    assert set(schema["routines"]["calc_p_rho"]["args"]) >= {"al","p","ph","pm1","t0","ids","ims","its"}
    assert set(schema["routines"]["advance_mu_t"]["args"]) >= {"muave","muts","mudf","t_ave","ww_1"}
    assert schema["routines"]["advance_w"]["declarations"]["t_2ave"]["intent"]=="inout"


def test_smallstep_fixture_builder_and_kernel_receipt_hashes():
    root=Path(__file__).resolve().parents[1]
    pins=json.loads((ORACLE_DIR/"oracle-sha256sums.json").read_text())
    assert len(pins)>=160
    assert "tools/smallstep_wrf471_oracle/build.py" in pins
    assert "gpuwm/core/kernels/acoustic.cu" in pins
    for name,digest in pins.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
