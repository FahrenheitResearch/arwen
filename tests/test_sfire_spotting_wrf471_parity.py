"""Original compiled firebrand helpers, including mixed precision arithmetic."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from conftest import requires_gpu


def load(name):
    root = Path(__file__).parents[1]/"tools/sfire_wrf471_oracle/fixtures/spotting"
    receipt = json.loads((root/"receipt.json").read_text())
    path = root/("spotting/"+name+".npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["cases"]["spotting/"+name]["sha256"]
    with np.load(path, allow_pickle=False) as f:
        return dict(f)


def load_driver(name):
    root=Path(__file__).parents[1]/"tools/sfire_wrf471_oracle/fixtures/spotting_driver"
    receipt=json.loads((root/"receipt.json").read_text())
    path=root/("spotting_driver/"+name+".npz")
    assert receipt["reference_kind"]=="explicitly corrected original WRF firebrand driver"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt["cases"]["spotting_driver/"+name]["sha256"]
    with np.load(path,allow_pickle=False) as f:
        return dict(f)


def load_mpi(rank):
    root=Path(__file__).parents[1]/"tools/sfire_wrf471_oracle/fixtures/spotting_mpi"
    receipt=json.loads((root/"receipt.json").read_text())
    name=f"spotting_mpi/rank_{rank}"
    path=root/(name+".npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt["cases"][name]["sha256"]
    with np.load(path,allow_pickle=False) as f:
        return dict(f)


@requires_gpu
def test_initial_firebrand_properties_match_native_wrf():
    import cupy as cp
    from gpuwm.core.sfire_spotting import properties
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = load("property")
    assert_bit_exact(cp.asnumpy(properties(f["inputs"],original_property=True)),f["prop"],"original firebrand property control")


@requires_gpu
def test_initial_firebrand_mass_matches_cubic_volume_control():
    import cupy as cp
    from gpuwm.core.sfire_spotting import properties
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f=load("property_corrected")
    assert_bit_exact(cp.asnumpy(properties(f["inputs"])),f["prop"],"corrected spherical firebrand mass")


@requires_gpu
@pytest.mark.parametrize("case", range(1, 5))
@pytest.mark.parametrize("mode", ["burnout", "termvel"])
def test_firebrand_physics_matches_native_wrf(case, mode):
    import cupy as cp
    from gpuwm.core.sfire_spotting import particle_physics
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = load(f"{mode}_{case}")
    prop, height = particle_physics(f["input"],f.get("height_in",np.ones(32,np.float32)),
        float(f["dt"]),*[f[k] for k in ("pressure", "density", "temperature", "wind")],mode=mode)
    assert_bit_exact(cp.asnumpy(prop),f["output"],f"firebrand {mode}")
    if mode == "termvel":
        assert_bit_exact(cp.asnumpy(height),f["height_out"],"firebrand settled height")


@requires_gpu
def test_particle_box_interpolation_matches_native_wrf():
    import cupy as cp
    from gpuwm.core.sfire_spotting import interpolate_box
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = load("interp_box")
    assert_bit_exact(cp.asnumpy(interpolate_box(f["coordinates"],f["corners"])),f["output"],"particle box interpolation")


@requires_gpu
def test_particle_meteorology_sampling_matches_native_wrf():
    import cupy as cp
    from gpuwm.core.sfire_spotting import sample_at_particles
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = load("sampling")
    out = sample_at_particles(f["coordinates"],f,origin=(-3,-3),tile=(1,9,1,7))
    assert_bit_exact(cp.asnumpy(out),f["output"],"particle meteorology sampling")


@requires_gpu
@pytest.mark.parametrize("case", range(1, 4))
def test_particle_advection_matches_native_wrf(case):
    import cupy as cp
    from gpuwm.core.sfire_spotting import advect_particles
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = load(f"advection_{case}")
    coords, prop, _ = advect_particles(f["coordinates"],f["life"],f["input"],f,float(f["dt"]),
        origin=(-3,-3),tile=(1,9,1,7),original_cleanup=True)
    assert_bit_exact(cp.asnumpy(prop),f["output"],"particle advected properties")
    assert_bit_exact(cp.asnumpy(coords),np.stack([f[k] for k in ("x_out","y_out","h_out")]),"particle advected coordinates")


@requires_gpu
@pytest.mark.parametrize("case", range(1, 7))
def test_release_height_distribution_matches_native_wrf(case):
    import cupy as cp
    from gpuwm.core.sfire_spotting import release_heights
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = load(f"release_{case}")
    coords, prop = release_heights(np.stack([f[k] for k in ("x_in","y_in","h_in")]),f["prop_in"],
        points=int(f["points"]),levels=int(f["levels"]),seed=int(f["seed"]))
    assert_bit_exact(cp.asnumpy(coords),np.stack([f[k] for k in ("x_out","y_out","h_out")]),"firebrand release coordinates")
    assert_bit_exact(cp.asnumpy(prop),f["prop_out"],"firebrand release properties")


@requires_gpu
@pytest.mark.parametrize("case", range(1, 7))
def test_packed_particle_generation_matches_native_wrf(case):
    import cupy as cp
    from gpuwm.core.sfire_spotting import generate_particles
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = load(f"generate_{case}")
    kwargs = dict(release_life=f["release_life"],release_source=f["release_source"]) if bool(f["imported"]) else {}
    coords, prop, ident, release, control = generate_particles(
        np.stack([f[k] for k in ("x_in","y_in","h_in")]),f["prop_in"],
        np.stack([f[k] for k in ("id_in","source_in","life_in")]),
        np.stack([f[k] for k in ("release_x","release_y","release_h")]),f["release_prop"],
        np.array([int(f["idmax_in"]),0,0],np.int32),**kwargs)
    assert_bit_exact(cp.asnumpy(coords),np.stack([f[k] for k in ("x_out","y_out","h_out")]),"generated firebrand coordinates")
    assert_bit_exact(cp.asnumpy(prop),f["prop_out"],"generated firebrand properties")
    assert np.array_equal(cp.asnumpy(ident),np.stack([f[k] for k in ("id_out","source_out","life_out")]))
    assert np.array_equal(cp.asnumpy(control),f["control_out"])
    assert_bit_exact(cp.asnumpy(release),np.stack([f[k] for k in ("release_x_out","release_y_out","release_h_out")]),"consumed release coordinates")


@requires_gpu
def test_approximate_release_rank_matches_native_wrf():
    import cupy as cp
    from gpuwm.core.sfire_spotting import approximate_order
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f = load("ranking")
    for order in range(0,25,6):
        assert_bit_exact(cp.asnumpy(approximate_order(f["values"],order)),f[f"order_{order}"],"native release rank")


@requires_gpu
@pytest.mark.parametrize("case",range(1,5))
def test_particle_column_preparation_matches_corrected_native_driver(case):
    import cupy as cp
    from gpuwm.core.sfire_spotting import prepare_native_fields
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f=load_driver(f"input_{case}")
    out=prepare_native_fields(f,p_top=50000.,rdx=float(f["rdx"]),rdy=float(f["rdy"]),use_theta_m=case==3)
    for name in ("u","v","w","pressure","theta","density","height","p8w"):
        assert_bit_exact(cp.asnumpy(out[name]),f["prepared_"+name],"firebrand prepared "+name)


@requires_gpu
@pytest.mark.parametrize("case",range(1,5))
@pytest.mark.parametrize("mapped_fields",[False,True])
def test_complete_firebrand_driver_matches_corrected_native_wrf(case,mapped_fields):
    import cupy as cp
    from gpuwm.core.sfire_spotting import (SpottingState,SpottingOptions,prepare_native_fields,
        PARTICLE_COORDS,PARTICLE_PROPERTIES,PARTICLE_IDENTIFIERS)
    from gpuwm.core.fp32_ulp import assert_bit_exact
    initial=load_driver(f"input_{case}")
    options=SpottingOptions(fs_array_maxsize=256,fs_firebrand_gen_lim=7,fs_firebrand_gen_dt=2,
        fs_firebrand_gen_levels=3,fs_firebrand_gen_mom3d_dt=1,fs_firebrand_gen_levrand=case==3)
    allocator=None
    if mapped_fields:
        from tilestream.hoststore import alloc_pinned_array
        from tilestream.sfire_spotting import mapped_host_array
        def allocator(shape,dtype):
            return mapped_host_array(alloc_pinned_array(shape,dtype))
    state=SpottingState((4,6),2,3,options,field_allocator=allocator)
    for keys,field in ((PARTICLE_COORDS,state.coordinates),(PARTICLE_PROPERTIES,state.properties),
                       (PARTICLE_IDENTIFIERS,state.identifiers)):
        for row,key in enumerate(keys):
            field[row]=cp.asarray(initial[key])
    for key in state.data:
        cp.copyto(state.data[key],cp.asarray(initial[key]))
    state.control[0]=int(initial["fs_gen_idmax"])
    state.control[1]=int(np.count_nonzero(initial["fs_p_id"]))
    fields=prepare_native_fields(initial,p_top=50000.,rdx=float(initial["rdx"]),rdy=float(initial["rdy"]),use_theta_m=case==3)
    fields["dt"]=.125
    base={key:initial[key] for key in ("fgip","fmc_g","nfuel_cat","fire_area")}
    if mapped_fields:
        payload={key:allocator((state.fine_shape[0]+2,state.fine_shape[1]+2),cp.float32)[1:-1,1:-1]
                 for key in (*base,"burnt_area_dt")}
        for key in base:
            cp.copyto(payload[key],cp.asarray(base[key]))
        base={key:payload[key] for key in base}
    for step in range(1,13):
        expected=load_driver(f"frame_{case}_{step}")
        burn=expected["burnt_area_dt"]
        if mapped_fields:
            cp.copyto(payload["burnt_area_dt"],cp.asarray(burn));burn=payload["burnt_area_dt"]
        state.advance_native(fields,{**base,"burnt_area_dt":burn},step=step,history_alarm=step%4==0)
        for key,value in state.arrays().items():
            if key=="control":
                continue
            actual=cp.asnumpy(value)
            if value.dtype==cp.int32:
                assert np.array_equal(actual,expected[key]),(case,step,key)
            else:
                assert_bit_exact(actual,expected[key],f"spotting driver {case}/{step}/{key}")
        actual_clocks=np.array([int(state.control[0].item()),state.last_gen_dt,int(state.count_reset)],np.int32)
        expected_clocks=np.array([int(expected["fs_gen_idmax"]),int(expected["fs_last_gen_dt"]),int(expected["fs_count_reset"])],np.int32)
        assert np.array_equal(actual_clocks,expected_clocks),(case,step,"firebrand clocks")


@requires_gpu
@pytest.mark.parametrize("case",range(1,4))
@pytest.mark.parametrize("profiles",[False,True])
def test_compact_atmosphere_preparation_matches_native_physical_columns(case,profiles):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core.sfire_spotting import prepare_atmosphere
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f=load_driver(f"input_{case}")
    nz,ny,nx=7,4,6
    core=(slice(None,nz),slice(4,4+ny),slice(4,4+nx))
    full=(slice(None),slice(4,4+ny),slice(4,4+nx))
    # Dry mass theta comes from compiled native normalization. The compact
    # representation retains a base profile and its perturbation.
    attrs=dict(u=cp.asarray(f["u"][:nz,4:4+ny,4:5+nx]),v=cp.asarray(f["v"][:nz,4:5+ny,4:4+nx]),
        w=cp.asarray(f["w"][full]),php=cp.asarray(f["ph"][full]),phb=cp.asarray(f["phb"][full]),
        p=cp.asarray(f["p"][core]),pb=cp.asarray(f["pb"][core]),
        thp=cp.asarray(f["prepared_theta_mass"][core]-np.float32(300.)),thb=cp.full(nz,300.,cp.float32),
        al=cp.asarray(f["al"][core]),alb=cp.asarray(f["alb"][core]),msft=cp.asarray(f["msftx"][4:4+ny,4:4+nx]),
        mup=cp.zeros((ny,nx),cp.float32),mub2d=cp.asarray(f["muts"][4:4+ny,4:4+nx]),p_top=50000.)
    attrs.update({key:cp.asarray(f[key][:nz]) for key in ("c1h","c2h","dnw","fnm","fnp")})
    if profiles:
        attrs["pb"]=cp.asarray(f["pb"][:nz,4,4])
        attrs["alb"]=cp.asarray(f["alb"][:nz,4,4])
        if case==1:
            attrs["phb"]=cp.asarray(f["phb"][:,4,4])
    attrs.update({key:cp.asarray(f["moist"][m][core]) for m,key in enumerate(("qv","qc","qr"))})
    got=prepare_atmosphere(SimpleNamespace(**attrs),SimpleNamespace(nx=nx,ny=ny,dx=100.,dy=100.))
    for key in ("u","v","w","pressure","theta","density","height","p8w"):
        # Boundary padding uses the available compact atmospheric cells.
        # Interior samples require no missing external mass or top-U halo.
        expected=f["prepared_"+key]
        a=got[key][:nz-1,5:ny+3,5:nx+3]
        b=expected[:nz-1,5:ny+3,5:nx+3]
        assert_bit_exact(cp.asnumpy(a),b,"compact prepared "+key)


@requires_gpu
def test_eight_neighbor_packets_match_actual_compiled_wrf_mpi_exchange():
    import cupy as cp
    from gpuwm.core.sfire_spotting import neighbor_packets,concatenate_neighbor_packets
    from gpuwm.core.fp32_ulp import assert_bit_exact
    refs=[load_mpi(rank) for rank in range(9)]
    sent=[]
    for f in refs:
        packets,counts=neighbor_packets(np.stack([f[k] for k in ("x","y","z")]),
            np.stack([f[k] for k in ("mass","diam","effd","temp","velocity")]),
            np.stack([f[k] for k in ("id","source","life")]),f["mask"],f["neighbors"],tile=tuple(f["tile"]))
        sent.append(packets)
        assert int(counts.sum().item())<=len(f["id"])
    for rank,f in enumerate(refs):
        arrivals=[]
        for source in f["neighbors"]:
            if int(source)<0:
                arrivals.append((cp.empty((8,0),cp.float32),cp.empty((3,0),cp.int32)))
            else:
                opposite=int(np.flatnonzero(refs[int(source)]["neighbors"]==rank)[0])
                arrivals.append(sent[int(source)][opposite])
        received_real,received_int=concatenate_neighbor_packets(arrivals)
        assert np.array_equal(np.array([p[0].shape[1] for p in arrivals],np.int32),f["received_counts"])
        expected_real=np.stack([f["received_"+k] for k in ("x","y","z","mass","diam","effd","temp","velocity")])
        expected_int=np.stack([f["received_"+k] for k in ("id","source","life")])
        assert_bit_exact(cp.asnumpy(received_real),expected_real,f"native MPI rank {rank} real packet")
        assert np.array_equal(cp.asnumpy(received_int),expected_int),(rank,"native MPI integer packet")
