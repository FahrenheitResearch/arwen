"""WRF module_advect_em.F:9495-10560, full-domain final-stage identity."""
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from gpuwm.core import chem_advect_mono as mono
from gpuwm.verify.chem_oracle import ORACLE_ROOT, load, ulp_table

CASES = load(ORACLE_ROOT / "core" / "advect_mono")
_MUTATED_MODULES = {}


def test_fixture_inventory_and_hashes():
    expected={f"b{b}_map{m}_front{f}_v{v}_c{c}" for b in range(3) for m in range(2)
        for f in range(2) for v in (3,5) for c in (1,2,3)} | {"bounds_v3","bounds_v5"}
    assert set(CASES)==expected
    root=ORACLE_ROOT/"core"/"advect_mono"
    covered=set()
    for line in (root/"fixture-sha256sums.txt").read_text(encoding="utf-8").splitlines():
        digest,relative=line.split(maxsplit=1)
        relative=relative.lstrip("*")
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
        covered.add(relative)
    assert covered=={p.relative_to(root).as_posix() for case in expected for p in (root/case).iterdir()}


def inputs(case):
    c = CASES[case]
    # WRF dumps (i,k,j); engine and CUDA store (k,j,i).
    volume = lambda n: np.ascontiguousarray(c[n].transpose(1,2,0))
    plane = lambda n: np.ascontiguousarray(c[n].T)
    coord = SimpleNamespace(**{n:c[src] for n,src in
        (("c1h","c1"),("c2h","c2"),("rdnw","rd"),("fnm","fnm"),("fnp","fnp"))})
    return dict(q=volume("q"),q0=volume("q0"),ru=volume("ru"),rv=volume("rv"),rw=volume("ww"),
        rw_implicit=volume("wi"),mut=plane("mut"),mu_old=plane("mu0"),mub=plane("mub"),
        coord=coord,dx=c["dx"],dy=c["dy"],dt=c["dt"],tend=volume("initial_tendency"),
        msft=plane("mx"),msfty=plane("my"),open_x=bool(c["boundary"]),open_y=bool(c["boundary"]),
        boundary="open" if c["boundary"]==2 else "specified",v_order=int(c["vorder"]))


def assert_outputs(case, result):
    c = CASES[case]
    for name in ("tendency","h_tendency","z_tendency"):
        actual = result[name]
        if name=="h_tendency" and c["boundary"]: actual=actual[:,1:-1,1:-1]
        expected = c[name].transpose(1,2,0)
        table = ulp_table(actual,expected)
        print(case,name,table)
        assert table==dict(max_ulp=0,n_nonzero=0,n=expected.size), (case,name,table)
        # ULP distance treats signed zeros alike; the word check also pins sign.
        np.testing.assert_array_equal(actual.view(np.uint32),expected.copy().view(np.uint32))


@pytest.mark.parametrize("case",sorted(CASES))
def test_cpu_oracle(case):
    assert_outputs(case,mono.np_advect_mono(**inputs(case)))


@pytest.mark.parametrize("vorder",(3,5))
def test_monotone_steps_and_pd_counterexample(vorder):
    p=inputs(f"bounds_v{vorder}")
    q=p["q0"].copy()
    lo,hi=q.min(),q.max()
    for _ in range(20):
        result=mono.np_advect_mono(**dict(p,q=q,q0=q))
        q=(p["mut"]*q+p["dt"]*result["tendency"])/p["mut"]
        assert q.min()>=lo and q.max()<=hi
    # Independent pinned PD Fortran output, not a surrogate limiter.
    c=CASES[f"bounds_v{vorder}"]
    pd=c["pd_tendency"].transpose(1,2,0)
    pd_q=(p["mut"]*p["q0"]+p["dt"]*pd)/p["mut"]
    print("pd_counterexample",vorder,float(pd_q.min()),float(pd_q.max()))
    assert pd_q.min()>=0
    assert pd_q.min()<lo
    assert pd_q.max()>hi
    if vorder==3:
        from gpuwm.verify.npref import np_pd_fluxes,np_pd_renorm_apply
        # The existing engine PD mirror is float64 and implements v_order=3.
        # It is a second positivity-only counterexample, not an FP32 oracle.
        fl=np_pd_fluxes(p["q0"],p["q0"],p["ru"],p["rv"],p["rw"],p["mut"],
            p["coord"],p["dx"],p["dy"],p["dt"])
        t=np_pd_renorm_apply(p["q0"],p["mut"],*fl,coord=p["coord"],dx=p["dx"],dy=p["dy"],dt=p["dt"])
        engine=(p["mut"]*p["q0"]+p["dt"]*t)/p["mut"]
        print("engine_pd_mirror_counterexample",float(engine.min()),float(engine.max()))
        assert engine.min()>=0 and engine.min()<lo and engine.max()>hi


def test_additive_contract():
    p=inputs("b0_map1_front1_v5_c2")
    nonzero=mono.np_advect_mono(**p)["tendency"]
    zero=mono.np_advect_mono(**dict(p,tend=np.zeros_like(p["q0"])))["tendency"]
    assert np.any(nonzero.view(np.uint32)!=zero.view(np.uint32))
    # Reference parity above pins the exact sequential subtractions with a
    # nonzero accumulator; this check rules out replacing that accumulator.
    still=p["tend"].copy()
    off=dict(p,ru=np.zeros_like(p["ru"]),rv=np.zeros_like(p["rv"]),rw=np.zeros_like(p["rw"]))
    np.testing.assert_array_equal(mono.np_advect_mono(**off)["tendency"],still)


def test_roundoff_residual_is_not_silently_clipped():
    # WRF's limiter budget and final tendency have different float32
    # associations. A general exact bound is not a float32 guarantee.
    p=inputs("bounds_v3")
    p.update(dx=np.float32(1000),dt=np.float32(20),
        mut=np.full_like(p["mut"],80000),mub=np.full_like(p["mub"],80000),
        ru=np.full_like(p["ru"],1600000))
    q=np.zeros_like(p["q0"]); q[:,:,6:]=1
    minimum=np.float32(0)
    for _ in range(20):
        t=mono.np_advect_mono(**dict(p,q=q,q0=q))["tendency"]
        q=(p["mut"]*q+p["dt"]*t)/p["mut"]
        minimum=min(minimum,q.min())
    print("unclipped_float32_residual",float(minimum),float(q.max()))
    assert minimum<0
    assert minimum>=np.float32(-1e-10)
    assert q.max()<=1


@pytest.mark.parametrize("keyword,value,match",(("h_order",3,"no other horizontal"),
    ("v_order",4,"other vertical"),("boundary","unknown","omit radiation")))
def test_unsupported_stencils_refuse(keyword,value,match):
    p=inputs("bounds_v3")
    with pytest.raises(ValueError,match=match):
        mono.np_advect_mono(**dict(p,**{keyword:value}))


def test_launcher_abi_and_workspace_guards(monkeypatch):
    """Exercise orchestration with host buffers and a recording CUDA stub."""
    class DeviceArray:
        def __init__(self,a):
            self.array=np.asarray(a,dtype=np.float32)
            self.shape=self.array.shape; self.dtype=self.array.dtype
            self.flags=self.array.flags; self.nbytes=self.array.nbytes
            self.data=SimpleNamespace(ptr=self.array.__array_interface__["data"][0])
            self.device=SimpleNamespace(id=0)

    def array(a,dtype=None):
        return a if isinstance(a,DeviceArray) else DeviceArray(np.array(a,dtype=dtype,copy=True))

    cp=SimpleNamespace(ndarray=DeviceArray,asarray=array,
        empty=lambda s,dtype:DeviceArray(np.empty(s,dtype)),
        ones=lambda s,dtype:DeviceArray(np.ones(s,dtype)),
        zeros=lambda s,dtype:DeviceArray(np.zeros(s,dtype)),
        full=lambda s,v,dtype:DeviceArray(np.full(s,v,dtype)),
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda:0),
                             get_current_stream=lambda:SimpleNamespace(ptr=0)))
    calls=[]
    monkeypatch.setattr(mono,"_cupy",lambda:cp)
    monkeypatch.setattr(mono,"_kernel",lambda name:
        (lambda grid,block,args:calls.append((name,grid,block,args))))
    p=inputs("b1_map1_front1_v5_c2")
    d={n:array(a) if isinstance(a,np.ndarray) else a for n,a in p.items()}
    coord=SimpleNamespace(**{n:array(a) for n,a in vars(p["coord"]).items()})
    nz,ny,nx=p["q"].shape
    fl=tuple(cp.empty(s,np.float32) for s in ((nz,ny,nx+1),(nz,ny,nx+1),
        (nz,ny+1,nx),(nz,ny+1,nx),(nz+1,ny,nx),(nz+1,ny,nx)))
    work=mono.launch_mono_fluxes(*(d[n] for n in ("q","q0","ru","rv","rw","mut")),
        coord,p["dx"],p["dy"],p["dt"],*fl,msft=d["msft"],msfty=d["msfty"],
        open_x=True,open_y=True,v_order=5,rw_implicit=d["rw_implicit"])
    args=(d["q0"],d["mu_old"],*fl,d["tend"],coord,p["dx"],p["dy"],p["dt"])
    with pytest.raises(ValueError,match="fresh matching"):
        mono.launch_mono_renorm_apply(*args,open_x=True,open_y=True)
    with pytest.raises(ValueError,match="bounds changed"):
        mono.launch_mono_renorm_apply(*args,workspace=work)
    with pytest.raises(ValueError,match="overlap"):
        alias=list(args); alias[8]=d["q0"]
        mono.launch_mono_renorm_apply(*alias,workspace=work,open_x=True,open_y=True)
    with pytest.raises(ValueError,match="overlap"):
        alias=list(args); alias[1]=DeviceArray(work.scratch[2].array[0])
        mono.launch_mono_renorm_apply(*alias,workspace=work,open_x=True,open_y=True)
    result=mono.launch_mono_renorm_apply(*args,workspace=work,mub=d["mub"],
        open_x=True,open_y=True,msft=d["msft"],msfty=d["msfty"])
    assert result["tendency"] is d["tend"]
    assert [c[0] for c in calls]==["mono_fluxes","mono_scales","mono_apply"]
    for name,grid,block,kernel_args in calls:
        assert len(kernel_args)==39
        assert kernel_args[0] is d["q"] and kernel_args[1] is d["q0"]
        assert kernel_args[11] is d["msft"] and kernel_args[12] is d["msfty"]
        assert tuple(kernel_args[16:22])==fl
        assert tuple(kernel_args[32:])==(nz,ny,nx,1,1,0,5)
        assert block==(128,1,1)
    assert calls[1][3][14] is d["mu_old"] and calls[1][3][15] is d["mub"]
    assert calls[1][3][26] is d["tend"]
    with pytest.raises(ValueError,match="reused extrema"):
        mono.launch_mono_renorm_apply(*args,workspace=work,open_x=True,open_y=True)


@pytest.mark.gpu
@pytest.mark.parametrize("case",sorted(CASES))
def test_gpu_oracle(case):
    from conftest import HAS_GPU
    if not HAS_GPU: pytest.skip("no authorized CUDA device")
    import cupy as cp
    p=inputs(case)
    coord=SimpleNamespace(**{n:cp.asarray(a) for n,a in vars(p["coord"]).items()})
    device={n:cp.asarray(a) if isinstance(a,np.ndarray) else a for n,a in p.items()}
    q,q0,ru,rv,rw,mut=(device[n] for n in ("q","q0","ru","rv","rw","mut"))
    nz,ny,nx=q.shape
    fl=tuple(cp.empty(s,np.float32) for s in ((nz,ny,nx+1),(nz,ny,nx+1),
        (nz,ny+1,nx),(nz,ny+1,nx),(nz+1,ny,nx),(nz+1,ny,nx)))
    work=mono.launch_mono_fluxes(q,q0,ru,rv,rw,mut,coord,p["dx"],p["dy"],p["dt"],*fl,
        msft=device["msft"],msfty=device["msfty"],open_x=p["open_x"],open_y=p["open_y"],
        v_order=p["v_order"],boundary=p["boundary"],rw_implicit=device["rw_implicit"])
    result=mono.launch_mono_renorm_apply(q0,device["mu_old"],*fl,device["tend"],coord,
        p["dx"],p["dy"],p["dt"],msft=device["msft"],msfty=device["msfty"],
        open_x=p["open_x"],open_y=p["open_y"],workspace=work,mub=device["mub"])
    assert_outputs(case,{n:cp.asnumpy(result[n]) for n in ("tendency","h_tendency","z_tendency")})


@pytest.mark.gpu
@pytest.mark.parametrize("case",sorted(CASES))
def test_gpu_mutation_fires(case,monkeypatch):
    """Make each device parity assertion fire on an authorized GPU run."""
    from conftest import HAS_GPU
    if not HAS_GPU: pytest.skip("no authorized CUDA device")
    import cupy as cp
    from gpuwm.core.kernels import module_source
    kind="scales" if case.startswith("bounds_") else "coefficient"
    device=cp.cuda.runtime.getDevice()
    key=device,kind
    if key not in _MUTATED_MODULES:
        # The exact string the engine loader compiles, with the same
        # options, so only the mutation differs from the graded kernel.
        source=module_source("mono_advection")
        mutated=(source.replace("37.0f/60.0f","36.0f/60.0f") if kind=="coefficient"
                 else source.replace("p.si[c]=1.0f; p.so[c]=1.0f;",
                                     "p.si[c]=1.0f; p.so[c]=1.0f; return;"))
        assert mutated!=source
        _MUTATED_MODULES[key]=cp.RawModule(code=mutated,options=("-std=c++17",))
    monkeypatch.setattr(mono,"_kernel",lambda name:_MUTATED_MODULES[key].get_function(name))
    with pytest.raises(AssertionError): test_gpu_oracle(case)
