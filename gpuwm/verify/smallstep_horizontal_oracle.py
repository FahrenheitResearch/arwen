"""Word-level measurements of the explicit acoustic driver against WRF.

The reference is a call into compiled, pinned ``module_small_step_em.F``.
NumPy here packs the real-state fixture, fills lateral ghosts, and converts
the engine's full-theta coordinate. It does not implement a WRF tendency.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

F = np.float32


def _vertical(raw, name, nz, default=None):
    value = raw.get(name)
    if value is None:
        if default is None:
            raise KeyError(name)
        value = default
    value = np.asarray(value, dtype=F).reshape(-1)
    if value.size < nz + 1:
        value = np.pad(value, (0, nz + 1 - value.size), mode="edge")
    return value[:nz + 1].copy()


def _face_mass(mass, axis, periodic):
    prior = np.roll(mass, 1, axis=axis)
    if not periodic:
        sl = [slice(None)] * 2
        sl[axis] = 0
        prior[tuple(sl)] = mass[tuple(sl)]
    core = F(.5) * (prior + mass)
    sl = [slice(None)] * 2
    sl[axis] = slice(0, 1) if periodic else slice(-1, None)
    tail = core[tuple(sl)] if periodic else mass[tuple(sl)]
    return np.concatenate((core, tail), axis=axis)


def make_horizontal_state(raw, metadata, *, mode="reference", boundary="periodic",
                          map_factors=True, top_lid=False, near_zero=False,first=True,
                          moist_loading=False):
    """Pack a realistic stage reference and smooth acoustic perturbations.

    ``mode=zero-reference`` isolates the perturbation equations. The normal
    case retains the observed reference winds and asks WRF itself to diagnose
    their reference omega, so its subtraction is also measured.
    """
    t = np.asarray(raw["T"], dtype=F)
    nz, ny, nx = t.shape
    periodic = boundary == "periodic"
    full_theta = F(300) + t
    pfull = np.asarray(raw["PB"], dtype=F) + np.asarray(raw["P"], dtype=F)
    if "AL" in raw and "ALB" in raw:
        alt = np.asarray(raw["AL"], dtype=F) + np.asarray(raw["ALB"], dtype=F)
    else:
        alt = F(287) * full_theta * np.power(pfull / F(100000), F(287 / 1004.5)) / pfull
    yy, xx = np.mgrid[:ny, :nx]
    # Deterministic, smooth pressure and geopotential waves over the observed
    # terrain. These are acoustic-size disturbances, never random states.
    wave = np.sin(F(2 * np.pi) * xx / F(nx)) * np.cos(F(2 * np.pi) * yy / F(ny))
    wave = wave.astype(F)
    taper = np.linspace(1, .15, nz, dtype=F)[:, None, None]
    pp = (F(2) * taper * wave).astype(F)
    phpp = (F(.02) * np.linspace(0, 1, nz + 1, dtype=F)[:, None, None] * wave).astype(F)
    mut = np.asarray(raw["MUB"], dtype=F) + np.asarray(raw["MU"], dtype=F)
    mup = np.asarray(raw["MU"], dtype=F).copy()
    mpp = (F(.003) * wave).astype(F)
    thpp = (F(.05) * taper * wave).astype(F)
    u = np.asarray(raw["U"], dtype=F).copy()
    v = np.asarray(raw["V"], dtype=F).copy()
    if periodic:
        u[..., -1] = u[..., 0]
        v[:, -1, :] = v[:, 0, :]
    if mode == "zero-reference":
        u.fill(0)
        v.fill(0)
    c1h = _vertical(raw, "C1H", nz, np.ones(nz + 1, dtype=F))
    c2h = _vertical(raw, "C2H", nz, np.zeros(nz + 1, dtype=F))
    dnw = _vertical(raw, "DNW", nz)
    rdnw = _vertical(raw, "RDNW", nz, F(1) / dnw)
    fnm = _vertical(raw, "FNM", nz)
    fnp = _vertical(raw, "FNP", nz)
    msft = np.asarray(raw.get("MAPFAC_M", np.ones((ny, nx))), dtype=F).copy()
    msfu = np.asarray(raw.get("MAPFAC_U", np.ones((ny, nx + 1))), dtype=F).copy()
    msfv = np.asarray(raw.get("MAPFAC_V", np.ones((ny + 1, nx))), dtype=F).copy()
    if not map_factors:
        msft.fill(1); msfu.fill(1); msfv.fill(1)
    if periodic:
        msfu[..., -1] = msfu[..., 0]
        msfv[-1, :] = msfv[0, :]
    muu = _face_mass(mut, 1, periodic)
    muv = _face_mass(mut, 0, periodic)
    upp = (F(.00001) * (c1h[:nz, None, None] * muu + c2h[:nz, None, None]) * np.asarray(raw["U"], dtype=F) / msfu).astype(F)
    vpp = (F(.00001) * (c1h[:nz, None, None] * muv + c2h[:nz, None, None]) * np.asarray(raw["V"], dtype=F) / msfv).astype(F)
    if periodic:
        upp[..., -1] = upp[..., 0]
        vpp[:, -1, :] = vpp[:, 0, :]
    if near_zero:
        for a in (pp, phpp, mpp, thpp, upp, vpp):
            a *= F(1e-12)
    ph = np.asarray(raw["PH"], dtype=F).copy()
    phb = np.asarray(raw["PHB"], dtype=F).copy()
    # The driver accepts php at full levels and reconstructs its half-level
    # coefficient. The WRF argument below is calc_php's actual half-level
    # field, so this regrouping has an independently measured rounding cost.
    rut=(F(.04)*upp).astype(F)
    rvt=(F(.04)*vpp).astype(F)
    # Interior independent mass sources are absent in the native slow driver
    # (dycore._add_slow_tendencies). An arbitrary nonzero source conflicts
    # with the closed-top omega condition and invalidates a scalar-offset
    # comparison in the top cell, so the realistic stage forcing is zero.
    rmut=np.zeros_like(wave)
    native_ft=(F(.01)*taper*wave).astype(F)
    # The canonical/full-theta reference-advection offset is filled from
    # compiled WRF's reference mass divergence before either comparison.
    full_ft=native_ft.copy()
    if boundary == "specified":
        for a in (rut,rvt,rmut,native_ft,full_ft):
            a[...,0,:]=0; a[...,-1,:]=0
            a[...,:,0]=0; a[...,:,-1]=0
    data = dict(u=u, v=v, u_pp=upp, v_pp=vpp, ru_t=rut, rv_t=rvt,
                mup=mup, mub2d=np.asarray(raw["MUB"], dtype=F).copy(), mu_pp=mpp,
                thp=t.copy(), thb=np.full_like(t, F(300)), th_pp=thpp,
                p_pp=pp, p_pp_old=F(.95)*pp, ph_pp=phpp, php=ph, phb=phb,
                alt=alt.astype(F), al_pp=(F(1e-7)*taper*wave).astype(F),
                pb=np.asarray(raw["PB"], dtype=F).copy(),p=pfull.copy(), rmu_t=rmut,
                rth_t=full_ft,native_ft=native_ft, ww_pp=np.zeros((nz+1,ny,nx),dtype=F),
                c1h=c1h[:nz], c2h=c2h[:nz], fnm=fnm, fnp=fnp,
                c1f=_vertical(raw,"C1F",nz),c2f=_vertical(raw,"C2F",nz),
                rdn=_vertical(raw,"RDN",nz),w=np.asarray(raw["W"],dtype=F).copy(),
                ht=np.asarray(raw.get("HGT",np.zeros((ny,nx))),dtype=F).copy(),
                w_pp=np.zeros((nz+1,ny,nx),dtype=F),
                rw_t=np.zeros((nz+1,ny,nx),dtype=F),rph_t=np.zeros((nz+1,ny,nx),dtype=F),
                dnw=dnw[:nz], rdnw=rdnw[:nz], msft=msft, msfu=msfu, msfv=msfv)
    # Native WRF surface weights if available, otherwise the coordinate's
    # quadratic extrapolation; this creates inputs, never oracle outputs.
    if all(k in raw for k in ("CF1", "CF2", "CF3")):
        cf = [F(np.asarray(raw[k]).reshape(-1)[0]) for k in ("CF1", "CF2", "CF3")]
    else:
        dn = _vertical(raw, "DN", nz)
        co1 = (F(2)*dn[1]+dn[2])/(dn[1]+dn[2])*dnw[0]/dn[1]
        co2 = dn[1]/(dn[1]+dn[2])*dnw[0]/dn[2]
        cf = [F(fnp[1]+co1), F(fnm[1]-co1-co2), F(co2)]
    data.update(zip(("cf1", "cf2", "cf3"), cf))
    if moist_loading:
        for engine,wrf in (("qv","QVAPOR"),("qc","QCLOUD"),("qr","QRAIN"),
                           ("qi","QICE"),("qs","QSNOW"),("qg","QGRAUP")):
            data[engine]=np.asarray(raw[wrf],dtype=F).copy()
    cfg = SimpleNamespace(nx=nx,ny=ny,nz=nz,dx=float(metadata.get("DX",metadata.get("dx",3000))),
                          dy=float(metadata.get("DY",metadata.get("dy",3000))),
                          open_x=boundary=="open",open_y=boundary=="open",
                          specified=boundary=="specified",nested=False,spec_zone=1,
                          top_lid=bool(top_lid),smdiv=.1,epssm=.1,moist_cq=bool(moist_loading),mp_physics=8,
                          damp_opt=0,dampcoef=0.,zdamp=5000.,relax_w=False,first=bool(first))
    return data, cfg


def _pack_args(oracle, data, cfg, full_theta_probe=False):
    nz,ny,nx=cfg.nz,cfg.ny,cfg.nx
    def arr(a):
        if np.asarray(a).ndim == 1 and len(a) == nz:
            a=np.concatenate((a,np.asarray(a[-1:],dtype=F)))
        return oracle.array(a)
    flags=dict(periodic_x=not cfg.open_x and not cfg.specified,
               specified=cfg.specified,nested=False,
               open_xs=cfg.open_x,open_xe=cfg.open_x,
               open_ys=cfg.open_y,open_ye=cfg.open_y)
    common=dict(c1h=arr(data["c1h"]),c2h=arr(data["c2h"]),
                fnm=arr(data["fnm"]),fnp=arr(data["fnp"]),rdnw=arr(data["rdnw"]),
                rdx=F(1/cfg.dx),rdy=F(1/cfg.dy),dts=F(.25),config_flags=flags)
    for key in ("c1f","c2f","c3h","c4h","c3f","c4f"):
        common[key]=arr(np.zeros(nz+1,dtype=F))
    mu=data["mub2d"]+data["mup"]
    periodic=flags["periodic_x"]
    muu=_face_mass(mu,1,periodic); muv=_face_mass(mu,0,periodic)
    msfx=arr(data["msfu"]); msfy=arr(data["msfv"])
    pressure=data["p_pp"]
    if not getattr(cfg,"first",True):
        # WRF calc_p_rho's explicit REAL pressure-history expression.
        # The engine driver applies it internally through pdmp on later
        # substeps; the original routine takes that weighted pressure.
        pressure=pressure+F(cfg.smdiv)*(pressure-data["p_pp_old"])
    uv=dict(common,u=arr(data["u_pp"]),v=arr(data["v_pp"]),
            ru_tend=arr(data["ru_t"]),rv_tend=arr(data["rv_t"]),
            p=arr(pressure),pb=arr(data["pb"]),ph=arr(data["ph_pp"]),
            php=arr(F(.5)*(data["phb"][:-1]+data["phb"][1:]+data["php"][:-1]+data["php"][1:])),
            alt=arr(data["alt"]),al=arr(data["al_pp"]),mu=arr(data["mu_pp"]),
            muu=arr(muu),muv=arr(muv),mudf=arr(np.zeros((ny,nx),dtype=F)),
            cqu=arr(data.get("cqu",np.ones_like(data["u_pp"]))),
            cqv=arr(data.get("cqv",np.ones_like(data["v_pp"]))),
            msfux=msfx,msfuy=msfx,msfvx=msfy,msfvy=msfy,
            msfvx_inv=arr(F(1)/data["msfv"]),emdiv=F(0),spec_zone=cfg.spec_zone,
            cf1=data["cf1"],cf2=data["cf2"],cf3=data["cf3"],
            non_hydrostatic=True,top_lid=cfg.top_lid)
    theta_offset=F(300)*data["c1h"][:,None,None]*data["mu_pp"][None]
    mass=dict(common,u=uv["u"],v=uv["v"],u_1=arr(data["u"]),v_1=arr(data["v"]),
              mu=arr(data["mu_pp"]),mut=arr(mu),muu=arr(muu),muv=arr(muv),
              muave=arr(np.zeros((ny,nx),dtype=F)),muts=arr(np.zeros((ny,nx),dtype=F)),
              mudf=arr(np.zeros((ny,nx),dtype=F)),mu_tend=arr(data["rmu_t"]),
              ww=arr(data["ww_pp"]),ww_1=arr(np.zeros_like(data["ww_pp"])),
              t=arr(data["th_pp"] if full_theta_probe else data["th_pp"]-theta_offset),
              t_1=arr(data["thb"]+data["thp"] if full_theta_probe else data["thp"]),
              t_ave=arr(np.zeros_like(data["th_pp"])),
              ft=arr(data["rth_t"] if full_theta_probe else data["native_ft"]),
              uam=arr(np.zeros_like(data["u_pp"])),vam=arr(np.zeros_like(data["v_pp"])),
              wwam=arr(np.zeros_like(data["ww_pp"])),dnw=arr(data["dnw"]),
              msfux=msfx,msfuy=msfx,msfvx=msfy,msfvy=msfy,
              msfvx_inv=arr(F(1)/data["msfv"]),msftx=arr(data["msft"]),msfty=arr(data["msft"]),
              epssm=F(cfg.epssm),step=1)
    return uv,mass


def run_horizontal_case(raw, metadata, library_path, *, full_theta_probe=False,
                        no_fma_probe=False, **case_options):
    """Call WRF and the real engine explicit launch path on the same words."""
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep_explicit
    from gpuwm.verify.smallstep_oracle import WRFOracle

    data,cfg=make_horizontal_state(raw,metadata,**case_options)
    if cfg.moist_cq:
        from gpuwm.core.acoustic import prepare_moist_cq
        provider=_device_state(data)
        cqu,cqv,_cqw,use_cq=prepare_moist_cq(provider,cfg)
        if not use_cq:
            raise AssertionError("real moisture state did not take native cq provider")
        # Inputs, not reference answers: use the actual stage provider's
        # face words identically in both the Fortran and CUDA calls.
        data["cqu"],data["cqv"]=cp.asnumpy(cqu),cp.asnumpy(cqv)
    oracle=WRFOracle(library_path,cfg.nx,cfg.ny,cfg.nz,
                     periodic=not cfg.open_x and not cfg.specified)
    uv,mass=_pack_args(oracle,data,cfg,full_theta_probe)
    # WRF diagnoses the fixed reference omega from the real winds. In the
    # actual substep this reference is subtracted after the total recurrence.
    ref={k:(v.copy(order="F") if isinstance(v,np.ndarray) else v) for k,v in mass.items()}
    for key in ("u","v","mu","ww","t","t_1","ft","mu_tend"):
        ref[key].fill(0)
    oracle.call("advance_mu_t",**ref)
    mass["ww_1"][...]=ref["ww"]
    dmdt_ref=oracle.extract(ref["mudf"],data["mu_pp"].shape)
    data["rth_t"] = data["native_ft"] + F(300)*data["c1h"][:,None,None]*dmdt_ref[None]/data["msft"][None]
    if full_theta_probe:
        mass["ft"][...]=oracle.array(data["rth_t"])
    oracle.call("advance_uv",**uv)
    got=horizontal_port_outputs(data,cfg,full_theta_probe=full_theta_probe,
                                no_fma_probe=no_fma_probe)
    # Routine isolation: both mass/theta calls receive the identical words
    # written by the engine momentum launch, so a pressure-gradient rounding
    # difference cannot be misreported as a second mass-equation difference.
    mass["u"]=oracle.array(got["u_pp"])
    mass["v"]=oracle.array(got["v_pp"])
    oracle.call("advance_mu_t",**mass)
    if not np.array_equal(mass["ww_1"].view(np.uint32),ref["ww"].view(np.uint32)):
        raise AssertionError("WRF unexpectedly changed its reference omega input")
    expected=dict(u_pp=oracle.extract(uv["u"],data["u_pp"].shape),
                  v_pp=oracle.extract(uv["v"],data["v_pp"].shape),
                  mu_pp=oracle.extract(mass["mu"],data["mu_pp"].shape),
                  th_pp=oracle.extract(mass["t"],data["th_pp"].shape),
                  ww_pp=oracle.extract(mass["ww"],data["ww_pp"].shape),
                  mudf=oracle.extract(mass["mudf"],data["mu_pp"].shape),
                  t_ave=oracle.extract(mass["t_ave"],data["th_pp"].shape),
                  muave=oracle.extract(mass["muave"],data["mu_pp"].shape),
                  muts=oracle.extract(mass["muts"],data["mu_pp"].shape))
    # These four INTENT(INOUT) arrays are never written by this routine.
    # Preserve them in the compiled fixture instead of silently omitting them.
    for key,shape in (("uam",data["u_pp"].shape),("vam",data["v_pp"].shape),
                      ("wwam",data["ww_pp"].shape)):
        invariant=oracle.extract(mass[key],shape)
        if np.any(invariant.view(np.uint32)):
            raise AssertionError("WRF unexpectedly changed "+key)
    return got,expected,data,cfg


def _device_state(data):
    """Minimal native-driver state retaining the actual scratch interface."""
    import cupy as cp
    class State(SimpleNamespace):
        def scratch(self,shape,slot):
            if slot not in self.buffers:
                self.buffers[slot]=cp.zeros(shape,dtype=F)
            return self.buffers[slot]
    return State(**{k:cp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in data.items()},
                 buffers={},has_msf=bool(np.any(data["msft"]!=F(1))))


def horizontal_port_outputs(data,cfg,*,full_theta_probe=False,no_fma_probe=False,
                            drop_pressure_term=False):
    """Use the forecast's actual explicit launch path, with fixture inputs."""
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep_explicit
    state=_device_state(data)
    mudf=cp.zeros((cfg.ny,cfg.nx),dtype=F)
    def launch():
        cq=(state.cqu,state.cqv,state.p_pp,True) if cfg.moist_cq else (state.p_pp,state.p_pp,state.p_pp,False)
        acoustic_substep_explicit(state,cfg,.25,getattr(cfg,"first",True),
                                 cq=cq,mudf=mudf)
    if no_fma_probe or drop_pressure_term:
        from unittest.mock import patch
        from gpuwm.core.kernels import get_kernel, module_source
        from gpuwm.verify.default_kernel_source import default_source
        # The probe is a default compile: mutate the branch it runs, not an
        # opt-in WRF-exact copy of the same pressure term.
        source=default_source(module_source("acoustic"))
        if drop_pressure_term:
            original="dpxy += rd * dphp *"
            if source.count(original)!=1:
                raise AssertionError("pressure-term mutation site drifted")
            source=source.replace(original,"dpxy += 0.0f * dphp *")
        options=("-std=c++17","--fmad=false") if no_fma_probe else ("-std=c++17",)
        probe=cp.RawModule(code=source,options=options)
        def probe_kernel(module,name):
            return probe.get_function(name) if module=="acoustic" else get_kernel(module,name)
        with patch("gpuwm.core.acoustic.get_kernel",probe_kernel):
            launch()
    else:
        launch()
    cp.cuda.get_current_stream().synchronize()
    got={k:cp.asnumpy(getattr(state,k)) for k in ("u_pp","v_pp","mu_pp","th_pp","ww_pp")}
    got["mudf"]=cp.asnumpy(mudf)
    got["t_ave"]=cp.asnumpy(state.buffers["acoustic_th_pp_old"])
    if not full_theta_probe:
        got["th_pp"] -= F(300)*data["c1h"][:,None,None]*got["mu_pp"][None]
        got["t_ave"] -= F(300)*data["c1h"][:,None,None]*cp.asnumpy(state.buffers["acoustic_mu_pp_old"])[None]
    # WRF leaves the specified frame's t_ave undefined; the wrapper supplies
    # zero there. The GPU's wider history capture is an internal workspace.
    if cfg.specified:
        got["t_ave"][:,0,:]=0; got["t_ave"][:,-1,:]=0
        got["t_ave"][:,:,0]=0; got["t_ave"][:,:,-1]=0
    # Capture the actual fused implicit-solve registers. Before trusting the
    # captures, require the instrumented solve to produce identical native
    # observable words for w, phi, pressure and inverse density.
    from gpuwm.core.acoustic import prepare_acoustic_coefficients
    from gpuwm.verify.smallstep_vertical_oracle import selected_vertical_launch
    coeff=prepare_acoustic_coefficients(state,cfg,.25,
                                       cq=(state.p,state.p,state.p,False))
    fields=("w_pp","ph_pp","p_pp","al_pp")
    before={key:getattr(state,key).copy() for key in fields}
    selected_vertical_launch(state,cfg,.25,coeff,no_fma=no_fma_probe)
    native={key:getattr(state,key).copy() for key in fields}
    for key in fields:
        getattr(state,key)[...]=before[key]
    t2buf=cp.zeros_like(state.th_pp)
    mubuf=cp.zeros((2,cfg.ny,cfg.nx),dtype=F)
    selected_vertical_launch(state,cfg,.25,coeff,no_fma=no_fma_probe,workspace=(t2buf,mubuf))
    for key in fields:
        if not bool(cp.array_equal(getattr(state,key).view(cp.uint32),native[key].view(cp.uint32))):
            raise AssertionError("diagnostic register capture changed "+key)
    got["muts"]=cp.asnumpy(mubuf[0])
    got["muave"]=cp.asnumpy(mubuf[1])
    return got


def measure_horizontal_case(raw,metadata,library_path,**case_options):
    got,want,_data,_cfg=run_horizontal_case(raw,metadata,library_path,**case_options)
    result={}
    for key in got:
        a,b=got[key],want[key]
        d=fp32_ulp_distance(a,b)
        loc=np.unravel_index(int(d.argmax()),d.shape)
        result[key]=dict(max_ulp=int(d.max()),different_words=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
                         words=int(a.size),max_abs=float(np.max(np.abs(a.astype(np.float64)-b))),
                         location=list(map(int,loc)),got=float(a[loc]),wrf=float(b[loc]))
    return result


def write_horizontal_receipt(path,cases,library_path,fixture_path=None):
    """Write measurements only. The parity test pins exact measured values."""
    results={}; fixture={}; descriptions=[]
    for name,raw,meta,opts in cases:
        got,want,data,cfg=run_horizontal_case(raw,meta,library_path,**opts)
        results[name]={}
        for key in got:
            a,b=got[key],want[key]
            d=fp32_ulp_distance(a,b)
            loc=np.unravel_index(int(d.argmax()),d.shape)
            results[name][key]=dict(max_ulp=int(d.max()),different_words=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
                                   words=int(a.size),max_abs=float(np.max(np.abs(a.astype(np.float64)-b))),
                                   location=list(map(int,loc)),got=float(a[loc]),wrf=float(b[loc]))
        if fixture_path is not None and not opts.get("no_fma_probe"):
            prefix=name+"__"
            for key,val in data.items():
                fixture[prefix+"in_"+key]=np.asarray(val,dtype=F)
            for key,val in want.items():
                fixture[prefix+"wrf_"+key]=val
            for key,val in got.items():
                fixture[prefix+"gpu_"+key]=val
            descriptions.append(dict(name=name,cfg=vars(cfg),full_theta_probe=bool(opts.get("full_theta_probe"))))
    Path(path).write_text(json.dumps(results,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if fixture_path is not None:
        fixture_path=Path(fixture_path)
        np.savez_compressed(fixture_path,**fixture)
        fixture_path.with_suffix(".json").write_text(json.dumps(descriptions,indent=2)+"\n",encoding="utf-8")
        files=(fixture_path,fixture_path.with_suffix(".json"),Path(path))
        fixture_path.with_suffix(".sha256.json").write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+"\n",encoding="utf-8")
    return results


def load_horizontal_fixture(path):
    path=Path(path)
    cases=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path,allow_pickle=False) as fixture:
        for case in cases:
            prefix=case["name"]+"__"
            data={key[len(prefix)+3:]:fixture[key].copy() for key in fixture.files if key.startswith(prefix+"in_")}
            for key in ("cf1","cf2","cf3"):
                data[key]=F(data[key])
            expected={key[len(prefix)+4:]:fixture[key].copy() for key in fixture.files if key.startswith(prefix+"wrf_")}
            yield case["name"],data,SimpleNamespace(**case["cfg"]),expected,case["full_theta_probe"]
