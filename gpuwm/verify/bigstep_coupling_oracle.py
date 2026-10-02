"""Word comparisons with compiled WRF big-step coupling routines.

No tolerance is applied here. Full output slices are measured as raw words;
tests separately pin numerical rounding and unused scratch divergences.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpuwm.verify.wrf471_fixtures import require_fixture_dir

# Test data in a source checkout, not package data (gpuwm.verify.wrf471_fixtures).
ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "wrf471_bigstep"
CASE_NAMES = ("real_periodic", "real_boundary", "map_extremes", "steep_terrain", "zero", "near_zero", "mass_perturbation", "cfl_explicit", "cfl_implicit")
SPECIES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")


def coupling_cases(directory=ORACLE_DIR):
    with np.load(require_fixture_dir(directory, "big-step coupling") / "state-real.npz") as data:
        original = {name: data[name].copy() for name in data.files}
    nz, ny, nx = original["T"].shape
    cases = []
    for name in CASE_NAMES:
        state = {key: value.copy() for key, value in original.items()}
        control = dict(periodic=name != "real_boundary", ieva=int(name == "cfl_implicit"), w_damping=1, dt=np.float32(12.), w_crit_cfl=np.float32(2.), dampcoef=np.float32(.2), zdamp=np.float32(5000.), rdx=np.float32(1.) / np.float32(3000.), rdy=np.float32(1.) / np.float32(3000.))
        # No random words: changes remain explicit perturbations of the real
        # WRF column. The extremes probe arithmetic and branch boundaries.
        if name == "map_extremes":
            for key in ("MAPFAC_M", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_U", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_V", "MAPFAC_VX", "MAPFAC_VY"):
                value = state[key]
                x = np.linspace(np.float32(.5), np.float32(2.), value.shape[-1], dtype=np.float32)
                value[:] = x
            state["MF_VX_INV"] = np.float32(1.) / state["MAPFAC_VX"]
        if name == "steep_terrain":
            displacement = np.linspace(np.float32(0.), np.float32(2500.), nx, dtype=np.float32)
            fraction = np.linspace(np.float32(1.), np.float32(0.), nz + 1, dtype=np.float32)
            state["PHB"] += fraction[:, None, None] * displacement[None, None, :] * np.float32(9.81)
        if name in ("zero", "near_zero"):
            magnitude = np.float32(0. if name == "zero" else 1.e-20)
            for key in (*SPECIES, "MU", "U", "V", "W", "PH"):
                state[key][:] = magnitude
        if name == "mass_perturbation":
            state["MU"][:] = np.linspace(np.float32(-137.321), np.float32(87.123), nx, dtype=np.float32)
        mut = state["MU"] + state["MUB"]
        massf = state["C1F"][:, None, None] * mut[None] + state["C2F"][:, None, None]
        wwd = np.zeros_like(state["W"])
        if name.startswith("cfl_"):
            cfl = np.array([0., np.nextafter(np.float32(1.), np.float32(0.)), 1., np.nextafter(np.float32(1.), np.float32(2.)), 1.5, 2., np.nextafter(np.float32(2.), np.float32(3.)), 2.5], np.float32)
            selected = np.resize(cfl, (nz - 1, ny, nx))
            # This construction is itself a pinned input, not an assertion
            # that inversion produces an exact threshold after rounding.
            wwd[1:nz] = selected * massf[1:nz] / (state["RDNW"][1:nz, None, None] * control["dt"])
            state["W"][1:nz] = np.where(np.indices((nz - 1, ny, nx))[2] % 2, np.float32(-3.), np.float32(3.))
        # Rayleigh's reference is a real profile at one retained column.
        fullz = (state["PH"][:, 0, 0] + state["PHB"][:, 0, 0]) / np.float32(9.81)
        bases = dict(ub=np.pad(state["U"][:, 0, 0], (0, 1), mode="edge"), vb=np.pad(state["V"][:, 0, 0], (0, 1), mode="edge"), tb=np.pad(state["T"][:, 0, 0], (0, 1), mode="edge"), zb=np.pad(np.float32(.5) * (fullz[:-1] + fullz[1:]), (0, 1), mode="edge"))
        cases.append(dict(name=name, fields=state, control=control, wwd=wwd, bases=bases))
    return cases


def load_coupling_oracle(directory=ORACLE_DIR):
    with np.load(require_fixture_dir(directory, "big-step coupling") / "coupling.npz") as data:
        return {key: data[key].copy() for key in data.files}


class _LaunchState(SimpleNamespace):
    """Real launch-path field adapter with the normal scratch/total-mu API."""
    def scratch(self, shape, name):
        import cupy as cp
        key = (name, tuple(shape))
        if key not in self._scratch:
            self._scratch[key] = cp.empty(shape, dtype=cp.float32)
        return self._scratch[key]

    def total_mu(self):
        return self.mup + self.mub2d


def coupling_port_outputs(case):
    """Execute the same Python launchers used by the RK stage driver."""
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.acoustic import prepare_moist_cq
    from gpuwm.core.dycore import (stage_fluxes, apply_w_damping,
        record_wrf_vertical_cfl, enable_wrf_cfl_recording,
        reset_wrf_cfl_recording, take_wrf_cfl)
    from gpuwm.core.ieva import stage_face_masses
    fields, control = case["fields"], case["control"]
    nz, ny, nx = fields["T"].shape
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000., dy=3000., ztop=20000., run_seconds=12., dt=float(control["dt"]), mp_physics=8, moist_cq=True, open_x=not control["periodic"], open_y=not control["periodic"], w_damping=control["w_damping"], zadvect_implicit=control["ieva"], w_crit_cfl=float(control["w_crit_cfl"]))
    bindings = dict(u="U", v="V", w="W", p="P", mup="MU", mub2d="MUB", c1h="C1H", c2h="C2H", c1f="C1F", c2f="C2F", dnw="DNW", rdnw="RDNW", msft="MAPFAC_M", msfu="MAPFAC_U", msfv="MAPFAC_V", qv="QVAPOR", qc="QCLOUD", qr="QRAIN", qi="QICE", qs="QSNOW", qg="QGRAUP")
    state = _LaunchState(**{key: cp.asarray(fields[value], dtype=cp.float32) for key, value in bindings.items()}, has_msf=True, _scratch={})
    state.rw_t = cp.zeros_like(state.w)
    mux, muy = stage_face_masses(state, cfg, state.total_mu())
    cqu, cqv, cqw, active = prepare_moist_cq(state, cfg)
    ru, rv, ww = stage_fluxes(state, cfg)
    wwd = cp.asarray(case["wwd"])
    apply_w_damping(state, cfg, wwd)
    reset_wrf_cfl_recording()
    enable_wrf_cfl_recording()
    record_wrf_vertical_cfl(state, cfg, wwd)
    maxv, maxh = take_wrf_cfl(cfg.grid_id)
    reset_wrf_cfl_recording()
    result = {"muu": cp.asnumpy(mux), "muv": cp.asnumpy(muy), "cqu": cp.asnumpy(cqu), "cqv": cp.asnumpy(cqv), "cqwr": cp.asnumpy(cqw), "ww": cp.asnumpy(ww), "rwd": cp.asnumpy(state.rw_t), "maxv": np.asarray(maxv, np.float32), "maxh": np.asarray(maxh, np.float32)}
    # The existing utility explicitly excludes WRF damp_opt=2. Calling that
    # option leaves every tendency zero. Pin the missing path's effect rather
    # than presenting its damp_opt=3 implicit relaxation as this routine.
    from gpuwm.core.diffusion import apply_rayleigh_damping
    apply_rayleigh_damping(state, SimpleNamespace(damp_opt=2))
    for field, shape in (("rayleigh_ru", fields["U"].shape), ("rayleigh_rv", fields["V"].shape), ("rayleigh_rw", fields["W"].shape), ("rayleigh_rt", fields["T"].shape)):
        result[field] = np.zeros(shape, np.float32)
    result.update(coupling_php_consumer(case))
    return result


def coupling_php_consumer(case):
    """Isolate calc_php's fused consumer through the real acoustic kernel.

    The two returned arrays are consuming momentum responses, not a missing
    standalone half-level geopotential array. Fortran's supplementary fixture
    computes that same isolated response from its actual calc_php words.
    """
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    f,c=case["fields"],case["control"]
    nz,ny,nx=f["T"].shape
    z=lambda shape:cp.zeros(shape,dtype=cp.float32)
    g=lambda key:cp.asarray(f[key],dtype=cp.float32)
    upp,vpp=z(f["U"].shape),z(f["V"].shape)
    zero_mass=z((nz,ny,nx))
    zero_full=z((nz+1,ny,nx))
    n=nz*(ny+1)*(nx+1)
    get_kernel("acoustic","advance_uv")(((n+255)//256,),(256,),
        (upp,vpp,z(f["U"].shape),z(f["V"].shape),
         zero_mass,zero_mass,zero_full,g("PH"),g("PHB"),
         g("AL")+g("ALB"),zero_mass,zero_mass,g("MU"),
         cp.ones((ny,nx),dtype=cp.float32),g("MUB"),
         g("C1H"),g("C2H"),g("FNM"),g("FNP"),g("RDNW"),
         zero_mass,zero_mass,np.int32(0),
         np.float32(f["CF1"].item()),np.float32(f["CF2"].item()),np.float32(f["CF3"].item()),np.int32(1),
         c["rdx"],c["rdy"],np.float32(1.),np.float32(0.),
         np.int32(0 if c["periodic"] else 1),np.int32(1),
         np.int32(nz),np.int32(ny),np.int32(nx)))
    return {"php_ru":cp.asnumpy(upp),"php_rv":cp.asnumpy(vpp)}


def coupling_php_order_trace(case):
    """Explanatory operator trace of the documented fused representation."""
    f,c=case["fields"],case["control"]
    p=f["PH"][:-1]+f["PH"][1:]
    b=f["PHB"][:-1]+f["PHB"][1:]
    output={}
    for axis,key,spacing in ((2,"php_ru",c["rdx"]),(1,"php_rv",c["rdy"])):
        count=p.shape[axis]
        right=np.arange(count+1)%count
        left=(np.arange(count+1)-1)%count
        grad=np.float32(.5)*(np.take(p,right,axis=axis)-np.take(p,left,axis=axis))
        grad+=np.float32(.5)*(np.take(b,right,axis=axis)-np.take(b,left,axis=axis))
        value=np.float32(0.)-((spacing*grad)*(-f["C1H"][:,None,None]))
        if not c["periodic"]:
            value[:,0,:]=value[:,-1,:]=value[:,:,0]=value[:,:,-1]=np.float32(0.)
        output[key]=value
    return output


def coupling_wrf_flux_trace(case):
    """Isolate transport coupling from Omega's real production column scan.

    This trace is explanatory, never the oracle. The expected words still
    come only from compiled WRF; supplying its float32 flux ordering to the
    production scan tests whether any remaining difference lies in the scan.
    """
    import cupy as cp
    from gpuwm.core.dycore import _omega_ref
    f,c = case["fields"],case["control"]
    nz,ny,nx = f["T"].shape
    def face_mass(axis):
        count = f["MU"].shape[axis]
        right = np.arange(count+1) % count
        left = (np.arange(count+1)-1) % count
        if not c["periodic"]:
            right=np.minimum(np.arange(count+1),count-1)
            left=np.maximum(np.arange(count+1)-1,0)
        ma,ba = np.take(f["MU"],right,axis=axis),np.take(f["MUB"],right,axis=axis)
        mb,bb = np.take(f["MU"],left,axis=axis),np.take(f["MUB"],left,axis=axis)
        return np.float32(.5) * (((ma+ba)+mb)+bb)
    mx,my = face_mass(1),face_mass(0)
    ru = ((f["C1H"][:,None,None]*mx[None]+f["C2H"][:,None,None])*f["U"]) / f["MAPFAC_UY"][None]
    rv = ((f["C1H"][:,None,None]*my[None]+f["C2H"][:,None,None])*f["V"]) * f["MF_VX_INV"][None]
    state=_LaunchState(p=cp.asarray(f["P"]),dnw=cp.asarray(f["DNW"]),c1h=cp.asarray(f["C1H"]),msft=cp.asarray(f["MAPFAC_MX"]),has_msf=True,_scratch={})
    cfg=SimpleNamespace(dx=3000.,dy=3000.)
    return cp.asnumpy(_omega_ref(state,cfg,cp.asarray(ru),cp.asarray(rv)))


def word_measurement(actual, expected):
    import hashlib
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    a = np.asarray(actual, np.float32)
    b = np.asarray(expected, np.float32)
    changed = a.view(np.uint32) != b.view(np.uint32)
    distance = fp32_ulp_distance(a, b)
    return dict(words=a.size, changed=int(changed.sum()), max_ulp=int(np.max(distance, initial=0)), max_abs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)), initial=0)), actual_sha256=hashlib.sha256(a.tobytes()).hexdigest(), reference_sha256=hashlib.sha256(b.tobytes()).hexdigest())
