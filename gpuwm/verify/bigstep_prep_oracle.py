"""Replay compiled WRF prep fixtures through the production engine launchers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.wrf471_fixtures import require_fixture_dir

# Test data in a source checkout, not package data (gpuwm.verify.wrf471_fixtures).
ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "wrf471_bigstep"
PREP_CASES = ("real-periodic2", "real-specified2", "real-periodic5", "real-specified5",
              "steep-terrain", "map-extremes", "zero-motion", "near-zero-motion")
PHY_MAPPING = {"rho":"rho", "th_phy":"theta", "p_phy":"eos_pressure", "pi_phy":"exner",
               "u_phy":"u", "v_phy":"v", "t_phy":"temperature", "t8w":"t8w",
               "z_at_w":"z_interface", "dz8w":"dz", "p_hyd":"pressure", "p_hyd_w":"p_interface"}


def load_prep_fixture(case):
    directory = require_fixture_dir(ORACLE_DIR, "big-step prep")
    with np.load(directory / f"prep-{case}.npz", allow_pickle=False) as f:
        return dict(f)


def prep_port_outputs(fixture, *, supplied_face_masses=False):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import _launch_slow_geopotential, _launch_slow_geopotential_faces
    from gpuwm.core.physics import _prepare_atmosphere
    from gpuwm.core.rrtmg_legacy import _t8w_columns

    data = {key[6:]:cp.asarray(value) for key,value in fixture.items() if key.startswith("input_")}
    nz,ny,nx = data["T"].shape
    state = SimpleNamespace()
    for attr, field in {"u":"U", "v":"V", "w":"W", "php":"PH", "phb":"PHB", "mup":"MU",
                        "mub2d":"MUB", "alt":"ALT", "qv":"QVAPOR", "qc":"QCLOUD", "qr":"QRAIN",
                        "qi":"QICE", "qs":"QSNOW", "qg":"QGRAUP", "rdnw":"RDNW", "fnm":"FNM",
                        "fnp":"FNP", "c1h":"C1H", "c2h":"C2H", "c1f":"C1F", "c2f":"C2F",
                        "dnw":"DNW", "msft":"MAPFAC_M", "msfu":"MAPFAC_U", "msfv":"MAPFAC_V"}.items():
        setattr(state,attr,data[field])
    state.qh = None
    state.has_msf = True
    state.p = data["P"] + data["PB"]
    state.p_top = np.float32(5000)
    state.cfn,state.cfn1 = np.float32(data["CFN"].get()[0]),np.float32(data["CFN1"].get()[0])
    state.rph_t = cp.zeros_like(data["PH"])
    state.total_mu = lambda: state.mub2d + state.mup
    state.total_theta = lambda: data["T"] + np.float32(300)
    state.scratch = lambda shape, name: cp.zeros(shape, dtype=cp.float32)
    cfg = RunConfig(nx=nx,ny=ny,nz=nz,dx=3000,dy=3000,ztop=20000,dt=12,run_seconds=12,
                    h_sca_adv_order=int(fixture["order"]),specified=bool(fixture["specified"]))
    if supplied_face_masses:
        _launch_slow_geopotential_faces(state,cfg,data["MUU"],data["MUV"])
    else:
        _launch_slow_geopotential(state,cfg,data["WW"],add_vertical=True)
    got = {"ph_tend":state.rph_t.get()}
    atmosphere = _prepare_atmosphere(state)
    atmosphere["eos_pressure"] = state.p
    # This is the existing radiation adapter's interface-temperature path.
    temp = atmosphere["temperature"].get().reshape(nz,-1).T
    z = atmosphere["z_interface"].get().reshape(nz+1,-1).T
    atmosphere["t8w"] = _t8w_columns(temp,z,data["FNM"].get(),data["FNP"].get()).T.reshape(nz+1,ny,nx)
    for output,key in PHY_MAPPING.items():
        a = atmosphere[key]
        got[output] = a.get() if hasattr(a,"get") else a
    return got


def measure_prep(fixture, outputs):
    measured = {}
    nz = fixture["input_T"].shape[0]
    for name,got in outputs.items():
        want = fixture[name]
        if got.shape[0] == nz:
            want = want[:nz]
        distance = fp32_ulp_distance(got,want)
        mismatch = np.ascontiguousarray(got,dtype="f4").view("u4") != np.ascontiguousarray(want,dtype="f4").view("u4")
        measured[name] = {"words":int(got.size),"different_words":int(mismatch.sum()),
                          "max_ulp":int(distance.max()), "max_abs":float(np.nanmax(np.abs(got.astype("f8")-want))),
                          "actual_sha256":hashlib.sha256(np.ascontiguousarray(got,dtype="<f4").tobytes()).hexdigest(),
                          "reference_sha256":hashlib.sha256(np.ascontiguousarray(want,dtype="<f4").tobytes()).hexdigest()}
        if name == "ph_tend":
            measured[name]["total_phi_control_max_ulp"] = int(fp32_ulp_distance(got,fixture["ph_tend_total_phi"]).max())
    return measured
