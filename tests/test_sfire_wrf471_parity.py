"""SFIRE kernels replay binary32 output from byte-unmodified compiled WRF."""
from pathlib import Path
import json
import numpy as np
import pytest
from conftest import requires_gpu
from tools.sfire_wrf471_oracle.fixture import ROOT, load, words


def _check(actual, expected, name, *, max_ulp=0):
    import cupy as cp
    result = words(cp.asnumpy(actual), expected)
    print(json.dumps({name: result}))
    assert result["nonfinite_differences"] == 0, (name, result)
    assert result["max_ulp"] <= max_ulp, (name, result)
    if max_ulp == 0:
        assert result["different_words"] == 0, (name, result)


@requires_gpu
def test_native_fuel_parameters_every_category():
    from gpuwm.core.sfire_phys import FuelTable, PARAM_FIELDS, set_fire_params
    case = load("params")
    params = set_fire_params(case["nfuel_cat_in"], case["fmc_g_in"], FuelTable(), fire_fmc_read=0)
    for name in (*PARAM_FIELDS, "fmc_g"):
        _check(params[name], case[name + "_out"], "params/" + name)


@requires_gpu
@pytest.mark.parametrize("advection", [0, 1])
def test_native_rothermel(advection):
    import cupy as cp
    from gpuwm.core.sfire_phys import fire_ros
    p, case = load("params"), load(f"ros_advection{advection}")
    params = {key: cp.asarray(p[key + "_out"]) for key in ("bbb", "betafl", "phiwc", "r_0", "ischap")}
    actual = fire_ros(*[case[key + "_in"] for key in ("propx", "propy", "vx", "vy", "dzdxf", "dzdyf")],
                      params, fire_advection=advection)
    for key, array in zip(("ros_base", "ros_wind", "ros_slope"), actual):
        _check(array, case[key + "_out"], f"ros_advection{advection}/" + key)


@requires_gpu
def test_native_heat_fluxes():
    from gpuwm.core.sfire_phys import heat_fluxes
    case, params = load("flux"), load("params")
    hfx, qfx = heat_fluxes(float(case["dt"]), params["fgip_out"], case["fuel_frac_burnt_in"], params["fmc_g_out"])
    _check(hfx, case["grnhft_out"], "flux/sensible")
    _check(qfx, case["grnqft_out"], "flux/latent")


@requires_gpu
def test_native_five_class_moisture_carried_steps():
    import cupy as cp
    from gpuwm.core.sfire_moisture import MoistureState, STATE_FIELDS, advance_moisture
    first = load("moisture_s0")
    state = MoistureState.zeros(first["rainc_in"].shape)
    for key in STATE_FIELDS:
        if key + "_in" in first:
            getattr(state, key)[...] = cp.asarray(first[key + "_in"])
    for step in range(3):
        case = load(f"moisture_s{step}")
        advance_moisture(state, float(case["dt"]), *[case[key + "_in"] for key in ("rainc", "rainnc", "t2", "q2", "psfc")],
                         initialize=step == 0, fmep_decay_tlag=float(case["fmep_decay_tlag"]))
        for key in STATE_FIELDS:
            _check(getattr(state, key), case[key + "_out"], f"moisture_s{step}/" + key)


@requires_gpu
def test_native_weighted_fire_grid_moisture():
    from gpuwm.core.sfire_phys import weighted_moisture
    case = load("moisture_interp")
    actual = weighted_moisture(case["classes_fire_out"],case["nfuel_cat_in"])
    _check(actual[1:-1,1:-1],case["fmc_g_out"][1:-1,1:-1],"moisture_interp/weighted")


@requires_gpu
def test_native_cell_fuel_integral():
    from gpuwm.core.sfire_core import fuel_left_cell_1
    case = load("fuel_cell")
    corners = ("00", "01", "10", "11")
    level = np.stack([case["lfn" + key + "_in"] for key in corners], axis=-1)
    ignition = np.stack([case["tign" + key + "_in"] for key in corners], axis=-1)
    fuel, area = fuel_left_cell_1(level, ignition, float(case["time_now"]), float(case["fuel_time"]))
    _check(fuel, case["fuel_frac_out"], "fuel_cell/fuel")
    _check(area, case["fire_area_out"], "fuel_cell/area")


@requires_gpu
@pytest.mark.parametrize("method,key", [(1,"upwind"),(2,"godunov"),(3,"eno")])
def test_native_sided_selectors(method, key):
    from gpuwm.core.sfire_core import select_sided
    case = load("selectors")
    out = select_sided(case["left_in"],case["right_in"],method=method)
    _check(out,case[key+"_out"],"selectors/"+key)


def _core_case(name):
    case = load(name)
    bounds = case["bounds"]
    domain = tuple(int(bounds[i] - bounds[4 + (i//2)*2]) for i in range(4))
    xlo,xhi,ylo,yhi = domain
    selection = np.s_[ylo:yhi+1,xlo:xhi+1]
    fields = ("vx", "vy", "dzdxf", "dzdyf", "bbb", "betafl", "phiwc", "r_0", "ischap")
    params = {key: case[key + "_in"] for key in fields}
    return case, params, domain, selection


@requires_gpu
@pytest.mark.parametrize("method", [1,2,3,5,6,7,8,9])
def test_native_level_set_tendency(method):
    from gpuwm.core.sfire_core import CoreOptions, tend_ls
    case, params, domain, selection = _core_case(f"tend_mode{method}")
    tend,ros,bound = tend_ls(case["lfn_in"], params, float(case["dx"]),float(case["dy"]),
                            options=CoreOptions(upwinding=method),domain=domain)
    _check(tend[selection],case["tend_out"][selection],f"tend_mode{method}/tend")
    _check(ros[selection],case["ros_out"][selection],f"tend_mode{method}/ros")
    _check(bound,case["tbound_out"],f"tend_mode{method}/bound")


@requires_gpu
def test_native_three_stage_level_set():
    from gpuwm.core.sfire_core import prop_ls_rk3
    case, params, domain, selection = _core_case("rk3")
    out,ros,bound = prop_ls_rk3(case["lfn_in"],params,float(case["dx"]),float(case["dy"]),
                               float(case["ts"]),float(case["dt"]),domain=domain)
    _check(out[selection],case["lfn_out"][selection],"rk3/lfn")
    _check(ros[selection],case["ros_out"][selection],"rk3/ros")
    _check(bound,case["tbound_out"],"rk3/bound")


@requires_gpu
@pytest.mark.parametrize("method", [1,2,3,4])
def test_native_three_stage_reinitialization(method):
    from gpuwm.core.sfire_core import CoreOptions, reinit_ls_rk3
    case, params, domain, selection = _core_case(f"reinit_mode{method}")
    out = reinit_ls_rk3(case["lfn_in"],case["lfn_2_in"],float(case["dx"]),float(case["dy"]),
                        options=CoreOptions(upwinding_reinit=method),domain=domain)
    _check(out[selection],case["lfn_out"][selection],f"reinit_mode{method}/lfn")


@requires_gpu
def test_native_refined_fuel_grid():
    from gpuwm.core.sfire_core import fuel_left
    case, params, domain, selection = _core_case("fuel_grid")
    fuel,area = fuel_left(case["lfn_in"],case["tign_in"],case["fuel_time_in"],float(case["ts"]),domain=domain)
    _check(fuel[selection],case["fuel_frac_out"][selection],"fuel_grid/fuel")
    _check(area[selection],case["fire_area_out"][selection],"fuel_grid/area")


@requires_gpu
def test_native_no_fire_initialization():
    from gpuwm.core.sfire_core import init_no_fire
    case = load("no_fire")
    state = init_no_fire(case["lfn_out"].shape,5.,7.,0.)
    for key in state:
        _check(state[key][1:-1,1:-1],case[key+"_out"][1:-1,1:-1],"no_fire/"+key)


def _line(case):
    from gpuwm.core.sfire_core import IgnitionLine
    return IgnitionLine(*map(float,case["line"]))


@requires_gpu
def test_native_nearest_ignition_segment():
    from gpuwm.core.sfire_core import nearest
    case = load("nearest")
    d,t = nearest(case["ax_in"],case["ay_in"],_line(case),unit_x=1.,unit_y=2.)
    _check(d,case["distance_out"],"nearest/distance")
    _check(t,case["time_out"],"nearest/time")


@requires_gpu
def test_native_time_dependent_ignition_line():
    import cupy as cp
    from gpuwm.core.sfire_core import ignite_fire
    case = load("ignite_line")
    lfn,tign = cp.asarray(case["lfn_in"]),cp.asarray(case["tign_in"])
    count = ignite_fire(lfn,tign,case["coord_x_in"],case["coord_y_in"],_line(case),0.,12.)
    _check(lfn,case["lfn_out"],"ignite_line/lfn")
    _check(tign,case["tign_out"],"ignite_line/tign")
    assert count == int(case["ignited_out"])


@requires_gpu
def test_native_ignition_time_crossing():
    import cupy as cp
    from gpuwm.core.sfire_core import tign_update
    case = load("ignition_time")
    tign = cp.asarray(case["tign_in"])
    tign_update(case["lfn_in"],case["lfn_after_in"],tign,12.,1.5)
    _check(tign,case["tign_out"],"ignition_time/tign")


@requires_gpu
def test_native_flame_diagnostics():
    from gpuwm.core.sfire_core import calc_flame_length
    case = load("flame")
    length,front,intensity = calc_flame_length(case["ros_in"],case["iboros_in"],case["fire_area_in"])
    _check(length[1:-1,1:-1],case["flame_length_out"][1:-1,1:-1],"flame/length")
    _check(front[1:-1,1:-1],case["ros_fl_out"][1:-1,1:-1],"flame/front")


@requires_gpu
def test_corrected_extinction_removes_original_negative_spread():
    import cupy as cp
    from gpuwm.core.sfire_phys import set_fire_params
    case = load("original_moisture_extinction_defect")
    assert np.all(case["r_0_out"] < 0) and np.all(case["iboros_out"] < 0)
    params = set_fire_params(case["nfuel_cat_in"],case["fmc_g_in"],fire_fmc_read=0)
    assert bool(cp.all(params["r_0"] == 0)) and bool(cp.all(params["iboros"] == 0))


@requires_gpu
def test_corrected_ros_cap_removes_original_negative_slope():
    import cupy as cp
    from gpuwm.core.sfire_phys import fire_ros
    p,case = load("params"),load("original_ros_cap_defect")
    assert np.any(case["ros_slope_out"] < 0)
    params = {key:cp.asarray(p[key+"_out"]) for key in ("bbb","betafl","phiwc","r_0","ischap")}
    base,wind,slope = fire_ros(*[case[key+"_in"] for key in ("propx","propy","vx","vy","dzdxf","dzdyf")],params)
    assert bool(cp.all(base >= 0)) and bool(cp.all(wind >= 0)) and bool(cp.all(slope >= 0))
    total=cp.asnumpy(base+wind+slope)
    print(json.dumps({"corrected_ros_cap":{"maximum":float(total.max()),
        "overshoot_words":int(np.count_nonzero(total>np.float32(6))),
        "max_ulp_above_six":int(np.max(total.view(np.uint32)-np.float32(6).view(np.uint32),where=total>np.float32(6),initial=0))}}))
    # Three binary32 contributions sum with two rounding operations.
    assert bool(cp.all(base+wind+slope <= np.nextafter(np.float32(6),np.float32(np.inf))))


@requires_gpu
def test_explicit_corrected_tg_fortran_control():
    from gpuwm.core.sfire_atm import truncated_gaussian, fire_tendency
    case = load("corrected/corrected_tg")
    prop = truncated_gaussian(case["dz8w"][:-1],case["z_at_w"],case["terrain"],
        peak=32.,upper=1000.,extinction=60.,active_levels=6,domain=(0,5,0,4))
    _check(prop,case["prop_heat_out"][:-1],"corrected_tg/prop")
    theta,water = fire_tendency(*[case[key] for key in ("grnhfx","grnqfx","canhfx","canqfx")],
        terrain=case["terrain"],z_at_w=case["z_at_w"],dz8w=case["dz8w"][:-1],mu=case["mu"],
        c1h=case["c1h"][:-1],c2h=case["c2h"][:-1],rho=case["rho"][:-1],
        fire_ext_grnd=60.,fire_ext_crwn=90.,crown_height=20.,fire_sfc_flx=1,
        fire_heat_peak=32.,fire_tg_ub=1000.)
    _check(theta,case["rthfrten_out"][:-1],"corrected_tg/theta")
    _check(water,case["rqvfrten_out"][:-1],"corrected_tg/water")
