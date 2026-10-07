"""GSL float32 fire injection and prep measured against pinned Fortran."""
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.verify.chem_oracle import ORACLE_ROOT, load, ulp_table
from gpuwm.verify.chem_fire_ref import add_emiss_burn, smoke_prep
from gpuwm.verify.chem_fire_ref import classify_fire, rules
from gpuwm.verify.chem_fire_ref import column_mass_change, carry_ebu, source_units, frp_diagnostic
from gpuwm.core.chem_fire import select_frames
from datetime import datetime, timedelta, timezone

ROOT = ORACLE_ROOT / "smoke"


def injection_inputs(case):
    volume = {n: case[n].transpose(1,2,0).copy() for n in ("initial","ebu","rho","dz")}
    plane = {n: case[n].T.copy() for n in ("hwp","prev","sw","ends","type")}
    return volume, plane


def prep_inputs(case):
    fields = {n: case[n].transpose(1,2,0).copy() for n in ("t","p","qv","z","z_at_w")}
    fields.update({n: case[n].T[:,None,:].copy() for n in ("u","v")})
    fields.update({n: case[n].reshape(1,-1).copy() for n in
                   ("pbl","oro","u10","v10","t2m","dpt2m","wetness","totprcp",
                    "totprcp_24hrs","swdown","snow")})
    fields["hour"] = case["hour"]
    return fields


@pytest.mark.parametrize("mode", (1,2))
def test_injection_cpu(mode):
    case = load(ROOT / "add_emiss_burn_gsl")[f"mode{mode}"]
    vol, plane = injection_inputs(case)
    chem = vol.pop("initial")
    coef = np.ones((1,20),np.float32); hist = coef.copy()
    for step in range(1,7):
        if step>1:
            vol["ebu"]=carry_ebu(carry,coef,True)
        before=chem.copy()
        chem,coef,hist = add_emiss_burn(chem, **vol, coef=coef,hist=hist,
            hwp=plane["hwp"],prev=plane["prev"],sw=plane["sw"],ends=plane["ends"],
            fire_type=plane["type"],dt=case["dt"],time=(step-1)*13*3600,
            mode=mode,minimum=case["minimum"])
        for name,out in (("chem",chem),("coef",coef),("hist",hist)):
            ref = case[f"{name}{step}"]
            ref = ref.transpose(1,2,0) if name=="chem" else ref.T
            measured=ulp_table(out,ref)
            assert measured=={"max_ulp":0,"n_nonzero":0,"n":out.size}, (name,step,measured)
        emitted=column_mass_change(before,chem,vol["rho"],vol["dz"])
        measured=ulp_table(emitted,case[f"emitted{step}"].T)
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":20},measured
        carry=carry_ebu(vol["ebu"],coef,False)
        measured=ulp_table(carry,case[f"carry{step}"].T[:,None,:])
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":carry.size},measured
        frp=frp_diagnostic(case["frp_mw"].T,coef)
        measured=ulp_table(frp,case[f"frp{step}"][None,:])
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":20},measured


@pytest.mark.parametrize("method", (1,2,3,4))
def test_prep_cpu(method):
    cases=load(ROOT / "smoke_prep_gsl")
    case=cases[f"method{method}_poison1"]
    out=smoke_prep(**prep_inputs(case),method=method)
    for name,value in out.items():
        measured=ulp_table(value,case[name].T)
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":value.size}, (name,measured)
        np.testing.assert_array_equal(case[name],cases[f"method{method}_poison2"][name])
    assert case["raw_kpbl"][1,0] == -777
    assert cases[f"method{method}_poison2"]["raw_kpbl"][1,0] == 777
    assert case["raw_kpbl"][2,0] == 59
    assert case["kpbl"][1,0] == case["kpbl"][2,0] == 58


def test_source_clocks():
    valid=datetime(2026,1,3,4,30,tzinfo=timezone.utc)
    hour=valid.replace(minute=0)
    available=tuple(hour-timedelta(hours=h) for h in range(30))
    with pytest.raises(ValueError,match="retired.*younger than a day"):
        select_frames("persistence_hourly",valid,available)
    assert select_frames("observed_hourly",valid,available)==(hour,)
    daily=select_frames("daily_mean_dcycle",valid,available)
    assert len(daily)==24 and daily[0]==hour-timedelta(hours=25) and daily[-1]==hour-timedelta(hours=2)
    for mode in ("observed_hourly","daily_mean_dcycle"):
        with pytest.raises(FileNotFoundError,match=r"missing hourly source file .*\.nc.*fabricate"):
            select_frames(mode,valid,())
    with pytest.raises(FileNotFoundError,match="202601030400_202601030459.nc"):
        select_frames("observed_hourly",valid,available[1:])
    with pytest.raises(ValueError,match="timezone"):
        select_frames("observed_hourly",valid.replace(tzinfo=None),available)
    with pytest.raises(ValueError,match="source clock"):
        select_frames("unknown",valid,available)


def test_the_trailing_window_ends_at_the_newest_hour_posted_by_the_start():
    from gpuwm.core.chem_fire import trailing_window
    start=datetime(2025,1,8,12,tzinfo=timezone.utc)
    available=tuple(start-timedelta(hours=h) for h in range(40))
    # RAVE's posting latency (7348 s) puts the newest readable hour at 09Z.
    window=trailing_window(start,available,7348.0)
    assert window[-1]==datetime(2025,1,8,9,tzinfo=timezone.utc)
    assert window[0]==datetime(2025,1,7,10,tzinfo=timezone.utc) and len(window)==24
    # Gaps inside the window are gaps; a fire's mean is over its seen hours.
    sparse=tuple(h for h in available if h.hour%3)
    assert all(h in sparse for h in trailing_window(start,sparse,7348.0))
    with pytest.raises(FileNotFoundError,match="no hourly source file is posted"):
        trailing_window(start,(start-timedelta(hours=30),),7348.0)
    with pytest.raises(ValueError,match="timezone"):
        trailing_window(start.replace(tzinfo=None),available,0.0)


def test_the_diurnal_curve_has_mean_one_and_peaks_at_the_band_hour():
    from gpuwm.core.chem_fire import diurnal_factor
    day=datetime(2025,1,8,tzinfo=timezone.utc)
    lon=np.array([-118.5,-105.0,-75.0])      # bands: 23Z, 22Z, 20Z peaks
    values=np.stack([diurnal_factor(np,lon,day+timedelta(minutes=30+60*h)) for h in range(24)])
    assert np.allclose(values.mean(axis=0),1.0,atol=0.01)
    assert list(values.argmax(axis=0))==[22,21,19]   # hour 22 holds 22:30Z, nearest 23:00
    assert values.max()>3.5 and values.min()<0.25


def test_the_retired_timing_is_refused_at_the_door_and_the_default_is_hrrr_style():
    from gpuwm.config import RunConfig, validate_chem_config
    assert RunConfig.__dataclass_fields__["fire_emission_mode"].default=="trailing_24h_dcycle"
    with pytest.raises(ValueError,match="retired.*Eaton"):
        validate_chem_config(_smoke_cfg(fire_emission_mode="persistence_hourly"))


def test_trailing_planes_average_the_hours_a_cell_burned(monkeypatch):
    cp=pytest.importorskip("cupy")
    try:
        cp.zeros(1)
    except Exception as error:
        pytest.skip(f"no device: {error}")
    from types import SimpleNamespace
    from gpuwm.core import chem_fire
    start=datetime(2025,1,8,12,tzinfo=timezone.utc)
    hours=[start-timedelta(hours=h) for h in range(3,30)]
    # cell 0 burns every hour at 2 kg; cell 1 only in the newest hour (a
    # young fire) at 6 kg; cell 2 never.
    def at(source,field,hour):
        young=6.0 if hour==datetime(2025,1,8,9,tzinfo=timezone.utc) else 0.0
        scale=1.0 if field=="PM25" else 10.0
        return np.array([[2.0*scale,young*scale,0.0]],np.float32)
    frames=SimpleNamespace(reference_time=start,available_hours=lambda s,f:frozenset(hours),at=at)
    ctx=SimpleNamespace(state=SimpleNamespace(chem=SimpleNamespace()),
                        met=lambda name:cp.asarray([[-118.5,-118.1,-118.0]],cp.float32))
    source=SimpleNamespace(name="rave-3km",time={"latency_s":7348.0})
    valid=datetime(2025,1,8,22,10,tzinfo=timezone.utc)
    mass,frp,hour=chem_fire.trailing_planes(ctx,frames,source,"PM25","FRP_MEAN",valid)
    factor=float(chem_fire.diurnal_factor(np,np.array([-118.5]),datetime(2025,1,8,22,30,tzinfo=timezone.utc))[0])
    mass=cp.asnumpy(mass)[0]; frp=cp.asnumpy(frp)[0]
    assert hour==datetime(2025,1,8,22,tzinfo=timezone.utc)
    assert np.allclose(mass,[2.0*factor,6.0*factor,0.0],rtol=1e-5)
    assert np.allclose(frp,[20.0*factor,60.0*factor,0.0],rtol=1e-5)


def test_rule_edges():
    data=rules()
    for rule in data["rules"]:
        if "group" not in rule:
            continue
        group=next(g for g in data["groups"] if g["name"]==rule["group"])
        category=group["categories"][0]-1
        threshold=np.float32(rule["gt"])
        for value in (np.nextafter(threshold,np.float32(0)),threshold,
                      np.nextafter(threshold,np.float32(1))):
            frac=np.zeros(20,np.float32);frac[category]=value
            actual=classify_fire(frac,np.float32(0),np.float32(0),1,0,data)
            if value>threshold:
                assert actual==rule["type"]
            else:
                assert actual!=rule["type"]
    region=next(r for r in data["rules"] if "longitude_gt" in r)
    zero=np.zeros(20,np.float32)
    east=np.float32(region["longitude_gt"])
    lo=np.float32(region["latitude_gt"]);hi=np.float32(region["latitude_lt"])
    period=np.float32(data["longitude_period"])
    for lon in (np.nextafter(east,np.float32(0)),east,np.nextafter(east,period)):
        for shifted in (lon, np.float32(lon-period)):
            actual=classify_fire(zero,np.float32((lo+hi)/2),shifted,1,0,data)
            assert (actual==region["type"])==(lon>east)
    for latitude in (np.nextafter(lo,np.float32(0)),lo,np.nextafter(lo,hi),
                     np.nextafter(hi,lo),hi,np.nextafter(hi,np.float32(90))):
        actual=classify_fire(zero,latitude,np.float32(east+1),1,0,data)
        assert (actual==region["type"])==(lo<latitude<hi)
    assert classify_fire(zero,np.float32((lo+hi)/2),np.float32(east+1),np.float32(0),np.float32(1e-6))==0


def test_classification_oracle():
    case=load(ROOT / "fire_rules_gsl")["edges"]
    measured_values=np.array([classify_fire(case["fractions"][i,:,0],case["latitude"][i,0],
        case["longitude"][i,0],case["emission"][i,0],case["minimum"]) for i in range(30)],np.int32)
    measured=ulp_table(measured_values,case["fire_type"][:,0])
    assert measured=={"max_ulp":0,"n_nonzero":0,"n":30},measured


@pytest.mark.gpu
@requires_gpu
def test_classification_kernel():
    import cupy as cp
    from gpuwm.core.kernels import load_module
    from gpuwm.core.chem_fire import _tables
    case=load(ROOT / "fire_rules_gsl")["edges"]
    module=load_module("chem_fire");data=_tables(module)
    frac=cp.asarray(case["fractions"].transpose(1,2,0).copy())
    out=cp.zeros((1,30),cp.int32)
    for shift in (0,-data["longitude_period"]):
        lon=cp.asarray((case["longitude"].T+np.float32(shift)).copy())
        module.get_function("chem_fire_classify")((1,),(128,),
            (frac,cp.asarray(case["latitude"].T.copy()),lon,cp.asarray(case["emission"].T.copy()),
             out,np.int32(30),np.int32(len(data["groups"])),np.int32(len(data["rules"])),
             np.float32(data["longitude_period"]),case["minimum"]))
        measured=ulp_table(cp.asnumpy(out),case["fire_type"].T)
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":30},measured


@pytest.mark.gpu
@requires_gpu
def test_unit_kernel():
    import cupy as cp
    from gpuwm.core.kernels import load_module
    f=np.float32
    kg=np.array([[.125,1,5]],f);mx=np.array([[1,1.2,.8]],f);my=np.array([[1,.9,1.1]],f)
    area=cp.zeros((1,3),cp.float32);flux=cp.zeros_like(area)
    load_module("chem_fire").get_function("chem_fire_units")((1,),(128,),
        (cp.asarray(kg),cp.asarray(mx),cp.asarray(my),flux,area,f(3000),f(3000),np.int32(3)))
    expected_area,expected_flux=source_units(kg,mx,my,f(3000),f(3000))
    for out,ref in ((area,expected_area),(flux,expected_flux)):
        measured=ulp_table(cp.asnumpy(out),ref)
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":3},measured


def test_coupling_and_hwp_refusals():
    from types import SimpleNamespace
    from gpuwm.core.chem_fire import init, _prep
    for option in ("add_fire_moist_flux","add_fire_heat_flux"):
        with pytest.raises(ValueError,match=option+".*water or energy"):
            init(SimpleNamespace(cfg=SimpleNamespace(**{option:True})))
    for method in (1,2,3,4):
        context=SimpleNamespace(cfg=SimpleNamespace(hwp_method=method),has=lambda n:False)
        with pytest.raises(ValueError,match="missing field t_phy.*wildfire potential"):
            _prep(context,0,None)
    short=SimpleNamespace(cfg=SimpleNamespace(),has=lambda n:True,
                          met=lambda n:np.zeros((2,1,1),np.float32),
                          diag={"fire_prev_rain":np.zeros((1,1),np.float32)})
    with pytest.raises(ValueError,match="at least three mass levels.*read outside"):
        _prep(short,0,None,method=0)


def test_a_row_with_two_fire_sources_is_refused(monkeypatch):
    # The row's diurnal carry and fire history are one plane each, so a
    # second fire source on the same row would overwrite the first's.
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import gpuwm.core.kernels as kernels
    from gpuwm.core import chem_fire
    monkeypatch.setattr(kernels, "load_module", lambda name: None)
    fire={"frp":"FRP"}
    source=SimpleNamespace(name="source", variables={"A":{"fire":fire},"B":{"fire":fire},"FRP":{}})
    row=SimpleNamespace(name="smoke", units="ug kg-1", emissions=(
        {"source":"source","field":"A","vertical":"plumerise","weight":1.0},
        {"source":"source","field":"B","vertical":"plumerise","weight":1.0}))
    table=SimpleNamespace(rows_for=lambda key:(row,), sources={"source":source})
    met=np.zeros((3,2,2),np.float32)
    diag={n:np.zeros((1,2,2),np.float32) for n in
          ("fire_hist_carry","fire_coef_carry","fire_cache_hour","fire_cache_ref")}
    frames=SimpleNamespace(reference_time=datetime(2025,1,8,6,tzinfo=timezone.utc),
                           available_hours=lambda source,field: frozenset())
    ctx=SimpleNamespace(table=table,cfg=SimpleNamespace(),frames=frames,diag=diag,
                        clock=SimpleNamespace(curr_secs=0.0),met=lambda name:met)
    with pytest.raises(ValueError,match="2 fire source fields.*its own row"):
        chem_fire.step(ctx,36.0,1)


def test_source_mass_identity():
    # Scalar float32 arithmetic, matching chem_fire_units and injection.
    f=np.float32
    masses=np.array([.125,1.0,5.0],np.float32)
    measured=[]
    for mass,mx,my in zip(masses,[1.0,1.2,.8],[1.0,.9,1.1]):
        area,flux=source_units(np.array([[mass]],f),np.array([[mx]],f),np.array([[my]],f),f(3000),f(3000))
        initial=np.zeros((59,1,1),f);ebu=initial.copy();ebu[0]=flux
        rho=np.full_like(initial,.9);dz=np.full_like(initial,100)
        plane=np.ones((1,1),f)
        chemistry,_,_=add_emiss_burn(initial,ebu,rho,dz,plane,plane,plane,plane,plane,
            plane,np.zeros((1,1),np.int32),f(3600),f(0),1,f(1e-6))
        recovered=f(column_mass_change(initial,chemistry,rho,dz)[0,0]*area[0,0])
        expected=f(mass*f(1e9))
        measured.append(ulp_table(recovered,expected))
    assert measured==[{"max_ulp":1,"n_nonzero":1,"n":1},
                      {"max_ulp":0,"n_nonzero":0,"n":1},
                      {"max_ulp":0,"n_nonzero":0,"n":1}], measured


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("method", (1,2,3,4))
def test_prep_kernel(method):
    import cupy as cp
    from gpuwm.core.kernels import load_module
    case=load(ROOT / "smoke_prep_gsl")[f"method{method}_poison1"]
    fields={n:cp.asarray(v) for n,v in prep_inputs(case).items() if n!="hour"}
    names=("kpbl","kpbl_thetav","uspdavg2d","windgustpot","hpbl2d","hwp")
    out={n:cp.zeros((1,12),cp.int32 if n.startswith("kpbl") else cp.float32) for n in names}
    order=("t","p","qv","u","v","z","z_at_w","pbl","oro","u10","v10",
           "t2m","dpt2m","wetness","totprcp","totprcp_24hrs","swdown","snow")
    load_module("chem_fire").get_function("chem_fire_prep")((1,),(128,),
        (*(fields[n] for n in order),*(out[n] for n in names),np.int32(59),np.int32(12),case["hour"],np.int32(method)))
    for name,value in out.items():
        measured=ulp_table(cp.asnumpy(value),case[name].T)
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":12},(name,measured)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mode", (1,2))
def test_injection_kernel(mode):
    import cupy as cp
    from gpuwm.core.kernels import load_module
    from gpuwm.core.chem_fire import _tables
    case=load(ROOT / "add_emiss_burn_gsl")[f"mode{mode}"]
    vol,plane=injection_inputs(case)
    vol={n:cp.asarray(v) for n,v in vol.items()}
    plane={n:cp.asarray(v) for n,v in plane.items()}
    module=load_module("chem_fire")
    data=_tables(module)
    coef=cp.ones((1,20),cp.float32);hist=coef.copy();removed=cp.zeros_like(coef)
    for step in range(1,7):
        removed.fill(0)
        if step>1:
            module.get_function("chem_fire_carry")((1,),(128,),
                (vol["ebu"],coef,np.int32(20),np.int32(59),np.int32(1)))
        module.get_function("chem_fire_cycle")((1,),(128,),
            (coef,hist,plane["hwp"],plane["prev"],plane["sw"],plane["ends"],plane["type"],
             np.int32(20),np.int32(len(data["night_decay"])),
             np.float32((step-1)*13*3600),np.float32(1),np.int32(mode)))
        module.get_function("chem_fire_inject")((1,),(128,),
            (vol["initial"],vol["ebu"],vol["rho"],vol["rho"],vol["dz"],coef,removed,
             np.int32(20),np.int32(59),np.int32(mode),case["dt"],case["minimum"],np.float32(1),np.int32(0),np.float32(1)))
        for name,out in (("chem",vol["initial"]),("coef",coef),("hist",hist)):
            ref=case[f"{name}{step}"]
            ref=ref.transpose(1,2,0) if name=="chem" else ref.T
            measured=ulp_table(cp.asnumpy(out),ref)
            assert measured=={"max_ulp":0,"n_nonzero":0,"n":ref.size},(name,step,measured)
        measured=ulp_table(cp.asnumpy(removed),case[f"emitted{step}"].T)
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":20},measured
        module.get_function("chem_fire_carry")((1,),(128,),
            (vol["ebu"],coef,np.int32(20),np.int32(59),np.int32(0)))
        measured=ulp_table(cp.asnumpy(vol["ebu"]),case[f"carry{step}"].T[:,None,:])
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":vol["ebu"].size},measured
        frp=cp.zeros_like(coef)
        module.get_function("chem_fire_frp_diag")((1,),(128,),
            (cp.asarray(case["frp_mw"].T.copy()),coef,frp,np.int32(20)))
        measured=ulp_table(cp.asnumpy(frp),case[f"frp{step}"][None,:])
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":20},measured


def test_daily_reducer_matches_the_srw_preprocessing_rules():
    """The Rust previous-day reduction (chem_fire_reduce.rs): emitting-hour
    mean with a two-hour floor on the count, latest fire age, finite-HWP
    mean and positive-rain sum (ufs-srweather-app 2ad2bc57 cycle.py:252-307,
    330-376), and the empty-history refusal."""
    import ctypes
    from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
    try:
        library = ctypes.CDLL(str(resolve_cpu_bridge()))
    except (OSError, RuntimeError, ValueError) as error:
        pytest.skip(f"CPU preprocessing bridge not loadable: {error}")
    if not hasattr(library, "gpuwm_fire_daily"):
        pytest.skip("this CPU preprocessing bridge predates the daily fire "
                    "reduction; rebuild tools/grib1_bridge to run it")
    from gpuwm.chem_fire_reduce import daily_inputs

    mass=np.zeros((24,1,4),np.float32)
    frp=np.zeros_like(mass)
    mass[0,0,0]=2;frp[0,0,0]=6
    mass[1:3,0,1]=[3,5];frp[1:3,0,1]=[4,8]
    mass[:,0,2]=1;frp[:,0,2]=2
    hwp=np.array([[[10,20,np.nan,0]],[[30,40,50,0]]],np.float32)
    rain=np.array([[[.01,.02,.03,.04]],[[.01,-1,.03,.04]]],np.float32)
    ages=np.arange(25,1,-1,dtype=np.float32)
    out=daily_inputs(mass,frp,ages,hwp,rain)
    expected=(np.array([[1,4,1,0]],np.float32),np.array([[3,6,2,0]],np.float32),
              np.array([[25,23,2,0]],np.float32),np.array([[20,30,50,0]],np.float32),
              np.array([[.02,.02,.06,0]],np.float32))
    for actual,reference in zip(out,expected):
        np.testing.assert_array_equal(actual,reference)
    try:
        daily_inputs(mass,frp,ages,hwp[:0],rain[:0])
    except ValueError as error:
        assert "prior HWP" in str(error)
    else:
        raise AssertionError("empty prior HWP must fail")


def _smoke_cfg(**overrides):
    from gpuwm.config import RunConfig
    values = dict(nx=12, ny=10, nz=6, dx=3000.0, dy=3000.0, ztop=12000.0,
                  dt=18.0, run_seconds=60.0, bl_pbl_physics=1,
                  chem_sets="smoke", chem_sources="rave-3km")
    values.update(overrides)
    return RunConfig(**values)


def test_the_shipped_smoke_configuration_is_admitted():
    from gpuwm.config import validate_chem_config
    validate_chem_config(_smoke_cfg())
    validate_chem_config(_smoke_cfg(fire_emission_mode="observed_hourly"))


@pytest.mark.parametrize("overrides, match", [
    # The fire source left out of chem_sources: the frames refuse it at the
    # first chem step, so the door refuses it before a preparation.
    (dict(chem_sources=""), "enables none of them"),
    # ebb_dcycle = 2 needs yesterday's model HWP, which no source row names.
    (dict(fire_emission_mode="daily_mean_dcycle"), "wildfire potential and rain"),
    # WRF-Chem's landuse arm needs mean_fct/firesize, which no row supplies.
    (dict(plume_fire_properties="landuse"), "mean_fct, firesize"),
    # GSL's emitted-number feedback to Thompson is not built: admitted, it
    # was a silent no-op under a flag that says it couples.
    (dict(mp_physics=28, aerosol_mp_coupling="emission"), "not in this build"),
])
def test_smoke_configurations_that_cannot_emit_are_refused_at_the_door(overrides, match):
    from gpuwm.config import validate_chem_config
    with pytest.raises(ValueError, match=match):
        validate_chem_config(_smoke_cfg(**overrides))


def test_the_fire_state_starts_on_a_rebuilt_nest_and_not_on_a_restore():
    # A nest moved in mid-run is built afresh with zero process arrays while
    # ktau is past 1; a zero diurnal coefficient would scale its carried
    # emission to nothing.  A restored checkpoint keeps what it carried.
    from types import SimpleNamespace
    from gpuwm.core.chem_fire import start_carry
    def ctx(hour):
        diag={n:np.zeros((1,2,2),np.float32) for n in ("fire_hist_carry","fire_coef_carry")}
        diag.update({n:np.full((1,2,2),hour,np.int32) for n in ("fire_cache_hour","fire_cache_ref")})
        return SimpleNamespace(state=SimpleNamespace(chem=SimpleNamespace()),diag=diag)
    moved=ctx(0)
    assert start_carry(moved,ktau=97) and moved.diag["fire_coef_carry"].min()==1
    assert moved.diag["fire_cache_hour"].max()==-1 and moved.state.chem.fire_carry_ready
    restored=ctx(482_000)
    restored.diag["fire_coef_carry"][...]=0.25
    assert not start_carry(restored,ktau=97) and restored.diag["fire_coef_carry"].max()==np.float32(0.25)
    first=ctx(0)
    assert start_carry(first,ktau=1) and first.diag["fire_hist_carry"].min()==1
    # Once a state is started, later steps read nothing from the device.
    first.diag["fire_cache_hour"][...]=0
    assert not start_carry(first,ktau=2)


def test_fresh_strip_carry_initializes_without_resetting_written_overlap():
    from types import SimpleNamespace
    from gpuwm.core.chem_fire import start_carry

    hour = np.array([[[482_000, 482_000, 0], [482_000, 482_000, 0]]], np.int32)
    diag = {"fire_cache_hour": hour,
            "fire_cache_ref": np.array([[[3, 3, 0], [3, 3, 0]]], np.int32),
            "fire_hist_carry": np.full((1, 2, 3), 0.5, np.float32),
            "fire_coef_carry": np.full((1, 2, 3), 0.25, np.float32)}
    ctx = SimpleNamespace(state=SimpleNamespace(chem=SimpleNamespace()), diag=diag)
    assert start_carry(ctx, ktau=97)
    np.testing.assert_array_equal(diag["fire_cache_hour"],
                                  [[[482_000, 482_000, -1], [482_000, 482_000, -1]]])
    np.testing.assert_array_equal(diag["fire_cache_ref"], [[[3, 3, -1], [3, 3, -1]]])
    np.testing.assert_array_equal(diag["fire_hist_carry"], [[[0.5, 0.5, 1], [0.5, 0.5, 1]]])
    np.testing.assert_array_equal(diag["fire_coef_carry"], [[[0.25, 0.25, 1], [0.25, 0.25, 1]]])


def test_a_rebuilt_nest_lifts_its_fires_at_once_and_a_restore_waits():
    # The plume bounds of a nest rebuilt in mid-run are zeros past ktau 2;
    # its first chem step runs the plume instead of waiting for the next
    # plumerisefire_frq boundary.  A restored or ongoing cache is >= 2.
    from gpuwm.core.chem_plumerise import rebuilt_mid_run
    zeros=np.zeros((2,3),np.int32); ongoing=np.full((2,3),2,np.int32)
    assert rebuilt_mid_run(zeros,ktau=200)
    assert not rebuilt_mid_run(zeros,ktau=1) and not rebuilt_mid_run(zeros,ktau=2)
    assert not rebuilt_mid_run(ongoing,ktau=200)
