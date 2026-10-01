"""CPU-only bitwise replay of all WRF v4.7.1 single-layer UCM fixtures."""
import csv
import gzip
from pathlib import Path

import numpy as np
import pytest
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.urban_ucm_ref import (
    F, FORCING_FIELDS, STATE_FIELDS, LAYER_FIELDS, OUTPUT_FIELDS, urban_step,
)

from gpuwm.verify.urban_ucm_oracle import UCM_VARIANTS

ROOT = Path(__file__).resolve().parents[1] / 'gpuwm/data/urban/oracle/ucm'
# The twelve column-oracle variants; the Noah and Noah-MP coupling fixtures
# beside them have their own replay at the end of this file.
FIXTURES = [ROOT / f'{name}.csv.gz' for name in UCM_VARIANTS]


def read_csv(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def load_fixture(path):
    name = path.name.removesuffix('.csv.gz')
    tables = {int(r['utype']): {k:F(float(v)) for k,v in r.items()}
              for r in read_csv(ROOT / (name + '-table.csv'))}
    switches = {r['name']:F(float(r['value'])) for r in read_csv(ROOT / (name + '-switches.csv'))}
    rows = [{k:(v if k == 'variant' else F(float(v))) for k,v in r.items()}
            for r in read_csv(path)]
    return tables, switches, rows


def initial_state(row):
    return {k: (np.array([row[f'{k}{i}_in'] for i in range(1,5)],dtype=np.float32)
                if k in LAYER_FIELDS else row[k+'_in']) for k in STATE_FIELDS}


def forcing(row):
    return {k:row['znt_in' if k == 'znt' else k] for k in FORCING_FIELDS}


def flattened(state):
    return {name:value for k,v in state.items()
            for name,value in ([(f'{k}{i}',x) for i,x in enumerate(v,1)]
                               if k in LAYER_FIELDS else [(k,v)])}


@pytest.mark.parametrize('path', FIXTURES, ids=lambda p:p.name)
@pytest.mark.parametrize('carry', [False, True], ids=['independent', 'carried'])
def test_oracle(path, carry):
    assert len(FIXTURES) == 12
    tables,switches,rows = load_fixture(path)
    states, maxima, failures = {}, {}, []
    for row in rows:
        key = (int(row['utype']),int(row['scenario']))
        if not carry or key not in states:
            states[key] = initial_state(row)
        state = states[key]
        out = urban_step(tables[key[0]],switches,forcing(row),state,jmonth=int(row['jmonth']))
        for field,value in {**out,**flattened(state)}.items():
            want=row[field]
            distance=int(fp32_ulp_distance([value],[want])[0])
            maxima[field]=max(maxima.get(field,0),distance)
            if F(value).view(np.uint32) != want.view(np.uint32):
                if len(failures)<20:
                    failures.append(f'utype={key[0]} scenario={key[1]} step={int(row["step"])} {field}: got {float(value):.9g} 0x{int(F(value).view(np.uint32)):08x}, want {float(want):.9g} 0x{int(want.view(np.uint32)):08x}, ULP={distance}')
    if failures:
        table='\n'.join(f'{k:18s} {maxima[k]}' for k in sorted(maxima))
        pytest.fail('Per-field max ULP:\n'+table+'\nFirst mismatches:\n'+'\n'.join(failures))


from gpuwm.verify.urban_ucm_ref import (
    noah_blend, noahmp_blend, noah_overrides, noahmp_overrides, RCP, _powi,
)


def driver_sample():
    # Binary fractions give hand-computed, exact blends.
    return dict(ts=F(304),qs=F(0.25),sh=F(16),lh=F(32),
                lh_kinematic=F(0.125),alb=F(0.5),g=F(-8),ust=F(0.5),
                gz1oz0=F(4),psim=F(2),psih=F(1),u10=F(3),v10=F(-4),
                th2=F(300),q2=F(0.25),znt=F(0.1),tr=F(303),cmr=F(0.01))


def common_inputs():
    return dict(frc_urb=F(0.25),ust=F(0.25),u1=F(0.3),v1=F(0.4),
                glw=F(400),rainbl=F(0.125),dt=F(60),declin=F(0.3),
                cosz=F(0.8),omg=F(0),xlat=F(40),znt=F(0.01),
                chs=F(0.001),chs2=F(0.03),cqs2=F(0.005))


def bits_equal(got,want):
    assert F(got).view(np.uint32)==F(want).view(np.uint32)


def test_noah_driver_blend_and_inputs():
    u=driver_sample()
    result=noah_blend(u,t1=F(296),sheat=F(8),eta_kinematic=F(0.0625),
        eta=F(16),ssoil=F(-4),albedok=F(0.25),q1=F(0.125),
        sfctmp=F(300),q2k=F(0.01),sfcprs=F(100000),zlvl=F(30),
        soldn=F(800),emissi=F(0.95),**common_inputs())
    expected=dict(albedo=0.3125,hfx=10,qfx=0.078125,lh=20,grdflx=-5,
                  tsk=298,q1=0.15625,ust=0.3125,ua_urb=1,
                  chs=0.01,chs2=0.03,cqs2=0.01,ssgd_urb=640,
                  ssgq_urb=160,akms_urb2d=0.1,tr_urb2d=303)
    for k,v in expected.items():bits_equal(result[k],F(v))
    bits_equal(result['rain_urb'],np.uint32(0x40f00001).view(np.float32))
    bits_equal(result['qsfc'],F(F(5)/F(27)))
    # Density operations at the call site, including the order of products.
    bits_equal(result['rhoo_urb'],F(F(100000)/F(F(F(287.04)*F(300))*F(F(1)+F(F(0.61)*F(0.01))))))
    assert u==driver_sample()  # Pure functions do not overwrite urban outputs.


def test_noahmp_driver_uses_cqs2_pressure_and_direct_qsfc():
    result=noahmp_blend(driver_sample(),albedo=F(0.25),hfx=F(8),qfx=F(0.0625),
        lh=F(16),grdflx=F(4),tsk=F(296),qsfc=F(0.125),t3d=F(300),
        qv=F(0.25),swdown=F(800),p8w_lower=F(100000),
        p8w_upper=F(90000),dz8w=F(60),**common_inputs())
    for k,v in dict(chs2=0.01,chs2_urb=0.01,qa_urb=0.2,za_urb=30,
                    grdflx=5,qsfc=0.15625,tsk=298,akms_urb2d=0.1).items():
        bits_equal(result[k],F(v))
    bits_equal(result['rhoo_urb'],F(F(95000)/F(F(F(287.04)*F(300))*F(F(1)+F(F(0.61)*F(0.2))))))


def test_noah_surface_overrides():
    got=noah_overrides(driver_sample(),chs=F(0.03),akms_urb2d=F(0.07))
    expected=dict(u10=3,v10=-4,psim=2,psih=1,gz1oz0=4,akhs=0.03,akms=0.07)
    assert set(got)==set(expected)
    for k,v in expected.items():bits_equal(got[k],F(v))


def test_noahmp_surface_overrides():
    u=driver_sample()
    got=noahmp_overrides(u,frc_urb=F(0.25),fvegxy=F(0.5),
        t2mvxy=F(296),t2mbxy=F(288),q2mvxy=F(0.125),q2mbxy=F(0.0625),
        psfc=F(100000),chs=F(0.03),akms_urb2d=F(0.07))
    # At 1000 hPa the pressure factor is exactly one.
    for k,v in dict(q2=0.1328125,t2=294,th2=294,akhs=0.03,akms=0.07).items():
        bits_equal(got[k],F(v))
    assert int(RCP.view(np.uint32))==0x3e924925
    # Non-unit pressure factor: hand-computed with the oracle libm, rather than
    # assuming double-precision exponentiation can stand in for powf.
    from gpuwm.core.noahmp_libm import powf
    e=F(powf(F(F(1e5)/F(95000)),F(2/7)))
    got=noahmp_overrides(u,frc_urb=F(0.25),fvegxy=F(0.5),
        t2mvxy=F(296),t2mbxy=F(288),q2mvxy=F(0.125),q2mbxy=F(0.0625),
        psfc=F(95000),chs=F(0.03),akms_urb2d=F(0.07))
    t2=F(F(219)+F(F(F(300)/e)*F(0.25)))
    bits_equal(got['t2'],t2);bits_equal(got['th2'],F(t2*e))


def test_refused_morphology_and_fatal_geometry():
    tables,switches,rows=load_fixture(FIXTURES[0]);r=rows[0]
    table=tables[int(r['utype'])];f=forcing(r);s=initial_state(r)
    with pytest.raises(NotImplementedError,match='distributed_aerodynamics_option'):
        urban_step(table,{**switches,'distributed_aerodynamics_option':True},f,s,jmonth=7)
    with pytest.raises(NotImplementedError,match='NUDAPT'):
        urban_step(table,switches,{**f,'mh_urb':F(1)},s,jmonth=7)
    with pytest.raises(ValueError,match=r'ZDC\+Z0C\+2\. >= ZA'):
        urban_step(table,switches,{**f,'za':F(1)},s,jmonth=7)


def test_defined_irrigation_time_without_anthropogenic_heat():
    tables,switches,rows=load_fixture(ROOT/'ucm-nlcd-griri.csv.gz')
    row=rows[0];table=tables[int(row['utype'])];f=forcing(row)
    # pi * 9/12 corresponds to local 21 h. AH=0 makes this defined-time
    # extension observable independently of WRF's AH-enabled fixture arm.
    f['omg']=F(F(F(3.14159)*F(9))/F(12))
    a=initial_state(row);b=initial_state(row)
    u0=urban_step(table,{**switches,'ahoption':F(0)},f,a,jmonth=7)
    u1=urban_step({**table,'ah':F(0)},{**switches,'ahoption':F(1)},f,b,jmonth=7)
    for k in u0:bits_equal(u0[k],u1[k])
    for k,v in flattened(a).items():bits_equal(v,flattened(b)[k])


def test_integer_power_order_and_correctly_rounded_intrinsics():
    from gpuwm.verify.urban_ucm_ref import _intrinsic
    x=F(1.1)
    bits_equal(_powi(x,3),F(x*F(x*x)))
    bits_equal(_powi(x,4),F(F(x*x)*F(x*x)))
    import math
    bits_equal(_intrinsic('atan',x),F(math.atan(float(x))))
    bits_equal(_intrinsic('log10',x),F(math.log10(float(x))))


def test_reference_imports_no_gpu_package():
    """In a fresh interpreter: the session may already hold CuPy for the
    GPU tests, which says nothing about what this module imports."""
    import subprocess
    import sys
    code = ("import sys, gpuwm.verify.urban_ucm_ref; "
            "sys.exit(any(k == 'cupy' or k.startswith('cupy.') "
            "for k in sys.modules))")
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0


# ---------------------------------------------------------------------------
# The Noah and Noah-MP coupling oracles, replayed on the CPU reference
# ---------------------------------------------------------------------------

def _coupling_rows(name):
    return [{k: (int(v) if k in ('step', 'case', 'ivgtyp', 'utype', 'tapped')
                 else F(float(v))) for k, v in r.items()}
            for r in read_csv(ROOT / name)]


def _check(got, want, label, failures):
    if F(got).view(np.uint32) != F(want).view(np.uint32):
        failures.append(f'{label}: got {float(got)!r}, want {float(want)!r}')


_POST_STATE = ['tr', 'tb', 'tg', 'tc', 'qc', 'uc', 'xxxr', 'xxxb', 'xxxg',
               'xxxc', 'cmcr', 'tgr', 'drelr', 'drelb', 'drelg', 'flxhumr',
               'flxhumb', 'flxhumg']
_POST_COEF = ['cmr', 'chr', 'cmc', 'chc', 'cmgr', 'chgr']
_POST_URB = ['ts', 'psim', 'psih', 'gz1oz0', 'u10', 'v10', 'th2', 'q2', 'ust']


def _compare_renewal(result, out, row, label, failures):
    for k in _POST_STATE:
        _check(result[k + '_urb2d'], row[k], f'{label} {k}', failures)
    for k in sorted(LAYER_FIELDS):
        for i in range(4):
            _check(result[k + '_urb3d'][i], row[f'{k}{i + 1}'],
                   f'{label} {k}{i + 1}', failures)
    for k in _POST_COEF:
        _check(result[k + '_sfcdif'], row[k], f'{label} {k}', failures)
    for k in _POST_URB:
        _check(result[k + '_urb2d'], row[k + '_urb'], f'{label} {k}_urb', failures)
    for k in ('sh', 'lh', 'g', 'rn'):
        _check(out[k], row[k + '_urb'], f'{label} {k}_urb', failures)
    _check(result['akms_urb2d'], row['akms_urb'], f'{label} akms_urb', failures)


def _urban_forcing(*, ta, qa, u1, v1, ssg, llg, rainbl, rhoo, za, omg, znt,
                   chs, chs2):
    """The forcing both drivers hand ``urban`` (noahdrv.F:1336-1360,
    noahmpdrv.F:3390-3415), built by the reference's own helper."""
    from gpuwm.verify.urban_ucm_ref import _driver_forcing
    call = _driver_forcing(ta=ta, qa=qa, u1=u1, v1=v1, soldn=ssg, glw=llg,
                           rainbl=rainbl, dt=F(60), rhoo=rhoo, za=za,
                           declin=F(0.37), cosz=F(0.8), omg=omg,
                           xlat=F(38.5), znt=znt, chs=chs, chs2=chs2)
    return {k: call[k + '_urb'] for k in FORCING_FIELDS}


def test_noah_coupling_oracle_on_the_cpu_reference():
    """ucm-noah.csv.gz (the lsm of WRF v4.7.1 with sf_urban_physics=1): each
    urban row from its recorded state and the rural values WRF held at the
    UCM entry, through urban_step and noah_blend, bit for bit."""
    from gpuwm.verify.urban_ucm_ref import _floors
    tables, switches, _ = load_fixture(ROOT / 'ucm-nlcd-default.csv.gz')
    failures = []
    rows = [r for r in _coupling_rows('ucm-noah.csv.gz') if r['tapped'] == 1]
    assert len(rows) == 60
    for r in rows:
        chs, chs2, cqs2 = _floors(r['tap_chs'], r['tap_chs2'], r['tap_cqs2'])
        rhoo = F(F(r['tap_sfcprs']) / F(F(F(287.04) * F(r['tap_sfctmp']))
                                         * F(F(1) + F(F(0.61) * F(r['tap_q2k'])))))
        forcing_ = _urban_forcing(
            ta=r['tap_sfctmp'], qa=r['tap_q2k'], u1=r['u1'], v1=r['v1'],
            ssg=r['tap_soldn'], llg=r['tap_glw'], rainbl=r['tap_rainbl'],
            rhoo=rhoo, za=r['tap_zlvl'], omg=r['omg'], znt=r['tap_znt'],
            chs=chs, chs2=chs2)
        state = initial_state(r)
        out = urban_step(tables[r['utype']], switches, forcing_, state, jmonth=7)
        result = noah_blend(
            {**out, **state}, frc_urb=r['frc'], t1=r['tap_t1'],
            sheat=r['tap_sheat'], eta_kinematic=r['tap_eta_kinematic'],
            eta=r['tap_eta'], ssoil=r['tap_ssoil'], albedok=r['tap_albedok'],
            q1=r['tap_q1'], sfctmp=r['tap_sfctmp'], q2k=r['tap_q2k'],
            sfcprs=r['tap_sfcprs'], zlvl=r['tap_zlvl'], soldn=r['tap_soldn'],
            rainbl=r['tap_rainbl'], emissi=F(0.95), ust=r['tap_ust'],
            u1=r['u1'], v1=r['v1'], glw=r['tap_glw'], dt=F(60),
            declin=F(0.37), cosz=F(0.8), omg=r['omg'], xlat=F(38.5),
            znt=r['tap_znt'], chs=r['tap_chs'], chs2=r['tap_chs2'],
            cqs2=r['tap_cqs2'])
        label = f"step {r['step']} case {r['case']}"
        for k in ('tsk', 'hfx', 'qfx', 'lh', 'grdflx', 'albedo', 'qsfc',
                  'ust', 'chs', 'chs2', 'cqs2'):
            _check(result[k], r[k], f'{label} {k}', failures)
        _compare_renewal(result, out, r, label, failures)
    assert not failures, '\n'.join(failures[:20])


def test_noahmp_coupling_oracle_on_the_cpu_reference():
    """ucm-noahmp.csv.gz (noahmp_urban plus surface_driver.F:3383-3405):
    each urban row through urban_step, noahmp_blend and noahmp_overrides."""
    from gpuwm.verify.urban_ucm_ref import _floors
    tables, switches, _ = load_fixture(ROOT / 'ucm-nlcd-default.csv.gz')
    failures = []
    rows = [r for r in _coupling_rows('ucm-noahmp.csv.gz') if r['utype'] > 0]
    assert len(rows) == 44
    for r in rows:
        chs, _, cqs2 = _floors(r['chs_in'], r['chs2_in'], r['cqs2_in'])
        qa = F(F(r['qv1']) / F(F(1) + F(r['qv1'])))
        rhoo = F(F(F(F(r['p8w2']) + F(r['p8w1'])) * F(0.5))
                 / F(F(F(287.04) * F(r['t3d1'])) * F(F(1) + F(F(0.61) * qa))))
        forcing_ = _urban_forcing(
            ta=r['t3d1'], qa=qa, u1=r['u1'], v1=r['v1'], ssg=r['swdown'],
            llg=r['glw'], rainbl=r['rainbl'], rhoo=rhoo,
            za=F(F(0.5) * F(r['dz8w1'])), omg=r['omg'], znt=r['znt_in'],
            chs=chs, chs2=cqs2)
        state = initial_state(r)
        out = urban_step(tables[r['utype']], switches, forcing_, state, jmonth=7)
        result = noahmp_blend(
            {**out, **state}, frc_urb=r['frc'], albedo=r['albedo_in'],
            hfx=r['hfx_in'], qfx=r['qfx_in'], lh=r['lh_in'],
            grdflx=r['grdflx_in'], tsk=r['tsk_in'], qsfc=r['qsfc_in'],
            ust=r['ust_in'], t3d=r['t3d1'], qv=r['qv1'], u1=r['u1'],
            v1=r['v1'], swdown=r['swdown'], glw=r['glw'], rainbl=r['rainbl'],
            dt=F(60), p8w_lower=r['p8w1'], p8w_upper=r['p8w2'],
            dz8w=r['dz8w1'], declin=F(0.37), cosz=F(0.8), omg=r['omg'],
            xlat=F(34.0), znt=r['znt_in'], chs=r['chs_in'],
            chs2=r['chs2_in'], cqs2=r['cqs2_in'])
        label = f"step {r['step']} case {r['case']}"
        for k in ('tsk', 'hfx', 'qfx', 'lh', 'grdflx', 'albedo', 'qsfc',
                  'ust', 'chs', 'chs2', 'cqs2'):
            _check(result[k], r[k], f'{label} {k}', failures)
        _compare_renewal(result, out, r, label, failures)
        over = noahmp_overrides(
            dict(out), frc_urb=r['frc'], fvegxy=r['fvegxy'],
            t2mvxy=r['t2mvxy'], t2mbxy=r['t2mbxy'], q2mvxy=r['q2mvxy'],
            q2mbxy=r['q2mbxy'], psfc=r['psfc'], chs=result['chs'],
            akms_urb2d=result['akms_urb2d'])
        for k in ('t2', 'th2', 'q2', 'u10', 'v10', 'psim', 'psih', 'gz1oz0',
                  'akhs', 'akms'):
            _check(over[k], r[k], f'{label} {k}', failures)
    assert not failures, '\n'.join(failures[:20])
