"""Independent new inputs and cycled starts; referee physics stays untouched."""
import gzip, importlib.util, json, pathlib, struct, subprocess, sys
import numpy as np
L=pathlib.Path('/work/pool-sfclay-noftz-mynn-mm5-eta')
S=L/'src'; T=S/'tools/mynn_pbl_gsd41_oracle'
spec=importlib.util.spec_from_file_location('columns',T/'columns_gsd41.py')
B=importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
regimes=[]
changes=[(0,'hot_humid_lowland',dict(th_ml=318.,ts=328.,qsurf=.027,hfx=700.,qfx=3e-4)),
 (9,'polar_very_cold',dict(th_ml=225.,ts=218.,qsurf=5e-5,qfloor=1e-7)),
 (17,'saturated_hot_cloud',dict(th_ml=307.,ts=313.,qsurf=.028,liq=(200.,4500.,.004))),
 (12,'mountain_500hpa',dict(ps=50000.,th_ml=322.,ts=275.,qsurf=.001)),
 (3,'sharp_stable_inversion',dict(dthml=.08,rmol=6.,ust=.025)),
 (1,'strong_dry_convection',dict(th_ml=322.,ts=335.,hfx=900.,qsurf=.0005,qfloor=1e-8)),
 (13,'storm_force_shear',dict(ust=2.,uref=45.,ushear=.025)),
 (14,'near_zero_shear',dict(ust=.011,uref=.01,ushear=0.,vref=0.)),
 (20,'subnormal_edge_values',dict(trace=(0.,9500.,float(np.nextafter(np.float32(0),np.float32(1))),float(np.finfo(np.float32).tiny)),snow=(0.,9500.,1e-44)))]
for idx,name,kwargs in changes:
    r=dict(B.REGIMES[idx]);r.update(kwargs);r['name']=name;regimes.append(r)
B.REGIMES=regimes
names,sca,pro=B.columns()
E=L/'extra'; E.mkdir(exist_ok=True)
meta={'names':names,'cases':[]}
for dt in (.001,1.,20.,120.,1200.):
    case=f'dt-{dt:g}'; d=E/case; d.mkdir(exist_ok=True)
    B.write(str(d/'in.bin'),6,dt)
    for form in ('sq','asis'):
        out=d/f'{form}.bin'
        if out.exists(): out.unlink()
        subprocess.run([str(L/'oracles/pbl'/form/'run_driver_columns_gsd41'),str(d/'in.bin'),str(out)],check=True)
        with gzip.open(str(out)+'.gz','wb') as f:f.write(out.read_bytes())
    meta['cases'].append(dict(name=case,dt=dt,nstep=6,cycling=False,qke=0.))
source=(T/'run_driver_columns_gsd41.F90').read_text()
for seed in (.08,.0001):
    case=f'cycling-{seed:g}';d=E/case;d.mkdir(exist_ok=True)
    B.write(str(d/'in.bin'),6,20.)
    text=source.replace('initflag, .false., .false.,','initflag, .false., .true.,')
    text=text.replace('sin_ = 0.0',f'sin_ = 0.0\n  sin_(:,1,:) = {seed}\n  sin_(:,7,:) = 2.0e-5\n  sin_(:,8,:) = 0.3')
    harness=d/'cycle.F90';harness.write_text(text)
    for form in ('sq','asis'):
        obj=L/'oracles/pbl'/form
        exe=d/f'cycle-{form}'
        subprocess.run(['gfortran','-O0','-ffp-contract=off','-ffree-form','-ffree-line-length-none','-I'+str(obj),str(harness),str(obj/'stub_wrf39.o'),str(obj/'module_bl_mynn.o'),'-o',str(exe)],cwd=d,check=True)
        out=d/f'{form}.bin'
        if out.exists(): out.unlink()
        subprocess.run([str(exe),str(d/'in.bin'),str(out)],check=True)
        with gzip.open(str(out)+'.gz','wb') as f:f.write(out.read_bytes())
    meta['cases'].append(dict(name=case,dt=20.,nstep=6,cycling=True,qke=seed))
(L/'receipts/extra-inputs.json').write_text(json.dumps(meta,indent=1))
print(json.dumps(meta,indent=1))
