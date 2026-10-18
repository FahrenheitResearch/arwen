import sys,json,subprocess,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np

D=Path(sys.argv[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gf_regime_columns as g

regimes=[
 g.Regime('sol-hot-saturated', 'very warm humid ocean',318,xland=2,hfx=90,qfx=2e-4,lapse=7.8,pbl_depth=1200,w_amp=.9,rh=((0,.98),(6000,.95),(14000,.7)),seed=701,t_trop_min=192),
 g.Regime('sol-high-6200m', 'high terrain moist elevated convection',284,ht=6200,psfc=47000,hfx=250,qfx=1e-4,lapse=8.4,pbl_depth=1800,w_amp=.45,rh=((0,.8),(4000,.75),(11000,.4)),seed=702,t_trop_min=190),
 g.Regime('sol-cold-inversion', 'cold dry inversion with descent',215,ht=2100,psfc=79000,pbl_kind='stable',stable_inv=18,pbl_depth=800,lapse=4.0,w_amp=-.15,rh=((0,.05),(5000,.03)),seed=703,t_trop_min=180),
 g.Regime('sol-cold-moist-ascent', 'cold moist unstable ice cloud',255,xland=2,hfx=350,qfx=9e-5,pbl_depth=1600,lapse=8.6,w_amp=.7,rh=((0,.99),(6000,.95),(12000,.5)),seed=704,t_trop_min=182),
 g.Regime('sol-hot-dry-desert', 'hot dry deep mixed layer',326,ht=-300,psfc=104000,hfx=700,qfx=0,pbl_depth=4500,lapse=8.2,w_amp=.02,rh=((0,.03),(5000,.05),(12000,.03)),seed=705,t_trop_min=195),
 g.Regime('sol-shallow-sharp-cap', 'moist shallow layer under strong cap',299,xland=2,hfx=40,qfx=8e-5,pbl_depth=900,cap_z=950,cap_dt=11,cap_depth=100,lapse=5.4,w_amp=.06,rh=((0,.95),(900,.99),(1500,.15),(6000,.1)),seed=706,t_trop_min=195),
]
original_heights=g.interface_heights
receipts=[]
for nz in (16,32,40,50,64,80):
 g.NZ=nz
 def grid(nz=nz):
  dz=np.geomspace(40,900,nz); dz*=21000/dz.sum()
  return np.r_[0,np.cumsum(dz)]
 g.interface_heights=grid
 cols=[]
 for ir,r in enumerate(regimes):
  for dt in (1e-6,.125,1,30,600,1800,7200):
   for ish in (0,1):
    kw=dict(kpbl_override=2) if 'shallow' in r.name else {}
    c=g.build_column(r,dx=(900,1800,3000,15000,75000,250000)[ir],dt=dt,ishallow=ish,ichoice=(ir*3)%17,variant=f'nz{nz}-dt{dt}-sh{ish}',**kw)
    cols.append(c)
 rng=np.random.default_rng(20261007+nz)
 for i in range(96):
  r=regimes[i%len(regimes)]
  r=replace(r,t_sfc=float(r.t_sfc+rng.uniform(-6,6)),w_amp=float(rng.uniform(-.5,1.8)),seed=10000+nz*100+i,
      hfx=float(rng.uniform(-150,800)),qfx=float(rng.uniform(-2e-5,2e-4)),u_shear=float(rng.uniform(-90,90)),v_shear=float(rng.uniform(-60,60)))
  kw={'kpbl_override':(1,nz-2,nz//2)[i%3]} if i%8==0 else {}
  c=g.build_column(r,dx=float(rng.uniform(800,180000)),dt=float(10**rng.uniform(-3,3.5)),ishallow=i%2,ichoice=i%17,variant=f'new-seed-{i}',**kw)
  cols.append(c)
 sc,ii,lv,pw=g.pack(cols)
 out=D/f'extra-nz{nz}';out.mkdir(exist_ok=True)
 with (out/'gf-regime-inputs.bin').open('wb') as f:
  np.array([g.MAGIC_IN,len(cols),nz],dtype='<i4').tofile(f)
  for a,dtype in ((sc,'<f4'),(ii,'<i4'),(lv,'<f4'),(pw,'<f4')):a.astype(dtype).tofile(f)
 manifest=dict(nz=nz,ncol=len(cols),columns=[dict(column=i,regime=c.regime,variant=c.variant,dt=c.dt,dx=c.dx,
     ishallow=c.ishallow,ichoice=c.ichoice,kpbl=c.kpbl,xland=c.sca['xland'],ht=c.sca['ht'],notes=c.notes) for i,c in enumerate(cols)])
 (out/'gf-regime-manifest.json').write_text(json.dumps(manifest,indent=1)+'\n')
 for kind in ('crgamma','stock','nan','bounds','safe','corrected'):
  p=subprocess.run([str(D/'regime'/f'run_{kind}'),str(out/'gf-regime-inputs.bin'),str(out/f'gf-regime-wrf-{kind}-k0.bin'),'0'],capture_output=True,text=True)
  (D/'receipts'/f'extra-nz{nz}-{kind}.log').write_text(p.stdout+p.stderr)
  if p.returncode: raise RuntimeError(f'{kind} nz{nz} exited {p.returncode}: {p.stderr}')
 cr=g.read_outputs(out/'gf-regime-wrf-crgamma-k0.bin');nan=g.read_outputs(out/'gf-regime-wrf-nan-k0.bin')
 receipts.append(dict(nz=nz,ncol=len(cols),steps=1,input_sha256=hashlib.sha256((out/'gf-regime-inputs.bin').read_bytes()).hexdigest(),
    nan_init_identical=all(np.array_equal(cr[k].view(np.uint32),nan[k].view(np.uint32)) for k in ('lev','sca','isc')),
    bounds_exit=0,reference_finite=bool(np.isfinite(cr['lev']).all() and np.isfinite(cr['sca']).all()),
    precipitating=int((cr['sca'][:,g.OUT_SCA.index('pratec')]>0).sum()),
    shallow_fired=int((cr['sca'][:,g.OUT_SCA.index('xmb_shallow')]>0).sum())))
 print(receipts[-1],flush=True)
(D/'receipts/extra-oracle-build.json').write_text(json.dumps(receipts,indent=1)+'\n')
