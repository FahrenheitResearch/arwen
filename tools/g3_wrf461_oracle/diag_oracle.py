"""Build and pack the public WRF time-average oracle on a CPU box."""
from pathlib import Path
import argparse, hashlib, json, struct, subprocess
import numpy as np

PIN = '2014aa2b39b1ad5fa7168ee2c731d02eba71532822879890b4d273c131c80af9'
FIELDS = ('t','p','qv','qc','qi','cloud','ice','rain')
OUTPUTS = ('gd_cloud_a','qc_cu','gd_cloud2_a','qi_cu','gd_cldfr','raincv_a','raincv_b')

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('wrf'); ap.add_argument('build'); ap.add_argument('fixtures'); a=ap.parse_args()
 source=Path(a.wrf)/'dyn_em/module_convtrans_prep.F'; build=Path(a.build).resolve(); fixtures=Path(a.fixtures)
 if hashlib.sha256(source.read_bytes()).hexdigest()!=PIN: raise ValueError('time-average WRF source hash changed')
 build.mkdir(parents=True,exist_ok=True); fixtures.mkdir(parents=True,exist_ok=True)
 here=Path(__file__).resolve().parent
 receipts=[]
 for opt in ('O0','O2'):
  flags=['-O0'] if opt=='O0' else ['-O2','-fno-fast-math','-ffp-contract=off','-fno-tree-vectorize']
  subprocess.run(['gfortran',*flags,'-ffree-form','-ffree-line-length-none','-c',str(source),'-o',str(build/f'module-{opt}.o'),'-J',str(build)],check=True)
  subprocess.run(['gfortran',*flags,'-ffree-form','-ffree-line-length-none','-I',str(build),str(here/'run_diag.F90'),str(build/f'module-{opt}.o'),'-o',str(build/f'run-{opt}')],check=True)
 for label,fqi,adapt in (('fixed',1,0),('adaptive',0,1)):
  nx=ny=8; nz=20; ns=12; avg=np.float32(4.)
  rng=np.random.default_rng(721)
  volume=(ns,ny,nz,nx); plane=(ns,ny,nx)
  data={'t':rng.uniform(245,310,volume).astype('f4'),'p':rng.uniform(35000,100000,volume).astype('f4'),'qv':rng.uniform(0,0.015,volume).astype('f4')}
  for field in ('qc','qi','cloud','ice'): data[field]=(rng.uniform(0,0.001,volume)*(rng.uniform(0,1,volume)>0.45)).astype('f4')
  data['rain']=rng.uniform(0,2,plane).astype('f4')
  dt=np.array([55,65,60,70,65,60,65,60,60,55,65,75] if adapt else [60]*ns,dtype='f4')
  curr=np.zeros(ns,dtype='f4')
  for n in range(1,ns): curr[n]=np.float32(curr[n-1]+dt[n-1])
  fin=build/f'{label}.bin'
  with fin.open('wb') as f:
   f.write(struct.pack('<6if',nx,ny,nz,ns,fqi,adapt,avg)); f.write(dt.tobytes()); f.write(curr.tobytes())
   for field in FIELDS: f.write(data[field].tobytes())
  for opt in ('O0','O2'): subprocess.run([str(build/f'run-{opt}'),str(fin),str(build/f'{label}-{opt}.out')],check=True)
  raw=(build/f'{label}-O0.out').read_bytes()
  assert raw==(build/f'{label}-O2.out').read_bytes(), 'WRF O0/O2 time-average disagreement'
  out={}; off=0
  for n in range(ns):
   for field in OUTPUTS:
    shape=(ny,nz,nx) if field not in ('raincv_a','raincv_b') else (ny,nx); size=int(np.prod(shape))
    out.setdefault(field,[]).append(np.frombuffer(raw,'<f4',size,off).reshape(shape).copy()); off+=4*size
  assert off==len(raw)
  packed={f'in_{k}':v for k,v in data.items()}; packed.update(dt=dt,curr=curr,avg=avg,fqi=np.int32(fqi),adapt=np.int32(adapt))
  packed.update({f'out_{k}':np.stack(v) for k,v in out.items()})
  np.savez_compressed(fixtures/f'diag-{label}.npz',**packed)
  receipts.append(dict(case=label,columns=64,steps=ns,words=len(raw)//4,source_sha256=PIN,output_sha256=hashlib.sha256(raw).hexdigest(),cross_build_equal=True))
 (build/'receipt.json').write_text(json.dumps(receipts,indent=2))
 print(json.dumps(receipts))
if __name__=='__main__': main()
