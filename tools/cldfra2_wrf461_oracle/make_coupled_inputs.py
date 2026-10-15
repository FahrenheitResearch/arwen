"""Deterministic column inputs only. No data decoding or regridding."""
from pathlib import Path
import sys
import numpy as np
out=Path(sys.argv[1]); nz=40; nc=64
F=np.float32
pw=np.linspace(101300,5000,nz+1,dtype=F)
p=(pw[:-1]+pw[1:])*F(.5)
z=np.linspace(0,16000,nz,dtype=F)
t=np.maximum(F(288.15)-F(.0065)*z,F(210))
tw=np.r_[t[:1],(t[:-1]+t[1:])*F(.5),t[-1:]].astype(F)
dz=np.full(nz,F(400)); exner=(p/F(1e5))**F(287/1004.5)
rho=p/(F(287)*t); qv=F(.012)*np.exp(-z/F(2500))
th=F(1e-6); lo=np.nextafter(th,F(0)); hi=np.nextafter(th,F(np.inf))
qc_values=[F(0),lo,th,hi,th*F(.5),hi*F(.5),-th,F(3e-5)]
qi_values=[F(2e-6),F(0),F(0),F(0),th*F(.5),hi*F(.5),F(2e-6),F(2e-5)]
def numbers(values): return ' '.join(f'{F(v):.9e}' for v in values)+'\n'
with (out/'lw-input.txt').open('w') as lw, (out/'sw-input.txt').open('w') as sw:
 lw.write(f'{nc} {nz}\n'); sw.write(f'{nc} {nz}\n')
 for c in range(nc):
  fqc=(c//8)%4 in (1,3); fqi=(c//8)%4 in (2,3)
  lw.write(f'{c+1} 2001 166 10 2 2 0 0 0 0 0 1 {int(fqc)} 1 {int(fqi)} 1 1 0\n')
  lw.write(numbers([166,0,0,.98,290,101300,1,0,0,0,5000]))
  sw.write(f'{c+1} 2001 166 10 2 2 0 2 0 0 0 2 0\n')
  sw.write(numbers([166,12,0,12,0,1370,1,0,0,.2,290,1,0,0,0]))
  for k in range(nz):
   i=(c+k)%8; qc=qc_values[i]; qi=qi_values[i]
   common=[p[k],t[k],dz[k],exner[k],rho[k],qv[k],qc,0,qi,0,0]
   # CF seed is deliberately wrong. The public Fortran overwrites it.
   row=common+[.375,0,0,0,0]
   lw.write(numbers(row)); sw.write(numbers(row))
  for pp,tt in zip(pw,tw): lw.write(numbers([pp,tt])); sw.write(numbers([pp,tt]))
