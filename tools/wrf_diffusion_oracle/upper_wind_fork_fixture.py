"""Build the saved-wind fixture from NOAA-EMC/HRRR v4.1.21 module_small_step_em.F:1705-1736. Args: source-file, build-dir, fixture-npz."""
from pathlib import Path
import hashlib, json, subprocess, sys, numpy as np
raw=Path(sys.argv[1]).read_bytes()
lines=raw.decode().splitlines()
block='\n'.join(s for s in lines[1704:1736] if not s.lstrip().startswith('!'))
assert 'qq(i,j)=sqrt((qq(i,j)-1.)*qq(i,j))' in block
head='''program oracle
implicit none
integer, parameter :: nx=7, ny=5, nz=6
real :: u_save(0:nx,1:nz,0:ny-1), v_save(0:nx-1,1:nz,0:ny)
real :: ph_1(0:nx-1,1:nz+1,0:ny-1), phb(0:nx-1,1:nz+1,0:ny-1)
real :: w(0:nx-1,1:nz+1,0:ny-1), w_save(0:nx-1,1:nz+1,0:ny-1)
real :: mut(0:nx-1,0:ny-1), qq(0:nx-1,0:ny-1), dampwt(1:nz+1)
real :: rlthresh, factor, htop,hk,hbot,hdepth,dampmag,pi,dts
real, parameter :: g=9.81
integer :: i,j,k,i_start,i_end,k_end,zone,substeps,step
character(512) :: a,b
call get_command_argument(1,a)
call get_command_argument(2,b)
open(10,file=trim(a),access='stream',form='unformatted',status='old')
read(10) zone,substeps,dts,hdepth,u_save,v_save,ph_1,phb
close(10)
i_start=zone; i_end=nx-1-zone; k_end=nz
pi=4.*atan(1.); dampmag=dts*0.2
w=0.; w_save=0.; mut=90000.
do step=1,substeps
do j=zone,ny-1-zone
'''
foot='''enddo
enddo
open(11,file=trim(b),access='stream',form='unformatted',status='replace')
write(11) u_save,v_save
close(11)
end program
'''
b=Path(sys.argv[2]); b.mkdir(parents=True, exist_ok=True)
(b/'oracle.f90').write_text(head+block+'\n'+foot)
flags=['-O0','-ffp-contract=off','-fcheck=bounds','-ffree-line-length-none']
subprocess.run(['gfortran',*flags,str(b/'oracle.f90'),'-o',str(b/'oracle')],check=True)
F=lambda x:np.ascontiguousarray(x.transpose(1,0,2))
store={}; rows=[]; c=0
for speed in (109.,110.,111.,130.,150.,200.):
 for depth in (5000.,15000.):
  nz,ny,nx=6,5,7; zone=int(depth==5000.); substeps=4 if speed>130 else 1
  u=np.full((nz,ny,nx+1),np.float32(speed)*np.float32(.8),np.float32)
  v=np.full((nz,ny+1,nx),np.float32(speed)*np.float32(.6),np.float32)
  u[:3]*=np.float32(.75); v[:3]*=np.float32(.75)
  php=np.zeros((nz+1,ny,nx),np.float32)
  phb=(np.array([0,2500,5000,8000,11000,14000,17000],np.float32)*np.float32(9.81))
  base3d=bool(c%2); phb_in=np.broadcast_to(phb[:,None,None],php.shape).copy()
  if base3d:
   phb_in+=np.arange(nx,dtype=np.float32)[None,None,:]*np.float32(45)
   phb=phb_in.copy()
  with (b/'case.bin').open('wb') as fh:
   np.array([zone,substeps],np.int32).tofile(fh)
   np.array([5.,depth],np.float32).tofile(fh)
   for x in (u,v,php,phb_in): F(x).tofile(fh)
  subprocess.run([str(b/'oracle'),str(b/'case.bin'),str(b/'out.bin')],check=True)
  out=np.fromfile(b/'out.bin',np.float32)
  nu=u.size; uout=out[:nu].reshape(ny,nz,nx+1).transpose(1,0,2).copy()
  vout=out[nu:].reshape(ny+1,nz,nx).transpose(1,0,2).copy()
  p=str(c)+'_'
  for key,val in dict(u=u,v=v,php=php,phb=phb,u_out=uout,v_out=vout,dts=5.,zdamp=depth,zone=zone,substeps=substeps).items(): store[p+key]=val
  rows.append(dict(speed=speed,depth=depth,base3d=base3d,zone=zone,substeps=substeps,changed=int((u.view('u4')!=uout.view('u4')).sum()+(v.view('u4')!=vout.view('u4')).sum())))
  c+=1
store['cases']=c
np.savez(Path(sys.argv[3]),**store)
meta=dict(source_sha256=hashlib.sha256(raw).hexdigest(),source_lines=[1705,1736],source_slice_sha256=hashlib.sha256(('\n'.join(lines[1704:1736])+'\n').encode()).hexdigest(),flags=flags,compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],rows=rows)
(b/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))