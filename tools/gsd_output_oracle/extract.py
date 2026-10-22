"""Extract the pinned GSD hail expressions; compile main.f90 with gfortran.
Usage: python extract.py SOURCE_DIR OUTPUT_DIR
No source rewriting other than replacing the three-dimensional scalar indexes.
"""
from pathlib import Path
import hashlib, sys
src=Path(sys.argv[1])/"module_diag_gsd.F"
out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
lines=src.read_text().splitlines()
def excerpt(a,b):
    return "\n".join(lines[a-1:b])+"\n"
# Scalarize the authority's loops, leaving every arithmetic expression intact.
thompson=excerpt(1498,1504).replace("temp_qg(i,k,j)","mass").replace("temp_ng(i,k,j)","number")
tail=excerpt(1614,1633).replace("temp_qg(i,k,j)","mass").replace("temp_ng(i,k,j)","number")
program="""program oracle
implicit none
integer :: n,ng,m
real :: xrho_g,xam_g,xbm_g,xmu_g,cge(3),cgg(3),zans1,mass,number
real :: hail_max_sp
real :: masses(7)=[1.e-7,1.e-6,1.e-5,1.e-4,1.e-3,1.e-2,0.1]
double precision :: N0exp,lam_exp,lamg,N0_g,sum_ng,sum_t,f_d,hail_max
integer,parameter :: ngbins=50
double precision,parameter :: thresh_conc=0.0005d0
double precision :: xxDx(ngbins+1),xxDg(ngbins),xdtg(ngbins)
"""+excerpt(1361,1364)+excerpt(1484,1490)+excerpt(1596,1605)+"""
do n=1,ngbins
 print '(A,1X,I3,2(1X,Z16.16))','BIN',n,transfer(xxDg(n),0_8),transfer(xdtg(n),0_8)
enddo
do m=1,size(masses)
 mass=masses(m)
 if(mass<1.e-6)then
  hail_max_sp=0.
 else
"""+thompson+tail+"""
  hail_max_sp=hail_max
 endif
 print '(A,1X,Z8.8,1X,Z8.8)','HAIL',transfer(mass,0),transfer(hail_max_sp,0)
enddo
contains
"""+"\n".join(lines[lines.index('      REAL FUNCTION GAMMLN(XX)'):])+"\n"
# The source's final END MODULE/#endif are outside the two extracted functions.
program=program[:program.rindex('END MODULE module_diag_gsd')]+"end program\n"
program="\n".join(l for l in program.splitlines() if not l.startswith('#'))+"\n"
(out/"main.f90").write_text(program)
(out/"source-sha256.txt").write_text(hashlib.sha256(src.read_bytes()).hexdigest()+"  module_diag_gsd.F\n")
