#!/usr/bin/env python3
"""Build independent eta fixtures from untouched WRF 4.6.1 Fortran routines.

Usage: python tools/wrf_eta_oracle.py --wrf-source /path/to/WRF --work /owned/work
Output is written to --output (tests/fixtures/wrf_eta_v461.json by default).
The source extraction is verbatim; only the module/driver/error stubs are ours.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

DRIVER = '''program driver
use eta_reference
implicit none
integer :: n, option, k
real :: top, dzmax, dzbot, stretch_s, stretch_u, base_temp
real, allocatable :: eta(:), given(:)
character(len=512) :: arg
call get_command_argument(1,arg); read(arg,*) n
call get_command_argument(2,arg); read(arg,*) option
call get_command_argument(3,arg); read(arg,*) top
call get_command_argument(4,arg); read(arg,*) dzmax
call get_command_argument(5,arg); read(arg,*) dzbot
call get_command_argument(6,arg); read(arg,*) stretch_s
call get_command_argument(7,arg); read(arg,*) stretch_u
call get_command_argument(8,arg); read(arg,*) base_temp
allocate(eta(n), given(n)); given=-1.
call compute_eta(eta,option,given,n,dzmax,dzbot,stretch_s,stretch_u, &
 top,9.81,100000.,-(1004.5-287.)/1004.5,50.,287.,1004.5, &
 base_temp,100000.,300.,200.,0.,-11., &
 1,2,1,2,1,n, 1,2,1,2,1,n, 1,2,1,2,1,n)
open(unit=31,file='eta.bin',access='stream',form='unformatted',status='replace')
write(31) eta
close(31)
end program
'''
STUBS = '''subroutine wrf_error_fatal(message)
character(len=*),intent(in)::message
print *,trim(message)
stop 2
end subroutine
subroutine wrf_message(message)
character(len=*),intent(in)::message
print *,trim(message)
end subroutine
subroutine wrf_debug(level,message)
integer,intent(in)::level
character(len=*),intent(in)::message
print *,trim(message)
end subroutine
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wrf-source', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('tests/fixtures/wrf_eta_v461.json'))
    a=p.parse_args()
    source=(a.wrf_source/'dyn_em/module_initialize_real.F').read_bytes()
    text=source.decode().replace('\r\n','\n')
    start=text.index('   SUBROUTINE compute_eta (')
    end=text.index('    END SUBROUTINE levels',start)+len('    END SUBROUTINE levels')
    extracted=text[start:end]
    a.work.mkdir(parents=True,exist_ok=True)
    (a.work/'reference.f90').write_text('module eta_reference\ncontains\n'+extracted+'\nend module\n'+STUBS+DRIVER)
    subprocess.run(['gfortran','-O0','-ffree-line-length-none','-fcheck=bounds',
                    'reference.f90','-o','eta-reference'],cwd=a.work,check=True)
    cases=[]
    for opt,n,top,maxdz,bot,s,u,temp in [
        (2,80,5000,1000,50,1.3,1.1,290),
        (2,50,5000,1000,50,1.3,1.1,290),
        (2,100,2000,800,25,1.2,1.08,290),
        (2,45,10000,1400,90,1.5,1.15,305),
        (1,80,5000,1000,50,1.3,1.1,290),
        (1,50,5000,1000,50,1.3,1.1,290),
        (1,100,2000,800,25,1.2,1.08,300),
        (1,45,10000,1400,90,1.5,1.15,280),
        (2,3,5000,1000,50,1.3,1.1,290),
        (1,20,1000,500,50,1.3,1.1,290),
    ]:
        args=[n,opt,top,maxdz,bot,s,u,temp]
        name=f'option{opt}-n{n}-top{top}'
        r=subprocess.run([str(a.work.resolve()/'eta-reference'),*map(str,args)],cwd=a.work,capture_output=True,text=True)
        (a.work/(name+'.log')).write_text(r.stdout+r.stderr)
        record=dict(name=name,e_vert=n,options=dict(auto_levels_opt=opt,p_top=top,
            max_dz=maxdz,dzbot=bot,dzstretch_s=s,dzstretch_u=u,base_temp=temp),exit_code=r.returncode)
        if r.returncode==0:
            data=(a.work/'eta.bin').read_bytes()
            assert len(data)==4*n
            record['eta_bits']=list(struct.unpack('<'+'I'*n,data))
            record['sha256']=hashlib.sha256(data).hexdigest()
        else: record['refusal_tail']=r.stdout.splitlines()[-1].strip()
        cases.append(record)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(authority='WRF v4.6.1 module_initialize_real.F compute_eta/levels',
        source_sha256=hashlib.sha256(source).hexdigest(),
        extracted_lf_sha256=hashlib.sha256(extracted.encode()).hexdigest(),
        driver_sha256=hashlib.sha256((STUBS+DRIVER).encode()).hexdigest(),
        compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
        cases=cases),indent=2)+'\n')
    print([(c['name'],c['exit_code']) for c in cases])

if __name__=='__main__':main()
