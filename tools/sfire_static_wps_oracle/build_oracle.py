"""Compile unmodified WPS interpolation routines for a Rust fixture anchor."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import urllib.request

parser=argparse.ArgumentParser(description='Build compiled WPS interpolation control')
parser.add_argument('--work-dir',type=Path,default=Path('work/sfire-static-wps-oracle'))
args=parser.parse_args()
root=args.work_dir.resolve()
root.mkdir(parents=True,exist_ok=True)
url='https://raw.githubusercontent.com/wrf-model/WPS/v4.6.0/geogrid/src/interp_module.F'
source=urllib.request.urlopen(url,timeout=120).read()
expected='cd3bf205f4870fb1deceefefd511bbc950ae97f78b08e43f68b7221333f6f424'
assert hashlib.sha256(source).hexdigest()==expected, 'WPS source bytes differ from the compiled fixture authority'
(root/'interp_module.F').write_bytes(source)
text=source.decode()
functions=[]
for name in ('nearest_neighbor','four_pt'):
    match=re.search(r'(?im)^\s*recursive function '+name+r'\(.*?^\s*end function '+name+r'\s*$',text,re.S|re.M)
    assert match,name
    functions.append(match.group())
stub='''
recursive function interp_sequence(xx, yy, izz, array, start_x, end_x, start_y, end_y, &
               start_z, end_z, msgval, interp_list, interp_opts, idx, mask_relational, maskval, mask_array) result(value)
implicit none
integer, intent(in)::izz,start_x,end_x,start_y,end_y,start_z,end_z,idx
real,intent(in)::xx,yy,msgval,array(start_x:end_x,start_y:end_y,start_z:end_z)
integer,intent(in)::interp_list(:),interp_opts(:)
real,intent(in),optional::maskval,mask_array(start_x:end_x,start_y:end_y)
character(len=1),intent(in),optional::mask_relational
real::value
value=msgval
end function interp_sequence
subroutine wps_points(xx,yy,values,out) bind(C)
use iso_c_binding
real(c_float),value::xx,yy
real(c_float),intent(in)::values(4)
real(c_float),intent(out)::out(2)
real::array(2,2,1)
integer::methods(1),options(1)
array(:,:,1)=reshape(values,[2,2])
methods=0;options=0
out(1)=nearest_neighbor(xx,yy,1,array,1,2,1,2,1,1,-9999.,methods,options,1)
out(2)=four_pt(xx,yy,1,array,1,2,1,2,1,1,-9999.,methods,options,1)
end subroutine wps_points
'''
unit='module static_oracle\ncontains\n'+'\n'.join(functions)+'\n'+stub+'\nend module static_oracle\n'
(root/'static_oracle.F90').write_text(unit)
command=['gfortran','-O0','-ffp-contract=off','-ffree-line-length-none','-shared','-fPIC',
         '-o','libstatic_oracle.so','static_oracle.F90']
subprocess.run(command,check=True,cwd=root)
lib=ctypes.CDLL(str(root/'libstatic_oracle.so'))
lib.wps_points.argtypes=[ctypes.c_float,ctypes.c_float,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
lib.wps_points.restype=None
f32=lambda x:ctypes.c_float(x).value
bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
corners=(ctypes.c_float*4)(507.125,498.75,514.375,510.0625)
cases=[]
for xx,yy in ((1.,1.),(2.,2.),(1.25,1.),(1.,1.5),(1.5,1.5),
             (1.1,1.7),(1.1234567,1.7654321),(1.99999988,1.00000012),
             (1.25,1.5),(1.75,1.5),(1.5,1.25),(1.5,1.75)):
    x=f32(xx);y=f32(yy);out=(ctypes.c_float*2)()
    lib.wps_points(x,y,corners,out)
    # Convert the native 1-based positions into the identical Rust fractions.
    u=f32(x-1.);v=f32(y-1.)
    cases.append(dict(u=bits(u),v=bits(v),corners=[bits(value) for value in corners],
                      nearest=bits(out[0]),four_point=bits(out[1])))
receipt=dict(schema='sfire-static-wps-oracle-v1',source_url=url,
    source_sha256=hashlib.sha256(source).hexdigest(),compile_command=command,
    compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
    scope='Unmodified nearest_neighbor and four_pt routines, with terminal interpolation-sequence stub',cases=cases)
(root/'native-fixture.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
