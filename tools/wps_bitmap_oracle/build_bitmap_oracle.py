"""Extract unchanged public WPS routines into an independent scalar driver."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess

WPS_SOURCE_SHA256='cd3bf205f4870fb1deceefefd511bbc950ae97f78b08e43f68b7221333f6f424'

def require_public_source(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=WPS_SOURCE_SHA256:
        raise ValueError('WPS v4.6.0 source hash differs from the independent reference pin')
    return digest

def extract_functions(text):
    functions=[]
    for name in ('nearest_neighbor','four_pt','four_pt_average'):
        match=re.search(r'(?ims)^\s*recursive function '+name+r'\(.*?^\s*end function '+name+r'\s*$',text)
        if not match:
            raise ValueError('public function missing: '+name)
        functions.append((name,match.group(),text[:match.start()].count('\n')+1))
    return functions

STUB='''recursive function interp_sequence(xx, yy, izz, array, start_x, end_x, &
 start_y,end_y,start_z,end_z,msgval,interp_list,interp_opts,idx,mask_relational,maskval,mask_array) result(value)
 implicit none
 integer,intent(in)::start_x,end_x,start_y,end_y,start_z,end_z,izz,idx
 real,intent(in)::xx,yy,msgval
 real,intent(in)::array(start_x:end_x,start_y:end_y,start_z:end_z)
 integer,intent(in)::interp_list(:),interp_opts(:)
 character(len=1),optional,intent(in)::mask_relational
 real,optional,intent(in)::maskval,mask_array(start_x:end_x,start_y:end_y)
 real::value
 value=msgval
end function interp_sequence
'''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    digest=require_public_source(a.source)
    text=a.source.read_text()
    a.out.mkdir(parents=True,exist_ok=True)
    bodies=[]
    for name,body,line in extract_functions(text):
        bodies.append(body)
        print('SOURCE_FUNCTION',name,'line',line,flush=True)
    module=a.out/'wps_ref.f90'
    module.write_text('module wps_ref\nimplicit none\ncontains\n'+STUB+'\n'.join(bodies)+'\nend module\n')
    print('PUBLIC_SOURCE_SHA256',digest,flush=True)
    command=['gfortran','-O0','-ffp-contract=off','-fno-tree-vectorize','-ffree-line-length-none',
             str(module.resolve()),str(Path(__file__).with_name('bitmap_oracle.f90').resolve()),
             '-o',str((a.out/'oracle').resolve())]
    print('COMMAND',repr(command),flush=True)
    subprocess.run(command,cwd=a.out,check=True)

if __name__=='__main__':
    main()
