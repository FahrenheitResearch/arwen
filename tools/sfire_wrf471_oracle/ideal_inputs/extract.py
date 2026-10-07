"""Compile the unchanged native formatted READ routines without model setup."""
from pathlib import Path
import argparse,hashlib,json

def routine(text,name):
 start=text.lower().index('subroutine '+name+'(')
 end=text.lower().index('end subroutine '+name,start)
 return text[start:end]+'end subroutine '+name+'\n'

def main():
 p=argparse.ArgumentParser();p.add_argument('--util',type=Path,required=True);p.add_argument('--initializer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 util=a.util.read_bytes();initial=a.initializer.read_bytes()
 assert hashlib.sha256(util).hexdigest()=='ab9499f12b305257fd62a0c76f299ee308b7fabc98c6fae073ba25f132ec8d27'
 assert hashlib.sha256(initial).hexdigest()=='c92d30ddb871bd2188fc7c37391b330eb6c9eca24a81abbfc34429fc5ab966b7'
 body=routine(util.decode(),'read_array_2d_real')+routine(initial.decode(),'read_sounding')
 suffix='''
subroutine check_mesh_2dim(its,ite,jts,jte,ims,ime,jms,jme)
integer,intent(in)::its,ite,jts,jte,ims,ime,jms,jme
if(its<ims.or.ite>ime.or.jts<jms.or.jte>jme)error stop 1
end subroutine
subroutine wrf_get_nproc(n)
integer,intent(out)::n
n=1
end subroutine
subroutine wrf_get_myproc(n)
integer,intent(out)::n
n=0
end subroutine
subroutine crash(text)
character(len=*),intent(in)::text
error stop text
end subroutine
subroutine message(text)
character(len=*),intent(in)::text
end subroutine
end module input_reference
'''
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'input_reference.F90').write_text('module input_reference\nimplicit none\ncontains\n'+body+suffix,newline='\n')
 (a.output/'native-source.json').write_text(json.dumps({'util_sha256':hashlib.sha256(util).hexdigest(),'initializer_sha256':hashlib.sha256(initial).hexdigest(),'extracted_routines_sha256':hashlib.sha256(body.encode()).hexdigest(),'scope':'Unchanged native formatted READ routines with serial execution and bounds/message service stubs'},indent=2)+'\n',newline='\n')
if __name__=='__main__':main()
