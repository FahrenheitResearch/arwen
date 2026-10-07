"""Compile the original ground extrapolation line with per-column inputs."""
from pathlib import Path
import argparse,hashlib,json


def extract(source,destination):
    source=Path(source).read_text()
    lines=[line for line in source.splitlines(keepends=True) if line.startswith('     grid%tsk(I,J)=')]
    if len(lines)!=1:
        raise ValueError('native ideal ground-temperature assignment changed')
    line=lines[0]
    text='''module native_surface_control
implicit none
type ground_record
 real::cf1,cf2,cf3
 real,allocatable::tsk(:,:)
end type
contains
subroutine extrapolate(source,coeff,output,nx,ny)
integer,intent(in)::nx,ny
real,intent(in)::source(nx,3,ny),coeff(3)
real,intent(out)::output(nx,ny)
type(ground_record)::grid
real::temp(3)
integer::I,J
allocate(grid%tsk(nx,ny));grid%cf1=coeff(1);grid%cf2=coeff(2);grid%cf3=coeff(3)
do J=1,ny
 do I=1,nx
  temp=source(I,:,J)
'''+line+''' enddo
enddo
output=grid%tsk
end subroutine
end module
'''
    Path(destination).write_text(text)
    return dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        native_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
        correction='Native temperature extrapolation applied per column to current temperature/theta when surface exchange is disabled',
        control_sha256=hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    print(json.dumps(extract(a.source,a.destination),indent=2))
