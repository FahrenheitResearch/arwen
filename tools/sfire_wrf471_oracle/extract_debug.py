"""Wrap the unmodified WRF default-REAL debug writer for native controls."""
from pathlib import Path
import argparse
import hashlib

SOURCE_SHA256 = "ab9499f12b305257fd62a0c76f299ee308b7fabc98c6fae073ba25f132ec8d27"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise SystemExit("The WRF 4.7.1 debug writer source hash differs")
    source = raw.decode("utf-8")
    begin = source.index("subroutine write_array_m3(")
    end = source.index("end subroutine write_array_m3", begin)
    routine = source[begin:end] + "end subroutine write_array_m3\n"
    prefix = """module debug_reference
implicit none
integer :: fire_print_file=1
contains
"""
    suffix = """
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
end module debug_reference
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prefix + routine + suffix, encoding="utf-8")


if __name__ == "__main__":
    main()
