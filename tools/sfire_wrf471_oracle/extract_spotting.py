"""Extract complete original firebrand helpers with their module constants."""
from pathlib import Path
import argparse
import hashlib
import json
import re

SOURCE_SHA256 = "e68e4b4ddbde6fdc117163e3b396cff3a11bfc7f2ac586de1525707ba82092e4"


def extract(source, destination):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Firebrand controls need the pinned WRF v4.7.1 module")
    header = raw[raw.index(b"    INTEGER, PARAMETER :: dp"):raw.index(b"CONTAINS")]
    begin = raw.index(b"    PURE &", raw.index(b"END SUBROUTINE firebrand_spotting_em_init"))
    end = raw.index(b"    SUBROUTINE firebrand_spotting_em_driver(", begin)
    group1 = raw[begin:end]
    begin = raw.index(b"    SUBROUTINE advect_xyz_m(")
    end = raw.index(b"    SUBROUTINE get_local_ijk(", begin)
    group2 = raw[begin:end]
    property_body=re.search(rb"(?mis)^\s*FUNCTION firebrand_property\(.*?END FUNCTION firebrand_property",group1).group()
    corrected_property=property_body.replace(b"FUNCTION firebrand_property",b"FUNCTION firebrand_property_corrected").replace(
        b"(prop%p_effd/1000.0_dp)**2",b"(prop%p_effd/1000.0_dp)**3")
    service = b"""module spotting_service
implicit none
type domain
 integer :: unused=0
end type
end module
module module_spotting_oracle
use, intrinsic :: ieee_arithmetic
use spotting_service
implicit none
"""
    result = service + header + b"contains\n" + group1 + group2 + corrected_property + b"""\nend module module_spotting_oracle
subroutine wrf_error_fatal(message)
character(len=*),intent(in)::message
print *,message
error stop 'native firebrand fatal condition'
end subroutine
"""
    Path(destination).write_bytes(result)
    receipt = dict(source_sha256=SOURCE_SHA256,
        constants_sha256=hashlib.sha256(header).hexdigest(),
        helper_group_sha256=[hashlib.sha256(v).hexdigest() for v in (group1, group2)],
        corrected_property_sha256=hashlib.sha256(corrected_property).hexdigest(),
        wrapper_sha256=hashlib.sha256(result).hexdigest(),
        extraction="unmodified contiguous native helper bodies and module constants; empty unused domain service type")
    Path(destination).with_suffix(".json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("destination")
    a = p.parse_args()
    extract(a.source, a.destination)
