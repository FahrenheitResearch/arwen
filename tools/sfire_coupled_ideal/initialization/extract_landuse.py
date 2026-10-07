"""Wrap the original complete WRF landuse_init routine with serial services."""
from pathlib import Path
import argparse
import hashlib
import json


def extract(source, destination):
    source = Path(source).read_text()
    marker = "   SUBROUTINE landuse_init("
    start = source.index(marker)
    end_marker = "   END SUBROUTINE landuse_init"
    end = source.index(end_marker,start)+len(end_marker)
    routine = source[start:end]
    output = "module native_landuse_control\nimplicit none\ninteger,parameter::IWORDSIZE=4,RWORDSIZE=4,LWORDSIZE=4\ncontains\n"
    output += routine + "\nend module\n"
    output += "subroutine wrf_dm_bcast_bytes(values,n)\ninteger(kind=1)::values(*)\ninteger::n\nend subroutine\n"
    Path(destination).write_text(output)
    return dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),
                control_sha256=hashlib.sha256(output.encode()).hexdigest(), correction="none")


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("source");p.add_argument("destination");args=p.parse_args()
    print(json.dumps(extract(args.source,args.destination),indent=2))
