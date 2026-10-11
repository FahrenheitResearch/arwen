"""Check that the transcription reports the original cu6 index-zero writes."""
from __future__ import annotations
import ctypes as c
import json
from pathlib import Path
import sys
import numpy as np
from check import fixture, host

def main():
    build = Path(sys.argv[1]).resolve()
    if sys.argv[2] == "--reference":
        lib=c.CDLL(str(build / "libcu6_checked.so"))
        lib.cu6_unsafe_flux.argtypes=[]
        lib.cu6_unsafe_flux()
        raise AssertionError("the checked WRF routine did not catch index 0")
    lib=c.CDLL(str(build / "libcu6_host.so"))
    lib.cu6_unsafe_flux.argtypes=[]
    lib.cu6_unsafe_flux.restype=c.c_int
    status=lib.cu6_unsafe_flux()
    receipt = {"backend": "cpu-transcription", "columns": 1, "nz": 4,
               "config": {"dt": 60., "cloud_top": 3, "active": False, "boundary": "cuflx"},
               "status": status, "pass": status < 0, "gpu_checked": False,
               "prevents": "module_cu_tiedtke.F:2282/2289 flux write at vertical index 0"}
    Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt))
    return int(status >= 0)

if __name__ == "__main__":
    raise SystemExit(main())
