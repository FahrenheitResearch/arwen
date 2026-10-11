"""Every tangent argument cu6 can reach: the Fortran's tan() against WOOF's.

The only tangent call in WRF v4.6.1 ``module_cu_tiedtke.F`` is the organized
detrainment at F:2939, ``tan(arg)`` with ``arg = 3.1415*(zzmzk/ztmzk)*0.5``
and ``0 <= zzmzk <= ztmzk`` (F:2931-2937), so ``arg`` lies in
``[0, fl(fl(3.1415)*0.5)]``.  This sweeps every float32 word from +0 up to
fl(pi/2), a superset of that range: 1,070,141,404 arguments.

The Fortran side is the oracle library's ``cu6_tan`` (gfortran ``tan`` on
REAL32, which binds the process libm's ``tanf``); the WOOF side is
``glibc_tanf`` from ``gpuwm/core/kernels/glibc_trig_flt32.cuh`` compiled
for the host by ``host.cpp``.  The receipt records the libc version and
the SHA256 of the libm actually mapped into the process, because the answer
is a property of that libm: glibc 2.41+ ships the CORE-MATH correctly
rounded tanf that WOOF transcribes; glibc 2.40 and older do not.

Usage: python tan_domain_check.py BUILD RECEIPT.json
"""
from __future__ import annotations
import ctypes as c
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import FP, libm_identity, pointer  # noqa: E402

LAST = int(np.array([np.pi / 2], dtype=np.float32).view(np.uint32)[0])  # 0x3fc90fdb
REACHABLE = np.float32(np.float32(3.1415) * np.float32(1.0)) * np.float32(0.5)
CHUNK = 1 << 24


def main():
    build = Path(sys.argv[1]).resolve()
    fortran = c.CDLL(str(build / "libcu6_oracle.so")).cu6_tan
    woof = c.CDLL(str(build / "libcu6_host.so")).cu6_tan
    for fn in (fortran, woof):
        fn.argtypes = [c.c_int, FP, FP]
        fn.restype = None
    differing = 0
    max_ulp = 0
    first = []
    reachable_differing = 0
    for start in range(0, LAST + 1, CHUNK):
        words = np.arange(start, min(start + CHUNK, LAST + 1), dtype=np.uint32)
        x = words.view(np.float32)
        a = np.empty_like(x)
        b = np.empty_like(x)
        fortran(x.size, pointer(x), pointer(a))
        woof(x.size, pointer(x), pointer(b))
        bad = a.view(np.uint32) != b.view(np.uint32)
        if not np.any(bad):
            continue
        # Every tangent here is finite and non-negative, so word distance is ULP distance.
        ulp = np.abs(a.view(np.uint32).astype(np.int64) - b.view(np.uint32).astype(np.int64))
        differing += int(np.count_nonzero(bad))
        reachable_differing += int(np.count_nonzero(bad & (x <= REACHABLE)))
        max_ulp = max(max_ulp, int(ulp.max()))
        for i in np.flatnonzero(bad)[: max(0, 8 - len(first))]:
            first.append({"arg": f"{int(words[i]):08x}", "fortran": f"{int(a.view(np.uint32)[i]):08x}",
                          "woof": f"{int(b.view(np.uint32)[i]):08x}"})
    result = {"arguments": LAST + 1, "last_argument_word": f"{LAST:08x}",
              "reachable_max_word": f"{int(np.array([REACHABLE]).view(np.uint32)[0]):08x}",
              "differing": differing, "differing_in_reachable_range": reachable_differing,
              "max_ulp": max_ulp, "first_differences": first, **libm_identity()}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return int(differing != 0)


if __name__ == "__main__":
    raise SystemExit(main())
