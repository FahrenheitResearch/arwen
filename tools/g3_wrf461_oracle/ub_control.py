"""The uninitialised-read control for the Grell 3D oracle.

usage: ub_control.py INPUT.bin ZEROED.bin POISONED.bin

WRF v4.6.1 G3DRV copies k22s/kbcons/ktops into K22_SHALLOW/KBCON_SHALLOW/
KTOP_SHALLOW for every band column (module_cu_g3.F:487-490), but
CUP_enss_3d only assigns them inside `if(ishallow_g3.eq.1)` (:1223-1522).
With ishallow = 0 the three diagnostics are therefore whatever was on the
stack: MEASURED, the -O0 and -O2 builds of the pinned source write
different words there.  The reference build pins the read to zero
(-finit-local-zero), which is the value G3DRV's own initialisation at
:237-241 states for "no shallow cloud"; the port implements that value.

This control proves the pin hides nothing else: a build that poisons every
uninitialised local (integers -123456, reals NaN, logicals true) must
differ from the zero-pinned build ONLY in those three fields, ONLY in the
band, and ONLY when ishallow = 0.  Exits nonzero otherwise.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_fixture_io import band, read_input, read_output  # noqa: E402

UB_FIELDS = {"k22_shallow", "kbcon_shallow", "ktop_shallow"}

inp = read_input(sys.argv[1])
a = read_output(sys.argv[2], inp["nx"], inp["ny"], inp["nz"], ensdim=inp["ensdim"])
b = read_output(sys.argv[3], inp["nx"], inp["ny"], inp["nz"], ensdim=inp["ensdim"])
bj, bi = band(inp["nx"], inp["ny"])
inside = np.zeros((inp["ny"], inp["nx"]), bool)
inside[bj, bi] = True
bad = []
moved = 0
for k in a:
    d = a[k].view(np.int32) != b[k].view(np.int32)
    if not d.any():
        continue
    if k in UB_FIELDS and inp["ishallow"] == 0 and not (d & ~inside).any():
        moved += int(d.sum())
        continue
    bad.append((k, int(d.sum())))
if bad:
    print("FATAL: outputs depend on uninitialised memory beyond the declared"
          " shallow-diagnostic read:", bad)
    sys.exit(9)
print(f"ub-control ok: {moved} declared words moved (ishallow={inp['ishallow']})")
