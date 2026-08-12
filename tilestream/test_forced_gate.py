"""The FORCED rung's battery door: specified BCs at 1, 2 and 4 ranks.

``tilestream.multigpu.gate_forced`` is the verdict machine; this module is
the ``tools/battery/tiles_gates.txt``-shaped runner (a ``main()`` that
prints the matrix and exits 0/1), because the gates file invokes modules,
not subcommands.  Knobs come from the environment:

    NX, NY, NZ, STEPS   domain and run length (defaults 220x168x49, 6)
    DEVICES             comma-separated card list ranks round-robin over
    SEAM                inert-seam mode for the positive arms (zeros)

Run directly::

    python -m tilestream.test_forced_gate
    DEVICES=0 STEPS=8 python -m tilestream.test_forced_gate
"""
from __future__ import annotations

import os
import sys

from tilestream import harness, multigpu


def main() -> int:
    nx = int(os.environ.get("NX", "220"))
    ny = int(os.environ.get("NY", "168"))
    nz = int(os.environ.get("NZ", str(harness.DEFAULT_NZ)))
    steps = int(os.environ.get("STEPS", "6"))
    raw = os.environ.get("DEVICES")
    devices = None if not raw else [int(v) for v in raw.split(",")
                                    if v.strip() != ""]
    seam = os.environ.get("SEAM", "zeros")
    ok, _results = multigpu.gate_forced(nx, ny, nz, steps, devices=devices,
                                        seam=seam)
    print("forced gate:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
