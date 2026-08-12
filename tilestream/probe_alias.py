"""Does a RESIDENT open-boundary run keep the periodic alias slot an alias?

``tilestream.spec``'s module docstring states ArWen's convention: on a
periodic axis there are only ``ny`` independent v faces and slot ``ny`` is an
ALIAS of slot 0.  ``harness.make_state`` seeds it that way, the tiled scatter
maintains it that way, and every y stencil in the dycore wraps as if it were
true.

This probe asks the resident model whether it still is after one step, on a
domain that is open in x and periodic in y.  It is a MONOLITHIC measurement:
no tiling, no store, no gather.  If the answer is no, the tiled/monolithic
disagreement at the four corner tiles is a property of the DYCORE and not of
the transport, and no halo can fix it.

    python -m tilestream.probe_alias open_x
"""

from __future__ import annotations

import sys

import numpy as np

from tilestream import harness
from tilestream import test_open_bc as gate


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    arm = argv[0] if argv else "open_x"
    nsteps = int(argv[1]) if len(argv) > 1 else 1
    nx, ny = 96, 64

    cfg = gate.open_cfg(nx, ny, "dry", arm, nz=25)
    state, _geo = gate.build_domain(cfg, arm, warmup=0)

    def report(tag):
        v = cp.asnumpy(state.v)
        u = cp.asnumpy(state.u)
        dv = np.abs(v[:, -1, :] - v[:, 0, :])
        du = np.abs(u[:, :, -1] - u[:, :, 0])
        cols = np.nonzero(dv.max(axis=0) > 0)[0]
        rows = np.nonzero(du.max(axis=0) > 0)[0]
        print(f"  {tag}: max|v[ny]-v[0]| = {dv.max():.6g} at x columns "
              f"{cols.tolist()[:12]} of {ny and nx}")
        print(f"  {tag}: max|u[nx]-u[0]| = {du.max():.6g} at y rows "
              f"{rows.tolist()[:12]}")

    print(f"{arm}: {nx}x{ny}x25, periodic_x="
          f"{not (cfg.open_x or cfg.specified)}, periodic_y="
          f"{not (cfg.open_y or cfg.specified)}")
    report("seeded ")
    harness.run_steps(state, cfg, nsteps)
    report(f"N={nsteps}   ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
