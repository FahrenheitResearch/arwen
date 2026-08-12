"""Monolithic per-field digests, so a change to SHARED physics code cannot hide.

The graph gate compares a tiled run against a monolithic run built from the
SAME source tree.  A change that moves both sides identically -- the health
ledger's finite check, the masked_clear that replaced cp.where, the memoised
Noah/RRTMGP constant tables -- is invisible to it by construction.  This runs
the ordinary resident stream path (no tiling, no graph, no ledger installed)
and prints one sha256 per carrier, so the base commit and the change can be
diffed directly.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.environ["ARWEN_TREE"])

import numpy as np


def main() -> int:
    import cupy as cp

    from tilestream import harness, physics_inventory as physinv
    from tilestream.test_gate import PHYSICS_RUNGS, NZ, SEED

    rungs = sys.argv[1].split("|")
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    nx, ny = 96, 80
    out = {}
    for rung in rungs:
        cfg = harness.make_config(nx, ny, NZ, **PHYSICS_RUNGS[rung])
        state, drv = physinv.default_builder(cfg, SEED)
        harness.run_steps(state, cfg, 1)
        harness.run_steps(state, cfg, nsteps)
        inv = physinv.carrier_inventory(state)
        digests = {}
        for k, v in sorted(inv.items()):
            a = cp.asnumpy(v) if isinstance(v, cp.ndarray) else np.asarray(v)
            digests[k] = hashlib.sha256(
                np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
        whole = hashlib.sha256(
            "".join(f"{k}:{d}" for k, d in sorted(digests.items())).encode()
        ).hexdigest()[:24]
        sc = physinv.carrier_scalars(state)
        out[rung] = {"whole": whole, "fields": digests,
                     "scalars": json.loads(json.dumps(sc, default=str))}
        print(f"{rung:24s} {whole}  ({len(digests)} carriers)", flush=True)
        del state, drv
        cp.get_default_memory_pool().free_all_blocks()
    with open(sys.argv[3], "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
