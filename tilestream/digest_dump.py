"""Dump SHA-256 per carrier for the MONOLITHIC reference and for the
DEFAULT (unshared) tiled path, so base and branch can be compared to each
other rather than each to its own in-process reference.

Both gates passing does not by itself prove the answer is unchanged: the
gate compares a tiled run to a reference computed in the SAME process, so a
change that moved both would pass on both trees.  This pins the actual
bytes.  Run it in each worktree and diff the JSON.
"""
import json
import sys
import warnings

import cupy as cp

from tilestream import driver, harness, physics_inventory as physinv
from tilestream import spec as tspec
from tilestream.test_gate import NZ, SEED, physics_reference

RUNG = "full+MYNN+Noah-MP"
NX, NY, NSTEPS = 96, 80, 3
TILE_NX, TILE_NY = 48, 40

cfg, start, start_scalars, ref_arrays, ref_scalars = physics_reference(
    RUNG, NX, NY, NSTEPS, nz=NZ, seed=SEED)
mono = physinv.field_digests(ref_arrays)

halo = harness.halo_radius(cfg)
specs = tspec.plan_tiles(NX, NY, TILE_NX, TILE_NY, halo, True)
store = {k: cp.asarray(v) for k, v in start.items()}
scalars = dict(start_scalars)
report = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    driver.run_tiled(
        store, cfg, TILE_NX, TILE_NY, halo=halo, nsteps=NSTEPS, nbuffers=2,
        report=report, inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
        tile_state_factory=lambda tc: driver.make_physics_tile_state(tc),
        scalars=scalars)
cp.cuda.runtime.deviceSynchronize()
tiled = physinv.field_digests(physinv.carrier_inventory(store))

out = {
    "rung": RUNG, "nx": NX, "ny": NY, "nz": NZ, "nsteps": NSTEPS,
    "tiles": len(specs),
    "radiation": scalars.get("call_counts", {}).get("radiation"),
    "cumulus": scalars.get("call_counts", {}).get("cumulus"),
    "monolithic": mono,
    "tiled_default_path": tiled,
    "scalars": {k: str(v) for k, v in sorted(scalars.items())},
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, sort_keys=True)
print(f"{len(mono)} monolithic digests, {len(tiled)} tiled digests, "
      f"radiation {out['radiation']}x cumulus {out['cumulus']}x -> {sys.argv[1]}")
