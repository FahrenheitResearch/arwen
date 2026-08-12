"""STRUCTURAL ATTACK 1: is any STREAMED CARRIER also an ARENA SLOT?

The sharing scheme's whole safety argument is that the arena holds only
step-local scratch -- written before read inside one step, dead at the tile
boundary.  But ``physics_inventory.carrier_inventory`` streams a set of
``scratch/*`` slots too (that is why forcing ``mp_rainnc`` into the arena is
a valid negative control).  If ANY name is in both sets, then the gather
writes a buffer whose storage every tile shares, and the scheme is wrong for
a reason no answer-watching control is guaranteed to catch.

This asks the question directly instead of trusting the audit table.
"""
import sys

sys.argv = [sys.argv[0]]

from tilestream import harness, physics_inventory as physinv, shared_workspace
from tilestream.test_gate import NZ, SEED, physics_cfg

RUNG = "full+MYNN+Noah-MP"

cfg = physics_cfg(RUNG)
# the shape test_share actually tiles with
tile_cfg = harness.tile_config(cfg, 64 + 32, 48 + 32)
state, driver = physinv.default_builder(tile_cfg, SEED)

carriers = set(physinv.carrier_inventory(state).keys())
arena = shared_workspace.arena_shapes(tile_cfg)
arena_names = set(arena)

# carrier keys look like "fields/xxx" or "scratch/xxx"; arena keys are bare
# slot names.
carrier_scratch = {k.split("/", 1)[1] for k in carriers if k.startswith("scratch/")}
carrier_other = {k for k in carriers if not k.startswith("scratch/")}

overlap = sorted(carrier_scratch & arena_names)

print(f"rung {RUNG}  tile {tile_cfg.nx}x{tile_cfg.ny}x{tile_cfg.nz}")
print(f"carriers total      : {len(carriers)}")
print(f"  of which scratch/ : {len(carrier_scratch)}")
print(f"  of which other    : {len(carrier_other)}")
print(f"arena-admitted slots: {len(arena_names)}")
print()
print(f"INTERSECTION (carrier AND arena): {len(overlap)}")
for name in overlap:
    print(f"   *** {name}  shape={arena[name]}")
if not overlap:
    print("   (empty -- no streamed carrier is backed by shared arena storage)")

# and the converse sanity check: the control's forced slot must really be a
# carrier, or the control proves nothing.
print()
print(f"mp_rainnc is a carrier      : {'mp_rainnc' in carrier_scratch}")
print(f"mp_rainnc is arena-admitted : {'mp_rainnc' in arena_names}")

# Which slots does the audit EXCLUDE that are also carriers?  Those are the
# ones the admission rule is actively protecting.
from gpuwm.core.preflight import scratch_slot_registry, scratch_slot_uses_arena

registry = scratch_slot_registry(tile_cfg, n_lbc_intervals=0)
protected = sorted(s for s in registry
                   if s in carrier_scratch and not scratch_slot_uses_arena(s))
print(f"carrier slots the audit EXCLUDES from the arena: {len(protected)}")
print("   " + ", ".join(protected[:24]))

# the arena's total footprint and MYNN's share, for the 77.6% claim
import math
tot = sum(4 * math.prod(s) for s in arena.values())
mynn = sum(4 * math.prod(s) for n, s in arena.items() if "mynn" in n.lower())
print()
print(f"arena bytes {tot / 2**20:.1f} MiB, of which mynn* "
      f"{mynn / 2**20:.1f} MiB ({100 * mynn / max(tot, 1):.1f}%)")
