# Full physics does not run out of the box on the 4x5070Ti machine

MEASURED there overnight. Two independent blockers, both cheap to clear, and
neither has anything to do with streaming or multi-GPU. Hit them before sizing a
run, not after.

## 1. `freezeH2O.dat` is missing

Blocks mp28, therefore every moist rung, therefore cumulus outright — the error
surfaces as *"cumulus physics requires a moist DomainState"*, which reads like a
configuration mistake and is not one.

Fix: copy the file from a sibling box that has it (the 4090 at
`ssh -p 40108 root@38.117.87.43` and the 5090 at `ssh -p 52002
root@47.198.198.205` both run full physics, so both have it). Find it with
`python -c "import gpuwm, pathlib; print(pathlib.Path(gpuwm.__file__).parent)"`
and search for the data directory the microphysics loads from.

## 2. RRTMGP refuses the harness column at nz >= 41

    ValueError: RRTMGP radiation supports at most 128 layers including the
                above-model column, got 140

The harness default model top pads to more layers than RRTMGP accepts. Two ways
out, and they are not equivalent:

* `ztop=20000.0` — what `test_gate.PHYSICS_MOIST` already uses, and the right fix.
  It is load-bearing, not decoration: with the 8 km harness default the padding
  overruns the limit. Copy the rung definitions from `test_gate` rather than
  writing your own.
* `nz <= 37` — works, but changes the vertical grid, so a run configured that way
  is not comparable to the 49-level runs everything else in this project quotes.
  If you use it, say so.

## Why this matters for sizing

A ns/cell measured at a rung that silently fell back to dry is not a full-physics
number, and full physics is 2-3x the cost per cell. Sizing a 1 km domain from a dry
rate will produce a domain that cannot finish. Confirm the rung you measured is the
rung you plan to run, and PRINT the radiation and cumulus fire counts to prove it:
at dt=6 s with radt=12 min, radiation fires every 120 steps.

## Everything else on that box is healthy

PCIe gen4 x16 per card verified under load (20/20 samples at 16 GT/s), solo H2D
28.45 GB/s = 90.3% of theoretical, single NUMA node so no binding needed, cgroup
memory limit 241.5 GiB. The card count and the RAM are real; only the physics data
and the model top need attention.
