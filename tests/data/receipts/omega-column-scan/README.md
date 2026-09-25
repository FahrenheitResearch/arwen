# Omega column-kernel receipts (2.7.6)

Receipts behind docs/manual/04-gpu-numerics.md section 4.3, CHANGELOG 2.7.6 and the
phase-2 pin ledger in tests/test_coriolis_map.py. All were produced on node-1
(RTX 5070 Ti, CUDA 13, cupy-cuda13x 14.2.0, Python 3.14.4) by
tools/omega_scan_receipt.py and tools/recapture_phase2_pin.py.

| file | what it is |
|---|---|
| validate-64x64x32-mp10.json | one-call Omega and 60-step field readings, legacy construction versus column kernel, Morrison (mp=10), seed 20260731 |
| validate-64x64x32-mp0.json | the same on the dry core (mp=0) |
| bench-mp10.json | whole-step timing, both arms alternating on one state, 8 repetitions of 20 steps, medians, at 250x200x49, 320x256x49 and 480x384x49 |
| kernel-480x384x49.json | the routine alone, retired construction versus column kernel |
| phase2-pin-shipped-vs-base.json | tests/data/phase2_step_regression.npz as shipped on 2026-09-03 against a capture at the 2.7.5 tip c36f4c1f1: the drift the pin already carried before this change |
| phase2-pin-base-vs-lane.json | capture at c36f4c1f1 against capture at the lane tip on the same card: the move attributable to the column kernel |
| phase2-pin-readings.json | per-entry readings of the re-pinned capture |

The two validate receipts record `environment.git_commit`
6880df717622b7bfe113a7081dc13f8d8eccc5d2, a node-1 working commit that is not on the
branch of record. The kernel they exercised is the one at 72f18213d (no later commit on
the branch touches gpuwm/core/dycore.py) and their field set is the moisture-complete
form of the receipt tool at 4ef7c4ac1. The bench receipt records 72f18213d. The next time
a card is used for this lane, regenerate the validate receipts at a branch-of-record hash.
