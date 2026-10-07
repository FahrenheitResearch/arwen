# Packed DA member forecasts

The public prepared regional driver has one complete-tree member-leg body.
The serial route and fresh UUID worker processes both call
`tools.da_member_leg.run_member_leg`. The retired `MemberPool` forecast
snapshot protocol is refused because it does not carry the whole model.
`jump_clock` remains a legacy pure reference only and is never a leg join.

Use the public prepared-authority and observation flags with:

```sh
python -m tools.da_cycle_prepared ... --members 32 --solve-device cuda \
  --forecast-members-per-card auto \
  --forecast-device-uuids GPU-... GPU-... GPU-... GPU-... GPU-... GPU-... GPU-... GPU-...
```

Production packing takes any member count on any number of physical cards
(it used to require 32 or 64 members on exactly eight cards, a limit that
named no breakage). Members beyond cards times the per-card width run in
later waves. The forecast-only memory owner prices the largest root plus
child trajectory, its boundary arrays, physics, perturbation workspace,
observations and CUDA process envelope. `auto` chooses four members per card
when four envelopes fit every card after a 4 GiB reserve, otherwise two,
otherwise one; refusal occurs only if one does not fit. A stated width
(`1`, `2` or `4`) that does not fit is refused. Four processes per card require an already active owned CUDA MPS
controller and full physical UUIDs in every child's `CUDA_VISIBLE_DEVICES`.
This uses the proven independent-process packing arrangement. It adds the
DA restart/increment task contract and complete-roster barrier; it does not
use the experimental native member-axis physics path.

The controller runs its unanalysed control first, then launches ordered
member waves. The original seed is `seed + member index`; wave and card slots
cannot change it. Every worker rebinds the ordinary prepared preflight,
checks immutable request and source hashes, restores the complete root and
child restart, applies pending increments once and advances the original
integer clocks. A carried child is built before restore; a newborn child is
built after the parent's pending analysis. Existing child corrections retain
`SINT(analysed parent) - SINT(restored parent)`.

Every result publishes its complete restart, exact clocks, array inventory,
H_Z(x), optional surface H(x), setup geometry and birth metadata before a
completion marker. All members must pass the hash and clock barrier before
the unchanged GPU LETKF runs. A failure or timeout stops only owned child
processes and cannot authorize a partial analysis.

Each child uses `GPUWM_MAPPED_ENGINE_THREADS=12` and an explicit decoder
memory budget. The aggregate decoder bound is at most 160 GiB, with 48 GiB
reserved for other resident work. Available host RAM is checked before each
wave; a smaller available budget reduces the decoder allowance. This is a
decoder bound, not a claim that total process RSS is bounded by that number.
One shared prepared cache is read in place. No member prepares another copy.

Packed cycles publish the complete accepted restart and pending state in a
two-slot recovery ring after each boundary. A failed later forecast or
analysis retains the previous accepted generation. Resume with the ordinary
`--resume-ensemble` door, the next absolute leg number and the remaining
observation/clock sequence. Restarting from the failed wave's partial member
outputs is refused. Successful runs remove automatic recovery scratch;
explicit `--save-ensemble` remains the caller's durable generation.

The cycle report records forecast plus I/O, analysis plus I/O, recovery I/O
and their total. The member wave receipt records the physical placement,
decoder budget, timeout and whether analysis was permitted. No 8-card storm
throughput or scientific gate PASS follows from source presence.

Tests and the bounded node-1 proof are recorded in the lane handoff. The
GPU proof compares every array's dtype, shape and raw bytes, plus exact clocks,
across serial and four-process MPS waves for two observed legs, nonzero LETKF
increments, nested birth, failed-wave source retention and resumed pending-once
continuation. The proof fixture uses a typed in-process setup seam; the
production JSON door cannot choose callbacks or imports. It does not establish
an eight-card 32-member storm result.
