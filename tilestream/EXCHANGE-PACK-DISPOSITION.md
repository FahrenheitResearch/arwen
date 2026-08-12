# The packed exchange stays the default, and the end-to-end number does not say why

`EXCHANGE-IS-COUNT-BOUND.md` records the measurement that made the exchange
packed. This file records what happened when the pack was measured again
from the integrated tree on a four-card box, and what the result does and
does not establish. The recommendation is unchanged. The reasoning behind
it is narrower than the numbers first appear.

Measured on node 6: 4x RTX 3090 24 GB, sm_86, driver 580.142, CUDA 13.0,
cupy 14.1.1, python 3.12.

## The disposition

The packed exchange remains the default. Nothing measured here argues
against it, and one thing measured here argues for leaving it alone.

## The microbenchmark still holds

At 2 GPUs the pack beats the unpacked reference by about 1.4x on the
exchange itself. That number is real and it is the number
`EXCHANGE-IS-COUNT-BOUND.md` predicts: the pack replaces one link crossing
per field with one per seam, and the exchange is bound by crossing count
rather than by bytes.

## The end-to-end difference is unattributed

At 4 GPUs, end to end, the packed default measured 2 to 4 percent SLOWER
than the direct path. This appeared in the lane's own run and again in the
verdict's independent re-drive, so it is not a single bad sample.

It cannot be the exchange. On this box the exchange is 0.2 to 0.7 percent
of a step. A change confined to the exchange cannot move the step by 2 to 4
percent in either direction, because the whole exchange is smaller than the
difference being explained.

The difference is therefore unattributed. It is step noise, dispatch
overhead, or something else that the exchange arm happens to correlate
with. It is not evidence about the pack. Treating it as evidence about the
pack would be reading a 2 to 4 percent effect out of a 0.2 to 0.7 percent
budget.

The honest statement is that the packed and direct paths are
indistinguishable end to end at this domain size on this hardware, and that
the microbenchmark is where the pack's advantage is visible.

## The exchange share is a floor, not a typical value

The cadence used for these runs fires radiation every step. Production
cadences fire it roughly 50 to 100 times less often. Radiation is a large
per-step cost, so running it every step inflates the step and shrinks the
exchange's share of it.

The 0.2 to 0.7 percent figure is therefore a FLOOR on the exchange share.
At a production radiation cadence the step is much cheaper and the
exchange's share is correspondingly larger. Any decision that depends on
the exchange being negligible has to be re-measured at the cadence it will
actually run.

## The transport under all of these numbers was host-staged

`cudaDeviceCanAccessPeer` is 0 for all twelve ordered device pairs on this
box. `cudaMemcpyPeerAsync` does not fail in that case: CUDA stages the
bytes through host memory and returns success. Every figure above,
including the 1.4x, was measured on a host-staged transport that was
labelled `peer`.

The measurements stand. The label was wrong, and the exchange front door
now prints the actual path per device pair rather than the requested one.
A box with working P2P has not been measured, and the pack's advantage
there is unmeasured rather than assumed.

## At 768 squared the argument is capacity, not throughput

A 768x768x49 full-physics domain needs about 47 GB of carriers. That does
not fit one 24 GB card and the single-card arm exits with an out-of-memory
error.

So at that size the case for four cards is that the problem does not
otherwise run. It is not a speed argument. Speed arguments require a
domain that fits on one card, which puts them at sizes where the exchange
share is small.

## The instrument wrote a completion marker over a failed arm

The scaling driver wrote its `SCALE_DONE` marker unconditionally after its
loop. The 768 arm exited non-zero on the out-of-memory error and the marker
appeared anyway, so a poller waiting on the marker saw a completed run and
a table with an arm silently missing.

Markers are now conditional on rc==0, and a failed arm writes a marker that
names the failure instead. The same discipline is in
`tools/battery/multigpu_crossdevice.sh`: the status is appended to the end
of the log by the run that produced it, so a status cannot outlive its run.
That failure is not hypothetical either. A hand re-run that redirected to
an existing log path left a passing log beside the failing status file of
the run before it, 3.5 minutes apart by mtime.
