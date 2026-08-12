# The halo exchange is bound by transfer COUNT, not bandwidth

Measured on 8x RTX 4090 (GeForce, no P2P, halos route through host RAM). This is
the single biggest remaining lever on multi-GPU throughput and it is a code fix,
not a hardware one.

## The evidence

At 9.63 Mcell, full physics, bit-exact and uncontended:

| config | Mc/card | plain step | exch / step ms | % exch | speedup |
|---|---|---|---|---|---|
| 1 GPU | 9.63 | 474.5 ms | — | — | 1.000 |
| 2x1 P=2 | 4.82 | 547.1 ms | 243.3 / 303.8 | 44.5% | **0.87** |
| 4x1 P=4 | 2.41 | 428.3 ms | 222.2 / 206.0 | 51.9% | 1.11 |
| 4x2 P=8 | 1.20 | 314.9 ms | 202.0 / 112.9 | 64.2% | 1.51 |

**Two GPUs are slower than one.** Compute falls 4.2x from P=1 to P=8 while
exchange falls only 243 -> 222 -> 202 ms. Exchange is nearly flat while the tile
perimeter shrinks — that is the signature of a cost bound by the NUMBER of
transfers, not by bytes.

Confirmed by volume: the exchange runs at **9-23% of the box's measured 52-67
GB/s PCIe ceiling — 4.4x to 11x off the floor** — while issuing **4,736 separate
unpinned copies per step at P=8** (74 carriers x 8 edges x 8 ranks).

Corroborating datapoint at 86.7 Mcell (contended, so directional only): 8x1
exchange is 26% higher than 4x2 against a **59% larger** halo volume. Volume up
59%, cost up only 26% — again count, not bytes.

## The fix, in payoff order

1. **PACK.** One fused kernel gathers every halo plane for a given neighbour into
   ONE contiguous device buffer. One D2H of that buffer, one H2D on the far side,
   one unpack kernel. Hundreds of small copies is precisely how you get 6 GB/s
   out of a 26 GB/s wire.
2. **PIN.** Pageable host staging forces an internal bounce buffer and roughly
   halves achievable bandwidth. Allocate the staging buffers pinned once, reuse.
3. **SEPARATE STREAMS** for D2H and H2D, non-blocking, so they overlap each other
   and compute.
4. **OVERLAP** the next substep's halo with the current interior compute.

## Why the exchange set cannot simply be trimmed instead

It must be every 3-D carrier plus all of `state/` = 74-75 arrays. Excluding the
2-D carriers made 23 carriers diverge, led by `state/mup`, the dycore's 2-D
column-mass prognostic — stale `mup` in the halo drags `p`, `al`, `alt`, `php`
and every microphysics number concentration with it. Dry exchanges only ~24
arrays, which is why the dry curve is not a guide here. See
[[NO-DRY-NUMBERS]].

## What this predicts

Packing and pinning alone should move 8-card efficiency from the measured 18.8%
at 9.63 Mcell (and 52.6% at 154.1 Mcell) substantially toward the PCIe floor.
Treat any projected figure as unmeasured until someone runs it.

## Correctness caveat that applies to every measurement above

Contention on this hardware changes RESULTS, not merely timings: "submission"
transfer ordering is bit-exact on an idle card and **wrong by 3.7e+02** on the
same 3x3 plan when another process shares the GPU. Every timing row must be
stamped with a box-wide CUDA-context count, and `nvidia-smi
--query-compute-apps` returns ZERO for a process that has launched but not yet
created its context — a start-only idle check is not enough.
