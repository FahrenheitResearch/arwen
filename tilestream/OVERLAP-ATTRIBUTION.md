# Where the streamed step's wall clock goes, measured before the overlap build

Task #166 inherits a receipt -- transfers fully exposed at ~0.628 s/step,
16-19% end to end, `asyncEngineCount == 1` -- measured on a box this lane no
longer has.  The build targets what IS, so this file is the re-measurement on
the box the build will be proven on, with the three named candidates (the
`tile_hook` bind, the ring waits, `_advance_clock`'s readback) measured
rather than suspected.  Instrument: `tilestream/overlap_attrib.py`; raw
per-step JSON in `tilestream/evidence/overlap-attrib/`.

## The venue, stated first because the ratios depend on it

Node 4: RTX 5080 16 GB, sm_120, CUDA 13.1 (runtime 13.0.20), cupy-cuda13x
14.1.1, python 3.12.  The gpu-mutex was held by this lane
(`/workspace/gpu-mutex/holder`) and `nvidia-smi` showed zero other compute
processes for every number below.

* **`asyncEngineCount == 2`, not 1.**  The old box's central constraint does
  not hold here: H2D and D2H each have an engine, and a raw probe confirms
  both run concurrently at full rate (2.34 GB/s H2D + 3.64 GB/s D2H =
  6.55 GB/s aggregate bidirectional).
* **The PCIe link is narrow.**  2.34 GB/s pinned H2D against the ~57 GB/s
  the tilestream receipts measured on the dev box; `nvidia-smi` reports the
  link at gen1 x16 (max gen5 x16).  Exposure FRACTIONS below are therefore
  upper bounds relative to a healthy x16 box -- the structure (what is
  exposed, what can hide, what is count-bound) transfers; the percentages
  do not.

## What ran

The reference case (ERA5 route, mp10 + YSU + Noah + RRTMG, specified
boundaries), one root domain at the case's own coverage, three spacings,
streamed with a pinned host store through the same `tilestream.driver
.TiledRun` internals `gpuwm.core.streaming.attach` drives, ONE sweep per
model step exactly as `execute_experiment` steps a streamed domain.  Tiling
and buffers are the production planner's own choice (`[tiles] mode = "on"`,
nothing pinned).  Cumulus off (all three arms are convection-allowing
spacings); nothing else changed from the case's physics.  CUDA events on
every gather / scatter / ring save / ring patch / `dycore.step`, on the
operation's own stream; perf_counter on the host-side candidates; exposure
is interval arithmetic, `|union(transfer) - union(compute)|`, never a
subtraction of sums.  Instrument overhead vs `--control` on the same
window: 1.6% / 0.9% / 0.1% (arms a/b/c quiet steps).

Sizes are bounded by the PREPARE stage, not the forecast: the slabbed
ingest's device high-water measures ~15-16 kB per mass cell here (5.55 GiB
at 648x576; realcase.py's 4090 receipt says 15.55 GiB at 960x960), so this
15.5 GiB card prepares at most ~0.9 M mass cells -- a 1296x1152 arm OOMed
in `init_at_rest` and was re-sized.  The streamed FORECAST's ceiling is far
higher; on this card the reference case's streaming envelope is
prepare-limited, which is its own finding (#158's work is the door).

## Attribution (quiet steps = no radiation fire; per-step means)

| | arm a | arm b | arm c |
|---|---|---|---|
| grid | 648x576x49 @ 4 km | 864x768x49 @ 3 km | 960x864x49 @ 2.7 km |
| planner's tiling | 4 tiles 324x288, nbuf 3 | 4 tiles 432x384, nbuf 2 | 6 tiles 320x432, nbuf 2 |
| wall ms | 4955 | 8331 | 11591 |
| compute busy ms | 2691 | 5965 | 8298 |
| transfer busy ms (union) | 4715 | 8009 | 11343* |
| **transfer exposed ms** | **2265 (46%)** | **2366 (28%)** | **3292 (28%)** |
| h2d / d2h ms | 2339 / 2046 | 3333 / 3645 | 4703 / 5169 |
| ring save / patch ms | 564 / 1183 | 367 / 1260 | 458 / 2581 |
| geography h2d ms | 105 | 925 | 961 |
| gathered / scattered GB per step | 4.33 / 3.38 | 7.26 / 6.01 | 9.24 / 7.52 |
| lazy tile_hook / set_scalars / _advance_clock ms | 0.005 / 0.010 / 0.009 | 0.008 / 0.012 / 0.011 | 0.014 / 0.018 / 0.011 |
| radiation step wall | 36.1 s (compute 34.9) | 61.5 s (59.7) | 74.3 s (72.6) |
| end-to-end exposed share (radt cadence in window) | 29% | 19% | 21% |

*arm c transfer-busy quoted from the all-steps mean; quiet-only differs <2%.

Where the exposure sits (arm c, instrument v3): fill 807 ms + drain 682 ms
= 45% of the exposed time is the pipeline's ends; the mid-sweep remainder
is ring_patch (1796 ms exposed) and d2h (2484 ms exposed, which contains
the drain).  Exposed-by-category: d2h 2484, ring_patch 1796, geo_h2d 729,
h2d 687, ring_save 124.

## The three named candidates, measured

* **`tile_hook` bind -- split verdict, and the split is the finding.**  The
  LAZY streaming bind this harness (and run_case_hrrr) uses costs
  0.005-0.014 ms/step: nothing.  The EAGER `gpuwm.ingest.lateral_bc
  .attach_lateral_boundaries`, which is what PRODUCTION `attach()` installs
  via `make_tile_hook`, costs 27-63 ms per bind on a real tile buffer with
  the same windowed tables -- at 6 binds/sweep (arm c) that is ~0.3 s per
  step, ~2.6% of the quiet wall, host-blocking, growing with interval
  count, tile count and boundary size.  The lazy variant
  (`attach_streaming_lateral_boundaries`) already exists; production attach
  simply does not use it.
* **ring waits -- confirmed, but as COPIES more than waits.**  ring_patch
  leaves 1.7-1.8 s/step exposed at arms b/c (15% of the quiet wall) and
  runs at 0.33 GB/s effective (0.86 GB in 2.58 s) against a 2.34 GB/s link:
  count-bound band copies serialized by the WAR/patch event chains, exactly
  the shape EXCHANGE-IS-COUNT-BOUND predicts.
* **`_advance_clock` readback -- exonerated.**  0.011 ms/step; it reads
  host-side driver counters after the sweep's own synchronize.
  `set_carrier_scalars` likewise (0.02 ms).

## The ceiling on THIS card

Two copy engines, one per direction.  Perfect overlap bounds the quiet step
at `max(compute, H2D-engine load, D2H-engine load)` (H2D engine = h2d + geo
+ ring_patch; D2H = d2h + ring_save):

| | arm a | arm b | arm c |
|---|---|---|---|
| engine loads h2d / d2h ms | 3627 / 2609 | 5519 / 4012 | 8246 / 5627 |
| perfect-overlap wall ms | 3627 (h2d-bound) | 5965 (compute-bound) | 8298 (compute-bound) |
| **speedup ceiling** | **1.37x** | **1.40x** | **1.40x** |
| overlappable of the exposed ms | 1329 of 2265 | 2366 of 2366 (all) | 3292 of 3292 (all) |
| link-bound residue ms | 936 | 0 | 0 |

At arms b and c the step is compute-bound at the ceiling: EVERY exposed
millisecond is overlappable in principle, worth ~1.4x on the quiet step and
~1.25x end to end at the production radiation cadence.  At arm a the
crippled link itself binds and 41% of the exposed time cannot be scheduled
away, only shrunk (fewer/larger copies) or out-run (wider link).  On a
healthy gen4/gen5 x16 box the same byte counts take ~10-20x less link time:
every arm is then compute-bound at ceiling, and the overlap build recovers
the whole exposed term.  An `asyncEngineCount == 1` card loses only
H2D-with-D2H concurrency, not copy-with-compute -- the levers below survive
the engine count.

## Against the inherited receipts

Directionally reproduced, absolutely different.  End-to-end exposed share
here is 19-29% against the receipts' 16-19%; per-step exposed is 2.3-3.3 s
against 0.628 s, because this box's link is ~20x slower than the box the
receipts came from.  The engine-count premise is dead on this card
(2, not 1).  The build must therefore target mechanisms that are
link-speed- and engine-count-independent, which the ranked list below is.

## Ranked fix list

1. **Cross-step pipelining at the sweep seam** (fill 0.8 s + drain 0.7 s +
   the d2h tail; ~45% of the exposed time at arm c).  A sweep today ends
   with stream syncs + `deviceSynchronize`, so the last tiles' scatters
   drain with no compute to hide under and the next step's first gathers
   fill against an idle GPU.  Let step N+1's gathers and first tile's
   compute issue while step N's scatters drain -- the clock epilogue
   (`_advance_clock`, 0.01 ms) and the `on_sweep` seam are the only true
   barriers, and both are cheap.  This is the single biggest lever and the
   one the per-step `StreamedDomain.__call__` -> `sweep(1)` structure
   currently forbids.
2. **Coalesce the D2H side: scatter + ring patch** (PACK-class).  d2h runs
   at 1.45 GB/s effective and ring_patch at 0.33 against a 3.64/2.34 GB/s
   link -- count-bound, not byte-bound.  Fewer, larger copies cut the
   engine load ~2.5x, which converts arm-a-class geometries from link-bound
   to compute-bound and shrinks what the overlap has to hide everywhere.
3. **Ring patch ordering.**  Patches wait on other tiles' save events
   mid-sweep (1.8 s exposed at arm c even though compute exists to hide
   under).  Issue patches on the next tile's gather stream (they are that
   tile's INPUT) rather than the producing tile's compute stream, so the
   event wait overlaps compute instead of gating the scatter.
4. **Geography prefetch/caching** (0.7 s exposed, 2.5 GB/step at arm c).
   Input-only, known ahead of the sweep: prefetch it with the carrier
   gather at depth, and prefer nbuffers that reduce buffer-tile changes
   when VRAM allows (arm a at nbuffers 3 paid 105 ms; arms b/c at
   nbuffers 2 paid ~950 ms).
5. **Switch production `attach()` to the lazy boundary bind** (~0.3 s/step
   at arm c, host-blocking).  `make_tile_hook` -> eager
   `attach_lateral_boundaries` re-uploads every interval on every
   buffer-tile change; `attach_streaming_lateral_boundaries` swaps host
   tables into one packed slot and is measured at ~0.01 ms.  One-line
   class of change, already proven in the realcase harness.
6. **Not levers:** `_advance_clock`, `set_carrier_scalars`, the lazy hook
   itself -- all measured below 0.02 ms/step.  Chasing them would be
   optimizing the noise floor.

Also carried out of this measurement, not an overlap item: the PREPARE
high-water (~15-16 kB/mass cell) is the binding constraint on how large a
reference-case domain this card can stream at all.
