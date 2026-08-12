# Never quote a dry number. Nobody runs dry.

## BANNED FIGURES — these are dry, do not restate them in any form

    9.98x / 9.5x / 9.47x   streaming capacity vs vanilla (2640^2 vs 858^2)
    1.954x / 1.90x         halo rings vs shadow store
    3.72x @ 46.5%          8-GPU dry scaling

All came from dry stores at 32.26 B/cell against full physics' ~272-279, so every
one is roughly **8.5x too generous in cells**. There is currently NO measured
full-physics replacement for the capacity multiplier. If you need one, MEASURE
it; do not deflate a dry number and quote the result.

What IS measured at full physics: 279.4 B/cell over 229 carriers; a 1536^2 x 49
streamed run at 31.93 GiB pinned; streamed tile ceilings of 384^2 (shipped) and
400^2 (tight layout) on a 12 GB card. None of those is a ceiling-vs-vanilla
ratio.

This list exists because the author of this file quoted 9.5x one hour after
writing the paragraph below explaining why it is wrong.


**Rule: every timing, every scaling curve, every capacity figure and every domain
multiplier that will be reported to a human must be measured at a real physics
rung — `full+MYNN+Noah-MP` (229 carriers, nz=49) unless there is a stated reason
for another.** Dry dynamics is not a product configuration. It has never been run
by a user and never will be.

This is not a style preference. Two curves in this project were measured dry and
both were wrong in ways that mattered:

* **The 8-GPU scaling curve.** Measured dry, it read 3.72x at 46% efficiency and
  was reported as a scaling collapse. Halo exchange is a near-FIXED cost per step
  (measured 24 ms at 10 Mcell, 45 at 40, 85 at 120 — barely tripling while the
  domain grew twelvefold, because GeForce has no working P2P so halos route
  through host memory with a launch cost that does not care about bytes). Dry
  compute is ~4.1 ns/cell, so that fixed exchange swamps it. Full physics is
  ~50 ns/cell — **twelve times more compute to hide the same exchange behind.**
  Substituting physics compute at the same domain predicts ~91% efficiency. The
  dry curve was pessimistic by roughly a factor of two, in the direction that
  would have killed a product decision.
* **Capacity multipliers.** Dry state is 32.26 B/cell; full physics is ~272.
  Every domain-size multiple quoted from a dry measurement is about **8.5x too
  generous in cells**. The 9.98x and the ring's 1.90x both came from dry stores.

## The one legitimate use

Isolating the transport. With almost no compute to hide behind, dry exposes
gather/scatter and exchange overhead cleanly, which is exactly what makes it a
good DIAGNOSTIC and a terrible HEADLINE. If you report a dry number for this
reason, label it `TRANSPORT DIAGNOSTIC, NOT A PRODUCT NUMBER` on the same line.

## Why agents reach for dry, and how to stop needing to

Every fallback so far has been friction, not judgement. Remove the friction:

1. **`freezeH2O.dat` missing** blocks mp28, hence every moist rung, hence cumulus.
   It surfaces as `cumulus physics requires a moist DomainState`, which reads like
   a configuration mistake and is not one. Copy the file from a box that runs full
   physics before doing anything else.
2. **RRTMGP refuses nz >= 41**: `at most 128 layers ... got 140`. The fix is
   `ztop=20000.0`, which `test_gate.PHYSICS_MOIST` already carries. Copy the rung
   definitions from `test_gate` rather than writing your own. Do NOT drop to
   nz <= 37 — that changes the vertical grid and makes the run incomparable to
   every 49-level number in this project.
3. **A full-physics state takes 25-50 s to build** against a couple of seconds for
   dry. Build it once, persist it to disk, and have every configuration load the
   same one — `scaling.py` already does this and it is also what makes digests
   comparable across processes.
4. **Full physics needs more VRAM**, so ceilings are lower and a sweep hits OOM
   sooner. That is the real answer, not a reason to measure something else.

## And the cadence, which is the same mistake wearing a different hat

A physics rung measured in a window where radiation and cumulus never fire is not
a physics measurement. Three false results in this project came from exactly that.
At dt=3 s with radt=12 min and cudt=5 min they fire every ~240 and ~100 steps, and
a radiation step costs 10-18x a normal one — the five radiation steps were **29%
of an entire forecast hour**. PRINT the fire counts on both sides of every
comparison, counted by instrumenting the driver rather than reading the config,
and refuse to report a number if either side shows zero.
