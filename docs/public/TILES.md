# Tiling a domain that does not fit on the card

**This page is not [Chunked forecast streaming](STREAMING.md).** That page
is `gpuwm stream PLAN.toml`, which follows an uploading HRRR cycle with
sealed hourly forecast legs. This page is the `[tiles]` table, which runs
one domain out of core on one card. The two share no configuration and no
code path, and either can be used without the other.

A domain's cost on the GPU is roughly a fixed 2.0–3.2 GiB per process plus
541–612 bytes per cell at full physics (measured; see
`tilestream/autoplan.py`).  A 32 GB card therefore holds about 924² × 49 of
`full+MYNN+Noah-MP` resident, and a 12 GB card about 483².  Beyond that,
ArWen used to have nothing to say.

`[tiles]` turns on the out-of-core mode: **the whole domain lives in
pinned host RAM and one tile of it at a time is cycled through the GPU**.
The forecast is not an approximation of the resident one — it is bit-exact
against it, carrier by carrier, at every physics rung, on a real Lambert
projection with real terrain and specified lateral boundaries.

## Turning it on

```toml
[tiles]
mode = "auto"
```

That is the whole configuration for a normal run.  Three modes:

| `mode`  | what happens |
|---------|--------------|
| `"off"` | the default.  The run is exactly the run it was before this feature existed — same call, same function, same bytes, same fingerprint. |
| `"auto"` | the planner (`tilestream.autoplan`) sizes the resident domain against the card.  It fits → **nothing streams**; it does not → the domain streams with the tiling the planner chose. |
| `"on"`  | stream regardless.  For benchmarks and for the bit-exactness proof; a forecast wants `"auto"`. |

`auto` does not stream a domain that fits, because streaming is not free.
Measured tiling tax against the identical resident run (dry, RTX 4090,
1024² × 49, 150 steps): tile 128 → 1.359×, tile 256 → **1.217×**, tile 512 →
1.346×.

The optional keys are for benchmarks and controls, not for forecasts:

```toml
[tiles]
mode = "on"
tile_nx = 512          # pin the tiling instead of asking the planner
tile_ny = 512
nbuffers = 2           # tile i+1's gather overlaps tile i's compute
store = "host"         # "device" keeps the store in VRAM: tiling without transport
write_mode = "ring"    # one store + a few per cent, vs "shadow"'s two stores
host_budget_bytes = 47244640256   # /proc/meminfo lies inside a container
```

A surface that is off must be empty: `mode = "off"` with a `tile_nx` set is a
refusal, not a hint, so a run cannot start streaming because a block was
inherited and a mode was flipped somewhere else.

`halo` is also accepted and **no forecast may set it**.  It is
`10 + 3*time_step_sound//2` and nothing else; a smaller one is silently wrong
*and faster*, which is how that defect hides.  Setting it warns.

## What it does not change

* **The answer.**  A streamed domain and a resident one produce identical
  bytes.  Held by `tilestream/test_gate.py` (51 physics cases, 14 rungs, every
  negative control) and `tilestream/test_join.py` (a real forecast
  configuration, 229 carriers).
* **The restart contract.**  `[tiles]` contributes nothing to the restart
  identity, deliberately: a checkpoint written by a resident run resumes
  streamed and one written streamed resumes resident.  That is the operation
  the mode exists for — a forecast that outgrew its card resuming on the card
  it outgrew — and binding the mode into the fingerprint would refuse it.
* **The loop.**  Output cadence, restart cadence, diagnostics and nest
  coupling are the model's, unchanged.  Streaming replaces one thing: the
  callable that takes one model step.

## The safety gate has to be told where the domain is

The run loop's per-substep gate — `nan_free`, `w_max`, the CFL, the `swdown`
peak and the radiation call count — used to read the resident `DomainState`.
Under a host store that state is never written again: it holds the condition
the store was FILLED from, so the gate was healthy at t=0 and healthy forever.
MEASURED at 672² × 49, tile 168, with the store poisoned through the tile hook
at step 50 and the state never touched: the run completed **200 of 200
substeps reporting `nan_free=True`**, with `w_max` pinned at 17.272114 for
every one of them, while the store ended with **22 579 196 of 22 579 200**
`w` cells non-finite.  The identical poison applied to a resident run's state
raised at substep 50.

Every quantity in that report is a max fold or an OR fold, so it is
associative and is now taken **per tile inside the sweep**
(`tilestream.health_fold`), over the memory the forecast is actually in.
Float max selects an operand and never rounds, so the folded report is
bit-identical to the whole-domain one — measured equal on every one of
250 substeps × 8 fields, with radiation and cumulus firing on both legs.
It costs a fraction of a millisecond per step and it REPLACES a whole-domain
reduction that a streamed run was paying for on data nobody read.

One observer is not folded: `StateHealthValidator` walks every field with
per-field bounds and cannot be folded this cheaply.  Under a host store it is
now explicitly **unarmed** — skipped, counted and warned about — rather than
silently passing, and `health_debug` refuses to start at all, because an
attribution mode that attributes nothing is worse than none.

## What it costs

Host RAM, and it is the binding constraint at every capacity limit measured.
A single store is 32.3 B/cell dry and 279.5 B/cell at the full carrier set;
against a measured 44.14 GiB pinned ceiling that is 5476² × 49 dry.  The
planner refuses with `resource="host"` rather than sizing a pinned store from
someone else's memory, and inside a container it will refuse to guess at all
unless `host_budget_bytes` is given — `/proc/meminfo` there reports the
**host's** RAM (measured: 503 GiB reported against a 241.7 GiB cgroup limit).

## What the capacity numbers on this page do and do not say

Every per-cell cost and card capacity above is measured, and each carries the
rung it was measured at. None of them is a capacity multiplier against a
vanilla resident run, and there is no measured full-physics multiplier to
quote. Two figures that look like one are not: the dry per-cell cost is about
8.5× more generous than the full-physics cost (32.26 against 279.5 B/cell),
and a prediction that dry scaling would merely be pessimistic was measured and
refuted — predicted 91% and 7.3×, measured 52.6% and 4.21×.

`tilestream/NO-DRY-NUMBERS.md` lists the specific values that may not be
quoted and why, and it governs this page.

## Where it is not finished

The seam (`gpuwm.core.streaming.make_stepper`) is wired into both prepared
forecast routes, and a configured domain that must stream gets either a
streamed stepper or a loud refusal.  Building that stepper needs a
domain-specific constructor — the store filled from the prepared state, tile
buffers built with the domain's own physics selectors, the geography
inventoried and the boundary tables windowed per tile.  `streaming.attach`
does all of it and `tilestream/test_join.py` drives it end to end, but the
real-data preparation route does not yet hand one in, so a real-data run with
`mode = "auto"` on an oversized domain currently **refuses with an
explanation** instead of streaming.  Refusing is the deliberate half of that:
the alternative failure mode is a silent resident run that dies at the
allocation the mode was turned on to avoid.
