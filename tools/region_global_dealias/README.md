# region-global-dealias

**Radar velocity dealiasing for a single sweep, as a dependency-free
WebAssembly module.** Flat typed arrays in and out, ~207 KB (76 KB gzipped),
no imports, no `wasm-bindgen`, no consumer build step. Runs on Node,
Deno, Bun, Cloudflare Workers, and in browsers — and as a plain Rust crate.

It solves **one sweep at a time**, so it suits feeds that deliver tilts
individually rather than as whole volumes.

```js
import { createDealiaser } from "region-global-dealias";

const dealiaser = await createDealiaser();

const result = dealiaser.dealias({
  velocity: observed,   // Float32Array, rows * gates, m/s, NaN = no data
  azimuth: azimuths,    // Float32Array, one degree value per ray
  nyquist: 24.1,        // m/s — a number, or one per ray for split cuts
  rows: 720,
  gates: 1192,
});

result.velocity; // Float32Array of unfolded velocities
result.stats;    // { gatesTotal, gatesFinite, gatesModified, maxAbsFold, wraps }
```

Install straight from GitHub — there is nothing to build:

```bash
npm install github:FahrenheitResearch/region-global-dealias
```

## What the algorithm is

Radar cannot distinguish a fast wind from one that wrapped past the Nyquist
velocity, so a 30 m/s outbound wind on a ±24 m/s radar is reported as 18 m/s
inbound. Dealiasing puts the wraps back.

This is a Rust port of Py-ART's `dealias_region_based` (Helmus & Collis 2016).
The sweep is segmented into regions that are internally fold-free, and the
whole region network is then unfolded *jointly* — merging the strongest region
pair first and re-weighting every remaining connection with what that merge
established, rather than committing each boundary from local information alone.
It also links regions separated by up to 100 gates of no-data, so a field
broken up by clutter gaps still resolves as one system.

See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for what this port changes
relative to upstream Py-ART, and for the BSD-3-Clause terms that come with it.

## Accuracy

Verified against BowEcho's production implementation of the same algorithm on
**86 real NEXRAD sweeps across 7 volumes — 16.6 million compared gates, zero
differing folds.** The regimes covered are clear air (empty and filled),
supercell, derecho, blizzard, and two hurricane eyewalls.

The engine is deterministic: identical input always produces byte-identical
output, on every run and every thread. A pinned golden checksum in the test
suite fails if that ever changes.

Velocities come back as exact `f32`. A gate the solver did not move is returned
bit-identically to what you supplied; a gate it did move differs by a whole
number of Nyquist intervals and nothing else.

## Performance

Measured with this package, in Node 22 on Linux, one thread, best-of-3 per
sweep. Every velocity sweep of each volume was solved; "worst sweep" is the
slowest single one, which is the number that matters for a per-tilt feed.

| Volume | Regime | Nyquist | Sweeps | Worst sweep | Whole volume |
|---|---|---|---:|---:|---:|
| KTLX 2013-05-20 | Moore EF5 supercell | 26.1 m/s | 14 | **10.5 ms** | 64 ms |
| KDLH 2020-01-20 | Clear air, near-empty | 11.6 m/s | 5 | 11.8 ms | 34 ms |
| KEAX 2026-06-09 | Derecho / QLCS | 24.1 m/s | 17 | 18.5 ms | 144 ms |
| KBOX 2022-01-29 | Blizzard, stratiform + LLJ | 24.0 m/s | 15 | 19.5 ms | 76 ms |
| KMBX 2026-01-20 | Clear air, filled | 11.6 m/s | 8 | 19.9 ms | 80 ms |
| KTBW 2022-09-28 | Hurricane Ian eyewall | 24.1 m/s | 19 | 20.8 ms | 164 ms |
| KUDX 2026-03-05 | Clear air, filled, low SNR | 11.8 m/s | 8 | **23.2 ms** | 96 ms |

So: **10 ms to 23 ms per sweep**, and a whole volume serially in 0.03–0.16 s.
For a feed delivering tilts seconds apart that is a rounding error. Use
[the pool](#worker-pool-nodejs) if you need to chew through whole volumes or
many sites at once.

### Against Py-ART

Same sweep, same defaults, same machine, dealias call only — decode excluded on
both sides, and Py-ART given an untimed warmup. Both are single-threaded.

| Sweep | Py-ART 2.2.5 (Python) | This port (wasm) | |
|---|---:|---:|---:|
| KEAX derecho, 0.86 M gates | 1,531 ms | **19.2 ms** | **80× faster** |
| KUDX clear air, 0.86 M gates | 2,085 ms | **26.4 ms** | **79× faster** |

With **zero differing folds** on all 720,774 compared gates. Same answers,
about eighty times faster, and no Python runtime to deploy.

Some of that is the language, but most of it is not: upstream Py-ART still
rescans its whole edge list on every merge, which is quadratic in the region
count. Before that was fixed here the gap was ~3×.

What drives the cost is **how much of the scene has returns**, not the Nyquist
velocity. The three clear-air volumes above all sit near 11.6 m/s and span
12 ms to 23 ms — the empty one is the cheapest volume in the set and the filled
ones are the most expensive. Low Nyquist alone is not the hard case; low
Nyquist *with a full boundary layer* is.

Native Rust is about 1.5× faster than wasm here (the 86-sweep corpus solves in
444 ms native vs 659 ms in wasm), so using the crate directly buys less than
you might expect.

### Hostile input

Cost scales with the number of distinct velocity regions, and real weather has
few of them relative to gate count. Data that is *not* weather — uniform noise,
a mis-scaled decode — produces orders of magnitude more regions and is
correspondingly slower: a synthetic pure-noise sweep of 0.86 M gates takes
~1.0 s versus ~20 ms for real data of the same size. It completes rather than
hanging, and `rows * gates` is capped at 2²² (4,194,304) so memory stays
bounded, but **if you accept sweeps from an untrusted source, run them through
[the pool](#worker-pool-nodejs) with your own timeout** rather than on the
request path.

## API

### `createDealiaser(options?) => Promise<Dealiaser>`

With no options the module is decoded from bytes embedded in the package — no
fetch, no filesystem, no bundler plugin.

| Option | Use |
|---|---|
| `module` | A pre-compiled `WebAssembly.Module`. Required on Cloudflare Workers. |
| `wasm` | Raw module bytes, if you would rather load `dealias.wasm` yourself. |

### `dealiaser.dealias(input, out?) => { velocity, stats }`

| Field | Type | Notes |
|---|---|---|
| `velocity` | `Float32Array` | `rows * gates`, row-major by ray. Non-finite = no data. |
| `azimuth` | `Float32Array` | Degrees, one per ray. Any range; normalized internally. |
| `nyquist` | `number` or array | m/s. One value, or one per ray for split-cut / dual-PRF. |
| `rows` | `number` | Rays. |
| `gates` | `number` | Gates per ray. |

Pass `out` (a `Float32Array` of `rows * gates`) to write in place and avoid an
allocation per sweep.

Returned `stats`:

| Field | Meaning |
|---|---|
| `gatesTotal` | `rows * gates`. |
| `gatesFinite` | Gates that had data. |
| `gatesModified` | Gates moved by at least one Nyquist interval. |
| `maxAbsFold` | Largest fold applied anywhere on the sweep. |
| `wraps` | Whether the rays closed a full circle — see [Sector feeds](#sector-feeds). |

Gates with no data pass through as `NaN`. A ray whose Nyquist is missing, zero,
or negative falls back to the sweep median; if there is no usable Nyquist
anywhere, velocities pass through untouched rather than erroring.

`rows * gates` is capped at 4,194,304 — about 4.9× a NEXRAD super-resolution
sweep. Larger requests throw rather than allocate.

Bad input throws `DealiasError`, which carries a `code` (the module's negative
return code, or `null` when validation failed in JavaScript).

### Upgrading from v0.1

The existing `dealias()` API and its region-global output are unchanged. Pin
the new release as usual:

```bash
npm install github:FahrenheitResearch/region-global-dealias#v0.2.0
```

No application changes are required unless you want the new refinement path.
To opt in, call `dealiasRift()` and supply the physical gate geometry shown
below. Keeping `dealias()` preserves the previous behavior exactly.

### Opt-in gate-resolution refinement (`dealiasRift`)

`dealias()` remains the stable region-global solver and produces the same
result as before. `dealiasRift()` is an additive, opt-in path for the harder
case where one connected region contains two different Nyquist branches. It
runs region-global first, then considers small gate-resolution proposals. An
automatic single-sweep proposal is applied only when an independent
wrapped-vortex fit selects the same branch and a bounded fusion cut lowers the
local energy; otherwise the region-global result is returned unchanged.

```js
const result = dealiaser.dealiasRift({
  velocity: observed,
  azimuth: azimuths,
  nyquist,
  rows,
  gates,
  rift: {
    firstGateMeters: 2125, // center range of gate zero
    gateSpacingMeters: 250,
  },
});

result.velocity;   // refined velocity, or the unchanged region-global result
result.folds;      // final absolute fold per gate
result.confidence; // nonzero only where a refinement was accepted
result.reasons;    // per-gate trigger, anchor, acceptance, or abstention flags
result.stats;      // ordinary stats plus ROI/refinement diagnostics
```

Physical gate geometry is required when automatic single-sweep refinement is
enabled. Set `automaticSingleSweep: false` for reference-only refinement. Up
to four already-projected reference fields may be supplied through
`context.references`, with kinds `caller`, `temporal`, `vertical`, or
`environmental`. Reference-driven proposals use their supplied quality and a
separate bounded fusion step rather than the single-sweep vortex fit.
Conflicting reference branches abstain rather than depending on input order.
Non-finite reference gates are absent; optional quality values run from 0 to
255.

The automatic detector is deliberately conservative. In the current hard-case
corpus it evaluated 80 low-level cuts from 20 NEXRAD volumes and accepted five
local corrections (442 gates), while leaving both controls and every tested
hurricane cut unchanged. Those results demonstrate bounded behavior on that
corpus, not universal meteorological truth. Mixed per-ray Nyquist transitions
currently abstain from the automatic vortex fit; reference-driven refinement
still uses each ray's resolved Nyquist interval.

On a Ryzen 9 9950X3D, native release builds with decode excluded completed the
five accepted full-sweep refinements in 58-97 ms median per cut. The ordinary
no-candidate path remains about 12 ms median; RIFT pays the extra model and cut
cost only after the strict trigger authorizes a compact ROI.

`reflectivity`, `spectrumWidth`, and `rhoHv` are accepted as reserved v1
context fields but do not currently authorize or reject an automatic proposal.

### `dealiaser.dispose()`

Releases the module-side scratch buffers. The instance is spent afterwards.

**One instance is not safe for concurrent use** — it owns one set of scratch
buffers. Use one per worker, or the pool.

## Worker pool (Node.js)

The solver is single-threaded, so volume throughput comes from running sweeps
in parallel. Each worker holds its own WebAssembly instance; nothing is shared.

```js
import { createDealiaserPool } from "region-global-dealias/pool";

const pool = await createDealiaserPool();       // one worker per core, less one

const sweeps = await Promise.all(
  volume.map((sweep) => pool.dealias(sweep)),
);

await pool.destroy();
```

Input arrays are copied to the worker by default. Pass `{ transfer: true }` to
hand the buffers over instead — faster for big sweeps, but it detaches your
arrays, so only do it if you will not read them again.

If a worker dies, its in-flight job rejects rather than hanging, and the pool
keeps running on the survivors. An input that cannot be sent rejects that one
job without costing the worker.

## Using it from Rust

The crate is the same code without the FFI layer, and has no dependencies:

```toml
[dependencies]
region-global-dealias = { git = "https://github.com/FahrenheitResearch/region-global-dealias" }
```

```rust
use region_global_dealias::solver;

// `observed` is rows * gates, row-major by ray; NaN means no data.
let unfolded = solver::dealias_sweep(&observed, &nyquist, rows, gates, &azimuths);
```

`solver::region_folds` returns the integer fold per gate if you would rather
apply them yourself, and `solver::sweep_wraps` exposes the seam test.

## Native C ABI

Cargo also builds a shared library with a plain C ABI. The complete constants,
structure layouts, nullability rules, and function declarations are in
[`region_global_dealias.h`](region_global_dealias.h). Existing hosts can keep
calling `bw_dealias`; opt-in hosts feature-detect `bw_rift_api_version() == 1`
and call `bw_dealias_rift_v1`.

The legacy ABI version remains 1. RIFT is versioned separately so additive
refinement work does not break existing native consumers.

## Runtimes

| Runtime | Works | Notes |
|---|---|---|
| Node 18+ | Yes | Nothing to configure. `/pool` needs Node. |
| Deno, Bun | Yes | Nothing to configure. |
| Browsers | Yes | Run it in a Web Worker — see `demo/index.html`. |
| Cloudflare Workers | Yes, with one line | See below. |

### Cloudflare Workers

workerd **blocks compiling WebAssembly at runtime**, so the embedded-bytes path
cannot work there. Import the module as an asset and hand it over:

```js
import module from "region-global-dealias/dealias.wasm";
import { createDealiaser } from "region-global-dealias";

const dealiaser = await createDealiaser({ module });
```

Wrangler compiles the `.wasm` at build time, which is what makes this legal.

### Bundlers

The default path embeds the module in JavaScript specifically to sidestep the
usual `.wasm` asset-resolution breakage, so most setups need no configuration.

`/pool` is the exception: it resolves `pool-worker.mjs` at runtime relative to
its own URL, which a bundler will not follow. Bundling a Node server that uses
the pool means marking `region-global-dealias` external, or copying
`pool-worker.mjs` into the output. The main entry has no such constraint.

## Sector feeds

`stats.wraps` reports whether the rays closed a full circle. When they did, the
seam between the last and first ray is available as an adjacency; when they did
not, the sweep is solved without it. Regions that straddle 0°/360° in a sector
feed therefore resolve independently, which can leave a seam artifact.

The check estimates ray spacing as `360 / rows`, so it assumes your rays span
the circle. A real sector at native ray spacing is classified correctly (a 90°
NEXRAD super-res sector is 180 rays and reads as open). A synthetic sweep with a
handful of rays crammed into a narrow arc can read as closed when it is not — if
you subsample rays, do not trust this flag.

If your feed delivers partial sectors that accumulate into a sweep, prefer
buffering until the sweep closes.

## What this does not do

- **It does not decode radar files.** It takes velocity arrays you already
  have. Level II decoding is a separate problem (bzip2-chunked Archive II,
  ODIM_H5, CfRadial…) and is not in this project.
- **It is not a whole-volume solver.** Solving every velocity tilt of a volume
  jointly, with the previous volume and an environmental wind profile as
  anchors, is strictly more information than one sweep carries and does better
  on ambiguous cases. That is a different, much larger piece of software.
- **It does not fix everything.** Isolated small regions with no strong
  connection to the rest of the sweep can still land on the wrong branch. This
  is an open problem across the field, not a quirk of this port — NOAA's
  operational ORPG fails the same cases.
- **On clear-air, low-SNR sweeps, treat the numbers with care.** Much of what
  looks like aliasing there is velocity noise, and "folds removed" measures the
  noise as much as the wind.

## Building from source

The committed `dealias.wasm` and `wasm-inline.mjs` are build outputs, so
consumers never need a Rust toolchain. To regenerate:

```bash
bash build.sh          # rebuild the artifacts
bash build.sh --check  # verify the committed ones are current
```

The build asserts the module has zero imports and exports exactly what the
wrapper expects, so a regression that would break one runtime fails the build
instead.

Tests:

```bash
cargo test    # solver + FFI
node --test   # JavaScript wrapper + pool
```

## License

`(MIT OR Apache-2.0) AND BSD-3-Clause`. This project's own code is MIT or
Apache-2.0, at your option. The solver is a derivative of **Py-ART**, which is
**BSD-3-Clause** — those terms travel with it and cannot be removed, so the BSD
conditions apply on top of whichever option you choose.

Redistributing this project, in source or compiled form, means shipping
[`PYART-LICENSE.txt`](PYART-LICENSE.txt) with it. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full notice, the list
of modifications this port makes, and the citation to use in published work.
