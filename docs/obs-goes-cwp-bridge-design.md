# Design note: the GOES CWP bridge (`rw_goes`)

**Status: the bridge is IMPLEMENTED-UNVERIFIED. The assimilation path is
DESIGN ONLY.** These are two different things and the distinction is the
whole point of this header: nothing here means GOES is assimilated.

**What exists** (`tools/rustwx/crates/rw-goes`, built on the patched
`rw-sat` from `evidence/upstream/rw-sat-cloud-products/`): the `rw_goes`
binary, with `list` and `fetch` against the live `noaa-goes19` bucket,
`cwp` (the 2 km `gpuwm-obs.goes-cwp.v2` pack), `cloud-top` (the 10 km
`gpuwm-obs.goes-cloudtop.v2` pack), and `verify` for both families. Both
pack families have been produced from live granules and re-proved by
`verify`; 32 unit tests are green; the crate builds
`--release --locked --offline`.

*Unverified* means exactly that: no obs-skill number, no comparison
against an independent CWP implementation beyond the DQF-count
cross-check, and no scientific review of the PROVISIONAL ice/mixed-phase
coefficients. Bytes move correctly; whether the observable is right for
assimilation is not yet established.

**What does not exist:** any use of these packs by the DA cycle. The
Python reader, the regrid to the model grid, superobbing, obs error, and
every line of the assimilation path below are DESIGN ONLY — unwritten.
No forecast has ever seen a GOES CWP observation.

## What it is

The satellite twin of `rw_nexrad`: a small Rust front-door crate under
`tools/rustwx/crates/rw-goes` that acquires ABI L2 cloud granules,
decodes them through the patched `rw-sat`, derives CWP, and emits one
flat pack per scan that the Python obs layer reads with `json` +
`numpy.frombuffer` and nothing else. Same vendor closure discipline as
`rw_nexrad`: `rw-sat` (patched) + already-vendored deps,
`cargo build --release --locked --offline` must hold, satellite ids and
sectors are data throughout — nothing branches on a particular case.

## Subcommands (cloned from `rw_nexrad`)

* `list` — `product_hour_prefix` listings for a (satellite, sector,
  product set, window); JSON receipt.
* `fetch` — anonymous S3 download into a content-addressed cache,
  sha256 recomputed from bytes on disk, atomic temp+rename via the
  rw-store pattern; JSON receipt.
* `cwp` — the workhorse: take one scan time; require the COD + CPS +
  ACTP granules of that exact scan start; decode each through
  `read_cloud_product_field` (DQF-gated, counts kept); **refuse unless
  the three fixed grids are bit-identical** (the same assertion the
  upstream fixture test makes — combining planes across grids would be
  fabrication); derive `cloud_water_path_plane`; write the pack.
  ACHA/CTP are accepted on the flags and refused wherever they are not
  on the trio's grid — which at CONUS is always; see "Two packs per
  scan" below.
* `cloud-top` — the same scan's ACHA/CTP on **their** fixed grid, as a
  second pack (`gpuwm-obs.goes-cloudtop.v2`). The vertical-placement
  inputs the operator spec wants, paired to the CWP pack rather than
  folded into it.
* A granule set that cannot be completed (COD published, CPS missing)
  is a hard error with the keys named, never a partial pack.

## The pack: `gpuwm-obs.goes-cwp.v2`

The `GPWMRDR1`-style layout carried over unchanged: 64-byte
little-endian header (new magic `GPWMGOES`, version u32, meta_len),
JSON metadata block, one contiguous `<f4` payload.

Metadata (schema `gpuwm-obs.goes-cwp.v2`) carries: satellite, sector,
scan start/end, the three source granule keys + sha256s, the
geostationary projection parameters and fixed-grid axes (the navigation
of record), the DQF rule actually applied per product (rule name AND
condemn mask — the pack must say how it was gated), the full
`DqfReport` and `CwpCounts` integers, and the CWP coefficient table
with the PROVISIONAL flags carried verbatim so no consumer can mistake
the ice branch for settled physics.

Payload planes, in declared order: `cwp` (g/m^2, NaN = no observation,
0.0 = clear-sky zero), `phase` (decoded, NaN = gated), `cod`, `cps`
(the gated inputs — consumers get to re-derive and re-gate), `lat`/`lon`
meshes from the rw-sat navigation, and then one `<product>_dqf` plane
per source (`cod_dqf`, `cps_dqf`, `actp_dqf`).

## Settled: the per-pixel DQF plane, and the v2 bump (2026-08-06)

v1 carried the DQF *counts* and the condemn mask and threw the
per-pixel plane away. Those counts are a summary: they say how many
pixels each cause condemned, not which. The DCOMP thin (256) and thick
(512) bits are deliberately **not** in the condemn mask — the operator
spec inflates observation error on them rather than gating them — so
from a v1 pack it was impossible to tell which pixels carried them, and
the assimilation lane's CWP observation operator recorded
`error_model.thin_thick_inflation.applied = false` for exactly that
reason. That is a hole this format punched, and v2 closes it.

Every source granule now contributes a `<product>_dqf` plane holding
its DQF **as published and ungated** — the flags are the evidence, so
the rule that reads them never masks them. `<f4` like every other
plane, which is lossless for a u16 flag word (every integer below 2^24
is exact in f32, verified over all 65,536 values). NaN means that DQF
pixel was itself fill or out of range — what `DqfReport::dqf_missing`
counts — and is never a stand-in for the real, meaningful value 0. Each
`sources` row names its plane in `dqf_plane`, so a consumer looks it up
instead of reconstructing the name, and `verify` refuses a pack whose
source row names a plane the pack does not hold.

The change is otherwise purely additive: the new planes append after
every v1 plane, so no existing plane is renamed, moved, retyped or
removed and no plane index shifts; the counts and the condemn mask are
unchanged and still there. The schema still bumps to `v2` — for both
families together, so "v2" means the same thing to a reader of either —
because "this pack has per-pixel DQF" is precisely what a consumer must
branch on. With the bump it can require v2 and fail closed; without it
the difference is silent and is discovered halfway through building an
error model.

Measured on the same scan (`s20262161801170`, full CONUS sector):
27,119 thin-flagged and 775,251 thick-flagged pixels of 3,750,000 —
all of it unrecoverable from a v1 pack. Read back through numpy, every
finite DQF pixel round-trips f32 -> u16 exactly, and the NaN count
matches the `dqf_missing` the metadata already claimed. The
assimilation lane reproduced all three counts independently from the
raw plane, and additionally checked `pixels_both = 0` before
multiplying factors together.

### Index the DQF plane over the whole pack, never over the CWP mask

The `_dqf` planes are full-grid planes and must be indexed over every
pixel of the pack. The obs-error inflation factor is needed per pixel
**before** superobbing decides which cells survive, so "pixels with a
DQF" and "pixels with a surviving CWP" are different populations and
are not interchangeable. An earlier version of this note said the NaN
DQF pixels were "gated out of `cwp` anyway"; that is true and
misleading, and following it would have read NaN as a bitfield for the
47,162 pixels whose flag was itself fill. `astype(np.uint16)` on a NaN
does not raise — it returns garbage quietly. Mask on `np.isfinite`
first, then read bits from the finite pixels; `dqf_missing` in the
metadata is the count of exactly those pixels, so it is what to check
the mask against.

The two DCOMP planes were bit-identical on the measured granule, but
that is an observation about one granule. The consuming operator takes
the union of the thin/thick sets across COD and CPS rather than relying
on the identity, which is the right call and the reason the format
keeps them as two planes.

### Pairing a windowed pack (nested campaigns): a worked example

This is a proven pattern, not a design intention — the assimilation
lane runs it in anger. The ordering matters and is the one thing worth
not rediscovering: **build the windowed `cwp` pack FIRST, then point
`cloud-top --pairs-with` at it.** The sibling block can only pin a pack
that already exists.

Step 1 — the 2 km pack, windowed to the nest:

```
rw_goes cwp --cod  OR_ABI-L2-CODC-M6_G19_s20262161801170_..._c20262161805324.nc \
            --cps  OR_ABI-L2-CPSC-M6_G19_s20262161801170_..._c20262161805325.nc \
            --actp OR_ABI-L2-ACTPC-M6_G19_s20262161801170_..._c20262161804390.nc \
            --window 900,512,600,384 \
            --out g19_c_20260804_1801_win.goespack
```

```
pack.bytes          = 7099461
pack.content_sha256 = 3fa3de3e1222292e70eb2357053076f200fcdd78ee62f9792c8864bd20b8c6e4
nx x ny             = 512 x 384          scan_start = 2026-08-04T18:01:17Z
```

Step 2 — the 10 km pack, pinned to the pack from step 1:

```
rw_goes cloud-top --acha OR_ABI-L2-ACHAC-M6_G19_s20262161801170_..._c20262161805241.nc \
                  --ctp  OR_ABI-L2-CTPC-M6_G19_s20262161801170_..._c20262161805240.nc \
                  --pairs-with g19_c_20260804_1801_win.goespack \
                  --out        g19_c_20260804_1801_win.cloudtop.goespack
```

```
pack.bytes          = 3618926
pack.content_sha256 = dfca3f38ce25a6551e447224f8da60a349b83e4dbd4a360697fa76e0b6891094
nx x ny             = 500 x 300
sibling = {"schema": "gpuwm-obs.goes-cwp.v2",
           "filename": "g19_c_20260804_1801_win.goespack",
           "content_sha256": "3fa3de3e1222292e70eb2357053076f200fcdd78ee62f9792c8864bd20b8c6e4",
           "nx": 512, "ny": 384, "window": [900, 512, 600, 384]}
```

The sibling block pins the *window's* digest, so the pairing is
provable rather than asserted: a consumer hashes the CWP pack it holds
and compares. A cloud-top built against the full-sector pack will
correctly fail to match a windowed one — different packs, different
digests — so always build the cloud-top against the window you actually
made. Scan identity (`satellite`, `sector`, `scan_start`) is checked
before the digest is recorded, and a mismatch on any of the three is
refused outright.

**The cloud-top stays full-sector (500x300) on purpose.** A 2 km window
has no exact 10 km counterpart in general: 512 two-km columns is 102.4
ten-km columns, so no integer box covers it exactly. Rounding one into
existence is precisely the silent resampling the two-pack split exists
to prevent, so this tool will not do it — the pack pairs a 512x384
window with the whole 500x300 grid and lets the consumer choose the
join. A caller who wants a smaller cloud-top passes `--window`
themselves with a box that covers the 2 km window; the pack records
whichever window it was given, on its own grid. Window indices are
per-pack and never comparable across the pair.

In use (assimilation lane, reported 2026-08-06): three windowed pairs
from live `noaa-goes19` scans at 2026-08-02 17:01 / 17:16 / 17:31Z,
window `1320,274,758,222` over the ktbw domain, each cloud-top paired
with its own window. Yield from one scan: 20,144 observations — 2,232
clear, 403 liquid, 17,509 ice — and zero phase-mixed cells at 1.5 km.
The full-sector cloud-top cost them nothing. The per-pixel DQF planes
turned 58,881 unattributed missing cloud tops into a stated reason,
which is the whole argument for shipping flags rather than summaries.

### Reading v1 after v2

`verify` reads every schema this tool has ever written, v1 included,
while `cwp` and `cloud-top` write v2 only. Receipts record pack
digests, and a digest nobody can re-verify is a dead receipt — dropping
a reader is a decision to invalidate history, and this format does not
make that decision as a side effect of adding a plane. The verify
receipt reports `per_pixel_dqf`, false for a v1 pack: a fact about the
version, not a fault in the pack. Re-proved live against the three v1
packs earlier receipts name (`584162902a2a…`, `bbd3a16f…`,
`e35a6d6c…`), all PASS.

## Settled: two packs per scan (coordinator, 2026-08-06)

Measured on real GOES-19 CONUS granules for scan `s20262161801170`,
during the `rw_goes` build: COD, CPS and ACTP are the **2 km** fixed
grid, 2500 x 1500. ACHA and CTP are the **10 km** fixed grid, 500 x
300. Same satellite, same sector, same scan start, same geostationary
projection — a different grid. The bit-identical-grid guard fired on
the first real trio+ACHA run, naming both shapes. That is the format
working, not a bug: the earlier "ACHA/CTP ride along as optional
layers" plan above was written before anyone had measured the two
grids, and it is not achievable without a resample.

**Ruling: SEPARATE PACK. The bridge never regrids to make one pack.**
A pack's whole promise is "every plane in here is bit-identically on
the grid this pack states". Silently upsampling cloud-top height to
2 km would break exactly the guarantee the format exists to make, and
would bury an interpolation choice inside an ingest tool where no
science reviewed it. The vertical-placement join happens at the
**consumer**, where the interpolation is explicit, chosen by the
science, and recorded in that stage's own receipt.

The second pack: schema `gpuwm-obs.goes-cloudtop.v2`, same `GPWMGOES`
container and header, its own grid metadata, projection and DQF rows.
Planes in declared order (plus `acha_dqf` / `ctp_dqf` at the end, per
the v2 section above): `cloud_top_height_m` (m), then
`cloud_top_pressure_hpa` (hPa), then `lat`/`lon` on **its** grid. Its
metadata carries `pairs_with_schema` and a `regrid` field that states
in words the resampling it did not do, so no consumer can read silence
as "the grids matched".

Pairing key: `(satellite, sector, scan_start)`, stated at the top level
of both packs. `rw_goes cloud-top --pairs-with <cwp pack>` additionally
checks that identity against the CWP pack and refuses on any
difference, then records the CWP pack's `content_sha256` and grid in a
`sibling` block — so a consumer can *prove* it paired the right two
files rather than trusting a file name. A `--window` index is per-pack
and is not comparable across the pair, because the grids differ.

`rw_goes verify` reads the family from the pack's own `schema`, never
from its file name, and emits `gpuwm-obs.goes-cwp-verify.v1` or
`gpuwm-obs.goes-cloudtop-verify.v1` accordingly.

Recorded by the coordinator on 2026-08-06 as packaging shape, not as a
registered scientific criterion; to be surfaced to Drew in the morning
summary.

## Python side

`gpuwm/obs/goes_cwp.py`: pack reader (header/schema check before any
payload byte — the `sweeps.py` discipline), then regrid to the target
model grid onto the `gpuwm-obs.radar-grid.v1` conventions:
`cwp_obs`/`cwp_mask`/`cwp_err` + `grid_identity_sha256`, masks int8
with no "maybe", counts for everything dropped. Superobbing is simple
areal averaging of CWP within a target cell with a minimum-fill
fraction; clear-sky zeros average with cloudy pixels only if the cell
is phase-uniform, otherwise the cell is masked and counted (a cell half
clear, half deep ice is not one observation). Per-type obs error
assigned here, per the operator spec.

## Settled: the DCOMP condemn-mask default (Drew, 2026-08-05)

The shipped default (condemn missing/fill DQF plus snow/sea-ice 8,
twilight 16, glint 64) keeps 1,361,938 of the 1,481,473 fixture-granule
retrievals, 91.93%. Shown that split, Drew ruled: "yeah if its 92% of
data truly then no point contam it with the 8% that might hurt it." The
default stands, and the condemned ~8% stays out on contamination-risk
grounds — those bits mark scenes that could degrade the analysis, so the
bridge does not admit them. The `cwp` subcommand therefore ships that
mask as its default and records rule name + condemn mask in every pack,
per the metadata contract above.

Settled means the default, not the science: which bits earn condemnation,
and whether some causes should inflate obs error rather than gate (the
operator spec already does that for the thin/thick bits, which are NOT in
the condemn mask), stay available as A/Bs for the obs-skill scoreboard.

## Open questions for Drew (carried from the upstream patch set)

1. Ice/mixed CWP coefficient — keep the PROVISIONAL 2/3·rho_i form, or
   adopt a published IWP–tau–r_e relation before the bridge builds?
2. Mesoscale cadence: COD/CTP have no meso sector, so meso-cadence CWP
   is impossible; CONUS 5-min is the ceiling. Acceptable for the WaH
   cycling window, or should the bridge also emit a phase+height-only
   meso pack at 1-min?
