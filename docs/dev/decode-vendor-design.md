# Decode-vendor design: the mapped engine moves to Rust

Status: skeleton landed on `lane/decode-vendor`; three port lanes branch
from it.  Ruling served: the Python boundary (Drew, 2026-08-16) — all
data-path processing is Drew's Rust or seeded from it; Python survives as
orchestration/CLI and CUDA driver code.

Donor: a private working checkout of `rusty-weather-consolidated` @
`integration/rw-consolidated-20260817` (`edf044c`), READ-ONLY.  Vendored
snapshot: `tools/rw_wps/` (see its `VENDOR.md` for the hash-verified file
list).  In-tree proof at vendor time: donor suite 65/65 green
(60 lib + 5 integration) under gpuwm's workspace manifest.

## 1. What exists today

### 1.1 gpuwm's Python engine (the behaviour of record)

`gpuwm/mapped_source.py` (5043 lines) consumes the sealed
`rw-wps.mapping.v1` contract and materializes canonical, hash-bound
source frames.  Its public entries — signatures FROZEN by this design:

| Entry | Returns |
|---|---|
| `decode_mapped_source(mapping_path, files, *, input_manifest, input_manifest_sha256, grib1_bridge, grib2_inventory, grib2_dump)` | `tuple[MappedSourceFrame, ...]` |
| `inspect_mapped_source(...)` (same shape) | `gpuwm-mapped-source-inspection-v1` dict |
| `mapped_frames_to_regular_snapshots(frames, *, soil_land_repair)` | regular-grid snapshots for the WRF-real ABI |
| `mapped_frame_receipt(frame)` | evidence dict |
| `gpuwm.mapped_composition.load_composition / decode_composed_source(...)` | `MappedSourceBundle` |
| `gpuwm.mapped_composition.mapped_composition_receipt(bundle)` | receipt dict |
| `gpuwm.mapped_direct.prepare_mapped_wrf(...)` and the `gpuwm-wrf-init` / `gpuwm prep` front door | prepared run |
| `gpuwm.member_prep.prepare_member(...)` / `gpuwm-member-prep` | member preparation |

Byte-processing already in Rust (exe bridges, kept): GRIB1 records via
`grib1_bridge`, GRIB2 inventory/dump via `grib2_inventory` /
`grib2_dump`, NetCDF via `rw_netcdf` (`gpuwm.netcdf_bridge`), the
acquisition codec (bzip2 twin staging) in `gpuwm.ingest.codec`.

Data-path steps still executed in Python/numpy — the port surface,
everything below moves behind the seam:

* GRIB2 selector→record matching on raw identifiers
  (`_grib2_wanted_indices`), selector-identity refusal, embedded
  valid-time/member extraction;
* declared-grid cross-check of every observed GRIB grid octet
  (`_validate_grid_declaration`, Lambert family, projected axis units
  `PROJECTED_AXIS_UNIT_M`);
* grid-relative wind rotation to the earth basis
  (`_rotate_grid_relative_winds`);
* record assembly into collections (`_assemble_grib`, `_decode_netcdf`
  above the bridge), transposition to target axes, declared level
  selection/ordering, layer-slice positioning;
* unit transforms and the closed derivation catalog
  (`_DERIVATION_ARGUMENTS`: rh→q, dewpoint forms, geopotential→height,
  layer-mass→volumetric soil moisture, soil surface node);
* canonical-frame invariants (`CanonicalField`, `MappedSourceFrame`
  post-init checks), missing-count accounting;
* soil column repair (`_nearest_soil_column_repair`);
* composition byte work: `_compose_terrain`, `_compose_bound_fields`,
  exact/binding subset index computation, cross-source partition decode;
* frame→regular-snapshot packing (`mapped_frames_to_regular_snapshots`).

Policy that STAYS Python (orchestration/receipts/refusal grammar):
authority snapshotting and hash-stability windows
(`_snapshot_authority` / `_require_authority_snapshot`), input-manifest
verification, decoder-path ladder resolution and ABI-marker handshakes
(`_build_grib2_tools`, `gpuwm.bridges`), receipts
(`mapped_frame_receipt`, `mapped_composition_receipt`, provenance
copies in `mapped_direct`), the member grammar and identity policy
(`member_grammar`, `member_prep` — verification/counting policy over
inventory evidence), CLI parsers, `prepare_mapped_wrf` orchestration,
and every user-facing refusal TEXT (the engine supplies class+message,
Python owns which exception type reaches the caller).

### 1.2 The donor engine (`tools/rw_wps/crates/rw-wps`)

* `mapping.rs` (2760 lines): the typed `rw-wps.mapping.v1` model —
  `NativeMapping`, selectors (GRIB1/GRIB2/NetCDF, scalar-or-list name
  specs), closed `Derivation` catalog, `TargetContract`,
  `validate_mapping`, and IN-PROCESS source inspection
  (`inventory_sources`, `inspect_source_fields`) reading real bytes
  through `grib_core::{Grib1File, Grib2File}` and netcrust — no
  subprocess tools.  WPS Vtables import only as non-executable row
  drafts.
* `lib.rs` (4561 lines): run configs for source kinds including
  `Mapped` (contract, format, mapping+composition paths, role-bound
  supplements/provenance, decoder config, member axis via
  `MemberCoordinate::{Dimension, EmbeddedMetadata}`), capability
  discovery, process orchestration with progress events, output
  receipts (`rw-wps.receipt.v1`), authoring argv builder.
* `main.rs`: CLI — `sources | namelist-support | template |
  mapping-template | mapping-validate | inventory | inspect |
  import-vtable | author-mapped | validate | plan | run`.
* Schema identity: already reconciled byte-for-byte to gpuwm's live
  authority files (donor constants name gpuwm's own schemas:
  `gpuwm-mapped-composition-v2`, `gpuwm-mapped-composition-inputs-v1`,
  `gpuwm-native-source-adapters-v1`); 65/65 green in-tree.

### 1.3 Where the two disagree in concept

1. **Direction of the seam.**  The donor's `run` treats gpuwm as "the
   native engine": Rust orchestrates, launches gpuwm, seals receipts
   over gpuwm's outputs.  gpuwm's ruling is the inverse: Python
   orchestrates, Rust decodes.  The vendor therefore keeps `mapping.rs`
   whole (it is the decode/validation engine we want), reuses `lib.rs`
   piecemeal (config/receipt types, progress-event grammar), and does
   NOT wire `run`'s engine-launching path into gpuwm — the mapped
   engine exe never invokes Python.
2. **Who reads bytes for inspection.**  Donor inspects in-process
   (grib-core as a library); gpuwm shells per-file to
   `grib2_inventory`/`grib2_dump` and re-parses TSV.  The donor shape
   wins behind the seam (one process, no TSV re-parse); the standalone
   TSV tools remain for audits and for the Python engine workaround.
3. **Decode-to-frames.**  The donor validates and inspects but does not
   materialize gpuwm's canonical frame set; frame materialization
   semantics (assembly, rotation, derivations, invariants, soil repair,
   composition) exist only in Python.  That port is new Rust code in
   the `mapped-engine` crate, seeded from `mapping.rs` types.
4. **Refusal surface.**  Donor: typed `RwWpsError` + validation
   `Diagnostic` lists.  gpuwm: Python exception classes with remedy
   text bound to install state.  Bridged by the refusal-class grammar
   (§3): the engine names class+message+remedy, Python maps class→
   exception type and re-words install remedies it alone can know.
5. **grib-core lineage.**  Two divergent descendants of one crate
   (§2).

## 2. grib-core convergence plan (lane 1)

Two copies exist:

* **gpuwm's** `tools/grib1_bridge/vendor/grib-core` — 167 unit tests.
  Unique hardening: fail-closed unknown/reserved
  `missing_value_management`, zero-bits-with-missing refusal,
  fail-closed unknown grid templates in the DECODE path, fail-closed
  unknown cited/local parameter tables, ECMWF table-128 resolution by
  cited version, NCEP-extension region hygiene + severe-weather rows,
  section-1 origin/production identity preservation, sign-magnitude
  fixed-surface scales (both surfaces), truncated-PDT refusals,
  bitmap raw-padding retention, spatial-differencing missing-cell
  exclusion + mode-zero semantics, zero-width-group missing markers,
  brightness-temperature units row, IEEE point-count agreement with
  section 5, reduced-grid-without-PL refusal, JPEG2000 (openjp2)
  feature wiring, row-window missing-cell NaN marking.
* **donor's** `tools/rw_wps/vendor/grib-core` — 155 unit tests.
  Unique hardening: complex/spatial packing correctness family
  (group-count vs section-5 points, primary/secondary missing skip the
  recurrence, primary-missing-is-not-a-difference, recurrence overflow
  is an error, width-zero groups classify both missing codes,
  zero-group constant row-window parity, row-window complex missing
  matches full decode), minute-resolution statistical intervals with
  exact durations, WMO code-table 4.4 fixed-duration units, wrapped
  global longitudes / equal-endpoint cyclic grids using the declared
  increment, `rotated_to_geographic` component handling, bounded
  second fixed surface capture, probability-threshold sign-magnitude.

True divergence after CRLF normalization: 9 files, 2664 changed lines
(`grib2/unpack.rs` 1460, `grib2/parser.rs` 592, `grib1/tables.rs` 249,
`grib2/tables.rs` 135, `grib2/grid.rs` 128, `grib1/parser.rs` 85,
mod files 14, `grib2/search.rs` 1).  `lib.rs`, `grib1/grid.rs`,
`grib1/unpack.rs` are byte-identical modulo line endings.

Procedure:

1. Base = gpuwm's copy (the stricter descendant; its consumers are the
   shipped bridges).  Graft donor-unique behaviour file by file in the
   order unpack → parser → tables → grid, porting the donor-unique
   tests with each graft (14 named donor-only tests + the donor
   variants of shared families).
2. Named semantic conflicts, resolved fail-closed per the refusal law,
   each with a written note in the crate:
   * unknown grid template: decode path REFUSES (gpuwm semantics);
     the donor's `grid_latlon` returns-empty behaviour survives only
     as the non-decode probe API if a donor caller needs it, never as
     a silent decode fallback.
   * IEEE (DRT 5.4) length: section-5 point count is the authority
     (gpuwm semantics); trailing section-7 padding beyond
     `count*width` is tolerated and recorded (donor semantics),
     short data refuses.  Donor's trim test is re-expressed against
     the superset struct (which carries `section5_num_data_points`).
   * table lookups: unknown cited/local tables refuse (gpuwm); donor
     rows that resolve through WMO 4.2/4.5 + NCEP extensions merge
     additively.
3. Gate (names the breakage: a superset that quietly dropped one
   side's hardening would decode HRRR/RAP/GDPS complex-spatial or
   ensemble octets differently between the two engines and the parity
   battery would chase phantom diffs): `cargo test` green on the union
   suite — every gpuwm test + every ported donor test — in
   `tools/grib1_bridge/vendor/grib-core`, then `cargo build --locked
   --offline --release` of all five grib1_bridge bins, then
   `cargo test -p rw-wps` green with the path dep flipped (§5).
4. End state: ONE crate at `tools/grib1_bridge/vendor/grib-core`;
   `tools/rw_wps/vendor/grib-core` deleted in the same integration
   commit that flips `crates/rw-wps/Cargo.toml`'s path dep.  Feature
   posture: `jpeg2000` stays on (RAP/NAM/GDPS), `png_codec` off, per
   the existing grib1_bridge reasoning.

## 3. The seam contract (normative)

Shape: an EXECUTABLE, `gpuwm_mapped_engine`, following the repo's
dominant bridge pattern (raw little-endian array stream + JSON
metadata, ladder resolution, static ABI marker) — not a dll: the
decode is a batch job over many files, the exe isolates a decode crash
from the driver process, and the artifact-verification law wants a
binary a user can run by hand.  Skeleton lives at
`tools/rw_wps/crates/mapped-engine`; Python constants at
`gpuwm/mapped_engine_bridge.py`.  Both encode THIS section; the parity
battery holds them to it.

### 3.1 Invocation

```
gpuwm_mapped_engine {decode|compose|inspect}
    --mapping MAPPING.json --input-list FILES.txt --output DIR
    [--composition COMPOSITION.json]
    [--supplement ROLE=PATH]... [--provenance ROLE=PATH]...
    [--input-manifest MANIFEST.json --input-manifest-sha256 HEX]
```

* `decode` mirrors `decode_mapped_source`; `compose` mirrors
  `decode_composed_source` (terrain composition, bound fields,
  contributing sources); `inspect` prints
  `gpuwm-mapped-source-inspection-v1` on stdout.
* `--input-list` is a UTF-8 file of one path per line
  (`read_input_list` grammar) — the argv-limit lesson from the 251-file
  icon-eu prep is baked in from day one.
* The engine decodes GRIB1/GRIB2/NetCDF in-process (superset grib-core
  + netcrust).  It never shells out and never invokes Python.

### 3.2 Outputs

* `DIR/frames.json`, schema `gpuwm-mapped-frameset-v1`: every scalar a
  `MappedSourceFrame` carries — valid_time, member, source_cycle,
  vertical kind/units/values, latitude/longitude axes,
  grid_fingerprint, mapping_sha256, input_sha256 map, full
  `SourceFrameHeader` including initialization_policies — plus per
  field: name, units, axes, location, staggering, shape, dtype `<f8`,
  byte offset+length into the stream, sha256 of the array bytes,
  missing_count, source_references.
* `DIR/frames.f64`: one little-endian float64 stream, fields packed
  row-major in `frames.json` order (the `grib1_bridge` idiom).
* stdout: one JSON object per line, schema
  `gpuwm-mapped-engine-progress-v1`; the last line is the run receipt.
* Refusal: nonzero exit; the LAST stderr line is one JSON object,
  schema `gpuwm-mapped-refusal-v1`:
  `{"schema", "class", "message", "remedy"}`.

What the READER verifies, and how many times it reads the bytes.  The
document carries two digests over the same stream: one per field, and
one over the whole file.  Because the fields are packed contiguously in
document order, the per-field digests already cover every byte — and
they bind dtype and shape besides — so `read_frameset` checks that the
field extents tile the stream exactly and then verifies the per-field
digests alone, spread over a small thread pool.  The whole-stream
digest is computed only for a frameset whose fields do NOT tile it,
where it is the one check that would see padding, a gap or a truncated
last field.  This is a reader-side strategy, not a contract change:
both writers still emit both digests, and a hand-run `frames.json` is
still verifiable either way.

### 3.3 Refusal classes

Enumerated in `gpuwm.mapped_engine_bridge.REFUSAL_CLASSES` (usage,
not_implemented [skeleton only], missing_input, mapping_invalid,
manifest_mismatch, selector_unmatched, grid_mismatch, decode_failed,
frame_invalid, forcing_series, authority_moved), each mapped to the
exception type the Python engine raises for the same condition today.
An engine refusal with an unlisted class is itself a defect and
re-raises as `RuntimeError` naming the unknown class.  Growing the
list is a contract change (marker review).  `forcing_series` grew the
list at the 2.5.0 release-candidate wave: the preparation front door
promoted the too-short-forcing-series `ValueError` to its own
`ForcingSeriesRefusal` (a `ValueError` subclass), and the 1:1 class
mapping had to follow -- the frameset schema is unchanged, so the ABI
marker does not move.

### 3.4 ABI marker and resolution

Marker literal `gpuwm-mapped-frameset-v1` (the output schema name — it
changes exactly when the frameset contract changes; a stale staged
binary must fail the static handshake, not write a shape the Python
side no longer reads: the 1.1.0 GFS series-file failure class).
Lane 3 registers `"gpuwm_mapped_engine"` in
`gpuwm.bridges.BRIDGE_ABI_MARKERS` and in the doctor estate.
Resolution ladder: `GPUWM_MAPPED_ENGINE_BIN` override (hard error if
missing), checkout `tools/rw_wps/target/{release,debug}`,
`libexec/bridges`, packaged dir, staged `~/.gpuwm/bridges` — the
`rw_netcdf` shape.

### 3.5 Python routing (signatures frozen)

* `decode_mapped_source` / `decode_composed_source` /
  `inspect_mapped_source` / `prepare_mapped_wrf` keep their exact
  signatures.  Default route: resolve engine, run it, `np.fromfile`
  the stream, rebuild `MappedSourceFrame` objects; the dataclass
  validators re-run in Python as a cross-check (cheap next to decode).
* Explicit `grib1_bridge` / `grib2_inventory` / `grib2_dump` arguments
  route that call through the Python engine: explicit tool
  configuration names the subprocess-tool route, and the existing
  pinned-tool test fixtures stay meaningful.
* Workaround: `GPUWM_MAPPED_ENGINE=python` or the front-door
  `--mapped-engine python` — documented AS A WORKAROUND (fixed means
  default: the bare default run is the Rust engine once lane 3's
  parity battery is green; shipping opt-in would be reported as a
  workaround, not a fix).
* Python re-verifies mapping/manifest authority hashes around the
  engine run (`authority_moved` window stays Python-owned).

## 4. Parity contract (lane 3's gate)

For every source in the model-gauntlet staging tree
(`$GPUWM_MODEL_GAUNTLET_STAGING`, `~/gpuwm-model-gauntlet-staging` by default)
(aifs, aigefs, aigfs, aigfs-hybrid, crosssource, gdas, gefs, gem-gdps,
icon, ifs, rap, rrfs — real staged agency bytes with facts.md) and the
repo-fixture-referenced earlier sources (era5 grib1+netcdf, gfs, hrrr,
20crv3 member), Python-engine prep vs Rust-engine prep through the SAME
frozen Python entries must produce:

1. **Field arrays: byte-identical** — per frame, per field, sha256 of
   the `<f8` array bytes equal, axes/shape/missing_count equal.  Both
   paths share grib-core lineage and must keep f64 arithmetic order in
   the ported derivations/rotations; where WRF/Python is proven wrong,
   never bit-exact to a bug: fix, document the divergence, ship the
   perturbed control beside the fixture.
2. **Manifests/receipts: canonically equal** — frame evidence,
   composition receipt, input manifests, inspection documents compared
   as canonical JSON after masking ONLY the enumerated engine-identity
   fields (decoder paths/hashes, engine name/version, timings).  The
   mask list is part of the battery, not ad hoc.
3. **Refusals: same class, same remedy** — the refusal battery drives
   every named refusal in both engines (bad mapping grammar, manifest
   drift, selector identity miss on the two-products-one-filename
   case, grid-declaration contradiction, truncated/corrupt GRIB,
   missing soil ladder, member identity violations) and asserts the
   Python exception TYPE and the remedy text match the Python engine's.
4. **Verdict default: an unexplained diff is FAIL.**  A diff is
   explained only by a committed note naming the defective side and
   the evidence; "close enough" does not exist here.
5. Member-axis coverage: at least one ensemble source prepped through
   `gpuwm-member-prep` per engine with identical member evidence rows.

The battery is a pytest module plus a driver script; it runs the real
exe (verify against the artifact) and is the release gate for flipping
the default.

### 4.1 Platform scope: a committed golden names the box that measured it

Every file under `tests/data/mapped_engine_goldens/` and
`tests/data/mapped_engine_compose_goldens/` carries a top-level
`measured_on` member (`{"system", "machine"}`), and the extractor stamps
it into anything it writes.

The reason is measured, not precautionary.  On node-1 (Linux) at
`7f0db8ab2` the battery was 26 failed / 65 passed / 46 skipped against
these Windows-measured files, and the split of the 26 is what settles
it: 4 python decode-golden, 4 rust decode-golden, 3 rust
inspection-golden, 7 python compose-golden, 7 rust compose-golden.  The
PYTHON reference engine misses exactly the goldens the Rust engine
misses.  Field by field on `rap-awip32` — 22 leaf diffs — every
DIRECTLY DECODED field hashes identical and only the DERIVED ones move:
`eastward_wind`, `northward_wind`, `eastward_wind_10m`,
`northward_wind_10m` (the grid-relative rotation) and
`specific_humidity` (the relative-humidity derivation), plus the frame
header hashes that carry them.  Two engines are not wrong in the same
direction on the same five fields; the platform's libm produced them.
Four of the rows fail identically at `41aec1590`, so it is not a
regression either.

So the rule the battery applies, per row:

* **on the platform named in the stamp** — compare against the
  committed golden byte for byte, exactly as before.  Nothing is
  masked, no tolerance is introduced, and a corrupted golden still
  takes the row red (proven with a two-way control: perturb one
  committed refusal message and the row on the measuring platform goes
  RED).
* **on any other platform** — run the LIVE dual-engine comparison
  instead, at the same strictness: the same source, the same digest
  function, the same fields, the Python engine against the Rust engine
  ON THAT BOX, byte-identical or FAIL.  That is the parity claim the
  battery exists to make, and it is the half that is portable.  The
  same control's other arm proves the fallback really is live: with the
  golden corrupted, the foreign-platform row stays GREEN because it
  never reads it.

What it never becomes is a skip.  Every row runs both engines and
clears a bar either way, and the row prints which reference it used and
why — including that cross-platform portability of the committed
numbers is open item GOLD-PORTABLE (task #157).  A golden with no stamp
is REFUSED by name rather than defaulted either way: defaulting to
"this box measured it" compares one platform's numbers against
another's and blames the engine, and defaulting to "some other box did"
silently stops comparing against the committed golden on the box that
measured it, which is the gate going quiet.

The stamp is deliberately coarse — operating system and machine
architecture, nothing finer.  It names the C runtime family whose libm
produces the derived fields; a libc version would make a golden stop
matching its own box after a system update, and a hostname would make
every box foreign.  Two boxes of one family with genuinely different
libm therefore compare against each other's numbers, which fails
loudly with a real numeric difference rather than passing quietly.

Re-measuring a golden on a different platform is a deliberate act:
`tools/extract_mapped_engine_goldens.py` names the platform change in
its refusal and `--force` is what moves a golden's home box.

## 5. Lanes (disjoint; branch from `lane/decode-vendor`)

| Lane | Branch | Files (exclusive) | Done means |
|---|---|---|---|
| 1. grib-core convergence | `lane/decode-vendor-gribcore` | `tools/grib1_bridge/vendor/grib-core/**` (superset), donor-test ports, `tools/grib1_bridge/vendor/VENDOR.md` | union suite green; five bridge bins rebuild `--locked --offline`; handoff note listing donor tests re-expressed |
| 2. rw-wps vendor + engine | `lane/decode-vendor-engine` | `tools/rw_wps/**` EXCEPT `vendor/grib-core` (crates/, workspace manifest, crates-io vendoring + `.cargo/config.toml`, `Cargo.lock`) | `decode/compose/inspect` implemented; crate-level goldens against staged real bytes; `cargo test` green offline; exe verified by hand on at least icon + one Lambert source + one NetCDF source |
| 3. Python rewiring | `lane/decode-vendor-python` | `gpuwm/mapped_engine_bridge.py`, `gpuwm/mapped_source.py`, `gpuwm/mapped_composition.py`, `gpuwm/mapped_direct.py`, `gpuwm/bridges.py` (marker+registration), `gpuwm/doctor.py`, front-door flag plumbing, `tests/**` parity battery, user docs, CHANGELOG | signatures unchanged; Rust default-on, Python engine behind the documented workaround; parity battery green on the full staging set |

Cross-lane authority: THIS document.  The two stub files already
encode the marker/schemas/refusal classes, so lanes 2 and 3 never edit
each other's files to stay in sync.  Integration steps (coordinator,
after lanes merge): flip `crates/rw-wps/Cargo.toml` grib-core path to
`../../../grib1_bridge/vendor/grib-core`, delete
`tools/rw_wps/vendor/grib-core`, rerun the union grib-core suite, the
rw-wps suite, the engine goldens, and the parity battery; then flip
the default engine and update the doctor estate + generated CLI
reference.

## 6. Known gaps carried forward

* Test counts in §2 are static `#[test]` counts.  Running gpuwm's
  grib-core suite standalone currently fails resolution offline: its
  manifest declares optional `libaec-sys`/`png`, absent from
  `tools/grib1_bridge/vendor/crates-io`, and cargo resolves optional
  deps even when the features are off.  Lane 1 makes the union suite
  runnable offline (vendor the two crates or drop the dead optional
  deps from the superset manifest) — the gate in §2 needs an
  executable suite, not a grep.

* The donor lib test `real_netcdf4_fixture_is_inventoried_through_
  hdf5_fallback` reaches `../rw-glm/tests/fixtures/OR_GLM-L2-LCFA_*.nc`;
  the 240 KB real fixture is vendored at the same relative spot so the
  donor test stays byte-identical (65/65 in-tree, verified).
* `tools/rw_wps` builds ONLINE only until lane 2 lands the crates-io
  vendor mirror; the shipped posture is `--locked --offline` like
  `tools/grib1_bridge`.
* The donor `rw-wps` CLI (`sources/plan/run`) is vendored but not a
  gpuwm front door; whether any of it ships is a separate ruling —
  nothing in this port depends on it.
* `mapped_frames_to_regular_snapshots` moves in lane 3 only if the
  parity battery shows the packing is a bottleneck; the ruling
  requires field BUILDING in Rust, and the frameset is the built
  product — snapshot packing is ABI adaptation and may stay Python in
  v1 (recorded here so nobody mistakes it for an oversight).

## 7. Contract addenda from lane 3 (the engine lane must implement)

Four things the §3 contract did not spell out, each found by wiring the
Python half against it.  All are in `gpuwm/mapped_engine_bridge.py`
already; the engine has to match.

1. **`--contributing-mapping ROLE=PATH`**, same ROLE=PATH grammar as
   `--supplement`/`--provenance`.  §3.1 enumerated supplements and
   provenance but not the third role-bound binding a cross-source
   composition carries: each contributing source's own sealed mapping,
   pinned by the composition's SHA-256.  Without it every
   `field_sources` binding resolves to no mapping and `compose`
   refuses.
2. **`DIR/composition.json`, schema
   `gpuwm-mapped-composition-evidence-v1`**, written by `compose`
   beside the frameset, carrying `alignment_receipt` and
   `contributing_sources`.  These are products of the byte work alone;
   `MappedSourceBundle` requires both, so a `compose` that wrote only a
   frameset could not produce a bundle and the composition receipt
   would have to be fabricated from values nobody measured.
3. **Axis round-trip hashes.**  `frames.json` carries each 1-D axis as
   JSON numbers PLUS the sha256 of its `<f8` bytes, and the reader
   refuses on mismatch.  Shortest-round-trip float printing makes the
   values exact on both sides; the hash is there so that if it ever is
   not, the result is a refusal rather than a quietly perturbed grid.
4. **The decoder a manifest seals.**  On the Rust route the decoder
   inventory is ONE role, `gpuwm_mapped_engine`, not the subprocess
   pair — the manifest binds the binary that actually read the bytes.
   `_decoder_inventory(..., engine=...)`, `author_input_manifest` and
   `_verify_manifest` agree on this, so a manifest sealed against the
   subprocess tools refuses on the Rust route instead of being replayed
   under a decoder that never ran.  `bridge_identity` grew a usage
   marker row for the engine so such a manifest can be authored at all.

Shared surfaces lane 3 touched outside its own file list, for the
assembler: `gpuwm/native_wrf_distribution.py` (one
`_BRIDGE_USAGE_MARKERS` row), `gpuwm/mapped_authoring.py` (engine-aware
decoder inventory), `gpuwm/bridges.py` (one `BRIDGE_ABI_MARKERS` row and
a note on why the engine is deliberately NOT in `BRIDGE_ENV` — that
map's consumers resolve through the grib1_bridge crate dir, and an entry
would answer from a staged copy while a fresh checkout build sat unused).

MEASURED seam cost, so the exe-versus-dll choice stays honest: a
two-frame 0.25-degree GDAS frameset is 3.55 GB; writing it takes 4.6 s
and reading it back 6.0 s on Drew's box, on top of a 26 s decode.  The
reader streams its hash and memory-maps the arrays rather than reading
the stream whole, so peak footprint does not double.

## 8. Integration: what the assembly measured and changed

The three lanes merged clean.  What follows is what only running them
TOGETHER, against real staged bytes, could show.

### 8.1 The seam's two halves did not agree

The frameset the engine wrote and the frameset gpuwm reads were written
by different lanes and had never met.  Three disagreements, all found by
the parity battery on the first joint run and all fixed on the ENGINE
side, because the reader's shape is the one this document specifies:

- `stream` was a bare file name beside a loose byte count; the reader
  wants one object carrying path, dtype, byte count and the whole-stream
  SHA-256, and it verifies that digest before it maps a byte.
- `grid_fingerprint`, `mapping_sha256` and `input_sha256` sat on the
  document; they belong on each frame, because the reader rebuilds one
  dataclass per frame and a frame that borrowed its provenance from an
  enclosing object could be replayed under another decode's.
- axes went out as bare arrays; they carry `{values, count, sha256}` so
  a JSON round trip that moved one bit refuses instead of quietly
  shifting a grid.

### 8.2 Three real defects the joint run exposed

1. **The longitude wrap.**  gpuwm has TWO `_wrap180` functions:
   `gpuwm.static.projection._wrap180` (a cut-zone comparison, what
   `ijll_lc` uses) and `gpuwm.mapped_source._wrap180`
   (`((v + 180) % 360) - 180`, what the mapped path applies to a
   DECLARED grid's corner).  They differ in the last two bits.  The port
   took the projection's, which moved `polei` by eight ULPs and every
   grid-relative wind component with it -- the arrays hashed differently
   while every printed decimal looked the same.  The crate's rotation
   golden had been compared within `1e-15` and passed throughout; it is
   compared exactly now, and `wrap180_declared` is pinned against both
   spellings.  The modulo form is the less accurate of the two and is
   REPRODUCED, not corrected: changing it would move every mapped
   source's winds and grid placement at once, which is a ruling to ask
   for, not a change to slip into a port.
2. **A missing refusal.**  The engine built frame headers without
   `gpuwm.source_frame`'s WRF-initial-state check, so on a source whose
   f00 carries the 3-D state and whose f06 carries only surface records
   it ran past the frame Python refuses and refused a later one for a
   different reason.  Both refuse, so nothing wrong was produced; they
   sent a reader to opposite ends of the source.  Ported.
3. **A numpy repr in a user-facing message.**  The vertical-coverage
   refusal interpolated numpy scalars, so the same source refused with
   `np.float64(100.0)` under numpy 2 and `100.0` under numpy 1.  Fixed
   on the PYTHON side (the message is about a level ladder, not a dtype)
   and the golden re-measured; the engine renders Python's `repr` for
   lists and floats, pinned by test.

### 8.3 The default, and how the last path joined it

`DEFAULT_ENGINE` is `rust`, and the engine declares GRIB1, GRIB2 and
NetCDF for all three subcommands -- every source FORMAT gpuwm reads,
through every verb it reads them with.  The capability TABLE has no
holes left in it.

`ENGINE_GAPS` is empty, and the difference between the two sentences
above is still the point of this section: the table says which formats
the engine can read; the gaps tuple says which FRONT DOORS reach it,
and every registered door now does.  The last entry out was the
exact-member door, `gpuwm prep --source 20crv3`, and it left carrying
its two gates rather than dropping them:

- **the ensemble identity** rides the composition input manifest as an
  explicit atomic pair (`member`/`member_identity`, optional keys of
  `gpuwm-mapped-composition-inputs-v1`): the route verifies its own
  member manifest, seals the filename member into the bridged generic
  manifest, and `compose` -- both engines -- stamps it onto every
  canonical frame and into the alignment receipt, refusing bytes whose
  product octets already encode a member;
- **the product identity** gate keeps its exact per-file contract and
  its own refusal wording in the route; its instrument on the bare
  default is the engine's fourth subcommand, `inventory` -- the raw
  per-record GRIB2 identity octets (`gpuwm-mapped-record-inventory-v1`),
  rendered in the same string spellings as the subprocess
  `grib2_inventory` TSV so one gate reads one spelling whichever
  instrument measured it.  `inventory` takes `--input-list` alone: it
  resolves no selectors and writes no frameset, so it takes no mapping
  and no output directory.

The route's own sealed member manifest stays the user-facing authority:
the prepared tree's evidence copy, identity chain and forecast leg bind
it, while the composition receipt seals the bridged twin (published
beside it as `composition-inputs.json` so the receipt's record can be
re-hashed from bytes).  `_predecoded_bundle` -- the single branch that
skipped `decode_composed_source` -- is gone with the entry.

`compose` was the last entry, and it closed the way the other two did:
a change to the engine plus one line of the declaration, on real-bytes
evidence.  What the engine gained is §2.1's composition byte work --
the `gpuwm-mapped-composition-v2` grammar, the
`gpuwm-mapped-composition-inputs-v1` manifest shape, partition decode,
the exact coordinate-subset solve, terrain composition and its receipt,
the cross-source bound-field borrow and its receipt, the union mapping,
and `composition.json`.  What it did NOT gain is the policy above the
seam: manifest verification, the authority hash window, the role
inventory, the contributing-mapping hash pins, the primary-versus-donor
field agreement, the soil-layer contract against the donor table, and
the two post-return checks all still run in
`gpuwm.mapped_composition`, around the call.

The evidence is `tests/test_mapped_engine_parity.py`'s
`test_the_rust_engine_reproduces_the_compose_golden`: all eleven
registered `mapped_composition_v1` sources with staged bytes, measured
through `decode_composed_source` (the real front door, not a hand-driven
exe), reproduce the Python engine's composed answer byte for byte --
frames, alignment receipt and contributing-source records, across all
three terrain clock rules, both cross-source shapes, and the one source
whose staged pressure ladder makes it refuse, which refuses with the
same sentence.

Two properties keep the declaration honest rather than a silent
fallback, and both predate this port:

- the engine declares what it implements (`gpuwm_mapped_engine
  capabilities`, schema `gpuwm-mapped-engine-capabilities-v1`), and
  `ENGINE_CAPABILITIES` mirrors that declaration with a test comparing
  the two against the BUILT binary, so the table cannot drift from the
  artifact;
- `compose` may be declared for a format only when `decode` already is
  (`test_a_declared_compose_format_is_a_declared_decode_format`).  The
  breakage that gate prevents is specific: `mapped_authoring` seals a
  preparation's decoder rows by asking the table about `decode`, while
  `decode_composed_source` resolves the decoders it will actually run by
  asking about `compose`.  A format declared for one and not the other
  makes those answers name different decoder role sets -- one in-process
  engine row versus the subprocess pair -- so the manifest would seal one
  binary and the composition verify against another, and
  `_verify_manifest` would refuse a preparation that is in fact correct.
- the OTHER direction of that rule is not symmetric, and
  `test_the_two_questions_name_one_decoder_inventory` measures it per
  format by driving both production sites.  A GRIB format survives a
  declared `decode` beside an undeclared `compose`, because the front
  door forwards its subprocess tools and an explicit tool pins the
  Python engine at both sites.  NetCDF has no subprocess tool to
  forward, so nothing brings the two answers into step: the manifest
  seals the in-process engine row and the composition verifies against
  an empty inventory.  That is why `compose` must be declared for NetCDF
  exactly while `decode` is, and it is the reason NetCDF is declared
  here without a compose golden -- no composed NetCDF source has staged
  bytes (`20crv3-cf` is the registry-coverage test's one written-down
  exemption).  GRIB1 has no packaged composed profile at all and is
  declared for the same shape of reason.  Every measured row is GRIB2,
  which is every packaged composed profile the registry carries.

GRIB1 was an entry here until `mapped-engine/src/grib1.rs` landed.  It
was held back because nothing but the out-of-process `grib1_bridge`
could read edition-1 records.  The port builds on
`grib_core::grib1::Grib1File` -- already vendored, already under the
union suite -- and feeds the SHARED `assemble_grib`, so there is no
second assembly path.  The evidence is the `era5-1974-grib1` battery
row: forty-two array sha256 identities across three frames and fourteen
fields, the grid fingerprint, the per-array axes/shape/missing/min/max
and source references, and the materialization refusal, all equal to
the golden extracted from the Python engine.  GRIB1 `decode` now spawns
no subprocess.

That was a statement about `decode`, and the mapped preparation route
asks for `compose`.  While `compose` was declared for no format,
`prepare_mapped_wrf` composed in Python on every call and read
edition-1 records through `grib1_bridge`, so a bare GRIB1 prep still
resolved that subprocess tool.  The compose declaration is what ended
that demand: a bare GRIB1 prep now spawns nothing.
`tests/test_mapped_direct.py` pins both directions so the claim here
cannot drift from the table.

NetCDF was another entry here until the corpus fix.  It had been held
back on real evidence -- the engine read one hand-made file and misread
gpuwm's own corpus -- and the two causes turned out to be one packaging
defect and one missing port, neither of them in the decode itself:

- `tools/rw_wps/Cargo.toml` was missing the `[patch.crates-io]`
  `hdf5-reader` redirect that `tools/rustwx/Cargo.toml` carries, so
  `netcrust`'s `netcdf-reader 0.3` dependency resolved the STOCK
  crates.io reader while the hardened vendored copy sat unused.  Every
  `netCDF4.Dataset(path, "w")` file refused with an HDF5 checksum
  mismatch.  `tests/test_hdf5_reader_convergence.py` now holds the
  redirect and the one-package lock.
- NetCDF-4 coordinate variables are HDF5 *dimension scales*, which
  netcrust's `variables()` does not report, so an ERA5 file offered
  every field and none of its axes.  `rw_netcdf`'s recovery -- public
  netcrust API only -- is grafted into `mapped-engine/src/ncdf.rs`.

The declaration rests on the whole Python NetCDF test set green under
`GPUWM_MAPPED_ENGINE=rust`, not on the single crate golden, which was
green throughout and therefore proved nothing about the corpus.  The
remaining post-cut item is to upstream the dimension-scale recovery into
`netcrust` itself and re-prove `rustwx`, so `rw_netcdf` and the mapped
engine stop carrying the same graft twice.

### 8.4 A packaging trap, twice

`.gitignore`'s `tools/rw_wps/vendor/**/target/` matched
`vendor/crates-io/cc-1.4.3/src/target/` -- four real crate source files
kept out of the mirror, so `cargo build --locked --offline` failed in a
fresh checkout on a missing checksum while every lane's own worktree
built fine.  The rule is anchored to the vendored crates that can be
built on their own.  Same shape as the `*.nc` rule that swallowed the
GLM fixture: a broad ignore pattern crossing into vendored source,
invisible until someone checks out clean.

### 8.5 The crate golden that was not reproducible

`mapped-engine`'s NetCDF golden pinned a raw `frame_header_sha256`.  The
canonical frame header quotes each field's source references and those
are ABSOLUTE paths, so that digest is an identity of the machine that
measured it.  The two GRIB cases hide it (their inputs sit at a fixed
path outside the repository); the NetCDF sample lives beside the golden
and therefore moves with the checkout, and the digest differed between
two worktrees of the SAME commit while every field array matched to the
byte.  Both sides then compared under the repository's one mask -- input
paths reduced to basenames, the rule
`tests/test_mapped_engine_parity.py` already used.

**The mask was half the story (2026-08-20).**  Carried to a second BOX
rather than a second worktree, the masked digest went red again: the
golden recorded on the Windows desktop failed on weather-node-1.  Two
findings, both measured on the same bytes:

* the netCDF case: 35 of 38 arrays bit-identical, and the three produced
  by the two `exp`-based humidity derivations apart by at most 3 ULP --
  4.1e-16 relative;
* the Lambert case: all four wind components differ, because the
  grid-relative wind rotation is `sin`/`cos`.

A transcendental's last bit is the box's libm.  Everything else in the
decode -- integer unpack plus IEEE add, multiply, divide -- is
bit-reproducible, which is why only these moved.

So the header digest is not the thing to make portable by masking harder.
Both engines now publish `frame_header_sha256_portable` beside the raw
digest under one declared, versioned rule
(`gpuwm-portable-frame-header-v1`, `gpuwm.source_frame.portable_frame_header`
and `mapped_engine::portable`): input paths reduced to basenames, and
each libm-dependent field's `data_reference` replaced by
`libm:<canonical_name>`.  The libm-dependent set is READ FROM THE
MAPPING -- fields with a declared `derivation`, plus the rotated wind
pairs when the declaration says the source publishes grid-relative
components -- so it is table work for a new model, not a code path.

What the portable digest still gates: grid, vertical coordinates, times,
policies, units, shapes, dtypes, the field roster by name, and the exact
array digest of every other field.  What it no longer gates is compared
by VALUE instead: the goldens record five statistics per libm-dependent
field (minimum, maximum, sum, sum of magnitudes, sum of squares) and
assert them within a declared 1e-12 relative tolerance -- four orders
above the measured cross-box spread, and far below any change of
formula, unit, level order or dependency.  The raw and masked digests
stay in each golden as DECLARED PER-BOX VALUES under `per_box`, with the
recording box named.

## 9. Concurrency inside the engine (normative)

A decode is spread over threads.  Nothing about that may reach an output
byte, so this section states the rule the crate obeys and the reason it
is a rule rather than a habit.

**THE RULE.**  Every parallel step in `mapped-engine` is an INDEXED
collect into pre-assigned slots, followed by a sequential drain in
document order.  Element *i* of the result comes from input *i* whatever
order the threads finished in, and the drain returns the FIRST error in
input order.  Determinism is therefore a property of the data structure,
not of the run: it survives a loaded box, a different worker count, and
a scheduler change in a future rayon.  `crates/mapped-engine/src/
threads.rs` holds the pool, the drain and two tests that pin the
property.

**WHAT IS SPREAD.**  The objects of an input list; the selected messages
of one object; the mapped fields of one assembly; the array digests of
one frame's fields, in all three places they are taken (the canonical
frame header, the frameset manifest, `inspect`'s per-field row); the
input file digests.

**WHAT IS NOT.**  The frameset WRITE.  Stream offsets are cumulative and
the whole-stream digest is order-dependent, so the bytes go out on one
thread; only the per-field hashing beside it moved.

**THE PROGRESS STREAM IS PART OF THE ANSWER.**  A caller parses it, so it
is reproduced line for line: an object that fails before it is
inventoried announces nothing (as the serial `?` did) and one that fails
after announces its inventory first.  That is why the per-file task
returns two shapes rather than one `Result`.

**ONE DECODER IS NOT RE-ENTRANT.**  `openjp2`'s HTJ2K path keeps state
in a process-global `static mut`, so GRIB2 Data Representation Template
5.40 messages take a mutex around `unpack_message`.  Two threads
decoding 5.40 codestreams at once could otherwise read each other's pass
state and land the corruption in float values that are finite, correctly
shaped and wrong -- the failure mode no downstream check catches.  Every
other template unpacks from `&self` into a fresh `Vec` and pays nothing.
The staged RAP case packs this way and is decoded at 1 and 32 workers to
the same bytes.

**THE WORKER COUNT** is the machine's parallelism capped at eight, and
the cap is a measured knee, not a guess -- the sweep is written into
`threads.rs` beside the constant.  `GPUWM_MAPPED_ENGINE_THREADS`
overrides it, which is what a caller running several engines at once
should use.

**THE GATE.**  A change here is proven by byte identity, not by a digest
comparison: `frames.json` diffs clean, `frames.f64` keeps its sha256 and
the progress stream is identical, on real staged bytes, AT MORE THAN ONE
WORKER COUNT.  A single-configuration test cannot see the defect this
gate exists to prevent.  `examples/decode_timing.rs` is the instrument
that says where the seconds went, before and after.
## 10. The prove-out: the full §4 sweep, measured (2026-08-17)

The battery grew from ten staged sources to SEVENTEEN, completing the
§4 list.  New rows and where their bytes come from:

| Row | Bytes | Answer |
|---|---|---|
| `gefs-ensemble-control` | staged control-member pgrb2a+pgrb2b pairs, two valid times | DECODED, 2 frames, 17 fields, embedded ensemble identity |
| `hrrr-prs` | two whole `wrfprs` CONUS files fetched from the AWS mirror into the staging tree (835 MB, SHA-256 manifest beside them) | DECODED, 2 frames, 19 fields -- the 1799x1059 Lambert grid under complex/spatial-differencing packing, the exact octet family §2 was hardened for |
| `twentycrv3-member-pl-sfc` | the 20CRv3 private member sample (`GPUWM_20CRV3_MEMBER_SAMPLE`) | DECODED, 2 frames, 18 fields |
| `aigfs-gdas-hybrid-pres` | the staged hybrid pressure product alone | REFUSED naming the surface state the file does not carry |
| `gfs-pressure-fixture` | the repository's own committed GFS fixture bytes -- the one row that runs on every checkout | REFUSED: the soil-ladder refusal (1 record vs 4 declared selectors) |
| `era5-1974-grib1` | the 1974 reference bundle (`GPUWM_TEST_WRF74_BUNDLE`) | REFUSED (terrain arrives by composition); inspection still hashes all 42 GRIB1 arrays |
| `era5-netcdf` | the CDO NetCDF oracle beside it | REFUSED (the file carries no surface geopotential, so `terrain_height` is unresolvable); compares REFUSAL parity only -- the Python engine refuses in selector resolution, before any array digest is recorded, so this row compares one sentence and zero field arrays.  The byte-level NetCDF evidence is the `netcdf-pressure-level` crate golden and the Python NetCDF suites under `GPUWM_MAPPED_ENGINE=rust`, not this row |

Both era5 rows were `ROUTED-PYTHON` skips when this section was first
written: the Rust-comparison tests did not silently skip, they asserted
the route and then skipped with the gap named.  As of the GRIB1 and
NetCDF ports there is nothing left for that guard to skip -- every row
in the battery compares the two engines for real, and `ROUTED-PYTHON`
appears on none of them.  The route assertion itself survives in
`test_a_bare_run_of_an_unported_format_routes_without_pretending`,
which now reads `gpuwm.mapped_source`'s format list against
`ENGINE_CAPABILITIES` and asserts BOTH directions, so it stays a live
gate with no unported format to point at: a declared format that a bare
run did not reach would fail it.

Everything else on the §4 checklist, measured on this box at this
commit:

* **Battery**: all four test families green over all seventeen rows
  (the two era5 rows route-pinned when this was written, genuinely
  compared since the GRIB1 and NetCDF ports); grib-core union suite 182
  green; rw-wps workspace 55+3+60+5 green, `--locked --offline`.
* **Member evidence** (§4.5): `gpuwm-member-prep` run twice on the
  staged GEFS control (pgrb2a+pgrb2b, f000+f003), once per engine.
  The receipts differ in ONE field, `prepared_at` -- a timestamp, on
  the enumerated mask -- and all four member evidence rows are
  byte-equal.  (Member verification reads identity octets through
  `grib2_inventory` on both routes by design, so this parity is
  structural; the dual run pins it against regression.)
* **Front door**: `gpuwm-mapped-inspect` run twice on the staged RAP
  pair -- `GPUWM_MAPPED_ENGINE=python` versus the bare default -- and
  the printed inspection documents are canonically equal after masking
  only `decoders`.
* **Sweep**: `tools/mapped_engine_parity_sweep.py` drives every row
  through the PUBLIC entries twice and compares frames, receipts and
  inspection documents, plus a driven refusal battery (grammar,
  missing input, corrupt GRIB, the NOMADS/S3 selector-identity twin).
  Measured result: 15 PASS + 2 ROUTED-PYTHON (the era5 rows, by
  declared gap), zero FAIL; all four refusal cases agree in class and
  sentence.  Full record in the DECODE-VENDOR report in Drew's
  Downloads and `sweep-record.json` beside the evidence charts.

Two refusal-parity defects the driven battery exposed, both fixed here:

1. **Undecodable bytes refused as different classes.**  The Python
   engine's subprocess wrapper raised `RuntimeError` for ANY nonzero
   decoder-tool exit, so corrupt GRIB bytes surfaced as a tool failure
   while the engine's `decode_failed` maps to `ValueError`.  Split on
   whether the tool produced a diagnostic: a diagnostic means the BYTES
   are the problem (`ValueError`, both routes agree); silence means the
   installation is (`RuntimeError`, kept).
2. **One broken mapping, two sentences.**  The Rust route handed
   grammar validation to the engine alone.  Both engine routes now run
   `load_mapping` -- the one Python validator -- before dispatch, so a
   grammar refusal is byte-identical whichever engine would have
   decoded, and the engine's validator remains behind it for hand-run
   invocations.

One EXPLAINED message divergence, by design: on corrupt bytes the
Python route quotes its subprocess tool (`GRIB2 inventory failed for
X: Error: "truncated GRIB2 envelope 87 at byte 2706453..."`) while the
engine speaks in-process (`truncated GRIB2 envelope 87 at byte 2706453
of X: ...`).  Same class, same grib-core diagnostic, different wrapper
-- the difference IS the masked decoder identity, and inventing a
subprocess sentence for a decode that never left the engine would lie
about which binary read the bytes (§7.4).  The sweep pins the shared
diagnostic instead.

### 10.1 The measured bench (this box, warm cache)

Decode through the public entry, Python engine vs Rust default:

| Case | Files | Python | Rust | Note |
|---|---|---|---|---|
| icon-eu (many small bz2 files) | 252 | 25.7 s | 19.6 s | Rust 1.31x faster; 0.102 -> 0.078 s/file.  Inspection 29.4 -> 21.7 s |
| rap-awip32 (small) | 2 | 4.7 s | 5.9 s | 58.4 Mcells: 0.080 -> 0.101 s/Mcell |
| hrrr-prs (largest frames) | 2 | 41.6 s | 70.0 s | 1002 Mcells: 0.042 -> 0.070 s/Mcell |
| gdas-pgrb2-0p25 | 2 | 20.2 s | 32.5 s | 444 Mcells |
| rrfs-prslev-2dfld | 4 | 50.0 s | 82.0 s | 1139 Mcells |
| refusing sources (7 GRIB2 rows) | -- | -- | -- | Rust equal or faster on every one |

The frame-heavy slowdown was the SEAM, not the decode: the exe wrote
the full `<f8` stream to disk, hashed every byte twice, and ran the
whole data path on one core (§7's measured 3.55 GB / 4.6 s write /
6.0 s read on a GDAS frameset, doubled again by hashing on both
sides).  The engine's in-process decode itself was already faster than
the subprocess pipeline -- the many-file case shows it the moment
per-file process launches stop dominating.  **This target is CLOSED**
by the parallel work of §9, which landed beside this sweep: the
redundant whole-stream hash is gone, per-field digests verify on a
pool, the five frameset copies are eliminated, and the data path runs
concurrently under §9's indexed-collect rule.  Measured on the same
box, warm, through the real engine: hrrr-prs 68.4 s to 18.6 s, gdas
28.2 s to 7.8 s -- the Rust default is now faster than the Python
engine on the frame-heavy rows this table recorded it losing, with
every output byte unchanged and both engines still reducing to the
same parity digest.  The default stayed `rust` throughout because
correctness parity was proven and the ruling is the Python boundary,
not a stopwatch.

## 11. The 20CRv3 member row, measured on the real private bytes (2026-08-18)

> **Provenance.** This section was measured BEFORE the member-door port
> landed (section 8's `ENGINE_GAPS`-is-empty world).  The measurements
> stand -- the engines' exact agreement, the header-hash pinning, the
> two-copy verification -- but where the text below describes the door
> as Python-decoding, `ENGINE_GAPS` as non-empty, or `--mapped-engine`
> as refused on the route, it describes the world it measured, and 11.3
> is the gap the port then closed.  The route now composes in the
> engine and forwards `--mapped-engine` like every other composed
> source.

Row `twentycrv3-member-pl-sfc` runs on this box and passes.  The full
battery at this section's base commit is **120 passed, 17 skipped** with
`GPUWM_20CRV3_MEMBER_SAMPLE` unset, and the four member rows are not
among the skips; re-run alone at the tip that carries this section they
are 4 passed -- the default path resolves and the golden reproduces.
That is worth stating plainly because it was believed otherwise, and
because a lane acting on the belief nearly traded a passing real-bytes
row for a sentinel default.  The box holds the archive TWICE, under two
folder names; the default names one of them.

What follows was measured by pointing the environment override at the
SECOND copy, which is what a second box would look like.  That the two
copies hold the same bytes is not assumed: 26 files and 30.84 MB by
count and size, and the golden's own `inputs` table -- measured under
copy A -- reproduces its SHA-256 for every file the row names when the
decode is run from copy B.  Three things came out of it, and only one of
them was the answer the row was written to get.

**The two engines agree exactly.**  Decoded at the same input path so no
path can differ between them, the Rust engine and the Python engine
produce parity digests with ZERO differing leaves, on both `decode` and
`inspect`: 2 frames, 18 fields each, every field array SHA-256, every
axis digest, the grid fingerprint, the input hashes and the frame header
hashes identical.  Section 11.2 extends the same comparison from the
row's two valid times to all thirteen.  (This is the DECODE seam; the
member front door does not reach it -- see 11.1.)

**The golden's frame header hash is pinned to the box that measured
it.**  Against the committed golden, four leaves differ and no others:
`decode.frames[0..1].header_sha256` and the same two values inside
`inspect.inspection.materialization.frame_header_sha256`.  Every array
digest, every input SHA-256 and every axis hash matches, which proves
the bytes are the ones the golden was measured on.  The cause is exact
rather than suspected: the canonical frame header carries each field's
`source_field` provenance as `<absolute input path>:<record index>`
(`mapped_source._materialize_frames`), and the header is hashed BEFORE
`machine_independent` reduces paths to basenames -- so the mask that the
battery's own docstring says covers "absolute input paths" cannot reach
inside a hash.  Substituting the original root back into the header dict
and re-hashing reproduces the committed golden bit for bit, both frames.

This is not a member-row property.  Sixteen goldens carry that hash --
six decode (`gdas-pgrb2-0p25`, `gefs-ensemble-control`, `hrrr-prs`,
`rap-awip32`, `rrfs-prslev-2dfld`, `twentycrv3-member-pl-sfc`) and ten
compose -- and every one of them is reproducible only on a box whose
input root is spelled the way the measuring box's was.  All sixteen pass
today because no root has moved.  What makes this row the demonstration
is that the SAME BYTES exist here at two paths, so the defect can be
shown without moving anything: decoded from copy A the golden
reproduces, decoded from copy B it fails on two hashes and nothing else.
Move the staging tree and fifteen more rows do what copy B just did.
Closing it is a change to what the PRODUCT records, not to the battery: the
header would have to carry `<basename>:<record index>`, in
`gpuwm.mapped_source` and in the engine's `frames.json` writer together
or the two stop agreeing, and all sixteen goldens re-measured behind it.
The goldens were deliberately left untouched here rather than moved
under `--force` by a lane that was measuring one row.

**A compose golden for this route cannot be committed yet, and the
reason is not the bytes.**  `--source 20crv3` runs
`twentycrv3_member_grib2_v1`, not `mapped_composition_v1`;
`composed_recipe` asserts the runner, because a compose row for a door
that never calls `decode_composed_source` would describe a route no user
can reach -- it would grade byte work that no `gpuwm prep` performs.
That door is the standing `ENGINE_GAPS` entry, and it is what the
compose exemption for this source actually is.  It closes when the
runner is ported, not when bytes appear -- and 11.3 measures how far
that is, which is nearer than the entry reads.

### 11.1 The member front door, dual-run on those bytes

`gpuwm prep --source 20crv3 --author-only` authored the sealed member
manifest from the real archive: 26 files, member `072`, 13 valid times
at a 10800 s cadence, `member_identity =
filename_memNNN_not_grib2_pdt` -- the `filename_member_manifest_v1`
ingest end to end on bytes whose PDT carries no ensemble identity at
all.

The run door was then driven three ways -- bare default,
`GPUWM_MAPPED_ENGINE=python`, `GPUWM_MAPPED_ENGINE=rust` -- against that
manifest.  All three child commands are byte-identical and all three
forward `--grib2-inventory` and `--grib2-dump`.  That is the gap
behaving as `ENGINE_GAPS` describes it, now measured through the real
door on real member bytes rather than a stub manifest.

It also surfaced a defect the estate had already declared fixed:
`--mapped-engine rust` was ACCEPTED on this route, exit 0, launching
that same Python-decoding child, while `docs/cli-reference.md` said
"`--mapped-engine` is refused on it".  A caller asking for the Rust
engine got the Python one and no sign of it.  The stopgap was a refusal
in `_required_twentycr_args` for both spellings; the port then retired
the refusal by making the flag TRUE instead -- the route composes in
the engine, follows the engine table, and forwards the flag to the
child, so asking for `rust` now gets rust.  Either way, the defect this
history names -- an asked-for engine that never arrives -- stays dead.

### 11.2 The whole series, not the battery's two times

The row names four files -- the `pl`+`sfc` pair at two valid times --
because a row names its inputs exactly.  The archive holds thirteen
times, and what it drives was measured separately: all 26 files through
both engines produce **13 frames, 18 fields each, 23 pressure levels,
one grid fingerprint** (`5711cdb0...3702`), spanning
1932-03-21T00:00:00 to 1932-03-22T12:00:00 at the 10800 s cadence the
adapter declares.  So the sample satisfies the packaged profile's level
ladder (`23-pressure-level-to-explicit-wrf-eta-v2`) and its cadence
(`uniform-paired-three-hour-member-series-v1`) over its whole span, not
just at the row's two times.

Engine parity was then measured over that whole span rather than over
the row: **234 field array SHA-256s across 13 frames, plus every axis
hash, grid fingerprint and frame header hash, IDENTICAL between the
Python engine and the Rust engine.**  Both decode at the same input
path, so nothing in that comparison can be a path artefact -- which is
what separates it from the golden comparison in section 11.

### 11.3 The size of the remaining gap, measured

`ENGINE_GAPS` reads as though the member door is waiting on machinery.
It is not.  The packaged profile `20crv3-member-grib2-v1` is already in
composition shape -- `composition_state: composed`, `data_role:
twentycrv3_in_band_surface`, `provenance_role:
twentycrv3_in_band_surface_provenance` -- and the composition authority
beside it is a `gpuwm-mapped-composition-v2` document declaring one
`terrain_height` supplement under `exact_coordinate_subset` +
`valid_time_exact` + `require_invariant_across_time`, plus the four-layer
conservative soil remap.  Every one of those is a clause the ported
compose implements.

Asked directly, it works: `decode_composed_source` on the packaged
authorities, the member `pl`+`sfc` pair as primary and the `sfc` files as
the terrain supplement, **on the bare default engine, composes 2 frames
of 18 fields** -- `terrain_height`, `soil_temperature` and
`volumetric_soil_moisture` among them, which is exactly the state a
`decode` of these bytes cannot reach alone.

So what stands between `gpuwm prep --source 20crv3` and the engine is
the ADAPTER, not the composition: `runner="twentycrv3_member_grib2_v1"`
and `gpuwm.twentycrv3_wrf`'s host-Python bundle build.  Porting it is a
routing change plus an equivalence proof (the sealed member manifest,
the member ordinal in the receipt and the alignment receipt all have to
come out the same), not new byte work.  That port has since landed --
section 8 describes the result: `ENGINE_GAPS` is empty, the member
identity rides the composition input manifest, and the last Python
data-path door in the preparation estate is closed.  This section's
measurement is why the entry could be called nearer than it read.
