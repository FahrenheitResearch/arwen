# Ensemble output contract

Schema: `gpuwm-ensemble-output.v2`. Each forecast domain writes
`<run>/<domain>/ensemble-manifest.json`. Paths in that manifest are relative
to `<run>`. The worker reads the manifest after each atomic replacement.
The run-level manifest may link these domain manifests.

| Entry | Meaning |
| --- | --- |
| `members_requested`, `member_order` | Complete roster, with zero-based stable member indices. Packing and device assignment never change this order. |
| `products` | Field identifiers, stored units, float32 thresholds, comparison, paintball, spaghetti and postage-stamp requests. |
| `events` | Named weather conjunctions, with every field, stored unit, float32 threshold and comparison declared. |
| `unavailable_products` | Diagnostics missing from the configured physics or incomplete accumulation windows. This never changes physics. |
| `frames[].valid_time`, `domain` | Exact output clock and forecast domain. Parent and child domains have separate grids. |
| `frames[].status` | `pending` until every member at this time has arrived and all declared files and Rust maps are finished; then `complete`. |
| `members_received`, `members_expected` | Explicit roster coverage. A pending frame has no public probabilities. A failed member is never dropped from the denominator. |
| `frames[].diagnostic_files` | Pending private scratch paths and the member indices they contain. They are never probability products or member histories. |
| `frames[].products` | Completed CDF5 aggregate files at `<domain>/ensemble/products/ensemble_<valid>.nc`. |
| `frames[].maps` | Finished Rust PNGs at `maps/<domain>/<product>/<valid-day>/`. |
| `keep_member_files`, `member_files` | Member histories are opt-in. The default list is empty. Each retained file has a member index, domain, path and byte count. |
| `deleted_scratch` | Exact paths and byte sizes of diagnostic scratch removed after the complete frame manifest is durable. |

The aggregate file uses `ensemble_contract=gpuwm-resident-ensemble-products.v1`,
`ensemble_members=N`, `valid_time`, `spread_ddof=1`, and
`probability_scale=fraction`. Every field has `*_mean`, `*_spread`, `*_min`,
`*_max`, and `*_finite_count` on `(south_north,west_east)`. Spread is zero
for one member. Any nonfinite member masks the aggregate cell; finite count
records coverage. Forecast member arrays are const inputs to reductions.

Threshold fields have `*_thresholds` and `*_probability`, with an explicit
comparison (`ge`, `gt`, `le`, or `lt`). Paintball is `uint64` membership words:
bit `position % 64` in word `position // 64`, where `position` indexes the
manifest's `member_order`. That order retains original global member IDs,
including sparse individual replay. The native reader keeps integer
words exact, including bits above 53. Optional `*_spaghetti` stores four
threshold corner bits per member cell. Optional `*_members` contains only
the requested two-dimensional headline diagnostic, used for Rust postage
stamp pages of up to 16 members. These planes exist only in the private
render-source CDF5 cache and are removed after rendering and durable manifest
publication. The public aggregate CDF5 omits them. These fields are not member wrfouts.

`NativeDiagnosticSpool.submit(fields, member_ids=..., valid_time=...)` accepts
contiguous resident float32 `(pack_member,y,x)` diagnostics. It writes small
CDF5 diagnostic packs through the Rust writer. `finish(valid_time,
available_bytes=...)` replays bounded row tiles through the Rust typed decoder,
reduces on the GPU in global member order, then renders through
`gpuwm.rustwx.run_ensemble_product_renderer`. This preserves the all-resident
two-pass mean and sample-spread arithmetic across cards and sequential packs.
Full-roster products stream as soon as a frame becomes complete. If packs run
whole forecasts sequentially, complete products appear while the last pack runs.
Input streaming remains the forecast runner's responsibility and stays enabled.
The optional `unavailable_fields` argument explicitly lists requested fields
missing at this time. Every pack at that time must have the same availability;
the frame records `available_fields` and `unavailable_fields` separately.
Replay reads each field/pack once into the requested 2-D postage cache, then
uploads bounded row tiles. Without postage, it retains at most one extra
complete 2-D member field during replay. The manifest records device replay
bytes, host product bytes and this extra host cache.

`HeadlineDiagnosticCollector` is the ordinary writer callback adapter. It takes
`state`, the actual `streamed` marker, `metadata`, the consumed `refl_field`,
`valid_time`, `grid_id`, `episode`, and `member_id`. Resident fields come from
the live physics diagnostics. Streamed fields come from the actual host or
deferred store frame. The stale resident state of a streamed domain is never
used. The callback completes before another sweep may change its borrowed
fields. Surface wind, precipitation, composite reflectivity, dewpoint and
relative humidity are diagnosed on the GPU. Gust requires a real gust field;
a running maximum of 10 m wind is never renamed as gust.

`capture_rain_counters(fields, valid_time=..., grid_id=..., episode=...,
member_id=..., latitude=..., longitude=...)` observes only the original
precipitation providers. The forecast runner seeds it before the first step
independently of `history_begin`, then observes exact requested QPF endpoints.
It creates no history or product frame. An output at an already captured
counter tick is allowed; duplicate output frames remain an error. The
optional `absent_zero_fields` names provider counters the original scheme
declares as structural zeros. A missed adaptive endpoint is unavailable;
this observer never clips a step or interpolates a counter.

The collector exposes `receipt()`, `finish_run()`, `require_complete()` and
`add_member_file(path, member_id=..., grid_id=...)`. Forecast completion and
product completeness remain separate. After a run, a missing aligned member
frame is `incomplete`. Member grids that differ at the same valid time are
`unavailable` with the coordinate mismatch recorded. Neither publishes a
partial-roster probability. Retired diagnostic scratch is removed by exact
owned paths. Coordinate changes use their own domain/grid manifest and maps.

The runner resolves diagnostics from the actual configured physics. Rolling
QPF windows have exact start/end output times; an incomplete window is listed
as unavailable rather than labeled as a full accumulation. The headline
threshold table supports QPF, surface wind and gust, reflectivity, temperature,
dewpoint, relative humidity and updraft helicity. Compound conditions use the
same complete-roster denominator. Threshold changes alter products only.
`HeadlineDiagnosticCollector(events={name: conditions})` accepts conjunctions
of `ThresholdCondition(field, units, threshold, comparison)` terms. Conditions
use the headline table's stored units. An event is unavailable when a required
diagnostic is unavailable, and nonfinite terms mask that member's event. No
fire ignition or fire spread quantity is inferred from a weather conjunction.

The source executor accepts `PreparedEnsembleSession(member_roster=...)` or
`run_prepared_ensemble(..., member_roster=...)`. The verified roster binds
original IDs, seeds, actual member inputs, initial fields and complete boundary
tables, with source and donor manifest hashes. Source metadata does not select
another atmosphere. Distinct native inputs use their original initializer;
only qualified common-source packs use the native shared-clock executor.

`retain_member_diagnostics=true` enables the native member-axis surface store
under `member-diagnostics/manifest.json`. Its thin CDF5 packs retain T2 in K,
U10 and V10 in m s-1, and cumulative precipitation since forecast start in mm.
Each pack carries the original uint64 member ID and seed, recipe/source
provenance, valid time, accumulation start, coordinate hash and coordinates.
These files contain no forecast volumes. Initial and exact hourly history
frames are retained; missing fields are explicit and never borrowed from
another member. Calibration runs must request initial and hourly history.
The archive manifest records each original member's forecast calendar and
lists missing member/grid/valid-time planes. Coverage is complete only after
finalization and every requested plane is present. Delayed domains retain
their actual initial time, even between hours. Spawned, retired or rearmed
domains have explicit dynamic coverage because a placeholder declaration
does not describe their realized episode windows. This coverage report does
not change the original clock or force additional history output.

The configuration door reads `[ensemble]` or run-plan `run_options.ensemble`.
`gpuwm go CONFIG --members N`, `gpuwm ensemble CONFIG --members N`, and
`gpuwm run CONFIG --members N` use that same request.

Every member of an N > 1 ensemble runs its own inputs. A member count with no
recipe takes the source's operational ensemble, where its adapter row declares
a runnable one (see `docs/ENSEMBLE_PROVIDERS.md`): the door prints the member
plan, fetches and prepares each member through that source's own chain, and
runs them, as it does for `--recipe time-lagged` and `--trajectories FILE`.
`--dry-run` prints the plan and fetches nothing. Where the row declares none,
the door exits 2 before any download and names the remedy. N members never
run one input: they would be N copies of one forecast, with zero spread and
probabilities of 0 or 1. For the same reason N > 1 is refused, by name and
before any work, at the doors that hold one input: `--wrfinput` and
`--met-em`, `--prepared-root`, `--restart`, `--data-dir`, `gpuwm resume`,
`gpuwm branch`, and a run plan on a route that prepares one trajectory.
`gpuwm sim` and a `[grid]`/`[dynamics]`/`[run]` config open no ensemble
session and refuse an `[ensemble]` table, whatever its member count, instead
of dropping it. One member is unchanged on the doors that open a session, and
the `--wrfinput` and `--met-em` doors carry a one-member request to their
worker.

Supervised execution
reads the captured configuration bytes. Aggregate-only is the default;
`--keep-member-files` retains full
member histories. A completed run links its domain product manifests from
`ensemble-run.json`. Product rendering is already complete at the normal
render-stage handoff and is verified there through the durable manifests.

The supervised success capsule records the path and SHA256 of
`ensemble-run.json`. Ensemble progress projects the original member times
onto their mean and sums original reported member steps. The run manifest retains
each original member's time, step, attempt and checkpoint ownership. A
partial member checkpoint does not represent a restart of the whole roster.
Only the enclosing run publishes completion after its products are durable.

`ensemble-run.json` ends in one of three states. `PASS` is a completed roster.
`failed` names what failed: `failed_members` lists each failing batch with its
original member IDs, card, wave, execution mode, error type and error text.
The error text begins with `member N:` (or `members N, M:` for a pack) when
the error's text is its plain message; `failed_members` always carries the
member IDs, so read the IDs from there. When several cards fail in one
wave, every one of them is listed. `interrupted` is a run stopped by Ctrl-C or
a stop signal: members not yet started are cancelled, running members end at
their next model step, and the door exits 130 with one sentence. In both
unfinished states `members_not_completed` lists the rest of the roster and no
aggregate product is closed as complete. A run with no supervising process
writes each member's start, progress and finish to standard error.

Requested `[simulated_radar]` products retain the original member runner,
volume history writer and live native radar consumer. They render during the
forecast through the original Rust radar path. This option requires temporary
member volume histories even with aggregate-only output. After that member's
original radar queue closes, the executor verifies its Rust manifest and every
listed artifact, then deletes only the original history paths returned by that
runner inside the new member directory. Failed radar work keeps its histories
for recovery. `keep_member_files=true` keeps those frames.

The run manifest's `member_radar_products` records the original member ID,
member directory, relative `radar/manifest.json`, manifest SHA256, verified
volume/image/loop artifacts, warnings and exact retired history paths and byte
sizes. These products are per-member simulated radar. They do not replace the
ensemble headline reflectivity probabilities. Radar's original native host
memory admission remains active; no polar volume is allocated on the GPU.
