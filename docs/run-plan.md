# `gpuwm run-plan` — driving gpuwm from a program

Every other front door in gpuwm talks to a person. `gpuwm run-plan`
talks to a program: one versioned JSON plan in, one append-only JSONL
event stream out. A GUI, a scheduler or a fleet controller drives it as
a subprocess and never parses a line of human output.

Nothing about the model changes. The plan is an **envelope** over the
config system — it resolves through the same loader `gpuwm run` uses and
executes through the same `runtime.run_experiment`. There is no second
config format.

```
gpuwm run-plan PLAN.json          # run it, stream events
python -m gpuwm.runplan PLAN.json # the same command, same flags

gpuwm run-plan --resolve  PLAN.json   # what will run, and every value nobody typed
gpuwm run-plan --estimate PLAN.json   # what it will cost
gpuwm run-plan --probe                # what this machine can do
```

Exit codes: **0** when the last event was `completed`, **1** when it was
`failed`, **2** on a refused plan (nothing started), **130** on Ctrl-C.

---

## The plan document — `gpuwm.run-plan.v1`

```json
{
  "schema": "gpuwm.run-plan.v1",
  "name": "overnight-run",
  "route": "experiment",
  "config": { "path": "conus3km.toml" },
  "output_root": "runs/overnight",
  "fetch":  { "args": ["--source", "gfs", "--cycle", "2026080700", "--hours", "12", "--out", "data"] },
  "run_options": { "device": 0, "dry_run": false, "restart": null, "health_debug": false }
}
```

| key | required | meaning |
|---|---|---|
| `schema` | yes | must be exactly `gpuwm.run-plan.v1` |
| `name` | yes | this run's label; carried on every event |
| `route` | yes | which existing front door executes it (see below) |
| `config` | yes | exactly one of `{"path": ...}`, `{"inline": "<TOML text>"}` or `{"intent": {...}}` |
| `output_root` | no | **the run directory itself**, not a parent. Default `out/run`, gpuwm run's own |
| `fetch` | no | `{"args": [...]}` — the argv list `gpuwm fetch` itself takes |
| `run_options` | no | the subset the route declares |

Relative paths resolve against **the plan file's own directory**, so a
plan and its config travel together.

Unknown top-level keys, unknown `run_options`, an unknown `route` and an
unknown `schema` are all **refused**, never ignored: a dropped key runs
a default under the name of your value.

`fetch.args` is validated by building gpuwm's **real** fetch parser and
handing it the list. There is no second copy of the fetch flag table
here, so a flag added to `gpuwm fetch` is accepted here on the same
commit, and a typo is refused before the run claims a directory.

### `config.intent` — submitting a shape instead of a config

A front end that collects "here, this big, this long, from this source"
has intent, not a config. `config.intent` hands that intent to the
`gpuwm domain` wizard, which writes the complete TOML — the same wizard,
the same refusals, the same emitted bytes a person gets from the CLI.

```json
"config": { "intent": {
  "point": "35.2,-97.4",
  "ladder": "12-3",
  "source": "era5",
  "cycle": "latest",
  "hours": 12,
  "vram_gib": 24
}}
```

Every intent key is a `gpuwm domain` flag, one to one. Nothing here is a
second config format, and no value is validated twice — the wizard's own
parser is what accepts or refuses each one.

| intent key | flag | intent key | flag |
|---|---|---|---|
| `point` | `--point LAT,LON` | `hours` | `--hours` |
| `polygon` | `--polygon` | `source` | `--source` |
| `buffer_km` | `--buffer-km` | `cycle` | `--cycle` |
| `projection` | `--projection` | `forecast_start_hour` | `--forecast-start-hour` |
| `name` | `--name` | `data_dir` | `--data-dir` |
| `card` | `--card` | `forcing` | `--forcing` |
| `vram_gib` | `--vram-gib` | `vtable` | `--vtable` |
| `ladder` | `--ladder` | `geog_root` | `--geog-root` |
| `root_dx_km` | `--root-dx` | `chain` | `--chain` |
| `physics_profile` | `--physics-profile` | | |

`point` or `polygon` is required (there is no default place), and so is
`cycle`. `--out` is deliberately **not** exposed: run-plan owns where the
generated config lands, so a plan cannot write outside its run directory.

The generated TOML is carried **verbatim** on the `resolved_plan` event
and in `--resolve`, as `generated_config`. The caller never typed it, so
it is the one thing they cannot look up — show it.

**There is no `nx`/`ny`.** Domain size is *fitted* by the wizard's VRAM
estimator from the ladder and the card budget; it is an output, not an
input. Each fitted size is reported in `automatic_resolutions` with
`basis: "fitted_to_vram_budget"`.

**Output cadence is a first-class control.** `history_interval_s` sets
the root domain's write interval and `nest_history_interval_s` every
nest's; they default to 3600 s and 900 s. Both map to new `gpuwm domain`
flags (`--history-interval`, `--nest-history-interval`).

The engine's rule is that a cadence must be a whole number of seconds
**and** a whole number of *that domain's* time steps, judged against the
exact rational `dt` — so a nest's cadence must divide the nest's step,
not the root's. The wizard round-trips its emitted bytes through the
real loader before writing, so a bad value is refused with the loader's
own sentence and no file lands.

#### Intent is ERA5-only on the `experiment` route

`gpuwm domain` writes a `[case_data]` table — the declared inputs a
config-driven run needs — **only for `source = "era5"`**, because the
config-driven route decodes native GRIB1 = ERA5 today. A GFS or HRRR
emission is a real config, but its consumer is the native/prepared front
door, and those are not run-plan routes yet. An intent plan naming
another source on this route is refused up front, with that reason.

### Routes

| route | what it is | source |
|---|---|---|
| `experiment` | the config-driven route: one experiment TOML with its `[case_data]` inputs, prepared and integrated in this process — what `gpuwm run CONFIG` executes | era5 |
| `prepared` | the native prepared-cache chain, in the documented order | **gfs** or **hrrr**, single domain |

**The credential-free path is `prepared` + `gfs`.** ERA5 needs a
Copernicus CDS key; GFS is public. A `prepared` plan therefore runs end
to end on a machine with no credentials at all.

The route reads the config's own `[fetch].source` and drives that
source's documented chain. Neither chain is re-implemented here.

**gfs** runs `gpuwm go` — the documented chain, and the only thing that
relays the integrity digests between stages without a person carrying
them. run-plan builds the same argparse namespace the `go` subcommand
builds and hands `go_main` an observer, with trees allowed: a
multi-domain config keeps the same preparation stages and dispatches
the forecast to `gpuwm-prepared-tree-forecast`, binding the sha256 of
the hierarchy document rw-wps left in the prepared root. `go`'s other
refusals (an ERA5 `[case_data]` config, another source) surface
verbatim as a `failed` event, and the interactive `gpuwm go` command
keeps its own tree refusal.

**hrrr** runs its own chain, because `go` refuses the source by
construction (`ORCHESTRATED_SOURCES = ("gfs",)`): fetch →
`tools.prepare_hrrr_wrf` → `prepared_single_domain_forecast`. The
stages and their order are the wizard's own printed chain
(`domain_wizard.hrrr_route_commands`), driven rather than printed, with
every stage's refusals left alone. It reuses `go`'s stage primitive, so
capture, heartbeats and failure replay are identical on both chains.

One thing is *added* to the HRRR chain: **`--wps-namelist`**. The
runner's HRRR manifest requires a `wps_namelist` role (the prepared
cache identity's `namelist_sha256` **is** that file's digest on this
route), and the preparer only publishes the portable bundle — `proof.json`,
the role-keyed source manifest, the experiment authority — when handed
that flag. The printed chain never passed it, so the bundle it produced
could not be read by the single-domain runner at all and HRRR was sent
to a benchmark script instead. Passing it makes a single-domain HRRR
bundle structurally identical to a GFS one at the run step: same runner,
same digests, same in-process observer.

The forecast stage's inputs come from the preparer's published
`portable_bundle` handoff (in `public-wrapper-result.json`), not from
re-derivation. Three of them are not guessable: `proof.json` lives at
the **output root**, not inside the prepared cache; `--prepared-root` is
that same root; and `--experiment-config` / `--wps-namelist` must be the
**published copies** (`experiment.toml`, `namelist.wps`), because the
runner checks each supplied file's name and digest against the portable
manifest. The relayed digests are cross-checked against `proof.json` on
disk before the forecast starts.

Multi-domain HRRR is refused with a sentence naming the limitation and
the chain that does run it (`gpuwm.hrrr_hierarchy_direct`, then
`gpuwm-prepared-tree-forecast`); see the note at the end of this file.
Multi-domain GFS runs, as above.

`physics_profile` is passed to the HRRR preparer only when the plan
states it (as an intent key or a run option). The route owns its own
physics gate, the emitted TOML records physics as numbers rather than a
profile id, and a default invented at this layer would silently outrank
the preparer's own.

`prepared` additionally takes the `data_dir` run option (where the
fetch lands). The `experiment` route does not: it declares its inputs
in `[case_data]` instead.

Routes are named generically. A route is a way of running the model,
never a particular experiment.

#### Moving nests (`[relocation]`) by route

A config with a `[relocation]` follow source (`[relocation.follow]` or
`[[relocation.move]]`) runs on the `experiment` route as before: the
case-data process holds the GEOG source and wires the real-data
relocation runner (footprint-rebuilt statics per move).

On the prepared chains the same config needs the bundle prepared with
**`--statics-corridor`**: the preparation seals child-resolution
statics over each child's whole parent extent beside the other
hierarchy artifacts, digest-bound into the preparation receipt, and the
tree runner (`gpuwm-prepared-tree-forecast`) then crops each new
footprint's statics out of that corridor at move time — the run stays
fully sealed, no runtime ingest. The GFS prepare stage adds the flag
itself whenever the experiment config declares a follow source (the
printed `rw-wps` line from `gpuwm fetch --author-front-door-manifest`
carries it too, from the same predicate). A bundle prepared *without*
a corridor refuses a follow config at the tree runner's preflight, with
the flag named as the remedy; a corridor that fails digest or geometry
verification refuses loudly rather than running the nest silently
static. Bounds-only `[relocation]` (no follow source) never needs a
corridor. The corridor is disk/host-side only (97 float64 planes,
776 bytes per corridor cell — a 900x900 corridor is ~629 MB); it adds
no GPU memory, so the VRAM gate is unchanged.

### `run_options`

| option | default | meaning |
|---|---|---|
| `device` | `null` | GPU index or full `GPU-…` UUID; sets `CUDA_VISIBLE_DEVICES` before anything can create a context |
| `dry_run` | `false` | resolve and validate, emit `resolved_plan`, stop before any device work |
| `restart` | `null` | a `gpuwmrst` checkpoint to continue from |
| `render_products` | `null` | which products the render stage draws — `gpuwm render --products`' own spec (a comma-separated list, or `all`), or `none` to skip rendering. Absent leaves the default set unchanged. `prepared` route only |
| `geog_root` | `null` | static geography tree (`prepared` route only) |
| `data_dir` | `null` | where the fetch lands (`prepared` route only) |
| `physics_profile` | `null` | passed to the HRRR preparer when stated (`prepared` route only) |
| `health_debug` | `false` | enable debug phase health attribution |

`run-plan` integrates in **this** process rather than re-executing under
gpuwm's own supervisor, so the pid in the manifest is the pid doing the
model work and the caller owns restart policy. That choice is reported
in `automatic_resolutions` on every run, never assumed.

---

## The event stream — `gpuwm.run-plan.event.v1`

One JSON object per line, appended to `<run_dir>/events.jsonl` **and**
mirrored verbatim to stdout. Every line carries the same four envelope
keys, with the event's own fields flattened alongside:

```json
{"schema_version":"gpuwm.run-plan.event.v1","sequence":7,"emitted_unix_ms":1786087223348,"event":"model_progress","domain":1,"outer_step":3,"model_seconds":180.0,"wall_seconds":4.21,"speed_x":42.75,"step_ms":1403.2,"phase":"post-d01-sync"}
```

`sequence` is monotonic **and dense**, starting at 1. A gap means a lost
or reordered line, never a skipped one, and the reader refuses it.

| `event` | fields | when |
|---|---|---|
| `plan_accepted` | `name`, `route`, `plan_source`, `plan_sha256`, `run_dir`, `manifest_path`, `events_path`, `pid`, `run_id` | first line, always |
| `resolved_plan` | `configuration`, `automatic_resolutions`, `config_sha256`, `config_source`, `run_options` | after the config loads, before any device work |
| `stage_started` | `stage`, `phase` | a stage opens |
| `stage_finished` | `stage`, `wall_seconds`, `phases`, (`receipts`, `outcome`) | that stage closes |
| `model_progress` | `domain`, `outer_step`, `model_seconds`, `wall_seconds`, `speed_x`, `step_ms`, `phase`, (`domains`) | each outer step |
| `output_committed` | `domain`, `valid_time`, `path` | a wrfout is durable on disk |
| `model_progress` (polled) | as above plus `source: "stage_progress_file"`, `step_ms: null` | a `prepared` stage that runs as a subprocess, sampled from its own progress file |
| `warning` | `code`, `message`, (`detail`) | anything worth saying, nothing worth stopping for |
| `completed` | `dry_run`, `run_dir`, `receipt_path`, `receipts`, `outputs_committed`, `summary` | last line, exit 0 |
| `failed` | `stage`, `error_class`, `message`, `remedy`, `run_dir`, `receipts` | last line, nonzero exit |

`stage` ∈ `fetch`, `prepare`, `initialize`, `forecast`, `finalize`.
`fetch` appears only when the plan declares one; every other stage
always emits its pair, so a stage timeline has no holes to interpret.
The finer pipeline phases inside a stage are not lost — they arrive on
`stage_started.phase` and the full ordered list on
`stage_finished.phases`.

`speed_x` and `step_ms` are `null` rather than an infinity when no wall
time has elapsed yet. A rate over no elapsed wall is undefined, not
large.

`domains` appears only on a domain tree with more than one clock: a
list of `{domain, model_seconds}` giving each grid's own clock, the
`domain`/`model_seconds` pair at the top level staying the root's. Its
absence means the root **is** the tree, so single-domain consumers see
the stream they always saw.

`output_committed` is raised at the moment the file is genuinely durable
— after the writer has fsynced it, validated it against its own
inventory and renamed it onto its final name — so the event never
announces a file that is merely queued. On the domain-tree route it
arrives from the per-domain writer thread, which is why the stream
serializes writes under a lock.

---

## `cycle: "latest"`

`latest` was already implemented across the fetch machinery before this
front door existed; run-plan does not reimplement it. What it adds is
that the **resolved** cycle is recorded rather than the question.

`resolve_latest_cycle` walks candidate cycles backwards (GFS/GDAS: 6-hourly,
~48 h; HRRR: hourly, 12 h) and accepts the first whose objects for the
**final requested forecast hour** are all published on S3 — for HRRR both
the `wrfnat` and its `wrfprs` sibling, since during a live publication
one can appear before the other. So **completeness is structural**: a
partially uploaded cycle cannot win, and the fetched window is complete
by construction. There is no "cycle still receiving files" state to
detect, because such a cycle is never selected.

run-plan resolves `latest` **before** the fetch stage runs and rewrites
the argv with the concrete cycle, for two reasons: a plan that records
`latest` records a question whose answer changes every six hours, and
resolving once removes the window where this front door could report one
cycle while the fetch downloads the next. It is the rule the wizard
already applies to its emitted `[fetch]` table — *the resolved cycle,
never the literal `latest`*.

The concrete cycle lands in `automatic_resolutions`:

```json
{"scope": "fetch", "key": "cycle", "value": "2026-08-07T12",
 "basis": "resolved_latest",
 "note": "the newest cycle whose objects for the final requested hour are all published; a partially uploaded cycle cannot win, so the window is complete by construction"}
```

If the resolved cycle is not the newest the clock allows, newer cycles
exist but are still publishing — the run will initialize from older data
than a caller may assume. That is a **`warning` event, never a refusal**:

```json
{"event": "warning", "code": "latest_cycle_is_not_the_newest",
 "cycle": "2026-08-07T00", "source": "gfs", "age_hours": 20, "last_hour": 6}
```

`latest` is matched case-insensitively here. `gpuwm fetch` compares bare
equality while the wizard and the interactive door lower-case first, so
`--cycle Latest` works on two of the three front doors today; a machine
interface should not inherit that coin flip.

ERA5 has no `latest` — it is a reanalysis published days late, and both
the fetch and the wizard refuse it by name.

---

## Planning before the data exists

`--resolve` and `--estimate` answer *what would this run be*, which has
to work before anything is downloaded. They load the config with the
input-existence check off, so the geometry, physics and VRAM estimate
come back from a plan whose forcing has not been fetched yet.

Nothing is skipped silently. The answer carries the full inventory:

```json
"inputs_present": false,
"declared_inputs": [
  {"role": "forcing", "path": ".../era5-combined.grib", "kind": "file", "present": false},
  {"role": "vtable",  "path": ".../Vtable.ERA5_CDO",    "kind": "file", "present": true},
  ...
]
```

A **run** keeps the check on. The `resolved_plan` event carries the same
inventory, then the fetch stage runs, and the gate fires after it — one
refusal naming every missing input, rather than discovering them one at
a time inside preparation.

## The minimum domain size

`--resolve` reports the floor, and it is **derived from the engine**, not
transcribed:

```json
"domain_size_floor": {
  "root_mass_points": {"nx": 60, "ny": 48},
  "nest_span_mass_points": 12,
  "clearance_rows": 10,
  "basis": "the wizard's fit loop bisects grid scale between _MIN_SCALE and _MAX_SCALE; ... Domain size is FITTED from the ladder and the VRAM budget -- there is no nx/ny input to set."
}
```

60 × 48 is the smallest layout that still hosts the deepest ladder with
full Davies/blend clearance; a nest span below 12 mass points is refused
outright; `clearance_rows` is `spec_bdy_width + blend_width`. The numbers
are computed from the wizard's own fit bracket at call time, so they move
when it moves — a constant copied into a front end would be right until
somebody tuned the bracket, then wrong silently.

When a shape does not fit, the refusal carries the wizard's own sentence
(which names the budget, the layout it bottomed out at, and what to
change) **and** this floor beside it.

---

## stdout is the machine channel

stdout carries JSONL and nothing else. Everything a person would want to
read — the resolved-config report, the wizard's sizing table and next
steps, warnings, the feedback advisory — goes to **stderr**.

This is enforced, not hoped for: `run_plan_main` binds the real stdout
for the event stream and then redirects `sys.stdout` to stderr for the
whole run. The pipeline and the wizard print with plain `print()` and
are correct to; they simply do not get the machine channel.

---

## Attaching to a running run

`<run_dir>/run-manifest.json` (`gpuwm.run-manifest.v1`) is written
before any work starts and names every stream a consumer may want,
including the two `run-plan` does not own:

```json
{
  "schema": "gpuwm.run-manifest.v1",
  "pid": 24188,
  "started_at_utc": "2026-08-07T18:00:12.104Z",
  "plan_sha256": "…",
  "run_dir":     "…/runs/overnight",
  "outputs_dir": "…/runs/overnight",
  "events_path":           "…/runs/overnight/events.jsonl",
  "events_schema":         "gpuwm.run-plan.event.v1",
  "progress_path":         "…/runs/overnight/run-progress.json",
  "progress_schema":       "gpuwm.run-progress/v1",
  "failure_capsule_path":  "…/runs/overnight/failure-capsule.json",
  "failure_capsule_schema":"gpuwm.failure-capsule/v3"
}
```

### Reattach: read the heartbeat, don't own the pipe

1. Read `run-manifest.json` for the paths and the pid.
2. Read `run-progress.json` for **current state**. That file is the
   authoritative anchor — atomically republished on every step, and what
   gpuwm's own recovery reads.
3. Replay `events.jsonl` from byte zero for **history**. It is the
   complete record, never rotated or truncated. Then tail it for live
   detail.

`run-plan` publishes **no progress state of its own**.
`gpuwm.supervisor` already writes `run-progress.json`, and it stays the
only writer of it: the run-plan observer *composes* with
`supervisor.RuntimeHeartbeat` rather than replacing it, so a run-plan
run leaves exactly the heartbeat a `gpuwm run` leaves.

A consumer that treats the event stream as the anchor will be wrong
exactly once: after a crash between the last event flush and the process
exit. The heartbeat is the thing that is durable by design.

A torn final line in `events.jsonl` means the writer died mid-flush.
`read_events` refuses it by default rather than trimming it, because a
reader that silently drops a partial line cannot tell "still going" from
"died here". Pass `allow_partial_tail=True` once you have established
which.

---

## Nothing silent: `automatic_resolutions`

Every value the pipeline chose on its own appears in the
`resolved_plan` event and in `--resolve`, one entry each:

```json
[
  {"scope":"plan","key":"output_root","value":"…/out/run","basis":"front_door_default"},
  {"scope":"run_options","key":"device","value":null,"basis":"schema_default"},
  {"scope":"experiment","key":"blend_width","value":5,"basis":"schema_default"},
  {"scope":"domain","grid_id":2,"key":"dt","value":20.0,"exact":"20",
   "basis":"derived_from_parent_time_step_ratio",
   "note":"parent d01 step divided by parent_time_step_ratio=3"},
  {"scope":"execution","key":"execution_mode","value":"in_process","basis":"front_door_contract"}
]
```

A front end can render that list and a reader can see, before the run,
every number nobody typed. The per-domain timestep is the flagship case:
it changes the model's answer, nobody writes it, and until now it
appeared only inside a printed report.

Library warnings (`gpuwm.explain.warn`) reach the stream as `warning`
events with `code: "library_warning"` — the same fact as fields rather
than as a line to recognize.

---

## Query modes

Each prints exactly one JSON document to stdout and runs nothing.

**`--resolve PLAN.json`** → `gpuwm.run-plan.resolved.v1`. The fully
resolved configuration (the objects the model will actually run, not a
re-read of the TOML), `automatic_resolutions`, and any warnings the load
produced. This is the "what will this do?" answer, and it creates
nothing on disk.

**`--estimate PLAN.json`** → `gpuwm.run-plan.estimate.v1`. VRAM from
`gpuwm.core.preflight`'s own itemization — the arithmetic `gpuwm check`
reports, on the CPU, with no CUDA context. Output-frame counts per
domain, which are exact. Where this package has no measured number
(bytes per frame, download size, wall time for an arbitrary
configuration) the field is `null` **with its `basis` stated**. A front
end showing an invented duration would be showing gpuwm's name on a
number gpuwm never measured.

**`--catalog`** → `gpuwm.run-plan.catalog.v1`. The renderer's product
catalog — what may go in `render_products`. Asked of the renderer, never
transcribed: on a box with the Rust engine that is its own
`--list-products` (150 slugs plus the `all`/`direct`/`derived`/`heavy`/
`windowed` group keywords); on a matplotlib-only box it is that engine's
five, and the document names which engine answered, because a picker
built against one and run against the other would offer products that do
not exist. The parse is checked against the renderer's own declared
count, and a disagreement is reported in `parse_warning` with the raw
output carried, rather than silently returning a short list.

**`--probe`** → `gpuwm.run-plan.probe.v1`. Device inventory (name, UUID,
driver, VRAM total/used/free) read through NVML only — **no CUDA context
is created**, so it is safe to poll on a busy card. Plus route and schema
inventories, and a readiness section delegating to `gpuwm doctor`.

That readiness section **does** create a context: doctor verifies the
estate by execution, including a subprocess that imports CuPy and runs a
2×2 matmul. Pass `--no-readiness` for the NVML-only document when
polling.

---

## Python API

```python
from gpuwm.runplan import (load_plan, resolve_plan, estimate_plan,
                           probe_environment, execute_plan,
                           EventStream, read_events)

plan = load_plan("PLAN.json")
resolution, exp, data = resolve_plan(plan)         # no device work
with EventStream(plan.run_dir / "events.jsonl") as events:
    exit_code = execute_plan(plan, events=events)

history = read_events(plan.run_dir / "events.jsonl")
```

`EventStream(path, mirror=...)` — `mirror` defaults to stdout; pass
`None` for the file only, or any writable stream.

## The observer seam

`run-plan` adds no private hooks. It uses the `progress_callback`
protocol `runtime.run_experiment` already accepted, plus one new
optional hook discovered the same way the existing ones are:

| hook | added by | raised at |
|---|---|---|
| `progress_callback(**event)` | pre-existing | each outer step |
| `.preparing(phase)` | pre-existing | each preparation phase |
| `.starting()` / `.complete()` / `.failed()` | pre-existing | lifecycle |
| `.output_committed(domain=, valid_time=, path=)` | **new** | `runtime._output_committed`, and the per-domain async writer once the file is durable |

Any object carrying those attributes works. A `progress_callback`
without `output_committed` is unaffected — the hook is discovered by
name, and absent means nothing happens.

---

## What the `prepared` route does not reach yet

Single-domain **gfs** and **hrrr** both run, and so do **GFS domain
trees**: `rw-wps` prepares the whole hierarchy in one call, and run-plan
owns the relay the manual chain used to need a person for — the
forecast stage reads the hierarchy document the preparation left in the
prepared root (`proof.json` or `receipt.json`, matched on schema
against the tree runner's own table) and binds its digest for
`gpuwm-prepared-tree-forecast`. The estimate is tree-aware
(`peak_envelope_bytes` beside the pool request) and `model_progress`
carries a per-domain clock list when the tree has more than one domain.
What remains unreachable:

**Multi-domain HRRR.** Needs a third stage
(`python -m gpuwm.hrrr_hierarchy_direct`) between the preparation and
the forecast, and the tree runner rather than the single-domain one.
Refused with that named.

The enabling seams for that are already in place: both
prepared runners take `observer=` on `run_*` and on `main()`, and
`PerDomainWrfoutWriters.attach_progress_callback` binds the output hook
late. What is missing is the chain, not the observability.

---

## Selective rendering

Both chains end in `gpuwm render`, and both take the same filter:

```json
"run_options": { "render_products": "composite_reflectivity,sbcape" }
"run_options": { "render_products": "all" }
"run_options": { "render_products": "none" }     // skip the stage
```

The value is `gpuwm render --products`' own spec, passed through
**verbatim** — this front door does not parse or validate it, because the
render command owns that vocabulary and a second copy of it here is the
enumeration drift `render.py`'s own catalog code already refuses to pay
for. Absent leaves the default set exactly as it was, so `gpuwm go`'s
own behaviour is unchanged.

`none` lives in the same field as the product list rather than in a
separate boolean, so "which products" has one answer and not two that
can disagree; it is not a product name in the catalog, so it cannot
collide with one.

Ask `run-plan --catalog` for the list. It is **not** an intent key:
intent mirrors `gpuwm domain`'s flags one for one, and the wizard writes
configs, not pictures.

The HRRR chain has no render step in its printed form; run-plan gives it
`go`'s, so the option means the same thing on both sources.
