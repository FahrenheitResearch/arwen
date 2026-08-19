# The pipeline, unbundled

ArWen's pipeline is three stages, and each one runs on its own terms:

| stage | command | takes | leaves |
|---|---|---|---|
| preprocessing | `gpuwm prep` | your source files, your `namelist.wps`, your experiment TOML | a **prepared tree** |
| simulation | `gpuwm sim` | a prepared tree | `wrfout` frames + `report.json` |
| rendering | `gpuwm render` | `wrfout` frames | product PNGs |

`gpuwm go` still exists and is still the recommended first command for
a new user: it runs the documented GFS chain end to end and carries
each stage's digests to the next so you never copy a hash by hand. What
this page documents is that `go` is a **composition** of the three
stages above, not the only way to reach them. Every stage is invocable
alone, with inputs you supply, from a script you own.

If you are integrating ArWen into an existing pipeline -- you pull your
own data, you author your own namelist, you have your own scheduler and
your own plotting -- these three commands are your interface, and this
page is the contract. You should not have to read our source to write
to it.

> ArWen is a research and educational tool, never a substitute for
> official warnings from your national meteorological service.

---

## The boundary object: a prepared tree

The prepared tree is what preprocessing writes and what the simulation
reads. It is a directory, and the only thing a caller has to know about
it is this:

**A finished preparation leaves a top-level document -- `proof.json` for
a single domain, `receipt.json` for a domain tree -- whose `schema`
field names the preparation that produced it.** That document is how
`gpuwm sim` learns which source prepared the tree and whether it is one
domain or a domain tree; it is why `sim` needs no `--source`, no cycle,
no area and no config to identify what it is looking at.

Every route publishes one, `--source hrrr` included: the native HRRR
preparation writes its own completion receipt
(`public-wrapper-result.json`) beside a `proof.json`, an
`experiment.toml` and a `namelist.wps`, so the two commands below are
the whole route.

A directory with neither a bindable document nor a route's completion
receipt is a partial or interrupted preparation, and `gpuwm sim` refuses
it rather than running it. That refusal is deliberate: a half-written
tree is the one input that can produce a forecast which looks finished
and is not. A tree whose completion receipt says the preparation
FINISHED but that carries no portable authorities is a different thing
and gets a different answer -- the route's own recorded reason, and the
command that publishes them -- because telling a reader their complete
tree was interrupted sends them hunting a crash that never happened.

The digests inside that document -- the input-manifest hash, the
prepared-cache content hash, and for a domain tree the preparation
receipt -- are what the forecast stage binds against. `gpuwm sim` reads
them off the file and hands them to the runner. **The runner still
recomputes every one of them and still refuses on any difference.**
Nothing is weakened by the relay; what you no longer have to be is a
checksum courier.

To see exactly what would be run, without running it:

```
gpuwm sim PREPARED_ROOT --experiment-config e.toml --wps-namelist n.wps \
          --outdir out/run --print-command
```

That prints one line. It is the real runner invocation with every
digest filled in. Copy it into your own script and you never have to
call `gpuwm sim` again -- that is the point of publishing it.

---

## Stage 1 -- `gpuwm prep`: preprocessing, on your data

`gpuwm prep` is the same program as the standalone `rw-wps` /
`gpuwm-wrf-init` console script: one parser, one implementation, two
spellings. Every flag one accepts, the other accepts.

**It downloads nothing.** You supply the files.

### Input contract

| you supply | flag |
|---|---|
| the source adapter to decode with | `--source MODEL` (`gpuwm prep --list-sources` prints every one, with its status and its evidence) |
| your GRIB/NetCDF files, in deterministic time order | `--input FILE` (repeatable) |
| your WPS namelist | `--wps-namelist namelist.wps` |
| your WRF namelist, when the route reads one | `--namelist-input namelist.input` |
| the resolved experiment TOML | `--experiment-config experiment.toml` |
| staged WPS_GEOG | `--geog-root DIR` |
| where the prepared tree goes | `--output-root DIR` |

For a source with no named adapter, the declarative mapped route takes
an explicit field/level/cadence contract instead of a built-in one:
`--descriptor`, `--mapping`, `--composition`, and
`--author-input-manifest` to write the hash manifest that binds your
files. `gpuwm prep --show-source mapped` prints what that route
declares and, honestly, what it does not certify.

> **Known limit, stated plainly.** Preparing an arbitrary source with
> your own mapping works. **Running the result does not yet.** The
> forecast stage certifies only the *packaged* mapping -- the one the
> `20crv3` route uses -- by pinning its mapping, composition and
> provenance authorities to digests shipped with this distribution. A
> mapping you authored fails that pin, and there is no second
> certificate for a caller-supplied one yet. `gpuwm sim` detects this
> case and says so at the door rather than deep in a hash comparison.
> It is not a flag you are missing. Today the route that prepares *and*
> runs end to end is `gpuwm prep --source 20crv3`.

If you already have a WRF namelist pair and no gpuwm config,
`gpuwm import-namelist namelist.wps namelist.input --output experiment.toml`
translates them and prints a substitution report naming everything it
had to change.

### Output contract

A prepared tree under `--output-root`, containing the top-level
`proof.json` described above. Nothing else about the layout is part of
this contract -- read the document, not the directory listing.

### Worked example

```
gpuwm prep --source gfs \
  --gfs-series data/gfs-series.tsv \
  --cycle 2026-07-29_18:00:00 \
  --bridge tools/grib1_bridge/target/release/gpuwm_gfs_grib2_bridge \
  --wps-namelist authority/namelist.wps \
  --experiment-config authority/experiment.toml \
  --source-manifest data/gfs-input-manifest.json \
  --source-manifest-sha256 <sha256 of that manifest> \
  --geog-root ~/WPS_GEOG \
  --output-root prepared/
```

`--dry-run` validates the arguments and prints the internal command
without running anything.

---

## Stage 2 -- `gpuwm sim`: the forecast, alone

```
gpuwm sim PREPARED_ROOT --experiment-config TOML --wps-namelist WPS --outdir DIR
```

**No fetching. No rendering. No network.** The stage does not import
the download machinery at all, and the whole command completes on a
machine whose sockets refuse to connect -- both are asserted by
`tests/test_stage_seams.py`, not merely promised here.

### Input contract

| you supply | flag |
|---|---|
| the prepared tree | positional `PREPARED_ROOT` |
| the experiment TOML the preparation was bound to | `--experiment-config` |
| the `namelist.wps` the preparation consumed | `--wps-namelist` (single domain; unused by the tree runner) |
| where output goes | `--outdir` |
| optionally, an assertion that the config IS a shipped suite | `--physics-profile ID` |

`--runner {auto,single,tree}` selects the runner arm. `auto` -- the
default -- reads it off the bundle's own schema and domain count. The
explicit values exist so a caller who believes they know better gets
refused precisely when they do not.

`--print-command` prints the exact runner line and exits, running
nothing and requiring no GPU.

### Output contract

Under `--outdir`:

- `wrfout/wrfout_d<NN>_<time>` -- history frames, WRF-shaped NetCDF.
- `report.json` -- the run's own validity verdict: health, stability
  and input-identity gates, plus `input.boundary_interval_seconds`,
  the real spacing of the lateral boundary times it integrated
  against. **This is the file to read to decide whether a run is
  usable.** `status` is the verdict.
- `progress.json` -- republished as the run advances.

### Watching it run

The stage runs the model **in this process**, so everything the model
prints reaches your terminal as it happens rather than through a pipe.
A script that wants typed events rather than prose should drive
`gpuwm run-plan PLAN.json` instead, which emits one append-only JSONL
event stream in which every fact the human output prints is a typed
field on a typed event.

---

## Stage 3 -- `gpuwm render`: pictures, from frames that already exist

```
gpuwm render out/run/wrfout/wrfout_d01_* --out out/render
```

Takes `wrfout` files -- ours, or another model's, as long as the
variables are there -- and writes PNGs. It runs nothing else and
fetches no forcing data.

- `--products LIST` -- comma-separated product names, or `all`.
- `--list-products` -- the engine's catalog with per-file availability,
  i.e. *why* each product is or is not renderable from your file,
  instead of rendering.
- `--engine {auto,rust,matplotlib}` -- the production Rust renderer, or
  the matplotlib workaround by name. `auto` refuses when the renderer is
  unusable rather than degrading to matplotlib weather fields.
- `--source-label TEXT` -- set this when rendering frames this model did
  not produce, so the sheet does not claim them.

---

## `gpuwm go`, as a composition

`gpuwm go CONFIG` runs six stages in order: authority, fetch, front-door
manifest, preprocessing, forecast, render. `gpuwm go CONFIG --dry-run`
prints all six commands, filled in, and runs none of them -- which is
the fastest way to see how the stages above compose for your config.

Stages 4, 5 and 6 of that chain are exactly `gpuwm prep`, `gpuwm sim`
and `gpuwm render`. Not "equivalent to": the same commands.
`tests/test_stage_seams.py` asserts that the forecast command `go`
composes is byte-identical to the one `gpuwm sim` composes for the same
prepared tree, on the single-domain arm and the domain-tree arm both.
If that ever stops being true, `go` has become a second implementation
and the test fails.

So the two routes are not a choice between a supported path and an
unsupported one. They are the same path, entered at different points.

### What the chain leaves behind: `<outdir>/events.jsonl`

A bare `gpuwm go` writes an append-only event stream beside the run.
You do not ask for it and there is no flag to turn it on.

It is the same grammar `gpuwm run-plan` writes -- schema
`gpuwm.run-plan.event.v1`, one JSON object per line, a dense `sequence`
-- so one reader replays either. `gpuwm.runplan.read_events` is that
reader; `gpuwm.chain_events.stage_walls` and
`gpuwm.chain_events.summarize` are two convenience views over it.

```json
{"schema_version":"gpuwm.run-plan.event.v1","sequence":6,"emitted_unix_ms":1786988412431,
 "event":"stage_finished","stage":"fetch","wall_seconds":11.271,"ok":true,"exit_code":0,
 "bytes":41284919,"downloaded_files":3,"bytes_per_second":4712880.4,
 "verified_files":0,"verified_bytes":0}
```

What is in it:

- one `stage_started`/`stage_finished` pair per stage, named `boot`,
  `authority`, `fetch`, `manifest`, `prepare`, `forecast`, `render`.
  `boot` is the CLI's own start-up plus the memory and geography gates.
- on `fetch`: `bytes`, and `bytes_per_second` **when this run actually
  downloaded something**. A re-run against an existing `--data-dir`
  verifies rather than transfers, and reports `verified_bytes` with a
  null bandwidth instead of dividing hash time into bytes.
- `first_products_ready` with `seconds_from_launch` -- time to first
  plot, measured by the process that published the pictures.
- one terminal `completed` (or `failed`), carrying every stage's wall,
  the process wall, how much of it the stages account for, and the
  forecast's own internals read back out of its `progress.jsonl`:
  `preflight_verify`, `restore_prepared_cache`, `initialize_physics`,
  step 1's wall, and `first_step_excess_seconds` when step 1 paid
  something the steady state does not (see PROGRESS.md).

The point of it is a before/after number. Every pre-sim stage in this
tree is being rewritten in Rust, and a rewrite without a measured
baseline is an opinion. `tests/data/pre-sim-stage-baseline.json` is the
pinned reading -- with the box, the card, the date, the version and the
case stated, because a number without those is not comparable to
anything.

---

## Driving it from your own script

The shape that works:

1. Fetch your own data, however you like.
2. Author `namelist.wps` and `namelist.input`, or take them from your
   existing run. `gpuwm import-namelist` turns that pair into an
   experiment TOML and tells you what it substituted.
3. `gpuwm prep ...` on your files. Check the exit code.
4. `gpuwm sim PREPARED_ROOT ... --outdir run/`. Check the exit code,
   then read `run/report.json` and check `status`.
5. Render when and how you like -- `gpuwm render`, or your own plotting
   against the `wrfout` files.

Every stage exits nonzero on refusal and prints one sentence saying
why; add `--explain` to any command for the mechanism behind the
sentence.
