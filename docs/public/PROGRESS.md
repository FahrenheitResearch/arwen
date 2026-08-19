# Watching a run, and knowing when a frame is safe to read

If you are driving ArWen from your own script, you need two things the
tool has to give you rather than make you guess:

1. **Where the run is** -- which model time step, on which domain, at
   which valid time, and how long that step took.
2. **When an output frame is finished** -- a signal you can poll that
   does not race the writer.

Both are on by default. You do not have to pass a flag to get them.

## The text stream: WRF's own sentences

The simulation prints one line per model time step per domain, in the
grammar `wrf.exe` writes to `rsl.out.0000`:

```
gpuwm: STARTING SIMULATION at 2026-05-20_18:00:00 for 3600 model seconds
d01 2026-05-20_18:00:12 gpuwm: domain start
Timing for main: time 2026-05-20_18:00:12 on domain   1:     0.06382 elapsed seconds  step 1
d02 2026-05-20_18:00:04 gpuwm: domain start
Timing for main: time 2026-05-20_18:00:04 on domain   2:     0.02015 elapsed seconds  step 1
Timing for main: time 2026-05-20_18:00:08 on domain   2:     0.01984 elapsed seconds  step 2
Timing for main: time 2026-05-20_18:00:12 on domain   2:     0.02011 elapsed seconds  step 3
Timing for main: time 2026-05-20_18:00:24 on domain   1:     0.06104 elapsed seconds  step 2
Timing for Writing wrfout_d01_2026-05-20_18_00_00 for domain   1:     0.24310 elapsed seconds
Timing for Writing restart for domain   1:     1.51204 elapsed seconds  gpuwmrst_d01_2026-05-20_19_00_00__3f9c1a20.npz
d01 2026-05-20_19:00:00 gpuwm: domain end, 300 steps
d02 2026-05-20_19:00:00 gpuwm: domain end, 900 steps
gpuwm: SUCCESS COMPLETE SIMULATION, 1200 steps, 182.4 wall seconds
```

That is a 1:3 nest (d01 at `dt = 12 s`, d02 at `dt = 4 s`), and the
interleaving is real rather than tidied: the parent steps once, the
child takes its three, and the child's valid time trails the parent's
until it catches up at the sync instant. `wrf.exe` prints the same
shape for the same reason.

A parser written against WRF keeps working:

```
Timing for main: time (\S+) on domain\s+(\d+):\s+([0-9.]+) elapsed seconds
```

still matches, because the one field WRF does not print -- the step
index -- is appended **after** WRF's sentence rather than inside it.

Two honest differences from `wrf.exe`, both deliberate:

- **History filenames carry no colons.** ArWen writes
  `wrfout_d01_2026-05-20_18_00_00`, not `...18:00:00`, so the same run
  directory works on Windows. Valid times *inside* log lines keep WRF's
  colon spelling, which is what a WRF-shaped parser expects.
- **`Timing for Writing` is a latency, not a blocking time.** WRF writes
  history synchronously and reports how long the model was stalled.
  ArWen writes it on a per-domain side thread and the model is never
  stalled, so "how long the write took" is not a number you can act on.
  What is reported instead is the seconds from the model reaching that
  frame's valid time to the file being durable -- "how far behind the
  run are my plots". The JSONL calls it `durable_after_seconds`.

## The machine stream: `progress.jsonl`

Beside the outputs, at `<outdir>/progress.jsonl` unless you move it.
Append-only, one JSON object per line, schema `gpuwm.step-log/v2`:

```json
{"schema":"gpuwm.step-log/v2","sequence":4,"emitted_unix_ms":1779300012431,
 "event":"step","text":"Timing for main: time 2026-05-20_18:00:24 on domain   1:     0.06104 elapsed seconds  step 2",
 "domain":1,"step":2,"valid_time":"2026-05-20_18:00:24","model_seconds":24.0,
 "step_wall_seconds":0.06104,"fraction":0.006667}
```

Three properties worth relying on:

- **`sequence` is dense and monotonic.** A gap means you lost a line;
  it never means a line was skipped. `gpuwm.progress_log.read_step_log`
  refuses a stream with a gap rather than quietly repairing it.
- **`text` is the line that was printed.** The two streams cannot
  disagree about what happened, because there is one formatting call
  feeding both.
- **The tag set is closed.** Every record is one of `run_start`,
  `phase`, `domain_start`, `step`, `output_written`, `restart_written`,
  `domain_end`, `run_end`, so a `switch` on `event` can be exhaustive.

### The road to step 1: `phase` records (new in v2)

The per-step stream used to begin at step 1 and say nothing about how
long getting there took -- which on a first run is most of the time you
spend waiting. `phase` records close that. Each carries a `name` and a
`wall_seconds`, in the same one-emit-two-streams shape as everything
else:

```json
{"schema":"gpuwm.step-log/v2","sequence":2,"emitted_unix_ms":1779300009911,
 "event":"phase","text":"gpuwm: phase preflight_verify:     1.90000 elapsed seconds",
 "name":"preflight_verify","wall_seconds":1.9}
```

The names a forecast emits:

| `name` | what it covers |
|---|---|
| `preflight_verify` | verifying the preparation receipt, its digests and the prepared cache against what was asked for |
| `restore_prepared_cache` | getting the prepared case off disk and onto the machine |
| `initialize_physics` | building the physics driver for the restored state |
| `kernel_compile` | how much of model step 1 went to compiling this card's kernels (see below) |

**`kernel_compile` is measured, not assumed.** It is emitted only when
two things are both true: the CuPy kernel cache said before the run that
a compile was coming (it was empty, or it held no entry for this card's
compute capability -- a card swap is the case that matters), and step 1
then took more than ten times the median of the steps that followed it.
Its `wall_seconds` is that excess. On every run, compile or not,
`run_end` carries `first_step_excess_seconds`: the same number, or
`null` when step 1 was ordinary.

Measured on the reference box, 2026-08-16: a card swap left 7,164
kernels cached for the previous card, so nothing was reusable, and 51 s
of recompilation ran inside step 1's wall with nothing anywhere naming
it.

**Upgrading from `gpuwm.step-log/v1`.** Nothing v1 emitted changed shape
-- v2 adds one tag and one `run_end` field. The schema string moved
anyway, because a v1 consumer meeting `phase` should refuse loudly
rather than silently skip a record, and the schema is how it does that.
`read_step_log` replays both.

One boundary, stated rather than implied: `run_end` reports the outcome
of the **integration**, emitted once the last frame is durable. Work
that happens after it -- writing the run receipt, the certification
capsule -- is outside the stream, so the process's **exit code** remains
the authority for "did the command succeed". `run_end` answers "did the
model finish", which is the question a progress consumer is asking.
A process that dies before reaching its own end still terminates the
stream, with `status: "INCOMPLETE"`.

`output_written` and `restart_written` are events in their own right.
You never have to infer "a file appeared" from a frame count changing.

On a domain tree, `restart_written` names the checkpoint set's **root**
member. That is deliberate rather than partial: the set is published
one domain at a time with d01 written last as its commit marker, so the
root member appearing is exactly the moment the whole set is durable.

## The completion signal: `ready/`

A size check or an mtime check races the writer. Poll this instead:

```
<outdir>/ready/wrfout_d01_2026-05-20_18_00_00.json
```

```json
{"schema":"gpuwm.frame-ready/v1","domain":1,
 "valid_time":"2026-05-20_18:00:00",
 "path":"C:/runs/case/wrfout/wrfout_d01_2026-05-20_18_00_00",
 "size_bytes":186401212,"published_unix_ms":1779300012009,
 "guarantee":"the named file was fsynced, self-validated and renamed onto its final name before this marker was published; ..."}
```

**What the marker guarantees.** ArWen publishes a history file by
writing a temporary, closing the netCDF handle, `fsync`ing it,
re-opening it and validating its variable inventory, and only then
`os.replace`-ing it onto its final name. The marker is written after
that -- and the marker itself is published by `tmp -> fsync ->
os.replace`, so you never observe a half-written marker either.

> **A marker that exists names a frame that is complete and readable.**

The converse does **not** hold. A marker can be missing for a frame that
is perfectly fine -- markers were switched off, the marker write failed,
the run pre-dates this feature. Treat a missing marker as "not yet",
never as "corrupt".

The directory is created when the run starts, before the first frame
lands, so a watcher never has to handle "the directory is not there yet",
and the markers sit in their own directory so they cannot appear in a
`wrfout_d01_*` glob.

**If you would rather not poll at all**, the same fact is an event:
`output_written` on `progress.jsonl` carries the same `path`,
`valid_time`, `domain` and `size_bytes`, emitted from the writer thread
at the same instant. Tail one file and you never touch the frame
directory.

### What was already safe, and what was not

Worth being exact, because it changes what you have to defend against.
ArWen's history writer has always published atomically, and its
in-flight temporary is hidden behind a leading dot, so a `wrfout_d01_*`
glob has never been able to see a partial file. What you could not do
before was ask *when*: every consumer polled the frames themselves, and
at least one in this tree ended up re-opening multi-hundred-megabyte
netCDF files looking for a completion attribute the writer had already
guaranteed. The marker and the event are that answer, published once, in
one small place.

What is still not safe, with or without markers: a size check, an mtime
check, and any "it showed up in the listing" test on a filesystem whose
rename is not atomic.

## The flags

On both simulation doors -- `python -m
gpuwm.prepared_single_domain_forecast` and `python -m
gpuwm.prepared_domain_tree_forecast` -- spelled and defaulted
identically:

| flag | default | what it does |
|---|---|---|
| `--progress-format` | `text` | `text` prints the sentences on stdout and still writes the JSONL; `jsonl` writes only the JSONL, leaving stdout free of sentences; `off` disables per-step reporting entirely |
| `--progress-output` | `<outdir>/progress.jsonl` | where the machine stream goes. `-` sends it to stdout, which with `--progress-format jsonl` gives you a pure record pipe |
| `--progress-every` | `1` | report every Nth step. The first and last step of every domain are always reported, and this thins only `step` records -- output, restart and domain events are never thinned |
| `--frame-markers` / `--no-frame-markers` | on | publish `ready/<frame>.json` |

### What `gpuwm go` does

`gpuwm go` runs the forecast stage with `--progress-format jsonl`, so
you still get `progress.jsonl` and the `ready/` markers under the run
directory -- every line, nothing thinned -- but the sentences stay off
stdout. That is not a preference: `go` captures the stage's stdout in
memory and discards it on success, and when `gpuwm run-plan` hosts the
same runner in-process, its stdout is reserved for its own event
stream. Tail the file instead -- `Get-Content -Wait
<run>/progress.jsonl` on Windows, `tail -f` elsewhere.

### A pure record pipe

For a script that wants JSON on stdout and nothing else, add two flags
to the invocation you already use:

```
python -m gpuwm.prepared_single_domain_forecast ... --progress-format jsonl --progress-output -
```

`--progress-output -` is refused together with `--progress-format text`,
because that would put the sentences and the records on one channel and
leave you parsing both.

## What it costs

Measured on the shipped default (sentences to a terminal **and** the
JSONL flushed to disk on every record): **10.4 microseconds per step
event**; 9.6 with `--progress-format jsonl`; 1.2 with
`--progress-every 10`. A GPU model step on the shapes this package runs
is milliseconds at the very fastest, so the default costs a fraction of
a percent, and `--progress-every` is there for the rare shape where
even that matters.

The per-step wall time is a `perf_counter` pair the executor takes
around the step it was already taking. **No device synchronisation is
added to produce it** -- syncing per step to get a "true" GPU time would
serialise the pipeline the timing exists to observe, and WRF's own
number is host wall time across the step's launches too.

## How this relates to the other progress artifacts

ArWen publishes four things, and they are not redundant:

| artifact | written by | what it is for |
|---|---|---|
| `run-progress.json` | the supervisor | the durable **reattach anchor**; atomically republished, one small read, what crash recovery consults |
| `progress.json` | each forecast runner | a **coarse stage sample** for a parent process watching a subprocess |
| `events.jsonl` | `gpuwm run-plan` | the **plan-level event stream**: stages, warnings, resolved configuration |
| `progress.jsonl` + `ready/` | the simulation itself | the **per-step layer** the other three never had |

Nothing above was replaced. If you are already reading one of them,
keep reading it.
