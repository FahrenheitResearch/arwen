# One run, one folder

Every run of `gpuwm go`, `gpuwm sim` and `gpuwm render` writes into its
own timestamped folder under the output directory you name. Run the same
configuration twice and you get two folders. Nothing is overwritten,
nothing is interleaved, and you do not have to invent a new directory
name each time.

This is the default. There is no flag to turn it on.

```
out/myarea/                                   <-- --outdir / --out (the case)
  data/                                       <-- the download, CACHED across runs
  latest-run.txt                              <-- one line: the newest run folder's name
  run-20260817-041233Z_i202607291800Z/        <-- the first run
    authority/
    prepared/
    run/            wrfout/, report.json, progress.jsonl, receipts
    png/            d02-3km/composite_reflectivity/2026-07-29/...
    events.jsonl
  run-20260817-134902Z_i202607291800Z/        <-- the second run, same config
    ...
```

The render layout is unchanged underneath. `<domain>/<product>/<valid-day>`
is exactly what it was; the run folder sits above it. See
[render-output-layout.md](render-output-layout.md) for that half.

## The stamp, exactly

```
run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z[-<NN>]
    \_______ launch _______/  \____ init ____/  \ordinal/
```

| part | what it is |
| --- | --- |
| `launch` | the wall-clock instant the command was invoked, UTC, to the second |
| `init` | the model's initialisation time (the cycle a forecast starts from), UTC, to the minute |
| `-NN` | `-02`, `-03`, ... appended only when the first two already name an existing folder |

Launch leads so a plain directory listing sorts into run order. Seconds,
not minutes, because two back-to-back runs of a short configuration land
inside the same minute.

`_i...` is present when the door knows the initialisation time and
**absent when it does not** — a wrong init time in a folder name is worse
than a missing one, because a reader believes it. Where each door reads
it from:

| door | init comes from |
| --- | --- |
| `gpuwm go` | the config's `[fetch] cycle` |
| `gpuwm sim` | the prepared bundle's own `proof.json` / `receipt.json` |
| `gpuwm render` | `SIMULATION_START_DATE` in the first wrfout that carries it |
| `gpuwm render --pair` | nothing — a comparison of two runs belongs to neither one's cycle, so the stamp is launch-only |

Everything but the ordinal is a pure function of `(launch, init)`, so a
caller that knows both can compute the folder name before the run starts.
`gpuwm.run_stamp.format_stamp` is the in-tree implementation if you would
rather import than transcribe.

## Finding the run you just started

Every door prints the folder before it writes anything:

`gpuwm go` claims the folder once, at the top, and its own render stage
draws into it — no second stamp inside:

```
go: run folder run-20260817-041233Z_i202607291800Z under out/myarea -- authority/, prepared/, run/ and png/ all land inside it, and the download is cached at out/myarea/data
render: layout nested -- out/myarea/run-20260817-041233Z_i202607291800Z/png/<domain>/<product>/<valid-day>/<file>.png
```

The download directory is the one exception to "everything lands inside
the run folder", and deliberately so: inputs are cached across runs of a
config while artifacts are separated by run. It defaults to `data/` beside
the run folders, and `--data-dir` moves it anywhere — the `cached at`
clause names whichever one is in force, so it is always the directory the
fetch stage actually writes to.

`gpuwm render` typed on its own is its own run, so it claims one:

```
render: run folder run-20260817-041233Z_i202605171800Z under out/myarea/png
render: layout nested -- out/myarea/png/run-.../<domain>/<product>/<valid-day>/<file>.png
```

From a script, read `latest-run.txt` — one line, the newest run folder's
**name**.

**It sits beside the run folders, in the directory you named on the
command line.** That is one rule, not a rule per door, and it is the
directory the `run-...` folder was created in:

| you typed | run folders land in | so the pointer is |
| --- | --- | --- |
| `gpuwm go ... --outdir out/myarea` | `out/myarea/` | `out/myarea/latest-run.txt` |
| `gpuwm sim ... --outdir out/myarea` | `out/myarea/` | `out/myarea/latest-run.txt` |
| `gpuwm render ... --out out/myarea/png` | `out/myarea/png/` | `out/myarea/png/latest-run.txt` |

`gpuwm render` typed on its own is the case that surprises people: its
`--out` is usually a `png/` directory *inside* the case, so the pointer
is inside `png/` too — not up at the case root. A script that reads
`out/myarea/latest-run.txt` after a render-only workflow finds nothing.
(When `go` runs its own render stage there is no separate pointer,
because that stage draws into the run folder `go` already claimed.)

For `gpuwm render` the pointer and the run folder are published only
after at least one PNG lands: a render that fails before drawing
anything removes its empty folder and leaves the pointer on the last run
that produced output, so a script following it never lands in a failed
render's empty tree.

```sh
run=$(cat out/myarea/latest-run.txt)              # after gpuwm go
ls "out/myarea/$run/run/wrfout"

run=$(cat out/myarea/png/latest-run.txt)          # after gpuwm render --out out/myarea/png
ls "out/myarea/png/$run"
```

```python
from gpuwm import run_stamp
run_dir = run_stamp.latest("out/myarea")           # pointer first, listing as fallback
run_dir = run_stamp.latest("out/myarea/png")       # ... the render-only workflow
```

`run_stamp.latest` takes the same directory you typed, and falls back to
listing the `run-*` names when no pointer was ever written. Sorting
those names alphabetically is the same order, because the launch field
leads.

## The download is not stamped

`out/myarea/data` holds **inputs** and stays put across runs. Stamping it
would re-download an identical GRIB set on every re-run of one config,
which is the opposite of what the timestamped folders are for. Artifacts
are separated by run; inputs are cached.

`--data-dir` still points at a download you already have, and is
unaffected by any of this.

## Pointing a door at an existing run folder

If the directory you pass to `--out`/`--outdir` is itself named like a run
folder — or sits anywhere **inside** one, such as `run-.../png` — it is
used **as given**. No second stamp level is inserted, because the run's
identity is already established at or above that path. What happens next
differs by door, and each says so:

| door | behaviour | why |
| --- | --- | --- |
| `gpuwm go` | **refuses** if the folder already holds an `authority/` tree | every stage is create-only; merging two runs into one tree publishes receipts describing neither |
| `gpuwm sim` | **refuses** if the folder already holds `report.json`, `wrfout/`, `progress.jsonl` or `receipts/` | two runs' wrfout frames merge silently under identical names, `report.json` ends up describing only the last one, and a render of that tree draws one product series out of two forecasts |
| `gpuwm render` | **resumes** — draws into the tree that is there | rendering is idempotent per filename, and adding products to an existing run's folder later (`--heavy`, another `--products` list) is a real workflow |

Both refusals name the folder, name what is already in it, and give the
remedy in the flag you typed.

## `--run-stamp off`, and why you probably do not want it

```sh
gpuwm go myarea.toml --outdir out/myarea --run-stamp off
gpuwm render out/myarea/run/wrfout/wrfout_d0* --out out/myarea/png --run-stamp off
```

writes straight into the output directory, which is what releases up to
v2.4.1 did. It exists for one reason: a consumer written against the old
fixed paths needs somewhere to stand while it is updated. It is a
compatibility escape hatch, not a supported alternative — with it on, a
second run of one configuration is refused against the first one's tree,
which is the report this feature answers.

`--run-stamp off --layout flat` together reproduce the v2.4.1 render
directory byte for byte: every picture directly under `--out`, no
intervening directory of any kind.

## What already understands it

* `gpuwm go`'s early render and its finalize render write into the same
  run folder, so `first-products.json` and the final tree agree. Both are
  stages of one run, not runs of their own: the run folder is claimed
  once by the chain, and neither stage stamps again inside it. A stage
  that did would put the frame published on the first committed history
  output and the frames drawn at the end into two `run-...` folders
  minutes apart, and the finalize stage could no longer prove the early
  frame had already been drawn.
* `gpuwm sim --print-command` prints the run folder in the `--outdir` of
  the line it emits, so a script that copies that line lands where
  `gpuwm sim` would have. The folder is **named, not created**: asking the
  question spends nothing, and the runner makes the directory when you run
  the line.
* `gpuwm render --pair A_DIR B_DIR` reads both input directories
  recursively, so it pairs a run folder against a flat directory during a
  migration.
