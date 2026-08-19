# walk-capture — see the experience before shipping it

`tools/walk_capture.py` runs an ordered list of real commands and
renders what happened into one self-contained HTML page: per-step
timings, the exact argv, full output, exit codes, files created, an
auto-flagged friction table, and time to first plot.

The thing it replaces is a summary written from memory by the person who
already knows the answers.  Nothing here mocks, replays, stubs or
reorders anything — the report's every number is a measurement of the
commands as executed, and the untruncated streams stay on disk beside
the capture.

The harness knows nothing about this product.  A persona is a walk
script, not a code path: a fresh user on Windows, an experienced
researcher on Linux, and someone upgrading from an older release are
three files.

    python -m tools.walk_capture run WALK.toml --out CAPTURE_DIR
    python -m tools.walk_capture report CAPTURE_DIR --html PAGE.html
    python -m tools.walk_capture validate WALK.toml

Exit codes: `0` the walk was clean, `1` something was flagged or failed,
`2` the harness refused the walk script.

A worked example ships at `tools/walks/cli-first-contact.toml`.

## The walk script

JSON or TOML, chosen by suffix, with identical keys.  An unknown key is
refused by name rather than ignored, because a misspelled key means the
report describes a run nobody asked for.

```toml
name = "Command line, first contact"          # required; names the report
persona = "Someone who has just installed this."
description = "What this walk is asking."
root = "{home}/.gpuwm/walks/first-contact"    # working dir + tracking scope

[settings]
slow_seconds = 5.0            # flag a step slower than this
quiet_seconds = 4.0           # flag a silence longer than this
timeout_seconds = 600         # per-step default; a step may override
stop_on_unexpected_failure = false
output_bytes_in_report = 12000   # excerpt bound; the log files are whole
image_suffixes = [".png"]     # what counts as a plot
plot_skip_dirs = ["site-packages", "dist-packages"]   # ... and what never does
track_files = true

[env]                         # every step gets these
PYTHONPATH = "{repo_root}"
env_remove = ["SOME_VAR"]     # ... and does not get these

[venv]                        # optional fresh-environment bootstrap
path = ".venv"
system_site_packages = false
recreate = true

[[steps]]
id = "doctor"                                  # names the log directory
intent = "Is this machine ready to run a forecast?"
command = "{python} -m gpuwm.cli doctor"
allow_failure = true                           # record it, keep walking
tags = ["install"]
```

### Per-step keys

| key | meaning |
|---|---|
| `id` | unique; names `steps/NN-<id>/` in the capture |
| `intent` | plain language, in the reader's terms, not ours |
| `command` | one line, tokenized by the harness (see below) |
| `argv` | explicit argument list; mutually exclusive with `command` |
| `shell` | opt in to the platform shell parsing the line |
| `cwd` | relative to the walk root unless absolute |
| `env` | overrides for this step only |
| `expect_failure` | this step is *supposed* to exit non-zero |
| `expect_exit_code` | pin the exact code |
| `allow_failure` | an unexpected failure does not end the walk |
| `timeout_seconds`, `slow_seconds` | per-step overrides |
| `tags` | free-text grouping |

### Placeholders

`{root}`, `{walk_dir}`, `{repo_root}`, `{python}` and `{home}` expand in
`root`, `command`, `argv`, `cwd` and every environment value.  They are
substituted literally, not through `str.format`, so braces in a command
line survive.

## What gets recorded, and why it is recorded that way

**Both streams, timestamped.**  Output is read incrementally and each
arrival is stamped, which is the only way to answer "how long did the
terminal show nothing?".  A buffered `capture_output=True` throws that
away.  The full streams go to `steps/NN-id/stdout.txt` and `stderr.txt`;
the report carries a bounded excerpt — head, tail, and an honest count
of the omitted middle.

**Files created and modified under the walk root**, diffed around each
step by `(mtime, size)`.  This is where the plots come from: the first
file with an image suffix gives **time to first plot**, the metric this
project treats as the headline UX number.  A walk that produces no image
files says so; it does not print a zero.

Images an *installer* wrote are not plots.  A walk that installs into a
venv under its own root unpacks thousands of shipped PNGs — matplotlib's
toolbar icons, scipy's test-fixture images — and counting them turned a
walk that rendered nothing into a reported "time to first plot: 13.4 s".
So the scan skips any image under a `site-packages`/`dist-packages`
directory (`plot_skip_dirs`) or under any directory holding a
`pyvenv.cfg`, which is what defines a virtual environment.  The files are
still tracked and still counted in *files created*; only the plot metric
ignores them, and the report says how many it ignored.

**Stdin is closed.**  A command that stops to ask a question gets EOF
and fails fast, and the report shows it.  A walk cannot answer prompts —
a door that needs one is a finding, not a configuration problem.

## expect_failure, in both directions

`expect_failure = true` says *this step is supposed to fail* — the typo,
the forgotten flag, the door that refuses on purpose.  Such a step is
recorded as `expected-failure`, raises no friction flag, and the walk
carries on.

The other direction is the one that earns its keep: a step declared
`expect_failure` that **succeeds** is flagged as friction, because the
walk's description of the product has gone stale.  That is how a walk
script notices that something got fixed.

`allow_failure = true` is different, and means *record this failure and
keep going*.  Use it where a non-zero exit is real friction you want in
the report rather than a wall you want the walk to stop at.

## Friction, and where each flag comes from

Every flag is derived from a measurement, and the thresholds it used are
printed in the report so a reader can disagree with them:

| flag | fires when |
|---|---|
| `nonzero` | the exit code is not what the step declared |
| `unexpected-success` | `expect_failure` and it exited 0 |
| `slow` | wall time at or over `slow_seconds` |
| `silent` | the longest gap between output arrivals reaches `quiet_seconds` (the gap before the first byte and after the last one both count) |
| `traceback` | stderr carries a Python traceback or a bare `SomeError:` line |
| `timeout` | the step outlived its timeout and was killed |
| `not-found` | the OS could not start the command at all |
| `skipped` | the walk stopped before reaching this step |

## Commands, quoting, and Windows

By default a `command` string never touches a shell.  It is tokenized by
one small rule — whitespace separates, double quotes group, everything
else including backslash is literal — and the tokens are handed to
`subprocess.Popen` as a list.

That rule exists because both obvious alternatives are wrong here.
`shlex.split` in POSIX mode eats the backslashes out of
`C:\walks\data`, turning a Windows path into a different one.
Handing the line to `cmd.exe` instead gives `&`, `|`, `^` and `>` inside
a directory name their shell meanings, so a directory legitimately named
`a & b` becomes two commands.  Neither happens: a quoted Windows path
arrives at the child byte for byte.

Three consequences worth knowing:

* An unbalanced double quote is refused, because the rest of the line
  would silently collapse into a single argument.
* A `.bat` or `.cmd` target carrying `&`, `|`, `<`, `>`, `^` or `"` in a
  later argument is refused.  Windows can only run batch files by handing
  a command *line* to `cmd.exe`, which re-parses it, so for that one
  target class argv is not a boundary and the harness has no safe
  quoting to offer.  Name the real executable, or set `shell = true` and
  own it deliberately.
* An argument needing a literal double quote uses the `argv` form, which
  skips tokenizing entirely.

`shell = true` remains available and is labelled as such in the report.

## The fresh-environment bootstrap

A `[venv]` table prepends a real, timed, captured step to the walk:
`python -m venv` with the flags the table asks for.  It is not
configuration that happens off-screen — venv creation is often the
slowest thing in a fresh-install walk, and it belongs on the timeline.

Afterwards every remaining step runs with the venv's `Scripts`/`bin`
first on `PATH` and `VIRTUAL_ENV` set.  On Windows that alone would not
be enough: `CreateProcess` resolves a bare executable name against the
*parent* process's `PATH`, not the one passed to the child, so a step
saying `python` would have silently run the outer interpreter and the
whole capture would have described the wrong install.  The harness
therefore resolves a bare `argv[0]` against the step's own `PATH` before
launching, and records the resolved path.

If the bootstrap fails the walk stops, because every later step would
have measured an install that is not the one under test.

## The capture directory

```
capture/
  capture.json          the whole record; the reports render from this alone
  report.html           self-contained, no external requests
  report.md             terse summary
  steps/01-version/stdout.txt
  steps/01-version/stderr.txt
  ...
```

`capture.json` is versioned; a capture from a different schema is
refused rather than rendered with mislabelled fields.  Because the
reports render from it alone, `report` can re-render an old capture with
new thresholds without re-running anything.

The capture directory must sit **outside** the walk root.  Inside it,
every step would see the previous step's log files as newly created user
output and the file-activity column would be the harness measuring
itself; that is refused.

## Running it against this tree

```
python -m tools.walk_capture run tools/walks/cli-first-contact.toml \
    --root  <scratch dir> \
    --out   <capture dir> \
    --label "2.5.0 pre-cut, Windows 11"
```

`--root`, `--slow-seconds`, `--quiet-seconds` and `--keep-going`
override the script, so the same persona can be re-walked on another box
without editing the file.

`tests/test_walk_capture.py` holds every promise on this page, against
real child processes rather than a mocked `subprocess`.
