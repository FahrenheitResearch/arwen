# Real terminal journeys

These recipes send keyboard input, bracketed paste, and mouse clicks to the
actual `arwen-tui` process. The Windows runner uses ConPTY through pywinpty;
the Linux code path uses a native pseudoterminal. The receipt names the actual
platform and backend. Windows receipts do not qualify Linux: the Linux backend
has not yet been exercised by this acceptance work. The runner observes emitted terminal cells,
created files, and the real CLI worker's completion receipts. It does not call
internal app methods or use the binary's snapshot mode.

Install `../requirements-tui-journeys.txt` into a test environment, then run:

```text
python tools/battery/tui_journeys.py --tui /path/to/arwen-tui --python /path/to/installed/python --recipe tools/battery/tui-journeys/repeat-forecast-planning.json --out /new/evidence/folder --scope installed-artifact --allow-gpu
```

Use a new evidence directory for every attempt. `--scope development` permits an
editable engine during implementation; `installed-artifact` refuses an editable
checkout. If the package uses a native distribution manifest, pass its path with
`--distribution-manifest`. The receipt binds the binary, engine provenance,
runner, recipe and all retained captures and command logs. Failures remain
failures, with their last visible screen and native command result preserved.

The ordinary and repeated planning recipes measure the local GPU to size their
configuration, so they require `--allow-gpu` on a GPU workstation. They create
configurations and run the real `go --dry-run` command; they do **not** qualify
data acquisition, forecast integration, restart, or weather rendering. Installed
release acceptance must run those additional workflows against actual inputs.

`repeat-forecast-planning-declared-capacity.json` performs the same A/B/A
journey, clicks the optional GPU-memory field in each settings summary, and
types `8`. Run it without `--allow-gpu` for CPU CI. This declares a planning
capacity; it does not measure or qualify a GPU. Keep the original recipe as
the separate test of automatic local-GPU measurement.

`repeat-default-filenames-declared-capacity.json` also accepts each proposed
filename unchanged. It checks that two creations produce `general.toml` and
`general-2.toml`, then reopens the first configuration and verifies its original
download cache. This covers repeated creation without asking the user to manage
output names.

`ordinary-forecast.json` creates an editable configuration from a mouse-selected
mode and typed location, source, UTC cycle, duration and filename containing a
space, then reviews its native launch plan. `repeat-forecast-planning.json` does
that twice in the same workspace with changed location and cycle, reopens the
first configuration, and checks the actual three CLI logs: the changed request
selects a different download cache and the reopened request selects its original
cache. The fixed dates make this planning-only check reproducible without asking
an archive for files.

Each action saves an HTML and text capture; `terminal.ansi` retains the raw output.
Click selectors must match visible labels uniquely, with optional row prefixes
or zero-based occurrences. Pasted text waits for visible echo before proceeding.
Long-running commands must be followed by `expect_jobs`, which checks their real
command identity, exact job count and exit codes rather than equating the
presence of a screen with success. A recipe cannot pass with unfinished native
jobs. Unexpected terminal exit fails even when the last expected text matched;
an intentional close must use `expect_exit` with the expected exit code.
Cleanup records and terminates only descendants observed under the launched
TUI, including workers in separate process groups, and a cleanup failure fails
the journey. Keyboard interruption retains a failed receipt. Do not
run a forecast recipe without deliberate input and resource provisioning.
