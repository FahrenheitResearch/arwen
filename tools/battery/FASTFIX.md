# Faster checks for small patches

`run_fastfix.py` coordinates the changed-area CPU tests and an optional check of
an installed wheel. It runs the selected tests in one pytest process, orders
files by recorded cost, and saves elapsed time, failures, skips, logs, JUnit,
source hashes and selection reasons in a new evidence directory. Installed
command checks run alongside the CPU batch.

This replaces the old manual sequence of selecting tests, launching individual
pytest processes and assembling their results. It does not publish or turn a
focused test result into a full release verdict.

## Run the patch checks

Start from the last accepted release and include the fix's regression test.
Review the plan first:

```text
python tools/battery/run_fastfix.py --base <accepted-revision> --output <new-plan-directory>
```

Run the batch and the installed command checks together:

```text
python tools/battery/run_fastfix.py --base <accepted-revision> --output <new-run-directory> --python <test-python> --installed-python <wheel-python> --engine-wheel <platform-wheel> --companion-wheel <data-wheel> --artifact-proof <verified-artifact-receipt> --execute
```

The output directory must be outside the checkout. The test interpreter needs
the battery's normal CPU dependencies. Installed checks require a clean,
committed source tree so the source tests and installed artifacts bind the same
revision. The installed interpreter must belong
to a separate environment containing the built engine and matching companion
wheel. The artifact verifier's receipt must bind both wheel hashes (include
both in its `--dist-dir`). Every installed wheel member is compared with the
expected bytes. The installed TUI must resolve inside that environment and
carry the tested source revision. The command
sweep uses isolated Python and rejects editable/source-tree installations.

For source checks before committing, add `--include-working-tree` and omit the
installed-check arguments. Such a run is
bound to the working bytes, including untracked shipped-source inputs, and is
labeled with its dirty state. A source
change during execution fails the run. `--source` permits an external copy of
the runner to measure a frozen checkout without modifying it.

The installed sweep enumerates the installed parser, checks help for every
registered command and nested command, executes the real module entry point
for help and version, and launches the native TUI to produce HTML and terminal
cell captures. It starts no forecast. A successful sweep proves those doors;
the workflow changed by the fix still needs its own real execution.

## Decide which expensive results need refreshing

The selector includes `always_files.txt` on every run. Direct Python imports
select affected test files; changed test files select themselves. Named Rust
and CUDA source readers also run when those source types change. This is a
focused selection, and does not cover every indirect consumer or data-file
reader. `broader_checks_required` names the extra work the runner can identify.

A small documentation correction should not require another long numerical
campaign. Reuse a previous campaign only when its model source, input fixtures,
numerical settings, dependencies, compiler and runtime assumptions still match.
The release report must name the original revision and receipt, explain the
unchanged inputs, and label the result **reused**. Never rewrite an old PASS as
though it ran on the patch. `run_fastfix.py` deliberately records no reused
results on its own.

A physics, kernel, forecast configuration, native build or dependency change
refreshes the affected numerical and platform checks. An unknown impact needs
review of the indirect consumers. A short test selection alone does not
establish that the rest of a release is unchanged.

Native builds retain dependency caches tied to the compiler and build image.
Owned binaries are rebuilt with the candidate's source stamp. Final package
verification and installed checks still run against the candidate artifacts.
This avoids recompiling unchanged dependencies without relabeling old binaries.

For the 2.7 release baseline, run the complete current pretag requirements in
`RELEASE_CHECKLIST.md`, including public Stage 1 and the separate no-CuPy job.
The baseline is what later patches can compare against. The previous version
of this document's blanket promise to publish before the full battery, and
its unimplemented post-publish alarm, are not the current release procedure.

## Read the result

- `PLAN_ONLY`: no tests ran.
- `FOCUSED_CHECKS_PASS`: the requested legs exited successfully. The receipt
  separately lists passed and skipped tests, installed checks that were not
  requested, and broader checks still required. This is not release approval.
- `FAIL`: a test/process failed, collection was empty, or the source changed.
  The logs and intermediate evidence remain for diagnosis.
- `INCOMPLETE`: tests skipped or a selected suite executed no tests. The
  command exits unsuccessfully and names the missing coverage in its receipt.

No elapsed-time target is a measured result. Use the receipt's actual wall time
for the current machine, environment and selection. Native builds, packaging,
installation and numerical work are separate costs unless the record includes
them. No guarantee of a complete release in a fixed number of minutes is made.

`fastfix.py --record-from <pytest.log>` can update per-file duration estimates
from an existing batched `--durations=0` report. It avoids the interpreter startup
cost of the older `--record` mode, which launched one process per test file.
Keep the measured revision and environment with the report.
