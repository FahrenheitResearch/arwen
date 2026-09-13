# Candidate contract check

`tools/release/precut_gate.py` runs the contract files named by the candidate's
publication workflow, plus the native distribution check. It is the check before
export. It does not establish that built distributions, desktop packages or an
installed forecast work. Receipts state that scope explicitly.

The runner requires a clean Git worktree, checks the commit and status operations,
and archives that exact commit. Archive creation finishes successfully before
transfer starts. Every attempt extracts into a new directory below the supplied
scratch directory. It never deletes or replaces another attempt's directory.

The test process must finish with a normal success or assertion-failure status.
Its fresh JUnit report must agree with that status, contain consistent case
counts, cover every selected file, contain a passing test, and contain no
collection, setup or teardown errors. Missing reports, interrupted runs and
transport failures cannot pass, including an interruption after some tests pass.
Human-readable terminal summaries are diagnostic output only.

One shadow-specific exception remains: the install-provenance test's assertion
that a `missing` identity should be `verified`. The exact test, assertion and
call-failure shape must agree. Another failure in that test is a real failure.
The exception still has to pass in the fresh installed release check.

The runner accepts `--tree`, `--node`, `--python` and `--remote-dir`, plus
`--evidence-dir` and optional `--cuda-path`. The surrounding release entry point
can provide the existing account, interpreter and storage defaults through
`main(defaults=...)`. Direct invocations must provide the execution account,
absolute scratch directory and evidence directory. Receipts retain the
`precut-gate-<commit>.json` name, record the process exit, structured report path
and digest, and use schema `arwen.precut-gate.v2`.

A PASS applies only to the named candidate commit and selected contract files.
An integrated commit requires its own check. A failure or incomplete result is
also recorded when the candidate identity and evidence location are available.
