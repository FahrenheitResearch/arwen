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

## The card stage

`tools/release/precut_gpu_gate.py` is the second check before export, and every
cut runs it after the contract check. The contract check runs with
`-m "not gpu"` under `GPUWM_NO_LOCAL_GPU=1`, so no test that needs a card runs
in it; the phase-2 step pin (`tests/test_coriolis_map.py`) shipped red in 2.7.4
and 2.7.5 and the mp=8 freeze (`tests/test_mp8_frozen.py`) shipped red in 2.7.4
and 2.7.5 for exactly that reason, and nothing in the procedure could see either.

The card stage takes the same `--tree`, `--node`, `--python`, `--remote-dir`,
`--evidence-dir` and optional `--cuda-path`, reuses the contract check's archive,
transfer, identity, receipt and strict-completion functions (it carries no second
copy), and runs the tests that `tests/gpu_pin_set.txt` declares on the node's
card without `GPUWM_NO_LOCAL_GPU` and with `-n 0`, because pins are captured
serially and two processes on one card are a discarded reading. That file is the
one place the set lives: `tests/test_gpu_pin_set.py` holds it to the tree, and
every test that asserts a pinned digest, fingerprint or bitwise/ULP baseline of
a GPU result belongs in it, with the exclusions stated in the file. A candidate
that predates the file is run with the gate's own copy, and the receipt's
`pin_set_source` says so.

The card is the stage's instrument, and the stage never reads it shared. The
probe that names the card also compiles one kernel, so an interpreter whose cupy
has no toolkit refuses the stage naming the toolkit (`--cuda-path` must point at
a directory with `include/` and `lib/libnvrtc`) instead of recording every pin
red with the same header error. Before the run the stage waits, up to nine
minutes, for the card to hold no other compute process, and refuses naming the
process if it still does; while the pins run it samples the card every two
seconds (its own process and that process's children excepted) and once more
after they end. The sample the probe saw, the sample the run started on, the
samples during and the sample after are all in the receipt, so a run that
waited names what it waited for, and a run during which another process held
the card is recorded as ERROR naming the process, never as a PASS and never as
a FAIL: that reading is discarded and the stage is run again on an idle card.

The receipt, `precut-gpu-gate-<commit>.json` with schema
`arwen.precut-gpu-gate.v1`, records the card name, driver, compute capability,
CUDA runtime, cupy version and the NVRTC build that compiled the kernels (with
its build id and library sha256, through the tree's own
`gpuwm.certify.compile_platform`) beside the commit, the occupancy samples and
the seconds waited, and every test in the set that skipped, by name with its
reason. A skip
whose reason says the run saw no card refuses the stage. A repeated run for one
commit never replaces an earlier receipt: it writes
`precut-gpu-gate-<commit>.<n>.json` and names the receipt it follows with its
sha256, so a discarded reading stays on record beside the run that replaced it.

A PASS applies to the named commit on the named card under the named NVRTC
build only. Every npz pin is kept per card in the registry `tests/_card_pins.py`
(the phase-2 step capture through `tests/_phase2_pin.py`), the two
surface-trajectory digests record the card they describe, the SASE device golden
pair is per card in `tests/sase_goldens.py`, and the Shin-Hong ULP table and the
bl=1 run-state hash are per NVRTC build and card. A card with no committed
capture or row skips that test with the reason, which the receipt then shows,
and an NVRTC build the Shin-Hong rows have never been measured under fails
naming itself, as those tests always did. The two axes differ because a new
card on the release node is a machine change that the receipt puts in front
of whoever cuts the release, while a new compiler build moves pinned bytes
silently and must be recorded before anything ships. A new card gets its captures from
`tools/recapture_phase2_pin.py --write` and `tools/recapture_card_pins.py --pin
NAME --write` on that card (both record the NVRTC build beside the card), each
with the reading the tool prints recorded in the test's ledger, the surface
digests from the two identity tools, and the per-card rows from two processes
on that card.

    python tools/release/precut_gpu_gate.py --tree <candidate worktree> \
        --node <account@node> --python <interpreter on the node> \
        --cuda-path <toolkit on the node with include/ and lib/libnvrtc> \
        --remote-dir <absolute scratch on the node> --evidence-dir <receipts>

The publication workflow in `.github/workflows/publish.yml` does not run this
stage, because that workflow has no card; it runs from a machine with a node
that has one. The receipts under `tests/data/receipts/pin-gates/` are the stage
run against integrate/2.7.6 at fc639c51f (FAIL: the phase-2 pin, the mp=8
freeze and every other pin in the state they shipped in) and against the lane
branch's tip (PASS, the declared set of 17 entries), both on node-1's RTX 4090
under NVRTC 13.3.33.
