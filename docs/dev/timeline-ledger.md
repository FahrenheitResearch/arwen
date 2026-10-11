# C1 v1

The schema and vocabulary live in `taxonomy.py`. `ledger.py` has no GPU,
numerical or field-processing dependencies. Boot and worker callers source
`tools/timeline/ledger.sh`; the shell emitter also works before installation.

Export `GPUWM_TIME_LEDGER=<workdir>/rental-ledger.jsonl` for the entire rental.
Both emitters append to that file. Without it the engine uses
`<run>/timeline.jsonl`. Context variables are `GPUWM_RENTAL_ID`,
`GPUWM_RUN_ID` and `GPUWM_PRODUCT`. Missing ids and unknown physical card
numbers are null. The emitter pid is the process doing the append, including
the Python helper launched by bash.

```python
from gpuwm.timeline import Ledger, span
ledger = Ledger("timeline.jsonl", run_id="run", product="regional")
ledger.emit("worker.claim", "mark", attrs={"source": "queue"})
with span("prep.inputs", ledger=ledger):
    prepare()
```

```bash
source tools/timeline/ledger.sh
ledger_emit boot.ready mark '{"source":"image"}'
```

One compact ASCII JSON record, terminated by LF, is written with O_APPEND
and a single write. POSIX writers also lock the file during the write.
Windows writers open the file with append-only access (FILE_APPEND_DATA
without FILE_WRITE_DATA), so the kernel places each record at the end; the
CRT's O_APPEND is a seek then a write, which lets concurrent writers
overwrite each other.
There is no fsync per event. Emission errors return false and produce one
stderr warning per process and path. Bash always returns zero and suppresses
subsequent warnings for the same path in that sourced shell. Events over
64 KiB are refused with the same best-effort behavior, preventing a bulk
receipt from becoming a ledger event. Disk exhaustion or a killed writer
can still truncate an append; consumers must report an incomplete final line.

The stage list is a contract, with generic descendants allowed under the
listed families. New information belongs in `attrs`. A change to the
envelope or vocabulary rules needs a schema version change.

Intervals nest and overlap. Never add durations to derive total wall time.
For a single forecast, FF is the measured `forecast.integrate` duration
divided by rental.requested-to-rental.terminated wall time, following C2.
Request latency is separately measured from rental.requested to rental.running.
A ledger without both rental boundary events has only a run-wall proxy.
For multiple runs, use the union of integration intervals, not their sum.
Use UTC across hosts. Monotonic clocks are comparable only on the same host
and boot. Rental ids must distinguish reused hosts after reboot.

Go stage labels map through `GO_STAGES`. Run-plan coarse stages carry
`attrs.role=envelope`; do not count them again beside their children.
The forecast process envelope is `post.forecast_process`. The runner's
startup envelope is `forecast.construct`, ending at the first measured
step start. It includes validation, restore, initialization and initial
output. It does not claim a separately measured model constructor call.

The prepared runner owns its existing progress.jsonl timer. The ledger
relays that metadata without changing its command or hosting mode.
`forecast.first_step` start is the step's publication time minus its measured
duration. `forecast.integrate` starts there and ends at the last measured
step's publication time. Boundaries are retrospective, millisecond UTC
resolution, and carry `attrs.basis=progress.jsonl.step_timer`.
The equivalent monotonic time uses the same host's current clock offset.
Append order therefore need not match boundary time. Sort before plotting.
Hosted callbacks batch metadata reads at 250 ms. Finalization and completion
drain the remaining records immediately, including a last step published
inside that polling interval. Sampling changes publication latency only;
boundaries retain their original measured timestamps.
With logging off, a thinned final step, or a lost progress log, coverage is
explicitly unknown. Those runs cannot establish a complete FF result.
Output intervals use existing frame submission-to-commit latency, including
async queue time. They are not blocking I/O time and can overlap integration.

Wave-1 parser slots are accepted but inert: `--stage-only`, `--warm-kernels`,
`--static-cache-root`, `--static-cache-key`, `--readiness-fold`, and
`--live-products-width`. The corresponding separate `SpeedHooks` bodies in
go_cli are their owners' implementation seams. Stage-only runs at go entry
before GPU admission, with plan=None; None means continue and an integer is
the completed stage-only exit code. Readiness receives the loaded config
payload. The other seams receive the claimed native go plan before stages
launch. They do not yet implement other registered route behavior.
