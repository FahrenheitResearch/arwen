# Prepared nested forecast recovery

The prepared nested runner automatically retries a typed full-state health
failure from the run's last committed restart. There are at most two retries.
This path addresses the Toronto and Boston nested d02 `w` health-gate failure
immediately after d01 synchronization (`post-d01-sync.d02`). It does not
replace the separate adaptive-clock fix for earlier d01 nonfinite failures.
The third health failure terminates the forecast with the original health
error and an exhausted recovery receipt. Memory, capacity, configuration,
untyped nonfinite and user interruption failures are never retried here.

For every adaptive domain, each retry caps future steps at half the smallest
of the previous cap, failed step and checkpoint step. The value is rounded
down to the 0.01 s adaptive clock lattice. Both CFL targets are halved. The
minimum clamp is lowered when needed so it cannot overrule the reduced cap.
The adaptive resume driver applies a retuned cap before the first resumed
step. No health bound, physics setting or restart identity check is relaxed.

Restart settings must write a checkpoint before failure. Production templates
should set `restart_interval_s = 3600.0`. A missing, torn, incompatible,
outside-run or unhealthy checkpoint ends recovery with its concrete reason.
The complete domain set is validated before any stored state is applied.
Arrays, physics state, clock state and coupler invalidation use the ordinary
`restore_tree_restart` contract.

Before restoring, asynchronous history writers are drained. Render consumers
must stop before history is rewound. Only this writer's registered frames
strictly after the restart are discarded, with their matching ready markers.
The frame bytes must still match their writer-completion proof. Earlier
history and unrelated files remain. Discarded paths, sizes and frame digests
are recorded in `evidence/stability-recovery.json`.

The same receipt records each health cause, domain, variable, restart model
time, retry number and clock policy before and after recovery. It is included
in `report.json`; failed-run receipts link to it. Hosted observers receive a
`stability_retry` warning record, and subprocess logs receive a JSON retry
line. Each retry has one retry number with a `validating_restore` start and
`resumed` outcome. A failed restore or output rewind records `refused` with
the original health cause and concrete refusal. These phases are written to
the receipt and subprocess output before replay can proceed. They remain
visible when no observer is attached. Repeated phases share the same retry
number and do not increase the two-retry bound.
An interruption during restore, writer draining or output rearm publishes an
`interrupted` terminal phase and preserves the original health cause before
propagating the interruption. It does not start another replay leg.
The site's existing event
forwarding preserves hosted warning metadata,
and its receipt upload includes `run/report.json`. No new normalized retry
count is added to the database run row. This lane does not qualify delivery
to the live database.

The prepared-cache comparison permits forecast-only changes to `epssm`,
`diff_6th_opt`, `diff_6th_factor` and `diff_6th_slopeopt`. Preparation constructs
the analysis and external boundary arrays without advancing an acoustic or
diffusion step. One verified prepared copy therefore serves these forecast
policies. Payload hashes and all other prepared identities remain checked.
Checkpoint restart identity continues to bind all four settings strictly.

Automatic recovery currently covers static prepared domain trees with
adaptive clocks. Moving, spawned, retired or delayed domains, active ensemble
capture, simulated radar history and standalone render consumers without a
rearm hook fail closed because rollback of their additional state has not
been qualified. Their ordinary forecast execution is unchanged. Fixed-clock
timesteps remain restart identity and are not changed automatically.

The CPU tests exercise real two-domain checkpoint writing and restoring,
bounded retry, selective error handling, policy reduction, torn checkpoints
and owned output rewind. `tools/proof_stability_retry.py` injects one out of
bound child `w` cell after a durable restart, asks the real full-state gate to
fail it, and requires default recovery and real solver completion. That
injection proves the recovery path. It does not prove that an archived
production incident or a complete revised city template is stable.

Run the GPU proof only on an authorized device through its ownership protocol:

```text
python tools/proof_stability_retry.py <ordinary prepared-tree arguments>
```

For an in-process prepared replay, wrap its ordinary runner call with
`tools.proof_stability_retry.health_fault_after_checkpoint(output_directory)`.
