# Operate a country suite

The suite controller runs existing forecast, products and observation-scorecard
doors. Its journal records the suite file hash on every row and cycle. Run the
controller on a box with its required packages and prepared inputs. Keep the
state directory on durable local storage. Use the same state directory for every
operation below. An unadmitted plan launches nothing.

| Step | Command | Pass condition |
|---|---|---|
| 0 | `pip install gpuwm` (and `gpuwm-hex` for hex rows) at paired versions, then `gpuwm check --json` on each box | Paired versions, cards, RAM and disk written to box receipts |
| 1 | `gpuwm suite init --region REGION --boxes box-r.json,box-g.json -o country.suite.toml` | Every referenced registry row exists |
| 2 | `gpuwm suite validate country.suite.toml` | Schema, regions, shapes and rates resolve; unknown keys exit 2 |
| 3 | `gpuwm suite plan country.suite.toml --json` | Complete suite admitted, or exit 2 names the budget and overrun |
| 4 | Prepare every row's inputs with the ordinary doors (`gpuwm go CONFIG --prepare-only`, `gpuwm-hex country --stage prepare`) and write their paths into `deployment.json` | The bindings document names each prepared input, DA and CPU-stage door |
| 5 | `gpuwm suite rehearse country.suite.toml --bindings deployment.json --from PAST_DATE --days 1` | Every cycle completed and observation scores cover every verification cycle |
| 6 | `gpuwm suite run country.suite.toml --bindings deployment.json --detach` | Journal advancing and heartbeat younger than 120 seconds |
| 7 | `gpuwm suite status --json`, `gpuwm suite events --follow` | Each row reports its cycle, state, lag and score |
| 8 | `gpuwm suite apply country.suite.toml` | Difference printed; new hash installed at a quiescent cycle boundary |

`REGION` is a row of `gpuwm/data/nwpsuite/regions.v1.json` (this release
ships `US` and `PH`; adding a region is a table row). With no `--boxes`, `init`
writes the template's boxes and the plan prices them from the shipped rate
table; plan against your own `gpuwm check --json` receipts before launch.

Automatic preparation and measured-rate calibration are not commands in this
release: preparation runs through the ordinary forecast and hex doors (step 4),
and rates come from the shipped rate table. An admitted reservation plan still
has `execution_ready=false` until its prepared, DA and CPU-stage bindings
exist; run and rehearse refuse it with `suite_execution_unbound` unless a
`--bindings deployment.json` document materializes those reservations.
Operation across two hosts is refused with `suite_transport_unbound`. See
[the binding contract](../suite-runtime-bindings.md) for prepared authority,
member placement, CPU stages and queue configuration. An explicit
`gpuwm.suite.runtime-fixture.v1` JSON plan is for door qualification only.

Configure each box's `launcher_prefix` as argv tokens for its admission queue.
`{cards}` is the row's requested card count. GPU launches without this field are
refused because they would bypass admission. Explicit prepared bindings must be
present before launch. `health_argv` is a bounded CPU command returning JSON with
`alive`, `cards`, `host_ram_available_bytes` and `disk_free_bytes`. Reported remote
metrics without a probe are null.

| Event | Response |
|---|---|
| Process crash | Three fresh-process retries, checkpoint command once, each degrade entry once, then `recovery_exhausted`. A failed child does not halt its parent. |
| Source late, exit 75 | Readiness waits to deadline minus expected runtime. Try the next configured source, then skip as `source_behind`. |
| DA arm failure | Record skipped with public-analysis initialization. Forecast dependencies may proceed. |
| Card lost | Replan by priority. Insufficient priority-1 capacity gives `priority_one_capacity`. |
| Box lost | Request the resolved `single_box_profile` at the boundary. Missing resolved profile gives `single_box_profile_missing`. |
| Disk below floor | Prune registered artifacts, then halt the lowest-priority nonmandatory row. |
| Refused door, exit 2 | Named `door_refused` halt. Repeating an unchanged refusal would waste cards. |
| Ambiguous lost worker identity | Halt. Inspect the launch token before resuming; replay could double-launch a door. |

`gpuwm suite halt ROW` prevents new launches and signals the identity-checked
worker group for any active row attempt. `gpuwm suite resume ROW` releases
the halt and starts an explicitly halted attempt with a new token. A scheduler
restart uses `gpuwm suite run --resume`. Detached workers retain the old launch
token and publish its result, so scheduler death does not relaunch a live door.
Active `intent.json`, `worker.json` and `result.json` are recovery anchors.
Terminal attempts can be pruned after their result has reached the journal.
A second controller for the same state directory returns exit 2 immediately
with `suite_scheduler_active`, preventing competing cycle launch decisions.

Status exit codes are 0 green, 1 degraded, 2 halted and 75 waiting on a source.
Scores are stored in the cycle receipt in the ledger, so pruning score files
does not remove the current health comparison. A rehearsal can accept scored
public-analysis forecasts after DA failure, with `degraded=true` and explicit
fallback records. Missing required forecasts or scores never pass acceptance.
Door exit codes are 0 completed, 1 failed, 2 refused, 75 source behind and
130 interrupted. Interrupting the CLI returns 130. A missing declared success
receipt fails the attempt. Door kinds and receipt checks are data in
`gpuwm/data/nwpsuite/door-kinds.v1.json`.

`on_state_change` is one argv command receiving the status document on stdin.
It runs on a change of aggregate or row state. Alert command failures are recorded
in the journal. The status file is `status.json`; history is `suite-ledger.jsonl`.

Retention defaults are history 36h, products 14d, analyses 7d, receipts 30d,
checkpoints latest per row, receipt cap 500MB and disk floor 500GB. Register raw
output categories with each runtime row's `artifact_globs`. Files under a launch
are inventoried after the worker finishes. Retention never follows links or
deletes shared prepared inputs. Receipt accounting combines registered receipts,
active JSON/JSONL/log/TOML/CSV files and control documents under the suite root.
An active worker checks every quarter second, prunes registered completed files,
and stops its door with `receipt_cap_exceeded` if its reserved share or the
combined cap remains exceeded. Active anchors and unregistered control files
are counted but never pruned. This is a monitored cap; a door can overshoot
between checks. Size active histories narrowly and use
door-level receipt sampling for long integrations. The journal compacts into an
atomic state snapshot, preserving unfinished jobs and terminal cycle watermarks.
Old event cursors receive `events_expired`. Snapshot and control documents
occupy receipt space. If they alone exceed the cap, increase the cap or reduce
the suite's retained configuration/state before resuming.

## Shown working

![First suite cycle, 3 km radar](img/whats-new-2.9.0/03-suite-cycle-12z-reflectivity.png)

Two consecutive regional 3 km cycles (2026-10-03 12Z and 18Z) launched by
`gpuwm suite run` through a GPU admission queue, each ending in Rust maps (the
12Z cycle's simulated radar, valid 15Z, above) and an observation scorecard.
The scheduler was killed mid-run and resumed without launching a cycle twice.
That run used an explicit runtime-fixture plan, the door-qualification form
described above, not an unattended country deployment.

MCP tools include `suite_plan`, `suite_status`, `suite_halt`, `suite_resume` and
`suite_rehearse`. Forecast admission accepts explicit physical `cards`; disjoint
sets can run simultaneously, while overlapping sets are refused with the holder
identity. Suite rehearsal is a CPU controller job; its rows acquire queue cards
individually.
