# Suite plan JSON contract

`gpuwm suite plan FILE --json` emits one JSON object on stdout. Exit 0 means
the complete operational suite fits. Exit 2 means refused. This contract is
`gpuwm.suite.plan.v1`; changes after the contract commit only add fields.
Consumers ignore unfamiliar fields. No command is executed by the planner.

Times are seconds from input **ready**, never a hard-coded posting clock.
The four blocks are UTC cycle hours 0, 6, 12 and 18. Hourly jobs carry a ready
offset within that block; the loop asks the source registry for actual ready.

| Field | Meaning |
|---|---|
| schema | Literal `gpuwm.suite.plan.v1` |
| suite, region | Resolved suite name and region registry id |
| suite_sha256 | SHA256 of canonical resolved suite JSON, null only in the initial stub |
| admitted, exit_code | Boolean and 0 or 2; launch nothing when false |
| block_seconds, cycle_hours | 21600, `[0,6,12,18]` |
| boxes | Per-box id, host, cards, memory, capacity and daily card-hours |
| rows | Resolved row id, shape, box, rate provenance, daily card-hours and dependencies |
| jobs | Each reservation's id, row, phase, cycle_hour, ready_offset_s, start_s, end_s, deadline_s, box, card_ids, rate, card_hours, argv, env and dependencies |
| measurement_slots | Unqualified experiments, outside operational totals; never automatic launches |
| refusals | Concrete row, reason, rate, budget, required, overrun, unit and message |
| daily_card_hours | Sum of admitted operational reservations across the four blocks |
| what_if | null or expression, baseline and delta daily card-hours |

`argv` is an array of tokens, not shell text. `env` carries selected physical
cards. Prepared path tokens such as `{prepared.mesh}` and `{cycle}` are explicit
bindings for the loop, not literal executable paths. A job with unresolved
bindings must be prepared before launch. A reservation can represent one
ensemble member even when the door launches a roster; the controller must use
the reservation group fields added by the final planner. Measurement slots
never become production just because projected times fit.

Refusals retain numeric budgets (seconds, MiB, cards or card-hours) and source rate
ids. `when-fits` rows still refuse the complete plan when enabled and unfit;
set `enabled=false` to declare a dormant candidate without blocking operations.
Products, verification, publishing and retention are CPU stages outside GPU
reservations, recorded as row policies and dependency stages by the controller.

The initial stub emits this envelope with `admitted=false`, empty placement,
and an `unpriced` refusal. It intentionally cannot license a run.

## Placement and controller bindings

The implemented planner emits the same envelope. Admission means that the
declared reservations fit the historical rates and memory admission. It does
not qualify a new engine build, stochastic arm, mesh, source, DA promotion or
product set. `execution_ready` remains false until the controller materializes
and validates every binding. Historical rate builds are retained verbatim;
`rate_build_matches_inventory` exposes disagreement rather than inventing a
rate for the installed build. Add measured rows and select their ids in data.

`boxes[].blocks` contains `cycle_hour`, `card_hours`, `budget_card_hours`, and
`lanes`, a mapping of physical card id to sorted `[start_s,end_s]` pairs.
Intervals are half-open. `jobs[].block_hour` identifies the block in which its
times are measured. Recurring jobs choose the earliest feasible placement,
then avoid their previous card group on ties, then sort by physical card id.
Rows sort by priority and id. Shadow DA fills remaining gaps after forecasts.
CPU products, verification, publishing and retention do not consume GPU slots.

`card_ids` are physical inventory indices. `logical_card_ids` are zero-based
indices inside the reservation's `CUDA_VISIBLE_DEVICES`. Door argv uses these
logical indices. Multi-card hex forecasts also require prepared layout and
partition bindings. Runtime-registered meshes require a validated mesh-rows
file, carried as `env.GPUWM_HEX_MESH_ROWS`; the catalog alone cannot bind geometry.
An inventory with insufficient eligible cards reports its eligible count as
the budget, requested count as required, and missing count as overrun. A card
whose memory is too small instead reports the memory deficit in MiB/card.
Rate-table edits with missing, zero or invalid forecast timing refuse before
placement, so missing timing cannot become a zero-length admitted job.
Optional timing qualifiers such as `all_in_measured` and `all_in_basis` retain
historical boundary limits alongside the capacity numbers; the controller
must qualify the installed build and workload before operational launch.

For CAM reservations, `recipe_reference`, `config_edits` and
`run_plan_document` describe the files the controller must produce. The latter
is the existing `gpuwm.run-plan.v1` envelope with route `prepared`. Physics
aliases resolve through the reference table to `[shared]` keys; the existing
experiment loader must validate the materialized TOML. Each ensemble job is
one perturbed member, with a distinct seed. Stochastic selections survive in
settings and config edits even where the base engine refuses their execution.
`overrides` retains all requested member selectors. Member ranges are zero-based
and inclusive. The planner prices the resulting per-member scheme through the
existing CPU memory estimator and refuses if that estimator cannot price it.

For hex ensemble jobs, the controller must produce a separate one-member
manifest per reservation, selecting `manifest_selection.member` from the source
manifest. `manifest_forecast_flags` carries its scheme overrides. Reusing the
first member manifest for all reservations would produce duplicate trajectories.
The argv binding names identify distinct artifacts for each member and cycle.

Shadow jobs carry an unresolved `da.shadow.command`, `background_dependency`
on the previous ensemble cycle, and measured `cpu_solve_seconds`. The suite
planner does not fabricate a JEDI launch command or promote a scientific arm.
Production DA without a qualified promotion receipt refuses. The controller
owns source posting probes, the previous-cycle ledger, CPU stages, prepared
file sharing, disk checks and retention, using the complete row `settings` and
suite `defaults`. A `from` reference with lagged members also requires reuse
of the previous hourly receipts, not new GPU runs.

Measurement slots have `launch_allowed=false`, projected `estimated_seconds`
and possible `windows` in the operational lane gaps and remaining 80% budget.
They are excluded from daily totals even if a projected window exists.

Rates include administrative `reservation_floor_s` where the plan rounds a
measured rate up for budgeting. `reservation_basis` cites that arithmetic.
The normal duration is startup plus all-in seconds per forecast hour times
lead; `all_in_includes_startup=true` prevents double counting a whole-run
measurement. Memory comes from measured peaks, a conservative measured box
capacity bound, or the engine CPU allocation estimator. A missing bound refuses.

Validation refusals also emit the envelope, with empty rows and placements.
Never launch the partial jobs of a refused plan. Numeric refusal overruns use
the earliest achievable finish when card gaps prevent meeting a deadline.

`--what-if` changes existing resolved row keys, validates them and reports the
baseline and delta for the returned reservations. A refused delta is not the
full requested workload: inspect `refusals` before interpreting its total.
The unchanged input suite is never written by a what-if query.

`[[override]].set` also accepts `door.FLAG` and `members[N..M].door.FLAG`.
FLAG is the public flag without its leading `--`, listed with parser provenance
in the shape table. Unknown names and invalid types refuse. Lead, source and
card-count aliases change the normalized row before rate selection and memory
admission. Per-member changes to those resources require a separate row.
Physical card flags belong to placement and cannot override reserved lanes.
Base flags that bypass admission or occupy ignored benchmark slots refuse,
citing the silent-drop defect. Other forecast-affecting flags require identical
`rates[].door_flags` provenance; adding that measured row is a data edit.
Output and receipt controls pass directly to argv. CAM flag overrides select
the existing `go` or `ensemble` alternate door over a materialized config.

`source_selection` retains the ordered IC and LBC ladders and the first-ready
rule. The controller must bind a prepared source receipt. CAM config edits
carry `{selected.fetch.source}` explicitly, so an inherited recipe fetch source
cannot silently replace the suite's source selection.
