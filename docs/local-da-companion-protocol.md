# Local DA companion protocol, version 1

Discover the engine contract with `gpuwm local-da --capabilities`. This
prints `arwen.companion-local-da.v1`, request/review schema names, field
rosters, coordinate order, defaults, the member limit from its owner, and
argument arrays for review, publication, launch and saved-review reading.
It performs no pricing, acquisition or device query.

Transport follows the existing companion query convention: launch the
installed engine executable with an argument array, never an interpolated
shell command; read exactly one JSON document from stdout; show stderr as
diagnostics. A nonzero process result is an error even when JSON parses.
Do not mistake `forecast_started: false` for successful completion.

## Request

Send UTF-8 JSON to stdin of `gpuwm local-da --request-json - --dry-run`:

```json
{
  "schema": "arwen.local-da-request.v1",
  "epoch": "2026-09-10T12:00:00Z",
  "point": [40.0, -100.0],
  "scale": 1,
  "budget_seconds": 3600,
  "forecast_seconds": 1800,
  "card": {
    "vram_gib": 10,
    "free_gib": 10,
    "host_gib": 32,
    "speed_factor": 1,
    "name": "declared card"
  },
  "obs_tables": [],
  "radar_grids": [],
  "satellite_grids": [],
  "base_seed": 0
}
```

Use `region: [west,south,east,north]` instead of `point`; never send both.
`free_gib` may be omitted. Optional `source` and `profile` are resolved by
the engine's authoring authorities. `cadence_seconds` is an explicit cadence
request, not an independently guessed value. Unknown top-level request and
card fields are refused. Do not combine JSON input with field-setting CLI
arguments; the JSON document is the field authority.

## The review document

One `gpuwm local-da --request-json - --dry-run` (equivalently
`--dry-run --json`; this door writes one JSON document on stdout either
way) answers with `arwen.local-da-plan.v1`. Its `schema` field carries that
string and a companion refuses any other value rather than guessing at the
shape. The document is the whole plan review: what was asked, every rung
that was priced, what each rung cost, which rung was taken, which
observation streams exist for that region and time, and what is unknown.

Twenty top-level fields, all always present. `gpuwm local-da
--capabilities` publishes the same roster as `review_fields`, so a
companion can check a build against the one it was written for.

| field | what it carries |
| --- | --- |
| `schema` | `arwen.local-da-plan.v1`, the document contract |
| `status` | `ready`; a review that is not ready is a refusal document instead |
| `forecast_started` | always `false` here; review allocates nothing and starts nothing |
| `request` | the request as the engine understood it, card included |
| `selected` | the exact requested rung, with its geometry, members, cadence and cycle count preserved |
| `changed_scale` | false for new reviews; retained so older saved reviews remain readable |
| `region_checks` | containment, dateline arc and span checks for the region or point |
| `alternatives` | the ladder: one row per rung, with its price and its verdict |
| `observations` | one row per observation stream, present or missing, with the reason |
| `observation_slots` | observation-space slots the analysis is sized for |
| `clock` | the cycle spine: anchor, tick rate, step and cycle ticks |
| `analysis_times` | the UTC valid time of every analysis the plan will make |
| `cadence_settings` | the applied localization and inflation from the cadence owner |
| `cadence_overrun` | that same owner's verdict on whether one cycle outruns the cadence, with `cost_basis` saying whether that verdict weighed a timing or a price |
| `memory` | forecast peak, analysis peak, host peak, each against its budget |
| `wall` | per-cycle, forecast, preparation and total seconds, with `measured` and `basis` |
| `configuration` | the authored `experiment.toml`, `ensemble.toml` and `experiment.namelist.wps` text |
| `inputs` | SHA-256 of every observation file the review was computed from |
| `warnings` | what the operator is owed but is not a reason to stop |
| `review_sha256` | the digest a launch confirmation is bound to |

### The ladder, its prices and its verdicts

`alternatives` is the ladder in requested order, lowest rung first, and it
is the evidence for the choice. Every row carries the same key set, and
`--capabilities` publishes it as `alternative_fields`: `scale`, `members`,
`dx_m`, `cycles`, `peak_bytes`, `cycle_seconds`, `total_seconds`, the
verdict `fits`, `priced`, the `reasons` behind a false verdict, the
`remedies` that answer those reasons in the same order, and `cadence`, the
cadence owner's own `gpuwm-da.cadence-overrun.v1` record for that rung.
A row whose `fits` is false names why in `reasons`: peak memory against the
usable budget, host arrays against declared host RAM, cycle cost against the
cadence, total cost against the time budget, or forcing beyond the source
horizon. `remedies[i]` is the way out of `reasons[i]`. Draw every row. The
rung that was not taken is how an operator reads what one more step would
cost.

`priced` is false on a rung the ladder stopped before it could price at
all: the rung's run needs forcing past the source horizon, and no arrays were sized
and no timing was derived. An unpriced row carries `null` in `members`,
`dx_m`, `cycles`, `peak_bytes`, `cycle_seconds`, `total_seconds` and
`cadence`, and carries its `reasons` and `remedies` as any other row does.
Read `priced` and draw such a row with its reason in place of its numbers;
`--capabilities` publishes the flag's name as
`unpriced_row_discriminator`. The keys are never absent, so a consumer
never needs a default for a missing one.

`selected` is the requested rung. Estimates cannot reduce its geometry,
resolution, members, cadence or finite cycle count. `fits` is a comparison
with declared resources, never permission to publish or launch. A ready
plan with `fits` false launches normally, with no override flag or extra
approval. Lower rungs are alternatives for an explicit later choice.

`memory.policy` and `wall.policy` are `advisory`. `wall.budget_seconds`
records the declared comparison target. `wall.budget_semantics` states
that it is not an enforced deadline: it neither shortens the finite cycle
run nor kills it when wall time exceeds the target. The run ends after its
authored cycles and forecast, or an explicit stop or actual failure.

`cadence_overrun.policy` is `queue`: every requested cycle is kept even
when its estimated duration exceeds cadence. `cost_basis` distinguishes
`estimated` from `measured`; the owner records projected backlog and its
basis. Neither a prediction nor a measured lag changes the selected domain.
Explicit cadence and forecast durations must be positive whole seconds:
saved analysis and ordinary forecast-output timestamps cannot preserve a
fractional second. Such a request receives a representation error instead
of being silently rounded. Supported explicit cadences are kept exactly.

`wall.measured` is false whenever the price comes from a recorded basis
rather than from timing this configuration, and `wall.basis` lists every
source line behind the number. A rung whose physics combination has no
recorded pace row at all is still priced, from the slowest rate on record,
and both `wall.basis` and `warnings` say so in those words. An unmeasured
price is a warning and never a disabled launch button.

### Observation streams, present and missing

`observations` is one row per stream the region and time could use, each
with `id`, `status` and `reason`. `ready` and `candidate` mean the route
exists for that region and time; `candidate` is coverage or capability, not
a received observation. Every other status is a stream that is NOT there,
with the reason in its own words: no site in range, no decoder installed, an
account-gated feed, a measurement label with no regional operator. Render
the missing rows as prominently as the present ones. Nothing in the engine
converts a missing stream into data, and when no route is available at all
the plan still runs and `warnings` says the cycles will be forecast-only
with explicit zero increments.

With no explicit satellite files, a new review selects automatic native
CWP discovery. Its row carries `binary_sha256` and `acquisition_policy`;
these are frozen with the review. Existing saved reviews retain their
recorded routes. Explicit local satellite files keep their own identity
and are not replaced by automatic data. See
[the source and quality contract](local-da-automatic-cwp.md).

### Refusals

Every refusal, from either door, is one JSON document on stdout with
`schema`, `error`, `code`, `details` and `forecast_started`, and the
process exits nonzero. A refusal carries the same `schema` string a
successful review carries, so tell the two apart by the presence of
`error`, which only a refusal has; `--capabilities` publishes that rule as
`refusal_discriminator`. The code roster is published under
`refusal_codes`, as `review`, `launch`, `run` and `fallback`, and what each
group leaves on disk is published beside it as `refusal_effects`. Read both
from there rather than from this page: the door checks its own roster
against its own source, so the published lists are the ones that cannot
drift.

A **review** refusal (`refusal_codes.review`) decides a configuration
before anything is written, fetched or allocated. There is no case
directory and no execution document; the way out is in the message text.
Estimated card, host-memory, cadence and wall-time overruns are advisories,
not refusal codes. `warnings` summarizes the selected rung's simultaneous
comparisons once per optional remedy; `alternatives` retains every reason.
`MISSING_DOMAIN` means the authored experiment has no
domain to price. `REVIEW_CONTRACT` means the door's own result, or one of
its ladder rows, drifted from the roster published beside it, which is a
defect in the engine, not in the request. `INVALID_PLAN` is a malformed
request. `FORCING_HORIZON` means the cycle run needs forcing past the
source's horizon; shorten the run or select a later initial cycle. It is
raised when the requested rung lacks the required forcing. A lower rung's
estimated resource overrun does not replace this concrete missing-data reason.
`details.alternatives` retains the available comparisons and horizon row.


A **launch** refusal (`refusal_codes.launch`) is raised after a review was
published, by the integrity checks a published case runs before its
execution document is opened. A case directory exists, `forecast_started`
is false, and the refusal writes no execution document, so if the
directory holds one it is from an earlier attempt. These divide into two
kinds. The saved case disagrees with its review: `PLAN_SCHEMA`,
`REVIEW_CHANGED`, `CONFIGURATION_CHANGED`, `ROSTER_CHANGED`,
`OBSERVATION_CHANGED`; the way out is to restore the reviewed bytes or
generate a new plan. `MISSING_GEOGRAPHY` names an absent preparation asset;
install that geography before launch. An estimated peak above currently
free VRAM only adds a launch warning. Missing devices and actual allocation
failures are still reported by their operation owners with recovery guidance.

A **run** refusal (`refusal_codes.run`) carries the same document and the
same four fields, and it is raised after the case's execution document is
open, by a check that reads what an earlier stage published and therefore
cannot be made at review. The case directory holds an
`arwen.local-da-execution.v1` document with status `FAILED`, and
`forecast_started` says whether a member had already begun, which it can
be: read it from the document rather than assuming it false. Treat a run
refusal as a coded run failure. `MISSING_MANIFEST`, `MISSING_SURFACE` and
`PREPARATION_CHANGED` mean preparation did not publish what the next stage
reads, before any member integrated. `OBSERVATION_WINDOW_CHANGED`,
`REFERENCE_CHANGED` and `SURFACE_CHANGED` mean a saved window, reference
grid or diagnostic record no longer matches the run it is being resumed
into; restore the reviewed bytes. `ANALYSIS_ROSTER` means a cycle did not
publish every reviewed member, so the short forecast was not started.
`OUTPUT_CHANGED` means published output frames failed inventory
verification after the forecast completed. The way out for all of them is
in the message text, and the execution document says how far the run got.
`CWP_OPERATOR_UNAVAILABLE` means explicit satellite input requires
condensate species absent from the prepared column. It is reported before
the first member integrates; automatic CWP instead records that missing
optional operator and lets other streams continue.
`CONTINUOUS_WINDOW_FAILED` means a continuous window's operation failed;
`error` names the window, the stage and the reason, `details` carries them
as fields, and launching the plan again resumes that window.

`refusal_codes.fallback` (`LOCAL_DA_ERROR`) carries anything the door did
not raise itself, so a switch on `code` always has an arm to land in. Its
group is whichever stage it escaped from, so read the case directory for
an execution document rather than assuming either way. Treat its `error`
text as the whole of what is known.

Nothing is refused for being unmeasured, unpriced or unvalidated. A rung
nobody has timed is priced from the most conservative basis on record and
run, and the review says so in `wall` and in `warnings`.

## Continuous cycling

A request may carry `continuous_windows`, a whole number of analysis
windows; `gpuwm local-da --continuous N` sets it. Zero, the default, is the
reviewed finite cycle described above. A positive count publishes the same
review with one more field, `continuous`, and launching that saved plan
runs a bounded continuous cycle instead of the finite one: window `i`
analyses at `epoch + (i + 1) * cadence_seconds`, restarting its forecast
leg from window `i - 1`'s analysis, assimilating the observations of its
own window, then running the reviewed short forecast and rendering it. The
initial condition, the analysed state and the scientific settings never
change between windows. Every window's products are rendered, in order,
and a window that is late is run late and reports its lag; nothing is
skipped to catch up.

The `continuous` review field carries `enabled`, `windows`, `status_schema`
(`arwen.local-da-continuous-status.v1`), `status_relative_path`
(`continuous/status.json`), `control_relative_path` (`continuous/stop`),
`window_relative_path` (`continuous/window_{index:06d}`), `product_policy`
(`every-window-in-order`) and `renewal_policy`. The publication record
gains `status_path` and `control_path`. `--capabilities` publishes the
continuous contract under `continuous`, with the `statuses` roster, the
`terminal_statuses` subset, and the argument arrays for `status_command`,
`stop_command` and `resume_command`.

### Forcing renewal

The reviewed `background` covers the finite cycle. When a window's short
forecast reaches past the forcing the case holds, that window renews the
forcing before it starts: the same source cycle is prepared again over a
longer window through the automatic preparation chain, into
`window_NNNNNN/forcing/`, and the renewal receipt
(`arwen.local-da-forcing-renewal.v1`, at `window_NNNNNN/forcing/renewal.json`)
records the selection, the fetch hints, the frames kept and the frames
appended. The renewed preparation is admitted only if it keeps the
reviewed geometry, physics, output and clock, initialises from the same
source cycle and lead, and carries every forcing frame already run under
before the appended ones; every member checkpoint of a continuous case
carries the preserved forcing-prefix contract, and a restart is refused at
the checkpoint if the live forcing changed a frame the checkpoint was
written under. A review whose `background.inputs.kind` is `prepared` or
`local` cannot be renewed, and a window that needs renewal on such a
review is refused by name: supply inputs that already cover the whole
continuous window, or review on an automatically prepared source. A review
saved before the background contract cannot be renewed either; publish a
new review. Frames the source has not published yet are waited for, with
the status document reading `WAITING_FORCING`, for at most one forcing
interval past the window's analysis time; after that the window fails and
names the frames it waited for.

### The status document and the stop request

`gpuwm local-da --status PLAN` prints the status document of a saved
continuous plan and exits; `gpuwm local-da --stop PLAN` writes a durable
stop request under `continuous/stop/` and prints the same document with
`stop_requested` true. A running controller honours the request between
operations and reads `STOPPED`; a launch made while no controller runs
clears the request and resumes, so a stopped plan resumes with a launch and
stops again only if asked again after that launch. `--launch PLAN` on a continuous plan resumes its
unfinished window through the same immutable decisions: an interrupted
analysis is recovered, not recomputed, and a window whose products were
committed is never re-rendered.

The document is `continuous/status.json`, rewritten atomically at every
stage change. `status` is one of `NOT_STARTED`, `PREPARING`,
`WAITING_TIME`, `WAITING_FORCING`, `OBSERVATIONS`, `ANALYZING`,
`FORECASTING`, `RENDERING`, `READY`, `STOPPING`, `STOPPED`, `COMPLETE`,
`INTERRUPTED` and `FAILED`; the last four are terminal. `windows`,
`completed_windows`, `remaining_windows`, `active_window` and
`analysis_time` place the run in its bounded sequence; `next_analysis_utc`
accompanies a wait; `lag_seconds` and `processing_seconds` describe the
window just completed; `latest_products` names the committed product
receipt of the newest completed window, whose `images` list is the
rendered pictures; `observation_usage` is the completed window's accepted
observation counts, in the shape the finite execution document uses;
`nowcast_score` is the live skill of every completed window, read off the
window receipts rather than off a snapshot, so a lead an earlier window
left pending appears as soon as a later pass scores it. A fault reading
those receipts is recorded as `nowcast_score_error`, a type and message,
beside a null `nowcast_score`, and never fails the window or the status
door: the score is a number about the run and not a gate on it.
`controller_alive` is decided by the held OS lock and the matching owner
identity in `controller_owner`, never by the age of the file: a non
terminal status whose owner is gone reads `INTERRUPTED`, and the document a
controller leaves behind when it exits already says `controller_alive` is
false, so the raw file never claims a run its process has left. `elapsed_seconds`,
`session_elapsed_seconds`, `current_stage`, `current_stage_elapsed_seconds`,
`stage_seconds` and `total_stage_seconds` are measured controller time,
across launches, with stopped downtime excluded, as `timing_scope` says.
`reason`, `failed_window` and `failed_stage` accompany `FAILED`.

Each window directory holds its immutable decisions: `inputs.json` (the
forcing generation and the assets it binds), `analysis.json` (the analysis
roster and the cycle receipt), `products.json` and `execution.json` (the
window's forecast, its rendered images, its observation usage and its
nowcast score) and `complete.json`, with `cycles/`, `forecast/` and
`products/` beside them in the same shapes the finite run publishes.
`nowcast-score.json` sits beside them and is the one document in the window
that is rewritten after the window commits, because a lead cannot be scored
before its own valid time; `complete.json` names it and carries the score as
it stood at completion, so the receipt is the current answer and the
completion record is the record at the time. `continuous/head.json` names the
newest completed window and `continuous/binding.json` the clock the
sequence was published under.

A continuous run that fails exits nonzero with the run refusal
`CONTINUOUS_WINDOW_FAILED`: `error` names the window, the stage and the
reason, `details` carries `window`, `stage` and `reason`, and
`forecast_started` says whether a member had begun. Completed windows,
their products and their analysis checkpoints are retained, and launching
the saved plan again resumes the failed window.

## The nowcast score

Every local DA run scores its own forecast against the radar, with no flag,
and the private workstation reads the same documents.

`nowcast-score.json` is `arwen.local-da-nowcast-score.v1`. A finite run
writes it in the case directory; a continuous run writes one per window at
`continuous/window_NNNNNN/nowcast-score.json`. It carries `analysis_time`,
`scored_utc`, the `evaluating_tree` that produced it (package, version and
the enclosing checkout's commit), the full `registration` with its
`registration_sha256`, the `primary` statistic (threshold, half width, box
length in metres and the sentence naming it), the `grid` it was scored on
with the scored interior cell count, the `model` record (the forecast
manifest, the domain, the member scored, the members available and the
reflectivity variable and reduction), the `observations` record (the bucket,
the product, the decode box, and every archive object taken with its key,
its `s3://` URI and its SHA-256), the `analysis_scan` the persistence
baseline was carried from, and `leads`.

Each entry of `leads` carries `lead_minutes`, `valid_time` and `status`, one
of `scored`, `pending`, `missing-obs` and `unavailable`. A scored lead adds
`scan` (the frame used, its offset from the requested instant, its SHA-256
and the archive object it came from), `observed_coverage_fraction`,
`interior_valid_fraction`, the full `fss` matrix over every registered
threshold and neighbourhood, `primary_fss`, `primary_fss_useful`, both base
rates, `scored_cells`, the `regrid` plan record, a `persistence` block with
the same matrix and primary for the analysis-time scan carried forward, and
`difference_primary`, which is the model primary scalar minus the
persistence one. An unscored lead carries `reason` instead and never carries
a score; a `missing-obs` lead below the coverage floor also carries
`minimum_observed_fraction` and every `candidate_frames` row the selection
rejected. `leads_scored`, `leads_pending`, `leads_missing_obs`,
`leads_unavailable`, `primary_by_lead`, `persistence_primary_by_lead` and
`difference_by_lead` are the same facts indexed for a consumer that only
wants the headline.

The compact form is `arwen.local-da-nowcast-summary.v1`: `receipt_path`,
`receipt_schema`, `analysis_time`, `primary`, `registration_sha256`, the
four lead lists, and `leads`, one row per lead carrying `lead_minutes`,
`valid_time`, `status`, `primary_fss`, `persistence_primary_fss`,
`difference_primary`, `primary_observed_base_rate`,
`primary_model_base_rate`, and `reason` when the lead is not scored.
The two base rates are in the compact row and not only in the receipt
because an FSS of 1 with `primary_observed_base_rate` 0.0 means the
radar found no echo at the primary threshold anywhere in the scored
interior, not that the forecast was perfect: persistence scores 1 there
too and `difference_primary` is 0.0, so without the base rate a clear
box and a skilful forecast are the same three numbers. The finite
execution document carries it under `nowcast_score`; a continuous window's
`execution.json`, `products.json` and `complete.json` carry it under the
same key; the status document carries `nowcast_score` with a `windows` list
of those summaries and an `open_leads` count. A summary's own `status` is
`scored` only when at least one lead of that window carries a number;
otherwise it is `pending`, `missing-obs` or `unavailable`, in the same words
the lead rows use, so a status document a reader only skims never shows
`scored` beside a window that scored nothing. A window whose score could not
be attempted at all carries `status` `unavailable` with the reason and no
receipt.

`gpuwm local-da --score PLAN` scores every still-unscored lead of a saved
plan now and prints the same summary per window. It exists because latency
is ordinary: the 60 minute lead of a real-time analysis cannot be scored
until an hour after that analysis. A running continuous controller does the
same pass itself as each window completes. `--capabilities` publishes the
contract under `nowcast_score`, with the two schemas, the registered leads,
the lead statuses and the `score_command` argument array.

## Review and confirmation

The field roster is the table above, and that table is the only one: a
second list here would be a second contract. Render selected members,
spacing, dimensions, cadence, cycles, total cost, both memory peaks and the
timing basis; render every observation status and reason. Older saved
reviews that selected a lower rung remain readable and must show that fact.

Publish the same request with `--out DIRECTORY`, without `--dry-run` and
without `--run`. This returns a newly computed review and
`publication.plan_path`. Availability or costs may differ from the earlier
preview. Display this published review before asking for launch approval;
never authorize a different `review_sha256` using an older confirmation.

After approval, run `gpuwm local-da --launch PLAN_PATH`. Do not hand the
TOML to the top-level `gpuwm cycle`: that command owns a different parent
adapter protocol. The local command composes the regional ensemble cycle
owner itself. Restart uses the same plan path. Keep the child process
handle for cancellation; an interrupted process does not count as a
successful analysis. Saved reviews can be rechecked with `--launch PATH
--dry-run`. Execution progress is atomically written to `execution.json`.
The terminal response uses `arwen.local-da-execution.v1` and identifies the
cycle/forecast manifests and product reports.

`COMPLETE` requires forecast frames and successful product rendering.
An unavailable or failed product stage leaves `FAILED` and its per-member
product report in `execution.json`; restarting retries products without
repeating a complete forecast. An older completed forecast with an empty
frame inventory is preserved, and its short forecast is rerun from the
saved analysis in an output recovery directory. Use `forecast_manifest`
from the execution document rather than assuming a fixed forecast folder.

## Desktop consumer

The desktop separates protocol records in `src/local_da.rs` from its
creation panel in `src/local_da_panel.rs`. Review, publication and launch
remain distinct actions over the engine-owned documents above. The
ensemble configuration is launched through the local command, not an
ordinary single-forecast action.

### Measured execution progress

The execution document may carry `warnings` from live preflight and
`cadence_progress` after a cycle finishes. Its `cost_basis` is `measured`;
`cycle_wall_seconds`, `mean_cycle_wall_seconds` and
`measured_cycles_this_launch` describe actual completed work.
`lag_seconds` is excess cycle wall time over the simulated cadence budget
for cycles completed during this launch, excluding preparation. It is not
observation age or a claim that a historical replay is current. On resume,
the measurement scope starts again and is stated in `scope`.
`remaining_cycles` and `projection` describe the remaining finite schedule
using the existing queue computation and the measured mean. The projection
is null after the last cycle. None of this telemetry changes configuration.

An actual operation failure preserves prior completed outputs and analysis
checkpoints, writes `FAILED` with the original `error` and a `recovery`
sentence, and exits nonzero. The CLI error carries that recovery sentence.
There is no automatic smaller-domain retry.

`observation_usage` contains completed cycle indices, `accepted_for_analysis`
and per-batch `accepted` counts from the analysis innovation masks after QC
and thinning. Those are accepted observation counts and not a skill number;
the skill of the forecast that follows is `nowcast_score` beside them. `cwp_accepted` is the column batch's accepted count. Zero is
explicit for a forecast-only cycle; older unreported counts remain null.
The included `routes` retain missing-feed reasons and optional
`observed_columns`, which counts source-QC columns before analysis. Display
these separately. Completed counts are recovered from the published cycle
manifest, so resuming does not erase them or count the same cycle twice.

The optional request selectors `source_cycle`, `source_product`, `source_member`,
`source_provider` and `forcing_cadence_hours` select the background product and
boundary frames. Source membership is separate from the regional ensemble size.
`source_root` supplies the existing local preparation handoff. `prepared_root`,
`prepared_config` and `prepared_namelist` identify an existing portable bundle
and its original authorities. `source_inputs` maps native roles to original
files when member verification needs those bytes. `supplements` carries the
preparation owner's `ROLE=PATH` inputs. Old requests may omit all these fields.

`background` records the frozen source selection, complete required boundary
window, preparation mode, actual supplied input bindings and exact fetch hints.
Launch reads that selection rather than selecting a newer publication cycle.
The `background_catalog` in discovery is projected from the live preparation,
product, member and acquisition owners. A recognized preparation does not imply
that automatic acquisition exists; supplied-input obligations remain explicit.
Old saved reviews without `background` retain their original preparation path
and authored fetch cycle. They are not silently reinterpreted as new selections.
