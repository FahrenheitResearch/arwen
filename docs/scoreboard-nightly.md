# Nightly CPU scoreboard

The native scorer owns continuous errors, ETS sufficient statistics and FSS.
The campaign harness calls its shared statistics and neighborhood code. The v1
clipped-mask policy and the visual harness complete-box policy retain their
different frozen support rules. No forecast arithmetic changes.

Run on a CPU host after the observation service seals its catalog:

```sh
python tools/scoreboard/nightly_job.py --catalog catalog.json --out nightly --as-of 2026-10-08T00:00:00Z
```

`catalog.json` has schema `woof.scoreboard.nightly-catalog.v1`, `complete: true`,
`window_start`, `window_end`, `comparator` and `runs`. The window must cover the
whole preceding 30 days. Public eligibility and finished state come from a
trusted run export, never bucket object presence or a template's current recipe.
The current public recent-run route has a five-row limit and cannot supply this
catalog. An export integration is required before unattended operation.

Each run has `run_id`, `is_public`, `deleted_at`, `template`, `status: finished`,
`init_time`, `observations_final_after` and `frames`. Each frame has `valid_time`,
`domain`, `baseline_model` and `score_request` with native `forecast`,
`observations` and `baseline` compact artifact paths. Export every expected frame,
including missing inputs, so each gap appears in the run receipt. The finality
watermark comes from the observation service's provider policy and sealed
revision. It is not inferred from elapsed time alone. Later provider corrections
require a revised catalog and yield one replacement contribution, not duplicate
statistics. Configure the comparator through metadata (HRRR for this deployment).

`nightly.json` (`woof.scoreboard.nightly.v2`) contains the seven- and 30-day
counts, a reason for each skip or partial run, the rolling 30-day template table
pooled across leads and runs (`templates`, with sufficient statistics), and
headline series per lead (`template_lead_series`) and per initialization day
(`template_day_series`). Each scored run also writes `summaries/<run>.json`
(`woof.scoreboard.run-summary.v1`): its own pooled table and its per-lead
headline series; the run row carries that file's sha256. The headline metrics are
request metadata (`headline`), defaulting to T2/Td2/10 m wind RMSE and bias,
1-hour ETS at 1 mm, 6-hour ETS at 5 mm and composite reflectivity FSS at 30 dBZ
and 25 km.

A frame may carry `score_requests` (a list) instead of one `score_request`: the
1-hour and 6-hour rain windows need their own forecast and baseline artifacts. A
metric two requests of one frame both carry is pooled once and reported as
`overlapping_score_requests`. A frame the driver cannot score carries
`skip_reason` (for example no comparator forecast at that lead) and still counts
in `frames_total`. After a frame's observations are final and it has scored, the
driver may delete its inputs and mark it `inputs_retired`: the verified receipt
is reused, and a missing or altered receipt becomes a named gap. Counts labeled scored include partial
runs with at least one paired frame; the per-run status and missing metrics are
visible. Domains and forecast leads remain separate, and both models use the
same valid hour and initialization. Pool sum-of-squares before taking RMSE and
pool contingency and fraction sums before ETS/FSS. Dry-event metrics remain null.

The ledger hashes the actual three input artifacts, score request and scorer
source revision, so a corrected scorer replaces old contributions. Warm
replays reuse verified score bytes. A process lock protects shared output;
publication writes are atomic. Output has no wall clock, elapsed time, cache hit
count or random identifier. The timing receipt is separate. Stable inputs and a
frozen cutoff produce identical publication and ledger bytes. The engine CLI
also exposes this through `gpuwm scoreboard --request nightly-request.json`.

The supplied timer is an installation proposal. It has not been enabled. The
job uses no GPU or forecast-worker lifetime. CPU hosting, acquisition and
storage still have costs; a zero-GPU receipt is not a zero-dollar claim. A host
cost estimate is `wall_seconds / 3600 * allocated_cpu_hour_price`, plus retrieval
and retention. Production deployment and source-worker extraction are owned by
the site service; this port does not deploy PR 145.
