# Forecast fraction and critical-path projection

`gpuwm timeline report rental-ledger.jsonl --output timeline.json` writes the
JSON report and `timeline.png` by default. `gpuwm timeline --help` and
`gpuwm timeline project --help` document both doors. No GPU is used.

FF is the union of `forecast.integrate` intervals across all cards divided
by `rental.terminated - rental.requested`. Parallel cards do not multiply
the numerator. The report includes rental wall, request to first step,
tail, and card-busy fraction. The denominator includes idle allocated cards.
Card-busy fraction requires intervals with `attrs.card_busy=true` and
`--cards N` or ledger `attrs.card_count`. Without those measurements it is
unknown. Integration card occupancy is reported separately. Neither value
claims hardware kernel utilization.

For an already-running sprint box, supply measured `--proxy-start UTC`
and optionally `--proxy-end UTC`. Without an explicit end, the last measured
output, post, upload, finish or artifact mark sets the endpoint. JSON and PNG
say PROXY. Rental FF and rental wall stay null. An incomplete rental never
closes at the parser's current time. A site launch-to-termination envelope
is labelled legacy rental and cannot establish rental-request FF.
The benchmark protocol's `worker.claim` and `finish` marks carrying a proxy
basis in `attrs.basis` also supply the proxy envelope automatically.

Supply multiple explicit files for one rental. Use `--rental-id` to select
one when a source contains several, and `--run-id` or `--product` to select
a product. Rental boundaries remain rental-wide. Append order is not event
order: same-host C1 intervals pair on monotonic time; their starting UTC
locates them across hosts. A clock adjustment is reported. Unmatched spans
and duration-only receipts never gain invented endpoints.
Non-forecast stages can pair across helper pids when exactly one open span
has the same host, rental, run, card, stage and span ID. This supports the
shell emitter's per-append helper processes. Ambiguous concurrent spans stay
unmatched. Forecast spans always preserve their process identity.

The same reader accepts engine `events.jsonl`, chain `progress.jsonl`
(including gzip), site cost JSON and scheduler receipts. Legacy step
emission time minus `step_wall_seconds` reconstructs first-step start;
the last step across domains closes the envelope. `model_progress` polling
samples and entire forecast stage timers do not establish integration.
Preparation plus forecast scheduler spans stay pipeline upper bounds.
Forecast-valid dates and source file mtimes never establish wall boundaries.

Small explicit sidecars use `gpuwm.timeline.sources.v1`, with `boundaries`
mapping a stage to `{at, evidence}`, `intervals` containing
`{stage, start, end, evidence, card, basis}`, and `unplaced` duration rows.
Sidecars document missing old events without changing the C1 emitter schema.
`tools/startup_timeline.py --ledger LEDGER --output timeline.json` delegates
to this calculator. Its existing startup-only arguments and report stay
unchanged.

`gpuwm timeline project --reference reference.json --deltas lane.json
--output projected.json` reschedules a dependency graph, with measured
and projected fractions side by side for each product. Repeat `--reference`
and `--deltas` for multiple products and lanes. `--png projected.png` writes
the projected waterfall for a single product. No delta file means baseline.

A reference uses `gpuwm.timeline.reference.v1`, `product`, `mode` (`rental`
or `proxy`), optional allocated `cards`, `measured_ff`,
`measured_wall_seconds`, and `nodes`. Each node has a unique `id`, `stage`,
measured `duration_seconds`, `evidence`, optional `card`, `depends_on` node
IDs and `lag_seconds`. Include all startup, gaps, barriers, output and tail
on the rental critical path. A resource ordering constraint is an explicit
predecessor edge. The envelope starts at zero and ends at the latest node.
The reference schedule must reproduce measured wall and FF within 1 percent.

The model requires measured dependency edges. Overlapping waterfall rows
alone do not reveal whether a nested timer includes another stage, whether
two jobs share a resource, or which stage releases a dependent job. Passing
an unannotated waterfall as a graph is refused because it could produce
unsupported rental savings. Deltas cannot shorten an unspecified dependency.

Each delta has a unique `lane`, `measured: true`, `evidence`, optional
`product`, a unique target `node` (or unambiguous `stage`), and either
`seconds_saved` or `removed_spans`. `depends_on` names prerequisite lanes.
An independent seconds-only delta removes the reference node's tail, so
independent estimates on the same work overlap. For disjoint measured work,
provide explicit `[start, end]` spans in that node's original coordinates.
A dependent seconds-only delta removes remaining work after its dependencies.
Its explicit spans must also be marginal and cannot re-claim predecessor work.
Missing prerequisites, cycles, duplicate lanes, ambiguous nodes, unmeasured
deltas and removals beyond retained work are refused with concrete reasons.

The graph is rescheduled after the union of removed work. A shorter
non-critical branch does not change wall or FF until it becomes critical.
An integration speedup changes both numerator and denominator, and can
reduce FF even when it saves wall. Never add `est_seconds_saved` across lanes.
