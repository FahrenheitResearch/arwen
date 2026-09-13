# Local rapid data assimilation

`gpuwm local-da` authors one externally forced regional domain, reviews
its resource costs, and composes the existing preparation, ensemble cycle,
checkpoint analysis and short-forecast paths. There is no second forecast
integrator or observation decoder.

Each forecast member writes its actual terminal state, even when the
forecast ends between history times. The frame uses the same diagnostics,
inventory and renderer as ordinary history output. A completed older
forecast with an empty frame inventory is preserved; restarting that case
reuses its saved analysis and produces the short forecast in an output
recovery directory. Missing frames or failed rendering leave execution
status `FAILED`, with the product failure recorded for retry.

## Review, publish, launch

```
gpuwm local-da --point 40,-100 --epoch 2026-09-10T12:00:00Z \
  --vram-gib 10 --host-gib 32 --budget-seconds 3600 --scale 1 --dry-run
```

The timestamp is an explicit example, not a latest-cycle resolver. It must
fall on the selected forcing source's initialization lattice. Review emits
one JSON document on stdout; diagnostics go to stderr. `--json` is accepted
and says so explicitly, since this door has no other output form. Every
field of that document, the ladder rows with their prices and verdicts, the
observation streams present and missing, the requested rung and its
advisories are written down once, in
[the companion protocol](local-da-companion-protocol.md), which
`gpuwm local-da --capabilities` names and whose field roster it publishes. Review does not
fetch observations, repair native libraries, write a run directory or open
a device. The installed dependencies and source/physics tables still have
to be present for the canonical forecast estimator to operate.

Replace `--dry-run` with `--out local-cycle` to publish the reviewed
configuration. Inspect the returned review before executing:

```
gpuwm local-da --launch local-cycle/local-da.json
```

Repeating that command uses the existing cycle and ensemble recovery
contracts. `--launch local-cycle/local-da.json --dry-run` validates and
reads the saved review without starting it. A generated directory contains
`experiment.toml`, `ensemble.toml`, `experiment.namelist.wps` and
`local-da.json`.
Their hashes, and any explicit observation input hashes, are checked again
before launch. `--run` combines publication and execution for a caller
that has explicitly requested both actions.

`--region west,south,east,north` replaces `--point`. East less than west
means the short arc across the dateline. The perimeter must fit the
projected domain with its boundary margin. The author never silently moves
the requested point or crops the requested region.

## The scale policy

Rung one has one forecast trajectory, one update, a 3 km domain with a
nominal 192 km span, and a 30 minute forecast. It uses eight independently
perturbed analysis states to prescribe a static covariance. Those states
are evaluated at the analysis instant only; they are not eight forecast
members. An observation innovation is evaluated against the actual
background, including for nonlinear operators. A zero innovation produces
an exact zero deterministic increment.

Above rung one, forecast membership grows as `2**(rung-1)`, bounded by the
ensemble owner's member limit. Spacing refines every third rung, the domain
span grows, and cadence derives from the cadence owner's reference
interval. The finite cycle count comes from the requested rung and is never
reduced by estimated cost. Review prices lower alternatives but preserves
the requested rung's region, resolution, members, cadence and duration.

The hierarchy is a stated policy, not a proof of optimal forecast skill.
A user may explicitly request any supported positive whole-second cadence.
Fractional seconds cannot be preserved by the analysis and forecast-output
timestamps and receive a named error instead of being floored. A cycle cost
above cadence is an advisory queue-lag projection, with fewer members,
coarser spacing and longer cadence listed as optional choices. Source horizon, geometry, missing inputs and inconsistent
clocks remain real refusal conditions.

Forecast members execute sequentially on one card. The analysis holds
multiple states simultaneously. Pricing separates those two memory peaks,
accounts for the full restart-field inventory and observation batches, and
bounds the localized solve workspace through its existing owner. The card's
compute multiplier is independent of its memory capacity. Timing uses the
existing forecast pace estimate plus a conservative extrapolation of the
recorded LETKF solve basis. Preparation has a stated allowance; network
latency is not bounded by it. Ten GiB fit tests inject forecast costs and
exercise the real analysis storage formula. They are not a measured device
admission certificate for the complete runtime. Estimated peaks above
declared or currently free memory are warnings, and actual allocation
failures remain operation failures with saved-state recovery guidance.

`--budget-seconds` is an advisory wall-time comparison target. It does not
change the finite cycle count or enforce a deadline. Execution records
measured per-cycle wall time and lag during this launch, excluding
preparation, and projects remaining finite work from the measured mean.
The chosen domain stays unchanged even when the run falls behind cadence.

## Observation paths and their limits

The regional registries supply radar coverage, native decoder identities
and surface networks. Missing streams are reported and skipped rather than
converted into synthetic observations. Available types enter the same
localized analysis, not independent increments added without accounting
for their combined covariance.

* Radar: the registered grid builder, per-radar radial velocity, the
  selected physics scheme's reflectivity operator, and clear-air products.
  Existing thinning, error, beam, positivity and paired-moment contracts
  remain in force. The reference geometry is a frozen initial field written
  and read through the existing native geometry paths.
* Surface: the existing nominal-time surface record supplies 2 m
  temperature and 10 m wind speed, evaluated against end-of-leg diagnostics.
  Vector components from neutral tables use earth-relative winds. Terrain
  elevation mismatch and absent diagnostics are counted rejections.
* Shared neutral tables: pressure-level temperature, wind and dewpoint,
  assigned-pressure satellite winds, and tangent-point refractivity use the
  installed `gpuwm-global` table decoder and operators. Each member supplies
  its own full pressure column. There is no vertical extrapolation. The
  default supported surface measurement labels are explicit; station
  pressure, sea-level pressure and unsupported platform quantities are not
  silently mapped to another measurement.
* Satellite cloud water: existing phase-aware cloud-water-path grids can
  enter through their registered adapter. Automatic regional satellite and
  sector selection is not implemented. Infrared/microwave radiances are not
  enabled: they need a regional column interface, coefficient identities,
  cloud treatment, bias/error configuration and numerical validation.
  Brightness temperature is never treated as a point air temperature.

The shared stream registry can be reused when its package and native doors
are installed. Account-gated streams, undecoded payloads and unbounded
subscriptions are reported by name. It does not create another ingest
stack. Explicit inputs use repeatable `--obs-table`, `--radar-grid` and
`--satellite-grid` arguments.

Observation windows are immutable after their first successful publication.
Their receipt binds the review, time, grid and exact input bytes. Resume
reads those same bytes. Quality control records future measurements,
arrival/publication cutoffs, invalid errors, gross limits, background checks,
duplicates, revision selection, thinning and out-of-domain observations.

**Current timing limitation:** neutral rows must have arrived and been
published by the analysis valid time. A fresh download made after that time
is excluded when its receipt time is recorded, including historical replay.
No receipt timestamp is rewritten or removed. A separate explicit
acquisition cutoff/replay policy is still needed for routine delayed live
cycling and retrospective fetched tables. Missing arrival metadata is
counted as latency unverified, not evidence of real-time availability.
The legacy surface record carries nominal times, not complete original
report/arrival timing. The current window is one cadence wide; delayed
upper-air and satellite reports may therefore be absent.

## Scientific and operational qualification

The deterministic covariance is prescribed, not an evolved ensemble
covariance. Covariance samples use the existing perturbation generator,
paired hydrometeor treatment and CPU equation-of-state refresh. Surface
covariance uses a frozen-transfer tangent approximation around the actual
end-of-leg diagnostics. The prior boundary conditions are shared.

Multiplicative condensate perturbations cannot create a storm where all
samples are cloud-free. Cloud initialization, displacement-aware updates,
additive inflation, balanced insertion, bias correction, correlated-error
operators, 4D trajectory sampling and forecast-skill evaluation are not
implemented by this patch. Refractivity uses fixed reference heights, and
radar gridding/localization also uses initial geometry rather than updated
member geopotential. These approximations need sensitivity tests.

The default member count, samples, localization, perturbation amplitudes
and error choices are a starting policy, not calibrated forecast skill.
Cadence-derived inflation is applied to every enabled observation family.
No-observation cycles publish explicit zero increments and a forecast-only
receipt. There is no new persistent no-observation staleness budget in this
regional launcher.

Products reuse the existing Rust renderer and hash-verified per-member
output inventories. No ensemble probability or mean-product aggregation is
added. Prepared-cache startup, native radar acquisition, reference-file
creation, device execution, desktop execution and rendered products still
need end-to-end qualification. A partial preparation without its final proof
needs manual recovery; only completed preparation and published cycles have
the tested recovery path. The lock is per run directory, not a cross-run
physical-card reservation. Online ingest size and timing are not a hard
upper bound. None of these limits is presented as a forecast-skill result.
