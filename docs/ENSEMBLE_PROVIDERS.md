# Ensemble input providers

The provider modules supply source recipes, native physical preparation,
posted member inputs and stochastic physics hooks. Native initial and boundary
inputs carry verified source, static, coordinate and unit authorities. The
source factory preserves original acquisition handoffs and preparation
configuration. Batched forecast integration and the observation campaign still
require their complete qualification. No observation-calibrated default for
recentered inputs is declared.

Public forecast requests refuse random perturbation descriptors and active
SPPT, SKEBS or SPP controls before source acquisition or GPU work. Their spread
amplitudes have not been calibrated against observations, so the resulting
ensemble spread and probabilities would be meaningless. Explicit all-off
controls remain ordinary forecasts. Operational-member, time-lagged and
multi-model source recipes retain their original source trajectories.
Automatic reference-noise fallbacks carry the same refusal. Component and
calibration-research code does not establish a usable forecast default.
The refusal covers only the random providers this line adds. An absent
perturbation, `perturbation = "none"` and the 2.8.4 references
`gpuwm.da.perturb` and `experimental-stub` keep their 2.8.4 behaviour,
paths and provenance warnings, including
`python -m tools.ensemble_forecast run/cycle`.

## Source recipes

`gpuwm.ensemble.recipes.build_recipe` resolves source rows and their existing
member grammars. Adding a deterministic input's operational ensemble is an
`ensemble_source` adapter entry. When a control is distributed separately,
`ensemble_control_source` identifies its deterministic source. A source mean
or spread is never a trajectory member.

| Recipe | Initial and boundary atmosphere |
| --- | --- |
| `input-ensemble` | One verified operational member, retained at every valid time. |
| `recentered` | Base atmosphere plus a bounded donor-member anomaly about a fixed full donor population. |
| `time-lagged` | Earlier cycles of one model, each covering the same initial and final valid times. |
| `multi-model` | Explicit distinct source/cycle trajectories on one target domain. |
| `control` | Repeated unchanged base source, useful as an identity control. |

Automatic N=1 selects the unchanged base. Automatic N>1 selects an input model's
declared operational ensemble. Other recipes require explicit selection until
observation-based evaluation establishes their defaults. Insufficient genuine
trajectories are reported before download. Members are never duplicated to
reach a requested count, except in the explicitly named control recipe.

Each `SourceTrajectory` includes its source, cycle and verified member ID.
`fetch_argv` builds arguments for the existing fetch door, including
`--as-posted`. A requested boundary beyond that cycle's declared horizon is
rejected. Acquisition paths are keyed by source identity. Replaying a member
with `recipe.select_members` preserves its original index and seed. Rebuilding
a differently ordered roster is a different recipe.

Time-lagged and multi-model recipes have front doors. Each member's own
trajectory is fetched and prepared by its source's ordinary chain, from a
copy of the config whose `[fetch]` table names that member's source, cycle
and start lead. The members then run through the ordinary ensemble session:

```console
gpuwm ensemble CONFIG --recipe time-lagged --members 4
gpuwm ensemble CONFIG --trajectories members.json
```

`gpuwm go` and `gpuwm run` take `--recipe`, `--trajectories` and `--members`
too. The config spelling is `[ensemble] recipe = "time-lagged"`, or
`[ensemble] trajectories` with one `{source, cycle}` entry per member for
multi-model, and a `gpuwm run-plan` plan whose config or
`run_options.ensemble` names a recipe is handed to the same door. The run
folder holds one prepared bundle per member under `members/`, the forecast
under `run/`, and `ensemble-recipe.json`, which records every member's
source, cycle, start lead, seed and bundle. These doors take a one-domain
config with a `[fetch]` table; a `[case_data]` config and the `--wrfinput`
and `--met-em` doors name one trajectory's files and are refused. A recipe
member is a real source trajectory, so the calibration refusal above does
not apply to it.

What makes two trajectories the same forecast is the model, the cycle and
the member. A source row names its model as `upstream_model_id`, so `hrrr`,
`hrrr-prs` and `hrrr-native` at one cycle are one HRRR run in three file
products: a multi-model roster that lists a run under several source ids is
refused, and two source ids of one model do not make it multi-model. A
time-lagged roster keeps the config's own `[fetch]` member (GEFS `p05`, for
example) at every lagged cycle. A member from another model is written that
model's own `[fetch]` table: its forcing cadence and, where its fetch crops
at the publisher, the domain's crop box.

Everything a request can be refused for is decided before the first fetch.
The door writes every member's config to a scratch folder and asks its
source's own chain plan, so `--dry-run` prints the member plan or the
refusal that names the member, and fetches nothing. A real run then asks
what the ordinary door asks before its fetch (the GPU runtime, the card,
the memory envelope once per source, the geography tree, the Rust renderer,
the products, and the disk for every member's download and prepared bundle)
before it claims its run folder.

On `gpuwm go` and `gpuwm ensemble` the launch flags mean what they mean for
one forecast:

| Flag | With a recipe |
| --- | --- |
| `--cycle` | Re-times the config, then plans the roster at that cycle. |
| `--readiness`, `--no-probe` | Answers for every member's window and runs nothing. The exit code is the worst member's (2 refused, 75 not yet, 0 ready). The document is the deciding member's `gpuwm.readiness.v1` with `state`, `ready`, `expected_ready_at`, `retry_after_seconds` and `refusal` answering for the roster, and `recipe.member_windows` holding each member's own document. |
| `--transport`, `--whole-cycle`, `--late-after-minutes` | Apply to every member's fetch. A host one member's source cannot pin is refused in the plan. |
| `--prepared-root`, `--restart`, `--data-dir`, `--supplement`, `--section`, `--keep-checkpoints` | Refused by name: each names one trajectory's bundle, checkpoint, download or donor file, or a product the ensemble does not draw. |

Under `gpuwm run` a recipe ensemble runs in the calling process, ahead of
the supervisor: there is no worker, no card lock, no fresh-process recovery
and no checkpoint to resume. `--outdir` is the case folder, which holds the
download caches and one stamped run folder per launch. `--gpu-uuid`,
`--restart`, `--supervisor-max-restarts`, `--prep-timeout`,
`--allow-shared-gpu`, `--health-debug`, `--preprocess-backend` and
`--directory-input-hash` are refused by name; `CUDA_VISIBLE_DEVICES` selects
the cards. `gpuwm resume` and `gpuwm branch` continue one checkpointed
trajectory, take neither recipe flag, and refuse a config that selects a
recipe.

A member's fetch, preparation or forecast that fails ends the run with one
line and that stage's exit code: 75 for a source lead that has not posted
within its budget (launch the same command again once it posts), 130 for a
Ctrl-C, and a stop signal's own code (143 for SIGTERM). `ensemble-recipe.json`
then says `failed` or `interrupted` and names the stage, the member and the
code under `failure`. A forecast failure takes its members from run control:
`member_id` is the failing member, and `member_ids` with `failed_members`
(the rows `run/ensemble-run.json` carries) name every member of a failed pack
or wave. The error class and text are the member's own, also when the runner
returned a code rather than raising.

A member count with no recipe (`gpuwm ensemble CONFIG --members 4`, or
`[ensemble] members = 4`) takes the automatic choice through the same doors:
the operational ensemble the source's adapter row declares. The door prints
the member plan and runs those members. Where the row declares none, or the
ensemble does not post the window's valid times, the door exits 2 before any
download and names the two recipes above as the remedy. N members never run
one input: they would be N copies of one forecast, with zero spread and
probabilities of 0 or 1.

A plain member count is a recipe run, so every flag in the table above and
every `gpuwm run` refusal above applies to it unchanged: `--cycle` re-times
the config and the members are planned at that cycle, `--readiness` answers
for the members' windows (the operational ensemble's, not the config's own
source), and `gpuwm run CONFIG --members N --gpu-uuid ...` is refused by
name. `gpuwm run-plan PLAN --readiness` answers for the members too when the
plan's config is on the `gpuwm go` chain. A refusal of the member plan
itself opens with the N copies sentence; a refusal of a later gate (the
card, memory, geography, the renderer, the disk) is that gate's own
sentence.

For example, this prints a source plan without fetching or using CUDA:

```console
python -m gpuwm.ensemble.recipes --source gfs --cycle 2026-10-01T00:00:00+00:00 --hours 12 --members 20
```

## Recentered physical fields

`NativePhysicalStore` captures actual horizontally mapped fields before native
real initialization. Native files and immutable manifests bind array bytes,
valid times, target geometry, captured statics, units, vector basis and vertical
coordinates. Pressure-level axes explicitly carry hPa; hybrid level indices
are dimensionless and require their actual Pa pressure field. Older stores
without this contract require an explicit source qualification certificate.

`RecenteredPhysicalPreparation` consumes a base store and the complete named
donor population. The native CPU bridge aligns donor pressure coordinates and
time brackets and converts donor humidity when needed. The base retains its
specific- or relative-humidity fields and its ordinary initializer branch.
Soil, surface properties and pressure coordinates stay with the base. Native
real initialization then reconstructs each member's model state and every
boundary knot. The bridge requires ensemble preparation ABI version 3.

The shared `recenter_field` operation accepts native CPU or CUDA float32
arrays after that alignment. Its standalone grid digest is only a caller
assertion; the store and preparation layers perform the source, coordinate,
time and unit verification. Equal array shapes alone never establish matching
physical coordinates.

The operator computes `base + scale * (member - full_population_mean)`.
The population order is canonical and independent of requested output count,
member order or card partition. A common per-cell scale enforces explicit
increment and physical-field bounds across the entire population, rather than
independently clipping members. Arithmetic uses binary64 with one final
float32 rounding. Limiting uses inward representable endpoints so the final
rounding respects the declared bounds. Mean preservation over the complete
donor population has float32 roundoff. A requested subset need not have zero
mean anomaly: recomputing its mean would change an individual member when N
or the card partition changes. Calibration must score the actual selected
subset, including any resulting ensemble-mean displacement.

Limits and amplitude are explicit `FieldBounds` values with units. They are
not a calibration. Apply the identical procedure to the initial valid time and
every boundary knot, then use normal native real initialization to construct
mass, thermodynamics and staggered state. Directly applying this operator to
an arbitrary prognostic field after initialization does not establish a
balanced recentered atmosphere.

Existing native source values outside physical humidity bounds can be admitted
through separate input bounds and remain unchanged in every member. Valid
in-range base values cannot acquire new out-of-range member values. Direct
NVRTC compilation preserves subnormal recentering inputs; native CPU and CUDA
checks cover original member selection, partitioning and limiting behavior.

## Source and runner facts are table rows

The ensemble code reads what a source or a preparation implementation can do
from tables and never from its name. A source or implementation with the same
capabilities is a row, not a branch. `tests/test_ensemble_table_law.py` walks
both tables and fails when ensemble code names a source or runner id.

| Fact | Where it is declared |
| --- | --- |
| A multi-member run from this source needs a fitted policy | `requires_ensemble_calibration` on the source row (`gpuwm/source_adapters.py`) |
| The implementation takes `--as-posted` | `as_posted` on the runner row (`gpuwm.source_cli.preparation_runners`) |
| It takes the native physical store options | `physical_stores`, `physical_base_prepared`, `provider_supplies_decode` on the runner row |
| It decodes from mapping and composition inputs | `composition_inputs` on the runner row |
| Prebuilt geography shared by members: the options and the build | `shared_geography` on the runner row |
| Its acquisition writes a role-keyed input manifest | `input_manifest` on the runner row |
| A decoder implements append-only lead admission | `DECODER_MODE_CONTRACTS` in `gpuwm/bridges.py`, keyed by executable |
| The physical field contract of a native implementation | a packaged document pinned in `gpuwm/source_authorities.py`, read by `gpuwm.ensemble.physical_fields` |
| What a preparation chain checks in a member config before its fetch | the chain's row in `gpuwm.regional_preparation.preparation_chain_reviews`, keyed by the chain IDs of `preparation_chains` |

A mapped source needs no contract document: its contract is derived from its
own packaged mapping.

## Posted physical inputs

`PostedPhysicalStream` publishes an immutable source plan followed by a sealed
one-frame native store and ready marker for each source knot. A frame binds
the actual posted and decoded evidence available at that time. Initial
preparation does not require a complete downloaded trajectory or its final
manifest. Later waits reuse the ordinary source producer's lifecycle.

`PostedPhysicalProvider` resolves the base knot and every fixed donor's exact
knot or time brackets before preparing selected original member indices.
Small immutable catalogs reuse the complete-store native operators. Serialized
provider plans can be reopened by native preparer child processes. Direct
recipes retain their exact source frame; recentered recipes use the complete
donor mean even for a one-member replay.

Native hooks verify the physical member frame against the actual grid, statics,
field contract and original source plan before each real initialization.
Source producer and member consumer reuse the original experiment and source
plan; member identity and runtime stochastic settings are separate authorities.
Deferred manifest values have explicit typed plan references. They are never
presented as completed input-manifest digests.

Prepared heads bind the provider plan, original member and initial full field
receipt. Each boundary segment binds both endpoint receipts. The final catalog
must contain every consumed knot and verified final source seals for all
trajectories. Portable source capsules replay the ordinary source seal checks
and hold consumed physical frames to their actual raw posted records.
Missing donors, changed members, changed source plans and incomplete source
seals are refused. The current native physical hooks support single domains.

`provider.source_context(original_member_index)` pins the ordinary source
head and exposes the existing interval and final-seal lifecycle. Recentered
members select the base context; direct members select their own source.
Native member preparation reuses the verified static geometry and solved
land/water fields from that context. Reuse checks all non-atmospheric physical
inputs and water metadata against the base, and keeps the original soil-repair
receipts. Each changed atmosphere still passes through native real
initialization at every forcing knot. Source markers and final authorities
are relayed from the ordinary source rather than decoded again.

`provider.set_wait_observer(writer_source_wait(writer))` attaches a member's
writer after its source contexts already exist. Source waits preserve the
upstream cause and check the member's own stop marker. Cancelling one member
does not stop the shared source or other members. A source that has arrived
but is still being prepared returns the member heartbeat to preparation.

Posted GFS consumers reuse the captured decoder authority and do not require
that executable to be installed. An explicit existing decoder still must
match the captured name and bytes. The full current input plan, including
the source cycle, posting schedule, experiment, WPS and static files, must
match the original checked source plan. Ordinary producers still require
their decoder, and member native initialization still requires its own
preparation backend.

The GFS producer, source factory and member consumer have passed a complete
archived-data run in which the member head passes forecast-reader preflight
before future source objects or the source seal exist. Early and final
preflights pass, and all 255 prepared arrays match both the source preparation
and the earlier complete-input preparation byte for byte. A guarded run
confirms that the member repeats no raw decoding, geographic construction,
horizontal mapping or soil preparation. Native initialization still runs for
each member knot. Separate native operator tests establish
streamed-versus-complete numerical identity for changed recentered members.
Other route and forecast qualification is tracked independently.

## Stochastic mechanisms

Public forecast and native-input doors recognise SPPT, SKEBS and SPP selectors and refuse active ones (exit 2) with the calibration reason until their spread amplitudes are calibrated against observations. This section describes the internal implementation; it does not make an uncalibrated run available.

`gpuwm.ensemble.stochastic` implements the vertically uniform WRF v4.6.1
spectral processes and explicit tendency hooks. It preserves the WRF reference
parameter equations, including the source's SKEBS normalization, while using
versioned counter-based Philox and cuFFT. These RNG and transform choices are
declared differences, not WRF byte identity. Binary64 shifted normalization
avoids the native coefficient underflow for correlation lengths much larger
than the domain.

`StochasticTimestepHook.before_timestep` advances once at a fixed model step.
`after_nonmicrophysics` adds SKEBS before multiplying by SPPT, in WRF order.
Its returned tendencies must be held through the RK stages. Microphysics
increments are outside this hook. Dry-mass-coupled tendencies require explicit
factors in the caller's staggered convention. The full pattern domain is the
mass grid plus one horizontal edge in each direction.

Counters bind the existing 64-bit member seed, physical stream, global spectral
index, absolute update step and rejection attempt. They do not contain a batch
position, launch shape or device ordinal. Checkpoints retain complete spectra,
configuration and completed update step. Disabled hooks return the original
tendency object and allocate no device state.

SPP patterns reach GF closure parameters, MYNN surface/PBL parameters and RUC
land parameters through explicit native consumers. The default WRF vertical
structure uses one independently seeded horizontal pattern for each selected
scheme and broadcasts it over that scheme's parameter or level axis. Optional
`spp_configs` bind an explicit amplitude configuration for every selected
scheme; partial maps are refused. Complete pattern configuration and spectra
are part of provider restart identity.

GF qualification compares the native closure equations directly: the upstream
driver reads its SPP field without passing it to those closures, and the
enabled consumer corrects that missing connection. MYNN and RUC checks cover
their native parameter sites and connected runtime behavior. Existing
default-off compiler inputs and checkpoint identities are preserved. Native
operator agreement and declared full-column numerical residuals are separate
from forecast skill. Vertical phase structures beyond WRF option 0 remain
unimplemented.

## Qualification still required

Field-transform identity, stochastic restart identity, source-member byte
verification and native WRF coefficient agreement do not establish forecast
skill. Whole-forecast N=1 and batched/single member identity must be qualified
on the connected executor and its actual prepared inputs. Calibration must
compare real forecast members
against observations, reporting effective member count, missing-data masks,
valid times, accumulation windows, sampling support and separate tuning and
evaluation cases. No amplitude should be selected from an unverified or
partially connected forecast path.
