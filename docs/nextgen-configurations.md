# NextGen configuration metadata

`gpuwm.hydro.config` resolves composition metadata before device preparation.
It reads `gpuwm/data/hydro/configurations.v1.json`, whose schema is
`gpuwm-nextgen-realizations-v1` and whose `realizations` array contains named
rows. New compositions use rows rather than case-specific resolver code.
This table is separate from the demo acquisition table, `demos.v1.json`.

## Available rows

| Row ID | Recorded scope |
|---|---|
| `ngen-baseline-noah-pet-cfe` | Pinned upstream baseline configuration; its runtime stage list is empty. |
| `ngen-baseline-noah-pet-smp-sft-cfe` | Pinned baseline including soil moisture profiles and freeze/thaw; its runtime stage list is empty. |
| `lower-colorado-hyfeatures` | Routing with prescribed lateral inflow and level-pool reservoirs; resource paths require explicit bindings. Dataset terms remain unresolved and the public demo is held. |
| `ngen-shipped-pet-cfe-48h` | Referee's adapted SLoTH/PET/CFE reference with Noah omitted; its runtime stage list is empty. |
| `ngen-shipped-noah-pet-cfe-48h` | Initialized three-catchment Noah/PET/CFE composition with 48 hourly diagnostic samples; requires five prepared-resource bindings. No channel or terrain routing. |

The baseline rows do not identify an operational NWM v4 calibration. Their
Russian River event metadata records observation/archive references, original
federal topology, source flowlines and partial catchment geometry. The topology
uses NHDPlus ComIDs rather than HYFeatures flowpath IDs. Source flowlines
cover all 1,445 selected reaches, preserving 32,270 source XYZM points and
direction. Catchment polygons cover 1,433 of 1,445 selected IDs; 12 gaps remain
explicit. The source-polyline and source-polygon manifests are distinct from
the resident network preparation format. Source lines do not supply hydraulic
roughness, cross sections, a verified vertical datum or divergence partition.
Complete geometry/parent overlap weights, model and hydraulic parameters,
forcing, warmup/state, reservoir controls and the selected native event
realization remain incomplete.

The row preserves a dated, immutable referee prerequisite snapshot and its
predecessor seal. The current snapshot acknowledges complete source lines
while retaining explicit missing hydraulic inputs. Later source availability
is recorded separately. Geometry byte comparisons establish the source
extraction, not a runnable hydrological event or a permitted Lower Colorado
demo dataset.

Configuration owner `6ddb681c709e48909c2d16b0663d849d4ab4c590` also
records a bounded NWM v3.1.6 RouteLink source candidate for Russian River.
All 1,445 requested original ComIDs match; an independent NetCDF-C row read
agrees on 140,165 bytes across 22 fields. This is a source extraction result.
The candidate remains private and unadmitted: its processing notes cite
Lynker Reference Fabric 3.0, whose underlying data terms remain unresolved.
The table records this hold separately from its permission to resolve
configuration metadata. It provides no public/default/demo acquisition route.

The subset has not been mapped to resident routing inputs. Native `Length`
has not become `length_m`; `Kchan` is an int16 channel coefficient in
millimetres per hour, not CFE soil conductivity. Static `Qi` does not supply
event warmup, and source start-node `alt` is not a hydraulic cross section.
There is no HYFeatures crosswalk or implicit NWM v3.0 coefficient identity.
Twelve source catchment polygons, parent overlap weights, selected model and
reservoir parameters, event forcing and initial state remain missing. No new
runtime stages or runnable case (c) follow from these source coefficients.
The `nwm-v3-conus` archive source belongs to retrospective analysis; it is
not a realization selector. No `nwm-v3` runtime row is provided here.

Every current row's admission permits configuration metadata only. A
successful resolution does not authorize acquisition of a referenced dataset,
load a native library, prepare GPU components or demonstrate a simulation.
The separate public demo admission remains held before acquisition.

## Resolver API

This metadata-only example needs no CUDA device or dataset:

```python
from gpuwm.hydro.config import configure, resolve_configuration

selected = resolve_configuration("ngen-baseline-noah-pet-cfe")
row = selected.as_dict()
settings = configure({"realization": selected.name})
assert settings["numerics"] == "fast"
assert settings["stages"] == []  # descriptive baseline, not runnable assembly
proof_settings = configure({"realization": selected.name, "numerics": "strict"})
assert proof_settings["numerics"] == "strict"
```

`resolve_configuration(name, *, table=None)` accepts the default table, a
table path, or a mapping. It rejects duplicate IDs, unknown IDs and rows whose
`admission.status` is not `"permitted"`. It returns a `Realization` with
`name`, an immutable `row`, and the canonical row's SHA-256. `as_dict()`
returns an independent mutable copy.

`configure(settings, *, table=None)` expands `settings["realization"]`.
Explicit settings recursively override mapping defaults. Lists, including
ordered stage lists, replace the entire default list. Input metadata is not
modified. `numerics` defaults to `"fast"`; explicitly supply `"strict"` for
strict mode. The runtime selects that process-wide mode before GPU preparation;
the resolver itself does not select a compiler or read a strict-mode flag.
The accepted values are exactly `"fast"` and `"strict"`; other values,
including booleans, are rejected.

The result includes `configuration_reference` and `realization_identity`.
The identity records the row and resolved-settings SHA-256 values, source
pins and admission. Reconfiguring unchanged expanded settings preserves the
identity. These are metadata identities, not numerical replay receipts.

## Binding resource paths

Rows may declare `settings_parameters`; callers supply exactly those names
as strings in `settings_bindings`. For example, the Lower Colorado row
declares `fabric_manifest`. Its value identifies an already prepared Rust
network manifest; the config resolver does not create or fetch one. The
manifest supplies native routing mode, initial state, lateral inflow and
output ID mapping. Configuration rows do not substitute guessed defaults.

`${parameter}` substitution runs once in runtime default fields after explicit
overrides. Binding values are literal, even if they contain `${...}` text.
Missing or extra names fail. The identity also records the bindings hash.
Use fresh input settings when changing bindings: expanded stage paths cannot
be distinguished from explicit overrides, so rebinding an expanded result
with changed bindings is rejected to prevent stale resource paths.

Native reference paths have a separate namespace: `reference_parameters`
and `reference_bindings`. `materialize_reference(name, bindings, *, table=None)`
returns only the bound reference metadata. Native `{{id}}` catchment tokens
remain intact. Native reference documents are provenance for development
tools; they do not become the product's runtime.

## Loading prepared routing resources

Driver `b17171776e139fccc7bbc09df5c7a782ad803ac9` integrates the hydrofabric
reader from owner `a05f4e303c148277a12735f3f2b298a0fd6c1b3e`.
`gpuwm.hydrofabric:prepare_resource` is a prepare-only stage: it loads the
manifest's typed payloads once onto the selected card. Routing owns subsequent
reservoir launches; the resource has no independent advance.

The production reader uses the `nextgen_hydrofabric` Rust library, ABI 1,
resolved through the bridge artifact paths or `GPUWM_HYDROFABRIC_LIBRARY`.
It does not require the offline replay tool. Rust verifies each payload's
length and declared SHA-256 before filling caller-owned transport memory;
Python binds metadata and transfers the verified bytes to CUDA.

The manifest must declare `auxiliary["network.json"].sha256`. Rust verifies
that roster before JSON parsing or CuPy import, and the loaded component's
restart identity includes `network_sha256`. Older manifests without the
roster hash require fresh preparation. Preserve identity equality when
resuming; do not remove this field to reuse an older checkpoint. The hashes
bind prepared content, not its data licence or physical suitability.

The owner's integration check initialized MC, 19 level-pool reservoirs and
history from 68 verified payloads with 148 registered arrays. It executed
zero routing steps. This establishes reader/assembler startup, not native
physics replay, a permitted public demo or complete hydro-step speed. Actual
parent geometry, physical parameters and source partition remain necessary
for a coupled realization.

## TOML settings files

`gpuwm hydro run CONFIG` reads the file's `[hydro]` table. Include
`enabled = true` for direct preparation or forecast coupling; standalone
`hydro run` enables it. Unrelated top-level tables are ignored by the hydro
reader. Module-specific keys are preserved and validated by their owners.

For `hydro run` and `go --hydro`, set `numerics = "strict"` inside `[hydro]`
for an explicit proof run; omit it for the fast default. The demo's
`--strict-numerics` option writes this same setting into its generated TOML.
It is not an option on `hydro run`.

The runtime selects the process compiler mode before importing GPU components.
Fast mode removes a kernel's `--fmad=false` option; strict mode preserves it.
Neither selection adds `--use_fast_math`. Check the run receipt's `numerics.mode`
for the effective mode. A resolved TOML setting or a compiler-policy check alone
does not establish numerical agreement with the native referee.

| Key | Meaning |
|---|---|
| `realization` | Row ID; supplies defaults before device preparation. |
| `numerics` | `"fast"` by default; `"strict"` explicitly selects proof mode. |
| `interval_seconds` | Positive cadence, normally inherited from the row; must match the parent LSM cadence when coupled. |
| `time_origin` | UTC epoch inherited by executable rows; prepared forcing and state must agree with it. |
| `driver_factory` | Defaults to `"gpuwm.hydro.assembly:prepare"`. |
| `output_factory` | Defaults to `"gpuwm.hydro.output:prepare"`; selected rows may override it. Writer-specific restart support needs separate evidence. |
| `[hydro.settings_bindings]` | Exactly the selected row's runtime resource names, with string values. |
| `[hydro.reference_bindings]` | Separate native reference metadata bindings, when requested. |
| `[[hydro.stages]]` | Optional complete replacement of the ordered stage list; module options belong under `[hydro.stages.options]`. |

For example, this is a settings template for your already prepared routing
resources, not a public acquisition recipe or a completed simulation:

```toml
[hydro]
enabled = true
realization = "lower-colorado-hyfeatures"

[hydro.settings_bindings]
fabric_manifest = "/prepared/routing/manifest.json"
```

The row inherits a 300-second interval and `2023-04-02T00:00:00Z` origin.
`fabric_manifest` must name the Rust-prepared resident network manifest.
A GeoPackage, routing YAML, source-polygon manifest or directory cannot
substitute for it. Path bindings are literal strings: the resolver neither
expands environment variables nor anchors paths to the TOML directory.
Use absolute prepared-resource paths. Generated `realization_identity` and
`configuration_reference` belong to resolver output, not user-authored keys.

## Initialized Noah/PET/CFE row

`ngen-shipped-noah-pet-cfe-48h` declares these five exact string bindings:

| Binding | Prepared resource |
|---|---|
| `forcing_manifest` | Rust-prepared forcing packet manifest for the ordered catchments. |
| `forcing_library` | Product forcing bridge shared library used by both forcing precisions. |
| `noah_initial_manifest` | Complete labelled Noah initial-state manifest from the Rust initializer. |
| `cfe_initial_manifest` | CFE initialized-state manifest. |
| `cfe_payload_root` | Directory containing the CFE manifest's state buffers. |

The row orders `forcing64`, `forcing32`, `pet`, `noah`, `cfe`, then `history`.
Its interval is 3,600 seconds, origin is `2015-12-01T00:00:00Z`, and catchment
order is `cat-67`, `cat-27`, `cat-52`. PET consumes float64 forcing with the
recorded per-catchment methods; Noah consumes float32 forcing. CFE consumes
Noah `water.qinsur` and PET in metres per second. Diagnostic history has
capacity for 48 updates and records each field's units: CFE runoff is metres
per step, storage is metres, and Noah/PET rates are metres per second.
These catchment quantities are not reach discharge in cubic metres per second.
The [forcing contract](nextgen-forcing.md) describes packet units, time
alignment and source-grid wind orientation; declaring a source row does not
prepare missing meteorological inputs or parent geometry.

This is the small initialized framework example, separate from the Lower
Colorado routing row and the Russian River event selection. It contains no
reach, reservoir or terrain stages. Fast remains the default; explicit strict
mode selects the mode used for the recorded native comparison. A strict
comparison or a speed measurement does not establish default-fast bitwise
agreement. Row resolution does not install component factories or initialize
their inputs. Its diagnostic output is not a demonstrated full-state restart.

## Runtime stages and evidence

The driver expands the selected row before device setup. Its assembly reads
ordered `stages`, each with `name`, an internal `gpuwm.module:factory`,
`options`, and optional `phase`, `launch_method` and `parallelism`.
`phase = "prepare"` creates shared checkpointed resources without a launch.
`"before"` and `"after"` launch around the parent land-surface exchange;
the default is `"after"`. Factories receive previously prepared components.
A reservoir advanced by routing hooks uses a preparation row to avoid a
second independent advance. Empty stage lists cannot prepare a simulation.

The current Lower Colorado metadata orders network preparation, reservoir
preparation and routing. It exercises prescribed lateral inflow, not runoff
generation. The public demo still needs a permitted replacement fabric,
matching prepared resources and a reviewed complete run. Input paths alone
do not fulfill those dependencies. See the [demo contract](nextgen-demo.md)
and [public guide](public/HYDRO.md).

`tools/nextgen/check_realization.py` compares source configuration hashes and
parsed metadata against the pinned reference documents, with optional
reference bindings. A pass means configuration metadata agrees; native BMI
replay, observation skill, restart/resume and GPU stage timing require their
own actual inputs and receipts. The descriptor fields in a row's
`supported_options` record component contracts; they do not assemble stages
or establish that every listed component is available in an installed build.

The retained ngen and t-route Apache-2.0 licences and NOAA terms are in
`gpuwm/data/hydro/nextgen-config/`. Software notices do not grant dataset
rights. Restricted research hydrofabric is excluded from public demo/default
downloads and this document provides no acquisition route for it.

TOPMODEL is excluded by the source-level Lancaster University commercial-use
restriction at `89e4dc529dfdf7f23b7365530973e9f238fb740d`; there is no
TOPMODEL factory, default or automatic replacement. The native reference
metadata contains `Qb_topmodel` and `Qv_topmodel` names inside SLoTH's
`model_params`, each set to zero. These are literal reference-provider values,
not a TOPMODEL implementation or a runtime selection. Preserve their provenance
when checking the pinned reference; do not interpret their names as available
physics. SoilFreezeThaw and SoilMoistureProfiles have independent notices and
remain separate from this exclusion.
