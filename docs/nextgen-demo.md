# NextGen demo table and driver contract

Docs-door owns `gpuwm/data/hydro/demos.v1.json` and
`gpuwm.hydro_demo.register_demo_cli(subparsers, *, runner=None)`.
This document describes the implemented contract. The current
`lower-colorado-tx` row is held before acquisition because the v201
GeoPackage's data terms remain unverified. There is no runnable public demo
configuration yet. Croton belongs to the withdrawn WRF-Hydro brief.

## Reference case and option coverage

Case (b) is `test/LowerColorado_TX_v4/test_AnA_V4_HYFeature_noDA.yaml`
at t-route `12a8eae0cdfed437143c590659fa7077605a5e70`. Keep this
no-assimilation configuration as the base reference. Its prescribed laterals
exercise network routing; they do not exercise catchment runoff generation.
The base case selects Muskingum-Cunge routing and level-pool reservoirs.
Hybrid diffusive routing, RFC forecasts, USGS/USACE reservoir persistence and
streamflow nudging are disabled. Captures of those other options require
actual native calls selected separately from this base configuration.

Every routing and reservoir option still needs its own captured native module
inputs and outputs and an exact replay result. An option absent from the base
case needs a separate module capture; a successful base-case run cannot qualify
it. This does not call for another whole forecast per option. Reservoir
preparation and launches belong to routing assembly; the demo registrar must
not advance the same reservoir independently. The driver owns runtime assembly
and CLI registration; the configuration owner owns realization rows and
settings bindings. Docs-door supplies acquisition metadata and the callback
below, with no numerical stage factory.

## Table fields

The top-level `schema` is exactly `gpuwm-nextgen-demos-v1`; `examples` is an
array of rows with unique `id` values. New examples are table rows.

| Row field | Type and use |
|---|---|
| `id` | String; CLI example ID, unique within the table. |
| `title` | Human-readable description; not used to select physics. |
| `admission.status` | Only `"permitted"` admits acquisition. Missing or other values refuse before cache/run directory creation or Rust asset I/O. The current row is `"held"`. |
| `admission.reason` | Human-readable reason for refusal; the current row names the unverified v201 data licence and pending public-domain replacement recipe. |
| `source` | Provenance object copied to the run receipt. Existing fields are `repository`, pinned `revision`, `licence_path`, `terms_path`, `software_licence`, and separate `data_licence`. Software terms do not clear data rights. |
| `seconds` | Default positive, finite simulation duration. `--seconds` overrides it. Simulation requires an exact multiple of `settings.interval_seconds`. |
| `settings` | Metadata copied into the generated `[hydro]` TOML table; see below. |
| `paths` | Map from settings keys to input-relative paths; the preparer emits absolute paths beneath the run's `inputs/` directory. |
| `assets` | Nonempty array of pinned input objects, described below. |

The table reader checks its schema, unique IDs and safe distinct asset paths;
it is not a general JSON Schema validator. Admission records a reviewed source
and its actual terms. Changing the status string does not grant rights or
bind missing physics. For the permanent public demo, the fabric owner must
supply the independently permitted source/member notices and preparation
recipe before the row can be admitted. Restricted research data is not a
public-demo source.

Each asset has `path` (distinct safe relative POSIX path), `url`, integer
`bytes`, and lowercase SHA-256 `sha256`. An upstream `git_blob` is retained as
provenance metadata; the transfer verifies byte count and SHA-256. The pinned
LICENSE and TERMS files are assets too. The preparer rejects absolute paths,
parent traversal, backslashes and colon-bearing paths. Rust performs transfer,
hashing and copying, with up to eight concurrent asset workers; Python handles
metadata and subprocess orchestration.

## Generated settings and runtime boundary

The current held row emits these settings:

| `[hydro]` key | Source or current value |
|---|---|
| `enabled` | `true`. |
| `realization` | `"lower-colorado-hyfeatures"`; expanded through the config owner's realization table. |
| `interval_seconds` | `300`. |
| `start_time` | `"2023-04-02T00:00:00Z"`. |
| `numerics` | `"fast"` by default; `"strict"` only with `--strict-numerics`. |
| `case_directory` | Absolute materialized example directory. |
| `routing_config` | Absolute materialized upstream no-assimilation YAML path. |
| `hydrofabric` | Absolute materialized GeoPackage path. |
| `forcing_directory` | Absolute materialized supplied lateral-inflow directory. |

`settings` is copied first; the preparer then sets `numerics` and expands
`paths`. Nested settings mappings are written as TOML tables, and ordered
stage mappings as TOML arrays of tables. JSON object syntax after `key =`
is not TOML. The current `paths` map only fills top-level input identities;
it does not prepare a routing resource or populate nested bindings.
Supplied lateral inflows exercise routing, not all runoff modules.
These input identities are insufficient to bind a working GPU realization.
The config, fabric, routing, forcing and output owners supply prepared stage
factories and resources. In particular, captured native routing options such
as `assume_short_ts` belong to the routing-resource contract; the demo does not
invent an override or infer their values.

The runtime configuration table is `configurations.v1.json`, independent of
this acquisition table. Its current Lower Colorado row requires
`settings_bindings.fabric_manifest` for already prepared network resources.
The held demo row does not supply that binding or produce such a manifest.
See the [configuration contract](nextgen-configurations.md); permitting data
acquisition alone would not complete the runtime assembly.
The runtime epoch key is `time_origin`; the held acquisition row's
`start_time` is descriptive metadata, not an override of the row's epoch.

On a permitted row the preparer returns `(config_path, run_directory, receipt)`
from `prepare(name, *, outdir, data_dir, strict_numerics=False)`. The root
registrar binds `runner=run_prepared`; simulation calls:

```python
runner(settings=config_path, seconds=seconds,
       outdir=run_directory / "output", device=device)
```

The runner returns its runtime receipt. The demo does not pass a restart to
this callback; restart is available through `hydro run`. Simulation duration
and a nonnegative device ordinal are checked before acquisition. Without a
bound runner the direct module command refuses simulation before acquisition;
`--prepare-only` only prepares inputs and never advances water.

The public orchestration helper is
`gpuwm.hydro.cli.run_prepared(*, settings, seconds, outdir, device=0, restart=None)`.
Its arguments are keyword-only. `settings` names a TOML file whose `[hydro]`
table supplies the settings; `outdir` identifies the selected writer's output
directory. `seconds`
must be a positive whole number of hydro intervals. The helper reads and
resolves the settings, selects fast numerics unless strict was requested,
prepares the driver and writer on the selected card, optionally restores
`restart`, advances the requested intervals and returns the runtime receipt.
The requested duration is additional simulation time after restoration.
Checkpoint compatibility and complete restore belong to the selected writer
and components; the helper signature alone does not prove them. Output and
checkpoint I/O occur outside the timed launch loop. This is the same helper
used by `hydro run` and the registered demo callback; docs-door does not supply
a second execution path.

## Rust asset bridge in a source checkout

Input preparation uses the existing `rw_fetch` bridge through
`gpuwm.geog_assets`. `GPUWM_RW_FETCH` selects an explicit executable. The
Python launcher checks its executable format, version and exact `--abi`
record before issuing asset work. The required ABI includes `asset` and
`hash`, along with the other declared fetch commands; a version number alone
does not establish compatibility.

For a Linux source checkout, build only the fetch target from the checked-in
vendor dependencies. Set `HYDRO_BUILD_ROOT` to your scratch directory and
run these commands from the repository root:

```sh
export HYDRO_BUILD_ROOT=/path/to/scratch/hydro-build
mkdir -p "$HYDRO_BUILD_ROOT/tmp" "$HYDRO_BUILD_ROOT/cargo-home"
cd tools/rustwx
CARGO_HOME="$HYDRO_BUILD_ROOT/cargo-home" CARGO_TARGET_DIR="$HYDRO_BUILD_ROOT/target" TMPDIR="$HYDRO_BUILD_ROOT/tmp" cargo build --release --locked --offline -j "$(nproc)" -p rw-fetch --bin rw_fetch
export GPUWM_RW_FETCH="$HYDRO_BUILD_ROOT/target/release/rw_fetch"
cd ../..
```

The bridge verifies expected byte count and SHA-256 for downloaded assets
and independent local copies. HTTP byte ranges and pins refer to the
identity representation; the transport requests `Accept-Encoding: identity`.
Keep the actual build command, source revision, binary digest, ABI result
and timed transfer/hash receipts when recording preparation evidence.

A working bridge clears the transport dependency. The current public demo
still refuses before bridge discovery because its dataset admission is held.
Verifying a permitted software notice through the bridge is transport
evidence; it does not prepare the dataset or advance the model.

## Run receipt and CLI

An admitted preparation creates an isolated run directory containing `inputs/`,
`hydro.toml` and `demo.json`. `demo.json` uses schema
`gpuwm-nextgen-demo-v1` and records `example`, `source`, `run_directory`,
`strict_numerics`, `status`, measured `stages`, and per-asset `path`, `bytes`,
`sha256`, `cache_reused`, and `seconds`. Successful preparation adds `settings`;
CLI execution adds `requested_seconds`. Simulation adds `runtime` and
`runtime_wall_seconds`; failures record `error` after a run directory exists.
A pre-acquisition refusal creates no run receipt. `prepared` means inputs
were materialized; `completed` requires the runner to return successfully.
Neither status alone establishes replay, physical skill or routing speed.

The driver CLI contract is:

```text
gpuwm hydro run CONFIG --seconds SECONDS --outdir DIR [--device N] [--restart FILE]
gpuwm go CONFIG --hydro [SETTINGS.toml]
```

These are syntax templates, not a runnable demo. `go --hydro` without a path
uses the forecast's `[hydro]` settings. In the current runtime, that table with
`enabled = true` also enables coupling when the flag is omitted. The parent
land-surface cadence must match the hydro interval. The demo registrar adds
`hydro demo EXAMPLE` with `--prepare-only`, `--outdir`, `--data-dir`, `--seconds`,
`--device`, and `--strict-numerics`; defaults and current availability are in
[the public guide](public/HYDRO.md).
