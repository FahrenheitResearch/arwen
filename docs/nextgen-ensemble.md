# NextGen ensemble output contract

This is the ensemble owner's implementation contract at
`1af2f96c98a2413a2fcd42eea33a2b619a00eb44`, including the float32 policy fix
`39c141f676db2a9fdf3750fae8a71a3a87bb791a`. Integration of these optional
collectors is required before using this contract in a build. This docs branch
does not contain them. The mass-launcher hook is a reviewable patch in
`tools/nextgen_mass_launcher.patch`; application to that launcher and actual
weather, WoFS, hurricane and multicard acceptance remain pending.

## Configuration and member state

In builds containing the collectors, an enabled `[hydro]` table for every
requested member automatically selects reach discharge probabilities through
the existing ensemble forecast door. Optional output policy is:

```toml
[ensemble.hydro]
thresholds_m3_s = [1, 10, 100]
device_budget_bytes = 268435456
```

These are the defaults. `device_budget_bytes` bounds the point-product CUDA
pool on each card; it is not a budget for the complete atmospheric and hydro
member state. Unknown keys are rejected. Thresholds must be distinct finite
nonnegative numbers; the budget must be a positive integer. Thresholds that
overflow or become identical after float32 rounding are rejected at setup.

Each member's own `[hydro]` table overrides inherited launcher settings.
The mass hook snapshots selected parent settings before workers start and
passes independent copies to members. A mixed enabled/disabled roster, or
explicit products without hydro enabled for every member, is rejected before
forecast work. An enabled member cannot silently become atmosphere-only when
its runtime is absent.

Each member owns its hydro state on its own card. Hydro remains on fixed root
grid 1; moving child nests do not add routing domains. The implementation
uses the original member forecast runner because packed atmospheric and
component banks do not carry this hydro state. Complete-state memory
admission and full atmospheric restart qualification remain pending.

## Driver identity and checkpoint restoration

Driver commit `d2f662947b467362a9acb1d2f01fca4746c7cc76` adds the original
ensemble member ID to hydro checkpoint identity. This newer driver change is
separate from the collector versions above and is not imported by this docs
change. It reads the existing member output scope during parent initialization;
there is no additional runtime flag or configuration table.

Each prepared driver keeps that member identity for its lifetime. A factory
cannot return another member's cached driver. A scoped checkpoint must match
the destination member; a checkpoint without the required member identity
cannot be restored into a scoped member. An ordinary unscoped run retains its
previous identity without adding a null member key.

The whole-atmosphere reader validates hydro identity before writing destination
arrays. A valid restore copies into the existing allocations. Prepared parent
geometry is rebuilt from the executed inputs; numerical hydro carriers remain
in the `hydro/` checkpoint namespace.

The owner's pooled RTX 4090 test used initialized parent physics and an
instrumented hydro stage: a member 7 checkpoint was refused by member 22 with
both atmospheric and hydro bytes unchanged. Restoring into member 7 preserved
device pointers and all 12 instrumented hydro bytes. Two targeted checks passed.
This qualifies the member-binding and restoration boundary, rather than native
hydrologic replay, a complete Noah/CFE restart or an actual weather ensemble.
Initialized standalone Noah/PET/CFE continuation has a [separate measured
scope](nextgen-output.md#initialized-standalone-continuation); it does not
establish atmosphere or ensemble continuation.

The ensemble owner retains the output-deadline callback. Driver binding does
not add another capture callback or host data processing to routing steps.

## Reach discharge probabilities

At output deadlines the producer supplies member ID, UTC, ordered int64 reach
IDs, native float32 or float64 discharge in m³/s, device and stream. CUDA
maintains integer counts, using one thread per reach and 256 threads per
block with event-ordered member submissions. Rust writes NetCDF at output
boundaries. This adds no product callback or host numerical work per routing
step.

The schema is `arwen.nextgen.reach-ensemble/v2`. Each requested threshold is
rounded once in CUDA to float32, round to nearest, ties to even. That exact
float32 value is both published and compared against native discharge using
`Q >= threshold`; float64 discharge is not narrowed. The fraction is
`exceedance_count / requested_member_count`, rounded to float32.

All members must supply every published deadline with identical reach IDs
and order. A missing member refuses finalization. If any requested member's
discharge is nonfinite at a reach, that reach's probability is NaN. Finite dry
members remain in the full requested-roster denominator. Counts retain the
finite contributions even when the probability is masked. Merging member
shards sums integer counts rather than averaging probabilities.

For ensemble root `OUT`, files are:

```text
OUT/nextgen/reach-exceedance-00000.nc
OUT/nextgen/reach-ensemble.json
```

NetCDF uses the Rust-backed CDF5 writer. With `R` reaches and `T` thresholds:

| NetCDF variable | Type and shape | Meaning |
|---|---|---|
| `feature_id` | int64 `[R]` | Exact ordered reach ID. |
| `feature_id_words` | uint32 `[R,2]` | Low/high words of the same two's-complement int64 ID. |
| `discharge_threshold` | float32 `[T]` | Actual comparison threshold, m³/s. |
| `exceedance_count` | uint32 `[T,R]` | Finite members satisfying `>=`. |
| `finite_count` | uint32 `[R]` | Members with finite discharge. |
| `probability` | float32 `[T,R]` | Fraction 0–1, or NaN under the complete-roster mask. |

The JSON manifest names the threshold buffer `threshold_m3_s`, pointing to
NetCDF variable `discharge_threshold`. Each frame carries UTC, requested
member IDs, denominator, `comparison = "ge"`, `missing_policy =
"complete_roster"`, paths, units, shapes and little-endian buffer dtypes.
Reach file paths are relative to this manifest's directory. Use the word
pair when a transport decodes numeric values through float64; that transport
cannot represent every int64 ID exactly.

Checkpoints retain integer counts, finite counts, frame/member cohorts and
exact IDs. Restore checks schema, roster and options. The previous v1
float64 threshold policy cannot be mixed with or restored into v2. Collector
continuation evidence does not establish complete atmosphere/Noah/CFE restart.

## Channel depth and HAND

Above-bed channel depth is archived separately, retaining native float32 or
float64 values and exact IDs:

```text
OUT/nextgen/channel-depth/...
OUT/nextgen/channel-depth-ensemble.json
```

Its schema is `arwen.nextgen.channel-depth-ensemble/v1`, units `m`,
`depth_reference = "local_channel_bed"` and `comparison = "gt"`. Each frame
lists the full requested roster; unavailable member sources remain null with
`missing_member_action = "mask_all_cells"`. Reach axes may differ by member,
so an explicit join is required. The manifest declares `root = ".."` for
source paths relative to the ensemble root.

This archive is transport for a renderer request. It does not supply terrain,
HAND, a reach-to-cell crosswalk or a dynamic 2D inundation solution. Discharge
cannot substitute for above-bed depth, and absolute reservoir elevation
cannot substitute for channel depth. See the [products request contract](nextgen-products.md)
for strict depth exceedance, complete-roster masks and separate dynamic/HAND
products. A missing depth source must not reduce the denominator.

The owner has qualified synthetic GPU capture, integer reduction, Rust output
and routing-state/product continuation. Those results do not qualify an
actual weather ensemble or demonstrate ensemble spread or flood-event skill.
The [development evidence](nextgen-development-evidence.md) records separately
scoped native comparisons and hydro-step timings.
