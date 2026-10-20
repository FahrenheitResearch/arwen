# Default-on hydro door

Drew, 2026-10-18 08:40Z: hydro is default on for every ArWen forecast and every
ensemble member whenever a hydrofabric covers the domain; `--hydro off` is the
opt-out; where no fabric exists hydro is off and the receipt names the gap.

## What runs

`gpuwm go` and `gpuwm ensemble` (the same entry, `gpuwm.hydro.runtime.go_scope`)
call `gpuwm.hydro.door.resolve_launch` before preparation. A runner started
without a launcher (`forecast_scope`) makes the same decision once.

| Case | Decision | Receipt `hydro_activation` |
|---|---|---|
| `--hydro off` | off, the table and the domain are never read | `reason: cli_off`, every member off |
| `--hydro FILE` or a `[hydro]` table naming `realization`, `stages` or `driver_factory` | on with those settings (unchanged behaviour) | `door.fabric: explicit settings` |
| `[hydro] enabled = false` | off | `reason: configuration_off` |
| nothing prepared (the default) | the door reads the parent grid and tries the provider rows | `fabric_covered` with the provider, or the named gap |
| bare `--hydro` and no covering fabric | refused before any work, naming every gap | (raises `HydroDoorRefusal`) |

Provider rows live in `gpuwm/data/hydro/fabric-providers.v1.json` (extra rows from
`$GPUWM_HYDRO_FABRIC_PROVIDERS`). A row is used only when all of these hold:

1. its coverage (WGS84 boxes, dateline-aware, or `global`) holds at least
   `min_covered_fraction` of a 33x33 lattice over the parent grid;
2. its data licence status is `verified`;
3. it names a coupled realization that `configurations.v1.json` admits;
4. the parent LSM is one coupled NextGen consumes (Noah, RUC or Noah-MP,
   `sf_surface_physics=2`, `3` or `4`);
5. its own tool answers the probe: the builder executable exists, has the
   `build` subcommand, `rw_hydro_world admit TABLE ROW` admits its source row, and
   its tile root exists (or, for a `prepared` row, its manifest exists).

The first such row wins. The door then runs the builder
(`rw_hydro_world build --sources TABLE --row ROW --bbox W,S,E,N
--data-root DIR --out FABRIC`), runs `rw_hydrofabric prepare`
on the result, caches it under `$GPUWM_HYDRO_FABRIC_CACHE` (default
`<experiment dir>/.gpuwm-hydro/<row>/<grid key>/`), and writes
`hydro-door.toml` binding the prepared manifest into the realization. A rerun on
the same grid and source root reuses the build. Each builder `paths` entry can
name an `argument` (for example `data_root`): the door resolves its environment
variable or default, checks the directory, and substitutes that value in the
row's argv template. Resolved paths enter the cache request, so changing the
source root rebuilds the fabric. No provider-specific adapter is needed.
The decision travels to every runner and member
process in `GPUWM_HYDRO_DOOR_DECISION_JSON`, never inside the settings, so the
sealed realization identity in checkpoints does not depend on how the run was
launched.

One hydro state per member is the coupling lane's resident driver: each member's
forecast prepares its own driver on its own card from its own settings copy, and
the driver's state is part of the atmosphere checkpoint.

## Cost and products in every coupled receipt

- `hydro_cost` (`gpuwm.hydro.cost`): CUDA events recorded around the driver's
  `before_lsm` and `advance` hooks on the hydro stream (the LSM between them is
  excluded), harvested with non-blocking queries; no device sync in the step
  loop. Reports ms per LSM call, ms per atmosphere step, the fraction of the
  atmosphere step, and each stage's parallelism. Above 5 percent it is the named
  defect `hydro-cost-over-budget` (owner astra-hydro-speed): the receipt and the
  forecast outputs are written first, then the run is refused with a non-zero exit.
- `hydro_products` (`gpuwm.hydro.forecast_products`): at the output boundary only,
  the realization's GPU history goes through the Rust writer, its renderer
  request gets the gauges from the fabric's `gauges.json` that sit on the routed
  roster, and the Rust renderer draws hydrographs and streamflow into
  `<forecast out>/hydro-products/d01-<dx>/<product>/<valid-day>/`. Every missing
  piece (no history stage, no renderer build, no matched gauge) is a named gap in
  the receipt and on stderr.

## Today

Every real domain is off with named gaps, because no provider row can pass yet:

| Domain | Decision | First gap (owner) |
|---|---|---|
| CONUS, Noah / RUC / Noah-MP | off, `no_covering_fabric` | complete source recipe, initialized same-network parameters/stores and admitted coupled realization (c-hydro-data, c-hydro-runoff, c-hydro-coupling) |
| Rhine, Noah / RUC / Noah-MP | off, `no_covering_fabric` | covering Copernicus source tiles and admitted physical coupled realization (hydro-world-fabric, c-hydro-coupling) |
| Fiji across the dateline, Noah / RUC / Noah-MP | off, `no_covering_fabric` | same as Rhine |

The global default row uses the Copernicus-derived network. Per the coordinator's
2026-10-18 11:50Z ruling, HydroSHEDS v1 is research / cross-check only: never
bundled, fetched by default, or used as the default source. Its limited WWF
licence remains a named refusal without an explicit research opt-in. The Rhine
run on that fabric is a pipeline proof, not a shippable fabric or a skill result.

The Rust global builder is merged. Physical source and realization admission
remain separate; a verified source licence alone never enables a forecast.

## Commands

```
# decisions and the coverage chart for any experiment files (CPU)
python -m tools.hydro_world_door.decide EXP.toml [...] --png coverage.png
# tests (CPU; the CUDA file skips without cupy)
python -m pytest -q tests/test_hydro_world_door.py tests/test_hydro_world_door_cuda.py
# on a card (an RTX 5070 Ti measured it) under the hydro pool's slot runner
env MPAS_UNIT=hydro-world-door python3 -B RUN/source/tools/hydro_world_door/node4_run.py RUN
```

Receipts: `receipts/door-decisions.json`, `receipts/door-coverage.png`,
`receipts/cpu-tests.json`, `receipts/node4-cost-meter.json`,
`receipts/case-b-coupled-run-blocker.json`.
