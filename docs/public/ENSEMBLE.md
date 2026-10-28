# Regional research ensembles

`gpuwm ensemble` creates a regional, convection-permitting ensemble from a source analysis or forecast cycle. The WoFS recipe defaults to 36 members, 3 km spacing, about 300 × 300 cells, a three-hour forecast and 15-minute histories:

```sh
gpuwm ensemble --recipe wofs --source era5 --init 2021-12-10T18 \
  --center 36.6,-88.3 --members 36 --hours 3
```

`--source` accepts `era5`, `hrrr`, `gfs` and `gefs`. Supply a UTC cycle supported by the source's archive. Availability and account quotas still apply. ERA5 automatically uses the keyless ARCO reanalysis when no CDS credentials exist. With CDS credentials it uses the ten encoded EDA members; `--era5-provider arco` explicitly selects reanalysis instead. Forecast-source preparation uses the existing source-specific download and preparation chains.

An existing domain config also works:

```sh
gpuwm ensemble regional.toml --recipe wofs --members 36
gpuwm go regional.toml --recipe wofs --members 36
```

The generated config and its `.ladder.json` explain the complete roster before acquisition. `--dry-run` prints the executable plan without downloading forcing or integrating a member. Existing explicitly selected ensemble recipes remain available.

A config may request a fractional-hour duration, such as `run_seconds = 600`. The WoFS recipe preserves that end time and acquires source knots enclosing it for native temporal interpolation. Restart snapshots retain full byte-hash verification; independent files are checked concurrently within the host's CPU allocation.

Forecast-source members prepare concurrently across the host's available CPUs; exact source/physics bindings share one preparation. On each card, WoFS runs as many independent member streams together as live memory admission allows, with a separate queue for each card. An explicit `max_ordinary_members_per_device` still selects a smaller concurrency limit. Checkpointed runs use the same placement, holding diagnostic submissions while each durable restart snapshot seals. Snapshots hard-link member checkpoint sets (write-once, so a seal adds no checkpoint bytes), copy other new files without changing live output metadata, then reuse unchanged sealed copies between generations.

## Member ladder

1. Use the source's own members at the requested cycle: all 31 GEFS members (control plus 30 perturbations), ten ERA5 EDA members with CDS, or the single deterministic analysis/forecast. GFS remains GFS.
2. Use earlier covering cycles, nearest first, up to `--max-lag-hours` (24 by default). HRRR, GFS and GEFS forecasts retain the requested valid-time window, with boundary knots bracketing it. ERA5 analyses have no forecast-cycle lag rung.
3. Fill to the requested count with seeded checkpoint branches of those source states. The initialized parent is saved and restored at model time zero, before perturbation. Parents are shared within a launch only when both the source trajectory and complete physics configuration agree; physics arms keep separate parents. Each fill member receives constant U and V offsets independently drawn in ±`--wind-amplitude` m/s (default 0.5), a surface draw, and a physics arm. These are illustrative research perturbations, with **uncalibrated spread**. No calibration refusal hides the available controls.

Surface fill scales soil moisture by a seeded value in [0.9, 1.1] and offsets sea-surface temperature by a seeded value in [−0.5, 0.5] K. The existing surface-state initializer applies the selected fixed values on the GPU. The control/source/lag members retain their original physics unless `--physics-arms all` is selected. Fill members rotate through these ordinary selectors:

| Arm | Microphysics | PBL | Surface layer |
|---|---:|---:|---:|
| base | config | config | config |
| thompson-ysu | 8 | 1 | 91 |
| thompson-mynn | 8 | 5 | 5 |
| thompson-myj | 8 | 2 | 2 |
| nssl2m-ysu | 18 | 1 | 91 |
| nssl2m-mynn | 18 | 5 | 5 |
| morrison-myj | 10 | 2 | 2 |
| wsm6-ysu | 6 | 1 | 91 |

Use `--physics-arms fill|all|off`, `--no-surface-fill`, `--wind-amplitude 0`, `--max-lag-hours 0` and `--ensemble-seed N` to choose the mechanisms. A fill with every mechanism disabled would duplicate trajectories and is rejected with that concrete explanation. An authored `member-roster` config selects other supported physics and surface combinations.

## Provenance label contract

There is one label format: **canonical compact JSON** with schema `gpuwm-member-provenance.v1`. Serialize object keys in ascending order, without separator spaces, without NaN/Infinity, and without the object's own top-level `label` key. A nested `parent` may retain its original label and structured ancestry. The complete string is the `GPUWM_ENSEMBLE_PROVENANCE` global attribute in each member's wrfout. The identical string is `provenance[i].label` in the ensemble request/receipt; structured fields alongside it are the same object. Aggregate NetCDF metadata retains the ordered member labels too.

Required fields are `schema`, zero-based `member`, `rung` (`source-member`, `time-lagged`, or `perturbation`), `source`, UTC `cycle` (`YYYY-MM-DDTHH`), `lag_hours`, `seed`, and `physics_arm`. Optional fields are `source_member` (the publisher's encoded selector), `physics` (ordinary run selectors), `surface` (fixed soil-moisture scale and SST offset), and `perturbation`. Missing optional fields mean that mechanism was not applied.

For uniform winds, `perturbation` contains `kind: "uniform-wind"`, `amplitude_m_s`, `applied` (the branch/application point), and `spread` (the calibration statement). This door adds `branch: {"kind":"initialized-checkpoint","model_elapsed_seconds":0}` for fill members. The member receipt's `checkpoint_branches` records the actual parent file, SHA-256, restore time and reuse status. A later checkpoint branch preserves source/cycle/member/lag ancestry and records its actual application point; it must not relabel a time-zero branch as a later forecast checkpoint. The seed and amplitude identify the deterministic wind draw; initialization receipts also record the kernel hash, axis keys and application status.

## Programmatic perturbation entry points

`LadderMember` and `RUNG_FILL` in `gpuwm.ensemble.wofs_recipe` construct checkpoint-child provenance. Add the parent ancestry and actual branch/application description, then recompute `row["label"] = provenance_label(row)` after all changes. Labels describe execution; constructing one does not apply a perturbation.

The existing execution APIs are separate mechanisms:

| Mechanism | Entry point | Controls |
|---|---|---|
| Uniform U/V offsets | `gpuwm.ensemble.batch_perturbation.initialize_member_winds` | `state`, `cfg`, `member_indices`, `seeds`, `phase="after_physics_before_step"`, `amplitude` in m/s |
| Soil moisture and SST | `gpuwm.ensemble.surface_recipe.apply_surface_recipe` | `state`, `value={"kind":"surface-state", ...}`, `member_id`, `seed`, optional `domain_id` |
| SPPT, SKEBS, SPP | `gpuwm.ensemble.stochastic_model.StochasticModelProvider.from_mapping` | `sppt`, `skebs`, `spp`, optional `spp_configs` and `wrf_seed_labels` |

For example, `StochasticModelProvider.from_mapping({"sppt": True, "skebs": True, "spp": {"pbl": 1}})` selects reference SPPT/SKEBS parameters and an SPP PBL consumer. Explicit parameter dictionaries override reference values. SPP must match a supported selected physics scheme. `gpuwm.ensemble.stochastic_authority.configure_member_inputs(provider, inputs)` applies its declared SPP switches before driver construction and returns the selected inputs and authority receipt. `provider.bind_model(model=model, member_id=member_id, seed=seed)` attaches the full-domain runtime owners. The ordinary ensemble session uses these APIs when `EnsembleRequest.stochastic` is supplied.

These are not interchangeable one-shot callbacks: SPPT/SKEBS/SPP run at the model's stochastic timestep seams. Normal surface initialization skips advanced states, and normal stochastic resume validates the original member, seed, configuration and spectra. A warm-checkpoint launcher must explicitly own branch-time surface application and any new stochastic owner; relabelling a checkpoint does not re-key its saved stochastic state. This WoFS door's measured fill remains at model time zero.

## Products

This door enables `nwp_diagnostics = 1` by default, retains member histories and diagnostics, and generates in-run mean/spread, exceedance probability, paintball, spaghetti and postage maps. Reflectivity and total precipitation also have a probability-matched mean (PMM), with equal mean ranks sharing their matched intensities. Reductions and plots run through the parallel Rust product engine. UH uses the model's `UP_HELI_MAX` diagnostic with the ensemble product threshold 75 m² s⁻². Explicit config settings remain selectable. A rolling QPF window appears once the run has accumulated that duration; unavailable fields are listed in the receipt.

These probabilities describe this roster's members; they are not an observation-calibrated probability forecast. Four-member validation maps therefore have probabilities in quarter-member increments.

For saved analysis checkpoints and completion-driven placement across all cards, use the [checkpoint research ensemble door](MASS-ENSEMBLE.md).
