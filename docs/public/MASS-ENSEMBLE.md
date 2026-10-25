# Checkpoint research ensembles

Start a whole research ensemble from saved analysis/forecast checkpoints in one command:

```sh
gpuwm ensemble --members 1000 --branch-from parents.json --cards all \
  --branch-hours 1 --outdir research-ensemble
```

The [any-init ensemble door](ENSEMBLE.md) creates source and physics diversity from an analysis in one command. This checkpoint door keeps each parent's grid, physics, forcing and developed model state, and branches it after restore with independently seeded uniform U/V offsets. The offsets default to ±0.5 m/s; `--wind-amplitude` sets their amplitude and `--ensemble-seed` sets the reproducible seed root. `--branch-surface` additionally applies the WoFS soil-moisture and SST draws to the restored surface. Different checkpoint parents can supply different physics arms; a checkpoint is never restored into another physics implementation. Spread is uncalibrated.

`parents.json` is a source-neutral list; relative paths resolve beside it:

```json
{
  "parents": [
    {
      "parent_id": "control",
      "checkpoint": "control/gpuwmrst_d01_2022-06-13_23_00_00.npz",
      "wrfinput": "control/run",
      "provenance": {
        "source": "gefs",
        "cycle": "2022-06-13T18",
        "source_member": "c00",
        "lag_hours": 0,
        "physics_arm": "base"
      }
    }
  ]
}
```

`wrfinput` contains the original `namelist.input`, `wrfinput_dNN` and `wrfbdy_d01` that prepared the checkpoint. A native prepared-tree parent instead supplies `prepared_root`, `experiment_config`, `experiment_config_sha256`, and either `preparation_receipt_sha256` or `prepared_head_sha256`. The forcing must cover the branch window. Checkpoint identity, physics, complete-set manifest and forcing checks run for every member. Existing checkpoint-format and producer compatibility rules apply.

Each parent can supply a positive `weight` (default equal). Allocation includes every parent when the child count permits, then fills families according to their weights. Family members divide their parent's probability mass: 100 children from 36 equal parents gives 28 families of three and eight of two, with child weights 1/108 and 1/72. A smaller run explicitly conditions its probabilities on the represented parents. Families are interleaved across member IDs; seeds do not depend on placement or completion order. `--cards all` uses every inherited visible card. `--cards 0,1` chooses ordinals within that visibility; GPU UUIDs also work. An empty or duplicate selection is refused because it cannot own distinct workers. `--dry-run` prints the complete roster and its provenance without opening a GPU.

Each selected card runs `--members-per-card` long-lived worker processes, one active forecast each. The default, `auto`, fits as many members as the card's measured free memory holds at the members' priced reservation (with a 2 GiB margin), up to 4 per card, and never more than the card's share of the roster or half the host cores per card; an unpriceable member runs one per card. The co-resident processes share the card through a private MPS controller this launch starts and stops itself (`--member-sharing auto|mps|contexts`); only this launch's workers name its pipe directory, so other jobs on the card keep ordinary contexts. Without MPS, ordinary contexts time-slice the card and measured slower than one member, so `auto` then keeps one member per card. Each worker's queue starts the next member immediately on completion; an idle card steals pending work from another queue. There are no cross-card waves. Members keep their seeds and perturbations whatever their placement, so a member's output does not depend on `--members-per-card`. `ensemble-set.json` records the chosen count, what bound it and the sharing mode. Existing native batching remains available through the ordinary ensemble route.

Each worker retains at most one parent's immutable input/checkpoint cache, loading another when its ancestry changes. This bounds host memory for large parent pools. Disk diagnostic writes and echo-centroid tracking use bounded host workers beside integration. The forecast emits small two-dimensional diagnostics every 15 minutes rather than full member histories. GPU reductions over the complete requested roster produce weighted exceedance probability and paintball for reflectivity, 10 m wind, branch-total precipitation and 2 m temperature, plus reflectivity and precipitation PMM for equal-weight sets. PMM is omitted for unequal weights. Native Rust writes and renders the products, one renderer process per map, as many at once as there are host cores. Missing or failed members prevent a misleading reduced-roster whole-set probability.

`tracks.json` contains the public tracker's 35 dBZ weighted echo-centroid fixes. The plume probability/paintball maps show members whose centroid passes within two grid cells at a saved valid time, without interpolation or storm-identity association. A centroid can be absent when no echo exceeds the threshold.

Each `members/member-NNNNN/receipt.json` records the canonical [member provenance label](ENSEMBLE.md#provenance-label-contract), parent ancestry, checkpoint SHA-256, perturbation receipt, physical card and timing. `ensemble-set.json` records the durable roster, compact member summaries, products and timing. `members_per_card_hour` includes input preparation, checkpoint setup, forecasts and worker cleanup; `end_to_end_members_per_card_hour` additionally includes whole-set products. Multiply neither by the member's forecast duration; the separate `member_forecast_hours_per_card_hour` metric does that explicitly.

Repeat a failed command with `--resume-ensemble RUN_DIRECTORY` to reuse completed members whose diagnostic payloads remain intact. The parent manifest, checkpoint identities, member recipe, duration and product request must match. Failed or incomplete members receive new numbered attempt directories; earlier receipts remain available. Resume throughput counts newly completed forecasts, excluding reused members.

`--cyclone-products REQUEST.json` selects moving-grid cyclone products instead of the default echo products. CUDA accumulates every-step time-max 10 m wind and coastal onshore-flow onset/duration on the request's fixed Earth grid. Rust makes the weighted hurricane-wind probability, MSLP track plume, moving-footprint animation frames, common-time reflectivity stamps and optional observed radar panel. NHC best track is a verification overlay. Matplotlib is used only for the pressure/wind time-series chart. The [Ian case](../../cases/ian-20220928/README.md) gives the complete request and run sequence.

Measured example (2026-10-17): 20 one-hour branches from two parents completed on one RTX 5070 Ti 16 GB, at 3 km on a 300×300×49 grid with 15 s model steps. The command was:

```sh
gpuwm ensemble --members 20 --branch-from parents.json --cards all \
  --branch-hours 1 --outdir research-ensemble
```

The forecast pipeline took 500.115 s: **143.97 members per card-hour**. Whole-set products took another 9.122 s, giving **141.39 members per card-hour including products**. Parent generation preceded the measured set. Median member time was 24.355 s, including 1.503 s of restore/setup. All 20 members contributed to four valid times; complete-roster probabilities and paintball masks matched independent counts exactly. A separate two-member surface branch verified both perturbations and provenance in retained native history files.

This qualifies the single-card checkpoint door. An actual 1,000-member/eight-card run remains unmeasured, including shared-storage contention and complete-roster product memory/disk requirements.

Measured co-residency (2026-10-24): 20 nine-hour branches of a 12-3 km AIFS parent (88×78 and 272×232 cells, 49 levels; one member peaks at 5.8 GB) on one RTX PRO 6000 98 GB, 16 host cores. One member per card: members 1,644.8 s, products 679.4 s, whole set 2,324.2 s. `auto` (4 per card, private MPS): members 1,054.5 s, products 49.6 s, whole set 1,104.2 s, 2.1x faster. All 20 members' diagnostic arrays (10,080) and final model-state digests, and all 4,900 maps, are byte-identical between the two. Four co-resident members in ordinary contexts ran 0.93x the one-member step rate; eight under MPS ran slower than four.
