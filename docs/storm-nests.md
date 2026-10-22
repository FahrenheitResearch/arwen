# Storm-following nests: `[storm_nests]`, `--storm-nests`, `--nest-at`

Fine nests that spawn on storms, follow them and stop on their own, on any
single-root configuration. The headline shape is the packaged HRRR v4 recipe
(`hrrr_v4_gsd41`, full CONUS at 3 km) with 1 km nests. Off unless asked.

## Asking for them

In the configuration:

```toml
[storm_nests]
slots = 3                      # automatic nests at once; "auto" prices them from the card
points = [{lat = 41.25, lon = -96.0, at = "2024-04-27T19:00:00"}]   # on command
```

Or on the command line, which overrides the table's `slots` and adds points:

```
gpuwm go hrrr.toml --storm-nests 3
gpuwm go hrrr.toml --storm-nests auto
gpuwm go hrrr.toml --nest-at 41.25,-96.0@2024-04-27T19:00 --nest-at 35.2,-97.4@3600
```

`gpuwm go` writes the expanded configuration to `<outdir>/storm-nests/` with a
`storm-nests.json` receipt beside it, and every later stage (preparation, the
statics corridor, the forecast) reads that copy.

## What happens

* **Automatic.** Each slot is a dormant nest. Every `cadence_s` (15 min) it
  looks at the parent's 2-5 km updraft helicity, the running maximum since its
  last look. When a cell reaches `spawn_threshold` (75 m2 s-2, the 3 km
  surrogate-severe threshold of Sobash et al. 2011 and 2016) in ground that no
  live nest already covers, the nest is born centred on that storm's UH
  centroid, from the live parent (WRF's own nest initialisation, with the
  nest's own 1 km terrain and land use). Two slots never claim one storm.
* **On command.** Each point is a nest born at that point at that time.
* **Following.** Every `follow_cadence_s` (15 min) a nest re-centres on its
  storm's UH centroid, moving at most the parent cells its overlap floor
  allows; reflectivity takes over when the UH is weak.
* **Stopping.** A nest retires when the UH under it stays at or below
  `retire_threshold` (25 m2 s-2) for `retire_sustained_s` (30 min) after it has
  lived `min_lifetime_s` (1 h). The receipt names why: `decayed`,
  `left-domain` (quiet with the nest held at the parent's edge), or `merged`
  (two nests overlapping by `merge_overlap` of the smaller one, after a storm
  merger: the younger retires). A retired slot re-arms after
  `rearm_cooldown_s` for up to `max_firings` storms.

## Every key

| key | default | meaning |
|---|---|---|
| `slots` | 3 | automatic slots, or `"auto"` (priced from free card memory, at most `max_slots`) |
| `max_slots` | 8 | cap for `"auto"` and for `slots` |
| `dx_km` | 1.0 | nest spacing; a whole fraction (>= 2) of the root's |
| `size_km` | 240 | nest side length |
| `detector` | `"uh"` | `"uh"` (2-5 km updraft helicity) or `"reflectivity"` (composite dBZ) |
| `spawn_threshold` | 75 | detector value that spawns a nest |
| `follow_threshold` / `fallback_threshold` | 50 / 25 | tracker thresholds |
| `retire_threshold` | 25 | quiet below this |
| `retire_sustained_s` / `min_lifetime_s` | 1800 / 3600 | how long quiet, and how young a nest is never retired |
| `merge_overlap` | 0.5 | footprint overlap that retires the younger of two nests |
| `earliest_s` / `latest_s` | one step / run end | when automatic slots may fire |
| `cadence_s` / `follow_cadence_s` | 900 / 900 | spawn look / follow look |
| `history_interval_s` | 900 | nest output interval |
| `radt` | 5 | nest radiation interval, minutes |
| `max_firings` / `rearm_cooldown_s` | 4 / 1800 | storms per slot, and the gap between them |
| `vram_gib` | card free memory | budget for `slots = "auto"` |
| `points` | none | `{lat, lon, at}` (ISO time) or `{lat, lon, at_s}` |

## What the nests run, and the parent

The nests run the parent's physics with five stated differences, each forced by
a named limit of a nest rebuilt mid-run: no cumulus scheme (`cu_physics = 0`,
convection resolved at 1 km), no GSL orographic drag (`gwd_opt = 0`; a child's
statics carry no sub-grid orography statistics), no lake model
(`sf_lake_physics = 0`; the rebuild does not carry the lake column), a
5-minute radiation interval, and WPS 30-arc-second terrain rather than the 30 m
overlay (a slot's corridor spans the whole parent, and the overlay over the
contiguous US is a ~3,000-tile, ~80 GB mosaic; a configuration that declares its
own `[static.highres]` keeps it). The parent keeps its own configuration except its
history interval, which drops to the follow cadence (the tracker's reflectivity
fallback is stashed at the parent's history cadence).

## Cost

* **Memory.** Every slot is reserved from the start (about 1.2 GiB for a
  240 km, 1 km nest under the HRRR recipe), so a spawn can never run the card
  out; `gpuwm check` prints one line per slot.
* **Preparation.** A dormant slot can fire anywhere in the parent, so it is
  prepared with a 1 km statics corridor over the whole parent (about 13 GB of
  host memory and disk for full CONUS). Slots of one shape share one build.
* **Compute.** A slot costs nothing until it fires. A live nest's cost per
  parent step is in the run receipt.

## Receipts

* `spawn_receipts.jsonl`: every boundary's decision, every birth (trigger
  evidence, placement, statics source, land-state interpolation) and every
  retirement row with its `stop_reason`.
* `relocation_receipts.dNN.json`: every follow move.
* `evidence/nest-lives.json`: one row per nest life: grid, episode, spawn time
  and place, track, stop time and reason.
* Nest history and pictures: `wrfout/dNN/episode-NNN/` and
  `<render>/dNN-1km/episode-NNN/<product>/...`.

## Routes

The `gpuwm go` prepared tree route (any `[fetch]` source on the mapped chain,
HRRR recipe included) and the case-data route (`gpuwm run`). Moving or spawned
nests cannot run on a split-card (`[devices]`) tree: the parent runs on one
card, streamed through host memory if it does not fit. The rap-native mapping
prepares at most four domains (`max_dom = 4` in its rw-wps mapping), so that
route carries the root and up to three nests, automatic and on-command
together. The GFS and ERA5 direct prepared caches still refuse spawn tables by
name.
