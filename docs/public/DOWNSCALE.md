# Offline downscaling (`gpuwm downscale`)

`gpuwm downscale` re-runs a finer nest from an **archived** parent run
-- the CUDA-native equivalent of WRF's offline `ndown` workflow. It
accepts ArWen and stock-WRF history files, proves the parent series
before touching the GPU, and advances only the child. This page is the
workflow plus the measured cost of the one shortcut people take
(coarse parent output cadence), so you can decide with numbers.

## From a fresh box to a downscaled nest, command by command

This is the complete verified path (walked end to end on a wheel
install, Windows 11 / RTX 3080, 2026-08-17).  Two facts shape it, and
knowing them up front saves the two dead ends everyone hits:

1. **The parent must be a run that wrote a gpuwm restart** -- that
   restart is the physics evidence `--parent-restart` binds.  A
   *single-domain* `gpuwm domain` emission disables restart writing
   (`restart_interval_s = 0`; the prepared single-domain runner writes
   no checkpoints even if you set it), so the README's one-domain
   quickstart run **cannot be downscaled**.  Emit a nest ladder: ladder
   emissions set hourly restarts, and the tree runner writes
   `gpuwmrst_d0N_*.npz` beside the wrfouts.
2. **A full-physics child needs a child-grid surface file**
   (`--child-surface-from`), and your own preparation already built
   one: rw-wps emits `wrfinput_d0N` for every nest under
   `<prepared>/wrf-native-input/`.  Derive the child at a nest's own
   geometry and that nest's `wrfinput` is the surface source.

```bash
# 1. Emit a ladder sized to your card at your point.  The nest's
#    default history cadence is already 900 s -- the downscale guidance
#    cadence; d01 writes hourly (add --history-interval 900 to densify).
gpuwm domain --point 39.7,-84.0 --card 12gb --ladder 12-3 \
  --source gfs --cycle latest --hours 6 --out myarea.toml

# 2-4. Fetch, then run the chain the closing block prints: materialize
#    the authority, author the front-door manifest (it prints the
#    complete rw-wps command), run rw-wps, then paste the
#    gpuwm-prepared-tree-forecast line rw-wps prints.  The parent run
#    directory now holds wrfout_d01/d02 frames and gpuwmrst_d0*.npz.

# 5. Derive the child and read the plan.  --dry-run prints placement
#    and writes the derived TOML without running; if the child needs a
#    surface source it says so here instead of after you walk away.
gpuwm downscale RUN/wrfout --parent-domain 1 \
  --parent-restart RUN/gpuwmrst_d01_<instant>__<set>.npz \
  --point 39.7,-84.0 --ratio 4 --child-size 120,96 \
  --hours 2 --out child-run --dry-run

# 6. Point --child-surface-from at the preparation's own wrfinput for
#    the nest whose geometry the child reuses, and run.  In the
#    verified walk the derived placement matched the ladder nest's
#    i/j_parent_start exactly (check the --dry-run line against your
#    emitted TOML's [[domain]] entries before trusting it).
gpuwm downscale RUN/wrfout --parent-domain 1 \
  --parent-restart RUN/gpuwmrst_d01_<instant>__<set>.npz \
  --point 39.7,-84.0 --ratio 4 --child-size 120,96 \
  --child-surface-from PREPARED/wrf-native-input/wrfinput_d02 \
  --hours 2 --out child-run
```

Measured on the walk above (120x96x49 child at 3 km, 2 h, full
physics): 480 child steps in 27 s wall on an RTX 3080, `report.json`
receipts complete.  To downscale the *innermost* nest to a brand-new
finer grid instead, emit a deeper ladder (`--ladder 12-3-1`) so the
preparation builds a `wrfinput_d03` on that finer grid -- the forecast
does not have to run the extra nest; only the preparation has to build
it.  A microphysics-only child (surface physics off in its
`--child-config`) runs with no surface source at all.

## The contract, in order

1. **Prove the parent.** Complete frame inventory, frozen geometry,
   regular cadence, one producer -- checked before anything runs.
2. **Prove the physics.** The parent's microphysics identity must come
   from companion evidence -- an ArWen restart file or the WRF
   `namelist.input` -- never inferred from the variable inventory.
   Cross-scheme conversion is explicit and fail-closed (active
   condensate is never paired with a fabricated zero number moment;
   NSSL targets require the official `calcnfromq` diagnosis).
3. **The boundary cadence defaults to the archive's own.** With no
   cadence flag the tool uses the parent history cadence and says so
   in one warning line; `--max-boundary-interval-seconds` bounds it
   explicitly and `--accept-parent-cadence` is the warning-free
   spelling of the default (one or the other, never both). Boundary
   cadence is the dominant error term (table below), so the line is
   worth reading -- but a runnable job is never refused for it.
4. **Full-physics children need a child-grid surface source**
   (`--child-surface-from`): a `wrfinput` or history file on the exact
   child grid supplies land identity and soil warm start -- mirroring
   `ndown`'s own requirement. Downscaling replaces the meteorology,
   never the land identity. Microphysics-only children run without
   one.  The in-product way to get that file is the preparation's own
   `wrf-native-input/wrfinput_d0N` (rw-wps emits one per nest; see the
   walkthrough above).  The refusal fires at the front door -- before
   any preprocessing -- and `--dry-run` warns instead of refusing so
   the derived placement can still be read.
5. **Every run writes `report.json`** with SHA-256 receipts of the
   parent frames, the physics evidence, the surface source, the
   boundary-clock identity, and the outputs.

## Two ways to specify the child

```bash
# Explicit: a child config plus its placement in the parent
gpuwm downscale RUNDIR --parent-domain 3 \
  --parent-restart RUNDIR/gpuwmrst_d03_....npz \
  --child-config child.toml --ratio 2 \
  --i-parent-start 151 --j-parent-start 151 \
  --child-surface-from wrfinput_child \
  --max-boundary-interval-seconds 900 --out child-run

# Derived: geometry around a point, physics inherited verbatim from
# the parent's restart evidence, extent fitted to a VRAM budget
gpuwm downscale RUNDIR --parent-domain 3 --parent-restart ... \
  --point 39.7,-84.0 --ratio 3 --card 24gb --hours 3 \
  --max-boundary-interval-seconds 900 --out child-run
```

`--card` takes the same tiers `gpuwm domain` does
(`12gb|16gb|24gb|32gb`, [HARDWARE.md](HARDWARE.md)) and `--vram-gib N`
covers anything between them; both feed the same budget model. Until
1.4 this command hardcoded a shorter list and rejected `--card 12gb`
while the wizard accepted it -- two commands quoting different tier
lists for the same concept.

Point mode writes the derived TOML beside `--out` for reuse, and
inherits the *parent's* per-domain tuning knobs verbatim (documented
in the TOML header). `--dry-run` prints the full plan -- frames,
physics binding, placement -- without running. In a controlled test,
point mode re-derived the reference run's actual d04 geometry (parent
start 151/151, 400x400 at 500 m, dt 2.5 s, correct physics) from
nothing but the point and the d03 restart evidence.

## The measured cadence cost

Acceptance measurement (2026-07-29, RTX 5090): the reference case's
1 km parent domain (501x501, Thompson), archived **hourly**, downscaled
to the run's own 500 m d04 geometry (400x400, ratio 2, dt 2.5 s, full
physics) over 3 h spanning convective initiation -- then scored against
the *live-nest* d04 of the same run on the interior grid (5-row rim
excluded). At F+0.0 the offline cold start matches the live nest on the
four metrics this table scores -- 0.000 K T2 MAE, 1.000 T2 correlation,
0.1 Pa PSFC MAE, 1.000 wind correlation (first row below). That row is
the whole scope of the statement: it is four comparator metrics on one
pair of runs, and no full-state digest was taken of the offline/live
pair. Read the rest of the table as forcing-path cost: hourly
interval-linear boundaries versus the live nest's every-parent-step
forcing.

| lead | T2 MAE (K) | T2 corr | PSFC MAE (Pa) | wind10 corr | refl corr | refl MAE (dBZ) | CSI@20 |
|---|---|---|---|---|---|---|---|
| F+0.0 | 0.000 | 1.000 | 0.1 | 1.000 | -- | -- | -- |
| F+0.5 | 0.133 | 0.992 | 37.5 | 0.978 | 0.922 | 0.31 | -- |
| F+1.0 | 0.127 | 0.988 | 23.0 | 0.957 | 0.906 | 0.80 | -- |
| F+1.5 | 0.156 | 0.989 | 26.6 | 0.943 | 0.845 | 1.20 | -- |
| F+2.0 | 0.253 | 0.977 | 23.9 | 0.916 | 0.262 | 3.05 | -- |
| F+2.5 | 0.289 | 0.975 | 29.8 | 0.871 | 0.248 | 13.06 | 0.000 |
| F+3.0 | 0.430 | 0.964 | 28.4 | 0.802 | 0.148 | 25.38 | 0.020 |

Read it as two regimes:

- **Mesoscale envelope: close.** T2 correlation never drops below
  0.96, PSFC MAE stays under 40 Pa, 10 m wind correlation is 0.80 at
  F+3 h. The child tracks the parent-constrained mesoscale state for
  the full window.
- **Convective scale: decorrelates at initiation.** The reference
  run's interior reflectivity maximum crosses 0 dBZ between F+1.5 and
  F+2.0; exactly there, reflectivity correlation collapses 0.845 ->
  0.262 and CSI at 20 dBZ is near zero. Both runs make storms -- in
  different places.

Run cost: 4,320 child steps, 66.3 min wall, 7.98 GiB pool peak, no
non-finite values, receipts complete.

## The 15-minute guidance

**Write parent history at 15-minute or denser cadence when you plan to
downscale.** The table above is the measured price of hourly
boundaries at 500 m across convective initiation. With hourly
boundaries, an offline child is honest for mesoscale downscaling --
temperature, pressure, wind envelope -- and is **not** a substitute
for a live nest at convective scale: cell placement inside the child
is decorrelated from what a live nest would produce. The CLI prints
this guidance as a warning whenever the archive is coarser than 15
minutes, however the cadence was chosen; `--accept-parent-cadence` is
recorded in `report.json` as
`boundary_cadence_provenance.accepted_parent_cadence: true` beside
the ceiling and the effective boundary interval (an explicit
`--max-boundary-interval-seconds` records `false` there instead).

Output cadence is a config decision on the parent run
(`history` interval per domain), so the time to make it is before the
parent runs, not after.

## Rendering the child

`gpuwm render` handles the child's wrfouts like any other run's, and
sub-hourly history cadences render exactly: with the rust engine (the
`tools/rustwx` build) every frame carries its precise
`valid_..._lead_...` stamp, so a 15-minute child cadence never rounds
to a fake whole hour. To eyeball what downscaling changed, render the
parent domain and the child into separate directories and compose
labeled pair sheets:

```bash
gpuwm render --pair out/parent/png out/child/png --out out/compare
```

## Known limits

- **No vertical remapping:** the child keeps the parent's eta levels;
  terrain is SINT-inherited from the parent.
- One child per invocation; nest the workflow by downscaling the
  downscaled run's own archive.
- Stock-WRF parents work (history + `namelist.input` as physics
  evidence), with the same explicit-cadence contract; a point-mode
  child of a WRF parent needs `--child-config` (a WRF namelist carries
  no complete ArWen run configuration).
- The child runs the same bound boundary-clock semantics as the
  production nest tree (WRF's `dtbc` recurrence), and its restarts
  record that identity -- checkpoints cannot silently mix with
  legacy-clock trajectories.

Full contract details:
[native-offline-child-contract.md](../native-offline-child-contract.md).
