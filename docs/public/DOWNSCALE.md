# Offline downscaling (`gpuwm downscale`)

`gpuwm downscale` re-runs a finer child grid from an **archived** parent
run -- the CUDA-native equivalent of WRF's offline `ndown` workflow. It
accepts gpuwm and stock-WRF parents, proves the parent series before
touching the GPU, and advances only the child.

The downscale command itself is one invocation and it is not where people
stall. What stalls people is producing a parent archive the command will
accept, and **that part differs by source**. This page walks one source
end to end with every command written out, names what the other routes
change, and puts each unfixed rough edge at the step where it bites.

## What the parent archive must contain

Two preconditions, both checked at the front door before any GPU work.

**1. A gpuwm restart file beside the history frames.** The restart is the
physics evidence `--parent-restart` binds -- the parent's microphysics
identity is never inferred from a variable inventory. Without it:

```
gpuwm downscale: parent physics must be bound from companion evidence: pass --parent-restart (gpuwm) or --parent-namelist (stock WRF)
```

This refusal fires on the real run **and** on `--dry-run`. Check for the
files before you plan anything else:

```bash
ls RUN/gpuwmrst_d0*.npz
```

The real filename pattern is `gpuwmrst_d0N_YYYY-MM-DD_HH_MM_SS.npz`, for
example `gpuwmrst_d01_2024-05-06_01_00_00.npz`. A stock-WRF parent uses
`--parent-namelist namelist.input` instead.

**2. For a full-physics child, nothing extra -- but a child-grid file
raises the fidelity.** Land identity and the soil warm start have to come
from somewhere, and `gpuwm downscale` now resolves that itself: with no
`--child-surface-from`, it takes them off the parent's own history frame
and puts them on the child grid through WRF's nest-birth operators, the
same route WRF uses for a nest with `input_from_file = .false.`. It says
so, once:

```
gpuwm downscale: child surface derived from RUN/wrfout_d01_... (21 fields, MODIFIED_IGBP_MODIS_NOAH)
warning: child surface state interpolated from the parent's own history rather than built on the child grid -- the child's land-use, soil category and landmask are the PARENT's ...
```

That warning is the cost, and it is real: the child's coastlines, lakes
and islands are its parent's, not the ones its own spacing could resolve.
`--child-surface-from` is the higher-fidelity route and takes a
`wrfinput` or history file whose grid **equals** the child's. The grid
check on that flag is exact:

```
gpuwm downscale: RUN\wrfout_d01_2024-05-06_00_00_00 south_north=48 does not match the child grid south_north=96; the surface source must be on the EXACT child grid
```

Where such a file comes from depends on your route -- see
[Route 2](#route-2-the-prepared-sources), which writes one per nest.

## Which route builds your parent -- ask the registry

Do not guess. Every source declares its route:

```bash
gpuwm sources              # all 32 rows, one line each
gpuwm sources era5         # one row in full
gpuwm sources --json       # the same facts as JSON
```

The field that decides your chain is `run_plan.intent_chain`. It takes
three values on this release:

| `intent_chain` | Sources | Parent is built by |
|---|---|---|
| `experiment` | `era5` | `gpuwm run CONFIG.toml` -- config-driven, no chain |
| `prepared:go` / `prepared:hrrr` | `gfs`, `hrrr` | `gpuwm go CONFIG.toml` end to end |
| `prepared:staged` | `hrrr-prs`, `gem-gdps`, `icon-eu`, `aigfs`, `ecmwf-open-data`, `aifs`, `rap`, `rrfs` | staged chain: authority, manifest, `rw-wps`, tree runner |

These are different enough that a walkthrough for one does not transfer.
Route 1 below is `era5`, written out completely. Route 2 covers the
`prepared:*` sources.

`mapped` is not a fourth route to a parent: it is the generic declarative
adapter for bytes you already have, it has no fetch route, and
`wizard_planable` is `no`, so `gpuwm domain` cannot plan it.

---

# Route 1: ERA5, from nothing to a downscaled child

Every command below was run in this order. Values are literal.

### 0. Find the console scripts

On an editable or `--user` install the console scripts land in your
**user** Scripts directory, which is often not on `PATH`:

```
C:\Users\<you>\AppData\Roaming\Python\Python313\Scripts\
```

`gpuwm.exe`, `rw-wps.exe` and `gpuwm-prepared-tree-forecast.exe` are
there. If `gpuwm` is not found, prepend that directory before continuing.

### 1. Emit the config FIRST, before fetching anything

The wizard computes the geographic box you must request. Fetching first
means fetching the wrong box.

```bash
gpuwm domain --point 39.0,-98.0 --source era5 --cycle 2024-05-06T00 \
  --hours 12 --history-interval 900 --vram-gib 6 \
  --geog-root C:\WPS_GEOG --name demo \
  --out demo.toml --data-dir era5-raw
```

2.6 s. Writes `demo.toml`, `demo.namelist.wps` and `Vtable.ERA5_CDO`, and
emits `nx = 60, ny = 48, dx = 12000.0`.

**Pass `--geog-root` explicitly.** Without it the wizard writes
`geog_root = "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG"` into the config, which
resolves to a path that may not hold your staged static data. This is a
**workaround**, not a fix: the emitted default is not validated at emission
time. `gpuwm fetch-geog` stages the nine datasets (~16 GiB unpacked) if you
do not have them.

**Sizing on a 10 GiB card.** The grid-independent envelope is about
3.24 GiB (CUDA context plus the kernel local-memory backing store), so
small budget changes move the grid a long way. Measured on an RTX 3080
10 GiB, Windows/WDDM:

| flag | fitted grid |
|---|---|
| `--vram-gib 10` | 406 x 326 |
| `--vram-gib 7` | 84 x 68 plus a nest |
| `--vram-gib 6` | 60 x 48 |
| `--vram-gib 4` | refused, naming the breakage |

`--card` offers only `{12gb,16gb,24gb,32gb}`, so a 10 GiB card has no tier
and must use `--vram-gib`. A WDDM desktop idles 2.7-4.6 GiB, so the card is
never empty; `--vram-gib 6` is a realistic starting budget on a 10 GiB
card with a desktop running. For grids larger than the card can hold, see
[TILES.md](TILES.md).

### 2. Turn restarts on -- the emitted default is 0

A single-domain emission writes:

```toml
restart_interval_s = 0.0
```

with no comment and no advisory. A parent that wrote no restart **cannot
be downscaled** (precondition 1 above). Edit line 17 of `demo.toml`:

```toml
restart_interval_s = 3600.0
```

This is a **workaround for an unfixed default**: the wizard should emit a
nonzero interval and does not. One edit is sufficient on this route --
`gpuwm run` honours it and writes `gpuwmrst_d01_*.npz` beside the wrfouts.
Nest-ladder emissions already default to `3600.0`.

### 3. Request the bytes

```bash
gpuwm fetch --source era5 --cycle 2024-05-06T00 --hours 12 \
  --area 34.30,-104.39,43.63,-91.61 --out era5-raw
```

Use the `--area` the wizard printed in step 1, **widened**. This command
downloads nothing -- ERA5 acquisition is manual because the Copernicus CDS
API needs a personal account key that gpuwm will not embed. It writes:

```
fetch era5: wrote era5-raw\era5-cds-request.json
fetch era5: wrote era5-raw\era5-cds-retrieve.py (runs the retrieval)
```

**Ask for more area than the wizard prints.** CDS snaps the requested box
inward to its grid. Requesting `34.30,-104.39,43.63,-91.61` delivered
`lat [34.5, 43.5] lon [-104.25, -91.75]` -- smaller on all four sides. The
preflight catches it later, but only after the download:

```
[spatial-coverage] domain point lat/lon=(34.0024, -105.184) lies outside forcing lat [34.5, 43.5] lon [-104.25, -91.75]
```

### 4. Retrieve, with the CDS client

`era5-cds-retrieve.py`, written beside the request, runs both requests and
concatenates their output. It resolves every path from its own location, so
the working directory does not matter and the files land in `--out`:

```bash
pip install cdsapi
python era5-raw/era5-cds-retrieve.py
```

The command is printed by step 3, with your absolute path filled in.

The client reads your key from `~/.cdsapirc` in the home directory of
**whichever interpreter runs it**. If you retrieve from WSL on a Windows
box, that is WSL's home, not `C:\Users\<you>`. `gpuwm sources era5` reports
the Windows path and whether a key is present there, so on a WSL retrieval
its "no key" line can be correct about Windows and irrelevant to the run.
The fetch step prints the WSL form too, with the path already translated to
`/mnt/c/...`:

```bash
wsl sh -c "python3 -u /mnt/c/wx/era5-demo/era5-raw/era5-cds-retrieve.py"
```

Run it in the foreground. `nohup ... &` inside `wsl sh -c` does not
survive: the job dies when `wsl.exe` returns and no log is written.

Measured on this walk: 95 s for a 2-time, 9x9-point request; both parts plus
`era5-combined.grib` written into `--out`, nothing in the shell's directory.

### 5. Validate

```bash
gpuwm fetch --source era5 --validate era5-combined.grib --area 34.30,-104.39,43.63,-91.61
```

The retrieval prints this command with your own `--area` already in it.
`era5 validation: PASS` lists the GRIB1 envelope count, the valid times, the
pressure ladder, the surface inventory, **and the delivered grid**:

```
ok: grid 9x9, 0.25 x 0.25 deg, lat [39.00, 41.00] lon [-99.00, -97.00]
ok: the delivered grid covers the requested box lat [39.00, 41.00] lon [-99.00, -97.00]
```

With `--area` given, a delivered box that falls more than one grid cell
short of the requested one on any edge is a FAIL, naming the edge and the
shortfall -- so the wrong-box file is caught here instead of at `gpuwm
check` or mid-run. One cell of tolerance is the CDS's own inward snap
(step 3). Without `--area` the extent is still reported, followed by a line
saying it was checked against nothing.

### 7. Check, then run the parent

Order matters: `gpuwm check` refuses before the bytes exist, so it comes
after retrieval, not before.

```bash
gpuwm check demo.toml
gpuwm run demo.toml --outdir demo-run
```

If the config is missing its bytes you get, at rc 0:

```
gpuwm check: forcing file ...\era5-combined.grib declared in [case_data] of ...\demo.toml does not exist.
```

`gpuwm run` is silent through preparation -- a banner, then roughly 40 s
with no output before anything else appears. That is expected.

Measured, RTX 3080 10 GiB, 12 h forecast at 60 x 48 x 49 and 12 km,
including ERA5 preprocessing: **51 s wall**, `status: complete`, 49
`wrfout_d01_*` frames at 900 s, 12 `gpuwmrst_d01_*.npz`. Machine-wide peak
4800 MiB VRAM, 1965 MiB host RSS.

Do not run `gpuwm go` here. It refuses ERA5 configs by name and points at
the right command:

```
gpuwm go: demo.toml declares a [case_data] table, which is the ERA5 config-driven route -- `gpuwm run` executes that one directly and needs no chain.
  remedy: gpuwm check demo.toml && gpuwm run demo.toml
```

### 8. Derive and run the child

```bash
gpuwm downscale demo-run --parent-domain 1 \
  --parent-restart demo-run/gpuwmrst_d01_2024-05-06_01_00_00.npz \
  --point 39.0,-98.0 --ratio 3 --child-size 120,96 \
  --vram-gib 6 --hours 2 --out child-run --dry-run
```

1.4 s, rc 0. It prints the placement and writes the derived TOML. A REAL
run writes it to `child-run/child.toml`, inside the run it describes. A
`--dry-run` cannot: the run claims `--out` for itself, so a dry run
writes `child-run.child.toml` beside it and says so.

**Re-running with the same `--out` is safe.** A run that refuses hands
the directory it claimed back, so the corrected command meets the tree
the first attempt found. An `--out` that already holds a run's output is
refused by name, saying what it holds and that you may pass a new `--out`
or remove the old directory; nothing in it is ever overwritten or merged
into, because the `report.json` a run publishes has to describe one run.

**Pass `--vram-gib` or `--card` here too.** `gpuwm downscale --card`
defaults to `24gb` while `gpuwm domain` measures the local card, so a
10 GiB owner who omits it gets a child sized for a card they do not have.

**You do not need to check the derived placement by hand.** For a 120 x 96
child at ratio 3 inside a 60 x 48 parent, centred, the tool printed
`{"ratio": 3, "i_parent_start": 11, "j_parent_start": 9}`, matching the
arithmetic exactly. `--dry-run` is worth running to read the plan; it is
not a required rehearsal, because the real run refuses at the front door in
about a second, before the GPU, with the same remedy text.

Then drop `--dry-run` to run it. Measured on the RTX 3080: `contract_pass`,
`child_shape [49, 96, 120]` at 4000 m, 49 boundary frames, first frame
`wrfout_d02_2024-05-06_00_00_00` at 44,803,168 B. Machine-wide peak
8859 MiB VRAM, 3074 MiB host RSS -- close enough to a 10 GiB card's ceiling
that a loaded desktop matters.

#### A full-physics ERA5 child needs no extra file

This used to be the dead end of the whole page: `--child-surface-from` was
mandatory for a full-physics child, and **no ERA5 route produced a file on
an arbitrary child grid**. `gpuwm run` -- the route ERA5 is steered onto --
writes no `wrf-native-input/`; the `rw-wps` prepared-tree route that does
write one needs a front-door manifest `gpuwm fetch` authors for `gfs` only;
and building an ERA5 nest at the child geometry to make one hit its own
refusal in `initialize_child`. The refusal demanded a file the product
could not make, and said so only after the parent forecast was paid for.

It is closed. The parent's own history already carries the nine surface
fields and the landuse identity attributes -- any run with a land-surface
scheme publishes them -- and the child grid is an exact refinement of a
parent window, so `gpuwm downscale` puts them where the child needs them
itself. Measured on node-1 (RTX 5070 Ti), the 12 km ERA5 parent from
`configs/era5_demo.toml` downscaled to a 72 x 54 child at 4 km with
`sf_surface_physics = 2`, `sf_sfclay_physics = 91`, `bl_pbl_physics = 1`
and both radiation streams:

```
gpuwm downscale RUN --parent-domain 1 --parent-restart RUN/gpuwmrst_d01_... \
  --point 35.3,-97.5 --ratio 3 --child-size 72,54 --hours 1 --out CHILD
```

```
gpuwm downscale: child surface derived from RUN/wrfout_d01_... (21 fields, MODIFIED_IGBP_MODIS_NOAH)
warning: child surface state interpolated from the parent's own history rather than built on the child grid ...
```

`"result": "PASS"`, 180 steps, `nan: false`, two `wrfout_d02_*` frames and
a final restart, 2.8 s wall. `report.json` carries the whole derivation
under `child_surface_source`: which fields were donor-copied, the masked
interpolator's branch counts per field, the parent frame's sha256.

**What you give up.** The child's land-use, soil category and landmask are
its parent's, carried down from the parent cell each child column sits in,
so coastlines, lakes and islands the child's 4 km spacing could resolve are
not resolved. That is WRF's own answer for a nest with
`input_from_file = .false.`, and `--child-surface-from` remains the
higher-fidelity route when you have a child-grid file (see
[Route 2](#route-2-the-prepared-sources), which writes one per nest).

A **microphysics-only child** is still available and still not a
substitute: zeroing `sf_surface_physics` alone is not enough, the check
reads `sf_surface_physics`, `sf_sfclay_physics` and `bl_pbl_physics`
together. In the 2.5.1 walk, with all three zeroed, the contract passed and
the run failed mid-integration with
`FloatingPointError: microphysics RAINNC contains a non-finite value`.
Prefer the full-physics child.

---

# Route 2: the prepared sources

For `intent_chain` values `prepared:go`, `prepared:hrrr` and
`prepared:staged`, the parent is built by the prepared pipeline rather than
by `gpuwm run`, and the downscale half of the page is unchanged.

Emit a config the same way, naming your source and a nest ladder:

```bash
gpuwm domain --point 39.7,-84.0 --card 12gb --ladder 12-3 \
  --source gfs --cycle latest --hours 6 --history-interval 900 \
  --geog-root C:\WPS_GEOG --out tree.toml --data-dir gfs-raw
```

Ladder emissions set `restart_interval_s = 3600.0` already, so step 2 of
Route 1 does not apply. The wizard closes by printing the rest of the
chain with your values filled in. It is four commands, not one, and they
run in this order:

```bash
# a. fetch the bytes, using the --area and --cycle the wizard printed
gpuwm fetch --source gfs --cycle 2026-08-20T18 --hours 6 \
  --area 14.95,-115.69,63.65,-52.31 --out gfs-raw

# b. confirm the sizing against the card that is actually present
gpuwm check tree.toml

# c. materialize the physics authority
python -m gpuwm.prepared_single_domain_forecast --materialize-authorities \
  --source gfs --base-experiment-config tree.toml \
  --base-wps-namelist tree.namelist.wps \
  --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1 \
  --output-directory tree-authority

# d. author the front-door manifest
gpuwm fetch --source gfs --author-front-door-manifest \
  --out gfs-raw \
  --wps-namelist tree-authority/namelist.wps \
  --experiment-config tree-authority/experiment.toml
```

Step (d) prints a complete `rw-wps` command; run it, adding your
`--geog-root`. `rw-wps` in turn prints the runner command with both of its
values filled in:

```bash
gpuwm-prepared-tree-forecast \
  --prepared-root <the directory rw-wps wrote> \
  --preparation-receipt-sha256 <the sha256 rw-wps printed>
```

Those last two values are produced by the preceding command and cannot be
known in advance; everything else above is complete as written. Section 3a
of [FIRST-LIGHT.md](FIRST-LIGHT.md) carries the same sequence.

For a single-domain `prepared:go` config (`--ladder 12`, the default),
`gpuwm go tree.toml` runs the whole chain end to end.

**Before downscaling, confirm the archive actually has restarts** --
`ls RUN/gpuwmrst_d0*.npz` -- rather than assuming the emission's interval
reached the runner.

On this route `--child-surface-from` is reachable: the preparation writes
`wrfinput_d0N` for every nest under `<prepared>/wrf-native-input/`. Derive
the child at a nest's own geometry and that nest's `wrfinput` is the
surface source. To downscale the innermost nest to a brand-new finer grid,
emit a deeper ladder (`--ladder 12-3-1`) so the preparation builds a
`wrfinput_d03` on that finer grid; the forecast does not have to run the
extra nest, only the preparation has to build it.

---

# Deriving the child: the two forms

```bash
# Explicit: a child config plus its placement in the parent
gpuwm downscale RUNDIR --parent-domain 3 \
  --parent-restart RUNDIR/gpuwmrst_d03_2024-05-06_01_00_00.npz \
  --child-config child.toml --ratio 2 \
  --i-parent-start 151 --j-parent-start 151 \
  --child-surface-from wrfinput_d04 \
  --max-boundary-interval-seconds 900 --out child-run

# Derived: geometry around a point, physics inherited verbatim from the
# parent's restart evidence, extent fitted to a VRAM budget
gpuwm downscale RUNDIR --parent-domain 3 \
  --parent-restart RUNDIR/gpuwmrst_d03_2024-05-06_01_00_00.npz \
  --point 39.7,-84.0 --ratio 3 --vram-gib 6 --hours 3 \
  --max-boundary-interval-seconds 900 --out child-run
```

`--card` takes the same tiers `gpuwm domain` does
(`12gb|16gb|24gb|32gb`, [HARDWARE.md](HARDWARE.md)) and `--vram-gib N`
covers anything between them; both feed the same budget model. Point mode
writes the derived TOML to `<out>/child.toml` for reuse, and inherits the
*parent's* per-domain tuning knobs verbatim (documented in the TOML
header).

`--hours` does not truncate the `parent_frames` list in the printed plan --
every archived frame is listed. The window lands in the derived config as
`run_seconds`.

## The contract, in order

1. **Prove the parent.** Complete frame inventory, frozen geometry,
   regular cadence, one producer -- checked before anything runs.
2. **Prove the physics.** The parent's microphysics identity must come
   from companion evidence -- a gpuwm restart file or the WRF
   `namelist.input` -- never inferred from the variable inventory.
   Cross-scheme conversion is explicit and fail-closed (active condensate
   is never paired with a fabricated zero number moment; NSSL targets
   require the official `calcnfromq` diagnosis).
3. **The boundary cadence defaults to the archive's own.** With no cadence
   flag the tool uses the parent history cadence and says so in one
   warning line:

   ```
   warning: using the parent archive's own 43200 s history cadence as the boundary cadence; pass --max-boundary-interval-seconds SECONDS to bound it
   warning: parent cadence 43200 s is coarser than the 900 s guidance for downscaling
   ```

   `--max-boundary-interval-seconds` bounds it explicitly and
   `--accept-parent-cadence` is the warning-free spelling of the default
   (one or the other, never both). Boundary cadence is the dominant error
   term (table below), so the line is worth reading -- but a runnable job
   is never refused for it.
4. **Full-physics children get their land identity and soil warm start
   automatically.** With `--child-surface-from` they come from a `wrfinput`
   or history file on the exact child grid, mirroring `ndown`'s own
   requirement; without it they are interpolated off the parent's own
   history frame, which is WRF's route for a nest with
   `input_from_file = .false.` and costs the child its own coastlines. The
   run says which one it used, once, and records the whole derivation in
   `report.json`. Microphysics-only children need neither. The surface is
   resolved at the front door, before any preprocessing, so a parent whose
   history cannot seed a child refuses at plan time naming the missing
   fields; `--dry-run` warns instead of refusing so the derived placement
   can still be read.
5. **Every run writes `report.json`** with SHA-256 receipts of the parent
   frames, the physics evidence, the surface source, the boundary-clock
   identity, and the outputs. A failed run writes `failure-capsule.json`
   carrying the config path and its sha256, every input hash, the GPU uuid
   and driver, the git commit, and `last_phase`.

## The measured cadence cost

Acceptance measurement, RTX 5090, 2026-07-29: a 1 km parent domain
(501 x 501, Thompson) archived **hourly**, downscaled to a 500 m child
(400 x 400, ratio 2, dt 2.5 s, full physics) over 3 h spanning convective
initiation, then scored against the *live-nest* child of the same run on
the interior grid (5-row rim excluded).

At F+0.0 the offline cold start matches the live nest on the four metrics
this table scores. That row is the whole scope of that statement: four
comparator metrics on one pair of runs, with no full-state digest taken.
Read the rest as forcing-path cost -- hourly interval-linear boundaries
versus the live nest's every-parent-step forcing.

| lead | T2 MAE (K) | T2 corr | PSFC MAE (Pa) | wind10 corr | refl corr | refl MAE (dBZ) | CSI@20 |
|---|---|---|---|---|---|---|---|
| F+0.0 | 0.000 | 1.000 | 0.1 | 1.000 | -- | -- | -- |
| F+0.5 | 0.133 | 0.992 | 37.5 | 0.978 | 0.922 | 0.31 | -- |
| F+1.0 | 0.127 | 0.988 | 23.0 | 0.957 | 0.906 | 0.80 | -- |
| F+1.5 | 0.156 | 0.989 | 26.6 | 0.943 | 0.845 | 1.20 | -- |
| F+2.0 | 0.253 | 0.977 | 23.9 | 0.916 | 0.262 | 3.05 | -- |
| F+2.5 | 0.289 | 0.975 | 29.8 | 0.871 | 0.248 | 13.06 | 0.000 |
| F+3.0 | 0.430 | 0.964 | 28.4 | 0.802 | 0.148 | 25.38 | 0.020 |

Two regimes:

- **Mesoscale envelope: close.** T2 correlation never drops below 0.96,
  PSFC MAE stays under 40 Pa, 10 m wind correlation is 0.80 at F+3 h. The
  child tracks the parent-constrained mesoscale state for the full window.
- **Convective scale: decorrelates at initiation.** Interior reflectivity
  maximum crosses 0 dBZ between F+1.5 and F+2.0; exactly there,
  reflectivity correlation collapses 0.845 -> 0.262 and CSI at 20 dBZ is
  near zero. Both runs make storms -- in different places.

Run cost: 4,320 child steps, 66.3 min wall, 7.98 GiB pool peak, no
non-finite values, receipts complete.

## The 15-minute guidance

**Write parent history at 15-minute or denser cadence when you plan to
downscale.** The table above is the measured price of hourly boundaries at
500 m across convective initiation. With hourly boundaries an offline child
is sound for mesoscale downscaling -- temperature, pressure, wind envelope
-- and is **not** a substitute for a live nest at convective scale: cell
placement inside the child is decorrelated from what a live nest would
produce.

The CLI prints this guidance whenever the archive is coarser than 15
minutes, however the cadence was chosen. `--accept-parent-cadence` is
recorded in `report.json` as
`boundary_cadence_provenance.accepted_parent_cadence: true` beside the
ceiling and the effective boundary interval; an explicit
`--max-boundary-interval-seconds` records `false` there instead.

Output cadence is a config decision on the parent run (`history` interval
per domain), so the time to make it is before the parent runs, not after.
On the ERA5 route that means `--history-interval 900` in step 1.

## Rendering the child

`gpuwm render` handles the child's wrfouts like any other run's, and
sub-hourly cadences render exactly: every frame carries its precise
`valid_..._lead_...` stamp, so a 15-minute child cadence never rounds to a
fake whole hour. To see what downscaling changed, render the parent domain
and the child into separate directories and compose labeled pair sheets:

```bash
gpuwm render --pair out/parent/png out/child/png --out out/compare
```

## Known limits

- **No vertical remapping.** The child keeps the parent's eta levels, and
  terrain is SINT-inherited from the parent rather than rebuilt at the
  child's resolution. Downscaling a coarse-terrain source gives the child
  that coarse terrain at a fine grid spacing.
- **A full-physics ERA5 child is not reachable on this release** -- see
  Route 1, step 8.
- **`gpuwm downscale --card` defaults to `24gb`** while `gpuwm domain`
  measures the local card.
- **A single-domain `gpuwm domain` emission sets
  `restart_interval_s = 0.0`**, so it produces a parent that cannot be
  downscaled until you edit it.
- One child per invocation; nest the workflow by downscaling the
  downscaled run's own archive.
- Stock-WRF parents work (history plus `namelist.input` as physics
  evidence) with the same explicit-cadence contract; a point-mode child of
  a WRF parent needs `--child-config`, because a WRF namelist carries no
  complete gpuwm run configuration.
- The child runs the same bound boundary-clock semantics as the production
  nest tree (WRF's `dtbc` recurrence), and its restarts record that
  identity, so checkpoints cannot silently mix with legacy-clock
  trajectories.

Full contract details:
[native-offline-child-contract.md](../native-offline-child-contract.md).
