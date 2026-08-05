# ArWen

ArWen is an independent, GPU-native implementation of a WRF-ARW-class
regional atmospheric model. It is not affiliated with or endorsed by
NCAR or UCAR. The name is a wordmark for "the ARW solver, GPU-native";
the Python package is currently named `gpuwm`.

> **Safety.** ArWen is a research and educational tool. It is never a
> substitute for official forecasts and warnings from your national
> meteorological service. Do not use it to make safety decisions.

Regional numerical weather prediction at convective scale has mostly
required institutional clusters, which means most of the world runs on
global-model guidance at 10-25 km. ArWen's aim is to put a verified,
kilometer-scale limited-area model on a single consumer GPU: pick a
point, size a nest ladder to your card, pull public analysis data, and
run a 1 km (or 500 m) simulation of your own area on hardware you own --
especially in places where no national convection-permitting model
exists.

![Paired CPU (WRF v4.6.1) and GPU (ArWen) composite reflectivity, same
initial state, same physics, +3 h](docs/public/img/hero-cpu-vs-gpu-reflectivity.png)

*Above: unchanged WRF v4.6.1 (left of each pair) and ArWen (right),
same case, matched physics. At +3 h on the 3 km domain the two models
agree to composite-reflectivity correlation 0.985, with the squall line
in the same place with the same structure and >=20 dBZ echo area
matching to 3 pixels in 14,227; the numbers behind this figure are in
[VERIFICATION.md](docs/public/VERIFICATION.md).*

## What it does

- Integrates a WRF-ARW-class compressible nonhydrostatic core (RK3,
  split-explicit acoustics, one-way static nesting) in FP32 on CUDA.
- Runs WRF v4.6.1-transcribed physics: 5 microphysics schemes, YSU,
  MYNN and scale-aware Shin-Hong PBL, Noah / Noah-MP / RUC land
  surface, RTE+RRTMGP and legacy RRTMG radiation, Kain-Fritsch cumulus
  ([PHYSICS.md](docs/public/PHYSICS.md)).
- Initializes directly from ERA5, GFS, or HRRR with a built-in fetch
  front door and a fail-closed Rust GRIB decode layer -- no WPS, no
  `real.exe` ([DATA.md](docs/public/DATA.md)).
- Sizes domains to your GPU with a measured VRAM model
  (`gpuwm domain`; [HARDWARE.md](docs/public/HARDWARE.md)).
- Runs independent forecasts concurrently on distinct GPUs with isolated
  output, temp, CuPy-cache, and driver-JIT-cache paths (`gpuwm multi-run`;
  [HARDWARE.md](docs/public/HARDWARE.md#independent-runs-on-multiple-gpus)).
- Renders reflectivity, T2, 10 m wind, and precipitation products;
  checkpoints and resumes; re-runs finer nests offline from archived
  parents ([DOWNSCALE.md](docs/public/DOWNSCALE.md)).
- Feeds unchanged stock WRF: the same preprocessor (`rw-wps`) emits
  `wrfinput_d0N`/`wrfbdy_d01` that WRF v4.6.1 has accepted and
  integrated, serial and MPI ([WRF-INTEROP.md](docs/public/WRF-INTEROP.md)).

Measured on one RTX 5090 (Windows 11, driver-default WDDM): a 6 h
forecast on a 250x200x49 12-km domain with a full physics suite
(Morrison two-moment microphysics, RTE+RRTMGP radiation, YSU, Noah,
Kain-Fritsch) completed in **3.6 minutes of wall time** using ~6.3 GiB
of device memory; GFS input acquisition took 9.7 s and rendering 16
product PNGs took 2.6 s (first-time-user acceptance transcript,
2026-07-29).

## Install

```bash
pip install 'gpuwm[all]'   # gpu + render extras
gpuwm setup                # prebuilt Rust decoders + the externalized physics tables
```

`gpuwm setup` runs `gpuwm fetch-bridges` then `gpuwm fetch-tables`,
verifies every artifact against the SHA-256 pins packaged in the wheel,
and finishes with the `gpuwm doctor` summary. It is re-run safe: what
is already staged and pin-valid is verified and skipped. The ~16 GB
WPS_GEOG static tree is not part of it -- `gpuwm setup --with-geog`
opts in and prints the size before anything downloads, or run
`gpuwm fetch-geog` later.

To upgrade later, `gpuwm update` prints the exact command for your
environment -- the distribution that provides this install and the
interpreter running it -- and nothing else: it downloads nothing and
replaces nothing, because rewriting a package's files underneath the
process importing them is how a half-upgraded install happens. Run the
line it prints from your shell. The staged bridges and physics tables
live under `~/.gpuwm`, outside the wheel, so an upgrade does not
re-download them.

Published PyPI wheels and sdists carry bridge pins generated from that
release's exact native bundles. GitHub's automatic source `.zip` and
`.tar.gz` archives are intentionally unpinned: install the PyPI artifact to
use `gpuwm fetch-bridges`, or clone the tag and build the vendored Rust
workspaces when installing from source.

Then a first forecast -- two commands, no placeholders:

```bash
# 1. Size a domain to your card at your point of interest.  The
#    emitted TOML records the cycle and fetch area, so nothing has to
#    be copied by hand.
gpuwm domain --point 35.3,-97.5 --card 24gb --ladder 12 \
  --source gfs --cycle latest --hours 6 --out configs/myarea.toml

# 2. Run the whole chain: fetch -> initialize -> GPU forecast -> PNGs.
#    (`--dry-run` prints the six underlying commands instead.)
gpuwm go configs/myarea.toml
```

Bare `gpuwm domain` at a terminal asks four questions and ends by
printing that exact `gpuwm go` line.  Nest ladders (12-3, 12-3-1, ...)
and the ERA5 route run stage by stage instead of through `go`; the
wizard's closing block prints each next command for the config it just
wrote, and the full walkthrough is
[FIRST-LIGHT.md](docs/public/FIRST-LIGHT.md).

For an uploading HRRR cycle, `gpuwm stream PLAN.toml` runs bounded,
crash-resumable hourly restart-extend legs. Each leg seals a forcing prefix,
rebuilds the prepared hierarchy, resumes the preceding tree checkpoint, and
publishes a hash-linked PASS timeline; forcing is never injected into a live
model process. See [Chunked forecast streaming](docs/public/STREAMING.md) for
the plan schema, GPU ownership, disk/cache accounting, and timeline semantics.

### Route status

Before every release cut, a route-coverage gate installs the built
wheel on a machine that has never seen this project and drives every
route this documentation advertises, filling each printed placeholder
from what the run itself printed rather than from knowledge of the
source. What that gate found for 1.4.0 is below. Routes marked Unreleased are
newer and report their separate campaign state explicitly.

**Supported** means the gate was green end to end from a wheel.
**Experimental** means the route works and has a named rough edge --
the edge is in the last column rather than in your way.

| Route | Status | What the gate found |
|---|---|---|
| Install: `pip install 'gpuwm[all]'` -> `gpuwm setup` -> `gpuwm doctor` | **Supported** | Green on a cold machine from this release's pinned bundle. |
| Parts: `gpuwm fetch-bridges`, `gpuwm fetch-tables`, re-run | **Supported** | Green, and re-running either is safe: what is already staged and pin-valid is verified and skipped. |
| `gpuwm fetch-geog` (the WPS_GEOG static tree) | **Supported** | Green. |
| GFS, single domain, through `gpuwm go` | **Supported** | Green: one command from fetch to PNGs. |
| GFS, single domain, stage by stage | **Supported** | Green through the forecast. |
| GFS, nest ladder -> the domain-tree runner | **Supported** | Green through the forecast. No page documents an authority-materialization step for the tree runner; the single-domain page's step does not transfer, and this route does not need it. |
| ERA5, from a config | **Supported** | Green: request template -> validate -> check -> run -> render. |
| `gpuwm import-namelist` (an existing WRF namelist pair) | **Supported** | Green on a real pair. Bring your own: nothing in the product emits a `namelist.input` to practise the importer on, and no example pair ships. |
| `gpuwm certify` / `gpuwm dual-run` | **Supported** | Green. |
| `gpuwm multi-run PLAN.toml` (Unreleased) | **Experimental** | An earlier candidate-wheel two-GPU campaign completed simultaneous HRRR/WSM6 and GFS/Thompson forecasts on distinct physical GPUs. The final post-1.4 wheel has not repeated that campaign, so no final-wheel readiness or performance claim is made. |
| `gpuwm stream PLAN.toml` (Unreleased) | **Experimental** | An earlier candidate wheel completed the bounded HRRR 12Z f001 -> 13Z f001 seam with initial and final health green; only 13Z was a fresh availability observation. The final post-1.4 wheel has not repeated it, and no within-cycle f001..f004 or combined HRRR-plus-GFS claim is made. |
| HRRR, single domain: fetch -> native preparation | **Experimental** | Fetch and the preparation are green from a wheel. The handoff to the forecast is not: the wizard's closing block says the preparation prints the forecast stage's arguments, and it does not print them -- so reaching a finished single-domain forecast means assembling that `tools/hrrr_single_domain_benchmark.py` command by hand out of the preparation's output tree. The nest-ladder route below needs no such step. |
| HRRR, nest ladder: preparation -> hierarchy -> tree forecast | **Supported** | Green end to end through the forecast: domain, fetch, preparation, hierarchy and the tree runner, each command copied from the one the previous stage printed. One value in the printed tree-runner command is not printed by any stage -- `--experiment-config-sha256`; the command names the file and you hash it yourself. |
| `gpuwm downscale` (an offline finer nest from an archived run) | **Experimental** | The dry run is green. Derived mode refuses when the child config enables surface physics and no child-grid surface source was given; the refusal names `--child-surface-from`. |
| `gpuwm enprod` (ensemble products) | **Experimental** | Green over an ensemble, and `--make-fixture` writes a synthetic one so you can try the suite. Member generation is undocumented: no public page prints a runnable command that produces members. |
| `gpuwm adapt` (an arbitrary but verified GRIB2 adapter) | **Experimental** | `--skeleton` is green and names its own gaps. Authoring needs a descriptor you complete by hand; the unfilled scaffold is refused as a scaffold rather than accepted. |

`gpuwm doctor` prints one line per item with the command that closes
each gap; `gpuwm doctor --explain` prints the full remedy block for
each, with the evidence behind it. Every command in this project takes
`--explain` and means the same thing by it.

Doctor reports the estate *and* the paths a run resolves: for each data
route, the exact decoder its preparation will launch, the byte transport
its fetch will pick, the identity its receipt will bind, and whether its
entry points import from a directory that is not a repository.
`gpuwm doctor --source hrrr` narrows the report to one route. That half
exists because a wheel install once read "no gaps" and then refused,
one command later, on a path the report had never resolved.

When something has already gone wrong, `gpuwm report` is the other half:
run it in the run directory and it collects the receipts, the failure,
the logs, this install's identity, the card and the free space into one
readable zip to attach to an issue. It is anonymous by construction --
usernames, home paths, hostnames, addresses and credential-shaped
strings are replaced by class placeholders, and your domain and dates
are kept because they are science, not identity. `gpuwm report
--dry-run` prints the manifest without writing anything, so you can see
what you would be sending first:
[reporting a problem](docs/public/REPORTING-A-PROBLEM.md).

The longer path -- the install scripts, the manual steps, and what each
piece is -- is below.

### The install scripts

One command from the checkout root. `install.sh` / `install.ps1`
create `.venv`, install the `[gpu,render]` extras, stage the
externalized Thompson tables (`gpuwm fetch-tables`: a one-time
~243 MiB release-asset download from a checkout, SHA-256-verified
before install, skipped when already present; `--no-fetch-tables` /
`-NoFetchTables` defers it), offer to install rustup when `cargo` is
missing (they ask first; `--yes` / `-Yes` consents), build the
vendored Rust GRIB bridges and the production render engine offline
(`--no-render` / `-NoRender` skips the renderer build), and finish
with `gpuwm doctor`. Re-running either script is safe: an existing
`.venv`, staged tables, and built bridges are reused.

```bash
git clone https://github.com/FahrenheitResearch/arwen gpuwm && cd gpuwm
bash install.sh         # PowerShell: .\install.ps1
```

`bash install.sh` is the universal form and works regardless of how
your checkout landed the file's mode bit. `./install.sh` works too;
if it ever answers `Permission denied`, use the `bash` form above.

The same scripts also run standalone -- POSIX:

```bash
curl -fsSL https://raw.githubusercontent.com/FahrenheitResearch/arwen/main/install.sh | sh
```

Windows (PowerShell):

```powershell
iwr -useb https://raw.githubusercontent.com/FahrenheitResearch/arwen/main/install.ps1 | iex
```

When piped like this, the script clones the repository into `./gpuwm`
(set `GPUWM_REPO_URL` to clone from a fork or mirror instead).

You need Python 3.11+ and git; for GPU runs, an NVIDIA card with
CUDA 12.x/13.x, field-verified through 13.2 driver stacks on sm_89 by
two independent nodes: the toolkit works out of the box with the
`cupy-cuda12x` pin, because minor-version compatibility plus CuPy's
system-NVRTC discovery covers it -- measured on a Linux RTX 4070 and a
4090, 2026-07-30. The
manual steps, if you prefer them:

POSIX:

```bash
git clone https://github.com/FahrenheitResearch/arwen gpuwm && cd gpuwm
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gpu,render]'
gpuwm fetch-tables
gpuwm fetch-geog       # WPS_GEOG static tree: ~1.3 GB down, ~16 GB unpacked
(cd tools/grib1_bridge && cargo build --release --locked --offline)
(cd tools/rustwx && cargo build --release --locked --offline)
gpuwm doctor
```

Windows (PowerShell):

```powershell
git clone https://github.com/FahrenheitResearch/arwen gpuwm; cd gpuwm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[gpu,render]'
gpuwm fetch-tables
gpuwm fetch-geog       # WPS_GEOG static tree: ~1.3 GB down, ~16 GB unpacked
cd tools\grib1_bridge; cargo build --release --locked --offline; cd ..\..
cd tools\rustwx; cargo build --release --locked --offline; cd ..\..
gpuwm doctor
```

**`pip install gpuwm` needs two more commands before it can read
weather data.** The wheel ships no compiled Rust, and every GRIB decode
-- ERA5, GFS, GDAS, HRRR, 20CRv3 -- goes through the fail-closed Rust
bridges built from `tools/grib1_bridge`, which a wheel does not carry.
Two commands close that gap:

```bash
pip install gpuwm
gpuwm setup             # runs both fetch steps below, then the doctor summary
```

`gpuwm setup` is a wrapper over the two commands, which still stand on
their own:

```bash
gpuwm fetch-bridges     # prebuilt decoders, renderer and fetch backbone
gpuwm fetch-tables      # the two externalized Thompson tables
gpuwm doctor            # what is still missing, and the command for each
```

`gpuwm fetch-bridges` downloads one bundle for your platform -- the
five GRIB decoders, the CPU preprocessing library, the Rust fetch
backbone and the batch renderer -- and verifies every artifact against
the SHA-256 pins packaged in the wheel before staging it into
`~/.gpuwm/bridges`, where the resolver already looks. A release
publishes bundles for Windows x86-64 and Linux x86-64; anywhere else,
and on any release that published none, the command says so by name and
the clone-and-`cargo build` route above is the answer. `--from DIR`
stages the same bundle offline. `gpuwm doctor` prints whichever of the
two remedies is true for your machine.

`[gpu]` installs CuPy (required by `gpuwm check`/`run` and the sizing
wizard); `[render]` installs the pinned `wrf-rust` package for
`gpuwm render`'s matplotlib fallback engine. The `tools/rustwx` build
is the production render engine (the vendored Rusty Weather renderer:
coast/state/county basemaps over a 324-entry vendored product catalog,
151 of whose products are implicit-render candidates on any file) --
`gpuwm render` uses it by default the moment it is built, and works
without it. `gpuwm doctor` then checks
each piece for real rather than by presence: it imports CuPy and the
render stack in subprocesses, probe-executes every bridge and the
renderer, loads the CPU library and reads its ABI,
hash-validates the staged Thompson tables, parses the Noah tables,
and requires each WPS_GEOG dataset's index file -- anything it can only
see (not prove) is labeled `present` instead of `ok`, and every gap
prints a remedy whose every line is a command or a `#` comment, so the
block survives being pasted whole. Most are exact commands; a few
cannot be, and say so rather than inventing one -- an unset
`GPUWM_CASE_DATA_ROOT` needs a path only you know. Details, wheel caveats, and the
sealed archives: [docs/install.md](docs/install.md).

## First light

The condensed path from nothing to pictures (full walkthrough with
measured timings: [FIRST-LIGHT.md](docs/public/FIRST-LIGHT.md)):

```bash
# 1. Size a domain ladder to your card at your point of interest
gpuwm domain --point 35.3,-97.5 --card 24gb --cycle 1999-05-03T12 \
  --hours 6 --out configs/myarea.toml

# 2. Get data (ERA5 shown; the wizard prints the exact command)
gpuwm fetch --source era5 --cycle 1999-05-03T12 --hours 6 \
  --area 25.4,-112.0,44.7,-83.0 --out data/myarea

# 3. Preflight, run, render
gpuwm check  configs/myarea.toml
gpuwm run    configs/myarea.toml --outdir out/myarea
gpuwm render out/myarea/wrfout_d01_* --out out/myarea/png
```

Live progress is `run-progress.json` in the output directory (atomic,
schema `gpuwm.run-progress/v1`); restart checkpoints are written every
`restart_interval_s` and `gpuwm resume` continues from the newest valid
one.  That is this route, the `[case_data]` route.  The prepared
single-domain forecaster writes no checkpoints at any
`restart_interval_s` -- `gpuwm check` names the limitation before you
spend the run; use a multi-domain config or a `[case_data]` experiment
when you need to resume.  The `tools/` runners write a different file: the domain-tree
route writes `<outdir>/evidence/progress.json` and the single-domain
runners write `<outdir>/progress.json`.

## What the output looks like

![ArWen 500 m composite reflectivity, 3 April 1974 17:30Z, +5h30m
lead](docs/public/img/showcase-d04-500m-reflectivity-1730z.png)

*Discrete supercells with 55-60 dBZ cores on the 500 m nest at a
sub-hourly valid time (+5 h 30 m) -- rendered by the built-in
production engine (`gpuwm render`).*

![Significant tornado parameter, 3 km domain,
18Z](docs/public/img/showcase-d02-stp-18z.png)

*Significant Tornado Parameter on the 3 km domain at +6 h from the same
run. The full catalog is 324 products (severe suite, isobaric charts,
surface fields, accumulations); `gpuwm render --list-products` shows
what any given file supports. The compute-expensive ECAPE family is
opt-in via `--heavy`.*

## Feature matrix

| Area | Shipped in this release |
|---|---|
| Dynamics | WRF-ARW-class RK3 split-explicit core, FP32, CUDA; one-way static nests on Lambert-conformal, Mercator, or polar-stereographic grids |
| Microphysics | Kessler, WSM6, Thompson (default; WRF tables SHA-256-pinned -- the two largest ship as release assets and `gpuwm fetch-tables` stages whichever are absent, run automatically by install), Morrison 2-moment, NSSL 2-moment |
| PBL / surface layer | YSU + MM5 (classic); MYNN PBL + MYNN surface layer (coupled pair); Shin-Hong scale-aware, the gray-zone option, with either MM5 surface layer |
| Land surface | Noah (4-layer), Noah-MP, RUC (9-level) |
| Radiation | RTE+RRTMGP (default); legacy RRTMG (WRF 4/4 transcription, verification tier); Dudhia SW |
| Cumulus | Kain-Fritsch (outer domains) |
| Data | ERA5 (CDS), GFS 0.25-deg (NOMADS), HRRR (NOMADS or AWS S3, incl. a live-cycle `--wait-for` mode) all initialize a run; GDAS 0.25-deg (NOMADS) is **fetch and decode only through f009 -- no initialization route** (`rw-wps --source gdas` refuses). Fail-closed Rust GRIB bridges; `gpuwm fetch` download front door. Plus an experimental, not-yet-stock-WRF-gated 20CRv3 ensemble-member route for GRIB2 files you supply yourself (no fetch route) -- see [DATA.md](docs/public/DATA.md) |
| Domains | `gpuwm domain` wizard: point + card -> sized experiment TOML (16/24/32 GiB tiers) |
| Products | `gpuwm render`, two engines: vendored Rusty Weather renderer (default when built), whose vendored catalog carries 324 entries; the runtime lister enumerates 151 of them as implicit-render candidates per file (the rest are explicit-opt-in ensemble/probabilistic families) -- reflectivity composite/1 km, surface T/Td/RH/MSLP/wind/PWAT/cloud-cover families, the 200-850 mb isobaric charts (height/temp/dewpoint/RH/absolute-vorticity + winds), CAPE/CIN/SRH/shear/STP severe suite, heavy ECAPE family (`--heavy`), and multi-hour windowed accumulations -- everything a file's stored fields prove out renders (measured on the committed 3 km UH-smoke case: 58/58 on a single frame, 238 renders / 0 failures across its four-frame store, transcripts retained in the development tree under `evidence/render-receipts/`; `--list-products` prints the per-file verdict with a field-level reason for every unavailable row), with coast/state/county basemaps and sub-hourly leads stamped; matplotlib fallback (composite reflectivity, T2, 10 m wind, accumulated precipitation); `--pair A B` composes two runs' PNGs into labeled comparison sheets |
| Lifecycle | `check` (input + VRAM preflight), `run`, `resume`, restart checkpoints, failure capsules.  **Checkpoints are written on the `[case_data]` route (`gpuwm run`) and on the multi-domain prepared route; a single-domain config with no `[case_data]` table runs on the prepared single-domain forecaster, which writes none** -- `gpuwm check` says so before the run |
| Operational HRRR | `gpuwm stream PLAN.toml`: bounded hourly chunked restart-extend with immutable leg chains, exact-hour cycle succession, independent disk-volume gates, and crash-safe replay |
| Downscaling | `gpuwm downscale`: offline finer nest from archived gpuwm or WRF history (ndown-class) |
| WRF interop | `rw-wps` emits `wrfinput`/`wrfbdy` consumed by unchanged WRF v4.6.1 (see boundaries) |
| Namelists | `gpuwm import-namelist`: WRF namelist pair -> experiment TOML with an explicit substitution report |

## Limits

Stated plainly, up front:

- **Projection and location.** Lambert conformal (both hemispheres),
  Mercator, and polar stereographic (both poles) run end to end --
  wizard, config, static build, ERA5/GFS ingest, native WRF export --
  and antimeridian-crossing domains are supported. What remains
  refused: domains containing or touching a pole (the lat-lon source
  interpolation and static-tile windowing are not pole-capable), and
  forcing footprints wider than 180 degrees of longitude.
  Latitude-longitude (cylindrical) and rotated grids stay unsupported
  and fail closed.
- **Projection maturity.** The new projections (Mercator, polar
  stereographic, southern-hemisphere Lambert) are oracle-verified and
  smoke-run verified -- transcription gates at binary64 against a
  Fortran oracle built from the pinned WRF v4.6.1
  `share/module_llxy.F`, plus short GPU smoke integrations -- not
  matched-run verified. The deep matched-run validation (the 1974
  reference family) exists for northern-hemisphere Lambert only.
- **Nesting.** Static nests. Children may start later on an exact
  parent-step and forcing-cadence seam. One-way is the supported
  default; two-way feedback (`feedback = 1`) ships as an EXPERIMENTAL
  path -- it runs, it is stamped as experimental in the run's own
  provenance, and one-way consumers refuse a feedback-modified parent.
  It feeds back dynamic state only, where WRF also feeds back hundreds
  of masked land-surface fields, so it is not a WRF-equivalent claim.
  No moving nests, no vertical refinement, no adaptive time step.
- **Precision.** The model state is FP32 (like WRF's default REAL).
  No end-to-end bit-identity with WRF is claimed anywhere; see
  [VERIFICATION.md](docs/public/VERIFICATION.md) for exactly what is
  claimed.
- **No data assimilation on the supported path.** `gpuwm run` cold-starts
  from public analyses only. v1.2 adds EXPERIMENTAL ensemble and DA
  machinery reachable **only** through experimental tools that nothing
  else calls -- `tools/ensemble_forecast.py` (perturbed members, cycling
  with an assimilation seam), `gpuwm enprod` (ensemble products), and
  `tools/da_synthetic_cycle.py` (the composition gate). None of it is on
  a certified forecast path, none of it has been calibrated against a
  verification archive, and the perturbation library imposes no mass or
  wind balance, perturbs no boundary forcing, and tapers laterally only.
  Run `python -m tools.ensemble_forecast run --help`, which prints the
  full limitation list before it does anything. See
  [PROVENANCE.md](PROVENANCE.md) for the register entry.
  The radar-DA nowcast built on that machinery
  ([quickstart](docs/da-nowcast-quickstart.md)) is EXPERIMENTAL and
  demo-grade on the same terms: UNSCORED, outside any registered
  campaign, and it says so on every figure it draws.
- **Data routes.** All three sources drive ArWen GPU forecasts; they
  differ in which door they use AND in which command runs the
  forecast. ERA5 uses the config door: `[case_data]` in the experiment
  TOML, read directly by `gpuwm run`. GFS and HRRR use the preprocessor
  door: `rw-wps` converts them to `wrfinput`/`wrfbdy`, which drive
  unchanged stock WRF and `gpuwm downscale` -- and which reach the
  ArWen GPU loop through
  `python -m gpuwm.prepared_single_domain_forecast` (one domain) or
  `python -m gpuwm.prepared_domain_tree_forecast` (a nest ladder),
  **not** through `gpuwm run`, which refuses a config with no
  `[case_data]` table. For single-domain GFS, `gpuwm go <config>` runs
  that whole sequence -- authority, fetch, front door, forecast,
  render -- so none of its digests has to be carried by hand; the
  sequence itself, and the routes `go` does not drive, are
  [FIRST-LIGHT.md 3a](docs/public/FIRST-LIGHT.md).
  HRRR remains CONUS (Lambert) only; worldwide points use GFS or ERA5,
  both global.
- **Verification depth.** One case (3 April 1974, ERA5, four domains to
  500 m) is deeply validated against WRF v4.6.1; other configurations
  inherit component-level evidence only. Physics options carry explicit
  per-option maturity labels ([PHYSICS.md](docs/public/PHYSICS.md)).
- **Resolved scale.** The innermost demonstrated grid is 500 m. That
  resolves supercell storm structure, cold pools, mesocyclone-scale
  rotation, and the environmental and morphological severe-weather
  diagnostics rendered on it -- 2-5 km updraft helicity, the
  Significant Tornado Parameter, CAPE/CIN/SRH/shear.
  It does not resolve tornado dynamics: the near-surface corner flow,
  suction vortices, or tornado-scale wind intensity, which the
  literature on tornado-like vortices places below roughly 25 m
  horizontal and 10 m vertical spacing.
  Read the STP/UH severe suite as a tornadic-supercell environment
  and mesocyclone-proxy diagnostic, not as a resolved-tornado claim:
  these are convection-permitting to sub-kilometer case studies,
  not tornado-resolving simulations.
  Sub-kilometer nests additionally sit in the PBL gray zone flagged
  in [FIRST-LIGHT.md](docs/public/FIRST-LIGHT.md) (a 3-D turbulence
  closure, SASE, is implemented but experimental -- see PHYSICS.md).
- **Platforms.** Developed and measured on Windows 11 + RTX 5090 and on
  Linux CUDA 12.x nodes. The sealed Windows archive is CPU-preprocessing
  only; Windows CUDA is exercised via the developer checkout.

## Verification

ArWen is gated against WRF v4.6.1 (commit `d66e442f`) at three levels:
bit-level kernel oracles against unmodified WRF Fortran, t=0
initialization parity, and matched-run forecast comparisons. A sample
of the measured results:

| Gate | Scope | Measured result |
|---|---|---|
| t=0 parity | 4 domains, 3 Apr 1974 case | T2 MAE 0.000 K, corr 1.000 on every domain vs the WRF initial state |
| Matched 6 h forecast | d02 (3 km), 15Z | composite refl corr 0.985; >=20 dBZ echo area within 3 pixels of WRF's 14,227 |
| Matched 6 h forecast | d03 (1 km), 18Z | T2 MAE 0.347 K; refl corr 0.715 (convective-scale chaos floor; see the page) |
| Component oracles | legacy RRTMG LW/SW engines | max ULP 0 vs the transcription oracle over the full fixture decks |
| Determinism | mid-run kill + relaunch | regenerated output frames SHA256-identical |

What these numbers mean, what is deliberately not claimed, and how to
reproduce them: [VERIFICATION.md](docs/public/VERIFICATION.md).

Consumer GeForce cards have no ECC memory, and running the forecast
twice and comparing bytes is what stands in for it. That comparison is a
transient-fault screen inside a fixed numerical environment, not an ECC
replacement: it cannot detect a fault that is identical in both runs.
What it does detect, what it does not, and the pin set that defines
"fixed environment": [DETERMINISM.md](docs/public/DETERMINISM.md).

## Documentation

- [First light walkthrough](docs/public/FIRST-LIGHT.md)
- [Verification](docs/public/VERIFICATION.md)
- [Determinism and the no-ECC dual-run screen](docs/public/DETERMINISM.md)
- [Physics options and maturity](docs/public/PHYSICS.md)
- [Configuration knobs (WRF namelist parity)](docs/public/CONFIGURATION.md)
- [Getting data](docs/public/DATA.md)
- [Hardware and VRAM sizing](docs/public/HARDWARE.md)
- [Offline downscaling](docs/public/DOWNSCALE.md)
- [Chunked forecast streaming](docs/public/STREAMING.md)
- [Driving stock WRF](docs/public/WRF-INTEROP.md)
- [Install and verify](docs/install.md)
- [Radar-DA nowcast quickstart (demo-grade, unscored)](docs/da-nowcast-quickstart.md)
- [CLI reference](docs/cli-reference.md)
- [Arbitrary but verified GRIB adapters](docs/arbitrary-verified-adapters.md)
- [What `gpuwm adapt` validates, and what it trusts](docs/adapt-validation-contract.md)
- [Migrating from WPS](docs/migrating-from-wps.md)
- [Community support matrix](docs/community-support-matrix.md)
- [Reporting a problem](docs/public/REPORTING-A-PROBLEM.md)

## Credits and provenance

ArWen was designed and directed by its author and implemented with
substantial use of AI coding agents (Anthropic's Claude, including
Claude Fable 5, with auditing by OpenAI models). All model code was
gated by verification against WRF v4.6.1 -- bit-level kernel oracles,
matched-run comparisons, and adversarial review -- rather than accepted
on generation. The verification methodology and its results are
documented in VERIFICATION.md.

The transcription authority for every WRF-derived mechanism is WRF
v4.6.1; deliberate deviations are registered in
[PROVENANCE.md](PROVENANCE.md). Radiation data files derive from AER's
RRTMG and the RTE+RRTMGP project; rendering uses the `wrf-rust`
package. See [NOTICE](NOTICE) for third-party acknowledgments.

## License

Apache License 2.0 ([LICENSE](LICENSE)). Third-party datasets, tables,
vendored components, and dependencies retain their own terms
([NOTICE](NOTICE)).
