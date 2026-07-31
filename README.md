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
- Runs WRF v4.6.1-transcribed physics: 5 microphysics schemes, YSU and
  MYNN PBL, Noah / Noah-MP / RUC land surface, RTE+RRTMGP and legacy
  RRTMG radiation, Kain-Fritsch cumulus
  ([PHYSICS.md](docs/public/PHYSICS.md)).
- Initializes directly from ERA5, GFS, or HRRR with a built-in fetch
  front door and a fail-closed Rust GRIB decode layer -- no WPS, no
  `real.exe` ([DATA.md](docs/public/DATA.md)).
- Sizes domains to your GPU with a measured VRAM model
  (`gpuwm domain`; [HARDWARE.md](docs/public/HARDWARE.md)).
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

Then a first forecast:

```bash
# 1. Size a nest ladder to your card. Ends by printing the exact
#    fetch/check/run commands for what it just wrote.
gpuwm domain --point 35.3,-97.5 --card 24gb --cycle 1999-05-03T12 \
  --hours 6 --out configs/myarea.toml

# 2. Get data (the wizard printed this line with the area filled in)
gpuwm fetch --source gfs --cycle ... --area ... --out data/myarea

# 3. Initialize. GFS/HRRR go through the native front door; ERA5 runs
#    straight from the config.
rw-wps --source gfs ...

# 4. Run
gpuwm run configs/myarea.toml --outdir out/myarea
```

`gpuwm doctor` prints one line per item with the command that closes
each gap; `gpuwm doctor --explain` prints the full remedy block for
each, with the evidence behind it. Every command in this project takes
`--explain` and means the same thing by it.

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
one.  The `tools/` runners write a different file: the domain-tree
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
| PBL / surface layer | YSU + MM5 (classic); MYNN PBL + MYNN surface layer (coupled pair) |
| Land surface | Noah (4-layer), Noah-MP, RUC (9-level) |
| Radiation | RTE+RRTMGP (default); legacy RRTMG (WRF 4/4 transcription, verification tier); Dudhia SW |
| Cumulus | Kain-Fritsch (outer domains) |
| Data | ERA5 (CDS), GFS 0.25-deg (NOMADS), HRRR (NOMADS or AWS S3, incl. a live-cycle `--wait-for` mode) all initialize a run; GDAS 0.25-deg (NOMADS) is **fetch and decode only through f009 -- no initialization route** (`rw-wps --source gdas` refuses). Fail-closed Rust GRIB bridges; `gpuwm fetch` download front door. Plus an experimental, not-yet-stock-WRF-gated 20CRv3 ensemble-member route for GRIB2 files you supply yourself (no fetch route) -- see [DATA.md](docs/public/DATA.md) |
| Domains | `gpuwm domain` wizard: point + card -> sized experiment TOML (16/24/32 GiB tiers) |
| Products | `gpuwm render`, two engines: vendored Rusty Weather renderer (default when built), whose vendored catalog carries 324 entries; the runtime lister enumerates 151 of them as implicit-render candidates per file (the rest are explicit-opt-in ensemble/probabilistic families) -- reflectivity composite/1 km, surface T/Td/RH/MSLP/wind/PWAT/cloud-cover families, the 200-850 mb isobaric charts (height/temp/dewpoint/RH/absolute-vorticity + winds), CAPE/CIN/SRH/shear/STP severe suite, heavy ECAPE family (`--heavy`), and multi-hour windowed accumulations -- everything a file's stored fields prove out renders (measured on the committed 3 km UH-smoke case: 58/58 on a single frame, 238 renders / 0 failures across its four-frame store, transcripts retained in the development tree under `evidence/render-receipts/`; `--list-products` prints the per-file verdict with a field-level reason for every unavailable row), with coast/state/county basemaps and sub-hourly leads stamped; matplotlib fallback (composite reflectivity, T2, 10 m wind, accumulated precipitation); `--pair A B` composes two runs' PNGs into labeled comparison sheets |
| Lifecycle | `check` (input + VRAM preflight), `run`, `resume`, restart checkpoints, failure capsules |
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
- **No data assimilation.** Cold starts from public analyses only; no
  observation ingest, no cycling, no ensemble machinery.
- **Data routes.** All three sources drive ArWen GPU forecasts; they
  differ in which door they use AND in which command runs the
  forecast. ERA5 uses the config door: `[case_data]` in the experiment
  TOML, read directly by `gpuwm run`. GFS and HRRR use the preprocessor
  door: `rw-wps` converts them to `wrfinput`/`wrfbdy`, which drive
  unchanged stock WRF and `gpuwm downscale` -- and which reach the
  ArWen GPU loop through
  `tools/prepared_single_domain_forecast.py` (one domain) or
  `tools/prepared_domain_tree_forecast.py` (a nest ladder), **not**
  through `gpuwm run`, which refuses a config with no `[case_data]`
  table. The complete GFS command sequence, in the order that works,
  is [FIRST-LIGHT.md § 3a](docs/public/FIRST-LIGHT.md). Each step
  prints the next one with its digests filled in.
  HRRR remains CONUS (Lambert) only; worldwide points use GFS or ERA5,
  both global.
- **Verification depth.** One case (3 April 1974, ERA5, four domains to
  500 m) is deeply validated against WRF v4.6.1; other configurations
  inherit component-level evidence only. Physics options carry explicit
  per-option maturity labels ([PHYSICS.md](docs/public/PHYSICS.md)).
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
- [Driving stock WRF](docs/public/WRF-INTEROP.md)
- [Install and verify](docs/install.md)
- [CLI reference](docs/cli-reference.md)
- [Arbitrary but verified GRIB adapters](docs/arbitrary-verified-adapters.md)
- [What `gpuwm adapt` validates, and what it trusts](docs/adapt-validation-contract.md)
- [Migrating from WPS](docs/migrating-from-wps.md)
- [Community support matrix](docs/community-support-matrix.md)

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
