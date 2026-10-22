# ArWen

ArWen is a GPU implementation of the WRF-ARW model family, with CUDA
forecast kernels and Rust data paths for weather inputs and products.
The Python package is `gpuwm`: `gpuwm go` takes a configuration from
analysis inputs through a regional forecast to rendered products.
WRF-exact results are demonstrated in strict mode for one 1-hour,
3 km configuration against stock scalar WRF 4.6.1 on glibc 2.39;
the [verification scope](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/README-EVIDENCE.md#wrf-comparison)
names the grid and physics used.

## What it does

- **Regional forecasts:** `gpuwm domain --source` selects GFS, HRRR, ERA5, RAP, GEFS and other registered sources; `gpuwm sources --json` lists their routes.
- **Source mappings:** the [source table](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/SOURCES.md) and [mapped-source guide](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/MAPPED-FORECAST.md) describe the inputs each route needs.
- **Nested grids:** `gpuwm domain --ladder 12-3` creates a nest; `[experiment] feedback = 1` selects the experimental two-way path.
- **Moving nests:** `gpuwm cyclone-setup` builds a cyclone configuration with a following nest; the [cyclone guide](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/CYCLONE-SETUP.md) covers track selection and sizing.
- **Ensembles:** `gpuwm go CONFIG --members N --recipe time-lagged` uses earlier source cycles; `--restart-roster JSON` continues the original members from checkpoints.
- **Products:** `gpuwm render RUN --series --products all` draws maps with the Rust renderer; `gpuwm export-grib2 RUN --out DIR` writes surface and pressure-level GRIB2.
- **Restart:** `gpuwm resume CONFIG --outdir RUN` finds a valid checkpoint; `gpuwm go CONFIG --restart CHECKPOINT` continues into a new output folder.
- **More than one GPU:** `gpuwm go CONFIG --devices N` splits supported grids into resident slabs; [multi-GPU operation](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/MULTIGPU.md) describes the supported combinations.

## Get ArWen

| Download | Requirements |
| --- | --- |
| [Windows desktop: GUI and TUI](https://github.com/FahrenheitResearch/arwen/releases/download/v2.9.0/ArWen-Desktop-1.0.10-Windows-x64.zip) | x86-64 Windows; compatible NVIDIA GPU and driver; internet for first setup |
| [Linux desktop: GUI and TUI](https://github.com/FahrenheitResearch/arwen/releases/download/v2.9.0/ArWen-Desktop-1.0.10-Linux-x64-pip.tar.gz) | x86-64 Linux, glibc 2.39 or newer, such as Ubuntu 24.04; Python 3.11 or newer; X11 or Wayland with working OpenGL |
| [Integration kit](https://github.com/FahrenheitResearch/arwen/releases/download/v2.9.4/ArWen-2.9.4-Integration-Kit.zip) | Developer guide, CLI/plan examples, catalogs, and client design notes |

The desktop archives contain the applications, native map library and map assets;
Windows includes a graphical setup launcher that downloads a private Python and
CUDA runtime on first use. The Python package `gpuwm` (below) carries the engine,
native processing tools and TUI in its Windows and Linux platform wheels. The
[release notes and checksums](https://github.com/FahrenheitResearch/arwen/releases/tag/v2.9.4)
list the exact artifacts and qualification records.

## Get running

Local forecasts need a compatible NVIDIA GPU and driver; the runs below used one RTX 5090 with 32 GiB VRAM.
Use Python 3.11 or newer on Linux x86-64 or Windows x86-64.
The platform wheels include the native tools; the desktop GUI is a separate download.

### Linux

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'gpuwm[all-cu12]==2.9.4'
gpuwm fetch-tables   # optional: stage Thompson tables ahead of the first run
gpuwm fetch-geog
gpuwm doctor
```

### Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install 'gpuwm[all-cu12]==2.9.4'
gpuwm fetch-tables   # optional: stage Thompson tables ahead of the first run
gpuwm fetch-geog
gpuwm doctor
```

Activating the environment puts `gpuwm` on PATH, and every command the
program prints starts with it; `gpuwm doctor` checks that.
`gpuwm setup --with-geog` runs the same staging (native tools, physics
tables and geography) as one resumable command.

`fetch-geog` sets up the global geography every forecast builds its terrain,
land use and soil from, once per computer. It downloads about 2.7 GB and
takes about 34 GB of disk once unpacked, including the land-use datasets
`gpuwm mesh` builds MPAS statics from. Every default source is permissively
licensed (the soil categories are SoilGrids250m, CC BY 4.0; the deep-soil
temperature is built on your machine from ERA5). Add `--list` to see each
dataset and its size without downloading. `--datasets wrf` stages only what
regional forecasts read (about 2.6 GB, about 28 GB unpacked); `gpuwm doctor` then
reports the mesh geography as missing. Without it a forecast is refused at
its start.

`gpuwm doctor` reports each remedy as a command or a `#` comment explaining
the manual step.
For CUDA 13, offline installation, source builds, data setup and desktop
launchers, use the [setup guide](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/SETUP.md).

For a developer installation from a source checkout, use `bash install.sh` on
Linux or `powershell -File install.ps1` on Windows. Source installations build
the native components and need the corresponding build tools. They make the
environment on free-threaded Python 3.14t, using a `python3.14t` already on
PATH or installing one with uv, because multi-card forecasts run about twice
as fast on it; `--python` (`-Python` on Windows) picks another interpreter. The explicit
`bash install.sh` invocation also works when an extracted source archive has
lost executable permission bits; use it if `./install.sh` reports
`Permission denied`.

Choose an area and write a three-hour GFS configuration:

```bash
gpuwm domain --point 37.5,-90 --point-extent-km 1500 --root-dx 12 --source gfs --cycle latest --hours 3 --out forecast.toml
```

Run the forecast with one command:

```bash
gpuwm go forecast.toml
```

The command fetches the inputs, prepares the domain, runs the model and
writes forecast files, maps and diagnostics in the run folder it prints.
Each launch gets a timestamped folder under `forecast-go` by default.
Use `gpuwm go forecast.toml --dry-run` to inspect the plan before fetching.

## Examples

These examples use the configuration above. Replace `RUN` with the exact
run folder printed by `gpuwm go`; use separate output roots for separate runs.

### Three members from GFS cycles

```bash
gpuwm go forecast.toml --members 3 --recipe time-lagged --outdir ensemble
```

This runs three source trajectories over the same forecast window and
writes ensemble products. Add `--keep-member-files` to retain full histories
for individual members.

### A parent and a 3 km nest

```bash
gpuwm domain --point 37.5,-90 --card 32gb --ladder 12-3 --source gfs --cycle latest --hours 3 --out nested.toml
gpuwm go nested.toml
```

The generated configuration carries the parent and child grids.
See [configuration](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/CONFIGURATION.md) for feedback and grid settings.
Sub-kilometer storm diagnostics are not resolved tornado dynamics;
the [scale note](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/RESOLVED-SCALE.md) explains their interpretation.

### Continue an interrupted run

```bash
gpuwm resume forecast.toml --outdir RUN
```

Use the original configuration and run folder. The restart identity check
rejects incompatible configurations, and the default retention is one complete
checkpoint set.

### Render a saved forecast

```bash
gpuwm render RUN --series --products t2,refl --out maps
```

This writes 2 m temperature and composite-reflectivity maps from saved frames.
`--products all` requests the full catalog; the render summary records products
that the available fields or forecast times cannot supply.

### Export forecast fields

```bash
gpuwm export-grib2 RUN --out grib2
```

The exporter writes surface and pressure-level files. The
[GRIB2 guide](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/grib2-export.md) describes fields and definition choices.

## Measured runs

These are individual Linux acceptance runs of the 2.9.2 candidate on one
RTX 5090, recorded on 2026-10-17. Times are complete command wall times,
including products, with geography installed and shared input caches as noted
in the [measurement record](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/README-EVIDENCE.md#measured-runs).

| Configuration | Forecast length | Command | Wall time |
| --- | --- | --- | --- |
| GFS, 12 km, 1500 km extent | 3 h | `gpuwm go` | 88.151 s |
| NCAR CONUS, 12 km, supplied WRF inputs | 12 h | `gpuwm run --wrfinput` | 137.915 s |
| NCAR CONUS, 12 km, supplied WRF restart | Final 1 h | `gpuwm run --restart-from` | 25.428 s |

These timings exclude installation and geography download. They describe
those configurations and cache states, not a throughput guarantee for other grids.

Whole-step driver runs of the 2.9.3 candidate, recorded on 2026-10-18 and
2026-10-19 (case B on one RTX 4090, no products; case A on
both cards; the HRRR-shape case on one RTX 5090 with products on):

| Configuration | Forecast length | Wall time |
| --- | --- | --- |
| Nested 24 h case A, 1600 parent and 4800 child steps | 24 h | 111.0 s process, 101.5 s integration on an RTX 5090 (products on cost 0 s); 144.1 s on the RTX 4090, 171.2 s there before the driver |
| Nested 24 h case B, 4800 steps | 24 h | 84.8 s (111.8 s before the driver) |
| HRRR-shape, 796 x 636 x 49 at 3 km, dt 15 s, 5760 steps, products on | 24 h | 1944.8 s whole command (2092.5 s before), 1495.1 s integration |

## Documentation

- [CLI manual](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/CLI-USER-MANUAL.md) and [option reference](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/CLI-OPTIONS.md).
- [WRF input and restart files](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/WRF-INPUTS.md).
- [Earth2Studio export](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/EXPORT-EARTH2.md): grids, consumer channels and GRIB2 fields.
- [Physics choices](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/PHYSICS.md), [hardware](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/HARDWARE.md) and [output variables](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/OUTPUT-VARIABLES.md).
- [Verification](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/VERIFICATION.md) and [release changes](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/CHANGELOG.md).
- [Desktop and terminal workspace](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/GUI.md).
- [Stage contract](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/PIPELINE-STAGES.md) for running preprocessing,
  the simulation and rendering on your own terms.
- [Smoke, dust and air quality](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/SMOKE-AND-AIR-QUALITY.md): wildfire
  smoke, dust, sea salt, carbon and CAMS air-quality species are an opt-in
  product, off unless a configuration names `chem_sets`; the worked example is
  in [CONFIGURATION.md](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/CONFIGURATION.md).

Report problems through [GitHub issues](https://github.com/FahrenheitResearch/arwen/issues)
with the version, platform, command and error text.
`gpuwm report RUN --output diagnostics.zip` collects a diagnostic bundle;
review it for private information before sharing, following the
[reporting guide](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/docs/public/REPORTING-A-PROBLEM.md).

ArWen is research software; use official meteorological services for safety warnings.

The engine is licensed under [Apache-2.0](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/LICENSE).
Third-party code and data retain their terms; see [NOTICE](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/NOTICE) and [provenance](https://github.com/FahrenheitResearch/arwen/blob/v2.9.4/PROVENANCE.md).
