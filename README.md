# ArWen

**Explore weather, create regional forecasts, and run them on your own hardware.**

ArWen combines a native desktop application, a terminal workspace, and a
CUDA-based atmospheric modeling engine. Use the desktop to inspect weather maps,
choose a forecast area or cyclone, review the setup, and follow your results.
Run the model on a local NVIDIA GPU or a Linux computer connected over SSH.

![ArWen Desktop showing an ERA5 temperature field](https://raw.githubusercontent.com/FahrenheitResearch/arwen/v2.7.3/docs/public/img/arwen-desktop-2.7.png)

*The Linux desktop displaying a real ERA5 field. Windows uses the same interface.*

## Get ArWen

| Download | Requirements |
| --- | --- |
| [Windows desktop: GUI and TUI](https://github.com/FahrenheitResearch/arwen/releases/download/v2.8.0/ArWen-Desktop-1.0.8-Windows-x64.zip) | x86-64 Windows; compatible NVIDIA GPU and driver; internet for first setup |
| [Linux desktop: GUI and TUI](https://github.com/FahrenheitResearch/arwen/releases/download/v2.8.0/ArWen-Desktop-1.0.8-Linux-x64-pip.tar.gz) | x86-64 Linux, glibc 2.39 or newer, such as Ubuntu 24.04; Python 3.11 or newer; X11 or Wayland with working OpenGL |
| [Integration kit](https://github.com/FahrenheitResearch/arwen/releases/download/v2.8.4/ArWen-2.8.4-Integration-Kit.zip) | Developer guide, CLI/plan examples, catalogs, and client design notes |

The desktop archives contain the applications, native map library and map assets.
Windows includes a graphical setup launcher that downloads a private Python and
CUDA runtime on first use. Linux uses the manual setup below. No Rust or C++
compiler is needed for these binary packages.

The Python package is named **`gpuwm`**. Its Windows and Linux platform wheels
include the engine, native processing tools, and TUI. The desktop GUI is the
separate download above. Check [release notes and checksums](https://github.com/FahrenheitResearch/arwen/releases/tag/v2.8.4)
for the exact artifacts and qualification records.

## Install the desktop for forecasts on your PC

### Windows

1. Download the Windows desktop ZIP and extract it into its own folder.
2. Open **ArWen.exe**. The setup window installs the matching engine and GPU
   dependencies, shows progress, and checks the local GPU.
3. When the workspace opens, choose **Create forecast**, draw an area, review
   the configuration, and run it on **Local computer**.

Python and the user-space CUDA components are managed by the application.
A compatible NVIDIA driver must already be installed. Keep the `resources`
folder beside `ArWen.exe`. Setup errors remain visible with retry and repair
controls; setup does not require terminal commands.

First setup also prepares the physics tables and global geography needed for
regional forecasts. Allow several gigabytes of downloads and at least 25 GB of
free disk space, plus space for weather inputs and forecast output.

### Linux

```bash
python3 -m venv ~/.local/share/arwen/venvs/2.8.0
~/.local/share/arwen/venvs/2.8.0/bin/python -m pip install 'gpuwm[all-cu12]==2.8.0'
~/.local/share/arwen/venvs/2.8.0/bin/python -m gpuwm.cli fetch-tables
~/.local/share/arwen/venvs/2.8.0/bin/python -m gpuwm.cli fetch-geog --datasets wrf
~/.local/share/arwen/venvs/2.8.0/bin/python -m gpuwm.cli doctor
```

`fetch-geog` sets up the global geography every forecast builds its terrain,
land use and soil from, once per computer. It downloads about 1.3 GB and
takes about 17 GB of disk once unpacked; add `--list` to see each dataset and
its size without downloading. Without it a forecast is refused at its start.

Extract the complete Linux tarball. From its application folder:

```bash
./Start\ ArWen.sh --python ~/.local/share/arwen/venvs/2.8.0/bin/python
```

If your distribution does not include Python's venv module, install its venv
package first; Ubuntu provides `python3-venv`.

The Linux launchers open the connected GUI and TUI. Use
`Start ArWen Terminal.sh` for the terminal alone. After activating a Python
environment, `gpuwm tui` also opens the terminal.

Settings and run caches live outside the application folder. Keep the application
files and notices together when moving or updating the desktop.

## GPU requirements and forecast data

The normal desktop installation above already includes the GPU dependencies.
Local forecasting requires a compatible NVIDIA GPU and driver. A card's VRAM
capacity alone does not establish driver or CUDA compatibility.

For a manual Python installation on a supported CUDA 13 GPU and driver, replace
`all-cu12` with `all-cu13` in the installation command. Install only one CuPy/CUDA major in each environment.
A driver reporting CUDA 13 support can also run CUDA 12 applications; that
display does not require switching a working CUDA 12 installation. See
[NVIDIA's compatibility guidance](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).

`gpuwm doctor` reports each remedy as a command or a `#` comment explaining
the manual step.

The GPU extra installs CuPy and the required user-space NVIDIA CUDA components
through their dependency packages. It does not install Python or the NVIDIA
device driver. Install one CuPy/CUDA major per environment. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) and
[ArWen hardware guidance](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/docs/public/HARDWARE.md).

Most scientific lookup data and the renderer's map assets (coastlines,
borders, state lines and counties) arrive with the automatically installed
`gpuwm-data` package. `gpuwm fetch-tables` obtains and hash-checks two additional
Thompson tables, about 314 MiB combined. Geography and forecast inputs are
separate: configure an existing geography installation or use `gpuwm fetch-geog`.
ERA5 through Copernicus CDS requires the user's CDS configuration and dataset
access. Deterministic ERA5 reanalysis also has a public, keyless ARCO route:
select `--era5-provider arco` when creating the domain or fetching ERA5.
Ensemble-member retrievals still require CDS. Remote operation requires an SSH
client and a configured, reachable forecast computer.

Local forecast integration runs on CUDA; it has no CPU fallback. The desktop,
terminal, native weather processing, and remote-control workflows can run on a
computer without a local CUDA installation.

Only an intentionally viewer-only or remote-control installation should use
`gpuwm[render]==2.8.0` without a GPU extra. It cannot execute local forecasts.

## Work with weather

- **Explore:** choose a supported source, field, area, and valid time; load a
  weather map and play available UTC frames.
- **Create forecast:** start from an area, an Explore map, a saved configuration,
  or a cyclone setup. Choose the forecast computer and inputs, review the
  resolved physics and memory requirements, then launch.
- **My forecasts:** return to active or saved runs, inspect domains and output
  times, view native fields and plots, and play the forecast history.
- **Terminal and CLI:** use `gpuwm tui` for an interactive workspace or the
  command interfaces for scripts and automation.

Forecast and cyclone setup use the shared preparation source catalog, with
geographic coverage, available times and required fields determined by the source. The installed source catalog is
available through `gpuwm sources --json`. Forecast-input support and Explore
preview support are separate; some registered forcing sources do not yet have
an Explore preview route.

Local data assimilation is experimental and hidden in the ordinary desktop.
Advanced users can enable its entry point with `gpuwm tui --enable-local-da`.
A local DA run scores its own forecast against the MRMS composite, and
`gpuwm local-da --continuous` cycles a regional analysis window after window,
with a durable stop and a resume. A score is a measurement of that run: it
masks a 9 km rim, scores one member, and is not a skill claim for the release.
Continuous cycling is qualified as a door and a state machine; analysis quality
is not part of that qualification.

Offline downscaling can retain the parent's physics or change the child's
microphysics scheme. A child of a different scheme is converted by the same
transition the live nest edge runs, on the parent archive before
interpolation, between any two of Kessler, WSM6, WDM6, Thompson, aerosol-aware
Thompson, Morrison, Milbrandt-Yau, NSSL and P3; WDM6 archives are read like
every other scheme's. The one edge without a contract is a microphysics-off
parent or child, which the review refuses by name.

For a configuration already prepared for your computer and inputs:

```bash
gpuwm go forecast.toml --dry-run
gpuwm go forecast.toml
```

The first command reviews the route; the second executes it. See the
[CLI manual](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/docs/public/CLI-USER-MANUAL.md)
and [TUI manual](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/docs/public/TUI-USER-MANUAL.md)
for the complete workflows, and the
[stage contract](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/docs/public/PIPELINE-STAGES.md)
for running preprocessing, the simulation and rendering on your own terms.

## Integrate ArWen into another application

The [integration guide](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/docs/integration-kit/integration-guide.md)
and [downloadable kit](https://github.com/FahrenheitResearch/arwen/releases/download/v2.8.4/ArWen-2.8.4-Integration-Kit.zip)
describe the CLI, run-plan documents, catalogs, and companion bridge boundaries.
The kit contains an example subprocess client and tests. It is a documented
integration surface, not a separate simulation engine or a replacement for
runtime validation.

Use the installed catalogs and validated plans when building another interface.
Preserve request identity, selected source/time/domain, cancellation, and the
distinction between a preview and an executing forecast.

## Science, compatibility, and current limits

ArWen independently implements a WRF-ARW-class regional atmospheric model and
WRF-derived physics on the GPU. Its numerical comparisons and qualification
apply to documented configurations and test cases. Matching WRF is code
verification; it does not validate a forecast against reality. Validation
requires observations. Existing ASOS and MRMS scores cover specific runs;
the broader observation scoreboard is in development. ArWen can inherit
WRF's published validation record only to the extent the two are statistically
indistinguishable for that configuration. No universal WRF equivalence or
general forecast-skill claim follows. Read the
[verification and validation record](docs/public/VERIFICATION.md),
[physics documentation](docs/public/PHYSICS.md),
and [2.8 changes](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/CHANGELOG.md).

2.8.0 adds ArWen in your browser, `gpuwm gui`, as a preview beside the desktop
app: a storm wiki of 17 cited events with a best simulation for each card size,
New forecast from any past date with a Physics step and a choice of vertical
levels, a map that places every picture on its own grid while the forecast
runs, a Machines page that draws a forecast's pictures on this computer or an
SSH host, and an optional assistant, off until you turn it on, that runs on a
language model on this computer. It
also adds `gpuwm warm-kernels`, ERA5 from Google's keyless store, and the
observation doors that build from this repository, and it fixes the HRRR soil
donor search over coasts, runs folders past the Windows 260-character path
limit, the source check that called a start available before the download
would accept it, and the other refusals the changelog lists. What this release
qualifies is its artifact and startup checks, a short GPU forecast, and the
scoped component evidence the verification records name. Outside that
qualification: general forecast skill, whole-model parity, the quality of a
forecast initialised from ICON global, and the assistant's answers, which come
from whichever model it is given. Three gaps are known and named rather than
qualified: the web GUI is a preview and the desktop app remains the full way to
run ArWen, what the fine nest costs at the nowcast door is not stated yet, and
the desktop weather map cannot draw ICON global fields, which does not affect
the forecast.

- **Resolved scale.** A 500 m nest resolves supercell storm
  structure, cold pools, mesocyclone-scale rotation, and the
  environmental and morphological severe-weather diagnostics drawn on
  it: 2-5 km updraft helicity, the Significant Tornado Parameter,
  CAPE/CIN/SRH/shear.
  It does not resolve tornado dynamics: the near-surface corner flow,
  suction vortices, or tornado-scale wind intensity, which the
  literature on tornado-like vortices places below roughly 25 m
  horizontal and 10 m vertical spacing.
  Read the STP/UH severe suite as a tornadic-supercell environment
  and mesocyclone-proxy diagnostic, not as a resolved-tornado claim:
  these are convection-permitting to sub-kilometer case studies,
  not tornado-resolving simulations.
  Sub-kilometer nests also sit in the PBL gray zone described in
  [FIRST-LIGHT.md](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/docs/public/FIRST-LIGHT.md).
- **Checkpoints:** 2.7 writes format 6. Format-5 checkpoints from older releases
  and previews are not compatible. Finish those runs with the version that
  created them, or start again from the original configuration and inputs.
- **Noah-MP memory basis:** `sf_surface_physics = 4` is priced from the
  per-thread frames its kernels compile to, which are readings of a compile
  platform (target architecture and NVRTC build). The NVRTC build is set by
  the cuda-toolkit release the package's `[ctk]` extra resolves to when pip
  runs, not by the machine. The desktop runtime is a CUDA-12 install. This
  release carries readings on both CUDA majors: sm_120 and sm_86 on NVRTC
  12.9.86, the build a CUDA-12 install resolves (`gpu-cu12`, `gpu`, `all`, and
  the desktop runtime), and sm_120 on the three builds the `gpu-cu13` extra has
  resolved to, 13.4.92 (what `pip install gpuwm[gpu-cu13]` installs today),
  13.4.59 and 13.3.33. On a recorded platform the estimate prices Noah-MP from
  that platform's own reading. On any other card the Noah-MP frames are priced the
  way every other kernel is priced on an unrecorded platform, from the ceiling
  over the recorded platforms, and `gpuwm check` states that basis beside the
  verdict ("Noah-MP local frames priced from the ceiling over the recorded
  platforms ...; not measured on this card"). Either way `gpuwm check` and
  `gpuwm run` admit Noah-MP with no flag; `tools/measure_noahmp_frames.py
  measure`, run on a card in its own environment, produces the row that makes
  its price a reading.
- **Playback:** the first pass may wait while remote frames enter the local
  cache. Startup compilation and input preparation also affect reported overall
  simulation speed.
- **Known interface issues:** some preview sources remain unavailable, exported
  dateline-crossing outlines can show a seam, and a background compact-store
  worker can report an error during the forecast-completion transition. The
  release qualification documents the retained limits.

ArWen is research and educational software. Use official meteorological services
for forecasts and safety warnings. The project is not affiliated with or endorsed
by NCAR or UCAR.

## Report an issue

Include the version, platform, what you attempted, and the relevant error.
`gpuwm report RUN_DIRECTORY --output diagnostics.zip` collects a diagnostic
report with recognized sensitive patterns redacted. **Review it before sharing:**
raw copied logs can contain local paths, usernames, hosts, and configuration
details, and no redaction tool guarantees anonymity. Never post credential files
or private keys. See
[reporting a problem](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/docs/public/REPORTING-A-PROBLEM.md).

## Development and licensing

Development combines human direction, AI-assisted implementation, and numerical
and runtime verification. The source, scientific provenance, tests, and release
qualification records are available for inspection.

For a developer installation from a source checkout, use `bash install.sh` on
Linux or `powershell -File install.ps1` on Windows. Source installations build
the native components and need the corresponding build tools. The explicit
`bash install.sh` invocation also works when an extracted source archive has
lost executable permission bits; use it if `./install.sh` reports
`Permission denied`.

The engine is licensed under the [Apache License 2.0](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/LICENSE).
The desktop application and its dependencies carry their own licenses and
notices in the desktop archives. Third-party code, tables, and datasets retain
their respective terms; see [NOTICE](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/NOTICE)
and [scientific provenance](https://github.com/FahrenheitResearch/arwen/blob/v2.8.0/PROVENANCE.md).
