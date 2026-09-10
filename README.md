# ArWen

**Explore weather, create regional forecasts, and run them on your own hardware.**

ArWen combines a native desktop application, a terminal workspace, and a
CUDA-based atmospheric modeling engine. Use the desktop to inspect weather maps,
choose a forecast area or cyclone, review the setup, and follow your results.
Run the model on a local NVIDIA GPU or a Linux computer connected over SSH.

![ArWen Desktop showing an ERA5 temperature field](https://raw.githubusercontent.com/FahrenheitResearch/arwen/v2.7.0/docs/public/img/arwen-desktop-2.7.png)

*The Linux desktop displaying a real ERA5 field. Windows uses the same interface.*

## Get ArWen

| Download | Requirements |
| --- | --- |
| [Windows desktop — GUI and TUI](https://github.com/FahrenheitResearch/arwen/releases/download/v2.7.0/ArWen-Desktop-1.0.0-rc3-Windows-x64-pip.zip) | x86-64 Windows; Python 3.11 or newer; working OpenGL graphics |
| [Linux desktop — GUI and TUI](https://github.com/FahrenheitResearch/arwen/releases/download/v2.7.0/ArWen-Desktop-1.0.0-rc3-Linux-x64-pip.tar.gz) | x86-64 Linux, glibc 2.39 or newer, such as Ubuntu 24.04; Python 3.11 or newer; X11 or Wayland with working OpenGL |
| [Integration kit](https://github.com/FahrenheitResearch/arwen/releases/download/v2.7.0/ArWen-2.7-Integration-Kit.zip) | Developer guide, CLI/plan examples, catalogs, and client design notes |

The desktop downloads are about 45 MB for Windows and 42 MB for Linux. They
contain the applications, native map library, map assets, and launchers. Pip
installs the engine and its dependencies separately. No Rust or C++ compiler is
needed for the supported binary packages.

The Python package is named **`gpuwm`**. Its Windows and Linux platform wheels
include the engine, native processing tools, and TUI. The desktop GUI is the
separate download above. Check [release notes and checksums](https://github.com/FahrenheitResearch/arwen/releases/tag/v2.7.0)
for the exact artifacts and qualification records.

## Install the desktop

A computer used to view weather and control a remote forecast node does not
need local CUDA. Create a dedicated Python environment and install the client
dependencies. These commands apply to the published 2.7.0 packages; each desktop
archive also includes instructions for installing from downloaded wheels.

### Windows

In PowerShell:

```powershell
py -3 -m venv "$env:LOCALAPPDATA\ArWen\venvs\2.7.0"
& "$env:LOCALAPPDATA\ArWen\venvs\2.7.0\Scripts\python.exe" -m pip install "gpuwm[render]==2.7.0"
```

Extract the complete Windows ZIP into its own folder. From that folder:

```powershell
& '.\Start ArWen.cmd' --python "$env:LOCALAPPDATA\ArWen\venvs\2.7.0\Scripts\python.exe"
```

If the Windows Python launcher `py` is unavailable, use the full path to your
installed Python executable when creating the environment.

### Linux

```bash
python3 -m venv ~/.local/share/arwen/venvs/2.7.0
~/.local/share/arwen/venvs/2.7.0/bin/python -m pip install 'gpuwm[render]==2.7.0'
```

Extract the complete Linux tarball. From its application folder:

```bash
./Start\ ArWen.sh --python ~/.local/share/arwen/venvs/2.7.0/bin/python
```

If your distribution does not include Python's venv module, install its venv
package first; Ubuntu provides `python3-venv`.

Both launchers open the connected GUI and TUI. Use `Start ArWen Terminal.cmd`
on Windows or `Start ArWen Terminal.sh` on Linux for the terminal alone.
After activating the Python environment, `gpuwm tui` also opens the terminal.

The archives use a documented manual setup. They do not include an installation
wizard or automatic updater. Settings and run caches live outside the application
folder; keep the application files and notices together when moving or updating it.

## Run forecasts on a GPU

Install the GPU dependencies **on the computer that will execute the model**.
That computer needs a compatible NVIDIA GPU and device driver. In its activated
Python environment, select the extra that matches the supported CUDA setup:

```bash
python -m pip install 'gpuwm[all-cu13]==2.7.0'  # CUDA 13
# Use 'gpuwm[all-cu12]==2.7.0' for a compatible CUDA 12 setup instead.
gpuwm fetch-tables
gpuwm doctor
```

The GPU extra installs CuPy and the required user-space NVIDIA CUDA components
through their dependency packages. It does not install Python or the NVIDIA
device driver. Install one CuPy/CUDA major per environment. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) and
[ArWen hardware guidance](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/docs/public/HARDWARE.md).

Most scientific lookup data arrives with the automatically installed
`gpuwm-data` package. `gpuwm fetch-tables` obtains and hash-checks two additional
Thompson tables, about 314 MiB combined. Geography and forecast inputs are
separate: configure an existing geography installation or use `gpuwm fetch-geog`.
ERA5 requires the user's CDS configuration and dataset access. Remote operation
requires an SSH client and a configured, reachable forecast computer.

Local forecast integration runs on CUDA; it has no CPU fallback. The desktop,
terminal, native weather processing, and remote-control workflows can run on a
computer without a local CUDA installation.

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

GFS, HRRR, and ERA5 workflows are included, with ERA5 member selection and
cyclone-follow setup among the 2.7 features. The installed source catalog is
available through `gpuwm sources --json`. Forecast-input support and Explore
preview support are separate; some registered forcing sources do not yet have
an Explore preview route.

For a configuration already prepared for your computer and inputs:

```bash
gpuwm go forecast.toml --dry-run
gpuwm go forecast.toml
```

The first command reviews the route; the second executes it. See the
[CLI manual](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/docs/public/CLI-USER-MANUAL.md)
and [TUI manual](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/docs/public/TUI-USER-MANUAL.md)
for the complete workflows.

## Integrate ArWen into another application

The [integration guide](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/docs/integration-kit/integration-guide.md)
and [downloadable kit](https://github.com/FahrenheitResearch/arwen/releases/download/v2.7.0/ArWen-2.7-Integration-Kit.zip)
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
apply to documented configurations and test cases. They do not establish
universal equivalence with WRF or validate every combination of physics,
input dataset, and hardware. Read the
[verification record](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/docs/public/VERIFICATION.md),
[physics documentation](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/docs/public/PHYSICS.md),
and [2.7 changes](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/CHANGELOG.md).

- **Checkpoints:** 2.7 writes format 6. Format-5 checkpoints from older releases
  and previews are not compatible. Finish those runs with the version that
  created them, or start again from the original configuration and inputs.
- **Physics admission:** some Noah-MP configurations currently refuse during
  memory estimation. The refusal does not silently replace the selected physics.
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
[reporting a problem](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/docs/public/REPORTING-A-PROBLEM.md).

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

The engine is licensed under the [Apache License 2.0](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/LICENSE).
The desktop application and its dependencies carry their own licenses and
notices in the desktop archives. Third-party code, tables, and datasets retain
their respective terms; see [NOTICE](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/NOTICE)
and [scientific provenance](https://github.com/FahrenheitResearch/arwen/blob/v2.7.0/PROVENANCE.md).
