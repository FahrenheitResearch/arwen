# Install and verify RW-WPS

RW-WPS does not yet publish a community package. There are two useful
installation modes: a developer checkout for inspection/tests and a sealed
Linux or Windows x86-64 runtime archive for real preprocessing. Neither mode
changes the scientific support matrix.

## Developer checkout

### One command

`install.sh` (POSIX) and `install.ps1` (PowerShell) at the repository root
perform the whole developer install: create `.venv` if absent, install
`-e '.[gpu,render]'` into it, stage the externalized Thompson tables
with `gpuwm fetch-tables` (downloads only what is absent -- ~243 MiB
from a checkout -- SHA-256-verified against the packaged pins before
install; `--no-fetch-tables`/`-NoFetchTables` or
`GPUWM_INSTALL_NO_FETCH_TABLES=1` defers it, and `gpuwm fetch-tables
--from DIR` stages them offline), offer to install rustup when `cargo` is
missing (they prompt first; `--yes`/`-Yes` or `GPUWM_INSTALL_YES=1`
consents non-interactively, and a non-interactive run without consent
fails with instructions instead of hanging), build the vendored Rust
GRIB bridges (`tools/grib1_bridge`) and the production render engine
(`tools/rustwx`) with `cargo build --release --locked --offline`
(`--no-render`/`-NoRender` or `GPUWM_INSTALL_NO_RENDER=1` skips the
renderer build; `gpuwm render` falls back to matplotlib until it is
built), and finish with `gpuwm doctor`, whose exit status the script
propagates. Both scripts are idempotent: re-running reuses the existing
`.venv` and the incremental cargo build. Run them from the checkout
root, or standalone (piped from the raw URL), in which case they clone
https://github.com/FahrenheitResearch/arwen into `./gpuwm`
(`GPUWM_REPO_URL` overrides the clone source). `GPUWM_PYTHON`
overrides the interpreter used to create the venv.

### Manual steps

Use Python 3.11 or newer. An editable developer checkout still installs the
monorepo metadata and its forecast/plot dependencies. The sealed release
builder instead creates a dedicated `rw-wps` wheel containing the source
adapters, initialization/export state, required setup kernels, and HRRR
preprocessing helpers; forecast executors and forecast-only physics data are
absent.

POSIX:

```bash
git clone https://github.com/FahrenheitResearch/arwen gpuwm && cd gpuwm
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gpu,render]'
gpuwm fetch-tables
(cd tools/grib1_bridge && cargo build --release --locked --offline)
gpuwm doctor
```

Windows (PowerShell):

```powershell
git clone https://github.com/FahrenheitResearch/arwen gpuwm; cd gpuwm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[gpu,render]'
gpuwm fetch-tables
cd tools\grib1_bridge; cargo build --release --locked --offline; cd ..\..
gpuwm doctor
```

`[gpu]` = CuPy for the CUDA runtime (`gpuwm check`/`run`, wizard
sizing); `[render]` = the pinned `wrf-rust` + matplotlib for `gpuwm
render`; add `[dev]` for the test suite:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q \
  tests/test_source_adapters.py \
  tests/test_namelist_compat.py \
  tests/test_native_wrf_distribution.py
```

**Wheel installs need two more commands for the data routes.**  A pip
wheel deliberately contains no compiled Rust, so `gpuwm check`/`run`
(ERA5 route) and the `rw-wps` GFS/HRRR front doors have nothing to
decode GRIB with until the artifacts are on the machine:

```bash
pip install gpuwm
gpuwm fetch-bridges
gpuwm fetch-tables
gpuwm doctor
```

`gpuwm fetch-bridges` downloads one release bundle for this platform --
the five GRIB decoders, the CPU preprocessing library, the `rw_fetch`
backbone and the `rw_wrfbatch` renderer -- verifies every artifact
against the exact size + SHA-256 pins packaged in the wheel, and stages
them into `~/.gpuwm/bridges`, the directory the resolver already
searches.  It is re-run safe (present, pin-valid artifacts are left
alone), an interrupted download resumes or restarts, and `--from DIR`
stages the same bundle -- or the loose artifacts -- from a local
directory with identical verification, for air-gapped installs.
`--dest DIR` stages elsewhere; `GPUWM_BRIDGE_ASSET_URL_BASE` points it
at a mirror.

Bundles are published for Windows x86-64 and Linux x86-64.  On any
other platform, and on any release that published no bundle, the
command says so by name and the route is the same one-time
`cargo build --release --locked --offline` from a source clone: point
the install at the built executables with the per-artifact environment
variables (`GPUWM_GRIB1_BRIDGE`, `GPUWM_GFS_GRIB2_BRIDGE`, ...) or copy
them into `~/.gpuwm/bridges/`.  `gpuwm doctor` prints whichever of the
two remedies is true for the machine it runs on.

The wheel and sdist also exclude the two externalized Thompson tables
(freezeH2O.dat, 243 MiB, and qr_acr_qg_V4.dat, 71 MiB -- together they
would put the artifacts over PyPI's per-file limit); run
`gpuwm fetch-tables` once after `pip install gpuwm` to stage them,
SHA-256-verified, into the installed package's table root (a checkout
already carries qr_acr_qg_V4.dat, so from a clone only freezeH2O.dat
downloads).
`gpuwm doctor` reports exactly which pieces are missing and prints the
command that fixes each one.

Install `'.[dev,geog]'` instead when exercising the source-tree experimental
high-resolution static raster prototype; it adds Rasterio and pyproj. That
extra does not turn the prototype into a stock-WRF-certified static source.

Build the vendored Rust decoder and CPU preprocessing workspace without
network access:

```bash
(
  cd tools/grib1_bridge
  cargo test --locked --offline
  cargo build --release --locked --offline
)
```

The crate emits the generic GRIB1/GRIB2 tools, source-specific bridges, and
the CPU shared library. Platform filenames differ (`.so`, `.dylib`, `.dll`).
Cargo must run from `tools/grib1_bridge` so it discovers the checked-in
`.cargo/config.toml` and uses `vendor/crates-io`; a populated developer Cargo
cache is not accepted as proof that the offline build is reproducible. The
Linux archive supplies CPU and CUDA setup paths. The Windows x86-64 archive
is deliberately CPU-only and uses PowerShell launchers; the named HRRR shell
pipeline is not portable to it yet.

## Sealed Linux archive

From a clean Linux x86-64 checkout, build the complete artifact with one
command. The Python wheel build and vendored Rust build do not access the
network; the Rust workspace is locked and built from its vendored sources.

```bash
python tools/build_rw_wps_release.py \
  --output-dir /tmp/rw-wps-0.1.1-linux-x86_64 \
  --archive /tmp/rw-wps-0.1.1-linux-x86_64.tar.gz
```

The underlying release builder accepts and independently probes one
source-matched wheel, five Rust bridge executables, and the Rust CPU library.
It refuses a dirty Git tree, wheel/source drift, missing bridge ABI markers,
CRLF shell payloads, and developer-specific absolute paths in installed text
files. It emits a deterministic gzip archive, `manifest.json`, and
`SHA256SUMS`.

Exercise the resulting archive from a temporary directory and a new virtual
environment with no CuPy, CUDA, or Matplotlib installation:

```bash
python tools/smoke_rw_wps_cpu_install.py \
  --archive /tmp/rw-wps-0.1.1-linux-x86_64.tar.gz \
  --receipt /tmp/rw-wps-clean-cpu-install.json
```

The smoke command installs only NumPy and netCDF4, extracts with traversal and
non-regular-entry checks, invokes `install.sh --skip-gpu`, runs the public CLI
from outside the checkout, and retains a receipt. Supply `--wheelhouse DIR`
for an offline dependency install.

After extracting an archive into a new directory:

```bash
python -m venv /opt/rw-wps-venv
/opt/rw-wps-venv/bin/python -m pip install 'numpy>=1.26' 'netCDF4>=1.6'

GPUWM_PYTHON=/opt/rw-wps-venv/bin/python ./install.sh --skip-gpu
./bin/rw-wps --version
./bin/rw-wps --show-support-matrix
./bin/rw-wps --namelist-support-report \
  --wps-namelist /case/namelist.wps \
  --namelist-input /case/namelist.input
```

`--skip-gpu` skips only CuPy/device verification. All wheel RECORD hashes,
archive hashes, decoder identities, GRIB2 tabular ABIs, CPU-library ABI,
helper inventories, and an actual serial-versus-parallel FP32 interpolation
through the native CPU library remain checked. ERA5, GFS, and mapped runs with
explicit `--preprocess-backend cpu` can retain setup/export state in NumPy and
do not import CuPy. `--preprocess-backend auto` selects CUDA only when a device
and the certified CUDA 12.x runtime family are available; otherwise it uses
the Rust/NumPy path. HRRR's public driver does not yet expose the common
backend selector.

For the CUDA path, prepare the environment with `cupy-cuda12x>=13.0`, omit
`--skip-gpu`, and verify the runtime receipt generated at install. The
launcher repeats archive, wheel, bridge, dependency, and selected-backend
checks rather than trusting an old receipt.

## Sealed Windows x86-64 CPU archive

From a clean Windows x86-64 checkout, build the dedicated wheel, all five PE
Rust bridges, and the Rust CPU DLL without network access to Cargo:

```powershell
python tools/build_rw_wps_windows_release.py `
  --output-dir C:\release\rw-wps-0.1.1-windows-x86_64 `
  --archive C:\release\rw-wps-0.1.1-windows-x86_64.zip
```

The ZIP writer fixes timestamps, modes, ordering, compression, and its single
top-level root. Rust uses the locked vendored workspace and remaps the source
checkout path. The distribution builder executes every PE bridge identity
probe, loads the CPU DLL, checks its ABI, and runs byte-identical one- and
three-worker FP32 interpolation before publishing the archive.

Exercise the archive in a new venv outside the checkout:

```powershell
python tools/smoke_rw_wps_cpu_install.py `
  --archive C:\release\rw-wps-0.1.1-windows-x86_64.zip `
  --receipt C:\release\rw-wps-windows-clean-cpu-install.json
```

The smoke extracts only regular ZIP entries beneath one root, rejects path
traversal and case-insensitive duplicate paths, installs only NumPy and
netCDF4, and proves CuPy and Matplotlib are absent. Manual installation is:

```powershell
$env:GPUWM_PYTHON = "C:\absolute\venv\Scripts\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -SkipGpu
.\bin\rw-wps.cmd --version
.\bin\rw-wps.cmd --show-support-matrix
```

The PowerShell installer and launcher re-hash every declared payload, reject
rooted/traversing/duplicate checksum paths, reject reparse points, refuse
existing install targets, bind the exact Python interpreter, and reject the
CUDA backend. The portable Windows envelope is the public CLI, descriptor and
namelist authoring, and pure Python/Rust CPU routes selected with
`--preprocess-backend cpu` or `auto`. The Bash-based named HRRR driver, a
Windows CUDA release, and real-source/stock-WRF Windows certification remain
open gates.

## What a successful install does not prove

Installation proves artifact integrity and runtime availability. It does not
prove that a new source mapping is meteorologically complete, that an
arbitrary namelist is supported, or that emitted files advance unchanged WRF.
Preserve the run's proof JSON, input manifest, static receipts, export
manifest, output hashes, exact WRF executable hash, and WRF smoke log as one
evidence set.

Before a public release, second clean Linux and Windows machines must
reproduce the platform archives and representative CPU/CUDA cases, and the
project owner must select a repository license. The dedicated wheel retains the
existing internal `gpuwm.*` module namespace for source compatibility, but its
distribution identity is `rw-wps` and its sealed payload excludes the model
driver, supervisor, dycore/physics executors, verification suite, and
forecast-only data tables.
