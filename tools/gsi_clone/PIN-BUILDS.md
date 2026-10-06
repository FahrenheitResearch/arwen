# Build record: further GSI pins beside the HRRR tag's build

`BUILD.md` records the container for operational HRRR v4.1.21's CPU tools.
This file records the second recipe in this folder: GSI and EnKF at any
other NOAA-EMC/GSI commit, on the same Intel toolchain stage, driven by one
pin file. Nothing here is a port; the programs are NOAA's, compiled from
their own sources with no source edits.

Claims are tagged MEASURED (run here, on the date given), CODE (file and line
in a NOAA source tree), PUBLIC (URL) or ESTIMATE.

## How it is laid out

| File | What it is |
|---|---|
| `pins/<pin>.pin` | The pin: GSI repository and commit, build modes, and every library as `name|version|source`. The only file that changes per pin. |
| `fetch_pin_sources.sh` | Downloads the pin's sources into `src-pins/<pin>/` and writes `SOURCES-<pin>.sha256` (committed). A `git+<repo>@<commit>#<paths>` source is taken from that commit, checked, with only the named paths. |
| `Dockerfile.pin` | `FROM` the toolchain stage of `Dockerfile`; checks the hashes, builds the libraries, fetches GSI at the commit and checks it, builds GSI. |
| `env-pin.sh` | Compilers and the pin's library prefix `/opt/gsi-pin/libs`. Nothing from the HRRR tag's `/opt/libs` or `/opt/nceplibs` is on a search path. |
| `build_pin_libs.sh` | One recipe per library name; versions come only from the pin. |
| `build_pin_gsi.sh` | GSI's own cmake line with the pin's `GSI_MODE` and each of its `ENKF_MODES`; checks the GSI tree is unedited afterwards. |
| `profiles/*.profile` | GSI namelist profiles (see below). |
| `case/run_singleob.sh`, `case/singleob_increments.sh` | GSI's single-observation mode under a profile, and the increment it made. |

Adding another GSI commit is a new `pins/*.pin` file and a fetch; adding
another system's analysis settings is a new `profiles/*.profile`.

Build, from this folder, after the toolchain stage of `Dockerfile` exists:

    ./fetch_pin_sources.sh pins/gsi-db90edf.pin --check
    podman build -f Dockerfile.pin --build-arg PIN=gsi-db90edf --build-arg JOBS=12 \
      -t gsi-clone-pin:gsi-db90edf .

`JOBS` is seen by the library step, so changing it rebuilds the libraries.

## Pin gsi-db90edf: the GSI RRFS v1 builds

| Input | Pin | Source of the value |
|---|---|---|
| GSI | NOAA-EMC/GSI `db90edfb9eedbb038a9093bf1ba896425a1ff383` (2026-04-16), checked after fetch | PUBLIC rrfs-workflow v1.0.25b (`e4a100b`) `sorc/Externals.cfg` [GSI] `hash = db90edf` |
| Build modes | `GSI_MODE=Regional`; EnKF built twice, `ENKF_MODE=WRF` and `ENKF_MODE=FV3REG`; Release; every other GSI option at its default (OpenMP off, cloud analysis off, MGBF on) | PUBLIC rrfs-workflow v1.0.25b `sorc/CMakeLists.txt` `GSI_ARGS` (Regional, FV3REG, nothing else) |
| CMake | 3.20.2, Kitware binary | PUBLIC rrfs-workflow v1.0.25b `versions/build.ver` `cmake_ver` |
| zlib 1.2.11, hdf5 1.14.0, netcdf-c 4.9.2 | source tarballs by SHA-256 | `build.ver` |
| netcdf-fortran 4.6.1 | source tarball by SHA-256 | GSI db90edf `modulefiles/gsi_common.lua` (build.ver names one netcdf version) |
| bacio 2.4.1, bufr 12.2.0, w3emc 2.12.0, ip 5.2.0, sigio 2.3.2, sfcio 1.4.1, nemsio 2.5.4, wrf_io 1.2.0, ncio 1.1.2, ncdiag 1.1.2 | GitHub tag archives by SHA-256 | `build.ver` |
| CRTM 2.4.0.1 | JCSDA/crtm commit `7ecad4866c400d7d0db1413348ee225cfa99ff36` (tag `v2.4.0_emc.3`), source paths only | see differences, item 3 |
| Compilers, MPI, MKL | the toolchain stage of `Dockerfile` (`intel-packages.txt`) | `BUILD.md` |

bacio 2.4.1 and zlib 1.2.11 hash the same as the HRRR recipe's copies
(MEASURED 2026-10-03, `SOURCES-gsi-db90edf.sha256` against `SOURCES.sha256`).

## How this differs from RRFS's operational build

Each item is outside GSI's sources.

1. **Compiler.** RRFS: `intel/19.1.3.304` (`build.ver`). Here: oneAPI
   2021.1.1 classic, as in `BUILD.md` item 1.
2. **MPI.** RRFS: `cray-mpich/8.1.19`. Here: Intel MPI 2021.1.1. The Intel
   MPI drivers stand in for the Cray `cc` and `ftn` wrappers for every
   library and for GSI.
3. **CRTM.** `build.ver` says `crtm_ver=2.4.0.2`. No public CRTM source tag
   has that name (PUBLIC JCSDA/crtm tag list, 2026-10-03). GSI db90edf's own
   WCOSS2 module file loads `crtm 2.4.0.1` and names `crtm_fix 2.4.0.2` for
   the coefficients (CODE `modulefiles/gsi_wcoss2.intel.lua`); spack maps
   2.4.0.1 to tag `v2.4.0_emc.3`. That is what is built. The tag's archive
   carries several GB of coefficient files that the library build does not
   read, so the source is taken from the tag's commit with `CMakeLists.txt`,
   `VERSION`, `LICENSE.md`, `COPYING`, `cmake`, `libsrc` and `test` only.
   CRTM's top-level `CMakeLists.txt` adds `test/` unconditionally and the
   tests need that data, so that one line is commented out, as spack's
   recipe does; nothing under `libsrc/` is touched.
4. **CRTM and OpenMP.** CRTM's CMake finds OpenMP when it is present and
   exports a link to `OpenMP::OpenMP_Fortran`. GSI built with OpenMP off (as
   RRFS builds it) never defines that target, and CMake's generate step
   stopped on every GSI and EnKF target (MEASURED 2026-10-03). The GSI
   configure gets a one-line project include,
   `find_package(OpenMP COMPONENTS Fortran)`, which defines the target. GSI's
   own sources get no OpenMP flag; the executables link the OpenMP runtime
   for CRTM's objects. Whether WCOSS2's CRTM module is built with OpenMP is
   not public.
5. **zlib include path.** GSI's C object library includes `zlib.h` and its
   CMake gives that library no zlib include folder (MEASURED 2026-10-03:
   `catastrophic error: cannot open source file "zlib.h"`). On WCOSS2 the
   Cray wrappers add each loaded module's include folder; `CPATH` adds the
   pin's library prefix here, for the GSI step only.
6. **hdf5 and netCDF.** Built with MPI (`--enable-parallel`), as WCOSS2's
   `hdf5-D` and `netcdf-D` modules are. netcdf-c is built without its
   remote-data, byte-range and Zarr layers (nothing in GSI or EnKF opens a
   URL) and without pnetcdf inside it. hdf5's command-line tools are not
   built.
7. **Not built:** `sp 2.4.0` (in `build.ver`; GSI db90edf does not ask for
   it, `ip` 5 carries what it used), bufr's utilities, every library's tests.

## Measured results

MEASURED 2026-10-03 on a 24-core host shared with other work, under
`nice`, `JOBS=12`, on the toolchain stage `30c13cda571a`:

| Step | Wall time |
|---|---|
| Libraries (`build_pin_libs.sh`) | about 3 min |
| GSI and EnKF, both modes (`build_pin_gsi.sh`) | about 7 min |

Image `gsi-clone-pin:gsi-db90edf`, `d73a5407a7fa`, 10.7 GB. CMake's summary
for both builds: `GSI_MODE Regional`, `OPENMP OFF`, `USE_GSDCLOUD OFF`,
`ENABLE_MKL ON`, compilers `Intel 2021.1.0.20201112`; `ENKF_MODE WRF`, then
`ENKF_MODE FV3REG`. The GSI tree is checked unedited after both builds
(`git status --porcelain --untracked-files=no` empty, HEAD
`db90edfb9eedbb038a9093bf1ba896425a1ff383`). Every executable resolves all
its shared libraries (`ldd`: 0 not found).

Executables in `/opt/gsi-pin/exec` (SHA-256; each embeds its build folder
and time, so the two `gsi.x` differ by 1,896 bytes and a rebuild gives new
hashes):

    f148fa12428ea619d7b9db6e3b002d697ca595dcd4e7fc097d1abbc37c2024c4  gsi.x.enkf-wrf-build      67,718,192 bytes
    48b2ded1c98a53e9dbecce4ad95265f113689abeb48b89fac30e6193c1640bca  enkf_wrf.x                48,633,224
    9079827f34a88928c2a9edb3700ed70a0b75c6cab8c4a6edb54d3e782d8fad6b  gsi.x.enkf-fv3reg-build   67,720,088
    f77fa21c50df22c7739514b37e4020e3a3c00a45280c5320099975316364b913  enkf_fv3reg.x             48,883,328

`enkf_fv3reg.x` is the EnKF RRFS itself builds; `enkf_wrf.x` is the same
source in WRF mode, the one that reads WOOF's WRF-format members. Neither
EnKF was run here.

Build problems met, each fixed outside GSI's sources (MEASURED 2026-10-03):
the hdf5 tarball lists its entries as `./hdf5-1.14.0/...`; CRTM's exported
OpenMP link (difference 4); GSI's missing zlib include folder (difference 5);
a shared library prefix that put ip's 4-byte modules in front of GSI
(fixed by one prefix per library, `build_pin_libs.sh`); and CMake reading
the `CMAKE_PREFIX_PATH` environment variable with `:` separators.

## Namelist profiles

A profile (`profiles/*.profile`) is a table of the namelist values,
variable table and fix-file mapping of one system's analysis, each value
with the file and line it comes from. `case/run_singleob.sh` reads only
profile keys and the background file itself.

| Profile | What it carries |
|---|---|
| `rrfs-v1.0.25b` | RRFS v1's conv_dbz deterministic analysis (rrfs-workflow v1.0.25b `scripts/exrrfs_analysis_gsi.sh`, `ush/gsiparm.anl.sh`, `ush/set_rrfs_config_SDL_VDL_MixEn.sh`): 15 percent static, four localization groups plus the scale-separation length (`s_ens_h=328.632,82.158,4.1079,4.1079,82.158`, `s_ens_v=3,3,-0.30125,-0.30125,0.0`, `nsclgrp=2`, `ngvarloc=2`, `naensloc=5`, `r_ensloccov4var=0.05`), 2 outer loops of 50, `grid_ratio=1` on WRF files (RRFS's FV3 ratio is 2), RRFS's QC and surface options, RRFS's control vector on WRF names (`profiles/anavinfo/arw_rrfs_conv.txt`), `convinfo.rrfs` and `errtable.rrfs` from GSI-fix `3ccc7626b6` |
| `hrrr-v4.1.21` | HRRR v4.1.21's hybrid analysis: 15 percent static, one localization length (`s_ens_h=110`, `s_ens_v=3`), HRRR's static B scales, HRRR's variable table and fix files |

### What GSI's WRF interface needs for RRFS's table (MEASURED 2026-10-03)

Running RRFS's analysis settings through GSI db90edf's WRF interface found
five things the FV3 path never meets. Each is handled in the profile or the
table, with the reason written beside it:

1. `grid_ratio`: RRFS analyses on a 6 km grid (`grid_ratio_fv3=2`) and its
   FV3 member reader interpolates; the WRF member reader requires members on
   the analysis-ensemble grid ("incorrect grid size in netcdf file"). On WRF
   files the analysis grid is the model grid.
2. `i_gsdcldanal_type`: RRFS's 0 makes GSI switch hydrometeor background I/O
   off on the WRF interface (`gsimod.F90:2141-2143`); 99 ("only read
   hydrometeor fields but no cloud analysis") is the same analysis there.
3. `dbz` must stay in the table: the WRF hydrometeor-ensemble reader
   deallocates its reflectivity array unconditionally
   (`cplr_get_wrf_mass_ensperts.f90:1738`, forrtl severe 153 without it).
4. `qnr`, `qni`, `qnc` must be in the table's met_guess (guess-only, as in
   HRRR's WRF table): the analysis writer writes them only when listed and
   the netCDF updater always reads them (end-of-file on `siganl` otherwise).
5. `t2m`: GSI renamed `th2m` to `t2m` in 2022, so HRRR v4.1.21's table takes
   the new name on db90edf.

The files must also carry `QNCLOUD` (members and background) and
`REFL_10CM` (background); WOOF's files do not (history with mp_physics 8 or
28, and the WRF input export). `case/add_zero_fields.sh` adds zero copies
for the test and says why they are inert for it.

### Single-observation test on a WOOF file (MEASURED 2026-10-03)

Background: WOOF's WRF v4.6.1 input export of a 302 x 286 x 49, 3 km
Mercator grid at 2024-01-25 00 UTC (SHA-256 `a1ea2003...`; the RRFS run uses
a copy with zero `QNCLOUD` and `REFL_10CM`). Ensemble: the same WOOF run's
12 hourly history files, 01 to 12 UTC (a time-lagged set, with
`l_ens_in_diff_time`, not a real ensemble). One temperature observation at
the domain centre, 500 hPa, innovation 1.0 K, error 0.8 K. 12 ranks.

| | HRRR profile | RRFS profile |
|---|---|---|
| Localization groups (`naensgrp`, `nsclgrp`, `ngvarloc`) | 1, 1, 1 | 4, 2, 2 (plus the scale-separation length: 5) |
| Observation used, initial J | 1, 1.5625 | 1, 1.5625 |
| Wall time | 16 s | 34 s |
| Peak T increment | 0.907 K | 0.825 K |
| T reach, 1/e (west-east, south-north) | 75 km, 102 km | 87 km, 150 km |
| T reach, 1/10, longest measured direction | 153 nominal km, 177 nominal km | 222 nominal km west-east; southward 423 nominal km, northward still above threshold at the 432 nominal km edge |
| W, cloud, rain, snow, graupel increments | none (not analysis variables) | nonzero: W 1.1e-3 m/s, rain 2.5e-7 kg/kg, cloud 2.1e-7, snow 2.1e-7, graupel 2.2e-7 |

Receipts (namelist, table, fit file, increments, standard output) are in
the lane report folder.

Named differences inside the profiles: RRFS's static B
(`rrfs_glb_berror.l127y770.f77`) is not public, so the public HRRR static B
stands in, with the format flag that matches it; `fed` is left out of the
RRFS table because GSI's WRF interface has no flash-extent density; HRRR's cloud
analysis type is 0 because the db90edf build has no cloud analysis; soil
nudging is off in both because it reads `SOILT1`, which the export does not
write. The profiles differ in the variable table, static length scales and
surface options as well as localization. The difference between their
increments is not an isolated localization-effect measurement.
`errtable.rrfs` is byte-identical to HRRR's `hrrr_nam_errtable.r3dv`
(MEASURED 2026-10-03, SHA-256 `9c5ddfd3...`).

## History background preparation and independent review

CODE: GSI db90edf's WRF converter reads `C3H`, `C4H`, `C3F`, `C4F`
at `src/gsi/cplr_wrf_netcdf_interface.f90:263-347`, `RDX` and `RDY`
at :400-433, and `XLAND` at :703. WOOF's saved history lacks those
records. `case/prepare_history.sh` uses the generic Rust `nc_rewrite`
`--fields-from` option to add exactly the profile's missing static fields
from the matching WRF input export. Existing variables, attributes and
valid time are kept. A different grid or replacing a history field is
refused. Build the CPU helper with:

    cargo build -p netcdf-writer --features rewrite --bin nc_rewrite

Then, on a CPU host with ncdump, using the history and its own input export:

    NC_REWRITE=<binary> bash case/prepare_history.sh profiles/<profile>.profile \
      <history> <input-template> <prepared-history>

The run script rejects missing static records before GSI and rejects a
normal exit with a NaN cost or analysis. GSI's end banner alone is not a
successful analysis. Temporary build files use TMPDIR and make jobs are
capped at the container's CPU quota.

MEASURED 2026-10-04: the two original input-export analyses reproduced
byte-for-byte. The rebuilt recipe's GSI also reproduced the original
analysis byte-for-byte. Changing only HRRR's horizontal localization from
110 to 20 km changed peak T from 0.906849 to 0.845636 K and failed the
full-file identity check. The Rust field-copy path preserved all 115
original history variables (116 after the QNCLOUD test shim) and the seven
added template records bit-for-bit, checked with the independent NetCDF-C
reader. A real history-background run is distinct from the input-export
test above; the report carries its final numbers and receipts.

MEASURED 2026-10-04 on the 01 UTC history background, 12 CPU ranks, with
the same temperature observation and time-lagged members:

| | HRRR profile | RRFS profile |
|---|---|---|
| Completed, finite analysis | yes | yes |
| Wall time under shared host load | 29 s | 58 s |
| Peak T increment | 0.906433 K | 0.824066 K |
| T 1/10 crossing west-east, negative / positive direction | 144 / 96 nominal km | 195 / 222 nominal km |
| T 1/10 crossing south-north, negative / positive direction | 177 / 138 nominal km | 423 nominal km / beyond the 432 nominal km edge |

An independent NetCDF-C scan found every value finite in all twelve
checked analysis fields for both outputs. This is a temperature-only
reference test on a 12-file time-lagged set, not an RRFS ensemble or a
cycling analysis.

PUBLIC source links: [RRFS GSI pin](https://github.com/NOAA-EMC/rrfs-workflow/blob/v1.0.25b/sorc/Externals.cfg#L30),
[build modes](https://github.com/NOAA-EMC/rrfs-workflow/blob/v1.0.25b/sorc/CMakeLists.txt#L286),
[localization values](https://github.com/NOAA-EMC/rrfs-workflow/blob/v1.0.25b/ush/set_rrfs_config_SDL_VDL_MixEn.sh#L38),
[namelist template](https://github.com/NOAA-EMC/rrfs-workflow/blob/v1.0.25b/ush/gsiparm.anl.sh),
[variable table](https://github.com/NOAA-EMC/GSI-fix/blob/3ccc7626b6/anavinfo.rrfs_conv_dbz),
[WRF converter](https://github.com/NOAA-EMC/GSI/blob/db90edf/src/gsi/cplr_wrf_netcdf_interface.f90).
