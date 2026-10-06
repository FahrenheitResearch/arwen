# Build record: NOAA-EMC/HRRR v4.1.21 CPU analysis tools in a container

This folder builds the analysis side of operational HRRR from NOAA's own
sources with no source edits, so it can run beside the GPU forecast model.
Nothing here is a port: the programs are NOAA's, compiled as NOAA ships them.

Claims are tagged MEASURED (run here, on the date given), CODE (file and line
in the tag), PUBLIC (URL) or ESTIMATE.

## What is built

Image stages, each a `podman build --target`:

| Stage | Holds | Status |
|---|---|---|
| `toolchain` | Intel compilers, MPI, MKL, third-party libraries, NCEPLIBS | MEASURED 2026-10-03: builds |
| `gsi` | `hrrr_gsi` (cloud analysis compiled in), `hrrr_enkf`, `hrrr_process_enkf`, `hrrr_initialens`, `hrrr_process_mosaic`, `hrrr_process_mosaic_enkf`, `hrrr_process_cloud`, `hrrr_process_lightning` | MEASURED 2026-10-03: builds; GSI, mosaic and cloud run |
| `full` | adds `hrrr_wrfarw_real`, `hrrr_wrfarw_fcst`, `hrrr_update_bc`, `hrrr_ref2tten`, `hrrr_wps_ungrib`, `hrrr_wps_metgrid`, `wgrib2` | MEASURED 2026-10-03: builds; real, ungrib, metgrid and wgrib2 run |

Executables land in `/opt/hrrr/exec`, build logs in `/opt/hrrr/build-logs`,
the unedited tree in `/opt/hrrr`.

## How to build

    ./fetch_sources.sh                 # downloads into ./src, writes SOURCES.sha256
    podman build --target toolchain --build-arg JOBS=10 -t gsi-clone-toolchain:v4.1.21 .
    podman build --target gsi       --build-arg JOBS=12 -t gsi-clone-gsi:v4.1.21 .
    podman build --target full      --build-arg JOBS=12 -t gsi-clone-full:v4.1.21 .

`SOURCES.sha256` and `SOURCES-run.sha256` are committed. The image build
verifies every tarball against them before unpacking, so a re-rolled
upstream archive stops the build.

## Pins

| Input | Pin | Where |
|---|---|---|
| HRRR sources | tag `v4.1.21`, commit `a131d6c7f6ebf57ae3c43f198229ad6afdf4d886`, checked after clone | `Dockerfile`, stage `gsi` |
| Base image | `ubuntu:20.04@sha256:c664f8f86ed5a386b0a340d981b8f81714e21a8b9c73f658c4bea56aa179d54a` (glibc 2.31) | `Dockerfile` |
| Ubuntu packages | `snapshot.ubuntu.com` at `20260901T000000Z` | `Dockerfile`, `APT_SNAPSHOT` |
| Intel compilers, MPI, MKL | 47 packages by exact version; the four asked for are `intel-oneapi-compiler-fortran-2021.1.1`, `intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2021.1.1`, `intel-oneapi-mpi-devel-2021.1.1`, `intel-oneapi-mkl-devel-2021.1.1` | `intel-packages.txt` |
| CMake | 3.18.4, Kitware binary | `SOURCES.sha256` |
| zlib 1.2.11, libpng 1.6.37, libjpeg 9c, jasper 2.0.25, hdf5 1.10.6, netcdf-c 4.7.4, netcdf-fortran 4.5.3, pnetcdf 1.12.2 | source tarballs by SHA-256 | `SOURCES.sha256`, `build_thirdparty.sh` |
| NCEPLIBS bacio 2.4.1, w3nco 2.4.1, bufr 11.4.0, g2 3.4.5, g2tmpl 1.10.0, wrf_io 1.1.1 | source tarballs by SHA-256 | `SOURCES.sha256`, `build_nceplibs.sh` |
| wgrib2 2.0.7 (`versions/run.ver`) | NOAA CPC tarball by SHA-256; gfortran 9 from the apt snapshot | `SOURCES-run.sha256`, `build_wgrib2.sh` |
| GSI's own libraries (bacio, bufr, CRTM v2.3.0, ip, nemsio, sfcio, sigio, sp, w3emc, w3nco) | the copies inside the tag, `sorc/hrrr_gsi.fd/libsrc` | the tag |

The library versions are the ones the tag's `versions/build.ver` loads on
WCOSS2 (CODE `versions/build.ver`). netcdf-fortran 4.5.3 is the Fortran layer
that ships inside WCOSS2's `netcdf/4.7.4` module (ESTIMATE: hpc-stack's
pairing for that release; `build.ver` names one netcdf version only).

Compiler banners inside the image (MEASURED 2026-10-03):
`ifort (IFORT) 2021.1 Beta 20201112`, `icc (ICC) 2021.1 Beta 20201112`,
`Intel(R) MPI Library for Linux* OS, Version 2021.1 Build 20201112`. CMake
identifies the compilers as `Intel 20.2.1.20201112`.

## How this differs from the operational build

Each of these is outside the sources.

1. **Compiler.** Operations: `intel/19.1.3.304` (CODE `versions/build.ver`).
   Here: oneAPI 2021.1.1, the first oneAPI release and the oldest classic
   ifort and icc Intel's package repository still serves (PUBLIC
   `https://apt.repos.intel.com/oneapi`). Results can differ in the last bits
   from NOAA's executables. Not measured: no NOAA executable is public to
   compare against.
2. **MPI.** Operations: `cray-mpich/8.1.7`. Here: Intel MPI 2021.1.1.
   `/opt/craywrap` supplies the Cray driver names `ftn`, `cc`, `CC` that the
   tag's makefiles call, over `mpiifort`, `mpiicc`, `mpiicpc`.
3. **Host.** GSI's CMake picks its build flags from the host name. Any name
   it does not list, WCOSS2's included, takes the `GENERIC` branch, which
   builds GSI's own copies of bacio, bufr, CRTM, ip, nemsio, sfcio, sigio,
   sp, w3emc and w3nco from `libsrc` (CODE
   `sorc/hrrr_gsi.fd/cmake/Modules/setHOST.cmake`; MEASURED 2026-10-03: the
   CMake log says `Host is set to GENERIC` and "building from libsrc" for
   each). So the GSI link uses the same libraries here as there.
4. **WCOSS2 paths.** The tag's WRF configure file names netcdf, hdf5 and
   pnetcdf by absolute WCOSS2 path. The image has those paths as links to
   `/opt/libs`, and the file is used as shipped.
5. **Make order.** The tag builds GSI and WRF with a serial make. Here a
   parallel pass runs first and the tag's serial make closes; see below.

## Build problems met, and what each needed

All MEASURED 2026-10-03 on a 24-core host.

- `snapshot.ubuntu.com` answers https only; the base image has no CA
  certificates. `ca-certificates` is installed from the image's own archive
  first.
- podman's default image format ignores `SHELL`, and Intel's `setvars.sh`
  needs bash. Every compiler step runs `bash -c ". /opt/gsi-env.sh && ..."`.
- netcdf-c 4.7.4 has no release tarball left on Unidata's server. The tag
  archive from GitHub is used and built with CMake.
- NCEPLIBS-g2's `find_package(Jasper)` fails without a libjpeg beside
  jasper. libjpeg 9c (`build.ver` `libjpeg_ver`) is built; jasper itself is
  built without its libjpeg codec because the tools' makefiles link jasper,
  png and zlib only.
- **GSI, parallel make.** With 12 jobs the first pass fails with 23
  `error #7002: Error in opening the compiled module file`. The cloud
  analysis sources (`gsdcloudanalysis.F90` and siblings) sit inside
  `#ifdef RR_CLOUDANALYSIS`; the define reaches the compiler through
  `COMPILE_FLAGS`, which CMake's Fortran dependency scanner does not read,
  so their `use` lines are not dependencies. `build_gsi.sh` runs two
  parallel passes that keep going, then the tag's serial make, which must
  end clean: second pass 0 errors, serial pass 0 errors.
- **WRF, parallel compile.** With `./compile -j 12`, `main/real_em.o` was
  compiled while `dyn_em/module_initialize_real.o` was still building, and
  `real.exe` never linked. `build_wrf.sh` runs the parallel compile, then
  the tag's own `./compile -j 1`.
- **Stack.** `hrrr_process_mosaic` keeps whole grids on the stack. Under the
  default 8 MB limit every rank is killed before it reads a level. The case
  scripts set `ulimit -s unlimited`.
- `hrrr_process_mosaic` stops with fewer than 33 MPI ranks (one per MRMS
  level; CODE `sorc/hrrr_process_mosaic.fd/process_NSSL_mosaic.f90:169-173`).
  The tag's count of 36 is kept; on 24 cores the ranks share them.

## Measured results

MEASURED 2026-10-03 on a 24-core host, builds under `nice`, from this
folder's recipe with nothing cached below the Intel layer:

| Stage | Wall time | Image |
|---|---|---|
| `toolchain` (Intel packages, third-party libraries, NCEPLIBS) | 6 min (`JOBS=8`) | `30c13cda571a`, 9.61 GB |
| `gsi` (GSI, EnKF, three preprocessors) | 2.5 min | `32861db9a663`, 10.8 GB |
| `full` (WRF real and wrf, update_bc, ref2tten, WPS, wgrib2) | 21 min, almost all of it WRF | `a552684367e0`, 13 GB |

"Sources untouched" is checked in the build and after it: `git status
--porcelain --untracked-files=no` in `/opt/hrrr` is empty after the GSI and
preprocessor steps (the build fails otherwise) and still empty in the
finished `full` image.

Executables of that image (SHA-256; each embeds its build time, so a rebuild
gives new hashes with the same sizes, as two builds of this recipe did):

    060b8cc42d66c0d23ddc86556e113ab23370415c3cc55ffcb441022273aa63d3  hrrr_gsi                  48,473,952 bytes
    6a855155e778fea3c45a055d8ca8b1562b22b39086f8dbcc2bcb447781a7b26a  hrrr_enkf                 12,323,368
    98d479da6a9c0e104e90c717c8b5c231f99c3c08353ae1a2e95efa0f256b7423  hrrr_process_enkf         24,598,848
    49d86b187726611d75943f91901705990012134a20c2b492c441c304b357571c  hrrr_initialens            2,638,560
    2e5b326423e469c38c06120efd3a11ee638fd714cc5f6d7e86743f4ba452cebb  hrrr_process_mosaic        3,278,712
    6b6c69ead930f35e7ea6fe0e22a553525a84052622d2a5d17000136f29c0dc45  hrrr_process_mosaic_enkf   3,409,592
    4dbba182b3c910061f76778a1868a095d7ab420f8bfbaa6d7839a93df857e872  hrrr_process_cloud         2,598,320
    73a5e3052b75b6bb3b7f6cdaa90078741f1a8aefadb66244b801b4ee4485273f  hrrr_process_lightning     2,382,016
    f30f80e1f56b66ca1f868d9641b41393c412f01f118371f17bb24ca3411828a0  hrrr_wrfarw_real         175,661,920
    107adbd50ab09b6162965d7f349d9217da72952a7900e58c3478ac9fec8aded3  hrrr_wrfarw_fcst         198,176,864
    6de97978d3a42a0640c5e1e414ca2c738e8312ad00beb13774704b622450d4fc  hrrr_update_bc             1,388,664
    6c6c1bccab29a0a4f9df26a85b83138a0a11ddb47ff36df37ffc2a340dd9367c  hrrr_ref2tten              1,985,960
    fe9e52f9733195294694e815d6756b5fba733a48b2faab588a801ed9a2e5be52  hrrr_wps_ungrib            2,403,608
    5c0810fcfb118bafad0c6b92f40c3954a037cc5837efff6fbff7d39b7e1914d1  hrrr_wps_metgrid           3,867,648
    1d4bd6fbc02d8c3017491c6b61fed8576820c2322cad532711725e5c19da54aa  wgrib2 (/opt/wgrib2/bin)

Programs run on the 2026-10-03 12 UTC public case (MEASURED 2026-10-03;
the case scripts are in `case/`, the results in the lane report):

- `hrrr_process_mosaic` on the 33 public MRMS levels writes
  `NSSLRefInGSI.bufr` (539,078,144 bytes); peak memory 16 GB on 36 ranks.
- `hrrr_process_cloud` on the public NASA LaRC cloud dump writes
  `NASALaRCCloudInGSI.bufr` (124,079,864 bytes, 1,897,663 columns) and ends
  `=== RAPHRRR PREPROCCESS SUCCESS ===`.
- `wgrib2`, `hrrr_wps_ungrib`, `hrrr_wps_metgrid`, `hrrr_wrfarw_real` make
  the cold-start guess from the public RAP native file:
  `SUCCESS COMPLETE REAL_EM INIT`, a 15,516,199,884-byte `wrfinput_d01` on
  the full 1799 x 1059 x 50 grid, 2 min 46 s on 8 ranks, about 100 GB of
  container memory at peak (page cache included).
- `hrrr_gsi` runs both of the tag's steps (variational, then cloud analysis)
  on that guess and the public observations: `PROGRAM GSI_ANL HAS ENDED`
  twice.
- `hrrr_process_lightning`, `hrrr_enkf`, `hrrr_update_bc`, `hrrr_ref2tten`
  and `hrrr_wrfarw_fcst` are built and not run here.

## The full stage

`build_wrf.sh` compiles the tag's WRF-ARW with its own
`configure.wrf.useme` and environment (`WRF_DFI_RADAR=1`, `WRF_SMOKE=1`,
`PNETCDF_QUILT=1`, `WRFIO_NCD_LARGE_FILE_SUPPORT=1`), then `update_bc` and
`ref2tten`, which link pieces of the built WRF tree. `build_wps.sh` builds
WPS 3.9.1 the same way. Two more things were needed, both MEASURED
2026-10-03:

- The Cray `ftn`, `cc` and `CC` drivers link the OpenMP runtime; the
  Intel MPI drivers do not. WPS links WRF's I/O library, which is compiled
  with `-qopenmp`, through a link line that names no OpenMP flag, and the
  link failed on `__kmpc_*` symbols. The `/opt/craywrap` drivers add
  `-liomp5 -lpthread` to link steps only; compile steps pass through
  untouched, so nothing is compiled with OpenMP that the tag does not
  compile with it.
- WPS 3.9.1's ungrib misreads the RAP file as NCEP publishes it (complex
  packing): metgrid then wrote pressures of 1.3e11 Pa and real.exe stopped
  at `p_top_requested < grid%p_top possible from data`. The tag's makeguess
  script repacks the file with wgrib2 first (`-set_grib_type s`), so the
  image carries wgrib2 2.0.7, the tag's `versions/run.ver` version, built
  from NOAA CPC's tarball with the GNU compilers (pinned in
  `SOURCES-run.sha256`; gfortran from the same apt snapshot).

## Run-time data (not in the image)

| Data | Source | SHA-256 |
|---|---|---|
| Fix files, `fix/conus` | PUBLIC `https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/hrrr.v4.1.21/fix/conus/` (the git tag has no fix folder) | below |
| CRTM 2.3.0 coefficients, 2,591,922,649 bytes | PUBLIC `https://ftp.emc.ncep.noaa.gov/jcsda/CRTM/REL-2.3.0/crtm_v2.3.0.tar.gz` | `881743f36aec16d5e5166d4f2f22968a92bf139e6ef188d4b9ede929ab0fde82` |

    b043b4a7b63e944e8e884a66329c611bad8ff42ae5060214afe9bf24382770ab  hrrr_anavinfo_arw_netcdf
    dd6969e9f62267a2d21861f88250a7da254a802f6cbc6e8b1ad04fd59c6ce61a  hrrr_berror_stats_global
    34b3a1baa541e427731a19b8f4c208786c1038a3bda36626596f18c37c604012  hrrr_current_bad_aircraft.txt
    164127d2b92436129a1ce62063263fcd54ff05ea4ce6165194eed12052a2dd84  hrrr_current_mesonet_uselist.txt
    a785b53fdb8493e376b88ece6e166ddd8046588dd1c0062181141bf2b012f6e2  hrrr_geo_em.d01.nc
    21313235544454c0144c36ec5672caf7ca059e5515657e4c99fb6df1e7f0bff9  hrrr_global_ozinfo.txt
    53402d69282a2b7346c19525e9397dafd9ce9f3d00fa23a7c4b90e5fd437c11c  hrrr_global_pcpinfo.txt
    d850ee5754f796f0e6d0225c6818d007e700cb160afc5f84ce6e8be44be4fb42  hrrr_global_satinfo.txt
    4de39c23dea0dfcdad07f71141371ab33675af6b06c9d4e458942ab9e4dcfdf3  hrrr_gsd_sfcobs_provider.txt
    9c5ddfd32751c93fb5e8689059f0e68d74360e5060b55d33035da90756f8b4fe  hrrr_nam_errtable.r3dv
    a91c8d24fd39d782fe865420ad330c1934dd9797c8f16dec6e79481283433f26  hrrr_nam_regional_convinfo

`case/prepare_crtm_fix.sh` lays the big-endian coefficient set out flat, the
shape the analysis script's `FIXcrtm` folder has. GSI is built with
`-convert big_endian` (CODE
`sorc/hrrr_gsi.fd/cmake/Modules/platforms/Generic.cmake`).

## Not built

`hrrr_full_cycle_surface`, `hrrr_update_gvf`, `hrrr_process_sst`,
`hrrr_process_imssnow`, `hrrr_process_fvcom`, `hrrr_prep_smoke`, the HRRRDAS
tools (`hrrr_cal_bcpert`, `hrrr_cal_ensemblemean`, `hrrr_copy_hrrrdas`,
`hrrrdas_wrfarw`) and the post-processing programs. Each has its own
`sorc/build_hrrr_*.sh` in the tag and builds the same way as the
preprocessors here (ESTIMATE; none was tried).
