# Build record: NOAA's RDASApp (JEDI for RRFS v2), MPAS only, in a container

RRFS v2's analysis is JEDI, built by NOAA-EMC/RDASApp. This folder builds
RDASApp at the commit the RRFS v2 workflow pins, for MPAS only, on a
generic Linux host, so it can run on CPU beside the GPU model. Nothing here
is a port: the programs are NOAA's and JCSDA's, built from their sources.

Claims are tagged MEASURED (run here, on the date given), CODE (file and
line in a source tree), PUBLIC (URL) or ESTIMATE.

## Files

| File | What it is |
|---|---|
| `rdasapp-1bfecf33.pin` | RDASApp repository and commit, dycore choice, test-data choice, toolchain image. |
| `Containerfile.toolchain` | JCSDA's public JEDI development image, pinned by digest, with ESMF's module files moved out of the shared include folder (the one environment change; the file says why). |
| `build_rdasapp.sh` | The steps of RDASApp's `build.sh -m MPAS -t NO -b NO` with the container's spack-stack in place of NOAA's module files. Uses workspace TMPDIR and caps jobs at the container's CPU quota. |
| `verify_sources.sh` | Checks submodule pins and compares every modified source with the exact default workaround copy. Rejects other source edits before compilation. |

Run, on a host with podman:

    podman build -f Containerfile.toolchain -t rdasapp-toolchain:jedi-1.9 .
    podman run --rm -v <work>:/work <CONTAINER from the pin> \
      bash /work/recipe/build_rdasapp.sh /work/recipe/rdasapp-1bfecf33.pin <jobs>

with this folder copied to `<work>/recipe`.

## Pins

| Input | Pin | Source of the value |
|---|---|---|
| RDASApp | `1bfecf33f005c7ca3c41414aba0d155ba1f8bc29` (2026-09-17), checked after fetch | PUBLIC rrfs-workflow branch `rrfs-mpas-jedi` at `85ea55c0`, `sorc/RDASApp` gitlink |
| Submodules | the gitlinks of that commit, recorded by the build in `logs/submodule-commits.txt`; among them oops `42d9192`, saber `715f6b4`, ioda `adb16b9`, ufo `32d8f9e`, vader `b027535`, crtm `e477c08`, mpas-jedi `20043d6`, MPAS `70bdd40` with MYNN-EDMF `c942418`, TEMPO `9adf9ef`, RUCLSM `74f7b5d`, UGWP `c1c893e` | the RDASApp commit |
| Toolchain | `docker.io/jcsda/docker-gnu-openmpi-dev` at digest `sha256:69d7552277b2...` (tag `1.9`, 2025-02-17): Ubuntu 24.04, gcc 13.3.0, Open MPI 5.0.5, spack-stack 1.9 with `jedi-mpas-env` (eckit 1.28.3, atlas 0.40.0, fckit 0.13.2, ecbuild 3.7.2, hdf5 1.14.3, parallel-netcdf 1.12.3) | PUBLIC Docker Hub; MEASURED image listing 2026-10-03 |

RDASApp's own MPAS copy is `70bdd40` (its `gsl/develop` gitlink), not the
fork commit the RRFS v2 forecast model builds (`1d6e370e`, `8.3.1-noaa`).
JEDI reads model files through its own copy's stream reader, so the two
MPAS commits in RRFS v2 differ by design. (MEASURED 2026-10-03, submodule
status.)

## How this differs from NOAA's build

1. **Host and toolchain.** RDASApp at this commit supports NOAA's machines
   only; `build.sh` lists a generic host but no module file exists for it
   (CODE `modulefiles/RDAS/` has hera, jet, orion, hercules, gaeac6, ursa,
   derecho, wcoss2 only). NOAA's module files load spack-stack 1.9.3 (for
   example `ursa.gnu.lua`: gcc 12.4.0, Open MPI 4.1.6). The container has
   spack-stack 1.9 with gcc 13.3.0 and Open MPI 5.0.5. Same stack line, not
   the same release.
2. **ESMF modules.** The container's single spack view puts ESMF 8.8.0's
   Fortran modules on every target's include path; MPAS's internal ESMF time
   manager has modules of the same names, and gfortran found the view's
   first (MEASURED 2026-10-03: `Symbol 'esmf_kind_i8' at (1) has no IMPLICIT
   type` in `MeatMod.F90`). `Containerfile.toolchain` moves the view's links
   to ESMF's 137 module files aside; on NOAA's hosts each package has its own
   prefix and the two never meet.
3. **Linker.** Ubuntu's gcc links with `--as-needed`; MPAS's `libsmiol.so`
   names no PnetCDF dependency, so `mpas_init_atmosphere` failed on every
   `ncmpi_*` symbol (MEASURED 2026-10-03). The build passes
   `-Wl,--no-as-needed`, the default on NOAA's RHEL-family hosts.
4. **Test data.** `-t NO`: no RRFS test data (git-lfs). JCB is skipped
   (`-b NO`), as it only generates RRFS test YAMLs. RDASApp's own test-data
   links point into a `fix/.agent` tree that NOAA's machines provide; here
   `fix/.agent/jcsda/mpas-jedi-data_3fff9f5_20251203` points at JCSDA's
   public `mpas-jedi_testinput_tier_1_3.1.0.jcsda.tar.gz` (SSEC, MD5
   `1c9999dc04609f0c129ed39bc644bddc`, the hash mpas-jedi's CMake pins). The
   four `480km_2stream` files compared are byte-identical to the LFS objects
   of `JCSDA-internal/mpas-jedi-data` at `3fff9f5` (MEASURED 2026-10-03,
   SHA-256 of each); the rest of the tree was not compared.
5. **Source edits:** only the workaround copies RDASApp's `build.sh` makes by
   default (`sorc/_workaround_` into fv3-jedi and ufo).

## Measured results

MEASURED 2026-10-03 on a 24-core host shared with other work, under
`nice`, 12 to 16 make jobs:

- **Build: complete.** Clone of RDASApp and its submodules, configure (which
  also downloads JCSDA's public CRTM, IODA and UFO test tarballs, about
  2.5 GB), and `make` to 100 percent: 779 targets, 248 executables in
  `build/bin`, among them `mpasjedi_variational.x`, `mpasjedi_enkf.x` (the
  GETKF and LETKF driver), `mpasjedi_hofx3d.x`, `mpasjedi_eda.x`,
  `mpasjedi_rtpp.x`. 2,338 tests registered. About 50 min wall from an empty
  folder, including the two fixes above. Build tree 14 GB, sources 6.2 GB.
  SHA-256 (each embeds its build folder):

      38e8b02ff1a42e0c947da7ea810e32d0f40595feae61fa0328c461cec110705b  mpasjedi_variational.x
      c509a85f0945920a263f002d3d77481255937558fd85691dc441c44febe8be15  mpasjedi_enkf.x
      c25b5d7378e01328120b40ee015d223e6ec099be2886f007ef678f8056eb3860  mpasjedi_hofx3d.x

- **Own tests that pass:** oops, 71 of 71 (every registered oops test but
  its coding-norms check, including its generic assimilation tests);
  `ufo_opr_reflectivity`, the radar reflectivity operator RRFS v2
  assimilates, on its inputs from `JCSDA-internal/ufo-data` at `7f295bd`
  (the snapshot RDASApp's fix links name), fetched with
  `fetch_jcsda_test_data.sh` and checked against their LFS SHA-256.
- **Own tests that do not pass, and why:**
  - mpas-jedi, 0 of 60. With no test data every test stops at MPAS's
    geometry; with mpas-jedi-data `29a655c` (fetched and checked) MPAS stops
    on `Stream 'da_state' was not defined in the Registry.xml file as an
    immutable stream`. mpas-jedi `20043d6`'s test streams declare `da_state`
    immutable; RDASApp's own MPAS (`8.3.1-noaa`, `70bdd40`) defines it as a
    mutable DA-cycling stream (`Registry.xml:1312`, `immutable="false"`).
    Declaring it the way the RRFS v2 workflow does (mutable, its 181-field
    list) moves the stop to MPAS's initial-condition read. So at these pins
    mpas-jedi's own tests do not run on the MPAS RDASApp builds; making
    them run means rewriting the test stream set for the fork, which was
    not done.
  - RDASApp's RRFS tests read `RDAS_DATA`, which exists only on NOAA's
    machines (`ush/init.sh` exits for a generic host); `-t NO` leaves them
    out.
  - `ufo_opr_radialvelocity` runs, but its result differs from the
    reference values stored in the `7f295bd` observation file (RMS
    13.7 m/s over 100 observations); `ufo_opr_aircraft` stops on the 1.10.0
    tarball's file layout. Both point to test data older than the code.
  - CRTM, 0 of 96: the coefficient files the configure step downloaded were
    never linked into the tests' `testinput` folder (`atms_npp.SpcCoeff.bin
    not found`). A data-placement gap, not tried further.

## Independent review

MEASURED 2026-10-04: the existing built tree rebuilt successfully, the
three executable hashes above stayed unchanged, all 71 registered OOPS
tests passed, and `ufo_opr_reflectivity` passed again. The test container
allows its root MPI launcher with `OMPI_ALLOW_RUN_AS_ROOT=1` and
`OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`; an initial run without those settings
failed only its MPI tests before launching them. No source was changed
for that setting. The MPAS geometry check still fails, so these results
meet the CPU build and own-test bar, not a working MPAS analysis or cycle.

PUBLIC source links: [workflow RDASApp pin](https://github.com/NOAA-EMC/rrfs-workflow/tree/85ea55c0c64a07e1891bcd8e050441e09d30b5ec/sorc),
[build and default workaround copies](https://github.com/NOAA-EMC/RDASApp/blob/1bfecf33f005c7ca3c41414aba0d155ba1f8bc29/build.sh#L264),
[submodule pins](https://github.com/NOAA-EMC/RDASApp/blob/1bfecf33f005c7ca3c41414aba0d155ba1f8bc29/.gitmodules),
[workflow test-data instructions](https://github.com/NOAA-EMC/rrfs-workflow/blob/85ea55c0c64a07e1891bcd8e050441e09d30b5ec/doc/build_and_run.md#L9).
