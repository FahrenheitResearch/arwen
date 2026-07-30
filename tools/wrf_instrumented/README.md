# Instrumented WRF v4.6.1 nest-force oracle (N1.5)

This directory builds the read-only bundle's WRF v4.6.1 source as a serial
GNU executable, instruments the first `med_force_domain` call, and compares
its child boundary tables with gpuwm arrays.  The dump includes value **and**
tendency tables on all four sides for `u/v/w/t/ph/mu`, every runtime-active
moist slot, and every runtime-active scalar slot.  It also carries parent and
child prognostic subsamples immediately before and after the force transaction,
the complete uncoupled parent/child input arrays immediately before coupling,
plus the child solve-side FP32 `dtbc` and six sampled
`bdy + dtbc*bdy_t` values before boundary application on every child solve in
one complete parent interval.  With `mp_physics=10`, the reader makes Morrison
`qni/qns/qnr/qng` mandatory.

The committed files are:

- `instrument-med-force.patch`: diff against the pristine, LF-normalized
  v4.6.1 `share/mediation_force_domain.F` and `dyn_em/solve_em.F`;
- `namelist.input.n1p5`: the bundle's d01+d02 geometry and vertical grid,
  shortened to one outer step and changed to Morrison (`mp=10`) plus YSU
  (`bl_pbl=1`), while retaining the bundle's revised-MM5 surface layer
  (`sf_sfclay_physics=91`) and explicitly selecting gpuwm's supported
  dry-theta/diagnostic scope (`use_theta_m=0`, `nwp_diagnostics=0`);
- `dump_reader.py`: strict binary reader, structural validator, and canonical
  NPZ exporter;
- `compare_nest_force.py`: exact N1.5/F10 comparator using the band imported
  from `gpuwm/verify/nest_gates.py`;
- `produce_nest_force_candidate.py`: CPU-only REAL-emulation mirror producer
  from the complete pre-coupling inputs to canonical comparator NPZ; and
- `fixtures/single-table-dump.bin.b64`: a compact synthetic binary-record pin
  used by the CPU-only harness tests.

No dump is written unless `GPUWM_WRF_ORACLE_DUMP` is nonempty.  The patch is
for the serial build only: a multi-process build would have multiple active
tasks racing for the same path.

Relative to the bundle's `namelists/namelist.input`, the complete set of
changes in `namelist.input.n1p5` is:

- restrict `max_dom` and every domain-valued list to d01+d02;
- shorten the 12-hour run to 1 minute (explicit zero day/hour/second fields,
  end time 12:01), change `interval_seconds` from 21600 to 60 for the
  persistence frame, and change d02 `history_interval` from 15 to 60;
- omit the unused zero `history_interval_s`/`history_begin` lists and the
  inactive `restart_interval` setting;
- change `mp_physics` from 55 to 10 and `bl_pbl_physics` from 11 to 1;
- change `nwp_diagnostics` from 1 to 0 in `&time_control`; and
- add `use_theta_m = 0` in `&dynamics` (the bundle omits it and therefore
  takes the Registry default 1).

All other d01+d02 geometry, dynamics, boundary, and physics settings are
unchanged; in particular, `sf_sfclay_physics` remains 91 on both domains.

The explicit thermodynamic setting is required for a valid gpuwm oracle.
Registry default `use_theta_m=1` (`Registry.EM_COMMON:2860`) makes the
boundary `t` table family the moist-theta `thm` state
(`Registry.EM_COMMON:211`), approximately a `(1 + 1.61*qv)` field difference
from gpuwm's dry-theta consumers.  Setting
`nwp_diagnostics=0` also records the case's own D2 cadence scope instead of
inheriting the bundle's unrelated diagnostic-product choice.

## Source rationale

The table hook is immediately after `force_domain_em_part2` returns in
`share/mediation_force_domain.F:167-178`, before either grid is uncoupled.
The new `PROG` records bracket the complete native transaction: the first pair
is written before parent coupling and the second after child then parent
uncoupling, `first_force` clearing, and the child dtbc reset
(`mediation_force_domain.F:111-206`).  That transaction order matters because
`couple_or_uncouple_em.F:117-348` forms and applies forward and reciprocal REAL
factors in separate passes, so the round trip is not generally bit-inert.
The full `INPT` records are written beside the first `PROG` pair, before the
parent or child coupling call.  They contain the six dry prognostics, every
active moist/scalar field, and the base/hybrid/map-factor arrays required by
gpuwm's non-mutating coupling mirror.  The producer deliberately ignores the
post-uncoupling samples as arithmetic operands.
The force callee reaches the Registry-generated `nest_forcedown_interp.inc` from
`external/RSL_LITE/force_domain_em_part2.F:306-309`.  The generated include
fills the six state-field table families at `:7-156`, then loops over the
complete runtime moist and scalar ranges at `:157-183` and `:211-237`.

The field selection comes from the Registry `bdy_interp:dt` flags:
`Registry/Registry.EM_COMMON:158-211` (`u/v/w/ph/t`), `:287-288` (`mu`),
`:451-470` (moist), and `:519-576` (scalars).  In particular, the active
Morrison number arrays are the Registry states `qni/qns/qnr/qng` at
`:523-536`.  The logical table crops and stagger lengths mirror the write
extents generated in `inc/wrf_bdyout.inc` (for example `u` at `:7-116` and
`mu` at `:887-951`), so halo/uninitialized storage is never serialized.
The subsequent uncouple call changes the prognostic grid state
(`dyn_em/couple_or_uncouple_em.F:270-348`) but never the boundary tables, so
the values captured at the hook are the final tables consumed by the nest.

The `DBDY` hook follows WRF's native recurrence at
`dyn_em/solve_em.F:371-372`, where `grid%dtbc` has just been incremented, and
precedes the first child relax/spec application at `:932-952`.  On the pinned
d01+d02 ratio-4 case it appends four records, one for each child solve in the
parent interval.  Each record contains the post-increment REAL dtbc and six
REAL expressions sampled across `u/v/t/w/ph/mu` boundary families.  These
extension records are deliberately outside the table comparator: the reader
validates and reports them, while `compare_nest_force.py` continues to compare
only the canonical table mapping.

## 1. Build in WSL

These are exact Bash steps for Ubuntu 24.04 WSL.  They modify only the
disposable WSL copy; the bundle stays read-only.

```bash
set -euo pipefail

SRC=/mnt/c/Users/drew/Downloads/WRF_1974_MP55_reference_bundle/WRF_source_v4.6.1_group
BUILD="$HOME/wrf-oracle-build"
REPO=/mnt/c/Users/drew/gpuwm

test "$(readlink -f "$SRC")" = "/mnt/c/Users/drew/Downloads/WRF_1974_MP55_reference_bundle/WRF_source_v4.6.1_group"
```

To make a fresh copy, `BUILD` must not exist:

```bash
test ! -e "$BUILD"
rsync -a "$SRC/" "$BUILD/"
rsync -ani --delete "$SRC/" "$BUILD/" | tee /tmp/wrf-copy-dry-run.txt
test ! -s /tmp/wrf-copy-dry-run.txt

cd "$BUILD"
find . -type f -exec grep -Il $'\r$' {} + > /tmp/wrf-crlf-files.txt
cp /tmp/wrf-crlf-files.txt crlf-normalized-files.txt
test "$(wc -l < crlf-normalized-files.txt)" -eq 4838
xargs -d '\n' sed -i 's/\r$//' < /tmp/wrf-crlf-files.txt
test "$(find . -type f -exec grep -Il $'\r$' {} + | wc -l)" -eq 0

test "$(nc-config --prefix)" = /usr
test "$(nf-config --prefix)" = /usr
export NETCDF=/usr
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
printf '32\n1\n' | ./configure 2>&1 | tee configure.log
test "${PIPESTATUS[1]}" -eq 0
test -s configure.wrf
grep -q -- '-fallow-argument-mismatch -fallow-invalid-boz' configure.wrf

./compile em_real -j 4 > compile.log 2>&1
# WRF make uses ignored errors and wrappers can return zero on failure.
grep -q 'Executables successfully built' compile.log
test -x main/wrf.exe
test -x main/real.exe
```

### Recover or re-run a used tree

`BUILD` is disposable; the reference bundle is not.  If the initial copy,
configure, or full compile stopped before producing `configure.wrf` and both
executables, remove only the incomplete disposable tree and repeat the fresh
copy block.  The path assertion prevents this cleanup from expanding beyond
the documented build directory:

```bash
test "$BUILD" = "$HOME/wrf-oracle-build"
if test -e "$BUILD" && \
   { test ! -s "$BUILD/configure.wrf" ||
     test ! -x "$BUILD/main/wrf.exe" ||
     test ! -x "$BUILD/main/real.exe"; }; then
  rm -rf -- "$BUILD"
fi
```

Once the full build exists, it is safe to keep the configured toolchain,
objects, executables, and compile logs.  The instrument patch changes only
`share/mediation_force_domain.F` and `dyn_em/solve_em.F`; restore both from the
read-only bundle through LF-normalized temporary files.  This recovers a fully
applied, partially applied, or rejected patch and then verifies both exact
pristine SHAs before another attempt:

```bash
set -euo pipefail
SRC=/mnt/c/Users/drew/Downloads/WRF_1974_MP55_reference_bundle/WRF_source_v4.6.1_group
BUILD="$HOME/wrf-oracle-build"
FORCE_SHA=aaef43f69eb810809eb890688b840ce894aea9ae1ae99f8266e4fa8c3b9f5518
SOLVE_SHA=e42df5d7db4b6ec4a3b8e2f228a8ec8f9a4c426656093bfcebe58a8de6c3e8f4
FORCE_RESET="$BUILD/share/.mediation_force_domain.F.reset.$$"
SOLVE_RESET="$BUILD/dyn_em/.solve_em.F.reset.$$"

test "$BUILD" = "$HOME/wrf-oracle-build"
test -s "$BUILD/configure.wrf"
test -x "$BUILD/main/wrf.exe" && test -x "$BUILD/main/real.exe"
tr -d '\r' < "$SRC/share/mediation_force_domain.F" > "$FORCE_RESET"
tr -d '\r' < "$SRC/dyn_em/solve_em.F" > "$SOLVE_RESET"
test "$(sha256sum "$FORCE_RESET" | awk '{print $1}')" = "$FORCE_SHA"
test "$(sha256sum "$SOLVE_RESET" | awk '{print $1}')" = "$SOLVE_SHA"
mv -f -- "$FORCE_RESET" "$BUILD/share/mediation_force_domain.F"
mv -f -- "$SOLVE_RESET" "$BUILD/dyn_em/solve_em.F"
rm -f -- "$BUILD/share/mediation_force_domain.F.orig" \
          "$BUILD/share/mediation_force_domain.F.rej" \
          "$BUILD/dyn_em/solve_em.F.orig" \
          "$BUILD/dyn_em/solve_em.F.rej"
test "$(sha256sum "$BUILD/share/mediation_force_domain.F" | awk '{print $1}')" = \
  "$FORCE_SHA"
test "$(sha256sum "$BUILD/dyn_em/solve_em.F" | awk '{print $1}')" = \
  "$SOLVE_SHA"
```

The case directory and dump are run outputs, not reusable inputs.  Before a
re-run, move exactly those recipe-owned paths aside so a pinned dump is never
silently overwritten.  A partially prepared case, partial logs, and a partial
dump are all recovered by the same block; unrelated review cases/dumps are
left alone.  The self-candidate NPZ is derived and may be deleted safely:

```bash
CASE="$BUILD/n1p5-case"
DUMP="$BUILD/dumps/n1p5-d01-d02-force001.bin"
if test -e "$CASE" || test -e "$DUMP"; then
  ARCHIVE="$BUILD/n1p5-previous-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  test ! -e "$ARCHIVE"
  mkdir -p "$ARCHIVE"
  test ! -e "$CASE" || mv -- "$CASE" "$ARCHIVE/case"
  test ! -e "$DUMP" || mv -- "$DUMP" "$ARCHIVE/force001.bin"
fi
rm -f -- "$BUILD/dumps/n1p5-self-candidate.npz"
```

For an already-pristine proven build, the restore is harmless.  In either a
fresh or recovered tree, apply with fuzz disabled and incrementally rebuild:

```bash
cd "$BUILD"
test "$(sha256sum share/mediation_force_domain.F | awk '{print $1}')" = \
  aaef43f69eb810809eb890688b840ce894aea9ae1ae99f8266e4fa8c3b9f5518
test "$(sha256sum dyn_em/solve_em.F | awk '{print $1}')" = \
  e42df5d7db4b6ec4a3b8e2f228a8ec8f9a4c426656093bfcebe58a8de6c3e8f4

patch -p1 --fuzz=0 --dry-run < "$REPO/tools/wrf_instrumented/instrument-med-force.patch"
patch -p1 --fuzz=0 < "$REPO/tools/wrf_instrumented/instrument-med-force.patch"

export NETCDF=/usr
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
./compile em_real -j 4 > n1p5-compile.log 2>&1
grep -q 'Executables successfully built' n1p5-compile.log
test -x main/wrf.exe
test -x main/real.exe
```

On the demonstrated tree, the incremental rebuild took about 7 seconds.  A fresh
serial GNU build took 534 seconds in the toolchain spike.

## 2. Prepare the pinned d01+d02 case

The bundle contains only the 12:00 `met_em` frame.  For this table-arithmetic
spike, make a disposable 12:01 persistence copy by changing its WRF `Times`
and date strings in place.  This is case setup, not a reference dataset.

```bash
set -euo pipefail
BUILD="$HOME/wrf-oracle-build"
REPO=/mnt/c/Users/drew/gpuwm
BUNDLE=/mnt/c/Users/drew/Downloads/WRF_1974_MP55_reference_bundle
CASE="$BUILD/n1p5-case"

test ! -e "$CASE"
mkdir -p "$CASE"
rsync -a "$BUILD/run/" "$CASE/"
cp "$REPO/tools/wrf_instrumented/namelist.input.n1p5" "$CASE/namelist.input"

# The Windows bundle stores this intended symlink as a 32-byte text pointer.
ln -sf CAMtr_volume_mixing_ratio.SSP245 "$CASE/CAMtr_volume_mixing_ratio"
test -L "$CASE/CAMtr_volume_mixing_ratio"

cp "$BUNDLE/met_em/met_em.d01.1974-04-03_12_00_00.nc" \
   "$CASE/met_em.d01.1974-04-03_12_00_00.nc"
cp "$BUNDLE/met_em/met_em.d02.1974-04-03_12_00_00.nc" \
   "$CASE/met_em.d02.1974-04-03_12_00_00.nc"
cp "$CASE/met_em.d01.1974-04-03_12_00_00.nc" \
   "$CASE/met_em.d01.1974-04-03_12_01_00.nc"
cp "$CASE/met_em.d02.1974-04-03_12_00_00.nc" \
   "$CASE/met_em.d02.1974-04-03_12_01_00.nc"
sed -i 's/1974-04-03_12:00:00/1974-04-03_12:01:00/g' \
   "$CASE/met_em.d01.1974-04-03_12_01_00.nc" \
   "$CASE/met_em.d02.1974-04-03_12_01_00.nc"
test "$(strings "$CASE/met_em.d01.1974-04-03_12_01_00.nc" | \
        grep -xc '1974-04-03_12:01:00')" -eq 2
test "$(strings "$CASE/met_em.d02.1974-04-03_12_01_00.nc" | \
        grep -xc '1974-04-03_12:01:00')" -eq 2
```

`nocolons=.true.` is why the filenames use underscores while the internal
WRF time strings use colons.

## 3. Run REAL and one outer WRF step

```bash
set -euo pipefail
BUILD="$HOME/wrf-oracle-build"
CASE="$BUILD/n1p5-case"
DUMP="$BUILD/dumps/n1p5-d01-d02-force001.bin"
mkdir -p "$BUILD/dumps"
test ! -e "$DUMP"

cd "$CASE"
ulimit -s unlimited
./real.exe > real.log 2>&1
grep -q 'SUCCESS COMPLETE REAL_EM INIT' real.log
test -s wrfinput_d01
test -s wrfinput_d02
test -s wrfbdy_d01

export GPUWM_WRF_ORACLE_DUMP="$DUMP"
./wrf.exe > wrf.log 2>&1
grep -q 'GPUWM N1.5 dump: .* tables=128' wrf.log
grep -q 'SUCCESS COMPLETE WRF' wrf.log
test -s "$DUMP"
EXPECTED_DUMP_SHA="$(tr -d ' \r\n' < /mnt/c/Users/drew/gpuwm/tools/wrf_instrumented/n1p5-dump.sha256)"
ACTUAL_DUMP_SHA="$(sha256sum "$DUMP" | awk '{print $1}')"
printf 'N1.5 dump SHA-256: %s\n' "$ACTUAL_DUMP_SHA"
test "$EXPECTED_DUMP_SHA" = "$ACTUAL_DUMP_SHA" || {
  echo 'controller must re-pin EXPECTED_DUMP_SHA after regeneration' >&2
  exit 1
}
```

**Single-source pin:** the expected dump SHA lives ONLY in
`tools/wrf_instrumented/n1p5-dump.sha256` (both recipe blocks read it);
re-pinning after a regeneration is a one-file edit.
**Superseded pins:** the older dump SHA-256
`049ece3d96bb912d094304e2616b2e7021f9ea0ed815da280d95eaead360ebd3`
is retained for provenance only.  Two independent amendments retired it:
(a) the old case ran `nwp_diagnostics=1` and omitted `use_theta_m`
(Registry default 1), so its `t` tables are moist-theta state and must not
feed gpuwm's dry-theta comparison; (b) the extended instrumentation's new
`PROG`/`DBDY` records (before/after med_force_domain prognostics + the
solve-side dtbc oracle) change the prior table-only hash regardless.  The
later diagnostics-only dump SHA-256
`441da3cba30fd1fb958488ba55c7e3aa0e5b11854a92666a3ba73f020f8022cc`
is also superseded: its 12-value `PROG` samples cannot independently reproduce
full boundary tables.  The current pin adds 50 complete `INPT` records (25
arrays for each domain) and was regenerated by the p5n15 controller lane.

The serial CPU `wrf.exe` step took about 230 seconds on the p5n15 controller;
that wall time is normal for this case and does not indicate a hang.  No GPU
is used.

Do not accept process exit status as the only success signal: both REAL and
WRF can exit zero after a fatal path in this source/build configuration.

## 4. Read and compare

The repo Python environment has NumPy.  From PowerShell at the repo root,
read the WSL dump through the distro UNC path:

```powershell
$dump = '\\wsl.localhost\Ubuntu-24.04\home\drew\wrf-oracle-build\dumps\n1p5-d01-d02-force001.bin'
$expected = (Get-Content tools/wrf_instrumented/n1p5-dump.sha256 -Raw).Trim()
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $dump).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "WRF oracle hash mismatch: $actual" }
python tools/wrf_instrumented/dump_reader.py $dump --compact
```

The compact report must show four prognostic samples in
`before d01, before d02, after d01, after d02` order and four boundary-clock
records with ordinals 1..4 and dtbc `15,30,45,60`.  It must also report 50
complete force inputs.  Legacy table-only and diagnostics-only dumps remain
readable, but the producer rejects them by naming their missing inputs.

The reader returns canonical `(vertical, along_boundary, bdy_width)` FP32
arrays.  A gpuwm candidate NPZ must contain exactly the same keys, named
`family.field.side.kind`; for example `state.u.xs.value`,
`moist.qv.ye.tendency`, and `scalar.qni.xe.value`.  Metadata entries may be
included only with names beginning `__`.

The controller producer and comparison commands are:

```powershell
$candidate = '\\wsl.localhost\Ubuntu-24.04\home\drew\wrf-oracle-build\dumps\gpuwm-nest-force001.npz'
python tools/wrf_instrumented/produce_nest_force_candidate.py $dump $candidate --summary
python tools/wrf_instrumented/compare_nest_force.py $dump $candidate --summary
```

The comparator imports both N1.5 gate rows and applies their registered F10
metric elementwise:

```text
abs(gpuwm - wrf) <= 1e-6 * (abs(wrf) + max(abs(wrf_table)))
```

Missing/extra tables, shape/dtype mismatch, or any NaN/Inf fails.  The final
report separately aggregates the maximum value-table and tendency-table
metrics.  Reference dumps remain outside git beside the controller-owned
bundle and must be pinned by path and SHA-256 before a gate run.
