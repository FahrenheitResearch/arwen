# Shin-Hong PBL oracle, WRF v4.6.1

The first WRF numbers ever produced for `bl_pbl_physics=11`.  Shin & Hong
(2015) is the scale-aware descendant of YSU: the same K-profile, PBL-height,
and entrainment machinery, plus a prescribed nonlocal heat-transport profile
and five partition functions (`pu`, `pq`, `pthnl`, `pthl`, `ptke`) that scale
nonlocal and local transport with `sqrt(dx*dy)` relative to the boundary-layer
depth.  The whole point of the scheme is that its answer MOVES with dx, so dx
is a fixture axis here, not a constant.

## What it builds

```
bash tools/shinhong_wrf461_oracle/build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
```

`WRF_SOURCE_ROOT/phys/module_bl_shinhong.F` must hash to
`99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56` -- the
byte-frozen file from the v4.6.1 release tarball (`v4.6.1.tar.gz`, sha256
`b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab562ce9dc50b5f2`); build.sh
refuses otherwise, so the fixture cannot come from an edited file.  The pin is
the file, not a git commit, because the campaign trees on the weather nodes
are tarball extractions.

Built and run 2026-08-03 on weather-node-1 (gfortran 15.2.0, glibc 2.43) from
the Thompson lane's pristine tarball extraction
(`thompson-oracle-lane/wrf461-lane-pristine`), the tree
`oracle-sha256sums.txt` records as `<wrf-4.6.1-checkout>`; every transfer
was sha256-verified in both directions.  Total new node data: < 10 MB.

`oracle-sha256sums.txt` carries the node's paths with their machine-specific
prefixes replaced, in the form the YSU, Noah and Morrison oracles already
use: `<wrf-4.6.1-checkout>` for the WRF extraction and `<oracle-workspace>`
for the build lane holding this directory's `.F90` sources.  Only the path
prefixes changed; every digest is the byte the node computed.

Four artefacts land in `gpuwm/data/shinhong/oracle/`:

| file | contents |
| --- | --- |
| `shinhong-levels.csv` | 30 cases x 6 dx x 40 levels: every per-level input and output of the 3-D `shinhong` entry (arm A: `ctopo=ctopo2=1`, the Registry `topo_wind=0` default), plus the arm-B (`ctopo=0.85, ctopo2=0.7`) momentum tendencies and `exch_h` |
| `shinhong-surface.csv` | per (case, dx): the surface-coupling inputs, `hpbl`, `kpbl`, `wstar`, `delta`, the `u10/v10` blend, both arms |
| `shinhong-partition.csv` | the five partition functions probed directly on a d x h grid spanning both clamps, including the `h == 0` early-out |
| `pow-probe.txt` | bit patterns of every real-exponent power form in the module beside its algebraic decompositions, at -O0 on the oracle's own toolchain |

`libmvec-report.txt` and `oracle-sha256sums.txt` are the receipts.

## The 3-D wrapper is the fixture boundary, on purpose

The YSU oracle drove the inner column scheme (`bl_ysu_run`) and documented
the wrapper identity separately.  Here the fixture calls `shinhong` itself,
so the ndiff-packed moisture array, the `rthblten = ttnp/pi` identity
(module_bl_shinhong.F:253), and the `u10/v10` ctopo2 blend (:1472) are all
inside the pinned boundary.

Optional-argument contract, established against the pinned source: the
wrapper subscripts `ctopo(ims,j)`, `ctopo2(ims,j)`, `regime(ims,j)`
unconditionally (:232, :242) and forwards `wstar`/`delta` to NON-optional
dummies of `shinhong2d`, so formal absence of any of them is an illegal
reference the moment the wrapper runs.  WRF's own driver
(module_pbl_driver.F:1302) always passes every one.  The oracle does too;
the port treats them as required and documents the formal-optionality gap
rather than implementing an arm WRF cannot reach.

## No stub_wrf.F90, on purpose

`module_bl_shinhong.F` has no USE statement and every CALL it makes
(`shinhong2d`, `tridi1n`, `tridin_ysu`, `mixlen`, `prodq2`, `vdifq`) plus the
five partition functions live in the same module.  `nm -u` on the -O0 object
is the receipt: `expf`, `powf`, `tanhf` and libgfortran, nothing else.

## libmvec: this module is NOT like YSU

WRF compiles `phys/` at `-O2 -ftree-vectorize`.  For `bl_ysu.F90` that
setting happened to vectorise nothing.  For `module_bl_shinhong.F` it pulls
in `_ZGVbN4vv_powf` -- glibc's 4-ULP vector pow -- so WRF's own shipping
build genuinely floats some pow loops off the scalar reference.  The oracle
is built at -O0 (scalar `powf`, the tightest defined reference), build.sh
fails on any `_ZGV*` symbol in the reference object, and the
`libmvec_positive_control.F90` arm proves the grep can fire.  The -O2 and
-Ofast objects are kept as evidence (`-Ofast` also rewrites `x**h1` to
`cbrtf`, same as YSU).

## pow semantics, measured not guessed

`pow-probe.txt` (gfortran 15.2.0, -O0, this exact toolchain):

* `x**2.` and `x**pfac` fold to `x*x` -- bitwise, negative bases included;
* `x**3.` and `x**4.` remain correctly-rounded `powf` calls, 1 ULP away from
  the multiply chains at some inputs -- a port that spells them as multiplies
  is wrong in the last ULP;
* `x**h1 == x**(1./3.)` (h1 = 0.33333333 rounds to float32 1/3);
* `x**0.` is exactly 1.0 for every probed x including 0;
* `x**(-1./2.)` agreed with `1/sqrt(x)` and `x**1.5` with `x*sqrt(x)` at
  every probed point (correctly-rounded powf), but the port spells them as
  pow, because two roundings are not a proof of equality everywhere;
* negative bases with integral real exponents (`dux**2.` at :971 with
  `dux < 0`) are prohibited by the Fortran standard and compiled to defined
  C99 `powf` semantics by gfortran: sign preserved for odd, dropped for even.

## Fixture coverage

30 cases x 6 grid spacings (100/250/500/1000/3000/9000 m), chosen to reach
branches rather than to look meteorological:

* dry convective deep and shallow; stable land weak and strong; near-neutral;
* `br` at `+0.0`, `-0.0`, `+/-1.4e-45`, and the smallest normal -- five
  probes on the `br.gt.0` sfcflg compare, because CUDA FTZ/DAZ and the
  sm_120 subnormal behaviour have already cost this project days;
* ocean columns driving the `brcr_sbro` Rossby form, one with subnormal
  `u10` and `-0.0` `v10`;
* `ust=hfx=qfx=0` exactly, and `ust` subnormal with `hfx=+0.0`, `qfx=-0.0`
  (case 13) -- WRF's own `prfac2` 0/0 fires there (`15.9*wstar3/ust3` with
  both cubes flushed to zero, :1010) and writes a NaN heat-tendency column;
  the same artifact the YSU fixture pins.  The NaN lanes are recorded
  exactly as WRF wrote them;
* strong capping (`deltaoh` ezfac/min clamps, :968-969) and weak capping
  (`deltaoh -> hpbl`, `rigs` below `-cpent`, the `entfmin` floor);
* `qc` exactly `0.01e-3` and one float32 step above it, straddling the
  in-cloud `imvdif` test (:1050); in-cloud Ri above the PBL with liquid AND
  ice; strong shear (`Ri < 0`); `prmax` clamp; `gamcrt`/`gamcrq` saturation;
* a 500 m dz grid saturating the `rlamdz` 300 m cap; a PBL that fills the
  column (`kpbl = kte`); subnormal and signed-zero moisture;
* `ust/wstar` placed at ~0.24, ~0.30, ~0.36, ~0.50 and ~1.04 so the `csfac`
  tanh ramp (:844-851) is sampled at its 1.0 floor, its transition, and its
  2.0 peak -- `cslen`, the h that four of the five partition functions see,
  depends on nothing else;
* uniform wind (`dux = dvx = 0` exactly: `rigs = rigsmax`, the ufxpbl/vfxpbl
  zero branches) with `corf = 0`; negative `corf` (the `f = max(corf,eps1)`
  clamp in mixlen, :2001); `dy = 4*dx` (the `sqrt(dx*dy)` contract); dt = 6 s
  beside the standard 45 s;
* input TKE at the `shinhonginit` cold-start floor everywhere, plus two
  developed-TKE profiles so `prodq2`/`vdifq` are exercised away from the
  `epsq2l` floor; `shinhong_tke_diag = 1` everywhere except cases 22/27
  (below).

`kpbl` spans 2 to 40 and `wstar` 0 to 5.36 m/s across the fixture; `delta`
reaches its 100 m clamp; the direct partition probes reach both the pmin and
pmax clamps of all five curves.

## Not covered, deliberately

* `shinhong_tke_diag = 1` with `kpbl = kte`.  WRF would read
  `q2xk(kpbl+1) == q2xk(kte+1)` (module_bl_shinhong.F:1557), one past the end
  of an array declared `kts:kte`.  Case 22 (PBL fills the column) therefore
  runs with `tke_diag = 0`; a fixture row there would carry whatever the
  stack held.  The port guards the read (`kpbl < kte`), which is the defined
  behaviour; the divergence is recorded in the registry warning instead.
* The exact `|rigs + cpent| <= 1e-6` pole test (:976).  Constructing a
  whole-column input whose `rigs` lands within 1e-6 of 0.4 in float32 is
  fixture-brittle; both sides of the pole are covered instead (cases 14/15
  bracket it, case 26 pins the `rigsmax` branch).
* `csfac = 1` via `wstar == 0` (:849).  Unreachable: the branch sits inside
  `pblflg`, and `pblflg` requires `sflux > 0`, which forces `wstar > 0`.
  Dead code in WRF, dead code in the port.

## Measuring the port

```
python tools/shinhong_wrf461_oracle/validate_shinhong_oracle.py [FIXTURE_DIR]
```

Prints per-field ULP; asserts nothing.  `tests/test_shinhong_wrf461_parity.py`
is the gate, and its module docstring carries the attribution for every
number in it.
