# plumerise_frp_gsl full-solver provenance

Reference: GSL ccpp-physics 3e6660c6df54e95a0871e990c2294dd397ae3860, Apache-2.0.
Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
glibc: Ubuntu GLIBC 2.43-2ubuntu2.4, 2.43.
Flags: -O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none;
WRF additionally uses -fallow-argument-mismatch and its EM preprocessor defines.
GSL reference machine stub: kind_phys=4. All public compiled or extracted files
pass their source SHA-256 checks. No -O0 object imports _ZGV symbols; the -Ofast
positive control imports _ZGVbN4v_expf.

## Original and observed solver

The original driver calls the real ebu_driver -> plumerise -> MAKEPLUME.
The observed build adds write-only trace storage for local ztopmax, per-group
bounds and outer time-loop counts. Every numerical statement stays unchanged.
Before publishing diagnostics, the driver compares every original/observed
emission word, not a tolerance. This comparison passed for every case.
`build_plume_observed.py` checks the original source pins before transformation
and records the generated observer-source hashes. There is no substituted plume.

WRF compile-only Registry stubs set biomass_burn_opt=0 and chem_opt=0. The
original driver receives a preinitialized generic three-row surface emission
vector; unavailable mechanism-specific species copies and MOZART scaling are
not exercised. Emissions are still processed by the original generic column
logic. GSL calls the real driver for its first row; the other rows replay its
species-independent, left-associated distribution statement.

## Cases and outputs

132 cases: nz=49 and 59 with different nonuniform spacing; dry and
moist well-mixed daytime layers, a stable nighttime inversion, strong wind
shear, high terrain with 750 hPa surface pressure, and a deep moist tropical
column. These are synthetic column inputs, not observed forecast qualification.
FRP 0; below/equal/above 1e7; 1e8; below/equal/above 1e9; 5e9; 2e10 and 3e10 before the cap. Windy/deep-PBL override is admitted and rejected across the ladder.

The binary writer records model-grid ebu, 200-level final w/t/qv/qc/qh/qi/radius,
ztopmax(2,4), steps(2,4) and group bounds. Only group slot 1 is used; slots 2..4 stay zero.
GSL additionally records final overridden k_min/k_max and flam_frac. `steps`
counts outer model iterations, not the three microphysics substeps.
Profile units: w m/s, t K, qv/qc/qh/qi kg/kg dry air, radius and tops m AGL.

## Measured float32 parity

Every output below is bit-identical, including exact integers and zero signs.
The full audit evaluates the same NumPy scalar statements with optional Numba
CPU acceleration (fastmath false, scalar glibc bindings). Ordinary tests use
unaccelerated NumPy and the portable glibc transcription. Every unique expf and
powf argument from all original solver calls was checked against the portable
helper; LIBM-ULP.json records zero ULP on the complete argument sets.

| Output | max ULP / mismatch | nonzero | words |
| --- | ---: | ---: | ---: |
| ebu | 0 | 0 | 21384 |
| ztopmax | 0 | 0 | 1056 |
| steps | 0 | 0 | 1056 |
| solver_k_min | 0 | 0 | 528 |
| solver_k_max | 0 | 0 | 528 |
| k_min | 0 | 0 | 132 |
| k_max | 0 | 0 | 132 |
| flam_frac | 0 | 0 | 132 |
| w | 0 | 0 | 26400 |
| t | 0 | 0 | 26400 |
| qv | 0 | 0 | 26400 |
| qc | 0 | 0 | 26400 |
| qh | 0 | 0 | 26400 |
| qi | 0 | 0 | 26400 |
| radius | 0 | 0 | 26400 |

The CPU-only rounding-shim audit of the assembled CUDA source also matches all
cases and both cached redistribution entries bitwise. This is explicitly not
an execution of the CUDA kernel on a GPU. Device tests are marked gpu and use
requires_gpu; they remain for integration to run.

## State definition and poison experiments

Each fixture column begins with zeroed module/coms state and a rebuilt 100 m
plume grid. All state is retained between imm passes and between active landuse
groups in that column. WRF's n_setgrid/ncall saved controls otherwise initialize
the grid only once, so the wrapper explicitly rebuilds it after a column reset.
GSL dbg_opt is false before any call; its unwritten icall is only a print gate
(module_plumerise.F90:80-93) and affects no computed value.

Field-by-field poison uses +1e10 for real scalars, -1e10 for arrays and 100000
for integers. Geometry arrays and zsurf are rebuilt canonical grid inputs,
not poisoned. Negative array poison keeps unwritten w tails below the plume-top
search threshold. All-fields and forward/reverse retained-state runs are also
recorded. STATE-CARRY.json lists every tested field; an empty result means no
monitored output changed, not that no memory read occurred.

WRF INITIAL:979-1000 writes only 1..kmt. scl_advectc:1294 reads rho(kmt+1) in
an unused top flux. set_flam_vert:377-403 consumes retained W_VMD tails for its
unused VMD diagnostic. WBAR/DQSDZ are read by initial WATERBAL, with no measured
output effect here. These same-column scalar carries remain exactly as written.
EVAPORATE:1998 and SUBLIMATE:2268 can read CVI(1); fallpart:1688-1707 writes CVI
only for k>=2. Index-isolated -1e10 CVI(1) poison changes one WRF case's emission
and top, plus counts in four cases. Poisoning CVI(2:nkp) changes none of the
monitored outputs. The defined port value of the never-written CVI(1) is zero.
Normal forward/reverse retained columns change saved tails, not emissions,
injection indices or plume tops, in both case sets. Zeroing a new column removes
that order dependence; deliberate same-column state remains.

A separate -finit-real=snan/-finit-integer=-999999 original/observed build changes
no published word in any case (LOCAL-POISON.json). Unwritten ixx is a disabled
WRF debug argument; GSL icall remains disabled debug state. No numerical local
read dependence was observed by this poison build.

## Precision and conventions

WRF's folded cpor=1/rcp is 0x405fffff; GSL's cp/rd is 0x40600000. They must
remain different. Fortran-folded words are recorded in plume-folded-words.json,
not re-folded by NVRTC. Integer powers use libgcc's square-and-multiply order.
The invalid negative finite base/noninteger powf word is explicitly 0xffc00000;
negative infinity follows glibc's special-value rule instead. This reproduces
unused WRF fallpart VTC calculations without a platform-dependent NaN sign.
GSL has use_last=1, a .01 dt floor and 1e-20 guards. WRF keeps its source variant.

GSL injection uses one-based exclusive upper bounds. WRF injects inclusively
per group and applies dz only above the surface, preserving surface ebu. The
landuse process reports the union of group bounds and sum(mean_fct), not a
normalized fraction; group-specific cache values reproduce the original sums.
Landuse incoming emissions are the source's smoldering convention; no total-mass
normalization or MOZART scaling was invented.
The port accepts per-column wind/PBL inputs instead of GSL's erroneous j=1
indexing at module_plumerise.F90:147-148. A row beyond j=1 therefore uses its own
column, an explicit driver bug correction. The column oracle has j=1.

## KIND=8 measurement, not a gate

The same float32 sounding inputs are promoted, then all GSL-kind calculations
use kind_phys=8. oracle_io converts outputs to f4 for reporting. Final k_min,
k_max and ztopmax are identical to KIND=4 in these 132 cases. Internal-profile
differences are measured in KIND8-DIFFERENCES.json:

k_min: max ULP 0, 0/132 different, max absolute 0.0.
k_max: max ULP 0, 0/132 different, max absolute 0.0.
ztopmax: max ULP 0, 0/1056 different, max absolute 0.0.
ebu: max ULP 2, 106/21384 different, max absolute 1.1920928955078125e-07.
w: max ULP 1805636118, 5977/26400 different, max absolute 0.12485361099243164.
t: max ULP 9069, 18272/26400 different, max absolute 0.1383819580078125.
qv: max ULP 3435078, 6592/26400 different, max absolute 3.491947427392006e-05.
qc: max ULP 894098357, 3264/26400 different, max absolute 4.1260384023189545e-05.
qh: max ULP 769912544, 2959/26400 different, max absolute 2.4619977921247482e-05.
qi: max ULP 2674598, 2122/26400 different, max absolute 2.8557959012687206e-05.
radius: max ULP 15042, 17118/26400 different, max absolute 3.67236328125.

## Reproduction commands

From the lane scratch root on the CPU node, with harness files from this patch:

```
bash harness/build_plume_oracles.sh ../aq-smoke/wrf-src final ../aq-smoke/venv/bin/python
GPUWM_NO_LOCAL_GPU=1 NUMBA_NUM_THREADS=1 PYTHONPATH=python:. ../aq-smoke/venv/bin/python audit_plume_cpu.py fixtures all-cpu.json
GPUWM_NO_LOCAL_GPU=1 PYTHONPATH=. ../aq-smoke/venv/bin/python audit_plume_state.py . ../aq-smoke/wrf-src
```

`fixtures` contains the observed WRF and KIND=4 GSL trees copied from `final`.
`python` is the scratch-only optional Numba install, removed at handoff. Source
copies are under the lane scratch; no other checkout or GPU is used.

From the work-folder root on the CPU host:

```
$env:GPUWM_NO_LOCAL_GPU='1'
$env:PYTHONPATH='.'
python -B tools/chem_wrf471_oracle/mutate_plume_parity.py
python -B -m pytest -q tests/test_chem_plumerise_wrf471_parity.py tests/test_chem_plumerise_frp_parity.py tests/test_chem_plumerise_process.py -m 'not gpu'
```

The NumPy test subsets are named in each test module: WRF nz49_s1_f05 and
nz59_s4_f04; GSL nz49_s1_f03 and nz59_s4_f05. Full-case execution takes longer
than one minute without acceleration. Mutation removes BURN's water conversion
factor in one arm at a time; both full-solver CPU tests fail and exact source
bytes are restored in finally. The earlier .85->.84 driver mutation is retained
in measure_ebu_distribute.py and the subset provenance.

Compiled public source hashes:

```
e784083e907aa1660b30db79dcac9dbe0d6a9632de8e672b491bdfdc33bf4396  gsl-ccpp-physics/physics/smoke_dust/module_smoke_plumerise.F90
b11265e926c10fd06e41d2cc84e6dcc0aca8337577eb2054870a6bed280f2b8c  gsl-ccpp-physics/physics/smoke_dust/module_zero_plumegen_coms.F90
f8c3fd2beb9157cdb5f3b3844981034a47ae5352ce9fa957afa8d0b1d7fa7994  gsl-ccpp-physics/physics/smoke_dust/module_plumerise.F90
27c5dc0503decb631f4e057241f35032b9a84b2f84616d8718dcee3c987ec577  gsl-ccpp-physics/physics/smoke_dust/rrfs_smoke_config.F90
```
