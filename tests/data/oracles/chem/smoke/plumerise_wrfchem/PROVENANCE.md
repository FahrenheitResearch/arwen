# plumerise_wrfchem full-solver provenance

Reference: WRF f52c197ed39d12e087d02c50f412d90d418f6186, public domain.
Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
glibc: Ubuntu GLIBC 2.43-2ubuntu2.4, 2.43.
Flags: -O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none;
WRF additionally uses -fallow-argument-mismatch and its EM preprocessor defines.
GSL reference machine stub: kind_phys=4. All public compiled or extracted files
pass their source SHA-256 checks. No -O0 object imports _ZGV symbols; the -Ofast
positive control imports _ZGVbN4v_expf.

## Original and observed solver

The original driver calls the real plumerise_driver -> plumerise -> MAKEPLUME.
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

108 cases: nz=49 and 59 with different nonuniform spacing; dry and
moist well-mixed daytime layers, a stable nighttime inversion, strong wind
shear, high terrain with 750 hPa surface pressure, and a deep moist tropical
column. These are synthetic column inputs, not observed forecast qualification.
Four groups individually, two mixtures, zero fraction sum, zero fire-size sum and zero emissions.

The binary writer records model-grid ebu, 200-level final w/t/qv/qc/qh/qi/radius,
ztopmax(2,4), steps(2,4) and group bounds. All four group slots are meaningful when active.
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
| ebu | 0 | 0 | 17496 |
| ztopmax | 0 | 0 | 864 |
| steps | 0 | 0 | 864 |
| solver_k_min | 0 | 0 | 432 |
| solver_k_max | 0 | 0 | 432 |
| w | 0 | 0 | 21600 |
| t | 0 | 0 | 21600 |
| qv | 0 | 0 | 21600 |
| qc | 0 | 0 | 21600 |
| qh | 0 | 0 | 21600 |
| qi | 0 | 0 | 21600 |
| radius | 0 | 0 | 21600 |

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
only for k>=2. Index-isolated -1e10 CVI(1) poison changes nz49_s4_f06: group 2 upper top
600 -> 900 m, upper injection index 6 -> 7, and six emission words. It changes one WRF case's emission
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
800c6015e9be32797cf6331f2ffd981bd408fe1d1139c3825fdc50a777743c4b  chem/module_chem_plumerise_scalar.F
f8146d4d94c672652718288c8765a652ce2e9ce71e92e6d32424a59f175e901f  chem/module_zero_plumegen_coms.F
9cfd34a5096048108e21b060d4f065389b09f5d0650439fd1d37245473265390  chem/module_plumerise1.F
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  share/module_model_constants.F
```
