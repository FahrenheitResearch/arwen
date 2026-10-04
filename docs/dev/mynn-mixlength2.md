# MYNN mixing length option 2

`bl_mynn_mixlength=2` selects the local mixing length from WRF v4.6.1
`phys/module_bl_mynn.F`, `mym_length` CASE(2), lines 2100-2232. Option 1
remains the default. Both the first-step initialization and the ordinary
turbulence call forward the selector on CPU and CUDA. The source is public
domain; its notice is retained in `licenses/LICENSE-WRF-public-domain.txt`.

The port preserves the source's TKE-weighted vertical integral, stable and
unstable eddy turnover times, mass-flux limits, free-atmosphere transition,
and LES-scale blend. It preserves FP32 expression order; CUDA divisions
use the existing `MYNN_DIV` rounded helper. The source's `Ugrid`, `Uonset`,
`cldavg`, and stable-branch `elb` temporaries do not reach an output and are
omitted. All inputs needed by this option already exist in the column API.

The Fortran harness compiles the unmodified WRF source with GNU Fortran
13.3.0 in Ubuntu 24.04. Source, harness, and output hashes are recorded in
`gpuwm/data/mynn/oracle/mixlength2-provenance.txt`. Run
`tools/mynn_pbl_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR` to rebuild.
The optional second argument `2` selects this branch in each harness.

| WRF comparison | Recorded columns | CPU maximum FP32 ULP |
|---|---:|---:|
| Mixing length | 8 x 12 levels | 0 |
| Initialization, including preserved QKE | 5 x 16 levels | 0 |
| Turbulence | 4 x 12 levels | 0 |
| Assembled driver, warm step | 5 x 30 levels | 0 |
| Assembled driver, cold step | 5 x 30 levels | 1 for stable-column QKE; 0 for every other field |

The one-ULP cold QKE difference is a declared numerical divergence from
this compiler's Fortran result. Its cause is an open item: no leaf has been
bisected against the Fortran for that column. The exact column checks include weak
buoyancy, tiny TKE, zero LES scale factor, deep boundary layers, and large
TKE. `tests/test_mynn_mixlength2.py` contains the comparisons and explicit
breakage comments. The CUDA leaf and initialization tests passed at exact
equality. Two independent GPU processes on RTX 4090 and RTX 5090, both
using NVRTC 13.4, measured identical assembled-driver maxima against WRF.
An RTX PRO 6000 (sm_120) with CuPy 14.2.0 and NVRTC 12.9 reproduced every
cold and warm maximum below with the same bounds:

| Option-2 CUDA field | Cold maximum ULP | Warm maximum ULP |
|---|---:|---:|
| Mixing length `el` | 4 | 0 |
| Heat diffusivity `exch_h` | 3 | 5 |
| Momentum diffusivity `exch_m` | 5 | 6 |
| Twice TKE `qke` | 1 | 2 |
| `tsq`, `qsq`, `cov` | 9, 10, 8 | 10, 4, 6 |
| Stability functions `sh`, `sm` | 5, 3 | 6, 4 |
| Cloud fraction | 32 | 32 |
| Subgrid liquid, ice | 5, 2 | 5, 2 |
| Vapor tendency | 1 | 0 |
| Cloud-liquid tendency | 9 | 4 |
| Wind, temperature, ice, ozone tendencies | 0 | 0 |
| PBL height, maximum mass flux | 1, 1 | 1, 0 |
| Other column diagnostics | 0 | 0 |

These are declared assembled-CUDA numerical residues, including existing
condensation and turbulence rounding. The new test pins these measured
field and step bounds instead of inheriting the broader option-1 limits.
Numerical column agreement alone does not establish forecast skill or
operational HRRR identity.

The new assembled comparison exposed an existing cold-start defect. WRF
passes vapor `sqv` into `mym_initialize`'s `qw` argument, while the port
passed total water `sqv+sqc+sqi`. Both CPU and CUDA now pass vapor for the
cold call, for either mixing-length option. This fix applies by default.
With option 1 the CPU assembled driver now agrees exactly with WRF on all
five cold and warm columns, including resolved liquid and ice clouds. The
old cold-cloud tolerance was removed from the test and Fortran validator.

The vapor input also changes the option-1 CUDA cold comparison. The earlier
comparison was CPU against CUDA on the same total-water input, so it was
consistent; its smaller readings were not taken against a wrong reference.
With vapor, the cloudy cold columns start from a different initialized
state. On the original four-column population, wind-tendency differences
then measure 3,276 ULP for `rublten` and 1,638 ULP for `rvblten`, with
maximum absolute errors 7.147900760e-8 and 4.452886060e-9 m/s². These are
declared CUDA residues on that new cold-cloud population, against a
CPU reference that now agrees with WRF. Only the two cold wind
ULP limits change; both also have measured absolute-error limits. The warm
bounds and every other prior option-1 bound remain unchanged. Both cold
wind limits also passed on RTX PRO 6000 with CuPy 14.2.0 and NVRTC 12.9.

## The model top

WRF's `mym_length` integrates `q*z` over the interfaces with a `DO WHILE`
that has no upper bound. When PBLH plus the entrainment layer (300 to
600 m) reaches the model top, the loop reads `dz` and `qkw` one level past
the column, which is undefined. Both mixing-length options end the integral
at the top interior interface instead, on CPU and CUDA alike, which is the
whole-column integral. The CPU reference used to raise there while the CUDA
kernels ran on; `tests/test_mynn_mixlength2.py` now pins the two to the
same bits for such columns.
