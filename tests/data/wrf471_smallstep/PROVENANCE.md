# WRF v4.7.1 smallstep reference

WRF source commit: `f52c197ed39d12e087d02c50f412d90d418f6186`, tag `v4.7.1`.
The reference compiles the byte-unmodified `dyn_em/module_small_step_em.F`
and `share/module_model_constants.F`. `tools/smallstep_wrf471_oracle/build.py`
refuses a different SHA256 for either source.

The build uses GNU Fortran 15.2.0, float32 default REAL, `-O0`,
`-ffp-contract=off`, and `-fcheck=bounds`. The only replacement module is a
data-only `grid_config_rec_type` with the fields read by the real source.
No arithmetic or reference subroutine is stubbed. Generated C wrappers retain
the complete original argument lists. The compiled reference library SHA256
is `5a26272a63f17b93fb752226a1db9ff5d461c193dc7d574bc75b05fccf95a086`.
Rebuilding twice into the same directory reproduced that library hash.

Source SHA256 pins:

| Source | SHA256 |
| --- | --- |
| `dyn_em/module_small_step_em.F` | `cabf1a177d50fb0096db79644af20cfe6d75217dbe63ab406a7e29bb54c17634` |
| `share/module_model_constants.F` | `5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062` |

`real-state.npz` retains an 8 by 7 crop with all 49 mass levels and 50 vertical
faces from the 2024-05-25 18Z WRF initialization and its 19Z history. The crop
begins at zero-based horizontal indices (20,20). Raw winds, theta, pressure,
geopotential, dry column mass, hybrid coefficients and map factors remain
float32. The engine's Rust NetCDF reader decodes the input files.

The history file omits inverse density. Its base inverse density is the
unchanged initial ALB; its total inverse density is reconstructed from the
recorded dry theta and pressure through the dry equation of state, then AL is
formed by subtracting ALB. This is an explicitly labelled input diagnostic,
never an expected comparison output. `cases.json` records both source file
hashes and this reconstruction. Other missing invariant coordinate inputs
come from the initial file.

Four stress transformations add steep surface slopes, map factors from 0.35
to 2.5, zero and near-zero perturbations, and southern latitude/Coriolis/wind
signs. Synthetic acoustic increments are small, deterministic disturbances
on these states. The input archives and reference arrays carry source hashes;
the SHA256 manifest pins all fixture and builder files together.

Fortran arrays use `(i,k,j)` memory with one lateral ghost row, the full
vertical stagger, explicit memory/domain/tile bounds, and correct duplicated
periodic faces. Engine arrays use `(k,j,i)`. The contract tests verify the
transpose and ghosts. All physical output words, including retained boundary
rows, are compared. Unstored WRF mass and coefficient workspaces are exposed
by test-only stores of the actual engine registers. Those stores are accepted
only after every ordinary output word agrees with the unobserved native launch.

Canonical comparisons keep WRF's theta-minus-300 representation and its
300 K constant. Full-theta and altered CUDA compilation controls are separate
causal witnesses; they never replace the canonical answers. The expected
arrays come from compiled Fortran. Source-level float32 traces explain
differences and must reproduce actual GPU words; they are not oracle answers.

The recorded GPU word receipts use an RTX 5090, CuPy 14.2.0 and CUDA runtime
13020. Exact GPU output pins describe that environment. Arithmetic counts can
change with another architecture or compiler. Normal tests use the packaged
Fortran fixtures and need no Fortran compiler. They pin measured differences
and native output words, with no tolerance acceptance rule.

These artifacts characterize routine answers and identified differences.
They do not establish full WRF trajectory identity or forecast skill.
