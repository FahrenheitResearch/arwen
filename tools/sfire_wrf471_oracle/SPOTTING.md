# Firebrand controls

The GPU implementation uses WRF v4.7.1 `module_firebrand_spotting.F` and
`module_firebrand_spotting_mpi.F`. The original files are copied unchanged in
`reference/spotting`. Their SHA-256 values are
`e68e4b4ddbde6fdc117163e3b396cff3a11bfc7f2ac586de1525707ba82092e4` and
`916bb9cd7b4405a49a4d1bcd4ae0adf98c33e568d55edf7956d1fb2afd4b3a21`.

`extract_spotting.py` extracts original routine bytes for the helper harness.
`build_spotting_driver.sh` compiles the complete original module with the
explicit source corrections recorded by `correct_spotting.py`, a minimal
grid/configuration service and checked array bounds. It also preserves the
original checked build's failure. No mathematical replacement is used as the
full-driver reference. The MPI harness compiles the unchanged MPI module.

The controls cover initial properties, burnout, settling, box interpolation,
particle meteorology, two-pass advection, seeded/unseeded release heights,
stable insertion, imported source/lifetime records, ID rollover, approximate
release ranking, full atmospheric column preparation and complete driver
frames. The full driver uses flat and varying terrain, moist theta, unequal
refinement and unequal horizontal spacing/metrics. Every frame includes
generation clocks, particle identifiers/properties, landing diagnostics and
history resets. Both device fields and mapped pinned-host fields are graded,
including padded fine-grid row strides. Receipts list every compared word.

The source's GNU Fortran random stream is reproduced by its seeded
xoshiro256** transition and initialization. Release generation reseeds the
stream, so checkpointed controls and the integer step clock determine the
next samples. The algorithm and seed packing follow
[the GNU Fortran intrinsic implementation](https://github.com/gcc-mirror/gcc/blob/releases/gcc-15.2.0/libgfortran/intrinsics/random.c).

## Default corrections

| Original behavior | Corrected behavior and control |
| --- | --- |
| Initial spherical mass uses diameter squared | Diameter cubed; original and corrected property controls are separate |
| Full-level theta assignment has a mass-level shape mismatch | Assignment restricted to physical mass levels; original checked driver fails and corrected driver executes |
| Ground height is overwritten before upper-level subtraction | Preserve the ground altitude while converting every level to AGL |
| Release UNPACK mask includes unselected cells | Use the release-potential mask for positive and zero threshold branches |
| Short random seed arrays are smaller than the intrinsic requires | Repeat the source seed pattern to fill the required words |
| Deposition domain tests use OR | Require all four bounds with AND |
| Fully consumed particles can become NaN and count as landings | Remove consumed/invalid particles before deposition |
| Zero landing likelihood is divided by zero | Preserve finite zero likelihood |
| Y transport uses X spacing and metric | Use Y spacing and metric; unequal-spacing driver control |
| Particle height metadata labels an index | Output AGL height in meters |
| All neighbor sends precede receives | Post receives before sends; the original large-packet control reaches its 15-second deadline while the production transport completes |

The original source remains available for each labeled defect control. Bare
runtime defaults use the corrections. The direct NVRTC route disables FMA
contraction and FTZ and uses round-to-nearest division and the existing
scalar math implementations.

## Ownership and execution

Native spotting runs after the completed atmosphere step and its halos,
before an external chemistry step, on the innermost fire domain. It is active
when `ifire=2` and `fs_firebrand_gen_lim>0`. Its original diagnostics measure
passive firebrand transport, deposits and spotting likelihood. The source
does not ignite the level set from those likelihoods.

There is one particle owner per domain. Native fire tiles partition the
same global fire state. An atmospheric streamed tile borrows the particle
owner and does not advance it. After all atmospheric scatters finish, the
global finalizer performs generation, transport, removal, deposition and
history once. Completed meteorology is copied only for occupied bounded
column groups with six-cell dependency halos. Fine fields and persistent
landing planes may remain mapped pinned-host bytes, with all field
arithmetic executed by the CUDA kernels. Release buffers are allocated
after the exact candidate count, including tied potentials.

Checkpoint payloads keep native float32 and int32 fields, generation clocks,
history-reset state, source controls and every particle property. History
uses the registry's coarse, fine, scalar and particle dimensions. Particle
vectors are published when `trackember` is enabled. The eight-neighbor GPU
packing order and optional MPI byte transport retain the native real and
integer row order. MPI transport receives and empty messages are tested on
the actual nine-rank topology and with packets larger than eager buffers.

`receipts/spotting-5090.json` is the wordwise routine/full-driver receipt.
`receipts/spotting-transport-mpi.json` records the large-message transport
control. Coupled disk/history and atmospheric streaming controls are separate
integration evidence, not a wildfire forecast skill measurement.
