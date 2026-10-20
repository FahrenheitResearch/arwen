Noah-OWP CUDA source regeneration

The reviewed source is NOAA-OWP/noah-owp-modular at
0242a96c2a90b2c1501ff56a1e457a932a49c8ac (Apache-2.0). The port's directory
retains LICENSE and TERMS.md. The source checkout and build intermediates live
under the lane's scratch directory. No native Fortran program or runtime
is installed by this tool or included in the generated CUDA implementation.

`lower.py --source CHECKOUT --work SCRATCH --output GENERATED --f2c TOOL`
lowers the numeric routines through an offline f2c compiler pass. Run
`cuda.py GENERATED gpuwm/hydro/noahowp` to produce physics.cu and schema.json.
The source revision is checked before lowering; the schema records input file
hashes, all field names/types/lower bounds and the routine inventory. The
compiler pass rejects unresolved identifiers and unsupported statements.
Default REAL literals and REAL parameter expressions remain float operands
through the offline pass, which otherwise folds compound constants in double.

The first specialization has four soil layers and three snow layers. Kernel
preparation specializes bounds; it never pads or changes the native profile
ordering. A device guard rejects incompatible counts and root bounds. Schema
slabs are field/level-major and catchment-minor. Calendar components and current
time are resident; domain.sim_datetimes is a setup schedule computed natively
from start + (itime-1)*dt and is represented by those numeric carriers.

Upstream numeric warnings become a resident status bit (1); STOP becomes fatal
bit 2; incompatible specialization is bit 4. Upstream local water-budget
accumulators only feed unused diagnostics. They become private initialized
locals rather than globals shared across catchments. No physical state or flux
is dropped. Fortran DATA iteration counts and tables are also private readonly
inputs to each thread. All 91 computational routines are generated.

Fast mode uses CUDA math. Strict mode disables contraction and reuses only the
existing Arm MIT / Sun fdlibm device math helpers and the MIT CORE-MATH
log10f in log10.cu to reproduce the pinned referee's native REAL libm behavior.
The referee uses the newer correctly rounded log10f; the older Sun log10f
differs in soil thermal conductivity on the sixth captured update.
Native captures, rather than this translation process, determine
whether the numerical implementation passes. Compilation alone is not parity.

Strict solar angles use the existing MIT Arm/CORE-MATH device trig header
(`glibc_trig_flt32.cuh`) to match the referee's glibc 2.43 float functions.
Fast mode continues to use CUDA builtins.
