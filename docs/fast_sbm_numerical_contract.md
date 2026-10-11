# FAST_SBM numerical contract

The selector is admitted through the physics registry. The shared source runs
full warm and cold tiles on the host and CUDA. Prognostic spectra travel through
native state, scalar transport and restart. End-to-end forecast checks are
reported separately from routine parity.

The WRF preset follows the pinned source and reproduces the recorded full
calls bit for bit in float32 and float64. The corrected column preset has the
catalogued deliberate differences plus D19 and D20 below. A finite budget is
not observation-based forecast qualification.

Source: WRF 4.6.1 and 4.8.0 module_mp_fast_sbm.F, LF SHA-256
fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e.
The live 4.6.1 rebuild reproduces 80 mapped streams; 140 live controls pass.
The combined refusal fixtures have no single source stream.

## D19: sedimentation volume

FALFLUXHUCM_Z uses mass-center spacing as divergence volume. On a stretched
grid that differs from DZ8W. The corrected preset uses dz8w*100 in CGS, and
the WRF preset keeps the original metric. The same bottom flux drives rain.
The regression checks physical rho*dz8w mass and has an active-state negative
control using the original metric.

## D20: collision and breakup mass

Native WRF breakup sweeps have mass ratios down to 0.9961623796583445. The
self-collision floors also add small numerical mass. The corrected preset
projects each nonnegative self-collision and breakup spectrum onto its
incoming total mass, preserving its shape, with the summation remainder in
an occupied bin. Negative breakup iterates retain their retry behavior.
Riming splinters take their mass from the dense-ice parent. The WRF preset
retains the source behavior. This is a declared numerical model correction,
not a claim of bitwise equivalence to a bug.

Three-hour, 900-call physical water residuals are -3.0571116753963507e-13
relative on warm columns and 2.269186636236018e-11 on cold columns. Geometry
and winds are fixed; there is no dycore or external tendency. The test gate
is 1e-9 relative and a disabled collision correction breaks it. Exit clipping
remains present.

## CUDA arithmetic and workspace

Direct NVRTC retains subnormals and disables contraction. RawModule appends
its flush-to-zero option after caller options. A bare RawModule run broke
eight float32 gates; direct NVRTC passed 17 replay tests and an edge-tile
memory check had zero errors. The corrected D19/D20 source passed all 17 device gates in default arithmetic.
The packaged production entry also matched the host over the warm and cold
edge replays.

The 200-level fixed density stack copy was removed. The prepared density and
its stride are reused. Tests cover 2, 200, 201 and 257 levels.

## Containment

Only public FSBM scheme code, tables, oracle harness and tests are carried.
The delivery report lists every file and content hash.

## Native input contract

The input reader preserves all 142 bin fields and six number fields. Native
WRF input and export geometry follow Registry/registry.sbm:318. A bulk-only
condensate analysis (HRRR, RAP, GFS) does not determine a unique spectrum,
so at the first call every cell whose 99 hydrometeor bins are empty and whose
bulk condensate exceeds 1e-13 kg/kg receives that condensate mass-exactly
with fixed declared shapes (gpuwm/core/fast_sbm_bulk_start.py: lognormal
cloud and ice, Marshall-Palmer rain, exponential snow and graupel on the
scheme's own mass grid). Cells that already hold bins (restarts, children of
a spectral parent) are untouched. The spectra then relax to the scheme's own
within its spin-up. Public metgrid inputs without condensate start with
WRF's zero hydrometeor spectra and its natural first-call aerosol profile.

Optional polar radar scattering is not implemented. Its concrete exclusion
is separate from the full warm and cold microphysics and is declared in the
reflectivity consumer table.

## Parallel preparation

The production path freezes each halo column independently, then converts
each active spectral column independently. Separate kernel launches order
those writes before the column driver reads neighbor state. There are no
shared writes between preparation blocks. The serial tile entry remains the
WRF carry-preserving oracle path. The same 22 default device checks compare
every output word in both precisions and in warm and cold edge tiles.

## Forecast and restart evidence

The public metgrid case ran 600 seconds through the ordinary CLI on the
corrected scheme, with 73 by 60 mass columns and 44 levels. Liquid cloud,
rain, ice, snow and dense ice all became active. Final health passed. A
300-second checkpoint resumed to the same canonical digest, including
all 288 arrays and scalar state. These are integration checks, not a
claim that the whole dycore is certified against stock WRF.
