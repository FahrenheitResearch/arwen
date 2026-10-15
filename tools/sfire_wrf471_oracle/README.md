# WRF v4.7.1 SFIRE native oracle

The original reference is WRF tag `v4.7.1`, annotated tag object
`3e7e865a1e59e067b6f8a3c55254c0b19a10597e`, peeled commit
`f52c197ed39d12e087d02c50f412d90d418f6186` from
<https://github.com/wrf-model/WRF>. The upstream filenames use
`module_fr_fire_*`. `SOURCES.sha256` pins every compiled original source,
the registry, driver source and supplied fuel namelists. The full source
archive SHA-256 is recorded in the corpus receipt.

The build compiles original model constants, error services, fire utilities,
fuel physics, the complete fire core, the six-pass model and atmospheric
exchange module. These files are byte-unmodified. Standalone service types
represent the unused serial domain/configuration objects; communication
services are single-rank no-ops. No fire equation, physical constant,
interpolation or boundary calculation is supplied by a service stub.
The atmospheric tracer service selects one tracer for this harness.

```sh
bash tools/sfire_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
python tools/sfire_wrf471_oracle/pack_fixtures.py BUILD_DIR \
    tools/sfire_wrf471_oracle/fixtures
```

The reference uses gfortran, four-byte REAL and INTEGER, `-O0`,
`-ffp-contract=off`, bounds checks and scalar libm. Build receipts record
the compiler, libc, flags, source hashes, harness hashes and every native
binary stream hash. Every reference object, including drivers, is checked
for vector libm symbols. The build refuses a destination inside the
read-only original source tree.

`oracle_io.F90` writes little-endian native words without record markers.
Packing preserves the words and records each array's stream and archive
hash. Device layouts are `(j,i)` and `(k,j,i)`; tracer arrays use
`(tracer,k,j,i)`. Scalar records retain rank zero. All physical and memory
bounds are retained by the core fixtures. The replay grades physical output
words; raw fixtures also retain halos and untouched output sentinels.

The original cases cover all 13 Anderson and 40 Scott-Burgan models plus
the no-fuel row, both wind/slope projection modes, sensible and latent heat,
five moisture classes carried through initialization and two timesteps,
partial-cell and refined-grid consumption, derivative selectors, eight
level-set schemes, RK3 propagation, four reinitialization schemes, initial
state, timed line ignition, nearest segment geometry, ignition-time
crossings, flame diagnostics and the full six-pass model through 60 steps.
The fuel table fixture changes only `nfuelcats=54`, making the source's
last Scott-Burgan row reachable. Original-default and corrected behavior
are kept separate in defect tests.

```sh
python -m pytest tests/test_sfire_wrf471_parity.py \
    tests/test_sfire_atm_wrf471_parity.py \
    tests/test_sfire_driver_wrf471_parity.py -v -s
```

Compiler controls use a separate build directory and the same original
sources. The signalling-NaN control must leave every fixture word unchanged.
The optimized control records every difference. It may call vector libm;
its symbol receipt explicitly identifies that and it is never the reference.

```sh
SFIRE_ORACLE_EXTRA_FLAGS='-finit-real=snan -finit-integer=-99999' \
    bash tools/sfire_wrf471_oracle/build.sh WRF_SOURCE_ROOT SNAN_BUILD
python tools/sfire_wrf471_oracle/verify_compiler_controls.py \
    BUILD_DIR SNAN_BUILD snan-receipt.json
```

`apply_corrected_tg.py` creates a separate, explicitly corrected atmospheric
source and a diff. It leaves the original unchanged. The corrected source
integrates the original approximate Gaussian CDF across mass layers,
normalizes the represented column, supplies the missing relative heights,
uses independent ground/crown sensible and latent sources, and repairs the
original smoke loop's single-column extent. Its fixtures and receipt live
under `fixtures/corrected/` and are labelled as corrected references.

```sh
python tools/sfire_wrf471_oracle/apply_corrected_tg.py \
    WRF_SOURCE_ROOT/phys/module_fr_fire_atm.F CORRECTED_ATM.F \
    --diff CORRECTED_TG.patch
SFIRE_CORRECTED_ATM_SOURCE=CORRECTED_ATM.F \
    bash tools/sfire_wrf471_oracle/build.sh WRF_SOURCE_ROOT CORRECTED_BUILD
python tools/sfire_wrf471_oracle/pack_fixtures.py CORRECTED_BUILD \
    tools/sfire_wrf471_oracle/fixtures/corrected --case corrected_tg
```

The original excessive-moisture fixture proves negative spread and reaction
intensity in the source; its GPU test requires zero spread above extinction.
The strong-wind fixture proves negative slope spread under the source cap;
its GPU test requires nonnegative contributions sharing the 6 m/s budget.
The original smoke fixture proves the skipped interior columns; its GPU
test requires every interior column to receive emissions. Constant-fire
fixtures expose the original inverted before/after propagation switch.
These defects are not targets for word equality.

Two arithmetic findings are covered by the controls. WRF's REAL `**2.0`
calls scalar `powf` at `-O0`; multiplying the value by itself changed two
intermediate reinitialization sign words in the carried model control.
The GPU retains the native real-power operation. The Gaussian math probe
also separates integer cube and polynomial words from `tanh` and CDF words.
The former were exact while a legacy float `tanh` changed 22 of 120 tanh
words by 1 ULP against glibc 2.43. The CORE-MATH implementation restored
zero-ULP agreement. The optimized arithmetic probe explicitly records
`_ZGVbN4v_tanhf`; those vector-library outputs are not the scalar reference.

The routine corpus and six-pass fire model do not establish full coupled
atmospheric agreement or observed perimeter skill. Those require the
separate compiled ideal case and real-data runs.

## Strict and default regime verdict

`regime_receipt.py` fails with exit 1 if a required regime, field, source
label or fixture is absent, a fixture hash changes, or any output word
differs. Strict and default run in separate processes. Both require zero
different words, with no tolerance or unregistered default exception.

```sh
bash tools/sfire_wrf471_oracle/build_regimes.sh WRF_ROOT LANE_ROOT
python -m tools.sfire_wrf471_oracle.regime_receipt \
    --fixtures LANE_ROOT/regime-fixtures --out LANE_ROOT/regime-receipt.json
```

Run the second command through the sprint GPU queue. The corpus includes
20 carried steps each for light-wind flat grass, strong-wind sloping timber,
rain-driven wet fuel, corrected central and Sethian normals, negative-x
split advection, and a strong-wind grass cap/boundary stress control. Timber
uses category 8 and a 0.25 s step; the other controls use category 1 and
1 s steps. The native caller selects supplied-fuel mode, and the grader
checks the actual fuel categories. The stress control retains CFL diagnostics
and is an arithmetic control rather than a forecast qualification.

Initialization and subsequent device state are computed independently.
Reference-produced level sets, fuel parameters and moisture arrays never
seed the replay. Each field carries full-field hashes, different-word counts
and maximum ULP. The same production `FireCoupler._feedback` path aggregates
the 2x2 fire cells and inserts heat and water into seven atmospheric mass
levels. Final NVRTC flags and compiled source/PTX hashes are recorded for
all six SFIRE modules, including the smoke source module.

`correct_regime_sources.py` creates separate, hashed corrected sources.
They retain the declared normal, split-advection, ignition-count, extinction,
spread-cap, chaparral-base, TG and monotonic-fuel divergences. The fuel
correction prevents rounded fuel recovery from creating negative heat and
smoke, as the existing compiled driver-burn capture demonstrates. The
standalone caller supplies the completed physical-boundary halo before
fuel quadrature, which requires its neighbor node state. Original WRF source
pins remain checked before every build; corrected source hashes are separate.

```sh
python -m tools.sfire_wrf471_oracle.regime_receipt --plant-one-ulp \
    --fixtures LANE_ROOT/regime-fixtures --out LANE_ROOT/planted-receipt.json
```

This negative control inserts `nextafterf` in one actual compiled heat-flux
kernel cell. It must exit 1 in both modes. It never changes a host output
array or a production source file. `run_lane.sh` builds, runs the zero-word
gate, runs this negative control and runs focused native/device regressions.
