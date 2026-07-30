# WRF v4.6.1 NSSL CUDA-slice oracle

This harness compiles NCAR WRF v4.6.1's
`phys/module_mp_nssl_2mom.F` at commit
`d66e442fccc04111067e29274c9f9eaccc3cef28` and calls its exact
`calc_eff_radius`, `calcnfromq`, `nssl_2mom_gs`, `sediment1d`, and `NUCOND`
routines after native
option-18 initialization.  The `nssl_2mom_gs` vectors independently isolate
warm-rain self-collection/breakup, Ziegler cloud-to-rain autoconversion, and
rain collection of cloud droplets.  A snow-only cold slab isolates the native
two-moment snow aggregation number sink.  An ice-only cold slab isolates
column-ice vapor deposition, latent heating, and the default `iscni=4`
100-micron depositional ice-to-snow conversion.  A separate warm-rain slab
isolates two-moment rain evaporation, vapor transfer, and latent cooling.
A coupled frozen slab exercises signed ice and snow vapor exchange, their
shared two-pass saturation limiter, latent heating/cooling, proportional
number loss during sublimation, and depositional ice-to-snow conversion.
The direct `NUCOND` slabs isolate clear-air warm-cloud activation, the
adjacent existing-cloud water adjustment, and the default `irenuc=2`
cloud-interior renucleation branch.  Together they cover two-pass saturation
adjustment, Twomey droplet nucleation, predicted-CCN depletion/restoration,
full and partial cloud evaporation, adaptive RK2 condensation, the native
vertical and half-condensed-mass renucleation gates, and droplet mean-mass
bounds.
Vapor is held above saturation to disable rain evaporation.  The
autoconversion slab begins with no rain, so accretion and rain
self-collection are identically zero during the process sweep.
The rain-, snow-, and ice-sedimentation columns directly exercise the native
adaptive multi-substep q/N fallout.  Rain covers the hybrid Method I+II
size-sorting correction; snow and ice cover the default Method-II number
bound and their distinct mass/number fall-speed splits.
An independent rain--cloud-ice slab exercises reciprocal `qraci` and `qiacr`
collection, the native freezing heat budget, transfer into predicted-volume
graupel, and the unmodified official source's mass-only legacy fallback.
A mixed-phase slab isolates native snow/graupel/hail melting, liquid soaking,
wet growth, and liquid shedding, including cold cloud riming and warm rain
collection on the dense categories.
A paired secondary-ice slab covers contact and homogeneous freezing,
type-II Hallett--Mossop splintering, riming-driven ice/snow-to-graupel
conversion, and the initialized-default graupel-to-hail conversion.

The official file must have SHA-256
`5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3`.
The build copies it and applies `visibility.patch`, which adds one PUBLIC
declaration and changes no executable statement.  This is necessary because
WRF calls these routines only from inside its public NSSL driver.  The
effective-radius output clamps are copied verbatim from the immediately
adjacent driver block at `module_mp_nssl_2mom.F:3172-3210`.

Run on Linux with GNU Fortran:

```sh
./build.sh /path/to/WRF-v4.6.1 /new/oracle-build
```

The effective-radius fixture covers empty and sub-threshold cells plus a wide
range of mass/number ratios.  The initial-state fixture covers empty,
sub-threshold, mass-only, negative-moment, and already initialized states for
all six hydrometeor categories, hail, CCN depletion, and density moments.
The self-collection fixture spans the 80/100/610/2000-micron native branch
boundaries, density variation, exponential breakup efficiency, process
shutoff, and the timestep number-depletion limiter.  The autoconversion
fixture spans cloud radii on both sides of the native 7.51-micron threshold,
several cloud-water and air-density states, timestep depletion limiting, and
the native post-process cloud/rain mean-volume bounds.
The sedimentation fixture spans empty levels, sharp and smooth rain profiles,
density and layer-depth variation, one- and multi-substep CFL branches,
surface export, and both rain-number correction paths.
The snow-sedimentation fixture spans the 10-micron and 10-mm native size
bounds, positive mass with a missing number moment, sparse/sharp/smooth
profiles, density and layer-depth variation, time steps from 0.5 to 300
seconds, surface export, and the default mass-weighted Method-II number
correction.
The ice-sedimentation fixture spans the native 6.88e-13 to 1e-8-kg particle
mass diagnosis, positive mass with a missing number moment,
sparse/sharp/smooth profiles, density and layer-depth variation, time steps
from 0.5 to 300 seconds, adjusted-Ferrier mass/number velocities, surface
export, and the default mass-weighted Method-II number correction.
The rain-cloud-accretion fixture spans the initiation-radius gate, both sides
of the 50-micron collector-radius split, cloud/rain distribution bounds,
density variation, and mass/number depletion limits.  Its oracle program uses
native NSSL namelist controls to disable autoconversion, rain
self-collection, and evaporation without changing the default accretion
equations.
The rain-evaporation fixture spans empty rain, the native mean-volume range,
temperature/pressure/density variation, relative humidities from 1 to 99
percent, Wisner ventilation, and the 10-percent mass/number depletion cap.
Only warm rain is populated, with autoconversion and self-collection disabled
through native namelist controls.
The clear-air activation fixture spans the native 0.4-percent adjustment gate,
temperature/pressure/density variation, vertical velocities from 0.05 to
20 m/s, CCN-limited activation, the initial 4-micron-radius limiter, and
cloud-free CCN restoration.  The accessibility patch exposes `NUCOND` but
changes no executable source statement.
The cloudy-water fixture spans complete and partial cloud evaporation,
near-saturation no-ops, condensation through the native 0.5-percent
cloud-interior-renucleation boundary, temperature/pressure/density and
timestep variation, predicted-CCN restoration, cloud cleanup, and both
droplet mean-mass bounds.  Rain and frozen moments are empty by construction.
The cloud-interior-renucleation fixture spans 0.51-89 percent initial water
supersaturation, vertical velocities from -2 to 30 m/s, predicted-CCN and
droplet-number variation, condensation-mass limiting, and time steps from
1 to 60 seconds.  Each case is a three-level direct `NUCOND` call with cloud
water but no rain or frozen moments, so the center-level output isolates the
default `irenuc=2` continuation without executable-source changes.
The primary-ice-nucleation fixture spans the strict 268.15-K and ice-
saturation gates, zero/negative/positive updrafts, reversed and sharp nuclei
profiles, vapor and 1e6-m-3 concentration limiting, layer depths from 100 to
2000 m, and time steps from 0.1 to 60 seconds.  Its three-level direct driver
call exposes WRF's positive-updraft upwind nuclei gradient (center minus
below), coupled vapor/ice/number/latent-heat updates, and zero snow guard.
The 48-row fixture SHA-256 is
`b30fed325a154c265e57b8ed037903f48ae00a44f7e6294b0c1d1e6b0ced3576`.
The cloud-ice-riming fixture isolates collection of cloud water by cloud ice
(`qiacw`).  It spans empty ice and cloud, the strict 15-micron cloud-droplet
and 30-micron ice-diameter gates, the exact 273.15-K freezing gate, active
cloud/ice sizes, temperature/pressure/density variation, time steps from
0.1 to 60 seconds, and the native ten-percent mass/number depletion cap.
Native namelist controls disable primary ice nucleation, ice autoconversion,
deposition, fragmentation, and neighboring frozen processes.  The clean
48-row fixture SHA-256 is
`d727d38d21fe51f8fbfb02a38e6f246c7621cd6f30dfde60b5247dd195a87ce5`.
The earlier v1 oracle is retained as rejected evidence: its 274-K case entered
the native cloud-ice melting branch and its 500-micron case entered the native
oversized-ice number cleanup, so it was not an isolated `qiacw` oracle and is
not used for CUDA acceptance.
The snow-cloud-riming fixture isolates collection of cloud water by snow
(`qsacw`).  It spans empty cloud/snow states, cloud diameters from 8 to 80
microns, snow diameters from 20 microns to 8 mm, temperature/pressure/density
variation, time steps from 0.1 to 60 seconds, and the native independent
ten-percent mass/number depletion caps.  Every temperature is below -15 C to
make native snow aggregation identically zero, while `depfac=0` and
`iglcnvs=0` suppress vapor exchange and snow-to-graupel conversion.  The
48-row fixture SHA-256 is
`39d7741240bc78a7b1a848c8842641fcf5abd9a3fe94f2ea0fc25db9d75a2029`.
The graupel-cloud-riming fixture isolates collection of cloud water by
predicted-density graupel (`qhacw`).  It spans empty cloud/graupel states,
cloud diameters from 4 to 80 microns, graupel diameters from 300 microns to
8 mm, graupel densities from 170 to 900 kg/m3, temperature/pressure/density
variation, time steps from 0.1 to 60 seconds, and the native independent
fifty-percent mass/number depletion caps.  The accepted v2 fixture raises the
smallest low-density graupel diameter from 100 to 300 microns so WRF's native
moment cleanup remains inactive; the rejected v1 evidence is retained
separately.  The clean 48-row fixture SHA-256 is
`326b64f1d65f0568fc69c5ca237bb0072893e095250ad3a93c7102435730b9df`.
The hail-cloud-riming fixture isolates collection of cloud water by
predicted-density hail (`qhlacw`).  It spans empty cloud/hail states, cloud
diameters from 4 to 80 microns, hail diameters from 300 microns to 40 mm, hail
densities from 500 to 900 kg/m3, cell depths from 250 to 1350 m,
temperature/pressure/density variation, time steps from 0.1 to 60 seconds,
and the native independent fifty-percent mass/number depletion caps.  The
40-mm, 60-second row exercises the qhlacw-specific hail fall-speed limit of
cell depth divided by time step.  The pre-admission v1 evidence is retained
separately because it did not reach the depletion cap.  The accepted clean
48-row v2 fixture SHA-256 is
`b7b825490ddb9ed25984bb52105a17e1da77f81a25e755c5af5cdcf7a0564b98`.
The rain--ice collection/freezing fixture spans empty rain and ice, the
100-micron rain-collector and 10-micron ice-efficiency gates, 150-micron
rain-tail interpolation, temperatures across the 268.15-K `qraci` mass gate
and 270.15-K intended `qiacr` gate, pressure/density variation, and time
steps from 0.1 to 60 seconds.  It exercises both independent ten-percent
depletion caps, graupel mass/number/volume growth, and latent heating.  WRF
v4.6.1 lines 16754--16886 contain a misplaced `ENDIF`; therefore the 271-K
coexistence row intentionally records the official legacy mass-only `qiacr`
fallback, while `craci` continues collecting ice number above 268.15 K.
The accepted v4 fixture adds a small-diameter, high-rain-water case so the
ice-mass limiter is reached.  Its SHA-256 is
`f764c255324955782f947b66831d168646e9fda6e99abfdb17a5d59fbde3b852`.
The frozen cross-collection fixture calls the same public process driver with
vapor fixed at the exact native liquid-saturation lookup node and valid
`evapfac=0`/`depfac=0` controls, so warm and frozen vapor exchange are inert.
All 80 rows remain below 242 K to exclude the later graupel--hail category
conversion.  The matrix spans snow collecting ice; graupel and hail
collecting ice, snow, and rain; the graupel snow-size/density/cloud gates;
the hail ice/cloud/temperature gate; the native two-moment snow--rain no-op;
and the mandatory hail `dz/dt` velocity cap.  Clean pair rows verify frozen
and liquid predicted-volume routing, including WRF's cold graupel--rain
500-kg/m3 rime-density path and hail--rain's 900-kg/m3 path.  Dedicated mixed
rows independently reach 30-percent ice depletion across snow/graupel/hail
and 20-percent snow/rain depletion across graupel/hail for both mass and
number, while a high-count interior dense-moment row remains an exact no-op.
The accepted fixture has zero vapor, cloud, or collector-number drift, zero
frozen-only temperature drift, and maximum five-category water closure of
`2.5125e-9`.  Its SHA-256 is
`8be24224229110df04bf719ad874552b01b490502b2e8584f416c7262f3c52ad`.
The melting/liquid-shedding fixture spans 24 physical cases at four time
steps.  It covers the strict 243.15-K wet-growth and 273.15-K melt gates;
snow, graupel, and hail ventilation/melting; retained-liquid soaking;
cold cloud riming; warm rain collection; rain-number routing; liquid
shedding; latent heating; predicted dense-category volume rewrites; and the
fifty-percent cloud source cap.  Neighboring nucleation, vapor exchange,
autoconversion, cross-collection, aggregation, and category-conversion
branches are disabled through native namelist controls.  The harness derives
the production `ventr`, `ventc`, and `c1sw` defaults from their initialization
formulas, rather than substituting test constants, and sets `ehs0=0` so hail
and graupel snow collection remains outside this process oracle.  Maximum
five-category water closure is `3.7252903e-9`; the accepted 96-row fixture
SHA-256 is
`59dde5e30cb8c4abfef1af28ed26e9649a08da0a1d4c257ae9c6599fa9ae4e4a`.
The secondary-ice/category-conversion fixture spans 38 physical cases at four
time steps.  Its eight official-driver runs comprise a true-off baseline,
single-family `contact`, `homogeneous`, `hm`, `ice_to_g`, `snow_to_g`, and
`g_to_h` modes, and an `all` mode.  The checked-in canonical fixture is the
initial state plus the all-minus-baseline tendency; the generated
`secondary-ice-conversions-family-isolated.csv` instead selects the matching
single-family run for each case as a diagnostic.  The full modes use
`ibfc=1`, `icfn=2`, `itype2=2`, `iglcnvi=1`, `iglcnvs=2`, and `ihlcnh=3`.
The true-off baseline uses `ibfc=3`, `icfn=0`, `itype2=0`, `iglcnvi=0`,
`iglcnvs=0`, and post-namelist `ihlcnh=-1`.

Those non-obvious off values are required by WRF v4.6.1.  For `ipconc=5`,
`ibfc=0` enables an alternate post-assembly cloud-freezing path, and
`ihlcnh=0` selects the legacy wet-growth graupel-to-hail branch.  `ibfc=3`
matches neither homogeneous-freezing path.  Initialization resolves a
nonpositive `ihlcnh` to 3 before reading `namelist.input`; writing `-1` in
that namelist occurs afterward and matches none of the conversion branches.
Earlier v3 and v4 candidates used `ibfc=0` and `ihlcnh=0` and are therefore
rejected, not acceptance references.  Their canonical SHA-256 values were
`6993b021c29806a6e66e0aa0688dbc72b1824f1692a45c647a03cbef2cb08ed2`
and
`6664260d03d9ea779aea01f4d6729f3fe21a7300c2b9369eeff0cd2891703625`.

`build.sh` audits every generated secondary-ice namelist against the
source-derived `nssl_mp_params` token set before combining outputs.  The
accepted 152-row canonical has zero vapor change, maximum total-water closure
of `5.8207660913467407e-10`, and SHA-256
`7a13ada35baad6698f85ed3bbdf00c3f610512127aed96dd45967fac6e0edd3f`.
The snow-aggregation fixture spans the strict snow-mass and -15/10/0 C
temperature boundaries, the linear and exponential efficiency branches,
minimum and maximum snow-size diagnosis, air-density and mixing-ratio
variation, and the native ten-percent timestep number-depletion cap.  Native
namelist controls disable deposition/sublimation, primary ice nucleation, and
collisional fragmentation; every other hydrometeor category is empty, so
snow mass is unchanged and the returned snow-number delta is `csacs` alone.
The ice-deposition fixture spans empty and strict-threshold ice, ice maximum
dimensions on both sides of 100 microns, ice relative humidities from 100.01
to 150 percent, temperature/pressure/density variation, and timesteps from
0.1 to 1000 seconds.  Native namelist controls disable primary nucleation and
snow fragmentation; all neighboring hydrometeor categories are empty, so the
returned mass, number, vapor, and temperature changes isolate positive cloud-
ice deposition and its default depositional conversion to snow.
The frozen-vapor fixture spans ice-only, snow-only, and mixed states;
deposition and sublimation; near-saturation and limiter-controlled cases;
temperature, pressure, density, and timesteps from 0.1 to 1000 seconds.  The
CUDA comparator preserves the official Fortran operation boundaries when it
constructs the dynamic 0.002-K lookup temperature, because a contracted FMA
can select the adjacent saturation-table entry at a half-step boundary.
The graupel/hail vapor fixture spans both dense frozen categories, signed
vapor exchange, predicted-density initialization, category-specific
deposition density, and native mass/number/volume bounds.  The Bigg fixture
spans the strict -5 C and 8-mm gates, 0.25-bin incomplete-gamma interpolation,
minimum transferred mass and number, partial and near-complete freezing,
latent heating, frozen-drop volume at 900 kg/m3, and final rain/graupel
two-moment bounds.  Native namelist controls disable optional snow routing,
splinter production, and neighboring collection/evaporation processes.
