# NextGen development measurements

This snapshot records development results inspected on 2026-10-18. Routing
uses integration `2c9b0295f99ac132a67759fa1515d838fef882f5`; the initialized
catchment result uses forcing owner `5e0598057fb46b5a15b3e1bc62aa8ed1dd6addba`
over integration `ee91d6c5fdd74e2fae423e76e97c39bc96e39b96`. These are separate
captured-input compositions. The public demo remains held before acquisition;
these results do not provide a downloadable, complete physical forecast.

## Strict comparisons and short runs

| Composition | Actual run | Native comparison | Measured strict step cost |
|---|---|---|---:|
| Initialized Noah/PET/CFE | Three catchments, 48 hourly intervals; initialized once, with no intermediate native resets | 0 of 864 differing stored words across six histories | 0.245834 ms |
| Muskingum-Cunge, level pool and RFC | 6,723 channel reaches, seven level pools and twelve RFC reservoirs; 12 intervals of 300 seconds with captured native lateral inflows | Seven NetCDF variables: 0 of 1,348,496 typed bytes; ordered names/types/shapes/typed attributes exact | 2.898381 ms |

Both explicitly select strict numerics on a shared-pool RTX 4090, include the
first enqueue without warmup, use two batch boundary waits and have no CUDA
graphs or per-step host numerical processing. The native routing recipe
repairs an uninitialized secant seed; comparison is against that pinned
referee recipe. The default-fast routing output differs from native; strict
agreement must not be transferred to fast. Physical HDF5 file bytes differ.

The catchment comparison includes surface runoff and PET in m/s, CFE runoff
and rain in metres per interval, and soil/groundwater storage in metres.
These quantities must retain their units rather than be relabelled as reach
discharge. Its authoritative native framework index has SHA-256
`5c5b23e6050e304fbfd98cb0096924e31ac030e99230ded3b83105cfa5547461`.
The [forcing contract](nextgen-forcing.md#evidence-limits)
records the corrected PET source, exact 720-hour module replay and changed
restart identity. Earlier continuation measurements retain their original
source pins.

Four current routing hydrographs were inspected. Model and native curves
overlap at all twelve samples: gauge 08117995 rises to about 0.000077 m³/s
against four zero observations; 08119500 rises from about 0.006 to 0.193 m³/s
with no observations; 08120500 stays near zero against four observations near
0.005918 m³/s. RFC lake 5632093 remains flat zero; its 0–1 axis shows no useful
variation, so exact typed comparison supplies the equality evidence. This
one-hour cold-start comparison establishes no flood skill.

## Native reference dependencies

The native case(a) reference uses SLoTH at ngen's gitlink
`48cd8082878fdb8a6f0e2337a5b375be5b71384c` to supply configured constants.
Its complete LICENSE states: “Software code created by U.S. Government
employees is not subject to copyright in the United States”, cites
17 U.S.C. §105, and retains the royalty-free, nonexclusive grant for use,
copying and derivative works outside the United States. Its BMI-CXX dependency
at gitlink `7c9a1d07bfb566027df337bbf83051da7fac4004` carries the MIT notice.
The referee retains both complete notices. These native tools are not installed
on a user path or supplied as a product runtime dependency.

The sealed framework index above records 576 actual public BMI updates across
SLoTH, Noah, PET and CFE, with six observed CSV files identical to the clean
native run. This is existing referee evidence, not a new run for this document.
Configured SLoTH constants do not establish physical soil-state producers for
a different network; that composition still needs its own parameters, states,
units and exchange boundaries. Software permissions do not clear the separate
case(b) GeoPackage or forcing-data rights.

## Reservoir carrier precision

Integration `ca16c7b3bfe95573ec30e884442783d8fb223537`, followed by
README-only `2c9b0295f99ac132a67759fa1515d838fef882f5`, imports reservoir
owner `7f2bf2a30c827e5249466234f213e2bc290fab52`. Assimilation arithmetic
uses double precision, but native t-route assigns persisted release, update
time and persistence time to float32 carriers. The CUDA kernel now reproduces
those assignments before storing their exact promoted values in float64
restart arrays. Both numerical modes preserve this boundary; it adds no
launch, allocation, transfer or synchronization to stepping.

Real independent native-call replay checks returned values and the production
kernel's committed carriers separately:

| Captured calls | Returned words | Committed words |
|---|---:|---:|
| RFC, 144 calls | 0 of 1,008 different | 0 of 1,008 different |
| Persistence, 48 calls | 0 of 288 different | 0 of 288 different |

Each call uses one CUDA thread in a 128-thread block. These are independently
fed native level-pool outputs, not a continuous network trajectory. The
persistence inputs contain missing observations; valid-observation USGS/USACE
persistence and the reservoir-to-diffusive edge remain unverified. The
integrated short trajectory below exercises seven level pools and twelve RFC
reservoirs; it is separate from the held no-assimilation demo row.

## Stage profiles

At integration `2c9b0295f99a`, the actual root CLI measured default fast
**1.530472 ms/step** and explicit strict **2.898381 ms/step**, twelve
300-second intervals, captured lateral inflows, 6,723 MC reaches, seven level
pools and twelve RFC reservoirs. Both used a shared-pool RTX 4090, no warmup,
338 ordinary launches and 320 prebound callbacks per step, two batch boundary
waits and no production timing events or CUDA graphs. Host work per step is
the launch sequence. These are not isolated-card measurements or full hydro.

This separate fresh first-interval profile includes enqueue gaps. Its event
brackets do not add up to batch throughput:

| Routing stage | Fast ms | Strict ms | GPU parallelism |
|---|---:|---:|---|
| Snapshot | 0.027584 | 0.025600 | 159 blocks of 128 threads |
| Muskingum-Cunge | 1.292192 | 1.693728 | 266 nonempty dependency groups, 128 threads per block |
| Reservoir gather | 0.072800 | 0.072960 | 17 ordered subsets, 128 threads per block |
| Level pool and assimilation | 0.132704 | 0.152768 | 17 ordered subsets, two kernels of 128 threads |
| Reservoir scatter | 0.060288 | 0.059744 | 17 ordered subsets, 128 threads per block |
| Reach publication | 0.004096 | 0.004096 | 53 blocks of 128 threads |
| Typed history | 0.005120 | 0.005120 | 27 blocks of 256 threads plus cursor |

Output is outside step timing: **81.122805 ms fast / 79.607237 ms strict**,
including D2H, the 148-array CDF5 checkpoint and Rust CSV/NetCDF publication.
Rust uses 24 Rayon formatting/transpose workers and serialized netCDF-C/HDF5-C
container calls. This run writes checkpoints; it does not repeat the
[separately pinned routing or standalone continuation
checks](nextgen-output.md#measured-routing-continuation).

The speed owner's earlier three-catchment measurements remain a separate
snapshot at revision `082aaaa5b613ce29ea91394b4bf331226c5a24e0`:
default fast **0.171880 ms/step**, explicit strict **0.240524 ms/step**,
48 intervals on a shared-pool RTX 4090, eight launches per interval. Its fresh
single-interval profiles are:

| Stage | Fast ms | Strict ms | GPU parallelism |
|---|---:|---:|---|
| Forcing float64 | 0.055456 | 0.040960 | One block, 128 threads |
| Forcing float32 | 0.014176 | 0.011264 | One block, 128 threads |
| PET | 0.019456 | 0.019648 | One block, 128 threads |
| Noah | 0.092160 | 0.167744 | One block, 64 threads |
| CFE | 0.014336 | 0.013568 | One block, 128 threads |
| History | 0.006144 | 0.006912 | One block, 128 threads plus cursor |

Only three catchments are active. These profiles also include enqueue gaps
and are not additive throughput measurements. Strict native agreement does
not establish default-fast agreement. Fast remains the product default.

## Remaining physical scope

The complete Lower Colorado runoff, routing and reservoir composition and
same-area 1 km terrain calculation still need their measured speed acceptance.
The separately pinned standalone continuation result does not establish
combined parent-atmosphere, runoff, network and terrain continuation or a
gauged flood-event simulation.

For the Russian River January 2023 event, original federal source flowlines
cover 1,445 of 1,445 selected reaches; catchment polygons cover 1,433 IDs.
Complete physical parameters, initialized state, forcing, reservoir controls,
parent overlap weights and the selected native event realization remain
missing. The existing 867 NWM-v3/USGS hourly pairs are a separate archive
comparison. Dataset rights and prepared-resource requirements remain as
described in the [public guide](public/HYDRO.md) and
[configuration contract](nextgen-configurations.md).
