# Thompson coefficient provenance

The numerical authority for this directory is NCAR WRF v4.6.1, git commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, file
`phys/module_mp_thompson.F`.

Classic Thompson is WRF `mp_physics=8` (`Registry/Registry.EM_COMMON:3024`).
It is not the aerosol-aware `mp_physics=28` or graupel/hail-predicting
`mp_physics=38` package.  The classic initialization omits optional `qng`,
selects the one-density graupel table (`idx_bg1=5`, 400 kg m-3), and generates
the following sequential-unformatted caches when they are absent:

- `qr_acr_qg_V4.dat`
- `qr_acr_qsV2.dat`
- `freezeH2O.dat`

Their exact record names, shapes, Fortran ordering, and byte counts are pinned
in `gpuwm/core/thompson_contract.py`.  Assets are admitted only after the
official source is compiled into the oracle harness, every record passes that
strict reader, and SHA-256 receipts are recorded.  The canonical table-set ID
is `wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1`; the four expected byte
digests are part of the runtime contract.

The `tables/` subdirectory holds the canonical generated coefficient set:
the three WRF cache files above plus `thompson_aux_tables.dat` (the smaller
thompson_init tables the pinned oracle harness writes to a fourth provenance
file), with `MANIFEST.sha256` (`sha256sum -c` format) committed beside them.
The two smallest ship as first-class package data; the two largest are
*externalized*: published as versioned GitHub release assets and staged
into this directory by `gpuwm fetch-tables` (gpuwm/table_assets.py),
verified against the same size and SHA-256 pins before an atomic
install.  `freezeH2O.dat` (243 MiB) exceeds GitHub's 100 MiB blob limit
and is absent from the public repository entirely; `qr_acr_qg_V4.dat`
(71 MiB) stays in the repository but is excluded from the wheel and
sdist, which PyPI's default per-file cap would otherwise reject.  The
development tree keeps both files tracked as the source of truth for
the release assets.  The bytes were produced by the
official-source oracle (`tools/thompson_wrf461_oracle/generate_tables.F90`) on
Ubuntu 24.04.4 with GNU Fortran 13.3.0 and are byte-identical to the
`CLASSIC_TABLE_ASSETS` pins in `gpuwm/core/thompson_contract.py`; every load
still re-validates size and SHA-256 before GPU setup.  This packaged root is
the default table root for `mp_physics=8`; the `GPUWM_THOMPSON_TABLE_ROOT`
environment variable remains as an override naming a directory with the same
validated bytes.

The `oracle/` subdirectory contains small direct-call comparison sets, not
coefficient tables.  The unmodified WRF public `mp_gt_driver` produced
the `warm`, `mixed`, and `ice` 24-level columns using a 10-second step.  Each
pair preserves the complete before/after column plus category precipitation,
effective radii, and reflectivity.  Their hashes are pinned by
`tests/test_thompson_contract.py`; the reproducible producer is
`tools/thompson_wrf461_oracle/run_column.F90`.  The additional `condense`
column holds temperature and vapor fixed vertically and starts with no
hydrometeors, isolating WRF's warm cloud-vapor saturation adjustment from
sedimentation and the remaining mixed-phase process network.
The `rain-sed` column is liquid-saturated, contains no cloud or frozen
hydrometeors, and selects a 45-micron rain median-volume diameter below the
self-collection threshold.  It therefore isolates WRF's mass/number rain
sedimentation and surface-water budget.  These official-WRF fixtures are
validation oracles only; gpuwm's executable implementation remains native
CUDA and does not call WRF at runtime.
The `ice-sed` column is ice-saturated and selects a 29-micron
mass-weighted ice diameter below the 30-micron ice-to-snow autoconversion
threshold while respecting WRF's ice-number ceiling.  It isolates
differential cloud-ice mass/number sedimentation, frozen surface
precipitation, and the column-water budget.
The `cloud-sed` column is liquid-saturated, contains only a deliberately
sub-autoconversion cloud-water layer, and has zero vertical velocity.  It
isolates WRF's fixed-number cloud-droplet mass sedimentation and conservative
vertical redistribution without surface precipitation.
The `cloud-condense-sed` column retains an entry-active, sub-autoconversion
cloud layer in slight liquid supersaturation and uses a deliberately shallow
sedimentation layer.  It is an adversarial composition gate for WRF's held
pre-adjustment volumetric cloud mass, post-adjustment mixing-ratio conversion,
and no-rain fall-speed density across saturation adjustment plus cloud fallout.
Its matched `cloud-condense-nofall` member differs only by upward velocity,
exposing the official WRF post-adjustment state so the small fallout increment
can be gated independently of cross-compiler saturation-adjustment rounding.
The matched `cloud-rain-condense-sed` / `cloud-rain-condense-nofall` pair adds
entry-active, sub-collection rain.  WRF's preceding rain-velocity pass then
refreshes the cloud fall-speed density after adjustment; the pair pins that
column-held `ANY(L_qr)` branch independently from rain transport.
The `condense-fall-attempt` column matches the original cloud-free `condense`
state but supplies zero vertical velocity.  Its post-call scientific state is
identical to `condense`: classic WRF holds `ANY(L_qc)=false` across saturation
adjustment, so freshly nucleated cloud cannot fall in that same call.
The `snow-sed` column is ice-saturated, contains only snow below 4.25 km, and
is cold enough to suppress melting.  It isolates WRF's Field et al. snow
moment conversion, one-moment mass sedimentation, timestep splitting, frozen
surface precipitation, and column-water budget.
The `graupel-sed` column has the same cold, ice-saturated isolation but
contains only classic fixed-density graupel.  It isolates WRF's diagnosed
exponential intercept, one-moment mass sedimentation, timestep splitting,
graupel surface precipitation, and column-water budget.
The `warm-auto` column is liquid-saturated, contains cloud water but no
incoming rain, and uses upward velocity to suppress cloud-droplet fallout.
It isolates Berry-Reinhardt cloud-to-rain autoconversion followed by the
already admitted two-moment rain sedimentation path.
The `rain-self` column is liquid-saturated, contains rain with a 500-micron
median-volume diameter and no other hydrometeors.  It isolates Seifert rain
self-collection's number sink followed by the admitted two-moment rain
sedimentation path.
The `warm-accrete` column adds a sub-autoconversion cloud layer to the same
500-micron rain distribution and uses upward velocity to suppress cloud
fallout.  It isolates rain-cloud accretion and simultaneous Seifert
self-collection followed by admitted rain sedimentation.  The accretion
kernel consumes the validated external FP64, Fortran-ordered
`t_Efrw(100,100)` array; the focused fixture's sparse test table contains only
the exact canonical entries exercised by this column and is not a production
table substitute.
The `warm-overlap` column raises that cloud layer above the autoconversion
threshold while retaining the 500-micron incoming rain distribution.  It is
the first unified-driver oracle: Berry-Reinhardt autoconversion, rain-cloud
accretion, and Seifert self-collection are all diagnosed from the same
incoming state, the shared cloud-water source cap is applied once, and the
categories are updated once before rain fallout.  The two legacy sequential
launch orders measurably disagree on the adversarial rate-cap gate, while the
fused CUDA network matches this official WRF column.  Its producer and hashes
are committed alongside the other direct-call oracles.
The `rain-snow-graupel-overlap` column is the first fused cold collision
network.  It activates both table-driven collision families plus Seifert rain
self-collection from one incoming rain state, applies WRF's shared rain-mass
bound once, and then exercises all three entry-time fallout paths.  The CUDA
gate preserves WRF's unusual limiter semantics: grouped mass scaling does not
scale either collision-number sink, and only the rain/graupel pair is
re-enforced after the bound.  The direct WRF producer regenerated both CSVs
twice with identical SHA-256 receipts.
The `rain-ice-graupel-overlap` column extends the same simultaneous source
boundary to table-driven Bigg rain freezing and rain collection of cloud ice.
It uses a 20-degree-supercooled, ice-saturated column so freezing, ice/rain
collection, graupel/rain collection, and rain self-collection all contribute
before the single cloud-ice and rain conservation groups.  Incoming ice and
graupel keep both frozen fallout paths eligible.  The official producer again
regenerated the fixtures twice with byte-identical SHA-256 receipts.
The `frozen-vapor-overlap` column activates simultaneous cloud-ice, snow, and
graupel sublimation from one 99.99%-ice-RH thermodynamic state.  Each species
is kept below the collection-table floor to exclude unrelated collisions, and
the shallow deficit forces WRF's shared frozen-vapor limiter.  All three
entry-time fallout paths then run.  Its direct-WRF fixtures were regenerated
twice with identical SHA-256 receipts.
The `cold-cloud-overlap` column activates the complete simultaneous cold
cloud-water source group: warm-rain autoconversion/accretion, table-driven
cloud freezing, snow/graupel riming, and Hallett-Mossop splinter production.
It pins WRF's single shared cloud-water limiter, including its deliberately
held number and splinter tendencies, before ice/snow/graupel/rain fallout.
Its direct-WRF fixtures were regenerated twice with byte-identical SHA-256
receipts.
The `frozen-vapor-nucleation-overlap` column combines non-aerosol Cooper ice
nucleation with simultaneous cloud-ice and snow deposition, table ice-to-snow
autoconversion from a 200-micron distribution, and snow collection of cloud
ice.  Its 50-second source step forces three layers through WRF's single
ice-saturation limiter while preserving WRF's held nucleation-number source.
A diagnostic 100,000-km layer thickness suppresses fallout to isolate the
interacting source rates and driver-order limiters.
The direct-WRF fixtures were regenerated twice with byte-identical SHA-256
receipts.
The `cold-ice-rain-overlap` column extends that 50-second, source-isolated
cold-ice group with a 1000-micron incoming rain distribution.  Its
200-micron cloud ice simultaneously undergoes deposition, lookup-table
ice-to-snow autoconversion, and rain collection while rain self-collection
and WRF's subsequent cloud-ice and rain mass caps remain active.  It pins
WRF's held collision-number tendencies and the deliberately pre-cap paired
graupel source; newly created snow and graupel are entry-inactive for fallout.
The incoming vapor is liquid-saturated, but deposition leaves some active
layers liquid-undersaturated, so the direct-call trajectory also retains
WRF's ordinary post-source rain evaporation and the entry-time ice/rain
fallout calls.  The 100,000-km diagnostic layers make that transport
dynamically negligible while preserving driver order.
The direct-WRF fixtures were regenerated twice with byte-identical SHA-256
receipts: `6e964a6cf7fef3801a80e4d241c4a2b402081b2681e44dcf75a4c0594a006c44`
for the column and
`6ea219231ce8d90cb4edb9c6fc31acb9d55368265ac3edcee0553b8591b6977d`
for the surface record.
The `cold-full-overlap` column is the first complete cold-source orchestration
gate.  At 240 K it combines supersaturated frozen-vapor exchange, Cooper
nucleation, ice-to-snow autoconversion, snow and rain collection of cloud
ice, Bigg rain freezing, both table-driven rain/snow and rain/graupel
collision families, and rain self-collection from one immutable incoming
state.  It then applies WRF's vapor, cloud-ice, rain, snow, and graupel bounds
in production order before one category and thermodynamic update.  The gate
also pins a previously cross-group-only ordering rule: WRF diagnoses rain
freezing first and subtracts those same-call `pni_rfz` crystals from the
Cooper nucleation target.  Ordinary post-source rain evaporation and every
entry-active frozen/rain fallout path remain in the direct-call trajectory;
100,000-km diagnostic layers keep transport dynamically negligible without
removing those calls.  The producer regenerated both files twice
byte-identically: `848d4c7f5efc13da67b7b1447a97eecd3acdbf1dc4cfebd05a56a748bd82af4b`
for the column and
`4d72461e4fc2f137889f9abf87a0d261b7a9c59a547383d00d3f723ac39dd71d`
for the surface record.  Because this adversarial vector forces the rain/ice
cap, it also inherits classic WRF's already documented pre-cap `prg_rci`
water-budget asymmetry.  The contract pins the resulting water creation
explicitly; it does not describe this exact-WRF trajectory as conservative.
The `cold-cloud-rain-overlap` column closes the remaining classic cold-source
interaction boundary by adding incoming cloud water to the same 240 K rain,
ice, and supersaturated-vapor state.  It simultaneously activates
Berry-Reinhardt autoconversion, rain/cloud accretion, Bigg cloud and rain
freezing, rain/ice collection, and Cooper nucleation.  In particular, it pins
WRF's driver ordering in which the Cooper target subtracts both same-call
`pni_wfz` and `pni_rfz` crystal sources before the shared species bounds.
Cloud saturation adjustment, rain evaporation, and all entry-active fallout
calls remain in the trajectory; 100,000-km diagnostic layers keep transport
small while still exposing held-density number sedimentation.  The direct WRF
v4.6.1 producer regenerated the fixtures byte-identically with SHA-256
`76da2ac7be669cd78b0b69f3940abb9e5ec817919c55f4d60da9f9893b4d9d96`
for the column and
`9097a49eeee045cff8f9aeec62ad2ac28147e9d0a4daad96fd72531d0fca0ec4`
for the surface record.
The `ice-auto` column is ice-saturated and contains one 200-micron
mass-weighted cloud-ice layer.  It isolates lookup-table ice-to-snow
autoconversion followed by the admitted ice and snow fallout paths.  The
kernel consumes the canonical external FP64, Fortran-ordered
`tps_iaus(64,55)` and `tni_iaus(64,55)` arrays.  The focused fixture's sparse
tables contain only its one active canonical bin and are not production table
substitutes.
The `rain-evap` column is warm, contains only a bounded 45-micron rain
distribution, and starts at 99.5% liquid-water relative humidity.  It avoids
the near-zero rapid-elimination and Seifert self-collection branches, thereby
isolating ordinary Srivastava-Coen rain evaporation followed by the admitted
two-moment rain fallout path.  Its composition fixture also pins WRF's
ordering detail: evaporation updates temperature and vapor before fallout,
but the rain mass/number carried into fallout retain the pre-evaporation
volumetric density.
The `snow-subl` column is cold, contains only snow below 4.25 km, and starts at
99.5% ice relative humidity.  It isolates ordinary Srivastava-Coen snow
sublimation, including the Field et al. first and ventilation moments, the
vapor source and sublimative cooling, followed by the admitted one-moment snow
fallout path.  Unlike the rain-evaporation composition, WRF recomputes the
snow volumetric state after the thermodynamic update before sedimentation.
The `graupel-subl` column applies the same cold 99.5%-ice-RH isolation to
classic fixed-density graupel.  It isolates ordinary Srivastava-Coen graupel
sublimation, including WRF's diagnosed exponential intercept, mass-proportional
internal number sink, sphere capacitance, ventilation term, vapor source, and
sublimative cooling, followed by admitted graupel fallout.  Classic
`mp_physics=8` does not expose graupel number as prognostic state: WRF
re-diagnoses it from updated graupel mass before sedimentation.

At runtime, `gpuwm/core/thompson_runtime.py` revalidates all four canonical
asset hashes, parses all 30 records, enforces exact FP64 Fortran layout and
immutability, uploads the complete 379,839,912-byte payload, and verifies a
per-record SHA-256 device round trip before caching one owner per CUDA device.
This closes the coefficient upload boundary but does not open the production
`mp_physics=8` selection gate.

WRF is distributed under its own license.  Any generated coefficient artifact
retains this provenance and the repository's canonical WRF license copy at
`gpuwm/data/wrf_radiation/LICENSE-WRF.txt`.
