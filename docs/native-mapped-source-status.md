# Declarative mapped-source engine status

This milestone extends gpuwm's strict `rw-wps.mapping.v1` consumer across real
GRIB1, GRIB2, and NetCDF source families.  The current v2 GFS composition has
now completed a fresh, exact-current-HEAD d01-d04 native CUDA initialization,
stock-WRF NetCDF export, and unchanged WRF v4.6.1 startup gate.  The older
ERA5 and GFS v1 results remain historical evidence only; they do not confer
their hashes on the current authorities.
These results do **not** yet imply arbitrary-input or arbitrary-physics
support, nor do they live-gate every possible nested layout.

## Implemented

- independent, recursive mapping validation with fail-closed selectors,
  coordinates, units, axes, missing-data policies, typed derivations, target
  requirements, and derivation-cycle checks;
- NetCDF dimension/time/member decoding with exact coordinate-dimension and
  unit binding, plus an ordered selector-stack contract for soil layers;
- generic GRIB1 decoding through the vendored Rust bridge;
- generic GRIB2 inventory and selective decoding for regular
  latitude/longitude GDT 0 inputs, including bounded-layer selectors, member,
  generating-process, fixed-surface, bitmap, and time-semantics checks;
- generating-process consistency at each `(valid_time, member)`, which accepts
  legitimate GFS analysis-to-forecast process changes while rejecting mixed
  identities inside one source frame;
- a soil-only `preserve_mask` policy for products such as GFS whose bounded
  soil layers intentionally omit ocean points; it cannot be selected for
  atmospheric or generic surface fields;
- canonical derivations for pressure, geopotential height, wind speed,
  relative humidity, and specific humidity;
- SHA-256-bound input manifests, canonical source-frame headers, inspection
  receipts, and translation into gpuwm's regular-grid interpolation / WRF-real
  initialization ABI;
- `gpuwm-mapped-composition-v2` joins for one or more supplement products,
  exact-one terrain ownership, exact valid-time matching, coordinate-subset
  binding, invariant-terrain checks, and decoder/provenance hashes;
- checked, source-neutral declarative ERA5 0-7/7-28/28-100/100-289 cm and
  GFS 0-10/10-40/40-100/100-200 cm soil contracts, including exact
  per-depth selector binding, remap semantics, and source-land/ocean policy;
- CUDA horizontal interpolation, WRF-real initialization, lateral-boundary
  construction, Noah initialization, prepared cache, and atomic stock-WRF
  NetCDF export through `gpuwm.mapped_direct.prepare_mapped_wrf`;
- parent-first one-way Lambert hierarchies up to the mapping's declared
  `max_dom`, with independently mapped child inputs, parent-barrier
  finalization, `wrfinput_d01..dNN`, and external LBCs only in `wrfbdy_d01`;
- a public `rw-wps --source mapped` command with exact primary, supplement,
  provenance, decoder, manifest, geography, experiment, backend, and hierarchy
  worker arguments.  At commit `6126ccae056222721eb261cc15860bda4110aa35`,
  the exact Linux bridge rebuild and focused current-source gate pass; a new
  offline *installed-distribution* archive check is still required before the
  older installed-bundle claim is extended to these exact bytes.
- a fail-closed `rw-wps.descriptor.v1` compiler that imports only exact GRIB
  selector rows from a WPS Vtable while leaving canonical meaning, units,
  axes, missing policy, derivations, cadence, soil, and target semantics
  explicit in the descriptor; and
- create-only public input-manifest authoring with installed-decoder discovery,
  bridge/GRIB2-tabular-ABI probing, stable-handle hashing, runtime-verifier
  round trip, authority recheck, and atomic no-clobber publication.

## Current-HEAD real GFS GRIB2 d01-d04 gate (v2)

The exact source identity was commit
`6126ccae056222721eb261cc15860bda4110aa35`, tree
`ee8043768273d055efa37ba513f32f7ffb2cf564`.  A blobless identity bundle,
canonical Git tree-object pack, and exact source archive independently bound
the on-node runtime; its tracked worktree was clean.  The rebuilt Rust bridge
tests passed, followed by 118/118 focused mapped/composition/source-adapter/
distribution tests.

The public mapped-source route consumed genuine GFS f000/f003 subsets with
SHA-256 values
`09cef1109ed1937dc875be578701babd31fcf8c0991d6c71c7d13a02958f097a`
and
`3cfed33a12a431105ad1a8fc8bf1b32675c1f8c48459b9c43d57795d500f4843`.
The current mapping SHA-256 is
`5b0f41a7f4ddee1116ce8310dfd67827761413908d45402e1f55f32facc61d86`;
the current v2 composition SHA-256 is
`e0c2adae105263b177d7e8f8bb87d0e99731bc4cda9cb6a4217971a0b49b18e1`.
Create-only manifest authoring produced manifest SHA-256
`14869949079528bd6b91b5e65213cfc0eeb4296bc095b5cdd3367301174bee8d`
before the manifest was consumed by the CUDA command.

The live-gated topology is a 120 x 100 x 49 mass-point d01 at 12 km plus
three 60 x 60 x 49, 4 km sibling children, each directly parented by d01 at
3:1.  It is a one-way `feedback=0` gate with root-only external boundaries;
it is not evidence for a deep d01-d02-d03-d04 chain or for configurations
beyond the mapping's declared `max_dom=4`.

Native CUDA preparation and export exited zero in 23.675 s, used 935,684 KiB
maximum host RSS, and had 1,100 MiB device memory observed during the active
command.  Its exact stock-WRF inputs are:

- `wrfbdy_d01`: 3,999,601 bytes, SHA-256
  `f4763c32e3e0665df274646f639419404c5345b5eb74bb5d31336566cac08e5a`;
- `wrfinput_d01`: 18,146,577 bytes, SHA-256
  `94940dc95da5dd63080d74b04d9623432b12d63ec0ad752a4fbd6e652850e545`;
- `wrfinput_d02`: 5,134,897 bytes, SHA-256
  `a62affea39ea98d51d40daf61ab5cd0306c372b9eb61894b109f250b35045da9`;
- `wrfinput_d03`: 5,134,897 bytes, SHA-256
  `3a27753fd1a22a911297022c3313388aa18dc89c4f5707da3b061ceafbf3ccb1`;
- `wrfinput_d04`: 5,134,897 bytes, SHA-256
  `bc1aaa27ee50a434a1ceba84cd36d6628c0a61aed1c91c134ea3cb4807a783b1`.

Unchanged serial stock WRF v4.6.1 at commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, executable SHA-256
`cfac96554c8f9796c7522aaf023131ea7681ddf12110a327e51a548958874089`,
consumed those five files.  It advanced all four domains through the common
+10 s endpoint, emitted `SUCCESS COMPLETE WRF`, and exited zero in 15.82 s
with 1,078,360 KiB maximum RSS.  Inputs were byte-identical before and after.
All 165 numeric variables in each of the four history files were finite, and
no NaN, CFL, fatal, segmentation, or dependency error was present.  This is a
startup/export compatibility gate, not a long-forecast skill claim.

The compact evidence archive is 85,622 bytes with SHA-256
`fdb0d07cbfb884be2e61aeaeb1349db1b464257b4284ec52c13072d9e2a15726`.
It binds the commands, source/decoder/authority/input hashes, timings, finite
audits, pre/post input manifests, four history hashes, and final verdict.

## Historical real GFS GRIB2 composition and stock-WRF gate (retired v1)

The current mapping at `configs/rw-wps-gfs-pressure-grib2.mapping.json` and the
retired v1 composition bytes formerly at
`configs/rw-wps-gfs-terrain.composition.json` were exercised against immutable
GFS f000 and f003 subsets for 2026-07-20 00 UTC. The v2 composition now at that
path has a different SHA-256 and remains ungated:

- f000: 5,559,455 bytes, SHA-256
  `24bf5b5b302962ed66da502e67a91b62120714a76ee4689c648c91539fb688e9`;
- f003: 5,539,980 bytes, SHA-256
  `9ada4e2f8e086773417236781b2211d9a6007f4350dc7ffb8eed0104d50b0f40`;
- GRIB2 inventory decoder SHA-256
  `0786d50fa2c6ddc26dd5cc8639709957fb57b69d4fa5b3cc3d1626fc19cafa83`;
- GRIB2 selective dump decoder SHA-256
  `3331d8df532beda921d9b2bd5700f7ab97ad8207318a41ddbaa60a35746689b3`.

The gate materialized two complete 18-field frames on a 141 by 221 grid with
21 pressure levels.  It selected the four bounded GFS soil slabs, preserved
the 13,374 bitmap-missing ocean points in each slab for downstream
land/water-aware initialization, and proved terrain bit-identical at f000 and
f003.  A repeat gate took 0.654 s.  The canonical receipt content SHA-256 is
`313638a9870542cc1f04eee70833ea5f6c77c8c7ad962ad0fa9b27d82455b981`.

The exact c700 Linux decoder build used for native export has inventory SHA-256
`0f26191f01b2d52e952206b82b361af609ff416517b828437c63ed2b7cbe297a`
and dump SHA-256
`0ff21b560aa01769ef8f884440161b79d775ad725dd366475f325a9175285f8c`.
Binding those platform-specific decoder bytes changes the receipt identity but
not the canonical frames; the native-export receipt content SHA-256 is
`28988c3579e84ee04f5dcec5cbce0926fc39f8bd7d7e4ff5454fc3e83b4c3413`.

The checked `configs/gfs_wrf_direct_proof.toml`, WPS geometry namelist, and
stock-WRF acceptance namelist define the 2026 three-hour experiment.  It
produced:

- `wrfinput_d01`: 71,452,829 bytes, SHA-256
  `54f8188371b8c8f0160ab4e4912d33e3b69ec56d7b975b55543e585660ce62a4`;
- `wrfbdy_d01`: 7,674,706 bytes, SHA-256
  `13259d40a6c52dd91a9d7da0acb8698ff97b7e8235573697daa78e53bf6f8b63`;
- 69.10 s cold preprocessing on an RTX 5070: 46.03 s uncached static
  geography, 1.684 s decode/compose, 11.81 s initialization of both forcing
  times, and 7.32 s NetCDF export;
- 1,635 MiB peak device memory in one-second `nvidia-smi` sampling;
- proof content SHA-256
  `a35b51894ee264a258899c20fe944065322a31052d7dab116a840a2e0fb024c7`.

Unchanged stock WRF v4.6.1 accepted both files, advanced from
00:00:00 to 00:00:10, emitted `SUCCESS COMPLETE WRF`, and exited zero in
9.76 s.  All required readback fields were finite, the input/namelist/oracle
hash manifest was identical before and after, and no NaN, infinity, CFL,
fatal, or segmentation marker occurred.  The history output SHA-256 is
`e5b7240a80dcd2d82686e765677bc2565cb21750f8c882e34751f73dfc95eee6`;
the sealed stock-oracle evidence SHA-256 is
`523c0b937932853d6a72ededec5c70efb2ab70c34fe47d91876c689db5de89fe`.

### Genuine two-domain mapped hierarchy

The same genuine GFS pair was routed through a mapping-bound hierarchy: d01
is 250 x 200 x 49 at 12 km, and d02 is 240 x 160 x 49 at 3 km with a 4:1
parent grid/time ratio.  The mapping declares `max_dom=4`; requests beyond
that limit, a different vertical count, missing lateral boundaries, a changed
forcing cadence, or sub-hour hierarchy forcing fail before static or GPU
initialization.

Native CUDA preparation completed in 94.85 s outer wall time and 93.96 s
instrumented time.  The instrumented breakdown was 50.45 s uncached root
geography, 1.761 s decode/composition, 11.85 s initialization of both root
forcing times, 10.23 s child initialization, 6.02 s verified hierarchy
artifact writing, and 12.94 s WRF export.  One-second sampling observed a
1,891 MiB peak on the RTX 5070.  Source donor-halo checks passed for both
domains.

The atomic output contains:

- `wrfinput_d01`: 71,452,829 bytes, SHA-256
  `54f8188371b8c8f0160ab4e4912d33e3b69ec56d7b975b55543e585660ce62a4`;
- `wrfinput_d02`: 46,696,175 bytes, SHA-256
  `cb907441851e543f10bd2f6aa3d0686db156489d9f7c282bce4937961bb90a8b`;
- root-only `wrfbdy_d01`: 7,674,706 bytes, SHA-256
  `13259d40a6c52dd91a9d7da0acb8698ff97b7e8235573697daa78e53bf6f8b63`.

The d01 files are byte-identical to the separately gated one-domain result.
The hierarchy proof content SHA-256 is
`c225be68e886f89bf20aed2989bbd24ab79e110048f282f47c4284d95ce3d92f`;
the hierarchy artifact manifest SHA-256 is
`c039977c2937b15b2427ee3a81989fe57455880405598f8b349bec0c14eeea8e`.

Unchanged stock WRF v4.6.1 accepted both initial files and the root boundary
file, advanced d01 at 5 s and d02 at 1.25 s to the common +10 s endpoint,
emitted `SUCCESS COMPLETE WRF`, and exited zero in 30.65 s.  The process used
3,309,892 KiB maximum RSS; all required fields in both history files were
finite; and no NaN, infinity, CFL, fatal, or segmentation marker occurred.
The complete input/authority hash manifest was identical before and after.
The d01/d02 history SHA-256 values are respectively
`e5b7240a80dcd2d82686e765677bc2565cb21750f8c882e34751f73dfc95eee6`
and `6824d6e115a66b57ee54ec3b5fc0346f51f1baa10598a0f0745ec2e5eb66bac3`;
the hardened sealed hierarchy oracle SHA-256 is
`90258b215677f256eea1df07320234fb73143136de7ebb326da947fe93cd7386`.
That seal independently binds each history file's domain ID, parent placement,
dimensions, grid spacing, time step, and valid time, and requires the exact
root-boundary plus per-domain-input file inventory.

### Public `rw-wps` hierarchy replay

Commit `95c2a086d7463079a3c2b22df83fa634686cc023` exposes the mapped
composition through the installed public `rw-wps --source mapped` route.  The
exact genuine GFS d01+d02 case above was replayed through that public command
on an RTX 5070, rather than by importing the internal Python helper.  It exited
zero in 98.312 s outer wall time and 96.827 s instrumented time; one-second
sampling observed a 1,883 MiB device-memory peak.  Its proof content SHA-256 is
`ac932ad69a52b6b1b02898628a2b91af2bcc32e378635d26357a0e7a18d8b407`
and the serialized proof SHA-256 is
`7d9a66d68be79b204c4f93e74e96094d75ab31f9e719d089406cfb08d4b40b15`.

The public replay's `wrfinput_d01`, `wrfinput_d02`, and `wrfbdy_d01` are
byte-identical to the three inputs bound by the hardened stock-WRF oracle
above.  This connects the public-route replay to the unchanged WRF v4.6.1 PASS
by exact artifact hashes, while preserving the command's conservative emitted
status `READY_NOT_YET_STOCK_WRF_GATED`: the command itself does not execute or
trust an external WRF oracle.

### Isolated installed-distribution gate

Commit `bbd0de104ff48689a962acffabeb732ceaecdc3d` additionally binds the
generic GRIB2 inventory and dump tabular ABIs during both distribution build
and every installed-runtime check.  This was added after a deliberately live
installed-bundle test exposed a stale inventory executable whose usage text
matched but whose table omitted the required `member` column.  The stale pair
now fails before packaging or execution.

All five Rust bridges, the deterministic CPU preprocessing library, and the
wheel were rebuilt from the exact source commit on Linux.  The resulting
standalone archive is 37,049,281 bytes with SHA-256
`1c482e86bc4ac317a1347184510c71a794c0928d7648cb84b57288cbedc3ea0a`.
Its offline isolated installation and CUDA/runtime/RECORD/bridge self-check
completed in 3.752 s; the installed wheel check bound 215 RECORD files.

The installed `bin/rw-wps`, using only its bundled wheel and decoder paths,
then replayed the genuine GFS d01+d02 case.  It exited zero with empty stderr
in 90.503 s outer wall time and 87.744 s instrumented time; one-second sampling
observed a 1,891 MiB device-memory peak.  Proof content SHA-256 is
`253bd85c14687829d8126f93145c48a037dcfe1a76f501c3fde00949e48f18c5`.
The three WRF artifacts are again byte-identical to the hardened stock-WRF
oracle inputs above.  This is an installed-bundle source-independence gate on
the retained Linux host, not yet a claim of extraction and execution on a
second clean machine.

The separate Rust RW-WPS frontend is sealed at commit
`ad501d016fe1ea6aeffcf641e04fae800dd1b5d0`.  It now expresses the same
versioned composition, typed-decoder, supplement/provenance, geography, and
hierarchy contract; 49 focused tests, 1,212 workspace tests, strict Clippy,
and a cross-language Python dry-run passed after independent fail-closed
review.

## Historical real ERA5 NetCDF gate (retired v1)

The three retained ERA5 NetCDF files initially failed closed: a final
`ncks -A` of a 1979 invariant land mask overwrote their common `time` and
`utc_date`, so the decoder correctly saw three duplicate 1979 frames.  The
checked `tools/repair_era5_appended_invariant_time.py` utility repairs a copy
only when the pressure-level extraction, monthly surface extraction, expected
hour indices, invariant append, and corrupt 1979 signature all agree.  Its
receipts prove all 25 non-time variable arrays are unchanged.

Corrected source SHA-256 values are:

- 1974-04-03 12 UTC:
  `fc61bb6988f9c355e5046cdb32113dbc54f9bd183b2a7e13baf3fb964980c95a`;
- 1974-04-03 18 UTC:
  `e5b34e89151fb33c397738a8e801d899eb9de6dac801115fec035d3a88f6726b`;
- 1974-04-04 00 UTC:
  `ef2fcbf506dfe6d024a47716219ac349c3b6a6d0b70e37b99e79af172e242943`.

Those files have no terrain.  The checked
`tools/build_era5_netcdf_terrain_supplement.py` utility decoded the genuine
ERA5 invariant GRIB1 geopotential product and produced a lossless NetCDF view,
SHA-256
`34222e29d3064895941a6b39c7ebac5930eb93b719ce5e8d24ba810483d435a2`.
Its receipt proves invariance across all three forcing times and a bit-exact
geopotential-to-height round trip.

The retained gate's v1 NetCDF mapping/composition pair materialized three
complete 19-field frames in 0.596 s.  Its mapping is still current, but its
composition SHA-256 differs from the v2 bytes now checked in. Grid, times,
levels, and terrain match
the independently decoded real GRIB1 slice.  The largest observed NetCDF vs
GRIB1 encoding residuals were 0.000214 K in air temperature, 0.046875 in
geopotential, 0.00478 m in terrain height, 0.046875 Pa in surface pressure,
and 2.08e-7 kg/kg in specific humidity.  The composition receipt content
SHA-256 is
`39802e50006423a9eac1e29c807d69356d2a854a72719dada937fb949c4fa2fc`.

## Native CUDA export and unchanged-stock-WRF gates

The earlier genuine ERA5 GRIB1 path produced `wrfinput_d01` and `wrfbdy_d01`
in 25.54 s and passed unchanged stock WRF v4.6.1 through 10 model seconds.

The new ERA5 NetCDF path produced:

- `wrfinput_d01`: 80,239,252 bytes, SHA-256
  `2969682af941f639a14e7ae43958e4d58f28b4003aafea951a1f90da35950554`;
- `wrfbdy_d01`: 14,502,299 bytes, SHA-256
  `2b530205289ccdb39ade2a29dee74f9d89020950f0c88f27c41efa561981b8d6`;
- 20.98 s total preprocessing wall time: 11.06 s native static geography,
  0.669 s decode/compose, 5.66 s CUDA initialization, and 3.03 s export;
- proof content SHA-256
  `53444f9ccf4950462468db05a8cb5d31e3928563232d870fa5e3a0bde7576db8`.

Unchanged stock WRF v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, executable SHA-256
`f0fb585bf37b72fbdcece562047934cb8386db3958f153d6e4e6876e5fd997ac`,
accepted both NetCDF-path products, advanced from 12:00:00 to 12:00:10,
emitted `SUCCESS COMPLETE WRF`, and exited zero in 10.42 s.  Input hashes were
identical before and after; no NaN, infinity, CFL, fatal, or segmentation
pattern was present.  The output SHA-256 is
`da6159a373a47829e7c207ea7186e16509bcee2ebf6419963c3c77f6a01dcf2c`.
The clean oracle gates are serial because the available stock executable is a
serial build.

## Remaining release gates

1. Merge and publish the sealed Python and Rust frontend commits in their
   canonical repositories, then extract and rerun the archive on a second
   clean Linux machine.
2. Re-run the unchanged-stock-WRF certification gate for the current ERA5 v2
   composition; the current GFS v2 d01-d04 sibling gate above is complete, but
   old ERA5 hashes are not inherited.
3. Live-gate deep-chain and additional sibling layouts, and explicitly reseal
   mappings before raising their current `max_dom=4` target ceiling.  Vertical
   nesting remains unsupported; all domains share one explicit eta grid.
4. Generalize the current terrain-only supplement join; declarative soil and
   descriptor/manifest authoring are implemented, but they do not make an
   incomplete product scientifically complete.

See `docs/native-mapped-source-authoring.md` for the exact authoring contract,
CLI, and certification boundary.
