# WRF v4.6.1 Thompson table oracle

This harness compiles the unmodified NCAR WRF v4.6.1
`module_mp_thompson.F` and `module_mp_radar.F` with the smallest possible
WRF-environment stubs, calls classic `thompson_init` without aerosol-aware or
hail-aware optional state, and writes all lookup tables consumed by
`mp_physics=8`.

It is an oracle and asset-generation tool, not a replacement implementation.
The two official WRF source files must come from tag v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`.  The build script never edits that
checkout and refuses to reuse existing table outputs.

On a Linux host with gfortran:

```sh
./build.sh /path/to/WRF-v4.6.1 /empty/build-and-output-directory
```

Expected classic-cache sizes with gfortran's default four-byte record markers:

- `qr_acr_qg_V4.dat`: 74,966,480 bytes
- `qr_acr_qsV2.dat`: 43,764,288 bytes
- `freezeH2O.dat`: 254,944,848 bytes
- `thompson_aux_tables.dat`: 6,164,536 bytes

Validate the values with `gpuwm.core.thompson_contract`; byte hashes are
compiler/platform receipts and are recorded separately from the source commit.
The deliberately unused/uninitialized `tnr_rev` allocation is not dumped.

The same build also runs deterministic 24-level, 10-second process columns
through WRF's public `mp_gt_driver`.  In addition to the broad `warm`, `mixed`,
and `ice` cases, the isolated fixtures cover saturation adjustment, five
fallout species, warm autoconversion, rain self-collection/accretion,
ice-to-snow autoconversion, ordinary subsaturated rain evaporation, and cold
snow/graupel vapor deposition and sublimation, cloud-ice deposition, and
classic deposition nucleation.
The `warm-frozen-subsat` column additionally composes simultaneous snow and
graupel in an ambient-warm, subsaturated layer, pinning WRF's same-call
graupel-melt reduction of subsequent rain evaporation.
Each `column-oracle/*-column.csv` preserves both the input and output state,
including number moments, temperature, effective radii, and reflectivity;
the matching `*-surface.csv` pins category-resolved precipitation.  These are
direct numerical comparison vectors for the CUDA port, not hand-transcribed
expected values.
