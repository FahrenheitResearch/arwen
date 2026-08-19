# General mapped-source authoring

RW-WPS can now compile an explicit source descriptor and author the exact
input manifest used by the native GRIB1, GRIB2, and NetCDF path. This removes
the need to edit a hash manifest by hand and avoids adding a Python adapter for
every named forecast model.

This is a general *configuration* path, not automatic scientific inference.
The emitted status is `VALIDATED_NOT_STOCK_WRF_CERTIFIED`. A new descriptor
does not inherit the retained ERA5/GFS stock-WRF gates merely because it uses
the same decoder.

## Authority boundary

`rw-wps.descriptor.v1` has the same coordinate, field, derivation, and target
semantics as executable `rw-wps.mapping.v1`. For GRIB input, each direct
field replaces `selectors` with `vtable_selectors`. Each reference must resolve
exactly one row in the supplied 11-column WPS Vtable. RW-WPS imports only the
numeric GRIB selector from that row. The descriptor remains the authority for:

- canonical field meaning and source/target units;
- axes, rank, staggering, and grid location;
- missing-data and mask policy;
- vertical inventory, ordering, derivations, and cadence;
- soil meaning and target WRF/domain requirements.

A Vtable's mnemonic, unit label, and description are retained in the
authoring receipt but are not treated as those scientific decisions. This is
important because real Vtables reuse names such as `TT` for both 3-D model
levels and 2-m temperature.

## Declarative soil geometry and remapping

The composition contract owns soil semantics independently of a source name.
`soil_layers` now declares:

- canonical temperature and volumetric-moisture fields;
- metre-valued, contiguous, shallow-to-deep source and target layer bounds,
  with the exact temperature and moisture selector bound to every source
  depth;
- either layer-bottom point interpolation to Noah midpoints with explicit
  surface/deep anchors, or conservative layer-mean overlap remapping with
  complete source coverage; and
- a fail-closed missing-data rule: source-land gaps are rejected before
  horizontal interpolation, while ocean soil is repaired after interpolation
  from skin temperature and unit moisture.

The checked ERA5 contract declares its historical WRF `module_soil_pre`
layer-bottom interpolation. The GFS contract declares conservative layer
means; because its four bounds exactly equal Noah's, that route remains a
bit-identical copy. No runtime branch refers to ERA5, GFS, or legacy soil slab
field names. Every GRIB1, GRIB2, and NetCDF selector is checked
position-by-position against the selector stored beside its declared source
depth; GRIB2 type-106 numeric bounds receive an additional semantic check.
The Rust frontend independently repeats the engine's selector, depth, order,
remap, unit, and ocean-policy checks. This incompatible replacement of the
old named-packing document is explicitly versioned as
`gpuwm-mapped-composition-v2`; v1 evidence remains replayable at its sealed
implementation commit but cannot be mistaken for this contract.
Changing the composition bytes intentionally invalidates the retained exact
mapping/composition stock-WRF certificate. The new contracts remain runnable
but are not rebound to that certificate until a fresh unchanged-stock-WRF gate
is recorded, even though focused tests prove byte-identical ERA5/GFS soil
state relative to the retired packing branches.

One GRIB direct field looks like this:

```json
{
  "air_temperature_2m": {
    "vtable_selectors": [{
      "metgrid_name": "TT",
      "grib2_level_type": 103,
      "level1": 2,
      "selector": {"level_value": 2.0}
    }],
    "units": {"source": "K", "target": "K"},
    "source_axes": ["y", "x"],
    "target_axes": ["y", "x"],
    "location": "surface",
    "staggering": "none",
    "missing": {"kind": "reject"}
  }
}
```

The extra explicit `selector.level_value` is required for numeric GRIB2
`Level1` rows because the Vtable does not provide a general physical-unit
contract for GRIB2 scaled fixed surfaces. Bounded layers require the atomic
`second_level_type` and `second_level_value` pair. Zero matches, multiple
matches, reused rows, overlapping broad/narrow direct selectors, hand-authored
GRIB selectors, incomplete surfaces, and unknown keys all fail before a
mapping is written. NetCDF descriptors keep
the existing explicit `name`/`standard_name` selectors and reject Vtables.

GRIB1 references must additionally declare both `selector.center` and
`selector.table_version`; a Vtable does not carry enough authority to infer
them. A numeric GRIB1 `Level1` cannot be changed by an override (a `*` row may
be narrowed explicitly). GRIB2 local-use codes 192--254 require an explicit
four-part Section 1 authority binding: `center`, `subcenter`,
`master_table_version`, and `local_table_version`. Both Rust inventory and
dump tools emit those values, and the Python runtime rejects any disagreement
between them. `local_table_version=255` cannot authorize a local-use code.
Missing/undefined identifier code 255 is also refused for required GRIB1
identifiers and for GRIB2 parameters, categories, disciplines, and second
surfaces.
GRIB2 `level_type=255` is retained only as the explicit no-fixed-surface case
and cannot carry fixed-surface values. Duplicate JSON object keys are rejected
at every nesting level rather than silently taking the last value.

## Create-only authoring

The public command can compile the mapping and create the input manifest in
one author-only call:

```bash
rw-wps --source mapped --source-format grib2 \
  --descriptor /case/source.descriptor.json \
  --vtable /case/Vtable.PRODUCT \
  --author-mapping /case/source.mapping.json \
  --composition /case/source.composition.json \
  --input /data/source-f000.grib2 \
  --input /data/source-f003.grib2 \
  --supplement terrain=/data/terrain-f000.grib2 \
  --supplement terrain=/data/terrain-f003.grib2 \
  --provenance terrain_provenance=/case/terrain-source.md \
  --author-input-manifest /case/input-manifest.json \
  --author-only
```

`--input-list FILE` says the same thing as the repeated `--input` flag as a
file -- one path per line, UTF-8, in the same deterministic time/file order,
blank lines skipped, nothing interpreted. It exists for command-line length:
Windows caps a whole command line at 32 KB, and a field-per-file source is
hundreds of `--input` flags per prepared state (a real ICON-EU cycle is 251).
Exactly one of the two spellings per invocation; both `rw-wps`/`gpuwm prep`
and the internal mapped runner accept it, so a `--dry-run` preview of a large
run stays pasteable. When the per-file spelling is used and the platform
refuses the internal relaunch's length, the front door retries once through a
temporary list file by itself -- a command line the platform accepts is never
rewritten.

Use the exact role names declared by the composition. For an installed
distribution the launcher discovers its bundled GRIB bridge paths relative to
the installation, not through `PATH`; explicit decoder flags that resolve to
any other file are rejected. Manifest authoring verifies the ELF identity,
usage probe, and the generic GRIB2 tabular ABI before hashing a decoder.
Author-only and dry-run modes do not require a CUDA device; end-to-end native
WRF file production still does.

The same flags may be supplied with the normal WPS geometry, geography,
experiment, and output arguments. In that form RW-WPS authors the create-only
contract, inserts the resulting manifest path and digest into the internal
argv, and immediately runs the native initializer. `--dry-run` is strictly
side-effect free and refuses authoring flags. Use `--author-only` to write the
contract without starting initialization, then use those emitted paths in a
separate `--dry-run` if an argv preview is wanted.

Authoring never overwrites a mapping, receipt, or manifest. Source order and
role partitions are preserved; every resolved file is bound by byte count and
lowercase SHA-256. Files are hashed through stable open handles, the candidate
manifest is checked by the runtime verifier, all authorities are rechecked,
and only then is it published with an atomic no-clobber link. A failed rewrite
therefore cannot replace a prior valid manifest.

The stable-handle checks detect persistent path, identity, metadata, or byte
drift visible during the before/after snapshots. They are not a filesystem
lock and do not claim to defeat a hostile same-user process that can swap an
authority to different bytes only while an external decoder reopens it and
restore the original perfectly before the final snapshot. Production inputs
and bundled decoders should therefore be staged read-only or protected from
concurrent writers while authoring and decoding run.

When mapping and manifest authoring are requested in one call, a manifest
failure removes only the exact mapping and receipt proven to have been created
by that invocation. A corrected retry therefore does not hit a stale partial
transaction.

The generic mapped runner is reported as runnable but not globally
stock-WRF-certified. Machine-readable stock-WRF evidence is keyed to exact
retained mapping/composition SHA-256 pairs; a newly authored pair never
inherits those gates by sharing a decoder or format.

## Which engine reads the bytes

Nothing above depends on it -- a mapping document is the same document
either way -- but it changes what a mapped command needs on the machine.

By default the mapped routes decode through `gpuwm_mapped_engine`, one
executable resolved through the usual bridge ladder
(`GPUWM_MAPPED_ENGINE_BIN`, this checkout's `tools/rw_wps/target`,
`libexec/bridges`, the packaged copy, then `~/.gpuwm/bridges`) and held
to the contract marker `gpuwm-mapped-frameset-v1`. It reads GRIB2 in
process, so a GRIB2 mapped command needs no `--grib2-inventory` /
`--grib2-dump` paths at all, and an input manifest seals the ENGINE as
its decoder rather than the subprocess pair.

Three paths are still decoded by the Python engine on a bare run: a
cross-source composition, a mapping whose `format` is `grib1`, and a
mapping whose `format` is `netcdf`. All three are unported work rather
than a decision, all three are printed by `gpuwm doctor`, and
`gpuwm_mapped_engine capabilities` is the binary's own statement of what
it can do -- so if you are authoring a GRIB1 or NetCDF mapping today,
nothing about the document changes, only which engine reads it.

`--mapped-engine python` (or `GPUWM_MAPPED_ENGINE=python`) runs the
Python decode path instead. It is a WORKAROUND for a decode the Rust
engine gets wrong, not a supported mode; manifests sealed against the
subprocess tools replay on it. Naming a decoder tool explicitly selects
it for that run, because those flags pin which binary reads the bytes.

`gpuwm doctor` reports the engine and which one a bare run uses. The
contract, the refusal-class table and the parity gate are in
`docs/dev/decode-vendor-design.md`.

## Current boundary

This slice removes source-family dispatch, manual manifest JSON, and the two
named four-layer soil packings from the mapped path. It does not yet generalize
the terrain-only supplement engine. Curvilinear GRIB grids, vertical nest
refinement, moving nests, and two-way feedback also remain unsupported. A
descriptor must materialize the existing complete canonical source frame and
pass the existing donor-grid, hierarchy, initialization, and export gates.
