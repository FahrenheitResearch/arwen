# Vendored GRIB decoder provenance

`grib-core/` is an exact source export of BowEcho's ratified hardened decoder:

- source: <https://github.com/FahrenheitResearch/rusty-weather>
- upstream revision: `fe9797f86c8958b5a625a4a7682c6b6aeff6b309`
- authoritative upstream `vendor/grib-core` Git tree pin:
  `7c036c2b18c0a9bb014eddd55cdacf84a244fed0`
- crate/version: `grib-core` 0.1.0
- license: MIT, as declared by the upstream workspace `Cargo.toml` at that
  revision.  The upstream `grib-core` subtree contains no separate license
  file; this provenance note preserves the exact declaration rather than
  inventing different package metadata.

The revision is the one BowEcho pinned after July 2026 GRIB1 hardening.  The
Git tree object above is the base content identity.  gpuwm carries narrow,
auditable local deltas required by the native GFS bridge:

- `src/grib2/parser.rs` exposes Section-1 origin/table/production identity,
  the analysis/forecast generating-process ids, and the second fixed-surface
  type/value.  It also corrects negative sign-magnitude fixed-surface scale
  decoding.  These fields bind the operational NCEP GFS product and validate
  the exact 0-10, 10-40, 40-100, and 100-200 cm GFS soil slabs.  It also
  enforces exact GRIB envelope coverage, the Section 1/3/4/5/6/7 state
  machine, Section-3/5/6 cardinality, bitmap padding and grid-bound bitmap
  reuse, complete PDT 4.0 fixed-surface descriptors, and the required DRT 5.0
  original-field-type octet.
  Patched-file SHA-256:
  `01b0b3df38b55d0327332cda6a0e1e444e822eaa29aaf8b003199bc1c2185c3d`.
- `src/grib2/mod.rs` re-exports that identification value object.
  Patched-file SHA-256:
  `6382513b312ddf4ba642568ff999ae661eec6e796f6f4059f80074d601150be8`.
- `src/grib2/search.rs` initializes the new identity in its pre-existing
  synthetic test messages. `src/grib2/unpack.rs` additionally enforces
  Section-5/7 simple- and complex-packing cardinality, checked bit reads, a
  signed 63-bit simple-packing limit, and exact bitmap consumption. Patched-
  file SHA-256 values are respectively
  `22254047da83fafcc06816170b7687127cb9f64bc7e82e4de2d239861c149547`
  and `1d0b9fd05104497fd0176a362235c9d08dc00abe9873ee81c104c4faac7f9d47`.
- `src/grib1/parser.rs` spells six `Option::is_none_or` checks as the
  semantically identical `Option::map_or(true, ...)`, retaining compatibility
  with the rental runtime's Rust 1.75 compiler.  Patched-file SHA-256:
  `48b817c88d1deca4e8d2eea80fbf6dea7b0393d44d0b5995f52b5fd9f1d9ae3a`.

The committed Cargo lock format is version 3 for the same Rust 1.75
compatibility.  The GRIB1 bridge retains its version-1 JSON metadata plus
little-endian-f64 value-stream protocol; the new GFS bridge has its own
fail-closed FP32 series protocol documented in the parent README.

`crates-io/` is generated from the committed `Cargo.lock` by `cargo vendor`.
Every package retains Cargo's `.cargo-checksum.json` plus any upstream license
files.  `.cargo/config.toml` replaces crates.io with that directory, enabling
`cargo build --locked --offline` from a clean checkout.
