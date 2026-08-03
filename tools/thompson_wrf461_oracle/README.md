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

The build compiles at `-O2 -fno-tree-vectorize`, and that second flag is
load-bearing.  From GCC 12 on, `-O2` implies `-ftree-vectorize`, and a
vectorised loop containing `exp`/`pow`/`log` links glibc's libmvec SIMD entry
points instead of the scalar routines.  libmvec is not bit-identical to scalar
libm, and which loops vectorise depends on the cost model, which depends on how
much unrelated source sits around them.  Growing `run_column.F90` therefore
changed the oracle's answer once already, leaving a fixture set that
contradicted itself: `warm`, `mixed` and `ice` were generated when the file was
227 lines and the base-state loop vectorised, and the other 43 columns after it
passed a thousand lines and the same `-O2` no longer vectorised it.  Two
fixtures at the same `dz` then disagreed by 1-2 ULP about
`p = p0*exp(-z/8000)`, which does not depend on the scenario at all, and no
single build could reproduce the whole set.

With `-fno-tree-vectorize` the binary links no libmvec symbol (`nm -D
run_column | grep _ZGV` prints nothing, and the build refuses to continue if it
does), and the fixtures become invariant: gfortran 12.5.0, 13.4.0, 14.3.0 and
15.2.0, at `-O1`, `-O2` and `-O3`, with and without `-ffp-contract=off`, all
produce a byte-identical 92-file set.  `FC` and `OPT_FLAGS` can be overridden to
reproduce that comparison; leave them alone to regenerate the committed
fixtures.

One residual sensitivity is measured rather than assumed, and the receipt's
`libc` field is the pin for it.  Running the *same* devectorised binary under
glibc 2.39 and glibc 2.43 reproduces 23 of 24 spot-checked scenarios byte for
byte; `cold-cloud-overlap-column.csv` differs in a single cell (`qg` at k2,
8.1e-08 relative, 1.5e-11 absolute).  The glibc-2.39 run reproduces the
originally committed version of that file exactly, which is what identifies the
cause: that one fixture was generated on an older libm.  It is not `log10f` --
glibc 2.43 did re-version `log10f`, but forcing the pre-2.43 implementation via
`-Wl,--wrap=log10f` changes nothing, so the difference is in another routine
that changed without a symbol-version bump.  A reader on a different glibc
should expect that one file to differ by ~1 ULP in one cell and nothing else.

That prediction was checked from the outside, by the mp=28 lane, on a
different machine and a different toolchain patch level, before it took these
fixtures.  A clean `build.sh` run against a fresh WRF v4.6.1 checkout with
GNU Fortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) under glibc 2.39 --
against the 13.4.0 / glibc 2.43 the committed receipt records -- reproduced
**91 of the 92** fixtures byte for byte, and the one that differed was
`cold-cloud-overlap-column.csv`.  Exactly the file named above, and no other.
That is the paragraph's claim confirmed independently rather than restated,
and it is also the strongest evidence that the other 91 are now genuinely
toolchain-invariant: two builds two glibc versions and two gfortran patch
levels apart agree on all of them.

Every run writes `PROVENANCE.txt`, a receipt naming the WRF commit and the
SHA-256 of both compiled `phys/*.F` files, the SHA-256 of every harness source
*and of `build.sh` itself*, the compiler and libc versions, the exact compile
and link lines, the binary and table-cache hashes, and a per-fixture hash plus
a rollup.  A copy is committed beside the fixtures, and
`tests/test_thompson_oracle_provenance.py` fails when the harness in the tree
no longer hashes to what the receipt recorded -- which is the check that would
have caught the stale fixtures on the commit that created them.

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
