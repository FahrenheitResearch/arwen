
## gpuwm addition: `met_intermediate`

`src/bin/met_intermediate.rs` is a gpuwm-authored binary in this
workspace, not part of the vendored `grib-core`.  It writes the WPS
version-5 intermediate format -- the file `metgrid` reads, and the file
MPAS's `init_atmosphere` reads directly through
`mpas_init_atm_read_met.F` -- from GRIB1 or GRIB2, driven by the same
Vtable files `ungrib` reads.  It lives here rather than in
`tools/rustwx` because it is a decoder-side tool and `grib-core` is
here; it adds no dependency and touches nothing vendored.

It is the seam that removes `ungrib.exe` from a real-data
initialization without asking anything downstream to move.  Parity
against `ungrib`'s own output on the same GRIB files, including three
characterised disagreements and two findings that an independent
ecCodes decode attributes to `ungrib` rather than to this tool, is
recorded in `evidence/rw-wps-met-intermediate.json`.

The producing centre it is given is a closed vocabulary, not free text.
`ungrib` reads that string out of the GRIB and then makes eight separate
decisions by substring test on it; this binary takes it from the command
line, so a label matching none of those tests would have turned every
repair off at once and still exited zero.  There is no default: an
absent or unknown `--map-source` is a refusal that names the labels it
knows, and each rule's on-or-off decision, with the reason, is printed
in the JSON receipt.  Both arms now match `ungrib`'s record set, record
order and every header field exactly; `evidence/rw-wps-met-intermediate.json`
carries the reconciliation and three characterised `ungrib` behaviours,
including soil over water being read out of uninitialised memory.

## CPU preparation execution and output

The bridge adds Rayon for persistent host interpolation and math workers,
and SHA-256 for independent prepared-array writes. Their registry sources
are copied unchanged from the same versions already vendored in `rw_wps`,
including their Cargo checksums and licenses. The shared GRIB decoder is
unchanged.

`src/glibc239_math.rs` transcribes the existing pinned FP32 power in
`gpuwm/core/noahmp_libm.py`, retaining its separate binary64 operations and
Arm MIT notice. It does not call the platform C library.

`src/prepared_io.rs` writes exact NPY metadata supplied by the caller and
streams payloads and hashes through 128 KiB per active worker. Inputs and
result slots are independent; atomic publication stays in input order.

`../preparation_resources.rs` is shared with the mapped engine. It belongs
to both crates' declared native build inputs, so source closures and binary
reuse checks must include it.

The separate `gpuwm_host_*` entries in `src/portable_math.rs` preserve the
host C-library contract of `gpuwm/core/host_libm.py`; they do not replace
the vendored portable math entries. `src/cold_start.rs` evaluates the
existing cold-start number formulas over independent cells, preserving
their FP32 and FP64 rounding points. `src/host_arrays.rs` casts setup
arrays directly into their destinations and builds the geopotential
residual and owned host snapshot without full-array cast temporaries.
It also retains the original pressure-face averaging, ordered moisture
sums, and subtract-then-cast state assignments over independent cells.

`src/preparation_fingerprints.rs` hashes immutable array bytes and packed
nonzero masks with bounded scratch. Independent fields may run in parallel;
each SHA-256 stream retains its original byte order. NaN and signed-zero
extrema keep NumPy's original reduction semantics in the Python wrapper.
