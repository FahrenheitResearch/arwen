
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
