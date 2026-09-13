# Device-sized validation scans

The immediate MYNN validation scan keeps its predicates, grid-stride loop,
per-array masks and current stream. Its grid now shares 32 blocks per device
SM across the arrays with integer floor division, capped by the longest
input's 128-thread block count. Device metadata is cached by device id and
queried only on first use. No scientific arithmetic or state storage changes.

The fixed eight-block cap left a six-array large-field group with only 48
blocks for the entire device. Eight warmed order-balanced trials, each with
15 complete validations including the immediate mask read, measured these
median milliseconds on a 70-SM RTX 5070 Ti:

| Arrays x elements | Per-array reductions | Fixed 8 | 8 x SM, ceil share | 32 x SM, floor share |
| --- | ---: | ---: | ---: | ---: |
| 6 x 18,000,000 | 0.547 | 5.251 | 0.590 | 0.546 |
| 6 x 1,620,000 | 0.063 | 0.226 | 0.034 | 0.029 |
| 4 x 819,200 | 0.045 | 0.123 | 0.023 | 0.020 |
| 50 x 76,800 | 0.440 | 0.050 | 0.041 | 0.041 |

Finite inputs, NaN and both infinities produced equal masks under all four
policies, and all inputs retained their values. Separate controls exercise
variable lengths, empty arrays, tail elements, signed zeros, subnormals,
custom predicates and concurrent streams. These timings describe validation
only; they make no forecast speed or scientific accuracy claim.

The native publication replacement tests also use a portable injection:
on Windows the held target is renamed aside before its address is replaced.
The test removes that aside file after the writer closes its descriptor,
before verifying the writer's own recovery copy. Otherwise an injection
fails with a sharing violation, or its own aside file could falsely satisfy
the recovery assertion. Production publication semantics are unchanged.

## Windows completed-digest reuse

Actual rapid same-size edits with restored modification times produced equal
Windows `FILE_BASIC_INFO.ChangeTime` values before and after the edit. In one
30-trial control, 18 changed files inherited the old digest because every
revision field compared equal. ChangeTime remains useful for detecting many
changes, but it cannot establish that the content is unchanged.

Windows revisions now require a full current-byte hash before a completed
digest is reused. The existing digest comparison rejects modified bytes and
keeps the original completion record. Linux retains the revision-based fast
path. This restores a Windows reread cost; the change makes no claim to have
eliminated that cost there. A deterministic collision control and native
filesystem trials preserve the reason for this distinction.
