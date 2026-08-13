# rw-fieldcmp

Two instruments that read paired history streams.

`rw_fieldcmp` reports what each arm looks like, frame by frame: the campaign
judge's metric table. `rw_runscore` reports how far apart the two arms are as
runs: one distance per registered metric, pooled over every scored time.

```
rw_fieldcmp <left-dir> <right-dir> [options]
rw_runscore <left-dir> <right-dir> --start WHEN --run-seconds N \
            --cadence-seconds N --domain LABEL=DX [options]
```

## rw_fieldcmp: the per-frame table

For every frame name present in both directories it reports, per named
surface field, each arm's mean / p10 / p50 / p99 / max and their differences;
per named accumulation field, each arm's domain sum and the percent ratio
between them; and, for a named volume field, the per-column composite's
coverage counts at each threshold plus its maximum. `--table` writes the
table to a file, `--json` writes every number the table rounds, `--threads`
sizes the worker pool. `--help` lists the rest.

Nothing about a case, a model configuration, or a physics suite is baked in.
The arm labels, the field labels, the accumulation list, the composite
variable and its thresholds all come from the command line; the defaults name
WRF-convention variables and nothing else.

## rw_runscore: the paired-run distances

Four metric classes, each pooled over the scored times and reported per
domain and per field:

- **state**, the RMSE of the low-pass-filtered difference over the domain
  interior;
- **boundary**, the RMSE of the difference between the two arms'
  per-interval increments, over the outer frame of cells;
- **object**, the gap in seconds between the two arms' first qualifying
  composite object;
- **neighbourhood**, the mean `1 - FSS` at a physical radius, on whichever
  domains `--neighborhood-domain` names.

Filter widths, neighbourhood radii and object sizes are physical, so each
domain's cell counts follow from its own `--domain LABEL=DX_M` spacing. The
frame ladder is checked against `--start`, `--run-seconds` and
`--cadence-seconds` before anything is opened, so a staging mistake is named
in a second rather than after half an hour of decoding.

Metric keys are `category:domain:subject`. The four category spellings and
the two subject spellings are command-line arguments, because a published
gate record's key strings are part of what it published; the defaults are
generic and name no domain. `--json` writes the distances, the geometry each
domain was scored under, the read counts, and the per-interval sums the
pooled numbers are built from.

## Why it is written the way it is

Each frame is opened once per arm and each variable is read once from that
open handle, including variables that are both scored and summed. Frames are
compared in parallel, the two arms of a frame are read concurrently, and the
volume reduction is spread across workers. Peak memory is one promoted volume
per frame-arm being read at once, which `--threads` bounds.

`rw_runscore` walks its ladder once. The reference reads every frame four
times per field, once as the current frame of its own interval and once as the
previous frame of the next, on each arm, reopening the file every time: 672
decodes of the state fields and 720 file opens for a four-domain seven-field
pair. This decodes each frame's fields once from one open handle and reduces
each one on the spot to the two things later times need from it, the scored
interior sum and the outer ring of cells, then drops it. That is 440 decodes
and 56 opens for the same pair, and the peak is a handful of arrays rather
than a ladder of them.

The arithmetic reproduces the reference's rather than merely being defensible,
because a verification instrument whose numbers move when it is reimplemented
cannot be used to judge anything. Five places carry that burden and are
commented where they are written:

- the block-of-128, eight-accumulator pairwise summation (`src/stats.rs`);
- the buffering *around* it. The reference's reduction hands its summation
  kernel 8192 elements at a time and adds each buffer's total into a running
  scalar, so an array longer than one buffer is summed as a sequence of trees
  rather than as one tree. Past about a million elements the two arrangements
  differ in the last bit or two, which is the size every paired-run metric
  works at. 8192 is a *settable* default in the reference stack: a caller that
  changes it changes the last bit of every large sum on both sides;
- the memory order the reference sums in (`src/gridops.rs`). Selecting the
  outer frame of cells from a stack of planes leaves the reference holding a
  column-major array, and its sums walk memory rather than logic, so the ring
  is packed cell-first here to match;
- the `linear` quantile with its virtual-index and interpolation rules, and
  the prefix-sum shape of the separable boxcar, which loses low-order bits
  where a direct windowed mean would not (`src/stats.rs`, `src/gridops.rs`);
- the single-precision accumulation the reference uses for domain sums. This
  is the one place where the reference is the less accurate of the two, so the
  JSON carries the double-precision sums beside the printed ones rather than
  instead of them.

The second and third of those were found by disagreeing with the reference by
one part in ten thousand million and refusing to call it a tolerance.

`src/numfmt.rs` reproduces the four-significant-digit general format the table
prints, including the fixed-versus-scientific choice and the two-digit
exponent, so a table produced here diffs cleanly against a shipped one.

## Parity and cost: `rw_runscore`

Measured against the Python paired-run scorer on a real two-arm WRF case from
the pinned reference build, seven frames per domain per arm staged on a
four-domain ladder, 6.0 GB of history, cache warm:

| | wall |
|---|---|
| reference scorer | 111 s median, 121 s in a separate round |
| `rw_runscore` | 16.7 s median, 16.8 s in that round |

All 61 distances are bit-identical: every pooled state RMSE, every pooled
boundary increment error, every object-timing difference, the neighbourhood
row. Not "agree to twelve digits", the same doubles.

Getting there took three corrections to what "reproduce the reference's
arithmetic" meant, each found by refusing to accept a one-bit disagreement as
a tolerance: the buffered reduction, the column-major memory order of the
boundary selection, and the prefix-sum shape of the boxcar. The first two are
described above; all three are commented where they are implemented. The
receipt is `evidence/verify-instrument/paired-run-score-parity.json` and
`tools/verify_runscore_parity.py` regenerates it.

One measurement worth carrying forward: the campaign door also re-hashes every
history frame against its run artifact before scoring, and on a warm cache
that leg costs about 2 s of the 122 s door. The metric arithmetic was the
whole cost, which is why replacing it was worth doing and why the frame
re-hash was not.

## Parity and cost: `rw_fieldcmp`

Measured against the reference judge on a two-domain, seven-frame paired case
(d01 250x200, d02 500x400, 50 levels, 14 frames per arm, 11.6 GB on disk),
both arms read from local NVMe with the file cache warm:

| | wall |
|---|---|
| reference judge | 2.03 s median of five, 1.90 s best |
| `rw_fieldcmp` | 0.77 s median of five, 0.67 s best |

2.6x on the median. The table is byte-identical across all 529 lines,
including the domain sums, the undefined ratios on the zero-accumulation
frame, and the absent-variable rows.

Two measurements worth carrying forward. Single-threaded, `rw_fieldcmp` takes
3.0 s, slower than the reference: the win here is entirely parallelism, not
arithmetic, because the reader promotes single-precision fields to double on
read where the reference hands them over in their on-disk dtype. And 0.48 s of
the 0.77 s is the reflectivity volumes alone, which are two orders of
magnitude larger than every surface field combined. Both point at the same
next lever, which is the reader rather than anything in this crate: a read
path that keeps single-precision fields single-precision would roughly halve
the bytes moved.
