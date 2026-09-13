# MYNN validation batching

Groups of at least four contiguous FP32 inputs share a read-only validation
scan. The caller clears its existing integer flags and the scan sets one
word per input. Pointers and lengths are launch arguments; compiled code is
shared, while status buffers remain with each workspace and stream. No
device descriptor or additional persistent field is allocated.

The same immediate read occurs at the same call site. Shape checks, input
conversion, finite/positive checks, sorted nonzero-field names and error
ordering are unchanged. Small groups, noncontiguous inputs, other dtypes and
custom predicates retain the original reductions. Explicit flush-to-zero
behavior matches the existing reduction predicates, including signed zeros
and positive subnormal comparisons. Scientific MYNN kernels are unchanged.

A warmed 48x32x50 MYNN/Noah/WSM6 control used the actual 50-input group and
three order-balanced parent/candidate pairs. Validation medians were about
445 versus 49 microseconds. Complete step deltas were -0.177, -0.238 and
-0.240 milliseconds, a median reduction of about 3.5 percent in this control.
Every carried-state, output-field and Rust-written history-file digest
matched. These measurements do not predict speed on another domain,
physics combination, device or storage system.

The inherited driver-fixture comparison still reports 3277 ULP for
`rublten` against its historical 819 ULP budget on this verification
configuration, identically before and after batching. Its numerical budget
is preserved. This performance change makes no new accuracy or WRF-parity
claim and introduces no deferred-health publication path.
