# Mapped atmospheric interpolation support

The mapped preparation consumer may retain a typed atmospheric window. It is
an operand in the **original source index space**, not a replacement source
grid. `AtmosphericWindow` carries the original shape and retained row/column
intervals. `WindowedAtmosphericSnapshot` retains the original complete axes,
projection, valid time and full-source pressure ladder.

The common provider unions the existing FP32 interpolation support for every
declared target's mass, U and V points. Each CPU/CUDA interpolation plan then
selects its own original support within that union and uses its existing local
FP32 index arrays. Coordinates are never recomputed from cropped axes. Both wind
components retain support for both wind staggers before the unchanged rotation.

Only the six mapped atmospheric quantities consumed by the regular join and its
five explicitly initialized zero hydrometeors use this representation. The
surface, soil and water fields retain their full arrays and existing searches,
including distant masked donors and soil repair. The quantity inventory follows
the shared interpolation consumer; it contains no source/model identities.

The full frameset reader remains available. For a window request it reads one
complete source plane at a time, hashes every original byte, checks the original
missing count and infinity contract, and retains only the requested rectangle.
Pressure must be finite and positive over the complete source field. Its median
is computed separately from every complete plane before discarding columns;
the crop median cannot replace the original ladder. Even fields ignored by the
join still pass their existing complete validation before a snapshot returns.

No window request preserves the original owning constructor and full reader.
Cyclic sources currently keep the full representation so each domain can retain
its existing orientation. Any target outside a published window reloads the
original atmosphere through its retained source provider and preserves surface
overlays. That provider retains the source lifetime without retaining a snapshot
cache cycle. The existing geometry/coverage refusals remain authoritative.

The Rust writer advertises `features.atmospheric_window` with schema
`gpuwm-mapped-atmospheric-window-v1`. Preparation supplies all target grids only
when this feature is present. Older engines use the full writer and the plane
reader above. An ordinary invocation without a support request keeps the original
`gpuwm-mapped-frameset-v1` document and byte layout.

With `--atmospheric-window stdio`, complete canonical materialization precedes a
flushed `atmospheric_window_request` progress event. It names the original axes,
projection, field inventory and geometry identity (frame index, grid fingerprint
and axis digests). The parent returns the existing consumer's support union and
echoes that identity, or returns full mode when local support is not proven. The
writer rejects changed geometry, out-of-bounds rectangles, repeated or absent
fields, fields outside the six consumed atmospheric quantities, and fields
without complete vertical/y/x axes. This exchange happens
after decompression, derivation and validation; it does not crop compressed
messages or weaken validation outside the retained rectangle.

When a rectangle is published, the frameset uses the distinct
`gpuwm-mapped-windowed-frameset-v1` schema. Each cropped field carries its actual
payload shape, digest and missing count, plus the original complete field's
shape, digest, missing count and validation marker. The source axes, header and
geometry remain full. Pressure carries a digest-bound ladder computed from the
complete original planes. The reader checks the descriptor and every retained
payload, while source receipts keep the original field digests. The composition
provider can rerun the same sealed full decode for an unexpected later request;
it verifies clock, geometry, input authority and original field identities before
accepting that fallback. Surface and soil publication remains complete.

This reduces `frames.f64` bytes and retained atmospheric arrays. The Rust decoder
still materializes the full canonical frame, so this alone does not reduce its
native decode residency. Whole-preparation time and peak RSS can be dominated by
other stages and must be measured independently of the stream reduction.

Permanent controls are `test_atmospheric_window.py` (CPU/reader/geometry/lifetime)
and `test_atmospheric_window_gpu.py` (actual CUDA original-index and all-field
horizontal comparisons). `test_mapped_atmospheric_writer.py` exercises the actual
native writer, descriptor/request refusals, original-source fallback and retained
payload validation. `test_mapped_atmospheric_writer_gpu.py` compares its actual
retained payloads with the full native writer through every CUDA horizontal field.
