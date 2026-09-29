# Following nests from a prepared hierarchy

The prepared hierarchy runner accepts both the existing `[relocation]` form
and independent `[domain.follow]` policies. Each follower retains its own
tracker, consultation cadence, bounds, cooldown, move history, and static
corridor. A domain must have one placement authority.

For an existing hierarchy config, add a follow table immediately after the
domain it controls. For example, this pressure tracker consults the live
850 hPa surface every 72 seconds:

```toml
[domain.follow]
field = "pressure"
level_hpa = 850.0
threshold = 1.0
search_margin_cells = 6
min_shift_cells = 1
max_shift_cells = 2
cooldown_seconds = 72.0
cadence_seconds = 72.0
max_move_parent_cells = 2
min_overlap_fraction = 0.5
```

These are example choices, not required settings. A second domain may have a
different cadence and bounds. Pressure tracking reads the live column;
UH/reflectivity tracking retains the existing requirement that the parent's
reflectivity output cadence can supply every consultation. Existing explicit
domain geometry, vertical levels, physics, and numerical settings remain
authoritative.

`gpuwm go` derives corridor preparation from every declared follower. When
calling `gpuwm prep` directly, include `--statics-corridor`. Each mover and
descendant carried by a move needs a verified corridor covering its new
ground. The runner checks corridor hashes and coordinate frames before
initialization. A descendant of another moving domain uses a root-anchored
corridor rather than treating its moving parent as stationary geography.

Each corridor covers the ground its domain can reach during the run, not the
whole parent. The reach is the declared footprint widened by the most the
mover can travel: its largest move at every cadence its cooldown allows over
`run_seconds`, the rows of a `[[relocation.move]]` itinerary, and
`reach_speed_m_s`, the average speed since the start the nest may not exceed
(default 40 m/s, the default of WRF's own `max_vortex_speed`). A domain riding
inside a mover reaches what its carrier reaches, and no reach leaves the
parent the mover sits in. The runner clamps a move past `reach_speed_m_s` and
names it in the receipt's `clamped_by`, so a nest never leaves its corridor.
The preparation prints each corridor's share of its frame, and says so when
the reach is the whole frame: a long run, or a dormant nest whose start is
chosen when it fires. A run that can reach further than the sealed corridor,
for example one extended on restart past the length it was prepared for, is
refused at load with both extents named. A corridor sealed before 2.8 covers
the whole frame and serves any reach.

```toml
[domain.follow]
# ... the tracker keys above ...
reach_speed_m_s = 15.0   # a storm known to move slowly: a smaller corridor
```

Each sealed corridor also records the build contract its field bytes were
produced under. For a corridor sealed before that contract, the runner
compares the crop at the child's prepared placement with the tree's own
sealed child statics. It uses the first relocation's shared-ground rule,
including acceptance of adjacent floating-point values within one ULP.
A parent-anchored corridor uses the child's placement in its parent; a
root-anchored corridor uses the child's composed origin in the root frame.
Receipt, digest, geometry and field-inventory checks still run first.

This check needs no geography and never calls the current static builder.
It also accepts applied high-resolution overlays when the corridor and
sealed child statics agree. A mismatch refuses the corridor, naming each
field and its mismatched cell count and giving the `rw-wps` rebuild line
with `--statics-corridor`. That refusal prevents the first relocation from
changing statics on shared ground. Later moves compare crops from the same
corridor, so a newer builder's numerical changes do not affect this check.

The comparison reruns at each run start, including a restart. Its work
scales with the prepared child footprint, rather than rebuilding the full
corridor. Verification never rewrites the sealed files.

In CPU checks of two real 2.7.4 preparations, each with a 900x900 corridor
and a 300x300 child, the comparison took 0.058 to 0.065 seconds. Loading
the corridor, including its digest check, took 0.889 to 0.907 seconds.
The comparison made no static-builder calls in either preparation.

Launch or continue through the ordinary prepared-data command:

```text
gpuwm sim PREPARED --experiment-config CONFIG --outdir NEW_RUN
gpuwm sim PREPARED --experiment-config CONFIG --restart CHECKPOINT --outdir CONTINUED_RUN
```

Continuation restores each follower's placement through its own initializer,
then its tracker state and move identity. History output adopts the actual
new coordinates after each move. The run receipt includes the per-domain
follow policies and ordered relocation events; individual follower receipts
are written separately.

The 2.7 acceptance witness exercised two resident moving children beneath
both a resident parent and a stationary parent using a host tile store.
The children moved at independent 72- and 108-second cadences, and a public
checkpoint continuation reproduced every remaining history and checkpoint
array exactly. This does not establish bounded-memory reconstruction of a
moving streamed child: that operation still requires its separate shared
store/geometry reconstruction implementation. The current parent donor
capture also materializes the parent state at move time, so this witness is
not a beyond-VRAM relocation claim.

Prepared followers in this route are live at the experiment start. The
existing prepared-route admission checks still name missing spawn-trigger
reservation/evaluation and delayed activation-epoch initialization. The
prepared executor does not currently run the case-data birth callback or
leg walker, so this change does not claim follower birth or delayed starts.
Those configurations continue to use the existing case-data lifecycle route.
