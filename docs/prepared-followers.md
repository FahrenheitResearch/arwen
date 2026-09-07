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
