# The cyclone setup document

`gpuwm cyclone-setup` prints one JSON document on stdout and nothing else;
everything it says while working goes to stderr. This page is that document:
what each kind carries, and which fields a client may rely on. The schema
string is `arwen.cyclone-setup.v2` on every kind.

The door authors a 12 km parent with a 3 km following nest centred on a
cyclone. It never starts a forecast: `forecast_started` is `false` on every
document it emits.

Two choices are the caller's: WHEN in the selected cycle the run begins
(`--start-hour`, default the analysis) and HOW BIG the following nest is
(`--nest-budget-gib`, default the preset nest). Both are described below.

## Any source, by table

The source is `--source`, and which sources may be named is not a list in this
door. It is the source registry intersected with the acquisition routes, so a
model becomes selectable here by getting a registry row and a fetch route, not
by getting a code path. `--list-sources` prints the current set.

Everything the document says about the source is read from that source's own
row: its published cycle hours, its forcing interval, its coverage window, its
member grammar, its recommended physics profile and its preparation recipe.
Two sources at the same point and cycle author the same grid, the same
vertical ladder, the same nest and the same follower; what differs is the
`[fetch]` block, the `[case_data]` recipe, the recommended physics, the
configuration name, and the model top where the source's certified inventory
floors it.

## Kinds

`kind` says which document this is. A client reads `kind` first.

### `sources` -- the menu

Emitted by `--list-sources`. `sources` is a list of rows, one per planable
source:

| field | meaning |
|---|---|
| `source` | canonical registry id; the exact string to pass back as `--source` |
| `label` | the source's display title, for a person |
| `members` | member ids in the route's own grammar, in route order; `[]` for a deterministic source |
| `default_member` | the member used when `--member` is omitted; `null` for a deterministic source |
| `forcing_interval_seconds` | the source's boundary cadence, in seconds |
| `cycle_hours` | the UTC hours this source initializes at; `[]` where the row declares no cycle grid |
| `coverage_envelope` | `[south, west, north, east]` in degrees for a regional source, `null` for a global one |
| `follow_statics` | how the chain this source runs on delivers the statics a moving nest travels over, in the run door's own word (`statics_corridor`, `case_data_ingest`, `retained_corridor`); `null` where that chain delivers none |
| `max_forecast_hour` | the last forecast lead this source's registry row declares, so a form can bound a `--start-hour` field; `0` means the row publishes analyses only and the flag does not apply to it |

Beside `sources`, the menu carries `nest_floor`: the nest a
`--nest-budget-gib` grows up from, as `dimensions`, `parent_cells`,
`parent_dimensions` and `ratio`. A budget under what that floor costs is
refused, so a form can bound its own field before anything is priced.

A member id is a string from `members`, never an index. The ordinal that
appears as `map_request.member` is a different thing (see below) and the two
are not interchangeable.

### `map` -- the selection map

Emitted by `--latest-map`. `map_request` names the exact field to draw:
`source`, `date`, `hour`, `forecast_hour` (the `--start-hour` lead, 0 for the
analysis), `member` (a numeric ordinal for the map product, 0 for a
deterministic source), `product` and `bounds`. The document repeats the lead
as `forecast_start_hour` and gives the moment it is valid at as `valid_time`,
so a picture drawn from this request can be labelled with the time the run
will begin at. `bounds` is the source's own coverage envelope clipped to the
drawable band, so a regional source does not offer a global frame to click in.
The document's top-level `member` is the resolved member id string.

### `configuration` -- the authored setup

The requested layout was admitted. `config_text` is the configuration TOML;
`domains` lists the authored grids; `follow` is the vortex-lock table the
following nest carries; `nest` is how big that nest is and what it costs;
`memory` carries the peak envelope, the budget and the basis it was sized
from -- and under `--nest-budget-gib` its `free_bytes` is the narrowed
allowance every number on the document was measured against, not the card's own
free memory, with `nest.budget_bound_by` saying whether the request or the card
set it; `streaming` says which road each domain takes. `forecast_start_hour`
and `start_time` say where in the cycle the run begins. `fitting.changed` is
`false`.

With `--out`, the file is written and `created` becomes `true`, beside a
`.namelist.wps` carrying the selected source's `interval_seconds`, a
`.cyclone.json` receipt, and -- for a source whose preparation recipe names one
-- a `.Vtable`. Nothing is ever overwritten.

### `proposal` -- a reduction to review

The requested layout was not admitted and a smaller one is proposed.
`fitting.changed` and `fitting.review_required` are `true`, `fitting.changes`
lists every field that moved, and `fitting.notice` says so in words. With
`--out`, a proposal exits 0 and writes nothing: pass
`--accept-fit fitting.fit_id` to save that exact reviewed proposal.

## Where the run begins

`--cycle` chooses which run of the source to initialize from. `--start-hour N`
chooses where in that run the forecast begins: the configuration is
initialized from lead N and forced from N onward at the source's own cadence,
and `start_time` becomes the cycle plus N hours. The default is 0, the
analysis, and a configuration authored without the flag is byte-identical to
one authored with `--start-hour 0`.

This is how a storm the model develops at a late lead is forecast at all. A
system that only exists at f186 of a run is initialized from f186; waiting for
the cycle whose analysis contains it is no longer the only route.

The lead grammar is the fetch route's. `[fetch] forecast_start_hour` is the
same key `gpuwm fetch` takes, the leads are resolved on the producer's own
published ladder, and a lead that producer skips is refused by that ladder
with its own words. Two bounds are checked before anything is fetched:

* a source whose registry row declares `max_forecast_hour = 0` publishes no
  leads at all, and the flag is refused naming that row -- every time in such
  a source is an analysis at its own valid time, so `--cycle` is the flag that
  moves it;
* a lead past the horizon of the cycle it is asked of is refused naming the
  horizon, the lead and the lead the window needs. The horizon is that
  CYCLE's, not the row's ceiling: several producers run a longer forecast on
  some cycle hours than on others.

`--latest-map` takes the same hour, so the centre is clicked on the field the
run is initialized from. With `--cycle latest` the cycle resolved is the newest
one whose analysis is complete, so a preview asked at a deep lead of that cycle
can name a field the producer has not published yet; name the earlier cycle to
draw it. Only the map is exposed to this: the configuration surface refuses
`--cycle latest` outright and asks for a resolved cycle first.

## How big the following nest is

The preset authors a 200x160 parent at 12 km and a 160x160 nest at 3 km, which
is 40 parent cells across. `--nest-budget-gib GIB` grows the nest, and only
the nest: it stays square, it registers on whole even parent cells, the parent
is untouched, and the largest size whose priced tree fits inside the budget
wins.

The budget is the memory the sized tree may occupy. It narrows the allowance
at the one place the card's own free memory enters, so every number downstream
is measured against it, the tile planner's tree road included, and the size
that comes back was admitted by the same estimator and the same headroom the
proposal is then reported against. A budget larger than what the card has free
is bound by the card instead, and the document says which of the two bound it.
A budget under the preset nest is refused, naming the floor in cells and in
parent cells, what that floor prices on this machine, the number that price was
compared against and the way out. That number is never the budget itself. On
`--tiles off` it is the fit target: the budget less the headroom the admission
leaves unspent. On the default `--tiles auto` the tree walk withholds the
following nest's rebuild transient before it compares anything, so what binds
is a smaller admission budget still, and the refusal quotes that one instead,
names the bytes withheld and for which grid, and carries the walk's own way
out beside the flag. Either way the quoted number is at or under the price, so
raising the flag to just over what the floor costs is never the advice given.

When it is the CARD and not the named budget that cannot hold the preset nest,
the refusal says that instead. A budget already larger than the card's free
memory never bound anything, so raising it moves nothing; `memory.bound_by`
reads `card` and `memory.reason` reads `nest-floor-card`, the sentence names
the card's own sizing budget as the bound, and the way out is to drop the flag.
That way out is measured before it is offered: the layout this door's own
reduction road then authors on that computer is printed in the refusal and
carried on `memory.unbudgeted_alternative`.

The nest is bounded by geometry as well as by memory: the parent has to keep
the nest's tracker search window clear of its own boundary and blend zone, so
the ladder ends where a larger nest would reach them.

THE MOVEMENT BOUNDS ARE DERIVED FROM THE SIZE. The overlap floor
(`min_overlap_fraction`) is the number that states the physics -- how much of
the child a move must keep -- and the per-axis maximum follows from it: an
overlap floor `f` admits a magnitude of `1 - sqrt(f)` of the nest's own width
in parent cells on the binding diagonal move. The preset's own maximum is that
number at its own size: 6 parent cells on a 40-parent-cell nest at a 0.7
floor. A nest grown past it carries the larger maximum its own width admits:

| nest, in parent cells | `max_move_parent_cells` |
|---|---|
| 40 (the preset) | 6 |
| 42 | 6 |
| 44, 46, 48 | 7 |
| 50, 52, 54 | 8 |
| 56, 58, 60 | 9 |

Read that before spending memory on a nest: the first step up buys ground
without buying reach, and the maximum moves at 44 parent cells. The tracker
search box grows with the nest, half its width on each side, and never below
the preset's own 20 cells.

It is one derivation in both directions, and the reduction ladder shares it: a
layout smaller than the preset is narrower, so its own floor admits fewer
parent cells, and a rung carrying the preset's 6 would declare a move the
floor then refuses. A 36-parent-cell nest admits 5 and a 24-parent-cell nest
admits 3. At the preset's own size the shipped table is emitted verbatim, so a
default run authors exactly the file it authored before a nest could be
resized.

The `nest` block on a `configuration` or a `proposal` carries the whole
decision:

| field | meaning |
|---|---|
| `dimensions` | the authored nest, in its own cells |
| `parent_cells` | the same nest in parent cells, which is the unit it is sized in |
| `floor_dimensions`, `floor_parent_cells` | the preset nest a budget grows up from |
| `ratio` | the parent-to-nest refinement ratio |
| `budget_gib` | the requested budget, or `null` when none was named |
| `budget_bytes` | the budget the tree was admitted against |
| `budget_bound_by` | `request` when the named budget bound it, `card` when the card's free memory did, `null` when no budget was named |
| `headroom_bytes` | the part of the budget the admission leaves unspent |
| `peak_envelope_bytes` | what the priced tree costs |
| `sized_to_budget` | whether a budget chose this nest |

The same three facts print on stderr as one `plan:` line, so a reader who is
about to save the file sees where the run begins, how big the nest ended up
and what it costs.

## Fields a client consumes

These are the fields the desktop reads, named exactly:

* on `sources` rows: `source`, `label`, `members`, `default_member`,
  `cycle_hours`, `coverage_envelope`, `forcing_interval_seconds`,
  `follow_statics`, `max_forecast_hour`; and beside them, the menu's own
  `nest_floor`;
* on a `configuration` or a `proposal`: `forecast_start_hour`, `start_time`
  and the `nest` block;
* on a `configuration` or a `proposal`: `follow_statics` at the top level,
  the block described under "The moving nest and the chain that feeds it";
* at the top level of a `map`, `configuration` or `proposal`: `member` -- the
  resolved member ID string, or `null` for a deterministic source;
* on a `map`: `map_request.member` -- a numeric ordinal for the map product.
  It is not the member id and must not be sent back as one.

`sources`, the top-level `member`, `forcing_interval_seconds`, `seed`,
`follow_statics`, `forecast_start_hour`, `start_time`, `nest`,
`max_forecast_hour` and `nest_floor` are
additions beside the existing keys. No key present before them moved or
changed meaning, so a reader written against the earlier document reads this
one; a kind it does not know is a kind it skips.

## The moving nest and the chain that feeds it

This door always authors a following nest, and a moving nest needs
child-resolution statics over the ground it travels. Which of this release's
preparation chains can deliver those is a table in the run door
(`gpuwm.runplan.source_follow_statics`), and this door reads that same table
rather than a second copy: a `configuration` or a `proposal` carries a
top-level `follow_statics` block, and every `sources` row carries the same
answer as one field, so a picker shows the limit where the source is chosen.

| field | meaning |
|---|---|
| `source` | the canonical registry id the answer is about |
| `chain` | the chain this source dispatches to, or `null` if the row reaches none |
| `delivery` | the delivery word, or `null` when the chain delivers no statics for a moving nest |
| `integrates_moving_nest` | `true` when the authored follower runs as authored |
| `reason` | why not, in the chain's or the row's own words; `null` when it does |
| `launch_refusal` | the sentence `gpuwm go` raises for a source that reaches no launch route at all, verbatim; `null` for every source that reaches a chain |
| `note` | the whole answer in one sentence, including the sources that do carry a moving nest |

`delivery` is `null` for two different reasons and `launch_refusal` is which
one. A source on a chain whose preparation seals no corridor launches: only
its following nest is refused, and dropping the follow source for a
bounds-only `[relocation]` runs the rest. A source that reaches no launch
route does not launch at all -- `gpuwm go` refuses it before it reads anything
in the file -- so the note quotes that refusal, names no way out that leaves
the source in place, and the comment written into the configuration is headed
`Launch route for this source:` instead.

This is not a refusal. The setup is authored, priced and reviewable on every
planable source, and a source whose chain delivers no corridor is still worth
authoring: the grid, the physics and the acquisition block are the same work.
What changes is that the document says, before anything is fetched, what
`gpuwm go` will do with this configuration: refuse its following nest at its
own plan review, or -- for a source with no launch route -- refuse the
configuration itself, in the words the launch will use. The same sentence is
written into the configuration file as a comment and printed once on stderr,
and the emitted `note` names the sources that do integrate a moving nest
today.

## Where the centre comes from

One function decides, with a stated fallback chain, and `seed.method` on the
document says which rung answered:

1. `--point LAT,LON` -- authoritative, and the only rung that needs no fields.
2. `--seed-fields NPZ` -- canonical arrays from the selected source analysis.
   Tried in order: the declared sea-level pressure minimum, then the 850 hPa
   cyclonic relative-vorticity maximum, then a 300/500 hPa warm anomaly.
   Which of those a source can offer is its registry row's own statement of
   what its route serves.
3. `--advisory-position LAT,LON` -- bounds the field search (`--seed-radius-km`,
   default 500 km) and is the final fallback when no diagnostic answers.

`seed.messages` records every rung that declined and why. A candidate centre
is a centre, not a tropical-cyclone classification or an intensity analysis.

The NPZ is loaded with object deserialization disabled and must carry three
identity scalars -- `source`, `cycle` and `member` (empty for a deterministic
source) -- matching the selection, plus matching 2-D `latitude` and `longitude`
degree arrays. Optional canonical fields are `mean_sea_level_pressure` (Pa),
`eastward_wind` and `northward_wind` (m/s) and `air_temperature` (K), with
upper-air arrays indexed `[level, y, x]`.

An upper-air array needs its level coordinate in the same file, or no plane
can be interpolated from it and the vorticity and warm-core rungs decline:
either `pressure_levels_pa`, a 1-D array of pascals as long as the array's
first axis, or a full `air_pressure` field of the same `[level, y, x]` shape.
Levels are interpolated in log-pressure and never extrapolated, so a target
level outside what the file brackets declines rather than guessing. A file
carrying only `mean_sea_level_pressure` reaches the first rung and no other.

## The one refusal that is about the storm

A centre outside the selected source's grid is refused, with the position and
the sources that do cover it named. That is a fact about the request, not a
budget: a regional source cannot initialize a cyclone it does not contain, and
the way out is a covering source.

Everything else this door refuses is the ordinary plan review -- an
unpublished cycle hour, a lead or a duration past the cycle's own horizon, a
lead the producer's ladder skips, a member the route's grammar does not have,
a budget under the nest floor, a layout the card cannot hold. All of them fire
while the configuration is still on the screen, before anything is fetched.

## The track the following nest writes

The authored configuration gives the following nest its own
`storm-track.d02.csv`. The track ends, with a stated reason, when the tracked
extremum reaches the parent-domain boundary and an enclosed centre is no
longer resolved -- for every tracked field, at whichever end of that field is
the centre. Missing signal alone is a gap in the record, not a termination,
and a termination ends the diagnostic stream without ending the forecast.
