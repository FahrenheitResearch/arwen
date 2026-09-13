# The cyclone setup document

`gpuwm cyclone-setup` prints one JSON document on stdout and nothing else;
everything it says while working goes to stderr. This page is that document:
what each kind carries, and which fields a client may rely on. The schema
string is `arwen.cyclone-setup.v2` on every kind.

The door authors a 12 km parent with a 3 km following nest centred on a
cyclone. It never starts a forecast: `forecast_started` is `false` on every
document it emits.

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

A member id is a string from `members`, never an index. The ordinal that
appears as `map_request.member` is a different thing (see below) and the two
are not interchangeable.

### `map` -- the selection map

Emitted by `--latest-map`. `map_request` names the exact analysis to draw:
`source`, `date`, `hour`, `forecast_hour` (always 0), `member` (a numeric
ordinal for the map product, 0 for a deterministic source), `product` and
`bounds`. `bounds` is the source's own coverage envelope clipped to the
drawable band, so a regional source does not offer a global frame to click in.
The document's top-level `member` is the resolved member id string.

### `configuration` -- the authored setup

The requested layout was admitted. `config_text` is the configuration TOML;
`domains` lists the authored grids; `follow` is the vortex-lock preset the
following nest carries; `memory` carries the peak envelope, the budget and the
basis it was sized from; `streaming` says which road each domain takes.
`fitting.changed` is `false`.

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

## Fields a client consumes

These are the fields the desktop reads, named exactly:

* on `sources` rows: `source`, `label`, `members`, `default_member`,
  `cycle_hours`, `coverage_envelope`, `forcing_interval_seconds`,
  `follow_statics`;
* on a `configuration` or a `proposal`: `follow_statics` at the top level,
  the block described under "The moving nest and the chain that feeds it";
* at the top level of a `map`, `configuration` or `proposal`: `member` -- the
  resolved member ID string, or `null` for a deterministic source;
* on a `map`: `map_request.member` -- a numeric ordinal for the map product.
  It is not the member id and must not be sent back as one.

`sources`, the top-level `member`, `forcing_interval_seconds`, `seed` and
`follow_statics` are
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
unpublished cycle hour, a duration past the cycle's own horizon, a member the
route's grammar does not have, a layout the card cannot hold. All of them fire
while the configuration is still on the screen, before anything is fetched.

## The track the following nest writes

The authored configuration gives the following nest its own
`storm-track.d02.csv`. The track ends, with a stated reason, when the tracked
extremum reaches the parent-domain boundary and an enclosed centre is no
longer resolved -- for every tracked field, at whichever end of that field is
the centre. Missing signal alone is a gap in the record, not a termination,
and a termination ends the diagnostic stream without ending the forecast.
