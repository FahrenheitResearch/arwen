## Recipes: the best run of an event for each card

An event page's first button runs the best simulation of that event that
fits the reader's card. The recipe store says what that simulation is for
each card size: the domains and nests, the start, the length, the source
and the physics, each one checked by the engine for that card's memory.

### Where recipes live

A recipe is a record in a `gpuwm.wiki.v1` document, in a list named
`recipes`. The seed ships one document per seed event
(`gpuwm/gui/seed/recipes/<event id>.json`); any other document under
`<runs root>/wiki/` may carry a `recipes` list too. A recipe attaches to
the event its `event` field names. It never replaces the event record
itself, and a document read later replaces an earlier recipe for the same
event, the same way a later event record replaces the seed's.

The old `recipe` field on an event stays as it is. When an event has a
recipe in the store, the page uses the card list and ignores the old field.

### Card classes

The card list is a table, one row per card size. The seed rows are 8, 12,
16, 24 and 32 GB; a 48 or 96 GB row is another row, not a code path. The
page picks the row with the largest `card_gb` that does not exceed the
memory of the card in this computer (`GET /api/system`) and whose
`disk_gib` fits the free disk that holds the runs, offers it as the first
button (`POST /api/wiki/simulate` with the event and the card size), and
puts the other sizes one click away. A reader with no card the
server can see picks a size by hand.

### A recipe

| field | meaning |
|---|---|
| `event` | the event id this recipe belongs to |
| `title`, `type` | copied from the event, for reading the file on its own |
| `key_hours` | `start`, `end` (ISO UTC) and `text`: the hours the simulation exists to capture; a cyclone's also has `key_time` and `key_is` (`landfall` or `peak`) |
| `ideal` | `rung`, `finest_km`, `text`: the layout this event gets when memory is no limit |
| `start`, `length_h`, `source` | the event's window; a card row that starts later says so in its own `start` and `length_h` |
| `cards` | one row per card size, below |
| `cite` | the rule that designed the recipe and the event's track or path record; for a cyclone also the landfall rule and the best-track row of the key landfall |

### A card row

| field | meaning |
|---|---|
| `card_gb` | the card size this row is for |
| `fits` | false when no layout fits; the row then carries `why` and `tried` only |
| `rung` | the name of the layout chosen, from the rule's ladder, best first |
| `domains` | the fitted grids: `grid_id`, `parent_id`, `dx_km`, `nx`, `ny`, `nz`, `width_km`, `height_km`, `parent_grid_ratio`, `following`, `cu_physics` (the cumulus switch the engine generated for that grid), `centre` |
| `start`, `length_h` | when this row's run starts (ISO UTC) and how many hours it runs; a late-start rung starts after the recipe's `start` |
| `source`, `source_why` | the analysis the run starts from, and why that one |
| `physics` | `profile` (the suite the engine bound) and `why` (the suite and where its cumulus scheme is on and off) |
| `fitted_grid` | the fitted grids in words |
| `memory` | `need_gib` and `budget_gib`: what the engine priced the run at for this card |
| `disk_gib`, `disk_basis` | the disk the run needs: its download, its preparation, its history files at the row's output intervals, the one checkpoint set run-plan keeps (two while a new one is written) and its rendered pictures, as the engine projects them (`gpuwm/disk_budget.py` and `gpuwm/download_budget.py`, also the `disk` block of `gpuwm run-plan --resolve`) from bytes per cell and picture bytes per frame measured on a real run (`tools/wiki_seed/runs/bytes-per-cell/sizes.json`) and from the object sizes and preparation bytes per cell measured on real downloads and preparations (`gpuwm/data/download-bytes.v1.json`); the basis names each part. The page compares it with the free disk before the first press, and run-plan refuses before its download a run whose projection exceeds the free disk |
| `download` | the row's download as the engine prices it: `bytes` left on disk, `transfer_bytes` moved over the network, `objects`, `leads` (forcing times), `source`, `mode`, and the `preparation_bytes` and `chain` of its preparation; the time estimate reads it |
| `disk_budget_gib` | the most disk a row of this card size may need: 40, 50, 60, 80 and 100 GiB for 8 to 32 GB; `disk_gib` is always at or below it |
| `output` | `history_interval_s` (outer grid), `nest_history_interval_s` and `keep_checkpoints`: the densest output the disk budget allows, also in `intent` |
| `key_cover_h` | a cyclone's 1 km row only: the hours before and after the key time that the best-track centre stays at least 60 km inside the 1 km grid's edge |
| `rings_km` | an ERA5 row with nests: for each nested grid, the km from its edge to its parent's edge on the thinnest side, read from the fitted grids |
| `est_minutes`, `est_total_minutes`, `est_basis`, `est_kind` | `est_minutes` is the forecast; `est_total_minutes` is the wait from the press to the last picture (download, preparation, forecast and pictures), which the page leads with, as "about 45 min to all pictures, 35 of it the forecast"; `est_kind` is `measured` when this row ran, else `estimated` from the measured runs of its kind of storm, with the download at the throughput measured on real downloads of its source, the preparation at the rate measured on real preparations of its chain (`tools/wiki_seed/runs/lead-in/runs.json`) and the pictures scaled by frames and grid points from the run that drew them (`tools/wiki_seed/estimate_times.py`); the basis names the card and the runs |
| `why` | what this layout captures, in plain words |
| `what_is_given_up` | what this card loses against `ideal`, or that it loses nothing |
| `finest_km` | the finest grid spacing in the row |
| `door` | how the engine builds it: `domain` or `cyclone-setup` |
| `intent`, `box` | for `domain`: the run plan's `config.intent`, and the box (`lat`, `lon`, `width_km`, `height_km`) written to `region.geojson` as its polygon |
| `args` | for `cyclone-setup`: `source`, `cycle`, `advisory_position`, `hours`, `card`, `isftcflx`, `history_interval`, `nest_history_interval`, and `nest_budget_gib` when the following nest is grown to the card; each key is the flag of the same name |
| `recipe` | the row again in the New forecast screen's words, see below |
| `better_rungs_refused` | each better layout the engine refused on this card: `rung`, `need_gib`, `reason` |
| `cite` | the fit record for this row, the rule, the event's track or path record, and a measured run when there is one |

`recipe` is a faithful copy of the row, not a simplification of it:
`source`, `cycle`, `hours`, `card`, the box (`lat`, `lon`, `width_km`,
`height_km`, around which the innermost grid is fitted), `dx_km` (the
outer grid's spacing), and, when the row has nests, `chain` and
`buffer_km` exactly as in `intent`; an ERA5 row adds `era5_provider`. A
`cyclone-setup` row has `following: true` and `cyclone_setup` (the `args`).
A screen that starts a row must pass `chain` and `buffer_km` through, or
start the row from `intent` and `box`; drawing only the box at `dx_km`
runs one grid, which is not the recipe.

### How a row reaches the engine

A `domain` row is a run plan whose `config.intent` is the row's `intent`
with `polygon` set to the box. The page writes the box to the run folder's
`region.geojson`, as New forecast does now, and runs `gpuwm run-plan`.
ERA5 runs on the `experiment` route, and the run downloads its own ERA5
from the keyless store; every other source runs on `prepared`.

Every row's intent carries its output intervals (`history_interval_s`,
`nest_history_interval_s`), the densest that keep the run within the card's
disk budget: a tornado's nests every 15, 20 or 30 minutes, a cyclone's
every one, two or three hours, since a supercell changes in minutes and a
cyclone run lasts two to five days. run-plan and `gpuwm go` keep one checkpoint
set by default (`run_options.keep_checkpoints`, `--keep-checkpoints` on
go), which is what the rows' disk figures assume; 0 keeps every hourly
set, which a later branch or downscale from an earlier checkpoint needs.

Every cyclone row sets `isftcflx: 1`, the surface layer's tropical cyclone
option over the sea (Donelan drag, which levels off in strong wind, with a
constant heat and moisture roughness), and says why in `physics.why`.

`buffer_km` is read by the engine from the box itself at every level, not
from the next inner grid. A 12 km grid meant to reach 500 km beyond a 3 km
grid whose own buffer is 300 is written `800,300,0`. The designer reads the
ring back from the fitted grids and refuses an ERA5 row whose 12 km grid
reaches less than 400 km beyond the grid inside it.

The intent never names a physics suite. Naming one makes the engine emit
that suite verbatim, cumulus scheme included, even on a 3 km grid; left
out, the engine binds the same suite (its default for the source) and
turns the cumulus scheme off on every grid finer than 4 km. The designer
reads the generated configuration back and refuses any row with a cumulus
scheme on a grid finer than 4 km.

A `cyclone-setup` row is `gpuwm cyclone-setup` with the row's `args` and
`--out <run folder>/cyclone.toml`; the run plan then names that file as
`config.path` on the `experiment` route. The nest follows the storm's
850 hPa circulation. The designer uses this row only when the storm's track
stays inside the fixed 12 km parent for the whole run, and says so in the
row's fit record.

### The rules

Three rules design every seed recipe. Each is a `rule` source in the
document, with its method in plain words.

- `rule-recipe-best-tornado`: start at the analysis at least 6 hours
  before the tornado and run to 3 hours after it ends. A 1 km grid covers
  the path and 120 km back along it, where the parent storm forms, inside a
  3 km grid that reaches 180 to 400 km beyond it. From HRRR, itself a 3 km
  analysis, the 3 km grid is the outer one. From ERA5 (about 31 km) a 12 km
  grid reaching 500 km beyond the 3 km one carries the large-scale flow, so
  the 3 km grid is not driven straight from 31 km boundaries. The ladder
  shrinks the 1 km grid, then drops it for a 3 km grid 1100, 800 or 650 km
  wide (inside the 12 km one for ERA5), then 4 km. No card gets a grid
  finer than 1 km: a fixed box cannot be sure to hold the simulated storm,
  and it multiplies the run time.
- `rule-recipe-landfall`: a cyclone's key landfall is the first best-track
  row, from six hours before the highest wind on, whose centre is within
  25 km of land (IBTrACS `DIST2LAND`), or that a US agency marks as a
  landfall. The table is built from the same IBTrACS rows the seed was
  built from, and each recipe cites the row it used.
- `rule-recipe-best-tc`: the key time is the key landfall when it comes
  after the highest wind, else the highest wind. Start 36 hours before the
  highest wind; stop 18 hours after it or 12 hours after the key landfall,
  whichever is later, at most 120 hours. A 1 km grid is the best-track path
  from 12 hours before to 12 hours after the key time, hour by hour, with
  240, 180, 130, 90 or 60 km of grid around it, and a rung is refused
  unless the centre stays at least 60 km inside the grid's edge for all of
  those hours (the eyewall, not only the centre, has to be on it); the row
  records the hours it holds in `key_cover_h`. The ladder, best first: the
  3 km grid holding the whole track with the three widest 1 km grids; the
  3 km grid holding only the hours from 24 hours before the key time, the
  12 km grid carrying the earlier ones, with the same 1 km grids; both
  again with the two narrowest; 3 km over the whole track with no 1 km
  grid; a following 3 km nest while the storm stays inside its parent; 3 km
  over the key hours only; a later start, 30 then 18 hours before the key
  time, with 3 km over the rest of the track; a run from 18 hours before to
  6 hours after the key time with a 3 km grid over just those hours (what a
  small card gets instead of 12 km over the whole storm); 12 km alone. The
  12 km grid reaches 700 km beyond the 3 km grid (500 for a later start)
  and 400 km around any hour the 3 km grid does not hold. A later start
  gives up spin-up time, and the peak itself when the key time is a later
  landfall; a row with no grid finer than 12 km says what that costs and
  what the smallest 3 km layout would have needed.

Each card takes the first rung the engine fits on it with at least 5% of
its budget to spare (`HEADROOM_FRACTION` in the designer), so a desktop or
browser on the same card does not turn the first press into a refusal, and
whose projected disk fits the card's disk budget (`DISK_BUDGET_GIB`) at one
of the allowed output intervals, so the first press does not fill a disk.

The source follows the date and the place: HRRR for United States tornadoes
from October 2014 on, since its archive is complete and its analysis already
holds the radar's storms; ERA5 for everything else, read from Google's
public copy (`era5_provider: arco`) so no key is needed. ERA5 is one
consistent analysis for every date, where an archived GFS exists only for
recent years.

A new kind of event is a new rule and new rows, not a new code path.

### Checking and rebuilding

Every row is checked by the engine, on the CPU, for a declared card of its
size: `gpuwm run-plan --resolve` for a `domain` row and `gpuwm
cyclone-setup` for a following nest. The fit record (`computed-fit-<event
id>-<size>gb`) keeps the fitted grid, the priced memory and the cumulus
switch per grid. A row the engine refuses is not written; the next layout
down the ladder is tried instead.

`tools/wiki_seed/measured_from_runs.py` turns kept run records into
`measured.json`. `tools/wiki_seed/estimate_times.py` then gives every row
that has not run a time scaled from the measured runs of its kind of
storm (grid points times time steps times hours), marked `est_kind:
estimated`, and cites `rule-recipe-time-estimate`.
`tools/wiki_seed/landfall_table.py` builds `landfall.json` from the IBTrACS
file; `tools/wiki_seed/design_recipes.py` designs and checks every event;
`tools/wiki_seed/build_recipes.py` writes the documents. The designer
reads `RECIPE_ENGINE_TREE` (the tree whose engine checks the rows),
`RECIPE_SEED_TREE` (the tree whose seed lists the events), `RECIPE_PYTHON`
(the interpreter that engine is installed in) and `RECIPE_WORK` (where the
fit cache and results go); the builder also reads `RECIPE_OUT`. Delete the
fit cache after any change to the engine's sizing, or the old answers are
reused.

A measured run lives in `measured.json` beside the tools, keyed
`<event id>/<card gb>`, and names the `rung` and `start` it ran. The
builder applies it only while the row still has that rung and start, so a
redesign never carries a stale time. The run's log and a small summary
(exit code, time per stage, field extremes per grid) stay under `runs/`
after its data is deleted.

`tests/test_gui_wiki.py` holds every recipe to these rules: every
`cite` id is a source in some document, every fitted row has `need_gib` at
or below `budget_gib` and `disk_gib` at or below `disk_budget_gib`, no grid
finer than 4 km has `cu_physics` on, every ERA5 row's 12 km grid reaches at
least 400 km beyond the grid inside it, every cyclone 1 km row holds the
centre for the hours its rule promises, the rows for one event never get
coarser as `card_gb` grows, and no row's text carries a machine path.
