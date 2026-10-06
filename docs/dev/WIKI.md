# The Weather Library's page store (`gpuwm.wiki.v1`)

`gpuwm gui` shows the Weather Library: event pages, pages for each kind of storm,
place pages, and every run in the forecasts folder as an article. The pages
are data, never hand-written HTML. The event atlas fills the Weather Library by writing
documents in this format; the page draws whatever the store holds.

## Where the store lives

`<runs root>/wiki/**/*.json`. Every file whose `schema` is `gpuwm.wiki.v1`
is read and merged, reloaded when a file changes.

- `wiki/seed.json` is the package's seed (`gpuwm/gui/seed/wiki-seed.json`,
  built by `tools/wiki_seed/build_seed.py`). The server puts it there at
  start and replaces it when the package's copy changes. It loads first.
- Any other document (the atlas writes `wiki/atlas/*.json`) loads after it,
  and a record with the same `id` replaces the seed's.

Runs are never written to. A run is linked to an event when its
`region.geojson` box holds a point of the event while its hours (from the
run's events, else its `plan.json`) overlap the event's window
(`gpuwm/gui/wiki.py: run_covers`).

## The rule every record keeps

Nothing is made up. Every fact names its sources by id (`cite`, or a key
ending in `_cite`), and every id must be in some document's `sources`. The
page draws a footnote for each and opens the record behind it. Prose is a
list of parts: `{"text": ...}` joining words with no numbers in them, and
`{"fact": id}` naming a fact of the same record. A part naming a missing
fact is dropped. `summary.generated` marks the prose as written by code.
`tests/test_gui_wiki.py` checks all of this on the seed.

## A document

```json
{
  "schema": "gpuwm.wiki.v1",
  "origin": "atlas",
  "built": "2026-09-25",
  "sources": {"<id>": {...}},
  "phenomena": [{...}],
  "places": [{...}],
  "events": [{...}]
}
```

### sources

| kind | fields |
|---|---|
| `dataset` | `title`, `publisher`, `url`, `licence`, `retrieved`, optional `citation` |
| `record` | a row of a dataset: `title`, `dataset` (a dataset source id), `url`, `licence`, `row` (the row's own fields) |
| `computed` | a statistic our code computed: `title`, `method` (plain words), `code` (file: function), `inputs` (source ids), `result` |
| `rule` | a rule our code applies (a recipe, a selection): `title`, `method`, `code` |

The page adds two kinds for run articles: `run-file` (a file of the run
folder, opened through the run-file endpoint) and `run-tree` (the run's
render folders).

### events

| field | meaning |
|---|---|
| `id`, `type`, `title` | `type` is a phenomenon id |
| `seed` | true for seed pages; the page says so |
| `added` | date the record entered the store (Recent changes) |
| `when` | `start`, `end`, `peak`, ISO UTC |
| `where` | `lat`, `lon`, `label`: the point maps and lists use |
| `region`, `decade`, `season` | filters; `season` is `dec-feb`, `mar-may`, `jun-aug` or `sep-nov` |
| `places` | place ids |
| `rarity` | `count` (events at least as strong in that place and season), `of`, `place`, `text`, `cite` |
| `facts` | `[{id, label, text, cite, infobox, value?, unit?}]` |
| `summary` | `{parts, generated}` |
| `geometry` | `track` (`[lon, lat, wind, time, class]` points) and `track_cite`, or `path` (`[[lon, lat], [lon, lat]]`) and `path_cite` |
| `observed` | `[{label, url, cite}]`: links to the observed records |
| `era5` | `statistics` (`{name: {text, cite}}`, filled by the atlas) and `quality` (`{flag, text, cite}`) |
| `recipe` | `source`, `cycle` (`YYYY-MM-DDTHH`), `hours`, `lat`, `lon`, `width_km`, `height_km`, `dx_km`, `cite`: what Simulate this loads into New forecast |

### phenomena and places

A phenomenon: `id`, `title`, `plural`, `what` (`{quote, cite}`), `detect`
(`{text, cite}`). A place: `id`, `title`, `kind` (`basin`, `country`,
`state`), optional `parent`, `cite`, and `counts`
(`{<phenomenon>: {count, text, cite}}`).

## Endpoints

`GET /api/library`, `/api/library/event/ID`, `/api/library/kind/ID`,
`/api/library/place/ID`, `/api/library/places`, `/api/library/run/RUN`,
`/api/library/search?q=&type=&region=&decade=&season=&place=&seed=&sort=`,
`/api/library/changes`, `/api/library/recipe/ID`. All read-only. The same paths
with `wiki` in place of `library` answer too: that was their name before the
page was named the Weather Library.

The search box takes words and matches them against titles, places and
facts. A plain-language answer from the assistant can take the same box
later; its answer must cite the same source ids.
