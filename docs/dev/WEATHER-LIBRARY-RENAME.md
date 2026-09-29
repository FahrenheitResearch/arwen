# The Weather Library: the page `gpuwm gui` serves, and its old name

The page `gpuwm gui` serves (storm pages, New forecast, My forecasts, Files,
Machines, Settings and the assistant panel) was called the storm wiki while it
was built. It is named the Weather Library everywhere a user sees it. ArWen
Desktop keeps its name and is not part of this.

`tools/dev/rename_weather_library.py` makes the rename, so it can be run again
on a tree other branches have changed since:

```
python tools/dev/rename_weather_library.py --check     what it would change; writes nothing
python tools/dev/rename_weather_library.py             make the change
python tools/dev/rename_weather_library.py --scan      old names left where a user reads them
python tools/dev/rename_weather_library.py --notes cut_inputs/release-notes-2.8.0.md
```

It prints each file and each place it changed. A second run changes nothing:
every file it changes is run through its rules again before anything is
written and must come out the same, and must still parse (JSON, Python, and
for a page script no name declared twice at its top level, and `node --check`
when Node is installed).

Sentences change after the rename, so an exact edit does not pin its lines:

- When an edit's result is there, or every mark of its result is (`made`: the
  lines it inserts, such as `MOVED_ROUTES` and `forwardedRoute` in `router.js`
  or `SECTIONS` in `api.py`), the edit is done. Lines it inserted and someone
  changed since are never inserted a second time.
- When neither its old text nor its result is there, its lines were reworded.
  It counts as made, and the run says so, when after the other rules the file
  holds every mark of its result and no old name: nothing the scan finds, and
  nothing its own `stale` patterns find (a tab title or brand that still says
  ArWen, a route string that is still `wiki`).
- Otherwise it refuses its file, and then nothing at all is written; the rule
  is rewritten for the file as it is now.

The word rules never read docstrings or comments outside the page's scripts;
the route rule renames the old API paths in the GUI package's docstrings, and
one exact edit renames the main page row of the API's endpoint table. It never
edits `tests/`, identifiers, the file formats and file names below, or a
released CHANGELOG section.

`tests/test_weather_library_name.py` holds the tree to it: no user-facing
string names the page by its old name outside the allow-list (`KEPT` in the
tool), the tool plans no change and no refusal on the tree, the old addresses
answer, and the help names the Weather Library.

## Decisions

Rename: the user reads the new name. Add: a place that named no page now
names the Weather Library. Redirect: the old address keeps working and
forwards to the new one. Keep: the old spelling stays, for the reason given.

### The page

| # | Where | Old | New | Decision |
|---|---|---|---|---|
| 1 | Header brand and every tab title (`screens.json` `app.name`) | ArWen | Weather Library | rename |
| 2 | Tab title before the scripts run (`index.html`) | ArWen | Weather Library | rename |
| 3 | Line a browser without scripts shows (`index.html`) | ArWen's page needs JavaScript. | The Weather Library needs JavaScript. | rename |
| 4 | Sidebar group (`wiki.json` `nav.wiki_group`) | Storm wiki | Weather Library | rename |
| 5 | Main page heading (`wiki.json` `main.title`) | Storm wiki | Weather Library | rename |
| 6 | Path root and the event, kind and place tab titles (`wiki.json` `name`) | Storm wiki | Weather Library | rename |
| 7 | Ctrl K search group (`screens.json` `shell.palette_wiki`) | Storm wiki | Weather Library | rename |
| 8 | Article line under an event's title (`wiki.json` `article.from`) | From the storm wiki | From the Weather Library | rename |
| 9 | Kind page list (`wiki.json` `kind.events`) | Every {kind} in the wiki | Every {kind} in the Weather Library | rename |
| 10 | Place page lede (`wiki.json` `place.lede_events`, `place.lede_one`) | events in the wiki | events in the Weather Library | rename |
| 11 | Run article with no event (`wiki.json` `run.events_none`) | None of the wiki's events | None of the Weather Library's events | rename |
| 12 | New forecast opened from an event (`wiki.json` `create.from_event`) | From the wiki: {title} | From the Weather Library: {title} | rename |
| 13 | My forecasts button (`screens.json` `runs.from_wiki`) | Pick an event from the wiki | Pick an event from the Weather Library | rename |
| 14 | Find an event, no match (`screens.json` `create.find_none`) | No event in the wiki matches. | No event in the Weather Library matches. | rename |
| 15 | Ctrl K assistant line (`screens.json` `shell.open_assistant_sub`) | Ask about the wiki, a forecast | Ask about a storm, a forecast | rename |
| 16 | The copy file's note for editors (`wiki.json` `about`) | The storm wiki's words | The Weather Library's words | rename |
| 17 | Assistant panel intro (`assistant.json` `intro`) | a storm in the wiki | a storm in the Weather Library | rename |
| 18 | Where a reply came from (`assistant.json` `from_wiki`) | the wiki | the Weather Library | rename |
| 19 | Assistant off line (`assistant.json` `off_line`) | the wiki, your forecasts | the storm pages, your forecasts | rename |
| 20 | Page not found (`api.py`) | No such wiki page. Go to the wiki's main page. | No such Weather Library page. Go to the Weather Library's main page. | rename |
| 21 | Too many live views, the reply's fix (`server.py`) | Close a few ArWen tabs and reload. | Close a few Weather Library tabs and reload. | rename |
| 22 | Seed sources' titles and methods (`wiki-seed.json`, `tools/wiki_seed/build_seed.py`) | chosen for the wiki, when the wiki was built | the Weather Library | rename |

### Addresses

| # | Where | Old | New | Decision |
|---|---|---|---|---|
| 23 | Main page's address | `#/wiki`, `#/wiki/...` | `#/library` | redirect: the shell forwards before drawing and the address bar shows the new one (`router.js` `MOVED_ROUTES`) |
| 24 | The page's API: main, event, kind, place, places, run, search, changes, recipe, simulate | `/api/wiki/...` | `/api/library/...` | redirect: both paths reach the same handler (`api.py` `Api.SECTIONS`) |
| 25 | Every other page address (`#/browse`, `#/places`, `#/changes`, `#/event/ID`, `#/kind/ID`, `#/place/ID`, `#/search`, `#/run/ID`) | | | keep: none carries the old name |

### The terminal

| # | Where | Old | New | Decision |
|---|---|---|---|---|
| 26 | The command | `gpuwm gui` | `gpuwm gui` | keep: it names what it opens by kind, not by the old name, and every doc, note and script that starts it keeps working; no alias is needed |
| 27 | `gpuwm gui` help | open ArWen in your browser | open the Weather Library in your browser | rename |
| 28 | The line `gpuwm gui` prints | ArWen is ready. | The Weather Library is ready. | rename |
| 29 | Port taken | close the other ArWen page server | close the other Weather Library page server | rename |
| 30 | `gpuwm --help` first-use guide | | `gpuwm gui` Open the Weather Library in your browser | add: the guide lists the page under Create and launch, written in `gpuwm/cli_help.py` by hand (the tool does not add it). The guide keeps its 30 lines, the most `tests/test_cli_first_use.py` allows, because `gpuwm resume` moves onto one line |
| 31 | `gpuwm assistant` help | the same assistant as gpuwm gui's panel | the same assistant as the Weather Library's panel | rename |
| 32 | `gpuwm gui` flags and the default port 8766 | | | keep: none carries the old name; there is no wiki subcommand or flag |

### The assistant

| # | Where | Old | New | Decision |
|---|---|---|---|---|
| 33 | Its instructions (`agent.py` `SYSTEM`) | search the wiki | search the Weather Library | rename: the model repeats these words to the user |
| 34 | Tool descriptions (`search_wiki`, `read_event`, `read_run`) | the storm wiki's events, One storm wiki event, its wiki article | the Weather Library's events, One Weather Library event, its Weather Library article | rename |
| 35 | The page `open_page` may open | `wiki` | `library` | rename; an old value reaching the page forwards (row 23) |
| 36 | Tool names `search_wiki`, `read_event` | | | keep: names the model calls, never words a user reads |

### Docs and release text

| # | Where | Old | New | Decision |
|---|---|---|---|---|
| 37 | `docs/public/GUI.md` title and text | ArWen in your browser, ArWen's page, storm wiki | the Weather Library, its main page, the storm pages; the sidebar's first group by its heading, the **Weather Library** group of storm pages | rename |
| 38 | `README.md` 2.8.0 paragraph and limits | ArWen in your browser, as a preview beside the desktop app, a storm wiki, the web GUI is a preview | the Weather Library, ArWen's graphical interface in your browser; ArWen Desktop keeps shipping and new features arrive in the Weather Library | rename |
| 39 | `CHANGELOG.md` 2.8.0 section | ArWen in your browser, a preview beside the desktop app, storm wiki, wiki event, wiki layout | the Weather Library | rename |
| 40 | `CHANGELOG.md` sections of released versions | | | keep: the record of what shipped |
| 41 | `NOTICE` seed credit | The storm wiki's seed pages, each wiki page | The Weather Library's seed pages, each Weather Library page | rename |
| 42 | `docs/dev/WIKI.md`, `WIKI-RECIPES.md`, `GUI-API.md`, `RELEASE-CUT.md` | storm wiki, the wiki, wiki article, the web GUI | the Weather Library | rename |
| 43 | Endpoints in `docs/dev` and the `api.py` table | `/api/wiki/...`, the wiki's main page | `/api/library/...`, the Weather Library's main page, with a line that the old paths answer | rename |
| 44 | 2.8.0 release notes (a cut input, outside the tree) | ArWen in your browser, a preview beside the desktop app, storm wiki, the web GUI | the Weather Library | rename at the cut with `--notes` |

### Files, formats and internal names

| # | Where | Decision |
|---|---|---|
| 45 | `<forecasts folder>/wiki/`, the page store's folder | keep: the event atlas writes there and a person's own page documents live there; moving it needs a migration that merges two stores, and a stale copy of the seed in the old folder would override the new seed's records |
| 46 | `wiki-run.json` in a run folder | keep: runs started from an event page carry it, and saved runs are read by that name |
| 47 | `gpuwm.wiki.v1`, the store's document schema | keep: a file format the atlas and saved documents are written in |
| 48 | `gpuwm/gui/seed/wiki-seed.json` and `gpuwm/gui/seed/recipes/` | keep: package data, mirrored into the store folder |
| 49 | `docs/public/GUI.md`, `docs/dev/WIKI.md`, `docs/dev/WIKI-RECIPES.md` file names | keep: linked from code and docs; the titles inside are renamed |
| 50 | `gpuwm/gui/wiki.py`, `Wiki`, `WikiMixin`, `Store`, the handler section `wiki` | keep: internal names |
| 51 | `wikipages.js`, `wikikit.js`, `wikimap.js` and the page's CSS classes | keep: internal names |
| 52 | `gpuwm/gui/copy/wiki.json` and its keys (`words.wiki.*`, `palette_wiki`, `from_wiki`, `wiki_group`) | keep: the file's stem is the key the page reads, and other branches add keys under it |
| 53 | `tools/wiki_seed/`, its user agent `gpuwm-wiki-seed` | keep: developer tooling |
| 54 | `tests/`: `test_gui_wiki.py`, fixtures of saved runs, the tests of `/api/wiki` | keep: the tool never edits tests; the old paths they call still answer |
| 55 | `gpuwm domain`, `gpuwm import-namelist`, `gpuwm go`, `events.jsonl` `model_progress` | keep: the hosted app depends on them; none carries the old name |
| 56 | `X-ArWen-Token`, the cookie, the server thread `arwen-gui-server` | keep: internal names |
| 57 | ArWen where it names the engine: "the graphical door of ArWen 2.8", "the one folder ArWen writes in there", "the assistant inside ArWen" | keep: the engine keeps its name |
| 58 | ArWen Desktop | keep: not part of this rename |

Counts: 36 rename, 1 add, 2 redirect, 19 keep (58 items).
