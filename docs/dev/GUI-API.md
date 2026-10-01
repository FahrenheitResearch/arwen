# gpuwm gui: the page's API

The endpoints the page's screens call, one section per area. Every POST
takes `"dry_run": true`, which checks the request and answers with what it
would do, with nothing run. Every error answers `{"ok": false, "message",
"fix"}`. The token rules are in `gpuwm/gui/auth.py`.

A machine path never reaches the page: the words "(a folder named in the page
server's log)" stand in for it, and the server writes the text it came from to
its own log (the terminal running `gpuwm gui`), once per text. A failed run's
`status.end` keeps the first line of the engine's `message` and the first
paragraph of its `remedy`. The Map shows both, and the Article lists them once,
as the facts "Why it stopped" and "What to do"; a stopped run's Article shows
its `message` under its title.

### Starting on this computer needs the geography tree

`POST /api/create/start` (on this computer), `POST /api/wiki/simulate` and
`POST /api/runs/RUN/start` check the geography tree the run will build its
static fields from (`$GPUWM_CASE_DATA_ROOT/WPS_GEOG`, or the tree the plan
names in `run_options.geog_root` or `config.intent.geog_root`) with the check
`gpuwm doctor` and `gpuwm go` make, before anything is written. Missing, the
answer is HTTP 409 with a `message` saying the computer has no geography data
yet; incomplete, the `message` says how many datasets are absent. Either way
the `fix` names `gpuwm fetch-geog --datasets wrf` and its download size, no run
folder is made, and a Ready run stays Ready. Dry runs answer the same.

## Physics composer

One engine answer to "what physics can I run and what does each choice
mean". The page and an assistant read the same two documents the terminal
prints; the GUI never holds a rule table of its own.

| Endpoint | Engine command |
| --- | --- |
| `GET /api/physics[?source=ID]` | `gpuwm physics-catalog --json [--source ID]` |
| `POST /api/physics/check` | `gpuwm physics-catalog --json --check JSON` |
| `POST /api/create/start` and `/api/create/fit` with `"preset"` | the preset's suite becomes `config.intent.physics_profile` in plan.json |
| (terminal) a mix into an experiment file | `gpuwm physics-catalog --check JSON --into EXPERIMENT.toml --out MIX.toml` |

### GET /api/physics

`gpuwm.physics-catalog.v1`:

- `families[]`: `id`, `name`, `what` (one plain line), `switch_keys` (the
  namelist switches it sets), `streams` (radiation only: longwave and
  shortwave), `schemes[]`.
- `schemes[]`: every option the physics registry has, runnable or not.
  - `id`, `label`, `description` (one plain line), `is_default`.
  - `choice`: the id to send back in a check. Radiation's `rte-rrtmgp`
    has two engines, listed in `variants[]` (`rte-rrtmgp:rte-rrtmgp`,
    `rte-rrtmgp:rrtmg_legacy`).
  - `implemented`, `reachability`, `blocker`: a scheme that cannot run
    says why in the registry's words. Grey it out and show `blocker`.
  - `maturity`, `scientific_evidence`, `notes[]` (the registry's warnings;
    long, show on demand).
  - `switches` (namelist values), `streams` (radiation: longwave and
    shortwave switch numbers).
  - `requires` and `requires_reasons`: the pairings the registry states
    (MYJ needs the Eta surface layer). Show them before a choice is made.
  - `cost`: `relative` to the default suite per model step, `measured`
    (true only when both rates are measurements), `words`, `rung`. Most
    schemes say "Unmeasured for this scheme. Planned at the default's
    rate.": the step rates only separate the MYNN and Noah-MP rung from
    the rest. Show `words`, never invent a number.
  - `resolution`: `quiet_at_dx_km` and `advisories[]` (`dx_km`,
    `headline`): where the wizard's own grid-spacing advice speaks about
    this family, sampled at 0.1 to 25 km.
  - `cards`: per card size (`12gb` .. `32gb`), the largest grid that stays
    resident at 49 levels (`columns`, `square_side`, `words`).
  - `vertical_levels`: the scheme's level-count bounds, when it has them.
  - `suites[]`: the registered suites that carry it.
  - `described_in`: whose settings the cost and card numbers were taken in.
- `suites[]`: every registered suite. `on_create_page` marks the ones the
  Create page can run; those also carry `day_only`, `day_only_reason`,
  `verification`, `summary` and `cost`.
- `presets[]`: `id`, `intention` (the button text), `suite`, `dx_km`,
  `why` (one line). Table data in `gpuwm/physics_catalog_v1.json`.
- `default_suite`, `source`, `cards`, `scheme_count`,
  `physics_registry_sha256`, `command`.

Cached for 10 minutes per source. About 1 s to build. An unknown
`source` is HTTP 400 naming the registered sources.

### POST /api/physics/check

Request (every field optional):

```json
{"suite": "a registered suite id; the source default when absent",
 "preset": "a preset id: its suite and dx_km, unless given here",
 "choices": {"pbl": "myj", "radiation": "rte-rrtmgp:rrtmg_legacy"},
 "settings": {"radt": 10},
 "dx_km": 3, "nz": 49, "source": "gfs", "card": "16gb",
 "cycle": "2026-09-24T12", "hours": 6, "lat": 35.0, "lon": -97.0}
```

`cycle`, `hours`, `lat` and `lon` are spelled as Create spells them.
Give `cycle`, `lat` and `lon` together (hours defaults to 6) and the
check also asks the load-time night door: a suite that runs shortwave
with no longwave is refused when the window includes local night at that
point.

`choices` are applied over the suite. Radiation by stream is
`"radiation": {"longwave": 4, "shortwave": 1}`. `settings` are applied
last, by the switch names an experiment file uses; a name no switch has
(`mp_physcs`) is HTTP 400 naming it and the nearest real names.

Reply: `{"ok": true, "command": ..., "check": {...}}` where `check` is
`gpuwm.physics-check.v1`:

- `valid`, `words` (one line to show), `base_suite`, `choices` (what was
  checked, family by family), `dx_km`, `nz`, `source`, `card`, `preset`,
  `window` (the cycle, hours and point the night door was asked at, or
  null).
- When valid: `resolved` (family to scheme, as the engine resolved it),
  `named_suite` (the suite this is, or null), `named_suite_label` (its
  plain name, for the page to show, with its cumulus part shown as
  "cumulus off" when the grid turns it off; null with `named_suite`),
  `plan_intent` (the run plan intent keys that run exactly this: for a
  named suite, `physics_profile`, plus `cumulus: "grid"` when no cumulus
  scheme was picked, so a root finer than the convection-permitting
  spacing runs none, as `words` says; for any other set,
  `physics_choices`, the choices as sent, plus `physics_profile` when
  the base suite is not the source's default; null only for a set
  that needs raw `settings`, which no intent key carries),
  `on_create_page` (true when `plan_intent` is not null: New forecast
  starts every such set),
  `day_only_reason`, `advisories[]` (`words`, a plain sentence for the
  page; `headline` and `sentence`, the engine's own), `cost`,
  `card_fit` when a card was given, `changed_from_suite` (the switches
  the mix sets differently from `base_suite`) and `experiment_toml` (the
  `[shared]` physics lines of the mix).
- `words` for a day-only suite starts "Runs by day only" and says either
  that the given window is all daylight or that a window with local night
  is refused.
- A valid mix with no named suite: `words` says no named set matches it.
  New forecast starts it as its schemes (see the Physics step below). In
  a terminal, `gpuwm domain --physics-choices JSON` writes it into a new
  experiment, and
  `gpuwm physics-catalog --check JSON --into EXPERIMENT.toml --out MIX.toml`
  writes an existing file with the mix in it (`[shared]` gets every physics
  switch but one every `[[domain]]` table states for itself; a
  `[[domain]]` table keeps its per-grid values except a switch the mix
  changes, and a nest that runs no cumulus keeps it off; quoted names such
  as the `["shared"]` of `gpuwm domain-fit` are the same tables and keys,
  and a line already holding the mix's value is left as written) after the
  real experiment loader has accepted it,
  keeps the root's cumulus as the file runs it unless the mix chooses
  cumulus (the command prints one line when the mix turns it on, off or
  to another scheme), and
  copies the experiment's `namelist.wps` under the new name, since the
  route reads it by that name. Without `--out` it prints the TOML only.
  An experiment with a `namelist.input` beside it (the HRRR route) is
  refused: that file states physics too and is not rewritten. Then
  `gpuwm go MIX.toml`, or a run plan with `config.path`.
- When refused: `refusal` (`door`, `owner`, `message`: the engine door
  that stopped it, word for word) and `neighbours[]`, at most five, each
  `words`, `choices` (send these back merged into your choices),
  `settings`, `changes` (`from`, `to` per family), `named_suite`,
  `advisories`. Neighbours that keep every family the caller chose come
  first, then fewest changes, then the registry's own remedy, then named
  suites.

Doors, in the order a run meets them: `registry` (a scheme the registry
declares unimplemented), `configuration` (`gpuwm.config.validate_run_config`),
`capability` (`physics_compat.validate_physics_capabilities`),
`radiation-off-land-surface`, `nocturnal-radiation` (when a window is
given), `source-route` (the source's emission gate, when `source` is
given).

A name the registry does not have (scheme, preset, suite or source) is
HTTP 400 with the list of names; the engine prints it as `{"error": ...}`
and exits 2, and the server's runner carries that document through. A
refused combination is HTTP 200 with `valid: false`: the refusal is the
answer. A check takes 0.05 s when valid and 1 to 5 s when refused (the
neighbour search runs every one- and two-family change through the same
doors).

### Presets on Create

`POST /api/create/start` and `/api/create/fit` take `"preset": ID`. The
preset fills `profile` (the suite) and `dx_km` when those are blank; an
explicit value wins. Every shipped preset's spacing is one Create can
start (the LES preset is 0.5 km, Create's finest outer grid); a preset
row finer than that is refused by name unless `dx_km` is given.

A suite the catalog marks `day_only` (the LES preset's) is checked
against the draft's window on `fit` and on `start`: the same
`physics-catalog --check` with `cycle`, `hours`, `lat` and `lon`, so a
window with local night is HTTP 422 in the night door's words before the
run is launched, not a refusal at load after it.

### The Physics step of New forecast

New forecast has a Physics step between How fine and Review. It draws
`GET /api/physics?source=ID` as one table per family, each scheme a row
with its description and its cost against the default, and asks
`POST /api/physics/check` with the draft's source, spacing, levels,
cycle, hours and centre whenever a row is picked. Every set the check
says runs can be started. One that the source offers as a suite goes into
the plan as that suite (`profile`, which becomes
`config.intent.physics_profile`); any other goes in as the rows picked
(`config.intent.physics_choices`).

`POST /api/create/start` also takes `"physics_choices": {family: scheme}`,
the rows picked. The server runs the check once more at the draft's own
values: a set that does not run is HTTP 422 in the check's words, and a
`profile` that is not the suite the choices make is HTTP 409. Nothing is
written before either. A set that makes a suite the source offers goes
into the plan as the check's `plan_intent`
(`config.intent.physics_profile`, and `config.intent.cumulus` when the
check left cumulus to the grid), so the run has the cumulus the Physics
step showed. Any other set that runs goes into the plan as
`config.intent.physics_choices`: `gpuwm domain --physics-choices` writes
those schemes over the source's default suite the way
`gpuwm physics-catalog --into` writes a mix, on every size its fit tries,
with the cumulus the check decided at the draft's spacing, and nothing
asserts a suite. `POST /api/create/fit` sizes a set that runs the same
way, and one the check refuses as the form's own set.
The set is kept beside the plan as `gui-physics.json`
(`gpuwm.gui-physics-choice.v1`: `suite` (null for a mix no set matches),
`label`, `base_suite`, `cumulus`, `choices`, `resolved`, `words`, `cost`,
`command`), and the run's wiki article lists it as its Physics fact,
citing that file and the plan ("Picked schemes, no named set" and the
schemes, for a mix). A run with no composed choice names its plan's
suite, or "The source's default set".

### The run manifest

`run-manifest.json` now carries `physics`: `suite`, `stated_by`,
`components`, `switches` and `physics_registry_sha256` when the plan
states a suite; `suite: null` with `stated_by` naming the route default
when it does not. A plan with `physics_choices` records `choices`,
`base_suite`, the `components` they resolve to, `switches` (those that
differ from the base) and `suite`, the named suite they make or null. A plan whose config is a file (`config.path` or
`config.inline`, the way a mix from `--into` runs) records that file's
physics: `switches` from `[shared]`, `inert` for the `[shared]` keys only
a scheme that is not running reads (`wsm6_hail_opt` under P3, say),
`domains[]` with each grid's own
values, and `suite` and `components` as the first domain resolves them
(`suite` null when no named suite matches).

`pid` has `process` beside it: `{pid, start, boot}`, the process's
creation time (and on Linux the boot it belongs to). The job record
(`gui-job.json`: `wrapper_pid`, `wrapper_process`), the card lock, the
queue's owned list and the assistant's `server.json` keep the same pair.
A run is `running` only while the process behind a recorded PID has that
identity; a PID that now names another program (reused after a crash or
a restart), or a record written without `process`, reads as ended, and
Stop answers 409 and signals nothing.
The escalation a minute after a Stop checks the group's leader again
before it sends SIGTERM.

A started job that ended before the engine wrote an end event (a plan the
engine refused at its start, a run killed outright) is `failed`, or
`stopped` when it was cancelled, never `ready`: `end` carries
`exit_code`, `stage` (`start` when the engine wrote nothing) and
`message`: the first plain line the engine printed at a refused start,
or the reason a traceback ended on (without the traceback).

## Run folders: what My forecasts lists and opens

`GET /api/runs` lists every run folder under the root, `NAME` or
`FOLDER/NAME`, newest first by its marker files. A folder is listed once,
by the path it resolves to. One that resolves outside the root (a link to
a folder elsewhere, at either depth) or deeper than two folders is not
listed. Past 4,000 runs the oldest are left off: the reply carries `total`
(every run found), `limit` and `truncated`, and My forecasts says the list
is cut. A folder whose records cannot be read is listed with the state
`unreadable` and a plain reason (the error goes to the page server's log),
and the other rows still load.

A run opens by the name its folder has: spaces, accents and names longer
than 64 characters included, URL-encoded as one path segment
(`FOLDER%2FNAME`). The plain-name rule (`[A-Za-z0-9][A-Za-z0-9._-]{0,63}`)
applies only to names the page creates. An id must stay under the root and
be spelled as the folder is on disk; `.`, `..`, dot folders, separators
and drive names are refused.

The render tree is `<render root>/<domain>/<product>/<valid-day>/*.png`,
and a nest that retires and re-arms files each life one segment deeper,
`<domain>/episode-NNN/<product>/<valid-day>/`. `GET /api/runs/RUN/pictures`
keeps `domain` and `episode` as separate keys (`episode` is `""` for a
nest with one life), `/pictures/list` takes `?episode=`, and every
picture's georeference is read from `render-georef.json` at the render
root.

## Starting data and the queue

### GET /api/sources/availability

`?time=YYYY-MM-DDTHH&hours=N`. Answers at once, never waiting for a data
server. Each row starts from the publication schedule and the archive
bounds; a start inside the last 24 hours, and then each source's newest
start, are put to the fetch's own object probe in the background (the same
rungs, URLs and rule as `gpuwm.fetch.cycle_publication_refusal`), with a
5 s timeout per HEAD, a 3 s limit on waiting for a NOMADS turn, a 15 s
deadline per source, one start's objects asked at once, and each object
asked once across every check the server runs. The newest start is asked
newest first, one start at a time, stopping at the first one a host holds
whole. A start more than a day old asks no server at all: its row is the
archive bounds', and its `confirmed` and `missing` are what an earlier
check found (`newest_basis` `unasked` when none is kept). Ask again while
`pending` is above zero; the page does, once a second.

One rule decides what each source offers, per source and length
(`gpuwm/gui/availability.py`):

- A start is **confirmed** when a check found it whole, when it is
  earlier than a start found whole (a source publishes its starts in
  order), or when no server is asked about it (a keyed archive, or a
  start more than a day old). A row never infers from its own start.
- A start is **due** when it is older than the time the source usually
  takes to publish a whole run: the cycle grid's `usual_delay_hours`,
  measured (GFS 6 h, GDAS 8 h, HRRR 3 h; a route-table source uses its
  measured lag).
- **Start** takes a start that is confirmed, or due and not found
  missing by a check.
- **Queue it** also takes a start from the last day that is not
  confirmed. The forecast is held, saying so, until Start would take the
  start, and then starts in its turn.
- The page **opens on** the newest start Start takes: the newest start a
  check confirmed whole, or the newest due start no check found missing,
  whichever is newer. It is the **Newest run** only once a check
  confirmed it whole and found every newer start on the source's hours,
  through the schedule's newest, not yet whole; while a newer start went
  unheard, nothing is the Newest run. The page works the opening start out
  from its own clock, the row's `usual_delay_hours` and `cycle_hours`, and
  the row's `confirmed` and `missing`, and a draft the person has not moved
  follows it. It takes the Newest run from the row's `newest_run`.

| field | meaning |
|---|---|
| `state` | what the row reads: `yes` (has it), `unknown` (may have it, or not yet), `no`. Each row is asked about the download its run makes (a 3-hour forecast from files that come every 6 hours downloads hours 0 and 6); a `no` the download itself would give, a run whose files thin out before the window ends, carries `fix` with the lengths that download, and a row whose check failed is `unknown` with `starts` `no` |
| `starts` | what takes this start: `now` (Start), `queue` (only Queue it, which holds the forecast until Start would take it), `no` (neither) |
| `why`, `note` | the reason, in the page's words; `note` says how a `yes` is known when no check answered (earlier than a start found whole: `inferred_from` names that start; or due) |
| `checking` | this row's check has not answered yet |
| `basis` | `checked` (a check answered, or the start is confirmed by another start's check), `schedule` (no server is asked), `unchecked` (a host was not heard in time), `checking` |
| `checked_age_s` | seconds since the check answered; answers are kept 10 minutes (90 s for "not published yet", 60 s when unheard) |
| `confirmed` | the newest start a check found whole, for this source and length; it never moves back when a later check hears fewer hosts. For a source no server is asked about, the schedule's own newest start |
| `missing` | newer starts a check found not yet whole |
| `due`, `usual_delay_hours` | the newest due start by the server's clock, and the delay it comes from |
| `opening`, `newest_run` | the start the rule opens on, by the server's clock, and that start when a check confirmed it whole and every newer start not yet whole (else null) |
| `newest_basis` | `checked` (the checks name the newest run, or every host answered), `schedule` (no server is asked), `checking` (the source's newest-start check is out), `unchecked` (a newer start went unheard; the page asks again every 15 s), `unasked` (the row's start is more than a day old, so today's newest start is not asked about, and no earlier check is kept) |
| `newest_checking` | the source's newest-start check is out |
| `pending` | rows still being checked |

`GET /api/sources` also answers at once. Each source row carries
`cycle_hours` and `usual_delay_hours`, so the page opens on the due start
of the first source by its own clock before any row answers;
`default_cycle` is the same rule's opening start by the server's clock,
`default_newest_run` names it once a check confirmed it whole and every
newer start not yet whole, and
`default_checking` is true while the check is out. `/api/create/fit` and
`/api/create/start` read the same answers; either may begin a check in the
background when none is fresh, and a fit never waits for it. Start and
Queue it wait for one check only: the chosen source's check of the start
they write, when that start is inside the last 24 hours and its check is
still out, for at most 20 s. Start then refuses, in words, a start that
only Queue it takes, and names the opening start; Queue it takes it and
`waits_for_data` says why the forecast waits. Their dry runs (Show
command) answer the same. Next on When goes on for any start Start or
Queue it takes; a check still out never holds it. A start more than a day
old asks no server, so a fit or an event's start for an old date is not held
by a check of today's runs.

Which requests may wait, and on what: the source list, the availability
rows, the queue listing and the Machines list never wait on a data
server, on SSH or on a start. `/api/create/fit` runs the engine's fit in
the request (a few seconds; a repeat of the same fit is answered from its
cache). Start, Queue it and an event's **Run the best simulation for this event** button
(`POST /api/wiki/simulate`) may wait up to 20 s for the one check above;
the button refuses a start only Queue it takes and points to Customise,
where Review offers Queue it. Machines **Check**, and a start on a
Machines node, wait on SSH by design. The engine's source list and
physics sets are kept in `<root>/.arwen-gui/engine/`, so a restarted page
server answers its first page from the last answer the same engine
command gave while it asks again; only a runs folder the engine never
answered for waits for its first answer.

### The queue

`POST /api/create/start` with `"queue": true` writes the run folder (plan,
box, physics) and a `gui-queued.json` marker instead of starting it; the
order lives in `<root>/.arwen-gui/queue.json`. Both are files in the
forecasts folder, so a queue survives a page or page-server restart. With
`"machine": "NAME"` the forecast is queued for that Machines node; every
check a start makes runs first. `need_gib` is the fit's card memory, which
decides whether the forecast fits beside another program on the card.

    GET  /api/queue                        ?machine=&need_gib=&source=&cycle=&hours= the queue in order, this
                                           computer's card, and "expect": where and when a new forecast would
                                           start, and whether Start or only Queue it takes that start
    POST /api/queue/RUN/up|down|remove     reorder, or remove (deletes the folder the queue wrote)

The page server's scheduler looks every 5 s. Per machine, the first
queued forecast starts once the card is free: no forecast holds the job
manager's card lock, no live line is in the OWNER file, and no other
program leaves less free memory than the fit needs (a Machines node: its
probe says idle). One forecast starts per card at a time. The job manager
checks its card lock, spawns the run and writes the lock under one
operating-system lock, so two starts in the same instant (two tabs, the
page and an MCP client) admit one and refuse the other. A forecast that
no longer fits (less than 5 GiB free on the disk that holds the forecasts,
less than a storm-following forecast's own `disk_gib`, or a fit bigger
than the card) is held with the reason in `held`, and the
ones behind it
keep their turn; it starts by itself once the reason is gone. So is one
whose start Start would not take yet (a start from the last day no check
has confirmed): each look asks the page's own checks, which ask the data
servers again every minute or two, and it starts in its turn once the start
is confirmed; its `awaits_data` is true, and My forecasts shows it as a wait,
not in the stop colour a forecast held for disk or card gets. `waiting` says
why the first one waits. Queuing a forecast,
Move up and Move down wake the scheduler, and a reorder rewrites every
forecast's `waiting` at once: the first one on a machine that is not held
takes the card's reason, the ones behind it wait for the forecasts ahead.
A forecast queued for this computer gets its `waiting` in the same
request, from the page's last card reading, before the scheduler looks.

`GET /api/queue` reads the order file and each queued folder's marker,
never the whole forecasts folder: the scheduler looks there for a queued
folder the order file does not list when it starts and once a minute.

The scheduler reads the card and starts a forecast outside the queue's
lock, so neither `GET /api/queue` nor a Queue it ever waits for a start
under way; a page is served the scheduler's last card reading.
A forecast being started has left the line (Remove answers 404); a start
that does not happen puts it back in its place, and so does the next page
server when the last one stopped in the middle of a start.

`gpuwm gui --owner-file PATH` (or `GPUWM_GPU_OWNER_FILE`) names the
card-sharing OWNER file; `GPUWM_GPU_OWNER_TAG` names this server's lines
(default `gpuwm-gui`). Every local start then takes the card there, a
queued one or Start on a free card: one test-and-append under the file's
locks (the file itself and a sibling `OWNER.lock` when there is one), a
line `TAG UTC pid PID bounded 720 min` handed to the run's job wrapper
(`gpuwm.mcp._jobwrap`, told of the file through its receipt's
`owner_file`), which removes the line when the run ends, whether or not
the page server is still running. The line is handed over under the
file's locks only while the wrapper still has the identity its job
recorded (PID and creation time); a run that already ended has the line
removed instead. The queue's `owned` list keeps each run's wrapper
identity and the exact line it handed on (`tag`, `utc`, `pid`), and the
scheduler removes that line, and no other under the same number, once
its wrapper has ended without removing it (a wrapper killed outright).
The server's own claim is kept in `owned` the same way before it is
made, with the server's identity and the time its line is written with,
so a page server that stops between claiming the card and handing the
line on leaves a record: the next scheduler removes that line once the
server has ended or its number names another process, which after a
reboot it often does. A record left by a server stopped before its line
was written names no line and removes none. A line of
the server's tag that no record names is removed once no process holds
its number; a line of another tag is never touched.
The line's format is unchanged, since other tools on a machine read it.
Another owner's live line holds the card, and a start on it is refused
with who holds it.

`expect` in `GET /api/queue`: `busy`, `why`, `ahead` (queued before it),
`running` and `seconds_left` (the forecast on the card), `held`,
`card_name`, `total_gib`, `free_gib`, `start_now` (true when the card
can take it now, beside another program if its fit leaves room), and,
with `source`, `cycle` and `hours`, `data`: `starts`, `why` and
`checking` for that start, from the rule above.

## Machines

Machines are the computers ArWen runs and draws forecasts on: this computer,
any number of SSH hosts, and cloud machines. The code is `gpuwm/gui/machines.py`
(the table, the checks, the transport), `gpuwm/gui/remote_runs.py` (forecasts on
machines, render workers, the follower), `gpuwm/gui/cloud.py` with
`gpuwm/gui/providers.json` (cloud machines as table data), and
`gpuwm/machine_agent.py` (the one file that runs on a machine).

### The table

Machines are rows of `~/.gpuwm/machines.toml` (or `$GPUWM_MACHINES_FILE`).
Adding a host is adding a row; nothing in the code knows any host.

```toml
[[machine]]
name = "gpu-box"                    # letters, digits, . _ - ; not "this-computer"
kind = "ssh"                        # "ssh", or a provider from providers.json ("aws")
host = "me@gpu-box.local"           # user@host or an ~/.ssh/config alias
port = 22                           # optional
identity = "~/.ssh/id_ed25519"          # optional key file on this computer
workspace = "/data/gpuwm"           # the one folder ArWen writes on the machine (default ~/gpuwm-machine)
python = "/data/gpuwm/venv/bin/python"     # optional; default <workspace>/venv/bin/python
owner_file = "/data/gpu-mutex/OWNER"       # optional card-sharing file (see below)
owner_tag = "gpuwm-gui"             # the word this computer's runs write in it
geog_root = "/data/WPS_GEOG"        # optional; becomes the plan's run_options.geog_root
data_dir = "/data/forcing"          # optional; becomes run_options.data_dir
env = { GPUWM_CASE_DATA_ROOT = "/data/cases" }   # optional environment for runs there
```

SSH is key authentication only. Every call is `ssh -o BatchMode=yes
-o StrictHostKeyChecking=yes`, which never asks for a password; a row with a
`password`, `passphrase` or `secret` field is refused. An unknown host key is
refused with "run ssh HOST once in a terminal and confirm the key", because
accepting it silently would send forecasts to whatever answers on that address.

**The OWNER file.** A machine whose cards are shared by several people or
lanes can keep one file listing who holds the card, one line each:
`<tag> <UTC> pid <n> bounded <m> min`. When a row names it, a forecast
there claims the card only when `nvidia-smi` shows no compute process and no
line's pid is alive, with one test-and-append under a file lock; it removes its
own line (and nobody else's) when it ends, and it is interrupted when it runs
past its bound. Without an OWNER file the card is claimed when nothing else
computes on it.

### Endpoints

    GET  /api/machines                     this computer plus every row, each checked (5 s cache; ?fresh=1)
    GET  /api/machines/NAME                one machine, checked now, plus its stored row
    GET  /api/providers                    cloud providers: label, needs, defaults, credentials, cli
    POST /api/machines/add                 {row fields..., replace?}   checked before it is saved
    POST /api/machines/NAME/check          check again now
    POST /api/machines/NAME/remove         remove the row (nothing on the machine is touched)
    POST /api/machines/NAME/install        {wheelhouse, from_machine?}
    POST /api/machines/NAME/start          cloud machine: launch or start, then the SSH check
    POST /api/machines/NAME/stop           cloud machine
    POST /api/machines/NAME/terminate      cloud machine
    POST /api/create/start                 + {machine, render_on, wait_min, bound_min}
    POST /api/runs/RUN/render              {machine, products?, render_section?}
    POST /api/runs/RUN/follow              follows a machine's run or drawing again (launches nothing else)
    POST /api/runs/RUN/stop                stops a run on whichever machine runs it

**A machine, as listed and checked** (`GET /api/machines` rows, `check` replies):

| field | meaning |
| --- | --- |
| `name`, `kind`, `host`, `workspace` | from the row |
| `state` | `idle`, `running` (a forecast of ours), `rendering`, `busy` (another owner holds the card), `offline` (with `detail` and `fix` saying why) |
| `detail` | one line: "idle", "running RUN", "the card is held by <OWNER line>" |
| `cards` | `[{index, name, memory_total_mib, memory_used_mib, driver}]` |
| `card` | the card class a plan is sized for: `12gb`, `16gb`, `24gb`, `32gb` |
| `vram_gib`, `cuda`, `compute_processes` | |
| `version_here`, `version_there`, `version_matches` | gpuwm on this computer and there |
| `offer` | present when the versions differ: `{action: "install", words}` |
| `install_extra` | the CuPy extra the machine's CUDA driver needs (`gpu-cu12`, `gpu-cu13`) |
| `install` | the last install there: `{state: installing, installed, failed, started_utc, seconds, log_tail}` |
| `disk_free_gib` | free space in the workspace |
| `owner_convention`, `owners` | whether the OWNER file exists, and its lines with `live` |
| `jobs` | `{forecasts: [{run, state, alive}], renders: [{job, run, state, alive, rendered}]}` |
| `round_trip_s` | how long the check took |

The first row is always `this-computer` (`kind: "local"`), from this computer's
card and the job manager's card lock. While the lock is held its `detail` is
"running RUN, started 2026-09-27 09:30 UTC", read from the lock's record and
the holding job's receipt (never from the lock's launch refusal, which is
written for an MCP client and names its job tools), and the row adds `run`
(the run's id, null when the job writes a folder that is not one of this
page's forecasts, which `detail` then says), `title` (the name the run reads
under), `url`, `started_utc` (when the job took the card) and `job_id`. The
Machines page links that line to the run's page. A lock file that cannot be
read holds the card too: the row is `busy`, and `fix` names the file to remove
if nothing runs on the card. A start the lock refuses (a forecast that took
the card a moment earlier, from another tab or an MCP client) is HTTP 409
naming the running forecast and when it started.

**Add.** The body is a row. An SSH row is checked before it is saved: a machine
that cannot be reached is not added, and the reply says why (`Not added. ...`
with a `fix`). A cloud row is saved as it is; it is checked once it starts.
`dry_run` answers with the row as it would be written.

**Install.** The two wheels of exactly this computer's version (the gpuwm Linux
wheel and its gpuwm_data companion) are taken from `wheelhouse`, a folder on this
computer or on `from_machine`, copied into `<workspace>/install/wheels` through
this computer, and installed into `<workspace>/venv` with the CuPy extra the
machine's driver needs. The install runs detached; the machine's `install` field
shows its progress, and when it finishes the row's `python` becomes the new venv.
A folder without the same version is refused: a plan made here would be read by
different code there.

### Forecasts on a machine

`POST /api/create/start` with `"machine": "NAME"` runs the forecast there. The
same draft fields as a local run; `card` defaults to the machine's class.

- `render_on`: the machine that draws the pictures (default: the same machine;
  `"none"` draws nothing). The render worker draws every frame as the forecast
  commits it, so the run itself is told to draw none.
- `wait_min` (default 0): with 0, a busy card is refused with the holder named.
  Above 0, the run waits up to that many minutes for the card and starts when it
  frees up; its status reads `running` with `phase` "waiting for the card: ...".
- `bound_min` (default 120): the bound written into the OWNER line; the run is
  interrupted past it.

The reply carries `argv`/`command` (the engine command on the machine),
`remote` (the mirror's record), `follower` (the detached job keeping the
mirror up to date), and `version_here`/`version_there`. A machine whose gpuwm
version is not this computer's is refused (dry run too), because the plan is
written here and run there: a plan written by one version and run by another
can mean different physics, fields or paths than the page showed. The fix is
Install on the machine's row.

**The mirror.** The run gets a folder under the runs root like any other, with
`gui-machine.json`:

    {machine, host, remote_rundir, launched_utc, alive, ended, checked_utc,
     events_offset, job: {state, supervisor_pid, engine_pid, exit_code, ...}}

The follower copies the machine's `events.jsonl` (by byte offset), its
`run-progress.json` and the tail of its engine output every 5 s, so
`GET /api/runs`, `/api/runs/RUN/status` and the event stream work unchanged.
Run rows now carry `machine` (`this-computer` for local runs) and `render`
(`{machine, state, pictures, worker_state, message}`), and `status` carries
`machine` and `machine_checked_utc`. `GET /api/runs/RUN` adds `remote` and
`render_job` (with `relays`, `batches`, `bytes_relayed`, `bytes_pulled`,
`first_fed_utc`, `first_picture_utc`).

A mirror is never `ready`: when the machine's job ended with no end event (it
was stopped while it waited for the card, refused, or killed), its state is
`stopped` or `failed` from the job's own state, and a mirror with no word from
the machine at all is `stale`. `POST /api/runs/RUN/start` refuses a mirror by
name ("This forecast belongs to NAME"), because its `plan.json` holds that
machine's paths and would otherwise run on this computer's card.

### Render workers

`POST /api/runs/RUN/render {"machine": "NAME", "products": "..."}` draws any run,
local or on a machine, running or finished, on any machine. `products` is
`gpuwm render --products` (omitted: the renderer's default set). New
forecast's named pictures and this endpoint accept `render_section` as
`lat,lon,lat,lon`. Every `xsec:` product requires that line before a draft
or render request is made; without it the product would draw nothing. The
draft writes it into `run_options.render_section`, and a separate Machines
render worker receives the same line. GUI lines use coordinates, since a
section file on this computer may be absent on the selected machine. A draft
sent with `products_named: true` ("Named pictures or a cross-section") and no
product names is refused, because an empty list would draw the standard set
while the page still shows the named choice. A draw that failed, finished or
was stopped is asked for again from no frames; one still going on that
machine is the one returned. `this-computer` draws with the interpreter the
page runs on, and its pictures are copied back folder by folder. The
Machines page's "Draw a forecast on a machine" panel is this endpoint's door.

Frames are the run's own `output_committed` events. When the render machine is
the forecast's machine, the worker reads them where they are; otherwise they are
copied through this computer (a tar stream from the source into the worker's
inbox). The worker runs `gpuwm render --series` (the Rust renderer) on each new
batch, with the previous frame of each domain as context, so windowed products
(hourly maxima, accumulations) come out right frame by frame. It starts with
the first committed frame, not at launch, so a run waiting for a card does not
hold an idle worker. Each picture is copied back
into `<run>/render-<machine>/<domain>/<product>/<valid-day>/` the moment the
worker lists it, so `GET /api/runs/RUN/pictures` and `/pictures/list` show the
pictures while the forecast runs. The follower ends by itself once the forecast
has ended and every picture is here; closing the page does not stop it.

A follower that ends before that (it gives up after about ten minutes of failed
calls, or its process was stopped) leaves the mirror as it last was. `status`
then carries `follow_lost: true`: the state, progress and pictures shown are
the last ones received, not the machine's state now. The map says so and
offers Resume updates, which is `POST /api/runs/RUN/follow`: it starts one
follower (two pages asking at once still start one, under the run folder's
`gui-follow.lock`) and launches no forecast and no new drawing. An open map
keeps reading while a render worker's state is `requested` or `rendering`, so
pictures that arrive after the forecast finished appear without a reload.

A relayed frame is a whole wrfout on the render machine's disk, so the worker
deletes each one once it is drawn, except the newest of each domain (the next
batch's context), and deletes the rest when the render ends. Only drawn frames
are deleted: a frame that arrives while a batch draws waits for the next batch,
and a frame fed just before the render's end is drawn before the worker
finishes. Frames a worker
reads in place (a render on the forecast's own machine) are never deleted.

**Measured (2026-09-25, desktop on the office LAN).** A 3-hour HRRR forecast,
104 x 102 points at 3 km and 49 levels, started through this API on an RTX
5070 Ti machine and drawn on another machine: 67 s on the card from claim to end
(fetch 4 s from its cache, prepare 35 s, forecast 25.5 s); four frames of 40 to
42 MB each relayed at about 7.5 MB/s through this computer; first picture on the
page 24 s after the first frame was fed, before the forecast had ended; 829
pictures (397 MB) back on this computer 53 s after the first frame was fed. The Machines
check of an SSH host takes 1 to 1.3 s; installing this version (140 MB of
wheels copied between two machines through this computer in 16.3 s, then pip)
took 50 s and 68 s.

### Cloud machines

A cloud row names a provider from `gpuwm/gui/providers.json`; the provider is a
table of its own CLI's command lines, filled from the row with `{{field}}`.
Adding a provider is adding an entry there. The AWS row:

```toml
[[machine]]
name = "cloud-a"
kind = "aws"
image = "ami-..."                  # the ArWen image (below)
instance_type = "g6e.xlarge"
region = "us-east-1"
profile = "default"                # your own AWS login on this computer
key_name = "my-key"                # the EC2 key pair whose private key you hold
security_group = "sg-..."          # allows SSH from this computer
subnet = "subnet-..."              # optional
price_per_hour_usd = 1.86          # what the cap is turned into minutes with
spend_cap_usd = 20
idle_stop_min = 30
ssh_user = "ubuntu"
workspace = "/opt/gpuwm"
```

`start` runs: `aws sts get-caller-identity`, `aws ec2 run-instances` (or
`start-instances` for an instance it already has), `aws ec2 wait
instance-running`, `describe-instances` for the address, `host_key`, then
`settle` and `set_cap` over SSH, then the same check as any SSH machine. The
address becomes the row's `host`, and from then on forecasts and renders use
the same path. `stop` and `terminate` run `settle` over SSH, then their one
provider call.

**Host keys.** A cloud machine's address changes on every start, so its SSH
host key is kept under the instance id: `host_key` reads the keys cloud-init
prints on the instance console (`aws ec2 get-console-output`), writes them to
`known_hosts` beside the machines file as `<instance_id> <type> <key>`, and sets
the row's `known_hosts` and `host_key_alias`; every SSH call then runs with
`UserKnownHostsFile` and `HostKeyAlias`, and `StrictHostKeyChecking=yes` still
holds. It waits up to 10 minutes for the keys and for SSH. A key that does not
match is a failed start (another machine may be answering on that address).
Any SSH row may set `known_hosts` and `host_key_alias` the same way.

**Money has a ceiling on the machine itself.** The first boot installs
`arwen-cap-guard`, a service that counts the minutes the machine runs in
`/var/lib/arwen-cap-guard/used_minutes` (on the machine's own disk, so the count
survives stops and starts, with one minute added per boot) and shuts the
machine down when the count reaches `cap_minutes` (what is left of the cap, at
the row's price) or after `idle_stop_min` minutes with no `python -m gpuwm` or
`machine_agent` process and nothing on the card. A render worker waiting for
frames (`machine_agent ... render-loop`) is not counted as work; its drawing
(`python -m gpuwm render`) is. So a cloud machine drawing for a forecast that
died before its end stops `idle_stop_min` after its last batch, not when the
worker gives up six hours later, and frames further apart than `idle_stop_min`
need a longer `idle_stop_min` on the row. The instance is launched with
`--instance-initiated-shutdown-behavior stop`, so a guard shutdown is a stop
and a stopped machine costs only its disk.

The guard's count only grows. The row counts the money: `spent_usd`
(settled), `guard_minutes_charged` (how much of the guard's count
`spent_usd` already covers), and while minutes are not settled, `started_utc`
and `cap_minutes_granted`. `settle` reads the guard's count and charges only
the minutes past `guard_minutes_charged`, so each minute is charged once
whichever start, stop or terminate reads it: a start, a stop through ArWen and
the next start charge one session once. `set_cap` gives the guard a cap of the
count already charged plus the minutes the money left buys; it never resets the
count, so the minutes between the read and the new cap are charged by the next
settle. A new machine settles too, so its minutes from boot to SSH are charged
before its cap is set.

When the count cannot be read, the minutes stay unsettled and are counted at
the most the machine can have run: the smaller of the time since `started_utc`
and `cap_minutes_granted`. A machine the guard stopped is counted that way
until its next start. A stop through ArWen freezes that bound at the time of the
stop plus one minute (the machine counts until it is down); a terminate charges
whatever is still unsettled, since nothing reads that machine again. The
listing shows unsettled money as `unsettled_usd`, and a start whose settled
plus unsettled money leaves under a minute is refused, naming both. A start
whose cap cannot be set on the machine (no guard, no passwordless sudo, SSH
refused) is a failed start: the instance is stopped again and its minutes since
the last read stay unsettled at their bound, because a machine without its cap
is a bill with no ceiling. A row with no price is refused (a cap that cannot be
turned into a time limit is not a cap).

Every action takes `dry_run`: the reply's `steps` list each call as `argv` and
`command`, in order, with `user_data` (the guard script), `cap_minutes`,
`remaining_usd` and `unsettled_usd`. Nothing runs and nothing is spent. From a
terminal: `gpuwm machines cloud start NAME --dry-run`.

Measured on two LAN machines (2026-09-25), with `shutdown` and `nvidia-smi`
stubbed and the real `pgrep`: on a machine with no gpuwm process the guard's
idle count rose 1, 2, 3 and it called the idle stop at its limit; on a machine
running a forecast it stayed at 0. The guard's own command line
(`/bin/bash /usr/local/bin/arwen-cap-guard`) does not match its test. Measured
again the same way on a LAN machine with no gpuwm process (2026-09-25): with
only a waiting render worker the idle count rose 1, 2; with a forecast
supervisor beside it, it stayed at 0.

**What the image must contain.**

- Ubuntu 22.04 or 24.04 with the NVIDIA driver for the instance's card, and
  `nvidia-smi` on the PATH.
- `python3` (3.11 or newer) and `python3-venv`.
- `<workspace>/venv` holding the gpuwm and gpuwm_data wheels of the version the
  desktop runs, installed with the extra the driver needs (`gpu-cu12` or
  `gpu-cu13`), so `gpuwm doctor` passes on first boot. (A version mismatch is
  still fixable after boot with Install.)
- The Rust bridges staged inside that wheel (the platform wheel carries them),
  so `gpuwm render` draws with `rw_wrfbatch` and no build step runs on the
  machine.
- The static data the routes read: `WPS_GEOG` (as `geog_root` on the row), the
  Thompson tables and RRTMGP data (in gpuwm_data), and any mesh files the MPAS
  port and the global model use, baked in so a first forecast does not wait on
  a download.
- The MPAS port (gpuwm-hex) and the global model packages of the matching
  release, in the same venv.
- The SSH user from the row with the key pair's public key, and passwordless
  `sudo` (the guard's cap is set with it; without it every start fails and the
  machine is stopped again).
- cloud-init's default of printing the SSH host keys on the console at boot
  (`-----BEGIN SSH HOST KEY KEYS-----`), which is how the desktop learns them.
- `pgrep` from procps (with `-a`) for the guard's idle test. The guard itself comes from the
  launch's user data; an image baked from a machine that already ran it keeps
  `/var/lib/arwen-cap-guard`, so remove that folder before baking.
- No credentials of any kind. The desktop's AWS login never leaves the desktop.

### From a terminal

    gpuwm machines list
    gpuwm machines add NAME USER@HOST [--workspace DIR] [--owner-file FILE] [--owner-tag WORD] ...
    gpuwm machines check NAME
    gpuwm machines install NAME --wheelhouse DIR [--from-machine NAME] [--dry-run]
    gpuwm machines cloud start|stop|terminate NAME [--dry-run]
    gpuwm machines remove NAME
    gpuwm machines follow --root ROOT RUN      (what the page starts in the background)

`gpuwm machines --json ...` prints the same documents the API answers with.

## The assistant

Code: `gpuwm/gui/assistant/`. Routes: `service.py`. The page panel:
`gpuwm/gui/static/js/assistant.js`, with `bridge.js` between the panel and
the Create form. Terminal door: `gpuwm assistant`.

### What it is

The person says what they want to see ("a squall line over the plains
tomorrow afternoon at 1 km"). The assistant reads the request once, then
answers every setting as a typed decision over ArWen's own tables and
fills the Create form with a checked plan, each field with a one-line
reason. It asks at most one question (the place, when none is named). It
never starts, stops or deletes a run, installs anything or reaches
another machine: Start and Stop come back as buttons whose request is the
page's own, sent only by the person's click. Downloading the model needs
`{"confirm": true}`, which only the panel's Download button sends.

### Optional, and off until turned on

The assistant ships off. Settings (`#/settings`, `settings.js`) and the
panel show, while it is off, the model this card would get with its
size and licence and a **Turn on** button; the sidebar entry reads
"Assistant (off)" with one line saying to turn it on in Settings, and
Ctrl K offers the same entry. `GET /api/session` carries
`assistant: {enabled}` so the sidebar knows before the panel opens.

Off, nothing is downloaded, no model server is started, no card memory
is used and no model server is asked anything: `install` (dry run
included), `load`, `say`, `decide` and `v1/systemone` answer HTTP 409
with `off: true`, message "The assistant is off." and a fix naming
Settings and the panel; the catalog's `detected` is empty (no port on
this computer is probed); status reads files only; a forecast's fit and
start never mention it (`make_room` and `card_warning` return None).
Settings may still be saved, so an endpoint can be set before turning
it on. Every other page works the same on or off.

`POST /api/assistant/enable` with `{"on": true}` saves the switch and
downloads nothing. When the bundled model is not on this computer yet,
its reply carries `install`, the same list as the install dry run (each
file, its size and licence), and the panel and Settings show that list
with a Download button: the download is still its own click.
`{"on": false}` stops the bundled server, so the model leaves the card,
and keeps the files; `POST /api/assistant/remove` deletes the home's
`models`, `server` and `downloads` folders and the server's log, and
nothing else in the home. `gpuwm assistant on` and `gpuwm assistant off
[--remove]` do the same from a terminal.

### Where its answers come from

Besides planning a forecast, the model has read tools over the page's
own data: `search_wiki` and `read_event` (the storm wiki's events: title,
summary, facts, rarity, runs; never map geometry), `read_run` (a run's
wiki article: source, start, length, grid, physics, levels, box,
pictures, each from the run's files), `run_status` and `list_runs`, and
`check_fit` (the engine's own fit). The panel names where each reply was
read from, and shows each typed decision as its options under letters
A, B, C..., the chosen one marked, each with its probability.

Turned on, with no model on this computer and no endpoint set, `GET /api/assistant`
says `downloaded: false` with the model's size and licence, and
`POST /api/assistant/say` is HTTP 409 at once ("The assistant's model is
not downloaded yet.", fix: Set up) with the install plan; the panel shows
both ways to get a model (Set up, or a model server such as LM Studio or
Ollama, with any found on this computer offered by name).

### Endpoints

| Method | Path | Body | Answer |
|---|---|---|---|
| GET | `/api/assistant` | | status (below) |
| GET | `/api/assistant/catalog` | | `models`, `brought` (models a person runs on their own server, never downloaded), `endpoints`, `decision_services`, `detected` (endpoints answering on this computer; empty while off), `server` (the llama.cpp build for this computer) |
| GET | `/api/assistant/conversations` | | `conversations`: `id`, `started_utc`, `turns`, `first` |
| GET | `/api/assistant/conversations/ID` | | the saved conversation: every turn as `say` returned it |
| POST | `/api/assistant/enable` | `on` (true or false) | status, `stopped`, and on with the model not here yet `install` (the dry-run list). Anything but a boolean is 400 |
| POST | `/api/assistant/remove` | | `freed_bytes`; status. With `dry_run`: `bytes`, `home`. 409 while a download runs |
| POST | `/api/assistant/settings` | any of `backend` (`bundled`, `endpoint`), `model` (a catalog id), `endpoint_url`, `endpoint_model`, `endpoint_kind` (`ollama` lets Start unload it), `key`, `decisions` (`local`, `jev`, `kev`), `decision_url`, `decision_key` | `settings` without the keys |
| POST | `/api/assistant/install` | `model`?, `confirm` | with `dry_run`: `items` (each `what`, `url`, `bytes`, `sha256`, `licence`), `bytes`, `home`. Without `confirm: true`: 409 carrying the same list as `install` |
| POST | `/api/assistant/load` | | loads the model on the card; status |
| POST | `/api/assistant/unload` | | `stopped`; status |
| POST | `/api/assistant/say` | `text`, `conversation`?, `form`? (the Create form as the person sees it, same keys as `/api/create/start`) | a turn (below) |
| POST | `/api/assistant/decide` | `question`: `{id, instructions, options: {id: description}}`, `state` | `answer`: one decision record |
| POST | `/api/assistant/v1/systemone` | the System One wire: `state`, `questions: {id: {type: "choice", instructions, criteria}}` | `answers: {id: {choice, probabilities, confidence}}` |

Status: `enabled`, `off` (the sentence the panel shows while off),
`card_gib` (the card the model is picked for, or null), `downloaded_bytes`
(what the assistant has downloaded into its home), `backend`,
`decisions`, `model` (`id`, `name`, `bytes`, `min_card_gib`, `licence`,
`note`, `quant`, `context`, `why`), `downloaded`, `server_found`,
`loaded`, `vram_gib` and `load_s` when loaded, `install` (`state`:
`idle`, `starting`, `downloading`, `unpacking`, `done`, `failed`, with
`what`, `bytes`, `total`, `message`), `forecasts_running` (run ids),
`note` (a sentence for the panel, such as the model having left the card
for a forecast), `endpoint`, `endpoint_model`, `decision_url` (not a
secret: the settings form shows it again, and the form's save sends only
the fields the person changed), `key_set`, `decision_key_set`, `home` (where the model is kept, in words such as
"the .arwen/assistant folder in your home folder", never a path; the same
words as the install list's `home`).

A turn: `conversation`, `user`, `reply`, `question` (set when the
assistant asks its one question; answer it with the same
`conversation`), `error`, `plan` (`fields`, `reasons`, `fit`, `fixes`,
`read`, `window_utc`, `machine`; each of `fixes` is `id`, `field` (the
`reasons` key it rewrote), `before`, `after` and `words`, the sentence
saying what was given up in the refusal's own numbers, which the reply
also carries), `decisions`, `tool_calls` (each `name`,
`args`, `ok`, `error`, `seconds`), `actions`, `seconds`, `llm` (`calls`,
`seconds`, `prompt_tokens`, `output_tokens`, `tokens_per_s`, `each`),
`model`, `note`.

A decision record: `question`, `instructions`, `options` (ids),
`choice` (always one of `options`), `probability`, `probabilities`
(per id, or null when the server gives no log probabilities),
`confidence` (System One's `(p_max - 1/K) / (1 - 1/K)`), `reason`,
`method` (`logprobs`, `systemone`, `only-option`, `unmeasured`),
`backend`, `seconds`.

### Actions: what the page does with a turn

| `type` | Fields | The page |
|---|---|---|
| `fill_create` | `fields`, `reasons`, `fit`?, `fixes` | sets those Create fields, marks each, shows its reason beside it, draws the fit. `reasons` keys: `when`, `box`, `ladder`, `source`, `physics`, `machine`, `start`, `card`. A reason whose key is some fix's `field` says what was given up; the page marks it apart (class `given-up`) |
| `go` | `page`, `run`? | opens that page |
| `confirm` | `action` (`start`, `stop`), `label`, `request` (`method`, `path`, `body`), `needs_name`, `command`?, `problem`? | shows a button. A `start` click POSTs the Create form as it is on screen (`bridge.readForm()`), so the person's own edits go with it; `request.body` is sent only when no Create form is mounted. A `stop` click POSTs `request` as given |
| `question` | `missing`, `text` | shows the question |

The Create form in `create.js` reads `fill_create` through
`bridge.onFill` and hands itself to the panel through
`bridge.setFormReader`; a redesigned Create page keeps both calls and
the assistant keeps working.

### The Create fields this lane added

`/api/create/fit` and `/api/create/start` take `ladder` (one of
`/api/sources`' `ladders`: the wizard's `12`, `12-3`, `12-3-1`,
`12-3-1-0.5`, `auto`) and `start_hour` (the cycle's forecast hour the run
starts from; the plan's intent `forecast_start_hour`). A ladder and
`dx_km` together are refused: the wizard fixes the spacing from the
ladder and refuses `--root-dx` beside `--ladder`. `/api/sources` rows now
carry `step_hours`, `coverage` (`west`, `east`, `south`, `north`, or
null for global), `coverage_words`, `status`, and each profile's
`status`.

A ladder and an event's `chain` together are refused with HTTP 400 that
names both (`gpuwm run-plan --resolve` refuses `--ladder` beside
`--chain`); New forecast's nest ladder replaces the event's `chain` and
`buffer_km` rather than sending them beside it.

A storm-following event's Customise sends `following: true` with the
layout's `event` and `recipe_card_gb` (the card size its Customise opened
with). The server reads that layout from the storm wiki and runs its own
`gpuwm cyclone-setup` options, as the event page's button runs them
(history intervals, surface flux and, for a nest that grows, a nest
budget set to the card); options the page sends of its own are not used.
`/api/create/fit` prices it with `gpuwm cyclone-setup --json`, and a
start runs the setup once in a scratch folder under the server's
`drafts` with the draft's `source`, `cycle`, `hours`, `start_hour` and
`card` replacing the event's, keeping every text file it writes: the
`cyclone.toml` the plan runs on the `experiment` route and the Vtable and
WPS namelist it names beside itself. A local start writes them into the
run folder; a start on a Machines node writes them into the mirror here
and sends them with the plan (`text_files`), and the node writes them as
text beside it. The plan names the pictures to draw (`render_products`,
the standard set unless the page says otherwise): a configuration file's
run draws nothing without it. A dry run answers the setup's argv as
`prepare`. `following` without an `event`, or with a layout the event
does not have for that card, is refused; a box or grid other than the
event's own is HTTP 422, and `following` beside a `ladder`, `chain`, `nz`,
`profile` or `physics_choices` is HTTP 400 (the setup sizes its own
grids, levels and physics). New forecast drops `following` when the grid,
box, levels or physics move off the event's own, and the Fine step and
Review say so. Once the setup's files are in the run folder, a local start resolves the
written plan (`gpuwm run-plan PLAN --resolve`) and reads the engine's own
projection of what the run writes (`disk.total_bytes`, the number
`gpuwm run-plan` refuses on before its download). A start that needs more
than the free disk under the runs root is HTTP 507, and its folder is
removed, as `/api/wiki/simulate` refuses the event's row; a queued one
keeps the projection as its queue marker's `disk_gib`, and the queue holds
it until the disk has the room.

Every Customise that keeps the event's own grids (its outer spacing and
nests) sends the layout's `event` and `recipe_card_gb`, following or not.
The server reads that best run's `intent` from the storm wiki and writes
every key of it that New forecast has no control for into the plan
(`DRAFT_INTENT_KEYS` in `gpuwm/gui/api.py` lists the ones it has), such as
`history_interval_s`, `nest_history_interval_s` and `isftcflx`, so a
Customise nobody changed plans exactly what `/api/wiki/simulate` plans.
An `event` whose best run the store does not hold for that card is HTTP
422. `/api/wiki/recipe/ID` reads every value New forecast has a control
for from the same `intent`, and adds `layout`: the best run's `domains`
(spacing, points and size of each grid), `physics` (`profile`, `why`) and
`output` (its history intervals), which How fine, Physics and Review show.

A start claims its run folder in one step (creating it) before writing
anything into it, so a second start under the same name, from another tab
or another page server on the same forecasts folder, is HTTP 409 rather
than a second plan written over the first. A queued start on a Machines
node starts in the folder the queue wrote only when the queue has marked
that folder as starting; a start that fails removes what it wrote there
(plan, region, physics choice and companion files) and the queue tries
again.

New forecast keeps its whole draft for the browser tab (session
storage, under the link it was drawn on), so a reload of any step comes
back to every choice. A link past When that the tab never drew opens at
Where with a notice. While `/api/sources` is on its way the link keeps
the `at` and `src` it came with (and a date picked since), and never
gains the browser's own guess at a start, so a reload in that wait, or a
second drawing of the same link, finds the same draft.

### The typed decisions

`plan.py` asks, in order: `day` (now, or one of the next eight local
dates), `part_of_day` (morning, afternoon, evening, overnight,
whole-day), `box_size` (300 to 2500 km), `ladder`, `source` (only
sources whose coverage holds the box and whose horizon reaches the
window), `physics` (the source's admissible profiles with their
verification status), `machine`, and, when Check the fit refuses
because the plan does not fit the card, `fix` (a later start dropped,
a coarser ladder, a smaller box, or a shorter window). A memory refusal
is known by the two numbers every wording of the engine's verdict
carries, `<need> GiB peak envelope` and `the <budget> GiB budget`, and
the reason beside the field quotes those two numbers, never an engine
line. Any other refusal (the install, the data, the computer) is no
question: the reply says it at once in plain words, without the
`--explain` pointer, traceback lines or folder paths. A coarser ladder is the next shallower row of
the wizard's `LADDER_RATIOS`, so a new ladder row coarsens in its place.
Local time at the place is solar time from the longitude, which can
differ from the clock there by about 2 hours, so a part of a day is
widened by 2 hours on each side and the `when` reason says so. Each question is the System One shape, 1 to
255 options; more than 26 are asked as group then member.

When the physics answer is the source's default profile, the plan
leaves `profile` empty, the form's "The source's default", which is the
same physics without naming it. Naming a profile on a nested ladder is
refused by the prepared tree forecast after fetch and preparation: the
wizard turns cumulus off on the nests and the forecast then asserts the
named suite on every domain. That engine conflict is open; until it is
settled, a non-default profile with a nested ladder reaches that refusal.

Local answers: the options are lettered, the reply is held by a JSON
schema to `{"choice": letter, "why": text}`, and the probabilities are
the log probabilities at the letter's token renormalised over the
letters. With `decisions` set to `jev` or `kev`, the same questions go
to `POST {url}/v1/systemone`. Kev's weights are Apache-2.0 (a LoRA on
Qwen3.5-4B-Base); Laya runs on Apple's MLX only, so it is not listed.

### Where the model runs, and the card

`catalog.json` lists models, llama.cpp builds, endpoints and decision
services; adding one is a row. The bundled model is picked by card
memory: the largest row whose `min_card_gib` the card reaches, with
0.6 GiB of slack because a card reports a little under its label (a
16 GB card reads 15.9 GiB). A row carries `quant`, `context`, `repo`,
`revision` (the pinned commit), `file`, `url` (the file at that
revision), `sha256` and `bytes` (the published file's, checked after
download), `licence` (`id`, `holder`, `url`), `request` (extra
chat-completions fields; every row sends `enable_thinking: false`),
`server_args` (extra llama-server flags, such as an 8-bit attention
cache), `note` and `why`. Each bundled row is Apache-2.0 or MIT.

| Card | Model | Quant | File | Context | Licence |
|---|---|---|---|---|---|
| under 8 GB | Qwen3.5 4B | Q4_K_M | 2.6 GiB | 16K | Apache-2.0 |
| 8 GB | Qwen3.5 9B | Q4_K_M, 8-bit cache | 5.3 GiB | 16K | Apache-2.0 |
| 12 GB (and no card found) | Qwen3.5 9B | Q6_K | 6.9 GiB | 16K | Apache-2.0 |
| 16 GB | Qwen3.8 27B | UD-IQ4_XS, 8-bit cache | 13.3 GiB | 16K | Apache-2.0 |
| 24 GB | Qwen3.8 27B | UD-Q5_K_M | 18.4 GiB | 16K | Apache-2.0 |
| 32 GB and up | Qwen3.8 27B | UD-Q6_K | 20.5 GiB | 32K | Apache-2.0 |

The files come from unsloth's GGUF repositories on Hugging Face.
Qwen3.8 27B (2026-08-14) is the newest open Qwen and uses the same
architecture and thinking switch as Qwen3.5; Qwen has published no 3.6,
3.7 or 3.8 model below 27B, and 3.7 has no open weights at all. Below
16 GB the rows stay on Qwen3.5, which scores above Gemma 4 E4B and
Gemma 4 12B with thinking off. Ternary Bonsai 2 27B (Qwen3.8 27B in
ternary weights, 5.5 GiB) would fit the 8 and 12 GB tiers, but no
official llama.cpp release loads its PQ2_0 and PTQ1_0 weight types, and
nothing published shows it ahead of Qwen3.5 9B with thinking off, so it
is a brought row.

`brought` rows (`id`, `name`, `maker`, `params_b`, `active_b`,
`min_card_gib`, `where`, `file_gib`, `licence`, `why_not_bundled`) are
strong models the assistant never downloads, listed with their licences
for a person to run on their own server: Ternary Bonsai 2 27B (only
PrismML's own llama.cpp build loads it), K2 Horizon 7B and MoVA
36B-A4B (upstream llama.cpp does not run them yet), G9v3 39B-A5B (no
llama.cpp release supports its architecture; community GGUFs load only
on a community branch), Muse Glimmer 30B (Apache-2.0 with Meta's usage
policy, and no thinking switch) and Qwen3.8 Flash Next 125B-A6B (the
Qwen Community License asks anyone running a Model-as-a-Service or AI
Work Assistant business for a separate licence from Qwen before
commercial use, and very large products to show the model's name, so it
is not Apache or MIT equivalent). Settings lists both tables. The
server listens on 127.0.0.1 with its record in the assistant home
(`GPUWM_ASSISTANT_HOME`, else `~/.arwen/assistant`). Starting a forecast
through the page stops the bundled server first (or unloads an Ollama
model) once the runner has accepted the start; a start refused because
another forecast runs leaves the model where it is. The start reply
carries `assistant: {unloaded, message}`. A model server on this
computer that the gui cannot unload (an endpoint on 127.0.0.1 or
localhost that is not Ollama with a model name, such as LM Studio or a
separate llama-server) is named instead: the start reply carries
`assistant: {unloaded: false, warning}` and `/api/create/fit` adds the
same sentence as `fit.warning` and to `fit.words`.
While a forecast runs, `say` on the bundled model answers 409 with a
sentence saying why, and the panel shows it.

An endpoint served by Ollama's `/v1` route returns no log
probabilities, so its decisions come back with `probabilities: null`
and method `unmeasured`; llama.cpp's server returns them.

### Guarding against text in data

Tool results and anything from a run folder reach the model between
`DATA` and `END DATA`, and the system prompt says they are data. The
model's words never become settings: `plan_forecast` passes the
person's own words to the planner, every setting is a listed id, and
`change_plan` values pass the API's own checks and the listed ids.

## Downscale

A finished or stopped run's map offers **Downscale**: a finer forecast inside it,
started and driven by the run's saved output (`gpuwm/gui/downscale.py`, the
`DownscaleMixin` in `gpuwm/gui/api.py`, the panel in `static/js/downscale.js`).

### GET /api/runs/RUN/downscale

What the run offers, read from its own event log and file names only (the server
opens no NetCDF file):

| Field | Meaning |
|---|---|
| `eligible`, `reason`, `fix` | whether the run can be downscaled, and in words why not |
| `domains` | one row per grid with saved frames: `id`, `frames`, `restart_sets`, `interval_s` (the most common gap between frames), `dx_km` (from the resolved plan), `first`, `last` |
| `domain` | the grid a finer forecast starts from by default: the finest one saved |
| `ratio`, `ratio_range` | the default refinement (3) and the accepted range |
| `output_minutes` | the parent's cadence, as the desktop panel's default output interval |
| `window_hours` | the saved window of the default grid |
| `name` | a free default name, `<run>-d02-x3` |
| `cards`, `products` | the card sizes the page offers and the default products (`all`) |

The reasons are the desktop panel's: a run on another machine, a run that has not
finished or stopped (its newest frame and checkpoint can be half written), no
saved frames, a single saved frame ("a finer forecast is driven between two saved
times"), no checkpoint ("the engine reads the parent's physics from its
checkpoint").

### POST /api/runs/RUN/downscale

| Key | Meaning |
|---|---|
| `mode` | `plan` reviews (the engine's `--dry-run`), `run` starts; default `run` |
| `lat`, `lon` | the finer forecast's centre; the engine sizes it to the card |
| `box` | `{west, south, east, north}` instead of a centre: its centre, and whole child cells sized as the desktop's `box_cells` sizes a drawn box (`--child-size`) |
| `domain` | the grid it starts from (default: the finest saved) |
| `ratio` or `dx_km` | the refinement, or the spacing it reaches (the nearest whole refinement); both must agree |
| `hours` | its length (default: the whole saved window) |
| `output_minutes` | its output interval (default: the parent's cadence) |
| `card` | `8gb` to `32gb` plans it for that card (`--card`); absent measures this computer's card (`--auto-vram`) |
| `products` | `all` (default), `none`, or product names separated by commas |
| `name` | the new run's name (default as above) |

Any other key is refused. The command is the one the terminal controller builds
for the same answers (`tools/arwen-tui/src/main.rs`, `downscale_guide`), token
for token:

```
python -m gpuwm downscale HISTORY --point=LAT,LON --parent-restart=latest
    --parent-domain=N --ratio=R [--child-size=NX,NY]
    --max-boundary-interval-seconds=B [--hours=H] [--output-interval-seconds=S]
    [--card=TIER] --tiles=auto --out ROOT/NAME [--dry-run] [--auto-vram]
    --render-products=SPEC
```

`B` is the chosen grid's `interval_s`, the cadence its frames were saved at: the
engine refuses frames that are not at that cadence, and records `B` as the
ceiling asked for. The page never sends `--accept-parent-cadence`, which
records a person's acceptance of the archive's cadence.

`"dry_run": true` returns that line and runs nothing. `mode: plan` runs it with
`--dry-run` and `--out` in a draft folder under `.arwen-gui/drafts/`, reads the
plan the engine wrote beside it, deletes the draft folder and answers `review`
(the grid, hours, memory, streaming verdict, the engine's warnings, and
`outline`, the planned grid's corners as `[lat, lon]`) with `start_argv`, the line
Start runs. `mode: run` makes `ROOT/NAME` empty (the engine adopts only an empty
`--out`), launches the line detached through the job manager holding the card,
writes the start record `.arwen-gui/starts/NAME.json` (`job`, `command`,
`parent`), and appends the command to the parent run's `commands.log`. The new
run lists at once as running; if the engine refuses it before it writes a
manifest, the run reads as failed with the engine's sentence from the job's
`stderr.log`.
