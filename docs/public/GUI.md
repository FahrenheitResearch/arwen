# The Weather Library: `gpuwm gui`

`gpuwm gui` starts a small server on your own computer and opens the Weather Library in
your normal browser. It is the graphical door of ArWen 2.8. Nothing is fetched
from the internet to draw the page, and nothing on it needs a build step.

```
gpuwm gui
```

It prints a link like `http://127.0.0.1:8766/?token=...` and opens it. The token
in the link is what lets your browser in; the page moves it into a cookie and
drops it from the address bar. Keep the link to yourself.

| Option | What it does |
| --- | --- |
| `--root DIR` | the folder that holds your forecasts. Default: `$GPUWM_RUNS_ROOT`, else `~/arwen-runs` |
| `--no-open` | print the link only; do not open a browser |
| `--port N` | listen on port N. Default 8766, or the next free one of ten; `0` picks any free port |
| `--bind ADDR` | the address to listen on. Default `127.0.0.1`, this computer only |
| `--allow-remote` | allow a `--bind` address other computers can reach. Refused without it |
| `--verbose` | log every request |

Press Ctrl+C in the terminal to close the page server. Closing it, or closing the
browser tab, never stops a forecast.

## A first forecast

Every forecast builds its grids' terrain, land use and soil from the geography
data, set up once per computer with `gpuwm fetch-geog --datasets wrf`. Until it
is, **Start** and **Run the best simulation for this event** refuse with that command and write
nothing.

1. `gpuwm gui`. The page opens on its main page; **New forecast** is in the
   sidebar under the forecasts group.
2. **Where**: click the map near latitude 39, longitude -97 to place the box, and
   pick Small (300 by 300 km). **Next**.
3. **When**: starting data GFS, a start time six or more hours old, 3 h long.
   **Next**.
4. **How fine**: Overview (12 km) and the engine default levels. Each choice is
   checked against your card with the engine's own fit; this one fits as 28 by 28
   points at 12 km, 49 levels, and the fitted grid is drawn solid over the box.
   **Next**.
5. **Physics**: leave every table on the data source's default set, or pick a
   row to change one part of the model. **Next**.
6. **Review and start**: the table repeats every choice, the physics set and the
   card memory it needs, where the forecast runs and when it starts. **Start
   forecast** opens the forecast's map while it runs. When the card is busy the
   buttons read **Start now** and **Queue it**; see [The queue](#the-queue).

The computer needs the static geography tree first (`gpuwm fetch-geog`); without
it the run stops at the prepare stage and the page says so.

## The pages

The sidebar has the **Weather Library** group of storm pages, your forecasts, **Machines**, the
**Assistant** (off until you turn it on) and **Settings**, and a search box
(Ctrl+K) that finds events, places, forecasts and pages and opens the
assistant.

- **Weather Library** opens on its main page: 17 cited events, 11 tropical cyclones and
  6 tornadoes, with pages by kind, by place and by recent change. Each event page
  gives its observed facts, its sources and its place on the map, and **Simulate
  this event** offers the best simulation for your card, started by
  **Run the best simulation for this event**: a plan for 8, 12, 16, 24
  and 32 GB cards, each accepted by the engine's own fit check, with its grids, its
  hours and the disk it needs, its download and preparation included. For a start
  from the last day it waits, as Start does, for the check of that start; a start
  not confirmed yet is sent to **Customise this run**, where Review offers **Queue it**.
  **Customise this run**, beside that button, opens New forecast with that plan
  filled in: its start data, start time and length (the event page says them in
  words), its nests or storm-following nest, its physics and how often it writes.
  Left unchanged, it starts the same plan as the button. The How fine step shows
  the event's grids as one choice, every grid named, and the step and Review say
  when another choice leaves them out. Like **Run the best simulation for this event**, it refuses to start a
  storm-following run that needs more disk than the disk holding your runs has free.
- **My forecasts** lists every forecast folder under the root, with its state:
  Imported, Queued, Ready, Running, Finished, Stopped, Failed or Stale. Forecasts
  waiting to start are listed above the others, in the order they start. Forecasts started
  from a terminal show here exactly like ones started from the page, whatever the
  folder is called: spaces, accents and long names open like any other. A running
  forecast shows its progress and the line that names its grids and the one that
  sets the pace. Click one to open it. A folder whose records cannot be read is
  listed as Unreadable and the others still load. A root with more than 4,000
  forecasts lists the newest 4,000 and says how many there are in all.
- **New forecast** is five steps beside a map: Where, When, How fine, Physics and
  Review and start. Where takes a click on the map, or **Find an event**, which searches the
  Weather Library and fills in the event's box, date and length. When takes any date
  from 1940 on: a calendar with month and year jumps, a typed date such as
  `1999-05-03`, the hour, and quick picks (the newest run, yesterday, a week ago,
  a year ago). Under it every data source says whether it holds that start and,
  when it does not, why: the date is before its archive begins, it is not
  published yet, the model does not run at that hour, or that run's files stop
  coming at the source's spacing before the forecast ends (the row names the
  lengths that download). Each row asks about the files the forecast downloads:
  a source whose files come every 6 hours (AIFS, AIGFS, AIGEFS) downloads hours 0
  and 6 for a 3 h forecast, so a start has it once hour 6 is published. Each row answers on its
  own: it says checking beside its name until that source's data server has
  answered, and the date box, the calendar, **Next** and the other rows work
  meanwhile. One rule decides what each source offers. **Start** takes a start a
  check found whole (or an earlier one: a source publishes in order), or one
  older than the time the source usually takes to publish a whole run (about
  6 hours for GFS, 8 for GDAS, 3 for HRRR) that no check found missing.
  **Queue it** also takes a start from the last day that is not confirmed yet:
  the forecast waits, saying so, until a check confirms the start, and then
  starts in its turn. The page opens on the newest start Start takes. It calls a
  start the newest run only once a check found it whole and every newer start not
  yet whole; while a newer start could not be checked, no start is called the
  newest run. A start more than a day old needs no check and is answered at
  once from each archive's bounds, with no data server asked. ERA5 holds every hour
  from 1940 and needs no account key. How fine checks each grid, and each choice
  of vertical levels, against the card size you pick (8, 12, 16, 24 or 32 GB) with
  `gpuwm run-plan PLAN.json --resolve` and draws the fitted grid over the box.
  Each choice and the Review step show the card memory the grid needs against
  the card's budget, for example `Fits: needs 5.9 of 14.5 GiB usable`, from
  the `memory` record that check returns for every data source. A choice too
  big for the card says how much it needs, from the figures the check's
  refusal carries.
  Physics is described below. Review shows when the forecast starts (the data's
  run plus the forecast hour set under More settings) and the data's run on its
  own row. **Start forecast** writes the run folder and launches
  `gpuwm run-plan PLAN.json`. The choices are kept for the browser tab, so a
  reload comes back to the same step with every choice.
- **Files** lists a forecast's pictures by kind, domain and valid day, one folder
  at a time, and opens each as the file the renderer wrote. A nest that retired
  and re-armed has one row per life (`d02-3km, life 2`). The run folder, the
  reattach files and the commands the page ran are on the same page.
- **Machines** lists this computer and every computer you added, with its state,
  its card and the gpuwm version there. See [Machines](#machines) below.
- **Assistant** opens a panel beside any page. It is optional and off until you
  turn it on; see [The assistant](#the-assistant) below.
- **Settings** turns the assistant on or off, removes its model, and lists the
  models it can download for each card and the ones you can bring.

An open forecast adds its own group to the sidebar:

- **Article**: what the forecast is, where and when it ran, its physics and its
  state. A forecast that failed gives why it stopped and what to do among its
  facts, in the same words as its Map; one that was stopped says why under its
  title.
- **Map**: the renderer's pictures placed at their true place on the map, with the
  time bar, **Play** and the arrow keys. While the forecast runs, the same page
  watches it: the time, the time left, the progress and the stage along the time
  bar, and each picture placed as the renderer writes it. Every grid is drawn as
  its frames are written, so on a nested run the grid chips at the top
  (`d02 · 3 km`, `d03 · 1 km`) appear and fill in while the forecast is still
  going; a chip turns its grid's pictures on or off. **Play**, the step buttons
  and the arrow keys go through the times a grid turned on has a picture at, and
  a time picked on the bar where none has one says so and offers the nearest time
  that has one. The live line names every
  grid and the one that sets the pace, for example
  `12 / 3 / 1 km, 21x real time, the 1 km grid sets the pace`. **Stop** is on the
  live line. The main maps come first (reflectivity, temperature, precipitation,
  wind, in the order `gpuwm/gui/copy/looks.json` lists them).
  A finished or stopped forecast's map has **Downscale** at the top right; see
  [Downscale](#downscale-a-finer-forecast-inside-a-finished-one) below.
- **Files**: the same page as above, for this forecast.

Every button is one `gpuwm` command. **Show command** under a button shows the
exact line before you press it: the page asks the server with `"dry_run": true`,
which returns the command and runs nothing. Every command the page ran is
appended to the run's `commands.log`, one line you can paste into a terminal to
run it again.

## Physics

The Physics step has one table per part of the model: microphysics, cumulus,
the boundary layer, the surface layer, the land surface, radiation and
turbulent mixing.
Each scheme is one row with its cost against the data source's own default
set, or "not measured" where no cost has been measured. The default set for the
run's finest grid runs unless you pick another row: the data source's own, or
below 1 km Thompson with MYNN and RUC wherever the source admits it, the set
`gpuwm domain` writes there; the step says when the two differ. With the auto
nest ladder the finest grid is the one its fit lands on for the card, so the
step, Start and the Article read the ladder the run gets. **Starting points**
offers whole sets to begin from.

Every pick is checked by the engine at this grid, place and time, the same
check `gpuwm physics-catalog --check` makes in a terminal. When the picks match
a set the data source offers, the step names that set, and it goes into the
plan with the cumulus the step showed: picking only microphysics at 3 km runs
no cumulus scheme, because the grid resolves storms itself. The Review row
names the set, and at Start the server checks the choice again and saves it
beside the plan as `gui-physics.json`; the forecast's Article names every
scheme and cites that file.

A mix that runs but matches no set starts too. The step says no named set
matches, the Review row lists the picked schemes, and the engine writes them
into the forecast's experiment over the default set for its finest grid, with the
cumulus the step showed. The Article names every scheme. In a terminal,
`gpuwm domain --physics-choices JSON` writes a mix into a new experiment the
same way, and this writes one into an experiment file you already have:

```
gpuwm physics-catalog --check MIX.json --into EXPERIMENT.toml --out MIX.toml
```

`--into` writes a mix that names no set over the default at the file's finest
grid. A file finer than 1 km written by 2.8.0 with no set named runs YSU and
Noah, and a mix written into it now lands over MYNN and RUC; to keep its
boundary layer and land surface, name its suite in the mix (`"suite"`).

Only a mix the engine refuses has no Start, and the step gives the engine's
reason and the nearest mixes that run. `gpuwm domain --cumulus grid` turns
cumulus off the same way for a named suite in a terminal.

## Downscale: a finer forecast inside a finished one

A finished or stopped forecast's **Map** has **Downscale** at the top right. It
runs a finer forecast inside that one, started and driven by the output the
forecast saved, with nothing fetched or prepared again:

1. Click the map for the finer forecast's centre, or press **Draw a box** and drag
   its extent. The engine sizes a centre to the card; a box becomes whole cells of
   the finer grid, sized as the desktop app sizes a drawn box.
2. Choose **How fine** (2 to 8 times finer than the grid it starts from, with the
   spacing each gives), **Hours** (blank runs the whole saved window), **Card**
   (blank measures this computer's card; a size plans it for a card of that size
   instead), **Pictures** and a **Name**. A forecast with nests starts from its
   finest grid unless you pick another.
3. **Review** asks the engine to plan it (`gpuwm downscale ... --dry-run`) and shows
   the grid, its hours, the card memory it needs, whether it streams and the
   engine's notes, and draws the planned grid on the map. Nothing runs, and
   nothing is written into your forecasts folder.
4. **Start** runs exactly what was reviewed as a new forecast in your forecasts
   folder and opens its map. A form changed after its review waits for another
   review before Start.

The form says how often the grid it starts from saved a frame, which is how
often the finer forecast's edges are driven. The command asks the engine for
exactly that cadence:

```
gpuwm downscale PARENT/chain/NAME/run/wrfout --point=LAT,LON --parent-restart=latest --parent-domain=1 --ratio=3 --max-boundary-interval-seconds=3600 --output-interval-seconds=3600 --tiles=auto --out RUNS/NEW-NAME --auto-vram --render-products=all
```

When those frames are further apart than 15 minutes, the engine's notes say the
cadence is coarser than its guidance for downscaling. That is a fact about the
parent, not a fault in the request: a finer forecast driven by hourly frames
keeps the large-scale weather close, and places its storms differently from one
driven every 15 minutes. A forecast with nests saves its nests every 15 minutes
by default, so a finer forecast started from a nest has no such note.

**Show command** shows it with your answers before you press anything. The finer
forecast reads the parent's newest checkpoint for its physics and its saved
frames for its start and its edges, so a forecast needs at least two saved frames
of the grid and one checkpoint; when one is missing, the panel says which.
A finer forecast is a forecast like any other: it lists under **My forecasts**,
its map places it inside its parent, and it can be downscaled again.

## Machines

The Machines page lists the computers that run and draw forecasts: this one, any
computer you reach by SSH, and cloud machines. Each row shows the machine's
state, its card, the gpuwm version there against the one here, and what it is
doing.

- **Add a machine** takes a name, an SSH host (`user@host`, or a Host name from
  your SSH config), a port, the one folder ArWen may write in there, and
  optionally where that machine keeps the static geography. The machine is
  checked over SSH before it is saved: its card, its free memory and its gpuwm
  version. Sign-in is by key only; a password is never asked for or kept, and a
  host key SSH does not already know is refused.
- **Check again** repeats that check. **Remove** drops the row and touches
  nothing on the machine. **Install this version** installs this computer's
  gpuwm version there: it asks for the folder that holds that version's
  gpuwm Linux wheel and its gpuwm_data wheel, and which machine the folder
  is on (this computer by default). **Find the wheels** names the two it
  found without copying anything; **Install** copies them and starts the
  install, and the machine's row shows the result. While the install is still
  putting in gpuwm's requirements, or when one of them is missing, the row says
  so and the machine takes no draw or forecast until they are all in; a
  forecast queued for it meanwhile waits in line.
- A cloud machine has **Start** and **Stop** on its row.
- **Draw a forecast on a machine** draws any forecast, running or finished, on
  any machine in the list. The pictures come back into the forecast's folder as
  they are drawn, and the forecast's map places them.

Machines are kept in `~/.gpuwm/machines.toml` (or the file
`$GPUWM_MACHINES_FILE` names), one row per machine. `gpuwm machines` lists,
adds, checks, installs and removes them in a terminal, and
`gpuwm machines cloud start NAME` starts a cloud machine.

## The queue

When the card is busy, or the start is one only **Queue it** takes (from the
last day, not confirmed yet), Review and start offers **Queue it** beside **Start
now**. The **Starts** row says when a queued forecast expects to start: once a
check confirms its start is published, when the forecast on the card finishes
(with the time left once the run knows it), after the forecasts ahead of it, or
when another program lets go of the card. **Start now** is offered only when
Start takes the start and the card has room for a second forecast beside what
runs there.

Queued forecasts are listed at the top of **My forecasts** in the order they
start, each with why it waits, and **Move up**, **Move down** and **Remove**
(Remove deletes the folder the queue wrote; nothing ran in it). Each is a run
folder with a `gui-queued.json` beside its plan, and the order is kept in
`.arwen-gui/queue.json` in the forecasts folder, so a queue survives closing the
page and restarting `gpuwm gui`. Moving one says at once why each one waits.

While `gpuwm gui` runs, open page or not, it looks at the queue every 5 seconds
and starts the first forecast once the card is free: no forecast from the page
holds it, no other program leaves too little card memory for the forecast's fit,
and no one else's line is in the OWNER file, when there is one. One forecast
starts at a time. A forecast queued for a machine on the Machines page starts on
that machine when its card is idle. A queued forecast that no longer fits (the
disk that holds the forecasts has less than 5 GiB free, or less than a
storm-following forecast writes, or its fit is bigger than the card) is held
with the reason, and the ones behind it keep their turn; it starts by itself
once the reason is gone. So is one whose start is not confirmed yet: the page
asks its data servers again every minute or two, and it starts in its turn once
a check confirms the start. My forecasts shows that one as an ordinary wait,
and a forecast held for disk or card in red.

On a card shared with other programs through an OWNER file, `gpuwm gui
--owner-file PATH` (or `GPUWM_GPU_OWNER_FILE`) makes every start from the page
take the card there: one line `TAG TIME pid PID bounded 720 min`, written under
the file's lock and removed when the forecast ends, also when `gpuwm gui` was
closed while it ran or stopped while starting it. `GPUWM_GPU_OWNER_TAG` sets the
tag (`gpuwm-gui` by default).

## The assistant

The assistant is optional, and it is off until you turn it on. Off, nothing is
downloaded, no model server runs and no card memory is used; the storm pages,
New forecast, your forecasts, Files and Machines work the same without it. The
sidebar reads **Assistant (off)** with a line saying to turn it on in
**Settings**, and Ctrl+K offers the same entry.

To turn it on, press **Turn on** in **Settings** or in the assistant panel, or
run `gpuwm assistant on`. Before you do, both show the model it would use for
your card, its size and its licence. Turning it on downloads nothing: it then
lists each file with its size and licence and waits for **Download**. You can
instead point it at a model server you already run. **Turn off** takes its
model off the card; **Remove its model** (or `gpuwm assistant off --remove`)
deletes what it downloaded.

Once on, the panel opens from **Assistant** in the sidebar, or from Ctrl+K and
then **Open the assistant**. Ask about a storm in the Weather Library, one of your
forecasts, or what fits this card. It answers from the Weather Library, the run
folder and the engine's fit check, and each answer says where it came from. A
decision it makes shows its options under letters, each with its probability,
or says the model server gives none. It can fill New forecast for you; nothing
runs until you press Start.

It runs on a language model on this computer, in one of two ways:

- a model server you already run, such as LM Studio, Ollama or llama.cpp: open
  **Settings** in the panel and give its address, or
  `gpuwm assistant use-endpoint URL --model NAME`. Ollama's server gives no
  probabilities, so its decisions show none;
- a model sized to your card that **Set up** (or `gpuwm assistant setup`)
  downloads once, after stating its size and licence, and checks against its
  published digest:

| Card | Model | Download | Licence |
|---|---|---|---|
| under 8 GB | Qwen3.5 4B, Q4_K_M | 2.6 GB | Apache-2.0 |
| 8 GB | Qwen3.5 9B, Q4_K_M | 5.3 GB | Apache-2.0 |
| 12 GB | Qwen3.5 9B, Q6_K | 6.9 GB | Apache-2.0 |
| 16 GB | Qwen3.8 27B, UD-IQ4_XS | 13.3 GB | Apache-2.0 |
| 24 GB | Qwen3.8 27B, UD-Q5_K_M | 18.4 GB | Apache-2.0 |
| 32 GB and up | Qwen3.8 27B, UD-Q6_K | 20.5 GB | Apache-2.0 |

Sizes are as the page shows them, where 1 GB is 2^30 bytes. **Settings** lists the same table
with each file's source, and the models you can bring on your own server with
their licences and why the assistant does not download them: Ternary Bonsai 2
27B, K2 Horizon 7B and MoVA 36B-A4B, G9v3 39B-A5B, Muse Glimmer 30B and
Qwen3.8 Flash Next. Some of them need a llama.cpp build other than the official
releases, as their rows say.

Turned on with no model set up, the panel says so at once and gives both ways.
`gpuwm assistant say "WORDS"` asks it from a terminal and prints the plan it
makes as a command to paste; it never starts a run. `gpuwm assistant unload`
takes the model off the card.

## The run folder is the truth

A run is a folder. `gpuwm run-plan` writes, in that folder:

- `run-manifest.json`: the process id and the path of every stream;
- `run-progress.json`: the current state, rewritten on every step;
- `events.jsonl`: every event from the start, one JSON object per line;
- the pictures, under `<domain>/<product>/<valid-day>/`.

The page adds `plan.json` (the plan it ran), `region.geojson` (the box you drew),
`gui-physics.json` (the physics you picked), `commands.log`, `gui-job.json` (the job it launched) and, after a Stop,
`gui-stop.json` (how it stopped). A downscale is the one exception: `gpuwm downscale`
claims only an empty folder, so the page keeps its record of that start (the job
and the command) in `.arwen-gui/starts/NAME.json` under the forecasts folder until
the finer forecast writes its own manifest and event log, and writes the command
into the parent forecast's `commands.log`. The page keeps nothing else. Reload the page,
restart the server, or start the server on a runs folder filled from a terminal,
and it reads the same answer from the same files.

A forecast is launched detached through the engine's one job manager, the same one
the MCP server (`arwen-mcp`) uses. A second forecast on the same card is refused with a sentence
that names the running one, and a new forecast refused that way leaves no folder behind, so the same
name starts once the card is free.

## Reattach from a terminal

A running forecast needs no page. From a terminal:

1. read `run-manifest.json` for the process id and the file paths;
2. read `run-progress.json` for where it is now;
3. read `events.jsonl` from the start for the history, then follow it
   (`tail -f events.jsonl` on Linux, `Get-Content events.jsonl -Wait` in
   PowerShell).

A forecast's Files page lists these three paths.

## Stop

`gpuwm run-plan` has no control file, so Stop acts on the process, and the page
says which way it did it:

- on Linux and macOS it sends Ctrl+C (`kill -INT`) to the run's process group. The
  forecast stops at its next step, writes its stopped event and exits 130. If it is
  still alive a minute later, the group is ended.
- on Windows it ends the run's process tree (`taskkill /F /T`), because a detached
  process there has no console a Ctrl+C could reach. The run folder keeps
  everything written up to then; the run shows as Stopped.

## Pictures

Every picture on the page is a PNG the Rust renderer (`rw_wrfbatch`, through
`gpuwm.rustwx`) wrote into the run folder. The server serves those files byte for
byte. It never draws, re-encodes or resizes a picture. If a run asked for no
pictures, or the renderer is not usable on this computer, the page says there are
none; `gpuwm run-plan --catalog` says why.

## Another computer

The server answers this computer only. To use a node's runs from your desktop,
start `gpuwm gui --no-open` on the node and open a tunnel:

```
ssh -L 8766:localhost:8766 NODE
```

then open the printed link on your desktop.

## Windows notes

- Edge, Chrome and Firefox all work. Nothing else needs installing.
- A forecast starts with no console window, in its own process group, so closing
  the terminal that runs `gpuwm gui` does not stop it.
- Port 8766 taken? The server tries the next nine and prints the one it used.

## Changing the words

The page's words live in `gpuwm/gui/copy/*.json`. Edit one and reload the browser.
`python -m gpuwm.gui.copy_lint` checks them.
