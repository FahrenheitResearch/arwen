# Quickstart: a live radar-DA nowcast

**What this is.**  Point it at a WSR-88D site and it assimilates that
radar's real Level-II volumes into a small GPU ensemble, then runs a
free forecast past the last observation and draws the result on a real
map.  It is a **demo**: ten members, one 3 km domain, no radiation, a
GFS background, and no velocity dealiasing -- UNSCORED, outside any
registered campaign, and no skill claim is made or implied.  The figures
say so on every panel, and the numbers on them (>=35 dBZ column counts,
FSS at 30 dBZ / 27 km) are diagnostics to look at, not scores to quote.

If you want the deeper version of that honesty statement -- how this
configuration differs from the Warn-on-Forecast System, and why its FSS
does not sit next to a published WoFS number -- read
[`da-vs-wofs.md`](da-vs-wofs.md) before you draw conclusions from
anything below.  The reference page for the pipeline itself is
[`da-nowcast-demo.md`](da-nowcast-demo.md); this page is the one you
follow the first time.

Every command on this page exists on the branch it ships on and was
exercised there.  Where output is quoted in a block, that output is real
and was captured from this tree; the long-running commands were verified
through their argument parsing and their cheap stages rather than by
spending a card on them.

---

## 1. Prerequisites

### The machine

| | |
|---|---|
| Python | 3.11 or newer |
| GPU | an NVIDIA card, CUDA 12.x/13.x, **required** -- the ensemble and the analysis both run on the device |
| VRAM | 16 GB is enough for the shipped defaults; see the measured table below |
| Rust toolchain | needed once, to build the radar front door (below) |
| Disk | a few GB for static geography, plus the case directory |

Measured on a rented RTX 4080 (16,376 MiB, sm_89), full six-cycle demo
at 136x134x49 and 3 km, receipts under `evidence/16gb-frontier/`:

| members | wall clock, whole run | whole-card peak |
|---|---|---|
| N=4 | 431 s | 15,946 MiB |
| N=10 (the default) | 727 s | 15,888 MiB |
| N=20 | 1163 s | 15,796 MiB |

All three fit in 16 GB, and the peak barely moves with N because the
members advance one after another rather than side by side.  What grows
with N is time, not memory.

So a 16 GB card runs the shipped demo shape unchanged -- nothing has to
be turned down.  What it is worth doing is telling the memory preflight
which card it is actually sizing for, with `--vram-gib 16`, instead of
letting the wizard guess a tier.  The 16 GB run returned the 32 GB
card's answer to within the case's own noise; the comparison, its
caveats, and the Ada-specific findings that came with it are in
[`da-nowcast-demo.md`](da-nowcast-demo.md).

### Install

Use a source checkout.  A pip install alone is not enough for this
demo -- it drives `python -m tools.*` entry points, which only a
checkout has.  The NEXRAD front door itself is no longer a reason:
`rw_nexrad` ships in the prebuilt bridge bundle, so `gpuwm
fetch-bridges` stages it like every other binary and no Rust toolchain
is required to get one.

```bash
git clone https://github.com/FahrenheitResearch/arwen gpuwm && cd gpuwm
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '.[gpu-cu12,render]'   # or gpu-cu13
gpuwm fetch-tables
gpuwm fetch-geog
```

`install.sh` / `install.ps1` at the repository root do all of that in one
step; see [`install.md`](install.md).  `gpuwm fetch-geog` stages the
static geography every prepared case is built from -- without it the run
fails at the `prepare` stage, and `gpuwm doctor` reports
`MISSING WPS_GEOG`.

### Get the radar front door (`rw_nexrad`)

Without this binary nothing can read a radar volume, so there are no
superobs, no analysis and no nowcast.  `gpuwm doctor` checks for it by
name and reports what its absence blocks, so you do not have to
discover it from a launcher that refuses to print a plan.

It ships in the prebuilt bundle:

```bash
gpuwm fetch-bridges
```

From a checkout you can build it instead, which is also what an
unsupported platform does:

```bash
cd tools/rustwx
cargo build --release --locked --offline --bin rw_nexrad
cd ../..
```

That produces `tools/rustwx/target/release/rw_nexrad` (`.exe` on
Windows), which gpuwm then finds on its own from a checkout.  If you
keep it somewhere else, point `GPUWM_RW_NEXRAD` at the file.

If you built one a while ago, rebuild it rather than hunting for the
copy: the wrapper pins the exact `--abi` contract line it was written
against, and a binary predating the live-chunk route fails that probe.
Both binaries answer `--version` with the same string, so re-pointing
`GPUWM_RW_NEXRAD` at another old copy cannot help.

Confirm it, and see what you can point at:

```bash
tools/rustwx/target/release/rw_nexrad sites | head -20
```

141 sites in the vendored table, each with an id, a name and a position.
The site is always an argument; nothing in this pipeline has a favourite
one.

### Check the box, and read one line in particular

```bash
gpuwm doctor
```

On a healthy install the radar-DA line reads:

```
  ok      radar-DA eigensolver: jacobi kernel ok, cuSOLVER ok
```

**This line exists because of a specific failure that hit this project
twice** -- once on a box whose NVIDIA wheels were installed but
invisible to a compiled extension, once on a rented node whose CUDA
install simply shipped without cuSOLVER -- and it is worth understanding
before you hit it too.  Both times the surrounding evidence said the GPU
was healthy, because for everything except the factorisation it was.

The analysis has to factor small symmetric matrices on the device.  CuPy's
core loads independently of cuBLAS and cuSOLVER, so on a box where the
factorisation cannot run at all, *`import cupy` still succeeds and
elementwise GPU arithmetic still works*.  Everything looks green and the
DA path dies anyway.  `gpuwm doctor` therefore solves one tiny
eigenproblem, in a fresh subprocess, with both solvers, and prints which
of them worked.

Four outcomes, all of them non-blocking, because an install that never
runs data assimilation never touches either solver:

| line | meaning | what to do |
|---|---|---|
| `jacobi kernel ok, cuSOLVER ok` | both work | nothing |
| `jacobi kernel ok, no cuSOLVER` | **the default analysis is fine.** This project's own batched Jacobi kernel is what runs at `--solve-device cuda`, so cuSOLVER is optional | nothing, unless you want `eigensolver='library'` or more than 64 members |
| a cuSOLVER-works-but-kernel-failed line | analyses still run through `eigensolver='library'` | report it; the bundled kernel should build wherever cupy does |
| `no device eigensolver` | DA cannot run on this device; plain forecasts are unaffected | install the wheels below |

`gpuwm doctor --explain` prints the full remedy for each line.  For the
last row the remedy leads with a diagnosis rather than an install,
because the commoner fault is a library that is **installed but
unreachable**, which installing again does not fix:

```bash
python -c "import sys,pathlib;print([str(p) for p in pathlib.Path(sys.prefix).rglob('*cusolver*') if p.is_dir()])"
```

If that prints a directory, the library is present and the loader cannot
see it: on Linux put that directory on `$LD_LIBRARY_PATH`; on Windows
`PATH` is not enough, because since Python 3.8 a compiled extension
resolves its dependants through `os.add_dll_directory()` only.  If it
prints nothing, install it:

```bash
# CUDA 12.x box (put 13 in the pin on a CUDA 13 box):
pip install --no-deps "nvidia-cusolver==12.*" "nvidia-cublas==12.*" "nvidia-cusparse==12.*"
```

The CUDA major belongs in the version pin, never in the package name.
NVIDIA has deprecated the `-cuXX` spellings, `-cu12` as well as `-cu13`:
`nvidia-cusolver-cu13` and `nvidia-cusolver-cu12` both resolve as
deprecation tombstones that install cleanly, report success, and supply
nothing.  Never install a suffixed name.

`gpuwm doctor` reads the major off the driver and prints this line with
the pin already filled in, so you do not have to choose.

Background, including why the same box can answer "is cuSOLVER
available?" differently depending on what the process did first, is in
[`da_jacobi_eigensolver.md`](da_jacobi_eigensolver.md).

---

## 2. The one-command path

```bash
python -m tools.da_nowcast run --site KXXX --window-end latest --out CASE_DIR
```

That is the whole command.  `KXXX` is any id from the site table;
`latest` derives the window from the newest volume the feed has.

It runs eight stages and writes a receipt for each under
`CASE_DIR/receipts/`: survey the site, size a domain around the echo,
fetch the GFS background, prepare the case, run a georeference forecast,
build one observation file per cycle, run six 15-minute LETKF cycles
with ten members, and draw the gallery.  Then it hands the case to a
detached verifier and returns.

You can stop anywhere with `--stop-after`, which is the cheapest way to
prove your install works without spending a card on it:

```bash
python -m tools.da_nowcast run --site KXXX --window-end latest \
    --out CASE_DIR --stop-after survey
```

That prints one line and exits -- this is real output from this branch:

```
survey: lag 13.6 min, echo gates 836, motion None
stopped after survey (receipts written)
```

`lag` is how far behind real time the feed's newest volume is, `echo
gates` is the >=35 dBZ census the domain is centred on, and `motion` is
the centroid displacement between two volumes when there are two to
compare.

### What you get at the end

`CASE_DIR/gallery/index.html`.  Open it in a browser.  It opens with a
dark banner reading `DEMO-GRADE NOWCAST -- UNSCORED, outside any
registered campaign, not campaign evidence. No skill claim is made or
implied.`, and then up to five figures, all drawn on the same county
level basemap with one reflectivity scale:

| file | what it shows |
|---|---|
| `00-lead-nowcast.png` | the story in one row: last observation, final applied analysis, free forecast |
| `01-percycle-strip.png` | observed vs analysis at every applied cycle, then the free legs |
| `02-cycle-numbers.png` | innovation RMS against the never-analysed control, ensemble spread, core counts |
| `03-verification.png` | free-forecast frames beside the observed composite at the same valid time, same colour scale, with counts and FSS per pair |
| `04-scorecard.png` | the same verification as a table: observed, forecast, control |

Beside them the renderer writes `_manifest.json` and, once any frame has
been graded, `_verification.json` (`gpuwm-da.nowcast-gallery-verification.v1`)
carrying the metric definitions and per-frame numbers.  The last two
figures do not exist yet on a run that has just finished -- see
[honesty labels](#5-what-the-labels-on-the-figures-mean).

Everything is CLI plus versioned JSON.  `CASE_DIR/nowcast-receipt.json`
(`gpuwm-da.nowcast.v1`) names every output and carries the verification
state machine; polling that one file is enough to drive a progress
display without parsing any transcript.

---

## 3. The interactive path: draw a box

Prefer to choose the domain yourself?  There is a local launcher page.
**It is a thin driver, not a second implementation** -- every answer on
it comes from the shipped thing that owns the question, and the command
it would run is printed on the page before you press anything.

```bash
python -m tools.da_nowcast_launcher serve --work-root RUNS_DIR --open
```

That serves a single self-contained document on `127.0.0.1:8765`
(`--port` to move it).  There is **no authentication**, which is why it
binds locally by default; `--host` anything else is a deliberate choice.
The page has no CDN, no external font and no external script -- the
basemap is inlined from the same vendored Natural Earth and US Census
assets the gallery draws with, so it loads with the network unplugged.

**Drag a box over the map** and, before anything is launched, the page
answers with four things:

1. **the grid your box fits** -- from the `gpuwm domain` wizard, which
   also runs the memory preflight for your card.  If the wizard refuses,
   the page shows you its refusal and offers no override;
2. **which radars cover it** -- from the vendored site table, with each
   site's distance from the box centre and whether its coverage of the
   box is full or partial;
3. **an estimated cost per cycle**;
4. **refusals and warnings**, each with the number that produced it.

The same check runs on the command line, which is how it was verified
here.  Note the `--box=` form: a leading minus sign makes argparse read
the value as another option.

```bash
python -m tools.da_nowcast_launcher plan --box="-99.5,34.5,-96.5,37.0"
```

```
box 271 x 278 km
grid 94x94x49 at 3 km, dt 15 s
cost ~29 s per assimilation + ~105 s per forecast refresh = ~135 s per cycle (N=10)
     one measured configuration scaled by cell-steps; a short leg costs more than this predicts, ...
site KTLX   80 km from the centre, partial coverage
site KVNX  111 km from the centre, partial coverage
...
warning (radar coverage): no single radar covers the whole box; the corners beyond 250 km
     will carry no observations and the model runs free there
GO
```

### What the cost estimate means, and what it does not

It is **one measurement scaled by cell-steps**, and the page prints its
basis beside it: 3.15 s per trajectory per 900 s leg on a 132x132x49
grid at dt 15 s, on an RTX 5090, from a real cycled run.  The control
counts as one more trajectory.  Two known ways it is wrong, both in the
direction of optimism:

- **a short leg costs more than it predicts**, because per-leg model
  wiring is folded into the reference rather than separated from it;
- **large ensembles cost more than it predicts.**  The integration half
  really is linear in N -- measured at 3.06 s per member-leg at N=10,
  N=20 *and* N=36, three significant figures, because members advance
  serially and each costs the same.  The **solve** half is not: the same
  sweep measured 6.3 s, 10.1 s and 71.9 s at those three sizes, because
  the eigendecomposition is cubic per active point and hides behind
  launch overhead until N gets large.  The estimate models it linear.
  Receipts: `evidence/da-demo/ensemble-size-sweep/`.

Treat it as an order of magnitude that tells you whether to bother, not
a promise.

Before you reach for the members slider: **bigger is not better here,
and that is measured, not assumed.**  N=20 buys about +0.0018 FSS inside
an across-member scatter of 0.0062-0.0074 for +82% wall clock, N=36
scores *below* N=10, and N=4 costs about 0.007 FSS while saving no VRAM
at all.  The full ladder, with receipts and the reason (FSS is computed
on the ensemble mean composite, and averaging pulls peaks down in
forecasts that already under-produce echo), is in
[`da-nowcast-demo.md`](da-nowcast-demo.md#the-defaults-and-the-measurements-behind-them).
That is why 10 is the shipped default rather than a round number.

The estimate is also a gate.  A cycle has to finish before the next
radar volume lands or the analysis falls further behind every time
round, so the page compares your estimated cycle against the 5-minute
nominal volume interval: comfortably inside is `ok`, over it is a
warning, and several times over it is a **refusal** -- because that is a
property of the box worth knowing before the card spends an hour
proving it.  A box too small to hold a storm and the ground it crosses
is refused too, and a box with no radar within range is refused for
having nothing to assimilate.

Pressing the button starts the continuous daemon below on your box, with
the same card sizing the page showed you a verdict for.  You can also
just write the page out and look at it:

```bash
python -m tools.da_nowcast_launcher page --out plan.html
```

---

## 4. The continuous path: keep it running

The one-command path produces one nowcast.  The daemon keeps producing
them: it takes each new volume as the radar publishes it, assimilates
it, refreshes the free forecast and **redraws the gallery in place at
one stable path**.  Refreshing that one page is the whole user
experience.

```bash
python -m tools.da_nowcast_auto start --site KXXX --out RUN_DIR
```

It detaches and returns immediately, printing where everything is:

```
daemon started, pid NNNN
  status : RUN_DIR/auto-status.json
  gallery: RUN_DIR/gallery/index.html
  log    : RUN_DIR/auto.log
  stop   : python -m tools.da_nowcast_auto stop --out RUN_DIR
```

What it does on its own:

- **bootstrap** one prepared case per epoch, up to the georeference
  forecast;
- **catch up** by assimilating every volume between the background's
  init hour and now, oldest first -- a real spin-up on real data rather
  than an hour of free integration.  While it is behind, cycles run
  observations only: a 90-minute forecast off a state the next volume is
  about to supersede is work nobody will look at, and the gallery says
  which mode it is in;
- **cycle** on the radar's own rhythm once current -- 4 to 6 minutes in
  a precipitation VCP, not a fixed clock;
- **refresh** the free legs from the newest analysis and redraw.

The ensemble is alive across cycles: each one resumes the generation the
last wrote, so the covariance is the one the cycling built rather than a
fresh perturbation every few minutes.  Free legs branch off that state
and never become it.

**Epochs end.**  A prepared case carries a finite window of lateral
boundary data.  Approaching it, the daemon says so on the page, boots a
new epoch on a newer background, and re-initialises the ensemble -- a
generation written against one prepared case cannot be restored into
another, and the identity check exists to stop anyone pretending
otherwise.

### Watching it

```bash
python -m tools.da_nowcast_auto status --out RUN_DIR
python -m tools.da_nowcast_auto status --out RUN_DIR --json
```

The status file is `gpuwm-da.nowcast-auto.v1` and carries state,
cycles completed, volumes behind, the current epoch, the last cycle's
timing, a rolling history, and any active notice.  Before anything has
started there, you get a plain refusal rather than an empty file:

```
da_nowcast_auto: RUN_DIR\auto-status.json does not exist; nothing has started here
```

To watch it work in the foreground instead of detached, run the same
arguments under `loop` -- that is literally what `start` spawns.

### Stopping it

```bash
python -m tools.da_nowcast_auto stop --out RUN_DIR
```

```
stop requested; the daemon finishes its current cycle and exits.
Nothing is killed: RUN_DIR\stop-requested
```

It writes a request file.  Nothing is signalled and nothing is
terminated: the daemon notices at the top of its next loop, finishes the
cycle it is in, and exits with the gallery holding the last analysis and
free forecast it produced.  `--max-cycles` and `--max-epochs` set
ceilings up front if you would rather it stop by itself.

---

## 5. What the labels on the figures mean

The free forecast runs *past the last observation*.  Its grade does not
exist when the run ends -- reality has not happened yet.  Rather than
leave that implicit, every forecast panel is stamped.

**`PAST LAST OBS`** (orange, on wide panels `PAST LAST OBS --
unverifiable yet`) means exactly what it says: this frame's valid time
is in the future relative to every observation the model has seen, so
there is nothing to compare it against.  It is not a claim that the
forecast is bad, and it is not a claim that it is good.

**`VERIFIED`** (green, on wide panels `VERIFIED AFTER THE FACT -- see
verification row`) replaces it once the archive covers that frame's
valid time and an observed composite has been built for it.  From then
on the frame appears in `03-verification.png` beside its observed
counterpart, on the same colour scale, with counts and FSS.

The transition is automatic.  When a run finishes it hands the case to a
detached rolling verifier, which polls the archive, grades each frame as
its valid time gets covered, re-renders the gallery in place, and stops
with a verdict.  `--no-verify` opts out.  You can also drive it by hand:

```bash
python -m tools.da_nowcast watch  --case-dir CASE_DIR   # roll until graded
python -m tools.da_nowcast verify --case-dir CASE_DIR   # one pass, then stop
```

Its state lives in the run's own receipt under `verification`:
`pending`, `rolling`, `complete`, `incomplete`, or `disabled`.  The
renderer is the only place counts and FSS are computed and the verifier
copies the rows it publishes, so the receipt and the figures cannot
disagree.

The daemon adds one more stamp.  A late volume, a skipped cycle or a
failed stage is written into the status file **and** onto the gallery
page as a banner, so a page you left open tells you it is running on
older data instead of quietly showing you a stale picture.  Nothing is
padded, no volume is re-used as a new one, and the model is never
advanced past data it does not have.

---

## 6. Troubleshooting

These are the failures actually hit while getting this working, in the
order you are likely to meet them.

**`no rw_nexrad front door`**

```
da_nowcast_launcher: no rw_nexrad front door: # build the NEXRAD Level-II
  front door once, from this checkout's root: ... cargo build --release
  --locked --offline ... which gpuwm then finds on its own (or set
  GPUWM_RW_NEXRAD to it).
```

The error carries its own remedy, spelled for your platform.  You should
not normally meet it: this binary ships in the bundle `gpuwm
fetch-bridges` stages, and `gpuwm doctor` checks for it by name, so a
green doctor report now does mean you have it.  Run `gpuwm
fetch-bridges`, or build it from a checkout.

If it fails an ABI probe rather than going missing, **rebuild it -- do
not re-point `GPUWM_RW_NEXRAD` at another copy.**  The wrapper pins the
exact contract line it was written against, so a binary predating the
live-chunk route fails the probe instead of failing later at the first
fetch; and because every such binary reports the same `--version`, the
copy you would point at instead fails identically.  `gpuwm doctor` says
this in those words.

**The solve fails but the GPU looks fine**

This is the masquerade described in section 1: elementwise CuPy works,
`import cupy` works, and the factorisation cannot run.  Do not debug it
by inference -- run `gpuwm doctor` and read the `radar-DA eigensolver`
line, which answers the question directly by solving a real eigenproblem
in a fresh subprocess with both solvers.  Then follow the table there:
in particular, `no cuSOLVER` on its own is **not** a problem, because
the default analysis uses this project's own kernel.

**A large-member run dies with `CUDA_ERROR_OUT_OF_MEMORY`, or starves
something else on the card**

Cap CuPy's pool:

```bash
export CUPY_GPU_MEMORY_LIMIT=12884901888   # 12 GiB
```

CuPy's default pool grows to fit and does not give the card back, so a
run that *fits* can still leave a 32 GB card with nothing on it for
anything else.  Measured at `--members 36` on a 32 GB card: uncapped,
the run leaves **945 MiB** free device-wide; with the 12 GiB cap above
it leaves **17,709 MiB**, at identical wall clock and identical solve
time.  That is 17.7 GB handed back for no measured cost.

The 945 MiB is the number to pay attention to if you run the continuous
daemon and a second job on the same card.  That is how the run these
figures come from ended after four legs: nothing was wrong with the run
itself, there was simply nothing left on the card for the daemon beside
it.

Read the cap as a ceiling the workload sits under, not as a requirement.
36 members do **not** need 14.9 GB; that figure is only where a 12 GiB
cap plus the resident trajectory happen to land, and measured live
demand on the same run is 3.9 GB.  12 GiB is a value proven to work with
large headroom rather than an optimum: smaller caps are still being
measured, and if you are tighter on memory it is worth trying one.

**Provenance.**  These two numbers were measured on a rented node and
their receipt is committed there, not in this tree, so nothing under
`evidence/` backs them today.  They are reported here because the
failure they describe is the one a first-time large-ensemble run is
most likely to hit; read them as a measurement whose receipt has not
landed rather than as a receipted claim.

**The archive is behind real time**

```
da_nowcast: KXXX: newest archived volume KXXX20260805_085732_V06 is 13.9 min old
(ceiling 1 min); pass --allow-stale to nowcast from a stale feed anyway
```

The front door measures how far behind the feed is and refuses to start
a *nowcast* on data that is not now.  The default ceiling is 15 minutes
(`--max-lag-seconds`).  This is expected behaviour, not a bug: the
archive bucket only gains a volume file when the volume **ends**, so its
newest object is on average half a volume period old and at worst a
whole one.  Polled every 30 s across more than one volume period, the
archive's own lag ran 18 / 197 / 406 s (min / median / max) on a 410 s
VCP and 4 / 95 / 186 s on a 197 s one -- so *when* you ask matters as
much as anything you configure.  Receipts: `evidence/da-demo/live-feed/`.

Your options, in order: wait for the next volume; raise
`--max-lag-seconds`; or pass `--allow-stale` to run anyway, which is the
right answer when you are reproducing a past event rather than nowcasting
a present one.  If instead you get `no volumes in the last N minutes --
site down, id wrong, or archive far behind`, check the site id against
`rw_nexrad sites` first.

The observation builder can also read the real-time chunk feed, which
publishes the same bytes as they are collected rather than when the
volume closes; the finished volumes on both routes are byte-identical.
The front door does not expose that choice today -- it uses the builder's
default (`auto`: prefer live, fall back to archive, recording why).
Driving `tools/obs_radar_grid_build.py` yourself with `--source live` is
the opt-in route, and mid-scan partial volumes are a further opt-in
(`--allow-partial`, off by default) documented in
[`da-nowcast-demo.md`](da-nowcast-demo.md#partial-volumes).

**The daemon stopped saying the run root moved**

```
stopped: the run root moved underneath the run -- the run root's HEAD moved
from abcd1234 to ef567890 while the daemon was running.  Runs and commits
belong in different worktrees.  The case and the gallery are intact; start
a new daemon against a tree nobody commits into.
```

The daemon fingerprints its run root's git HEAD and uncommitted-file
count when it starts, and stops loudly if either moves.  It is not a
purity claim -- it is there because a commit landing in the tree a
forecast is executing from silently changes what the forecast is.  If
you develop in the same checkout you run from, you will hit this.  The
fix is `--run-root` pointing at a worktree nobody commits into:

```bash
git worktree add ../gpuwm-run HEAD
python -m tools.da_nowcast_auto start --site KXXX --out RUN_DIR --run-root ../gpuwm-run
```

The case and the gallery are intact when this fires; only the daemon
stopped.

**`MISSING WPS_GEOG` / the run dies at `prepare`**

`gpuwm fetch-geog`.  Static geography is not bundled.  Note also that
prepared cases are **host-bound** by receipted finding: prepare on the
box that will run, and do not point the front door at a case prepared
elsewhere.

**A cycle failed but the daemon kept going**

By design.  A failed cycle leaves the analysis clock where it was, so
retrying is right for a transient fault; the daemon says so on the page,
retries on the next poll, and gives up after
`--max-consecutive-failures` in a row.  A failed *render* is treated
even more gently -- it is a failed picture, and the analysis behind it
is finished and carried forward, so the gallery keeps the last drawing
that worked and says which cycle it came from.

---

## 7. Beyond the defaults

Two capabilities are optional, off unless asked for, and each changes
what a run costs:

- **A fine one-way nest over the free forecast.**  It runs inside the
  parent over the free legs only, inheriting the analysis rather than
  resampling it, and the parent is proven bitwise unchanged by its
  presence.  Reached with the `--nest-*` flags on
  `tools/da_cycle_prepared.py`; cost scales as ratio^3 per covered
  parent cell times the number of members carrying it (default: the
  control only).  See [`da-nested-forecast.md`](da-nested-forecast.md).
- **The eigensolver, which has no CLI flag.**  At `--solve-device cuda`
  and 2 to 64 members, this project's batched Jacobi kernel is simply
  what runs, which is why cuSOLVER is optional for radar DA.  Because
  that is a default rather than a choice, every leg of every cycle
  report names the solver that produced it under `filter.eigensolver`,
  beside `filter.max_jacobi_sweeps` (0 under the library solver; a value
  climbing toward the sweep cap means the localised matrix is worse
  conditioned than expected).  The two solvers agree to about 1e-11
  relative, **not** bitwise, so a receipt banked before this became the
  default reproduces only under `eigensolver='library'`.  See
  [`da_jacobi_eigensolver.md`](da_jacobi_eigensolver.md).

Sites, times, boxes and buckets are all **arguments**.  No radar-site
name appears anywhere in this machinery, its defaults or its
identifiers, and the test suite checks that mechanically.
