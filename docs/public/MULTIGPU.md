# One forecast across cards

`[devices]` splits a prepared forecast into permanently resident slabs, one
domain or every grid of a nested tree. Each slab is built at its own
compute-window size. A split domain stays in one pinned host store and is never
restored onto a card whole. The slabs of a domain share one adaptive clock and
exchange a wide halo after each model step.

```toml
[devices]
count = 2
# grid = "1x2"
# ids = [0, 1]
# transport = "auto"
# domains = [1, 2]
```

`count` is a positive slab count. Its default, 1, takes the existing
one-card road. `grid` is `GYxGX`, with a product equal to `count`. When omitted,
the planner chooses the factor pair with the shortest seams. `ids` gives one
visible CUDA card id per slab, in row order; omitted, it is `0..count-1`.
Repeated ids are allowed, and their memory prices are summed on that card.

`gpuwm go CONFIG --devices N` and `gpuwm sim ... --devices N` replace the table's
count and print which count won. A count that contradicts explicit `grid` or
`ids` is refused. `--devices 1` selects the one-card road when the table has no
split-only keys. Remove those keys when turning a configured split off.

A prepared bundle binds its experiment TOML byte for byte, so a split for an
existing bundle travels beside it: `gpuwm sim PREPARED --devices-table JSON`
takes the table as JSON (for example `{"count": 2, "ids": [0, 1]}`), for a
single domain or a tree. Neither relay changes the restart identity.

## Nested trees

On a tree, every grid is split by the same `count`, `grid` and `ids` unless
`domains` names the grids to split. A grid left out runs resident on the first
card of `ids`, so `domains = [2]` splits a large nest under a parent that fits
one card. A split parent needs its nests split too (see the refusals below).

- The parent forces its nest every parent step. A split nest's slabs each take
  their own window of the rolling boundary tables the nest coupler fills, and a
  slab on another card than those tables copies its window across with a
  device-to-device copy, which needs no peer access.
- Two-way feedback writes into a split parent through its host store, which
  goes back to the parent's slabs before the parent's next step.
- The coupler reads a split grid through its host store, so each parent step
  drains the split parent and nest to the host and the next steps copy them
  back to their slabs. That is correct and slow: on the one-card proof tree
  (2.25 km 288 x 288 parent, 750 m 216 x 216 nest) it was 250 drains and 250
  copies back per grid over 2 h. Reading only the coupler's windows from the
  slabs is the next step for nested speed.
- A split nest's halo covers the same seam fiction a specified root's does:
  the dependency radius plus `spec_zone` (one cell at the standard
  `spec_zone = 1`). The relaxation frame is masked on a seam side, so it is
  not paid. Measured on a 3 km HRRR-lattice crop split 2x1: halo 20 against
  the former 28 (and 22 and 19 as probes), every hourly history
  byte-identical over 12 h across all four and the one-card run.

Refused by name on a tree, before anything restores:

- `[relocation]` (moving or following nests): a move rebuilds the nest from
  its live parent, and nothing rebuilds a split grid's slabs.
- A split grid that starts after the run does: its start rebuilds a `[tiles]`
  store domain, not slabs. Leave it out of `domains` to run it resident.
- Trees initialized by an external door (wrfinput, met_em): those doors
  restore every grid resident first.
- A split parent over a resident nest: the nest coupler serves a resident
  nest from its parent's resident arrays, and a split parent keeps only host
  store arrays. Add the nest to `domains`, or leave the parent whole.

`gpuwm go TREE --devices N` and a `[devices]` table in a tree config run the
tree split: the plan names each grid's road (`d01 split 1x2, d02 split 1x2`, or
`d01 resident on card 0`), the memory gate prices every card before the
download, and the count reaches the tree runner. `gpuwm check TREE --devices N`
prints the same per-card prices, grid by grid. A bundle already prepared runs
split through `gpuwm sim PREPARED --devices-table JSON`, as a single domain does.

## What a card needs and what it is given

Each card's admission line prices state, seams and the loader template at the
narrowest MYNN workspace width (8,192 columns per rank). That price is the one
that refuses a forecast. Room left on the card then goes to a wider MYNN
workspace (up to 98,304 columns per rank), which makes the boundary layer
faster and never changes a byte, and then to frame snapshots. The line names
the widened part (`of which N GiB is MYNN workspace widened into free
memory`). A closing `devices:` line gives both figures per model column:
`required` for capacity planning and `admitted` for what this box holds. The
receipt carries both as `footprint`. Measured for one 1 km grid (1800 x 1400 x
50, HRRR physics), the required figure is 60.9, 65.6, 68.7 and 82.9 KB per
column on 2, 3, 4 and 8 cards. On 8 x RTX 5090 the admitted figure is 101.8,
because the MYNN workspace widens into the rest of each card. More cards cost
more per column for three reasons: the halo cells each slab repeats (27 cells
wide when those figures were taken; 6% extra columns on 3 cards, 13% on 8),
the fixed per-card tables and allocator headroom, and the seam buffers. The
halo no longer pays the relaxation frame (19 cells on that grid against the
27, about 30% less halo), so the repeated columns and the price fall with it.

## Output

A history frame does not stop the model to copy the whole store. At an output
time each slab snapshots only the frame's own fields (and the reflectivity) on
its compute stream. A separate stream copies that snapshot to the host store
while the next step runs. The snapshot buffers are reused after their copies
finish. They are bounded at 4 GiB per slab, included in memory admission. A
frame exceeding that bound copies directly from the live arrays and fences
the next step until the copy finishes.
The writer thread then derives T, P and PSFC on the host and writes the file
while the model steps on. The next copy into the store waits until the writer
has released the previous frame. Running maxima the frame resets (`UP_HELI_MAX`)
are zeroed on the slabs.

A reader that needs the whole store (the final digest, a restart, a nest
coupler) still drains it, and the next step copies it back to the slabs. The
forecast receipt's `devices.output_road` counts both roads: frame downloads,
their bytes, the seconds the stepping thread spent issuing them and the writer
spent waiting, retained snapshot bytes, direct-copy fallbacks, whole-store
drains, and copies back.

## Transport

`auto` chooses peer copies when both directions can reach the other card and
CUDA staging otherwise. When the run's cards sit on more than one NUMA node
(two sockets), `auto` stages every cross-card pair: on a two-socket box the
2x2 exchange's concurrent peer copies, mixing in-socket and cross-socket pairs,
took 245 ms per step against 15 ms staged. The receipt names that reason on
each staged pair. `peer` requires direct access and refuses an unreachable
pair. `staged` leaves peer access off for CUDA staging. `host` explicitly copies
through pinned host buffers. Slabs on the same card use device copies. Read the
actual per-pair path in the receipt rather than treating the requested transport
as a measurement.

## Host threads

Each slab steps on its own host thread, which issues that card's kernel
launches, readbacks and halo copies. On a standard CPython build those threads
share one interpreter lock and take turns: CuPy gives the lock up around every
CUDA call, about 3,900 times per slab in an ordinary HRRR step and about
140,000 in a radiation step, so adding cards adds lock traffic. On a two-socket
4-card box the radiation steps took 13-25 s under the lock and 5-8 s without it.

Run multi-card forecasts under a free-threaded Python 3.14 (`python3.14t`).
`install.sh` and `install.ps1` make `.venv` on it by default: a `python3.14t`
already on PATH, or one uv finds or installs (`--python` / `-Python` picks
another interpreter). For a pip install, make the environment with
`uv venv --seed --python 3.14t` and install into it with
`pip install --prefer-binary 'gpuwm[gpu-cu13]'`. CuPy, NumPy and netCDF4
publish free-threaded wheels. `cftime` publishes one for 1.6.5 but not 1.6.6,
so the installers pass `--prefer-binary`, and a pip install needs it too:
without it pip builds cftime 1.6.6 from source, which needs a C compiler. The render
extra's `wrf-rust` publishes none, so it builds from source with the Rust
toolchain the installers already require. The slab threads then run at once.
Importing netCDF4 would switch the lock back on, so the `gpuwm` command line
and both prepared runners (single domain and domain tree) re-run once with
`PYTHON_GIL=0` on a free-threaded build (every netCDF4 session in the
process is already serialized by its own lock). On Windows, where `exec`
cannot keep the process, the re-run is a child bound to the first process by
a job object, so stopping either one stops both. An explicit `PYTHON_GIL` in
the environment is respected, including `PYTHON_GIL=1`. A multi-card run on a
locked interpreter prints one `[devices]` line saying so; the run continues.
`gpuwm doctor` reports which build the environment runs.

On a host with more than one NUMA node, each slab thread is bound to the CPUs
sysfs lists as local to its card. Nothing is bound on a one-node host or when
the firmware reports no locality. The receipt's `host_threads` records the
interpreter and whether its lock was on, and `rank_placements` records each
slab's card, node, CPUs and whether the binding took.

## Admission and receipt

Split grids with `sf_surface_mosaic = 1` or `slope_rad = 1` are refused before
restoring or allocating ranks. Noah mosaic needs land-use tiles built from
`LANDUSEF`, which the rank factory does not build. Terrain-shadow radiation
needs full-domain terrain and held radiation state that slabs do not carry.
These schemes remain available on resident grids. Leave an affected grid out
of `[devices] domains`, or run the experiment without a split.

`gpuwm check --devices` requires a declared budget or a measured budget for
every selected card. Missing budgets return exit 2 with a named refusal and
the `--budget-gib` remedy, including when the output is JSON.

The memory gate prices each compute window with the resident estimator and the
same physics and retained forcing intervals. It sums ranks on repeated card ids,
adds each channel's send and receive buffers, and adds the retained loader
template on the first card. Host pricing includes the pinned store, geography,
retained forcing tables, and explicit host staging buffers. Before preparation,
the host carrier census is a conservative bound; at the forecast stage the
prepared store supplies its actual array inventory. On a tree, each split grid
is priced that way, each resident grid as a resident domain on the first card,
and the sums are compared with every card's free memory.

`gpuwm check CONFIG --devices N` prices the run split into `N` slabs and prints
every card's envelope and the host store before a run. `gpuwm check` on a config
with `[devices]`, a devices-enabled `go --dry-run`, and the forecast stage print
the same per-card totals and verdicts. The forecast's `devices` receipt contains
options, resolved grid, rank shapes, cards, halo, admission terms, actual
transport, the stepper report, clock history, per-card pool high-water samples
after committed steps, and the output road. Samples can miss transient peaks.
A tree's receipt carries the same per split grid.

## Gates

`python -m tilestream.ranks_gate` runs the cross-card gates: `gate` (identity
against the unsplit domain over grids, step modes and rungs), `controls` (each
a deliberate fault that must change the answer), `loop` (the real adaptive
driver), `transport` and `bench`. `bench --devices A --devices B` measures the
split on cards A and B against card A alone; without `--devices` both slabs
share card 0, which measures what the split itself costs. The wrong-card
control hands every slab the first card's cached device tables, allocated from
the first card's stream-ordered memory pool, which no other card can read even
after CuPy turns peer access on by itself. On two cards it must stop on a CUDA
error or change the answer; on one card it must leave the answer unchanged,
which shows the rebinding alone changes nothing. It runs in a child process
because that error poisons its process.

## Refusals

An enabled `[tiles]` table beside `count > 1` is refused: one domain has one road,
and two planners would price different stores. Per-domain `devices` tables,
offline children, downscale, and routes without the ranked builder are refused
before a silent one-card allocation can occur. Nonexistent card ids are refused
before rank allocation and at the download door's probe. An interior too narrow
for the halo or boundary frame is refused because it cannot serve a seam from a
unique owned interior.

## Byte identity gate

The required contract is identical wrfout bytes, restart bytes, and canonical
state digests from the same prepared case on the same card. The release gate must
compare complete files by SHA256 and carriers bit for bit. CPU front-door tests
prove table transport, default authority bytes, fingerprints, restart exemptions,
and admission arithmetic. They do not prove forecast byte identity.

The one-card proof configuration exercises the split without a second card:

```toml
[devices]
count = 2
ids = [0, 0]
```

It buys no memory capacity: both rank prices are added on card 0. With no table,
authority bytes, fingerprints, and plans retain the base behavior, and
`tests/test_devices_door.py` holds that front-door contract.
