# Measured CUDA preparation peaks (the preparation price's calibration)

`gpuwm/ingest/preparation_price.py` prices a CUDA preparation before its first
device allocation, and `MEASURED_PREPARATION_PEAKS` there holds the four peaks
below; `tests/test_preparation_price.py` holds the price at or above every one
of them. This page is where those numbers come from.

## What was measured

- Engine: integrate/2.8 at `7645b5125`, before the price existed. Every run
  pins `--preprocess-backend`, and each receipt's `selection` block records
  `requested: cuda` (or `cpu`) and `reason: named by the caller`.
- Card: one NVIDIA H100 80 GB (Linux), otherwise empty (4 MiB in use before
  every run). The largest peak is 37.5 GB, so the card constrained nothing.
- Input: HRRR pressure-level files, cycle 2026-09-27 21Z, f00 to f06 (7 forcing
  times), the same cycle for every run.
- Route: `gpuwm prep --source hrrr-prs` (the mapped front door `gpuwm go` takes
  for this source), with the inputs from
  `gpuwm fetch --source hrrr-prs --cycle 2026-09-27T21 --hours 6`.
- Instruments: card = nvidia-smi `memory.used` for the whole card, every
  0.2 s; pool = the CuPy default pool's exact high-water marks from an
  allocator hook read after every allocation (reserved = what the pool holds
  from the device, live = bytes in arrays still referenced); host RSS =
  `/usr/bin/time -v` maximum resident set of the largest process.

## The peaks

GB are 1e9 bytes.

| Case | Grid (nx x ny x nz) | Backend | Card peak | Pool reserved peak | Pool live peak | Where the peak is | Host RSS |
|---|---|---|---|---|---|---|---|
| 3 km CONUS | 1792 x 1024 x 55 | cuda | 37.46 | 36.80 | 31.10 | the model-state build of one forcing time (each of the 7 reaches it) | 32.7 GB |
| 6 km CONUS | 896 x 512 x 59 | cuda | 10.18 | 9.52 | 8.26 | the same, per forcing time | 16.9 GB |
| nest | the 6 km root + a 480 x 480 x 59 child, ratio 3 | cuda | 12.58 | 11.91 | 10.87 | initializing the child (parent residue held) | 17.3 GB |
| tiled nest | the nest with the child `tiles = { mode = "on" }` | cuda | 12.56 | 11.89 | 10.87 | the same as the nest | 17.7 GB |
| 3 km CONUS | 1792 x 1024 x 55 | cpu | no CuPy allocation | 0 | 0 | | 48.3 GB |

Card peak minus pool reserved is 0.66 GB on every CUDA run: the CUDA context
and the kernels outside the pool. The price charges the context at the
reference profile's 0.75 GiB.

## What the timelines say

- Forcing states are not held at once on this route. Each forcing time is
  built, copied off and released before the next (the pool drops back between
  times: the 3 km case to 19.2 GB reserved, the 6 km case to 4.7 GB). The peak
  is one forcing time's model-state build and does not grow with the number of
  times.
- Single-domain peak per nx x ny x nz cell: live 305 B (6 km) and 308 B (3 km);
  pool reserved 352 and 365 B. The two shapes agree within 1% live, so the live
  term scales with cells.
- The interpolation before the state build holds more reserve than live
  (3 km: 19.2 GB reserved against 8.0 live; 6 km: 4.7 against 2.5) and stays
  under the state-build peak on both shapes.
- Hierarchy: after the root's last time the pool keeps 6.72 GB live while the
  child initializes; the 13.6 M-cell child adds 4.15 GB live (305 B per child
  cell, the same per-cell term as a root) to 10.87 live and 11.91 reserved.
- Tiles do not change preparation on this route: the tiled child prepares
  exactly like the resident one.
- The 24 GB failure the price exists for follows from these numbers: the
  3 km case needs 36.8 GB reserved, well over a 24 GB card, where the 6 km
  case needs 10.2 GB.

## Per stage

From the engine's `GPUWM_PREP_EVENT` lines. Card, pool reserved and pool live
in GB.

| Stage | 3 km cuda: wall / card / reserved / live | 6 km cuda | nest cuda | 3 km cpu wall |
|---|---|---|---|---|
| Decode and compose source | 206 s / 0 / 0 / 0 | 225 s / 0 | 232 s / 0 | 223 s |
| Prepare root static fields | 303 s / 0 / 0 / 0 | 98 s / 0 | 112 s / 0 | 124 s |
| Initialize root forcing states | 146 s / 37.29 / 36.63 / 31.10 | 42 s / 10.18 / 9.52 / 8.26 | 167 s / 10.38 / 9.72 / 8.26 | 229 s |
| Publish prepared head | 24 s / 37.29 / 36.63 / 24.55 | 7 s | | 25 s |
| Initialize remaining forcing states | 259 s / 37.46 / 36.80 / 31.10 | 111 s / 10.18 / 9.52 / 8.26 | | 400 s |
| Initialize child domains | | | 37 s / 12.58 / 11.91 / 10.87 | |
| Write hierarchy artifacts | 48 s / 0.56 | 18 s / 0.56 | 39 s / 12.58 / 11.91 / 9.99 | 50 s |

The 6 km and nest CUDA runs shared the host's cores with the 3 km CPU run, so
their walls carry that overlap; their card and pool peaks are their own.

## The other routes, one card run each (A101)

The fit above is the mapped route's. The other rows of `PREPARATION_ROUTES`
carry the same residual and headroom unmeasured, so the routes with a term
the mapped runs never exercised were each run once on a card at a reference
shape. `MEASURED_ROUTE_PEAKS` holds the results, and
`tests/test_preparation_price.py` re-prices each row and holds the price at
or above its peak.

- Engine: integrate/2.8 at `cc3cb0ad6`. Every door that takes the flag ran
  with `--preprocess-backend cuda`; `gpuwm run` has no such flag, and its
  receipt shows `auto` choosing the card.
- Card: one RTX 5070 Ti 16 GB (Linux), held through the card lock for each
  run, so no other process was on it.
- Predicted: the price the door decided on, `selection.device_fit.need_bytes`
  in the receipt (the run route logs the same record).
- Actual: the preparation's own card memory from nvidia-smi compute-apps
  every 0.2 s, with every process it spawned (the native HRRR boundary
  workers) summed into it, up to the end of the preparation. On the two
  routes that go on to a forecast in the same process, the end was stamped
  by an instrument line: the downscale child just before its state is built
  (after the parent interpolation and the boundaries), and `gpuwm run` just
  before the physics attach. What follows is the forecast's and is priced
  there.

| Route | Shape | Predicted | Actual | Pool reserved at the end | Margin |
|---|---|---|---|---|---|
| hrrr-native (`gpuwm prep --source hrrr`, `--prepare-workers 2`) | 3 km 556 x 444 x 49, f00 to f02 | 6.947 | 4.496 | not read | +54.5% |
| downscale-child | 3 km parent 408 x 420 x 49 (16 fields) to a 1 km child 450 x 450 x 49 | 3.898 | 1.791 | 1.520 | +118% |
| experiment (`gpuwm run`, ERA5 from ARCO) | 12 km 500 x 400 x 49, mp 10, two forcing times | 4.656 | 4.261 | 3.907 | +9.3% |

GB are 1e9 bytes.

- hrrr-native: the price binds on the boundary-worker phase (the kept f00
  state, two analyses and two workers at a full context each, 6.95 GB); the
  card peaked at 4.50 GB while building the f00 state, which the price puts
  at 5.70 GB. Each spawned worker held 0.26 to 0.31 GB on the card, context
  included, against the 0.80 GB context plus strip the price gives it, and
  the main process held 3.1 to 3.3 GB beside them.
- downscale-child: the price counts every parent field twice on the parent's
  extent and levels plus the child's fields at both ladders. The card never
  held that much at once: the interpolation peaked at 1.79 GB, less than
  half the price. The price is safe and loose; the route decides `auto` on
  the CPU earlier than it needs to.
- experiment: the whole card peak is 9.3% under the price, but only because
  this card's context (0.35 GB) is under the 0.80 GB the price charges for
  it. The pool reserved 3.907 GB against a pooled price of 3.853 GB: 1.217
  times the itemized arrays, past the 1.20 headroom the mapped route
  measured (live was 2.70 GB, under the 3.21 GB itemized, so the residual
  is not what ran short). The experiment row now carries its own pool
  headroom of 1.25, which prices this run at 4.817 GB (pooled 4.014 GB,
  13.0% over the card peak).

Not measured here, with the reason:

- era5 (`gpuwm prep --source era5`, the FP64 `source_transform`): that door
  reads the CDS combined GRIB1, which needs a CDS key. The keyless ARCO
  download is NetCDF with specific humidity, and it feeds `gpuwm run` (the
  experiment row above), where no humidity conversion runs.
- gfs: already anchored by the 24 GB failure (1.17 times its itemization
  reserved).
- met_em: missing input. The measuring node holds no WPS metgrid set this
  line's met_em door accepts: the five sets there were made for another
  line of work, and the door's namelist importer refuses each one before
  it prepares anything (all five carry five namelist keys this line does
  not map, one pairs a 221 x 221 namelist with 101 x 101 met_em files, and
  all five write several assignments on one namelist line, whose later
  keys the importer reports missing). A measurement needs a WPS run
  (geogrid, ungrib, metgrid) of its own, at a reference shape.
- experiment-host-store: missing input. The route runs only when a
  `gpuwm run` case's streaming plan puts its state in a host store; the
  ERA5 case the experiment row was measured on was not kept on the node,
  so a measurement needs that case fetched again and run with a streamed
  `[tiles]` configuration.

`gpuwm check` prices a config whose forcing is its own `[case_data]` on the
experiment row's pool headroom (`preflight.config_preparation_route`), the
same 1.25 the `gpuwm run` door decides on; every other config keeps the
1.20 the other rows carry.
