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
