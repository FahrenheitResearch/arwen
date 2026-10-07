# Nowcast frames: `gpuwm-obs.nowcast-frames.v1`

Status: next release (2.8.7+), unproven until the box test.

This is the plug point between a 2D reflectivity source and WOOF's radar
heating. Any such source writes these files: a machine-learning nowcast
(StormScope today), the observed MRMS composite (`rw_mrms frames`, for the
oracle arm), or a future nowcast. The window adapter
(`rw_nexrad grid-composite`) reads only this format, so adding a source
means writing these files. It needs no new adapter code path.

## Layout

```
<root>/nowcast.json                     receipt
<root>/inputs.json                      every input object read, with its stamp
<root>/<YYYYmmddTHHMMZ>/refc.f32        one frame per valid time
<root>/<YYYYmmddTHHMMZ>/<other>.f32     optional extra variables of the same shape
```

- A frame file is raw little-endian float32 in C order, shape
  `[members, ny, nx]`. Row `j`, column `i` is cell `(j, i)` of the lattice
  below. There is no header.
- Values are dBZ. NaN means no coverage (outside the radar network or the
  satellite view). Any finite value below 5 dBZ counts as clear air:
  MRMS writes -99 for observed clear air, and StormScope's clear air comes
  out near -10.
- The directory name is the frame's valid time in UTC, to the minute.
- A writer creates each file under a `.partial` name and renames it into
  place only when every member is written, so a reader never sees half a
  frame.

## `nowcast.json`

| Key | Meaning |
|---|---|
| `schema` | `gpuwm-obs.nowcast-frames.v1` |
| `status` | `READY` is the only consumable status. `RUNNING`, `REFUSED` and `FAILED` are written by the runner so a marker-driven chain can see where a run stopped; `refusal` then says why. |
| `source` | `kind` (`nowcast` or `observed`), `id` (for example `stormscope-3km-10min` or `mrms-composite`), `package`, `revision` (the weights revision), `code` (package versions) |
| `sampler`, `seed` | for a nowcast: the diffusion sampler settings and the seed |
| `issue_time` | the time the nowcast is issued from, UTC |
| `latest_input_time` | the newest stamp of any input object read |
| `causal` | true only when every input stamp is at or before `issue_time` |
| `variable`, `units`, `members` | `refc`, `dBZ`, ensemble size |
| `lattice` | the source grid, see below |
| `frames[]` | `{valid, lead_minutes, file, shape, dtype, sha256, extra}`; `extra` maps each further variable to `{file, sha256, dtype}` |
| `inputs` | `{file: "inputs.json", sha256}` |
| `history` | for a nowcast: the history-slot policy, the slots, the satellite, how lightning entered, the seed rule |
| `timing` | seconds for fetch and load, seconds per step (the first step includes compilation), device and CUDA version |
| `peak_device_bytes` | the largest device memory the run reserved |

`dtype` is `float32-le`. `lead_minutes` is the frame's valid time minus
`issue_time`, and frames are listed in increasing lead with no gap.

### Lattice

Either a Lambert conformal lattice on a sphere:

```
{kind: lambert, truelat1, truelat2, stand_lon, ref_lat, ref_lon, earth_radius_m,
 nx, ny, dx_m, dy_m, x0_m, y0_m, row_order, corners_latlon}
```

- Projection coordinates are metres from the projection origin at
  (`ref_lat`, `stand_lon`); `ref_lon` repeats `stand_lon`. With cone
  constant n, `x = rho sin(n (lon - stand_lon))` and
  `y = rho0 - rho cos(n (lon - stand_lon))`.
- Cell `(j, i)` has its centre at `x = x0_m + i dx_m`, `y = y0_m + j dy_m`.
  `dx_m` and `dy_m` are signed. `row_order` is `south_to_north` when
  `dy_m` is positive and `north_to_south` when it is negative.

or a latitude and longitude lattice:

```
{kind: latlon, lat0, lon0, dlat, dlon, nx, ny, row_order, corners_latlon}
```

with cell `(j, i)` at `lat0 + j dlat`, `lon0 + i dlon`.

`corners_latlon` holds the centres of the four corner cells as
`[lat, lon]` in degrees, longitude in [-180, 180):
`j0_i0`, `j0_in` (row 0, last column), `jn_i0` (last row, column 0) and
`jn_in`. A reader recomputes them from the lattice numbers and refuses the
lattice if any lands more than 100 m away.

The adapter maps every WOOF mass cell through latitude and longitude into
the source lattice, using each side's own projection and earth radius.
StormScope's grid is HRRR's on a 6,371,229 m sphere and WOOF's WPS grid
uses 6,370,000 m, so two grids are never taken to be the same lattice
because both say "HRRR 3 km".

## `inputs.json`

Schema `gpuwm-obs.nowcast-inputs.v1`. `objects[]` lists every object a data
source read: `stream`, `bucket`, `key`, `stamp`, `satellite`, `slot` (the
history slot it was read for), `sha256` and `bytes` of the local copy the
package decoded. `slots[]` gives, for each stream and slot, what it
resolved to and how far off nominal it was.

Stamps by stream (StormScope row in `tools/nowcast/sources.toml`):

| Stream | Stamp | Why |
|---|---|---|
| ABI MCMIP (satellite) | scan end | the frame holds pixels up to the end of the scan |
| MRMS composite and base | product time in the name | the file is the analysis at that time |
| GLM LCFA (lightning) | file start | the gridded source keeps only events before the bin end, so a file that starts at the issue time contributes no event |

## The StormScope runner

```
<venv-e2s>/bin/python tools/nowcast/run_nowcast.py --source stormscope-3km-10min \
    --issue 2026-10-01T18:00Z --minutes 120 --members 8 --seed 1 --device cuda:0 --out ROOT
```

- It runs in its own venv, built by `tools/nowcast/setup_venv.sh` from
  `tools/nowcast/requirements-stormscope.txt`. The engine venv never
  imports it. The pinned torch 2.13.0 has no CUDA 12.8 wheel and its
  CUDA 12.6 wheel has no Blackwell kernels, so the box driver must
  support CUDA 13.0 (driver 580 or newer). `setup_venv.sh` checks the
  driver first and then proves torch sees the card, carries its
  architecture and loads NATTEN, all before the 6 GB weight download.
- `--plan-only` checks the source row, the weights pin, the satellite and
  the history plan, prints them, and stops. It needs no model package.
- `--batch-members N` runs the ensemble in rollouts of N members when all
  of them do not fit on one card. Each rollout seeds with the seed plus
  the index of its first member.
- History: six frames at 10 minutes, ending at the issue time. Each slot
  takes the newest object that is complete at or before the slot. MRMS
  files carry stamps about 40 s past the even minute, so the 18:00 slot
  reads the file stamped about 17:58:40, not the one stamped 18:00:40.
  The runner drops every radar candidate stamped after the slot from the
  source's nearest-first list, so a missing file falls back to the one
  before it, never to a file after the slot.
  ABI reads the scan that ends before the slot, and GLM reads the 5-minute
  bin that ends at the slot. Every stream is therefore shifted by at most
  5 minutes against the model's training convention, which took the
  nearest object; the box test measures what that costs.
- Observed lightning enters only in the six starting frames. The rollout
  then carries the model's own predicted lightning.
- The package fetches and decodes its own satellite, radar and lightning
  inputs. The runner adds no decoder; it taps each source's read call so
  every object is on the ledger. Swapping in arrays decoded by `rw_goes`
  and `rw_mrms` changes nothing in this contract.
- Output is a raw dump of the model's output tensor with two sets of
  cells set to NaN: cells outside the model's valid mask, and cells where
  any history slot of the observed composite had no radar coverage
  (NaN, or MRMS -999). StormScope fills unobserved input cells with clear
  air, so without the second mask a radar outage would come out as a
  confident clear-air nowcast. The receipt's `history.no_coverage` block
  counts those cells per slot. `refc` is the frame; `refc_base` and
  `glm_density` are extras. The heating does not read the extras in v1.
- A lightning bin is accepted only when the GLM files actually read cover
  it with no gap longer than `max_gap_seconds` (21 s, one missing 20 s
  file); a longer outage would be counted as no lightning.
- The rollout holds the engine's own lock on its card (the same
  `GPUWM_GPU_LOCK_ROOT` file a forecast arm takes), and takes it only
  after the CPU fetch and the weight load, so the card is never held
  idle during preparation.

### Refusals

Each refusal stops the run before the card is used where it can, writes a
`REFUSED` receipt, and exits with status 3.

| Refusal | Breakage it prevents |
|---|---|
| an input stamped after the issue time | a hindcast scored as a forecast would overstate skill |
| two history slots resolving to the same object or stamp | the model would be told the storms stood still |
| a slot more than 5 minutes off nominal | the model would be told the storms jumped |
| a slot with no object, or two objects for one single-object slot | an empty frame would read as clear sky, or the ledger could not say which object fed the frame |
| an object no stream describes, or one read outside every slot | its time cannot be shown to be causal |
| a weights revision other than the pinned one | the receipt would name weights that did not make the frames |
| a satellite that was not GOES-East at the issue time, or an issue time from 2025-04-04 to 2025-04-07 | the model would be conditioned on the wrong scan geometry (the package's satellite and lightning sources disagree on the handover date) |
| model variables or input times that differ from the source row | channels or history frames would be fed in the wrong order |
| a lattice whose spacing is not uniform, or whose corners do not reproduce | the adapter would heat the wrong columns |
| `--minutes` not a multiple of the step | the last window would end between frames |
| an issue time off the 5-minute grid, a history request before the input archive (2020-10-14), or a history window that crosses a GOES-East period | the satellite or radar source would fail mid-fetch on the rented box, or the history would mix two scan geometries |
| a lightning bin the read files do not cover (a gap over 21 s) | an outage of the lightning source would be read as no lightning |
| a history field with no finite value | the model would refuse the non-finite input only after the weights were on the card |
| a card another run holds | two runs on one card would both run out of memory |
| a root whose receipt is already READY | frames would change under a receipt something may already have read |

## A NetCDF nowcast as a frames root

A runner that writes NetCDF instead of this contract (a 2D `refc` variable
in dBZ over `[time, y, x]` or `[time, member, y, x]`, the time coordinate
carrying CF `units` like `hours since 2026-10-01 19:00:00`, and cell-centre
`lat`/`lon` beside it) is converted by `rw_nowcast_frames from-netcdf`:

```
rw_nowcast_frames from-netcdf --netcdf FILE --out ROOT --source-id ID \
    --lattice lambert:truelat1=..,truelat2=..,stand_lon=..,ref_lat=..,earth_radius_m=.. \
    [--receipt RUNNER.json] [--issue TIME] [--causal true|false]
```

`--lattice` names the projection the grid is fitted in (`lambert:` with
its five numbers, or `latlon`); the lattice block is fitted from the
file's own coordinates and refused when the grid is not rectilinear in
that projection or its corners do not reproduce. The issue time and the
causal flag come from the flags, else the runner receipt, else the file's
`init` and `causal` attributes; a receipt that disagrees with the file's
attributes is refused, and so is a frame set that cannot say both.
`_FillValue` and `missing_value` become NaN (no coverage). The receipt's
`objects[]` become `inputs.json`; `sampler` is null and `history` carries
only the receipt's causal rule and violations, because an ad-hoc runner's
receipt records nothing more. The in-tree runner writes the contract
directly and needs no converter.

`rw_nowcast_frames` ships in the platform wheel's bridge bundle beside
`rw_compare` (and `gpuwm fetch-bridges` stages it on any other install);
`GPUWM_RW_NOWCAST_FRAMES` names a build of your own, and a checkout's
`cargo build --release` in `tools/rustwx` builds it with its siblings.
Python callers resolve it with `gpuwm.rustwx_lanes.require_nowcast_frames_bin()`,
which refuses a build whose `--abi` line differs, and `gpuwm doctor`
reports it on the `nowcast frames converter` line.

## License

The StormScope weights are under the NVIDIA Open Model License. Outputs
are not Derivative Models and NVIDIA claims no rights in them. The runner
pulls the weights from Hugging Face at run time. If the weights are ever
baked into a WOOF image, the image needs a copy of the license and a
NOTICE line: "Licensed by NVIDIA Corporation under the NVIDIA Open Model
License".
