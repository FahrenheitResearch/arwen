# Machine-learning datasets from a run: `gpuwm ml-export`

`gpuwm ml-export` turns a run's history files into a training dataset:
one Zarr store per domain, on standard pressure levels, with the variable
names, units and attributes ERA5 and WeatherBench 2 use. Every grid opens
with `xarray.open_zarr`. Regular latitude-longitude exports with WB2 names
and forecast layout have WeatherBench 2's evaluation dimensions without
variable renaming; projected native grids retain `y` and `x` dimensions
with two-dimensional latitude and longitude coordinates.

It reads any wrfout-shaped history file: a regional run's own, the frames a
hex run converts its mesh history into, and a global run's tapes. Nothing is
recomputed: the dataset is a conversion of what the run wrote.

```
gpuwm ml-export my-run/run/wrfout --out ml/my-run                 # WB13 levels, the run's own grid
gpuwm ml-export my-run-wrfout.zip --out ml/my-run --zip           # straight from the run page's ZIP
gpuwm ml-export run/wrfout --out ml/x --grid latlon:0.25          # a regular 0.25 degree grid
gpuwm ml-export run/wrfout --out ml/x --levels model --variables +vertical_velocity
gpuwm ml-export --list                                            # level sets, variables, naming schemes
```

Inputs can be history files, folders holding them (searched for
`wrfout_dNN_*`), gzip-compressed history files, or ZIPs whose members are
history files. A compressed or archived frame is unpacked just before it is
read and deleted just after, so the disk holds one frame at a time.

## Options

| option | values | default |
|---|---|---|
| `--levels` | `wb13`, `era5-37`, `model`, `model:1-20` or `model:1,2,4,8`, or a list of hPa such as `500,700,850` | `wb13` |
| `--variables` | `default`, `all`, `+NAME,...` (the defaults plus these), `NAME,...` (exactly these); a table id or a name under any scheme | the rows marked default |
| `--grid` | `native`, `latlon`, `latlon:DEG` | `native` |
| `--regrid` | `bilinear`, `area-mean` (with `latlon`) | `bilinear` |
| `--names` | `wb2` (WeatherBench 2 long names), `era5` (ERA5 short names) | `wb2` |
| `--layout` | `analysis` (`time` is the valid time), `forecast` (WeatherBench 2's `time` and `prediction_timedelta`) | `analysis` |
| `--domains` | `d01,d02,...` | every domain |
| `--every`, `--start`, `--end` | frames every N hours from the run's start; first and last valid time | every frame |
| `--config` | the run's configuration file; its SHA-256 is recorded | none |
| `--zip` | also write `<out>-ml.zip` | off |
| `--overwrite`, `--skip-unavailable`, `--threads` | replace an earlier export; write what the files can make and record the rest; worker threads | off, off, every core |
| `--append`, `--finalize` | add frames to the export in `--out` one call at a time, then close it | |

`--append` then `--finalize` writes exactly the bytes one call over the same
frames writes; a service converting frames as they arrive uses it.

## What the dataset holds

On pressure levels, dims `(time, level, y, x)`:

| WB2 name | ERA5 name | units | made from |
|---|---|---|---|
| `geopotential` | `z` | m\*\*2 s\*\*-2 | PH + PHB averaged onto mass levels |
| `temperature` | `t` | K | (T + 300)(p / 10^5)^(2/7) |
| `u_component_of_wind`, `v_component_of_wind` | `u`, `v` | m s\*\*-1 | U, V unstaggered and rotated to earth-relative by SINALPHA, COSALPHA |
| `specific_humidity` | `q` | kg kg\*\*-1 | vapour per kilogram of moist air |
| `vertical_velocity` (option) | `w` | Pa s\*\*-1 | omega, as ERA5's `w` is |
| `relative_humidity` (option) | `r` | % | the IFS mixed-phase definition, not capped at 100 |

At the surface, dims `(time, y, x)`: `2m_temperature`, `10m_u_component_of_wind`,
`10m_v_component_of_wind` (earth-relative), `mean_sea_level_pressure`,
`surface_pressure`, `total_precipitation` (over the interval since the
previous time, metres of water, empty at the first time),
`total_precipitation_6hr` (left out when no exported time lies 6 h before
another) and `total_column_water_vapour`. Static, dims `(y, x)`:
`geopotential_at_surface` and `land_sea_mask`. With `--levels model` the
model's own mass levels are kept with no interpolation and `pressure` is
added.

For a moving native grid, interval and six-hour precipitation compare
accumulators at the same geographic points. Newly exposed ground has NaN
where no earlier frame covers it. Lambert, polar, Mercator and regular
latitude-longitude grids support this alignment; a moved curved or
rotated-pole latitude-longitude grid is refused for precipitation because
subtracting unmatched ground would publish false rainfall.

Every variable carries `long_name`, `units`, `standard_name`, `short_name`
and `ecmwf_param_id`; level variables say how they were interpolated and
filled.

Every list here is a table (`gpuwm/data/ml_export/variables.json`,
`levels.json`, `spacings.json`, `names.json`): a new level set, a new
naming scheme, or a new variable an existing operator can make (any raw
history field with a scale, any of the derived quantities above) is a new
row.

## Levels, the ground and the model top

Inside the column, values are linear in ln(pressure) between the two model
levels around the target: the same column walk the renderer's pressure-level
charts use.

Below the lowest model level, temperature and geopotential follow ECMWF
extrapolation (Trenberth, Berry and Buja 1993,
NCAR/TN-396, the formulas NCL's `vinth2p_ecmwf` and GeoCAT's
`interp_hybrid_to_pressure(extrapolate=True)` implement), and every other
field takes the lowest model level's value. This rule does not make the
model's filled values identical to ERA5: its terrain, pressure and lowest
model-level state can differ. A `below_ground` mask (uint8, 1 under the
surface) marks points below the surface, so

```python
ds.where(ds.below_ground == 0)
```

keeps only model column. On the run's own grid, a level is under the ground
where its pressure exceeds the surface pressure.

A level above the model lid has no model data and is left out of the
dataset rather than filled, because a whole level of fill quietly poisons
the per-level normalisation every ML loader computes. The levels left out
are printed and listed in `levels_dropped_above_model_top`. With a 50 hPa
lid, `era5-37` keeps 50 hPa and loses 1 to 30 hPa. Between the top model
level and the lid, geopotential is interpolated up to the lid's own value
and other fields hold the top level's.

## Grids

`--grid native` keeps the run's grid. A projected domain has `(y, x)` with
2-D `latitude` and `longitude`, and, when the grid is uniform in the CF
projection's metres, 1-D `y` and `x` and a `crs` variable cartopy and MetPy
read. A stationary global tape keeps its regular grid as 1-D `latitude`
(ascending) and `longitude` (0 to 360). A moving nest's latitude, longitude,
surface geopotential and land mask vary with time. Moving projected grids
also carry absolute `projection_x_coordinate` and `projection_y_coordinate`
along time; moving regular grids use `y` and `x` dimensions with geographic
coordinates at each time. Stationary terrain and land mask stay two-dimensional.

`--grid latlon[:DEG]` regrids onto a regular latitude-longitude lattice.
Without DEG the spacing is the row of `spacings.json` nearest the run's
grid spacing. The lattice sits on integer multiples of DEG, so exports of
different runs share points, and a spacing dividing 0.25 degree lands on
ERA5's own points. The box is the largest lattice-aligned rectangle inside
the domain (and outside its lateral boundary rows, when the files state
them), so no point is filled. `bilinear` samples the model grid in its own
index space through WRF's projection; `area-mean` averages a block of those
samples over each cell, weighted by cos(latitude), for targets much coarser
than the model. Winds are interpolated as earth-relative components.

Each domain of a nested run is its own dataset (`d01.zarr`, `d02.zarr`).

## Opening it

```python
import xarray as xr
ds = xr.open_zarr("ml/my-run/d02.zarr")
ds["temperature"].sel(level=850)
```

The export folder is itself a Zarr group over its domains, and so is the
ZIP's root, so the archive opens in place:

```python
import xarray as xr, zarr
ds = xr.open_zarr(zarr.storage.ZipStore("ml/my-run-ml.zip", mode="r"), group="my-run/d02.zarr")
```

Zarr format 2 with consolidated metadata, which xarray reads under
zarr-python 2.18 and 3. Chunks hold one time and every level of one
variable, so one training sample is one read per variable; arrays are
float32, compressed losslessly with Blosc (Zstandard, byte shuffle). The
ZIP stores its entries uncompressed so it can be read in place.

`README.txt` beside the datasets says the same, and `ml-export-receipt.json`
lists every input file by name and SHA-256, each frame's time and size, the
table rows used and every variable left out with its reason.

## Provenance

The dataset's attributes carry the engine and its version, the run's
configuration digest (SHA-256 of `--config`, or of a fixed list of the
history files' configuration attributes), the initial-condition source and
cycle, the domain, the grid spacing, the level set and levels left out, the
interpolation, below-ground and model-top rules, the horizontal grid and a
digest of the input files. No machine paths, host names or GPU identities.

## Refusals

| refused | because otherwise |
|---|---|
| a variable whose history fields are absent (for example QVAPOR from a trimmed history) | the dataset would silently lack a variable that was asked for; `--skip-unavailable` writes the rest and records the omission |
| frames of one domain from two grids | two grids would be stacked into one array |
| two frames at one valid time, or a time before one already written | two frames would be written into one time slot |
| a projection rebuilt from the attributes that misplaces the file's own coordinates by more than 0.01 cell | every regridded value would land in the wrong place |
| `latlon` on a rotated-pole grid or a moving nest, or finer than a global tape | the regrid would sample the wrong ground, or add bytes and no information |
| an accumulation that falls by more than 0.01 mm between frames | negative rain would be published |
| an export folder holding files this exporter did not write, or an earlier export without `--overwrite` | someone's files, or an earlier dataset, would be overwritten |
| a frame larger than the host's available memory | the host would kill the export part-way |

Exit status: 0 done, 2 refused (one sentence naming the breakage), 3 the
exporter binary is not built or staged (the message names what supplies
it), 1 failed.

## How it was checked

- **The same file, another implementation.** GeoCAT's
  `interp_hybrid_to_pressure` (log-pressure, ECMWF extrapolation) on 3,000
  columns from two frames of a regional run and 1,000 from a hex run's
  frame, every WB13 level: temperature within 1.6e-5 K, geopotential
  within 0.015 m\*\*2 s\*\*-2, winds within 3.2e-6 m s\*\*-1 and specific
  humidity within 5e-10, above and below ground.
- **Against ERA5.** The first frame of an ERA5-started 12 km run, exported
  on ERA5's own 0.25 degree points and compared with ERA5's pressure levels
  where both are above ground (3,950 points). Mean differences at every
  level from 100 to 1000 hPa: temperature within 0.1 K, each wind
  component within 0.5 m s\*\*-1, specific humidity within 2.4e-5; 2 m
  temperature and 10 m winds within 0.001 K and 0.001 m s\*\*-1.
  Geopotential is 1 to 7.5 m above ERA5's from 850 hPa up; at 500, 300 and
  200 hPa, 87 to 95 percent of that is already in the run's own history
  file (its geopotential against the height its own temperatures give),
  not added by the export. Written as height instead of geopotential, with the levels
  flipped, or with grid-relative winds, the same comparison moves by
  50,000 m\*\*2 s\*\*-2, 28 K and a near-threefold wind error.
- **Area mean.** On that run's 0.25 degree area-mean grid, the box means of
  2 m temperature, total column water vapour and surface pressure equal the
  model grid's (each cell weighted by its true area) within 6e-5 of their
  value, and the light, patchy hourly precipitation within 1.6 percent; on a
  global tape at 1 degree, every one of them within 5e-6.
- **Opening.** Every export above, unzipped and in place from its ZIP,
  opens with `xarray.open_zarr` under zarr-python 2.18 and 3.4 with times
  decoded and one chunk per variable per sample.

## Speed and size

The default dataset on WB13, measured on shared 24-core hosts with eight
worker threads, every frame of each set:

| source | grid | per frame | stored per frame |
|---|---|---|---|
| regional, 12 km | 150 x 120 x 49 | 0.22 s | 3.2 MB |
| regional, 250 m | 580 x 364 x 84 | 5.4 s | 32 MB |
| hex, 0.97 km window | 248 x 248 x 55 | 0.91 s | 9.7 MB |
| global, 0.5 degree tape | 720 x 360 x 40 | 4.5 s | 50 MB |

Time scales with columns times model levels, and the 250 m frame peaked
at 2.7 GB of memory. Stored size scales with the number of columns: about
150 to 200 bytes per column per frame (smooth fields compress further),
and `era5-37` two to two and a half times WB13.
