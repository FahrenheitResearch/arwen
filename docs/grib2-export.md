# GRIB2 export

WOOF writes its histories as WMO GRIB2 (FM 92, edition 2) through four
doors, all of which run the same native exporter, `rw_grib2export`:

| Door | What it exports |
| --- | --- |
| `gpuwm export-grib2 INPUT... --out DIR` | history files, run folders, `.gz` histories or a history ZIP |
| `gpuwm go CONFIG.toml --grib2` | each history frame as the forecast writes it |
| `run_options.grib2 = true` (run-plan document) | the same, from a plan |
| `gpuwm render HISTORY... --grib2-out DIR` | the frames a render draws |

The exporter is new in 2.8.7 and is Apache-2.0 throughout. Every product is
computed by the WOOF post-processor (`tools/rustwx/crates/rw-post`), written
from published science; the packing is the vendored wx-core GRIB2 writer.

## Output

Per domain and valid time:

- `wrfsfc_dNN_YYYYMMDDTHHMMSSZ.grib2`: surface, column and single-level fields;
- `wrfprs_dNN_YYYYMMDDTHHMMSSZ.grib2`: height, temperature, dewpoint,
  relative humidity and the two wind components on each pressure level;
- `manifest.json`: the catalog, and for every frame the files, each field's
  GRIB2 code, level, units and method with its published source, the
  device each group ran on, the timings and every omitted field with the
  reason it was omitted.

`--zip` adds `<DIR>-grib2.zip` beside the folder (STORED, ZIP64 when
needed). A live export (`--grib2`) writes `<run>/grib2/` and finishes with
`<run>/grib2-grib2.zip`.

## Where it computes

`--post-device auto` (the default) computes on the first visible GPU when
the frame fits in its free memory and on the CPU otherwise; `gpu` requires
a card and `cpu` never opens one. The GPU kernels are prebuilt for every
supported architecture, so no CUDA toolkit is needed. The CPU and GPU
paths return the same bits, so the same frame gives byte-identical GRIB2
files on either. During `go --grib2` the exporter sees the forecast's own
card and uses it when there is room.

Measured on a 600 x 353 x 50 CONUS 9 km WOOF frame (279 messages): 6.6 s
with one RTX 5090 (post-processing about 1 s, the rest reading the history
and packing) and 12.1 s on 8 CPU threads.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--fields standard\|surface\|pressure\|all` | `standard` | `surface` writes only `wrfsfc`, `pressure` only `wrfprs`; `standard` and `all` write both |
| `--levels 1000,850,...` | 100 to 1000 by 25 | pressure levels in hPa |
| `--packing complex\|simple` | `complex` | data representation template 5.3 (spatial differencing, order 2) or 5.0 |
| `--bits N` | 20 | bits per packed value (1 to 31) |
| `--winds grid\|earth` | `grid` | wind components relative to the model grid (flag 0x08 set) or to east/north, rotated with SINALPHA/COSALPHA |
| `--post-device auto\|gpu\|cpu` | `auto` | see above |
| `--gust auto\|similarity\|tke` | `auto` | surface gust method: `auto` is the mixed-layer TKE gust where the history carries TKE (TKE_PBL or QKE), else the similarity gust; the frame manifest's `gust` says which ran (RULINGS 7, scored against ASOS) |
| `--definitions woof\|renderer\|arwen\|upp` | `woof` | who computes the fields: `woof` is the WOOF post-processor; `renderer` (or `arwen`, the same) takes the fields the product maps draw from the renderer's own diagnostics (see below); `upp` is a one-release alias of `woof` and prints the rename |
| `--extrema-interval-seconds N` | off | also write maximum and minimum updraft helicity and maximum 10 m wind over the N seconds ending at each frame (3600 for hourly); see below |
| `--upp-control srw\|rapr` | `srw` | level label of composite reflectivity: `srw` writes the entire-atmosphere layer 200, `rapr` writes 10 as RAP and HRRR files do; nothing else changes |
| `--domains`, `--times`, `--start`, `--end` | all | which frames to write |
| `--append`, `--finalize` | | incremental export: append writes new frames only (frames already exported are skipped), finalize writes the final manifest and the ZIP |
| `--list [--json]` | | print the catalog |
| `--threads N` | all cores | CPU worker threads |

## Grid

Grids keep WOOF's own geometry: Lambert conformal (template 3.30), polar
stereographic (3.20), Mercator (3.10) and regular latitude-longitude (3.0),
encoded on the 6,370,000 m sphere WRF projects on (shape of the Earth 1),
rows written south first (scanning mode 0x40). First and last points are
the history's own XLAT and XLONG. A decoder therefore places every point
where WOOF put it; an independent decoder (ecCodes) reproduces XLAT/XLONG
within 1.2e-5 degrees on the frame above. Rotated latitude-longitude
histories are refused (template 3.1 is not encoded).

## Renderer definitions

`--definitions renderer` (also spelled `arwen`) is for GRIB2 whose numbers
match the PNG maps. The rows the map renderer draws are computed by the
renderer's own diagnostics (`rw_wrfbatch`, the same `getvar` call and units
the maps use): terrain, surface pressure, 2 m temperature, dewpoint and
relative humidity, sea-level pressure, precipitable water, surface-based,
mixed-layer and most-unstable CAPE and CIN, LCL height, 0-1 and 0-3 km
helicity, 0-1 and 0-6 km bulk shear, STP, EHI, composite reflectivity, low,
middle and high cloud and PBL height (the model's own `PBLH`). CIN is
written negative, as in every other row. Every other row, the pressure
levels and all vector components stay WOOF's, so the file holds the same
messages under the same GRIB2 codes; `manifest.json` names, per frame,
which rows the renderer computed and which it could not.

## Extrema windows

A history carries running extremes under their WRF names (`UP_HELI_MAX`,
`UP_HELI_MIN`, the 0-2, 0-3 and 1-6 km variants, `WSPD10MAX`): each frame
holds the extreme since the previous history write. With
`--extrema-interval-seconds N` the exporter writes, for each frame, the
maximum (or minimum) over the N seconds ending at it, as product template
4.8 with the statistical process and the window in the message (hourly
maximum and minimum updraft helicity 0.7.199 and 0.7.200, maximum 10 m wind
speed 0.2.1). Frames written every 15 minutes make an hourly window from
four frames; a window equal to the history interval passes one frame
through.

The history does not record its own output interval, so each frame's
window is read off the frame sequence: all the frames the request finds
(every file of a run folder, before `--times`), plus the frames earlier
`--append` calls recorded. A window that the frames cannot build exactly
is omitted with the reason, never relabelled: frames spaced wider than the
window, a window that reaches back before the run start, or a frame of the
window that is not available. Pass the whole run folder: a hand-picked
subset of a finer history looks like a coarser one. The planes of recent
frames are kept in `<out>/.woof-grib2-extrema/` so a live export can build
its windows one frame at a time.

## Identification

Section 1 names centre 7 with GRIB master table 2 and local table 1, because
the catalog's local parameters and level types (for example MAPS sea-level
pressure 0.3.198 and the entire-atmosphere level 200) are entries of that
centre's local tables. WOOF's own shear components (category 192) are
written under centre 255, so no decoder labels them with another centre's
meaning. The reference time is the history's `SIMULATION_START_DATE`; the
forecast time is in hours when whole, else minutes, else seconds. Total
precipitation is an accumulation from the start (product template 4.8).

The forecast generating process identifier (product template octet 14) is
254 on every message, so a WOOF file is told apart from any NCEP model
(HRRR is 83, RAP 105, RRFS 134). NCEP's Office Note 388 Table A reserves
231 to 254 and assigns none of them; the manifest records the value and
its meaning.

Missing values are a GRIB2 bitmap: a carrier the history does not have
omits its products (never zero), and a field missing in every cell is
listed as omitted rather than written.

## Catalog `woof-post/v1`

Surface file. Group letters follow the post-processor's modules: A column
state and surface fields, B pressure levels and sea-level pressure, C
parcels and severe indices, D reflectivity, cloud and visibility, E
boundary layer and isotherm levels. Section and D numbers refer to the
post-processor specification; the method strings are the ones the
manifest records.

| id | group | GRIB2 | level | units | method |
| --- | --- | --- | --- | --- | --- |
| `terrain` | A | 0.3.5 | 1/0 | gpm | HGT, WRF geometric metres reported as gpm (D2) |
| `surface_pressure` | A | 0.3.0 | 1/0 | Pa | PSFC (D3) |
| `t2` | A | 0.0.0 | 103/2 | K | surface scheme T2; else TH2 (p2/p0)^(Rd/cp), p2 hypsometric from PSFC (D4, AMS Glossary) |
| `td2` | A | 0.0.6 | 103/2 | K | inverse AERK Magnus dewpoint of Q2 at shelter pressure, capped at T2 (JAM 35 (1996) 601, D4, D5) |
| `rh2` | A | 0.1.1 | 103/2 | % | RH over water, AERK Magnus, clipped [0,100] (JAM 35 (1996) 601, WMO-No. 8, D5) |
| `potential_temperature2` | A | 0.0.2 | 103/2 | K | TH2 |
| `specific_humidity2` | A | 0.1.0 | 103/2 | kg kg-1 | Q2/(1+Q2), Q2 floored at 0 |
| `skin_temperature` | A | 0.0.0 | 1/0 | K | TSK |
| `snow_water` | A | 0.1.13 | 1/0 | kg m-2 | SNOW, floored at 0 |
| `snow_depth` | A | 0.1.11 | 1/0 | m | SNOWH, floored at 0 |
| `snow_cover` | A | 0.1.42 | 1/0 | % | SNOWC x 100; else Noah depletion curve of SNOW and the VEGPARM.TBL SNUP threshold (JGR 108 (2003) 8851, D26) |
| `sensible_heat_flux` | A | 0.0.11 | 1/0 | W m-2 | HFX, upward positive |
| `latent_heat_flux` | A | 0.0.10 | 1/0 | W m-2 | LH; RUC land model: QFX x 2.501e6 (D28), upward positive |
| `ground_heat_flux` | A | 2.0.10 | 1/0 | W m-2 | GRDFLX |
| `friction_velocity` | A | 0.2.30 | 1/0 | m s-1 | UST |
| `downward_longwave` | A | 0.5.3 | 1/0 | W m-2 | GLW |
| `downward_shortwave` | A | 0.4.7 | 1/0 | W m-2 | SWDOWN, 0 where the Solar Energy 40 (1988) 227 solar zenith cosine <= 0 (D27) |
| `vegetation_fraction` | A | 2.0.4 | 1/0 | % | VEGFRA in percent |
| `vegetation_type` | A | 2.0.198 | 1/0 | index | IVGTYP |
| `sea_ice_fraction` | A | 10.2.0 | 1/0 | proportion | SEAICE, not thresholded |
| `roughness_length` | A | 2.0.1 | 1/0 | m | ZNT |
| `vegetation_min` | A | 2.0.231 | 1/0 | % | SHDMIN in percent |
| `vegetation_max` | A | 2.0.232 | 1/0 | % | SHDMAX in percent |
| `leaf_area_index` | A | 0.7.198 | 1/0 | 1 | LAI |
| `u10` | history | 0.2.2 | 103/10 | m s-1 | U10, the surface layer scheme's 10 m wind (history copy) |
| `v10` | history | 0.2.3 | 103/10 | m s-1 | V10, the surface layer scheme's 10 m wind (history copy) |
| `total_precipitation` | history | 0.1.8 | 1/0 | kg m-2 | RAINNC + RAINC plus the bucket counters times BUCKET_MM, accumulated from the simulation start (product template 4.8) |
| `mslp` | B | 0.3.1 | 101/0 | Pa | SPEC 5.2 D6: NMC reduction of Wea. Forecasting 13 (1998) 833 and Mon. Wea. Rev. 123 (1995) 59 from ground pressure and height |
| `maps_mslp` | B | 0.3.198 | 101/0 | Pa | SPEC 5.2: MAPS reduction of Mon. Wea. Rev. 118 (1990) 2099 from the 700 hPa temperature, 1-2-1 smoothing of Rev. Geophys. 8 (1970) 359 (D9) |
| `pwat` | B | 0.1.3 | 200/0 | kg m-2 | SPEC 5.3: (1/g) sum of q dp over model layers (AMS Glossary, precipitable water) |
| `sbcape` | C | 0.7.6 | 1/0 | J kg-1 | SPEC 6.1-6.2: surface parcel at shelter pressure (D10), Bolton (1980) eq. 43 pseudoadiabat by Newton iteration (MWR 136 (2008) 2764), virtual temperature buoyancy (WAF 9 (1994) 625) |
| `sbcin` | C | 0.7.7 | 1/0 | J kg-1 | SPEC 6.1-6.2: surface parcel CIN, negative, 0 with no LFC (D12) |
| `mlcape` | C | 0.7.6 | 108/9000-108/0 | J kg-1 | SPEC 6.2: 90 hPa mixed-layer parcel (WAF 17 (2002) 885, D11) |
| `mlcin` | C | 0.7.7 | 108/9000-108/0 | J kg-1 | SPEC 6.2: 90 hPa mixed-layer parcel CIN (D11, D12) |
| `mucape` | C | 0.7.6 | 108/30000-108/0 | J kg-1 | SPEC 6.2: most-unstable parcel, maximum theta-e in the lowest 300 hPa (SPC) |
| `mucin` | C | 0.7.7 | 108/30000-108/0 | J kg-1 | SPEC 6.2: most-unstable parcel CIN (D12) |
| `cape_best180` | C | 0.7.6 | 108/18000-108/0 | J kg-1 | SPEC 6.2 D13: six 30 hPa layer parcels in the lowest 180 hPa, maximum theta-e lifted |
| `cin_best180` | C | 0.7.7 | 108/18000-108/0 | J kg-1 | SPEC 6.2 D13: best-180 parcel CIN |
| `lcl_height` | C | 0.3.5 | 5/0 | gpm | SPEC 6.2: lowest 30 hPa layer parcel, Bolton (1980) eq. 21 LCL, height above ground |
| `storm_motion_u` | C | 0.2.27 | 103/0-103/6000 | m s-1 | SPEC 6.3, RULINGS 2: Bunkers internal dynamics (WAF 15 (2000) 61), 0-6 km height-weighted mean wind, 7.5 m s-1 deviation |
| `storm_motion_v` | C | 0.2.28 | 103/0-103/6000 | m s-1 | SPEC 6.3, RULINGS 2: Bunkers internal dynamics right mover |
| `srh_0_1km` | C | 0.7.8 | 103/1000-103/0 | m2 s-2 | SPEC 6.3: storm-relative helicity (16th Conf. Severe Local Storms (1990) 588) from 10 m to 1 km, right mover |
| `srh_0_3km` | C | 0.7.8 | 103/3000-103/0 | m2 s-2 | SPEC 6.3: storm-relative helicity from 10 m to 3 km, right mover |
| `shear_u_0_1km` | C | 0.192.3 | 103/1000-103/0 | m s-1 | SPEC 6.3 D14: 10 m to 1 km vector wind difference |
| `shear_v_0_1km` | C | 0.192.4 | 103/1000-103/0 | m s-1 | SPEC 6.3 D14: 10 m to 1 km vector wind difference |
| `shear_u_0_6km` | C | 0.192.3 | 103/6000-103/0 | m s-1 | SPEC 6.3, RULINGS 3: 10 m to 6 km AGL vector wind difference (SPC) |
| `shear_v_0_6km` | C | 0.192.4 | 103/6000-103/0 | m s-1 | SPEC 6.3, RULINGS 3: 10 m to 6 km AGL vector wind difference (SPC) |
| `bulk_shear_0_1km` | C | 0.192.1 | 103/1000-103/0 | m s-1 | SPEC 6.3: magnitude of the 0-1 km shear vector |
| `bulk_shear_0_6km` | C | 0.192.2 | 103/6000-103/0 | m s-1 | SPEC 6.3: magnitude of the 0-6 km shear vector |
| `stp` | C | 0.7.211 | 14/0 | 1 | SPEC 6.3, RULINGS 4: fixed-layer STP (WAF 18 (2003) 1243, WAF 27 (2012) 1136, SPC), floored at 0 |
| `ehi_0_1km` | C | 0.7.9 | 103/1000-103/0 | 1 | SPEC 6.3 D16: sbCAPE x SRH 0-1 km / 160000 (SHARP v1.50, NWS 1991) |
| `composite_reflectivity` | D | 0.16.196 | 200/0 (10/0 with `--upp-control rapr`) | dBZ | SPEC 7.1: column maximum of the reflectivity volume (AMS Glossary) |
| `reflectivity_1km` | D | 0.16.195 | 103/1000 | dBZ | SPEC 7.1, RULINGS 5: linear Z interpolated to 1000 m AGL, then dBZ |
| `low_cloud` | D | 0.6.3 | 214/0 | % | SPEC 7.2, RULINGS 1: maximum cloud fraction in the ISCCP low layer (BAMS 80 (1999) 2261) |
| `mid_cloud` | D | 0.6.4 | 224/0 | % | SPEC 7.2, RULINGS 1: maximum cloud fraction in the ISCCP middle layer |
| `high_cloud` | D | 0.6.5 | 234/0 | % | SPEC 7.2, RULINGS 1: maximum cloud fraction in the ISCCP high layer |
| `cloud_base` | D | 0.3.5 | 2/0 | gpm | SPEC 7.3 D20: lowest level with cloud fraction > 0.01 and condensate > 1e-6 kg kg-1 (ECMWF 228023) |
| `cloud_top` | D | 0.3.5 | 3/0 | gpm | SPEC 7.3 D20: highest level with the same thresholds |
| `cloud_ceiling` | D | 0.3.5 | 215/0 | gpm | SPEC 7.3 D19: lowest height where cloud fraction reaches 0.5 (FMH-1, ECMWF ceiling) |
| `visibility` | D | 0.19.0 | 1/0 | m | SPEC 7.4, RULINGS 6 and 10: hydrometeor extinction (JAM 38 (1999) 385), FRAM-L RH haze clear-air baseline (BAMS 90 (2009) 341), 550 nm aerosol extinction where the history carries it (the frame's group_d_sources.aerosol says which), 5 percent contrast meteorological optical range (WMO-No. 8) |
| `pbl_height` | E | 0.3.18 | 1/0 | m | SPEC 8: bulk Richardson number 0.25 (BLM 81 (1996) 245) with b u*^2, b = 100 (D23) |
| `gust` | E | 0.2.22 | 1/0 | m s-1 | SPEC 8 D22, RULINGS 7 (ASOS-scored): mixed-layer TKE gust (MWR 129 (2001) 5) where the history carries TKE, else surface-layer similarity gust (ECMWF IFS Part IV, BLM 11 (1977) 355); the frame's gust record says which |
| `u80` | E | 0.2.2 | 103/80 | m s-1 | SPEC 8 D25: linear in height between bracketing mass levels, 10 m wind below the lowest level |
| `v80` | E | 0.2.3 | 103/80 | m s-1 | SPEC 8 D25: linear in height between bracketing mass levels |
| `freezing_height` | E | 0.3.5 | 4/0 | gpm | SPEC 8: lowest warm-to-cold 273.15 K crossing, linear in height (AMS Glossary, freezing level) |
| `freezing_pressure` | E | 0.3.0 | 4/0 | Pa | SPEC 8: pressure of that crossing, linear in ln p |
| `highest_freezing_height` | E | 0.3.5 | 204/0 | gpm | SPEC 8 D24, RULINGS 8: highest crossing below the WMO lapse-rate tropopause (500-50 hPa search) |
| `highest_freezing_pressure` | E | 0.3.0 | 204/0 | Pa | SPEC 8 D24, RULINGS 8: pressure of that crossing |
| `minus10_height` | E | 0.3.5 | 20/263 | gpm | SPEC 8 D24: highest 263.15 K crossing below the tropopause |
| `minus20_height` | E | 0.3.5 | 20/253 | gpm | SPEC 8 D24: highest 253.15 K crossing below the tropopause |

With `--extrema-interval-seconds N`, for each carrier the history has
(product template 4.8 over the N s window):

| id | carrier | GRIB2 | level | process |
| --- | --- | --- | --- | --- |
| `max_updraft_helicity_2_5km` | UP_HELI_MAX | 0.7.199 | 103/5000-103/2000 | maximum |
| `min_updraft_helicity_2_5km` | UP_HELI_MIN | 0.7.200 | 103/5000-103/2000 | minimum |
| `max_updraft_helicity_0_3km` | UP_HELI_MAX03 | 0.7.199 | 103/3000-103/0 | maximum |
| `min_updraft_helicity_0_3km` | UP_HELI_MIN03 | 0.7.200 | 103/3000-103/0 | minimum |
| `max_updraft_helicity_0_2km` | UP_HELI_MAX02 | 0.7.199 | 103/2000-103/0 | maximum |
| `min_updraft_helicity_0_2km` | UP_HELI_MIN02 | 0.7.200 | 103/2000-103/0 | minimum |
| `max_updraft_helicity_1_6km` | UP_HELI_MAX16 | 0.7.199 | 103/6000-103/1000 | maximum |
| `min_updraft_helicity_1_6km` | UP_HELI_MIN16 | 0.7.200 | 103/6000-103/1000 | minimum |
| `max_wind_speed_10m` | WSPD10MAX | 0.2.1 | 103/10 | maximum |

Pressure file, every requested level:

| id | GRIB2 | level | units | method |
| --- | --- | --- | --- | --- |
| `height` | 0.3.5 | 100/<level Pa> | gpm | SPEC 5.1: hypsometric from the bracketing interface with the layer-mean virtual temperature (NCAR/TN-396); below ground per [R12] and the membrane reduction (D8) |
| `temperature` | 0.0.0 | 100/<level Pa> | K | SPEC 5.1: linear in ln p between bracketing mass levels; below ground 6.5 K km-1 (ICAO, NCAR/TN-396) |
| `dewpoint` | 0.0.6 | 100/<level Pa> | K | SPEC 5.1: AERK Magnus dewpoint of the interpolated q (JAM 35 (1996) 601) |
| `rh` | 0.1.1 | 100/<level Pa> | % | SPEC 5.1: RH over water from the interpolated q and T (WMO-No. 8); lowest level held below ground (D7) |
| `u` | 0.2.2 | 100/<level Pa> | m s-1 | SPEC 5.1: linear in ln p; lowest level held below ground (D7) |
| `v` | 0.2.3 | 100/<level Pa> | m s-1 | SPEC 5.1: linear in ln p; lowest level held below ground (D7) |

Group D (reflectivity, cloud and visibility) runs from the same shared
column state. The cloud fraction is the history's `CLDFRA` when present,
otherwise a diagnosed fraction (Xu and Randall 1996); the manifest's
`group_d_sources` says which, and which aerosol carriers fed visibility.
Visibility is always written. Where the history carries an aerosol
extinction carrier (`AEXTC55`, `EXTCOF55` or `AOD3D_SMOKE`), its 550 nm
extinction is added to the hydrometeor and haze terms. Without one, the
visibility is the hydrometeor extinction plus a clear-air baseline: the
FRAM-L relative-humidity haze relation (Gultepe et al., BAMS 90 (2009)
341), whose dry-air value lies beyond the 20 km cap.
`group_d_sources.aerosol` says which one each frame used.

## Method choices still open

Each is one switch in `rw-post` and is recorded in the manifest: the gust
method (`--gust`, scored against ASOS gusts), the 0-6 km shear form (10 m
to 6 km endpoints or layer means), the STP floor at zero, reflectivity
interpolation in Z or dBZ, the visibility contrast threshold, the
clear-air visibility baseline used without an aerosol carrier (to be picked
by the ASOS visibility score) and the tropopause search window of the
highest-freezing and isotherm levels.
