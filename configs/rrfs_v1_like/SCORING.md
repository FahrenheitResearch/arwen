# RRFS-like boundary comparison

Arm R starts from and takes boundaries from the same posted RRFS cycle.
Arm G starts from that RRFS analysis and takes hourly GFS boundaries from
the cycle six hours earlier. Arm GS starts from GFS and is a separate arm.
The deterministic physics profile declares the missing SAS, smoke, dust and
ozone parts. The dycore, step and coupling remain differences.

For the 2026-10-03 12 UTC cycle and an 18-hour window:

```sh
gpuwm fetch --source rrfs --cycle 2026-10-03T12 --hours 18 --cadence 1 --out rrfs-files
gpuwm fetch --source gfs --cycle 2026-10-03T06 --forecast-start-hour 6 --hours 18 --cadence 1 --p-top-pa 1000 --area '19,-136,55,-58' --out gfs-files
```

Use `conus_tier_c_arm_r.toml` for arm R. For arm G, the mapped preparation
uses `gfs_hourly_pressure_fields.mapping.json`,
`../rw-wps-gfs-terrain.composition.json`, and a separate analysis declaration:

```json
{
  "schema": "gpuwm-initial-source-v1",
  "source": "rrfs",
  "input_files": [
    "rrfs-files/rrfs.t12z.prslev.3km.f000.conus.grib2",
    "rrfs-files/rrfs.t12z.2dfld.3km.f000.conus.grib2"
  ],
  "supplements": {
    "rrfs_prslev_2dfld_in_band_surface": [
      "rrfs-files/rrfs.t12z.2dfld.3km.f000.conus.grib2"
    ]
  }
}
```

Both files belong in the primary analysis inventory. The pressure file has
no surface or soil fields. The repeated surface file also supplies the
composition's explicitly bound terrain role.

The hourly mapping carries every level the 1000 Pa fetch declares and a
3600-second boundary interval. Keep that level declaration equal to the
fetch manifest if the requested source top changes. Do not replace a source
with a different start under the same arm label.

```sh
gpuwm prep --source mapped --source-format grib2 \
  --mapping configs/rrfs_v1_like/gfs_hourly_pressure_fields.mapping.json \
  --composition configs/rw-wps-gfs-terrain.composition.json \
  --input-list gfs-files/forcing-files.txt \
  --supplement gfs_valid_time_terrain=GFS_F006 \
  --supplement gfs_valid_time_terrain=GFS_F007 \
  --provenance gfs_valid_time_terrain_provenance=terrain-provenance.json \
  --author-input-manifest gfs-inputs.json --initial-inputs initial.json \
  --wps-namelist configs/rrfs_v1_like/conus_tier_c_arm_gs.namelist.wps \
  --experiment-config configs/rrfs_v1_like/conus_tier_c_arm_gs.toml \
  --geog-root GEOG --output-root prepared-g \
  --preprocess-backend cpu --preprocess-workers 4 --stock-wrf-export off
gpuwm go configs/rrfs_v1_like/conus_tier_c_arm_gs.toml \
  --prepared-root prepared-g \
  --wps-namelist configs/rrfs_v1_like/conus_tier_c_arm_gs.namelist.wps
```

`forcing-files.txt` lists the fetched f006 through f024 files, one path per
line. Repeat the terrain supplement for every one of those files, rather
than just the two lines shown in the grammar above. Terrain provenance
names that same cycle, file inventory and valid-time range. Preparation
binds the separate initial source in its retained evidence. The prepared
bundle governs the forecast's start; a bare GS profile still starts from GFS.

For the comparison, build `tools/posted_forecast_reader` and pass
`posted_surface_fields.json` to `tools/obs_battery_score.py` with the native
reader library. Fetch the same MRMS and ASOS window for all arms. Use one
registration, lead set, interior mask and threshold table. Export and
intersect each arm's station admission before the final scores, as described
in the reader's README. Record input hashes and the actual reader artifact.

The native posted reader covers instantaneous reflectivity and the ASOS
surface variables. Interval precipitation needs a separately declared
accumulation operator. A short reader proof is not a scored forecast day,
and the current profile trims one row from each product-grid edge.

## Receipts every run keeps

A run of these profiles is reproducible only with these three facts in its
receipts, beside the input hashes:

- The WPS namelist of the cut, as a file. The full-grid namelists are the
  committed `*.namelist.wps` beside each profile; a smaller cut writes its
  own and keeps it.
- The commit of the tree that ran. A tree exported with `git archive`
  reports its version as `0+unknown`, so record `git rev-parse HEAD` of the
  exporting checkout next to the run log.
- The geography overlay. GSL drag (`gwd_opt = 3`) reads `varsso_10m`,
  `orogwd_10m` and `orogwd3_10m`, and the lake model reads lake depth.
  The overlay is NCAR's WPS_GEOG downloads
  `https://www2.mmm.ucar.edu/wrf/src/wps_files/varsso_10m.tar.bz2`,
  `https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd_10m.tar.bz2` and
  `https://www2.mmm.ucar.edu/wrf/src/wps_files/orogwd3_10m.tar.bz2`,
  unpacked under the GEOG root, plus
  `gpuwm fetch-geog --datasets lake_depth --source ncar`. Record each
  archive's SHA-256.
