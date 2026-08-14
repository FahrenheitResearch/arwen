# Every option, every door

The complete command-line surface, read off the parsers themselves.  It exists because a flag that appears in no document is a feature nobody can reach: `--parent-namelist` gated the whole stock-WRF-parent route, `--tiles` gated the only streamed prepared route, and neither was written down anywhere.

`tests/test_docs_extras_agree_with_code.py` holds this page against the parsers in both directions, so it cannot fall behind the code and cannot name a flag that was removed.  Regenerate it with `python -m tools.build_cli_options_doc` after changing any option.

Options are listed with the help text the tool itself prints.  Positional arguments and `--help` are omitted; run any door with `--help` for its usage line.

## `gpuwm adapt`

| option | what it does |
|---|---|
| `--descriptor JSON` | completed rw-wps.descriptor.v1 document |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--grib2-dump EXE` | expert override paired with --grib2-inventory |
| `--grib2-inventory EXE` | expert override paired with --grib2-dump |
| `--input GRIB2` | actual GRIB2 input (repeat for every file in the series) |
| `--output-dir DIR` | directory for create-only adapter authorities and manifest |
| `--skeleton JSON` | create a review-required descriptor scaffold and stop |
| `--vtable VTABLE` | 11-column WPS Vtable selector authority. Required, and never defaulted: this command adapts arbitrary sources, and quietly reaching for a GFS Vtable would mis-map every other product. A worked GFS example installs with the package -- <gpuwm package>\authorities\Vtable.GFS.rw-wps |

## `gpuwm cases`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit the registry as JSON for a front end |

## `gpuwm certify`

| option | what it does |
|---|---|
| `--band BAND` | acceptance band for this configuration; it is addressed by the configuration's SHA-256, and certify refuses a band keyed to another one |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--metrics-csv CSV` | matched-comparison metrics CSV for the run |
| `--out-verdict VERDICT` | write the verdict document here (it is printed either way) |
| `--run-capsule CAPSULE` | certification-capsule.json written by the run |
| `--wrf-reference-manifest MANIFEST` | WRF reference manifest naming the executable, build recipe, namelists and reference wrfout bytes the comparison was made against |

## `gpuwm check`

| option | what it does |
|---|---|
| `--alloc` | construct every persistent allocation on the device, zero steps, report measured vs estimate (N0; GPU required) |
| `--budget-gib GIB` | CPU-mode measured budget (free VRAM minus reserve) for the estimate<=budget leg |
| `--column-chunk COLS` | RRTMGP chunk override (the first over-budget lever) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--forcing-interval-s S` | forcing cadence sizing the root's eager LBC tables (default ERA5 6-hourly) |
| `--json` | machine-readable report |
| `--rail-mib MIB` | whole-machine device residency ceiling: the budget is additionally capped at RAIL minus what every other process on the card already holds (read from NVML before this process touches CUDA). A property of the host, so there is no default |
| `--reserve-gib GIB` | override the calibrated reserve policy with a flat reserve |
| `--vram-gib GIB` | physical VRAM total of the card being sized for. A CEILING on the free figure, never a source of one: a declared --budget-gib plus the reserve can otherwise synthesise more free VRAM than the card physically has |

## `gpuwm doctor`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit the checks as JSON |
| `--source {era5,gfs,hrrr}` | report only this data route's own resolution (repeatable) alongside the shared estate: the exact decoder its preparation will launch, and the byte transport its fetch will use. Omitted, every route this build knows is reported (era5, gfs, hrrr) |

## `gpuwm domain`

| option | what it does |
|---|---|
| `--ack ID` | declare a governed experiment, written verbatim into the emitted [experiment].acknowledgements. Repeatable. This door used to write the nocturnal declaration for you, which silenced the load guard at check/run/go/run-plan and both prepared runners for the life of the file; it no longer does, and refuses instead. The id it accepts is asymmetric-radiation-nocturnal-window-v1: a longwave-OFF suite over a window that includes local night, which you are running deliberately as a daytime validation experiment |
| `--buffer-km KM[,KM...]` | with --polygon, nonnegative geometry buffer in kilometres; one value applies to every domain, or supply exactly one outer-to-inner value per level. With --ladder auto, a multi-value list selects the preset of that depth (default: zero) |
| `--card {12gb,16gb,24gb,32gb}` | GPU tier; sets the VRAM budget (default 24gb when --vram-gib is absent) |
| `--chain R1,R2,...` | custom nest refinement ratios, integers in [2, 8] (e.g. --root-dx 3 --chain 4 for 3 km -> 750 m); omit for a single domain at --root-dx. Sized by the same estimator fit loop as the presets |
| `--cycle YYYY-MM-DDTHH|latest` | the forcing CYCLE (UTC), which is the run's start time unless --forecast-start-hour moves it; 'latest' probes the public mirrors for the newest complete gfs/hrrr cycle covering the whole window and prints what it picked (needs network; era5 must name an explicit time) |
| `--data-dir DIR` | where fetched forcing lives/will live (default data/<name>) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--forcing GRIB` | era5: explicit forcing GRIB path(s) already on disk (default <data-dir>/era5-combined.grib) |
| `--forecast-start-hour K` | gfs/gdas/hrrr: initialize the run from the cycle's f{K} FORECAST lead instead of its analysis, so start_time = cycle + K h and the boundaries come from f{K+i}. This is how a window deep in a forecast (say f174..f240) is reached without integrating from f000. The initial condition is then itself a K-hour forecast, and every receipt says so |
| `--geog-root DIR` | staged WPS_GEOG tree (default ${GPUWM_CASE_DATA_ROOT}/WPS_GEOG) |
| `--history-interval SECONDS` | how often the ROOT domain writes a wrfout, in seconds (default 3600). Must be a whole number of seconds and a whole number of that domain's time steps -- the loader checks both against the exact rational dt and refuses the emitted file otherwise, before it is written |
| `--hours N` | forecast length (run_seconds = N*3600) |
| `--ladder {12,12-3,12-3-1,12-3-1-0.5,auto}` | preset nest dx chain in km (default: 12 -- one 12 km domain, the shape `gpuwm go` runs end to end, same as the interactive session). Nest trees are explicit opt-in: a deeper preset, `auto` (the deepest preset that fits the card), or --root-dx / --chain for anything else; their closing block names the tree runner they route to |
| `--name` | experiment name (default derived from the center) |
| `--nest-history-interval SECONDS` | the same, for every NESTED domain (default 900). Nests write more often than the root by default because resolving what the root cannot, over a shorter window, is the point of running one. Ignored for a single-domain ladder |
| `--out TOML` | emitted experiment TOML path |
| `--physics-profile {morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1,nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1,nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1,thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1,thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1,wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1,wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1,thompson-mp8-ysu-mm5-noah-validation-v1,wsm6-ysu-mm5-noah-no-radiation-v1,wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1,wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1,wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1}` | shipped physics suite to emit; taken verbatim from the registry the prepared-forecast runner validates against, so the emitted config passes its guard as written. Read the names: the *-no-radiation-* and *-validation-* profiles run reduced physics with longwave OFF and are NOT nocturnally valid -- selecting one for a window that includes local night is REFUSED unless you declare it yourself with --ack. NOT every profile runs on every route: --source gfs cannot prepare wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1 or wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1 or wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1 -- the wizard refuses those pairings and names the missing component rather than emitting a config the front door would reject. (gfs/era5 default: morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1, full RTE+RRTMGP + Kain-Fritsch, nocturnally valid; hrrr default: thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1, Thompson + full RRTMG lw+sw, no cumulus at 3 km, nocturnally valid) |
| `--point LAT,LON` | domain center in decimal degrees. \|lat\| 90 is refused, and so is any center whose FITTED domain reaches the pole -- a domain containing one is unsupported -- so the usable limit is set by the domain's size, not by the center, and lands well short of 90 (near \|lat\| 72 on the default card and ladder, further equatorward as either grows). The refusal names the fitted size when it fires; the projection is auto-selected from \|lat\| (<25 Mercator, 25-60 Lambert conformal, >60 polar stereographic) unless --projection is set. Negative (southern/western) values work in both forms: --point -33.87,151.21 and --point=-33.87,151.21 |
| `--polygon GEOJSON` | local GeoJSON Polygon, MultiPolygon, Feature, or FeatureCollection; the minimum antimeridian-aware bounds supply the center and every emitted level is fitted around the geometry |
| `--projection {auto,lambert,mercator,polar}` | map projection override (default: auto by center latitude; all three are oracle-gated against WRF v4.6.1 module_llxy) |
| `--root-dx KM` | custom root grid spacing in km [0.05, 200]; use with --chain instead of --ladder |
| `--source {gfs,hrrr,era5}` | forcing source for the [fetch] hints and (era5) the [case_data] declarations |
| `--vram-gib N` | total VRAM in GiB (alternative to --card) |
| `--vtable` | era5: Vtable override (default: the packaged Vtable.ERA5_CDO, copied beside the TOML) |

## `gpuwm downscale`

`--parent-namelist` (with `--parent-namelist-domain`) is the entire stock-WRF-parent route this command's own summary advertises: without it, only a gpuwm parent run can be downscaled.  `--tiles {on,auto}` and `--child-size` are the only way to stream a `--point`-derived child, because this command authors the child TOML itself and a `[tiles]` table you wrote by hand would be overwritten.

| option | what it does |
|---|---|
| `--accept-parent-cadence` | accept the archive's own cadence as the ceiling (prints the 15-min guidance when coarser); mutually exclusive with --max-boundary-interval-seconds |
| `--card {12gb,16gb,24gb,32gb}` | VRAM tier for --point sizing (default 24gb; the same tiers `gpuwm domain` accepts) |
| `--child-config` | legacy RunConfig TOML for the child (specified=true, nested=false) |
| `--child-size NX[,NY]` | explicit child extent for --point |
| `--child-surface-from` | child-grid wrfinput/history file with land identity + soil warm start (required for surface-physics children) |
| `--dry-run` | validate contracts, derive/print the plan, write the derived TOML, run nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--health-interval-seconds` | _(the parser declares no help text for this option)_ |
| `--hours` | --point run window in hours (default: the full parent archive window) |
| `--i-parent-start` | _(the parser declares no help text for this option)_ |
| `--j-parent-start` | _(the parser declares no help text for this option)_ |
| `--max-boundary-interval-seconds` | explicit ceiling on acceptable parent cadence (the scientific cadence contract); mutually exclusive with --accept-parent-cadence |
| `--out` | create-only output directory for the child run (report.json, wrfout frames, restart) |
| `--output-interval-seconds` | --point child history cadence (default: the parent cadence) |
| `--parent-domain` | parent domain id when the directory carries several (e.g. 3 for the innermost archived parent) |
| `--parent-namelist` | stock-WRF namelist.input of the parent run |
| `--parent-namelist-domain` | domain column of --parent-namelist (default 1) |
| `--parent-restart` | gpuwm restart of the parent run (authoritative physics evidence) |
| `--point LAT,LON` | derive the child around this point instead of --child-config (gpuwm parents only) |
| `--preprocess-backend {cuda,cpu}` | _(the parser declares no help text for this option)_ |
| `--ratio` | refinement ratio (child-config placement: required; --point default 3) |
| `--tiles {on,auto}` | write [tiles] mode into the config --point derives, so the child integrates out of a pinned host store instead of resident ('on' always, 'auto' when tilestream.autoplan says it does not fit) |
| `--vram-gib` | explicit VRAM capacity for --point sizing |

## `gpuwm dual-run`

| option | what it does |
|---|---|
| `--capsule-a CAPSULE` | _(the parser declares no help text for this option)_ |
| `--capsule-b CAPSULE` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--out-report REPORT` | write the comparison document here |

## `gpuwm enprod`

| option | what it does |
|---|---|
| `--accept-status LIST` | comma-separated manifest member statuses to accept (default DONE,complete); any other status is a refusal naming the members |
| `--domain dNN` | which domain to plot when members hold more than one (default: the single domain present, else a refusal) |
| `--dpi N` | PNG resolution (default 150) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--field LIST` | comma-separated product fields: refl, uh, precip, t2, wspd10, or 'all' (default refl; 'refl,uh' is the severe-convective pair) |
| `--make-fixture DIR` | write a synthetic ensemble (members + manifest) to DIR and exit, for exercising the suite without a real ensemble |
| `--members N` | --make-fixture member count (default 5) |
| `--nan-policy {mask,refuse}` | what to do with a non-finite member value -- NaN or +/-Inf (default mask): 'mask' excludes it from every reduction at that point, shrinks the denominator with it, and stamps the resulting coverage on the panel; 'refuse' fails the whole product naming the members |
| `--neighborhood-km LIST` | comma-separated neighborhood radii in km for the probability product (default 0 = point probability). Each member is reduced to its maximum within the radius before the ensemble fraction is taken |
| `--out DIR` | output directory for the PNGs (default out/enprod) |
| `--pmm-tie-rule {flat-index,average}` | how the probability-matched mean resolves equal means (default flat-index): 'flat-index' is Ebert's algorithm exactly and paints a deterministic but meaningless row-major gradient across a plateau; 'average' gives every point in a tie the group's mean intensity and gives up the exact pooled distribution |
| `--products LIST` | comma-separated products: mean, spread, prob, paintball, pmm, or 'all' (default) |
| `--source-label TEXT` | model/provenance label stamped on every plot (default ArWen) |
| `--threshold LIST` | comma-separated exceedance thresholds in the field's own units; default is the field's own (refl 40 dBZ, uh 75 m2 s-2). Every threshold gets its own probability and paintball plot |
| `--timeidx N|all` | index into the valid times every member shares, or 'all' (default) |

## `gpuwm fetch`

| option | what it does |
|---|---|
| `--accept-inventory-change` | proceed when the live provider inventory yields a different record count than this ArWen was certified against. Without it such a mismatch is a refusal naming both counts; with it the live count becomes the bar and the fetch manifest records the acceptance |
| `--all-levels` | gfs/gdas only: take every isobaric level the product publishes instead of choosing a ladder. On the default NOMADS grib-filter transport this selects every level; with --mode full-file the whole object already carries every level and this declares them all for the decode. Either way level subsetting stays an opt-in bandwidth saver rather than a ceiling on the model top |
| `--area LAT0,LON0,LAT1,LON1` | bounding box corners in degrees (order free); allow several degrees of margin beyond the outer domain -- for gfs, 15 deg (the front door must prove every model lake's nearest source-water donor lies inside the crop; `gpuwm domain` suggests areas with this margin built in) |
| `--author-front-door-manifest` | author the front-door input manifest for the fetched series; requires --wps-namelist and --experiment-config (--bridge defaults to the built decoder this install resolves) |
| `--bridge EXE` | built gfs_grib2_bridge executable; omit it and the same resolver `gpuwm go` uses finds the one this install has (checkout build, libexec, then ~/.gpuwm/bridges -- see gpuwm doctor) |
| `--cache-dir DIR` | --engine rust only (hrrr, gfs/gdas --mode full-file): wx-core disk cache root, keyed by URL and byte range, so a re-run or an overlapping window re-reads bytes instead of re-downloading them |
| `--cadence {1,3,6}` | forecast-hour cadence: gfs 1 or 3 (default 3); gdas 1, 3, or 6 (default 3, and it does not apply to --hours 0, which is the analysis alone); era5 template 1, 3, or 6 (default 6); hrrr is hourly |
| `--cycle YYYY-MM-DDTHH|latest` | model cycle (UTC); 'latest' resolves the newest complete cycle from the AWS Open Data listing (gfs/hrrr only) |
| `--engine {auto,rust,python}` | hrrr, and gfs/gdas --mode full-file: which downloader moves the bytes. 'rust' is the vendored rw_fetch backbone (16 MiB parallel range GETs, .idx coalescing, the cross-process NOMADS rate governor, a disk cache); 'python' is the stdlib transport and always works; 'auto' (default) uses the backbone when it is built |
| `--experiment-config TOML` | the experiment TOML the front door will consume |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--force-refetch` | move every existing file in --out aside (nothing is deleted) and re-download this request. The receipts go first -- fetch-manifest.json, SHA256SUMS, the series -- so an interrupted force can never leave a manifest behind claiming payloads it has already replaced; then payloads, .idx indexes, stale parts and anything else in the directory. Files already set aside by an earlier quarantine are left untouched, and subdirectories are yours. Required when re-fetching a different area/cycle into the same --out |
| `--forecast-start-hour K` | gfs/gdas/hrrr: the forecast lead the window BEGINS at (default f000, the analysis). --hours stays the window length, so --forecast-start-hour 174 --hours 66 fetches f174..f240 and nothing before it; an experiment whose start_time is cycle+K is then initialized from f{K} with its boundaries from f{K+i}. With --author-front-door-manifest on an already-fetched --out, this authors the manifest over that tail of the existing series instead of re-downloading it |
| `--hours N` | forecast window length: hours 0..N are fetched. gdas is certified for fetch and decode through f009 -- there is no gdas ingest route, so those files stop at the decoder -- and it is the one source that also accepts --hours 0, because its f000 is an analysis |
| `--manifest-out JSON` | manifest path (default <out>/gfs-input-manifest.json) |
| `--mode {auto,full-file,idx-subset}` | the byte transport. hrrr (--engine rust): 'full-file' is the default -- the whole object in parallel range GETs, which is the pipeline this product is built on; 'idx-subset' is the opt-in bandwidth saver: it selects records instead of taking the file, saves transfer volume, costs wall clock, and refuses rather than silently degrading when the index cannot carry the selection; 'auto' is the probe rule -- take the whole file when the .idx is absent, malformed, or provably shorter than the object -- which is what an install without the rust backbone falls back to. gfs/gdas: 'full-file' takes the whole pgrb2.0p25 objects from the S3 archive (either engine); omitted, the NOMADS grib-filter crop remains the default, and 'auto'/'idx-subset' refuse -- .idx record subsetting of the raw objects is not a certified GFS route |
| `--out DIR` | output directory (created; complete files are skipped on re-run) |
| `--p-top-pa PA` | gfs/gdas only: the model top (Pa) the fetched atmosphere must reach. The pressure ladder is extended upward along whatever the live inventory publishes until a level sits at or above it, so --p-top-pa 5000 fetches the 70 and 50 hPa levels the certified 100 hPa ladder stops short of. Omitted, the certified 21-level ladder is fetched exactly as before (a 10000 Pa source top). A top the product cannot serve refuses and names the deepest it can |
| `--point LAT,LON` | center point; requires --radius-km |
| `--radius-km KM` | half-width of the box around --point |
| `--source {gfs,gdas,hrrr,era5}` | public data source |
| `--static-input NPZ` | optional prebuilt static cache (with --static-receipt); omit when the front door builds statics from --geog-root |
| `--static-receipt JSON` | receipt for --static-input |
| `--transport {auto,nomads,s3}` | hrrr only: download host. Both serve byte-identical files and .idx indexes; 'nomads' (nomads.ncep.noaa.gov, roughly the newest 48 h) publishes each hour first, 's3' is the full AWS archive, and 'auto' (default) probes NOMADS for the requested window and falls back to S3 |
| `--validate GRIB` | era5 only: validate user-supplied GRIB1 file(s) against what gpuwm ingest expects instead of fetching |
| `--wait-for` | hrrr only: live-cycle mode -- download each forecast hour as it publishes (polling at most every 30 s), so preparation can start before the cycle finishes publishing; on timeout the manifest still records the complete fetched prefix and a re-run resumes |
| `--wait-timeout-minutes MIN` | hrrr --wait-for only: give up after this long (default 90 min), reporting exactly which hours were fetched |
| `--wps-namelist WPS` | the namelist.wps the front door will consume (e.g. the gpuwm domain output) |

## `gpuwm fetch-bridges`

| option | what it does |
|---|---|
| `--dest DIR` | stage into DIR instead of ~/.gpuwm/bridges (gpuwm finds the default on its own; anywhere else needs the per-artifact environment variables) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from DIR` | stage from a local directory instead of downloading (offline installs): either the bundle archive or the artifacts loose in it; verification is identical |
| `--keep-bundle` | keep the verified archive under <dest>/.fetch-bridges after staging (default: remove it) |
| `--list` | print the platform, bundle and per-artifact staged state, then exit without touching the network |

## `gpuwm fetch-geog`

| option | what it does |
|---|---|
| `--allow-upstream-drift` | accept an NCAR archive whose bytes no longer match the packaged pin (recorded as unpinned; refused outside a sanity size band); never applies to the mirror |
| `--bundle` | fetch NCAR's single geog_high_res_mandatory.tar.gz (2.6 GiB) instead of the per-dataset tarballs and extract the requested datasets from it (fallback; NCAR only) |
| `--datasets all|NAME,NAME` | which datasets to stage (default all nine the static builder opens) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--keep-archives` | keep the verified tarballs under <root>/.fetch-geog after extraction (default: remove each one after its datasets validate) |
| `--list` | print the dataset/size/source table and per-dataset staged state, then exit without touching the network |
| `--root DIR` | geog root to stage into (default: $GPUWM_CASE_DATA_ROOT/WPS_GEOG, exactly what gpuwm doctor checks and wizard configs reference) |
| `--source {hf,ncar}` | download host: 'hf' (default) is the ArWen mirror on Hugging Face (CDN bandwidth, pinned bytes); 'ncar' is the upstream NCAR server (single host, can be slow, no upstream checksums -- the same pins are enforced) |

## `gpuwm fetch-tables`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from DIR` | stage from a local directory instead of downloading (offline installs); verification is identical |

## `gpuwm go`

`--no-memory-gate` is the only escape from the pre-fetch memory gate.  The gate runs before the chain downloads anything, and on a box whose card it cannot see it declines to refuse rather than blocking a run that would have worked.

| option | what it does |
|---|---|
| `--data-dir DIR` | reuse an existing `gpuwm fetch` download instead of fetching into <outdir>/data |
| `--dry-run` | print the six commands, filled in, and exit without running any of them |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--geog-root DIR` | staged WPS_GEOG tree (default: the one `gpuwm fetch-geog` stages into) |
| `--no-memory-gate` | skip the before-the-fetch memory check that refuses a configuration whose binding phase cannot fit this card's free VRAM |
| `--outdir DIR` | root for the authority, prepared and run trees (default <config-stem>-go beside the config) |

## `gpuwm import-namelist`

| option | what it does |
|---|---|
| `--ack ID` | declared-experiment acknowledgement id to write into [experiment].acknowledgements of the resolved TOML (repeatable). WRF namelists cannot spell gpuwm governance declarations, so an import that needs one -- e.g. shortwave-on/longwave-off physics across a window that includes local night -- names the id it wants in its refusal |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--name NAME` | [experiment].name for the resolved TOML (default derived from start time and domain count) |
| `--output TOML` | write the resolved experiment TOML here (omit to print the report only) |
| `--rrtmg-variant {rte-rrtmgp,rrtmg_legacy}` | implementation for a WRF RRTMG 4/4 request: the established RTE+RRTMGP substitution (default, unchanged output) or the exact legacy-RRTMG port (fails closed at physics setup until its compute kernels land) |

## `gpuwm ingest`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output NPZ` | initialized-state NPZ output |

## `gpuwm multi-run`

`--preflight {estimate,alloc,off}` is the only override of the plan's own preflight mode.

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--preflight {estimate,alloc,off}` | override the plan's gpuwm check mode: estimate, alloc, or off |
| `--summary SUMMARY.json` | summary path relative to the plan (default PLAN.summary.json) |

## `gpuwm obs`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `gpuwm obs asos`

Takes no options of its own.

## `gpuwm obs goes`

Takes no options of its own.

## `gpuwm obs mrms`

Takes no options of its own.

## `gpuwm obs odim`

Takes no options of its own.

## `gpuwm obs opera`

Takes no options of its own.

## `gpuwm obs radar`

Takes no options of its own.

## `gpuwm obs radar doctor`

Takes no options of its own.

## `gpuwm obs radar grid`

| option | what it does |
|---|---|
| `--clear-air-from-censor` | build clear-air zeroes from the decoder's own gate codes as well as from finite below-floor gates. Needs a pack carrying censor planes (v2 or v3); a v1 pack is a hard error rather than a silent fallback. Range-folded and ambiguous gates stay excluded either way |
| `--dealias` | unfold radial velocity per sweep before gridding instead of masking every gate that might be folded. Requires scipy |
| `--grid-wrfout WRFOUT` | the wrfout whose georeference the observations are gridded onto; its SHA-256 joins the receipt |
| `--max-elevation-deg DEG` | likewise: the elevation ceiling, stated rather than defaulted |
| `--max-range-km KM` | THE range authority, required rather than defaulted: a build that quietly picked a different range than the one it is compared against produces a plausible, wrong answer |
| `--out NC` | observation file to write |
| `--overwrite` | replace an existing --out |
| `--pack PACK` | the sweep pack, from `pack` or from rw_nexrad |

## `gpuwm obs radar nyquist`

| option | what it does |
|---|---|
| `--file H5` | one ODIM file; geometry is read, no payload |

## `gpuwm obs radar pack`

| option | what it does |
|---|---|
| `--dir DIR` | directory of single-sweep ODIM files (SCAN) to assemble into one volume, as Germany publishes them. Mutually exclusive with --file |
| `--file H5` | one whole-volume ODIM file (PVOL). Mutually exclusive with --dir |
| `--max-elevation-deg DEG` | drop cuts above this elevation. The 90-degree birdbath a Dutch volume opens with is a calibration cut, not an observation of anything a model column exists for |
| `--max-range-km KM` | trim gates beyond this range |
| `--out PACK` | sweep pack to write |
| `--quantities Q,Q` | ODIM quantity names to carry (DBZH,VRADH). Omitting this carries every quantity in the volume, which is nine of them on a Dutch scan |
| `--stamp YYYYmmddTHHMMSSZ` | which volume in --dir to assemble, in the spelling `volumes` reports. Required only when the directory holds more than one: taking the newest silently would put a volume nobody asked for behind an ordinary-looking record |

## `gpuwm obs radar sites`

| option | what it does |
|---|---|
| `--bbox W,S,E,N` | select the sites inside this lon/lat box |
| `--no-velocity` | do not require a radial-velocity moment in the assimilability check, for a reflectivity-only assimilation |
| `--require-assimilable` | also run the assimilability check over the selection and report its refusal in full. Without this the verdict field is null rather than true: a check that did not run has no verdict |
| `--site ID` | select one site by its table id |

## `gpuwm obs radar volumes`

| option | what it does |
|---|---|
| `--dir DIR` | directory of ODIM .h5 files, not searched recursively |

## `gpuwm obs stage4`

Takes no options of its own.

## `gpuwm render`

`--pair-labels`, `--pair-subtitle` and `--pair-title` title and label the paired CPU-vs-GPU figure `--pair` composes.

| option | what it does |
|---|---|
| `--dpi N` | PNG resolution, matplotlib engine (default 150) |
| `--engine {auto,rust,matplotlib}` | render engine: the vendored Rusty Weather renderer (campaign plot quality; 151 implicit-render catalog candidates per file) or the matplotlib fallback; 'auto' (default) uses rust whenever its binary is built and probes as runnable |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--heavy` | rust engine: also compute the heavy ECAPE product family at import (SBECAPE/SBNCAPE/SBECIN, ECAPE SCP/EHI/...; adds substantial per-frame import time) |
| `--list-products` | list the engine's product catalog with per-file availability (why each product is or is not renderable from this wrfout) instead of rendering |
| `--out DIR` | output directory for the PNGs (default out/render) |
| `--pair ('A_DIR', 'B_DIR')` | compose two runs' rendered PNG directories into labeled side-by-side comparison sheets (no wrfout arguments) |
| `--pair-labels ('LEFT', 'RIGHT')` | panel labels (default: the two directory names) |
| `--pair-subtitle TEXT` | optional pair-sheet subtitle |
| `--pair-title TITLE` | pair-sheet title (default 'Paired comparison') |
| `--products LIST` | comma-separated products: refl, t2, wind10, precip, olr, or 'all' (default); with the rust engine, raw catalog slugs (sbcape, srh_0_1km, ...) also work and 'all' renders its full catalog |
| `--size WxH` | output pixels, rust engine (default 1200x900) |
| `--source-label TEXT` | model/provenance label stamped on every plot (default 'ArWen <the executing version>'); set it when rendering wrfout files this model did not produce, so the sheet does not claim them |
| `--timeidx N|all` | frame index within each file, or 'all' (default) |

## `gpuwm report`

| option | what it does |
|---|---|
| `--dry-run, --list` | print the manifest -- everything that would be included, redacted and reported missing -- and write nothing |
| `--exit-code N` | the exit status the failing command returned, recorded in the manifest (nothing on disk records it) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--log FILE` | an additional log file to include, for output that was redirected outside the run directory (repeatable) |
| `--output PATH` | where to write the zip: a file path, or a directory to name it in (default: the current directory, falling back to the system temporary directory and then your home directory if a write is refused) |

## `gpuwm resume`

| option | what it does |
|---|---|
| `--allow-shared-gpu` | UNSUPPORTED: permit another substantial CUDA compute context; device verification and the GPUWM UUID lock remain enforced |
| `--directory-input-hash {inventory,content}` | how declared directory inputs (the static geography tree) are bound to this run's identity: 'inventory' (default) uses relative path, size, and mtime; 'content' reads every file and uses its SHA-256. Use 'content' when two runs being compared for byte identity stage their geography separately, and when an mtime-preserving change to that tree must not go unnoticed (docs/public/DETERMINISM.md). Also settable as GPUWM_DIRECTORY_INPUT_HASH. |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from CKPT|latest` | explicit gpuwmrst_*.npz checkpoint, or 'latest' (default) to take the newest set in --outdir whose members validate |
| `--gpu-uuid GPU-UUID` | physical GPU UUID to lock (required on multi-GPU hosts) |
| `--health-debug` | enable debug phase health attribution hooks |
| `--no-supervise` | run the experiment in this process (escape hatch; disables fresh-process recovery and exclusive-GPU supervision) |
| `--outdir OUT` | the interrupted run's wrfout/checkpoint directory (default out/run) |
| `--prep-timeout SECONDS` | optional preparation heartbeat timeout; default is no timeout until integration begins |
| `--supervisor-max-restarts N` | fresh-process recovery attempts (default 3) |

## `gpuwm run`

`--allow-shared-gpu`, `--gpu-uuid`, `--prep-timeout` and `--supervisor-max-restarts` are the command-line spellings of settings STREAMING.md documents only as run-plan keys.

| option | what it does |
|---|---|
| `--allow-shared-gpu` | UNSUPPORTED: permit another substantial CUDA compute context; device verification and the GPUWM UUID lock remain enforced |
| `--directory-input-hash {inventory,content}` | how declared directory inputs (the static geography tree) are bound to this run's identity: 'inventory' (default) uses relative path, size, and mtime; 'content' reads every file and uses its SHA-256. Use 'content' when two runs being compared for byte identity stage their geography separately, and when an mtime-preserving change to that tree must not go unnoticed (docs/public/DETERMINISM.md). Also settable as GPUWM_DIRECTORY_INPUT_HASH. |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--gpu-uuid GPU-UUID` | physical GPU UUID to lock (required on multi-GPU hosts) |
| `--health-debug` | enable debug phase health attribution hooks |
| `--no-supervise` | run the experiment in this process (escape hatch; disables fresh-process recovery and exclusive-GPU supervision) |
| `--outdir OUT` | wrfout output directory |
| `--prep-timeout SECONDS` | optional preparation heartbeat timeout; default is no timeout until integration begins |
| `--restart RST` | resume from a gpuwmrst restart file written by an earlier run of the SAME config (only the forecast length / output and restart cadence may differ); restart writing itself is the restart_interval_s config key |
| `--supervisor-max-restarts N` | fresh-process recovery attempts (default 3) |

## `gpuwm run-plan`

| option | what it does |
|---|---|
| `--catalog` | print the renderer's product catalog as one JSON document -- what may be put in the render_products run option -- and run nothing; needs no plan |
| `--estimate` | print this plan's VRAM estimate and output-frame counts as one JSON document, and run nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--no-readiness` | with --probe, report the device inventory only: the NVML-only half, safe to poll on a card that is busy |
| `--probe` | print this machine's device inventory and runtime-estate readiness as one JSON document; needs no plan. The device inventory is NVML only and creates no CUDA context; the readiness half runs `gpuwm doctor`'s checks, which verify by execution and do create one |
| `--resolve` | print the fully resolved configuration plus every automatic resolution as one JSON document, and run nothing |

## `gpuwm setup`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from DIR` | stage the bridge and table artifacts from a local directory instead of downloading (offline installs); verification is identical. Does not apply to --with-geog, which has its own --source |
| `--with-geog` | also stage the WPS_GEOG static geography (~1.3 GB compressed, ~16 GB unpacked); the size is printed before anything downloads |

## `gpuwm static`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output NPZ` | static-field NPZ output |

## `gpuwm stream`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `gpuwm update`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `gpuwm verify`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--outdir OUT` | directory for the PNG and wrfout NetCDF output (omit to compute metrics only) |

## `gpuwm version`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--offline` | skip the PyPI lookup entirely (it is already skipped silently whenever the network does not answer) |
| `--pypi-timeout SECONDS` | seconds to wait for the index (default 2.0) |

## `gpuwm-mapped-inspect`

| option | what it does |
|---|---|
| `--grib1-bridge` | _(the parser declares no help text for this option)_ |
| `--grib2-dump` | _(the parser declares no help text for this option)_ |
| `--grib2-inventory` | _(the parser declares no help text for this option)_ |
| `--input` | _(the parser declares no help text for this option)_ |
| `--input-manifest` | _(the parser declares no help text for this option)_ |
| `--input-manifest-sha256` | _(the parser declares no help text for this option)_ |
| `--mapping` | _(the parser declares no help text for this option)_ |

## `gpuwm-prepared-forecast`

`--tiles JSON` is the only way to stream this route: its hash-bound experiment cannot carry a `[tiles]` table, so the table rides on the flag.  `--render-products` (with `--render-dir`) is render-on-first-committed-frame, off by absence.  `--materialize-authorities` and `--show-capabilities` each select a DIFFERENT program with its own options and must be the first argument on the line.

| option | what it does |
|---|---|
| `--ack` | registry-owned expert acknowledgement id; repeat as needed. The hash-bound experiment's acknowledgements array delivers the same consent |
| `--domain-bundle` | explicit hierarchy d01 bundle; if omitted it is derived from the hash-bound domain-artifacts manifest |
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--history-interval-seconds` | history cadence; must equal the hash-bound experiment's d01 history_interval_s, and defaults to it when omitted |
| `--io-mode {history}` | _(the parser declares no help text for this option)_ |
| `--materialize-authorities` | create one hash-receipted named-source experiment/WPS authority pair for an exact physics profile, then exit. Run it first on the line and with --help after it for that mode's own options |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--physics-profile` | optional assertion that the hash-bound experiment IS this shipped suite, refused on any switch drift; omitted, the experiment's own physics runs as written and its WRF-verification status is reported, never gating |
| `--prepared-content-sha256` | _(the parser declares no help text for this option)_ |
| `--prepared-root` | _(the parser declares no help text for this option)_ |
| `--proof-sha256` | _(the parser declares no help text for this option)_ |
| `--render-dir DIR` | where --render-products publishes; defaults to OUTDIR/png. Ignored without --render-products |
| `--render-products SPEC` | `gpuwm render --products`' own spec -- a comma-separated product list, or `all`, or `none` -- for the FIRST frame this run commits, rendered on a worker thread while the forecast is still integrating. Absent is off, and off is the default: there is deliberately no second switch, so "which products" has one answer that cannot disagree with itself. The first frame is the analysis at t = 0, durable before a single step is integrated |
| `--run-seconds` | forecast length; must equal the hash-bound experiment's run_seconds, and defaults to it when omitted |
| `--show-capabilities` | print this runner's capability JSON and exit; it must be the only argument |
| `--source {20crv3,era5,gfs,hrrr}` | _(the parser declares no help text for this option)_ |
| `--source-manifest-sha256` | _(the parser declares no help text for this option)_ |
| `--stream-init {auto,resident,store}` | which road a STREAMED forecast builds its domain on. `resident` restores the prepared cache into one full-domain DomainState, attaches physics to it and lets the streaming seam copy it into the pinned host store -- the road with the parity proof, and the one that caps the domain at the size of the CARD rather than of the machine (MEASURED at nz = 49: the prepared case costs about 15 780 B/column, so 1024x1024 is refused on a 16 GB card while the streamed forecast it would have fed needs about 6 GiB). `store` fills the same store one ROW SLAB at a time and never allocates a domain-shaped device array, so the ceiling is the machine's pinned RAM. `auto`, the default, prices the resident state from the cache's own state/* manifest times the measured physics headroom and takes the resident road wherever it fits inside 0.80 of the card's free memory. Meaningful only when the run streams: with [tiles] off the resident state IS the domain and this flag changes nothing |
| `--tiles JSON` | the [tiles] table this forecast integrates under, as a JSON object with the keys gpuwm.core.streaming.StreamingOptions takes (mode/tile_nx/tile_ny/nbuffers/halo/store/write_mode/pipeline/vram_budget_bytes/host_budget_bytes). For the caller whose hash-bound experiment cannot carry one: the native HRRR chain hands this runner the authority its preparer BUILT, which has no [tiles] table, so a user's block had nowhere to ride. Validated by the same StreamingOptions.from_mapping the config front door uses, and binds no identity -- omitted, the hash-bound experiment's own table (usually none) runs |
| `--wps-namelist` | _(the parser declares no help text for this option)_ |

## `gpuwm-prepared-forecast --materialize-authorities`

| option | what it does |
|---|---|
| `--base-experiment-config` | _(the parser declares no help text for this option)_ |
| `--base-wps-namelist` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output-directory` | _(the parser declares no help text for this option)_ |
| `--physics-profile` | shipped suite to materialize into the experiment; omitted, the base config's own physics is published unchanged and its WRF-verification status is reported |
| `--source {20crv3,era5,gfs,hrrr}` | _(the parser declares no help text for this option)_ |

## `gpuwm-prepared-tree-forecast`

`--sealed-forcing-extension` selects the append-only forcing-prefix checkpoint contract.

| option | what it does |
|---|---|
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--experiment-config-sha256` | _(the parser declares no help text for this option)_ |
| `--health-debug` | _(the parser declares no help text for this option)_ |
| `--io-mode {history,none}` | _(the parser declares no help text for this option)_ |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--preparation-receipt-sha256` | _(the parser declares no help text for this option)_ |
| `--prepared-root` | _(the parser declares no help text for this option)_ |
| `--restart` | resume from any member of a gpuwmrst checkpoint set written by an earlier run of this prepared tree; only the forecast length (run_seconds) and the output/restart cadence (history_interval_s, restart_interval_s) may differ from the run that wrote it -- the same contract `gpuwm run --restart` publishes. Anything else is refused by name |
| `--sealed-forcing-extension` | write/restore checkpoints using the explicit append-only forcing-prefix contract |
| `--show-capabilities` | print this runner's capability JSON and exit; it must be the only argument |

## `gpuwm-wrf-runtime-check`

| option | what it does |
|---|---|
| `--bridge-dir` | _(the parser declares no help text for this option)_ |
| `--contract` | _(the parser declares no help text for this option)_ |
| `--receipt` | _(the parser declares no help text for this option)_ |
| `--skip-gpu` | _(the parser declares no help text for this option)_ |

## `rw-wps`

`--validate-physics-plan`, `--canonical-physics-plan-output`, `--extend-root-preparation`, `--sealed-prepared-cache`, `--domain-source-orography`, `--validate-hrrr-domain` and `--no-stock-wrf-export` are gates on the preprocessing route; each is off unless named.

| option | what it does |
|---|---|
| `--ack` | registry-owned expert physics acknowledgement id; repeatable |
| `--author-input-manifest` | create an exact mapped or 20CRv3 input manifest; conflicts with an existing --source-manifest/--source-manifest-sha256 pair |
| `--author-mapping` | create-only path for a mapping compiled from --descriptor; the adjacent *.authoring.json receipt binds descriptor/Vtable bytes |
| `--author-only` | author the requested create-only mapped contract or 20CRv3 member manifest and exit; requires --author-input-manifest and does not need run geometry |
| `--bridge` | prebuilt gpuwm all-Rust source-specific GRIB bridge executable |
| `--canonical-physics-plan-output PATH` | create an exact canonical UTF-8 copy of the plan validated by --validate-physics-plan; refuses an existing output |
| `--child-workers` | bounded CPU worker budget for parallel d02..dNN initialization (1..32) |
| `--composition` | strict gpuwm-mapped-composition-v2 product join contract |
| `--cpu-preprocess-bridge` | _(the parser declares no help text for this option)_ |
| `--cycle` | GFS cycle in YYYY-MM-DD_HH:MM:SS form |
| `--descriptor` | explicit rw-wps.descriptor.v1 science contract; requires --author-mapping and, for GRIB, --vtable |
| `--domain-source-orography DNN=PATH` | ERA5 hierarchy source-orography binding; repeat once for every domain (d01..dNN). All bindings use --source-orography-variable |
| `--domain-spec` | strict gpuwm-hrrr-target-domain-v1 Lambert root-domain JSON; nested layouts come from --wps-namelist/--namelist-input |
| `--dry-run` | validate route-specific arguments and print the exact internal command |
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--extend-root-preparation` | sealed HRRR predecessor to extend by exactly one forcing hour |
| `--forecast-end-hour` | inclusive absolute HRRR source lead |
| `--forecast-start-hour` | absolute cycle-relative HRRR lead used for model time zero |
| `--geog-root` | WPS_GEOG root used to build a domain-specific native static cache; requires --domain-spec and replaces --static-cache/--static-receipt |
| `--gfs-series` | tab-separated HOUR and GFS GRIB2 path inventory |
| `--grib` | combined ERA5 GRIB1 series |
| `--grib2-dump` | _(the parser declares no help text for this option)_ |
| `--grib2-inventory` | _(the parser declares no help text for this option)_ |
| `--hierarchy-workers` | bounded mapped d02..dNN initialization workers (1..32) |
| `--history-interval-seconds` | positive output cadence used by HRRR preparation and the prepared-cache forecast identity |
| `--input` | mapped source file; repeat in deterministic time/file order |
| `--list-sources` | print the provenance-bound source capability manifest as JSON |
| `--mapping` | strict rw-wps.mapping.v1 field/coordinate/target contract |
| `--namelist-input` | _(the parser declares no help text for this option)_ |
| `--namelist-support-report` | classify --wps-namelist/--namelist-input and print the exact stock-WRF versus gpuwm support report as JSON |
| `--no-stock-wrf-export` | prepare the forecast only, and do not attempt the bonus unchanged-WRF wrfinput/wrfbdy export of a domain tree |
| `--output-root` | _(the parser declares no help text for this option)_ |
| `--physics-profile` | optional assertion that the experiment IS this shipped single-domain suite, refused on any switch drift; omitted, the config's own physics is prepared as written and its WRF-verification status is reported (the HRRR route still requires a shipped profile: its cold-start evidence contract is profile-keyed) |
| `--pipeline-workers` | _(the parser declares no help text for this option)_ |
| `--prepare-workers` | _(the parser declares no help text for this option)_ |
| `--preprocess-backend {cuda,cpu,auto}` | select CUDA or deterministic parallel CPU preprocessing |
| `--preprocess-workers` | _(the parser declares no help text for this option)_ |
| `--provenance ROLE=PATH` | composition provenance binding |
| `--root-preparation` | sealed output of the native HRRR root-preparation command; enables parallel d01..dNN hierarchy export for max_dom 1..21; the two namelists remain the topology authority |
| `--run-seconds` | _(the parser declares no help text for this option)_ |
| `--sealed-prepared-cache` | opt in to a prefix-sealed operational HRRR root preparation |
| `--show-physics-registry` | print the canonical GPUWM-owned physics registry v2 as JSON |
| `--show-source MODEL` | print one source declaration as JSON |
| `--show-support-matrix` | print the versioned native WRF compatibility matrix as JSON |
| `--source MODEL` | native source adapter id |
| `--source-format {grib1,grib2,netcdf}` | input format; must agree with the sealed rw-wps.mapping.v1 document |
| `--source-manifest, --source-sha256s` | SHA-256 file manifest covering every downloaded source file |
| `--source-manifest-sha256, --source-sha256s-sha256` | expected SHA-256 of --source-sha256s |
| `--source-orography` | _(the parser declares no help text for this option)_ |
| `--source-orography-variable` | _(the parser declares no help text for this option)_ |
| `--source-root` | _(the parser declares no help text for this option)_ |
| `--source-top-pressure-pa` | smallest pressure represented by the selected source; used by --namelist-support-report to reject vertical extrapolation |
| `--static-cache` | _(the parser declares no help text for this option)_ |
| `--static-input` | _(the parser declares no help text for this option)_ |
| `--static-receipt` | _(the parser declares no help text for this option)_ |
| `--statics-corridor GRID_IDS` | also seal child-resolution statics over each child's whole parent extent (the moving-nest corridor); bare flag covers every child domain, or pass comma-separated child grid ids (e.g. 2,3). Required before the prepared tree runner will honor a [relocation] follow source |
| `--stock-wrf-namelist-input` | unchanged-stock-WRF namelist matching the native hierarchy except for the certified LW and moist-theta representation selections |
| `--supplement ROLE=PATH` | composition supplement binding; repeat roles for multiple files |
| `--valid-time` | initial UTC time in WRF form YYYY-MM-DD_HH:MM:SS. On --source hrrr this is the CYCLE; model time zero is cycle + --forecast-start-hour and is derived for every stage |
| `--validate-hrrr-domain PATH` | validate a strict HRRR target domain and its complete native interpolation window |
| `--validate-physics-plan PATH` | validate and resolve a gpuwm-physics-plan-v2 JSON document |
| `--version` | show program's version number and exit |
| `--vtable` | ERA5 GRIB1 Vtable |
| `--wps-namelist` | standard WPS geometry/static-selection namelist |

## `gpuwm-wrf-init`

The same program as `rw-wps`, under its other installed name; every option above applies.
