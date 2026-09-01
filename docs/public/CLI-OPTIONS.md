# Every option, every door

The complete command-line surface, read off the parsers themselves.  It exists because a flag that appears in no document is a feature nobody can reach: `--parent-namelist` gated the whole stock-WRF-parent route, `--tiles` gated the only streamed prepared route, and neither was written down anywhere.

`tests/test_docs_extras_agree_with_code.py` holds this page against the parsers in both directions, so it cannot fall behind the code and cannot name a flag that was removed.  Regenerate it with `python -m tools.build_cli_options_doc` after changing any option.

Everything is listed with the help text the tool itself prints.  A door's positional arguments come first, in the order they are written on the command line and under the names its `--help` usage line gives them: `[NAME]` is optional, `NAME [NAME ...]` repeats.  Where an argument is restricted to a fixed set of values, run that door with `--help` for the list -- it is read from the tree at run time rather than pinned here.  `--help` itself is omitted.

## `arwen-mcp`

Takes no options of its own.

## `gpuwm adapt`

| option | what it does |
|---|---|
| `--descriptor JSON` | completed rw-wps.descriptor.v1 document |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--grib2-dump EXE` | expert override paired with --grib2-inventory |
| `--grib2-inventory EXE` | expert override paired with --grib2-dump |
| `--input FILE` | your own GRIB2 or NetCDF input file, matching the descriptor's declared format (repeat for every file in the series) |
| `--output-dir DIR` | directory for create-only adapter authorities and manifest |
| `--skeleton JSON` | create a review-required descriptor scaffold and stop |
| `--vtable VTABLE` | 11-column WPS Vtable selector authority. Required for GRIB descriptors, and never defaulted: this command adapts arbitrary sources, and quietly reaching for a GFS Vtable would mis-map every other product. Must be omitted for NetCDF descriptors, whose selectors name CF variables directly. A worked GFS example installs with the package -- <gpuwm package>\authorities\Vtable.GFS.rw-wps |

## `gpuwm branch`

| argument | what it does |
|---|---|
| `CONFIG` | the config the source run used; the branch edits a copy of it and the restart identity check refuses any other |

| option | what it does |
|---|---|
| `--allow-shared-gpu` | UNSUPPORTED: permit another substantial CUDA compute context; device verification and the GPUWM UUID lock remain enforced |
| `--directory-input-hash {inventory,content}` | how declared directory inputs (the static geography tree) are bound to this run's identity: 'inventory' (default) uses relative path, size, and mtime; 'content' reads every file and uses its SHA-256. Use 'content' when two runs being compared for byte identity stage their geography separately, and when an mtime-preserving change to that tree must not go unnoticed (docs/public/DETERMINISM.md). Also settable as GPUWM_DIRECTORY_INPUT_HASH. |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from CKPT|latest` | explicit gpuwmrst_*.npz checkpoint to branch from, or 'latest' (default) for the newest valid set in --from-run |
| `--from-run RUNDIR` | the source run's output directory -- where its gpuwmrst_*.npz checkpoints are. Optional only when --from names a checkpoint file explicitly |
| `--gpu-uuid GPU-UUID` | physical GPU UUID to lock (required on multi-GPU hosts) |
| `--health-debug` | enable debug phase health attribution hooks |
| `--no-supervise` | run the experiment in this process (escape hatch; disables fresh-process recovery and exclusive-GPU supervision) |
| `--outdir OUT` | the NEW run's output directory; it must be empty and must not be inside the source run |
| `--prep-timeout SECONDS` | optional preparation heartbeat timeout; default is no timeout until integration begins |
| `--prepare-only` | write the branch run directory, its config and its receipts, then stop without integrating -- the price-it-first step a what-if screen shows before committing a card |
| `--set KEY=VALUE` | a setting to change in the branched run, repeatable. Changeable from a checkpoint: run_seconds, restart_interval_s, acknowledgements, relocation.*, tiles.*, output.*, domain.<grid_id>.history_interval_s, domain.<grid_id>.tiles.*, domain.<grid_id>.output.*. Everything else is refused by name, because the restart identity binds it |
| `--supervisor-max-restarts N` | fresh-process recovery attempts (default 3) |

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

| argument | what it does |
|---|---|
| `CONFIG` | experiment TOML (or legacy RunConfig TOML, wrapped as a one-domain experiment) |

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

## `gpuwm cycle`

| option | what it does |
|---|---|
| `--accept-snap-offset-seconds` | largest analysis-time offset from the parent-step lattice this run will accept by name (default 0.0: the time must land on a step) |
| `--allow-placement-clamp` | accept a placement clamped into the parent instead of refusing it (default off; a clamp is always receipted either way) |
| `--analysis-increment CYCLE=PATH` | the analysis increment applied at CYCLE, as an npz keyed by PROGNOSTIC FIELD NAME. Repeatable. Use CYCLE=null for an explicit NULL ARM (a zero increment that must be bit-stable through the anchor). The three-hash ingestion gate needs both arms to mean anything |
| `--child-dt-seconds` | child model step; must divide the parent step exactly (default: the parent step) |
| `--child-dx-m` | child grid spacing in metres (default 1000.0); must divide --parent-dx-m exactly |
| `--child-nx` | child grid points west-east (default 199) |
| `--child-ny` | child grid points south-north (default 199) |
| `--child-slots` | identically shaped dormant nests reserved at t=0 (default 0). The RESERVATION is fixed and the PLACEMENT is arbitrary: that is what keeps VRAM deterministic while a child can be anywhere |
| `--cycle-seconds` | model seconds between cycle boundaries; must be a whole number of parent steps |
| `--cycles` | how many cycles to run |
| `--dry-run` | print the boundary lattice and the resolved child ratios, refuse invalid combinations, and write nothing |
| `--epoch-anchor ISO8601` | the parent init's config_start_time (UTC); the ONLY datetime in the cycling spine, every other time is an integer tick from it |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--max-forecast-only-cycles` | consecutive cycles allowed with no analysis before the run halts STALE_ANALYSIS_BUDGET_EXHAUSTED (default 3): forecast-only is legitimate, forecast-only forever is a run that stopped being a DA cycle |
| `--min-separation-km` | two children are never planted on one storm (default 40.0). A request inside this radius of an assigned child is REFUSED by name, never silently dropped |
| `--no-resume` | start again at cycle 1 |
| `--parent-dt-seconds` | parent model step; must be a whole number of milliseconds (default 120.0) |
| `--parent-dx-m` | the parent's grid spacing in metres. Required with a placement provider: the child/parent refinement ratio is derived from it and a guessed spacing silently changes every placement |
| `--parent-geo-file PATH` | XLAT/XLONG for the parent's mass grid: a radar-grid NetCDF or an npz carrying both. Defaults to the first --placement-obs-file, whose grid IS the target model grid by contract |
| `--parent-kind {mpas-cuda,mpas-cuda-frames,arwen,replay}` | which engine advances the parent |
| `--parent-mesh-id ID` | the mesh identity written into every anchor. No default: an identity the spine guessed is an identity no downstream reader can trust |
| `--parent-state GLOB` | the parent's state series: a glob of npz frames in the SAME on-disk shape tools/cycle_mpas_leg.py --history reads (flat npz: prognostic fields, time_seconds, and the derived diagnostics beside them). Required for a real run; --dry-run does not need it |
| `--placement-obs-field NAME` | which observation plane the obs provider places on (default z_obs). The radar-grid contract ships z_obs, z_max and z_mean side by side and calls the choice the consumer's; an absent name is refused naming what the file does carry |
| `--placement-obs-file PATH` | radar-grid observation file the obs provider places on. Repeatable, ONE PER CYCLE in order; a single file is reused for every cycle. A storm that never moves between cycles is the defect that hid child retirement for a week |
| `--placement-provider {tracker,schedule,obs,none}` | where each cycle's child placements come from (default none: parent-only cycling) |
| `--placement-threshold` | trigger value a peak must reach to earn a child, in the placement field's own units (default 40.0) |
| `--placement-tracker-field NAME` | which PARENT plane the tracker provider places on (default composite_reflectivity) |
| `--port-config PATH` | the port's case configuration JSON. Required for a model parent kind |
| `--port-root PATH` | the MPAS port checkout the forecast worker runs from. Required for a model parent kind |
| `--port-steps` | dycore steps per cycle boundary. Required for a model parent kind; the step RECEIPTS the worker returns are counted against this number, and a leg that ran fewer steps than asked cannot earn the mpas-cuda stamp |
| `--port-timeout` | seconds to wait for one forecast segment (default: no timeout) |
| `--resume` | continue after the last completed cycle in the ledger (default) |
| `--retire-below-strength` | a child with less than this much signal under it is retired and its reservation returns to the pool. Required with a placement provider and deliberately has NO default: its units are the trigger field's, so a default would be a hardcoded threshold for somebody else's field |
| `--root` | cycle root; the ledger, anchors and per-cycle receipts all live here |

## `gpuwm doctor`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit the checks as JSON |
| `--source {20crv3,20crv3-cf,aifs,aigefs,aigfs,ecmwf-open-data,era5,era5-l137,gdas,gefs,gem-gdps,gfs,hgefs,hiresw,href,hrrr,hrrr-ak,hrrr-prs,icon-eu,mapped,nam,nbm,rap,refs,rrfs,rrfs-a,rrfs-firewx,rrfs-public,rtma,sref,urma,wrf}` | report only this data route's own resolution (repeatable) alongside the shared estate: what its preparation will decode with, and the byte transport its fetch will use. The choices are the source registry -- the same list `gpuwm fetch` and `gpuwm prep` take. Omitted, every route this build knows is reported |

## `gpuwm domain`

| option | what it does |
|---|---|
| `--ack ID` | declare a governed experiment, written verbatim into the emitted [experiment].acknowledgements. Repeatable. This door used to write the nocturnal declaration for you, which silenced the load guard at check/run/go/run-plan and both prepared runners for the life of the file; it no longer does, and refuses instead. The id it accepts is asymmetric-radiation-nocturnal-window-v1: a longwave-OFF suite over a window that includes local night, which you are running deliberately as a daytime validation experiment |
| `--buffer-km KM[,KM...]` | with --polygon, nonnegative geometry buffer in kilometres; one value applies to every domain, or supply exactly one outer-to-inner value per level. With --ladder auto, a multi-value list selects the preset of that depth (default: zero) |
| `--card {12gb,16gb,24gb,32gb}` | GPU tier; sets the VRAM budget with no local probe. With neither --card nor --vram-gib the wizard MEASURES the local card's capacity (short-lived probe, suppressed by GPUWM_NO_LOCAL_GPU) and refuses, naming both flags, when there is nothing to measure |
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
| `--physics-profile {morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1,nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1,nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1,thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1,thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1,p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1,wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1,wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1,thompson-mp8-ysu-mm5-noah-validation-v1,wsm6-ysu-mm5-noah-no-radiation-v1,wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1,wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1,wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1}` | shipped physics suite to emit; taken verbatim from the registry the prepared-forecast runner validates against, so the emitted config passes its guard as written. Read the names: the *-no-radiation-* and *-validation-* profiles run reduced physics with longwave OFF and are NOT nocturnally valid -- selecting one for a window that includes local night is REFUSED unless you declare it yourself with --ack. NOT every profile runs on every route: --source hrrr cannot prepare morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1 or nssl2-mp18-ysu-mm5-noah-kf-rte-rrtmgp-validation-candidate-v1 or nssl2-mp18-ysu-mm5-noah-kf-rrtmg-legacy-validation-candidate-v1 or wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1 or wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1 or wsm6-mynn-mynn-noah-no-radiation-implemented-unverified-v1 or wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1 or wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1; --source gfs cannot prepare wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1 or wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1 or wsm6-mynn-mynn-ruc-no-radiation-implemented-unverified-v1 -- the wizard refuses those pairings and names the missing component rather than emitting a config the front door would reject. (--source era5, the default source, binds morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1; every source has its own computed default and its own admissible set -- `gpuwm run-plan --physics-profiles` prints the whole table) |
| `--point LAT,LON` | domain center in decimal degrees. \|lat\| 90 is refused, and so is any center whose FITTED domain reaches the pole -- a domain containing one is unsupported -- so the usable limit is set by the domain's size, not by the center, and lands well short of 90 (near \|lat\| 72 on the default card and ladder, further equatorward as either grows). The refusal names the fitted size when it fires; the projection is auto-selected from \|lat\| (<25 Mercator, 25-60 Lambert conformal, >60 polar stereographic) unless --projection is set. Negative (southern/western) values work in both forms: --point -33.87,151.21 and --point=-33.87,151.21 |
| `--polygon GEOJSON` | local GeoJSON Polygon, MultiPolygon, Feature, or FeatureCollection; the minimum antimeridian-aware bounds supply the center and every emitted level is fitted around the geometry |
| `--projection {auto,lambert,mercator,polar}` | map projection override (default: auto by center latitude; all three are oracle-gated against WRF v4.6.1 module_llxy) |
| `--root-dx KM` | custom root grid spacing in km [0.05, 200]; use with --chain instead of --ladder |
| `--source SOURCE` | forcing source: any registered source id or alias -- hrrr, hrrr-prs, gem-gdps, icon-eu, gfs, gdas, gefs, aigfs, aigefs, ecmwf-open-data, aifs, rap, rrfs, era5, era5-l137, 20crv3, 20crv3-cf today (`gpuwm prep --list-sources` lists the whole registry). It sets the boundary cadence written into the companion namelist.wps, bounds the domain by the source's own grid where that grid is regional, and (era5) declares [case_data]. A source `gpuwm fetch` cannot download yet emits the same geometry with the acquisition step named instead of a [fetch] table |
| `--vram-gib N` | total VRAM in GiB (alternative to --card) |
| `--vtable` | era5: Vtable override (default: the packaged Vtable.ERA5_CDO, copied beside the TOML) |

## `gpuwm downscale`

`--parent-namelist` (with `--parent-namelist-domain`) is the entire stock-WRF-parent route this command's own summary advertises: without it, only a gpuwm parent run can be downscaled.  `--tiles {on,auto}` and `--child-size` are the only way to stream a `--point`-derived child, because this command authors the child TOML itself and a `[tiles]` table you wrote by hand would be overwritten.

| argument | what it does |
|---|---|
| `parent [parent ...]` | parent wrfout directory or explicit history files |

| option | what it does |
|---|---|
| `--accept-parent-cadence` | accept the archive's own cadence as the ceiling (prints the 15-min guidance when coarser); mutually exclusive with --max-boundary-interval-seconds |
| `--card {12gb,16gb,24gb,32gb}` | VRAM tier for --point sizing (default 24gb; the same tiers `gpuwm domain` accepts) |
| `--child-config` | legacy RunConfig TOML for the child (specified=true, nested=false) |
| `--child-size NX[,NY]` | explicit child extent for --point |
| `--child-surface-from` | child-grid wrfinput/history file with land identity + soil warm start (required for surface-physics children) |
| `--dry-run` | validate contracts, derive/print the plan, write the derived TOML, run nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--health-interval-seconds` | model seconds between child health lines (CFL, w_max, NaN check; default 60) |
| `--hours` | --point run window in hours (default: the full parent archive window) |
| `--i-parent-start` | 1-based west-east parent index of the child's southwest corner (required with --child-config; --point derives it) |
| `--j-parent-start` | 1-based south-north parent index of the child's southwest corner (required with --child-config; --point derives it) |
| `--max-boundary-interval-seconds` | explicit ceiling on acceptable parent cadence (the scientific cadence contract); mutually exclusive with --accept-parent-cadence |
| `--out` | create-only output directory for the child run (report.json, wrfout frames, restart) |
| `--output-interval-seconds` | --point child history cadence (default: the parent cadence) |
| `--parent-domain` | parent domain id when the directory carries several (e.g. 3 for the innermost archived parent) |
| `--parent-namelist` | stock-WRF namelist.input of the parent run |
| `--parent-namelist-domain` | domain column of --parent-namelist (default 1) |
| `--parent-restart` | gpuwm restart of the parent run (authoritative physics evidence) |
| `--point LAT,LON` | derive the child around this point instead of --child-config (gpuwm parents only) |
| `--preprocess-backend {cuda,cpu}` | where the parent-to-child interpolation runs (default cuda; cpu reproduces it off-GPU for verification) |
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

| argument | what it does |
|---|---|
| `[ENS_ROOT]` | ensemble root holding member_NNN/ run directories and ensemble-manifest.json (schema gpuwm-ensemble-manifest.v1) |

| option | what it does |
|---|---|
| `--accept-status LIST` | comma-separated manifest member statuses to accept (default DONE,complete); any other status is a refusal naming the members |
| `--domain dNN` | which domain to plot when members hold more than one (default: the single domain present, else a refusal) |
| `--dpi N` | PNG resolution (default 150) |
| `--engine {auto,rust,matplotlib}` | which renderer draws the panels (default auto). The render law puts weather fields on the Rust renderer; 'auto' uses rw_ensbatch when this checkout has built it and falls back to matplotlib with the reason named, 'rust' refuses rather than substituting, and 'matplotlib' selects the fallback outright |
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
| `--cadence {1,3,6}` | forecast-hour cadence: gfs 1 or 3 (default 3); gdas 1, 3, or 6 (default 3, and it does not apply to --hours 0, which is the analysis alone); era5 template 1, 3, or 6 (default 6); hrrr is hourly. On a table route the accepted cadences and the default are the row's own -- a cadence off the publisher's ladder refuses and names the ladder |
| `--cycle YYYY-MM-DDTHH|latest` | model cycle (UTC); 'latest' resolves the newest cycle this source can serve, from the initialization grid and publication lag its registry row or route declares -- probed against the mirrors where the source publishes objects to probe, and taken from the declared lag where it does not (a reanalysis published on a delay has a latest, and it is that delay). A source that declares neither is refused by name |
| `--engine {auto,rust,python}` | hrrr, and gfs/gdas --mode full-file: which downloader moves the bytes. 'rust' is the vendored rw_fetch backbone (16 MiB parallel range GETs, .idx coalescing, the cross-process NOMADS rate governor, a disk cache); 'python' is the stdlib transport and always works; 'auto' (default) uses the backbone when it is built |
| `--experiment-config TOML` | the experiment TOML the front door will consume |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--fetch-workers N` | how many FILES are in flight at once (default 6; every source but era5, which is a manual CDS retrieval). Bounded per host on top of the pool: NOMADS is capped at 2 in-flight requests and every request still passes the node-wide 2.5 s spacing governor, so concurrency overlaps service time without raising the request rate against a fragile public host. Every file keeps the exact serial verification -- envelope walk, record bar, sha256 -- and one failed file still refuses by name. 1 is the serial transport: a knob, not a workaround. The manifest receipts files, bytes, workers, wall and the effective speedup under 'concurrency' |
| `--force-refetch` | move every existing file in --out aside (nothing is deleted) and re-download this request. The receipts go first -- fetch-manifest.json, SHA256SUMS, the series -- so an interrupted force can never leave a manifest behind claiming payloads it has already replaced; then payloads, .idx indexes, stale parts and anything else in the directory. Files already set aside by an earlier quarantine are left untouched, and subdirectories are yours. Required when re-fetching a different area/cycle into the same --out |
| `--forecast-start-hour K` | every forecast source: the forecast lead the window BEGINS at (default f000, the analysis). --hours stays the window length, so --forecast-start-hour 174 --hours 66 fetches f174..f240 and nothing before it; an experiment whose start_time is cycle+K is then initialized from f{K} with its boundaries from f{K+i}. With --author-front-door-manifest on an already-fetched --out, this authors the manifest over that tail of the existing series instead of re-downloading it |
| `--hours N` | forecast window length: hours 0..N are fetched. gdas is certified for fetch and decode through f009 -- there is no gdas ingest route, so those files stop at the decoder. --hours 0 is the analysis alone, which gdas accepts and every table route accepts (its f000 is an initial state on its own, and it is also how a hybrid source's donor is fetched). A window past the cycle's own horizon refuses and names both the horizon and which cycles reach farther |
| `--manifest-out JSON` | manifest path (default <out>/gfs-input-manifest.json) |
| `--member ID` | ensemble routes (gefs, aigefs): which member to fetch (default the control). Member identity is a PATH component for these products, so the files land under their declared upstream-relative paths and `gpuwm-member-prep --inputs` reads the directory as published |
| `--mode {auto,full-file,idx-subset}` | the byte transport. hrrr (--engine rust): 'full-file' is the default -- the whole object in parallel range GETs, which is the pipeline this product is built on; 'idx-subset' is the opt-in bandwidth saver: it selects records instead of taking the file, saves transfer volume, costs wall clock, and refuses rather than silently degrading when the index cannot carry the selection; 'auto' is the probe rule -- take the whole file when the .idx is absent, malformed, or provably shorter than the object -- which is what an install without the rust backbone falls back to. gfs/gdas: 'full-file' takes the whole pgrb2.0p25 objects from the S3 archive (either engine); omitted, the NOMADS grib-filter crop remains the default, and 'auto'/'idx-subset' refuse -- .idx record subsetting of the raw objects is not a certified GFS route |
| `--out DIR` | output directory (created; complete files are skipped on re-run) |
| `--p-top-pa PA` | gfs/gdas only: the model top (Pa) the fetched atmosphere must reach. The pressure ladder is extended upward along whatever the live inventory publishes until a level sits at or above it, so --p-top-pa 5000 fetches the 70 and 50 hPa levels the certified 100 hPa ladder stops short of. Omitted, the certified 21-level ladder is fetched exactly as before (a 10000 Pa source top). A top the product cannot serve refuses and names the deepest it can |
| `--point LAT,LON` | center point; requires --radius-km |
| `--radius-km KM` | half-width of the box around --point |
| `--source MODEL` | public data source: aifs, aigefs, aigfs, ecmwf-open-data, era5, gdas, gefs, gem-gdps, gfs, hrrr, hrrr-prs, icon-eu, rap, rrfs. Registry aliases work too (gdps, ifs, hrrr-wrfprs). A registered source with no public bytes -- the 20CRv3 every-member archive, the generic 'mapped' adapter -- refuses by name and points at `gpuwm prep --source-root` |
| `--static-input NPZ` | optional prebuilt static cache (with --static-receipt); omit when the front door builds statics from --geog-root |
| `--static-receipt JSON` | receipt for --static-input |
| `--transport {auto,aws,dwd,ecmwf,msc,nomads,s3}` | pin one rung of the source's endpoint ladder. Every NCEP source declares an ORDERED ladder -- the operational server (nomads.ncep.noaa.gov) while it still holds the cycle, the AWS archive behind it -- and the default walks it. Retention decides which rungs are asked: a cycle older than the operational window goes straight to the archive. Throughput decides which one serves: each requested object is HEADed on the archive first and taken there when the archive already has it, because the operational server's head start is spent once both hosts have the same bytes; an object the archive has not caught up with comes from the operational server. A refusal, a 403/503 or a Retry-After moves to the next rung either way. Both hosts serve byte-identical objects under identical keys, so the choice never changes the data. Naming a host here is a decision: it skips the probe, disables fall-through, and refuses in that host's own words. A host a source does not carry refuses and lists the ones it does, because for some products the second copy is a DIFFERENT product (see `gpuwm fetch --source aigfs`) |
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
| `--datasets all|CONSUMER|NAME,NAME` | which datasets to stage (default 'all', every pin -- the 10 above). A consumer name stands for one door's whole set: 'wrf' is the 9 the WRF static builder opens, 'mesh' is what gpuwm mesh needs for the static half of its pair. Use '--datasets wrf' to skip the ~12 GiB Noah-MP soil archive that only gpuwm mesh reads |
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
| `--wif` | also stage QNWFA_QNIFA_SIGMA_MONTHLY.dat (215 MiB), the global monthly aerosol climatology the mp_physics=28 WIF ingest reads (aer_init_opt=1 with wif_input_opt=1), into ~/.gpuwm/wif under the same SHA-256 contract. Opt-in: it is an input dataset, not a coefficient table, and no default install opens it |
| `--wif-only` | with --wif, stage only that dataset and leave the coefficient tables alone |
| `--wif-root DIR` | stage the WIF dataset into DIR instead of ~/.gpuwm/wif (same meaning as GPUWM_WIF_DATA_ROOT) |

## `gpuwm go`

`--no-memory-gate` is the only escape from the pre-fetch memory gate.  The gate runs before the chain downloads anything, and on a box whose card it cannot see it declines to refuse rather than blocking a run that would have worked.

| argument | what it does |
|---|---|
| `CONFIG` | a single-domain GFS experiment TOML emitted by `gpuwm domain --source gfs` -- the wizard's default emission is exactly this shape (--physics-profile optional: an unbound config's own suite runs as written) |

| option | what it does |
|---|---|
| `--data-dir DIR` | reuse an existing `gpuwm fetch` download instead of fetching into <outdir>/data |
| `--dry-run` | print the six commands, filled in, and exit without running any of them |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--geog-root DIR` | staged WPS_GEOG tree (default: the one `gpuwm fetch-geog` stages into) |
| `--no-memory-gate` | skip the before-the-fetch memory check that refuses a configuration whose binding phase cannot fit this card's free VRAM |
| `--outdir DIR` | the case directory: it holds the cached download and one timestamped run folder per run, each carrying that run's authority, prepared, run and png trees (default <config-stem>-go beside the config). Point it at an existing run-... folder and that folder is used as given |
| `--products LIST` | which products the render stage draws: a comma-separated list of catalog slugs, 'all' (the default -- the renderer's whole catalog), or 'none' to stop after the forecast. The same spelling `gpuwm render --products` takes |
| `--run-stamp {on,off}` | put this run's authority, prepared, run and png trees in its own timestamped folder under --outdir (default on): --outdir/run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z/ (launch instant UTC, then the model initialisation time; the _i part is omitted when the run's init time cannot be read). Successive runs of one configuration then never overwrite or interleave each other. 'off' writes straight into --outdir, which is what releases up to 2.4.1 did; it is kept only for a consumer still written against that and is a workaround, not a supported alternative |

## `gpuwm import-namelist`

| argument | what it does |
|---|---|
| `WPS` | WPS namelist.wps (projection + nest layout) |
| `INPUT` | WRF namelist.input (domains/physics/dynamics/bdy_control) |

| option | what it does |
|---|---|
| `--ack ID` | declared-experiment acknowledgement id to write into [experiment].acknowledgements of the resolved TOML (repeatable). WRF namelists cannot spell gpuwm governance declarations, so an import that needs one -- e.g. shortwave-on/longwave-off physics across a window that includes local night -- names the id it wants in its refusal |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--name NAME` | [experiment].name for the resolved TOML (default derived from start time and domain count) |
| `--output TOML` | write the resolved experiment TOML here (omit to print the report only) |
| `--rrtmg-variant {rte-rrtmgp,rrtmg_legacy}` | implementation for a WRF RRTMG 4/4 request: the established RTE+RRTMGP substitution (default, unchanged output) or the exact legacy-RRTMG port (fails closed at physics setup until its compute kernels land) |

## `gpuwm ingest`

| argument | what it does |
|---|---|
| `CONFIG` | experiment TOML ([experiment]/[[domain]] tables, as emitted by `gpuwm domain` or `gpuwm import-namelist`; config-driven runs declare their inputs in [case_data]). A legacy [run]-table RunConfig naming a registered case is also accepted. |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output NPZ` | initialized-state NPZ output |

## `gpuwm mesh`

| option | what it does |
|---|---|
| `--allow-rough-mesh` | WORKAROUND: emit a mesh rougher than the stated smoothness bound. Reported as a workaround in the receipt, never as a setting |
| `--background-km KM` | cell spacing far from every refinement region; with no --refine this is a uniform mesh at that spacing |
| `--card NAME` | size the mesh to this card's measured device footprint; --list-cards prints the ones that have been measured |
| `--cells N` | exact cell count, skipping the device model entirely |
| `--clobber` | replace an existing --out |
| `--dry-run` | size and cost the request, apply both gates, write nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--geog DIR` | WPS_GEOG archive the static's terrain, land use, soil, green-ness and albedo come from. Defaults to $GPUWM_WPS_GEOG, then ~/.local/share/gpuwm/WPS_GEOG |
| `--list-cards` | print the cards in the sizing table and the largest mesh each measured one holds, then exit |
| `--name TEXT` | label carried into the grid file's provenance attributes |
| `--no-static` | WORKAROUND: write only the grid. The result is NOT runnable -- the mesh registry pins grid AND static, so a lone grid file is refused before any dycore sees it |
| `--nominal-dx-m M` | the nominal spacing the static DECLARES, in metres. Defaults to the grid's own implied value. This scalar is compared FP32-bit-exactly by the mesh registry, so a declaration the grid disagrees with is refused |
| `--out GRID.nc` | grid file to write (not needed with --dry-run or --list-cards) |
| `--receipt JSON` | write the measured receipt here as well as to stdout |
| `--refine LAT,LON,KM,KM` | refine a circle: LAT,LON,RADIUS_KM,SPACING_KM and optionally a fifth TRANSITION_KM. Repeatable -- each row is one region and adding one is data, not a code path |
| `--refine-box LAT0,LAT1,LON0,LON1,KM` | refine a latitude/longitude box: LAT0,LAT1,LON0,LON1,SPACING_KM and optionally a sixth TRANSITION_KM. Repeatable |
| `--spec SPEC.json` | read the whole resolution spec from a JSON document instead of building it from --background-km/--refine |
| `--static-out STATIC.nc` | where the matching static goes. Defaults to --out with a .static.nc suffix beside the grid, because the mesh registry admits the two as a pair |
| `--sweeps N` | relaxation budget passed to the generator |
| `--tolerance X` | relaxation convergence tolerance passed to the generator |
| `--triangulation {rebuild,incremental}` | how the Delaunay is kept between relaxation sweeps. rebuild (the default) rebuilds it every sweep and is the arm every registered mesh was generated with -- the only one that reproduces a pinned SHA-256. incremental keeps the facets and repairs them by Lawson flips: the same triangulation, much faster, and a DIFFERENT FILE, because each cell keeps the ring rotation a rebuild re-rolls. For a mesh that has never existed |
| `--vram-gib X` | device budget in GiB, instead of the named card's total memory (for a card that is shared with something else); needs --card, because the fixed term is per card |

## `gpuwm multi-run`

`--preflight {estimate,alloc,off}` is the only override of the plan's own preflight mode.

| argument | what it does |
|---|---|
| `PLAN.toml` | versioned plan with one or more [[run]] entries |

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

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged; gpuwm's own flags must come before the instrument name |

Takes no options of its own.

## `gpuwm obs goes`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged; gpuwm's own flags must come before the instrument name |

Takes no options of its own.

## `gpuwm obs mrms`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged; gpuwm's own flags must come before the instrument name |

Takes no options of its own.

## `gpuwm obs odim`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged; gpuwm's own flags must come before the instrument name |

Takes no options of its own.

## `gpuwm obs opera`

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged; gpuwm's own flags must come before the instrument name |

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

| argument | what it does |
|---|---|
| `ARGS ...` | arguments passed to the instrument's binary unchanged; gpuwm's own flags must come before the instrument name |

Takes no options of its own.

## `gpuwm prep`

| option | what it does |
|---|---|
| `--ack` | registry-owned expert physics acknowledgement id; repeatable |
| `--author-input-manifest` | create an exact mapped or 20CRv3 input manifest; conflicts with an existing --source-manifest/--source-manifest-sha256 pair |
| `--author-mapping` | create-only path for a mapping compiled from --descriptor; the adjacent *.authoring.json receipt binds descriptor/Vtable bytes |
| `--author-only` | author the requested create-only mapped contract or 20CRv3 member manifest and exit; requires --author-input-manifest and does not need run geometry |
| `--bridge` | prebuilt gpuwm all-Rust source-specific GRIB bridge executable; omitted on the era5/gfs routes it resolves through the shared bridge ladder (environment override, a checkout build, staged bridges under ~/.gpuwm/bridges) exactly as gpuwm go does |
| `--canonical-physics-plan-output PATH` | create an exact canonical UTF-8 copy of the plan validated by --validate-physics-plan; refuses an existing output |
| `--child-workers` | bounded CPU worker budget for parallel d02..dNN initialization (1..32) |
| `--composition` | strict gpuwm-mapped-composition-v2 product join contract |
| `--contributing-mapping ROLE=PATH` | cross-source composition: a contributing source's own mapping document under the mapping_role its field_sources binding declares; bytes must hash to the composition's pinned SHA-256 |
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
| `--grib2-dump` | override the GRIB2 dump tool; omitted, it resolves through the shared bridge ladder exactly as --grib2-inventory does |
| `--grib2-inventory` | override the GRIB2 inventory tool; omitted, it resolves through the shared bridge ladder (GPUWM_GRIB2_INVENTORY, a checkout build, the wheel's bundled copy, then the staged ~/.gpuwm/bridges) |
| `--hierarchy-workers` | bounded mapped d02..dNN initialization workers (1..32) |
| `--history-interval-seconds` | positive output cadence used by HRRR preparation and the prepared-cache forecast identity |
| `--input` | mapped source file; repeat in deterministic time/file order |
| `--input-list` | file naming the mapped source files, one path per line, in the same deterministic time/file order the repeated --input flag spells; the spelling that keeps a field-per-file source's hundreds of inputs inside the Windows 32 KB command-line limit |
| `--list-sources` | print the provenance-bound source capability manifest as JSON |
| `--mapped-engine {rust,python}` | which engine decodes mapped source bytes; omitted, the default engine runs. `python` is a documented WORKAROUND -- the slower Python decode path, kept reachable so a decode the Rust engine gets wrong has a way around it while the defect is fixed -- not a supported mode to prefer |
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
| `--vtable` | ERA5 GRIB1 Vtable |
| `--wps-namelist` | standard WPS geometry/static-selection namelist |

## `gpuwm render`

`--pair-labels`, `--pair-subtitle` and `--pair-title` title and label the paired CPU-vs-GPU figure `--pair` composes.

| argument | what it does |
|---|---|
| `[WRFOUT ...]` | wrfout NetCDF file(s) written by gpuwm run |

| option | what it does |
|---|---|
| `--annotate FILE.json` | rust engine: override the panel title and the three subtitle slots (title, title_suffix, subtitle_left, subtitle_center, subtitle_right). A short badge belongs in the centre slot; anything sentence-length belongs on the left, which owns the row's width |
| `--barbs` | rust engine: draw the wind as BARBS, overruling both the automatic choice and any inherited RUSTWX_WIND_STREAMLINES |
| `--dpi N` | PNG resolution, matplotlib engine (default 150) |
| `--engine {auto,rust,matplotlib}` | render engine: the vendored Rusty Weather renderer (campaign plot quality; 151 implicit-render catalog candidates per file) or the matplotlib workaround; 'auto' (default) uses rust whenever its binary is built and probes as runnable, and REFUSES otherwise rather than drawing weather fields with matplotlib -- 'matplotlib' asks for that workaround by name and announces itself |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--heavy` | rust engine: also compute the heavy ECAPE product family at import (SBECAPE/SBNCAPE/SBECIN, ECAPE SCP/EHI/...; adds substantial per-frame import time) |
| `--layout {nested,flat}` | how the PNGs are arranged inside this render's run folder: 'nested' (default) files each picture at <run folder>/<domain>/<product>/<valid-day>/<file>.png (domain as d02-3km / d05-111m / native_grid, valid-day as YYYY-MM-DD), so a run's thousands of frames are separated by nest, by chart and by day and a script can predict a path without globbing; 'flat' writes every picture directly into the run folder, which is what releases up to 2.4.1 did and is kept only for consumers still written against it (with --run-stamp off it is the v2.4.1 tree exactly) |
| `--list-products` | list the engine's product catalog with per-file availability (why each product is or is not renderable from this wrfout) instead of rendering |
| `--out DIR` | where the PNGs go (default out/render). Each render claims its own timestamped run folder under it, so two renders never overwrite each other; point it at an existing run-... folder to draw into that one |
| `--overlays FILE.json` | rust engine: draw map overlays given in geographic DEGREES on every panel -- lines, closed boxes, markers, labels and range rings. This is the seam a boundary-zone frame, tile seams, storm-report markers and radar sites needed; the schema is documented in tools/rustwx/crates/rustwx-products/src/geographic_overlays.rs. Omitted, the renderer runs no overlay code and the PNGs are byte-identical |
| `--pair ('A_DIR', 'B_DIR')` | compose two runs' rendered PNG directories into labeled side-by-side comparison sheets (no wrfout arguments) |
| `--pair-labels ('LEFT', 'RIGHT')` | panel labels (default: the two directory names) |
| `--pair-subtitle TEXT` | optional pair-sheet subtitle |
| `--pair-title TITLE` | pair-sheet title (default 'Paired comparison') |
| `--products LIST` | comma-separated products: refl, t2, wind10, precip, olr, or 'all' (default); with the rust engine, raw catalog slugs (sbcape, srh_0_1km, ...) also work and 'all' renders its full catalog |
| `--run-stamp {on,off}` | put this run's PNGs in its own timestamped folder under --out (default on): --out/run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z/ (launch instant UTC, then the model initialisation time; the _i part is omitted when the run's init time cannot be read). Successive runs of one configuration then never overwrite or interleave each other. 'off' writes straight into --out, which is what releases up to 2.4.1 did; it is kept only for a consumer still written against that and is a workaround, not a supported alternative |
| `--size WxH` | output pixels, rust engine (default 1200x900) |
| `--source-label TEXT` | model/provenance label stamped on every plot (default 'ArWen <the executing version>'); set it when rendering wrfout files this model did not produce, so the sheet does not claim them |
| `--streamlines` | rust engine: draw the wind as STREAMLINES instead of barbs on every product that carries a wind layer. Without either flag the engine keeps its automatic choice (streamlines on curvilinear and projected grids, barbs on plain lat/lon), and the RUSTWX_WIND_STREAMLINES environment variable still works; this flag and --barbs outrank it |
| `--timeidx N|all` | frame index within each file, or 'all' (default) |

## `gpuwm report`

| argument | what it does |
|---|---|
| `[RUNDIR]` | the run directory to collect from (default: the current directory, so `gpuwm report` with no arguments inside a run works; when the current directory holds no receipt but out/run below it does, that one is read and the manifest says so) |

| option | what it does |
|---|---|
| `--dry-run, --list` | print the manifest -- everything that would be included, redacted and reported missing -- and write nothing |
| `--exit-code N` | the exit status the failing command returned, recorded in the manifest (nothing on disk records it) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--log FILE` | an additional log file to include, for output that was redirected outside the run directory (repeatable) |
| `--output PATH` | where to write the zip: a file path, or a directory to name it in (default: the current directory, falling back to the system temporary directory and then your home directory if a write is refused) |

## `gpuwm resume`

| argument | what it does |
|---|---|
| `CONFIG` | the SAME config the interrupted run used; the restart identity check refuses any other |

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

| argument | what it does |
|---|---|
| `CONFIG` | experiment TOML ([experiment]/[[domain]] tables, as emitted by `gpuwm domain` or `gpuwm import-namelist`; config-driven runs declare their inputs in [case_data]). A legacy [run]-table RunConfig naming a registered case is also accepted. |

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

| argument | what it does |
|---|---|
| `[PLAN.json]` | a gpuwm.run-plan.v1 document: which route to execute, which config to execute it with, and where the outputs land |

| option | what it does |
|---|---|
| `--catalog` | print the renderer's product catalog as one JSON document -- what may be put in the render_products run option -- and run nothing; needs no plan |
| `--estimate` | print this plan's VRAM estimate and output-frame counts as one JSON document, and run nothing |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--no-readiness` | with --probe, report the device inventory only: the NVML-only half, safe to poll on a card that is busy |
| `--physics-profiles` | print the per-source physics menu as one JSON document -- every registered source crossed with every shipped physics suite, saying which pairings this product can actually prepare, why each refused one is refused, which suite that source's bare run binds, and which suites run shortwave with longwave off -- and run nothing; needs no plan |
| `--probe` | print this machine's device inventory and runtime-estate readiness as one JSON document; needs no plan. The device inventory is NVML only and creates no CUDA context; the readiness half runs `gpuwm doctor`'s checks, which verify by execution and do create one |
| `--resolve` | print the fully resolved configuration plus every automatic resolution as one JSON document, and run nothing |
| `--sources` | print the source registry as one JSON document -- every registered source, what each one's row declares, and which run-plan route can drive it from an intent -- and run nothing; needs no plan |

## `gpuwm setup`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--from DIR` | stage the bridge and table artifacts from a local directory instead of downloading (offline installs); verification is identical. Does not apply to --with-geog, which has its own --source |
| `--with-geog` | also stage the WPS_GEOG static geography (~2.2 GB compressed, ~30 GB unpacked); the size is printed before anything downloads |

## `gpuwm sim`

| argument | what it does |
|---|---|
| `PREPARED_ROOT` | a prepared tree written by `gpuwm prep --output-root`, by the rw-wps console script, or by `gpuwm go`'s preparation stage |

| option | what it does |
|---|---|
| `--experiment-config TOML` | the experiment TOML this preparation was bound to (the tree runner binds its digest; the single-domain runner binds it through the proof) |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--io-mode {history}` | history output (the only mode this seam offers; `--io-mode none` is a runner-level diagnostic, reachable through --print-command) |
| `--outdir DIR` | where this forecast's output goes. By default it is the parent of one timestamped run folder per forecast, so running the same prepared tree twice never merges two runs' wrfout frames and report.json. Point it at an existing run-... folder and that folder is used as given -- and refused if a forecast is already in it |
| `--physics-profile ID` | optional assertion that the hash-bound experiment IS this shipped suite; omitted, the experiment's own suite runs as written |
| `--print-command` | print the exact runner command, with every digest filled in, and exit without running it -- the documented boundary a third-party script writes to. The --outdir in that line is this run's own timestamped folder, named but not created: asking the question spends nothing, and the runner makes the directory when you run the line |
| `--progress-format {text,jsonl,off}` | how the run reports progress. Omitted, the runner's own default applies, which is the WRF-shaped `Timing for main:` line per step per domain on this terminal -- watching it run is the reason to run the stage alone. `jsonl` is what `gpuwm go` passes, because it owns the runner's stdout; pass it here when you are hosting this stage the same way |
| `--run-stamp {on,off}` | put this run's wrfout, report.json and receipts in its own timestamped folder under --outdir (default on): --outdir/run-<YYYYMMDD>-<HHMMSS>Z_i<YYYYMMDD><HHMM>Z/ (launch instant UTC, then the model initialisation time; the _i part is omitted when the run's init time cannot be read). Successive runs of one configuration then never overwrite or interleave each other. 'off' writes straight into --outdir, which is what releases up to 2.4.1 did; it is kept only for a consumer still written against that and is a workaround, not a supported alternative |
| `--runner {auto,single,tree}` | which runner arm to use. 'auto' (default) reads it off the bundle's own schema and domain count; the explicit values exist for a caller who knows better and wants to be refused precisely when they do not |
| `--wps-namelist WPS` | the namelist.wps this preparation consumed; required for a single-domain forecast, unused by the tree runner |

## `gpuwm sources`

| argument | what it does |
|---|---|
| `[ID]` | print ONE row in full, named by its registry id or any alias it declares (omit for the listing) |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--json` | emit the registry document instead of the table -- the gpuwm.run-plan.sources.v1 schema, narrowed to the one row when ID is given |

## `gpuwm spectral`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `gpuwm spectral check`

| argument | what it does |
|---|---|
| `RECEIPT.json` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--rehash-inputs` | _(the parser declares no help text for this option)_ |

## `gpuwm spectral cross-box`

| argument | what it does |
|---|---|
| `RECEIPT.json` | _(the parser declares no help text for this option)_ |
| `OTHER-RECEIPT.json` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--tolerance` | override the declared tolerance (1e-12); the default is measured, so a campaign widening it says why in its record |

## `gpuwm spectral pins`

Takes no options of its own.

## `gpuwm spectral plot`

| argument | what it does |
|---|---|
| `RECEIPT.json` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--output-dir DIR` | _(the parser declares no help text for this option)_ |

## `gpuwm spectral register`

| argument | what it does |
|---|---|
| `SPEC.toml` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--output REGISTRATION.json` | _(the parser declares no help text for this option)_ |

## `gpuwm spectral run`

| argument | what it does |
|---|---|
| `SPEC.toml` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--plot-dir DIR` | _(the parser declares no help text for this option)_ |
| `--receipt RECEIPT.json` | _(the parser declares no help text for this option)_ |
| `--registration REGISTRATION.json` | _(the parser declares no help text for this option)_ |

## `gpuwm spectral score`

| argument | what it does |
|---|---|
| `REGISTRATION.json` | _(the parser declares no help text for this option)_ |

| option | what it does |
|---|---|
| `--output RECEIPT.json` | _(the parser declares no help text for this option)_ |
| `--plot-dir DIR` | _(the parser declares no help text for this option)_ |

## `gpuwm spectral-op`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `gpuwm spectral-op benchmark`

| option | what it does |
|---|---|
| `--backend {numpy,cupy}` | _(the parser declares no help text for this option)_ |
| `--dx-m` | _(the parser declares no help text for this option)_ |
| `--dy-m` | _(the parser declares no help text for this option)_ |
| `--levels` | _(the parser declares no help text for this option)_ |
| `--nx` | _(the parser declares no help text for this option)_ |
| `--ny` | _(the parser declares no help text for this option)_ |
| `--output` | _(the parser declares no help text for this option)_ |
| `--repeats` | _(the parser declares no help text for this option)_ |

## `gpuwm spectral-op calibrate`

| option | what it does |
|---|---|
| `--dt-s` | _(the parser declares no help text for this option)_ |
| `--input` | _(the parser declares no help text for this option)_ |
| `--output` | _(the parser declares no help text for this option)_ |
| `--protect-wavelength-m` | _(the parser declares no help text for this option)_ |

## `gpuwm spectral-op check`

| argument | what it does |
|---|---|
| `receipt` | _(the parser declares no help text for this option)_ |

Takes no options of its own.

## `gpuwm spectral-op pins`

Takes no options of its own.

## `gpuwm spectral-op response`

| option | what it does |
|---|---|
| `--dt-s` | _(the parser declares no help text for this option)_ |
| `--e-fold-time-s` | _(the parser declares no help text for this option)_ |
| `--maximum-damping-fraction` | _(the parser declares no help text for this option)_ |
| `--maximum-wavelength-m` | _(the parser declares no help text for this option)_ |
| `--minimum-wavelength-m` | _(the parser declares no help text for this option)_ |
| `--order` | _(the parser declares no help text for this option)_ |
| `--output` | _(the parser declares no help text for this option)_ |
| `--protect-wavelength-m` | _(the parser declares no help text for this option)_ |
| `--reference-wavelength-m` | _(the parser declares no help text for this option)_ |
| `--samples` | _(the parser declares no help text for this option)_ |
| `--wavelength-m` | _(the parser declares no help text for this option)_ |

## `gpuwm speedrun`

| argument | what it does |
|---|---|
| `[COURSE]` | the course id to run (`--list` shows them). A course is a row in the shipped course table plus its two asset files; adding one is table work |

| option | what it does |
|---|---|
| `--cold-cache-dir DIR` | EMPTY this directory and point CUPY_CACHE_DIR at it for the run, so a cold-cache record can be set on a machine whose own cache is warm. It never touches the inherited cache |
| `--compare ('A', 'B')` | compare two records. REFUSED, by name, when they are not records of the same course, the same product set and the same compile mode |
| `--compile-mode {cold,warm}` | which kernel-cache class this record belongs to (default: whatever the course declares). The door MEASURES the cache before the clock starts and refuses a mismatch, because the one-time NVRTC compile is roughly a minute and it is always inside the clock -- a cold record and a warm record are different records and are never compared |
| `--determinism ('ARM_A', 'ARM_B')` | the dual-run byte screen: two capsules from two runs of one course on one machine. These cards carry no ECC, so this is the only thing that may set a determinism claim |
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--geog-root DIR` | staged WPS_GEOG tree (default: the one `gpuwm fetch-geog` stages into) |
| `--json` | with --list, emit the table as JSON, including each course's digest and product-set digest |
| `--leaderboard CAPSULE` | emit the SPEEDRUN.md tables for these capsules, one table per comparability class |
| `--list` | list the courses, their product sets and the off-the-clock command that stages each course's bytes |
| `--out DIR` | where the run tree and the capsule go (default speedrun/<course>). The capsule is written as speedrun-capsule.json inside the run's own timestamped folder |
| `--staged DIR` | the directory holding this course's already-staged input bytes. Required to run a course: the clock starts here, so the download must have happened before the door is called |
| `--verify CAPSULE` | verify one capsule's seal and evidence and print its record line, instead of running anything |

## `gpuwm static`

| argument | what it does |
|---|---|
| `CONFIG` | experiment TOML ([experiment]/[[domain]] tables, as emitted by `gpuwm domain` or `gpuwm import-namelist`; config-driven runs declare their inputs in [case_data]). A legacy [run]-table RunConfig naming a registered case is also accepted. |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |
| `--output NPZ` | static-field NPZ output |

## `gpuwm stream`

| argument | what it does |
|---|---|
| `PLAN.toml` | strict gpuwm-stream-plan-v1 orchestration plan |

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `gpuwm update`

| option | what it does |
|---|---|
| `--explain` | print the full reasoning, alternate routes and per-item evidence behind this command's output, instead of the default one-line-per-item summary |

## `gpuwm verify`

| argument | what it does |
|---|---|
| `case` | verification case to run |

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

## `gpuwm-member-prep`

| option | what it does |
|---|---|
| `--cycle YYYY-MM-DDTHH` | the model cycle whose member is staged, in UTC; it selects the declared upstream-relative paths under --inputs and is recorded in the staging receipt |
| `--describe SET` | print one packaged member set's declared members, statistics and products, then exit |
| `--grib2-inventory` | override the resolved inventory executable |
| `--inputs ROOT` | root holding fetched files at their declared upstream-relative paths |
| `--list-member-sets` | print the packaged member sets and exit |
| `--member ID` | declared member id |
| `--member-set SET` | packaged member-set id (see --list-member-sets) |
| `--members-document JSON` | explicit rw-wps.members.v1 document instead of a packaged set (its SHA-256 is recorded in the receipt) |
| `--output ROOT` | root the member-addressed prepared tree is written under |
| `--products P,P` | comma-separated declared products (default: all declared) |
| `--steps H,H,...` | forecast hours to prepare, e.g. 0,3,6 |
| `--verify-only FILE` | verify one file's bytes against --member and print the evidence; nothing is staged |

## `gpuwm-prepared-forecast`

`--tiles JSON` is the only way to stream this route: its hash-bound experiment cannot carry a `[tiles]` table, so the table rides on the flag.  `--render-products` (with `--render-dir`) is render-on-first-committed-frame, off by absence.  `--materialize-authorities` and `--show-capabilities` each select a DIFFERENT program with its own options and must be the first argument on the line.

| option | what it does |
|---|---|
| `--ack` | registry-owned expert acknowledgement id; repeat as needed. The hash-bound experiment's acknowledgements array delivers the same consent |
| `--domain-bundle` | explicit hierarchy d01 bundle; if omitted it is derived from the hash-bound domain-artifacts manifest |
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--frame-markers` | publish OUTDIR/ready/<frame>.json after each history frame is fsynced, self-validated and renamed into place (the default). A marker that exists names a frame that is complete and readable, which is the signal to poll for instead of racing the writer with a size check |
| `--history-interval-seconds` | history cadence; must equal the hash-bound experiment's d01 history_interval_s, and defaults to it when omitted |
| `--io-mode {history}` | _(the parser declares no help text for this option)_ |
| `--materialize-authorities` | create one hash-receipted named-source experiment/WPS authority pair for an exact physics profile, then exit. Run it first on the line and with --help after it for that mode's own options |
| `--no-frame-markers` | do not publish frame-ready markers |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--physics-profile` | optional assertion that the hash-bound experiment IS this shipped suite, refused on any switch drift; omitted, the experiment's own physics runs as written and its WRF-verification status is reported, never gating |
| `--prepared-content-sha256` | _(the parser declares no help text for this option)_ |
| `--prepared-root` | _(the parser declares no help text for this option)_ |
| `--progress-every N` | report every Nth model step (default 1, WRF's own cadence). The first and last step of every domain are always reported, and this thins ONLY `step` records -- output, restart and domain events are never thinned |
| `--progress-format {text,jsonl,off}` | how this run reports its progress. `text` (the default) prints one WRF-shaped `Timing for main:` line per model time step per domain on stdout and ALSO writes the machine stream to OUTDIR/progress.jsonl; `jsonl` writes only that stream, leaving stdout free of sentences; `off` disables per-step reporting entirely |
| `--progress-output PATH` | where the machine stream is written; defaults to OUTDIR/progress.jsonl. Append-only JSONL at gpuwm.step-log/v3, one record per printed line, with a dense `sequence` so a consumer can detect a lost line. `-` sends the records to stdout instead of to a file, which with --progress-format jsonl is a pure record pipe |
| `--proof-sha256` | _(the parser declares no help text for this option)_ |
| `--render-dir DIR` | where --render-products publishes; defaults to OUTDIR/png. Ignored without --render-products |
| `--render-products SPEC` | `gpuwm render --products`' own spec -- a comma-separated product list, or `all`, or `none` -- for the FIRST frame this run commits, rendered on a worker thread while the forecast is still integrating. Absent is off, and off is the default: there is deliberately no second switch, so "which products" has one answer that cannot disagree with itself. The first frame is the analysis at t = 0, durable before a single step is integrated |
| `--run-seconds` | forecast length; must equal the hash-bound experiment's run_seconds, and defaults to it when omitted |
| `--show-capabilities` | print this runner's capability JSON and exit; it must be the only argument |
| `--source {20crv3,20crv3-cf,aifs,aigefs,aigfs,ecmwf-open-data,era5,era5-l137,gdas,gefs,gem-gdps,gfs,hrrr,hrrr-prs,icon-eu,rap,rrfs}` | _(the parser declares no help text for this option)_ |
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
| `--source {20crv3,20crv3-cf,aifs,aigefs,aigfs,ecmwf-open-data,era5,era5-l137,gdas,gefs,gem-gdps,gfs,hrrr,hrrr-prs,icon-eu,rap,rrfs}` | _(the parser declares no help text for this option)_ |

## `gpuwm-prepared-tree-forecast`

`--sealed-forcing-extension` selects the append-only forcing-prefix checkpoint contract.

| option | what it does |
|---|---|
| `--experiment-config` | _(the parser declares no help text for this option)_ |
| `--experiment-config-sha256` | _(the parser declares no help text for this option)_ |
| `--frame-markers` | publish OUTDIR/ready/<frame>.json after each history frame is fsynced, self-validated and renamed into place (the default). A marker that exists names a frame that is complete and readable, which is the signal to poll for instead of racing the writer with a size check |
| `--health-debug` | _(the parser declares no help text for this option)_ |
| `--io-mode {history,none}` | _(the parser declares no help text for this option)_ |
| `--no-frame-markers` | do not publish frame-ready markers |
| `--outdir` | _(the parser declares no help text for this option)_ |
| `--preparation-receipt-sha256` | _(the parser declares no help text for this option)_ |
| `--prepared-root` | _(the parser declares no help text for this option)_ |
| `--progress-every N` | report every Nth model step (default 1, WRF's own cadence). The first and last step of every domain are always reported, and this thins ONLY `step` records -- output, restart and domain events are never thinned |
| `--progress-format {text,jsonl,off}` | how this run reports its progress. `text` (the default) prints one WRF-shaped `Timing for main:` line per model time step per domain on stdout and ALSO writes the machine stream to OUTDIR/progress.jsonl; `jsonl` writes only that stream, leaving stdout free of sentences; `off` disables per-step reporting entirely |
| `--progress-output PATH` | where the machine stream is written; defaults to OUTDIR/progress.jsonl. Append-only JSONL at gpuwm.step-log/v3, one record per printed line, with a dense `sequence` so a consumer can detect a lost line. `-` sends the records to stdout instead of to a file, which with --progress-format jsonl is a pure record pipe |
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
| `--bridge` | prebuilt gpuwm all-Rust source-specific GRIB bridge executable; omitted on the era5/gfs routes it resolves through the shared bridge ladder (environment override, a checkout build, staged bridges under ~/.gpuwm/bridges) exactly as gpuwm go does |
| `--canonical-physics-plan-output PATH` | create an exact canonical UTF-8 copy of the plan validated by --validate-physics-plan; refuses an existing output |
| `--child-workers` | bounded CPU worker budget for parallel d02..dNN initialization (1..32) |
| `--composition` | strict gpuwm-mapped-composition-v2 product join contract |
| `--contributing-mapping ROLE=PATH` | cross-source composition: a contributing source's own mapping document under the mapping_role its field_sources binding declares; bytes must hash to the composition's pinned SHA-256 |
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
| `--grib2-dump` | override the GRIB2 dump tool; omitted, it resolves through the shared bridge ladder exactly as --grib2-inventory does |
| `--grib2-inventory` | override the GRIB2 inventory tool; omitted, it resolves through the shared bridge ladder (GPUWM_GRIB2_INVENTORY, a checkout build, the wheel's bundled copy, then the staged ~/.gpuwm/bridges) |
| `--hierarchy-workers` | bounded mapped d02..dNN initialization workers (1..32) |
| `--history-interval-seconds` | positive output cadence used by HRRR preparation and the prepared-cache forecast identity |
| `--input` | mapped source file; repeat in deterministic time/file order |
| `--input-list` | file naming the mapped source files, one path per line, in the same deterministic time/file order the repeated --input flag spells; the spelling that keeps a field-per-file source's hundreds of inputs inside the Windows 32 KB command-line limit |
| `--list-sources` | print the provenance-bound source capability manifest as JSON |
| `--mapped-engine {rust,python}` | which engine decodes mapped source bytes; omitted, the default engine runs. `python` is a documented WORKAROUND -- the slower Python decode path, kept reachable so a decode the Rust engine gets wrong has a way around it while the defect is fixed -- not a supported mode to prefer |
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
