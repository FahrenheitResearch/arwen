# Changelog

## 2.8.6 (2026-10-05)

New:

- Multi-card `[devices]` forecasts step their slabs at once under a free-threaded Python 3.14 (`python3.14t`): each slab's host thread issues its own card's work instead of taking turns on the interpreter lock. The `gpuwm` command line and the prepared runner re-execute once with `PYTHON_GIL=0` on a free-threaded build; receipts and `gpuwm doctor` record the interpreter and lock state. Full HRRR physics on the 1797 x 1057 x 50 grid: 4 x RTX PRO 6000 130.0 s per forecast hour against 333.8 before, 8 x RTX 5090 83.8 s. Histories are byte-identical across 2, 4 and 8 cards and both interpreters.
- Published static files can be selected through `[static] source`. HRRR configurations select the pinned operational CONUS static file by default on its native grid and contained cuts. Soil, land use, terrain, monthly vegetation, lakes and drag fields retain the source values. Analyzed vegetation fraction comes from the initialization GRIB.
- `terrain_clock = "pinned"` preserves the configured clock at launch. With `use_adaptive_time_step = false`, `time_step` and `time_step_sound` run exactly as written; a selected adaptive controller still updates its live clock. The terrain clock and the steep-ground substep rule still read the domain and record what they would have done as advice in the `terrain_clock` and `acoustic_substeps` receipts (`clock = "pinned"`, `advice` with `applied = false`) and print it, but apply nothing. `"measured"`, the default, is unchanged. The off-centering floor is read in both modes.
- `gpuwm import-namelist` reads a single-domain WRF namelist whose adaptive clock has `starting_time_step = max_time_step = min_time_step` (HRRR v4: 20/20/20) as the fixed step it integrates. It writes `use_adaptive_time_step = false`, that step, the acoustic substep count WRF derives for it from the step and the grid's largest map factor (`solve_em.F`; 6 for HRRR's 1799 x 1059 grid at 20 s, where its namelist spells 4), and `terrain_clock = "pinned"`, and the import report lists each substitution. A ranged adaptive clock and any nested namelist import as before.
- The named GSD v4.1 MYNN form converts moisture output with updated vapour and the actual original mixing ratios. The source pre-mixing condensate heat and negative-condensate clipping are available together through explicit `bl_mynn_cloud_tendency_form = "gsd_41"`; its `wrf_461` default keeps the conserving repair and mixed-condensate heat. HRRR import and recipe defaults keep that defect option off. Generic MYNN remains byte-identical.
- The GSD v4.1 MYNN TKE predictor now uses the source zero-gradient top, 1e-4 floor and uncapped solution, with its 1e-12 initialization production floor. All 96 oracle TKE words match the unmodified source. This applies through the named HRRR importer and explicit source recipe; the global MYNN default remains wrf_461.
- The optional GSD v4.1 MYNN form ports the complete plume population, surface excess, trigger, motion and height tapers, entrainment, scale awareness and density-free transport block. Default MYNN PTX remains byte-identical. Fork oracles cover 17 plume columns and eight clear transport columns; day and night forecast cuts are recorded in the lane report. This remains a partial fork version.
- The named hrrr_wrf.nl importer selects `bl_mynn_version = "gsd_41"`, including when the legacy MYNN budget key is omitted. Explicit source metadata wins. The native-spacing HRRR recipe and `configs/recipes/hrrr_v4_gsd41.toml` explicitly select that source version with mixing length 2 and legacy RRTMG. Generic defaults and existing compositions retain `wrf_461`. The source version remains partial; see `docs/dev/mynn-gsd41.md`.
- `bl_mynn_version = "gsd_41"` runs the MYNN boundary layer of the operational RAP/HRRR WRF branch (GSD MYNN v4.1): its dew and frost water, mixing length 2 and subgrid stratus and shallow-cumulus cloud. Checked against that branch's Fortran (stratus bit for bit, mixing length within 2 ULP). On a 3 km cut of the operational configuration, daytime 2 m dewpoint bias against the operational model fell from +0.93/+0.95 K to +0.69/+0.85 K at f01/f02. `"wrf_461"` stays the default, byte for byte.
- Sun-angle land albedo updates `ALBSOL` and `ALBBCKSOL` on radiation steps, supplies shortwave radiation and RUC, and preserves the fractional sea-ice blend. New key `alb_sol` accepts 0 or 1, with general default 0. HRRR namelist imports honor their `alb_sol = 1`, the shipped HRRR demos and the new solar-albedo monthly RUC legacy-RRTMG template select 1 explicitly, and other configurations retain their prior albedo behavior.
- HRRR plans retain the fork physics default on fresh installs. The fetch stages the required monthly aerosol climatology in its shared cache before forcing transfer, with exact size and SHA-256 verification. Dry runs report the pending dependency. Fixed UCAR input downloads work without bridge-release pins; explicit mirrors and offline copies remain supported. `[fetch] wif = false` retains the actionable offline refusal.
- New key `ruc_qvg_cold_start` (`"wrf"` default, `"air"`): `air` starts the RUC ground vapour of a cold start from the lowest-level air with no ground condensate, the fallback of the operational RAP/HRRR branch of LSMRUC, which cycles the ground vapour. `wrf` keeps public WRF's saturation at the skin times moisture availability. SOILTEMP carries the old ground vapour as storage, so `air` pulls the skin toward the air's dewpoint on the first steps (measured 2.7 K colder after 20 steps on a moist test column); on a 3 km cut of an operational-HRRR start it moves the 2 m dewpoint by -0.013 to +0.018 K.
- New key `ruc_2m_diagnostic` (`"flux"` default, `"log_profile"`): `log_profile` selects the operational RUC logarithmic 2 m block. The fork copies its alternate diagnostics into the main fields. HRRR namelist imports and the HRRR recipes select this form; generic runs retain `flux`.
- Importing a supported namelist named `hrrr_wrf.nl` selects legacy RRTMG and the operational cloud wrapper form. The shipped `hrrr_configuration_clock.toml` recipe selects aerosol optics 3, shortwave interpolation 1, the 15 minute interval and analyzed Thompson aerosol input. Generated route namelists retain both analyzed aerosol flags. The selected cloud form uses the source SW fallback radii, snow optical mass and non-restart radii; it retains scheme radii during the MYNN merge. Generic requests keep their previous defaults. Forecast and station improvement require the paired qualification described in the shortwave report.
- Generated adaptive nests explicitly declare a maximum step growth of 5 percent. Their existing parent growth, 1.2/0.84 CFL targets and WRF per-resolution bounds are retained; generated single-domain and fixed-clock documents are unchanged. Generated children already resolved to 5 percent, so this makes their policy explicit. A configuration that sets a child's growth to 51 percent itself keeps that value and is named by `gpuwm check`. A 1 km and 500 m nested forecast that failed at 51 percent completed 26 hours at 5 percent with no measured cost; lower CFL targets and tighter per-resolution step caps were measured at about 2.1 times the run time and are not adopted.
- `diff_6th_form`, `diff_6th_factor2`, `mp_zero_out` (with `mp_zero_out_thresh` and `mp_zero_out_all`) and `upper_wind_limiter_form` select the sixth-order filter, microphysics zero-out and upper wind limiter of the operational RAP/HRRR WRF branch (`"noaa_wrf39"`). Defaults are the WRF v4.6.1 forms and no zero-out; the HRRR namelist importer and HRRR recipes select the branch forms. At HRRR's 20 s step the acoustic substep count is the six WRF derives from the grid, not the four written in the namelist.
- `v_sca_adv_order`, `v_mom_adv_order` and `h_mom_adv_order` set WRF's vertical scalar, vertical momentum and horizontal momentum advection orders (defaults 3, 3 and 5). The HRRR recipes select fifth-order vertical advection.
- `thompson_version = "wrf_39_noaa"` runs the Thompson microphysics of the operational RAP/HRRR WRF branch with its own lookup tables, which the first run acquires from pinned public source with hash checks. The default stays `"wrf_461"`; the HRRR importer and recipes select the branch version. `thompson_fork_snow_fall` keeps WRF v4.6.1's melting-snow fall speed (`"blend"`) by default; the branch's own expression, which is singular just above freezing, is available only by explicit selection.
- `mynn_sfclay_variant = "gsl_wrf39"` runs the MYNN surface layer of the operational branch; the default is `"wrf_461"` and the HRRR doors select the branch form. `fractional_seaice = 1` uses WRF's fractional sea-ice threshold with RUC.
- `gpuwm fetch --source 20crv3-cf` reads NOAA PSL 20th Century Reanalysis v3 ensemble-mean subsets (1836 to 2015, three-hourly) with native fields and four soil layers, so a forecast can start from a historical date.
- A finished forecast is verified against observations by default: station dots and scorecards, and MRMS reflectivity and hourly rain panels, read, scored and drawn in Rust. The run waits at most 2 seconds for it at the end; observation downloads and scoring continue in the background and `gpuwm verify-visuals RUN_DIRECTORY` runs it on demand. A verification failure never changes a forecast's result, and `GPUWM_VERIFY_VISUALS=0` turns it off.
- Packed ensembles carry a parent and its nest together, with each member's own nest forcing and feedback, adaptive clock, MYNN, RUC and radiation; operations not yet packed fall back to the member-by-member path and the run records which. Ensemble products can be reduced as members finish by a separate CPU process, with every delivered file byte-identical.
- An interrupted streamed nested forecast or ensemble continues from its last checkpoint: the nested continuation and each member resume with identical arrays, and completed members are skipped.
- `[physics_params]` (or `GPUWM_PHYSICS_PARAMS`) applies a named set of physics parameter values, recorded in the run's identity and output attributes. Default runs are unchanged.
- A regional rain scorer: Rust hourly rain, echo and fractions-skill-score kernels with a native MRMS decoder and a command-line scorer.
- Comparison sheets can draw MRMS and RRFS as reference panels and take several references in a stated order. The renderer's WOOF themes, layout plans and run-difference and pair sheets are part of the main line, and maps and sheets name the model WOOF for files this engine wrote.

Fixed:

- The GPU memory watcher reads `nvidia-smi` on its own thread, not the forecast's: a nested forecast on a busy 8-card host ran 130.5 s per forecast hour, now 8.0.
- Isobaric heights no longer read high: every pressure-level height paired a layer's mean height with its mid-level pressure, so 500 hPa heights read 5.3 m high on a 3 km frame, and the 500 hPa difference from HRRR's own 12 h forecast fell from +6.2 m to +0.9 m with the fix. Heights are now read between the layer interfaces for height charts, pressure-level planes, sounding heights, post-processed files and ML export. Answers change for every isobaric and sounding height product.
- The same height fix reaches the moving-nest vortex tracker, the local DA GNSS-RO refractivity operator, the verification maps and the flagship products, all through one Rust reader shipped as the bundled library `rw_isobaric` (with a bit-identical CUDA copy for the tracker). Tracker centre heights and RO answers change; tracker centre positions barely move.
- Standalone Rust NetCDF builds use the same corrected HDF5 reader as the engine, so compressed NetCDF-4 files with dense attributes read without a fractal-heap checksum mismatch. Consumer workspaces no longer need to patch the facade's reader dependencies.
- `[devices]` with `transport = "auto"` stages every cross-card halo copy when the run's cards sit on more than one NUMA node. Concurrent peer copies that mixed in-socket and cross-socket card pairs took 245 ms per HRRR exchange against 15 ms staged, so four cards ran slower than two. The receipt names the reason on each staged pair; an explicit transport is unchanged.
- On hosts with more than one NUMA node, each slab's thread runs on the CPUs local to its card.
- The halo exchange packs and unpacks each seam in one launch and moves each slab's seams as soon as its step returns, instead of 3,568 pitched copies per HRRR step issued after the slowest slab. The bytes moved are unchanged.
- A domain that is a source's own declared Lambert grid (the same mass dimensions, every point on its own source cell; for `hrrr-prs` and `hrrr-native`, the full 1799 x 1059 HRRR grid) prepares untrimmed. Each point copies its own source cell and the outermost u and v faces take the edge cell, and the preparation proof records `source_pairing = "identity"`. It was refused because its outermost cell corners and faces lie half a cell past the source's outermost points, and the advice was a one-row trim that moved the boundary zone one row inward. Every other domain is interpolated as before.
- Detached terminal workers restore their interrupt handler so Stop ends their owned process tree gracefully.
- GRIB2 readers preserve valid complex-packed all-missing fields as bitmap NaNs.
- Hex initialization geometry and reconstruction coefficients run in Rust. All four arrays preserve their float32 or float64 bits; the measured 157,859-cell geometry job fell from 28.46 s to 0.223 s.
- Observation FSS, contingency and station scoring maths run in Rust, including station report matching and quality screens. Float64 reduction and threshold results preserve the previous bits.
- Noah soil liquid-water initialization runs in Rust with the previous float64 arithmetic and frozen-soil threshold.
- Hex forecast topology checks, terrain metrics and initialized vertical momentum run in Rust. Scalar operation order, mixed float32/float64 promotion, per-edge storage rounding and validation failure order are preserved.
- The shared NetCDF reader accepts arrays above 134,217,728 values. Numeric reads use checked 64-bit shapes, bounded hyperslabs and fallible dense-result allocation. WPS, Zarr and renderer tools use the same reader source; the former volume guards are retired.
- Chunked NetCDF-4 reads decode signed and unsigned 64-bit coordinates and preserve each integer type's signedness when promoting values. `gpuwm adapt` can author adapters from sources with int64 time coordinates.
- Direct NVRTC radiation, terrain-drag and stochastic surface compilations retain their exact binary images across processes. Cache identity binds source, options, architecture and compiler/runtime build; corrupt or incompatible entries compile normally.
- Prepared host-store startup verifies read-only array mappings with bounded Rust hashing and reuses those mappings for slab reads and coordinate/base arrays. Complete payload digests and changed-file checks remain enforced.
- Store initialization batches rows within the existing complete memory envelope and retains a one-row template. Unknown device budgets keep the original column cap.
- Host health gates scan dense floating fields in Rust without allocating derived arrays or masks. Status classes, auxiliary rounding and first-failure attribution retain their existing rules.
- Repeated CPU vertical interpolation reuses bounded immutable pressure geometry. Each field keeps the original interpolation expression and validation; changed pressures and older bridges use the original path.
- Legacy radiation interpolates ozone latitude and time in Rust using the two needed months. It preserves separate float32 operations and avoids full twelve-month grids and NumPy intermediates for each moving tile. Older bridges and unusual floating-point policies retain the original path.
- RUC's `log_profile` diagnostic set applies the final 2 m humidity bound over land and water after the lake model, matching the operational fork. Q2 cannot exceed 1.05 times the lowest-level vapour mixing ratio. The generic `flux` form retains the land-only bound.
- The operational HRRR namelist importer and both HRRR configuration recipes select prescribed monthly LAI and background albedo, the fork's ground-vapour cold start and its logarithmic 2 m diagnostic. RUC no longer replaces prescribed LAI with its vegetation-table seasonal value on those configuration paths. Generic configuration defaults are unchanged. A prepared native bundle can be reused across these runtime selectors.
- Checkpoint configuration echoes accept their own omitted default filter and microphysics zero-out settings. Explicit changes to those settings still refuse continuation of a different model.
- RUC irrigation can select the operational WRF v4.0-4.5 crop-fraction floor with `ruc_irrigation = "wrf_45"`. The generic default remains `"wrf_461"`, the previous per-step relaxation under `mosaic_lu = 1`. The operational HRRR namelist importer and shipped `configs/recipes/conus_hrrr_configuration.toml` and `hrrr_configuration_cut.toml` select `"wrf_45"` explicitly. Custom configurations keep the generic form unless they name the switch. In one cycle the fork form made station dewpoint 0.05 to 0.10 K moister, so it is not a global default. SOILPROP stays globally `"wrf_45"`, as shipped in 2.8.5.
- RUC snow can select the operational WRF v4.0-4.5 conductivity, cover, albedo and melt set with `ruc_snow = "wrf_45"`; the default stays `"wrf_461"`, and the HRRR importer and recipes select `"wrf_45"`. On one 3 km cut, snow-covered T2 bias against HRRR fell from +0.68/+1.13 K to +0.31/+0.48 K at f01/f02, while ASOS RMSE rose from 0.94/1.05 K to 0.97/1.13 K. Checkpoints from the earlier default are refused on continuation.
- High-resolution geography fetches serialize each shared cache entry across processes, use unique staging files, validate payloads before atomic publication, and verify cached hashes before reuse. Derived terrain and land-cover windows and extracted archive rasters use the same publication discipline.
- The Rust high-resolution raster path reads source blocks and streams derived GeoTIFF tiles instead of retaining full terrain mosaics and encoded files. Merge registration accepts separate field buffers, and native merges borrow their inputs without cloning both field sets.
- Physics setup repairs damaged packaged tables in its selected user cache through verified atomic replacement. Explicit table roots still refuse damaged files.
- `gpuwm check` now names an adaptive nest whose `max_step_increase_pct` is above 5 (WRF's own nest value is 51, so imported namelists carry it): after an output alarm a child at that growth has been reproduced taking steps about three times its spacing rule and failing with a non-finite vertical velocity hours into a nested forecast, while 5 ran the same forecast to completion at the same pace. It is an advisory and blocks nothing; generated nests already use 5.
- A nest keeps its freshly measured CFL when an output alarm lands on a full, unshortened step, instead of continuing on stale step memory, which had ended a nested forecast with a non-finite state in its parent domain. Step-memory protection now applies only to steps the alarm actually shortened.
- A nested health failure is retried at most twice from a verified checkpoint with a lowered step ceiling, and every retry and its outcome is written to the run receipt. Nest forcing observations are reset after a restore from checkpoint.
- An HRRR-native start no longer refuses when the source's top moist-pressure level sits a fraction of a millipascal above the dry model top. The top source value is used within about 0.026 Pa of the top pressure, with no extrapolation; a real gap is still refused by name.
- A completed forecast keeps its exit status and final-state digest when another process deletes history frames during the run; the digest comes from model state and the receipt carries per-frame hashes.
- HRRR-configuration runs select aerosol optics 3, shortwave interpolation 1 and the operational cloud form: on a full-grid cycle the surface shortwave excess against HRRR fell from about +31 to +4 W/m2. Prescribed monthly leaf area and albedo with the operational RUC forms bring the afternoon 2 m dewpoint bias from about +0.5 to +0.8 K down to HRRR's own on two full-grid cycles, with 2 m temperature error unchanged.
- A physics template's own parameters win over its component options. The GSD v4.1 MYNN suite declared mixing length 2 and scalar boundary-layer mixing and ran the component's 1 and off.
- A prepared bundle whose domain is its source's own grid is accepted by the forecast runner; its proof carried a pairing entry the runner refused as unknown.
- Simulated-radar requests survive preparation: the published experiment document keeps its `[simulated_radar]` table.
- Process start to first forecast output on a 1797 x 1057 x 50 case fell from 113.8 to 26.6 minutes, with identical output. Prepared arrays stay mapped through about five file descriptors instead of one per array, so cases with more than a thousand arrays run at the usual 1,024 open-file limit, and running out of descriptors is reported as a resource limit, not as cache corruption.
- High-resolution static preparation streams source reprojection through bounded windows: a full-domain 48-worker preparation that climbed to 322 GiB now holds about 160 MiB per reader, with all 21 static fields byte-identical.
- The namelist importer accepts 43 inert `&time_control` output-stream settings it used to refuse, and still refuses an unknown physics-generation selector. Scalar mixing accepts an explicit `mix_full_fields = false` on a zero base state.
- The 2-D Smagorinsky kernel dispatches on 32-bit indices, about 1.6 percent faster per step and byte-identical. A rounding difference on Blackwell cards is removed.
- The Weather Library name and its compatible browser and API routes return.
- GFS preparation budgets its decode against host memory instead of the card's, and holds a window of decoded leads instead of every one. With `GPUWM_FETCH_POLICY=aws` a 240 h run reads 81 whole-globe leads; on a 64 GiB host with a 96 GB card it planned against 93.93 GiB and was killed out of memory, and it now prepares within the host. Prepared values are identical.
- A packed data-assimilation step resumed with `--resume-ensemble` reproduces the uninterrupted run. Computing the radar observation operator no longer zeroes graupel below its reflectivity threshold in the model state, which also changes data-assimilation trajectories from previous builds.
- The memory preflight prices the operational-branch Thompson sedimentation frames and the operational sixth-order filter's working set at their real size. The HRRR recipes had been under-priced, by about 0.5 GiB on a 32 GB card for sedimentation alone.
- Legacy RRTMG no longer builds a twelve-month ozone latitude grid at start-up that nothing read once the radiation call took its two months from the live latitude in Rust. The host memory it held is freed: 326 MiB on the 401 x 301 HRRR configuration cut (peak resident memory of a 30 minute run 4.29 to 3.79 GiB) and 5,145 MiB on the full 1799 x 1059 HRRR grid. Forecasts are byte-identical, single card and split across cards.
- Two gpuwm versions on one machine no longer overwrite each other's Rust bridges. A released install stages its bundle in `~/.gpuwm/bridges/<release>-<bundle digest>`; the older flat directory is read only for files whose bytes match this release's pin and is never written. A bundle that fails verification for any member installs none of them, and `gpuwm doctor` names the directory in use. The `GPUWM_*` bridge overrides are unchanged.
- Windowed products (6 h precipitation and the other accumulation and maximum windows) on a regular latitude-longitude grid, such as a WOOF Global or GFS map, are drawn the way the direct products beside them are. A whole-globe 6 h precipitation map no longer streaks across the globe where rain crosses the date line, and a regional crop (North America, Europe) no longer shrinks the precipitation panel to a thin strip. Projected grids (HRRR, WRF Lambert) draw byte-identical pictures, as do the direct products.
- Scalar transport at vertical order 5 (`v_sca_adv_order = 5`, the HRRR recipes) no longer creates mass. Its positive-definite final stage read the downstream cell at face Courant numbers up to 1, drained empty cells, and the zero clamp turned the drained amount into new water; every such face now takes the upwind flux, as WRF 4.7.1 does. In a 12 h 3 km forecast the clamps had added 181,000 t of cloud ice; now 25 t. Answers change for order-5 runs; order 3, the generic default, is unchanged.
- The GF, New Tiedtke, urban, UH and Noah mosaic oracle fixtures moved from the package to the source repository's `tests/data`, so the platform wheels fit PyPI's 100 MB file limit again.
- A WPS namelist whose `ref_x`/`ref_y` name the grid centre imports to the same bytes on Linux and Windows: `ref_lat` and `ref_lon` are carried exactly instead of through a projection round trip (Linux had emitted 38.49999999999998 for 38.5).

## 2.8.5 (2026-10-03)

New:

- `rap-native` fetches and prepares RAP's 13 km hybrid-level columns through the shared source tables. `hrrr-native` supplies a native hybrid analysis, with its same-cycle pressure product supplying the soil column. `gpuwm prep --initial-inputs JSON` starts a single mapped domain from a separately declared analysis while keeping every lateral frame, including the first, from the boundary source. Both sources are bound into the prepared evidence.
- `use_rap_aero_icbc = true` takes aerosol number initial and boundary values from the analyzed input, with the operational nearest-neighbor and missing-value fallback sequence. A separate initial analysis uses the boundary analysis's own pressure column for aerosol interpolation. The operational surface-emission formula continues to read the monthly climatology. Missing analyzed aerosol fields are refused before a climatology can replace them.
- MYNN supports WRF mixing length 2 and `scalar_pblmix = 1` for aerosol-aware Thompson number scalars. The scalar option uses WRF's post-PBL diffusion through MYNN exchange coefficients. Both settings import from WRF namelists.
- RUC mosaic land use and soil (`mosaic_lu`, `mosaic_soil`) and WRF's CLM lake model (`sf_lake_physics = 1`), with column oracles, restart state and resident-rank carriers. WRF's off defaults remain unchanged. Lake-depth geography follows WPS sampling.
- `gpuwm ensemble CONFIG --members N` (also `gpuwm go` and `gpuwm run` with `--members N`, or `[ensemble] members = N` in the config) runs N members, each from its own source trajectory. With no recipe named, the members come from the source's operational ensemble (`gefs` for `gfs`, `aigefs` for `aigfs`, `ecmwf-ens` for `ecmwf-open-data`) and are fetched and prepared like a named recipe; the member plan prints first, and `--dry-run` prints only the plan. A source with no operational ensemble exits 2 before any download and names `--recipe time-lagged` and `--trajectories FILE`, because N copies of one forecast have zero spread and probabilities of 0 or 1; `--wrfinput`, `--met-em`, `gpuwm resume`, `gpuwm branch` and a `gpuwm run-plan` route that prepares one trajectory refuse N > 1 for the same reason. A plain member count answers the same flags as a recipe: `--cycle` re-times the config and plans the members at that cycle, `--readiness` answers for the members' windows, and `--prepared-root`, `--restart`, `--data-dir` and the `gpuwm run` supervision flags are refused by name. The run draws ensemble mean, spread, minimum and maximum, threshold probabilities, paintball and postage-stamp maps through the Rust renderer while it runs, and writes member wrfouts only with `--keep-member-files`. Members whose physics is Thompson, YSU, MM5 surface layer, Noah and legacy RRTMG on a fixed clock with `nwp_diagnostics = 0`, and that share one lateral-forcing field set, advance as one native batch per card; every product file is byte-identical to the same members run one at a time. On an RTX 5090, four time-lagged members of a 150x120 3 km one-hour case forecast in 22.0 s packed against 39.5 s one at a time, plus a 3.4 s bootstrap per member. Other members run one after another, and `ensemble-run.json` records why. Ctrl-C stops a running ensemble at the next model step, cancels members not yet started and exits 130 with one sentence. A failed member is named in the error, `ensemble-run.json` lists `failed_members`, and no aggregate product is published for a failed or interrupted run. Each finished member releases its card memory before the next starts. `gpuwm sim`, the prepared runners started on their own and `[grid]/[dynamics]/[run]` configs refuse an `[ensemble]` table by name, as 2.8.4 did.
- Ensemble member source recipes: operational members (`gefs` for `gfs`, `aigefs` for `aigfs`, `ecmwf-ens` for `ecmwf-open-data`), time-lagged members (earlier cycles of one model covering the same window) and multi-model members (distinct source and cycle trajectories). Each member keeps its own source trajectory at every boundary time, and members are never repeated to reach a count: one model run listed under several of its source ids (`hrrr`, `hrrr-prs`, `hrrr-native`) is refused as one trajectory. `python -m gpuwm.ensemble.recipes --source SOURCE --cycle CYCLE --hours H --members N --recipe time-lagged` prints a member plan without downloading, and `--member` names the base member. `gpuwm ensemble CONFIG --recipe time-lagged --members N` runs them, as do `gpuwm go` and `gpuwm run` with `--recipe`, `[ensemble] recipe = "time-lagged"` in the config, and a `gpuwm run-plan` plan that names the recipe: each member's cycle is fetched and prepared by its source's ordinary chain, and the members run through the ensemble session with aggregate products. A time-lagged roster keeps the config's own `[fetch]` member at every cycle. Multi-model takes its member list from `--trajectories FILE` (JSON or TOML entries of `source`, `cycle` and, where the source has members, `member`) or from `[ensemble] trajectories`, and the planning command takes the same `--trajectories FILE`. `--dry-run` prints the member plan without fetching. The run folder's `ensemble-recipe.json` records each member's source, cycle and prepared bundle. The recipe doors run one-domain configs with a `[fetch]` table.
- Recipe ensembles refuse before the first fetch. Every member's config and chain plan is checked (a member from another model gets that model's own fetch table and the domain's crop box), then the card, memory, geography, renderer, product and disk checks run for all members. On `gpuwm go` and `gpuwm ensemble`, `--cycle` re-times the roster, `--readiness` answers for every member's window and runs nothing, `--transport`, `--whole-cycle` and `--late-after-minutes` reach every member's fetch, and the flags that name one trajectory (`--prepared-root`, `--restart`, `--data-dir`, `--supplement`, `--section`, `--keep-checkpoints`) are refused by name. Under `gpuwm run` a recipe ensemble runs unsupervised and cannot be resumed: `--outdir` is the case folder holding one stamped run folder per launch, the supervision flags are refused by name, and `gpuwm resume` and `gpuwm branch` refuse a recipe config. A failed member stage exits with its own code (75 for a source lead not posted yet, 130 for a stop) and `ensemble-recipe.json` records the failure.
- `gpuwm fetch --source ecmwf-ens` fetches ECMWF open-data ENS perturbed members p01 to p50. `gpuwm fetch --source rrfs-ens` fetches the five public RRFS ensemble members (m001 to m005, 3 km CONUS); preparation refuses them because their public files stop at 250 hPa and carry no soil or land state.
- Random ensemble perturbations are refused before any download, process or GPU work. SPPT, SKEBS and SPP (`sppt`, `skebs`, `spp`, `spp_conv`, `spp_pbl`, `spp_lsm`), `rand_perturb`, the `pert_*` switches and table-form `[ensemble] perturbation` descriptors exit 2 at every door that loads a config (`gpuwm ensemble`, `go`, `run`, `run-plan`, `sim`, `branch`, `downscale` and `python -m tools.ensemble_forecast`) and at `gpuwm import-namelist`, which names the key, with the reason: their spread amplitudes have not been calibrated against observations, so ensemble spread and probabilities would be meaningless. An absent perturbation and `perturbation = "none"` behave as in 2.8.4. The 2.8.4 overlay file and the provider names `gpuwm.da.perturb` and `experimental-stub` run through `python -m tools.ensemble_forecast` as in 2.8.4; the gpuwm doors refuse them (exit 2, as in 2.8.4) and name that command.
- Simulated radar draws ZDR, CC and KDP PPIs when the volume carries them, where the same gate's reflectivity is drawn. Radar colours follow the AWIPS reflectivity table (top 85 dBZ) and a green-red velocity table on simulated radar PPIs and on the composite, 1 km, ensemble and observed reflectivity maps. `[simulated_radar] color_tables = "classic"`, `gpuwm render --radar-colors classic` or `RUSTWX_RADAR_COLORS=classic` brings back the earlier colours byte for byte.
- New pages: `docs/ensemble-output-contract.md` (ensemble run paths, manifest and products), `docs/ENSEMBLE_PROVIDERS.md` (member sources and recipes) and `docs/ensemble-wrf-stochastic-import.md` (how WRF `&stoch` controls map, and why active ones are refused).

Fixed:

- RUC soil water no longer refills a dry top soil level from below at the WRF v4.6.1 rate. v4.6.1's SOILPROP normalises diffusivity and conductivity by total moisture over porosity, 2.5 to 8 times the v4.0 to v4.5 value in dry soil, so evaporation ran high and the near-surface air turned moist. New key `ruc_soilprop`, default `"wrf_45"`: the v4.0 to v4.5 form, also the operational RAP and HRRR form, with diffusivity and conductivity over the moisture above the residual. `"wrf_461"` keeps the v4.6.1 form by name and reproduces the earlier output byte for byte. In one cycle (2026-10-02 21Z, full HRRR configuration), 2 m dewpoint bias against 2,287 / 2,255 stations went from +1.16 / +1.59 K to +0.92 / +1.22 K at f01 / f02, and 2 m temperature bias went from -0.51 / -0.56 to -0.40 / -0.42 K. A RUC checkpoint written by an earlier build is refused on resume by its land-surface algorithm identity. Answers change for every RUC run.
- Multi-card `[devices]` forecasts and streamed `[tiles]` forecasts pass sub-grid orographic statistics and topographic wind inputs to each tile, including its halo, so `gwd_opt` and `topo_wind` initialize on split domains. Topographic wind curvature uses the true neighboring terrain across stored row slabs and prepared slab seams. The terrain-drag kernels and their WRF arithmetic are unchanged.
- MYNN initializes turbulent moments using water vapor, matching WRF's driver. It previously included cloud condensate in that input, changing cold-start cloudy columns. Answers change for those columns.
- Native hybrid analysis can initialize at its own model-top interface. Tables distinguish that interface from the highest mass level, and the shared CPU and CUDA vertical operators co-locate pressure endpoints within the existing float32 rounding bound. Requests above that bound still refuse.
- The aerosol-aware Thompson cold start names the field and the cell count when an analyzed aerosol number, or the droplet number seeded from it, is not a number. It previously failed with a NumPy index error.
- Preparation honors explicit worker counts within CPU and memory limits, records effective pools, and runs host interpolation, cold-start numbers, setup copies and prepared-file writes in reusable Rust workers.
- Mapped CUDA preparation uses bounded batches with host-retained results, so grids larger than a card's preparation workspace can prepare on that card.
- Split forecasts start from the sealed initial state and first boundary interval, then wait only for later intervals as they post. Preparation reserves its next batch while the forecast runs.
- RUC bypasses lake columns only when the lake model is selected. With `sf_lake_physics = 0`, it runs its water branch instead of leaving those columns unadvanced. Answers change for RUC runs with lake cells (LAKEMASK = 1) when `sf_lake_physics = 0`, the default. RUC checkpoints written by 2.8.4 or earlier are refused on resume, by name.
- Prepared forecasts that split one domain across GPUs size each rank's MYNN column batches against its card's memory budget, reducing host launch gaps without changing column arithmetic.
- Single-domain multi-GPU admission prices the selected workspaces and rolling boundary buffers, counts existing allocations once, and limits output snapshots to the remaining card memory.
- Legacy RRTMG longwave applies each stratospheric optical-depth correction once, on its owning g-point thread. Parallel band 4 and band 7 calculations previously raced on shared outputs and could change upper-level heating and outgoing longwave radiation. Answers change. Legacy RRTMG checkpoints written by 2.8.4 or earlier are refused on resume, by name.
- `[devices]` admission prices the MYNN tile workspace and selected microphysics kernel tier, counts returned MYNN rates once, and includes the RUC soil and snow workspaces and physics coefficient tables. Aerosol microphysics drops unused diagnostic arrays. Ranks whose complete carrier inventory exceeds the history snapshot budget copy frames directly to the host and fence the next producer until that copy completes.
- History writes and store downloads select the card that owns each CUDA stream, event and buffer. Submission preserves the producing stream. Resident and store-direct runs can use card sets that do not start at card 0 without failing at the first history write.
- Forecasts run one after another in one process no longer leave their microphysics ring tables and Noah land-surface tables on the card.

## 2.8.4 (2026-10-03)

New:

- `[simulated_radar]` writes native radar volumes and PPI loops while a forecast runs under `gpuwm go`, `gpuwm run` or `gpuwm sim`, the routes WOOF drives. Sites, scan ladders, timing, fields and formats are configurable. Level II and CfRadial 1 are the defaults; CfRadial 2 and ODIM are optional. Before fetching, a forecast checks `rw_simradar` and the scan's host memory and prices listed sites' radar output into its disk check; ensemble and local-cycling members refuse the table by name. `gpuwm simulated-radar` replays saved full-column histories. The manifest carries immutable artifact paths, hashes, scan times and source revisions. Off by default.
- `gpuwm simulated-radar --describe` reports installed native support and companion requirements. `--estimate` reports scan samples, memory and an output-size bound for quoting before processing history.
- `zadvect_implicit_variant = "wrf_legacy"` selects the older WRF implicit vertical-advection split and scalar mass weighting. The default remains `"wrf_471"`; selecting the older variant changes forecast answers.

Fixed:

- Legacy RRTMG shortwave follows WRF on columns with no layer above the troposphere switch (a model top below about 190 hPa, as on shallow or idealized domains): the solar source of bands 16, 17, 27, 28 and 29 is zero there, as WRF leaves it. The batched path read leftover GPU memory for bands 16 and 27, giving NaN or impossible surface fluxes, and every path computed bands 17, 28 and 29. Answers change only on such columns.
- Scientific documentation distinguishes code verification, solution verification and validation against observations. WRF comparisons name their reference version, tested build and limits; historical runs and composition exemptions no longer imply current forecast accuracy. Physics profiles use comparison or daytime names. Old profile IDs and labels are accepted as aliases wherever an ID is entered (flags, run and stream plans, case catalogs, the physics check, New forecast), and a preparation sealed under an old ID still matches and extends. Suites with no current matched WRF run, the defaults included, now report `supported-not-wrf-verified`, and a prepared forecast prints one line saying so; the run continues. WRF evidence moves to `docs/public/wrf-comparison/`, with stubs at the old paths. Observation-scoring badges say `SCORED`; execution qualification and observation validation remain separate.

## 2.8.3 (2026-10-02)

New:

- `[devices]` in a single-domain or tree config, or `--devices N` on `gpuwm go`, splits one forecast across cards, each card's memory checked before the run and the split recorded in an execution receipt. In a tree, `domains` names the grids that split (default all), a split nest takes its parent's forcing each parent step, and two-way feedback reaches a split parent; `gpuwm sim --devices N` or `--devices-table` sets the split for a prepared forecast without changing its prepared configuration. `gpuwm check CONFIG --devices N` prices every card, grid and the pinned host memory, exiting 2 for a card with no declared or measured VRAM budget. Frames reach the host while the model steps on. A split parent over a resident nest, moving nests and late-starting split grids are refused by name, and split grids refuse Noah mosaic and slope radiation (`slope_rad = 1`) before allocation, naming the missing state; resident grids keep both. Off by default.
- `gpuwm run-plan` now starts GFS and native `hrrr` single domains and trees from the first posted hours, as `gpuwm go` does, and both start one `hrrr-prs`, `rap`, `rrfs`, `icon-eu` or `gem-gdps` domain the same way; a `[tiles]` root, as `gpuwm cyclone-setup` writes, waits at each seam, not for the whole cycle. Output unchanged. A forecast failing before a later lead times out keeps its own error, the timeout reported beside it.
- `GPUWM_FETCH_POLICY=aws` pins each source with an AWS host in the route table to that host (no NOMADS fall-through; a source with no AWS host keeps its usual host); "latest" is the newest cycle whose first hours are on AWS (`--whole-cycle` keeps the whole-cycle rule), and forecasts still stream from their first posted hours. Off by default: unset, fetching is unchanged. `gpuwm fetch` asks a host refusing HEAD with a one-byte GET.
- `gpuwm import-namelist` accepts the WPS sections GUI writers add (`domain_wizard`, `mod_levs`, `plotfmt`) and a bare `30s` geography, and fills 17 of 33 formerly required keys from WRF's Registry and WPS defaults; an unknown section is still refused by name.
- `diff_opt = 1`, WRF's coordinate-surface horizontal diffusion, runs with `km_opt` 2 and 4 and either `mix_full_fields`, and restarts and streams. Off by default (`diff_opt = 2`).
- `topo_wind` (under YSU, not with BEP or BEP+BEM) and `gwd_opt` = 1 or 3 (under any PBL scheme) run WRF v4.7.1's sub-grid terrain drag on the GPU, the coefficients and the KIM and GSL drag matching WRF's Fortran bit for bit on test columns; the static builder adds the orographic statistics from WPS geography, and namelists setting either key import. The orographic statistics come out bit-identical on Linux and Windows. A streamed `[tiles]` domain stops at its first drag call. Off by default.
- `gpuwm ml-export` turns a run's history (regional, hex or global, or the run page's ZIP) into one Zarr store per domain on WeatherBench 2's 13 pressure levels (ERA5's 37 or the model's own as options), with ERA5 and WeatherBench 2 names and units, earth-relative winds, a `below_ground` mask over ERA5's below-ground fill, levels above the model top listed and left out, an optional regular latitude-longitude grid, and a ZIP that opens in place; `xarray.open_zarr` reads it under zarr-python 2.18 and 3. On ERA5's 0.25 degree points, an ERA5-started run's first frame is within 0.1 K and 0.5 m/s of ERA5's temperature and winds in the mean at every level from 100 to 1000 hPa, and the levels match GeoCAT to float32 round-off above and below ground. Variables, level sets, spacings and naming schemes are tables. On Windows, a ZIP's history files named with WRF's colons unpack under the underscore spelling. See `docs/ml-export.md`.
- The GPU preparation backend rotates winds, interpolates vertically and builds the Thompson cold start on the card in the CPU route's arithmetic, so a card-prepared start matches the CPU route's arrays; forecasts prepared on the card change. On a full HRRR grid, card forcing preparation took 154 s instead of 640 s.
- The acoustic, advection, horizontal diffusion and per-step glue kernels move less memory each step, with byte-identical output; the horizontal diffusion change alone takes an hour of an 800x600x50 domain from 95.7 to 88.4 s on an RTX 5090.

Fixed:

- Legacy RRTMG longwave radiation keeps the correct spacing above the model top. A pressure guard had compressed valid layers and changed their temperatures and heating. Answers change.
- `gpuwm go --cycle` accepts an exact match to a prepared bundle, checkpoint or declared forcing's actual initial time and records it as a no-op; conflicting or unreadable input times remain refused. Declared forcing fetched by the run is checked after acquisition and before preparation.
- Moist forecasts apply WRF's moisture correction to the pressure terms (`moist_cq`) in every shipped profile and every configuration leaving it unset, passive vapour with microphysics off included; dry states bypass it. Unmatched configurations (most imported WRF namelists) had run without it and WSM6 and Kessler profiles had disabled it; an explicit setting still wins. Over five 20-member WRF ensemble cases, forecasts sit inside WRF's own rounding envelope in 76% of field-hours (17% before the fix), against 65% for WRF built with fast math (Intel ifx -fp-model fast). Answers change.
- The dynamical core fixes six places where it differed from compiled WRF v4.7.1: the vertical-velocity damping constant, `w_damp`'s float32 rounding, the geopotential tendency's outer open and specified rows, vertical-velocity advection at an open model top, map factors in the mapped open-boundary kernels, and diffusion (sixth-order filtering, boundary deformation, TKE at the outer rows, prescribed surface heat flux). Answers change; new tests compare each routine's output words with compiled WRF.
- Surface vertical velocity on open or specified edge rows takes WRF's clamped terrain slope instead of the interior slope copied outward, which doubled its terrain-following term there; periodic domains and interior cells are unchanged; answers change.
- Land points cap 2 m humidity at 1.05 times the lowest level's vapour after the surface schemes, as WRF's surface driver does; answers change for 2 m humidity.
- `wrfinput` starts initialize Noah soil liquid water from soil moisture in unfrozen soil and terrain-following vertical wind from the surface flow, as WRF does, not the file's zeros (first-hour latent heat flux on a 3 km case: 239 to 0.4 W/m2 RMSE against WRF), and with `usemonalb` off keep the seasonal background albedo WRF's land-use initialization sets, not the file's monthly value. Answers change for these forecasts.
- `gpuwm downscale` children of saved runs, `--point` included, build their own terrain, land use and soil at their own spacing (Copernicus GLO-30 terrain at 1 km or finer) instead of running on the parent's interpolated terrain, and the parent's state, every boundary frame included, is blended and rebalanced onto them as WRF's `ndown` does, within a few float32 epsilons of compiled WRF v4.7.1 on column checks. The child needs a WPS_GEOG tree (`gpuwm fetch-geog`, or `--geog-root`) and is refused without one; `--parent-terrain` keeps the old route; the child config records it in `[static]`. Answers change for these children.
- Downscaled children and DA nested forecasts take the steep-terrain acoustic substeps and `epssm` floor from their own terrain, keeping automatic parent choices; lower explicit `epssm` values are refused. Answers change for a child whose own terrain is steep enough for the rule where its parent's was not.
- Adaptive steps land exactly on history, radiation, cumulus, surface and boundary-layer times (a history interval need only be whole seconds), prepared forecasts honor adaptive history alarms, and delayed child domains start on time; answers change for adaptive forecasts whose history or physics times fell between steps.
- Disk estimates price the history each domain's physics and `history_vars` write in its output window: trimmed runs that fit start, and runs that cannot fit are refused up front.
- Prepared forecasts and checks, a forecast that starts while its source posts included, price BEP+BEM's workspace from urban column counts, so a forecast that fits stays on the card; configuration checks keep and explain their upper bound.
- A resident nest under a streamed or split parent no longer stops before its first restore with a KeyError from the shared scratch arena.
- A streamed forecast's final digest records the resident forecast's model time; a fractional adaptive step had ended it 0.27 ms late after 3 h, so two identical forecasts had different digests.
- Every checkpoint writer (resident, streamed, out-of-core) records the run configuration alike, so two checkpoints of one state no longer differ in their header.
- CI runs Stage 1 without CuPy, pins oracle builds and gates publication on the same commit; SASE accounting needs no CuPy, Windows receipt families match paths, and releases link retained GPU results.

## 2.8.1 (2026-10-01)

New:

- `sf_urban_physics = 1` runs WRF's single-layer urban canopy model on city cells under Noah or Noah-MP; the high-resolution land cover keeps its urban classes (Local Climate Zones or NLCD). A grid whose first level sits inside a class's canopy, WRF's own stop, is refused when the config loads. Off by default.
- `sf_urban_physics = 2` runs WRF's multi-layer BEP model with YSU or MYJ: building drag, heat and turbulence enter the lowest levels. Under YSU the surface drag outside the city is applied once, where WRF applies it twice. Off by default.
- `sf_urban_physics = 3` adds WRF's building energy model to BEP: indoor temperature, air conditioning, windows, and direct and diffuse sunlight on streets and walls. Off by default.
- A domain's terrain can stay unsmoothed or take any WPS GEOGRID.TBL smoother: `[[domain]] static`, `gpuwm domain --terrain-smoothing`, or GEOGRID.TBL import.
- `smooth_precision = "wps-float32"` or `--terrain-smoothing-precision` makes default terrain smoothing match `geogrid.exe` exactly; the float64 default is unchanged.
- WRF namelists with `history_begin`/`history_end`, `smooth_cg_topo`, `topo_shading` or `slope_rad` now import and run, matching WRF v4.7.1 (all off by default); an import names every unsupported key at once; a restart may change the history window wherever it may change the cadence.
- Noah mosaic land use: `sf_surface_mosaic = 1` runs `mosaic_cat` land-use tiles per cell, matching WRF v4.7.1 apart from three WRF defects it does not copy (soil layer index, full-weight accumulations, empty land cells). With `sf_urban_physics = 1` the urban canopy follows WRF and runs only where a cell is mostly urban; `mosaic_urban_canopy = "every_tile"` also runs it in the town tiles of mostly rural cells. Off by default.
- The UW moist-turbulence boundary layer (`bl_pbl_physics = 9`) runs on the card and imports from WRF namelists, matching WRF v4.7.1 on column checks.
- Physics and the dycore run faster, output byte-identical: per call, legacy RRTMG 7x, RUC 7 to 22x, Noah-MP 5.8x, MYNN 4.5x (a `[tiles]` buffer keeps its 2.8.0 width and memory), NSSL 1.9 to 3.1x, P3 1.5 to 2.2x, RTE-RRTMGP 1.6x, Thompson 1.3 to 1.6x, Morrison 1.2 to 1.5x, and Milbrandt-Yau, KF, New Tiedtke, Shin-Hong and GF 1.1 to 1.5x; nest forcing and the dycore's bookkeeping take fewer launches.
- `[shared] adaptive_nest_lattice = true` picks the adaptive root step with the fewest nest cell-steps: 6% faster on a three-domain 250 m tree; answers change.
- `[shared] zadvect_implicit = 1`, or the WRF namelist key, runs WRF's implicit-explicit vertical advection, matching WRF v4.7.1's routines word for word on column checks but for two corrected boundary terms; default off; answers change. Its `w_crit_cfl` (where w-damping starts; WRF suggests 2.0) imports and runs as WRF v4.7.1 uses it; default 1.0.
- `gpuwm fetch` takes each lead once a host has it (NOMADS's 403 beside another host's 404 means not yet posted); a lead past its late time exits 75. Leads move in order, dated when a host first holds them or marked already up when first asked; `aigfs` waits for its GDAS donor; event streams carry every source's leads.
- `latest` needs only a cycle's first leads; `--whole-cycle` keeps the old rules, `--readiness` says whether a run can start and `gpuwm go --cycle` names the cycle.
- `gpuwm go` on a GFS single domain prepares and forecasts from a cycle's first posted hours, taking each later hour as it posts; output unchanged.
- A single native `hrrr` domain, and domain trees from native `hrrr`, `hrrr-prs`, other mapped sources and GFS, prepare the start and nests first, then one boundary interval at a time; the forecast starts at that head and waits for any interval not yet ready, output identical. Storm-following trees, `gpuwm cyclone-setup` ones included, start at the head too; a `[tiles]` tree waits only when its root streams from host memory. A GFS tree, storm-following included, prepares as its leads post.
- `gpuwm sources` shows each source's posting shape and lateness budget; HRRR DA plans on expected times; `latest` asks about AIFS f360 from +5 h 26 min, not +7 h 30 min.
- `gpuwm run --preprocess-backend` or `[case_data] preprocess_backend` pins where a config run's root prepares, so compared runs on a busy card share one preparation.
- The source decode uses every core: HRRR and similar sources decode several forecast times at once, GFS every hour at once, and `--preprocess-workers` sets the count. Prepared arrays are unchanged; a 48 h HRRR decode on 24 cores took 140 s instead of 480.
- Each source's posting (shape, lateness budget covering measured late posting) shows in `gpuwm sources`; HRRR DA plans on expected times; `latest` asks about AIFS f360 from +5 h 26 min, not +7 h 30 min.

Fixed:

- New forecast and assistant fits price MYNN workspace from the selected card's memory instead of the host card or an off-card cap.
- BEP+BEM reserves its full local-memory frame on both supported CUDA compiler families; the default compiler needs 5,128 bytes per thread on Blackwell.
- Adaptive nested runs shorten the parent step when an exact child divide would exceed the substep bound. Short output and stop-time remainders stay within their deadlines, and hosted clock failures retain their cause in the error event.
- `gpuwm downscale --point` from an adaptive-clock parent gives the child its own fixed step (5 s per km, landing on its output intervals) instead of the parent checkpoint's live step over the ratio, which could be refused.
- Hex and global maps name their model where they said WRF; `rw_mpas_convert --model-label` sets it.
- The desktop's storm-following plans on GFS name the prepared route, so `gpuwm run-plan` runs them instead of refusing the configuration.
- A chained single-domain forecast keeps its sealed preparation's long step, restarting on the seal when a later boundary moves it.
- Native `hrrr` roots take their terrain smoothing, and an urban run with a high-resolution carrier prepares under `gpuwm go` and run-plan and restores from its seal.
- Two runs of one mapped-source download into sibling run folders now share one prepared cache; its identity no longer names the run folder.
- A GRIB2 mapped source's decode fits its threads, chosen before decoding, and its parallel valid times to free memory or a systemd or batch limit.
- Blackwell cards (RTX 50-series, RTX PRO) divide by kernel constants IEEE-correctly, not through reciprocal multiplies up to one ULP off: answers change, and steps take about 2% longer (1.7% on an RTX PRO 6000).
- The Blackwell division check reads CUDA 12 and CUDA 13 compiler output correctly while still rejecting inexact reciprocal rewrites.
- A nest spawned mid-run is priced against the card's free memory before it is built, and a card that cannot hold it stops with the sizes named instead of a CUDA out-of-memory. Moving a nest from Python with no `staging` argument stages through host memory, as `gpuwm run` does.
- A source that publishes cloud water, rain, ice, snow and graupel at every forcing time (`hrrr`, `hrrr-prs`, `icon-d2`, your own mapping) brings them in through the outer grid's edges, not water vapour alone. A three-hour HRRR mountain winter case held 12.8 million t of snow after two hours, now 26.9 (HRRR's analysis: 34.6). GFS byte-identical; an older prepared forecast runs on its old edges and says so.
- Every memory check prices those hydrometeor edge tables (`[tiles]`, the resident check, `gpuwm stream`, `gpuwm multi-run`, the preparation check, a streamed forecast's host store): 80 MB of card and 160 MB of host per forcing interval on HRRR's 3 km grid.
- A snow-covered land column whose top soil sits more than 30 K below its skin is rebuilt linear in depth from skin to deep soil, keeping its moisture. On 2017-01-19 15Z over Idaho the first 3 h 2 m temperature error at 13 snowy ASOS stations fell from 3.76 K to 1.65 K. Rebuilt columns are printed; a current HRRR analysis prepares byte for byte as before.
- A nest on its parent's levels keeps its float64 base state, so the float32 pressure correction reaches it, spawned and moving nests included: surface geopotential error fell from 2.2e-4 m to 2.3e-13 m on a 12/3 km HRRR tree.
- The renderer's three time-axis fixtures (`tools/rustwx/crates/rw-wrfbatch/tests/time_axis_fixtures/*.nc`) no longer carry a machine path in `history`; the release's path scan now reads NetCDF attributes.
- A forecast too big for the card is refused before anything is allocated, naming need and free memory: single-domain `gpuwm run`, `gpuwm downscale`, the prepared and tree runners, and the cache, wrfinput and idealized loaders price it on the envelope `gpuwm go` checks. `--no-memory-gate` skips it; a pinned `[tiles]` refusal names the largest tile that fits.
- The forecast memory gate counts pool headroom once: no legacy-RRTMG slack, a measured 1.13x margin (HRRR physics with RTE-RRTMGP: 1.19x), urban arrays at their size; chained preparation checks the envelope.
- `gpuwm domain` refuses a RUC suite on a source whose soil RUC cannot start from (GEM GDPS: one layer), before the download. A nested `--source hrrr` domain runs a RUC suite: its hierarchy stage now pins RUC's nine soil layers.
- A domain finer than 1 km with no physics named runs Thompson, MYNN and RUC on both radiation streams (the suite that kept coastal fog), not the source's YSU and Noah, unless the source's soil cannot start RUC; below 500 m the step stays fixed. New forecast, `gpuwm physics-catalog --check` (new `finest_dx_km` and `domains` keys), `gpuwm domain --physics-choices` and the manifest agree. Name its suite to keep a 2.8.0 sub-km file's YSU and Noah.
- A nest from `gpuwm domain` or a DA cycle's nested forecast no longer gets stronger sixth-order damping than its parent: each takes the smaller of its parent's value and the 0.12, 0.10, 0.08, 0.06 ladder.
- A stock-WRF export with no physics named (`python -m gpuwm.wrf_direct` without `--physics-profile`, the ERA5 export, domain-tree exports) writes its prepared cache's schemes into each wrfinput and wrfbdy, not the bundled file's WSM6 and Dudhia.
- A forecast with nests that names its suite (New forecast, a Starting point, the assistant, a run plan) runs from every source, suite on the outer grid and nests as `gpuwm domain` wrote them; it was refused after preparation everywhere but GFS and ERA5 ("cu_physics selected 0 expected 1").
- A `gpuwm run` domain tree with no `[tiles]` is priced against the card's free memory before anything is downloaded, as a single domain is; `--no-memory-gate` skips it.
- A `gpuwm run` tree whose delayed nest streams through `[tiles]` runs past the nest's start ("the state object handed to the streamed stepper is not the one it was attached to") without two copies of it in memory.
- A pinned `[tiles]` tiling decides without loading the tile planner, as its documentation says; its tile count and redundancy are unchanged.
- On `gpuwm run`, a nest starting after the forecast is priced before its rebuild and a card that cannot hold it stops with the sizes named; `--no-memory-gate` skips it. With legacy RRTMG a waiting nest no longer keeps its parent's replaced radiation.
- A `gpuwm run` preparation on the card prices the allocator reserve at 1.25x its arrays, not 1.20 (a 12 km ERA5 run on an RTX 5070 Ti reserved 3.91 GB against 3.85); `gpuwm check` agrees. Measured peaks: `docs/dev/a65-preparation-peaks.md`.
- Under the adaptive clock, steep high ground under strong crest winds caps `max_time_step` only where three-hour adaptive runs saw a longer step stop, on a 2 and 3 km map measured to 40 s. A 2.25 km ridge of slope 0.30 or 0.37 under a 3.6 to 3.8 km crest and 20 to 30 m/s keeps 30 s; the `terrain_clock` record names each domain's reading.
- A domain whose `epssm` is a default (unset, or a namelist listing fewer values than domains) takes the measured floor over steep ground; a lower chosen value is refused.
- `gpuwm check`, `gpuwm run-plan --estimate` and `gpuwm domain` count the edge tables in every figure: 21.6 MB more on a 550 x 550 x 49 HRRR Morrison config.
- A native `hrrr` configuration with no route namelists beside it runs: the run writes them into its folder. A value they cannot carry is refused at `gpuwm go --dry-run`, naming the fix.
- `gpuwm warm-kernels --all-profiles` warms all 28 shipped suites and exits 0; suites with no longwave warm on their declared constant longwave. A failing suite is named with its remedy and the pass goes on, exiting 1.
- `gpuwm cycle --parent-kind mpas-cuda` prices the MPAS port's device memory for the mesh against the card's free memory before the first leg and refuses a mesh that cannot fit with both numbers; it runs against port trees in the `src/hexcore` layout, and refuses a tree with neither package by name.
- `gpuwm render` imports each frame once: nine long windows over a 37-frame run took 25.7 CPU-s instead of 98.9; every picture byte-identical.
- A nest starting after the forecast draws, live and through `gpuwm render`, with leads from the forecast's start (every frame was refused, "conflicting WRF references"); `gpuwm cells` no longer dates it early.
- The 10 m wind maximum pictures draw WRF's stored `WSPD10MAX` where a history has it (`nwp_diagnostics = 1`); otherwise they are titled "largest hourly snapshot, no stored max".
- `simulated_ir_satellite` from a wrfout shows cloud tops: WRF-Python's `wrfcttcalc` had per-gram absorption applied to kilograms, so most cloud drew the ground's temperature. The median over cloud tops on a 3 km Thompson run moved from 306 K to 243 K, matching WRF-Python to 2e-5 K; snow no longer counts as cloud ice, and a file without cloud ice draws.
- A domain tree with a `[perturbation]` block prepares from `hrrr-prs`, native `hrrr`, the other mapped sources and ERA5, as from GFS: the bubbles are recorded as deferred (`gpuwm-initial-perturbation-deferred-v1`) and applied at the start. A single-domain preparation still refuses the block, earlier on native HRRR. An ERA5 tree's companion WRF files are optional; `gpuwm research create` makes warm-bubble scenarios from any source.
- 34 tests red on Linux for reasons outside their code now pass and check what they did; no shipped code changed.
- `gpuwm local-da` finer than 1 km (rung 10 and up) with no physics named runs the sub-km default (Thompson, MYNN, RUC); a published local DA case on `--source hrrr` launches.
- A prepared DA cycle's memory check counts each leg's analysis, on the card or staged through host, and each member perturbation's cuFFT work areas, now released after each draw.
- Results no longer depend on NumPy 2.5's AVX-512 math: a CPU-prepared state's pressure takes glibc's `powf`, its `hypsometric_opt = 2` geopotential, density and pressure glibc 2.39's `log1pf`, the Python Lambert, Mercator and polar stereographic grids the C library's tan, atan, atan2, asin, acos, log, exp and pow, and a preparation's base-state, moisture and temperature math one Rust library.
- A Thompson start's seeded rain number and the `mp_physics = 28` droplet number now take glibc's `powf` and the C library's `pow`, not NumPy's, which moved them on AVX-512 Linux.
- RUC's 2 m temperature (T2, TH2) and cold-start frozen soil water take glibc's `powf` and `logf` on every machine, not NumPy's, whose AVX-512 math differs on cloud Linux machines.
- The default hybrid coordinate (`hybrid_opt = 2`) takes WRF's closed-form cubic in plain float64, so its coefficients are the same on every machine.
- The standalone `rw-wps` package installs cleanly (its check refused every clean install, and on Python 3.11 its console scripts); it and an installed gpuwm now read their own version beside another gpuwm install, not the other's. A finished preparation there says it prepares inputs only, and `gpuwm.obs.write_radar_grid` writes there.
- `--cycle latest` waits each publisher's measured delay per cycle hour and no longer skips posted cycles of seven sources; ARCO ERA5 follows its store's end; ECMWF's 503 SlowDown is retried up to 10 minutes; IFS past f144 and GEFS past f240 fetch 6-hourly.
- An AIFS forecast starting past f000 prepares (the fetch adds f000's land mask and terrain); a GEM GDPS window past f000 fetches its statics from PT000H.
- A storm-following nest's 1 h rain draws at hours it moved (4 of 12 drew); ground it moves onto, and a spawned nest, start from the parent's rain total.
- A chained forecast waiting for a boundary interval says so on its heartbeat, progress file, events, run page and My forecasts, with model time; `gpuwm go`'s watchdog times the wait by its own bound instead of stopping it after 120 s. Under `gpuwm run-plan`, a failed forecast's heartbeat says it waits for the preparation to end, and `prepare_sealed` lands with the seal. A checkpoint written before the preparation finishes no longer stops the run ("discontinuous preserved forcing frame"), and resuming from it matches the uninterrupted run. Forcing checks record each interval's true end frame, so renewing or extending forcing across a hydrometeor that clears out is accepted.
- Two CUDA preparations of the same inputs publish identical caches: free memory, CPU and worker counts and nest preparation time stay in the receipt, out of the cache identity.
- Preparations from 2.8.0 on are no longer refused as different physics when only registry wording, other schemes, a scheme's partners or off-by-default additions changed; refusals name the change.
- A global source's pressure levels are staged only where domains clear of its longitude seam read; GEM GDPS decodes only there: a 6 h preparation took 32 s and 1.5 GB, not 175 s and 12 GB.
- `gpuwm run --wrfinput` no longer refuses a `USE_THETA_M=1` pair whose real.exe rounds moist theta differently; mismatched pairs are still refused.
- Every door reading a namelist, `gpuwm import-namelist` included, now reads several keys per line and element or section assignments, which were refused.
- A WRF namelist with WUDAPT Local Climate Zones imports as one: `geog_data_res = 'cglc_modis_lcz'` builds that land cover through `[static.highres]`, where it was dropped and the run built MODIS land use under `use_wudapt_lcz = 1`; `num_land_cat = 61` imports with the urban canopy on. The report says the block also replaces terrain and soil; a `geog_data_res` token the engine builds nowhere is refused by name. `gpuwm prep` and `gpuwm go` build that land cover without terrain smoothing, where they refused the token.
- WDM6 rain carrying no number no longer evaporates to feed ice deposition that took vapour below zero, stopping two convective forecasts at step 4; answers change.
- `gpuwm go` no longer warns on every prepared run that a preparation phase has no stage, and the fetch route table's posting measurements no longer name the machine that watched.
- `hrrr-prs`, `rap`, `rrfs` and `icon-d2` take `--cadence 3` or `6` (or `cadence` in `[fetch]`), fetching and preparing only those leads; it was refused though each hour is published. A preparation made before this change still runs as its source, not refused or relabelled `mapped`.
- Route table sources take any `--cadence` their publisher posts and preparation takes (`hrrr-prs` 2 h, `aigefs` 12 h, hourly `icon-global` to f078); a refusal names the lead or decode.
- `gpuwm go` under `GPUWM_NO_LOCAL_GPU` refused with "GPU readiness is info: device not touched"; it now names the variable and says to unset it.
- `zadvect_implicit = 1` no longer goes NaN in three steps over a steep ridge: two boundary terms of its w solve take consistent units, departing from WRF v4.7.1; answers change.
- Near the memory floor, `gpuwm cyclone-setup --nest-budget-gib` refused the preset nest the command kept without the flag; both now judge it alike, and `nest.headroom_bytes` names only headroom held back.

## 2.8.0 (2026-09-26)

New:

- A domain at 1 km or finer now builds its terrain from Copernicus GLO-30 (about 30 m) by default instead of 30 arc-second GMTED, when the configuration declares no `[static.highres]` block. Coarser domains keep the 30 arc-second terrain, and land use and soil are unchanged. Before the download the console names the tiles, how many are already cached, about how many MB the rest will take and the cache folder (your user cache, `gpuwm/highres-cache`). A declared block is taken as written: `enabled = false` keeps the 30 arc-second terrain, and the new `max_dx_m` key limits a block to domains at or finer than that spacing. A prepared cache sealed before this change for a tree with a domain at 1 km or finer and no `[static]` block is refused and must be prepared again, once.
- `gpuwm warm-kernels` compiles the forecast GPU kernels for this card before the first forecast: 128 kernels in 49.0 s on an empty cache on an RTX 5070 Ti, 0 in 1.1 s once warm.
- `gpuwm run-plan` shows the first run's kernel compile as `kernel_compile_progress` warning events, one per GPU module.
- `gpuwm downscale-parent RUN` lists every domain the run wrote with its spacing and frame count and names the finest as `default_parent`, which it uses when `--parent-domain` is left out; `gpuwm downscale` on a multi-domain run still asks for `--parent-domain`.
- `gpuwm gui` opens ArWen in your browser on port 8766, beside the desktop app: the storm wiki, New forecast, My forecasts, Files and a map for each run. Start launches `gpuwm run-plan` detached, so closing the page never stops a forecast.
- The storm wiki opens with 17 cited events, 11 tropical cyclones and 6 tornadoes, each with its observed facts, its sources and its place on the map.
- Each wiki event offers a plan for 8, 12, 16, 24 and 32 GB cards, 85 in all, each accepted by the engine's fit check, with its grids, hours and disk.
- New forecast offers a choice of vertical levels, each priced by the engine for the grid above it; a choice too big for the card says how much it needs.
- A run's map places each picture on the grid it was drawn on, storm-following nests included, over coastlines, borders, states, lakes and US counties served with the page.
- `rw_atms` and the `rw_igra2`, `rw_amv`, `rw_ndbc`, `rw_gnssro` and `rw_wis2` observation doors build from this repository's source, and `rw_goes` and `rw_asos` gain the subcommands the Arwen Global observation streams call.
- The page reads everything from the run folder, so a reload, a server restart or a forecast started from a terminal shows the same way, and Watching streams the run's events live.
- A forecast with pictures draws every grid's frames as they are written, with bounded concurrent renders beside the forecast, on every route that writes history; the end-of-run render draws only what is missing.
- Every button on the page shows its exact `gpuwm` command before you press it, and every command it ran is appended to the run's `commands.log`, ready to paste into a terminal.
- Releases reuse unchanged native binaries: a binary built at an earlier commit of the same line passes the release checks when every file its build reads is unchanged, so a Python-only release ships without recompiling.
- New forecast starts from any past date and hour, from a calendar, a typed date or quick picks; each data source says whether it holds that start, and **Find an event** fills in the box, date and length from the storm wiki.
- Every source New forecast offers knows where its archive starts or how long its publisher keeps it, so a start a source does not hold is refused with the reason before anything runs.
- ERA5 is offered for every hour from 1940 to its newest day, from Google's analysis-ready store with no account key. A `config.intent` with `source = "era5"` and no `era5_provider` uses that store; `era5_provider = "cds"` keeps the Copernicus service.
- The page sizes a run for an 8 GB card as well as 12, 16, 24 and 32 GB.
- New forecast has a Physics step: one table per part of the model, each scheme with its cost against the default, checked by the engine at the run's grid, place and time; every set the check passes can start. `gpuwm physics-catalog` prints the same check in a terminal.
- A finished or stopped forecast's map in `gpuwm gui` has **Downscale**: click a centre or draw a box and choose how fine, how long and for which card; **Review** runs `gpuwm downscale --dry-run` and draws the planned grid, and **Start** runs it as a new forecast.
- A Machines page lists this computer and any SSH host you add with its card and state, draws any forecast's pictures on any of them and starts or stops a cloud machine; `gpuwm machines` does the same in a terminal.
- An optional assistant panel, off until you turn it on; off, nothing is downloaded and no model runs. It answers from the storm wiki, the run folder and the engine's fit check, on a local server or an Apache-2.0 model sized to the card.
- A busy card offers Queue it beside Start now: queued forecasts wait in order in My forecasts, survive a restart and start by themselves when the card is free. `gpuwm gui --owner-file PATH` shares the card with other programs.
- A front end can pass the settings the engine takes: `--point-extent-km` on `gpuwm domain` and `gpuwm domain-fit --point`, `tiles`, `ack` and `point_extent_km` in a run plan's `config.intent`, and one declared row per config key (`gpuwm.config.declared_key_rows()`).
- `[static.highres]` takes land use from CGLC-MODIS-LCZ, WRF's 100 m global land cover (CC BY 4.0), by default, so outside the United States it replaces land use and soil as well as terrain; `landcover_source = "annual-nlcd"` keeps the 30 m US map.
- `--source icon-d2` starts from DWD's 2.2 km ICON-D2.
- Two named physics suites put Thompson microphysics with MYNN surface layer and boundary layer over the RUC land surface: `thompson-mp8-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1`, with radiation by day and night, and `thompson-mp8-mynn-mynn-ruc-dudhia-implemented-unverified-v1`, a daytime suite. `gpuwm domain`, the Physics step and `gpuwm physics-catalog` offer them wherever the WSM6 MYNN and RUC suites run (hrrr and era5), and the catalog's new **Coastal fog and low stratus** preset starts the first. They are offered, not picked: the default suite does not change with grid spacing.
- A single-domain GFS, ERA5 or mapped forecast starts once its first two forcing times are prepared, while later boundaries are prepared beside it; history files are byte-identical. `GPUWM_CHAINED_PREP=0` turns it off.
- `rw_asos fetch|decode --product asos1min` takes the surface stream from the archive's one-minute ASOS pages for the same frozen station table and the same `gpuwm-obs.asos-surface.v2` seam, so a DA cycle of a few minutes gets a report from those stations at every analysis instead of at the few after each METAR: decode strides the valid times by one minute and each report serves the minute it was taken (`--step-minutes` sets another stride), and the record's provenance names the product (`iem-asos-1min`). On a four-minute cycle of a storm-scale DA run the one-minute pages took 2 m dewpoint to all thirty analyses on both grids, where the METAR stream reached ten on the parent and eight on the child. Decode refuses a CSV whose time column is the other product's, `table` refuses a one-minute CSV (it carries no altimeter setting), and a network query is refused because the route answers only by station. `python tools/obs_fetch_asos.py --product asos1min` drives the same door, and `rw_asos --abi` names the product so an older binary is refused as stale.
- `gpuwm domain --clock auto|adaptive|fixed` sets how a run steps, and New forecast offers the same choice. By default (`auto`) a grid from 500 m to 12 km outside the tropics now steps adaptively: each grid starts at its usual step and follows the flow between 3 and 8 s per km of its spacing, with the terrain clock still capping it over steep ground. `--clock fixed` keeps one step; `adaptive` on a grid it cannot run, such as the tropical 2.5 s per km clock, is refused with the grid named.

Fixed:

- A CUDA preparation too big for the card no longer stops minutes in with a CuPy out-of-memory while it builds the forcing states (1792 x 1024 x 55 on a 24 GB card). Every door that prepares on the card (GFS, ERA5, mapped sources and 20CRv3, met_em, native HRRR with its boundary workers, the root of a `gpuwm run` case and the downscale child's parent interpolation) prices the preparation before its first device allocation against the card's free memory: under `--preprocess-backend auto` one that does not fit prepares on the CPU, says why in one line and records it in the receipt's `selection.device_fit`, and an explicit `--preprocess-backend cuda` is refused by name with the CPU command. Those doors all default to `auto` now, `gpuwm go` prepares such a run on the CPU instead of refusing it, and the preparation estimate `gpuwm go`, `gpuwm check` and `gpuwm domain` read is itemized from the same terms, measured on four preparations (1792 x 1024 x 55 GFS: 43.5 GiB before, 33.9 now). A `gpuwm run` case holds one forcing state on the card while it prepares, not two, and met_em no longer prints a memory warning and then allocates.
- An `mp_physics = 28` forecast that starts from an analysis with cloud water, such as HRRR's pressure-level files, now starts that water as cloud droplets the way WRF's real.exe does (`make_DropletNumber`): for 0.01 to 0.3 g/kg with no aerosol yet, 13 to 870 drops per cm3 of about 8 um over land and 2 to 150 of about 15 um over water, where it used to start 0.01 to 0.7 drops per cm3 of about 89 um, drizzle that rained out and froze in the first minutes.
- A Thompson forecast (`mp_physics = 8` or `28`) that starts from an analysis with rain or ice, such as HRRR's pressure-level files, now gives that rain and ice a starting number the way WRF's real.exe does (`make_RainNumber`, `make_IceNumber`): rain colder than -2 C starts as drops of about 0.3 mm and warmer rain near 0.9 mm for 0.1 g/m3, and ice takes a crystal size from its temperature. `mp_physics = 8` used to start both numbers at zero and `mp_physics = 28` at 1 mm drops and 5 um crystals whatever the temperature. An analysed rain, ice or droplet number above zero is kept, and one at zero is filled like a missing one instead of left at zero. The native HRRR preparation (`gpuwm prep --source hrrr`) accepts that start, holding each seeded number to zero where its rain, ice or cloud water is zero and to a finite number above zero, counted against the seed receipt, where it is not; it refused every Thompson start with analysed rain or ice. The stock WRF files the preparation writes (`wrf-native-input/wrfinput_d01`) now carry those numbers in `QNRAIN`, `QNICE` and `QNCLOUD` instead of zeros, and Morrison's (`mp_physics = 10`) and P3's (`mp_physics = 50`) prepared number moments as well.
- Frames of about a million columns or more at 55 levels, such as a 1132 x 906 or a 1792 x 1024 3 km domain, now draw their pressure-level charts: the renderer sizes its pressure-level volumes to the memory the host has available instead of a fixed 4 GiB, and a frame that fits in 4 GiB draws exactly as before.
- A `gpuwm run-plan` or hosted `gpuwm go` forecast marks each history write, checkpoint read-back and end-of-run step in its progress file instead of showing integrating through them, and the progress line calls the read-back "checking checkpoint". The render's skip note gives each skipped product its own reason without the file path, which `--explain` still shows, a series render lists a left-out pressure-level volume once instead of once per frame, and a crash in building that volume names the pressure-level products as the size limit does. The met_em memory advisory reads the same host-memory check `gpuwm go` refuses on.
- In a container, or under any other memory limit on Linux (cgroup v1 or v2, on the process's own cgroup or a parent such as a systemd slice), the renderer and the engine's host-memory checks count only the memory left under that limit instead of the host's free memory, so a render no longer sizes its pressure-level volumes or its worker count past the limit and gets killed.
- Under a memory limit on the process's own cgroup or a parent, such as a systemd scope or slice, a streamed domain's host store and host-RAM price and the `gpuwm mesh` static-field build are sized to that limit instead of the whole computer's memory. In a 2 GiB scope on a 30 GiB computer the planner read 32.8 GB of host RAM and the mesh build no limit at all; both now read 2 GiB.
- Under a memory limit on the process's own cgroup or a parent, the streamed domain's pinned host store checks each store against the room left under that limit instead of the whole computer's free memory, including when no budget was passed. In a 2 GiB scope on a 30 GiB computer it read 25.97 GiB available and admitted a 3 GiB pinned store that cannot fit under the limit; it now refuses it and names the limit.
- `gpuwm domain --tiles auto` sizes a streamed domain to the host RAM `gpuwm go` admits, lateral-boundary tables included, and `gpuwm check` fails a config over it instead of passing it; a frame too large for the renderer's pressure-level limit names the pressure-level products it left out in the render summary.
- A `gpuwm run-plan` or hosted `gpuwm go` forecast marks each history write, checkpoint read-back and end-of-run step in its progress file instead of showing integrating through them, and the progress line calls the read-back "checking checkpoint". The render's skip note gives each skipped product its own reason, a series render lists a left-out pressure-level volume once instead of once per frame, and a crash in building that volume names the pressure-level products as the size limit does. The met_em memory advisory reads the same host-memory check `gpuwm go` refuses on.
- `gpuwm domain --tiles auto` sizes a streamed domain to the host RAM `gpuwm go` admits, lateral-boundary tables included, and `gpuwm check` fails a config over it instead of passing it; a frame too large for the memory the renderer's pressure-level volumes may use names the pressure-level products it left out in the render summary.
- An event page's **Customise this run** now opens New forecast with everything **Run the best simulation for this event** runs, and a Customise nobody changes starts the same plan. It had shown a 12, 3 and 1 km run as one 12 km grid, and started it without the best run's history intervals or a cyclone's sea-surface flux option: the 16 GB layout of the 22 May 2011 Missouri tornado would have written 95 GiB instead of 53. How fine and Review now name every grid and the map shows all three; the button sits beside **Run the best simulation for this event** and the event page says the start data, start time and length.
- `gpuwm domain --tiles auto` sizes a streamed domain to the host RAM `gpuwm go` admits, lateral-boundary tables included, and `gpuwm check` fails a config over it instead of passing it; a frame too large for the renderer's 4 GiB pressure-level limit names the pressure-level products it left out in the render summary.
- ICON-D2 reads DWD's 65 model levels, so the model top is 60 hPa instead of 200 hPa and the forecast starts with DWD's cloud, ice, rain, snow and graupel; the download is about five times larger.
- A moving nest without a statics corridor can now be prepared on one operating system and moved on another. The WPS sampling projection uses the same arithmetic on every machine. Statics from earlier builds differ, so prepare moving trees again with the current release before running them. When loaded from disk, incompatible prepared statics and Python fallback sampling are refused before integration with the preparation command. Preparation and moves in the same process can use matching Python sampling.
- An ICON-D2 preparation no longer stages DWD's six raw moisture fractions beside the humidity and mixing ratios made from them, so its scratch stream holds 827 layers per valid time instead of 1,217, about 360 MB less per valid time on a 465 x 246 point source window, and the disk check asks for that much less.
- HRRR cycles from before July 2018 (HRRRv1 and HRRRv2) prepare again through `--source hrrr-prs` and start with the cloud ice their files publish. Those files carry cloud ice under its older GRIB2 code (CICE), so every such cycle, 2017-01-19 among them, was refused with `lacks required fields ['cloud_ice_mixing_ratio']`.
- The native `--source hrrr` route prepares HRRR cycles from before July 2018 too, reading the cloud ice their model-level files publish under CICE; the bridge refused every such cycle with `missing required field QI hybrid level 1`. `tools/download_hrrr_native_subset.py` and `gpuwm fetch --source hrrr --mode idx-subset` select those records as well. Current cycles decode exactly as before.
- A land soil column whose source soil temperature lies outside 170 to 400 K is rebuilt linear in depth from the skin temperature to the deep soil temperature and keeps its moisture. This follows WRF's real.exe rebuild with its deep-temperature sign corrected, because real.exe's own formula takes the column toward 0 K. The preparation prints how many columns were rebuilt and where, and the proof of a mapped preparation or of a single-domain `--source hrrr` preparation records it as `soil_temperature_repair`; ERA5 and GFS proofs carry no such receipt, only the printed line. HRRRv2 analyses over western snowpack carry soil down to 60 K, so 2017-01-19 00Z over Idaho was refused on both HRRR routes. A missing soil temperature is still refused, and so is one outside that range on an open-water cell of the `--source hrrr` source; an ice-covered lake cell carries the same snowpack column as the land around it and is admitted.
- A high-resolution join that cannot write the fetch folder it recorded now fetches into `<output-root>.highres-cache` beside the bundle, so child and corridor receipts name a folder that exists and a rebuild at the same output root reuses the tiles; delete that folder to reclaim its space.
- A high-resolution join that cannot write the fetch folder it recorded now fetches into a folder beside the bundle named from the first ten characters of the output folder's name, a six-character hash of the whole name and `.highres` (`joined-6b963c.highres` for an output folder called `joined`), so child and corridor receipts name a folder that exists, a long output name no longer pushes them past Windows' 259-character path limit, and a rebuild at the same output root reuses the tiles; delete that folder to reclaim its space.
- Preparing a domain tree from HRRR, GFS, ERA5 or a mapped source refuses an output folder so deep that a file it writes would pass Windows' 259-character path limit on a computer with long paths off, before any source is decoded, and names the longest path and how many characters shorter the output folder, or the folder holding it, has to be. The preparation used to run to the end and publish, and the forecast then failed to open its own inputs as if they were missing: a 92-character output name in a 125-character folder put them at 277 characters. The check counts the published tree, the staging it is written in (for a short output name the unchanged-WRF export's staging is the deepest), and, for an HRRR tree fetching high-resolution statics beside the output folder, that folder's receipts. An output folder in the extended `\\?\` spelling, which `gpuwm go` and `gpuwm run-plan` hand over for a deep run folder, opens at any length and is not refused. A single-domain preparation is not affected.
- Re-preparing a data folder prepared by an older release now works after its packaged mappings or tools change: the old `inputs.json` is kept unchanged and the new binding is written beside it.
- When `gpuwm prep --source-root` replaces a folder's `inputs.json` because its fetched files changed, it also removes the `inputs.<digest>.json` an earlier upgrade wrote for the replaced data, and the REPLACED line names each file it removed, so the folder no longer holds a manifest for data it does not have.
- `gpuwm go` records forecast progress in `run-progress.json`, shows nested progress and finalizing phases, and stops a stalled forecast with exit 124 while keeping its diagnostics. Cold first-step compilation remains preparation, and finished forecasts get the render queues' own wait allowance while their pictures finish.
- `gpuwm go` writes `run-progress.json` for its forecast, shows nested progress and the finishing steps, and stops a forecast that stops making progress with exit 124, keeping the forecast's own error lines. A first run's kernel compilation and the wait for live pictures do not count as a stall.
- `gpuwm go` and `gpuwm run` no longer stop a large forecast as stalled while it writes a history frame or a checkpoint between two model steps, where a finished 1132 x 906 x 55 streamed forecast was stopped with exit 124 while writing its last 5.24 GB frame: each write reports its own progress, with a time limit set by its size, and a write that truly stops is still stopped.
- The ERA5 fetch line `gpuwm domain` prints carries `--retrieve`, so running it as printed downloads the forcing instead of writing a CDS request template, and the ERA5 refusals that quote a fetch line quote one that downloads.
- A statics corridor sealed before 2.7.5 loads when it matches its own prepared tree; one that does not is refused naming the field and the re-preparation.
- Switching only a diagnostic output (`tke_budget`, `hmix_k_diag`, `sase_flux_diag`, `nwp_diagnostics`) no longer refuses an HRRR single-domain prepared bundle or re-prepares a stage whose arrays are identical.
- An adaptive-clock forecast whose starting step is longer than the first output, restart or boundary interval, or than the whole run, now lands on it instead of skipping that frame or stepping past the stop time.
- Sealed preparations with detailed terrain can be joined, extended and run on another machine without matching the original fetch folder. New child and corridor data use a writable local fetch folder, recorded in the join receipt.

- A GDAS, ICON-EU, GEFS, ECMWF IFS or GEM forecast prepares at any whole multiple of its source's boundary cadence, so `gpuwm domain --source gdas --cadence 3` runs on 3-hourly boundaries instead of passing the door and failing preparation with "mapped cadence 10800 seconds differs from target contract". A cadence a source's preparation does not take is refused by `gpuwm domain`, `gpuwm fetch` and the `[fetch]` check before any download, and a bare `gpuwm fetch --source gdas` takes the hourly ladder the source publishes.
- A pip install draws coastlines, borders, state lines and counties on every picture, where a Linux wheel install drew none: the map assets ship in `gpuwm-data`, and a renderer that finds none says so in a `render_basemap_missing` warning.
- The check that a start is published through its last hour asks that hour's files side by side: 51 s to about 8 s for GEM, 89 s to 15 s for ICON-EU.
- Historical IFS open data cycles resolve their dated layout, and cycles the preparation cannot read are refused by name instead of reported unpublished.
- A fetch of an ICON-EU or ICON global cycle DWD has already dropped (its open-data server keeps about a day and has no archive) says the cycle is no longer on the server and names the newest one to use, where it said "not published yet" and left you waiting for a cycle that would never appear. Any source's cycle older than its newest complete one is told waiting will not bring it, and the GUI's date page no longer offers to queue a run for a start the publisher skipped.
- No RFMIP file ships or sits in the repository: RRTMGP reads its trace-gas and ozone climatology from a 136-number table derived from the RFMIP inputs (CC BY-SA 4.0), bit-identical to before, and existing RRTMGP checkpoints resume unchanged.
- The WPS_GEOG notices credit ISRIC SoilGrids250m (CC BY 4.0) for the Noah-MP soil archive and state FAO's terms for the soil texture maps.
- `gpuwm cells analyze` is optional: without a titan binary it is off and says so in one line. `gpuwm cells export` and `gpuwm cells catalog` work without it.
- A run's map across the 180th meridian is framed on the few degrees the forecast covers, not 359, with its pictures and outlines where the view looks.
- `gpuwm downscale --point` takes longitudes from 0 to 360 (`39.5,276` places the child where `39.5,-84` does), and `--ratio` below 1 is refused as it is read, not with a ZeroDivisionError traceback.
- A parent history file that cannot be read, was cut off or holds a value its writer never set is refused naming the file, variable and cell.
- `gpuwm fetch-tables` replaces a damaged `CCN_ACTIVATE.BIN` and `gpuwm doctor` reports it, and a table download from a server that stops sending gives up after three 60 s tries and names `--from`.
- A long forecast's final render starts on Windows: a command past 32,000 characters hands its frames to the renderer in a file, and a stage that cannot start fails with its reason.
- Warm bubbles above 10 K run, and prepared trees keep their pressure.
- Shipped docs, comments and tests no longer carry personal names, internal tool names or paths into one machine's files.
- The map's basemap builder no longer needs shapely, which no dependency table declared; it simplifies the coastlines itself and rebuilds the shipped basemap byte for byte.
- `gpuwm render --products windowed` no longer refuses a whole-hour series with "49 per-hour products selected; GUI ceiling is 32".
- `render-georef.json` records every picture of a run, where each renderer launch rewrote it from its own pictures and a 6 h two-grid run listed 60 of 646; each launch now merges its pictures in under a lock.
- A render into a folder that refuses writes warns and goes on, where the map metadata lock kept a core busy forever.
- Two `gpuwm run-plan` or `gpuwm go` launches into one run folder no longer interleave `events.jsonl`: the second is refused, naming the process that holds it.
- A run whose `events.jsonl` ends in a line cut short by a killed writer replays again. The next launch moves the cut bytes to `events.jsonl.torn-<UTC time>` and says so in an `event_tail_recovered` warning.
- Two `gpuwm downscale` runs given one `--out` no longer both go ahead, and the refused one no longer deletes the other's `child.toml`.
- A stopped `gpuwm downscale` reads as stopped, not failed, without half-drawn pictures.
- Two bridge setups into one folder take turns instead of deleting each other's partial download, and `gpuwm fetch-bridges --dest` on a folder you cannot write answers in words, not a traceback.
- Two assistant messages sent close together in one conversation are both kept.
- The Thompson aerosol column tests hold each card to its own measured residuals, so they pass on an RTX 4090 and an RTX 5070 Ti, and the mp=28 versus mp=8 check compares its near-cancellation level in ULPs.
- The `wp08-freeze` Thompson column fixture's worst-ULP pin carries the card's measured 5 ULP instead of the host build's 3, so the ULP table test passes on a card.
- A job the MCP job manager launched can be stopped with Ctrl+C on Linux, where a background server's runs ignored it; the wrapper outlives the stop and records exit 130.
- A job whose wrapper was ended no longer reads as running: the job manager reaps its wrappers on Linux.
- `gpuwm run-plan` reports what a prepared run wrote, where its completion summary read the wrong folder and a run that wrote two frames completed with `wrfout_count` 0.
- Opening a new version regenerates the terminal binary's licence notices, so the licence gate passes on every commit of the new line.
- The line-ending writer test no longer fails on the public tree, which does not carry the four evidence scripts it listed.
- Seven more tests no longer fail on the public tree: the ones that read a file only the development tree carries skip there and name the file.
- The run report behind the observation battery's 3 km speed anchor now ships with the public receipts, so `tools/battery_route_preflight.py` prints its wall-clock projection on a public install instead of dropping it as "not found", and `tools/flagship/evidence_pack.py` says at once that the nesting ledger documents it needs are in development checkouts only, instead of calling one a missing manifest artifact.
- The device-memory fit behind the observation battery's speed anchors, and the 4 Hz memory samples it rests on, now ship with the public receipts, so the fit source `tools/battery_route_preflight.py` names in every receipt is a file a public install has, where it named notes only development checkouts carry.
- The publication workflow's platform smoke checks out the full history, so a release that carries reused native binaries is not refused by the smoke after its tag is public.
- A fit the engine refuses reads as a plain sentence (too big for the card, or the box leaves the source's area); the engine's full text, with its paths, goes to the page server's log.
- An HRRR start over a coast no longer fails its preparation with "no valid surface-matched HRRR donor within 8 cells": a root's soil donor search takes 24 cells, stopping at HRRR's own edge. A cell that found a donor within 8 cells keeps the same donor.
- An HRRR nest over a small island that HRRR's land mask has as sea prepares: such a cell takes the nearest HRRR land cell past the radius, and the receipt lists every donor more than 8 cells away.
- Grell-Freitas `clos_choice` 1 to 16, `km_opt = 2` under a `km_opt = 2` parent, and SASE on a domain tree run with a not-yet-verified warning instead of a refusal.
- An HRRR domain tree runs the Grell-Freitas `clos_choice` and `ishallow` its configuration sets, where the nest preparation stopped after the download; the route's namelists carry both.
- A `[shared]` `clos_choice` or `ishallow` reaches the Grell-Freitas domains of a tree whose other domains run no Grell-Freitas, where it was refused on those.
- HRRR and GFS runs with WDM6, Kessler or Milbrandt-Yau microphysics no longer stop at preparation ("the HRRR preparation published no portable bundle", "unsupported direct-export microphysics").
- `gpuwm go` shows each configuration warning once, where every stage printed its own copy.
- An HRRR tree on a shortwave-only suite passes its acknowledgements to the nest preparation, which refused it.
- An HRRR domain tree with a storm-following or moving nest prepares its root, where the root preparation stopped with "grid_id = 2 in [relocation] ... is not a domain of this experiment". The root is prepared without the nest's `[relocation]` table, and the tree still moves the nest.
- ERA5 from Google's keyless store no longer runs an older Zarr reader an earlier release left in `~/.gpuwm/bridges`, which refused every request.
- A forecast whose runs folder passes the Windows 260-character path limit runs to the end, where the GFS download and every later stage failed with FileNotFoundError; every stage gets the folder in its extended `\\?\` spelling.
- A full-file HRRR download no longer fails when one connection stops sending: a connection that delivers under 64 KiB in 30 s is dropped and its piece resumed (`RUSTWX_DOWNLOAD_STALL_SECONDS`, `RUSTWX_DOWNLOAD_STREAMS`).
- Every disk and time estimate counts the download and the preparation, and `gpuwm run-plan` refuses before its download a run the free disk cannot hold; `--estimate` said a prepared plan downloads nothing. The defect was in 2.7.7.
- A streamed forecast's host memory no longer grows with its tile count: a 206x204 GFS 3 km forecast held 27.7 GB at 1,190 tiles and now holds 2.8 GiB. The defect was in 2.7.7.
- A frame whose renderer drew some pictures and then failed is drawn again at the end of the run, and a failed render publishes only the pictures its receipt names.
- Live pictures keep pace with a 3 km nest: frames draw up to three at a time within the available CPU and memory, and pictures are unchanged.
- The first picture of a forecast started from the page or a downscale draws with every processor but the forecast's, as `gpuwm go` already did, and live pictures count a container's CPU limit and draw fewer frames at a time when free memory is short.
- Live pictures of a large grid draw only as many frames at once as the free memory holds. Every frame was priced at 640 MiB, measured on small grids, so a computer with 16 GiB free drew three 880x704x55 frames at once, each holding 7.0 GiB; each frame is now priced from its own grid. The CPU limit is also read from a systemd slice, a nested group and cgroup v1, not only from a container's root.
- A dormant nest below a permanent nest fires on a storm on its own grid and comes back after its cooldown.
- On Windows a history file under a folder with non-ASCII letters is checked through the native reader, so a valid frame is no longer failed or quarantined.
- An MCP job whose program cannot start ends with exit 127 and the reason in its stderr, where it read as running.
- `gpuwm fetch` never publishes another day's files under a new cycle, reuses a finished folder offline only while every file matches its receipt, and refuses a damaged file by name with `--force-refetch` as the way out.
- `gpuwm fetch --cycle latest` asks the host `--transport` pins, `--wait-for` waits for a named HRRR cycle, and an HRRR 404 past a host's retention names the host that keeps the cycle.
- `gpuwm fetch` of a named cycle on GEFS, RAP, AIGFS, AIGEFS and the other table routes asks whether it is published before the first download.
- An experiment's `[fetch]` table takes `transport`, the host `gpuwm fetch --transport` pins, where it was refused as an unknown key; `gpuwm go` and `gpuwm run-plan` fetch from that host, `gpuwm go --transport` overrides it, and the plan says which one it used.
- A timed-out server check no longer calls a published start unpublished.
- A run plan's `fetch.args` is read by the fetch command's own parser (`--source=gfs`, `--cycle=latest`), and a relative file in its intent is read beside the plan.
- A finished forecast no longer leaves about 0.8 GB of renderer working stores on disk on Windows past the 260-character limit. Stores a render could not remove are removed once every render exits, or by the next run, and named in a `render_scratch_left` warning.
- The aerosol-aware Thompson suite (`thompson-aerosol-mp28-myj-eta-noah-rte-rrtmgp-v1`) runs on the prepared single-domain route; a machine without the WIF aerosol dataset is still refused by name.
- An aerosol-aware Thompson forecast no longer stops on the health check with 1e15 aerosol per kg atop its outer row, which now takes no vertical transport, as in WRF.
- Both prepared routes offer New Tiedtke (`cu_physics = 16`) as a cumulus scheme. The hand-typed list had missed it.
- Land beside a reservoir, river or irrigated field no longer stops a 2.7.7 preparation with "declarative mapped soil moisture is missing or outside 0..1 on land": the mapping takes a weighted mean where WPS's sixteen-point operator would leave the physical range.
- `gpuwm physics-catalog --into` writes into the quoted `["shared"]` file `gpuwm domain-fit` writes, and a misspelled setting is refused by name.
- Sources deeper than 63 levels, such as ERA5's 137 model levels, prepare on the CUDA backend. The vertical interpolation runs at 64, 160 or 256 levels, and a deeper column uses the CPU bridge.
- Downscaled children no longer make edge storms.
- `[tiles] mode = "auto"` no longer streams a domain in tiles too small to run. When no tiling within the planner's 4.0x redundancy limit fit, auto dropped the limit: a 206x204 GFS 3 km forecast that missed its resident budget by 0.3 GB ran in 1,190 tiles at 49.95x the necessary work, 237 to 547 s per 15 s step against 0.6 to 1.9 s with tiles off, and the review quoted 0.45 to 1.2 s. Auto now keeps the limit, on a nested tree too: the domain runs resident inside the 0.5 GiB kept back for other programs when the card's measured free memory holds it, and is refused before the download otherwise. `[tiles] max_redundancy = false` still streams past the limit. Every streamed plan states its tile count and redundancy in the review, in `gpuwm go`'s log and on the run log line, and the quoted pace counts both: 11 to 121 s per step for a 598-tile plan that ran at 11.1 s. The defect was in 2.7.7.
- New forecast's source list answers for a 3-hour forecast. For AIFS, AIGFS and AIGEFS, whose files come every 6 hours, the check asked the download's lead planner about a 3-hour window inside the page's request; the planner refuses that window, so the whole list, GFS included, and the fit for the same draft answered with a server error. No run asks for that window: a run's download is rounded up to whole steps of the source's file spacing, hours 0 and 6 for a 3-hour forecast from 6-hourly files, and such a forecast runs. The check now asks about the download the run makes, a source whose check fails is only that row's answer (a fit or a start from it is refused in those words, never with a server error), and a start the download would refuse is that row's no in the download's words. Where a model's files stop coming at its spacing before the forecast ends, the row says so first: ICON-EU past 30 hours starts only at 00, 06, 12 and 18 UTC, and ECMWF IFS past 144 hours and GEFS past 240 hours name the lengths that download.
- A forecast whose forecasts folder is on an exFAT drive no longer stops at its first history file with "per-domain wrfout writer failed". A rename there can give a file a new id, and publication compared that id; it now compares length and last write time through the writer's open handle, and still refuses a changed file.
- Preparation, ERA5 downloads, multi-run summaries, DA analysis records and research bundles publish on an exFAT drive, which has no hard links (preparing ICON-EU there failed with "[WinError 1] Incorrect function"): the finished file is renamed without replacing another.
- A forecast stopped because another program changed one of its output files says what changed and what to do on its map page and article, instead of only "Failed". The advice no longer says to retry.
- The default plot set no longer asks for `10m_wind_gusts`, `precipitation_type` and `cloud_cover`. No wrfout carries a gust, a precipitation category or a total cloud fraction, so every run drew 20 of its 24 pictures. General asks for 22 products, `cloud_cover_levels` in place of the total, and every preset names only products a local run draws; the snow preset names its stored snow depth as `var:wrf_snowh`, the spelling the store uses.
- `all` draws every named product the run can draw, and no longer every stored variable beside them: an 18 h run filed 137 of its 204 product folders under names such as `var_wrf_t2_1222df9c491fb635`. The stored variables are the `variables` keyword, and a variable's folder is its own name (`var_wrf_t2`).
- `all` and `windowed` leave out the windows a run cannot close. An 18 h run was asked for 41 windows ending at F024 or F048 and skipped each one on every frame, 805 times, advising an HRRR extended cycle. A window named on a run too short for it is refused with the run's length: `this run's stored frames end at F018`, and a downloaded model's frame too early for a window names the hour the window needs instead of advising an HRRR cycle whatever the model.
- Each skip line of a series render names the file of its own frame. Every line of a 19-frame series named the last file, so an F000 reason read against the F018 frame.
- The end of a run names every product that drew no picture in any of its render passes. It named only the last pass's skips: `qpf_1h`, with 24 pictures, was listed and the three products with none were not. `render-summary.json` lists them as `undrawn_families`.
- A grid writing history more often than hourly draws its windowed pictures, with 1 h maxima over the whole hour. Every window there was refused, and a 15-minute nest's `uh_2to5km_1h_max` from whole hours alone read low on 31,623 of 90,000 cells.
- `gpuwm run-plan --catalog` carries `local_run`: the products a local run can draw, each with the first forecast hour it can exist at, and the reason for every other product. `run-plan PLAN --catalog` narrows it to that plan's length, and the terminal's plot picker offers that list instead of every model's products.
- A warm forecast no longer announces a GPU kernel compile that is not happening. On a card older than the RTX 50 series with the CUDA 12 runtime, every run printed "the kernel cache holds 158 entry(s), none of them for this card -- compiling GPU kernels for sm_86 (the cache carries sm_5 ...)" and set its status to compiling while it was already stepping. CUDA 12 writes those cards' kernels in an ELF layout that keeps the card's SM in the low byte of the header's flags, and the cache check read the second byte, which that layout sets to 5 for every card. The check now reads the byte each measured layout names, and an entry in a layout it does not know counts as possibly this card's, never as another card's. `gpuwm speedrun --compile-mode warm` no longer refuses a warm cache as cold: it looked the card up as `sm_86` where the cache census spells it `86`, so it counted 0 kernels for this card.
- `gpuwm go` from GFS, `gpuwm downscale`, ensemble members and the `--wrfinput` and `--met-em` doors drew only the analysis frame while the forecast ran, and a nested GFS `gpuwm go` none. On an RTX 5070 Ti a GFS 3 km `gpuwm go` draws each frame within 10 s of writing it, a 9 and 3 km GFS nest all seven frames before its forecast ends, and a 250 m child each frame within 19 s. The defect was in 2.7.7.
- `gpuwm cycle` draws each boundary it keeps while the next leg runs, where 2.7.5 to 2.7.7 drew nothing, from the planes the parent carries and nothing else: a parent with no terrain gets no 0 m Terrain Height map. A parent on an MPAS mesh still draws nothing and says why.
- A download counts every file from its first line (`0 of 38 files done`, not `0 of 6`), never counts a failed file as done, and ends at its first failed file, where 2.7.7 downloaded on for 190 s and about 3 GB. An existence check that loses its connection or meets a throttled NOMADS is asked again instead of reported as `no source served this object`, and the GFS full-file progress line stops when the download ends. The defects were in 2.7.7.
- Engine commands on Windows no longer print `UserWarning: CUDA path could not be detected` when CuPy's CUDA 12 toolkit comes from pip, as in the desktop's install. CuPy loads CUDA there all the same, and the line headed every failed command's text in the desktop. 2.7.7 printed it too.
- A downscale of a forecast started from the page or with `gpuwm run-plan` is named after that forecast. Such a forecast writes its frames under `<run>/chain/run/wrfout`, and the child took its name from the nearest folder that was not `run` or `wrfout`, so every child read `Downscale of chain · d02 ×12 · 0.25 km` in the run list and on its parent line. A child now takes the name in its parent's `run-manifest.json`, the name the run list shows for that forecast, and the parent's run folder where there is no manifest; a downscale of a downscale adds its grid to its parent's name, as in `Downscale of Front Range 3 km · d02 ×3 · 1 km · d03 ×3 · 0.333 km`. A parent given by a relative path, such as `chain/run/wrfout` typed inside its run folder, is named the same way, and the child's plan and manifest record the parent's folder and checkpoint as absolute paths: the page finds a downscale's grid through its parent's run folder, and the relative path named none, so the child's map had no grid. The defect was in 2.7.7.
- A start refused because a forecast holds the card no longer leaves a run folder behind. **Simulate this event** and New forecast's Start wrote the folder before the launch was refused, so My forecasts and the event page listed a Ready forecast nobody asked for, the event page's next press started a `-2` copy beside it, and New forecast refused the same name as taken. The event page's button now checks the card before it writes anything or sets up a storm-following nest, and a start refused at launch removes the folder it wrote, so the next press once the card is free starts under the same name. A Ready forecast started on a busy card keeps its folder. 2.7.7 has no `gpuwm gui` and did not have this.
- The page: the Physics check with several families picked no longer fails on Linux with "File name too long"; a downscale Review enables Start only for the settings it checked; search keeps a typed "?" and every word after it; Files shows only the folder picked last and says when its list cannot be read, with Try again; a reorder made while the queue reads the card starts the forecast moved to the front; and a Stop on Windows that could not end the run keeps the card claimed and says so instead of reporting it stopped.
- Without the geography data, Start and **Simulate this event** refuse with `gpuwm fetch-geog --datasets wrf` and write nothing, instead of launching a forecast that fails at once. A failed forecast's page shows the remedy its refusal states, not "fix the plan document" or nothing, and a stopped forecast's Article says why it stopped.
- A downscaled child run with no `--tiles` records the price its review gave. The run priced such a child on no card, so the estimator charged the context and kernel memory of the largest card it knows: a 552x552x49 child on a 15.47 GiB RTX 5070 Ti recorded a 15.86 GiB envelope in its `child_streaming_decision` event and `report.json`, more than the card it then ran on, where the review and a `--tiles=auto` run priced it at 13.90 GiB and it peaked at 11.57 GiB. The run now reads the card before preparing the child whatever `[tiles]` says, prices the child on it, and records the card and its free memory beside the price. The defect was in 2.7.7.
- Forecast geography sets up in a deeper folder, and a failure on this computer no longer reads as a download failure. The resume record's staging copy was the record's own name plus a process number and a clock stamp, 35 characters longer than the record, so a geography folder of 150 characters or more failed every setup on its second archive with "download failed from https://huggingface.co/...; the partial file is kept and a re-run resumes", when nothing had been written, the mirror was fine and every re-run failed the same way. Staging files now take a short random name claimed exclusively, and extraction stages under one too, so a geography folder of up to 175 characters holds the WRF geography where 149 was the most before. A write this computer refuses names the path's length and, where Windows' 259-character limit refused it, how many characters shorter the folder has to be. A failed download promises a resume only when it left a partial file the next run resumes, and `gpuwm fetch-bridges` reports its failures the same way. The defect was in 2.7.7.
- A downscale started from a forecast's page asks for the boundary cadence the parent's frames have, shown on the form (`--max-boundary-interval-seconds`), instead of sending `--accept-parent-cadence` for a choice nobody made. The 900 s guidance note over hourly frames stays because it holds: hourly against 15-minute edges left a 250 m child's storms correlated at 0.44 after 2 h.
- A downscaled child that blows up no longer names a cell it did not measure. The refusal said `W went non-finite at cell (k=0, j=0, i=200)` for a block of W that had gone on every level from the south edge to row 60: that cell was the lowest index of every non-finite value the health check found, so it was always the block's lowest level and southmost row, and W was named only because it is first in the survey's list. The first sentence now gives the fields found non-finite at the check after step N, the k, j and i ranges they fall inside with the cell count, and whether that box touches a lateral edge; a single cell is named only when one cell went. The message also says where |w| was largest at the last check that measured it and how far that is from the nearest edge, and every `child_step` event line carries that place as `w_max_cell` and `w_max_edge`. The defect was in 2.7.6 and 2.7.7.
- A downscaled child keeps one checkpoint set (an 11 h 250 m child kept all 11, 28.5 GB) and is refused before it starts when it would not fit on the disk under `--out`; its review shows the figure.
- `[tiles] mode = "auto"` no longer streams a domain in tiles too small to run: it keeps the 4.0x redundancy limit and runs the domain resident when free memory holds it, or refuses it before the download.
- New forecast's source list answers for a 3-hour forecast, where AIFS, AIGFS and AIGEFS, with files every 6 hours, made the whole list answer with a server error; a failing source is now only its own row's answer.
- A forecast whose forecasts folder is on an exFAT drive no longer stops at its first history file with "per-domain wrfout writer failed".
- Preparation, ERA5 downloads, run summaries, DA analysis records and research bundles publish on an exFAT drive, which has no hard links, and `gpuwm stream` refuses such a work folder at its start.
- A forecast stopped because another program changed one of its output files says what changed and what to do, instead of only "Failed".
- The default plot set no longer asks for `10m_wind_gusts`, `precipitation_type` and `cloud_cover`, which no wrfout carries, so every run drew 20 of 24 pictures; every preset now names only products a local run draws.
- `all` draws every named product the run can draw and no longer every stored variable beside them; the stored variables are the `variables` keyword.
- `all` and `windowed` leave out the windows a run cannot close, where an 18 h run skipped 41 on every frame, and a window named on a run too short for it is refused with the run's length.
- Each skip line of a series render names its own frame's file, where every line named the last file.
- The end of a run names every product that drew no picture in any render pass, not only the last pass's skips, and `render-summary.json` lists them as `undrawn_families`.
- `gpuwm run-plan --catalog` carries `local_run`: the products a local run can draw and the reason for every other, and the terminal's plot picker offers that list.
- A warm forecast on a card older than the RTX 50 series with CUDA 12 no longer announces a GPU kernel compile that is not happening, and `gpuwm speedrun --compile-mode warm` no longer refuses a warm cache as cold.
- `gpuwm go` from GFS, `gpuwm downscale`, ensemble members and the `--wrfinput` and `--met-em` doors draw every frame as it is written, where they drew only the analysis frame and a nested GFS `gpuwm go` none.
- `gpuwm cycle` draws each boundary it keeps while the next leg runs, where 2.7.5 to 2.7.7 drew nothing.
- A download counts every file from its first line, never counts a failed file as done and stops at its first failure, and an existence check that loses its connection asks again. The defects were in 2.7.7.
- Engine commands on Windows no longer print `UserWarning: CUDA path could not be detected` when CuPy's CUDA 12 toolkit comes from pip, as in the desktop's install.
- A downscale of a forecast started from the page or with `gpuwm run-plan` takes its name from the parent's `run-manifest.json` and records the parent's paths as absolute, so its map finds its grid. The defect was in 2.7.7.
- A start refused because a forecast holds the card no longer leaves a Ready forecast behind: **Simulate this event** checks the card before it writes anything, and a start refused at launch removes its folder.
- Without the geography data, Start and **Simulate this event** refuse with `gpuwm fetch-geog --datasets wrf` and write nothing, instead of launching a forecast that fails at once. A failed forecast's page shows the remedy its refusal states, not "fix the plan document" or nothing, and a stopped forecast's Article says why it stopped.
- The page: the Physics check with several families picked runs on Linux, a downscale Review enables Start only for the settings it checked, and a Stop on Windows that could not end the run says so.
- A downscaled child run with no `--tiles` is priced on the card it runs on, as its review priced it; a 552x552x49 child on a 15.47 GiB RTX 5070 Ti recorded a 15.86 GiB envelope. The defect was in 2.7.7.
- Forecast geography sets up in a folder of up to 175 characters, where one of 150 or more failed every setup as "download failed".
- A downscale started from a forecast's page asks for the parent frames' own boundary cadence, shown on the form (`--max-boundary-interval-seconds`), instead of sending `--accept-parent-cadence` for a choice nobody made.
- A downscaled child that blows up names the fields, their k, j and i ranges and where |w| was largest, not a cell it did not measure.
- Moving-nest statics corridors cover only reachable ground (`reach_speed_m_s`).
- GFS runs default to 50 hPa tops, fetched automatically.
- A start carrying analyzed cloud (HRRR) no longer fails preparation with "hydrometeor disposition falsely excluded or included WRF target support".
- `[static.highres]` prepares a domain that reaches past a source's coverage, where it stopped with `high-resolution land cover lacks N target values`: those cells take the 30-arc-second baseline, blended over five cells, with one warning. `gpuwm go` also accepts a config carrying `[static]`.
- `gpuwm go` from GFS prepares and runs Kessler, Milbrandt-Yau and WDM6, which stopped in preparation because the unchanged-WRF files have no package for them, and keeps the checkpoint sets `--keep-checkpoints` names, one by default. An HRRR domain whose interpolation fits is no longer refused near HRRR's edge for its soil donor search, a nest that starts late is charged disk only for the frames it writes, and a donor receipt over HRRR's whole grid is written.
- `gpuwm fetch --mode auto` takes the default transport on every source, and a GDAS window may begin on any lead it publishes.
- Checkpoints are pruned again after a nest retires, and a failed forecast's report states the model time it failed at.
- Bulk shear from a WRF run is drawn in knots: it was stored in m/s and read off the knot colour bar, so a 22.7 m/s peak showed 22.7 kt where it is 44.1 kt, on the named pictures and on `var:` pictures. Ensemble mean and PMM panels of 2 m temperature, 10 m wind and precipitation are drawn in the degF, kt and inches their bars state. A narrow nest keeps its valid time in the header, `gpuwm render` shows the renderer's warnings, and a folder drawn through a long run keeps publishing past 4,096 render receipts.
- In `gpuwm gui`, Show command beside Start and Queue it gives the line that runs on the machine picked under Run it on, and stays open when another machine is picked; an assistant Settings field edited while Save is on its way keeps the newer value.
- A preparation whose scratch disk cannot hold the decoded frame stream is refused after the first valid time, before the stream is written, naming the folder, the bytes it needs and the bytes free, with `GPUWM_COMPOSE_SCRATCH` as the remedy. A GEM 48 h run decoded for 14 minutes and then failed as `FileNotFoundError` telling the user to supply an input file; a disk that fills anyway is now the same refusal.
- A tree prepared on one machine runs on another, and observations gridded on one machine assimilate on another. The forecast was refused with `native static receipt geometry differs from target` when the two machines' math libraries differed in a latitude's last digit; grids now match on their exact definition, computed positions within a thousandth of a cell. A grid moved by a cell or with another spacing or size is still refused.
- A `gpuwm run`, `resume` or `branch` forecast that is finishing (writing its last history frames, checking and digesting its final state, hashing its frames) is no longer stopped as stalled and run again from its last checkpoint: each finishing step names the bytes it moves and may take the step limit plus those bytes at 8 MiB/s, so a step that stops answering is still stopped. A run resumed at its last step is no longer stopped by `--prep-timeout` while it finishes.
- NSSL microphysics (`mp_physics = 18`) runs with `use_adaptive_time_step = true`; every such forecast stopped at the first step change with `NSSL production binding timestep differs from runtime`. New Tiedtke, the WSM6 and WDM6 SR check and `[spectral_numerics]` also follow the changing step.
- `gpuwm go` on GFS refuses a run its disks cannot hold before it claims a run folder or downloads a byte, with the same disk check `gpuwm run-plan` makes. A picture render the run stops waiting for is ended and publishes nothing, where its late finish replaced the pictures the end-of-run render drew and its process outlived the run, and a stopped or failed `gpuwm run-plan` run ends or collects its first-frame render as well as the every-frame one.
- AIFS cycles that publish geopotential on pressure levels and no geopotential height (every cycle sampled from May 2025 to February 2026) prepare instead of refusing with `lacks required fields ['geopotential_height']`: the height is read from the geopotential. Where a pressure-level source publishes neither at a level, that level's height is integrated from the source's own temperature, humidity, surface pressure and terrain height, counted in the preparation receipt (`completed_fields`) with one warning.
- A lake whose downloaded area holds no source water no longer stops preparation with "N water cells have no admissible water temperature" (a default GDAS run, ICON global), nor an island whose area holds no source land with a near-zero TSK: the cell takes the source's skin temperature on the other surface there, counted in the receipt and said in one line. A GFS lake whose nearest water could lie past the fetched area is counted instead of refused.
- A `gpuwm domain` memory refusal names a lighter `--physics-profile` or a shallower ladder only when it fits the same card with the same `--physics-choices`, so following the advice is no longer refused again (a 3 km domain on a 5 GB card advised a suite needing 4.00 GiB against 3.75). A shallower ladder is named by the flags that request it. When nothing lighter fits it says so and names the free memory a larger card needs.
- An editable install of an extracted source archive (no `.git`) binds its run identity to the content of the files it lists, so `gpuwm doctor` no longer reports run provenance missing; `gpuwm research hardware` and `gpuwm domain-tiles` suggest only options they accept; the install guide says platform wheels carry the compiled Rust tools.
- A land cell whose soil map says water keeps its land use as its vegetation, as in WRF, and takes silty clay loam soil; it ran as mixed forest, on 2,152 of the 25,600 cells of a 500 m nest over a coastal city. The stock-WRF export writes the same soil.
- `gpuwm prep --source era5-l137 --source-root DIR --experiment-config X --wps-namelist Y` runs as documented, and `gpuwm go X --data-dir DIR` binds the same folder: its model-level files and the `era5-combined.grib` `gpuwm fetch --source era5 --retrieve` wrote beside them, with the manifest and output root written for you. `gpuwm domain` prints the request and the fetch that fill the folder, the fetch refusal prints lines that run, and lakes, sea and snow start from the same state as `--source era5`. Run again, the line prepares into a new folder beside the last one instead of exiting 78, and replaces `DIR/inputs.json` only when the folder's files changed. The `gpuwm sim` line `gpuwm prep` prints draws every product from each output as it lands, where it drew nothing.
- In `gpuwm gui`, Remove takes a queued forecast out of line even when its folder cannot be deleted, and says so, where it came back at the next look and started; a queued forecast that starts at once is answered as queued, not with HTTP 500; a disk that fills while a forecast is saved starts nothing and frees its name (HTTP 507); and without CuPy, Start is off and a forecast queued here is held naming the install, where it was launched and failed at its first step.
- `gpuwm gui`: a finished downscale shows its grid and levels without a false warning, a map time no grid turned on has a picture at says so and Play steps past it, the header, map controls and time bar fit a phone-wide window with the run framed between them, a cleared or changed search no longer brings old choices back for Enter to open, a machine still installing gpuwm's requirements takes no draw or forecast, and a forecast or downscale whose start failed only after it launched says it started, keeps the card, and is not launched again from the queue.
- Colour bars label every tick with its own value (a narrow 200 hPa height bar printed `12.2` at all 14 ticks); raw hail and maximum helicity keep their file units, so hail draws in inches; the live viewer draws `var:<field>` selections, from the desktop too; a long product list no longer stops picture publication; `--cache-bytes` and the desktop viewer cache take any positive budget.
- A forecast runs six acoustic substeps per step on a domain whose terrain is steeper than four hold (a slope of 0.70 in any direction at the shipped `epssm` 0.5, from the engine's measured stability map), and says so: a generated 500 m Andes forecast stopped at model second 40 with four and now completes. Flatter domains run as before, bit for bit. Over the highest ground the derived etac also keeps every layer at least 15 percent as deep as over flat ground: a generated 1 km forecast under Aconcagua, ordered at etac 0.2 with one layer 1 percent deep, stopped at model second 350 and now completes. The dry-mass check on specified and nested grids no longer reports a false 4 percent loss.
- IFS open-data starts from earlier publications of the product prepare. Cycles without the 10 hPa level stopped with `air_temperature vertical coverage mismatch; missing=[1000.0]` (the level is in Pa); older cycles also spell their soil layers differently and carry no surface geopotential. The column is built on the levels the files carry, the older soil records are read through `record_aliases`, and terrain is derived from each column's height at its surface pressure; the receipt records each, with one warning.
- A source checkout's bridge build is judged against the files cargo compiled it from. After a pull that changed a file only another binary of the same crate reads, the CPU preprocessing library and the GRIB2 tools were refused as older than their sources even after the `cargo build` the refusal printed.
- A prepared cache or HRRR preparation extended on a drive without hard links (exFAT, or across drives) copies the earlier hours and checks every copy byte for byte, where it refused; a copy the disk cannot hold is refused by name first, and `gpuwm stream` runs on such a drive. Switching `tke_budget`, `sase_flux_diag` or `hmix_k_diag` no longer refuses an unchanged prepared cache, and a nested restart takes the diagnostic switches a single domain takes. A mapped source's terrain and borrowed fields join across the 0/360 seam of a global donor, and terrain with missing cells outside the requested area is no longer refused.
- The command line refuses a NaN, infinite, negative or zero value by the option's name where the option cannot use it, and names a typed path that is missing or a folder, at exit 2 instead of a traceback or a silent default; `gpuwm cycle` refusals print as sentences, `gpuwm obs <instrument>` passes `--help` and leading switches to its binary, `gpuwm spectral-op response` writes `null` for an undamped row, the terminal names a snapshot size it cannot take, and doctor on a box without CuPy keeps `gpuwm domain --card` and `gpuwm check --vram-gib` open.
- Land use, vegetation, soil and land mask pictures (`var:wrf_lu_index`, `var:wrf_ivgtyp`, `var:wrf_isltyp`, `var:wrf_landmask` and the other category planes) draw each cell's own code, with one colour bar band per code labelled with that code. They were interpolated into codes no cell holds, under a continuous bar. Smooth colour bars end on the colour the map draws at their top, where they stopped one colour short (a land mask of 1 was yellow on the map and green on the bar).
- A tiled GFS domain is sized and admitted against the computer's RAM as well as its card. Its preparation runs on the CPU and holds no card memory, so nothing weighed the RAM it needs, and a 48 GiB card on a 12 GiB computer was sized a 902x720x76 root whose preparation peaks above 14 GiB. `gpuwm domain` now fits the RAM, `gpuwm go` and `gpuwm check` refuse before the download when the arrays the preparation must hold at once exceed it and warn when only its estimated peak does, and the memory line names the preparation's host RAM.
- A forecast over steep ground under a strong wind at crest height takes a shorter time step, or six acoustic substeps, where the engine's measured stability map says its configured pair does not hold. The wind is read from the run's own start and boundary data, the step is divided by a whole number so every output still lands on a step, and each changed domain prints one line. Generated 3 km forecasts under a 79 m/s Himalayan jet and a 45 m/s Andes jet, and 1 km forecasts under Aconcagua in a 66 m/s jet, stopped between model second 60 and hour 3; they now finish. On the adaptive clock the six substeps are a floor under the count it takes from its step (`min_time_step_sound`), which fell back to four: the 1 km Aconcagua forecast stopped there at model second 135 and now finishes. Other domains run as before, bit for bit.
- A prepared forecast whose outer domain carries boundary data or map factors for another grid than its start state is refused before it starts, naming the domain and both grids, where the steep-terrain time-step reading stopped with a bare `operands could not be broadcast together` error.
- The estimated host RAM of a nested tree prepared on the CPU no longer counts every nest's preparation at once, so `gpuwm domain` stops shrinking nested tiled ladders further than they need; on 2 and 3 domain trees the estimate sits 1.01 to 1.12 times the measured peak.
- A preparation on the CPU starts at most eight worker threads unless `--preprocess-workers` names more, the count its host RAM estimate was measured at. It started one per CPU, and on a 64-vCPU machine a 744x594x49 GFS domain peaked at 7.79 GiB against the 7.50 GiB `gpuwm check` and `gpuwm domain` sized for; it now peaks at 7.10 GiB, and on a shared 64-vCPU machine it prepared no slower.
- The installers build the mapped-source decode engine and the dealiasing library, so doctor on a source install no longer reports those routes MISSING.
- `gpuwm go` refuses a misspelled `--products` name at review, before anything is downloaded; it ran the whole forecast and then drew no picture of any product (`unknown product 'compsite_reflectivity'`). Every reader of a product list now splits it the way the renderer does, so `gpuwm remote sync-native-plots` and a remote run's map preparation no longer refuse `xsec:QCLOUD=0.01,0.1/wa` or `xsec:wa=1,2,5@5` as the invalid products `0.1/wa` and `5@5`.
- `gpuwm fetch` ends with a `gpuwm prep` line that runs as printed once the flags it names are added. HRRR's named four of the six its door needs and printed no command name, so it was refused for `--domain-spec` and `--namelist-input`; the table routes printed `--source` and `--input-list` only, leaving out the supplement and manifest flags; GDAS printed a sentence and no command. 20CRv3 manifest authoring does the same, and no longer names the GRIB2 tool pair as yours, which pinned the Python decoder.
- In `gpuwm gui`, a run on another machine whose updates stopped says so and offers Resume updates instead of reading as live, a finished forecast's map keeps reading while its pictures are still being drawn, the engine's run warnings (a warm bubble above 10 K) show on the map and the article, and a late My forecasts reply no longer undoes Move up.
- RTE-RRTMGP no longer dims sunshine under MYNN subgrid cloud, and both RRTMG engines size MYNN's cloud water and ice as cloud in every layer it fills, beside a trace of resolved cloud too.
- `gpuwm go` and `gpuwm run-plan` price a preparation's decoded frame stream (about 82 GB for GEM GDPS over 48 h) and refuse before the download when its scratch disk cannot hold it or `GPUWM_COMPOSE_SCRATCH` names no folder, naming the folder and the variable; that refusal and a preparation's own reach `gpuwm gui` and the run's failed event as the sentence, its remedy and its folder instead of `prepare failed (exit 78)`. The supercooled water path and hydrometeor maps draw in their colour bars' colours, and a negative interpolated vapour value at or above 1100 hPa is floored and counted instead of refused.
- `gpuwm go --section` (and `render_section` in a run plan) cuts the `xsec:` products in `--products` along a line, on every picture drawn as frames land and at the end; an `xsec:` product with no line is refused at review before anything is downloaded, where it ran the whole forecast and drew nothing. The closing note names each product that drew no picture with its own reason, and the terminal's plot picker keeps `xsec:QCLOUD=0.01,0.1/wa` as one product instead of listing `0.1/wa`.
- A resumed forecast keeps the rain a moving parent carried into its nest's edge, draws its first new hour's `qpf_1h` and `qpf_total` from the history saved before its checkpoint, and is admitted on disk for what it still writes, not the whole run. A moving nest's high-resolution terrain is cut on one pixel grid for every footprint, so its first move is no longer refused with `footprint-rebuilt statics differ`.
- AI-GEFS files from NOMADS prepare instead of refusing at the member check or with `lacks required fields ['surface_pressure']`. AI-GEFS and AIGFS derive surface pressure at every lead from their own mean-sea-level pressure; AIGFS held the analysis value at every lead, and AI-GEFS from AWS read one up to 59 hPa off.
- A domain across the 180th meridian prepares from every mapped global source as it does from GFS, where it was refused with "target points fall outside the source grid" (ICON: "mapped longitude coordinate is not a regular axis"), and an ICON window whose only land is small islands is no longer refused as a missing soil field. `gpuwm go` also downloads for a domain south of the equator, where it stopped with "argument --area: expected one argument".
- AI-GEFS cycles from January to late April 2026, which the AWS mirror serves re-encoded by another GRIB writer, prepare instead of refusing with `0 of 177 GRIB message(s) ... match this mapping's selectors` or at the member check.
- Ocean domains whose only land is atolls or cays at 0 m prepare instead of being refused as missing terrain, preparations over small islands a coarse source holds as sea finish in about a minute instead of running past 30, and island land whose source area holds no land starts at its skin temperature and its soil's field capacity instead of 285 K and saturated.
- A start carrying analyzed cloud no longer fails preparation with "hydrometeor disposition falsely excluded or included WRF target support" (HRRR on 2.7.7): the receipt follows the interpolation that wrote the state and counts where its prediction differs.
- `[static.highres]` prepares a domain that reaches past a source's coverage (out to sea, across a national border, over an unpublished terrain tile): those cells take the 30-arc-second baseline, blended over five cells at the edge, with counts and bounds per field in the receipt and one warning. It stopped with `high-resolution land cover lacks N target values`. `gpuwm go` also accepts a config carrying `[static]`.
- `gpuwm fetch --mode auto` takes the default transport on every source instead of being refused; a GDAS window may begin on any lead it publishes, and a single lead accepts `--cadence`; an ensemble member's fetch folder reused for a new window prepares that window beside the old one; and the observation archive reader refuses a member size it cannot hold instead of wrapping it or allocating it whole.
- Checkpoints are pruned again after a nest retires; a run that draws no pictures is not charged for them at the disk check; a failed rainfall pass or a picture removed by hand no longer spoils a render's publication; `--products` keeps a section beside `all` and each section's whole level list (`xsec:QCLOUD=0.01,0.1/wa`); and a failed forecast's report states the model time it failed at.
- A large domain's forcing states prepare on many CPUs instead of one core: every CPU on the card route, and the preparation's worker count (at most eight unless `--preprocess-workers` names more) on the CPU route. Each forcing time's float64 setup columns run in parallel (the same bytes as serial), the hydrometeor disposition receipt is built only for the start time, whose result is the one kept, and the masked soil and skin mapping evaluates only the cells still waiting for a value. The prepared state is byte-identical. On one 24-core machine the forcing stage of an 896 x 512 x 59 HRRR domain with seven forcing times took 591 s against 760 s, and of a 1792 x 1024 x 55 one with two times 398 s against 911 s. The worker threads cost host memory: each worker pool hands its free pages back to the system when it closes, and peak host memory still rises by about 0.9 GB on the first domain (17.2 to 18.0 GB) and 1.3 GB on the second (46.0 to 47.2 GB). `--preprocess-backend cuda` runs the same host steps.
- `install.sh`, `install.ps1` and the manual install steps build the mapped-source decode engine and the dealiasing library, so doctor on a source install no longer reports the default decode and dealiasing routes MISSING, and the installers' closing doctor sees the new `.venv`. Doctor no longer calls a working `[gpu-cu12]` install the wrong extra on a CUDA 13 driver.
- An all-land GFS or GDAS area prepares instead of stopping at `missing-value policy mismatch`; AIGFS and AIGEFS `--cycle latest` picks a cycle whose GDAS analysis is out; `gpuwm sim --print-command` prints a PowerShell line on Windows; `--forecast-start-hour 0` authors its manifest; `gpuwm research --explain hardware` keeps `--explain`.
- A frame drawn as it lands keeps its composite reflectivity when an earlier frame of the series, such as the analysis frame, stores none, and a request with no windowed product draws each whole hour alone; the lead-1 h frame of a 3 km CONUS run lost its reflectivity and took 77 s against 14 s, and a render of a whole run from its analysis frame drew none.
- The fit check reserves less card memory for RRTMGP radiation, because the radiation solver is now priced at the 3,600 bytes per thread every card compiles it to rather than an older 5,152: on an RTX 5090 a forecast with RRTMGP reserves up to 8 MiB less with Morrison or Kessler microphysics and up to 386 MiB less with P3 or none.
- Preparation runs on the card on CUDA 13 as well as CUDA 12, where `--preprocess-backend auto` sent a CUDA 13 box to the CPU; whenever `auto` prepares on the CPU it says why in one line and in the preparation receipt.
- The masked soil, snow, skin temperature and sea-ice mapping runs in the Rust preprocessing library across CPU threads, under both preprocessing backends, instead of single-core NumPy, and so does the native HRRR route's soil mapping (its land-only bilinear stencil, the search for the nearest HRRR land cell, and the stencil's report). The mapped fields, their repair counts and the stencil report are byte-identical. On a shared 64-core machine, at the CPU backend's default of eight threads, the masked fields of one forcing time took 0.6 s against 10.0 s on a 1792 x 1024 x 55 HRRR domain, whose forcing stage took 390 s against 458 s. A CPU preprocessing library built before this is refused by name with the rebuild remedy, and `gpuwm doctor` reports it missing. `--preprocess-workers` now sets the threads of these host steps under `--preprocess-backend cuda` instead of being refused there.
- The lake skin temperature search and the water-temperature assembly run in the Rust preprocessing library across CPU threads, under both preprocessing backends, instead of single-core NumPy: the labelling of water bodies, the source cells each body owns, the search for the nearest source water cell of each model lake, the bilinear blend of source water temperature over each water body's donors, the sweep that closes a body's remaining holes from its own cells, the repair of water cells whose provider left no admissible temperature, the per-body assembly, and the corner blend of a water-temperature overlay. The CPU backend's bounded surface-nearest search moved too. They take `--preprocess-workers` like every other host step, and every value is byte-identical at any thread count. When two water bodies claimed one source cell equally, which one owned it depended on the machine's NumPy sort; the higher label now wins everywhere. On a shared 24-core machine the lake search of a 1792 x 1024 GFS domain (46,539 lakes) took 0.010 s against 0.36 s, the whole water-temperature assembly of that domain (3,820 bodies) 0.10 s against 4.3 s, and on a 1792 x 1024 target the blend of one water body took 14 ms against 142 ms, the hole sweep of a large body 2.9 s against 17.1 s, and the labelling of 15,987 water bodies 0.01 s against 0.64 s. A CPU preprocessing library built before this is refused naming the entries it lacks, with the rebuild remedy, and `gpuwm doctor` reports it missing.
- `--preprocess-backend auto`, the default, prepares on the CPU when the card is busy (50% utilization or more), when the preparation's priced device memory exceeds the card's free memory, or when the card cannot be opened by CUDA, where it prepared on the card anyway: beside a running forecast on an RTX 5090 that took 796 to 990 s against 678 s on the CPU, and on a card another program had filled its first allocation failed with out of memory. The line it prints and the preparation receipt name the reading, and each child of a GFS domain tree prepared on the root's backend records the root's reason.
- New forecast keeps **Named pictures or a cross-section** when the Product names box is emptied and refuses to start until a picture is named; it had started the run with the standard set while the page still showed the named choice, and a reload switched the choice back. The terminal no longer publishes or draws a downscale with the standard set when its plots name a cross-section with no line; it says the line is missing. A remote run that asked only for cross-sections shows a note in its map and plot progress instead of preparing the node's default maps.
- The standalone RW-WPS package builds again: its build refused on imports that only a forecast reaches (the check of a statics corridor sealed by an older build, and the saved history a resumed forecast draws from).
- The standalone RW-WPS package builds again with chained preparation and the steep-terrain clock in the tree. It has no forecast to start early, so a GFS, ERA5 or mapped preparation publishes when it is complete and never needs the forecast's memory check, which the package does not carry.
- The standalone RW-WPS package's `--preprocess-backend auto` reads the card's load as `gpuwm` does: it prepares on the CPU when the card is busy (50% utilization or more), when the preparation's priced device memory exceeds the card's free memory, or when the card cannot be opened by CUDA, and says which in its line and receipt, where it prepared on that card anyway.
- An HRRR domain tree prepares on a computer without CuPy or a usable card. HRRR preparation now defaults to `--preprocess-backend auto`, as GFS, ERA5 and mapped preparation do, so it prepares on the CPU when no card is usable instead of refusing, and the step that builds the nests from the prepared d01 no longer needs CUDA; it used to stop with `No module named 'cupy'` before the first nest. A CPU HRRR preparation of two or more hours also runs through: each boundary hour prepared on its share of the CPU threads was refused at the first boundary hour with "preprocessing backend receipt differs from the resolved public selector".
- On Windows, the rebuild command `gpuwm doctor` prints for an out-of-date static-field library is joined with `;` and pastes into Windows PowerShell 5.1, which rejected the `&&` it carried.
- A fresh `pip install gpuwm[gpu-cu13]` compiles with NVRTC 13.4.92 since cuda-toolkit 13.4.2, and this release carries readings of that compiler on sm_120 (RTX 50 series) cards: `gpuwm check` prices Noah-MP (`sf_surface_physics = 4`) from them and says they were measured, where it charged the ceiling over other compilers and said the frames were not measured on this card.
- On Windows, the build command printed when `gpuwm fetch --mode` needs the Rust fetch backbone, when an observation product cannot load the Rust NetCDF writer, and when HRRR preparation falls back to the slow NumPy mapping pastes into Windows PowerShell 5.1, which rejected the `&&` it carried; `gpuwm go` on an ERA5 config prints `gpuwm check X; if ($?) { gpuwm run X }` there, so the run still starts only after the check passes.
- On a machine with more than one card, a card whose `nvidia-smi` row reads `[N/A]` is no longer measured with another card's memory and load. `--preprocess-backend auto` keeps the card when the memory probe itself fails on a card CUDA opened, where it moved the preparation to the CPU saying CUDA could not open the card, and `gpuwm check` names that failure where it said no CUDA device answered. The terminal answers a native plot request for a remote run that asked only for cross-sections with the node's note, where it said the plots were still being prepared.
- A single-domain forecast starts beside its preparation only when the machine's RAM also holds every boundary interval and the forecast process itself, so a large domain on a 16 GiB machine starts after preparation instead of running out of memory. Ctrl-C in `gpuwm go` during such a forecast stops its preparation too and reports a stop, not a failed forecast, and a domain tree, met_em, native HRRR or `gpuwm run` says why its forecast waits for preparation.
- `gpuwm go` on an ERA5 config quotes the config's path in the `gpuwm check` and `gpuwm run` line it prints, so a config in a folder with a space pastes as one argument in PowerShell and in a POSIX shell, where it split into two.
- The LETKF sums its ensemble-space products in one fixed order on every card, so two cards given the same members and observations write the same analysis bytes. The library route summed them in a per-card order, and a convective cycle grew that last-bit difference into a different storm (footprint rain 0.31 against 0.20 from byte-identical model legs). `LetkfConfig.matmul = "library"` keeps the old route as an A/B arm.
- A radar analysis no longer leaves vapour above saturation for the model's first step to condense into warming, and no longer dries its members doing so. Both increment writers cap the resulting vapour at liquid saturation at the resulting temperature where the increment moved theta or vapour, keeping the background's own supersaturation ratio (on a storm-scale 1 km child's first analysis, 5.03 Mt of supersaturated vapour went to 0.003 Mt; condensed, it had warmed 1 to 3 km by 1.4 to 2.1 K over the storm box). A per-member cap alone cuts the members' upper tail wherever the mean sits near saturation and removed 30 to 44 g m-2 per member per analysis, so the analysis first bounds the whole ensemble with a mean-preserving rule on the headroom: a member over its limit ends at it, the members with headroom take the difference, and the ensemble-mean vapour the filter produced is kept exactly unless that mean is itself over the mean limit. On by default; the analysis receipt carries `saturation_bound` and each write's receipt `saturation`.
- A radar analysis stops writing an under-dispersed radial-velocity ensemble's innovations into theta and vapour. Where a Vr batch's innovation variance is more than 3 times its ensemble plus observation error variance over all its gates, the batch is withheld from theta and vapour in the columns whose local ratio exceeds 2; theta and vapour there come from the same analysis without that batch, and every other field and column keeps the joint solve. On a storm-scale four-minute cycle the first analysis put 6.44 Mt of vapour into a box where no radar saw a storm; the gate at these thresholds takes 0.79 Mt out, as clean as keeping Vr off theta and vapour everywhere, which starved the cycled storm (footprint rain 0.083 against 0.396). The first analysis's batches measured 4.68 (1 km child) and 3.72 (3 km parent) and every cycled batch at most 1.97, so once the ensemble has cycled nothing is withheld; the cycle of record ran with the gate on. On by default; `--velocity-dispersion-gate RATIO|none` and `--velocity-dispersion-batch-gate RATIO|none` set or switch off either condition, every analysis receipt records the ratios per batch (`velocity_dispersion`), and an A/B bundle recorded before the gate replays without it. Any number of radars can be gated in one analysis, and each solve without a withheld set runs only over the box around the columns it writes and the observations within their reach.
- The prepared DA cycle (`tools/da_cycle_prepared.py`) frees each ensemble forecast before it builds the next, so a domain whose one forecast fits the card no longer runs out of memory building the second. Before its first upload it checks that its largest forecast fits, and refuses a domain that cannot with the sizes it needs and the memory the card has free.
- An HRRR domain tree with two-way nesting (`feedback = 1`) prepares from native HRRR, where it was refused before anything was downloaded. The route's namelists carry the config's `feedback`, `smooth_option` and adaptive time step, and a nest may set its own step limits and growth.
- A high-resolution terrain, land-cover or soil download that loses its connection, is cut short or meets a busy host (HTTP 408, 429 or 5xx) is asked again up to four times, 2, 4, 8 and 16 s apart, resuming the bytes already received. A preparation used to end with a raw `Connection reset by peer` traceback at the first dropped connection. A host that stays unreachable is refused with the file, the host and what to do, and `on_refuse = "fallback-30s"` does not turn that into baseline terrain. A download cut short no longer lands in the cache as a whole file, and preparations sharing one cache download each file once instead of writing into one staging file together.
- A preparation that finds another preparation downloading the same high-resolution terrain, land-cover or soil file waits for as long as that download keeps growing, and stops only when it has not grown for 600 s (`GPUWM_FETCH_LOCK_TIMEOUT_S`), naming the other preparation and the bytes it has. It gave up after 600 s however fast the file was still arriving, so on the 2.28 GB default land-cover file any link slower than about 3.8 MB/s ended parallel preparations on a fresh cache with a refusal.
- `gpuwm fetch` waits out a host that is briefly unavailable. A file that meets HTTP 408, 429 or 5xx, a dropped connection or a transfer cut short is asked for again up to four more times, 2, 4, 8 and 16 s apart (or longer when the host asks for it with Retry-After), for every source: the table sources such as IFS, RRFS, ICON and GDPS, and GFS and HRRR through the Rust fetch. An IFS fetch from the AWS mirror used to stop after three tries over 6 s (2 and 4 s apart) when the mirror answered HTTP 503 for a few seconds, although the same command 20 s later completed. A host that stays down is refused with the file, each host asked, its last answer, how many times it was asked and how long the fetch waited, and the files already downloaded are kept.
- GFS preparations started at the same time from one download no longer break each other: `gpuwm prep --source gfs` without `--source-manifest` writes its input manifest beside its `--output-root` as `<output-root name>.gfs-input-manifest.json`, and `gpuwm go` writes it in its own run folder, instead of both writing one `gfs-input-manifest.json` in the download that a later preparation replaced while an earlier one was still running.
- A config whose domain lies outside a regional source's coverage is refused by `gpuwm go`, `gpuwm run-plan` and `gpuwm prep` before anything is downloaded or decoded, naming the grid the source covers; it used to be refused only at the root forcing stage, after the whole cycle had been fetched and decoded.
- A forecast's map page and its row in My forecasts say where its preparation is while it prepares: files and bytes downloaded (or which ERA5 time is being read), the step it is on and, for a nested run, which grid's start state it is building, "compiling GPU kernels", how long the step has been going, and for a forecast that starts beside its preparation, how many boundary times are ready. Before, the line read "Preparing the grid · forecast hour 0 of 15" and did not change until the first model step.
- A forecast with the MYJ boundary layer and its Eta surface layer (`bl_pbl_physics = 2`, `sf_sfclay_physics = 2`) is priced with the two turbulence columns and 17 surface fields it holds, so `gpuwm check`, `gpuwm domain`, `gpuwm run-plan --estimate` and the fit check before a run stop admitting a grid that does not fit the card; the estimate left out 0.87 GiB at 1792x1024x55.
- A nest that starts later than the forecast no longer holds two copies of itself on the card when it starts. It was built at the start of the run and built again from the analysis at its own start time while the first build was still held, so a tree that fitted the card could run out of memory the moment the nest began. On a 240 x 216 3 km HRRR tree with a 180 x 180 1 km nest starting an hour in, the card's peak fell from 4,899 to 4,337 MiB, and every history file is unchanged.
- A forecast's map page and its row in My forecasts say where its preparation is while it prepares: files and bytes downloaded (or which ERA5 time is being read), the step it is on and, for a nested run, which grid's start state it is building, "compiling GPU kernels", how long that step has been going (between steps, how long the stage has), and for a forecast that starts beside its preparation, how many boundary times are ready. `gpuwm go` also prints the reading of the starting data and each grid's start state as steps in its terminal. Before, the line read "Preparing the grid · forecast hour 0 of 15" and did not change until the first model step.
- `[fetch] transport = "auto"` and `gpuwm go --transport auto` are read as the default they spell out, not as a host named auto: the plan no longer says the fetch pins host auto, and the run reuses the download of the same config without the key instead of downloading the cycle again into a new folder. `--transport auto` over a `[fetch] transport` that names a host lets the fetch choose the host again. A remote review of a plan whose `run_options.transport` pins another host downloads into its own folder on the node, where the second review's fetch was refused for a folder that already held a different request. An HRRR file asked for again through the Rust fetch counts its asks against the four it will make (`asking again (1 of 4)`), where it said 1 of 5 and stopped after the fourth.
- A forecast from a source whose preparation runs as its own program (HRRR pressure levels, RAP, RRFS, ICON, IFS, GEM, GEFS, GDAS, AIFS and the other staged sources) now shows the step its preparation is on and, when the forecast starts beside its preparation, how many boundary times are ready, on its map page and in My forecasts, whether it was started from the page, `gpuwm run-plan` or `gpuwm go`. Before, such a run showed "Preparing the grid" and nothing else from the end of the download to the first model step. A step's time now reads right after the step, before "compiling GPU kernels".
- `gpuwm go --transport auto` over a `[fetch] transport` that names a host says so in the plan (`the fetch walks the host ladder, from --transport auto, over [fetch] transport = 's3'`), where it said nothing about the host. `gpuwm go --transport auto` on a `[case_data]` config or with `--prepared-root`, and a run plan's `run_options.transport = "auto"` beside its `prepared_root`, are taken instead of refused as a host that nothing would ask, and a remote review's expected download names the host the plan asks rather than the table's.
- A plain HRRR or GFS forecast (`source = "hrrr"` or `"gfs"`) now shows its preparation's steps on its map page and in My forecasts as they happen: the child domains' start state and the hierarchy files of a nested HRRR run, and the boundary times a GFS run writes, with "boundary times 4 of 16 ready" while its forecast steps beside them. `gpuwm go`, which runs those preparations as stages, read a stage's output only once the stage had ended, and the GFS stage kept its steps to its own log, so these runs showed "Preparing the grid" until the preparation was over. A failed stage's last lines, on the page and in the terminal, now show its refusal instead of the preparation's raw step records, and `gpuwm go --explain` says each step of a preparation running beside its forecast once.
- A wrfout with no `Times` variable, such as one cut or post-processed by another tool, is labelled with the valid time in its file name (`wrfout_d01_2025-03-16_00_00_00` or `wrfout_d01_2025-03-16_00:00:00`); the renderer labelled it with the forecast's start instead, so a 00Z hour-12 frame read `Valid 03/15 12Z F000`. A file with no time in its name either is refused, naming the file.
- A wrfout cut or converted to NetCDF-4 by `ncks`, whose `Times` is then stored as NetCDF-4 characters or as a NetCDF-4 string, is rendered; it was refused with `Times decoded 19 records, expected 1`, or with an unsupported datatype when `Times` was a string.
- With a positive boundary-layer cadence (`bldt` above 0), the memory estimate charges the retained boundary-layer output only to YSU (`bl_pbl_physics = 1`), the one scheme that keeps it; MYJ, MYNN and Shin-Hong runs were charged about ten grid arrays they never hold, so a run that fitted the card at the margin could be refused.
- The pinned host store of a tiled run leaves an eighth of the machine free beside it on a machine or memory limit under 64 GiB, where it held back a fixed 8 GiB: a container under about 8.5 GiB refused every store, and one under about 20 GiB could refuse a store the planner had sized for it. Machines of 64 GiB or more keep the 8 GiB reserve. Its refusal names the container's memory limit only when that limit set the figure it quotes.
- A GFS, ERA5 or mapped-source preparation with `--preprocess-backend cuda` whose domains alone are too big for the card is refused within seconds of reading the config, before its inputs are decoded, where it was refused only after the decode and the static fields (501 s at 1792 x 1024 x 55 on a 24 GB card); `auto` moves such a preparation to the CPU at the same point. The full price, read once the inputs are decoded, still decides every other case.
- `gpuwm go` says a preparation whose estimate exceeds the card's free memory "may prepare on the CPU", where it said it "prepares on the CPU": the preparation decides from its own price when it starts, and one priced under the card's free memory still prepares on the card.
- IFS and AIFS open data (`ecmwf-open-data`, `aifs`) and GEM (`gem-gdps`) downloads know how long the publishers keep a cycle: ECMWF's own server about 72 h and MSC Datamart about 696 h (29 days). An ECMWF cycle older than 72 h now downloads straight from the AWS mirror instead of first asking ECMWF's server, which no longer has it, and `--cycle latest` looks back 72 h for ECMWF and 696 h for GEM, where it looked back 48 h.
- The GFS NOMADS subset download and the HRRR byte-range download retry the way every other download does: a file the host says is missing (HTTP 404) ends the download at once, where the GFS subset asked for it four more times over about 50 s, and a dropped connection or a busy host is asked again up to four times, 2, 4, 8 and 16 s apart.
- A single-domain HRRR forecast (`source = "hrrr"` with one grid) shows its preparation's steps on its map page and in My forecasts: the root static fields, the start state, "Boundary times k of m" and the companion WRF files, where it showed only the stage. `gpuwm go` on a GFS start prints each preparation step in the terminal under the prepare stage, where it printed only the heartbeat.
- A global model's forecast, such as Arwen Global's, prepares its static fields: the static-field builder takes a `rows` grid, explicit latitude rows on an evenly spaced longitude ring, which is how a global model's Gaussian grid is described. Every global statics build was refused with `grid spec JSON: unknown variant`, because the builder knew only the lambert, mercator and polar projections.

## 2.7.7 (2026-09-25)

Fixed:

- A single-domain run from a configuration file wrote every forecast frame and then stopped with `RealCaseRunSummary.__init__() got an unexpected keyword argument 'moisture_floor_receipts'` before drawing a picture, whichever door started it, the desktop GUI included. It finishes and renders again. Nested runs were not affected. The defect was in 2.7.5 and 2.7.6.
- A GFS 0.25 degree start refused every object of a cycle whose soil moisture decoded 0.00038 above 1.0 on a saturated cell. NCEP packs that field in steps of 0.001 from a reference that can leave 1.0 off the grid, and the bound check accepted a tenth of a step. One packing step past the bound is now clamped to 1.0; a wider excursion still refuses.
- `gpuwm domain`'s check refused typhoon and cyclone boxes from ERA5 that the preparation runs, because ERA5 leaves SST missing on every cell about one fifth land or more and small islands with open water all round were read as open-water holes. A missing SST on a cell with any land fraction is coastal; a hole over open water still refuses.

## 2.7.6 (2026-09-22)

New:

**Speed and numerics, measured on an RTX 5070 Ti**

- Omega is diagnosed one column per thread in WRF's own operation order: a whole step is 5.1 to 7.0 percent cheaper, 1.07, 2.02 and 5.33 ms per step at 250x200x49, 320x256x49 and 480x384x49 under Morrison. The summation order changed with it: one Omega call moves by at most 6.0e-7 of its maximum, the dynamic fields by 1.7e-4 to 7.2e-4 of theirs after 60 steps, and of the moisture fields only Morrison's cloud droplet number (2.5e-6 of its maximum). Proposed and measured by weiserhase in issue #5.

**Modern radiation on every route**

- `--source hrrr` defaults to `thompson-mp8-ysu-mm5-noah-rte-rrtmgp-v1`, with RTE+RRTMGP in place of the legacy RRTMG pair, so every route, door and catalog case shares one radiation default. The legacy suite stays selectable on every door as `--physics-profile thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`. A proposal that types the engine through `ra_rrtmg_variant` composes instead of being refused.
- It buys domain: `gpuwm domain --source hrrr --ladder 12-3 --hours 6 --card 16gb` fits 268x216 at 12 km with a 536x432 nest at 3 km (13.76 GiB peak forecast envelope) against 184x146 and 368x288 (13.74 GiB) on the legacy suite, 2.15 times the root cells; both engines still bound a longwave column at 128 radiation layers.

**The nowcast front door**

- The fine nest is reachable from the nowcast door: `--nest-half-width-km` (or `--nest-nx`/`--nest-ny`) turns it on, and `--nest-ratio`, `--nest-members`, `--nest-i-parent-start`, `--nest-j-parent-start`, `--nest-history-interval-s` and `--nest-acknowledge` are forwarded to the cycle driver. The receipt's `sizing.nested_free_forecast` reports what was asked for.
- The receipt says where the forecast is: `outputs.forecast_frames` names the frame directory and which legs are the free forecast, and the nowcast pages show the `gpuwm render --engine rust` line that draws one. Each frame is a one-level composite snapshot; full fields come from the prepared forecast (`tools/da_cycle_prepared.py`).
- The fine nest's forecast can be drawn: the cycle writes `wrfout_legNN_<trajectory>_dNN.nc` beside the child's `.npz`, and `gpuwm render --engine rust` draws its composite reflectivity into a `d02-1km` folder.

**Rendering**

- `gpuwm render --section-top-km N` sets how tall a vertical cut is drawn, 1 to 40 km, on every render route; every cross section had been fitted to 14 km. Saying nothing keeps 14 km, and `render-summary.json` records `section_tops_km`.
- `gpuwm render` draws twelve column products from a wrfout: the 0, -10 and -20 C isotherm heights (metres above sea level, blank where a column never gets that cold), the supercooled liquid water path over the whole column and the 0 to 3 and 3 to 6 km layers (g m-2 on the model's own layer mass), each hydrometeor's column maximum (g kg-1, one fixed 0.01 to 5 bar) and `simulated_ir_satellite`, whose stored field nothing had written.

**Preparation**

- A run whose terrain the configured vertical coordinate cannot order derives one that can instead of stopping: `etac` becomes the largest value WRF's cubic hybrid supports for the highest ground any domain, or a following nest's corridor, can touch; preparation prints the choice, the receipt carries the numbers, every domain takes it, and terrain no positive `etac` orders is still refused.
- `tools/build_stretched_eta_ladder.py --dt` with `--price-band` reports a ladder's vertical Courant number and the step that would hold it: 200 m layers under the 15 s step chosen for 680 m ones ran the limiter at 1.41 on 606 cells and halved a storm's peak updraft from 33.9 to 18.5 m/s. The DA cycle's preflight names the updraft at which they start being limited. Both read the full-level layer a parcel crosses, not the midpoint spacing: on the shipped 49-level ladder the thinnest convective layer is 630.30 m, not 614.90 m, and its Courant number at 15 s under 33.9 m/s 0.8068, not 0.8270.

**Downscaling**

- `gpuwm downscale` states the LES regime at the door: a child at or below 250 m spacing that inherits its parent's ladder, runs a 1-D boundary-layer scheme with no 3-D closure, or mixes vertically by no route (`bl_pbl_physics = 0` with `km_opt` 1 or 4) is told so with the ways out: `--child-levels N,STRETCH`, `km_opt = 3` or `km_opt = 2` with `bl_pbl_physics = 0` in `--child-config`, and `--child-surface-from`.

Fixed:

**Resume and run directories**

- `gpuwm resume` reads the configuration out of the run directory it was given: as typed, with a hidden `.toml`, against `--outdir`, then from the configuration the run recorded (`child.toml`, `experiment.toml` or `captured-config-<run id>.toml`); `--explain` says which, and a refusal names every path tried. A path that cannot be classified (a link loop, a permission wall, a gone mount) names its errno and the way out.
- `gpuwm resume` refuses a downscaled child's run directory by name: a finished child is given the `gpuwm render ... --series` line for its frames, and one that stopped inside its forecast is sent back to `gpuwm downscale`.
- The prepared-tree runner imports the `tilestream` that shipped with it from any directory, so `python -m gpuwm.prepared_domain_tree_forecast ...` no longer dies with `ImportError: cannot import name 'ValidatedStreamedRestart'` beside an older checkout.

**Initialization and forcing**

- An ERA5 window sized for a 16 GiB card initializes and runs instead of refusing with "specific humidity must be finite in [0, 1)": the check judges the mapped field against the interpolation operator's envelope and floors every admitted value at WRF's qv_min, and a source small enough to window no longer stops preparation with an `AttributeError`.
- `gpuwm check` admits the forcing `gpuwm run` runs: ERA5's descending pressure levels are accepted on both doors.
- `--source hrrr-prs` starts a run with the hydrometeor masses the `wrfprs` file carries (cloud water, cloud ice, rain, snow, graupel), mapped bilinearly; the route began dry. Vertical velocity still starts at zero and the boundaries omit the five.
- `gpuwm run --wrfinput` and `run --met-em` write their run report again instead of failing at the end with `AttributeError: 'WrfTreeInputs' object has no attribute 'physics_profile_assertion'`. The wrfbdy/wrfinput seam refusal names `USE_THETA_M` and what it compared, and says WRF 3.7 to 3.9.1.1 wrote a dry `T_BXS` where 4.0 onward writes a moist one.
- A specified domain forced with an aerosol boundary table keeps its outermost row on that table. Under `mp_physics = 28` the `nwfa` and `nifa` boundary row was never put back: on a 2.7 km domain over 81 levels one corner grew from 6.94e8 kg-1 to 5.13e10 kg-1 in an hour, a factor of 74, until the health gate stopped the forecast; the same hour now holds it at 6.96e8 kg-1.
- An `mp_physics = 28` cold start with analysed condensate closes cloud droplet, rain and ice number the scheme's way, counted under `hydrometeor_initialization.cold_start_moment_closure` in the prepared cache; the orphan state it replaces read 145.38 dBZ between steps, 62.89 dBZ as prepared.
- `--soil-source` pointed at the run's WPS directory recovers the soil column when the producing table declares more layers than that cycle stacked, naming the met_em and the layer authority chosen. A layer that cannot be converted is refused with what it converted, pointing at `source_quantity`/`source_units` and `source_layer_bounds_m`.
- The prepared single-domain and domain-tree doors offer `icon-global`, with the same fourteen physics profiles as `icon-eu`.
- The standalone rw-wps bundle carries `gdt101_remap` and `rw_fetch`, so `rw-wps --source icon-global` runs on a bundle install; a build that omits a bridge is refused naming it.
- `docs/public/CONFIGURATION.md` documents `ra_lw_physics` and `ra_sw_physics`, the two per-domain override keys it accepted and never listed; the page promised 61 keys and printed 59.

**Nowcast replay**

- `python -m tools.da_nowcast run --window-end <ISO>` is about that hour: the volumes, the lag, the echo census and the storm motion that sites the domain come from the window asked for, not the clock (one archived hour's 15 volumes and motion 8.27 m/s toward 166.7 became 18 volumes and 15.14 m/s toward 146.5). Only `--window-end latest` reads the clock, the receipts record `clock` and `listing_window`, and a malformed `--window-end` is refused before any download.

**Data assimilation**

- A reflectivity or hydrometeor analysis runs under Thompson (`mp_physics` 8 and 28): a cell given mass and no number moment is repaired the scheme's own way instead of refused (one cycle had refused 1,834 cells), and the reflectivity operator diagnoses the graupel number itself instead of raising "requires the same-call classic graupel number shadow".
- The positivity policy covers every number concentration and volume the prognostic contract carries (`nwfa`, `nifa`, Milbrandt-Yau's hail number, WDM6's CCN number, P3's rime mass and volume); a filter increment had driven one to -1.005e8 per kilogram and the next leg refused the state. A field the policy has no opinion about is refused at the merge; the moment conditioning reads the running scheme's condensate threshold (1e-12 under Thompson, 1e-14 under Morrison), rescales a number only from an active background and declines a result above the health gate's ceiling.
- The hot start's insertion and the filter's increment are bounded as one analysis; each alone was admissible while their sum put water vapour at -1.02e-4 kg/kg. The correction carried down to a nested child is bounded against the child's own background, which had received -1.014e-5 kg/kg on a boundary row.
- The 1 km child runs on every cycling leg and is corrected by its parent's analysis instead of being rebuilt from it; on the free legs alone it lived 90 to 165 seconds. An analysis that changed nothing leaves the child bitwise alone.
- A forecast leg that begins part way through a run makes the radiation call its land surface needs, so a cycle under a both-streams profile no longer dies at its first surface call with "GLW has no producer and Noah is about to consume it"; a nest born on a later leg counts its steps from its birth.
- A cycling analysis leg is a restart: soil, surface, accumulators and held physics tendencies carry across each leg through the checkpoint set `gpuwm run --restart` reads, so two 60 s legs equal one 120 s run byte for byte; a 2.7.5 ensemble generation is refused by name.
- `rw_nexrad verify` measures each pack array against its own declared dtype, so a pack written by `decode --censor-flags` verifies and `--clear-air-from-censor` keeps every radar it was given.

**Observation times and heights**

- A radar volume is dated by its own radials: each cut and the volume carry `start_time` and `end_time` beside the header's `valid_time` (one real volume: header 12:02:36Z, last radial 12:09:14Z), and the nowcast survey and the cycle admit a volume only once its last radial is collected.
- A surface report carries the instant it was taken and serves one valid time: `rw_asos decode` writes `gpuwm-obs.asos-surface.v2` with `observation_time` on every report, and one 12:54 report is no longer written under both 12:00 and 13:00 with a 3600 s window.
- A cloudy satellite pixel is gridded beneath its retrieved cloud top, not at the ground under the satellite's line of sight: a 9 km top over 35 N 97 W moves 9,724 m, and on one GOES-19 CONUS scan 1,933,447 pixels moved a median 6,165 m. A pixel with no retrieved top stays at the ground and is counted.

**The HRRR route and its configuration doors**

- Every door that publishes an HRRR configuration (edits, `domain-fit` and `domain-tiles` copies, `cyclone-setup`, catalog cases, research workspaces, cycling directories and retained drafts) writes the whole file set its route reads (`route_companions` in the result); they wrote only the TOML and `namelist.wps`, so the run refused every such edit. A per-domain change to a tree-wide setting (`mp_physics`, `sf_sfclay_physics`, `bldt`, `diff_6th_opt`, `isfflx`) is refused with "Select All domains to change them", a configuration whose suite the namelists cannot state is refused naming `moist_cq`, and a nest added through the domain editor inherits its parent's radiation cadence.
- A configuration that spells its radiation through the aggregate `ra_physics` selector, or carries the live cumulus interval with cumulus off, matches the profile it names on every door: 9 of 22 profiles had been refused on the namelist route with "selected physics differs from profile". An HRRR replay of the nowcast door's default profile completes domain to cycle in 114 s of stage time on an RTX 4090.
- The nowcast front door and `gpuwm domain` offer the same physics suites; the warm-rain suite had been refused at one as invalid. A shortwave-only suite prepares on the HRRR background too, and `gpuwm domain` writes a namelist set the importer reads back for the New Tiedtke, ArWen boundary-layer closure, prognostic-TKE and 20CRv3 suites.
- A preparation that fails puts its reason in the file it points at: `python -m gpuwm.source_cli` printed `Details: <path>` and left that file empty.

**Sizing and memory**

- `gpuwm domain`, `gpuwm check` and `gpuwm run-plan --estimate` price a plan on one device and print one number, naming the card priced (`device_profile`, `device_total_bytes`). On a declared 16 GiB card the 12/3 km ladder was sized at 13.73 GiB and refused by its own check at 14.72 GiB (exit 4); it now emits and checks at 13.74 GiB, domains 184x146 and 368x288 rather than 204x162 and 408x320, about 19 percent fewer cells.
- A machine whose card can be read but whose kernels cannot be compiled prices the card it reads and says so, instead of quoting the reference card: one RTX 4090 read 1,353,931,428 bytes from `gpuwm check --json` and 1,205,295,780 from `gpuwm run-plan --estimate`.

**Downscaling**

- A downscaled child that blows up says what, where and how: the carriers that went non-finite, the cell and model second, the last w_max and CFL health checks, and the command that draws the frames on disk; the refusal had been `offline child became non-finite at step N` alone. `report.json` carries the same, a gone reading as `null` beside `w_max_state`, never `NaN`.
- A downscaled child that does not finish keeps the pictures it drew: `DID-NOT-FINISH.txt` atop the picture folder says where the forecast stopped, why, and which frames were written, and `report.json` carries `result` FAIL, the reason and a `products` block (`status` `KEPT`, `pictures_on_disk`, the banner's path). A child composed with `--child-config` is recognised by what the run wrote, not by a `child.toml` beside it.
- `gpuwm downscale` runs a child whose parent used legacy RRTMG with `o3input = 2` instead of refusing it for a missing parent ozone field: the child evaluates the packaged ozone climatology on its own grid, `report.json` names the routing under `child_ozone_routing`, and `o3input = 0` keeps the wrapper's own profile; two 10-minute runs differing only in that field end at most 0.0086 K apart.
- `gpuwm downscale --child-levels` runs on the default preprocess backend instead of dying on its first boundary frame with a `TypeError`.

**Rendering**

- `gpuwm render` and `gpuwm downscale` draw the products they can: a `mesh:` or `meshdiff:` term, an `xsec:` term without `--section`, a `var:` product no store carries and any slug the catalog refuses are dropped before the renderer launches, named with the reason in `render-summary.json`; `--products composite_reflectivity,mesh:cell_area` had exited 1 with no pictures, and the `snow` preset's `var:SNOW` had failed a thirteen-frame child render after 143 pictures. A request left with nothing to draw is refused by name.
- A failed child render says what the renderer said, in the refusal and `report.json`, and counts the pictures on disk; a picture tree that cannot be read is reported as unreadable with the error, not as empty, and so is an availability listing.
- The terminal preset picker no longer calls a product undrawable from the renderer build's fileless import plan, which called sixteen of the `snow` preset's twenty-one products undrawable on a run that then drew 143 pictures of them.
- A vertical cut's fill is drawn on a colour bar fitted to the air it holds; the bar started at zero for every all-positive field, so a three-kilometre cut of air temperature in kelvin used 30 of its 447 rungs, 6.7 percent, and read as one dark red. It now uses 60.4 percent, a fourteen-kilometre cut 88.6 from 28.6. `render-summary.json` records each cut's `lo`, `hi` and rule under `section_fills` (eight rows, `additional_section_fills` counting the rest); a renderer build predating that line is refused at the handshake.
- `xsec:QCLOUD=0.01,0.1/wa` is one product on its own 0.01 to 0.1 g kg-1 bar on every frame, a no-signal frame included (`rule=named` in `render-summary.json`); the splitter had cut the list at its last comma, refused the command line and ignored the list. `xsec:QCLOUD~log` prints 0.1, 1, 10 g kg-1 at its decades under a (log scale) label, not -1, 0, 1 beside g kg-1 under a log10 prefix.
- `render-georef.json` carries a transform for a regional panel whose map a post-render pass pushed past the image's edge, clipped to the surviving pixels within two pixels of drawn markers; only a rectangle with no surviving pixel is withheld.
- The 2 m temperature plate reads at its hot end: the fixed -60 to 120 F ramp drew everything at or above 100 F in neutral ink; those three anchors are now a pink band (on one field's 235,789 cells at or above 100 F, median saturation 0.032 before, 0.925 after), the sixteen anchors at or below 90 F are unchanged byte for byte, and the surface temperature, windowed 2 m and 2 m range products take the repair.
- A mixing ratio is drawn in g kg-1 on every route: a stored kg kg-1 plane through `var:` or `mesh:` is converted on its units attribute, never its name, so the plane at 1.0071096e-3 to 3.8782053e-3 kg kg-1 whose fourteen ticks all read `0` reads 1.0 to 3.9 g kg-1, as a cut of the same air does, where the decade remedy alone had read it 1.2 to 3.8 against `1e-3 kg kg-1`. A generic `var:<stored 2-D variable>` panel whose values still sit below a tick's one decimal after that is drawn on the decade its legend states (2e-7 to 8.3e-7 kg kg-1 reads 200 to 830 against `1e-6 g kg-1`), and a `mesh:<field>:colmax@LO..HI` panel takes the same remedy on the band it was clamped to.

**Reports and the terminal**

- A run's row in the terminal carries the forecast lead it starts from, so an analysis start and a twelve-hour forecast off the same cycle no longer read as one row.
- `gpuwm report` removes a Windows profile reached through a mounted drive (`/mnt/<letter>/Users/<name>`).
- A stale Rust shared library is refused by the name its workspace declares, `gpuwm_preprocess_cpu` rather than `libgpuwm_preprocess_cpu`.

**Source checkouts**

- A checkout's bridge build is checked against its sources, so a `git pull` that moves the Rust half is refused up front, naming the binary, the file that moved past it and the cargo command, instead of a mid-run decode error. `gpuwm doctor` reports the same line; wheels and bundles are untouched.
- `pip install 'gpuwm[dev]'` installs pyyaml, which four release contract tests need, and `gpuwm doctor` judges it by importing `yaml`.
- Three tests in a source checkout pass again: the additive-dissipation switch test compares an explicit `true` against an explicit `false`, the resident-estimate test reads the closure's two added items off the domain estimate's own itemization, and the clock-module audit finds no reflection in the domain-tree builder. No model bytes and nothing the model allocates changed.

**Microphysics**

- Aerosol-aware Thompson (`mp_physics = 28`) agrees with WRF v4.6.1's own Fortran process rate by process rate on saved real-data columns, reflectivity within 0.024 dB; it had been up to 8.7 dB off in 505 to 939 cells per frame.
- Classic Thompson (`mp_physics = 8`) agrees with the same Fortran on the same columns, reflectivity within 0.045 dB; it had been up to 43.9 dB off.
- The repairs are WRF's own rules: condensate at or below 1e-12 kg/kg is zeroed on entry and on exit, a column with no microphysics is left untouched, vapour is floored at 1e-10 kg/kg, ice numbers stay inside their bounds, melting snow falls at its blended speed, collected ice uses the 12.9 micron minimum crystal mass and the rain fallout opens on WRF's `L_qr` gate.
- The `wp08-freeze` column fixture now clears the flat gate: 18 of 22 WRF column fixtures clear it with nothing held out and 19 of 22 as gated.
- `tools/thompson_real_column_parity` runs WRF's Fortran beside the port on the CPU, and a 42-column fixture for each scheme holds the port to it.

**Licence notices**

- `NOTICE` and the Grell-Freitas gamma note no longer describe the gamma as derived from glibc, because it is this project's own work.

Known limits:

- What the fine nest costs at the nowcast door is not stated yet.
- `gpuwm cycle` shows no picture while it runs.
- The local DA nowcast score masks a 9 km rim and scores one member, which the receipt names.
- The desktop's weather map cannot draw ICON global fields; the forecast itself is unaffected.

Detail behind every row: [the 2.7.6 development record](docs/2.7.6-development-record.md).

## 2.7.5 (2026-09-16)

New:

**ICON global forcing**

- `--source icon-global` runs DWD's 13 km global ICON like any other source, from `gpuwm domain`, `fetch`, `prep` and `go`. Aliases `icon`, `icon-13km`, `dwd-icon`. Eighteen pressure levels, surface and near-surface state, the TERRA soil column, sea ice and snow, three-hourly to f180 from 00 and 12 UTC and to f120 from 06 and 18 UTC. Data: Deutscher Wetterdienst, opendata.dwd.de, CC BY 4.0.
- The icosahedral remapper `gdt101_remap` joins the bridge bundle and reads any source on WMO grid template 101. A bundle built before this release does not carry it, and preparation says so by name.
- A new producer on that grid is authority documents and a table row, not code.

**Local DA: a skill number, and continuous cycling**

- Every local DA run scores its own forecast against the MRMS composite, with no flag: neighbourhood FSS at 15, 30, 45 and 60 minutes, at 20, 30 and 40 dBZ, in 9 km and 27 km boxes. Beside every score sits the radar-persistence baseline and the difference, so the number reads as skill over the last scan. The receipt is `nowcast-score.json` beside each window; `--status` shows it.
- A lead that cannot be scored yet stays `pending` with its reason, never a zero. `gpuwm local-da --score PLAN` fills it in later; MRMS scans cache inside the case.
- `gpuwm local-da --continuous WINDOWS` cycles a regional analysis: each window restarts from the last analysis, assimilates its observations, runs the short forecast and renders it. `--status` says where it is, `--stop` is durable, `--launch` resumes without recomputing an analysis.
- Boundary forcing renews itself from the same source cycle when a window runs past it, and a checkpoint can carry the forcing prefix it ran under, so cycling restarts inside a forcing interval.

**Cyclone quick-start**

- `gpuwm cyclone-setup --start-hour N` begins a run at forecast lead N of the selected cycle instead of at its analysis, so a storm the model only develops at a late lead can be forecast now. Every registered source that publishes leads takes it, on its own published ladder; a lead past that cycle's horizon is refused naming the horizon. `--latest-map` takes the same hour, so the centre is clicked on the field the run starts from.
- `gpuwm cyclone-setup --nest-budget-gib GIB` sizes the following nest to a memory budget instead of leaving it at the preset 160x160 whatever the card holds: a 16 GB card was running a nest that fits in under 4 GB. The nest grows square in whole parent cells to the largest layout whose priced tree the budget admits, its movement maximums and search box are derived for the size it reaches, and a budget the card or the preset floor refuses is named with what bound it, what the floor costs here and the way out.

**Offline downscaling and mixed-scheme nesting**

- Offline downscaling changes the child's microphysics scheme the way live nesting does, for the initial state and every boundary frame: WDM6, P3, Milbrandt-Yau, NSSL and every other ported scheme, not NSSL only.
- A nest edge that changes scheme runs off a streamed parent, and two-way feedback runs across it. Both default-on.

**Ensemble and render products**

- `gpuwm enprod` files its panels in the same `<out>/<domain>/<product>/<valid-day>/` tree as every other product, on both engines, with a `render-summary.json`. `--domain dNN` works on the rust engine, `--dpi 300` renders 2400x1800, and `--field` takes any field the engine lists.
- The terminal plot catalog says, per preset, which products this install will not draw and why.

**Remote execution**

- `gpuwm remote sync-outputs` retrieves a run's whole committed output set and resumes if interrupted. `remote list-products` prints what the node's renderer serves, and a selection may name the renderer's `var:`, `xsec:` and `mesh:` families or nothing for the node's default set.
- `--device` names the card a remote run uses. `remote resume` takes the same inputs `start` takes, and a resumed map run stays a map run.
- A launch is a named attempt, so a retry answers with the job already created instead of starting a second forecast on the same card. Long reviews and transfers are no longer cut off: the deadline measures silence.
- `remote status` says which route a job planned and why, and names what could not start or be read. A node too old for what the client sends is named with the version to update to.

**Speed and memory, measured on an RTX 5070 Ti**

- MYNN runs at 8,192 columns per launch, the swept optimum: 15.5 percent cheaper per root cycle than the 16,384 of 2.7.4 on half the workspace. A 1,500-step nested pair went from 2,877 s to 2,390 s with every history frame byte-identical. `GPUWM_MYNN_COLUMN_CHUNK` overrides it.
- The shortwave chain keeps its chunk workspace instead of rebuilding it: about 50 GB of memset per nested radiation event becomes about 0.35 GB.
- The RUC land surface admits a whole call's inputs with one device read instead of 568.
- Forecast receipts split GPU memory into this process, other processes and the card, with the seconds a co-tenant was present, so a run that grew is told from a card that filled up underneath it.

**Elsewhere**

- The terminal's banner and `arwen-tui --help` read the version from the crate, so both say 2.7.5.

Fixed:

**Installed wheels**

- The manylinux wheel's bridge binaries install executable under `pip`. Installs made with `uv` never showed this; Windows was never affected.
- The installed cycle bridge keeps site-packages on its path, so the first leg of a cycle no longer dies with `No module named 'cupy'`.
- `rw_mpas_convert`'s receipt records the converted frame's own digest, not the source history's.

**ERA5 and CDS credentials**

- The terminal's CDS key panel saves a key again. It ran `python -m gpuwm`, which had nothing to run, so a key typed into the product was written nowhere.
- An ERA5 fetch that cannot reach the CDS client says which file it looked for, whether it exists, its encoding, and cdsapi's own error, with the way out. The retired v2 endpoint and `UID:KEY` shape are reported as not configured, and saving a token over them writes the current endpoint.
- The keyless ARCO provider works again: Google's ERA5 Zarr spells its level unit `Hectopascal(hPa)`, which was compared as text against `hPa`, so every request refused before the first chunk. A configuration written for ARCO needs no hand edit: `gpuwm domain` wrote a `.grib` forcing name while the fetch publishes `.nc`.

**Running and stopping**

- A forecast stopped mid-stage exits 130, not 1, so the desktop reads it as stopped and the run can still be downscaled after the app is reopened. Runs stopped under 2.7.4 stay failed.
- A shared GPU is priced against this run's reservation instead of refused for being shared. `--allow-shared-gpu` is accepted and has no effect.

**Moving nests**

- A following nest holds where it is instead of jumping when its search box carries no closed circulation. A cyclone quick forecast started where no cyclone was proposed a 37 parent-cell move toward a ridge at its first cadence and ended there; it now runs the six hours out with the nest where it was placed, and every hold records what it declined on.
- A move that would leave too little of the nest overlapping is made as large as the overlap floor allows instead of ending the run, on the tracked nest and on a containment slide alike, and the receipt names the bound that cut it. Naming a placement outright is still refused, now with the per-axis move that floor implies and the knob that widens it.
- The cyclone quick-start's movement maximum is derived from its overlap floor and from the nest it actually proposes, so it is always a move the nest can make: 6 parent cells at the full layout, 5 on a card that fits down to a 144-cell nest, 3 at the smallest layout the door proposes. A fitted proposal shows both numbers among its reviewed changes.

**Preparation and data**

- A conformant IEEE-packed GRIB2 message decodes instead of being refused, and a stale mapped decode engine is named as a rebuild rather than blamed on the publisher's bytes.
- A following nest no longer refuses its first move when its parent spans the antimeridian. Source pixels are binned at their canonical column, so a crop of the sealed statics corridor equals the statics built directly for that footprint byte for byte; a corridor sealed before this fix is refused at load naming the re-preparation. The stencil averages also reduce in a fixed order, so two builds of the same ground agree to the bit.
- A real-data initialization is no longer refused because the vertical operator undershot a sharp dry slot: vapour taken below zero is floored at WRF's own `qv_min_value` and receipted, for the parent and every spawned child.
- Radial velocity is dealiased by default at every radar door; `--no-dealias` turns it off on the nowcast doors and the radar-grid tools. A coherent fold used to reach the analysis as a plausible wrong wind field.
- A Noah-MP option that reaches no gpuwm code is set with one line saying it changes nothing, rather than refused.
- A spawned child carries its own aerosol receipt across the nest boundary.

**Rendering**

- Frames at forecast hour 1000 and beyond are filed correctly instead of left flat at the render root.
- A research recipe is no longer refused for naming a product the packaged table does not list: the renderer's catalog is the vocabulary.
- `enprod --engine auto` refuses rather than drawing weather fields with matplotlib when the rust engine is missing. A run whose history dropped `OLR`, or a downscale asking for `xsec:` products, is refused at plan review rather than after the run.
- `simulated_ir_satellite` left the general and hurricane presets and carries its reason: no forward radiative-transfer operator exists on the history-import route.
- The terminal workspace's Plot history guide pins `--engine rust` on the timeline it builds. `--series` renders a whole timeline into one renderer store, so windowed products can be differenced across frames (`qpf_6h` is F012 minus F006); the matplotlib engine renders one file at a time, holds no store and carries no windowed product, and `--timeidx` would index inside each file. The pair is refused rather than silently ignored.

**Remote execution**

- A remote forecast started from a configuration already on the node serves its frames, stores, maps and gallery instead of six doors refusing it.
- A node whose card memory could not be measured is priced against its most conservative capacity and launched, not refused. Inputs too large for the staging manifest travel by verified transfer. A saved case naming any registered source has its forcing relocated to the node cache.

Known limits:

- `gpuwm cycle` shows no picture while it runs.
- The standalone rw-wps bundle does not carry `gdt101_remap`; `gpuwm doctor` names the gap.
- The desktop's weather map cannot draw ICON global fields; the forecast itself is unaffected.
- The local DA nowcast score masks a 9 km rim and scores one member, which the receipt names.

Detail behind every row: [the 2.7.5 development record](docs/2.7.5-development-record.md).
## 2.7.4 (2026-09-13)

- Full-world map bounds retain all longitude columns instead of collapsing to one meridian. The desktop regenerates affected normalized map caches, including AIFS and IFS cyclone previews.

- Desktop cyclone selection displays the selected model's pressure/wind preview, with visible loading, errors and retry. Map overlay controls adjust barb size, spacing, line weight and visibility.
- A standalone downscaled run supplies its own archived geometry when used as a parent. The desktop can save and review child physics/settings separately, and failed forecasts show their error directly.
- Offline Morrison-to-NSSL startup repairs bounded interpolation roundoff from valid parent fields and invokes the existing NSSL initializer for missing moments. Seven retained parent boundary frames passed conversion; this is not a forecast-skill validation.
- Known limitation: offline physics transitions remain narrower than live nesting. Mixed offline conversion targets NSSL (18) from WSM6 (6), Thompson (8), Morrison (10), or aerosol-aware Thompson (28). Other mixed transitions and WDM6 offline archives remain unsupported; some configuration reviews can succeed before initialization reports that limitation. Broader shared conversion support is deferred beyond this release.
- Release tooling runs on Python 3.11: the version-bump tool no longer relies on a 3.13-only file API, and the dev extra declares pytest-xdist, which the pre-cut gate and its controls run under.

This corrective release keeps local data assimilation experimental. The ordinary
Desktop interface hides its entry point; advanced users can enable it by starting
the TUI with `--enable-local-da`. Backend capabilities remain available, while
continuous local cycling is outside this release's completion scope.

Remote resume now honors supplied geography, prepared-input, WPS and product
overrides. Remote memory estimates remain visible as advice without blocking the
requested launch. Standalone RW-WPS shares source preparation facts with ArWen
without importing forecast or assimilation orchestration.

Overlapping direct WRF and met_em launches receive separate output attempts.
Supervised workers retain their parent's output ownership, and interrupted
processes release that ownership automatically so unused empty folders can be
used again. Existing forecast files and checkpoint generations are preserved.

Fetch metadata publication tolerates short-lived Windows readers of its prior
manifest or recovery JSON. Persistent write errors remain visible, with the
previous publication preserved.

New:
- WRF cold inputs whose upstream soil layer amounts were interpolated without conversion recover from matching original metgrid layers and their producing Vtable. The native reader verifies the source domain, time and reconstruction, converts before interpolation, preserves consistent liquid state, and records source hashes for restart identity. Discovery is automatic beside the input files; `--soil-source DIR` locates original WPS inputs stored elsewhere. Originals and already physical volume fractions are unchanged.
- Regional assimilation accepts background choices from the shared preparation catalog, with explicit product, source member, provider, boundary cadence and supplied input authorities. One selected cycle and complete forcing window survive review, publication and execution; source membership is separate from regional ensemble size. Existing native and staged preparation chains return verified inputs without starting a forecast, and stale owned preparations recover into new directories before member state is committed.
- Local assimilation automatically discovers native satellite cloud-water-path inputs for each window, preserving explicit files, source intervals, quality and uncertainty. Overlaps contribute one observation per target column; missing feeds remain visible, and frozen inputs survive recovery. Completed cycle reports expose actual accepted analysis batch counts separately from downloaded and source-QC counts.
- `gpuwm cyclone-setup --source SOURCE` authors the 12/3 km moving-nest cyclone from any source the registry and the acquisition routes both carry, not from one model. Every source-derived fact on the emitted document is read from that source's own row: its published cycle hours (so an hourly source takes an hourly cycle and a twice-daily one says which two hours it publishes), its forcing interval, which now prices the tree admission and stamps `interval_seconds` in the emitted `namelist.wps` instead of a constant patched into the rendered text, its coverage window, its recommended physics profile, the model top its certified inventory floors, and its preparation recipe, which for a source that declares one writes its own `.Vtable` beside the configuration. A centre outside the selected source's grid is the one refusal about the storm rather than the computer, and it names the position and the sources that do cover it. `--member` selects an ensemble member in the route's own grammar, carries it into the acquisition request and the map, and the staged chain verifies the member identity of the GRIB messages that arrive before preparation consumes them. Adding another model to this door is a registry row and a route, with no code path of its own. `docs/public/CYCLONE-SETUP.md` documents the document.
- `gpuwm cyclone-setup --list-sources` emits a third document kind, `sources`: one row per planable source carrying its canonical id, its display title, its member ids and default member, its forcing interval, the UTC hours it initializes at, its coverage envelope or `null` for a global source, and how the chain that source runs on delivers the statics a moving nest travels over. The schema string does not move, because the kind and the fields beside it are additions and no existing key changed meaning.
- Every cyclone document says what the run door will do with the following nest it authors. `gpuwm cyclone-setup` always authors a moving nest, and only some of this release's preparation chains can deliver the child-resolution statics such a nest travels over. Which ones is a table in the run door, and the setup door now reads that same table: a `configuration` or `proposal` carries a `follow_statics` block (the chain, the delivery word, whether the nest is integrated as authored, and why not when it is not), every `--list-sources` row carries the same answer, the sentence is written into the configuration file beside the nest it describes and printed once on stderr, and it names the sources whose chain does carry a moving nest. A source that reaches no launch route at all is a different answer and says so: the block carries `launch_refusal`, the sentence `gpuwm go` itself raises for that source, and the note quotes it rather than describing statics, because the nest is not what stops such a configuration and dropping the nest would not start it. Nothing new is refused: the setup is authored, priced and reviewable on every planable source, and what changed is that the limit is stated where the source is chosen instead of at the launch.
- The cyclone centre can be found rather than clicked. `--seed-fields NPZ` takes canonical arrays from the selected source analysis, bound to it by three identity scalars, and locates a candidate centre through one stated chain: the declared sea-level pressure minimum, then the 850 hPa cyclonic relative-vorticity maximum, then a 300/500 hPa warm anomaly. `--advisory-position LAT,LON` bounds that search within `--seed-radius-km` (default 500) and is the final fallback when no diagnostic answers. The document's `seed` block says which rung answered and records every rung that declined and why. An explicit `--point` still wins over all of them, and a candidate centre is a centre, not a cyclone classification or an intensity analysis.
- The following nest authored by `gpuwm cyclone-setup` writes its own `storm-track.d02.csv`, and each independent follower may declare a `[[domain]].follow.track` of its own, refused at load if two of them would write the same file. The track ends with a stated reason when the tracked extremum reaches the parent-domain boundary and an enclosed centre is no longer resolved, for every tracked field and at whichever end of that field is the centre; the stream ends there rather than appending rows from a clipped edge cell, and the run and the nest's movement are unaffected. Missing signal alone stays a gap in the record, not a termination.
- A run-plan intent naming an ensemble source is drivable on the prepared route. It used to be refused for carrying a member set at all, on the reasoning that member selection belonged elsewhere and an intent had no member axis; the route's own grammar supplies the default or selected member and the staged chain verifies the member identity of the bytes that arrive, so the fact the refusal rested on is gone.
- `gpuwm local-da` reviews, publishes and runs a local rapid data assimilation case for one point or one bounded region. One `--dry-run` prices a ladder of rungs against a declared card, preserves the requested rung and reports memory, cadence and wall-time comparisons as advice, says which observation streams exist for that region and time and which do not, and writes one `arwen.local-da-plan.v1` document; `--out` publishes the configuration and `--launch` runs it through the existing regional ensemble cycle owner. `--capabilities` publishes the companion contract, including `refusal_codes` split into the review codes that decide a configuration before anything is written, the launch codes a published case raises before its execution document is opened, and the run codes it raises after, with `refusal_effects` saying per group whether the refusal leaves an `arwen.local-da-execution.v1` document behind and whether `forecast_started` can be true, and `docs/local-da-companion-protocol.md` documents every field of the review, the ladder rows with their prices and verdicts, and all three kinds of refusal.
- A local DA review carries `cadence_overrun`, the cadence owner's own `gpuwm-da.cadence-overrun.v1` record for the selected rung, and each ladder row carries its own; a rung that exceeds its cadence remains runnable and carries a queue-lag projection.
- A source whose registry row declares a composed preparation profile and a local input contract runs from bytes already on disk. Review resolves the input root from `data_dir`, `--data-dir` or `[fetch].source_root`, checks the declared inputs, their valid times and their bytes before any stage starts, and the staged chain prepares from the same `prep-arguments.json` handoff a downloaded source publishes. Being downloadable stops being the condition for reaching a prepared chain, and `gpuwm domain` emits a `[fetch]` table for such a source instead of stating a gap in the file's header.
- The ensemble member chosen in a domain intent survives into the run. `gpuwm domain --member` records it in `[fetch]`, the acquisition route's own declared member set validates it, review reports which member was selected and on what basis, and the staged preparation verifies the selected tree's grammar identity, cycle, member, expected inventory and file bytes before preparation reads it. The blanket refusal of an ensemble row at plan review is retired; an invalid member and a statistical product are still refused.
- One preparation dispatch table owns each runner's required arguments, command, chain and local input contract, and the prep door and the run planner both read it. A registry row naming a runner that already exists becomes intent-drivable with no new branch in the planner.
- GDAS acquires and prepares through the same structured handoff as the table routes. The container fetch publishes its ordered inputs and the in-band surface role its packaged composition declares, so `gpuwm go` on a GDAS configuration reaches the staged chain instead of being refused as analysis-only.
- ERA5 boundary cadence takes any positive whole number of hours that divides the requested window and satisfies the selected product's clock, instead of 1, 3 or 6. A cadence coarser than six hours warns once per review and runs. The EDA product keeps its three-hourly clock, as a multiple.
- The two WRF-input worker doors take `--products` and `--render-dir`, and the four per-step progress flags (`--progress-format`, `--progress-output`, `--progress-every`, `--no-frame-markers`), and carry every one of them through the supervised re-launch that does the run. `gpuwm run` itself does not register these flags yet, so today they are given by running `python -m gpuwm.wrfinput_forecast` or `python -m gpuwm.metem_forecast` directly; through `gpuwm run` the defaults apply, which draw the default catalog and publish frame-ready markers.

Fixed:
- Windows table-driven acquisition and preparation handoffs preserve full cache identities and upstream filenames beyond the legacy path limit. Completed downloads can publish and reuse their verification receipts in deep output folders, including ICON-EU field files and ensemble member trees.
- WRF input soil water explicitly labelled as layer mass or equivalent water depth is converted to volume fraction by the native reader using the file's own soil-layer thicknesses. Already volumetric fields retain their exact values, liquid water uses its own declared units, and the import receipt records the conversion. This does not infer units from large values or repair soil whose producer has already discarded its source quantity and geometry.
- WDM6 checks its actual substep count before converting it to a signed counter. Unrepresentable counts or invalid density/thickness stop the affected column and report a failed call through the existing health path; normal schedules are unchanged and no iteration cap is applied.
- Physics maturity documentation states that implemented options have different evidence coverage. A maturity label alone does not establish an independent Fortran oracle or a matched forecast; existing per-option measurements remain unchanged.
- Verification documentation distinguishes actual per-routine oracle coverage and historical comparisons from current forecast skill. It preserves the initial-state failures and measured fine-grid differences without claiming an unmeasured cause or a calibrated chaos allowance.
- Native `met_em` preparation reports estimated VRAM and host-store overruns as advice, retaining the requested domain, levels and physics. Structural input checks and actual preparation errors remain enforced; both warnings are visible before preparation and recorded with its memory estimates.
- Windows output receipts rehash current bytes before reusing a completed digest. Native ChangeTime can remain equal across a rapid same-size edit with restored modification time; trusting that collision returned an old digest for changed contents. Linux revision-based reuse remains unchanged.
- MYNN validation distributes its unchanged scan over a cached device-sized grid, avoiding the fixed eight-block regression on large arrays. Windows publication replacement controls now reach the intended descriptor-recovery boundary and exclude their own injected aside copy from the recovery assertion.
- Named table-route fetches reuse complete byte-verified local inventories after remote retention expires. Missing or changed objects still go through normal recovery before preparation receives a complete handoff.
- Local cycling selects a source cycle whose requested initial and boundary frames are actually available, while preserving the chosen initialization time. An older source cycle uses the corresponding later leads, and fetch duration no longer includes the initial lead twice.
- Vertical colorbar units stay within the legend column and cannot cover right-aligned timestamps. Longer labels use the complete vertical placement; map and tick positions are preserved.
- Classic Thompson retains rain mass and number concentrations across cloud adjustment until rain evaporation actually refreshes them, and preserves the positive-condensation decision that suppresses same-call rain evaporation. This corrects fallout after strong cloud evaporation; warm collision heating and canonical tables remain unchanged. Checkpoints name the corrected rain-density history.
- MYNN ordinary mixing length now shares the rounded initialization routine, removing compiler-dependent duplication and preserving the existing driver error limits. Checkpoints identify the changed PBL algorithm. OLR tests now follow the existing checkpoint-only diagnostic owner so held radiation output survives between calls.
- Native map colorbars show the displayed field's units, including temperature, reflectivity and wind. Headerless vertical legends rotate complete longer labels into the side margin without covering numeric ticks; data values, color levels, map geometry and overlays are preserved.
- MYNN batches large native validation groups into one read-only scan while retaining per-field flags and immediate error reporting. Scientific arithmetic, output fields and workspace sizes are unchanged; a small warmed comparison preserved state and output bytes with lower validation overhead.
- Asynchronous forecast output computes exact file digests after native publication while the next step can reuse its staging buffers. Final receipts verify those writer-bound records without rereading unchanged payloads. Replaced files cannot inherit a writer identity, and earlier outputs without a fresh record retain the full stable-file check.
- Local preparation publishes and verifies the reviewed input order through completion, and recovers a damaged owned list while preserving prior files. GFS accepts positive uniform subsets of its actual published leads through acquisition and both preparation readers, while retaining missing-lead and time-bracketing checks. Preset and soil documentation no longer claim retired Milbrandt-Yau radiation or nested RUC restrictions.
- Analysis recovery checks that the retained receipt, member roster, method and policies agree with their bound run before publishing any remaining member. An omitted binding cannot hide an existing enclosing cycle authority. Contradictory records preserve the existing analyses and cannot describe a smaller ensemble or a different method as the original decision.
- Finite assimilation retains an immutable analysis decision and method receipt before publishing any member. Recovery verifies the entire member set, current inputs and policies, then completes pending publication without repeating assimilation or replacing committed bytes. The local CLI also returns structured error JSON when a preparation stage fails.
- Morrison freezing remains finite when an intermediate exceeds FP32 range, while all competing sinks share the same donor budget. Cloud freezing retains its log-space moments, final evaporation and sublimation return water to vapor, and in-range number moments are preserved. These default corrections carry a new checkpoint algorithm identity; earlier checkpoints cannot silently continue under changed physics. Tiny column tests establish finite transfers and water budgets, while WRF trajectory agreement remains unresolved.
- Cloud-water-path uncertainty inflation follows the source numeric quality flags: 256 is thick cloud and 512 is thin cloud. The two labels were previously reversed.
- Local assimilation keeps the requested domain, resolution, members, cadence and cycle count even when estimates exceed declared memory or wall-time targets. Live free-memory comparisons also warn without refusing. Execution records measured cycle lag, and actual failures retain prior outputs and checkpoints for recovery. The finite wall-time target is explicitly advisory, with no automatic downgrade or override hurdle.
- Domain authoring carries explicit ERA5 product, provider, member and boundary cadence through the saved configuration and acquisition command. Reanalysis remains the default; a member alone cannot change the product. A latest request binds one cycle and the selected member before authoring, and a local-input intent retains its printed staging directory.
- Configuration review rejects malformed time-window values before any source-specific planner. The local-input check and execution review inspect the same directory and byte identities, and an authority replacement cannot reuse an unverified cache entry by retaining its size and timestamp.
- Registry diagnostics recognize an unknown or unimplemented option before considering route permissions. General route explanations derive from the route's own declarations; an ordinary prepared route cannot acquire the benchmark's immutable-comparison explanation. Noah-MP throughput advice no longer excludes a component override, while its surface-coupling constraints remain enforced.
- WDM6 rain sedimentation preserves column water and rain number across every interface and records the actual surface fallout. The previous update could discard much of a thin-layer rain column at ordinary physics timesteps by limiting an arrival against rain already removed from the cell above. Its time schedule now covers changing fall speeds and initially dry receiving layers through the scheme's declared maximum speed and each layer's depth. The corrections are on by default and carry a new checkpoint algorithm identity; prior checkpoints cannot silently continue with changed sedimentation.
- Local assimilation resource advice includes simultaneous card, host-memory and cadence limits once each, instead of allowing two cadence reasons or a later source horizon to hide a live memory limit. The companion protocol describes the same ordering.
- Ensemble member legs publish their actual terminal state through the normal history writer, including short forecasts resumed between history times. Intermediate cadence is preserved, cadence-aligned endpoints are written once, and terminal reflectivity uses the output-time microphysics handshake. Local assimilation no longer reports a complete execution when its forecast frames or rendered products are missing.
- The contract check before export refuses interrupted tests, incomplete reports and failed archive or transfer operations instead of recording a passing candidate from partial terminal output. It checks a fresh structured result for every selected file, narrows the shadow-provenance exception to the documented assertion, and names its candidate-contract scope in the receipt; built packages and installed forecast workflows still require their own checks.
- Table-driven acquisition retries temporary connection failures and service errors across its declared mirrors, with at most three attempts per endpoint and bounded waits. Missing objects are not repeatedly requested. Starting the same forecast again reuses completed files after verifying their payload and SHA-256, including files that finished after an earlier transfer failed; only missing or damaged files are fetched again.
- A downloaded object is validated before replacing its destination. An incomplete or invalid response cannot replace a previously complete file, and an unfinished acquisition publishes no preparation manifest.
- Forecast retries keep prior frames, pictures and receipt addresses unchanged in their original generation. Output inventories reject concurrent file replacement, skipped renders do not publish old pictures, and older completions cannot demote the latest published run. Preparation reuse preserves authority bytes and timestamps; changed settings select a new generation automatically.
- Metgrid checkpoints bind a verified scientific preparation identity independently of available-memory observations. Raw receipt integrity and scientific, source, cache and runtime identities remain checked.
- Standard Noah-MP cold-start input records are classified from the input registry instead of rejecting the selected land surface scheme. Its snow, canopy and soil initialization survives restoration; placeholder state and recomputed diagnostics cannot overwrite initialized fields. Explicit stepped state and meaningful untagged prognostic state are refused before GPU selection because this door does not restore the full WRF restart inventory. The import receipt records each additional surface field's disposition.
- NetCDF reader selection and release staging require character records as well as numeric arrays, so an older decoder cannot reach WRF time-record reads. Default resolution can select a compatible reader already available later in the search order.
- A per-domain follower whose evaluation cadence the watched parent's history stash cannot serve is refused when the configuration loads, in the same words, instead of when the relocation runners were being built. The way out is unchanged and named in the refusal: set that follower's `cadence_seconds` to a whole multiple of the watched parent's `history_interval_s`, or set that domain's `history_interval_s` to a value the cadence divides into. A configuration that ran before still runs; what moved is where the refusal arrives, and the refusal names the table the knob is actually in, `[[domain]].follow` for a per-domain follower and `[relocation]` for a whole-run one, in the same words at the load door and at the runner build. The same check is what both doors call, so neither can admit what the other refuses, and it applies only to the trackers that read the stash: a pressure tracker reduces from the live column and is no longer refused for a cadence that has nothing to do with it.
- `member` in a `[fetch]` block is validated against the named source's own acquisition route rather than refused as an option that applies to one reanalysis only. A source with no ensemble route still refuses it, naming that, and a member the route's grammar does not have is refused at configuration review with the grammar named.
- One launcher serves every desktop package. `tools/release/launch_arwen.py` selects the runtime a package carries when it carries one, and takes `--python` (or `ARWEN_PYTHON`) and verifies the installed engine against `ENGINE-PYTHON.json` when it does not, on Windows and on Linux, with the folder preferences, the terminal restore after a killed controller, the notices and the `--verify-launcher` record the same code on both routes. Until now this file was the bundled-runtime variant alone, and each cut shipped a second launcher maintained outside the repository, so a repair committed here reached the packages a release late or not at all.
- A package built for another computer, a package naming a platform this launcher has no programs for, a folder holding no readable `ARWEN-DESKTOP.json`, a package that never finished assembly, and a manifest that does not record the controller, the application, the weather program, the map library, the C runtime or the runtime document are each refused by name at start, with the way out, instead of failing later on a file nothing verified. A `--companion` or `--open-companion` handed through to the launcher is refused with the start that does work rather than the breakage alone. `docs/desktop-package-launcher.md` records what the launcher decides and what the cut tooling should stop lifting.
- The launcher's help text names the engine version out of the package manifest rather than a version written into the file, and `Start ArWen.cmd`, `Start ArWen Terminal.cmd` and the new `Start ArWen.sh` and `Start ArWen Terminal.sh` each prefer the runtime the package carries, otherwise pass `--python` or `ARWEN_PYTHON` through, and name ArWen and the way out when the computer has no Python at all instead of leaving the shell to report a missing command.
- A physics combination with no recorded pace row is priced and run instead of refused. The pace owner answers `conservative` callers from the slowest rate on record, marks the estimate substituted and unmeasured, names the missing rung and the row that stood in for it in the basis, and the local DA review warns once and stays ready. The conservative answer is a caller's choice and the local DA door is the caller that makes it; the forecast preflight and the run planner still take the unpriced answer and stay silent, so outside this door the remedy is still opt-in.
- An analysis carrying only attributed point observations is admitted on the batches it was handed rather than on a flag set before anybody knew. `extra_observations` becomes a statement the caller may leave unset, `analysis_sources` is the one function the config refusal and the analysis call both ask, and an analysis with no source and no batch is refused before a checkpoint is opened.
- Every `gpuwm local-da` option says what it does in `--help` and in the CLI reference.
- A published local DA case names its WPS namelist `experiment.namelist.wps`, which is the name the preparation owner reads beside the experiment. A bare `namelist.wps` made every real launch stop at its first stage; one roster now serves the publisher and the saved-plan reader.
- A hydrometeor perturbation counts the depleted moment pairs it CREATED, not the ones the background arrived with. A cold start carrying hydrometeor mass with no number concentration was refused for a condition the perturbation neither made nor touched; those cells are left alone and reported in the draw receipt as `depleted_pairs_in_background` beside `negative_points_in_background`.
- `--obs-table` on an installation without the shared neutral-table package is refused with both ways out named, instead of printing a bare import failure: install `gpuwm-global` to read neutral tables, or review and run on the radar and surface routes without the flag.
- A windowed radar document is assimilated on the accelerator. Staging a batch onto the device rebuilt it field by field without its reach window, so a `gpuwm-obs.radar-grid.v1` document carrying a window arrived at the filter declaring none while carrying window-sized arrays, and the analysis stopped on a shape it had produced itself. The staging now carries every field the batch declares, so the next one added cannot go missing the same way. The CPU arm was never affected.
- The local DA observation tests no longer take a module-level `importorskip`, which emptied the whole file wherever the shared observation-table package is absent. The dependent half is its own file with a stated skip reason, its items are still collected, and a new guard fails any local DA test file that empties itself at collection.
- A `gpuwm-da.cadence-overrun.v1` record says whether its verdict weighed a timing or a price. The cadence owner documented its cycle cost as a measured wall time while its one caller in the tree hands it a reviewed estimate, so a reader could take a projection for a stopwatch reading. `check_overrun` takes `cost_basis`, defaulting to `measured`, the record carries it, and a local DA review states `estimated` because that is what a ladder rung is. The verdict itself is unchanged.
- `gpuwm run` on a domain tree takes its `[tiles]` admission before the fetch, from the same call the plan review prices with, instead of deciding the tree's roads at build time from the run's own memory ledger. A tree `gpuwm check` admits is no longer refused minutes later, after authority, fetch, manifest and prepare have all been paid for; the review and both run doors now weigh one envelope against one budget and the decision the user was shown is the decision the run executes. The build pass consumes that decision rather than taking a second one, and marks the moving subtree on the live tree whether the road was decided there or handed in, so the two paths leave the tree in the same state.
- `gpuwm run` on a single domain takes that same admission, from the same function the plan review and the prepared forecast call, priced from the configuration alone and taken before the case is fetched. It used to price this arm from a second estimate with the schedule's retained forcing interval count folded in and no device profile, and to take it only after the input catalog was built and every forcing snapshot decoded: MEASURED up to 188,362,088 bytes apart on a 12 km root, a band in which `gpuwm check` admitted the domain resident and the run then refused it with the download already spent. The review surfaces, `gpuwm go`'s single-domain forecast and `gpuwm run` now weigh one envelope against one budget, and a domain carrying its own `tiles = {...}` table is judged on that table at every one of them.
- `gpuwm go`'s prepared single-domain forecast judges the domain on the `tiles = {...}` table that governs it, the way the plan review, the run plan and `gpuwm run` already did. A single domain carrying its own table was judged there on the tree-wide `[tiles]` table instead, so one configuration had two answers and the second arrived only after the prepared cache had been restored.
- A tree whose domains run adaptive clocks is admitted on the acoustic reach it will actually walk. The admission a door takes before the fetch resolves each domain's largest map factor from the projection and the latitude span, the way the live domain resolves it from its own `MAPFAC_U`/`MAPFAC_V`, instead of assuming a unit factor: a unit factor prices a tile halo narrower than the domain's per-step dependency radius, which does not fail the run, it makes the tile interiors wrong and the run faster. Plan review, the door and the build pass now hand the executor the same tiling.
- `gpuwm domain`, `gpuwm check` and the starter template judge a `[tiles]` admission on the card they measured. A caller that probed the device and handed its profile beside a planner machine built without one had the shared admission priced against the reference card instead, which on a measured RTX 3080 at 6.54 GiB free refused a domain that fits by 606,218,976 bytes.
- A domain carrying its own `tiles = {...}` table is judged on that table by the plan review, the run plan's execution block and the root's streamed envelope, not on the tree-wide `[tiles]`. A root whose own table asked for streaming under a tree-wide `mode = "off"` was reported as needing no decision at all by the review while every run door decided it.
- `gpuwm.render.sweep_abandoned_scratch` requires the ownership token the door minted and refuses the plain working-store prefix, which matches every store in a delivery including a concurrent render's live one. The minted token is eight hex digits rather than a whole uuid, because it rides on every working-store path.
- Plan review asks the renderer for its product catalog once per process instead of spawning `rw_wrfbatch --list-products` for every question, keyed on the renderer binary itself, so a restaged renderer is asked afresh.
- A render stage that exits nonzero clears the working stores it left in the delivery's `.render-scratch` sibling and says how many and where; they held no product and nothing later read them. The door names its own stage's stores with a token it mints and hands down, and sweeps only stores carrying that token, so a concurrent render into the same case keeps the store it is working in whenever it opened it. The delivered tree is still kept as the evidence of what failed.
- `gpuwm cyclone-setup` quotes the budget `--tiles auto` was actually judged against when it names auto as the way to keep the requested coverage, rather than the whole-process allowance the auto walk had already withheld a moving nest's rebuild from.
- A forecast stage that drew the first frame and then failed clears the working stores that early render left beside the delivery, the way the render stage already does, and says how many and where. It mints its own ownership token and hands it down, so it sweeps only its own stores and a concurrent render into the same case keeps the store it is working in.
- The plan review says which allocation it priced for a domain whose own `tiles = {...}` table streams it under a tree-wide `mode = "off"`; that sentence used to be omitted entirely for such a configuration, so the report quoted streamed figures with nothing saying they were streamed. A nested tree's sentence prices the road it describes from the same admission the verdict printed beside it is asked of, rather than from an estimate of its own.
- `gpuwm fetch --source gdas --cadence N` no longer truncates its window in silence. A cadence that does not divide `--hours` dropped the window's final frame, fetched successfully and left the run bounded by a shorter series; it is now refused, naming the cadences the published ladder serves for that window. The `[fetch]` table and the flag plan the window through one function, so a table accepted at config load cannot be refused at the fetch.
- `area`, `point` and `radius_km` on a local-input configuration were read, accepted and then had nothing to apply to. Each is refused by name, with where the crop actually happens. `source_root` on a source `gpuwm fetch` downloads is refused the same way.
- The refusal past the GDAS publication limit states its remedy at the default width and its mechanism under `--explain` again, instead of one unlayered sentence.
- A `[fetch]` table spelling a source alias reached plan review without the local-input admission its registry id carries, and the staged chain then looked for an acquisition route that does not exist. Both lookups resolve the registry id first.
- A local-input `[fetch]` table missing `source`, `cycle` or `hours` failed inside review with a dictionary key error. The plan is refused and the missing lines are named.
- `gpuwm fetch` on a container source whose composition binds no in-band surface role refused after the whole download, inside the manifest publisher. The same check runs at argument validation, before any bytes move.
- Plan review re-read and re-hashed every packaged profile's composition document, for every registry row, on every configuration load. Each authority is byte-verified on every use; the parsed composition is reused only for the same verified bytes.
- The shipped data page and the `gpuwm domain --source` help described retired refusals: that GDAS has no initialization front door and reports `"runnable": false`, and that a source `gpuwm fetch` cannot download gets the acquisition step named instead of a `[fetch]` table. Both now describe the staged mapped chain and the local-input table the doors actually take.
- A `[fetch]` table spelling its cadence as a decimal or a quoted number was accepted at configuration load, skipped the window check entirely and was refused later at the fetch. The spelling is refused where the window is planned, so the two doors agree for every spelling a table can carry.
- `gpuwm domain` wrote a `cadence` key into the `[fetch]` table of a source whose own `gpuwm fetch` refuses one, so a configuration that passed `gpuwm check` was refused at stage 1 of the run it had just been checked for. Whether a fetch takes a cadence at all is one question now, asked by the flag, by the `[fetch]` table at configuration load and by the front door that writes such a table; a source prepared from bytes already on disk still carries the spacing its staging check needs.
- A `[fetch]` table spelling its `hours` as a decimal or a quoted number was accepted at configuration load and then skipped the window check, the way a cadence spelling did: `hours = 9.0` with `cadence = 2` loaded and named exactly the silently truncated window that check exists to prevent. Both operands of that check are refused where the window is planned, and so is a length `gpuwm fetch` will not take. The length is checked ahead of every source's own window planner, so no planner reads a spelling it cannot count with.
- An ERA5 `[fetch]` table whose window is not a whole multiple of its cadence is refused at configuration load rather than at the retrieval, through the function that plans the retrieval's valid times. Widening the accepted cadences had widened the set of tables that loaded clean and were refused later. The `--era5-product` help stops naming a single fixed cadence for the EDA product.
- `gpuwm fetch --cadence` given a spacing the GFS 0.25-degree product is not published on states its remedy at the default width and its mechanism under `--explain`, instead of one unlayered sentence. Taking any whole hour on the flag is what made that refusal reachable at the door, where argparse used to answer for every source at once.
- `out` on a local-input configuration was read, accepted and applied to nothing, beside the three crop keys already refused by name. It names where a download would write and nothing is downloaded, so it is refused with `source_root` named as the key that does carry the input directory, and `gpuwm domain` stops emitting it for such a source.
- The six composition suites reach the single-domain route: Milbrandt-Yau with New Tiedtke, WDM6 with Grell-Freitas, the SASE closure-supplied suite, and the three large-eddy closures (TKE 1.5-order, 3D Smagorinsky, constant K). Each is an implemented option's only named front door, and each was offered on the domain-tree route alone, so on a single domain nothing named them for any source. They are offered on every source that names any suite, in one order, derived from the route's own declaration rather than written per source.
- `gpuwm sim` takes `--tiles JSON` and `--stream-init auto|resident|store` for a single-domain prepared run, validated by the runner's own parser before anything is claimed, and passes `--physics-profile` through to the prepared domain-tree runner, which asserts the named suite on every hash-bound domain and records the assertion in its report.
- `gpuwm doctor` reports what the ERA5 route can actually do on this computer: whether `cdsapi` imports and a credential file or the `CDSAPI_URL`/`CDSAPI_KEY` pair is present for `--retrieve`, whether the keyless ARCO reader is staged and ABI-compatible, and that the request document is always available. It used to print that there was no transport to report on.

- The three Noah-MP expert suites are offered on every source the single-domain runner supports that names any suite at all, not on one. A route that declares any expert list is exhaustive, so on an ERA5 single domain a plan naming one was told the template was off-route instead of being handed the acknowledgement advisory the option exists to raise.
- Land surface `off`, surface layer `off` and the analytic clear-sky radiation proxy stop being published as registry-`unreachable` while a hand-written config ran all three. Each is `component-override` on the routes that declare it, and each keeps its degradation as a plan-review warning naming what turning the component off costs.
- The prepared single-domain route admits the component overrides its own declaration lists: cumulus, microphysics and turbulence vary freely, and land surface, PBL, radiation and surface layer resolve to any option the route lists for them. The 2026-07-31 ruling removed that runner's profile whitelist and the route declaration was never widened to match, so plan review refused configurations the runner runs, including a component map that merely restated the base template's own composition. Noah-MP component selection follows its state and surface-coupling constraints, with expert acknowledgements advisory. The analytic clear-sky radiation proxy is written in the hash-bound experiment config or run on the domain-tree route, which declares `ra_lw_physics` and `ra_sw_physics` as expert selectors.
- A route refusal names the option it turns away, what the route declares instead, and the way to the option: the registered templates on this route and source that carry it, or the runner that admits per-domain overrides. The blanket "runner accepts only its immutable template_id" described no runner, and "runner route does not allow this component option to vary per domain" named none of the three.
- The single-domain physics menu is every fixed-template route's own declaration rather than one route's replay table, so a suite a route declares is offered by that route's own menu.
- `--cycle latest` resolves on ERA5 and on every table route, from the route's declared publication delay when no object can be probed, with the basis printed in the same line as the resolved cycle. Both used to refuse the word outright.
- `--hours 0` fetches the analysis alone on every source, not on one. Only a forecast manifest still needs two forcing times, and that refusal names what a single frame leaves empty.
- `--cadence` is checked against each source's own published ladder instead of a fixed 1|3|6 list: ERA5 HRES takes any positive whole-hour cadence, ERA5 EDA any positive multiple of its native 3 h interval, a table route the spacings its row names, and every source refuses a duration the cadence does not divide, naming the final time that would otherwise be dropped.
- An ERA5 request is written one day at a time, so a window whose cadence does not divide 24, or that ends part way through a day, cannot retrieve times that were never asked for. CDS crosses every date with every clock time in one request; the retrieval and the request document now carry the same per-day schedule instead of computing two.
- A named cycle is checked for publication on the endpoint and for the member actually requested, before any byte moves, and one complete endpoint settles it. A control member or a lagging mirror could authorise downloading a different member from a still-incomplete pinned endpoint.
- `gpuwm sim` composes and validates its runner command before claiming a run directory, so a refusal no longer leaves an allocated directory behind.
- A `[fetch]` hints table and the command line are validated by one function, so a window, cadence, member, crop or start hour the CLI would refuse is refused identically at configuration load, and a partial hint table still validates.
- Every source the single-domain runner supports appears in the capability inventory with a row of its own, instead of being absent or inheriting another source's. A packaged source with nothing measured on it reports the suites its route carries for every packaged profile, with the limitation saying no source-specific verification is claimed.
- A route that serves a source declares which templates it offers there. The single-domain route served three sources it declared nothing for, so the launcher's per-source offer and the registry could not be compared for them at all. The native route declares the eleven suites the packaged pressure-level profile of the same producer declares. A source with nothing measured on it is priced from the suites that route names for every source it has measured, not emptied: an empty list is published as "this source reaches no named suite", and it took the six composition suites and the three Noah-MP expert suites away from the two packaged profiles that had no measurement of their own. The caller-supplied composition row, whose physics the caller states, is the one source that names none.
- A route refusal that points at an expert template also names its optional acknowledgement, without turning that advisory into a second refusal.
- `ra_physics` is no longer offered as a per-domain parameter on the domain-tree route. It is the pre-split spelling of the radiation pair, which every radiation option states, so overriding it per domain could only restate or contradict the option already selected, and plan review refused eighty-two configurations the run door admits. The route publishes the omission with the component and the way to the value.
- `gpuwm tui` reports the release the engine beside it runs: the terminal crate declares 2.7.4.
- `--cycle latest` on an analysis source resolves the same date the calendar publishes as its boundary. The window came off the newest published analysis twice, once in the calendar and once in the resolver, so a 240-hour ERA5 request started ten days before the boundary shown beside it, with nothing said; the terminal's Latest button ran that start. The boundary the calendar publishes moves with it: a window that is not a whole number of cycle intervals now ends on the last published frame rather than one interval past it, which costs up to one interval of recency and never asks for a frame that is not there.
- A component override the route turns away is refused with that route's own reason for that component. Both fixed-template routes printed "runs its registered templates unchanged", which describes the sealed benchmark route alone: the prepared single-domain route printed it while admitting a cumulus, microphysics, turbulence or PBL override on the same plan, and pointed away from the reason the exclusion has. The reason is declared per component, so a refusal about land surface no longer carries two sentences about what an analytic radiation proxy leaves out.
- A per-domain setting a route defers is refused with the component it belongs to and the way to the value, instead of "runner route does not accept this per-domain setting" alone. A setting no route defers names where the ones it does take per domain are declared.
- A template the route declares its runner cannot build a product for is refused at plan review, naming the runner and the reason, instead of failing as `unsupported physics profile` after preparation has been paid for.
- A component option refused for the SOURCE it would run on (microphysics `off` on a native HRRR column) is reported under its own code, because a `RunConfig` carries no source identity and the per-domain authority cannot mirror that one refusal.
- A domain tree whose domains name different template ids that resolve the same values runs. The refusal compared template LABELS and named no breakage; what it stood in for is checked on resolved values now, and names the two values and the loader that cannot express them.
- A fetch refusal about flags says it is about the command line rather than about a `[fetch]` table the user never wrote, and the one-frame HRRR window names a way out the door that refused can take: the acquisition door, which admits a one-frame window, and `--forecast-hours` where that is the window flag the caller has. It named an internal keyword, and then a flag pair belonging to a door that cannot raise it.
- A `gpuwm run --wrfinput` or `gpuwm run --met-em` forecast draws itself. Both doors arm the shared early render when products are named, run the same finalize render stage the chain runs, and publish `latest-run.txt` once a picture is on disk; with no product spec the default catalog is drawn at the end and `none` is the opt-out. An install that cannot draw says so at plan review with `gpuwm setup` as the remedy, instead of finishing a whole forecast and then declining to draw it.
- A `gpuwm run --wrfinput` or `gpuwm run --met-em` run claims its output directory at plan review, before a card is selected or locked. A directory that already holds a run is refused in one sentence naming it and the two ways out, where it used to fail as `[Errno 17] File exists` inside the worker child after a GPU had been reserved for it.
- `--restart` into the run's own output directory works on both WRF-input doors. The resumed attempt writes into a new `segment-NNN` generation, leaving prior frames, input records and receipts together and unchanged, even when an older checkpoint replays an already published time. The checkpoint reference keeps its original location and the forecast and renderer use the new generation. A changed preparation request gets a separate owned generation, retaining the previous input tree and its authority documents.
- The terminal ArWen opens keeps the terminal description the caller stands in. `TERM` and `COLORTERM` are given to a package's children only when the caller's terminal names neither, so a 256-colour terminal is no longer told it is truecolor and a package started from the application still gets a terminal the controller can draw in. `NO_COLOR` reaches them too: it is the user's instruction and ArWen's terminal implements it, and a launcher that removed it made a package print colour a user had asked it not to. `Start ArWen.sh` and `Start ArWen Terminal.sh` are committed executable, so the archive carries their mode out of the tree rather than out of a list of file names in the cut tooling.
- An imported research-proposal catalog is read for the nest tree its domains declare instead of for what they are called. The case importer resolves each preset's tree from the `parent_id` each domain row carries, so any id vocabulary and any list order import, and a case whose recipe is a subtree keeps its own ids rather than having to be renumbered to `d01`, `d02`, `d03` first. The converted tier is the same geometry as before: domains are ordered parent before child, `grid_id` follows that depth, and each nest ratio is the child's spacing divided by its resolved parent's. The refusal that admitted only the literal sequential naming is retired; what remains is four named refusals, each quoting the offending domain's own id: a `parent_id` this preset does not declare, a closed `parent_id` chain, more than one outermost domain, and two nests declared on one parent. The per-domain import issues name the catalog's own domain id instead of the row's position.
- Two nests on one parent are still refused when a catalog tier is imported, and the refusal now says why: a tier's geometry is one nest ladder (`root_dx_km` plus `nest_ratios`), which no single chain spells, and the way out is one recipe per branch under `source_domain_recipes` or building the sibling nests with `gpuwm domain`. Until now that tree was refused for its id sequence, which said nothing about what actually cannot be emitted.
- An interrupted lifecycle episode's leftovers are swept out of publication. A domain that retires and re-arms files its history frames under `dNN/episode-NNN/`, and the start-of-run sweep that moves orphan temporaries and unfinished final frames into `.quarantine` globbed one level deep, so an episode's leftovers stayed put under published names for the next reader to open as though they were products. Discovery is now one recursive helper in the module that writes that layout, used for all three of the sweep's patterns, and it skips dot-prefixed directories, which is the rule the rendered-picture reader already applies and is what keeps the sweep from re-sweeping its own quarantine on the next run. The helper also says what IS a frame, from the name and before anything is opened: the readiness receipts published at `ready/<frame>.json` repeat a frame's basename, and a recursive walk that handed them to the reader filed a valid receipt under `.quarantine` labelled an incomplete product, destroying the one positive signal a live consumer is told it may trust. A frame that is complete, at the run root or in an episode folder, is left alone. The go chain's render feed reads the same tree with its own one-level glob and is unchanged by this.
- `gpuwm downscale --tiles` and `--child-levels` work beside `--child-config` instead of being refused as a pairing. The flag is resolved against the file the caller supplied, by one function each that plan review and the runner both call: the file decides when the flag is absent, the flag decides when the file declares no `[tiles]` table and no `eta_levels`, an agreement is a no-op, and a disagreement is one warning naming both values with the flag winning as the later and more specific statement. Only the ABSENCE of the table is silence: a `[tiles]` table that spells `mode = "off"` out loud has named a mode, so `--tiles on` or `--tiles auto` beside it is a disagreement and earns that warning rather than replacing the statement without a word. A `[tiles]` block that pins a tiling under the mode that owns it -- `tile_nx`, `tile_ny`, `nbuffers` or `halo` beside `mode = "on"` -- keeps that tiling when the flag agrees and loses it with the mode when the flag names one that cannot carry it, and the same one warning names the keys it dropped and both ways of keeping them (drop the flag, or delete the keys), because the flag is what imposed the mode and the file cannot be the statement asked to change. The warning is said once per invocation however many doors resolve the same configuration. The file on disk is never rewritten, so its `[tiles]` knobs, its `[output]` history selection and its comments survive. `downscale-plan.json` carries `child_levels_override`, `effective_nz` and `effective_eta_levels` beside `child_config`, `report.json` carries the same three beside `child_config_sha256`, and the plan and the price are taken on the resolved ladder and the resolved mode rather than on the file's. A malformed `N[,STRETCH]` and a level count given without a stretch still refuse, at CLI argument validation and before `--out` is reserved.
- A downscale child config that sets `nested = true`, or leaves `specified = false`, is refused at plan review on `gpuwm downscale --dry-run` instead of after the run has started or after the whole parent archive has been interpolated, and the refusal names the breakage: the offline route forces the domain from an external LBC mirror, so the run's root must take specified boundaries or it has no Davies forcing and drifts freely off the archive. The rule was spelled twice, in two different wordings, in two files and fired at neither plan-review door; it is now one function every door calls, and it says that sub-nests of a downscale child are declared as domains in an experiment TOML.
- A surface snapshot's every 2-D field now reaches the wrfout beside it. `gpuwm.io.surface_wrfout` published a curated list of names and dropped the rest without a word, so `tilestream.bigdomain`'s `COSZEN` carrier was lost from every tile-streamed frame while the list carried a `Q2` row neither lane produces. The shape is the contract now: an entry that is a 2-D array of the frame's own grid is published under its own key, the writer types it and the render door draws it (`--products var:<name>`), and only the two keys whose wrfout spelling differs (`WMAX` to `W_UP_MAX`, `WMIN` to `W_DN_MAX`) still need a row. A lane that grows a carrier gets it in the wrfout with no table edit.
- `write_surface_wrfout` says what did not reach the file instead of discarding it quietly. It returns a `SurfaceWrfoutWrite`, which IS the written path (it extends `Path`, so `.name`, `.stat()` and `open()` work on it and a caller that wants only the file needs no edit) and additionally carries `passed_through` and a `skipped` map of key to reason. The DA-cycle lane prints the `skipped` map at write time, so a 3-D array in a file that declares itself surface-only is named where the run can see it rather than found later as a missing panel. The `*_units` siblings and the bookkeeping scalars are in neither list: they are not fields. The one refusal is unchanged and now covers every 2-D field rather than only the listed ones, naming a field placed on a grid it was not computed on. `T` and `MU` are placed before the shape rule because they are the file's structure rather than a stand-in, so a same-named surface plane is reported instead of replacing the mass coordinate.
- A remote job is no longer unmade by the computer correcting its own clock. `bound_manifest` and the completion evidence ordered two wall-clock stamps written by two processes seconds apart -- the job record's `created_at` and the wrapper's `started_at` against the runner's own `started_at_utc` -- so a machine that steps CLOCK_REALTIME back a second or two while resynchronising turned a live run into "Remote run manifest does not match this job's saved plan, process and output identity" and a terminal receipt, with no way out. An ordering read inside the window a host may correct itself in is no longer evidence and no longer refuses; outside it, a manifest that predates its own job is refused as before, and what binds a manifest to one job stays the per-job token its process carries, which no clock can move. The measured settlement case times the wrapper's cleanup ladder on CLOCK_MONOTONIC at both ends, the rule the code under test already keeps, instead of subtracting two wall-clock readings taken six seconds apart and reporting a four second cleanup that had in fact run its whole course.
- `gpuwm remote` resolves its OpenSSH client for the platform it was asked about rather than the one it is running on. `ssh_executable` took the environment and the platform as arguments, then handed the PATH search to `shutil.which`, which reads the path separator and `PATHEXT` from the interpreter's own platform: a Windows PATH separated by `;` whose entries carry an extension was read as one POSIX entry and found nothing, so the two halves of one choice answered for two different platforms. The search now honours both arguments, and an environment carrying no PATH has no PATH to search.
- A typed `gpuwm remote` option is refused for what is wrong with it, before this computer is consulted. `ssh_command` resolved the OpenSSH client between the workspace check and the port check, so on a machine without `ssh` installed `--port 65536`, a relative `--ssh-config` and a relative `--identity` all came back as "OpenSSH client 'ssh' is unavailable; install it and retry", which names neither the mistake nor the way out. Every typed option is now checked first, through one function both the review and the run reach, and the missing client refuses only once the configuration is whole.
- The compositor behind `gpuwm render --pair` no longer sends a reader to an extra that cannot supply what they are missing. `gpuwm/pair_compose.py` said Pillow "ships with the `[render]` extra"; `[render]` is the rust renderer and has never contained Pillow, which arrives with matplotlib instead. The refusal at the door already declined to name an extra, so the two doors disagreed about one fact and the module a reader opens carried the wrong half. It now says where Pillow really comes from and names no extra, and a test sweeps every extra that docstring names against what the extra actually installs, so the claim cannot come back in a different wording.
- The high-resolution static overlay runs on every projection this program builds. `[static.highres] enabled = true` on a Mercator or polar-stereographic domain refused with `unsupported-projection` and stopped the case, on a statement about where the path had been proven rather than about anything it could not do: the overlay resamples through the grid's own map projection and was never specific to Lambert. The gate now rejects only an object that is not a projected grid at all. A projection this program cannot build is still refused where it always was, at configuration load, with its own blockers named.
- A coastal domain takes high-resolution land use instead of being refused. The land-cover crosswalk has one open water class and could not tell a lake from the sea, so every domain with ocean in it was refused outright, which in practice meant most coastlines. Open water now becomes WRF ocean category 17 where the domain's own 30-arc-second baseline water field says ocean and inland lake category 21 everywhere else, and the receipt carries both cell counts and names the field that decided. The split is the only behaviour available: the discriminating mask is a required input with no default, both the production door and the bounded geography pilot derive it from the same function, and a caller that has no ocean in its footprint says so with an empty mask and reads the same audit.
- A domain sitting on 180 degrees is described by the longitude range it occupies rather than by a bounding box claiming the whole planet, so the one-degree terrain tile enumerators return the handful of tiles either side of the line and the near-global elevation sources report no overshoot for it. The two blanket refusals of such a domain, at static production and at tile enumeration, are gone. One step remains: the derived mosaic window is still written in the cut -180 to 180 frame, so a domain whose footprint is continued past 180 refuses naming `dateline-window-unbuilt`, with the reason and the way out. That refusal is made on the footprint itself, before a plan is resolved and before a single tile is enumerated or fetched, so no domain downloads a mosaic it is then told cannot be built.
- `gpuwm render --heavy` on a wrfout computes the mixed-layer and most-unstable entraining-parcel grids (`mlecape`, `muecape`, `mlecin`) and the three ECAPE / derived-CAPE ratio pairs, which wrf-core exposes no diagnostic for and which therefore reached no panel at all on this lane while running normally on the GRIB one. The import's own surface planes and isobaric sounding volumes are assembled into the products-side surface/pressure input pair and handed to the shared heavy recipe entry point, so both lanes run one copy of the recipes; only the slugs this hour does not already carry are filled, so wrf-core's own surface-based diagnostics keep their names. A file with no 3-D fields, or missing a surface plane, degrades to a note naming the input and what produces it, and still writes its 2-D fields and its soundings. Levels outside a column (below ground, above the model top) stay unset and are dropped per column rather than invented, and where a split `wrf3d` file carries only lowest-model-level stand-ins for the 2 m and 10 m fields the grids are computed from those with that basis stated once in the import notes.
- `--list-products` gives every excluded heavy (ECAPE) row its own reason instead of one sentence for the whole family, so a permanent exclusion and a gap a re-import fills are no longer indistinguishable. Each row names the grid the catalog looked for, what computes it, and the route that does: the three `*_ecape_native_cape_ratio` pairs divide by the source model's own decoded CAPE plane, which a wrfout does not carry (wrf-core's CAPE is diagnostic, not native), and point at a GRIB surface ingest that does; the parcel grids, the derived-CAPE ratio pairs and the composites each name the volumes and surface planes they are solved from and the `--heavy` import or GRIB heavy lane that produces them.

## 2.7.3 (2026-09-10)

New:
- `downscale-plan.json` carries `child_outline`, the four corner latitude and longitude pairs of the child footprint read from the parent's own grid, beside `parent_domain`, `child_grid_id` and a `streaming` block that says whether the child runs resident or streamed, why, and on what budget. A front end can draw exactly what will run and show the verdict before anything is fetched.
- A downscaled forecast is itself a parent. The child writes a checkpoint set at every `restart_interval_s` inside its window and at its end, under the instant naming `--parent-restart latest` discovers, so `gpuwm downscale CHILD_RUN --parent-domain 2 --parent-restart latest --point ...` derives a grid 3 grandchild from it. The desktop's per-run downscale block carries `parent_domain` and counts the run's own frames and sets, and its downscale request accepts `parent_domain`.
- The finer forecast is drawn on the run's map. Downscale from this run arms the map's area gesture; the box dragged inside the parent becomes the child's centre and its size in whole child cells at the chosen ratio and is outlined at once; Review prices that size on this computer's measured card and the engine's own child outline replaces the box, with the memory verdict and whether the child runs resident or streams said in words. Fit to this GPU stays as the second choice and no point has to be clicked before the window opens. A downscaled run is offered Downscale from this run like any other, so a forecast is downscaled again as many times as wanted.
- Settings lets the user choose where new forecasts and downscaled forecasts are written and where the managed Python, engine, physics tables and geography install, beside the existing data folder, each with Apply and Open folder; the first run of the Windows desktop offers the install location before any download starts. Existing forecasts stay where they are and stay listed; a saved folder that cannot be used at start is named with the reason and the way out and the profile's own folder holds.
- A finished forecast that kept restart checkpoints can be downscaled to a finer child without leaving the desktop. My forecasts offers the child's area, spacing and window, the engine writes `downscale-plan.json` and the child's run receipts, and the child is listed and opened in the run browser like any other forecast, with progress on every browser tick. `gpuwm downscale` stays the command-line route, and a refusal it prints is recorded on the job so the engine's own sentence is what a reviewer reads.
- A forecast configuration can be saved as a named setup and started again at a new time. `gpuwm companion-setups save --config CONFIG --library DIR --name NAME` copies an opened configuration into the desktop's setup library; `gpuwm companion-setups start SETUP.toml --cycle CYCLE --forecast-start-hour N --hours H --out NEW.toml` writes a new configuration with the new clock, cadence and output folder and every other table verbatim, validated by the ordinary configuration parser before it is published. The desktop's Saved setups tab does both, and a refusal is shown as the engine wrote it.
- Milbrandt-Yau (`mp_physics = 9`) radiates its own cloud, ice and snow effective radii under RTE+RRTMGP. The radii are the scheme's own moment ratios evaluated from the transported number concentrations on every radiation call; ice and snow merge into the single RRTMGP ice species by number as Morrison's do. The refusal of mp=9 against the 4/4 pair on the default variant is retired in the config door, the registry constraint and the composition walk. A bare mp=9 configuration with RTE+RRTMGP now validates and runs. The legacy RRTMG arm is unchanged and computes its own radii, so the two arms radiate different cloud radii for this scheme by design. The companion physics door's special summary and "use WRF RRTMG" repair for that refusal are removed with it; every refused draft is summarised and repaired the same way.
- A forecast requested from a map point is sized to a domain rather than to whatever the card holds. The point fit carries a maximum extent per axis and the projection's own polar envelope, and both shrink the layout instead of refusing it, so an ordinary mid-latitude point with streaming on is no longer grown across a pole and then refused. The bound is stated once in the plan summary. An area that was actually drawn to a pole, and a point so close to one that even the minimum layout contains it, are still refused, and each refusal names what moves it.
- Every memory refusal the domain-fitting door raises carries the term that bound it, so a caller can read what refused without parsing the sentence.
- `--card` on `gpuwm domain`, `gpuwm downscale`, `gpuwm cyclone-setup` and the starter template takes any card spelling: a tier (`16gb`), a size written into the name (`10gb`, `"RTX 3080 10GB"`), or a model with a recorded size (`"RTX 3080"`, `rtx3080`, `"5070 Ti"`); `--vram-gib` beside a card is the capacity and the card is a label. Only a spelling that carries no capacity at all is refused, with both ways to give it one. Until now the flag accepted four tier names and refused every real card name.
- `gpuwm check` exits 6 when the run door's own `[tiles]` walk refuses the configured execution plan, with that refusal as its last line, instead of printing the refusal beside a fit verdict and exiting 0; `gpuwm go` on the same file was stopping at the refusal the check had already printed.
- `[tiles] max_redundancy` is a configuration key: a number replaces the tile planner's halo-work limit and `false` lifts it, so the planner's refusal of a domain too small to tile efficiently names a way out a configuration can take.
- WDM6 (`mp_physics = 16`) and aerosol-aware Thompson (`mp_physics = 28`) reach the laterally forced routes and the mixed nest edge. `mp_physics = 28` is admitted on the HRRR-forced routes, a stock-WRF `mp_physics = 28` wrfinput export writes all fourteen members instead of six, the WDM6 and Thompson-aerosol nest edges under and over the other schemes are ratified with their entry closures named in the transition receipt, and the mp=28 aerosol dataset precondition is asked at every door that commits to a forecast, before anything is fetched.
- The desktop's forecast output folder is the user's to choose. The packaged launcher and the Windows bootstrap read `forecast_output_folder` from the desktop preferences and start the controller with `--output` on that folder, and the controller's new `--saved-runs FOLDER`, given once for the profile's own run folder and once for every folder listed in `saved_run_folders`, keeps the forecasts made in each earlier folder listed and openable in My forecasts, so forecasts already made stay where they are and stay visible however often the folder moves. A saved folder that cannot be created at start (its drive absent, its share unreachable) is named with the reason and the way out, and the controller opens on the profile's own folder so the desktop and its Settings still open.
- Milbrandt-Yau (`mp_physics = 9`) reaches every door the other schemes reach: a Milbrandt-Yau domain nests under or over a domain running another scheme, the radar assimilation operator simulates its reflectivity with the scheme's own Z block, an offline child can be cut from a Milbrandt-Yau parent, and the WRF compatibility matrix and the renderer's precipitation catalog carry it. A stated clear-air floor that disagrees with the scheme's own is refused at configuration.

Changed:
- Editing a configuration whose domain encloses a pole is refused at the companion doors and at plan review with the reason and the way out.
- `gpuwm cyclone-setup` documents carry schema `arwen.cyclone-setup.v2` with the new proposal kind and `--accept-fit FIT_ID`; `--out` on a reduced fit exits 0 and writes nothing, and the desktop reads v1 and v2.
- Noah-MP (`sf_surface_physics = 4`) is priced on CUDA-12 installs from readings taken on the RTX 3080 class and on sm_120 with NVRTC 12.9.86; a card with no reading is priced from the recorded ceiling with the basis named at plan review, never refused.
- Radar assimilation refusals land at plan review before any ensemble leg: a cycle that names a scheme the operator cannot simulate, or a clear-air floor the scheme does not carry, is refused when the cycle is planned, with the arm to turn off named. A checkpointed run is asked once per domain, before step 0, whether its physics can be named in a checkpoint.
- The streamed composite reflectivity derives its species from the active scheme's own operator and prefers the scheme's stashed dBZ, so every scheme's composite renders from the product step.

Fixed:
- A nested forecast under `[tiles]` set to `auto`, the cyclone quick-start included, runs resident when the whole configured tree fits the card, instead of being refused by the tile planner's shared process and radiation floor charged before any domain is priced (the reported case: a 12/3 km moving-nest cyclone refused by a 6,978,986,310 byte tile floor against a 6,855,065,600 byte admission budget, while the same tree needs 5,141,378,237 bytes resident under that budget, and ran). The tree road now asks the resident question first, as the single-domain road already did, and consults the planner only where that answer is no. The refusal that remains names what refused and the budget it was weighed against in one sentence and the way out in the next, and no longer tells the user to re-run with `--tiles off`; the tile road's shared floor is quoted only where that floor is what exceeds the budget, a refusal is classified by the refusal that ended the search rather than by the first one it met, and a refusal raised on a tiling rather than on memory keeps the planner's own sentence and points at the geometry instead of at the card. Where `mode = "on"` is what put a fitting tree on the tiled road, the refusal names the table that set it -- the domain's own, or the tree-wide one where that is where the mode comes from -- and says to set it to `auto`. The plan review and the run door now price the admission from one estimate against one device profile, so a tree the review admits is not refused at run start after the download, and the run takes that one admission through to the steppers it builds instead of deciding a second time. An automatic tree that moves a nest withholds that nest's rebuild from the admission budget, weighs the pinned host snapshot the move stages through against the page-lockable allowance on the resident road as well as the tiled one, records both, and names the withholding in any refusal whose budget carries one -- with the way out being the nest, not the card, where the tree fits and only the move does not. The cyclone `--tiles on` door prices the mode it recommends, so it offers `--tiles auto` only where auto admits. The cyclone document carries a `streaming` block saying which road each domain takes.
- A downscaled forecast renders its products. The child's frames are drawn into `png/<domain>/<product>/<valid-day>/` beside them, the first frame while the run is still integrating and the rest when it finishes, through the same render stage, product catalog and default set (`all`) a forecast takes; its event stream carries the `finalize` render stage and the render summary, so a downscaled run's rendered picture count is filled in the run view like a forecast's. `gpuwm downscale --render-products LIST|all|none` chooses the set or keeps only the frames, the derived plan says which, and the desktop's `launch_downscale` request takes `render_products`, defaulting to the session's own plot selection. A requested set that drew nothing is a refusal naming the render command, and so is a render stage that exits nonzero: the forecast keeps its `PASS` and `report.json` gains a `products` block naming the command that draws the saved frames. A product slug the catalog does not carry is refused at plan review, on the dry run too; a computer with no staged renderer is refused when a run is asked, before a frame is opened, with `--render-products none` named as the way out. A child that does not pass publishes no picture: what the early render drew is withdrawn, and every exit waits for that render rather than leaving it running. Until now a completed child held frames, checkpoints and receipts and no picture at all.
- The downscaled child's `[tiles]` decision is taken before preprocessing, on a card measured before the process has allocated anything, from the same estimator the plan review priced with. A child the review admitted is no longer refused at run start by the tile planner's fixed-cost sentence computed against the memory the run had already spent (the reported case: a 138x138x49 child admitted at 2.90 GiB against 7.32 GiB free, then refused at "no tile fits in 3.49 GiB" after the archive had been interpolated). A card that is genuinely too small refuses at plan review and at run start before the child's own state was built on the device, with the measured figure and the way out. With `[tiles]` set to `auto`, the plan's `memory.fits` is judged on the same budget as its `streaming` decision, so the two blocks cannot disagree about one child; with `[tiles]` set to `on` the memory block keeps the fit ceiling and the streaming block says the child streams. A child clock that is not a whole number of steps (`output_interval_s`, `restart_interval_s`) is refused at plan review as well, in the runner's own words, with the whole multiple of `dt` to choose named.
- `restart_interval_s` in a downscaled child's configuration is honoured instead of accepted and dropped. The child used to write one final checkpoint under a name no discovery recognised, so no downscaled run could be downscaled although the door's own refusal named that setting as the remedy. With `--parent-domain N` and `--parent-restart latest` the child's physics is bound from that domain's own member of the newest checkpoint set, not from the set's root member.
- `gpuwm downscale --child-size` with `--auto-vram` prices the given extent on the measured local card and reports the fit, with the `gpu_sizing` receipt filled as on the fitted route; the refusal that required fitted sizing beside measurement is retired, and the same holds for `--child-config`. Only `--card` and `--vram-gib` stay exclusive with measuring, as two declarations of one budget. The desktop companion's `launch_downscale` request measures the card by default whenever no capacity is declared, an explicit size or configuration included, and an explicit size beside a declared capacity is priced on that capacity instead of refused.
- A `--child-size` that reaches past the parent's interior around the point shrinks to the largest centered extent the parent holds there and the plan's warnings say what was asked, what it became and why; only a child that cannot exist at all is refused. An extent that is not a whole number of refinement cells is rounded down to one and said so.
- Refusals whose only reason was a missing measurement, recording or validation are gone: the estimator prices from its most conservative recorded basis and says so at plan review, a pairing no oracle has paired runs with its mapping named and a warning, and every mixed nest edge between ported schemes resolves to its ratified closure.
- A nest whose microphysics differs from its parent's runs without an opt-in key. `nest_microphysics_transition` left at its default resolves the edge to the closure that pair takes and the transition receipt records the requested and the effective policy; naming the id of a different edge is refused with the id the pair takes. Until now the default refused every mixed edge for want of the key.
- An aerosol-aware Thompson (`mp_physics = 28`) forecast reaches the end of a run on both routes: the shared scratch arena of a domain tree hands the scheme's two integer entry slots out as same-width views instead of refusing the dtype, and the single-domain runner checks a prepared cache's boundary inventory against the fields the configuration forces (`nwfa`/`nifa` included) instead of a six-field constant that refused every mp=28 cache.
- A domain tree whose root streams while a child stays resident no longer dies sizing the shared arena: the child's forcing slots are sized from the whole configured tree, not from the resident domains alone.
- A radiation configuration that names the same engine in `ra_physics` and in `ra_lw_physics`/`ra_sw_physics` is admitted; only a contradiction between the two spellings is refused, with the way out.
- The standalone RW-WPS wheel imports again: the saved-setups module is staged, and the nest-edge module reads the three mp=28 seeding constants from an import-free constants module the wheel stages instead of from the table contract it does not.
- The CLI reference page carries every 2.7.3 flag (`--parent-restart PATH|latest` among them).
- A saved setup with a fractional-second forcing cadence is refused with the field to change and the way to take the source's own interval.
- Noah-MP (`sf_surface_physics = 4`) configurations are no longer refused at plan review on every card. The memory estimator prices the composed Noah-MP translation units the model actually launches, from a per-thread local-frame reading taken on the card's own compile platform (target architecture and NVRTC build). A card whose platform has no reading is priced from the ceiling over the recorded platforms, and plan review names that basis beside the verdict. Readings ship for sm_86 and sm_120 at NVRTC 12.9.86, the build a CUDA-12 install resolves, and for sm_120 at 13.4.59 and 13.3.33, the builds the `gpu-cu13` extra has resolved to. The NVRTC build is set by the cuda-toolkit release the package's `[ctk]` extra resolves to, and a gate keeps the recorded builds in step with that resolution; `python tools/measure_noahmp_frames.py measure` takes a reading for another card.
- `gpuwm downscale --point --auto-vram` prices a Noah-MP parent on the card it measured instead of refusing it as a declared card. Every Noah-MP kernel factory compiles through one site, so the source handed to the compiler at run time is the source the frame reading was taken from.
- A streamed forecast is priced at the window its tile buffer actually holds instead of at a single-row rectangle no buffer allocates, so a forecast with pinned tiles that fits is no longer refused before its inputs are fetched. The refusal that remains prints the terms it priced.
- Refusals in the physics doors are held to what this tree can actually run. A sweep of 68 refusals across the configuration, plan-review and run doors classified 38 as drift between independent scheme tables rather than a real incompatibility. The pairing tables stop refusing combinations that run, a per-source land-surface offer that no route reached is removed, a dry planetary-boundary-layer run reaches the driver instead of being admitted at one door and refused at the next, and each refusal that remains names the breakage it prevents and the way out.
- The physics panel says why an option is closed instead of greying it out with a single note under the whole panel. `gpuwm companion-domains --availability` applies each installed option to the current draft, hands the result to the ordinary configuration parser and reports the first door that refuses, so a closed option and the refusal a save would produce cannot say different things. The option payload also carries the couplings a scheme requires, so a panel can say what a scheme must be paired with, and each repair names exactly the edit it performs.
- A remote forecast that finished is no longer written up as a permanent failure. The producer-completion transition reaches every remote watcher and door, a runner-exit window that cannot be proved keeps its job pending instead of writing a terminal receipt, and a stop request is answered on every pass of the compact viewer's loop rather than after the next revalidation.
- A domain that crosses the antimeridian is exported as its two seam-cut parts instead of one ring whose edge circles the world, so a narrow domain on the seam is no longer published to map consumers as its wide complement. A longitude that needs no turn is carried through unchanged. A perimeter that encircles a pole, and a perimeter with no area, are refused by name with what to do next.
- The desktop controller ends once its workspace has closed and no job is running, so the next ArWen start gets a fresh controller and a freshly resolved runtime instead of reattaching to a finished one. A terminal window the desktop had revealed is a view of the session, not a reason to keep it alive.
- Every exit from the terminal workspace restores the terminal: return, error, panic, and the interrupt, terminate and hang-up signals alike. Panic text goes to `controller.log`. The desktop launcher writes the restore bytes after the controller exits, so a controller that was killed outright still hands back a usable shell instead of one left on the alternate screen and still reporting mouse motion.
- A run viewer opened from Runs removes its own session directory when it closes, so opening saved runs no longer fills the output root with finished sessions. A viewer that cannot remove its directory leaves a recognisable closed session, which the controller sweeps when it starts and when it exits. Live sessions, other sessions, job receipts and the frame caches are never touched.
- Milbrandt-Yau forecasts can be checkpointed and resumed: the scheme is named in the checkpoint identity tables, and plan review now asks whether a run can be checkpointed at all instead of leaving the answer to the first checkpoint write.
- Every plan-review refusal names the scheme it turns away, what would break and the way out. Route refusals name the option, what the route declares and the templates or runner that reach the option; pairing and pinned-setting refusals carry the reason the run door gives. A physics combination matrix holds every registry composition, per route and source, to two outcomes: launchable, or refused with those three names.
- A per-domain radiation change on an RRTMG template no longer passes plan review and fails at run start. The non-RRTMG radiation options resolve the RRTMG receipt token themselves, and the run door's token refusal names what the token selects and both ways out.

## 2.7.2 (2026-09-10)

Fixed:
- My forecasts lists the accepted forecast immediately and follows it through acquisition, preparation, simulation, rendering and its final state. Saved runs whose receipts cannot be validated are listed with the reason instead of disappearing. A worker's plain working-directory spelling is no longer mistaken for a different directory.
- Windows desktop: the ArWen terminal is available from the desktop. Show progress in TUI, Connections in TUI and Credentials & connection settings open the terminal window of the same controller and forecast; Ctrl+Q hides it again without stopping anything.
- Opening ArWen while a forecast still runs in the background reopens the running controller's workspace instead of starting a second controller.

## 2.7.1 (2026-09-10)

- Windows desktop setup installs and verifies its own Python and CUDA runtime through a graphical launcher. Local setup checks the GPU before opening the forecast workspace.
- The desktop controller can serve the GUI without an open terminal, retaining forecast review, command acknowledgments, progress and worker lifetime handling.
- GUI startup diagnostics are saved per session. Engine version probes have a deadline, and reopening an active workspace preserves its existing connection.
- Local progress waits for the worker startup handshake before reading its process receipt.
- Completed and claimed request history no longer consumes the pending-request scan limit. Compact-viewer workers service queued jobs beyond large historical directory collections.
- A compact-cache member removed during eviction is treated as unavailable, while permission and metadata errors remain visible.
- The desktop identifies pole-crossing forecast areas separately from GPU memory failures and gives a map-based remedy.

## 2.7.0 (2026-09-10)

New:
- Case catalogs can be imported from JSON, TOML or supported ZIP documents, searched and opened in the terminal. Source-specific initialization times, domain tiers and scientific settings create an editable native configuration with the original catalog and selection receipt preserved. Catalog recommendations remain reviewable; the engine validates the selected configuration. See [case catalogs](docs/public/CASE_CATALOG.md).
- The terminal's source-aware UTC calendar shows archive bounds and available cycle hours, with direct date entry and a latest-period check. Sources without a completeness probe label publication estimates explicitly. Domain guides show the root-to-child spacing sequence and explain point-sized grids and GeoJSON footprints.
- Research workspaces organize 12 weather families, 23 submodes and 38 questions into 114 configurations, with at least three recommendations per question. Native creation uses the shared 8/12/16/24/32 GiB policy, measured free memory and study-area constraints, and retains the exact diagnostics and review receipt. Existing-parent and supplied-state methods keep their required input routes. See [Research workspaces](docs/public/RESEARCH-WORKSPACES.md).
- The terminal adds a searchable research hierarchy, shared solarpunk palette, compact and spacious layouts, a reviewed Scenario editor, and a guide for plotting real existing history. Mouse and keyboard actions reach the same native commands; saves preserve the source configuration.
- Native feature following supports potential temperature, water vapour, cloud water, rain water and vertical velocity, with explicit extrema, column reductions or model levels. Units and source availability are validated; streamed states supply their current fields, and attribute policies bind restart identity.
- Fresh GFS domain-tree preparation accepts reviewed warm-bubble scenarios while keeping prepared source arrays unchanged. The runtime applies the perturbation once; checkpoint restoration does not reapply it. Unsupported single-domain and companion stock-WRF scenario routes refuse explicitly.
- The terminal workspace defaults to 25 general plot products and offers Tornado, Hurricane, Snow, Rain and Wind selections, plus a searchable custom selector. Choices persist per configuration or remote node; command-line defaults remain unchanged.
- Terminal Nodes profiles control ArWen on Linux through OpenSSH. Reviewed starts and checkpoint resumes bind the selected inputs; durable jobs keep running when the terminal closes, with reconnectable status, logs and verified Stop. Existing configurations and data remain on the node.
- `gpuwm tui` opens the native Rust terminal workspace, with clickable actions, keyboard navigation, five-question configuration creation, editable starting presets and the full TOML editor. The Domains editor exposes nest geometry, delayed starts, storm-following and lifecycle settings; an offline downscale guide reviews the exact command before running it. Existing ArWen configurations, prepared folders, WRF inputs and WPS `met_em` have explicit entry points. The interface uses the same preparation, validation and forecast commands as the CLI. See [the terminal workspace](tools/arwen-tui/README.md).
- Existing WRF `wrfinput`/`wrfbdy` pairs and WPS `met_em` can initialize ArWen's common forecast runtime. Requested namelist settings and vertical coordinates are preserved; shared Rust initialization implements WRF's automatic eta options when the input requires them.
- Native preparation accepts an explicitly declared mean-sea-level pressure donor when the requested pressure operation needs a field absent from the primary product. Quantity, time, grid and source identity are validated; native-only preparation retains its previous result when no donor is supplied.
- `gpuwm go CONFIG --outdir OUTPUT` brings native preparation, forecasting and requested pictures into one launch. Progress names the current stage; detailed diagnostics go into the run's log. A failed render now fails the launch instead of reporting successful pictures that do not exist.
- Public prepared forecasts can continue through `gpuwm sim --restart`, `gpuwm go --restart` and `gpuwm resume`, retaining the original prepared bundle and canonical model, physics, tracking and diagnostic state. Single-domain and hierarchy continuations validate checkpoint identity before changing live state.
- Prepared delayed children use the analysis at their declared start time and activate at the exact forecast boundary. They can start with streamed storage and continue through restart. Trigger-based creation and fixed delayed caches beneath a moving ancestor remain unsupported in the prepared route.
- Supplied metgrid number and aerosol fields, predicted-CCN inflow, and supported RUC/MYNN state fields reach their declared initialization and boundary consumers. Scalar gas overrides retain the selected radiation spectra and restart identity; active MYNN `qke_adv` transport is not included.
- Domain sizing accepts `--nz` and `--tiles` before fitting the domain. The generated configuration and `gpuwm check` use the same declared free-memory budget. For supplied GRIB data, a native metadata inventory provides the actual forcing cadence and retained boundary count.
- A terminal memory refusal offers **Fit domain** or **Tile streaming**. `gpuwm domain-tiles CONFIG --out NEW` reviews a tile plan without changing the domain area, grid spacing, physics or clocks; `--write` creates a separate configuration and receipt. The planner selects tile dimensions against available GPU memory and system RAM. Explicit tile choices remain protected, and starting the forecast checks memory again.
- `gpuwm downscale --child-levels N,STRETCH` and a child configuration's `eta_levels` give an offline child its own vertical ladder. Preparation conservatively remaps the initial state and boundary frames once; a matching ladder preserves the existing result. Vertical refinement in a live nest still requires a vertical boundary operator; use the offline downscale command for that case.
- Adaptive time stepping uses WRF's controls on ArWen's clock, with exact output times, elapsed-time radiation and cumulus scheduling, and controller state carried through restart and relocation. More damping and diffusion settings can vary by domain. The configured starting step is retained when WRF's automatic starting-step sentinel is used; each domain uses its own measured CFL. See [adaptive time stepping](docs/ADAPTIVE-TIMESTEP.md) for the controls, supported settings and measured behavior.
- `gpuwm render --theme NAME|FILE.json` changes fonts, labels, backgrounds and map styling, with `default` and `dark` built in. Existing weather color scales remain unless explicitly overridden. `xsec:` products and `--section` draw vertical cross-sections with terrain masking, isotherms, overlays and optional logarithmic scales in the normal domain/product output layout.

Fixed:
- Terminal help and quit remain reachable in forms and small windows. Guide answers survive Help and Escape, the editor supports undo and redo, and monochrome mode retains focus cues. Local starts remain responsive; Stop and ordinary Quit track the owned process through completion. Saved node profiles are shared across working directories and can be removed without stopping remote jobs.
- Research recommendations use diagnostics supported by the native forecast renderer, with separate notes for questions requiring further analysis. Moving-grid studies disclose finite coverage and tracking-radius limits, and retain a storm-track CSV by default. Supplied-input creation errors report the missing input or conflicting path plainly.
- Source installers install the matching local data companion and build the terminal workspace on Linux and Windows. Binary notices include the locked terminal dependencies, and doctor describes the changes when upgrading from 2.6.5 to 2.7.0.
- Forecast health checks reject non-finite fields and folded vertical layers before publishing the affected history frame or checkpoint. Final summaries report checked state health. Prepared adaptive trees reserve tile halos from the complete static map fields before allocation, and moving streamed children retain the canonical held PBL rates used by Grell-Freitas and New Tiedtke.
- Prepared HRRR water temperatures reach the Noah soil initializer, explicit soil-texture downscaling choices remain effective, and WDM6 supplied-state initialization applies the required initial CCN field to the production state.
- Terminal-created forecasts enable periodic checkpoints, repeated creation chooses an unused output name, and archived downscaling accepts the actual parent cadence while retaining the requested child output interval.
- Named memory-budget refusals and multiline preparation failures retain their reasons and remedies at the command line. `go --supplement` carries explicit pressure donors through preparation, and doctor distinguishes compilation, kernel launch and result-validation failures.
- Repeated forecasts automatically use separate download caches when their source, cycle, area or acquisition settings differ. Matching requests reuse verified inputs; old flat data folders and incomplete or incompatible caches are preserved. The wizard's default launch command uses this same automatic behavior, while an explicitly selected data directory stays explicit.
- Archived downscaling renders legitimate optional restart defaults, checks the full child time window against the parent archive, binds restart domain/grid evidence, and validates explicit child surface placement, including dateline and polar geometries.
- Cross-platform native static replay compares dimensionless rotation fields at unit-vector rounding scale near zero. Exact input hashes and grid identity remain bound; regenerated geometry remains authoritative, and changed spacing or placement still refuses.
- `gpuwm render --series` combines compatible saved history files for windowed rainfall products, and `go` supplies earlier frames to later rendering stages. Existing pictures retain their original bytes. Products still skip an insufficient time window explicitly; invalid inputs, unknown products and requests producing no image return a nonzero exit.
- Platform-installed renderers find map overlays staged by `gpuwm fetch-bridges`, even when the preferred executable is inside the wheel. Explicit map-asset settings remain in effect.
- New Forecast and Fit use currently available GPU memory when detecting the local card, so a new domain does not immediately exceed Run's memory budget. Explicit card sizes retain their declared-capacity behavior, and fitting preserves the selected physics and timing. Clicking a failed command's message opens its error details and log, with explicit Copy details and Copy log actions. Domain sizing remedies appear only for a confirmed memory refusal, so unrelated failures do not suggest shrinking the domain.
- Preparation accepts the quoted TOML table and key names written by Fit without appending duplicate physics sections. Stream continuation recognizes the same quoted and escaped clock declarations. Both writers preserve complete multiline values and unrelated settings, and validate the generated configuration before publishing it.
- Native preparation no longer requires stock-WRF export to succeed. Requested ArWen physics remain selected, and an explicitly requested unsupported WRF export is checked before expensive preparation.
- Automatic memory planning uses the configured resident requirement and shared domain, tile and nesting allocations. It retains the initial device observation and applies an available device-wide memory cap to the selected GPU. Fitting automatic layouts are tried promptly, with a complete mixed-layout fallback; explicit tile settings remain unchanged. See [automatic planning](docs/AUTOMATIC-STREAMING-SEARCH.md).
- Streamed execution supports ordinary adaptive single-domain runs, stationary nesting with both domains streamed, and moving streamed children with bounded reconstruction and public restart. Fixed tracking windows retain their contents through checkpoints. Supported moving prepared runs use a validated statics corridor; domains can exceed the GPU's resident storage capacity through explicit tile settings.
- Independently selected longwave and shortwave settings, CAM ozone, gas controls and component geography reach the prepared runtime without replacement physics. Classic radiation workspaces are priced with the configured column cap, and tall legacy shortwave columns use the corrected shared implementation.
- Preparation avoids repeated metadata materialization and redundant immutable-frame copies. Native atmospheric windows retain the exact required source support while preserving complete source validation; canonical fields are released after their final publication consumer. Changed run-only controls and random seeds can reuse verified preparation, while changed scientific inputs rebuild it and preserve superseded output. Raw decoding and source assembly can still retain a complete source time; these changes do not promise a general preparation speedup.
- Editable source installs no longer require release bridge pins while generating package metadata. Distributable wheels still verify their native assets when built. Windows available-memory checks and cold CUDA compilation checks now run before a forecast can proceed on a misleading successful preflight.
- Forecast failures retain the failed stage, exit status or signal, and an actionable diagnostic. Memory reporting distinguishes device memory from host memory; a declared budget cannot increase the physical capacity of a GPU. Streaming plans price the actual resident and streamed domains and retained forcing intervals.
- Exporting a long prepared series no longer retains every memory mapping and exhausts the process's file-descriptor limit. Live consumers keep their arrays valid while unused mappings can close. Configured native physics selections now travel through preparation and its validation receipt together.
- Below-terrain extrapolation of the source column no longer clamps the interpolated total pressure to the target dry pressure. real.exe has no such step, and the clamp is replaced by a refusal that names a non-finite or non-positive interpolated pressure. The affected case's potential-temperature error falls from about 3.56 K to 0.00053 K against the independent WRF initialization reference; the unchanged control retains identical data. This measures initialization agreement, not forecast skill.
- Nested preparation, source coverage, preflight bounds and water-overlay sampling use the same geographic-to-source transform. Projected coordinate axes are no longer read as degrees. A domain outside the source is reported in the correct coordinate system.
- Fine vertical grids retain pressure accuracy by preserving layer-thickness and pressure-coefficient residuals otherwise lost in float32 cancellation. Advection is unchanged. Inflow perturbations retain their configured refresh duration under adaptive steps, and incompatible inflow/PBL pairings are reported during configuration loading.
- PBL and land-surface timing follow the live model step. Held PBL and radiation tendencies survive skipped physics calls, tiled transport and checkpoints, and are recoupled to the new dry mass after a moving domain relocates. The original physics cadence is preserved.
- Adaptive checkpoints no longer reject their own derived acoustic step count. A resume may change supported controller targets and bounds, with each change reported, while changing the controller mode remains a model-state incompatibility. Nested step selection avoids prime-factor combinations that could silently collapse a child's step and forecast speed.
- Relocated checkpoint identity is checked against the original recorded components and the current configuration. Scripted moves before a restart are retired so later moves can execute. All relevant NetCDF/HDF5 users share the synchronization needed to prevent concurrent access from crashing relocation.
- Restart payloads are validated before state mutation and written through uniquely named, durable temporary files. The streamed single-domain output path reads current host-store fields; output-only diagnostics no longer make its checkpoint inventory fail. Streamed summaries and final digests read the live clock and state.
- Prepared single-domain and nested caches share the compatibility rules for older configuration headers. Fields with recorded, harmless legacy defaults no longer invalidate every older preparation; real identity changes still report the differing fields. A temporary provenance lookup failure or an untracked output file no longer makes an unchanged forecast implementation appear to change during a run.
- `GPUWM_COMPOSE_SCRATCH` also controls mapped decode and inspection scratch, allowing preparation to use a disk with sufficient space. Incomplete caches are distinguished from valid reusable data, and cleanup preserves files the run does not own.
- NSSL rain freezing preserves the separately rounded mass update before rain-number limiting. This corrects the reproduced cancellation error without changing global compiler arithmetic or loosening the WRF reference tolerance.
- Grell-Freitas uses ArWen's independently checked gamma implementation. Its deliberate difference from WRF's library gamma, and the paired-reference comparison option, are documented in [the gamma comparison](docs/gf_gamma_known_delta.md). The glibc-derived implementation and two glibc-only fragments are removed. Source and binary distributions now carry the applicable license and attribution texts, including those for ported NumPy and MPAS code and the companion physics tables.
- Portable frame inventories reject noncanonical backslash paths consistently. Generated CLI documentation uses portable paths. Decoder validation, static/output metadata and physics coverage records are corrected; numerical reference tests retain their declared platform distinctions rather than silently accepting changed results.

### Results that move from 2.6.5

- Dry mixing tendencies carry WRF's rk_addtend_dry map-factor division (1/msf) on mapped grids with km_opt = 2, 3 or 4. Every shipped real-data configuration selects km_opt = 4, so the u, v, w and theta mixing rows move by the local map factor and 2.6.5 mapped-grid bitwise baselines no longer reproduce.
- Real-data initialization loads the hydrostatic pressure recurrence with total water (vapour plus every analyzed condensate species), as module_initialize_real.F does. The initial pressure moves wherever the forcing carries hydrometeors: native HRRR, wrfinput, met_em and any analyzed_species route. Measured worst move 4.34 Pa on the native HRRR case.
- The MM5 surface layer (sf_sfclay_physics = 1 or 91) publishes USTM unconditionally, so the km_opt = 2 TKE surface drag with a PBL scheme on reads the real u* instead of zero. That configuration's TKE budget and mixing coefficients change at first order.
- Kain-Fritsch (cu_physics = 1) shallow feedback tendencies divide by the unrounded 2400 s TIMEC that module_cu_kfeta.F re-sets before its feedback loop. Whenever dt does not divide 2400 s all six shallow tendencies move; at dt = 90 s they were 1.23 percent low.
- NSSL two-moment (mp_physics = 18): the Bigg rain-freezing graupel gate is 1e-12 kg/kg (WRF's live override of 1e-7), the fused ice-nucleation vertical velocity is WRF's single interface average 0.5 (w[k] + w[k+1]), and the low-temperature cnuc term reads the raw staggered w. Transfers move in the 1e-12 to 1e-7 window and primary ice nucleation moves wherever w has vertical shear.
- Morrison (mp_physics = 10) rebuilds the PGAM reference density from the current temperature, pres / (287.15 T), instead of the density frozen at entry. Cloud effective radius moves and RRTMG radiation moves with it.
- Grell-Freitas (cu_physics = 3) keeps the k = kbcon layer in the undilute CAPE integral, as WRF does. On the measured column aa0 moved from 0 to 0.0349, which changes where convection triggers.
- Grell-Freitas (cu_physics = 3) evaluates gamma with ArWen's correctly rounded float32 implementation; the glibc-derived one is removed. On the 216-column capture RAINCV and PRATEC move by at most 7.3 percent, median 1.6 percent.
- P3 (mp_physics = 50) spells three integer exponents as multiplications, as gfortran compiles them, instead of routing them through powf; results move at the last-bit level.
- MYNN surface layer (sf_sfclay_physics = 5): the stable psih table's 1/1.1 exponent is folded in float32 as gfortran folds it, which moves 268 of the 1001 table words by up to 3 ULP; the table's pinned SHA-256 changed.
- Every non-root domain owns its acoustic carrier ww_pp instead of sharing a view of the root's, so a child no longer overwrites its parent's held boundary flux. Multi-domain results change, and resident GPU memory grows by one (nz+1, ny, nx) float32 field per non-root domain; a tree that just fitted a card in 2.6.5 can now be refused.
- ERA5 preparation fetches the lake-model mixed-layer temperature and uses it for lake cells with ice-free lake state, replacing the skin temperature fallback, so lake surface temperatures move on every fresh ERA5 case with lakes. A lake cell whose source lake state is frozen, partially frozen or missing retains the existing analysed-water or skin-temperature fallback. The preparation receipt reports the affected cells, including wholly frozen lakes. A 2.6.5 ERA5 GRIB cache without the lake messages keeps its fallback.
- Prepared HRRR initialization plans a soil-texture mesh on the projected source grid, so soil moisture and deep soil temperature are downscaled onto domains finer than the 3 km source. A 3 km domain from HRRR is unchanged; a 1 km domain moves.
- A nested child's soil column is built on the configured layer count (num_soil_layers, resolved per land-surface scheme) instead of the nine-layer default; six-layer RUC children used to fail forecast loading.
- wrfout files carry sixteen more global attributes (NUM_LAND_CAT, GMT, JULYR, JULDAY and the twelve *_PATCH_* extents), and SIMULATION_START_DATE on a delayed-start nest is the run's origin rather than the domain's own start, so lead times read from that attribute agree across domains.
- [tiles] mode = "auto" with any of tile_nx, tile_ny, nbuffers or halo pinned is refused when the configuration loads. 2.6.5 accepted and silently ignored the pinned keys (nbuffers = 2 planned 3; a pinned tiling streamed a domain that fits). Pin them under mode = "on", or cap auto with vram_budget_bytes.
- `gpuwm go` applies explicit native planner refusals and the admission budget (free GPU memory minus a 0.5 GiB margin for other processes). Memory refusals require a measured card. An unexpected failure in the tile-planning report is announced and admission uses the resident price; that report error alone does not refuse a runnable configuration.
- Opt-in, diff_6th_opt = 1 or 2: the moisture, number, volume and TKE rows are normalised on dt/3 and damp three times harder than the four dry rows, as WRF's rk_scalar_tend does.
- Opt-in, feedback = 1: child-to-parent feedback restricts the raw prognostic fields with theta rebased, without mass weighting, as WRF does; two-way runs move.

### Restart and import compatibility

- A 2.6.5 checkpoint cannot be resumed by 2.7.0. The restart identity compares the run configuration key by key and 2.7.0 added thirteen fields (eta_levels and the twelve adaptive-time-step controls), and the member inventory now requires fields/ustm on every MM5 surface-layer run and held/gf_rthblten plus held/gf_rqvblten on every Grell-Freitas and New Tiedtke run. 2.7.0 writes restart format 6. Format-5 checkpoints, including earlier 2.7.0 previews, are refused with a compatibility explanation. Start a fresh run from the original configuration and inputs; checkpoint migration is not included in this release.
- Preparation rebuilds a prepared cache written by 2.6.5 for a native-HRRR or other condensate-carrying source, because this release prepares those sources differently; the superseded output is preserved.
- A namelist that omits input_from_file, or gives it a short per-domain column, keeps WRF's Registry default of .true. for the omitted entries on import-namelist, run --wrfinput and run --met-em. Explicit values retain their normal validation.
- use_theta_m = 1 in an imported namelist is no longer refused on the met_em and wrfinput doors. It is emitted as 0 because those doors construct the dry-theta state themselves. The import report explicitly identifies this as a reasoned substitution, without claiming equivalence to WRF's moist-theta integration.

## 2.6.5 (2026-09-02)

Fixed:
- `gpuwm fetch-geog` stages the soil archive as `WPS_GEOG/soilgrids/<dataset>` and every reader looked for `WPS_GEOG/<dataset>`: doctor's "WPS_GEOG (MPAS static)" row, the mesh door's pre-flight and `rw_mpas_static` itself. An install that ran doctor's own remedy was told it was still five datasets short and `gpuwm mesh` refused the static half of the pair. All three read both layouts now, off the pin table's own child declarations; the static built from a staged tree is byte-identical to one built from a flat tree.
- `gpuwm mesh` gives its runnability verdict again. The door read the static engine's absent-field list from a receipt key the unified writer never wrote, so every pair built since then was announced "NOT MEASURED" after a successful build; the list is now derived from the variables the engine wrote, and a receipt with neither is still reported as not measured rather than runnable.

## 2.6.4 (2026-09-02)

New:
- `gpuwm cells` turns a run's history into storm-cell objects through the titan storm-cell engine: `gpuwm cells export` resamples `REFL_10CM` (and temperature) onto a fixed height ladder and writes titan's checksummed volume stream with a receipt naming the ladder and the interpolation rule; `gpuwm cells analyze` runs `titan analyze` into the render folder layout (`<case>/<domain>/cells/<first-valid-day>/`), refusing by name when no titan binary resolves (`--titan`, `GPUWM_TITAN`, the bridge directories, then the PATH); `gpuwm cells catalog` writes one row per cell per frame joining titan's identity, track, age, area, echo tops, motion, trend and forecast footprint to ArWen's own attributes sampled inside the cell's footprint -- peak vertical velocity in m/s and ft/min, cloud top and base heights and temperatures, freezing and -5/-10/-15/-20 C levels, supercooled liquid water path -- as CSV, JSON with units and provenance, WGS84 GeoJSON, and one renderer overlay file per frame for `gpuwm render --overlays`. The door sizes titan's trend window and gap tolerance to the series' measured cadence (a radar profile's 1,800 s window holds one point of hourly history, so every trend came out zero), records the change in the receipt, and never overrides a `--titan-config` key; the catalog carries titan's fitted trend motion (`trend_*`) beside the tracker's Kalman state (`motion_*`), named apart because on a constant-velocity test storm the trend is exact and the state is not. The MCP server gains `arwen_cells` (the chain as a job) and `arwen_cells_catalog` (the catalog as JSON). Documented in `docs/cells.md`, which states the resolution caveat: a model resolves an updraft only when the core spans four to six cells, so peak `W` from a 3 km run is a grid-limited bound.
- `cu_physics = 16` selects New Tiedtke cumulus, the WRF v4.6.1 `module_cu_ntiedtke` scheme ported to CUDA: all 21 stages and the assembled pipeline reproduce the byte-frozen Fortran bitwise over an 18-case, 6-spacing oracle corpus that ships under `gpuwm/data/ntiedtke/oracle/`; the scheme requires a PBL scheme and `cudt_minutes = 0` and refuses by name otherwise; its state rides across a restart; chunking is proven not to imprint (nine of nine output files byte-identical at two chunk widths on a real two-domain case). `ntiedtke_tiedtke_closure` (default off) runs its kernels with classic Tiedtke's deep closure. Documented in `docs/cumulus-new-tiedtke.md`, which records the scheme's grid-thickness deep first guess and its effect on a tropical-cyclone core.
- Per-level diabatic heating is written for every configuration (`H_DIABATIC`, `RTHRATLW`, `RTHRATSW`); the three were computed each step and discarded, and writing them changes no forecast value.

Fixed:
- Nest relocation freed neither the dropped physics driver nor its cumulus adapter: the two held each other in a reference cycle, so every relocation left the whole driver graph resident (measured per relocation: Grell-Freitas 35 MiB, New Tiedtke 61 MiB, Kain-Fritsch 1.3 MiB). The adapter now releases its driver reference at relocation and the receipt carries `cumulus_workspace_bytes`; output is byte-identical before and after.
- Prepared trees written before `run.p3_backend` joined `RunConfig` were refused with `these identity fields differ: run.p3_backend`. The field joins the tolerant identity table: it is scoped to `mp_physics = 50`, so a tree written before it existed could not have selected a backend, and a tree that carries a value is still compared strictly.

## 2.6.3 (2026-09-01)

New:
- `gpuwm doctor` measures the install root against Windows' 260-character
  path ceiling, where a deep root on a box without long paths enabled
  makes every kernel source read fail as a missing file. Windows only,
  skipped by name elsewhere.
- The MPAS column-batch physics seam transports aerosol-aware Thompson
  (`mp_physics = 28`): WRF's eleven scalars, graupel receipts, the surface
  aerosol emission pair on the restart payload, and the scheme's own
  aerosol initialization at the seam's first microphysics call on the
  caller's state, so a batch that brings no aerosol data is a valid batch.

Fixed:
- 2.6.2 was tagged but never published: a test in the suite rewrote
  `os.name` process-wide, so every path operation raised on Linux and
  the release job died. 2.6.3 is that release plus the fix, and no
  2.6.2 artifacts reached PyPI.
- `gpuwm fetch-tables` stages `CCN_ACTIVATE.BIN` into the table root, so
  `mp_physics = 28` runs from a wheel install. The command used to report
  the root complete and byte-valid while every aerosol-aware forecast
  died at its first microphysics step. It carries the classic set's pin.
- `gpuwm fetch-tables --wif` no longer lets the optional dataset leg
  cancel the mandatory one. Only `--wif-only` skips the coefficient
  tables, the exit code is the worst of the legs that ran, and a summary
  names each leg staged or refused.
- `gpuwm fetch-tables --wif` resolves `QNWFA_QNIFA_SIGMA_MONTHLY.dat`
  from the versioned release-assets base, honoring
  `GPUWM_BRIDGE_ASSET_URL_BASE`, where it used to answer 404. A refusal
  names the file, the URL and two ways to supply it.
- A bridge staged in `~/.gpuwm/bridges` by an earlier release no longer
  reaches a door: every ladder checks it against this install's pins and
  re-fetches by default. Offline it refuses with the file, its revision
  and the replacing command; overrides, in-tree builds and wheel-shipped
  binaries are never judged.
- The Rust observation engine `rw_obsgrid` reads the windowed radar-grid
  layout natively. 2.6.1 made that layout what the ingest path writes
  while the engine still implemented the whole-domain one, so every
  freshly built grid was refused at the schema check. Older files stay
  readable.
- `gpuwm doctor` reports a skewed `gpuwm-data` companion as a blocking
  row carrying the exact pip line, rather than dying in an uncaught
  `ImportError`. The base-dependency line no longer prints `ok` over the
  pinned companion.
- The companion version lock compares PEP 440 public versions, so a
  wheel built with a local label accepts the `gpuwm-data` it was cut
  against, with a typeable remedy. A genuine release
  mismatch is still refused.
- The GPU preprocessing remedy names the certified CUDA pair instead of
  an uncertified extra. It states that such a box has no
  certified pair today and that the CPU backend runs there.
- A config's relative file references resolve from the config's own
  directory as well as the working directory, so a config that ships
  beside its data loads from anywhere. A value that already resolves is
  left byte for byte alone, so run fingerprints do not move.
- A host-only run no longer aborts at its first step boundary when CuPy
  is installed and every device is masked. The pool trim is an
  optimisation, so an unreachable pool now skips it instead of raising.
- Morrison deposition-freezing nucleation is bounded by the vapour excess
  over ice saturation. The unbounded term drove qv negative below about
  159 K, beyond a regional lid; it also engages between about 190 and
  213 K, which a cold tropopause under the default 50 hPa lid reaches.
  That divergence from WRF is declared and measured; no parity pin moved.
- The WPS_GEOG mirror carries the Noah-MP `soilgrids` archive, so a
  mesh-door `gpuwm fetch-geog` no longer falls from the default source to
  a 404. The publisher refuses to stage a pin table it cannot render.
- The two rw-mpas golden helper scripts compile again: a helper sat
  inside each module docstring, leaving both unterminated and the
  Rust tests' goldens impossible to regenerate. The whole-tree
  source scan now fails on a file it cannot parse.
- The RRTMGP cloud-fraction refusal names the scheme it refuses instead
  of citing only WRF line numbers, so the reader learns which arm is
  unsupported.

## 2.6.1 (2026-08-31)

New:
- An MCP server: `arwen-mcp` (also `python -m gpuwm.mcp`) serves the
  CLI doors as tools over stdio, so an LLM agent can plan, price,
  fetch, prep, forecast and render. Every tool runs the real command
  line or reads a receipt a door already wrote; refusals arrive as the
  CLI's own sentences, verbatim. Long work runs as detached jobs that
  survive a server restart, with one GPU job at a time per card. The
  SDK is the new `[mcp]` extra (`pip install gpuwm[mcp]`); see
  docs/mcp-server.md.
- The MPAS column-batch seam carries P3: `microphysics_scheme = "p3"`
  selects it and the seam transports P3's eight species scalars
  (rimed-ice pair included) through both phases, beside the unchanged
  WSM6 default. Restored P3 seams continue bit-identically under the
  same v2 payload; the scheme rides the payload identity, so a payload
  restored under the wrong scheme refuses by name.
- Level-2 regional spectral numerics: an optional `[spectral_numerics]`
  table runs scale-selective spectral hyperdiffusion and divergent-mode
  wind damping once per completed slow large step, in `off` (bitwise
  inert, the default) / `shadow` (hash-bound receipts, state untouched)
  / `apply` (opt-in, budget-gated) modes, with per-step receipts bound
  into the run capsule, a `gpuwm spectral-op` evidence door that stays
  CPU-reachable without CuPy, and loud refusals on streamed domains,
  false periodic declarations and the loops that cannot honor the hook.
  See [LEVEL2_SPECTRAL_NUMERICS.md](docs/public/LEVEL2_SPECTRAL_NUMERICS.md).
- `gpuwm cycle`: continuous assimilation cycling as a door. A cycle
  clock schedules analyses, a ledger records every cycle's inputs and
  verdicts, a supervisor drives fetch, assimilation and forecast legs,
  and anchored child domains follow the feature being cycled.
- Nowcast skill is scored against an operational baseline, with
  structure metrics beside the existing point verification.
- Windowed all-radar ingest: assimilation cost follows the sites in
  the active window, not the domain.
- Overlap handover: successive nest windows hand their interior over
  during the overlap instead of restarting the child cold.
- Static builds below a 1,310 m pixel diagonal no longer refuse: the
  static builder derives categorical fields by supersampling, keyed to
  each mesh's measured minimum edge. Demonstrated end to end with a
  937.5 m mesh generated, static-built, and run.
- Nineteen measured sizing configurations land for sources the
  configuration wave had not covered, ensemble members and reanalyses
  included, with headers stating what was actually measured.

Fixed:
- `gpuwm prep` no longer grows its host memory with the length of the
  forcing window. The mapped engine reads, decodes, composes, writes and
  drops one valid time before asking for the next, so the peak follows
  one valid time rather than the number of them, and the identity pass
  admits objects in bounded parallel batches under a 64 MiB staged-byte
  budget instead of one at a time. Measured on public DWD ICON-EU
  bytes, a 3 h and a 6 h window both peak at 2.32 GiB against 7.21 and
  12.68 GiB before, memory stays flat at both windows, and the frames
  are byte-identical on compressed and uncompressed sources alike.
  Most of the per-time wall cost is recovered (the inventory pass runs
  2.9x faster); a compressed field-per-file source still pays roughly a
  quarter over its pre-fix wall, because each record is decompressed
  once for identity and once for decode. Uncompressed multi-message
  sources are unaffected.
- `gpuwm check` on a nested tree with `[tiles]` prices the mixed road the
  run door actually takes (child streamed, parent resident) instead of
  only the fully resident tree. It used to exit 1 against trees that fit
  and complete. The report now names each domain's road, tiling and
  claim, prints that plan on a refusal too, and publishes it as
  `tree_road` in `--json`.
- A `[tiles]` configuration the loader refuses now reaches the terminal
  as a refusal with its remedy intact, on every door, instead of a
  twenty-line Python traceback.
- A restored MPAS column-batch seam no longer refuses its first
  radiation-not-due step ("GLW has no producer"): the restart payload
  now carries the radiation carrier provenance (restart schema v2), and
  the seam's GPU round-trip suite joins the release battery so the
  question is asked at every cut.
- P3 (mp_physics = 50) no longer computes NaN supersaturation ratios on
  its first step: the cold-start saturation mixing ratios are floored
  at 1e-20 (the remedy newer P3 releases carry themselves), pinning the
  step-1 ratios at exactly -1, fully subsaturated. Default-on in all
  three arms and inert from step 2 onward; runs are bit-identical to
  2.6.0 from step 2 on, and the documented step-1 delta is that the
  trace-condensate clip can now fire where NaN made every comparison
  false.
- A cycling DA run no longer analyses on the CPU by default while the
  device solve sits behind a flag. `RadarAssimilationConfig.solve_device`
  and `tools/da_cycle_prepared.py --solve-device` both default to `auto`,
  which takes the card when the process can reach one and numpy when it
  cannot. `host` and `cuda` are still honoured verbatim -- pin `host` to
  reproduce a receipt banked on numpy, pin `cuda` to make a missing card
  an error rather than a silent slower run -- and `GPUWM_NO_LOCAL_GPU`
  outranks the probe, so a CPU-only run never opens a device context.
  Every analysis receipt now carries the request, the device that ran,
  and the sentence explaining the resolution.
- Velocity dealiasing uses each radial's own Nyquist velocity. The
  reader already detected per-radial disagreement, then dealiased
  against one scalar anyway.
- A radar with no innovations no longer reports the whole network's
  residual as its own.
- The continuous nowcast daemon no longer crashes at the first
  observation stage of every cycle: its observation command now passes
  the dealiasing choice the stage requires, and a test binds the call.
- Data assimilation reads P3's own 10 cm reflectivity instead of
  refusing an mp=50 background as having no reflectivity formulation.
- Clear-air reflectivity carries one floor value; a P3 field could
  previously hold two different clear-air sentinels at once.
- mp=50 composes with RTE+RRTMGP radiation: the adapter couples P3's
  own two-radius cloud optics (no snow radius, matching the scheme's
  own inventory), the refusal that outlived the gap is retired, and
  the accepted composition set grows by exactly those pairings.
- A P3 lookup table with the wrong version is refused as a version
  mismatch instead of being reported as byte corruption, and `gpuwm
  doctor` now checks the P3 table.
- The P3 oracle suite fails when physical processes are switched off:
  the previous fixture set stayed green with five processes disabled,
  and the new fixture plus per-process checks close that blindness.
- Mesh generation receipts print small trace values in scientific
  notation instead of rendering them as 0.0000.
- The mesh ladder's location check credits grading and insertion
  instead of a fixed one-percent band, so it stops refusing meshes it
  graded correctly seconds earlier; the re-aimed check is proven in
  both directions.

## 2.6.0 (2026-08-30)

New:
- P3 microphysics (`mp_physics = 50`): one ice category carrying a
  predicted rime mass fraction and rime density. Runs on the card by
  default; `p3_backend` selects `cuda`, `fused`, or the `reference` CPU
  transcription, with the two device arms byte-identical on every
  fixture and -fmad=false because contraction measurably moves the
  numbers.
- P3 is measured against WRF's own Fortran: unmodified module_mp_p3.F
  (v4.5.2) at -O0 over twelve fixtures. 4 of 12 bit-identical, five more
  within 2-7 ULP; the two long mixed-phase cases bifurcate exactly as
  the Fortran does against itself under a one-ULP nudge. The executable
  oracle ships in the tree. Maturity stays implemented-unverified; the
  open items are named in the registry row.
- P3 reaches every shipped seam: native namelist import (mp=51/52 stay
  refused by name), the stock-WRF initialization inventory, real-data
  starts with analyzed hydrometeors, nest births that seed the rime
  pair, an mp=50 wrfinput handoff, the RW-WPS support report, and
  legacy RRTMG's one-category-ice special case.
- P3 nest edges run in both directions against every ported partner, a
  defined and documented ArWen extension where WRF forbids mixed
  selectors outright: mass-conserving merge on entry, rime-state split
  on exit, with P3's own calc_bulkRhoRime holding qirim <= qitot and
  the [50, 900] density bound by construction on CPU and CUDA alike.
- The native-HRRR route admits P3: the microphysics admission is
  derived from the ported set (minus the named mp=28 aerosol block),
  and a registered preparation profile
  (p3-mp50-ysu-mm5-noah-rrtmg-legacy-v1) makes it a template selection.
- The offline downscale reads an mp=50 parent; the one gap the
  composition creates (a cross-scheme offline edge) refuses by name.
- Milbrandt-Yau (mp=9) joins analyzed-hydrometeor initialization: a
  cloudy analysis starts with its five decoded species instead of
  condensate-free.
- A cu_physics=3 import prints a scheme-generation notice (this is WRF
  v4.6.1's Grell-Freitas, not v4.8.0's Grell-Freitas-Li, and the two
  namelists are byte-indistinguishable); cugd_avedx != 1 is refused by
  name.
- Mesh generation is faster on both arms: the default arm stays
  byte-identical to every pinned mesh at 1.28x-1.71x, and
  `gpuwm mesh --triangulation incremental` serves never-built hi-res
  construction at 4.28x on a 1.7M-cell build. The crate test floor
  rises to 418.
- Fine-mesh unlock: generated meshes store binary64 points and the
  dual-edge floor becomes 400 coordinate quanta, so sub-kilometre
  meshes generate instead of being refused.
- rw-netcdf is no longer the weak crate: mutant survival drops 67.8 to
  4.6 percent, the mutation debt ratchets to 986, and the crate's test
  floor rises 7 to 43.
- A mutation-testing leg joins the battery: cargo-mutants runs over the
  Rust a commit changed. Its first estate run gave up six real defects,
  all fixed.
- Dycore symmetry gates (x-reflection, x/y transpose,
  well-balancedness-at-rest) and five further battery gates for faults
  nothing had caught, including a listed file that stops collecting now
  failing the leg.

Fixed:
- The mesh steepest-gradient gate sampled a lattice too coarse to see
  refinement transitions and admitted a spec at 7.3x its ceiling.
  Density fields now declare where they vary, partial probe coverage
  refuses before a number is read, and 157 re-measured specs moved 5
  verdicts, all admit-to-refuse.
- `gpuwm downscale` of an mp=10 parent died at its first radiation call
  on rounding-scale negative moments; offline children now route
  through the same positive-definite tolerance policy as online ones.
- Writers landing on tracked files stop translating newlines, and the
  line-ending hook reads worktree bytes and arms itself on first use.
- The RRTM longwave adapter (ra_lw_physics=1) resolved cloud-fraction
  species flags as a fused pair, so a supercooled Kessler-class cloud
  radiated as clear sky. The flags now resolve per scheme; P3 is
  byte-identical before and after.
- `rw_mpas_mesh --dry-run` prints its JSON record alone again.
- The registry, manual and public docs stop calling mp=50 CPU-only and
  oracle-less.
- Rendering a Rust-written classic history tape no longer dies at the
  Times variable: the renderer's NetCDF reader gains a real text read
  for character arrays.
- A prepared domain tree written by an earlier version binds again
  after upgrade: five newer experiment fields had joined the identity
  without tolerance entries and blamed the experiment file.
- The two-domain HRRR forecast tool runs its own shipped configuration
  again; its constant downward-longwave guard had refused it since the
  guard landed.
- CF time units that declare a real UTC offset (`+05:30`, `-0600`, `Z`)
  decode to the instant they name in both Rust readers. The dump tool
  used to drop the calendar silently and the mapped engine refused the
  file; offsets are bounded at 18 hours and a malformed designator is
  refused with the reason.

## 2.5.8 (2026-08-27)

New:
- Graded (variable-resolution) MPAS meshes generate on a hierarchical
  Goldberg ladder, and a mesh from this generator has now completed a
  6 h full-physics forecast (151,649 cells, 20 km core in an 80 km
  background, 180/180 steps finite, byte-identical across two runs) --
  the first variable-resolution mesh this project generated that ran.
  The mechanism: level 0 is the proven uniform Goldberg arm, each level
  halves the spacing by count-targeted midpoint insertion under a
  level-clamped field, and the transition annuli are drained by
  count-changing defect surgery with hysteresis, per-site op caps and
  halting by counting. The canonical 15-to-60 km spec delivers 224,210
  cells against 224,208 predicted with min dvEdge/dcEdge 0.0407 and zero
  edges under 0.04; the Fibonacci arm is retired to a control instrument.
- Every mesh relaxation samples min dvEdge/dcEdge each sweep, stamps the
  trajectory into the receipt, and refuses a collapsing tail with the
  TRiSK amplification named.
- mesh304_probe grows --from-grid, --surgery insert-delete, --emit-grid
  and --seed hierarchy arms, so recorded failures, their repairs and the
  graded generator all measure through one census.
- `rw_mpas_lbc`: the MPAS v8.4.1 lateral-boundary-condition stream in Rust.
  Per boundary time it consumes one source-agnostic WPS intermediate plus the
  regional initial-conditions file, runs the `init_atm_case_lbc` pipeline and
  writes `lbc.<time>.nc` full-mesh CDF-5 files with native-identical layout,
  header offsets and attribute block. No source model is named anywhere in
  the producer; cadence arrives as `--interval` rows and header metadata as
  `--config-attrs` table data.
- `rw_mpas_lbc compare`: grades a produced lbc file against a native one from
  its own CDF-5 byte walk, per-variable raw-slab comparison with measured
  max-ULP/abs/rel and location, plus a header field-by-field diff. Against
  the three native case-9 files of the 2026-08-25 regional oracle: xtime,
  Time, lbc_qc, lbc_qr byte-exact; lbc_theta within 11 ULP; lbc_qv 7.8e-8
  abs; lbc_u 1.5e-5 m/s; lbc_w 4.3e-6 m/s; lbc_rho 2.3e-4 rel, against the
  native case-7/case-9 pair's own 1.5e-4 on that field. Headers agree on 68
  of 70 attributes; `file_id`, `history` and the added `gpuwm_provenance`
  are the declared identity set.
- One documented divergence, deliberate: native case 9 runs every boundary
  time in one process, so after the first time its isobaric/model-level
  branch test reads stale `p_fg` content. The producer is stateless per time
  and takes the branch each file's own content selects; it matches the native
  files on the oracle series, and a source whose level table changed
  mid-series would diverge from the native quirk.
- `rw_store::netcdf_classic::create_with_min_header`: the classic writer can
  reserve header space (the `h_minfree` convention) so data sections land at
  the offsets SMIOL reserves; `create` is unchanged.
- `rw_mpas_lbc` produces a lateral boundary from another run's own output, so
  one run can drive the next. The parent's decoupled prognostic state is
  sampled onto the child's own cells and edges and remapped onto the child's
  own levels; nothing is rebuilt, because the lbc stream's contents already
  are the model's state. Density is remapped in the log, a one-cell edge
  takes its height column from the cell that exists, and both divergences are
  named in the receipt. Measured against the 2026-08-25 regional oracle: a
  child that is its parent gets theta, u and w back bit for bit over 831k
  values, rho within 2 ulp and qv differing only at the six points the parent
  held below zero; two runs are byte-identical; and a 120 km regional
  forecast drives a 24 km child across five hourly frames in 2.1 s on CPU.
- A driving source is now a row of `registry/driving-sources.json`, not a
  code path: the row names the reader, the grid family, the state kind, and
  the map from canonical roles to whatever that source calls them.
  `--source ROW` selects it and `rw_mpas_lbc list-sources` prints the
  registry; `--source-registry FILE` merges rows by name. Three rows ship. A
  frame with sixteen variables renamed, driven through a row that exists only
  in a JSON file, produced a boundary file bit-identical on all seven fields
  to the built-in spelling; so did a frame in this program's own history
  spelling and container, read through `--parent-grid`.
- `rw_mpas_lbc --coincidence-snap no` measures the transfer operator's own
  accuracy at a coincident point instead of taking the snap's word for it.
- The RUC land-surface column is parametric over WRF's admitted soil-level
  counts {6, 9}, on the host and on the device. The geometry, the ingest
  remap and the CUDA column all size themselves from the profile instead of
  a literal nine. Both geometries are oracle-matched to WRF 4.7.1 real.exe
  ZS/DZS, including the float32 artifact DZS(7) = 0.4999999 at nine levels.
  Six levels carries NO bit-exactness claim: there is no six-level Fortran
  oracle, and the previous nine-level blocker becomes a warn saying so. The
  surviving refusal, a count outside {6, 9}, is byte-identical before and
  after (sha256 805e6e5433...c97a1828).
- A real `mp_physics = 28` run initializes its aerosol state from WRF's
  global monthly water/ice-friendly climatology
  (QNWFA_QNIFA_SIGMA_MONTHLY.dat) instead of thompson_init's synthetic
  CCN/IN profile, because a forecast started from one is a different
  forecast. The synthetic profile survives as a named fallback, announced in
  the receipt. The dataset is opt-in to stage: `gpuwm fetch-tables --wif`
  (with `--wif-only` and `--wif-root DIR`) puts it under `~/.gpuwm/wif` on
  the coefficient tables' SHA-256 contract.
  `wif_climatology_path` and `mp28_aerosol_source` are the config fields,
  configs/demos/mp28_wif_climatology.toml is the demo, and the cyclic
  bilinear regrid and the IFV=5 intermediate reader are Rust.
- A run's report.json says which aerosol dataset the run used, in three
  states rather than one: a scheme with no particle-number fields carries NO
  KEY AT ALL, so its report is byte for byte what it was; a searched-and-empty
  mp=28 run says so with the search that came up empty; and a writer holding
  no ingest receipt records "not recorded" rather than "no dataset". No schema
  version moves, so every stored receipt and hash stays valid.
- `bl_mynn_mixscalars` leaves MYNN's single-value option identity and is
  admitted at {0, 1}. At 1, MYNN's own five qn-family tridiagonal solves
  and the DMP_mf updraft-flux accumulation run on the aerosol-aware
  Thompson species nc/ni/nwfa/nifa, on CPU and on device. The frozen DMP unit
  stays frozen: the sibling kernel is the pinned one plus complete tagged
  lines, and a test strips them and compares bytes.
- A pre-commit hook asks the index the question the line-ending gate asks
  HEAD, so a patch script that rewrites a whole file's line endings for a
  two-line edit is caught at commit time rather than at release time.
- `rw_mpas_mesh --cull-parent` cuts a limited-area mesh out of an existing
  global grid or static file instead of generating one, byte-matching the
  native MPAS-Limited-Area v2.2 cull for the same region, down to the mask
  rings, the parent-subset ordering and the METIS graph file. `--region` is the crate's existing Shape row (cap, lat_lon_box,
  polygon) as pure JSON, so a new region is a row and never a code path.
  Byte-identical against three pinned native culls from both the test harness
  and the exe. One documented divergence: stored-0 connectivity maps to 0
  rather than the native wrap to the last index, and the receipt records when
  a parent would expose it (neither pinned parent does).
- rw_mpas_static admits a culled mesh. The blanket outermost-ring refusal is
  lifted for the reason it named: the native mpas_in_cell containment is
  ported and gates every source pixel whose nearest cell is on the outermost
  ring, so a pixel whose true owner was culled reaches nobody instead of the
  rim. The culler's sentinel geometry is admitted with every violation
  named.
- `rw_mpas_convert --window mesh` derives the render window from the mesh, so
  a refined core anywhere on earth renders at its own resolution. `--window
  focus` was a hardcoded CONUS box: measured through the binary on a
  126,103-cell grid with a 4.53 km core in the tropical Atlantic, focus and
  global both gathered at about 26.6 km mean nearest-cell distance and
  neither saw the core. The refined region is every cell within twice the
  mesh minimum spacing, centred on those cells' unit-vector mean so a core on
  the antimeridian stays there.
- `rw_mpas_convert --compose SOURCES.json` puts a coarse run and each fine
  run on one grid, rendered once, and says which run drew each point,
  composed as data rather than as pictures so no seam comes from the drawing.
- `rw_wrfbatch` writes `<out-dir>/render-georef.json` on a bare run, with no
  flag: per PNG, the image size, the map's pixel rectangle inside it and the
  projection. A panel used to publish its transform nowhere.
- rw_mpas_init writes model lineage. model_name, core_name, version and
  git_version are optional switches defaulting to this engine's own identity,
  so a caller that says nothing still writes a usable boundary source. Never
  "mpas".

Fixed:
- The boundary producer's default source is the incumbent intermediate row,
  so an invocation written before the registry existed produces the same
  bytes; verified byte-identical against the row spelling.
- `gpuwm fetch-bridges` stages `rw_mpas_lbc`. Its source has been in the tree
  since the limited-area lane landed and every release cut built it, but no
  published bundle carried it, so the only way to obtain the binary was a Rust
  toolchain and a source checkout. It is the other half of the limited-area
  route -- `rw_mpas_mesh --cull-parent` cuts the mesh, this produces the
  boundary series that mesh is driven by -- so the cull shipped output nothing
  could run. Bundle roster, contract marker, resolver row and doctor line all
  added; the binary answers `--abi` and `--version` like its four siblings.
- The gpuwm wheel ships `docs/mpas-seam.md` as package data, at
  `<site-packages>/docs/mpas-seam.md`. An external consumer verifies the
  physics seam by hashing sixteen engine files keyed on repository paths and
  runs that one key set against both a checkout and an install; fifteen keys
  resolved under both and this one resolved under the checkout only, because
  no distribution placed the document anywhere. The install path is the
  repository path so the same key answers for both roots.

- The generator's absolute dvEdge floor re-anchors from 7,500 m to 200 m
  (ruling 2026-08-25). The old anchor guarded the port's retired rtol 2e-5
  load check and refused meshes the port loads and runs, the published
  x4.163842 at its measured 1,170 m included; the new floor is 115x the live
  check's measured 1.732 m absolute tolerance, and it survives as a
  noise-fraction bound because a stored dual length is quantisation noise
  long before the loader objects. The refusal names the surviving breakages,
  the dvEdge/dcEdge >= 0.02 gate is now stated as the port's own
  DualEdgePolicy admission floor, and a dislocation mesh is refused naming
  DualEdgeAdmissionError and the measured TRiSK amplification. The static
  receipt's FP32 agreement reading mirrors the live check (rtol 9.54e-7 plus
  the absolute floor) and reports tolerance use instead of a dead boundary.
  Measured: a generated 654,432-cell pair with 12,732 edges past the retired
  bound, worst absolute 0.634 m, is accepted whole by the live port loader,
  and x4.163842 (dv/dc 0.0336) clears the default emit gate. Evidence:
  evidence/2026-08-25-stale-guards-engine/.
- The mesh-door frame-staleness watchdog now compares the frame cut against
  the MPAS port's live ARWEN_BUILD_COMMIT (read from the port checkout)
  instead of the sizing table's own frozen measured-against record, which by
  construction predates the cut and kept the watchdog green through two pin
  moves (stale-guard audit 2026-08-25, finding 2). The corrected watchdog is
  deliberately red on this tree until the frame and every residue are
  re-measured in one session at the live pin (finding 1, needs the RTX 5090).
- The mesh door's short-edge verdict now states the MPAS port's live
  contract: the 1.73 m FP32 storage atol and the 0.02 dvEdge/dcEdge
  admission floor, instead of the rtol 2e-5 / atol 0.0 comparison the port
  retired on 2026-08-23 (stale-guard audit 2026-08-25, finding 3). The static
  engine's fp32_metric_agreement receipt now reports the live readings, and
  receipts measured against the retired contract report NOT MEASURED instead
  of being quoted.
- gpuwm downscale sizes the standalone child with the live affine
  envelope -- the same estimate_experiment arithmetic the domain wizard
  and gpuwm check price with -- instead of the retired flat reserve and
  1.75x multiplicative floor (stale-guard audit 2026-08-25, finding 4).
  Measured on the RTX 3080 10 GiB through the real doors: on a real
  386x308 12 km rte-rrtmgp parent the retired pair admitted 282x282
  while the card ran the affine fit's 342x342 whole (47% more child
  area). On a legacy-RRTMG parent the disagreement reverses (retired
  324x324 against the live envelope's 258x258) and both children ran,
  so the retired pair's error has no consistent sign -- which is why it
  is replaced by the affine model rather than re-tuned. Evidence:
  evidence/2026-08-25-stale-guards-engine/finding4-README.md.
- RUC ground heat flux was about five times too large at six soil levels:
  ruc_soil_finalize computed dzstop from the literal nine-level spacing.
  The frozen digest pin moves for it.
- The frozen-window digest gate was red on Linux and green on Windows from
  pristine source, and the cause is the C library, not this tree. Measured
  with the compiler excluded: every value handed to `powf` is bit-identical
  across the two boxes (0 of 36,000 differ), and the differing library
  answers move 1,684 of 36,000 latitudes by at most 4 ULP, 1.42e-14 degrees.
  The focus window now carries a per-platform digest row keyed by the (OS, C
  environment) pair; adding a row is additive, no existing row is ever
  edited, and an unmeasured platform REFUSES and prints the digest it took
  rather than passing silently. The portable half is a geometry gate holding
  the grid corners to 1e-9 degrees everywhere. The global window needs no
  table and says why: its coordinates call no transcendental at all.
- The published pages catch up: CONFIGURATION.md gains the one way to supply
  your own mp=28 aerosol, PHYSICS.md and the manual drop MYNN from twelve
  pinned knobs to eleven, and the generated CLI reference is re-rendered for
  the three table-staging flags. One page told a reader to run `gpuwm
  import`, which is not a command; the door is `gpuwm import-namelist`.
- One authored source file had carried CRLF unbooked since it landed, which
  held the line-ending gate red. Normalized to LF: 4,563 CR bytes removed and
  nothing else.
- One cell with no valid source pixel no longer costs a whole static. MODIS
  albedo is land-only, so at the Antarctic sea-ice margin the land-use
  archive calls a cell ice while every albedo pixel in it is fill, and one
  cell refused a 128,019-cell mesh that took 435 s to generate. A cell with
  no sample now takes the mean of the neighbours that have one, spreading
  over the mesh's own connectivity, and mask-excluded cells conduct a value
  without keeping it, so an ice margin across open water is still reached.
  The refusal is not removed, it moves to the case it was written for, a
  field with no valid pixel anywhere to carry from, and it names the
  stranded cells. Nothing that builds today changes: a 112,676-cell static
  rebuilt with the patched binary is byte-identical, 215,446,200 B, sha256
  4cfa3008...c902f10.

Changed:
- The tropical root clock (30 s inside the tropics) is re-justified on the
  instrument that replaced the one it was found wrong on: re-run at the
  motivating case under the corrected v1.1 co-located |w|/dz monitor, the
  un-halved 60 s clock peaks at vertical CFL 1.997 against the halved
  clock's 0.976 -- twice the vertical Courant limit, sustained through
  6 h. The guard stays, and its cost is corrected to the +73% forecast
  wall measured here rather than the +22% previously quoted.
- POOL_SLACK_FRACTION keeps its 0.20 with a post-#310 re-measurement
  recorded beside it: legacy-lane retention has fallen and now shrinks
  with grid size (0.213 / 0.150 / 0.019 at 60x48 / 110x88 / 290x232),
  so the value is no longer a worst-case bound. Not re-fitted, because
  the new rows are 2 h and the calibration rows are 6 h; the like-for-like
  protocol is a named follow-up.
- The RRTMGP 128-layer ceiling now names its owner: the LW Planck-source
  kernel's fixed per-thread pfrac[128]. planck_sources refuses a taller
  column loudly before any launch (it used to come back as uninitialized
  source arrays that read as valid radiation), the LW door keeps its
  refusal with the owner named, and the SW door's copy of the ceiling is
  retired: the SW solvers compile RRTMGP_MAX_LAYERS to the run's own nlay
  and hold no fixed per-layer storage, proven by a 160-layer transparent-
  atmosphere run reproducing the analytic direct beam (stale-guard audit
  2026-08-25, RRTMGP nlay adjudication).
- Stage-IV ZERO_TOLERANCE_MM re-justified in place: four archive objects
  (01h summer/winter/spring, 24h; 1,964,589 dry cells) decode every dry
  cell to exactly 0.0 through the current Rust path with
  clamped_negative_cells 0, so the clamp is unexercised on the measured
  archive; it stays as the boundary between a dry observation and a
  deleted one, with the receipt counter keeping any file that exercises
  it visible (stale-guard audit 2026-08-25).
- The mesh generator stops rebuilding one polygon on every evaluation. The
  resolution field has one definition with the per-shape constants solved
  once, and three pure hoists come off a measured profile in which 56.6% of
  all cycles sat inside libm. Measured on 24 cores, CPU only, on a
  112,676-cell parent spec: 322.4 s to 166.9 s.
  EQUIVALENCE IS THE POINT -- the emitted grid is byte-identical to the
  baseline (sha256 36406384...39a7d7b7, 154,154,208 B), held there without a
  run by two gates: the prepared field against a verbatim copy of the
  retired arithmetic by f64 bit pattern, and the new filtered orientation
  sign against the exact predicate. A mesh registry pins grid files by
  SHA-256, so one moved bit makes every registered digest downstream
  unreproducible.

## 2.5.7 (2026-08-25)

Fixed:
- The default model top moves from 100 hPa to 50 hPa (p_top 5000 Pa, the
  WRF Registry default), taking the damp_opt=3 sponge base from ~11 km
  to ~15.6 km AGL. The 100 hPa lid damped anvil-layer updrafts and
  smeared convective structure on a bare default run; measured on a
  3 km HRRR A/B vs MRMS, the 50 hPa lid restores updraft tops and core
  counts at equal-or-better FSS and no VRAM or wall cost. Emissions are
  bounded per source: a source whose certified ladder stops at 100 hPa
  (GFS pgrb2, 20CRv3 NetCDF) keeps its covered top.
- A tree with [relocation.containment] can be resumed: the checkpoint
  records the parent slides in the move chain, the slid parent is put
  back with its own initializer, and a restore is no longer judged by
  the per-move steering cap.
- A resumed or branched run of a moving nest keeps the precipitation
  totals a move shifted into the domain edge; the restore migration
  zeroes only what no relocation carries.

The moving-nest restart fixes were contributed by Roch.

## 2.5.6 (2026-08-25)

2.5.5 was tagged but never published: the public snapshot commit lost four
vendored Rust build files to the build-directory ignore rule, so the release
cut could not build the bridges and nothing reached PyPI. The tag stays where
it is, because tags here are forward-only. Everything 2.5.5 carried ships
here.

A run that narrates its own life, and radiation that fits the card.

New:
- Nest lifecycle in the per-step stream (schema gpuwm.step-log/v3):
  nest_spawned, nest_retired, nest_rearmed, nest_moved, containment_moved
  and track_fix, each with the domain, its step count, the valid time and
  the position. v1 and v2 streams replay unchanged.
- The delivered render layout gains an episode segment
  (out/domain/episode-NNN/product/valid-day), so two lives of one nest
  publishing the same valid time no longer overwrite each other.
  Non-episodic runs keep their exact paths.
- gpuwm branch: a what-if door that seeds a new run from an existing
  checkpoint, with --set overrides. A pinned setting is refused by name
  with the changeable list in the sentence.
- The MPAS seam publishes refl10cm and q2 in the default history stream,
  computed inside the history step's own microphysics call, and the
  rw-mpas converter maps REFL_10CM and Q2. Reflectivity products read the
  model's own field instead of a hydrometeor fallback.

Fixed:
- The vortex track CSV is tail-safe while the run writes it: every row
  reaches the OS at its newline, and the header is readable before the
  first fix.
- The legacy-RRTMG batch chunk width sizes itself to the device it runs
  on instead of paying the largest card's workspace everywhere. Results
  are byte-identical at any width. Measured 2026-08-24 on an RTX 5070 Ti
  (gpuwm-hex x1.40962): peak process memory 6,724.0 to 5,604.0 MiB.
- Grell-Freitas column arrays leave the fully-resident kernel frame on
  the MPAS column-batch path, shrinking the port's fixed device
  reservation.
- Offline downscaling from a WSM6 parent works; the transport table
  refused a scheme its own contract admits.

## 2.5.4 (2026-08-24)

Two-way nests and nest slots that open and close on the storm.

New:
- Two-way nest feedback. feedback = 1 runs WRF's feedback transaction
  with WRF's parent smoothers, pinned bitwise to the Fortran, on both
  the native and prepared-tree runners. Off by default; feedback = 0
  runs are byte-identical to 2.5.3.
- Pressure spawn and retire triggers. A dormant nest slot opens when a
  low deepens and retires when it fills.
  configs/cyclone_nest_slots_12km.toml ships three slots with rearm.
- Storm-following upgrades. The tracker centres on the 850 hPa height
  minimum inside a 50 km disc and refines through a finer nest; a
  containment nest slides the mover's parent while the mover stays
  earth-fixed.
- Statics corridors. --statics-corridor seals child-resolution statics
  at preparation time, so a prepared bundle hosts a moving nest with no
  runtime geography.
- A track file. [relocation.track] writes the vortex track from the
  same fix that steers the nest. output_level picks the reported
  surfaces without touching what steers, and the eight-surface cap is
  gone.
- Tracker defaults: level_hpa 850, radius_km 50. Sea-level tracking is
  level_hpa = 0; the threshold bands are disjoint, so a config that
  means one and reads as the other refuses at load.
- --cycle latest resolves for every registered source from its own
  registry row. A reanalysis has a latest like anything else.
- A prepared-tree run with a live follower checkpoints and resumes,
  including across a move, byte-identical.

Fixed:
- A moved nest keeps its surface-radiation carriers; the shipped
  radiation defaults no longer refuse at the first move.
- Spawn receipts append per boundary instead of rewriting the ledger.
- run-plan --estimate on a missing config path refuses by name instead
  of a traceback.
- A resumed run publishes real outgoing longwave instead of zeros.
- A forecast whose nest moved can be resumed; the move chain replays on
  restore.
- Sizing and pace estimates work on an install with no GPU runtime.

Two-way feedback, the storm-following upgrades, the track file and the
OLR restart carry were contributed by Roch.

## 2.5.3 (2026-08-24)

Small cards and living nests.

New:
- Living nests. Nests spawn on storm triggers, follow their storm, retire,
  and rearm during a run, on the GPU, with rendered products per episode.
- Restart. Runs checkpoint and restore with bit-identical continuation,
  proven across multi-segment splits.
- Expected pace before you commit. run-plan and check state the expected
  seconds per step and wall clock for your domain on your card, from
  measured rates, and name the domain size that would run resident.
- Latest-cycle fetches ask NOMADS first and route around lagging mirrors
  automatically. Receipts name the host that served every file.
- WPS intermediates from ECMWF open data and AIFS. Ordinal soil layers
  decode through the standard tables, so both sources are table work.
- Per-file download progress with real byte counts.

Fixed:
- VRAM sizing now prices the whole run, including the radiation working
  set that fires after the first minutes. A card that passes the check
  finishes the run.
- Streamed initialization no longer keeps a resident copy of the domain,
  which paged 10 GB cards into unusability.
- check accepts every streamed domain that go can run. The alloc gate is
  tile-aware.
- A stale or spent VRAM budget refuses up front with every term named,
  instead of failing after the download and preparation.
- First-step status reports what the engine is actually doing.
- Fetch receipts claim every file they write, and an interrupted forced
  refetch can no longer leave a stale receipt beside rejected data.
- A run that failed past the fetch comes back without re-downloading. The
  fetch verifies and skips what is already on disk and the receipt says
  which files it skipped, preparation is reused when nothing feeding it
  moved, and superseded forecast output is moved aside rather than deleted.
- The standalone RW-WPS wheel ships the nest lifecycle module.
- The ERA5 model-level route is documented alongside every other source.
- A projected or Gaussian grid refuses by name instead of minting a
  mis-georeferenced lat-lon intermediate. Uniform lat-lon output is
  unchanged.

## 2.5.2 (2026-08-21)

New:
- Radiation is about 1.9x faster at the shipped default column chunk
  and the allocation estimate drops roughly 1.5 GB, with output that is
  byte-identical to 2.5.1. Measured on an RTX 5070 Ti at chunk 3125:
  187.96 and 187.77 s before, 96.69 and 96.50 s after, first run after a
  kernel-source change discarded. The gain depends on the chunk, from
  1.66x at 8192 to 5.00x at 512, because the optimised path is flat
  across that range where the previous one was not. Byte identity was
  confirmed on two cards, with perturbed-kernel and perturbed-input
  controls both differing. Contributed by Roch.
- The first over-budget lever moves with the radiation workspace, which
  falls from 2,051,400,000 to 719,650,000 B at `column_chunk = 6250`.
  `gpuwm check` against a 19.5 GiB budget now names `--column-chunk 3125`,
  one halving of the configured 6250, where the same case had to be walked
  back two halvings to 1562 before. The whole chunk ladder flattened with
  it: 6250 down to 256 buys 824 MB where it used to buy 2,293 MB.
- `gpuwm fetch --source era5` writes `era5-cds-retrieve.py` beside the
  request document and prints the one command that runs it. The script
  retrieves both CDS parts and byte-concatenates them into
  `era5-combined.grib`, resolving every path from its own location, so it
  produces the same files from any working directory and under any
  interpreter -- including a WSL python3 reached from Windows, whose
  command is printed with the path already translated. The request's two
  targets are absolute paths under `--out` instead of bare leaf names.
- `gpuwm fetch --source era5 --validate` reports the delivered
  geographic extent -- grid shape, spacing and corners -- and, when
  `--area` is given, FAILS a file whose grid falls more than one grid
  cell short of the requested box on any edge, naming the edge and the
  shortfall. One cell is the provider's own inward snap onto its native
  grid. Without `--area` the extent is printed followed by a line saying
  it was checked against nothing. Two different grids in one file set
  fail: the two requests were not made with the same area.
- `gpuwm doctor` checks that the directory holding the `gpuwm` console
  script is on PATH, and prints the exact line that fixes it. An
  editable or `--user` install writes the launchers where PATH does not
  look, and every command this product prints then answers that `gpuwm`
  is not recognized.

Fixed:
- `gpuwm downscale` runs a full-physics child without `--child-surface-from`.
  The child's land identity and soil warm start are taken from the parent's
  own history frame and put on the child grid by WRF's nest-birth operators:
  categories and the landmask by donor-cell copy, the surface/soil family by
  the Registry's masked land interpolator. Previously the flag was mandatory
  and pointed at a file no command could produce for a config-driven parent,
  so a full-physics child of a `[case_data]` run was unreachable; measured on
  a 12 km ERA5 parent, the same invocation refused before and finishes now.
  The flag stays as the higher-fidelity route -- a derived child inherits its
  parent's coastlines and lakes, which the run warns about once and records
  in `report.json`. The surface is resolved before the parent archive is
  decoded, so a parent that cannot seed a child refuses at plan time instead
  of after the preprocessing.
- `gpuwm downscale --point` writes the derived child config to
  `<out>/child.toml`, inside the run it describes, instead of
  `<out>.child.toml` beside it. `--dry-run` still writes beside `--out`,
  because reserving that directory during a plan would make the run that
  follows refuse, and it now says so.
- Static fields no longer depend on which source window a build happened to
  read. On a geography source that wraps in x, points below the window
  origin were shifted by the global width and shifted straight back by the
  tile interpolator, a lossy round trip in floating point, so two builds
  covering the same ground sampled the source at coordinates differing in
  their last bits. A moving nest whose parent-extent statics corridor
  crossed the source's wrap seam therefore refused every relocation
  (HGT_M differing in 11581 of 39204 shared cells by up to 2.8e-11 m) and
  wrote no frames after the first move. Measured on the same tree after
  the fix: 3519936 cells compared per move, zero mismatches, two moves
  completed, 26 frames.

## 2.5.1 (2026-08-20)

New:
- `gpuwm sources` lists every registered forcing source and what this
  release can do with each one; `gpuwm sources ID` prints one row in full,
  by id or alias. `--json` serves the same document as
  `gpuwm run-plan --sources`.
- Registry rows carry a display name and their credential prerequisites,
  emitted through `--sources`: what must be configured before a source's
  bytes can be acquired, where it lives on this machine, whether it is
  there, and what breaks without it. Existence only -- a key's value is
  never read. A source that needs an account key is one registry row; no
  front end carries an exception table for it.
- run-plan intents are composed from the registry row instead of a
  hardcoded list: icon-eu, rap, rrfs, hrrr-prs, gem-gdps, aifs, aigfs and
  ecmwf-open-data join gfs/hrrr/era5 through the same door, eleven of the
  eighteen runnable rows carrying an intent. The prepared route grows a
  staged fetch->prep->sim chain composed from the fetch
  route's own bound handoff (prep-arguments.json, new beside
  prep-command.txt); --sources rows carry intent_chain and intent_refusal.
  An intent naming an undrivable source is refused with the missing
  registry fact (member set, no acquisition route, no cadence), never a
  hardcoded-list bounce.
- `gpuwm run-plan --sources` prints the source registry as one JSON
  document (`gpuwm.run-plan.sources.v1`): every registered row, its
  coverage, maturity and fetch route, and whether a run-plan intent can
  drive it. A front end picks from the engine's list instead of its own.
- `[output]` selects which variables the wrfout files carry:
  `preset = "full"` (the default), `"minimal"`, `"severe"`, or your own
  `history_vars` / `history_drop` lists; the same table sits inline on a
  `[[domain]]`, so a 1 km child can write surface fields while its parent
  keeps everything. Config resolution warns which render products a
  selection costs, per domain, naming product and variable; the render
  front door refuses those products by name with the remedy. Trimmed runs
  stamp `GPUWM_HISTORY_PRESET` / `GPUWM_HISTORY_SELECTION` /
  `GPUWM_HISTORY_DROPPED`; a default run stamps nothing. Measured on a
  two-domain forecast: 35% off the 3 km parent's frames, 81% off the
  1 km child's.
- `gpuwm render --streamlines` / `--barbs` choose how wind is drawn on
  every product carrying a wind layer. Neither given, the automatic
  per-grid choice is unchanged; `RUSTWX_WIND_STREAMLINES` still works and
  the flags outrank it.
- `gpuwm run-plan --physics-profiles` answers which physics suites run on
  a given source and which that source's own row defaults to, computed
  from the registry rather than a per-source table.
- `gpuwm speedrun` runs a named course and emits a signed record. Two
  courses ship, `regional-12km-6h` and `nested-12km-3km-3h`, with the
  records under `evidence/speedrun/`. See SPEEDRUN.md.
- `era5-l137` is a registered runnable source: ERA5 on its native 137
  model levels, aliases `era5-model-level` and `era5-ml`. It declares the
  Copernicus CDS key as a credential; acquisition runs through the
  provider's own client, so the row carries no gpuwm fetch route and no
  run-plan intent.
- Twenty-two example configurations across eleven sources, two each for
  ecmwf-open-data, era5, era5-l137, gdas, gem-gdps, gfs, hrrr, hrrr-prs,
  icon-eu, rap and rrfs. Each was run end to end -- fetch, prepare,
  forecast, render -- and each header records the domain, the `gpuwm
  check` peak envelope against a 10 GiB budget, and the peak device
  memory and stage timings its run measured. The forecasts were executed
  on a 16 GiB RTX 5070 Ti against a computed 10 GiB budget; these are
  configurations SIZED for a 10 GiB budget, not configurations proven on
  a 10 GiB card. Two need non-default flags and are workarounds, not
  fixes: gdas requires `gpuwm fetch --mode full-file --cadence 1`
  because the default NOMADS subset emits 21 of the profile's 33
  isobaric levels and the fetch door's default cadence is three-hourly
  against an hourly contract; ecmwf-open-data has no one-command chain
  and must be driven as separate fetch, prep and sim stages.
- The prepared-tree runner and `gpuwm stream` refuse a delayed
  `[[domain]] start_time` by name and point at `gpuwm run`; both restore
  every domain from a prepared cache and cannot activate one mid-run.

Fixed:
- A run that reaches its stop tick reports success. Restarting from a run's
  own final checkpoint is completion, not an error: the route finalizes and
  exits 0 on both the tree and single-domain paths. A checkpoint from beyond
  the stop is still refused, and that refusal names the breakage and the key
  to change.
- The stretch after the last model step -- drain, device synchronize,
  trajectory digest, receipts, and the hash pass over every emitted frame --
  publishes a named finalizing heartbeat, and the frame hash counts itself
  off frame by frame. It used to publish nothing, so on a large run the
  supervisor's stale-integration watchdog killed a worker that was finishing
  normally. Emitted frames are hashed once per run, not twice.
- A restore the worker refused is no longer relaunched. Three fresh processes
  rediscovering one refusal also overwrote each other's failure capsules,
  which destroyed the evidence for the first failure. Crashes elsewhere still
  get their fresh-process recovery attempts.
- The no-fetch-route refusal pointed at `gpuwm sources`, which did not exist.
  It does now, and the sentence names the one-row spelling too.
- A cargo build that cannot finish (an artifact another process holds open, a
  build lock, an incomplete vendor set) refuses by name with the artifact and
  a remedy, instead of relaying a compiler warning wall under "could not
  decode/merge forcing inputs". `gpuwm check --json` always writes a JSON
  document to stdout; a failing input preflight used to leave it empty, so a
  build error reached a calling program as a JSON parse error. The preflight
  also tells a decoder that could not be BUILT from data that is wrong, and
  says which failures below it measured an empty catalog rather than inputs.
- The per-thread local-frame table is a ceiling over named compile platforms
  instead of one box's reading; six rows were under-pricing sm_86 cards by up
  to 0.17 GiB of backing store, and one kernel module had no row at all.
  `gpuwm check`'s binding-phase line no longer charges the CUDA context and
  backing store twice -- on a loaded 10 GiB card it called a configuration
  1.61 GiB over budget that `gpuwm go` ran to completion.
- A delivered render tree contains only products: the Rust renderer's working
  store moved out of the output directory into a sibling scratch directory,
  cleaned on exit, so it cannot be left among the pictures and cannot race a
  copy of a tree being rendered into.
- The `gpuwm domain` next-steps pointer and interactive prompt derive their
  credential line from the registry row instead of a hardcoded per-source
  branch, so any source that declares one gets the pointer.
- The MPAS render-bridge Python pair refuses to run as a product path, exits
  78 and prints the Rust command that supersedes it; it stays reachable as
  the parity reference behind an explicit flag.
- VRAM sizing charged a run's CUDA context and kernel backing store
  twice -- once inside the peak envelope and again in the budget it was
  compared against -- so a 10 GiB card was refused the smallest domain
  the hrrr ladder can express. The same grid now fits with room to spare.
  A 550 x 550 x 49 full-physics run is admitted on a 15.24 GiB card
  instead of refused; the streamed-admission pin moves to 576 x 576, the
  smallest domain that genuinely does not fit, and the newly admitted run
  is pinned as its own statement because it is user-visible.
- A GRIB2 source with no hybrid vertical coordinate reads again. The
  Section-4 pv octets became a required column of every inventory, so a
  single-surface record such as a sea-surface temperature failed a round
  trip with "missing required columns ['pv']" with no vertical
  coordinate to lose. pv is required of hybrid-level records only
  (level_type 105, 118, 119), where a decoder that cannot state them is
  refused by name, naming the level type, the column and the rebuild.
  Both directions are tested.
- The standalone RW-WPS preprocessing wheel builds again. Reading a
  config imported the forecast output writer: the history vocabulary was
  built at module scope through a function-local import of the wrfout
  writer, which pulls in the supervisor and netCDF4, so importing a
  preparation module dragged in the whole executor. The metadata table
  moves to the data module that owns it, and the boundary is pinned
  against the artifact.
- No shipped file carries a developer's absolute paths. The shipped
  docstrings take a `<user>` template and the affected tests assemble
  their fixtures from a user fragment, so the strings the functions under
  test receive are unchanged real spellings.
- The domain wizard measured your card and then priced its kernel memory
  against a different one, disagreeing with `gpuwm check` on the same
  machine. Both doors price the card that is present.
- The Linux memory envelope charged no allowance for pool retention and
  sat below the measured peak of every instrumented forecast. The
  allowance follows the radiation scheme rather than the operating
  system, and is charged wherever that scheme runs.
- The CUDA context was priced from one 2026-07 reading of one card,
  which under-charged a Linux RTX 5090 by 215 MiB. It is measured on the
  card in the machine.
- On Windows, free VRAM was read from an instrument that counts memory
  the driver would have to evict from your desktop -- 5.7 GiB of a
  10 GiB card. Budgets use the smaller, machine-wide figure.
- Grell-Freitas cumulus receives the boundary-layer forcing WRF's own
  cumulus driver hands it; the engine built those rates every step and
  fed the scheme zeros. Every boundary-layer scheme supplies them now,
  not only YSU. Measured, one forecast hour from the same analysis with
  YSU: at 12 km domain-total convective rainfall moves 12.8% and cloud
  water 22% of its own spatial variation, with single points moving up
  to 2.5 m/s in the 10 m wind and 1.2 K at 2 m; at 3 km, where the
  scheme damps itself, convective rainfall moves 3.3%. No shipped
  physics profile selects Grell-Freitas, so a default run is unchanged;
  the two shipped configs that do select it both pair it with YSU and
  change. On the MPAS seam shallow convection is on, dx is per column,
  and the advective forcing pair arrives too; the ArWen dycore exports
  no advective pair, so RTHFTEN/RQVFTEN stay zero there.
- A nest that starts later than the experiment now runs to completion.
  Its first history frame is that domain's own analysis frame and carries
  no REFL_10CM, exactly as d01's frame at t = 0 does; every frame after
  it carries the field. The 2.5.0 upfront refusal named this crash and
  retires with it.
- `gpuwm check` sizes the selected PBL scheme against its kernel column
  bound. A 130-level YSU configuration used to pass check and both
  preparation stages, then die on the first physics call.
- The health gate's potential-temperature ceiling follows the configured
  model lid instead of a flat 600 K, so a deep-top initial state such as
  a 20 hPa top is no longer refused as runaway; the refusal names the
  lid-derived threshold it applied.
- Render: a 2-D variable you added to your own file is drawable by name
  with `--products var:`. The science import kept only the fields its
  fixed catalogs named; every stored `(Time, south_north, west_east)`
  plane is now imported, which also makes the surface suite renderable
  with no new product code.
- `gpuwm doctor` checks the staged Rust artifacts against this release's
  pins, not just that they exist and launch; a stale `~/.gpuwm/bridges`
  is a blocking finding with `gpuwm fetch-bridges` as the remedy.
  `python -m gpuwm.doctor` prints the report and returns the console
  script's exit code instead of silent 0. Slow probes name themselves on
  stderr while they run; the report itself is unchanged.
- The mapped preparation route prints its ready-to-run forecast command
  with all three digests filled in, like the GFS route; it used to finish
  silently. `--proof-sha256` refuses by name, states that it wants the
  sha256 of the proof.json file, prints both digests, and says outright
  when the value given is the document's own `proof_content_sha256`.
- The generated CLI reference covers every installed console script;
  `gpuwm-member-prep` and its twelve options were named by no document.
- `gpuwm go` no longer redraws the frame its early render already
  published, and `time to first plot` is checked against the pictures in
  the render tree before it is printed.
- `tools/check_case_token_leakage.py` scans the zones it names and
  reports the file count; passing a subtree used to scan zero files and
  always print green.
- `tools/ci_test_replay.py` builds its venv with setuptools, so the
  replay runs on Python 3.12+.

## 2.5.0 (2026-08-19)

New:
- The whole data path runs on Rust by default: GRIB1, GRIB2 and NetCDF
  decode, composition, regrid, the static warp, `wrfinput`/`wrfbdy` and
  product NetCDF writing, and verification renders. Compiled crates on
  a bare install, no toolchain. Each leg keeps a named Python workaround,
  none the default.
- `gpuwm domain --source` reads the source registry: any registered id or
  alias, cadence from its own row, a refusal naming what a row is missing;
  a new row needs no wizard code. See [SOURCES.md](docs/public/SOURCES.md).
- Regional sources carry their native grid as table data: a domain outside
  coverage refuses at plan time, naming the offending point and covered
  window. The fitted ladder, the `--area` gate and the suggested box read
  one definition.
- `gpuwm domain` emits a runnable `gpuwm fetch` step for every routed
  source; one with no public bytes emits the same geometry, the
  hand-staging step named.
- The forecast horizon each source declares is enforced at the wizard:
  `--hours` past it refuses naming the horizon, `--forecast-start-hour`
  refuses on a row with no leads.
- The wizard says when the source, not the card, stopped the domain search;
  a saturated fit cannot look comfortable.
- Mapped sources decode and compose inside `gpuwm_mapped_engine`: a bare
  `gpuwm prep --source <id>` reads its bytes with no decoder flags and no
  subprocess per file.
- The 20CRv3 exact-member route composes in the Rust engine: the sealed
  member rides the composition manifest into every frame and receipt, and
  a bare member prep resolves no subprocess decoder; `ENGINE_GAPS` is
  empty, every prep door decoding on the engine by default.
- `--mapped-engine rust|python` on `gpuwm prep --source mapped` and
  `gpuwm.mapped_direct`, `GPUWM_MAPPED_ENGINE` for library callers. `python`
  is the documented workaround; an unrecognised name refuses.
  `gpuwm_mapped_engine capabilities` prints what the build implements.
- The mapped parity battery: seventeen real-byte sources decoded by the
  Python engine, reduced to per-field SHA-256 the Rust engine must
  reproduce exactly; goldens are extracted, never authored.
- The static-field builder runs on Rust by default: everything in
  `gpuwm/static` processing bytes executes in the `static-fields`
  cdylib, byte-identical to the numpy reference. `GPUWM_STATIC_PYTHON=1`
  falls back and says so.
- One run, one folder: `go`, `sim` and `render` each claim a timestamped
  `run-...` subdirectory, a rerun gets a second tree, `latest-run.txt`
  names the newest. Default-on; `--run-stamp off` is the workaround. See
  `docs/run-output-folders.md`.
- Adding a model whose mapping can be written down is table work: three
  SHA-256-pinned JSON documents in `gpuwm/authorities` plus one adapter
  row. Every source below is proven on real production bytes: prep, GPU
  forecast to PASS, renders.
- Ensembles enter as one SHA-256-pinned `rw-wps.members.v1` document plus
  a registry row; members are verified at decode on the GRIB ensemble
  identity octets, and a mean or spread claimed as a member refuses by
  name.
- Cross-source composition: a field may come from another packaged
  source's decode (soil, terrain, the land mask under an AI atmosphere).
  The mapping declares each gap `composition_bound`, `field_sources` binds
  it to one contributor, same-grid borrowing only; every other shape
  refuses by name.
- New initialization sources, each `gpuwm prep --source <id>`: `gdas`,
  `rap`, `rrfs`, `icon-eu`, `gem`, `ecmwf-open-data` (aliases `ecmwf`,
  `ifs`), `aifs`, `gefs`, `aigefs`, and `hrrr-prs`, the public hourly
  wrfprs product beside the native `hrrr` route. `aigfs`, NCEP's
  GraphCast AI forecast, refuses a solo init naming the seven canonicals
  it does not publish; its hybrid profile borrows them from the same
  cycle's GDAS. `20crv3-cf` reads the 20th Century Reanalysis v3, the
  only route starting a case before the satellite era.
- Mapped NetCDF sources may span many files, publish a quantity on fewer
  levels than another, reuse a variable name across quantities, and stack
  an N-layer quantity in one variable.
- `rw_ensbatch`, the first ensemble renderer on the Rust path: N member
  wrfout sets in, mean, spread and paintball panels out through
  `rw_wrfbatch`'s projection, basemap and PNG writer; bitwise-identical
  members refuse, so an all-zero spread is never published as measured.
- `gpuwm fetch --source` accepts every registered source with public
  bytes: hosts, key grammar and file sets are table data read by one
  engine, whole files pooled by default (`--fetch-workers N`, NOMADS
  capped at 2), each output directory carrying the ordered `--input-list`
  and a runnable `gpuwm prep`. See [DATA.md](docs/public/DATA.md).
- The wrfout writer runs on the Rust classic NetCDF writer by default;
  `GPUWM_WRFOUT_WRITER=python` is the named workaround, and a box without
  the library refuses naming the build.
- `pip install gpuwm` also installs `gpuwm-data`, the companion carrying
  the RRTMGP and Thompson tables past PyPI's 100 MiB cap; pinned `==`, a
  skewed companion refuses by name.
- Building a wheel refuses while `bridge-pins.json` declares no release
  and no platforms; `GPUWM_ALLOW_UNPINNED_WHEEL=1` builds an unpublishable
  dev wheel.
- `gpuwm go` persists stage timings to `<outdir>/events.jsonl`,
  default-on, and prints time to first plot.
- `[ingest] soil_texture_downscale = false` restores stock-WRF soil behaviour
  byte-for-byte.
- `gpuwm render --layout flat` restores the single output directory 2.4.1
  wrote.
- `docs/public/DOWNSCALE.md` opens with the fresh-box to downscaled-nest
  walkthrough, verified command-by-command against 2.4.1.
- `gpuwm spectral` says at which physical wavelengths a forecast and its
  reference disagree, and how: banded power, amplitude ratios, correlation,
  coherence, phase error, rotational versus divergent kinetic energy.
  CPU-only; thresholds only from a predeclared known-good population.
  See [SPECTRAL_VERIFICATION.md](docs/public/SPECTRAL_VERIFICATION.md).
- The observation battery's remap runs on the `obs-regrid` crate by
  default, both registered operators, with `GPUWM_OBSREGRID_PYTHON=1` the
  scipy workaround.
- The high-resolution static warp is Rust: decode, mosaic, projection and
  warp run in the `static-fields` crate by default, honouring
  `RasterPixelIsPoint`; `rasterio` and `pyproj` are the fallback, in an
  extra.
- `wrfinput` and `wrfbdy` are written by the Rust classic-tape writer through
  one shared facade, with a dual-write verification door.
- `gpuwm spectral`'s field reads, the preflight orography read and the
  gridded radar and GOES products run on the Rust NetCDF path; every
  remaining Python NetCDF reader is declared, and an undeclared one fails
  a gate.
- `docs/manual/` is the scientific manual: nine chapters from WRF lineage
  to the operational envelope, every number citing its receipt.
- `tools/build_rw_wps_release.py` stages the modules the standalone RW-WPS
  wheel imports, and names the executor it refuses.
- Downloads count their bytes: `gpuwm fetch-tables` and the fetch routes
  print a running size line per object, and a redirected front door streams
  lines instead of buffering to exit.
- `gpuwm doctor --source` accepts every registered source id: a registry
  route answers in one line, deferring to the engine's verdict, so an
  unbuilt binary is one gap with one remedy.
- After an upgrade, `gpuwm doctor` opens with what changed since the
  last version run on this box; `gpuwm run --help` points at it.
- `docs/public/WITHOUT-A-GPU.md`: install to verified preparation with no
  card, every command measured, no flags: the bare defaults fall to CPU
  and say so.
- `docs/public/GLOSSARY.md` maps the product's vocabulary to the WPS nouns
  a WRF user knows: prep is WPS plus `real.exe`; sim is `wrf.exe`.

Fixed:
- README no longer says "No moving nests": storm-following relocation
  shipped in 1.8.0, so the nesting limits name what is absent (vertical
  refinement, adaptive time step).
- `gpuwm render --engine auto` refuses instead of quietly drawing weather
  fields on matplotlib, and the verify door's synoptic panels come from
  `rw_wrfbatch`.
- The shipped public pages name the detailed WRF-comparison case by its role, not
  an internal token and date; the reproduction section keeps every literal
  artifact name a reader types.
- A run's pictures land in one folder: `gpuwm go` claims its run folder
  once and renders into it, and `gpuwm render` pointed inside a `run-...`
  folder adds no second stamp level.
- A rendered product stays in the `<domain>/<product>/<valid-day>` tree
  past Windows' 260-character path limit, and `--pair` reads frames at any
  path length.
- A wheel install reported a missing mapped decode engine no command could
  supply; `gpuwm fetch-bridges` stages it, offered by `gpuwm doctor` only
  when the bundle carries it.
- The release battery compiles and tests the static-fields crate and
  mapped-engine workspace, and their goldens carry no developer's absolute
  path, so a second box reproduces them.
- `gpuwm domain` sizes to the largest domain one source crop can feed: the
  180-degree single-crop limit is a fit bound, not an exit-2 refusal.
- The Windows VRAM envelope is measured, not a 1.75x guess: a forecast
  fitting a 10 GiB card passes bare `gpuwm check` and `gpuwm go`, and
  the oversized-domain advisory ranks lighter profiles by priced envelope
  instead of pointing `--vram-gib` at the gate that refused.
- `gpuwm fetch` hint tables in `experiment.toml` accept every routed
  source and refuse an unrouted spelling with the vocabulary that would
  have worked.
- The staged DA sweep plan names its case by `${CASE_ROOT}`
  (`GPUWM_DA_SWEEP_CASE_ROOT` or `--case-root`), carries no machine path,
  and refuses an unbound case before a card is taken.
- Grid-relative winds on a declared Lambert source rotated with the
  projection module's longitude wrap instead of the mapped path's, moving
  every component by a few ULPs.
- A mapped frame that cannot initialize WRF is refused where built,
  so the reported frame is the faulty one; a vertical-coverage refusal
  no longer spells its levels through numpy's scalar repr, which varied by
  numpy major.
- The mapped engine ran on one core, copied the decoded field set five
  times and hashed every frameset byte twice; it runs concurrently now,
  every output byte unchanged at every worker count.
- The two engines refuse alike: undecodable GRIB bytes raise `ValueError`
  on both, and a broken mapping document runs one grammar validator before
  dispatch.
- `python -m build` swept staged native binaries into the sdist;
  `MANIFEST.in` prunes them and `setup.py` refuses an sdist of a staged
  tree.
- `tools/rw_wps` is covered by the vendored-registry gate, comparing each
  locked version against the vendored manifest, not crate names only.
- The release machine-path scan read files by suffix, missing `.cu`,
  `.cuh`, `.wps` and a JSON-escaped Windows marker; the bytes decide
  scope now.
- `tools/verify_release_artifacts.py` measures every distribution file
  against PyPI's cap, refuses oversizes before a cut, and rewrites the
  staging document's wheel sizes from that measurement.
- `gpuwm sim` no longer refuses the run folder it just created: both
  runners accept an empty directory, refusing only a populated one.
- `gpuwm domain --source hrrr` closes with the two shipped commands, not
  `tools/` paths no wheel contains and hand-pasted digests.
- `--statics-corridor` on the hrrr prep door was dropped silently, so a
  front-door moving nest sealed no child-resolution statics; forwarded
  now, and the route publishes its portable authorities on every run.
- `gpuwm sim` stopped calling a finished preparation partial: the
  completeness check reads the route's own receipt first, and a state with
  neither the restored surface nor the native soil pair refuses by name.
- `pip install 'gpuwm[gpu-cu12]'`/`[gpu-cu13]` install a matching CUDA
  toolkit beside the CuPy wheel; a box with a driver and no toolkit no
  longer dies at `Failed to find CUDA headers`.
- `gpuwm doctor` stops prescribing a wheel reinstall for missing CUDA
  headers: the remedy leads with `cupy-cuda13x[ctk]` at the driver's
  major.
- `pip install 'gpuwm[gpu-cu13,render]'` installed no gpuwm on Python
  3.14; `[render]` resolves `wrf-rust` with no environment marker, so 3.10
  through 3.14 install.
- `gpuwm doctor` derives its wheel-matrix sentence from the recorded
  interpreter ceiling, not a constant beside it.
- A moving nest cold-restarted every driver-held physics array at
  relocation, wiping precipitation memory (also 2.4.1). Continuation
  moves with the nest; only the exposed strip cold-starts. Default-on.
- A nested domain starting later than the experiment refuses at config
  load, naming the activation-epoch crash.
- A cross-domain wrfout writer abort names its own root cause, and a
  missing-Rust-artifact remedy names its own crate.
- `gpuwm.hrrr_hierarchy_direct` grew `--ack`, so the shortwave-only profiles
  no longer refuse their own wizard-emitted namelists.
- The preparation door owns its refusals: out-of-coverage grids and
  short staged series are their own classes, sentence plus remedy on
  stderr, exit `78`, not tracebacks.
- `--source gem` selects the Canadian global route by the model's common
  name; it is an alias row now, and the CLI reference carries the
  alias table.
- `rw_wrfbatch` draws the whole domain of a latitude/longitude grid, the
  box holding every drawn row, not a thin inscribed band; projected
  grids untouched.
- `rw_ensbatch` reads a member's whole forecast: the roster kept only the
  newest outfile, so `--frames 0` drew the final valid time under the
  first's name.
- `gpuwm-wrf-init --source gfs` answers an undecodable experiment config
  by naming the decode fault with line and column.
- `tools/stage_wheel_bridges.py --dest DIR` stages a bridge directory from
  the same `BUNDLED_ARTIFACTS` declaration the wheel reads, so
  `~/.gpuwm/bridges` needs no hand-filling.
- The AI-ensemble adapter row's default product no longer names the
  ensemble mean, the silent wrong answer the member capability refuses.
- Watching a forecast's progress file cannot kill it: the `progress.json`
  republish retries its replace, so a Windows reader racing the rename
  costs at most a stale heartbeat.
- A 2.4.x archive downscales again: the `sase_additive_dissipation`
  default flip made the non-SASE guard refuse the value 2.4.x restart
  headers record.
- A config-validation refusal inside `gpuwm downscale --point` carries the
  validation error's own sentence, not a VRAM answer, and
  a surface-physics child with no `--child-surface-from` refuses naming an
  in-product remedy.
- `gpuwm render --pair` composes the parent-vs-child compare documents:
  unmatched keys pair across the domain token where each side's stripped
  key is unambiguous; two nests never cross-pair.
- The wizard's nest-ladder closing block prints the runnable chain with
  real paths, and the four `gpuwm downscale` options with no help text
  declare themselves.
- `gpuwm prep`'s GRIB2 routes no longer demand `--grib2-inventory` or
  `--grib2-dump`: every route resolves both through the bridge ladder; the
  flags stay overrides and a stale binary refuses by name.
- A field-per-file source prepares on Windows: hundreds of `--input` flags
  overflowed CreateProcess's 32 KB limit, so the pairs move into a list
  file, `--input-list FILE`.
- The render skip notice stopped denying its own exit code: with nothing
  drawn it says so and points at `--list-products`.
- Grell-Freitas (`cu_physics = 3`) carries a measured regime note.
- WPS's own sixteen_pt overshoot on a sharp soil-moisture gradient no
  longer refuses a whole preparation: within a 0.05 margin the value
  clamps to the bound with a counted advisory.
- Shin-Hong's SGS TKE carrier is floored at the scheme's cold-start value
  at the PBL seam, closing the one input class that reproduced the
  historical non-finite tke failure.
- The first-run kernel-compile notice fires when the CuPy cache holds no
  entry for this card, timed as its own `kernel_compile` phase.
- `proof.json` records `static_build` and `verify_inputs` inside `total`,
  and `fetch-manifest.json` splits download from verify, so hash time is
  never reported as bandwidth.
- `gpuwm go` is a composition of `gpuwm sim` again; `--progress-format`
  expresses the only axis where they differ.
- The soil state no longer carries the forcing model's 0.25-degree mesh:
  soil moisture crosses the resolution change as Noah's SRATIO wetness
  against the 30 arc-second texture, nothing leaving [SMCDRY, SMCMAX].
  Default-on.
- `gpuwm render` files every picture at
  `<--out>/<domain>/<product>/<valid-day>/` by default, both engines,
  filenames and bytes unchanged.
- The LES mixing remedy is default-on: a domain leaving `mix_isotropic`
  unset that violates the measured stability criterion runs isotropic
  mixing, announced at load; a written 0 still resumes an old-default
  checkpoint.
- The engine's compose scratch lands beside the prep output, not in
  system temp where tmpfs boxes died on a disk quota;
  `GPUWM_COMPOSE_SCRATCH` overrides it and refuses a missing directory by
  name.
- Bare-default preprocessing reaches the CPU backend on a box with no
  CuPy: `--preprocess-backend auto` is the default, the CPU fall is
  announced, and an explicit `cuda` with no CuPy refuses by name.
- `gpuwm domain` sizes from a declared budget (`--card`, `--vram-gib`) or
  a measured card, otherwise refusing with both flags named; the silent 24
  GiB assumption is gone, and `GPUWM_NO_LOCAL_GPU=1` has one definition,
  honored before touching any card.
- A failed render publishes nothing: the run folder and the
  `latest-run.txt` pointer appear only once the first picture lands.
- Refusal tails print the reader's own invocation plus `--explain`,
  re-runnable as printed, on every door.
- `--bridge` is optional on the era5 and gfs direct routes: omitted, it
  resolves through the staged-bridge ladder; an empty ladder refuses
  naming what it searched.
- An install ahead of PyPI gets its own `gpuwm version` sentence, not
  upgrade advice.
- The mapped prep door refuses in sentences at exit `78`: a missing path
  is named under its flag with the commands producing it; an explicit
  eta ladder is adopted at its own level count; a config with none gets
  one refusal naming both level counts and the two reconciling doors.
- The GFS direct adapter's eta-ladder refusal reads like the mapped
  door's: both level counts, both vocabularies, both reconciling doors.
- The GFS route leaves the four-file front door: `gpuwm prep --source gfs`
  authors and digest-binds its input manifest from the fetched directory,
  so the printed prep command runs as printed; the `--source-manifest`
  pair pins an existing manifest, and half the pair refuses.
- Pasting the printed prep command again re-authors nothing: identical
  bytes answer `MATCHED` with the same digest; differing bytes refuse
  naming both digests and the remedy.
- `gpuwm check` with no measurable card names the missing budget and the
  flag to type, never a bare exit code.
- The wizard reconciles its own cadences: a profile cadence landing on a
  fractional root step snaps to the nearest whole-step value, spoken; a
  hand-written pair keeps the loader's refusal.
- A refused `gpuwm go` names its run folder first, the same line the
  dry-run promises.
- `gpuwm --version` answers like `gpuwm version`, aliases print on one
  line, doctor paths keep one separator style, and the ahead-of-PyPI
  sentence names each version once.
- A refusal that is already a sentence keeps its own words; a bare path
  still gets its exception class named.
- A bare `gpuwm setup` names the 16 GB WPS_GEOG download it skips,
  so the estate gap reads as expected.
- `gpuwm_mapped_engine` exits 2 on a refusal, the sibling bridges' usage
  exit. Exit 3 made the release cut's execute-probe (doctor, the artifact
  verifier, the publish bridges job) refuse a healthy binary; the seam
  contract, always nonzero-plus-refusal-JSON, is unchanged.

## 2.4.1 (2026-08-14)

2.4.0 was tagged but never published: the publish workflow's own test job
refused it on two rows of the CLI suite that assert `gpuwm verify` and
`gpuwm run` exit 0, on a runner with no GPU extra installed, where both
now correctly refuse at exit 2. Nothing reached PyPI and no asset was
written. The tag stays where it is, because tags here are forward-only.
Everything 2.4.0 carried ships here, plus the fix below.

Fixed:
- Those two tests declare the capability estate they are about, instead
  of measuring the runner's. The refusals they tripped over are the
  headline of this release and are correct; the tests were asserting the
  pre-refusal exit codes. The workflow's own test list now sits on the
  release battery's stage-1 list, and the checklist gained the leg that
  runs it with no GPU extra installed, which is the only environment in
  which that class is visible.

New:
- European radar reaches the product. `gpuwm obs radar` decodes an ODIM
  polar volume into the `gpuwm-obs.radar-sweeps.v3` pack the existing
  sweeps reader accepts, reports the per-sweep Nyquist and its
  provenance, and superobs the pack onto a model domain for the LETKF
  adapter. The route was complete from the decoder down to the adapter
  in 2.3.3 and had no door, which by this project's rule means it did
  not exist. `rw_odim` is a bundled artifact now, so the whole route is
  reachable from `pip install gpuwm` plus `gpuwm fetch-bridges`, with no
  Rust toolchain and no source checkout.
- The frozen European radar site table carries an antenna altitude for
  every one of its 136 sites. The MeteoGate locations endpoint publishes
  a 2-D point and the OSCAR/Surface registry it links answers
  `totalCount: 0`, so every row used to read `elevation_m: null` -- and
  beam height above ground is a function of antenna height above mean sea
  level, so a site assimilated without one places every gate at the wrong
  altitude with nothing that looks like a failure. The heights are read
  out of `/where/height` in the volumes themselves. The table refuses a
  null rather than substituting a zero.
- Germany's split-file volumes assemble. Some national feeds publish one
  file per elevation and quantity rather than one volume file;
  `gpuwm obs radar pack --dir` assembles them into one volume. It refuses
  a directory holding two nominal times without `--stamp` rather than
  taking the newest, and an assembled pack quotes a manifest digest over
  its members instead of a file digest it cannot have.
- `gpuwm obs` is the door onto every observation front door. Each
  instrument -- MRMS, Stage-IV, ASOS/METAR, GOES ABI, the European
  composite and European polar volumes -- resolves its binary through the
  shared ladder and takes its arguments unchanged; bare `gpuwm obs`
  prints where each one resolved and whether it speaks this release's
  record contract. All six are bundled artifacts, so the command their
  refusals name can supply them.
- `gpuwm certify` refuses an empty kernel manifest and NVRTC compile
  drift. A manifest with no kernels in it used to satisfy every
  comparison it was asked to make.

Fixed:
- `gpuwm doctor` detects what is missing and means its exit code. It
  reports every extra the packaging declares, exercises every claim it
  makes rather than inferring from presence, and reports `untested`
  where it cannot exercise something instead of `ok`.
- Refusals fire at the front door, before expensive work, and name
  remedies that exist. `gpuwm go`, `gpuwm run`, `gpuwm render`,
  `gpuwm run-plan --probe` and both prepared runners take their
  requirements from one capability registry, so no two doors can
  disagree about what is missing or what installs it.
- No remedy names an extra that installs nothing. scipy is a base
  dependency, so the messages that sent readers to `gpuwm[obs]` and
  `gpuwm[dealias]` -- both deliberately empty and deliberately retained,
  so install lines already written down keep resolving -- now name the
  package. `gpuwm render --pair` no longer says Pillow comes with
  `[render]`; it comes with matplotlib, which is base.
- The mapped and 20CRv3 GRIB2 routes resolve `grib2_inventory` and
  `grib2_dump` through the same ladder every other artifact uses. They
  shelled `cargo build` into a directory no wheel contains, so
  `gpuwm-mapped-inspect` and `gpuwm adapt` died on a bare
  `NotADirectoryError` while both tools sat staged and pin-valid and
  doctor reported them `ok`. Doctor reports the route, not just the
  files.
- `gpuwm fetch-bridges --help` names what it actually stages. The
  inventory is derived from the artifact table, so it cannot drift on the
  next addition.
- Every documented flag and command is held against the parsers by test,
  in both directions, so a page cannot name a flag that does not exist
  and a flag cannot exist in no page.
- scipy, pyshp, huggingface_hub and h5py are declared where they are
  used, and the parity between what the wheel imports and what it
  declares is a gate rather than a review step.

Known issues:
- European radar observations are assimilated and land in the right
  place with the right footprint, but the analysis magnitude is not yet
  validated. Proven end to end on three real KNMI Den Helder volumes:
  97.8 percent of the increment falls within 200 km of the radar and 100
  percent within 215 km with none beyond, localisation matches the
  configured 12 km, all 7,475 radial-velocity observations enter the
  solve, and a negative control returns bitwise zero. Not proven: the
  size of the increment. The ensemble spread is constructed rather than
  cycled -- `mean(d^2)/(spread^2 + sigma_o^2)` is 9.6 -- so the
  increment's magnitude is not a defensible analysis. A cycled ensemble
  is the largest remaining gap.
- Reflectivity and clear-air assimilation are unexercised on the
  European route: there is no host reflectivity operator at
  `mp_physics=10`.
- A locally built wheel ships `bridge-pins.json` declaring no platforms,
  because pins are generated at release time, so `gpuwm setup` and
  `gpuwm fetch-bridges` refuse on any build that did not come through
  the release workflow. Use the published artifact, or point
  `GPUWM_RW_WRFBATCH` at a real staged binary.

## 2.3.3 (2026-08-14)

Fixed:
- `pip install gpuwm` now carries everything the high-resolution terrain
  path needs. 2.3.2 shipped worldwide 30 m terrain that no documented
  install could reach: rasterio and pyproj lived in an optional `geog`
  extra, `[all]` deliberately excluded it, and every published quickstart
  one-liner omitted it. Following `docs/public/HIGHRES-TERRAIN.md` on a
  documented install downloaded 160.7 MiB of Copernicus tiles and then
  died on a bare `ModuleNotFoundError`. Both libraries are ordinary
  runtime dependencies now, so every install line the project publishes
  reaches the feature.
- The terrain path refuses a missing geography stack BEFORE it fetches
  anything, as a named `HighresRefusal` naming the exact command to run.
  The old check sat after the download, was not a refusal type, and
  escaped `gpuwm static` as a raw traceback at exit 1. It is one sentence
  at exit 2 now, with nothing fetched, and `on_refuse = "fallback-30s"`
  deliberately does not swallow it: a coverage policy must not silently
  answer an install question with 900 m terrain.
- `gpuwm doctor` reports `geography stack (rasterio + pyproj)`, so the
  product can answer this before a run rather than after a download. It
  imports both in subprocesses, which also catches an installed-but-broken
  GDAL or PROJ that a presence check calls green.
- `gpuwm static` no longer demands a forcing GRIB and a Vtable it never
  opens. It builds geography from `geog_root` and the WPS namelist, so
  requiring a whole forecast cycle on disk first was a gate on bytes the
  command does not read, sitting across the terrain path. Both keys are
  still declared, and `gpuwm run` still refuses absent forcing at the
  front door.
- `docs/public/HIGHRES-TERRAIN.md` carries an install line and a worked
  example that produces real terrain: one 40 x 40 km Alpine domain at
  1 km, about 80 MB of tiles, verified end to end from a PyPI install.
  Its reproduce commands now use `python -m tools.<module>`, which works
  off a wheel; `python tools/<file>.py` needed a source checkout nobody
  installing from PyPI has.
- `docs/install.md` documents every extra `gpuwm` declares, including
  `[obs]` and `[dealias]`, which code told users to install and no
  document mentioned.

Known issues:
- The `geog` extra is retained and empty so `pip install 'gpuwm[geog]'`
  keeps working. It adds nothing.
- Outside the United States a high-resolution run still replaces terrain
  only. Land use and soil remain the 30-arc-second baseline, unchanged
  from 2.3.2.

## 2.3.2 (2026-08-14)

New:
- High-resolution terrain now works anywhere in the world. Copernicus DEM
  GLO-30 is the default outside the United States: 30 m ground sampling
  from 90 S to 84 N, fetched anonymously, with no account, no registration
  and no API key. Inside the conterminous United States nothing changes --
  `terrain_source = "auto"` still selects USGS 3DEP, and every existing
  domain builds exactly the bytes it built before.
- `terrain_source` in `[static.highres]` selects the elevation source by
  name: `auto`, `copernicus-dem-glo30`, `srtm-gl1` or `usgs-3dep-13as`.
  SRTM is served through OpenTopography's anonymous mirror and declares
  its own limits, so asking for it above 60 N or below 56 S is refused by
  name rather than answered with a hole.
- Coverage is checked against the source you actually asked for instead of
  one fixed footprint. A domain that runs off the edge is refused with the
  source named, the source's own footprint printed, and the overshoot given
  per edge, so the message says which way to move and by how much.
- Every run records which elevation source it used and which geoid that
  source's heights are measured against: Copernicus DEM on EGM2008, SRTM on
  EGM96, USGS 3DEP on NAVD88. The three differ regionally, so the receipt
  keeps them apart instead of treating the numbers as one quantity.
- Outside the United States, terrain is upgraded and land cover is not.
  Land use and soil stay at the older global data, deliberately: no freely
  licensed worldwide land-cover set separates an inland lake from the open
  sea, and getting that wrong at a coastline does more damage to a forecast
  than coarse land use does. The console line, the run's receipt and
  `docs/public/HIGHRES-TERRAIN.md` all say so at the point of use, so a
  30 m terrain map is never mistaken for a 30 m land-use map.
- `docs/public/HIGHRES-TERRAIN.md` documents the sources, their limits,
  their datums and what changing between them does to a real domain,
  measured on one 50 x 50 km domain at 500 m over the Colorado Front Range
  built three ways. Copernicus reads about 3.5 m higher than 3DEP under
  forest and 7 cm lower above treeline, because Copernicus sees the canopy
  and 3DEP sees the ground; once that one offset is removed, 9,999 of
  10,000 cells agree within 20 m, median slopes agree to 0.09 %, and the
  two datasets place ridges and valleys the same way in 99.06 % of cells.
  Both disagree with the 900 m global baseline they replace in one cell out
  of seven.

Fixed:
- Building one domain through two different elevation sources keeps both
  receipts. The receipt path was keyed on the grid and the domain only, so
  the second build silently overwrote the first build's provenance -- and
  building the same domain twice to compare sources is exactly what the
  documentation tells you to do. The source is part of the key now.
- The documentation described USGS 3DEP as a surface model. It is bare
  earth. That is corrected, and it is the difference the canopy measurement
  above rests on.

Known issues:
- The worldwide default is a surface model, so a forested domain built from
  Copernicus DEM sits roughly 3.5 m higher than bare ground, tracking the
  canopy and vanishing above treeline. There is no freely licensed global
  bare-earth elevation set at this resolution; inside the United States,
  `usgs-3dep-13as` is bare earth and remains the default there.

## 2.3.1 (2026-08-14)

2.3.0 was tagged but never published: the RW-WPS staging gate refused the
standalone preprocessing wheel, so nothing reached PyPI. The tag stays where
it is, because tags here are forward-only. Everything 2.3.0 carried ships
here, plus the first fix below.

New:
- Forcing a nest off a streamed parent now moves O(child footprint) per
  parent step, not O(parent). The coupler pulls exactly the two windows
  FORCE reads -- the field and the coupling mass over the child's
  footprint plus an 8-parent-cell halo -- through the store seam, and the
  executor's duplicate projection is gone, so a relocation moves the
  window by rewriting the child's placement and nothing else. Measured on
  the 5070 Ti through `execute_experiment` (256x256x49 streamed parent,
  72x72 child at 3:1, 12 parent steps): bit-identical to the all-resident
  tree in every carrier of both domains, one-way and two-way, the parent
  store bitwise unchanged by nest presence, and the corridor moved
  58.0 MiB where whole-parent pulls would have moved 2352 MiB, a 40.5x
  reduction that grows with the parent. A halo-starved window is the
  negative control, and every run carries the traffic in
  `coupler.force_sync_bytes` as a receipt.
- The second concurrent-nesting shape streams: a RESIDENT parent can
  drive a TILE-STREAMED child, one run, coupled every parent step, one-way
  or two-way. FORCE pulls only the child's boundary frame from its pinned
  host store (four strips, `spec_bdy_width + 8` cells, the child-side
  mirror of the streamed-parent footprint corridor;
  `coupler.force_sync_bytes` is the receipt) and writes the same
  full-perimeter rolling tables the resident path writes; the child's tile
  buffers consume them through per-buffer packed table windows
  (`gpuwm.core.nest_stream`) that re-copy at kernel-launch time whenever
  the FORCE generation moves, so a buffer can never apply a previous
  interval's forcing. Two-way feedback restricts out of the child's store
  into the resident parent. Gated bit-identical to the all-resident tree
  through `execute_experiment` in every carrier of both domains, with
  stale-store, stale-tables and starved-frame negative controls firing in
  both directions.
- Roads are assigned per domain, and the tree is priced together.
  `steppers_for_tree` prices each domain against the budget its
  predecessors left: resident claims, streamed tile working sets, and the
  coupling corridor's slots. Every decision's receipt names its road, its
  claim and the budget spent before it, so a tree that cannot fit refuses
  with the arithmetic instead of dying at the allocation. A pinned tiling
  still asks no planner and probes no card. What remains refused by name:
  a coupling edge with BOTH ends streamed (ungated), a nest that also
  carries a tabulated boundary series, and `[relocation]` targeting a
  streamed child.
- The offline child can be asked to stream. `[tiles]` joins the tables a
  RunConfig TOML may carry -- the `gpuwm downscale` child schema refused
  it as unknown, so the route whose domain most outgrows its card was
  the one route that could not ask -- and the child route wires the seam
  exactly as the prepared single-domain runner does: decide once, hand
  the decision to `make_stepper`, record it. The three observers are
  wired with it (refresh on the history cadence and before the final
  read; a store-aware stability observer), `report.json` gains a `tiles`
  verdict, and `gpuwm downscale --tiles {on,auto}` writes the block into
  the config `--point` derives -- mode only, never a tiling, because the
  plan belongs to the card the run meets and not to the machine that
  derived it. A child that configures nothing is byte-for-byte the run it
  was.
- The last two nesting shapes are reachable from the product front door.
  `[tiles]` is now a PER-DOMAIN surface as well as a tree-wide one: a
  `[[domain]]` row carries its own `tiles = { ... }`, and the override
  replaces rather than merges, so "stream the parent, keep the child
  resident" and its roles-flipped twin are both things a user can write in
  a config instead of shapes that only a test could reach. Both run
  through `gpuwm run <config> --outdir <dir>` with no workaround flags and
  with `feedback = 1`. Run-plan delivers the same surface, and
  `docs/public/TILES.md` documents it.
- The tree decision RESERVES its children before a streamed parent picks
  its tile. A streamed parent's tile search maximises the compute window
  against whatever budget it is shown, so pricing parent-first let it take
  the largest clean tile that fit the whole card (3.94 of 4.00 GiB, 98.5%)
  and its resident child then met "no tile fits in 0.06 GiB". Only the
  order of the arithmetic made that shape unreachable. The reservation
  constrains the TILE, never the VERDICT, so a tree that decided
  all-resident decides all-resident still.

Fixed:
- The standalone RW-WPS preprocessing wheel builds again. The
  nest-streaming work above gave `gpuwm/core/streaming.py`, which that
  wheel stages, four new function-local imports of two modules it
  deliberately does not carry, and the builder's unresolved-import scan
  refused the staging. All four sit behind a
  running forecast, so both modules are recorded as forecast-only in the
  builder's exception table rather than staged. Build tooling only: no
  product source changed and the wheel's behaviour is unchanged; it was
  simply unbuildable at 2.3.0.
- Two-way nest feedback runs under a streamed parent. The executor
  refused `feedback = 1` whenever the parent streamed, claiming nothing
  projects a restriction back into the store; the coupler had carried
  exactly that projection since the same merge, proven bit-identical at
  the coupler level, and the two truths were never in one gate. The
  refusal is gone and the dispatch is mode-blind. Measured through
  `execute_experiment` on a 5070 Ti (256x256x49 parent streamed from a
  pinned host store at tile 128, 72x72 child at 3:1, 12 parent steps,
  radiation/cumulus/PBL/LSM firing on both sides): one-way and two-way
  runs are bit-identical to the all-resident tree in every carrier of
  both domains, the feedback arms demonstrably differ from the one-way
  arms, a stale-coupler negative control diverges, and the parent's store
  is bitwise unchanged by the presence of a one-way nest. What a streamed
  parent genuinely cannot do still refuses by name in the coupler: a
  cross-scheme microphysics edge, and a streamed child.
- `[tiles]` streaming starts on native Windows. Two stacked defects made
  every planner-driven streamed run refuse before the config was read:
  `autoplan.Machine.detect` sourced host RAM only from `/proc/meminfo`
  and the cgroup files, so a Windows box was wrongly told it was
  "containerised with no cgroup memory limit"; and `streaming.decide`
  probed the machine before applying the configured `vram_budget_bytes` /
  `host_budget_bytes` overrides, so the keys whose documented purpose is
  to override the probe were never read where it raised. Both halves are
  fixed default-on: host RAM comes from
  `GlobalMemoryStatusEx` (`ullTotalPhys`) on win32, and the configured
  budgets are applied before the machine is consulted, on every platform.
  Measured on an RTX 5090 at 438x350x49 with `mode = "on"` and nothing
  else, planner-chosen 219x175 x3 buffers: dry and full-physics rungs
  both bit-exact against the resident run, twice per rung on this no-ECC
  card with matching state digests.
- The ERA5 direct door's soil hand-off resolves the orography the GRIB
  carries. `rw-wps --source era5` accepted the wizard's config since the
  `[case_data]` fix, decoded the GRIB and built the statics, then died
  in `preprocess_noah_soil` with "terrain and source_orography must be
  provided together": `gpuwm fetch --source era5` writes the invariant
  geopotential INTO `era5-combined.grib`, so the route declares no
  separate orography artifact and the soil call forwarded a None beside
  a real HGT_M. The horizontal stage had already remapped SOILGEO onto
  the mass grid as SOURCE_OROGRAPHY; `soil_source_orography()` now
  resolves exactly that field once, beside the guard that depends on it.
  The declared artifact still outranks the embedded record.
- The root runtime and the nested child stop dropping the same
  orography. Both gated the pair defensively, so where the direct door
  refused loudly they silently skipped WRF's `adjust_soil_temp_new`
  altogether while their own met fields carried SOURCE_OROGRAPHY. Nests
  feel this hardest: the finest grids carry the sharpest terrain
  disagreement with a ~31 km source. Both now resolve through
  `soil_source_orography`; a source that declares no orography anywhere
  keeps the historical no-adjustment path. The direct door and the
  wizard chain are now bit-identical in TSK and TSLB on the case that
  exposed the gap, and the chain route's land TSLB moves by the
  expected -5.28..+3.70 K (rms 0.46 K).
- gpuwm's own wrfout warm-starts gpuwm's own child. The history writer
  published six fields of a surface source; ISLTYP, TMN and VEGFRA --
  required by `--child-surface-from`'s reader -- were never named, so
  `gpuwm downscale` refused gpuwm output as a child surface source.
  Fixed at the source: five rows join the history inventory behind the
  land-surface-scheme gate the soil rows already sit behind, IVGTYP and
  SEAICE ride along (an absent SEAICE was substituted with zeros, so
  ice-covered columns warm-started ice-free with nothing said), every
  string is transcribed from the pinned WRF v4.6.1 Registry, and
  ISOILWATER joins the global attributes stock WRF writes. A child's own
  history is itself a valid `--child-surface-from` file now.
- A streamed `[tiles]` domain consumes its lateral boundaries on the
  domain's BOUND Davies clock, bit-identical to the resident run.  Every
  production real-data root binds a DomainClock to its external LBC
  mirror (WRF's post-increment dtbc, `dt..T_bdy`); the streamed tile
  hook silently DROPPED that binding when it converted each buffer, so
  the buffers fell back to the retired `0..T-dt` path and every
  clock-bound streamed run forced its boundaries ONE TIMESTEP LATE, a
  constant phase error with no NaN and no refusal.  Measured on the
  offline child at
  t+15 min: 41 of 76 wrfout fields differed (W by 0.0043 m/s on a
  0.174 m/s field); at 438x350x49 with the clock bound, 9/9 carriers,
  max|d| 0.43.  The hook now rebinds the domain's clock onto every buffer
  it converts, and both external attach functions preserve an existing
  binding across re-attachment.  The unbound compatibility semantics
  (direct era5/gfs routes without a DomainClock) are proven unmoved, and
  nested children were never affected.
- A streamed domain's `refresh_state` lands the scattered `diag/*`
  members back on the physics driver, so routes that publish frames off
  the DomainState (`gpuwm.io.wrfout.state_frame`; the offline child)
  emit the OLR the tiles computed instead of the snapshot's zeros --
  measured: streamed OLR max 0.0 against 292.77 W/m2 resident, with the
  production StoreFrame route publishing the same store correctly the
  whole time.  A store diagnostic with no destination on the state is
  now refused rather than silently skipped.
- A tree whose `[tiles]` block cannot run is refused when it is read, not
  when it has been paid for, and the refusal now names the shape that is
  actually unsupported: a coupling edge with BOTH ends streamed, which no
  gate has driven. `mode = "on"` streams every grid unconditionally, so on
  a tree it forces exactly that shape on every edge; it is now refused by
  `build_experiment`, the one load every front door shares, naming the
  mechanism and the three ways out, rather than by the tile builder after
  a fetch and two preparations had already been paid for. `mode = "auto"`
  stays accepted over a tree: streamed children and streamed parents are
  both legal roads now, and if the pricing ever streams both ends of one
  edge the walk refuses at DECISION time with both grids named. The
  earlier admission refused every nested `mode = "on"` config for being
  nested and short-circuited every nest under `auto` to resident without
  asking the planner; both are gone, so the child road is reachable from
  the front doors that carry it and a nest's receipt line carries the
  planner's own arithmetic.
- A 1024x1024 streamed forecast now runs from a bare default command.
  `--stream-init auto` priced the resident road from the prepared cache's
  `state/*` manifest, the serialized prognostics and not a `DomainState`,
  so at 1024x1024x49 it read 4 716 B per column where the state alone
  measures 11 276.5: auto called a 6.45 GiB fit and the resident road died
  with 16 276 726 272 B allocated, with nothing re-routing after the
  refusal. Auto now prices this domain's own columns at the cost the whole
  route was measured at, 15 780 B per column at nz = 49, and takes the
  larger of that and the manifest term. The measured term predicts that
  refusal to 0.4%. 384x384 still prices at 2.17 GiB and still takes the
  resident road; the crossover on a 16 GB card sits near 768x768, and the
  receipt names which term set the price.
- The store-direct road runs the full-state health gate. It used to arm
  only the per-step stability fold -- u, w and theta finiteness plus the
  two Courant terms -- and declare the descriptor gate's per-field bounds
  out of reach. The domain is in the pinned host store, so the gate runs
  there instead, over the same fields under the same `rule_for_field`
  rules: moisture ranges, coupled-mass positivity, the geopotential and
  specific-volume limits, the soil bounds, and `gpu_integer_policy`'s
  exclusions and its refusal of an undeclared integer dtype. Armed at
  `initialized.d01` and `final.d01`, where the resident road runs it, and
  a failure is terminal on both. A store that turned out to be slab-height
  rather than domain-shaped is refused rather than reported as a pass. The
  executor's periodic full-state gate stays unarmed on cost, with the
  measured cost of one whole-store pass in the receipt beside the reason.
- Installing gpuwm no longer DOWNGRADES a user's wrf-rust. The `[render]`
  extra pinned `wrf-rust==0.2.35`, so `pip install 'gpuwm[render]'` (and
  `[all]`, `[all-cu12]`, `[all-cu13]`, which route through it) replaced a
  working newer core with the certified one -- reported from the field
  against 0.2.38. The extra now declares the window `>=0.2.35,<0.3`: the
  floor is what the products are certified against, the ceiling is where
  upstream is free to change the diagnostic surface. The four runtime
  checks that compared `== "0.2.35"` and refused anything else test the
  same window now, sourced from one module (`gpuwm/science_core.py`)
  instead of four copies of a string. Proven at both ends: the resolver
  leaves an existing 0.2.38 alone, and the suites pass on 0.2.35 and
  0.2.38 alike. Score files now report the version that ACTUALLY read the
  run beside the window that was required, rather than naming the floor
  regardless. `__version__` remains recorded and never gated: wrf-rust
  0.2.35 shipped the attribute reading `0.2.34`.
- `[tiles]` streaming no longer runs in silence when only a per-domain
  table asks for it. `streaming_receipt` was keyed on the TREE-WIDE
  `[tiles]` table, so a tree whose tree-wide mode is `off` and whose child
  carries `tiles = { mode = "on" }` streamed that grid and produced an
  EMPTY receipt: no line named the grid that tiled. The receipt is keyed
  on the per-domain DECISIONS now -- the tree-wide table is a default, not
  the answer. Where the tree-wide mode did not ask for what happened, the
  summary names the domains that overrode it. A tree the tree-wide table
  governs keeps its existing receipt bytes exactly.
- Four files reached the release line with CRLF line endings after a lane
  edited them on a Windows worktree, three of them product source
  (`gpuwm/core/streaming.py`, `gpuwm/experiment.py`, `gpuwm/runtime.py`).
  `.gitattributes` says `* -text`, so the flip entered the object database
  and every hash the product takes of those files would have disagreed
  with a Linux clone's. Normalised back to LF.

- The red-on-revert harness graded the tree it was written in rather than
  the tree it was run from: it opened with an absolute path to its lane
  worktree, so a copy carried onto any other checkout kept reverting and
  testing the ORIGINAL source and reporting PROVEN about it. It now
  derives the tree from its own location and refuses if `gpuwm` resolves
  outside it, and its restore leg round-trips bytes instead of writing
  back a CRLF copy of every LF file it touched.

- `gpuwm doctor`'s CUDA advice is installable. A support case on Ubuntu
  read `MISSING cupy (GPU runtime) ... Failed to find CUDA headers` and
  was offered `pip install 'gpuwm[gpu-cu13]'`: the wheel was already
  installed and no gpuwm extra has ever carried a CUDA toolkit, so pip
  reported success and the fault survived. That branch now routes on the
  import's own message -- the header/`CUDA_PATH`/`nvcc` signature gets the
  toolkit remedy (`conda install -c nvidia cuda-toolkit=<the major this
  box's driver serves>`, conda-forge and the NVIDIA pip wheels named
  beside it), a missing shared library keeps the wheel remedy, and a
  message carrying neither prints BOTH labelled by symptom rather than
  guessing. Separately, no remedy names a suffixed NVIDIA package any
  more: NVIDIA has deprecated `-cu12` as well as `-cu13` and both
  spellings install cleanly and supply nothing, so the affected lines now
  print `nvidia-cusolver`/`nvidia-cuda-runtime`/`nvidia-cuda-nvrtc` with
  the detected major in a version pin instead of in the package name.
  `docs/da-nowcast-quickstart.md` is corrected to what the code emits, and
  the tombstone rule now covers install lines in every tracked file under
  `docs/`, so a doc may warn about a tombstone but may not tell anyone to
  install one.

Battery:
- `tests/test_streaming_clock_arming.py` is registered on the GPU pytest
  shard. It shipped with its lane on no list, so until now the test
  existed and no leg ran it.
- The two front-door nesting legs are scripted GPU gates in
  `tools/battery/tiles_gates.txt`: each drives `gpuwm run` twice for the
  corruption screen, then digest-compares against an all-resident control
  of the same tree, and fails when the receipt says nothing streamed.
  Their case files carry no card budget and no machine-specific path.
- Five more files join the CPU stage-1 list, all found on no list at the
  cut: the CUDA-header tombstone and docs guards, the wrf-rust pin guard
  and its two consumers, and the gate that keeps the GPU shard list
  honest, itself an entry on nothing.
- `tests/test_native_wrf_distribution.py` joins
  `tools/battery/always_files.txt`, keeping its stage-1 entry, so every
  lane runs the RW-WPS staging boundary rather than only a cut. It stages
  the wheel by path and AST-walks the result, so it has no import edge to
  the modules whose imports it constrains: on this release's own range, 87
  product files touched, 289 tests selected, this suite not among them. It
  is the gate that burned 1.8.8 and 2.3.0, both times catchable in a lane
  in 18 seconds. `RELEASE_CHECKLIST.md` now makes stage-1 green at the
  stamped tip, before any tag, the first per-cut item.

## 2.2.1 (2026-08-13)

Fixed:
- ERA5 preparation no longer refuses the cache it just wrote. Whenever the
  soil-moisture floor fired, which is the common case on clipped ERA5, the
  prepared cache recorded a receipt the reader did not expect and the run
  stopped on its own output. ERA5 domains of 448x448 and larger could not
  be prepared at all.
- `gpuwm doctor` reports a missing CUDA header tree as a missing header
  tree. It compiles two kernels from a cold cache and says whether the
  CuPy wheel or the toolkit headers are the gap, then prints an external
  toolkit and `CUDA_PATH` as the remedy. The previous advice reinstalled
  the CuPy wheel, which carries the compiler and no headers, so it could
  not fix what it named.
- `gpuwm doctor` chooses CUDA library wheels by the CUDA major the box
  serves. A CUDA 13 box is no longer handed the `cu12` spellings, and
  never the `cu13` ones, which install cleanly and supply nothing.
- The Rust fetch cache stores one copy of each object instead of two. A
  whole-file fetch reaches the cache under two key shapes and stored two
  complete payloads at two paths; they now share one content-addressed
  copy, with the second key a hard link to it (a pointer file where the
  filesystem cannot hard-link). Measured on a 193.6 MB HRRR object: the
  cache fell from 3.00x to 2.00x the payload, and the fetch directory
  plus its cache from 4.00x to 3.00x. The fetch record and the HRRR
  manifest now carry a `dedup` block reporting bytes written, bytes
  deduplicated and reference entries.
- The ERA5 direct door accepts the config `gpuwm domain --source era5`
  writes. The adapter loaded the experiment through a path that split
  off `[fetch]` but refused `[case_data]`, so the wizard's own emission
  was refused by the wizard's own source adapter. The adapter now
  consumes `[case_data]` and uses its declarations to default the
  matching inputs: a declared geog_root or source orography needs no
  flag.
- Every front door that loads only the experiment portion of a config
  now validates and detaches `[case_data]` and `[static]` instead of
  refusing the file they sit in.
- A table that is present but not consumed by a loading path is reported
  as exactly that; the "does not have a table" refusal is reserved for
  tables that are genuinely unknown.
- An ERA5 forcing series that begins before the experiment start hour is
  trimmed loudly instead of refused. A series that does not contain the
  start hour refuses by naming the decoded window, the expected hour,
  and both fixes.
- The ERA5 refusals for a missing source terrain and a missing geography
  root name the exact missing input and every accepted remedy.
- The composed Thompson + Shin-Hong suite no longer dies with a
  non-finite TKE error on real forecast cases. Shin-Hong's SGS TKE
  chain is a diagnostic no tendency reads, and its transcribed WRF
  arithmetic divides by quantities that legitimately reach zero; a
  non-finite confined to that diagnostic is now repaired to the
  scheme's own cold-start floor with a one-line advisory, while a
  non-finite in any consumed output still stops the run exactly as
  before.
- Releases no longer fail on PyPI index propagation lag: the final
  promotion check now retries its PyPI read with the same bounded
  backoff the publish job already uses.

New:
- A wizard-emits/adapter-accepts round-trip gate: for each source (ERA5,
  GFS, HRRR), the config `gpuwm domain` writes must be accepted by that
  source's own adapter through config validation. It runs in the
  stage-1 battery on every cut.

## 2.2.0 (2026-08-13)

New:
- A fast-fix lane, for shipping a correctness fix in under 90 minutes.
  `tools/battery/fastfix.py` takes the range a fix spans and prints the
  test files the change can break, cheapest first, with the reason each
  was selected and an estimated wall time. Selection is by direct
  imports: measured against the alternative, a transitive walk selects
  three quarters of the test estate and finds nothing extra.
  `tools/battery/FASTFIX.md` documents the procedure, including what it
  does not yet cover.
- Repository-scanning gates run unconditionally, from
  `tools/battery/always_files.txt`. Citation checkers, receipt
  regenerators and line-ending gates read the tree instead of importing
  it, so no selector can reach them; running them every time is the only
  correct answer.
- A GPU test leg, `tools/battery/gpu_shard_files.txt`. The dynamics
  core had no automated gate on any list: sixth-order diffusion and its
  boundary faces, both Smagorinsky closures, the open lateral boundary,
  the acoustic solver, the mapped mass closure and the turbulence budget
  all skip on a machine without a card. First run on a 16 GB sm_120
  card: 314 passed, 121 seconds. The file that checks a gpuwm history
  frame with an independent WRF reader now runs too, 18 passed, where it
  had never run anywhere.
- The projected-source horizontal mapping runs in the packaged Rust
  bridge. `gpuwm_indexed_interp_f32` takes the donor as an exact integer
  pair plus its FP32 fraction, which is the shape a projected source
  needs and the reason this route had no Rust boundary at all: it selects
  its donor in FP64, and a coordinate just below an integer can advance
  that donor once it is rounded to FP32. The 4x4 stencil and both `oned`
  sweeps are fused per point across every core, with no intermediate
  array. Measured on a real HRRR CONUS window against the real les.km3
  d02 child (1059x1799 source, 608x488 target, 50 levels): the complete
  `interpolate_hrrr_to_lambert` apply set falls from 32.37 s to 0.301 s
  on 32 workers and to 2.334 s on one, and a single 3-D parabolic apply
  from 3.764 s to 0.027 s.
- The port is pinned to the NumPy operator it replaces, not to a
  tolerance. On that same real window and child, 136,187,136 output
  elements compare bit-identical across every method, both ranks, real
  values and an adversarial arm carrying zeros, negative zeros, the
  1e-20 sentinel itself, subnormal operands, NaN donors and overflowing
  products. `tests/test_hrrr_projected_operator.py` gates the same
  comparison on the raw uint32 view, so a signed zero cannot hide in it.
- The bridge now carries two `oned` missing-value predicates and says
  which authority each serves. The regular-grid entry keeps the CUDA one,
  which flushes subnormal operands; the projected entry uses the host
  IEEE one, which flushes only the product, because that is what the
  NumPy operator on this route does and reproducing it is the whole
  point. They differ only when an operand is subnormal or infinite and
  the host product is not, and both the divergence and the test that
  holds the line are named in `tools/grib1_bridge/src/lib.rs`.
- An install whose bridge predates the new entry keeps the NumPy mirror
  and is told so in one line, once, naming the rebuild. `gpuwm doctor`
  reports the same gap on the library itself rather than calling it
  simply verified, because the ABI integer cannot express an addition.
  The HRRR mapping report records `projected_horizontal_operator`, so a
  receipt can no longer say "cpu" for two implementations an order of
  magnitude apart.
- Legacy RRTMG shortwave runs its FP32 arithmetic 1.62x faster on the
  GPU and returns the same bits. The unit's 679 add/sub/mul sites carried
  a subnormal countermeasure that emulated every FP32 operation through
  FP64: decode a possibly subnormal operand from its bits, do the work in
  double, re-encode the result, about twenty instructions where one would
  do, at FP64's 1:64 rate on GeForce. They now compile to `add.rn.f32`,
  `sub.rn.f32` and `mul.rn.f32` written as inline PTX without the `.ftz`
  modifier, which is the same correctly-rounded binary32 operation with
  gradual underflow that the emulation was computing, and the same idiom
  `dycore.cu`, `acoustic.cu` and `mynn_pbl.cu` already use. Measured on
  an RTX 5070 Ti (sm_120) over 8,450 real fixture columns at 50 layers,
  shortwave kernel time per call falls 609.98 ms to 376.40 ms: the
  dominant `rsw_spcvmc_gpt_b` 511.07 to 371.70, `rsw_taumol_b` 43.38 to
  1.31, `rsw_cldprmc_b` 25.10 to 1.13, `rsw_spc_accum_b` 29.01 to 1.78.
  Every output field is byte-identical to the emulation at 169, 1,690,
  8,450 and 16,900 columns.
- The armor is retained at every one of the 679 sites, not traded away.
  A witness build counts, per source line, how many times each site runs,
  how often it sees or makes a subnormal, and whether the PTX and FP64
  arms ever disagree. Over a batch spanning a full diurnal `coszen`
  ladder from 0.02 to 0.99: 304,204,382 macro calls, 2,166,356 with a
  subnormal operand, 1,659,113 producing a subnormal result, and zero
  mismatches between the two arms. The subnormals are real and they are
  concentrated exactly where the file said they would be, in the
  direct-beam transmittance chain and what consumes it: `rsw_reftra`,
  `rsw_vrtqdr`, `rsw_spcvmc_body` and `rsw_spc_accum_body` hold every one
  of them and carry 79.7% of all calls, while `setcoef`, `taumol`,
  `cldprmc`, `sfluxzen`, `inatm` and `post` witness none.
- `tests/test_rrtmg_sw_rn_identity.py` proves on the CPU that the FP64
  emulation is ordinary IEEE binary32 round-to-nearest, over structured
  operand classes, an exhaustive sweep of the low-order subnormal range,
  a mantissa sweep that lands products on subnormal rounding ties, and a
  Sterbenz cancellation sweep. Two red-on-revert arms drop either half of
  the armor, the operand decode or the result encode, and show the same
  check going red.

Fixed:
- The early-render and time-to-first-plot suite ran none of its tests.
  A module-level skip for an optional package sat two thirds of the way
  down the file, and a module-level skip applies to the whole module, so
  the 21 tests defined above it were skipped as well. The suite is on
  the per-cut list and had reported success while executing nothing
  since the feature shipped. It now collects 36 tests and passes them.
- Two Noah-MP kernel suites had the same defect and four more tests
  that had never executed on any machine, including one whose own
  documentation said it needed no GPU and ran everywhere. Their
  device-free checks now live where they run.
- A new gate refuses the whole class: a module-level skip below a test
  definition fails, naming the file, the line, and how many tests it
  silently takes with it.
- The wheel-contents suite no longer skips when its build dependency is
  absent. Five of its nine assertions were skipping in the release
  environment, and those five are the ones that check that declared data
  files actually reach the wheel. An absent dependency is now a failure
  with a one-line remedy.
- The per-merge test cost falls by about 800 seconds with no assertion
  removed or weakened. The health-descriptor ceiling gate keeps running
  on every merge, over the neighbourhood of the measured peak, and
  re-derives that peak rather than reading it back; the exhaustive sweep
  moves to the per-cut tier. The GPU-marker gate reuses one collection
  instead of spawning several hundred. The registry citation checker
  indexes the tree once instead of walking it per citation.
- Seven test files whose subject or replacement was re-verified are
  retired. Three further candidates are kept and the reasons recorded:
  one rested on a duplicate that does not exist.
- Degrading to the Python fetch transport is no longer silent. The
  missing-backbone branch of the engine selection returned the stdlib
  transport with no message at all, which is the branch every install
  without the bridges bundle takes; it is also the expensive one,
  because that transport has no whole-file mode, so a `--mode
  full-file` request quietly became `.idx` subsetting. Selection now
  emits one `warning:` line naming the measured tax (560 s for one
  419 MB HRRR file against 27-35 s taken whole, roughly 16x) and the
  fix, before any bytes move. Nothing is refused.
- The line is said at selection time rather than by one command, so the
  GFS full-file command, the streamer's preflight and every library
  caller of the front door get it; the HRRR command's own near-duplicate
  is gone. `select_fetch_engine` returns the reason beside the engine;
  `resolve_fetch_engine` keeps its two-value shape for the callers that
  only want the pair.
- The fetch manifest records `engine_selection` beside `engine` --
  `rust`, `python-requested`, or `python-fallback` -- so a receipt
  distinguishes a transport somebody chose from one an install
  inherited. It is not called `transport`, because `--transport`
  already names the host on this front door.
- `gpuwm doctor` prices the missing backbone instead of only naming it:
  the check line carries the same measurement and the receipt field to
  look for.
- The shortwave unit no longer depends on a compiler flag to keep its
  subnormals. `tools/ftz_receipt` measures that each of the six
  compiler-emitted FP32 mechanisms it probes flushes under CuPy's appended
  `-ftz=true`, and that inline PTX without `.ftz` does not. The module
  currently escapes the
  append by compiling through `compile_using_nvrtc` rather than
  `RawModule`, and that escape is load-bearing but incidental: it was
  adopted because NVRTC 13 started rejecting the duplicate flag, and a
  build-route change would silently reintroduce the flush. Writing the
  instructions as PTX makes the subnormal contract a property of the
  source instead of a property of the option tuple.
- Streaming works from the product front doors. `[tiles]` had only ever
  run through the tilestream engineering route, and every path a user
  actually takes failed somewhere between admission and the first output
  frame. The shipped default physics suite failed first and hardest: its
  radiation is legacy RRTMG, a plain class whose constructor requires a
  start time and per-column latitude and longitude, and the per-buffer
  twin builder could rebuild only dataclasses and empty constructors. So
  every `[tiles]` forecast of the default suite raised StreamingRefused
  inside TiledRun before taking a step, and the docstring that should
  have caught it asserted legacy RRTMG was a dataclass. The builder now
  carries explicit constructor recipes, audited two ways: a recipe must
  name every constructor parameter, and its twin must agree with the
  domain's adapter on every non-volatile scalar. An adapter with no
  recipe is still refused. Measured on a 16 GB card through the prepared
  single-domain runner, 550x550x49 at 3 km with the default suite: 240
  streamed steps, one forecast hour, health green, no NaN, peak 15828
  MiB.
- The tile-buffer warm-up handed the radiation an atmosphere that did not
  exist. Buffers were built on `make_vertical_coord`'s default stretch
  while the domain's own eta table was imposed on top, so their 3-D base
  state described one atmosphere and their pressure another, and then a
  throwaway step integrated that against the domain's real lateral
  forcing. RTE+RRTMGP's gas tables refused the result at 120.3 K to
  407.3 K, and legacy RRTMG, which has no equivalent validator,
  integrated the same field in silence. The `[160, 355] K` range is the
  tables' own and is unchanged. Buffers are built on the domain's eta
  table now, and the warm-up step is gone: it existed only to allocate
  two lazily-created carriers, which are allocated directly instead.
  Verified with RTE+RRTMGP at 550x550x49, `mode = "on"`: 240 streamed
  steps, health green.
- `[tiles] mode = "on"` was refused at admission by the routes that
  support it. The refusal was written when the prepared routes passed no
  streamed-domain builder; they have passed one since, but the refusal
  outlived the wiring and went on rejecting the one mode that asks for
  streaming unconditionally, with a message asserting the route wired no
  builder while `mode = "auto"` streamed through that very builder.
  `gpuwm go` mirrored it before the download, so the front door most
  users type was the last place that rejected streaming. `gpuwm run`
  still refuses, because it reads `[tiles]` at no point.
- A streamed forecast's final health gate and canonical digest reported
  the initial condition. Under a host store the DomainState is the
  snapshot that filled the store, and neither the whole-field health
  validator nor the trajectory digest has a tile-interior form, so both
  answered for the analysis on a run that had integrated for an hour.
  The runner copies the domain back from the store before they read it,
  and the receipt records how many carriers moved.
- A streamed run could not publish reflectivity. `refl_10cm` is rebuilt
  scratch, correctly absent from the carrier manifest and therefore from
  the store, so each tile computed its own window and the transport had
  nowhere to join them. The sweep refused the first due frame, which on
  an hourly cadence is an hour into a healthy forecast. The slot is
  primed on the domain and on every buffer before the store is sized.
- Legacy RRTMG's ozone grid is carried by a checkpoint and no longer by a
  sweep. The adapter recomputes it from the climatology on every
  radiation call and reads it only from a child domain, which streaming
  refuses outright, so a warmed buffer had it while the prepared domain
  did not and the inventories differed by exactly that key.
- The prepare stage holds one forcing time resident instead of two. The
  loop retained the start time's state for its whole duration while every
  later time was built underneath it. Measured at 800x800x49 with
  Morrison, a 6-hour GFS chain: resident 14.67 GiB to 7.66 GiB, peak
  envelope 23.92 GiB to 15.86 GiB. A 24-hour chain and any ERA5 chain
  still price above 16 GiB; the binding term left is
  `domain_boundary_snapshot`'s full-domain host copy.
- The HRRR route can configure `[tiles]`. It had no way to, and said
  nothing: the block was silently dropped and the run went resident.
  Carried as a runner flag rather than rendered into the published
  authority, which is hash-bound, so that a bundle prepared streamed can
  still be re-run resident. Multi-domain plans that cannot stream are
  refused at config validation, before the fetch, naming every nest and
  the grid that can stream.
- The prepared front door renders its first committed frame. It published
  frames with nothing watching, so reaching a first plot needed an
  external watcher polling `GPUWM_WRITE_COMPLETE`.
- Legacy RRTMG compiles its shortwave CUDA engine once per process rather
  than once per adapter. Streaming builds one adapter per tile buffer, so
  an uncached engine multiplied both the NVRTC compile and the resident
  device tables that streaming exists to save.
- A streamed forecast publishes `OLR`. It had been dropped from every
  frame a streamed run wrote, silently: the field is produced by
  radiation, published to output and read back by nothing, so it is not
  a carrier, so the transport neither gathered nor scattered it and the
  store had nowhere to join the tiles' windows. The frame was written
  short and the run still reported forecast validity PASS with health
  status bits clear, so 73 of a resident run's 74 variables was
  indistinguishable from success. The transport now carries the
  output-only diagnostics the same way it carries reflectivity, which
  leaves the trajectory bit-identical because nothing reads them back,
  and `XKMH`/`XKHH` are carried with them when the eddy-viscosity
  diagnostic is on.
- A streamed frame that cannot publish a field a resident run writes is
  refused instead of written short. The refusal was documented but never
  implemented: the writer took the short field list as its schema and
  the file validated. It now names the missing rows and the one-line
  remedy.
- Carrier provenance on a streamed frame states the radiation that
  actually ran. `GLW` and `SWDOWN` were byte-identical to a resident run
  while the same file's provenance attributes read `unwritten` with no
  update time, because the export path read the driver on the domain
  state and a streamed domain's radiation runs on the tile buffers. The
  ledger itself was always current; only the reader was stale.
- `gpuwm go` and `gpuwm check` price a streamed run as streamed. Every
  memory term described a domain resident in VRAM, so a configuration
  with `[tiles]` was refused by the default front door on the strength of
  a number describing a run that was not going to happen -- and the
  configuration refused was the one streaming exists to make possible.
  The forecast term is now the streamed envelope, the tile working set
  and the pinned store, taken from the same measured model the run
  attaches with. Preprocessing is not streamed and keeps its own term.
- `gpuwm check` no longer tells users that `[tiles] mode = "on"` is
  refused by the forecast routes. It has not been since the routes were
  wired, and the same advisory also claimed the estimator had no model
  of a streamed domain, which is no longer true either.
- `UP_HELI_MAX` is bit-identical under tiling. It was the one field of 75
  a streamed run did not reproduce exactly, by one part in ten billion at
  a single cell -- and that cell sat on a tile seam, which is what named
  the cause. The diagnostic is evaluated at the end of a step from a
  stencil two cells wide, at the one moment when a tile's outermost halo
  has already been consumed by that step, and the halo carried exactly
  two cells of spare margin. A run that emits the diagnostic now widens
  its halo by the diagnostic's own reach, with nothing to configure. The
  forecast is unchanged either way; the tile windows grow by four cells
  on each axis.

## 2.1.1 (2026-08-13)

Fixed:
- The release verifier proves every bundled library through the ABI
  symbol that library itself declares. It used to ask every library
  for the CPU preprocessing library's symbol, which the region global
  dealiasing library does not export, and it had no branch at all for
  a vendored artifact, whose freshness is proved by its contract
  marker rather than by a source stamp. Either defect refuses a
  correct set of release artifacts, and both were reached only while
  preparing 2.1.0, so 2.1.0 was tagged and never published. 2.1.1 is
  the published form of that release and carries the same contents.
  A library artifact that declares no ABI handshake is now refused by
  name instead of being probed with another library's, one table
  declares each handshake for both the verifier and the release
  workflow, and a local probe loads every such library before a tag
  exists.

## 2.1.0 (2026-08-13)

New:
- Correlation-coefficient QC is available at the radar grid-build front
  door as `--cc-qc`, off by default. Its rule is per moment. In
  reflectivity a gate is dropped only where RhoHV and reflectivity are
  both low, so hail cores, the melting layer and tornadic debris survive
  as echo. In velocity there is no reflectivity shield: a low-RhoHV gate
  loses its velocity at every reflectivity, because a scatterer that does
  not move with the air is not a wind observation. Off is byte-identical
  to a build from before the flag existed.
- The debris-signature fringe keeps its velocity. A velocity gate
  survives the mask where RhoHV sits between the debris floor and the
  velocity threshold, reflectivity sits between 30 dBZ and the shield,
  and the gate lies inside the neighbourhood of a clustered velocity
  couplet in the same sweep. All five conditions must hold. Low RhoHV
  alone never qualifies, and the debris core at and above the shield is
  untouched. The exemption is on with `--cc-qc`;
  `--cc-no-tds-fringe-exempt` restores the strict rule.
- The mask's account is per radar, not per file. Every exempted gate is
  counted, and so is every candidate each criterion turned away, so a
  reader can reconstruct the strict-rule total from the receipt without
  rerunning anything. The `cc_qc` provenance key appears only when at
  least one radar's mask ran.
- Radar velocity dealiasing has a new default engine. `--dealias` runs
  the vendored `region-global-dealias` crate
  (`tools/region_global_dealias`), a Rust port of Py-ART's
  `dealias_region_based`, through a C ABI, with its refinement pass on.
  Verified fold-for-fold identical to Py-ART 2.2.5 across 1,355,617
  velocity gates of a real Level-II volume. There it keeps 3,894 more
  velocity cells than the previous `vad-region` engine, which rejects
  both gates of that volume's strongest couplet, and the dealias stage
  falls from 18.3 seconds to 62 ms. `--dealias-engine vad-region` still
  selects the old solver, `--no-dealias-refinement` turns the refinement
  off, and the engine that ran is recorded in `provenance.dealias`.
- Dealiased velocities are bounded by physics on both engines. A gate
  unfolded past `max_speed_ms` (75 m/s) is rejected as
  `speed_out_of_range` and counted, never clamped and never passed: 115
  gates of that volume, worst 114.5 m/s, couplet untouched.
  `gpuwm fetch-bridges` stages the library the default engine needs, and
  `gpuwm doctor` blocks without it.
- `tools/dealias_engine_compare.py` runs both engines, the masking-only
  and bound-lifted arms over one volume, reporting each arm's wall clock,
  gate account, gridded `vr` field, per-cell differences, and what each
  did to the strongest couplet in the data.
- Radial-velocity dealiasing runs about 21x faster on a real volume, and
  decides exactly what it decided before. Its coarse VAD seed search --
  1.13 million wrapped-cost cosines per range band, 1214 bands in a
  KTLX volume -- moved into the existing preprocessing cdylib as
  `gpuwm_dealias_coarse_cost_f64`, which rides a plane-rotation
  recurrence along the speed axis instead of calling a cosine per
  candidate, and evaluates a whole sweep's bands in parallel. Measured
  back to back on one KTLX 2013-05-20 volume: the band-fit stage 25.91 s
  to 1.21 s, the dealiasing cost of the volume 29.97 s to 2.63 s, the
  whole observation front door 32.05 s to 5.00 s, with the gridded
  output sha256 and every dealias count identical to the NumPy path and
  to the unported tree. The kernel never ranks candidates: it shortlists
  them, `_coarse_shortlisted` proves from its own error bound that no
  excluded candidate can be in the answer, and the surviving shortlist is
  ranked by the original NumPy expression -- widening to the exhaustive
  search when the proof cannot be made. Without the native library the
  pure-NumPy search runs as before, and the sweep's stats say which path
  it took.
- Wall-clock instrumentation reaches the observation, verification and
  I/O paths, which had none. `GPUWM_PERF_TIMING=1` turns on a per-stage
  clock inside the dealiasing pipeline, the paired-frame reader and
  comparison, the wrfout frame writer, the restart writer, the static
  builder and the HRRR bridge verification; `GPUWM_PERF_TIMING_OUT=<path>`
  writes the receipt at exit. Off by default, and off costs a
  module-global read per probe. Nested stages are subtracted from their
  parent so a receipt says where the wall clock actually sat.
- `tools/perf_obs_timing.py` prices the observation front door on one real
  radar volume, gridding it with dealiasing off and on so the capability's
  cost is a difference rather than an assertion.
- The paired-frame judge is a Rust program. `rw_fieldcmp` prints the
  metric table from two directories of history frames: per-field
  quantiles and their differences, accumulation sums and their ratio,
  composite coverage at each threshold. `--threads` sizes the pool,
  `--table` and `--json` write it out, and arm labels, fields, composite
  and thresholds all come from the command line. On a paired
  verification set: 2.6 times sooner, all 529 table lines identical to
  the Python judge's.
- The paired-run judge is born Rust. `rw_runscore`, in the same crate as
  the frame judge, scores two run directories against each other over a
  registered metric set: pooled low-pass state RMSE and pooled boundary
  increment error per domain and per field, the first-object timing
  difference per domain, and the mean neighbourhood skill distance on
  whichever domains carry it. The ladder, the spacings, the field list,
  every threshold and every metric-key spelling arrive on the command
  line; the defaults name WRF-convention variables and generic keys, and
  no campaign, case or physics suite is baked in.
- It walks the ladder once. The Python scorer reads every frame four
  times per field, once as the current frame of its own interval and once
  as the previous frame of the next, on each arm, reopening the file every
  time: 672 decodes of the state fields and 720 file opens for a
  four-domain seven-field pair. The Rust one decodes each frame's fields
  once from one open handle, reduces each on the spot to the scored
  interior sum and the outer ring of cells the next interval needs, drops
  the field, and runs the frames in parallel. Same pair: 440 decodes, 56
  opens.
- Measured on a real two-arm WRF pair from the pinned node-1 build, seven
  frames per domain per arm staged on the registered four-domain ladder,
  6.0 GB of history, cache warm: all 61 distances are bit-identical to the
  Python scorer, and the comparison costs 16.7 s against the reference's
  111 s. Three independent timing rounds put it between 6.5x and 7.2x.
- `tools/verify_runscore_parity.py` runs both scorers on one pair and
  reports every metric that differs, with the gap in representable
  doubles rather than a relative epsilon that hides a wrong answer at
  small magnitudes. It times both in the same pass, so the parity claim
  and the speed claim come from the same bytes, and it records how the
  pair was staged. The receipt for the run above is
  `evidence/verify-instrument/paired-run-score-parity.json`.
- What the Rust scorer does not replace: the campaign door also re-hashes
  every history frame against its run artifact before it scores anything.
  That leg is still the Python one's. It is cheaper than it looks on a
  warm cache, about 2 s of the 122 s door on this pair, so the metric
  arithmetic is where the time went and where it was taken from.
- A stored hour's plots render at the same time instead of one after
  another, in a private pool the met kernels' own parallel work nests
  inside. The width is the smallest of the box's physical cores, the
  hour's product count, and what free memory affords at half of what is
  free; a width that memory rather than cores decided says so and names
  the override. `RUSTWX_BATCH_RENDER_THREADS` sets it for this loop,
  `RUSTWX_RENDER_THREADS` for the renderer as a whole. Measured on a
  337-plot two-frame gallery from a local WRF run: 85 s to 9 s on a
  16-core box, every PNG byte-identical to the serial render.
- Both console streams report in catalog order rather than completion
  order, at every width. The progress events, the summary's output list
  and `rw_wrfbatch`'s stdout read as the serial pass did, and so now do
  the advisories a render writes to stderr about itself. Cancellation
  still stops new plots from starting and lets in-flight ones finish,
  now up to one per worker rather than one in total.
- A release gate changed: the battery now tests the Rust workspaces this
  project ships, `tools/rustwx` and `tools/grib1_bridge`, which no cut
  had ever compiled. `tools/battery/cargo_gates.txt` lists them per
  package with a shard and a test floor;
  `tools/battery/run_cargo_gates.py` runs the leg offline under an
  external `CARGO_TARGET_DIR`, reporting COULD NOT RUN rather than a
  pass when cargo is absent.
  `tests/test_cargo_gate_manifest.py` keeps list and members in step, so
  a new crate cannot land ungated.
- The LETKF analysis times itself. `gpuwm.da.letkf.analyze` records its
  setup, its chunk loop and its finish, and splits the chunk loop at the
  phase boundary into the localisation weighting (evaluated at every
  gridpoint) and the batched transform (at the active ones only). The
  five numbers reach every cycle report through the `filter` block of
  `assimilate_radar_grid`'s provenance, beside the host-to-device staging
  and unstaging the device arm pays. There was no wall clock anywhere in
  `gpuwm/da/` before this, so a report could say the analysis was most of
  a DA leg and not say what part of the analysis that was.
- `tools/da_solve_ab.py` A/Bs one LETKF analysis between solve devices on
  byte-identical inputs. Each arm runs in a fresh subprocess, the bundle's
  every input file is re-digested before either arm reads it, and the
  receipt carries per-stage wall clock, per-field increment agreement and
  the observation counts that say the two arms saw the same analysis. It
  judges as well as measures: bitwise-identical increments across two
  devices are reported as a device arm that silently ran the host path,
  not as agreement, and a bundle built by the harness rather than dumped
  from a real leg is labelled synthetic in the verdict.
- `tools/da_cycle_prepared.py --dump-analysis-bundle DIR` copies each
  leg's analysis inputs -- the staged member checkpoints, that leg's
  observation file, the history file its observations were gridded onto --
  into a bundle the A/B replays. A leg whose analysis needs a forward
  operator that files alone cannot rebuild says so and dumps nothing.
- MEASURED, on a synthetic bundle at a radar-sparse shape (86,400
  gridpoints, 6.5% of them active, 10 members, two radars, numpy):
  the localisation weighting is 19.9 s of a 22.6 s chunk loop and the
  batched transform is 2.2 s. The phase that scales with the WHOLE
  domain dominates the phase that scales with the observed part, so a
  faster batched eigensolver -- in any language -- is bidding for 9% of
  this analysis. `evidence/da-solve-ab/host-baseline-sparse.json`.
- MEASURED, same bundle, host against cuda on one RTX 5090, medians of
  three: the analysis is 16.63 s on the host arm and 2.17 s on the device
  arm. The device wins the localisation weighting 14.41 to 0.33 and the
  transform 1.65 to 0.84, and pays 0.27 s to stage. Sharing the card with
  a GPU load taxes the device analysis 2x, to 4.29 s, and leaves the host
  analysis unmoved. The arms agree to 2.6e-15 relative, worst field, with
  different eigensolvers and identical observation counts.
  `evidence/da-solve-ab/README.md`.

Fixed:
- Verifying an HRRR bridge publication hashes its payloads concurrently.
  The check still covers every payload the manifest lists, so what it
  proves is unchanged, but the reads no longer run one file at a time.
  Measured here at 1.18 GB/s serial against 3.51 GB/s on eight workers.
  A single lead's snapshot load re-verified the whole sealed
  publication -- all 25 leads on a 24-hour case -- and that read is what
  made two of six cases take 267 and 280 seconds in a stage the other
  four finished in 10 to 12.
- The Rust judge's summation reproduced the reference's pairwise tree but
  not the buffering around it. The reference's reduction hands its
  summation kernel 8192 elements at a time and adds each buffer's total
  into a running scalar, so an array longer than one buffer is summed as
  a sequence of trees rather than as one tree, and past about a million
  elements the two arrangements differ in the last bit or two. Every
  paired-run metric works at that size. Both `rw_fieldcmp` and
  `rw_runscore` now buffer, in single precision as well as double, with
  the block totals computed concurrently and added in order so the answer
  does not depend on the worker count. The buffer length is a settable
  default in the reference stack, which is recorded where the constant
  lives: a caller that changes it moves the last bit of every large sum
  on both sides.
- Selecting the outer frame of cells from a stack of planes leaves the
  reference holding a column-major array, and its sums walk memory rather
  than logic. The ring is packed cell-first to match. Before this was
  traced, three boundary metrics sat one bit away from the reference with
  no explanation.
- The live bundle smoke counted only the binaries it staged. `fetch_bundle`
  stages every pinned binary and every pinned map asset, so the assertion
  read 9 against a true 47 on the published v2.0.0 bundle. It is derived
  from the manifest now. Nothing offline had ever packed a bundle carrying
  assets, which is why it survived; a hermetic test covers that half now.
  The same test skipped two different states through one branch: a tree
  that has never been cut still skips, a cut tree missing a bundle for a
  supported platform fails.
- Three `rustwx-products` tests drove one process-global disclosure with
  nothing serializing them, so one test's reset landed between another's
  set and its assertion about one run in five. The new cargo leg is what
  made it visible. The product was never involved.
- The whole-report doctor paste fixture did not know
  `tools/region_global_dealias`, so the new region-dealias check believed
  a source checkout was a wheel and printed a `git clone` the paste
  contract refuses.
- The release verifier could not reason about a vendored artifact. It
  required every binary in a bridge bundle to carry this release's
  source-revision stamp, but the region-global dealias library is a
  verbatim upstream crate frozen at a recorded commit, deliberately
  unstamped, and proved instead by the contract marker naming the ABI it
  speaks. The packer already knew that; the verifier did not, and would
  have refused a correct bundle. It asks a vendored artifact for its
  marker now, and refuses one carrying the wrong marker or none, so the
  staleness question is still asked of every binary. Its receipt names
  which binaries each proof covered instead of asserting one blanket
  claim, so the receipt schema is `gpuwm-release-artifact-proof-v2`.
  `tests/test_verify_release_artifacts.py` and the release-snapshot
  suites joined the battery's stage-1 list, which had never carried the
  machinery a cut itself runs.

## 2.0.0 (2026-08-12)

New:
- A streamed sweep's transfers run beside its compute. Each tile buffer
  carries a copy-in and a copy-out stream next to its compute stream,
  ordered by the event chain `tilestream/overlap.py` derives from the
  ring plan. When nothing armed on the run needs a hard sweep barrier,
  the end-of-sweep device synchronization is deferred too: the next
  step's gathers chain on the previous step's scatter events and the
  pipeline stays primed across steps. Anything that reads the store
  drains first, through `TiledRun.store` or `TiledRun.drain()`.
  `overlap="on"` is the default and changes no arithmetic; `"off"`
  keeps the single-stream loop as the reference; `"unchained"` drops
  the chain, wrong on purpose, as the negative control.
- `tilestream/test_overlap.py` gates the event chain without a card. The
  sweep is simulated under the driver's own dependencies and driven by
  adversarial schedules; the full chain must match the monolithic
  reference under every adversary, each wait class cut must fire the
  construction-time checker and either move the digest or be proven
  transitively implied, and the unchained control must move the digest.
  Listed in `tools/battery/tiles_gates.txt` as a cpu gate.
- A domain too large for the card runs as a streamed forecast. The state
  lives in pinned host RAM and cycles through the GPU one tile at a time,
  under the `[tiles]` table. `mode = "auto"` streams only when the domain
  does not fit, and unknown keys in the table are refused by name.
- `docs/public/TILES.md` documents the streamed run. `gpuwm stream` and
  `docs/public/STREAMING.md` remain the HRRR cycle-following feature. Both
  pages, and both `stream` help texts, open by naming the other: the two
  share no configuration and no code path.
- Health is reduced per tile. `health_partial_tile` folds a tile's nan,
  w_max and CFL record after its step and before its interior is
  scattered, so the run loop's gate reduces over the host store instead of
  over a state the sweep never writes.
- `tools/battery/tiles_gates.txt` enumerates the four `[tiles]` gates, the
  shard each runs on, the environment each one's real acceptance needs, and
  the measured card floor for the graph section. Two gates have a rung or
  geometry selector whose default is the weaker setting, so each is listed
  twice. `tests/test_tiles_gate_manifest.py` gates the list.
- Stage 1 gains seven CPU-only suites from this work, each with its reason
  inline: the `[tiles]` option surface, spawn-at-trigger under `[tiles]`,
  the gate list, the gates' green-on-nothing preconditions, the GLW
  declaration contract, the physics allocation inventory, and the
  distribution's import closure.
- `tilestream/` carries the harness, its engineering records and the
  evidence logs behind every figure stated here. None of it is product
  surface and no gate covers it.
- Specified (externally forced) lateral boundaries run through the
  multi-GPU decomposition. `plan_split` takes per-axis periodicity; a
  non-periodic axis clamps edge ranks at the domain, so the real boundary
  sits on the rank's own array edge, and no wrap seam is exchanged there.
- Each forced rank attaches the domain's `LateralBoundaries` windowed to
  its array: the domain's own tables on true edges, inert tables on
  interior seams. The halo is padded by `max(spec_zone, relax_zone)` so
  seam-side boundary fiction stays inside the throwaway ring.
- Forced decompositions that cannot be right are refused by name: forcing
  missing on a specified config, forcing on a periodic config, nested
  forcing, a quarantine-defeating halo, Davies zone mismatch between the
  forcing and the config, and ranks narrower than the relaxation frame.
- The gate battery gains a FORCED rung alongside the periodic one:
  `tilestream.test_forced_gate` runs the specified case at 1, 2 and 4
  ranks against the resident digest, with a poison-seam control that must
  match and a scaled-edge control that must differ. Stage 1 gains
  `tests/test_multigpu_specified_bc.py`, the CPU-hermetic geometry,
  windowing, corner and refusal suite.
- Every radiative field a land-surface scheme reads carries a source and a
  last-producer time, and the check runs immediately before the scheme
  consumes it. Sources are `radiation_scheme`, `declared_constant`,
  `external_array`, `analytic_geometry`, `wrf_compat_zero` and `unwritten`.
  The consumer matrix states what each scheme reads: Noah GLW+SWDOWN, RUC
  GLW+GSW, Noah-MP GLW+SWDOWN+COSZEN, no land surface nothing. A scheme not
  in the matrix refuses rather than defaulting to requiring nothing.
- `surface_radiation_policy` defaults to `"required"` for every run. The
  escape `"wrf_compat_zero"` consumes unsourced carriers at their
  allocation fill, is never selected automatically, labels every carrier it
  admits, appears in the run receipt, and is refused on a resume that
  changed it. It is an experimental forcing, not a valid configuration for
  a real case. A declared constant GLW is an experimental forcing on the
  same terms, not a statement that radiation is available.
- The check counts the consumer's cells before refusing, with the same
  land and sea-ice predicate the schemes dispatch on. A domain with zero
  land and zero ice cells feeds its land-surface scheme nothing, so an
  all-water idealised run with radiation off is admitted. One land or
  ice cell restores the full refusal, and a state whose footprint cannot
  be read runs the check in full.
- Per-carrier provenance in the run receipt and the output metadata.
  The prepared runner's `report.json` carries one row per carrier with
  its source, last producer time and a representative value, and every
  wrfout file carries `GPUWM_SURFACE_RADIATION_POLICY` plus
  `GPUWM_CARRIER_<NAME>_SOURCE` and `GPUWM_CARRIER_<NAME>_LAST_UPDATE`
  globals, stamped per frame from the live contract.

Fixed:
- Classic RRTM+Dudhia (1/1) radiation no longer stalls large nests on
  host dispatch. The RRTM longwave solver re-dispatched its whole CuPy
  op graph once per fixed 512-column chunk and re-uploaded its
  coefficient tables in every chunk -- about 3.2 million kernel
  launches per radiation step on a 500x400 nest, GPU idle 93 percent
  of that wall. The chunk is now sized from free device memory at the
  first eager solve of each column geometry and cached (512-column
  floor, whole-grid cap, explicit `column_chunk` still pins it); the
  tables move to the device once per process. Byte identical across
  chunk sizes, pinned by the existing chunking test; the RTX 3090
  re-measure reproduced the pre-fix trajectory digests bit for bit.
  The 2-domain NSSL steady step fell from 39.0 to 4.2 seconds and
  Morrison from 37.9 to 3.0; a window that took 807 seconds takes 165.
  The reported 16x NSSL-vs-Morrison gap was a benchmark config
  confound (d01-only versus d01+d02 walls); on matched configs NSSL
  costs 1.0-1.4x Morrison before and after this fix.
- The RRTM chunk auto-sizer no longer queries device memory inside CUDA
  graph capture, where `cudaMemGetInfo` is refused. A captured solve
  reuses the eagerly cached chunk; a cold capture takes the 512-column
  floor. Byte identical by chunk invariance, with a bit-exact capture
  test.
- The carrier-contract consumption check runs inside CUDA graph capture,
  where its blocking footprint and finiteness reads are refused. Under
  an active health ledger the footprint read is skipped and the check
  runs in full (fail-closed); the
  finiteness verdict is recorded into the ledger, re-accumulated on
  replay, and raised at the sweep drain.
- A streamed run survives its second output frame. The carrier
  contract's produced-at ledger now rides the domain clock's round trip
  through the tile sweep, the graph stepper's scalar records and the
  streamed checkpoint header. Fresh tile buffers previously kept their
  build-time stamps, so the freshness law refused hour-N consumption as
  stale GLW while radiation ran on every tile. The law is unchanged;
  pre-contract streamed checkpoints take the one-time producer
  refresh.
- The tiles harness and the fixtures behind the conformance, RUC
  runtime, MYJ and MYNN pairing suites declare the shortwave carriers
  the carrier contract made law, the way they declared GLW:
  `declared_swdown_kwargs`, `declared_carrier_kwargs` and
  `declare_offline_gsw` read the contract's own consumer matrix and
  declare the allocation zeros those rungs always ran on, so no gate
  digest moves.
- Both tiles real-case preparers assemble their water temperature before
  soil preprocessing, the mainline ERA5 route's own `assemble_for_route`
  call at full-domain scope so slab seams cannot split a connected body.
  Every streamed ERA5-with-SST run was refused at prepare without it.
  Policy, route and receipt reach the soil router,
  under a structural test.
- The committed GLW no-op digest test skips, naming the condition, when
  its probe subprocess cannot see a CUDA device the test process itself
  holds: visibility withheld from child processes, not a probe verdict.
  Every other nonzero exit still fails.
- The production streamed attach bound lateral boundaries EAGERLY on
  every buffer-tile change: every forcing interval re-validated, re-packed
  and re-uploaded, 27-63 ms per bind, host-blocking, ~0.3 s per step at
  the attribution run's largest arm. `make_tile_hook` now uses the
  single-slot streaming bind the real-case harness already carried the
  digest proof for, measured at ~0.01 ms per bind.
- The restart manifest's KF-expiry guard entered `array.device` as a CUDA
  context whenever the attribute existed. NumPy 2 gives every host array a
  `device`, the string `"cpu"`, so checkpointing host-resident state raised
  a TypeError. The guard now enters only a device that is a context
  manager; the CuPy owning-card entry is unchanged.
- The dycore left the periodic staggered alias slot stale between steps.
- RRTMGP memoised two device arrays without keying them to the card, so a
  second device read the first one's arrays.
- The map-factor division left `gpuwm/core/physics.py` without the alias
  it undoes.
- `plan_split` rejected the 1x1 identity plan.
- The restart inventory path ran a device-blind reduction.
- `kf` allocated `w0avg` at the first due cumulus call, so the carrier set
  changed identity mid-run. It is allocated at construction instead, which
  is why the slice gate now reassembles 246 arrays over 229 carriers where
  it reassembled 245 over 228.
- The streamed real-case route died at its first tile change:
  `tilestream/realcase.py` bound `tile_hook` to the superseded
  three-argument contract. The consolidated gate is periodic by design and
  never calls `tile_hook`, so it held 233 PASS / 0 FAIL throughout.
- The standalone RW-WPS preprocessing wheel staged `gpuwm/experiment.py`
  without `gpuwm.core.streaming`, which its field default requires.
- No `[tiles]` gate reports success over an empty comparison. Two empty
  carrier maps hashed equal and were recorded bit-exact, a reassembly
  verdict of `not bad` passed at zero carriers checked, and three verdict
  lines asserted totality while stating no size. Each now carries a
  declared precondition with a floor of one, and each passing verdict
  states how much it covered.
- The `health` entry in the mp=8 frozen-kernel census described a source
  this tree does not carry: two re-pins landed on opposite sides of a
  merge and only one was recorded. Recomputed over the merged source.
- The gate's Noah rungs declare their downward longwave instead of
  inheriting the constant that `initialize_physics` used to supply in
  silence. Every digest is unchanged and the receipt names an origin.
- A graph negative control that ran out of memory held the memory through
  the controls after it, by way of the caught exception's traceback.
- The wheel shipped `gpuwm/core/streaming.py` and not the `tilestream`
  package it calls, so `[tiles]` mode `auto` and `on` raised
  `ModuleNotFoundError` from a clean install while mode `off` worked.
  Every import is function-local and every gate runs from the repository
  root, which is why nothing caught it. `tilestream` is in the
  distribution now. `tools/les1m_probe.py` rode the same gap.
- `tilestream` resolved only when the working directory was the repository
  root, editable installs included. It resolves from any directory now.
- `tests/test_tiles_distribution.py` gates the property rather than the
  fix: the shipped package set must be closed under importing, and any
  file a shipped module opens beside itself must be declared package data.
  It is on stage 1.
- The physics allocation inventory still recorded `w0avg` against
  `kf.update_trigger_history` after the fix above moved it to
  `kf.ensure_trigger_history`. Stage 1 was green because that suite was
  not on the list. Row corrected, suite added.
- The gate list omitted the seam gate's `GRID=2x2` entry, so a battery run
  only ever split one axis. Both geometries discriminate.
- The card-floor note claimed the consolidated gate needs more than 16 GB,
  and its replacement claimed 16 GB is enough on any idle card. Both were
  wrong. An idle 16 GB RTX 4080 reports 233 checks passed and 0 failed. An
  idle 16 GB RTX 5080 refuses the graph section's capture allocations with
  15.2 GiB free, at 1.1 MB in one run and 201.6 MB in another. The note
  carries the measured matrix and says which cell an operator is reading.
- A graph capture the card would not serve was reported as a failed
  negative control, so the consolidated gate exited 1 on an idle 16 GB RTX
  5080. A row the card cannot run is MACHINE-LIMITED: counted on its own,
  printed where it happened, and listed in the verdict, which reads GATE
  PASSED WITH A COVERAGE HOLE and names every unevaluated row. A control
  that fails for any other reason, the harness included, still fails the
  gate.
- `[tiles]` decisions and refusals were worded as "streaming", which is
  the name of the unrelated `gpuwm stream` feature. They say `[tiles]`.
- The four gate runners printed "Box idle throughout" whenever at most one
  CUDA context was on the box, including when that one was somebody
  else's. The verdict separates this gate's own context from every other
  by pid, reports the split, and says UNKNOWN when nvidia-smi cannot be
  read.
- The consolidated gate then took its start count after its own first
  device call, so every run called the box not idle and its verdict
  provisional, an empty box included. The count is read before anything
  touches a card, as the other three runners already did.
- `tests/test_tiles_distribution.py` routed both load-bearing tests
  through `pytest.importorskip("setuptools")`, and the project virtualenv
  carries no setuptools, so the suite skipped in the environment it runs
  in. Each setuptools measurement now has a tomllib reading of the same
  declaration beside it, and a third test holds the two against each
  other. Reverting the include list to the state that shipped the defect
  turns the file red with setuptools absent and with setuptools present.
- `tests/test_tiles_gate_manifest.py` stayed green when `GRID=2x2` or
  `PHYSICS=1` was deleted from the gate list: entry identity is module
  plus environment, and every other test walks whatever the file says.
  Both rows are pinned by name.
- The check is at the consumer, so it covers direct `initialize_physics`
  callers, restarts and DA cycles, not only configs loaded through a front
  door. The config-load guards stay and are unchanged; they refuse earlier
  and more helpfully, and they are no longer the only line.
- `set_forcing` refuses a radiative carrier on a driver assembled
  without a contract, in the same sentence family as the radiation and
  consumption seams, instead of crashing on an attribute error after
  writing the buffer.
- Zero shortwave at night passes on source and age, never on the value
  looking plausible. The same zero after sunrise with no live producer
  refuses before the land-surface call, naming the carrier, the consumer
  and the fix.
- `gsw` joins the generic forcing setter. RUC reads net shortwave, so an
  offline-forced RUC run could not supply the one shortwave carrier its
  land surface consumes and integrated zeros.
- Radiation-free Noah-MP gets an analytic COSZEN provider on the radiation
  cadence. It was written once at startup, so a twelve-hour run computed
  canopy radiative transfer against the sun angle of its first minute.
- The 300 W m-2 GLW buffer fill can no longer be consumed. It stays as the
  allocation value so healthy trajectories are byte-identical, and its
  source is `unwritten` until a producer writes it. `swdown`'s 0.0 default
  becomes `None` on the same argument: the buffer is still zeros, and the
  contract can now tell a silent default from a declared zero.
- Carrier provenance and age are serialized into the restart driver header,
  two scalars per carrier, so no array key moves. A checkpoint written
  before the contract forces one producer refresh rather than guessing.
  Proven through `write_restart` and `restore_restart` themselves:
  a mid-interval checkpoint resumes with source and age bit-equal to the
  uninterrupted run, a header without the mapping forces the refresh,
  and a resume under a changed policy is refused.

Record correction:
- The 1.6.2 nocturnal dewpoint collapse ran under a silent fixed
  300 W m-2 GLW with shortwave-only radiation. It did not run under
  GLW = 0. Full LW+SW radiation removed the symptom and remains the
  default for real-data runs. The direct mechanism is proven for land and
  shoreline columns. Over open water the mechanism does not apply in a
  single surface call: SFCLAY builds its surface endpoint from TSK and
  PSFC alone (`gpuwm/core/kernels/sfclay.cu:255`) and TSK over water is
  prescribed, so a 120 W m-2 longwave change leaves QSFC, QFX, Q2 and
  TSK bit-identical (`tests/test_open_water_longwave_invariant.py`).
  Open-water attribution beyond one call is not claimed.
- The instrument for this class is
  `gpuwm/core/surface_moisture_ledger.py`, which records the final writer
  of Q2 by name, that writer's own inputs, and its formula re-evaluated in
  FP64 from the published inputs. `tools/surface_moisture_ledger.py`
  decomposes a difference between two arms into named terms and exits
  non-zero on an unexplained remainder above 0.1 g/kg. The decomposition
  is provider-aware: Noah SFCDIAGS rows expand the Noah identity
  (QSFC, QFX flux route, rho times cqs2 route, cross), SFCLAY-family
  rows expand theirs, because the first land A/B put the whole QFX route
  into the remainder under the SFCLAY terms.
- The accounting ran on a two-arm column A/B on node 5 differing only in
  the declared constant GLW, 300 versus 420 W m-2 over land for two
  hours: dQ2 +3.49 g/kg, dTd2 +4.8 K, dominant named term QSFC (skin
  saturation humidity, +5.07 g/kg, partially offset by the flux route),
  unexplained remainder 0.0 across all 1440 column-times, exit 0.
  Corrupting one published Q2 in the arm file flips the tool to exit 1.
  Receipt: `docs/public/receipts/surface-moisture-ledger-accounting.json`.
  This is column-tier evidence generated for the accounting; the
  historical case's data was never staged and no case-tier accounting is
  claimed.

Evidence:
- Ported from a tree whose branch point is byte-identical to this line at
  1.8.7 over all 13,585 shared paths, re-measured at the merge tip before
  anything moved. 229 clean adds, 40 modified files, 12 collisions, 10 of
  them merging with zero conflicts.
- No capacity multiplier against a resident run is stated here, because
  none has been measured. `tilestream/NO-DRY-NUMBERS.md` lists the figures
  that may not be quoted and why.

## 1.9.1 (2026-08-11)

Three defects the twin verification of the 1.9.0 ports caught, each
closed with a class instrument so the pattern cannot ship again.

Fixed:
- Milbrandt-Yau (`mp_physics = 9`) real cases run. Seven scheme-keyed
  tables lacked mp=9 arms: the preflight shape manifest, so an accepted
  configuration could not build its real-case workspace; the acoustic
  moist sum; the held-mixing slot registry; the nest forcing inventory;
  the coupled-scalar allowlist; the reflectivity consume membership; and
  the end-of-run digest's nest-slot membership, where a nested run
  integrated its full window and then reported failure. The last three
  also lacked arms for WDM6, P3 or aerosol-aware Thompson. The three
  hand-copied coupling sets and the digest membership now read one
  shared inventory, and a new stage-1 suite proves for every
  `mp_physics` value the loader accepts that the declared closure
  builds, classifies, couples, consumes its reflectivity and digests.
  WDM6's CCN pair and Milbrandt-Yau's hail-number pair joined the
  restart manifest with it.
- WDM6 and Milbrandt-Yau own the SR roundoff envelope WSM6 already had.
  The WDM6 kernel forms SR with the identical positive-sum expression as
  WSM6, but the validator granted the proven envelope to WSM6 alone, so
  a healthy WDM6 run died on WRF expression-order roundoff, one ULP
  above 1.0, at its first frozen-dominated column. The envelope is now
  keyed on the audited shared-expression family with an exact analytic
  bound per member. Morrison's expression provably cannot exceed 1.0 in
  float32 and keeps the tight range.
- RRTM longwave with Dudhia shortwave (1/1) finishes cleanly. The
  longwave adapter stored `p_top` as a NumPy scalar, which the restart
  classifier treats as an unclassified array, so every 1/1 run reported
  failure in the end-of-run digest after a perfect integration. The
  value is coerced exactly as the legacy-RRTMG adapter coerces its own,
  the composed pair's sub-adapters and the RRTM coefficient tables are
  classified, and a new suite walks every registered radiation callable
  through the restart classifier.

Known:
- The time-zero history frame publishes the declared constant downward
  longwave, 300 W m-2, where WRF writes 0.0 before the first radiation
  call. Recorded here for the surface-radiation carrier contract work;
  the first radiation call overwrites it.

## 1.9.0 (2026-08-10)

Six WRF schemes ported, plus the 1.8.9 recalibration batch and a
coherent water-temperature default on every forcing route.

Assembly notes for this line, including the inherited-red inventory and
the artefact regeneration order, are in
`docs/release-1.9-assembly-notes.md`.

New:
- NOAHMP_GLACIER ported (`module_sf_noahmp_glacier.F`, 3,080 lines,
  the byte-frozen WRF v4.6.1 tree). An active land column whose
  `IVGTYP` equals `ISICE_TABLE` now dispatches to the dedicated glacier
  column instead of refusing: `gpuwm/core/noahmp_glacier.py` is the
  FP32 host authority with file:line anchors,
  `gpuwm/core/kernels/noahmp_glacier.cu` the CUDA batch, bitwise twins
  over a multi-regime battery chained 12 steps. Pinned identity
  `opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_gla=1`; WRF's own
  balance gates (ERRSW/ERRENG 0.01 W m-2, ERRWAT 0.1 mm) are
  transcribed as raises. Two defined replacements for WRF's undefined
  reads (glacier-column HCPCT and EFLXB) are documented in the module.
  `guard_noahmp_glacier_columns` stays as the backstop and refuses only
  when the path is disabled. The step census reports `glacier` counts
  and the execution path that answered them.
- Noah-MP sea-ice threshold is configurable:
  `NoahmpRuntimeParameters(xice_threshold=...)`, threaded through
  `initialize_physics(xice_threshold=...)` and
  `run_mpas_column_batch(xice_threshold=...)`. WRF's Registry 0.5 stays
  the default; the chosen value joins the restart identity and the
  classification receipt.
- Native XLAND input: `initialize_physics(xland=...)` and
  `run_mpas_column_batch(xland=...)` consume the caller's XLAND
  verbatim; derivation from landmask remains only as the documented
  fallback when no xland is passed. `driver.surface_classification`
  (and the seam's `surface_classification` property) names which source
  decided and counts every class: sea ice, open water, SFLX land,
  glacier land.
- MPAS physics seam: `gpuwm.core.physics.run_mpas_column_batch` builds a
  persistent two-phase column-batch seam for an external MPAS CUDA core.
  Phase 1 runs legacy RRTMG, revised MO, Noah-MP, YSU and GF/KF through
  the ARW driver's own orchestration (held radiation, KF NCA holds,
  buckets, counters) and returns raw A-grid du/dv/dtheta/dq* rates with
  no mass coupling, face interpolation or map factors. Phase 2 runs WSM6
  in place on the caller's six species plus theta. Restart export and
  restore round-trip every persisted item and a restored seam continues
  bit-identically. Contract and semantics in `docs/mpas-seam.md`.
- WRF RRTM longwave (`ra_lw_physics = 1`), as WRF's classic pair with
  Dudhia shortwave (`ra_sw_physics = 1`). Line transcription of
  `module_ra_rrtm.F`; column smoke of the shipped seams only, no oracle
  comparison against the Fortran. This was the last schema-legal selector
  value that reached no accepted run.
- The NSSL 2-moment variant family: `nssl_hail_on`, `nssl_ccn_on`,
  `nssl_density_on` and `nssl_3moment` beside the default lane. Hail-off
  and diagnosed-CCN carry column smoke and treatment proofs, no oracle.
  `nssl_hail_on = 2`, `nssl_density_on = 1` and `nssl_3moment = 1` are
  refused by name.
- The MYJ PBL (`bl_pbl_physics = 2`) with the Eta similarity surface layer
  (`sf_sfclay_physics = 2`), admitted only as the 2/2 pair, which is WRF
  v4.6.1's own rule. Float32 CPU authority transcribed from the byte-frozen
  `module_bl_myjpbl.F`, with a CUDA translation unit agreeing inside a
  stated tolerance. TKE cold-starts at WRF's `epsq2` = 0.2. Declared
  divergence: interface heights are carried above ground, not above sea
  level, which cancels to within 69 ULP over 4.4 km terrain with `KPBL`
  unchanged. No oracle comparison against the Fortran.
- Milbrandt-Yau 2-moment microphysics (`mp_physics = 9`). Graupel and hail
  are separate prognostic categories and all twelve moments transport.
  Column smoke on three seeding layouts, water budget closing to 1.3e-4
  relative or better, plus a mutation control. No oracle. Per-domain
  override only.
- P3 one-category microphysics (`mp_physics = 50`). A 15-step 3-column
  integration stays finite and non-negative, conserves total water against
  surface precipitation to 1e-4 relative, and holds rime mass at or below
  ice mass with rime density in [50, 900]. WRF's
  `p3_lookupTable_1.dat-v5.4_2momI` is packaged verbatim and SHA-256
  validated at load; nothing yet checks that the interpolation of it
  reproduces WRF's. CPU only, no CUDA mirror. Per-domain override only.
  Siblings 51, 52 and 53 are refused by name.
- WDM6 double-moment warm rain (`mp_physics = 16`). CUDA kernel and
  `wdm6init` transcribed line by line from `module_mp_wdm6.F` with
  file:line citations, a float64 coefficient block pinning the kernel's
  baked FP32 literals, and a column smoke asserting WDM6's own bounds,
  water conservation to the surface flux, and CCN activation moving number
  from `nn` into `nc`. No oracle. WDM5 (14) and WDM7 (26) are refused by
  name. Per-domain override only.
- `mp_physics = 9` with RTE+RRTMGP is refused for absent cloud-optics
  coupling. WRF leaves `has_reqc`/`has_reqi`/`has_reqs` at 0 for
  MILBRANDT2MOM and the scheme's effective-radius block is commented out,
  so there are no scheme radii to hand RRTMGP. The refusal names two
  remedies and both are measured to work: `ra_rrtmg_variant =
  'rrtmg_legacy'`, or the Dudhia pair. Milbrandt-Yau under MYNN runs on
  either of them.
- `mp_physics = 50` with RTE+RRTMGP is refused for absent cloud-optics
  coupling. WRF sets `has_reqs` to 0 for P3 and its single ice category
  carries no separate snow species, so there is no snow radius to hand
  RRTMGP. The pairing was previously admitted and failed at the first
  radiation call. The refusal names the same two remedies as the
  Milbrandt-Yau one and both are measured to work.
- The composition walk grows to 9781 combinations, 990 accepted against 18
  registered templates and 8791 refused under 19 distinct rules. Every one
  of the 19 rules carries a demonstrated remedy pair, and every admitted
  value of every axis now reaches an accepted run.
- `water_temperature_policy` in `[case_data]` declares which provider decides
  water temperature: `era5_class_coherent` (the default that silence
  selects), `wrf_compat` (the historical per-cell selector, byte-for-byte,
  for stock-WRF certification and parity batteries), and `external_overlay`
  (the declared-overlay machinery, unchanged, which a declared overlay
  selects on its own).
- An advisory line names the policy, the water-cell count, the connected-body
  count and the per-provider cell counts on any domain the coherent policy
  touches, and states that a declared overlay with an observational analysis
  remains the higher-accuracy option. It warns and never blocks.
- `tools/water_temperature_probe.py` emits the decision surface for one
  domain of a case: mapped SST, mapped SKINTEMP, their difference, the
  validity mask the old selector consumed, the assembled field, the provider
  ids and TSK, under both policies side by side.

Fixed:
- Restart identity holds across process boundaries. The legacy RRTMG LW
  buffer march kept subtracting 4 mb below a column top shallower than
  the nominal p_top that sized it, `log(play)` went NaN in setcoef, the
  tropospheric layer count inflated, and the band kernels read outside
  their coefficient tables, so those columns' longwave answer depended
  on whatever freed device-pool memory happened to hold. A restore in a
  fresh process met different pool history than the run that wrote the
  checkpoint and continued differently. The march now takes the largest
  per-column step that keeps every buffer level positive (bitwise
  unchanged for in-contract tops), the setcoef kernel clamps
  nonpositive pressures before the log, and a pool-poison GPU test pins
  phase-1 bitwise invariance to freed-pool contents.
- WSM6 is priced at the tier it compiles rather than at a flat row. The
  kernel frame under-priced local memory above 64 levels by 446 MiB on
  4-domain LES configs. WSM6 and WDM6 both price per domain that selects
  them, at 112 and 152 bytes per level, so a 40-level WSM6 child beside a
  100-level parent takes its own bound.
- The WSM6 front-door level bound is bound to the kernel's own ladder
  instead of a transcribed constant, so the two cannot drift apart.
- The decoder door asks the bridge-contract question itself. Resolution
  checked existence only, which let a preparation run against a non-ABI
  stand-in.
- `PSFC` follows the surface switch rather than the existence of a driver.
  wrfout published a fabricated surface pressure for runs with surface
  physics off.
- A 0 K deep soil temperature is refused at ingest. The water fill passed
  every finiteness check and reached the land surface as weather.
- The duplicated periodic staggered face is pinned to face 0, exactly.
- Five NSSL selectors entered the restart identity of every run, including
  runs that select no NSSL scheme. They join `SCHEME_SCOPED_RUN_FIELDS`
  beside WDM6's pair, so an experiment written before these schemes existed
  keeps its fingerprint and its checkpoints stay resumable.
- Out-of-schema `mp_physics` values all recite the accepted menu. The P3
  siblings named their missing physics without it, so a value refusal and a
  composition refusal read the same to anything classifying by message.
- mp18's ring HAILNC accumulated where WRF's is exactly zero. The clipped
  tile guard's slot family did not carry the NSSL hail pair, so the ring
  retained hail the driver never wrote. Both hail-bearing schemes now share
  one slot family. The per-domain allocation estimate rises by 8244 bytes.
- P3 (`mp_physics = 50`) and MYJ (`bl_pbl_physics = 2` with
  `sf_sfclay_physics = 2`) can be run. Both were admitted by the config
  loader and absent from the memory preflight's kernel module tables, so
  `gpuwm check` refused them and a run died on the same call. P3 prices at
  the shared moisture validator, since it is a host transcription that
  launches no kernel of its own; MYJ prices at `myjpbl` and `myjsfc`,
  9,232 and 0 bytes per thread.
- The Milbrandt-Yau RTE+RRTMGP refusal reaches the registry. It lived only
  as prose in an option's extensions, so a launcher offered 320 component
  combinations that the runtime refuses after the prepare. Registry
  constraints gain a conditional kind, `refused_when`, because the coupling
  is a conjunction of a radiation option and an `ra_rrtmg_variant` value
  and none of the three existing kinds can state one.
- Lakes and coastal water no longer initialize from a per-cell mixture of two
  differently-mapped fields on the ERA5 route. The shipped chain chose SST
  where the mapped SST passed a 170..400 K test and ERA5 SKINTEMP everywhere
  else. METGRID.TBL maps SST with `sixteen_pt+four_pt` and `fill_missing=0.`,
  both operators need a fully usable stencil, and an SST analysis is missing
  over land, so every water cell within two source cells of a coastline took
  the fill and flipped to SKINTEMP. Over lakes ERA5 SKINTEMP is the FLake
  model state, several K warmer and quantized on the 0.25 degree cell, so the
  lake came out a quilt of two providers joined along the validity boundary.
- `gpuwm/ingest/water_temperature.py` now assembles the finished field before
  soil preprocessing runs. Surface classes come from the target statics, each
  class is labelled into connected components, and one provider is chosen per
  component: the ERA5 analysis on that component's own donors where they
  cover at least half of it, otherwise the coherent skin field for the whole
  body. Donors are selected by component identity, so a lake cannot take
  ocean water and no water cell can take a land donor. Every water cell
  carries a provider id in `WATER_TEMP_SOURCE`.
- Measured on Lake Erie 1985-05-31 12Z, 3 km nest, 2684-cell lake, in the 18Z
  forecast frame of this run and of the control: intra-lake adjacent-cell P99
  7.32 K to 0.13 K, cells stepping more than 1 K 5.18 % to 0.00 %,
  shore-to-midlake TSK gap +5.21 K to +0.46 K, lake TSK range 284.51..294.86 K
  to 284.49..289.41 K. The stock WRF v4.6.1 oracle at 1 km sits at P99 0.60 K
  and 0.65 % above 1 K, so the default now reaches oracle class. No declared
  file and no network access are involved.
- The guarantee is enforced at the soil preprocessing routers, the one seam
  every forcing route crosses, and not route by route. A route that hands the
  router a raw SST beside its SKINTEMP with no assembled water temperature is
  refused by name and told where the decision belongs. A route added later
  cannot reinstate the per-cell selector by omission.
- Lakes are named by the selected land-use table's own ISLAKE on every route.
  A table with no inland-water class is a declared state the advisory states
  out loud, never a silently empty lake mask.
- The met_em / rw-wps route and the 20CRv3 route that rides it now carry the
  lake mask and the policy into the assembly, so the lake-never-takes-ocean
  guarantee holds there and not only on ERA5.
- The GFS route forwarded no assembled field into soil while printing the
  coherent-policy advisory, so the historical per-cell selector ran behind a
  receipt saying it had not. The route now assembles after its lake skin is
  resolved and the router consumes that field. A test refuses the advisory on
  any route that does not consume what it assembled.
- Water temperature under `era5_class_coherent` interpolates the source
  analysis, so cells where the mapped SST was valid are not bit-identical to
  the previous default. Measured on the reproducing case: mean +0.002 K, max
  0.169 K, 0 of 1955 bit-identical. Unverified on a real coastal ocean domain.
  Declare `wrf_compat` to reproduce an archived run byte for byte.
- The ERA5 direct adapter run with `--static-input` and no `--geog-root`
  cannot resolve a land-use table, because the prebuilt static cache carries
  numeric fields only. That run says so in one line and keeps the historical
  selector. Passing `--geog-root` restores the default treatment.
- Connected-component labels are a pure function of two invariant statics but
  were recomputed for every forcing time, 0.19 s at 550 squared and 0.61 s at
  1000 squared on the launch-to-first-plot path. They are computed once per
  domain and reused.

## 1.8.9 (2026-08-10)

1.8.8 was tagged but never published: the RW-WPS staging gate refused the
standalone preprocessing wheel, so nothing reached PyPI. The tag stays where
it is, because tags here are forward-only. Everything 1.8.8 carried ships
here, plus the first fix below.

New:
- MYNN is selectable for a real forecast. Three shipped suites pair the
  MYNN PBL and MYNN surface layer with RTE+RRTMGP longwave AND shortwave,
  one per land surface the no-radiation MYNN family already covered:
  `wsm6-mynn-mynn-noah-rte-rrtmgp-implemented-unverified-v1`,
  `wsm6-mynn-mynn-ruc-rte-rrtmgp-implemented-unverified-v1` and the expert
  `wsm6-mynn-mynn-noahmp-rte-rrtmgp-expert-only-v1`. Every named MYNN suite
  before this ran shortwave with longwave off, which is refused over a night
  window unless declared, so picking MYNN from a menu meant picking the
  nocturnally invalid class. Each new row is its no-radiation sibling with
  only the radiation block changed (`ra_lw_physics` 0 to 4, `ra_sw_physics` 1
  to 4, `radt` 1.0 to 12.0). MYNN's pinned option identity is unchanged.
- Physics composability is now a measurement.
  `docs/public/receipts/physics-composition-walk.json` records 6536 physics
  combinations pushed through `gpuwm.experiment.build_experiment` one at a
  time: 749 accepted (741 distinct suites against 18 registered templates),
  5787 refused, 15 distinct refusal rules, and zero accepted runs whose
  resolved `RunConfig` differs from what the file asked for. Every admitted
  value of every selector reaches an accepted run except `ra_lw_physics = 1`,
  which is unported and says so. Regenerate with
  `tools/report_physics_composition_walk.py --table`.

Fixed:
- The standalone RW-WPS preprocessing wheel builds again, which is why
  1.8.8 was withdrawn. Two imports added during that line reached modules
  the wheel does not carry, one inside a refusal that fires when a config
  loads: a standalone user who tripped it got an import error instead of
  the sentence written for them. Both are resolved at the source rather
  than waived, and the staging scan that caught them now runs in the
  first-stage battery instead of only at a release cut.
- Two refusals you could not act on. `km_opt = 2` with a PBL scheme on
  offered `km_opt 3/4`, but the branch above refuses 3 for the same reason;
  it names `km_opt = 4` now and says why 3 is not an option. The coupled
  LW/SW adapter rule, the most frequent refusal in the configuration space,
  named no key and no offending value; it opens with `ra_lw_physics=<got>`
  and `ra_sw_physics=<got>` now and lists the admitted pairs. A standing
  check requires every refusal to name a selector and to lead somewhere that
  runs.
- `docs/public/PHYSICS.md` read as though the shipped suites were the only
  suites. It
  says on the page now that the tables are a catalogue and not a gate,
  defines `reachability.state` by quoting the registry, and prints what a
  config naming each registry-unreachable option actually gets: two are
  genuinely unreachable, three are accepted by the loader and merely off the
  menus. The revised MM5 surface layer was one of the three: the page said no
  config could reach it, while `sf_sfclay_physics = 1` is accepted and runs.
  The MYNN rows called the 5/5 PBL/surface-layer pairing mandatory both ways
  when only the surface layer restricts its partner; what is pinned is MYNN's
  twelve `bl_mynn_*`/`icloud_bl` knobs, each with one implemented value.
  `tests/test_physics_md_reachability_claims.py` derives all of it from the
  registry, the walk receipt and `build_experiment`.
- `--materialize-authorities` no longer replaces a physics value your config
  states. It deleted all 26 profile-owned keys and rewrote them from the
  named profile, silently, into a file every later stage binds by hash: a
  config saying WSM6 with no longwave ran as Morrison with RRTMG and every
  downstream check passed. A named profile now SUPPLIES the keys a config is
  silent about and REFUSES the ones it states differently, naming each key,
  both values and both remedies, before the fetch. Silent agreement is
  unchanged.
- `gpuwm go` and `gpuwm run-plan`'s prepared route derive the forwarded
  `--physics-profile` from the whole config, not the root domain alone. The
  wizard's `--ladder` trees declare nests that deliberately depart from the
  root suite, so the root-only derivation composed a stage-1 command the
  refusal above is guaranteed to refuse. A config the profile contradicts
  nowhere carries the assertion end to end; one that says more runs unnamed,
  as its own suite.
- A refusal whose root already IS the named profile no longer offers the
  flag it just refused as the remedy. It says to omit `--physics-profile`,
  and no longer suggests editing an LES nest's resolved-turbulence keys back
  to a PBL suite.
- `ra_rrtmg_variant` is governed under a profile that resolves the RRTMG
  (4, 4) pair without pinning the variant. A config declaring the other
  variant kept it and ran legacy RRTMG under a profile named rte-rrtmgp,
  silently; it refuses by name now. A declaration matching the resolved
  variant is kept where it was written.
- `gpuwm certify` exited 0 on a metrics CSV with a header and no data rows,
  because every row condition is a statement about all rows and is true of
  none. A declared condition, `the_comparison_is_not_empty`, runs before the
  per-row ones and names what it found against what it required, and
  `rederive_verdict` refuses an empty comparison independently. The floor is
  one row and one comparison, so a run shorter than the band's longest lead
  is still a legitimate partial comparison.
- `gpuwm dual-run` reported `capsules are identical field for field` on two
  empty documents, and it is the only detector this project has for the
  silent memory corruption a card with no ECC cannot report. An empty
  comparison is refused by name and by arm, a zero-byte capsule is refused
  with its byte count, and every passing screen states its size
  (`... (71 compared quantities)`), with `compared_count` in the written
  report. A half-empty pair is still a divergence, not a refusal.
- The certification schema version moved with the condition set it declares.
  `the_comparison_is_not_empty` was added while `verdict_schema_version`
  stayed at `1.0.0`, and `rederive_verdict` compares that set for exact
  equality, so a genuine document from the previous release rederived
  `False`: the same silent answer a forged one gets. The version is `1.1.0`,
  `CONDITIONS_BY_SCHEMA_VERSION` carries the older set, an unknown version
  fails closed naming itself, and every refusal carries a reason. The
  minimum-comparison floor applies on every version.
- `docs/public/CERTIFICATION.md` listed seven conditions where the code
  declares eight, and described `dual-run` as two outcomes where there are
  three. Both are corrected, in `docs/public/DETERMINISM.md` as well, and a
  test pins the public condition table to
  `gpuwm.certify.verdict.CONDITIONS`.
- The bundled renderer is checked against a contract, like every other
  bundled binary. `rw_wrfbatch` was asked only whether it started, so two
  builds two days and 4 MB apart, neither from this tree, printed the usage
  line, were reported `verified`, and drew the plots. It answers
  `--abi` now with the handshake `rw_fetch` and `rw_nexrad` have always
  answered, pinned in `gpuwm.rustwx.RENDERER_ABI_MARKER`. `gpuwm doctor`
  reports a mismatch with the rebuild remedy and lets the run proceed,
  matplotlib being the documented fallback; the rust render suite skips a
  foreign engine rather than substituting it.
- `gpuwm render --engine rust` went around that check. Only `--engine auto`
  probed, so the caller who pinned the engine to be sure of the real
  renderer was the one who did not get it. Both forms ask the same
  question now: `auto` falls back with the reason, `rust` refuses with exit
  2. `--list-products` with no wrfout resolves its own engine and prints the
  same refusal instead of a traceback.
- The release battery's stage-1 list gained the other half of its radiation
  gate. It proved a shortwave-on/longwave-off pairing is refused and proved
  nothing about radiation running: forcing the Dudhia heating rate to zero
  left the entire pre-amendment list green (901 passed, 6 skipped).
  `tests/test_wrf_legacy_radiation.py` is listed now; it drives the
  production `PhysicsDriver` radiation seam on the CPU and fails on that
  mutation, and `tests/test_stage1_manifest.py` pins both halves.
- Downward longwave no longer has a silent default. Any run with
  `ra_lw_physics = 0` integrated a fixed 300 W m-2 GLW for its whole
  forecast: `initialize_physics` defaulted `glw = 300.0` and no production
  call site passed anything else. Radiative equilibrium at 300 W m-2 is
  269.7 K; a Gulf-coast October night runs near 410 W m-2, or 291.6 K. That
  deficit craters skin temperature, takes the surface saturation humidity
  with it, and produced the reported nocturnal dewpoint collapse. `glw` has
  NO default now: a longwave scheme owns the field, or the caller types a
  constant or hands over a source array, or nothing reads or publishes it.
  Anything else is refused, naming the selectors, the number it would have
  fabricated, and three remedies. WRF v4.6.1 refuses the same shortwave-only
  pairing outright (`phys/module_radiation_driver.F:2245`).
- Radiation entirely off is covered by the same rule, and is where gpuwm
  deliberately parts from WRF. That suite attached no radiation adapter,
  while Noah, Noah-MP and RUC read downward longwave every surface step, so
  the run integrated an undeclared constructor seed. WRF's answer there is
  0 W m-2, an absent atmosphere rather than a thin one, and a column under
  it cools without bound; gpuwm does not copy that. With a land surface
  attached the pairing is refused at load, and a declared run integrates
  `gpuwm.physics_compat.DECLARED_CONSTANT_GLW_WM2`, 300 W m-2, named in the
  run receipt. With no land surface nothing reads the field and nothing
  fires. A run with a longwave scheme is byte-for-byte what it was.
- A real experiment whose downward longwave would be consumed (a land
  surface with no longwave scheme) or published (shortwave on with no
  longwave scheme, so the GLW row reaches every wrfout frame) refuses at
  config load, at the one loader every front door shares, with its own
  acknowledgement: `constant-downward-longwave-v1`. The load guard and
  `initialize_physics` refuse the SAME selector set, so a config either fails
  at the door or runs. This is deliberately NOT the 1.7.1 nocturnal token,
  which is checked before any physics is inspected and so lifted this
  question too. Every resolved-configuration report names each domain's
  downward-longwave source on a `radiation.downward_longwave` line.
- `configs/era5_wrf_direct_proof.toml`, `configs/gfs_wrf_direct_proof.toml`
  and `configs/gfs_wrf_hierarchy_proof.toml` ship with real radiation (legacy
  RRTMG on both streams, matching their paired stock-WRF namelists, which now
  say the same) and carry no acknowledgement at all.
  `tools/ens_sweep/kdmx_case.toml`, a 04Z-10Z Iowa window that is night end
  to end, moves to `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`. The LES
  records, the battery shape smoke, the init demos and the MYNN no-radiation
  probes keep their physics, their bytes being the provenance of committed
  results, and declare the constant instead.
- A shipped config may not carry an acknowledgement token silently. Each one
  must be answered by a `# JUSTIFY <token>:` block of real length in the
  same file, gated by
  `tests/test_shipped_acknowledgement_justifications.py`.
- `gpuwm enprod` stamped `EXPERIMENTAL (v1.2 ensemble, uncalibrated)` onto
  every ensemble panel, frozen at the release the suite was written for, so a
  1.8.7 plot claimed to come from a 1.2 ensemble. The warning stays and is
  still true; the version comes from the running engine now.
- The wizard no longer writes the acknowledgement itself. `gpuwm domain` wrote
  `acknowledgements = ["asymmetric-radiation-nocturnal-window-v1"]` into the
  emitted `[experiment]` by itself, and every later door reads that line for
  the life of the file. It refuses an asymmetric profile over a night window
  now and names both ways forward; `--ack <id>` is what writes the line.
- The shipped LES configs no longer carry the setting that caused the 1.6.0
  anisotropic-mixing instability. 1.6.0 closed that defect with a criterion
  and left every example config on the wrong side of it: the two 250 m nested
  trees at `mix_upper_bound*(dz_max/dx)^2 = 0.702`, and both 100 m tornado
  trees at 4.23 on d04, 17x the limit and the exact configuration that
  aborted a run at step 5467 with `w = 239.48 m/s`. All four run
  `mix_isotropic = 1` on their LES children now. That key is inside the
  restart fingerprint, so these trees run from t = 0, and none has been
  re-scored: every published number for them was measured on the old bytes.
- The configuration a committed receipt was produced under is archived at
  `configs/frozen/`, pinned by sha256, and it loads. Each of the three
  records there declares the constant downward longwave its run always had,
  so the loader accepts the file, the sha256 gate passes, and reproducing
  one of those runs gives the same physics the archived run had. Those three
  are why `gpuwm check` passes on three more shipped configurations than it
  did. No physics selector moved and no committed receipt was edited, so a
  record now carries two digests: the as-run one its receipts name, and the
  on-disk one a reproduction passes. `configs/frozen/README.md` publishes
  both and says which is which. `docs/public/LES.md` and
  `docs/public/GRAYZONE-NEST.md` say now that their measured numbers belong
  to the archived configuration rather than to the configs those pages name.
- `tests/test_shipped_configs_mixing_stability.py` fails if any config under
  `configs/` arrives on the exposed mixing path, and it runs on every release
  cut; the criterion used to be checked only when somebody loaded a config
  and read stderr. The frozen records are the one exemption, each pinned to
  its content hash.
- The criterion is no longer skipped on a config that writes no eta ladder.
  `km_opt = 3` with `mix_isotropic = 0` at `dx = 100` m and no `eta_levels`
  is the exact shape the criterion exists for, and it used to load and pass
  `gpuwm check` in silence, reported clean. The depths are resolved the way
  the model resolves them now: uniform interfaces from `nz`, with `ztop`
  converted to a pressure by the exact inverse of the relation the depths
  are read with. This reverses a 1.6.0 ruling that skipped such coordinates;
  downstream, a skip was indistinguishable from a pass.
- Where no depth can be produced at all, the criterion says so instead of
  going quiet. A model top above the analytic base state's ~24.6 km ceiling
  has no representable pressure, so such a domain sits on the exposed path
  with no number to judge it by. There is still no ratio, because a
  fabricated depth in a stability criterion is worse than none; an advisory
  fires instead, naming what could not be resolved and the two remedies:
  declare `eta_levels` and `p_top`, or set `mix_isotropic = 1`.
- `gpuwm check` repeats the mixing advisory in its report and under
  `advisories` in `--json`; the 1.6.0 run had been warned at config load,
  hours before it died. The advisory is tiered now: a ratio that inverts the
  2dx mode (above 1/4) reads differently from one that grows it (above 1/2),
  so 0.3 and 4.2 no longer look alike. Still not a refusal.

CONTRACT CHANGE: a config that declares a suite and is materialized under a
different named profile used to succeed and now refuses. Edit the config's
physics to the profile's values, name the profile the config already is, or
omit `--physics-profile` to publish the config's own suite.

CONTRACT CHANGE, prepared-cache identity: a config declaring `radt_minutes`
keeps it now (no profile pins that key; the old materializer deleted it),
which moves `prepared_domain_config_identity` for that key. A prepared tree
built from such a config before this change must be prepared once more. No
shipped config declares `radt_minutes`, and the kept value is physically
inert under every shipped profile.

CONTRACT CHANGE, version identity: running gpuwm from a source checkout
while a DIFFERENT gpuwm version is pip-installed now refuses at config load,
because every receipt would otherwise be stamped with a release that is not
executing. Bind the tree to its own metadata with `pip install -e <tree>`, or
run the installed copy. A borrowed version that AGREES warns one line and
still runs, so ordinary worktrees beside an editable install are unaffected,
and a plain wheel install never sees this at all. The refusal fires at the
one load every front door shares, and it names both versions, both locations
and the one-line fix.

Changed (compatibility):
- **A real config that runs a land-surface model with no radiation is now
  refused at load.** This is a load-time break of files that worked before,
  and it is wider than "configs that set `ra_*_physics = 0`": radiation
  defaults to off, so **any** real config (one with a `[projection]`) that
  selects a land surface and simply OMITS a radiation selector is in the
  class. That is the larger group and the likelier surprise, so an omitting
  config is told `NO RADIATION SELECTOR SET AT ALL` and offered the missing
  line as the first remedy, instead of being quoted two zeros its author
  never wrote. Idealized experiments (no `[projection]`) are not guarded. To
  keep such a run, both claims have to be declared, in one edit, in
  `[experiment]`:

  ```
  acknowledgements = [
    "radiation-off-land-surface-v1",
    "constant-downward-longwave-v1",
  ]
  ```

  Two tokens, because there are two claims: this run has no radiation
  scheme, and the number its surface integrates is one the config declares. The
  first token alone is still refused. Two shipped configs are in the class,
  and each declares the pair in-file with its reason.
- **Two refusal layers, in a fixed order, and an exit code that tells them
  apart.** The declaration guard above answers at config load, before any
  route logic runs: a token-less radiation-off config never becomes an
  experiment at all. Route admission (`gpuwm.hrrr_route_inputs`) answers
  second, and only about configs that loaded. Tools carrying both report it
  the same way `tools/battery_route_preflight.py` does: exit 2 with no
  receipt for the config error, exit 1 with a `REFUSED` receipt naming the
  gate for a route-refused run.
- **The radar-DA storm nowcast's default physics profile changed, and it
  roughly doubles the VRAM plan.** `wsm6-ysu-mm5-noah-no-radiation-v1`
  (shortwave on, longwave off) was the default of a product whose cases are
  overwhelmingly nocturnal; it is now the HRRR route's own
  `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`, because the nowcast's
  background is HRRR. On the shipped 132x132x49 3 km parent that moves the
  machine peak envelope from 3606 MiB to 6993 MiB (+3.3 GiB, 1.94x) and the
  allocation estimate from 618 MiB to 3499 MiB. **A member count or VRAM plan
  measured before this has to be re-measured, not extrapolated.** All six
  surfaces move together, including `tools/da_nest_cost.py`, the route's
  documented VRAM gate, which imports the constant now rather than restating
  it. `--physics-profile` still takes any shipped profile.

## 1.8.7 (2026-08-08)

New:
- The first plot lands while the forecast is still running. The first
  frame a forecast commits is the analysis -- WRF's history alarm is
  true at `t = 0`, so it is written before a single step is integrated
  -- and it used to sit finished on disk until the finalize stage,
  which waits for the last timestep. A run that sets `render_products`
  now renders that frame the moment `output_committed` fires for it, on
  a worker thread, concurrent with the forecast. Default ON where it
  applies and inert everywhere else: a plan naming no products (the
  default) or `none` behaves exactly as before, and the `experiment`
  route has no such option. There is no second switch.
- Every run receipt now carries a time-to-first-plot number.
  `first_products_ready` names the frame, the pictures and
  `seconds_from_plan_accepted` -- wall clock from the instant the plan
  was accepted, which is the instant the person who launched it started
  waiting, to the instant the pictures were readable. It is measured by
  the engine rather than by a stopwatch outside it, and it is emitted
  before the frame is digested and before the receipt is written,
  because both are finalize bookkeeping and hashing a 362 MB history
  frame would have put a second of it inside the number. The
  `completed` event repeats it as `first_products_seconds`, null and
  not zero when a run published nothing early, so comparing two runs
  does not mean scanning two event streams.
- Finalize does not redo the early work, and does not take that on
  trust. The early render leaves `first-products.json` naming the frame
  it read and every PNG it wrote, all by sha256. Finalize drops that
  frame from its own list only when the frame still hashes to the
  recorded digest, every recorded picture is on disk hashing to its
  recorded digest, and `render_products` has not changed. Any other
  answer -- a deleted picture, an edited frame, a different product
  spec, a render still running -- and the frame is rendered again. Six
  mutations are pinned to void the claim. Pictures are published by
  `os.replace` out of a scratch directory, so a reader watching the
  output directory sees a whole PNG or none.
- The release battery's stage-1 curated list ships in the repository,
  at `tools/battery/stage1_files.txt`, gated by
  `tests/test_stage1_manifest.py`. Until now that list existed only as
  an array inside per-assembly scratchpad scripts, propagated by copy
  from one assembly to the next: a lost scratchpad silently lost every
  amendment ever made to it and nothing anywhere failed when it did.
  The file carries each entry's reason inline, the gate checks that
  every path exists and nothing is listed twice, and the two entries
  whose absence caused a real miss are pinned by name against a
  tidy-up. This release is the first assembly to consume it, and the
  first to amend it in the repository rather than in a scratchpad.
- Analysis-hour-first fetch ordering is now a contract rather than an
  accident. Every ladder builder already returned an ascending range
  from the forecast-start lead and every fetch loop already walked it
  in order, so reordering buys zero seconds -- there was nothing to
  reorder. The property is pinned because it is what any future overlap
  work stands on, and pinned alongside it is the property such work
  would have to preserve: `SHA256SUMS`, which preparation binds through
  `--source-manifest-sha256`, is byte-identical whatever order the
  hours land in. That is proven by fetching the same window forwards
  and backwards and comparing bytes, not by reading the `sorted` call
  that makes it true, and a second test keeps order independence from
  becoming order blindness by requiring the receipt to claim only hours
  that completed.

Evidence: measured on a real seven-frame `d01-12km` run, 362 MB
analysis frame, products `refl,t2,wind10,precip`. Early render of the
analysis frame alone, 17.0 s and 4 pictures; the finalize render that
used to do all seven frames, 116.0 s and 28 pictures; the finalize
render of the remaining six, 100.0 s. So the split costs nothing
measurable and the 17 s hides under a running forecast, while the first
plot stops waiting for the last timestep. All four early pictures are
byte-identical to the same frame rendered inside the seven-frame batch.
The renderer never touches the card: it is a separate `python -m
gpuwm.cli render` process driving the CPU-side Rust binary, and it
creates no CUDA context -- `cuCtxGetCurrent` returns
`CUDA_ERROR_NOT_INITIALIZED`, interrogated from the driver rather than
read off the module table. Handing the committed frame to the worker
returns to the writer thread in 0.0007 s, and every telemetry path
swallows into a `warning` so no render can fail a forecast. Why the
larger overlap win was not taken -- preparation is one pass that seals
the initial condition and the boundary tables together, and the sealed
manifest binds the role set, so preparing from a subset is a different
digest against a runner that binds exactly one -- is recorded in
`docs/run-plan.md` rather than left for the next reader to rediscover.

## 1.8.6 (2026-08-08)

New:
- Moving nests run on both prepared routes, driven from `gpuwm
  run-plan`. 1.8.4 landed the statics corridor on `gpuwm go` and the
  tree runner; what was left was the front door. The prepared GFS chain
  already composed `--statics-corridor` on its prepare stage and nobody
  had checked it from this door, so that is now pinned by driving the
  real chain for a follow plan and reading the composed command back.
  The prepared HRRR chain could not do it at all, and now does.
- `gpuwm.hrrr_hierarchy_direct` takes `--statics-corridor`, bare or as
  comma-separated child grid ids, spelled exactly as the GFS door
  spells it. It seals the set into
  `hierarchy-artifacts/statics-corridor/` -- the same relative path the
  GFS bundle uses -- and binds its receipt into `receipt.json` beside
  `artifact_receipt`, exactly as the GFS chain binds it into
  `proof.json`. It is that stage and not `tools/prepare_hrrr_wrf`
  because a corridor is child-resolution statics over the parent
  extent, and the root preparer never sees a child: the hierarchy stage
  requires `--geog-root`, builds `d02..dNN`, and already holds the
  verified GEOG catalog. Without the flag the receipt has no such key
  and the bundle is byte-for-byte what it was.
- The tree runner reads `statics_corridor` out of whichever hierarchy
  document matched its pinned digest and cannot tell the two chains
  apart, which was true by construction in 1.8.4 and is now a
  measurement: the corridor refusal matrix -- no corridor, uncovered
  child, failed verification, accepted -- is parametrized over both the
  GFS `proof.json` and the HRRR `receipt.json`.
- `gpuwm run-plan --resolve` reports the decision instead of leaving a
  caller to guess or launch and find out. A `moving_nest` record names
  the chain, the delivery, the relocating grid and whether a corridor
  will be built, with a matching `automatic_resolutions` entry (scope
  `preparation`, basis `relocation_follow`) whose note names the stage
  that actually carries the flag -- on HRRR that is the hierarchy
  stage, and "the prepare stage" would have sent a reader to a tool
  that refuses it. `moving_nest` is `null` for a config that moves no
  nest.
- `gpuwm run-plan --estimate` prices the corridor. It is the one
  preparation artifact whose size cannot be inferred from the domain
  sizes a caller already has, being parent extent at child resolution:
  a 45x45 nest at `parent_grid_ratio = 3` on a 398x320 root is a
  1194x960 corridor at 889 MB. The new `corridor` block prices every
  child, because run-plan passes the flag bare and the preparation
  reads that as every child domain, and it prices them through the
  preparation's own child selection and the corridor module's own
  arithmetic rather than a second copy of either. `corridor_cost` is
  held equal to a real build's sealed `host_bytes` by a test instead of
  by agreeing today. The corridor adds no GPU residency, so the `vram`
  block and the VRAM gate are unchanged with and without it; that was
  verified by pricing one experiment on both chains and requiring the
  two figures to be equal, not assumed.

Fixed:
- One emitter, one child selection, one predicate. Both chains now call
  `gpuwm.static.corridor.emit_statics_corridor_set`, the child
  selection moved to `validated_corridor_selection` in that module
  (`source_hierarchy` held one copy and run-plan's pricing reached in
  past a leading underscore to read it), and the follow predicate that
  had three copies kept equal by a drift test is now the single
  `config_declares_follow_source` that `gpuwm go`, the printed `rw-wps`
  line and run-plan all call. The tree runner's own inline copy of that
  predicate is gone -- it was the one place that decides whether a
  bundle is accepted, holding its own reading of the sentence every
  preparation door reads from the module.
- A moving nest on a nested HRRR tree no longer fetches, prepares a
  root and builds a hierarchy before dying at the forecast preflight.
  `_FOLLOW_STATICS_DELIVERY` declares per chain where a moving nest's
  statics come from -- corridor, live ingest, or nowhere -- under a
  completeness guard, and `_execute_prepared_route` dispatches through
  the same `_chain_key` the refusal is decided on, so a config cannot
  be judged as one chain and run as the other. With HRRR moving from
  `None` to `statics_corridor` the refusal table is empty: no chain
  this front door dispatches to refuses a moving nest today. The
  refusal machinery and both completeness tests stay for a chain added
  tomorrow that cannot feed one, and the refusal is now called directly
  so that retained machinery is exercised rather than merely retained.
- `docs/run-plan.md` no longer says nested HRRR is refused. It shipped
  in c45e24924 without the doc catching up, which by 1.8.6 directly
  contradicted the section above it.
- `docs/run-plan.md` now says not to touch a checkout a run is reading
  from. The HRRR hierarchy stage samples `git status --short` when it
  publishes and refuses a tree with anything uncommitted in it, which
  is deliberate -- a bundle stamped with a commit it was not built from
  is a false provenance claim -- but the stage runs minutes into a run,
  `git status --short` counts untracked files, and a commit landing in
  the worktree mid-run therefore kills the run.

Evidence: crop-versus-direct bitwise identity re-proven for a corridor
built through the HRRR chain's own grids and catalog, 5 placements x 14
fields, with the instrument checked by a planted single-ULP flip
caught as exactly one differing word. Emission determinism, the
receipt-bytes round trip, a tampered-cache refusal, and the real
`prepare_hrrr_hierarchy` sealing and binding. Live on a real 18Z HRRR
cycle: both moves executed off the sealed corridor, the parent bitwise
unchanged, 42 of 42 fields bit-identical against direct builds, and the
`--estimate` figure equal to the sealed bytes exactly.

## 1.8.5 (2026-08-08)

Fixed:
- HRRR no longer publishes a negative 2 m mixing ratio over high
  terrain. A nested 12-3 km tree over eastern Colorado cleared both
  preparation stages on 1.8.4 and was then refused by the tree
  forecast's own input check, `prepared near-surface surface_qv is
  outside the physical range 0.0..0.2`. The guard was right and the
  data was WRF-faithful, which is why nothing upstream caught it.
  HRRR's GRIB2 packing quantises 2 m specific humidity to 1e-5, so
  over source orography of 2557..3535 m it decodes to exactly zero
  beside neighbours three orders of magnitude larger; METGRID.TBL
  routes SPECHUMD through the overshooting `sixteen_pt` operator,
  which undershoots such a stencil below zero, and
  `module_initialize_real.F` converts it with `qv_gc = sh_gc/(1 -
  sh_gc)` and no floor before assigning `grid%q2` verbatim. Two child
  cells came out at -1.79e-05 and -1.48e-05. Nested GFS completed the
  same placement only because its lane builds the same quantity
  through `_saturation_mixing_ratio`, which floors at WRF's
  `qv_min_value` inline; the specific-humidity lane had no floor at
  all, and that asymmetry is the whole defect, since one guard written
  against the floored lane's contract was applied to both. The
  published `surface_qv` is now floored at that same `qv_min_value`,
  at the publication point and nowhere else: `sfcprs2`, `integ_moist`
  and the vertical-interpolation pseudo-level are uses real.exe is
  defined for and keep the raw value, so no prognostic field moves on
  any lane, including the explicit `use_sh_qv=True` lane whose
  prognostic qv is the interpolated surface value. Cells at or above
  the floor are byte-untouched, so no existing artifact changes; 1 of
  63 arrays moved in the refusing tree's cache and a control tree was
  byte-identical across all 270. The divergence is recorded on
  `RealInitResult.surface_moisture_floor` and printed, like the
  soil-moisture floor beside it. Single-domain HRRR carried the
  identical latent defect on the same lane behind the same guard, and
  was unexposed only because the shipped cell runs over flat Oklahoma;
  a 3 km root over the same Colorado terrain reproduces both negative
  cells. The near-surface refusal now names the offending cell count
  and the observed extremes, because reading them back off the old
  message cost a full re-run of a two-stage preparation.
- The two `km_opt` registry citations point at the LES-closure
  docstrings again. 1.8.4 shipped with `tests/test_physics_registry.py`
  red: both anchors in `gpuwm/core/dycore.py` were 29 lines stale
  because the solver perf work added the cached `couple_momentum`
  kernel well above them. The code moved, not the claim, and the two
  want opposite fixes, so it was checked before repointing: both
  docstrings are byte-identical between 1.8.3 and 1.8.4, as is the
  whole of `dycore.py` from the first anchor to end of file. Only the
  line numbers move, 824 to 853 and 914 to 943, in the registry
  warning text and in the checker's paired RESOLVED rows, which must
  move together or the checker fails from the other side. The citation
  checker now reports 85 citations with 0 failing, and the curated
  first-stage battery carries that test so the next drift is caught by
  a release gate rather than by a sweep.

## 1.8.4 (2026-08-08)

New:
- `gpuwm run-plan` runs nested HRRR end to end, the last route that
  refused. An HRRR tree is not a GFS tree with a different source:
  rw-wps is not on the path at all, the root preparation is
  `tools.prepare_hrrr_wrf`, and a third stage builds d02..dNN from the
  sealed d01 before the same `gpuwm-prepared-tree-forecast` the GFS
  tree route already drives takes over. The branch is taken after the
  root preparation, which both HRRR depths share, so neither fetch nor
  that preparation is duplicated. The hierarchy writes `receipt.json`
  where rw-wps writes `proof.json`, and the document resolver added
  for the GFS tree matches on schema rather than filename, so the
  receipt relay needed no new machinery. Render is now one helper
  shared by both HRRR arms, so `render_products` -- including `none`
  -- means the same thing however the forecast was produced.
- Moving nests run on the prepared routes. A prepared tree is
  digest-bound and carries no ingest inputs, so a `[relocation]`
  follow source had nothing to rebuild child statics from and the tree
  runner refused it outright. `--statics-corridor` closes the gap at
  preparation time, when the geography source is still on hand:
  parent-extent child-resolution statics per child, built through the
  same `build_static` the domain statics use, written
  byte-deterministically and sealed with a SHA-256 receipt. Corridor
  crops are bitwise the direct footprint build, pinned by a
  crop-vs-direct identity test across the build's gcell, categorical,
  interpolation and smoother paths with a planted one-ULP
  perturbation proving the instrument fires. At forecast time a
  follow source is accepted only over a verified corridor (receipt
  against proof, cache digest, geometry and grid-arithmetic probes);
  a corridor-less bundle keeps the refusal and now names the flag,
  and a failed verification refuses loudly rather than running a
  silently static nest. Opt-in throughout: without the flag the
  bundle is byte-for-byte what it was. `gpuwm go` and run-plan pass
  it whenever the config declares a follow source, and the printed
  rw-wps chain derives it from the same predicate, so the pasted and
  the driven chain cannot drift on it.
- `prepare_mapped_wrf` accepts a prebuilt native static cache
  (`--static-input` / `--static-receipt`). It was the last direct
  adapter with no such bypass: it called `build_static` every cycle
  and then serialized the exact artifact it had no way to read back.
  The receipt is verified against the resolved cache and the target
  grid before the load, and the proof records `root_static_provider`
  as `prebuilt-hash-bound-cache`.
- The solver step and the preprocessing that feeds it are faster, with
  every output byte unchanged. The surveyed arm took the
  representative full-physics trace from 41.345 to 37.514
  ms/model-step, and a fused coupled-scalar update, a fused
  `couple_momentum`, a Morrison active-span substep sweep and a
  chunked state gate landed on top of it, for about 13% off the step
  in total. `validate_full_state` on a seeded 250x200x49 mp10 state
  went 5.25 to 0.52 ms per check. `build_static` against the
  reference WPS_GEOG tree at d01 251x201 went 18.701 to 5.359 s
  (3.49x), which pays on every path a static cache cannot serve: the
  cold first cycle, geometry changes, moving and spawned nests, and
  child domains. The byte proofs: forecast SHA-256 unchanged on the
  seeded mp10 and dry lanes, ValidationReports bitwise identical
  across a 14-case corruption-injection battery, and all 14 static
  field digests identical in every arm.

Fixed:
- Nested GFS preparation no longer refuses on inland water. An
  ordinary 12-3 km tree died 15.9 s into its prepare stage with 73
  columns disagreeing between land mask and soil category, and the 73
  were reservoirs and rivers scattered across the whole child rather
  than an edge artifact, so this refused essentially any CONUS child
  holding inland water, moving nest or not. The soil-temperature
  lookup was an inline chain that did not know the GFS spelling; it is
  now one table of per-source spellings read by both reconciler call
  sites, the nested child and the root case. The SST argument had no
  fallback and the GFS lane carries no SST field, so WRF's second arm
  could never fire; it now takes real.exe's own precedence, SST then
  the skin temperature that `module_initialize_real.F` substitutes
  where SST has no valid support. The reconciler's soil-temperature
  and SST reads also went straight to `np.asarray` on values that can
  arrive device-resident, which CuPy refuses; both are marshalled to
  the host now, like the rest of that module. That refusal was
  unreachable until a column actually disagreed, so it sat behind the
  reconciler's own hot path instead of failing at setup.
- A chain stage no longer imports from the caller's directory. Every
  prepared-chain stage is spawned as `python -m MODULE`, and `-m`
  prepends the current directory to `sys.path` ahead of the installed
  package, so a chain started inside a source checkout imported that
  checkout; a live run was hijacked exactly that way. The stage cwd is
  deliberately the caller's directory so a relative `--out` means what
  the person typing it meant, so moving it was not the fix.
  `PYTHONSAFEPATH` separates the two: the child still resolves paths
  against the caller's directory and no longer imports from it.
- Progress telemetry reports real numbers on every prepared route.
  `speed_x` was null and `wall_seconds` 0.0 on all 181 progress events
  of a completed run, because the baselines were armed on the stage
  transition and every prepared chain opens the forecast stage itself
  before handing the runner over. They arm on the first progress call
  now, which is true however the stage was opened.
- A completed tree run no longer summarizes itself as zero. The
  summary knew only the single-domain runner's `progress.json` and
  `report.json`; the tree runner writes a certification capsule and
  neither of those, so a run that finished 10800 model seconds and
  wrote 17 frames reported `completed_seconds` 0.0, and the heartbeat
  fed from it published a model time of zero beside an `outer_step` of
  180. Completed seconds now come from the observer, which saw every
  step on either route, and the capsule is read for frames and named
  in the summary. Where no receipt states a status, status stays null
  and `status_basis` says why.
- `gpuwm run-plan --resolve` no longer reports an authored table as a
  schema default. It decided "the author did not type this" by looking
  only inside the `[experiment]` table, but several fields are
  authored as their own top-level table, so a config declaring
  `[relocation]` was reported with the schema's `enabled: false`. For
  a moving-nest plan that reads exactly backwards. A top-level table
  whose name is a field name now counts as the author spelling that
  field, taken from the document rather than from a second list.
- `relocation_receipts.json` carries one run-end summary row. On a
  leg-walking route the executor returns once per leg, so a live
  12-move run left two byte-identical summary rows in a 25-row list.
  The superseded row is dropped and the new one appended, so the
  summary stays last and current.

## 1.8.3 (2026-08-08)

New:
- `gpuwm run-plan` runs nested GFS end to end. A multi-domain config on
  the `prepared` route keeps the same preparation stages (rw-wps builds
  the whole hierarchy in one call) and dispatches the forecast to
  `gpuwm-prepared-tree-forecast`, with run-plan owning the receipt
  relay the manual chain needed a person for: the sha256 of the
  hierarchy document rw-wps left in the prepared root, matched on
  schema against the tree runner's own table, plus the experiment
  config's own digest. The interactive `gpuwm go` command keeps its
  tree refusal; only run-plan, which dispatches to the tree runner,
  drives trees. Multi-domain HRRR remains refused with its by-hand
  chain named.
- `gpuwm run-plan --estimate` reports `peak_envelope_bytes` beside the
  pool request. That is the tree-aware figure: it adds a per-nest term
  and, on WDDM, the measured footprint floor. A tree priced on the
  pool request alone reads as fitting a card it does not fit.
- `model_progress` events carry each grid's own clock on a domain tree
  (a `domains` list of `{domain, model_seconds}`). Absent on a single
  domain, where the root is the tree, so existing consumers see the
  stream they always saw.

Fixed:
- Native HRRR prepared runs complete on Windows. The prepared-cache
  identity keyed source digests by `str(path.relative_to(...))`, which
  is backslashed on Windows, and the reader looks names up by
  forward-slash constants, so the forecast handoff always failed with
  "decode identity omits [...]" on a cache that was in fact complete.
  Every producer now writes POSIX keys and the validator reads both
  spellings, so caches already sealed on Windows under 1.8.2 stay
  restorable. A dict carrying one path under both spellings is refused
  rather than merged.
- `sealed_extension_fingerprint` no longer differs between Windows and
  Linux for identical code: the tree runner's runtime source identity
  used OS-native separators in its keys. Linux fingerprints do not
  move; Windows converges onto the existing value.

## 1.8.2 (2026-08-08)

1.8.1 was tagged but never published: its release workflow's own test
job failed on the runner, so PyPI received nothing and the release was
withdrawn to a draft. The tag stays where it is, because tags here are
forward-only. Everything 1.8.1 carried ships in this release, plus the
fix below.

Fixed:
- `gpuwm doctor`'s printed report no longer depends on whether the box
  running the tests has an NVIDIA driver. The report suite substituted a
  stand-in for `sys` that carried two attributes, and the CUDA-major
  remedy added in 1.8.1 reads a third (`sys.platform`). Whether anything
  reached it depended on `GPUWM_NO_LOCAL_GPU`, so the gap was
  unreachable on a developer box and fatal on a driverless runner. The
  stand-in now delegates every attribute it was not deliberately given
  to the real module, and the report test pins the driver arrangement
  itself instead of inheriting the host's.

## 1.8.1 (2026-08-07, tagged but not published)

New:
- `gpuwm run-plan` reaches HRRR. The `prepared` route takes
  `source = "hrrr"` for a single domain, so the credential-free machine
  path is no longer GFS-only. 1.8.0 shipped this route explicitly scoped
  to GFS and said so; the preparation side is what had to change, not
  run-plan.
- A plan chooses what to render. The `render_products` run option takes
  the product names a run should draw instead of drawing the default
  set, so a fleet controller asking for four fields does not pay for the
  whole catalog.
- `gpuwm run-plan --catalog` prints the render catalog as one JSON
  document and runs nothing, the same shape as `--resolve`,
  `--estimate` and `--probe`. A front end offering a product picker can
  populate it from the build rather than from a list it maintains.
- `gpuwm version` says which code is actually running: the import path,
  whether it resolves inside a checkout or a site-packages install, the
  distribution version, and whether pip would move it. An install whose
  behaviour does not match its version is the first thing to rule out in
  a bug report, and it was the one thing no command reported.
- The GPU install extra names its CUDA major. CuPy ships one wheel per
  CUDA major and the wheel must match the major the box serves: a
  `cupy-cuda12x` wheel on a CUDA-13-only box imports cleanly, compiles
  kernels, and then dies at the first cuBLAS load. A pip extra cannot
  detect a CUDA major, so this does not pretend to choose one:
  `gpuwm[gpu-cu12]` and `gpuwm[gpu-cu13]` name it, with `[all-cu12]` and
  `[all-cu13]` beside them. `[gpu]` and `[all]` still resolve to cu12,
  as aliases rather than a third policy, because every install that
  already works names them.
- `gpuwm doctor` reads the box's CUDA major straight off the driver,
  with or without CuPy installed, and prints the extra that matches it.

Fixed:
- A supervisor guard refusal reads as a refusal. A run refused by a
  physics or configuration guard surfaced as a crash capsule, so the
  headline a user saw was a stack trace rather than the sentence the
  guard wrote for them.
- A `[relocation.follow]` cadence that the reflectivity stash cannot
  serve is refused when the config loads, not at the first cadence that
  needs the fallback. The tracker's composite-reflectivity fallback
  reads a plane the microphysics stashes at history cadence, so a follow
  cadence off that lattice ran fine until the first evaluation where
  updraft helicity was under threshold and then refused mid-run.
- A two-phase DA run that stops early still writes the receipt its
  verifier needs, so the run can be graded instead of being unscoreable.
- A domain tree that cannot be resolved is left to its own validator
  rather than being reported by the tracker, which is not the component
  that knows what is wrong with it.
- An island domain whose source land-sea mask rounds all its land away
  now initializes from the real land fraction instead of refusing. WPS's
  `make_zero_or_one` binarizes the mask, so a sub-grid island whose land
  fraction never reaches 0.5 anywhere in an all-ocean crop left the
  binarized donor set empty over the whole domain, and the nest took a
  METGRID-fill `TSK = 0.0`. The fall back to the discarded fraction fires
  only when that donor set is empty across the entire crop, and it says
  so in a per-domain receipt. The nonphysical-TSK refusal that used to be
  the only symptom now names the offending cells and the fill that
  produced them. The CONUS reference domain is byte-identical across this
  change.

## 1.8.0 (2026-08-07)

New:
- Storm-following moving nests. A `[relocation]` table, default off,
  moves a child domain by a whole number of parent cells at cycle
  boundaries. A moving nest here is a sequence of static nests joined by
  a re-grid, not WRF's continuous per-step motion; those namelist keys
  stay refused and now point at the discrete mechanism instead of
  saying "non-goal". The overlap between the outgoing and incoming
  footprints is transplanted bitwise, because a whole-parent-cell shift
  leaves every overlapped cell's donor index and sub-cell offset
  unchanged: measured 0 mismatches over 18,714,960 cells across 25
  restart-contract fields on an RTX 5090, with the treatment proven by
  56 % of those same cells differing from a cold start. A restart across
  a move promises nothing, by ruling, and the relocation bounds stay out
  of the restart fingerprint.
- `[relocation.follow]` decides the WHEN and the WHERE from the running
  model's own fields. Updraft helicity (`up_heli_max`, WRF's
  UP_HELI_MAX, reset each history interval) is the primary vote, because
  rotation is what a tornado nest exists to follow; column-max
  reflectivity is the fallback for the window before a mesocyclone
  exists. Three configured guards keep a jittering centroid from
  spending the run on relocation spin-up: a dead band below
  `min_shift_cells`, a per-event clamp at `max_shift_cells`, and a
  cooldown that burns on executed moves rather than proposals. Every
  cadence appends a receipt carrying the centroid evidence and the
  decision, so a run's move/hold history is auditable afterwards. Every
  follow key is required and unknown keys refuse.
- A relocated nest rebuilds its statics for the new footprint, at nest
  resolution, from the nest's own static source rather than inheriting
  parent-interpolated terrain: over-land storm-following at 2 km lives
  on resolved terrain. The footprint grid comes from an exact
  whole-cell translation of the reference grid, so a cell two placements
  share evaluates to bitwise-identical coordinates and therefore
  bitwise-identical statics; the preparer asserts that equality on every
  move rather than assuming it, which is what lets the bitwise overlap
  transplant survive a statics rebuild. The atmosphere on the fresh
  strip is adjusted to the rebuilt terrain by the same sequence a t = 0
  real child runs, and land state moves by landmask-aware donor fill.
- Off-centre nest placement is proven placement-independent, in both
  directions. The parent-to-child exchange geometry carries no centred
  placement assumption: force tables are exact on linear fields at every
  quadrant and at the minimum legal edge margins for ratios 3 and 5, the
  residual tables are identical between centred and off-centre
  placements, and the whole force transaction is translation-equivariant
  at the bit level. The one genuine concentric-nest assumption in the
  tree was an LES case module's own placement instrument, and it now
  uses the runner's resolver. A relocated nest is off-centre by
  definition, so this is the moving nest's standing regression bed.
- Spawn-at-trigger nests. A `[[domain]]` can declare a `spawn` block --
  `trigger = "uh" | "reflectivity" | "time"` with a threshold and a
  model-time window -- and stay dormant until the trigger fires, at
  which point it materializes at the tracker-chosen position and follows
  the storm from then on. The nest is pre-declared, not mid-run
  allocated: the memory plan prices it exactly as if it existed, so
  `gpuwm check` refuses honestly at planning time and prints one
  advisory line per dormant nest naming what its declaration costs.
  Spawning is activation, not allocation. A declared-but-never-triggered
  nest costs its reserved VRAM for the whole run and zero compute; that
  is the contract, not a leak.
- The fired nest gets own-grid statics for its footprint, an atmosphere
  interpolated from the CURRENT parent -- so it is born inside the storm
  its trigger saw, not inside a stale analysis -- and terrain adoption
  byte-for-byte the real-data child adjustment sequence. Placement is
  the storm-core weighted centroid around the loudest qualifying cell in
  the search box, so with two storms in one box it lands on the stronger
  storm rather than between them, and a watch ignores signal inside
  another active nest's footprint.
- A nest born mid-run takes its land surface state the way WRF gives one
  to a nest with no `wrfinput` of its own: parent interpolation through
  the mask-aware operator WRF names per field in its Registry, so a
  mixed land/water cell averages only the corners matching the child's
  own land-use class. That is the operator WRF runs at every nest birth
  and at every moving-nest leading edge. ArWen keeps the half of
  `fine_input_stream = 2` it can have, the own-grid statics, which is
  what makes the mask load-bearing: the child's categories are resolved
  at the child's dx and disagree with the parent's wherever finer
  terrain resolves a coast, lake or island the parent smoothed away. The
  receipt counts every branch per mask family and publishes the
  island/lake fallbacks. This lifted the real-data spawn refusal on the
  run route; routes that neither reserve nor watch still refuse a spawn
  declaration by name rather than dropping it.
- `gpuwm run-plan`, a machine interface. One versioned JSON plan in, one
  append-only JSONL event stream out, so a GUI, a scheduler or a fleet
  controller drives gpuwm as a subprocess and never parses a line of
  human output. The plan is an envelope over the existing config system,
  not a second config format: it resolves through the same loader
  `gpuwm run` uses and executes through the same runtime. Unknown
  top-level keys, unknown run options, an unknown route and an unknown
  schema are all refused rather than ignored, and `fetch.args` is
  validated by building gpuwm's real fetch parser.
- `config.intent` submits a shape instead of a config -- a point or
  polygon, a ladder, a source, a cycle, a card budget -- and the `gpuwm
  domain` wizard writes the complete TOML, the same wizard with the same
  refusals and the same emitted bytes a person gets from the CLI. Every
  intent key is a wizard flag, one to one. The generated TOML is carried
  verbatim on the `resolved_plan` event, because the caller never typed
  it and it is the one thing they cannot look up.
- Every value the pipeline chose on its own appears in
  `automatic_resolutions` with its basis, one entry each, before the run
  starts. The per-domain time step is the flagship case: it changes the
  model's answer, nobody writes it, and until now it appeared only
  inside a printed report. `cycle: "latest"` is resolved to a concrete
  cycle before the fetch runs and the resolved value is what gets
  recorded, so a plan does not archive a question whose answer changes
  every six hours.
- `--resolve`, `--estimate` and `--probe` each print exactly one JSON
  document and run nothing. They work before anything is downloaded, and
  they carry the full declared-input inventory instead of skipping the
  existence check silently. `--probe` reads the device inventory through
  NVML only, creating no CUDA context, so it is safe to poll on a busy
  card. Where this package has no measured number the field is `null`
  with its basis stated, rather than an invented one under gpuwm's name.
- A run-plan run is reattachable. `run-manifest.json` is written before
  any work starts and names every stream a consumer may want, including
  the two run-plan does not own. run-plan publishes no progress state of
  its own: the supervisor stays the only writer of `run-progress.json`,
  the run-plan observer composes with the existing heartbeat rather than
  replacing it, and a run-plan run leaves exactly the heartbeat a `gpuwm
  run` leaves. `sequence` on the event stream is monotonic and dense, so
  a gap means a lost line and the reader refuses it; a torn final line is
  refused rather than trimmed.
- `gpuwm domain --history-interval` and `--nest-history-interval` make
  output cadence a first-class control. They were bare literals with no
  knob. A cadence must be a whole number of seconds and a whole number
  of that domain's own time steps, judged against the exact rational
  `dt`, and the wizard round-trips its emitted bytes through the real
  loader before writing, so a bad value is refused with the loader's own
  sentence and no file lands.
- The HRRR route stages microphysics lookup tables for every profile
  whose microphysics reads them, at profile-binding time -- before the
  fetch and before preprocessing -- through the packaged table ladder,
  SHA-256 checked. It resolved tables for one profile behind two
  environment variables before, which is why its default had to be a
  suite that needs none. The resolved set is recorded in the physics
  receipt as `microphysics_table_authority`.
- `gpuwm domain --source hrrr` defaults to
  `thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1`: Thompson microphysics,
  RRTMG longwave and shortwave, no cumulus at 3 km. That is the
  operational HRRR composition (NOAA/GSL CCPP `HRRR_suite`); gpuwm
  diverges on YSU rather than MYNN-EDMF and Noah rather than RUC.
- The wizard offers both legacy-RRTMG Thompson suites. It offered
  neither, which is why no full-radiation suite could be an HRRR
  default.
- A sizing refusal names a lighter `--physics-profile` beside the
  shallower ladder and the bigger card.

Fixed:
- No door emits an asymmetric radiation pairing (shortwave on, longwave
  off) as a default. This was corrected to exclude HRRR in 1.7.1; it is
  now true everywhere, and the claim's test covers all three sources
  instead of two.
- The HRRR route no longer writes a nocturnal experiment declaration on
  the operator's behalf for its own default. A declaration by silence
  declares nothing. Explicitly selecting an asymmetric suite for a night
  window still writes it, which is what it always should have meant.
- A bare interactive `hrrr` session emitted a config its own route
  refused (`d01 cu_physics=1`, Kain-Fritsch on a convection-permitting
  grid). Its covering test checked registry reachability and maturity,
  both true of that suite, and never ran the route's physics gate; it
  does now.
- An HRRR config carrying a legacy-RRTMG suite could not be emitted at
  all: the emission round-trip re-imported its namelists without the
  RRTMG variant, resolved 4/4 to RTE+RRTMGP, and reported the
  difference as though the emitted files were wrong. It inherits the
  variant from the authoritative experiment, as it already did for
  acknowledgements.
- A wizard header called every `ra_*_physics = 4` suite RTE+RRTMGP,
  including the legacy-RRTMG ones.
- An installed-but-unloadable CuPy no longer kills the command line. An
  absent CuPy raises `ImportError` and was always survived; the
  documented rental trap -- a `cupy-cuda12x` wheel on a CUDA-13 box, a
  missing NVRTC DLL -- raises `RuntimeError` or `OSError` from deep
  inside the import, so `gpuwm` died on exactly the installs
  `run-plan --probe` exists to diagnose. It is deferred, not swallowed:
  the original error is kept and named at the use site, with the
  wheel/CUDA-major mismatch called out, because installed-and-broken has
  a different fix from absent and must not get the "install it"
  sentence.
- A `prepared` run-plan run that succeeded reported `failed`. The
  chain's own exit convention was read as the run's.
- Nothing but JSONL reaches stdout on a run-plan run. The resolved-config
  report, the wizard's sizing table, warnings and the feedback advisory
  all go to stderr, enforced by binding the real stdout for the event
  stream and redirecting `sys.stdout` for the whole run, rather than by
  asking every printer to behave.
- `gpuwm go` runs its forecast in process, so the documented chain is
  observable end to end instead of going dark inside a subprocess.
- Output cadence no longer steers a storm-following nest. The tracker and
  the spawn watch read updraft helicity, and they had been reading WRF's
  `UP_HELI_MAX` diagnostic, which is zeroed by the history writer. Its
  accumulation window was therefore the output interval, so changing
  `history_interval_s` changed the peak and the centroid the tracker saw,
  and the nest went somewhere else. Each consumer now owns a window
  folded from the same cal_helicity columns in the same pass and reset by
  that consumer at every evaluation, accepted or held, so the window is
  exactly "the strongest rotation since I last looked". Two windows, not
  one, because relocation evaluates on cadence boundaries and spawn on
  leg boundaries: a shared window would let whichever consumer read first
  blind the other. Thresholds keep their meaning, and the shipped demo
  configs are unaffected because their cadences were already aligned.
  The windows are not restart-carried: a restart starts them empty and
  the first evaluation after it may under-read, the same posture already
  ruled for a restart across a move or a spawn.
- The offline child route refuses `ra_rrtmg_variant='rrtmg_legacy'` with
  `o3input = 2` instead of silently using the wrong ozone. That pairing
  means "take ozone interpolated from the parent", and this route has no
  resident parent; it was building the radiation adapter without one,
  which made it evaluate a fresh climatology on the child's own latitudes
  and then report `"ozone_routing": "root-climatology"` for a nested
  domain. `o3input = 2` is the default and `gpuwm downscale` copies it
  from the parent, so this was the ordinary path. The refusal names both
  remedies.

## 1.7.1 (2026-08-06)

Fixed:
- The wizard's real-case default physics is
  `morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1` on the gfs/era5 doors:
  the registry's user-ready `wrf-matched-run` template with both
  radiation streams on, the profile FIRST-LIGHT's worked example and
  the interactive session already used. It replaces the unshipped
  "product default suite". The HRRR door keeps its route-constrained
  WSM6 default. Neither the gfs nor the era5 door emits an asymmetric
  radiation pairing (shortwave on, longwave off) as a default. (The
  HRRR door was fixed in the following release; see Unreleased.)
- A real experiment whose window includes local night refuses to load
  with `ra_sw_physics > 0` and `ra_lw_physics == 0`, naming the
  physics, the matched profile, and both remedies. A field report
  proved the failure: a wizard-emitted 48 h case bound
  `thompson-mp8-ysu-mm5-noah-validation-v1` (longwave OFF, shortwave
  Dudhia), shortwave heated the surface by day, nothing balanced the
  surface's upward longwave at night, skin temperature cratered, and
  2 m dewpoints read in the 50s F inside a 70s airmass. The guard
  lives in the one config load every front door shares
  (`gpuwm.experiment.build_experiment`), so `gpuwm run`, `gpuwm go`,
  `gpuwm check`, both prepared runners, the DA drivers and the
  wizard's own sizing loop all refuse identically. Existing configs
  carrying the pairing refuse at load with the remedy named.
- Asymmetric pairings stay selectable, loudly: a daylight-only window
  loads unguarded, and a night window runs as a declared experiment
  with `acknowledgements = ["asymmetric-radiation-nocturnal-window-v1"]`
  in `[experiment]`. The wizard writes that declaration itself when an
  asymmetric profile is selected explicitly for a night window, so the
  file it emits still loads everywhere.
- The native HRRR route writes the same declaration. Its default suite
  is asymmetric by route constraint -- the HRRR root preparer stages
  no microphysics tables for the full-radiation profiles, and eight of
  the thirteen profiles it accepts run Dudhia shortwave with longwave
  off -- and it builds its experiment in code rather than from a config
  file, so it declares a night window itself and the experiment
  authority it publishes carries the line. Five full-radiation profiles
  remain selectable on that route for a nocturnally valid HRRR run.
- Every wizard-emitted config header now states whether its suite is
  nocturnally valid, and `docs/public/PHYSICS.md` carries a
  per-profile nocturnal-validity table.

## 1.7.0 (2026-08-06)

New:

- Radar data assimilation for nowcasting, as a product surface. A
  WoFS-style ensemble Kalman filter cycles an ensemble against live
  observations and then runs free forecasts from the analysis.
  `gpuwm-da-nowcast run --da full` selects the certified full-stack
  configuration in one flag: radial velocity, reflectivity, clear-air
  echo, velocity dealiasing, surface observations and GOES cloud water
  path. `--without <stream>` subtracts one stream from that preset, so
  a denial experiment is one word rather than a hand-assembled flag
  list, and `--da vr` is radial velocity alone. The default stays
  `custom`, meaning the individual flags remain the whole story for
  anyone already driving them.
- Every assimilated stream is counted. The cycle records how many
  observations each stream actually contributed, and a full-stack claim
  is refused when a stream contributed nothing, so a run cannot report
  a configuration it did not perform.
- Velocity dealiasing on the radial-velocity path. Folded gates above
  0.8 * Nyquist carry a mesocyclone's couplet, and masking them removes
  the signal the analysis is for. Gates the unfolder cannot resolve are
  still dropped and counted. The same unfolding is applied to the
  observation-scoring composites, so a run is graded against a truth field
  built the way it was fed. Needs the new `[dealias]` extra.
- GOES cloud water path is assimilated beside radar, from the model's
  own column integral, and an observed-clear column can remove cloud
  the model invented.
- Validation against MRMS observations. `--truth mrms` grades a run against the
  national mosaic rather than against a composite assembled from the
  same radars that fed it, which is the honest grader for a domain
  wider than one radar's reach.
- HRRR is the default background for the nowcast route.
- Adjusted initial conditions: `[[perturbation.bubbles]]` in the
  experiment TOML adds warm theta bubbles to a real-data initial state
  at prepare time -- WRF's idealized cosine-squared bubble, placed in
  geographic coordinates, with optional `rh_preserve` qv adjustment.
  Applied per domain through each domain's own init, so a bubble
  inside a nest arrives on the nest's own grid. An absent block is
  byte-inert; refusals are loud (out-of-domain center, nonpositive or
  oversized amplitude, a bubble that touches zero cells); what was
  actually written lands in `initial-perturbation.json` per domain.

- Warm bubbles. A `[perturbation]` block applies theta bubbles to the
  initial state, on the single-domain route and on the prepared
  domain-tree route. An absent block changes nothing at all, including
  the restart fingerprint.
- High-resolution static geography for US interior domains: 10 m
  terrain, 30 m land cover and 250 m soils, fetched as on-demand tiles
  with recorded provenance. Off by default. Coastal domains refuse, or
  fall back to the 30 arc second baseline when asked to.
- The render vocabulary grows from 55 products to 164, with proper
  colortable resolution: an operational table where the product has
  one, then a curated table, then a generic full-finite-range style.
  Any stored 2-D plane with no named product renders as `var:<name>`.

Fixed:
- The turbulence selector `km_opt` is a registry component carrying an
  option for every value it accepts, including 0. A plan selecting the
  SASE closure, which requires `km_opt = 0`, previously failed to
  resolve the component at all.
- `km_opt = 3` is admitted with the PBL off only. Its vertical exchange
  pair is applied by a PBL-off-gated operator, so with a PBL scheme on
  only the horizontal half of the closure ran, under the name of the
  3-D one.
- The registry and `validate_run_config` now agree about every one of
  the 46,080 registered component combinations. They disagreed about
  3,944 of them: 3,360 where the registry refused `km_opt = 3` under a
  PBL and the loader did not, 512 where the SASE closure was paired
  with the MYNN surface layer that WRF admits only under the MYNN PBL
  or none, and 72 dry SASE plans the registry called launchable and the
  loader refused.
- An enabled `[static.highres]` block on a route that cannot apply it
  refuses by name instead of silently producing the baseline.
- The native-HRRR static verifier accepts a whole-second rational-clock
  payload, which it had been refusing outright.
- The nested-domain child soil reconciler reads the native lane's soil
  temperature field.

- Large-eddy runs with WRF's anisotropic mixing lengths no longer go
  unstable and abort. With `mix_isotropic = 0` the mixing lengths are
  per-axis, so on a grid whose vertical spacing is far finer than its
  horizontal spacing the vertical exchange coefficient is applied
  through a horizontal operator and amplifies the 2dx vertical-velocity
  mode until the run fails. The condition is now stated as a criterion
  on `mix_upper_bound * (dz_max / dx)^2`, a configuration that violates
  it warns by name at setup and says which of the two knobs to move,
  and `mix_isotropic = 1` mixes on one length instead of three.
- CUDA kernel compiler version is part of the certified numeric
  fingerprint. A silent NVRTC downgrade moved the CUDA column's
  measured distance from WRF on two fields with no source change, and
  nothing in the capsule recorded it. The capsule now carries the full
  four-part NVRTC build, its internal changelist, and the SHA-256 of
  the loaded NVRTC library; the shinhong parity baselines and the
  off-path byte pin are recorded per NVRTC build rather than as one
  row.
- `--surface-obs` without a sigma carried the file and assimilated
  nothing.
- The dealiasing account is a per-radar list, so a multi-radar run
  reports each radar's unfolded and dropped gate counts instead of one
  collapsed number.
- scipy is declared in the new `[dealias]` extra, which the velocity
  dealiaser needs. It was imported but undeclared.

## 1.6.3 (2026-08-06)

New:

- An optional high-resolution water-temperature overlay for the ERA5
  route. ERA5's water-surface temperature is a coarse analysis, and
  over lakes and along coasts it paints through to the near-surface
  fields as blocky dewpoint and temperature artifacts. This is a
  well-known WRF + ERA5 limitation, not something specific to this
  model's ingest; docs/water-temperature-overlay.md cites the community
  threads and their remedies. The overlay is the high-resolution
  substitution remedy implemented natively: a user-supplied gridded
  water-temperature analysis replaces ERA5 SST and SKINTEMP over water
  source cells before horizontal interpolation, on the direct adapter
  route and on `gpuwm run`, with per-snapshot replacement counts
  reported. Off by default; configured on nothing, this code never
  runs.

Fixed:

- Certification capsules on CUDA-13 boxes no longer report the CuPy
  version as unavailable. The capsule pinned the literal distribution
  name cupy-cuda12x; it now records whichever CuPy distribution pip
  actually installed (cupy, cupy-cuda12x, cupy-cuda13x).

## 1.6.2 (2026-08-06)

New:

- A `[gpu-cu13]` install extra for CUDA-13-only machines. The existing
  `[gpu]` extra stays on CuPy's CUDA-12 wheel, which a CUDA-13-only box
  imports cleanly and then fails at its first cuBLAS load, deep inside
  a run. `gpuwm doctor` now performs that first load on purpose, in an
  isolated subprocess probe, and when it fails it reports the installed
  wheel's CUDA major, the box's, and the exact pip commands that put
  the matching wheel on. The load is the judgment, not the version
  numbers: a newer driver serving an older wheel is a working install
  and is never refused. Under `GPUWM_NO_LOCAL_GPU` the probe does not
  run and the report says the pairing went unjudged.

Fixed:

- Shoreline land cells next to water no longer initialize far below
  wilting-point soil moisture on the ERA5 route. Ingest built the soil
  column with a different soil category than the land model then
  integrated: the ingest/land-surface category reconciliation ran after
  the soil moisture floor had been applied, so a coastal cell whose
  category changed kept a floor that belonged to neither category. One
  rulebook now resolves the reconciled category before the soil column
  is built, on the root domain and on nest initialization, and the
  floor reads the reconciled category's own air-dry value. On the
  domain this was verified on, every affected cell initializes at its
  category's air-dry value and no other cell moved.
- The demo/gallery renderer's shapefile reader (pyshp) is declared in
  the `[render]` extra. It was imported but undeclared, so a fresh
  install that followed the quickstart exactly could still crash at the
  first basemap it drew.

## 1.6.1 (2026-08-05)

Fixed:

- ERA5-forced runs on coastal domains could still die at step 0 with a
  non-finite tendency, which 1.6.0 did not fix. Where the land mask and
  the soil category disagree, an ordinary occurrence at a coastline, a
  land cell carries the water soil category. That category has no
  air-dry value, so 1.6.0's air-dry floor skipped exactly those cells
  and left soil moisture at 0.0, which Noah's thermal conductivity then
  divides by. On the reported domain 192 cells were in that state, and
  are now none: a land cell whose category has no air-dry value takes
  WRF v4.6.1's own 0.005 constant, and real land keeps its category's
  air-dry value as before. A run also refuses at the start, by name,
  when a required forcing variable is absent.

## 1.6.0 (2026-08-05)

New:

- LES completion program, packages P1 to P4. Every verdict below was
  scored against acceptance bands registered before the runs existed.
  - Moist LES verification (P1): matched cloud-topped boundary layer
    comparisons against a pristine-source WRF oracle, with the
    saturation mutation control and both negative controls fired. 29 of
    32 metric-arm comparisons landed inside the registered bands; the
    two findings behind the three misses are published beside the
    passes, and no band was widened.
  - Vertical levels (P2): the LES level bound is now a compiled tier
    ladder. Every configuration at or below 128 levels compiles the
    kernel it always did, proven bit-identical; 160-level and 192-level
    runs carry a full device acceptance battery including restart
    bit-identity; above 256 the solver refuses loudly before any launch.
  - Inflow seeding (P3): an LES nest child can seed inflow turbulence
    with cell-block boundary perturbations, per domain, default off,
    and the off path is gated byte-identical to a build without the
    mechanism. The measured need is on the record: the shipped nested
    child's turbulence development fetch is larger than its own domain.
  - Gray-zone chain (P4): the scale-aware parent chain campaign ran end
    to end and ships with its findings section as the status of record.
    The gray-zone nest configuration ships unblessed and says so.
- Observation scoring instrument, first light. A registered reflectivity
  skill battery (fractions skill score against radar composites, frozen
  parameters, a measured persistence floor, and a wrong-day negative
  control measured on real observations) plus the scoring command line.
  Exactly one end-to-end case is scored: model S_refl 0.4618, above the
  persistence floor of 0.1598 and below the registered useful-skill line
  of 0.5249. This is first light for the instrument. No campaign ran and
  none is claimed.
- Radar data assimilation foundation, experimental. Ensemble
  initial-condition perturbations, radar observation operators,
  hot-start reflectivity increments, a batched LETKF gated by a twin
  experiment, a hydrometeor positivity policy, ensemble products, and a
  production NEXRAD Level II pipeline from archive bytes to filter-ready
  observation files. Every entry point stamps status experimental and
  nothing is wired into a default forecast route.
- WoFS at Home (WaH), a radar-DA nowcast that runs on one card. It is
  demo-grade, and that label is a measurement rather than modesty; the
  paragraph below it is what the label means. One command takes a
  WSR-88D site from Level II bytes through ensemble analysis to a
  rendered forecast gallery. An interactive launcher draws the domain as
  a box on a map instead of writing one. A daemon cycles the whole thing
  continuously and says on the page when it stops and why. The
  observation route reads the real-time chunk feed, which publishes
  bytes as they are collected rather than when a volume closes, and
  falls back to the archive when the live prefix is short; a finished
  volume is byte-identical on the two routes. Nested forecasts ride the
  same path, and the analysis carries its own Jacobi eigensolver so it
  does not need cuSOLVER present to solve.
  Exactly one case is verified. KDMX, ten members plus a never-analysed
  control, 3 km spacing, six 15-minute cycles and a 90-minute free
  forecast, 506 s for the whole three-hour exercise on one RTX 5090. One
  radar. Radial velocity only, which is the weaker half of radar DA,
  because reflectivity is what places and maintains storms. Two analysis
  variables, u and v. Fractions skill score 0.72 to 0.77 against 0.24 to
  0.34 for a cold start that never saw an observation: an internally
  controlled measurement against doing nothing, and not a comparison
  against persistence, optical-flow nowcasting, WRFDA, or the
  operational HRRR. One draw per arm, so no error bar. No dual-run byte
  comparison was made for any DA run, so the standing no-ECC corruption
  screen has not been applied to any of these numbers. No velocity
  dealiasing: fold risk is masked and counted, not unwrapped.
  [docs/da-nowcast-quickstart.md](docs/da-nowcast-quickstart.md) is the
  path a stranger can follow, and
  [docs/da-vs-wofs.md](docs/da-vs-wofs.md) sets the system beside
  Warn-on-Forecast line by line, including every line where it is
  smaller.
- Grell-Freitas cumulus, `cu_physics = 3`. A scale-aware deep and shallow
  convection scheme, selectable per domain on the tree route. Bitwise
  against WRF v4.6.1 at the whole-driver boundary over a committed
  216-column oracle: 208 columns word for word on the float32 CPU
  authority and on CUDA, the remaining 8 bounded at 34 ULP by the
  driver's own mixed-precision constants. Never a default and no
  template selects it. Not verified in any real-case forecast: no scored
  comparison against observations has been run with it, so the registry
  label is implemented-unverified with scientific evidence none.
  Selecting it costs 3.20 GiB of device local-memory backing store,
  priced by `gpuwm check` under NON-POOL, which today rules out
  four-domain configurations on a 32 GB card.
- OLR renders as synthetic satellite infrared in the matplotlib fallback
  renderer.

Fixed:

- ERA5-forced runs could die at step 0 with a non-finite tendency. An
  ERA5 soil-moisture value a hair below zero, from GRIB packing or
  interpolation undershoot, was clipped to exactly 0.0 on a land point.
  Noah's thermal conductivity divides by soil moisture three times, so
  one such cell produced NaN conductivity, then NaN ground heat flux,
  then NaN sensible heat flux, and the run failed blaming the PBL
  scheme. Each land layer below its soil category's air-dry value is now
  floored to that value before the derived liquid-water split, on every
  Noah-geometry source. Water, sea ice and healthy land are byte
  untouched: on a mixed land, water, ice and frozen domain all ten state
  arrays hash identically with and without the fix. Where WRF v4.6.1
  differs it is recorded rather than imitated: WRF resets a whole column
  to a constant on a top-layer trigger, this floors per layer at the
  category value, and on the case that was reported 109 of 142,644 land
  cell-layers differ as a result.
- The refusal that run produced named the wrong scheme. A non-finite
  surface heat flux handed to YSU leaves as a non-finite tendency, and
  the message named only that tendency, pointing at the
  boundary-layer scheme when the defect was in the surface layer feeding
  it. 1,474,560 fuzzed columns of finite input never produce a
  non-finite tendency at all, so when one appears an input is the cause.
  The refusal now names the degenerate input it is an image of. Which
  tendency carries it depends on surface stability: over a stable
  surface the heat flux reaches the potential temperature tendency and
  nothing else, which is the shape that run reported; over an unstable
  surface it also sets the convective velocity scale, so it reaches the
  momentum and moisture tendencies too and the refusal names the wind
  tendency instead. The named input is the heat flux either way.
- Two silent wrong answers on CuPy's compile route, measured on sm_120,
  where FP32 subnormals are flushed in each of the six mechanisms the
  receipt measures. A friction velocity of 1e-13 is an ordinary float32
  whose cube is subnormal and flushes to zero, turning a quotient into
  NaN and laundering it into an exchange coefficient of 1000 m2/s where
  the float64 authority says 131; validation saw nothing. And a
  roughness length small enough to overflow the surface-layer
  logarithm's quotient sent an infinity into every downstream similarity
  quantity. Both now have defined answers, proven bit-exact on the
  healthy path under pinned fixtures and an adversarial sweep. Where WRF
  v4.6.1 leaves the degenerate case undefined, the defined behaviour is
  documented as a divergence rather than reproduced.
- A failure capsule can be read without the tree that produced it. It
  now embeds its own configuration, namelist and vtable text, size
  capped. A first-step failure is also labelled as stepping rather than
  as writer initialization, which had sent one real user-bug diagnosis
  down the wrong path.
- Twelve defects in the observation emitters and scoring path, found
  while hardening the instrument: among them frame selection under radar
  outages, a frozen station table, an archive manifest that describes
  the archive rather than one pull, and the WRF comparison arm being
  told to produce the reflectivity it is scored on.
- Release receipts record exactly the 16 bytes that are a device UUID;
  they previously carried three neighbouring bytes that can change when
  the adapter re-enumerates.
- The release snapshot builder answers --help instead of running.
- `gpuwm doctor` checks for the NEXRAD front door by name, and that
  binary now ships in the bridge bundle. A machine without it previously
  got a green report while being unable to ingest a single radar
  observation, which is every DA route, live and archived. Absent blocks
  and says what it blocks. Stale says to rebuild rather than to
  re-point, because every such binary reports the same version string,
  so pointing at another copy fails identically.
- The NoahMP column-slab preflight priced its device transient by CuPy
  pool growth, which reads zero whenever the pool already holds a block
  big enough to serve the request. It now reads allocation demand at the
  allocator boundary. No gate was widened: the negative controls at
  1,024 and at 16,384 columns both still fail.
- 91 restart gates that need no device were being skipped by a
  whole-module GPU mark, and a collection that crashed under the marker
  leak gate made that gate pass vacuously instead of failing.
- One scorer of record for the skill battery. The two scorers already
  shared the kernel, but each carried its own copy of the constants, and
  the copies disagreed where it matters: the same named half-width is a
  27 km box at 3 km spacing and a 13.5 km box at 1.5 km. The constants
  are now derived in one place, proven inert against the committed
  composites, and held there by a guard that fails when a literal is put
  back.
- A sweep receipt described its scored field as one member when the
  published figure is the ensemble mean. The label is corrected and no
  number was recomputed.

High-resolution terrain statics (hash-bound raster overrides for
topography) are present and unchanged in this release.

## 1.5.2 (2026-08-03)

ArWen gains its first scale-aware boundary-layer scheme: Shin-Hong 2015,
`bl_pbl_physics = 11`. A namelist asking for it now gets it â€” native
import, no substitution to YSU â€” and the scheme's subgrid/resolved
partition adapts continuously with the grid spacing, which is what the
1 km-to-100 m "gray zone" between mesoscale and LES resolutions needs.

The evidence ships with the claim. The CPU reference implementation is
bitwise against WRF v4.6.1's own module â€” max ULP 0, every output
column, both terrain-flux arms â€” and the CUDA port holds the heating
tendency bitwise on top of it. Beyond WRF-fidelity, the partition
itself was measured: a 3200 m â†’ 100 m grid-spacing ladder, six
independent seeds per rung, scored against acceptance bands registered
before any run existed, drawn from the published similarity envelope.
Every gated rung landed in its band, and at 100 m the scheme reproduces
standard behaviour to within measured noise â€” scale-awareness in the
gray zone without corruption at the LES end. The full ladder â€” every
seed, every band, the determinism digests â€” is at
[docs/public/receipts/grayzone/](docs/public/receipts/grayzone/README.md),
and [PHYSICS.md](docs/public/PHYSICS.md) states exactly what is proven
and what is not. The honest label stays **implemented-unverified**: no
matched free-running forecast comparison has been run with it, and the
ladder is one idealized case on one card.

Surface-layer pairing rules mirror WRF's own â€” Shin-Hong runs with
`sf_sfclay_physics` 1 or 91, and a namelist pairing it with anything
else is refused by name with the values that would work, citing the
line of WRF that enforces the same thing.

One small fix rides along: the note printed when a GFS fetch domain
crosses 0Â° longitude now says it is informational â€” the full-band
widening is handled, the only cost is download size, and the run
continues unchanged. Field reports were reading the old wording as an
error.

## 1.5.1 (2026-08-03)

A hardening release. The published wheel's first week in the field â€” a
1 km nest on an RTX PRO 6000, cross-architecture runs on an RTX 4090, a
5060 Ti, and real HRRR forecasts on an RTX 5070 Ti â€” sent back every
rough edge fixed here, and one thing the 1.5.0 notes had to stop
claiming is now true.

GFS gained the full-file transport HRRR has had all along: `gpuwm fetch
--source gfs --mode full-file` takes the whole `pgrb2.0p25` objects
from the AWS S3 archive â€” no NOMADS rate governor, no spatial crop, and
the archive's reach is years where NOMADS keeps about ten days â€”
through the Rust backbone's parallel range GETs or the stdlib
transport. The raw objects differ from the NOMADS crops in row order
*and* packing; both differences are certified in `gfs_grib2_bridge`
against committed matched pairs of the same fields from the same cycle,
including a bitmap-carrying SOILW pair that pins complex-packing
missing-value semantics cell for cell, bit for bit. Everything outside
that proven envelope still refuses by name, the NOMADS crop remains the
default transport unchanged, and `gpuwm doctor --source gfs` now
reports the GFS route â€” decoder and byte transports â€” the way it
reports HRRR's.

The wizard now emits what the runner accepts: a bare `gpuwm domain`
produces one 12 km domain â€” the shape `gpuwm go` runs â€” instead of the
deepest nest tree that fits the card; nest trees are explicit opt-in,
and refusals name the tree runner with a complete remedy. Sizing a card
that is not in the machine is priced against the conservative measured
reference profile instead of a per-class discount, and is labelled an
estimate everywhere it appears. **Migration:** a config sized near a
card's ceiling that previously certified as fitting can now be honestly
refused â€” re-run the wizard to get a grid the estimate stands behind.

First-run polish, from the first cross-architecture field run of the
published wheel: with `GPUWM_CASE_DATA_ROOT` unset, the case-data root
now follows the platform â€” the XDG data directory on Linux and macOS
instead of a literal `~/Downloads`, which stays the Windows default
unchanged. `gpuwm fetch-geog` announces the whole bill â€” datasets still
needed, download bytes, unpacked size â€” before the first byte moves.
The first forecast on a machine says up front that it is compiling GPU
kernels for the local card and what that costs (typically 1â€“3 minutes,
cached for every later run), where that time used to pass under a stale
status and read as a hang. `pip show gpuwm` answers `Apache-2.0` â€” the
wheel carries the SPDX license expression; building from source now
needs setuptools â‰¥ 77. Rendered PNGs are named `arwen_*`, and `--pair`
reads both spellings so older render directories still pair.

From the 5070 Ti findings: `gpuwm update` exists â€” print-only, it names
the exact upgrade command for your install and what stays preserved
across upgrades; the generated HRRR chain prints the two experimental
Thompson-aerosol exports before preparation instead of leaving them to
be discovered from a refusal; and the benchmarks grew up â€” an existing
`--outdir` is refused in a sentence rather than a traceback
(`--allow-existing` reuses without clobbering), the GPU name is decoded
text instead of a bytes repr, provenance resolves from the installed
wheel instead of demanding a git checkout, the aerosol-aware scheme is
selectable with its fixed 362 MiB table cost priced into the estimate,
and the cold first step is reported apart from the warmed rate so short
runs stop extrapolating misleadingly.

wrfout frames now carry `OLR` â€” the top-of-atmosphere outgoing longwave
the longwave scheme was already computing â€” with WRF's registry
metadata, present exactly when the attached longwave scheme produces it
and absent with radiation off. It is output-only and rebuilt rather
than restart-carried, so checkpoints on disk stay loadable; a resumed
run reports zeros until its next radiation call â€” a deliberate,
documented divergence from WRF's restart handling.

The LES comparison receipts ship with the repository â€” now true: forty
curated receipts and an index at
[docs/public/receipts/les/](docs/public/receipts/les/README.md) back
every WRF-comparison, realisation-spread and determinism number
[LES.md](docs/public/LES.md) cites, byte-identical to the originals
except for relativized machine paths, with every omission stated and
justified. The CBL receipt's VRAM figure splits into `vram_pool_gib`
(screened) and `vram_device_gib` (environmental), so the documented
dual-run corruption screen stops false-positiving on a shared card. And
the single-card determinism known-limit is retired: the same seed is
bit-identical across three cards â€” two of them different sm_120
silicon â€” and seed-equivalent against sm_89, receipt and claim boundary
in the same directory.

Bookkeeping regenerated on the assembled line: the flush-to-zero claim
census and route registers re-keyed after 1.5.0's line drift, and one
stale kernel frame pin corrected to its measured value â€” re-measured
under two NVRTC majors and offline nvcc before it moved. The NVRTC
widening boundary in the Thompson control tests moves to 13.1, where
exact agreement was actually measured.

## 1.5.0 (2026-08-03)

This release makes the model usable at the scales storm work actually
happens at. Two large-eddy-simulation closures are now selectable â€”
`km_opt = 3` (3-D Smagorinsky) and `km_opt = 2` (1.5-order prognostic
TKE), both per-domain, so a coarse parent running a PBL scheme can carry
a PBL-off LES child in one tree. Both closures were run head-to-head
against an independently built WRF v4.6.1 `em_les` reference on the same
case, same grid, same instrument, across repeated realisations. A nested 250 m LES
configuration is included so the capability is reachable without
hand-assembly, and [LES.md](docs/public/LES.md) states plainly what is
proven and what is not.

Aerosol-aware Thompson microphysics (`mp_physics = 28`) ships with WRF's
own `CCN_ACTIVATE.BIN` table, so a clean install validates and runs it
with no extra download. Its honest label is **implemented-unverified**:
every device kernel is pinned bitwise against an instrumented WRF oracle
column by column, a matched free-running forecast comparison has been
run and published with its limits stated, and a t=0 read-back digest
backs the initial-state numbers â€” but the matched comparison did not
meet the bar to raise the maturity label, and the pages say so; a
third, pre-registered distribution gate across independent runs reached
the same verdict by its own declared rule
([validation/mp28-distribution-gate.md](docs/public/validation/mp28-distribution-gate.md)).
Against
the 22 pinned oracle fixture columns, 17 of 22 clear a flat bitwise
gate and 18 of 22 clear it with the three named allowances applied; the
residuals are itemized, with mechanisms, in the evidence page. Start at
[validation/mp28-column-evidence.md](docs/public/validation/mp28-column-evidence.md).

A third turbulence option, the SASE closure (`bl_pbl_physics = 900`), is
selectable and GPU-resident, verified against the GABLS1 stable
boundary-layer intercomparison. It is **EXPERIMENTAL and not
WRF-verified**, is selected run-wide only (never per nest), and the
configuration reference says exactly where it is and is not admitted.
With it comes `km_opt = 0` as a deliberate research control â€” no
horizontal mixing operator at all â€” behind a written acknowledgement so
it cannot be reached by a mis-set switch.

The domain wizard stops leaving hardware on the table: its internal
search ceiling used to size 64, 96 and 180 GiB cards identically, and
now a larger card buys a larger domain, with an explicit warning in the
one case the search bound rather than memory decides. Sizing below 4 km
with a cumulus scheme active earns a printed advisory (double-counted
convection, and the per-domain switch that fixes it); 4â€“10 km earns a
softer note; neither refuses. HRRR-driven runs got a matching cleanup:
the wizard derives one coverage envelope from the grid itself and clamps
the emitted fetch area inward so every emitted hint passes the same
validator that will judge it, the coverage fitter reserves the
soil-donor search margin instead of discovering it missing later, and
sub-hour forcing windows use one endpoint convention end to end. One
tightening to note: a hand-edited fetch area that names ground outside
the real grid â€” including clamps that previous releases accepted â€” is
now refused; re-emit the area with this release's wizard, which
produces a valid box unaided.

Receipts and diagnostics got more honest. The VRAM receipt now reports a
true high-water mark from a 50 ms watcher, labeled as what it is, with a
separate pool-peak entry â€” not a post-trim boundary value that
understated the peak. Sea-level pressure is terrain-robust: the
below-terrain extrapolation is smoothed with a terrain-keyed filter, so
high-terrain MSLP fields stop ringing. A first run stages the WPS_GEOG
datasets through the installer with byte accounting, and `gpuwm go`
prints what each stage holds and what a failed stage left on disk. An
interrupt anywhere in the `go` chain â€” including the pre-download memory
gate â€” exits with one sentence and code 130.

The release workflow accepts both cut motions again: a re-run of a
partially published cut adopts the assets it already uploaded by content
contract, the post-upload index proof rides out registry lag with
bounded backoff, and the cut refuses bridge binaries that are not
provably built from the release commit.

## 1.4.1 (2026-08-02)

A correction and operations release. No new physics scheme and no kernel
change: every forecast 1.4.0 could produce, 1.4.1 produces identically.

### Corrections

Two statements in 1.4.0's documentation were wrong and are corrected here.
FP32 subnormal flushing on sm_120 was described as a hardware property no
compile flag could change; it follows the compile options actually emitted, and
the same source built without an appended `-ftz=true` keeps full IEEE
subnormals on the same card. The CPU reference binary was described as an Intel
`ifx` build; it is GCC 15.2.0. Neither the countermeasures nor the reference
stream change.

### Fixed

- `gpuwm dual-run` returned nonzero on a byte-identical pair, so the
  determinism screen the documentation describes could not be passed.
- `--products wind10` returned a sea-level pressure analysis on the rust
  engine and a wind map on the matplotlib engine. A standalone
  `10m_wind_speed_and_direction` product was added: speed fill with barbs and
  streamlines, 0-60 kt, unmasked.
- A `pip install` shipped no map assets, so plots drew weather over a blank
  frame. The assets now travel with the renderer, and a renderer that cannot
  find them fails with a message instead of reporting success.
- The domain wizard emitted configurations that `gpuwm check` then rejected.
  It sized against a reserve the verifier does not use and against nameplate
  card capacity rather than usable capacity.
- The VRAM model had no intercept and priced preprocessing on the root domain
  only, so it under-predicted on multi-domain trees.
- An unrecognized `[case_data]` key was dropped silently; it is refused now.
- `gpuwm render --list-products` no longer requires a `wrfout` first.
- Every `wrfout` now records what it was initialized from, so a plot separated
  from its run directory still carries its provenance.
- Restart and preflight refusals exit with a message rather than a traceback,
  and a `run_seconds` off the root-domain step grid is refused at config time.
- Thompson (mp=8) moves closer to WRF v4.6.1; sparse HRRR cloud analyses no
  longer fail a mass-loss gate they should have passed.
- No ArWen or WRF quantity can reach a label reading "wind gust": WRF v4.6.1
  defines no gust. Operational GRIB gust fields keep the name, because that is
  the issuing centre's own field.

### Added

- `gpuwm multi-run PLAN.toml` â€” one independent forecast per physical GPU.
- `gpuwm stream PLAN.toml` â€” sealed hourly HRRR forecast streaming.
- `gpuwm report` â€” a single redacted file describing a failure.

### Running long forecasts on consumer cards

Consumer GPUs do not correct memory errors. On one such card, six 6-hour runs
from byte-identical inputs each ended differently, and two duplicate 1-hour
runs of one configuration wrote byte-different output. Run a long forecast
twice and compare the results byte for byte; a single run on hardware without
ECC is not by itself a reproducible result.

### Known limits

- The VRAM envelope carries no RTX 5090 datapoint.
- `10m_wind_1h_max` read from a `wrfout` store is a lower bound on the true
  maximum, not the maximum.
- Upgrading ArWen does not upgrade a bridge staged by an earlier version.
  Re-stage it with `gpuwm fetch-bridges`.
- Checkpointing is not reachable on the single-domain prepared route, and
  three of the configuration changes `gpuwm run --restart` documents as
  supported do not take effect.

## 1.4.0 (2026-08-01)

This release removes the two things that most often stopped a run that
was going to be fine. Any physics suite the engine implements now runs:
the whitelist of named suites is gone, and verification status is a
label reported on the receipt rather than a gate. And 33 user-facing
refusals became one-line warnings that continue -- a hard refusal is
now reserved for the cases where the run would be garbage.

Preprocessing no longer grows with the length of the run. Ingest used
to hold every forcing time resident at once; it streams now, so its
peak is FLAT in forcing-time count where it used to climb. Measured
against a real 1.3.1 install on identical inputs and the same idle
card: **3.98 -> 4.00 GiB across 2 and 5 forcing times on 1.4.0, where
1.3.1 went 4.80 -> 6.33 GiB.** That is the property that matters on a
small card -- a 24 h window costs the same preprocessing VRAM as a 2 h
one -- and it is what the headline should have said. On the longer
windows the streaming change was developed against (CONUS 12 km
414x330x49, nine 3-hourly GFS times, RTX 5090) the same change is a
14.93 -> 4.56 GiB reduction, but ~3x is a property of a nine-time
window, not of the release: at the 2-12 h windows a first run actually
uses it is 1.2x-1.6x. `gpuwm check` and the `gpuwm domain` fit loop
also price ingest as its own phase -- so a domain that cannot fit is
refused before anything is downloaded rather than after.

A run may now start at a forecast lead instead of only at a cycle's
analysis: `start_time` may be cycle + K hours, the initial condition
comes from f{K}, and the GRIB bridge enforces instantaneous-only
fields for that path.

The HRRR route works end to end from a wheel: the wizard emits the
complete input set for the nested route, per-domain history cadence is
supported, sizing is coverage-aware and refuses an off-grid domain
before the download, transport prefers S3, ocean-edge domains prepare,
and one cell of snow-depth interpolation overshoot is repaired instead
of losing a preparation.

Nests inherit the physics profile's `epssm` instead of a hardcoded
value. That one line was a first-minute blow-up on sub-kilometre nests
over steep terrain.

The externalized Thompson tables stage outside the install, so a wheel
upgrade no longer deletes what you just downloaded, and a missing
table is a named preflight refusal instead of a mid-forecast
traceback. `jsonschema` is a declared dependency, and
certification-capsule emission can no longer cost you a completed
forecast. `gpuwm doctor` now resolves the exact execution paths a run
will use, per data route.

A substantial part of what follows is fixes for issues reported by an
external tester, who ran the published wheel with no git checkout
anywhere near it.

### Performance

- The full-physics step is faster on the same hardware and the same
  inputs, from a profile-first pass that removed per-step scalar
  device-to-host transfers, batched output validation, and reused
  scratch buffers in place. The representative trace, its method and
  its receipts are in [PERF-SURVEY.md](PERF-SURVEY.md). No numerical
  result changed: the pass is measured against a bit-identical base.

### The HRRR route, and the nest that blew up

- **The HRRR hierarchy prints the digest its own chain asks for.** The
  emitted multi-domain chain ends in `--preparation-receipt-sha256
  <printed by the hierarchy>`, and the forecast runner checks that
  digest against the hierarchy's `receipt.json` -- but the hierarchy
  printed only timings and the three WRF file digests, so the one
  placeholder in that chain no stage filled. Walking the printed route
  end to end on a two-domain run is what found it: the chain could not
  be completed without hashing a file by hand. It is printed now, taken
  from the published receipt rather than from the in-memory payload, so
  it is a digest of the bytes the next stage reads. In the same pass the
  single-domain block stopped pointing at FIRST-LIGHT section 3a for its
  forecast stage: 3a is the GFS route, which that same block tells the
  reader not to use, and it never mentions the benchmark runner. It now
  says the arguments come from what the preparation wrapper prints,
  which is where they actually come from.
- **Every nest gets the physics profile's `epssm`.** `gpuwm domain`
  wrote `epssm = 0.1` on every nest while the root took 0.5 from the
  profile -- stripping the vertical-acoustic off-centering exactly where
  nest terrain is steepest. On a wizard 3 km -> 750 m ladder over steep
  terrain, w grew -15 -> -289 -> -976 m/s to non-finite in seven
  acoustic substeps at the child's single steepest cell (34.6 degrees;
  the 3 km parent smooths the same peak to 15.7 and survived). The
  reported nested-forecast blow-up was that one line; the PBL guard was
  the messenger. The value is now written per domain in the emitted
  TOML, so it is visible where it is set and stays overridable there.
- **`gpuwm domain --source hrrr` emits every input the HRRR routes
  read.** It used to emit one of five, and the `namelist.wps` it did
  emit was missing `&share/interval_seconds` -- the one key the
  domain-tree route's first gate requires. The wizard now writes the
  target-domain document, the native `namelist.input` and its stock-WRF
  twin beside the config, all derived from the emitted experiment and
  re-imported through the real importer before they are left on disk;
  binds a route-compatible physics profile when none is named (and
  refuses an incompatible one at emission, naming the switches); and
  prints the actual HRRR chain with every value it knows already bound.
  A multi-domain HRRR emission used to be told to prepare with `rw-wps`,
  which is the GFS front door.
- **An HRRR domain is sized against HRRR's grid, not only your card.**
  HRRR's native grid is finite and the interpolation halo needs real
  source cells outside the target on every side, so a ladder sized
  purely against VRAM could be a legal, well-sized experiment that no
  HRRR fetch can force -- found by running the wizard's own output: a
  3 km root whose halo ran nine rows off the top of the grid, discovered
  by the root preparation after the download. The fit loop now asks the
  source's own window function about every candidate, a point HRRR
  cannot force at any size is refused before anything is written, and a
  domain the grid stopped rather than the card says which bound it hit.
- **One cell of interpolation overshoot no longer loses a preparation.**
  Snow water and snow depth are physically non-negative, so a negative
  mapped value is the horizontal operator overshooting across the snow
  line, not data. A nested HRRR preparation died on one cell of 88 844
  at -4.9 cm beside a 44.5 m maximum. It is repaired at zero, on the
  same terms soil moisture already was; an overshoot beyond what a
  bounded stencil can produce still refuses, now with the count, the
  most negative value and the field maximum in the sentence.
- **A coastal HRRR domain prepares.** Any domain whose 5-cell boundary
  strip was entirely water died in soil mapping with a message naming
  neither the strip nor the domain -- and the four strips are mapped
  independently, so one Pacific-facing edge killed the whole
  preparation. A water target needs no land donor: its soil is the
  target-water fill. The land-window diagnostics are recorded as absent
  with the reason, the finiteness and physical-range checks now run in
  every case, and the refusal that remains (target land with no
  reachable source land) names the grid, the count and the remedy.
- **A frozen soil node no longer discards a prepared hierarchy.** The
  stock-WRF export's frozen-soil refusal is typed as
  `StockWrfExportUnsupported`, and the HRRR hierarchy asks for an
  optional export like every other route, so a state gpuwm integrates
  but the WRF file format cannot represent is recorded as a refused
  export manifest instead of destroying a complete, verified
  preparation.
- **`--transport auto` is paired with the byte mode it runs.** It probed
  NOMADS first; both hosts serve identical bytes, but S3 is several
  times faster for the whole-file default (a measured three-hour window:
  54 minutes over NOMADS, about three over S3). `auto` now prefers S3
  and falls back to NOMADS only when S3 cannot serve the window yet;
  `--wait-for`, which is what NOMADS's head start is for, keeps its
  NOMADS-first polling. When NOMADS is chosen with `--mode full-file`
  the cost is named before the first byte moves, and `gpuwm fetch` says
  who chose the byte mode -- the backbone cannot tell a typed flag from
  a caller's default, and used to call both an "operator override".
- **`--pipeline-workers` is refused at parse time.** `prepare_hrrr_wrf`
  advertised 1..64 while the decoder it spawns accepts 1..13, so an
  impossible request cost a geometry receipt and a static build before
  anything refused it.

### Front doors, packaging and documentation

- `gpuwm fetch-tables` stages outside the install. It used to write its
  two downloads *inside* site-packages, so the 315 MiB a wheel user had
  just paid for was deleted by the next `pip install --upgrade` or venv
  rebuild -- and the nested-GFS forecast that followed died on a bare
  `FileNotFoundError` naming a path in site-packages, after the fetch
  and preprocessing had already been paid for. Staging now lands in
  `~/.gpuwm/tables/thompson`, beside `~/.gpuwm/bridges`, and the staged
  root is completed from the package so it is whole; a complete
  packaged root (every clone, and every wheel staged before this
  change) still answers first and is left alone. The gap itself is
  refused at `--materialize-authorities` -- step 2 of the six the
  documentation prints, before any download -- in one sentence naming
  the table and `gpuwm fetch-tables`, and the domain-tree runner's
  preflight says the same thing rather than raising.
- `gpuwm fetch --author-front-door-manifest` resolves the bridge itself
  when `--bridge` is omitted, through the same `gpuwm.bridges` ladder
  `gpuwm go` has always used. FIRST-LIGHT's stage-by-stage route
  printed `tools/grib1_bridge/target/release/gfs_grib2_bridge`, which
  exists only in a checkout, so a wheel user following the documented
  long form was refused after paying for the fetch. That page now
  documents the resolved default, and its step 5/step 6 pair names the
  `--outdir` rw-wps actually prints (`<output-root>-forecast`) instead
  of a third path neither stage produces.
- `gpuwm adapt --skeleton` names its own gaps, and `--descriptor`
  refuses the unfilled scaffold as a scaffold. The placeholders are
  deliberate, but the reader met them as
  `descriptor.adapt.model_top_pa must be a positive finite number` --
  a type error about one field, with three more waiting one run apart,
  and two (`name`, `target.name`) that passed every validator and would
  have shipped an adapter called `REPLACE_WITH_ADAPTER_NAME`. One
  function now reports them, so the skeleton's list and the authoring
  gate's list cannot drift.
- `rw-wps --namelist-support-report` refuses a missing namelist in one
  sentence. It is step one of `docs/migrating-from-wps.md` and answered
  the commonest possible mistake with a five-frame traceback out of
  `pathlib`, while `gpuwm import-namelist` answered the identical
  condition cleanly; both now go through one function.
- `gpuwm downscale --card` accepts the tiers `gpuwm domain` accepts.
  It hardcoded `16gb|24gb|32gb` and rejected `12gb` -- a tier the
  product advertises and this command already maps -- so two front
  doors quoted different tier lists for one concept.
- The wizard's oversized-footprint advisory measures latitude too, and
  names the flags that matter. A Linux `--card 24gb --ladder 12` sized
  a 144x88 degree domain behind a hemispheric fetch box; the advisory
  saw only longitude, and offered `--card 12gb` rather than saying what
  the printed `--area` is or why shrinking it alone would starve the
  domain it feeds. Still an advisory, still never a refusal.
- DATA.md prices HRRR at what it costs. The documented ~0.4 GB per
  forecast hour was one object in a subset mode that is no longer the
  default; the default pulls the whole `wrfnat` **and** `wrfprs` pair,
  measured 704 MB + 427 MB, so ~1.1 GB an hour and ~21 GB for f00..f18.
  Both modes are now priced separately. `docs/native_source_adapters.md`
  says that its `share/configs/...` paths are distribution-root paths
  and names the checkout's `configs/` equivalents; FIRST-LIGHT's
  custom-ladder examples no longer hide the required `--cycle` and
  `--out` behind an ellipsis -- nor does HARDWARE.md's short version --
  and the page no longer offers `gpuwm domain --explain` as a way to
  list profiles, which is a usage error rather than a listing.

### Physics freedom, memory and initialization

- 33 user-facing refusals are one-line warnings that continue. The
  config loader, preflight, the environment-production path and
  certification-capsule emission now say what they found and carry on,
  because none of those findings was ever evidence that the run would
  be wrong. `jsonschema` is declared in `pyproject.toml` instead of
  assumed, and a missing one costs a capsule rather than a finished
  forecast. The README quickstart is two commands.
- Any physics suite the engine implements runs. The whitelist that
  admitted a fixed list of named suites is gone, and verification
  status -- `readiness`, `warning_only` -- is reported metadata on the
  receipt rather than a gate: an unverified combination is labelled,
  not refused. `export_prepared_wrf` gains an `experiment_config_suite`
  contract, expert-tuple governance is applied uniformly behind
  `--ack`, and the Thompson table checks are keyed on `mp_physics == 8`
  rather than on a suite name. An unacknowledged expert tuple warns and
  runs, with both accepted spellings of the acknowledgement in the line
  and `acknowledged: false` on the receipt; a caller who *names* an
  expert `--physics-profile` asked for that gate and still meets it.
- Preprocessing no longer peaks at roughly double the forecast. The
  initialization loop held every forcing time's horizontally
  interpolated analysis and every time's complete `DomainState`
  resident until the last one existed, purely so the lateral-boundary
  builder could see them all at once; only the first time is used
  downstream. It now streams: `StateBoundaryFrames` takes each state's
  four `spec_bdy_width` perimeter frames as it is built and the state
  is released immediately. On a measured CONUS 12 km case (414x330x49,
  nine 3-hourly GFS times, RTX 5090) the preprocessing peak fell from
  **14.93 GiB to 4.56 GiB**, below the same case's 7.10 GiB forecast
  peak, with the prepared cache's 451 arrays and both wrfinput_d01 and
  wrfbdy_d01 byte-identical to the pre-change run. Applies to the GFS,
  ERA5, mapped-source and config-driven (`gpuwm run`) ingest paths.

  **The 3.3x in that sentence is a property of a nine-forcing-time
  window, not of the release.** The reduction a user sees is whatever
  the streaming saves at *their* window length, and the honest general
  statement is the shape rather than the ratio: 1.3.1's preprocessing
  peak GROWS with the number of forcing times and 1.4.0's does not.
  Measured on one card against a real 1.3.1 install, identical GRIB2
  inputs, identical WPS_GEOG tree, identical config (242x194 @ 12 km +
  480x384 @ 3 km):

  | forcing times | window | 1.3.1 peak | 1.4.0 peak | reduction |
  |---|---|---|---|---|
  | 2 | 2 h | 4.80 GiB | **3.98 GiB** | 1.20x |
  | 5 | 12 h | 6.33 GiB | **4.00 GiB** | 1.58x |

  1.3.1 climbs ~0.51 GiB per extra forcing time; 1.4.0 is flat to
  0.02 GiB. Extrapolating 1.3.1's measured slope, the ratio reaches 3x
  at about 13 forcing times (a ~36 h window). For the 2-12 h windows a
  first run actually uses, expect 1.2x-1.6x -- and expect it not to get
  worse as you lengthen the run, which is the part that is new.
- `gpuwm check` and `gpuwm domain` price **every phase**, not just the
  forecast. The report carries an INGEST line and a one-sentence
  binding-phase verdict, the wizard's fit loop sizes against the
  largest phase, and a config whose forcing product this estimator has
  not modelled (HRRR's native-hybrid-level lane) is reported NOT PRICED
  rather than scored as free. Previously a domain sized to "fit your
  card" could pass `gpuwm check` and then run out of memory during
  preprocessing, a phase nothing had estimated.
- `gpuwm go` runs that check **before the fetch stage**. A binding
  phase that cannot fit the card's free VRAM is refused with its
  number and the re-sizing command, with nothing downloaded; over
  budget but inside free VRAM warns and proceeds. `--no-memory-gate`
  skips it.
- GFS: a run may be initialized from any forecast lead the fetch
  carries, not only from the cycle's f000 analysis. `start_time` may be
  `cycle + K` for any K in the fetched forecast-hour set: the initial
  condition comes from f{K}, the lateral boundaries from f{K+i}, and
  the model clock starts at `start_time`. `gpuwm fetch
  --forecast-start-hour K` plans the window (`--hours` stays its
  length, so `--forecast-start-hour 174 --hours 66` fetches
  f174..f240), the same flag on `--author-front-door-manifest` cuts a
  front-door manifest to that tail of an existing fetch instead of
  re-downloading it, `gpuwm domain --forecast-start-hour K` emits the
  config, and `gpuwm go` carries the lead through its chain. Before
  this, `rw-wps --source gfs` refused any experiment whose start_time
  was not the cycle -- so reaching a window deep in a forecast meant
  integrating to it. Admission warns rather than blocks, and every
  receipt records cycle, lead and start together
  (`gpuwm-gfs-initial-condition-provenance-v1`); a forecast lead is
  never relabelled as an analysis. A run at lead 0 is unchanged, proved
  by a byte-identical prepared cache, wrfinput/wrfbdy, final state and
  history frames against the base commit.
- Docs: added an explicit resolved-scale disclaimer -- the 500 m nest
  and the STP/UH severe suite are tornadic-supercell environment and
  mesocyclone-proxy diagnostics, not resolved tornado dynamics -- to
  README Limits and to VERIFICATION.md section 6 Not-claimed.

### The HRRR route on a pip install

A field report ran the published wheel -- no git checkout anywhere near
it -- through the HRRR to nested-d02 route and hit five failures in a
row that `gpuwm doctor` had just called "no gaps". Every one of them is
a resolution the report never performed.

- `gpuwm fetch --source hrrr` takes whole objects by default
  (`--mode full-file` through the Rust backbone) instead of `.idx`
  record subsets. The old `auto` default resolved to `idx-subset`
  against every healthy host, because its probe rule only takes the
  whole file when the index cannot carry the selection; one measured
  419 MB file cost 560 s that way against 27-35 s taken whole.
  Subsetting is the opt-in bandwidth saver it was always documented as,
  it says in one line what it costs when chosen, and an install without
  the backbone is told before the download rather than after it.
- The HRRR preparation wrapper resolves its decoder through the same
  ladder `gpuwm doctor` reports -- environment override, checkout
  build, `libexec/bridges`, `~/.gpuwm/bridges` -- instead of a cargo
  workspace under `site-packages` that no wheel has. A green doctor
  line and a runnable preparation are now the same claim.
- A run binds its identity from the installed wheel (distribution
  version plus the digests pip wrote into `RECORD`) when it is not a
  git checkout, instead of running `git rev-parse` with the working
  directory in `site-packages` and exiting 128. A venv nested inside an
  unrelated repository is treated as "not a checkout of this tree"
  rather than being bound to that repository's commit.
- The sealed-runtime manifest is validated against its whole schema at
  the first read, by one validator every consumer calls, and the
  refusal names every missing field at once. A minimal hand-authored
  document used to satisfy the identity gate, run a preparation for
  minutes, and die at a third consumer on `contract.platform.backends`.
- The stock-WRF export no longer takes a successful preparation down
  with it. `--skip-stock-wrf-export` never attempts it;
  `--require-stock-wrf-export` makes a refusal fatal; by default a
  refusal is one warning line and the prepared cache and its PASS
  report stand. The converter's own named refusals are unchanged.
- `gpuwm doctor` resolves the paths a run will use: the exact decoder
  per data route, the byte transport `gpuwm fetch` will pick, the
  identity path a receipt will bind, the complete manifest schema, and
  whether the route entry points import from a directory that is not a
  repository. `--source SOURCE` narrows the report to one route; every
  route is in the default estate, because behind a flag these would
  have been invisible to the person this was written for.

### One d01 across the HRRR nested route

The same field report's other half. Seven contract contradictions on the
HRRR to nested-d02 route, six of them the same shape: two parts of this
route answering one question differently, each needing a hand
workaround.

- The single-domain root preparer reads **d01's** column entry of every
  WRF per-domain array, not the last one. `gpuwm domain`'s own emitted
  ladder writes `diff_6th_factor = 0.08, 0.10`; the preparer read 0.10
  -- d02's value, while preparing d01 -- and refused with a sentence
  about the number, never saying which domain it had looked at. A
  refusal on a column that is not uniform now says so and names the fix.
- `num_land_cat` and `fractional_seaice` no longer hit the unmapped-key
  refusal, which they did despite gpuwm's own stock-WRF namelist writer
  emitting both and a shipped hierarchy config carrying both.
  `num_land_cat` is validated against the one land-use identity gpuwm
  builds -- the 21-category MODIS set its static builder stamps and
  every wrfout carries -- and recorded; `fractional_seaice` is reported
  with the branch each initialization route actually runs. Neither is
  silently dropped, and a genuinely unmapped key still refuses by name.
- The hierarchy's run-length ceiling is the sealed root preparation's
  own forcing inventory. It was pinned at 43200 s with a sentence about
  "the native f00..f12 forcing horizon", while its own docstring
  promised not to pin a duration and while the source window, the fetch
  and the forcing-hour check below it all handled f00..f24.
- Per-domain history cadence is supported: a 3 km parent written hourly
  beside a 1 km child written every 15 minutes -- the ladder's whole
  point, and already implemented by the experiment loader, the clock and
  the tree runner -- was refused after preparation as a drift.
- `moist_cq` and `top_lid` have one authority: the shipped physics
  profile. WRF's namelist has no `moist_cq` key and gpuwm's `top_lid`
  default is deliberately not WRF's Registry default, so something must
  decide them; the profiles decided one way and the importer invented
  another, opposite for every WSM6/Kessler/MYNN/RUC/Noah-MP suite, so a
  public root and a public hierarchy built from the *same* namelist
  could not produce matching d01 identities. A test asserts every
  shipped profile agrees with the single answer.
- Prepared-cache identity asks the partition gpuwm already publishes
  twice instead of keeping a third opinion, through one normalization
  shared by the hierarchy's root binding and the tree preflight. Write
  cadences, a trajectory-inert diagnostic toggle, and a `cudt_minutes`
  that is dead at `cu_physics = 0` and spelled differently by the
  importer and the profiles were all compared by exact equality.
- `write_prepared_cache` accepts a read-only mapping. It crashed on the
  `MappingProxyType` hydrometeor record that `restore_prepared_cache`
  deliberately hands back, with "Object of type mappingproxy is not JSON
  serializable" from inside a nested publication, naming no field. A set
  still refuses, having no canonical order.

Every refusal removed here has a negative control beside it: a garbage
namelist key, a USGS category count, an active `cudt_minutes`, a real
trajectory difference and a run longer than the sealed forcing window
all still fail closed.

### Fixed

- `gpuwm doctor` no longer hangs forever probing a present-but-invalid
  executable on Windows. `subprocess.run`'s timeout bounds the wait and
  not `CreateProcess`, so a file with a corrupt image header could make
  the loader raise a modal dialog inside that call which no timeout
  reaches and no hidden-window session dismisses; a release battery
  froze there twice. Every probe in the package -- the bridges, the
  renderer, the fetch backbone -- now reads the ELF/PE header first and
  reports "present but not a native executable" from the bytes, with a
  fail-fast process error mode behind it for the loader dialogs a
  header cannot predict (a missing DLL). The verdict for a genuine
  bridge is unchanged.

### Sending one file when something breaks

`gpuwm report` collects, from a run directory, everything that explains
a failure -- the receipts, the typed failure with its traceback, the
supervisor's worker logs, the resolved config, this install's provenance
through the same resolver a run receipt uses, the Python/CuPy/driver
versions, the card and its memory, and the free space on every volume
involved -- into one plain zip to attach to an issue. Run with no
arguments it reads the directory it is standing in; `--dry-run` prints
the manifest and writes nothing.

It is anonymous by construction. Usernames, home-directory prefixes,
hostnames, IP and MAC addresses, e-mail addresses, credential-shaped
strings and every environment variable outside an allowlist are replaced
by class placeholders in path names, in log text and inside JSON
receipts, and the manifest counts what went by class. Coordinates,
dates, grid shapes, physics choices and SHA-256 digests are kept: a
chosen domain is scientific content, not identity, and a bundle whose
digests were shredded could not be matched to a receipt.

It reads only what ArWen writes. Collection is by allowlist rather
than by sweep, a deny-set refuses dot-files, private-configuration
directories, credential names and key-shaped suffixes before anything
opens or lists them, paths are resolved before they are tested so a
symlink cannot carry a target past the check, and the command refuses
outright in a directory holding nothing this product wrote. Refusals
are counted by class and never named, because a file name can be the
secret. Redaction is the second line of defence, not the first.

It is built for the failure it exists for. The archive is assembled in
memory and written once, relocating to the temporary or home volume if
the first location refuses, so a full disk costs a relocation rather
than the bundle; a missing artifact is a named line in the manifest with
the route that would have written it, never an exception; and when a
volume is nearly full the manifest says so beside the note that an empty
`pipeline producer exited 1:` message is a known erasure rather than
additional information. Model output and input data are never copied --
`wrfout`, restarts and NPZ caches are inventoried by name and size.

## 1.3.1 (2026-07-31)

Four commands take a fresh machine to a rendered forecast: `pip
install gpuwm[gpu,render]`, `gpuwm setup`, `gpuwm domain`, `gpuwm go`.
Physics combination admission follows WRF v4.6.1's own compatibility
verdicts, the run-time parameter ledger is closed out with every
unimplemented parameter refusing by name, polar stereographic and
Mercator projections run end to end, and mixed microphysics across a
nest edge runs under a named transition policy.

### The four-command chain

- Both forecast runners live in the installed package. `python -m
  gpuwm.prepared_single_domain_forecast` and `python -m
  gpuwm.prepared_domain_tree_forecast` work from any directory, with
  console scripts `gpuwm-prepared-forecast` and
  `gpuwm-prepared-tree-forecast`; the `tools/` scripts a checkout
  carries delegate to the same module objects, so no second copy
  exists to drift. Every command the CLI prints is one a pip install
  can execute. In 1.3.0 the chain's last two stages were spelled as
  relative script paths only a git checkout has.
- `gpuwm go <config>` runs the whole single-domain GFS chain as six
  stages in one invocation -- authority, fetch, manifest, prepare,
  forecast, render -- and prints a heartbeat every 20 seconds with
  the stage's elapsed time and, where the stage publishes one, its
  model seconds. Rendering is the sixth stage rather than a command
  to paste; a missing `render` extra is named instead of skipped.
- A refused stage speaks in sentences. A reused output directory is
  refused up front with the directory named, why it is create-only,
  and both ways out; in 1.3.0 this surfaced as a raw
  `FileExistsError` traceback from a flag the caller never typed.
- `gpuwm go` drives one domain from GFS. A multi-domain tree is
  refused up front and the refusal names
  `python -m gpuwm.prepared_domain_tree_forecast`.

### The wizard composes with `go`

- `gpuwm domain` at a bare prompt asks its questions and emits a
  config `gpuwm go` accepts: one 12 km domain on the wrf-matched-run
  `morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1` profile, with both
  supplied defaults stated in the session. This changes the
  bare-session emission -- in 1.3.0 it was a four-domain ladder on
  the product default suite, a shape the single-domain chain refuses
  twice over. Pass `--ladder` for nests and `--physics-profile` for
  another suite; a session with any explicit flag keeps the flag
  path's own defaults.
- The closing next-steps block branches on source: a GFS emission
  ends at `gpuwm go`, an ERA5 emission keeps `gpuwm run`, and HRRR
  and multi-domain trees get their route named. In 1.3.0 the block
  ended at `gpuwm run` for every source, and `gpuwm run` refuses GFS
  configs by design.
- A domain sized to fill the card prints an advisory when its fetch
  box exceeds 90 degrees of longitude, naming `--vram-gib` for a
  smaller first run. Sizing itself is unchanged, and continental
  domains under the threshold pass without comment.

### WRF v4.6.1 combination admission

The combination authority is a 2,400-cell transcription of WRF
v4.6.1's own PBL, surface-layer, and land-surface admission: 1,080
cells legal, 360 legal with WRF's silent reconfiguration named, 600
fatal, 360 not expressible in WRF. Three pairings 1.3.0 refused are
admitted because WRF admits them -- PBL off with the MYNN surface
layer, and MYNN PBL with either MM5 surface layer, each across the
four routed land models -- and refusals cite the matrix cell that
produced them. An edge policy is not a loophole: a PBL/surface-layer
pairing the matrix calls fatal refuses whatever the nest declares.

### Run-time parameters

The declared WRF namelist parameter set stands at 158: 104
implemented, and each of the 54 others refuses by name with the
blocking dependency cited. Five parameters whose runtime branches
already existed are wired end to end: `isfflx`, `o3input`,
`use_mp_re`, `seaice_albedo_default` (replacing a live-path
literal), and `rdmaxalb`.

### Projections

Polar stereographic and Mercator domains run end to end -- wizard
emission, WPS namelist, static-field preparation, map factors, and
earth-relative wind rotation -- verified against WPS geogrid
references for both families. A bare interactive session at high
latitude emits `map_proj = "polar"` with a matching namelist.

### Nest edges

All 20 ordered pairs of ported microphysics schemes run across a
parent-child edge under an explicit `nest_microphysics_transition`
policy, and a mixed edge without one refuses naming the setting.
Each policy's mass and number handling is a registry row with its
WRF citation, not a code path a config reaches silently.

### Byte-stable checkouts

`.gitattributes` declares `* -text`: a clone materializes the
committed bytes on every platform. That immediately exposed one
shipped defect, fixed here -- the packaged GFS Vtable's hash
contract named the file's CRLF form, which only a Windows checkout
produced. The contract names the committed bytes, and
`tests/test_line_ending_stability.py` holds every hashed-contract
file to its blob.

### Fixed

- `gpuwm go`'s render stage handed the render CLI the wrfout
  directory rather than its frames, failing after a successful
  forecast. Found driving this release's own acceptance chain; the
  frames are enumerated, and `--dry-run` still prints the glob.
- `--materialize-authorities` refused a reused output directory with
  a raw traceback; a missing base input did the same. Both are
  sentences.
- The packaged GFS Vtable hash pin, above.

The four certified profiles are byte-identical to 1.3.0: their
pinned two-step trajectory hashes did not move, and the registry
regenerates to the tracked bytes.
## 1.3.0 (2026-07-31)

Ten couplings that in 1.2.x produced silently non-WRF forcing are
implemented from WRF v4.6.1 source transcription, the MYNN surface
layer pairs with the RUC and Noah-MP land models, HRRR analyses
initialize NSSL-2 with their hydrometeors retained, and a
hand-authored physics tuple outside the registry's ratified set -- one
that is otherwise complete and executable -- can be run with one short
acknowledgement. The command surface answers first: `gpuwm doctor` and
the domain wizard print one line per finding with the full prose one
flag away, and `gpuwm setup` stages a fresh install in one command.

### Changed forecasts

Configurations using MYNN, RUC, Noah-MP, or NSSL-2 produce different
trajectories than 1.2.x where their forcing or initialization follows
WRF in 1.3.0:

- MYNN receives the real snow mixing ratio when the microphysics
  carries one (WRF `FLAG_QS`); its boundary-layer cloud fields
  (`QC_BL`/`QI_BL`/`CLDFRA_BL`) are consumed by all three radiation
  implementations with WRF's `icloud_bl=1` merge semantics.
- RUC runs WRF-ARW's per-species precipitation partition (rain, snow,
  graupel reach the soil model separately), the lake bypass, the
  fractional sea-ice deblend/reblend, and radiation-time `GSW`.
- Noah-MP receives all six WRF precipitation rates and the radiation
  driver's carried `COSZEN` instead of a per-call recomputation;
  glacier columns are refused before the forecast starts rather than
  mid-run.
- NSSL-2 initialized from a native HRRR analysis starts from the five
  retained analyzed species whichever radiation implementation is
  selected, RTE included; in 1.2.x those fields started from zero.
- NSSL-2 with legacy RRTMG activates ice and snow cloud optics, as WRF
  does; the 1.2.x exclusion was not WRF-faithful.
- Coastal native-HRRR domains where the analysis landmask disagrees
  with RUC's soil-category requirements are reconciled from the
  evidence in the analysis, or refused before the run starts with the
  points named; in 1.2.x this route crashed inside the RUC soil model.

The four certified profiles (WSM6+Dudhia, Thompson+Dudhia,
Morrison+RTE, NSSL+RTE on Noah/YSU/MM5) are byte-identical to 1.2.x:
their two-step trajectory hashes are pinned in the suite and did not
move through any of these changes.

### MYNN surface pairings

In 1.3.0, `sf_sfclay_physics = 5` pairs with `sf_surface_physics = 3`
(RUC) and `4` (Noah-MP), following the write-back ownership sequence
of WRF's surface driver: which component owns the fluxes, exchange
coefficients, and 2-m diagnostics at each point in the step is
transcribed from `phys/module_surface_driver.F`, tested against
independent source transcription, and proven with short GPU
integrations of both pairings. The WRF namelist importer admits a
complete MYNN 5/5 stack (`bl_pbl_physics = 5` with
`sf_sfclay_physics = 5`) paired with any ported microphysics scheme,
where 1.2.x admitted it only with WSM6. The MYNN half-suite refusals
(MYNN surface without MYNN PBL, and the reverse) remain.

### HRRR analyzed hydrometeors

In 1.3.0, native HRRR preparation retains analyzed QC/QR/QI/QS/QG for
NSSL-2 (in 1.2.x only WSM6/Thompson/Morrison), retains QC/QR for
Kessler under WRF real.exe's Registry policy with the frozen species
accounted in a receipt, and refuses microphysics-off with a statement
of what cannot be represented. The preparation receipt binds decoded
source to initialized state per species -- byte hashes, nonzero masks,
extrema -- and states its own evidentiary strength: a species with no
source mass is reported VACUOUS, not proven. An independent check
exists outside this repository: WRF's own real.exe, run on the same
source bytes, initializes the five species with nonzero-cell counts
within 0.06% of this route's.

### Expert acknowledgement for unratified tuples

A hand-authored domain tree whose resolved physics tuple falls outside
the registry's declared reachability -- but is otherwise complete and
executable -- refuses with the exact line to add, instead of silently
running or silently failing at first call. Acknowledging is one
repeatable flag, `--ack expert-tuple-v1`, or one configuration line,
`acknowledgements = ["expert-tuple-v1"]` under `[experiment]`. The
receipt records which IDs were used and how they were supplied. The
acknowledgement is a provenance gate, not a bypass: component
availability, half-suite coupling, soil and land-model compatibility,
geometry, vertical bounds, and runtime-budget refusals all still
apply, and an acknowledged tuple that fails one of them refuses with
that check's own message. The stock-WRF export is layered the same
way: a physics tuple outside the exporter's stock coverage refuses
the export by name while the forecast itself prepares, and
`--no-stock-wrf-export` skips the export with a NOT_REQUESTED
receipt. Registry-ratified tuples run without acknowledgement, and
two joined the ratified set: Noah-MP on GFS as an expert template
(the runner already advertised it), and NSSL-2 with legacy RRTMG as a
fixed profile at wrf-matched-run-candidate maturity, buildable end to end
on both the GFS and native HRRR routes -- the runner builds its
configuration from the profile's complete declared switch inventory,
and a profile switch without a declared forwarding home is a named
refusal.

### Refusals moved to where they can be read

Scheme vertical limits (MYNN needs nz >= 5, Kain-Fritsch 8..128, WSM6
2..80, the radiation implementations' layer caps, and the rest) are
checked at configuration validation and preparation, naming the
component and both bounds -- in 1.2.x a first-timestep abort.
Microphysics-off plans on real-data routes are decided by the plan
validator, which states what a moist-but-scheme-less state holds,
instead of being deferred to source initialization.

### A command surface that answers first

`gpuwm doctor` prints one line per finding -- status, subject, the one
command that closes it -- and folds repeats that share a remedy; the
full evidence and pasteable remedy blocks are unchanged under
`--explain` (byte-identical to 1.2.x's output, held by a golden
fixture). The domain wizard ends with a numbered three-command block
and nothing after it. `gpuwm setup` runs `fetch-bridges` and
`fetch-tables` with one status line each; geography stays opt-in
(`--with-geog`, size printed first). `pip install gpuwm[all]` installs
the GPU and render extras together. Scripts that parsed doctor's
default text should use `--json`, which is unchanged and complete. The
HRRR decoder's air-gapped build invokes Cargo from the crate-local
directory so the vendored-source configuration is discovered; building
the bridge without network access works from a bare checkout, where in
1.2.x it could fail to find the vendored crates.

### Fixed

- 1.2.3 cannot complete a prepared GFS single-domain forecast: its
  fetch writes front-door manifests carrying pressure-ladder
  provenance (`pressure_levels_hpa`, `top_pressure_pa`) that its own
  runner refuses as an identity mismatch. The 1.3.0 runner compares
  manifest identity strictly on model, product, and cycle; validates
  the declared ladder through the fetch lane's own validator against
  the experiment's model top; and records
  `input.source_manifest_identity` in the run report. An unknown
  `source` field is still a refusal, never a silently dropped key.

### Evidence and its edges

Every reopened coupling passed preparation, a real GPU integration,
and its receipts on GFS and/or native HRRR data, and matched WRF
v4.6.1 reference runs are banked for trajectory comparison. The
reference runs' retained forcing bytes match the ArWen-side inputs by
hash, with two recorded exceptions: the WRF-side HRRR preparation
added the MSLMA field, fetched by byte range from the same archive
objects and hash-recorded; and the WPS static-geography trees on the
two preparation hosts are not identical. Four edges are stated rather
than implied:

- The Dudhia-shortwave radiation selector (`ra_lw_physics = 0`,
  `ra_sw_physics = 1`) has no exact WRF counterpart -- WRF's longwave
  selector has no "off" -- so its reference runs bracket the
  configuration between RRTM-longwave and radiation-off.
- Trajectory comparison against WRF is incomplete, so no combination
  is described as WRF-equivalent in this release. Maturity labels
  state what each combination has passed.
- HRRR with Kessler: the preparation retention policy above exists,
  but no registry template or profile admits the pairing in 1.3.0;
  its front-door admission is deferred.
- Checkpoint/restart content sealing (tracked internally as K-02)
  remains open and out of scope for 1.3.0.

## 1.2.3 (2026-07-30)

The 1.2.2 workflow stopped at its test gate on two defects the ubuntu
runner exposed: a doctor test that asserted the complete-tree Thompson
table state on a checkout where the externalized 243 MiB asset is
legitimately absent, and a remedy-contract test that flipped the shell
flag without re-deriving the build hints frozen from it at import, so
POSIX remedies were judged by PowerShell rules. Both tests now assert
each platform's own truth, negative-controlled, run against the public
byte-mirror under Python 3.11 and 3.13 on both shell arms. No runtime
code change.

## 1.2.2 (2026-07-30)

The 1.2.1 release stopped in its workflow before any wheel reached
PyPI: the release tooling wrote pyproject.toml with a UTF-8 byte-order
mark, which pip's TOML parser refuses at line 1. Rewritten without the
mark; the release tooling now writes both files byte-order-mark-free.
No code change.

## 1.2.1 (2026-07-30)

The 1.2.0 release workflow stopped at its test gate before any wheel
reached PyPI: `gpuwm/core/preflight.py` carried an f-string expression
broken across a line break, which Python 3.12 accepts and the supported
3.11 floor rejects. 1.2.1 is 1.2.0 with that statement rewritten in
3.11-compatible form and the whole tree compile-checked under 3.11.
No other change.

## 1.2.0 (2026-07-30)

`pip install gpuwm` followed by `gpuwm fetch-bridges` puts the compiled
GRIB decoders, the CPU preprocessing library, the fetch backbone and the
batch renderer onto a machine with no clone and no Rust toolchain.
History files carry the variable names, NetCDF types, axes, precipitation
accumulators and physics-selector globals a WRF reader expects, which
changes what a v1.1.x file looked like. `gpuwm fetch --source gfs`
accepts the model top the case asks for instead of stopping at 100 hPa.
The fetch and cache state machines take a single-writer lock, publish
atomically, verify cached bytes where they are used, and pace NOMADS from
the Python transport as well as the Rust one. Two documentation pages
state where the determinism claim and the `gpuwm adapt` trust boundary
stop.

### Breaking: what a reader of v1.1.x output must change

- **43 Noah-MP fields are written under WRF's external names.** The
  history writer upper-cased the internal symbol, so `tvxy` went out as
  `TVXY`. WRF's external name is the Registry `dname` column upper-cased
  (`tools/gen_wrf_io.c:331-334`), which makes those fields `TV`, `TG`,
  `ISNOW`, `TSNO`, `ZSNSO`, `SNICE`, `SNLIQ`, `PGS`, `T2V`, `Q2B`,
  `RUNSF` and so on. Two fields keep the symbol spelling, `qsnowxy` and
  `qrainxy`, because that is what their `dname` is. Anything keyed to the
  old `*XY` spellings reads a v1.2.0 file and finds nothing; anything
  keyed to WRF's names reads it and finds the field.
- **`EL_PBL`, `EXCH_H` and `EXCH_M` are written on `bottom_top_stag`,
  one level taller.** The Registry declares them `Z`-staggered, and WRF
  writes them at `nz+1` with the top interface left at the Registry cold
  value of zero, which is what its own PBL driver leaves there. A script
  that hard-coded `nz` levels for these three MYNN fields reads `nz+1`.
- **`--force-refetch` moves every regular file in `--out` aside.** The
  help text has always described a whole-directory replacement; the
  implementation quarantined per forecast hour, so a shorter forced
  window left earlier hours canonical and unlisted, and an interrupt
  mid-sweep left a manifest claiming bytes that had been replaced. The
  sweep is now receipts first (`fetch-manifest.json`, `SHA256SUMS`, the
  series) and payloads second, over every regular file in the directory,
  including index files, selector files, stale parts and files the
  operator put there. Nothing is deleted -- everything is renamed aside,
  files an earlier quarantine already set aside are left alone, and
  subdirectories are untouched. A directory that was half a fetch and
  half someone's notes comes back with the notes renamed.
- **Four integer fields are written as `NC_INT` with `FieldType 106`.**
  `KTOP_PLUME` and `KPBL` (MYNN), `ISNOW` and `PGS` (Noah-MP) are
  `state integer` rows in the Registry and are allocated as 32-bit
  integers in the model; they had been shipping as float32 with
  `FieldType 104`. Values were already integral, so the change is to type
  and metadata, not to content -- but a reader that assumed float32 for
  every variable except `ITIMESTEP` sees a different dtype.
- **Projection globals are single precision.** `DX`, `DY`, `TRUELAT1`,
  `TRUELAT2`, `STAND_LON`, `CEN_LAT`, `CEN_LON`, `MOAD_CEN_LAT`,
  `POLE_LAT` and `POLE_LON` are written `NC_FLOAT`, which is what stock
  WRF writes; they had been `NC_DOUBLE`. Both consumers tolerate either,
  but a byte-level comparison against a v1.1.x file differs.
- **A fractional history or restart cadence is refused at config
  admission.** `history_interval_s` and `restart_interval_s` must be a
  whole number of seconds. A quarter-second cadence on a quarter-second
  step divides evenly into steps and was accepted, and then wrote three
  distinct model instants onto one file name, because the filename and
  the `Times` string carry whole seconds. Sub-second history remains
  unsupported; it is refused rather than silently collapsed. The writers
  refuse a `valid_time` with a nonzero microsecond and a `Times` value
  longer than 19 characters for the same reason.

### Prebuilt Rust artifacts arrive as a download

- **Added: `gpuwm fetch-bridges`.** A pip install has never been able to
  read a GRIB file. The wheel ships no compiled Rust, so the five
  decoders, the CPU preprocessing library, the `rw_fetch` backbone and
  the `rw_wrfbatch` renderer had exactly one route onto a machine: clone
  the repository, install a Rust toolchain, and run `cargo build` in two
  workspaces. The artifacts are published as one release bundle per
  platform, and this command downloads the one matching this OS and
  architecture, verifies every artifact against the size + SHA-256 pins
  packaged in the wheel, and stages it into `~/.gpuwm/bridges` -- the
  directory the resolver already searched. Bundles are published for
  Windows x86-64 and Linux x86-64; the platform check asks what this box
  can execute and nothing else, and anywhere without a bundle keeps the
  build-from-source route, which works everywhere.
- **Every byte is checked three ways before it is installed**: the exact
  size, the SHA-256 pin, and -- for the decoders that declare one -- the
  contract marker `gpuwm doctor` looks for in an already-built binary.
  A staged file that fails any of the three is deleted rather than
  installed, and members are read out of the archive by their exact
  pinned filename, so nothing can be written anywhere the pins did not
  name. An interrupted download resumes against the partial, or restarts
  when the server ignores the range rather than appending to bytes it did
  not extend. `--from DIR` stages the same bundle -- or the loose
  artifacts -- from a local directory, offline, under identical checks.
- **A stale artifact is replaced, not refused.** This is the one place
  the contract differs from `gpuwm fetch-tables`, deliberately: a physics
  table that does not match its pin is the operator's file and is never
  overwritten, while a bridge executable that does not match is a build
  from a different release. Leaving that one in place is exactly the skew that
  made 1.1.0 preparations fail with a message blaming a file gpuwm had
  just written correctly. The replacement still happens only after the
  new bytes pass all three checks.
- **`gpuwm doctor` offers it first, and only when it is true.** Every
  Rust-artifact remedy on a wheel install leads with
  `gpuwm fetch-bridges` and keeps the clone-and-build block beneath it,
  commented out -- printed because it is the only route on a platform
  with no bundle, commented because a reader who pastes the whole report
  must not also compile what the line above already staged. The offer
  appears only when the pins this wheel carries name a bundle for this
  platform, so the report never advertises a command that would refuse.
- **Pins are generated during the release cut**, by
  `tools/build_bridge_bundle.py`, from the exact bytes the release
  uploads, before the wheel is built -- the bundles are compiled from the
  commit being released, so they cannot be pinned any earlier. The
  release workflow builds both platforms with
  `cargo build --release --locked`, uploads the bundles and their
  manifest to the release *before* publishing the wheel that points at
  them, and re-reads its own pins as the last step before PyPI.
  `RELEASE_CHECKLIST.md` carries the same sequence for a cut driven by
  hand. A tree that has not been through a cut declares no platform,
  which is what an unpinned document says instead of carrying a hash
  nobody computed.

### Scheme output a WRF reader can read

- **Every emitted scheme field has a record**, in
  `gpuwm/io/wrf_output_schema.py`: 85 fields across MYNN, Noah-MP and
  RUC, each carrying WRF's external name, its NetCDF type and
  `FieldType`, its stagger, and its description and units transcribed
  from the pinned v4.6.1 Registry with the Registry line cited. Where
  WRF's own strings are visibly wrong they are transcribed as they are,
  because a gpuwm file that disagreed with a WRF file about the same
  field is the failure this table exists to prevent. The writer refuses
  to publish a field whose Registry stagger contradicts the axis the
  dimension table gave it, refuses a float payload for a field WRF
  declares integer, and refuses a scheme field that has no record rather
  than shipping it without metadata.
- **The six precipitation accumulators WRF always writes are always
  written.** `RAINC`, `RAINSH`, `RAINNC`, `SNOWNC`, `GRAUPELNC` and
  `HAILNC` are core per-domain history fields in `Registry.EM_COMMON`
  with no package gate, so WRF writes all six whatever the physics
  selection is, filling them with zeros when nothing produced the
  quantity. gpuwm emitted each one only when the scheme that fills it was
  routed, so `RAINC` was absent from every `cu_physics=0` run, `HAILNC`
  from everything but NSSL2, and `RAINSH` from every file it had ever
  written. Every precipitation recipe in wrf-python and wrf-rust reads
  `RAINC + RAINNC` unconditionally, which is why a wrfout that omits
  `RAINC` fails an entire product family. The six are emitted always,
  zero-filled when the producer is absent. `RAINSH` is always zero and
  the code says why: gpuwm implements no shallow cumulus, and zero is
  what WRF writes for `shcu_physics=0`.
- **Eleven physics-selector globals are stamped into every history
  file.** `MP_PHYSICS`, `RA_LW_PHYSICS`, `RA_SW_PHYSICS`,
  `SF_SFCLAY_PHYSICS`, `SF_URBAN_PHYSICS`, `SF_SURFACE_PHYSICS`,
  `SF_SURFACE_MOSAIC`, `SF_OCEAN_PHYSICS`, `BL_PBL_PHYSICS`,
  `CU_PHYSICS` and `SHCU_PHYSICS`, as `NC_INT`, every one of them present
  in stock WRF output and cited to its Registry line. Four are constants
  because gpuwm has no such option to select, which makes WRF's "off"
  value the true one -- and it is what makes the accumulator zeros
  legible, since `SHCU_PHYSICS=0` beside `RAINSH` at zero answers a
  question neither answers alone. The radiation pair is resolved through
  `gpuwm.config.radiation_scheme_ids` rather than copied from the config,
  because gpuwm's `-1/-1` is a legacy sentinel meaning "use the aggregate
  spelling" and is not a WRF scheme id. The globals are written only when
  the writer was given a resolved run configuration, so an idealized
  caller receives none.
- **Identity in the files themselves.** wrfout carries a `GPUWM_VERSION`
  global; the restart header records the producing distribution, its
  version, and the restart format version. `TITLE` remains the caller's
  configured output title, which is why it never identified the producer.
  This is producer identity only: nothing in this release hashes or
  otherwise binds checkpoint or history *contents*.
- **A restart clock must be a real, finite, non-negative number.**
  Elapsed seconds are checked on write and on restore for every format --
  refusing NaN, both infinities, negatives, booleans and numeric strings
  -- and the header is written with `allow_nan=False`, so it cannot
  express the value at all. The idealized `Times` helper rolls over the
  calendar instead of emitting `0001-01-32` past the end of a month.
- **A checkpoint is fsynced before it becomes visible.** The standalone
  restart temporary and the feedback receipt are flushed and fsynced
  before the atomic rename, and the receipt stages under a unique name.
  Funnelling every publisher through one durability helper is recorded
  and not done.

### The GFS route follows the case's model top

- **`gpuwm fetch --source gfs --p-top-pa PA`** names the model top the
  fetched atmosphere must reach. The certified 21-level ladder is
  extended upward along whatever the live index publishes until a level
  sits at or above the requested top, which is the condition
  `gpuwm.vertical_contract` enforces at ingest. Requesting 5000 Pa adds
  the 70 and 50 hPa levels and nothing else. `--all-levels` takes the
  whole published ladder. An unsatisfiable request is refused by name,
  stating the deepest top the source offers.
- **The route stopped at 10000 Pa, with no flag and no receipt.** Every
  ArWen GFS run was capped at a 100 hPa model top by a hardcoded level
  list, the fetch manifest never recorded which ladder was taken, and the
  user met the consequence three steps later at ingest ("source
  atmosphere stops at 10000 Pa but requested p_top is 5000 Pa") with no
  mention of the fetch that chose it. The ladder and the source top are
  recorded in the fetch receipt and in the front-door manifest, the
  record-count bar is derived from the request rather than fixed at 124,
  the decoder derives its ladder from the source and reports what it
  decoded, and the vertical contract reads the declared ladder instead of
  a constant. A decode whose ladder the input manifest does not declare
  is refused.
- **Extending, never replacing.** Every level a certified run already
  used is present at every requested top, which is what lets the bridge
  and the front door check "is this the certified ladder, extended
  upward?" rather than trusting a number. Whole GRIB objects remain the
  default transfer shape; level subsetting stays an opt-in bandwidth
  saver and is no longer a ceiling on the model top.

### Fetch and cache state machines

- **One writer per output directory.** `gpuwm fetch` takes an
  OS-enforced lock -- a Windows byte-range lock or a POSIX `flock` -- on
  a file kept outside the output tree, keyed on the resolved target path.
  The CLI holds it across the prior-request guard and the transfer it
  authorises, and `fetch_gfs`, `fetch_hrrr`, `fetch_geog` and the table
  stager take it themselves, so a direct library caller gets the same
  contract. A second process queues and then refuses, naming the holder's
  pid. The lock is an OS lock because the kernel releases it when the
  holder dies: a crashed fetch must not leave a directory permanently
  unfetchable. `GPUWM_FETCH_LOCK_TIMEOUT_S` bounds the wait, default
  600 s, `0` to fail fast; `GPUWM_FETCH_LOCK_ROOT` moves the lock root.
- **Receipts describe only what finished.** An HRRR wait-timeout no
  longer lists a half-fetched hour in `files` and `SHA256SUMS` while
  `forecast_hours` omits it; the partial product stays on disk unclaimed
  and is re-verified under the ordinary bars on the next run. HRRR
  publishes a manifest after every completed hour, as GFS already did, so
  a kill after hour zero leaves a usable receipt. Assembly happens inside
  the unique staging directory, so a canonical `.part` is never created,
  and a legacy one left by an older release is swept by force.
- **Nothing is published under a name another writer could be using.**
  `atomic_write_text`/`atomic_write_bytes` stage under a name unique per
  process and per call, fsync, then rename, and fsync the containing
  directory where the platform has one -- Windows exposes no directory
  handle through the standard library, so there the guarantee is that the
  rename is atomic, not that it is durable across power loss. Quarantine
  proves `<name>.rejected-<stamp>` free before renaming, because
  nanosecond stamps collide inside a tick and the rename destroyed the
  older evidence. Table staging names are unique per writer and the
  stage-verify-install sequence runs under the table-root lock.
- **The raw download cache verifies at use.** Entries carry a sidecar
  recording the exact key, the byte count and a checksum; a read
  re-checks all three and renames a failing entry aside so the next
  request refetches it. Payloads land by atomic rename from a per-call
  staging name. A range response is validated before it is adopted: 206,
  a matching `Content-Range`, and the exact byte count asked for. A
  failed quarantine leaves the file in place and reports, rather than
  deleting the evidence.
- **The NOMADS governor fails closed, and paces the Python transport
  too.** A governor state file that exists but does not parse is read as
  "a request just happened" rather than as an empty state, and a request
  whose shared record fails to land still waits out the gap locally --
  both strictly stricter than before, in Rust and in Python. The GFS CGI
  transport, the HRRR index and range transports and the availability
  probes `--wait-for` polls all route through a Python governor speaking
  the same protocol over the same state files as the Rust client, so a
  Rust fetch and a Python fetch on one node pace each other. The NOMADS
  range pool narrows to one worker, since the governor serialises those
  requests anyway.
- **Geography reuse compares the tile corpus, not just the index.** File
  count and total bytes are checked against the extraction receipt, which
  catches missing, truncated, extra and added tiles; an install with no
  receipt keeps the index-only bar and says so. A same-size mutation of a
  tile's contents still passes, and closing that needs a content digest
  of a multi-gigabyte tree, which is recorded as a deliberate
  verification mode rather than something to run on every command that
  opens WPS_GEOG. Each archive's provenance entry is published as it
  lands, before the verified archive is removed, and publication re-reads
  and merges the canonical manifest under the geography-root lock instead
  of overwriting it. A resume is bound by sidecar to the first response's
  URL, ETag and Last-Modified, is sent with `If-Range`, and refuses a 206
  that does not start at the requested offset.

### Two pages that say where a claim stops

- **`docs/public/DETERMINISM.md`.** Consumer cards have no ECC, and
  running the forecast twice and comparing bytes is what stands in for
  it. The page states that as a transient-fault screen inside a fixed
  numerical environment and not as an ECC replacement, because equality
  cannot detect a fault that is identical in both runs. It names the
  seven undetected fault classes, the pin set that "fixed environment"
  actually means, the three mechanisms that make the pin set necessary
  (library-owned reduction order, FMA contraction and CUDA math
  functions, FP32 subnormal flushing), what each compared surface covers
  and omits, and the six known improvements that are recorded and not
  shipped -- the largest being that no fail-closed comparator command
  exists in this release. Linked from the README, `HARDWARE.md`, and
  `VERIFICATION.md`.
- **`docs/adapt-validation-contract.md`.** `gpuwm adapt` proves the
  emitted files implement your descriptor and that your GRIB files
  satisfy it. It does not prove the descriptor is a correct physical
  reading of them. The page gives every input dimension in two columns
  -- validated for you, trusted from your declaration -- and a
  self-check the reader can run for each trusted row. Wired into the
  adapt parser's description and epilog and into both of the command's
  completion messages, so it is found at the point of use.

### Input and checkpoint identity

- **`--directory-input-hash content`** (also `GPUWM_DIRECTORY_INPUT_HASH`)
  binds a declared directory input -- in practice the static geography
  tree -- by each file's SHA-256 instead of its mtime. The default stays
  `inventory`, which is cheap enough to run before every launch on a
  multi-GB tree but has two known modes that matter to a dual-run
  comparison: a byte-identical copy staged separately compares
  different, and a change that preserves path, size, and mtime compares
  equal. Every recorded hash carries the algorithm that produced it, and
  the `inventory` record layout is unchanged, so digests from earlier
  releases still compare equal.
- **Checkpoint discovery no longer ties.** Sets are ordered by valid
  time, then nanosecond mtime, then set id. Two sets at one model
  instant with tied second-resolution mtimes previously fell back on
  filesystem discovery order, which made the choice of resume point a
  property of the filesystem rather than of the run.

### WRF-Runner interoperability

Verified against namelist pairs generated by WRF-Runner (the
collaborator's workflow tool, branch New-PC-Updates) running its own
code, and against its plotting pipeline and viewer consuming gpuwm
history files unchanged.

- **`gpuwm import-namelist` no longer leaks a traceback on an unported
  selector.** A runner-generated pair carrying `ra_lw_physics=1` (the
  unported WRF RRTM longwave) crashed the CLI with a stack trace where
  every neighbouring refusal printed one actionable line; the
  `NotImplementedError` refusals from `validate_run_config` land on the
  uniform CLI refusal boundary -- message on stderr, exit 2, no partial
  output file.
- **The namelist support report classifies what WRF-Runner namelists
  actually carry, and gates two things it used to wave through.**
  `io_form_auxinput2`, `override_restart_timers`, `iofields_filename`
  and `ignore_iofields_warning` classify as runtime-only instead of
  eight lines of `UNCLASSIFIED_NAMELIST_SETTING` noise;
  `sf_surface_mosaic`/`usemonalb`/`rdlai2d` are state-relevant and
  value-gated to their WRF defaults with the exact selector named.
  Two new fail-closed codes: `NEST_INPUT_STREAM_UNSUPPORTED`
  (`fine_input_stream` nonzero -- WRF's delayed-nest-start pattern needs
  a per-nest input file RW-WPS does not produce) and
  `FDDA_INPUT_NOT_PRODUCED` (`grid_fdda` active -- `wrffdda_d0N` is a
  real.exe product, and the report previously classified the request
  runtime-only while blessing an export that cannot feed it).

### Release plumbing

- **The publish workflow runs tests.** The release path had no test gate
  -- a cut built a wheel and shipped it. It runs the
  packaging-and-contract suite first (what ships in the wheel, what
  doctor promises, what both fetch commands verify before installing),
  and runs it on pull requests too, because a gate whose first execution
  is a release cut is a gate that ambushes the cut. It is not the whole
  suite: the model's own tests need a CUDA GPU and staged case data, and
  five test modules import CuPy at module scope, so collecting everything
  on a GPU-less runner fails before a single test runs.
- **The job that writes release assets is not the job that publishes to
  PyPI.** Uploading the bridge bundles needs `contents: write`, and
  Trusted Publishing needs `id-token: write`; they are separate jobs, so
  neither credential is held by a job that has the other. The ordering is
  unchanged and is now a property of the job graph: the assets job
  uploads the bundles and writes the pins computed from those exact
  bytes, and only then does the publish job build the wheel around them
  and push it.
- **The RW-WPS standalone project stops reaching the forecast side.**
  `gpuwm/resume.py` is no longer staged into it -- nothing that wheel
  ships imports it, no entry point exposes it, and its own lookups are a
  module that wheel forbids and one it does not stage. `gpuwm/doctor.py`
  stays, because a preprocessing install is exactly the one that needs
  to be told which bridge is missing, and it reads its nine dataset
  names from `gpuwm.geog_assets` rather than reaching through the domain
  wizard into the CLI.
- **Both direct-proof descriptors validate against the physics
  registry again.** `configs/gfs_wrf_direct_proof.toml` and
  `configs/era5_wrf_direct_proof.toml` resolved through the legacy
  aggregate radiation spelling, which then demanded `ra_physics = 4`;
  naming `ra_lw_physics = 0` and `ra_sw_physics = 1` in each resolves
  them to `dudhia-shortwave`, and in each `radt` moves 12.0 to 1.0 and
  `diff_6th_factor` moves 0.12 to 0.08 -- the values the profile they
  identify as declares. The two descriptors are deliberately equal apart
  from source, start, and run length, and a test binds that equality.
- **The vendored `wx-core` is 0.3.10** and publishes a capability probe
  -- what this build's NOMADS governor is configured to do, read from the
  places the pacing code reads -- so a consumer can demonstrate the
  pacing rather than infer it from where the crate lives. Two copies of
  that crate shared version 0.3.9 and only one carried the governor; a
  dependency-graph reorder can no longer swap in the governorless copy
  under the same version string.

## Unreleased â€” physics reopening battery fixes

### The documented HRRR front door was dead, and the suite could not see it

- **Fixed:** `tools/prepare_hrrr_wrf.py` demanded receipt schema
  `gpuwm-hrrr-microphysics-initialization-v1` while
  `tools/hrrr_single_domain_benchmark.py` had moved to `v2`, so a fully
  successful preparation still raised `RuntimeError: HRRR preparation
  omitted deterministic cold-start evidence` -- for all seven profiles,
  taking the `gpuwm.wrf_direct` stock-WRF export and every
  `--root-preparation` tree with it. The consumer now validates the
  **fields**: the QC/QR/QI/QS/QG partition, the per-species
  decoded-source and initialized-state fingerprints, and the retention
  claim each species makes about itself.
- **Older schemas are deliberately not accepted.** `main` runs the
  producer itself and reads back the `report.json` that invocation just
  wrote, so an older receipt cannot reach this consumer and accepting one
  would widen the gate for a case that cannot occur.
- **The test shape was the defect.** `tests/test_prepare_hrrr_wrf.py`
  built the receipt it wanted instead of consuming the producer's, so the
  two were never bound to one schema constant. It now runs the real
  producer -- decoded native fields through `initialize_real`, then
  through the same `_initial_hrrr_microphysics_receipt` the benchmark
  binds into `report.json` -- so a future schema bump breaks the consumer
  the moment it lands.

### A receipt now states its own evidentiary strength

- The analyzed-hydrometeor retention gate is one-sided (`source nonzero
  > 0 and live nonzero == 0 -> raise`), so on a cloud-free analysis it
  cannot fire and a green receipt proves nothing. Each species is now
  recorded `PROVEN` or `VACUOUS` with the counts behind the verdict, and
  the domain carries a `retention_evidence_summary`. No threshold is
  invented for "negligible" beyond exact zero; the numbers an acceptance
  harness would need to impose one are published instead. Schema
  `gpuwm-hrrr-microphysics-initialization-v3`.

### RUC no longer meets a land column carrying water soil

- **Fixed:** a land column with `SOILTYP = 14` reached the RUC cold start
  and produced a non-finite `MAVAIL`, killing native HRRR RUC on its
  first surface call with zero model time advanced. WRF never lets that
  column exist: `dyn_em/module_initialize_real.F:3608-3650` reconciles it
  at `real.exe` time -- to land (`IVGTYP 5`, `ISLTYP 8`) from its soil
  temperature, to water from its SST, and to
  `wrf_error_fatal('mismatch_landmask_ivgtyp')` with neither -- and RUC
  assumes that without re-verifying it (`soilvegin`,
  `phys/module_sf_ruclsm.F:6973-6984`, has no `else` for `isltyp == 14`,
  so the column keeps zeroed soil parameters and `:913` evaluates
  `0./0.`). `gpuwm/core/landuse.py`, the transcription of that same cold
  start, now performs the reconciliation, which fixes every source at one
  seam rather than clamping a NaN where WRF has no clamp.

### MYNN is reachable with something other than WSM6

- `gpuwm import-namelist` had no mapping for `bl_pbl_physics = 5`, so the
  MYNN suite was expressible only through the one fixed profile that pins
  `mp_physics = 6`. The readiness authority already admitted the coupled
  5/5 pair and the dispatch already ran it; only the importer lagged. No
  gate is widened -- the half-suite, MYNN+RUC and MYNN+Noah-MP refusals
  are unchanged and pinned by tests.

### The air-gapped Rust build

- **Fixed, and the cause was not a missing crate.** Cargo finds the
  crates.io -> `vendor/crates-io` replacement in the crate's own
  `.cargo/config.toml` by walking up from the **working directory**,
  never from `--manifest-path`, so a build driven from the repository
  root bypasses the vendored registry and resolves against crates.io.
  `tools/prepare_hrrr_wrf.py`'s decoder auto-build,
  `tools/prepare_hrrr_500_native.sh` and `docs/benchmark.md` now `cd`
  into the crate as README/`install.sh`/`install.ps1`/CONTRIBUTING always
  did. A new test refuses any shipped build site that reaches a vendored
  workspace through `--manifest-path`, and checks every locked external
  package is actually vendored.

## 1.1.2 (2026-07-30)

### A saturated soil cell is packing, not corruption

- **Fixed, and this is why 1.1.2 exists:** a pip user's GFS preparation
  died with `RuntimeError: GFS Rust bridge failed: GFS_SM010040 value
  1.0000000019073487 outside [0,1]`. Nothing was wrong with their data.
  GRIB2 reconstructs every value as `(R + X * 2^E) * 10^-D`, so the
  representable values sit on a grid of spacing `2^E * 10^-D` and the
  encoder rounds onto it. A cell whose soil is physically **saturated**
  is encoded at exactly 1.0 and decodes one step above it -- and that
  reported number is exactly one step of that record's own grid
  (`E = -19`, `D = 3`, one step = 1.9073486328125e-9), which is the whole
  argument. It is a statement about the packing, not about the data.
- **How it is fixed, and how far:** every bound now declares what it
  **is**. A *physical* bound is a saturating limit real cells sit on;
  a value past it by no more than the tolerance derived from that
  record's own quantum is clamped onto the bound, counted, and
  published. A *sanity* bound is a slack plausibility range no real
  value approaches; it is offered nothing and refuses exactly as before.
  The offer cannot be talked upward by the record: it is the quantum,
  raised only to the round-off floor of the bound's own magnitude -- so
  a field with a slack ceiling cannot buy extra room at its physical
  floor -- and capped at 1e-4 of the field's declared range, so integer
  packing (one step = the whole range) or template 5.4's placeholder
  scale slots cannot widen the gate. `1.05` still refuses, and now says
  by how much and against what offer.
- **The exposure was never one field.** In the GFS bridge: `RH`, `RH2`,
  `SNOW` and `SNOWH` at zero; `XICE` and all four `GFS_SM` soil layers
  at both ends; and `LANDSEA`, whose 0/1 codes were matched against a
  fixed 1e-9 that the same grid can overshoot. In the HRRR bridge the
  identical shape: every field declared non-negative (a hydrometeor
  mixing ratio is exactly zero across most of a domain) and the
  `LANDSEA`/`XICE`/`SOILW` unit fractions. `Q2` keeps its exact ceiling
  -- one kg/kg of water vapour is an impossibility, not a limit real
  cells sit on, so no quantization argument applies to it.
- **The clamping is auditable, not invisible.** `gate.tsv` gains
  `bound_clamp_total`, `bound_clamp_max_excursion` and
  `bound_clamp_fields` (per field: count and worst excursion); both
  bridges' inventory manifests gain `clamped` and `max_excursion`
  columns. A run with no clamps says so with zeros and an empty field
  list, which is a stronger statement than silence.
- **And the refusal explains itself.** `GFS Rust bridge failed:
  <stderr>` was true and unreadable on its own -- a soil value of 1.05
  means nothing to someone who did not write the decoder, and the
  obvious reading, that the range is too tight, is the one action that
  must not be taken. An out-of-range refusal now carries what the number
  is, that a bound-kissing value is already clamped against the record's
  own packing step, and that a value past *that* points at the
  downloaded bytes: re-fetch and re-run. Every remedy line is a comment,
  so the block pastes whole. Failures that are not bound refusals gain
  nothing.

### Three surfaces that were telling users something untrue

- **Fixed:** `gpuwm.__version__` was a hand-typed `"0.1.1"` that four
  releases walked past. It is read from the installed distribution's
  metadata now, because two surfaces quote it to say which release is
  speaking: the prepared-cache provenance refusal, which told a 1.1.1
  user *"this is gpuwm 0.1.1"* -- a sentence whose entire job is to name
  the release -- and `rw-wps --version`. The test pins it to the
  metadata **and** rejects a release-shaped literal in the module, since
  a constant that matches today's install passes the first check and
  rots at the next cut, which is exactly what happened. Cache content
  digests are unaffected: the writer's version stamp was already outside
  the hashed basis.
- **Fixed:** `gpuwm doctor` announced *"NO basemap assets found"* on
  installs where `rw_wrfbatch` was drawing the coastlines it said were
  missing. Doctor probed one path -- the checkout's own
  `tools/rustwx/assets/basemap` -- while the renderer walks its own
  candidate list. Doctor now walks the same list in the same order: the
  two `RUSTWX_*` overrides, then `assets/basemap` (and the macOS
  `Resources` layout) under each of the first eight ancestors of the
  **executable's** directory -- which is how a build at
  `tools/rustwx/target/release` reaches the crate's assets two levels up
  -- then the working-directory walk, then the crate path it used to
  check alone. Doctor already resolved the renderer in order to report
  on it, so it had the missing fact all along. The warning survives for
  the case it was written for: nothing found anywhere still says so, and
  still names the environment variable.
- **Fixed:** the 20CRv3 authoring step printed nothing at all. The GFS
  route ends by printing the whole front-door command with its digest
  filled in, and every mapped authoring step prints an `AUTHORED` line;
  a user who had just watched a 20CRv3 manifest be written still had to
  locate it and compute its SHA-256 by hand. It now prints the
  `AUTHORED` line and a `next:` block: the half it knows
  (`--source-manifest` and `--source-manifest-sha256`, bound and exact)
  as a pasteable fragment, and the half it cannot know named in
  comments. It does not print a whole command, because 20CRv3 authoring
  deliberately **refuses** `--wps-namelist`, `--geog-root`,
  `--experiment-config`, `--output-root` and the two GRIB2 tool paths --
  those values do not exist in that process, and a command with
  placeholders in it fails when pasted.

### The same class, everywhere it existed

- **Fixed:** an audit reproduced the reported `1.0000000019` refusal on
  the HRRR and mapped routes, not only GFS. The strict `[0,1]` gates on
  HRRR source `LANDSEA`/`SOILW`, on the mapped soil output, on the HRRR
  soil nodes and on declarative mapped soil moisture all read a
  saturated cell as corruption. The bridges derive their tolerance from
  a record's own packing parameters; by these seams the packing is gone
  and only an array remains, so the head-room there is the round-off the
  pipeline demonstrably carries -- a few float32 ulps of the bound's own
  magnitude, the same constant the HRRR soil report already used for its
  convex-hull comparison. It moves cells that are AT a bound back onto
  it and leaves everything else untouched, so each existing refusal
  still sees, and still refuses, exactly what it did before.
- **Fixed:** mapped land fraction gained the check it never had. It was
  tested for finiteness and then thresholded at 0.5, so a mis-scaled
  unit transform delivering `2.0` was read as land without complaint. It
  is a fraction and is admitted as one. (The netCDF test fixture's own
  land fraction walked to 1.25 -- its helper adds 0.25 per time step to
  every variable -- and is corrected to a physical value.)
- **Fixed:** the stale version was not only printed, it was **sealed
  in**. The standalone RW-WPS wheel's pyproject carried a hardcoded
  `0.1.1` while `_installed_record_receipt` refuses unless that
  distribution's version equals `gpuwm.__version__`; the two agreed by
  coincidence. The moment the constant started telling the truth, a
  hardcoded version there would have failed the very seal it feeds. It
  is stamped from the package now, and a test proves a freshly sealed
  native contract, a prepared-cache writer stamp and that wheel all
  carry the distribution version.

### Commands that survive being pasted

- **Fixed:** a successful HRRR fetch printed a front-door command ending
  in a literal `...`, which its own consumer rejects with `unrecognized
  arguments: ...`. A GFS fetch without `--author-front-door-manifest`
  printed a template carrying `GFS_GRIB2_BRIDGE_EXE`, `NAMELIST_WPS` and
  `EXPERIMENT_TOML` and called it "next". Both now print the bound half
  as a real command and name the rest in comments.
- **Fixed:** the materialized GFS front-door command, the wizard's
  `next:` and `check:` lines, and both prepared-forecast commands
  interpolated paths bare, so a perfectly valid `--out`, config or
  `--outdir` containing a space split into two arguments the moment the
  command was pasted. They render POSIX display form and quote when a
  shell would split, exactly as `rw-wps --dry-run` already did. The new
  test shell-parses the printed line back to argv through a path with a
  space in it -- the old checks were lexical and could not see this.
- **Fixed:** the missing-CuPy remedy put a parenthesised alternative on
  the same physical line as the command it followed. Separate command
  and comment lines, which is doctor's form.
- **Fixed:** `--force-refetch` said it moves "every existing file in
  `--out`" aside. It moves the files that fetch would write; manifests,
  forecast hours you did not request and unrelated files stay. The
  behaviour was right and the scope word was not.

### Pages that described a build we no longer ship

- **Fixed:** the physics route table was labelled "state of play in
  v1.0.1" and claimed the GFS/HRRR door could prepare only YSU + MM5
  surface layer + Noah. v1.1.0 removed that coupling. The table now
  lists the shipped routes, names the one deliberate withdrawal (GFS +
  RUC, whose initialization the GFS route cannot supply), and points at
  `rw-wps --show-physics-registry` as the authority it summarizes.
  `DATA.md` repeated the same claim and now agrees.
- **Fixed:** README, `CONFIGURATION.md` and `PHYSICS.md` said two-way
  feedback is absent, `0 only`, or rejected at load, while 1.1.1 runs
  and stamps an experimental feedback path. They describe what ships,
  with its limits: experimental, stamped in run provenance, refused by
  one-way consumers, and feeding back dynamic state only where WRF also
  feeds back hundreds of masked land-surface fields.
- **Fixed:** `HARDWARE.md` kept the 3 GiB reserve for 12 and 16 GiB
  cards that v1.1.0 replaced with a flat 4 GiB, and said `gpuwm check`
  "still warns rather than blocks" after it began exiting 4 on an
  exceeded budget. Both corrected, with the distinction spelled out: a
  script reading the exit status is blocked, a person reading the output
  is advised, and nothing prevents a later `gpuwm run`.
- **Fixed:** the announcement draft said every physics scheme is "gated
  three ways". The registry says otherwise for MYNN, Noah, Noah-MP and
  RUC, and says so in warnings it prints on request. The draft now
  states which gates each option has passed and points at the registry.
- **Fixed:** README promised that every doctor gap "prints the exact
  command that fixes it". Doctor's own contract -- every remedy LINE is
  a command or a `#` comment -- is the accurate one, and one gap
  (`GPUWM_CASE_DATA_ROOT`) needs a path only the user knows.
- **Fixed:** `gpuwm domain --help` offered "any land point on earth"
  while the parser refuses a pole and the South Pole is land. The help
  now says what the gate does. The gate is unchanged.

### The last front door that said nothing

- **Fixed:** three sources reach the prepared-forecast runners -- GFS,
  ERA5 and 20CRv3 -- and two of them ended a successful preparation by
  printing a complete, hash-bound run command. ERA5 dumped its proof
  document and stopped, so a user who had just prepared a bundle
  reconstructed three SHA-256 values by hand, one of them findable only
  by grepping the JSON. It routes through the same shared printer now.
  The mapped route stays deliberately silent: `mapped` is not a
  `--source` either runner accepts, so a `next:` there would lead
  straight to a refusal -- the GDAS dead end this project already
  decided against.

### Guidance that still named a withdrawn gate

- **Fixed:** the RUC physics template's own warning said it is "OFFERED
  FOR ... gfs" while the route matrix withdrew GFS in v1.1.1 -- a
  GFS-initialised RUC forecast prepares and then cannot take its first
  step. The admission tests checked the matrix and never the prose
  against it, so the template kept telling users what the gate had
  stopped offering. The warning now names exactly the sources the matrix
  reaches (HRRR and ERA5) and explains the withdrawal, and a test
  asserts prose-and-matrix agreement so the two cannot drift apart
  again.
- **Tightened:** the pasteability guard added earlier this release now
  runs over BOTH prepared next-command branches -- single-domain and the
  multi-domain hierarchy -- with the config, namelist and output all in
  a directory whose name contains a space, shell-parsed back to argv.
  And the README's doctor claim ("a remedy whose every line is a command
  or a `#` comment") is now bound to doctor's real behaviour by a test
  that proves a genuine comment-only gap exists and rejects the old
  "every gap prints the exact command" overclaim.

### Carried, not fixed here

- Ordinary missing-file arguments still produce raw tracebacks on
  several public CLIs rather than a sentence. The fix pattern exists in
  this codebase; the surface is wide enough to want its own pass.
- Install and remedy guidance still points at mutable `main` rather than
  the released tag. Pinning it is a release-process decision, not a
  patch-lane one.

## 1.1.1 (2026-07-30)

### The GFS front door stops applying a single-domain gate to nests

- **Fixed, and this is why 1.1.1 exists:** 1.1.0 made the GFS front door
  apply the prepared **single-domain** forecast runner's physics profile
  whitelist to **every** configuration, including multi-domain ones. A
  `max_dom = 2` config that prepared cleanly on 1.0.1 was refused with a
  raw `ValueError` traceback minutes after 1.1.0 shipped. Three things
  followed from one mis-scoped call. The wizard's own note beside every
  emitted config -- *"the multi-domain (domain-tree) runner has no such
  whitelist and runs the suite above as written"* -- became false. The
  product's **default** suite (Thompson MP8 + Kain-Fritsch + RTE+RRTMGP)
  is deliberately not in that whitelist, so the config `gpuwm domain`
  emits when nobody names a profile could not pass the GFS front door at
  all. And the refusal arrived as a stack trace. The whitelist now gates
  single-domain preparation only, which is the boundary the wizard was
  describing all along.
- **Unchanged:** a single-domain config still meets the whitelist
  exactly as it did in 1.1.0, defaulting to the WSM6 profile when the
  caller names none, and an explicit `--physics-profile` is still
  enforced on either route -- a gate you asked for is not a gate to
  drop. On a domain tree it binds the root, because the wizard's own
  nested emission of a shipped profile turns cumulus off on the inner
  domains. Nothing the multi-domain path refused before 1.1 is accepted
  now.
- **New:** the multi-domain preparation records what each domain
  selected. `gpuwm-gfs-native-hierarchy-proof-v1` -> **`-v2`**, carrying
  a `gpuwm-front-door-physics-selection-multi-domain-v1` receipt: one
  selector set per domain (a child chooses its own cumulus and radiation
  cadence), the registry's semantic SHA-256, and the registry's names
  for each selection where it has them. Where it does not -- the
  committed two-domain descriptor selects the legacy aggregate radiation
  spelling with radiation off, which the registry has no option for --
  the receipt carries the blocker text instead of a guess, and the run
  proceeds. Naming is provenance here, not permission: requiring the
  registry to name a tree's physics would have been the same regression
  in better clothes. Both prepared-forecast runners accept v1 and v2 as
  distinct schemas and never promote v1 by inference, the rule the
  direct proof's v2 already lives under.
- **Fixed:** every refusal `python -m gpuwm.gfs_direct` makes now reaches
  you as one sentence and exit code **2** -- what the 20CRv3 door already
  costs for the same class of refusal -- instead of a traceback. That
  covers the two node-8 tracebacks on this path: the physics scoping
  above, and a manifest hash-bound to a namelist that was since
  re-pointed. The gates are untouched; only how they arrive is.

### RUC on the GFS route refuses at preparation, not mid-forecast

- **Fixed:** the RUC land-surface template was selectable through the
  GFS front door and could not complete a forecast. It prepared cleanly
  -- proof PASS, `land_surface: ruc-lsm`, nine soil layers, 339 MB of
  prepared state -- and then died 2.8 s into integration with `mavail
  must be finite`, having advanced no model time. The fail-closed guard
  did its job and the partial output was labelled
  `PARTIAL_NOT_RUN_PASS`, so nothing wrong was produced; what was wrong
  was spending the whole preparation to reach a refusal. The GFS route
  now refuses the pairing **before** any GRIB is decoded, with a
  registry-cited blocker that names what was observed. `gpuwm domain`
  refuses to emit the pairing at all, so the config never gets written.
- **Scoped to the evidence, deliberately:** RUC on the **ERA5** and
  **HRRR** routes is unchanged and still offered. It was not exercised
  by the run that found this, and withdrawing it on the inference that
  it shares the defect would refuse a path nobody has shown to be
  broken. Completing the GFS route's RUC land/soil initialisation is a
  v1.2 item, and the registry, the runner's capability declaration, and
  the front door now say the same thing about which sources offer it.
- **Mechanism worth knowing about:** the registry has always published
  which templates each route offers each source, but until now only plan
  validation and the GUI read that declaration -- nothing consulted it
  before a preparation ran. That gap is what let "selectable" and
  "usable" drift apart. Preparation now enforces the same declaration.

### Upgrading no longer invalidates what you already prepared

- **Fixed, and it affected every user with a prepared tree:** 1.1.0 gave
  every domain an optional per-domain `start_time` for staggered nest
  starts. The prepared-cache identity is compared by strict equality, so
  a header written by 1.0.1 -- before the field existed -- could never
  match again, and **every 1.0.1-era prepared tree became unrunnable
  under 1.1.0**, refused with `d01 cache domain config differs from
  experiment`. That sentence names the user's experiment TOML, which was
  innocent; the cause was a package upgrade. A field diff of a real
  preserved tree found exactly one added key and zero value differences
  among the eleven shared keys and ~110 `run` fields.
- **How it is fixed, and how narrowly:** a field the header does not
  carry, whose live value means the feature is not in use, describes the
  same prepared state as a header written before the field existed --
  for `start_time`, a domain whose start is the experiment's start,
  i.e. no delayed start. That case is accepted and the tolerated field
  names are recorded in the run's provenance. Everything else still
  refuses: a `start_time` that is genuinely late, any value that
  differs, any field the header carries and this build does not (a cache
  from a *newer* gpuwm), and every source/static/namelist/bridge digest,
  which are hashes of bytes and were never relaxed.
- **New:** cache headers now stamp the gpuwm that wrote them, outside
  the hashed content basis so that every existing cache's digest keeps
  verifying exactly as before. Residual mismatches now refuse with the
  honest cause -- `prepared by <version>, this is <version>; these
  identity fields differ: [...]` -- instead of pointing at a
  configuration file for a package difference.

### `gpuwm doctor` catches a bridge that predates the contract

- **Fixed:** the wheel ships no Rust, so upgrading the Python half
  leaves yesterday's bridge binaries in place. 1.1.0 changed the GFS
  series file from two columns to three, and `gpuwm doctor` reported
  every 1.0.1-era bridge `ok` because it only asked whether they
  launched -- after which each preparation died with `series line 1 must
  be HOUR<TAB>GRIB2`, blaming the series file gpuwm had just written
  correctly. Each bridge now declares a marker of the contract it
  speaks, and doctor reports a bridge that lacks it as **MISSING** with
  the rebuild remedy. The check is static, so it works on the binaries
  already on disk. `gpuwm.native_wrf_distribution` has applied this
  mechanism before sealing a distribution since before 1.1; there is now
  one table, shared, so the two surfaces cannot drift again.

### `gpuwm doctor`'s bootstrap wires what it builds

- **Fixed, pip installs:** pasting the whole report on a pip-only
  machine and running every line of it left doctor still reporting six
  MISSING bridges. Every line was honest -- the wiring step was offered
  as two `#` alternatives, copy into `~/.gpuwm/bridges` **or** set
  `GPUWM_<X>_BRIDGE`, because it genuinely is a choice -- but a choice
  printed entirely in comments means "run all the commands" does not
  close the gap the commands were for. The copy is now the printed
  **command**, correct for the shell you are in (`mkdir -p` / `cp`;
  `New-Item -ItemType Directory -Force` / `Copy-Item`, because Windows
  PowerShell 5.1 has neither of the first pair), and the environment
  variable is the `#` alternative beneath it. The destination is the
  default bridge directory spelled out in full rather than through
  `$HOME`, so the path you paste is the path gpuwm searches even where
  the two disagree.

### `gpuwm adapt` stops emitting a descriptor its own battery rejects

- **Fixed:** `gpuwm adapt --skeleton` gave four 3-D fields their
  **surface** counterpart's selector in addition to their own --
  `air_temperature` claimed the 2 m row that `air_temperature_2m` also
  claimed, and the same for relative humidity and both wind components
  -- so filling in the scaffold honestly and running the battery got
  `Vtable line 11 is assigned more than once`. The prefix rule that
  correctly collects `soil_temperature_0_0.1m` under `soil_temperature`
  was swallowing `air_temperature_2m` under `air_temperature`; it now
  takes the **longest** matching canonical name, which separates the
  four pairs without a hand-maintained exception list and leaves the
  four-layer soil collection intact.
- **Fixed:** that refusal named the Vtable and a line of it that was
  perfectly fine. Both claims live in the **descriptor**, so the message
  now names the descriptor and both fields that claimed the row -- which
  matters most when, as here, the tool itself wrote the descriptor.
- **Fixed, pip installs:** `configs/Vtable.GFS.rw-wps` -- the worked
  example the adapt flow is documented against -- lived outside any
  package, so the wheel never carried it and the documented command
  named a path only a checkout has. It ships beside the 20CRv3
  authorities now, under the same recursive package-data glob and the
  same byte-verified contract, and `gpuwm adapt --help` prints its
  resolved path on the install you are running. `--vtable` stays
  required and is deliberately never defaulted: this command adapts
  arbitrary sources, and quietly reaching for a GFS Vtable would
  mis-map every other product.

### `feedback = 1` says where it is supported, before you build

- **Fixed, guidance only -- no gate changed:** `feedback = 1` is a legal
  schema value, so a config could be authored, pass `gpuwm check` with a
  clean exit 0 and output identical to its one-way twin, and only then
  be refused at preparation -- after a 26 s hierarchy build -- by the
  prepared-hierarchy route, which supports static one-way nests only.
  `gpuwm check` now prints an advisory (and reports it under
  `advisories` in `--json`) naming the restriction: it changes no exit
  code and blocks nothing. And the preparation refusal now names the
  route that *does* run experimental two-way feedback -- the native
  experiment-runner route, `gpuwm run` -- instead of stating only what
  it will not do.

### Refusals

- **Fixed:** a `--preparation-receipt-sha256` that matched nothing
  printed `proof.json digest differs` once per accepted (file, schema)
  pair -- three identical lines, four once the GFS hierarchy proof grew
  a v2 -- and named neither the file it read nor the digest it found.
  One note per file now, naming the resolved path and the observed
  SHA-256 (or the schema it carries), plus the list of schemas that
  would have been accepted.
- **Fixed:** `rw-wps --source rap` printed
  `status=adapter_mapping_required: adapter_mapping_required` -- the
  reason fell back to the status value already on the line, and a
  doubled token reads as a truncated message. An adapter with nothing to
  add now says nothing rather than saying it twice. Adapters that do
  have something to say (`gdas`'s notes, the composition family's
  requirement) are unchanged, as is the paragraph underneath that
  explains the refusal.

## 1.1.0 (2026-07-30)

### `gpuwm check` tells the truth about VRAM

- **Fixed:** `gpuwm check` printed "observed peak envelope 12.98 GiB
  exceeds the WDDM budget 11.64 GiB" and exited **0**, so every script
  wrapping it read green out of a report whose own prose said the run
  might not fit. That case now exits **4** -- nonzero, and distinct from
  1 because no gate failed and the levers differ. A harder verdict still
  wins: 1 (a leg FAILED), 2 (nothing evaluable) and 3 (`--alloc`
  aborted) all outrank it. The warning line names the code it will exit
  with, so the reader of the text and the reader of `echo $?` learn the
  same thing.
- **Fixed:** the wizard's `--card 16gb` tier reconstructed a *notional*
  free-VRAM figure of 16.68 GiB -- more than a 16 GB card physically
  has -- because `--budget-gib` plus the reserve is arithmetic that never
  saw the card. `gpuwm check` gained `--vram-gib`, a ceiling and never a
  source: a declared free figure is now clamped to the smaller of the
  named card's capacity and a live NVML reading of it, reported as
  `CAPPED` in text and `free_bytes_capped_to_physical_bytes` in `--json`.
  Without `--vram-gib` nothing is clamped, because `--budget-gib` is how
  you size for a machine that is not this one.
- **Fixed:** `gpuwm check` sized its estimate and its observed-peak
  envelope without knowing the card, so on a 12 GiB card it applied the
  32 GiB machine's pool constants and the 1.75 Windows envelope factor
  while the wizard applied the small-card constants and 1.45 -- the two
  surfaces disagreed by 6.9 GiB on the same config. The wizard now passes
  its card size to `check`, and `check` sizes for the card it is told
  about.
- **Fixed:** the wizard's flat small-card reserve was 3.0 GiB, but the
  reserve policy `gpuwm check` actually applies charges 3.5-3.6 GiB on
  those same 12 and 16 GiB cards -- so the wizard sized layouts against a
  budget the preflight would never grant, and certified them anyway
  because of the exit-code bug above. The 12 and 16 GiB tiers now reserve
  4.0 GiB, the figure the 24 GiB tier already used.
- **Changed, and you will see it:** put together, the four fixes above
  mean **`gpuwm domain` emits smaller domains on the 12 and 16 GiB tiers
  than 1.0.1 did** for the same request. Nothing about your card
  changed; what changed is that the wizard and the preflight now size
  against the same card, the same reserve and the same envelope factor,
  and the answer they agree on is the smaller one. 1.0.1's larger
  domains on those tiers were sized against a budget the preflight would
  not have granted and a free-VRAM figure the card did not have -- and
  the preflight said so and exited 0 anyway. The 24 and 32 GiB tiers are
  unchanged. If you have a 1.0.1 domain that ran, it still runs; it is
  the *suggested* size that moved.

### Remedies

- **Fixed, pip installs:** `gpuwm render`'s matplotlib-fallback notice
  still ended `Build it with: cd tools/rustwx; cargo build ...` -- a
  directory a wheel install does not carry. The 1.0.1 remedy contract
  fixed that everywhere except here, because this call site assembled
  its own string. It routes through the install-aware machinery now, in
  a new one-line form: the `cargo` one-liner where the crate exists, and
  a pointer to `gpuwm doctor` where it does not, because the honest
  answer there is a whole bootstrap and this notice is contractually one
  physical line.
- **Fixed:** `bridges.sources_present()` answered for
  `tools/grib1_bridge` no matter which crate it was asked about, so a
  tree carrying one crate and not the other could be handed a `cd` into
  the missing one. It now takes the crate it is asked about;
  `install_aware_build_hint` passes it through.
- **Fixed:** `gpuwm doctor` says its remedy blocks run "as printed, in
  the order printed", and they did not. A `cd` into a crate never came
  back, so the block after it resolved its relative paths somewhere
  else; the repeated `git clone` a pip-only machine gets once per gap
  errored on every repeat; and two remedies ended with prose fused onto
  the end of a command line, which the shell hands to `cargo` as
  arguments. Every block now returns to the directory it started in,
  the clone carries a note telling you to skip it when the directory is
  already there, and every physical line of every remedy is either a
  command to run or a `#` comment. The claim is now enforced by a test
  that pastes the *whole* report as one sequence, in both shells,
  twice.
- **Fixed:** continuation lines of a multi-line remedy were printed at
  whatever indentation they were composed with -- 0, 2 or 4 spaces --
  so a block could start under the `remedy:` label and then jump to
  column 0. The whole block lines up now.

### The rust renderer says what is wrong

- **Fixed:** `rw_wrfbatch` answered an unknown `--products` slug, a file
  that is not a wrfout, a path that does not exist, and a bare
  `--list-products` with the *same* line -- its usage string -- and exit
  1. Three of the four never named the thing that was wrong. Each now
  names the problem: an unknown product names the token, the five group
  keywords and where to get the full list; an unusable input names the
  path and why, in the same wording matplotlib's engine uses
  (`<path>: unreadable wrfout (...)`).
- **Fixed:** a `--products` typo used to cost a full wrfout import
  before anything checked it, and then reported `No supported WRF files
  selected` -- about a file, not about the typo. The command line is
  checked before any file is opened.
- **Fixed:** the usage line was printed *after* the message, and
  `gpuwm render` reports the renderer's last line as the cause -- so
  every argument mistake reached you as a usage string. Usage first,
  reason last, as argparse does it.
- **Changed:** `rw_wrfbatch` exits **2** on a bad command line, matching
  what the matplotlib engine costs for the same mistake. `gpuwm render`
  still exits 1 either way, so scripts wrapping the front door see no
  change.
- **New:** `rw_wrfbatch --list-products` with no store, no output
  directory and no input prints the product vocabulary you may pass to
  `--products`. The store-aware listing -- which of those the frames you
  imported can actually render, and why not -- is unchanged and still
  needs a store.

### Guidance catches up with the gates

Every gate in this group was already right; the guidance around them
was not, and three of the four surfaced as tracebacks.

- **Fixed:** the GFS front door's own suggested `--outdir` was
  `<prepared_root>/forecast` -- a *child* of the preparation, while both
  prepared-forecast runners declare `--prepared-root` a protected input
  and refuse any output directory overlapping it. The front door's own
  next-command was therefore refused, as a raw traceback. The suggestion
  is now a genuine sibling, `<prepared_root>-forecast`, and the
  pasteable test no longer just looks at the printed text: it runs the
  runner's actual guard over the printed `--outdir`.
- **Fixed:** both runners' protected-input refusals reach the user as
  one sentence naming the problem and a directory that works, and exit
  2, instead of a traceback out of `claim_output_directory`.
- **Fixed:** the 20CRv3 front door rejected the wizard's default TOML
  with a raw `unknown table(s)/top-level key(s) ['case_data']`
  traceback. It now checks config compatibility *before* decoding any
  GRIB2 and refuses in one sentence that names both the incompatibility
  (`[case_data]` declares the ERA5 config-driven run path, which this
  door does not read) and a supported route to a config it accepts.
- **Fixed:** the 20CRv3 front door printed no run instruction at all,
  while the GFS door printed a complete hash-bound command -- after the
  1932 hindcast had just run end to end. Both doors now use the same
  printer, with `--source 20crv3`; every digest is resolved, so the
  pasteable contract holds without a holed command.
- `gpuwm domain` still has no `--source 20crv3` option. DATA.md's 20CRv3
  section now states the supported config route explicitly
  (`gpuwm domain --source gfs`, which emits no `[case_data]` table), and
  the front door's refusal names the same route.

### Any-combo front-door physics

- The single-domain GFS and HRRR front doors carry the *selected*
  registered physics profile all the way through configuration
  materialization, preparation, native WRF export, and content-hashed
  provenance. The exporter no longer applies a stock
  `bl_pbl_physics=1 / sf_sfclay_physics=91 / sf_surface_physics=2`
  identity gate when a front-door profile is supplied: it resolves each
  selector through the physics registry and checks the component's
  `implemented` declaration, required settings, pairings, and runtime
  readiness rails, failing closed with a registry JSON pointer and the
  published blocker when a component is unavailable.
- Three profiles that were declared but unreachable are now reachable on
  the prepared single-domain route: WSM6 + MYNN 5/5 + Noah, WSM6 +
  YSU/MM5 + RUC LSM (with its required nine soil layers), and WSM6 +
  YSU/MM5 + Noah-MP behind the registry-owned
  `noahmp-host-column-throughput-v1` expert acknowledgement. **This
  closes the 1.0.1 known issue** where the registry marked MYNN, RUC and
  Noah-MP `reachability: template` while `--physics-profile` rejected
  them: the two surfaces now agree, and they agree by making the schemes
  preparable rather than by hiding them.
- Provenance: `gpuwm-gfs-direct-wrf-proof-v2` ->
  `-v3` and `gpuwm-native-direct-wrf-export-v2` -> `-v3`, both carrying a
  `gpuwm-front-door-physics-selection-v1` receipt (profile, registry
  semantic SHA-256, resolved components, selectors, complete runtime
  settings, maturity, accepted acknowledgements). Preflight accepts v2
  and v3 as distinct schemas and fails closed on anything else; exact
  proof-file SHA-256 verification is unchanged for both.
- Stock-profile export identity, stated exactly: at integration time the
  pre-change exporter (`c16aed0e` `gpuwm/wrf_direct.py`) and this one
  were each run against *one* identical fully specified stock-profile
  prepared cache, static cache, geometry receipt, valid time and
  boundary cadence, and the two artifacts came out byte-identical --
  949,606-byte `wrfinput_d01` and 539,836-byte `wrfbdy_d01`, matching
  SHA-256 both sides. That comparison is **not** reproducible from this
  tree: it needed the archived base exporter and a real prepared cache,
  neither of which the release contains, and no committed fixture
  renders a stock export. The four digests and the exact identity inputs
  are recorded in the internal ledger
  `PRODUCT-V11-ANYCOMBO-20260730.md`. What the release *does* gate is
  the surrounding contract -- v2 proofs keep their historical shape, v3
  fails closed on absent, mismatched or future-schema physics receipts,
  and callers that omit a profile keep the legacy stock-only exporter --
  and that is tested (`tests/test_prepared_single_domain_forecast.py`).

### `gpuwm adapt`: arbitrary but verified GRIB2 sources

- New `gpuwm adapt` turns a WPS Vtable, an explicit
  `rw-wps.descriptor.v1`, and the caller's own GRIB2 files into a
  create-only runnable mapped adapter bundle. Its machine status is
  `runnable_mapping_not_stock_wrf_certified` -- it never widens or
  inherits a stock-WRF certification gate.
- Publication is gated on a battery that runs before anything is
  written: Vtable compilation into exact numeric GRIB2 selectors, a
  record inventory at every valid time (complete pressure sets, bounded
  soil selectors, no duplicates or member mixing, uniform cadence equal
  to the declared boundary interval), exact target units/axes/location,
  an *executable* decode of the selected records through the real GRIB2
  decoder, soil selector parity with contiguous surface-down coverage
  and no synthesized layers, one shared GDT 0 grid at scan `0x40`, an
  explicit source top at or above the model top, and stable before/after
  identity for every input and both decoder executables.
- Documented in `docs/arbitrary-verified-adapters.md`, including the GDT
  boundary and the runnable-versus-certified definition.

### GDAS

- `gfs_grib2_bridge` no longer infers forecast process 96 from
  `hour > 0`. The generic series contract is now
  `HOUR<TAB>GRIB2[<TAB>FORECAST_PROCESS_ID]`: a two-column legacy row
  declares analysis process 81 only, and ID 96 must be declared
  explicitly and stays inside the certified `{81, 96}` capability set.
  `gpuwm fetch` writes the per-row declaration. Certified against a real
  NOMADS proof corpus -- f000/f003/f006/f009 of `gdas.20260729/12/atmos`,
  124 messages each, frozen at centre 7, tables 2/1, PDT 4.0, GDT 3.0 /
  shape 6 / 0.25 degree / scan `0x40`, DRT 5.0 -- committed under
  `tests/fixtures/gdas-process-id/`. Each forecast sample is also
  required to fail under the undeclared analysis-only ID-81 policy. The
  corpus includes the **endpoint** of the certified span, so f009 rests
  on committed bytes rather than on the ladder constant.
- With the bridge re-certified, `gpuwm fetch --source gdas` serves the
  full **f000..f009** span again rather than 1.0.1's analysis-only
  window. The 1.0.1 scoping existed because the bridge was certified
  only against the analysis tag; that is the re-certification event it
  named, and it has now happened with real samples. Past f009 still
  refuses up front and says why.
- **What GDAS still is not:** a front door. `rw-wps --source gdas` has
  no ingest route and refuses. Every public surface now says so in the
  same words -- `docs/public/DATA.md`'s opening routing summary as well
  as its GDAS section, the README feature table, the `fetch` subcommand
  summary, and `fetch --help`'s `--hours` and `--cadence` text, which
  1.0.1 left describing GDAS as analysis-only. A test pins that help
  text to the registry's own `max_forecast_hour` so the two cannot drift
  apart again.
- The opt-in live GDAS smoke (`GPUWM_NETWORK_TESTS=1`) now fetches the
  hours it asserts about. It downloaded f000 alone, asserted the series
  had exactly one row, then required that row to carry both declared
  process IDs -- unsatisfiable by construction, so enabling it could
  only ever fail. It fetches the whole f000..f009 ladder from a live
  cycle and checks the census, the record bar and the declared process
  ID on every hour.

### Nests: delayed starts and sub-hour forcing

- `DomainConfig` carries an optional per-domain `start_time`. The root
  must still equal `[experiment].start_time` and an omitted child value
  inherits it, so existing configurations are unchanged; a child may
  start later than its parent, never before it and never outside the
  experiment run. `gpuwm import-namelist` reads real-shaped per-domain
  WRF `start_year..start_second` columns and writes a child `start_time`
  only where it differs from the root.
- The scheduler keeps a delayed child dormant while the parent advances,
  then at its seam runs the ordinary nest initialization against live
  parent state and begins normal stepping and output -- no child history
  before that point, and the parent's schedule and history are
  untouched. Restart headers carry `domain_start_time`,
  `domain_start_ticks` and `domain_lifecycle`; a checkpoint taken before
  the seam restores as `NOT_STARTED` and activates exactly once.
- **The whole-hour forcing refusal is gone**, because it was never a
  numerical requirement -- it came from representing the forcing
  inventory as integer hours. Generic mapped hierarchies now use exact
  `forcing_offsets_seconds` when the cadence is sub-hour. The contract
  that IS enforced: integer-second boundary interval, an exact integer
  number of root steps (root Davies/LBC state resets at a top-of-step
  seam), and every delayed child start on both an exact parent-step
  boundary and an exact global forcing seam. Refusals report the
  offending ratio, e.g. `cadence/dt = 31/6`.

### Feedback: experimental two-way nest restriction

- `[experiment].feedback` accepts `1` as an **experimental** capability
  (default remains `0`; `smooth_option` remains restricted to zero
  because parent smoothing is unimplemented). Launch prints the
  experimental warning, the run writes `feedback-provenance.json`, and
  every per-domain wrfout carries `GPUWM_FEEDBACK`,
  `GPUWM_FEEDBACK_VALUE`, `GPUWM_FEEDBACK_STOCK_WRF_CERTIFIED = 0` and
  `GPUWM_FEEDBACK_CERTIFICATION`. Nothing is stamped at `feedback = 0`.
- `feedback = 0` byte identity, and how to re-run it. At integration
  time an archived pre-feedback tree (`e7bf4d88`) and the lane tip
  (`80561c6e`) each ran the same two-domain deterministic schedule at
  `feedback = 0`, and every serialized state byte matched -- whole-tree
  SHA-256 `727ac476e0ebbf97a89be350a151c593e2b9447cc5d9b14fe0436d5f89e47557`,
  with the per-domain digests recorded in the internal ledger
  `PRODUCT-V11-FEEDBACK-20260730.md`. The release cannot re-run the base
  half of that comparison -- it does not contain the base tree -- but it
  no longer has to take the result on trust either: that pre-change
  digest is now frozen in
  `tests/test_feedback.py::test_feedback_zero_output_is_pinned_and_costs_nothing`,
  which re-derives it from the shipped code and additionally proves the
  dormant path is free (the same run with the feedback call path removed
  entirely produces the identical digest, and the coupler records zero
  transactions).
- The transaction is schedule-owned and three-phase -- restrict `MU`
  first, couple each remaining child prognostic into the existing nest
  scratch arena, call WRF's own `copy_fcn` transliteration over the
  exact registration, uncouple into the parent only on the feedback
  rectangle, refresh parent diagnostics -- at synchronized exact-integer
  clocks, including at a delayed child's activation seam. Restart
  continuation is SHA-identical to an uninterrupted run.
- Operator classes were verified against tagged WRF v4.6.1 source: mass
  (`copy_fcn`, 16-point average at ratio 4, odd centered path at ratio
  3), U on x-faces and V on y-faces all **match**. The masked/integer
  class is a **documented divergence**: ArWen's authoritative
  `nest_field_kinds` inventory contains no masked or integer feedback
  fields, so stock WRF's `LU_INDEX`/`TSK`/`TSLB`/`SMOIS`/... restriction
  has no ArWen counterpart. The `copy_fcnm`/`copy_fcni` kernels remain
  present and exact for a future explicit inventory extension.
- Every one-way consumer fails closed on a feedback-modified parent:
  offline-child/downscale, the static source-hierarchy exporter, RW-WPS
  stock export, and the explicitly-one-way prepared-tree runner.
- `tools/compare_feedback_signature.py` (schema
  `gpuwm-feedback-signature-comparison-v2`) compares a four-run
  ArWen/WRF feedback-on/off matrix over the parent overlap with the
  child specified zone excluded, annotating every row with operator
  class, WRF routine, stagger, stencil and source-point count. It is
  signature comparison, not bit or amplitude certification, and the
  thresholds are synthetic defaults until the real reference pair is
  run. The experimental label stands until then.

### Hygiene

- **Fixed, a fabricated CFL:** the shared one-readback health reduction
  now computes `max(|w_upper| / dz_cell)` on device, with `dz_cell`
  taken from that same cell's live `(php + phb)` geopotential faces.
  The old formula combined global extrema, so 100 m/s aloft over its own
  1,000 m layer reported CFL 100 against an unrelated 10 m surface
  layer; the co-located term reports 1 and passes, while 11 m/s over the
  actual 10 m first layer reports 11 and fails. Both single-domain
  runners share one threshold predicate: exactly 10 passes, the first
  representable FP32 value above 10 fails, and the independent 150 m/s
  `w_max` and non-finite guards are unchanged.
- **Fixed, interrupted fetches:** the GFS/GDAS fetch loop atomically
  rewrites its series and fetch manifest after every verified hour, so
  every published prefix contains only files that passed the full GRIB2
  envelope walk, the resolved record-count bar, and SHA-256.
  `KeyboardInterrupt` is now a traceback-free operational error that
  names each verified file (hour, bytes, digest), separately names any
  `.part` or otherwise unverified payload still on disk, and prints the
  exact `gpuwm fetch` command to resume the original
  cycle/window/cadence/area/output. A first-hour interrupt publishes an
  empty request-identity manifest rather than recreating the old
  manifestless-resume trap.
- **Now loud:** a requested box that crosses the prime meridian cannot
  be one NOMADS request (the CGI accepts a single `[0,360]` interval),
  so it falls back to the full longitude band. That fallback was already
  correct and silent; before any bytes move it now names the requested
  box, the actual band being sent, and `360 / requested_width` as the
  exact longitude-span amplification -- with the compressed-byte factor
  explicitly labelled data-dependent. A 20-degree UK-style box reports
  18x; an equal-width dateline box stays narrow and prints nothing. The
  same area and amplification are recorded in the fetch manifest. No
  unverified two-request stitcher was introduced.

## 1.0.1 (2026-07-30)

- **Documented, not new:** the 20CRv3 ensemble-member route. The
  adapter has shipped and is runnable, but no front-facing document
  mentioned it, so nobody could find it. README and DATA now describe
  it in the registry's own terms -- runnable and experimental, not yet
  accepted by unchanged stock WRF, one exact member bound by a
  filename-plus-hash manifest, paired three-hourly pressure/surface
  analyses, one-way Lambert nests through `max_dom = 4` -- plus what the
  registry does not say: **there is no fetch route and the inputs are
  not self-serve.** Every-member
  20CRv3 GRIB2 generally needs access to the NOAA-CIRES-DOE archive
  holdings; the publicly downloadable mean/spread NetCDF products are a
  different family and are not inputs to this route.

### Known issues

- The physics registry marks the MYNN, RUC and Noah-MP components
  `reachability: template` while `--physics-profile` rejects them as
  invalid choices. Both surfaces are telling the truth about different
  things -- the schemes exist and are declared, but the prepared
  single-domain runner has no profile row for them -- so they are
  visible without yet being preparable on that route. Decoupling the
  two surfaces is scheduled for the next release; until then, the
  profile list in `gpuwm domain --help` is the authority on what the
  prepared route will accept.

- **Fixed, pip installs:** `gpuwm doctor`'s six bridge remedies told a
  pip user to `cd tools/grib1_bridge`, a directory a wheel install does
  not contain, and named no repository anywhere -- so the one machine
  state where *nothing* can decode GRIB got a remedy that reads as a
  broken package. Every Rust remedy is now aware of which install it is
  printing on: in a checkout it stays the single `cargo build` line and
  names the real output path; on a pip install it prints the whole
  bootstrap: the Rust install and the PATH activation that makes cargo
  usable in the shell already running (rustup only edits the login
  profile), then `git clone`, then the build -- about two minutes end
  to end. Every emitted line is either a command spelled for the shell
  this platform actually has -- Windows PowerShell 5.1 cannot parse
  `&&`, so it gets `;` and separate lines -- or a `#` comment; nothing
  is prose fused onto a command, which is what `install Rust: winget
  ... (or https://rustup.rs)` was. The renderer and fetch-backbone
  remedies share the same builder, so they no longer carry `<clone>`
  placeholders, and the pip-extra and `fetch-geog` hints put their
  explanation on `#` lines instead of after the command. README's
  install section says plainly that a pip-only install refuses every
  real data source until the bridges are built. Doctor's closing line
  claims only what it can prove: every remedy line is a command to run
  as printed, in order, or a `#` comment.
- `gpuwm doctor` reads the render extra's version from installed
  package metadata rather than the module's `__version__` attribute,
  which lagged a release (0.2.34 reported on a machine with wrf-rust
  0.2.35 installed).

- **Fixed:** a platform gpuwm has never measured now gets the
  *conservative* VRAM accounting, not the optimistic one. Detection
  recognized Windows (with Cygwin and MSYS) and gave everything else
  the Linux envelope -- so Darwin, a BSD, or any future platform name
  was priced with the numbers that omit 4.12 GiB of fixed pool
  constants, on no evidence at all. Those numbers are three runs on two
  Linux cards, not a default. An unrecognized platform now takes the
  Windows envelope and the wizard prints one line naming the platform
  and saying the accounting may size a smaller domain than the machine
  can run. WSL and Linux containers report `linux` and are unaffected.
- **Fixed:** `rw_fetch`'s probe receipt recorded an absent `.idx` (404)
  as one that arrived and failed validation, so the printed reason said
  the index was malformed when it simply was not published yet. The
  transport half of the index read now tags its own failures instead of
  the caller re-deriving them from prose -- the old prefix match could
  never fire, because the error text it looked for is always preceded
  by the error type's own `HTTP error: `. Transport choice was already
  correct (both classes take the full file); this is receipt fidelity.

- **Fixed:** the front door's printed next-command was not pasteable.
  It filled in the three SHA-256 values -- the whole point of printing
  it -- and then asked for `--physics-profile <the profile this config
  was materialized for>` and `--outdir OUTPUT_DIR`. The profile is now
  resolved by asking the same table the runner's own guard asks, and
  the outdir names a real directory beside the preparation, so the
  command runs exactly as printed. The multi-domain branch printed a
  runner name and a fragment; it now prints that runner's whole
  command, digest and all. A config bound to no shipped profile gets
  prose saying so, never a command that cannot run -- and the same
  treatment reached `gpuwm fetch --author-front-door-manifest`, whose
  `rw-wps` line carried `--output-root OUTPUT_DIR` and `--geog-root
  WPS_GEOG_DIR`.

- **Fixed:** an HRRR inventory-change refusal on the Rust backbone said
  "Nothing was downloaded" when the payload was already on disk. That
  transport can only report a record census after it has written the
  object, so the tripwire necessarily fires late -- and it left an
  unverified GRIB plus its `.idx` in a directory with no manifest,
  which the next ordinary run also refused, for a different and
  confusing reason. The refusal now quarantines both files
  (`*.inventory-change-<ns>`, nothing deleted) *before* raising, and
  says what is on disk and where. Refusals that genuinely precede any
  transfer -- the GFS route derives its bar from the live index first --
  still say nothing was downloaded, because there it is true.
- **Fixed:** a completed HRRR resume erased the record that an
  inventory change had been accepted. Re-running the command on a
  directory whose files are all present downloads nothing, resolves no
  record bars, and republished the manifest with `record_bars: []` --
  losing `inventory_change_accepted: true` from the fetch that really
  did accept it. The prior manifest's bars now seed the run, and any
  product actually re-fetched replaces its own entry.

- **No shipped file carries one machine's absolute paths any more, and
  the release build now refuses to let one back in.** 51 lines across
  25 tracked files still held a developer's WSL/Windows/POSIX home --
  executable defaults in the WRF-oracle build scripts, a committed
  `nvidia-smi` process capture, oracle provenance notes, a hardcoded
  scratch directory. Files a stranger can use were parameterized
  (`$WRF_TREE`, `$WRF_SOURCE_ROOT`, `$GPUWM_REPO`, `RW_STORE_ROOT`,
  `SNOW_PROBE_SCRATCH` -- all fail-closed with a message naming what to
  set); the campaign ensemble harness, which needs a privately-built
  WRF oracle and a private reference bundle and so cannot run outside
  the campaign at all, is excluded from the release instead. The
  snapshot builder now scans every staged text file for the *shape* of
  a per-user absolute path and FAILs the build on any hit, because an
  exclusion manifest cannot name the file that grows one next.

- **Fixed, data loss:** `gpuwm render` silently overwrote one domain
  with another. Two nests of one run share model, init cycle, forecast
  hour, and (on the whole-hour axis) everything else the filename
  carried, so rendering two domains at one lead into one directory left
  ONE PNG -- with no error, no warning, and exit code 0. Every output
  filename now carries a domain + resolution token
  (`..._d02-3km_composite_reflectivity_...`; sub-kilometre nests as
  `_d05-111m_`) on both engines, read from the file's own `GRID_ID` and
  `DX`. Both engines degrade by the same three rules: identity plus a
  usable `DX` gives `d02-3km`; identity without one gives a bare `d02`;
  and a file whose domain identity cannot be established keeps the
  generic `native_grid` slug rather than being labelled `d01` on no
  evidence. On the rust engine, which imports several inputs into one
  store and renders them as one run, inputs that disagree get no token
  at all. The matplotlib engine renders each input separately, so it
  instead refuses -- naming both files and the remedy -- when two
  inputs it could not identify would write the same output name, rather
  than letting the second overwrite the first.
- `gpuwm render --pair` keys on the domain token as well as the product
  slug, so a directory holding several nests cannot pair a 3 km panel
  against a 333 m one. Sheet names gain the domain
  (`d02-3km_sbcape-pair.png`); the panel labels are unchanged.
- Plot subtitles gain the grid spacing the file declares -- `ÃŽâ€x 3 km`
  at and above a kilometre, `ÃŽâ€x 111 m` below it -- on both engines and
  on all four rust render lanes (direct, derived, heavy, windowed).
- Locally-imported runs are labelled **ArWen** instead of the GDEX
  fetch source inherited from the `wrf` store-model identity, which
  this lane never fetched from. New `gpuwm render --source-label TEXT`
  renames it for rendering stock-WRF files honestly.
- The matplotlib fallback notice is now genuinely one line, and carries
  the three things needed to act on it: the engine actually in use, the
  products available against the rust catalog, and the exact
  `cargo build --release --locked --offline` command. The multi-line
  remedy stays in `gpuwm doctor`, where it can be read at leisure.
- **Windows cards at or below 12 GiB are now sized as an EXPERIMENTAL
  tier instead of being refused.** The refusal was never "your card is
  too small": at the wizard's smallest layout, 4.12 GiB of a 5.38 GiB
  projection was grid-independent constants measured on one 32 GiB
  RTX 5090 running campaign-scale forecasts, so no ladder could fit and
  no smaller grid could help. Small Windows cards are now priced like
  Linux -- itemized alloc estimate under the 1.45 envelope -- plus a
  single reduced 1.5 GiB fixed reserve, and every sizing prints an
  honest pioneer warning: the accounting is extrapolated from one much
  larger machine, the worst case is paging or a clean out-of-memory
  failure (neither corrupts a forecast), and please report your
  measured peak. Windows cards of 16 GiB and up are unchanged, and so
  is `gpuwm check`, which warns rather than blocks as before.
- FIRST-LIGHT, README, and HARDWARE named `run-progress.json` as *the*
  progress file. It is the config-driven `gpuwm run` route's file only:
  the domain-tree runner writes `<outdir>/evidence/progress.json` and
  the single-domain runners write `<outdir>/progress.json`. All three
  documents now state the truth for every route.
- The single-domain CFL safety gate now reduces
  `max(dt*|w_upper|/dz_cell)` with velocity and live geopotential
  thickness from the same cell. It no longer combines a global
  upper-level updraft with the unrelated thinnest surface layer.
  Genuine thin-layer violations, non-finite geometry, the CFL 10
  threshold, and the independent 150 m/s vertical-speed guard remain
  fail-closed.
- GFS/GDAS fetches now atomically refresh their series and fetch
  manifest after every verified hour. Ctrl-C reports the exact
  digest-bound prefix and any unverified partial file on disk, exits
  without a Python traceback, and prints the exact resume command; a
  good completed hour is never discarded merely because a later hour
  was interrupted.
- A GFS/GDAS box crossing 0 degrees longitude no longer silently widens
  to NOMADS' full `0..360` band. The verification-preserving full-band
  fallback now prints and manifests the requested box, fetched band,
  and exact longitude-span amplification (while labelling compressed
  bytes as data-dependent). Splitting remains deferred because
  concatenating two 124-record grids is not one geometry-valid
  124-record product.

- New `gpuwm fetch-geog`: downloads and stages the nine WPS_GEOG
  static datasets the static builder opens (~1.3 GB download, ~16 GB
  unpacked) -- previously an entirely manual step and the launch-day
  pilot's #1 finding. Default source is the ArWen Hugging Face mirror
  (CDN bandwidth); `--source ncar` fetches upstream; both are verified
  against packaged size + SHA-256 pins (NCAR publishes no checksums;
  the pins were computed from UCAR's bytes on 2026-07-29). Resumable
  (HTTP Range), idempotent, safe extraction, per-dataset WPS `index`
  validation (doctor's own bar), and a local
  `geog-fetch-manifest.json` audit record. `gpuwm doctor` and the
  wizard now print this command as the WPS_GEOG remedy; DATA.md
  documents both routes, the exact manual URLs, and per-dataset
  provenance/attribution.
- The WPS_GEOG mirror lives at
  `huggingface.co/datasets/deepguess/wps-geog-arwen`.
- New `rw_fetch`, a Rust download backbone built from the vendored
  `tools/rustwx` workspace and driven by `gpuwm fetch --engine rust`
  (HRRR today; `auto` uses it when built, the Python transport stays
  the always-available fallback). It brings 16 MiB parallel range GETs,
  `.idx` range coalescing, a cross-process NOMADS rate governor
  (2.5 s minimum request gap plus a node-wide cooldown, shared by every
  process on the machine), and a URL+range disk cache (`--cache-dir`).
  `gpuwm doctor` probes it, including an exact fetch-record ABI check
  so a stale binary fails before a download rather than after one.
- New `gpuwm fetch --mode auto|full-file|idx-subset`: the byte
  transport, chosen by **probe** rather than by any time constant. The
  last indexed message's declared length is read from its own header
  and one byte past it is requested; an index that provably ends where
  the object ends earns a range subset, and an index that is absent,
  malformed, short, or unprovable gets the whole file. `idx-subset`
  refuses rather than silently degrading. Full-file HRRR hours feed
  `hrrr_grib2_bridge` unchanged -- it selects by field identity, not by
  file size.
- Record-count bars are now **derived from the live provider
  inventory**, with the certified counts (124 GFS, 561 + 18 HRRR) kept
  as a tripwire: agreement is silent, disagreement names both numbers
  and refuses until `--accept-inventory-change` makes the live count
  the bar and records the acceptance in the fetch manifest. Never
  silently adapt, never mystery-break.
- NOMADS and AWS do **not** publish identical HRRR `.idx` field names:
  NOMADS says `CLWMR` where S3 says `CLMR`. gpuwm treats that as an
  alias (one role, either spelling) instead of hard-failing
  `--transport auto`, and an inventory a host publishes that gpuwm
  genuinely does not recognise now falls back to the other host with an
  explanation rather than ending the run. DATA.md's byte-identical
  claim is corrected: the GRIB files are identical, the indexes are
  not.
- `gfs_grib2_bridge` accepts both published row orders -- the
  grib-filter crop's south-to-north `0x40` and the raw archive's
  north-to-south `0x00` -- and normalizes to one on decode, with the
  gate receipt recording `source_scan_mode` and whether a flip
  happened. Proved bit-identical (no tolerance) against a committed
  matched pair in `tests/fixtures/gfs-scan-order/`. Every other scan
  mode stays fail-closed, and so does GRIB2 template 5.3, which is what
  actually still blocks the raw S3 archive.
- New `gpuwm fetch --source gdas`, **scoped to analysis init**: the GFS
  assimilation cycle through the certified GFS container -- same grid,
  codes, 124-record census, centre and tables -- so it reuses the
  certified mapping, bridge and front door with a source tag. The
  accepted window is `--hours 0`, the f000 analysis, because that is
  what has been verified end to end: NCEP tags GDAS forecast hours with
  a different forecast generating process than the analysis, and the
  fail-closed `gfs_grib2_bridge` is certified against the analysis tag.
  A request past f000 -- on the CLI or in a config `[fetch]` table --
  refuses up front, says why, and points at `--source gfs` (certified
  through f384) rather than downloading files the bridge would reject.
  f000 is an analysis-quality initial state at identical cost and
  through identical machinery; no comparative forecast-impact claim is
  made, because none has been measured.

## 1.0.0 -- first public release (2026-07-29)

ArWen: an independent, GPU-native implementation of a WRF-ARW-class
regional model. Not affiliated with or endorsed by NCAR/UCAR. Research
and educational tool; never a substitute for official warnings.

### Worldwide projections

- Mercator, polar stereographic (both poles), and southern-hemisphere
  Lambert conformal grids now run end to end: `gpuwm domain`
  auto-selects the projection from the point latitude (below 25
  degrees absolute Mercator, 25-60 hemisphere-correct Lambert, above
  60 polar stereographic; `--projection` overrides), the config schema
  gains `[projection] map_proj = "lambert" | "mercator" | "polar"`
  (the `[shared]` integer keeps the WRF convention 1=lambert, 2=polar,
  3=mercator and must agree), and namelist import/emission, the
  WPS_GEOG static build, ERA5/GFS ingest, and the native WRF direct
  export (`MAP_PROJ`/`MAP_PROJ_CHAR` derived per projection) all
  follow.
- Antimeridian-crossing domains are supported: `gpuwm fetch --area`
  reads a longitude pair spanning more than 180 degrees as the
  complementary box crossing 180E, `--point` boxes wrap across the
  seam, and GFS antimeridian crops decode onto a continuous axis.
- New wizard ladder `12` emits a single 12 km domain
  (`restart_interval_s = 0`, the portable prepared-forecast contract).
- `gpuwm check` on a config without `[case_data]` (the GFS/HRRR wizard
  emissions) now prints that the input preflight is not applicable and
  certifies the memory preflight (exit 0) instead of refusing.
- Maturity, stated plainly: the new projections (Mercator, polar
  stereographic, southern-hemisphere Lambert) are oracle-verified and
  smoke-run verified -- transcription gates at binary64 against a
  Fortran oracle built from the pinned WRF v4.6.1
  `share/module_llxy.F` (tools/llxy_wrf461_oracle, fixtures in
  tests/data/llxy_oracle, gates in tests/test_projection_oracle.py
  with measured max-ULP ceilings) plus short GPU smoke integrations --
  NOT matched-run verified. The detailed matched-run verification (the 1974
  reference family, geo_em byte-level gates) remains
  northern-hemisphere Lambert only.
- Genuine limits that remain: domains containing or touching a pole
  are refused; forcing footprints wider than 180 degrees of longitude
  are refused; latitude-longitude (cylindrical) and rotated grids fail
  closed; HRRR remains CONUS Lambert only (worldwide points use GFS or
  ERA5, both global).

### The model

- WRF-ARW-class compressible nonhydrostatic core (RK3, split-explicit
  acoustics) in FP32 on CUDA; one-way static Lambert-conformal nests
  with WRF-recurrent boundary-clock semantics.
- Physics transcribed from WRF v4.6.1 (`d66e442f`) with per-option
  maturity labels (docs/public/PHYSICS.md): Kessler, WSM6, Thompson
  (wrf-matched-run; WRF tables SHA-256-pinned -- the two largest are
  published as release assets and staged by `gpuwm fetch-tables`,
  which install runs automatically), Morrison
  2-moment, NSSL 2-moment (wrf-matched-run-candidate) microphysics; YSU and
  MYNN PBL; MM5 and MYNN surface layers; Noah, Noah-MP, and RUC land
  surface; RTE+RRTMGP (default) and legacy-RRTMG (verification tier)
  radiation; Kain-Fritsch cumulus.
- Verification against WRF v4.6.1 at three levels -- component ULP
  oracles, a t=0 full-state digest, matched 6 h four-domain forecast
  to 500 m -- with published decay tables and explicit non-claims
  (docs/public/VERIFICATION.md). Measured: 6 h on a 250x200x49 domain
  with full physics in 3.6 min on one RTX 5090.

### The product surface (all new in this release)

- `gpuwm fetch` -- GFS (NOMADS subsets), HRRR (AWS byte-range), and
  ERA5 (CDS request templates + validation); resumable, manifested,
  refuses changed requests instead of silently reusing files.
- `gpuwm domain` -- point + GPU tier to a sized experiment TOML via
  the real VRAM estimator (16/24/32 GiB tiers, measured 1.75 peak
  envelope; docs/public/HARDWARE.md).
- `gpuwm check` -- input preflight (decode envelopes, coverage, geog
  tiles, table hashes) plus itemized VRAM preflight with `--alloc`
  device verification.
- `gpuwm run` / `resume` -- supervised forecasts, atomic
  `run-progress.json`, failure capsules, restart checkpoints with
  fail-closed identity checks.
- `gpuwm render` -- composite reflectivity, T2, 10 m wind, and
  accumulated precipitation PNGs via the `wrf-rust` package.
- `UP_HELI_MAX` -- WRF's 2-5 km updraft-helicity running-max diagnostic
  (`nwp_diagnostics = 1`; wizard configs enable it), oracle-pinned to
  WRF v4.6.1 `cal_helicity` at max ULP 0, trajectory-inert by test,
  restart-carried, reset each history frame; unlocks the renderer's UH
  product family.
- `gpuwm downscale` -- offline finer-nest re-runs from archived ArWen
  or stock-WRF history (ndown-class), explicit boundary-cadence
  contract, measured cadence-cost table (docs/public/DOWNSCALE.md).
- `gpuwm import-namelist` -- WRF namelist pair to experiment TOML with
  a structured substitution report and a one-sweep missing-key census.
- `gpuwm doctor` -- whole-estate diagnosis (CuPy, bridges, tables,
  data roots) with copy-pasteable remedies.
- `rw-wps` -- the native preprocessor: HRRR/GFS/ERA5/mapped sources to
  `wrfinput_d0N`/`wrfbdy_d01` directly (no WPS, no `real.exe`);
  unchanged stock WRF v4.6.1 has accepted and integrated its outputs,
  serial and MPI, nests through d06, within documented boundaries
  (docs/public/WRF-INTEROP.md).

### Packaging and platforms

- Git-checkout install (Windows and POSIX) with one vendored, locked,
  offline Rust build; `[gpu]` and `[render]` extras; the pip wheel
  documented honestly as needing the same bridge build.
- `gpuwm fetch-tables` -- stages the externalized table assets (the
  two largest Thompson tables: freezeH2O.dat, 243 MiB, over GitHub's
  blob limit and PyPI's per-file cap; qr_acr_qg_V4.dat, 71 MiB,
  excluded from the wheel/sdist for the same cap but still in the
  repository) from the version-pinned release assets, verified against
  the packaged SHA-256 pins before an atomic install; refuses
  mismatched bytes and never overwrites an existing file.  The install
  scripts run it automatically; `--from DIR` covers offline installs;
  `gpuwm doctor` prints it as the exact remedy while a table is
  missing.
- Sealed Linux runtime archive and CPU-only Windows x86-64 archive
  with deterministic, hash-manifested builders.
- Uniform CLI refusal contract: documented refusals exit 2 with a
  one-line message, never a traceback.

### Known limits (stated in full in README and VERIFICATION)

Lambert conformal, Mercator, and polar stereographic projections
(pole-containing domains, footprints wider than 180 degrees of
longitude, and lat-lon/rotated grids refused; non-Lambert and
southern-hemisphere grids at oracle + smoke maturity, not matched-run
verified); one-way static nests; FP32; no data assimilation; ERA5
drives the config-driven GPU loop (GFS/HRRR feed the native
preprocessor front door; HRRR is CONUS-only); one case deeply
compared with WRF, component evidence elsewhere.

## Pre-1.0 development

Pre-release development history, internal milestone evidence, and
per-change hashes are recorded in PROVENANCE.md and the focused status
documents rather than duplicated here.
