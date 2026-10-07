Public recent convection case

The active case is 241 x 241 x 49 mass points at 3 km, centered at 31 N, 89 W. Its mass-point span is 720 km on each axis. It starts from the 2026-10-03 21Z native HRRR analysis. Native RAP from that same 21Z cycle supplies every hourly boundary through 09Z. The independent 00Z HRRR analysis is fetched as a reference and is never inserted into the 21Z initial state. Six hourly analyses finish at 03Z, followed by a six-hour free forecast to 09Z. Set `--da-hours` to 3, 4 or 5 for an earlier issuance with a matching six-hour horizon.

Author the case on the execution box. The saved folders here contain actual CPU-authored geometry and profile authorities. Their data status is pending. Each `case-plan.json` and `authority/authority-receipt.json`, and the two pricing receipts, record the author machine's absolute paths, so the public release tree leaves them out (RELEASE-EXCLUDE.txt); a development checkout keeps them. The box author command creates local paths and fresh authorities, including all of those files. It creates no invented data hashes.

```sh
python -m tools.da_recent_case --out /workspace/shared-prepared/da-recent-286/case --nx 241 --ny 241 --latitude 31 --longitude=-89 --da-hours 6 --shared-root /workspace/shared-prepared/da-recent-286 --geog-root /workspace/shared-prepared/WPS_GEOG
```

The public profile is `thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1`, with Thompson MP28, nine-layer RUC, MYNN gsd_41, legacy RRTMG, `moist_cq=true`, `aer_opt=3`, `swint_opt=1`, and native analyzed aerosols. `base/` plus `authority/` record the complete named-profile materialization. Native RAP's named source profile supplies its decoder and mapping authorities.

Use `observation-case.json` and `model-object-requests.json` with the shipped `tools.da_recent_observations inventory` command. The one fetch controller obtains the frozen model, radar and MRMS inventory. Execute the exact `prepare_argv` list in `case-plan.json` after fetching. Every RAP knot has an explicit `rap_native_in_band_surface` binding. The RAP mapped source door seals its prepared cache by default. It refuses the HRRR-only `--sealed-prepared-cache` and `--physics-profile` CLI options; the preparation uses the materialized authority pair instead.

Both baseline public GEOG datasets and the pinned HRRR static source are required in the one shared geography root. Reuse the shared kit's baseline datasets. The named static source is fetched by:

```sh
python -m gpuwm fetch-geog --static-source hrrr-conus-v4 --root /workspace/shared-prepared/WPS_GEOG
```

Prepare native observations against the actual `prepared/wrf-native-input/wrfinput_d01`. This is the full 3D native georeference, including PH/PHB. The finalizer refuses a replacement grid. The availability-audited radar roster is frozen in the observation case; two usable causal native radar volumes are required per slot. Surface rows come from the native METAR seam, with native screening. Publication timing remains a recorded historical replay policy.

Acquisition `bbox` is the native full-cell corner footprint, including the outer half-cells accepted by the observation operator. `model_mass_bbox` separately records the mass-point center extrema. The edge regression includes the actual PIB station at 31.4671 N, 89.3371 W, which the former center-only bbox missed.

Truth acquisition has its own frozen extent. `truth_bbox` comes from the full-cell corners of a larger virtual grid on the model's exact native lattice, without expanding the model or its donor inputs. The native `static-fields` translated-grid and latitude/longitude/MAPFAC operators enlarge the integer rim until the ground-length bound exceeds 75 km. For the active case the rim is 27 cells, 81 km projected, with a 79.360 km ground-length bound. The independent truth bbox is `[-93.7889407927, 26.6538182469, -83.7585322620, 35.2250863284]` in W,S,E,N order. Native grid definition, translation, map-factor bound and binary hash are frozen in `truth_padding_provenance`. The MRMS adapter uses this truth extent, and refuses an observed object clipped at its edge or outside forecast coverage instead of trimming a footprint.

Bind actual prepared and observation bytes after preparation:

```sh
python -m tools.da_recent_case finalize --plan /workspace/shared-prepared/da-recent-286/case/case-plan.json --prepared-root /workspace/shared-prepared/da-recent-286/prepared --slots /workspace/shared-prepared/da-recent-286/observations/slots.json --fetch-receipt /workspace/shared-prepared/da-recent-286/fetch-receipt.json --engine-sha ACTUAL_40_DIGIT_EXPORT_SHA --run-out /workspace/da-recent-286/results --manifest /workspace/shared-prepared/da-recent-286/case/run-manifest.json
```

This CPU step hashes the actual proof, portable source manifest, prepared header, grid, radar and surface bytes. It calls the public full-cache forecast preflight and native radar-grid reader, checks all absolute clocks and the surface seam, and records the actual independent HRRR donor. Raw download files may already be retired after preparation; the complete fetch receipt stays. Remaining raw files are rehashed. `tools.da_recent_run` consumes the resulting run manifest and runs both the packed DA ensemble and matched clean-start ensemble.

The small integration smoke uses `--smoke --nx 33 --ny 33 --smoke-run-hours 2`. It runs four members, one hourly observed leg, then 120 seconds free. Its observation request still declares six requested hourly slots, and inventory uses `--first-slot-only`. The smoke omits the unused independent 00Z reference. It does not qualify rain skill.

The actual Linux CPU price for the active 241 point case is 10.6889399486 GiB per member, using the public conservative absent RTX 5090 profile. Two members require 21.3778798971 GiB within the 28 GiB shared-card budget. Thirty-two members run in two 16-member waves across eight cards. Four members do not fit this estimate. MPS is required only when four members fit. `native-linux-pricing-241.json` is the actual Linux receipt, with `gpu_started=false`; this is an admission estimate, not a measured GPU peak. FFT plan workspace was not measured, and actual observation analysis/device/host headroom must still pass live admission. The host array floors for 32 members plus one control mirror are 8.09 GiB for serialized mirrors, 9.51 GiB for float64 analysis priors, and 0.75 GiB for H(Z) and surface planes. These are array floors, not worker RSS upper bounds. Decode processes need the deck's thread and RAM caps.

`requested-601-center31` records the literal 601 point domain. Native HRRR coverage refuses its required source window `i=869..1473,j=-39..565`, because valid source rows begin at zero. `large-601-center32p2` moves the center north to 32.2 N while retaining the 31 N, 89 W focus inside. Its native donor coverage passes, but its 26.9084515683 GiB per-member estimate refuses two members per card. No extrapolation, filled donor region, or packed-fit claim is made for either large scenario. Actual Linux receipts for 201, 301, 401 and 601 points are in `native-linux-pricing.json`.

The author can parameterize another recent cycle, grid, explicit `--radar-sites` roster and `--surface-networks KS_ASOS OK_ASOS` list. With no `--radar-sites` the roster is every WSR-88D whose 250 km reach touches the domain; with no `--surface-networks` the networks are every ASOS network in the frozen table whose extent meets the domain box plus 0.5 deg, and a domain none reaches is refused by name. The surface network authority records whether the caller supplied the list (`caller`) or the domain box chose it (`domain-bbox`). Every new case must still pass native source coverage, anonymous object inventory, exact valid-time composition, public preparation, causal observation checks and runtime admission.
