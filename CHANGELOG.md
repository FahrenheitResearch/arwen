# Changelog

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
- Plot subtitles gain the grid spacing the file declares -- `Δx 3 km`
  at and above a kilometre, `Δx 111 m` below it -- on both engines and
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
  NOT matched-run verified. The deep matched-run validation (the 1974
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
  (model-validated; WRF tables SHA-256-pinned -- the two largest are
  published as release assets and staged by `gpuwm fetch-tables`,
  which install runs automatically), Morrison
  2-moment, NSSL 2-moment (validation-candidate) microphysics; YSU and
  MYNN PBL; MM5 and MYNN surface layers; Noah, Noah-MP, and RUC land
  surface; RTE+RRTMGP (default) and legacy-RRTMG (verification tier)
  radiation; Kain-Fritsch cumulus.
- Verification against WRF v4.6.1 at three levels -- component ULP
  oracles, t=0 initialization parity, matched 6 h four-domain forecast
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
validated, component evidence elsewhere.

## Pre-1.0 development

Pre-release development history, internal milestone evidence, and
per-change hashes are recorded in PROVENANCE.md and the focused status
documents rather than duplicated here.
