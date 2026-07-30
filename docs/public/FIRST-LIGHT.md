# First light

This is the full path from a bare machine to rendered forecast
products, with real wall-clock timings. Every number on this page was
measured in a first-time-user acceptance transcript on 2026-07-29:
RTX 5090 (32 GiB), Windows 11, Python 3.13, a warm pip cache, and no
prior ArWen state. Your times will vary with network, disk, and card;
the shape of the path will not.

> ArWen is a research and educational tool, never a substitute for
> official warnings from your national meteorological service.

## 0. What you need

- Python 3.11+, a Rust toolchain (`cargo`), git.
- For the GPU forecast loop: an NVIDIA card with CUDA 12.x/13.x
  (tested through 13.0) and the
  `[gpu]` extra (CuPy). 8 GiB VRAM is enough for a first single-domain
  run; nest ladders are sized to your card in step 2.
- Disk: budget several GB. In the acceptance transcript the whole tree
  (venvs, data, outputs) reached 7.6 GB, dominated by hourly output
  frames at ~198 MB each.
- Static geography: the WPS_GEOG tree, staged by `gpuwm fetch-geog`
  (~1.3 GB one-time download, ~16 GB unpacked; see
  [DATA.md](DATA.md#static-geography-wps_geog)).

## 1. Install (measured: under 2 minutes with a warm cache)

| step | measured wall |
|---|---|
| `git clone` (local) | 3.0 s |
| `python -m venv` + `pip install -e '.[gpu,render]'` | ~25 s cached; a fresh machine downloads ~150 MB (numpy, matplotlib, netCDF4, CuPy) |
| `gpuwm fetch-tables` (externalized Thompson tables, a one-time ~243 MiB release-asset download from a checkout, SHA-256 verified; a no-op once staged) | connection-speed bound; instant when already present |
| `cargo build --release --locked --offline` in `tools/grib1_bridge` | ~8 s (vendored workspace, no network) |
| `cargo build --release --locked --offline` in `tools/rustwx` (the production render engine; `--no-render` skips it) | 67 s from clean (measured 2026-07-29, same box; vendored workspace, no network) |
| `gpuwm doctor` | seconds |

One command does all of it -- `bash install.sh` (POSIX; the universal
form, mode-bit independent) or `.\install.ps1`
(PowerShell) from the checkout root: venv, `[gpu,render]` extras, the
offline Rust builds (the `tools/grib1_bridge` GRIB bridges and the
`tools/rustwx` render engine; `--no-render` / `-NoRender` skips the
renderer, the long pole of install), and a closing `gpuwm doctor`; it
offers rustup if `cargo` is missing and is safe to re-run. The
equivalent manual steps:

POSIX:

```bash
git clone https://github.com/FahrenheitResearch/arwen gpuwm && cd gpuwm
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[gpu,render]'
gpuwm fetch-tables
gpuwm fetch-geog       # WPS_GEOG static tree: ~1.3 GB down, ~16 GB unpacked
(cd tools/grib1_bridge && cargo build --release --locked --offline)
(cd tools/rustwx && cargo build --release --locked --offline)
gpuwm doctor
```

Windows (PowerShell):

```powershell
git clone https://github.com/FahrenheitResearch/arwen gpuwm; cd gpuwm
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[gpu,render]'
gpuwm fetch-tables
gpuwm fetch-geog       # WPS_GEOG static tree: ~1.3 GB down, ~16 GB unpacked
cd tools\grib1_bridge; cargo build --release --locked --offline; cd ..\..
cd tools\rustwx; cargo build --release --locked --offline; cd ..\..
gpuwm doctor
```

`gpuwm doctor` checks CuPy, the render extra, the rust render engine,
all five Rust bridge executables, the CPU library, the packaged physics
tables, and your data-root layout, and prints the exact command that
fixes anything missing. Run it until it is clean; everything downstream
assumes it is. (The `tools/rustwx` build is skippable: `gpuwm render`
falls back to matplotlib without it, and doctor labels that state
`info`, not a gap.)

## 2. Size a domain to your card (measured: 1.7 s)

```bash
gpuwm domain --point 35.3,-97.5 --card 24gb --cycle 1999-05-03T12 \
  --hours 6 --out configs/myarea.toml
```

The wizard centers a nest ladder on your point and bisects the grid
sizes through the real VRAM estimator until the projected machine peak
fits your card's budget. Output on the 24 GB tier (Windows):

```
  domain    dx        mass grid      dt         resident
  d01     12.000 km   170 x 136        60 s     0.65 GiB
  d02      3.000 km   336 x 272        15 s     2.31 GiB
  d03      1.000 km   360 x 294         5 s     2.69 GiB
  d04      0.500 km   288 x 236       5/2 s     1.74 GiB
  itemized alloc estimate 7.17 GiB; footprint projection 11.29 GiB
    x 1.75 observed peak envelope = 19.75 GiB
    envelope factor: windows (measured, 1 WDDM run)
  budget 20.00 GiB (24 GiB card - 4 GiB reserve); headroom 0.25 GiB
```

The same command on Linux prints `x 1.45 observed peak envelope` over a
projection that drops two Windows-only pool constants, and sizes a much
larger grid -- roughly one card tier's worth. Three instrumented Linux
runs measured the true peak at 1.15-1.32x the itemized alloc estimate,
against a Windows model that predicted 2.1x it
([HARDWARE.md](HARDWARE.md)). `--card 12gb` is a Linux tier for exactly
this reason.

It emits the experiment TOML, a matching `namelist.wps`, and prints the
exact `gpuwm fetch` command for the data it needs. `--ladder` picks the
depth (`12`, `12-3`, `12-3-1`, `12-3-1-0.5`, or `auto`; the
single-domain `12` ladder emits `restart_interval_s = 0`, the portable
prepared-forecast contract); `--vram-gib N` covers cards between the
named tiers (`--card 12gb|16gb|24gb|32gb`). How the sizing model works and where each platform's
envelope factor comes from: [HARDWARE.md](HARDWARE.md).

The presets are shortcuts, not the whole product. `--root-dx KM` and
`--chain R1,R2,...` build any ladder from an arbitrary root spacing and
a chain of integer refinement ratios, sized by the same estimator fit
loop and validated by the same `gpuwm check`:

```bash
gpuwm domain --point=35.3,-97.5 --card 24gb \
    --root-dx 3 --chain 4 ...          # 3 km -> 750 m
gpuwm domain --point=35.3,-97.5 --card 32gb \
    --root-dx 3 --chain 3,3,3 ...      # 3 km -> 1 km -> 333 m -> 111 m
```

Root `dt` follows the same convention at any spacing (5 s per km, 2.5
in the tropics) and is carried exactly, including half seconds, through
WRF's rational clock keys. When any domain lands below 1 km with a 1-D
PBL scheme active the wizard prints a **gray-zone advisory** -- not a
refusal -- into both the file and stdout: below that spacing the largest
boundary-layer eddies are partly resolved by the dynamics while the PBL
scheme parameterizes them as if they were not, and the proper tool at
those scales is a 3-D turbulence closure (SASE, planned).

The wizard works worldwide: it picks the projection from your point's
latitude -- Mercator below 25 degrees, hemisphere-correct Lambert
conformal from 25 to 60, polar stereographic above 60, in either
hemisphere -- and `--projection` overrides the choice.
Antimeridian-crossing footprints are handled. It still refuses what
the pipeline cannot stand behind: domains containing or touching a
pole, and forcing footprints wider than 180 degrees of longitude. The
new projections (Mercator, polar stereographic, southern-hemisphere
Lambert) are oracle-verified and smoke-run verified, not matched-run
verified ([VERIFICATION.md](VERIFICATION.md)).

## 3. Get data

Two routes, honestly distinguished (details and disk sizing:
[DATA.md](DATA.md)):

- **ERA5 (the GPU forecast route).** `gpuwm fetch --source era5` emits
  the exact two-part CDS request template plus instructions; you
  retrieve with your own Copernicus account, then validate with
  `gpuwm fetch --source era5 --validate FILE...` (seconds, catches a
  wrong retrieval before anything expensive).
- **GFS / HRRR (the native preprocessor route, measured: 9.7 s).**

```bash
gpuwm fetch --source gfs --cycle latest --hours 6 \
  --point 35.2,-97.4 --radius-km 350 --out data/gfs-latest
```

resolved the newest complete cycle, downloaded three ~127 KB subset
files, and wrote a SHA-256 manifest and the series inventory in 9.7 s.
Every fetch is resumable; re-running with a different area or cycle
into the same directory refuses with the exact difference rather than
silently keeping the old files.

GFS/HRRR data feeds `rw-wps`, the native initialization front door
(measured: 44.7 s for a two-domain d01+d02 build -- 30-arcsec static
fields, decode, and initialization on the deterministic Rust CPU
backend), which emits `wrfinput_d01..dNN` + `wrfbdy_d01`. Those files
drive unchanged stock WRF ([WRF-INTEROP.md](WRF-INTEROP.md)) and the
downscale tool. The config-driven `gpuwm run` loop in section 5 below
runs from ERA5 only -- **GFS and HRRR reach the GPU through a different
runner, documented in section 3a.**

## 3a. GFS -> GPU forecast: the complete route

`gpuwm run` refuses a GFS config by design (no `[case_data]` table).
The GPU forecast for a prepared GFS product is launched by a runner
under `tools/`, and **the order below matters**: the runner binds the
experiment config into the prepared cache, so materializing the physics
*after* preprocessing means preprocessing again.

Three first-time-user pilots ran this route on rented Linux 4090s and
4070s on 2026-07-30; the ordering, the flags, and the timings are
theirs.

```bash
# 1. Size the domain.  --physics-profile emits a config the runner
#    accepts as written; without it you get the product default suite,
#    which the SINGLE-domain runner refuses (the wizard says so, and
#    lists what every profile actually runs).
gpuwm domain --point=35.3,-97.5 --card 24gb --ladder 12     --source gfs --cycle latest --hours 6     --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1     --out configs/myarea.toml

# 2. Materialize the exact physics authority.  BEFORE rw-wps, not after.
python tools/prepared_single_domain_forecast.py --materialize-authorities     --source gfs     --base-experiment-config configs/myarea.toml     --base-wps-namelist configs/myarea.namelist.wps     --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1     --output-directory work/myarea-authority

# 3. Fetch, then author the front-door manifest.  Step 3 prints the
#    complete rw-wps command with its digest already filled in.
gpuwm fetch --source gfs --cycle <RESOLVED> --hours 6     --area=<THE BOX THE WIZARD PRINTED> --out data/myarea
gpuwm fetch --source gfs --author-front-door-manifest --out data/myarea     --bridge tools/grib1_bridge/target/release/gfs_grib2_bridge     --wps-namelist work/myarea-authority/namelist.wps     --experiment-config work/myarea-authority/experiment.toml

# 4. Run the front door (paste the line step 3 printed, plus these).
rw-wps ... --geog-root $GPUWM_CASE_DATA_ROOT/WPS_GEOG     --output-root out/myarea-init

# 5. Run the forecast.  rw-wps finishes by printing THIS command with
#    all three digests filled in -- copy it rather than retyping.
python tools/prepared_single_domain_forecast.py     --source gfs --prepared-root out/myarea-init     --proof-sha256 <printed> --source-manifest-sha256 <printed>     --prepared-content-sha256 <printed>     --experiment-config work/myarea-authority/experiment.toml     --wps-namelist work/myarea-authority/namelist.wps     --physics-profile morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1     --io-mode history --outdir out/myarea-run

# 6. Render.
gpuwm render out/myarea-run/wrfout/<frame> --out out/myarea-png
```

Measured on a Linux RTX 4070 12 GB (node 3, GFS 2026-07-29 18Z, single
domain 342x272x49 at 12 km):

| stage | wall |
|---|---:|
| `gpuwm fetch` (3 files, 42.2 MB) | 50.3 s |
| `--materialize-authorities` | < 1 s |
| `rw-wps` (30-arcsec static + decode + init, 48 CPU cores) | 2 m 52 s |
| GPU forecast, 6 simulated hours, 2160 steps at dt 10 s | **523 s** |
| `gpuwm render` (4 products, 1 frame) | 21.8 s |

**Multi-domain** products go to `tools/prepared_domain_tree_forecast.py`
instead, which takes `--prepared-root` + `--preparation-receipt-sha256`
and has no physics-profile whitelist -- it runs the wizard's default
suite as written. `rw-wps` names whichever runner applies to the proof
it just wrote, with the digests filled in.

**Watch progress** at `<outdir>/evidence/progress.json` (domain tree) or
`<outdir>/progress.json` (single domain), not the `run-progress.json`
that section 5's config-driven route writes.

## 4. Preflight (measured: 9.2 s)

```bash
gpuwm check configs/myarea.toml
```

Two gates in one command: the input preflight (in the transcript, 19
checks -- real GRIB decode envelopes, level/temporal/spatial coverage,
geog tile coverage and hashes, physics table hashes) and the itemized
VRAM preflight (predicted 12.78 GiB peak envelope against a 27.25 GiB
measured budget in the transcript). `check` failing is the tool working:
it names the missing input or the memory shortfall and the remedy
before you spend GPU time.

## 5. Run (measured: 6 h forecast in 3.6 min)

```bash
gpuwm run configs/myarea.toml --outdir out/myarea
```

The acceptance run: 6 simulated hours on a 250x200x49 12-km domain
with Morrison + RTE+RRTMGP + YSU + Noah + Kain-Fritsch at dt = 60 s
(the wizard default at the time of the transcript; it now emits
Thompson mp8 in the microphysics slot, Morrison stays selectable) --
361 steps in ~200 s of wall time (~0.55 s/step including output),
completed on the first attempt, ~6.3 GiB of device memory above the
desktop baseline, seven hourly wrfout frames of ~198 MB each.

While it runs:

- **Watch `run-progress.json`** in `--outdir` (schema
  `gpuwm.run-progress/v1`): status, model-elapsed seconds, outer step,
  newest durable output frame, newest checkpoint. It is rewritten
  atomically through the run. Do not watch a redirected stdout -- it is
  block-buffered and can stay empty until exit while the run is
  healthy. This filename belongs to the config-driven `gpuwm run`
  route only; the `tools/` runners of section 3 write
  `<outdir>/evidence/progress.json` (domain tree) or
  `<outdir>/progress.json` (single domain).
- On failure the supervisor writes `failure-capsule.json` beside it
  with the exception, the step, and the state needed to report or
  resume.

## 6. Render (measured: 2.6 s for 16 PNGs)

```bash
gpuwm render out/myarea/wrfout_d01_* --out out/myarea/png
```

With the `tools/rustwx` build from step 1, this runs the production
Rusty Weather engine by default.  Its vendored catalog carries 324
entries; 151 of them are implicit-render candidates the runtime
lister evaluates against every file (the rest are explicit-opt-in
ensemble/probabilistic families): reflectivity composite and 1 km,
the 2 m temperature/dewpoint/RH families with 10 m wind variants,
MSLP + winds, PWAT, cloud cover, the 200/250/300/500/700/850 mb chart
families (height/temperature/dewpoint/RH/absolute vorticity with
winds), SB/ML/MU CAPE and CIN, SRH, bulk shear, STP, the heavy ECAPE
family (`--heavy`), and multi-hour windowed accumulations (run-total
QPF, wind/UH maxima, ...) on whole-hour multi-frame runs.  Whatever
your output fields prove out renders -- measured on the committed
3 km UH-smoke case: 58/58 on a single frame and 238 renders with 0
failures across its four-frame store (receipt transcripts retained in
the development tree under `evidence/render-receipts/`) -- and
`gpuwm render --list-products FILE` prints the whole catalog with the
per-file verdict and the exact field-level reason anything is
unavailable.
Charts draw coast/state/county basemaps, and sub-hourly output
cadences are stamped exactly (`valid_..._lead_003h30m00s`) in filename
and subtitle.  The first line of output names the engine in use.

Every filename carries the domain and its resolution
(`..._d02-3km_composite_reflectivity_...`; sub-kilometre nests read as
`_d05-111m_`), so several nests of one run render into one directory
without colliding, and the plot subtitle carries the same spacing as
`Δx 3 km`.  Plots are labelled **ArWen**; pass `--source-label` when
rendering wrfout files this model did not produce, so the sheet does
not claim them.

Without that build, the matplotlib fallback renders four products per
frame -- composite reflectivity (NWS color scale), 2 m temperature,
10 m wind speed and barbs, accumulated precipitation -- via the
`wrf-rust` package (`pip install 'gpuwm[render]'` if you skipped the
extra; the error message names it). In the transcript, 4 files x 4
products took 2.6 s.

To compare two runs product-by-product (a rerun, a physics variant, a
CPU WRF twin), render each into its own directory and compose labeled
side-by-side sheets:

```bash
gpuwm render --pair out/runA/png out/runB/png --out out/compare
```

## 7. Checkpoint and resume (measured: 65.6 s + 46.1 s)

```bash
# a 2 h leg writing checkpoints every simulated hour
gpuwm run configs/myarea.toml --outdir out/myarea    # restart_interval_s = 3600
# ... later, extend run_seconds in the config, then:
gpuwm resume configs/myarea.toml --outdir out/myarea
```

In the transcript the 2 h leg took 65.6 s and wrote 2 checkpoints;
after extending the config to 3 h, `gpuwm resume` found the newest
valid checkpoint set, verified its identity against the config
(fingerprints, physics identity, boundary-clock semantics -- mismatches
refuse loudly), and completed the third hour in 46.1 s. Torn or
truncated checkpoint sets are skipped with printed reasons.

## 8. Where to go next

- Downscale an archived run to a finer nest:
  [DOWNSCALE.md](DOWNSCALE.md) (write parent history at 15-minute
  cadence if you plan to).
- Drive unchanged stock WRF from the same preprocessor:
  [WRF-INTEROP.md](WRF-INTEROP.md).
- Import an existing WRF namelist pair:
  `gpuwm import-namelist namelist.wps namelist.input` -- emits an
  experiment TOML plus an explicit substitution report of every option
  it mapped or refused.
- Understand what the model is verified against before you trust a
  picture: [VERIFICATION.md](VERIFICATION.md).
