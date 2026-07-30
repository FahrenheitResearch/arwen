# Provenance of the vendored Rusty Weather renderer

## Source

Everything under `crates/` and `vendor/` (except `crates/rw-wrfbatch`
and `vendor/crates-io`, below) is Drew's own **rusty-weather**
workspace (MIT -- `LICENSE` in this directory is its license file),
vendored from the RW-Studio worktree at

    rust-weather-rwwps-studio @ commit 31b681e
    ("Preflight HRRR domains and isolate Python caches")
    PLUS its uncommitted working-tree state

The working-tree state is deliberate, not sloppy: the campaign's paired
CPU-vs-GPU product sheets were rendered by `rw_wrf_batch.exe` (sha256
`aa220a5a...`) built from exactly that tree, whose uncommitted diff is
the exact-time (sub-hourly) rendering support -- `RwsExactTime`
threaded through `StoreFieldSource`, `hour_presentation` valid/lead
stamps, and the removal of the blanket exact-time refusals from
`store_render::open` and `stored_run_hours`.

Vendored subset (the transitive closure `rw-wrfbatch` needs, nothing
more):

- `crates/`: rustwx-core, rustwx-io, rustwx-models, rustwx-products,
  rustwx-render, rustwx-contour, rustwx-calc, rw-store, rw-ingest,
  rusty-weather (library only)
- `vendor/`: grib-core, netcrust (with its vendored hdf5-reader),
  wx-core, wx-field, wx-math, wx-radar, metrust, ecape-rs
- `assets/basemap/`: Natural Earth 10m/110m (public domain) and US
  Census counties (public domain); `upstream-lock.json` is the source
  workspace's own provenance record
- Fonts: Source Sans 3 (OFL), embedded by rustwx-render
  (`crates/rustwx-render/assets/fonts/`, license alongside)

NOT vendored: the egui UI stack (rusty-weather-ui, rw-ui, rw-sim,
rw-sat, rw-glm, ...), rustwx-sounding/sharprs, the eleven rusty-weather
command-line bins and their 9.8 MB test fixtures.

## Deliberate divergences from the source tree

1. `crates/rusty-weather/Cargo.toml` is trimmed to the library target:
   the bins and their deps (clap, image, mimalloc, libmimalloc-sys,
   ecape-rs, rw-ingest, windows-sys) are dropped, as are the bin-only
   build.rs and `src/bin`, `src/main.rs`, `contour_mode.rs`,
   `domain.rs`, `region.rs`, `tests/`.
2. `crates/rw-wrfbatch` is new (not in the source workspace): main.rs
   is the Studio worktree's untracked
   `crates/rusty-weather-ui/src/bin/rw_wrf_batch.rs` adapted for gpuwm
   -- all stored frames render by default (`BatchHourScope::AllStored`
   instead of the minimum stored slot), `--frames N` selects the Nth
   stored frame (ordinal index, matching `gpuwm render --timeidx`), and
   the batch work limits are sized to the request instead of the GUI
   click ceilings.  The five modules beside it (grib_import,
   local_import, postproc_severe, wrf_process, wrf_volumes) are
   verbatim copies from rusty-weather-ui.
3. `crates/rusty-weather/src/render_all.rs::hour_presentation`: the
   exact-time subtitle drops zero seconds ("+003:30" not "+003:30:00")
   so the valid time survives the renderer's subtitle width limit.
4. `crates/rusty-weather/src/windowed_store.rs`: the stale test
   asserting `stored_run_hours` rejects exact-time axes (already
   contradicted by the source tree's own working-tree diff) is updated
   to the completed contract -- ordinal slots are listable, per-frame
   rendering proceeds, and fixed-hour WINDOWED accumulations alone
   remain refused on exact-time axes.
5. The workspace root `Cargo.toml` lists only the vendored subset.
6. Windowed products are axis-gated, not model-gated:
   `batch_render::inspect_renderable_products` and the "all"-keyword
   clear list/run the windowed lane for ANY model whose store has more
   than one whole-hour frame (`windowed_store::windowed_axis_ready`),
   with per-plane availability proven at compute time as ever.  The
   source tree gated the lane on `ModelId::Hrrr`.
7. `windowed_store::read_source_plane` accepts the wrfout import
   lane's names for physically identical planes: `apcp` (run-total
   APCP) behind `apcp_run_total`, `relative_humidity_2m` behind
   `rh_2m`, and `updraft_helicity_2to5km` (Registry units spelling
   `m2/s2`) as a third UH fallback.  Plane fidelity is tracked per
   source: the GRIB lane's `uh_2to5km` snapshot keeps its honest
   "lower bound" note, while `updraft_helicity_2to5km` (WRF
   UP_HELI_MAX, reset each history frame) is labeled what it is --
   the exact per-history-interval max, i.e. the exact trailing 1 h
   max at the hourly cadence the whole-hour windowed axis folds.
8. The wrfout import publishes per-level canonical selector planes:
   the sounding volumes' 200/250/300/500/700/850 hPa levels
   (height/temperature/dewpoint/u/v -- byte-identical to the volume
   levels), plus chart-level RH and absolute vorticity interpolated at
   those six levels only (same bracket/lerp walk, one extra native
   field resident at a time), plus the three cloudfrac planes under
   their canonical cloud-cover selectors.  This is what unlocks the
   isobaric chart recipe families on model output.
9. `wrf_process::wrf_product_slug` maps wrf-core's heavy ECAPE
   diagnostics onto the recipe slugs they are (same ecape-rs solver,
   parcel_type default "sb"): `ecape`->`sbecape`, `ncape`->`sbncape`,
   `ecape_cin`->`sbecin`, `ecape_ehi`->`ecape_ehi_0_1km`, and
   `ehi`->`ehi_0_1km` (SBCAPE x SRH1 / 160000, the catalog's sb
   proxy).  Deliberately NOT mapped: the lapse rates (wrf-core
   defaults to plain-T; the catalog's grids are virtual-temperature)
   and `lcl`/`lfc`/`el` (parcel identity unverified) -- they stay
   under `wrf_*` browse names.
10. `rw_wrfbatch --list-products` (new) enumerates the complete
    product catalog with per-store availability, and
    `direct/planning.rs::direct_recipe_requires_explicit_opt_in` is
    `pub` so the lister applies the exact planner filter.
11. Store-backed availability is model-identity-free end to end.
    `rustwx_models::plot_recipe_store_requirements` (new) enumerates a
    recipe's required fields/selectors without a `ModelId`;
    `direct/planning.rs::store_direct_recipe_slugs` (new) lists every
    non-opt-in direct candidate without the per-model fetch-plan
    filter.  `render_all::partition_products` (model parameter
    dropped), `batch_render::inspect_renderable_products`,
    `store_render::render_direct_recipes_from_store`, and
    `rw_wrfbatch --list-products` all decide from STORED FIELDS; the
    `gated` listing status and every "no fetch plan for model" reason
    are gone -- a non-renderable product always names its missing
    fields.  The per-model served-field tables in `rustwx-models`
    (e.g. `selector_supported_for_model`'s HRRR-only smoke and
    simulated-IR rows) remain for FETCH planning only: they state
    which operational source's downloadable files carry a field -- an
    input fact -- and no store-backed render or listing decision
    consults them.
12. The canonical catalog's windowed support rows are
    input-requirement gates: every non-QPF windowed entry derives its
    per-source support from `HrrrWindowedProduct::input_selectors`
    (new) instead of a hardcoded HRRR-only row, so sources serving the
    input fields (including the wrfout lane's `wrf` identity for 2-5
    km UH) list as supported and the rest are blocked with the missing
    input field named.  QPF windows keep curated lists (1-hour APCP
    bucket availability is a source fact an input selector cannot
    express).  The inventory fixture regenerates via
    `RW_UPDATE_PRODUCT_CATALOG_FIXTURE=1`.
13. `vendor/wx-core/src/download/cache.rs`: the raw `DiskCache` used by
    `rw_fetch --cache-dir` verifies entries AT USE and publishes them
    atomically.  The source version wrote the canonical file directly
    with `std::fs::write` and returned whatever `std::fs::read` found,
    with no key, length or checksum consulted -- so a killed writer
    left a prefix at the canonical name that the next run adopted as
    the object, and a persistently bad entry was never displaced, so
    every retry read the same poison and the outer validation failed
    forever.  Each entry now carries a `.meta` sidecar recording the
    exact cache key, byte count and an FNV-1a-64 checksum of the
    payload; `get` re-checks all three, `has` requires the sidecar, and
    a failing entry is renamed aside (never deleted) so the next
    request refetches.  The payload lands by atomic rename from a
    staging name unique per process and per call.  The checksum is an
    integrity check against truncation, partial writes and bit rot --
    not a cryptographic seal, and nothing treats it as one: upstream
    authenticity remains the caller's manifest digest and the GRIB
    envelope/record-count bars above this layer remain the completeness
    gate.  Audited as ArWen v1.1.1 finding F-7 / lying state LS-7.
14. `vendor/wx-core/src/download/client.rs`: the cross-process NOMADS
    rate governor fails CLOSED on its own failures.  `read_nomads_state`
    used to map an unreadable or malformed state file to `(0, 0)`,
    indistinguishable from "nobody has fetched yet" -- which puts
    `last_request + gap` in 1970 and waves the request straight through,
    so corrupting one small file disabled the pacing that protects a
    shared public service.  It now distinguishes an absent file (a real
    zero) from a present-but-unusable one (read as "a request just
    happened"), and `pace_request` charges that penalty exactly once
    before republishing a sound state, so an unparseable file cannot
    spin the caller forever.  `write_nomads_state` returns whether the
    bytes landed instead of discarding the result; a process whose
    record did not land absorbs the minimum gap locally rather than
    leaving the next process free to send immediately.  Audited as
    finding F-6 / lying state LS-8.  The governor's constants,
    protocol, file names and env overrides are unchanged -- ArWen's new
    `gpuwm/nomads_governor.py` speaks the same protocol over the same
    files so the Rust and Python transports pace each other.
15. `vendor/wx-core/src/download/client.rs::get_range` validates the
    ANSWER before adopting it: the status must be 206, the
    `Content-Range` must begin with the requested span, and the body
    must be exactly the requested number of bytes (the first two
    clauses only, for open-ended ranges).  The source version accepted
    whatever came back -- a 200 with the whole object, a 206 for some
    other span, a short body -- cached it and concatenated it into the
    caller's subset, leaving the outer GRIB bars to reject the assembly
    with nothing to say about which chunk was wrong while the bad bytes
    stayed cached.  These are the same three clauses ArWen's Python
    range transport has always applied.  Audited as finding F-7.
16. `crates/rustwx-io/src/cache.rs::quarantine_cache_path` no longer
    deletes the original when the quarantine rename fails.  The source
    version fell through to `fs::remove_file`, which turns quarantine
    into deletion exactly when quarantine matters -- a full disk, an
    uncreatable directory, a file held open elsewhere -- and destroys
    the evidence of what arrived in the cases hardest to reproduce.  A
    file that cannot be moved aside is now left in place and reported;
    current-schema entries are re-verified against their recorded
    request identity, length and SHA-256 at load, so it stays refused
    on every use rather than served.  Audited as finding F-8.
17. `vendor/wx-core` is version **0.3.10**, and `crates/rw-fetch` and
    `crates/rustwx-io` pin `version = "0.3.10"` beside their `path`.
    The source tree and `vendor/crates-io/wx-core` both said `0.3.9`,
    and those two crates are NOT the same code: the crates.io copy has
    no NOMADS governor at all -- no pacing, no cooldown, no state file --
    and no verified-at-use download cache.  Only the path dependency
    kept the right one in the graph, and one version string covering two
    behaviours is a trap that resolves silently.  The bump plus the
    version pin make the governorless copy unable to satisfy the
    requirement at all.  `vendor/crates-io` itself is untouched, as this
    file requires.

    The pin is belt; the braces are a capability probe.  `wx-core` now
    exports `download::nomads_governor()` (what this build's governor is
    configured to do, read from the same places the pacing reads) and
    `download::pace_nomads_request()` (the pacing entry point itself), and
    `rw-fetch`'s `the_resolved_wx_core_actually_paces_nomads` test makes
    the resolved crate DEMONSTRATE the spacing -- two paced calls against
    a scratch state file must be a configured gap apart and must leave
    node-wide state, while a non-NOMADS host must not be paced at all.
    A governorless wx-core fails to link; a wx-core whose governor
    stopped working fails the test.  Neither asserts which copy was
    resolved, which is the point.

## vendor/crates-io

`cargo vendor --locked vendor/crates-io` output: every crates.io and
git dependency in `Cargo.lock`, checked in so a clean clone builds with
`--offline` (source replacement in `.cargo/config.toml`).  The three
git dependencies (wrf-core from wrf-rust, ecape-rs, metrust-py) are all
FahrenheitResearch -- Drew's own -- repositories pinned to the exact
revisions the campaign renderer used; wrf-core rev `93bd2ca6` is the
same pure-Rust wrfout reader lineage as the pip `wrf-rust` package the
matplotlib engine uses.

Do not edit anything under `vendor/crates-io` -- each crate carries a
`.cargo-checksums.json` that cargo validates at build time.
