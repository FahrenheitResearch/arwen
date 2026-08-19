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
- `vendor/`: netcrust (with its vendored hdf5-reader), wx-core,
  wx-field, wx-math, wx-radar, metrust, ecape-rs
- NOT under `vendor/` any more: `grib-core`.  The GRIB decoder is ONE
  crate for the whole repository and lives at
  `tools/grib1_bridge/vendor/grib-core`, whose `VENDOR.md` carries its
  per-file SHA-256 delta record and its Rust 1.75 spellings.  The four
  consumers here (`rustwx-io`, `rustwx-products`, `rw-obs`,
  `rw-wrfbatch`) reach it by path with their feature postures unchanged
  -- defaults on for the first three, `default-features = false` for
  `rw-wrfbatch`.  The copy that used to sit here lacked
  `missing_value_management`, so every complex-packed message this
  workspace read decoded the encoder's missing-value markers as
  physical data.  Do not re-vendor it.
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
    The store is also CONTENT-ADDRESSED: the payload lives once, under
    `content/`, named by the same FNV-1a-64 and length the sidecar
    records, and each key entry is a hard link to it (a pointer file
    naming it, where the filesystem cannot hard-link, probed rather than
    assumed).  One full-file fetch reaches this cache under two key
    shapes -- URL-only from `get_bytes_parallel_whole`, URL+ranges from
    `get_ranges` -- and the source version stored two complete copies of
    the object at two paths.  Dedup is decided on the CONTENT and never
    on the two key shapes meaning the same thing, because a range list
    that does not cover the object legitimately yields different bytes.
    A reference is verified exactly as strictly as a payload: the bytes
    it resolves to are checked against its own sidecar, and one that
    dangles or resolves to foreign content is a miss and is set aside,
    never a wrong-bytes hit; a canonical payload that no longer hashes to
    its own name goes aside with it, so a refetch is not linked straight
    back onto the poison.  `remove` now takes the whole entry: it deleted
    the payload and left the sidecar describing a file that no longer
    existed.  `size` counts a shared payload where it lives rather than
    once per name.  `rw_fetch`'s record gained a `dedup` block reporting
    bytes written, bytes deduplicated and reference entries, and ArWen's
    HRRR fetch manifest carries the sum.
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
18. `crates/rw-fetch` and `crates/rw-wrfbatch` (the two binaries a
    gpuwm bridge bundle ships) gain a `build.rs` and a
    `GPUWM_BRIDGE_SOURCE_REV_STAMP` static: each binary embeds
    `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>` -- the workspace
    checkout's clean HEAD -- so the gpuwm release cut
    (`tools/build_bridge_bundle.py pin --source-rev`) can prove a
    staged binary was built from the commit being released by reading
    bytes, never by executing it.  The 1.4.1 cut nearly shipped
    platform zips from two different source revisions; nothing checked.
    The stamp is gpuwm release machinery, not renderer behaviour, and
    is absent upstream.

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

18. Locally imported runs name their own grid in the plot headline, and
    never a source catalog's dataset id.  The source workspace composed
    a `wrf`-identity title by appending the GDEX dataset token
    (`MSLP / 10m Winds (d612005)`), falling back to that literal when no
    product override named one.  It guarded against the fallback by
    sniffing `subtitle_right_override` for the phrase "local WRF
    NetCDF" -- which this lane's subtitle never says, because
    `rw_wrfbatch --source-label` replaces that line with the local
    model's own name.  So every isobaric and 2 m frame ArWen rendered
    carried the id of an NCAR archive it had never read.  The guard is
    now an explicit declaration rather than a string sniff:
    `rustwx_products::shared_context::TitleProvenance` rides on the
    direct, derived and windowed batch requests, defaults to
    `SourceCatalog` (fetched batches are byte-unchanged), and
    `rw_wrfbatch` declares `LocalImport { grid_label }` from the same
    `GRID_ID`/`DX` reading that already names the output files -- so the
    headline reads `MSLP / 10m Winds (d02 750 m)`.  When the file
    declares no usable grid, the headline carries no parenthetical at
    all: no parenthetical is honest where a borrowed one is not.

19. A terrain product (`terrain_height`, `RenderStyle::WeatherTerrain`).
    The catalog had no orography frame, so nothing in it showed a
    viewer the ground the forecast was standing on.  Both import lanes
    already published the plane (`HGT` -> `orography`, surface
    geopotential height), so this is a recipe, a hypsometric fill scale
    and three guard corrections: `should_render_overlay_only` and
    `viewer::operational_style_for_store_variable` excluded the whole
    `GeopotentialHeight` canonical field because the ISOBARIC member is
    contour-analysis-only, and the surface member -- a different
    physical quantity -- was caught by the same net.  Because the field
    is a property of the grid rather than of the forecast time,
    `batch_render` renders it on the first selected hour and announces
    the skip for the rest (`direct_recipe_is_time_invariant`); a stated
    single render and a silent drop must not look alike.

8. `crates/rw-nexrad` is new (not in the source workspace): the
   NEXRAD Level-II acquisition and decode front door.  `vendor/wx-radar`
   arrived with a Level-II *parser* and no downloader; the anonymous-S3
   client here is the source workspace's own `rw-sat`/`rw-glm`
   `ListObjectsV2` pattern retold for a bucket keyed by day directory
   per site.  No new external crate: every dependency was already in the
   vendor closure.
9. `vendor/wx-radar/src/level2.rs::parse_radial_block_nyquist` reads the
   Message-31 RAD block's Nyquist velocity at byte 16, not the byte 26
   the source tree used.  Byte 26 is the low half of the vertical
   calibration constant; reinterpreting it as an int*2 scaled by 100
   yields physically impossible Nyquist velocities -- one KTLX volume
   reported 620.7, 450.8 and 313.9 m/s for cuts whose real values are
   8-32 m/s.  The corrected layout was established from the bytes of a
   real volume rather than a reading of the ICD: the block's own LRTUP
   field says 28, and only one arrangement of (unambiguous range, two
   noise-level floats, Nyquist, flags, two calibration floats) both fits
   in 28 bytes and decodes every field to a sane number.  This is a bug
   fix, not an adaptation, and it is safe to carry: no other crate in
   this workspace reads the field, so the renderer's output is
   unchanged.  It is listed here because a diff against the source tree
   must show it as intentional.  The crate's own tests pin the offsets
   with the verbatim bytes of that volume's first RAD block.
10. `vendor/wx-radar/src/level2.rs` gains a `ParseMode`, and with it a
    `Level2File::parse_strict` beside the existing `parse`.  `parse`
    keeps the source tree's behaviour exactly -- it is the renderer's
    parser, and a renderer that draws 95% of a damaged volume beats one
    that draws nothing -- while `parse_strict` refuses a volume, by
    name, the moment any part of it contradicts what it declares about
    itself.  `crates/rw-nexrad` is the only strict caller.

    The lenient path is unchanged in one respect only: a bzip2 LDM block
    that will not decompress used to be substituted into the message
    stream *still compressed*, so a decompression failure was fed to the
    Message-31 parser as though it were radial bytes.  That is not
    leniency, it is manufacturing radials out of noise, so the block now
    contributes nothing (lenient) or refuses the volume (strict).

    What strict adds: a per-radial envelope, from the tighter of the
    message header's size and the Message-31 header's own radial length,
    with every data block pointer, moment header and gate span required
    to lie inside it; the mandatory VOL/ELV/RAD constant blocks required
    by name, so that a generic block is no longer treated as a Nyquist
    source merely because its type byte is `R`; the RAD block's own
    LRTUP required to be 20 or 28, the two layouts whose byte 16..18
    Nyquist position has been verified; unsupported Message-31
    compression, undefined data word widths, and declared lengths that
    leave the volume all refused; and a step to the next message by the
    declared size rather than rounded up to a 2432-byte legacy record.

    VOL and ELV were required by *name* only, which left a radial free to
    carry a constant block of any length declaring any generic-format
    version and still be called conforming.  Both are now parsed.  The
    supported VOL layouts were settled the same way the RAD offsets were
    -- from the bytes of real archived volumes, not from a reading of the
    ICD.  Nine volumes off `unidata-nexrad-level2`, six sites, 2017
    through 2026, produced exactly two `(version, LRTUP)` combinations:
    version 2.0 in 44 bytes (KTLX 2017-05-16, 2018-05-01, 2019-05-20,
    2020-05-22, 2021-05-26) and version 3.0 in 52 (KGWX 2022-03-30,
    KDDC 2024-05-06, KLZK 2025-03-14, KFWS 2026-07-29, KTLX 2026-07-29).
    Version 3.0 appends eight bytes to the 2.0 layout and moves nothing,
    so the pairing is the contract: a block declaring 3.0 in 44 bytes, or
    2.0 in 52, is stating two incompatible things about where its fields
    are.  The gate has teeth beyond the label -- the latitude and
    longitude at bytes 8..16 must be a place on Earth, which four bytes
    read at the wrong offset almost never are.  ELV is required to be its
    only possible layout, 12 bytes.  The crate's tests pin the verbatim
    VOL blocks of the KTLX 2019 and KFWS 2026 volumes and the ELV block
    of the former; `rw_nexrad decode` was run against both whole volumes
    and returned READY.

    **Corrected 2026-07-30, and the correction is the lesson.**  That
    first survey sampled 2017 onward, found only 2.0 and 3.0, and
    concluded the version 1.0 the crate's synthetic fixture declared was
    invented.  It is not: 1.0 is the 2011-2014 era, and it was outside
    the sampling window rather than outside the archive.  Extending the
    window to the years the pre-2016 `.gz` keys cover produced it at
    once, from KTLX 2011-05-24, 2012-04-14, 2013-05-20 and 2014-04-27 --
    the last being the Moore tornado day.  `KNOWN_VOL_LAYOUTS` now reads
    1.0/44, 2.0/44 and 3.0/52, and 2015-05-06 is where 2.0 (and RAD
    LRTUP 28) begins.  A survey is evidence only over the range it
    covers, and the range has to be the range the product will read.

    The original caution still holds in the other direction: had the
    allow-list been written from the fixture *alone* it would have
    admitted 1.0 and refused 2.0 and 3.0, i.e. every volume from 2015 on.
    Either way the fixture was not evidence; the archive is.

12. `vendor/wx-radar/src/level2.rs` learns the second Archive-II shape.
    `decompress_archive2` walked an LDM block table unconditionally, but
    the pre-2016 keys -- everything the archive stores as `..._V06.gz` or
    `..._V03.gz`, roughly 2011 through 2016 -- hold no block table at
    all: once gunzipped, the messages follow the 24-byte volume header
    directly.  Walking a block table over those bytes yields zero-length
    "blocks" and a volume that decodes to nothing, silently.

    `Level2File::is_ldm_framed` decides between the two from the first
    block's payload: an LDM table always opens with the bzip2 metadata
    block, so it begins `BZh`, while in the unframed shape those bytes
    are the first message's CTM header.  Verified against real volumes
    from 2011, 2012, 2013, 2014, 2015 and 2016 (all unframed) and 2017,
    2019 and 2026 (all LDM+bzip2).  It is `pub` and `#[doc(hidden)]` so
    `rw-nexrad`'s framing check can be pinned against it -- the framing
    record and the decoder must agree about which shape they read.

    Strict mode also now refuses a volume that framed cleanly and yielded
    no Message-31 radial at all.  The renderer may draw zero sweeps; an
    observation front door that publishes zero sweeps has published
    nothing and called it READY.

    The crate's own `archive2_volume` test fixture stored its messages
    uncompressed behind a length word, which is neither real shape -- an
    LDM block is always a bzip2 stream.  It now bzip2s them, so the
    fixture is an LDM volume and the tests exercise the layout they name.

    The source tree bounded all of these against the end of the
    decompressed volume instead, so a missing, reordered or corrupt
    pointer read a *neighbouring* radial's bytes and returned plausible
    geometry or a plausible Nyquist velocity, with every stage
    reporting success.  Outer LDM framing cannot see that: the enclosing
    block lengths are valid while the inner pointer is not.

13. `vendor/wx-radar/src/level2.rs::parse_vol_block` now RETURNS the
    site it validates (`VolSite`: latitude, longitude, site ground
    height at byte 16, feedhorn height AGL at byte 18), and
    `Level2File` carries it as `vol_site`.  Strict parsing refuses a
    volume whose radials disagree about the radar's own position.  The
    two height fields are bounded arithmetically (site -450..4500 m
    MSL, feedhorn <= 200 m) the same way the lat/lon fields are: real
    surveys always satisfy them, misaligned bytes almost never do.

    `crates/rw-nexrad`'s site resolution ladder becomes: explicit
    `--site-latlon` override, then the volume's own VOL block, then the
    vendored table.  The VOL block is the self-describing route the
    audit preferred: `site + feedhorn` is the beam origin with no
    downstream addition, which the table cannot supply (130/141
    elevations are placeholders and the populated ones are ground-only).
    A table entry disagreeing with the volume by more than 0.05 deg is
    named in the fix's `source` string.  KPBZ's longitude, transcribed
    16.9 km east of the radar, is corrected in `sites.rs` regardless
    (-80.0178 -> -80.2183, the ROC/NCEI survey value).

14. `crates/rw-obs` is new (not in the source workspace): ArWen's
    observation front doors, one binary per instrument -- `rw_asos`
    (IEM ASOS/AWOS METAR), `rw_mrms`, `rw_stage4`.  They share a crate
    because they share substance: one pack container, one seam time
    spelling, one HTTP agent, one fetch-receipt shape.

    **`rw_asos` is not a crate of Drew's that was copied in, and this
    entry exists so a later reader does not go looking for one.**  Drew's
    `rusty-weather` workspace has no ASOS crate, module or binary -- the
    only METAR code in it is `vendor/metrust/src/io/metar.rs`, a report
    parser this lane does not call.  What `rw_asos` retells headlessly is
    the *query pattern* of the GUI's IEM fetch: the `asos.py` endpoint,
    `tz=Etc/UTC`, `format=onlycomma`, `missing=empty`,
    `report_type=3&report_type=4`, unpadded date components, and
    resolving CSV columns by name are all kept verbatim.  Its own module
    header states the three deliberate departures (the query is bounded
    by a frozen station list rather than pulling every station on Earth
    and filtering locally; the station table is frozen and hashed at
    registration rather than re-read from each fetch; `mslp` and `p01i`
    are additionally requested).  So the lineage is Drew's stack by
    pattern and gpuwm's by authorship, exactly as `rw-nexrad` is.

    Provenance of the code in *this* tree: authored in gpuwm on
    `integration/obs-battery` (`00461023`, 2026-08-03, "three
    observation front doors, decided against real bytes"), ported here
    from that branch's tip `326aa23f` on 2026-08-06 under Drew's "yes
    vendor" authorisation of the same date -- the surface-obs DA lane
    consumes `{case}/surface-obs/surface.v1.json` and could not produce
    one on this line.  MIT, the workspace license, as gpuwm-authored
    code in a gpuwm-authored crate; no third-party attribution rides
    with it because no third-party source does.  Every dependency was
    already in the vendor closure, so it builds `--locked --offline`
    like the rest.

    One adaptation was needed and it is the whole diff against
    `integration/obs-battery`: `src/bin/mrms.rs` called
    `rw_nexrad::s3::list_s3_objects(agent, bucket, prefix, None)`, which
    that branch's older `rw-nexrad` had and this line's newer one has
    refactored into a request builder -- `list_s3(agent,
    ListRequest::new(bucket, prefix))?.objects`.  Same roster, still
    assembled only from pages that proved themselves complete.  `rw_asos`
    itself is byte-identical to the origin branch.

15. `crates/rw-nexrad` gains a library target (`src/lib.rs`) exposing
    the four modules the binary already had.  `rw-obs` needs the
    anonymous-S3 client -- `s3::{build_agent, parse_time, list_s3,
    object_url, boxed_error, hex_sha256}` -- and the alternative to
    exposing it is a second S3 client, re-earning the bugs that
    module's nine-round fix-then-attack audit already fixed.  `main.rs`
    swapped four `mod` lines for one `use rw_nexrad::{decode, live,
    pack, s3}`; no item moved, changed signature, or changed behaviour,
    and the crate's 104 tests pass unchanged.  This is the same move
    `integration/obs-battery` made for the same reason, redone here
    because that branch's `rw-nexrad` predates this line's `live.rs`.
16. `crates/netcdf-writer` is new (not in the source workspace), and it
    is the first WRITE capability in a workspace that until now only
    read NetCDF.  `vendor/netcrust`, its vendored `hdf5-reader` and the
    crates.io `netcdf-reader` are read-only by design, and
    `crates/rw-netcdf` is `inventory` plus `dump`; the only writer was
    `crates/rw-store/src/netcdf3.rs`, a CDF-2 emitter with `NC_FLOAT`
    as its only variable type and no record dimension, which is
    precisely why it cannot write a wrfout (`Times` is `NC_CHAR` on an
    unlimited dimension).  The new crate is seeded from that module's
    byte grammar and extends it to the whole classic format: CDF-1,
    CDF-2 and CDF-5 containers, all eleven classic external types for
    variables and attributes, a record dimension with netCDF-C's
    stride rules (including the one-record-variable special case), and
    `numrecs` patched at `finish`.  It has no dependencies on the
    default feature; the optional `rewrite` feature adds netcrust and
    netcdf-reader for the `nc_rewrite` parity harness only, and the
    `cdylib` target is the ctypes seam `gpuwm/io/nc_writer_bridge.py`
    loads.  `rw-store/netcdf3.rs` is left in place: it is the only
    writer `rw-ingest`'s hour exporter calls, and moving that caller is
    a separate change from introducing the library.
17. **Ensemble products and observation grids: one new crate, two new
    binaries, one new render-request field pair.**  2.5.0's render law
    puts every weather-field product plot on this renderer, and two
    families had no way in.  `gpuwm/da/enprod.py` said so outright --
    *"the vendored rust renderer's catalog has no ensemble entries, so
    this module is matplotlib-only ... there is no second engine to
    switch to"* -- and `tools/da_level2_render.py` /
    `tools/da_nowcast_render.py` were re-implementing this renderer's own
    Filled style in matplotlib while borrowing its assets and palette.

    Nothing here changes what `rw_wrfbatch` draws.  Every new capability
    is a new crate, a new binary, or an `Option` field whose `None` --
    what every caller that does not ask for it produces -- executes no
    code.  That claim is a GATE, not an assertion:
    `tools/rustwx_render_regression_gate.py` renders the repo's own
    fixture wrfout with `--products all` and byte-compares every PNG
    against a baseline recorded from the pre-change binary.  MEASURED:
    the base commit's binary (`d996b9dbe`, rw_wrfbatch sha256
    `0c20002f50beb9fa...`, built in a scratch tree of its own) against
    this commit's (`77c7b674941c1418...`) -- 43 of 43 PNGs byte-identical,
    catalog 22 products unchanged.  The concrete breakage it prevents is a shared
    `StaticPlotDesign` / style-ladder / frame-aspect change made for an
    ensemble or observation product silently restyling every product the
    renderer already draws -- which nothing else in the tree would catch,
    because the render tests assert names, sizes and catalog counts and
    all three survive a restyle untouched.

    **`crates/rustwx-ensemble` is new** (not in the source workspace):
    the ensemble reduction mathematics with no I/O -- mean, sample spread
    (ddof=1), neighbourhood-maximum exceedance probability (NMEP),
    probability-matched mean with both tie rules, the disc-neighbourhood
    geometry, the missingness bookkeeping, and the paintball palette.  It
    is a direct transcription of `gpuwm/da/enprod.py`'s reduction layer,
    which CLAUDE.md's 2.5.0 Python boundary puts on the Rust side of the
    line.  Kept I/O-free so every documented policy is unit-testable
    against the exact probes that module's docstrings name, and the
    twenty-one tests are those probes: the `[10, NaN, 0]` NMEP monotonicity
    case, the `[[100, 1, NaN], [90, 2, 0]]` PMM case that used to return
    `[NaN, 90, 1]`, the `+Inf` stack whose mean and spread contradicted
    their own coverage stamp, the per-point ddof minimum, and the
    footprint count that distinguishes a clipped disc from a
    replicate-padded one.  Cross-checked against the Python: all ten
    probe reductions agree value for value.

    **`crates/rw-wrfbatch` gains a library target and two more binaries.**
    `src/lib.rs` re-exports the five wrfout modules the binary already
    had (`main.rs` keeps its own `#[path]` declarations and is not built
    from the library, so the renderer's bytes are unaffected) plus three
    new modules: `panel.rs` (one arbitrary 2-D plane through the SAME
    projected map, basemap, `StaticPlotDesign`, and PNG writer
    `store_render::render_generic_store_variable` uses), `scales.rs`
    (colour scales for quantities the catalog has no ladder for, because
    they are not weather variables: a standard deviation, a probability,
    an integer tally), `obs_grid.rs` (a native `gpuwm-obs.radar-grid.v1`
    reader), and `annotate.rs` (a re-export shim, not a second schema).

    * `rw_ensbatch` -- N member wrfouts at ONE valid time -> mean,
      spread, NMEP, PMM, paintball.  `rw_wrfbatch`'s positional list
      means "these files are one run", which is why `gpuwm.rustwx`
      already drives it one file per invocation with its own store root;
      N members cannot be expressed in that contract at all, so this is a
      sibling and not a flag.  Each member imports into its own store;
      the field's canonical `FieldSelector` resolves the plane, and the
      panel's colour ladder is resolved from that store's own variable
      metadata (`operational_style_for_store_variable`) so an ensemble
      mean of reflectivity wears the same NWS dBZ table the deterministic
      panel wears.  Paintball is N `ContourLayer`s at one level with one
      stable colour per member NUMBER -- expressible today, no new render
      primitive.
    * `rw_obsgrid` -- the `gpuwm-obs.radar-grid.v1` observation grid, read
      NATIVELY.  This is the one input gap closed by a reader rather than
      a writer, and deliberately: the file is already classic NetCDF on
      the model mass grid, and wrapping observations in a wrfout would
      make `TitleProvenance::LocalImport` state that a model produced
      them.  Products: observed composite reflectivity (masked column max
      over OBSERVED levels only -- the mask is load-bearing, since the
      dense array's unobserved cells hold a fill value), coverage depth,
      radar-overlap count, and radial velocity at each column's lowest
      observed level on a symmetric diverging scale.
    * Site identifiers come from the file's `provenance` JSON global
      attribute, POSITION-CHECKED against `radar_lat`/`radar_lon` (0.1 deg)
      before a label is attached.  The obvious source, the `radar_id`
      variable, is unreadable: the writer emits `NETCDF4_CLASSIC`, where a
      `(radar, nchar)` `NC_CHAR` array arrives through netcrust's HDF5
      path as `DataType::String` and every numeric read of it is refused
      by type.  (A classic-container char array reads fine as byte codes
      -- that is how the wrfout lane reads `Times` -- so this is a
      container difference, not a schema one.)  A site whose provenance
      coordinates disagree keeps its positional name and the decline is
      printed: a label on the wrong marker is worse than no label.

    **`--overlays FILE.json` and `--annotate FILE.json` on `rw_wrfbatch`.**
    `rustwx-render` has carried every primitive these describe since
    before this vendoring -- `ProjectedLineOverlay`,
    `ProjectedPointOverlay`, `ProjectedPlaceLabel`, the three subtitle
    slots -- and all of them take PROJECTED coordinates, which is why
    nothing outside the renderer could reach them and why four separate
    matplotlib modules in this repository hand-rolled the same overlays
    (radar sites and range rings, SPC storm reports, a domain-boundary
    box, tile seams).  The seam is:

    * `rustwx-render::project_geographic_points_with_options` (new,
      additive, no existing caller): `(lat, lon)` degrees into the same
      projected space `build_projected_map_with_options` produces from the
      same mesh and options, reusing `resolved_projector` rather than
      rebuilding one, so a marker cannot land in a different frame than
      the fill it annotates.
    * `rustwx-products::direct::projection`: `build_projected_map_with_projection`'s
      option build is EXTRACTED into `presentation_projected_map_options`
      (and the full-domain branch's into
      `full_domain_projected_map_options`), pure extractions with no
      behaviour change, so the new `project_points_with_projection` reuses
      the exact same options rather than a hand-copied second copy.
    * `rustwx-products::geographic_overlays` (new module): the JSON schema
      -- lines, closed boxes, markers, place labels, and range rings that
      expand to closed lines by a spherical small-circle walk -- plus the
      title/subtitle overrides.  Classified `StablePublic` in the crate's
      own public-surface map, because the CLI deserialises straight into
      it.
    * `DirectBatchRequest` and `StoreRenderConfig`/`BatchRenderRequest`
      each gain `geographic_overlays` and `panel_annotations`, both
      `Option` and both defaulted `None`.  They are applied at the direct
      lane's existing save site (immediately after
      `apply_place_label_overlay_with_density_styling`, which has always
      had exactly this shape) and in the generic `var:` lane, using the
      same bounds and aspect ratio the panel's own projected map was built
      with.  The derived, heavy and windowed lanes do NOT apply them --
      see "not done", below.

    **`batch_render::native_grid_domain_from_coordinates`** (new, a pure
    extraction of `native_grid_domain`'s body) so the two sibling binaries
    frame their panels on the EXACT same bounds arithmetic -- the
    antimeridian-aware longitude choice, the degenerate-extent padding,
    the pole clamp -- rather than a second, subtly different copy.  Two
    framings of one grid is how a member-mean panel stops overlaying the
    deterministic panel it is meant to be compared against.

    Per-file SHA-256 delta (first 16 hex), pre-change -> this commit:

        Cargo.toml                                    7b59e87735c1163d -> f0a52c1979c82420
        crates/rustwx-render/src/lib.rs               25ed8737a6dfb232 -> 9b5c91bc4a9a46c8
        crates/rustwx-render/src/projected_map.rs     2d7281b01804d6b0 -> b38bca098af918c6
        crates/rustwx-products/src/lib.rs             96cf7da0b6391573 -> 85e3b5cbe070c4b1
        crates/rustwx-products/src/direct.rs          1c0daae3de05a779 -> 1f2cb0e694453a42
        crates/rustwx-products/src/direct/projection.rs
                                                      ae9c63f3f668d42d -> d1ecd6505a899e4a
        crates/rustwx-products/src/direct/types.rs    1550103e51e9b382 -> 69c4bd7221a0fa9f
        crates/rustwx-products/src/direct/tests.rs    896facb2d63ba9ff -> 0b53baffbb5af7e6
        crates/rustwx-products/src/non_ecape.rs       34927b6d38253cc6 -> a28ba30701535f77
        crates/rustwx-products/src/non_ecape/tests.rs 1340bc9f81b7db52 -> 79e23821d4209f7c
        crates/rustwx-products/src/tests.rs           d0af4a596444e4a2 -> 3bf8f9474cf250b6
        crates/rustwx-products/src/geographic_overlays.rs
                                                        (new file)   -> d00eae35dc548af3
        crates/rusty-weather/src/batch_render.rs      9cf93e2c148fc172 -> 7dd7a9548e6eb891
        crates/rusty-weather/src/render_all.rs        6f850ccb1f16c178 -> a74777e3ba60b8c9
        crates/rusty-weather/src/store_render.rs      2a3d54fa2f18c742 -> 64f2daf89c32d310
        crates/rw-wrfbatch/Cargo.toml                 ea3cbe6e5e909f7b -> e1d2984ace2d3a6f
        crates/rw-wrfbatch/src/main.rs                9e8c595c9d2dc15f -> 8e673415944c7ed2
        crates/rw-wrfbatch/src/lib.rs                   (new file)   -> 3c32ac48a6dcddaf
        crates/rw-wrfbatch/src/annotate.rs              (new file)   -> 54c595f55cc2565a
        crates/rw-wrfbatch/src/panel.rs                 (new file)   -> 8adc394850cdafea
        crates/rw-wrfbatch/src/scales.rs                (new file)   -> 2a304ea1a5660bd3
        crates/rw-wrfbatch/src/obs_grid.rs              (new file)   -> 3f45e1af176d9bfc
        crates/rw-wrfbatch/src/bin/ensbatch.rs          (new file)   -> 1b4c98d256a31618
        crates/rw-wrfbatch/src/bin/obsgrid.rs           (new file)   -> fadf563b3bfd1b32
        crates/rustwx-ensemble/Cargo.toml               (new file)   -> b9c7ae3595718d25
        crates/rustwx-ensemble/src/lib.rs               (new file)   -> 9b8fc20dbf1fc0df

    No verbatim-vendored upstream file outside `crates/rustwx-render`,
    `crates/rustwx-products` and `crates/rusty-weather` is touched, and
    nothing under `vendor/` is touched at all.

    **Deliberately NOT done, so a later reader does not go looking:**

    * The derived, heavy and windowed render lanes do not apply
      `--overlays`/`--annotate`.  The direct lane covers composite
      reflectivity, 2 m temperature, 10 m wind, QPF and terrain -- every
      product the two calling lanes overlay -- and the generic `var:`
      lane covers the rest of a wrfout's stored planes.  Extending it is
      three more `DerivedBatchRequest`/`HrrrWindowedBatchRequest` fields
      and three more save sites, all of the same shape.
    * There is no FOOTER BAND.  `MapRenderRequest` has a title and three
      subtitle slots and no fourth text region, so
      `bigdomain_render.py`'s multi-paragraph `IC_FOOTER` honesty block
      has nowhere honest to go and is not silently squeezed into a
      subtitle.  Note also that the renderer centres `subtitle_center` on
      the PANEL rather than in the gap between the other two slots, so a
      long centre string overlaps them; the schema's doc comment says so.
    * Multi-panel COMPOSITION (lead rows, 3xN strips, verification pairs,
      evolution strips, 6-panel sheets) stays out of Rust entirely.
      `gpuwm/pair_compose.py` already pastes finished PNGs with Pillow
      ("pixels are pasted, never recomputed"), which is the same
      operation over panels these binaries draw.
    * The ANALYSIS charts named in the recon stay on matplotlib, which
      the render law permits: seam-roughness statistics, innovation-RMS
      and spread line charts, the per-frame verification table, and
      FSS-vs-neighbourhood curves.  None of them is a weather field.

  - **Column extremes of vertical velocity join the raw-extras table**
    (`crates/rw-wrfbatch/src/wrf_process.rs`, proving lane, 2026-08-17).
    TWO TABLE ROWS, no code path -- `W_UP_MAX` and `W_DN_MAX` appended to
    `RAW_EXTRA_CATALOG`, which already carried `WSPD10MAX` and
    `UP_HELI_MAX`: the same family of WRF Registry column/period maxima,
    read verbatim, published as `wrf_w_up_max` / `wrf_w_dn_max` and drawn
    through the generic `var:` lane.

    WHY: a SURFACE snapshot has no profile, so a column maximum is the
    only way the updraft reaches a panel at all, and the tile-streamed
    lane's `wmax` figure was the last weather field that lane had no
    production renderer for.  gpuwm's writer publishes its own extremes
    under the Registry's spellings so a reader who knows wrfout knows
    them; the divergence from WRF is the averaging WINDOW (WRF's are
    running maxima between history writes, gpuwm's are the frame's own
    instant) and the file states that in a `GPUWM_W_EXTREME_SEMANTICS`
    global attribute rather than leaving a reader to assume.

    Additive by construction: a wrfout that does not carry the variables
    is unchanged, which the pixel gate measured -- `record` against a
    binary built from this file's pre-change state (sha256
    `1c32fff0fa8086a6...`) and `check` against the rebuilt one
    (`89599e48a863c98d...`) reported `COMPARED pngs=43 identical=43`,
    same 22-product catalog both sides.

        Per-file SHA-256 delta (first 16 hex), pre-change -> this commit:

        crates/rw-wrfbatch/src/wrf_process.rs
                                                      3e66e5c0b18135c4 -> f34a32a64328a777

  - **The domain frame of a RESAMPLED grid is its bounding box, not the
    rectangle inscribed in it** (`crates/rustwx-render/src/render.rs`,
    renderer-defect lane, 2026-08-17).

    `compute_projected_domain_frame_rect` fed the projected grid's
    coverage mask to `inner_rect_from_coverage`, which erodes every row
    covered across less than 90% of the rect's width and every column
    covered across less than 90% of its height, three times.  That is the
    largest axis-aligned rectangle INSCRIBED in the footprint, and it is
    the right answer for the grid the source workspace draws: a wrfout on
    its OWN map projection is a rectangle on screen, so the erosion only
    trims the antialiased rim.

    A regular latitude/longitude field is not that grid.  It is resampled
    per pixel into a PRESENTATION projection (the inverse-raster path),
    and for a regional window the adaptive chooser picks a conic -- so its
    footprint is a curved quadrilateral whose top and bottom edges are
    arcs.  On a curve the erosion does not converge on the rim, it eats
    real domain, and `DomainFrame::clear_outside` then erases everything
    outside the band it lands on.  MEASURED on a 0.25 deg 20-55N/130-60W
    field at 1600x900: 256 of 900 rows drawn, the map reduced to a sliver
    with the title and colourbar following it down the canvas.  Reported
    twice by the round-2 gauntlet (`MODEL-GAUNTLET2-SCORECARD-2026-08-17`,
    "rw_wrfbatch MAP_PROJ=6 draws only ~half the south_north rows"), and
    worked around in that lane by stamping a lat/lon file `MAP_PROJ = 3`.

    The frame now inscribes only when the footprint IS a screen rectangle,
    which is exactly when no inverse raster is in play
    (`opts.inverse_projected_grid.is_none()`); otherwise it is the
    bounding box of the coverage, inset as before.  The two agree
    whenever the footprint is a rectangle, so nothing the renderer
    already drew can move -- MEASURED, not asserted: `record` against a
    binary built from this tree with the flag pinned to the old arm and
    `check` against the fixed one reported `COMPARED pngs=43
    identical=43`, same 22-product catalog both sides.  A second arm on
    the defect itself: a `MAP_PROJ = 3` lat/lon file renders
    byte-identical across the change while the `MAP_PROJ = 6` twin does
    not.

    Known limit, named rather than hidden: a genuinely CURVILINEAR mesh
    (a rotated-pole grid whose XLAT/XLONG are not rectilinear) takes
    neither branch's protection, because it does not qualify for the
    inverse raster either.  A non-default POLE_LAT/POLE_LON over a
    rectilinear mesh does qualify and is fixed (MEASURED: 38 bands, the
    same as the unrotated twin).

        Per-file SHA-256 delta (first 16 hex), pre-change -> this commit:

        crates/rustwx-render/src/render.rs
                                                      8d314f9181c54733 -> 2d6f305afdf3dec5

  - **A member is a time series, so `rw_ensbatch` imports all of it**
    (`crates/rw-wrfbatch/src/bin/ensbatch.rs`, renderer-defect lane,
    2026-08-17).

    `newest_wrfout` returned `found.pop()` -- the last wrfout by name in a
    member directory -- on the premise that "the last by name is the
    latest valid time, which `--frames` then indexes into".  The premise
    is wrong about what `--frames` indexes: it indexes the member's
    STORE, and the store only ever received the one file that was
    imported.  With `frames_per_outfile = 1` (WRF's default, and what
    every member run in this tree wrote) every member's store held
    exactly one frame, so `--frames 0` drew the FINAL valid time under
    the name of the first and `--frames 1` and up refused with "its store
    has 1 frame(s)".  Reported by the round-2 gauntlet's AI-ensemble lane
    against real GEFS-shaped products.

    `member_wrfout_series` returns every wrfout of the one domain,
    ordered by the model timestamp in the filename, and the whole list
    goes into one `spawn_process_paths` call -- rw-store merges files
    written into one run, and a member's own frames are exactly that.
    The mixed-nest refusal is unchanged and still names both domains.
    `--member NUMBER=PATH` accepts a member DIRECTORY as well as a single
    file, so the explicit route and the manifest route describe the same
    thing.  MEASURED on the staged three-member 00Z hybrid ensemble,
    seven single-frame wrfouts each: pre-fix one frame per member and the
    `--frames 0` panel byte-identical to the post-fix `--frames 6` panel;
    post-fix seven frames per member and three distinct panels at frames
    0, 3 and 6.  The `--abi` marker is unchanged because the vocabulary
    the Python half parses did not change.

        Per-file SHA-256 delta (first 16 hex), pre-change -> this commit:

        crates/rw-wrfbatch/src/bin/ensbatch.rs
                                                      107d11b1117714ce -> 0020a250b792120b

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

## crates/static-fields lane-3 decode dependencies (2026-08-17)

The high-resolution raster substrate (`crates/static-fields`
`src/raster/**`, the GeoTIFF decode/encode + warp that replaces
rasterio/GDAL in `gpuwm/static/highres*.py`) declares three crates,
every one ALREADY a `Cargo.lock` + `vendor/crates-io` resident of this
workspace before the lane landed -- no new package, no new registry
fetch, `--offline` clean-clone builds unchanged:

- `miniz_oxide` (0.8.9): the zlib streams inside deflate-compressed
  GeoTIFF strips/tiles, both decode of the real source tiles and the
  crate's own byte-deterministic derived-window writer;
- `weezl` (0.1.12): LZW decode only, for sources GDAL compressed that
  way (the TIFF-flavoured early-change codec);
- `sha2` (0.10.9): BoundRaster provenance verification, which moves to
  Rust with the compute so no unverified byte is ever decoded.

The GeoTIFF container itself is a purpose-written reader/writer in
`src/raster/geotiff.rs` (classic + BigTIFF, striped + tiled,
predictor none/horizontal/floating-point, the closed GeoKey CRS
inventory), chosen over the crates.io `tiff` crate because the decode
envelope is deliberately narrow and refuse-by-name; its parity gates
run against real Copernicus DEM windows in
`crates/static-fields/tests/fixtures/highres/` (see
`generate_goldens.py` there for the provenance of every expected
byte).

## crates/obs-regrid — seeded from Drew's rustwx-regrid (2026-08-18)

`crates/obs-regrid` is the Rust half of `gpuwm/verify/obs/regrid.py`,
the observation battery's remap.  It declares **no dependencies at
all**, so it adds nothing to the vendor closure and nothing to
`Cargo.lock` beyond its own entry.

It is not a verbatim vendoring.  Its seed is `crates/rustwx-regrid` in
Drew's consolidated workspace
(`%USERPROFILE%\rusty-weather-consolidated`), and what was taken is
the SHAPE rather than the arithmetic:

- the plan-as-data discipline (`plan.rs`): a remap between two fixed
  grids is a fixed mapping, built once and applied many;
- `apply_into_*` writing a caller-owned buffer, which is what lets the
  ctypes seam fill preallocated numpy arrays instead of round-tripping;
- the bounded-distance validation and its refusal class
  (`nearest.rs::build_nearest_weights`);
- the error taxonomy (`error.rs`), minus `UnsupportedGeometry`: this
  engine takes lat/lon arrays directly rather than a geometry trait
  object, so no geometry can decline to expose a centre.

What is written here rather than taken, and why the crate could not
simply be vendored: `rustwx-regrid` regrids STRUCTURED grids with
sparse weights and a NaN missing policy, while the battery remaps
scattered curvilinear observation swaths with an explicit validity
field.  Its `Nearest` is index arithmetic on a regular lat/lon spec
with a brute-force fallback and a haversine-kilometre bound; this one
is a k-d tree over unit vectors with a chord bound.  Its
`Conservative` is area overlap; this crate's `cell_average` is a
REVERSE assignment (each source cell to its nearest destination
centre, then a mean per destination cell), which is a different
operator answering a different question.

The k-d tree (`kdtree.rs`) is purpose-written rather than taken from
crates.io for one reason stated in its own header: an off-the-shelf
neighbour search resolves exact ties by its own undocumented traversal
order, which is precisely the defect this port exists to remove.

Nothing under `vendor/` changed for this crate.
