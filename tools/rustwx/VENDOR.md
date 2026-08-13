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
