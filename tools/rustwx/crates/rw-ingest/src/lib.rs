//! Live GRIB -> `.rws` ingest as a library: the exact fetch/extract/derive/
//! write flow the `rw_ingest`/`rw_batch` bins have always run, extracted
//! from their `#[path]`-shared modules so interactive hosts (the
//! rusty-weather-ui shell) can run ingests in-process on background
//! threads with progress events and cancellation.
//!
//! Layout mirrors the old bin-side modules one-to-one:
//! - [`ingest_hour`] (re-exported at the crate root): per-hour
//!   [`fetch_hour`] / [`process_fetched_hour`] / [`ingest_hour()`],
//!   [`IngestConfig`], [`planned_store_variables`], [`parse_hours`].
//! - [`ingest_profile`]: what one run fetches/extracts/computes/stores
//!   (`full` / `sounding` / `view` presets + overrides + validation).
//! - [`ingest_compute`]: the derived/heavy precompute over the products
//!   decode lane.
//! - [`size_estimate`]: exact (`walk_hour_sizes`) and predictive
//!   (`estimate` against a [`size_estimate::Calibration`]) sizing.
//! - [`throttle`]: polite scheduling — the bins' process-wide knobs plus
//!   the per-thread / dedicated-pool variants for interactive hosts.
//! - [`events`] (re-exported at the crate root): [`IngestEvent`] progress
//!   stream, [`IngestStage`], [`IngestError`] (with a `Cancelled` variant),
//!   and [`print_event`] — the sink that reproduces the bins' historical
//!   stdout/stderr lines byte-for-byte.

mod events;
pub mod ingest_hour;
pub mod throttle;

// Child modules of `ingest_hour` historically; kept reachable both ways.
pub use ingest_hour::ingest_compute;
pub use ingest_hour::ingest_profile;
pub use ingest_hour::size_estimate;

pub use events::{IngestError, IngestEvent, IngestStage, NEVER_CANCEL, print_event};
pub use ingest_hour::{
    FetchedHour, IngestConfig, IngestedHour, PlannedStoreVariables, SpilledFetchedHour,
    VolumeSummary, cache_state, fetch_hour, ingest_hour as ingest_hour_serial, parse_hours,
    planned_store_variables, process_fetched_hour, validate_forecast_hours,
};

/// Short git SHA (plus `-dirty`) of the build that produced this crate, the
/// same stamp `write_hour_from_fields_with_derived` records in `run.json`.
pub fn build_sha() -> &'static str {
    env!("RW_BUILD_SHA")
}

/// One product file to fetch for an hour and the roles its messages serve.
/// The per-hour flow has two extraction roles — a pressure-source file (the
/// 3D isobaric volumes + render-grade isobaric planes + the prs-side thermo
/// decode) and a surface-source file (the 2D surface set + the surface-side
/// thermo decode). HRRR splits them across two physical files (`prs`/`sfc`);
/// GFS's single `pgrb2.0p25` carries both, so one entry sets both roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductFetch {
    /// Product token passed to [`rustwx_core::ModelRunRequest::new`].
    pub product: &'static str,
    /// The 2D surface field extraction reads this file.
    pub surface_source: bool,
    /// The 3D volume / isobaric-plane extraction reads this file.
    pub pressure_source: bool,
    /// `.idx` substring patterns selecting only the GRIB messages this file's
    /// roles need. Empty = fetch the whole file (HRRR/GFS — preserving their
    /// historical byte-identical whole-file fetch). Non-empty = the fetch path
    /// passes these as `FetchRequest.variable_patterns`, which triggers the
    /// existing AWS/Google `.idx` message-subset fetch (ranged GET of just the
    /// matched messages, cache keyed by the pattern set — see
    /// `rustwx_io::fetch_bytes_with_cache`). RRFS-A's NA files are 4.3 GB
    /// (`prs-na`) + 9.1 GB (`nat-na`), so subsetting is mandatory: the surface
    /// subset is ~1.8% of the file, the pressure subset ~86% (the isobaric
    /// volumes are inherently most of the pressure file).
    pub idx_patterns: &'static [&'static str],
}

/// The per-model fetch plan: which product file(s) one hour downloads and
/// which extraction roles each serves. HRRR keeps its historical two-file
/// pair (pressure = `prs`, surface = `sfc`) in that exact order so its
/// fetch URLs and extraction sequence stay byte-identical; single-file
/// analysis products fetch once and serve both roles from the same bytes.
///
/// Models that are not ingest-supported (see [`ingest_supported`]) return an
/// error rather than a plan — callers gate on `ingest_supported` first, so
/// this is a defensive guard, not the primary check.
pub fn fetch_plan(model: rustwx_core::ModelId) -> Result<Vec<ProductFetch>, IngestError> {
    use rustwx_core::ModelId;
    match model {
        ModelId::Hrrr | ModelId::HrrrAk => Ok(vec![
            ProductFetch {
                product: "prs",
                surface_source: false,
                pressure_source: true,
                idx_patterns: &[],
            },
            ProductFetch {
                product: "sfc",
                surface_source: true,
                pressure_source: false,
                idx_patterns: &[],
            },
        ]),
        ModelId::Gfs | ModelId::Gdas => Ok(vec![ProductFetch {
            product: "pgrb2.0p25",
            surface_source: true,
            pressure_source: true,
            idx_patterns: &[],
        }]),
        ModelId::Gefs => Ok(vec![ProductFetch {
            product: "pgrb2ap5/gec00",
            surface_source: true,
            pressure_source: true,
            idx_patterns: &[],
        }]),
        ModelId::Aigfs => Ok(vec![
            ProductFetch {
                product: "pres",
                surface_source: false,
                pressure_source: true,
                idx_patterns: &[],
            },
            ProductFetch {
                product: "sfc",
                surface_source: true,
                pressure_source: false,
                idx_patterns: &[],
            },
        ]),
        ModelId::Aigefs => Ok(vec![
            ProductFetch {
                product: "pres/avg",
                surface_source: false,
                pressure_source: true,
                idx_patterns: &[],
            },
            ProductFetch {
                product: "sfc/avg",
                surface_source: true,
                pressure_source: false,
                idx_patterns: &[],
            },
        ]),
        ModelId::Hgefs => Ok(vec![
            ProductFetch {
                product: "pres/avg",
                surface_source: false,
                pressure_source: true,
                idx_patterns: &[],
            },
            ProductFetch {
                product: "sfc/avg",
                surface_source: true,
                pressure_source: false,
                idx_patterns: &[],
            },
        ]),
        ModelId::EcmwfOpenData => Ok(vec![ProductFetch {
            product: "oper",
            surface_source: true,
            pressure_source: true,
            idx_patterns: &[],
        }]),
        ModelId::Rap => Ok(vec![ProductFetch {
            product: "awp130pgrb",
            surface_source: true,
            pressure_source: true,
            // Keep RAP whole-file for now: the shared gridded fetch path
            // documents old subset attempts missing pressure-level winds.
            idx_patterns: &[],
        }]),
        ModelId::Nam => Ok(vec![ProductFetch {
            product: "awip3d",
            surface_source: true,
            pressure_source: true,
            idx_patterns: &[],
        }]),
        // RRFS-A: the only files carrying RRFS surface fields are the NA pair
        // (recon-verified — `prslev.conus` is pressure-only, `natlev.conus`
        // 404s). Both are the SAME rotated-pole grid (GRIB template 1,
        // 4881x2961); they are cropped to a CONUS box at ingest (see
        // `model_crop_box`). The files are huge (prs-na 4.3 GB, nat-na 9.1 GB),
        // so each carries `.idx` patterns to subset-fetch only the messages its
        // role needs.
        ModelId::RrfsA => Ok(vec![
            ProductFetch {
                product: "prs-na",
                surface_source: false,
                pressure_source: true,
                idx_patterns: RRFS_PRS_IDX_PATTERNS,
            },
            ProductFetch {
                product: "nat-na",
                surface_source: true,
                pressure_source: false,
                idx_patterns: RRFS_NAT_IDX_PATTERNS,
            },
        ]),
        other => Err(events::other(format!(
            "model '{other}' has no ingest fetch plan (not ingest-supported)"
        ))),
    }
}

/// `.idx` message-selection patterns for the RRFS-A `prs-na` (pressure) file:
/// the isobaric volume field types the ingest plan decodes (T/RH/DPT for the
/// dewpoint→rh fallback, U/V, geopotential height) plus absolute vorticity
/// (the sparse per-level planes). Each is a bare `VARIABLE` token (no level
/// filter) so it matches every isobaric level of that field via
/// [`wx_core::download::find_entries`]; the level subset the profile stores is
/// realized at decode.
///
/// **Pattern format (load-bearing):** `find_entries` parses a pattern as
/// `VARIABLE[:level-substring]`, matching the GRIB variable name EXACTLY and the
/// level as a substring — it does NOT wrap the token in `.idx` field colons.
/// A `:VAR:` form would split at the leading colon into an empty variable name
/// and match nothing (the fetcher would then silently fall back to a whole-file
/// GET — a ~4 GB regression). So these are bare `TMP`, `RH`, … exactly as every
/// other model's `idx_patterns` are written.
///
/// Over-fetches the few stratospheric levels above the stored set (substring
/// patterns can't express "≥100 mb"), so the realized subset is ~69% of the
/// file (measured against the live f001 `.idx`, 2026-06-11) — the isobaric
/// volumes ARE most of the pressure file.
const RRFS_PRS_IDX_PATTERNS: &[&str] = &["TMP", "RH", "DPT", "UGRD", "VGRD", "HGT", "ABSV"];

/// `.idx` message-selection patterns for the RRFS-A `nat-na` (surface) file: the
/// 2D surface set the ingest plan extracts plus the trailing 1 h window messages
/// (APCP 0-1 h acc, MXUPHL 2-5 km 0-1 h max, WIND 10 m 0-1 h max). `MSLET`
/// (mean sea level) covers the `mslp` selector (`PARAMETER_MSLP` matches it).
///
/// Same `VARIABLE[:level-substring]` format as [`RRFS_PRS_IDX_PATTERNS`] — bare
/// variable names and `VAR:level` tokens, never `:VAR:level:` (see that doc for
/// why colon-wrapping silently disables subsetting). Level-qualified entries
/// pin the height/surface variant (e.g. `TMP:2 m above ground`); bare entries
/// (`REFC`, `MSLET`, `PWAT`, the categorical precip flags) match the single
/// message of that variable.
///
/// `CAPE` (bare) pulls the native CAPE planes (surface / 90-0 mb ML /
/// 255-0 mb MU, plus a harmless 180-0 mb layer) that the heavy native-ECAPE
/// ratio recipes consume; `TCDC:entire atmosphere` is deliberately
/// level-qualified — bare `TCDC` would drag ~60 per-hybrid-level cloud planes
/// (hundreds of MB) instead of the 2 entire-atmosphere messages. LCDC/MCDC/HCDC
/// exist in `natlev.na` only at their cloud-layer levels (live-idx verified
/// 2026-06-11 — the original recon's "only TCDC" claim was wrong), so bare
/// tokens are exact.
///
/// Tiny: ~2.6% of the 9.2 GB file (measured against the live f001 `.idx`:
/// 33 messages, ~226 MB).
const RRFS_NAT_IDX_PATTERNS: &[&str] = &[
    "TMP:2 m above ground",
    "DPT:2 m above ground",
    "RH:2 m above ground",
    "SPFH:2 m above ground",
    "UGRD:10 m above ground",
    "VGRD:10 m above ground",
    "REFC",
    "MSLET",
    "PRES:surface",
    "HGT:surface",
    "GUST:surface",
    "PWAT",
    "APCP:surface",
    "CRAIN",
    "CSNOW",
    "CICEP",
    "CFRZR",
    "VIS:surface",
    "MXUPHL:5000-2000 m above ground",
    "MXUPHL:3000-0 m above ground",
    "WIND:10 m above ground",
    "MAXUW:10 m above ground",
    "MAXVW:10 m above ground",
    "CAPE",
    "TCDC:entire atmosphere",
    "LCDC",
    "MCDC",
    "HCDC",
];

/// The geographic CONUS crop box for a model whose native ingest domain is
/// larger than CONUS (RRFS-A's North America rotated-pole grid), as
/// `(west, east, south, north)` degrees. `None` = no crop (HRRR, GFS — the
/// native grid is already the store grid).
///
/// RRFS-A: the NA files (4881x2961, GRIB template 1 rotated-pole, unrotated to
/// curvilinear geographic by grib-core) cover all of North America. Cropping to
/// this box at ingest keeps the store HRRR-class (~5.1M cells, ~2.7x HRRR)
/// instead of 14.5M. The box is chosen so RRFS-CONUS coverage ⊇ HRRR-CONUS:
/// HRRR's CONUS Lambert grid spans roughly lat 21.1..52.6, lon -134.1..-60.9;
/// these bounds (21.0..53.5, -134.5..-60.5) bound it with a small margin.
/// Realized on the native rotated index grid as a contiguous block
/// (~1736x2931 cells; the rotated grid is skewed so the index block
/// over-covers the geographic rectangle — the true rectangle is fully inside).
/// Determinism: the crop index range is a pure function of the grid's per-cell
/// coordinates (first/last row+col whose geographic point lies in the box),
/// computed once per hour — no per-run floating-point branch.
pub fn model_crop_box(model: rustwx_core::ModelId) -> Option<(f64, f64, f64, f64)> {
    use rustwx_core::ModelId;
    match model {
        ModelId::RrfsA => Some((-134.5, -60.5, 21.0, 53.5)),
        _ => None,
    }
}

/// Whether this crate can ingest `model` today. Backed by [`fetch_plan`]:
/// a model is ingest-supported exactly when a per-model fetch plan exists
/// for it (HRRR's `prs`/`sfc` pair, GFS's single `pgrb2.0p25`). UI pickers
/// gate enablement on this so the list self-updates as plans land.
pub fn ingest_supported(model: rustwx_core::ModelId) -> bool {
    fetch_plan(model).is_ok()
}

/// Crate-local profiling scope: expands to `puffin::profile_scope!` under
/// the `profiling` feature and to nothing otherwise, so call sites stay
/// clean and headless bins compile puffin out entirely.
#[cfg(feature = "profiling")]
macro_rules! profile_scope {
    ($($arg:tt)*) => {
        puffin::profile_scope!($($arg)*);
    };
}
#[cfg(not(feature = "profiling"))]
macro_rules! profile_scope {
    ($($arg:tt)*) => {};
}
pub(crate) use profile_scope;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_sha_is_stamped() {
        assert!(!build_sha().is_empty());
    }

    #[test]
    fn fetch_plan_models_are_ingest_supported() {
        use rustwx_core::ModelId;
        let enabled = [
            ModelId::Hrrr,
            ModelId::HrrrAk,
            ModelId::Rap,
            ModelId::Gfs,
            ModelId::Gdas,
            ModelId::Gefs,
            ModelId::Aigfs,
            ModelId::Aigefs,
            ModelId::Hgefs,
            ModelId::EcmwfOpenData,
            ModelId::Nam,
            ModelId::RrfsA,
        ];
        for model in enabled {
            assert!(
                ingest_supported(model),
                "{model} should be ingest-supported"
            );
        }
        // Every other user-facing model stays gated until its fetch plan lands.
        for model in rustwx_models::supported_models() {
            if !enabled.contains(&model) {
                assert!(
                    !ingest_supported(model),
                    "{model} must stay gated until its fetch plan exists"
                );
            }
        }
        // A model with no plan (e.g. NBM) is explicitly unsupported.
        assert!(!ingest_supported(ModelId::Nbm));
    }

    #[test]
    fn whole_file_models_carry_no_idx_patterns() {
        use rustwx_core::ModelId;
        // These plans intentionally keep whole-file fetches (empty patterns).
        for model in [
            ModelId::Hrrr,
            ModelId::HrrrAk,
            ModelId::Rap,
            ModelId::Gfs,
            ModelId::Gdas,
            ModelId::Gefs,
            ModelId::Aigfs,
            ModelId::Aigefs,
            ModelId::Hgefs,
            ModelId::EcmwfOpenData,
            ModelId::Nam,
        ] {
            for entry in fetch_plan(model).expect("plan") {
                assert!(
                    entry.idx_patterns.is_empty(),
                    "{model} product {} must fetch whole-file (no idx subset)",
                    entry.product
                );
            }
        }
        // Whole-file first-wave models have no crop box.
        assert!(model_crop_box(ModelId::Hrrr).is_none());
        assert!(model_crop_box(ModelId::HrrrAk).is_none());
        assert!(model_crop_box(ModelId::Rap).is_none());
        assert!(model_crop_box(ModelId::Gfs).is_none());
        assert!(model_crop_box(ModelId::Gdas).is_none());
        assert!(model_crop_box(ModelId::Nam).is_none());
    }

    #[test]
    fn fetch_plan_rrfs_a_is_the_na_pair_with_subset_patterns() {
        use rustwx_core::ModelId;
        let plan = fetch_plan(ModelId::RrfsA).expect("RRFS-A plan");
        assert_eq!(plan.len(), 2, "RRFS-A fetches prs-na + nat-na");
        // Pressure role first (the historical pressure-then-surface order).
        assert_eq!(plan[0].product, "prs-na");
        assert!(plan[0].pressure_source && !plan[0].surface_source);
        assert_eq!(plan[1].product, "nat-na");
        assert!(plan[1].surface_source && !plan[1].pressure_source);
        // Both files are huge → both MUST subset-fetch.
        assert!(
            !plan[0].idx_patterns.is_empty() && !plan[1].idx_patterns.is_empty(),
            "RRFS-A NA files (4.3+9.1 GB) must subset-fetch"
        );
        // The surface plan must reach the trailing 1 h window messages and the
        // honest MSLET→mslp message.
        let nat = plan[1].idx_patterns;
        assert!(nat.iter().any(|p| p.contains("APCP:surface")));
        assert!(nat.iter().any(|p| p.contains("MXUPHL:5000-2000 m")));
        assert!(nat.iter().any(|p| p.contains("WIND:10 m above ground")));
        assert!(nat.iter().any(|p| p.contains("MSLET")));
        // The pressure plan must reach the isobaric volume field types.
        let prs = plan[0].idx_patterns;
        for need in ["TMP", "RH", "UGRD", "VGRD", "HGT"] {
            assert!(prs.contains(&need), "prs subset missing {need}");
        }
    }

    #[test]
    fn rrfs_a_has_a_conus_crop_box_bounding_hrrr() {
        use rustwx_core::ModelId;
        let (w, e, s, n) = model_crop_box(ModelId::RrfsA).expect("RRFS-A crop box");
        // (west, east, south, north). Must bound HRRR's CONUS coverage with a
        // margin so RRFS-CONUS ⊇ HRRR-CONUS.
        assert!(w <= -134.0 && e >= -61.0, "lon box must bound HRRR conus");
        assert!(s <= 21.1 && n >= 52.6, "lat box must bound HRRR conus");
        assert!(w < e && s < n, "box must be well-ordered");
    }

    /// Replica of `wx_core::download::find_entries`'s match rule (split the
    /// pattern on its FIRST colon into an exact variable name + a level
    /// substring) so this crate can assert its `.idx` patterns actually select
    /// messages WITHOUT taking a dev-dep on wx-core. Kept deliberately tiny and
    /// in lock-step with the real parser; the real parser is itself covered by
    /// a regression test in `vendor/wx-core/src/download/idx.rs`.
    fn idx_line_matches(pattern: &str, variable: &str, level: &str) -> bool {
        let (var_pat, level_pat) = match pattern.find(':') {
            Some(i) => (&pattern[..i], Some(&pattern[i + 1..])),
            None => (pattern, None),
        };
        variable == var_pat && level_pat.is_none_or(|lp| level.contains(lp))
    }

    /// REGRESSION (root cause of the first live ingest fetching whole 4.3+9.1 GB
    /// files instead of subsetting): the patterns MUST be bare
    /// `VARIABLE[:level]` tokens, NOT colon-wrapped `:VARIABLE:level:`.
    /// `find_entries` splits on the first colon, so a leading colon yields an
    /// empty variable name that matches nothing, and the fetcher then silently
    /// falls back to a whole-file GET. This test proves the live-idx field rows
    /// (variable, level taken verbatim from the 2026-06-11 f001 `.idx` files)
    /// are selected by the current patterns, and that the old `:VAR:` framing
    /// would select nothing.
    #[test]
    fn rrfs_a_idx_patterns_select_real_idx_rows() {
        // (variable, level) rows that must be reachable, verbatim from the live
        // natlev.na / prslev.na .idx files.
        let nat_rows = [
            ("TMP", "2 m above ground"),
            ("DPT", "2 m above ground"),
            ("UGRD", "10 m above ground"),
            ("VGRD", "10 m above ground"),
            ("REFC", "entire atmosphere (considered as a single layer)"),
            ("MSLET", "mean sea level"),
            ("APCP", "surface"),
            ("MXUPHL", "5000-2000 m above ground"),
            ("MXUPHL", "3000-0 m above ground"),
            ("WIND", "10 m above ground"),
            // Native CAPE planes for the heavy native-ECAPE ratio recipes.
            ("CAPE", "surface"),
            ("CAPE", "90-0 mb above ground"),
            ("CAPE", "255-0 mb above ground"),
            // Cloud cover: TCDC entire-atmosphere + the per-layer LCDC set
            // (natlev carries them; the original recon missed them).
            ("TCDC", "entire atmosphere (considered as a single layer)"),
            ("LCDC", "low cloud layer"),
            ("MCDC", "middle cloud layer"),
            ("HCDC", "high cloud layer"),
        ];
        // Bare `TCDC` must NOT be a pattern: natlev carries ~60 per-hybrid-level
        // TCDC planes and an unqualified token would fetch them all.
        assert!(
            !RRFS_NAT_IDX_PATTERNS.contains(&"TCDC"),
            "TCDC must stay level-qualified (entire atmosphere)"
        );
        for (var, level) in nat_rows {
            assert!(
                RRFS_NAT_IDX_PATTERNS
                    .iter()
                    .any(|p| idx_line_matches(p, var, level)),
                "nat patterns select nothing for {var}:{level}"
            );
        }
        // prslev.na isobaric rows (one example level per field type).
        let prs_rows = [
            ("TMP", "500 mb"),
            ("RH", "850 mb"),
            ("DPT", "700 mb"),
            ("UGRD", "250 mb"),
            ("VGRD", "250 mb"),
            ("HGT", "500 mb"),
            ("ABSV", "500 mb"),
        ];
        for (var, level) in prs_rows {
            assert!(
                RRFS_PRS_IDX_PATTERNS
                    .iter()
                    .any(|p| idx_line_matches(p, var, level)),
                "prs patterns select nothing for {var}:{level}"
            );
        }
        // No pattern may be colon-wrapped — that is the exact shape that
        // silently disabled subsetting and must never regress.
        for pattern in RRFS_NAT_IDX_PATTERNS.iter().chain(RRFS_PRS_IDX_PATTERNS) {
            assert!(
                !pattern.starts_with(':') && !pattern.ends_with(':'),
                "idx pattern {pattern:?} is colon-wrapped (matches nothing in find_entries)"
            );
            // And prove the colon-wrapped form really would match nothing, so
            // the guard above is meaningful and not cosmetic.
            let wrapped = format!(":{pattern}:");
            assert!(
                !idx_line_matches(&wrapped, "TMP", "2 m above ground")
                    && !idx_line_matches(&wrapped, "REFC", "entire atmosphere"),
                "colon-wrapped {wrapped:?} unexpectedly matched"
            );
        }
    }

    #[test]
    fn fetch_plan_hrrr_is_the_historical_two_file_pair() {
        use rustwx_core::ModelId;
        for (model, pressure, surface) in [
            (ModelId::Hrrr, "prs", "sfc"),
            (ModelId::HrrrAk, "prs", "sfc"),
            (ModelId::Aigfs, "pres", "sfc"),
            (ModelId::Aigefs, "pres/avg", "sfc/avg"),
            (ModelId::Hgefs, "pres/avg", "sfc/avg"),
        ] {
            let plan = fetch_plan(model).expect("split pressure/surface plan");
            assert_eq!(plan.len(), 2, "{model} fetches pressure + surface");
            // Order is load-bearing: pressure (prs) first, surface (sfc) second,
            // matching the historical fetch sequence.
            assert_eq!(plan[0].product, pressure);
            assert!(plan[0].pressure_source && !plan[0].surface_source);
            assert_eq!(plan[1].product, surface);
            assert!(plan[1].surface_source && !plan[1].pressure_source);
        }
    }

    #[test]
    fn fetch_plan_single_file_models_serve_both_roles() {
        use rustwx_core::ModelId;
        for (model, product) in [
            (ModelId::Rap, "awp130pgrb"),
            (ModelId::Gfs, "pgrb2.0p25"),
            (ModelId::Gdas, "pgrb2.0p25"),
            (ModelId::Gefs, "pgrb2ap5/gec00"),
            (ModelId::EcmwfOpenData, "oper"),
            (ModelId::Nam, "awip3d"),
        ] {
            let plan = fetch_plan(model).expect("single-file plan");
            assert_eq!(plan.len(), 1, "{model} fetches one product file");
            assert_eq!(plan[0].product, product);
            assert!(
                plan[0].surface_source && plan[0].pressure_source,
                "the one {model} file serves both the surface and pressure roles"
            );
        }
    }

    #[test]
    fn fetch_plan_rap_is_one_file_serving_both_roles() {
        use rustwx_core::ModelId;
        let plan = fetch_plan(ModelId::Rap).expect("RAP plan");
        assert_eq!(plan.len(), 1, "RAP fetches a single awp130pgrb file");
        assert_eq!(plan[0].product, "awp130pgrb");
        assert!(
            plan[0].surface_source && plan[0].pressure_source,
            "the one RAP file serves both the surface and pressure roles"
        );
        assert!(plan[0].idx_patterns.is_empty());
    }

    #[test]
    fn fetch_plan_rejects_unsupported_model() {
        use rustwx_core::ModelId;
        let err = fetch_plan(ModelId::Nbm).expect_err("NBM has no fetch plan");
        assert!(!err.is_cancelled());
        assert!(
            err.to_string().contains("no ingest fetch plan"),
            "got: {err}"
        );
    }

    /// The GFS fetch-plan token resolves to a well-formed AWS GRIB URL
    /// through the same `ModelRunRequest` -> `resolve_urls` path the ingest
    /// fetch uses. AWS is GFS source priority 2 (NOMADS is 1), so the test
    /// picks the AWS entry explicitly and asserts the exact archive URL.
    #[test]
    fn gfs_fetch_plan_token_resolves_a_well_formed_aws_url() {
        use rustwx_core::{CycleSpec, ModelId, ModelRunRequest, SourceId};
        let plan = fetch_plan(ModelId::Gfs).expect("GFS plan");
        let cycle = CycleSpec::new("20260414", 18).expect("valid cycle");
        let request =
            ModelRunRequest::new(ModelId::Gfs, cycle, 12, plan[0].product).expect("GFS request");
        let urls = rustwx_models::resolve_urls(&request).expect("GFS urls resolve");
        let aws = urls
            .iter()
            .find(|url| url.source == SourceId::Aws)
            .expect("AWS is a GFS source");
        assert_eq!(
            aws.grib_url,
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260414/18/atmos/gfs.t18z.pgrb2.0p25.f012"
        );
    }

    #[test]
    fn added_fetch_plan_tokens_resolve_well_formed_urls() {
        use rustwx_core::{CycleSpec, ModelId, ModelRunRequest, SourceId};
        let cycle = CycleSpec::new("20260414", 18).expect("valid cycle");
        let cases = [
            (
                ModelId::HrrrAk,
                6,
                "prs",
                SourceId::Aws,
                "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260414/alaska/hrrr.t18z.wrfprsf06.ak.grib2",
            ),
            (
                ModelId::Rap,
                6,
                "awp130pgrb",
                SourceId::Aws,
                "https://noaa-rap-pds.s3.amazonaws.com/rap.20260414/rap.t18z.awp130pgrbf06.grib2",
            ),
            (
                ModelId::Gdas,
                3,
                "pgrb2.0p25",
                SourceId::Aws,
                "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.20260414/18/atmos/gdas.t18z.pgrb2.0p25.f003",
            ),
            (
                ModelId::Gefs,
                6,
                "pgrb2ap5/gec00",
                SourceId::Aws,
                "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20260414/18/atmos/pgrb2ap5/gec00.t18z.pgrb2a.0p50.f006",
            ),
            (
                ModelId::Aigfs,
                6,
                "pres",
                SourceId::Nomads,
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod/aigfs.20260414/18/model/atmos/grib2/aigfs.t18z.pres.f006.grib2",
            ),
            (
                ModelId::Aigefs,
                6,
                "sfc/avg",
                SourceId::Nomads,
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/aigefs.20260414/18/ensstat/products/atmos/grib2/aigefs.t18z.sfc.avg.f006.grib2",
            ),
            (
                ModelId::Hgefs,
                6,
                "pres/avg",
                SourceId::Nomads,
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hgefs/prod/hgefs.20260414/18/ensstat/products/atmos/grib2/hgefs.t18z.pres.avg.f006.grib2",
            ),
            (
                ModelId::EcmwfOpenData,
                6,
                "oper",
                SourceId::Ecmwf,
                "https://data.ecmwf.int/forecasts/20260414/18z/ifs/0p25/oper/20260414180000-6h-oper-fc.grib2",
            ),
            (
                ModelId::Nam,
                6,
                "awip3d",
                SourceId::Aws,
                "https://noaa-nam-pds.s3.amazonaws.com/nam.20260414/nam.t18z.awip3d06.tm00.grib2",
            ),
        ];
        for (model, hour, product, source, expected) in cases {
            let request =
                ModelRunRequest::new(model, cycle.clone(), hour, product).expect("request");
            let urls = rustwx_models::resolve_urls(&request).expect("urls resolve");
            let resolved = urls
                .iter()
                .find(|url| url.source == source)
                .unwrap_or_else(|| panic!("{model} {source:?} URL missing"));
            assert_eq!(resolved.grib_url, expected);
        }
    }
}
