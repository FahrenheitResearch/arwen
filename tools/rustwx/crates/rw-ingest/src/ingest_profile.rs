//! Ingest profiles: WHICH subset of the full per-hour ingest plan a run
//! fetches, extracts, computes, and stores. Shared via `#[path]` inclusion
//! as a child of `ingest_hour` (the single inclusion point, so every bin
//! sees one set of types).
//!
//! A profile is the customization surface for model-data packs: volumes
//! (the 3D isobaric variables), the isobaric level step (25 or 50 hPa),
//! the 2D surface field set (everything, or a named subset), and the two
//! compute stages (derived, heavy). Three named presets exist:
//!
//! * `full` — today's default ingest, unchanged: all 5 volumes at 25 hPa
//!   steps, every 2D field (surface set + trailing 1 h windows + vorticity
//!   planes + direct-recipe isobaric planes), derived AND heavy stages.
//! * `sounding` — the point-sounding pack: all 5 volumes plus the 7 surface
//!   fields a sounding/hodograph view needs; no derived, no heavy, none of
//!   the render-grade 2D planes.
//! * `view` — the 2D map pack: every 2D field including the derived grids,
//!   NO volumes, no heavy.
//!
//! Validation happens HERE, not mid-ingest: the derived/heavy stages decode
//! their thermo inputs from the full surface + pressure files, so a profile
//! that stores only a named surface subset (and therefore skips the prs
//! 2D planes) excludes their inputs and must be rejected up front.

use rustwx_core::{CanonicalField, FieldSelector};

/// The two supported isobaric level steps (hPa) over the 100..=1000 range.
pub const LEVEL_STEPS_HPA: [u16; 2] = [25, 50];

/// One 3D isobaric variable choice, mapping to the stable store names the
/// full ingest has always written (`temperature_iso`, `dewpoint_iso`,
/// `u_iso`, `v_iso`, `height_iso`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeChoice {
    Temperature,
    Dewpoint,
    UWind,
    VWind,
    GeopotentialHeight,
}

impl VolumeChoice {
    /// Every volume, in the order the full ingest has always stored them.
    pub const ALL: [VolumeChoice; 5] = [
        VolumeChoice::Temperature,
        VolumeChoice::Dewpoint,
        VolumeChoice::UWind,
        VolumeChoice::VWind,
        VolumeChoice::GeopotentialHeight,
    ];

    /// The GRIB field this volume extracts.
    pub fn field(self) -> CanonicalField {
        match self {
            VolumeChoice::Temperature => CanonicalField::Temperature,
            VolumeChoice::Dewpoint => CanonicalField::Dewpoint,
            VolumeChoice::UWind => CanonicalField::UWind,
            VolumeChoice::VWind => CanonicalField::VWind,
            VolumeChoice::GeopotentialHeight => CanonicalField::GeopotentialHeight,
        }
    }

    /// The stable store variable name (dewpoint may fall back to `rh_iso`
    /// at ingest when the file realizes fewer than two dewpoint levels).
    pub fn store_name(self) -> &'static str {
        match self {
            VolumeChoice::Temperature => "temperature_iso",
            VolumeChoice::Dewpoint => "dewpoint_iso",
            VolumeChoice::UWind => "u_iso",
            VolumeChoice::VWind => "v_iso",
            VolumeChoice::GeopotentialHeight => "height_iso",
        }
    }
}

/// The 2D surface field set a profile stores: everything the full ingest
/// plan carries (surface plan + trailing 1 h windows + vorticity planes +
/// direct-recipe isobaric planes), or a named subset of the surface plan
/// (names from [`surface_plan`]; the prs-sourced planes and trailing
/// windows ride only with `All`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldSet {
    All,
    Named(Vec<String>),
}

/// One ingest profile: what to fetch/extract/compute/store per hour.
#[derive(Debug, Clone, PartialEq)]
pub struct IngestProfile {
    pub volumes: Vec<VolumeChoice>,
    /// Isobaric level step over 100..=1000 hPa: 25 (37 levels) or 50 (19).
    pub level_step_hpa: u16,
    pub surface_fields: FieldSet,
    /// Run the non-heavy derived compute stage (29 recipe grids).
    pub derived: bool,
    /// Run the heavy ECAPE compute stage (16 recipe grids).
    pub heavy: bool,
}

/// The 7 surface fields the `sounding` preset stores: the 2 m/10 m state +
/// pressure trio a sounding/hodograph view needs (orography anchors the
/// column AGL heights; mslp labels the chart).
pub const SOUNDING_SURFACE_FIELDS: [&str; 7] = [
    "temperature_2m",
    "dewpoint_2m",
    "u_10m",
    "v_10m",
    "surface_pressure",
    "mslp",
    "orography",
];

impl IngestProfile {
    /// Today's default ingest, unchanged: everything, both compute stages.
    pub fn full() -> Self {
        Self {
            volumes: VolumeChoice::ALL.to_vec(),
            level_step_hpa: 25,
            surface_fields: FieldSet::All,
            derived: true,
            heavy: true,
        }
    }

    /// The point-sounding pack: 5 volumes + 7 surface fields, no compute
    /// stages, no render-grade 2D planes.
    pub fn sounding() -> Self {
        Self {
            volumes: VolumeChoice::ALL.to_vec(),
            level_step_hpa: 25,
            surface_fields: FieldSet::Named(
                SOUNDING_SURFACE_FIELDS
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
            ),
            derived: false,
            heavy: false,
        }
    }

    /// The 2D map pack: every 2D field including derived grids, no
    /// volumes, no heavy stage.
    pub fn view() -> Self {
        Self {
            volumes: Vec::new(),
            level_step_hpa: 25,
            surface_fields: FieldSet::All,
            derived: true,
            heavy: false,
        }
    }

    /// Preset lookup by CLI name.
    pub fn preset(name: &str) -> Result<Self, String> {
        match name {
            "full" => Ok(Self::full()),
            "sounding" => Ok(Self::sounding()),
            "view" => Ok(Self::view()),
            other => Err(format!(
                "--profile: unknown preset '{other}' (expected full, sounding, or view)"
            )),
        }
    }

    /// Candidate isobaric levels (hPa) for every volume; absent levels are
    /// simply not stored (same partial-extraction behavior as always).
    pub fn candidate_levels(&self) -> Vec<u16> {
        (100..=1000)
            .step_by(usize::from(self.level_step_hpa))
            .collect()
    }

    /// Whether the profile stores the full 2D plan: the trailing 1 h window
    /// fields, the vorticity planes, and the direct-recipe isobaric planes
    /// ride only with `FieldSet::All`.
    pub fn includes_full_2d(&self) -> bool {
        matches!(self.surface_fields, FieldSet::All)
    }

    /// Whether one named surface-plan field is stored under this profile.
    pub fn includes_surface_field(&self, name: &str) -> bool {
        match &self.surface_fields {
            FieldSet::All => true,
            FieldSet::Named(names) => names.iter().any(|have| have == name),
        }
    }

    /// Whether one volume field is stored under this profile.
    pub fn includes_volume_field(&self, field: CanonicalField) -> bool {
        self.volumes.iter().any(|choice| choice.field() == field)
    }

    /// Whether the ingest needs the pressure ("prs") product file at all:
    /// volumes, the prs-sourced 2D planes, or either compute stage (their
    /// thermo decode reads the raw prs bytes).
    pub fn needs_prs(&self) -> bool {
        !self.volumes.is_empty() || self.includes_full_2d() || self.derived || self.heavy
    }

    /// Validate the profile as a whole. Rules:
    /// 1. `level_step_hpa` must be 25 or 50.
    /// 2. No duplicate volume choices.
    /// 3. A named surface set must be non-empty and name only known
    ///    surface-plan fields.
    /// 4. The derived/heavy stages need the full 2D surface set and the
    ///    prs file (their thermo pair decodes from both family files); a
    ///    named-subset profile excludes those inputs.
    /// 5. The heavy stage builds on the derived stage.
    pub fn validate(&self) -> Result<(), String> {
        if !LEVEL_STEPS_HPA.contains(&self.level_step_hpa) {
            return Err(format!(
                "profile: level step {} hPa is not supported (expected 25 or 50)",
                self.level_step_hpa
            ));
        }
        for (index, choice) in self.volumes.iter().enumerate() {
            if self.volumes[..index].contains(choice) {
                return Err(format!(
                    "profile: duplicate volume '{}'",
                    choice.store_name()
                ));
            }
        }
        if let FieldSet::Named(names) = &self.surface_fields {
            if names.is_empty() {
                return Err(
                    "profile: the named surface field set is empty; every hour needs at \
                     least one surface field to carry the grid"
                        .to_string(),
                );
            }
            for name in names {
                if !surface_plan().iter().any(|(have, _)| have == name) {
                    return Err(format!(
                        "profile: unknown surface field '{name}' (known fields: {})",
                        surface_plan()
                            .iter()
                            .map(|(have, _)| *have)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }
        if (self.derived || self.heavy) && !self.includes_full_2d() {
            return Err(
                "profile: the derived/heavy stages need the full 2D surface set and the \
                 prs file as compute inputs, but this profile stores only a named surface \
                 subset; use the full or view preset, or disable the compute stages"
                    .to_string(),
            );
        }
        if self.heavy && !self.derived {
            return Err(
                "profile: the heavy stage builds on the derived stage; enable derived \
                 (drop --no-derived) or disable heavy (--no-heavy)"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// One-line human description for CLI banners.
    pub fn describe(&self) -> String {
        let volumes = if self.volumes.is_empty() {
            "no volumes".to_string()
        } else {
            format!(
                "{} volume(s) @ {} hPa steps ({} levels)",
                self.volumes.len(),
                self.level_step_hpa,
                self.candidate_levels().len()
            )
        };
        let surface = match &self.surface_fields {
            FieldSet::All => "all 2D fields".to_string(),
            FieldSet::Named(names) => format!("{} named surface field(s)", names.len()),
        };
        format!(
            "{volumes}, {surface}, derived {}, heavy {}",
            if self.derived { "on" } else { "off" },
            if self.heavy { "on" } else { "off" },
        )
    }
}

/// CLI override flags applied on top of a preset (the composable surface:
/// `--profile NAME [--level-step N] [--no-derived] [--heavy|--no-heavy]`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ProfileOverrides {
    /// `--level-step N` (25 or 50).
    pub level_step_hpa: Option<u16>,
    /// `--no-derived`: skip the derived compute stage.
    pub no_derived: bool,
    /// `--heavy` (Some(true)) / `--no-heavy` (Some(false)) / neither (None).
    pub heavy: Option<bool>,
}

/// Resolve a preset name + override flags into a validated profile.
pub fn resolve_profile(
    preset: &str,
    overrides: &ProfileOverrides,
) -> Result<IngestProfile, String> {
    let mut profile = IngestProfile::preset(preset)?;
    if let Some(step) = overrides.level_step_hpa {
        if !LEVEL_STEPS_HPA.contains(&step) {
            return Err(format!(
                "--level-step: {step} hPa is not supported (expected 25 or 50)"
            ));
        }
        profile.level_step_hpa = step;
    }
    if overrides.no_derived {
        profile.derived = false;
    }
    if let Some(heavy) = overrides.heavy {
        profile.heavy = heavy;
    }
    profile
        .validate()
        .map_err(|err| format!("--profile {preset}: {err}"))?;
    Ok(profile)
}

/// 2D fields pulled from the surface ("sfc") product file, with their stable
/// store names. These mirror the selector constructors the rustwx-models
/// plot-recipe catalog uses for the same HRRR fields. (Moved here from
/// `ingest_hour` so profile validation and the ingest plan share one list.)
///
/// CAPE has no plan entry: it is sounding-derived here (no CAPE
/// CanonicalField) and ships through the derived precompute stage instead
/// (`sbcape`/`mlcape`/`mucape`... — see `compute_derived_grids`).
///
/// `apcp_run_total` is the plain TotalPrecipitation selection: the sfc file
/// carries two APCP accumulations that both end at hour h (0->h run total
/// and the trailing (h-1)->h hour); they tie on match score and the run
/// total wins as first in file order. The trailing 1 h window is stored
/// separately as `apcp_1h` via a dedicated re-select (see `ingest_hour`).
///
/// Lightning flash density is deliberately absent: rustwx-io has no
/// structured selector for it (HRRR exposes LTNG, a non-dimensional
/// lightning flag, and LTNGSD strike density — not flash density), and the
/// recipe catalog blocks the slug for HRRR for the same mislabeling reason.
pub fn surface_plan() -> Vec<(&'static str, FieldSelector)> {
    vec![
        (
            "temperature_2m",
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
        ),
        (
            "dewpoint_2m",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
        ),
        (
            "u_10m",
            FieldSelector::height_agl(CanonicalField::UWind, 10),
        ),
        (
            "v_10m",
            FieldSelector::height_agl(CanonicalField::VWind, 10),
        ),
        (
            "composite_reflectivity",
            FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity),
        ),
        (
            "mslp",
            FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        ),
        // --- surface state & moisture (feeds SurfaceInputs-derived products) ---
        (
            "rh_2m",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2),
        ),
        (
            "wind_gust_10m",
            FieldSelector::height_agl(CanonicalField::WindGust, 10),
        ),
        (
            "surface_pressure",
            FieldSelector::surface(CanonicalField::Pressure),
        ),
        (
            "orography",
            FieldSelector::surface(CanonicalField::GeopotentialHeight),
        ),
        // --- precipitation & precip type ---
        (
            "apcp_run_total",
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
        ),
        (
            "categorical_rain",
            FieldSelector::surface(CanonicalField::CategoricalRain),
        ),
        (
            "categorical_freezing_rain",
            FieldSelector::surface(CanonicalField::CategoricalFreezingRain),
        ),
        (
            "categorical_ice_pellets",
            FieldSelector::surface(CanonicalField::CategoricalIcePellets),
        ),
        (
            "categorical_snow",
            FieldSelector::surface(CanonicalField::CategoricalSnow),
        ),
        // --- moisture column, clouds, visibility ---
        (
            "pwat",
            FieldSelector::entire_atmosphere(CanonicalField::PrecipitableWater),
        ),
        (
            "cloud_cover_low",
            FieldSelector::entire_atmosphere(CanonicalField::LowCloudCover),
        ),
        (
            "cloud_cover_mid",
            FieldSelector::entire_atmosphere(CanonicalField::MiddleCloudCover),
        ),
        (
            "cloud_cover_high",
            FieldSelector::entire_atmosphere(CanonicalField::HighCloudCover),
        ),
        (
            "cloud_cover_total",
            FieldSelector::entire_atmosphere(CanonicalField::TotalCloudCover),
        ),
        (
            "visibility",
            FieldSelector::surface(CanonicalField::Visibility),
        ),
        // --- convection, smoke, satellite (also in wrfnat; sfc carries them
        //     too, so they ride this fetch — see the ingest_hour module doc) ---
        (
            "reflectivity_1km",
            FieldSelector::height_agl(CanonicalField::RadarReflectivity, 1000),
        ),
        (
            "uh_2to5km",
            FieldSelector::height_layer_agl(CanonicalField::UpdraftHelicity, 2000, 5000),
        ),
        (
            "smoke_8m",
            FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8),
        ),
        (
            "smoke_column",
            FieldSelector::entire_atmosphere(CanonicalField::ColumnIntegratedSmoke),
        ),
        (
            "simulated_ir",
            FieldSelector::nominal_top(CanonicalField::SimulatedInfraredBrightnessTemperature),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_preset_is_todays_behavior() {
        let full = IngestProfile::full();
        assert_eq!(full.volumes, VolumeChoice::ALL.to_vec());
        assert_eq!(full.level_step_hpa, 25);
        assert_eq!(full.surface_fields, FieldSet::All);
        assert!(full.derived && full.heavy);
        assert_eq!(full.candidate_levels().len(), 37, "100..=1000 step 25");
        assert_eq!(full.candidate_levels()[0], 100);
        assert_eq!(*full.candidate_levels().last().unwrap(), 1000);
        full.validate().expect("full preset validates");
    }

    #[test]
    fn sounding_preset_stores_volumes_and_seven_surface_fields() {
        let sounding = IngestProfile::sounding();
        assert_eq!(sounding.volumes.len(), 5);
        assert!(!sounding.derived && !sounding.heavy);
        assert!(!sounding.includes_full_2d());
        for name in SOUNDING_SURFACE_FIELDS {
            assert!(
                sounding.includes_surface_field(name),
                "sounding must include '{name}'"
            );
        }
        assert!(!sounding.includes_surface_field("composite_reflectivity"));
        sounding.validate().expect("sounding preset validates");
    }

    #[test]
    fn view_preset_is_all_2d_no_volumes_no_heavy() {
        let view = IngestProfile::view();
        assert!(view.volumes.is_empty());
        assert!(view.includes_full_2d());
        assert!(view.derived && !view.heavy);
        assert!(view.needs_prs(), "view still needs prs for the 2D planes");
        view.validate().expect("view preset validates");
    }

    #[test]
    fn every_sounding_surface_field_is_a_known_plan_name() {
        let plan = surface_plan();
        for name in SOUNDING_SURFACE_FIELDS {
            assert!(
                plan.iter().any(|(have, _)| *have == name),
                "'{name}' must exist in surface_plan()"
            );
        }
    }

    #[test]
    fn level_step_50_yields_19_levels() {
        let mut profile = IngestProfile::sounding();
        profile.level_step_hpa = 50;
        let levels = profile.candidate_levels();
        assert_eq!(levels.len(), 19);
        assert_eq!(levels[0], 100);
        assert_eq!(*levels.last().unwrap(), 1000);
        assert!(levels.windows(2).all(|pair| pair[1] - pair[0] == 50));
    }

    #[test]
    fn validate_rejects_bad_level_step() {
        let mut profile = IngestProfile::full();
        profile.level_step_hpa = 10;
        let message = profile.validate().unwrap_err();
        assert!(message.contains("10 hPa"), "got: {message}");
    }

    #[test]
    fn validate_rejects_duplicate_volumes() {
        let mut profile = IngestProfile::full();
        profile.volumes.push(VolumeChoice::Temperature);
        let message = profile.validate().unwrap_err();
        assert!(message.contains("duplicate volume"), "got: {message}");
        assert!(message.contains("temperature_iso"), "got: {message}");
    }

    #[test]
    fn validate_rejects_empty_named_set_and_unknown_names() {
        let mut profile = IngestProfile::sounding();
        profile.surface_fields = FieldSet::Named(Vec::new());
        let message = profile.validate().unwrap_err();
        assert!(message.contains("empty"), "got: {message}");

        profile.surface_fields = FieldSet::Named(vec!["not_a_field".to_string()]);
        let message = profile.validate().unwrap_err();
        assert!(
            message.contains("unknown surface field 'not_a_field'"),
            "got: {message}"
        );
    }

    #[test]
    fn validate_rejects_derived_or_heavy_on_a_named_subset() {
        let mut profile = IngestProfile::sounding();
        profile.heavy = true;
        let message = profile.validate().unwrap_err();
        assert!(
            message.contains("named surface subset"),
            "heavy on sounding must name the excluded inputs, got: {message}"
        );

        let mut profile = IngestProfile::sounding();
        profile.derived = true;
        assert!(profile.validate().is_err());
    }

    #[test]
    fn validate_rejects_heavy_without_derived() {
        let mut profile = IngestProfile::full();
        profile.derived = false;
        let message = profile.validate().unwrap_err();
        assert!(
            message.contains("heavy stage builds on the derived stage"),
            "got: {message}"
        );
    }

    #[test]
    fn resolve_profile_applies_overrides() {
        let profile = resolve_profile(
            "sounding",
            &ProfileOverrides {
                level_step_hpa: Some(50),
                ..Default::default()
            },
        )
        .expect("sounding @ 50 resolves");
        assert_eq!(profile.level_step_hpa, 50);
        assert_eq!(profile.candidate_levels().len(), 19);

        let profile = resolve_profile(
            "full",
            &ProfileOverrides {
                heavy: Some(false),
                ..Default::default()
            },
        )
        .expect("full --no-heavy resolves");
        assert!(profile.derived && !profile.heavy);

        // --no-derived alone on full leaves heavy dangling: clear error.
        let message = resolve_profile(
            "full",
            &ProfileOverrides {
                no_derived: true,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(message.contains("--no-heavy"), "got: {message}");

        // --no-derived --no-heavy composes into a plain-extraction full.
        let profile = resolve_profile(
            "full",
            &ProfileOverrides {
                no_derived: true,
                heavy: Some(false),
                ..Default::default()
            },
        )
        .expect("full --no-derived --no-heavy resolves");
        assert!(!profile.derived && !profile.heavy);
    }

    #[test]
    fn resolve_profile_rejects_heavy_on_sounding_with_a_clear_error() {
        let message = resolve_profile(
            "sounding",
            &ProfileOverrides {
                heavy: Some(true),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(message.contains("--profile sounding"), "got: {message}");
        assert!(message.contains("named surface subset"), "got: {message}");
    }

    #[test]
    fn resolve_profile_rejects_unknown_preset_and_bad_step() {
        let message = resolve_profile("everything", &ProfileOverrides::default()).unwrap_err();
        assert!(
            message.contains("unknown preset 'everything'"),
            "got: {message}"
        );

        let message = resolve_profile(
            "full",
            &ProfileOverrides {
                level_step_hpa: Some(30),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(message.contains("--level-step"), "got: {message}");
    }

    #[test]
    fn describe_names_the_shape() {
        assert_eq!(
            IngestProfile::full().describe(),
            "5 volume(s) @ 25 hPa steps (37 levels), all 2D fields, derived on, heavy on"
        );
        assert_eq!(
            IngestProfile::view().describe(),
            "no volumes, all 2D fields, derived on, heavy off"
        );
        let mut sounding = IngestProfile::sounding();
        sounding.level_step_hpa = 50;
        assert_eq!(
            sounding.describe(),
            "5 volume(s) @ 50 hPa steps (19 levels), 7 named surface field(s), derived off, heavy off"
        );
    }
}
