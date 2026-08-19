//! Declarative source semantics for arbitrary meteorological datasets.
//!
//! A mapping is deliberately richer than a WPS Vtable: it binds data selectors
//! to units, axes, staggering, missing-data behavior, and a target initialization
//! contract. WPS Vtables can only be transcribed into hash-bound,
//! non-executable row drafts; their missing semantics are reported rather than
//! guessed silently.

use grib_core::grib1::Grib1File;
use grib_core::grib2::Grib2File;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::{RwWpsError, io_error};

pub const MAPPING_SCHEMA: &str = "rw-wps.mapping.v1";
pub const INSPECTION_SCHEMA: &str = "rw-wps.source-inspection.v1";
pub const INVENTORY_SCHEMA: &str = "rw-wps.source-inventory.v1";
pub const VALIDATION_SCHEMA: &str = "rw-wps.mapping-validation.v1";
pub const VTABLE_ROW_DRAFT_SCHEMA: &str = "rw-wps.vtable-row-draft.v1";

/// Non-executable transcription of a classic WPS Vtable.
///
/// This deliberately does not share the [`NativeMapping`] schema. A Vtable
/// does not define enough scientific semantics to author an executable
/// mapping, so this document only preserves its rows and identifies decisions
/// that an explicit descriptor still has to make.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VtableRowDraft {
    pub schema: String,
    pub name: String,
    pub executable: bool,
    pub source: VtableSourceBinding,
    pub rows: Vec<VtableDraftRow>,
    pub conflicts: Vec<VtableDraftConflict>,
    pub unresolved: Vec<VtableDraftIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VtableSourceBinding {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VtableDraftRow {
    pub id: String,
    pub line: usize,
    #[serde(default)]
    pub grib1: Option<VtableGrib1Draft>,
    #[serde(default)]
    pub grib2: Option<VtableGrib2Draft>,
    pub metgrid_name: String,
    pub units: String,
    pub description: String,
    #[serde(default)]
    pub extra_columns: Vec<String>,
    #[serde(default)]
    pub conflict_ids: Vec<String>,
    pub unresolved: Vec<VtableDraftIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VtableGrib1Draft {
    pub parameter: String,
    pub level_type: String,
    pub level1: String,
    pub level2: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VtableGrib2Draft {
    pub discipline: String,
    pub category: String,
    pub parameter: String,
    pub level_type: String,
    pub level1: String,
    pub level2: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VtableDraftConflict {
    pub id: String,
    pub kind: String,
    pub metgrid_name: String,
    pub row_ids: Vec<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VtableDraftIssue {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SourceFormat {
    Grib1,
    Grib2,
    Netcdf,
}

impl SourceFormat {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Grib1 => "grib1",
            Self::Grib2 => "grib2",
            Self::Netcdf => "netcdf",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NativeMapping {
    pub schema: String,
    pub name: String,
    pub format: SourceFormat,
    pub coordinates: CoordinateMapping,
    pub fields: BTreeMap<String, FieldMapping>,
    #[serde(default)]
    pub derivations: Vec<NamedDerivation>,
    pub target: TargetContract,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CoordinateMapping {
    pub horizontal: HorizontalCoordinates,
    pub vertical: VerticalCoordinate,
    pub time: TimeCoordinate,
    #[serde(default)]
    pub member: Option<MemberCoordinate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HorizontalCoordinates {
    EmbeddedGrid,
    Variables {
        latitude: VariableSelector,
        longitude: VariableSelector,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VerticalCoordinate {
    pub kind: VerticalKind,
    #[serde(default)]
    pub selector: Option<VariableSelector>,
    pub units: String,
    #[serde(default)]
    pub positive: Option<PositiveDirection>,
    #[serde(default)]
    pub levels: Vec<f64>,
    #[serde(default)]
    pub hybrid_a_field: Option<String>,
    #[serde(default)]
    pub hybrid_b_field: Option<String>,
    #[serde(default)]
    pub surface_pressure_field: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerticalKind {
    Pressure,
    HybridSigmaPressure,
    ModelLevel,
    Height,
    SoilDepth,
    EmbeddedLevels,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PositiveDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TimeCoordinate {
    EmbeddedMetadata,
    Dimension {
        selector: DimensionSelector,
        units: String,
        #[serde(default)]
        calendar: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MemberCoordinate {
    EmbeddedMetadata,
    Dimension { selector: DimensionSelector },
}

/// A declared source name: either one exact name or an ordered list of
/// alternates. gpuwm's live mapping authorities spell provider renames as
/// alternates (e.g. ERA5 `time`/`valid_time`, `level`/`pressure_level`), so a
/// scalar-only schema hard-fails on the very files it exists to describe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum NameSpec {
    One(String),
    Alternates(Vec<String>),
}

impl NameSpec {
    pub fn candidates(&self) -> &[String] {
        match self {
            Self::One(name) => std::slice::from_ref(name),
            Self::Alternates(names) => names,
        }
    }

    pub fn matches(&self, candidate: &str) -> bool {
        self.candidates().iter().any(|name| name == candidate)
    }

    /// True when no candidate can ever match: empty list or all-empty names.
    pub fn is_unmatchable(&self) -> bool {
        self.candidates().iter().all(|name| name.is_empty())
    }
}

impl From<String> for NameSpec {
    fn from(value: String) -> Self {
        Self::One(value)
    }
}

impl From<&str> for NameSpec {
    fn from(value: &str) -> Self {
        Self::One(value.to_owned())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DimensionSelector {
    #[serde(default)]
    pub name: Option<NameSpec>,
    #[serde(default)]
    pub standard_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "format", rename_all = "lowercase", deny_unknown_fields)]
pub enum VariableSelector {
    Grib1 {
        parameter: u8,
        #[serde(default)]
        table_version: Option<u8>,
        #[serde(default)]
        center: Option<u8>,
        #[serde(default)]
        level_type: Option<u8>,
        #[serde(default)]
        level_value: Option<f64>,
    },
    Grib2 {
        discipline: u8,
        category: u8,
        parameter: u8,
        #[serde(default)]
        center: Option<u16>,
        #[serde(default)]
        subcenter: Option<u16>,
        #[serde(default)]
        master_table_version: Option<u8>,
        #[serde(default)]
        local_table_version: Option<u8>,
        #[serde(default)]
        level_type: Option<u8>,
        #[serde(default)]
        level_value: Option<f64>,
        #[serde(default)]
        second_level_type: Option<u8>,
        #[serde(default)]
        second_level_value: Option<f64>,
        #[serde(default)]
        member: Option<u8>,
    },
    Netcdf {
        #[serde(default)]
        name: Option<NameSpec>,
        #[serde(default)]
        standard_name: Option<String>,
    },
}

impl VariableSelector {
    fn format(&self) -> SourceFormat {
        match self {
            Self::Grib1 { .. } => SourceFormat::Grib1,
            Self::Grib2 { .. } => SourceFormat::Grib2,
            Self::Netcdf { .. } => SourceFormat::Netcdf,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Grib1 {
                parameter,
                table_version,
                center,
                level_type,
                level_value,
            } => format!(
                "GRIB1 parameter={parameter} table={table_version:?} center={center:?} level_type={level_type:?} level={level_value:?}"
            ),
            Self::Grib2 {
                discipline,
                category,
                parameter,
                center,
                subcenter,
                master_table_version,
                local_table_version,
                level_type,
                level_value,
                second_level_type,
                second_level_value,
                member,
            } => format!(
                "GRIB2 discipline={discipline} category={category} parameter={parameter} center={center:?} subcenter={subcenter:?} master_table={master_table_version:?} local_table={local_table_version:?} level_type={level_type:?} level={level_value:?} second_level_type={second_level_type:?} second_level={second_level_value:?} member={member:?}"
            ),
            Self::Netcdf {
                name,
                standard_name,
            } => format!("NetCDF name={name:?} standard_name={standard_name:?}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FieldMapping {
    #[serde(default)]
    pub selectors: Vec<VariableSelector>,
    #[serde(default)]
    pub derivation: Option<String>,
    #[serde(default)]
    pub selector_stack_axis: Option<AxisRole>,
    pub units: UnitTransform,
    pub source_axes: Vec<AxisRole>,
    pub target_axes: Vec<AxisRole>,
    pub location: GridLocation,
    #[serde(default)]
    pub staggering: Staggering,
    pub missing: MissingPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UnitTransform {
    pub source: String,
    pub target: String,
    #[serde(default = "one")]
    pub scale: f64,
    #[serde(default)]
    pub offset: f64,
}

const fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum AxisRole {
    Time,
    Member,
    Vertical,
    Y,
    X,
    Soil,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GridLocation {
    Mass,
    UFace,
    VFace,
    Surface,
    Soil,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Staggering {
    #[default]
    None,
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MissingPolicy {
    Reject,
    Attribute { name: String },
    Value { value: f64 },
    PreserveMask,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
// `deny_unknown_fields` cannot be placed on a container that uses `flatten`:
// serde would reject the flattened `operation` tag before `Derivation` sees
// it. `Derivation` itself denies unknown fields, so the combined JSON object
// remains strict (covered by the round-trip and negative tests below).
pub struct NamedDerivation {
    pub name: String,
    #[serde(flatten)]
    pub operation: Derivation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Derivation {
    Copy {
        source: String,
    },
    WindSpeed {
        u: String,
        v: String,
    },
    SpecificHumidityFromRh {
        relative_humidity: String,
        temperature: String,
        pressure: String,
    },
    RelativeHumidityFromDewpoint {
        dewpoint: String,
        temperature: String,
    },
    GeopotentialHeight {
        geopotential: String,
        #[serde(default = "standard_gravity")]
        gravity_m_s2: f64,
    },
    PressureFromVerticalCoordinate,
    SpecificHumidityFromDewpoint {
        dewpoint: String,
        temperature: String,
        pressure: String,
    },
}

const fn standard_gravity() -> f64 {
    9.80665
}

impl Derivation {
    fn dependencies(&self) -> Vec<&str> {
        match self {
            Self::Copy { source } => vec![source],
            Self::WindSpeed { u, v } => vec![u, v],
            Self::SpecificHumidityFromRh {
                relative_humidity,
                temperature,
                pressure,
            } => vec![relative_humidity, temperature, pressure],
            Self::RelativeHumidityFromDewpoint {
                dewpoint,
                temperature,
            } => vec![dewpoint, temperature],
            Self::GeopotentialHeight { geopotential, .. } => vec![geopotential],
            Self::PressureFromVerticalCoordinate => Vec::new(),
            Self::SpecificHumidityFromDewpoint {
                dewpoint,
                temperature,
                pressure,
            } => vec![dewpoint, temperature, pressure],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetContract {
    pub name: String,
    pub physics_suite: String,
    pub max_dom: u16,
    pub require_lateral_boundaries: bool,
    #[serde(default)]
    pub target_vertical_levels: Option<u16>,
    #[serde(default)]
    pub soil_layer_count: Option<u16>,
    #[serde(default)]
    pub boundary_interval_seconds: Option<u32>,
    pub required_fields: Vec<FieldRequirement>,
    #[serde(default = "default_pressure_requirement")]
    pub pressure_requirement: PressureRequirement,
    #[serde(default)]
    pub policy_controlled_fields: Vec<String>,
    #[serde(default)]
    pub initialization_policies: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PressureRequirement {
    AirPressure,
    HybridCoordinate,
    AirPressureOrHybridCoordinate,
}

const fn default_pressure_requirement() -> PressureRequirement {
    PressureRequirement::AirPressureOrHybridCoordinate
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FieldRequirement {
    pub name: String,
    pub axes: Vec<AxisRole>,
    pub location: GridLocation,
    pub target_units: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: String,
    #[serde(default)]
    pub field: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub schema: String,
    pub verdict: String,
    pub errors: Vec<Diagnostic>,
    pub warnings: Vec<Diagnostic>,
}

impl ValidationReport {
    pub fn passed(&self) -> bool {
        self.errors.is_empty()
    }
}

pub fn read_mapping(path: &Path) -> Result<NativeMapping, RwWpsError> {
    let bytes = fs::read(path).map_err(|error| io_error(path, error))?;
    serde_json::from_slice(&bytes).map_err(RwWpsError::from)
}

pub fn validate_mapping(mapping: &NativeMapping) -> ValidationReport {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut error = |code: &str, field: Option<&str>, message: String| {
        errors.push(Diagnostic {
            severity: Severity::Error,
            code: code.to_owned(),
            field: field.map(str::to_owned),
            message,
        });
    };

    if mapping.schema != MAPPING_SCHEMA {
        error(
            "schema_mismatch",
            None,
            format!(
                "schema must be {MAPPING_SCHEMA:?}, got {:?}",
                mapping.schema
            ),
        );
    }
    if mapping.name.trim().is_empty() {
        error(
            "empty_name",
            None,
            "mapping name must not be empty".to_owned(),
        );
    }
    if mapping.target.max_dom == 0 {
        error(
            "invalid_domain_contract",
            None,
            "target.max_dom must be positive".to_owned(),
        );
    }
    if mapping.target.target_vertical_levels == Some(0) {
        error(
            "invalid_target_vertical_levels",
            None,
            "target_vertical_levels must be positive".to_owned(),
        );
    } else if mapping.target.target_vertical_levels.is_none() {
        error(
            "target_vertical_levels_missing",
            None,
            "bind target_vertical_levels to the authoritative namelist/domain contract".to_owned(),
        );
    }
    if mapping.target.soil_layer_count == Some(0) {
        error(
            "invalid_soil_layer_count",
            None,
            "soil_layer_count must be positive".to_owned(),
        );
    } else if mapping.target.soil_layer_count.is_none() {
        error(
            "soil_layer_count_missing",
            None,
            "bind soil_layer_count to the selected land-surface physics".to_owned(),
        );
    }
    if mapping.target.require_lateral_boundaries {
        match mapping.target.boundary_interval_seconds {
            Some(value) if value > 0 => {}
            _ => error(
                "boundary_interval_missing",
                None,
                "lateral boundaries require a positive boundary_interval_seconds".to_owned(),
            ),
        }
    }

    validate_coordinate_selectors(mapping, &mut error);

    let derivations = mapping
        .derivations
        .iter()
        .map(|item| (item.name.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    if derivations.len() != mapping.derivations.len() {
        error(
            "duplicate_derivation",
            None,
            "derivation names must be unique".to_owned(),
        );
    }

    for (name, field) in &mapping.fields {
        let direct = !field.selectors.is_empty();
        let derived = field.derivation.is_some();
        if direct == derived {
            error(
                "ambiguous_field_source",
                Some(name),
                "declare either one-or-more selectors or one derivation, but not both/neither"
                    .to_owned(),
            );
        }
        for selector in &field.selectors {
            validate_selector(mapping.format, selector, name, &mut error);
        }
        if let Some(stack_axis) = field.selector_stack_axis {
            if mapping.format != SourceFormat::Netcdf || !direct {
                error(
                    "selector_stack_axis_source",
                    Some(name),
                    "selector_stack_axis requires a direct NetCDF field".to_owned(),
                );
            }
            if stack_axis != AxisRole::Soil {
                error(
                    "selector_stack_axis_unsupported",
                    Some(name),
                    "selector_stack_axis currently supports only soil".to_owned(),
                );
            }
            if field.selectors.len() < 2 {
                error(
                    "selector_stack_axis_cardinality",
                    Some(name),
                    "selector_stack_axis requires multiple selectors".to_owned(),
                );
            }
            if !field.source_axes.contains(&stack_axis) {
                error(
                    "selector_stack_axis_missing_from_source_axes",
                    Some(name),
                    "selector_stack_axis is absent from source_axes".to_owned(),
                );
            }
        }
        if let Some(derivation) = field.derivation.as_deref() {
            if !derivations.contains_key(derivation) {
                error(
                    "unknown_derivation",
                    Some(name),
                    format!("derivation {derivation:?} is not declared"),
                );
            }
        }
        if field.units.source.trim().is_empty() || field.units.target.trim().is_empty() {
            error(
                "missing_units",
                Some(name),
                "source and target units must both be explicit".to_owned(),
            );
        }
        if !field.units.scale.is_finite() || !field.units.offset.is_finite() {
            error(
                "nonfinite_conversion",
                Some(name),
                "unit scale and offset must be finite".to_owned(),
            );
        }
        if field.source_axes.is_empty() || field.target_axes.is_empty() {
            error(
                "missing_axes",
                Some(name),
                "source_axes and target_axes must both be explicit".to_owned(),
            );
        }
        for (label, axes) in [
            ("source_axes", &field.source_axes),
            ("target_axes", &field.target_axes),
        ] {
            let unique_axes = axes.iter().collect::<BTreeSet<_>>();
            if unique_axes.len() != axes.len() {
                error(
                    "duplicate_axis",
                    Some(name),
                    format!("{label} contains a duplicate: {axes:?}"),
                );
            }
        }
        if let MissingPolicy::Value { value } = field.missing {
            if !value.is_finite() {
                error(
                    "nonfinite_fill",
                    Some(name),
                    "missing-data fill value must be finite".to_owned(),
                );
            }
        }
        if let MissingPolicy::Attribute { name: attribute } = &field.missing {
            if mapping.format != SourceFormat::Netcdf {
                error(
                    "attribute_missing_policy_format",
                    Some(name),
                    "attribute-based missing data is only valid for NetCDF sources".to_owned(),
                );
            }
            if attribute.trim().is_empty() {
                error(
                    "empty_missing_attribute",
                    Some(name),
                    "missing-data attribute name must not be empty".to_owned(),
                );
            }
        }
        if matches!(field.missing, MissingPolicy::PreserveMask)
            && field.location != GridLocation::Soil
        {
            error(
                "preserve_mask_location",
                Some(name),
                "preserve_mask is restricted to soil fields repaired by the land/water-aware initializer"
                    .to_owned(),
            );
        }
    }

    for item in &mapping.derivations {
        if item.name.trim().is_empty() {
            error(
                "empty_derivation_name",
                None,
                "derivation name must not be empty".to_owned(),
            );
        }
        for dependency in item.operation.dependencies() {
            if !mapping.fields.contains_key(dependency) {
                error(
                    "missing_derivation_dependency",
                    Some(&item.name),
                    format!("dependency {dependency:?} is not a mapped field"),
                );
            }
        }
        if let Derivation::GeopotentialHeight { gravity_m_s2, .. } = &item.operation {
            if !gravity_m_s2.is_finite() || *gravity_m_s2 <= 0.0 {
                error(
                    "invalid_gravity",
                    Some(&item.name),
                    "gravity_m_s2 must be finite and positive".to_owned(),
                );
            }
        }
        if matches!(item.operation, Derivation::PressureFromVerticalCoordinate)
            && mapping.coordinates.vertical.kind != VerticalKind::Pressure
        {
            error(
                "pressure_derivation_coordinate",
                Some(&item.name),
                "pressure_from_vertical_coordinate requires a pressure vertical coordinate"
                    .to_owned(),
            );
        }
    }

    if let Some(cycle) = derivation_cycle(mapping, &derivations) {
        error(
            "derivation_cycle",
            cycle.first().map(String::as_str),
            format!("derived fields form a cycle: {}", cycle.join(" -> ")),
        );
    }

    let target_requirements = mapping
        .target
        .required_fields
        .iter()
        .map(|requirement| (requirement.name.as_str(), requirement))
        .collect::<BTreeMap<_, _>>();
    for expected in canonical_wrf_requirements() {
        match target_requirements.get(expected.name.as_str()) {
            None => error(
                "target_contract_missing_canonical_requirement",
                Some(&expected.name),
                "WRF initialization contracts may add physics requirements but cannot remove the canonical source-frame minimum"
                    .to_owned(),
            ),
            Some(actual) if **actual != expected => error(
                "target_contract_changes_canonical_requirement",
                Some(&expected.name),
                format!(
                    "canonical requirement must remain axes={:?} location={:?} units={:?}",
                    expected.axes, expected.location, expected.target_units
                ),
            ),
            Some(_) => {}
        }
    }
    let target_policy_fields = mapping
        .target
        .policy_controlled_fields
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for canonical in canonical_policy_fields() {
        if !target_policy_fields.contains(canonical.as_str()) {
            error(
                "target_contract_missing_canonical_policy",
                Some(&canonical),
                "WRF initialization contracts cannot omit a canonical policy-controlled field"
                    .to_owned(),
            );
        }
    }

    let mut requirement_names = BTreeSet::new();
    for requirement in &mapping.target.required_fields {
        if !requirement_names.insert(requirement.name.as_str()) {
            error(
                "duplicate_requirement",
                Some(&requirement.name),
                "target contract contains the same required field more than once".to_owned(),
            );
        }
        let Some(field) = mapping.fields.get(&requirement.name) else {
            error(
                "required_field_missing",
                Some(&requirement.name),
                format!(
                    "required by target physics/domain contract {:?} (suite {:?}, max_dom={})",
                    mapping.target.name, mapping.target.physics_suite, mapping.target.max_dom
                ),
            );
            continue;
        };
        if field.target_axes != requirement.axes {
            error(
                "required_field_axes",
                Some(&requirement.name),
                format!(
                    "expected target axes {:?}, got {:?}",
                    requirement.axes, field.target_axes
                ),
            );
        }
        if field.location != requirement.location {
            error(
                "required_field_location",
                Some(&requirement.name),
                format!(
                    "expected location {:?}, got {:?}",
                    requirement.location, field.location
                ),
            );
        }
        if field.units.target != requirement.target_units {
            error(
                "required_field_units",
                Some(&requirement.name),
                format!(
                    "target contract requires units {:?}, mapping produces {:?}",
                    requirement.target_units, field.units.target
                ),
            );
        }
    }

    let has_air_pressure = mapping.fields.contains_key("air_pressure");
    let has_hybrid = mapping.coordinates.vertical.kind == VerticalKind::HybridSigmaPressure;
    let pressure_satisfied = match mapping.target.pressure_requirement {
        PressureRequirement::AirPressure => has_air_pressure,
        PressureRequirement::HybridCoordinate => has_hybrid,
        PressureRequirement::AirPressureOrHybridCoordinate => has_air_pressure || has_hybrid,
    };
    if !pressure_satisfied {
        error(
            "pressure_requirement_missing",
            Some("air_pressure"),
            format!(
                "target contract requires {:?}; map canonical air_pressure or declare a complete hybrid coordinate as applicable",
                mapping.target.pressure_requirement
            ),
        );
    }

    let mut controlled_names = BTreeSet::new();
    for field in &mapping.target.policy_controlled_fields {
        if !controlled_names.insert(field.as_str()) {
            error(
                "duplicate_policy_controlled_field",
                Some(field),
                "policy-controlled field is listed more than once".to_owned(),
            );
        }
        if !mapping.fields.contains_key(field)
            && mapping
                .target
                .initialization_policies
                .get(field)
                .is_none_or(|policy| policy.trim().is_empty())
        {
            error(
                "initialization_policy_missing",
                Some(field),
                "field is absent, so the target contract requires an explicit initialization policy"
                    .to_owned(),
            );
        }
    }
    for (field, policy) in &mapping.target.initialization_policies {
        if policy.trim().is_empty() {
            error(
                "empty_initialization_policy",
                Some(field),
                "initialization policy must not be empty".to_owned(),
            );
        }
    }

    if matches!(
        mapping.coordinates.vertical.kind,
        VerticalKind::HybridSigmaPressure
    ) {
        for (label, value) in [
            (
                "hybrid_a_field",
                &mapping.coordinates.vertical.hybrid_a_field,
            ),
            (
                "hybrid_b_field",
                &mapping.coordinates.vertical.hybrid_b_field,
            ),
            (
                "surface_pressure_field",
                &mapping.coordinates.vertical.surface_pressure_field,
            ),
        ] {
            if value.is_none() {
                error(
                    "incomplete_hybrid_coordinate",
                    None,
                    format!("vertical.{label} is required for hybrid_sigma_pressure"),
                );
            }
        }
    }
    if mapping.coordinates.vertical.units.trim().is_empty() {
        error(
            "vertical_units_missing",
            None,
            "vertical coordinate units must be explicit".to_owned(),
        );
    }
    if mapping
        .coordinates
        .vertical
        .levels
        .iter()
        .any(|value| !value.is_finite())
    {
        error(
            "nonfinite_vertical_level",
            None,
            "explicit vertical levels must all be finite".to_owned(),
        );
    }
    let unique_levels = mapping
        .coordinates
        .vertical
        .levels
        .iter()
        .map(|value| value.to_bits())
        .collect::<BTreeSet<_>>();
    if unique_levels.len() != mapping.coordinates.vertical.levels.len() {
        error(
            "duplicate_vertical_level",
            None,
            "explicit vertical levels must be unique".to_owned(),
        );
    }
    if mapping.fields.is_empty() {
        warnings.push(Diagnostic {
            severity: Severity::Warning,
            code: "empty_mapping".to_owned(),
            field: None,
            message: "no data fields are mapped".to_owned(),
        });
    }

    ValidationReport {
        schema: VALIDATION_SCHEMA.to_owned(),
        verdict: if errors.is_empty() { "PASS" } else { "FAIL" }.to_owned(),
        errors,
        warnings,
    }
}

fn derivation_cycle(
    mapping: &NativeMapping,
    derivations: &BTreeMap<&str, &NamedDerivation>,
) -> Option<Vec<String>> {
    fn visit(
        field: &str,
        mapping: &NativeMapping,
        derivations: &BTreeMap<&str, &NamedDerivation>,
        active: &mut Vec<String>,
        complete: &mut BTreeSet<String>,
    ) -> Option<Vec<String>> {
        if let Some(index) = active.iter().position(|item| item == field) {
            let mut cycle = active[index..].to_vec();
            cycle.push(field.to_owned());
            return Some(cycle);
        }
        if complete.contains(field) {
            return None;
        }
        let derivation_name = mapping.fields.get(field)?.derivation.as_deref()?;
        let derivation = derivations.get(derivation_name)?;
        active.push(field.to_owned());
        for dependency in derivation.operation.dependencies() {
            if let Some(cycle) = visit(dependency, mapping, derivations, active, complete) {
                return Some(cycle);
            }
        }
        active.pop();
        complete.insert(field.to_owned());
        None
    }

    let mut complete = BTreeSet::new();
    for field in mapping.fields.keys() {
        if let Some(cycle) = visit(field, mapping, derivations, &mut Vec::new(), &mut complete) {
            return Some(cycle);
        }
    }
    None
}

fn validate_coordinate_selectors(
    mapping: &NativeMapping,
    error: &mut impl FnMut(&str, Option<&str>, String),
) {
    match &mapping.coordinates.horizontal {
        HorizontalCoordinates::EmbeddedGrid if mapping.format == SourceFormat::Netcdf => error(
            "netcdf_coordinates_missing",
            None,
            "NetCDF mappings must select latitude and longitude variables".to_owned(),
        ),
        HorizontalCoordinates::Variables {
            latitude,
            longitude,
        } => {
            validate_selector(mapping.format, latitude, "latitude", error);
            validate_selector(mapping.format, longitude, "longitude", error);
        }
        HorizontalCoordinates::EmbeddedGrid => {}
    }
    match (&mapping.format, &mapping.coordinates.time) {
        (SourceFormat::Netcdf, TimeCoordinate::EmbeddedMetadata) => error(
            "netcdf_time_missing",
            None,
            "NetCDF mappings must declare a time dimension/coordinate".to_owned(),
        ),
        (SourceFormat::Grib1 | SourceFormat::Grib2, TimeCoordinate::Dimension { .. }) => error(
            "grib_time_not_embedded",
            None,
            "GRIB time is message metadata and must use embedded_metadata".to_owned(),
        ),
        _ => {}
    }
    if let Some(selector) = mapping.coordinates.vertical.selector.as_ref() {
        validate_selector(mapping.format, selector, "vertical", error);
    } else if mapping.format == SourceFormat::Netcdf {
        error(
            "netcdf_vertical_missing",
            None,
            "NetCDF mappings must select a vertical coordinate variable".to_owned(),
        );
    }
    for (label, selector) in [
        (
            "time",
            match &mapping.coordinates.time {
                TimeCoordinate::EmbeddedMetadata => None,
                TimeCoordinate::Dimension { selector, .. } => Some(selector),
            },
        ),
        (
            "member",
            match mapping.coordinates.member.as_ref() {
                Some(MemberCoordinate::Dimension { selector }) => Some(selector),
                Some(MemberCoordinate::EmbeddedMetadata) | None => None,
            },
        ),
    ] {
        if let Some(selector) = selector {
            if selector.name.as_ref().is_none_or(NameSpec::is_unmatchable)
                && selector.standard_name.as_deref().is_none_or(str::is_empty)
            {
                error(
                    "empty_dimension_selector",
                    Some(label),
                    "dimension selector needs name and/or standard_name".to_owned(),
                );
            }
        }
    }
    if matches!(
        (&mapping.format, mapping.coordinates.member.as_ref()),
        (
            SourceFormat::Netcdf,
            Some(MemberCoordinate::EmbeddedMetadata)
        ) | (
            SourceFormat::Grib1 | SourceFormat::Grib2,
            Some(MemberCoordinate::Dimension { .. })
        )
    ) {
        error(
            "member_coordinate_format",
            Some("member"),
            "NetCDF members use a dimension; GRIB members use embedded metadata".to_owned(),
        );
    }
}

fn validate_selector(
    format: SourceFormat,
    selector: &VariableSelector,
    field: &str,
    error: &mut impl FnMut(&str, Option<&str>, String),
) {
    if selector.format() != format {
        error(
            "selector_format_mismatch",
            Some(field),
            format!(
                "mapping format is {}, but selector is {}",
                format.id(),
                selector.format().id()
            ),
        );
    }
    match selector {
        VariableSelector::Netcdf {
            name,
            standard_name,
        } => {
            if name.as_ref().is_none_or(NameSpec::is_unmatchable)
                && standard_name.as_deref().is_none_or(str::is_empty)
            {
                error(
                    "empty_netcdf_selector",
                    Some(field),
                    "NetCDF selector needs name and/or standard_name".to_owned(),
                );
            }
        }
        VariableSelector::Grib1 { level_value, .. } => {
            if level_value.is_some_and(|value| !value.is_finite()) {
                error(
                    "nonfinite_selector_level",
                    Some(field),
                    "GRIB selector level_value must be finite".to_owned(),
                );
            }
        }
        VariableSelector::Grib2 {
            level_value,
            second_level_type,
            second_level_value,
            ..
        } => {
            if level_value.is_some_and(|value| !value.is_finite()) {
                error(
                    "nonfinite_selector_level",
                    Some(field),
                    "GRIB selector level_value must be finite".to_owned(),
                );
            }
            if second_level_type.is_some() != second_level_value.is_some() {
                error(
                    "incomplete_second_fixed_surface",
                    Some(field),
                    "second_level_type and second_level_value are an atomic pair".to_owned(),
                );
            }
            if second_level_value.is_some_and(|value| !value.is_finite()) {
                error(
                    "nonfinite_second_selector_level",
                    Some(field),
                    "GRIB2 selector second_level_value must be finite".to_owned(),
                );
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceInspection {
    pub schema: String,
    pub verdict: String,
    pub format: SourceFormat,
    pub files: Vec<FileInspection>,
    pub selectors: Vec<SelectorInspection>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceInventory {
    pub schema: String,
    pub format: SourceFormat,
    pub files: Vec<FileInspection>,
    pub record_count: usize,
    pub unique_records: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileInspection {
    pub path: PathBuf,
    pub byte_count: u64,
    pub records: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectorInspection {
    pub field: String,
    pub selector: String,
    pub matches: usize,
    pub status: String,
}

#[derive(Debug, Clone)]
enum InventoryRecord {
    Grib1 {
        parameter: u8,
        table_version: u8,
        center: u8,
        level_type: u8,
        level_value: f64,
    },
    Grib2 {
        discipline: u8,
        category: u8,
        parameter: u8,
        level_type: u8,
        level_value: f64,
        second_level_type: Option<u8>,
        second_level_value: Option<f64>,
        member: Option<u8>,
    },
    Netcdf {
        name: String,
        standard_name: Option<String>,
        attributes: BTreeSet<String>,
    },
    NetcdfDimension {
        name: String,
    },
}

impl InventoryRecord {
    fn describe(&self) -> String {
        match self {
            Self::Grib1 {
                parameter,
                table_version,
                center,
                level_type,
                level_value,
            } => format!(
                "GRIB1 parameter={parameter} table={table_version} center={center} level_type={level_type} level={level_value}"
            ),
            Self::Grib2 {
                discipline,
                category,
                parameter,
                level_type,
                level_value,
                second_level_type,
                second_level_value,
                member,
            } => format!(
                "GRIB2 discipline={discipline} category={category} parameter={parameter} level_type={level_type} level={level_value} second_level_type={second_level_type:?} second_level={second_level_value:?} member={member:?}"
            ),
            Self::Netcdf {
                name,
                standard_name,
                ..
            } => format!("NetCDF variable={name} standard_name={standard_name:?}"),
            Self::NetcdfDimension { name } => format!("NetCDF dimension={name}"),
        }
    }
}

pub fn inventory_sources(
    format: SourceFormat,
    paths: &[PathBuf],
) -> Result<SourceInventory, RwWpsError> {
    let (records, files) = load_inventory(format, paths)?;
    let unique_records = records
        .iter()
        .map(InventoryRecord::describe)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(SourceInventory {
        schema: INVENTORY_SCHEMA.to_owned(),
        format,
        files,
        record_count: records.len(),
        unique_records,
    })
}

fn load_inventory(
    format: SourceFormat,
    paths: &[PathBuf],
) -> Result<(Vec<InventoryRecord>, Vec<FileInspection>), RwWpsError> {
    if paths.is_empty() {
        return Err(RwWpsError::Config(
            "at least one source file is required for inspection".to_owned(),
        ));
    }
    let mut records = Vec::new();
    let mut files = Vec::new();
    for path in paths {
        let metadata = fs::metadata(path).map_err(|error| io_error(path, error))?;
        if !metadata.is_file() {
            return Err(RwWpsError::Config(format!(
                "source {} is not a regular file",
                path.display()
            )));
        }
        let before = records.len();
        match format {
            SourceFormat::Grib1 => inspect_grib1(path, &mut records)?,
            SourceFormat::Grib2 => inspect_grib2(path, &mut records)?,
            SourceFormat::Netcdf => inspect_netcdf(path, &mut records)?,
        }
        files.push(FileInspection {
            path: path.clone(),
            byte_count: metadata.len(),
            records: records.len() - before,
        });
    }
    Ok((records, files))
}

pub fn inspect_sources(
    mapping: &NativeMapping,
    paths: &[PathBuf],
) -> Result<SourceInspection, RwWpsError> {
    let fields = mapping.fields.keys().cloned().collect();
    inspect_source_fields(mapping, paths, &fields)
}

/// Inspect only the declared field partition against one source-file inventory.
///
/// Composed inputs deliberately assign some direct fields to supplements.  The
/// full mapping remains the validation authority, while this narrower view
/// prevents a selector from being satisfied by the wrong input partition.
pub fn inspect_source_fields(
    mapping: &NativeMapping,
    paths: &[PathBuf],
    fields: &BTreeSet<String>,
) -> Result<SourceInspection, RwWpsError> {
    let report = validate_mapping(mapping);
    if !report.passed() {
        return Err(RwWpsError::Config(format!(
            "mapping validation failed with {} error(s); fix the mapping before source inspection",
            report.errors.len()
        )));
    }
    if fields.is_empty() {
        return Err(RwWpsError::Config(
            "source field partition must not be empty".to_owned(),
        ));
    }
    let known_fields = mapping.fields.keys().cloned().collect();
    let unknown = fields
        .difference(&known_fields)
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(RwWpsError::Config(format!(
            "source field partition names unknown mapping fields {unknown:?}"
        )));
    }

    let (records, files) = load_inventory(mapping.format, paths)?;
    Ok(inspect_inventory_partition(mapping, records, files, fields))
}

fn inspect_inventory_partition(
    mapping: &NativeMapping,
    records: Vec<InventoryRecord>,
    files: Vec<FileInspection>,
    fields: &BTreeSet<String>,
) -> SourceInspection {
    let mut selectors = Vec::new();
    let mut errors = Vec::new();
    for (field, mapped) in &mapping.fields {
        if !fields.contains(field) || mapped.derivation.is_some() {
            continue;
        }
        let matches = mapped
            .selectors
            .iter()
            .map(|selector| {
                records
                    .iter()
                    .filter(|record| selector_matches(selector, record))
                    .count()
            })
            .sum::<usize>();
        let status = if matches == 0 {
            errors.push(format!(
                "field {field:?}: no source record matches any declared selector"
            ));
            "MISSING"
        } else {
            "FOUND"
        };
        selectors.push(SelectorInspection {
            field: field.clone(),
            selector: mapped
                .selectors
                .iter()
                .map(VariableSelector::describe)
                .collect::<Vec<_>>()
                .join(" OR "),
            matches,
            status: status.to_owned(),
        });
        validate_field_metadata(field, mapped, &records, &mut errors);
    }
    match &mapping.coordinates.horizontal {
        HorizontalCoordinates::EmbeddedGrid => {}
        HorizontalCoordinates::Variables {
            latitude,
            longitude,
        } => {
            inspect_selector_group(
                "@coordinate.latitude",
                std::slice::from_ref(latitude),
                &records,
                &mut selectors,
                &mut errors,
            );
            inspect_selector_group(
                "@coordinate.longitude",
                std::slice::from_ref(longitude),
                &records,
                &mut selectors,
                &mut errors,
            );
        }
    }
    if let Some(vertical) = mapping.coordinates.vertical.selector.as_ref() {
        inspect_selector_group(
            "@coordinate.vertical",
            std::slice::from_ref(vertical),
            &records,
            &mut selectors,
            &mut errors,
        );
    }
    if let TimeCoordinate::Dimension { selector, .. } = &mapping.coordinates.time {
        inspect_dimension(
            "@coordinate.time",
            selector,
            &records,
            &mut selectors,
            &mut errors,
        );
    }
    if let Some(MemberCoordinate::Dimension { selector }) = &mapping.coordinates.member {
        inspect_dimension(
            "@coordinate.member",
            selector,
            &records,
            &mut selectors,
            &mut errors,
        );
    }
    selectors.sort_by(|left, right| left.field.cmp(&right.field));
    SourceInspection {
        schema: INSPECTION_SCHEMA.to_owned(),
        verdict: if errors.is_empty() { "PASS" } else { "FAIL" }.to_owned(),
        format: mapping.format,
        files,
        selectors,
        errors,
    }
}

fn inspect_selector_group(
    field: &str,
    declared: &[VariableSelector],
    records: &[InventoryRecord],
    selectors: &mut Vec<SelectorInspection>,
    errors: &mut Vec<String>,
) {
    let matches = declared
        .iter()
        .map(|selector| {
            records
                .iter()
                .filter(|record| selector_matches(selector, record))
                .count()
        })
        .sum::<usize>();
    let status = if matches == 0 {
        errors.push(format!(
            "{field}: no source record matches the declared selector"
        ));
        "MISSING"
    } else {
        "FOUND"
    };
    selectors.push(SelectorInspection {
        field: field.to_owned(),
        selector: declared
            .iter()
            .map(VariableSelector::describe)
            .collect::<Vec<_>>()
            .join(" OR "),
        matches,
        status: status.to_owned(),
    });
}

fn validate_field_metadata(
    field: &str,
    mapping: &FieldMapping,
    records: &[InventoryRecord],
    errors: &mut Vec<String>,
) {
    let MissingPolicy::Attribute { name } = &mapping.missing else {
        return;
    };
    let matched = records.iter().filter(|record| {
        mapping
            .selectors
            .iter()
            .any(|selector| selector_matches(selector, record))
    });
    for record in matched {
        match record {
            InventoryRecord::Netcdf { attributes, .. } if attributes.contains(name) => {}
            InventoryRecord::Netcdf {
                name: variable, ..
            } => errors.push(format!(
                "field {field:?}: NetCDF variable {variable:?} lacks declared missing-data attribute {name:?}"
            )),
            _ => errors.push(format!(
                "field {field:?}: attribute missing-data policy cannot be verified for a non-NetCDF record"
            )),
        }
    }
}

fn inspect_dimension(
    field: &str,
    declared: &DimensionSelector,
    records: &[InventoryRecord],
    selectors: &mut Vec<SelectorInspection>,
    errors: &mut Vec<String>,
) {
    let matches = records
        .iter()
        .filter(|record| match record {
            InventoryRecord::NetcdfDimension { name } => {
                declared.name.as_ref().is_some_and(|value| value.matches(name))
            }
            InventoryRecord::Netcdf {
                name,
                standard_name,
                ..
            } => {
                declared.name.as_ref().is_some_and(|value| value.matches(name))
                    || declared
                        .standard_name
                        .as_ref()
                        .is_some_and(|value| Some(value) == standard_name.as_ref())
            }
            _ => false,
        })
        .count();
    let status = if matches == 0 {
        errors.push(format!(
            "{field}: no NetCDF dimension/coordinate matches name={:?} standard_name={:?}",
            declared.name, declared.standard_name
        ));
        "MISSING"
    } else {
        "FOUND"
    };
    selectors.push(SelectorInspection {
        field: field.to_owned(),
        selector: format!(
            "NetCDF dimension name={:?} standard_name={:?}",
            declared.name, declared.standard_name
        ),
        matches,
        status: status.to_owned(),
    });
}

fn inspect_grib1(path: &Path, records: &mut Vec<InventoryRecord>) -> Result<(), RwWpsError> {
    let file = Grib1File::open(path)
        .map_err(|error| RwWpsError::Config(format!("GRIB1 {}: {error}", path.display())))?;
    if file.messages.is_empty() {
        return Err(RwWpsError::Config(format!(
            "GRIB1 {} contains no parseable messages",
            path.display()
        )));
    }
    records.extend(
        file.messages
            .into_iter()
            .map(|message| InventoryRecord::Grib1 {
                parameter: message.pds.parameter,
                table_version: message.pds.table_version,
                center: message.pds.center_id,
                level_type: message.pds.level_type,
                level_value: f64::from(message.pds.level_value),
            }),
    );
    Ok(())
}

fn inspect_grib2(path: &Path, records: &mut Vec<InventoryRecord>) -> Result<(), RwWpsError> {
    let path_text = path.to_string_lossy();
    let file = Grib2File::open(&path_text)
        .map_err(|error| RwWpsError::Config(format!("GRIB2 {}: {error}", path.display())))?;
    if file.messages.is_empty() {
        return Err(RwWpsError::Config(format!(
            "GRIB2 {} contains no parseable messages",
            path.display()
        )));
    }
    records.extend(
        file.messages
            .into_iter()
            .map(|message| InventoryRecord::Grib2 {
                discipline: message.discipline,
                category: message.product.parameter_category,
                parameter: message.product.parameter_number,
                level_type: message.product.level_type,
                level_value: message.product.level_value,
                // grib-core adaptation (lane 2, routed through the superset's
                // own accessor at integration).  gpuwm's crate — the one
                // superset, and the crate the shipped bridges decode
                // through — spells "no second fixed surface" as the WMO
                // missing code 255 with a zero value, where the donor
                // snapshot spelled it `None`.  The sentinel is translated
                // HERE rather than widened into `Some(255)`, because
                // `Some(255)` would announce a second fixed surface of type
                // "missing" and every selector that matches on the absence
                // of one would stop matching.  `second_fixed_surface()` keeps
                // the 255 test inside grib-core so this call site cannot
                // drift from the crate's own reading of Code Table 4.5.
                second_level_type: message
                    .product
                    .second_fixed_surface()
                    .map(|(kind, _)| kind),
                second_level_value: message
                    .product
                    .second_fixed_surface()
                    .map(|(_, value)| value),
                member: message.product.perturbation_number,
            }),
    );
    Ok(())
}

fn inspect_netcdf(path: &Path, records: &mut Vec<InventoryRecord>) -> Result<(), RwWpsError> {
    let record_start = records.len();
    let file = netcrust::open(path)
        .map_err(|error| RwWpsError::Config(format!("NetCDF {}: {error}", path.display())))?;
    match (file.dimensions(), file.variables()) {
        (Ok(dimensions), Ok(variables)) if !variables.is_empty() => {
            records.extend(dimensions.into_iter().map(|dimension| {
                InventoryRecord::NetcdfDimension {
                    name: dimension.name().to_owned(),
                }
            }));
            records.extend(variables.into_iter().map(|variable| {
                InventoryRecord::Netcdf {
                    name: variable.name().to_owned(),
                    standard_name: variable
                        .attribute("standard_name")
                        .and_then(|attribute| attribute.as_string())
                        .map(str::to_owned),
                    attributes: variable
                        .attributes()
                        .iter()
                        .map(|attribute| attribute.name().to_owned())
                        .collect(),
                }
            }));
        }
        (dimensions, variables) => {
            let hdf5 = hdf5_reader::Hdf5File::open(path).map_err(|hdf5_error| {
                RwWpsError::Config(format!(
                    "NetCDF {} metadata failed (dimensions={:?}, variables={:?}); HDF5 fallback failed: {hdf5_error}",
                    path.display(),
                    dimensions.err(),
                    variables.err()
                ))
            })?;
            let root = hdf5.root_group().map_err(|error| {
                RwWpsError::Config(format!(
                    "NetCDF4/HDF5 {} root group: {error}",
                    path.display()
                ))
            })?;
            inspect_hdf5_group(&root, "", records).map_err(|error| {
                RwWpsError::Config(format!(
                    "NetCDF4/HDF5 {} inventory: {error}",
                    path.display()
                ))
            })?;
        }
    }
    if !records[record_start..]
        .iter()
        .any(|record| matches!(record, InventoryRecord::Netcdf { .. }))
    {
        return Err(RwWpsError::Config(format!(
            "NetCDF {} contains no discoverable variables",
            path.display()
        )));
    }
    Ok(())
}

fn inspect_hdf5_group(
    group: &hdf5_reader::group::Group,
    prefix: &str,
    records: &mut Vec<InventoryRecord>,
) -> Result<(), hdf5_reader::error::Error> {
    for dataset in group.datasets()? {
        let name = if prefix.is_empty() {
            dataset.name().to_owned()
        } else {
            format!("{prefix}/{}", dataset.name())
        };
        let standard_name = dataset
            .attribute("standard_name")
            .ok()
            .and_then(|attribute| attribute.read_string().ok());
        records.push(InventoryRecord::Netcdf {
            name: name.clone(),
            standard_name,
            attributes: dataset
                .attributes()
                .into_iter()
                .map(|attribute| attribute.name)
                .collect(),
        });
        if dataset
            .attribute("CLASS")
            .ok()
            .and_then(|attribute| attribute.read_string().ok())
            .as_deref()
            == Some("DIMENSION_SCALE")
        {
            records.push(InventoryRecord::NetcdfDimension { name });
        }
    }
    for child in group.groups()? {
        let child_prefix = if prefix.is_empty() {
            child.name().to_owned()
        } else {
            format!("{prefix}/{}", child.name())
        };
        inspect_hdf5_group(&child, &child_prefix, records)?;
    }
    Ok(())
}

fn selector_matches(selector: &VariableSelector, record: &InventoryRecord) -> bool {
    const LEVEL_EPSILON: f64 = 1.0e-6;
    match (selector, record) {
        (
            VariableSelector::Grib1 {
                parameter,
                table_version,
                center,
                level_type,
                level_value,
            },
            InventoryRecord::Grib1 {
                parameter: actual_parameter,
                table_version: actual_table,
                center: actual_center,
                level_type: actual_level_type,
                level_value: actual_level,
            },
        ) => {
            parameter == actual_parameter
                && table_version.is_none_or(|value| value == *actual_table)
                && center.is_none_or(|value| value == *actual_center)
                && level_type.is_none_or(|value| value == *actual_level_type)
                && level_value.is_none_or(|value| (value - *actual_level).abs() <= LEVEL_EPSILON)
        }
        (
            VariableSelector::Grib2 {
                discipline,
                category,
                parameter,
                level_type,
                level_value,
                second_level_type,
                second_level_value,
                member,
                // center/subcenter/table versions are declarative provenance;
                // the GRIB2 inventory record does not yet carry them.
                ..
            },
            InventoryRecord::Grib2 {
                discipline: actual_discipline,
                category: actual_category,
                parameter: actual_parameter,
                level_type: actual_level_type,
                level_value: actual_level,
                second_level_type: actual_second_level_type,
                second_level_value: actual_second_level,
                member: actual_member,
            },
        ) => {
            let second_surface_matches = match (second_level_type, second_level_value) {
                (None, None) => actual_second_level_type.is_none(),
                (Some(expected_type), Some(expected_value)) => {
                    Some(*expected_type) == *actual_second_level_type
                        && actual_second_level
                            .is_some_and(|actual| (*expected_value - actual).abs() <= LEVEL_EPSILON)
                }
                _ => false,
            };
            discipline == actual_discipline
                && category == actual_category
                && parameter == actual_parameter
                && level_type.is_none_or(|value| value == *actual_level_type)
                && level_value.is_none_or(|value| (value - *actual_level).abs() <= LEVEL_EPSILON)
                && second_surface_matches
                && member.is_none_or(|value| Some(value) == *actual_member)
        }
        (
            VariableSelector::Netcdf {
                name,
                standard_name,
            },
            InventoryRecord::Netcdf {
                name: actual_name,
                standard_name: actual_standard_name,
                ..
            },
        ) => {
            name.as_ref().is_none_or(|value| value.matches(actual_name))
                && standard_name
                    .as_ref()
                    .is_none_or(|value| Some(value) == actual_standard_name.as_ref())
        }
        _ => false,
    }
}

/// Transcribe a classic 11-column WPS Vtable into a hash-bound row draft.
///
/// The result is intentionally non-executable. It preserves both GRIB
/// editions and their raw level columns, but never guesses canonical field
/// names, units, axes, missing-data behavior, derivations, or a WRF target
/// contract. Those decisions belong in an explicit descriptor compiled by the
/// Python authoring authority.
pub fn import_wps_vtable(
    bytes: &[u8],
    name: impl Into<String>,
) -> Result<VtableRowDraft, RwWpsError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| RwWpsError::Config(format!("Vtable is not valid UTF-8: {error}")))?;
    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("GRIB1")
            || trimmed.starts_with("Param")
            || trimmed.starts_with('-')
        {
            continue;
        }

        let mut columns = line
            .split('|')
            .map(|value| value.trim().to_owned())
            .collect::<Vec<_>>();
        let mut unresolved = Vec::new();
        if columns.len() < 11 {
            unresolved.push(draft_issue(
                "malformed_column_count",
                format!(
                    "row has {} columns; an 11-column Vtable row is required before descriptor authoring",
                    columns.len()
                ),
            ));
            columns.resize(11, String::new());
        }
        let extra_columns = if columns.len() > 11 {
            unresolved.push(draft_issue(
                "extra_columns",
                format!(
                    "row has {} extra column(s); their meaning must be resolved explicitly",
                    columns.len() - 11
                ),
            ));
            columns[11..].to_vec()
        } else {
            Vec::new()
        };

        let grib1_present = columns[..4].iter().any(|value| !value.is_empty());
        let grib2_present = columns[7..11].iter().any(|value| !value.is_empty());
        let grib1 = grib1_present.then(|| VtableGrib1Draft {
            parameter: columns[0].clone(),
            level_type: columns[1].clone(),
            level1: columns[2].clone(),
            level2: columns[3].clone(),
        });
        let grib2 = grib2_present.then(|| VtableGrib2Draft {
            discipline: columns[7].clone(),
            category: columns[8].clone(),
            parameter: columns[9].clone(),
            level_type: columns[10].clone(),
            level1: columns[2].clone(),
            level2: columns[3].clone(),
        });

        if !grib1_present && !grib2_present {
            unresolved.push(draft_issue(
                "selector_missing",
                "row has neither a complete GRIB1 nor GRIB2 selector",
            ));
        }
        if grib1_present && columns[..2].iter().any(String::is_empty) {
            unresolved.push(draft_issue(
                "grib1_selector_incomplete",
                "GRIB1 parameter and level type are not both present",
            ));
        }
        if grib2_present && columns[7..11].iter().any(String::is_empty) {
            unresolved.push(draft_issue(
                "grib2_selector_incomplete",
                "GRIB2 discipline, category, parameter, and level type are not all present",
            ));
        }
        if grib1_present && grib2_present {
            unresolved.push(draft_issue(
                "cross_edition_equivalence_unverified",
                "the Vtable pairs GRIB1 and GRIB2 codes, but their scientific equivalence has not been independently established",
            ));
        }
        if columns[2] == "*" {
            unresolved.push(draft_issue(
                "level_inventory_unresolved",
                "wildcard Level1 requires an explicit vertical inventory and ordering contract",
            ));
        } else if grib2_present && !columns[2].is_empty() {
            unresolved.push(draft_issue(
                "grib2_level_value_unresolved",
                "numeric Level1 does not by itself establish the GRIB2 scaled fixed-surface value",
            ));
        }
        if !columns[3].is_empty() {
            unresolved.push(draft_issue(
                "layer_bounds_unresolved",
                "Level1/Level2 require an explicit atomic bounded-layer selector",
            ));
        }
        if columns[4].is_empty() {
            unresolved.push(draft_issue(
                "metgrid_name_missing",
                "row has no metgrid mnemonic",
            ));
        }

        rows.push(VtableDraftRow {
            id: format!("vtable-line-{}", line_index + 1),
            line: line_index + 1,
            grib1,
            grib2,
            metgrid_name: columns[4].clone(),
            units: columns[5].clone(),
            description: columns[6].clone(),
            extra_columns,
            conflict_ids: Vec::new(),
            unresolved,
        });
    }
    if rows.is_empty() {
        return Err(RwWpsError::Config(
            "Vtable contains no data rows to preserve".to_owned(),
        ));
    }

    let mut rows_by_mnemonic = BTreeMap::<String, Vec<usize>>::new();
    for (index, row) in rows.iter().enumerate() {
        if !row.metgrid_name.is_empty() {
            rows_by_mnemonic
                .entry(row.metgrid_name.clone())
                .or_default()
                .push(index);
        }
    }
    let mut conflicts = Vec::new();
    for (mnemonic, indexes) in rows_by_mnemonic {
        if indexes.len() < 2 {
            continue;
        }
        let conflict_id = format!("mnemonic-reused-{mnemonic}");
        let row_ids = indexes
            .iter()
            .map(|index| rows[*index].id.clone())
            .collect::<Vec<_>>();
        for index in &indexes {
            rows[*index].conflict_ids.push(conflict_id.clone());
            rows[*index].unresolved.push(draft_issue(
                "metgrid_mnemonic_reused",
                format!(
                    "metgrid mnemonic {mnemonic:?} is used by multiple rows and cannot identify a canonical field"
                ),
            ));
        }
        conflicts.push(VtableDraftConflict {
            id: conflict_id,
            kind: "metgrid_mnemonic_reused".to_owned(),
            metgrid_name: mnemonic.clone(),
            row_ids,
            message: format!(
                "{mnemonic:?} names multiple selector/level rows; a descriptor must bind each intended field explicitly"
            ),
        });
    }

    let digest = Sha256::digest(bytes);
    Ok(VtableRowDraft {
        schema: VTABLE_ROW_DRAFT_SCHEMA.to_owned(),
        name: name.into(),
        executable: false,
        source: VtableSourceBinding {
            bytes: u64::try_from(bytes.len()).expect("usize fits in u64 on supported targets"),
            sha256: format!("{digest:x}"),
        },
        rows,
        conflicts,
        unresolved: vec![
            draft_issue(
                "non_executable_row_draft",
                "this document cannot be supplied where rw-wps.mapping.v1 is required",
            ),
            draft_issue(
                "canonical_semantics_unresolved",
                "Vtable mnemonics, unit labels, and descriptions are evidence, not canonical field semantics",
            ),
            draft_issue(
                "coordinate_and_axis_contract_unresolved",
                "horizontal, vertical, time/member coordinates, axis order, staggering, and missing-data policy are absent",
            ),
            draft_issue(
                "target_contract_unresolved",
                "derivations, required fields, physics, soil/vertical levels, cadence, domains, and lateral-boundary policy are absent",
            ),
        ],
    })
}

fn draft_issue(code: &str, message: impl Into<String>) -> VtableDraftIssue {
    VtableDraftIssue {
        code: code.to_owned(),
        message: message.into(),
    }
}

pub fn wrf_real_contract(max_dom: u16, physics_suite: impl Into<String>) -> TargetContract {
    TargetContract {
        name: "gpuwm/wrf-real initialization".to_owned(),
        physics_suite: physics_suite.into(),
        max_dom,
        require_lateral_boundaries: true,
        target_vertical_levels: None,
        soil_layer_count: None,
        boundary_interval_seconds: None,
        required_fields: canonical_wrf_requirements(),
        pressure_requirement: PressureRequirement::AirPressureOrHybridCoordinate,
        policy_controlled_fields: canonical_policy_fields(),
        initialization_policies: BTreeMap::new(),
    }
}

fn canonical_wrf_requirements() -> Vec<FieldRequirement> {
    let three_d = [
        ("air_temperature", "K"),
        ("specific_humidity", "kg kg-1"),
        ("eastward_wind", "m s-1"),
        ("northward_wind", "m s-1"),
        ("geopotential_height", "m"),
    ];
    let surface = [
        ("surface_pressure", "Pa", GridLocation::Surface),
        ("terrain_height", "m", GridLocation::Surface),
        ("skin_temperature", "K", GridLocation::Surface),
        ("air_temperature_2m", "K", GridLocation::Surface),
        ("specific_humidity_2m", "kg kg-1", GridLocation::Surface),
        ("eastward_wind_10m", "m s-1", GridLocation::Surface),
        ("northward_wind_10m", "m s-1", GridLocation::Surface),
        ("land_fraction", "1", GridLocation::Surface),
        ("soil_temperature", "K", GridLocation::Soil),
        ("volumetric_soil_moisture", "m3 m-3", GridLocation::Soil),
    ];
    let mut required_fields = three_d
        .into_iter()
        .map(|(name, units)| FieldRequirement {
            name: name.to_owned(),
            axes: vec![AxisRole::Vertical, AxisRole::Y, AxisRole::X],
            location: GridLocation::Mass,
            target_units: units.to_owned(),
        })
        .collect::<Vec<_>>();
    required_fields.extend(
        surface
            .into_iter()
            .map(|(name, units, location)| FieldRequirement {
                name: name.to_owned(),
                axes: if location == GridLocation::Soil {
                    vec![AxisRole::Soil, AxisRole::Y, AxisRole::X]
                } else {
                    vec![AxisRole::Y, AxisRole::X]
                },
                location,
                target_units: units.to_owned(),
            }),
    );
    required_fields
}

fn canonical_policy_fields() -> Vec<String> {
    [
        "cloud_water_mixing_ratio",
        "rain_water_mixing_ratio",
        "cloud_ice_mixing_ratio",
        "snow_mixing_ratio",
        "graupel_or_hail_mixing_ratio",
        "vertical_velocity",
        "snow_water_equivalent",
        "snow_depth",
        "sea_ice_fraction",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

pub fn mapping_template(format: SourceFormat) -> NativeMapping {
    let selector = match format {
        SourceFormat::Grib1 => VariableSelector::Grib1 {
            parameter: 11,
            table_version: None,
            center: None,
            level_type: Some(100),
            level_value: None,
        },
        SourceFormat::Grib2 => VariableSelector::Grib2 {
            discipline: 0,
            category: 0,
            parameter: 0,
            center: None,
            subcenter: None,
            master_table_version: None,
            local_table_version: None,
            level_type: Some(100),
            level_value: None,
            second_level_type: None,
            second_level_value: None,
            member: None,
        },
        SourceFormat::Netcdf => VariableSelector::Netcdf {
            name: Some(NameSpec::from("temperature")),
            standard_name: Some("air_temperature".to_owned()),
        },
    };
    let horizontal = match format {
        SourceFormat::Netcdf => HorizontalCoordinates::Variables {
            latitude: VariableSelector::Netcdf {
                name: Some(NameSpec::from("latitude")),
                standard_name: Some("latitude".to_owned()),
            },
            longitude: VariableSelector::Netcdf {
                name: Some(NameSpec::from("longitude")),
                standard_name: Some("longitude".to_owned()),
            },
        },
        SourceFormat::Grib1 | SourceFormat::Grib2 => HorizontalCoordinates::EmbeddedGrid,
    };
    let mut fields = BTreeMap::new();
    fields.insert(
        "air_temperature".to_owned(),
        FieldMapping {
            selectors: vec![selector],
            derivation: None,
            selector_stack_axis: None,
            units: UnitTransform {
                source: "K".to_owned(),
                target: "K".to_owned(),
                scale: 1.0,
                offset: 0.0,
            },
            source_axes: if format == SourceFormat::Netcdf {
                vec![AxisRole::Time, AxisRole::Vertical, AxisRole::Y, AxisRole::X]
            } else {
                vec![AxisRole::Vertical, AxisRole::Y, AxisRole::X]
            },
            target_axes: vec![AxisRole::Vertical, AxisRole::Y, AxisRole::X],
            location: GridLocation::Mass,
            staggering: Staggering::None,
            missing: MissingPolicy::Reject,
        },
    );
    fields.insert(
        "air_pressure".to_owned(),
        FieldMapping {
            selectors: Vec::new(),
            derivation: Some("pressure-from-vertical-coordinate".to_owned()),
            selector_stack_axis: None,
            units: UnitTransform {
                source: if format == SourceFormat::Grib1 {
                    "hPa".to_owned()
                } else {
                    "Pa".to_owned()
                },
                target: "Pa".to_owned(),
                scale: if format == SourceFormat::Grib1 {
                    100.0
                } else {
                    1.0
                },
                offset: 0.0,
            },
            source_axes: vec![AxisRole::Vertical, AxisRole::Y, AxisRole::X],
            target_axes: vec![AxisRole::Vertical, AxisRole::Y, AxisRole::X],
            location: GridLocation::Mass,
            staggering: Staggering::None,
            missing: MissingPolicy::Reject,
        },
    );
    NativeMapping {
        schema: MAPPING_SCHEMA.to_owned(),
        name: format!("custom-{}", format.id()),
        format,
        coordinates: CoordinateMapping {
            horizontal,
            vertical: VerticalCoordinate {
                kind: VerticalKind::Pressure,
                selector: if format == SourceFormat::Netcdf {
                    Some(VariableSelector::Netcdf {
                        name: Some(NameSpec::from("level")),
                        standard_name: Some("air_pressure".to_owned()),
                    })
                } else {
                    None
                },
                units: if format == SourceFormat::Grib1 {
                    "hPa".to_owned()
                } else {
                    "Pa".to_owned()
                },
                positive: Some(PositiveDirection::Down),
                levels: Vec::new(),
                hybrid_a_field: None,
                hybrid_b_field: None,
                surface_pressure_field: None,
            },
            time: if format == SourceFormat::Netcdf {
                TimeCoordinate::Dimension {
                    selector: DimensionSelector {
                        name: Some(NameSpec::from("time")),
                        standard_name: Some("time".to_owned()),
                    },
                    units: "hours since ...".to_owned(),
                    calendar: Some("proleptic_gregorian".to_owned()),
                }
            } else {
                TimeCoordinate::EmbeddedMetadata
            },
            member: None,
        },
        fields,
        derivations: vec![NamedDerivation {
            name: "pressure-from-vertical-coordinate".to_owned(),
            operation: Derivation::PressureFromVerticalCoordinate,
        }],
        target: wrf_real_contract(1, "edit-to-match-namelist"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VTABLE: &str = r#"
GRIB1| Level| From |  To  | metgrid  | metgrid  | metgrid |GRIB2|GRIB2|GRIB2|GRIB2|
Param| Type |Level1|Level2| Name     | Units    | Desc    |Discp|Catgy|Param|Level|
-----+------+------+------+----------+----------+---------+-----+-----+-----+-----+
 130 | 100  |   *  |      | T        | K        | Temp    |  0  |  0  |  0  | 100 |
 129 | 100  |   *  |      | GHT      | m        | Height  |  0  |  3  |  5  | 100 |
 157 | 100  |   *  |      | RH       | %        | RH      |  0  |  1  |  1  | 100 |
 129 |   1  |   0  |      | GHT      | m        | Terrain |  0  |  3  |  5  |   1 |
 130 | 103  |   2  |      | TT       | K        | T2      |  0  |  0  |  0  | 103 |
"#;

    #[test]
    fn mapping_templates_json_round_trip_into_validation_for_every_format() {
        for format in [
            SourceFormat::Grib1,
            SourceFormat::Grib2,
            SourceFormat::Netcdf,
        ] {
            let template = mapping_template(format);
            let encoded = serde_json::to_vec_pretty(&template).unwrap();
            let decoded: NativeMapping = serde_json::from_slice(&encoded).unwrap();

            assert_eq!(decoded, template, "JSON round-trip drift for {format:?}");
            assert_eq!(
                validate_mapping(&decoded),
                validate_mapping(&template),
                "validation drift after JSON round-trip for {format:?}"
            );
        }
    }

    #[test]
    fn every_typed_derivation_json_variant_round_trips() {
        let variants = [
            Derivation::Copy {
                source: "temperature".to_owned(),
            },
            Derivation::WindSpeed {
                u: "u".to_owned(),
                v: "v".to_owned(),
            },
            Derivation::SpecificHumidityFromRh {
                relative_humidity: "rh".to_owned(),
                temperature: "temperature".to_owned(),
                pressure: "pressure".to_owned(),
            },
            Derivation::RelativeHumidityFromDewpoint {
                dewpoint: "dewpoint".to_owned(),
                temperature: "temperature".to_owned(),
            },
            Derivation::GeopotentialHeight {
                geopotential: "geopotential".to_owned(),
                gravity_m_s2: 9.80665,
            },
            Derivation::PressureFromVerticalCoordinate,
            Derivation::SpecificHumidityFromDewpoint {
                dewpoint: "dewpoint".to_owned(),
                temperature: "temperature".to_owned(),
                pressure: "pressure".to_owned(),
            },
        ];

        for (index, operation) in variants.into_iter().enumerate() {
            let named = NamedDerivation {
                name: format!("derivation-{index}"),
                operation,
            };
            let encoded = serde_json::to_vec(&named).unwrap();
            let decoded: NamedDerivation = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(decoded, named);
        }
    }

    #[test]
    fn flattened_typed_derivations_still_reject_unknown_fields() {
        let error = serde_json::from_str::<NamedDerivation>(
            r#"{
                "name": "copy-temperature",
                "operation": "copy",
                "source": "air_temperature",
                "bogus": true
            }"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field `bogus`"));

        let error = serde_json::from_str::<NamedDerivation>(
            r#"{
                "name": "copy-temperature",
                "operation": "copy",
                "source": "air_temperature",
                "pressure": "air_pressure"
            }"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field `pressure`"));
    }

    #[test]
    fn vtable_import_is_hash_bound_and_never_an_executable_mapping() {
        let draft = import_wps_vtable(VTABLE.as_bytes(), "ecmwf").unwrap();
        assert_eq!(draft.schema, VTABLE_ROW_DRAFT_SCHEMA);
        assert!(!draft.executable);
        assert_eq!(draft.source.bytes, VTABLE.len() as u64);
        assert_eq!(draft.source.sha256.len(), 64);
        assert_eq!(draft.rows.len(), 5);
        let encoded = serde_json::to_value(&draft).unwrap();
        assert!(encoded.get("fields").is_none());
        assert!(encoded.get("coordinates").is_none());
        assert!(encoded.get("target").is_none());
        assert!(
            draft
                .unresolved
                .iter()
                .any(|issue| issue.code == "non_executable_row_draft")
        );
    }

    #[test]
    fn real_hrrr_tt_3d_and_2m_rows_remain_distinct_and_unresolved() {
        // Exact selector/level columns from WPS Vtable.raphrrr. The reused TT
        // mnemonic is the unsafe case where an executable importer could merge
        // rows or assign canonical meaning without a descriptor.
        let hrrr = r#"
GRIB1| Level| From |  To  | metgrid  | metgrid  | metgrid                 |GRIB2|GRIB2|GRIB2|GRIB2|
Param| Type |Level1|Level2| Name     | Units    | Description             |Discp|Catgy|Param|Level|
  11 | 109  |   *  |      | TT       | K        | Temperature             |  0  |  0  |  0  | 105 |
  11 | 105  |   2  |      | TT       | K        | Temperature at 2 m      |  0  |  0  |  0  | 103 |
"#;
        let draft = import_wps_vtable(hrrr.as_bytes(), "hrrr").unwrap();
        assert_eq!(draft.rows.len(), 2);
        assert_ne!(draft.rows[0].id, draft.rows[1].id);
        assert_eq!(draft.rows[0].description, "Temperature");
        assert_eq!(draft.rows[1].description, "Temperature at 2 m");
        assert_eq!(draft.rows[0].grib1.as_ref().unwrap().level_type, "109");
        assert_eq!(draft.rows[0].grib1.as_ref().unwrap().level1, "*");
        assert_eq!(draft.rows[1].grib1.as_ref().unwrap().level_type, "105");
        assert_eq!(draft.rows[1].grib1.as_ref().unwrap().level1, "2");
        assert_eq!(draft.rows[0].grib2.as_ref().unwrap().level_type, "105");
        assert_eq!(draft.rows[1].grib2.as_ref().unwrap().level_type, "103");
        let conflict = draft
            .conflicts
            .iter()
            .find(|conflict| conflict.metgrid_name == "TT")
            .unwrap();
        assert_eq!(conflict.row_ids.len(), 2);
        for row in &draft.rows {
            assert!(row.conflict_ids.contains(&conflict.id));
            assert!(
                row.unresolved
                    .iter()
                    .any(|issue| issue.code == "metgrid_mnemonic_reused")
            );
        }
    }

    #[test]
    fn vtable_import_preserves_malformed_rows_as_unresolved_evidence() {
        let malformed = "11|109|*||TT|K|Temperature|0|0|0|105\nnot eleven columns\n";
        let draft = import_wps_vtable(malformed.as_bytes(), "malformed").unwrap();
        assert_eq!(draft.rows.len(), 2);
        assert_eq!(draft.rows[1].metgrid_name, "");
        assert!(
            draft.rows[1]
                .unresolved
                .iter()
                .any(|issue| issue.code == "malformed_column_count")
        );
    }

    #[test]
    fn contract_validation_reports_every_missing_required_field() {
        let mapping = mapping_template(SourceFormat::Grib2);
        let report = validate_mapping(&mapping);
        assert_eq!(report.verdict, "FAIL");
        assert!(report.errors.iter().any(|item| {
            item.code == "required_field_missing"
                && item.field.as_deref() == Some("surface_pressure")
        }));
        assert!(report.errors.len() > 10);
    }

    #[test]
    fn mismatched_selector_format_fails_closed() {
        let mut mapping = mapping_template(SourceFormat::Grib2);
        mapping.fields.get_mut("air_temperature").unwrap().selectors =
            vec![VariableSelector::Netcdf {
                name: Some(NameSpec::from("t")),
                standard_name: None,
            }];
        let report = validate_mapping(&mapping);
        assert!(
            report
                .errors
                .iter()
                .any(|item| item.code == "selector_format_mismatch")
        );
    }

    #[test]
    fn inventory_matching_obeys_all_grib2_selector_keys() {
        let record = InventoryRecord::Grib2 {
            discipline: 0,
            category: 0,
            parameter: 0,
            level_type: 100,
            level_value: 500.0,
            second_level_type: None,
            second_level_value: None,
            member: Some(7),
        };
        let selector = VariableSelector::Grib2 {
            discipline: 0,
            category: 0,
            parameter: 0,
            center: None,
            subcenter: None,
            master_table_version: None,
            local_table_version: None,
            level_type: Some(100),
            level_value: Some(500.0),
            second_level_type: None,
            second_level_value: None,
            member: Some(7),
        };
        assert!(selector_matches(&selector, &record));
    }

    #[test]
    fn inventory_matching_requires_explicit_second_fixed_surface_bounds() {
        let record = InventoryRecord::Grib2 {
            discipline: 2,
            category: 0,
            parameter: 2,
            level_type: 106,
            level_value: 0.1,
            second_level_type: Some(106),
            second_level_value: Some(0.4),
            member: None,
        };
        let bounded = VariableSelector::Grib2 {
            discipline: 2,
            category: 0,
            parameter: 2,
            center: None,
            subcenter: None,
            master_table_version: None,
            local_table_version: None,
            level_type: Some(106),
            level_value: Some(0.1),
            second_level_type: Some(106),
            second_level_value: Some(0.4),
            member: None,
        };
        let mut unbound = bounded.clone();
        let VariableSelector::Grib2 {
            second_level_type,
            second_level_value,
            ..
        } = &mut unbound
        else {
            unreachable!()
        };
        *second_level_type = None;
        *second_level_value = None;
        let mut wrong_upper_bound = bounded.clone();
        let VariableSelector::Grib2 {
            second_level_value, ..
        } = &mut wrong_upper_bound
        else {
            unreachable!()
        };
        *second_level_value = Some(1.0);

        assert!(selector_matches(&bounded, &record));
        assert!(!selector_matches(&unbound, &record));
        assert!(!selector_matches(&wrong_upper_bound, &record));
    }

    #[test]
    fn field_partition_cannot_be_satisfied_by_supplement_only_record() {
        let mut mapping = mapping_template(SourceFormat::Grib2);
        let mut terrain = mapping.fields["air_temperature"].clone();
        terrain.selectors = vec![VariableSelector::Grib2 {
            discipline: 0,
            category: 3,
            parameter: 5,
            center: None,
            subcenter: None,
            master_table_version: None,
            local_table_version: None,
            level_type: Some(1),
            level_value: Some(0.0),
            second_level_type: None,
            second_level_value: None,
            member: None,
        }];
        mapping.fields.insert("terrain_height".to_owned(), terrain);
        let terrain_record = InventoryRecord::Grib2 {
            discipline: 0,
            category: 3,
            parameter: 5,
            level_type: 1,
            level_value: 0.0,
            second_level_type: None,
            second_level_value: None,
            member: None,
        };
        let primary = inspect_inventory_partition(
            &mapping,
            vec![terrain_record.clone()],
            Vec::new(),
            &BTreeSet::from(["air_temperature".to_owned()]),
        );
        assert_eq!(primary.verdict, "FAIL");
        assert!(
            primary
                .errors
                .iter()
                .any(|error| error.contains("air_temperature"))
        );

        let supplement = inspect_inventory_partition(
            &mapping,
            vec![terrain_record],
            Vec::new(),
            &BTreeSet::from(["terrain_height".to_owned()]),
        );
        assert_eq!(supplement.verdict, "PASS");
    }

    #[test]
    fn derived_field_cycles_are_rejected() {
        let mut mapping = mapping_template(SourceFormat::Grib2);
        mapping.target.required_fields.clear();
        let mut field_a = mapping.fields.remove("air_temperature").unwrap();
        field_a.selectors.clear();
        field_a.derivation = Some("a-from-b".to_owned());
        let mut field_b = field_a.clone();
        field_b.derivation = Some("b-from-a".to_owned());
        mapping.fields.insert("A".to_owned(), field_a);
        mapping.fields.insert("B".to_owned(), field_b);
        mapping.derivations = vec![
            NamedDerivation {
                name: "a-from-b".to_owned(),
                operation: Derivation::Copy {
                    source: "B".to_owned(),
                },
            },
            NamedDerivation {
                name: "b-from-a".to_owned(),
                operation: Derivation::Copy {
                    source: "A".to_owned(),
                },
            },
        ];
        assert!(
            validate_mapping(&mapping)
                .errors
                .iter()
                .any(|item| item.code == "derivation_cycle")
        );
    }

    #[test]
    fn real_netcdf4_fixture_is_inventoried_through_hdf5_fallback() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../rw-glm/tests/fixtures/OR_GLM-L2-LCFA_G19_s20261620805000_e20261620805200_c20261620805214.nc",
        );
        let inventory = inventory_sources(SourceFormat::Netcdf, &[fixture]).unwrap();
        assert!(inventory.record_count > 50);
        assert!(
            inventory
                .unique_records
                .iter()
                .any(|record| record.contains("flash_lat"))
        );
    }

    #[test]
    fn declared_netcdf_missing_attribute_must_exist() {
        let mut mapping = mapping_template(SourceFormat::Netcdf);
        let field = mapping.fields.get_mut("air_temperature").unwrap();
        field.missing = MissingPolicy::Attribute {
            name: "_FillValue".to_owned(),
        };
        let records = vec![InventoryRecord::Netcdf {
            name: "temperature".to_owned(),
            standard_name: Some("air_temperature".to_owned()),
            attributes: BTreeSet::new(),
        }];
        let mut errors = Vec::new();
        validate_field_metadata("T", field, &records, &mut errors);
        assert!(errors.iter().any(|error| error.contains("_FillValue")));
    }

    #[test]
    fn target_contract_cannot_remove_canonical_wrf_requirements() {
        let mut mapping = mapping_template(SourceFormat::Grib2);
        mapping
            .target
            .required_fields
            .retain(|field| field.name != "terrain_height");
        assert!(validate_mapping(&mapping).errors.iter().any(|item| {
            item.code == "target_contract_missing_canonical_requirement"
                && item.field.as_deref() == Some("terrain_height")
        }));
    }
}
