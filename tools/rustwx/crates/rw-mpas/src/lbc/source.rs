//! The driving-source registry: what a boundary producer is allowed to be
//! told about where its numbers come from.
//!
//! A driving source is **described, never coded**.  A row names the reader
//! that opens the source's files, the geometry that samples them, the kind of
//! state they carry, and — the part that does the real work — the map from the
//! canonical roles this producer needs to whatever that source calls them.
//! Adding a model is adding a row.
//!
//! That is not a claim made in a comment.  Three rows ship, and two of them
//! differ **only** in the map and one column:
//!
//! * `wps-intermediate` — the source-agnostic intermediate every registered
//!   external model already arrives as, on a regular latitude/longitude grid,
//!   carrying a first-guess state.  This is the incumbent row and the
//!   producer's default.
//! * `unstructured-native-stream` — an unstructured spherical mesh carrying a
//!   prognostic state, spelled the way MPAS v8.4.1's own history, init and
//!   restart streams spell it (`u`, `theta`, `rho`, ..., with `xtime`).
//! * `unstructured-port-stream` — the same source kind, spelled the way this
//!   program's own forecast history spells it: the edge wind is `normal_u`,
//!   and there is no `xtime` variable, so the valid time is the caller's.
//!
//! The distance between the last two is two map entries and one column.  A
//! fourth unstructured model — anyone's, ours or not — is the same distance
//! away, and [`crate::lbc::parent`]'s tests pin that by driving the whole
//! producer from a row that exists only in a test fixture.
//!
//! ## Why geometry and not model
//! The row does not name a model.  It names a *grid family*, because that is
//! the only thing a sampler can actually be written against: a regular
//! latitude/longitude array, a projected array, an unstructured sphere.  Every
//! model is a row over one of those; a model that shares a family with an
//! existing row costs no code at all.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{MpasError, MpasResult};

/// The rows that ship with the producer.  A registry file supplied on the
/// command line is merged over this by name.
pub const BUILT_IN_REGISTRY: &str = include_str!("../../registry/driving-sources.json");

/// The row the producer uses when the caller names none: the incumbent
/// source-agnostic intermediate.  Naming it here rather than leaving it
/// implicit is what keeps the receipt able to say which row ran.
pub const DEFAULT_ROW: &str = "wps-intermediate";

/// The file reader a row selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Reader {
    /// The WPS intermediate format: self-describing slab records.
    #[serde(rename = "wps-intermediate")]
    WpsIntermediate,
    /// A NetCDF file whose variables are named by the row's map.
    #[serde(rename = "named-variables")]
    NamedVariables,
}

/// The horizontal sampler a row selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Geometry {
    /// A regular latitude/longitude array, sampled by the sixteen-point and
    /// search kernels `init::hinterp` already carries.
    #[serde(rename = "regular-latlon")]
    RegularLatLon,
    /// Cell centres and their dual triangulation on a sphere, sampled by
    /// [`crate::lbc::sphere`].
    #[serde(rename = "unstructured-sphere")]
    UnstructuredSphere,
}

/// What the source's numbers *are*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StateKind {
    /// Temperature, humidity, height and pressure on the source's own levels,
    /// from which the boundary state must be built.
    #[serde(rename = "first-guess")]
    FirstGuess,
    /// The boundary fields themselves, in the same decoupled convention the
    /// lbc stream is written in: dry density, dry potential temperature,
    /// mixing ratios, edge-normal wind, vertical velocity.
    #[serde(rename = "prognostic")]
    Prognostic,
}

/// A canonical role.  The producer asks for roles; the row says what this
/// source calls them.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Role {
    #[serde(rename = "cell-latitude")]
    CellLatitude,
    #[serde(rename = "cell-longitude")]
    CellLongitude,
    #[serde(rename = "edge-latitude")]
    EdgeLatitude,
    #[serde(rename = "edge-longitude")]
    EdgeLongitude,
    #[serde(rename = "edge-normal-angle")]
    EdgeNormalAngle,
    #[serde(rename = "cells-on-vertex")]
    CellsOnVertex,
    #[serde(rename = "cells-on-cell")]
    CellsOnCell,
    #[serde(rename = "edges-on-cell")]
    EdgesOnCell,
    #[serde(rename = "edges-per-cell")]
    EdgesPerCell,
    /// Height of every layer interface, `nCells` by `nVertLevelsP1`.
    #[serde(rename = "interface-height")]
    InterfaceHeight,
    #[serde(rename = "dry-density")]
    DryDensity,
    #[serde(rename = "dry-potential-temperature")]
    DryPotentialTemperature,
    #[serde(rename = "vapour-mixing-ratio")]
    VapourMixingRatio,
    #[serde(rename = "cloud-mixing-ratio")]
    CloudMixingRatio,
    #[serde(rename = "rain-mixing-ratio")]
    RainMixingRatio,
    #[serde(rename = "edge-normal-wind")]
    EdgeNormalWind,
    #[serde(rename = "vertical-velocity")]
    VerticalVelocity,
    /// The source's own record of its valid time.
    #[serde(rename = "valid-time")]
    ValidTime,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Role::CellLatitude => "cell-latitude",
            Role::CellLongitude => "cell-longitude",
            Role::EdgeLatitude => "edge-latitude",
            Role::EdgeLongitude => "edge-longitude",
            Role::EdgeNormalAngle => "edge-normal-angle",
            Role::CellsOnVertex => "cells-on-vertex",
            Role::CellsOnCell => "cells-on-cell",
            Role::EdgesOnCell => "edges-on-cell",
            Role::EdgesPerCell => "edges-per-cell",
            Role::InterfaceHeight => "interface-height",
            Role::DryDensity => "dry-density",
            Role::DryPotentialTemperature => "dry-potential-temperature",
            Role::VapourMixingRatio => "vapour-mixing-ratio",
            Role::CloudMixingRatio => "cloud-mixing-ratio",
            Role::RainMixingRatio => "rain-mixing-ratio",
            Role::EdgeNormalWind => "edge-normal-wind",
            Role::VerticalVelocity => "vertical-velocity",
            Role::ValidTime => "valid-time",
        }
    }

    /// Why the producer needs this role, in one clause, so a refusal can say
    /// what breaks rather than only what is missing.
    pub fn because(self) -> &'static str {
        match self {
            Role::CellLatitude | Role::CellLongitude => {
                "without it no target point can be located in the source mesh"
            }
            Role::EdgeLatitude | Role::EdgeLongitude => {
                "without it the wind transfer has no source positions to weight"
            }
            Role::EdgeNormalAngle => {
                "without it a source edge's wind is a number with no direction, and projecting \
                 it onto the target's normal is meaningless"
            }
            Role::CellsOnVertex => {
                "without the dual triangulation there is no triangle to interpolate over, and \
                 the transfer would fall back to nearest-cell steps at the boundary"
            }
            Role::CellsOnCell | Role::EdgesOnCell | Role::EdgesPerCell => {
                "without the source's edge neighbourhood the wind fit has no patch to fit over"
            }
            Role::InterfaceHeight => {
                "without the source's own layer heights every column would be remapped against \
                 an invented vertical coordinate"
            }
            Role::DryDensity => "the boundary carries dry density and there is no substitute for it",
            Role::DryPotentialTemperature => {
                "the boundary carries dry potential temperature and there is no substitute for it"
            }
            Role::VapourMixingRatio => {
                "a boundary with no water vapour dries the child's inflow from the first step"
            }
            Role::CloudMixingRatio | Role::RainMixingRatio => {
                "an absent condensate slot is written as zero, which is the reference's own \
                 behaviour; this role is optional"
            }
            Role::EdgeNormalWind => {
                "a boundary with no wind is a wall, and the child would spin its own circulation \
                 against it"
            }
            Role::VerticalVelocity => {
                "the boundary carries vertical velocity; zeroing it discards the parent's own \
                 motion at exactly the ring where the child has nothing else to go on"
            }
            Role::ValidTime => {
                "without the source's own valid time a frame cannot be checked against the \
                 boundary time it was asked for"
            }
        }
    }
}

/// The roles a prognostic unstructured source must map, and the ones it may
/// leave out.
pub const PROGNOSTIC_REQUIRED: [Role; 15] = [
    Role::CellLatitude,
    Role::CellLongitude,
    Role::EdgeLatitude,
    Role::EdgeLongitude,
    Role::EdgeNormalAngle,
    Role::CellsOnVertex,
    Role::CellsOnCell,
    Role::EdgesOnCell,
    Role::EdgesPerCell,
    Role::InterfaceHeight,
    Role::DryDensity,
    Role::DryPotentialTemperature,
    Role::VapourMixingRatio,
    Role::EdgeNormalWind,
    Role::VerticalVelocity,
];

/// One row of the registry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceRow {
    pub name: String,
    pub reader: Reader,
    pub geometry: Geometry,
    pub state: StateKind,
    /// The role map.  Absent for a first-guess row, whose reader is
    /// self-describing.
    #[serde(default)]
    pub variables: BTreeMap<Role, String>,
    /// One line saying what this source is, carried into the receipt.
    #[serde(default)]
    pub notes: String,
}

impl SourceRow {
    /// The source's own name for a role.
    pub fn name_of(&self, role: Role) -> Option<&str> {
        self.variables.get(&role).map(String::as_str)
    }

    /// The source's name for a role, or a refusal that says what breaks.
    pub fn require(&self, role: Role) -> MpasResult<&str> {
        self.name_of(role).ok_or_else(|| {
            MpasError::Refusal(format!(
                "the driving-source row \"{}\" maps no variable to the role {}: {}.  A row is \
                 the whole description of a source, so this is a missing table entry, not a \
                 missing code path",
                self.name,
                role.label(),
                role.because()
            ))
        })
    }

    /// Check the row is internally coherent before anything opens a file.
    pub fn validate(&self) -> MpasResult<()> {
        match (self.reader, self.geometry, self.state) {
            (Reader::WpsIntermediate, Geometry::RegularLatLon, StateKind::FirstGuess) => {
                if !self.variables.is_empty() {
                    return Err(MpasError::Refusal(format!(
                        "the driving-source row \"{}\" reads self-describing intermediates but \
                         also maps {} variable name(s).  The map would be silently ignored, and \
                         a row whose entries do nothing is a row that lies about what it does",
                        self.name,
                        self.variables.len()
                    )));
                }
                Ok(())
            }
            (Reader::NamedVariables, Geometry::UnstructuredSphere, StateKind::Prognostic) => {
                for role in PROGNOSTIC_REQUIRED {
                    self.require(role)?;
                }
                Ok(())
            }
            (reader, geometry, state) => Err(MpasError::Refusal(format!(
                "the driving-source row \"{}\" asks for reader {reader:?} with geometry \
                 {geometry:?} carrying a {state:?} state, and no sampler is written for that \
                 combination.  Two combinations exist: a self-describing intermediate on a \
                 regular latitude/longitude grid carrying a first guess, and named variables on \
                 an unstructured sphere carrying a prognostic state",
                self.name
            ))),
        }
    }
}

/// The loaded registry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Registry {
    pub rows: Vec<SourceRow>,
}

impl Registry {
    /// The rows compiled into the producer.
    pub fn built_in() -> MpasResult<Registry> {
        serde_json::from_str(BUILT_IN_REGISTRY).map_err(|e| {
            MpasError::Refusal(format!("the built-in driving-source registry does not parse: {e}"))
        })
    }

    /// Merge a registry file over the built-in rows, by name.  A file row with
    /// the same name as a built-in one replaces it; a new name is appended.
    pub fn with_file(path: &Path) -> MpasResult<Registry> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            MpasError::Refusal(format!(
                "cannot read the driving-source registry {}: {e}",
                path.display()
            ))
        })?;
        let extra: Registry = serde_json::from_str(&text).map_err(|e| {
            MpasError::Refusal(format!(
                "{} is not a driving-source registry: {e}",
                path.display()
            ))
        })?;
        let mut merged = Registry::built_in()?;
        for row in extra.rows {
            match merged.rows.iter_mut().find(|r| r.name == row.name) {
                Some(slot) => *slot = row,
                None => merged.rows.push(row),
            }
        }
        Ok(merged)
    }

    /// Select a row by name, validated.
    pub fn row(&self, name: &str) -> MpasResult<&SourceRow> {
        let row = self.rows.iter().find(|r| r.name == name).ok_or_else(|| {
            let known: Vec<&str> = self.rows.iter().map(|r| r.name.as_str()).collect();
            MpasError::Refusal(format!(
                "no driving-source row is named \"{name}\".  The registry holds: {}.  Adding a \
                 source is adding a row to the registry, not adding a switch here",
                known.join(", ")
            ))
        })?;
        row.validate()?;
        Ok(row)
    }

    pub fn names(&self) -> Vec<&str> {
        self.rows.iter().map(|r| r.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_registry_parses_and_every_row_validates() {
        let reg = Registry::built_in().unwrap();
        assert!(reg.rows.len() >= 3, "rows: {:?}", reg.names());
        for row in &reg.rows {
            row.validate()
                .unwrap_or_else(|e| panic!("row {} does not validate: {e}", row.name));
        }
    }

    #[test]
    fn the_default_row_is_the_incumbent_intermediate() {
        let reg = Registry::built_in().unwrap();
        let row = reg.row(DEFAULT_ROW).unwrap();
        assert_eq!(row.reader, Reader::WpsIntermediate);
        assert_eq!(row.state, StateKind::FirstGuess);
    }

    /// The claim the arbitrary acceptance test actually makes, checked: the
    /// two unstructured rows differ only in the map, not in reader, geometry
    /// or state kind.
    #[test]
    fn two_unstructured_sources_differ_only_in_their_variable_names() {
        let reg = Registry::built_in().unwrap();
        let a = reg.row("unstructured-native-stream").unwrap();
        let b = reg.row("unstructured-port-stream").unwrap();
        assert_eq!(a.reader, b.reader);
        assert_eq!(a.geometry, b.geometry);
        assert_eq!(a.state, b.state);
        let differing: Vec<Role> = PROGNOSTIC_REQUIRED
            .into_iter()
            .chain([Role::ValidTime])
            .filter(|&r| a.name_of(r) != b.name_of(r))
            .collect();
        // The edge wind is spelled differently, and one of them records its
        // own valid time while the other does not.
        assert!(
            differing.contains(&Role::EdgeNormalWind),
            "differing: {differing:?}"
        );
        assert!(
            differing.len() <= 3,
            "two spellings of one source kind should differ in a handful of entries, not \
             {}: {differing:?}",
            differing.len()
        );
    }

    /// The arbitrary acceptance test, at this seam: if a model's name ever
    /// reaches the code, the seam has been lost and the next model will need
    /// code too.
    ///
    /// The registry file is data and is exempt — naming sources is what it is
    /// for. Everything that executes is not.
    #[test]
    fn no_source_model_is_named_anywhere_in_the_code_that_runs() {
        let code = [
            ("sphere.rs", include_str!("sphere.rs")),
            ("parent.rs", include_str!("parent.rs")),
            ("source.rs", include_str!("source.rs")),
        ];
        // Spellings of driving sources this program can already reach, plus
        // the families the acceptance test names as the ones to come.
        let models = [
            "gfs", "hrrr", "rap", "rrfs", "nam", "gefs", "gdas", "icon", "aifs", "ecmwf", "era5",
            "gem", "hiresw", "href", "sref", "rtma", "urma", "nbm", "refs",
        ];
        for (file, text) in code {
            // The scan stops at the test module: this very list has to spell
            // the names out, and a test that fails on its own assertion text
            // is testing nothing.
            let runs = text.split("#[cfg(test)]").next().unwrap_or(text);
            let lower = runs.to_ascii_lowercase();
            for model in models {
                // Word-ish match: the bare name, not a substring of a longer
                // identifier that happens to contain it.
                let hit = lower.match_indices(model).any(|(i, _)| {
                    let before = lower[..i].chars().next_back();
                    let after = lower[i + model.len()..].chars().next();
                    let boundary = |c: Option<char>| {
                        c.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_')
                    };
                    boundary(before) && boundary(after)
                });
                assert!(
                    !hit,
                    "{file} names the driving source \"{model}\".  A source enters through a \
                     registry row; the moment one is spelled in the code, the next one needs \
                     code too"
                );
            }
        }
    }

    #[test]
    fn an_unknown_row_name_lists_the_registry_and_says_where_to_add_one() {
        let reg = Registry::built_in().unwrap();
        let err = reg.row("a-model-nobody-registered").unwrap_err().to_string();
        assert!(err.contains("a-model-nobody-registered"), "{err}");
        assert!(err.contains(DEFAULT_ROW), "{err}");
        assert!(err.contains("adding a row"), "{err}");
    }

    #[test]
    fn a_prognostic_row_missing_a_role_is_refused_with_the_consequence_named() {
        let mut row = Registry::built_in()
            .unwrap()
            .row("unstructured-native-stream")
            .unwrap()
            .clone();
        row.variables.remove(&Role::VerticalVelocity);
        let err = row.validate().unwrap_err().to_string();
        assert!(err.contains("vertical-velocity"), "{err}");
        assert!(err.contains("discards the parent's own motion"), "{err}");
        assert!(err.contains("missing table entry"), "{err}");
    }

    #[test]
    fn a_combination_with_no_sampler_is_refused_by_name() {
        let row = SourceRow {
            name: "half-described".to_string(),
            reader: Reader::WpsIntermediate,
            geometry: Geometry::UnstructuredSphere,
            state: StateKind::Prognostic,
            variables: BTreeMap::new(),
            notes: String::new(),
        };
        let err = row.validate().unwrap_err().to_string();
        assert!(err.contains("no sampler is written"), "{err}");
    }

    #[test]
    fn an_intermediate_row_carrying_a_map_is_refused_rather_than_ignoring_it() {
        let mut row = Registry::built_in().unwrap().row(DEFAULT_ROW).unwrap().clone();
        row.variables.insert(Role::DryDensity, "rho".to_string());
        let err = row.validate().unwrap_err().to_string();
        assert!(err.contains("silently ignored"), "{err}");
    }

    #[test]
    fn a_registry_file_replaces_a_row_by_name_and_appends_a_new_one() {
        let dir = std::env::temp_dir().join(format!(
            "rw-mpas-registry-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("extra.json");
        let base = Registry::built_in().unwrap();
        let mut clone = base.row("unstructured-native-stream").unwrap().clone();
        clone.name = "a-fourth-model".to_string();
        clone
            .variables
            .insert(Role::EdgeNormalWind, "edge_wind".to_string());
        let doc = Registry { rows: vec![clone] };
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();

        let merged = Registry::with_file(&path).unwrap();
        assert_eq!(merged.rows.len(), base.rows.len() + 1);
        assert_eq!(
            merged.row("a-fourth-model").unwrap().name_of(Role::EdgeNormalWind),
            Some("edge_wind")
        );
        // The built-in rows are untouched.
        assert_eq!(
            merged.row("unstructured-native-stream").unwrap().name_of(Role::EdgeNormalWind),
            base.row("unstructured-native-stream").unwrap().name_of(Role::EdgeNormalWind)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
