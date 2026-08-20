//! The `rw-wps.mapping.v1` document, as the decode path reads it.
//!
//! This is the port of the READ side of `gpuwm.mapped_source.load_mapping`
//! — the accessors the data path uses, plus the structural refusals a
//! decoder cannot proceed without.  The full authoring grammar (allowed
//! key sets, cross-field policy, the vocabulary a mapping author is held
//! to) stays in Python: `load_mapping` is the mapping's validator of
//! record and lane 3 re-runs it around every engine call, so duplicating
//! all 670 lines of it here would create a second authority that could
//! disagree with the first.  What lives here is what the DECODE needs to
//! be well defined, and every read of a malformed value refuses by name.

use crate::node::Node;
use crate::refusal::{mapping_invalid, Result};

pub const MAPPING_SCHEMA: &str = "rw-wps.mapping.v1";

pub const GRID_FAMILY_REGULAR: &str = "regular_latitude_longitude";
pub const GRID_FAMILY_LAMBERT: &str = "lambert_conformal";

/// Axis unit for projected regular grids: one unit = 100 km along the
/// projection plane (`mapped_source.PROJECTED_AXIS_UNIT_M`).
pub const PROJECTED_AXIS_UNIT_M: f64 = 100_000.0;

/// Canonical wind pairs a grid-relative source must rotate to the earth
/// basis at decode time (`mapped_source._ROTATED_WIND_PAIRS`).
pub const ROTATED_WIND_PAIRS: [(&str, &str); 2] = [
    ("eastward_wind", "northward_wind"),
    ("eastward_wind_10m", "northward_wind_10m"),
];

/// `resolution_flags` bit 0x08: vector components are grid-relative.
pub const GRIB2_GRID_RELATIVE_WIND_BIT: u8 = 0x08;

/// One mapping document plus the identity the frames carry.
pub struct Mapping {
    pub doc: Node,
    pub sha256: String,
    pub path: String,
}

impl Mapping {
    pub fn load(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            crate::refusal::missing_input(format!("cannot read mapping {path}: {error}"))
        })?;
        let sha256 = crate::digest::bytes_sha256(&bytes);
        let doc = Node::parse(&bytes)
            .map_err(|error| mapping_invalid(format!("mapping {path} is not JSON: {error}")))?;
        if !doc.is_object() {
            return Err(mapping_invalid(format!(
                "mapping {path} must be a JSON object"
            )));
        }
        let mapping = Self {
            doc,
            sha256,
            path: path.to_owned(),
        };
        let schema = mapping.string("schema")?;
        if schema != MAPPING_SCHEMA {
            return Err(mapping_invalid(format!(
                "mapping {path} declares schema {schema:?}; this engine \
                 implements {MAPPING_SCHEMA}"
            )));
        }
        let format = mapping.format()?;
        if !matches!(format, "grib1" | "grib2" | "netcdf") {
            return Err(mapping_invalid(format!(
                "unsupported mapped source format {format:?}"
            )));
        }
        Ok(mapping)
    }

    pub fn string(&self, key: &str) -> Result<&str> {
        self.doc
            .get(key)
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid(format!("mapping.{key} must be a string")))
    }

    pub fn format(&self) -> Result<&str> {
        self.string("format")
    }

    pub fn name(&self) -> Result<&str> {
        self.string("name")
    }

    pub fn target(&self) -> Result<&Node> {
        self.doc
            .get("target")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.target must be an object"))
    }

    pub fn coordinates(&self) -> Result<&Node> {
        self.doc
            .get("coordinates")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.coordinates must be an object"))
    }

    pub fn vertical(&self) -> Result<&Node> {
        self.coordinates()?
            .get("vertical")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.coordinates.vertical must be an object"))
    }

    /// `mapping.coordinates.vertical.kind`.
    pub fn vertical_kind(&self) -> Result<&str> {
        self.vertical()?
            .get("kind")
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid("mapping.coordinates.vertical.kind must be a string"))
    }

    /// Inline `vertical.hybrid_a`/`hybrid_b` literal arrays — the
    /// declared-data fallback coefficient channel.  `None` when neither
    /// is declared; both-or-neither is the contract.
    pub fn hybrid_literals(&self) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
        let vertical = self.vertical()?;
        let read = |key: &str| -> Result<Option<Vec<f64>>> {
            let Some(node) = vertical.field(key) else {
                return Ok(None);
            };
            node.items()
                .iter()
                .map(|item| {
                    item.as_f64().ok_or_else(|| {
                        mapping_invalid(format!(
                            "vertical.{key} must be a non-empty numeric list"
                        ))
                    })
                })
                .collect::<Result<Vec<f64>>>()
                .map(Some)
        };
        match (read("hybrid_a")?, read("hybrid_b")?) {
            (None, None) => Ok(None),
            (Some(a), Some(b)) => Ok(Some((a, b))),
            _ => Err(mapping_invalid(
                "vertical.hybrid_a and vertical.hybrid_b must be declared \
                 together; one half of a coefficient pair prices no pressure",
            )),
        }
    }

    /// The declared vertical ladder, in the mapping's own order.  Empty
    /// means "take the levels the file offers" (`_assemble_grib`).
    pub fn declared_levels(&self) -> Result<Vec<f64>> {
        let vertical = self.vertical()?;
        let Some(levels) = vertical.field("levels") else {
            return Ok(Vec::new());
        };
        levels
            .items()
            .iter()
            .map(|item| {
                item.as_f64().ok_or_else(|| {
                    mapping_invalid("mapping.coordinates.vertical.levels must be numbers")
                })
            })
            .collect()
    }

    /// Field names in declared order.
    pub fn field_names(&self) -> Result<Vec<String>> {
        let fields = self
            .doc
            .get("fields")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?;
        Ok(fields
            .entries()
            .iter()
            .map(|(name, _)| name.clone())
            .collect())
    }

    pub fn field(&self, name: &str) -> Result<FieldSpec<'_>> {
        let raw = self
            .doc
            .get("fields")
            .and_then(|fields| fields.get(name))
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid(format!("mapping.fields.{name} must be an object")))?;
        Ok(FieldSpec {
            name: name.to_owned(),
            raw,
        })
    }

    pub fn fields(&self) -> Result<Vec<FieldSpec<'_>>> {
        self.field_names()?
            .into_iter()
            .map(|name| self.field(&name))
            .collect()
    }

    pub fn derivation(&self, name: &str) -> Option<&Node> {
        self.doc
            .field("derivations")?
            .items()
            .iter()
            .find(|item| item.get("name").and_then(Node::as_str) == Some(name))
    }

    pub fn required_field_names(&self) -> Result<Vec<String>> {
        let target = self.target()?;
        let required = target
            .get("required_fields")
            .filter(|node| node.is_array())
            .ok_or_else(|| mapping_invalid("mapping.target.required_fields must be an array"))?;
        required
            .items()
            .iter()
            .map(|item| {
                item.get("name")
                    .and_then(Node::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        mapping_invalid("mapping.target.required_fields[].name must be a string")
                    })
            })
            .collect()
    }

    pub fn soil_layer_count(&self) -> Result<Option<i64>> {
        let Some(node) = self.target()?.field("soil_layer_count") else {
            return Ok(None);
        };
        node.as_i64()
            .map(Some)
            .ok_or_else(|| mapping_invalid("mapping.target.soil_layer_count must be an integer"))
    }

    /// The mapping's normalized grid declaration, defaulting to regular
    /// (`mapped_source._mapping_grid_declaration`).
    pub fn grid_declaration(&self) -> Result<GridDeclaration> {
        let Some(grid) = self.doc.field("grid") else {
            return Ok(GridDeclaration::regular());
        };
        let family = grid
            .get("family")
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid("mapping.grid.family must be a string"))?;
        if family == GRID_FAMILY_REGULAR {
            return Ok(GridDeclaration::regular());
        }
        if family != GRID_FAMILY_LAMBERT {
            return Err(mapping_invalid(format!(
                "unsupported mapping.grid.family {family:?}; supported: \
                 {GRID_FAMILY_REGULAR}, {GRID_FAMILY_LAMBERT}"
            )));
        }
        let wind_basis = grid
            .get("wind_basis")
            .and_then(Node::as_str)
            .unwrap_or_default();
        if !matches!(wind_basis, "earth_relative" | "grid_relative_with_rotation") {
            return Err(mapping_invalid(
                "a lambert_conformal grid must declare wind_basis \
                 'earth_relative' or 'grid_relative_with_rotation'",
            ));
        }
        let parameters = grid
            .get("parameters")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.grid.parameters must be an object"))?;
        let number = |key: &str| -> Result<f64> {
            parameters
                .get(key)
                .and_then(Node::as_f64)
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    mapping_invalid(format!(
                        "mapping.grid.parameters.{key} must be a finite number"
                    ))
                })
        };
        let integer = |key: &str| -> Result<i64> {
            parameters.get(key).and_then(Node::as_i64).ok_or_else(|| {
                mapping_invalid(format!("mapping.grid.parameters.{key} must be an integer"))
            })
        };
        let lambert = LambertParameters {
            latin1: number("latin1")?,
            latin2: number("latin2")?,
            lov: number("lov")?,
            lat1: number("lat1")?,
            lon1: number("lon1")?,
            dx_m: number("dx_m")?,
            dy_m: number("dy_m")?,
            nx: integer("nx")?,
            ny: integer("ny")?,
            earth_radius_m: number("earth_radius_m")?,
            shape_of_earth: integer("shape_of_earth")?,
        };
        if lambert.dx_m <= 0.0 || lambert.dy_m <= 0.0 {
            return Err(mapping_invalid(
                "mapping.grid.parameters dx_m/dy_m must be positive",
            ));
        }
        if !(6.2e6..=6.5e6).contains(&lambert.earth_radius_m) {
            return Err(mapping_invalid(
                "mapping.grid.parameters.earth_radius_m must be a plausible \
                 Earth radius in metres",
            ));
        }
        let span_units = (lambert.nx as f64 * lambert.dx_m).max(lambert.ny as f64 * lambert.dy_m)
            / PROJECTED_AXIS_UNIT_M;
        if span_units >= 180.0 {
            return Err(mapping_invalid(format!(
                "declared projected grid spans {:.0} km; the projected-axis \
                 representation covers grids below 18,000 km",
                span_units * PROJECTED_AXIS_UNIT_M / 1000.0
            )));
        }
        Ok(GridDeclaration {
            family: GRID_FAMILY_LAMBERT,
            wind_basis: wind_basis.to_owned(),
            parameters: Some(lambert),
        })
    }
}

/// The normalized `mapping.grid` block.
#[derive(Debug, Clone)]
pub struct GridDeclaration {
    pub family: &'static str,
    pub wind_basis: String,
    pub parameters: Option<LambertParameters>,
}

impl GridDeclaration {
    pub fn regular() -> Self {
        Self {
            family: GRID_FAMILY_REGULAR,
            wind_basis: "earth_relative".to_owned(),
            parameters: None,
        }
    }

    pub fn is_lambert(&self) -> bool {
        self.family == GRID_FAMILY_LAMBERT
    }

    pub fn rotates_winds(&self) -> bool {
        self.wind_basis == "grid_relative_with_rotation"
    }
}

/// The declared Lambert source grid, in GRIB2's own vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct LambertParameters {
    pub latin1: f64,
    pub latin2: f64,
    pub lov: f64,
    pub lat1: f64,
    pub lon1: f64,
    pub dx_m: f64,
    pub dy_m: f64,
    pub nx: i64,
    pub ny: i64,
    pub earth_radius_m: f64,
    pub shape_of_earth: i64,
}

impl LambertParameters {
    /// `(y_axis, x_axis)` of the declared projected grid, in 100-km units
    /// (`mapped_source._projected_axes`).
    pub fn projected_axes(&self) -> (Vec<f64>, Vec<f64>) {
        let y = (0..self.ny)
            .map(|index| index as f64 * self.dy_m / PROJECTED_AXIS_UNIT_M)
            .collect();
        let x = (0..self.nx)
            .map(|index| index as f64 * self.dx_m / PROJECTED_AXIS_UNIT_M)
            .collect();
        (y, x)
    }
}

/// One `mapping.fields[name]` entry.
#[derive(Debug, Clone)]
pub struct FieldSpec<'a> {
    pub name: String,
    pub raw: &'a Node,
}

impl<'a> FieldSpec<'a> {
    pub fn selectors(&self) -> &'a [Node] {
        self.raw.field("selectors").map(Node::items).unwrap_or(&[])
    }

    pub fn derivation(&self) -> Option<&'a str> {
        self.raw.field("derivation").and_then(Node::as_str)
    }

    pub fn provider(&self) -> Option<&'a str> {
        self.raw.field("provider").and_then(Node::as_str)
    }

    pub fn time_binding(&self) -> Option<&'a str> {
        self.raw.field("time_binding").and_then(Node::as_str)
    }

    pub fn selector_stack_axis(&self) -> Option<&'a str> {
        self.raw.field("selector_stack_axis").and_then(Node::as_str)
    }

    pub fn location(&self) -> Result<&'a str> {
        self.raw
            .get("location")
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid(format!("fields.{}.location must be a string", self.name)))
    }

    pub fn staggering(&self) -> &'a str {
        self.raw
            .field("staggering")
            .and_then(Node::as_str)
            .unwrap_or("none")
    }

    pub fn source_axes(&self) -> Result<Vec<String>> {
        self.axes("source_axes")
    }

    pub fn target_axes(&self) -> Result<Vec<String>> {
        self.axes("target_axes")
    }

    fn axes(&self, key: &str) -> Result<Vec<String>> {
        self.raw
            .get(key)
            .and_then(Node::as_string_list)
            .ok_or_else(|| {
                mapping_invalid(format!("fields.{}.{key} must be a list of axis names", self.name))
            })
    }

    pub fn units_target(&self) -> Result<&'a str> {
        self.raw
            .get("units")
            .and_then(|units| units.get("target"))
            .and_then(Node::as_str)
            .ok_or_else(|| {
                mapping_invalid(format!("fields.{}.units.target must be a string", self.name))
            })
    }

    pub fn unit_scale(&self) -> f64 {
        self.raw
            .get("units")
            .and_then(|units| units.field("scale"))
            .and_then(Node::as_f64)
            .unwrap_or(1.0)
    }

    pub fn unit_offset(&self) -> f64 {
        self.raw
            .get("units")
            .and_then(|units| units.field("offset"))
            .and_then(Node::as_f64)
            .unwrap_or(0.0)
    }

    /// The declared missing policy: `reject`, `allow`, `value`, or
    /// `landmask_water` (`mapped_source` missing-policy grammar).
    pub fn missing_kind(&self) -> Result<&'a str> {
        self.raw
            .get("missing")
            .and_then(|missing| missing.get("kind"))
            .and_then(Node::as_str)
            .ok_or_else(|| {
                mapping_invalid(format!("fields.{}.missing.kind must be a string", self.name))
            })
    }

    pub fn missing_value(&self) -> Result<f64> {
        self.raw
            .get("missing")
            .and_then(|missing| missing.get("value"))
            .and_then(Node::as_f64)
            .ok_or_else(|| {
                mapping_invalid(format!("fields.{}.missing.value must be a number", self.name))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping_from(text: &str) -> Mapping {
        let bytes = text.as_bytes().to_vec();
        Mapping {
            sha256: crate::digest::bytes_sha256(&bytes),
            doc: Node::parse(&bytes).unwrap(),
            path: "<test>".to_owned(),
        }
    }

    #[test]
    fn declared_field_order_is_the_document_order_not_alphabetical() {
        let mapping = mapping_from(
            r#"{"fields": {"zulu": {}, "alpha": {}, "mike": {}}}"#,
        );
        assert_eq!(
            mapping.field_names().unwrap(),
            vec!["zulu".to_owned(), "alpha".to_owned(), "mike".to_owned()]
        );
    }

    #[test]
    fn an_absent_grid_block_declares_the_regular_family() {
        let mapping = mapping_from(r#"{"fields": {}}"#);
        let declaration = mapping.grid_declaration().unwrap();
        assert_eq!(declaration.family, GRID_FAMILY_REGULAR);
        assert!(!declaration.rotates_winds());
    }

    #[test]
    fn a_lambert_grid_without_a_wind_basis_refuses() {
        let mapping = mapping_from(
            r#"{"grid": {"family": "lambert_conformal", "parameters": {}}}"#,
        );
        let refusal = mapping.grid_declaration().unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID);
        assert!(refusal.message.contains("wind_basis"));
    }

    #[test]
    fn projected_axes_are_in_hundred_kilometre_units() {
        let parameters = LambertParameters {
            latin1: 50.0,
            latin2: 50.0,
            lov: 253.0,
            lat1: 1.0,
            lon1: 214.5,
            dx_m: 32463.0,
            dy_m: 32463.0,
            nx: 3,
            ny: 2,
            earth_radius_m: 6371229.0,
            shape_of_earth: 6,
        };
        let (y, x) = parameters.projected_axes();
        assert_eq!(y.len(), 2);
        assert_eq!(x.len(), 3);
        assert_eq!(x[1], 32463.0 / 100_000.0);
    }
}
