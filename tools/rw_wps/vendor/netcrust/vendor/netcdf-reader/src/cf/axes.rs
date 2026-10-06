//! CF axis identification.
//!
//! Determines the role of each coordinate variable (T, X, Y, Z) based on:
//! - The `axis` attribute (most explicit)
//! - The `standard_name` attribute (e.g., "latitude", "longitude", "time")
//! - The `units` attribute (e.g., "degrees_north", "degrees_east")
//! - The `positive` attribute for vertical axes
//!
//! Priority follows CF Conventions Table 1: axis > standard_name > units > positive.

use crate::types::NcVariable;

/// The axis role of a coordinate variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfAxisType {
    /// Time axis.
    T,
    /// Longitude / easting axis.
    X,
    /// Latitude / northing axis.
    Y,
    /// Vertical axis.
    Z,
    /// Cannot be determined.
    Unknown,
}

/// Identify the CF axis type of a variable.
///
/// Checks attributes in CF priority order:
/// 1. `axis` attribute ("X", "Y", "Z", "T")
/// 2. `standard_name` attribute
/// 3. `units` attribute
/// 4. `positive` attribute (vertical indicator)
pub fn identify_axis(var: &NcVariable) -> CfAxisType {
    // 1. Explicit axis attribute (highest priority)
    if let Some(attr) = var.attribute("axis") {
        if let Some(val) = attr.value.as_string() {
            match val.trim().to_uppercase().as_str() {
                "X" => return CfAxisType::X,
                "Y" => return CfAxisType::Y,
                "Z" => return CfAxisType::Z,
                "T" => return CfAxisType::T,
                _ => {}
            }
        }
    }

    // 2. standard_name attribute
    if let Some(attr) = var.attribute("standard_name") {
        if let Some(val) = attr.value.as_string() {
            match val.trim() {
                "latitude" => return CfAxisType::Y,
                "longitude" => return CfAxisType::X,
                "time" => return CfAxisType::T,
                "altitude"
                | "height"
                | "depth"
                | "air_pressure"
                | "atmosphere_hybrid_sigma_pressure_coordinate"
                | "atmosphere_ln_pressure_coordinate"
                | "atmosphere_sigma_coordinate"
                | "ocean_sigma_coordinate"
                | "ocean_s_coordinate"
                | "ocean_double_sigma_coordinate" => return CfAxisType::Z,
                "projection_x_coordinate" | "grid_longitude" => return CfAxisType::X,
                "projection_y_coordinate" | "grid_latitude" => return CfAxisType::Y,
                _ => {}
            }
        }
    }

    // 3. units attribute
    if let Some(attr) = var.attribute("units") {
        if let Some(val) = attr.value.as_string() {
            let lower = val.trim().to_lowercase();
            // Latitude units
            if matches!(
                lower.as_str(),
                "degrees_north"
                    | "degree_north"
                    | "degree_n"
                    | "degrees_n"
                    | "degreen"
                    | "degreesn"
            ) {
                return CfAxisType::Y;
            }
            // Longitude units
            if matches!(
                lower.as_str(),
                "degrees_east" | "degree_east" | "degree_e" | "degrees_e" | "degreee" | "degreese"
            ) {
                return CfAxisType::X;
            }
            // Time units (contains "since")
            if lower.contains(" since ") {
                return CfAxisType::T;
            }
            // Pressure units (common vertical)
            if matches!(
                lower.as_str(),
                "pa" | "hpa" | "mbar" | "millibar" | "bar" | "atm"
            ) {
                return CfAxisType::Z;
            }
        }
    }

    // 4. positive attribute (vertical axis indicator)
    if let Some(attr) = var.attribute("positive") {
        if let Some(val) = attr.value.as_string() {
            let lower = val.trim().to_lowercase();
            if lower == "up" || lower == "down" {
                return CfAxisType::Z;
            }
        }
    }

    CfAxisType::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NcAttrValue, NcAttribute, NcType, NcVariable};

    fn make_var(attrs: Vec<NcAttribute>) -> NcVariable {
        NcVariable {
            name: "test".into(),
            dimensions: vec![],
            dtype: NcType::Float,
            attributes: attrs,
            data_offset: 0,
            _data_size: 0,
            is_record_var: false,
            record_size: 0,
        }
    }

    #[test]
    fn test_axis_attribute() {
        let var = make_var(vec![NcAttribute {
            name: "axis".into(),
            value: NcAttrValue::Chars("X".into()),
        }]);
        assert_eq!(identify_axis(&var), CfAxisType::X);
    }

    #[test]
    fn test_standard_name_latitude() {
        let var = make_var(vec![NcAttribute {
            name: "standard_name".into(),
            value: NcAttrValue::Chars("latitude".into()),
        }]);
        assert_eq!(identify_axis(&var), CfAxisType::Y);
    }

    #[test]
    fn test_standard_name_time() {
        let var = make_var(vec![NcAttribute {
            name: "standard_name".into(),
            value: NcAttrValue::Chars("time".into()),
        }]);
        assert_eq!(identify_axis(&var), CfAxisType::T);
    }

    #[test]
    fn test_units_degrees_north() {
        let var = make_var(vec![NcAttribute {
            name: "units".into(),
            value: NcAttrValue::Chars("degrees_north".into()),
        }]);
        assert_eq!(identify_axis(&var), CfAxisType::Y);
    }

    #[test]
    fn test_units_time_since() {
        let var = make_var(vec![NcAttribute {
            name: "units".into(),
            value: NcAttrValue::Chars("days since 1970-01-01".into()),
        }]);
        assert_eq!(identify_axis(&var), CfAxisType::T);
    }

    #[test]
    fn test_positive_up() {
        let var = make_var(vec![NcAttribute {
            name: "positive".into(),
            value: NcAttrValue::Chars("up".into()),
        }]);
        assert_eq!(identify_axis(&var), CfAxisType::Z);
    }

    #[test]
    fn test_unknown() {
        let var = make_var(vec![]);
        assert_eq!(identify_axis(&var), CfAxisType::Unknown);
    }

    #[test]
    fn test_axis_takes_precedence() {
        // axis="X" should win over standard_name="latitude"
        let var = make_var(vec![
            NcAttribute {
                name: "axis".into(),
                value: NcAttrValue::Chars("X".into()),
            },
            NcAttribute {
                name: "standard_name".into(),
                value: NcAttrValue::Chars("latitude".into()),
            },
        ]);
        assert_eq!(identify_axis(&var), CfAxisType::X);
    }
}
