//! Normalize explicitly labelled layer water using the file's layer geometry.

use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct Conversion {
    pub source_units: String,
    pub target_units: &'static str,
    pub thickness_variable: String,
    pub thickness_units: String,
    pub layer_dimension: String,
    pub water_density_kg_m3: f64,
    pub thickness_m: Vec<f64>,
}

fn compact(units: &str) -> String {
    units.to_lowercase().chars().filter(|c| !c.is_whitespace() && !matches!(c, '^' | '*' | '(' | ')')).collect()
}

pub(crate) fn equivalent_depth_scale(units: &str) -> Option<f64> {
    match compact(units).as_str() {
        "kgm-2" | "kg/m2" | "mm" => Some(0.001),
        "m" => Some(1.0),
        _ => None,
    }
}

fn length_scale(units: &str) -> Option<f64> {
    match compact(units).as_str() {
        "m" | "meter" | "meters" | "metre" | "metres" => Some(1.0),
        "cm" | "centimeter" | "centimeters" | "centimetre" | "centimetres" => Some(0.01),
        "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => Some(0.001),
        _ => None,
    }
}

pub(crate) fn normalize(file: &netcrust::File, variable: &netcrust::Variable,
                       values: &mut [f64], thickness_name: &str) -> Result<Conversion, String> {
    let name = variable.name();
    let units = variable.attribute("units").and_then(|a| a.as_string())
        .ok_or_else(|| format!("{name}: layer water conversion requires declared units"))?;
    let scale = equivalent_depth_scale(units)
        .ok_or_else(|| format!("{name}: units {units:?} do not identify layer water mass or equivalent depth"))?;
    let thickness = file.variable(thickness_name)
        .ok_or_else(|| format!("{name}: layer water requires its thickness variable {thickness_name}"))?;
    let thickness_units = thickness.attribute("units").and_then(|a| a.as_string())
        .ok_or_else(|| format!("{thickness_name}: layer thickness needs declared length units"))?;
    let depth_scale = length_scale(thickness_units)
        .ok_or_else(|| format!("{thickness_name}: units {thickness_units:?} do not identify a layer thickness"))?;
    let thickness_dims: Vec<_> = thickness.dimensions().iter()
        .filter(|dim| !(dim.name() == "Time" && dim.len() == 1)).collect();
    if thickness_dims.len() != 1 {
        return Err(format!("{thickness_name}: layer thickness must have one layer dimension, optionally one Time record"));
    }
    let layer_dimension = thickness_dims[0].name();
    let axis = variable.dimensions().iter().position(|dim| dim.name() == layer_dimension)
        .ok_or_else(|| format!("{name}: no dimension matches {thickness_name}'s {layer_dimension}"))?;
    let shape = variable.shape();
    let array = file.read_array_f64(thickness_name)
        .map_err(|error| format!("cannot decode {thickness_name}: {error}"))?;
    let mut depths = array.into_values();
    if depths.len() != shape[axis] || depths.is_empty() {
        return Err(format!("{name}: water and thickness layer counts differ"));
    }
    let number = |key| thickness.attribute(key).and_then(|a| a.as_f64());
    let packed_scale = number("scale_factor").unwrap_or(1.0);
    let packed_offset = number("add_offset").unwrap_or(0.0);
    for depth in &mut depths {
        if number("_FillValue").is_some_and(|v| v == *depth)
            || number("missing_value").is_some_and(|v| v == *depth) {
            return Err(format!("{thickness_name}: layer thickness contains missing data"));
        }
        *depth = (*depth * packed_scale + packed_offset) * depth_scale;
        if !depth.is_finite() || *depth <= 0.0 {
            return Err(format!("{thickness_name}: every layer thickness must be finite and positive"));
        }
    }
    let stride = shape[axis + 1..].iter().product::<usize>();
    for (index, value) in values.iter_mut().enumerate() {
        if value.is_finite() {
            *value = (*value * scale) / depths[(index / stride) % depths.len()];
            if !value.is_finite() {
                return Err(format!("{name}: layer water conversion overflowed a finite value"));
            }
        }
    }
    Ok(Conversion { source_units: units.to_string(), target_units: "m3 m-3",
        thickness_variable: thickness_name.to_string(), thickness_units: thickness_units.to_string(),
        layer_dimension: layer_dimension.to_string(), water_density_kg_m3: 1000.0,
        thickness_m: depths })
}
