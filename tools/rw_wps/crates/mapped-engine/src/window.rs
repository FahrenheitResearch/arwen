//! Optional publication support requested in original source index space.
//!
//! The parent supplies the existing interpolation plan's exact stencil after
//! seeing validated source geometry and the resolved canonical inventory.
//! Every field is validated over its full source before its publication;
//! no compressed-message validation or derivation is cropped.
use std::collections::BTreeSet;
use std::io::{BufRead, Write};

use serde_json::{json, Value};

use crate::frames::MappedSourceFrame;
use crate::refusal::{frame_invalid, Result};

pub const SCHEMA: &str = "gpuwm-mapped-atmospheric-window-v1";
pub const FRAMESET_SCHEMA: &str = "gpuwm-mapped-windowed-frameset-v1";

// The same six source operands consumed by the Python regular-join window ABI.
// Having vertical/y/x axes alone does not authorize publication of a diagnostic
// or a prognostic field through a consumer that cannot read that representation.
const CANONICAL_ATMOSPHERIC_FIELDS: &[&str] = &[
    "air_pressure",
    "air_temperature",
    "eastward_wind",
    "geopotential_height",
    "northward_wind",
    "specific_humidity",
];

#[derive(Debug, Clone)]
pub struct Window {
    pub rows: [usize; 2],
    pub columns: [usize; 2],
    pub fields: BTreeSet<String>,
    pub source_shape: [usize; 2],
}

fn pair(value: &Value, name: &str) -> Result<[usize; 2]> {
    let values = value
        .get(name)
        .and_then(Value::as_array)
        .filter(|v| v.len() == 2)
        .ok_or_else(|| frame_invalid(format!("atmospheric window {name} needs two integers")))?;
    let a = values[0].as_u64().and_then(|v| usize::try_from(v).ok());
    let b = values[1].as_u64().and_then(|v| usize::try_from(v).ok());
    match (a, b) {
        (Some(a), Some(b)) => Ok([a, b]),
        _ => Err(frame_invalid(format!(
            "atmospheric window {name} needs two integers"
        ))),
    }
}

fn parse_inventory(
    response: &Value,
    geometry: &Value,
    source_shape: [usize; 2],
    inventory: &[String],
) -> Result<Option<Window>> {
    if response.get("schema").and_then(Value::as_str) != Some(SCHEMA)
        || response.get("geometry") != Some(geometry)
    {
        return Err(frame_invalid(
            "atmospheric window response changed its source geometry identity",
        ));
    }
    if response.get("mode").and_then(Value::as_str) == Some("full") {
        return Ok(None);
    }
    if response.get("mode").and_then(Value::as_str) != Some("window") {
        return Err(frame_invalid("unknown atmospheric window response mode"));
    }
    if pair(response, "source_shape")? != source_shape {
        return Err(frame_invalid(
            "atmospheric window source shape differs from its frame",
        ));
    }
    let rows = pair(response, "rows")?;
    let columns = pair(response, "columns")?;
    for (bounds, size) in [(rows, source_shape[0]), (columns, source_shape[1])] {
        if bounds[0] >= bounds[1] || bounds[1] > size {
            return Err(frame_invalid(
                "atmospheric window must lie inside its original source grid",
            ));
        }
    }
    let requested = response
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| frame_invalid("atmospheric window requires an explicit field inventory"))?;
    let mut fields = BTreeSet::new();
    for value in requested {
        let name = value
            .as_str()
            .ok_or_else(|| frame_invalid("window field names must be strings"))?;
        if !inventory.iter().any(|field| field == name) {
            return Err(frame_invalid(format!("window names absent canonical field {name}")));
        }
        if !CANONICAL_ATMOSPHERIC_FIELDS.contains(&name) {
            return Err(frame_invalid(format!(
                "window field {name} is outside the atmospheric regular-join inventory"
            )));
        }
        if !fields.insert(name.to_owned()) {
            return Err(frame_invalid(format!("window repeats field {name}")));
        }
    }
    if fields.is_empty() {
        return Err(frame_invalid(
            "an empty atmospheric field inventory must use full mode",
        ));
    }
    Ok(Some(Window {
        rows,
        columns,
        fields,
        source_shape,
    }))
}

pub fn parse(response: &Value, geometry: &Value, frame: &MappedSourceFrame) -> Result<Option<Window>> {
    let inventory = frame.fields.iter().map(|field| field.name.clone()).collect::<Vec<_>>();
    let window = parse_inventory(response, geometry,
        [frame.latitude.len(), frame.longitude.len()], &inventory)?;
    if let Some(window) = &window {
        for field in &frame.fields { window.validate_field(field)?; }
    }
    Ok(window)
}

pub fn request(
    frame: &MappedSourceFrame,
    index: usize,
    fingerprint: &str,
) -> Result<Option<Window>> {
    let inventory = frame.fields.iter().map(|field| field.name.clone()).collect::<Vec<_>>();
    let window = request_for_source(&frame.latitude, &frame.longitude,
        &frame.header["grid"], &inventory, index, fingerprint)?;
    if let Some(window) = &window {
        for field in &frame.fields { window.validate_field(field)?; }
    }
    Ok(window)
}

/// The writer asks once the source axes and the complete dependency plan
/// are known. Each requested field's actual axes are checked before a
/// byte of that field is written, and the final manifest is published
/// only after every canonical field and the whole header pass validation.
pub fn request_for_source(
    latitude: &[f64], longitude: &[f64], grid: &Value, inventory: &[String],
    index: usize, fingerprint: &str,
) -> Result<Option<Window>> {
    let geometry = json!({
        "frame_index": index, "grid_fingerprint": fingerprint,
        "latitude_sha256": crate::frames::axis_document(latitude)["sha256"],
        "longitude_sha256": crate::frames::axis_document(longitude)["sha256"],
    });
    let event = json!({
        "schema": crate::PROGRESS_SCHEMA, "event": "atmospheric_window_request",
        "contract": SCHEMA, "geometry": geometry,
        "latitude": crate::frames::axis_document(latitude),
        "longitude": crate::frames::axis_document(longitude),
        "grid": grid,
        "fields": inventory,
    });
    let mut output = std::io::stdout().lock();
    writeln!(output, "{event}")
        .and_then(|_| output.flush())
        .map_err(|e| frame_invalid(format!("cannot request atmospheric support: {e}")))?;
    drop(output);
    let mut line = String::new();
    let count = std::io::stdin()
        .lock()
        .read_line(&mut line)
        .map_err(|e| frame_invalid(format!("cannot read atmospheric support response: {e}")))?;
    if count == 0 {
        return Err(frame_invalid(
            "atmospheric support requester closed before replying",
        ));
    }
    let response: Value = serde_json::from_str(&line)
        .map_err(|e| frame_invalid(format!("invalid atmospheric support response: {e}")))?;
    parse_inventory(&response, &geometry, [latitude.len(), longitude.len()], inventory)
}

impl Window {
    pub fn validate_field(&self, field: &crate::derive::CanonicalField) -> Result<()> {
        if self.fields.contains(&field.name) && (field.axes != ["vertical", "y", "x"]
            || field.values.shape().len() != 3
            || field.values.shape()[1..] != self.source_shape) {
            return Err(frame_invalid(format!(
                "window field {} must retain vertical/y/x source axes", field.name)));
        }
        Ok(())
    }

    pub fn shape(&self, levels: usize) -> [usize; 3] {
        [
            levels,
            self.rows[1] - self.rows[0],
            self.columns[1] - self.columns[0],
        ]
    }

    pub fn crop(&self, values: &[f64], levels: usize) -> Vec<f64> {
        let [ny, nx] = self.source_shape;
        let mut result = Vec::with_capacity(self.shape(levels).iter().product());
        for level in 0..levels {
            for row in self.rows[0]..self.rows[1] {
                let start = (level * ny + row) * nx;
                result.extend_from_slice(&values[start + self.columns[0]..start + self.columns[1]]);
            }
        }
        result
    }

    pub fn document(&self) -> Value {
        json!({"schema": SCHEMA, "source_shape": self.source_shape,
               "rows": self.rows, "columns": self.columns, "fields": self.fields,
               "operation": "regular-parabolic-bilinear-original-fp32-support"})
    }
}

pub fn pressure_levels(values: &[f64], plane_size: usize) -> Result<Vec<f64>> {
    if plane_size == 0
        || values.len() % plane_size != 0
        || values.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(frame_invalid(
            "mapped air pressure must be finite and positive over its full source",
        ));
    }
    values
        .chunks_exact(plane_size)
        .map(|source| {
            let mut plane = source.to_vec();
            let middle = plane.len() / 2;
            let even = plane.len() % 2 == 0;
            let (lower, upper, _) = plane.select_nth_unstable_by(middle, f64::total_cmp);
            let median = if even {
                (*lower
                    .iter()
                    .max_by(|a, b| a.total_cmp(b))
                    .expect("nonempty lower half")
                    + *upper)
                    / 2.0
            } else {
                *upper
            };
            Ok(median / 100.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_rows_preserve_level_and_column_order() {
        let window = Window {
            rows: [1, 3],
            columns: [2, 5],
            fields: BTreeSet::new(),
            source_shape: [4, 6],
        };
        let values: Vec<f64> = (0..48).map(f64::from).collect();
        assert_eq!(
            window.crop(&values, 2),
            vec![8., 9., 10., 14., 15., 16., 32., 33., 34., 38., 39., 40.]
        );
    }

    #[test]
    fn pressure_ladder_uses_full_planes_and_preserves_even_mean_order() {
        assert_eq!(
            pressure_levels(&[100., 400., 200., 300., 500., 800., 600., 700.], 4).unwrap(),
            vec![2.5, 6.5]
        );
        assert_eq!(pressure_levels(&[300., 100., 200.], 3).unwrap(), vec![2.]);
        for bad in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(pressure_levels(&[100., 200., 300., bad], 4).is_err());
        }
    }
}
