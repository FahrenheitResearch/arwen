//! The closed derivation catalog and the canonical field it produces.
//!
//! Port of `mapped_source._evaluate_derivation` plus the two humidity
//! relations it borrows from `gpuwm.ingest.real`.  Every expression keeps
//! numpy's operation ORDER, because the parity contract compares sha256 of
//! the resulting bytes: `values * scale + offset` stays two operations,
//! `100.0 * exp(k * (1/T - 1/D))` keeps its parenthesisation, and the
//! Bolton chain keeps its `max(candidate, 1e-6)` floor on the same side of
//! the validity test.

use ndarray::{ArrayD, IxDyn};

use crate::array;
use crate::assemble::DecodedCollection;
use crate::model::FieldSpec;
use crate::node::Node;
use crate::refusal::{frame_invalid, mapping_invalid, Result};

/// WRF `module_model_constants` Teten/Bolton saturation constants.
const SVP1: f64 = 0.6112;
const SVP2: f64 = 17.67;
const SVP3: f64 = 29.65;
const SVPT0: f64 = 273.15;

/// `mapped_source.CanonicalField`.
#[derive(Debug, Clone)]
pub struct CanonicalField {
    pub name: String,
    pub units: String,
    pub axes: Vec<String>,
    pub location: String,
    pub staggering: String,
    pub values: ArrayD<f64>,
    pub missing_count: usize,
    pub source_references: Vec<String>,
}

impl CanonicalField {
    /// The `__post_init__` invariants: rank agrees with the axes, no
    /// infinities survive, and the recorded missing count is the array's.
    pub fn validated(self) -> Result<Self> {
        if self.values.ndim() != self.axes.len() {
            return Err(frame_invalid(format!(
                "{} rank {} differs from axes {:?}",
                self.name,
                self.values.ndim(),
                self.axes
            )));
        }
        if self.values.iter().any(|value| value.is_infinite()) {
            return Err(frame_invalid(format!("{} contains infinity", self.name)));
        }
        if array::count_nan(&self.values) != self.missing_count {
            return Err(frame_invalid(format!(
                "{} missing count does not match its data",
                self.name
            )));
        }
        Ok(self)
    }
}

/// `gpuwm.ingest.real._surface_relative_humidity` — ungrib's 2 m RH from
/// dewpoint (WPS `rrpr.F:compute_rh_dewpt`), deliberately unclipped.
pub fn surface_relative_humidity(dewpoint: &[f64], temperature: &[f64]) -> Vec<f64> {
    let xlv_over_rv = 2.5e6 / 461.5;
    dewpoint
        .iter()
        .zip(temperature.iter())
        .map(|(d, t)| 100.0 * (xlv_over_rv * (1.0 / t - 1.0 / d)).exp())
        .collect()
}

/// `gpuwm.ingest.real._saturation_mixing_ratio` (WRF `rh_to_mxrat1`).
pub fn saturation_mixing_ratio(
    temperature: &[f64],
    pressure: &[f64],
    relative_humidity: &[f64],
) -> Vec<f64> {
    temperature
        .iter()
        .zip(pressure.iter())
        .zip(relative_humidity.iter())
        .map(|((t, p), rh)| {
            let rh = rh.clamp(0.0, 100.0);
            let es_hpa =
                (rh * 0.01) * (10.0 * SVP1) * (SVP2 * (t - SVPT0) / (t - SVP3)).exp();
            let candidate = 0.622 * es_hpa / (p / 100.0 - es_hpa);
            // rh_to_mxrat1's own EPS = 0.622, NOT module ep_2 = 0.62175.
            let valid = *t != 0.0 && es_hpa.is_finite() && es_hpa < p / 100.0;
            if valid {
                candidate.max(1.0e-6)
            } else {
                1.0e-6
            }
        })
        .collect()
}

/// `mapped_source._specific_humidity_from_rh`.
fn specific_humidity_from_rh(
    relative_humidity: &[f64],
    temperature: &[f64],
    pressure: &[f64],
) -> Vec<f64> {
    saturation_mixing_ratio(temperature, pressure, relative_humidity)
        .into_iter()
        .map(|mixing_ratio| mixing_ratio / (1.0 + mixing_ratio))
        .collect()
}

fn dependency<'a>(
    operation: &Node,
    label: &str,
    available: &'a std::collections::BTreeMap<String, CanonicalField>,
) -> Option<&'a CanonicalField> {
    let name = operation.get(label)?.as_str()?;
    available.get(name)
}

/// `mapped_source._evaluate_derivation`.
///
/// `Ok(None)` is the Python `except KeyError: continue` branch: a
/// dependency this pass has not derived yet, which the caller retries.
pub fn evaluate_derivation(
    operation: &Node,
    available: &std::collections::BTreeMap<String, CanonicalField>,
    collection: &DecodedCollection,
    field: &FieldSpec<'_>,
    name: &str,
) -> Result<Option<(ArrayD<f64>, Vec<String>, Vec<String>)>> {
    let kind = operation
        .get("operation")
        .and_then(Node::as_str)
        .ok_or_else(|| mapping_invalid(format!("derivation for {name} has no operation")))?;
    let source_axes = field.source_axes()?;
    let target_axes = field.target_axes()?;

    let (raw, axes, references): (ArrayD<f64>, Vec<String>, Vec<String>) = match kind {
        "copy" => {
            let Some(source) = dependency(operation, "source", available) else {
                return Ok(None);
            };
            (
                source.values.clone(),
                source.axes.clone(),
                source.source_references.clone(),
            )
        }
        "wind_speed" => {
            let (Some(u), Some(v)) = (
                dependency(operation, "u", available),
                dependency(operation, "v", available),
            ) else {
                return Ok(None);
            };
            if u.axes != v.axes || u.values.shape() != v.values.shape() {
                return Err(frame_invalid(format!(
                    "{name} wind derivation dependencies disagree"
                )));
            }
            let u_flat = array::contiguous(&u.values);
            let v_flat = array::contiguous(&v.values);
            let values: Vec<f64> = u_flat
                .iter()
                .copied()
                .zip(v_flat.iter().copied())
                .map(|(left, right)| left.hypot(right))
                .collect();
            let mut references = u.source_references.clone();
            references.extend(v.source_references.iter().cloned());
            (
                ArrayD::from_shape_vec(IxDyn(u.values.shape()), values)
                    .expect("shape preserved elementwise"),
                u.axes.clone(),
                references,
            )
        }
        "geopotential_height" => {
            let Some(geopotential) = dependency(operation, "geopotential", available) else {
                return Ok(None);
            };
            let gravity = operation
                .field("gravity_m_s2")
                .and_then(Node::as_f64)
                .unwrap_or(9.80665);
            if !gravity.is_finite() || gravity <= 0.0 {
                return Err(mapping_invalid(format!("{name} declares invalid gravity")));
            }
            let values: Vec<f64> = array::contiguous(&geopotential.values)
                .into_iter()
                .map(|value| value / gravity)
                .collect();
            (
                ArrayD::from_shape_vec(IxDyn(geopotential.values.shape()), values)
                    .expect("shape preserved elementwise"),
                geopotential.axes.clone(),
                geopotential.source_references.clone(),
            )
        }
        "pressure_from_vertical_coordinate" => {
            if source_axes != ["vertical", "y", "x"] {
                return Err(mapping_invalid(format!(
                    "{name} pressure derivation currently requires source_axes \
                     ['vertical','y','x']"
                )));
            }
            let levels = &collection.vertical_values;
            let rows = collection.latitude.len();
            let columns = collection.longitude.len();
            let mut values = Vec::with_capacity(levels.len() * rows * columns);
            for level in levels {
                values.extend(std::iter::repeat_n(*level, rows * columns));
            }
            (
                ArrayD::from_shape_vec(IxDyn(&[levels.len(), rows, columns]), values)
                    .expect("broadcast shape is exact"),
                source_axes.clone(),
                vec!["@coordinate.vertical".to_owned()],
            )
        }
        "relative_humidity_from_dewpoint" => {
            let (Some(dewpoint), Some(temperature)) = (
                dependency(operation, "dewpoint", available),
                dependency(operation, "temperature", available),
            ) else {
                return Ok(None);
            };
            if dewpoint.axes != temperature.axes {
                return Err(frame_invalid(format!(
                    "{name} dewpoint/temperature axes disagree"
                )));
            }
            let values = surface_relative_humidity(
                &array::contiguous(&dewpoint.values),
                &array::contiguous(&temperature.values),
            );
            let mut references = dewpoint.source_references.clone();
            references.extend(temperature.source_references.iter().cloned());
            (
                ArrayD::from_shape_vec(IxDyn(dewpoint.values.shape()), values)
                    .expect("shape preserved elementwise"),
                dewpoint.axes.clone(),
                references,
            )
        }
        "specific_humidity_from_rh" | "specific_humidity_from_dewpoint" => {
            let (Some(temperature), Some(pressure)) = (
                dependency(operation, "temperature", available),
                dependency(operation, "pressure", available),
            ) else {
                return Ok(None);
            };
            let (relative_humidity, axes, references) = if kind == "specific_humidity_from_rh" {
                let Some(humidity) = dependency(operation, "relative_humidity", available) else {
                    return Ok(None);
                };
                let mut references = humidity.source_references.clone();
                references.extend(temperature.source_references.iter().cloned());
                references.extend(pressure.source_references.iter().cloned());
                (
                    array::contiguous(&humidity.values).into_owned(),
                    humidity.axes.clone(),
                    references,
                )
            } else {
                let Some(dewpoint) = dependency(operation, "dewpoint", available) else {
                    return Ok(None);
                };
                let mut references = dewpoint.source_references.clone();
                references.extend(temperature.source_references.iter().cloned());
                references.extend(pressure.source_references.iter().cloned());
                (
                    surface_relative_humidity(
                        &array::contiguous(&dewpoint.values),
                        &array::contiguous(&temperature.values),
                    ),
                    dewpoint.axes.clone(),
                    references,
                )
            };
            if axes != temperature.axes || axes != pressure.axes {
                return Err(frame_invalid(format!(
                    "{name} humidity derivation dependency axes disagree"
                )));
            }
            let values = specific_humidity_from_rh(
                &relative_humidity,
                &array::contiguous(&temperature.values),
                &array::contiguous(&pressure.values),
            );
            (
                ArrayD::from_shape_vec(IxDyn(temperature.values.shape()), values)
                    .expect("shape preserved elementwise"),
                axes,
                references,
            )
        }
        "volumetric_soil_moisture_from_layer_mass" => {
            let Some(layer_mass) = dependency(operation, "layer_mass", available) else {
                return Ok(None);
            };
            let Some(soil_axis) = layer_mass.axes.iter().position(|axis| axis == "soil") else {
                return Err(mapping_invalid(format!(
                    "{name} layer-mass derivation requires a soil axis on its \
                     layer_mass dependency"
                )));
            };
            let bounds: Vec<(f64, f64)> = operation
                .get("layer_bounds_m")
                .map(Node::items)
                .unwrap_or(&[])
                .iter()
                .map(|pair| {
                    let items = pair.items();
                    match (
                        items.first().and_then(Node::as_f64),
                        items.get(1).and_then(Node::as_f64),
                    ) {
                        (Some(top), Some(bottom)) => Ok((top, bottom)),
                        _ => Err(mapping_invalid(format!(
                            "{name} layer_bounds_m entries must be [top, bottom] numbers"
                        ))),
                    }
                })
                .collect::<Result<Vec<(f64, f64)>>>()?;
            if layer_mass.values.shape()[soil_axis] != bounds.len() {
                return Err(mapping_invalid(format!(
                    "{name} declares {} soil layer bounds but its layer_mass \
                     column has {} layers",
                    bounds.len(),
                    layer_mass.values.shape()[soil_axis]
                )));
            }
            let density = operation
                .field("water_density_kg_m3")
                .and_then(Node::as_f64)
                .unwrap_or(1000.0);
            let thickness: Vec<f64> = bounds
                .iter()
                .map(|(top, bottom)| bottom - top)
                .collect();
            let shape = layer_mass.values.shape().to_vec();
            let inner: usize = shape[soil_axis + 1..].iter().product();
            let values: Vec<f64> = array::contiguous(&layer_mass.values)
                .iter()
                .copied()
                .enumerate()
                .map(|(position, value)| {
                    let layer = (position / inner.max(1)) % bounds.len();
                    value / (density * thickness[layer])
                })
                .collect();
            (
                ArrayD::from_shape_vec(IxDyn(&shape), values).expect("shape preserved elementwise"),
                layer_mass.axes.clone(),
                layer_mass.source_references.clone(),
            )
        }
        "soil_surface_node_from_shallowest" => {
            let Some(source) = dependency(operation, "source", available) else {
                return Ok(None);
            };
            let Some(soil_axis) = source.axes.iter().position(|axis| axis == "soil") else {
                return Err(mapping_invalid(format!(
                    "{name} surface-node derivation requires a soil axis on its \
                     source dependency"
                )));
            };
            if soil_axis != 0 {
                return Err(mapping_invalid(format!(
                    "{name} surface-node derivation currently requires the soil \
                     axis first; got {:?}",
                    source.axes
                )));
            }
            let shape = source.values.shape().to_vec();
            let plane: usize = shape[1..].iter().product();
            let flat = array::contiguous(&source.values);
            let mut values = Vec::with_capacity(flat.len() + plane);
            values.extend_from_slice(&flat[..plane]);
            values.extend_from_slice(&flat);
            let mut grown = shape.clone();
            grown[0] += 1;
            (
                ArrayD::from_shape_vec(IxDyn(&grown), values)
                    .expect("one extra layer of the same plane"),
                source.axes.clone(),
                source.source_references.clone(),
            )
        }
        other => {
            return Err(mapping_invalid(format!(
                "unsupported derivation operation '{other}'"
            )))
        }
    };

    if axes != source_axes {
        return Err(frame_invalid(format!(
            "derived {name} produced axes {axes:?}, expected {source_axes:?}"
        )));
    }
    let converted = array::unit_transform(raw, field.unit_scale(), field.unit_offset(), name)?;
    let converted = array::transpose_to_target(converted, &source_axes, &target_axes, name)?;
    let mut deduplicated: Vec<String> = Vec::new();
    for reference in references {
        if !deduplicated.contains(&reference) {
            deduplicated.push(reference);
        }
    }
    Ok(Some((converted, target_axes, deduplicated)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dewpoint_relative_humidity_is_the_unclipped_ungrib_relation() {
        // T = D means saturation: exactly 100 %, and the relation is NOT
        // clipped, so a dewpoint above the temperature exceeds 100.
        let saturated = surface_relative_humidity(&[290.0], &[290.0]);
        assert!((saturated[0] - 100.0).abs() < 1e-9, "{}", saturated[0]);
        let supersaturated = surface_relative_humidity(&[292.0], &[290.0]);
        assert!(supersaturated[0] > 100.0, "{}", supersaturated[0]);
    }

    #[test]
    fn saturation_mixing_ratio_floors_at_the_wrf_minimum() {
        let dry = saturation_mixing_ratio(&[250.0], &[100_000.0], &[0.0]);
        assert_eq!(dry[0], 1.0e-6);
    }

    #[test]
    fn saturation_mixing_ratio_clips_relative_humidity_into_zero_to_hundred() {
        let over = saturation_mixing_ratio(&[290.0], &[100_000.0], &[150.0]);
        let hundred = saturation_mixing_ratio(&[290.0], &[100_000.0], &[100.0]);
        assert_eq!(over, hundred);
    }

    #[test]
    fn specific_humidity_stays_below_its_mixing_ratio() {
        let mixing = saturation_mixing_ratio(&[295.0], &[95_000.0], &[80.0]);
        let specific = specific_humidity_from_rh(&[80.0], &[295.0], &[95_000.0]);
        assert!(specific[0] < mixing[0]);
        assert!((specific[0] - mixing[0] / (1.0 + mixing[0])).abs() < 1e-18);
    }
}
