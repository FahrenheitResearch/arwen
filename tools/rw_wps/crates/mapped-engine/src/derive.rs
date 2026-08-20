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

/// Hydrostatic constants — `mapped_source._HYDROSTATIC_RD` and friends:
/// ECMWF's own model-level geopotential build-up, so derived heights
/// agree with the provider's archived pressure-level z.
const HYDROSTATIC_RD: f64 = 287.06;
const HYDROSTATIC_VIRTUAL: f64 = 0.609133;
/// The provider's top-of-model clamp: a full hybrid ladder's top
/// interface is 0 Pa, whose logarithm does not exist, so the top full
/// level integrates against 0.1 Pa with alpha = ln 2.
const HYDROSTATIC_TOP_PA: f64 = 0.1;

/// `mapped_source._hybrid_half_level_pressure`: the (count, y, x)
/// ladder p = A + B * ps, gated strictly increasing top-first.
fn hybrid_half_level_pressure(
    collection: &DecodedCollection,
    surface_pressure: &[f64],
    name: &str,
) -> Result<Vec<f64>> {
    if collection.hybrid_a.is_empty() || collection.hybrid_b.is_empty() {
        return Err(frame_invalid(format!(
            "{name} requires resolved hybrid A/B coefficients, which this \
             decoded collection does not carry"
        )));
    }
    if surface_pressure
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(frame_invalid(format!(
            "{name} requires finite positive surface pressure to price \
             the hybrid ladder"
        )));
    }
    let count = collection.hybrid_a.len();
    let plane = surface_pressure.len();
    let mut ladder = Vec::with_capacity(count * plane);
    for level in 0..count {
        let a = collection.hybrid_a[level];
        let b = collection.hybrid_b[level];
        for pressure in surface_pressure {
            ladder.push(a + b * pressure);
        }
    }
    let monotonic = (1..count).all(|level| {
        (0..plane).all(|cell| ladder[level * plane + cell] > ladder[(level - 1) * plane + cell])
    });
    if !monotonic {
        return Err(frame_invalid(format!(
            "{name} hybrid pressure must increase strictly from the top \
             of the atmosphere downward at every cell; the resolved A/B \
             ladder does not (check coefficient order against the \
             declared levels)"
        )));
    }
    Ok(ladder)
}

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
    vertical: &Node,
) -> Result<Option<(ArrayD<f64>, Vec<String>, Vec<String>)>> {
    let kind = operation
        .get("operation")
        .and_then(Node::as_str)
        .ok_or_else(|| mapping_invalid(format!("derivation for {name} has no operation")))?;
    let source_axes = field.source_axes()?;
    let target_axes = field.target_axes()?;
    let vertical_kind = vertical
        .get("kind")
        .and_then(Node::as_str)
        .unwrap_or_default();

    // The declared surface-pressure channel both hybrid derivations
    // consume.  `Ok(None)` when the field is not yet available — the
    // caller's fixpoint loop retries after composition injection.
    let surface_pressure_dependency = || -> Result<Option<&CanonicalField>> {
        let pressure_name = vertical
            .get("surface_pressure_field")
            .and_then(Node::as_str)
            .ok_or_else(|| {
                mapping_invalid(format!(
                    "{name} requires vertical.surface_pressure_field on a \
                     hybrid_sigma_pressure coordinate"
                ))
            })?;
        let Some(resolved) = available.get(pressure_name) else {
            return Ok(None);
        };
        if resolved.axes != ["y", "x"] {
            return Err(frame_invalid(format!(
                "{name} requires the declared surface pressure field \
                 '{pressure_name}' on ('y', 'x') axes; got {:?}",
                resolved.axes
            )));
        }
        Ok(Some(resolved))
    };

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
            let rows = collection.latitude.len();
            let columns = collection.longitude.len();
            if vertical_kind == "hybrid_sigma_pressure" {
                // p = A + B*ps on the resolved ladder: half-level
                // interfaces average to full levels; full-level
                // coefficients state the level pressure directly.
                let Some(pressure) = surface_pressure_dependency()? else {
                    return Ok(None);
                };
                let surface = array::contiguous(&pressure.values);
                let ladder = hybrid_half_level_pressure(collection, &surface, name)?;
                let levels = collection.vertical_values.len();
                let plane = surface.len();
                let values: Vec<f64> = if ladder.len() == (levels + 1) * plane {
                    (0..levels * plane)
                        .map(|position| {
                            0.5 * (ladder[position] + ladder[position + plane])
                        })
                        .collect()
                } else {
                    ladder
                };
                let mut references = vec!["@coordinate.vertical.hybrid".to_owned()];
                references.extend(pressure.source_references.iter().cloned());
                (
                    ArrayD::from_shape_vec(IxDyn(&[levels, rows, columns]), values)
                        .expect("ladder shape is exact"),
                    source_axes.clone(),
                    references,
                )
            } else {
                let levels = &collection.vertical_values;
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
        }
        "geopotential_height_hydrostatic" => {
            let (Some(temperature), Some(humidity), Some(surface_height)) = (
                dependency(operation, "temperature", available),
                dependency(operation, "specific_humidity", available),
                dependency(operation, "surface_geopotential_height", available),
            ) else {
                return Ok(None);
            };
            let Some(pressure) = surface_pressure_dependency()? else {
                return Ok(None);
            };
            if source_axes != ["vertical", "y", "x"] {
                return Err(mapping_invalid(format!(
                    "{name} hydrostatic derivation currently requires \
                     source_axes ['vertical','y','x']"
                )));
            }
            if temperature.axes != source_axes || humidity.axes != source_axes {
                return Err(frame_invalid(format!(
                    "{name} hydrostatic derivation requires temperature and \
                     specific humidity on ('vertical', 'y', 'x') axes"
                )));
            }
            if surface_height.axes != ["y", "x"] {
                return Err(frame_invalid(format!(
                    "{name} hydrostatic derivation requires surface \
                     geopotential height on ('y', 'x') axes"
                )));
            }
            let gravity = operation
                .field("gravity_m_s2")
                .and_then(Node::as_f64)
                .unwrap_or(9.80665);
            if !gravity.is_finite() || gravity <= 0.0 {
                return Err(mapping_invalid(format!("{name} declares invalid gravity")));
            }
            let levels = collection.vertical_values.len();
            let surface = array::contiguous(&pressure.values);
            let plane = surface.len();
            let ladder = hybrid_half_level_pressure(collection, &surface, name)?;
            if ladder.len() != (levels + 1) * plane {
                return Err(frame_invalid(format!(
                    "{name} hydrostatic integration requires half-level \
                     interface coefficients: {levels} levels need {} A/B \
                     values, this source resolves {}",
                    levels + 1,
                    ladder.len() / plane.max(1)
                )));
            }
            // ECMWF's model-level build-up, in the Python engine's own
            // operation order: virtual temperature per full level,
            // geopotential accumulated interface to interface from the
            // surface upward, the full level placed by its alpha.
            let temperature_flat = array::contiguous(&temperature.values);
            let humidity_flat = array::contiguous(&humidity.values);
            let virtual_temperature: Vec<f64> = temperature_flat
                .iter()
                .zip(humidity_flat.iter())
                .map(|(t, q)| t * (1.0 + HYDROSTATIC_VIRTUAL * q))
                .collect();
            if virtual_temperature
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(frame_invalid(format!(
                    "{name} hydrostatic integration requires finite positive \
                     virtual temperature"
                )));
            }
            let log2 = 2.0f64.ln();
            let surface_flat = array::contiguous(&surface_height.values);
            let mut phi_half: Vec<f64> = surface_flat
                .iter()
                .map(|value| gravity * value)
                .collect();
            let mut values = vec![0.0f64; levels * plane];
            for level in (0..levels).rev() {
                for cell in 0..plane {
                    let below = ladder[(level + 1) * plane + cell];
                    let above = ladder[level * plane + cell];
                    let positive_above = above > 0.0;
                    let log_ratio = (below
                        / if positive_above { above } else { HYDROSTATIC_TOP_PA })
                        .ln();
                    let alpha = if positive_above {
                        1.0 - (above / (below - above)) * log_ratio
                    } else {
                        log2
                    };
                    let energy = HYDROSTATIC_RD * virtual_temperature[level * plane + cell];
                    values[level * plane + cell] = (phi_half[cell] + energy * alpha) / gravity;
                    phi_half[cell] += energy * log_ratio;
                }
            }
            let mut references = vec!["@derived.hydrostatic".to_owned()];
            references.extend(temperature.source_references.iter().cloned());
            references.extend(humidity.source_references.iter().cloned());
            references.extend(surface_height.source_references.iter().cloned());
            references.extend(pressure.source_references.iter().cloned());
            let rows = collection.latitude.len();
            let columns = collection.longitude.len();
            (
                ArrayD::from_shape_vec(IxDyn(&[levels, rows, columns]), values)
                    .expect("integration shape is exact"),
                source_axes.clone(),
                references,
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

    // ---- hybrid_sigma_pressure: the numbers proven in the Python
    // reference (tests/test_mapped_hybrid_vertical.py) ----

    const NLEV: usize = 3;
    const A_HALF: [f64; 4] = [0.0, 6000.0, 4000.0, 0.0];
    const B_HALF: [f64; 4] = [0.0, 0.24, 0.7, 1.0];
    const PS: f64 = 100_000.0;
    /// p_half = A + B*ps = [0, 30000, 74000, 100000] Pa.
    const P_HALF: [f64; 4] = [0.0, 30_000.0, 74_000.0, 100_000.0];
    /// Full level = mean of its bounding interfaces.
    const P_FULL: [f64; 3] = [15_000.0, 52_000.0, 87_000.0];

    fn constant_field(name: &str, axes: &[&str], shape: &[usize], value: f64) -> CanonicalField {
        let count: usize = shape.iter().product();
        CanonicalField {
            name: name.to_owned(),
            units: String::new(),
            axes: axes.iter().map(|axis| (*axis).to_owned()).collect(),
            location: "mass".to_owned(),
            staggering: "none".to_owned(),
            values: ArrayD::from_shape_vec(IxDyn(shape), vec![value; count]).unwrap(),
            missing_count: 0,
            source_references: vec![format!("@test.{name}")],
        }
    }

    fn hybrid_collection() -> DecodedCollection {
        DecodedCollection {
            latitude: vec![48.0, 49.0],
            longitude: vec![16.0, 17.0],
            vertical_values: vec![1.0, 2.0, 3.0],
            direct: std::collections::BTreeMap::new(),
            source_cycles: std::collections::BTreeMap::new(),
            grid_fingerprint: "fixture-grid".to_owned(),
            hybrid_a: A_HALF.to_vec(),
            hybrid_b: B_HALF.to_vec(),
        }
    }

    fn hybrid_available() -> std::collections::BTreeMap<String, CanonicalField> {
        let mut available = std::collections::BTreeMap::new();
        available.insert(
            "surface_pressure".to_owned(),
            constant_field("surface_pressure", &["y", "x"], &[2, 2], PS),
        );
        available.insert(
            "air_temperature".to_owned(),
            constant_field("air_temperature", &["vertical", "y", "x"], &[NLEV, 2, 2], 280.0),
        );
        available.insert(
            "specific_humidity".to_owned(),
            constant_field("specific_humidity", &["vertical", "y", "x"], &[NLEV, 2, 2], 0.0),
        );
        available.insert(
            "terrain_height".to_owned(),
            constant_field("terrain_height", &["y", "x"], &[2, 2], 100.0),
        );
        available
    }

    fn parse_node(text: &str) -> Node {
        Node::parse(text.as_bytes()).unwrap()
    }

    fn three_d_field_node() -> Node {
        parse_node(
            r#"{"source_axes": ["vertical", "y", "x"],
                "target_axes": ["vertical", "y", "x"],
                "units": {"source": "Pa", "target": "Pa"},
                "location": "mass", "missing": {"kind": "reject"}}"#,
        )
    }

    fn vertical_node() -> Node {
        parse_node(
            r#"{"kind": "hybrid_sigma_pressure", "units": "1",
                "positive": "down",
                "surface_pressure_field": "surface_pressure"}"#,
        )
    }

    #[test]
    fn hybrid_pressure_is_the_mean_of_its_bounding_half_levels() {
        let collection = hybrid_collection();
        let available = hybrid_available();
        let operation = parse_node(
            r#"{"name": "p", "operation": "pressure_from_vertical_coordinate"}"#,
        );
        let field_node = three_d_field_node();
        let field = FieldSpec {
            name: "air_pressure".to_owned(),
            raw: &field_node,
        };
        let vertical = vertical_node();
        let (values, axes, references) = evaluate_derivation(
            &operation, &available, &collection, &field, "air_pressure", &vertical,
        )
        .unwrap()
        .expect("dependencies are present");
        assert_eq!(axes, vec!["vertical", "y", "x"]);
        let flat = array::contiguous(&values);
        for (level, expected) in P_FULL.iter().enumerate() {
            for cell in 0..4 {
                assert_eq!(flat[level * 4 + cell], *expected);
            }
        }
        assert_eq!(references[0], "@coordinate.vertical.hybrid");
    }

    #[test]
    fn a_non_monotonic_hybrid_ladder_refuses_by_name() {
        let mut collection = hybrid_collection();
        collection.hybrid_a = vec![0.0, 4000.0, 6000.0, 0.0];
        collection.hybrid_b = vec![0.0, 0.7, 0.24, 1.0];
        let available = hybrid_available();
        let operation = parse_node(
            r#"{"name": "p", "operation": "pressure_from_vertical_coordinate"}"#,
        );
        let field_node = three_d_field_node();
        let field = FieldSpec {
            name: "air_pressure".to_owned(),
            raw: &field_node,
        };
        let vertical = vertical_node();
        let refusal = evaluate_derivation(
            &operation, &available, &collection, &field, "air_pressure", &vertical,
        )
        .unwrap_err();
        assert!(refusal.message.contains("strictly"), "{}", refusal.message);
    }

    #[test]
    fn hydrostatic_height_matches_the_isothermal_analytic_answer() {
        // Constant virtual temperature telescopes the half-level
        // accumulation to the analytic z(p) = z_s + (Rd Tv / g) ln(ps/p);
        // ECMWF's alpha places each full level between its interfaces,
        // with alpha = ln 2 against 0.1 Pa at the top.  Same expected
        // numbers as the Python reference test.
        let collection = hybrid_collection();
        let available = hybrid_available();
        let operation = parse_node(
            r#"{"name": "z", "operation": "geopotential_height_hydrostatic",
                "temperature": "air_temperature",
                "specific_humidity": "specific_humidity",
                "surface_geopotential_height": "terrain_height"}"#,
        );
        let field_node = parse_node(
            r#"{"source_axes": ["vertical", "y", "x"],
                "target_axes": ["vertical", "y", "x"],
                "units": {"source": "m", "target": "m"},
                "location": "mass", "missing": {"kind": "reject"}}"#,
        );
        let field = FieldSpec {
            name: "geopotential_height".to_owned(),
            raw: &field_node,
        };
        let vertical = vertical_node();
        let (values, _axes, references) = evaluate_derivation(
            &operation, &available, &collection, &field, "geopotential_height", &vertical,
        )
        .unwrap()
        .expect("dependencies are present");
        let flat = array::contiguous(&values);
        let tv = 280.0;
        let gravity = 9.80665;
        let z_half_1 = 100.0 + (HYDROSTATIC_RD * tv / gravity) * (P_HALF[3] / P_HALF[1]).ln();
        let alpha_bottom = 1.0
            - (P_HALF[2] / (P_HALF[3] - P_HALF[2])) * (P_HALF[3] / P_HALF[2]).ln();
        let expected_bottom = 100.0 + (HYDROSTATIC_RD * tv / gravity) * alpha_bottom;
        let expected_top = z_half_1 + (HYDROSTATIC_RD * tv / gravity) * 2.0f64.ln();
        for cell in 0..4 {
            assert!(
                (flat[2 * 4 + cell] - expected_bottom).abs() < 1e-9 * expected_bottom,
                "bottom {} vs {}",
                flat[2 * 4 + cell],
                expected_bottom
            );
            assert!(
                (flat[cell] - expected_top).abs() < 1e-9 * expected_top,
                "top {} vs {}",
                flat[cell],
                expected_top
            );
            // heights increase upward
            assert!(flat[cell] > flat[4 + cell] && flat[4 + cell] > flat[2 * 4 + cell]);
        }
        assert_eq!(references[0], "@derived.hydrostatic");
    }

    #[test]
    fn hydrostatic_height_requires_interface_coefficients() {
        let mut collection = hybrid_collection();
        collection.hybrid_a = vec![5000.0, 2000.0, 0.0];
        collection.hybrid_b = vec![0.1, 0.5, 0.99];
        let available = hybrid_available();
        let operation = parse_node(
            r#"{"name": "z", "operation": "geopotential_height_hydrostatic",
                "temperature": "air_temperature",
                "specific_humidity": "specific_humidity",
                "surface_geopotential_height": "terrain_height"}"#,
        );
        let field_node = parse_node(
            r#"{"source_axes": ["vertical", "y", "x"],
                "target_axes": ["vertical", "y", "x"],
                "units": {"source": "m", "target": "m"},
                "location": "mass", "missing": {"kind": "reject"}}"#,
        );
        let field = FieldSpec {
            name: "geopotential_height".to_owned(),
            raw: &field_node,
        };
        let vertical = vertical_node();
        let refusal = evaluate_derivation(
            &operation, &available, &collection, &field, "geopotential_height", &vertical,
        )
        .unwrap_err();
        assert!(refusal.message.contains("interface"), "{}", refusal.message);
    }
}
