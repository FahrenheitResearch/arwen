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

/// Canonical 3-D fields a pressure-level frame completes from its own
/// state when the source leaves them out, at some levels or at all of
/// them.  Keyed on canonical names, so every pressure-level source is
/// served without a line of per-source code: a source that publishes the
/// field keeps every value it publishes, and only the levels it does not
/// publish are derived.
pub const PRESSURE_COMPLETED_FIELDS: [&str; 1] = ["geopotential_height"];

/// What the hypsometric completion reads, by canonical name and unit.
pub const HYPSOMETRIC_OPERANDS: [(&str, &str); 5] = [
    ("air_temperature", "K"),
    ("specific_humidity", "kg kg-1"),
    ("air_pressure", "Pa"),
    ("surface_pressure", "Pa"),
    ("terrain_height", "m"),
];

/// The first reference of a completed field, as
/// `@completed.hypsometric:<values derived>`.  The preparation receipt
/// counts the derived values from it.
pub const HYPSOMETRIC_COMPLETION_REFERENCE: &str = "@completed.hypsometric";

/// Standard gravity, the constant geopotential height is defined by.
const STANDARD_GRAVITY: f64 = 9.80665;

/// Geopotential height where a pressure-level source leaves it out.
///
/// `seed` is the source's own field, NaN on the levels (or cells) the
/// source did not publish, or `None` when it published none of it.  Every
/// finite seed value is kept as it is.  Each missing value is placed by
/// the hypsometric equation, dz = -(Rd / g) Tv dln(p), with virtual
/// temperature taken linear in ln(p) between the published levels:
///
/// * from the nearest level of the same column that carries the source's
///   own height (the level below on a tie), so a derived level sits on
///   the source's own heights wherever there are any;
/// * from the surface otherwise, with the source's surface pressure and
///   terrain height as the anchor and the virtual temperature at the
///   surface interpolated between the levels that bracket it (held at the
///   nearest level when the surface lies outside the ladder).
///
/// The level ladder is the frame's own `air_pressure`, one pressure per
/// level.  `Ok(None)` means an operand is absent, so there is nothing to
/// integrate and the frame keeps its missing-field refusal.
pub fn complete_geopotential_height(
    seed: Option<&CanonicalField>,
    operands: &impl CanonicalFields,
    name: &str,
    units: &str,
) -> Result<Option<(ArrayD<f64>, Vec<String>, Vec<String>)>> {
    let mut resolved: Vec<&CanonicalField> = Vec::with_capacity(HYPSOMETRIC_OPERANDS.len());
    for (operand, unit) in HYPSOMETRIC_OPERANDS {
        let Some(field) = operands.get_field(operand) else {
            return Ok(None);
        };
        if field.units != unit {
            return Err(frame_invalid(format!(
                "{name} is derived hydrostatically from {operand} in {unit}; \
                 this frame carries it in {}",
                field.units
            )));
        }
        resolved.push(field);
    }
    let &[temperature, humidity, pressure, surface_pressure, terrain] = resolved.as_slice() else {
        unreachable!("five operands resolved");
    };
    if units != "m" {
        return Err(frame_invalid(format!(
            "{name} is completed in metres; the mapping declares {units}"
        )));
    }
    let column_axes = ["vertical", "y", "x"];
    for field in [temperature, humidity, pressure].into_iter().chain(seed) {
        if field.axes != column_axes {
            return Err(frame_invalid(format!(
                "{name} is derived hydrostatically on ('vertical', 'y', 'x') \
                 axes; {} has {:?}",
                field.name, field.axes
            )));
        }
    }
    for field in [surface_pressure, terrain] {
        if field.axes != ["y", "x"] {
            return Err(frame_invalid(format!(
                "{name} is derived hydrostatically from {} on ('y', 'x') \
                 axes; it has {:?}",
                field.name, field.axes
            )));
        }
    }
    let shape = temperature.values.shape().to_vec();
    let plane = shape[1] * shape[2];
    let levels = shape[0];
    for field in [humidity, pressure].into_iter().chain(seed) {
        if field.values.shape() != shape.as_slice() {
            return Err(frame_invalid(format!(
                "{name} hydrostatic operands disagree in shape: {} is {:?}, \
                 air_temperature is {shape:?}",
                field.name,
                field.values.shape()
            )));
        }
    }
    for field in [surface_pressure, terrain] {
        if field.values.shape() != &shape[1..] {
            return Err(frame_invalid(format!(
                "{name} hydrostatic operands disagree in shape: {} is {:?}, \
                 the column plane is {:?}",
                field.name,
                field.values.shape(),
                &shape[1..]
            )));
        }
    }
    let pressure_flat = array::contiguous(&pressure.values);
    let mut level_pressure = Vec::with_capacity(levels);
    for level in 0..levels {
        let row = &pressure_flat[level * plane..(level + 1) * plane];
        let first = row[0];
        if !first.is_finite() || first <= 0.0 || row.iter().any(|value| *value != first) {
            return Err(frame_invalid(format!(
                "{name} is derived hydrostatically on a pressure ladder with \
                 one positive pressure per level; air_pressure level {level} \
                 is not one"
            )));
        }
        level_pressure.push(first);
    }
    // Bottom first: the highest pressure is the lowest level.
    let mut order: Vec<usize> = (0..levels).collect();
    order.sort_by(|left, right| level_pressure[*right].total_cmp(&level_pressure[*left]));
    let log_pressure: Vec<f64> = order.iter().map(|level| level_pressure[*level].ln()).collect();
    // For each position, the other positions nearest in ln(p) first, the
    // lower one first on a tie.  One ladder, so one list per position.
    let candidates: Vec<Vec<usize>> = (0..levels)
        .map(|position| {
            let mut others: Vec<usize> = (0..levels).filter(|other| *other != position).collect();
            others.sort_by(|left, right| {
                (log_pressure[position] - log_pressure[*left])
                    .abs()
                    .total_cmp(&(log_pressure[position] - log_pressure[*right]).abs())
                    .then(left.cmp(right))
            });
            others
        })
        .collect();
    let temperature_flat = array::contiguous(&temperature.values);
    let humidity_flat = array::contiguous(&humidity.values);
    let surface_flat = array::contiguous(&surface_pressure.values);
    let terrain_flat = array::contiguous(&terrain.values);
    let mut values: Vec<f64> = match seed {
        Some(field) => array::contiguous(&field.values).into_owned(),
        None => vec![f64::NAN; levels * plane],
    };
    let factor = HYDROSTATIC_RD / STANDARD_GRAVITY;
    let mut virtual_temperature = vec![0.0f64; levels];
    let mut thickness = vec![0.0f64; levels];
    let mut known = vec![false; levels];
    let mut completed = 0usize;
    for cell in 0..plane {
        for position in 0..levels {
            known[position] = values[order[position] * plane + cell].is_finite();
        }
        if known.iter().all(|flag| *flag) {
            continue;
        }
        for position in 0..levels {
            let index = order[position] * plane + cell;
            let value = temperature_flat[index] * (1.0 + HYDROSTATIC_VIRTUAL * humidity_flat[index]);
            if !value.is_finite() || value <= 0.0 {
                return Err(frame_invalid(format!(
                    "{name} is derived hydrostatically where the source leaves \
                     it out, which needs finite positive virtual temperature; \
                     air_temperature and specific_humidity give {} at \
                     column {cell}",
                    crate::refusal::python_float_repr(value)
                )));
            }
            virtual_temperature[position] = value;
        }
        let surface = surface_flat[cell];
        let height = terrain_flat[cell];
        if !surface.is_finite() || surface <= 0.0 || !height.is_finite() {
            return Err(frame_invalid(format!(
                "{name} is derived hydrostatically where the source leaves it \
                 out, which needs finite positive surface_pressure and finite \
                 terrain_height; column {cell} has {} Pa and {} m",
                crate::refusal::python_float_repr(surface),
                crate::refusal::python_float_repr(height)
            )));
        }
        thickness[0] = 0.0;
        for position in 1..levels {
            thickness[position] = thickness[position - 1]
                + 0.5
                    * (virtual_temperature[position - 1] + virtual_temperature[position])
                    * (log_pressure[position - 1] - log_pressure[position]);
        }
        let log_surface = surface.ln();
        let top = levels - 1;
        let surface_thickness = if log_surface >= log_pressure[0] {
            thickness[0] - virtual_temperature[0] * (log_surface - log_pressure[0])
        } else if log_surface <= log_pressure[top] {
            thickness[top] + virtual_temperature[top] * (log_pressure[top] - log_surface)
        } else {
            let below = (0..top)
                .find(|position| {
                    log_pressure[*position] >= log_surface
                        && log_surface > log_pressure[position + 1]
                })
                .expect("an interior surface lies between two levels");
            let weight = (log_pressure[below] - log_surface)
                / (log_pressure[below] - log_pressure[below + 1]);
            let at_surface = virtual_temperature[below]
                + weight * (virtual_temperature[below + 1] - virtual_temperature[below]);
            thickness[below]
                + 0.5 * (virtual_temperature[below] + at_surface) * (log_pressure[below] - log_surface)
        };
        for position in 0..levels {
            if known[position] {
                continue;
            }
            let anchor = candidates[position].iter().copied().find(|other| known[*other]);
            let value = match anchor {
                Some(other) => {
                    values[order[other] * plane + cell]
                        + factor * (thickness[position] - thickness[other])
                }
                None => height + factor * (thickness[position] - surface_thickness),
            };
            values[order[position] * plane + cell] = value;
            completed += 1;
        }
    }
    let mut references = vec![format!("{HYPSOMETRIC_COMPLETION_REFERENCE}:{completed}")];
    for field in seed.into_iter().chain([temperature, humidity, pressure, surface_pressure, terrain]) {
        for reference in &field.source_references {
            if !references.contains(reference) {
                references.push(reference.clone());
            }
        }
    }
    Ok(Some((
        ArrayD::from_shape_vec(IxDyn(&shape), values).expect("column shape is exact"),
        column_axes.iter().map(|axis| (*axis).to_owned()).collect(),
        references,
    )))
}

/// The pressure-level fields a frame of this mapping completes.
///
/// Empty on every other vertical kind: a hybrid or model-level source
/// declares its own hydrostatic derivation, and the completion integrates
/// between pressure levels.
pub fn completed_fields(
    vertical_kind: &str,
    required: &std::collections::BTreeSet<String>,
) -> Vec<&'static str> {
    if vertical_kind != "pressure" {
        return Vec::new();
    }
    PRESSURE_COMPLETED_FIELDS
        .iter()
        .copied()
        .filter(|name| required.contains(*name))
        .collect()
}

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
        Self::validate_values(&self.name, &self.axes, &self.values, self.missing_count)?;
        Ok(self)
    }

    /// Check an owned decoder field without copying its array merely to
    /// construct the canonical wrapper. The streaming writer moves that
    /// array only when its first consumer asks for it.
    pub fn validate_values(
        name: &str,
        axes: &[String],
        values: &ArrayD<f64>,
        missing_count: usize,
    ) -> Result<()> {
        if values.ndim() != axes.len() {
            return Err(frame_invalid(format!(
                "{} rank {} differs from axes {:?}",
                name,
                values.ndim(),
                axes
            )));
        }
        if values.iter().any(|value| value.is_infinite()) {
            return Err(frame_invalid(format!("{} contains infinity", name)));
        }
        if array::count_nan(values) != missing_count {
            return Err(frame_invalid(format!(
                "{} missing count does not match its data",
                name
            )));
        }
        Ok(())
    }
}

/// Derivations read their operands. A writer can lend the exact cached
/// operands without making new arrays to satisfy an owning map's type.
pub trait CanonicalFields {
    fn get_field(&self, name: &str) -> Option<&CanonicalField>;
}

impl CanonicalFields for std::collections::BTreeMap<String, CanonicalField> {
    fn get_field(&self, name: &str) -> Option<&CanonicalField> {
        self.get(name)
    }
}

impl CanonicalFields for std::collections::BTreeMap<String, &CanonicalField> {
    fn get_field(&self, name: &str) -> Option<&CanonicalField> {
        self.get(name).copied()
    }
}

/// The closed catalog's field operands, in the same terms as evaluation.
/// Missing operand declarations remain an unresolved dependency, just as
/// `evaluate_derivation` returns `None`. Coordinate-only pressure has no
/// field dependency; hybrid pressure also consumes the declared surface.
pub fn derivation_dependencies(operation: &Node, vertical: &Node, name: &str) -> Result<Option<Vec<String>>> {
    let kind = operation.get("operation").and_then(Node::as_str)
        .ok_or_else(|| mapping_invalid(format!("derivation for {name} has no operation")))?;
    let labels: &[&str] = match kind {
        "copy" | "soil_surface_node_from_shallowest" | "height_from_interfaces" | "mass_fraction_rebase" => &["source"],
        "wind_speed" => &["u", "v"],
        "geopotential_height" => &["geopotential"],
        "pressure_from_vertical_coordinate" => &[],
        "geopotential_height_hydrostatic" =>
            &["temperature", "specific_humidity", "surface_geopotential_height"],
        "relative_humidity_from_dewpoint" => &["dewpoint", "temperature"],
        "specific_humidity_from_rh" => &["temperature", "pressure", "relative_humidity"],
        "specific_humidity_from_dewpoint" => &["temperature", "pressure", "dewpoint"],
        "volumetric_soil_moisture_from_layer_mass" => &["layer_mass"],
        "surface_pressure_from_sea_level" =>
            &["sea_level_pressure", "level_height", "pressure", "surface_height"],
        other => return Err(mapping_invalid(format!("unsupported derivation operation '{other}'"))),
    };
    let mut dependencies = Vec::with_capacity(labels.len() + 1);
    for label in labels {
        let Some(name) = operation.get(label).and_then(Node::as_str) else { return Ok(None); };
        dependencies.push(name.to_owned());
    }
    if kind == "mass_fraction_rebase" {
        for value in operation.get("exclude").map(Node::items).unwrap_or_default() {
            let dependency = value.as_str().ok_or_else(|| mapping_invalid("mass_fraction_rebase.exclude must name fields"))?;
            dependencies.push(dependency.to_owned());
        }
    }
    if kind == "geopotential_height_hydrostatic" || (kind == "pressure_from_vertical_coordinate"
        && vertical.get("kind").and_then(Node::as_str) == Some("hybrid_sigma_pressure")) {
        let pressure = vertical.get("surface_pressure_field").and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid(format!(
                "{name} requires vertical.surface_pressure_field on a hybrid_sigma_pressure coordinate")))?;
        dependencies.push(pressure.to_owned());
    }
    dependencies.sort();
    dependencies.dedup();
    Ok(Some(dependencies))
}

/// The operand whose shape a derivation's result keeps, as
/// `evaluate_derivation` builds it, and whether the result carries one
/// soil layer more than that operand (the surface node stacked above the
/// shallowest layer).  `None` for a derivation built on the frame's own
/// grid and ladder instead of an operand's shape (pressure from the
/// vertical coordinate, hydrostatic height).
///
/// The frame writer sizes its stream from this before it derives a
/// value.  Sizing every derived soil field on the first soil column
/// instead gave ICON's layer-mass moisture the nine layers of its soil
/// temperature, not the eight it writes, so the up-front disk figure
/// disagreed with the stream the writer then wrote.
pub fn derivation_shape_operand(operation: &Node) -> Option<(&'static str, bool)> {
    match operation.get("operation").and_then(Node::as_str)? {
        "copy" => Some(("source", false)),
        "soil_surface_node_from_shallowest" => Some(("source", true)),
        "wind_speed" => Some(("u", false)),
        "geopotential_height" => Some(("geopotential", false)),
        "relative_humidity_from_dewpoint" => Some(("dewpoint", false)),
        "specific_humidity_from_rh" | "specific_humidity_from_dewpoint" => Some(("temperature", false)),
        "volumetric_soil_moisture_from_layer_mass" => Some(("layer_mass", false)),
        _ => None,
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

/// Below this surface height WRF reduces sea-level pressure along the
/// lowest layer's pressure gradient instead of interpolating.
const SEA_LEVEL_SHALLOW_M: f64 = 50.0;

/// `mapped_source._surface_pressure_from_sea_level`: WRF real's
/// `sfcprs3` relation in f64, one column at a time.
///
/// `level_height` and `pressure` are level-major `(levels, plane)`
/// arrays in the source's own level order; the levels are ordered by
/// the first column's pressure, highest first, and every column must
/// keep that order strictly.  A surface under 50 m takes sea-level
/// pressure plus the lowest layer's gradient times its height; a
/// higher one is interpolated in log pressure between the two levels
/// whose heights bracket it, or, when it lies below every level,
/// between sea level and the second level (WRF's own choice) or the
/// first level under the sea-level pressure.  The loop bounds keep
/// WRF's excluded top levels.
pub fn surface_pressure_from_sea_level(
    sea_level_pressure: &[f64],
    level_height: &[f64],
    pressure: &[f64],
    surface_height: &[f64],
    columns: usize,
    name: &str,
) -> Result<Vec<f64>> {
    let plane = surface_height.len();
    if plane == 0 || sea_level_pressure.len() != plane {
        return Err(frame_invalid(format!(
            "{name} requires sea-level pressure and surface height on the same plane"
        )));
    }
    let levels = pressure.len() / plane;
    if levels < 2 || pressure.len() != levels * plane || level_height.len() != pressure.len() {
        return Err(frame_invalid(format!(
            "{name} requires level heights and pressures on at least two \
             common levels"
        )));
    }
    let mut order: Vec<usize> = (0..levels).collect();
    order.sort_by(|left, right| {
        pressure[right * plane]
            .partial_cmp(&pressure[left * plane])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut values = Vec::with_capacity(plane);
    for cell in 0..plane {
        let z = |k: usize| level_height[order[k] * plane + cell];
        let p = |k: usize| pressure[order[k] * plane + cell];
        let zm = surface_height[cell];
        let slp = sea_level_pressure[cell];
        let refuse = |reason: &str| {
            frame_invalid(format!(
                "{name} cannot reduce sea-level pressure at row {}, column {}: {reason}",
                cell / columns.max(1),
                cell % columns.max(1)
            ))
        };
        if !zm.is_finite() || !slp.is_finite() || slp <= 0.0 {
            return Err(refuse("the sea-level pressure or surface height is not finite and positive"));
        }
        for k in 0..levels {
            if !z(k).is_finite() || !p(k).is_finite() || p(k) <= 0.0 {
                return Err(refuse("a level height or pressure is not finite and positive"));
            }
            if k > 0 && p(k - 1) <= p(k) {
                return Err(refuse("the level pressures do not keep one strict order"));
            }
        }
        let interpolate = |zl: f64, zu: f64, pl: f64, pu: f64| {
            ((pl.ln() * (zm - zu) + pu.ln() * (zl - zm)) / (zl - zu)).exp()
        };
        let result = if zm < SEA_LEVEL_SHALLOW_M {
            slp + (p(0) - p(1)) / (z(0) - z(1)) * zm
        } else if let Some(k) =
            (0..levels.saturating_sub(2)).find(|&k| z(k) <= zm && z(k + 1) > zm)
        {
            interpolate(z(k), z(k + 1), p(k), p(k + 1))
        } else if slp >= p(0) {
            interpolate(0.0, z(1), slp, p(1))
        } else if let Some(k) =
            (0..levels.saturating_sub(3)).find(|&k| slp >= p(k + 1) && slp < p(k))
        {
            interpolate(0.0, z(k + 1), slp, p(k + 1))
        } else {
            return Err(refuse("no level brackets the surface height or the sea-level pressure"));
        };
        if !result.is_finite() || result <= 0.0 {
            return Err(refuse("the reduced pressure is not finite and positive"));
        }
        values.push(result);
    }
    Ok(values)
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
    available: &'a impl CanonicalFields,
) -> Option<&'a CanonicalField> {
    let name = operation.get(label)?.as_str()?;
    available.get_field(name)
}

/// `mapped_source._evaluate_derivation`.
///
/// `Ok(None)` is the Python `except KeyError: continue` branch: a
/// dependency this pass has not derived yet, which the caller retries.
pub fn evaluate_derivation(
    operation: &Node,
    available: &impl CanonicalFields,
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
        let Some(resolved) = available.get_field(pressure_name) else {
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
        "mass_fraction_rebase" => {
            let Some(source) = dependency(operation, "source", available) else { return Ok(None); };
            let names = operation.get("exclude").map(Node::items).unwrap_or_default();
            if names.is_empty() {
                return Err(mapping_invalid("mass_fraction_rebase requires excluded mass-fraction fields"));
            }
            let mut excluded = Vec::new();
            for key in names {
                let key = key.as_str().ok_or_else(|| mapping_invalid("mass_fraction_rebase.exclude must name fields"))?;
                let Some(value) = available.get_field(key) else { return Ok(None); };
                excluded.push(value);
            }
            for fraction in std::iter::once(source).chain(excluded.iter().copied()) {
                if fraction.units != "kg kg-1" || fraction.axes != source.axes
                    || fraction.values.shape() != source.values.shape()
                    || fraction.values.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(frame_invalid(format!("{name} requires finite nonnegative mass fractions on identical axes")));
                }
            }
            let mut denominator = ArrayD::from_elem(source.values.raw_dim(), 1.0);
            let mut references = source.source_references.clone();
            for fraction in excluded {
                denominator -= &fraction.values;
                references.extend(fraction.source_references.clone());
            }
            if denominator.iter().any(|v| *v <= 0.0) {
                return Err(frame_invalid(format!("{name} excluded mass fractions leave no positive reference mass")));
            }
            (&source.values / &denominator, source.axes.clone(), references)
        }
        "height_from_interfaces" => {
            let Some(source) = dependency(operation, "source", available) else { return Ok(None); };
            if source.axes != ["half_level", "y", "x"] || source.units != "m" {
                return Err(frame_invalid(format!("{name} requires interface heights in m on half_level,y,x axes")));
            }
            let heights = array::contiguous(&source.values);
            let nz = collection.vertical_values.len();
            let plane = collection.latitude.len() * collection.longitude.len();
            if source.values.shape() != [nz + 1, collection.latitude.len(), collection.longitude.len()]
                || heights.iter().any(|v| !v.is_finite()) {
                return Err(frame_invalid(format!("{name} requires N+1 finite interface heights")));
            }
            let ascending = heights[plane] > heights[0];
            let mut values = Vec::with_capacity(nz * plane);
            for i in 0..nz * plane {
                let delta = heights[i + plane] - heights[i];
                if delta == 0.0 || (delta > 0.0) != ascending {
                    return Err(frame_invalid(format!("{name} interface heights must be strictly ordered without crossing layers")));
                }
                values.push(0.5 * (heights[i] + heights[i + plane]));
            }
            (ArrayD::from_shape_vec(vec![nz, collection.latitude.len(), collection.longitude.len()], values)
                .map_err(|e| frame_invalid(e.to_string()))?,
             vec!["vertical".into(), "y".into(), "x".into()], source.source_references.clone())
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
        "surface_pressure_from_sea_level" => {
            let (Some(sea_level), Some(height), Some(pressure), Some(surface)) = (
                dependency(operation, "sea_level_pressure", available),
                dependency(operation, "level_height", available),
                dependency(operation, "pressure", available),
                dependency(operation, "surface_height", available),
            ) else {
                return Ok(None);
            };
            if sea_level.axes != ["y", "x"] || surface.axes != ["y", "x"] {
                return Err(frame_invalid(format!(
                    "{name} requires sea-level pressure and surface height on \
                     ('y', 'x') axes"
                )));
            }
            if height.axes != ["vertical", "y", "x"]
                || pressure.axes != height.axes
                || pressure.values.shape() != height.values.shape()
                || height.values.shape()[1..] != *surface.values.shape()
                || sea_level.values.shape() != surface.values.shape()
            {
                return Err(frame_invalid(format!(
                    "{name} requires level heights and pressures on one \
                     ('vertical', 'y', 'x') grid over the surface plane"
                )));
            }
            let columns = surface.values.shape()[1];
            let values = surface_pressure_from_sea_level(
                &array::contiguous(&sea_level.values),
                &array::contiguous(&height.values),
                &array::contiguous(&pressure.values),
                &array::contiguous(&surface.values),
                columns,
                name,
            )?;
            let mut references = vec!["@derived.sea_level_reduction".to_owned()];
            references.extend(sea_level.source_references.iter().cloned());
            references.extend(height.source_references.iter().cloned());
            references.extend(pressure.source_references.iter().cloned());
            references.extend(surface.source_references.iter().cloned());
            (
                ArrayD::from_shape_vec(IxDyn(surface.values.shape()), values)
                    .expect("one value per surface cell"),
                surface.axes.clone(),
                references,
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

/// Standard gravity for the surface-height derivation (m s-2).
const SURFACE_HEIGHT_GRAVITY: f64 = 9.80665;

/// What [`height_at_surface_pressure`] made, and how, for the receipt.
#[derive(Debug, Clone)]
pub struct SurfaceHeight {
    /// Terrain height (m) on the surface fields' `(y, x)` shape.
    pub values: ArrayD<f64>,
    pub cells: usize,
    /// Cells whose surface lies below the deepest pressure level, where
    /// the layer mean comes from the 2 m temperature and dewpoint.
    pub below_ladder: usize,
    pub minimum: f64,
    pub maximum: f64,
}

/// `mapped_source._height_at_surface_pressure`: the height of each
/// column's surface pressure on its own pressure-level geopotential
/// height, for a source that publishes no surface geopotential.
///
/// The hypsometric equation from the first level above the ground down to
/// the surface pressure, with the layer's mean virtual temperature taken
/// between that level and the surface.  At the surface the temperature
/// and humidity are interpolated linearly in log pressure between the two
/// levels that bracket it; below the deepest level they come from the
/// 2 m temperature and the dewpoint's specific humidity, the surface
/// fields WPS and real extrapolate from.  Measured against the surface
/// geopotential a current ECMWF open-data analysis does publish (1,038,240
/// cells on its 13 levels below 10 hPa): bias -0.35 m, RMS 1.7 m, 99th
/// percentile 7.5 m, largest 43 m; RMS 4.6 m over terrain above 1500 m.
///
/// A surface above the highest level is refused: the column says
/// nothing about the air below such a surface.
#[allow(clippy::too_many_arguments)]
pub fn height_at_surface_pressure(
    levels_pa: &[f64],
    geopotential_height: &ArrayD<f64>,
    temperature: &ArrayD<f64>,
    specific_humidity: &ArrayD<f64>,
    surface_pressure: &ArrayD<f64>,
    surface_temperature: &ArrayD<f64>,
    surface_dewpoint: &ArrayD<f64>,
) -> Result<SurfaceHeight> {
    let surface_shape = surface_pressure.shape().to_vec();
    if surface_shape.len() != 2 {
        return Err(frame_invalid(format!(
            "terrain_height from surface pressure needs (y, x) surface \
             fields; surface pressure has shape {surface_shape:?}"
        )));
    }
    let cells = surface_shape[0] * surface_shape[1];
    let column_shape = [levels_pa.len(), surface_shape[0], surface_shape[1]];
    for (label, field) in [
        ("geopotential height", geopotential_height),
        ("temperature", temperature),
        ("specific humidity", specific_humidity),
    ] {
        if field.shape() != column_shape {
            return Err(frame_invalid(format!(
                "terrain_height from surface pressure needs {label} on \
                 the {} declared levels over the surface grid \
                 {column_shape:?}; got {:?}",
                levels_pa.len(),
                field.shape()
            )));
        }
    }
    for (label, field) in [
        ("surface temperature", surface_temperature),
        ("surface dewpoint", surface_dewpoint),
    ] {
        if field.shape() != surface_shape.as_slice() {
            return Err(frame_invalid(format!(
                "terrain_height from surface pressure needs {label} on the \
                 surface grid {surface_shape:?}; got {:?}",
                field.shape()
            )));
        }
    }
    if levels_pa.is_empty() || levels_pa.iter().any(|level| !(level.is_finite() && *level > 0.0)) {
        return Err(frame_invalid(
            "terrain_height from surface pressure needs positive pressure levels",
        ));
    }
    // Deepest (largest pressure) first.
    let mut order: Vec<usize> = (0..levels_pa.len()).collect();
    order.sort_by(|left, right| levels_pa[*right].total_cmp(&levels_pa[*left]));
    let highest = levels_pa[order[order.len() - 1]];

    let heights = array::contiguous(geopotential_height);
    let heights: &[f64] = &heights;
    let temperatures = array::contiguous(temperature);
    let temperatures: &[f64] = &temperatures;
    let humidities = array::contiguous(specific_humidity);
    let humidities: &[f64] = &humidities;
    let pressures = array::contiguous(surface_pressure);
    let surface_temperatures = array::contiguous(surface_temperature);
    let dewpoints = array::contiguous(surface_dewpoint);
    let scale = HYDROSTATIC_RD / SURFACE_HEIGHT_GRAVITY;

    let mut values = Vec::with_capacity(cells);
    let mut below_ladder = 0usize;
    let mut above_ladder = 0usize;
    let mut unreadable = 0usize;
    for cell in 0..cells {
        let pressure = pressures[cell];
        if !(pressure.is_finite() && pressure > 0.0) {
            unreadable += 1;
            values.push(f64::NAN);
            continue;
        }
        // The first level above the ground: the deepest with p < ps.
        let Some(position) = order.iter().position(|level| levels_pa[*level] < pressure) else {
            above_ladder += 1;
            values.push(f64::NAN);
            continue;
        };
        let above = order[position];
        let at = |field: &[f64], level: usize| field[level * cells + cell];
        let (surface_t, surface_q) = if position == 0 {
            below_ladder += 1;
            let dewpoint = dewpoints[cell];
            let vapour_hpa =
                (10.0 * SVP1) * (SVP2 * (dewpoint - SVPT0) / (dewpoint - SVP3)).exp();
            let humidity = 0.622 * vapour_hpa / (pressure / 100.0 - 0.378 * vapour_hpa);
            (surface_temperatures[cell], humidity)
        } else {
            let below = order[position - 1];
            let weight = (pressure.ln() - levels_pa[above].ln())
                / (levels_pa[below].ln() - levels_pa[above].ln());
            let t_above = at(temperatures, above);
            let q_above = at(humidities, above);
            (
                t_above + (at(temperatures, below) - t_above) * weight,
                q_above + (at(humidities, below) - q_above) * weight,
            )
        };
        let virtual_above =
            at(temperatures, above) * (1.0 + HYDROSTATIC_VIRTUAL * at(humidities, above));
        let virtual_surface = surface_t * (1.0 + HYDROSTATIC_VIRTUAL * surface_q);
        let height = at(heights, above)
            - scale * (0.5 * (virtual_above + virtual_surface)) * (pressure / levels_pa[above]).ln();
        if !height.is_finite() {
            unreadable += 1;
        }
        values.push(height);
    }
    if above_ladder > 0 {
        return Err(frame_invalid(format!(
            "terrain_height from surface pressure: {above_ladder} of {cells} \
             cells have a surface pressure at or above the highest level \
             ({highest} Pa), so the column says nothing about the air below \
             their surface"
        )));
    }
    if unreadable > 0 {
        return Err(frame_invalid(format!(
            "terrain_height from surface pressure: {unreadable} of {cells} \
             cells have no finite surface pressure or column to derive from"
        )));
    }
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(SurfaceHeight {
        values: ArrayD::from_shape_vec(IxDyn(&surface_shape), values)
            .expect("one value per surface cell"),
        cells,
        below_ladder,
        minimum,
        maximum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dry isothermal column: z(p) = (Rd/g) T ln(p0/p) exactly, so the
    /// hypsometric surface height is exact inside and below the ladder.
    fn isothermal_column(levels: &[f64], cells: usize, t: f64) -> (ArrayD<f64>, ArrayD<f64>, ArrayD<f64>) {
        let scale = HYDROSTATIC_RD / SURFACE_HEIGHT_GRAVITY;
        let mut heights = Vec::new();
        for level in levels {
            heights.extend(std::iter::repeat_n(scale * t * (100_000.0 / level).ln(), cells));
        }
        let shape = [levels.len(), 1, cells];
        (
            ArrayD::from_shape_vec(IxDyn(&shape), heights).unwrap(),
            ArrayD::from_elem(IxDyn(&shape), t),
            ArrayD::from_elem(IxDyn(&shape), 0.0),
        )
    }

    #[test]
    fn surface_height_is_the_column_height_at_the_surface_pressure() {
        // Declared out of order on purpose: the ladder is sorted, not trusted.
        let levels = [5000.0, 100_000.0, 85_000.0, 50_000.0, 92_500.0, 70_000.0];
        let t = 250.0;
        let (heights, temperature, humidity) = isothermal_column(&levels, 3, t);
        // Inside the ladder, between two levels, and below the deepest.
        let pressure = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![80_000.0, 92_500.0, 103_000.0]).unwrap();
        let surface_t = ArrayD::from_elem(IxDyn(&[1, 3]), t);
        // A dewpoint this cold carries no measurable vapour.
        let dewpoint = ArrayD::from_elem(IxDyn(&[1, 3]), 100.0);
        let derived = height_at_surface_pressure(
            &levels, &heights, &temperature, &humidity, &pressure, &surface_t, &dewpoint,
        )
        .unwrap();
        let scale = HYDROSTATIC_RD / SURFACE_HEIGHT_GRAVITY;
        for (cell, surface) in [80_000.0f64, 92_500.0, 103_000.0].iter().enumerate() {
            let expected = scale * t * (100_000.0 / surface).ln();
            let got = derived.values[[0, cell]];
            assert!((got - expected).abs() < 1e-6, "cell {cell}: {got} vs {expected}");
        }
        assert_eq!(derived.cells, 3);
        assert_eq!(derived.below_ladder, 1);
        assert!(derived.minimum < 0.0 && derived.maximum > 1500.0);
    }

    #[test]
    fn a_surface_above_the_highest_level_is_refused_by_count() {
        let levels = [85_000.0, 100_000.0];
        let (heights, temperature, humidity) = isothermal_column(&levels, 2, 280.0);
        let pressure = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![90_000.0, 60_000.0]).unwrap();
        let surface_t = ArrayD::from_elem(IxDyn(&[1, 2]), 280.0);
        let dewpoint = ArrayD::from_elem(IxDyn(&[1, 2]), 270.0);
        let refusal = height_at_surface_pressure(
            &levels, &heights, &temperature, &humidity, &pressure, &surface_t, &dewpoint,
        )
        .unwrap_err();
        assert!(refusal.message.contains("1 of 2 cells"), "{}", refusal.message);
        assert!(refusal.message.contains("85000 Pa"), "{}", refusal.message);
    }

    #[test]
    fn humid_air_below_the_ladder_is_thicker_than_dry_air() {
        let levels = [85_000.0, 100_000.0];
        let (heights, temperature, humidity) = isothermal_column(&levels, 1, 290.0);
        let pressure = ArrayD::from_elem(IxDyn(&[1, 1]), 102_000.0);
        let surface_t = ArrayD::from_elem(IxDyn(&[1, 1]), 290.0);
        let height = |dewpoint: f64| {
            height_at_surface_pressure(
                &levels,
                &heights,
                &temperature,
                &humidity,
                &pressure,
                &surface_t,
                &ArrayD::from_elem(IxDyn(&[1, 1]), dewpoint),
            )
            .unwrap()
            .values[[0, 0]]
        };
        // Below the 1000 hPa level the surface is lower; moist air makes
        // the layer thicker, so the surface lies further down still.
        assert!(height(288.0) < height(100.0));
        assert!(height(100.0) < 0.0);
    }

    #[test]
    fn layer_mass_uses_declared_thickness_without_a_magnitude_guess_or_clip() {
        let mut mass = constant_field("soil_water", &["soil", "y", "x"], &[2, 1, 2], 0.0);
        mass.units = "kg m-2".to_owned();
        mass.location = "soil".to_owned();
        mass.values = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![5.0, 7.5, 10.0, 80.0]).unwrap();
        let available = std::collections::BTreeMap::from([("soil_water".to_owned(), mass)]);
        let operation = parse_node(r#"{
            "operation": "volumetric_soil_moisture_from_layer_mass",
            "layer_mass": "soil_water", "layer_bounds_m": [[0.0, 0.01], [0.01, 0.05]]
        }"#);
        let declaration = parse_node(r#"{
            "source_axes": ["soil", "y", "x"], "target_axes": ["soil", "y", "x"],
            "units": {"source": "m3 m-3", "target": "m3 m-3"},
            "location": "soil", "missing": {"kind": "reject"}
        }"#);
        let field = FieldSpec { name: "soil_fraction".to_owned(), raw: &declaration };
        let (values, axes, _) = evaluate_derivation(
            &operation, &available, &hybrid_collection(), &field, "soil_fraction", &vertical_node(),
        ).unwrap().unwrap();
        assert_eq!(array::contiguous(&values), vec![0.5, 0.75, 0.25, 2.0]);
        assert_eq!(axes, vec!["soil", "y", "x"]);
    }

    /// One column on five isobaric levels, listed top first as a
    /// pressure-level source declares them: 500 to 1000 hPa.
    const SLP_LEVELS_PA: [f64; 5] = [50_000.0, 70_000.0, 85_000.0, 92_500.0, 100_000.0];
    const SLP_HEIGHTS_M: [f64; 5] = [5_600.0, 3_000.0, 1_500.0, 800.0, 100.0];

    fn sea_level_column(slp: f64, terrain: f64) -> Vec<f64> {
        surface_pressure_from_sea_level(
            &[slp], &SLP_HEIGHTS_M, &SLP_LEVELS_PA, &[terrain], 1, "surface_pressure",
        )
        .unwrap()
    }

    #[test]
    fn a_low_surface_takes_sea_level_pressure_along_the_lowest_gradient() {
        // (1000 - 925 hPa) over (100 - 800 m), times 30 m, added to MSLP.
        let expected = 101_300.0 + (100_000.0 - 92_500.0) / (100.0 - 800.0) * 30.0;
        assert_eq!(sea_level_column(101_300.0, 30.0), vec![expected]);
        assert_eq!(sea_level_column(101_300.0, 0.0), vec![101_300.0]);
    }

    #[test]
    fn a_bracketed_surface_is_interpolated_in_log_pressure() {
        let zm = 1_200.0;
        let expected = ((92_500.0f64.ln() * (zm - 1_500.0) + 85_000.0f64.ln() * (800.0 - zm))
            / (800.0 - 1_500.0))
            .exp();
        let got = sea_level_column(101_300.0, zm);
        assert_eq!(got, vec![expected]);
        assert!(got[0] < 92_500.0 && got[0] > 85_000.0, "{}", got[0]);
    }

    #[test]
    fn a_surface_under_every_level_interpolates_from_sea_level_to_the_second() {
        // 60 m sits below the 1000 hPa height (100 m) and MSLP is above
        // 1000 hPa, so WRF interpolates between sea level and 925 hPa.
        let zm = 60.0;
        let expected = ((101_300.0f64.ln() * (zm - 800.0) + 92_500.0f64.ln() * (0.0 - zm))
            / (0.0 - 800.0))
            .exp();
        assert_eq!(sea_level_column(101_300.0, zm), vec![expected]);
    }

    #[test]
    fn the_source_level_order_does_not_change_the_answer() {
        let reversed_heights: Vec<f64> = SLP_HEIGHTS_M.iter().rev().copied().collect();
        let reversed_levels: Vec<f64> = SLP_LEVELS_PA.iter().rev().copied().collect();
        for terrain in [0.0, 30.0, 60.0, 1_200.0] {
            let reversed = surface_pressure_from_sea_level(
                &[101_300.0], &reversed_heights, &reversed_levels, &[terrain], 1, "surface_pressure",
            )
            .unwrap();
            assert_eq!(reversed, sea_level_column(101_300.0, terrain));
        }
    }

    #[test]
    fn a_column_out_of_the_first_columns_order_refuses_by_position() {
        // Two cells; the second column's pressures run the other way.
        let pressure = [85_000.0, 100_000.0, 92_500.0, 92_500.0, 100_000.0, 85_000.0];
        let height = [1_500.0, 100.0, 800.0, 800.0, 100.0, 1_500.0];
        let refusal = surface_pressure_from_sea_level(
            &[101_300.0, 101_300.0], &height, &pressure, &[10.0, 10.0], 2, "surface_pressure",
        )
        .unwrap_err();
        assert!(refusal.message.contains("row 0, column 1"), "{}", refusal.message);
        assert!(refusal.message.contains("strict order"), "{}", refusal.message);
    }

    #[test]
    fn the_sea_level_derivation_reads_its_four_declared_operands() {
        let collection = DecodedCollection {
            latitude: vec![40.0],
            longitude: vec![-100.0, -99.75],
            vertical_values: SLP_LEVELS_PA.to_vec(),
            direct: std::collections::BTreeMap::new(),
            source_cycles: std::collections::BTreeMap::new(),
            grid_fingerprint: "fixture-grid".to_owned(),
            hybrid_a: Vec::new(),
            hybrid_b: Vec::new(),
        };
        let mut available = std::collections::BTreeMap::new();
        let mut slp = constant_field("air_pressure_at_mean_sea_level", &["y", "x"], &[1, 2], 101_300.0);
        slp.location = "surface".to_owned();
        let mut terrain = constant_field("terrain_height", &["y", "x"], &[1, 2], 0.0);
        terrain.values = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![30.0, 1_200.0]).unwrap();
        let mut height = constant_field("geopotential_height", &["vertical", "y", "x"], &[5, 1, 2], 0.0);
        height.values = ArrayD::from_shape_vec(
            IxDyn(&[5, 1, 2]),
            SLP_HEIGHTS_M.iter().flat_map(|value| [*value, *value]).collect(),
        )
        .unwrap();
        let mut pressure = constant_field("air_pressure", &["vertical", "y", "x"], &[5, 1, 2], 0.0);
        pressure.values = ArrayD::from_shape_vec(
            IxDyn(&[5, 1, 2]),
            SLP_LEVELS_PA.iter().flat_map(|value| [*value, *value]).collect(),
        )
        .unwrap();
        let operation = parse_node(
            r#"{"name": "psfc", "operation": "surface_pressure_from_sea_level",
                "sea_level_pressure": "air_pressure_at_mean_sea_level",
                "level_height": "geopotential_height",
                "pressure": "air_pressure",
                "surface_height": "terrain_height"}"#,
        );
        let field_node = parse_node(
            r#"{"source_axes": ["y", "x"], "target_axes": ["y", "x"],
                "units": {"source": "Pa", "target": "Pa"},
                "location": "surface", "missing": {"kind": "reject"}}"#,
        );
        let field = FieldSpec { name: "surface_pressure".to_owned(), raw: &field_node };
        let vertical = parse_node(r#"{"kind": "pressure", "units": "Pa", "positive": "down"}"#);
        available.insert(slp.name.clone(), slp);
        available.insert(height.name.clone(), height);
        available.insert(pressure.name.clone(), pressure);
        // Terrain is borrowed in a composed frame and may arrive last:
        // until it does the derivation waits rather than refusing.
        assert!(evaluate_derivation(
            &operation, &available, &collection, &field, "surface_pressure", &vertical,
        )
        .unwrap()
        .is_none());
        available.insert(terrain.name.clone(), terrain);
        let (values, axes, references) = evaluate_derivation(
            &operation, &available, &collection, &field, "surface_pressure", &vertical,
        )
        .unwrap()
        .expect("every operand is present");
        assert_eq!(axes, vec!["y", "x"]);
        let flat = array::contiguous(&values);
        assert_eq!(flat[0], sea_level_column(101_300.0, 30.0)[0]);
        assert_eq!(flat[1], sea_level_column(101_300.0, 1_200.0)[0]);
        assert_eq!(references[0], "@derived.sea_level_reduction");
        assert!(references.contains(&"@test.terrain_height".to_owned()));
        let dependencies = derivation_dependencies(&operation, &vertical, "surface_pressure")
            .unwrap()
            .unwrap();
        assert_eq!(
            dependencies,
            vec!["air_pressure", "air_pressure_at_mean_sea_level", "geopotential_height", "terrain_height"]
        );
    }

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

    #[test]
    fn height_interfaces_preserve_columns_and_reject_crossings() {
        let collection = hybrid_collection();
        let mut source = constant_field("coordinate_height", &["half_level", "y", "x"], &[4, 2, 2], 0.0);
        source.units = "m".to_owned();
        source.values = ArrayD::from_shape_vec(IxDyn(&[4, 2, 2]),
            [23000.0, 19000.0, 3000.0, 0.0].iter()
                .flat_map(|height| (0..4).map(move |cell| height + cell as f64)).collect()).unwrap();
        let mut available = std::collections::BTreeMap::new();
        available.insert("coordinate_height".to_owned(), source);
        let operation = parse_node(r#"{"operation":"height_from_interfaces","source":"coordinate_height"}"#);
        let field_node = parse_node(r#"{"source_axes":["vertical","y","x"],"target_axes":["vertical","y","x"],"units":{"source":"m","target":"m"}}"#);
        let field = FieldSpec { name: "geopotential_height".to_owned(), raw: &field_node };
        let vertical = parse_node(r#"{"kind":"model_level","units":"1"}"#);
        let (values, axes, _) = evaluate_derivation(&operation, &available, &collection,
            &field, "geopotential_height", &vertical).unwrap().unwrap();
        assert_eq!(axes, ["vertical", "y", "x"]);
        let expected: Vec<f64> = [21000.0, 11000.0, 1500.0].iter()
            .flat_map(|height| (0..4).map(move |cell| height + cell as f64)).collect();
        assert_eq!(array::contiguous(&values), expected);
        available.get_mut("coordinate_height").unwrap().values[[2, 0, 0]] = 25000.0;
        let error = evaluate_derivation(&operation, &available, &collection,
            &field, "geopotential_height", &vertical).unwrap_err();
        assert!(error.message.contains("crossing layers"));
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

    // ---- complete_geopotential_height: the isothermal answers pinned in
    // tests/test_pressure_height_completion.py ----

    const COLUMN_PRESSURE: [f64; 3] = [100_000.0, 85_000.0, 50_000.0];
    const COLUMN_TEMPERATURE: f64 = 250.0;
    const COLUMN_HUMIDITY: f64 = 0.004;
    const COLUMN_SURFACE_PRESSURE: f64 = 95_000.0;
    const COLUMN_TERRAIN: f64 = 500.0;

    fn isothermal_height(pressure: f64) -> f64 {
        let virtual_temperature =
            COLUMN_TEMPERATURE * (1.0 + HYDROSTATIC_VIRTUAL * COLUMN_HUMIDITY);
        COLUMN_TERRAIN
            + HYDROSTATIC_RD / STANDARD_GRAVITY
                * virtual_temperature
                * (COLUMN_SURFACE_PRESSURE / pressure).ln()
    }

    fn isothermal_operands() -> std::collections::BTreeMap<String, CanonicalField> {
        let column = ["vertical", "y", "x"];
        let mut pressure = constant_field("air_pressure", &column, &[3, 1, 2], 0.0);
        pressure.values = ArrayD::from_shape_vec(
            IxDyn(&[3, 1, 2]),
            COLUMN_PRESSURE.iter().flat_map(|level| [*level, *level]).collect(),
        )
        .unwrap();
        let mut operands = std::collections::BTreeMap::new();
        for (name, units, axes, shape, value) in [
            ("air_temperature", "K", &column[..], &[3, 1, 2][..], COLUMN_TEMPERATURE),
            ("specific_humidity", "kg kg-1", &column[..], &[3, 1, 2][..], COLUMN_HUMIDITY),
            ("surface_pressure", "Pa", &["y", "x"][..], &[1, 2][..], COLUMN_SURFACE_PRESSURE),
            ("terrain_height", "m", &["y", "x"][..], &[1, 2][..], COLUMN_TERRAIN),
        ] {
            let mut field = constant_field(name, axes, shape, value);
            field.units = units.to_owned();
            operands.insert(name.to_owned(), field);
        }
        pressure.units = "Pa".to_owned();
        operands.insert("air_pressure".to_owned(), pressure);
        operands
    }

    #[test]
    fn a_column_with_no_height_is_integrated_from_the_surface() {
        let operands = isothermal_operands();
        let (values, axes, references) =
            complete_geopotential_height(None, &operands, "geopotential_height", "m")
                .unwrap()
                .unwrap();
        assert_eq!(axes, vec!["vertical", "y", "x"]);
        assert_eq!(references[0], "@completed.hypsometric:6");
        let values = array::contiguous(&values);
        for (level, pressure) in COLUMN_PRESSURE.iter().enumerate() {
            for cell in 0..2 {
                let observed = values[level * 2 + cell];
                assert!(
                    (observed - isothermal_height(*pressure)).abs() < 1e-9,
                    "level {level}: {observed} against {}",
                    isothermal_height(*pressure)
                );
            }
        }
    }

    #[test]
    fn published_heights_are_kept_and_anchor_the_missing_levels() {
        let operands = isothermal_operands();
        let published = 5600.0;
        let mut seed = constant_field("geopotential_height", &["vertical", "y", "x"], &[3, 1, 2], f64::NAN);
        seed.units = "m".to_owned();
        seed.values = ArrayD::from_shape_vec(
            IxDyn(&[3, 1, 2]),
            vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, published, published],
        )
        .unwrap();
        seed.missing_count = 4;
        let (values, _axes, references) =
            complete_geopotential_height(Some(&seed), &operands, "geopotential_height", "m")
                .unwrap()
                .unwrap();
        assert_eq!(references[0], "@completed.hypsometric:4");
        assert!(references.contains(&"@test.geopotential_height".to_owned()));
        let values = array::contiguous(&values);
        assert_eq!(values[4], published);
        assert_eq!(values[5], published);
        let offset = published - isothermal_height(COLUMN_PRESSURE[2]);
        for level in 0..2 {
            let expected = isothermal_height(COLUMN_PRESSURE[level]) + offset;
            assert!((values[level * 2] - expected).abs() < 1e-9, "{}", values[level * 2]);
        }
    }

    #[test]
    fn a_frame_without_surface_pressure_has_nothing_to_integrate() {
        let mut operands = isothermal_operands();
        operands.remove("surface_pressure");
        assert!(complete_geopotential_height(None, &operands, "geopotential_height", "m")
            .unwrap()
            .is_none());
    }

    #[test]
    fn completion_is_limited_to_pressure_levels_that_require_height() {
        let required: std::collections::BTreeSet<String> =
            ["geopotential_height".to_owned()].into_iter().collect();
        assert_eq!(completed_fields("pressure", &required), vec!["geopotential_height"]);
        assert!(completed_fields("hybrid", &required).is_empty());
        assert!(completed_fields("pressure", &std::collections::BTreeSet::new()).is_empty());
    }
}
