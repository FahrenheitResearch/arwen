//! Table-declared sparse-column completion. No source or model dispatch.
//!
//! Interpolate AI-minus-donor anomalies in log pressure. A cosine of the
//! distance to a published AI pressure tapers each anomaly to zero across
//! the declared pressure-distance band. Thus missing column segments use
//! the donor and every published AI knot retains its value. All columns
//! run through the mapped engine's Rayon pool.
use std::collections::BTreeMap;
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Value};
use crate::assemble::{DecodedCollection, DirectValue, TimeKey};
use crate::node::Node;
use crate::refusal::{frame_invalid, mapping_invalid, Result};

#[derive(Debug, Clone)]
pub struct FieldRule {
    pub primary_field: String,
    representation: String,
}

#[derive(Debug, Clone)]
pub struct VerticalMerge {
    pub fields: BTreeMap<String, FieldRule>,
    band: [f64; 2],
    primary_scale: f64,
    donor_scale: f64,
    max_rh: f64,
    column_pressure: bool,
}

fn number(node: &Node, name: &str) -> Result<f64> {
    node.get(name).and_then(Node::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| mapping_invalid(format!("vertical_merge.{name} must be finite")))
}

impl VerticalMerge {
    pub fn parse(node: &Node) -> Result<Self> {
        let allowed = ["fields", "method", "pressure_band_hpa", "primary_pressure_scale",
            "donor_pressure_scale", "max_relative_humidity"];
        if !node.is_object() || node.entries().len() != allowed.len() + usize::from(node.get("pressure_field").is_some())
            || node.entries().iter().any(|(key, _)| !allowed.contains(&key.as_str()) && key != "pressure_field") {
            return Err(mapping_invalid("vertical_merge must declare fields, method, pressure_band_hpa, pressure scales and max_relative_humidity"));
        }
        if node.get("method").and_then(Node::as_str) != Some("cosine_pressure_distance_log_anomaly") {
            return Err(mapping_invalid("unsupported vertical_merge transition method"));
        }
        let band = node.get("pressure_band_hpa").map(Node::items).unwrap_or(&[]);
        let band: Vec<f64> = band.iter().filter_map(Node::as_f64).collect();
        if band.len() != 2 || !band.iter().all(|x| x.is_finite()) || band[0] < 0.0 || band[1] <= band[0] {
            return Err(mapping_invalid("vertical_merge pressure_band_hpa must be increasing nonnegative distances"));
        }
        let primary_scale = number(node, "primary_pressure_scale")?;
        let donor_scale = number(node, "donor_pressure_scale")?;
        if ![1.0, 100.0].contains(&primary_scale) || ![1.0, 100.0].contains(&donor_scale) {
            return Err(mapping_invalid("vertical_merge pressure scales must convert Pa or hPa to Pa"));
        }
        let max_rh = number(node, "max_relative_humidity")?;
        if !(1.0..=1.05).contains(&max_rh) {
            return Err(mapping_invalid("vertical_merge max_relative_humidity must be in [1, 1.05]"));
        }
        let fields = node.get("fields").filter(|x| x.is_object() && !x.entries().is_empty())
            .ok_or_else(|| mapping_invalid("vertical_merge needs field rules"))?;
        let mut rules = BTreeMap::new();
        for (name, rule) in fields.entries() {
            let primary_field = rule.get("primary_field").and_then(Node::as_str)
                .filter(|s| !s.is_empty()).ok_or_else(|| mapping_invalid("vertical_merge needs primary_field"))?;
            let representation = rule.get("representation").and_then(Node::as_str).unwrap_or("");
            if rule.entries().len() != 2 || !["canonical", "relative_humidity_percent"].contains(&representation)
                || (representation == "relative_humidity_percent" && name != "specific_humidity") {
                return Err(mapping_invalid("vertical_merge field representation must be canonical, or RH percent for specific_humidity"));
            }
            rules.insert(name.clone(), FieldRule { primary_field: primary_field.into(), representation: representation.into() });
        }
        let column_pressure = match node.get("pressure_field").and_then(Node::as_str) {
            None => false,
            Some("air_pressure") => true,
            _ => return Err(mapping_invalid("vertical_merge.pressure_field must name air_pressure")),
        };
        Ok(Self { fields: rules, band: [band[0], band[1]], primary_scale, donor_scale, max_rh, column_pressure })
    }

    pub fn validate_coordinates(&self, primary: &Node, donor: &Node) -> Result<()> {
        if self.column_pressure {
            if [primary, donor].iter().any(|coordinate|
                coordinate.get("kind").and_then(Node::as_str) != Some("model_level")
                || coordinate.get("units").and_then(Node::as_str) != Some("1"))
                || self.primary_scale != 1.0 || self.donor_scale != 1.0 {
                return Err(mapping_invalid("column pressure splice requires matching model_level index coordinates and pressure fields in Pa"));
            }
            return Ok(());
        }
        for (coordinate, scale) in [(primary, self.primary_scale), (donor, self.donor_scale)] {
            let expected = match coordinate.get("units").and_then(Node::as_str) {
                Some("Pa") => 1.0,
                Some("hPa") => 100.0,
                _ => return Err(mapping_invalid("vertical_merge requires pressure in Pa or hPa")),
            };
            if coordinate.get("kind").and_then(Node::as_str) != Some("pressure") || scale != expected {
                return Err(mapping_invalid("vertical_merge pressure scale disagrees with pressure coordinate"));
            }
        }
        Ok(())
    }

    pub fn apply(&self, slice: &mut DecodedCollection, donor: &[DirectValue],
                 donor_levels: &[f64], key: &TimeKey, binding: &str) -> Result<Value> {
        use rayon::prelude::*;
        let primary_p: Vec<f64> = slice.vertical_values.iter().map(|x| x * self.primary_scale).collect();
        let donor_p: Vec<f64> = donor_levels.iter().map(|x| x * self.donor_scale).collect();
        if !self.column_pressure {
            monotonic(&primary_p)?;
            monotonic(&donor_p)?;
        }
        let cells = slice.latitude.len() * slice.longitude.len();
        let get = |name: &str| slice.direct.get(&(key.0, key.1.clone(), name.to_owned()));
        let temperature = get("air_temperature");
        let primary_pressure = get("air_pressure");
        let primary_pressure_missing = primary_pressure.map(|field| field.values.iter().filter(|p| !p.is_finite()).count()).unwrap_or(0);
        let donor_pressure = donor.iter().find(|field| field.name == "air_pressure");
        let mut primary_pressure_reordered_columns = 0usize;
        let column_pressures = if self.column_pressure {
            let ai = primary_pressure.ok_or_else(|| frame_invalid("column splice needs primary air_pressure"))?;
            let baseline = donor_pressure.ok_or_else(|| frame_invalid("column splice needs donor air_pressure"))?;
            check_column(ai, primary_p.len(), cells)?;
            check_column(baseline, donor_p.len(), cells)?;
            let ai = crate::array::contiguous(&ai.values);
            let base = crate::array::contiguous(&baseline.values);
            // Model indices identify missing pressure values only. Predicted
            // pressure remains authoritative wherever published and finite.
            let completed: Vec<f64> = crate::threads::install(|| (0..ai.len()).into_par_iter().map(|i| {
                if ai[i].is_finite() { return ai[i]; }
                let level = i / cells;
                donor_p.iter().position(|index| *index == primary_p[level])
                    .map(|k| base[k * cells + i % cells]).unwrap_or(f64::NAN)
            }).collect());
            let invalid = crate::threads::install(|| (0..cells).into_par_iter().find_first(|cell| {
                (0..primary_p.len()).any(|k| !completed[k*cells+cell].is_finite() || completed[k*cells+cell] <= 0.0)
                    || monotonic(&(0..donor_p.len()).map(|k| base[k*cells+cell]).collect::<Vec<_>>()).is_err()
            }));
            if let Some(cell) = invalid { return Err(frame_invalid(format!("column splice requires positive finite primary pressure and strictly monotonic positive donor pressure at cell {cell}"))); }
            primary_pressure_reordered_columns = crate::threads::install(|| (0..cells).into_par_iter().filter(|cell| {
                monotonic(&(0..primary_p.len()).map(|k| completed[k*cells+cell]).collect::<Vec<_>>()).is_err()
            }).count());
            Some((completed, base.to_vec()))
        } else { None };
        let inputs: Vec<_> = self.fields.iter().map(|(name, rule)| {
            let baseline = donor.iter().find(|value| &value.name == name)
                .ok_or_else(|| frame_invalid(format!("vertical_merge donor lacks {name}")))?;
            check_column(baseline, donor_p.len(), cells)?;
            let primary = get(&rule.primary_field);
            if let Some(value) = primary { check_column(value, primary_p.len(), cells)?; }
            if rule.representation == "relative_humidity_percent" {
                let t = temperature.ok_or_else(|| frame_invalid("RH splice needs primary temperature"))?;
                check_column(t, primary_p.len(), cells)?;
            }
            Ok((name, rule, baseline, primary))
        }).collect::<Result<Vec<_>>>()?;
        let results: Vec<Result<(DirectValue, Value)>> = crate::threads::install(|| inputs.par_iter().map(|(name, rule, baseline, primary)| {
            let base = crate::array::contiguous(&baseline.values);
            let ai = primary.map(|value| crate::array::contiguous(&value.values));
            let temp = temperature.map(|value| crate::array::contiguous(&value.values));
            let words = donor_p.len() * cells;
            // Each column builds its anomaly knots once, then evaluates every
            // donor level. Parallel columns avoid a serial core while the GPU
            // waits and avoid rebuilding the knots once per output level.
            let mut columns = vec![0.0; words];
            let statistics = columns.par_chunks_mut(donor_p.len()).enumerate().map(|(cell, column)| {
                let column_primary;
                let column_donor;
                let (primary_p, donor_p) = if let Some((ai, base)) = &column_pressures {
                    column_primary = (0..primary_p.len()).map(|k| ai[k*cells+cell]).collect::<Vec<_>>();
                    column_donor = (0..donor_p.len()).map(|k| base[k*cells+cell]).collect::<Vec<_>>();
                    (&column_primary, &column_donor)
                } else { (&primary_p, &donor_p) };
                let mut knots: Vec<(f64, f64)> = if self.column_pressure && *name == "air_pressure" {
                    // Pressure is the output grid coordinate. The primary
                    // pressure locates forecast knots; it is not an anomaly
                    // to add back onto that same coordinate.
                    Vec::new()
                } else { primary_p.iter().enumerate().filter_map(|(k, pressure)| {
                    let ai = ai.as_ref()?;
                    let mut value = ai[k*cells+cell];
                    if rule.representation == "relative_humidity_percent" {
                        let t = temp.as_ref().unwrap()[k*cells+cell];
                        if !t.is_finite() { return None; }
                        value = specific_humidity(t, *pressure, value * 0.01);
                    }
                    if !value.is_finite() { return None; }
                    let donor_value = interpolate(donor_p, |j| base[j*cells+cell], *pressure)?;
                    Some((*pressure, value - donor_value))
                }).collect() };
                // Pressure samples need not arrive in pressure order. Keep
                // each value paired with its own pressure, and average only
                // coincident pressure anomalies before interpolation.
                let coincident = if self.column_pressure { order_pressure_knots(&mut knots) } else { 0 };
                let mut sum = 0.0;
                let mut donor_only = 0usize;
                let mut primary_only = 0usize;
                for (level, p) in donor_p.iter().copied().enumerate() {
                    let weight = if knots.is_empty() { 0.0 } else {
                        let distance = knots.iter().map(|(pressure, _)| (p-pressure).abs()/100.0).fold(f64::INFINITY, f64::min);
                        let x = ((distance-self.band[0])/(self.band[1]-self.band[0])).clamp(0.0, 1.0);
                        0.5*(1.0+(std::f64::consts::PI*x).cos())
                    };
                    let anomaly = if weight == 0.0 { 0.0 } else { interpolate_clamped(&knots, p) };
                    column[level] = base[level*cells+cell] + weight*anomaly;
                    sum += 1.0-weight;
                    donor_only += usize::from(weight <= 1e-12);
                    primary_only += usize::from(weight >= 1.0-1e-12);
                }
                (sum, donor_only, primary_only, coincident)
            }).reduce(|| (0.0, 0usize, 0usize, 0usize), |a,b| (a.0+b.0,a.1+b.1,a.2+b.2,a.3+b.3));
            if columns.par_iter().any(|value| !value.is_finite()) {
                return Err(frame_invalid(format!("vertical_merge {name} produced a nonfinite value")));
            }
            let output: Vec<f64> = (0..words).into_par_iter().map(|index|
                columns[(index % cells)*donor_p.len() + index/cells]).collect();
            let donor_fraction = statistics.0 / words as f64;
            let donor_only_count = statistics.1;
            let primary_only_count = statistics.2;
            let mut references = baseline.references.clone();
            if let Some(primary) = primary { references.extend(primary.references.iter().cloned()); }
            references.push(format!("vertical_merge:{binding}:cosine_pressure_distance_log_anomaly"));
            let values = ArrayD::from_shape_vec(IxDyn(&[donor_p.len(), slice.latitude.len(), slice.longitude.len()]), output)
                .map_err(|error| frame_invalid(error.to_string()))?;
            let value = DirectValue { name: (*name).clone(), valid_time: key.0, member: key.1.clone(), source_cycle: slice.source_cycles[key],
                axes: vec!["vertical".into(), "y".into(), "x".into()], values, missing_count: 0, references };
            Ok((value, json!({"field": name, "donor_fraction": donor_fraction, "value_count": words,
                "donor_only_value_count": donor_only_count, "primary_only_value_count": primary_only_count,
                "coincident_primary_pressure_samples_averaged": statistics.3})))
        }).collect());
        let mut results = crate::threads::in_order(results)?;
        let t = results.iter().find(|(field, _)| field.name == "air_temperature")
            .ok_or_else(|| frame_invalid("vertical_merge needs air_temperature to validate moisture"))?;
        let t = crate::array::contiguous(&t.0.values).to_vec();
        let mut q_min = f64::INFINITY;
        let mut rh_max: f64 = 0.0;
        let mut clipped = 0usize;
        for (field, _) in &mut results {
            if field.name == "specific_humidity" {
                let q = field.values.as_slice_mut().unwrap();
                for (index, value) in q.iter_mut().enumerate() {
                    let p = column_pressures.as_ref().map(|(_, base)| base[index])
                        .unwrap_or(donor_p[index / cells]);
                    let cap = specific_humidity(t[index], p, self.max_rh);
                    let bounded = value.clamp(0.0, cap);
                    if bounded != *value { clipped += 1; }
                    *value = bounded;
                    q_min = q_min.min(*value);
                    rh_max = rh_max.max(relative_humidity(t[index], p, *value));
                }
            }
        }
        if !q_min.is_finite() { return Err(frame_invalid("vertical_merge needs specific_humidity to validate moisture")); }
        let field_receipts: Vec<_> = results.iter().map(|(_, receipt)| receipt.clone()).collect();
        let donor_fraction = field_receipts.iter().map(|r| r["donor_fraction"].as_f64().unwrap()).sum::<f64>() / field_receipts.len() as f64;
        let surface_values: usize = donor.iter().filter(|value| !self.fields.contains_key(&value.name)).map(|value| value.values.len()).sum();
        let atmospheric_values = donor_p.len() * cells * self.fields.len();
        let binding_donor_fraction = (donor_fraction * atmospheric_values as f64 + surface_values as f64)
            / (atmospheric_values + surface_values) as f64;
        for rule in self.fields.values() {
            if !self.fields.contains_key(&rule.primary_field) {
                slice.direct.remove(&(key.0, key.1.clone(), rule.primary_field.clone()));
            }
        }
        for (value, _) in results { slice.direct.insert((key.0, key.1.clone(), value.name.clone()), value); }
        slice.vertical_values = donor_levels.to_vec();
        slice.hybrid_a.clear(); slice.hybrid_b.clear();
        Ok(json!({"binding": binding, "valid_time": crate::frames::naive_isoformat(key.0),
            "transition_method": "cosine_pressure_distance_log_anomaly", "pressure_band_hpa": self.band,
            "pressure_band_semantics": "distance_to_nearest_published_primary_pressure",
            "pressure_monotonicity": "PASS",
            "pressure_monotonicity_scope": "donor_output_pressure_coordinate",
            "pressure_coordinate": if self.column_pressure { "air_pressure_field" } else { "pressure_levels" },
            "output_levels": donor_levels,
            "output_levels_pa": if self.column_pressure { Value::Null } else { json!(donor_p) },
            "primary_pressure_missing_values_filled": primary_pressure_missing,
            "primary_pressure_nonmonotonic_column_count": primary_pressure_reordered_columns,
            "primary_pressure_sample_ordering": if self.column_pressure { "pressure_value_pairs_sorted; coincident_pressure_anomalies_averaged" } else { "declared_monotonic_pressure_levels" },
            "donor_fraction": donor_fraction, "donor_fraction_definition": "mean interpolation weight over spliced atmospheric values",
            "binding_donor_fraction": binding_donor_fraction,
            "binding_donor_fraction_definition": "mean donor weight over all bound output values, including surface and soil",
            "fields": field_receipts, "moisture_validity": {"status": "PASS", "q_min": q_min, "relative_humidity_max": rh_max,
                "maximum_allowed_relative_humidity": self.max_rh, "bounded_value_count": clipped}}))
    }
}

fn order_pressure_knots(knots: &mut Vec<(f64, f64)>) -> usize {
    knots.sort_by(|a,b| b.0.total_cmp(&a.0));
    let original = knots.len();
    let mut written = 0;
    let mut read = 0;
    while read < original {
        let pressure = knots[read].0;
        let mut end = read + 1;
        let mut sum = knots[read].1;
        while end < original && knots[end].0 == pressure {
            sum += knots[end].1;
            end += 1;
        }
        knots[written] = (pressure, sum / (end-read) as f64);
        written += 1;
        read = end;
    }
    knots.truncate(written);
    original-written
}

fn check_column(value: &DirectValue, levels: usize, cells: usize) -> Result<()> {
    if value.axes != ["vertical", "y", "x"] || value.values.len() != levels * cells {
        return Err(frame_invalid(format!("vertical_merge {} must have declared (vertical,y,x) geometry", value.name)));
    }
    Ok(())
}
fn monotonic(levels: &[f64]) -> Result<()> {
    if levels.is_empty() || levels.iter().any(|x| !x.is_finite() || *x <= 0.0)
        || !(levels.windows(2).all(|p| p[1] > p[0]) || levels.windows(2).all(|p| p[1] < p[0])) {
        return Err(frame_invalid("vertical_merge requires strictly monotonic positive pressure columns"));
    }
    Ok(())
}
fn interpolate(levels: &[f64], sample: impl Fn(usize) -> f64, p: f64) -> Option<f64> {
    if p < levels.iter().copied().fold(f64::INFINITY, f64::min)
        || p > levels.iter().copied().fold(0.0, f64::max) { return None; }
    if let Some(i) = levels.iter().position(|level| *level == p) { return Some(sample(i)); }
    let i = levels.windows(2).position(|pair| p >= pair[0].min(pair[1]) && p <= pair[0].max(pair[1]))?;
    let x = (p.ln() - levels[i].ln()) / (levels[i + 1].ln() - levels[i].ln());
    Some(sample(i) * (1.0 - x) + sample(i + 1) * x)
}
fn interpolate_clamped(knots: &[(f64, f64)], p: f64) -> f64 {
    for pair in knots.windows(2) {
        if p >= pair[0].0.min(pair[1].0) && p <= pair[0].0.max(pair[1].0) {
            let x = (p.ln() - pair[0].0.ln()) / (pair[1].0.ln() - pair[0].0.ln());
            return pair[0].1 * (1.0 - x) + pair[1].1 * x;
        }
    }
    knots.iter().min_by(|a,b| (a.0-p).abs().total_cmp(&(b.0-p).abs())).unwrap().1
}
fn specific_humidity(t: f64, p: f64, rh: f64) -> f64 {
    let e = rh.max(0.0) * 611.2 * (17.67 * (t-273.15)/(t-29.65)).exp();
    let w = 0.622 * e / (p-e).max(1.0);
    w / (1.0+w)
}
fn relative_humidity(t: f64, p: f64, q: f64) -> f64 {
    let w = q / (1.0-q);
    let e = p*w/(0.622+w);
    e/(611.2*(17.67*(t-273.15)/(t-29.65)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pressure_samples_keep_their_values_when_sorted_and_coincident_knots_average() {
        let mut knots = vec![(66896.640625, 2.0), (66906.796875, 4.0),
            (66000.0, 6.0), (66000.0, 10.0)];
        assert_eq!(order_pressure_knots(&mut knots), 1);
        assert_eq!(knots, vec![(66906.796875, 4.0), (66896.640625, 2.0), (66000.0, 8.0)]);
        assert_eq!(interpolate_clamped(&knots, 66896.640625), 2.0);
        assert_eq!(interpolate_clamped(&knots, 66000.0), 8.0);
    }
    #[test]
    fn model_indices_fill_missing_pressure_then_splice_each_columns_pressure() {
        let node = Node::parse(br#"{
            "method":"cosine_pressure_distance_log_anomaly", "pressure_band_hpa":[0,100],
            "primary_pressure_scale":1,"donor_pressure_scale":1,"max_relative_humidity":1.05,
            "pressure_field":"air_pressure", "fields":{
                "air_temperature":{"primary_field":"air_temperature","representation":"canonical"},
                "specific_humidity":{"primary_field":"specific_humidity","representation":"canonical"},
                "air_pressure":{"primary_field":"air_pressure","representation":"canonical"}}}"#).unwrap();
        let merge = VerticalMerge::parse(&node).unwrap();
        let coordinate = Node::parse(br#"{"kind":"model_level","units":"1"}"#).unwrap();
        merge.validate_coordinates(&coordinate, &coordinate).unwrap();
        let time = chrono::NaiveDate::from_ymd_opt(2024,5,21).unwrap().and_hms_opt(18,0,0).unwrap();
        let key = (time, None);
        let value = |name: &str, levels: usize, words: Vec<f64>| DirectValue {
            name:name.into(),valid_time:time,member:None,source_cycle:time,
            axes:vec!["vertical".into(),"y".into(),"x".into()],
            missing_count:words.iter().filter(|x| !x.is_finite()).count(), references:Vec::new(),
            values:ArrayD::from_shape_vec(IxDyn(&[levels,1,2]), words).unwrap(),
        };
        let primary = [value("air_pressure",2,vec![100000.,95000.,f64::NAN,f64::NAN]),
            value("air_temperature",2,vec![300.,305.,270.,275.]),
            value("specific_humidity",2,vec![0.001;4])];
        let donor = [value("air_pressure",4,vec![100000.,95000.,80000.,70000.,50000.,45000.,20000.,18000.]),
            value("air_temperature",4,vec![290.,295.,280.,285.,260.,265.,220.,225.]),
            value("specific_humidity",4,vec![0.001;8])];
        let mut slice = DecodedCollection {latitude:vec![0.],longitude:vec![0.,0.03],
            vertical_values:vec![1.,3.],direct:primary.into_iter().map(|v| ((time,None,v.name.clone()),v)).collect(),
            source_cycles:BTreeMap::from([(key.clone(),time)]),grid_fingerprint:"test".into(),hybrid_a:Vec::new(),hybrid_b:Vec::new(),original_pressure_hpa:None};
        let receipt = merge.apply(&mut slice, &donor, &[1.,2.,3.,4.], &key, "test").unwrap();
        let t = &slice.direct[&(time,None,"air_temperature".into())].values;
        assert_eq!(t.as_slice().unwrap(), &[300.,305.,280.,285.,270.,275.,220.,225.]);
        assert_eq!(slice.direct[&(time,None,"air_pressure".into())].values, donor[0].values);
        assert_eq!(receipt["primary_pressure_missing_values_filled"],2);
        assert_eq!(receipt["pressure_coordinate"],"air_pressure_field");
        assert_eq!(slice.vertical_values,vec![1.,2.,3.,4.]);
    }
}
