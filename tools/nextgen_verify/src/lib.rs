// SPDX-License-Identifier: Apache-2.0
// Original NextGen observation verification; no hydrologic model source used.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub type Timeline = BTreeMap<i64, f64>;
pub type Hydrographs = BTreeMap<String, Timeline>;
pub mod geo;
pub mod matching;
pub mod netcdf;
pub mod world;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub station_id: String,
    pub time: String,
    pub flow_m3s: f64,
    #[serde(default)]
    pub quality: String,
}
/// A gauge row. `station_id` is the bare agency number for USGS (the v2.0.1
/// `Gages-<station>` form) and `AGENCY:<number>` for every other agency, so
/// one table can hold gauges from several countries without collisions.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Station {
    pub station_id: String,
    #[serde(default)]
    pub feature_id: Option<String>,
    #[serde(default)]
    pub time_series_id: Option<String>,
    /// Source row that serves this station; omitted means the agency-less (USGS) rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Explicit offset for sources that label values with a local date only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utc_offset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latitude: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longitude: Option<f64>,
    /// A drainage area given in the row must cite where it comes from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drainage_area_km2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drainage_area_source: Option<String>,
}
impl Station {
    /// The agency's own number (the part after `AGENCY:`), used in source URLs.
    pub fn number(&self) -> &str {
        self.station_id.split_once(':').map_or(&self.station_id, |(_, n)| n)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Layout {
    #[serde(default)]
    pub records: String,
    #[serde(default)]
    pub station: Option<String>,
    #[serde(default)]
    pub station_prefix: String,
    #[serde(default)]
    pub feature: Option<String>,
    #[serde(default)]
    pub time: String,
    #[serde(default)]
    pub flow: String,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub fixed_unit: Option<String>,
    pub unit_factors: BTreeMap<String, f64>,
    #[serde(default)]
    pub filters: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub optional_filters: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub time_series: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub netcdf: Option<netcdf::Layout>,
    /// Records carry no station field; they belong to the station requested.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub station_from_request: bool,
    /// Quality pointer (record, or document with a leading `^`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// Quality text the source documents for every value (for example raw, unchecked data).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_quality: Option<String>,
    /// `time` holds a local date only; the station row's `utc_offset` stamps the day start.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub time_from_date: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Source {
    pub id: String,
    #[serde(default)]
    pub url: String,
    #[serde(default = "world::empty_layout")]
    pub layout: Layout,
    #[serde(default)]
    pub links: String,
    #[serde(default)]
    pub relation: String,
    #[serde(default)]
    pub href: String,
    /// Agency code for `AGENCY:<number>` station ids and `Gages-AGENCY:<number>`; none for USGS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agency: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub licence: Option<world::Licence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<world::Request>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<world::Container>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<world::Metadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_gate: Option<world::OwnerGate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discover: Option<world::Discover>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Table {
    pub schema: String,
    #[serde(default)]
    pub sources: Vec<Source>,
    pub stations: Vec<Station>,
    #[serde(default)]
    pub layouts: BTreeMap<String, Layout>,
    /// Another table (path relative to this file) whose source rows and
    /// layouts this table uses, so a station list needs no copy of the sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources_from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matching: Option<matching::Rules>,
}
pub fn timestamp(s: &str) -> Result<i64> {
    DateTime::parse_from_rfc3339(s)
        .map(|time| time.timestamp_millis())
        .map_err(|error| {
            format!(
                "invalid timestamp {s:?}: expected RFC3339 with an explicit UTC offset (Z or +/-HH:MM); {error}"
            )
            .into()
        })
}
pub fn utc(t: i64) -> String {
    DateTime::<Utc>::from_timestamp_millis(t)
        .unwrap()
        .to_rfc3339()
}
pub fn scalar(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}
// JSON pointer plus wildcard segments: table work adds a source or row layout.
pub fn select<'a>(root: &'a Value, pointer: &str) -> Vec<&'a Value> {
    if pointer.is_empty() {
        return vec![root];
    }
    if !pointer.starts_with('/') {
        return vec![];
    }
    let mut layer = vec![root];
    for key in pointer[1..].split('/') {
        let key = key.replace("~1", "/").replace("~0", "~");
        layer = layer
            .into_iter()
            .flat_map(|v| {
                if key == "*" {
                    match v {
                        Value::Array(a) => a.iter().collect(),
                        Value::Object(o) => o.values().collect(),
                        _ => vec![],
                    }
                } else if let Some((k, want)) = key.strip_prefix('[').and_then(|x| x.strip_suffix(']')).and_then(|x| x.split_once('=')) {
                    // `[key=value]`: the members whose `key` equals `value`.
                    let members: Vec<&Value> = match v {
                        Value::Array(a) => a.iter().collect(),
                        Value::Object(o) => o.values().collect(),
                        _ => vec![],
                    };
                    members.into_iter().filter(|m| m.get(k).and_then(scalar).as_deref() == Some(want)).collect()
                } else {
                    v.pointer(&format!("/{}", key.replace('~', "~0").replace('/', "~1")))
                        .into_iter()
                        .collect()
                }
            })
            .collect();
    }
    layer
}
pub fn field(root: &Value, path: &str) -> Option<String> {
    if path.contains('[') {
        return select(root, path).into_iter().find_map(scalar);
    }
    root.pointer(path).and_then(scalar)
}
/// A layout pointer: relative to the record, or to the whole document when it starts with `^`.
pub fn lookup(document: &Value, record: &Value, path: &str) -> Option<String> {
    match path.strip_prefix('^') {
        Some(p) => field(document, p),
        None => field(record, path),
    }
}
pub fn normalize(
    root: &Value,
    spec: &Layout,
    stations: &[Station],
) -> Result<(Vec<Observation>, usize)> {
    let mut values = Vec::new();
    let mut rejected = 0;
    for record in select(root, &spec.records) {
        if spec
            .filters
            .iter()
            .any(|(p, allowed)| field(record, p).is_none_or(|v| !allowed.contains(&v)))
            || spec.optional_filters.iter().any(|(p, allowed)| {
                field(record, p).is_some_and(|v| !v.is_empty() && !allowed.contains(&v))
            })
        {
            rejected += 1;
            continue;
        }
        let station = stations.iter().find(|s| {
            (spec.station_from_request && stations.len() == 1)
                || spec
                    .station
                    .as_ref()
                    .and_then(|p| field(record, p))
                    .is_some_and(|id| {
                        let id = id.strip_prefix(&spec.station_prefix).unwrap_or(&id);
                        id == s.station_id || id == s.number()
                    })
                || spec
                    .feature
                    .as_ref()
                    .and_then(|p| field(record, p))
                    .is_some_and(|id| s.feature_id.as_ref() == Some(&id))
        });
        let Some(station) = station else {
            rejected += 1;
            continue;
        };
        if let Some(id) = &station.time_series_id {
            if spec
                .time_series
                .as_ref()
                .and_then(|p| field(record, p))
                .as_ref()
                != Some(id)
            {
                rejected += 1;
                continue;
            }
        }
        let Some(q) = field(record, &spec.flow)
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|q| q.is_finite() && *q >= 0.)
        else {
            rejected += 1;
            continue;
        };
        let unit = spec
            .unit
            .as_ref()
            .and_then(|p| lookup(root, record, p))
            .or_else(|| spec.fixed_unit.clone())
            .ok_or("missing declared discharge units")?;
        let factor = *spec
            .unit_factors
            .get(&unit)
            .ok_or_else(|| format!("unsupported discharge unit {unit}"))?;
        if !factor.is_finite() || factor <= 0. {
            return Err("unit factor must be finite and positive".into());
        }
        let flow_m3s = q * factor;
        if !flow_m3s.is_finite() {
            return Err("nonfinite converted discharge".into());
        }
        let mut t = field(record, &spec.time).ok_or("missing UTC time")?;
        if spec.time_from_date {
            let offset = station.utc_offset.as_ref().ok_or_else(|| {
                format!(
                    "station {} has date-only values; its table row must declare utc_offset",
                    station.station_id
                )
            })?;
            if t.len() != 10 {
                return Err(format!("expected a YYYY-MM-DD date, found {t:?}").into());
            }
            t = format!("{t}T00:00:00{offset}");
        }
        let mut quality: Vec<String> = spec.fixed_quality.iter().cloned().collect();
        quality.extend(spec.quality.as_ref().and_then(|p| lookup(root, record, p)));
        quality.extend(
            spec.filters
                .keys()
                .chain(spec.optional_filters.keys())
                .filter_map(|p| field(record, p)),
        );
        values.push(Observation {
            station_id: station.station_id.clone(),
            time: utc(timestamp(&t)?),
            flow_m3s,
            quality: quality.join(";"),
        });
    }
    Ok((values, rejected))
}
pub fn index(rows: &[Observation]) -> Result<Hydrographs> {
    let mut out: Hydrographs = BTreeMap::new();
    for row in rows {
        if row.station_id.is_empty() || !row.flow_m3s.is_finite() || row.flow_m3s < 0. {
            return Err("invalid station/discharge".into());
        }
        let t = timestamp(&row.time)?;
        let series = out.entry(row.station_id.clone()).or_default();
        if let Some(before) = series.insert(t, row.flow_m3s) {
            if before.to_bits() != row.flow_m3s.to_bits() {
                return Err(format!(
                    "conflicting discharge at {} {}; select a time_series_id in table",
                    row.station_id, row.time
                )
                .into());
            }
        }
    }
    Ok(out)
}
pub fn ordered(rows: &[Observation]) -> Result<Vec<Observation>> {
    index(rows)?;
    let mut unique = BTreeMap::new();
    for row in rows {
        let t = timestamp(&row.time)?;
        let mut normalized = row.clone();
        normalized.time = utc(t);
        unique
            .entry((row.station_id.clone(), t))
            .or_insert(normalized);
    }
    Ok(unique.into_values().collect())
}
#[derive(Debug, Serialize)]
pub struct Metrics {
    pub samples: usize,
    pub nse: Option<f64>,
    pub kge_2009: Option<f64>,
    pub rmse_m3s: Option<f64>,
    pub mean_ratio: Option<f64>,
    pub correlation: Option<f64>,
    pub peak_timing_seconds: Option<f64>,
    pub differing_f64_words: usize,
    pub reasons: Vec<String>,
}
pub fn metrics(pairs: &[(i64, f64, f64)]) -> Metrics {
    let n = pairs.len();
    let mut out = Metrics {
        samples: n,
        nse: None,
        kge_2009: None,
        rmse_m3s: None,
        mean_ratio: None,
        correlation: None,
        peak_timing_seconds: None,
        differing_f64_words: pairs
            .iter()
            .filter(|(_, a, b)| a.to_bits() != b.to_bits())
            .count(),
        reasons: vec![],
    };
    if n == 0 {
        out.reasons.push("no exact UTC pairs".into());
        return out;
    }
    let mean_a = pairs.iter().map(|p| p.1).sum::<f64>() / n as f64;
    let mean_b = pairs.iter().map(|p| p.2).sum::<f64>() / n as f64;
    let (mut va, mut vb, mut cov, mut se) = (0., 0., 0., 0.);
    for &(_, a, b) in pairs {
        va += (a - mean_a) * (a - mean_a);
        vb += (b - mean_b) * (b - mean_b);
        cov += (a - mean_a) * (b - mean_b);
        se += (a - b) * (a - b);
    }
    out.rmse_m3s = Some((se / n as f64).sqrt());
    if mean_b != 0. {
        out.mean_ratio = Some(mean_a / mean_b);
    }
    if n < 2 {
        out.reasons
            .push("fewer than two exact pairs; NSE/KGE/peak timing undefined".into());
        return out;
    }
    let peak = |column: usize| {
        let mut best = (
            pairs[0].0,
            if column == 1 { pairs[0].1 } else { pairs[0].2 },
        );
        for &(t, a, b) in &pairs[1..] {
            let q = if column == 1 { a } else { b };
            if q > best.1 {
                best = (t, q);
            }
        }
        best.0
    };
    if va > 0. && vb > 0. {
        out.peak_timing_seconds = Some((peak(1) - peak(2)) as f64 / 1000.);
    } else {
        out.reasons
            .push("constant series: peak timing undefined".into());
    }
    if vb > 0. {
        out.nse = Some(1. - se / vb);
    } else {
        out.reasons.push("constant reference: NSE undefined".into());
    }
    if va > 0. && vb > 0. && mean_b != 0. {
        let r = cov / (va * vb).sqrt();
        let alpha = (va / vb).sqrt();
        let beta = mean_a / mean_b;
        out.correlation = Some(r);
        out.kge_2009 =
            Some(1. - ((r - 1.).powi(2) + (alpha - 1.).powi(2) + (beta - 1.).powi(2)).sqrt());
    } else {
        out.reasons
            .push("constant series or zero reference mean: KGE undefined".into());
    }
    out
}
#[derive(Serialize)]
pub struct Assessment {
    pub station_id: String,
    pub model_samples: usize,
    pub reference_samples: usize,
    pub observation_samples: usize,
    pub model_vs_reference: Metrics,
    pub model_vs_observations: Metrics,
    pub reference_vs_observations: Metrics,
    pub complete_reference_support: bool,
}
pub fn assess(
    station: &str,
    model: &Hydrographs,
    native: &Hydrographs,
    obs: &Hydrographs,
) -> Assessment {
    let empty = Timeline::new();
    let m = model.get(station).unwrap_or(&empty);
    let r = native.get(station).unwrap_or(&empty);
    let o = obs.get(station).unwrap_or(&empty);
    let mr: Vec<_> = m
        .iter()
        .filter_map(|(&t, &q)| r.get(&t).map(|&v| (t, q, v)))
        .collect();
    let shared: BTreeSet<_> = m
        .keys()
        .filter(|t| r.contains_key(t) && o.contains_key(t))
        .copied()
        .collect();
    let mo: Vec<_> = shared.iter().map(|&t| (t, m[&t], o[&t])).collect();
    let ro: Vec<_> = shared.iter().map(|&t| (t, r[&t], o[&t])).collect();
    Assessment {
        station_id: station.into(),
        model_samples: m.len(),
        reference_samples: r.len(),
        observation_samples: o.len(),
        complete_reference_support: !m.is_empty() && m.keys().eq(r.keys()),
        model_vs_reference: metrics(&mr),
        model_vs_observations: metrics(&mo),
        reference_vs_observations: metrics(&ro),
    }
}
