//! A standing scorecard over native verification receipts and recorded pairs.
//! Cases are the bootstrap units. Stations, leads and pixels stay together.
//! Positive improvement means the candidate has the lower RMSE or higher FSS.

use crate::verification_io::{read_json, source_identity};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "gpuwm.obs-scorecard.manifest.v1";
pub const RECEIPT: &str = "gpuwm.obs-scorecard.receipt.v1";

fn default_replicates() -> usize {
    5000
}
fn default_confidence() -> f64 {
    0.95
}
fn default_minimum() -> usize {
    4
}
fn default_metric() -> String {
    "rmse".into()
}
fn default_direction() -> String {
    "lower".into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Bootstrap {
    #[serde(default = "default_replicates")]
    pub replicates: usize,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_minimum")]
    pub minimum_blocks: usize,
}
impl Default for Bootstrap {
    fn default() -> Self {
        Self {
            replicates: 5000,
            confidence: 0.95,
            seed: 0,
            minimum_blocks: 4,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    pub id: String,
    pub source: String,
    pub quantity: String,
    pub units: String,
    #[serde(default = "default_metric")]
    pub statistic: String,
    #[serde(default = "default_direction")]
    pub direction: String,
    #[serde(default)]
    pub threshold: Option<f64>,
    #[serde(default)]
    pub width_km: Option<f64>,
    #[serde(default)]
    pub required: bool,
}

/// Column names and filters are data. No forecast label chooses a reader.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairCsv {
    pub source: String,
    pub path: PathBuf,
    pub station_column: String,
    pub time_column: String,
    pub quantity_column: String,
    pub units_column: String,
    pub observed_column: String,
    pub candidate_column: String,
    pub reference_column: String,
    #[serde(default)]
    pub quantity_map: BTreeMap<String, String>,
    #[serde(default)]
    pub filters: BTreeMap<String, String>,
    /// A digest binds this replay to the retained export, not to its raw inputs.
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    /// Related cycles or domains share a block. They never inflate independence.
    pub block: String,
    #[serde(default)]
    pub requests: Vec<PathBuf>,
    #[serde(default)]
    pub receipts: Vec<PathBuf>,
    #[serde(default)]
    pub pairs: Option<PairCsv>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub candidate: String,
    pub reference: String,
    pub cases: Vec<Case>,
    pub metrics: Vec<Metric>,
    #[serde(default)]
    pub bootstrap: Bootstrap,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub acquisition: Vec<Value>,
}

impl Manifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!("expected {SCHEMA}"));
        }
        if self.candidate.trim().is_empty()
            || self.reference.trim().is_empty()
            || self.candidate == self.reference
        {
            return Err("distinct candidate and reference labels are required to prevent comparing an arm to itself".into());
        }
        let b = &self.bootstrap;
        if b.replicates < 100
            || b.replicates > 1_000_000
            || !b.confidence.is_finite()
            || !(0.0..1.0).contains(&b.confidence)
            || b.confidence == 0.0
            || b.minimum_blocks < 2
        {
            return Err("bootstrap needs 100..1000000 replicates, confidence in (0,1), and at least two independent blocks".into());
        }
        let mut ids = BTreeSet::new();
        for c in &self.cases {
            if !component(&c.id) || c.block.trim().is_empty() || !ids.insert(&c.id) {
                return Err("case ids must be unique safe path components and blocks must be nonempty; duplicate cases would inflate the score".into());
            }
            if c.requests.is_empty() && c.receipts.is_empty() && c.pairs.is_none() {
                return Err(format!(
                    "case {} has no recorded pairs, requests or receipts",
                    c.id
                ));
            }
            if c.pairs.is_some() && (!c.requests.is_empty() || !c.receipts.is_empty()) {
                return Err(format!(
                    "case {} mixes pair and native modes, which would double count samples",
                    c.id
                ));
            }
        }
        if ids.is_empty() || self.metrics.is_empty() {
            return Err("scorecard needs cases and metrics".into());
        }
        ids.clear();
        for m in &self.metrics {
            let expected_direction = match m.statistic.as_str() {
                "fss" => "higher",
                "bias" => "diagnostic",
                _ => "lower",
            };
            if m.direction != expected_direction {
                return Err(format!(
                    "metric {} direction must be {expected_direction}; reversing {} would label a worse forecast as an improvement",
                    m.id, m.statistic
                ));
            }
            if !component(&m.id)
                || !ids.insert(&m.id)
                || m.source.is_empty()
                || m.units.is_empty()
                || m.quantity.is_empty()
            {
                return Err("metric ids must be unique safe components, with source, quantity and units metadata".into());
            }
            if !["rmse", "bias", "mae", "fss"].contains(&m.statistic.as_str())
                || !["lower", "higher", "diagnostic"].contains(&m.direction.as_str())
                || (m.statistic == "bias" && m.direction != "diagnostic")
                || (m.statistic == "fss"
                    && (m.threshold.is_none_or(|v| !v.is_finite())
                        || m.width_km.is_none_or(|v| !v.is_finite() || v <= 0.0)))
            {
                return Err(format!(
                    "metric {} needs a supported statistic and direction; bias is diagnostic, FSS needs threshold and width",
                    m.id
                ));
            }
        }
        Ok(())
    }
}
fn component(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
}
pub fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.into()
    } else {
        root.join(path)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseScore {
    pub case: String,
    pub block: String,
    pub metric: String,
    pub candidate: Option<f64>,
    pub reference: Option<f64>,
    pub count: usize,
    pub status: String,
    pub reason: Option<String>,
}

fn number(value: &Value, key: &str) -> Option<f64> {
    value[key].as_f64().filter(|v| v.is_finite())
}
fn mean(values: &[f64]) -> f64 {
    rw_field_sum(values) / values.len() as f64
}
fn rw_field_sum(values: &[f64]) -> f64 {
    // Stable input order, independent of worker completion. No forecast arithmetic changes.
    values.iter().copied().sum()
}
fn statistic(errors: &[f64], name: &str) -> f64 {
    let mut stats=rw_scoreboard::Stats::default();
    for &error in errors { stats.add_error(error); }
    stats.metric(name).unwrap_or(f64::NAN)
}
fn row(c: &Case, m: &Metric, a: &[f64], b: &[f64], count: usize) -> CaseScore {
    let valid = !a.is_empty() && a.len() == b.len();
    let candidate = valid.then(|| {
        if m.statistic == "fss" {
            mean(a)
        } else {
            statistic(a, &m.statistic)
        }
    });
    let reference = valid.then(|| {
        if m.statistic == "fss" {
            mean(b)
        } else {
            statistic(b, &m.statistic)
        }
    });
    CaseScore {
        case: c.id.clone(),
        block: c.block.clone(),
        metric: m.id.clone(),
        candidate,
        reference,
        count,
        status: if valid { "measured" } else { "missing" }.into(),
        reason: (!valid).then(|| {
            format!(
                "no common finite {} {} support in both arms",
                m.source, m.quantity
            )
        }),
    }
}

pub fn native_scores(
    c: &Case,
    manifest: &Manifest,
    paths: &[PathBuf],
) -> Result<(Vec<CaseScore>, Vec<Value>), String> {
    let mut samples: BTreeMap<(String, String, String), (f64, f64, f64)> = BTreeMap::new();
    let mut radar: BTreeMap<String, Vec<(f64, f64, usize)>> = BTreeMap::new();
    let mut grid_errors: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    let mut missing_reasons: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut instants = BTreeSet::new();
    let mut identities = Vec::new();
    for path in paths {
        let receipt = read_json(path)?;
        if receipt["schema"] != crate::verification::RECEIPT_SCHEMA {
            return Err(format!(
                "{} is not a native verification receipt",
                path.display()
            ));
        }
        let time = receipt["valid_time"]
            .as_str()
            .ok_or("native receipt valid_time missing")?;
        let domain = receipt["domain"]
            .as_str()
            .ok_or("native receipt domain missing")?;
        let native: crate::verification::VerificationRequest = receipt
            .get("scoring_request")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| e.to_string())?
            .unwrap_or_else(|| crate::verification::default_station_request(time, domain));
        if !instants.insert((
            domain.to_string(),
            time.to_string(),
            native.station_source.clone(),
        )) {
            return Err(
                "duplicate native source/domain/time receipt would double count a lead".into(),
            );
        }
        identities.push(source_identity(path)?);
        for sample in receipt["station_samples"]
            .as_array()
            .ok_or("native receipt lacks station_samples")?
        {
            let id = sample["station_id"]
                .as_str()
                .ok_or("sample station_id missing")?;
            let forecasts = sample["forecast"]
                .as_array()
                .ok_or("sample forecast missing")?;
            let arm = |label: &str| {
                forecasts
                    .iter()
                    .find(|f| f["label"].as_str() == Some(label))
            };
            for metric in manifest.metrics.iter().filter(|m| m.statistic != "fss") {
                if metric.source != native.station_source {
                    continue;
                }
                // Only sample ids with a passed native paired QC row may enter.
                let passed = receipt["stations"].as_array().is_some_and(|rows| {
                    rows.iter().any(|r| {
                        r["quantity"].as_str() == Some(&metric.quantity)
                            && r["units"].as_str() == Some(&metric.units)
                            && r["status"].as_str() == Some("ready")
                    })
                });
                if !passed {
                    continue;
                }
                let Some(obs) = number(&sample["values"], &metric.quantity) else {
                    continue;
                };
                let Some(a) =
                    arm(&manifest.candidate).and_then(|f| number(&f["values"], &metric.quantity))
                else {
                    continue;
                };
                let Some(b) =
                    arm(&manifest.reference).and_then(|f| number(&f["values"], &metric.quantity))
                else {
                    continue;
                };
                // Apply the same QC bounds as the native paired row, not merely its overall status.
                if !crate::verification::observation_valid_at(
                    &native,
                    &metric.quantity,
                    &sample["values"],
                    sample["observation_time"]
                        .as_str()
                        .ok_or("sample actual observation time missing")?,
                )? {
                    continue;
                }
                let key = (metric.id.clone(), time.to_string(), id.to_string());
                if samples.insert(key, (obs, a, b)).is_some() {
                    return Err("duplicate station/lead/metric would double count a sample".into());
                }
            }
        }
        for metric in manifest
            .metrics
            .iter()
            .filter(|m| m.source == native.radar_source && m.statistic == "fss")
        {
            for r in receipt["radar"]
                .as_array()
                .ok_or("native receipt radar missing")?
            {
                if r["quantity"].as_str() != Some(&metric.quantity)
                    || r["units"].as_str() != Some(&metric.units)
                    || number(r, "threshold") != metric.threshold
                    || number(r, "width_km") != metric.width_km
                {
                    continue;
                }
                let arms = r["arms"].as_array().ok_or("radar arms missing")?;
                let arm = |label: &str| {
                    arms.iter()
                        .find(|v| v["label"].as_str() == Some(label))
                        .and_then(|v| number(v, "fss"))
                };
                if let (Some(a), Some(b)) = (arm(&manifest.candidate), arm(&manifest.reference)) {
                    let support = |label: &str| {
                        arms.iter()
                            .find(|v| v["label"].as_str() == Some(label))
                            .and_then(|v| v["count"].as_u64())
                    };
                    let count = support(&manifest.candidate)
                        .ok_or("FSS scored-neighborhood count missing")?;
                    if Some(count) != support(&manifest.reference) {
                        return Err("FSS arms have different paired neighborhood support".into());
                    }
                    radar
                        .entry(metric.id.clone())
                        .or_default()
                        .push((a, b, count as usize));
                } else {
                    missing_reasons
                        .entry(metric.id.clone())
                        .or_default()
                        .push(format!(
                            "{} {}: {}",
                            metric.source,
                            time,
                            r["status"].as_str().unwrap_or("missing-native-score")
                        ));
                }
            }
        }
        for metric in manifest
            .metrics
            .iter()
            .filter(|m| m.source == native.radar_source && m.statistic != "fss")
        {
            for row in receipt["grid_errors"].as_array().into_iter().flatten() {
                if row["quantity"].as_str() != Some(&metric.quantity)
                    || row["units"].as_str() != Some(&metric.units)
                {
                    continue;
                }
                let key = match metric.statistic.as_str() {
                    "rmse" => "sum_squared_error",
                    "mae" => "sum_absolute_error",
                    _ => "sum_error",
                };
                let arms = row["arms"].as_array().ok_or("grid score arms missing")?;
                let a = arms
                    .iter()
                    .find(|v| v["label"].as_str() == Some(&manifest.candidate));
                let b = arms
                    .iter()
                    .find(|v| v["label"].as_str() == Some(&manifest.reference));
                if let (Some(a), Some(b)) = (a, b) {
                    let count = a["count"].as_u64().ok_or("grid error count missing")? as usize;
                    if a["count"] != b["count"] {
                        return Err("grid error arms have different paired support".into());
                    }
                    if let (Some(av), Some(bv)) = (number(a, key), number(b, key)) {
                        let entry = grid_errors
                            .entry(metric.id.clone())
                            .or_insert((0, 0.0, 0.0));
                        entry.0 += count;
                        entry.1 += av;
                        entry.2 += bv;
                    }
                }
            }
        }
    }
    let scores = manifest
        .metrics
        .iter()
        .map(|m| {
            if m.statistic == "fss" {
                let values = radar.get(&m.id).cloned().unwrap_or_default();
                let mut score = row(
                    c,
                    m,
                    &values.iter().map(|v| v.0).collect::<Vec<_>>(),
                    &values.iter().map(|v| v.1).collect::<Vec<_>>(),
                    values.iter().map(|v| v.2).sum(),
                );
                if score.status == "missing" {
                    if let Some(reasons) = missing_reasons.get(&m.id) {
                        score.reason = Some(reasons.join("; "));
                    }
                }
                score
            } else if grid_errors.contains_key(&m.id) {
                let (count, a, b) = grid_errors.get(&m.id).copied().unwrap_or((0, 0.0, 0.0));
                let convert = |v: f64| {
                    if m.statistic == "rmse" {
                        (v / count as f64).sqrt()
                    } else {
                        v / count as f64
                    }
                };
                let mut score = row(c, m, &[], &[], count);
                if count > 0 {
                    score.candidate = Some(convert(a));
                    score.reference = Some(convert(b));
                    score.status = "measured".into();
                    score.reason = None;
                }
                score
            } else {
                let values = samples
                    .iter()
                    .filter(|((id, _, _), _)| id == &m.id)
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>();
                row(
                    c,
                    m,
                    &values.iter().map(|v| v.1 - v.0).collect::<Vec<_>>(),
                    &values.iter().map(|v| v.2 - v.0).collect::<Vec<_>>(),
                    values.len(),
                )
            }
        })
        .collect();
    Ok((scores, identities))
}

pub fn csv_scores(
    c: &Case,
    manifest: &Manifest,
    root: &Path,
) -> Result<(Vec<CaseScore>, Value), String> {
    let spec = c.pairs.as_ref().ok_or("pair CSV absent")?;
    let path = resolve(root, &spec.path);
    let identity = source_identity(&path)?;
    if identity["sha256"].as_str() != Some(&spec.sha256) {
        return Err(format!(
            "{} digest differs from manifest; recorded sample support changed",
            path.display()
        ));
    }
    let records = rw_obs::csv::read(&path)?;
    let header = records.first().ok_or("pair CSV has no header")?;
    let col = |s: &str| {
        header
            .iter()
            .position(|h| h == s)
            .ok_or_else(|| format!("{} missing column {s}", path.display()))
    };
    let id = col(&spec.station_column)?;
    let time = col(&spec.time_column)?;
    let q = col(&spec.quantity_column)?;
    let units = col(&spec.units_column)?;
    let obs = col(&spec.observed_column)?;
    let a = col(&spec.candidate_column)?;
    let b = col(&spec.reference_column)?;
    let filters = spec
        .filters
        .iter()
        .map(|(k, v)| Ok((col(k)?, v)))
        .collect::<Result<Vec<_>, String>>()?;
    let mut values: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for record in records.iter().skip(1) {
        if record.len() != header.len() {
            return Err(
                "pair CSV row length differs from header; sample columns would shift".into(),
            );
        }
        if filters.iter().any(|(i, v)| &record[*i] != v.as_str()) {
            continue;
        }
        let quantity = spec
            .quantity_map
            .get(&record[q])
            .map(String::as_str)
            .unwrap_or(&record[q]);
        let applicable = manifest
            .metrics
            .iter()
            .filter(|m| m.source == spec.source && m.quantity == quantity && m.statistic != "fss")
            .collect::<Vec<_>>();
        if applicable.is_empty() {
            continue;
        }
        let parse = |i: usize| record[i].parse::<f64>().ok().filter(|v| v.is_finite());
        let (Some(o), Some(av), Some(bv)) = (parse(obs), parse(a), parse(b)) else {
            continue;
        };
        crate::verification::parse_time(&record[time])?;
        if record[id].is_empty() {
            return Err("pair CSV has empty station/sample id".into());
        }
        if !seen.insert((
            record[id].to_string(),
            record[time].to_string(),
            quantity.to_string(),
        )) {
            return Err(format!(
                "duplicate pair {}/{}/{} would inflate sample support",
                &record[id], &record[time], quantity
            ));
        }
        for m in applicable {
            if record[units] != m.units {
                return Err(format!(
                    "{} units {} differ from {}",
                    m.id, &record[units], m.units
                ));
            }
            values
                .entry(m.id.clone())
                .or_default()
                .push((av - o, bv - o));
        }
    }
    let scores: Vec<CaseScore> = manifest
        .metrics
        .iter()
        .map(|m| {
            let values = values.get(&m.id).cloned().unwrap_or_default();
            row(
                c,
                m,
                &values.iter().map(|v| v.0).collect::<Vec<_>>(),
                &values.iter().map(|v| v.1).collect::<Vec<_>>(),
                values.len(),
            )
        })
        .collect();
    if scores.iter().any(|r| {
        r.candidate.is_some_and(|v| !v.is_finite()) || r.reference.is_some_and(|v| !v.is_finite())
    }) {
        return Err(
            "recorded CSV residual reduction overflows float64; it cannot produce a measured score"
                .into(),
        );
    }
    Ok((scores, identity))
}

/// SplitMix64 with rejection sampling. Published RNG and quantile semantics
/// make the same seed independent of external RNG package versions.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn draw(state: &mut u64, n: usize) -> usize {
    let n = n as u64;
    let threshold = n.wrapping_neg() % n;
    loop {
        let v = next(state);
        if v >= threshold {
            return (v % n) as usize;
        }
    }
}
fn quantile(values: &[f64], probability: f64) -> f64 {
    let position = (values.len() - 1) as f64 * probability;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    values[low] + (values[high] - values[low]) * (position - low as f64)
}
pub fn interval(values: &[f64], config: &Bootstrap) -> Value {
    if values.len() < 2 {
        return json!({"status":"not_measured","estimate":(!values.is_empty()).then(||mean(values)),"blocks":values.len(),"reason":"at least two independent blocks are needed for a case bootstrap"});
    }
    let mut state = config.seed;
    let mut samples = (0..config.replicates)
        .map(|_| {
            mean(
                &(0..values.len())
                    .map(|_| values[draw(&mut state, values.len())])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    let alpha = (1.0 - config.confidence) / 2.0;
    let lower = quantile(&samples, alpha);
    let upper = quantile(&samples, 1.0 - alpha);
    let qualified = values.len() >= config.minimum_blocks;
    json!({"status":if qualified {"measured"} else {"exploratory"},"estimate":mean(values),"lower":lower,"upper":upper,"blocks":values.len(),"confidence":config.confidence,"replicates":config.replicates,"seed":config.seed,"verdict":if !qualified {"too_few_blocks"} else if lower>0.0 {"favors_candidate"} else if upper<0.0 {"favors_reference"} else {"indistinguishable"}})
}

pub fn aggregate(
    manifest: &Manifest,
    scores: &[CaseScore],
    identities: Vec<Value>,
    manifest_identity: Value,
) -> Value {
    let mut comparisons = Vec::new();
    for metric in &manifest.metrics {
        let rows = scores
            .iter()
            .filter(|r| r.metric == metric.id)
            .collect::<Vec<_>>();
        let mut blocks: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        for r in &rows {
            if let (Some(a), Some(b)) = (r.candidate, r.reference) {
                blocks.entry(r.block.clone()).or_default().push((a, b));
            }
        }
        // A partially missing block cannot change its cycle weighting between metrics silently.
        blocks.retain(|block, values| {
            values.len() == manifest.cases.iter().filter(|c| &c.block == block).count()
        });
        let a = blocks
            .values()
            .map(|v| mean(&v.iter().map(|v| v.0).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let b = blocks
            .values()
            .map(|v| mean(&v.iter().map(|v| v.1).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        let differences = a
            .iter()
            .zip(&b)
            .map(|(a, b)| {
                if metric.direction == "higher" {
                    a - b
                } else {
                    b - a
                }
            })
            .collect::<Vec<_>>();
        let ci = if metric.direction == "diagnostic" {
            json!({"status":"diagnostic","reason":"signed bias has no preferred direction"})
        } else {
            interval(&differences, &manifest.bootstrap)
        };
        comparisons.push(json!({"metric":metric,"candidate":(!a.is_empty()).then(||mean(&a)),"reference":(!b.is_empty()).then(||mean(&b)),"case_count":rows.iter().filter(|r|r.status=="measured").count(),"block_ids":blocks.keys().collect::<Vec<_>>(),"improvement":ci}));
    }
    let missing = scores
        .iter()
        .filter(|r| {
            r.status == "missing"
                && manifest
                    .metrics
                    .iter()
                    .any(|m| m.id == r.metric && m.required)
        })
        .map(|r| format!("{}/{}", r.case, r.metric))
        .collect::<Vec<_>>();
    let optional = scores
        .iter()
        .filter(|r| {
            r.status == "missing"
                && manifest
                    .metrics
                    .iter()
                    .any(|m| m.id == r.metric && !m.required)
        })
        .map(|r| format!("{}/{}", r.case, r.metric))
        .collect::<Vec<_>>();
    json!({"schema":RECEIPT,"manifest":manifest_identity,"configuration":manifest,"candidate":manifest.candidate,"reference":manifest.reference,"case_scores":scores,"comparisons":comparisons,"sources":identities,"missing_required":missing,"missing_optional":optional,"coverage_status":if !missing.is_empty(){"incomplete"}else if !optional.is_empty(){"partial"}else{"complete"},"bootstrap_unit":"independent case block, equal block weight and equal case weight within block","error_convention":"forecast minus observation; positive improvement favors candidate","rng":"SplitMix64, unbiased rejection indices, linear percentile quantile","notes":manifest.notes})
}

pub fn write_outputs(out: &Path, document: &Value) -> Result<(), String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    std::fs::write(
        out.join("scorecard.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(document).map_err(|e| e.to_string())?
        ),
    )
    .map_err(|e| e.to_string())?;
    let mut csv = std::fs::File::create(out.join("scorecard.csv")).map_err(|e| e.to_string())?;
    rw_obs::csv::write_row(
        &mut csv,
        &[
            "case",
            "block",
            "metric",
            "candidate",
            "reference",
            "count",
            "status",
            "reason",
        ]
        .map(str::to_string),
    )?;
    for r in document["case_scores"].as_array().unwrap() {
        rw_obs::csv::write_row(
            &mut csv,
            &[
                r["case"].as_str().unwrap().into(),
                r["block"].as_str().unwrap().into(),
                r["metric"].as_str().unwrap().into(),
                r["candidate"]
                    .as_f64()
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                r["reference"]
                    .as_f64()
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
                r["count"].to_string(),
                r["status"].as_str().unwrap().into(),
                r["reason"].as_str().unwrap_or("").into(),
            ],
        )?;
    }
    let mut md = format!(
        "Observation scorecard: {}\n\nCandidate: {}. Reference: {}. Positive improvement favors candidate.\n\n| Metric | Candidate | Reference | Improvement | CI | Blocks | Status |\n| --- | ---: | ---: | ---: | --- | ---: | --- |\n",
        document["coverage_status"].as_str().unwrap(),
        document["candidate"].as_str().unwrap(),
        document["reference"].as_str().unwrap()
    );
    let format_value = |v: &Value| {
        v.as_f64()
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "missing".into())
    };
    for r in document["comparisons"].as_array().unwrap() {
        let ci = &r["improvement"];
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} to {} | {} | {} |\n",
            r["metric"]["id"].as_str().unwrap(),
            format_value(&r["candidate"]),
            format_value(&r["reference"]),
            format_value(&ci["estimate"]),
            format_value(&ci["lower"]),
            format_value(&ci["upper"]),
            r["block_ids"].as_array().unwrap().len(),
            ci["status"].as_str().unwrap()
        ));
    }
    md.push_str("\nIntervals resample whole independent blocks. Overlapping cycles must share a block. Exploratory intervals do not establish improvement. Missing fields are listed in scorecard.csv.\n");
    std::fs::write(out.join("scorecard.md"), md).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bootstrap_hand_check_and_repeatability() {
        let config = Bootstrap {
            replicates: 5000,
            confidence: 0.95,
            seed: 17,
            minimum_blocks: 4,
        };
        let result = interval(&[1.0, 3.0], &config);
        assert_eq!(result["estimate"], 2.0);
        assert_eq!(result["lower"], 1.0);
        assert_eq!(result["upper"], 3.0);
        assert_eq!(result["status"], "exploratory");
        assert_eq!(result, interval(&[1.0, 3.0], &config));
        assert_eq!(interval(&[1.0], &config)["status"], "not_measured");
    }
    #[test]
    fn rmse_is_pooled_before_case_bootstrap() {
        assert!((statistic(&[1.0, 3.0], "rmse") - 5.0f64.sqrt()).abs() < 1e-15);
        assert_eq!(statistic(&[-1.0, 3.0], "bias"), 1.0);
        assert_eq!(statistic(&[-1.0, 3.0], "mae"), 2.0);
    }
    #[test]
    fn reversed_skill_direction_is_not_accepted() {
        let mut m:Manifest=serde_json::from_value(json!({"schema":SCHEMA,"candidate":"A","reference":"B","cases":[{"id":"one","block":"one","receipts":["one.json"]}],"metrics":[{"id":"score","source":"mrms","quantity":"composite_reflectivity","units":"dBZ","statistic":"fss","threshold":35.0,"width_km":9.0}]})).unwrap();
        assert!(m.validate().unwrap_err().contains("label a worse forecast"));
        m.metrics[0].direction = "higher".into();
        m.validate().unwrap();
    }
    #[test]
    fn blocks_prevent_related_cycles_inflating_independence() {
        let manifest:Manifest=serde_json::from_value(json!({"schema":SCHEMA,"candidate":"A","reference":"B","cases":[{"id":"one","block":"shared","receipts":["one.json"]},{"id":"two","block":"shared","receipts":["two.json"]}],"metrics":[{"id":"t","source":"asos","quantity":"temperature_2m","units":"K","required":true}]})).unwrap();
        manifest.validate().unwrap();
        let scores = manifest
            .cases
            .iter()
            .map(|c| CaseScore {
                case: c.id.clone(),
                block: c.block.clone(),
                metric: "t".into(),
                candidate: Some(1.),
                reference: Some(2.),
                count: 100000,
                status: "measured".into(),
                reason: None,
            })
            .collect::<Vec<_>>();
        let output = aggregate(&manifest, &scores, vec![], Value::Null);
        assert_eq!(
            output["comparisons"][0]["improvement"]["status"],
            "not_measured"
        );
        assert_eq!(
            output["comparisons"][0]["block_ids"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        let mut partial = scores.clone();
        partial[0].candidate = None;
        partial[0].reference = None;
        partial[0].status = "missing".into();
        let output = aggregate(&manifest, &partial, vec![], Value::Null);
        assert!(
            output["comparisons"][0]["block_ids"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(output["coverage_status"], "incomplete");
    }
}
