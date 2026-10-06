//! Paired station and neighbourhood verification with common support.

use chrono::{DateTime, NaiveDateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::verification_io::{ArmData, ArmSpec, RadarSpec, StationObservation};

pub const REQUEST_SCHEMA: &str = "gpuwm.verify-visuals.request.v1";
pub const RECEIPT_SCHEMA: &str = "gpuwm.verify-visuals.receipt.v1";
pub const INVENTORY_SCHEMA: &str = "gpuwm.verify-visuals.inventory.v1";
pub const STATION_QUANTITIES: &[(&str, &str, &str)] = &[
    ("temperature_2m", "2 m temperature", "K"),
    ("dewpoint_2m", "2 m dewpoint", "K"),
    ("wind_speed_10m", "10 m wind speed", "m s-1"),
];
pub const RADAR_QUANTITIES: &[(&str, &str, &str, &[f64])] = &[
    (
        "composite_reflectivity",
        "Composite reflectivity",
        "dBZ",
        &[20.0, 35.0],
    ),
    (
        "precipitation_1h",
        "1 h precipitation",
        "mm h-1",
        &[1.0, 5.0],
    ),
];

fn default_widths() -> Vec<f64> {
    vec![3.0, 9.0, 27.0]
}
fn default_window() -> i64 {
    600
}
fn default_radar_window() -> i64 {
    240
}
fn default_mode() -> String {
    "observed".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationRequest {
    pub schema: String,
    pub valid_time: String,
    pub domain: String,
    #[serde(default)]
    pub dx_km: f64,
    #[serde(default)]
    pub dy_km: Option<f64>,
    pub arms: Vec<ArmSpec>,
    #[serde(default)]
    pub stations_path: Option<PathBuf>,
    #[serde(default)]
    pub station_table_path: Option<PathBuf>,
    #[serde(default)]
    pub radar: Vec<RadarSpec>,
    #[serde(default = "default_widths")]
    pub widths_km: Vec<f64>,
    #[serde(default = "default_window")]
    pub station_window_seconds: i64,
    #[serde(default = "default_radar_window")]
    pub radar_window_seconds: i64,
    #[serde(default)]
    pub out_root: Option<PathBuf>,
    #[serde(default = "default_mode")]
    pub station_mode: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct StationArmScore {
    pub label: String,
    pub bias: Option<f64>,
    pub rmse: Option<f64>,
    pub count: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct StationScore {
    pub quantity: String,
    pub units: String,
    pub status: String,
    pub arms: Vec<StationArmScore>,
    pub winner: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ForecastSample {
    pub label: String,
    pub values: BTreeMap<String, f64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct StationSample {
    #[serde(flatten)]
    pub observation: StationObservation,
    pub forecast: Vec<ForecastSample>,
}
#[derive(Clone, Debug, Serialize)]
pub struct RadarArmScore {
    pub label: String,
    pub fss: Option<f64>,
    pub count: usize,
    pub event_cells: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct RadarScore {
    pub quantity: String,
    pub units: String,
    pub threshold: f64,
    pub width_km: f64,
    pub actual_width_km: f64,
    pub actual_height_km: f64,
    pub width_cells: usize,
    pub height_cells: usize,
    pub observed_event_cells: usize,
    pub common_cells: usize,
    pub status: String,
    pub arms: Vec<RadarArmScore>,
    pub winner: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Artifact {
    pub product: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

impl Artifact {
    pub fn new(product: String, path: PathBuf) -> Result<Self, String> {
        let identity = crate::verification_io::source_identity(&path)?;
        Ok(Self {
            product,
            path,
            bytes: identity["bytes"].as_u64().unwrap(),
            sha256: identity["sha256"].as_str().unwrap().into(),
        })
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct VerificationReceipt {
    pub schema: String,
    pub valid_time: String,
    pub domain: String,
    pub request_sha256: String,
    pub error_convention: String,
    pub station_interpolation: String,
    pub temperature_method: String,
    pub wind_metric: String,
    pub neighborhood_support: String,
    pub sources: Vec<Value>,
    pub stations: Vec<StationScore>,
    pub station_samples: Vec<StationSample>,
    pub station_drops: Vec<Value>,
    pub radar: Vec<RadarScore>,
    pub artifacts: Vec<Artifact>,
    pub skipped: Vec<String>,
}

pub fn parse_time(text: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(text)
        .map(|v| v.with_timezone(&Utc))
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S").map(|v| v.and_utc()))
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").map(|v| v.and_utc()))
        .map_err(|e| format!("invalid verification time {text}: {e}"))
}

impl VerificationRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != REQUEST_SCHEMA {
            return Err(format!("expected {REQUEST_SCHEMA}"));
        }
        parse_time(&self.valid_time)?;
        if self.arms.is_empty() {
            return Err("verification requires at least one forecast arm".into());
        }
        if !self.dx_km.is_finite()
            || self.dx_km < 0.0
            || !self.radar.is_empty() && self.dx_km == 0.0
        {
            return Err("verification spacing must be positive kilometres".into());
        }
        if self.dy_km.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            return Err("verification row spacing must be positive kilometres".into());
        }
        if self.station_window_seconds <= 0 || self.radar_window_seconds < 0 {
            return Err(
                "verification time windows must be nonnegative, with a positive station window"
                    .into(),
            );
        }
        if self.widths_km.is_empty() || self.widths_km.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("neighbourhood widths must be positive kilometres".into());
        }
        if self.station_mode != "observed" && self.station_mode != "error" {
            return Err("station_mode must be observed or error".into());
        }
        for (i, arm) in self.arms.iter().enumerate() {
            if arm.label.trim().is_empty() || self.arms[..i].iter().any(|a| a.label == arm.label) {
                return Err("forecast arm labels must be nonempty and unique".into());
            }
        }
        Ok(())
    }
}

pub fn paired_station_scores(
    request: &VerificationRequest,
    arms: &[ArmData],
    observations: &[StationObservation],
) -> Result<(Vec<StationScore>, Vec<StationSample>, Vec<Value>), String> {
    let target = parse_time(&request.valid_time)?;
    let frozen = if let Some(path) = &request.station_table_path {
        let table = crate::verification_io::read_json(path)?;
        Some(
            table["stations"]
                .as_array()
                .ok_or("frozen station table has no stations")?
                .iter()
                .filter_map(|s| s["station_id"].as_str().map(str::to_string))
                .collect::<std::collections::BTreeSet<_>>(),
        )
    } else {
        None
    };
    let mut samples = Vec::new();
    let mut drops = Vec::new();
    for observation in observations {
        if frozen
            .as_ref()
            .is_some_and(|set| !set.contains(&observation.station_id))
        {
            drops.push(serde_json::json!({"station_id":observation.station_id,"reason":"outside-frozen-station-set"}));
            continue;
        }
        let time = parse_time(&observation.observation_time)?;
        let offset = (time - target).num_seconds().abs();
        if offset > request.station_window_seconds {
            drops.push(serde_json::json!({"station_id":observation.station_id,"reason":"outside-time-window","offset_seconds":offset}));
            continue;
        }
        let mut forecast = Vec::new();
        for arm in arms {
            let position = arm
                .grid
                .as_ref()
                .and_then(|g| g.position(observation.latitude, observation.longitude));
            if arm.grid.is_some() && position.is_none() {
                forecast.push(ForecastSample {
                    label: arm.label.clone(),
                    values: BTreeMap::new(),
                });
                continue;
            }
            let mut values = BTreeMap::new();
            for &(q, _, _) in STATION_QUANTITIES {
                let value = arm
                    .points
                    .get(&observation.station_id)
                    .and_then(|p| {
                        if q == "temperature_2m" {
                            p.values.get("temperature_2m_raw").copied()
                        } else {
                            p.values.get(q).copied()
                        }
                    })
                    .or_else(|| {
                        let grid = arm.grid.as_ref()?;
                        grid.sample(arm.fields.get(q)?, position?)
                    });
                if let Some(value) = value.filter(|v| v.is_finite() && v.abs() < 1e20) {
                    values.insert(q.into(), value);
                }
            }
            forecast.push(ForecastSample {
                label: arm.label.clone(),
                values,
            });
        }
        if forecast.iter().any(|f| f.values.is_empty()) {
            drops.push(serde_json::json!({"station_id":observation.station_id,"reason":"forecast-support-missing"}));
        }
        samples.push(StationSample {
            observation: observation.clone(),
            forecast,
        });
    }
    let mut scores = Vec::new();
    for &(q, _, units) in STATION_QUANTITIES {
        let low = if q == "wind_speed_10m" { 0.0 } else { 233.15 };
        let high = if q == "wind_speed_10m" { 75.0 } else { 328.15 };
        let paired: Vec<_> = samples
            .iter()
            .filter(|s| {
                s.observation
                    .values
                    .get(q)
                    .is_some_and(|v| v.is_finite() && *v >= low && *v <= high)
                    && !(q == "dewpoint_2m"
                        && s.observation
                            .values
                            .get("temperature_2m")
                            .is_some_and(|t| s.observation.values[q] > *t))
                    && s.forecast.iter().all(|f| f.values.contains_key(q))
            })
            .collect();
        let count = paired.len();
        let scored: Vec<_> = arms
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let (sum, squares) = paired.iter().fold((0.0, 0.0), |(sum, squares), s| {
                    let error = s.forecast[i].values[q] - s.observation.values[q];
                    (sum + error, squares + error * error)
                });
                StationArmScore {
                    label: a.label.clone(),
                    bias: (count > 0).then(|| sum / count as f64),
                    rmse: (count > 0).then(|| (squares / count as f64).sqrt()),
                    count,
                }
            })
            .collect();
        let winner = unique_winner(
            &scored
                .iter()
                .map(|s| (s.label.clone(), s.rmse))
                .collect::<Vec<_>>(),
            false,
        );
        scores.push(StationScore {
            quantity: q.into(),
            units: units.into(),
            status: if count == 0 {
                "missing"
            } else if arms.len() < 2 {
                "unpaired"
            } else {
                "ready"
            }
            .into(),
            arms: scored,
            winner,
        });
    }
    Ok((scores, samples, drops))
}

fn unique_winner(values: &[(String, Option<f64>)], higher: bool) -> Option<String> {
    if values.len() < 2 {
        return None;
    }
    if values.iter().any(|(_, v)| v.is_none()) {
        return None;
    }
    let best = values.iter().min_by(|a, b| {
        let order = a.1.unwrap().total_cmp(&b.1.unwrap());
        if higher {
            order.reverse()
        } else {
            order
        }
    })?;
    let best_value = best.1.unwrap();
    let tied = values
        .iter()
        .filter(|(_, v)| (v.unwrap() - best_value).abs() <= 1e-12)
        .count();
    (tied == 1).then(|| best.0.clone())
}

fn integral(values: &[f64], ny: usize, nx: usize) -> Vec<f64> {
    let mut out = vec![0.0; (ny + 1) * (nx + 1)];
    for y in 0..ny {
        let mut row = 0.0;
        for x in 0..nx {
            row += values[y * nx + x];
            out[(y + 1) * (nx + 1) + x + 1] = out[y * (nx + 1) + x + 1] + row;
        }
    }
    out
}
fn rectangle(sums: &[f64], nx: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    sums[y1 * (nx + 1) + x1] - sums[y0 * (nx + 1) + x1] - sums[y1 * (nx + 1) + x0]
        + sums[y0 * (nx + 1) + x0]
}

/// FSS uses a common finite/coverage mask across every arm. Only complete
/// square neighbourhoods wholly within that mask contribute; no missing pixel
/// becomes a correct negative and no edge neighbourhood changes its area.
pub fn paired_fss(
    quantity: &str,
    units: &str,
    threshold: f64,
    width_km: f64,
    dx_km: f64,
    ny: usize,
    nx: usize,
    observed: &[f64],
    valid: &[bool],
    forecasts: &[(String, Vec<f64>)],
) -> Result<RadarScore, String> {
    paired_fss_rect(
        quantity, units, threshold, width_km, dx_km, dx_km, ny, nx, observed, valid, forecasts,
    )
}

fn nearest_odd_width(width_km: f64, spacing_km: f64) -> usize {
    (((width_km / spacing_km - 1.0) * 0.5).round().max(0.0) as usize) * 2 + 1
}

pub fn paired_fss_rect(
    quantity: &str,
    units: &str,
    threshold: f64,
    width_km: f64,
    dx_km: f64,
    dy_km: f64,
    ny: usize,
    nx: usize,
    observed: &[f64],
    valid: &[bool],
    forecasts: &[(String, Vec<f64>)],
) -> Result<RadarScore, String> {
    let cells = nx
        .checked_mul(ny)
        .ok_or("verification grid dimensions overflow")?;
    if observed.len() != cells
        || valid.len() != cells
        || forecasts.iter().any(|(_, v)| v.len() != cells)
    {
        return Err("FSS arrays and grid differ".into());
    }
    let width = nearest_odd_width(width_km, dx_km);
    let half = width / 2;
    let height = nearest_odd_width(width_km, dy_km);
    let half_y = height / 2;
    let common: Vec<bool> = (0..cells)
        .map(|i| {
            valid[i]
                && observed[i].is_finite()
                && forecasts
                    .iter()
                    .all(|(_, v)| v[i].is_finite() && v[i].abs() < 1e20)
        })
        .collect();
    let common_cells = common.iter().filter(|&&v| v).count();
    let observed_event_cells = (0..cells)
        .filter(|&i| common[i] && observed[i] >= threshold)
        .count();
    let mask = integral(
        &common.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        ny,
        nx,
    );
    let obs = integral(
        &(0..cells)
            .map(|i| f64::from(common[i] && observed[i] >= threshold))
            .collect::<Vec<_>>(),
        ny,
        nx,
    );
    let arm_integrals: Vec<_> = forecasts
        .iter()
        .map(|(_, values)| {
            integral(
                &(0..cells)
                    .map(|i| f64::from(common[i] && values[i] >= threshold))
                    .collect::<Vec<_>>(),
                ny,
                nx,
            )
        })
        .collect();
    let mut numerator = vec![0.0; forecasts.len()];
    let mut denominator = vec![0.0; forecasts.len()];
    let mut count = 0;
    if width <= nx && height <= ny {
        for y in half_y..ny - half_y {
            for x in half..nx - half {
                let (x0, y0, x1, y1) = (x - half, y - half_y, x + half + 1, y + half_y + 1);
                let area = (width * height) as f64;
                if rectangle(&mask, nx, x0, y0, x1, y1) != area {
                    continue;
                }
                count += 1;
                let o = rectangle(&obs, nx, x0, y0, x1, y1) / area;
                for (i, a) in arm_integrals.iter().enumerate() {
                    let f = rectangle(a, nx, x0, y0, x1, y1) / area;
                    numerator[i] += (f - o).powi(2);
                    denominator[i] += f * f + o * o;
                }
            }
        }
    }
    let arms: Vec<_> = forecasts
        .iter()
        .enumerate()
        .map(|(i, (label, values))| RadarArmScore {
            label: label.clone(),
            fss: (count > 0 && denominator[i] > 0.0)
                .then(|| (1.0 - numerator[i] / denominator[i]).clamp(0.0, 1.0)),
            count,
            event_cells: (0..cells)
                .filter(|&j| common[j] && values[j] >= threshold)
                .count(),
        })
        .collect();
    let status = if count == 0 {
        "missing-support"
    } else if observed_event_cells == 0 {
        "no-observed-events"
    } else if forecasts.len() < 2 {
        "unpaired"
    } else {
        "ready"
    };
    let winner = if status == "ready" {
        unique_winner(
            &arms
                .iter()
                .map(|a| (a.label.clone(), a.fss))
                .collect::<Vec<_>>(),
            true,
        )
    } else {
        None
    };
    Ok(RadarScore {
        quantity: quantity.into(),
        units: units.into(),
        threshold,
        width_km,
        actual_width_km: width as f64 * dx_km,
        actual_height_km: height as f64 * dy_km,
        width_cells: width,
        height_cells: height,
        observed_event_cells,
        common_cells,
        status: status.into(),
        arms,
        winner,
    })
}

pub fn mapped_fields(
    target: &crate::verification_io::GridData,
    source: &crate::verification_io::GridData,
    values: &[f64],
) -> Result<Vec<f64>, String> {
    if target.nx == source.nx
        && target.ny == source.ny
        && target.lat == source.lat
        && target.lon == source.lon
    {
        return Ok(values.to_vec());
    }
    let mapping = crate::compare::match_grids(
        &target.lat,
        &target.lon,
        target.ny,
        target.nx,
        &source.lat,
        &source.lon,
        source.ny,
        source.nx,
    )?;
    Ok(mapping
        .source_index
        .par_iter()
        .enumerate()
        .map(|(target_index, &i)| {
            if i == u32::MAX
                || source
                    .position_from_nearest(
                        f64::from(target.lat[target_index]),
                        f64::from(target.lon[target_index]),
                        i as usize,
                    )
                    .is_none()
            {
                f64::NAN
            } else {
                values.get(i as usize).copied().unwrap_or(f64::NAN)
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn naive_packaged_times_are_utc_with_either_separator() {
        let expected = parse_time("2026-10-04T12:00:00Z").unwrap();
        assert_eq!(parse_time("2026-10-04 12:00:00").unwrap(), expected);
        assert_eq!(parse_time("2026-10-04T12:00:00").unwrap(), expected);
        assert_ne!(parse_time("2026-10-04 13:00:00").unwrap(), expected);
        assert!(parse_time("04/10/2026 12:00:00").is_err());
    }
    #[test]
    fn target_outside_source_lattice_never_becomes_a_valid_dry_negative() {
        use crate::verification_io::GridData;
        let source = GridData {
            lat: vec![30., 30., 31., 31.],
            lon: vec![-100., -99., -100., -99.],
            ny: 2,
            nx: 2,
            projection: None,
        };
        let target = GridData {
            lat: vec![30.25, 30.25, 30.75, 30.75],
            lon: vec![-100.25, -99.75, -100.25, -99.75],
            ny: 2,
            nx: 2,
            projection: None,
        };
        let old = crate::compare::match_grids(
            &target.lat,
            &target.lon,
            2,
            2,
            &source.lat,
            &source.lon,
            2,
            2,
        )
        .unwrap();
        assert_ne!(old.source_index[0], u32::MAX);
        let values = mapped_fields(&target, &source, &[0.; 4]).unwrap();
        assert!(values[0].is_nan());
        assert!(values[2].is_nan());
        assert_eq!(values[1], 0.);
        assert_eq!(values[3], 0.);
        let validity = values.iter().map(|v| v.is_finite()).collect::<Vec<_>>();
        let score = paired_fss(
            "precipitation_1h",
            "mm h-1",
            1.,
            1.,
            1.,
            2,
            2,
            &values,
            &validity,
            &[("A".into(), vec![0.; 4]), ("B".into(), vec![0.; 4])],
        )
        .unwrap();
        assert_eq!(score.common_cells, 2);
        assert_eq!(score.arms[0].count, 2);
        assert!(score.winner.is_none());
    }
    #[test]
    fn station_pairing_uses_raw_temperature_and_never_fabricates_wind() {
        use crate::verification_io::{ArmData, PointForecast};
        let request:VerificationRequest=serde_json::from_value(serde_json::json!({"schema":REQUEST_SCHEMA,"valid_time":"2030-01-01T01:00:00","domain":"d01","arms":[{"label":"A","kind":"points","path":"a"},{"label":"B","kind":"points","path":"b"}]})).unwrap();
        let point = |id: &str, raw: f64| PointForecast {
            station_id: id.into(),
            terrain_m: None,
            values: [
                ("temperature_2m".into(), 330.0),
                ("temperature_2m_raw".into(), raw),
            ]
            .into(),
        };
        let arm = |label: &str, points: Vec<PointForecast>| ArmData {
            label: label.into(),
            grid: None,
            fields: BTreeMap::new(),
            points: points
                .into_iter()
                .map(|p| (p.station_id.clone(), p))
                .collect(),
            provenance: Value::Null,
        };
        let arms = vec![
            arm("A", vec![point("S1", 302.), point("S2", 305.)]),
            arm("B", vec![point("S1", 301.)]),
        ];
        let observation = |id: &str| StationObservation {
            station_id: id.into(),
            latitude: 30.,
            longitude: -100.,
            elevation_m: 100.,
            observation_time: "2030-01-01T00:55:00".into(),
            values: [
                ("temperature_2m".into(), 300.),
                ("wind_speed_10m".into(), 3.),
            ]
            .into(),
        };
        let (scores, _, _) =
            paired_station_scores(&request, &arms, &[observation("S1"), observation("S2")])
                .unwrap();
        assert_eq!(scores[0].arms[0].count, 1);
        assert_eq!(scores[0].arms[0].bias, Some(2.));
        assert_eq!(scores[0].arms[1].bias, Some(1.));
        assert_eq!(scores[0].winner, Some("B".into()));
        assert!(scores[2]
            .arms
            .iter()
            .all(|a| a.count == 0 && a.rmse.is_none()));
        assert!(scores[2].winner.is_none());
    }
    #[test]
    fn rectangular_neighbourhoods_preserve_physical_axis_widths() {
        let o = vec![2.; 25];
        let s = paired_fss_rect(
            "precipitation_1h",
            "mm h-1",
            1.,
            3.,
            1.,
            2.,
            5,
            5,
            &o,
            &[true; 25],
            &[("A".into(), o.clone()), ("B".into(), o.clone())],
        )
        .unwrap();
        assert_eq!(s.width_cells, 3);
        assert_eq!(s.height_cells, 1);
        assert_eq!(s.actual_width_km, 3.);
        assert_eq!(s.actual_height_km, 2.);
        assert_eq!(s.arms[0].count, 15);
    }
    #[test]
    fn no_event_has_no_winner_or_perfect_score() {
        let s = paired_fss(
            "composite_reflectivity",
            "dBZ",
            20.,
            3.,
            1.,
            3,
            3,
            &[0.; 9],
            &[true; 9],
            &[("A".into(), vec![0.; 9]), ("B".into(), vec![0.; 9])],
        )
        .unwrap();
        assert_eq!(s.status, "no-observed-events");
        assert!(s.winner.is_none());
        assert!(s.arms.iter().all(|a| a.fss.is_none()));
    }
    #[test]
    fn perfect_and_displaced_event_have_correct_fss() {
        let o = vec![0., 0., 0., 0., 30., 0., 0., 0., 0.];
        let mut b = vec![0.; 9];
        b[5] = 30.;
        let s = paired_fss(
            "composite_reflectivity",
            "dBZ",
            20.,
            1.,
            1.,
            3,
            3,
            &o,
            &[true; 9],
            &[("A".into(), o.clone()), ("B".into(), b)],
        )
        .unwrap();
        assert_eq!(s.arms[0].fss, Some(1.));
        assert_eq!(s.arms[1].fss, Some(0.));
        assert_eq!(s.winner, Some("A".into()));
    }
    #[test]
    fn missing_coverage_never_becomes_dry_negative() {
        let s = paired_fss(
            "precipitation_1h",
            "mm h-1",
            1.,
            3.,
            1.,
            3,
            3,
            &[0.; 9],
            &[true, true, true, true, false, true, true, true, true],
            &[("A".into(), vec![0.; 9]), ("B".into(), vec![0.; 9])],
        )
        .unwrap();
        assert_eq!(s.arms[0].count, 0);
        assert_eq!(s.common_cells, 8);
        assert_eq!(s.status, "missing-support");
    }
    #[test]
    fn mask_is_common_to_every_arm() {
        let o = vec![2.; 9];
        let mut b = o.clone();
        b[4] = f64::NAN;
        let s = paired_fss(
            "precipitation_1h",
            "mm h-1",
            1.,
            1.,
            1.,
            3,
            3,
            &o,
            &[true; 9],
            &[("A".into(), o.clone()), ("B".into(), b)],
        )
        .unwrap();
        assert_eq!(s.common_cells, 8);
        assert!(s.arms.iter().all(|a| a.count == 8 && a.fss == Some(1.)));
        assert!(s.winner.is_none());
    }
}
