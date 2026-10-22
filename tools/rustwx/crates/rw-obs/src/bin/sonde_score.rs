//! `rw_sonde_score` -- a forecast's history frames scored against radiosondes.
//!
//! The observation side is the neutral table `rw_igra2 table` writes
//! (`gpuwm-obs.table.v2`): one row per station, sounding, level and
//! variable, each level at its own time (release time plus elapsed time).
//! Only `sonde_level` rows with a pressure are read; every other row is
//! counted and left alone.
//!
//! The model side is WRF history frames (`wrfout_d0N_*`, any NetCDF
//! container netcrust reads).  For each sonde level:
//!
//! * the frame is the one whose valid time is nearest the level's own time,
//!   within `--match-seconds` (default 1800 s); a level with no frame that
//!   close is dropped by name;
//! * the column is the mass point nearest the station (great-circle over
//!   `XLAT`/`XLONG`); a station whose nearest point is on the domain's
//!   outermost `--rim-cells` (default 5, the specified and relaxation zone)
//!   is dropped, never scored against boundary-forced values;
//! * the model value is interpolated to the level's pressure linearly in
//!   ln(p) between the two model levels that bracket it; a pressure below
//!   the model's lowest level or above its top is dropped, never
//!   extrapolated;
//! * temperature is `(T + 300) * (p / 1e5)^(Rd/cp)` with `p = P + PB`;
//!   dewpoint is the inverse Magnus form over the vapour pressure
//!   `e = qv p / (0.622 + qv)` (the same constants the engine's surface
//!   dewpoint uses: 6.112 hPa, 17.67, 243.5 C); winds are destaggered to the
//!   mass point and rotated earth-relative with `COSALPHA`/`SINALPHA`
//!   (WRF's own uvmet rotation).  A frame without the rotation fields
//!   scores no wind and says so.
//!
//! Output: per variable and per mandatory level (plus `all`), the count
//! and the sums of error, absolute error and squared error (scoreboard-v2's
//! sufficient statistics, so runs pool by addition), with bias and RMSE
//! derived, every drop by reason, and every input by SHA-256.
//!
//! ```text
//! rw_sonde_score --table OBS.csv --frame WRFOUT [--frame WRFOUT ...] --out SCORES.json
//!                [--match-seconds 1800] [--rim-cells 5] [--label NAME]
//! ```

use std::collections::BTreeMap;
use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use chrono::{DateTime, NaiveDateTime, Utc};
use serde_json::json;

use rw_obs::table::{MEAS_SONDE_LEVEL, VAR_DEWPOINT, VAR_TEMPERATURE, VAR_WIND_U, VAR_WIND_V};
use rw_obs::{err, hex_sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const SCHEMA: &str = "gpuwm-obs.sonde-scores.v1";
const RD_OVER_CP: f64 = 287.0 / 1004.5;
/// The mandatory levels a row is grouped under; any other level pools into
/// `all` only.
const MANDATORY_HPA: [i64; 10] = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150];

#[derive(Clone, Debug)]
struct ObsRow {
    station: String,
    lat: f64,
    lon: f64,
    level_pa: f64,
    time: DateTime<Utc>,
    variable: String,
    value: f64,
}

struct Frame {
    path: PathBuf,
    sha256: String,
    valid: DateTime<Utc>,
    nx: usize,
    ny: usize,
    nz: usize,
    lat: Vec<f64>,
    lon: Vec<f64>,
    pressure: Vec<f64>,
    temperature: Vec<f64>,
    qv: Vec<f64>,
    /// Earth-relative mass-point winds, when the frame carries the rotation.
    wind: Option<(Vec<f64>, Vec<f64>)>,
}

fn parse_time(text: &str) -> Result<DateTime<Utc>> {
    let text = text.trim().trim_end_matches('Z');
    for format in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d_%H:%M:%S"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return Ok(naive.and_utc());
        }
    }
    Err(err(format!("unreadable time {text:?}")))
}

/// The table's sonde-level rows of the four scored variables; the rest counted.
fn read_table(path: &PathBuf, skipped: &mut BTreeMap<String, u64>) -> Result<Vec<ObsRow>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|line| !line.starts_with('#') && !line.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| err("observation table is empty"))?
        .split(',')
        .collect();
    let column = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| err(format!("observation table has no {name} column")))
    };
    let (station, lat, lon, level, time, variable, value, measurement) = (
        column("station_id")?,
        column("latitude_deg")?,
        column("longitude_deg")?,
        column("level_pa")?,
        column("valid_time")?,
        column("variable")?,
        column("value")?,
        column("measurement")?,
    );
    let wanted = [VAR_TEMPERATURE, VAR_DEWPOINT, VAR_WIND_U, VAR_WIND_V];
    let mut rows = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != header.len() {
            return Err(err(format!("observation row has {} fields, header {}", fields.len(), header.len())));
        }
        if fields[measurement] != MEAS_SONDE_LEVEL || !wanted.contains(&fields[variable]) {
            *skipped.entry(format!("not a scored sonde row ({})", fields[measurement])).or_default() += 1;
            continue;
        }
        let Ok(level_pa) = fields[level].parse::<f64>() else {
            *skipped.entry("sonde row without a pressure".into()).or_default() += 1;
            continue;
        };
        rows.push(ObsRow {
            station: fields[station].to_string(),
            lat: fields[lat].parse()?,
            lon: fields[lon].parse()?,
            level_pa,
            time: parse_time(fields[time])?,
            variable: fields[variable].to_string(),
            value: fields[value].parse()?,
        });
    }
    Ok(rows)
}

fn first_record(file: &netcrust::File, name: &str) -> Result<(Vec<f64>, Vec<usize>)> {
    let array = file
        .read_array_f64_first_record_or_all(name)
        .map_err(|e| err(format!("history frame variable {name}: {e}")))?;
    let shape = array.shape().to_vec();
    Ok((array.into_values(), shape))
}

fn read_frame(path: PathBuf) -> Result<Frame> {
    let bytes = std::fs::read(&path)?;
    let sha256 = hex_sha256(&bytes);
    let file = netcrust::File::from_bytes(&bytes)
        .map_err(|e| err(format!("{}: not a NetCDF history frame ({e})", path.display())))?;
    let times = file.read_strings("Times")?;
    let valid = parse_time(times.first().ok_or_else(|| err("history frame has no Times record"))?)?;
    let (lat, horizontal) = first_record(&file, "XLAT")?;
    let (lon, _) = first_record(&file, "XLONG")?;
    let (ny, nx) = match horizontal.as_slice() {
        [.., ny, nx] => (*ny, *nx),
        _ => return Err(err("XLAT is not two-dimensional")),
    };
    let (p, mass) = first_record(&file, "P")?;
    let (pb, _) = first_record(&file, "PB")?;
    let (theta, _) = first_record(&file, "T")?;
    let (qv, _) = first_record(&file, "QVAPOR")?;
    let nz = match mass.as_slice() {
        [.., nz, y, x] if *y == ny && *x == nx => *nz,
        _ => return Err(err("mass fields are not [bottom_top, south_north, west_east]")),
    };
    let pressure: Vec<f64> = p.iter().zip(&pb).map(|(a, b)| a + b).collect();
    let temperature = theta
        .iter()
        .zip(&pressure)
        .map(|(t, p)| (t + 300.0) * (p / 1.0e5).powf(RD_OVER_CP))
        .collect();
    let wind = match (file.variable("COSALPHA"), file.variable("SINALPHA")) {
        (Some(_), Some(_)) => {
            let (u, ushape) = first_record(&file, "U")?;
            let (v, vshape) = first_record(&file, "V")?;
            if ushape.as_slice() != [nz, ny, nx + 1] || vshape.as_slice() != [nz, ny + 1, nx] {
                return Err(err("U/V are not on the staggered C grid of the mass fields"));
            }
            let (cosa, _) = first_record(&file, "COSALPHA")?;
            let (sina, _) = first_record(&file, "SINALPHA")?;
            let mut ue = vec![0.0; nz * ny * nx];
            let mut ve = vec![0.0; nz * ny * nx];
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let um = 0.5 * (u[(k * ny + j) * (nx + 1) + i] + u[(k * ny + j) * (nx + 1) + i + 1]);
                        let vm = 0.5 * (v[(k * (ny + 1) + j) * nx + i] + v[(k * (ny + 1) + j + 1) * nx + i]);
                        let (c, s) = (cosa[j * nx + i], sina[j * nx + i]);
                        let m = (k * ny + j) * nx + i;
                        ue[m] = um * c - vm * s;
                        ve[m] = vm * c + um * s;
                    }
                }
            }
            Some((ue, ve))
        }
        _ => None,
    };
    Ok(Frame { path, sha256, valid, nx, ny, nz, lat, lon, pressure, temperature, qv, wind })
}

fn nearest_cell(frame: &Frame, lat: f64, lon: f64) -> (usize, usize, f64) {
    let (sl, cl) = (lat.to_radians().sin(), lat.to_radians().cos());
    let mut best = (0, 0, f64::INFINITY);
    for j in 0..frame.ny {
        for i in 0..frame.nx {
            let m = j * frame.nx + i;
            let (plat, plon) = (frame.lat[m].to_radians(), frame.lon[m].to_radians());
            let c = sl * plat.sin() + cl * plat.cos() * (plon - lon.to_radians()).cos();
            let distance = c.clamp(-1.0, 1.0).acos() * 6_371_000.0;
            if distance < best.2 {
                best = (j, i, distance);
            }
        }
    }
    best
}

/// The column value at `level_pa`, linear in ln(p); None outside the column.
fn at_pressure(frame: &Frame, field: &[f64], j: usize, i: usize, level_pa: f64) -> Option<f64> {
    let (nx, ny) = (frame.nx, frame.ny);
    let column = |k: usize| (k * ny + j) * nx + i;
    for k in 0..frame.nz - 1 {
        let (p0, p1) = (frame.pressure[column(k)], frame.pressure[column(k + 1)]);
        if level_pa <= p0 && level_pa >= p1 {
            let w = (p0.ln() - level_pa.ln()) / (p0.ln() - p1.ln());
            return Some(field[column(k)] * (1.0 - w) + field[column(k + 1)] * w);
        }
    }
    None
}

fn dewpoint_k(qv: f64, pressure_pa: f64) -> f64 {
    let q = qv.max(0.0);
    let e_hpa = (q * pressure_pa / 100.0 / (0.622 + q)).max(1.0e-10);
    let ln_e = (e_hpa / 6.112).ln();
    243.5 * ln_e / (17.67 - ln_e) + 273.15
}

#[derive(Default)]
struct Sums {
    count: u64,
    sum_error: f64,
    sum_absolute_error: f64,
    sum_squared_error: f64,
}

impl Sums {
    fn add(&mut self, error: f64) {
        self.count += 1;
        self.sum_error += error;
        self.sum_absolute_error += error.abs();
        self.sum_squared_error += error * error;
    }
    fn json(&self) -> serde_json::Value {
        let n = self.count as f64;
        json!({"count": self.count, "sumError": self.sum_error,
               "sumAbsoluteError": self.sum_absolute_error, "sumSquaredError": self.sum_squared_error,
               "bias": if n > 0.0 { self.sum_error / n } else { f64::NAN },
               "rmse": if n > 0.0 { (self.sum_squared_error / n).sqrt() } else { f64::NAN }})
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut table, mut frames, mut out, mut label) = (None, Vec::new(), None, String::from("model"));
    let (mut match_seconds, mut rim) = (1800i64, 5usize);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| err(format!("{flag} needs a value")));
        match flag.as_str() {
            "--table" => table = Some(PathBuf::from(value()?)),
            "--frame" => frames.push(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--label" => label = value()?,
            "--match-seconds" => match_seconds = value()?.parse()?,
            "--rim-cells" => rim = value()?.parse()?,
            "--help" | "-h" => {
                println!("rw_sonde_score --table OBS.csv --frame WRFOUT [--frame ...] --out SCORES.json \
                          [--match-seconds 1800] [--rim-cells 5] [--label NAME]");
                return Ok(());
            }
            other => return Err(err(format!("unknown argument {other}"))),
        }
    }
    let table = table.ok_or_else(|| err("--table is required"))?;
    let out = out.ok_or_else(|| err("--out is required"))?;
    if frames.is_empty() {
        return Err(err("at least one --frame is required"));
    }
    let mut skipped = BTreeMap::new();
    let rows = read_table(&table, &mut skipped)?;
    let frames = frames.into_iter().map(read_frame).collect::<Result<Vec<_>>>()?;
    let mut drops: BTreeMap<String, u64> = BTreeMap::new();
    let mut sums: BTreeMap<(String, String), Sums> = BTreeMap::new();
    let mut stations_scored = std::collections::BTreeSet::new();
    let mut winds: BTreeMap<(String, String, i64), (Option<f64>, Option<f64>, usize)> = BTreeMap::new();
    for row in &rows {
        let Some(frame_index) = frames
            .iter()
            .enumerate()
            .map(|(n, f)| (n, (f.valid - row.time).num_seconds().abs()))
            .filter(|(_, gap)| *gap <= match_seconds)
            .min_by_key(|(_, gap)| *gap)
            .map(|(n, _)| n)
        else {
            *drops.entry("no frame within the match window".into()).or_default() += 1;
            continue;
        };
        let frame = &frames[frame_index];
        let (j, i, _) = nearest_cell(frame, row.lat, row.lon);
        if j < rim || i < rim || j + rim >= frame.ny || i + rim >= frame.nx {
            *drops.entry("station outside the domain interior".into()).or_default() += 1;
            continue;
        }
        let model = match row.variable.as_str() {
            v if v == VAR_TEMPERATURE => at_pressure(frame, &frame.temperature, j, i, row.level_pa),
            v if v == VAR_DEWPOINT => at_pressure(frame, &frame.qv, j, i, row.level_pa)
                .map(|q| dewpoint_k(q, row.level_pa)),
            _ => {
                let Some((ue, ve)) = &frame.wind else {
                    *drops.entry("frame carries no COSALPHA/SINALPHA for earth-relative wind".into()).or_default() += 1;
                    continue;
                };
                let field = if row.variable == VAR_WIND_U { ue } else { ve };
                at_pressure(frame, field, j, i, row.level_pa)
            }
        };
        let Some(model) = model else {
            *drops.entry("level outside the model column".into()).or_default() += 1;
            continue;
        };
        stations_scored.insert(row.station.clone());
        let hpa = (row.level_pa / 100.0).round() as i64;
        let tags: Vec<String> = if MANDATORY_HPA.contains(&hpa) && (row.level_pa / 100.0 - hpa as f64).abs() < 0.01 {
            vec![hpa.to_string(), "all".into()]
        } else {
            vec!["all".into()]
        };
        if row.variable == VAR_WIND_U || row.variable == VAR_WIND_V {
            // The vector error needs both components of one level.
            let key = (row.station.clone(), row.time.to_rfc3339(), (row.level_pa * 10.0).round() as i64);
            let entry = winds.entry(key).or_insert((None, None, frame_index));
            let pair = if row.variable == VAR_WIND_U { &mut entry.0 } else { &mut entry.1 };
            *pair = Some(model - row.value);
            for tag in &tags {
                sums.entry((row.variable.clone(), tag.clone())).or_default().add(model - row.value);
            }
            if let (Some(du), Some(dv)) = (entry.0, entry.1) {
                for tag in &tags {
                    sums.entry(("wind_vector_m_s".into(), tag.clone())).or_default().add(du.hypot(dv));
                }
            }
        } else {
            for tag in &tags {
                sums.entry((row.variable.clone(), tag.clone())).or_default().add(model - row.value);
            }
        }
    }
    let scores: Vec<serde_json::Value> = sums
        .iter()
        .map(|((variable, level), s)| json!({"variable": variable, "level_hpa": level, "model": label,
                                            "error": "model minus observation", "stats": s.json()}))
        .collect();
    let document = json!({
        "schema": SCHEMA,
        "method": "nearest mass column, linear in ln(p), no extrapolation; dewpoint by inverse Magnus \
                   (6.112 hPa, 17.67, 243.5 C); winds destaggered and rotated earth-relative \
                   (COSALPHA/SINALPHA); wind_vector_m_s is the magnitude of the vector error",
        "match_seconds": match_seconds, "rim_cells": rim,
        "table": {"path": table.display().to_string(), "sha256": hex_sha256(&std::fs::read(&table)?),
                  "sonde_rows": rows.len(), "rows_not_scored": skipped},
        "frames": frames.iter().map(|f| json!({"path": f.path.display().to_string(), "sha256": f.sha256,
            "valid_time": f.valid.to_rfc3339(), "grid": [f.nz, f.ny, f.nx], "earth_relative_wind": f.wind.is_some()}))
            .collect::<Vec<_>>(),
        "stations_scored": stations_scored,
        "drops": drops,
        "scores": scores,
    });
    std::fs::write(&out, serde_json::to_vec_pretty(&document)?)?;
    println!("{}", serde_json::to_string(&json!({"status": "PASS", "out": out.display().to_string(),
        "stations_scored": document["stations_scored"], "drops": document["drops"]}))?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rw_sonde_score: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dewpoint_matches_the_engine_surface_form() {
        // 14 g/kg at 1000 hPa dews between 18 and 21 C (rustwx-calc's own check).
        let td = dewpoint_k(0.014, 100_000.0) - 273.15;
        assert!(td > 18.0 && td < 21.0, "{td}");
        assert!(dewpoint_k(-1.0e-6, 50_000.0).is_finite());
    }

    #[test]
    fn interpolation_is_linear_in_log_pressure_and_never_extrapolates() {
        let frame = Frame {
            path: PathBuf::new(), sha256: String::new(), valid: Utc::now(), nx: 1, ny: 1, nz: 3,
            lat: vec![0.0], lon: vec![0.0], pressure: vec![100_000.0, 85_000.0, 50_000.0],
            temperature: vec![300.0, 290.0, 260.0], qv: vec![0.0; 3], wind: None,
        };
        let field = frame.temperature.clone();
        assert_eq!(at_pressure(&frame, &field, 0, 0, 85_000.0), Some(290.0));
        let w = (100_000f64.ln() - 92_500f64.ln()) / (100_000f64.ln() - 85_000f64.ln());
        let got = at_pressure(&frame, &field, 0, 0, 92_500.0).unwrap();
        assert!((got - (300.0 * (1.0 - w) + 290.0 * w)).abs() < 1e-9);
        assert_eq!(at_pressure(&frame, &field, 0, 0, 101_000.0), None);
        assert_eq!(at_pressure(&frame, &field, 0, 0, 40_000.0), None);
    }

    #[test]
    fn table_times_parse_in_every_spelling_the_doors_write() {
        for text in ["2026-08-20T00:00:00Z", "2026-08-20 00:00:00", "2026-08-20_00:00:00"] {
            assert_eq!(parse_time(text).unwrap().to_rfc3339(), "2026-08-20T00:00:00+00:00");
        }
    }
}
