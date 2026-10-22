//! Solar product door: native readers, Rayon compute, production map panels.
use crate::{
    panel::{render_panel_bounds, PanelRequest},
    verification_io::{load_arm_at, validate_arm_time, ArmSpec, GridData},
};
use chrono::{DateTime, Datelike, NaiveDate, Timelike, Utc};
use rayon::prelude::*;
use rustwx_calc::solar::{self, Cell, PvSystem};
use rustwx_render::{
    Color, ColorScale, DiscreteColorScale, ExtendMode, LegendControls, RenderDensity,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    out: PathBuf,
    frames: Vec<Frame>,
    #[serde(default)]
    systems: Option<Vec<PvSystem>>,
    #[serde(default)]
    stations: Vec<Station>,
    #[serde(default)]
    render_times: Vec<String>,
    #[serde(default)]
    bounds: Option<(f64, f64, f64, f64)>,
    #[serde(default)]
    products: Vec<String>,
    #[serde(default = "gap")]
    max_gap_seconds: i64,
    #[serde(default = "domain")]
    domain: String,
    #[serde(default = "width")]
    width: u32,
    #[serde(default = "height")]
    height: u32,
}
#[derive(Clone, Deserialize)]
struct Frame {
    #[serde(flatten)]
    spec: ArmSpec,
    #[serde(default)]
    time_index: Option<usize>,
}
/// Expand every record in a native file unless an explicit record is selected.
fn expand_frames(frames: &[Frame]) -> Result<Vec<(ArmSpec, usize)>, String> {
    let mut out = Vec::new();
    for f in frames {
        if f.spec.kind == "netcdf" || f.spec.kind == "wrfout" {
            let nc = netcrust::open(&f.spec.path).map_err(|e| e.to_string())?;
            let times = nc.read_strings("Times").map_err(|e| e.to_string())?;
            let indices: Vec<_> = f
                .time_index
                .map_or_else(|| (0..times.len()).collect(), |i| vec![i]);
            if indices.is_empty() {
                return Err("history file contains no native times".into());
            }
            for i in indices {
                let text = times
                    .get(i)
                    .ok_or("history time_index is out of range")?
                    .replace('_', "T");
                let mut spec = f.spec.clone();
                if let Some(want) = &spec.valid_time {
                    if timestamp(want)? != timestamp(&text)? {
                        return Err(
                            "requested valid_time differs from selected native record".into()
                        );
                    }
                }
                spec.valid_time = Some(text);
                out.push((spec, i));
            }
        } else {
            if f.time_index.is_some_and(|i| i != 0) {
                return Err("time_index applies to native history records".into());
            }
            out.push((f.spec.clone(), 0));
        }
    }
    Ok(out)
}
fn gap() -> i64 {
    3600
}
fn domain() -> String {
    "d01".into()
}
fn width() -> u32 {
    1200
}
fn height() -> u32 {
    900
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Station {
    id: String,
    network: String,
    latitude: f64,
    longitude: f64,
    files: Vec<PathBuf>,
}
#[derive(Clone, Deserialize, Serialize)]
struct ObservationRow {
    id: String,
    url: String,
    header_lines: usize,
    year: usize,
    doy: usize,
    month: usize,
    day: usize,
    hour: usize,
    minute: usize,
    zenith: usize,
    ghi: usize,
    ghi_qc: usize,
    dni: usize,
    dni_qc: usize,
    dhi: usize,
    dhi_qc: usize,
    good_qc: i32,
    missing_below: f64,
    cadence_seconds: i64,
    tolerance_seconds: i64,
    terms: String,
}
#[derive(Clone, Deserialize, Serialize)]
struct Product {
    slug: String,
    quantity: String,
    units: String,
    minimum: f64,
    maximum: f64,
}
fn systems() -> Result<Vec<PvSystem>, String> {
    serde_json::from_str(include_str!("../../rustwx-calc/data/pv_systems.json"))
        .map_err(|e| e.to_string())
}
fn products() -> Result<Vec<Product>, String> {
    serde_json::from_str(include_str!("../data/solar-products.json")).map_err(|e| e.to_string())
}
fn observations() -> Result<Vec<ObservationRow>, String> {
    serde_json::from_str(include_str!("../data/solar-observations.json")).map_err(|e| e.to_string())
}
fn timestamp(text: &str) -> Result<DateTime<Utc>, String> {
    crate::verification::parse_time(text)
}
fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
#[derive(Clone)]
struct Obs {
    time: i64,
    values: [f64; 3],
}
fn read_observations(station: &Station, row: &ObservationRow) -> Result<Vec<Obs>, String> {
    let mut out = Vec::new();
    for path in &station.files {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let header: Vec<_> = text
            .lines()
            .nth(1)
            .ok_or("observation missing station header")?
            .split_whitespace()
            .collect();
        let lat = header
            .first()
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or("bad observation header latitude")?;
        let lon = header
            .get(1)
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or("bad observation header longitude")?;
        if (lat - station.latitude).abs() > 0.05 || (lon - station.longitude).abs() > 0.05 {
            return Err(format!(
                "{} observation header location differs from station row",
                station.id
            ));
        }
        for (line, text) in text.lines().enumerate().skip(row.header_lines) {
            if text.trim().is_empty() {
                continue;
            }
            let v: Vec<f64> = text
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|e| format!("{}:{}: {e}", path.display(), line + 1))?;
            let max = [
                row.year, row.doy, row.month, row.day, row.hour, row.minute, row.zenith, row.ghi,
                row.ghi_qc, row.dni, row.dni_qc, row.dhi, row.dhi_qc,
            ]
            .into_iter()
            .max()
            .unwrap();
            if v.len() <= max {
                return Err(format!(
                    "{}:{}: short radiation record",
                    path.display(),
                    line + 1
                ));
            }
            let date =
                NaiveDate::from_ymd_opt(v[row.year] as i32, v[row.month] as u32, v[row.day] as u32)
                    .ok_or("invalid observation date")?;
            if date.ordinal() != v[row.doy] as u32 {
                return Err("observation day-of-year differs from calendar date".into());
            }
            let time = date
                .and_hms_opt(v[row.hour] as u32, v[row.minute] as u32, 0)
                .ok_or("invalid observation time")?
                .and_utc()
                .timestamp();
            let mut values = [f64::NAN; 3];
            for (i, (field, qc)) in [
                (row.ghi, row.ghi_qc),
                (row.dni, row.dni_qc),
                (row.dhi, row.dhi_qc),
            ]
            .into_iter()
            .enumerate()
            {
                if v[qc] == f64::from(row.good_qc)
                    && v[field].is_finite()
                    && v[field] > row.missing_below
                {
                    // NOAA thermopile nighttime offsets can be negative with good QC.
                    // Preserve the measured value; clipping would bias verification.
                    values[i] = v[field];
                }
            }
            out.push(Obs { time, values });
        }
    }
    out.sort_by_key(|o| o.time);
    if out.windows(2).any(|p| p[0].time == p[1].time) {
        return Err("duplicate observation timestamps".into());
    }
    Ok(out)
}
fn matched(obs: &[Obs], time: i64, tolerance: i64) -> [f64; 3] {
    let mut sum = [0.0; 3];
    let mut count = [0; 3];
    let lo = obs.partition_point(|o| o.time < time - tolerance);
    let hi = obs.partition_point(|o| o.time <= time + tolerance);
    for o in &obs[lo..hi] {
        for i in 0..3 {
            if o.values[i].is_finite() {
                sum[i] += o.values[i];
                count[i] += 1;
            }
        }
    }
    std::array::from_fn(|i| {
        if count[i] > 0 {
            sum[i] / f64::from(count[i])
        } else {
            f64::NAN
        }
    })
}
#[derive(Default)]
struct Metric {
    n: usize,
    bias: f64,
    abs: f64,
    squared: f64,
}
impl Metric {
    fn add(&mut self, a: f64, b: f64) {
        if a.is_finite() && b.is_finite() {
            let e = a - b;
            self.n += 1;
            self.bias += e;
            self.abs += e.abs();
            self.squared += e * e;
        }
    }
    fn value(&self) -> Value {
        if self.n == 0 {
            json!({"pairs":0})
        } else {
            json!({"pairs":self.n,"bias":self.bias/self.n as f64,"mae":self.abs/self.n as f64,"rmse":(self.squared/self.n as f64).sqrt()})
        }
    }
}
struct Daily {
    start: i64,
    end: i64,
    seconds: i64,
    ghi: Vec<f32>,
    ac: Vec<Vec<f32>>,
}
fn plane(cells: &[Cell], q: &str, system: usize) -> Vec<f32> {
    cells
        .par_iter()
        .map(|c| match q {
            "ghi" => c.ghi,
            "dni" => c.dni,
            "dhi" => c.dhi,
            "clear" => c.clear,
            "index" => c.index,
            "poa" => c.poa[system],
            "dc" => c.dc[system],
            "ac" => c.ac[system],
            _ => f32::NAN,
        })
        .collect()
}
fn scale(min: f64, max: f64) -> ColorScale {
    let colors = [
        [30, 35, 70],
        [35, 85, 145],
        [30, 145, 170],
        [80, 185, 140],
        [175, 210, 100],
        [240, 215, 70],
        [250, 165, 45],
        [235, 90, 35],
        [170, 30, 45],
        [100, 0, 60],
    ];
    ColorScale::Discrete(DiscreteColorScale {
        levels: (0..=10)
            .map(|i| min + (max - min) * f64::from(i) / 10.0)
            .collect(),
        colors: colors
            .iter()
            .map(|c| Color::rgba(c[0], c[1], c[2], 255))
            .collect(),
        extend: ExtendMode::Both,
        mask_below: None,
    })
}
fn draw(
    r: &Request,
    grid: &GridData,
    slug: &str,
    row: &Product,
    values: Vec<f32>,
    time: &str,
    note: &str,
    system: Option<&PvSystem>,
) -> Result<PathBuf, String> {
    let dt = timestamp(time)?;
    let out = r
        .out
        .join(&r.domain)
        .join(slug)
        .join(dt.format("%Y%m%d").to_string())
        .join(format!("{}.png", dt.format("%Y%m%dT%H%M%SZ")));
    fs::create_dir_all(out.parent().unwrap()).map_err(|e| e.to_string())?;
    let factor = if row.units == "MW" || row.units == "MW/min" || row.units == "MWh" {
        system.map_or(1.0, |s| s.dc_mw)
    } else {
        1.0
    };
    render_panel_bounds(
        PanelRequest {
            lat_deg: &grid.lat,
            lon_deg: &grid.lon,
            projection: grid.projection.as_ref(),
            ny: grid.ny,
            nx: grid.nx,
            values,
            product_slug: slug.into(),
            title: slug.replace('_', " "),
            display_units: row.units.clone(),
            scale: scale(row.minimum * factor, row.maximum * factor),
            cbar_tick_step: None,
            legend: LegendControls::default(),
            render_density: RenderDensity::default(),
            subtitle_left: format!("Valid {time}"),
            subtitle_center: None,
            subtitle_right: note.into(),
            width: r.width,
            height: r.height,
            contours: vec![],
            colorbar: true,
            overlays: None,
            annotations: None,
            out_path: out.clone(),
        },
        r.bounds,
    )?;
    println!("RENDERED\t{}", out.display());
    Ok(out)
}
fn stats(values: &[f32]) -> Value {
    let (n, sum, min, max) = values
        .par_iter()
        .filter(|v| v.is_finite())
        .map(|&v| (1_u64, f64::from(v), v, v))
        .reduce(
            || (0, 0.0, f32::INFINITY, f32::NEG_INFINITY),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2.min(b.2), a.3.max(b.3)),
        );
    json!({"finite":n,"missing":values.len()as u64-n,"mean":if n>0{Some(sum/n as f64)}else{None},"minimum":if n>0{Some(min)}else{None},"maximum":if n>0{Some(max)}else{None}})
}
fn run(path: &Path) -> Result<(), String> {
    let whole = Instant::now();
    let r: Request = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if r.frames.is_empty()
        || r.max_gap_seconds <= 0
        || r.width < 200
        || r.height < 200
        || r.domain.is_empty()
        || !r
            .domain
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    {
        return Err("solar needs frames, positive max gap, valid domain and canvas >=200".into());
    }
    if let Some((w, e, s, n)) = r.bounds {
        if ![w, e, s, n].iter().all(|v| v.is_finite()) || w >= e || s >= n || s < -90.0 || n > 90.0
        {
            return Err("invalid map bounds (west,east,south,north)".into());
        }
    }
    let frames = expand_frames(&r.frames)?;
    let systems = r.systems.clone().unwrap_or(systems()?);
    if systems.is_empty() {
        return Err("no PV systems".into());
    }
    for s in &systems {
        s.validate()?;
    }
    let mut ids = std::collections::BTreeSet::new();
    for s in &systems {
        if !ids.insert(s.id.clone()) {
            return Err("duplicate PV system id".into());
        }
    }
    let product_rows = products()?;
    let obs_rows = observations()?;
    let mut station_obs = Vec::new();
    for station in &r.stations {
        let row = obs_rows
            .iter()
            .find(|o| o.id == station.network)
            .ok_or("unknown radiation observation network")?;
        station_obs.push((read_observations(station, row)?, row.clone()));
    }
    let mut selected = std::collections::BTreeSet::new();
    for row in &product_rows {
        if row.slug.contains("{system}") {
            for s in &systems {
                selected.insert(row.slug.replace("{system}", &s.id));
            }
        } else {
            selected.insert(row.slug.clone());
        }
    }
    for slug in &r.products {
        if !selected.contains(slug) {
            return Err(format!("unknown solar product {slug}"));
        }
    }
    let wanted = |slug: &str| r.products.is_empty() || r.products.iter().any(|p| p == slug);
    fs::create_dir_all(&r.out).map_err(|e| e.to_string())?;
    let mut source_receipts = Vec::new();
    let mut frame_receipts = Vec::new();
    let mut daily: BTreeMap<i64, Daily> = BTreeMap::new();
    let mut previous: Option<(i64, Vec<Cell>)> = None;
    let mut geometry: Option<GridData> = None;
    let mut station_indices = Vec::new();
    let mut csv=String::from("station,valid_time,split,model_ghi,observed_ghi,model_dni,observed_dni,model_dhi,observed_dhi,model_ghi_ramp_wm2_min,observed_ghi_ramp_wm2_min\n");
    let mut metrics: BTreeMap<String, Metric> = BTreeMap::new();
    let mut last_station: Vec<Option<(i64, f64, f64)>> = vec![None; r.stations.len()];
    let mut maps = Vec::new();
    let (mut read_seconds, mut compute_seconds, mut render_seconds) = (0.0, 0.0, 0.0);
    for (spec, time_index) in &frames {
        let stage = Instant::now();
        let arm = load_arm_at(spec, *time_index)?;
        let grid = arm.grid.as_ref().ok_or("solar input has no native grid")?;
        let text = spec
            .valid_time
            .as_deref()
            .or_else(|| arm.provenance["metadata"]["valid_time"].as_str())
            .ok_or("solar frame missing valid time")?;
        let time = timestamp(text)?;
        validate_arm_time(spec, &arm, text)?;
        if previous.as_ref().is_some_and(|p| time.timestamp() <= p.0) {
            return Err("solar frames must have strictly increasing native times".into());
        }
        if let Some(old) = &geometry {
            if old.nx != grid.nx || old.ny != grid.ny || old.lat != grid.lat || old.lon != grid.lon
            {
                return Err("solar temporal products need a fixed Earth grid; regrid moving output before integration".into());
            }
        } else {
            for s in &r.stations {
                grid.position(s.latitude, s.longitude)
                    .ok_or_else(|| format!("station {} lies outside the source grid", s.id))?;
                let index = (0..grid.lat.len())
                    .into_par_iter()
                    .min_by(|&a, &b| {
                        let d = |i: usize| {
                            (f64::from(grid.lat[i]) - s.latitude).powi(2)
                                + ((f64::from(grid.lon[i]) - s.longitude + 180.0).rem_euclid(360.0)
                                    - 180.0)
                                    .powi(2)
                                    * s.latitude.to_radians().cos().powi(2)
                        };
                        d(a).total_cmp(&d(b))
                    })
                    .ok_or("empty grid")?;
                station_indices.push(index);
            }
            geometry = Some(grid.clone());
        }
        let field = |q: &str| {
            arm.fields
                .get(q)
                .ok_or_else(|| format!("{} missing required field {q}", spec.path.display()))
        };
        let ghi = field("solar_ghi")?;
        let air = field("temperature_2m")?;
        let wind = field("wind_speed_10m")?;
        let dni = arm.fields.get("solar_dni");
        let dhi = arm.fields.get("solar_dhi");
        if dni.is_some() != dhi.is_some() {
            return Err("native SWDDNI and SWDDIF must both decode; refusing a mixed native/decomposed split".into());
        }
        if spec.kind == "netcdf" || spec.kind == "wrfout" {
            let nc = netcrust::open(&spec.path).map_err(|e| e.to_string())?;
            for q in ["solar_dni", "solar_dhi"] {
                if spec
                    .fields
                    .get(q)
                    .is_some_and(|name| nc.variable(name).is_some())
                    && !arm.fields.contains_key(q)
                {
                    return Err(format!("present native {q} failed to decode"));
                }
            }
        }
        let split = if dni.is_some() {
            "native_SWDDNI_SWDDIF"
        } else {
            "Erbs_hourly"
        };
        read_seconds += stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let leap = NaiveDate::from_ymd_opt(time.year(), 2, 29).is_some();
        let cells = solar::compute(
            solar::Inputs {
                lat: &grid.lat,
                lon: &grid.lon,
                ghi,
                dni: dni.map(Vec::as_slice),
                dhi: dhi.map(Vec::as_slice),
                air_k: air,
                wind_ms: wind,
                day: time.ordinal(),
                days_in_year: if leap { 366 } else { 365 },
                utc_seconds: f64::from(time.num_seconds_from_midnight()),
            },
            &systems,
        )?;
        let mut ramps = vec![vec![f32::NAN; cells.len()]; systems.len()];
        if let Some((t0, old)) = &previous {
            for i in 0..systems.len() {
                ramps[i] = solar::ramp(
                    &plane(old, "ac", i),
                    &plane(&cells, "ac", i),
                    time.timestamp() - t0,
                    r.max_gap_seconds,
                )?;
            }
            let mut start = *t0;
            let old_ghi = plane(old, "ghi", 0);
            let current_ghi = plane(&cells, "ghi", 0);
            let old_ac: Vec<_> = (0..systems.len()).map(|i| plane(old, "ac", i)).collect();
            let current_ac: Vec<_> = (0..systems.len()).map(|i| plane(&cells, "ac", i)).collect();
            while start < time.timestamp() {
                let day = start.div_euclid(86400) * 86400;
                let end = time.timestamp().min(day + 86400);
                let d = daily.entry(day).or_insert_with(|| Daily {
                    start,
                    end: start,
                    seconds: 0,
                    ghi: vec![0.0; cells.len()],
                    ac: vec![vec![0.0; cells.len()]; systems.len()],
                });
                solar::integrate(
                    &mut d.ghi,
                    &old_ghi,
                    &current_ghi,
                    *t0,
                    time.timestamp(),
                    start,
                    end,
                    r.max_gap_seconds,
                    0.001,
                )?;
                for i in 0..systems.len() {
                    solar::integrate(
                        &mut d.ac[i],
                        &old_ac[i],
                        &current_ac[i],
                        *t0,
                        time.timestamp(),
                        start,
                        end,
                        r.max_gap_seconds,
                        1.0,
                    )?;
                }
                d.end = end;
                if time.timestamp() - t0 <= r.max_gap_seconds {
                    d.seconds += end - start;
                }
                start = end;
            }
        }
        for (i, station) in r.stations.iter().enumerate() {
            let (obs, row) = &station_obs[i];
            let observed = matched(obs, time.timestamp(), row.tolerance_seconds);
            let c = &cells[station_indices[i]];
            let model = [f64::from(c.ghi), f64::from(c.dni), f64::from(c.dhi)];
            let (mut mr, mut or) = (f64::NAN, f64::NAN);
            if let Some((old, mg, og)) = last_station[i] {
                let dt = time.timestamp() - old;
                if dt <= r.max_gap_seconds {
                    mr = (model[0] - mg) * 60.0 / dt as f64;
                    or = (observed[0] - og) * 60.0 / dt as f64;
                }
            }
            last_station[i] = Some((time.timestamp(), model[0], observed[0]));
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{}\n",
                station.id,
                time.to_rfc3339(),
                split,
                model[0],
                observed[0],
                model[1],
                observed[1],
                model[2],
                observed[2],
                mr,
                or
            ));
            for (j, q) in ["ghi", "dni", "dhi"].iter().enumerate() {
                metrics
                    .entry(format!("{}/{q}", station.id))
                    .or_default()
                    .add(model[j], observed[j]);
                if c.clear >= 20.0 {
                    metrics
                        .entry(format!("{}/{q}_daylight", station.id))
                        .or_default()
                        .add(model[j], observed[j]);
                }
            }
            metrics
                .entry(format!("{}/ghi_ramp_wm2_min", station.id))
                .or_default()
                .add(mr, or);
        }
        compute_seconds += stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let mut summary = BTreeMap::new();
        let render = r.render_times.is_empty()
            || r.render_times
                .iter()
                .any(|t| timestamp(t).is_ok_and(|t| t == time));
        for row in product_rows
            .iter()
            .filter(|p| !p.quantity.starts_with("daily"))
        {
            let indices: Vec<_> = if row.slug.contains("{system}") {
                (0..systems.len()).collect()
            } else {
                vec![0]
            };
            for i in indices {
                let slug = row.slug.replace("{system}", &systems[i].id);
                let values = match row.quantity.as_str() {
                    "ramp" => ramps[i].clone(),
                    "event" => ramps[i]
                        .par_iter()
                        .map(|v| {
                            if v.is_finite() {
                                if f64::from(v.abs())
                                    >= systems[i].ac_rating_mw()
                                        * systems[i].ramp_fraction_per_minute
                                {
                                    1.0
                                } else {
                                    0.0
                                }
                            } else {
                                f32::NAN
                            }
                        })
                        .collect(),
                    q => plane(&cells, q, i),
                };
                summary.insert(slug.clone(), stats(&values));
                if render && wanted(&slug) {
                    maps.push(draw(
                        &r,
                        grid,
                        &slug,
                        row,
                        values,
                        &time.to_rfc3339(),
                        split,
                        Some(&systems[i]),
                    )?);
                }
            }
        }
        render_seconds += stage.elapsed().as_secs_f64();
        frame_receipts
            .push(json!({"valid_time":time.to_rfc3339(),"split":split,"statistics":summary}));
        source_receipts.push(arm.provenance);
        println!(
            "SOLAR_FRAME\t{}\t{}x{}\t{}",
            time.to_rfc3339(),
            grid.nx,
            grid.ny,
            split
        );
        previous = Some((time.timestamp(), cells));
    }
    let stage = Instant::now();
    let grid = geometry.ok_or("no solar grid")?;
    let mut day_receipts = Vec::new();
    for (day, mut d) in daily {
        let complete = d.start == day && d.end == day + 86400 && d.seconds == 86400;
        if !complete {
            d.ghi.par_iter_mut().for_each(|v| *v = f32::NAN);
            for a in &mut d.ac {
                a.par_iter_mut().for_each(|v| *v = f32::NAN);
            }
        }
        let valid = DateTime::<Utc>::from_timestamp(day, 0)
            .ok_or("daily time overflow")?
            .to_rfc3339();
        let mut summary = BTreeMap::new();
        for row in product_rows
            .iter()
            .filter(|p| p.quantity.starts_with("daily"))
        {
            let indices: Vec<_> = if row.slug.contains("{system}") {
                (0..systems.len()).collect()
            } else {
                vec![0]
            };
            for i in indices {
                let slug = row.slug.replace("{system}", &systems[i].id);
                let values = if row.quantity == "daily_ghi" {
                    d.ghi.clone()
                } else {
                    d.ac[i].clone()
                };
                summary.insert(slug.clone(), stats(&values));
                if wanted(&slug) && complete {
                    maps.push(draw(
                        &r,
                        &grid,
                        &slug,
                        row,
                        values,
                        &valid,
                        "UTC daily trapezoidal energy",
                        Some(&systems[i]),
                    )?);
                }
            }
        }
        day_receipts.push(json!({"date":valid,"complete":complete,"covered_seconds":d.seconds,"statistics":summary}));
    }
    render_seconds += stage.elapsed().as_secs_f64();
    fs::write(r.out.join("verification.csv"), csv).map_err(|e| e.to_string())?;
    let station_meta:Vec<_>=r.stations.iter().zip(station_indices).map(|(s,i)|json!({"id":s.id,"network":s.network,"latitude":s.latitude,"longitude":s.longitude,"cell":i,"grid_latitude":grid.lat[i],"grid_longitude":grid.lon[i],"files":s.files})).collect();
    let metrics: BTreeMap<_, _> = metrics.iter().map(|(k, m)| (k, m.value())).collect();
    write_json(
        &r.out.join("receipt.json"),
        &json!({"schema":"gpuwm.solar-products.v1","command":std::env::args().collect::<Vec<_>>(),"grid":[grid.ny,grid.nx],"frames":frame_receipts,"sources":source_receipts,"systems":systems,"stations":station_meta,"verification":metrics,"observation_matching":"mean of QC=0 one-minute end-time values within table tolerance; nearest native model cell; instantaneous forecast compared to short observation mean","daily":day_receipts,"maps":maps,"stages_seconds":{"read":read_seconds,"compute":compute_seconds,"render":render_seconds,"whole":whole.elapsed().as_secs_f64()},"parallelism":{"compute":"Rayon grid cells","threads":rayon::current_num_threads(),"render":"Rayon raster","read":"native per-frame readers"},"peak_vram_mib":0,"gpu_compute":false,"methods":{"decomposition":"native pair when present; otherwise Erbs hourly","poa":"isotropic sky and ground; monofacial, no external shading","temperature":"Faiman, module-height wind power law exponent 0.2","power":"PVWatts v5 DC and inverter equations; not the full PVWatts Fuentes/Perez model","clear_sky":"Haurwitz empirical GHI; index missing below 20 W/m2 clear sky","ramp":"signed endpoint MW/min; table AC fraction/min threshold; gaps missing","energy":"UTC trapezoids, midnight splitting, full 86400-second coverage required"}}),
    )?;
    Ok(())
}
pub fn cli(args: &[String]) -> Option<Result<(), String>> {
    match args.first()?.as_str() {
        "--solar-catalog" => Some((|| {
            println!(
                "{}",
                json!({"systems":systems()?,"products":products()?,"observations":observations()?})
            );
            Ok(())
        })()),
        "--solar-products" => Some(if args.len() == 2 {
            run(Path::new(&args[1]))
        } else {
            Err("usage: rw_wrfbatch --solar-products REQUEST.json".into())
        }),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observations_preserve_good_qc_offsets_and_reject_missing_or_bad_qc() {
        let path =
            std::env::temp_dir().join(format!("solar-observations-{}.dat", std::process::id()));
        fs::write(
            &path,
            "station\n40.0 -88.0\n2030 90 3 31 0 0 0 100 -6.0 0 0 0 -9999.9 0 50.0 1\n",
        )
        .unwrap();
        let station = Station {
            id: "test".into(),
            network: "surfrad".into(),
            latitude: 40.0,
            longitude: -88.0,
            files: vec![path.clone()],
        };
        let obs = read_observations(&station, &observations().unwrap()[0]).unwrap();
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].values[0], -6.0);
        assert!(obs[0].values[1].is_nan());
        assert!(obs[0].values[2].is_nan());
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn multi_record_history_selects_native_flux_and_time() {
        use netcdf_writer::{NcFormat, NcType, NcWriter, Schema, VarData};
        let path = std::env::temp_dir().join(format!("solar-records-{}.nc", std::process::id()));
        let mut schema = Schema::new(NcFormat::Offset64);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let date = schema.def_dim("DateStrLen", 19, false).unwrap();
        let y = schema.def_dim("south_north", 2, false).unwrap();
        let x = schema.def_dim("west_east", 2, false).unwrap();
        let times = schema
            .def_var("Times", NcType::Char, &[time, date])
            .unwrap();
        let keys = [
            "XLAT", "XLONG", "SWDOWN", "SWDDNI", "SWDDIF", "T2", "U10", "V10",
        ];
        let vars: Vec<_> = keys
            .iter()
            .map(|k| schema.def_var(k, NcType::Float, &[time, y, x]).unwrap())
            .collect();
        let mut w = NcWriter::create(&path, schema).unwrap();
        for (i, text) in [b"2030-03-31_12:00:00", b"2030-03-31_13:00:00"]
            .iter()
            .enumerate()
        {
            w.write_record(i as u64, times, VarData::Char(*text))
                .unwrap();
            for (j, value) in [
                40.0,
                -88.0,
                100.0 + i as f32 * 300.0,
                200.0,
                50.0,
                290.0,
                3.0,
                4.0,
            ]
            .iter()
            .enumerate()
            {
                w.write_record(i as u64, vars[j], VarData::F32(&[*value; 4]))
                    .unwrap();
            }
        }
        w.finish().unwrap();
        let frame:Frame=serde_json::from_value(json!({"label":"test","kind":"wrfout","path":path,"fields":{"solar_ghi":"SWDOWN","solar_dni":"SWDDNI","solar_dhi":"SWDDIF","temperature_2m":"T2","wind_speed_10m":"WSPD10","dewpoint_2m":"unused"}})).unwrap();
        let expanded = expand_frames(&[frame]).unwrap();
        assert_eq!(expanded.len(), 2);
        let arm = load_arm_at(&expanded[1].0, expanded[1].1).unwrap();
        assert_eq!(arm.fields["solar_ghi"], [400.0; 4]);
        assert_eq!(arm.fields["solar_dni"], [200.0; 4]);
        assert_eq!(arm.fields["wind_speed_10m"], [5.0; 4]);
        validate_arm_time(&expanded[1].0, &arm, "2030-03-31T13:00:00").unwrap();
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn observation_rows_have_distinct_direct_columns() {
        let rows = observations().unwrap();
        assert_eq!(rows[0].dni, 12);
        assert_eq!(rows[1].dni, 10);
        assert_eq!(rows[0].good_qc, 0);
    }
    #[test]
    fn missing_and_qc_are_not_zero() {
        let o = vec![
            Obs {
                time: 60,
                values: [100.0, f64::NAN, 50.0],
            },
            Obs {
                time: 120,
                values: [200.0, 300.0, f64::NAN],
            },
        ];
        assert_eq!(matched(&o, 90, 30), [150.0, 300.0, 50.0]);
        assert!(matched(&o, 1000, 30)[0].is_nan());
    }
    #[test]
    fn product_rows_and_systems_are_reachable() {
        for s in systems().unwrap() {
            s.validate().unwrap();
        }
        let p = products().unwrap();
        assert!(p.iter().any(|r| r.quantity == "daily_ac"));
        assert!(p.iter().all(|r| r.minimum < r.maximum));
    }
}
