//! Station or station-group reference CSV evaluation; weather/data processing
//! stays native. Column mappings and source/site metadata are request data.
use crate::{Result, hash, publish};
use chrono::{DateTime, NaiveDate};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    reference: PathBuf,
    forecast: Vec<PathBuf>,
    rows: PathBuf,
    receipt: PathBuf,
    start_date: NaiveDate,
    end_date: NaiveDate,
    utc_offset_hours: i32,
    date_column: String,
    columns: BTreeMap<String, String>,
    reference_name: String,
    fuel_model: char,
    slope_class: usize,
    provenance: String,
    research_reference_license: String,
    // A station-group time series must never be relabeled as one station.
    aggregation: String,
    latitude: Option<f64>,
    longitude: Option<f64>,
}
pub fn run(path: &Path) -> Result<()> {
    let wall = Instant::now();
    let r: Request = serde_json::from_slice(&fs::read(path)?)?;
    if r.rows.exists() || r.receipt.exists() {
        return Err("preserving existing NFDRS verification artifacts".into());
    }
    if r.reference_name.trim().is_empty()
        || r.provenance.trim().is_empty()
        || r.research_reference_license.trim().is_empty()
        || r.columns.is_empty()
        || r.forecast.is_empty()
        || r.start_date > r.end_date
        || !(-12..=14).contains(&r.utc_offset_hours)
        || !matches!(r.aggregation.as_str(), "regional_mean" | "nearest_station")
    {
        return Err(
            "NFDRS verification requires source/license, dates, columns and explicit aggregation"
                .into(),
        );
    }
    for key in r.columns.keys() {
        if !crate::PRODUCTS.contains(&key.as_str()) {
            return Err(format!("unknown verification product {key}").into());
        }
    }
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(&r.reference)?;
    let header = reader.headers()?.clone();
    let col = |s: &str| {
        header
            .iter()
            .position(|n| n == s)
            .ok_or_else(|| format!("missing reference column {s}"))
    };
    let dc = col(&r.date_column)?;
    let cols: Vec<_> = r
        .columns
        .iter()
        .map(|(p, c)| Ok((p.clone(), col(c)?)))
        .collect::<Result<_>>()?;
    let mut obs = BTreeMap::new();
    for row in reader.records() {
        let row = row?;
        let Some(ds) = row.get(dc).and_then(|s| s.get(..10)) else {
            continue;
        };
        let d = NaiveDate::parse_from_str(ds, "%Y-%m-%d")?;
        if d < r.start_date || d > r.end_date {
            continue;
        }
        let values: Vec<Option<f64>> = cols
            .iter()
            .map(|(_, i)| {
                row.get(*i)
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|v| v.is_finite() && *v >= 0.)
            })
            .collect();
        if obs.insert(d, values).is_some() {
            return Err(format!("ambiguous reference date {d}").into());
        }
    }
    let mut pairs = vec![];
    let mut sources = vec![];
    let mut axes = None;
    let mut days = std::collections::BTreeSet::new();
    for p in &r.forecast {
        let n = netcrust::open(p)?;
        let config = n
            .attribute("nfdrs_configuration")
            .and_then(|v| v.as_string().map(str::to_string))
            .ok_or("forecast is missing NFDRS configuration")?;
        let config: fire_indices::nfdrs::Config = serde_json::from_str(&config)?;
        if config.fuel_model != r.fuel_model || config.slope_class != r.slope_class {
            return Err("reference/forecast NFDRS fuel model or slope class mismatch".into());
        }

        let lat = n.read_f64("latitude")?;
        let lon = n.read_f64("longitude")?;
        if let Some((a, b)) = &axes {
            if a != &lat || b != &lon {
                return Err("NFDRS verification grid changed".into());
            }
        } else {
            axes = Some((lat.clone(), lon.clone()));
        }
        let point = if r.aggregation == "nearest_station" {
            let sy = r.latitude.ok_or("station latitude required")?;
            let sx = r.longitude.ok_or("station longitude required")?;
            if !sy.is_finite()
                || !sx.is_finite()
                || sy < lat.iter().copied().fold(f64::INFINITY, f64::min)
                || sy > lat.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                || sx < lon.iter().copied().fold(f64::INFINITY, f64::min)
                || sx > lon.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            {
                return Err("reference station outside forecast grid".into());
            }
            let near = |a: &[f64], v: f64| {
                a.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| (*a - v).abs().total_cmp(&(*b - v).abs()))
                    .unwrap()
                    .0
            };
            Some(near(&lat, sy) * lon.len() + near(&lon, sx))
        } else {
            None
        };
        let fields: Vec<Vec<f64>> = cols
            .par_iter()
            .map(|(v, _)| n.read_f64(v))
            .collect::<std::result::Result<_, _>>()?;
        let count = lat.len() * lon.len();
        let times = n.read_f64("time")?;
        if fields.iter().any(|f| f.len() != count * times.len()) {
            return Err("bad NFDRS forecast dimensions".into());
        }
        for (ti, t) in times.iter().enumerate() {
            let d = (DateTime::from_timestamp(*t as i64, 0).ok_or("bad time")?
                + chrono::Duration::hours(r.utc_offset_hours as i64))
            .date_naive();
            let Some(o) = obs.get(&d) else { continue };
            if !days.insert(d) {
                return Err("overlapping NFDRS forecasts".into());
            }
            for (j, (product, _)) in cols.iter().enumerate() {
                let Some(o) = o[j] else { continue };
                let f = &fields[j][ti * count..(ti + 1) * count];
                let forecast = if let Some(i) = point {
                    f[i]
                } else {
                    let (sum, area) = f
                        .par_iter()
                        .enumerate()
                        .filter(|(_, v)| v.is_finite())
                        .map(|(i, v)| {
                            let weight = lat[i / lon.len()].to_radians().cos();
                            (v * weight, weight)
                        })
                        .reduce(|| (0., 0.), |(a, b), (c, d)| (a + c, b + d));
                    sum / area
                };
                if forecast.is_finite() {
                    pairs.push((d.to_string(), product.clone(), forecast, o));
                }
            }
        }
        sources.push(json!({"path":p,"sha256":hash(p)?}));
    }
    if pairs.is_empty() {
        return Err("no matching NFDRS reference/forecast values".into());
    }
    let mut csv = csv::Writer::from_writer(Vec::new());
    csv.write_record([
        "date",
        "reference_name",
        "aggregation",
        "product",
        "forecast",
        "station_derived_reference",
        "error",
    ])?;
    let mut metrics = BTreeMap::new();
    for (d, p, f, o) in &pairs {
        csv.write_record([
            d,
            &r.reference_name,
            &r.aggregation,
            p,
            &f.to_string(),
            &o.to_string(),
            &(f - o).to_string(),
        ])?;
    }
    for (p, _) in &cols {
        let rows: Vec<_> = pairs.iter().filter(|(_, q, _, _)| q == p).collect();
        let n = rows.len() as f64;
        if n == 0. {
            continue;
        }
        let f = rows.iter().map(|x| x.2).sum::<f64>();
        let o = rows.iter().map(|x| x.3).sum::<f64>();
        let err = rows.iter().map(|x| x.2 - x.3).sum::<f64>();
        let ff = rows.iter().map(|x| x.2 * x.2).sum::<f64>();
        let oo = rows.iter().map(|x| x.3 * x.3).sum::<f64>();
        let fo = rows.iter().map(|x| x.2 * x.3).sum::<f64>();
        let denom = ((n * ff - f * f) * (n * oo - o * o)).max(0.).sqrt();
        metrics.insert(p,json!({"n":rows.len(),"bias":err/n,"mae":rows.iter().map(|x|(x.2-x.3).abs()).sum::<f64>()/n,"rmse":(rows.iter().map(|x|(x.2-x.3).powi(2)).sum::<f64>()/n).sqrt(),"correlation":if denom>0.{Some((n*fo-f*o)/denom)}else{None}}));
    }
    publish(&r.rows, &csv.into_inner()?)?;
    publish(
        &r.receipt,
        &serde_json::to_vec_pretty(
            &json!({"schema":"arwen.nfdrs-reference-verification.v1","reference":r.reference,"reference_sha256":hash(&r.reference)?,"reference_name":r.reference_name,"fuel_model":r.fuel_model,"slope_class":r.slope_class,"provenance":r.provenance,"research_reference_license":r.research_reference_license,"aggregation":r.aggregation,"station_location":[r.latitude,r.longitude],"forecasts":sources,"rows":r.rows,"rows_sha256":hash(&r.rows)?,"metrics":metrics,"wall_seconds":wall.elapsed().as_secs_f64(),"parallelism":{"forecast_read":rayon::current_num_threads().min(cols.len()),"spatial_reduction":rayon::current_num_threads(),"reference_csv":1},"limitations":["station-weather-derived indices, not measured fire behaviour","regional means and station-group means have different spatial sampling","reference initialization, observation time and independent Nelson discretization affect agreement"]}),
        )?,
    )?;
    Ok(())
}
