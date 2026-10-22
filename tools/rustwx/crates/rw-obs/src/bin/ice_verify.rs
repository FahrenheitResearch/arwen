//! Bind native saved-frame ice products to surface precipitation-type reports.
//! No radial-ice validation is inferred from a precipitation-type report.
use chrono::{DateTime, NaiveDateTime};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    io::Write,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
fn fail(s: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    s.into().into()
}
fn time(s: &str) -> Result<i64> {
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Ok(t.timestamp());
    }
    for f in [
        "%Y-%m-%d_%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ] {
        if let Ok(t) = NaiveDateTime::parse_from_str(s, f) {
            return Ok(t.and_utc().timestamp());
        }
    }
    Err(fail(format!("invalid observation/product UTC time {s:?}")))
}
fn reports_freezing_rain(codes: &str, metar: &str) -> bool {
    // FZRANO is a failed sensor, not a positive weather report. Remarks
    // describe earlier conditions and cannot establish this report's type.
    codes
        .split_whitespace()
        .chain(metar.split(" RMK ").next().unwrap_or("").split_whitespace())
        .any(|v| v.contains("FZRA") && !v.contains("FZRANO") && !v.starts_with("RE"))
}
#[derive(Clone)]
struct Observation {
    source: String,
    id: String,
    at: i64,
    lat: f64,
    lon: f64,
    fzra: bool,
    cold: Option<bool>,
    ice: Option<f64>,
}
fn asos(path: &Path) -> Result<Vec<Observation>> {
    let rows = rw_obs::csv::read(path).map_err(fail)?;
    let h = rows.first().ok_or_else(|| fail("ASOS CSV is empty"))?;
    let col = |n: &str| {
        h.iter().position(|v|v==n).ok_or_else(||fail(format!("ASOS CSV needs {n}; request station, valid, lat, lon, tmpf, wxcodes, metar from IEM")))
    };
    let (station, valid, lat, lon) = (col("station")?, col("valid")?, col("lat")?, col("lon")?);
    let wx = h.iter().position(|v| v == "wxcodes");
    let metar = h.iter().position(|v| v == "metar");
    if wx.is_none() && metar.is_none() {
        return Err(fail(
            "ASOS CSV has no weather codes or METAR; absence of a type field cannot mean no freezing rain",
        ));
    }
    let tmp = h.iter().position(|v| v == "tmpf");
    let ice = h.iter().position(|v| v == "ice_accretion_1hr");
    let mut out = Vec::new();
    for r in rows
        .iter()
        .skip(1)
        .filter(|r| !r.iter().all(|s| s.is_empty()))
    {
        if r.len() != h.len() {
            return Err(fail("ASOS CSV row width differs from its header"));
        }
        let get = |c: Option<usize>| c.map(|i| r[i].trim()).unwrap_or("");
        let clean = |c| {
            let v = get(c);
            if v == "M" { "" } else { v }
        };
        let code = format!("{} {}", clean(wx), clean(metar));
        if code.trim().is_empty() || code.trim() == "M" {
            continue;
        }
        let number = |c: Option<usize>| get(c).parse::<f64>().ok().filter(|v| v.is_finite());
        let (y, x) = (r[lat].parse::<f64>()?, r[lon].parse::<f64>()?);
        out.push(Observation {
            source: "ASOS".into(),
            id: r[station].clone(),
            at: time(&r[valid])?,
            lat: y,
            lon: x,
            fzra: reports_freezing_rain(clean(wx), clean(metar)),
            cold: number(tmp).map(|v| v < 32.0),
            ice: number(ice).filter(|v| *v >= 0.0).map(|v| v * 25.4),
        });
    }
    Ok(out)
}
fn mping(path: &Path) -> Result<Vec<Observation>> {
    let doc: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let rows = doc
        .as_array()
        .or_else(|| doc.get("results").and_then(Value::as_array))
        .ok_or_else(|| fail("mPING export must be a report array or API results array"))?;
    let mut out = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let label=r.get("description").and_then(|v|v.as_str().or_else(||v.get("name").and_then(Value::as_str))).ok_or_else(||fail("mPING report has no descriptive label; numeric type codes need an authoritative label table"))?.to_lowercase();
        let is_precip = ["rain", "drizzle", "snow", "ice pellets", "sleet", "none"]
            .iter()
            .any(|w| label.contains(w));
        if !is_precip {
            continue;
        }
        let point = r
            .get("geometry")
            .or_else(|| r.get("geom"))
            .and_then(|v| v.get("coordinates"))
            .and_then(Value::as_array)
            .ok_or_else(|| fail("mPING report lacks GeoJSON coordinates"))?;
        if point.len() != 2 {
            return Err(fail("mPING point needs longitude,latitude"));
        }
        let at = r
            .get("observation_time")
            .or_else(|| r.get("obtime"))
            .and_then(Value::as_str)
            .ok_or_else(|| fail("mPING report lacks observation_time/obtime"))?;
        out.push(Observation {
            source: "mPING".into(),
            id: r
                .get("id")
                .map(Value::to_string)
                .unwrap_or_else(|| format!("report-{i}")),
            at: time(at)?,
            lat: point[1]
                .as_f64()
                .ok_or_else(|| fail("mPING latitude is not numeric"))?,
            lon: point[0]
                .as_f64()
                .ok_or_else(|| fail("mPING longitude is not numeric"))?,
            fzra: label.contains("freezing rain"),
            cold: None,
            ice: None,
        });
    }
    Ok(out)
}
struct Frame {
    path: PathBuf,
    sha: String,
    at: i64,
    lat: Vec<f64>,
    lon: Vec<f64>,
    rate: Vec<f64>,
    cold: Vec<f64>,
    flat: Vec<f64>,
    start: i64,
}
fn read(path: &Path) -> Result<Frame> {
    let f = netcrust::open(path)?;
    let attr = |name: &str| {
        f.attribute(name)
            .and_then(|a| a.as_string().map(str::to_string))
            .ok_or_else(|| fail(format!("{} missing {name}", path.display())))
    };
    if attr("ensemble_contract")? != "gpuwm-resident-ensemble-products.v1" {
        return Err(fail(
            "ice verification needs the native ensemble product contract",
        ));
    }
    let sampling: Value = serde_json::from_str(&attr("ice_accretion_sampling")?)?;
    if sampling["method"] != "saved-frame-right-endpoint" {
        return Err(fail(
            "ice verification does not recognize this accumulation sampling method",
        ));
    }
    let intervals = sampling["intervals_hours"]
        .as_array()
        .ok_or_else(|| fail("ice product lacks sampled intervals"))?;
    let hours = intervals
        .iter()
        .filter_map(Value::as_f64)
        .fold(0.0_f64, f64::max);
    let at = time(&attr("valid_time")?)?;
    let field = |name: &str| -> Result<Vec<f64>> { Ok(f.read_array_f64(name)?.values().to_vec()) };
    let coordinate = f.variable("XLAT").ok_or_else(|| fail("missing XLAT"))?;
    let shape = coordinate.shape();
    if shape.len() != 2 || shape.contains(&0) {
        return Err(fail(
            "native ice coordinates require a nonempty two-dimensional grid",
        ));
    }
    let check_geometry = |name: &str, threshold: Option<usize>| -> Result<()> {
        let v = f
            .variable(name)
            .ok_or_else(|| fail(format!("missing {name}")))?;
        let names: Vec<_> = v.dimensions().iter().map(|d| d.name()).collect();
        let threshold_name = name
            .strip_suffix("_probability")
            .map(|n| format!("{n}_threshold"));
        let (dims, expected) = if let Some(n) = threshold {
            (
                vec![
                    threshold_name.as_deref().unwrap(),
                    "south_north",
                    "west_east",
                ],
                vec![n, shape[0], shape[1]],
            )
        } else {
            (vec!["south_north", "west_east"], shape.to_vec())
        };
        if names != dims || v.shape() != expected {
            return Err(fail(format!(
                "{name} has dimensions {names:?} and shape {:?}; expected {dims:?} {expected:?}; transposition would bind reports to wrong points",
                v.shape()
            )));
        }
        Ok(())
    };
    check_geometry("XLAT", None)?;
    check_geometry("XLONG", None)?;
    check_geometry("ice_accretion_horizontal_mean", None)?;
    let lat = field("XLAT")?;
    let lon = field("XLONG")?;
    if lat.is_empty()
        || lat
            .iter()
            .zip(&lon)
            .any(|(y, x)| !y.is_finite() || !x.is_finite() || y.abs() > 90.0 || x.abs() > 180.0)
    {
        return Err(fail(
            "native ice coordinates must contain finite valid latitude/longitude points",
        ));
    }
    let rate = field("freezing_rain_rate_probability")?;
    let cold = field("temperature2_below_freezing_probability")?;
    let flat = field("ice_accretion_horizontal_mean")?;
    if rate
        .iter()
        .chain(&cold)
        .any(|p| p.is_finite() && !(0.0..=1.0).contains(p))
    {
        return Err(fail("native ice event probabilities must lie in [0,1]"));
    }
    for name in [
        "freezing_rain_rate_probability",
        "temperature2_below_freezing_probability",
    ] {
        let v = f
            .variable(name)
            .ok_or_else(|| fail("missing probability"))?;
        let wanted = if name.starts_with("temperature") {
            "lt"
        } else {
            "ge"
        };
        if v.attribute("comparison").and_then(|a| a.as_string()) != Some(wanted) {
            return Err(fail(format!(
                "{name} does not use {wanted}; comparison reversal would invert the event"
            )));
        }
    }
    let thresholds = field("freezing_rain_rate_thresholds")?;
    if thresholds.first().is_none_or(|v| (*v - 0.1).abs() > 1e-6) {
        return Err(fail(
            "freezing-rain verification needs the first rate threshold to be 0.1 mm/h",
        ));
    }
    let t = field("temperature2_below_freezing_thresholds")?;
    if t.len() != 1 || (t[0] - 273.15).abs() > 1e-4 {
        return Err(fail(
            "below-freezing verification requires one 273.15 K threshold",
        ));
    }
    check_geometry("freezing_rain_rate_probability", Some(thresholds.len()))?;
    check_geometry("temperature2_below_freezing_probability", Some(1))?;
    if lon.len() != lat.len()
        || cold.len() != lat.len()
        || flat.len() != lat.len()
        || rate.len() != lat.len() * thresholds.len()
    {
        return Err(fail(
            "ice verification arrays have inconsistent native geometry",
        ));
    }
    Ok(Frame {
        path: path.into(),
        sha: rw_obs::hex_sha256(&std::fs::read(path)?),
        at,
        lat,
        lon,
        rate,
        cold,
        flat,
        start: at - (hours * 3600.0).round() as i64,
    })
}
fn distance(y: f64, x: f64, a: f64, b: f64) -> f64 {
    let dy = (a - y).to_radians();
    let dx = (b - x).to_radians();
    let h = (dy / 2.0).sin().powi(2)
        + y.to_radians().cos() * a.to_radians().cos() * (dx / 2.0).sin().powi(2);
    12742.0 * h.clamp(0.0, 1.0).sqrt().asin()
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--help" {
        println!(
            "rw_ice_verify REQUEST.json OUTDIR\nREQUEST: frames [native NetCDF paths], asos_csv (optional), mping_json (optional), max_distance_km (default 15). Type reports bind to saved-frame intervals; ASOS 1 h horizontal ice binds only to hourly increments. Missing mPING yields PARTIAL. No type report validates radial magnitude."
        );
        return Ok(());
    }
    if args.len() != 3 {
        return Err(fail("usage: rw_ice_verify REQUEST.json OUTDIR"));
    }
    let req: Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let paths: Vec<PathBuf> = serde_json::from_value(
        req.get("frames")
            .cloned()
            .ok_or_else(|| fail("request needs frames"))?,
    )?;
    let mut frames = paths
        .par_iter()
        .map(|p| read(p))
        .collect::<Result<Vec<_>>>()?;
    frames.sort_by_key(|f| f.at);
    if frames.len() < 2 || frames.windows(2).any(|p| p[0].at >= p[1].at) {
        return Err(fail(
            "verification needs unique advancing native frames including the start",
        ));
    }
    if frames
        .windows(2)
        .any(|p| p[0].lat != p[1].lat || p[0].lon != p[1].lon)
    {
        return Err(fail(
            "moving native footprints need a per-frame station plan; this verifier binds a fixed regional grid",
        ));
    }
    let radius = req
        .get("max_distance_km")
        .and_then(Value::as_f64)
        .unwrap_or(15.0);
    if !radius.is_finite() || radius <= 0.0 {
        return Err(fail("max_distance_km must be finite and positive"));
    }
    let mut observations = Vec::new();
    let mut sources = Vec::new();
    for (key, reader) in [
        ("asos_csv", asos as fn(&Path) -> Result<Vec<Observation>>),
        ("mping_json", mping),
    ] {
        if let Some(path) = req.get(key).and_then(Value::as_str) {
            let p = Path::new(path);
            observations.extend(reader(p)?);
            sources
                .push(json!({"kind":key,"path":p,"sha256":rw_obs::hex_sha256(&std::fs::read(p)?)}));
        }
    }
    if observations.is_empty() {
        return Err(fail("no usable ASOS or mPING precipitation-type reports"));
    }
    if observations.iter().any(|o| {
        !o.lat.is_finite() || !o.lon.is_finite() || o.lat.abs() > 90.0 || o.lon.abs() > 180.0
    }) {
        return Err(fail(
            "observation coordinates are not valid latitude/longitude",
        ));
    }
    // One ASOS station serves one interval, positive if any report says FZRA.
    let mut groups: BTreeMap<(String, String, usize), Observation> = BTreeMap::new();
    for o in observations {
        let k = frames.partition_point(|f| f.at < o.at);
        if k == 0 || k == frames.len() || o.at <= frames[k - 1].at {
            continue;
        }
        let key = (o.source.clone(), o.id.clone(), k);
        groups
            .entry(key)
            .and_modify(|a| {
                a.fzra |= o.fzra;
                if (frames[k].at - o.at).abs() < (frames[k].at - a.at).abs() {
                    a.at = o.at;
                    a.cold = o.cold;
                    a.ice = o.ice;
                }
            })
            .or_insert(o);
    }
    let rows: Vec<_> = groups.into_iter().collect();
    let pairs:Vec<Value>=rows.par_iter().map(|((_,_,k),o)|{
        let f=&frames[*k];let previous=&frames[*k-1];
        let (cell,d)=f.lat.iter().zip(&f.lon).enumerate().map(|(i,(&y,&x))|(i,distance(o.lat,o.lon,y,x))).min_by(|a,b|a.1.total_cmp(&b.1)).unwrap();
        let p=f.rate[cell];let keep=d<=radius && p.is_finite() && f.start==previous.at;
        let hourly=f.at-previous.at==3600 && (f.at-o.at).abs()<=600;
        let model_ice=f.flat[cell]-previous.flat[cell];
        json!({"source":o.source,"id":o.id,"frame":f.path,"observation_unix":o.at,"frame_unix":f.at,"distance_km":d,"matched":keep,
            "observed_freezing_rain":o.fzra,"probability_freezing_rain":if keep {Some(p)}else{None},
            "observed_below_freezing":o.cold,"probability_below_freezing":if keep && (f.at-o.at).abs()<=600 && f.cold[cell].is_finite(){Some(f.cold[cell])}else{None},
            "observed_horizontal_ice_1h_mm":if hourly{o.ice}else{None},"model_horizontal_ice_1h_mean_mm":if keep&&hourly&&model_ice.is_finite()&&model_ice>=0.0{Some(model_ice)}else{None}})
    }).collect();
    let mut metrics = BTreeMap::new();
    let kinds: BTreeSet<_> = pairs.iter().filter_map(|p| p["source"].as_str()).collect();
    for kind in kinds {
        let matched: Vec<_> = pairs
            .iter()
            .filter(|p| p["source"] == kind && p["matched"] == true)
            .collect();
        let n = matched.len();
        let brier: f64 = matched
            .iter()
            .map(|p| {
                let y = if p["observed_freezing_rain"] == true {
                    1.0
                } else {
                    0.0
                };
                (p["probability_freezing_rain"].as_f64().unwrap() - y).powi(2)
            })
            .sum();
        let positives = matched
            .iter()
            .filter(|p| p["observed_freezing_rain"] == true)
            .count();
        let ice_errors: Vec<_> = matched
            .iter()
            .filter_map(|p| {
                Some(
                    p["model_horizontal_ice_1h_mean_mm"].as_f64()?
                        - p["observed_horizontal_ice_1h_mm"].as_f64()?,
                )
            })
            .collect();
        let cold_errors: Vec<_> = matched
            .iter()
            .filter_map(|p| {
                Some(
                    (p["probability_below_freezing"].as_f64()?
                        - if p["observed_below_freezing"].as_bool()? {
                            1.0
                        } else {
                            0.0
                        })
                    .powi(2),
                )
            })
            .collect();
        metrics.insert(kind.to_owned(),json!({"matched_intervals":n,"positive_freezing_rain_intervals":positives,"brier":if n>0{Some(brier/n as f64)}else{None},"below_freezing_pairs":cold_errors.len(),"below_freezing_brier":if cold_errors.is_empty(){None}else{Some(cold_errors.iter().sum::<f64>()/cold_errors.len() as f64)},"horizontal_ice_1h_pairs":ice_errors.len(),"horizontal_ice_1h_mae_mm":if ice_errors.is_empty(){None}else{Some(ice_errors.iter().map(|v|v.abs()).sum::<f64>()/ice_errors.len() as f64)}}));
    }
    let complete = ["ASOS", "mPING"].iter().all(|s| {
        metrics
            .get(*s)
            .and_then(|v| v["matched_intervals"].as_u64())
            .unwrap_or(0)
            > 0
    });
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out)?;
    let mut file = std::fs::File::create(out.join("pairs.jsonl"))?;
    for p in &pairs {
        writeln!(file, "{}", serde_json::to_string(p)?)?;
    }
    let receipt = json!({"schema":"gpuwm-ice-observation-binding.v1","status":if complete{"BOUND"}else{"PARTIAL"},"sources":sources,"frames":frames.iter().map(|f|json!({"path":f.path,"sha256":f.sha,"time":f.at})).collect::<Vec<_>>(),"metrics":metrics,"max_distance_km":radius,"method":"nearest native grid point; report belongs to (previous frame, current frame]; ASOS presence grouped by station and interval; point reports represent interval occurrence only","radial_ice_magnitude_verified":false,"magnitude_note":"ASOS I groups report elevated horizontal ice, not radial ice; mPING precipitation type does not measure ice magnitude","pairs":pairs.len()});
    std::fs::write(
        out.join("receipt.json"),
        serde_json::to_string_pretty(&receipt)? + "\n",
    )?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
fn main() {
    std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    if let Err(e) = run() {
        eprintln!("rw_ice_verify: {e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn present_weather_excludes_failed_sensor_and_remarks() {
        assert!(reports_freezing_rain("-FZRA", ""));
        assert!(reports_freezing_rain(
            "",
            "KAAA 311200Z 02010KT -FZRAPL OVC020 M02/M03 RMK AO2"
        ));
        assert!(!reports_freezing_rain(
            "",
            "KAAA 311200Z 02010KT OVC020 M02/M03 RMK AO2 FZRANO"
        ));
        assert!(!reports_freezing_rain(
            "",
            "KAAA 311200Z 02010KT -SN OVC020 M02/M03 RMK FZRAB15E40"
        ));
        assert!(!reports_freezing_rain("REFZRA", ""));
        assert!(!reports_freezing_rain("-SN", ""));
    }
    #[test]
    fn archive_times_and_native_distance_are_utc_and_geodesic() {
        assert_eq!(
            time("2023-01-31T12:00:00Z").unwrap(),
            time("2023-01-31 12:00").unwrap()
        );
        assert!((distance(30.0, -97.0, 31.0, -97.0) - 111.1949).abs() < 0.001);
        assert!(time("2023-02-31 12:00").is_err());
    }
}
