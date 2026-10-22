//! AERONET V3 direct-sun Level 2 AOD: spectral interpolation and explicit
//! space/time collocation, without extrapolation or Python array processing.
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    observations: Vec<PathBuf>,
    frames: Vec<PathBuf>,
    max_distance_km: f64,
    max_time_gap_seconds: i64,
}
#[derive(Clone, Serialize)]
struct Observation {
    site: String,
    valid_time: String,
    latitude: f64,
    longitude: f64,
    aod_550: f64,
    wavelengths_nm: [f64; 2],
    #[serde(skip)]
    epoch: i64,
}
struct Frame {
    time: i64,
    lat: Vec<f64>,
    lon: Vec<f64>,
    aod: Vec<f64>,
}
#[derive(Serialize)]
struct Pair {
    site: String,
    valid_time: String,
    latitude: f64,
    longitude: f64,
    observed_aod_550: f64,
    model_aod_550: f64,
    distance_km: f64,
    model_cell: usize,
    bracket_epoch_seconds: [i64; 2],
    observed_wavelengths_nm: [f64; 2],
}
// RFC4180 single-line fields; the published service uses one row per line.
// Multiline quoted station names are refused, not split into false observations.
fn csv(line: &str) -> Result<Vec<String>, String> {
    let mut fields = vec![];
    let mut field = String::new();
    let mut quoted = false;
    let mut closed = false;
    let mut chars = line.trim_end_matches('\r').chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field.push(c);
            }
        } else if c == ',' {
            fields.push(std::mem::take(&mut field));
            closed = false;
        } else if c == '"' && field.is_empty() && !closed {
            quoted = true;
        } else if closed {
            return Err("characters after CSV closing quote".into());
        } else {
            field.push(c);
        }
    }
    if quoted {
        return Err("unterminated CSV quote".into());
    }
    fields.push(field);
    Ok(fields)
}
fn number(row: &[String], index: usize) -> Result<f64, String> {
    row[index]
        .parse::<f64>()
        .ok()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("invalid numeric AERONET column {index}"))
}
fn observations(path: &Path) -> Result<(Vec<Observation>, usize), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !text
        .lines()
        .take(8)
        .any(|l| l.trim() == "Version 3: AOD Level 2.0")
    {
        return Err(format!(
            "{}: requires AERONET Version 3 direct-sun AOD Level 2.0",
            path.display()
        ));
    }
    let mut lines = text.lines().skip_while(|l| !l.starts_with("AERONET_Site,"));
    let header = csv(lines.next().ok_or("missing AERONET CSV header")?)?;
    let col = |name: &str| {
        header
            .iter()
            .position(|s| s == name)
            .ok_or_else(|| format!("missing AERONET column {name}"))
    };
    let site = col("AERONET_Site")?;
    let date = col("Date(dd:mm:yyyy)")?;
    let time = col("Time(hh:mm:ss)")?;
    let lat = col("Site_Latitude(Degrees)")?;
    let lon = col("Site_Longitude(Degrees)")?;
    let quality = col("Data_Quality_Level")?;
    let mut bands = vec![];
    for (i, h) in header.iter().enumerate() {
        if let Some(band) = h.strip_prefix("AOD_").and_then(|h| h.strip_suffix("nm")) {
            if let Some(w) = header
                .iter()
                .position(|h| h == &format!("Exact_Wavelengths_of_AOD(um)_{band}nm"))
            {
                bands.push((i, w));
            }
        }
    }
    let mut result = vec![];
    let mut excluded = 0;
    for (r, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = csv(line).map_err(|e| format!("{} row {}: {e}", path.display(), r + 1))?;
        if row.len() != header.len() {
            return Err(format!(
                "{} row {}: CSV width mismatch",
                path.display(),
                r + 1
            ));
        }
        if row[quality] != "lev20" {
            return Err("non-Level-2 observation in Level-2 download".into());
        }
        let dt = NaiveDateTime::parse_from_str(
            &format!("{} {}", row[date], row[time]),
            "%d:%m:%Y %H:%M:%S",
        )
        .map_err(|e| e.to_string())?;
        let latitude = number(&row, lat)?;
        let longitude = number(&row, lon)?;
        if latitude.abs() > 90. || longitude.abs() > 180. || row[site].is_empty() {
            return Err("invalid station location/name".into());
        }
        let mut spectral = vec![];
        for &(a, w) in &bands {
            let aod = number(&row, a)?;
            let nm = number(&row, w)? * 1000.;
            if aod >= 0. && nm > 0. {
                spectral.push((nm, aod));
            }
        }
        spectral.sort_by(|a, b| a.0.total_cmp(&b.0));
        let lower = spectral.iter().rev().find(|p| p.0 <= 550.);
        let upper = spectral.iter().find(|p| p.0 >= 550.);
        let (a, b) = match (lower, upper) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                excluded += 1;
                continue;
            }
        };
        // Log-log interpolation uses the measured band centres, never nominal
        // 500/675 labels. Zero extinction has no logarithm; exact 550 is valid.
        let value = if a.0 == b.0 {
            a.1
        } else if a.1 > 0. && b.1 > 0. {
            let f = (550_f64 / a.0).ln() / (b.0 / a.0).ln();
            (a.1.ln() + (b.1.ln() - a.1.ln()) * f).exp()
        } else {
            excluded += 1;
            continue;
        };
        result.push(Observation {
            site: row[site].clone(),
            valid_time: dt.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            latitude,
            longitude,
            aod_550: value,
            wavelengths_nm: [a.0, b.0],
            epoch: dt.and_utc().timestamp(),
        });
    }
    Ok((result, excluded))
}
fn frame(path: &Path) -> Result<Frame, String> {
    let file = netcrust::File::open(path).map_err(|e| e.to_string())?;
    let read = |n: &str| super::compose_source::values(&file, n);
    let (lat, shape) = read("XLAT")?;
    if shape.len() != 3 || shape[0] != 1 || shape[1] < 2 || shape[2] < 2 {
        return Err("AOD verification expects one Time/y/x frame per file".into());
    }
    let (lon, ls) = read("XLONG")?;
    let (aod, ash) = read("AOD5502D")?;
    if ls != shape || ash != shape {
        return Err("AOD5502D/XLAT/XLONG shapes differ".into());
    }
    if aod.iter().any(|&x| x < 0.)
        || lat.iter().any(|x| x.abs() > 90.)
        || lon.iter().any(|x| x.abs() > 360.)
    {
        return Err("invalid AOD or coordinates".into());
    }
    let times = file.read_strings("Times").map_err(|e| e.to_string())?;
    if times.len() != 1 {
        return Err("expected one WRF valid time".into());
    }
    let time = NaiveDateTime::parse_from_str(times[0].trim_matches('\0'), "%Y-%m-%d_%H:%M:%S")
        .map_err(|e| e.to_string())?
        .and_utc()
        .timestamp();
    Ok(Frame {
        time,
        lat,
        lon,
        aod,
    })
}
fn distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let a = ((lat2 - lat1).to_radians() * 0.5).sin().powi(2)
        + lat1.to_radians().cos()
            * lat2.to_radians().cos()
            * ((lon2 - lon1).to_radians() * 0.5).sin().powi(2);
    6371. * 2. * a.clamp(0., 1.).sqrt().asin()
}
fn metrics(pairs: &[&Pair]) -> serde_json::Value {
    let n = pairs.len() as f64;
    if n == 0. {
        return json!({"pairs":0});
    }
    let errors = pairs
        .iter()
        .map(|p| p.model_aod_550 - p.observed_aod_550)
        .collect::<Vec<_>>();
    json!({"pairs":pairs.len(),"bias":errors.iter().sum::<f64>()/n,"mae":errors.iter().map(|e|e.abs()).sum::<f64>()/n,"rmse":(errors.iter().map(|e|e*e).sum::<f64>()/n).sqrt(),"observed_mean":pairs.iter().map(|p|p.observed_aod_550).sum::<f64>()/n,"model_mean":pairs.iter().map(|p|p.model_aod_550).sum::<f64>()/n})
}
pub(super) fn run(manifest: &Path, out: &Path, jobs: usize) -> Result<(), String> {
    let started = std::time::Instant::now();
    let m: Manifest = serde_json::from_slice(&fs::read(manifest).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if m.schema != "gpuwm-aeronet-aod-v1"
        || m.observations.is_empty()
        || !(1..=64).contains(&jobs)
        || !m.max_distance_km.is_finite()
        || m.max_distance_km <= 0.
        || m.max_time_gap_seconds <= 0
    {
        return Err("invalid AERONET manifest or workers (1..64)".into());
    }
    let mut obs = vec![];
    let mut spectral_excluded = 0;
    for path in &m.observations {
        let (mut o, n) = observations(path)?;
        obs.append(&mut o);
        spectral_excluded += n;
    }
    obs.sort_by(|a, b| (&a.site, a.epoch).cmp(&(&b.site, b.epoch)));
    if obs
        .windows(2)
        .any(|p| p[0].site == p[1].site && p[0].epoch == p[1].epoch)
    {
        return Err("duplicate station/time observation; use nonoverlapping downloads".into());
    }
    if obs.is_empty() {
        return Err("no Level-2 observations with spectral coverage of 550 nm".into());
    }
    let next = AtomicUsize::new(0);
    let results = Mutex::new(vec![]);
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(m.frames.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= m.frames.len() {
                    break;
                }
                let result = frame(&m.frames[i]);
                results.lock().unwrap().push(result);
            });
        }
    });
    let mut frames = results
        .into_inner()
        .unwrap()
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    frames.sort_by_key(|f| f.time);
    if frames.windows(2).any(|p| p[0].time == p[1].time) {
        return Err("duplicate model valid times".into());
    }
    if frames
        .iter()
        .any(|f| f.lat != frames[0].lat || f.lon != frames[0].lon)
    {
        return Err("model grid changes during station collocation".into());
    }
    let mut locations = BTreeMap::new();
    let mut pairs = vec![];
    let mut outside_space = 0;
    let mut outside_time = 0;
    for o in &obs {
        if frames.is_empty() {
            break;
        }
        let key = (o.latitude.to_bits(), o.longitude.to_bits());
        let (cell, dist) = *locations.entry(key).or_insert_with(|| {
            frames[0]
                .lat
                .iter()
                .zip(&frames[0].lon)
                .enumerate()
                .map(|(i, (&lat, &lon))| (i, distance(o.latitude, o.longitude, lat, lon)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
        });
        if dist > m.max_distance_km {
            outside_space += 1;
            continue;
        }
        let bracket = match frames.binary_search_by_key(&o.epoch, |f| f.time) {
            Ok(i) => Some((i, i)),
            Err(i) if i > 0 && i < frames.len() => Some((i - 1, i)),
            _ => None,
        };
        let (a, b) = match bracket {
            Some((a, b)) if frames[b].time - frames[a].time <= m.max_time_gap_seconds => (a, b),
            _ => {
                outside_time += 1;
                continue;
            }
        };
        let weight = if a == b {
            0.
        } else {
            (o.epoch - frames[a].time) as f64 / (frames[b].time - frames[a].time) as f64
        };
        let value = frames[a].aod[cell] + weight * (frames[b].aod[cell] - frames[a].aod[cell]);
        pairs.push(Pair {
            site: o.site.clone(),
            valid_time: o.valid_time.clone(),
            latitude: o.latitude,
            longitude: o.longitude,
            observed_aod_550: o.aod_550,
            model_aod_550: value,
            distance_km: dist,
            model_cell: cell,
            bracket_epoch_seconds: [frames[a].time, frames[b].time],
            observed_wavelengths_nm: o.wavelengths_nm,
        });
    }
    let stations = pairs
        .iter()
        .map(|p| p.site.clone())
        .collect::<BTreeSet<_>>();
    let by_station = stations
        .iter()
        .map(|s| {
            (
                s.clone(),
                metrics(&pairs.iter().filter(|p| &p.site == s).collect::<Vec<_>>()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let summary = json!({"schema":"gpuwm-aeronet-aod-verification-v1","quality":"AERONET V3 direct-sun Level 2.0","spectral_operator":"log-log interpolation of positive AOD at bracketing exact measured wavelengths to 550 nm; no extrapolation","spatial_operator":"nearest great-circle model mass point within declared distance","temporal_operator":"linear interpolation between bracketing model frames; no extrapolation","max_distance_km":m.max_distance_km,"max_time_gap_seconds":m.max_time_gap_seconds,"observations":obs.len(),"spectral_excluded":spectral_excluded,"outside_distance":outside_space,"outside_time":outside_time,"scored":!pairs.is_empty(),"metrics":metrics(&pairs.iter().collect::<Vec<_>>()),"by_station":by_station,"workers":jobs.min(m.frames.len()),"wall_seconds":started.elapsed().as_secs_f64(),"input_manifest":m.observations,"model_files":m.frames});
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    for (name, value) in [
        ("observations.json", serde_json::to_value(&obs).unwrap()),
        ("pairs.json", serde_json::to_value(&pairs).unwrap()),
        ("summary.json", summary),
    ] {
        fs::write(out.join(name), serde_json::to_vec_pretty(&value).unwrap())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quoted_fields_and_dateline_distance() {
        assert_eq!(
            csv("\"a,b\",\"c\"\"d\",x").unwrap(),
            vec!["a,b", "c\"d", "x"]
        );
        assert!(csv("\"bad").is_err());
        assert!(csv("\"x\"junk").is_err());
        assert!((distance(0., 179.9, 0., -179.9) - 22.238985).abs() < 1e-5);
    }
    #[test]
    fn exact_band_centres_and_missing_spectral_coverage() {
        let dir = std::env::temp_dir().join(format!("aeronet-spectrum-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("obs.txt");
        let prefix="AERONET Data Download (Version 3 Direct Sun)\nVersion 3: AOD Level 2.0\nAERONET_Site,Date(dd:mm:yyyy),Time(hh:mm:ss),Site_Latitude(Degrees),Site_Longitude(Degrees),Data_Quality_Level,AOD_500nm,AOD_675nm,Exact_Wavelengths_of_AOD(um)_500nm,Exact_Wavelengths_of_AOD(um)_675nm\n";
        fs::write(&path,format!("{prefix}Station,15:03:2022,12:00:00,40,-3,lev20,1,0.5,0.5,0.675\nStation,15:03:2022,12:01:00,40,-3,lev20,-999,0.5,-999,0.675\n")).unwrap();
        let (o, excluded) = observations(&path).unwrap();
        assert_eq!(o.len(), 1);
        assert_eq!(excluded, 1);
        let expected = (550_f64 / 500.).powf(-2_f64.ln() / (675_f64 / 500.).ln());
        assert!((o[0].aod_550 - expected).abs() < 1e-14);
        assert_eq!(o[0].valid_time, "2022-03-15T12:00:00Z");
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn collocation_interpolates_time_and_excludes_unbracketed_samples() {
        use netcdf_writer::{NcFormat, NcType, NcWriter, Schema, VarData};
        let root = std::env::temp_dir().join(format!("aeronet-collocation-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let fixture = |name: &str, time: &str, value: f64| {
            let path = root.join(name);
            let mut schema = Schema::new(NcFormat::Classic);
            let t = schema.def_dim("Time", 1, false).unwrap();
            let y = schema.def_dim("y", 2, false).unwrap();
            let x = schema.def_dim("x", 2, false).unwrap();
            let c = schema.def_dim("DateStrLen", 19, false).unwrap();
            let lat = schema.def_var("XLAT", NcType::Double, &[t, y, x]).unwrap();
            let lon = schema.def_var("XLONG", NcType::Double, &[t, y, x]).unwrap();
            let aod = schema
                .def_var("AOD5502D", NcType::Double, &[t, y, x])
                .unwrap();
            let stamp = schema.def_var("Times", NcType::Char, &[t, c]).unwrap();
            let mut writer = NcWriter::create(&path, schema).unwrap();
            writer
                .write_var(lat, VarData::F64(&[40., 40., 41., 41.]))
                .unwrap();
            writer
                .write_var(lon, VarData::F64(&[-3., -2., -3., -2.]))
                .unwrap();
            writer.write_var(aod, VarData::F64(&[value; 4])).unwrap();
            writer
                .write_var(stamp, VarData::Char(time.as_bytes()))
                .unwrap();
            writer.finish().unwrap();
            path
        };
        let a = fixture("a.nc", "2022-03-15_12:00:00", 0.4);
        let b = fixture("b.nc", "2022-03-15_13:00:00", 0.8);
        let obs = root.join("obs.txt");
        fs::write(&obs,"AERONET Data Download (Version 3 Direct Sun)\nVersion 3: AOD Level 2.0\nAERONET_Site,Date(dd:mm:yyyy),Time(hh:mm:ss),Site_Latitude(Degrees),Site_Longitude(Degrees),Data_Quality_Level,AOD_550nm,Exact_Wavelengths_of_AOD(um)_550nm\nStation,15:03:2022,12:30:00,40,-3,lev20,0.5,0.55\nStation,15:03:2022,13:30:00,40,-3,lev20,0.5,0.55\nFar,15:03:2022,12:30:00,0,0,lev20,0.5,0.55\n").unwrap();
        let manifest = root.join("request.json");
        let mut doc = json!({"schema":"gpuwm-aeronet-aod-v1","observations":[obs],"frames":[b,a],"max_distance_km":10.,"max_time_gap_seconds":3600});
        fs::write(&manifest, serde_json::to_vec(&doc).unwrap()).unwrap();
        let out = root.join("score");
        run(&manifest, &out, 2).unwrap();
        let summary: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("summary.json")).unwrap()).unwrap();
        assert_eq!(summary["metrics"]["pairs"], 1);
        assert_eq!(summary["outside_time"], 1);
        assert_eq!(summary["outside_distance"], 1);
        assert!((summary["metrics"]["bias"].as_f64().unwrap() - 0.1).abs() < 1e-14);
        doc["max_time_gap_seconds"] = json!(1800);
        fs::write(&manifest, serde_json::to_vec(&doc).unwrap()).unwrap();
        run(&manifest, &out, 2).unwrap();
        let summary: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("summary.json")).unwrap()).unwrap();
        assert_eq!(summary["scored"], false);
        doc["observations"].as_array_mut().unwrap().push(json!(obs));
        fs::write(&manifest, serde_json::to_vec(&doc).unwrap()).unwrap();
        assert!(run(&manifest, &out, 2)
            .unwrap_err()
            .contains("duplicate station/time"));
        fs::remove_dir_all(root).unwrap();
    }
}
