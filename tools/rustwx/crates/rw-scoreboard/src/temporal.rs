//! Bracketed baseline fields, with explicit temporal provenance.
use crate::*;

fn paired(a: &Artifact, b: &Artifact) -> Result<()> {
    if a.method_id != METHOD
        || b.method_id != METHOD
        || a.support_hash != b.support_hash
        || !same_cycle(a, b)?
        || a.stations != b.stations
    {
        return Err(fail(
            "temporal endpoints differ in method, cycle or frozen support",
        ));
    }
    Ok(())
}
fn input<'a>(r: &'a Value, key: &str, alternative: &str) -> Result<&'a str> {
    r[key]
        .as_str()
        .or_else(|| r[alternative].as_str())
        .ok_or_else(|| fail(format!("{key} endpoint is required")))
}
fn value(a: f64, b: f64, w: f64, reflectivity: bool) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let v = if reflectivity {
        let z = (1. - w) * 10f64.powf(a / 10.) + w * 10f64.powf(b / 10.);
        10. * z.log10()
    } else {
        a + (b - a) * w
    };
    v.is_finite().then_some(v)
}
pub fn interpolate(r: &Value) -> Result<Value> {
    let a = read_artifact(input(r, "lower", "input")?)?;
    let b = read_artifact(input(r, "upper", "next_input")?)?;
    paired(&a, &b)?;
    let at = time(&a.valid_time)?;
    let bt = time(&b.valid_time)?;
    let target = time(get_str(r, "valid_time")?)?;
    if bt <= at || target < at || target > bt {
        return Err(fail(
            "temporal interpolation requires two ordered endpoints bracketing the target, never extrapolation",
        ));
    }
    let weight = (target - at) as f64 / (bt - at) as f64;
    let mut out = a.clone();
    out.valid_time = iso(target);
    let init = time(
        out.init_time
            .as_deref()
            .ok_or_else(|| fail("baseline interpolation requires a cycle"))?,
    )?;
    if target < init {
        return Err(fail("interpolated forecast valid time precedes its cycle"));
    }
    out.lead_minutes = ((target - init) / 60) as u32;
    out.lead_seconds = (target - init) as u64;
    out.source_sha256 = digest(&serde_json::to_vec(&(
        "linear-time-v1",
        &a.source_sha256,
        &b.source_sha256,
        target,
    ))?);
    out.diagnostics = json!({"temporal_interpolation":"linear_instantaneous_and_cumulative_linear_Z_reflectivity","source_valid_times":[a.valid_time,b.valid_time],"source_sha256":[a.source_sha256,b.source_sha256],"upper_weight":weight});
    out.points.clear();
    out.grids.clear();
    out.drops.clear();
    let points: BTreeMap<_, _> = b.points.iter().map(|p| (&p.station_id, p)).collect();
    for p in &a.points {
        let Some(q) = points.get(&p.station_id) else {
            out.drops.push(
                json!({"station_id":p.station_id,"reason":"missing_upper_temporal_endpoint"}),
            );
            continue;
        };
        if let (Some(x), Some(y)) = (p.terrain_m, q.terrain_m) {
            if (x - y).abs() > 0.01 {
                return Err(fail("baseline terrain changes between temporal brackets"));
            }
        }
        let mut values = BTreeMap::new();
        for (name, x) in &p.values {
            if name == "wind_speed_10m" || name == "precipitation" {
                continue;
            }
            if let Some(y) = q.values.get(name) {
                if let Some(v) = value(*x, *y, weight, name == "reflectivity") {
                    values.insert(name.clone(), v);
                }
            }
        }
        if let (Some(u), Some(v)) = (values.get("u_wind_10m"), values.get("v_wind_10m")) {
            values.insert("wind_speed_10m".into(), u.hypot(*v));
        }
        out.points.push(Point {
            station_id: p.station_id.clone(),
            valid_time: out.valid_time.clone(),
            observation_time: None,
            observation_times: BTreeMap::new(),
            observation_sources: BTreeMap::new(),
            values,
            terrain_m: p.terrain_m,
        });
    }
    for g in &a.grids {
        if g.variable == "precipitation" {
            continue;
        }
        let Some(h) = b
            .grids
            .iter()
            .find(|h| h.variable == g.variable && h.window_minutes == g.window_minutes)
        else {
            continue;
        };
        if g.spec != h.spec || g.values.len() != h.values.len() {
            return Err(fail("temporal endpoint verification grids differ"));
        }
        out.grids.push(Grid {
            spec: g.spec.clone(),
            variable: g.variable.clone(),
            window_minutes: g.window_minutes,
            values: g
                .values
                .iter()
                .zip(&h.values)
                .map(|(a, b)| {
                    a.zip(*b)
                        .and_then(|(a, b)| value(a, b, weight, g.variable == "reflectivity"))
                })
                .collect(),
        });
    }
    write_artifact(&out, get_str(r, "output")?)
}
pub(crate) fn difference(current: f64, previous: f64) -> Option<f64> {
    if !current.is_finite() || !previous.is_finite() || current - previous < -0.0001 {
        None
    } else {
        Some((current - previous).max(0.))
    }
}
pub fn accumulate(r: &Value) -> Result<Value> {
    let current = read_artifact(input(r, "current", "next_input")?)?;
    let previous = read_artifact(input(r, "previous", "input")?)?;
    paired(&current, &previous)?;
    let window = r["window_minutes"]
        .as_u64()
        .ok_or_else(|| fail("accumulate window_minutes required"))?;
    if window == 0 || time(&current.valid_time)? - time(&previous.valid_time)? != window as i64 * 60
    {
        return Err(fail(
            "cumulative endpoints do not span the exact requested precipitation window",
        ));
    }
    let mut out = current.clone();
    out.grids
        .retain(|g| g.variable != "precipitation" && g.variable != "precipitation_total");
    let mut found = false;
    for c in current
        .grids
        .iter()
        .filter(|g| g.variable == "precipitation_total")
    {
        let p = previous
            .grids
            .iter()
            .find(|g| g.variable == "precipitation_total" && g.spec == c.spec)
            .ok_or_else(|| fail("previous cumulative precipitation grid absent or different"))?;
        if c.values.len() != p.values.len() {
            return Err(fail(
                "cumulative precipitation endpoint grid lengths differ",
            ));
        }
        out.grids.push(Grid {
            spec: c.spec.clone(),
            variable: "precipitation".into(),
            window_minutes: window as u32,
            values: c
                .values
                .iter()
                .zip(&p.values)
                .map(|(a, b)| a.zip(*b).and_then(|(a, b)| difference(a, b)))
                .collect(),
        });
        found = true;
    }
    if !found {
        return Err(fail(
            "current cumulative precipitation grid absent; extract with include_cumulative=true",
        ));
    }
    let points: BTreeMap<_, _> = previous.points.iter().map(|p| (&p.station_id, p)).collect();
    for p in &mut out.points {
        p.values.remove("precipitation");
        if let (Some(current), Some(previous)) = (
            p.values.get("precipitation_total").copied(),
            points
                .get(&p.station_id)
                .and_then(|p| p.values.get("precipitation_total"))
                .copied(),
        ) {
            if let Some(v) = difference(current, previous) {
                p.values.insert("precipitation".into(), v);
            }
        }
        p.values.remove("precipitation_total");
    }
    out.source_sha256 = digest(&serde_json::to_vec(&(
        "cumulative-difference-v1",
        &current.source_sha256,
        &previous.source_sha256,
        window,
    ))?);
    out.diagnostics = json!({"accumulation_start":previous.valid_time,"accumulation_end":current.valid_time,"accumulation_window_minutes":window,"endpoint_sha256":[previous.source_sha256,current.source_sha256],"endpoint_temporal_provenance":[previous.diagnostics,current.diagnostics]});
    write_artifact(&out, get_str(r, "output")?)
}
/// Observation matching v2 (`obs-match-v2`): the forecast at each station report's own time. The scored hour's
/// forecast (`forecast`, valid at the hour) keeps its grids, accumulations and terrain; each station's instantaneous
/// value of each variable is replaced by the linear interpolation in time to the time of the report `observations`
/// selected for that station and variable (observe/merge with a wider `match_seconds`), between the two hourly outputs
/// that bracket it: `previous` (one output step before the hour) for a report before the hour, `next` for one after.
/// A report outside the bracket, or with no output on its side, drops that variable for that station (named
/// `no_bracketing_output`); a report exactly at the hour keeps the hour's value. 10 m wind speed is the speed of the
/// interpolated components at the wind report's time. Never extrapolates.
pub fn to_reports(r: &Value) -> Result<Value> {
    let current = read_artifact(get_str(r, "forecast")?)?;
    let observations = read_artifact(get_str(r, "observations")?)?;
    let previous = r["previous"].as_str().map(read_artifact).transpose()?;
    let next = r["next"].as_str().map(read_artifact).transpose()?;
    let ct = time(&current.valid_time)?;
    if time(&observations.valid_time)? != ct || observations.support_hash != current.support_hash {
        return Err(fail("observations are not for the forecast's hour and frozen support"));
    }
    for (side, b) in [("previous", &previous), ("next", &next)] {
        if let Some(b) = b {
            paired(&current, b)?;
            let bt = time(&b.valid_time)?;
            if (side == "previous" && bt >= ct) || (side == "next" && bt <= ct) {
                return Err(fail(format!("{side} output is on the wrong side of the scored hour")));
            }
        }
    }
    let side_points = |a: &Option<Artifact>| -> BTreeMap<String, Point> {
        a.as_ref().map(|a| a.points.iter().map(|p| (p.station_id.clone(), p.clone())).collect()).unwrap_or_default()
    };
    let before = side_points(&previous);
    let after = side_points(&next);
    let before_t = previous.as_ref().map(|a| time(&a.valid_time)).transpose()?;
    let after_t = next.as_ref().map(|a| time(&a.valid_time)).transpose()?;
    let reports: BTreeMap<_, _> = observations.points.iter().map(|p| (p.station_id.clone(), p)).collect();
    // The value of `name` at time t for this station, from the hour's value and the bracketing output's.
    let at = |station: &str, name: &str, x: f64, t: i64| -> Option<f64> {
        if t == ct {
            return Some(x);
        }
        let (bt, points) = if t < ct { (before_t?, &before) } else { (after_t?, &after) };
        let y = *points.get(station)?.values.get(name)?;
        let (lo_t, lo, hi_t, hi) = if t < ct { (bt, y, ct, x) } else { (ct, x, bt, y) };
        if t < lo_t || t > hi_t {
            return None;
        }
        value(lo, hi, (t - lo_t) as f64 / (hi_t - lo_t) as f64, false)
    };
    let mut out = current.clone();
    let (mut moved, mut kept, mut dropped, mut largest) = (0u64, 0u64, 0u64, 0i64);
    for p in &mut out.points {
        let Some(o) = reports.get(&p.station_id) else { continue };
        let report_time = |name: &str| -> Result<Option<i64>> {
            o.observation_times.get(name).or(o.observation_time.as_ref()).map(|t| time(t)).transpose()
        };
        let mut values = BTreeMap::new();
        for (name, x) in &p.values {
            if name == "wind_speed_10m" {
                continue;
            }
            if name.starts_with("precipitation") || !o.values.contains_key(name) {
                values.insert(name.clone(), *x);
                continue;
            }
            let Some(t) = report_time(name)? else {
                values.insert(name.clone(), *x);
                continue;
            };
            match at(&p.station_id, name, *x, t) {
                Some(v) => {
                    values.insert(name.clone(), v);
                    if t == ct { kept += 1 } else { moved += 1 }
                    largest = largest.max((t - ct).abs());
                }
                None => {
                    dropped += 1;
                    out.drops.push(json!({"station_id":p.station_id,"variable":name,"reason":"no_bracketing_output","report_time":iso(t)}));
                }
            }
        }
        if let Some(&speed) = p.values.get("wind_speed_10m") {
            let t = if o.values.contains_key("wind_speed_10m") { report_time("wind_speed_10m")? } else { None };
            match t {
                None => {
                    values.insert("wind_speed_10m".into(), speed);
                }
                Some(t) => {
                    let u = p.values.get("u_wind_10m").and_then(|&x| at(&p.station_id, "u_wind_10m", x, t));
                    let v = p.values.get("v_wind_10m").and_then(|&x| at(&p.station_id, "v_wind_10m", x, t));
                    match (u, v) {
                        (Some(u), Some(v)) if t != ct => {
                            values.insert("wind_speed_10m".into(), u.hypot(v));
                            moved += 1;
                            largest = largest.max((t - ct).abs());
                        }
                        _ if t == ct => {
                            values.insert("wind_speed_10m".into(), speed);
                            kept += 1;
                        }
                        _ => {
                            dropped += 1;
                            out.drops.push(json!({"station_id":p.station_id,"variable":"wind_speed_10m","reason":"no_bracketing_output","report_time":iso(t)}));
                        }
                    }
                }
            }
        }
        p.values = values;
    }
    let sha = |a: &Option<Artifact>| a.as_ref().map(|a| a.source_sha256.clone());
    out.source_sha256 = digest(&serde_json::to_vec(&(
        "linear-to-report-time-v1",
        sha(&previous),
        &current.source_sha256,
        sha(&next),
        &observations.source_sha256,
    ))?);
    out.diagnostics = json!({"observation_matching":"obs-match-v2","temporal_interpolation":"linear-to-report-time-v1",
        "source_valid_times":[previous.as_ref().map(|a| a.valid_time.clone()), current.valid_time, next.as_ref().map(|a| a.valid_time.clone())],
        "source_sha256":[sha(&previous), current.source_sha256, sha(&next)],"observations_sha256":observations.source_sha256,
        "values_moved_to_report_time":moved,"values_at_the_hour":kept,"values_without_bracket":dropped,"largest_offset_seconds":largest});
    write_artifact(&out, get_str(r, "output")?)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporal_reflectivity_uses_power() {
        let v = value(0., 20., 0.5, true).unwrap();
        assert!((v - 10. * 50.5f64.log10()).abs() < 1e-12);
        assert!(v > 10.);
    }
    fn artifact(valid: &str, points: Vec<(&str, Vec<(&str, f64)>, Vec<(&str, &str)>)>) -> Artifact {
        Artifact {
            schema: "woof.scoreboard.artifact.v1".into(), method_id: METHOD.into(), valid_time: valid.into(),
            init_time: Some("2026-10-02T12:00:00Z".into()), domain: "d01".into(), lead_minutes: 0, lead_seconds: 0,
            source_sha256: format!("src-{valid}"), support_hash: "support".into(), stations: vec![],
            points: points.into_iter().map(|(id, values, times)| Point {
                station_id: id.into(), valid_time: valid.into(), observation_time: None,
                observation_times: times.into_iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                observation_sources: BTreeMap::new(), values: values.into_iter().map(|(k, v)| (k.to_string(), v)).collect(), terrain_m: None,
            }).collect(),
            tombstones: vec![], grids: vec![], drops: vec![], diagnostics: Value::Null,
        }
    }
    #[test]
    fn reports_off_the_hour_get_the_forecast_at_their_own_time() {
        let dir = std::env::temp_dir().join(format!("rw-to-reports-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = |n: &str| dir.join(n).to_string_lossy().to_string();
        let f = |t: f64, u: f64| vec![
            ("MMMX", vec![("temperature_2m", t), ("u_wind_10m", u), ("v_wind_10m", 0.), ("wind_speed_10m", u.abs()), ("precipitation", 1.)], vec![]),
            ("MMTO", vec![("temperature_2m", t + 1.), ("u_wind_10m", u), ("v_wind_10m", 0.), ("wind_speed_10m", u.abs())], vec![]),
        ];
        for (name, valid, t, u) in [("prev", "2026-10-02T14:00:00Z", 280., 2.), ("cur", "2026-10-02T15:00:00Z", 284., 4.), ("next", "2026-10-02T16:00:00Z", 290., 8.)] {
            write_artifact(&artifact(valid, f(t, u)), &path(name)).unwrap();
        }
        // MMMX reported 15 minutes before the hour, MMTO 45 minutes after it.
        let obs = artifact("2026-10-02T15:00:00Z", vec![
            ("MMMX", vec![("temperature_2m", 283.), ("wind_speed_10m", 3.)], vec![("temperature_2m", "2026-10-02T14:45:00Z"), ("wind_speed_10m", "2026-10-02T14:45:00Z")]),
            ("MMTO", vec![("temperature_2m", 286.)], vec![("temperature_2m", "2026-10-02T15:45:00Z")]),
        ]);
        write_artifact(&obs, &path("obs")).unwrap();
        to_reports(&json!({"forecast": path("cur"), "previous": path("prev"), "next": path("next"), "observations": path("obs"), "output": path("out")})).unwrap();
        let out = read_artifact(&path("out")).unwrap();
        let mx = &out.points.iter().find(|p| p.station_id == "MMMX").unwrap().values;
        assert!((mx["temperature_2m"] - 283.).abs() < 1e-12, "{}", mx["temperature_2m"]); // 3/4 of the way from 280 to 284
        assert!((mx["wind_speed_10m"] - 3.5).abs() < 1e-12); // |u| at 14:45 = 2 + 0.75 * 2
        assert_eq!(mx["precipitation"], 1.); // accumulations stay at the hour
        let to = &out.points.iter().find(|p| p.station_id == "MMTO").unwrap().values;
        assert!((to["temperature_2m"] - 289.5).abs() < 1e-12); // 285 + 0.75 * (291 - 285)
        assert_eq!(to["wind_speed_10m"], 4.); // no wind report: the hour's value, unscored anyway
        // Without the next output, the report after the hour has no bracket: dropped and named, never extrapolated.
        to_reports(&json!({"forecast": path("cur"), "previous": path("prev"), "observations": path("obs"), "output": path("out2")})).unwrap();
        let out2 = read_artifact(&path("out2")).unwrap();
        assert!(!out2.points.iter().find(|p| p.station_id == "MMTO").unwrap().values.contains_key("temperature_2m"));
        assert!(out2.drops.iter().any(|d| d["station_id"] == "MMTO" && d["reason"] == "no_bracketing_output"));
        std::fs::remove_dir_all(&dir).ok();
    }
    #[test]
    fn interval_reset_is_missing() {
        assert_eq!(difference(2., 3.), None);
        assert_eq!(difference(2., 2.00001), Some(0.));
        assert_eq!(difference(3., 2.), Some(1.));
    }
}
