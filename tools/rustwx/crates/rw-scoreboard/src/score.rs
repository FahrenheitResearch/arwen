use crate::*;
use rw_scoreboard::Stats;
fn metrics(s: &Stats) -> Value {
    if s.count == 0 {
        return json!({"n":0,"mae":null,"rmse":null,"bias":null});
    }
    let n = s.count as f64;
    json!({"n":s.count,"mae":s.sum_absolute_error/n,"rmse":(s.sum_squared_error/n).sqrt(),"bias":s.sum_error/n})
}
fn ratio(a: f64, b: f64) -> Option<f64> {
    if b == 0. { None } else { Some(a / b) }
}
fn contingency(f: &[Option<f64>], o: &[Option<f64>], valid: &[bool], threshold: f64) -> Stats {
    let (mut h, mut m, mut a, mut c) = (0, 0, 0, 0);
    let mut s = Stats::default();
    for k in 0..f.len() {
        if valid[k] {
            let fv = f[k].unwrap();
            let ov = o[k].unwrap();
            s.add(fv, ov);
            match (fv >= threshold, ov >= threshold) {
                (true, true) => h += 1,
                (false, true) => m += 1,
                (true, false) => a += 1,
                _ => c += 1,
            }
        }
    }
    s.hits = Some(h);
    s.misses = Some(m);
    s.false_alarms = Some(a);
    s.correct_negatives = Some(c);
    s
}
fn categorical(s: &Stats) -> Value {
    let (h, m, a, c) = (
        s.hits.unwrap() as f64,
        s.misses.unwrap() as f64,
        s.false_alarms.unwrap() as f64,
        s.correct_negatives.unwrap() as f64,
    );
    let random = (h + m) * (h + a) / (h + m + a + c);
    json!({"hits":h,"misses":m,"false_alarms":a,"correct_negatives":c,"bias":ratio(h+a,h+m),"ets":ratio(h-random,h+m+a-random),"csi":ratio(h,h+m+a),"pod":ratio(h,h+m),"far":ratio(a,h+a)})
}
fn fss(f:&[Option<f64>],o:&[Option<f64>],valid:&[bool],spec:&GridSpec,threshold:f64,scale_km:f64)->Stats {
    let dy=6371008.8*spec.dlat.to_radians().abs();
    let hy=(((scale_km*1000./dy-1.)/2.).round().max(0.) as usize).min(spec.ny);
    let widths=(0..spec.ny).map(|j| {
        let dx=6371008.8*spec.dlon.to_radians().abs()*(spec.south+j as f64*spec.dlat).to_radians().cos().abs();
        (((scale_km*1000./dx-1.)/2.).round().max(0.) as usize).min(spec.nx)
    }).collect::<Vec<_>>();
    rw_scoreboard::fractions(&f.iter().map(|v|v.unwrap_or(f64::NAN)).collect::<Vec<_>>(),&o.iter().map(|v|v.unwrap_or(f64::NAN)).collect::<Vec<_>>(),valid,spec.nx,spec.ny,threshold,&widths,hy,false)
}
/// Gross-error limits for point observations, per native variable: an
/// observation farther than this from EVERY scored model (the forecast, and
/// the baseline when the score is paired) is a bad report, not a forecast
/// error, and is rejected and counted in `gross_error_rejected_by_variable`.
///
/// The breakage this prevents, named: one METAR station pressure hundreds of
/// hPa wrong passes the 30-110 kPa climate screen in `observe::screen` and set
/// both models' station-pressure RMSE to 566 Pa at f06 and 1,502 Pa at f12 of
/// the 2026-10-02 00Z WOOF-vs-HRRR comparison, against 70-100 Pa at every
/// other lead. 50 hPa is far outside any forecast error at a lead of a day or
/// two (the RMSE is under 1.2 hPa) and well inside a mis-keyed altimeter or a
/// station-elevation error. The rule asks every model, so it cannot favour one.
const GROSS_ERROR_LIMITS: &[(&str, f64)] = &[("pressure_surface", 5_000.0)];

fn gross_error_limit(variable: &str) -> Option<f64> {
    GROSS_ERROR_LIMITS
        .iter()
        .find(|(name, _)| *name == variable)
        .map(|(_, limit)| *limit)
}

fn gross_error(variable: &str, observed: f64, forecast: f64, baseline: Option<f64>) -> bool {
    let Some(limit) = gross_error_limit(variable) else {
        return false;
    };
    (observed - forecast).abs() > limit
        && baseline.map_or(true, |b| (observed - b).abs() > limit)
}

fn gross_error_limits_json() -> Value {
    Value::Object(
        GROSS_ERROR_LIMITS
            .iter()
            .map(|(name, limit)| ((*name).to_string(), json!(limit)))
            .collect(),
    )
}

fn alias(name: &str) -> &str {
    match name {
        "temperature_2m" => "t2",
        "temperature_2m_raw" => "t2_raw",
        "dewpoint_2m" => "td2",
        "wind_speed_10m" | "u_wind_10m" | "v_wind_10m" => "wind10",
        "pressure_surface" => "surface_pressure",
        _ => name,
    }
}
fn selected(request: &Value, name: &str) -> bool {
    let selector = request["variable"].as_str();
    if selector == Some("wind10") && matches!(name, "u_wind_10m" | "v_wind_10m") {
        return false;
    }
    selector.is_none_or(|v| v == name || v == alias(name))
}
fn pair_rows(
    pairs: &mut Vec<Value>,
    variable: &str,
    window: u32,
    w: &Stats,
    b: Option<&Stats>,
    metric: &str,
    units: &str,
    threshold: Option<f64>,
    scale: Option<f64>,
) {
    pairs.push(json!({"variable":alias(variable),"nativeVariable":variable,"windowMinutes":window,"metric":metric,"units":units,"threshold":threshold,"scaleKm":scale,"woof":w,"baseline":b}));
}
pub fn run(request: &Value) -> Result<Value> {
    let forecast = read_artifact(get_str(request, "forecast")?)?;
    if forecast.method_id != METHOD {
        return Err(fail("scoreboard-v1 cannot score artifacts from a different frozen QC and neighborhood method"));
    }
    let observations = read_artifact(get_str(request, "observations")?)?;
    let baseline = request["baseline"]
        .as_str()
        .map(read_artifact)
        .transpose()?;
    let forecast_lead = actual_lead_seconds(&forecast)?;
    let policy = request["baseline_policy"].as_str().unwrap_or("same_cycle");
    if !["same_cycle", "latest_at_or_before_init"].contains(&policy) {
        return Err(fail("unknown baseline cycle policy"));
    }
    let mut baseline_gap = None;
    let mut baseline_lead = None;
    if let Some(b) = baseline.as_ref() {
        let wi = time(
            forecast
                .init_time
                .as_deref()
                .ok_or_else(|| fail("paired forecast initialization is unknown"))?,
        )?;
        let bi = time(
            b.init_time
                .as_deref()
                .ok_or_else(|| fail("baseline initialization is unknown"))?,
        )?;
        if bi > wi {
            return Err(fail("baseline cycle is newer than the WOOF initialization"));
        }
        if policy == "same_cycle" && bi != wi {
            return Err(fail(
                "older baseline cycle requires explicit latest_at_or_before_init policy",
            ));
        }
        baseline_gap = Some((wi - bi) as u64);
        baseline_lead = Some(actual_lead_seconds(b)?);
    }
    for a in std::iter::once(&observations).chain(baseline.iter()) {
        if a.method_id != forecast.method_id || time(&a.valid_time)? != time(&forecast.valid_time)?
        {
            return Err(fail("forecast and observation method/valid-time mismatch"));
        }
        if a.support_hash != forecast.support_hash {
            return Err(fail(
                "frozen forecast, baseline and observation support differs",
            ));
        }
    }
    let fc: BTreeMap<_, _> = forecast.points.iter().map(|p| (&p.station_id, p)).collect();
    let obs: BTreeMap<_, _> = observations
        .points
        .iter()
        .map(|p| (&p.station_id, p))
        .collect();
    let base: BTreeMap<_, _> = baseline
        .iter()
        .flat_map(|a| a.points.iter())
        .map(|p| (&p.station_id, p))
        .collect();
    let mut pairs = vec![];
    let mut summary = vec![];
    let mut cohort = vec![];
    let mut dropped = BTreeMap::<String, u64>::new();
    let mut gross_rejected = BTreeMap::<String, u64>::new();
    let mut coverage_rows = vec![];
    let (mut forecast_samples, mut baseline_samples, mut observation_samples) = (0u64, 0u64, 0u64);
    for (variable, units) in [
        ("temperature_2m", "K"),
        ("temperature_2m_raw", "K"),
        ("dewpoint_2m", "K"),
        ("wind_speed_10m", "m s-1"),
        ("u_wind_10m", "m s-1"),
        ("v_wind_10m", "m s-1"),
        ("pressure_surface", "Pa"),
    ] {
        if !selected(request, variable) {
            continue;
        }
        let mut w = Stats::default();
        let mut b = Stats::default();
        let mut ids = vec![];
        let mut samples = vec![];
        let (mut fcount, mut bcount, mut ocount) = (0u64, 0u64, 0u64);
        for station in &forecast.stations {
            let observation = obs
                .get(&station.station_id)
                .and_then(|p| p.values.get(variable))
                .copied();
            let fv = fc
                .get(&station.station_id)
                .and_then(|p| p.values.get(variable))
                .copied();
            let bv = base
                .get(&station.station_id)
                .and_then(|p| p.values.get(variable))
                .copied();
            fcount += u64::from(fv.is_some_and(f64::is_finite));
            bcount += u64::from(bv.is_some_and(f64::is_finite));
            ocount += u64::from(observation.is_some_and(f64::is_finite));
            match (observation, fv, bv) {
                (Some(o), Some(f), bval)
                    if o.is_finite()
                        && f.is_finite()
                        && (baseline.is_none() || bval.is_some_and(f64::is_finite))
                        && gross_error(variable, o, f, bval) =>
                {
                    *gross_rejected.entry(variable.into()).or_default() += 1;
                }
                (Some(o), Some(f), bval)
                    if o.is_finite()
                        && f.is_finite()
                        && (baseline.is_none() || bval.is_some_and(f64::is_finite)) =>
                {
                    w.add(f, o);
                    if let Some(v) = bval {
                        b.add(v, o);
                    }
                    ids.push(station.station_id.clone());
                    let point = obs[&station.station_id];
                    samples.push(json!({"station_id":station.station_id,"observation_time":point.observation_times.get(variable).or(point.observation_time.as_ref())}));
                }
                _ => {
                    *dropped.entry(variable.into()).or_default() += 1;
                }
            }
        }
        forecast_samples += fcount;
        baseline_samples += bcount;
        observation_samples += ocount;
        coverage_rows.push(json!({"variable":alias(variable),"nativeVariable":variable,"windowMinutes":0,"forecastSamples":fcount,"baselineSamples":bcount,"observationSamples":ocount}));
        if w.count > 0 {
            let paired = baseline.as_ref().map(|_| &b);
            for metric in ["mae", "rmse", "bias"] {
                pair_rows(
                    &mut pairs, variable, 0, &w, paired, metric, units, None, None,
                );
            }
            summary.push(json!({"variable":variable,"woof":metrics(&w),"baseline":paired.map(metrics),"station_ids":ids,"frozen_station_count":forecast.stations.len(),"excluded_station_count":forecast.stations.len()as u64-w.count}));
            cohort.push(json!({"variable":variable,"samples":samples}));
        }
    }
    let keys: std::collections::BTreeSet<_> = forecast
        .grids
        .iter()
        .chain(observations.grids.iter())
        .chain(baseline.iter().flat_map(|a| a.grids.iter()))
        .filter(|g| g.variable != "precipitation_total" && selected(request, &g.variable))
        .map(|g| (g.variable.clone(), g.window_minutes))
        .collect();
    for (variable, window) in keys {
        let count = |artifact: &Artifact| {
            artifact
                .grids
                .iter()
                .find(|g| g.variable == variable && g.window_minutes == window)
                .map_or(0, |g| {
                    g.values
                        .iter()
                        .filter(|v| v.is_some_and(f64::is_finite))
                        .count() as u64
                })
        };
        let (f, b, o) = (
            count(&forecast),
            baseline.as_ref().map_or(0, &count),
            count(&observations),
        );
        forecast_samples += f;
        baseline_samples += b;
        observation_samples += o;
        coverage_rows.push(json!({"variable":variable,"windowMinutes":window,"forecastSamples":f,"baselineSamples":b,"observationSamples":o}));
    }
    for fg in &forecast.grids {
        if fg.variable == "precipitation_total" || !selected(request, &fg.variable) {
            continue;
        }
        let og = observations
            .grids
            .iter()
            .find(|g| g.variable == fg.variable && g.window_minutes == fg.window_minutes);
        let Some(og) = og else {
            summary.push(json!({"variable":fg.variable,"window_minutes":fg.window_minutes,"state":"observation_variable_unavailable"}));
            continue;
        };
        let bg = baseline.as_ref().and_then(|a| {
            a.grids
                .iter()
                .find(|g| g.variable == fg.variable && g.window_minutes == fg.window_minutes)
        });
        if baseline.is_some() && bg.is_none() {
            summary.push(json!({"variable":fg.variable,"window_minutes":fg.window_minutes,"state":"baseline_variable_unavailable"}));
            continue;
        }
        if og.spec != fg.spec || bg.is_some_and(|g| g.spec != fg.spec) {
            return Err(fail(
                "gridded cohorts must share the exact verification lattice",
            ));
        }
        let n = fg
            .spec
            .nx
            .checked_mul(fg.spec.ny)
            .ok_or_else(|| fail("grid cell count overflow"))?;
        if fg.values.len() != n || og.values.len() != n || bg.is_some_and(|g| g.values.len() != n) {
            return Err(fail("gridded artifact length differs from lattice"));
        }
        let valid: Vec<_> = (0..n)
            .map(|k| {
                fg.values[k].is_some_and(f64::is_finite)
                    && og.values[k].is_some_and(f64::is_finite)
                    && bg.is_none_or(|g| g.values[k].is_some_and(f64::is_finite))
            })
            .collect();
        if !valid.iter().any(|v| *v) {
            continue;
        }
        let thresholds: Vec<f64> = if let Some(a) = request["thresholds"].as_array() {
            a.iter()
                .map(|v| v.as_f64().ok_or_else(|| fail("threshold is not numeric")))
                .collect::<Result<_>>()?
        } else if fg.variable == "precipitation" {
            vec![0.1, 1., 5., 10., 25.]
        } else {
            vec![20., 30., 40.]
        };
        let scales: Vec<f64> = if let Some(a) = request["scales_km"].as_array() {
            a.iter()
                .map(|v| v.as_f64().ok_or_else(|| fail("scale is not numeric")))
                .collect::<Result<_>>()?
        } else {
            vec![1., 3., 10., 25., 50.]
        };
        let registered = if fg.variable == "precipitation" {
            vec![0.1, 1., 5., 10., 25.]
        } else {
            vec![20., 30., 40.]
        };
        if thresholds.iter().any(|t| !registered.contains(t))
            || scales.iter().any(|s| ![1., 3., 10., 25., 50.].contains(s))
        {
            return Err(fail(
                "threshold or neighbourhood differs from frozen scoreboard-v1",
            ));
        }
        let units = if fg.variable == "precipitation" {
            "mm"
        } else {
            "dBZ"
        };
        let mut continuous = Stats::default();
        let mut continuous_base = Stats::default();
        for k in 0..n {
            if valid[k] {
                continuous.add(fg.values[k].unwrap(), og.values[k].unwrap());
                if let Some(b) = bg {
                    continuous_base.add(b.values[k].unwrap(), og.values[k].unwrap());
                }
            }
        }
        for metric in ["mae", "rmse", "bias"] {
            pair_rows(
                &mut pairs,
                &fg.variable,
                fg.window_minutes,
                &continuous,
                bg.map(|_| &continuous_base),
                metric,
                units,
                None,
                None,
            );
        }
        summary.push(json!({"variable":fg.variable,"window_minutes":fg.window_minutes,"units":units,"woof":metrics(&continuous),"baseline":bg.map(|_|metrics(&continuous_base)),"valid_cells":continuous.count,"missing_cells":n as u64-continuous.count}));
        for threshold in thresholds {
            let w = contingency(&fg.values, &og.values, &valid, threshold);
            let b = bg.map(|g| contingency(&g.values, &og.values, &valid, threshold));
            for metric in ["frequency_bias", "ets"] {
                pair_rows(
                    &mut pairs,
                    &fg.variable,
                    fg.window_minutes,
                    &w,
                    b.as_ref(),
                    metric,
                    "1",
                    Some(threshold),
                    None,
                );
            }
            summary.push(json!({"variable":fg.variable,"window_minutes":fg.window_minutes,"threshold":threshold,"units":units,"woof":categorical(&w),"baseline":b.as_ref().map(categorical),"valid_cells":w.count,"missing_cells":n as u64-w.count}));
            for scale in &scales {
                if !scale.is_finite()
                    || *scale <= 0.
                    || fg.spec.dlat.abs() < 1e-12
                    || fg.spec.dlon.abs() < 1e-12
                {
                    return Err(fail(
                        "FSS scales and geographic lattice spacing must be positive finite",
                    ));
                }
                let dy = 6371008.8 * fg.spec.dlat.to_radians().abs();
                let dxs: Vec<f64> = (0..fg.spec.ny)
                    .map(|j| {
                        6371008.8
                            * fg.spec.dlon.to_radians().abs()
                            * (fg.spec.south + j as f64 * fg.spec.dlat)
                                .to_radians()
                                .cos()
                                .abs()
                    })
                    .collect();
                let max_cell = dxs.iter().copied().fold(dy, f64::max);
                if scale * 1000. < max_cell {
                    summary.push(json!({"variable":fg.variable,"window_minutes":fg.window_minutes,"threshold":threshold,"scale_km":scale,"state":"below_lattice_resolution","largest_pixel_width_m":max_cell}));
                    continue;
                }
                let w = fss(&fg.values, &og.values, &valid, &fg.spec, threshold, *scale);
                let b = bg.map(|g| fss(&g.values, &og.values, &valid, &fg.spec, threshold, *scale));
                let actual = *scale;
                pair_rows(
                    &mut pairs,
                    &fg.variable,
                    fg.window_minutes,
                    &w,
                    b.as_ref(),
                    "fss",
                    "1",
                    Some(threshold),
                    Some(actual),
                );
                let hy =
                    (((scale * 1000. / dy - 1.) / 2.).round().max(0.) as usize).min(fg.spec.ny);
                let widths: Vec<f64> = dxs
                    .iter()
                    .map(|dx| {
                        let hx = (((scale * 1000. / dx - 1.) / 2.).round().max(0.) as usize)
                            .min(fg.spec.nx);
                        (2 * hx + 1) as f64 * dx
                    })
                    .collect();
                if let Some(pair) = pairs.last_mut() {
                    pair["windowHeightKm"] = json!((2 * hy + 1) as f64 * dy / 1000.);
                    pair["windowWidthMinKm"] =
                        json!(widths.iter().copied().fold(f64::INFINITY, f64::min) / 1000.);
                    pair["windowWidthMaxKm"] =
                        json!(widths.iter().copied().fold(0., f64::max) / 1000.);
                }
                summary.push(json!({"variable":fg.variable,"window_minutes":fg.window_minutes,"threshold":threshold,"scale_km":actual,"actual_box_height_m":(2*hy+1)as f64*dy,"actual_box_width_m_range":[widths.iter().copied().fold(f64::INFINITY,f64::min),widths.iter().copied().fold(0.,f64::max)],"woof_fss":ratio(w.fractions_squared_error.unwrap(),w.fractions_reference.unwrap()).map(|d|1.-d),"baseline_fss":b.as_ref().and_then(|s|ratio(s.fractions_squared_error.unwrap(),s.fractions_reference.unwrap()).map(|d|1.-d)),"valid_cells":w.count}));
            }
        }
        cohort
            .push(json!({"variable":fg.variable,"window_minutes":fg.window_minutes,"valid":valid}));
    }
    let comparison = digest(&serde_json::to_vec(&(
        METHOD,
        &forecast.support_hash,
        &forecast.valid_time,
        &cohort,
    ))?);
    let result = json!({"schema":"woof.scoreboard.scores.v1","methodId":METHOD,"validTime":forecast.valid_time,"domain":forecast.domain,"leadSeconds":forecast_lead,"supportHash":forecast.support_hash,"comparison_support_hash":comparison,"paired":baseline.is_some(),"baseline_policy":policy,"baseline_init_time":baseline.as_ref().map(|b|b.init_time.clone()),"baseline_lead_seconds":baseline_lead,"baseline_cycle_gap_seconds":baseline_gap,"cycle_gap_seconds":baseline_gap,"forecast_sha256":forecast.source_sha256,"observations_sha256":observations.source_sha256,"baseline_sha256":baseline.as_ref().map(|a|a.source_sha256.clone()),"pairs":pairs,"summary":summary,"dropped_by_variable":dropped,"gross_error_rejected_by_variable":gross_rejected,"gross_error_limits":gross_error_limits_json(),"coverage":{"forecastSamples":forecast_samples,"baselineSamples":baseline_samples,"observationSamples":observation_samples},"coverage_by_variable":coverage_rows});
    if let Some(path) = request["output"].as_str() {
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fss_handles_missing_and_dry() {
        let spec = GridSpec {
            south: 0.,
            west: 0.,
            dlat: 0.01,
            dlon: 0.01,
            nx: 2,
            ny: 2,
            dx_m: 1000.,
        };
        let a = vec![Some(1.), None, Some(0.), Some(0.)];
        let b = vec![Some(0.), Some(0.), Some(1.), Some(0.)];
        let s = fss(&a, &b, &[true, false, true, true], &spec, 0.5, 3.);
        assert!(
            (1. - s.fractions_squared_error.unwrap() / s.fractions_reference.unwrap() - 1.).abs()
                < 1e-12
        );
        let d = vec![Some(0.); 4];
        let s = fss(&d, &d, &[true; 4], &spec, 1., 1.);
        assert_eq!(s.fractions_reference, Some(0.));
    }
    #[test]
    fn contingency_undefined_is_null() {
        let s = contingency(&[Some(0.)], &[Some(0.)], &[true], 1.);
        assert!(categorical(&s)["ets"].is_null());
    }
    #[test]
    fn a_station_pressure_far_from_every_model_is_rejected() {
        // a mis-keyed report 300 hPa off: rejected, paired or not
        assert!(gross_error("pressure_surface", 70_000.0, 100_000.0, Some(100_050.0)));
        assert!(gross_error("pressure_surface", 70_000.0, 100_000.0, None));
        // an ordinary forecast error is scored
        assert!(!gross_error("pressure_surface", 100_000.0, 100_150.0, Some(99_900.0)));
        // close to ONE model: kept, so the rule cannot favour either side
        assert!(!gross_error("pressure_surface", 94_000.0, 100_000.0, Some(94_100.0)));
        // variables without a limit are untouched
        assert!(!gross_error("temperature_2m", 200.0, 300.0, Some(300.0)));
        assert_eq!(gross_error_limits_json()["pressure_surface"], json!(5_000.0));
    }
}
