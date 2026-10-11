use crate::support::{Locator, sample};
use crate::*;
use rustwx_core::{CanonicalField as F, FieldSelector as S, ModelId};
use std::io::Read;

struct Native {
    locator: Locator,
    fields: BTreeMap<String, Vec<f64>>,
    valid_time: Option<String>,
    init_time: Option<String>,
    init_source: String,
}
fn plane(f: &netcrust::File, name: &str) -> Result<(usize, usize, Vec<f64>)> {
    let a = f.read_array_f64_first_record_or_all(name)?;
    let shape = a.shape();
    if shape.len() < 2 || shape[..shape.len() - 2].iter().product::<usize>() != 1 {
        return Err(fail(format!("{name} is not a single 2-D record")));
    }
    let (nx, ny) = (shape[shape.len() - 1], shape[shape.len() - 2]);
    let mut values = a.into_values();
    mask_encoded_missing(f, name, &mut values);
    Ok((nx, ny, values))
}
fn mask_encoded_missing(f: &netcrust::File, name: &str, values: &mut [f64]) {
    if let Some(var) = f.variable(name) {
        for key in ["_FillValue", "missing_value"] {
            if let Some(fill) = var.attribute(key).and_then(|a| a.as_f64()) {
                for v in &mut *values {
                    if *v == fill {
                        *v = f64::NAN;
                    }
                }
            }
        }
    }
}
fn wrf(path: &str, request: &Value) -> Result<Native> {
    let f = netcrust::File::open(path)?;
    let times = f.read_strings("Times")?;
    let valid_time = times
        .first()
        .map(|s| time(s).map(iso))
        .transpose()?
        .ok_or_else(|| {
            fail("WRF Times is empty, so the frame cannot be placed on the forecast clock")
        })?;
    let valid_stamp = time(&valid_time)?;
    if valid_stamp != time(get_str(request, "valid_time")?)? {
        return Err(fail("requested valid_time differs from WRF Times"));
    }
    let (mut init, mut init_source) = (None, String::new());
    for name in ["SIMULATION_START_DATE", "START_DATE"] {
        if let Some(text) = f
            .attribute(name)
            .and_then(|a| a.as_string().map(str::to_string))
        {
            init = Some(time(text.trim_matches('\0').trim())?);
            init_source = name.into();
            break;
        }
    }
    if init.is_none() && f.variable("XTIME").is_some() {
        let elapsed = f
            .read_array_f64_first_record_or_all("XTIME")?
            .values()
            .first()
            .copied()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .ok_or_else(|| {
                fail("WRF XTIME is missing or negative, so the forecast cycle cannot be inferred")
            })?;
        let units = f
            .variable("XTIME")
            .and_then(|v| {
                v.attribute("units")
                    .and_then(|a| a.as_string().map(str::to_string))
            })
            .unwrap_or_else(|| "minutes".into());
        let scale = if units.starts_with("seconds") {
            1.0
        } else if units.starts_with("minutes") {
            60.0
        } else if units.starts_with("hours") {
            3600.0
        } else if units.starts_with("days") {
            86400.0
        } else {
            return Err(fail("WRF XTIME units cannot locate the forecast cycle"));
        };
        let elapsed_seconds = (elapsed * scale).round();
        if elapsed_seconds > i64::MAX as f64 {
            return Err(fail("WRF XTIME overflows the forecast clock"));
        }
        init = Some(valid_stamp - elapsed_seconds as i64);
        init_source = "Times minus XTIME".into();
    }
    if let Some(supplied) = request["init_time"].as_str() {
        let supplied = time(supplied)?;
        if init.is_some_and(|native| native != supplied) {
            return Err(fail(
                "requested init_time differs from the native WRF simulation start",
            ));
        }
        if init.is_none() {
            init = Some(supplied);
            init_source = "request, native start metadata absent".into();
        }
    }
    let init=init.ok_or_else(||fail("WRF carries no simulation start or usable XTIME, so an unresolved latest run cannot bind a forecast cycle"))?;
    if valid_stamp < init {
        return Err(fail("native WRF valid time precedes the simulation start"));
    }
    let (nx, ny, lat) = plane(&f, "XLAT")?;
    let (_, _, lon) = plane(&f, "XLONG")?;
    let mut fields = BTreeMap::new();
    for (name, var) in [
        ("temperature_2m_raw", "T2"),
        ("mixing_ratio_2m", "Q2"),
        ("u_wind_10m", "U10"),
        ("v_wind_10m", "V10"),
        ("pressure_surface_raw", "PSFC"),
        ("terrain_m", "HGT"),
    ] {
        if f.variable(var).is_some() {
            let (fnx, fny, v) = plane(&f, var)?;
            if (fnx, fny) != (nx, ny) {
                return Err(fail(format!("{var} is not on the mass grid")));
            }
            fields.insert(name.into(), v);
        }
    }
    if fields.contains_key("u_wind_10m") && fields.contains_key("v_wind_10m") {
        let u = fields.remove("u_wind_10m").unwrap();
        let v = fields.remove("v_wind_10m").unwrap();
        if f.variable("COSALPHA").is_some() && f.variable("SINALPHA").is_some() {
            let (_, _, c) = plane(&f, "COSALPHA")?;
            let (_, _, s) = plane(&f, "SINALPHA")?;
            if c.len() != u.len() || s.len() != u.len() {
                return Err(fail("wind rotation coefficients differ from mass grid"));
            }
            fields.insert(
                "u_wind_10m".into(),
                (0..u.len()).map(|k| u[k] * c[k] - v[k] * s[k]).collect(),
            );
            fields.insert(
                "v_wind_10m".into(),
                (0..u.len()).map(|k| v[k] * c[k] + u[k] * s[k]).collect(),
            );
        } else if f
            .attribute("MAP_PROJ")
            .and_then(|a| a.as_f64())
            .is_some_and(|v| v == 0.0 || v == 6.0)
        {
            fields.insert("u_wind_10m".into(), u);
            fields.insert("v_wind_10m".into(), v);
        }
        // A projected wind with no orientation is absent, never silently east.
    }
    if let (Some(q), Some(p)) = (
        fields.get("mixing_ratio_2m"),
        fields.get("pressure_surface_raw"),
    ) {
        let td = q.iter().zip(p).map(|(q, p)| dewpoint(*q, *p)).collect();
        fields.insert("dewpoint_2m".into(), td);
    }
    if f.variable("RAINNC").is_some() {
        let (_, _, mut rain) = plane(&f, "RAINNC")?;
        if f.variable("RAINC").is_some() {
            let (_, _, r) = plane(&f, "RAINC")?;
            for (k, v) in rain.iter_mut().enumerate() {
                *v += r[k];
            }
        }
        let bucket = f
            .attribute("BUCKET_MM")
            .and_then(|a| a.as_f64())
            .unwrap_or(0.);
        if bucket > 0. {
            for name in ["I_RAINNC", "I_RAINC"] {
                if f.variable(name).is_some() {
                    let (_, _, r) = plane(&f, name)?;
                    for (k, v) in rain.iter_mut().enumerate() {
                        *v += r[k] * bucket;
                    }
                }
            }
        }
        fields.insert("precipitation_total".into(), rain);
    }
    if f.variable("REFL_10CM").is_some() {
        let a = f.read_array_f64_first_record_or_all("REFL_10CM")?;
        let mut values = a.into_values();
        mask_encoded_missing(&f, "REFL_10CM", &mut values);
        if values.len() % (nx * ny) != 0 {
            return Err(fail("reflectivity volume shape mismatch"));
        }
        let mut composite = vec![f64::NEG_INFINITY; nx * ny];
        for z in values.chunks(nx * ny) {
            for (k, v) in z.iter().enumerate() {
                if v.is_finite() {
                    composite[k] = composite[k].max(*v);
                }
            }
        }
        fields.insert("reflectivity".into(), composite);
    }
    Ok(Native {
        locator: Locator::new(nx, ny, lat, lon)?,
        fields,
        valid_time: Some(valid_time),
        init_time: Some(iso(init)),
        init_source,
    })
}
pub fn dewpoint(q: f64, p: f64) -> f64 {
    if !q.is_finite() || q < 0. || p <= 0. {
        return f64::NAN;
    }
    let e = q * p / (0.622 + q);
    if e <= 0. {
        return f64::NAN;
    }
    let l = (e / 611.2).ln();
    273.15 + 243.5 * l / (17.67 - l)
}
fn grib(path: &str, request: &Value) -> Result<Native> {
    let requested_init = time(get_str(request, "init_time")?)?;
    let mut bytes = std::fs::read(path)?;
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut inflated = vec![];
        flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut inflated)?;
        bytes = inflated;
    }
    let mut parsed = grib_core::grib2::Grib2File::from_bytes(&bytes)?;
    let wanted_window = request["window_minutes"].as_u64().unwrap_or(0);
    if wanted_window > 0 || request["include_cumulative"].as_bool() == Some(true) {
        let mut filtered = vec![];
        let mut offset = 0usize;
        for message in &parsed.messages {
            if bytes.get(offset..offset + 4) != Some(b"GRIB") {
                return Err(fail("GRIB subset is not a contiguous message stream"));
            }
            let length =
                u64::from_be_bytes(bytes[offset + 8..offset + 16].try_into().unwrap()) as usize;
            let precip = grib_surface_name(message) == Some("precipitation_total");
            let selected_precip = if !request["previous_input"].is_null()
                || request["include_cumulative"].as_bool() == Some(true)
            {
                message.product.forecast_time == 0
            } else {
                message.product.statistical_time_range_seconds() == Some(wanted_window * 60)
            };
            if !precip || selected_precip {
                filtered.extend_from_slice(&bytes[offset..offset + length]);
            }
            offset += length;
        }
        bytes = filtered;
        parsed = grib_core::grib2::Grib2File::from_bytes(&bytes)?;
    }
    let model: ModelId = serde_json::from_value(request["model"].clone()).map_err(|_| {
        fail("GRIB model must name a Rust registry model, for example Hrrr, Gfs, Rap, Nam")
    })?;
    let specs = [
        ("temperature_2m_raw", S::height_agl(F::Temperature, 2)),
        ("dewpoint_2m", S::height_agl(F::Dewpoint, 2)),
        ("u_wind_10m", S::height_agl(F::UWind, 10)),
        ("v_wind_10m", S::height_agl(F::VWind, 10)),
        ("pressure_surface_raw", S::surface(F::Pressure)),
        ("terrain_m", S::surface(F::GeopotentialHeight)),
        ("precipitation_total", S::surface(F::TotalPrecipitation)),
        (
            "reflectivity",
            S::entire_atmosphere(F::CompositeReflectivity),
        ),
    ];
    let selected = rustwx_io::extract_field_values_partial_from_model_bytes_at_forecast_hour(
        model,
        &bytes,
        None,
        &specs.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
        request["forecast_hour"].as_u64().map(|v| v as u16),
    )?;
    let first = selected
        .extracted
        .first()
        .ok_or_else(|| fail("no scoring variables found in GRIB"))?;
    let grid = &selected.grids[first.grid_index].grid;
    let locator = Locator::new(
        grid.shape.nx,
        grid.shape.ny,
        grid.lat_deg.iter().map(|v| *v as f64).collect(),
        grid.lon_deg.iter().map(|v| *v as f64).collect(),
    )?;
    let mut fields = BTreeMap::new();
    for field in &selected.extracted {
        if field.grid_index != first.grid_index {
            return Err(fail("scoring GRIB fields use different native grids"));
        }
        let name = specs.iter().find(|(_, s)| *s == field.selector).unwrap().0;
        fields.insert(
            name.into(),
            field.values.iter().map(|v| *v as f64).collect(),
        );
    }
    if request["include_cumulative"].as_bool() == Some(true)
        && request["forecast_hour"].as_u64() == Some(0)
        && !fields.contains_key("precipitation_total")
    {
        fields.insert(
            "precipitation_total".into(),
            vec![0.0; locator.nx * locator.ny],
        );
    }
    // Validate the byte metadata, including accumulation interval. A subset
    // file is still a real GRIB file and must not inherit caller timing.
    let requested = time(get_str(request, "valid_time")?)?;
    for message in &parsed.messages {
        let p = &message.product;
        if let Some(name) = grib_surface_name(message) {
            if let Some(init) = request["init_time"].as_str() {
                if message.reference_time.and_utc().timestamp() != time(init)? {
                    return Err(fail(
                        "GRIB reference cycle differs from requested init_time",
                    ));
                }
            }
            let units = match p.time_range_unit {
                0 => 60,
                1 => 3600,
                2 => 86400,
                10 => 10800,
                11 => 21600,
                12 => 43200,
                13 => 1,
                _ => return Err(fail("calendar-relative GRIB lead cannot be scored")),
            };
            let valid = message.reference_time.and_utc().timestamp()
                + p.forecast_time as i64 * units as i64
                + p.statistical_time_range_seconds().unwrap_or(0) as i64;
            if valid != requested {
                return Err(fail(format!(
                    "{name} GRIB valid time {} differs from requested {}",
                    iso(valid),
                    iso(requested)
                )));
            }
            if name == "precipitation_total"
                && request["previous_input"].is_null()
                && request["include_cumulative"].as_bool() != Some(true)
            {
                let minutes = p.statistical_time_range_seconds().unwrap_or(0) / 60;
                let wanted = request["window_minutes"].as_u64().unwrap_or(0);
                if wanted > 0 && minutes != wanted {
                    fields.remove("precipitation_total");
                }
            }
        }
    }
    Ok(Native {
        locator,
        fields,
        valid_time: Some(iso(requested)),
        init_time: Some(iso(requested_init)),
        init_source: "GRIB reference cycle validated".into(),
    })
}
fn grib_surface_name(m: &grib_core::grib2::Grib2Message) -> Option<&'static str> {
    let p = &m.product;
    match (
        m.discipline,
        p.parameter_category,
        p.parameter_number,
        p.level_type,
    ) {
        (0, 0, 0, 103)
        | (0, 0, 6, 103)
        | (0, 2, 2, 103)
        | (0, 2, 3, 103)
        | (0, 3, 0, 1)
        | (0, 3, 5, 1) => Some("surface"),
        (0, 1, 8, 1) => Some("precipitation_total"),
        _ => None,
    }
}
fn native(path: &str, request: &Value) -> Result<Native> {
    match request["format"].as_str().unwrap_or("wrf") {
        "wrf" => wrf(path, request),
        "grib" => grib(path, request),
        _ => Err(fail("extract format must be wrf or grib")),
    }
}
fn precipitation_at(current: &Native, previous: &Native, lat: f64, lon: f64) -> Option<f64> {
    let a = sample(
        current.fields.get("precipitation_total")?,
        &current.locator.locate(lat, lon)?,
    )?;
    let b = sample(
        previous.fields.get("precipitation_total")?,
        &previous.locator.locate(lat, lon)?,
    )?;
    crate::temporal::difference(a, b)
}
pub fn run(request: &Value) -> Result<Value> {
    let path = get_str(request, "input")?;
    let bytes = std::fs::read(path)?;
    let stations = load_stations(request)?;
    let grid: Option<GridSpec> = serde_json::from_value(request["grid"].clone())?;
    let mut artifact = new_artifact(request, &bytes, stations, &grid)?;
    let mut data = native(path, request)?;
    if let Some(init) = data.init_time.as_ref() {
        artifact.init_time = Some(init.clone());
        let seconds = time(&artifact.valid_time)? - time(init)?;
        if seconds < 0 || seconds / 60 > u32::MAX as i64 {
            return Err(fail(
                "native forecast clock cannot define an unsigned whole-minute lead",
            ));
        }
        artifact.lead_minutes = (seconds / 60) as u32;
        artifact.lead_seconds = seconds as u64;
        artifact.diagnostics = json!({"init_time_source":data.init_source,"native_init_time":init,"native_lead_minutes":artifact.lead_minutes});
    }
    if let Some(values) = data.fields.get_mut("reflectivity") {
        for v in values {
            *v = 10f64.powf(*v / 10.);
        }
    }
    let mut previous_data = None;
    if let Some(previous) = request["previous_input"].as_str() {
        let window = request["window_minutes"]
            .as_u64()
            .ok_or_else(|| fail("previous_input requires exact window_minutes"))?;
        let mut previous_request = request.clone();
        previous_request["init_time"] = json!(artifact.init_time);
        let previous_time = time(&artifact.valid_time)? - window as i64 * 60;
        previous_request["valid_time"] = json!(iso(previous_time));
        if let Some(h) = request["forecast_hour"].as_u64() {
            previous_request["forecast_hour"] = json!(
                h.checked_sub(window / 60)
                    .ok_or_else(|| fail("precipitation interval precedes cycle"))?
            );
        }
        let prev = native(previous, &previous_request)?;
        let previous_bytes = std::fs::read(previous)?;
        let current_sha256 = artifact.source_sha256.clone();
        let previous_sha256 = digest(&previous_bytes);
        artifact.source_sha256 = digest(&serde_json::to_vec(
            &json!({"current_sha256":current_sha256,"previous_sha256":previous_sha256,"window_minutes":window}),
        )?);
        artifact.diagnostics = json!({"current_sha256":current_sha256,"previous_sha256":previous_sha256,"accumulation_window_minutes":window,"accumulation_start":iso(previous_time),"accumulation_end":artifact.valid_time,"init_time_source":data.init_source,"native_init_time":artifact.init_time,"native_lead_minutes":artifact.lead_minutes,"accumulation_spatial_operator":"independent_endpoint_interpolation_on_fixed_support"});
        previous_data = Some(prev);
    } else if request["format"] == "grib"
        && request["window_minutes"].as_u64().unwrap_or(0) > 0
        && request["include_cumulative"].as_bool() != Some(true)
    {
        if let Some(r) = data.fields.remove("precipitation_total") {
            data.fields.insert("precipitation".into(), r);
        }
    }
    let max_elevation = 500.0;
    for station in &artifact.stations {
        let Some(s) = data.locator.locate(station.lat, station.lon) else {
            artifact
                .drops
                .push(json!({"station_id":station.station_id,"reason":"outside_domain"}));
            continue;
        };
        let terrain = data.fields.get("terrain_m").and_then(|v| sample(v, &s));
        let mut values: BTreeMap<String, f64> = data
            .fields
            .iter()
            .filter_map(|(name, v)| {
                if name == "mixing_ratio_2m"
                    || name == "terrain_m"
                    || (name == "precipitation_total"
                        && request["include_cumulative"].as_bool() != Some(true))
                {
                    None
                } else {
                    sample(v, &s)
                        .and_then(|v| {
                            if name == "reflectivity" {
                                let v = 10. * v.log10();
                                v.is_finite().then_some(v)
                            } else {
                                Some(v)
                            }
                        })
                        .map(|v| (name.clone(), v))
                }
            })
            .collect();
        if let Some(previous) = previous_data.as_ref() {
            if let Some(rain) = precipitation_at(&data, previous, station.lat, station.lon) {
                values.insert("precipitation".into(), rain);
            }
        }
        if let Some(terrain) = terrain {
            let dz = station.elevation_m - terrain;
            if dz.abs() <= max_elevation {
                if let Some(t) = values.get("temperature_2m_raw").copied() {
                    values.insert("temperature_2m".into(), t - 0.0065 * dz);
                    if let Some(p) = values.get("pressure_surface_raw").copied() {
                        values.insert(
                            "pressure_surface".into(),
                            p * (-9.80665 * dz / (287.05 * (t - 0.00325 * dz))).exp(),
                        );
                    }
                }
            } else {
                artifact.drops.push(json!({"station_id":station.station_id,"reason":"terrain_mismatch_adjusted_only","elevation_difference_m":dz}));
            }
        } else {
            artifact.drops.push(json!({"station_id":station.station_id,"reason":"missing_model_terrain_adjusted_only"}));
        }
        if let (Some(u), Some(v)) = (values.get("u_wind_10m"), values.get("v_wind_10m")) {
            values.insert("wind_speed_10m".into(), u.hypot(*v));
        }
        artifact.points.push(Point {
            station_id: station.station_id.clone(),
            valid_time: artifact.valid_time.clone(),
            observation_time: None,
            observation_times: BTreeMap::new(),
            observation_sources: BTreeMap::new(),
            values,
            terrain_m: terrain,
        });
    }
    if let Some(spec) = grid {
        if spec.nx.checked_mul(spec.ny).is_none() || spec.nx * spec.ny > 4_000_000 {
            return Err(fail("requested verification grid exceeds 4 million cells"));
        }
        let stencils: Vec<_> = (0..spec.ny)
            .flat_map(|j| (0..spec.nx).map(move |i| (j, i)))
            .map(|(j, i)| {
                data.locator.locate(
                    spec.south + j as f64 * spec.dlat,
                    spec.west + i as f64 * spec.dlon,
                )
            })
            .collect();
        for name in ["precipitation", "reflectivity", "precipitation_total"] {
            if name == "precipitation" {
                if let Some(previous) = previous_data.as_ref() {
                    if data.fields.contains_key("precipitation_total")
                        && previous.fields.contains_key("precipitation_total")
                    {
                        let values = (0..spec.ny)
                            .flat_map(|j| (0..spec.nx).map(move |i| (j, i)))
                            .map(|(j, i)| {
                                precipitation_at(
                                    &data,
                                    previous,
                                    spec.south + j as f64 * spec.dlat,
                                    spec.west + i as f64 * spec.dlon,
                                )
                            })
                            .collect();
                        artifact.grids.push(Grid {
                            spec: spec.clone(),
                            variable: name.into(),
                            window_minutes: request["window_minutes"].as_u64().unwrap_or(0) as u32,
                            values,
                        });
                    }
                    continue;
                }
            }
            if name == "precipitation_total"
                && request["include_cumulative"].as_bool() != Some(true)
            {
                continue;
            }
            if let Some(values) = data.fields.get(name) {
                artifact.grids.push(Grid {
                    spec: spec.clone(),
                    variable: name.into(),
                    window_minutes: if name == "precipitation" {
                        request["window_minutes"].as_u64().unwrap_or(0) as u32
                    } else {
                        0
                    },
                    values: stencils
                        .iter()
                        .map(|s| {
                            s.as_ref().and_then(|s| sample(values, s)).and_then(|v| {
                                if name == "reflectivity" {
                                    let v = 10. * v.log10();
                                    v.is_finite().then_some(v)
                                } else {
                                    Some(v)
                                }
                            })
                        })
                        .collect(),
                });
            }
        }
    }
    let _ = data.valid_time;
    write_artifact(&artifact, get_str(request, "output")?)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moving_rain_uses_physical_locations() {
        let make = |longitude: Vec<f64>, rain: Vec<f64>| Native {
            locator: Locator::new(2, 2, vec![0., 0., 1., 1.], longitude).unwrap(),
            fields: BTreeMap::from([("precipitation_total".into(), rain)]),
            valid_time: None,
            init_time: None,
            init_source: String::new(),
        };
        let previous = make(vec![0., 1., 0., 1.], vec![1., 2., 1., 2.]);
        let current = make(vec![0.5, 1.5, 0.5, 1.5], vec![4.5, 5.5, 4.5, 5.5]);
        assert!((precipitation_at(&current, &previous, 0.5, 0.75).unwrap() - 3.).abs() < 1e-12);
        assert!(precipitation_at(&current, &previous, 0.5, 1.25).is_none());
        assert!(precipitation_at(&current, &previous, 0.5, 0.25).is_none());
    }
}
