use crate::*;
use std::io::Read;
const KNOT: f64 = 0.5144444444444445;
#[derive(Clone, Serialize, Deserialize)]
struct Report {
    station: Station,
    t: i64,
    values: BTreeMap<String, f64>,
    source: String,
    rejected: std::collections::BTreeSet<String>,
}

#[derive(Serialize, Deserialize)]
struct ObservationArchive {
    schema: String,
    method_id: String,
    source_sha256: String,
    diagnostics: Value,
    reports: Vec<Report>,
}

// RFC 4180 field escaping, including embedded newlines and doubled quotes.
fn csv(text: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = vec![];
    let mut row = vec![];
    let mut field = String::new();
    let mut quote = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if quote && chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quote = !quote;
                }
            }
            ',' if !quote => {
                row.push(std::mem::take(&mut field));
            }
            '\n' if !quote => {
                row.push(std::mem::take(&mut field));
                if !row.iter().all(String::is_empty) {
                    rows.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
            }
            '\r' if !quote => {}
            _ => field.push(c),
        }
    }
    if quote {
        return Err(fail("unterminated quoted observation CSV field"));
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}
fn num(s: &str) -> Option<f64> {
    s.parse::<f64>().ok().filter(|v| v.is_finite())
}
fn screen(values: &mut BTreeMap<String, f64>) {
    values.retain(|name, v| match name.as_str() {
        "temperature_2m" | "temperature_2m_raw" | "dewpoint_2m" => (233.15..=328.15).contains(v),
        "wind_speed_10m" => (0.0..=75.0).contains(v),
        "pressure_surface" => (30000.0..=110000.0).contains(v),
        _ => v.is_finite(),
    });
    if let (Some(t), Some(td)) = (values.get("temperature_2m"), values.get("dewpoint_2m")) {
        if td > t {
            values.remove("dewpoint_2m");
        }
    }
}
fn pressure(altimeter_pa: f64, elevation: f64) -> f64 {
    altimeter_pa * (1.0 - 0.0065 * elevation / 288.15).powf(5.255877)
}
fn radar_value(v: f64) -> f64 {
    if !v.is_finite() || v <= -900.0 || v > 1e20 {
        f64::NAN
    } else if v <= -90.0 {
        -35.0
    } else {
        v
    }
}
fn reports_dwd(bytes: &[u8]) -> Result<(Vec<Report>, Value)> {
    let mut decoded = vec![];
    let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
        flate2::read::GzDecoder::new(bytes)
            .take(128 * 1024 * 1024 + 1)
            .read_to_end(&mut decoded)?;
        if decoded.len() > 128 * 1024 * 1024 {
            return Err(fail("DWD SYNOP archive exceeds bounded expansion"));
        }
        decoded.as_slice()
    } else {
        bytes
    };
    let root: Value = serde_json::from_slice(bytes)?;
    if root["type"] != "FeatureCollection" {
        return Err(fail("DWD SYNOP requires decoded FeatureCollection"));
    }
    let features = root["features"]
        .as_array()
        .ok_or_else(|| fail("DWD SYNOP features missing"))?;
    let mut reports = vec![];
    let mut invalid = 0u64;
    let mut heights_excluded = 0u64;
    for feature in features {
        let coords = &feature["geometry"]["coordinates"];
        let (Some(lon), Some(lat), Some(elevation_m)) =
            (coords[0].as_f64(), coords[1].as_f64(), coords[2].as_f64())
        else {
            invalid += 1;
            continue;
        };
        if lat.abs() > 90. || lon.abs() > 360. || elevation_m.abs() > 10000. {
            invalid += 1;
            continue;
        }
        let props = feature["properties"]
            .as_object()
            .ok_or_else(|| fail("DWD SYNOP properties missing"))?;
        let mut entries: Vec<_> = props
            .iter()
            .filter_map(|(key, v)| {
                key.strip_prefix("data_")
                    .and_then(|n| n.parse::<usize>().ok())
                    .map(|n| (n, v))
            })
            .collect();
        entries.sort_by_key(|(n, _)| *n);
        let value = |name: &str| {
            entries
                .iter()
                .find_map(|(_, v)| (v["shortname"].as_str() == Some(name)).then_some(&v["value"]))
        };
        let integer = |name: &str| value(name).and_then(Value::as_u64);
        let (Some(year), Some(month), Some(day), Some(hour), Some(minute)) = (
            integer("year"),
            integer("month"),
            integer("day"),
            integer("hour"),
            integer("minute"),
        ) else {
            invalid += 1;
            continue;
        };
        let Some(date) = chrono::NaiveDate::from_ymd_opt(year as i32, month as u32, day as u32)
        else {
            invalid += 1;
            continue;
        };
        let Some(t) = date.and_hms_opt(hour as u32, minute as u32, 0) else {
            invalid += 1;
            continue;
        };
        let local = value("wigosLocalIdentifierCharacter")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(String::from)
            .or_else(|| {
                Some(format!(
                    "{:02}{:03}",
                    integer("blockNumber")?,
                    integer("stationNumber")?
                ))
            });
        let Some(local) = local else {
            invalid += 1;
            continue;
        };
        let station_id = format!(
            "DWD_SYNOP_{}",
            local
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect::<String>()
        );
        let mut values = BTreeMap::new();
        let mut offered = std::collections::BTreeSet::<String>::new();
        let mut height = None::<f64>;
        let mut speed = None::<f64>;
        let mut direction = None::<f64>;
        for (_, entry) in entries {
            let name = entry["shortname"].as_str().unwrap_or("");
            let numeric = entry["value"].as_f64().filter(|v| v.is_finite());
            let unit = entry["unit"].as_str().unwrap_or("");
            match name {
                "heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform" => {
                    height = if unit == "m" { numeric } else { None }
                }
                "airTemperature" | "dewpointTemperature" => {
                    let variable = if name == "airTemperature" {
                        "temperature_2m"
                    } else {
                        "dewpoint_2m"
                    };
                    offered.insert(variable.into());
                    if height.is_some_and(|h| (h - 2.).abs() <= 0.01) && unit == "K" {
                        if let Some(v) = numeric {
                            values.entry(variable.into()).or_insert(v);
                        }
                    } else {
                        heights_excluded += 1;
                    }
                }
                "windSpeed" => {
                    offered.extend(
                        ["wind_speed_10m", "u_wind_10m", "v_wind_10m"]
                            .into_iter()
                            .map(String::from),
                    );
                    if height.is_some_and(|h| (h - 10.).abs() <= 0.01) && unit == "m/s" {
                        speed = numeric;
                    } else {
                        heights_excluded += 1;
                    }
                }
                "windDirection" => {
                    offered.extend(["u_wind_10m", "v_wind_10m"].into_iter().map(String::from));
                    if height.is_some_and(|h| (h - 10.).abs() <= 0.01)
                        && matches!(unit, "deg" | "degree")
                    {
                        direction = numeric.filter(|v| (0.0..=360.0).contains(v));
                    } else {
                        heights_excluded += 1;
                    }
                }
                "nonCoordinatePressure" => {
                    offered.insert("pressure_surface".into());
                    if unit == "Pa" {
                        if let Some(v) = numeric {
                            values.entry("pressure_surface".into()).or_insert(v);
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(s) = speed.filter(|v| (0.0..=75.0).contains(v)) {
            values.insert("wind_speed_10m".into(), s);
            if let Some(d) = direction {
                values.insert("u_wind_10m".into(), -s * d.to_radians().sin());
                values.insert("v_wind_10m".into(), -s * d.to_radians().cos());
            }
        }
        screen(&mut values);
        let rejected = offered
            .into_iter()
            .filter(|name| !values.contains_key(name))
            .collect();
        reports.push(Report {
            station: Station {
                station_id,
                lat,
                lon,
                elevation_m,
            },
            t: t.and_utc().timestamp(),
            values,
            source: "dwd-synop-germany".into(),
            rejected,
        });
    }
    Ok((
        reports,
        json!({"source":"dwd-synop-germany","attribution":"© Deutscher Wetterdienst","license_url":"https://kunden.dwd.de/ankonda/info/copyright.jsf","scope":"German DWD decoded SYNOP bulletins only","features":features.len(),"invalid_metadata_features":invalid,"unsupported_sensor_height_fields":heights_excluded,"temperature_height_required_m":2,"wind_height_required_m":10}),
    ))
}
fn reports_csv(bytes: &[u8], format: &str, stations: &[Station]) -> Result<Vec<Report>> {
    let mut unwrapped = vec![];
    let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
        flate2::read::GzDecoder::new(bytes).read_to_end(&mut unwrapped)?;
        unwrapped.as_slice()
    } else {
        bytes
    };
    let text = std::str::from_utf8(bytes)?;
    let rows = csv(text)?;
    let head = rows
        .iter()
        .position(|r| r.iter().any(|c| c == "station" || c == "station_id"))
        .ok_or_else(|| fail("observation CSV has no station header"))?;
    let columns = &rows[head];
    let get = |row: &Vec<String>, names: &[&str]| -> String {
        for name in names {
            if let Some(i) = columns.iter().position(|c| c == name) {
                return row.get(i).cloned().unwrap_or_default();
            }
        }
        String::new()
    };
    let mut out = vec![];
    for row in &rows[head + 1..] {
        let id = get(row, &["station", "station_id"]);
        if id.is_empty() {
            continue;
        }
        let stamp = get(
            row,
            &["valid", "valid(UTC)", "valid_time", "observation_time"],
        );
        let Ok(t) = time(&stamp) else {
            continue;
        };
        let raw = if let (Some(lat), Some(lon), Some(elevation_m)) = (
            num(&get(row, &["lat", "latitude_deg", "latitude"])),
            num(&get(row, &["lon", "longitude_deg", "longitude"])),
            num(&get(row, &["elevation", "elevation_m", "elev"])),
        ) {
            Some(Station {
                station_id: id.clone(),
                lat,
                lon,
                elevation_m,
            })
        } else {
            None
        };
        let station = if stations.is_empty() {
            let Some(raw) = raw else {
                continue;
            };
            raw
        } else if let Some(s) = stations.iter().find(|s| {
            raw.as_ref()
                .map_or(s.station_id == id, |r| station_alias(s, r))
        }) {
            s.clone()
        } else {
            continue;
        };
        let mut values = BTreeMap::new();
        let mut offered = std::collections::BTreeSet::<String>::new();
        if format == "table" {
            let variable = get(row, &["variable"]);
            let name = match variable.as_str() {
                "temperature" | "temperature_2m" => "temperature_2m",
                "dewpoint" | "dewpoint_2m" => "dewpoint_2m",
                "u_wind" | "u_wind_10m" => "u_wind_10m",
                "v_wind" | "v_wind_10m" => "v_wind_10m",
                "pressure" | "station_pressure" => "pressure_surface",
                _ => continue,
            };
            offered.insert(name.into());
            if let Some(value) = num(&get(row, &["value"])) {
                values.insert(name.into(), value);
            }
        } else {
            for (column, awc_column, names) in [
                ("tmpf", "temp_c", &["temperature_2m"][..]),
                ("dwpf", "dewpoint_c", &["dewpoint_2m"][..]),
                (
                    "sknt",
                    "wind_speed_kt",
                    &["wind_speed_10m", "u_wind_10m", "v_wind_10m"][..],
                ),
                ("alti", "altim_in_hg", &["pressure_surface"][..]),
                ("mslp", "mslp", &["pressure_msl"][..]),
            ] {
                if columns.iter().any(|c| c == column || c == awc_column) {
                    offered.extend(names.iter().map(|s| s.to_string()));
                }
            }
            for (column, awc_column, name) in [
                ("tmpf", "temp_c", "temperature_2m"),
                ("dwpf", "dewpoint_c", "dewpoint_2m"),
            ] {
                if let Some(value) = num(&get(row, &[column, awc_column])) {
                    values.insert(
                        name.into(),
                        if format == "awc_csv" {
                            value + 273.15
                        } else {
                            (value - 32.0) * 5.0 / 9.0 + 273.15
                        },
                    );
                }
            }
            if let Some(speed) = num(&get(row, &["sknt", "wind_speed_kt"])) {
                let speed = speed * KNOT;
                values.insert("wind_speed_10m".into(), speed);
                if let Some(direction) = num(&get(row, &["drct", "wind_dir_degrees"])) {
                    let rad = direction.to_radians();
                    values.insert("u_wind_10m".into(), -speed * rad.sin());
                    values.insert("v_wind_10m".into(), -speed * rad.cos());
                }
            }
            if let Some(alti) = num(&get(row, &["alti", "altim_in_hg"])) {
                values.insert(
                    "pressure_surface".into(),
                    pressure(alti * 3386.389, station.elevation_m),
                );
            }
            if let Some(mslp) = num(&get(row, &["mslp"])) {
                values.insert("pressure_msl".into(), mslp * 100.0);
            }
        }
        screen(&mut values);
        let rejected = offered
            .into_iter()
            .filter(|name| !values.contains_key(name))
            .collect();
        let source = if format == "table" {
            let named = get(row, &["source"]);
            if named.is_empty() {
                "neutral-table".into()
            } else {
                named
            }
        } else if format == "awc_csv" {
            "awc-metar".into()
        } else {
            "iem-metar".into()
        };
        out.push(Report {
            station,
            t,
            values,
            source,
            rejected,
        });
    }
    Ok(out)
}
fn reports_awc(bytes: &[u8], frozen: &[Station]) -> Result<Vec<Report>> {
    let rows: Vec<Value> = serde_json::from_slice(bytes)?;
    let mut out = vec![];
    for r in rows {
        let (Some(id), Some(lat), Some(lon), Some(elevation_m)) = (
            r["icaoId"].as_str(),
            r["lat"].as_f64(),
            r["lon"].as_f64(),
            r["elev"].as_f64(),
        ) else {
            continue;
        };
        let Some(t) = r["obsTime"]
            .as_i64()
            .or_else(|| r["reportTime"].as_str().and_then(|s| time(s).ok()))
        else {
            continue;
        };
        let mut values = BTreeMap::new();
        for (column, name) in [("temp", "temperature_2m"), ("dewp", "dewpoint_2m")] {
            if let Some(v) = r[column].as_f64() {
                values.insert(name.into(), v + 273.15);
            }
        }
        if let Some(v) = r["wspd"].as_f64() {
            let speed = v * KNOT;
            values.insert("wind_speed_10m".into(), speed);
            if let Some(direction) = r["wdir"].as_f64() {
                values.insert("u_wind_10m".into(), -speed * direction.to_radians().sin());
                values.insert("v_wind_10m".into(), -speed * direction.to_radians().cos());
            }
        }
        if let Some(v) = r["altim"].as_f64() {
            values.insert("pressure_surface".into(), pressure(v * 100.0, elevation_m));
        }
        screen(&mut values);
        let rejected = [
            "temperature_2m",
            "dewpoint_2m",
            "wind_speed_10m",
            "u_wind_10m",
            "v_wind_10m",
            "pressure_surface",
        ]
        .into_iter()
        .filter(|name| !values.contains_key(*name))
        .map(String::from)
        .collect();
        let raw = Station {
            station_id: id.into(),
            lat,
            lon,
            elevation_m,
        };
        let station = if frozen.is_empty() {
            raw
        } else if let Some(station) = frozen.iter().find(|s| station_alias(s, &raw)) {
            station.clone()
        } else {
            continue;
        };
        out.push(Report {
            station,
            t,
            values,
            source: "awc-metar".into(),
            rejected,
        });
    }
    Ok(out)
}
pub fn station_alias(a: &Station, b: &Station) -> bool {
    let names = a.station_id == b.station_id
        || (a.station_id.len() == 3
            && b.station_id.len() == 4
            && b.station_id.strip_prefix('K') == Some(a.station_id.as_str()))
        || (b.station_id.len() == 3
            && a.station_id.len() == 4
            && a.station_id.strip_prefix('K') == Some(b.station_id.as_str()));
    if !names {
        return false;
    }
    let Ok(v) = obs_regrid::unit_vectors(&[a.lat, b.lat], &[a.lon, b.lon]) else {
        return false;
    };
    let d =
        ((v[0][0] - v[1][0]).powi(2) + (v[0][1] - v[1][1]).powi(2) + (v[0][2] - v[1][2]).powi(2))
            .sqrt();
    obs_regrid::arc_from_chord(d) <= 5000.
}
fn reports_madis(bytes: &[u8]) -> Result<(Vec<Report>, Value)> {
    let mut raw = bytes.to_vec();
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut v = vec![];
        flate2::read::GzDecoder::new(raw.as_slice()).read_to_end(&mut v)?;
        raw = v;
    }
    let file = netcrust::File::from_bytes(&raw)?;
    let lat = file.read_array_f64("latitude")?.into_values();
    let n = lat.len();
    let read = |name: &str| -> Result<Vec<f64>> {
        let v = file.read_array_f64(name)?.into_values();
        if v.len() != n {
            return Err(fail(format!("MADIS {name} record length mismatch")));
        }
        Ok(v)
    };
    let strings = |name: &str| -> Result<Vec<String>> {
        let v = file.read_strings(name)?;
        if v.len() == n {
            return Ok(v
                .into_iter()
                .map(|s| s.trim_matches('\0').trim().into())
                .collect());
        }
        if v.len() == 1 && v[0].len() == n {
            return Ok(v[0].chars().map(|c| c.to_string()).collect());
        }
        Err(fail(format!("MADIS {name} text record length mismatch")))
    };
    let lon = read("longitude")?;
    let elevation = read("elevation")?;
    let stamp = read("observationTime")?;
    let restrictions = read("restrictedDataLevel")?;
    let ids = strings("providerId")?;
    let providers = strings("dataProvider")?;
    let license: Value =
        serde_json::from_str(include_str!("../data/madis-public-providers-v1.json"))?;
    let licensed: std::collections::BTreeSet<_> = license["providers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    let restriction_fill = file
        .variable("restrictedDataLevel")
        .and_then(|v| v.attribute("_FillValue").and_then(|a| a.as_f64()));
    let mut columns = vec![];
    for (var, name, qc) in [
        ("temperature", "temperature_2m", "temperatureDD"),
        ("dewpoint", "dewpoint_2m", "dewpointDD"),
        ("stationPressure", "pressure_surface", "stationPressureDD"),
    ] {
        if file.variable(var).is_some() && file.variable(qc).is_some() {
            columns.push((name, read(var)?, strings(qc)?));
        }
    }
    // The archive's generic wind is at variable sensor height. Only its
    // explicit 10 m fields qualify for the 10 m forecast comparison.
    let wind = if file.variable("windSpeed10").is_some() {
        Some(read("windSpeed10")?)
    } else {
        None
    };
    let direction = if file.variable("windDir10").is_some() {
        Some(read("windDir10")?)
    } else {
        None
    };
    let wind_qc = if file.variable("windSpeedDD").is_some() {
        Some(strings("windSpeedDD")?)
    } else {
        None
    };
    let dir_qc = if file.variable("windDirDD").is_some() {
        Some(strings("windDirDD")?)
    } else {
        None
    };
    let sanitize = |s: &str| {
        s.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>()
    };
    let passed = |s: &str| matches!(s, "C" | "S" | "V" | "K" | "G");
    let mut out = vec![];
    let mut restricted = 0u64;
    let mut counts = BTreeMap::<String, u64>::new();
    let mut raw_counts = BTreeMap::<String, u64>::new();
    let mut public_records = 0;
    let mut licensed_missing = 0;
    for k in 0..n {
        *raw_counts.entry(providers[k].clone()).or_default() += 1;
        let missing = !restrictions[k].is_finite()
            || restriction_fill == Some(restrictions[k])
            || restrictions[k] == -9999.0;
        if restrictions[k] == 0.0 {
        } else if missing && licensed.contains(providers[k].as_str()) {
            licensed_missing += 1;
        } else {
            restricted += 1;
            continue;
        }
        if !lat[k].is_finite()
            || lat[k].abs() > 90.
            || !lon[k].is_finite()
            || lon[k].abs() > 360.
            || !elevation[k].is_finite()
            || elevation[k].abs() > 10000.
            || !stamp[k].is_finite()
            || stamp[k] <= 0.
            || stamp[k] > 4e9
        {
            continue;
        }
        public_records += 1;
        *counts.entry(providers[k].clone()).or_default() += 1;
        let station_id = format!("MADIS_{}_{}", sanitize(&providers[k]), sanitize(&ids[k]));
        let mut values = BTreeMap::new();
        for (name, values_in, qc) in &columns {
            if passed(&qc[k]) && values_in[k].abs() < 1e20 && values_in[k] != -9999. {
                values.insert((*name).into(), values_in[k]);
            }
        }
        if let (Some(w), Some(qc)) = (&wind, &wind_qc) {
            if passed(&qc[k]) && (0.0..=75.0).contains(&w[k]) {
                values.insert("wind_speed_10m".into(), w[k]);
                if let (Some(d), Some(qc)) = (&direction, &dir_qc) {
                    if passed(&qc[k]) && (0.0..=360.0).contains(&d[k]) {
                        values.insert("u_wind_10m".into(), -w[k] * d[k].to_radians().sin());
                        values.insert("v_wind_10m".into(), -w[k] * d[k].to_radians().cos());
                    }
                }
            }
        }
        screen(&mut values);
        let mut offered: std::collections::BTreeSet<String> = columns
            .iter()
            .map(|(name, _, _)| name.to_string())
            .collect();
        if wind.is_some() {
            offered.extend(
                ["wind_speed_10m", "u_wind_10m", "v_wind_10m"]
                    .into_iter()
                    .map(String::from),
            );
        }
        let rejected = offered
            .into_iter()
            .filter(|name| !values.contains_key(name))
            .collect();
        out.push(Report {
            station: Station {
                station_id,
                lat: lat[k],
                lon: lon[k],
                elevation_m: elevation[k],
            },
            t: stamp[k] as i64,
            values,
            source: "madis-mesonet".into(),
            rejected,
        });
    }
    Ok((
        out,
        json!({"madis_records":n,"restricted_or_unknown_records_excluded":restricted,"public_records":public_records,"public_records_by_provider":counts,"raw_records_by_provider":raw_counts,"licensed_missing_restriction_records":licensed_missing,"license_table":license,"qc_allowed":["C","S","V","K","G"],"restriction_level_required":"0 or missing for explicitly licensed category-3 provider","wind_height_required_m":10}),
    ))
}
fn mrms(request: &Value, bytes: &[u8], artifact: &mut Artifact) -> Result<()> {
    let mut raw = bytes.to_vec();
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut v = vec![];
        flate2::read::GzDecoder::new(raw.as_slice()).read_to_end(&mut v)?;
        raw = v;
    }
    let grib = grib_core::grib2::Grib2File::from_bytes(&raw)?;
    if grib.messages.len() != 1 {
        return Err(fail(
            "MRMS observation object must contain exactly one registered product",
        ));
    }
    let msg = &grib.messages[0];
    let requested = time(&artifact.valid_time)?;
    let actual = msg.reference_time.and_utc().timestamp();
    let tolerance = request["match_seconds"].as_i64().unwrap_or(120);
    if (actual - requested).abs() > tolerance {
        return Err(fail(format!(
            "MRMS nominal time {} outside requested match window",
            iso(actual)
        )));
    }
    let variable = get_str(request, "variable")?;
    if !["precipitation", "reflectivity"].contains(&variable) {
        return Err(fail("MRMS variable must be precipitation or reflectivity"));
    }
    let window = request["window_minutes"].as_u64().unwrap_or(0) as u32;
    // 180 minutes: MRMS publishes MultiSensor_QPE_03H_Pass2, whose encoded
    // duration is checked against the window just below like the others.
    if variable == "precipitation" && !matches!(window, 60 | 180 | 360) {
        return Err(fail(
            "MRMS QPE window must be the registered 60, 180 or 360 minutes",
        ));
    }
    if let Some(duration) = msg.product.statistical_time_range_seconds() {
        if duration != window as u64 * 60 {
            return Err(fail(
                "MRMS encoded accumulation duration differs from requested window",
            ));
        }
    }
    let spec: GridSpec = serde_json::from_value(request["grid"].clone())?;
    let values = grib_core::grib2::unpack_message_normalized(msg)?;
    let (lat, lon) = grib_core::grib2::grid_latlon(&msg.grid)?;
    // Crop before building the remap tree; full CONUS MRMS is not retained in
    // a worker artifact. The extra 2 cells ensure complete bilinear stencils.
    let nx = msg.grid.nx as usize;
    let ny = msg.grid.ny as usize;
    let minlat = spec
        .south
        .min(spec.south + (spec.ny - 1) as f64 * spec.dlat);
    let maxlat = spec
        .south
        .max(spec.south + (spec.ny - 1) as f64 * spec.dlat);
    let west = spec.west;
    let east = spec.west + (spec.nx - 1) as f64 * spec.dlon;
    let mut bounds = None::<(usize, usize, usize, usize)>;
    for k in 0..lat.len() {
        let l = (lon[k] + 180.).rem_euclid(360.) - 180.;
        if lat[k] >= minlat && lat[k] <= maxlat && l >= west.min(east) && l <= west.max(east) {
            let i = k % nx;
            let j = k / nx;
            bounds = Some(match bounds {
                None => (i, i, j, j),
                Some((a, b, c, d)) => (a.min(i), b.max(i), c.min(j), d.max(j)),
            });
        }
    }
    let (a, b, c, d) = bounds.ok_or_else(|| fail("verification region outside MRMS coverage"))?;
    let (a, b, c, d) = (
        a.saturating_sub(2),
        (b + 2).min(nx - 1),
        c.saturating_sub(2),
        (d + 2).min(ny - 1),
    );
    let mut clat = vec![];
    let mut clon = vec![];
    let mut vals = vec![];
    for j in c..=d {
        for i in a..=b {
            let k = j * nx + i;
            clat.push(lat[k]);
            clon.push(lon[k]);
            let v = values[k] as f64;
            vals.push(if variable == "reflectivity" {
                radar_value(v)
            } else if v.is_finite() && v >= 0.0 && v < 1e20 {
                v
            } else {
                f64::NAN
            });
        }
    }
    let locator = crate::support::Locator::new(b - a + 1, d - c + 1, clat, clon)?;
    if variable == "reflectivity" {
        for v in &mut vals {
            *v = 10f64.powf(*v / 10.);
        }
    }
    let out = (0..spec.ny)
        .flat_map(|j| (0..spec.nx).map(move |i| (j, i)))
        .map(|(j, i)| {
            locator
                .locate(
                    spec.south + j as f64 * spec.dlat,
                    spec.west + i as f64 * spec.dlon,
                )
                .and_then(|s| crate::support::sample(&vals, &s))
                .and_then(|v| {
                    if variable == "reflectivity" {
                        let v = 10. * v.log10();
                        v.is_finite().then_some(v)
                    } else {
                        Some(v)
                    }
                })
        })
        .collect();
    artifact.grids.push(Grid {
        spec,
        variable: variable.into(),
        window_minutes: window,
        values: out,
    });
    Ok(())
}
pub fn run(request: &Value) -> Result<Value> {
    let bytes = std::fs::read(get_str(request, "input")?)?;
    let frozen = load_stations(request)?;
    let grid: Option<GridSpec> = serde_json::from_value(request["grid"].clone())?;
    let format = get_str(request, "format")?;
    if format == "mrms" {
        let mut a = new_artifact(request, &bytes, frozen, &grid)?;
        mrms(request, &bytes, &mut a)?;
        return write_artifact(&a, get_str(request, "output")?);
    }
    let mut source_sha256 = digest(&bytes);
    let (reports, diagnostics) = match format {
        "iem_csv" | "awc_csv" | "table" => (reports_csv(&bytes, format, &frozen)?, json!({})),
        "awc_json" => (reports_awc(&bytes, &frozen)?, json!({})),
        "madis_netcdf" => reports_madis(&bytes)?,
        "dwd_synop_geojson" => reports_dwd(&bytes)?,
        "archive" => {
            if !bytes.starts_with(b"WFSBOBS1") {
                return Err(fail("normalized observation archive magic mismatch"));
            }
            let mut decoded = vec![];
            zstd::stream::read::Decoder::new(&bytes[8..])?
                .take(512 * 1024 * 1024 + 1)
                .read_to_end(&mut decoded)?;
            if decoded.len() > 512 * 1024 * 1024 {
                return Err(fail("observation archive exceeds bounded expansion"));
            }
            let archive: ObservationArchive = serde_json::from_slice(&decoded)?;
            if archive.schema != "woof.scoreboard.observation-archive.v1"
                || archive.method_id != METHOD
            {
                return Err(fail(
                    "normalized observation archive schema/method mismatch",
                ));
            }
            source_sha256 = archive.source_sha256;
            (archive.reports, archive.diagnostics)
        }
        _ => {
            return Err(fail(
                "observe format must be iem_csv, awc_csv, awc_json, table, madis_netcdf, dwd_synop_geojson, archive or mrms",
            ));
        }
    };
    let mut archive_receipt = None;
    if let Some(path) = request["archive_output"].as_str() {
        let archive = ObservationArchive {
            schema: "woof.scoreboard.observation-archive.v1".into(),
            method_id: METHOD.into(),
            source_sha256: source_sha256.clone(),
            diagnostics: diagnostics.clone(),
            reports: reports.clone(),
        };
        let mut encoded = b"WFSBOBS1".to_vec();
        encoded.extend(zstd::stream::encode_all(
            serde_json::to_vec(&archive)?.as_slice(),
            3,
        )?);
        archive_receipt = Some(
            json!({"archive_path":path,"archive_sha256":digest(&encoded),"archive_bytes":encoded.len(),"archive_reports":reports.len(),"archive_source_sha256":source_sha256}),
        );
        std::fs::write(path, encoded)?;
    }
    let mut stations: BTreeMap<String, Station> = frozen
        .iter()
        .map(|s| (s.station_id.clone(), s.clone()))
        .collect();
    if frozen.is_empty() {
        for r in &reports {
            stations
                .entry(r.station.station_id.clone())
                .or_insert_with(|| r.station.clone());
        }
    }
    let mut a = new_artifact(request, &bytes, stations.values().cloned().collect(), &grid)?;
    a.source_sha256 = source_sha256;
    a.diagnostics = diagnostics;
    a.diagnostics["normalized_reports"] = json!(reports.len());
    a.diagnostics["report_time_min"] = json!(reports.iter().map(|r| r.t).min().map(iso));
    a.diagnostics["report_time_max"] = json!(reports.iter().map(|r| r.t).max().map(iso));
    let mut report_times = BTreeMap::<String, std::collections::BTreeSet<i64>>::new();
    for r in &reports {
        report_times
            .entry(r.station.station_id.clone())
            .or_default()
            .insert(r.t);
    }
    let mut fast_stations = 0u64;
    let mut min_gap = None::<i64>;
    for times in report_times.values() {
        let times: Vec<_> = times.iter().copied().collect();
        let mut fast = false;
        for pair in times.windows(2) {
            let gap = pair[1] - pair[0];
            if gap > 0 {
                min_gap = Some(min_gap.map_or(gap, |b| b.min(gap)));
                fast |= gap <= 300;
            }
        }
        fast_stations += u64::from(fast);
    }
    a.diagnostics["stations_with_reports_at_most_300s_apart"] = json!(fast_stations);
    a.diagnostics["minimum_positive_report_gap_seconds"] = json!(min_gap);
    let target = time(&a.valid_time)?;
    let tolerance = request["match_seconds"].as_i64().unwrap_or(600);
    let accepted: std::collections::BTreeSet<_> = reports
        .iter()
        .flat_map(|r| {
            r.values.keys().map(move |name| {
                (
                    r.station.station_id.clone(),
                    name.clone(),
                    r.t,
                    r.source.clone(),
                )
            })
        })
        .collect();
    let mut rejected = BTreeMap::new();
    let mut best: BTreeMap<(String, String), (i64, i64, f64, String)> = BTreeMap::new();
    for mut r in reports {
        if !stations.contains_key(&r.station.station_id) {
            if let Some(canonical) = stations.values().find(|s| station_alias(s, &r.station)) {
                r.station = canonical.clone();
            } else {
                continue;
            }
        }
        let distance = (r.t - target).abs();
        if distance > tolerance {
            continue;
        }
        for name in &r.rejected {
            let key = (
                r.station.station_id.clone(),
                name.clone(),
                r.t,
                r.source.clone(),
            );
            if !accepted.contains(&key) {
                rejected.insert(key, "missing_or_quality_rejected");
            }
        }
        for (name, value) in r.values {
            let key = (r.station.station_id.clone(), name);
            let candidate = (distance, r.t, value, r.source.clone());
            if best.get(&key).is_none_or(|v| {
                (distance, r.t, value.to_bits(), &r.source) < (v.0, v.1, v.2.to_bits(), &v.3)
            }) {
                best.insert(key, candidate);
            }
        }
    }
    for ((station_id, variable, t, source_id), reason) in rejected {
        a.tombstones.push(ObservationRetraction {
            station_id: station_id.clone(),
            variable: variable.clone(),
            observation_time: iso(t),
            source_id: source_id.clone(),
            reason: reason.into(),
        });
        if variable == "temperature_2m" {
            a.tombstones.push(ObservationRetraction {
                station_id,
                variable: "temperature_2m_raw".into(),
                observation_time: iso(t),
                source_id,
                reason: reason.into(),
            });
        }
    }
    for station in &a.stations {
        let mut values = BTreeMap::new();
        let mut observed = None;
        let mut observation_times = BTreeMap::new();
        let mut observation_sources = BTreeMap::new();
        for ((_, name), (_, t, value, source)) in best
            .range((station.station_id.clone(), String::new())..)
            .take_while(|((id, _), _)| id == &station.station_id)
        {
            values.insert(name.clone(), *value);
            observation_times.insert(name.clone(), iso(*t));
            observation_sources.insert(name.clone(), source.clone());
            observed = Some(*t);
        }
        if let Some(t) = values.get("temperature_2m").copied() {
            values.insert("temperature_2m_raw".into(), t);
            if let Some(t) = observation_times.get("temperature_2m").cloned() {
                observation_times.insert("temperature_2m_raw".into(), t);
            }
            if let Some(source) = observation_sources.get("temperature_2m").cloned() {
                observation_sources.insert("temperature_2m_raw".into(), source);
            }
        }
        if !values.contains_key("wind_speed_10m") {
            if let (Some(u), Some(v)) = (values.get("u_wind_10m"), values.get("v_wind_10m")) {
                if observation_times.get("u_wind_10m") == observation_times.get("v_wind_10m") {
                    values.insert("wind_speed_10m".into(), u.hypot(*v));
                    if let Some(t) = observation_times.get("u_wind_10m").cloned() {
                        observation_times.insert("wind_speed_10m".into(), t);
                    }
                    if let Some(source) = observation_sources.get("u_wind_10m").cloned() {
                        observation_sources.insert("wind_speed_10m".into(), source);
                    }
                }
            }
        }
        screen(&mut values);
        if values.is_empty() {
            a.drops.push(
                json!({"station_id":station.station_id,"reason":"no_quality_report_in_window"}),
            );
        } else {
            a.points.push(Point {
                station_id: station.station_id.clone(),
                valid_time: a.valid_time.clone(),
                observation_time: observed.map(iso),
                observation_times,
                observation_sources,
                values,
                terrain_m: None,
            });
        }
    }
    let mut sample_counts = BTreeMap::<String, u64>::new();
    for point in &a.points {
        for name in point.values.keys() {
            *sample_counts.entry(name.clone()).or_default() += 1;
        }
    }
    a.diagnostics["selected_samples_by_variable"] = json!(sample_counts);
    let mut receipt = write_artifact(&a, get_str(request, "output")?)?;
    if let Some(archive) = archive_receipt {
        for (key, value) in archive.as_object().unwrap() {
            receipt[key] = value.clone();
        }
    }
    Ok(receipt)
}
pub fn merge(request: &Value) -> Result<Value> {
    let inputs = request["inputs"]
        .as_array()
        .ok_or_else(|| fail("merge inputs array required"))?;
    let mut artifacts = vec![];
    let mut source_bytes = vec![];
    for input in inputs {
        let path = input
            .as_str()
            .ok_or_else(|| fail("merge input must be path"))?;
        let bytes = std::fs::read(path)?;
        source_bytes.extend_from_slice(digest(&bytes).as_bytes());
        artifacts.push(read_artifact(path)?);
    }
    if artifacts.is_empty() {
        return Err(fail("merge requires at least one artifact"));
    }
    let stations = load_stations(request)?;
    let grid: Option<GridSpec> = serde_json::from_value(request["grid"].clone())?;
    let mut out = new_artifact(request, &source_bytes, stations, &grid)?;
    let target = time(&out.valid_time)?;
    let tolerance = request["match_seconds"].as_i64().unwrap_or(600);
    let inventory: std::collections::BTreeSet<_> =
        out.stations.iter().map(|s| s.station_id.clone()).collect();
    // Resolve corrections at the source/report/variable identity first. A
    // wholly absent report creates no event and cannot withdraw another feed.
    let mut events = BTreeMap::<(String, String, i64, String), (usize, Option<f64>)>::new();
    for (priority, artifact) in artifacts.iter().enumerate() {
        if artifact.method_id != METHOD {
            return Err(fail("merge method mismatch"));
        }
        for point in &artifact.points {
            if !inventory.contains(&point.station_id) {
                continue;
            }
            for (name, value) in &point.values {
                let observed = point
                    .observation_times
                    .get(name)
                    .or(point.observation_time.as_ref())
                    .unwrap_or(&point.valid_time);
                let t = time(observed)?;
                let delta = (t - target).abs();
                if delta > tolerance {
                    continue;
                }
                let source = point
                    .observation_sources
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| "legacy-observation".into());
                let key = (point.station_id.clone(), name.clone(), t, source);
                if value.is_finite() && events.get(&key).is_none_or(|b| priority < b.0) {
                    events.insert(key, (priority, Some(*value)));
                }
            }
        }
        for tombstone in &artifact.tombstones {
            let t = time(&tombstone.observation_time)?;
            if !inventory.contains(&tombstone.station_id) || (t - target).abs() > tolerance {
                continue;
            }
            let key = (
                tombstone.station_id.clone(),
                tombstone.variable.clone(),
                t,
                tombstone.source_id.clone(),
            );
            if events.get(&key).is_none_or(|b| priority < b.0) {
                events.insert(key, (priority, None));
            }
        }
        for g in &artifact.grids {
            if artifact.valid_time == out.valid_time && Some(&g.spec) == grid.as_ref() {
                if !out
                    .grids
                    .iter()
                    .any(|v| v.variable == g.variable && v.window_minutes == g.window_minutes)
                {
                    out.grids.push(g.clone());
                }
            }
        }
    }
    let mut selected = BTreeMap::<(String, String), (i64, i64, usize, f64, String)>::new();
    for ((id, name, t, source), (priority, value)) in events {
        if let Some(value) = value {
            let delta = (t - target).abs();
            let key = (id, name);
            if selected
                .get(&key)
                .is_none_or(|b| (delta, t, priority, &source) < (b.0, b.1, b.2, &b.4))
            {
                selected.insert(key, (delta, t, priority, value, source));
            }
        } else {
            out.tombstones.push(ObservationRetraction {
                station_id: id,
                variable: name,
                observation_time: iso(t),
                source_id: source,
                reason: "missing_or_quality_rejected".into(),
            });
        }
    }
    for station in &out.stations {
        let mut values = BTreeMap::new();
        let mut times = BTreeMap::new();
        let mut sources = BTreeMap::new();
        for ((_, name), (_, t, _, value, source)) in selected
            .range((station.station_id.clone(), String::new())..)
            .take_while(|((id, _), _)| id == &station.station_id)
        {
            values.insert(name.clone(), *value);
            times.insert(name.clone(), iso(*t));
            sources.insert(name.clone(), source.clone());
        }
        if !values.is_empty() {
            out.points.push(Point {
                station_id: station.station_id.clone(),
                valid_time: out.valid_time.clone(),
                observation_time: times.values().next().cloned(),
                observation_times: times,
                observation_sources: sources,
                values,
                terrain_m: None,
            });
        }
    }
    write_artifact(&out, get_str(request, "output")?)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn escaped_csv_is_real_csv() {
        let a = csv("station,name,valid\nA,\"a,b\"\"c\",2026-01-01 00:00\n").unwrap();
        assert_eq!(a[1][1], "a,b\"c");
    }
    #[test]
    fn screen_supersaturated_dewpoint() {
        let mut v = BTreeMap::from([
            ("temperature_2m".into(), 280.),
            ("dewpoint_2m".into(), 282.),
        ]);
        screen(&mut v);
        assert!(!v.contains_key("dewpoint_2m"));
    }
    #[test]
    fn mrms_no_echo_is_valid() {
        assert!(radar_value(-999.).is_nan());
        assert_eq!(radar_value(-99.), -35.);
        assert_eq!(radar_value(-31.5), -31.5);
        assert_eq!(radar_value(45.), 45.);
    }
    #[test]
    fn dwd_numeric_descriptor_order_binds_sensor_height_and_missing_fields() {
        let mut properties = json!({
            "data_1":{"shortname":"wigosLocalIdentifierCharacter","value":"01234 "},
            "data_2":{"shortname":"year","value":2026},"data_3":{"shortname":"month","value":10},
            "data_4":{"shortname":"day","value":2},"data_5":{"shortname":"hour","value":8},"data_6":{"shortname":"minute","value":0},
            "data_9":{"shortname":"heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform","value":2.0,"unit":"m"},
            "data_10":{"shortname":"airTemperature","value":290.0,"unit":"K"},
            "data_11":{"shortname":"dewpointTemperature","value":null,"unit":"K"},
            "data_12":{"shortname":"heightOfSensorAboveLocalGroundOrDeckOfMarinePlatform","value":10.0,"unit":"m"},
            "data_13":{"shortname":"windDirection","value":90.0,"unit":"deg"},"data_14":{"shortname":"windSpeed","value":5.0,"unit":"m/s"}
        });
        let encode = |props: Value| {
            serde_json::to_vec(&json!({"type":"FeatureCollection","features":[{"geometry":{"coordinates":[10.0,50.0,100.0]},"properties":props}]})).unwrap()
        };
        let (reports, _) = reports_dwd(&encode(properties.clone())).unwrap();
        assert_eq!(reports[0].station.station_id, "DWD_SYNOP_01234");
        assert_eq!(reports[0].values["temperature_2m"], 290.0);
        assert!((reports[0].values["u_wind_10m"] + 5.0).abs() < 1e-12);
        assert!(reports[0].rejected.contains("dewpoint_2m"));
        properties["data_9"]["value"] = json!(1.0);
        properties["data_12"]["value"] = json!(20.0);
        let (reports, _) = reports_dwd(&encode(properties)).unwrap();
        assert!(reports[0].values.is_empty());
        assert!(reports[0].rejected.contains("temperature_2m"));
        assert!(reports[0].rejected.contains("wind_speed_10m"));
    }
}
