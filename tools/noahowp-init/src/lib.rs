// SPDX-License-Identifier: Apache-2.0
// Initialization follows NOAA-OWP/noah-owp-modular 0242a96c; see port NOTICE.
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::{c_char, CStr},
    fs,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, String>;
type Values = BTreeMap<String, Vec<String>>;
type Groups = BTreeMap<String, Values>;
type State = BTreeMap<String, Vec<f64>>;
mod snow;
fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn read(p: &Path) -> Result<Vec<u8>> {
    fs::read(p).map_err(|e| format!("{}: {e}", p.display()))
}
fn string(v: &Value, k: &str) -> Result<String> {
    v[k].as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("missing string {k}"))
}
fn tokens(text: &str) -> Result<Vec<String>> {
    let mut out = vec![];
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        if c.is_whitespace() || c == ',' {
            continue;
        }
        if c == '!' {
            while let Some(c) = it.next() {
                if c == '\n' {
                    break;
                }
            }
            continue;
        }
        if c == '\'' || c == '"' {
            let mut s = String::new();
            let mut closed = false;
            while let Some(d) = it.next() {
                if d == c {
                    if it.peek() == Some(&c) {
                        it.next();
                        s.push(c)
                    } else {
                        closed = true;
                        break;
                    }
                } else {
                    s.push(d)
                }
            }
            if !closed {
                return Err("unterminated string".into());
            }
            out.push(format!("\"{s}\""));
            continue;
        }
        if "&/=()*".contains(c) {
            out.push(c.to_string());
            continue;
        }
        let mut s = c.to_string();
        while let Some(&d) = it.peek() {
            if d.is_whitespace() || ",!&/=()*\"'".contains(d) {
                break;
            }
            s.push(d);
            it.next();
        }
        out.push(s);
    }
    Ok(out)
}
fn unquote(s: &str) -> String {
    s.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(s)
        .to_owned()
}
fn namelist(text: &str) -> Result<Groups> {
    let t = tokens(text)?;
    let mut groups = Groups::new();
    let mut i = 0;
    while i < t.len() {
        if t[i] != "&" {
            i += 1;
            continue;
        }
        i += 1;
        if i == t.len() {
            return Err("missing group".into());
        }
        let group = t[i].to_lowercase();
        i += 1;
        let mut fields = Values::new();
        while i < t.len() && t[i] != "/" {
            let key = t[i].to_lowercase();
            i += 1;
            if t.get(i).map(String::as_str) != Some("=") {
                return Err(format!(
                    "expected '=' after {group}.{key}; indexed assignments unsupported"
                ));
            }
            i += 1;
            let mut v = vec![];
            while i < t.len() && t[i] != "/" && t.get(i + 1).map(String::as_str) != Some("=") {
                if t.get(i + 1).map(String::as_str) == Some("*") {
                    let n = t[i].parse::<usize>().map_err(|_| "invalid repeat count")?;
                    if n > 100000 || i + 2 >= t.len() {
                        return Err("invalid repeat value".into());
                    }
                    v.extend(std::iter::repeat_n(unquote(&t[i + 2]), n));
                    i += 3
                } else {
                    v.push(unquote(&t[i]));
                    i += 1
                }
            }
            if v.is_empty() || fields.insert(key.clone(), v).is_some() {
                return Err(format!("empty or duplicate {group}.{key}"));
            }
        }
        if i == t.len() {
            return Err(format!("unterminated group {group}"));
        }
        i += 1;
        if groups.insert(group.clone(), fields).is_some() {
            return Err(format!("duplicate group {group}"));
        }
    }
    Ok(groups)
}
fn atom(s: &str) -> Result<f64> {
    match s.trim() {
        ".true." => Ok(1.),
        ".false." => Ok(0.),
        "huge(1)" => Ok(i32::MAX as f64),
        "huge(1.0)" => Ok(f32::MAX as f64),
        "huge(1.d0)" => Ok(f64::MAX),
        s => s
            .replace(['D', 'd'], "e")
            .parse::<f32>()
            .map(|x| x as f64)
            .map_err(|_| format!("invalid number {s}")),
    }
}
fn get<'a>(v: &'a Values, k: &str) -> Result<&'a [String]> {
    v.get(k)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing namelist/table field {k}"))
}
fn number(v: &Values, k: &str) -> Result<f32> {
    let a = get(v, k)?;
    if a.len() != 1 {
        return Err(format!("{k} must be scalar"));
    }
    Ok(atom(&a[0])? as f32)
}
fn index(v: &Values, k: &str) -> Result<usize> {
    let x = number(v, k)?;
    if x < 1. || x.fract() != 0. || x > 100000. {
        return Err(format!("invalid one-based index {k}"));
    }
    Ok(x as usize - 1)
}
fn vals(v: &Values, k: &str) -> Result<Vec<f64>> {
    get(v, k)?.iter().map(|s| atom(s)).collect()
}
fn set(s: &mut State, k: &str, v: Vec<f64>) -> Result<()> {
    let a = s
        .get_mut(k)
        .ok_or_else(|| format!("initializer field absent from schema {k}"))?;
    if v.len() == 1 {
        a.fill(v[0])
    } else if v.len() == a.len() {
        *a = v
    } else {
        return Err(format!("wrong length {k}: {} != {}", v.len(), a.len()));
    }
    Ok(())
}
fn put(s: &mut State, k: &str, v: f32) -> Result<()> {
    set(s, k, vec![v as f64])
}
fn take(s: &State, k: &str) -> f32 {
    s[k][0] as f32
}

/// Optional prepared soil profile, in K, on the existing four soil layers.
/// Snow layers (-2..0) and surface/canopy temperatures are separate state.
fn soil_temperature(state: &mut State, values: &Values) -> Result<()> {
    if values.contains_key("soil_temperature_k") {
        let profile = vals(values, "soil_temperature_k")?;
        if profile.len() != 4 || profile.iter().any(|x| !x.is_finite() || *x <= 0.) {
            return Err(
                "soil_temperature_k requires exactly 4 finite positive Kelvin values".into(),
            );
        }
        let stc = state.get_mut("energy.stc").ok_or("missing energy.stc")?;
        if stc.len() != 7 {
            return Err("energy.stc requires 3 snow + 4 soil layers".into());
        }
        stc[3..].copy_from_slice(&profile);
    }
    Ok(())
}
fn date(s: &str) -> Result<([i32; 5], f64)> {
    if s.len() != 12 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err("date must be YYYYMMDDhhmm".into());
    }
    let y = s[..4].parse::<i32>().unwrap();
    let m = s[4..6].parse::<i32>().unwrap();
    let d = s[6..8].parse::<i32>().unwrap();
    let h = s[8..10].parse::<i32>().unwrap();
    let min = s[10..12].parse::<i32>().unwrap();
    let leap = |y: i32| y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let lengths = [
        31,
        if leap(y) { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=12).contains(&m) || d < 1 || d > lengths[(m - 1) as usize] || h > 23 || min > 59 {
        return Err(format!("invalid date {s}"));
    }
    let mut days = 0i64;
    if y >= 1970 {
        for yy in 1970..y {
            days += if leap(yy) { 366 } else { 365 }
        }
    } else {
        for yy in y..1970 {
            days -= if leap(yy) { 366 } else { 365 }
        }
    }
    days += lengths[..(m - 1) as usize]
        .iter()
        .map(|&n| n as i64)
        .sum::<i64>()
        + d as i64
        - 1;
    Ok((
        [y, m, d, h, min],
        (days * 86400 + h as i64 * 3600 + min as i64 * 60) as f64,
    ))
}
fn soil_table(text: &str, class: &str) -> Result<Values> {
    let lines: Vec<_> = text.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.trim().trim_matches(['\'', '"']) == class)
        .ok_or_else(|| format!("soil class {class} absent"))?;
    let n = tokens(lines.get(start + 1).ok_or("soil table truncated")?)?[0]
        .parse::<usize>()
        .map_err(|_| "invalid soil category count")?;
    let keys = [
        "bexp", "smcdry", "f1", "smcmax", "smcref", "psisat", "dksat", "dwsat", "smcwlt", "quartz",
        "bvic", "axaj", "bxaj", "xxaj", "bdvic", "bbvic", "gdvic",
    ];
    let mut out = Values::new();
    for k in keys {
        out.insert(k.into(), vec![]);
    }
    for i in 0..n {
        let t = tokens(lines.get(start + 2 + i).ok_or("soil table truncated")?)?;
        if t.len() < 18 || t[0].parse::<usize>().ok() != Some(i + 1) {
            return Err("invalid soil row".into());
        }
        for (j, k) in keys.iter().enumerate() {
            out.get_mut(*k).unwrap().push(t[j + 1].clone());
        }
    }
    Ok(out)
}
fn general_table(text: &str) -> Result<Values> {
    let lines: Vec<_> = text.lines().map(str::trim).collect();
    let mut out = Values::new();
    for (i, l) in lines.iter().enumerate() {
        if let Some(k) = l.strip_suffix("_DATA") {
            let mut v = vec![];
            let mut j = i + 1;
            while j < lines.len() && !lines[j].ends_with("_DATA") {
                v.extend(tokens(lines[j])?);
                j += 1
            }
            if k == "SLOPE" {
                if v.len() < 2 {
                    return Err("missing slope table".into());
                }
                v.remove(0);
            }
            out.insert(k.to_lowercase(), v);
        }
    }
    Ok(out)
}
fn path(base: &Path, s: &str) -> PathBuf {
    let p = Path::new(s);
    if p.is_absolute() {
        p.to_owned()
    } else {
        base.join(p)
    }
}
fn init(
    config: &Value,
    base: &Path,
    schema: &Value,
    rules: &Value,
    default_parameters: Option<&Path>,
) -> Result<(State, Value)> {
    let config_path = path(base, &string(config, "namelist")?);
    let bytes = read(&config_path)?;
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let groups = namelist(text)?;
    let mut n = Values::new();
    for (_, fields) in groups {
        for (k, v) in fields {
            if n.insert(k.clone(), v).is_some() {
                return Err(format!("duplicate namelist field {k}"));
            }
        }
    }
    if let Some(overrides) = config["overrides"].as_object() {
        for (k, v) in overrides {
            let vs = if let Some(a) = v.as_array() {
                a.clone()
            } else {
                vec![v.clone()]
            };
            n.insert(
                k.to_lowercase(),
                vs.iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| v.to_string())
                    })
                    .collect(),
            );
        }
    }
    if number(&n, "nsoil")? != 4. || number(&n, "nsnow")? != 3. {
        return Err("resident Noah-OWP requires 4 soil / 3 snow layers".into());
    }
    let mut state = State::new();
    for f in schema["fields"].as_array().ok_or("bad schema")? {
        state.insert(
            string(f, "name")?,
            vec![0.; f["count"].as_u64().ok_or("bad count")? as usize],
        );
    }
    for (k, v) in rules["defaults"].as_object().unwrap() {
        if state.contains_key(k) {
            set(&mut state, k, vec![atom(v.as_str().unwrap())?])?
        }
    }
    for (k, v) in rules["transfer"].as_object().unwrap() {
        if state.contains_key(k) && v != "zsoil" {
            set(&mut state, k, vals(&n, v.as_str().unwrap())?)?
        }
    }
    let dir = if let Some(s) = config["parameter_dir"].as_str() {
        path(base, s)
    } else if let Some(dir) = default_parameters {
        dir.to_owned()
    } else {
        path(config_path.parent().unwrap(), &get(&n, "parameter_dir")?[0])
    };
    let mut evidence = vec![json!({"path":config_path,"sha256":sha(&bytes)})];
    let mut texts = vec![];
    for k in ["soil_table", "general_table", "noahowp_table"] {
        let p = dir.join(&get(&n, k)?[0]);
        let b = read(&p)?;
        evidence.push(json!({"path":p,"sha256":sha(&b)}));
        texts.push(String::from_utf8(b).map_err(|e| e.to_string())?);
    }
    let soil = soil_table(&texts[0], &get(&n, "soil_class_name")?[0])?;
    let general = general_table(&texts[1])?;
    let mp = namelist(&texts[2])?;
    let prefix = match get(&n, "veg_class_name")?[0].as_str() {
        "MODIFIED_IGBP_MODIS_NOAH" => "modis",
        "USGS" => "usgs",
        _ => return Err("unknown vegetation class".into()),
    };
    let veg = mp
        .get(&format!("{prefix}_veg_parameters"))
        .ok_or("missing vegetation parameters")?;
    let categories = mp
        .get(&format!("{prefix}_veg_categories"))
        .ok_or("missing vegetation categories")?;
    let rad = mp
        .get("rad_parameters")
        .ok_or("missing radiation parameters")?;
    let global = mp
        .get("global_parameters")
        .ok_or("missing global parameters")?;
    let months = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    for (k, entries) in rules["parameter_tables"].as_object().unwrap() {
        let mut result = vec![];
        for entry in entries.as_array().unwrap() {
            let t = entry["table"].as_str().unwrap();
            let ix = entry["index"].as_str();
            let selected = if t == "laim" || t == "saim" {
                let p = if t == "laim" { "lai" } else { "sai" };
                let j = index(&n, "vegtyp")?;
                months
                    .iter()
                    .map(|m| {
                        let v = get(veg, &format!("{p}_{m}"))?;
                        atom(v.get(j).ok_or("vegetation index outside monthly table")?)
                    })
                    .collect::<Result<Vec<_>>>()?
            } else if let Some(ix) = ix {
                let parts: Vec<_> = ix.split(',').map(str::trim).collect();
                let mut table = t.to_owned();
                let (values, j) = if parts[0] == "namelist%vegtyp" {
                    if parts.len() == 2 {
                        table = format!("{t}_{}", if parts[1] == "1" { "vis" } else { "nir" });
                    }
                    (veg, index(&n, "vegtyp")?)
                } else if parts[0] == "namelist%isltyp" {
                    (&soil, index(&n, "isltyp")?)
                } else if parts[0] == "namelist%soilcolor" {
                    table = format!("{t}_{}", if parts[1] == "1" { "vis" } else { "nir" });
                    (rad, index(&n, "soilcolor")?)
                } else if parts[0] == "1" {
                    (&general, 0)
                } else {
                    return Err(format!("unknown table selection {ix}"));
                };
                vec![atom(
                    get(values, &table)?
                        .get(j)
                        .ok_or_else(|| format!("index outside {table}"))?,
                )?]
            } else {
                let value = general
                    .get(t)
                    .or_else(|| global.get(t))
                    .or_else(|| rad.get(t))
                    .or_else(|| categories.get(t))
                    .or_else(|| veg.get(t));
                if let Some(v) = value {
                    v.iter().map(|s| atom(s)).collect::<Result<Vec<_>>>()?
                } else if t.starts_with("lcz_") {
                    vec![-99999.]
                } else {
                    return Err(format!("missing parameter table {t}"));
                }
            };
            result.extend(selected);
        }
        set(&mut state, k, result)?;
    }
    for (k, v) in rules["constants"].as_object().unwrap() {
        set(&mut state, k, vec![atom(v.as_str().unwrap())?])?
    }
    let kdt = take(&state, "parameters.refkdt") * take(&state, "parameters.dksat")
        / take(&state, "parameters.refdk");
    put(&mut state, "parameters.kdt", kdt)?;
    let frzx = 0.15f32
        * (take(&state, "parameters.smcmax") / take(&state, "parameters.smcref"))
        * (0.412f32 / 0.468f32);
    put(&mut state, "parameters.frzx", frzx)?;
    let threshold = match number(&n, "precip_phase_option")? as i32 {
        2 => 273.16f32 + 2.2f32,
        5 | 6 => 273.16f32 + number(&n, "rain_snow_thresh")?,
        _ => 273.16f32,
    };
    put(&mut state, "parameters.rain_snow_thresh", threshold)?;
    put(&mut state, "parameters.zwt_init", number(&n, "zwt")?)?;
    let dz = vals(&n, "dzsnso")?;
    if dz.len() != 7 {
        return Err("dzsnso requires 3 snow + 4 soil values".into());
    }
    let mut z = vec![];
    let mut sum = 0f32;
    for d in &dz[3..] {
        if *d <= 0. {
            return Err("soil thickness must be positive".into());
        }
        sum += *d as f32;
        z.push((-sum) as f64)
    }
    set(&mut state, "domain.zsoil", z.clone())?;
    let mut zsnso = vec![0.; 3];
    zsnso.extend(z);
    set(&mut state, "domain.zsnso", zsnso)?;
    let liquid = vals(&n, "sh2o")?;
    let ice = vals(&n, "sice")?;
    if liquid.len() != 4 || ice.len() != 4 {
        return Err("soil moisture requires 4 layers".into());
    }
    let total: Vec<_> = liquid
        .iter()
        .zip(ice)
        .map(|(&a, b)| ((a as f32) + (b as f32)) as f64)
        .collect();
    set(&mut state, "water.smc", total.clone())?;
    set(&mut state, "water.smc_init", total)?;
    // Apply after native cold-start constants; otherwise STC=298 overwrites it.
    soil_temperature(&mut state, &n)?;
    let snow_recipe = snow::initialize(&mut state, &n)?;
    let (parts, start) = date(&get(&n, "startdate")?[0])?;
    let (_, end) = date(&get(&n, "enddate")?[0])?;
    let dt = number(&n, "dt")?;
    if !dt.is_finite()
        || dt <= 0.
        || end < start
        || (((end - start) / dt as f64).round() * dt as f64 - (end - start)).abs() > 1e-5
    {
        return Err("positive dt must divide simulation duration".into());
    }
    for (k, v) in [
        "start_year",
        "start_month",
        "start_day",
        "start_hour",
        "start_minute",
    ]
    .iter()
    .zip(parts)
    {
        set(&mut state, &format!("domain.{k}"), vec![v as f64])?
    }
    set(&mut state, "domain.start_datetime", vec![start])?;
    set(&mut state, "domain.end_datetime", vec![end])?;
    set(
        &mut state,
        "domain.ntime",
        vec![((end - start) / dt as f64) as i32 as f64 + 1.],
    )?;
    if !(1. ..=4.).contains(&take(&state, "parameters.nroot")) {
        return Err("NROOT outside resident 1..4 allocation".into());
    }
    Ok((
        state,
        json!({"id":string(config,"id")?,"inputs":evidence,"start_datetime":start,"end_datetime":end,"dt_seconds":dt,"overrides":config["overrides"],"snow_initialization":snow_recipe}),
    ))
}

#[cfg(test)]
mod soil_temperature_tests {
    use super::*;

    fn state() -> State {
        State::from([
            ("energy.stc".into(), vec![298.; 7]),
            ("energy.tg".into(), vec![298.]),
        ])
    }

    #[test]
    fn absent_profile_preserves_native_initialization() {
        let mut s = state();
        let before = s.clone();
        soil_temperature(&mut s, &Values::new()).unwrap();
        assert_eq!(s, before);
    }

    #[test]
    fn profile_targets_only_soil_levels_and_rounds_like_native_real() {
        let mut s = state();
        let v = Values::from([(
            "soil_temperature_k".into(),
            ["260.123456789", "270", "280", "290"]
                .map(str::to_owned)
                .to_vec(),
        )]);
        soil_temperature(&mut s, &v).unwrap();
        assert_eq!(
            s["energy.stc"],
            vec![298., 298., 298., 260.123456789f32 as f64, 270., 280., 290.]
        );
        assert_eq!(s["energy.tg"], vec![298.]);
    }

    #[test]
    fn invalid_profile_refuses_before_mutating_state() {
        for data in [
            vec!["270"],
            vec!["270"; 3],
            vec!["270"; 5],
            vec!["270", "NaN", "270", "270"],
            vec!["270", "inf", "270", "270"],
            vec!["270", "0", "270", "270"],
            vec!["270", "-1", "270", "270"],
            vec!["270", "1e99", "270", "270"],
        ] {
            let mut s = state();
            let before = s.clone();
            let v = Values::from([(
                "soil_temperature_k".into(),
                data.into_iter().map(str::to_owned).collect(),
            )]);
            assert!(soil_temperature(&mut s, &v).is_err());
            assert_eq!(s, before);
        }
    }
}
pub fn prepare(request: &Path, destination: &Path) -> Result<Value> {
    prepare_with_parameters(request, destination, None)
}

pub fn prepare_with_parameters(
    request: &Path,
    destination: &Path,
    default_parameters: Option<&Path>,
) -> Result<Value> {
    let before = std::time::Instant::now();
    let bytes = read(request)?;
    let spec: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let configs = spec["catchments"]
        .as_array()
        .ok_or("request requires catchments[]")?;
    if configs.is_empty() {
        return Err("empty catchment roster".into());
    }
    let base = request.parent().unwrap_or(Path::new("."));
    let schema: Value =
        serde_json::from_str(include_str!("../../../gpuwm/hydro/noahowp/schema.json")).unwrap();
    let rules: Value = serde_json::from_str(include_str!("rules.json")).unwrap();
    let columns = configs
        .par_iter()
        .map(|c| init(c, base, &schema, &rules, default_parameters))
        .collect::<Result<Vec<_>>>()?;
    let mut ids = std::collections::BTreeSet::new();
    for (_, m) in &columns {
        if !ids.insert(m["id"].to_string()) {
            return Err("duplicate catchment ID".into());
        }
    }
    if destination.exists() {
        return Err(format!(
            "destination already exists: {}",
            destination.display()
        ));
    }
    fs::create_dir_all(destination).map_err(|e| e.to_string())?;
    let n = columns.len();
    let mut files = serde_json::Map::new();
    for (dtype, width) in [("f32", 4), ("i32", 4), ("f64", 8)] {
        let size = schema["fields"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["dtype"] == dtype)
            .map(|f| f["offset"].as_u64().unwrap() + f["count"].as_u64().unwrap())
            .max()
            .unwrap() as usize;
        let mut payload = vec![0u8; size * n * width];
        for f in schema["fields"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["dtype"] == dtype)
        {
            let name = f["name"].as_str().unwrap();
            let offset = f["offset"].as_u64().unwrap() as usize;
            for (c, (s, _)) in columns.iter().enumerate() {
                for (j, &v) in s[name].iter().enumerate() {
                    let p = ((offset + j) * n + c) * width;
                    match dtype {
                        "f32" => payload[p..p + 4].copy_from_slice(&(v as f32).to_le_bytes()),
                        "i32" => payload[p..p + 4].copy_from_slice(&(v as i32).to_le_bytes()),
                        _ => payload[p..p + 8].copy_from_slice(&v.to_le_bytes()),
                    }
                }
            }
        }
        let filename = format!("{dtype}.slab");
        fs::write(destination.join(&filename), &payload).map_err(|e| e.to_string())?;
        files.insert(dtype.into(),json!({"path":filename,"dtype":dtype,"shape":[size,n],"bytes":payload.len(),"sha256":sha(&payload)}));
    }
    let metadata: Vec<_> = columns.iter().map(|(_, m)| m).collect();
    let manifest = json!({"schema":"arwen.noahowp.initial.v1","source_revision":rules["source_revision"],"request_sha256":sha(&bytes),"schema_sha256":sha(include_bytes!("../../../gpuwm/hydro/noahowp/schema.json")),"rules_sha256":sha(include_bytes!("rules.json")),"parameter_sha256":sha(&serde_json::to_vec(&metadata).unwrap()),"columns":metadata,"files":files,"parallelism":{"threads":rayon::current_num_threads(),"method":"Rayon parallel catchment initialization"},"elapsed_ms":before.elapsed().as_secs_f64()*1000.});
    fs::write(
        destination.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(manifest)
}
/// Setup-only, verified payload transport. Failure leaves destination untouched.
#[no_mangle]
pub unsafe extern "C" fn arwen_noahowp_read_into(
    file: *const c_char,
    hash: *const c_char,
    dest: *mut u8,
    len: usize,
) -> i32 {
    if file.is_null() || hash.is_null() || dest.is_null() {
        return 1;
    }
    let r = std::panic::catch_unwind(|| -> Result<()> {
        let p = CStr::from_ptr(file).to_str().map_err(|e| e.to_string())?;
        let expected = CStr::from_ptr(hash).to_str().map_err(|e| e.to_string())?;
        let b = read(Path::new(p))?;
        if b.len() != len || sha(&b) != expected {
            return Err("payload size/digest mismatch".into());
        }
        std::ptr::copy_nonoverlapping(b.as_ptr(), dest, len);
        Ok(())
    });
    match r {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 2,
        Err(_) => 3,
    }
}
