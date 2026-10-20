// SPDX-License-Identifier: Apache-2.0
use nextgen_verify::*;
use rayon::prelude::*;
use reqwest::blocking::Client;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
fn read<T: DeserializeOwned>(p: impl AsRef<Path>) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn save(p: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<()> {
    fs::write(p, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn fingerprint(p: impl AsRef<Path>) -> Result<Value> {
    let p = p.as_ref();
    Ok(json!({"path":p,"sha256":format!("{:x}",Sha256::digest(fs::read(p)?))}))
}
fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}
fn measurement(name: &str, begin: Instant, parallel: &str) -> Value {
    json!({"stage":name,"wall_ms":begin.elapsed().as_secs_f64()*1000.,"parallelism":parallel,"finished_unix_ms":unix_ms().to_string()})
}
fn table(p: &str) -> Result<Table> {
    let mut t: Table = read(p)?;
    if t.schema != "nextgen-gauge-table-v1" {
        return Err("unrecognized gauge table schema".into());
    }
    if let Some(from) = &t.sources_from {
        let path = Path::new(p).parent().unwrap_or(Path::new(".")).join(from);
        let base: Table = read(&path)?;
        if base.schema != "nextgen-gauge-table-v1" {
            return Err("sources_from names a file that is not a gauge table".into());
        }
        let own: BTreeSet<_> = t.sources.iter().map(|s| s.id.clone()).collect();
        t.sources.extend(base.sources.into_iter().filter(|s| !own.contains(&s.id)));
        for (k, v) in base.layouts {
            t.layouts.entry(k).or_insert(v);
        }
        if t.matching.is_none() {
            t.matching = base.matching;
        }
    }
    let source_ids: BTreeSet<_> = t.sources.iter().map(|s| &s.id).collect();
    if source_ids.len() != t.sources.len() {
        return Err("source ids must be unique".into());
    }
    if let Some(bad) = t.stations.iter().find(|s| s.source.as_ref().is_some_and(|x| !source_ids.contains(x))) {
        return Err(format!("station {} names unknown source {:?}", bad.station_id, bad.source).into());
    }
    let ids: BTreeSet<_> = t.stations.iter().map(|s| &s.station_id).collect();
    if ids.len() != t.stations.len() || t.stations.is_empty() {
        return Err("station rows must be nonempty and unique".into());
    }
    Ok(t)
}
fn source<'a>(t: &'a Table, id: &str) -> Result<&'a Source> {
    t.sources
        .iter()
        .find(|s| s.id == id)
        .ok_or_else(|| format!("unknown source {id}").into())
}
fn safe_name(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' }).collect()
}
/// POST a form, then follow one redirect to the same origin (the download link).
fn post_form(client: &Client, url: &str, form: &[(String, String)], follow: bool, agent: Option<&str>) -> Result<Vec<u8>> {
    let mut last = String::new();
    for attempt in 0..3 {
        let mut request = client.post(url).form(form);
        if let Some(a) = agent {
            request = request.header(reqwest::header::USER_AGENT, a);
        }
        match request.send() {
            Ok(response) => {
                let status = response.status();
                if status.is_redirection() && follow {
                    let location = response
                        .headers()
                        .get(reqwest::header::LOCATION)
                        .and_then(|v| v.to_str().ok())
                        .ok_or("redirect without location")?;
                    let next = reqwest::Url::parse(url)?.join(location)?;
                    if next.origin() != reqwest::Url::parse(url)?.origin() {
                        return Err("form download redirected to another origin".into());
                    }
                    return download_as(client, next.as_str(), agent);
                }
                if status.is_success() {
                    return Ok(response.bytes()?.to_vec());
                }
                if status.as_u16() != 429 && !status.is_server_error() {
                    return Err(format!("HTTP{status}: POST {url}").into());
                }
                last = format!("HTTP{status}");
            }
            Err(e) => last = e.to_string(),
        }
        if attempt < 2 {
            std::thread::sleep(Duration::from_millis(500 * (attempt + 1)));
        }
    }
    Err(format!("form request failed after 3 attempts: {url}: {last}").into())
}
fn download_as(client: &Client, url: &str, agent: Option<&str>) -> Result<Vec<u8>> {
    let mut last = String::new();
    for attempt in 0..4 {
        let mut request = client.get(url);
        if let Some(a) = agent {
            request = request.header(reqwest::header::USER_AGENT, a);
        }
        if reqwest::Url::parse(url)?.host_str() == Some("api.waterdata.usgs.gov") {
            if let Ok(key) = std::env::var("USGS_API_KEY") {
                request = request.header("X-Api-Key", key);
            }
        }
        match request.send() {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    return Ok(response.bytes()?.to_vec());
                }
                if status.as_u16() != 429 && !status.is_server_error() {
                    return Err(format!("HTTP{status}: {url}").into());
                }
                // A rate limit names its own wait: honour a short one, report a long one.
                let wait = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.trim().parse::<u64>().ok());
                if status.as_u16() == 429 {
                    match wait {
                        Some(w) if w <= 60 => {
                            std::thread::sleep(Duration::from_secs(w));
                            last = format!("HTTP{status} (waited {w} s)");
                            continue;
                        }
                        Some(w) => {
                            return Err(format!(
                                "HTTP{status}: {url}: the service asks to retry after {w} s (rate limit; for USGS set USGS_API_KEY)"
                            )
                            .into())
                        }
                        None => {}
                    }
                }
                last = format!("HTTP{status}");
            }
            Err(e) => last = e.to_string(),
        }
        // Some national services answer 504 intermittently; back off 0.5, 1, 2 s.
        if attempt < 3 {
            std::thread::sleep(Duration::from_millis(500 << attempt));
        }
    }
    Err(format!("request failed after 4 attempts: {url}: {last}").into())
}
fn url_value(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}
fn fetch(a: &[String]) -> Result<()> {
    if a.len() != 5 {
        return Err("fetch TABLE SOURCE START_UTC END_UTC OUT".into());
    }
    let t = table(&a[0])?;
    let spec = source(&t, &a[1])?;
    let (start, end) = (timestamp(&a[2])?, timestamp(&a[3])?);
    if start >= end {
        return Err("start must precede end".into());
    }
    let out = Path::new(&a[4]);
    fs::create_dir_all(out)?;
    let began = Instant::now();
    let client = Client::builder()
        .timeout(Duration::from_secs(45))
        .user_agent("gpuwm-nextgen-gauge-verification/0.1")
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    world::admit(spec)?;
    let stations: Vec<&Station> = t.stations.iter().filter(|s| world::belongs(s, spec)).collect();
    if stations.is_empty() {
        return Err(format!("the table has no station rows for source {}", spec.id).into());
    }
    let method = spec.request.as_ref().map_or("GET", |r| r.method.as_str());
    if !["GET", "POST_FORM"].contains(&method) {
        return Err(format!("unknown request method {method}").into());
    }
    let agent = spec.request.as_ref().and_then(|r| r.user_agent.as_deref());
    let raw_value = |v: &str| v.to_string();
    // Network politeness: at most 8 requests in flight against one agency's service.
    let in_flight = spec.request.as_ref().and_then(|r| r.max_in_flight).unwrap_or(8).max(1);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(stations.len().clamp(1, in_flight)).build()?;
    // One station's failure (no archive for that series, a refused owner) is
    // recorded and the others still land; the command then exits non-zero.
    let outcomes: Vec<(String, Result<_>)> = pool.install(|| stations
        .par_iter()
        .map(|s| (s.station_id.clone(), (|| -> Result<_> {
            let s: &Station = s;
            let safe = safe_name(&s.station_id);
            let meta = match &spec.metadata {
                Some(m) => {
                    let url = world::expand(&m.url, s, &a[2], &a[3], None, &url_value)?;
                    let raw = download_as(&client, &url, m.user_agent.as_deref())?;
                    let name = format!("{safe}-metadata.json");
                    fs::write(out.join(&name), &raw)?;
                    let doc: Value = serde_json::from_slice(&raw)?;
                    Some((doc, json!({"file":name,"url":url,"sha256":format!("{:x}",Sha256::digest(&raw)),"raw_bytes":raw.len()})))
                }
                None => None,
            };
            let info = match world::station_info(spec, s, meta.as_ref().map(|m| &m.0)) {
                Ok(i) => Some(i),
                Err(e) if spec.metadata.is_some() => return Err(e),
                Err(_) => None,
            };
            let mut url = world::expand(&spec.url, s, &a[2], &a[3], meta.as_ref().map(|m| &m.0), &url_value)?;
            let form: Vec<(String, String)> = match &spec.request {
                Some(r) if r.method == "POST_FORM" => r
                    .form
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), world::expand(v, s, &a[2], &a[3], meta.as_ref().map(|m| &m.0), &raw_value)?)))
                    .collect::<Result<_>>()?,
                _ => vec![],
            };
            let origin = reqwest::Url::parse(&url)?.origin();
            let (mut visited, mut rows, mut pages, mut rejects, mut owners) = (BTreeSet::new(), Vec::new(), Vec::new(), 0, BTreeSet::new());
            loop {
                if !visited.insert(url.clone()) {
                    return Err("cyclic source pagination".into());
                }
                // A response that is not the declared container (an error page from a
                // crowded service) is retried after a pause, then reported.
                let mut tries = 0;
                let (raw, page) = loop {
                    let raw = if method == "POST_FORM" {
                        post_form(&client, &url, &form, spec.request.as_ref().is_some_and(|r| r.follow_same_origin_redirect), agent)?
                    } else {
                        download_as(&client, &url, agent)?
                    };
                    match world::decode(&raw, spec.container.as_ref()) {
                        Ok(page) => break (raw, page),
                        Err(e) if tries < 3 => {
                            tries += 1;
                            eprintln!("{}: undecodable response ({e}); retry {tries}", s.station_id);
                            std::thread::sleep(Duration::from_secs(2 << tries));
                        }
                        Err(e) => {
                            fs::write(out.join(format!("{safe}-rejected-response.bin")), &raw)?;
                            return Err(format!("{}: {e}", s.station_id).into());
                        }
                    }
                };
                let ext = if spec.container.is_some() { "zip" } else { "json" };
                let name = format!("{safe}-{:04}.{ext}", pages.len());
                fs::write(out.join(&name), &raw)?;
                if let Some(gate) = &spec.owner_gate {
                    owners.insert(world::owner_attribution(gate, &page, &s.station_id)?);
                }
                let (mut decoded, rejected) = normalize(&page, &spec.layout, std::slice::from_ref(s))?;
                decoded.retain(|r| timestamp(&r.time).is_ok_and(|t| t >= start && t <= end));
                rejects += rejected;
                rows.extend(decoded);
                pages.push(json!({"file":name,"url":url,"form":form,"sha256":format!("{:x}",Sha256::digest(&raw)),"raw_bytes":raw.len()}));
                if method != "GET" || spec.links.is_empty() {
                    break;
                }
                let next = select(&page, &spec.links)
                    .into_iter()
                    .find(|v| field(v, &spec.relation).as_deref() == Some("next"))
                    .and_then(|v| field(v, &spec.href));
                match next {
                    Some(next) => {
                        let resolved = reqwest::Url::parse(&url)?.join(&next)?;
                        if resolved.origin() != origin {
                            return Err("pagination changed source origin".into());
                        }
                        url = resolved.to_string();
                    }
                    None => break,
                }
            }
            let rows = ordered(&rows)?;
            let mut info = info;
            if let Some(i) = info.as_mut() {
                if !owners.is_empty() {
                    i.attribution = owners.iter().cloned().collect::<Vec<_>>().join("; ");
                }
            }
            let receipt = json!({"station":s,"samples":rows.len(),"rejected":rejects,"metadata":meta.as_ref().map(|m|&m.1),"pages":pages,"owners":owners});
            Ok((rows, receipt, info))
        })()))
        .collect());
    let mut jobs = Vec::new();
    let mut failed = Vec::new();
    for (id, r) in outcomes {
        match r {
            Ok(j) => jobs.push(j),
            Err(e) => failed.push(json!({"station_id": id, "error": e.to_string()})),
        }
    }
    let rows: Vec<_> = jobs.iter().flat_map(|p| p.0.iter().cloned()).collect();
    save(out.join("observations.json"), &rows)?;
    let infos: Vec<_> = jobs.iter().filter_map(|j| j.2.clone()).collect();
    save(out.join("stations.json"), &infos)?;
    let mut lines: BTreeSet<String> = infos.iter().map(|i| i.attribution.clone()).filter(|a| !a.is_empty()).collect();
    if let Some(l) = &spec.licence {
        lines.insert(format!("Licence: {}", l.licence));
    }
    fs::write(out.join("ATTRIBUTION.txt"), lines.into_iter().collect::<Vec<_>>().join("\n") + "\n")?;
    let missing: Vec<_> = jobs
        .iter()
        .filter(|(rows, _, _)| rows.is_empty())
        .map(|(_, r, _)| r["station"]["station_id"].clone())
        .collect();
    let receipt = json!({"schema":"nextgen-observation-fetch-v1","source":spec,"table":fingerprint(&a[0])?,"start":a[2],"end":a[3],"samples":rows.len(),"stations":jobs.iter().map(|(_,r,_)|r).collect::<Vec<_>>(),"station_records":infos.len(),"missing_stations":missing,"failed_stations":failed,"stages":[measurement("fetch_decode_normalize",began,&format!("Rayon{}workers across{}stations; ordered pages",pool.current_num_threads(),stations.len()))],"routing_step_host_work":false});
    save(out.join("fetch.json"), &receipt)?;
    println!(
        "{} observations; missing{}; failed{}; {}",
        rows.len(),
        missing.len(),
        failed.len(),
        out.display()
    );
    for f in &failed {
        eprintln!("{}: {}", f["station_id"], f["error"]);
    }
    if !missing.is_empty() || !failed.is_empty() {
        return Err("stations without observations; receipt saved".into());
    }
    Ok(())
}
fn ingest(a: &[String]) -> Result<()> {
    if a.len() != 4 {
        return Err("import TABLE LAYOUT FILES_JSON OUT".into());
    }
    let t = table(&a[0])?;
    let spec = t.layouts.get(&a[1]).ok_or("unknown layout")?;
    let files: Vec<PathBuf> = read(&a[2])?;
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    let began = Instant::now();
    let jobs: Result<Vec<_>> = files
        .par_iter()
        .map(|p| -> Result<_> {
            if let Some(nc) = &spec.netcdf {
                let (rows, rejected) = nextgen_verify::netcdf::decode(p, nc, spec, &t.stations)?;
                return Ok((rows, json!({"input":fingerprint(p)?,"rejected":rejected,"decoder":"pure Rust netcdf-reader"})));
            }
            let document = if p.extension().and_then(|s| s.to_str()) == Some("csv") {
                let mut csv = csv::Reader::from_path(p)?;
                let names = csv.headers()?.clone();
                let mut records = Vec::new();
                for row in csv.records() {
                    let row = row?;
                    let record = names
                        .iter()
                        .zip(row.iter())
                        .map(|(k, v)| (k.to_owned(), Value::String(v.into())))
                        .collect::<serde_json::Map<_, _>>();
                    records.push(Value::Object(record));
                }
                Value::Array(records)
            } else {
                read(p)?
            };
            let (rows, rejected) = normalize(&document, spec, &t.stations)?;
            Ok((rows, json!({"input":fingerprint(p)?,"rejected":rejected})))
        })
        .collect();
    let jobs = jobs?;
    let rows = ordered(
        &jobs
            .iter()
            .flat_map(|(r, _)| r.iter().cloned())
            .collect::<Vec<_>>(),
    )?;
    let have: BTreeSet<_> = rows.iter().map(|r| r.station_id.as_str()).collect();
    let missing: Vec<_> = t
        .stations
        .iter()
        .filter(|s| !have.contains(s.station_id.as_str()))
        .map(|s| &s.station_id)
        .collect();
    save(out.join("hydrographs.json"), &rows)?;
    save(
        out.join("import.json"),
        &json!({"schema":"nextgen-streamflow-import-v1","table":fingerprint(&a[0])?,"layout":spec,"inputs":jobs.iter().map(|j|&j.1).collect::<Vec<_>>(),"samples":rows.len(),"missing_stations":missing,"stages":[measurement("decode_join_normalize",began,&format!("Rayon{}workers across{}files",rayon::current_num_threads(),files.len()))],"routing_step_host_work":false}),
    )?;
    println!("{} imported samples, missing{}", rows.len(), missing.len());
    if !missing.is_empty() {
        return Err("requested station features missing; receipt saved".into());
    }
    Ok(())
}
fn xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
fn hydrograph(station: &str, series: [&Timeline; 3]) -> String {
    named_hydrograph(station, series, ["GPU", "Native", "USGS"])
}
fn named_hydrograph(station: &str, series: [&Timeline; 3], labels: [&str; 3]) -> String {
    let run_series = if series[0].is_empty() && series[1].is_empty() {
        &series[..]
    } else {
        &series[..2]
    };
    let times: Vec<_> = run_series.iter().flat_map(|s| s.keys()).copied().collect();
    let lo = times.iter().min().copied().unwrap_or(0);
    let hi = times.iter().max().copied().unwrap_or(lo);
    let observed_max = series
        .iter()
        .flat_map(|s| s.range(lo..=hi))
        .map(|(_, q)| *q)
        .fold(0., f64::max);
    let qmax = if observed_max > 0. { observed_max } else { 1. };
    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="420"><rect width="1000" height="420" fill="white"/><g font-family="sans-serif" font-size="14"><text x="65" y="25">Station {} — discharge (m³/s); UTC; exact matching</text><text x="65" y="405">{} — {}</text>"##,
        xml(station),
        utc(lo),
        utc(hi)
    );
    for k in 0..5 {
        let y = 350. - k as f64 * 70.;
        let q = qmax * k as f64 / 4.;
        let label = if qmax < 0.1 {
            format!("{q:.3e}")
        } else {
            format!("{q:.2}")
        };
        svg += &format!(
            r##"<path d="M65,{y}H970" stroke="#ddd"/><text x="5" y="{y}">{label}</text>"##
        );
    }
    for (i, (s, (name, color))) in series
        .iter()
        .zip(labels.into_iter().zip(["#c22", "#26c", "#111"]))
        .enumerate()
    {
        if s.is_empty() {
            continue;
        }
        let name = xml(name);
        let coords: Vec<_> = s
            .range(lo..=hi)
            .map(|(&t, &q)| {
                format!(
                    "{:.2},{:.2}",
                    if hi == lo {
                        517.5
                    } else {
                        65. + 905. * (t - lo) as f64 / (hi - lo) as f64
                    },
                    350. - 280. * q / qmax
                )
            })
            .collect();
        svg += &format!(
            r##"<text x="{}" y="50" fill="{color}">{name}</text><polyline points="{}" fill="none" stroke="{color}" stroke-width="2"/>"##,
            650 + i * 110,
            coords.join(" ")
        );
        if coords.len() == 1 {
            let (x, y) = coords[0].split_once(',').unwrap();
            svg += &format!(
                r##"<circle cx="{x}" cy="{y}" r="{}" stroke="{color}" fill="none" stroke-width="2"/>"##,
                8 - i * 2
            );
        }
    }
    svg + "</g></svg>"
}
fn compare(a: &[String]) -> Result<()> {
    if a.len() != 4 || a[1].trim().is_empty() {
        return Err("compare PREDICTION_JSON LABEL OBSERVATIONS_JSON OUT".into());
    }
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    let began = Instant::now();
    let paths = [&a[0], &a[2]];
    let inputs: Result<Vec<_>> = paths
        .par_iter()
        .map(|p| index(&read::<Vec<Observation>>(p)?))
        .collect();
    let inputs = inputs?;
    let load = measurement(
        "load_index",
        began,
        &format!("Rayon{}workers/2inputs", rayon::current_num_threads()),
    );
    let began = Instant::now();
    let stations: BTreeSet<_> = inputs.iter().flat_map(|s| s.keys()).cloned().collect();
    let reports: Result<Vec<_>> = stations.par_iter().map(|station| -> Result<_> {
        let empty = Timeline::new();
        let predicted = inputs[0].get(station).unwrap_or(&empty);
        let observed = inputs[1].get(station).unwrap_or(&empty);
        let pairs: Vec<_> = predicted.iter()
            .filter_map(|(&t, &q)| observed.get(&t).map(|&o| (t, q, o))).collect();
        let filename = format!("comparison-{}.svg", &format!("{:x}", Sha256::digest(station.as_bytes()))[..12]);
        fs::write(out.join(&filename), named_hydrograph(station,
            [predicted, &empty, observed], [&a[1], "", "Observations"]))?;
        Ok(json!({"station_id":station,"prediction_samples":predicted.len(),
            "observation_samples":observed.len(),"unmatched_prediction_samples":predicted.len()-pairs.len(),
            "unmatched_observation_samples":observed.len()-pairs.len(),
            "scores":metrics(&pairs),"picture":filename}))
    }).collect();
    let reports = reports?;
    let valid = !reports.is_empty()
        && reports
            .iter()
            .all(|r| r["scores"]["samples"].as_u64().unwrap_or(0) > 0);
    save(
        out.join("comparison.json"),
        &json!({"schema":"nextgen-observation-comparison-v1",
        "scope":"prediction against observations only; no native/GPU replay or hydrologic acceptance",
        "tool_source_rev":env!("GPUWM_BRIDGE_SOURCE_REV"),"prediction_label":a[1],
        "inputs":paths.iter().map(fingerprint).collect::<Result<Vec<_>>>()?,"stations":reports,
        "observation_support":"two-way exact UTC intersection; no interpolation",
        "chart_support":"full input series within prediction time span; scores use exact pairs only",
        "peak_timing":"prediction minus observations in seconds; positive late; earliest tied peak on shared support",
        "kge":"Gupta2009: correlation, stdev ratio, mean ratio",
        "stages":[load,measurement("align_score_analysis_svg",began,
            &format!("Rayon{}workers across stations; ordered f64 reductions",rayon::current_num_threads()))],
        "routing_step_host_work":false}),
    )?;
    println!(
        "{} stations; observation comparison valid={valid}",
        stations.len()
    );
    if !valid {
        return Err(
            "no exact observation pairs for requested stations; comparison retained".into(),
        );
    }
    Ok(())
}
fn score(a: &[String]) -> Result<()> {
    if a.len() != 4 {
        return Err("score MODEL_JSON NATIVE_JSON OBSERVATIONS_JSON OUT".into());
    }
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    let began = Instant::now();
    let inputs: Result<Vec<_>> = a[..3]
        .par_iter()
        .map(|p| index(&read::<Vec<Observation>>(p)?))
        .collect();
    let series = inputs?;
    let load = measurement(
        "load_index",
        began,
        &format!("Rayon{}workers/3inputs", rayon::current_num_threads()),
    );
    let began = Instant::now();
    let stations: BTreeSet<_> = series.iter().flat_map(|s| s.keys()).cloned().collect();
    let reports: Vec<_> = stations
        .par_iter()
        .map(|s| assess(s, &series[0], &series[1], &series[2]))
        .collect();
    let scoring = measurement(
        "align_score",
        began,
        &format!(
            "Rayon{}workers across stations; ordered f64 reductions",
            rayon::current_num_threads()
        ),
    );
    let began = Instant::now();
    let charts: Result<Vec<_>> = stations
        .par_iter()
        .map(|s| -> Result<_> {
            let empty = Timeline::new();
            let data = [0, 1, 2].map(|i| series[i].get(s).unwrap_or(&empty));
            let filename = format!(
                "station-{}.svg",
                format!("{:x}", Sha256::digest(s.as_bytes()))[..12].to_owned()
            );
            fs::write(out.join(&filename), hydrograph(s, data))?;
            Ok(json!({"station_id":s,"file":filename}))
        })
        .collect();
    let valid = !reports.is_empty()
        && reports
            .iter()
            .all(|r| r.model_vs_reference.samples > 0 && r.model_vs_observations.samples > 0);
    let bitwise = !reports.is_empty()
        && reports
            .iter()
            .all(|r| r.complete_reference_support && r.model_vs_reference.differing_f64_words == 0);
    save(
        out.join("scores.json"),
        &json!({"schema":"nextgen-gauge-scores-v1","tool_source_rev":env!("GPUWM_BRIDGE_SOURCE_REV"),"inputs":a[..3].iter().map(fingerprint).collect::<Result<Vec<_>>>()?,"stations":reports,"complete_support_bitwise_f64":bitwise,"all_stations_have_observation_pairs":valid,"observation_support":"same three-way exact UTC intersection; no interpolation","peak_timing":"model minus reference in seconds; positive late; earliest tied peak; null for one sample or constant series","kge":"Gupta2009: correlation, stdev ratio, mean ratio","stages":[load,scoring,measurement("analysis_svg",began,&format!("Rayon{}workers across stations",rayon::current_num_threads()))],"charts":charts?,"routing_step_host_work":false}),
    )?;
    println!(
        "{} stations; full-support f64bitwise={bitwise}",
        stations.len()
    );
    if !valid {
        return Err(
            "no matching reference/observation samples for requested stations; scores retained"
                .into(),
        );
    }
    Ok(())
}
fn stations(a: &[String]) -> Result<()> {
    if a.len() != 5 {
        return Err("stations ATTRIBUTES_JSON RECORDS_PTR STATION_PTR FEATURE_PTR OUT_JSON".into());
    }
    let began = Instant::now();
    let root: Value = read(&a[0])?;
    let jobs: Vec<_> = select(&root, &a[1])
        .par_iter()
        .filter_map(|r| {
            let station = field(r, &a[2])?;
            let station = station.trim();
            if station.is_empty() {
                return None;
            }
            let feature = field(r, &a[3])?;
            Some(Station {
                station_id: station.into(),
                feature_id: Some(feature),
                time_series_id: None,
                ..Default::default()
            })
        })
        .collect();
    let mut rows = std::collections::BTreeMap::new();
    for row in jobs {
        if let Some(before) = rows.insert(row.station_id.clone(), row.clone()) {
            if before.feature_id != row.feature_id {
                return Err(format!(
                    "station{}maps multiple features; select metadata rows",
                    row.station_id
                )
                .into());
            }
        }
    }
    save(&a[4], &rows.values().collect::<Vec<_>>())?;
    save(
        Path::new(&a[4]).with_extension("receipt.json"),
        &json!({"input":fingerprint(&a[0])?,"records":a[1],"station_field":a[2],"feature_field":a[3],"stations":rows.len(),"stage":measurement("station_feature_join",began,&format!("Rayon{}workers across metadata rows",rayon::current_num_threads()))}),
    )?;
    println!("{} station/feature table rows", rows.len());
    Ok(())
}
fn observations(a: &[String]) -> Result<()> {
    if a.len() != 2 {
        return Err("observations OBSERVATIONS_JSON OUT".into());
    }
    let out = Path::new(&a[1]);
    fs::create_dir_all(out)?;
    let began = Instant::now();
    let data = index(&read::<Vec<Observation>>(&a[0])?)?;
    // Station names from the fetch's common station records, when present beside the observations.
    let names: std::collections::BTreeMap<String, String> = Path::new(&a[0])
        .parent()
        .map(|p| p.join("stations.json"))
        .filter(|p| p.is_file())
        .and_then(|p| read::<Vec<world::StationInfo>>(p).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|i| (i.station_id.clone(), i.name.unwrap_or(i.station_id)))
        .collect();
    let jobs:Result<Vec<_>>=data.par_iter().map(|(station,s)|->Result<_>{let empty=Timeline::new();let filename=format!("observed-{}.svg",format!("{:x}",Sha256::digest(station.as_bytes()))[..12].to_owned());let agency=station.split_once(':').map_or("USGS",|(a,_)|a);fs::write(out.join(&filename),named_hydrograph(station,[&empty,&empty,s],["","",agency]))?;let peak=s.iter().max_by(|a,b|a.1.total_cmp(b.1).then_with(||b.0.cmp(a.0))).ok_or("empty observed series")?;Ok(json!({"station_id":station,"name":names.get(station),"samples":s.len(),"peak_m3s":peak.1,"peak_utc":utc(*peak.0),"picture":filename}))}).collect();
    fs::write(out.join("observed-all.svg"), panel(&data, &names))?;
    save(
        out.join("observed.json"),
        &json!({"scope":"gauge observations only; no native/GPU skill claim","input":fingerprint(&a[0])?,"stations":jobs?,"panel":"observed-all.svg","stage":measurement("observed_analysis_charts",began,&format!("Rayon{}workers across stations",rayon::current_num_threads()))}),
    )?;
    Ok(())
}
fn replay(a: &[String]) -> Result<()> {
    if a.len() != 6 {
        return Err("replay TABLE SOURCE STATION RAW_JSON EXPECTED_JSON OUT".into());
    }
    let t = table(&a[0])?;
    let src = source(&t, &a[1])?;
    let station = t
        .stations
        .iter()
        .find(|s| s.station_id == a[2])
        .ok_or("unknown station")?;
    let out = Path::new(&a[5]);
    fs::create_dir_all(out)?;
    let began = Instant::now();
    let (rows, rejected) = normalize(
        &read::<Value>(&a[3])?,
        &src.layout,
        std::slice::from_ref(station),
    )?;
    let actual = index(&rows)?;
    let expected = index(&read::<Vec<Observation>>(&a[4])?)?;
    let pass = actual.keys().eq(expected.keys())
        && actual.iter().all(|(s, v)| {
            expected.get(s).is_some_and(|e| {
                v.keys().eq(e.keys()) && v.iter().all(|(t, q)| q.to_bits() == e[t].to_bits())
            })
        });
    save(
        out.join("replay.json"),
        &json!({"scope":"USGS decode and UTC/SI normalization, not hydrologic module proof","bitwise_f64":pass,"samples":rows.len(),"rejected":rejected,"raw":fingerprint(&a[3])?,"expected":fingerprint(&a[4])?,"stage":measurement("decode_replay",began,"one recorded page; fetch parallelizes stations")}),
    )?;
    println!("{} real observation words bitwise={pass}", rows.len());
    if !pass {
        return Err("decoded observation word mismatch".into());
    }
    Ok(())
}
/// Viridis-like ramp: one distinct colour per series, dark = largest mean flow.
fn ramp(i: usize, n: usize) -> String {
    const STOPS: [[f64; 3]; 5] = [[68., 1., 84.], [59., 82., 139.], [33., 145., 140.], [94., 201., 98.], [253., 231., 37.]];
    let x = if n > 1 { i as f64 / (n - 1) as f64 * 0.92 } else { 0.0 } * 4.0;
    let k = (x.floor() as usize).min(3);
    let f = x - k as f64;
    let c: Vec<u8> = (0..3).map(|j| (STOPS[k][j] + (STOPS[k + 1][j] - STOPS[k][j]) * f).round() as u8).collect();
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}
/// Every station's observed series on one chart (m3/s, UTC), legend by mean flow.
fn panel(data: &Hydrographs, names: &std::collections::BTreeMap<String, String>) -> String {
    let lo = data.values().filter_map(|s| s.keys().next()).min().copied().unwrap_or(0);
    let hi = data.values().filter_map(|s| s.keys().next_back()).max().copied().unwrap_or(lo + 1).max(lo + 1);
    let qmax = data.values().flat_map(|s| s.values()).fold(0.0f64, |a, b| a.max(*b)).max(1e-9);
    let mut order: Vec<_> = data.iter().collect();
    let mean = |s: &Timeline| s.values().sum::<f64>() / s.len().max(1) as f64;
    order.sort_by(|a, b| mean(b.1).total_cmp(&mean(a.1)).then_with(|| a.0.cmp(b.0)));
    let (w, h, l, r, t, b) = (1440.0, 600.0, 80.0, 540.0, 40.0, 50.0);
    let (pw, ph) = (w - l - r, h - t - b);
    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="{w}" height="{h}" fill="white"/><g font-family="sans-serif" font-size="13"><text x="{l}" y="24" font-size="15">Observed discharge (m³/s), UTC, {} to {}</text>"##,
        utc(lo),
        utc(hi)
    );
    for k in 0..=5 {
        let y = t + ph - ph * k as f64 / 5.0;
        svg += &format!(r##"<path d="M{l},{y:.1}H{:.1}" stroke="#ddd"/><text x="8" y="{:.1}">{:.0}</text>"##, l + pw, y + 4.0, qmax * k as f64 / 5.0);
    }
    let days = ((hi - lo) as f64 / 86_400_000.0).ceil() as i64;
    let step = if days > 20 { 5 } else { 1 };
    let day0 = lo - lo.rem_euclid(86_400_000);
    for d in (0..=days).step_by(step as usize) {
        let tm = day0 + d * 86_400_000;
        if tm < lo || tm > hi {
            continue;
        }
        let x = l + pw * (tm - lo) as f64 / (hi - lo) as f64;
        svg += &format!(r##"<path d="M{x:.1},{t}V{:.1}" stroke="#eee"/><text x="{:.1}" y="{:.1}" font-size="11">{}</text>"##, t + ph, x - 18.0, t + ph + 16.0, &utc(tm)[5..10]);
    }
    for (i, (station, s)) in order.iter().enumerate() {
        let color = ramp(i, order.len());
        let pts: Vec<String> = s
            .iter()
            .map(|(&tm, &q)| format!("{:.1},{:.1}", l + pw * (tm - lo) as f64 / (hi - lo) as f64, t + ph - ph * q / qmax))
            .collect();
        let label = names.get(*station).cloned().unwrap_or_else(|| (*station).clone());
        svg += &format!(
            r##"<polyline points="{}" fill="none" stroke="{color}" stroke-width="1.6"/><rect x="{:.1}" y="{:.1}" width="14" height="4" fill="{color}"/><text x="{:.1}" y="{:.1}">{} ({}) mean {:.0}</text>"##,
            pts.join(" "),
            l + pw + 14.0,
            t + 6.0 + 18.0 * i as f64,
            l + pw + 34.0,
            t + 11.0 + 18.0 * i as f64,
            xml(&label),
            xml(station),
            mean(s)
        );
    }
    svg + "</g></svg>"
}
fn match_cmd(a: &[String]) -> Result<()> {
    if a.len() != 4 && a.len() != 5 {
        return Err("match TABLE FABRIC_GPKG STATIONS_JSON OUT [OBSERVATIONS_JSON]".into());
    }
    let t = table(&a[0])?;
    let rules = t.matching.clone().ok_or("the gauge table has no matching rules")?;
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    let began = Instant::now();
    let (reaches, projection, srs, _fabric) = matching::load_fabric(&a[1])?;
    let load = measurement("read_fabric_decode_geometry", began, &format!("Rayon{}workers across{}flowpaths (GeoPackage read: 1 thread)", rayon::current_num_threads(), reaches.len()));
    let stations: Vec<world::StationInfo> = read(&a[2])?;
    // Mean observed flow per gauge (optional) feeds the discharge plausibility rule.
    let mean_flows: std::collections::HashMap<String, f64> = match a.get(4) {
        Some(p) => index(&read::<Vec<Observation>>(p)?)?
            .into_iter()
            .filter(|(_, s)| !s.is_empty())
            .map(|(k, s)| (k, s.values().sum::<f64>() / s.len() as f64))
            .collect(),
        None => Default::default(),
    };
    let began = Instant::now();
    let matches = matching::match_all(&stations, &reaches, &projection, &rules, &mean_flows);
    let stage = measurement("index_match", began, &format!("Rayon{}workers across{}stations; grid index", rayon::current_num_threads(), stations.len()));
    let began = Instant::now();
    let hl = matching::hydrolocations(&matches, srs);
    save(out.join("gauges.json"), &json!({"schema":"nextgen-gauge-bindings-v1", "rows": matches.iter()
        .filter_map(|m| m.matched.as_ref().map(|r| json!({"feature_id":r.flowpath_id,
            "station_id":m.station_id,"matching_rule":m.rule,"distance_m":r.distance_m,
            "drainage_area_qualified":m.gauge_area_km2.is_some()}))).collect::<Vec<_>>()}))?;
    save(out.join("matches.json"), &matches)?;
    save(out.join("hydrolocations.json"), &hl)?;
    fs::write(out.join("match-map.svg"), match_map(&reaches, &matches, &stations))?;
    let draw = measurement("write_rows_map", began, "1 thread (two JSON files and one SVG)");
    let matched = matches.iter().filter(|m| m.matched.is_some()).count();
    save(
        out.join("match.json"),
        &json!({"schema":"nextgen-gauge-match-v1","tool_source_rev":env!("GPUWM_BRIDGE_SOURCE_REV"),"fabric":fingerprint(&a[1])?,"stations":fingerprint(&a[2])?,"observations":a.get(4).map(fingerprint).transpose()?,"table":fingerprint(&a[0])?,"rules":rules,"projection":format!("{projection:?}"),"flowpaths":reaches.len(),"gauges":stations.len(),"matched":matched,"unmatched":matches.iter().filter(|m|m.matched.is_none()).map(|m|json!({"station_id":m.station_id,"reason":m.reason})).collect::<Vec<_>>(),"stages":[load,stage,draw]}),
    )?;
    println!("{matched} of {} gauges matched to {} flowpaths; {}", stations.len(), reaches.len(), out.display());
    Ok(())
}
fn match_map(reaches: &[matching::Reach], matches: &[matching::Match], stations: &[world::StationInfo]) -> String {
    let pts: Vec<[f64; 2]> = matches.iter().map(|m| m.projected_xy).collect();
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in &pts {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let pad = ((x1 - x0).max(y1 - y0) * 0.15).max(20_000.0);
    // Extra room on the east for gauge labels.
    let (x0, y0, x1, y1) = (x0 - pad, y0 - pad, x1 + pad + 0.6 * (x1 - x0 + 2.0 * pad), y1 + pad);
    let (w, h) = (900.0, 900.0 * (y1 - y0) / (x1 - x0)).max_by_aspect();
    let sx = |x: f64| (x - x0) / (x1 - x0) * w;
    let sy = |y: f64| h - (y - y0) / (y1 - y0) * h;
    let matched: BTreeSet<&str> = matches.iter().filter_map(|m| m.matched.as_ref().map(|c| c.flowpath_id.as_str())).collect();
    let mut svg = format!(r##"<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0}" height="{:.0}"><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12">"##, h + 30.0);
    let in_view = |r: &matching::Reach| r.line.iter().any(|p| p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1);
    // Dense fabrics: draw flowpaths draining at least 100 km2 (and every matched one).
    let min_area = if reaches.iter().filter(|r| in_view(r)).count() > 4000 { 100.0 } else { 0.0 };
    for r in reaches {
        if !in_view(r) || (r.area_outlet_km2 < min_area && !matched.contains(r.id.as_str())) {
            continue;
        }
        let width = (r.area_outlet_km2.max(1.0).log10() - 1.5).clamp(0.3, 4.0);
        let color = if matched.contains(r.id.as_str()) { "#d62728" } else { "#4a7fb5" };
        let d: Vec<String> = r.line.iter().map(|p| format!("{:.1},{:.1}", sx(p[0]), sy(p[1]))).collect();
        svg += &format!(r##"<polyline points="{}" fill="none" stroke="{color}" stroke-width="{width:.2}"/>"##, d.join(" "));
    }
    for (m, s) in matches.iter().zip(stations) {
        let (x, y) = (sx(m.projected_xy[0]), sy(m.projected_xy[1]));
        let label = s.name.clone().unwrap_or_else(|| s.station_id.clone());
        let mark = if m.matched.is_some() { "#111" } else { "#ff7f0e" };
        svg += &format!(r##"<circle cx="{x:.1}" cy="{y:.1}" r="4" fill="{mark}"/>"##);
        // Label every gauge on small maps; on crowded maps only the unmatched ones.
        if matches.len() <= 30 || m.matched.is_none() {
            svg += &format!(r##"<text x="{:.1}" y="{:.1}">{}</text>"##, x + 6.0, y - 4.0, xml(&label));
        }
    }
    svg += &format!(r##"<text x="8" y="{:.0}">Flowpaths{} (blue, width by drainage area), matched flowpath red, gauges black (orange = unmatched). Fabric CRS metres.</text></g></svg>"##, h + 20.0, if min_area > 0.0 { " draining 100 km2 or more" } else { "" });
    svg
}
trait Aspect {
    fn max_by_aspect(self) -> (f64, f64);
}
impl Aspect for (f64, f64) {
    fn max_by_aspect(self) -> (f64, f64) {
        if self.1 > 1100.0 {
            (self.0 * 1100.0 / self.1, 1100.0)
        } else {
            self
        }
    }
}
/// Compare matches with a fabric's own gauge links (flowpath_attributes.rl_gages and
/// hydrolocations at nexuses): the check that the matcher reproduces a curated fabric.
fn match_check(a: &[String]) -> Result<()> {
    if a.len() != 3 {
        return Err("match-check FABRIC_GPKG MATCHES_JSON OUT_JSON".into());
    }
    let (fabric, _) = hydro_world::gpkg::read(&a[0])?;
    let matches: Vec<Value> = read(&a[1])?;
    let mut truth: std::collections::BTreeMap<String, BTreeSet<String>> = Default::default();
    for f in &fabric.flowpath_attributes {
        if let (Some(id), Some(g)) = (&f.id, &f.rl_gages) {
            for s in g.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                truth.entry(s.to_string()).or_default().insert(id.clone());
            }
        }
    }
    let nex_to: std::collections::HashMap<&str, &str> = fabric.nexus.iter().filter_map(|n| Some((n.id.as_deref()?, n.toid.as_deref()?))).collect();
    for h in &fabric.hydrolocations {
        if h.hl_reference.as_deref() != Some("Gages") {
            continue;
        }
        let (Some(link), Some(nex)) = (&h.hl_link, &h.id) else { continue };
        let e = truth.entry(link.clone()).or_default();
        for f in fabric.flowpaths.iter().filter(|f| f.toid.as_deref() == Some(nex.as_str())) {
            e.extend(f.id.clone());
        }
        if let Some(to) = nex_to.get(nex.as_str()) {
            e.insert(to.to_string());
        }
    }
    let rows: Vec<Value> = matches
        .iter()
        .map(|m| {
            let id = m["station_id"].as_str().unwrap_or_default();
            let got = m["matched"]["flowpath_id"].as_str();
            let want = truth.get(id);
            let verdict = match (got, want) {
                (_, None) => "no fabric link",
                (None, Some(_)) => "unmatched",
                (Some(g), Some(w)) if w.contains(g) => "agrees",
                _ => "differs",
            };
            json!({"station_id":id,"matched":got,"fabric_links":want,"verdict":verdict,"distance_m":m["matched"]["distance_m"],"point_error":m["matched"]["point_error"]})
        })
        .collect();
    let count = |v: &str| rows.iter().filter(|r| r["verdict"] == v).count();
    let summary = json!({"agrees":count("agrees"),"differs":count("differs"),"unmatched":count("unmatched"),"no_fabric_link":count("no fabric link")});
    save(&a[2], &json!({"fabric":fingerprint(&a[0])?,"matches":fingerprint(&a[1])?,"summary":summary,"stations":rows}))?;
    println!("{summary}");
    Ok(())
}
/// Common station records (location, drainage area, licence) without fetching values.
/// A station whose metadata request fails is reported, not fatal.
fn describe(a: &[String]) -> Result<()> {
    if a.len() != 3 {
        return Err("describe TABLE SOURCE OUT".into());
    }
    let t = table(&a[0])?;
    let spec = source(&t, &a[1])?;
    world::admit(spec)?;
    let out = Path::new(&a[2]);
    fs::create_dir_all(out)?;
    // Resume: records already in OUT/stations.json (an earlier, rate-limited run) are kept.
    let kept: Vec<world::StationInfo> = read(out.join("stations.json")).unwrap_or_default();
    let have: BTreeSet<String> = kept.iter().map(|k| k.station_id.clone()).collect();
    let stations: Vec<&Station> = t.stations.iter().filter(|s| world::belongs(s, spec) && !have.contains(&s.station_id)).collect();
    let began = Instant::now();
    let client = Client::builder().timeout(Duration::from_secs(45)).user_agent("gpuwm-nextgen-gauge-verification/0.1").redirect(reqwest::redirect::Policy::none()).build()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(stations.len().clamp(1, 8)).build()?;
    let results: Vec<(String, Result<world::StationInfo>, Value)> = pool.install(|| {
        stations
            .par_iter()
            .map(|s| {
                let s: &Station = s;
                let r = (|| -> Result<(world::StationInfo, Value)> {
                    let Some(m) = &spec.metadata else {
                        return Ok((world::station_info(spec, s, None)?, Value::Null));
                    };
                    let url = world::expand(&m.url, s, "", "", None, &url_value)?;
                    let raw = download_as(&client, &url, m.user_agent.as_deref())?;
                    let name = format!("{}-metadata.json", safe_name(&s.station_id));
                    fs::write(out.join(&name), &raw)?;
                    let doc: Value = serde_json::from_slice(&raw)?;
                    Ok((world::station_info(spec, s, Some(&doc))?, json!({"file":name,"url":url,"sha256":format!("{:x}",Sha256::digest(&raw))})))
                })();
                match r {
                    Ok((i, page)) => (s.station_id.clone(), Ok(i), page),
                    Err(e) => (s.station_id.clone(), Err(e), Value::Null),
                }
            })
            .collect()
    });
    let mut infos: Vec<_> = kept.clone();
    infos.extend(results.iter().filter_map(|(_, r, _)| r.as_ref().ok().cloned()));
    infos.sort_by(|a, b| a.station_id.cmp(&b.station_id));
    let failed: Vec<_> = results.iter().filter_map(|(id, r, _)| r.as_ref().err().map(|e| json!({"station_id":id,"error":e.to_string()}))).collect();
    save(out.join("stations.json"), &infos)?;
    save(
        out.join("describe.json"),
        &json!({"schema":"nextgen-station-describe-v1","source":spec.id,"table":fingerprint(&a[0])?,"described":infos.len(),"kept_from_earlier_run":kept.len(),"failed":failed,"pages":results.iter().map(|r|&r.2).collect::<Vec<_>>(),"stage":measurement("metadata_fetch_decode",began,&format!("Rayon{}workers across{}stations",pool.current_num_threads(),stations.len()))}),
    )?;
    println!("{} described, {} failed; {}", infos.len(), failed.len(), out.display());
    Ok(())
}
/// Station rows for every gauge a source publishes inside a box.
fn discover(a: &[String]) -> Result<()> {
    if a.len() != 4 {
        return Err("discover TABLE SOURCE MINLON,MINLAT,MAXLON,MAXLAT OUT_TABLE_JSON".into());
    }
    let t = table(&a[0])?;
    let spec = source(&t, &a[1])?;
    world::admit(spec)?;
    let d = spec.discover.as_ref().ok_or_else(|| format!("source {} has no discover row", spec.id))?;
    let b: Vec<f64> = a[2].split(',').map(|v| v.trim().parse::<f64>()).collect::<std::result::Result<_, _>>()?;
    let bbox: [f64; 4] = b.try_into().map_err(|_| "bbox needs four numbers")?;
    let began = Instant::now();
    let client = Client::builder().timeout(Duration::from_secs(90)).user_agent("gpuwm-nextgen-gauge-verification/0.1").redirect(reqwest::redirect::Policy::none()).build()?;
    let url = d.url.replace("{bbox}", &format!("{},{},{},{}", bbox[0], bbox[1], bbox[2], bbox[3]));
    let raw = download_as(&client, &url, d.user_agent.as_deref())?;
    let doc: Value = serde_json::from_slice(&raw)?;
    let rows = world::discovered(d, spec, &doc, bbox);
    let table_path = fs::canonicalize(&a[0])?;
    save(
        &a[3],
        &json!({"schema":"nextgen-gauge-table-v1","sources_from":table_path,"stations":rows,
            "discovered":{"source":spec.id,"bbox":bbox,"url":url,"sha256":format!("{:x}",Sha256::digest(&raw)),"raw_bytes":raw.len(),
            "stage":measurement("discover",began,"1 request; parse 1 thread")}}),
    )?;
    println!("{} stations from {} inside {:?}", rows.len(), spec.id, bbox);
    Ok(())
}
fn run() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    let Some(command) = a.first() else {
        return Err("rw_nextgen_verify fetch|import|compare|score|replay|observations|stations|match|match-check|discover ...".into());
    };
    match command.as_str() {
        "--version" => {
            println!(
                "rw_nextgen_verify nextgen-gauge-table-v1 GPUWM_BRIDGE_SOURCE_REV={}",
                env!("GPUWM_BRIDGE_SOURCE_REV")
            );
            Ok(())
        }
        "stations" => stations(&a[1..]),
        "observations" => observations(&a[1..]),
        "fetch" => fetch(&a[1..]),
        "import" => ingest(&a[1..]),
        "score" => score(&a[1..]),
        "compare" => compare(&a[1..]),
        "replay" => replay(&a[1..]),
        "match" => match_cmd(&a[1..]),
        "match-check" => match_check(&a[1..]),
        "discover" => discover(&a[1..]),
        "describe" => describe(&a[1..]),
        _ => Err("unknown verification command".into()),
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offset_query_values_do_not_become_spaces() {
        assert_eq!(
            url_value("2023-01-05T00:00:00+00:00"),
            "2023-01-05T00%3A00%3A00%2B00%3A00"
        );
    }
    #[test]
    fn observation_chart_does_not_claim_model_series() {
        let empty = Timeline::new();
        let observed = Timeline::from([(timestamp("2023-01-05T00:00:00Z").unwrap(), 1.)]);
        let svg = hydrograph("g", [&empty, &empty, &observed]);
        assert!(svg.contains("USGS"));
        assert!(!svg.contains(">GPU<") && !svg.contains(">Native<"));
        assert!(svg.contains("<circle"));
    }
    #[test]
    fn dry_and_low_flow_axes_are_readable() {
        let empty = Timeline::new();
        let dry = Timeline::from([(0, 0.), (1000, 0.)]);
        assert!(hydrograph("g", [&empty, &empty, &dry]).contains(">0.25</text>"));
        let low = Timeline::from([(0, 0.), (1000, 0.01)]);
        assert!(hydrograph("g", [&empty, &empty, &low]).contains(">2.500e-3</text>"));
    }
}
