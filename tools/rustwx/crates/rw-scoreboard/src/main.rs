//! Incremental forecast extraction and paired observation scoring.
//! The station bilinear and masked FSS rules are the verification battery's
//! registered operators, with raw and elevation-adjusted temperature separate.
mod extract;
mod observe;
mod score;
mod support;
mod temporal;
mod nightly;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, io::Read, path::Path};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub const METHOD: &str = "scoreboard-v1";
pub const ABI: &str = "rw_scoreboard --request JSON schema=woof.scoreboard.request.v1 method=scoreboard-v1 actions=extract,observe,score,plan,merge,interpolate,accumulate,interpolate_to_reports columns=WFSBCOL1 additional_actions=nightly nightly_publication=woof.scoreboard.nightly.v2";
pub const SOURCE_REV: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
/// Embed the build revision for release artifact verification, as every bundled entry point does.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str = SOURCE_REV;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Station {
    #[serde(alias = "id", alias = "icaoId")]
    pub station_id: String,
    #[serde(alias = "latitude", alias = "latitude_deg")]
    pub lat: f64,
    #[serde(alias = "longitude", alias = "longitude_deg")]
    pub lon: f64,
    pub elevation_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GridSpec {
    pub south: f64,
    pub west: f64,
    pub dlat: f64,
    pub dlon: f64,
    pub nx: usize,
    pub ny: usize,
    #[serde(default = "default_dx")]
    pub dx_m: f64,
}
fn default_dx() -> f64 {
    1000.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub station_id: String,
    pub valid_time: String,
    #[serde(default)]
    pub observation_time: Option<String>,
    #[serde(default)]
    pub observation_times: BTreeMap<String, String>,
    #[serde(default)]
    pub observation_sources: BTreeMap<String, String>,
    pub values: BTreeMap<String, f64>,
    #[serde(default)]
    pub terrain_m: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid {
    pub spec: GridSpec,
    pub variable: String,
    pub window_minutes: u32,
    pub values: Vec<Option<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub schema: String,
    pub method_id: String,
    pub valid_time: String,
    pub init_time: Option<String>,
    pub domain: String,
    pub lead_minutes: u32,
    #[serde(default)]
    pub lead_seconds: u64,
    pub source_sha256: String,
    pub support_hash: String,
    pub stations: Vec<Station>,
    pub points: Vec<Point>,
    #[serde(default)]
    pub tombstones: Vec<ObservationRetraction>,
    #[serde(default)]
    pub grids: Vec<Grid>,
    #[serde(default)]
    pub drops: Vec<Value>,
    #[serde(default)]
    pub diagnostics: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationRetraction {
    pub station_id: String,
    pub variable: String,
    pub observation_time: String,
    pub source_id: String,
    pub reason: String,
}

pub fn fail(msg: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::other(msg.into()).into()
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn get_str<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| fail(format!("{key} is required")))
}
pub fn time(s: &str) -> Result<i64> {
    if let Ok(v) = chrono::DateTime::parse_from_rfc3339(s) {
        return Ok(v.timestamp());
    }
    for f in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d_%H:%M:%S"] {
        if let Ok(v) = chrono::NaiveDateTime::parse_from_str(s, f) {
            return Ok(v.and_utc().timestamp());
        }
    }
    Err(fail(format!("invalid UTC time {s}")))
}
pub fn iso(t: i64) -> String {
    chrono::DateTime::from_timestamp(t, 0)
        .unwrap()
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string()
}
pub fn read_artifact(p: &str) -> Result<Artifact> {
    let bytes = std::fs::read(p)?;
    if !bytes.starts_with(b"WFSBCOL1") {
        return Ok(serde_json::from_slice(&bytes)?);
    }
    if bytes.len() < 16 {
        return Err(fail("columnar artifact header is truncated"));
    }
    let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    if n > bytes.len() - 16 {
        return Err(fail("columnar metadata length exceeds artifact bytes"));
    }
    let bounded = |bytes: &[u8], limit: u64| -> Result<Vec<u8>> {
        let mut out = vec![];
        zstd::stream::read::Decoder::new(bytes)?
            .take(limit + 1)
            .read_to_end(&mut out)?;
        if out.len() as u64 > limit {
            return Err(fail(
                "columnar expansion exceeds bounded metadata/payload capacity",
            ));
        }
        Ok(out)
    };
    let meta: Value = serde_json::from_slice(&bounded(&bytes[16..16 + n], 64 * 1024 * 1024)?)?;
    let mut artifact: Artifact = serde_json::from_value(meta["artifact"].clone())?;
    let names: Vec<String> = serde_json::from_value(meta["columns"].clone())?;
    let raw = bounded(&bytes[16 + n..], 512 * 1024 * 1024)?;
    let mut cursor = 0usize;
    let mut next = || -> Result<f64> {
        let end = cursor
            .checked_add(8)
            .ok_or_else(|| fail("column offset overflow"))?;
        let b = raw
            .get(cursor..end)
            .ok_or_else(|| fail("columnar data is truncated"))?;
        cursor = end;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    };
    for name in &names {
        for p in &mut artifact.points {
            let v = next()?;
            if v.is_finite() {
                p.values.insert(name.clone(), v);
            }
        }
    }
    for grid in &mut artifact.grids {
        let count = grid
            .spec
            .nx
            .checked_mul(grid.spec.ny)
            .ok_or_else(|| fail("columnar grid size overflow"))?;
        for _ in 0..count {
            let v = next()?;
            grid.values.push(v.is_finite().then_some(v));
        }
    }
    if cursor != raw.len() {
        return Err(fail("columnar payload has unexplained trailing bytes"));
    }
    Ok(artifact)
}
pub fn support_hash(stations: &[Station], grid: &Option<GridSpec>) -> Result<String> {
    let mut s = stations.to_vec();
    s.sort_by(|a, b| a.station_id.cmp(&b.station_id));
    Ok(digest(&serde_json::to_vec(&(METHOD, s, grid))?))
}
pub fn new_artifact(
    request: &Value,
    bytes: &[u8],
    stations: Vec<Station>,
    grid: &Option<GridSpec>,
) -> Result<Artifact> {
    let valid = iso(time(get_str(request, "valid_time")?)?);
    Ok(Artifact {
        schema: "woof.scoreboard.artifact.v1".into(),
        method_id: METHOD.into(),
        valid_time: valid,
        init_time: request["init_time"]
            .as_str()
            .map(|s| time(s).map(iso))
            .transpose()?,
        domain: request["domain"].as_str().unwrap_or("d01").into(),
        lead_minutes: request["lead_minutes"].as_u64().unwrap_or(0) as u32,
        lead_seconds: request["lead_seconds"]
            .as_u64()
            .unwrap_or_else(|| request["lead_minutes"].as_u64().unwrap_or(0) * 60),
        source_sha256: digest(bytes),
        support_hash: support_hash(&stations, grid)?,
        stations,
        points: vec![],
        tombstones: vec![],
        grids: vec![],
        drops: vec![],
        diagnostics: json!({}),
    })
}
pub fn load_stations(v: &Value) -> Result<Vec<Station>> {
    let data = if let Some(path) = v["stations"].as_str() {
        let d: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if d.is_array() {
            d
        } else {
            d["stations"].clone()
        }
    } else {
        v["stations"].clone()
    };
    let mut stations: Vec<Station> = if data.is_null() {
        vec![]
    } else {
        serde_json::from_value(data)?
    };
    stations.sort_by(|a, b| a.station_id.cmp(&b.station_id));
    if stations
        .windows(2)
        .any(|s| s[0].station_id == s[1].station_id)
    {
        return Err(fail("duplicate station identity in frozen support"));
    }
    if stations.iter().any(|s| {
        !s.lat.is_finite() || s.lat.abs() > 90.0 || !s.lon.is_finite() || !s.elevation_m.is_finite()
    }) {
        return Err(fail("station coordinates/elevations must be finite"));
    }
    Ok(stations)
}
pub fn same_cycle(a: &Artifact, b: &Artifact) -> Result<bool> {
    Ok(a.init_time.as_deref().map(time).transpose()?
        == b.init_time.as_deref().map(time).transpose()?)
}
pub fn actual_lead_seconds(a: &Artifact) -> Result<u64> {
    if let Some(init) = a.init_time.as_deref() {
        let seconds = time(&a.valid_time)? - time(init)?;
        if seconds < 0 {
            return Err(fail("forecast valid time precedes its initialization"));
        }
        Ok(seconds as u64)
    } else if a.lead_seconds > 0 {
        Ok(a.lead_seconds)
    } else {
        Ok(a.lead_minutes as u64 * 60)
    }
}
pub fn write_artifact(artifact: &Artifact, path: &str) -> Result<Value> {
    let bytes = serde_json::to_vec(artifact)?;
    let semantic_sha256 = digest(&serde_json::to_vec(&(
        &artifact.method_id,
        &artifact.valid_time,
        &artifact.support_hash,
        &artifact.points,
        &artifact.tombstones,
        &artifact.grids,
    ))?);
    std::fs::write(path, &bytes)?;
    // A compact columnar object: metadata names the f64 columns in a zstd
    // frame. Station identity, time and missingness remain in the metadata.
    let mut names = std::collections::BTreeSet::new();
    for p in &artifact.points {
        names.extend(p.values.keys().cloned());
    }
    let names: Vec<_> = names.into_iter().collect();
    let mut raw = vec![];
    for name in &names {
        for p in &artifact.points {
            raw.extend_from_slice(
                &p.values
                    .get(name)
                    .copied()
                    .unwrap_or(f64::NAN)
                    .to_le_bytes(),
            );
        }
    }
    for grid in &artifact.grids {
        for value in &grid.values {
            raw.extend_from_slice(&value.unwrap_or(f64::NAN).to_le_bytes());
        }
    }
    let mut metadata = artifact.clone();
    for p in &mut metadata.points {
        p.values.clear();
    }
    for g in &mut metadata.grids {
        g.values.clear();
    }
    let meta = serde_json::to_vec(
        &json!({"schema":"woof.scoreboard.columns.v1","columns":names,"artifact":metadata}),
    )?;
    let meta = zstd::encode_all(meta.as_slice(), 3)?;
    let compressed = zstd::encode_all(raw.as_slice(), 3)?;
    let column_path = format!("{path}.columns.zst");
    let mut output = b"WFSBCOL1".to_vec();
    output.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    output.extend_from_slice(&meta);
    output.extend_from_slice(&compressed);
    std::fs::write(&column_path, &output)?;
    Ok(
        json!({"schema":"woof.scoreboard.receipt.v1","artifact_path":path,"content_sha256":digest(&bytes),"semantic_sha256":semantic_sha256,"bytes":bytes.len(),"columnar_path":column_path,"columnar_sha256":digest(&output),"columnar_bytes":output.len(),"support_hash":artifact.support_hash,"points":artifact.points.len(),"grids":artifact.grids.len(),"drops":artifact.drops.len(),"init_time":artifact.init_time,"lead_minutes":artifact.lead_minutes,"lead_seconds":actual_lead_seconds(artifact)?,"diagnostics":artifact.diagnostics}),
    )
}
fn main() {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    if let Err(e) = run() {
        eprintln!("rw_scoreboard: {e}");
        std::process::exit(1)
    }
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "--abi") {
        println!("{ABI}\n{SOURCE_REV}");
        return Ok(());
    }
    if args.len() != 3 || args[1] != "--request" {
        return Err(fail("usage: rw_scoreboard --request REQUEST.json"));
    }
    let req: Value = serde_json::from_slice(&std::fs::read(Path::new(&args[2]))?)?;
    if req["method_id"].as_str().is_some_and(|m| m != METHOD) {
        return Err(fail("method_id differs from frozen scoreboard-v1"));
    }
    let result = match get_str(&req, "action")? {
        "extract" => extract::run(&req)?,
        "observe" => observe::run(&req)?,
        "merge" => observe::merge(&req)?,
        "score" => score::run(&req)?,
        "nightly" => nightly::run(&req)?,
        "plan" => plan(&req)?,
        "interpolate" => temporal::interpolate(&req)?,
        "accumulate" => temporal::accumulate(&req)?,
        "interpolate_to_reports" => temporal::to_reports(&req)?,
        other => return Err(fail(format!("unknown action {other}"))),
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn plan(req: &Value) -> Result<Value> {
    let stations = if let Some(inputs) = req["station_inventories"].as_array() {
        let mut union = BTreeMap::<String, Station>::new();
        for input in inputs {
            let mut source = req.clone();
            source["stations"] = input.clone();
            for station in load_stations(&source)? {
                let alias = if station.station_id.len() == 3 {
                    Some(format!("K{}", station.station_id))
                } else if station.station_id.len() == 4 {
                    station.station_id.strip_prefix('K').map(str::to_string)
                } else {
                    None
                };
                if alias
                    .as_ref()
                    .and_then(|id| union.get(id))
                    .is_none_or(|s| !observe::station_alias(s, &station))
                {
                    union.entry(station.station_id.clone()).or_insert(station);
                }
            }
        }
        union.into_values().collect()
    } else {
        load_stations(req)?
    };
    let bbox = req["bbox"]
        .as_array()
        .ok_or_else(|| fail("plan bbox [west,south,east,north] required"))?;
    if bbox.len() != 4 {
        return Err(fail("plan bbox must have four bounds"));
    }
    let bounds: Vec<f64> = bbox
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| fail("bbox bound must be numeric")))
        .collect::<Result<_>>()?;
    let stations: Vec<_> = stations
        .into_iter()
        .filter(|s| {
            s.lat >= bounds[1]
                && s.lat <= bounds[3]
                && if bounds[0] <= bounds[2] {
                    s.lon >= bounds[0] && s.lon <= bounds[2]
                } else {
                    s.lon >= bounds[0] || s.lon <= bounds[2]
                }
        })
        .collect();
    let grid: Option<GridSpec> = serde_json::from_value(req["grid"].clone())?;
    let result = json!({"schema":"woof.scoreboard.plan.v1","method_id":METHOD,"support_hash":support_hash(&stations,&grid)?,"stations":stations,"grid":grid});
    std::fs::write(get_str(req, "output")?, serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}
