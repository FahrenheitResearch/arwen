//! Modified t-route output contracts at 12a8eae0cdfed437143c590659fa7077605a5e70
//! (Apache-2.0). Serialization and history transpose are Rust data-path work.
use crate::{
    Frame, NexusPlan, Result,
    netcdf::{self, Attribute as A, Dataset, Dimension, Variable},
};
use chrono::{DateTime, NaiveDateTime, TimeDelta, Timelike};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

/// Routing clocks are UTC. Preserve legacy native spelling and accept explicit
/// RFC3339 offsets without changing the physical instant or output precision.
pub(crate) fn parse_origin(value: &str) -> Result<NaiveDateTime> {
    if let Ok(origin) = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S") {
        return Ok(origin);
    }
    let origin = DateTime::parse_from_rfc3339(value)?.naive_utc();
    if origin.nanosecond() != 0 {
        return Err(
            "routing output origin has fractional seconds that native headers cannot represent"
                .into(),
        );
    }
    Ok(origin)
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub output_directory: PathBuf,
    pub origin: String,
    pub interval_seconds: u64,
    pub cadence_steps: u64,
    #[serde(default)]
    pub final_time_seconds: Option<f64>,
    pub frames: usize,
    pub capacity: usize,
    pub reaches: usize,
    pub dtype: String,
    pub ids_offset: usize,
    pub flow_offset: usize,
    pub velocity_offset: Option<usize>,
    pub depth_offset: Option<usize>,
    pub nudge_offset: Option<usize>,
    pub types: Vec<String>,
    pub formats: Vec<String>,
    #[serde(default)]
    pub nexus_ids: Vec<String>,
    #[serde(default)]
    pub candidate_ids: Vec<String>,
    #[serde(default)]
    pub provenance: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub parquet: Option<crate::parquet::Options>,
    #[serde(default)]
    pub renderer: Option<Renderer>,
}

impl Plan {
    pub fn time_seconds(&self, frame: usize) -> Result<f64> {
        let nominal = (frame as u64 + 1)
            .checked_mul(self.cadence_steps)
            .and_then(|n| n.checked_mul(self.interval_seconds))
            .ok_or("history clock overflow")? as f64;
        if frame + 1 == self.frames {
            if let Some(seconds) = self.final_time_seconds {
                let previous = frame as f64 * self.cadence_steps as f64 * self.interval_seconds as f64;
                if !seconds.is_finite() || seconds <= previous || seconds > nominal {
                    return Err("final history time must follow the preceding frame within its output period".into());
                }
                return Ok(seconds);
            }
        }
        Ok(nominal)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Renderer {
    pub domain: String,
    pub source_label: String,
    #[serde(default)]
    pub gauges: Vec<String>,
}

/// Apply an authoritative output-ID table without changing numerical history.
pub fn remap_ids(
    input: &Path,
    crosswalk: &std::collections::BTreeMap<i64, i64>,
    output: &Path,
) -> Result<serde_json::Value> {
    let bytes = fs::read(input)?;
    if bytes.len() % 8 != 0 {
        return Err("partial int64 output-ID word".into());
    }
    let ids = bytes
        .par_chunks_exact(8)
        .map(|word| i64::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    let source = ids
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    if source.len() != ids.len() || crosswalk.keys().any(|id| !source.contains(id)) {
        return Err("output-ID crosswalk source IDs missing/duplicated".into());
    }
    let mapped = ids
        .par_iter()
        .map(|id| crosswalk.get(id).copied().unwrap_or(*id))
        .collect::<Vec<_>>();
    if mapped
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != mapped.len()
    {
        return Err("output-ID crosswalk creates duplicate IDs".into());
    }
    let mut bytes = Vec::with_capacity(bytes.len());
    for id in mapped {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(
        serde_json::json!({"operation":"remap-ids","features":ids.len(),"explicit_crosswalk_rows":crosswalk.len(),
        "scope":"authoritative metadata table only; original ordering and numerical history unchanged"}),
    )
}

/// Boundary transport for named resident-history dumps. All raw array reading,
/// packing and numerical ID decoding stays in Rust, never in orchestration.
pub fn pack_history(
    mut plan: Plan,
    files: &std::collections::BTreeMap<String, PathBuf>,
    directory: &Path,
) -> Result<serde_json::Value> {
    let width = match plan.dtype.as_str() {
        "f32le" => 4,
        "f64le" => 8,
        _ => return Err("unsupported history precision".into()),
    };
    if plan.reaches == 0 || plan.frames == 0 || plan.frames > plan.capacity {
        return Err("invalid packed history extent".into());
    }
    if files
        .keys()
        .any(|key| !["reach_id", "flow", "velocity", "depth", "nudge"].contains(&key.as_str()))
    {
        return Err("unknown named history field".into());
    }
    let count = plan
        .capacity
        .checked_mul(plan.reaches)
        .and_then(|v| v.checked_mul(width))
        .ok_or("history size overflow")?;
    let mut bytes = Vec::new();
    let mut offsets = std::collections::BTreeMap::new();
    for name in ["reach_id", "flow", "velocity", "depth", "nudge"] {
        if let Some(path) = files.get(name) {
            let raw = fs::read(path)?;
            let size = if name == "reach_id" {
                plan.reaches.checked_mul(8).ok_or("ID size overflow")?
            } else {
                count
            };
            if raw.len() != size {
                return Err(
                    format!("history {name} bytes differ from declared native extent").into(),
                );
            }
            offsets.insert(name, bytes.len());
            bytes.extend_from_slice(&raw);
        }
    }
    plan.ids_offset = *offsets
        .get("reach_id")
        .ok_or("explicit reach IDs missing")?;
    plan.flow_offset = *offsets.get("flow").ok_or("flow history missing")?;
    plan.velocity_offset = offsets.get("velocity").copied();
    plan.depth_offset = offsets.get("depth").copied();
    plan.nudge_offset = offsets.get("nudge").copied();
    plan.candidate_ids = values(&bytes, plan.ids_offset, plan.reaches, 8)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b).to_string())
        .collect();
    fs::create_dir(directory)?;
    plan.output_directory = directory.join("products");
    fs::write(directory.join("snapshot.raw"), &bytes)?;
    fs::write(
        directory.join("request.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"operation":"series-write","plan":plan}))?,
    )?;
    Ok(
        serde_json::json!({"operation":"pack-history","bytes":bytes.len(),"fields":offsets,
        "scope":"opaque actual resident history words; decimal candidate IDs from explicit int64 roster"}),
    )
}

/// Offline qualification of actual native/GPU histories. Align only by exact
/// explicit IDs: source and output network orders need not be identical.
pub fn compare(
    expected: &Plan,
    reference: &[u8],
    candidate: &Plan,
    actual: &[u8],
    fields: &[&str],
) -> Result<serde_json::Value> {
    let started = Instant::now();
    let origin = parse_origin(&expected.origin)?;
    let candidate_origin = parse_origin(&candidate.origin)?;
    if expected.dtype != candidate.dtype
        || !matches!(expected.dtype.as_str(), "f32le" | "f64le")
        || origin != candidate_origin
        || expected.frames != candidate.frames
        || expected.interval_seconds != candidate.interval_seconds
        || expected.cadence_steps != candidate.cadence_steps
        || expected.frames == 0
        || expected.frames > expected.capacity
        || candidate.frames > candidate.capacity
    {
        return Err("history native precision, clock, cadence or extent differs".into());
    }
    let width = if expected.dtype == "f32le" { 4 } else { 8 };
    let roster = |plan: &Plan, bytes: &[u8]| -> Result<Vec<i64>> {
        let ids = values(bytes, plan.ids_offset, plan.reaches, 8)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect::<Vec<_>>();
        if ids.is_empty() || ids.iter().collect::<std::collections::HashSet<_>>().len() != ids.len()
        {
            return Err("history has empty/duplicate explicit feature IDs".into());
        }
        Ok(ids)
    };
    let expected_ids = roster(expected, reference)?;
    let actual_ids = roster(candidate, actual)?;
    let indices = actual_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect::<std::collections::HashMap<_, _>>();
    if expected_ids.len() != actual_ids.len()
        || expected_ids.iter().any(|id| !indices.contains_key(id))
    {
        let missing = expected_ids
            .iter()
            .filter(|id| !indices.contains_key(id))
            .take(16)
            .collect::<Vec<_>>();
        return Err(format!("history feature rosters differ: native {}, candidate {}; first missing native IDs {missing:?}",expected_ids.len(),actual_ids.len()).into());
    }
    if fields.is_empty()
        || fields
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != fields.len()
    {
        return Err("history comparison needs explicit unique fields".into());
    }
    let offset = |plan: &Plan, name: &str| -> Result<usize> {
        match name {
            "flow" => Some(plan.flow_offset),
            "velocity" => plan.velocity_offset,
            "depth" => plan.depth_offset,
            "nudge" => plan.nudge_offset,
            _ => return Err(format!("unknown history field {name}").into()),
        }
        .ok_or_else(|| format!("history field {name} not provided").into())
    };
    let verdicts = fields.par_iter().map(|name| -> Result<_> {
        let l=values(reference,offset(expected,name)?,expected.capacity.checked_mul(expected.reaches).ok_or("history overflow")?,width)?;
        let r=values(actual,offset(candidate,name)?,candidate.capacity.checked_mul(candidate.reaches).ok_or("history overflow")?,width)?;
        let mut different=0u64;
        let mut witnesses=Vec::new();
        for t in 0..expected.frames {
            for (i,id) in expected_ids.iter().enumerate() {
                let li=(t*expected.reaches+i)*width;
                let ri=(t*candidate.reaches+indices[id])*width;
                if l[li..li+width] != r[ri..ri+width] {
                    different+=1;
                    if witnesses.len()<8 {
                        witnesses.push(serde_json::json!({"frame":t,"feature_id":id,
                        "native_le_bytes":l[li..li+width],"candidate_le_bytes":r[ri..ri+width],
                        "native_value":value(l,t*expected.reaches+i,width).to_string(),
                        "candidate_value":value(r,t*candidate.reaches+indices[id],width).to_string()}));
                    }
                }
            }
        }
        Ok(serde_json::json!({"field":name,"words":expected.frames*expected.reaches,
            "different_words":different,"witnesses":witnesses}))
    }).collect::<Result<Vec<_>>>()?;
    let different: u64 = verdicts
        .iter()
        .map(|v| v["different_words"].as_u64().unwrap())
        .sum();
    let unselected = ["flow", "velocity", "depth", "nudge"]
        .into_iter()
        .filter(|name| !fields.contains(name) && offset(expected, name).is_ok())
        .collect::<Vec<_>>();
    Ok(
        serde_json::json!({"operation":"compare-series","matches":different==0,
        "dtype":expected.dtype,"frames":expected.frames,"features":expected.reaches,
        "origin_utc":origin.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "words":fields.len()*expected.frames*expected.reaches,"different_words":different,
        "alignment":"exact explicit int64 IDs, full roster; no inferred crosswalk",
        "fields":verdicts,"uncompared_native_fields":unselected,
        "parallelism":format!("{} Rayon workers; independent field comparisons",rayon::current_num_threads()),
        "comparison_ms":started.elapsed().as_secs_f64()*1000.}),
    )
}
fn values(bytes: &[u8], offset: usize, count: usize, width: usize) -> Result<&[u8]> {
    let end = count
        .checked_mul(width)
        .and_then(|n| offset.checked_add(n))
        .ok_or("series size overflow")?;
    bytes
        .get(offset..end)
        .ok_or_else(|| "incomplete series snapshot".into())
}
fn value(bytes: &[u8], index: usize, width: usize) -> f64 {
    if width == 8 {
        f64::from_le_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap())
    } else {
        f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap()) as f64
    }
}
fn atomic_text(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<fs::File>) -> Result<()>,
) -> Result<()> {
    let partial = path.with_extension("csv.partial");
    let mut file = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)?,
    );
    let result = (|| {
        write(&mut file)?;
        file.flush()?;
        file.get_ref().sync_all()?;
        fs::hard_link(&partial, path)?;
        Ok(())
    })();
    let _ = fs::remove_file(partial);
    result
}
// Pandas' round-trip float32 formatter needs a native witness, so this door
// preserves all native-real digits in the independent analysis CSV. It is never
// labelled a byte-identical t-route CSV without the referee comparison.
fn number(v: f64) -> String {
    if v.is_nan() {
        String::new()
    } else if v.is_finite() && v.fract() == 0. {
        format!("{v:.1}")
    } else {
        v.to_string()
    }
}

fn native_csv_number(v: f64, width: usize) -> String {
    if v.is_nan() {
        return String::new();
    }
    if !v.is_finite() {
        return v.to_string();
    }
    // Match the native-real shortest round-trip representation used by Pandas,
    // rather than printing a float32 after widening it to double.
    let scientific = if width == 4 {
        format!("{:e}", v as f32)
    } else {
        format!("{v:e}")
    };
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if !(-4..16).contains(&exponent) {
        return format!("{mantissa}e{exponent:+03}");
    }
    let mut out = if width == 4 {
        (v as f32).to_string()
    } else {
        v.to_string()
    };
    if !out.contains('.') {
        out.push_str(".0");
    }
    out
}

/// Capture the actual native output-writer inputs from its uncompressed typed
/// NetCDF variables. This is a referee tool, not a product routing adapter.
pub fn capture_native(path: &Path, directory: &Path) -> Result<serde_json::Value> {
    let (schema, bytes) = netcdf::read(path)?;
    let reaches = schema
        .dimensions
        .iter()
        .find(|d| d.name == "feature_id")
        .ok_or("native feature_id dimension missing")?
        .len;
    let frames = schema
        .dimensions
        .iter()
        .find(|d| d.name == "time")
        .ok_or("native time dimension missing")?
        .len;
    let variable = |name: &str, dtype: &str, dims: &[&str]| -> Result<&Variable> {
        let var = schema
            .variables
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| format!("native variable {name} missing"))?;
        if var.dtype != dtype || var.dimensions != dims {
            return Err(format!("native variable {name} precision/order differs").into());
        }
        Ok(var)
    };
    let ids = variable("feature_id", "i64le", &["feature_id"])?;
    let time = variable("time", "f64le", &["time"])?;
    let origin = time
        .attributes
        .iter()
        .find(|a| a.name == "units" && a.dtype == "char")
        .and_then(|a| a.text.strip_prefix("seconds since "))
        .ok_or("native clock origin missing")?;
    let times = values(&bytes, time.offset, frames, 8)?;
    let interval = value(times, 0, 8);
    if !interval.is_finite()
        || interval < 1.
        || interval.fract() != 0.
        || interval > u32::MAX as f64
        || (0..frames).any(|t| value(times, t, 8) != (t + 1) as f64 * interval)
    {
        return Err("native clock needs consecutive whole-second intervals".into());
    }
    let feature_type = schema
        .variables
        .iter()
        .find(|v| v.name == "type")
        .ok_or("native type variable missing")?;
    if feature_type.dimensions != ["feature_id"] {
        return Err("native type order differs".into());
    }
    let types = match feature_type.dtype.as_str() {
        "string" => feature_type.strings.clone(),
        "char" => values(&bytes, feature_type.offset, reaches, 1)?
            .iter()
            .map(|b| std::str::from_utf8(std::slice::from_ref(b)).map(str::to_owned))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        _ => return Err("native type encoding differs".into()),
    };
    let mut raw = values(&bytes, ids.offset, reaches, 8)?.to_vec();
    let mut offsets = Vec::new();
    for name in ["flow", "velocity", "depth", "nudge"] {
        let var = variable(name, "f32le", &["feature_id", "time"])?;
        let source = values(&bytes, var.offset, reaches * frames, 4)?;
        offsets.push(raw.len());
        let mut field = vec![0; reaches * frames * 4];
        field
            .par_chunks_mut(reaches * 4)
            .enumerate()
            .for_each(|(t, dest)| {
                for r in 0..reaches {
                    dest[r * 4..r * 4 + 4]
                        .copy_from_slice(&source[(r * frames + t) * 4..(r * frames + t) * 4 + 4]);
                }
            });
        raw.extend_from_slice(&field);
    }
    let plan = Plan {
        final_time_seconds: None,
        output_directory: directory.join("rewritten"),
        origin: origin.into(),
        interval_seconds: interval as u64,
        cadence_steps: 1,
        frames,
        capacity: frames,
        reaches,
        dtype: "f32le".into(),
        ids_offset: 0,
        flow_offset: offsets[0],
        velocity_offset: Some(offsets[1]),
        depth_offset: Some(offsets[2]),
        nudge_offset: Some(offsets[3]),
        types,
        formats: vec!["troute_netcdf".into()],
        nexus_ids: vec![],
        candidate_ids: values(&bytes, ids.offset, reaches, 8)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|id| i64::from_le_bytes(*id).to_string())
            .collect(),
        provenance: serde_json::Map::new(),
        parquet: None,
        renderer: None,
    };
    fs::create_dir(directory)?;
    fs::write(directory.join("snapshot.raw"), raw)?;
    fs::write(
        directory.join("request.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"operation":"series-write","plan":plan}))?,
    )?;
    Ok(
        serde_json::json!({"operation":"capture-troute-netcdf","frames":frames,"reaches":reaches,
        "scope":"actual native typed Q/V/D/nudge writer inputs; bit-preserving Rust transpose"}),
    )
}
pub fn write(plan: &Plan, payload: &[u8]) -> Result<serde_json::Value> {
    let started = Instant::now();
    // A bad later entry used to fail only after preceding formats had already
    // published files. Validate the whole roster before creating any output.
    if plan.formats.is_empty() {
        return Err("history needs at least one output format".into());
    }
    let mut formats = std::collections::HashSet::new();
    for format in &plan.formats {
        if !matches!(
            format.as_str(),
            "troute_parquet"
                | "troute_csv"
                | "candidate_json"
                | "nexus_csv"
                | "troute_netcdf"
                | "analysis_csv"
                | "renderer_request"
        ) {
            return Err(format!("format {format:?} has no verified writer yet").into());
        }
        if !formats.insert(format) {
            return Err(format!("duplicate output format {format:?}").into());
        }
    }
    let width = match plan.dtype.as_str() {
        "f64le" => 8,
        "f32le" => 4,
        _ => return Err("unsupported history precision".into()),
    };
    if plan.frames == 0
        || plan.frames > plan.capacity
        || plan.reaches == 0
        || plan.cadence_steps == 0
    {
        return Err("invalid history extent/cadence".into());
    }
    if plan.types.len() != plan.reaches {
        return Err("explicit feature type roster required".into());
    }
    let ids = values(payload, plan.ids_offset, plan.reaches, 8)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect::<Vec<_>>();
    if ids.iter().collect::<std::collections::HashSet<_>>().len() != ids.len() {
        return Err("duplicate output feature IDs".into());
    }
    let node_ids = ids.iter().map(i64::to_string).collect::<Vec<_>>();
    if formats.contains(&"renderer_request".to_string()) {
        if let Some(renderer) = &plan.renderer {
            if renderer.domain.trim().is_empty()
                || renderer.source_label.trim().is_empty()
                || renderer.gauges.iter().any(|id| !node_ids.contains(id))
                || renderer
                    .gauges
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    != renderer.gauges.len()
            {
                return Err("renderer needs nonempty labels and unique gauges in the explicit publication roster".into());
            }
        }
    }
    let count = plan
        .capacity
        .checked_mul(plan.reaches)
        .ok_or("history overflow")?;
    let flow = values(payload, plan.flow_offset, count, width)?;
    let field = |offset: Option<usize>| -> Result<Option<&[u8]>> {
        offset.map(|o| values(payload, o, count, width)).transpose()
    };
    let velocity = field(plan.velocity_offset)?;
    let depth = field(plan.depth_offset)?;
    let nudge = field(plan.nudge_offset)?;
    let origin = parse_origin(&plan.origin)?;
    let frames = (0..plan.frames)
        .map(|frame| -> Result<Frame> {
            let seconds = plan.time_seconds(frame)?;
            let date = origin
                .checked_add_signed(TimeDelta::microseconds((seconds * 1e6).round() as i64))
                .ok_or("timestamp overflow")?;
            Ok(Frame {
                index: frame.try_into()?,
                timestamp: date.format("%Y-%m-%d %H:%M:%S%.f").to_string(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    fs::create_dir_all(&plan.output_directory)?;
    let mut receipts = Vec::new();
    for format in &plan.formats {
        match format.as_str() {
            "renderer_request" => {
                let directory = plan.output_directory.join("renderer-history");
                fs::create_dir(&directory)?;
                let dtype = if width == 8 { "<f8" } else { "<f4" };
                // Bit-preserving frame slices, never a final-state broadcast or
                // a float32 conversion. Relative paths keep the output movable.
                let renderer_frames = frames
                    .par_iter()
                    .enumerate()
                    .map(|(t, frame)| -> std::result::Result<serde_json::Value, String> {
                        let filename = format!("flow-{t:06}.{}", plan.dtype);
                        let begin = t * plan.reaches * width;
                        atomic_text(&directory.join(&filename), |file| {
                            file.write_all(&flow[begin..begin + plan.reaches * width])?;
                            Ok(())
                        }).map_err(|e| e.to_string())?;
                        Ok(serde_json::json!({
                            "valid_time":format!("{}Z",frame.timestamp.replace(' ',"T")),
                            "discharge_m3_s":{"path":format!("renderer-history/{filename}"),"dtype":dtype}
                        }))
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                crate::sync_directory(&directory)?;
                let domain = plan
                    .renderer
                    .as_ref()
                    .map_or("hydro", |r| r.domain.as_str());
                let label = plan.renderer.as_ref().map_or(
                    "NextGen routing history; m3/s; UTC interval ends; no terrain bound",
                    |r| r.source_label.as_str(),
                );
                let gauges = plan
                    .renderer
                    .as_ref()
                    .map_or(&[][..], |r| r.gauges.as_slice());
                let path = plan.output_directory.join("renderer-request.json");
                atomic_text(&path, |file| {
                    serde_json::to_writer_pretty(
                        &mut *file,
                        &serde_json::json!({
                            "schema":"gpuwm.nextgen-products.v1", "out_dir":"rendered-products",
                            "domain":domain,"source_label":label,"node_ids":node_ids,
                            "gauges":gauges,"frames":renderer_frames
                        }),
                    )?;
                    writeln!(file)?;
                    Ok(())
                })?;
                receipts.push(serde_json::json!({"operation":"renderer-request","path":path,
                    "schema":"gpuwm.nextgen-products.v1","frames":plan.frames,
                    "reaches":plan.reaches,"dtype":dtype,"units":"m3 s-1",
                    "scope":"exact captured Q frames in explicit int64 publication order; no reference, observations or mixed channel/reservoir depth inferred"}));
            }
            "troute_parquet" => receipts.push(crate::parquet::write(plan, payload, &ids, width)?),
            "troute_csv" => {
                let (velocity, depth) = (
                    velocity.ok_or("native CSV needs velocity")?,
                    depth.ok_or("native CSV needs depth")?,
                );
                let rows = ids
                    .par_iter()
                    .enumerate()
                    .map(|(r, id)| {
                        let mut row = id.to_string();
                        for t in 0..plan.frames {
                            for field in [flow, velocity, depth] {
                                row.push(',');
                                row.push_str(&native_csv_number(
                                    value(field, t * plan.reaches + r, width),
                                    width,
                                ));
                            }
                        }
                        row.push('\n');
                        row
                    })
                    .collect::<Vec<_>>();
                let path = plan.output_directory.join("flowveldepth.csv");
                atomic_text(&path, |file| {
                    for t in 0..plan.frames {
                        let step = (t as u64 + 1)
                            .checked_mul(plan.cadence_steps)
                            .and_then(|n| n.checked_sub(1))
                            .ok_or("CSV step overflow")?;
                        for key in ["q", "v", "d"] {
                            write!(file, ",\"({step}, '{key}')\"")?;
                        }
                    }
                    writeln!(file)?;
                    for row in &rows {
                        file.write_all(row.as_bytes())?;
                    }
                    Ok(())
                })?;
                receipts.push(serde_json::json!({"operation":"troute-csv","path":path,
                    "scope":"native tuple-column header, native-real shortest decimals; qualify against native bytes"}));
            }
            "candidate_json" => {
                if plan.candidate_ids.len() != plan.reaches
                    || plan.candidate_ids.iter().any(|s| s.is_empty())
                    || plan
                        .candidate_ids
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        != plan.reaches
                    || plan.provenance.is_empty()
                {
                    return Err(
                        "candidate JSON needs explicit unique candidate IDs and provenance".into(),
                    );
                }
                // Format independent time slices in parallel, retaining capture
                // and feature order. Non-finite output is explicitly missing.
                let samples = frames
                    .par_iter()
                    .enumerate()
                    .map(|(t, frame)| {
                        plan.candidate_ids
                            .iter()
                            .enumerate()
                            .map(|(r, id)| {
                                let v = value(flow, t * plan.reaches + r, width);
                                serde_json::json!({"feature_id":id,
                            "time_utc":format!("{}Z",frame.timestamp.replace(' ',"T")),
                            "streamflow_m3_s":if v.is_finite() {Some(v)} else {None}})
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                let path = plan.output_directory.join("candidate.json");
                atomic_text(&path, |file| {
                    serde_json::to_writer(
                        &mut *file,
                        &serde_json::json!({
                        "schema":"arwen.point-flow.v1", "units":"m3 s-1",
                        "provenance":plan.provenance, "samples":samples}),
                    )?;
                    writeln!(file)?;
                    Ok(())
                })?;
                receipts.push(serde_json::json!({"operation":"candidate-json",
                    "path":path,"samples":plan.frames*plan.reaches,
                    "schema":"arwen.point-flow.v1"}));
            }
            "nexus_csv" => {
                if plan.nexus_ids.len() != plan.reaches {
                    return Err("nexus output needs explicit IDs in the capture order".into());
                }
                let mut bytes = vec![0u8; plan.reaches * plan.frames * width];
                // ngen labels the contribution with the formulation step's
                // index/start time; t-route above labels the interval end.
                let nexus_frames = (0..plan.frames)
                    .map(|t| -> Result<Frame> {
                        let step = (t as u64 + 1)
                            .checked_mul(plan.cadence_steps)
                            .and_then(|n| n.checked_sub(1))
                            .ok_or("nexus step overflow")?;
                        let seconds = step
                            .checked_mul(plan.interval_seconds)
                            .ok_or("nexus time overflow")?;
                        let date = origin
                            .checked_add_signed(TimeDelta::seconds(seconds.try_into()?))
                            .ok_or("nexus timestamp overflow")?;
                        Ok(Frame {
                            index: step.try_into()?,
                            timestamp: date.format("%Y-%m-%d %H:%M:%S").to_string(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                bytes
                    .par_chunks_mut(plan.frames * width)
                    .enumerate()
                    .for_each(|(reach, dest)| {
                        for t in 0..plan.frames {
                            let src = (t * plan.reaches + reach) * width;
                            dest[t * width..(t + 1) * width]
                                .copy_from_slice(&flow[src..src + width]);
                        }
                    });
                receipts.push(crate::write_nexus(
                    &NexusPlan {
                        version: 1,
                        ids: plan.nexus_ids.clone(),
                        frames: nexus_frames,
                        dtype: plan.dtype.clone(),
                        data: PathBuf::new(),
                        output_directory: plan.output_directory.join("nexus"),
                    },
                    &bytes,
                )?);
            }
            "troute_netcdf" => {
                let fields = [
                    ("flow", "Flow", "m3 s-1", Some(flow)),
                    ("velocity", "Velocity", "m/s", velocity),
                    ("depth", "Depth", "m", depth),
                    ("nudge", "Streamflow Nudge Value", "m3 s-1", nudge),
                ];
                if fields.iter().any(|f| f.3.is_none()) {
                    return Err(
                        "native t-route NetCDF requires flow, velocity, depth and nudge arrays"
                            .into(),
                    );
                }
                let maxlen = plan.types.iter().map(|s| s.len()).max().unwrap();
                if maxlen == 0 {
                    return Err("empty feature type".into());
                }
                let mut bytes = Vec::new();
                let mut schema = Dataset {
                    format: "netcdf4".into(),
                    dimensions: vec![
                        Dimension {
                            name: "feature_id".into(),
                            len: plan.reaches,
                            unlimited: false,
                        },
                        Dimension {
                            name: "time".into(),
                            len: plan.frames,
                            unlimited: false,
                        },
                        Dimension {
                            name: "type_strlen".into(),
                            len: maxlen,
                            unlimited: false,
                        },
                    ],
                    variables: vec![],
                    attributes: vec![
                        A::text("TITLE", "OUTPUT FROM T-ROUTE"),
                        A::text(
                            "file_reference_time",
                            &origin.format("%Y-%m-%d_%H:%M:%S").to_string(),
                        ),
                        // netCDF4-Python stores setncatts("") as one NUL,
                        // witnessed by the real LowerColorado native output.
                        A::text("code_version", "\0"),
                    ],
                };
                let time = Variable {
                    name: "time".into(),
                    dtype: "f64le".into(),
                    dimensions: vec!["time".into()],
                    attributes: vec![
                        A::f64("_FillValue", -9999.),
                        A::text("long_name", "valid output time"),
                        A::text("standard_name", "time"),
                        A::text(
                            "units",
                            &format!("seconds since {}", origin.format("%Y-%m-%d %H:%M:%S")),
                        ),
                        A::f64("missing_value", -9999.),
                    ],
                    offset: bytes.len(),
                    strings: vec![],
                };
                for t in 0..plan.frames {
                    bytes.extend_from_slice(&plan.time_seconds(t)?.to_le_bytes());
                }
                schema.variables.push(time);
                schema.variables.push(Variable {
                    name: "feature_id".into(),
                    dtype: "i64le".into(),
                    dimensions: vec!["feature_id".into()],
                    attributes: vec![A::text("long_name", "Segment ID")],
                    offset: bytes.len(),
                    strings: vec![],
                });
                for id in &ids {
                    bytes.extend_from_slice(&id.to_le_bytes());
                }
                let string_type = if maxlen == 1 { "char" } else { "string" };
                schema.variables.push(Variable {
                    name: "type".into(),
                    dtype: string_type.into(),
                    dimensions: vec!["feature_id".into()],
                    attributes: vec![A::text("long_name", "Type")],
                    offset: bytes.len(),
                    strings: if maxlen == 1 {
                        vec![]
                    } else {
                        plan.types.clone()
                    },
                });
                if maxlen == 1 {
                    for s in &plan.types {
                        bytes.push(s.as_bytes()[0]);
                    }
                }
                for (key, long, units, source) in fields {
                    schema.variables.push(Variable {
                        name: key.into(),
                        dtype: "f32le".into(),
                        dimensions: vec!["feature_id".into(), "time".into()],
                        attributes: vec![
                            A::f32("_FillValue", -9999.),
                            A::text("long_name", long),
                            A::text("units", units),
                            A::f64("missing_value", -9999.),
                        ],
                        offset: bytes.len(),
                        strings: vec![],
                    });
                    let mut data = vec![0u8; plan.reaches * plan.frames * 4];
                    let source = source.unwrap();
                    data.par_chunks_mut(plan.frames * 4)
                        .enumerate()
                        .for_each(|(reach, dest)| {
                            for t in 0..plan.frames {
                                dest[t * 4..t * 4 + 4].copy_from_slice(
                                    &(value(source, t * plan.reaches + reach, width) as f32)
                                        .to_le_bytes(),
                                );
                            }
                        });
                    bytes.extend_from_slice(&data);
                }
                let transpose_ms = started.elapsed().as_secs_f64() * 1000.;
                let mut r = netcdf::write(
                    &plan.output_directory.join("flowveldepth.nc"),
                    &schema,
                    &bytes,
                )?;
                r["history_transpose_elapsed_ms"] = transpose_ms.into();
                receipts.push(r);
            }
            "analysis_csv" => {
                let path = plan.output_directory.join("streamflow.csv");
                atomic_text(&path, |file| {
                    writeln!(file, "feature_id,time_utc,streamflow_m3_s")?;
                    for (t, frame) in frames.iter().enumerate() {
                        for (r, id) in ids.iter().enumerate() {
                            writeln!(
                                file,
                                "{id},{}Z,{}",
                                frame.timestamp.replace(' ', "T"),
                                number(value(flow, t * plan.reaches + r, width))
                            )?;
                        }
                    }
                    Ok(())
                })?;
                receipts.push(serde_json::json!({"operation":"analysis-csv","path":path,"schema":"feature_id,time_utc,streamflow_m3_s; UTC RFC3339; native precision"}));
            }
            _ => unreachable!("validated output format roster"),
        }
    }
    crate::sync_directory(&plan.output_directory)?;
    Ok(
        serde_json::json!({"operation":"series-write","frames":plan.frames,"reaches":plan.reaches,"boundary_ms":started.elapsed().as_secs_f64()*1000.,"parallelism":format!("{} Rayon workers transpose/format; native Rust NetCDF containers; ordered file I/O",rayon::current_num_threads()),"receipts":receipts,"scope":"serialization only; native replay qualification separate"}),
    )
}
