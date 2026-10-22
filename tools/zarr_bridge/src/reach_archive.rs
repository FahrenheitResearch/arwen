// SPDX-License-Identifier: Apache-2.0
//! Original reach/time archive analysis. No hydrological model source is used.
use super::{Result, SourceArray, fail, moment, numbers, open};
use chrono::{DateTime, SecondsFormat, TimeZone, Utc};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
use zarrs::storage::{
    MaybeBytes, MaybeBytesIterator, ReadableStorage, ReadableStorageTraits, StorageError, StoreKey,
    byte_range::ByteRangeIterator,
};

#[derive(Clone, Deserialize)]
struct Archive {
    id: String,
    store: String,
    feature: String,
    time: String,
    field: String,
    units: String,
    cadence_seconds: i64,
    #[serde(default)]
    provenance: BTreeMap<String, String>,
}
#[derive(Deserialize)]
struct Link {
    archive_feature_id: i64,
    candidate_feature_id: String,
    #[serde(default)]
    station_id: Option<String>,
}
#[derive(Deserialize)]
struct Job {
    source: Archive,
    cache: Option<PathBuf>,
    start: Option<String>,
    end: Option<String>,
    #[serde(default)]
    reaches: Vec<Link>,
    candidate: Option<PathBuf>,
    workers: Option<usize>,
}
#[derive(Deserialize)]
struct Point {
    feature_id: String,
    time_utc: String,
    streamflow_m3_s: Option<f64>,
}
#[derive(Deserialize)]
struct Candidate {
    schema: String,
    units: String,
    #[serde(default)]
    kind: CandidateKind,
    label: Option<String>,
    #[serde(default)]
    provenance: BTreeMap<String, String>,
    samples: Vec<Point>,
}
#[derive(Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CandidateKind {
    #[default]
    Model,
    Observations,
}
fn message(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn parallel<K: Sync, T: Send>(
    items: &[K],
    workers: usize,
    f: impl Fn(&K) -> std::result::Result<T, String> + Sync,
) -> std::result::Result<Vec<T>, String> {
    let next = AtomicUsize::new(0);
    let slots = Mutex::new((0..items.len()).map(|_| None).collect::<Vec<Option<T>>>());
    let errors = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..workers.min(items.len()) {
            threads.push(scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    match f(&items[i]) {
                        Ok(v) => slots.lock().unwrap()[i] = Some(v),
                        Err(e) => errors.lock().unwrap().push(e),
                    }
                }
            }));
        }
        for t in threads {
            if t.join().is_err() {
                errors
                    .lock()
                    .unwrap()
                    .push("archive worker panicked".into());
            }
        }
    });
    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e);
    }
    Ok(slots
        .into_inner()
        .unwrap()
        .into_iter()
        .map(Option::unwrap)
        .collect())
}
struct CaptureStore {
    upstream: ReadableStorage,
    root: PathBuf,
    sequence: AtomicUsize,
}
impl ReadableStorageTraits for CaptureStore {
    fn get(&self, key: &StoreKey) -> std::result::Result<MaybeBytes, StorageError> {
        let path = self.root.join(key.as_str());
        match fs::read(&path) {
            Ok(v) => return Ok(Some(v.into())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            Err(e) => return Err(StorageError::Other(e.to_string())),
        }
        let Some(bytes) = self.upstream.get(key)? else {
            return Ok(None);
        };
        let write = || -> std::io::Result<()> {
            fs::create_dir_all(path.parent().unwrap())?;
            let tmp = path.with_extension(format!(
                "capture-{}-{}",
                std::process::id(),
                self.sequence.fetch_add(1, Ordering::Relaxed)
            ));
            fs::write(&tmp, &bytes)?;
            fs::rename(tmp, path)?;
            Ok(())
        };
        write().map_err(|e| StorageError::Other(e.to_string()))?;
        Ok(Some(bytes))
    }
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> std::result::Result<MaybeBytesIterator<'a>, StorageError> {
        let Some(bytes) = self.get(key)? else {
            return Ok(None);
        };
        let size = bytes.len() as u64;
        let mut parts = Vec::new();
        for r in ranges {
            let (a, b) = (r.start(size) as usize, r.end(size) as usize);
            if a > b || b > bytes.len() {
                return Err(StorageError::Other(
                    "archive byte range exceeds captured object".into(),
                ));
            }
            parts.push(Ok(bytes.slice(a..b)));
        }
        Ok(Some(Box::new(parts.into_iter())))
    }
    fn size_key(&self, key: &StoreKey) -> std::result::Result<Option<u64>, StorageError> {
        Ok(self.get(key)?.map(|b| b.len() as u64))
    }
    fn supports_get_partial(&self) -> bool {
        false
    }
}
fn store(job: &Job) -> Result<ReadableStorage> {
    let src = &job.source.store;
    let network = src.starts_with("https://") || src.starts_with("http://");
    let upstream: ReadableStorage = if network {
        Arc::new(super::http_store::HttpStore::new(src)?)
    } else {
        Arc::new(zarrs::filesystem::FilesystemStore::new(src)?)
    };
    if let Some(root) = &job.cache {
        if !network {
            return fail(
                "capture cache is for remote archives; use source.store for offline replay",
            );
        }
        fs::create_dir_all(root)?;
        let identity = root.join(".archive-origin");
        if identity.exists() {
            if fs::read_to_string(&identity)? != *src {
                return fail("archive cache belongs to a different source URL");
            }
        } else {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(identity)?;
            f.write_all(src.as_bytes())?;
        }
        Ok(Arc::new(CaptureStore {
            upstream,
            root: root.clone(),
            sequence: AtomicUsize::new(0),
        }))
    } else {
        Ok(upstream)
    }
}
fn timestamp(text: &str) -> Result<i64> {
    let t = DateTime::parse_from_rfc3339(text)?;
    if t.timestamp_subsec_nanos() != 0 {
        return fail(
            "reach samples require whole UTC seconds; fractional times cannot be silently rounded",
        );
    }
    Ok(t.timestamp())
}
fn iso(t: i64) -> String {
    Utc.timestamp_opt(t, 0)
        .single()
        .unwrap()
        .to_rfc3339_opts(SecondsFormat::Secs, true)
}
fn cf_time(array: &SourceArray, value: f64) -> Result<i64> {
    let attrs = array.attributes();
    let calendar = attrs
        .get("calendar")
        .and_then(Value::as_str)
        .unwrap_or("standard");
    if !["standard", "gregorian", "proleptic_gregorian"].contains(&calendar) {
        return fail(format!("unsupported archive calendar {calendar}"));
    }
    let units = attrs
        .get("units")
        .and_then(Value::as_str)
        .ok_or("archive time units missing")?;
    let (unit, origin) = units
        .split_once(" since ")
        .ok_or("archive time lacks CF since epoch")?;
    let factor = match unit {
        "seconds" | "second" => 1.,
        "minutes" | "minute" => 60.,
        "hours" | "hour" => 3600.,
        "days" | "day" => 86400.,
        _ => return fail("unsupported CF time unit"),
    };
    let epoch = moment(origin, false)?.and_utc().timestamp();
    let delta = value * factor;
    if !delta.is_finite() || delta.fract() != 0. || delta.abs() > i64::MAX as f64 {
        return fail("nonintegral/nonfinite archive timestamp");
    }
    epoch
        .checked_add(delta as i64)
        .ok_or_else(|| "archive timestamp overflow".into())
}
fn recorded(stages: &mut Vec<Value>, name: &str, start: Instant, parallelism: String) {
    stages.push(json!({"stage":name,"wall_ms":start.elapsed().as_secs_f64()*1000.,"parallelism":parallelism}));
}
fn write_json(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
fn svg(
    link: &Link,
    times: &[i64],
    reference: &[Option<f64>],
    candidate: &[Option<f64>],
    label: &str,
    observations: bool,
) -> String {
    let high = reference
        .iter()
        .chain(candidate)
        .flatten()
        .copied()
        .fold(0., f64::max)
        .max(
            if reference
                .iter()
                .chain(candidate)
                .flatten()
                .all(|v| *v == 0.)
            {
                1.
            } else {
                f64::MIN_POSITIVE
            },
        )
        * 1.08;
    let low = reference
        .iter()
        .chain(candidate)
        .flatten()
        .copied()
        .fold(0., f64::min);
    let x = |i: usize| 80. + 940. * i as f64 / (times.len().max(2) - 1) as f64;
    let y = |v: f64| 350. - 240. * (v - low) / (high - low);
    let line = |values: &[Option<f64>], colour: &str| {
        let mut path = String::new();
        let mut pen = false;
        for (i, value) in values.iter().enumerate() {
            if let Some(v) = value {
                path.push_str(&format!(
                    "{} {:.3} {:.3} ",
                    if pen { "L" } else { "M" },
                    x(i),
                    y(*v)
                ));
                pen = true;
            } else {
                pen = false;
            }
        }
        let points = values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                v.map(|v| {
                    format!(
                        "<circle cx='{:.3}' cy='{:.3}' r='2' fill='{colour}'/>",
                        x(i),
                        y(v)
                    )
                })
            })
            .collect::<String>();
        format!("<path d='{path}' fill='none' stroke='{colour}' stroke-width='2'/>{points}")
    };
    let mut ticks = String::new();
    for i in 0..=4 {
        let v = low + (high - low) * i as f64 / 4.;
        let yy = y(v);
        let label = if v != 0. && v.abs() < 0.01 {
            format!("{v:.2e}")
        } else {
            format!("{v:.2}")
        };
        ticks.push_str(&format!("<path d='M80 {yy} H1020' stroke='#dde3eb'/><text x='70' y='{}' text-anchor='end'>{label}</text>",yy+4.));
    }
    format!(
        "<svg xmlns='http://www.w3.org/2000/svg' width='1100' height='490' viewBox='0 0 1100 490'><rect width='1100' height='490' fill='white'/><g font-family='sans-serif' font-size='14' fill='#152235'><text x='80' y='30' font-size='21'>Archive reach {} / paired series {} / gauge {}</text><text x='80' y='60'>Streamflow (m³/s) — hourly comparison; UTC</text>{ticks}{}{}<text x='80' y='380'>{}</text><text x='1020' y='380' text-anchor='end'>{}</text><text x='80' y='420' fill='#0671bc'>Blue: published archive</text><text x='470' y='420' fill='#d94b00'>Orange: {}</text><text x='80' y='460'>{}</text></g></svg>",
        link.archive_feature_id,
        xml(&link.candidate_feature_id),
        xml(link.station_id.as_deref().unwrap_or("unspecified")),
        line(reference, "#0671bc"),
        line(candidate, "#d94b00"),
        iso(times[0]),
        iso(*times.last().unwrap()),
        xml(label),
        if observations {
            "Missing samples stay gaps. Scores evaluate the archive against observations; no NextGen output."
        } else {
            "Missing samples stay gaps. Model agreement does not establish observational skill."
        }
    )
}
fn score(times: &[i64], r: &[Option<f64>], c: &[Option<f64>], observations: bool) -> Value {
    let paired = r
        .iter()
        .zip(c)
        .filter_map(|(r, c)| Some(((*r)?, (*c)?)))
        .collect::<Vec<_>>();
    if paired.is_empty() {
        return json!({"paired":0,"expected":times.len(),"status":"no_paired_samples"});
    }
    let n = paired.len() as f64;
    let mean = paired.iter().map(|(r, _)| r).sum::<f64>() / n;
    let sse = paired.iter().map(|(r, c)| (c - r).powi(2)).sum::<f64>();
    let var = paired.iter().map(|(r, _)| (r - mean).powi(2)).sum::<f64>();
    let complete = paired.len() == times.len();
    let peak = |v: &[Option<f64>]| {
        v.iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| (i, v)))
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
    };
    let lag = if complete {
        peak(c)
            .zip(peak(r))
            .map(|((c, _), (r, _))| times[c] - times[r])
    } else {
        None
    };
    let mut value = json!({"paired":paired.len(),"expected":times.len(),"status":if complete{"complete"}else{"incomplete"},"bias_m3_s":paired.iter().map(|(r,c)|c-r).sum::<f64>()/n,"rmse_m3_s":(sse/n).sqrt(),"prediction":if observations{"archive"}else{"candidate"},"reference":if observations{"observations"}else{"archive"},"bias_definition":"prediction minus reference","differing_decoded_f64_words":paired.iter().filter(|(r,c)|r.to_bits()!=c.to_bits()).count()});
    value[if observations {
        "nse_against_observations"
    } else {
        "nse_against_model_reference"
    }] = json!(if var > 0. { Some(1. - sse / var) } else { None });
    value[if observations {
        "archive_minus_observed_peak_seconds"
    } else {
        "candidate_minus_archive_peak_seconds"
    }] = json!(lag);
    value
}
fn output_directory(requested: &Path) -> Result<PathBuf> {
    match fs::create_dir(requested) {
        Ok(()) => return Ok(requested.to_path_buf()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && requested.is_dir() => (),
        Err(e) => return Err(e.into()),
    }
    // Repeated ordinary runs retain earlier results. Atomic creation also
    // handles two launches choosing the same requested directory concurrently.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    for sequence in 0_u64.. {
        let path = requested.join(format!("run-{stamp}-{}-{sequence}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(e) => return Err(e.into()),
        }
    }
    unreachable!()
}
pub(super) fn run(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        return fail("usage: rw_zarr reach-archive REQUEST.json OUTPUT_DIRECTORY");
    }
    let job: Job = serde_json::from_slice(&fs::read(&args[0])?)?;
    let output = output_directory(Path::new(&args[1]))?;
    let workers = job
        .workers
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        })
        .clamp(1, 64);
    let source = &job.source;
    if source.cadence_seconds <= 0 || source.units != "m3 s-1" {
        return fail("reach archive requires positive cadence and m3 s-1 streamflow units");
    }
    let storage = store(&job)?;
    let mut stages = Vec::new();
    let start = Instant::now();
    let arrays = parallel(
        &[
            source.feature.as_str(),
            source.time.as_str(),
            source.field.as_str(),
        ],
        3,
        |n| open(&storage, n).map_err(message),
    )?;
    let (features, time, field) = (&arrays[0], &arrays[1], &arrays[2]);
    if features.shape().len() != 1
        || time.shape().len() != 1
        || time.shape()[0] < 2
        || field.shape().len() != 2
    {
        return fail(
            "archive must have one-dimensional feature/time coordinates and two-dimensional flow",
        );
    }
    let actual = field
        .attributes()
        .get("units")
        .and_then(Value::as_str)
        .unwrap_or("")
        .replace(' ', "");
    if !["m3s-1", "m^3s^-1", "m3/s"].contains(&actual.as_str()) {
        return fail(format!("archive flow units {actual:?} do not match m3 s-1"));
    }
    let dims = field
        .attributes()
        .get("_ARRAY_DIMENSIONS")
        .and_then(Value::as_array)
        .ok_or("flow dimension names missing")?;
    if dims.len() != 2 {
        return fail("flow must name exactly its two coordinate dimensions");
    }
    let t_axis = dims
        .iter()
        .position(|v| v.as_str() == Some(&source.time))
        .ok_or("flow time dimension missing")?;
    let f_axis = dims
        .iter()
        .position(|v| v.as_str() == Some(&source.feature))
        .ok_or("flow feature dimension missing")?;
    if t_axis == f_axis
        || field.shape()[t_axis] != time.shape()[0]
        || field.shape()[f_axis] != features.shape()[0]
    {
        return fail("flow coordinate shapes disagree");
    }
    recorded(
        &mut stages,
        "metadata",
        start,
        "3 independent array workers".into(),
    );
    let start = Instant::now();
    let endpoints = parallel(&[0, 1, 2], 3, |i| match i {
        0 => numbers(time, std::slice::from_ref(&(0..1))).map_err(message),
        1 => numbers(time, std::slice::from_ref(&(time.shape()[0] - 1..time.shape()[0]))).map_err(message),
        _ => numbers(features, std::slice::from_ref(&(0..features.shape()[0].min(3)))).map_err(message),
    })?;
    let first = cf_time(time, endpoints[0][0])?;
    let last = cf_time(time, endpoints[1][0])?;
    let duration = i64::try_from(time.shape()[0] - 1)?
        .checked_mul(source.cadence_seconds)
        .ok_or("archive duration overflow")?;
    if first.checked_add(duration) != Some(last) {
        return fail("archive endpoints disagree with declared regular cadence");
    }
    recorded(
        &mut stages,
        "coverage_and_feature_preview",
        start,
        "3 workers: two time endpoints and first three feature IDs".into(),
    );
    let coverage = json!({"schema":"arwen.reach-archive-coverage.v1","source_id":source.id,"first_utc":iso(first),"last_utc":iso(last),"time_count":time.shape()[0],"feature_count":features.shape()[0],"feature_preview":endpoints[2],"flow_metadata":field.metadata(),"time_attributes":time.attributes(),"stages":stages});
    write_json(output.join("coverage.json"), &coverage)?;
    let (Some(begin), Some(end)) = (&job.start, &job.end) else {
        if job.start.is_some() || job.end.is_some() {
            return fail("event start and end must be supplied together");
        }
        println!("{}", serde_json::to_string(&coverage)?);
        return Ok(());
    };
    let (begin, end) = (timestamp(begin)?, timestamp(end)?);
    if begin < first || end > last || begin > end {
        return fail(format!(
            "requested event is outside archive coverage {} through {}; select an overlapping event, never substitute operational output",
            iso(first),
            iso(last)
        ));
    }
    if (begin - first) % source.cadence_seconds != 0 || (end - first) % source.cadence_seconds != 0
    {
        return fail("event boundaries are off the archive cadence");
    }
    if job.reaches.is_empty() {
        return fail("event requires explicit archive/candidate feature mapping rows");
    }
    let row_ids = job
        .reaches
        .iter()
        .map(|r| r.archive_feature_id)
        .collect::<BTreeSet<_>>();
    if row_ids.len() != job.reaches.len() {
        return fail("duplicate archive feature mapping");
    }
    let (lo, hi) = (
        ((begin - first) / source.cadence_seconds) as u64,
        ((end - first) / source.cadence_seconds) as u64 + 1,
    );
    let times = (lo..hi)
        .map(|i| first + i as i64 * source.cadence_seconds)
        .collect::<Vec<_>>();
    let start = Instant::now();
    let coordinates = parallel(&[0, 1], 2, |kind| {
        if *kind == 0 {
            numbers(features, std::slice::from_ref(&(0..features.shape()[0]))).map_err(message)
        } else {
            numbers(time, std::slice::from_ref(&(lo..hi))).map_err(message)
        }
    })?;
    for (i, v) in coordinates[1].iter().enumerate() {
        if cf_time(time, *v)? != times[i] {
            return fail("event time coordinate is not the declared regular cadence");
        }
    }
    let metadata = serde_json::to_value(field.metadata())?;
    if metadata.get("zarr_format").and_then(Value::as_u64) != Some(2)
        || metadata.get("chunks").and_then(Value::as_array).is_none()
    {
        return fail("reach archive extraction currently requires a regular Zarr v2 chunk grid");
    }
    let chunk = field.chunk_shape(&[0, 0])?;
    let mut groups: BTreeMap<u64, Vec<(usize, u64)>> = BTreeMap::new();
    let index_chunks = coordinates[0]
        .chunks(coordinates[0].len().div_ceil(workers).max(1))
        .enumerate()
        .collect::<Vec<_>>();
    let positions = parallel(&index_chunks, workers, |(block, values)| {
        let mut found = Vec::new();
        let base = block * coordinates[0].len().div_ceil(workers).max(1);
        for (i, v) in values.iter().enumerate() {
            if !v.is_finite() || v.fract() != 0. || v.abs() > 9007199254740991. {
                return Err("feature coordinate is not an exact integer".into());
            }
            if row_ids.contains(&(*v as i64)) {
                found.push((*v as i64, (base + i) as u64));
            }
        }
        Ok(found)
    })?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    for (r, link) in job.reaches.iter().enumerate() {
        let positions = positions
            .iter()
            .filter(|(id, _)| *id == link.archive_feature_id)
            .collect::<Vec<_>>();
        if positions.len() != 1 {
            return fail(format!(
                "archive feature {} has {} matches",
                link.archive_feature_id,
                positions.len()
            ));
        }
        let p = positions[0].1;
        groups
            .entry(p / u64::from(chunk[f_axis]))
            .or_default()
            .push((r, p));
    }
    recorded(
        &mut stages,
        "coordinates_and_feature_join",
        start,
        format!("2 coordinate workers; {workers} feature scan workers"),
    );
    let start = Instant::now();
    let tasks = groups
        .into_iter()
        .flat_map(|(f, rows)| {
            (lo / u64::from(chunk[t_axis])..=(hi - 1) / u64::from(chunk[t_axis]))
                .map(move |t| (t, f, rows.clone()))
        })
        .collect::<Vec<_>>();
    let slabs = parallel(&tasks, workers, |(t, f, rows)| {
        let a = (*t * u64::from(chunk[t_axis])).max(lo);
        let b = ((*t + 1) * u64::from(chunk[t_axis])).min(hi);
        let left = *f * u64::from(chunk[f_axis]);
        let right = ((*f + 1) * u64::from(chunk[f_axis])).min(features.shape()[0]);
        let ranges = if t_axis == 0 {
            vec![a..b, left..right]
        } else {
            vec![left..right, a..b]
        };
        let values = numbers(field, &ranges).map_err(message)?;
        let mut result = Vec::new();
        for (r, p) in rows {
            for ti in a..b {
                let i = if t_axis == 0 {
                    (ti - a) * (right - left) + p - left
                } else {
                    (p - left) * (b - a) + ti - a
                };
                let v = values[i as usize];
                result.push((*r, (ti - lo) as usize, v.is_finite().then_some(v)));
            }
        }
        Ok(result)
    })?;
    let mut reference = vec![vec![None; times.len()]; job.reaches.len()];
    for (r, t, v) in slabs.into_iter().flatten() {
        reference[r][t] = v;
    }
    recorded(
        &mut stages,
        "archive_flow",
        start,
        format!(
            "{} distinct chunk tasks; {} concurrent workers; zarrs codec threads",
            tasks.len(),
            workers.min(tasks.len())
        ),
    );
    let start = Instant::now();
    let candidate = job
        .candidate
        .as_ref()
        .map(|p| -> Result<Candidate> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
        .transpose()?;
    let mut lookup = BTreeMap::new();
    if let Some(c) = &candidate {
        if c.schema != "arwen.point-flow.v1" || c.units != "m3 s-1" {
            return fail("candidate schema/units must be arwen.point-flow.v1 and m3 s-1");
        }
        for p in &c.samples {
            if p.streamflow_m3_s.is_some_and(|v| !v.is_finite()) {
                return fail("candidate nonfinite flow must be null");
            }
            if lookup
                .insert(
                    (p.feature_id.clone(), timestamp(&p.time_utc)?),
                    p.streamflow_m3_s,
                )
                .is_some()
            {
                return fail("duplicate candidate feature/time sample");
            }
        }
    }
    let jobs = job.reaches.iter().enumerate().collect::<Vec<_>>();
    let observations = candidate
        .as_ref()
        .is_some_and(|c| c.kind == CandidateKind::Observations);
    let label = candidate
        .as_ref()
        .and_then(|c| c.label.as_deref())
        .unwrap_or(if observations {
            "observations"
        } else if candidate.is_some() {
            "candidate"
        } else {
            "candidate pending (archive only)"
        });
    let results = parallel(&jobs, workers, |(r, link)| {
        let values = times
            .iter()
            .map(|t| {
                lookup
                    .get(&(link.candidate_feature_id.clone(), *t))
                    .copied()
                    .flatten()
            })
            .collect::<Vec<_>>();
        fs::write(
            output.join(format!("reach-{}.svg", link.archive_feature_id)),
            svg(link, &times, &reference[*r], &values, label, observations),
        )
        .map_err(message)?;
        let scores = if observations {
            score(&times, &values, &reference[*r], true)
        } else {
            score(&times, &reference[*r], &values, false)
        };
        Ok((scores, values))
    })?;
    recorded(
        &mut stages,
        "candidate_join_scores_hydrographs",
        start,
        format!(
            "{} reach workers; JSON control interchange decode",
            workers.min(jobs.len())
        ),
    );
    let start = Instant::now();
    let mut table = String::from(
        "archive_feature_id,candidate_feature_id,time_utc,archive_m3_s,candidate_m3_s\n",
    );
    let mut samples = Vec::new();
    let mut summaries = Vec::new();
    for (r, link) in job.reaches.iter().enumerate() {
        summaries.push(json!({"archive_feature_id":link.archive_feature_id,"candidate_feature_id":link.candidate_feature_id,"station_id":link.station_id,"scores":results[r].0}));
        for (i, t) in times.iter().enumerate() {
            let a = reference[r][i];
            let b = results[r].1[i];
            table.push_str(&format!(
                "{},\"{}\",{},{},{}\n",
                link.archive_feature_id,
                link.candidate_feature_id.replace('"', "\"\""),
                iso(*t),
                a.map(|v| v.to_string()).unwrap_or_default(),
                b.map(|v| v.to_string()).unwrap_or_default()
            ));
            samples.push(json!({"feature_id":link.archive_feature_id.to_string(),"time_utc":iso(*t),"streamflow_m3_s":a}));
        }
    }
    fs::write(output.join("paired.csv"), table)?;
    write_json(
        output.join("reference.json"),
        &json!({"schema":"arwen.point-flow.v1","units":"m3 s-1","provenance":source.provenance,"samples":samples}),
    )?;
    let empty = BTreeMap::new();
    let cp = candidate.as_ref().map(|c| &c.provenance).unwrap_or(&empty);
    let factors = source
        .provenance
        .keys()
        .chain(cp.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let differences=factors.into_iter().map(|factor|{let a=source.provenance.get(&factor);let b=cp.get(&factor);json!({"factor":factor,"archive":a,"candidate":b,"status":if a.is_none()||b.is_none(){"unknown"}else if a==b{"same_declared"}else{"different_declared"}})}).collect::<Vec<_>>();
    recorded(
        &mut stages,
        "reduced_tables",
        start,
        "coordinator; reduced event records only".into(),
    );
    let report = json!({"schema":"arwen.reach-archive-comparison.v1","output_directory":output,"status":if candidate.is_some(){"comparison_complete"}else{"reference_only"},"comparison_kind":if observations{"archive_against_observations"}else{"candidate_against_archive"},"source_id":source.id,"start":iso(begin),"end":iso(end),"results":summaries,"provenance":differences,"interpretation":"Declared differences are context, not isolated causes; archive is model output, not observations.","stages":stages,"routing_step_ms":null,"routing_step_work":"none; offline Rust analysis","eye_review":"pending; open every SVG before acceptance"});
    write_json(output.join("report.json"), &report)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
