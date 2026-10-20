//! Fixed-grid segment composition. No resampling or invented boundary frames.
use chrono::{DateTime, Utc};
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
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    latitude: String,
    longitude: String,
    time: String,
    terrain: String,
    terrain_authority: PathBuf,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    cadence_seconds: i64,
    required_columns: BTreeMap<String, Column>,
    required_surfaces: BTreeMap<String, String>,
    supplement_role: String,
    segments: Vec<Segment>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Column {
    units: String,
    vertical_coordinate: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Segment {
    id: String,
    mesh: PathBuf,
    init: PathBuf,
    frames: Vec<PathBuf>,
}
#[derive(Clone, Serialize)]
struct Frame {
    file: PathBuf,
    segment: String,
    mesh: PathBuf,
    init: PathBuf,
    valid_time: DateTime<Utc>,
    variables: BTreeSet<String>,
    columns: BTreeMap<String, Range>,
    wall_seconds: f64,
}
#[derive(Clone, Serialize)]
struct Range {
    minimum: f64,
    maximum: f64,
}
struct Grid {
    lat: Vec<f64>,
    lon: Vec<f64>,
    terrain: Vec<f64>,
    vertical_axes: BTreeMap<String, (Vec<f64>, String)>,
}

fn resolve(root: &Path, path: &Path) -> Result<PathBuf, String> {
    fs::canonicalize(root.join(path)).map_err(|e| format!("{}: {e}", root.join(path).display()))
}
pub(super) fn values(file: &netcrust::File, name: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
    let variable = file
        .variable(name)
        .ok_or_else(|| format!("missing {name}"))?;
    let array = variable.array_f64().map_err(|e| e.to_string())?;
    let shape = array.shape().to_vec();
    let scale = variable
        .attribute("scale_factor")
        .and_then(|a| a.as_f64())
        .unwrap_or(1.0);
    let offset = variable
        .attribute("add_offset")
        .and_then(|a| a.as_f64())
        .unwrap_or(0.0);
    let fill = variable.attribute("_FillValue").and_then(|a| a.as_f64());
    let missing = variable.attribute("missing_value").and_then(|a| a.as_f64());
    if !scale.is_finite() || scale == 0.0 || !offset.is_finite() {
        return Err(format!("{name}: invalid CF packing"));
    }
    let mut data = array.into_values();
    for value in &mut data {
        if !value.is_finite() || fill == Some(*value) || missing == Some(*value) {
            return Err(format!(
                "{name}: missing/nonfinite value in required full field"
            ));
        }
        *value = *value * scale + offset;
        if !value.is_finite() {
            return Err(format!("{name}: nonfinite unpacked value"));
        }
    }
    Ok((data, shape))
}
fn units(file: &netcrust::File, name: &str, expected: &str) -> Result<(), String> {
    let actual = file.variable(name).and_then(|v| {
        v.attribute("units")
            .and_then(|a| a.as_string())
            .map(str::to_owned)
    });
    if actual.as_deref() != Some(expected) {
        return Err(format!("{name}: units {actual:?}, expected {expected}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

    fn fixture(path: &Path, seconds: f64, terrain: f64, w: f64) {
        fixture_levels(path, seconds, terrain, w, &[1., 2., 3.]);
    }
    fn fixture_levels(path: &Path, seconds: f64, terrain: f64, w: f64, levels: &[f64]) {
        let mut schema = Schema::new(NcFormat::Classic);
        let t = schema.def_dim("Time", 1, false).unwrap();
        let z = schema.def_dim("level", 3, false).unwrap();
        let y = schema.def_dim("y", 2, false).unwrap();
        let x = schema.def_dim("x", 2, false).unwrap();
        let mut variable = |name, dims: &[usize], units: &str| {
            let id = schema.def_var(name, NcType::Double, dims).unwrap();
            schema
                .put_var_attr(id, "units", AttrValue::Text(units.into()))
                .unwrap();
            id
        };
        let time = variable("Time", &[t], "seconds since 1970-01-01 00:00:00");
        let level = variable("level", &[z], "1");
        let lat = variable("lat", &[y], "degrees_north");
        let lon = variable("lon", &[x], "degrees_east");
        let hgt = variable("HGT", &[t, y, x], "m");
        let vel = variable("W", &[t, z, y, x], "m s-1");
        let mut writer = NcWriter::create(path, schema).unwrap();
        for (id, values) in [
            (time, vec![seconds]),
            (level, levels.to_vec()),
            (lat, vec![39.0, 40.0]),
            (lon, vec![-85.0, -84.0]),
            (hgt, vec![terrain; 4]),
            (vel, vec![w; 12]),
        ] {
            writer.write_var(id, VarData::F64(&values)).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn real_netcdf_segments_order_deduplicate_and_refuse_terrain_or_cadence_changes() {
        let root = std::env::temp_dir().join(format!("segment-composer-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        for (name, time, w) in [
            ("a0.nc", 0.0, 1.0),
            ("a1.nc", 300.0, 2.0),
            ("b0.nc", 300.0, 2.5),
            ("b1.nc", 600.0, 3.0),
        ] {
            fixture(&root.join(name), time, 100.0, w);
        }
        fs::write(root.join("mesh-a"), "mesh a provenance").unwrap();
        fs::write(root.join("mesh-b"), "mesh b provenance").unwrap();
        let manifest = root.join("segments.json");
        let document = json!({"schema":"gpuwm-segment-source-v1","latitude":"lat","longitude":"lon","time":"Time","terrain":"HGT","terrain_authority":"a0.nc","start":"1970-01-01T00:00:00Z","end":"1970-01-01T00:10:00Z","cadence_seconds":300,"required_columns":{"W":{"units":"m s-1","vertical_coordinate":"level"}},"required_surfaces":{},"supplement_role":"surface","segments":[{"id":"a","mesh":"mesh-a","init":"mesh-a","frames":["a1.nc","a0.nc"]},{"id":"b","mesh":"mesh-b","init":"mesh-b","frames":["b0.nc","b1.nc"]}]});
        fs::write(&manifest, serde_json::to_vec(&document).unwrap()).unwrap();
        let out = root.join("composed");
        compose(&manifest, &out, 2).unwrap();
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("composition.json")).unwrap()).unwrap();
        assert_eq!(receipt["frame_count"], 3);
        assert_eq!(receipt["duplicates_removed"].as_array().unwrap().len(), 1);
        assert_eq!(receipt["frames"][1]["segment"], "b");
        assert_eq!(receipt["frames"][1]["columns"]["W"]["maximum"], 2.5);
        assert_eq!(receipt["parallelism"]["frame_reader_threads"], 2);
        fixture(&root.join("b1.nc"), 600.0, 101.0, 3.0);
        assert!(compose(&manifest, &root.join("bad-terrain"), 2)
            .unwrap_err()
            .contains("terrain changes"));
        fixture_levels(&root.join("b1.nc"), 600.0, 100.0, 3.0, &[1., 2., 4.]);
        assert!(compose(&manifest, &root.join("bad-levels"), 2)
            .unwrap_err()
            .contains("vertical axes differ"));
        fixture(&root.join("b1.nc"), 900.0, 100.0, 3.0);
        assert!(compose(&manifest, &root.join("bad-time"), 2)
            .unwrap_err()
            .contains("outside requested window"));
        assert!(compose(&manifest, &out, 2)
            .unwrap_err()
            .contains("already exists"));
        fs::remove_dir_all(root).unwrap();
    }
}
fn axis(file: &netcrust::File, name: &str) -> Result<Vec<f64>, String> {
    let (data, shape) = values(file, name)?;
    if data.len() < 2
        || shape != [data.len()]
        || !(data.windows(2).all(|w| w[0] < w[1]) || data.windows(2).all(|w| w[0] > w[1]))
    {
        return Err(format!(
            "{name}: expected a finite strictly monotonic 1D axis"
        ));
    }
    Ok(data)
}
fn grid(file: &netcrust::File, m: &Manifest) -> Result<Grid, String> {
    units(file, &m.latitude, "degrees_north")?;
    units(file, &m.longitude, "degrees_east")?;
    units(file, &m.terrain, "m")?;
    let lat = axis(file, &m.latitude)?;
    let lon = axis(file, &m.longitude)?;
    let (terrain, shape) = values(file, &m.terrain)?;
    if shape != [lat.len(), lon.len()] && shape != [1, lat.len(), lon.len()] {
        return Err("terrain must have exactly one full geographic plane".into());
    }
    let mut vertical_axes = BTreeMap::new();
    for column in m.required_columns.values() {
        let name = &column.vertical_coordinate;
        if !vertical_axes.contains_key(name) {
            let coordinates = axis(file, name)?;
            let units = file
                .variable(name)
                .and_then(|v| {
                    v.attribute("units")
                        .and_then(|a| a.as_string())
                        .map(str::to_owned)
                })
                .ok_or_else(|| format!("{name}: vertical axis has no units"))?;
            vertical_axes.insert(name.clone(), (coordinates, units));
        }
    }
    Ok(Grid {
        lat,
        lon,
        terrain,
        vertical_axes,
    })
}
fn inspect(
    path: &Path,
    segment: &Segment,
    m: &Manifest,
    reference: &Grid,
) -> Result<Frame, String> {
    let started = Instant::now();
    let (file, _) = super::open(path)?;
    let observed = grid(&file, m)?;
    if observed.lat != reference.lat || observed.lon != reference.lon {
        return Err(format!("{}: geographic grid differs from terrain authority; export each segment on the same fixed lat/lon grid",path.display()));
    }
    if observed.vertical_axes != reference.vertical_axes {
        return Err(format!(
            "{}: vertical axes differ from the composition authority",
            path.display()
        ));
    }
    if observed.terrain != reference.terrain {
        let delta = observed
            .terrain
            .iter()
            .zip(&reference.terrain)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        return Err(format!("{}: terrain changes by up to {delta} m; use a common geographic terrain authority before composition",path.display()));
    }
    let (times, shape) = values(&file, &m.time)?;
    if shape != [1] || times.len() != 1 {
        return Err(format!(
            "{}: export one full-field file per valid time (one Time record)",
            path.display()
        ));
    }
    let time_variable = file.variable(&m.time).ok_or("missing time")?;
    let time_units = time_variable
        .attribute("units")
        .and_then(|a| a.as_string())
        .ok_or("missing CF time units")?;
    if let Some(calendar) = time_variable
        .attribute("calendar")
        .and_then(|a| a.as_string())
    {
        if !["standard", "gregorian", "proleptic_gregorian"].contains(&calendar) {
            return Err(format!("unsupported calendar {calendar}"));
        }
    }
    let time_ref = super::parse_cf_units(time_units)?.ok_or("time is not CF reference time")?;
    let time_strings = super::decode_times(&time_ref, &times)?;
    let valid_time = DateTime::parse_from_rfc3339(&time_strings[0])
        .map_err(|e| e.to_string())?
        .with_timezone(&Utc);
    let mut columns = BTreeMap::new();
    for (name, column) in &m.required_columns {
        units(&file, name, &column.units)?;
        let levels = axis(&file, &column.vertical_coordinate)?;
        let (data, shape) = values(&file, name)?;
        if shape != [1, levels.len(), reference.lat.len(), reference.lon.len()] {
            return Err(format!(
                "{name}: expected one complete time/vertical/y/x column, got {shape:?}"
            ));
        }
        columns.insert(
            name.clone(),
            Range {
                minimum: data.iter().copied().fold(f64::INFINITY, f64::min),
                maximum: data.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            },
        );
    }
    for (name, expected_units) in &m.required_surfaces {
        units(&file, name, expected_units)?;
        let (_, shape) = values(&file, name)?;
        if shape != [1, reference.lat.len(), reference.lon.len()] {
            return Err(format!(
                "{name}: expected one complete surface plane, got {shape:?}"
            ));
        }
    }
    let variables = file
        .variables()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|v| v.name().to_owned())
        .collect();
    Ok(Frame {
        file: path.to_owned(),
        segment: segment.id.clone(),
        mesh: segment.mesh.clone(),
        init: segment.init.clone(),
        valid_time,
        variables,
        columns,
        wall_seconds: started.elapsed().as_secs_f64(),
    })
}

pub(super) fn compose(manifest: &Path, output: &Path, jobs: usize) -> Result<(), String> {
    let started = Instant::now();
    if output.exists() {
        return Err(format!(
            "{} already exists; composition preserves prior receipts",
            output.display()
        ));
    }
    let root = manifest.parent().unwrap_or(Path::new("."));
    let mut m: Manifest = serde_json::from_slice(&fs::read(manifest).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if m.schema != "gpuwm-segment-source-v1"
        || m.cadence_seconds <= 0
        || m.end <= m.start
        || (m.end - m.start).num_seconds() % m.cadence_seconds != 0
        || m.required_columns.is_empty()
        || m.supplement_role.is_empty()
        || m.supplement_role.contains('=')
    {
        return Err(
            "invalid segment-source schema, window, cadence, required columns or supplement role"
                .into(),
        );
    }
    m.terrain_authority = resolve(root, &m.terrain_authority)?;
    let (authority, _) = super::open(&m.terrain_authority)?;
    let reference = grid(&authority, &m)?;
    let mut ids = BTreeSet::new();
    let mut tasks = Vec::new();
    for (index, segment) in m.segments.iter_mut().enumerate() {
        if segment.id.is_empty() || !ids.insert(segment.id.clone()) {
            return Err("segment IDs must be nonempty and distinct".into());
        }
        segment.mesh = resolve(root, &segment.mesh)?;
        segment.init = resolve(root, &segment.init)?;
        if segment.frames.is_empty() {
            return Err(format!("segment {} has no exports", segment.id));
        }
        for path in &segment.frames {
            tasks.push((index, resolve(root, path)?));
        }
    }
    if tasks.is_empty() {
        return Err("no segment exports".into());
    }
    let jobs = if jobs == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        jobs
    }
    .min(tasks.len());
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some((segment, path)) = tasks.get(index) else {
                    break;
                };
                let result = inspect(path, &m.segments[*segment], &m, &reference);
                results.lock().unwrap().push((index, result));
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|(index, _)| *index);
    let results: Vec<_> = results
        .into_iter()
        .map(|(_, r)| r)
        .collect::<Result<_, _>>()?;
    let mut endpoints = BTreeMap::new();
    for frame in &results {
        let bounds = endpoints
            .entry(frame.segment.clone())
            .or_insert((frame.valid_time, frame.valid_time));
        bounds.0 = bounds.0.min(frame.valid_time);
        bounds.1 = bounds.1.max(frame.valid_time);
    }
    let mut ordered = BTreeMap::<DateTime<Utc>, Frame>::new();
    let mut duplicates = Vec::new();
    let mut inventory: Option<BTreeSet<String>> = None;
    for frame in results {
        if frame.valid_time < m.start || frame.valid_time > m.end {
            return Err(format!(
                "{} lies outside requested window",
                frame.file.display()
            ));
        }
        if let Some(expected) = &inventory {
            if &frame.variables != expected {
                return Err(format!(
                    "{}: field inventory changes between frames; moments cannot silently disappear",
                    frame.file.display()
                ));
            }
        } else {
            inventory = Some(frame.variables.clone());
        }
        if let Some(previous) = ordered.get(&frame.valid_time) {
            if previous.segment == frame.segment {
                return Err(format!(
                    "duplicate valid time inside segment {}",
                    frame.segment
                ));
            }
            // Later manifest segment wins at a shared boundary; only endpoints
            // may overlap. Keep both ranges so the remap jump remains visible.
            if previous.valid_time != endpoints[&previous.segment].1
                || frame.valid_time != endpoints[&frame.segment].0
            {
                return Err("segments overlap away from a shared endpoint".into());
            }
            duplicates
                .push(json!({"valid_time":frame.valid_time,"discarded":previous,"retained":frame}));
        }
        ordered.insert(frame.valid_time, frame);
    }
    let expected = ((m.end - m.start).num_seconds() / m.cadence_seconds + 1) as usize;
    if ordered.len() != expected
        || ordered
            .keys()
            .enumerate()
            .any(|(i, t)| *t != m.start + chrono::Duration::seconds(i as i64 * m.cadence_seconds))
    {
        return Err(format!("expected {expected} endpoint-inclusive frames every {} s; found {} unique times (no temporal interpolation is performed)",m.cadence_seconds,ordered.len()));
    }
    let frames: Vec<_> = ordered.into_values().collect();
    let inputs = frames
        .iter()
        .map(|f| f.file.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if inputs.iter().any(|p| p.contains('\n') || p.contains('\r')) {
        return Err("input-list paths cannot contain line breaks".into());
    }
    let supplements = inputs
        .iter()
        .map(|p| format!("{}={p}", m.supplement_role))
        .collect::<Vec<_>>();
    // Publish the receipt last. Original exports are read-only and remain the
    // bytes prep seals in its own input manifest; no multi-GB copying/hashing.
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    fs::write(output.join("inputs.txt"), inputs.join("\n") + "\n").map_err(|e| e.to_string())?;
    let receipt = json!({"schema":"gpuwm-segment-composition-v1","manifest":manifest,"terrain_authority":m.terrain_authority,"terrain_policy":"exactly_invariant_on_identical_coordinates","duplicate_policy":"later_manifest_segment_at_shared_endpoint","duplicates_removed":duplicates,"frames":frames,"supplements":supplements,"cadence_seconds":m.cadence_seconds,"frame_count":expected,"parallelism":{"frame_reader_threads":jobs},"wall_seconds":started.elapsed().as_secs_f64(),"weather_data":"unchanged original full-field exports; prep hashes the input bytes"});
    fs::write(
        output.join("composition.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string(&receipt).map_err(|e| e.to_string())?
    );
    Ok(())
}
