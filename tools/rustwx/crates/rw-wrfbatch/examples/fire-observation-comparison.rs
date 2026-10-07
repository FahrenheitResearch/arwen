//! Compare provenance-bound geographic perimeters with native SFIRE history.
//! All geometry, history decoding and measurements run in Rust. Distances use
//! EPSG:5070 and explicit length sampling; this is a hindcast comparison tool.
use chrono::{NaiveDateTime, TimeZone, Utc};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use static_fields::raster::{transform_points, Crs};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

type Point = [f64; 2];
type Segment = [Point; 2];
type Ring = Vec<Point>;
type Polygon = Vec<Ring>;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}
fn file_digest(path: &Path) -> Result<(String, u64), String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut state = Sha256::new();
    let mut count = 0;
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        state.update(&buffer[..n]);
        count += n as u64;
    }
    Ok((format!("{:x}", state.finalize()), count))
}
fn crs() -> Crs {
    Crs::AlbersConusNad83 {
        lat_1: 29.5,
        lat_2: 45.5,
        lat_0: 23.,
        lon_0: -96.,
        false_easting: 0.,
        false_northing: 0.,
    }
}
fn project(points: &mut [Point]) -> Result<(), String> {
    if points.iter().any(|p| {
        !p[0].is_finite()
            || !p[1].is_finite()
            || !(-180.0..=180.0).contains(&p[0])
            || !(-90.0..=90.0).contains(&p[1])
    }) {
        return Err("coordinates must be finite geographic longitude/latitude degrees".into());
    }
    let mut x = points.iter().map(|p| p[0]).collect::<Vec<_>>();
    let mut y = points.iter().map(|p| p[1]).collect::<Vec<_>>();
    transform_points(&Crs::Geographic, &crs(), &mut x, &mut y).map_err(|e| e.to_string())?;
    for (i, p) in points.iter_mut().enumerate() {
        *p = [x[i], y[i]];
    }
    if points.iter().flatten().any(|x| !x.is_finite()) {
        return Err("coordinates leave the EPSG:5070 analysis projection".into());
    }
    Ok(())
}
fn ring(value: &Value) -> Result<Ring, String> {
    let mut points = value
        .as_array()
        .ok_or("ring is not an array")?
        .iter()
        .map(|p| {
            Ok([
                p.get(0)
                    .and_then(Value::as_f64)
                    .ok_or("ring longitude is missing")?,
                p.get(1)
                    .and_then(Value::as_f64)
                    .ok_or("ring latitude is missing")?,
            ])
        })
        .collect::<Result<Vec<_>, String>>()?;
    if points.len() < 4 || points.first() != points.last() {
        return Err("observed rings must be closed with at least three vertices".into());
    }
    project(&mut points)?;
    Ok(points)
}
fn polygon(value: &Value) -> Result<Polygon, String> {
    let rings = value.as_array().ok_or("polygon rings are not an array")?;
    if rings.is_empty() {
        return Err("polygon has no outer ring".into());
    }
    rings.iter().map(ring).collect()
}
fn polygons(geometry: &Value) -> Result<Vec<Polygon>, String> {
    let coordinates = geometry
        .get("coordinates")
        .ok_or("geometry lacks coordinates")?;
    match geometry.get("type").and_then(Value::as_str) {
        Some("Polygon") => Ok(vec![polygon(coordinates)?]),
        Some("MultiPolygon") => coordinates
            .as_array()
            .ok_or("multipolygon is not an array")?
            .iter()
            .map(polygon)
            .collect(),
        _ => Err("observations must be Polygon or MultiPolygon geometry".into()),
    }
}
fn area(points: &[Point]) -> f64 {
    if points.len() < 3 {
        return 0.;
    }
    let o = points[0];
    (0..points.len())
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            (a[0] - o[0]) * (b[1] - o[1]) - (b[0] - o[0]) * (a[1] - o[1])
        })
        .sum::<f64>()
        .abs()
        * 0.5
}
fn length(segment: &Segment) -> f64 {
    (segment[1][0] - segment[0][0]).hypot(segment[1][1] - segment[0][1])
}
fn observed_measure(polygons: &[Polygon]) -> Result<(f64, Vec<Segment>), String> {
    let mut total = 0.;
    let mut segments = Vec::new();
    for poly in polygons {
        let mut a = 0.;
        for (index, ring) in poly.iter().enumerate() {
            a += if index == 0 { area(ring) } else { -area(ring) };
            segments.extend(
                ring.windows(2)
                    .filter(|s| s[0] != s[1])
                    .map(|s| [s[0], s[1]]),
            );
        }
        if a <= 0. {
            return Err("observed polygon holes exhaust the outer-ring area".into());
        }
        total += a;
    }
    if segments.is_empty() {
        return Err("observed perimeter has no nonzero edges".into());
    }
    Ok((total, segments))
}

#[derive(Clone, Copy)]
struct Vertex {
    p: Point,
    v: f64,
    id: usize,
}
fn crossing(mut a: Vertex, mut b: Vertex) -> Vertex {
    if a.id > b.id {
        std::mem::swap(&mut a, &mut b);
    }
    if a.v == 0. {
        return a;
    }
    if b.v == 0. {
        return b;
    }
    let t = a.v / (a.v - b.v);
    Vertex {
        p: [
            a.p[0] + t * (b.p[0] - a.p[0]),
            a.p[1] + t * (b.p[1] - a.p[1]),
        ],
        v: 0.,
        id: usize::MAX,
    }
}
fn clip_triangle(vertices: [Vertex; 3], fourth_value: f64) -> Vec<Vertex> {
    if vertices.iter().all(|p| p.v > 0.)
        || (vertices.iter().all(|p| p.v == 0.) && fourth_value > 0.)
    {
        return Vec::new();
    }
    let mut output = Vec::new();
    for index in 0..3 {
        let a = vertices[index];
        let b = vertices[(index + 1) % 3];
        if (a.v <= 0.) != (b.v <= 0.) {
            output.push(crossing(a, b));
        }
        if b.v <= 0. {
            output.push(b);
        }
    }
    output
}
fn segment_key(segment: Segment) -> [u64; 4] {
    let mut a = [segment[0][0].to_bits(), segment[0][1].to_bits()];
    let mut b = [segment[1][0].to_bits(), segment[1][1].to_bits()];
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    [a[0], a[1], b[0], b[1]]
}
fn model_geometry(
    points: &[Point],
    lfn: &[f64],
    shape: [usize; 2],
) -> Result<(f64, Vec<Segment>), String> {
    let [ny, nx] = shape;
    if ny < 2 || nx < 2 || points.len() != nx * ny || lfn.len() != nx * ny {
        return Err(
            "LFN and projected coordinates require matching grids of at least 2 by 2".into(),
        );
    }
    if lfn.iter().any(|v| !v.is_finite()) {
        return Err("LFN contains nonfinite values".into());
    }
    if (0..ny).any(|j| lfn[j * nx] <= 0. || lfn[j * nx + nx - 1] <= 0.)
        || (0..nx).any(|i| lfn[i] <= 0. || lfn[(ny - 1) * nx + i] <= 0.)
    {
        return Err(
            "burned LFN reaches the stored-grid boundary; its perimeter is truncated".into(),
        );
    }
    let mut total = 0.;
    let mut edges: HashMap<[u64; 4], Segment> = HashMap::new();
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let ids = [
                j * nx + i,
                j * nx + i + 1,
                (j + 1) * nx + i + 1,
                (j + 1) * nx + i,
            ];
            for (triangle, fourth) in [
                ([ids[0], ids[1], ids[2]], ids[3]),
                ([ids[0], ids[2], ids[3]], ids[1]),
            ] {
                let clipped = clip_triangle(
                    triangle.map(|id| Vertex {
                        p: points[id],
                        v: lfn[id],
                        id,
                    }),
                    lfn[fourth],
                );
                let clipped_area = area(&clipped.iter().map(|v| v.p).collect::<Vec<_>>());
                total += clipped_area;
                if clipped_area == 0. {
                    continue;
                }
                for k in 0..clipped.len() {
                    let a = clipped[k];
                    let b = clipped[(k + 1) % clipped.len()];
                    if a.v == 0. && b.v == 0. && a.p != b.p {
                        let edge = [a.p, b.p];
                        let key = segment_key(edge);
                        // A shared zero edge with fire on both sides is interior.
                        if edges.remove(&key).is_none() {
                            edges.insert(key, edge);
                        }
                    }
                }
            }
        }
    }
    let mut segments = edges.into_iter().collect::<Vec<_>>();
    segments.sort_by_key(|entry| entry.0);
    Ok((total, segments.into_iter().map(|entry| entry.1).collect()))
}
fn point_distance_squared(p: Point, segment: &Segment) -> f64 {
    let a = segment[0];
    let d = [segment[1][0] - a[0], segment[1][1] - a[1]];
    let den = d[0] * d[0] + d[1] * d[1];
    let t = if den == 0. {
        0.
    } else {
        ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / den
    }
    .clamp(0., 1.);
    (p[0] - a[0] - t * d[0]).powi(2) + (p[1] - a[1] - t * d[1]).powi(2)
}
fn bounds(segments: &[Segment]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in segments.iter().flatten() {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}
enum Tree {
    Leaf([f64; 4], Vec<Segment>),
    Node([f64; 4], Box<Tree>, Box<Tree>),
}
impl Tree {
    fn new(mut segments: Vec<Segment>) -> Self {
        let b = bounds(&segments);
        if segments.len() <= 12 {
            return Self::Leaf(b, segments);
        }
        let axis = usize::from(b[3] - b[1] > b[2] - b[0]);
        segments.sort_by(|a, b| (a[0][axis] + a[1][axis]).total_cmp(&(b[0][axis] + b[1][axis])));
        let right = segments.split_off(segments.len() / 2);
        Self::Node(b, Box::new(Self::new(segments)), Box::new(Self::new(right)))
    }
    fn box_distance(&self, p: Point) -> f64 {
        let b = match self {
            Self::Leaf(b, _) | Self::Node(b, _, _) => b,
        };
        (b[0] - p[0]).max(0.).max(p[0] - b[2]).powi(2)
            + (b[1] - p[1]).max(0.).max(p[1] - b[3]).powi(2)
    }
    fn nearest(&self, p: Point, best: &mut f64) {
        if self.box_distance(p) > *best {
            return;
        }
        match self {
            Self::Leaf(_, segments) => {
                for s in segments {
                    *best = best.min(point_distance_squared(p, s));
                }
            }
            Self::Node(_, left, right) => {
                let (first, second) = if left.box_distance(p) <= right.box_distance(p) {
                    (left, right)
                } else {
                    (right, left)
                };
                first.nearest(p, best);
                second.nearest(p, best);
            }
        }
    }
    fn distance(&self, p: Point) -> f64 {
        let mut d = f64::INFINITY;
        self.nearest(p, &mut d);
        d.sqrt()
    }
}
struct Samples {
    values: Vec<(f64, f64)>,
    maximum: f64,
    bound: f64,
}
fn sample(source: &[Segment], target: &Tree, spacing: f64) -> Samples {
    let mut result = Samples {
        values: Vec::new(),
        maximum: 0.,
        bound: 0.,
    };
    for s in source {
        let n = (length(s) / spacing).ceil().max(1.) as usize;
        let weight = length(s) / n as f64;
        result.bound = result.bound.max(weight * 0.5);
        for p in s {
            result.maximum = result.maximum.max(target.distance(*p));
        }
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            let p = [
                s[0][0] + t * (s[1][0] - s[0][0]),
                s[0][1] + t * (s[1][1] - s[0][1]),
            ];
            let d = target.distance(p);
            result.maximum = result.maximum.max(d);
            result.values.push((d, weight));
        }
    }
    result
}
fn summary(samples: &[(f64, f64)]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total = samples.iter().map(|s| s.1).sum::<f64>();
    let mean = samples.iter().map(|s| s.0 * s.1).sum::<f64>() / total;
    let mut accumulated = 0.;
    let mut p95 = 0.;
    for (d, weight) in sorted {
        accumulated += weight;
        p95 = d;
        if accumulated >= 0.95 * total {
            break;
        }
    }
    json!({ "mean_m": mean, "p95_m": p95, "sample_count": samples.len(), "boundary_length_m": total })
}
fn distances(observed: &[Segment], modeled: &[Segment], spacing: f64) -> Value {
    if modeled.is_empty() {
        return json!({ "status": "no-modeled-perimeter", "mean_m": null, "p95_m": null, "hausdorff_m": null });
    }
    let a = sample(observed, &Tree::new(modeled.to_vec()), spacing);
    let b = sample(modeled, &Tree::new(observed.to_vec()), spacing);
    let directed_a = summary(&a.values);
    let directed_b = summary(&b.values);
    let mut both = a.values.clone();
    both.extend_from_slice(&b.values);
    let maximum = a.maximum.max(b.maximum);
    let bound = a.bound.max(b.bound);
    json!({ "status": "measured", "observed_to_modeled": directed_a, "modeled_to_observed": directed_b,
        "symmetric_equal_direction_mean_m": 0.5 * (directed_a["mean_m"].as_f64().unwrap() + directed_b["mean_m"].as_f64().unwrap()),
        "symmetric_max_direction_p95_m": directed_a["p95_m"].as_f64().unwrap().max(directed_b["p95_m"].as_f64().unwrap()),
        "symmetric_length_weighted": summary(&both), "hausdorff_sampled_m": maximum,
        "hausdorff_upper_bound_m": maximum + bound, "sampling_error_bound_m": bound })
}

fn epoch_ms(time: &str) -> Result<i64, String> {
    let time = time.trim_end_matches(['\0', ' ']);
    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(time) {
        return Ok(parsed.timestamp_millis());
    }
    let parsed = NaiveDateTime::parse_from_str(time, "%Y-%m-%d_%H:%M:%S")
        .map_err(|e| format!("native UTC Times {time:?}: {e}"))?;
    Ok(Utc.from_utc_datetime(&parsed).timestamp_millis())
}
fn nearest(times: &[i64], observed: i64) -> usize {
    times
        .iter()
        .enumerate()
        .min_by_key(|(_, time)| ((**time as i128 - observed as i128).abs(), **time))
        .unwrap()
        .0
}
fn plane(
    file: &netcrust::File,
    name: &str,
    record: usize,
    shape: Option<[usize; 2]>,
) -> Result<(Vec<f64>, [usize; 2]), String> {
    let data = file
        .read_array_f64_record_or_all(name, record as u64)
        .map_err(|e| format!("{name}: {e}"))?;
    let dims = data.shape();
    let dims = if dims.len() == 3 && dims[0] == 1 {
        &dims[1..]
    } else {
        dims
    };
    if dims.len() != 2 || dims.contains(&0) {
        return Err(format!(
            "{name} requires a nonempty fine-grid plane, got {:?}",
            data.shape()
        ));
    }
    let actual = [dims[0], dims[1]];
    if shape.is_some_and(|expected| expected != actual) {
        return Err(format!(
            "{name} shape {actual:?} differs from fire coordinates {shape:?}"
        ));
    }
    Ok((data.into_values(), actual))
}
fn footprint(points: &[Point], shape: [usize; 2]) -> Ring {
    let [ny, nx] = shape;
    let mut ring = Vec::new();
    ring.extend((0..nx).map(|i| points[i]));
    ring.extend((1..ny).map(|j| points[j * nx + nx - 1]));
    ring.extend((0..nx - 1).rev().map(|i| points[(ny - 1) * nx + i]));
    ring.extend((1..ny - 1).rev().map(|j| points[j * nx]));
    ring.push(ring[0]);
    ring
}
fn inside(p: Point, ring: &[Point]) -> bool {
    let mut result = false;
    for s in ring.windows(2) {
        if point_distance_squared(p, &[s[0], s[1]]) <= 1e-12 {
            return true;
        }
        if (s[0][1] > p[1]) != (s[1][1] > p[1])
            && p[0] < s[0][0] + (p[1] - s[0][1]) * (s[1][0] - s[0][0]) / (s[1][1] - s[0][1])
        {
            result = !result;
        }
    }
    result
}
struct Frame {
    path: PathBuf,
    record: usize,
    time: i64,
    hash: String,
}
struct Options {
    observations: PathBuf,
    contract: PathBuf,
    output: PathBuf,
    histories: Vec<PathBuf>,
    spacing: f64,
    max_delta: f64,
}
fn options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut o = Options {
        observations: PathBuf::new(),
        contract: PathBuf::new(),
        output: PathBuf::new(),
        histories: Vec::new(),
        spacing: 5.,
        max_delta: f64::NAN,
    };
    while let Some(key) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{key} lacks its value"))?;
        match key.as_str() {
            "--observations" => o.observations = value.into(),
            "--contract" => o.contract = value.into(),
            "--output" => o.output = value.into(),
            "--history" => o.histories.push(value.into()),
            "--sample-spacing-m" => {
                o.spacing = value.parse().map_err(|_| "sample spacing is not numeric")?
            }
            "--max-time-delta-s" => {
                o.max_delta = value
                    .parse()
                    .map_err(|_| "maximum time delta is not numeric")?
            }
            _ => return Err(format!("unknown argument {key}")),
        }
    }
    if o.observations.as_os_str().is_empty()
        || o.contract.as_os_str().is_empty()
        || o.output.as_os_str().is_empty()
        || o.histories.is_empty()
        || !o.spacing.is_finite()
        || o.spacing <= 0.
        || !o.max_delta.is_finite()
        || o.max_delta < 0.
    {
        return Err("usage: fire-observation-comparison --observations GEOJSON --contract JSON --history WRFOUT [--history WRFOUT ...] --output JSON --max-time-delta-s SECONDS [--sample-spacing-m 5]".into());
    }
    let output = if o.output.exists() {
        o.output.canonicalize().map_err(|e| e.to_string())?
    } else {
        let parent = o
            .output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        parent
            .canonicalize()
            .map_err(|e| format!("output directory: {e}"))?
            .join(o.output.file_name().ok_or("output path lacks a filename")?)
    };
    if std::iter::once(&o.observations)
        .chain(std::iter::once(&o.contract))
        .chain(o.histories.iter())
        .any(|input| input.canonicalize().is_ok_and(|p| p == output))
    {
        return Err(
            "output path must differ from the scientific input paths to prevent overwrite".into(),
        );
    }
    Ok(o)
}
fn run(o: Options) -> Result<Value, String> {
    let raw = read(&o.observations)?;
    let contract_bytes = read(&o.contract)?;
    let contract: Value = serde_json::from_slice(&contract_bytes).map_err(|e| e.to_string())?;
    let raw_hash = digest(&raw);
    if contract["input_sha256"].as_str() != Some(&raw_hash) {
        return Err("observed source hash differs from its captured source contract".into());
    }
    if contract["time_format"].as_str() != Some("unix_ms")
        || contract["coordinate_reference_system"].as_str() != Some("EPSG:4326")
    {
        return Err(
            "source contract must declare UTC unix_ms times and EPSG:4326 coordinates".into(),
        );
    }
    let property = contract["time_property"]
        .as_str()
        .ok_or("source contract lacks timestamp property")?;
    let data: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if data["type"].as_str() != Some("FeatureCollection") {
        return Err("observed source must be a GeoJSON FeatureCollection".into());
    }
    let features = data["features"]
        .as_array()
        .ok_or("observed source has no features")?;
    if features.is_empty() {
        return Err("observed source has no perimeters".into());
    }
    let mut frames = Vec::new();
    let mut input_receipts = Vec::new();
    for path in &o.histories {
        let (hash, bytes) = file_digest(path)?;
        let file = netcrust::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let times = file
            .read_strings("Times")
            .map_err(|e| format!("Times: {e}"))?;
        if times.is_empty() {
            return Err("native history has no UTC Times records".into());
        }
        for (record, time) in times.iter().enumerate() {
            frames.push(Frame {
                path: path.clone(),
                record,
                time: epoch_ms(time)?,
                hash: hash.clone(),
            });
        }
        input_receipts.push(json!({ "file": path.file_name().and_then(|x| x.to_str()), "sha256": hash, "bytes": bytes, "records": times.len() }));
    }
    frames.sort_by(|a, b| (a.time, &a.path, a.record).cmp(&(b.time, &b.path, b.record)));
    if frames.windows(2).any(|pair| pair[0].time == pair[1].time) {
        return Err("history inputs contain duplicate UTC records; select one fire domain and one version of each output".into());
    }
    let times = frames.iter().map(|f| f.time).collect::<Vec<_>>();
    let mut records = Vec::new();
    for feature in features {
        let observed = feature["properties"][property]
            .as_i64()
            .ok_or("observed feature lacks UTC unix_ms time")?;
        let geometry = feature
            .get("geometry")
            .ok_or("observed feature lacks geometry")?;
        let polys = polygons(geometry)?;
        let (observed_area, observed_edges) = observed_measure(&polys)?;
        let frame = &frames[nearest(&times, observed)];
        let delta = (frame.time as i128 - observed as i128) as f64 * 0.001;
        if delta.abs() > o.max_delta {
            return Err(format!("observation {observed} is {:.3} seconds from the nearest native history record, exceeding explicit maximum {}", delta.abs(), o.max_delta));
        }
        let file = netcrust::File::open(&frame.path).map_err(|e| e.to_string())?;
        if file
            .attribute("FIRE_COORDINATE_MODE")
            .and_then(|a| a.as_string().map(str::to_string))
            .as_deref()
            == Some("metric")
        {
            return Err(
                "metric fire coordinates cannot be compared with geographic observations".into(),
            );
        }
        let (lat, shape) = plane(&file, "FXLAT", frame.record, None)?;
        if shape.iter().any(|n| *n < 2) {
            return Err(
                "fire-grid coordinates require at least 2 by 2 points for contour geometry".into(),
            );
        }
        let (lon, _) = plane(&file, "FXLONG", frame.record, Some(shape))?;
        let mut points = lon
            .into_iter()
            .zip(lat)
            .map(|(x, y)| [x, y])
            .collect::<Vec<_>>();
        project(&mut points)?;
        let grid_ring = footprint(&points, shape);
        if polys
            .iter()
            .flatten()
            .flatten()
            .any(|p| !inside(*p, &grid_ring))
        {
            return Err("observed perimeter leaves the stored fire-grid footprint; spatial coverage would truncate its growth".into());
        }
        let (lfn, _) = plane(&file, "LFN", frame.record, Some(shape))?;
        let (fraction, _) = plane(&file, "FIRE_AREA", frame.record, Some(shape))?;
        if fraction
            .iter()
            .any(|f| !f.is_finite() || !(0.0..=1.0).contains(f))
        {
            return Err("FIRE_AREA must contain finite fractions from zero to one".into());
        }
        let attr = |name: &str| {
            file.attribute(name)
                .and_then(|a| a.as_f64())
                .filter(|x| x.is_finite() && *x > 0.)
                .ok_or_else(|| format!("native nominal area needs positive {name} metadata"))
        };
        let dx = (attr("DX")? as f32) / (attr("SR_X")? as f32);
        let dy = (attr("DY")? as f32) / (attr("SR_Y")? as f32);
        if !dx.is_finite() || !dy.is_finite() || dx <= 0. || dy <= 0. {
            return Err("WRF REAL fine spacing must remain finite and positive".into());
        }
        let native_area = fraction.iter().sum::<f64>() * f64::from(dx) * f64::from(dy);
        let (modeled_area, modeled_edges) = model_geometry(&points, &lfn, shape)?;
        records.push(json!({ "observed_utc_epoch_ms": observed, "observed_utc": Utc.timestamp_millis_opt(observed).single().ok_or("observed UTC outside supported calendar")?.to_rfc3339(),
            "geometry_sha256": digest(&serde_json::to_vec(geometry).map_err(|e| e.to_string())?),
            "model_utc_epoch_ms": frame.time, "model_utc": Utc.timestamp_millis_opt(frame.time).single().unwrap().to_rfc3339(),
            "model_minus_observation_s": delta, "outside_model_time_range": observed < times[0] || observed > *times.last().unwrap(),
            "history_file": frame.path.file_name().and_then(|p| p.to_str()), "history_sha256": frame.hash, "history_record": frame.record,
            "fine_grid_shape": shape, "wrf_nominal_fine_dx_m": dx, "wrf_nominal_fine_dy_m": dy,
            "observed_area_m2": observed_area, "observed_perimeter_m": observed_edges.iter().map(length).sum::<f64>(),
            "observed_polygon_count": polys.len(), "observed_ring_count": polys.iter().map(Vec::len).sum::<usize>(),
            "modeled_native_FIRE_AREA_m2": native_area, "modeled_projected_LFN_area_m2": modeled_area,
            "native_FIRE_AREA_error_m2": native_area - observed_area, "native_FIRE_AREA_error_percent": 100. * (native_area - observed_area) / observed_area,
            "projected_LFN_area_error_m2": modeled_area - observed_area, "projected_LFN_area_error_percent": 100. * (modeled_area - observed_area) / observed_area,
            "modeled_perimeter_m": modeled_edges.iter().map(length).sum::<f64>(), "modeled_perimeter_segment_count": modeled_edges.len(),
            "perimeter_distance": distances(&observed_edges, &modeled_edges, o.spacing) }));
    }
    records.sort_by_key(|r| r["observed_utc_epoch_ms"].as_i64().unwrap());
    let (executable_hash, _) = file_digest(&std::env::current_exe().map_err(|e| e.to_string())?)?;
    Ok(
        json!({ "schema": "sfire-observation-comparison-v1", "analysis_crs": "EPSG:5070",
        "comparison_executable_sha256": executable_hash,
        "source_contract": contract, "observation_input_sha256": raw_hash, "source_contract_sha256": digest(&contract_bytes),
        "history_inputs": input_receipts, "max_time_delta_s": o.max_delta, "sample_spacing_m": o.spacing,
        "methods": { "temporal": "Nearest actual UTC output, earlier output on an equal-distance tie; no time interpolation.",
            "model_geometry": "LFN<=0 piecewise-linear clipping of two triangles per projected fine-grid quad, diagonal from lower-left to upper-right; shared interior zero edges cancel. An identically zero triangle inherits the sign of the quad's fourth node, preserving exact-node outer and hole corners; all-zero quads are included. Zero-area clips contribute no boundary edges.",
            "native_area": "Sum of stored FIRE_AREA fractions times WRF REAL DX/SR_X and DY/SR_Y; distinct from projected contour area and initialized burned-center count.",
            "distance": "Exact point-to-segment distances at length-weighted segment midpoints; symmetric mean and p95 pool both directions by boundary length. Endpoints also contribute to sampled Hausdorff. True Hausdorff is bounded above by reported sampled maximum plus at most half a sample interval.",
            "limitations": "Observed time is the published polygon timestamp, not ignition time. Projection applies a geographic NAD83/WGS84 ballpark pass-through. LFN triangulation and finite spatial resolution affect contour area. Suppression, fuel age, observation uncertainty and atmospheric forcing remain physical interpretation limits. No IoU is claimed." },
        "comparisons": records }),
    )
}
fn main() {
    let result = options().and_then(|o| {
        let output = o.output.clone();
        let document = run(o)?;
        Ok((document, output))
    });
    match result {
        Ok((document, output)) => match serde_json::to_vec_pretty(&document)
            .map_err(|e| e.to_string())
            .and_then(|bytes| std::fs::write(output, bytes).map_err(|e| e.to_string()))
        {
            Ok(()) => println!(
                "{} genuine observed perimeters compared in Rust",
                document["comparisons"].as_array().unwrap().len()
            ),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn square(x: f64, y: f64, size: f64) -> Ring {
        vec![
            [x, y],
            [x + size, y],
            [x + size, y + size],
            [x, y + size],
            [x, y],
        ]
    }
    fn edges(r: &Ring) -> Vec<Segment> {
        r.windows(2).map(|s| [s[0], s[1]]).collect()
    }
    #[test]
    fn holes_and_multipolygons_retain_area_and_every_boundary() {
        let polygons = vec![
            vec![square(0., 0., 10.), square(2., 2., 2.)],
            vec![square(20., 0., 3.)],
        ];
        let (a, e) = observed_measure(&polygons).unwrap();
        assert_eq!(a, 105.);
        assert_eq!(e.iter().map(length).sum::<f64>(), 60.);
        assert_eq!(distances(&e, &e, 0.5)["hausdorff_sampled_m"], 0.);
    }
    #[test]
    fn exact_node_zero_square_has_no_internal_diagonal_perimeter() {
        let points = (0..5)
            .flat_map(|j| (0..5).map(move |i| [i as f64, j as f64]))
            .collect::<Vec<_>>();
        let lfn = points
            .iter()
            .map(|p| (p[0] - 2.).abs().max((p[1] - 2.).abs()) - 1.)
            .collect::<Vec<_>>();
        let (a, e) = model_geometry(&points, &lfn, [5, 5]).unwrap();
        assert_eq!(a, 4.);
        assert_eq!(e.iter().map(length).sum::<f64>(), 8.);
        assert_eq!(
            distances(&edges(&square(1., 1., 2.)), &e, 0.1)["hausdorff_sampled_m"],
            0.
        );
    }
    #[test]
    fn translated_square_distance_and_bound_match_analytic_values() {
        let a = edges(&square(0., 0., 10.));
        let b = edges(&square(20., 0., 10.));
        let result = distances(&a, &b, 0.5);
        assert_eq!(result["hausdorff_sampled_m"], 20.);
        assert_eq!(result["hausdorff_upper_bound_m"], 20.25);
        assert!(
            (result["symmetric_length_weighted"]["mean_m"]
                .as_f64()
                .unwrap()
                - 15.)
                .abs()
                < 1e-12
        );
        assert_eq!(result["symmetric_length_weighted"]["p95_m"], 20.);
    }
    #[test]
    fn truncated_fire_and_empty_fire_are_explicit() {
        let points = (0..3)
            .flat_map(|j| (0..3).map(move |i| [i as f64, j as f64]))
            .collect::<Vec<_>>();
        let (a, e) = model_geometry(&points, &vec![1.; 9], [3, 3]).unwrap();
        assert_eq!(a, 0.);
        assert!(e.is_empty());
        assert_eq!(
            distances(&edges(&square(0., 0., 1.)), &e, 1.)["status"],
            "no-modeled-perimeter"
        );
        let mut l = vec![1.; 9];
        l[0] = -1.;
        assert!(model_geometry(&points, &l, [3, 3])
            .unwrap_err()
            .contains("truncated"));
    }
    #[test]
    fn modeled_hole_retains_its_inner_boundary() {
        let points = (0..9)
            .flat_map(|j| (0..9).map(move |i| [i as f64, j as f64]))
            .collect::<Vec<_>>();
        let lfn = points
            .iter()
            .map(|p| {
                let d = (p[0] - 4.).abs().max((p[1] - 4.).abs());
                (d - 3.).max(1. - d)
            })
            .collect::<Vec<_>>();
        let (a, e) = model_geometry(&points, &lfn, [9, 9]).unwrap();
        assert_eq!(a, 32.);
        assert_eq!(e.iter().map(length).sum::<f64>(), 32.);
    }
    #[test]
    fn closest_actual_timestamp_uses_earlier_tie_and_fractional_utc() {
        assert_eq!(nearest(&[0, 1000, 2000], 1500), 1);
        assert_eq!(epoch_ms("2026-07-16_00:12:31").unwrap(), 1784160751000);
        assert_eq!(epoch_ms("2026-07-16T00:12:31.719Z").unwrap(), 1784160751719);
    }
    #[test]
    fn spatial_tree_matches_direct_segment_distances() {
        let all = (0..70)
            .map(|i| {
                [
                    [i as f64, (i % 7) as f64],
                    [i as f64 + 0.7, (i % 7) as f64 + 2.],
                ]
            })
            .collect::<Vec<_>>();
        let tree = Tree::new(all.clone());
        for i in 0..150 {
            let p = [i as f64 * 0.5, -1.3];
            let direct = all
                .iter()
                .map(|s| point_distance_squared(p, s))
                .fold(f64::INFINITY, f64::min)
                .sqrt();
            assert_eq!(tree.distance(p), direct);
        }
    }
    #[test]
    fn native_classic_history_roundtrip_preserves_time_and_distinct_area_measures() {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let folder = std::env::temp_dir().join(format!(
            "sfire-observation-comparison-{}",
            std::process::id()
        ));
        std::fs::create_dir(&folder).unwrap();
        let history = folder.join("native-history.nc");
        let observed = folder.join("observed.geojson");
        let contract = folder.join("source-contract.json");
        let mut x = (0..5)
            .flat_map(|_| (0..5).map(|i| -1800000. + i as f64 * 100.))
            .collect::<Vec<_>>();
        let mut y = (0..5)
            .flat_map(|j| (0..5).map(move |_| 1400000. + j as f64 * 100.))
            .collect::<Vec<_>>();
        transform_points(&crs(), &Crs::Geographic, &mut x, &mut y).unwrap();
        let observed_ring = [6, 8, 18, 16, 6].map(|id| vec![x[id], y[id]]);
        let data = json!({"type":"FeatureCollection","features":[{"type":"Feature","properties":{"observed_ms":1784160750719_i64},"geometry":{"type":"Polygon","coordinates":[observed_ring]}}]});
        let bytes = serde_json::to_vec(&data).unwrap();
        std::fs::write(&observed, &bytes).unwrap();
        let source = json!({"input_sha256":digest(&bytes),"time_property":"observed_ms","time_format":"unix_ms","coordinate_reference_system":"EPSG:4326"});
        std::fs::write(&contract, serde_json::to_vec(&source).unwrap()).unwrap();
        let mut schema = Schema::new(NcFormat::Offset64);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let text = schema.def_dim("DateStrLen", 19, false).unwrap();
        let ny = schema.def_dim("south_north_subgrid", 5, false).unwrap();
        let nx = schema.def_dim("west_east_subgrid", 5, false).unwrap();
        for (name, value) in [("DX", 500.), ("DY", 500.), ("SR_X", 5.), ("SR_Y", 5.)] {
            schema
                .put_global_attr(name, AttrValue::Doubles(vec![value]))
                .unwrap();
        }
        schema
            .put_global_attr("FIRE_COORDINATE_MODE", AttrValue::Text("geographic".into()))
            .unwrap();
        let stamp = schema
            .def_var("Times", NcType::Char, &[time, text])
            .unwrap();
        let vars = ["FXLONG", "FXLAT", "LFN", "FIRE_AREA"].map(|name| {
            schema
                .def_var(name, NcType::Double, &[time, ny, nx])
                .unwrap()
        });
        let lfn = (0..5)
            .flat_map(|j| {
                (0..5).map(move |i| ((i as f64 - 2.).abs().max((j as f64 - 2.).abs()) - 1.) * 100.)
            })
            .collect::<Vec<_>>();
        let fractions = lfn
            .iter()
            .map(|v| if *v <= 0. { 1. } else { 0. })
            .collect::<Vec<_>>();
        let mut writer = NcWriter::create(&history, schema).unwrap();
        for (record, stamp_text) in [b"2026-07-16_00:12:00", b"2026-07-16_00:13:00"]
            .iter()
            .enumerate()
        {
            writer
                .write_record(record as u64, stamp, VarData::Char(*stamp_text))
                .unwrap();
            for (var, data) in vars.iter().zip([&x, &y, &lfn, &fractions]) {
                writer
                    .write_record(record as u64, *var, VarData::F64(data))
                    .unwrap();
            }
        }
        writer.finish().unwrap();
        let opts = || Options {
            observations: observed.clone(),
            contract: contract.clone(),
            histories: vec![history.clone()],
            output: folder.join("comparison.json"),
            spacing: 5.,
            max_delta: 60.,
        };
        let result = run(opts()).unwrap();
        let record = &result["comparisons"][0];
        assert_eq!(record["history_record"], 1);
        assert!((record["model_minus_observation_s"].as_f64().unwrap() - 29.281).abs() < 1e-12);
        assert_eq!(record["modeled_native_FIRE_AREA_m2"], 90000.);
        assert!((record["modeled_projected_LFN_area_m2"].as_f64().unwrap() - 40000.).abs() < 1e-3);
        assert!(
            record["perimeter_distance"]["hausdorff_sampled_m"]
                .as_f64()
                .unwrap()
                < 1e-7
        );
        let mut narrow = opts();
        narrow.max_delta = 20.;
        assert!(run(narrow)
            .unwrap_err()
            .contains("exceeding explicit maximum"));
        let mut bad = source;
        bad["input_sha256"] = json!("wrong");
        std::fs::write(&contract, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(run(opts()).unwrap_err().contains("source hash differs"));
        for path in [&history, &observed, &contract] {
            let size = std::fs::metadata(path).unwrap().len();
            std::fs::remove_file(path).unwrap();
            println!(
                "deleted owned test fixture {} ({size} bytes)",
                path.display()
            );
        }
        std::fs::remove_dir(&folder).unwrap();
    }
}
