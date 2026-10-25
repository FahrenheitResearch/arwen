//! Original Apache-2.0 NextGen output products. No UCAR implementation or
//! fixture is used. HAND definition: Renno et al. (2008),
//! doi:10.1016/j.rse.2008.03.018. All numeric work is output-cadence Rust/Rayon.
use crate::panel::{layout_path, render_panel_cells, PanelRequest};
use rayon::prelude::*;
use rustwx_render::{
    Color, ColorScale, DiscreteColorScale, ExtendMode, LegendControls, LegendMode,
};
use serde::Deserialize;
use std::{
    collections::HashSet,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

pub const ABI: &str =
    "gpuwm.nextgen-products.v1 node-map HAND terrain-depth ensemble-depth hydrograph component-series streamflow-map";

/// Preserve prior products on ordinary repeated requests, including symlink
/// destinations. A fresh sibling is claimed with create_dir, not a name probe.
fn output_directory(requested: &Path) -> Result<PathBuf, String> {
    match std::fs::symlink_metadata(requested) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(requested).map_err(|e| e.to_string())?;
            return Ok(requested.to_path_buf());
        }
        Err(e) => return Err(e.to_string()),
        Ok(meta) if meta.is_dir() && !meta.file_type().is_symlink() => {
            if std::fs::read_dir(requested)
                .map_err(|e| e.to_string())?
                .next()
                .transpose()
                .map_err(|e| e.to_string())?
                .is_none()
            {
                return Ok(requested.to_path_buf());
            }
        }
        Ok(_) => {}
    }
    // Resolve ordinary directories so requests ending in '.' or '..' choose
    // actual siblings. Keep symlink names intact and never write through them.
    let anchor = if requested.is_dir()
        && !std::fs::symlink_metadata(requested)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
    {
        std::fs::canonicalize(requested).map_err(|e| e.to_string())?
    } else {
        requested.to_path_buf()
    };
    let parent = anchor.parent().unwrap_or(&anchor);
    let name = anchor.file_name().unwrap_or_else(|| std::ffi::OsStr::new("products"));
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    for attempt in 0..1000 {
        let mut sibling = name.to_os_string();
        sibling.push(format!("-hydro-products-{stamp}-{}-{attempt}", std::process::id()));
        let out = parent.join(sibling);
        match std::fs::create_dir(&out) {
            Ok(()) => {
                eprintln!("NextGen products: preserving {}; writing {}", requested.display(), out.display());
                return Ok(out);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    Err("could not claim a fresh products directory".into())
}

#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct Buffer {
    path: PathBuf,
    /// Raw <f4 (default), <f8, <i4, <i8, or a NetCDF variable.
    dtype: Option<String>,
    variable: Option<String>,
    record: Option<usize>,
    /// Explicit little-endian i64 gather indices, including padded layouts.
    indices: Option<PathBuf>,
    missing: Option<f64>,
}
impl Buffer {
    fn values(&self, root: &Path) -> Result<Vec<f64>, String> {
        let data = if let Some(variable) = &self.variable {
            let f = netcrust::open(root.join(&self.path)).map_err(|e| e.to_string())?;
            if let Some(record) = self.record {
                // The request explicitly selects axis zero, including ordinary
                // hydro [time, node] data. The WRF convenience reader ignores
                // records on arrays with fewer than three dimensions.
                let shape = f
                    .variable(variable)
                    .ok_or_else(|| format!("missing NetCDF variable metadata: {variable}"))?
                    .shape();
                if shape.first().is_none_or(|&count| record >= count) {
                    return Err(format!(
                        "NetCDF {variable} shape {shape:?} has no leading-axis record {record}"
                    ));
                }
                let mut selections = vec![netcrust::NcSliceInfoElem::Index(record as u64)];
                selections.extend((1..shape.len()).map(|_| netcrust::NcSliceInfoElem::Slice {
                    start: 0,
                    end: u64::MAX,
                    step: 1,
                }));
                f.read_array_f64_slice(variable, &netcrust::NcSliceInfo { selections })
            } else {
                f.read_array_f64(variable)
            }
            .map_err(|e| e.to_string())?
            .values()
            .to_vec()
        } else {
            let bytes = std::fs::read(root.join(&self.path)).map_err(|e| e.to_string())?;
            let kind = self.dtype.as_deref().unwrap_or("<f4");
            let size = match kind {
                "<i4" | "<f4" => 4,
                "<i8" | "<f8" => 8,
                _ => return Err(format!("unknown product buffer dtype {kind}")),
            };
            if bytes.len() % size != 0 {
                return Err("partial product buffer word".into());
            }
            bytes
                .par_chunks_exact(size)
                .map(|b| match kind {
                    "<i4" => i32::from_le_bytes(b.try_into().unwrap()) as f64,
                    "<i8" => i64::from_le_bytes(b.try_into().unwrap()) as f64,
                    "<f4" => f32::from_le_bytes(b.try_into().unwrap()) as f64,
                    _ => f64::from_le_bytes(b.try_into().unwrap()),
                })
                .collect::<Vec<_>>()
        };
        if let Some(indices) = &self.indices {
            let raw = std::fs::read(root.join(indices)).map_err(|e| e.to_string())?;
            if raw.len() % 8 != 0 {
                return Err("partial product gather index".into());
            }
            raw.par_chunks_exact(8)
                .map(|b| {
                    let i = i64::from_le_bytes(b.try_into().unwrap());
                    if i < 0 {
                        return Err("negative product gather index".into());
                    }
                    data.get(i as usize)
                        .copied()
                        .ok_or("product gather index exceeds buffer".into())
                })
                .collect()
        } else {
            Ok(data)
        }
    }
    fn numeric(&self, root: &Path, n: usize) -> Result<Vec<f64>, String> {
        let v = self.values(root)?;
        if v.len() != n {
            return Err(format!("{} has {} words, expected {n}; supply an explicit indices map for padded/node layouts", self.path.display(), v.len()));
        }
        Ok(v.into_par_iter()
            .map(|v| if Some(v) == self.missing { f64::NAN } else { v })
            .collect())
    }
    fn floats(&self, root: &Path, n: usize) -> Result<Vec<f32>, String> {
        Ok(self
            .numeric(root, n)?
            .into_par_iter()
            .map(|v| v as f32)
            .collect())
    }
    fn integers(&self, root: &Path, n: usize) -> Result<Vec<i64>, String> {
        let v = self.values(root)?;
        if v.len() != n {
            return Err("product topology/map length differs".into());
        }
        v.into_par_iter()
            .map(|v| {
                if !v.is_finite() || v.fract() != 0. || v.abs() > 9007199254740991. {
                    Err("product topology index must be an exactly represented integer".into())
                } else {
                    Ok(v as i64)
                }
            })
            .collect()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Terrain {
    nx: usize,
    ny: usize,
    latitude: Buffer,
    longitude: Buffer,
    elevation_m: Option<Buffer>,
    downstream_cell: Option<Buffer>,
    /// Zero-based node index at drainage cells; -1 elsewhere.
    channel_node: Option<Buffer>,
    stream_order: Option<Buffer>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {
    valid_time: String,
    /// Engineering clocks may label elapsed time without implying an observed event.
    time_label: Option<String>,
    discharge_m3_s: Option<Buffer>,
    reference_m3_s: Option<Buffer>,
    observed_m3_s: Option<Buffer>,
    /// Authoritative dynamic terrain-2d depth, grid order, in metres.
    terrain_depth_m: Option<Buffer>,
    /// Optional HAND estimate uses above-bed channel depth in node order.
    channel_depth_m: Option<Buffer>,
    #[serde(default)]
    member_terrain_depth_m: Vec<Buffer>,
    #[serde(default)]
    member_channel_depth_m: Vec<Buffer>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    out_dir: PathBuf,
    domain: String,
    source_label: String,
    #[serde(default)]
    node_ids: Vec<String>,
    #[serde(default)]
    gauges: Vec<String>,
    terrain: Option<Terrain>,
    frames: Vec<Frame>,
    #[serde(default = "threshold")]
    flood_threshold_m: f32,
    #[serde(default = "depth_scale")]
    depth_scale_max_m: f32,
}
fn depth_scale() -> f32 {
    2.0
}

struct TerrainData {
    lat: Vec<f32>,
    lon: Vec<f32>,
    channel: Option<Vec<i64>>,
    drainage: Option<(Vec<f32>, Vec<i64>, Vec<i64>, Vec<f32>)>,
}

fn threshold() -> f32 {
    0.1
}

/// Parallel pointer jumping to the first drainage cell on the supplied flow
/// graph. No geometric nearest-channel approximation and no serial grid walk.
fn height_above_drainage(
    z: &[f32],
    next: &[i64],
    channel: &[i64],
) -> Result<(Vec<f32>, Vec<Option<usize>>), String> {
    let n = z.len();
    if next.len() != n || channel.len() != n || next.iter().any(|&j| j < -1 || j >= n as i64) {
        return Err("HAND downstream topology differs from DEM".into());
    }
    let mut jump: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !z[i].is_finite() {
                -1
            } else if channel[i] >= 0 {
                i as i64
            } else {
                next[i]
            }
        })
        .collect();
    let mut anchor: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| (z[i].is_finite() && channel[i] >= 0).then_some(i))
        .collect();
    let rounds = usize::BITS - n.max(1).leading_zeros() + 1;
    for _ in 0..rounds {
        let updates: Vec<_> = (0..n)
            .into_par_iter()
            .map(|i| {
                let j = jump[i];
                if anchor[i].is_some() || j < 0 {
                    (jump[i], anchor[i])
                } else {
                    (jump[j as usize], anchor[j as usize])
                }
            })
            .collect();
        (jump, anchor) = updates.into_par_iter().unzip();
    }
    if (0..n)
        .into_par_iter()
        .any(|i| z[i].is_finite() && anchor[i].is_none() && jump[i] >= 0)
    {
        return Err("channel-free cycle in HAND drainage graph".into());
    }
    let h = (0..n)
        .into_par_iter()
        .map(|i| {
            if z[i].is_finite() {
                anchor[i].map_or(f32::NAN, |j| (z[i] - z[j]).max(0.))
            } else {
                f32::NAN
            }
        })
        .collect();
    Ok((h, anchor))
}
fn exceedance(members: &[Vec<f32>], n: usize, threshold: f32) -> Vec<f32> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut wet = 0usize;
            for member in members {
                let d = member[i];
                if !d.is_finite() || d < 0. {
                    return f32::NAN;
                }
                wet += usize::from(d > threshold);
            }
            wet as f32 / members.len() as f32
        })
        .collect()
}
fn words(path: &Path, v: &[f32]) -> Result<(), String> {
    let b: Vec<_> = v.par_iter().map(|v| v.to_le_bytes()).collect();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
    for b in b {
        f.write_all(&b).map_err(|e| e.to_string())?;
    }
    f.flush().map_err(|e| e.to_string())
}

struct ChartMetadata<'a> {
    title: &'a str,
    source: &'a str,
    first_label: &'a str,
    last_label: &'a str,
    axis_label: &'a str,
    zero_based: bool,
    nonnegative: bool,
}

fn chart(
    path: &Path,
    metadata: ChartMetadata<'_>,
    positions: &[f64],
    series: &[(&str, Vec<f32>)],
) -> Result<(), String> {
    use image::{Rgba, RgbaImage};
    let mut image = RgbaImage::from_pixel(1200, 640, Rgba([250, 251, 253, 255]));
    let valid = |v: f32| v.is_finite() && (!metadata.nonnegative || v >= 0.);
    let extrema = series
        .iter()
        .flat_map(|(_, v)| v)
        .copied()
        .filter(|&v| valid(v))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    let (lower, upper) = if !extrema.0.is_finite() {
        (0., 1.)
    } else if metadata.zero_based {
        (
            extrema.0.min(0.),
            if extrema.1 > 0. { extrema.1 * 1.1 } else { 1. },
        )
    } else {
        let margin = ((extrema.1 - extrema.0) * 0.1).max(extrema.1.abs().max(1.) * 1e-6);
        (extrema.0 - margin, extrema.1 + margin)
    };
    let first = positions[0];
    let span = (positions[positions.len() - 1] - first).max(f64::EPSILON);
    let point = |i: usize, v: f32| {
        (
            100 + ((positions[i] - first) / span * 1040.).round() as i32,
            510 - ((v - lower) / (upper - lower) * 390.).round() as i32,
        )
    };
    let line = |img: &mut RgbaImage, a: (i32, i32), b: (i32, i32), c: [u8; 4]| {
        let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
        for k in 0..=steps {
            let x = a.0 + (b.0 - a.0) * k / steps;
            let y = a.1 + (b.1 - a.1) * k / steps;
            if x >= 0 && y >= 0 && x < 1200 && y < 640 {
                img.put_pixel(x as u32, y as u32, Rgba(c));
            }
        }
    };
    let precision = (-((upper - lower) / 5.).log10()).ceil().clamp(3., 8.) as usize;
    for i in 0..=5 {
        let y = 510 - i * 78;
        line(&mut image, (100, y), (1140, y), [210, 215, 225, 255]);
        let tick = lower + (upper - lower) * i as f32 / 5.;
        let label = if tick != 0. && (tick.abs() < 0.01 || tick.abs() >= 10000.) {
            format!("{tick:.3e}")
        } else {
            format!("{tick:.precision$}")
        };
        rustwx_render::draw_text_line(&mut image, &label, 4, y - 8, Color::BLACK, 3);
    }
    let mut legend = Vec::new();
    for (name, values) in series {
        let (c, r) = match *name {
            "USGS" => ([185, 50, 80, 255], 7),
            "referee" => ([235, 125, 20, 255], 5),
            _ => ([25, 80, 185, 255], 2),
        };
        if !values.iter().any(|&v| valid(v)) {
            continue;
        }
        legend.push(format!(
            "{name} {}",
            if *name == "USGS" {
                "red"
            } else if *name == "referee" {
                "orange"
            } else {
                "blue"
            }
        ));
        for i in 0..values.len() {
            if !valid(values[i]) {
                continue;
            }
            let p = point(i, values[i]);
            if i > 0 && valid(values[i - 1]) {
                line(&mut image, point(i - 1, values[i - 1]), p, c);
            }
            for dx in -r..=r {
                for dy in -r..=r {
                    line(&mut image, (p.0 + dx, p.1 + dy), (p.0 + dx, p.1 + dy), c);
                }
            }
        }
    }
    rustwx_render::draw_text_line(&mut image, metadata.title, 100, 20, Color::BLACK, 6);
    rustwx_render::draw_text_line(&mut image, &legend.join(" | "), 100, 65, Color::BLACK, 3);
    rustwx_render::draw_text_line(&mut image, metadata.source, 100, 92, Color::BLACK, 3);
    // Anchor endpoint labels to their samples. Right alignment keeps a full
    // timestamp inside the plot; ticks make the exact sample positions clear.
    for (label, anchor, right_aligned) in [
        (metadata.first_label, 100, false),
        (metadata.last_label, 1140, true),
    ] {
        line(&mut image, (anchor, 515), (anchor, 525), [80, 85, 95, 255]);
        let x = if right_aligned {
            anchor - rustwx_render::text_width(label, 3) as i32
        } else {
            anchor
        };
        rustwx_render::draw_text_line(&mut image, label, x, 535, Color::BLACK, 3);
    }
    rustwx_render::draw_text_line(&mut image, metadata.axis_label, 100, 590, Color::BLACK, 3);
    image.save(path).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentSeriesRequest {
    schema: String,
    out_dir: PathBuf,
    domain: String,
    source_label: String,
    charts: Vec<ComponentChart>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentChart {
    id: String,
    quantity: String,
    units: String,
    /// Path label only; "source-time" is appropriate for an unverified timezone.
    clock_label: String,
    axis_label: String,
    positions: Vec<f64>,
    sample_labels: Vec<String>,
    #[serde(default)]
    zero_based: bool,
    series: Vec<ComponentTrace>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentTrace {
    label: String,
    values: Buffer,
}

fn component_series(root: &Path, r: ComponentSeriesRequest) -> Result<(), String> {
    let wall = Instant::now();
    if r.schema != "gpuwm.component-series.v1" || r.charts.is_empty() {
        return Err("empty/invalid component series request".into());
    }
    let mut paths = HashSet::new();
    for c in &r.charts {
        if c.id.is_empty()
            || c.quantity.is_empty()
            || c.units.is_empty()
            || c.axis_label.is_empty()
            || c.clock_label.is_empty()
            || c.positions.is_empty()
            || c.positions.len() != c.sample_labels.len()
            || c.positions.iter().any(|v| !v.is_finite())
            || c.positions.windows(2).any(|v| v[1] <= v[0])
            || c.series.is_empty()
        {
            return Err(format!("invalid component chart metadata: {}", c.id));
        }
        let p = layout_path(
            &root.join(&r.out_dir),
            &r.domain,
            "component_series",
            &c.clock_label,
            &c.id,
        );
        if !paths.insert(p) {
            return Err("component chart paths collide after sanitizing IDs".into());
        }
        let labels: HashSet<_> = c.series.iter().map(|s| &s.label).collect();
        if labels.len() != c.series.len() || labels.iter().any(|s| s.is_empty()) {
            return Err("component series labels must be nonempty and unique".into());
        }
    }
    let requested_out = root.join(&r.out_dir);
    let out = output_directory(&requested_out)?;
    let charts = r.charts.par_iter().map(|c| -> Result<_, String> {
        let start = Instant::now();
        let values = c.series.par_iter().map(|s| s.values.numeric(root, c.positions.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let load_ms = start.elapsed().as_secs_f64() * 1000.;
        let model = c.series.iter().position(|s| s.label == "model");
        let reference = c.series.iter().position(|s| s.label == "referee");
        let (mut compared, mut different) = (0usize, 0usize);
        if let (Some(m), Some(n)) = (model, reference) {
            for (&a, &b) in values[m].iter().zip(&values[n]) {
                if a.is_finite() && b.is_finite() {
                    compared += 1;
                    different += usize::from(a.to_bits() != b.to_bits());
                }
            }
        }
        let p = layout_path(&out, &r.domain, "component_series", &c.clock_label, &c.id);
        std::fs::create_dir_all(p.parent().unwrap()).map_err(|e| e.to_string())?;
        let start = Instant::now();
        let display: Vec<_> = c.series.iter().zip(&values)
            .map(|(s, v)| (s.label.as_str(), v.iter().map(|&v| v as f32).collect())).collect();
        chart(&p, ChartMetadata {
            title: &format!("{} ({})", c.quantity, c.units),
            source: &r.source_label,
            first_label: &c.sample_labels[0],
            last_label: &c.sample_labels[c.sample_labels.len() - 1],
            axis_label: &c.axis_label,
            zero_based: c.zero_based,
            nonnegative: false,
        }, &c.positions, &display)?;
        let render_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let csv = p.with_extension("csv");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&csv).map_err(|e| e.to_string())?);
        writeln!(file, "position,sample_label,series,value").map_err(|e| e.to_string())?;
        let quote = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
        for (j, trace) in c.series.iter().enumerate() {
            for (i, &v) in values[j].iter().enumerate() {
                let text = if v.is_finite() { v.to_string() } else { String::new() };
                writeln!(file, "{},{},{},{}", c.positions[i], quote(&c.sample_labels[i]), quote(&trace.label), text)
                    .map_err(|e| e.to_string())?;
            }
        }
        file.flush().map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"id":c.id,"quantity":c.quantity,"units":c.units,
            "axis_label":c.axis_label,"sample_count":c.positions.len(),"outputs":[p,csv],
            "comparison":{"compared_finite_f64_words":compared,"different_words":different},
            "timings_ms":{"load":load_ms,"render":render_ms,"csv":start.elapsed().as_secs_f64()*1000.}}))
    }).collect::<Result<Vec<_>, _>>()?;
    let receipt = serde_json::json!({"schema":"gpuwm.component-series.receipt.v1",
        "requested_output_directory":requested_out,"output_directory":out,
        "renderer":"rw_wrfbatch","source_label":r.source_label,"source_licence":"original Apache-2.0",
        "charts":charts,"parallelism":format!("{} Rayon workers; charts and traces parallel", rayon::current_num_threads()),
        "wall_ms":wall.elapsed().as_secs_f64()*1000.,"full_precision_csv_dtype":"<f8",
        "time_policy":"positions and labels supplied explicitly; no timezone inferred",
        "routing_step_host_work":"none; output boundary only","graphs":false});
    let p = out.join("nextgen-products-receipt.json");
    std::fs::write(&p, serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
    println!("NEXTGEN_RECEIPT\t{}", p.display());
    Ok(())
}

pub fn run(path: &Path) -> Result<(), String> {
    let wall = Instant::now();
    let root = path.parent().unwrap_or(Path::new("."));
    let request: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if request["schema"] == "gpuwm.component-series.v1" {
        return component_series(
            root,
            serde_json::from_value(request).map_err(|e| e.to_string())?,
        );
    }
    let r: Request = serde_json::from_value(request).map_err(|e| e.to_string())?;
    if r.schema != "gpuwm.nextgen-products.v1"
        || r.frames.is_empty()
        || !r.flood_threshold_m.is_finite()
        || r.flood_threshold_m < 0.
        || !r.depth_scale_max_m.is_finite()
        || r.depth_scale_max_m <= 0.
    {
        return Err("invalid NextGen products request".into());
    }
    let unique: HashSet<_> = r.node_ids.iter().collect();
    if unique.len() != r.node_ids.len()
        || r.node_ids
            .iter()
            .any(|id| id.is_empty() || id.contains(['\n', '\r']))
    {
        return Err("product node IDs must be unique nonempty single-line IDs".into());
    }
    let stamps = r
        .frames
        .iter()
        .map(|f| {
            chrono::DateTime::parse_from_rfc3339(&f.valid_time)
                .map_err(|e| e.to_string())
                .and_then(|t| {
                    if t.offset().local_minus_utc() == 0 {
                        Ok(t.timestamp())
                    } else {
                        Err("product valid time must be UTC".into())
                    }
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if stamps.windows(2).any(|s| s[1] <= s[0]) {
        return Err("product valid times must increase".into());
    }
    let requested_out = root.join(&r.out_dir);
    let out = output_directory(&requested_out)?;
    let mut timings = Vec::new();
    let mut outputs = Vec::new();
    let mut record = |stage: &str, start: Instant| {
        timings.push(serde_json::json!({"stage":stage,"wall_ms":start.elapsed().as_secs_f64()*1000.,"parallelism":format!("{} Rayon workers",rayon::current_num_threads())}));
    };
    let start = Instant::now();
    let terrain = r
        .terrain
        .as_ref()
        .map(|t| {
            let n =
                t.nx.checked_mul(t.ny)
                    .filter(|&n| n > 0)
                    .ok_or("empty/overflow terrain")?;
            let lat = t.latitude.floats(root, n)?;
            let lon = t.longitude.floats(root, n)?;
            let channel = t.channel_node.as_ref().map(|b| b.integers(root, n)).transpose()?;
            if channel.as_ref().is_some_and(|c| c.par_iter().any(|&i| i < -1 || i >= r.node_ids.len() as i64)) {
                return Err("terrain channel_node exceeds explicit node roster".into());
            }
            let drainage = match (&t.elevation_m, &t.downstream_cell, &channel, &t.stream_order) {
                (None, None, _, None) => None,
                (Some(z), Some(next), Some(channel), Some(order)) => {
                    let z = z.floats(root, n)?;
                    let next = next.integers(root, n)?;
                    let order = order.floats(root, n)?.into_par_iter().enumerate().map(|(i,v)| {
                        if channel[i] >= 0 && v.is_finite() && v > 0. && v.fract() == 0. {v} else {f32::NAN}
                    }).collect();
                    Some((z, next, channel.clone(), order))
                }
                _ => return Err("HAND requires elevation_m, downstream_cell, channel_node and stream_order together; omit all for depth-only maps".into()),
            };
            Ok::<_, String>(TerrainData { lat, lon, channel, drainage })
        })
        .transpose()?;
    record("load_static", start);
    let start = Instant::now();
    let hand = terrain
        .as_ref()
        .and_then(|t| t.drainage.as_ref())
        .map(|(z, next, c, _)| height_above_drainage(z, next, c))
        .transpose()?;
    record("HAND", start);
    let render = |slug: &str,
                  time: &str,
                  values: &[f32],
                  units: &str,
                  title: &str|
     -> Result<Vec<PathBuf>, String> {
        let t = r
            .terrain
            .as_ref()
            .ok_or("map requires terrain coordinates")?;
        let data = terrain.as_ref().unwrap();
        let (lat, lon) = (&data.lat, &data.lon);
        let upper = match slug {
            "streamflow_map" => 100_000.,
            "flood_probability" | "hand_flood_probability" | "inundation_extent" => 1.,
            "stream_order" => values
                .par_iter()
                .copied()
                .filter(|v| v.is_finite())
                .reduce(|| 1., f32::max),
            "HAND" => 100.,
            _ => r.depth_scale_max_m,
        };
        let bins = if slug == "stream_order" {
            upper as usize
        } else if slug == "inundation_extent" {
            2
        } else {
            10
        };
        let levels = if slug == "streamflow_map" {
            // Stable physical thresholds reveal tributaries beside continental
            // outlets; a per-frame linear maximum hid almost all inland flow.
            vec![0., 0.1, 1., 10., 100., 300., 1_000., 3_000., 10_000., 30_000., 100_000.]
        } else if slug == "stream_order" {
            (0..=bins).map(|i| i as f64 + 0.5).collect()
        } else if slug == "inundation_extent" {
            vec![-0.5, 0.5, 1.5]
        } else {
            (0..=bins)
                .map(|i| upper as f64 * i as f64 / bins as f64)
                .collect()
        };
        let colors = (0..bins)
            .map(|i| {
                let s = (i * 9 / bins.saturating_sub(1).max(1)) as u8;
                Color::rgba(245 - s * 24, 250 - s * 17, 255 - s * 8, 255)
            })
            .collect();
        let scale = ColorScale::Discrete(DiscreteColorScale {
            levels,
            colors,
            extend: ExtendMode::Max,
            mask_below: None,
        });
        let p = layout_path(
            &out,
            &r.domain,
            slug,
            &time[..10],
            &format!("{slug}_{time}"),
        );
        render_panel_cells(PanelRequest {
            lat_deg: lat,
            lon_deg: lon,
            projection: None,
            ny: t.ny,
            nx: t.nx,
            values: values.to_vec(),
            product_slug: slug.into(),
            title: if slug == "streamflow_map" {
                format!("{title} ({units}); fixed discharge thresholds")
            } else {
                format!("{title} ({units}); top color >= {upper}")
            },
            display_units: units.into(),
            scale,
            cbar_tick_step: if matches!(slug, "stream_order" | "inundation_extent") {
                Some(1.)
            } else {
                None
            },
            legend: LegendControls {
                mode: if slug == "streamflow_map" {
                    LegendMode::Thresholds
                } else if matches!(slug, "stream_order" | "inundation_extent") {
                    LegendMode::Categories
                } else {
                    LegendMode::Stepped
                },
                ..Default::default()
            },
            render_density: Default::default(),
            subtitle_left: r
                .frames
                .iter()
                .find(|f| f.valid_time == time)
                .and_then(|f| f.time_label.clone())
                .unwrap_or_else(|| time.into()),
            subtitle_center: None,
            subtitle_right: r.source_label.clone(),
            width: 1200,
            height: 900,
            contours: vec![],
            colorbar: true,
            overlays: None,
            annotations: None,
            out_path: p.clone(),
        })?;
        let raw = p.with_extension("f32le");
        words(&raw, values)?;
        Ok(vec![p, raw])
    };
    let start = Instant::now();
    if let (Some((_, _, _, order)), Some((h, _))) =
        (terrain.as_ref().and_then(|t| t.drainage.as_ref()), &hand)
    {
        for (slug, v, u, title) in [
            ("HAND", h, "m", "Height above first downstream drainage"),
            ("stream_order", order, "1", "Stream order"),
        ] {
            outputs.extend(render(slug, &r.frames[0].valid_time, v, u, title)?);
        }
    }
    record("render_static", start);
    let n = r.node_ids.len();
    let mut rows = Vec::new();
    let mut differences = 0usize;
    let mut compared = 0usize;
    for f in &r.frames {
        let start = Instant::now();
        let q = f
            .discharge_m3_s
            .as_ref()
            .map(|b| b.numeric(root, n))
            .transpose()?
            .unwrap_or_else(|| vec![f64::NAN; n]);
        if f.discharge_m3_s.is_some() {
            let qp = layout_path(
                &out,
                &r.domain,
                "streamflow",
                &f.valid_time[..10],
                &format!("streamflow_{}", f.valid_time),
            )
            .with_extension("f32le");
            std::fs::create_dir_all(qp.parent().unwrap()).map_err(|e| e.to_string())?;
            let display: Vec<_> = q.par_iter().map(|&v| v as f32).collect();
            words(&qp, &display)?;
            let full = qp.with_extension("f64le");
            let mut file =
                std::io::BufWriter::new(std::fs::File::create(&full).map_err(|e| e.to_string())?);
            for value in &q {
                file.write_all(&value.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
            file.flush().map_err(|e| e.to_string())?;
            outputs.extend([qp, full]);
        }
        let optional = |b: &Option<Buffer>| {
            b.as_ref()
                .map(|b| b.numeric(root, n))
                .transpose()
                .map(|v| v.unwrap_or_else(|| vec![f64::NAN; n]))
        };
        let reference = optional(&f.reference_m3_s)?;
        let obs = optional(&f.observed_m3_s)?;
        let check = q
            .par_iter()
            .zip(&reference)
            .filter(|(_, b)| b.is_finite())
            .map(|(a, b)| (1usize, usize::from(a.to_bits() != b.to_bits())))
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        compared += check.0;
        differences += check.1;
        record("load_node_outputs", start);
        let start = Instant::now();
        let mut planes = Vec::new();
        if f.discharge_m3_s.is_some() {
            if let Some(channel) = terrain.as_ref().and_then(|t| t.channel.as_ref()) {
                planes.push(("streamflow_map", channel.par_iter().map(|&i| {
                    if i < 0 { f32::NAN } else { q[i as usize] as f32 }
                }).collect(), "m3 s-1", "Routed streamflow on prepared channel cells"));
            }
        }
        let cells = r.terrain.as_ref().map_or(0, |t| t.nx * t.ny);
        if let Some(b) = &f.terrain_depth_m {
            let d = b
                .floats(root, cells)?
                .into_par_iter()
                .map(|v| {
                    if v.is_finite() && v >= 0. {
                        v
                    } else {
                        f32::NAN
                    }
                })
                .collect::<Vec<_>>();
            planes.push((
                "inundation_depth",
                d.clone(),
                "m",
                "Dynamic terrain-2d inundation depth",
            ));
            planes.push((
                "inundation_extent",
                d.into_par_iter()
                    .map(|v| {
                        if v.is_finite() {
                            if v > 0. {
                                1.
                            } else {
                                0.
                            }
                        } else {
                            f32::NAN
                        }
                    })
                    .collect(),
                "1",
                "Dynamic terrain-2d inundation extent",
            ));
        }
        if let Some(b) = &f.channel_depth_m {
            let stage = b.floats(root, n)?;
            let (h, anchor) = hand
                .as_ref()
                .ok_or("HAND depth estimate requires terrain")?;
            let c = &terrain.as_ref().unwrap().drainage.as_ref().unwrap().2;
            let d = (0..cells)
                .into_par_iter()
                .map(|i| {
                    anchor[i].map_or(f32::NAN, |a| {
                        let s = stage[c[a] as usize];
                        if s >= 0. && s.is_finite() {
                            (s - h[i]).max(0.)
                        } else {
                            f32::NAN
                        }
                    })
                })
                .collect();
            planes.push((
                "hand_depth",
                d,
                "m",
                "HAND estimate from above-bed channel depth",
            ));
        }
        if !f.member_terrain_depth_m.is_empty() {
            if cells == 0 {
                return Err("ensemble terrain depth requires terrain grid".into());
            }
            let members = f
                .member_terrain_depth_m
                .par_iter()
                .map(|b| b.floats(root, cells))
                .collect::<Result<Vec<_>, _>>()?;
            planes.push((
                "flood_probability",
                exceedance(&members, cells, r.flood_threshold_m),
                "fraction",
                "Complete-roster terrain depth exceedance probability",
            ));
        }
        if !f.member_channel_depth_m.is_empty() {
            let (height, anchor) = hand
                .as_ref()
                .ok_or("channel ensemble requires HAND terrain")?;
            let channel = &terrain.as_ref().unwrap().drainage.as_ref().unwrap().2;
            let members = f
                .member_channel_depth_m
                .par_iter()
                .map(|b| b.floats(root, n))
                .collect::<Result<Vec<_>, _>>()?;
            let p = (0..cells)
                .into_par_iter()
                .map(|i| {
                    let Some(a) = anchor[i] else {
                        return f32::NAN;
                    };
                    let node = channel[a] as usize;
                    let mut count = 0usize;
                    for member in &members {
                        let stage = member[node];
                        if !stage.is_finite() || stage < 0. {
                            return f32::NAN;
                        }
                        count += usize::from((stage - height[i]).max(0.) > r.flood_threshold_m);
                    }
                    count as f32 / members.len() as f32
                })
                .collect();
            planes.push((
                "hand_flood_probability",
                p,
                "fraction",
                "Complete-roster HAND depth exceedance estimate",
            ));
        }
        record("depth_probability", start);
        let start = Instant::now();
        let paths = planes
            .par_iter()
            .map(|(slug, v, u, title)| {
                let title = if *slug == "flood_probability" {
                    format!(
                        "{title}; {} members; > {} m",
                        f.member_terrain_depth_m.len(),
                        r.flood_threshold_m
                    )
                } else if *slug == "hand_flood_probability" {
                    format!(
                        "{title}; {} members; > {} m",
                        f.member_channel_depth_m.len(),
                        r.flood_threshold_m
                    )
                } else {
                    (*title).to_string()
                };
                render(slug, &f.valid_time, v, u, &title)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for p in paths {
            outputs.extend(p);
        }
        rows.push((q, reference, obs));
        record("render_frame", start);
    }
    let start = Instant::now();
    let directory = out
        .join(crate::panel::safe_component(&r.domain, "domain"))
        .join("hydrograph")
        .join(&r.frames[0].valid_time[..10]);
    std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    let csv_rows = r
        .node_ids
        .par_iter()
        .enumerate()
        .map(|(i, id)| {
            let id = format!("\"{}\"", id.replace('"', "\"\""));
            let mut text = String::new();
            for (t, (q, reference, obs)) in rows.iter().enumerate() {
                let value = |v: f64| {
                    if v.is_finite() && v >= 0. {
                        v.to_string()
                    } else {
                        String::new()
                    }
                };
                text.push_str(&format!(
                    "{},{},{},{},{}\n",
                    r.frames[t].valid_time,
                    id,
                    value(q[i]),
                    value(reference[i]),
                    value(obs[i])
                ));
            }
            text
        })
        .collect::<Vec<_>>();
    if r.frames.iter().any(|f| {
        f.discharge_m3_s.is_some() || f.reference_m3_s.is_some() || f.observed_m3_s.is_some()
    }) {
        let csv = directory.join("streamflow.csv");
        let mut file = std::fs::File::create(&csv).map_err(|e| e.to_string())?;
        writeln!(
            file,
            "valid_time,node_id,discharge_m3_s,reference_m3_s,observed_m3_s"
        )
        .map_err(|e| e.to_string())?;
        for row in csv_rows {
            file.write_all(row.as_bytes()).map_err(|e| e.to_string())?;
        }
        outputs.push(csv);
    }
    let charts = r
        .gauges
        .par_iter()
        .map(|id| {
            let i = r
                .node_ids
                .iter()
                .position(|v| v == id)
                .ok_or_else(|| format!("gauge {id} absent from explicit node roster"))?;
            let p = directory.join(format!(
                "{i}-{}.png",
                crate::panel::safe_component(id, "gauge")
            ));
            chart(
                &p,
                ChartMetadata {
                    title: &format!("Streamflow: {id} (m3/s)"),
                    source: &r.source_label,
                    first_label: &chrono::DateTime::from_timestamp(stamps[0], 0)
                        .unwrap()
                        .to_rfc3339(),
                    last_label: &chrono::DateTime::from_timestamp(stamps[stamps.len() - 1], 0)
                        .unwrap()
                        .to_rfc3339(),
                    axis_label: "UTC; missing values break lines; only supplied times are samples",
                    zero_based: true,
                    nonnegative: true,
                },
                &stamps.iter().map(|&v| v as f64).collect::<Vec<_>>(),
                &[
                    ("USGS", rows.iter().map(|row| row.2[i] as f32).collect()),
                    ("referee", rows.iter().map(|row| row.1[i] as f32).collect()),
                    ("model", rows.iter().map(|row| row.0[i] as f32).collect()),
                ],
            )?;
            Ok(p)
        })
        .collect::<Result<Vec<_>, String>>()?;
    outputs.extend(charts);
    record("hydrographs_csv", start);
    let receipt = serde_json::json!({"schema":"gpuwm.nextgen-products.receipt.v1","requested_output_directory":requested_out,"output_directory":out,"renderer":"rw_wrfbatch","source_label":r.source_label,"source_licence":"original Apache-2.0","outputs":outputs,"node_ids":r.node_ids,"raw_dtype":"<f4","full_precision_streamflow_dtype":"<f8","comparison_dtype":"<f8 (f32 sources exactly promoted)","grid_shape":r.terrain.as_ref().map(|t|[t.ny,t.nx]),"timings":timings,"wall_ms":wall.elapsed().as_secs_f64()*1000.,"streamflow_comparison":{"compared_words":compared,"different_words":differences},"probability_threshold_m":r.flood_threshold_m,"probability_missing_policy":"any missing/negative member masks cell; fixed complete denominator","routing_step_host_work":"none; output boundary only","graphs":false});
    let p = out.join("nextgen-products-receipt.json");
    std::fs::write(&p, serde_json::to_vec_pretty(&receipt).unwrap()).map_err(|e| e.to_string())?;
    println!("NEXTGEN_RECEIPT\t{}", p.display());
    Ok(())
}
pub fn cli(args: &[String]) -> Option<Result<(), String>> {
    match args.first().map(String::as_str) {
        Some("--nextgen-products-abi") => Some({
            println!("{ABI}");
            Ok(())
        }),
        Some("--nextgen-products") => Some(
            args.get(1)
                .ok_or("--nextgen-products needs request JSON".into())
                .and_then(|p| run(Path::new(p))),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn connected_drainage_and_complete_member_threshold() {
        let (height, anchors) =
            height_above_drainage(&[9., 3., 11., 1.], &[1, -1, 3, -1], &[-1, 0, -1, 1]).unwrap();
        assert_eq!(height, vec![6., 0., 10., 0.]);
        assert_eq!(anchors, vec![Some(1), Some(1), Some(3), Some(3)]);
        assert!(height_above_drainage(&[1., 2.], &[1, 0], &[-1, -1]).is_err());
        let p = exceedance(&[vec![0., 0.2, f32::NAN], vec![0.1, 0.4, 0.]], 3, 0.1);
        assert_eq!(&p[..2], &[0., 1.]);
        assert!(p[2].is_nan());
    }
    #[test]
    fn explicit_node_map_rejects_implicit_truncation() {
        let dir = std::env::temp_dir().join(format!("nextgen-products-map-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        words(&dir.join("q.raw"), &[99., 7., 88., 3.]).unwrap();
        std::fs::write(
            dir.join("map.raw"),
            [3i64.to_le_bytes(), 1i64.to_le_bytes()].concat(),
        )
        .unwrap();
        let mut source: Buffer =
            serde_json::from_value(serde_json::json!({"path":"q.raw"})).unwrap();
        assert!(source.floats(&dir, 2).is_err());
        source.indices = Some("map.raw".into());
        assert_eq!(source.floats(&dir, 2).unwrap(), vec![3., 7.]);
        let native = 1.0f64 + 2.0f64.powi(-40);
        std::fs::write(dir.join("native.raw"), native.to_le_bytes()).unwrap();
        let source: Buffer =
            serde_json::from_value(serde_json::json!({"path":"native.raw","dtype":"<f8"})).unwrap();
        assert_eq!(
            source.numeric(&dir, 1).unwrap()[0].to_bits(),
            native.to_bits()
        );
        assert_ne!(
            (source.floats(&dir, 1).unwrap()[0] as f64).to_bits(),
            native.to_bits()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
