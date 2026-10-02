//! Close an export: every array's metadata, the coordinates, the group's
//! attributes, `.zmetadata` last; then `README.txt`, the receipt and, when
//! asked, the ZIP.

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::error::Result;
use crate::export::{self, is_accumulation, Progress, MASK_NAME};
use crate::{frame, grid};
use crate::ops::Op;
use crate::request::{Layout, LevelKind, Request, VariableKind, VariableRow};
use crate::state::{self, DomainState, State};
use crate::times;
use crate::zarr::{self, ArrayMeta, Dtype};
use crate::zipout;

pub struct Closed {
    pub bytes: u64,
    pub zip: Option<(PathBuf, u64)>,
}

pub const VERTICAL_INTERPOLATION: &str =
    "linear in ln(pressure) between the two model mass levels that bracket the level in each column";
pub const BELOW_GROUND_TEMPERATURE: &str = "below the lowest model level: the ECMWF rule (Trenberth, Berry and Buja 1993, NCAR/TN-396, equation 16), as NCL vinth2p_ecmwf and GeoCAT interp_hybrid_to_pressure(extrapolate=True) implement it; the below_ground mask flags points under the surface";
pub const BELOW_GROUND_GEOPOTENTIAL: &str = "below the lowest model level: the ECMWF rule (Trenberth, Berry and Buja 1993, NCAR/TN-396, equation 15), as NCL vinth2p_ecmwf and GeoCAT interp_hybrid_to_pressure(extrapolate=True) implement it; the below_ground mask flags points under the surface";
pub const BELOW_GROUND_LOWEST: &str =
    "below the lowest model level: the lowest model level's value; the below_ground mask flags points under the surface";
pub const MODEL_TOP_RULE: &str = "between the top model mass level and the model lid (P_TOP when the history file states it): geopotential linear in ln(pressure) up to the lid's own geopotential, every other field held at the top level's value; without P_TOP, geopotential extends hydrostatically at the top level's temperature; levels above the lid are left out of the dataset and listed in levels_dropped_above_model_top";

fn time_dims(layout: Layout) -> Vec<String> {
    match layout {
        Layout::Analysis => vec!["time".into()],
        Layout::Forecast => vec!["time".into(), "prediction_timedelta".into()],
    }
}

fn time_shape(layout: Layout, frames: usize) -> Vec<usize> {
    match layout {
        Layout::Analysis => vec![frames],
        Layout::Forecast => vec![1, frames],
    }
}

fn ones(n: usize) -> Vec<usize> {
    vec![1; n]
}

fn space_dims(domain: &DomainState) -> (String, String) {
    if domain.regrid.is_some() || (domain.native.regular.is_some() && domain.latlon_versions == 1) {
        ("latitude".into(), "longitude".into())
    } else {
        ("y".into(), "x".into())
    }
}

fn two_d_latlon(domain: &DomainState) -> bool {
    domain.regrid.is_none() && (domain.native.regular.is_none() || domain.latlon_versions > 1)
}

fn attrs_of(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// The `accumulation_period` text for a row, from the frames' spacing.
fn accumulation_period(row: &VariableRow, times: &[i64]) -> String {
    match Op::parse(&row.op) {
        Ok(Op::Accumulation(crate::ops::Period::Hours(h))) => format!("PT{h}H"),
        _ => {
            let steps: Vec<i64> = times.windows(2).map(|w| w[1] - w[0]).collect();
            match steps.first() {
                Some(&first) if steps.iter().all(|&s| s == first) => {
                    if first % 3600 == 0 {
                        format!("PT{}H", first / 3600)
                    } else if first % 60 == 0 {
                        format!("PT{}M", first / 60)
                    } else {
                        format!("PT{first}S")
                    }
                }
                Some(_) => "the interval since the previous time step (the output interval varies)".into(),
                None => "the output interval (one frame: no interval behind it)".into(),
            }
        }
    }
}

fn variable_attrs(
    row: &VariableRow,
    domain: &DomainState,
    layout: Layout,
    names: &str,
) -> Map<String, Value> {
    let mut a = Map::new();
    a.insert("long_name".into(), json!(row.long_name));
    a.insert("units".into(), json!(row.units));
    if let Some(s) = &row.standard_name {
        a.insert("standard_name".into(), json!(s));
    }
    if let Some(s) = &row.short_name {
        a.insert("short_name".into(), json!(s));
    }
    if let Some(p) = row.param_id {
        a.insert("ecmwf_param_id".into(), json!(p));
    }
    if let Some(c) = &row.comment {
        a.insert("comment".into(), json!(c));
    }
    a.insert("table_row".into(), json!(row.id));
    a.insert("naming".into(), json!(names));
    if domain.native.crs.is_some() && domain.regrid.is_none() {
        a.insert("grid_mapping".into(), json!("crs"));
    }
    let mut coordinates = Vec::new();
    if two_d_latlon(domain) {
        coordinates.push("latitude longitude");
        if domain.latlon_versions > 1 && domain.native.crs.is_some() {
            coordinates.push("projection_x_coordinate projection_y_coordinate");
        }
    }
    if row.kind != VariableKind::Static || domain.latlon_versions > 1 {
        coordinates.push("init_time");
        if layout == Layout::Analysis {
            coordinates.push("lead_time");
        }
    }
    if !coordinates.is_empty() {
        a.insert("coordinates".into(), json!(coordinates.join(" ")));
    }
    if row.kind == VariableKind::Level && !domain.levels_kept.is_empty() {
        a.insert("vertical_interpolation".into(), json!(VERTICAL_INTERPOLATION));
        let rule = match row.below_ground.as_deref() {
            Some("ecmwf-temperature") => BELOW_GROUND_TEMPERATURE,
            Some("ecmwf-geopotential") => BELOW_GROUND_GEOPOTENTIAL,
            _ => BELOW_GROUND_LOWEST,
        };
        a.insert("below_ground_rule".into(), json!(rule));
    }
    if is_accumulation(row) {
        a.insert("accumulation_period".into(), json!(accumulation_period(row, &domain.times)));
        a.insert("cell_methods".into(), json!("time: sum"));
        if domain.latlon_versions > 1 {
            a.insert("moving_grid_accumulation".into(), json!("the earlier accumulator is aligned to the current geographic points before subtraction; newly exposed ground outside the earlier footprint is missing"));
        }
    }
    a
}

fn write_coordinates(store: &Path, out: &Path, domain: &DomainState, request: &Request) -> Result<u64> {
    let layout = request.layout;
    let frames = domain.times.len();
    let mut bytes = 0u64;
    let whole_minutes = domain.times.iter().all(|t| (t - domain.init_time) % 60 == 0);
    let (unit, div) = if whole_minutes { ("minutes", 60) } else { ("seconds", 1) };
    let init_cf = times::cf(domain.init_time);
    let leads: Vec<i64> = domain.times.iter().map(|t| (t - domain.init_time) / div).collect();
    match layout {
        Layout::Analysis => {
            let meta = ArrayMeta {
                shape: vec![frames],
                chunks: vec![frames],
                dtype: Dtype::I64,
                dims: vec!["time".into()],
                attrs: attrs_of(&[
                    ("standard_name", json!("time")),
                    ("long_name", json!("valid time")),
                    ("units", json!(format!("{unit} since {init_cf}"))),
                    ("calendar", json!("proleptic_gregorian")),
                    ("axis", json!("T")),
                ]),
            };
            bytes += zarr::write_small(store, "time", &meta, &zarr::i64_bytes(&leads))?;
            let meta = ArrayMeta {
                shape: vec![frames],
                chunks: vec![frames],
                dtype: Dtype::I64,
                dims: vec!["time".into()],
                attrs: attrs_of(&[
                    ("long_name", json!("time since initialization")),
                    ("units", json!(unit)),
                    ("dtype", json!("timedelta64[ns]")),
                ]),
            };
            bytes += zarr::write_small(store, "lead_time", &meta, &zarr::i64_bytes(&leads))?;
        }
        Layout::Forecast => {
            let meta = ArrayMeta {
                shape: vec![1],
                chunks: vec![1],
                dtype: Dtype::I64,
                dims: vec!["time".into()],
                attrs: attrs_of(&[
                    ("standard_name", json!("forecast_reference_time")),
                    ("long_name", json!("initialization time")),
                    ("units", json!(format!("{unit} since {init_cf}"))),
                    ("calendar", json!("proleptic_gregorian")),
                ]),
            };
            bytes += zarr::write_small(store, "time", &meta, &zarr::i64_bytes(&[0]))?;
            let meta = ArrayMeta {
                shape: vec![frames],
                chunks: vec![frames],
                dtype: Dtype::I64,
                dims: vec!["prediction_timedelta".into()],
                attrs: attrs_of(&[
                    ("standard_name", json!("forecast_period")),
                    ("long_name", json!("lead time")),
                    ("units", json!(unit)),
                    ("dtype", json!("timedelta64[ns]")),
                ]),
            };
            bytes += zarr::write_small(store, "prediction_timedelta", &meta, &zarr::i64_bytes(&leads))?;
        }
    }
    // init_time: a scalar, seconds since 1970.
    let init_meta = ArrayMeta {
        shape: vec![],
        chunks: vec![],
        dtype: Dtype::I64,
        dims: vec![],
        attrs: attrs_of(&[
            ("long_name", json!("initialization time")),
            ("units", json!("seconds since 1970-01-01 00:00:00")),
            ("calendar", json!("proleptic_gregorian")),
        ]),
    };
    zarr::write_meta(store, "init_time", &init_meta)?;
    bytes += zarr::write_scalar_chunk(store, "init_time", &zarr::i64_bytes(&[domain.init_time]), Dtype::I64)?;

    // Levels.
    match request.levels.kind {
        LevelKind::Pressure if !domain.levels_kept.is_empty() => {
            let levels: Vec<i32> = domain.levels_kept.iter().map(|&l| l as i32).collect();
            let meta = ArrayMeta {
                shape: vec![levels.len()],
                chunks: vec![levels.len().max(1)],
                dtype: Dtype::I32,
                dims: vec!["level".into()],
                attrs: attrs_of(&[
                    ("long_name", json!("pressure level")),
                    ("standard_name", json!("air_pressure")),
                    ("units", json!("hPa")),
                    ("positive", json!("down")),
                    ("axis", json!("Z")),
                ]),
            };
            bytes += zarr::write_small(store, "level", &meta, &zarr::i32_bytes(&levels))?;
        }
        LevelKind::Model if !domain.model_levels.is_empty() => {
            let levels: Vec<i32> = domain.model_levels.iter().map(|&l| l as i32).collect();
            let meta = ArrayMeta {
                shape: vec![levels.len()],
                chunks: vec![levels.len().max(1)],
                dtype: Dtype::I32,
                dims: vec!["model_level".into()],
                attrs: attrs_of(&[
                    ("long_name", json!("model mass level, 1 at the bottom")),
                    ("positive", json!("up")),
                    ("axis", json!("Z")),
                ]),
            };
            bytes += zarr::write_small(store, "model_level", &meta, &zarr::i32_bytes(&levels))?;
            if let Some(eta) = &domain.eta {
                let meta = ArrayMeta {
                    shape: vec![eta.len()],
                    chunks: vec![eta.len()],
                    dtype: Dtype::F64,
                    dims: vec!["model_level".into()],
                    attrs: attrs_of(&[("long_name", json!("eta on mass levels (ZNU)")), ("units", json!("1"))]),
                };
                bytes += zarr::write_small(store, "eta", &meta, &zarr::f64_bytes(eta))?;
            }
        }
        _ => {}
    }

    // Horizontal coordinates.
    let lat_attrs = attrs_of(&[
        ("standard_name", json!("latitude")),
        ("long_name", json!("latitude")),
        ("units", json!("degrees_north")),
    ]);
    let lon_attrs = attrs_of(&[
        ("standard_name", json!("longitude")),
        ("long_name", json!("longitude")),
        ("units", json!("degrees_east")),
    ]);
    let one_d = match (&domain.regrid, &domain.native.regular) {
        (Some(r), _) => Some((r.lat.clone(), r.lon.clone())),
        (None, Some(reg)) if domain.latlon_versions == 1 => Some((reg.lat.clone(), reg.lon.clone())),
        (None, Some(_)) => None,
        (None, None) => None,
    };
    if let Some((lat, lon)) = one_d {
        let mut a = lat_attrs.clone();
        a.insert("axis".into(), json!("Y"));
        let meta = ArrayMeta { shape: vec![lat.len()], chunks: vec![lat.len()], dtype: Dtype::F64, dims: vec!["latitude".into()], attrs: a };
        bytes += zarr::write_small(store, "latitude", &meta, &zarr::f64_bytes(&lat))?;
        let mut a = lon_attrs.clone();
        a.insert("axis".into(), json!("X"));
        let meta = ArrayMeta { shape: vec![lon.len()], chunks: vec![lon.len()], dtype: Dtype::F64, dims: vec!["longitude".into()], attrs: a };
        bytes += zarr::write_small(store, "longitude", &meta, &zarr::f64_bytes(&lon))?;
    } else {
        let (ny, nx) = (domain.meta.ny, domain.meta.nx);
        let moving = domain.latlon_versions > 1;
        for (name, prefix, attrs) in [("latitude", "xlat", &lat_attrs), ("longitude", "xlong", &lon_attrs)] {
            if moving {
                let mut dims = time_dims(layout);
                dims.extend(["y".to_string(), "x".to_string()]);
                let mut shape = time_shape(layout, frames);
                shape.extend([ny, nx]);
                let mut chunks = ones(time_dims(layout).len());
                chunks.extend([ny, nx]);
                let meta = ArrayMeta { shape, chunks, dtype: Dtype::F32, dims, attrs: attrs.clone() };
                zarr::write_meta(store, name, &meta)?;
                for (t, &version) in domain.latlon_version.iter().enumerate() {
                    let plane = state::load_plane(out, &format!("{}-{prefix}-{version}.f64", domain.id))?.unwrap_or_default();
                    let plane = match &domain.native.regular {
                        Some(reg) => grid::reorder(&plane, nx, reg),
                        None => plane,
                    };
                    let values: Vec<f32> = plane.iter().map(|&v| v as f32).collect();
                    let mut index = match layout {
                        Layout::Analysis => vec![t],
                        Layout::Forecast => vec![0, t],
                    };
                    index.extend([0, 0]);
                    bytes += zarr::write_chunk(store, name, &index, &zarr::f32_bytes(&values), Dtype::F32)?;
                }
            } else {
                let plane = state::load_plane(out, &format!("{}-{prefix}-0.f64", domain.id))?.unwrap_or_default();
                let values: Vec<f32> = plane.iter().map(|&v| v as f32).collect();
                let meta = ArrayMeta { shape: vec![ny, nx], chunks: vec![ny, nx], dtype: Dtype::F32, dims: vec!["y".into(), "x".into()], attrs: attrs.clone() };
                bytes += zarr::write_small(store, name, &meta, &zarr::f32_bytes(&values))?;
            }
        }
        if let (Some(x), Some(y), Some(crs)) = (&domain.native.x, &domain.native.y, &domain.native.crs) {
            if moving {
                // Absolute projection axes move with the nest.  They are
                // auxiliary coordinates: x/y remain the array indices.
                for (name, axis, size, standard_name) in [
                    ("projection_x_coordinate", "x", nx, "projection_x_coordinate"),
                    ("projection_y_coordinate", "y", ny, "projection_y_coordinate"),
                ] {
                    let mut shape = time_shape(layout, frames);
                    shape.push(size);
                    let mut chunks = ones(time_dims(layout).len());
                    chunks.push(size);
                    let mut dims = time_dims(layout);
                    dims.push(axis.into());
                    let meta = ArrayMeta {
                        shape, chunks, dtype: Dtype::F64, dims,
                        attrs: attrs_of(&[("standard_name", json!(standard_name)), ("units", json!("m"))]),
                    };
                    zarr::write_meta(store, name, &meta)?;
                    for (t, &version) in domain.latlon_version.iter().enumerate() {
                        let lat = state::load_plane(out, &format!("{}-xlat-{version}.f64", domain.id))?.unwrap_or_default();
                        let lon = state::load_plane(out, &format!("{}-xlong-{version}.f64", domain.id))?.unwrap_or_default();
                        let placed = grid::describe(&domain.meta, &lat, &lon);
                        let values = if axis == "x" { placed.x } else { placed.y }
                            .ok_or_else(|| crate::error::refuse(format!(
                                "domain {} moved onto coordinates that do not fit its projection axes, so a single grid mapping would place its fields incorrectly", domain.id
                            )))?;
                        let mut index = match layout {
                            Layout::Analysis => vec![t],
                            Layout::Forecast => vec![0, t],
                        };
                        index.push(0);
                        bytes += zarr::write_chunk(store, name, &index, &zarr::f64_bytes(&values), Dtype::F64)?;
                    }
                }
            } else {
            let meta = ArrayMeta {
                shape: vec![x.len()],
                chunks: vec![x.len()],
                dtype: Dtype::F64,
                dims: vec!["x".into()],
                attrs: attrs_of(&[("standard_name", json!("projection_x_coordinate")), ("units", json!("m")), ("axis", json!("X"))]),
            };
            bytes += zarr::write_small(store, "x", &meta, &zarr::f64_bytes(x))?;
            let meta = ArrayMeta {
                shape: vec![y.len()],
                chunks: vec![y.len()],
                dtype: Dtype::F64,
                dims: vec!["y".into()],
                attrs: attrs_of(&[("standard_name", json!("projection_y_coordinate")), ("units", json!("m")), ("axis", json!("Y"))]),
            };
            bytes += zarr::write_small(store, "y", &meta, &zarr::f64_bytes(y))?;
            }
            let meta = ArrayMeta { shape: vec![], chunks: vec![], dtype: Dtype::I32, dims: vec![], attrs: crs.clone() };
            zarr::write_meta(store, "crs", &meta)?;
            bytes += zarr::write_scalar_chunk(store, "crs", &zarr::i32_bytes(&[0]), Dtype::I32)?;
        }
    }
    Ok(bytes)
}

fn group_attrs(domain: &DomainState, request: &Request, omitted: &[(String, String)]) -> Map<String, Value> {
    let meta = &domain.meta;
    let mut inputs: Vec<&str> = domain.inputs.iter().map(|i| i.sha256.as_str()).collect();
    inputs.sort_unstable();
    inputs.dedup();
    let input_sha = frame::sha256_bytes(inputs.join("\n").as_bytes());
    let version = meta.gpuwm_version.clone();
    let engine = meta.source_engine.as_deref().unwrap_or("model history");
    let source = match &version {
        Some(v) => format!("{engine} {v}"),
        None => format!("{engine} (the history files state no version)"),
    };
    let horizontal = match &domain.regrid {
        Some(r) => r.description.clone(),
        None => domain.native.description.clone(),
    };
    let mut a = Map::new();
    a.insert("Conventions".into(), json!("CF-1.8"));
    a.insert("title".into(), json!(format!("Model forecast, domain {}", domain.id)));
    a.insert("source".into(), json!(source));
    a.insert(
        "history".into(),
        json!(format!(
            "{} ml-export {}: {}",
            request.provenance.created_utc, request.provenance.exporter_version, request.provenance.options
        )),
    );
    a.insert("exporter".into(), json!(format!("{} ml-export {}", request.provenance.engine, request.provenance.exporter_version)));
    a.insert("model_config_sha256".into(), json!(domain.config_digest));
    a.insert("model_config_digest_kind".into(), json!(domain.config_digest_kind));
    a.insert("initial_condition_source".into(), json!(meta.ic_source.clone().unwrap_or_else(|| "unknown".into())));
    a.insert("initial_condition_cycle".into(), json!(meta.ic_cycle.clone().unwrap_or_else(|| "unknown".into())));
    a.insert("init_time".into(), json!(times::iso(domain.init_time)));
    a.insert("domain_id".into(), json!(domain.id));
    a.insert("parent_domain_id".into(), json!(meta.parent.clone().unwrap_or_else(|| "none".into())));
    a.insert("grid_spacing_m".into(), json!(meta.dx));
    a.insert(
        "lateral_boundary_rows".into(),
        match meta.spec_bdy_width {
            Some(w) => json!(w),
            None => json!("not stated by the history files"),
        },
    );
    a.insert("level_set".into(), json!(request.levels.set));
    if request.levels.kind == LevelKind::Pressure && !domain.levels_kept.is_empty() {
        a.insert("levels_dropped_above_model_top".into(), json!(domain.levels_dropped));
        a.insert("model_top_hpa".into(), json!((domain.lid_pa / 10.0).round() / 10.0));
        a.insert(
            "model_top_source".into(),
            json!(if domain.lid_stated { "P_TOP" } else { "the top model mass level (the history files state no P_TOP)" }),
        );
        a.insert("vertical_interpolation".into(), json!(VERTICAL_INTERPOLATION));
        a.insert(
            "below_ground_rule".into(),
            json!("temperature and geopotential: the ECMWF rule (Trenberth, Berry and Buja 1993, NCAR/TN-396) ERA5's pressure levels are filled with; every other field: the lowest model level's value; below_ground = 1 where the level's pressure exceeds the surface pressure"),
        );
        a.insert("model_top_rule".into(), json!(MODEL_TOP_RULE));
    }
    a.insert("horizontal_grid".into(), json!(horizontal));
    if let Some(preset) = &meta.history_preset {
        a.insert("history_preset".into(), json!(preset));
    }
    a.insert("input_sha256".into(), json!(input_sha));
    a.insert("naming".into(), json!(request.names));
    a.insert(
        "layout".into(),
        json!(match request.layout {
            Layout::Analysis => "analysis",
            Layout::Forecast => "forecast",
        }),
    );
    if !omitted.is_empty() {
        a.insert(
            "variables_omitted".into(),
            json!(omitted.iter().map(|(id, why)| format!("{id}: {why}")).collect::<Vec<_>>()),
        );
    }
    a
}

/// Close one domain's store; returns (bytes of coordinates and metadata
/// written now, omitted variables).
fn close_domain(out: &Path, domain: &DomainState, request: &Request) -> Result<(u64, Vec<(String, String)>)> {
    let store = out.join(format!("{}.zarr", domain.id));
    let layout = request.layout;
    let frames = domain.times.len();
    let (ydim, xdim) = space_dims(domain);
    let (ny, nx) = (domain.out_ny, domain.out_nx);
    let mut omitted: Vec<(String, String)> = Vec::new();
    let level_dim = match request.levels.kind {
        LevelKind::Pressure => "level",
        LevelKind::Model => "model_level",
    };
    let levels = match request.levels.kind {
        LevelKind::Pressure => domain.levels_kept.len(),
        LevelKind::Model => domain.model_levels.len(),
    };
    for var in &domain.variables {
        let row = request.variables.iter().find(|r| r.id == var.id).expect("state rows come from the request");
        if let Some(why) = &var.omitted {
            omitted.push((var.id.clone(), why.clone()));
            continue;
        }
        if let Ok(Op::Accumulation(period)) = Op::parse(&row.op) {
            let key = match period {
                crate::ops::Period::Interval => "interval".to_string(),
                crate::ops::Period::Hours(h) => format!("{h}h"),
            };
            let found = domain.period_partners.get(&key).map(|v| v.iter().any(|&b| b)).unwrap_or(false);
            if !found && !matches!(period, crate::ops::Period::Interval) {
                let _ = std::fs::remove_dir_all(store.join(&var.name));
                omitted.push((
                    var.id.clone(),
                    format!(
                        "no frame has a frame {key} before it among the exported times, so every value would be empty"
                    ),
                ));
                continue;
            }
        }
        let (shape, chunks, dims) = match row.kind {
            VariableKind::Level => {
                let mut shape = time_shape(layout, frames);
                shape.extend([levels, ny, nx]);
                let mut chunks = ones(shape.len() - 3);
                chunks.extend([domain.levels_per_chunk, ny, nx]);
                let mut dims = time_dims(layout);
                dims.extend([level_dim.to_string(), ydim.clone(), xdim.clone()]);
                (shape, chunks, dims)
            }
            VariableKind::Surface => {
                let mut shape = time_shape(layout, frames);
                shape.extend([ny, nx]);
                let mut chunks = ones(shape.len() - 2);
                chunks.extend([ny, nx]);
                let mut dims = time_dims(layout);
                dims.extend([ydim.clone(), xdim.clone()]);
                (shape, chunks, dims)
            }
            VariableKind::Static if domain.latlon_versions > 1 => {
                let mut shape = time_shape(layout, frames);
                shape.extend([ny, nx]);
                let mut chunks = ones(time_dims(layout).len());
                chunks.extend([ny, nx]);
                let mut dims = time_dims(layout);
                dims.extend([ydim.clone(), xdim.clone()]);
                (shape, chunks, dims)
            }
            VariableKind::Static => {
                // During append these carry every time in case a later
                // frame moves.  A stationary export retains its usual 2-D
                // representation and exactly the first plane's bytes.
                for t in 0..frames {
                    let key = match layout {
                        Layout::Analysis => format!("{t}.0.0"),
                        Layout::Forecast => format!("0.{t}.0.0"),
                    };
                    let path = store.join(&var.name).join(key);
                    if t == 0 {
                        if path.is_file() {
                            std::fs::rename(&path, store.join(&var.name).join("0.0"))?;
                        }
                    } else if path.is_file() {
                        std::fs::remove_file(path)?;
                    }
                }
                (vec![ny, nx], vec![ny, nx], vec![ydim.clone(), xdim.clone()])
            }
        };
        let meta = ArrayMeta {
            shape,
            chunks,
            dtype: Dtype::F32,
            dims,
            attrs: variable_attrs(row, domain, layout, &request.names),
        };
        zarr::write_meta(&store, &var.name, &meta)?;
    }
    if request.levels.kind == LevelKind::Pressure && store.join(MASK_NAME).is_dir() {
        let mut shape = time_shape(layout, frames);
        shape.extend([levels, ny, nx]);
        let mut chunks = ones(shape.len() - 3);
        chunks.extend([domain.levels_per_chunk, ny, nx]);
        let mut dims = time_dims(layout);
        dims.extend([level_dim.to_string(), ydim.clone(), xdim.clone()]);
        let mut attrs = attrs_of(&[
            ("long_name", json!("level lies below the ground surface")),
            ("flag_values", json!([0, 1])),
            ("flag_meanings", json!("above_ground below_ground")),
            ("comment", json!("1 where the level's pressure exceeds the surface pressure: the values there were filled by the below-ground rule, not taken from the model column. Not an ERA5 variable; loaders that do not know it can ignore it.")),
        ]);
        if two_d_latlon(domain) {
            attrs.insert("coordinates".into(), json!("latitude longitude"));
        }
        let meta = ArrayMeta { shape, chunks, dtype: Dtype::U8, dims, attrs };
        zarr::write_meta(&store, MASK_NAME, &meta)?;
    }
    let bytes = write_coordinates(&store, out, domain, request)?;
    zarr::write_group(&store, &group_attrs(domain, request, &omitted))?;
    zarr::consolidate(&store)?;
    Ok((bytes, omitted))
}

fn readme(request: &Request, state: &State, slug: &str, zip: bool) -> String {
    let mut text = String::new();
    text.push_str("Machine-learning export\n=======================\n\n");
    text.push_str(&format!(
        "Written by ml-export {} from model history files. One Zarr (format 2, consolidated) dataset per domain.\n\n",
        request.provenance.exporter_version
    ));
    text.push_str("Open it with xarray:\n\n");
    text.push_str("    import xarray as xr\n");
    for id in state.domains.keys() {
        text.push_str(&format!("    ds = xr.open_zarr(\"{slug}/{id}.zarr\")\n"));
    }
    if zip {
        let first = state.domains.keys().next().cloned().unwrap_or_else(|| "d01".into());
        text.push_str("\nor straight from the ZIP, without unpacking it:\n\n");
        text.push_str("    import xarray as xr, zarr\n");
        text.push_str(&format!(
            "    ds = xr.open_zarr(zarr.storage.ZipStore(\"{slug}-ml.zip\", mode=\"r\"), group=\"{slug}/{first}.zarr\")\n"
        ));
    }
    text.push_str("\nDatasets:\n");
    for (id, domain) in &state.domains {
        let grid = domain.regrid.as_ref().map(|r| r.description.clone()).unwrap_or_else(|| domain.native.description.clone());
        text.push_str(&format!(
            "  {id}: {} times from {} to {}, {} x {} points, {grid}\n",
            domain.times.len(),
            domain.times.first().map(|&t| times::iso(t)).unwrap_or_default(),
            domain.times.last().map(|&t| times::iso(t)).unwrap_or_default(),
            domain.out_ny,
            domain.out_nx,
        ));
        if request.levels.kind == LevelKind::Pressure && !domain.levels_kept.is_empty() {
            text.push_str(&format!(
                "      levels (hPa): {:?}; left out above the model top ({:.1} hPa): {:?}\n",
                domain.levels_kept,
                domain.lid_pa / 100.0,
                domain.levels_dropped
            ));
        }
        for var in &domain.variables {
            if let Some(why) = &var.omitted {
                text.push_str(&format!("      not written: {} ({why})\n", var.id));
            }
        }
    }
    text.push_str("\nVariables:\n");
    for row in &request.variables {
        let name = row.names.get(&request.names).cloned().unwrap_or_default();
        text.push_str(&format!("  {name:<32} {:<14} {}\n", row.units, row.long_name));
    }
    if request.levels.kind == LevelKind::Pressure {
        text.push_str(&format!(
            "\nOn pressure levels: {VERTICAL_INTERPOLATION}. Below the lowest model level temperature and geopotential follow the ECMWF rule ERA5's pressure levels are filled with (Trenberth, Berry and Buja 1993), and every other field takes the lowest model level's value. The {MASK_NAME} mask is 1 where a level lies under the ground surface; drop those points with ds.where(ds.{MASK_NAME} == 0) if you want only model column.\n"
        ));
    }
    text.push_str("\nThe receipt (ml-export-receipt.json) lists every input file by name and SHA-256, the time each frame took, and every variable left out and why.\n");
    text
}

fn receipt(request: &Request, state: &State, omitted: &[(String, Vec<(String, String)>)], bytes: u64, zip: &Option<(PathBuf, u64)>) -> Value {
    let domains: Vec<Value> = state
        .domains
        .values()
        .map(|d| {
            let left_out = omitted.iter().find(|(id, _)| *id == d.id).map(|(_, v)| v.clone()).unwrap_or_default();
            json!({
                "id": d.id,
                "frames": d.inputs.iter().map(|i| json!({
                    "valid": times::iso(i.valid),
                    "input": i.name,
                    "sha256": i.sha256,
                    "time_index": i.time_index,
                    "seconds": (i.seconds * 1000.0).round() / 1000.0,
                    "read_seconds": (i.read_seconds * 1000.0).round() / 1000.0,
                    "bytes_written": i.bytes_written,
                })).collect::<Vec<_>>(),
                "grid": [d.out_ny, d.out_nx],
                "horizontal_grid": d.regrid.as_ref().map(|r| r.description.clone()).unwrap_or_else(|| d.native.description.clone()),
                "levels": d.levels_kept,
                "levels_dropped_above_model_top": d.levels_dropped,
                "model_levels": d.model_levels,
                "model_top_hpa": if d.lid_pa > 0.0 { Some((d.lid_pa / 10.0).round() / 10.0) } else { None },
                "variables": d.variables.iter().filter(|v| v.omitted.is_none() && !left_out.iter().any(|(id, _)| *id == v.id)).map(|v| v.name.clone()).collect::<Vec<_>>(),
                "omitted": left_out.iter().map(|(id, why)| json!({"variable": id, "reason": why})).collect::<Vec<_>>(),
                "model_config_sha256": d.config_digest,
                "model_config_digest_kind": d.config_digest_kind,
            })
        })
        .collect();
    json!({
        "schema": "ml-export.receipt/v1",
        "exporter": format!("ml-export {}", request.provenance.exporter_version),
        "engine": request.provenance.engine,
        "created_utc": request.provenance.created_utc,
        "options": request.provenance.options,
        "levels": request.levels,
        "grid": request.grid,
        "names": request.names,
        "layout": request.layout,
        "table_rows": request.variables,
        "domains": domains,
        "bytes": bytes,
        "zip": zip.as_ref().map(|(p, b)| json!({"name": p.file_name().map(|n| n.to_string_lossy().into_owned()), "bytes": b})),
    })
}

fn folder_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                match e.file_type() {
                    Ok(t) if t.is_dir() => stack.push(e.path()),
                    Ok(_) => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                    Err(_) => {}
                }
            }
        }
    }
    total
}

pub fn finalize(out: &Path, state: &State, zip: bool, progress: Progress<'_>) -> Result<Closed> {
    let request = &state.request;
    let mut omitted_all = Vec::new();
    for domain in state.domains.values() {
        let (_, omitted) = close_domain(out, domain, request)?;
        omitted_all.push((domain.id.clone(), omitted));
    }
    let _ = std::fs::remove_dir_all(state::state_dir(out));
    let slug = export::slug(out);
    // The export folder is itself a group over its domains' stores.
    let folder_attrs = json!({
        "title": "Model forecast, every domain",
        "exporter": format!("ml-export {}", request.provenance.exporter_version),
        "domains": state.domains.keys().collect::<Vec<_>>(),
    });
    zarr::write_document(&out.join(".zgroup"), &json!({"zarr_format": 2}))?;
    zarr::write_document(&out.join(".zattrs"), &folder_attrs)?;
    let tree = zarr::consolidated_tree(out, "")?;
    zarr::write_document(&out.join(".zmetadata"), &tree)?;
    std::fs::write(out.join("README.txt"), readme(request, state, &slug, zip))?;
    let bytes_before_receipt = folder_bytes(out);
    let receipt_path = out.join("ml-export-receipt.json");
    let provisional = receipt(request, state, &omitted_all, bytes_before_receipt, &None);
    std::fs::write(&receipt_path, serde_json::to_string_pretty(&provisional).unwrap_or_default() + "\n")?;
    let mut zipped = None;
    if zip {
        let files = zipout::collect(out, &slug)?;
        let archive = export::zip_path(out);
        // The archive's root is a group too, consolidated over everything,
        // so `ZipStore` + `group="<slug>/dNN.zarr"` opens under zarr 3 (which
        // reads the root first) as well as zarr 2 (which reads the path).
        let mut root_tree = zarr::consolidated_tree(out, &format!("{slug}/"))?;
        if let Some(metadata) = root_tree.get_mut("metadata").and_then(Value::as_object_mut) {
            metadata.insert(".zgroup".into(), json!({"zarr_format": 2}));
            metadata.insert(".zattrs".into(), json!({}));
        }
        let memory = vec![
            (".zgroup".to_string(), zarr::json_bytes(&json!({"zarr_format": 2}))),
            (".zattrs".to_string(), zarr::json_bytes(&json!({}))),
            (".zmetadata".to_string(), zarr::json_bytes(&root_tree)),
        ];
        let size = zipout::write_with(&archive, &files, &memory)?;
        progress(json!({"event": "zip", "name": archive.file_name().map(|n| n.to_string_lossy().into_owned()), "bytes": size}));
        zipped = Some((archive, size));
    }
    let bytes = folder_bytes(out);
    progress(json!({
        "event": "finalized",
        "domains": state.domains.keys().collect::<Vec<_>>(),
        "bytes": bytes,
    }));
    Ok(Closed { bytes, zip: zipped })
}
