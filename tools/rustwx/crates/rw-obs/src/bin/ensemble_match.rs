//! Freeze observation geometry once, then match every member through that plan.
//! All diagnostics, projection, interpolation and precipitation arithmetic are
//! native. The caller supplies paths, member identities and the scoring window.
use obs_regrid::{apply_plan, build_plan, Method};
use rw_obs::pack::{decode_pack, payload_digest, validate_arrays, ArrayEntry, GEO_SCHEMA};
use rw_obs::precipitation::{instant, read_frame, verify};
use rw_obs::seam::seam_time;
use rw_obs::{err, hex_sha256};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use static_fields::projection::{GridSpec, ProjectedGrid, ProjectionKind};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use wrf_core::{getvar, ComputeOpts, WrfFile};
#[path = "ensemble_match/diagnostics.rs"]
mod diagnostics;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const PLAN_SCHEMA: &str = "gpuwm.ensemble-match-plan.v1";
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

#[derive(Deserialize)]
struct Prepare {
    grid: PathBuf,
    surface: PathBuf,
    precipitation_geometry: PathBuf,
    /// observation-to-model for a fine observation grid, model-to-observation
    /// when observations have a coarser native support than the model.
    precipitation_direction: String,
    interior_rim_m: f64,
    boundary_rows: usize,
    elevation_tolerance_m: f64,
    max_regrid_distance_m: f64,
}
#[derive(Clone, Serialize, Deserialize)]
struct Reference {
    path: PathBuf,
    sha256: String,
}
fn reference(path: &Path) -> Result<Reference> {
    Ok(Reference {
        path: path.to_owned(),
        sha256: hex_sha256(&std::fs::read(path)?),
    })
}
fn check_reference(item: &Reference) -> Result<()> {
    if reference(&item.path)?.sha256 != item.sha256 {
        return Err(err(format!(
            "source digest changed: {}",
            item.path.display()
        )));
    }
    Ok(())
}
#[derive(Serialize, Deserialize)]
struct Station {
    station_id: String,
    latitude: f64,
    longitude: f64,
    elevation_m: f64,
}
#[derive(Deserialize)]
struct Report {
    station_id: String,
    valid_time: String,
    observation_time: String,
    values: BTreeMap<String, f64>,
    flags: Vec<String>,
}
#[derive(Deserialize)]
struct Surface {
    schema: String,
    status: String,
    provenance: rw_obs::seam::Provenance,
    station_table_sha256: String,
    valid_times: Vec<String>,
    match_seconds: i64,
    min_report_rate: f64,
    max_screen_rate: f64,
    stations: Vec<Station>,
    reports: Vec<Report>,
}
fn surface(path: &Path) -> Result<Surface> {
    let record: Surface = serde_json::from_slice(&std::fs::read(path)?)?;
    if record.schema != "gpuwm-obs.asos-surface.v2"
        || record.status != "READY"
        || record.provenance.is_stub
        || record.match_seconds != 600
        || record.min_report_rate != 0.8
        || record.max_screen_rate != 0.05
        || record.valid_times.is_empty()
        || record.station_table_sha256.len() != 64
    {
        return Err(err(
            "surface record does not carry the registered native QC and matching contract",
        ));
    }
    if hex_sha256(&std::fs::read(&record.provenance.uri)?) != record.provenance.sha256 {
        return Err(err("surface raw report digest differs"));
    }
    let mut identities = BTreeSet::new();
    for station in &record.stations {
        if station.station_id.is_empty()
            || station.station_id.contains(['\t', '\n', '\r'])
            || !identities.insert(&station.station_id)
        {
            return Err(err(
                "surface station identities must be unique, nonempty TSV labels",
            ));
        }
    }
    let mut matched = BTreeSet::new();
    for report in &record.reports {
        if !identities.contains(&report.station_id)
            || !matched.insert((&report.station_id, &report.valid_time))
            || !record.valid_times.contains(&report.valid_time)
            || (instant(&report.valid_time)? - instant(&report.observation_time)?)
                .num_seconds()
                .abs()
                > record.match_seconds
        {
            return Err(err(
                "surface report has unknown station, duplicate match or invalid time offset",
            ));
        }
    }
    Ok(record)
}
#[derive(Serialize, Deserialize)]
struct StationPosition {
    station_id: String,
    x: f64,
    y: f64,
    keep: bool,
    drop_reason: String,
    terrain_offset_m: Option<f64>,
}
#[derive(Serialize, Deserialize)]
struct Remap {
    source_index: Vec<i64>,
    reachable: Vec<bool>,
    source_shape: (usize, usize),
    destination_shape: (usize, usize),
    max_distance_m: f64,
    max_used_distance_m: f64,
}
#[derive(Serialize, Deserialize)]
struct Plan {
    schema: String,
    grid: Reference,
    surface: Reference,
    precipitation_geometry: Reference,
    nx: usize,
    ny: usize,
    grid_geometry_sha256: String,
    precipitation_geometry_sha256: String,
    grid_interior: Vec<bool>,
    precipitation_target_mask: Vec<bool>,
    stations: Vec<StationPosition>,
    remap: Remap,
    precipitation_direction: String,
    parameters: Value,
}

fn field(file: &WrfFile, name: &str) -> Result<Vec<f64>> {
    let values = file.read_var(name, 0)?;
    if values.len() != file.nxy() || values.iter().any(|x| !x.is_finite()) {
        return Err(err(format!("{name} must fill one finite mass-grid plane")));
    }
    Ok(values)
}
fn geometry_hash(lat: &[f64], lon: &[f64]) -> String {
    let bytes = lat
        .iter()
        .chain(lon)
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    hex_sha256(&bytes)
}
fn projection(file: &WrfFile, latitude: &[f64], longitude: &[f64]) -> Result<ProjectedGrid> {
    let kind = match file.global_attr_i32("MAP_PROJ")? {
        1 => ProjectionKind::Lambert,
        2 => ProjectionKind::Polar,
        3 => ProjectionKind::Mercator,
        value => {
            return Err(err(format!(
                "MAP_PROJ {value} is not a supported regional observation grid"
            )))
        }
    };
    let spec = GridSpec {
        kind,
        ref_lat: latitude[0],
        ref_lon: longitude[0],
        truelat1: file.global_attr_f64("TRUELAT1")?,
        truelat2: file.global_attr_f64("TRUELAT2")?,
        stand_lon: file.global_attr_f64("STAND_LON")?,
        dx: file.dx,
        dy: file.dy,
        e_we: file.nx as i64 + 1,
        e_sn: file.ny as i64 + 1,
        known_x: 1.0,
        known_y: 1.0,
        moad_cen_lat: file
            .global_attr_f64("MOAD_CEN_LAT")
            .or_else(|_| file.global_attr_f64("CEN_LAT"))?,
        moad_cen_lon: file.global_attr_f64("CEN_LON")?,
        lat_deg: Vec::new(),
        lon0_deg: 0.0,
        dlon_deg: 0.0,
    };
    let projected = ProjectedGrid::new(spec)?;
    let mut worst = 0.0f64;
    for index in 0..latitude.len() {
        let (x, y) = projected.latlon_to_ij(latitude[index], longitude[index]);
        worst = worst
            .max((x - 1.0 - (index % file.nx) as f64).abs())
            .max((y - 1.0 - (index / file.nx) as f64).abs());
    }
    if !worst.is_finite() || worst > 0.01 {
        return Err(err(format!(
            "WRF projection differs from file coordinates by {worst} cells, limit 0.01"
        )));
    }
    Ok(projected)
}
#[derive(Deserialize)]
struct Geo {
    schema: String,
    status: String,
    content_sha256: String,
    arrays: BTreeMap<String, ArrayEntry>,
}
fn geo(path: &Path) -> Result<(Geo, Vec<f64>, Vec<f64>, (usize, usize))> {
    let (meta, payload): (Geo, Vec<u8>) = decode_pack(&std::fs::read(path)?)?;
    if meta.schema != GEO_SCHEMA
        || meta.status != "READY"
        || meta.content_sha256 != payload_digest(&payload)
    {
        return Err(err(
            "observation geometry schema, readiness or digest differs",
        ));
    }
    validate_arrays(&meta.arrays, payload.len())?;
    let read = |name: &str| -> Result<Vec<f64>> {
        let entry = meta
            .arrays
            .get(name)
            .ok_or_else(|| err(format!("geometry has no {name}")))?;
        if entry.dtype != "<f8" || entry.shape.len() != 2 {
            return Err(err("geometry needs float64 two-dimensional coordinates"));
        }
        let values = payload[entry.offset..entry.offset + entry.bytes]
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values.iter().any(|x| !x.is_finite()) {
            return Err(err("nonfinite observation coordinate"));
        }
        Ok(values)
    };
    let latitude = read("latitude")?;
    let longitude = read("longitude")?;
    let shape = meta.arrays["latitude"].shape.clone();
    if shape != meta.arrays["longitude"].shape {
        return Err(err("observation latitude and longitude shapes differ"));
    }
    Ok((meta, latitude, longitude, (shape[0], shape[1])))
}
fn prepare(request: &Prepare) -> Result<Plan> {
    if !request.interior_rim_m.is_finite()
        || request.interior_rim_m < 0.0
        || !request.elevation_tolerance_m.is_finite()
        || request.elevation_tolerance_m <= 0.0
    {
        return Err(err("invalid interior or terrain tolerance"));
    }
    let file = WrfFile::open(&request.grid)?;
    let latitude = field(&file, "XLAT")?;
    let longitude = field(&file, "XLONG")?;
    if !file.dx.is_finite() || file.dx <= 0.0 || !file.dy.is_finite() || file.dy <= 0.0 {
        return Err(err("grid spacing must be finite and positive"));
    }
    let terrain = field(&file, "HGT")?;
    let land = field(&file, "LANDMASK")?;
    if land.iter().any(|value| *value != 0.0 && *value != 1.0) {
        return Err(err("LANDMASK must contain only the native 0/1 land flags"));
    }
    let projected = projection(&file, &latitude, &longitude)?;
    let rim_x = request.boundary_rows + (request.interior_rim_m / file.dx).ceil() as usize;
    let rim_y = request.boundary_rows + (request.interior_rim_m / file.dy).ceil() as usize;
    if 2 * rim_x >= file.nx || 2 * rim_y >= file.ny {
        return Err(err("interior exclusion leaves no forecast cells"));
    }
    let interior = (0..file.nxy())
        .map(|k| {
            k % file.nx >= rim_x
                && k % file.nx < file.nx - rim_x
                && k / file.nx >= rim_y
                && k / file.nx < file.ny - rim_y
        })
        .collect::<Vec<_>>();
    let observation = surface(&request.surface)?;
    let stations = observation
        .stations
        .iter()
        .map(|station| {
            let (px, py) = projected.latlon_to_ij(station.latitude, station.longitude);
            let (x, y) = (px - 1.0, py - 1.0);
            let mut pos = StationPosition {
                station_id: station.station_id.clone(),
                x,
                y,
                keep: false,
                drop_reason: String::new(),
                terrain_offset_m: None,
            };
            if !x.is_finite()
                || !y.is_finite()
                || x < 0.0
                || y < 0.0
                || x > (file.nx - 1) as f64
                || y > (file.ny - 1) as f64
            {
                pos.drop_reason = "outside-domain".into();
                return pos;
            }
            let index = y.round_ties_even() as usize * file.nx + x.round_ties_even() as usize;
            if !interior[index] {
                pos.drop_reason = "outside-interior-mask".into();
                return pos;
            }
            if land[index] < 0.5 {
                pos.drop_reason = "not-land".into();
                return pos;
            }
            let offset = terrain[index] - station.elevation_m;
            pos.terrain_offset_m = Some(offset);
            if offset.abs() > request.elevation_tolerance_m {
                pos.drop_reason = "terrain-mismatch".into();
                return pos;
            }
            pos.keep = true;
            pos
        })
        .collect::<Vec<_>>();
    let (observed, obs_lat, obs_lon, obs_shape) = geo(&request.precipitation_geometry)?;
    let model_shape = (file.ny, file.nx);
    let native_plan = match request.precipitation_direction.as_str() {
        "observation-to-model" => build_plan(
            Method::CellAverage,
            &obs_lat,
            &obs_lon,
            obs_shape,
            &latitude,
            &longitude,
            model_shape,
            request.max_regrid_distance_m,
        )?,
        "model-to-observation" => build_plan(
            Method::CellAverage,
            &latitude,
            &longitude,
            model_shape,
            &obs_lat,
            &obs_lon,
            obs_shape,
            request.max_regrid_distance_m,
        )?,
        _ => {
            return Err(err(
                "precipitation direction must be observation-to-model or model-to-observation",
            ))
        }
    };
    let target_mask = if request.precipitation_direction == "observation-to-model" {
        interior
            .iter()
            .zip(&native_plan.reachable)
            .map(|(a, b)| *a && *b)
            .collect()
    } else {
        let mut mask = native_plan.reachable.clone();
        for (index, &target) in native_plan.source_index.iter().enumerate() {
            if target >= 0 && !interior[index] {
                mask[target as usize] = false;
            }
        }
        mask
    };
    let remap = Remap {
        source_index: native_plan.source_index,
        reachable: native_plan.reachable,
        source_shape: native_plan.source_shape,
        destination_shape: native_plan.destination_shape,
        max_distance_m: native_plan.max_distance_m,
        max_used_distance_m: native_plan.max_used_distance_m,
    };
    Ok(Plan {
        schema: PLAN_SCHEMA.into(),
        grid: reference(&request.grid)?,
        surface: reference(&request.surface)?,
        precipitation_geometry: reference(&request.precipitation_geometry)?,
        nx: file.nx,
        ny: file.ny,
        grid_geometry_sha256: geometry_hash(&latitude, &longitude),
        precipitation_geometry_sha256: observed.content_sha256,
        grid_interior: interior,
        precipitation_target_mask: target_mask,
        stations,
        remap,
        precipitation_direction: request.precipitation_direction.clone(),
        parameters: json!({"surface_interpolation":"bilinear","projection":"static-fields module_llxy with whole-grid 0.01-cell validation",
        "precipitation_remap":"obs-regrid cell_average, reverse centre assignment, not area-overlap conservative",
        "interior_rim_m":request.interior_rim_m,"boundary_rows":request.boundary_rows,"elevation_tolerance_m":request.elevation_tolerance_m,
        "dx_m":file.dx,"dy_m":file.dy,"excluded_x_rows":rim_x,"excluded_y_rows":rim_y,"surface_source_qc":"rw_asos v2"}),
    })
}

#[derive(Deserialize)]
struct Member {
    id: String,
    file: PathBuf,
    #[serde(default)]
    initial_file: Option<PathBuf>,
}
#[derive(Deserialize)]
struct Match {
    plan: PathBuf,
    quantity: String,
    valid_time: String,
    #[serde(default)]
    precipitation_observation: Option<PathBuf>,
    members: Vec<Member>,
}
fn valid_time(file: &WrfFile, expected: &str) -> Result<()> {
    let times = file.times()?;
    if times.len() != 1 || instant(&times[0].replace('_', "T"))? != instant(expected)? {
        return Err(err(
            "forecast must contain exactly the requested valid time",
        ));
    }
    Ok(())
}
fn check_grid(file: &WrfFile, plan: &Plan) -> Result<()> {
    if file.nx != plan.nx
        || file.ny != plan.ny
        || geometry_hash(&field(file, "XLAT")?, &field(file, "XLONG")?) != plan.grid_geometry_sha256
    {
        return Err(err(
            "forecast coordinates differ from the frozen matching grid",
        ));
    }
    Ok(())
}
fn bilinear(values: &[f64], nx: usize, ny: usize, x: f64, y: f64) -> Result<f64> {
    if values.len() != nx * ny
        || nx == 0
        || ny == 0
        || !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x > (nx - 1) as f64
        || y > (ny - 1) as f64
    {
        return Err(err("bilinear position or field is outside the frozen grid"));
    }
    let i0 = (x.floor() as usize).min(nx.saturating_sub(2));
    let j0 = (y.floor() as usize).min(ny.saturating_sub(2));
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let (tx, ty) = (x - i0 as f64, y - j0 as f64);
    Ok(values[j0 * nx + i0] * (1.0 - tx) * (1.0 - ty)
        + values[j0 * nx + i1] * tx * (1.0 - ty)
        + values[j1 * nx + i0] * (1.0 - tx) * ty
        + values[j1 * nx + i1] * tx * ty)
}
fn rain_total(file: &WrfFile) -> Result<Vec<f64>> {
    if file.global_attr_f64("BUCKET_MM").unwrap_or(-1.0) > 0.0 {
        return Err(err(
            "bucketed rainfall requires the bucket counters and is not supported by this matcher",
        ));
    }
    let nc = field(file, "RAINNC")?;
    let c = field(file, "RAINC")?;
    let values = nc.iter().zip(c).map(|(a, b)| a + b).collect::<Vec<_>>();
    if values.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(err("forecast precipitation total is negative or nonfinite"));
    }
    Ok(values)
}
fn accumulation(end: &[f64], start: &[f64]) -> Result<Vec<f64>> {
    if end.len() != start.len() {
        return Err(err("precipitation endpoints have different shapes"));
    }
    let values = end
        .iter()
        .zip(start)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    if values.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(err(
            "forecast precipitation accumulator decreased or became nonfinite",
        ));
    }
    Ok(values)
}
fn check_seam(quantity: &str, values: &[f64]) -> Result<()> {
    let (low, high) = rw_obs::seam::seam_bounds(quantity)
        .ok_or_else(|| err("quantity has no registered physical seam bounds"))?;
    if values
        .iter()
        .any(|value| value.is_finite() && (*value < low || *value > high))
    {
        return Err(err(format!(
            "{quantity} leaves the declared physical seam bounds [{low},{high}]"
        )));
    }
    Ok(())
}
fn remap(plan: &Remap, values: &[f64], valid: &[bool]) -> Result<(Vec<f64>, Vec<bool>)> {
    let count = plan.destination_shape.0 * plan.destination_shape.1;
    let mut output = vec![0.0; count];
    let mut mask = vec![false; count];
    apply_plan(
        Method::CellAverage,
        &plan.source_index,
        &plan.reachable,
        plan.source_shape,
        plan.destination_shape,
        values,
        valid,
        &mut output,
        &mut mask,
    )?;
    Ok((output, mask))
}
fn matching(request: &Match, out: &Path) -> Result<Value> {
    matching_with_archive(request, out, None)
}
fn matching_with_archive(
    request: &Match,
    out: &Path,
    archive: Option<&diagnostics::Archive>,
) -> Result<Value> {
    let plan_bytes = std::fs::read(&request.plan)?;
    let plan: Plan = serde_json::from_slice(&plan_bytes)?;
    if plan.schema != PLAN_SCHEMA {
        return Err(err("unknown frozen matching plan"));
    }
    check_reference(&plan.grid)?;
    check_reference(&plan.surface)?;
    check_reference(&plan.precipitation_geometry)?;
    if request.members.is_empty() {
        return Err(err("matching needs at least one member"));
    }
    let mut ids = BTreeSet::new();
    for member in &request.members {
        if member.id.is_empty() || member.id.contains(['\t', '\n', '\r']) || !ids.insert(&member.id)
        {
            return Err(err(
                "member identities must be distinct, nonempty TSV labels",
            ));
        }
    }
    let end = seam_time(instant(&request.valid_time)?);
    let mut source_receipts = Vec::new();
    let mut observation_receipts = vec![serde_json::to_value(&plan.surface)?];
    let is_precip = request.quantity == "precipitation_accumulation";
    let (units, diagnostic) =
        match request.quantity.as_str() {
            "temperature_2m" => ("K", "t2"),
            "wind_speed_10m" => ("m s-1", "wspd10"),
            "precipitation_accumulation" => ("mm", ""),
            _ => return Err(err(
                "matcher supports temperature_2m, wind_speed_10m and precipitation_accumulation",
            )),
        };
    let mut sample_ids = Vec::new();
    let mut observed = Vec::new();
    let mut observation_mask = Vec::new();
    let mut observation_times = Vec::new();
    let mut precipitation_start = None;
    if is_precip {
        let path = request
            .precipitation_observation
            .as_ref()
            .ok_or_else(|| err("precipitation matching needs an observation pack"))?;
        let meta = verify(path)?;
        let frame = read_frame(path)?;
        if meta.valid_time != end || meta.geometry_sha256 != plan.precipitation_geometry_sha256 {
            return Err(err(
                "precipitation observation time or geometry differs from the frozen request",
            ));
        }
        precipitation_start = Some(meta.accumulation_start.clone());
        observation_receipts.push(serde_json::to_value(reference(path)?)?);
        let (values, valid) = if plan.precipitation_direction == "observation-to-model" {
            remap(&plan.remap, &frame.values, &frame.valid)?
        } else {
            (frame.values, frame.valid)
        };
        if values.len() != plan.precipitation_target_mask.len() {
            return Err(err("precipitation remap target shape differs"));
        }
        for index in 0..values.len() {
            if plan.precipitation_target_mask[index] {
                sample_ids.push(format!("{end}:cell:{index}"));
                observed.push(values[index]);
                observation_mask.push(valid[index]);
            }
        }
    } else {
        let record = surface(&plan.surface.path)?;
        if !record.valid_times.contains(&end) {
            return Err(err(
                "surface timeline does not contain the requested valid time",
            ));
        }
        for pos in plan.stations.iter().filter(|pos| pos.keep) {
            let report = record
                .reports
                .iter()
                .find(|r| r.station_id == pos.station_id && r.valid_time == end);
            let value = report
                .and_then(|r| r.values.get(&request.quantity))
                .copied();
            sample_ids.push(format!("{end}:station:{}", pos.station_id));
            observed.push(value.unwrap_or(0.0));
            observation_mask.push(
                value.is_some_and(|x| x.is_finite())
                    && report.is_some_and(|r| {
                        !r.flags.iter().any(|flag| flag.contains(&request.quantity))
                    }),
            );
            observation_times.push(json!({"station_id":pos.station_id,"observation_time":report.map(|r|&r.observation_time)}));
        }
    }
    if sample_ids.is_empty() {
        return Err(err("no locations survive the frozen matching plan"));
    }
    let mut columns = Vec::new();
    for (member_index, member) in request.members.iter().enumerate() {
        let raw_values = if let Some(archive) = archive {
            let (values, receipt) = archive.field(
                member_index,
                &plan,
                &request.quantity,
                precipitation_start.as_deref(),
            )?;
            source_receipts.push(receipt);
            values
        } else {
            let file = WrfFile::open(&member.file)?;
            valid_time(&file, &end)?;
            check_grid(&file, &plan)?;
            source_receipts.push(json!({"member_id":member.id,"frame":reference(&member.file)?}));
            if is_precip {
                let initial = member.initial_file.as_ref().ok_or_else(|| {
                    err("each precipitation member needs its own accumulation-start frame")
                })?;
                let first = WrfFile::open(initial)?;
                valid_time(&first, precipitation_start.as_ref().unwrap())?;
                check_grid(&first, &plan)?;
                source_receipts.push(
                    json!({"member_id":member.id,"accumulation_start_frame":reference(initial)?}),
                );
                accumulation(&rain_total(&file)?, &rain_total(&first)?)?
            } else {
                let output = getvar(&file, diagnostic, Some(0), &ComputeOpts::default())?;
                let core_units = if diagnostic == "wspd10" { "m/s" } else { units };
                if output.shape != vec![plan.ny, plan.nx] || output.units != core_units {
                    return Err(err(format!(
                        "science-core diagnostic {diagnostic} units/shape differ: {}",
                        output.units
                    )));
                }
                output.data
            }
        };
        check_seam(&request.quantity, &raw_values)?;
        let values = if is_precip {
            let values = if plan.precipitation_direction == "model-to-observation" {
                remap(&plan.remap, &raw_values, &plan.grid_interior)?.0
            } else {
                raw_values
            };
            values
                .into_iter()
                .zip(&plan.precipitation_target_mask)
                .filter_map(|(value, keep)| keep.then_some(value))
                .collect::<Vec<_>>()
        } else {
            plan.stations
                .iter()
                .filter(|pos| pos.keep)
                .map(|pos| bilinear(&raw_values, plan.nx, plan.ny, pos.x, pos.y))
                .collect::<Result<Vec<_>>>()?
        };
        if values.len() != sample_ids.len() {
            return Err(err("member matched sample count differs"));
        }
        columns.push(values);
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(std::fs::File::create(out)?);
    write!(writer, "sample_id\tweight\tobserved")?;
    for member in &request.members {
        write!(writer, "\t{}", member.id)?;
    }
    writeln!(writer)?;
    for index in 0..sample_ids.len() {
        write!(
            writer,
            "{}\t1\t{}",
            sample_ids[index],
            if observation_mask[index] {
                observed[index].to_string()
            } else {
                "NaN".into()
            }
        )?;
        for column in &columns {
            write!(
                writer,
                "\t{}",
                if column[index].is_finite() {
                    column[index].to_string()
                } else {
                    "NaN".into()
                }
            )?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(
        json!({"schema":"gpuwm.ensemble-match-receipt.v1","plan":{"path":request.plan,"sha256":hex_sha256(&plan_bytes)},
        "quantity":request.quantity,"units":units,"valid_time":end,"accumulation_start":precipitation_start,"member_ids":request.members.iter().map(|m|&m.id).collect::<Vec<_>>(),
        "forecast_field_reader":if archive.is_some(){diagnostics::CONTRACT}else{"wrf-core one-frame history"},
        "samples":sample_ids.len(),"observed_samples":observation_mask.iter().filter(|x|**x).count(),"source_receipts":source_receipts,"observation_receipts":observation_receipts,
        "observation_times":observation_times,"matched":reference(out)?,"parameters":plan.parameters}),
    )
}
fn run() -> Result<()> {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(err(
            "usage: rw_ensemble_match prepare|match|match-diagnostics REQUEST.json OUTPUT.json|OUTPUT.tsv",
        ));
    }
    let body = std::fs::read(&args[1])?;
    let out = Path::new(&args[2]);
    let mut receipt = match args[0].as_str() {
        "prepare" => {
            let plan = prepare(&serde_json::from_slice(&body)?)?;
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(out, serde_json::to_vec_pretty(&plan)?)?;
            json!({"schema":PLAN_SCHEMA,"plan":reference(out)?,"stations_kept":plan.stations.iter().filter(|s|s.keep).count(),
                "stations_dropped":plan.stations.iter().filter(|s|!s.keep).count(),"precipitation_target_cells":plan.precipitation_target_mask.iter().filter(|x|**x).count(),"parameters":plan.parameters})
        }
        "match" => matching(&serde_json::from_slice(&body)?, out)?,
        "match-diagnostics" => {
            let (request, archive) = diagnostics::resolve(&serde_json::from_slice(&body)?)?;
            matching_with_archive(&request, out, Some(&archive))?
        }
        _ => return Err(err("unknown matching operation")),
    };
    receipt["request"] = json!({"path":args[1],"sha256":hex_sha256(&body)});
    receipt["executable"] = serde_json::to_value(reference(&std::env::current_exe()?)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("rw_ensemble_match: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    use rw_obs::pack::{write_pack, PayloadBuilder, GRID_SCHEMA};
    use rw_obs::precipitation::{write_frame, Frame, Grid, Mask, Meta};
    use rw_obs::seam::Provenance;
    #[test]
    fn bilinear_matches_registered_expression_and_boundaries() {
        let values = [1.0, 3.0, 5.0, 7.0];
        assert_eq!(bilinear(&values, 2, 2, 0.25, 0.5).unwrap(), 3.5);
        assert_eq!(bilinear(&values, 2, 2, 1.0, 1.0).unwrap(), 7.0);
        assert!(bilinear(&values, 2, 2, 1.01, 0.0).is_err());
    }
    #[test]
    fn precipitation_is_endpoint_difference_without_clipping_resets() {
        assert_eq!(
            accumulation(&[5.0, 9.0], &[1.0, 2.0]).unwrap(),
            vec![4.0, 7.0]
        );
        assert!(accumulation(&[1.0], &[2.0]).is_err());
        assert!(accumulation(&[f64::NAN], &[0.0]).is_err());
        assert!(check_seam("temperature_2m", &[20.0]).is_err());
        assert!(check_seam("temperature_2m", &[293.15]).is_ok());
        assert!(check_seam("wind_speed_10m", &[200.0]).is_err());
    }
    #[test]
    fn shared_remap_keeps_missing_separate_from_dry() {
        let plan = Remap {
            source_index: vec![0, 0, 1],
            reachable: vec![true, true],
            source_shape: (1, 3),
            destination_shape: (1, 2),
            max_distance_m: 1000.0,
            max_used_distance_m: 0.0,
        };
        let (values, valid) = remap(&plan, &[2.0, 4.0, 0.0], &[true, true, false]).unwrap();
        assert_eq!(values, vec![3.0, 0.0]);
        assert_eq!(valid, vec![true, false]);
    }

    fn fixture_directory() -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ensemble-match-test-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
    fn fixture_projection() -> ProjectedGrid {
        ProjectedGrid::new(GridSpec {
            kind: ProjectionKind::Lambert,
            ref_lat: 38.0,
            ref_lon: -98.0,
            truelat1: 30.0,
            truelat2: 60.0,
            stand_lon: -98.0,
            dx: 3000.0,
            dy: 3000.0,
            e_we: 9,
            e_sn: 9,
            known_x: 1.0,
            known_y: 1.0,
            moad_cen_lat: 38.0,
            moad_cen_lon: -98.0,
            lat_deg: vec![],
            lon0_deg: 0.0,
            dlon_deg: 0.0,
        })
        .unwrap()
    }
    fn write_fixture_forecast(path: &Path, hour: usize, member: usize) {
        let mut schema = Schema::new(NcFormat::Offset64);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let text = schema.def_dim("DateStrLen", 19, false).unwrap();
        let ny = schema.def_dim("south_north", 8, false).unwrap();
        let nx = schema.def_dim("west_east", 8, false).unwrap();
        schema
            .put_global_attr("MAP_PROJ", AttrValue::Ints(vec![1]))
            .unwrap();
        for (name, value) in [
            ("DX", 3000.0),
            ("DY", 3000.0),
            ("TRUELAT1", 30.0),
            ("TRUELAT2", 60.0),
            ("STAND_LON", -98.0),
            ("CEN_LAT", 38.0),
            ("CEN_LON", -98.0),
            ("MOAD_CEN_LAT", 38.0),
        ] {
            schema
                .put_global_attr(name, AttrValue::Doubles(vec![value]))
                .unwrap();
        }
        let times = schema
            .def_var("Times", NcType::Char, &[time, text])
            .unwrap();
        let mut ids = BTreeMap::new();
        for (name, units) in [
            ("XLAT", "degrees_north"),
            ("XLONG", "degrees_east"),
            ("HGT", "m"),
            ("LANDMASK", ""),
            ("T2", "K"),
            ("U10", "m s-1"),
            ("V10", "m s-1"),
            ("RAINNC", "mm"),
            ("RAINC", "mm"),
        ] {
            let id = schema
                .def_var(name, NcType::Float, &[time, ny, nx])
                .unwrap();
            schema
                .put_var_attr(id, "units", AttrValue::Text(units.into()))
                .unwrap();
            ids.insert(name, id);
        }
        let projection = fixture_projection();
        let mut writer = NcWriter::create(path, schema).unwrap();
        writer
            .write_record(
                0,
                times,
                VarData::Char(format!("2000-01-01_{hour:02}:00:00").as_bytes()),
            )
            .unwrap();
        for (name, id) in ids {
            let values = (0..64)
                .map(|k| {
                    let (lat, lon) =
                        projection.ij_to_latlon((k % 8 + 1) as f64, (k / 8 + 1) as f64);
                    (match name {
                        "XLAT" => lat,
                        "XLONG" => lon,
                        "HGT" => 20.0,
                        "LANDMASK" => 1.0,
                        "T2" => 290.0 + 2.0 * member as f64 + (k % 8) as f64 + 2.0 * (k / 8) as f64,
                        "U10" => 3.0,
                        "V10" => 4.0,
                        "RAINNC" => {
                            if hour == 0 {
                                1.0
                            } else {
                                5.0 + 2.0 * member as f64
                            }
                        }
                        "RAINC" => {
                            if hour == 0 {
                                0.5
                            } else {
                                1.0 + 0.5 * member as f64
                            }
                        }
                        _ => unreachable!(),
                    }) as f32
                })
                .collect::<Vec<_>>();
            writer.write_record(0, id, VarData::F32(&values)).unwrap();
        }
        writer.finish().unwrap();
    }

    fn write_fixture_diagnostic(
        path: &Path,
        grid: &Path,
        member_id: u64,
        seed: u64,
        temperature_offset: f32,
        rain: f32,
        wrong: &str,
    ) -> Value {
        let grid = WrfFile::open(grid).unwrap();
        let mut lat = field(&grid, "XLAT")
            .unwrap()
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        let lon = field(&grid, "XLONG")
            .unwrap()
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        if wrong == "geometry" {
            lat[0] += 0.01;
        }
        let geometry = hex_sha256(
            &lat.iter()
                .chain(&lon)
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let provenance = json!({"member_id":member_id,"seed":seed,"source":"analytical fixture"});
        let mut schema = Schema::new(NcFormat::Cdf5);
        let member = schema.def_dim("member", 1, false).unwrap();
        let ny = schema.def_dim("south_north", 8, false).unwrap();
        let nx = schema.def_dim("west_east", 8, false).unwrap();
        for (name, value) in [
            (
                "member_diagnostic_contract",
                diagnostics::CONTRACT.to_string(),
            ),
            (
                "valid_time",
                if wrong == "time" {
                    "2000-01-01T02:00:00"
                } else {
                    "2000-01-01T01:00:00+00:00"
                }
                .into(),
            ),
            (
                "precipitation_accumulation_start",
                "2000-01-01T00:00:00".into(),
            ),
            ("geometry_sha256", geometry.clone()),
            ("member_provenance", provenance.to_string()),
        ] {
            schema
                .put_global_attr(name, AttrValue::Text(value))
                .unwrap();
        }
        let id_var = schema
            .def_var("member_id", NcType::UInt64, &[member])
            .unwrap();
        let seed_var = schema
            .def_var("member_seed", NcType::UInt64, &[member])
            .unwrap();
        let mut variables = BTreeMap::new();
        for (name, units) in [
            ("XLAT", "degrees_north"),
            ("XLONG", "degrees_east"),
            ("T2", "K"),
            ("U10", "m s-1"),
            ("V10", "m s-1"),
            ("RAIN_TOTAL", "mm"),
        ] {
            let dims = if name.starts_with("XL") {
                vec![ny, nx]
            } else {
                vec![member, ny, nx]
            };
            let id = schema.def_var(name, NcType::Float, &dims).unwrap();
            schema
                .put_var_attr(
                    id,
                    "units",
                    AttrValue::Text(
                        if wrong == "units" && name == "T2" {
                            "C"
                        } else {
                            units
                        }
                        .into(),
                    ),
                )
                .unwrap();
            variables.insert(name, id);
        }
        let mut writer = NcWriter::create(path, schema).unwrap();
        writer
            .write_var(
                id_var,
                VarData::U64(&[if wrong == "member" {
                    member_id + 1
                } else {
                    member_id
                }]),
            )
            .unwrap();
        writer
            .write_var(
                seed_var,
                VarData::U64(&[if wrong == "seed" { seed - 1 } else { seed }]),
            )
            .unwrap();
        for (name, id) in variables {
            let values = match name {
                "XLAT" => lat.clone(),
                "XLONG" => lon.clone(),
                "T2" => (0..64)
                    .map(|i| 290.0 + temperature_offset + (i % 8) as f32 + 2.0 * (i / 8) as f32)
                    .collect(),
                "U10" | "V10" => vec![1.0; 64],
                "RAIN_TOTAL" => vec![rain; 64],
                _ => unreachable!(),
            };
            writer.write_var(id, VarData::F32(&values)).unwrap();
        }
        writer.finish().unwrap();
        json!({"path":path.file_name().unwrap().to_str().unwrap(),"bytes":std::fs::metadata(path).unwrap().len(),
            "sha256":reference(path).unwrap().sha256,"member_id":member_id,"seed":seed,"grid_id":1,"episode":0,
            "valid_time":"2000-01-01T01:00:00","geometry_sha256":geometry})
    }
    #[test]
    fn native_forecast_observation_matching_checks_times_members_masks_and_intervals() {
        let root = fixture_directory();
        let initial = root.join("initial.nc");
        let a = root.join("member-a.nc");
        let b = root.join("member-b.nc");
        write_fixture_forecast(&initial, 0, 0);
        write_fixture_forecast(&a, 1, 0);
        write_fixture_forecast(&b, 1, 1);
        let source = root.join("observed.raw");
        std::fs::write(&source, b"analytical observation fixture").unwrap();
        let provenance = Provenance::new(
            "analytical-fixture",
            "native-matcher-test",
            source.to_string_lossy(),
            hex_sha256(&std::fs::read(&source).unwrap()),
            "2000-01-01T01:00:00",
        );
        let file = WrfFile::open(&initial).unwrap();
        let lat = field(&file, "XLAT").unwrap();
        let lon = field(&file, "XLONG").unwrap();
        let mut builder = PayloadBuilder::new();
        builder.push_f64("latitude", &lat, vec![8, 8]);
        builder.push_f64("longitude", &lon, vec![8, 8]);
        let (payload, arrays) = builder.finish();
        let geometry_digest = payload_digest(&payload);
        let geometry = root.join("geometry.obspack");
        write_pack(&geometry,&json!({"schema":GEO_SCHEMA,"status":"READY","content_sha256":geometry_digest,"arrays":arrays}),&payload).unwrap();
        let projection = fixture_projection();
        let (station_lat, station_lon) = projection.ij_to_latlon(3.25, 4.5);
        let surface_path = root.join("surface.json");
        std::fs::write(&surface_path,serde_json::to_vec(&json!({"schema":"gpuwm-obs.asos-surface.v2","status":"READY","provenance":provenance,
            "station_table_sha256":"a".repeat(64),"valid_times":["2000-01-01T01:00:00"],"match_seconds":600,"min_report_rate":0.8,"max_screen_rate":0.05,
            "stations":[{"station_id":"synthetic","latitude":station_lat,"longitude":station_lon,"elevation_m":20.0}],
            "reports":[{"station_id":"synthetic","valid_time":"2000-01-01T01:00:00","observation_time":"2000-01-01T00:55:00",
                "values":{"temperature_2m":300.0,"wind_speed_10m":6.0},"flags":[]}]})).unwrap()).unwrap();
        let plan = prepare(&Prepare {
            grid: initial.clone(),
            surface: surface_path,
            precipitation_geometry: geometry,
            precipitation_direction: "observation-to-model".into(),
            interior_rim_m: 0.0,
            boundary_rows: 0,
            elevation_tolerance_m: 100.0,
            max_regrid_distance_m: 1.0,
        })
        .unwrap();
        assert_eq!(plan.stations.iter().filter(|s| s.keep).count(), 1);
        assert_eq!(
            plan.precipitation_target_mask
                .iter()
                .filter(|x| **x)
                .count(),
            64
        );
        let plan_path = root.join("plan.json");
        std::fs::write(&plan_path, serde_json::to_vec(&plan).unwrap()).unwrap();
        let members = vec![
            Member {
                id: "member-a".into(),
                file: a,
                initial_file: Some(initial.clone()),
            },
            Member {
                id: "member-b".into(),
                file: b,
                initial_file: Some(initial),
            },
        ];
        let mut request = Match {
            plan: plan_path,
            quantity: "temperature_2m".into(),
            valid_time: "2000-01-01T01:00:00".into(),
            precipitation_observation: None,
            members,
        };
        let out = root.join("matched.tsv");
        let receipt = matching(&request, &out).unwrap();
        assert_eq!(receipt["observed_samples"], 1);
        let text = std::fs::read_to_string(&out).unwrap();
        let row = text.lines().nth(1).unwrap().split('\t').collect::<Vec<_>>();
        assert_eq!(row[2], "300");
        assert!((row[3].parse::<f64>().unwrap() - 299.25).abs() < 0.001);
        assert!((row[4].parse::<f64>().unwrap() - 301.25).abs() < 0.001);
        request.quantity = "wind_speed_10m".into();
        matching(&request, &out).unwrap();
        assert!(std::fs::read_to_string(&out)
            .unwrap()
            .lines()
            .nth(1)
            .unwrap()
            .ends_with("\t6\t5\t5"));
        let precipitation = root.join("rain.obspack");
        let mut observed = vec![true; 64];
        observed[0] = false;
        write_frame(
            &precipitation,
            Frame {
                values: vec![2.0; 64],
                valid: observed,
                meta: Meta {
                    schema: GRID_SCHEMA.into(),
                    status: "READY".into(),
                    quantity: "precipitation_accumulation".into(),
                    units: "mm".into(),
                    valid_time: "2000-01-01T01:00:00".into(),
                    accumulation_start: "2000-01-01T00:00:00".into(),
                    accumulation_seconds: 3600,
                    accumulation_hours: 1.0,
                    provenance,
                    geometry_sha256: geometry_digest,
                    grid: Grid {
                        kind: "analytical-fixture".into(),
                        nx: 8,
                        ny: 8,
                        source_nx: 8,
                        source_ny: 8,
                        i_start: 0,
                        j_start: 0,
                    },
                    sentinels: Mask {
                        masked_cells: 0,
                        zero_cells: 0,
                        positive_cells: 0,
                        observed_fraction: 0.0,
                    },
                    value_min_mm: 0.0,
                    value_max_mm: 0.0,
                    arrays: BTreeMap::new(),
                    payload_bytes: 0,
                    content_sha256: String::new(),
                    source_frames: vec![],
                },
            },
        )
        .unwrap();
        request.quantity = "precipitation_accumulation".into();
        request.precipitation_observation = Some(precipitation);
        let receipt = matching(&request, &out).unwrap();
        assert_eq!(receipt["samples"], 64);
        assert_eq!(receipt["observed_samples"], 63);
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(text.lines().nth(1).unwrap().ends_with("\tNaN\t4.5\t7"));
        assert!(text.lines().nth(2).unwrap().ends_with("\t2\t4.5\t7"));
        // The production surface archive has no WRF projection/header fields.
        // It must use the already frozen plan and original diagnostic planes.
        let diag_a = root.join("diagnostic-a.nc");
        let diag_b = root.join("diagnostic-b.nc");
        let seed_a = u64::MAX;
        let seed_b = 9_007_199_254_740_993u64;
        let a_entry = write_fixture_diagnostic(&diag_a, &plan.grid.path, 0, seed_a, 0.0, 2.25, "");
        let b_entry = write_fixture_diagnostic(&diag_b, &plan.grid.path, 19, seed_b, 2.0, 4.25, "");
        let manifest_path = root.join("diagnostics-manifest.json");
        let mut manifest = json!({"schema":diagnostics::CONTRACT,"member_order":[0,19],
            "member_metadata":[{"member_id":0,"seed":seed_a,"source":"analytical fixture"},
                {"member_id":19,"seed":seed_b,"source":"analytical fixture"}],
            "fields":{"T2":"K","U10":"m s-1","V10":"m s-1","RAIN_TOTAL":"mm"},
            "precipitation_accumulation_start":"2000-01-01T00:00:00Z","forecast_volume_fields":false,
            "files":[a_entry,b_entry],"unavailable":[]});
        let mut diagnostic_request = json!({"plan":request.plan,"quantity":"temperature_2m",
            "valid_time":"2000-01-01T01:00:00+00:00","precipitation_observation":request.precipitation_observation,
            "archive":{"root":root,"manifest":{"path":manifest_path,"sha256":""},"grid_id":1,"episode":0},
            "members":[{"id":"control-0","member_id":0,"seed":seed_a},{"id":"member-19","member_id":19,"seed":seed_b}]});
        let repin = |manifest: &Value, request: &mut Value| {
            std::fs::write(&manifest_path, serde_json::to_vec(manifest).unwrap()).unwrap();
            request["archive"]["manifest"]["sha256"] =
                json!(reference(&manifest_path).unwrap().sha256);
        };
        let evaluate = |request: &Value| -> Result<Value> {
            let (request, archive) =
                diagnostics::resolve(&serde_json::from_value(request.clone())?)?;
            matching_with_archive(&request, &out, Some(&archive))
        };
        repin(&manifest, &mut diagnostic_request);
        let receipt = evaluate(&diagnostic_request).unwrap();
        assert_eq!(receipt["source_receipts"][1]["seed"].as_u64(), Some(seed_b));
        let text = std::fs::read_to_string(&out).unwrap();
        let row = text.lines().nth(1).unwrap().split('\t').collect::<Vec<_>>();
        assert!((row[3].parse::<f64>().unwrap() - 299.25).abs() < 0.001);
        assert!((row[4].parse::<f64>().unwrap() - 301.25).abs() < 0.001);
        diagnostic_request["quantity"] = json!("wind_speed_10m");
        evaluate(&diagnostic_request).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        let row = text.lines().nth(1).unwrap().split('\t').collect::<Vec<_>>();
        assert!((row[3].parse::<f64>().unwrap() - 1.4142135381698608).abs() < 1e-14);
        diagnostic_request["quantity"] = json!("precipitation_accumulation");
        evaluate(&diagnostic_request).unwrap();
        assert!(std::fs::read_to_string(&out)
            .unwrap()
            .lines()
            .nth(1)
            .unwrap()
            .ends_with("\tNaN\t2.25\t4.25"));
        diagnostic_request["quantity"] = json!("temperature_2m");
        for (wrong, message) in [
            ("seed", "uint64 member_seed"),
            ("member", "uint64 member_id"),
            ("units", "units"),
            ("geometry", "coordinates"),
            ("time", "time differs"),
        ] {
            manifest["files"][1] =
                write_fixture_diagnostic(&diag_b, &plan.grid.path, 19, seed_b, 2.0, 4.25, wrong);
            repin(&manifest, &mut diagnostic_request);
            assert!(
                evaluate(&diagnostic_request)
                    .unwrap_err()
                    .to_string()
                    .contains(message),
                "{wrong}"
            );
        }
        manifest["files"][1] =
            write_fixture_diagnostic(&diag_b, &plan.grid.path, 19, seed_b, 2.0, 4.25, "");
        repin(&manifest, &mut diagnostic_request);
        diagnostic_request["members"][1]["seed"] = json!(seed_b - 1);
        assert!(evaluate(&diagnostic_request)
            .unwrap_err()
            .to_string()
            .contains("seed differs"));
        diagnostic_request["members"][1]["seed"] = json!(seed_b);
        manifest["unavailable"] =
            json!([{"member_id":19,"grid_id":1,"episode":0,"valid_time":"2000-01-01T01:00:00Z"}]);
        repin(&manifest, &mut diagnostic_request);
        assert!(evaluate(&diagnostic_request)
            .unwrap_err()
            .to_string()
            .contains("explicitly unavailable"));
        manifest["unavailable"] = json!([]);
        manifest["files"].as_array_mut().unwrap().pop();
        repin(&manifest, &mut diagnostic_request);
        assert!(evaluate(&diagnostic_request)
            .unwrap_err()
            .to_string()
            .contains("missing diagnostic member"));
        std::fs::write(&manifest_path, b"{}").unwrap();
        assert!(evaluate(&diagnostic_request)
            .unwrap_err()
            .to_string()
            .contains("manifest digest"));
        request.members[1].id = "member-a".into();
        assert!(matching(&request, &out)
            .unwrap_err()
            .to_string()
            .contains("distinct"));
        request.members[1].id = "member-b".into();
        request.valid_time = "2000-01-01T02:00:00".into();
        assert!(matching(&request, &out).is_err());
    }
}
