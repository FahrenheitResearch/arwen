//! `rw_zarr soiltemp-climatology`: the default deep-soil temperature dataset
//! of the WPS_GEOG tree, produced from a public ERA5 climatology.
//!
//! WRF and MPAS read one deep-soil temperature, SOILTEMP, a 1-degree global
//! field of annual-mean air temperature reduced to a sea-level reference, and
//! take TMN = SOILTEMP - 0.0065 * HGT at each land cell.  NCAR's
//! `soiltemp_1deg` is ISLSCP-I's 1987-88 ECMWF analysis mean, whose licence
//! no source states, so the default tree holds it out.  This command builds
//! the same quantity from an hourly ERA5 climatology (WeatherBench 2's
//! 1990-2019 climatology on Google Cloud, under the Licence to Use Copernicus
//! Products) in the SAME on-disk layout -- 1-degree cells centred on
//! -179.5..179.5 / -89.5..89.5, two 180x180 tiles with a 3-cell border,
//! big-endian unsigned 16-bit words at a 0.01 K scale, 0 = no land -- so every
//! reader of the old dataset (the WRF static builder, rw_mpas_static's native
//! fixed-layout reader) reads this one unchanged.
//!
//! The reduction is the original's: the annual mean of 2 m temperature over
//! every hour and day of the climatology (day 366 weighted by the request's
//! leap-day weight), plus lapse * surface height, with the surface height the
//! source's own `geopotential_at_surface / g`.  The 1-degree cell value is the
//! land-weighted, area-conservative mean of the source cells it overlaps
//! (weight = overlap of sin(latitude) x overlap of longitude x land fraction),
//! so a coastal land cell carries land air temperature, not the sea's.  A cell
//! whose land fraction is at or below the request's floor is written 0, the
//! dataset's declared missing value, which is how the original marks ocean.
//!
//! Every read is a whole chunk along the day-of-year axis, read concurrently
//! (`GPUWM_RW_ZARR_WORKERS` bounds the transfers, default 8); the sums are
//! f64 and reduced in block order, so the output does not depend on the
//! worker count.
use super::{dimensions, fail, http_store, numbers, open, read_workers, Result, SourceArray};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::Path,
    sync::{atomic::{AtomicUsize, Ordering}, Arc, Mutex},
};
use zarrs::storage::ReadableStorage;

pub const REQUEST_SCHEMA: &str = "arwen.geog-soiltemp-climatology.request.v1";
pub const RECEIPT_SCHEMA: &str = "arwen.geog-soiltemp-climatology.v1";
/// The fixed layout every reader of `soiltemp_1deg` expects (the MPAS native
/// reader hard-codes it: rw-mpas `GeogDataset::open_native_soil_temperature`).
const NX: usize = 360;
const NY: usize = 180;
const TILE: usize = 180;
const BORDER: usize = 3;
const SCALE: f64 = 0.01;

#[derive(Deserialize)]
struct Request {
    schema: String,
    climatology_store: String,
    temperature: String,
    temperature_units: String,
    static_store: String,
    geopotential: String,
    land_sea_mask: String,
    lapse_rate_k_per_m: f64,
    gravity_m_s2: f64,
    leap_day_weight: f64,
    land_fraction_min: f64,
    description: String,
}

fn store(url: &str) -> Result<ReadableStorage> {
    Ok(if url.starts_with("https://") {
        Arc::new(http_store::HttpStore::new(url)?)
    } else {
        Arc::new(zarrs::filesystem::FilesystemStore::new(url)?)
    })
}

fn coordinate_values(store: &ReadableStorage, name: &str) -> Result<Vec<f64>> {
    let array = open(store, name)?;
    if array.shape().len() != 1 {
        return fail(format!("{name} coordinate must be one dimensional"));
    }
    let values = numbers(&array, &[0..array.shape()[0]])?;
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return fail(format!("{name} coordinate is empty or nonfinite"));
    }
    Ok(values)
}

/// The chunk length of `axis`, from Zarr v2 `chunks` or v3 `chunk_grid`.
fn chunk_len(array: &SourceArray, axis: usize) -> Option<u64> {
    let metadata = serde_json::to_value(array.metadata()).ok()?;
    let chunks = metadata.get("chunks").or_else(|| {
        metadata.get("chunk_grid")?.get("configuration")?.get("chunk_shape")
    })?;
    chunks.as_array()?.get(axis)?.as_u64().filter(|n| *n > 0)
}

/// `values` (laid out over `dims`, the two spatial axes in either order)
/// as a latitude-major plane.
fn lat_major(values: &[f64], dims: &[String], ny: usize, nx: usize) -> Result<Vec<f64>> {
    match (dims[0].as_str(), dims[1].as_str()) {
        ("latitude", "longitude") => Ok(values.to_vec()),
        ("longitude", "latitude") => {
            let mut out = vec![0.0; ny * nx];
            for i in 0..nx {
                for j in 0..ny {
                    out[j * nx + i] = values[i * ny + j];
                }
            }
            Ok(out)
        }
        _ => fail(format!("spatial dimensions must be latitude and longitude, found {dims:?}")),
    }
}

/// Cell edges of a strictly monotonic latitude axis, clipped to the poles.
fn latitude_edges(lat: &[f64]) -> Result<Vec<(f64, f64)>> {
    let n = lat.len();
    if n < 2 {
        return fail("latitude needs at least two cells");
    }
    let ascending = lat[1] > lat[0];
    if lat.windows(2).any(|w| (w[1] > w[0]) != ascending || w[1] == w[0]) {
        return fail("latitude must be strictly monotonic");
    }
    let mid = |a: f64, b: f64| 0.5 * (a + b);
    Ok((0..n).map(|j| {
        let below = if j == 0 { lat[0] - (lat[1] - lat[0]) / 2.0 } else { mid(lat[j - 1], lat[j]) };
        let above = if j + 1 == n { lat[n - 1] + (lat[n - 1] - lat[n - 2]) / 2.0 } else { mid(lat[j], lat[j + 1]) };
        let (lo, hi) = if below < above { (below, above) } else { (above, below) };
        (lo.max(-90.0), hi.min(90.0))
    }).collect())
}

/// Overlap length of [a0,a1] with [b0,b1] on a 360-degree circle.
fn circular_overlap(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    let mut total = 0.0;
    for shift in [-720.0, -360.0, 0.0, 360.0, 720.0] {
        let lo = a0.max(b0 + shift);
        let hi = a1.min(b1 + shift);
        if hi > lo {
            total += hi - lo;
        }
    }
    total
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 2 {
        return fail("usage: rw_zarr soiltemp-climatology REQUEST.json OUTPUT_DIR");
    }
    let request: Request = serde_json::from_slice(&fs::read(&args[0])?)?;
    let receipt = produce(request, Path::new(&args[1]))?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

fn produce(request: Request, output: &Path) -> Result<Value> {
    if request.schema != REQUEST_SCHEMA {
        return fail(format!("request schema must be {REQUEST_SCHEMA}"));
    }
    if output.exists() {
        return fail(format!("output already exists: {}", output.display()));
    }
    for (name, value) in [("lapse_rate_k_per_m", request.lapse_rate_k_per_m),
                          ("gravity_m_s2", request.gravity_m_s2),
                          ("leap_day_weight", request.leap_day_weight),
                          ("land_fraction_min", request.land_fraction_min)] {
        if !value.is_finite() || value < 0.0 {
            return fail(format!("{name} must be finite and non-negative"));
        }
    }
    if request.gravity_m_s2 == 0.0 || request.land_fraction_min >= 1.0 {
        return fail("gravity must be positive and the land-fraction floor below 1");
    }
    let clim = store(&request.climatology_store)?;
    let stat = store(&request.static_store)?;

    // The temperature cube: [hour, dayofyear] leading, two spatial axes after.
    let temperature = open(&clim, &request.temperature)?;
    let dims = dimensions(&temperature)?;
    let shape = temperature.shape().to_vec();
    if dims.len() != 4 || shape.len() != 4 {
        return fail(format!("{} must have four dimensions", request.temperature));
    }
    let mut lead = [dims[0].as_str(), dims[1].as_str()];
    lead.sort_unstable();
    if lead != ["dayofyear", "hour"] {
        return fail(format!("{} must lead with hour and dayofyear, found {dims:?}", request.temperature));
    }
    if let Some(units) = temperature.attributes().get("units").and_then(Value::as_str) {
        if !units.trim().eq_ignore_ascii_case(&request.temperature_units) {
            return fail(format!("{} units {units:?} disagree with declared {:?}",
                                request.temperature, request.temperature_units));
        }
    }
    let doy_axis = dims.iter().position(|d| d == "dayofyear").unwrap();
    let doy = coordinate_values(&clim, "dayofyear")?;
    let hours = coordinate_values(&clim, "hour")?;
    if doy.len() as u64 != shape[doy_axis] || hours.len() as u64 != shape[1 - doy_axis] {
        return fail("dayofyear/hour coordinates disagree with the temperature shape");
    }
    if doy.iter().any(|d| *d < 1.0 || *d > 366.0 || d.fract() != 0.0) {
        return fail("dayofyear values must be whole days 1..366");
    }
    let lat = coordinate_values(&clim, "latitude")?;
    let lon = coordinate_values(&clim, "longitude")?;
    let spatial = dims[2..].to_vec();
    let (ny, nx) = (lat.len(), lon.len());
    let expected_spatial: Vec<u64> = spatial.iter()
        .map(|d| if d == "latitude" { ny as u64 } else { nx as u64 }).collect();
    if shape[2..] != expected_spatial[..] {
        return fail("latitude/longitude coordinates disagree with the temperature shape");
    }
    let plane = ny * nx;

    // The static fields must sit on exactly the climatology's grid.
    for (name, values) in [("latitude", &lat), ("longitude", &lon)] {
        let other = coordinate_values(&stat, name)?;
        if other.len() != values.len()
            || other.iter().zip(values.iter()).any(|(a, b)| (a - b).abs() > 1e-6) {
            return fail(format!("static store {name} differs from the climatology's"));
        }
    }
    let read_static = |name: &str| -> Result<Vec<f64>> {
        let array = open(&stat, name)?;
        let sdims = dimensions(&array)?;
        if sdims.len() != 2 {
            return fail(format!("{name} must be two dimensional"));
        }
        let ranges: Vec<_> = array.shape().iter().map(|n| 0..*n).collect();
        let values = numbers(&array, &ranges)?;
        if values.len() != plane || values.iter().any(|v| !v.is_finite()) {
            return fail(format!("{name} is not a complete finite field on the grid"));
        }
        lat_major(&values, &sdims, ny, nx)
    };
    let geopotential = read_static(&request.geopotential)?;
    let land = read_static(&request.land_sea_mask)?;
    if land.iter().any(|f| !(0.0..=1.0).contains(f)) {
        return fail(format!("{} must be a fraction in [0, 1]", request.land_sea_mask));
    }

    // The annual mean: blocks of whole chunks along the day-of-year axis.
    let block = chunk_len(&temperature, doy_axis).unwrap_or(30).min(shape[doy_axis]);
    let starts: Vec<u64> = (0..shape[doy_axis]).step_by(block as usize).collect();
    let weight_of = |d: f64| if d > 365.0 { request.leap_day_weight } else { 1.0 };
    let partial: Mutex<Vec<Option<Vec<f64>>>> = Mutex::new(vec![None; starts.len()]);
    let next = AtomicUsize::new(0);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let workers = read_workers(starts.len());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= starts.len() || failure.lock().unwrap().is_some() {
                    break;
                }
                let d0 = starts[k];
                let d1 = (d0 + block).min(shape[doy_axis]);
                let mut ranges: Vec<_> = shape.iter().map(|n| 0..*n).collect();
                ranges[doy_axis] = d0..d1;
                let values = match numbers(&temperature, &ranges) {
                    Ok(v) => v,
                    Err(e) => {
                        *failure.lock().unwrap() = Some(format!("days {}..{}: {e}", d0 + 1, d1));
                        break;
                    }
                };
                let (n0, n1) = ((ranges[0].end - ranges[0].start) as usize,
                                (ranges[1].end - ranges[1].start) as usize);
                let mut sum = vec![0.0f64; plane];
                let mut bad = 0usize;
                for a in 0..n0 {
                    for b in 0..n1 {
                        let day = if doy_axis == 0 { a } else { b } as u64 + d0;
                        let w = weight_of(doy[day as usize]);
                        let offset = (a * n1 + b) * plane;
                        for (s, v) in sum.iter_mut().zip(&values[offset..offset + plane]) {
                            if v.is_finite() { *s += w * v } else { bad += 1 }
                        }
                    }
                }
                if bad > 0 {
                    *failure.lock().unwrap() = Some(format!(
                        "days {}..{} carry {bad} missing temperature values", d0 + 1, d1));
                    break;
                }
                eprintln!("soiltemp: days {}..{} read ({}/{} blocks)", d0 + 1, d1, k + 1, starts.len());
                partial.lock().unwrap()[k] = Some(sum);
            });
        }
    });
    if let Some(message) = failure.into_inner().unwrap() {
        return fail(message);
    }
    let mut total = vec![0.0f64; plane];
    for sum in partial.into_inner().unwrap() {
        let sum = sum.ok_or("a climatology block was not read")?;
        for (t, s) in total.iter_mut().zip(sum) {
            *t += s;
        }
    }
    let weight_sum = hours.len() as f64 * doy.iter().map(|d| weight_of(*d)).sum::<f64>();
    let mean_2m = lat_major(&total.iter().map(|s| s / weight_sum).collect::<Vec<_>>(),
                            &spatial, ny, nx)?;
    let sea_level: Vec<f64> = mean_2m.iter().zip(&geopotential)
        .map(|(t, z)| t + request.lapse_rate_k_per_m * z / request.gravity_m_s2).collect();

    // Conservative, land-weighted remap onto the fixed 1-degree cells.
    let lat_edges = latitude_edges(&lat)?;
    let dlon = 360.0 / nx as f64;
    let spacing_ok = lon.windows(2).all(|w| ((w[1] - w[0]) - dlon).abs() < 1e-6 * dlon.max(1.0));
    if !spacing_ok {
        return fail("longitude must be uniformly spaced around the whole globe");
    }
    let rad = std::f64::consts::PI / 180.0;
    let lat_weights: Vec<Vec<(usize, f64)>> = (0..NY).map(|jj| {
        let (t0, t1) = (-90.0 + jj as f64, -89.0 + jj as f64);
        lat_edges.iter().enumerate().filter_map(|(j, (s0, s1))| {
            let (lo, hi) = (s0.max(t0), s1.min(t1));
            (hi > lo).then(|| (j, (hi * rad).sin() - (lo * rad).sin()))
        }).collect()
    }).collect();
    let lon_weights: Vec<Vec<(usize, f64)>> = (0..NX).map(|ii| {
        let (t0, t1) = (-180.0 + ii as f64, -179.0 + ii as f64);
        lon.iter().enumerate().filter_map(|(i, c)| {
            let w = circular_overlap(c - dlon / 2.0, c + dlon / 2.0, t0, t1);
            (w > 0.0).then_some((i, w))
        }).collect()
    }).collect();
    let mut kelvin = vec![f64::NAN; NY * NX];
    let (mut land_cells, mut min_k, mut max_k, mut sum_k) = (0usize, f64::INFINITY, f64::NEG_INFINITY, 0.0);
    for jj in 0..NY {
        for ii in 0..NX {
            let (mut area, mut land_area, mut num) = (0.0, 0.0, 0.0);
            for (j, wy) in &lat_weights[jj] {
                for (i, wx) in &lon_weights[ii] {
                    let w = wy * wx;
                    let f = land[j * nx + i];
                    area += w;
                    land_area += w * f;
                    num += w * f * sea_level[j * nx + i];
                }
            }
            if area <= 0.0 {
                return fail(format!("1-degree cell ({jj}, {ii}) overlaps no source cell"));
            }
            if land_area / area > request.land_fraction_min {
                let value = num / land_area;
                kelvin[jj * NX + ii] = value;
                land_cells += 1;
                min_k = min_k.min(value);
                max_k = max_k.max(value);
                sum_k += value;
            }
        }
    }
    let word = |value: f64| -> Result<u16> {
        if value.is_nan() {
            return Ok(0);
        }
        let scaled = (value / SCALE).round();
        if !(1.0..=65535.0).contains(&scaled) {
            return fail(format!("{value} K does not fit the dataset's 16-bit words"));
        }
        Ok(scaled as u16)
    };

    let staging = output.with_extension("partial");
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    let side = TILE + 2 * BORDER;
    let mut tiles = Vec::new();
    for t in 0..NX / TILE {
        let mut bytes = Vec::with_capacity(side * side * 2);
        for r in 0..side {
            // Rows beyond a pole repeat the pole row; columns wrap the globe.
            let jj = (r as isize - BORDER as isize).clamp(0, NY as isize - 1) as usize;
            for c in 0..side {
                let ii = (t * TILE + c + NX - BORDER) % NX;
                bytes.extend_from_slice(&word(kelvin[jj * NX + ii])?.to_be_bytes());
            }
        }
        let name = format!("{:05}-{:05}.{:05}-{:05}", t * TILE + 1, (t + 1) * TILE, 1, TILE);
        fs::write(staging.join(&name), &bytes)?;
        tiles.push(json!({"name": name, "bytes": bytes.len()}));
    }
    let description = request.description.replace('"', "'");
    let index = format!(
        "type=continuous\nprojection=regular_ll\nmissing_value=0.0\ndx=1.0\ndy=1.0\n\
         known_x=1.0\nknown_y=1.0\nknown_lat=-89.5\nknown_lon=-179.5\nwordsize=2\n\
         tile_x={TILE}\ntile_y={TILE}\ntile_z=1\ntile_bdr={BORDER}\nscale_factor={SCALE}\n\
         units=\"Kelvin\"\ndescription=\"{description}\"\n");
    {
        let mut file = fs::File::create(staging.join("index"))?;
        file.write_all(index.as_bytes())?;
        file.sync_all()?;
        // Closed before the rename: NTFS refuses to rename a directory
        // while a handle inside it is open ("Access is denied", os error
        // 5; the 2.9.4 Windows gate), where Linux allows it.
    }
    fs::rename(&staging, output)?;
    Ok(json!({
        "schema": RECEIPT_SCHEMA,
        "climatology_store": request.climatology_store,
        "temperature": {"variable": request.temperature, "dimensions": dims, "shape": shape,
                         "units": request.temperature_units, "days": doy.len(), "hours": hours.len(),
                         "block_days": block, "blocks": starts.len(), "workers": workers},
        "static_store": request.static_store,
        "geopotential": request.geopotential,
        "land_sea_mask": request.land_sea_mask,
        "source_grid": {"latitudes": ny, "longitudes": nx},
        "reduction": {"mean": "every hour and day of year, day 366 weighted leap_day_weight",
                      "leap_day_weight": request.leap_day_weight,
                      "sea_level": "T2m_mean + lapse_rate * geopotential_at_surface / gravity",
                      "lapse_rate_k_per_m": request.lapse_rate_k_per_m,
                      "gravity_m_s2": request.gravity_m_s2,
                      "remap": "area-conservative, land-weighted, onto 1-degree cells",
                      "land_fraction_min": request.land_fraction_min},
        "output": {"cells": NX * NY, "land_cells": land_cells, "missing_cells": NX * NY - land_cells,
                   "min_k": min_k, "max_k": max_k,
                   "mean_k": if land_cells > 0 { sum_k / land_cells as f64 } else { f64::NAN },
                   "tiles": tiles, "index": index},
    }))
}
