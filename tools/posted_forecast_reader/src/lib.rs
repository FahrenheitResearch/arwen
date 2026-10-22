//! Posted forecasts at the observation seam. Record identities and units are
//! table data. Rust owns decoding, coordinates, conversions and derived fields.

use chrono::Duration;
use grib_core::grib2::{grid_latlon, unpack_message, Grib2File, Grib2Message, GridDefinition};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, CStr, CString};
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selector {
    discipline: u8,
    category: u8,
    parameter: u8,
    level_type: u8,
    level_value: f64,
    center: u16,
    subcenter: u16,
    master_table_version: u8,
    local_table_version: u8,
    product_template: u16,
}

impl Selector {
    fn matches(&self, m: &Grib2Message) -> bool {
        let p = &m.product;
        let i = &m.identification;
        m.discipline == self.discipline && p.parameter_category == self.category
            && p.parameter_number == self.parameter && p.level_type == self.level_type
            && p.level_value == self.level_value && p.second_level_type == 255
            && i.center_id == self.center && i.subcenter_id == self.subcenter
            && i.master_table_version == self.master_table_version
            && i.local_table_version == self.local_table_version
            && p.template == self.product_template
            && p.perturbation_number.is_none() && p.derived_forecast_type.is_none()
            && p.probability_type.is_none() && p.percentile_value.is_none()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Field {
    units: String,
    minimum: f64,
    maximum: f64,
    factor: f64,
    offset: f64,
    selector: Selector,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Derived {
    operation: String,
    inputs: [String; 2],
    units: String,
    minimum: f64,
    maximum: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Table {
    schema: String,
    label: String,
    sources: Vec<String>,
    fields: BTreeMap<String, Field>,
    derived: BTreeMap<String, Derived>,
}

fn bounded(name: &str, values: &[f64], low: f64, high: f64) -> Result<()> {
    if !low.is_finite() || !high.is_finite() || low > high {
        return Err(format!("{name}: invalid declared bounds"));
    }
    if let Some((k, v)) = values.iter().enumerate().find(|(_, v)| !v.is_finite() || **v < low || **v > high) {
        return Err(format!("{name}: cell {k} is {v}, outside [{low}, {high}]; refusing a missing value or units accident at the scoring seam"));
    }
    Ok(())
}

fn valid_time(m: &Grib2Message) -> Result<String> {
    // Only instantaneous, deterministic records are admitted. An accumulation,
    // ensemble statistic or hourly maximum is not an instantaneous field.
    if m.product.template != 0 {
        return Err("only instantaneous deterministic PDT 4.0 fields are supported; interval and ensemble products need their own declared operators".into());
    }
    let seconds = match m.product.time_range_unit {
        0 => 60, 1 => 3600, 2 => 86400, 10 => 10800, 11 => 21600, 12 => 43200, 13 => 1,
        v => return Err(format!("forecast time unit {v} is not a fixed duration")),
    };
    let delta = Duration::seconds(i64::from(m.product.forecast_time) * seconds);
    Ok(m.reference_time.checked_add_signed(delta).ok_or("forecast time overflows")?
        .format("%Y-%m-%dT%H:%M:%S").to_string())
}

fn validate_envelopes(bytes: &[u8]) -> Result<()> {
    let mut offset = 0usize;
    if bytes.is_empty() { return Err("posted GRIB2 input is empty".into()); }
    while offset < bytes.len() {
        let tail = &bytes[offset..];
        if tail.len() < 20 || &tail[..4] != b"GRIB" || tail[7] != 2 {
            return Err(format!("invalid GRIB2 envelope at byte {offset}; a partial or foreign file cannot be scored"));
        }
        let declared = u64::from_be_bytes(tail[8..16].try_into().unwrap());
        let length = usize::try_from(declared).map_err(|_| "GRIB2 length exceeds address space")?;
        if length < 20 || length > tail.len() || &tail[length-4..length] != b"7777" {
            return Err(format!("truncated GRIB2 envelope at byte {offset}; incomplete archive bytes cannot be scored"));
        }
        offset = offset.checked_add(length).ok_or("GRIB2 offset overflows")?;
    }
    Ok(())
}

pub struct Reader {
    grid: GridDefinition,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    land: Vec<u8>,
    frames: BTreeMap<String, BTreeMap<String, Vec<f64>>>,
    metadata: CString,
}

impl Reader {
    pub fn open(table_path: &str, paths: &[String]) -> Result<Self> {
        let table_bytes = fs::read(table_path).map_err(|e| e.to_string())?;
        let table: Table = serde_json::from_slice(&table_bytes).map_err(|e| e.to_string())?;
        if table.schema != "posted-grib2-fields-v1" || table.fields.is_empty() || paths.is_empty() {
            return Err("a posted-grib2-fields-v1 table, fields and inputs are required".into());
        }
        let required = [("temperature_2m", "K"), ("dewpoint_2m", "K"),
            ("wind_speed_10m", "m s-1"), ("composite_reflectivity", "dBZ"),
            ("terrain_height", "m"), ("land_mask", "1")];
        let mut units: BTreeMap<String, String> = table.fields.iter().map(|(n,f)| (n.clone(),f.units.clone())).collect();
        for (name, f) in &table.derived {
            if units.insert(name.clone(), f.units.clone()).is_some() {
                return Err(format!("{name}: both decoded and derived"));
            }
            if f.operation != "hypot" || f.inputs.iter().any(|n| !table.fields.contains_key(n)) {
                return Err(format!("{name}: hypot needs two decoded fields"));
            }
            if f.inputs.iter().any(|n| table.fields[n].units != f.units) {
                return Err(format!("{name}: hypot inputs and result need the same units"));
            }
        }
        for (name, expected) in required {
            if units.get(name).map(String::as_str) != Some(expected) {
                return Err(format!("{name}: the scoring seam requires {expected}"));
            }
        }
        let mut frames: BTreeMap<String, BTreeMap<String, Vec<f64>>> = BTreeMap::new();
        let mut reference = None;
        let mut grid: Option<GridDefinition> = None;
        let mut provenance: Vec<Value> = Vec::new();
        let mut files = BTreeSet::new();
        for path in paths {
            let canonical = fs::canonicalize(path).map_err(|e| e.to_string())?;
            if !files.insert(canonical) { return Err("the input inventory repeats a file".into()); }
            let bytes = fs::read(path).map_err(|e| e.to_string())?;
            validate_envelopes(&bytes)?;
            let digest = format!("{:x}", Sha256::digest(&bytes));
            let file = Grib2File::from_bytes(&bytes).map_err(|e| e.to_string())?;
            for (index, m) in file.messages.iter().enumerate() {
                for (name, spec) in &table.fields {
                    if !spec.selector.matches(m) { continue; }
                    let time = valid_time(m)?;
                    if reference.is_some_and(|r| r != m.reference_time) {
                        return Err("selected records mix forecast cycles; that would mix different forecast ages in one score".into());
                    }
                    reference = Some(m.reference_time);
                    if grid.as_ref().is_some_and(|g| g != &m.grid) {
                        return Err("selected records do not share the exact GRIB grid definition; pointwise scores would be misregistered".into());
                    }
                    // The decoder's Lambert coordinates are i-consecutive with
                    // positive x and y. Other scanning orders must be normalized
                    // by a reader that understands their matching coordinates.
                    if m.grid.template != 30 || m.grid.scan_mode != 0x40 || m.grid.is_reduced {
                        return Err("this reader supports Lambert template 3.30 with positive i/j, i-consecutive scan 0x40; another grid or scan would pair values with the wrong scoring coordinates".into());
                    }
                    if m.grid.nx < 2 || m.grid.ny < 2 || m.grid.dx <= 0.0 || m.grid.dy <= 0.0 {
                        return Err("a scoring grid needs positive spacing and at least two points on each axis".into());
                    }
                    grid = Some(m.grid.clone());
                    let mut values = unpack_message(m).map_err(|e| e.to_string())?;
                    if values.len() != m.grid.nx as usize * m.grid.ny as usize {
                        return Err(format!("{name}: decoded count differs from grid shape"));
                    }
                    if !spec.factor.is_finite() || !spec.offset.is_finite() {
                        return Err(format!("{name}: unit conversion is not finite"));
                    }
                    for value in &mut values { *value = *value * spec.factor + spec.offset; }
                    bounded(name, &values, spec.minimum, spec.maximum)?;
                    if frames.entry(time.clone()).or_default().insert(name.clone(), values).is_some() {
                        return Err(format!("{name}: duplicate selected record at {time}; refusing an ambiguous comparison"));
                    }
                    provenance.push(json!({"field":name,"valid_time":time,"file":Path::new(path).file_name().unwrap().to_string_lossy(),"sha256":digest,"record_index":index}));
                }
            }
        }
        let grid = grid.ok_or("no records match the field table")?;
        for (time, frame) in &mut frames {
            for name in table.fields.keys() {
                if !frame.contains_key(name) { return Err(format!("{name}: no selected record at {time}; refusing a partial scored frame")); }
            }
            for (name, spec) in &table.derived {
                let values: Vec<f64> = frame[&spec.inputs[0]].iter().zip(&frame[&spec.inputs[1]])
                    .map(|(x,y)| x.hypot(*y)).collect();
                bounded(name, &values, spec.minimum, spec.maximum)?;
                frame.insert(name.clone(), values);
            }
        }
        let first = frames.values().next().unwrap();
        for frame in frames.values() {
            for name in ["terrain_height", "land_mask"] {
                if frame[name] != first[name] { return Err(format!("{name}: changes between frames; a fixed station admission would no longer describe every lead")); }
            }
        }
        let land = first["land_mask"].iter().map(|v| match *v {
            0.0 => Ok(0), 1.0 => Ok(1), _ => Err("land_mask must be an exact 0/1 flag".to_string())
        }).collect::<Result<Vec<_>>>()?;
        let (latitude, mut longitude) = grid_latlon(&grid).map_err(|e| e.to_string())?;
        for value in &mut longitude { *value = (*value + 180.0).rem_euclid(360.0) - 180.0; }
        let metadata = CString::new(json!({"schema":"posted-forecast-reader-v1", "reader":"grib-core-rust", "label":table.label,"sources":table.sources,
            "field_table_sha256":format!("{:x}", Sha256::digest(&table_bytes)),"shape":[grid.ny,grid.nx],"dx_m":grid.dx,
            "valid_times":frames.keys().collect::<Vec<_>>(),"init_time":reference.unwrap().format("%Y-%m-%dT%H:%M:%S").to_string(),
            "units":units,"selected_records":provenance,"derived_fields":table.derived.keys().collect::<Vec<_>>(),
            "reflectivity_operator":"posted composite reflectivity; no recomputation"}).to_string()).map_err(|e| e.to_string())?;
        Ok(Self { grid, latitude, longitude, land, frames, metadata })
    }

    fn field(&self, time: &str, name: &str) -> Result<&[f64]> {
        match name {
            "latitude" => Ok(&self.latitude), "longitude" => Ok(&self.longitude),
            _ => self.frames.get(time).and_then(|f| f.get(name)).map(Vec::as_slice)
                .ok_or_else(|| format!("no declared {name} at {time}")),
        }
    }

    fn locate(&self, latitude: f64, longitude: f64) -> Result<(f64,f64)> {
        if !latitude.is_finite() || !longitude.is_finite() || latitude.abs() >= 90.0 {
            return Err("station coordinates must be finite and short of the poles".into());
        }
        let g = &self.grid;
        let a = g.latin1.to_radians(); let b = g.latin2.to_radians();
        let n = if (g.latin1-g.latin2).abs() < 1e-6 { a.sin() }
            else { (a.cos().ln()-b.cos().ln()) / ((std::f64::consts::FRAC_PI_4+b/2.0).tan().ln()-(std::f64::consts::FRAC_PI_4+a/2.0).tan().ln()) };
        let f = a.cos() * (std::f64::consts::FRAC_PI_4+a/2.0).tan().powf(n) / n;
        let project = |lat:f64,lon:f64| {
            let rho = 6371229.0*f/(std::f64::consts::FRAC_PI_4+lat.to_radians()/2.0).tan().powf(n);
            let theta = n*((lon-g.lov+180.0).rem_euclid(360.0)-180.0).to_radians();
            (rho*theta.sin(), -rho*theta.cos())
        };
        let (x0,y0) = project(g.lat1,g.lon1); let (x,y) = project(latitude,longitude);
        let result = ((x-x0)/g.dx, (y-y0)/g.dy);
        if !result.0.is_finite() || !result.1.is_finite() { return Err("station projection is singular".into()); }
        Ok(result)
    }
}

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn set_error(error: String) { ERROR.with(|v| *v.borrow_mut() = CString::new(error.replace('\0', "?")).unwrap()); }
unsafe fn text<'a>(p: *const c_char) -> Result<&'a str> {
    if p.is_null() { return Err("null text argument".into()); }
    CStr::from_ptr(p).to_str().map_err(|e| e.to_string())
}

#[no_mangle]
pub extern "C" fn posted_reader_error() -> *const c_char { ERROR.with(|v| v.borrow().as_ptr()) }
#[no_mangle]
pub extern "C" fn posted_reader_abi_version() -> u32 { 1 }

/// Paths are a JSON string array. Call close exactly once for a non-null handle.
#[no_mangle]
pub unsafe extern "C" fn posted_reader_open(table: *const c_char, paths: *const c_char) -> *mut Reader {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let paths: Vec<String> = serde_json::from_str(text(paths)?).map_err(|e| e.to_string())?;
        Reader::open(text(table)?, &paths)
    }));
    match result {
        Ok(Ok(reader)) => Box::into_raw(Box::new(reader)),
        Ok(Err(e)) => {set_error(e); std::ptr::null_mut()},
        Err(_) => {set_error("native reader panicked".into()); std::ptr::null_mut()},
    }
}

#[no_mangle]
pub unsafe extern "C" fn posted_reader_close(reader: *mut Reader) { if !reader.is_null() { drop(Box::from_raw(reader)); } }
#[no_mangle]
pub unsafe extern "C" fn posted_reader_metadata(reader: *const Reader) -> *const c_char { (*reader).metadata.as_ptr() }
#[no_mangle]
pub unsafe extern "C" fn posted_reader_field(reader: *const Reader, time: *const c_char, name: *const c_char, len: *mut usize) -> *const f64 {
    let result = text(time).and_then(|t| text(name).and_then(|n| (*reader).field(t,n)));
    match result { Ok(v) => { *len=v.len(); v.as_ptr() }, Err(e) => {set_error(e); *len=0; std::ptr::null()} }
}
#[no_mangle]
pub unsafe extern "C" fn posted_reader_land(reader: *const Reader) -> *const u8 { (*reader).land.as_ptr() }
#[no_mangle]
pub unsafe extern "C" fn posted_reader_locate(reader: *const Reader, lat: f64, lon: f64, x: *mut f64, y: *mut f64) -> i32 {
    match (*reader).locate(lat,lon) { Ok((a,b)) => { *x=a; *y=b; 0 }, Err(e) => {set_error(e); 1} }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_values_and_unit_accidents_cannot_be_scored() {
        assert!(bounded("temperature", &[280.0], 150.0, 350.0).is_ok());
        assert!(bounded("temperature", &[20.0], 150.0, 350.0).is_err());
        assert!(bounded("temperature", &[f64::NAN], 150.0, 350.0).is_err());
        assert!(bounded("temperature", &[280.0], f64::NAN, 350.0).is_err());
    }
}
