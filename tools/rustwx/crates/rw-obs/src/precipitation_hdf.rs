//! Table-defined HDF5 (and netCDF-4) precipitation grids with explicit axes and periods.
//!
//! Every product is one row of `data/precipitation-hdf-products.json`; a new
//! source is a new row, never a new code path.  A row declares:
//!
//! * the value variable, its units (`mm/hr` rate or `mm` accumulation) and its
//!   axis order (`time,lon,lat` or `time,lat,lon`), with the attribute that
//!   states the order when the file carries one (`axes_attribute`).  A file
//!   with no such attribute is held to its shape: the order is only accepted
//!   when the latitude and longitude axes have different lengths, so an
//!   unprovable order is refused rather than guessed;
//! * the period: either native half-open time bounds (`time_bounds_variable`)
//!   or a time coordinate (`time_variable`) whose value plus a declared
//!   `period_end_offset_seconds` is the period end, with the declared
//!   `accumulation_seconds` before it.  A multi-record file is decoded one
//!   record at a time, chosen by the requested period end;
//! * identity checks: a header string of `key=value;` pairs
//!   (`root_header_attribute` + `required_header_values`) and/or exact root
//!   attributes (`required_root_attributes`);
//! * an optional quality index kept beside the field.
use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use chrono::{DateTime, Duration, Utc};
use hdf5_reader::{Datatype, Hdf5File};
use serde::{Deserialize, Serialize};

use crate::pack::{
    decode_pack, payload_digest, write_pack, PayloadBuilder, GEO_SCHEMA, GRID_SCHEMA,
};
use crate::precipitation::{instant, write_frame, Frame, Grid, Mask, Meta};
use crate::seam::{
    seam_time, wrap_longitude, Provenance, QUANTITY_PRECIPITATION_ACCUMULATION, UNITS_MM,
};
use crate::{absolute_uri, err, hex_sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Product {
    pub name: String,
    pub source: String,
    pub value_variable: String,
    pub value_units: String,
    pub value_axes: String,
    /// The value variable's attribute that states its axis order, when the
    /// file carries one (HDF-EOS `DimensionNames`).
    #[serde(default)]
    pub axes_attribute: Option<String>,
    pub latitude_variable: String,
    pub longitude_variable: String,
    /// Native half-open period bounds, one `[start, end]` pair per record.
    #[serde(default)]
    pub time_bounds_variable: Option<String>,
    /// A time coordinate, one value per record, when the file has no bounds.
    #[serde(default)]
    pub time_variable: Option<String>,
    pub time_units: String,
    pub time_origin: String,
    /// Seconds in one unit of the time values (3600 for "hours since").
    #[serde(default = "one_second")]
    pub time_value_seconds: i64,
    /// Period end = time value + this offset (time-coordinate rows only).
    #[serde(default)]
    pub period_end_offset_seconds: i64,
    pub accumulation_seconds: i64,
    #[serde(default)]
    pub quality_variable: Option<String>,
    #[serde(default)]
    pub quality_range: Option<[f64; 2]>,
    #[serde(default)]
    pub root_header_attribute: Option<String>,
    #[serde(default)]
    pub header_start_key: Option<String>,
    #[serde(default)]
    pub header_last_included_key: Option<String>,
    #[serde(default)]
    pub header_last_included_tick_milliseconds: i64,
    #[serde(default)]
    pub required_header_values: BTreeMap<String, String>,
    /// Root attributes that must read exactly as declared (product identity
    /// and, where the file states it in prose, the period definition).
    #[serde(default)]
    pub required_root_attributes: BTreeMap<String, String>,
}

fn one_second() -> i64 {
    1
}

/// The value layouts this decoder reads, by declared axis order.
const AXES: [&str; 2] = ["time,lon,lat", "time,lat,lon"];
/// The value units this decoder reads: a rate over the period, or its total.
const UNITS: [&str; 2] = ["mm/hr", "mm"];

pub fn product(name: &str) -> Result<Product> {
    #[derive(Deserialize)]
    struct Table {
        schema: String,
        products: Vec<Product>,
    }
    let table: Table =
        serde_json::from_str(include_str!("../data/precipitation-hdf-products.json"))?;
    if table.schema != "gpuwm-obs.precipitation-hdf-products.v1" {
        return Err(err("unknown HDF precipitation table schema"));
    }
    table
        .products
        .into_iter()
        .find(|row| row.name == name)
        .ok_or_else(|| err(format!("no HDF precipitation metadata for {name}")))
}

fn values(dataset: &hdf5_reader::Dataset) -> Result<Vec<f64>> {
    macro_rules! read {
        ($t:ty) => {
            dataset
                .read_array::<$t>()?
                .iter()
                .map(|&value| value as f64)
                .collect()
        };
    }
    Ok(match dataset.dtype() {
        Datatype::FloatingPoint { size: 4, .. } => read!(f32),
        Datatype::FloatingPoint { size: 8, .. } => read!(f64),
        Datatype::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => read!(i8),
        Datatype::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => read!(u8),
        Datatype::FixedPoint {
            size: 2,
            signed: true,
            ..
        } => read!(i16),
        Datatype::FixedPoint {
            size: 2,
            signed: false,
            ..
        } => read!(u16),
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => read!(i32),
        Datatype::FixedPoint {
            size: 4,
            signed: false,
            ..
        } => read!(u32),
        other => {
            return Err(err(format!(
                "unsupported numeric precipitation dataset {other:?}"
            )))
        }
    })
}

fn text(dataset: &hdf5_reader::Dataset, key: &str) -> Result<String> {
    Ok(dataset
        .attribute(key)?
        .read_string()?
        .trim_matches('\0')
        .trim()
        .to_string())
}

fn number(dataset: &hdf5_reader::Dataset, key: &str) -> Option<f64> {
    dataset.attribute(key).ok()?.read_as_f64().ok()
}

fn unpacked(dataset: &hdf5_reader::Dataset, require_fill: bool) -> Result<Vec<f64>> {
    let fill = number(dataset, "_FillValue");
    let missing = number(dataset, "missing_value");
    // COARDS files (the CPC analyses) state `missing_value` and no
    // `_FillValue`; either one names the empty cell, and a variable that
    // states neither could not tell an ocean cell from a dry one.
    if require_fill && fill.is_none() && missing.is_none() {
        return Err(err(
            "precipitation variable must state its fill or missing value",
        ));
    }
    let scale = number(dataset, "scale_factor").unwrap_or(1.0);
    let offset = number(dataset, "add_offset").unwrap_or(0.0);
    if !scale.is_finite() || !offset.is_finite() || scale == 0.0 {
        return Err(err("invalid CF packing scale or offset"));
    }
    Ok(values(dataset)?
        .into_iter()
        .map(|raw| {
            if !raw.is_finite() || fill == Some(raw) || missing == Some(raw) {
                f64::NAN
            } else {
                raw * scale + offset
            }
        })
        .collect())
}

fn root_text(file: &Hdf5File, key: &str) -> Result<String> {
    Ok(file
        .root_group()?
        .attribute(key)?
        .read_string()?
        .trim_matches('\0')
        .trim()
        .to_string())
}

pub fn inspect(path: &Path, definition: &Product) -> Result<serde_json::Value> {
    let file = Hdf5File::open(path)?;
    let mut variables = BTreeMap::new();
    let names = [
        Some(&definition.value_variable),
        Some(&definition.latitude_variable),
        Some(&definition.longitude_variable),
        definition.time_bounds_variable.as_ref(),
        definition.time_variable.as_ref(),
        definition.quality_variable.as_ref(),
    ];
    for name in names.into_iter().flatten() {
        let dataset = file.dataset(name)?;
        let mut attributes = BTreeMap::new();
        for attr in dataset.attributes() {
            let value = match attr.read_string() {
                Ok(text) => serde_json::Value::String(text),
                Err(_) => match attr.read_as_f64() {
                    Ok(number) => serde_json::json!(number),
                    Err(_) => serde_json::json!(format!("{:?}", attr.datatype)),
                },
            };
            attributes.insert(attr.name, value);
        }
        variables.insert(name.clone(),serde_json::json!({"shape":dataset.shape(),"dtype":format!("{:?}",dataset.dtype()),"attributes":attributes}));
    }
    let mut root = BTreeMap::new();
    for key in definition
        .root_header_attribute
        .iter()
        .chain(definition.required_root_attributes.keys())
    {
        root.insert(key.clone(), root_text(&file, key)?);
    }
    Ok(serde_json::json!({"schema":"gpuwm-obs.precipitation-hdf-inventory.v1",
        "source_sha256":hex_sha256(&std::fs::read(path)?),"variables":variables,"root_attributes":root}))
}

fn offset(origin: DateTime<Utc>, value: f64, unit_seconds: i64, what: &str) -> Result<DateTime<Utc>> {
    let seconds = value * unit_seconds as f64;
    if !seconds.is_finite() || seconds.fract() != 0.0 || seconds.abs() > i64::MAX as f64 / 2.0 {
        return Err(err(format!(
            "precipitation {what} is not a finite whole number of seconds"
        )));
    }
    let shift = Duration::try_seconds(seconds as i64)
        .ok_or_else(|| err(format!("precipitation {what} overflows")))?;
    origin
        .checked_add_signed(shift)
        .ok_or_else(|| err(format!("precipitation {what} overflows")))
}

/// Every record's half-open period, from native bounds or the time coordinate.
fn read_periods(file: &Hdf5File, definition: &Product) -> Result<Vec<(DateTime<Utc>, DateTime<Utc>)>> {
    let origin = instant(&definition.time_origin)?;
    let unit = definition.time_value_seconds;
    if unit <= 0 {
        return Err(err("precipitation time unit must be a positive number of seconds"));
    }
    let periods = match (&definition.time_bounds_variable, &definition.time_variable) {
        (Some(name), None) => {
            let dataset = file.dataset(name)?;
            let shape = dataset.shape();
            if shape.len() != 2 || shape[1] != 2 || text(&dataset, "units")? != definition.time_units {
                return Err(err(
                    "precipitation time bounds differ from the declared native time axis or unit",
                ));
            }
            let bounds = unpacked(&dataset, false)?;
            bounds
                .chunks(2)
                .map(|pair| {
                    Ok((
                        offset(origin, pair[0], unit, "start offset")?,
                        offset(origin, pair[1], unit, "end offset")?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        }
        (None, Some(name)) => {
            let dataset = file.dataset(name)?;
            if dataset.shape().len() != 1 || text(&dataset, "units")? != definition.time_units {
                return Err(err(
                    "precipitation time coordinate differs from the declared native time axis or unit",
                ));
            }
            let span = Duration::try_seconds(definition.accumulation_seconds)
                .ok_or_else(|| err("precipitation accumulation overflows"))?;
            unpacked(&dataset, false)?
                .into_iter()
                .map(|value| {
                    let stamp = offset(origin, value, unit, "time value")?;
                    let end = offset(stamp, definition.period_end_offset_seconds as f64, 1, "period end")?;
                    Ok((end - span, end))
                })
                .collect::<Result<Vec<_>>>()?
        }
        _ => {
            return Err(err(
                "a precipitation product declares exactly one of time_bounds_variable and time_variable",
            ))
        }
    };
    if periods.is_empty() {
        return Err(err("precipitation file holds no time record"));
    }
    for (start, end) in &periods {
        if (*end - *start).num_seconds() != definition.accumulation_seconds {
            return Err(err(
                "precipitation time bounds have the wrong accumulation duration",
            ));
        }
    }
    Ok(periods)
}

/// The product identity the row declares: header pairs and exact root attributes.
fn check_identity(
    file: &Hdf5File,
    definition: &Product,
    period: (DateTime<Utc>, DateTime<Utc>),
) -> Result<()> {
    for (key, expected) in &definition.required_root_attributes {
        if root_text(file, key)? != *expected {
            return Err(err(format!(
                "precipitation root attribute {key} differs from the declared product identity"
            )));
        }
    }
    let Some(header_name) = &definition.root_header_attribute else {
        return Ok(());
    };
    let header = file.root_group()?.attribute(header_name)?.read_string()?;
    let fields = header
        .split(';')
        .filter_map(|item| item.trim().split_once('='))
        .map(|(key, value)| (key.trim(), value.trim()))
        .collect::<BTreeMap<_, _>>();
    for (key, expected) in &definition.required_header_values {
        if fields.get(key.as_str()) != Some(&expected.as_str()) {
            return Err(err(format!(
                "precipitation header {key} differs from the declared product identity"
            )));
        }
    }
    let from_header = |key: &Option<String>| -> Result<Option<DateTime<Utc>>> {
        let Some(key) = key else { return Ok(None) };
        let value = fields
            .get(key.as_str())
            .ok_or_else(|| err(format!("precipitation header has no {key}")))?;
        Ok(Some(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc)))
    };
    let header_start = from_header(&definition.header_start_key)?;
    let header_end = from_header(&definition.header_last_included_key)?
        .map(|last| last + Duration::milliseconds(definition.header_last_included_tick_milliseconds));
    if header_start.is_some_and(|start| start != period.0) || header_end.is_some_and(|end| end != period.1) {
        return Err(err(
            "precipitation granule header disagrees with the native half-open time bounds",
        ));
    }
    Ok(())
}

pub fn decode(
    path: &Path,
    definition: &Product,
    bounds: Option<[f64; 4]>,
    expected_end: Option<DateTime<Utc>>,
    geometry: &Path,
    out: &Path,
) -> Result<Meta> {
    // The supported layouts are explicit metadata, never inferred.
    if !AXES.contains(&definition.value_axes.as_str())
        || !UNITS.contains(&definition.value_units.as_str())
    {
        return Err(err(format!(
            "the HDF precipitation converter reads value axes {AXES:?} in units {UNITS:?}; \
             product {} declares {} in {}",
            definition.name, definition.value_axes, definition.value_units
        )));
    }
    let file = Hdf5File::open(path)?;
    let periods = read_periods(&file, definition)?;
    let record = match expected_end {
        Some(wanted) => periods
            .iter()
            .position(|(_, end)| *end == wanted)
            .ok_or_else(|| {
                err("requested endpoint differs from every precipitation record interval")
            })?,
        None if periods.len() == 1 => 0,
        None => {
            return Err(err(format!(
                "the file holds {} precipitation records; name the one to decode with \
                 --valid-time (its period end)",
                periods.len()
            )))
        }
    };
    let (start, end) = periods[record];
    check_identity(&file, definition, (start, end))?;
    let precipitation = file.dataset(&definition.value_variable)?;
    if text(&precipitation, "units")? != definition.value_units {
        return Err(err(
            "precipitation units differ from the native product declaration",
        ));
    }
    if let Some(attribute) = &definition.axes_attribute {
        if text(&precipitation, attribute)? != definition.value_axes {
            return Err(err(
                "precipitation dimension order differs from the native product declaration",
            ));
        }
    }
    let lat_dataset = file.dataset(&definition.latitude_variable)?;
    let lon_dataset = file.dataset(&definition.longitude_variable)?;
    if lat_dataset.shape().len() != 1
        || lon_dataset.shape().len() != 1
        || text(&lat_dataset, "units")? != "degrees_north"
        || text(&lon_dataset, "units")? != "degrees_east"
    {
        return Err(err(
            "precipitation coordinates are not declared one-dimensional geographic axes",
        ));
    }
    let latitude = unpacked(&lat_dataset, false)?;
    let longitude = unpacked(&lon_dataset, false)?
        .into_iter()
        .map(wrap_longitude)
        .collect::<Vec<_>>();
    let (nx, ny, nt) = (longitude.len(), latitude.len(), periods.len());
    let lat_first = definition.value_axes == "time,lat,lon";
    // With no attribute stating the order, equal axis lengths would let a
    // transposed field pass every shape test: refuse rather than guess.
    if definition.axes_attribute.is_none() && nx == ny {
        return Err(err(
            "the axis order is held only by the field's shape, and equal latitude and \
             longitude lengths cannot prove it",
        ));
    }
    let expected_shape = if lat_first {
        [nt as u64, ny as u64, nx as u64]
    } else {
        [nt as u64, nx as u64, ny as u64]
    };
    if precipitation.shape() != expected_shape
        || latitude
            .iter()
            .chain(longitude.iter())
            .any(|x| !x.is_finite())
    {
        return Err(err("precipitation data and geographic axis shapes differ"));
    }
    let quality = match &definition.quality_variable {
        Some(name) => {
            let dataset = file.dataset(name)?;
            if dataset.shape() != precipitation.shape()
                || definition
                    .axes_attribute
                    .as_ref()
                    .map(|attribute| text(&dataset, attribute))
                    .transpose()?
                    .is_some_and(|axes| axes != definition.value_axes)
            {
                return Err(err(
                    "precipitation quality index does not share the declared field axes",
                ));
            }
            Some(unpacked(&dataset, true)?)
        }
        None => None,
    };
    let quality_range = match (&quality, definition.quality_range) {
        (Some(_), Some(range)) => Some(range),
        (None, None) => None,
        _ => {
            return Err(err(
                "a precipitation quality variable and its range are declared together",
            ))
        }
    };
    let values = unpacked(&precipitation, true)?;
    let [west, south, east, north] = bounds.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
    let columns = (0..nx)
        .filter(|&i| longitude[i] >= west && longitude[i] <= east)
        .collect::<Vec<_>>();
    let rows = (0..ny)
        .filter(|&j| latitude[j] >= south && latitude[j] <= north)
        .collect::<Vec<_>>();
    if columns.is_empty() || rows.is_empty() {
        return Err(err("bbox selects no precipitation observations"));
    }
    // The grid record names a source window by its first index and size, so
    // the selection must be one contiguous run on each axis.
    if columns.windows(2).any(|w| w[1] != w[0] + 1) || rows.windows(2).any(|w| w[1] != w[0] + 1) {
        return Err(err(
            "bbox selects a non-contiguous window of the source grid (it crosses the file's \
             longitude seam); split it at the seam",
        ));
    }
    let grid = Grid {
        kind: "rectilinear_latlon".into(),
        nx: columns.len(),
        ny: rows.len(),
        source_nx: nx,
        source_ny: ny,
        i_start: columns[0],
        j_start: rows[0],
    };
    let mut lat_out = Vec::new();
    let mut lon_out = Vec::new();
    let mut amount = Vec::new();
    let mut observed = Vec::new();
    let mut retained_quality = Vec::new();
    let mut quality_valid = Vec::new();
    let hours = definition.accumulation_seconds as f64 / 3600.0;
    let per_period = if definition.value_units == "mm/hr" { hours } else { 1.0 };
    let record_base = record * nx * ny;
    for &j in &rows {
        for &i in &columns {
            let index = record_base + if lat_first { j * nx + i } else { i * ny + j };
            let value = values[index];
            let q = quality.as_ref().map(|q| q[index]);
            let quality_ok = q.map_or(true, |q| q.is_finite());
            let valid = value.is_finite() && quality_ok;
            if (value.is_finite() && value < 0.0)
                || q.zip(quality_range).is_some_and(|(q, [low, high])| {
                    q.is_finite() && (q < low || q > high)
                })
            {
                return Err(err(
                    "precipitation value or source quality index is outside its declared range",
                ));
            }
            lat_out.push(latitude[j]);
            lon_out.push(longitude[i]);
            amount.push(if valid { value * per_period } else { 0.0 });
            observed.push(valid);
            if let Some(q) = q {
                retained_quality.push(if q.is_finite() { q } else { 0.0 });
                quality_valid.push(q.is_finite());
            }
        }
    }
    let mut builder = PayloadBuilder::new();
    builder.push_f64("latitude", &lat_out, vec![grid.ny, grid.nx]);
    builder.push_f64("longitude", &lon_out, vec![grid.ny, grid.nx]);
    let (geo_payload, geo_arrays) = builder.finish();
    let geometry_sha256 = payload_digest(&geo_payload);
    if geometry.exists() {
        let (meta, payload): (serde_json::Value, Vec<u8>) = decode_pack(&std::fs::read(geometry)?)?;
        if meta["schema"] != GEO_SCHEMA || payload != geo_payload {
            return Err(err(
                "existing precipitation geometry differs from this granule",
            ));
        }
    } else {
        write_pack(
            geometry,
            &serde_json::json!({"schema":GEO_SCHEMA,"status":"READY","grid":grid,"arrays":geo_arrays,"payload_bytes":geo_payload.len(),"content_sha256":geometry_sha256}),
            &geo_payload,
        )?;
    }
    let meta = Meta {
        schema: GRID_SCHEMA.into(),
        status: "READY".into(),
        quantity: QUANTITY_PRECIPITATION_ACCUMULATION.into(),
        units: UNITS_MM.into(),
        valid_time: seam_time(end),
        accumulation_start: seam_time(start),
        accumulation_seconds: definition.accumulation_seconds,
        accumulation_hours: hours,
        provenance: Provenance::new(
            &definition.source,
            &definition.name,
            absolute_uri(path),
            hex_sha256(&std::fs::read(path)?),
            seam_time(Utc::now()),
        ),
        geometry_sha256,
        grid: grid.clone(),
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
        source_frames: Vec::new(),
    };
    let written = write_frame(
        out,
        Frame {
            meta,
            values: amount,
            valid: observed,
        },
    )?;
    // Keep the unthresholded native quality index as a companion pack. Its role
    // is provenance and later stratification, not an invented confidence cutoff.
    if let (Some(name), Some(range)) = (&definition.quality_variable, quality_range) {
        let mut builder = PayloadBuilder::new();
        builder.push_f64(
            "precipitation_quality_index",
            &retained_quality,
            vec![grid.ny, grid.nx],
        );
        builder.push_mask("valid", &quality_valid, vec![grid.ny, grid.nx]);
        let (payload, arrays) = builder.finish();
        let quality_path = out.with_extension("quality.obspack");
        write_pack(
            &quality_path,
            &serde_json::json!({"schema":"gpuwm-obs.precipitation-quality.v1","valid_time":written.valid_time,
            "source_variable":name,"range":range,"source_sha256":written.provenance.sha256,
            "geometry_sha256":written.geometry_sha256,"arrays":arrays,"payload_bytes":payload.len(),"content_sha256":payload_digest(&payload)}),
            &payload,
        )?;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metadata_defines_rate_units_axes_and_half_hour_period() {
        let product = product("imerg-final-v07b").unwrap();
        assert_eq!(product.value_axes, "time,lon,lat");
        assert_eq!(product.accumulation_seconds, 1800);
        assert_eq!(product.value_units, "mm/hr");
        assert_eq!(product.required_header_values["ProductVersion"], "V07B");
        assert_eq!(product.axes_attribute.as_deref(), Some("DimensionNames"));
        assert_eq!(product.time_bounds_variable.as_deref(), Some("/Grid/time_bnds"));
        assert!(product.time_variable.is_none());
        assert_eq!(product.time_value_seconds, 1);
        assert!(super::product("imerg-early").is_err());
    }

    #[test]
    fn cpc_gauge_row_is_a_daily_lat_lon_total_ending_at_12z() {
        let product = product("cpc-unified-conus-rt").unwrap();
        assert!(AXES.contains(&product.value_axes.as_str()));
        assert_eq!(product.value_axes, "time,lat,lon");
        assert_eq!(product.value_units, "mm");
        assert!(product.axes_attribute.is_none(), "netCDF-4 carries no DimensionNames");
        assert!(product.time_bounds_variable.is_none());
        assert_eq!(product.time_variable.as_deref(), Some("time"));
        assert_eq!(product.accumulation_seconds, 86_400);
        assert!(product.quality_variable.is_none() && product.quality_range.is_none());
        assert!(product.required_root_attributes["Comments"].contains("12z of previous day"));
        // 1972320 h after 1800-01-01 is 2025-01-01 00Z, the day stored; its
        // total runs 12Z 2024-12-31 to 12Z 2025-01-01.
        let origin = instant(&product.time_origin).unwrap();
        let stamp = offset(origin, 1_972_320.0, product.time_value_seconds, "t").unwrap();
        assert_eq!(stamp.to_rfc3339(), "2025-01-01T00:00:00+00:00");
        let end = offset(stamp, product.period_end_offset_seconds as f64, 1, "end").unwrap();
        assert_eq!(end.to_rfc3339(), "2025-01-01T12:00:00+00:00");
    }

    #[test]
    fn a_time_value_that_is_not_whole_seconds_is_refused() {
        let origin = instant("1800-01-01T00:00:00Z").unwrap();
        assert!(offset(origin, 0.5, 1, "t").is_err());
        assert!(offset(origin, f64::NAN, 3600, "t").is_err());
        assert!(offset(origin, 0.5, 3600, "t").is_ok());
    }
}
