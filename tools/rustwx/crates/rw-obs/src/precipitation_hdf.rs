//! Table-defined HDF5 precipitation rates with explicit axes and time bounds.
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
    pub latitude_variable: String,
    pub longitude_variable: String,
    pub time_bounds_variable: String,
    pub time_units: String,
    pub time_origin: String,
    pub accumulation_seconds: i64,
    pub quality_variable: String,
    pub quality_range: [f64; 2],
    pub root_header_attribute: String,
    pub header_start_key: String,
    pub header_last_included_key: String,
    pub header_last_included_tick_milliseconds: i64,
    pub required_header_values: BTreeMap<String, String>,
}

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
    if require_fill && fill.is_none() {
        return Err(err("precipitation variable must state its fill value"));
    }
    let missing = number(dataset, "missing_value");
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

pub fn inspect(path: &Path, definition: &Product) -> Result<serde_json::Value> {
    let file = Hdf5File::open(path)?;
    let mut variables = BTreeMap::new();
    for name in [
        &definition.value_variable,
        &definition.latitude_variable,
        &definition.longitude_variable,
        &definition.time_bounds_variable,
        &definition.quality_variable,
    ] {
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
    Ok(
        serde_json::json!({"schema":"gpuwm-obs.precipitation-hdf-inventory.v1","source_sha256":hex_sha256(&std::fs::read(path)?),"variables":variables,
        "header":file.root_group()?.attribute(&definition.root_header_attribute)?.read_string()?}),
    )
}

fn read_period(file: &Hdf5File, definition: &Product) -> Result<(DateTime<Utc>, DateTime<Utc>)> {
    let dataset = file.dataset(&definition.time_bounds_variable)?;
    if dataset.shape() != [1, 2] || text(&dataset, "units")? != definition.time_units {
        return Err(err(
            "precipitation time bounds differ from the declared native time axis or unit",
        ));
    }
    let bounds = unpacked(&dataset, false)?;
    if bounds.len() != 2
        || bounds.iter().any(|value| {
            !value.is_finite() || value.fract() != 0.0 || value.abs() > i64::MAX as f64
        })
    {
        return Err(err(
            "precipitation time bounds are not two finite whole seconds",
        ));
    }
    let origin = instant(&definition.time_origin)?;
    let start_offset = Duration::try_seconds(bounds[0] as i64)
        .ok_or_else(|| err("precipitation start offset overflows"))?;
    let end_offset = Duration::try_seconds(bounds[1] as i64)
        .ok_or_else(|| err("precipitation end offset overflows"))?;
    let start = origin
        .checked_add_signed(start_offset)
        .ok_or_else(|| err("precipitation start time overflows"))?;
    let end = origin
        .checked_add_signed(end_offset)
        .ok_or_else(|| err("precipitation end time overflows"))?;
    if (end - start).num_seconds() != definition.accumulation_seconds {
        return Err(err(
            "precipitation time bounds have the wrong accumulation duration",
        ));
    }
    let header = file
        .root_group()?
        .attribute(&definition.root_header_attribute)?
        .read_string()?;
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
    let from_header = |key: &str| -> Result<DateTime<Utc>> {
        let value = fields
            .get(key)
            .ok_or_else(|| err(format!("precipitation header has no {key}")))?;
        Ok(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc))
    };
    let header_end = from_header(&definition.header_last_included_key)?
        + Duration::milliseconds(definition.header_last_included_tick_milliseconds);
    if from_header(&definition.header_start_key)? != start || header_end != end {
        return Err(err(
            "precipitation granule header disagrees with the native half-open time bounds",
        ));
    }
    Ok((start, end))
}

pub fn decode(
    path: &Path,
    definition: &Product,
    bounds: Option<[f64; 4]>,
    expected_end: Option<DateTime<Utc>>,
    geometry: &Path,
    out: &Path,
) -> Result<Meta> {
    let file = Hdf5File::open(path)?;
    let (start, end) = read_period(&file, definition)?;
    if expected_end.is_some_and(|wanted| wanted != end) {
        return Err(err(
            "requested endpoint differs from the precipitation granule interval",
        ));
    }
    let precipitation = file.dataset(&definition.value_variable)?;
    if text(&precipitation, "units")? != definition.value_units
        || text(&precipitation, "DimensionNames")? != definition.value_axes
    {
        return Err(err(
            "precipitation units or dimension order differ from the native product declaration",
        ));
    }
    // The supported axis layout is explicit metadata, never inferred by equal sizes.
    if definition.value_axes != "time,lon,lat" || definition.value_units != "mm/hr" {
        return Err(err(
            "the HDF precipitation converter requires an explicit time,lon,lat rate in mm/hr",
        ));
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
    let (nx, ny) = (longitude.len(), latitude.len());
    if precipitation.shape() != [1, nx as u64, ny as u64]
        || latitude
            .iter()
            .chain(longitude.iter())
            .any(|x| !x.is_finite())
    {
        return Err(err("precipitation data and geographic axis shapes differ"));
    }
    let quality_dataset = file.dataset(&definition.quality_variable)?;
    if quality_dataset.shape() != precipitation.shape()
        || text(&quality_dataset, "DimensionNames")? != definition.value_axes
    {
        return Err(err(
            "precipitation quality index does not share the declared field axes",
        ));
    }
    let rates = unpacked(&precipitation, true)?;
    let quality = unpacked(&quality_dataset, true)?;
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
    for &j in &rows {
        for &i in &columns {
            let index = i * ny + j;
            let q = quality[index];
            let rate = rates[index];
            let valid = rate.is_finite() && q.is_finite();
            if (rate.is_finite() && rate < 0.0)
                || (q.is_finite()
                    && (q < definition.quality_range[0] || q > definition.quality_range[1]))
            {
                return Err(err(
                    "precipitation rate or source quality index is outside its declared range",
                ));
            }
            lat_out.push(latitude[j]);
            lon_out.push(longitude[i]);
            amount.push(if valid { rate * hours } else { 0.0 });
            observed.push(valid);
            retained_quality.push(if q.is_finite() { q } else { 0.0 });
            quality_valid.push(q.is_finite());
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
        "source_variable":definition.quality_variable,"range":definition.quality_range,"source_sha256":written.provenance.sha256,
        "geometry_sha256":written.geometry_sha256,"arrays":arrays,"payload_bytes":payload.len(),"content_sha256":payload_digest(&payload)}),
        &payload,
    )?;
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
        assert!(super::product("imerg-early").is_err());
    }
}
