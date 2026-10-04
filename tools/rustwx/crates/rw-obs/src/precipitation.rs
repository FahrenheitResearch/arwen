//! Native precipitation observation decoding and exact interval accumulation.
//!
//! Product semantics are table data. Missing source cells remain missing through
//! every accumulation, and overlapping or incomplete periods are refused rather
//! than rescaled into a precipitation total.
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use grib_core::grib2::{grid_latlon, unpack_message, Grib2File};
use serde::{Deserialize, Serialize};

use crate::pack::{
    decode_pack, payload_digest, validate_arrays, write_pack, ArrayEntry, PayloadBuilder,
    GEO_SCHEMA, GRID_SCHEMA,
};
use crate::seam::{
    seam_bounds, seam_time, wrap_longitude, Provenance, QUANTITY_PRECIPITATION_ACCUMULATION,
    UNITS_MM,
};
use crate::{absolute_uri, err, gunzip_if_wrapped, hex_sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Product {
    pub name: String,
    pub source: String,
    pub discipline: u8,
    pub category: u8,
    pub parameter: u8,
    pub center: u16,
    pub level_type: u8,
    pub level_value: f64,
    pub units: String,
    pub accumulation_seconds: i64,
    pub missing_values: Vec<f64>,
}

pub fn product(name: &str) -> Result<Product> {
    #[derive(Deserialize)]
    struct Table {
        schema: String,
        products: Vec<Product>,
    }
    let table: Table = serde_json::from_str(include_str!("../data/precipitation-products.json"))?;
    if table.schema != "gpuwm-obs.precipitation-products.v1" {
        return Err(err("unknown precipitation product table schema"));
    }
    let selected = table
        .products
        .into_iter()
        .find(|row| row.name == name)
        .ok_or_else(|| err(format!("no precipitation metadata for product {name:?}")))?;
    if selected.units != UNITS_MM || selected.accumulation_seconds <= 0 {
        return Err(err(
            "precipitation metadata needs millimetres and a positive accumulation period",
        ));
    }
    Ok(selected)
}

pub fn instant(raw: &str) -> Result<DateTime<Utc>> {
    rw_nexrad::s3::parse_time(raw)
}

pub fn bbox(raw: &str) -> Result<[f64; 4]> {
    let values = raw
        .split(',')
        .map(|item| item.parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.len() != 4
        || values.iter().any(|x| !x.is_finite())
        || values[0] < -180.0
        || values[2] >= 180.0
        || values[0] >= values[2]
        || values[1] < -90.0
        || values[3] > 90.0
        || values[1] >= values[3]
    {
        return Err(err("bbox must be finite W,S,E,N within the longitude/latitude range, without crossing the antimeridian"));
    }
    Ok([values[0], values[1], values[2], values[3]])
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Grid {
    pub kind: String,
    pub nx: usize,
    pub ny: usize,
    pub source_nx: usize,
    pub source_ny: usize,
    pub i_start: usize,
    pub j_start: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Mask {
    pub masked_cells: usize,
    pub zero_cells: usize,
    pub positive_cells: usize,
    pub observed_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SourceFrame {
    pub path: String,
    pub sha256: String,
    pub accumulation_start: String,
    pub valid_time: String,
    pub source: Provenance,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Meta {
    pub schema: String,
    pub status: String,
    pub quantity: String,
    pub units: String,
    pub valid_time: String,
    pub accumulation_start: String,
    pub accumulation_seconds: i64,
    pub accumulation_hours: f64,
    pub provenance: Provenance,
    pub geometry_sha256: String,
    pub grid: Grid,
    pub sentinels: Mask,
    pub value_min_mm: f64,
    pub value_max_mm: f64,
    pub arrays: BTreeMap<String, ArrayEntry>,
    pub payload_bytes: usize,
    pub content_sha256: String,
    #[serde(default)]
    pub source_frames: Vec<SourceFrame>,
}

#[derive(Clone)]
pub struct Frame {
    pub meta: Meta,
    pub values: Vec<f64>,
    pub valid: Vec<bool>,
}

fn mask(values: &[f64], valid: &[bool]) -> Result<(Mask, f64, f64)> {
    if values.len() != valid.len() || values.is_empty() {
        return Err(err(
            "precipitation values and mask must share a nonempty grid",
        ));
    }
    let (low, high) = seam_bounds(QUANTITY_PRECIPITATION_ACCUMULATION).unwrap();
    let mut zero = 0;
    let mut wet = 0;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for (&value, &observed) in values.iter().zip(valid) {
        if !value.is_finite() {
            return Err(err("precipitation pack contains a nonfinite stored value"));
        }
        if !observed {
            continue;
        }
        if value < low || value > high {
            return Err(err(format!(
                "observed precipitation {value} mm is outside [{low},{high}]"
            )));
        }
        if value == 0.0 {
            zero += 1;
        } else {
            wet += 1;
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    if zero + wet == 0 {
        return Err(err("precipitation field has no observed cells"));
    }
    Ok((
        Mask {
            masked_cells: values.len() - zero - wet,
            zero_cells: zero,
            positive_cells: wet,
            observed_fraction: (zero + wet) as f64 / values.len() as f64,
        },
        minimum,
        maximum,
    ))
}

pub fn write_frame(path: &Path, mut frame: Frame) -> Result<Meta> {
    let shape = vec![frame.meta.grid.ny, frame.meta.grid.nx];
    if frame.meta.grid.nx.checked_mul(frame.meta.grid.ny) != Some(frame.values.len()) {
        return Err(err("precipitation field shape differs from its grid"));
    }
    let (statistics, minimum, maximum) = mask(&frame.values, &frame.valid)?;
    let mut builder = PayloadBuilder::new();
    builder.push_f64("values", &frame.values, shape.clone());
    builder.push_mask("valid", &frame.valid, shape);
    let (payload, arrays) = builder.finish();
    frame.meta.sentinels = statistics;
    frame.meta.value_min_mm = minimum;
    frame.meta.value_max_mm = maximum;
    frame.meta.arrays = arrays;
    frame.meta.payload_bytes = payload.len();
    frame.meta.content_sha256 = payload_digest(&payload);
    write_pack(path, &frame.meta, &payload)?;
    Ok(frame.meta)
}

pub fn read_frame(path: &Path) -> Result<Frame> {
    let raw = std::fs::read(path)?;
    let (meta, payload): (Meta, Vec<u8>) = decode_pack(&raw)?;
    if meta.schema != GRID_SCHEMA
        || meta.quantity != QUANTITY_PRECIPITATION_ACCUMULATION
        || meta.units != UNITS_MM
    {
        return Err(err(
            "accumulation input is not a precipitation observation pack in mm",
        ));
    }
    if meta.status != "READY" || meta.provenance.is_stub {
        return Err(err(
            "precipitation scoring needs ready observations, not stub or incomplete inputs",
        ));
    }
    if meta.content_sha256 != payload_digest(&payload) || meta.payload_bytes != payload.len() {
        return Err(err(
            "precipitation observation payload digest or length differs",
        ));
    }
    validate_arrays(&meta.arrays, payload.len())?;
    let shape = vec![meta.grid.ny, meta.grid.nx];
    let values_entry = meta
        .arrays
        .get("values")
        .ok_or_else(|| err("precipitation pack has no values array"))?;
    let valid_entry = meta
        .arrays
        .get("valid")
        .ok_or_else(|| err("precipitation pack has no validity array"))?;
    if values_entry.shape != shape
        || valid_entry.shape != shape
        || values_entry.dtype != "<f8"
        || valid_entry.dtype != "<u1"
    {
        return Err(err(
            "precipitation pack arrays differ from its grid or typed seam",
        ));
    }
    let values = payload[values_entry.offset..values_entry.offset + values_entry.bytes]
        .chunks_exact(8)
        .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mask_bytes = &payload[valid_entry.offset..valid_entry.offset + valid_entry.bytes];
    if mask_bytes.iter().any(|&value| value > 1) {
        return Err(err("precipitation validity is not boolean"));
    }
    let valid = mask_bytes
        .iter()
        .map(|&value| value == 1)
        .collect::<Vec<_>>();
    let (statistics, minimum, maximum) = mask(&values, &valid)?;
    if statistics != meta.sentinels || minimum != meta.value_min_mm || maximum != meta.value_max_mm
    {
        return Err(err(format!("precipitation mask counts or extrema differ from the stored values: computed {statistics:?}, [{minimum:?},{maximum:?}], recorded {:?}, [{:?},{:?}]",meta.sentinels,meta.value_min_mm,meta.value_max_mm)));
    }
    let seconds = (instant(&meta.valid_time)? - instant(&meta.accumulation_start)?).num_seconds();
    if seconds <= 0
        || seconds != meta.accumulation_seconds
        || meta.accumulation_hours != seconds as f64 / 3600.0
    {
        return Err(err(
            "precipitation pack accumulation period disagrees with its endpoints",
        ));
    }
    if meta.geometry_sha256.len() != 64
        || !meta.geometry_sha256.bytes().all(|c| c.is_ascii_hexdigit())
    {
        return Err(err("precipitation pack needs its native geometry digest"));
    }
    Ok(Frame {
        meta,
        values,
        valid,
    })
}

pub fn decode(
    file: &Path,
    definition: &Product,
    bounds: Option<[f64; 4]>,
    expected_end: Option<DateTime<Utc>>,
    geometry: &Path,
    out: &Path,
) -> Result<Meta> {
    let raw = std::fs::read(file)?;
    let (bytes, _) = gunzip_if_wrapped(&raw, "precipitation observation")?;
    let parsed = Grib2File::from_bytes(&bytes)?;
    let matches = parsed
        .messages
        .iter()
        .filter(|message| {
            message.discipline == definition.discipline
                && message.product.parameter_category == definition.category
                && message.product.parameter_number == definition.parameter
                && message.identification.center_id == definition.center
                && message.product.level_type == definition.level_type
                && message.product.level_value == definition.level_value
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(err(format!(
            "product {} needs exactly one matching GRIB observation; found {}",
            definition.name,
            matches.len()
        )));
    }
    let message = matches[0];
    if message.grid.template != 0
        || message.product.template != 0
        || message.product.forecast_time != 0
    {
        return Err(err("precipitation table expects a regular-latlon observation with PDT0 and zero forecast lead; another time encoding needs explicit metadata"));
    }
    let end = message.reference_time.and_utc();
    if expected_end.is_some_and(|expected| expected != end) {
        return Err(err(
            "requested precipitation endpoint differs from the GRIB observation time",
        ));
    }
    let start = end - Duration::seconds(definition.accumulation_seconds);
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    let cells = nx
        .checked_mul(ny)
        .ok_or_else(|| err("precipitation grid overflows"))?;
    let raw_values = unpack_message(message)?;
    let (raw_latitude, raw_longitude) = grid_latlon(&message.grid)?;
    if raw_values.len() != cells || raw_latitude.len() != cells || raw_longitude.len() != cells {
        return Err(err(
            "native precipitation values and geometry have different shapes",
        ));
    }
    let (mut i0, mut i1, mut j0, mut j1) = (0, nx, 0, ny);
    if let Some([west, south, east, north]) = bounds {
        let columns = (0..nx)
            .filter(|&i| {
                let lon = wrap_longitude(raw_longitude[i]);
                lon >= west && lon <= east
            })
            .collect::<Vec<_>>();
        let rows = (0..ny)
            .filter(|&j| raw_latitude[j * nx] >= south && raw_latitude[j * nx] <= north)
            .collect::<Vec<_>>();
        if columns.is_empty() || rows.is_empty() {
            return Err(err("bbox selects no precipitation cells"));
        }
        i0 = columns[0];
        i1 = columns[columns.len() - 1] + 1;
        j0 = rows[0];
        j1 = rows[rows.len() - 1] + 1;
    }
    let grid = Grid {
        kind: "regular_latlon".into(),
        nx: i1 - i0,
        ny: j1 - j0,
        source_nx: nx,
        source_ny: ny,
        i_start: i0,
        j_start: j0,
    };
    let mut latitude = Vec::new();
    let mut longitude = Vec::new();
    let mut values = Vec::new();
    let mut valid = Vec::new();
    for j in j0..j1 {
        for i in i0..i1 {
            let index = j * nx + i;
            latitude.push(raw_latitude[index]);
            longitude.push(wrap_longitude(raw_longitude[index]));
            let value = raw_values[index];
            let observed = value.is_finite() && !definition.missing_values.contains(&value);
            if observed && value < 0.0 {
                return Err(err(format!(
                    "unknown negative precipitation sentinel {value}"
                )));
            }
            values.push(if observed { value } else { 0.0 });
            valid.push(observed);
        }
    }
    let mut geometry_builder = PayloadBuilder::new();
    geometry_builder.push_f64("latitude", &latitude, vec![grid.ny, grid.nx]);
    geometry_builder.push_f64("longitude", &longitude, vec![grid.ny, grid.nx]);
    let (geo_payload, geo_arrays) = geometry_builder.finish();
    let geometry_sha256 = payload_digest(&geo_payload);
    let geo_meta = serde_json::json!({"schema":GEO_SCHEMA,"status":"READY","grid":grid,
        "arrays":geo_arrays,"payload_bytes":geo_payload.len(),"content_sha256":geometry_sha256});
    if geometry.exists() {
        let (existing, payload): (serde_json::Value, Vec<u8>) =
            decode_pack(&std::fs::read(geometry)?)?;
        if existing["schema"] != GEO_SCHEMA || payload != geo_payload {
            return Err(err("existing precipitation geometry differs from this field; use a distinct geometry pack"));
        }
    } else {
        write_pack(geometry, &geo_meta, &geo_payload)?;
    }
    let (statistics, minimum, maximum) = mask(&values, &valid)?;
    let meta = Meta {
        schema: GRID_SCHEMA.into(),
        status: "READY".into(),
        quantity: QUANTITY_PRECIPITATION_ACCUMULATION.into(),
        units: UNITS_MM.into(),
        valid_time: seam_time(end),
        accumulation_start: seam_time(start),
        accumulation_seconds: definition.accumulation_seconds,
        accumulation_hours: definition.accumulation_seconds as f64 / 3600.0,
        provenance: Provenance::new(
            &definition.source,
            &definition.name,
            absolute_uri(file),
            hex_sha256(&raw),
            seam_time(Utc::now()),
        ),
        geometry_sha256,
        grid,
        sentinels: statistics,
        value_min_mm: minimum,
        value_max_mm: maximum,
        arrays: BTreeMap::new(),
        payload_bytes: 0,
        content_sha256: String::new(),
        source_frames: Vec::new(),
    };
    write_frame(
        out,
        Frame {
            meta,
            values,
            valid,
        },
    )
}

pub fn accumulate(
    paths: &[PathBuf],
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    out: &Path,
) -> Result<Meta> {
    if paths.is_empty() || end <= start {
        return Err(err("accumulation requires inputs and an increasing window"));
    }
    if out.exists()
        && paths
            .iter()
            .any(|path| path.canonicalize().ok() == out.canonicalize().ok())
    {
        return Err(err(
            "accumulation output must not replace an input observation pack",
        ));
    }
    let mut frames = paths
        .iter()
        .map(|path| Ok((path.clone(), read_frame(path)?)))
        .collect::<Result<Vec<_>>>()?;
    frames.sort_by(|left, right| {
        left.1
            .meta
            .accumulation_start
            .cmp(&right.1.meta.accumulation_start)
    });
    let first = &frames[0].1;
    let mut result = first.clone();
    result.values.fill(0.0);
    result.valid.fill(true);
    result.meta.source_frames.clear();
    let mut next = start;
    for (path, frame) in &frames {
        if instant(&frame.meta.accumulation_start)? != next {
            return Err(err("precipitation input periods overlap or leave a gap; no rescaling or double counting is allowed"));
        }
        if frame.meta.geometry_sha256 != result.meta.geometry_sha256
            || frame.meta.grid != result.meta.grid
        {
            return Err(err("precipitation inputs have different native geometry"));
        }
        next = instant(&frame.meta.valid_time)?;
        for index in 0..result.values.len() {
            result.valid[index] &= frame.valid[index];
            if result.valid[index] {
                result.values[index] += frame.values[index];
            } else {
                result.values[index] = 0.0;
            }
        }
        result.meta.source_frames.push(SourceFrame {
            path: absolute_uri(path),
            sha256: hex_sha256(&std::fs::read(path)?),
            accumulation_start: frame.meta.accumulation_start.clone(),
            valid_time: frame.meta.valid_time.clone(),
            source: frame.meta.provenance.clone(),
        });
    }
    if next != end {
        return Err(err(
            "precipitation inputs do not end at the requested accumulation endpoint",
        ));
    }
    let manifest = out.with_extension("sources.json");
    let body = serde_json::to_vec_pretty(&result.meta.source_frames)?;
    if let Some(parent) = manifest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&manifest, &body)?;
    result.meta.provenance = Provenance::new(
        "derived-observation",
        "sum-nonoverlapping-precipitation-intervals",
        absolute_uri(&manifest),
        hex_sha256(&body),
        seam_time(Utc::now()),
    );
    result.meta.valid_time = seam_time(end);
    result.meta.accumulation_start = seam_time(start);
    result.meta.accumulation_seconds = (end - start).num_seconds();
    result.meta.accumulation_hours = result.meta.accumulation_seconds as f64 / 3600.0;
    write_frame(out, result)
}

pub fn verify(path: &Path) -> Result<Meta> {
    verify_inner(
        path,
        &mut std::collections::BTreeSet::new(),
        &mut BTreeMap::new(),
    )
}

fn verify_inner(
    path: &Path,
    active: &mut std::collections::BTreeSet<PathBuf>,
    checked: &mut BTreeMap<PathBuf, Meta>,
) -> Result<Meta> {
    let canonical = path.canonicalize()?;
    if let Some(meta) = checked.get(&canonical) {
        return Ok(meta.clone());
    }
    if active.len() >= 64 || !active.insert(canonical.clone()) {
        return Err(err(
            "precipitation source graph is cyclic or exceeds 64 accumulation levels",
        ));
    }
    let frame = read_frame(path)?;
    let origin = Path::new(&frame.meta.provenance.uri);
    if hex_sha256(&std::fs::read(origin)?) != frame.meta.provenance.sha256 {
        return Err(err(
            "precipitation source bytes differ from their recorded digest",
        ));
    }
    for source in &frame.meta.source_frames {
        if hex_sha256(&std::fs::read(&source.path)?) != source.sha256 {
            return Err(err("accumulation source pack digest differs"));
        }
        let child = verify_inner(Path::new(&source.path), active, checked)?;
        if child.accumulation_start != source.accumulation_start
            || child.valid_time != source.valid_time
            || serde_json::to_value(&child.provenance)? != serde_json::to_value(&source.source)?
        {
            return Err(err(
                "accumulation source metadata differs from its verified pack",
            ));
        }
    }
    active.remove(&canonical);
    checked.insert(canonical, frame.meta.clone());
    Ok(frame.meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn product_table_pins_identity_units_period_and_missing_values() {
        let hour = product("MultiSensor_QPE_01H_Pass2_00.00").unwrap();
        assert_eq!(
            (hour.discipline, hour.category, hour.parameter),
            (209, 6, 37)
        );
        assert_eq!(hour.units, "mm");
        assert_eq!(hour.accumulation_seconds, 3600);
        assert_eq!(hour.missing_values, vec![-1.0, -3.0]);
        assert_eq!(
            product("MultiSensor_QPE_12H_Pass2_00.00")
                .unwrap()
                .accumulation_seconds,
            43200
        );
        assert!(product("MergedReflectivityQCComposite_00.50").is_err());
    }
    #[test]
    fn missing_cells_and_dry_cells_are_different_observations() {
        let (counts, minimum, maximum) = mask(&[0.0, 0.0, 2.5], &[false, true, true]).unwrap();
        assert_eq!(
            (
                counts.masked_cells,
                counts.zero_cells,
                counts.positive_cells
            ),
            (1, 1, 1)
        );
        assert_eq!((minimum, maximum), (0.0, 2.5));
        assert!(mask(&[-1.0], &[true]).is_err());
        assert!(mask(&[0.0], &[false]).is_err());
    }
    #[test]
    fn bbox_is_not_silently_wrapped_or_reversed() {
        assert_eq!(bbox("-100,30,-90,40").unwrap(), [-100.0, 30.0, -90.0, 40.0]);
        for raw in ["170,-10,-170,10", "0,0,NaN,10", "10,20,0,30", "0,0,10,95"] {
            assert!(bbox(raw).is_err());
        }
    }

    fn directory() -> PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rw-precip-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn fixture(
        directory: &Path,
        label: &str,
        start: &str,
        end: &str,
        values: Vec<f64>,
        valid: Vec<bool>,
    ) -> PathBuf {
        let raw = directory.join(format!("{label}.raw"));
        std::fs::write(&raw, label).unwrap();
        let seconds = (instant(end).unwrap() - instant(start).unwrap()).num_seconds();
        let meta = Meta {
            schema: GRID_SCHEMA.into(),
            status: "READY".into(),
            quantity: QUANTITY_PRECIPITATION_ACCUMULATION.into(),
            units: UNITS_MM.into(),
            valid_time: end.into(),
            accumulation_start: start.into(),
            accumulation_seconds: seconds,
            accumulation_hours: seconds as f64 / 3600.0,
            provenance: Provenance::new(
                "test-fixture",
                "analytical",
                absolute_uri(&raw),
                hex_sha256(label.as_bytes()),
                "2000-01-01T00:00:00",
            ),
            geometry_sha256: "a".repeat(64),
            grid: Grid {
                kind: "test".into(),
                nx: 2,
                ny: 1,
                source_nx: 2,
                source_ny: 1,
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
            source_frames: Vec::new(),
        };
        let path = directory.join(format!("{label}.obspack"));
        write_frame(
            &path,
            Frame {
                meta,
                values,
                valid,
            },
        )
        .unwrap();
        path
    }

    #[test]
    fn accumulation_sorts_intervals_and_preserves_missing_cells() {
        let root = directory();
        let a = fixture(
            &root,
            "first",
            "2000-01-01T00:00:00",
            "2000-01-01T00:30:00",
            vec![1.0, 2.0],
            vec![true, true],
        );
        let b = fixture(
            &root,
            "second",
            "2000-01-01T00:30:00",
            "2000-01-01T01:00:00",
            vec![3.0, 0.0],
            vec![true, false],
        );
        let out = root.join("total.obspack");
        accumulate(
            &[b, a],
            instant("2000-01-01T00:00:00").unwrap(),
            instant("2000-01-01T01:00:00").unwrap(),
            &out,
        )
        .unwrap();
        let result = read_frame(&out).unwrap();
        assert_eq!(result.values, vec![4.0, 0.0]);
        assert_eq!(result.valid, vec![true, false]);
        assert_eq!(result.meta.accumulation_seconds, 3600);
        verify(&out).unwrap();
    }

    #[test]
    fn missing_overlapping_wrong_grid_and_short_periods_are_refused() {
        let root = directory();
        let a = fixture(
            &root,
            "first",
            "2000-01-01T00:00:00",
            "2000-01-01T01:00:00",
            vec![1.0, 2.0],
            vec![true, true],
        );
        let b = fixture(
            &root,
            "second",
            "2000-01-01T02:00:00",
            "2000-01-01T03:00:00",
            vec![3.0, 4.0],
            vec![true, true],
        );
        let out = root.join("total.obspack");
        let start = instant("2000-01-01T00:00:00").unwrap();
        let end = instant("2000-01-01T03:00:00").unwrap();
        assert!(accumulate(&[a.clone(), b], start, end, &out)
            .unwrap_err()
            .to_string()
            .contains("gap"));
        assert!(accumulate(&[a.clone(), a.clone()], start, end, &out)
            .unwrap_err()
            .to_string()
            .contains("overlap"));
        assert!(accumulate(&[a.clone()], start, end, &out)
            .unwrap_err()
            .to_string()
            .contains("endpoint"));
        assert!(accumulate(&[a.clone()], start, end, &a)
            .unwrap_err()
            .to_string()
            .contains("replace"));
        let c = fixture(
            &root,
            "third",
            "2000-01-01T01:00:00",
            "2000-01-01T02:00:00",
            vec![1.0, 2.0],
            vec![true, true],
        );
        let mut altered = read_frame(&c).unwrap();
        altered.meta.geometry_sha256 = "b".repeat(64);
        write_frame(&c, altered).unwrap();
        assert!(accumulate(
            &[a, c],
            start,
            instant("2000-01-01T02:00:00").unwrap(),
            &out
        )
        .unwrap_err()
        .to_string()
        .contains("geometry"));
    }

    #[test]
    fn source_float_extrema_round_trip_without_changing_stored_values() {
        let root = directory();
        // An actual half-hour IMERG maximum. The default JSON parser can round
        // this shortest decimal representation to the adjacent f64 value.
        let maximum = 23.014999389648438;
        let path = fixture(
            &root,
            "roundtrip",
            "2000-01-01T00:00:00",
            "2000-01-01T00:30:00",
            vec![0.0, maximum],
            vec![true, true],
        );
        let frame = read_frame(&path).unwrap();
        assert_eq!(frame.values[1].to_bits(), maximum.to_bits());
        assert_eq!(frame.meta.value_max_mm.to_bits(), maximum.to_bits());
        verify(&path).unwrap();
    }

    #[test]
    fn source_corruption_is_not_hidden_by_a_valid_pack() {
        let root = directory();
        let path = fixture(
            &root,
            "single",
            "2000-01-01T00:00:00",
            "2000-01-01T01:00:00",
            vec![0.0, 2.0],
            vec![true, true],
        );
        verify(&path).unwrap();
        std::fs::write(root.join("single.raw"), "changed").unwrap();
        assert!(verify(&path)
            .unwrap_err()
            .to_string()
            .contains("source bytes"));
    }

    #[test]
    fn nested_accumulation_rechecks_the_ultimate_raw_observations() {
        let root = directory();
        let a = fixture(
            &root,
            "nested-first",
            "2000-01-01T00:00:00",
            "2000-01-01T00:30:00",
            vec![1.0, 2.0],
            vec![true, true],
        );
        let b = fixture(
            &root,
            "nested-second",
            "2000-01-01T00:30:00",
            "2000-01-01T01:00:00",
            vec![3.0, 4.0],
            vec![true, true],
        );
        let hourly = root.join("nested-hour.obspack");
        accumulate(
            &[a, b],
            instant("2000-01-01T00:00:00").unwrap(),
            instant("2000-01-01T01:00:00").unwrap(),
            &hourly,
        )
        .unwrap();
        let c = fixture(
            &root,
            "nested-third",
            "2000-01-01T01:00:00",
            "2000-01-01T02:00:00",
            vec![5.0, 6.0],
            vec![true, true],
        );
        let total = root.join("nested-total.obspack");
        accumulate(
            &[hourly, c],
            instant("2000-01-01T00:00:00").unwrap(),
            instant("2000-01-01T02:00:00").unwrap(),
            &total,
        )
        .unwrap();
        verify(&total).unwrap();
        std::fs::write(root.join("nested-first.raw"), "changed").unwrap();
        assert!(verify(&total)
            .unwrap_err()
            .to_string()
            .contains("source bytes"));
    }
}
