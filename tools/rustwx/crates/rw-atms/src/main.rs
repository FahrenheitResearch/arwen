//! `rw_atms` -- the ATMS front door.
//!
//! Three passes over the JPSS ATMS Sensor Data Records (IDPS HDF5
//! granules: `SATMS_*` brightness temperatures, `GATMO_*` geolocation),
//! shaped like the `rw_netcdf` pair so one idiom reads every format:
//!
//! * `rw_atms inventory FILE` prints a JSON description of any HDF5 file:
//!   every group and dataset with its path, shape, dtype and attributes.
//!   No values are read.  It exists so the decode below can be checked
//!   against the file's own vocabulary rather than a format memo.
//! * `rw_atms decode OUTDIR SDR GEO [SDR GEO ...]` decodes granule pairs
//!   (an SDR and the GEO file of the same granule, paired by position)
//!   into flat little-endian arrays concatenated along the scan axis --
//!   brightness temperature in kelvin with the IDPS scale and offset
//!   applied and every fill replaced by NaN, beam latitude, longitude,
//!   satellite zenith and azimuth, solar zenith, and the beam time as
//!   Unix seconds -- plus a `metadata.json` receipt naming each input,
//!   its SHA-256, the factors applied and the array shapes.
//! * `rw_atms thin OUTDIR DECODED --latitudes FILE --nlon N --origin-unix-s T
//!   --bin-s S [--max-zenith DEG]` colocates the decoded beams onto a
//!   grid of latitude rings (one centre per line in FILE, any monotonic
//!   order) crossed with `N` equally spaced longitudes from 0 degrees, in
//!   time bins of `S` seconds from `T`; every cell's per-channel mean,
//!   standard deviation and count are written with the mean geometry.
//!   This is the "one report per grid cell per hour" thinning of the
//!   observation design, done where the design puts it.
//!
//! IDPS time is IET: microseconds since 1958-01-01T00:00:00 TAI.  The
//! decode converts to Unix seconds with the 37 s TAI-UTC offset in force
//! since 2017 and writes that constant into the receipt.
//!
//! Everything is read through `hdf5-reader`, the vendored pure-Rust HDF5
//! decoder netcrust already rides.  There is no C HDF5 library anywhere
//! in this path.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use hdf5_reader::group::Group;
use hdf5_reader::{Attribute, Dataset, Datatype, Hdf5File};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// The contract marker, readable straight out of the binary bytes.
const ABI: &str = concat!(
    "gpuwm-rw-atms-inventory-v1\tgroups\tdatasets\tattributes",
    "\tgpuwm-rw-atms-decode-v1\tgranules\tarrays\tbrightness_temperature_k",
    "\tgpuwm-rw-atms-thin-v1\tcells\ttb_mean_k",
);

const INVENTORY_SCHEMA: &str = "gpuwm-rw-atms-inventory-v1";
const DECODE_SCHEMA: &str = "gpuwm-rw-atms-decode-v1";
const THIN_SCHEMA: &str = "gpuwm-rw-atms-thin-v1";

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: embedded as a literal so
/// the release cut can read the source revision out of the bytes.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const USAGE: &str = "\
usage: rw_atms inventory FILE
       rw_atms decode OUTPUT_DIR SDR_FILE GEO_FILE [SDR_FILE GEO_FILE ...]
       rw_atms decode OUTPUT_DIR @PAIRS_FILE      (one SDR<TAB>GEO line per pair)
       rw_atms thin OUTPUT_DIR DECODED_DIR --latitudes FILE --nlon N
                    --origin-unix-s SECONDS --bin-s SECONDS [--max-zenith DEG]
       rw_atms --abi | --help

  inventory  print a JSON description of any HDF5 file (no values are read)
  decode     decode ATMS SDR/GEO granule pairs into OUTPUT_DIR as flat
             little-endian arrays plus metadata.json
  thin       colocate decoded beams onto latitude rings x N longitudes x
             time bins and write the cell means with counts and spreads
";

const SDR_GROUP: &str = "/All_Data/ATMS-SDR_All";
const GEO_GROUP: &str = "/All_Data/ATMS-SDR-GEO_All";
const CHANNELS: usize = 22;
const FOVS: usize = 96;
/// Beam k of a scan points at (k - 47.5) * 1.11 degrees from nadir.
const SCAN_STEP_DEG: f64 = 1.11;
/// Seconds from 1958-01-01 (the IET epoch) to 1970-01-01.
const IET_EPOCH_TO_UNIX_S: f64 = 378_691_200.0;
/// TAI minus UTC, in force since 2017-01-01 (no leap second since).
const TAI_MINUS_UTC_S: f64 = 37.0;
/// IDPS 16-bit fills occupy the top of the unsigned range (65528..=65535).
const U16_FILL_FLOOR: u16 = 65528;
/// IDPS float fills are -999.x sentinels.
const F32_FILL_CEILING: f32 = -999.0;

#[derive(Serialize)]
struct Provenance {
    tool: &'static str,
    source_rev: &'static str,
}

fn provenance() -> Provenance {
    Provenance {
        tool: "rw_atms",
        source_rev: env!("GPUWM_BRIDGE_SOURCE_REV"),
    }
}

// ---------------------------------------------------------------- inventory

#[derive(Serialize)]
struct Inventory {
    schema: &'static str,
    metadata: Provenance,
    file: String,
    sha256: String,
    user_block_bytes: usize,
    groups: Vec<GroupRecord>,
    datasets: Vec<DatasetRecord>,
}

#[derive(Serialize)]
struct GroupRecord {
    path: String,
    attributes: BTreeMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct DatasetRecord {
    path: String,
    shape: Vec<u64>,
    dtype: String,
    attributes: BTreeMap<String, serde_json::Value>,
}

fn dtype_name(dtype: &Datatype) -> String {
    match dtype {
        Datatype::FixedPoint { size, signed, .. } => {
            format!("{}{}", if *signed { "int" } else { "uint" }, u32::from(*size) * 8)
        }
        Datatype::FloatingPoint { size, .. } => format!("float{}", u32::from(*size) * 8),
        Datatype::String { .. } => "string".to_string(),
        Datatype::Compound { fields, .. } => format!("compound[{}]", fields.len()),
        Datatype::Array { base, dims } => format!("array{:?}<{}>", dims, dtype_name(base)),
        Datatype::Enum { base, .. } => format!("enum<{}>", dtype_name(base)),
        Datatype::VarLen { base } => format!("varlen<{}>", dtype_name(base)),
        Datatype::Opaque { size, .. } => format!("opaque[{size}]"),
        Datatype::Reference { .. } => "reference".to_string(),
        Datatype::Bitfield { size, .. } => format!("bitfield{}", u32::from(*size) * 8),
    }
}

fn number(value: f64) -> serde_json::Value {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

fn numbers<T: Copy + Into<f64>>(values: Vec<T>) -> serde_json::Value {
    if values.len() == 1 {
        number(values[0].into())
    } else {
        serde_json::Value::Array(values.into_iter().map(|v| number(v.into())).collect())
    }
}

fn attribute_json(attribute: &Attribute) -> serde_json::Value {
    let unreadable = |what: &str| serde_json::Value::String(format!("<unreadable {what}>"));
    match &attribute.datatype {
        Datatype::String { .. } => match attribute.read_strings() {
            Ok(mut strings) if strings.len() == 1 => serde_json::Value::String(strings.remove(0)),
            Ok(strings) => serde_json::Value::Array(
                strings.into_iter().map(serde_json::Value::String).collect(),
            ),
            Err(_) => match attribute.read_string() {
                Ok(text) => serde_json::Value::String(text),
                Err(_) => unreadable("string"),
            },
        },
        Datatype::FloatingPoint { size: 4, .. } => attribute
            .read_1d::<f32>()
            .map(numbers)
            .unwrap_or_else(|_| unreadable("float32")),
        Datatype::FloatingPoint { size: 8, .. } => attribute
            .read_1d::<f64>()
            .map(numbers)
            .unwrap_or_else(|_| unreadable("float64")),
        Datatype::FixedPoint { size, signed, .. } => match (size, signed) {
            (1, true) => attribute.read_1d::<i8>().map(numbers),
            (1, false) => attribute.read_1d::<u8>().map(numbers),
            (2, true) => attribute.read_1d::<i16>().map(numbers),
            (2, false) => attribute.read_1d::<u16>().map(numbers),
            (4, true) => attribute.read_1d::<i32>().map(numbers),
            (4, false) => attribute.read_1d::<u32>().map(numbers),
            (8, true) => attribute
                .read_1d::<i64>()
                .map(|v| numbers(v.into_iter().map(|x| x as f64).collect::<Vec<f64>>())),
            (8, false) => attribute
                .read_1d::<u64>()
                .map(|v| numbers(v.into_iter().map(|x| x as f64).collect::<Vec<f64>>())),
            _ => Ok(unreadable("integer")),
        }
        .unwrap_or_else(|_| unreadable("integer")),
        other => serde_json::Value::String(format!("<{}>", dtype_name(other))),
    }
}

fn attributes_json(attributes: &[Attribute]) -> BTreeMap<String, serde_json::Value> {
    attributes
        .iter()
        .map(|attribute| (attribute.name.clone(), attribute_json(attribute)))
        .collect()
}

fn child_path(parent: &str, name: &str) -> String {
    if parent.is_empty() || parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

fn walk(group: &Group, path: &str, inventory: &mut Inventory) -> Result<(), String> {
    // An attribute block the reader cannot parse is recorded, not fatal:
    // the datasets are what the decode needs, and an inventory that dies
    // on a group's attributes says nothing about them.
    let attributes = match group.attributes() {
        Ok(attributes) => attributes_json(&attributes),
        Err(error) => BTreeMap::from([(
            "<attributes unreadable>".to_string(),
            serde_json::Value::String(error.to_string()),
        )]),
    };
    inventory.groups.push(GroupRecord {
        path: if path.is_empty() { "/".to_string() } else { path.to_string() },
        attributes,
    });
    let (groups, datasets) = group
        .members()
        .map_err(|e| format!("{path}: members: {e}"))?;
    for dataset in datasets {
        inventory.datasets.push(DatasetRecord {
            path: child_path(path, dataset.name()),
            shape: dataset.shape().to_vec(),
            dtype: dtype_name(dataset.dtype()),
            attributes: attributes_json(&dataset.attributes()),
        });
    }
    for child in groups {
        let child_name = child.name().to_string();
        walk(&child, &child_path(path, &child_name), inventory)?;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

/// IDPS files carry a 1024-byte XML user block ahead of the HDF5
/// signature (`<HDF_UserBlock>...`), and the superblock's addresses are
/// relative to the signature, not to byte zero.  `hdf5-reader` looks for
/// the signature at byte zero only, so the file is opened from the
/// signature onward, where every address is relative to the buffer.  The
/// HDF5 format allows the signature at 0, 512, 1024, 2048, ... bytes.
fn open_hdf5(path: &Path) -> Result<(Hdf5File, usize), String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut offset = 0usize;
    loop {
        if offset + HDF5_SIGNATURE.len() > bytes.len() {
            return Err(format!("{}: no HDF5 signature found", path.display()));
        }
        if bytes[offset..offset + HDF5_SIGNATURE.len()] == HDF5_SIGNATURE {
            break;
        }
        offset = if offset == 0 { 512 } else { offset * 2 };
    }
    let file = if offset == 0 {
        Hdf5File::from_vec(bytes)
    } else {
        Hdf5File::from_vec(bytes[offset..].to_vec())
    }
    .map_err(|e| format!("{}: open: {e}", path.display()))?;
    Ok((file, offset))
}

const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0D, 0x0A, 0x1A, 0x0A];

fn inventory(path: &Path) -> Result<Inventory, String> {
    let (file, user_block) = open_hdf5(path)?;
    let mut record = Inventory {
        schema: INVENTORY_SCHEMA,
        metadata: provenance(),
        file: path.display().to_string(),
        sha256: sha256_file(path)?,
        user_block_bytes: user_block,
        groups: Vec::new(),
        datasets: Vec::new(),
    };
    let root = file
        .root_group()
        .map_err(|e| format!("{}: root group: {e}", path.display()))?;
    walk(&root, "", &mut record)?;
    Ok(record)
}

// ------------------------------------------------------------------- decode

fn dataset(file: &Hdf5File, path: &str) -> Result<Dataset, String> {
    file.dataset(path).map_err(|e| format!("{path}: {e}"))
}

fn read_f32(file: &Hdf5File, path: &str) -> Result<(Vec<u64>, Vec<f32>), String> {
    let ds = dataset(file, path)?;
    let shape = ds.shape().to_vec();
    let values = match ds.dtype() {
        Datatype::FloatingPoint { size: 4, .. } => ds
            .read_array::<f32>()
            .map_err(|e| format!("{path}: read f32: {e}"))?
            .iter()
            .copied()
            .collect::<Vec<f32>>(),
        Datatype::FloatingPoint { size: 8, .. } => ds
            .read_array::<f64>()
            .map_err(|e| format!("{path}: read f64: {e}"))?
            .iter()
            .map(|v| *v as f32)
            .collect(),
        other => {
            return Err(format!(
                "{path}: expected a float dataset, found {}",
                dtype_name(other)
            ))
        }
    };
    Ok((shape, values))
}

fn read_i64(file: &Hdf5File, path: &str) -> Result<(Vec<u64>, Vec<i64>), String> {
    let ds = dataset(file, path)?;
    let shape = ds.shape().to_vec();
    let values = match ds.dtype() {
        Datatype::FixedPoint { size: 8, signed: true, .. } => ds
            .read_array::<i64>()
            .map_err(|e| format!("{path}: read i64: {e}"))?
            .iter()
            .copied()
            .collect::<Vec<i64>>(),
        Datatype::FixedPoint { size: 8, signed: false, .. } => ds
            .read_array::<u64>()
            .map_err(|e| format!("{path}: read u64: {e}"))?
            .iter()
            .map(|v| *v as i64)
            .collect(),
        other => {
            return Err(format!(
                "{path}: expected a 64-bit integer dataset, found {}",
                dtype_name(other)
            ))
        }
    };
    Ok((shape, values))
}

/// Raw 16-bit counts with the IDPS fill range mapped to None.
fn read_counts(file: &Hdf5File, path: &str) -> Result<(Vec<u64>, Vec<Option<u32>>), String> {
    let ds = dataset(file, path)?;
    let shape = ds.shape().to_vec();
    let values = match ds.dtype() {
        Datatype::FixedPoint { size: 2, signed: false, .. } => ds
            .read_array::<u16>()
            .map_err(|e| format!("{path}: read u16: {e}"))?
            .iter()
            .map(|v| (*v < U16_FILL_FLOOR).then_some(u32::from(*v)))
            .collect::<Vec<Option<u32>>>(),
        Datatype::FixedPoint { size: 2, signed: true, .. } => ds
            .read_array::<i16>()
            .map_err(|e| format!("{path}: read i16: {e}"))?
            .iter()
            .map(|v| (*v >= 0).then_some(*v as u32))
            .collect(),
        other => {
            return Err(format!(
                "{path}: expected a 16-bit integer dataset, found {}",
                dtype_name(other)
            ))
        }
    };
    Ok((shape, values))
}

fn fill_to_nan(values: Vec<f32>) -> Vec<f32> {
    values
        .into_iter()
        .map(|v| if v <= F32_FILL_CEILING || !v.is_finite() { f32::NAN } else { v })
        .collect()
}

fn iet_to_unix(iet_us: i64) -> f64 {
    if iet_us <= 0 {
        return f64::NAN;
    }
    iet_us as f64 * 1.0e-6 - IET_EPOCH_TO_UNIX_S - TAI_MINUS_UTC_S
}

fn expect_shape(what: &str, shape: &[u64], expected: &[u64]) -> Result<(), String> {
    if shape != expected {
        return Err(format!(
            "{what}: shape {shape:?} does not match the expected {expected:?}"
        ));
    }
    Ok(())
}

struct Granule {
    sdr: PathBuf,
    geo: PathBuf,
    sdr_sha256: String,
    geo_sha256: String,
    user_block_bytes: (usize, usize),
    spacecraft: String,
    nscan: usize,
    factors: Vec<f32>,
    tb: Vec<f32>,
    lat: Vec<f32>,
    lon: Vec<f32>,
    zenith: Vec<f32>,
    azimuth: Vec<f32>,
    solar_zenith: Vec<f32>,
    time: Vec<f64>,
}

fn spacecraft_of(path: &Path) -> String {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let mut parts = name.split('_');
    let _product = parts.next();
    parts.next().unwrap_or("unknown").to_string()
}

fn decode_granule(sdr_path: &Path, geo_path: &Path) -> Result<Granule, String> {
    let (sdr, sdr_user_block) = open_hdf5(sdr_path)?;
    let (geo, geo_user_block) = open_hdf5(geo_path)?;

    let (tb_shape, counts) = read_counts(&sdr, &format!("{SDR_GROUP}/BrightnessTemperature"))?;
    if tb_shape.len() != 3 || tb_shape[1] as usize != FOVS || tb_shape[2] as usize != CHANNELS {
        return Err(format!(
            "{}: BrightnessTemperature shape {tb_shape:?}, expected [scans, {FOVS}, {CHANNELS}]",
            sdr_path.display()
        ));
    }
    let nscan = tb_shape[0] as usize;
    let (_, factors) = read_f32(&sdr, &format!("{SDR_GROUP}/BrightnessTemperatureFactors"))?;
    if factors.len() < 2 || factors.len() % 2 != 0 || nscan % (factors.len() / 2) != 0 {
        return Err(format!(
            "{}: BrightnessTemperatureFactors has {} values for {nscan} scans",
            sdr_path.display(),
            factors.len()
        ));
    }
    let granules_in_file = factors.len() / 2;
    let scans_per_granule = nscan / granules_in_file;
    let mut tb = Vec::with_capacity(counts.len());
    for (index, count) in counts.iter().enumerate() {
        let scan = index / (FOVS * CHANNELS);
        let block = scan / scans_per_granule;
        let scale = factors[2 * block];
        let offset = factors[2 * block + 1];
        tb.push(match count {
            Some(raw) => *raw as f32 * scale + offset,
            None => f32::NAN,
        });
    }

    let beam_shape = [nscan as u64, FOVS as u64];
    let read_beam = |name: &str| -> Result<Vec<f32>, String> {
        let (shape, values) = read_f32(&geo, &format!("{GEO_GROUP}/{name}"))?;
        expect_shape(&format!("{}: {name}", geo_path.display()), &shape, &beam_shape)?;
        Ok(fill_to_nan(values))
    };
    let lat = read_beam("Latitude")?;
    let lon = read_beam("Longitude")?;
    let zenith = read_beam("SatelliteZenithAngle")?;
    let azimuth = read_beam("SatelliteAzimuthAngle")?;
    let solar_zenith = read_beam("SolarZenithAngle")?;

    // Beam time from the SDR when it carries one per beam, else the GEO
    // scan mid time broadcast across the scan.
    let time: Vec<f64> = match read_i64(&sdr, &format!("{SDR_GROUP}/BeamTime")) {
        Ok((shape, values)) if shape == beam_shape => values.into_iter().map(iet_to_unix).collect(),
        _ => {
            let (shape, values) = read_i64(&geo, &format!("{GEO_GROUP}/MidTime"))?;
            expect_shape(&format!("{}: MidTime", geo_path.display()), &shape, &[nscan as u64])?;
            values
                .into_iter()
                .flat_map(|v| std::iter::repeat_n(iet_to_unix(v), FOVS))
                .collect()
        }
    };

    Ok(Granule {
        sdr: sdr_path.to_path_buf(),
        geo: geo_path.to_path_buf(),
        sdr_sha256: sha256_file(sdr_path)?,
        geo_sha256: sha256_file(geo_path)?,
        user_block_bytes: (sdr_user_block, geo_user_block),
        spacecraft: spacecraft_of(sdr_path),
        nscan,
        factors,
        tb,
        lat,
        lon,
        zenith,
        azimuth,
        solar_zenith,
        time,
    })
}

#[derive(Serialize)]
struct GranuleRecord {
    sdr: String,
    geo: String,
    sdr_sha256: String,
    geo_sha256: String,
    user_block_bytes: (usize, usize),
    spacecraft: String,
    scan_offset: usize,
    scan_count: usize,
    factors: Vec<f32>,
}

#[derive(Serialize)]
struct ArrayRecord {
    filename: String,
    shape: Vec<usize>,
    dtype: &'static str,
    units: &'static str,
}

#[derive(Serialize)]
struct ChannelStats {
    valid: usize,
    min_k: Option<f32>,
    max_k: Option<f32>,
}

#[derive(Serialize)]
struct DecodeMetadata {
    schema: &'static str,
    metadata: Provenance,
    tai_minus_utc_s: f64,
    fill_policy: &'static str,
    scan_count: usize,
    fov_count: usize,
    channel_count: usize,
    granules: Vec<GranuleRecord>,
    arrays: BTreeMap<String, ArrayRecord>,
    channels: Vec<ChannelStats>,
}

fn write_le<T: Copy>(dir: &Path, filename: &str, values: &[T], to_bytes: impl Fn(T) -> Vec<u8>)
    -> Result<(), String> {
    let path = dir.join(filename);
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<T>());
    for value in values {
        bytes.extend_from_slice(&to_bytes(*value));
    }
    fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn write_f32(dir: &Path, filename: &str, values: &[f32]) -> Result<(), String> {
    write_le(dir, filename, values, |v: f32| v.to_le_bytes().to_vec())
}

fn write_f64(dir: &Path, filename: &str, values: &[f64]) -> Result<(), String> {
    write_le(dir, filename, values, |v: f64| v.to_le_bytes().to_vec())
}

fn write_i32(dir: &Path, filename: &str, values: &[i32]) -> Result<(), String> {
    write_le(dir, filename, values, |v: i32| v.to_le_bytes().to_vec())
}

fn decode(outdir: &Path, pairs: &[(PathBuf, PathBuf)]) -> Result<DecodeMetadata, String> {
    fs::create_dir_all(outdir).map_err(|e| format!("{}: {e}", outdir.display()))?;
    let granules: Vec<Granule> = pairs
        .par_iter()
        .map(|(sdr, geo)| decode_granule(sdr, geo))
        .collect::<Result<Vec<_>, String>>()?;

    let total: usize = granules.iter().map(|g| g.nscan).sum();
    let mut tb = Vec::with_capacity(total * FOVS * CHANNELS);
    let mut lat = Vec::with_capacity(total * FOVS);
    let mut lon = Vec::with_capacity(total * FOVS);
    let mut zenith = Vec::with_capacity(total * FOVS);
    let mut azimuth = Vec::with_capacity(total * FOVS);
    let mut solar_zenith = Vec::with_capacity(total * FOVS);
    let mut time = Vec::with_capacity(total * FOVS);
    let mut granule_index = Vec::with_capacity(total);
    let mut records = Vec::with_capacity(granules.len());
    let mut offset = 0usize;
    for (index, granule) in granules.into_iter().enumerate() {
        records.push(GranuleRecord {
            sdr: granule.sdr.display().to_string(),
            geo: granule.geo.display().to_string(),
            sdr_sha256: granule.sdr_sha256,
            geo_sha256: granule.geo_sha256,
            user_block_bytes: granule.user_block_bytes,
            spacecraft: granule.spacecraft,
            scan_offset: offset,
            scan_count: granule.nscan,
            factors: granule.factors,
        });
        granule_index.extend(std::iter::repeat_n(index as i32, granule.nscan));
        offset += granule.nscan;
        tb.extend_from_slice(&granule.tb);
        lat.extend_from_slice(&granule.lat);
        lon.extend_from_slice(&granule.lon);
        zenith.extend_from_slice(&granule.zenith);
        azimuth.extend_from_slice(&granule.azimuth);
        solar_zenith.extend_from_slice(&granule.solar_zenith);
        time.extend_from_slice(&granule.time);
    }

    let mut channels = Vec::with_capacity(CHANNELS);
    for channel in 0..CHANNELS {
        let mut valid = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for beam in 0..(total * FOVS) {
            let value = tb[beam * CHANNELS + channel];
            if value.is_finite() {
                valid += 1;
                min = min.min(value);
                max = max.max(value);
            }
        }
        channels.push(ChannelStats {
            valid,
            min_k: (valid > 0).then_some(min),
            max_k: (valid > 0).then_some(max),
        });
    }

    let mut arrays = BTreeMap::new();
    let mut record = |name: &str, shape: Vec<usize>, dtype: &'static str, units: &'static str| {
        arrays.insert(
            name.to_string(),
            ArrayRecord {
                filename: format!("{name}.{}", dtype.trim_start_matches('<')),
                shape,
                dtype,
                units,
            },
        );
    };
    record("brightness_temperature_k", vec![total, FOVS, CHANNELS], "<f4", "K");
    record("latitude_deg", vec![total, FOVS], "<f4", "degrees_north");
    record("longitude_deg", vec![total, FOVS], "<f4", "degrees_east");
    record("satellite_zenith_deg", vec![total, FOVS], "<f4", "degrees");
    record("satellite_azimuth_deg", vec![total, FOVS], "<f4", "degrees");
    record("solar_zenith_deg", vec![total, FOVS], "<f4", "degrees");
    record("beam_time_unix_s", vec![total, FOVS], "<f8", "s");
    record("granule_index", vec![total], "<i4", "1");

    write_f32(outdir, "brightness_temperature_k.f4", &tb)?;
    write_f32(outdir, "latitude_deg.f4", &lat)?;
    write_f32(outdir, "longitude_deg.f4", &lon)?;
    write_f32(outdir, "satellite_zenith_deg.f4", &zenith)?;
    write_f32(outdir, "satellite_azimuth_deg.f4", &azimuth)?;
    write_f32(outdir, "solar_zenith_deg.f4", &solar_zenith)?;
    write_f64(outdir, "beam_time_unix_s.f8", &time)?;
    write_i32(outdir, "granule_index.i4", &granule_index)?;

    let metadata = DecodeMetadata {
        schema: DECODE_SCHEMA,
        metadata: provenance(),
        tai_minus_utc_s: TAI_MINUS_UTC_S,
        fill_policy: "IDPS fills (u16 >= 65528, floats <= -999, non-positive IET) are NaN",
        scan_count: total,
        fov_count: FOVS,
        channel_count: CHANNELS,
        granules: records,
        arrays,
        channels,
    };
    let text = serde_json::to_string_pretty(&metadata).map_err(|e| e.to_string())?;
    fs::write(outdir.join("metadata.json"), text)
        .map_err(|e| format!("{}: {e}", outdir.join("metadata.json").display()))?;
    Ok(metadata)
}

// --------------------------------------------------------------------- thin

struct ThinOptions {
    latitudes: Vec<f64>,
    nlon: usize,
    origin_unix_s: f64,
    bin_s: f64,
    max_zenith_deg: f64,
}

#[derive(Clone)]
struct Accumulator {
    count: u32,
    tb_sum: [f64; CHANNELS],
    tb_sq: [f64; CHANNELS],
    tb_count: [u32; CHANNELS],
    lat: f64,
    lon_x: f64,
    lon_y: f64,
    zenith: f64,
    azimuth_x: f64,
    azimuth_y: f64,
    solar_zenith: f64,
    time: f64,
    scan_abs: f64,
}

impl Accumulator {
    fn new() -> Self {
        Accumulator {
            count: 0,
            tb_sum: [0.0; CHANNELS],
            tb_sq: [0.0; CHANNELS],
            tb_count: [0; CHANNELS],
            lat: 0.0,
            lon_x: 0.0,
            lon_y: 0.0,
            zenith: 0.0,
            azimuth_x: 0.0,
            azimuth_y: 0.0,
            solar_zenith: 0.0,
            time: 0.0,
            scan_abs: 0.0,
        }
    }
}

fn read_flat_f32(dir: &Path, name: &str, expected: usize) -> Result<Vec<f32>, String> {
    let path = dir.join(format!("{name}.f4"));
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != expected * 4 {
        return Err(format!(
            "{}: {} bytes, expected {} f32 values",
            path.display(),
            bytes.len(),
            expected
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_flat_f64(dir: &Path, name: &str, expected: usize) -> Result<Vec<f64>, String> {
    let path = dir.join(format!("{name}.f8"));
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != expected * 8 {
        return Err(format!(
            "{}: {} bytes, expected {} f64 values",
            path.display(),
            bytes.len(),
            expected
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// Index of the ring whose centre is nearest to `lat` in a monotonic
/// ring set (either direction).
fn nearest_ring(rings: &[f64], ascending: bool, lat: f64) -> usize {
    let n = rings.len();
    let key = if ascending { lat } else { -lat };
    let cmp = |value: f64| if ascending { value } else { -value };
    // Binary search for the first ring >= key in the ascending view.
    let mut lo = 0usize;
    let mut hi = n;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cmp(rings[mid]) < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        0
    } else if lo >= n {
        n - 1
    } else if (cmp(rings[lo]) - key).abs() < (key - cmp(rings[lo - 1])).abs() {
        lo
    } else {
        lo - 1
    }
}

#[derive(Serialize)]
struct ThinMetadata {
    schema: &'static str,
    metadata: Provenance,
    decoded_dir: String,
    decode_metadata_sha256: String,
    ring_count: usize,
    nlon: usize,
    origin_unix_s: f64,
    bin_s: f64,
    max_zenith_deg: f64,
    beams_in: usize,
    beams_placed: usize,
    beams_rejected_geometry: usize,
    beams_rejected_zenith: usize,
    cell_count: usize,
    arrays: BTreeMap<String, ArrayRecord>,
}

fn thin(outdir: &Path, decoded: &Path, options: &ThinOptions) -> Result<ThinMetadata, String> {
    let metadata_path = decoded.join("metadata.json");
    let metadata_text = fs::read_to_string(&metadata_path)
        .map_err(|e| format!("{}: {e}", metadata_path.display()))?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_text)
        .map_err(|e| format!("{}: {e}", metadata_path.display()))?;
    if metadata["schema"].as_str() != Some(DECODE_SCHEMA) {
        return Err(format!(
            "{}: schema {} is not {DECODE_SCHEMA}",
            metadata_path.display(),
            metadata["schema"]
        ));
    }
    let nscan = metadata["scan_count"]
        .as_u64()
        .ok_or_else(|| "metadata.json lacks scan_count".to_string())? as usize;
    let nbeam = nscan * FOVS;
    let tb = read_flat_f32(decoded, "brightness_temperature_k", nbeam * CHANNELS)?;
    let lat = read_flat_f32(decoded, "latitude_deg", nbeam)?;
    let lon = read_flat_f32(decoded, "longitude_deg", nbeam)?;
    let zenith = read_flat_f32(decoded, "satellite_zenith_deg", nbeam)?;
    let azimuth = read_flat_f32(decoded, "satellite_azimuth_deg", nbeam)?;
    let solar_zenith = read_flat_f32(decoded, "solar_zenith_deg", nbeam)?;
    let time = read_flat_f64(decoded, "beam_time_unix_s", nbeam)?;

    let rings = &options.latitudes;
    if rings.len() < 2 {
        return Err("--latitudes needs at least two rings".to_string());
    }
    let ascending = rings[1] > rings[0];
    for pair in rings.windows(2) {
        if (pair[1] > pair[0]) != ascending {
            return Err("--latitudes must be monotonic".to_string());
        }
    }
    let dlon = 360.0 / options.nlon as f64;

    let mut cells: HashMap<(i32, u32, u32), Accumulator> = HashMap::new();
    let mut placed = 0usize;
    let mut rejected_geometry = 0usize;
    let mut rejected_zenith = 0usize;
    for beam in 0..nbeam {
        let (la, lo, ze, az, sz, t) = (
            lat[beam] as f64,
            lon[beam] as f64,
            zenith[beam] as f64,
            azimuth[beam] as f64,
            solar_zenith[beam] as f64,
            time[beam],
        );
        if !(la.is_finite() && lo.is_finite() && ze.is_finite() && t.is_finite() && la.abs() <= 90.0)
        {
            rejected_geometry += 1;
            continue;
        }
        if ze > options.max_zenith_deg {
            rejected_zenith += 1;
            continue;
        }
        let bin = ((t - options.origin_unix_s) / options.bin_s).floor() as i32;
        let j = nearest_ring(rings, ascending, la) as u32;
        let lon_wrapped = lo.rem_euclid(360.0);
        let i = ((lon_wrapped / dlon).round() as usize % options.nlon) as u32;
        let fov = beam % FOVS;
        let scan_angle = (fov as f64 - 47.5) * SCAN_STEP_DEG;
        let acc = cells.entry((bin, j, i)).or_insert_with(Accumulator::new);
        acc.count += 1;
        acc.lat += la;
        acc.lon_x += lo.to_radians().cos();
        acc.lon_y += lo.to_radians().sin();
        acc.zenith += ze;
        if az.is_finite() {
            acc.azimuth_x += az.to_radians().cos();
            acc.azimuth_y += az.to_radians().sin();
        }
        acc.solar_zenith += if sz.is_finite() { sz } else { 0.0 };
        acc.time += t;
        acc.scan_abs += scan_angle.abs();
        for channel in 0..CHANNELS {
            let value = tb[beam * CHANNELS + channel] as f64;
            if value.is_finite() {
                acc.tb_sum[channel] += value;
                acc.tb_sq[channel] += value * value;
                acc.tb_count[channel] += 1;
            }
        }
        placed += 1;
    }

    let mut keys: Vec<(i32, u32, u32)> = cells.keys().copied().collect();
    keys.sort_unstable();
    let ncell = keys.len();
    let mut cell_bin = Vec::with_capacity(ncell);
    let mut cell_j = Vec::with_capacity(ncell);
    let mut cell_i = Vec::with_capacity(ncell);
    let mut count = Vec::with_capacity(ncell);
    let mut tb_mean = Vec::with_capacity(ncell * CHANNELS);
    let mut tb_std = Vec::with_capacity(ncell * CHANNELS);
    let mut tb_count = Vec::with_capacity(ncell * CHANNELS);
    let mut lat_mean = Vec::with_capacity(ncell);
    let mut lon_mean = Vec::with_capacity(ncell);
    let mut zenith_mean = Vec::with_capacity(ncell);
    let mut azimuth_mean = Vec::with_capacity(ncell);
    let mut solar_mean = Vec::with_capacity(ncell);
    let mut scan_abs_mean = Vec::with_capacity(ncell);
    let mut time_mean = Vec::with_capacity(ncell);
    for key in &keys {
        let acc = &cells[key];
        let n = acc.count as f64;
        cell_bin.push(key.0);
        cell_j.push(key.1 as i32);
        cell_i.push(key.2 as i32);
        count.push(acc.count as i32);
        for channel in 0..CHANNELS {
            let m = acc.tb_count[channel] as f64;
            if m > 0.0 {
                let mean = acc.tb_sum[channel] / m;
                let var = (acc.tb_sq[channel] / m - mean * mean).max(0.0);
                tb_mean.push(mean as f32);
                tb_std.push(var.sqrt() as f32);
            } else {
                tb_mean.push(f32::NAN);
                tb_std.push(f32::NAN);
            }
            tb_count.push(acc.tb_count[channel] as i32);
        }
        lat_mean.push((acc.lat / n) as f32);
        lon_mean.push((acc.lon_y.atan2(acc.lon_x).to_degrees().rem_euclid(360.0)) as f32);
        zenith_mean.push((acc.zenith / n) as f32);
        azimuth_mean.push(
            if acc.azimuth_x != 0.0 || acc.azimuth_y != 0.0 {
                acc.azimuth_y.atan2(acc.azimuth_x).to_degrees().rem_euclid(360.0) as f32
            } else {
                f32::NAN
            },
        );
        solar_mean.push((acc.solar_zenith / n) as f32);
        scan_abs_mean.push((acc.scan_abs / n) as f32);
        time_mean.push(acc.time / n);
    }

    fs::create_dir_all(outdir).map_err(|e| format!("{}: {e}", outdir.display()))?;
    write_i32(outdir, "cell_bin.i4", &cell_bin)?;
    write_i32(outdir, "cell_j.i4", &cell_j)?;
    write_i32(outdir, "cell_i.i4", &cell_i)?;
    write_i32(outdir, "count.i4", &count)?;
    write_f32(outdir, "tb_mean_k.f4", &tb_mean)?;
    write_f32(outdir, "tb_std_k.f4", &tb_std)?;
    write_i32(outdir, "tb_count.i4", &tb_count)?;
    write_f32(outdir, "lat_mean_deg.f4", &lat_mean)?;
    write_f32(outdir, "lon_mean_deg.f4", &lon_mean)?;
    write_f32(outdir, "zenith_mean_deg.f4", &zenith_mean)?;
    write_f32(outdir, "azimuth_mean_deg.f4", &azimuth_mean)?;
    write_f32(outdir, "solar_zenith_mean_deg.f4", &solar_mean)?;
    write_f32(outdir, "scan_angle_abs_mean_deg.f4", &scan_abs_mean)?;
    write_f64(outdir, "time_mean_unix_s.f8", &time_mean)?;

    let mut arrays = BTreeMap::new();
    let mut record = |name: &str, shape: Vec<usize>, dtype: &'static str, units: &'static str| {
        arrays.insert(
            name.to_string(),
            ArrayRecord {
                filename: format!("{name}.{}", dtype.trim_start_matches('<')),
                shape,
                dtype,
                units,
            },
        );
    };
    record("cell_bin", vec![ncell], "<i4", "1");
    record("cell_j", vec![ncell], "<i4", "1");
    record("cell_i", vec![ncell], "<i4", "1");
    record("count", vec![ncell], "<i4", "1");
    record("tb_mean_k", vec![ncell, CHANNELS], "<f4", "K");
    record("tb_std_k", vec![ncell, CHANNELS], "<f4", "K");
    record("tb_count", vec![ncell, CHANNELS], "<i4", "1");
    record("lat_mean_deg", vec![ncell], "<f4", "degrees_north");
    record("lon_mean_deg", vec![ncell], "<f4", "degrees_east");
    record("zenith_mean_deg", vec![ncell], "<f4", "degrees");
    record("azimuth_mean_deg", vec![ncell], "<f4", "degrees");
    record("solar_zenith_mean_deg", vec![ncell], "<f4", "degrees");
    record("scan_angle_abs_mean_deg", vec![ncell], "<f4", "degrees");
    record("time_mean_unix_s", vec![ncell], "<f8", "s");

    let result = ThinMetadata {
        schema: THIN_SCHEMA,
        metadata: provenance(),
        decoded_dir: decoded.display().to_string(),
        decode_metadata_sha256: format!("{:x}", Sha256::digest(metadata_text.as_bytes())),
        ring_count: rings.len(),
        nlon: options.nlon,
        origin_unix_s: options.origin_unix_s,
        bin_s: options.bin_s,
        max_zenith_deg: options.max_zenith_deg,
        beams_in: nbeam,
        beams_placed: placed,
        beams_rejected_geometry: rejected_geometry,
        beams_rejected_zenith: rejected_zenith,
        cell_count: ncell,
        arrays,
    };
    let text = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
    fs::write(outdir.join("metadata.json"), text)
        .map_err(|e| format!("{}: {e}", outdir.join("metadata.json").display()))?;
    Ok(result)
}

fn parse_thin_options(args: &[String]) -> Result<ThinOptions, String> {
    let mut latitudes: Option<PathBuf> = None;
    let mut nlon: Option<usize> = None;
    let mut origin: Option<f64> = None;
    let mut bin_s: Option<f64> = None;
    let mut max_zenith = 90.0f64;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--latitudes" => latitudes = Some(PathBuf::from(value)),
            "--nlon" => nlon = Some(value.parse().map_err(|e| format!("--nlon {value}: {e}"))?),
            "--origin-unix-s" => {
                origin = Some(value.parse().map_err(|e| format!("--origin-unix-s {value}: {e}"))?)
            }
            "--bin-s" => bin_s = Some(value.parse().map_err(|e| format!("--bin-s {value}: {e}"))?),
            "--max-zenith" => {
                max_zenith = value.parse().map_err(|e| format!("--max-zenith {value}: {e}"))?
            }
            other => return Err(format!("unknown thin option {other}")),
        }
        index += 2;
    }
    let latitudes_path = latitudes.ok_or("--latitudes is required")?;
    let text = fs::read_to_string(&latitudes_path)
        .map_err(|e| format!("{}: {e}", latitudes_path.display()))?;
    let latitudes = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.parse::<f64>().map_err(|e| format!("{}: {line}: {e}", latitudes_path.display())))
        .collect::<Result<Vec<f64>, String>>()?;
    let nlon = nlon.ok_or("--nlon is required")?;
    if nlon < 2 {
        return Err("--nlon must be at least 2".to_string());
    }
    let bin_s = bin_s.ok_or("--bin-s is required")?;
    if !(bin_s > 0.0) {
        return Err("--bin-s must be positive".to_string());
    }
    Ok(ThinOptions {
        latitudes,
        nlon,
        origin_unix_s: origin.ok_or("--origin-unix-s is required")?,
        bin_s,
        max_zenith_deg: max_zenith,
    })
}

// --------------------------------------------------------------------- main

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("--abi") => {
            println!("{ABI}");
            Ok(())
        }
        Some("--help") | Some("-h") | None => {
            print!("{USAGE}");
            Ok(())
        }
        Some("inventory") => {
            let path = args.get(1).ok_or(USAGE)?;
            let record = inventory(Path::new(path))?;
            let text = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(text.as_bytes()).and_then(|_| handle.write_all(b"\n"))
                .map_err(|e| e.to_string())
        }
        Some("decode") => {
            let outdir = args.get(1).ok_or(USAGE)?;
            // `@FILE` names a list of pairs, one `SDR<TAB>GEO` line each:
            // a day is 2,701 pairs and does not belong on a command line.
            let rest: Vec<String> = match args.get(2) {
                Some(list) if list.starts_with('@') && args.len() == 3 => {
                    let path = &list[1..];
                    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
                    text.lines()
                        .map(str::trim)
                        .filter(|line| !line.is_empty())
                        .flat_map(|line| line.split('\t').map(str::to_string).collect::<Vec<_>>())
                        .collect()
                }
                _ => args[2..].to_vec(),
            };
            if rest.is_empty() || rest.len() % 2 != 0 {
                return Err("decode needs SDR GEO pairs".to_string());
            }
            let pairs: Vec<(PathBuf, PathBuf)> = rest
                .chunks_exact(2)
                .map(|pair| (PathBuf::from(&pair[0]), PathBuf::from(&pair[1])))
                .collect();
            let metadata = decode(Path::new(outdir), &pairs)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": DECODE_SCHEMA,
                    "granules": metadata.granules.len(),
                    "scan_count": metadata.scan_count,
                    "outdir": outdir,
                })
            );
            Ok(())
        }
        Some("thin") => {
            let outdir = args.get(1).ok_or(USAGE)?;
            let decoded = args.get(2).ok_or(USAGE)?;
            let options = parse_thin_options(&args[3..])?;
            let metadata = thin(Path::new(outdir), Path::new(decoded), &options)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": THIN_SCHEMA,
                    "cell_count": metadata.cell_count,
                    "beams_placed": metadata.beams_placed,
                    "beams_rejected_geometry": metadata.beams_rejected_geometry,
                    "beams_rejected_zenith": metadata.beams_rejected_zenith,
                    "outdir": outdir,
                })
            );
            Ok(())
        }
        Some(other) => Err(format!("unknown command {other}\n{USAGE}")),
    }
}

fn main() {
    // Read once so the linker keeps the stamp: a `pub static` nothing
    // reads is dropped from a bin, and the bundle pinner then finds no
    // source revision in rw_atms and has to be told to skip it by name.
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(message) = run(&args) {
        eprintln!("rw_atms: {message}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stamp_carries_the_marker_the_bundle_pinner_reads() {
        assert!(GPUWM_BRIDGE_SOURCE_REV_STAMP.starts_with("GPUWM_BRIDGE_SOURCE_REV="));
    }

    #[test]
    fn the_abi_names_all_three_schemas() {
        for schema in [INVENTORY_SCHEMA, DECODE_SCHEMA, THIN_SCHEMA] {
            assert!(ABI.contains(schema), "ABI does not name {schema}");
        }
    }

    #[test]
    fn nearest_ring_holds_in_both_directions_and_at_the_ends() {
        let ascending = [-60.0, -30.0, 0.0, 30.0, 60.0];
        assert_eq!(nearest_ring(&ascending, true, -90.0), 0);
        assert_eq!(nearest_ring(&ascending, true, 14.0), 2);
        assert_eq!(nearest_ring(&ascending, true, 16.0), 3);
        assert_eq!(nearest_ring(&ascending, true, 90.0), 4);
        let descending = [60.0, 30.0, 0.0, -30.0, -60.0];
        assert_eq!(nearest_ring(&descending, false, 90.0), 0);
        assert_eq!(nearest_ring(&descending, false, 16.0), 1);
        assert_eq!(nearest_ring(&descending, false, -90.0), 4);
    }

    #[test]
    fn fill_values_and_non_finite_values_become_nan() {
        let out = fill_to_nan(vec![250.0, F32_FILL_CEILING, f32::INFINITY]);
        assert_eq!(out[0], 250.0);
        assert!(out[1].is_nan());
        assert!(out[2].is_nan());
    }

    #[test]
    fn a_non_positive_iet_is_not_a_time() {
        assert!(iet_to_unix(0).is_nan());
        assert!(iet_to_unix(-5).is_nan());
        assert!(iet_to_unix(1_000_000).is_finite());
    }
}
