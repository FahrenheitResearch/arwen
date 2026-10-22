//! Source-neutral export of a composed canonical frame to WPS intermediate v5.
//! This is a consumer of the shared frameset, with no model or hex adapter.
use crate::{
    digest,
    refusal::{frame_invalid, usage, Result},
};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

const SURFACE: f32 = 200100.;
const MISSING: f32 = -1.0e30;

#[derive(Deserialize)]
struct Binding {
    field: String,
    wps: String,
    units: String,
    axis: String,
    #[serde(default)]
    transform: String,
    #[serde(default)]
    missing: String,
}
#[derive(Deserialize)]
struct Contract {
    schema: String,
    bindings: Vec<Binding>,
}
#[derive(Deserialize)]
struct Field {
    name: String,
    units: String,
    axes: Vec<String>,
    shape: Vec<usize>,
    dtype: String,
    offset: u64,
    length: u64,
    sha256: String,
}
#[derive(Deserialize)]
struct Frame {
    valid_time: String,
    member: Option<String>,
    source_cycle: String,
    #[serde(deserialize_with = "coordinate")]
    latitude: Vec<f64>,
    #[serde(deserialize_with = "coordinate")]
    longitude: Vec<f64>,
    vertical_kind: String,
    vertical_units: String,
    #[serde(deserialize_with = "coordinate")]
    vertical_values: Vec<f64>,
    header: Value,
    fields: Vec<Field>,
}

// Canonical coordinates bind their JSON values to the original f64 bytes.
// Consume that same descriptor rather than assuming a bare JSON array.
fn coordinate<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<f64>, D::Error> {
    #[derive(Deserialize)]
    struct Coordinate {
        values: Vec<f64>,
        count: usize,
        sha256: String,
    }
    let c = Coordinate::deserialize(d)?;
    let bytes: Vec<u8> = c.values.iter().flat_map(|v| v.to_le_bytes()).collect();
    if c.count != c.values.len() || digest::bytes_sha256(&bytes) != c.sha256 {
        return Err(serde::de::Error::custom("coordinate count/digest differs"));
    }
    Ok(c.values)
}
#[derive(Deserialize)]
struct Stream {
    path: String,
    dtype: String,
    bytes: u64,
    sha256: String,
}
#[derive(Deserialize)]
struct Frameset {
    schema: String,
    stream: Stream,
    frames: Vec<Frame>,
}

fn io_error(e: impl std::fmt::Display) -> crate::refusal::Refusal {
    frame_invalid(e.to_string())
}
fn read_json(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(io_error)
}
fn axis(values: &[f64], name: &str) -> Result<f64> {
    if values.len() < 3 || values.iter().any(|v| !v.is_finite()) {
        return Err(frame_invalid(format!(
            "{name} needs >=3 finite coordinates"
        )));
    }
    let d = values[1] - values[0];
    if d == 0. || values.windows(2).any(|w| (w[1] - w[0] - d).abs() > 1.0e-7) {
        return Err(frame_invalid(format!(
            "{name} must be regular and monotone"
        )));
    }
    Ok(d)
}
fn fixed(out: &mut Vec<u8>, value: &str, width: usize) {
    out.extend(value.as_bytes().iter().copied().take(width));
    out.resize(out.len() + width.saturating_sub(value.len()), b' ');
}
fn record(w: &mut impl Write, bytes: &[u8]) -> Result<()> {
    let n = u32::try_from(bytes.len()).map_err(io_error)?.to_be_bytes();
    w.write_all(&n)
        .and_then(|_| w.write_all(bytes))
        .and_then(|_| w.write_all(&n))
        .map_err(io_error)
}
struct Slab {
    field: String,
    units: String,
    level: f32,
    values: Vec<f32>,
}
fn write_slab(w: &mut impl Write, s: &Slab, f: &Frame, dy: f64, dx: f64) -> Result<()> {
    record(w, &5i32.to_be_bytes())?;
    let mut h = Vec::new();
    fixed(&mut h, &f.valid_time.replace('T', "_"), 24);
    h.extend(0f32.to_be_bytes());
    fixed(&mut h, "rw-wps composed canonical state", 32);
    fixed(&mut h, &s.field, 9);
    fixed(&mut h, &s.units, 25);
    fixed(&mut h, &format!("canonical {}", s.field), 46);
    h.extend(s.level.to_be_bytes());
    h.extend((f.longitude.len() as i32).to_be_bytes());
    h.extend((f.latitude.len() as i32).to_be_bytes());
    h.extend(0i32.to_be_bytes());
    record(w, &h)?;
    let mut p = Vec::new();
    fixed(&mut p, "SWCORNER", 8);
    for v in [f.latitude[0], f.longitude[0], dy, dx, 6371.229] {
        p.extend((v as f32).to_be_bytes());
    }
    record(w, &p)?;
    record(w, &0i32.to_be_bytes())?;
    let mut data = Vec::with_capacity(s.values.len() * 4);
    for v in &s.values {
        data.extend(v.to_be_bytes());
    }
    record(w, &data)
}
fn read_field(stream: &Path, f: &Field) -> Result<Vec<f64>> {
    let count = f
        .shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| frame_invalid("field shape overflow"))?;
    if f.dtype != "<f8" || count.checked_mul(8).map(|v| v as u64) != Some(f.length) {
        return Err(frame_invalid(format!(
            "{} has invalid dtype/length",
            f.name
        )));
    }
    let mut file = File::open(stream).map_err(io_error)?;
    file.seek(SeekFrom::Start(f.offset)).map_err(io_error)?;
    let mut bytes = vec![0; f.length as usize];
    file.read_exact(&mut bytes).map_err(io_error)?;
    let values: Vec<f64> = bytes
        .as_chunks::<8>().0.iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    if digest::array_sha256(&f.shape, &values) != f.sha256 {
        return Err(frame_invalid(format!("{} digest differs", f.name)));
    }
    Ok(values)
}

/// `wps --frameset DIR --composition JSON --contract JSON --valid-time ISO --out MET`.
/// Global coverage is mandatory: the target reader wraps longitude periodically.
pub fn run(arguments: &[String]) -> Result<Value> {
    if !arguments.len().is_multiple_of(2) {
        return Err(usage("wps takes --frameset DIR --composition JSON --contract JSON --valid-time ISO --out MET"));
    }
    let mut args = BTreeMap::new();
    for pair in arguments.as_chunks::<2>().0 {
        if ![
            "--frameset",
            "--composition",
            "--contract",
            "--valid-time",
            "--out",
            "--member",
        ]
        .contains(&pair[0].as_str())
            || args.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err(usage(format!("unknown/duplicate wps argument {}", pair[0])));
        }
    }
    let need = |key: &str| {
        args.get(key)
            .copied()
            .ok_or_else(|| usage(format!("wps requires {key}")))
    };
    let dir = Path::new(need("--frameset")?);
    let out = PathBuf::from(need("--out")?);
    let manifest = read_json(&dir.join("frames.json"))?;
    let set: Frameset = serde_json::from_slice(&manifest).map_err(io_error)?;
    if set.schema != crate::FRAMESET_SCHEMA
        || set.stream.dtype != "<f8"
        || set.stream.path != "frames.f64"
    {
        return Err(frame_invalid(
            "wps requires a canonical non-windowed frameset",
        ));
    }
    let stream = dir.join(&set.stream.path);
    if fs::metadata(&stream).map_err(io_error)?.len() != set.stream.bytes
        || digest::file_sha256(
            stream
                .to_str()
                .ok_or_else(|| frame_invalid("non-UTF8 frameset path"))?,
        )
        .map_err(io_error)?
            != set.stream.sha256
    {
        return Err(frame_invalid("frameset stream size/digest differs"));
    }
    let selected: Vec<_> = set
        .frames
        .iter()
        .filter(|f| {
            f.valid_time == need("--valid-time").unwrap_or_default()
                && f.member.as_deref() == args.get("--member").copied()
        })
        .collect();
    if selected.len() != 1 {
        return Err(frame_invalid(format!(
            "wps requires one matching valid-time/member, found {}",
            selected.len()
        )));
    }
    let f = selected[0];
    chrono::NaiveDateTime::parse_from_str(&f.valid_time, "%Y-%m-%dT%H:%M:%S").map_err(io_error)?;
    let dy = axis(&f.latitude, "latitude")?;
    let dx = axis(&f.longitude, "longitude")?;
    if dx <= 0.
        || (dx * f.longitude.len() as f64 - 360.).abs() > 1.0e-6
        || f.latitude.iter().copied().fold(f64::INFINITY, f64::min) > -90. + dy.abs()
        || f.latitude.iter().copied().fold(f64::NEG_INFINITY, f64::max) < 90. - dy.abs()
    {
        return Err(frame_invalid("wps global init requires uncropped longitude (one 360-degree period, no duplicate seam) and pole-to-pole latitude"));
    }
    if f.header["grid"]["projection"] != "regular_latitude_longitude"
        || f.header["grid"]["wind_basis"] != "earth_relative"
        || f.vertical_kind != "pressure"
        || !["hPa", "Pa"].contains(&f.vertical_units.as_str())
    {
        return Err(frame_invalid(
            "wps requires regular lat-lon, earth-relative winds and pressure levels",
        ));
    }
    let levels: Vec<f32> = f
        .vertical_values
        .iter()
        .map(|v| (*v * if f.vertical_units == "hPa" { 100. } else { 1. }) as f32)
        .collect();
    if levels.is_empty()
        || levels
            .iter()
            .any(|v| !v.is_finite() || *v <= 0. || *v >= SURFACE)
        || levels.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(frame_invalid(
            "pressure levels must be unique, positive and increasing",
        ));
    }
    let contract_bytes = read_json(Path::new(need("--contract")?))?;
    let contract: Contract = serde_json::from_slice(&contract_bytes).map_err(io_error)?;
    if contract.schema != "rw-wps.intermediate-fields.v1" {
        return Err(frame_invalid("unsupported WPS field contract"));
    }
    let composition_bytes = read_json(Path::new(need("--composition")?))?;
    let composition: Value = serde_json::from_slice(&composition_bytes).map_err(io_error)?;
    let layers = composition["soil_layers"]["source_layers"]
        .as_array()
        .ok_or_else(|| frame_invalid("composition must declare soil source_layers"))?;
    if composition["soil_layers"]["depth_units"] != "m" {
        return Err(frame_invalid("soil depth units must be m"));
    }
    let soil: Vec<(usize, usize)> = layers
        .iter()
        .map(|l| {
            let depth = |key: &str| -> Result<usize> {
                let v = l[key]
                    .as_f64()
                    .ok_or_else(|| frame_invalid("invalid soil bound"))?
                    * 100.;
                if !v.is_finite() || !(0. ..=999.).contains(&v) || (v - v.round()).abs() > 1.0e-6 {
                    return Err(frame_invalid(
                        "WPS soil bounds require whole centimetres in 0..999",
                    ));
                }
                Ok(v.round() as usize)
            };
            let top = depth("top")?;
            let bottom = depth("bottom")?;
            if bottom <= top {
                return Err(frame_invalid("soil bounds must increase"));
            }
            Ok((top, bottom))
        })
        .collect::<Result<_>>()?;
    if soil.is_empty() || soil[0].0 != 0 || soil.windows(2).any(|w| w[0].1 != w[1].0) {
        return Err(frame_invalid(
            "soil layers must be contiguous from the surface",
        ));
    }
    let nxy = f.latitude.len() * f.longitude.len();
    // Field readers, hashes and float conversion run across the shared Rayon pool.
    let slabs: Vec<Vec<Slab>> = contract
        .bindings
        .par_iter()
        .map(|b| -> Result<Vec<Slab>> {
            let matches: Vec<_> = f.fields.iter().filter(|v| v.name == b.field).collect();
            if matches.len() != 1 {
                return Err(frame_invalid(format!(
                    "WPS binding {} requires exactly one canonical field",
                    b.field
                )));
            }
            let field = matches[0];
            let values = read_field(&stream, field)?;
            let (axes, shape) = match b.axis.as_str() {
                "pressure" => (
                    vec!["vertical", "y", "x"],
                    vec![levels.len(), f.latitude.len(), f.longitude.len()],
                ),
                "surface" => (vec!["y", "x"], vec![f.latitude.len(), f.longitude.len()]),
                "soil" => (
                    vec!["soil", "y", "x"],
                    vec![soil.len(), f.latitude.len(), f.longitude.len()],
                ),
                _ => return Err(frame_invalid("unknown WPS axis")),
            };
            if field.units != b.units || field.axes != axes || field.shape != shape {
                return Err(frame_invalid(format!(
                    "{} units/axes/shape differ from WPS contract",
                    b.field
                )));
            }
            if !["", "binary_land_fraction"].contains(&b.transform.as_str()) {
                return Err(frame_invalid("unknown WPS transform"));
            }
            values
                .chunks(nxy)
                .enumerate()
                .map(|(k, values)| {
                    let name = if b.axis == "soil" {
                        format!("{}{:03}{:03}", b.wps, soil[k].0, soil[k].1)
                    } else {
                        b.wps.clone()
                    };
                    let values = values
                        .iter()
                        .map(|v| -> Result<f32> {
                            if v.is_nan() {
                                if b.missing != "preserve_mask" {
                                    return Err(frame_invalid(format!(
                                        "{} has missing values forbidden by the WPS contract",
                                        b.field
                                    )));
                                }
                                return Ok(MISSING);
                            }
                            if !v.is_finite() || !(*v as f32).is_finite() {
                                return Err(frame_invalid(format!(
                                    "{} has unrepresentable values",
                                    b.field
                                )));
                            }
                            if b.transform == "binary_land_fraction" {
                                if !(0. ..=1.).contains(v) {
                                    return Err(frame_invalid("land_fraction outside [0,1]"));
                                }
                                Ok(if *v >= 0.5 { 1. } else { 0. })
                            } else {
                                Ok(*v as f32)
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Slab {
                        field: name,
                        units: b.units.clone(),
                        level: if b.axis == "pressure" {
                            levels[k]
                        } else {
                            SURFACE
                        },
                        values,
                    })
                })
                .collect()
        })
        .collect::<Result<_>>()?;
    if out.exists() {
        return Err(frame_invalid(format!(
            "refusing to overwrite {}",
            out.display()
        )));
    }
    let mut file = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(&out)
            .map_err(io_error)?,
    );
    let result = (|| {
        for slab in slabs.iter().flatten() {
            write_slab(&mut file, slab, f, dy, dx)?;
        }
        file.flush().map_err(io_error)
    })();
    if result.is_err() {
        drop(file);
        let _ = fs::remove_file(&out);
        result?;
    }
    let receipt = json!({"schema":"rw-wps.intermediate-export.v1","format_version":5,"output":{"path":out,"bytes":fs::metadata(&out).map_err(io_error)?.len(),"sha256":digest::file_sha256(out.to_str().unwrap()).map_err(io_error)?},"frameset_sha256":digest::bytes_sha256(&manifest),"stream_sha256":set.stream.sha256,"composition_sha256":digest::bytes_sha256(&composition_bytes),"field_contract_sha256":digest::bytes_sha256(&contract_bytes),"valid_time":f.valid_time,"source_cycle":f.source_cycle,"member":f.member,"global":true,"nx":f.longitude.len(),"ny":f.latitude.len(),"nfglevels":levels.len()+1,"nfgsoillevels":soil.len(),"soil_layers_cm":soil,"fields":slabs.iter().flatten().map(|s|json!({"field":s.field,"level":s.level,"missing":s.values.iter().filter(|v|**v==MISSING).count()})).collect::<Vec<_>>()});
    Ok(receipt)
}
