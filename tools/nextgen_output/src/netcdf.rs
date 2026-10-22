//! Typed NetCDF snapshots on the project's Rust CDF-5 / NetCDF-4 writer and reader.
use crate::Result;
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    ffi::CString,
    fs,
    os::raw::c_int,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};
static SERIAL: AtomicU64 = AtomicU64::new(0);
fn name(value: &str) -> Result<CString> {
    Ok(CString::new(value)?)
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Dimension {
    pub name: String,
    pub len: usize,
    #[serde(default)]
    pub unlimited: bool,
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Attribute {
    pub name: String,
    pub dtype: String,
    #[serde(default)]
    pub text: String,
    /// Native little-endian bytes, including the original float bit patterns.
    #[serde(default)]
    pub bytes: Vec<u8>,
}
impl Attribute {
    pub fn text(name: &str, value: &str) -> Self {
        Self {
            name: name.into(),
            dtype: "char".into(),
            text: value.into(),
            bytes: vec![],
        }
    }
    pub fn f64(name: &str, value: f64) -> Self {
        Self {
            name: name.into(),
            dtype: "f64le".into(),
            text: String::new(),
            bytes: value.to_le_bytes().to_vec(),
        }
    }
    pub fn f32(name: &str, value: f32) -> Self {
        Self {
            name: name.into(),
            dtype: "f32le".into(),
            text: String::new(),
            bytes: value.to_le_bytes().to_vec(),
        }
    }
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Variable {
    pub name: String,
    pub dtype: String,
    pub dimensions: Vec<String>,
    #[serde(default)]
    pub attributes: Vec<Attribute>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub strings: Vec<String>,
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub format: String,
    pub dimensions: Vec<Dimension>,
    pub variables: Vec<Variable>,
    #[serde(default)]
    pub attributes: Vec<Attribute>,
}

pub fn datatype(dtype: &str) -> Result<(c_int, usize)> {
    Ok(match dtype {
        "i8" => (1, 1),
        "char" => (2, 1),
        "i16le" => (3, 2),
        "i32le" => (4, 4),
        "f32le" => (5, 4),
        "f64le" => (6, 8),
        "u8" => (7, 1),
        "u16le" => (8, 2),
        "u32le" => (9, 4),
        "i64le" => (10, 8),
        "u64le" => (11, 8),
        "string" => (12, 0),
        _ => return Err(format!("unsupported NetCDF type {dtype}").into()),
    })
}
impl Dataset {
    pub fn elements(&self, var: &Variable) -> Result<usize> {
        var.dimensions.iter().try_fold(1usize, |n, key| {
            let d = self
                .dimensions
                .iter()
                .find(|d| &d.name == key)
                .ok_or("undefined dimension")?;
            n.checked_mul(d.len)
                .ok_or_else(|| "dimension size overflow".into())
        })
    }
    pub fn validate(&self, payload: &[u8]) -> Result<()> {
        if cfg!(target_endian = "big") {
            return Err("little-endian snapshot transport required".into());
        }
        if !["netcdf4", "cdf5"].contains(&self.format.as_str()) {
            return Err("unsupported container".into());
        }
        let mut dims = std::collections::HashSet::new();
        for dim in &self.dimensions {
            if dim.len == 0 || !dims.insert(&dim.name) {
                return Err("empty or duplicate dimension".into());
            }
            name(&dim.name)?;
        }
        let mut vars = std::collections::HashSet::new();
        let mut ranges = Vec::new();
        for var in &self.variables {
            if !vars.insert(&var.name) {
                return Err("duplicate variable".into());
            }
            name(&var.name)?;
            let (_, width) = datatype(&var.dtype)?;
            let count = self.elements(var)?;
            if width == 0 {
                if var.strings.len() != count {
                    return Err("string variable count differs".into());
                }
                for s in &var.strings {
                    name(s)?;
                }
            } else {
                let size = count.checked_mul(width).ok_or("array size overflow")?;
                let end = var
                    .offset
                    .checked_add(size)
                    .ok_or("array offset overflow")?;
                if end > payload.len() {
                    return Err(format!("incomplete variable {}", var.name).into());
                }
                ranges.push((var.offset, end));
            }
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (a, b) in ranges {
            if a != end {
                return Err("snapshot has gaps or overlapping fields".into());
            }
            end = b;
        }
        if end != payload.len() {
            return Err("snapshot has unclaimed bytes".into());
        }
        Ok(())
    }
}

fn nc_type(dtype: &str) -> Result<NcType> {
    Ok(match dtype {
        "i8" => NcType::Byte,
        "char" => NcType::Char,
        "i16le" => NcType::Short,
        "i32le" => NcType::Int,
        "f32le" => NcType::Float,
        "f64le" => NcType::Double,
        "u8" => NcType::UByte,
        "u16le" => NcType::UShort,
        "u32le" => NcType::UInt,
        "i64le" => NcType::Int64,
        "u64le" => NcType::UInt64,
        _ => return Err(format!("{dtype} has no classic NetCDF representation").into()),
    })
}
fn attribute(a: &Attribute) -> Result<AttrValue> {
    let (_, width) = datatype(&a.dtype)?;
    if a.dtype == "char" {
        return Ok(AttrValue::Text(a.text.clone()));
    }
    if width == 0 || !a.bytes.len().is_multiple_of(width) {
        return Err("invalid attribute byte count".into());
    }
    Ok(match a.dtype.as_str() {
        "i8" => AttrValue::Bytes(
            a.bytes
                .chunks_exact(1)
                .map(|b| i8::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "i16le" => AttrValue::Shorts(
            a.bytes
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "i32le" => AttrValue::Ints(
            a.bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "f32le" => AttrValue::Floats(
            a.bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "f64le" => AttrValue::Doubles(
            a.bytes
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "u8" => AttrValue::UBytes(
            a.bytes
                .chunks_exact(1)
                .map(|b| u8::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "u16le" => AttrValue::UShorts(
            a.bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "u32le" => AttrValue::UInts(
            a.bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "i64le" => AttrValue::Int64s(
            a.bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        "u64le" => AttrValue::UInt64s(
            a.bytes
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        _ => return Err("unsupported attribute type".into()),
    })
}
fn write_values(
    writer: &mut NcWriter,
    id: usize,
    dtype: &str,
    bytes: &[u8],
    record: Option<(usize, usize)>,
) -> Result<()> {
    macro_rules! push {
        ($ty:ty, $variant:ident) => {{
            let values: Vec<$ty> = bytes
                .par_chunks_exact(std::mem::size_of::<$ty>())
                .map(|b| <$ty>::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if let Some((count, slab)) = record {
                for r in 0..count {
                    writer.write_record(
                        r as u64,
                        id,
                        VarData::$variant(&values[r * slab..(r + 1) * slab]),
                    )?;
                }
            } else {
                writer.write_var(id, VarData::$variant(&values))?;
            }
        }};
    }
    match dtype {
        "i8" => push!(i8, I8),
        "char" => push!(u8, Char),
        "i16le" => push!(i16, I16),
        "i32le" => push!(i32, I32),
        "f32le" => push!(f32, F32),
        "f64le" => push!(f64, F64),
        "u8" => push!(u8, U8),
        "u16le" => push!(u16, U16),
        "u32le" => push!(u32, U32),
        "i64le" => push!(i64, I64),
        "u64le" => push!(u64, U64),
        _ => return Err("unsupported variable type".into()),
    }
    Ok(())
}

fn write_classic(path: &Path, schema: &Dataset, payload: &[u8]) -> Result<()> {
    let mut output = Schema::new(NcFormat::Cdf5);
    for d in &schema.dimensions {
        output.def_dim(&d.name, d.len, d.unlimited)?;
    }
    for a in &schema.attributes {
        output.put_global_attr(&a.name, attribute(a)?)?;
    }
    for v in &schema.variables {
        let dims = v
            .dimensions
            .iter()
            .map(|key| {
                schema
                    .dimensions
                    .iter()
                    .position(|d| &d.name == key)
                    .ok_or("undefined dimension")
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let id = output.def_var(&v.name, nc_type(&v.dtype)?, &dims)?;
        for a in &v.attributes {
            output.put_var_attr(id, &a.name, attribute(a)?)?;
        }
    }
    let mut writer = NcWriter::create(path, output)?;
    for (id, v) in schema.variables.iter().enumerate() {
        let count = schema.elements(v)?;
        let size = count * datatype(&v.dtype)?.1;
        let record = v
            .dimensions
            .first()
            .and_then(|key| {
                schema
                    .dimensions
                    .iter()
                    .find(|d| &d.name == key && d.unlimited)
            })
            .map(|d| (d.len, count / d.len));
        write_values(
            &mut writer,
            id,
            &v.dtype,
            &payload[v.offset..v.offset + size],
            record,
        )?;
    }
    writer.finish()?;
    Ok(())
}
fn write_netcdf4(path: &Path, schema: &Dataset, payload: &[u8]) -> Result<()> {
    use netcdf_writer::netcdf4 as nc4;
    if schema.dimensions.iter().any(|d| d.unlimited) {
        return Err("NetCDF-4 snapshot writer cannot preserve unlimited dimensions; use CDF5 for record output".into());
    }
    fn attrs(items: &[Attribute]) -> Result<Vec<nc4::Attribute<'_>>> {
        items
            .iter()
            .map(|a| {
                Ok(nc4::Attribute {
                    name: &a.name,
                    dtype: nc_type(&a.dtype)?,
                    bytes: if a.dtype == "char" {
                        a.text.as_bytes()
                    } else {
                        &a.bytes
                    },
                })
            })
            .collect()
    }
    let dimensions = schema
        .dimensions
        .iter()
        .map(|d| nc4::Dimension {
            name: &d.name,
            len: d.len,
        })
        .collect::<Vec<_>>();
    let variables = schema
        .variables
        .iter()
        .map(|v| {
            let values = if v.dtype == "string" {
                nc4::Values::Strings(&v.strings)
            } else {
                let size = schema.elements(v)? * datatype(&v.dtype)?.1;
                nc4::Values::Primitive(nc_type(&v.dtype)?, &payload[v.offset..v.offset + size])
            };
            Ok(nc4::Variable {
                name: &v.name,
                dimensions: v
                    .dimensions
                    .iter()
                    .map(|key| {
                        schema
                            .dimensions
                            .iter()
                            .position(|d| &d.name == key)
                            .ok_or("undefined dimension")
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                attributes: attrs(&v.attributes)?,
                values,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    nc4::write(path, &dimensions, &variables, &attrs(&schema.attributes)?)?;
    Ok(())
}

pub fn write(path: &Path, schema: &Dataset, payload: &[u8]) -> Result<serde_json::Value> {
    let started = Instant::now();
    schema.validate(payload)?;
    let validation_ms = started.elapsed().as_secs_f64() * 1000.;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let partial = path.with_file_name(format!(
        ".{}.partial-{}-{serial}",
        path.file_name()
            .ok_or("missing filename")?
            .to_string_lossy(),
        std::process::id()
    ));
    // Reserve an owned temporary name. Publication never replaces an existing file.
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)?;
    let result = (|| -> Result<f64> {
        let data_started = Instant::now();
        if schema.format == "netcdf4" {
            write_netcdf4(&partial, schema, payload)?;
        } else {
            write_classic(&partial, schema, payload)?;
        }
        let payload_write_ms = data_started.elapsed().as_secs_f64() * 1000.;
        fs::OpenOptions::new()
            .write(true)
            .open(&partial)?
            .sync_all()?;
        fs::hard_link(&partial, path)?;
        crate::sync_directory(parent)?;
        Ok(payload_write_ms)
    })();
    let cleanup = fs::remove_file(&partial);
    let payload_write_ms = result?;
    cleanup?;
    Ok(
        serde_json::json!({"operation":"netcdf-write", "variables":schema.variables.len(), "payload_bytes":payload.len(),
        "container_and_fsync_ms":started.elapsed().as_secs_f64()*1000., "validation_ms":validation_ms,
        "payload_write_ms":payload_write_ms, "container_format":schema.format, "container_backend":"netcdf-writer (Rust)",
        "parallelism":format!("{} Rayon workers for typed payload conversion; ordered file I/O",rayon::current_num_threads())}),
    )
}

fn attributes(items: &[netcdf_reader::NcAttribute]) -> Result<Vec<Attribute>> {
    use netcdf_reader::NcAttrValue as V;
    items
        .iter()
        .map(|a| {
            let mut out = Attribute {
                name: a.name.clone(),
                dtype: String::new(),
                text: String::new(),
                bytes: vec![],
            };
            macro_rules! numeric {
                ($v:expr, $dtype:expr) => {{
                    out.dtype = $dtype.into();
                    for x in $v {
                        out.bytes.extend_from_slice(&x.to_le_bytes());
                    }
                }};
            }
            match &a.value {
                V::Chars(s) => {
                    out.dtype = "char".into();
                    out.text = s.clone();
                }
                V::Bytes(v) => numeric!(v, "i8"),
                V::Shorts(v) => numeric!(v, "i16le"),
                V::Ints(v) => numeric!(v, "i32le"),
                V::Floats(v) => numeric!(v, "f32le"),
                V::Doubles(v) => numeric!(v, "f64le"),
                V::UBytes(v) => numeric!(v, "u8"),
                V::UShorts(v) => numeric!(v, "u16le"),
                V::UInts(v) => numeric!(v, "u32le"),
                V::Int64s(v) => numeric!(v, "i64le"),
                V::UInt64s(v) => numeric!(v, "u64le"),
                V::Strings(_) => return Err("string attributes not captured".into()),
            }
            Ok(out)
        })
        .collect()
}

/// Read in each variable's native type: no float or integer widening.
pub fn read(path: &Path) -> Result<(Dataset, Vec<u8>)> {
    let file = netcdf_reader::NcFile::open(path)?;
    let mut schema = Dataset {
        format: match file.format() {
            netcdf_reader::NcFormat::Cdf5 => "cdf5",
            netcdf_reader::NcFormat::Nc4 | netcdf_reader::NcFormat::Nc4Classic => "netcdf4",
            _ => return Err("expected NetCDF4 or CDF5".into()),
        }
        .into(),
        dimensions: file
            .dimensions()?
            .iter()
            .map(|d| {
                Ok(Dimension {
                    name: d.name.clone(),
                    len: d.size.try_into()?,
                    unlimited: d.is_unlimited,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        variables: vec![],
        attributes: attributes(file.global_attributes()?)?,
    };
    let mut payload = Vec::new();
    for v in file.variables()? {
        let dtype = match v.dtype.classic_type_code() {
            Some(code) => [
                "", "i8", "char", "i16le", "i32le", "f32le", "f64le", "u8", "u16le", "u32le",
                "i64le", "u64le",
            ][code as usize],
            None if file.read_variable_char_bytes(&v.name).is_ok() => "char",
            None => "string",
        };
        let mut var = Variable {
            name: v.name.clone(),
            dtype: dtype.into(),
            dimensions: v.dimensions.iter().map(|d| d.name.clone()).collect(),
            attributes: attributes(&v.attributes)?,
            offset: payload.len(),
            strings: vec![],
        };
        macro_rules! values {
            ($ty:ty) => {
                for x in file.read_variable::<$ty>(&v.name)?.iter() {
                    payload.extend_from_slice(&x.to_le_bytes());
                }
            };
        }
        match dtype {
            "i8" => values!(i8),
            "i16le" => values!(i16),
            "i32le" => values!(i32),
            "f32le" => values!(f32),
            "f64le" => values!(f64),
            "u8" => values!(u8),
            "u16le" => values!(u16),
            "u32le" => values!(u32),
            "i64le" => values!(i64),
            "u64le" => values!(u64),
            "char" => payload.extend_from_slice(&file.read_variable_char_bytes(&v.name)?),
            "string" => var.strings = file.read_variable_as_strings(&v.name)?,
            _ => unreachable!(),
        }
        schema.variables.push(var);
    }
    schema.validate(&payload)?;
    Ok((schema, payload))
}
