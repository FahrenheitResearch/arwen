//! `nc_rewrite IN.nc OUT.nc [--format cdf2|cdf5|auto]`
//!
//! Read a NetCDF file with Drew's Rust readers and write it back out
//! through `netcdf-writer`. This is the PARITY instrument for the writer:
//! a real wrfout goes in, a classic-format file comes out, and the two
//! are compared variable by variable, attribute by attribute, and
//! render by render (`tools/nc_rewrite_parity.py`,
//! `tools/nc_rewrite_render_parity.py`).
//!
//! Read side, and why it is two crates:
//!
//! * **structure** comes from `netcrust`, the sanctioned read facade --
//!   its NetCDF-4 metadata reconstruction (dimension overrides, the raw
//!   HDF5 index fallback) is the hardened part;
//! * **values** come from `netcdf-reader` directly, because netcrust's
//!   public read surface is `f64` only (gap D2 in the reset map). A
//!   parity harness may not promote an `f32` field through `f64`: the
//!   value survives, but NaN payloads and the exact bit pattern need not,
//!   and byte-comparing renders is the whole point.
//!
//! Memory: variables are read whole. A wrfout history frame is one record
//! and a few tens of megabytes per variable, so this is bounded by the
//! largest single variable, not by the file.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

const USAGE: &str = "usage: nc_rewrite IN.nc OUT.nc [--format cdf2|cdf5|auto]";

fn main() -> ExitCode {
    match run() {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("nc_rewrite: {message}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FormatChoice {
    Cdf2,
    Cdf5,
    Auto,
}

fn run() -> Result<String, String> {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().ok_or(USAGE)?);
    let output = PathBuf::from(args.next().ok_or(USAGE)?);
    let mut choice = FormatChoice::Auto;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--format" => {
                choice = match args.next().as_deref() {
                    Some("cdf2") => FormatChoice::Cdf2,
                    Some("cdf5") => FormatChoice::Cdf5,
                    Some("auto") => FormatChoice::Auto,
                    other => {
                        return Err(format!("unknown --format {other:?}; {USAGE}"));
                    }
                }
            }
            other => return Err(format!("unknown argument {other:?}; {USAGE}")),
        }
    }

    let structure = netcrust::open(&input).map_err(|err| format!("open {input:?}: {err}"))?;
    let values = netcdf_reader::NcFile::open(&input)
        .map_err(|err| format!("open {input:?} for typed reads: {err}"))?;

    let dims = structure
        .dimensions()
        .map_err(|err| format!("dimensions: {err}"))?;
    let vars = structure
        .variables()
        .map_err(|err| format!("variables: {err}"))?;
    let gattrs = structure
        .attributes()
        .map_err(|err| format!("global attributes: {err}"))?;

    // How many records the source carries: netcrust reports the record
    // dimension's length as the record count.
    let numrecs = dims
        .iter()
        .find(|dim| dim.is_unlimited())
        .map(|dim| dim.len() as u64)
        .unwrap_or(0);

    let needs_cdf5 = vars
        .iter()
        .any(|var| map_dtype(var.dtype()).is_some_and(is_cdf5_only));
    let formats: Vec<NcFormat> = match choice {
        FormatChoice::Cdf2 => vec![NcFormat::Offset64],
        FormatChoice::Cdf5 => vec![NcFormat::Cdf5],
        FormatChoice::Auto if needs_cdf5 => vec![NcFormat::Cdf5],
        // CDF-2 first, CDF-5 only if the sizes demand it. A file the
        // whole world can read beats a file only new readers can.
        FormatChoice::Auto => vec![NcFormat::Offset64, NcFormat::Cdf5],
    };

    let mut last_error = String::new();
    for (attempt, &format) in formats.iter().enumerate() {
        let mut schema = Schema::new(format);
        let mut dimid_of: HashMap<String, usize> = HashMap::new();
        for dim in &dims {
            let dimid = schema
                .def_dim(dim.name(), dim.len(), dim.is_unlimited())
                .map_err(|err| format!("dimension '{}': {err}", dim.name()))?;
            dimid_of.insert(dim.name().to_string(), dimid);
        }
        for attr in &gattrs {
            let value = map_attr(attr.value())
                .map_err(|err| format!("global attribute '{}': {err}", attr.name()))?;
            schema
                .put_global_attr(attr.name(), value)
                .map_err(|err| format!("global attribute '{}': {err}", attr.name()))?;
        }
        let mut varids = Vec::with_capacity(vars.len());
        for var in &vars {
            let ty = map_dtype(var.dtype())
                .ok_or_else(|| format!("variable '{}': {:?} has no classic external type; \
                                        classic NetCDF cannot represent it", var.name(), var.dtype()))?;
            let ids: Vec<usize> = var
                .dimensions()
                .iter()
                .map(|dim| {
                    dimid_of
                        .get(dim.name())
                        .copied()
                        .ok_or_else(|| format!("variable '{}' uses unknown dimension '{}'",
                                               var.name(), dim.name()))
                })
                .collect::<Result<_, _>>()?;
            let varid = schema
                .def_var(var.name(), ty, &ids)
                .map_err(|err| format!("variable '{}': {err}", var.name()))?;
            for attr in var.attributes() {
                let value = map_attr(attr.value()).map_err(|err| {
                    format!("attribute '{}' of '{}': {err}", attr.name(), var.name())
                })?;
                schema.put_var_attr(varid, attr.name(), value).map_err(|err| {
                    format!("attribute '{}' of '{}': {err}", attr.name(), var.name())
                })?;
            }
            varids.push((varid, var));
        }

        let mut writer = match NcWriter::create(&output, schema) {
            Ok(writer) => writer,
            Err(err) if attempt + 1 < formats.len() => {
                last_error = err.to_string();
                continue;
            }
            Err(err) => return Err(format!("create {output:?}: {err}")),
        };

        for (varid, var) in &varids {
            copy_variable(&values, &mut writer, *varid, var, numrecs)?;
        }
        writer
            .finish()
            .map_err(|err| format!("finish {output:?}: {err}"))?;

        let label = match format {
            NcFormat::Classic => "CDF-1",
            NcFormat::Offset64 => "CDF-2",
            NcFormat::Cdf5 => "CDF-5",
        };
        let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
        let retried = if last_error.is_empty() {
            String::new()
        } else {
            format!(" (retried after: {last_error})")
        };
        return Ok(format!(
            "REWROTE {} -> {} format={label} dims={} vars={} gattrs={} records={numrecs} \
             bytes={size}{retried}",
            input.display(),
            output.display(),
            dims.len(),
            vars.len(),
            gattrs.len(),
        ));
    }
    Err(format!("no classic container fit this file: {last_error}"))
}

/// A copy of `NcType`'s CDF-5-only predicate, kept here because the crate
/// does not export it (it is an internal container rule, not an API).
fn is_cdf5_only(ty: NcType) -> bool {
    matches!(
        ty,
        NcType::UByte | NcType::UShort | NcType::UInt | NcType::Int64 | NcType::UInt64
    )
}

/// NetCDF-4 stores `NC_CHAR` as an HDF5 fixed-length string of size 1, and
/// the reader reports that as `DataType::String` -- the size is gone by
/// the time it reaches us. So a dimensioned `String` variable is treated
/// as `NC_CHAR`, which is what a wrfout's `Times` is, and
/// [`char_bytes`] refuses at read time if the shape says otherwise.
fn map_dtype(dtype: &netcrust::DataType) -> Option<NcType> {
    Some(match dtype {
        netcrust::DataType::I8 => NcType::Byte,
        netcrust::DataType::Char | netcrust::DataType::String => NcType::Char,
        netcrust::DataType::I16 => NcType::Short,
        netcrust::DataType::I32 => NcType::Int,
        netcrust::DataType::F32 => NcType::Float,
        netcrust::DataType::F64 => NcType::Double,
        netcrust::DataType::U8 => NcType::UByte,
        netcrust::DataType::U16 => NcType::UShort,
        netcrust::DataType::U32 => NcType::UInt,
        netcrust::DataType::I64 => NcType::Int64,
        netcrust::DataType::U64 => NcType::UInt64,
        _ => return None,
    })
}

fn map_attr(value: &netcrust::AttributeValue) -> Result<AttrValue, String> {
    use netcrust::AttributeValue as V;
    Ok(match value {
        V::Chars(text) => AttrValue::Text(text.clone()),
        // A NetCDF-4 file can carry a variable-length string attribute.
        // Classic has no string type, so a single string becomes NC_CHAR
        // text and an array of them is refused rather than joined -- a
        // join would silently change the attribute's meaning.
        V::Strings(items) if items.len() == 1 => AttrValue::Text(items[0].clone()),
        V::Strings(items) => {
            return Err(format!(
                "is an array of {} strings; classic NetCDF has no string type and \
                 joining them would change the attribute",
                items.len()
            ))
        }
        V::Bytes(v) => AttrValue::Bytes(v.clone()),
        V::Shorts(v) => AttrValue::Shorts(v.clone()),
        V::Ints(v) => AttrValue::Ints(v.clone()),
        V::Floats(v) => AttrValue::Floats(v.clone()),
        V::Doubles(v) => AttrValue::Doubles(v.clone()),
        V::UBytes(v) => AttrValue::UBytes(v.clone()),
        V::UShorts(v) => AttrValue::UShorts(v.clone()),
        V::UInts(v) => AttrValue::UInts(v.clone()),
        V::Int64s(v) => AttrValue::Int64s(v.clone()),
        V::UInt64s(v) => AttrValue::UInt64s(v.clone()),
    })
}

/// Read one variable in its native type and push it through the writer,
/// record by record when it rides the record dimension.
fn copy_variable(
    values: &netcdf_reader::NcFile,
    writer: &mut NcWriter,
    varid: usize,
    var: &netcrust::Variable,
    numrecs: u64,
) -> Result<(), String> {
    let name = var.name();
    let is_record = var
        .dimensions()
        .first()
        .is_some_and(|dim| dim.is_unlimited());
    let slab_elems: usize = if is_record {
        var.dimensions().iter().skip(1).map(|d| d.len()).product()
    } else {
        var.dimensions().iter().map(|d| d.len()).product()
    };

    macro_rules! copy {
        ($t:ty, $variant:path) => {{
            let array = values
                .read_variable::<$t>(name)
                .map_err(|err| format!("read '{name}': {err}"))?;
            let flat = array.into_raw_vec_and_offset().0;
            push(writer, varid, name, is_record, numrecs, slab_elems, &flat, |slice| {
                $variant(slice)
            })?;
        }};
    }

    match var.dtype() {
        netcrust::DataType::F32 => copy!(f32, VarData::F32),
        netcrust::DataType::F64 => copy!(f64, VarData::F64),
        netcrust::DataType::I32 => copy!(i32, VarData::I32),
        netcrust::DataType::I16 => copy!(i16, VarData::I16),
        netcrust::DataType::I8 => copy!(i8, VarData::I8),
        netcrust::DataType::U8 => copy!(u8, VarData::U8),
        netcrust::DataType::U16 => copy!(u16, VarData::U16),
        netcrust::DataType::U32 => copy!(u32, VarData::U32),
        netcrust::DataType::I64 => copy!(i64, VarData::I64),
        netcrust::DataType::U64 => copy!(u64, VarData::U64),
        netcrust::DataType::Char | netcrust::DataType::String => {
            let flat = char_bytes(values, var)?;
            push(writer, varid, name, is_record, numrecs, slab_elems, &flat, VarData::Char)?;
        }
        other => {
            return Err(format!(
                "variable '{name}': {other:?} has no classic external type"
            ))
        }
    }
    Ok(())
}

/// Rebuild a char variable's exact byte block.
///
/// Neither reader hands back raw `NC_CHAR` bytes -- that is gap D1 in the
/// reset map, and it bites here too. Both offer the variable decoded to
/// strings with trailing NULs trimmed, and the two containers decode it
/// to DIFFERENT shapes:
///
/// * **classic** treats the last dimension as the string length, so a
///   `Times(Time, DateStrLen)` comes back as one string per record;
/// * **NetCDF-4** stores `NC_CHAR` as an HDF5 fixed string of size 1, so
///   the same variable comes back as one string per CHARACTER.
///
/// Both are accepted, told apart by the count, and anything that is
/// neither is refused by name rather than guessed at.
fn char_bytes(
    values: &netcdf_reader::NcFile,
    var: &netcrust::Variable,
) -> Result<Vec<u8>, String> {
    let name = var.name();
    let width = var
        .dimensions()
        .last()
        .map(|dim| dim.len())
        .filter(|len| *len > 0)
        .ok_or_else(|| format!("char variable '{name}' has no trailing length dimension"))?;
    let total: usize = var.dimensions().iter().map(|dim| dim.len()).product();
    let strings = values
        .read_variable_as_strings(name)
        .map_err(|err| format!("read '{name}' as strings: {err}"))?;

    let chunk = if strings.len() == total.div_ceil(width) {
        width // classic: one string per row
    } else if strings.len() == total {
        1 // NetCDF-4: one string per character
    } else {
        return Err(format!(
            "char variable '{name}' decoded to {} string(s); expected {} \
             (one per row of {width}) or {total} (one per character)",
            strings.len(),
            total.div_ceil(width)
        ));
    };

    let mut flat = Vec::with_capacity(total);
    for (index, text) in strings.iter().enumerate() {
        let bytes = text.as_bytes();
        if bytes.len() > chunk {
            return Err(format!(
                "char variable '{name}' element {index} is {} bytes but its slot is {chunk}",
                bytes.len()
            ));
        }
        flat.extend_from_slice(bytes);
        flat.resize(flat.len() + (chunk - bytes.len()), 0);
    }
    if flat.len() != total {
        return Err(format!(
            "char variable '{name}' rebuilt {} byte(s) but the shape wants {total}",
            flat.len()
        ));
    }
    Ok(flat)
}

#[allow(clippy::too_many_arguments)]
fn push<'a, T, F>(
    writer: &mut NcWriter,
    varid: usize,
    name: &str,
    is_record: bool,
    numrecs: u64,
    slab_elems: usize,
    flat: &'a [T],
    wrap: F,
) -> Result<(), String>
where
    F: Fn(&'a [T]) -> VarData<'a>,
{
    if !is_record {
        return writer
            .write_var(varid, wrap(flat))
            .map_err(|err| format!("write '{name}': {err}"));
    }
    let want = slab_elems
        .checked_mul(numrecs as usize)
        .ok_or_else(|| format!("variable '{name}' element count overflows usize"))?;
    if flat.len() != want {
        return Err(format!(
            "variable '{name}' read back {} value(s) but {numrecs} record(s) of \
             {slab_elems} were expected",
            flat.len()
        ));
    }
    for record in 0..numrecs {
        let start = record as usize * slab_elems;
        writer
            .write_record(record, varid, wrap(&flat[start..start + slab_elems]))
            .map_err(|err| format!("write record {record} of '{name}': {err}"))?;
    }
    Ok(())
}
