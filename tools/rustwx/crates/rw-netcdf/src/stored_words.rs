//! Exact stored words for resident history frames. No CF decoding or promotion.
use netcrust::DataType as T;
use serde::Serialize;
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Serialize)]
struct Record {
    name: String,
    filename: String,
    shape: Vec<usize>,
    dtype: &'static str,
}

pub fn run(path: &Path, output: &Path, names: &[String]) -> Result<(), String> {
    let (file, _) = super::open(path)?;
    fs::create_dir_all(output).map_err(|e| e.to_string())?;
    let mut records = Vec::new();
    for (index, name) in names.iter().enumerate() {
        let variable = file
            .variable(name)
            .ok_or_else(|| format!("unknown variable {name}"))?;
        let filename = format!("{index}.bin");
        let mut writer =
            BufWriter::new(fs::File::create(output.join(&filename)).map_err(|e| e.to_string())?);
        macro_rules! numeric {
            ($ty:ty, $dtype:literal) => {{
                let array = file
                    .read_array::<$ty>(name)
                    .map_err(|e| format!("{name}: {e}"))?;
                for value in &array {
                    writer
                        .write_all(&value.to_le_bytes())
                        .map_err(|e| e.to_string())?;
                }
                $dtype
            }};
        }
        let dtype = match variable.dtype() {
            T::I8 => numeric!(i8, "|i1"),
            T::U8 => numeric!(u8, "|u1"),
            T::I16 => numeric!(i16, "<i2"),
            T::U16 => numeric!(u16, "<u2"),
            T::I32 => numeric!(i32, "<i4"),
            T::U32 => numeric!(u32, "<u4"),
            T::I64 => numeric!(i64, "<i8"),
            T::U64 => numeric!(u64, "<u8"),
            T::F32 => numeric!(f32, "<f4"),
            T::F64 => numeric!(f64, "<f8"),
            T::Char => {
                let (bytes, _) = super::character_bytes(&file, &variable, false)?;
                writer.write_all(&bytes).map_err(|e| e.to_string())?;
                "|S1"
            }
            T::String => {
                // NetCDF-4 fixed S1 arrays are exposed as String by the
                // pinned HDF5 metadata reader. Admit only one-byte slots;
                // variable-width strings have no fixed stored-word ABI.
                let (bytes, width) = super::character_bytes(&file, &variable, true)?;
                if width.is_some() {
                    return Err(format!(
                        "{name}: stored-words requires single-byte character slots"
                    ));
                }
                writer.write_all(&bytes).map_err(|e| e.to_string())?;
                "|S1"
            }
            other => return Err(format!("{name}: stored-words does not support {other:?}")),
        };
        writer.flush().map_err(|e| e.to_string())?;
        records.push(Record {
            name: name.clone(),
            filename,
            shape: variable.shape(),
            dtype,
        });
    }
    let metadata =
        serde_json::json!({"schema": "gpuwm-rw-netcdf-stored-words-v1", "variables": records});
    fs::write(
        output.join("metadata.json"),
        serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
