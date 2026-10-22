//! Zarr format 2 on a file system: one directory per array, a `.zarray` and
//! a `.zattrs` in it, chunk files named by their grid index, and the group's
//! consolidated `.zmetadata` written last.
//!
//! Format 2 because xarray, zarr-python 2.18 and zarr-python 3 all open it
//! and a great deal of ML code still pins zarr 2.  `_ARRAY_DIMENSIONS` on
//! every array is xarray's format-2 convention for dimension names.

use rayon::prelude::*;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::blosc;
use crate::error::{fail, Result};

/// Zstandard level inside the Blosc frames.
pub const CLEVEL: i32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    U8,
    I32,
    I64,
    F64,
}

impl Dtype {
    pub fn zarr(self) -> &'static str {
        match self {
            Dtype::F32 => "<f4",
            Dtype::U8 => "|u1",
            Dtype::I32 => "<i4",
            Dtype::I64 => "<i8",
            Dtype::F64 => "<f8",
        }
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::I64 | Dtype::F64 => 8,
        }
    }

    /// The `.zarray` fill value.  Floats are NaN-filled.  Integers carry
    /// none: xarray turns a declared fill value into a mask, which would
    /// erase every 0 in the uint8 `below_ground` mask.
    fn fill(self) -> Value {
        match self {
            Dtype::F32 | Dtype::F64 => Value::String("NaN".into()),
            _ => Value::Null,
        }
    }
}

/// Everything a `.zarray` and `.zattrs` say about one array.
#[derive(Debug, Clone)]
pub struct ArrayMeta {
    pub shape: Vec<usize>,
    pub chunks: Vec<usize>,
    pub dtype: Dtype,
    pub dims: Vec<String>,
    pub attrs: Map<String, Value>,
}

impl ArrayMeta {
    fn zarray(&self) -> Value {
        json!({
            "chunks": self.chunks,
            "compressor": blosc::zarr_compressor(CLEVEL),
            "dimension_separator": ".",
            "dtype": self.dtype.zarr(),
            "fill_value": self.dtype.fill(),
            "filters": Value::Null,
            "order": "C",
            "shape": self.shape,
            "zarr_format": 2,
        })
    }

    fn zattrs(&self) -> Value {
        let mut attrs = self.attrs.clone();
        attrs.insert(
            "_ARRAY_DIMENSIONS".into(),
            Value::Array(self.dims.iter().cloned().map(Value::String).collect()),
        );
        Value::Object(attrs)
    }
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value).map_err(|e| fail(format!("{e}")))?;
    let mut file = fs::File::create(path)
        .map_err(|e| fail(format!("could not write {}: {e}", path.display())))?;
    file.write_all(text.as_bytes())?;
    file.write_all(b"\n")?;
    Ok(())
}

/// The chunk key for a chunk-grid index.
pub fn chunk_key(index: &[usize]) -> String {
    index
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

/// Compress and write one chunk; returns the bytes written.
pub fn write_chunk(
    store: &Path,
    array: &str,
    index: &[usize],
    raw: &[u8],
    dtype: Dtype,
) -> Result<u64> {
    let dir = store.join(array);
    fs::create_dir_all(&dir)?;
    let frame = blosc::compress(raw, dtype.size(), CLEVEL)?;
    let path = dir.join(chunk_key(index));
    fs::write(&path, &frame)
        .map_err(|e| fail(format!("could not write {}: {e}", path.display())))?;
    Ok(frame.len() as u64)
}

/// Little-endian bytes of a typed slice.
fn little_endian_bytes<T: Copy + Sync, const N: usize>(
    values: &[T],
    encode: impl Fn(T) -> [u8; N] + Sync,
) -> Vec<u8> {
    let mut bytes = vec![0; values.len() * N];
    bytes
        .par_chunks_mut(N)
        .zip(values.par_iter())
        .for_each(|(out, value)| out.copy_from_slice(&encode(*value)));
    bytes
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    little_endian_bytes(values, f32::to_le_bytes)
}

pub fn f64_bytes(values: &[f64]) -> Vec<u8> {
    little_endian_bytes(values, f64::to_le_bytes)
}

pub fn i64_bytes(values: &[i64]) -> Vec<u8> {
    little_endian_bytes(values, i64::to_le_bytes)
}

pub fn i32_bytes(values: &[i32]) -> Vec<u8> {
    little_endian_bytes(values, i32::to_le_bytes)
}

/// Compress and write the one chunk of a 0-d array (key `0`).
pub fn write_scalar_chunk(store: &Path, array: &str, raw: &[u8], dtype: Dtype) -> Result<u64> {
    let dir = store.join(array);
    fs::create_dir_all(&dir)?;
    let frame = blosc::compress(raw, dtype.size(), CLEVEL)?;
    fs::write(dir.join("0"), &frame)?;
    Ok(frame.len() as u64)
}

/// Write an array's `.zarray` and `.zattrs`.
pub fn write_meta(store: &Path, array: &str, meta: &ArrayMeta) -> Result<()> {
    let dir = store.join(array);
    fs::create_dir_all(&dir)?;
    write_json(&dir.join(".zarray"), &meta.zarray())?;
    write_json(&dir.join(".zattrs"), &meta.zattrs())?;
    Ok(())
}

/// Write a whole small array as one chunk plus its metadata.
pub fn write_small(store: &Path, array: &str, meta: &ArrayMeta, raw: &[u8]) -> Result<u64> {
    debug_assert_eq!(meta.shape, meta.chunks);
    write_meta(store, array, meta)?;
    let zeros = vec![0usize; meta.shape.len()];
    write_chunk(store, array, &zeros, raw, meta.dtype)
}

/// Write the group's `.zgroup` and `.zattrs`.
pub fn write_group(store: &Path, attrs: &Map<String, Value>) -> Result<()> {
    fs::create_dir_all(store)?;
    write_json(&store.join(".zgroup"), &json!({"zarr_format": 2}))?;
    write_json(&store.join(".zattrs"), &Value::Object(attrs.clone()))?;
    Ok(())
}

/// Every `.zgroup`, `.zattrs` and `.zarray` under `root`, keyed by its
/// path relative to `root` with `prefix` in front, as one consolidated
/// metadata document.  The export folder and the ZIP's root are groups
/// over their domains' stores, so a reader that opens the root and walks
/// to a domain (xarray with zarr 3 does exactly that for `group=`) finds
/// the whole hierarchy in one read.
pub fn consolidated_tree(root: &Path, prefix: &str) -> Result<Value> {
    let mut metadata = Map::new();
    let mut stack = vec![root.to_path_buf()];
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type()?.is_dir() {
                if !name.starts_with('.') {
                    stack.push(path);
                }
            } else if matches!(name.as_str(), ".zgroup" | ".zattrs" | ".zarray") {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|e| fail(format!("{e}")))?
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                found.push((format!("{prefix}{relative}"), path));
            }
        }
    }
    found.sort();
    for (key, path) in found {
        let text = fs::read_to_string(&path)?;
        let value: Value =
            serde_json::from_str(&text).map_err(|e| fail(format!("{}: {e}", path.display())))?;
        metadata.insert(key, value);
    }
    Ok(json!({"metadata": Value::Object(metadata), "zarr_consolidated_format": 1}))
}

/// `value` as the bytes [`write_json`] would write.
pub fn json_bytes(value: &Value) -> Vec<u8> {
    let mut text = serde_json::to_string_pretty(value).unwrap_or_default();
    text.push('\n');
    text.into_bytes()
}

/// Write `value` to `path` as [`write_json`] does.
pub fn write_document(path: &Path, value: &Value) -> Result<()> {
    write_json(path, value)
}

/// Every metadata document in the store, consolidated into `.zmetadata`.
/// Written last, so a store with a `.zmetadata` is a finished store.
pub fn consolidate(store: &Path) -> Result<()> {
    let mut metadata = Map::new();
    let mut read = |relative: String, path: PathBuf| -> Result<()> {
        let text = fs::read_to_string(&path)?;
        let value: Value =
            serde_json::from_str(&text).map_err(|e| fail(format!("{}: {e}", path.display())))?;
        metadata.insert(relative, value);
        Ok(())
    };
    for name in [".zgroup", ".zattrs"] {
        read(name.to_string(), store.join(name))?;
    }
    let mut arrays: Vec<String> = fs::read_dir(store)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join(".zarray").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    arrays.sort();
    for array in arrays {
        for name in [".zarray", ".zattrs"] {
            let path = store.join(&array).join(name);
            if path.is_file() {
                read(format!("{array}/{name}"), path)?;
            }
        }
    }
    write_json(
        &store.join(".zmetadata"),
        &json!({"metadata": Value::Object(metadata), "zarr_consolidated_format": 1}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_keys_are_dot_separated() {
        assert_eq!(chunk_key(&[3, 0, 0, 0]), "3.0.0.0");
        assert_eq!(chunk_key(&[0]), "0");
    }

    #[test]
    fn integer_arrays_declare_no_fill_value() {
        let meta = ArrayMeta {
            shape: vec![2, 3],
            chunks: vec![2, 3],
            dtype: Dtype::U8,
            dims: vec!["y".into(), "x".into()],
            attrs: Map::new(),
        };
        assert_eq!(meta.zarray()["fill_value"], Value::Null);
        let float = ArrayMeta {
            dtype: Dtype::F32,
            ..meta
        };
        assert_eq!(float.zarray()["fill_value"], Value::String("NaN".into()));
        assert_eq!(float.zattrs()["_ARRAY_DIMENSIONS"], json!(["y", "x"]));
    }
}

/// Reuse the written Blosc chunks with format-3 metadata. Earth2Studio 0.19's
/// ZarrBackend reads dimension_names, whereas its xarray source reads CF attrs.
/// The v2 key codec keeps every chunk byte and filename unchanged.
pub fn promote_v3(store: &Path) -> Result<()> {
    let attrs: Value = serde_json::from_slice(&fs::read(store.join(".zattrs"))?)
        .map_err(|e| fail(e.to_string()))?;
    let mut metadata = Map::new();
    for entry in fs::read_dir(store)? {
        let path = entry?.path();
        if !path.join(".zarray").is_file() {
            continue;
        }
        let old: Value = serde_json::from_slice(&fs::read(path.join(".zarray"))?)
            .map_err(|e| fail(e.to_string()))?;
        let mut attributes: Value = serde_json::from_slice(&fs::read(path.join(".zattrs"))?)
            .map_err(|e| fail(e.to_string()))?;
        let dims = attributes
            .as_object_mut()
            .unwrap()
            .remove("_ARRAY_DIMENSIONS")
            .unwrap();
        let (dtype, size) = match old["dtype"].as_str().unwrap_or("") {
            "<f4" => ("float32", 4),
            "<f8" => ("float64", 8),
            "<i8" => ("int64", 8),
            "<i4" => ("int32", 4),
            "|u1" => ("uint8", 1),
            d => return Err(fail(format!("no format-3 data type for {d}"))),
        };
        let fill = if old["fill_value"].is_null() {
            json!(0)
        } else {
            old["fill_value"].clone()
        };
        let meta = json!({"zarr_format":3,"node_type":"array","shape":old["shape"],
            "data_type":dtype,"chunk_grid":{"name":"regular","configuration":{"chunk_shape":old["chunks"]}},
            "chunk_key_encoding":{"name":"v2","configuration":{"separator":"."}},
            "fill_value":fill,"codecs":[{"name":"bytes","configuration":{"endian":"little"}},
                {"name":"blosc","configuration":{"typesize":size,"cname":"zstd","clevel":CLEVEL,"shuffle":"shuffle","blocksize":0}}],
            "dimension_names":dims,"attributes":attributes});
        write_json(&path.join("zarr.json"), &meta)?;
        metadata.insert(
            path.file_name().unwrap().to_string_lossy().into_owned(),
            meta,
        );
        fs::remove_file(path.join(".zarray"))?;
        fs::remove_file(path.join(".zattrs"))?;
    }
    write_json(
        &store.join("zarr.json"),
        &json!({"zarr_format":3,"node_type":"group","attributes":attrs,
        "consolidated_metadata":{"kind":"inline","must_understand":false,"metadata":metadata}}),
    )?;
    for name in [".zgroup", ".zattrs", ".zmetadata"] {
        fs::remove_file(store.join(name))?;
    }
    Ok(())
}
