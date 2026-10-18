//! Typed stored-word NetCDF extraction for subsystem replay.
//! Python consumes only these Rust-produced resident-array bridge files.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    path::PathBuf,
};

#[derive(Deserialize)]
struct Request {
    output: PathBuf,
    arrays: Vec<Array>,
}
#[derive(Deserialize)]
struct Array {
    name: String,
    source: PathBuf,
    source_sha256: String,
    variable: String,
    dtype: String,
}

fn npy(shape: &[usize], dtype: &str, words: &[u8]) -> Result<Vec<u8>, String> {
    let tuple = shape
        .iter()
        .map(|x| format!("{x},"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut header =
        format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({tuple}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let length = u16::try_from(header.len()).map_err(|_| "NPY header too large")?;
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(words);
    Ok(bytes)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let request = std::env::args()
        .nth(1)
        .ok_or("usage: rw_mpas_replay_arrays REQUEST.json")?;
    let req: Request = serde_json::from_slice(&fs::read(request)?)?;
    fs::create_dir_all(&req.output)?;
    let mut files = HashMap::new();
    let mut rows = Vec::new();
    for array in req.arrays {
        if array.name.is_empty()
            || !array
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(
                "array name must be a simple identifier to keep output in its directory".into(),
            );
        }
        if !files.contains_key(&array.source) {
            let mut input = fs::File::open(&array.source)?;
            let mut hash = Sha256::new();
            let mut buffer = vec![0u8; 1024 * 1024];
            loop {
                let n = input.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            let digest = format!("{:x}", hash.finalize());
            if digest != array.source_sha256 {
                return Err("source words differ from sealed publication".into());
            }
            files.insert(
                array.source.clone(),
                (netcrust::File::open(&array.source)?, digest),
            );
        }
        let (file, digest) = &files[&array.source];
        if digest != &array.source_sha256 {
            return Err("inconsistent source hash in request".into());
        }
        let var = file
            .variable(&array.variable)
            .ok_or("missing replay variable")?;
        let mut shape = var.shape();
        // A file with multiple time records must be selected explicitly by a
        // future request schema; silently flattening Time corrupts cell layout.
        if var.dimensions().first().is_some_and(|d| d.name() == "Time") {
            if shape[0] != 1 {
                return Err("replay requires exactly one stored Time record".into());
            }
            shape.remove(0);
        }
        let words: Vec<u8> = match array.dtype.as_str() {
            "<f4" if var.dtype() == &netcrust::DataType::F32 => file
                .read_array::<f32>(&array.variable)?
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
            "<i4" if var.dtype() == &netcrust::DataType::I32 => file
                .read_array::<i32>(&array.variable)?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            _ => {
                return Err(
                    "stored dtype must match requested f32/i32; promotion changes replay words"
                        .into(),
                );
            }
        };
        let bytes = npy(&shape, &array.dtype, &words)?;
        let path = req.output.join(format!("{}.npy", array.name));
        fs::File::create_new(&path)?.write_all(&bytes)?;
        rows.push(
            serde_json::json!({"name": array.name, "source": array.source,
            "source_sha256": digest,
            "variable": array.variable, "dtype": array.dtype, "shape": shape,
            "words": words.len()/4, "word_sha256": format!("{:x}", Sha256::digest(&words)),
            "npy_sha256": format!("{:x}", Sha256::digest(&bytes))}),
        );
    }
    fs::File::create_new(req.output.join("arrays.json"))?.write_all(&serde_json::to_vec_pretty(
        &serde_json::json!({"schema": "mpas.replay-arrays/v1", "arrays": rows}),
    )?)?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_signed_zero_and_nan_payload() {
        let words = [0x80000000u32, 0x7fc01234u32, 0x3f800000u32];
        let raw: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
        let out = npy(&[1, 3], "<f4", &raw).unwrap();
        let offset = 10 + u16::from_le_bytes([out[8], out[9]]) as usize;
        assert_eq!(offset % 64, 0);
        assert_eq!(&out[offset..], &raw);
        assert!(
            std::str::from_utf8(&out[10..offset])
                .unwrap()
                .contains("(1, 3,)")
        );
    }
}
