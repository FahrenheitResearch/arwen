//! Sealed output packets (little-endian raw arrays + manifest with SHA-256)
//! and CFE parameter rows.
use crate::cfe::{self, Params};
use crate::Result;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut block = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Write an existing catchment-major history without flattening or copying the
/// entire continental matrix. Independent file handles own disjoint extents.
pub fn write_history(dir: &Path, rows: &[&[f64]], hours: usize) -> Result<Value> {
    if rows.iter().any(|r| r.len() != hours) {
        return Err("runoff history rows do not share the declared hour axis".into());
    }
    let bytes = rows
        .len()
        .checked_mul(hours)
        .and_then(|n| n.checked_mul(8))
        .ok_or("runoff history size overflow")?;
    let path = dir.join("runoff_replay.f64");
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?
        .set_len(bytes as u64)?;
    rows.par_chunks(1024)
        .enumerate()
        .try_for_each(|(i, part)| -> Result<()> {
            let mut data = Vec::with_capacity(part.len() * hours * 8);
            for row in part {
                for value in *row {
                    data.extend_from_slice(&value.to_le_bytes());
                }
            }
            let mut file = std::fs::OpenOptions::new().write(true).open(&path)?;
            file.seek(SeekFrom::Start((i * 1024 * hours * 8) as u64))?;
            file.write_all(&data)?;
            Ok(())
        })?;
    Ok(
        json!({"name":"runoff_replay", "file":"runoff_replay.f64", "dtype":"<f8",
        "shape":[rows.len(),hours], "bytes":bytes, "sha256":sha256_file(&path)?,
        "units":"m per hour", "meaning":"CFE Qout at every replay hour; catchment-major, chronological within each catchment"}),
    )
}

/// Transpose chronological slabs into the unchanged canonical catchment-major
/// packet, using at most 1024 catchment histories per Rayon worker.
pub fn combine_history(
    dir: &Path,
    slabs: &[(std::path::PathBuf, usize)],
    catches: usize,
) -> Result<Value> {
    let hours: usize = slabs.iter().map(|(_, h)| h).sum();
    for (path, h) in slabs {
        if std::fs::metadata(path)?.len() != (catches * h * 8) as u64 {
            return Err("runoff slab extent does not match its catchment/hour axes".into());
        }
    }
    let bytes = catches
        .checked_mul(hours)
        .and_then(|n| n.checked_mul(8))
        .ok_or("runoff history size overflow")?;
    let path = dir.join("runoff_replay.f64");
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?
        .set_len(bytes as u64)?;
    (0..catches.div_ceil(1024))
        .into_par_iter()
        .try_for_each(|block| -> Result<()> {
            let first = block * 1024;
            let count = (catches - first).min(1024);
            let mut data = vec![0u8; count * hours * 8];
            let mut hour = 0;
            for (part, h) in slabs {
                let mut file = std::fs::File::open(part)?;
                file.seek(SeekFrom::Start((first * h * 8) as u64))?;
                let mut source = vec![0; count * h * 8];
                file.read_exact(&mut source)?;
                for k in 0..count {
                    data[(k * hours + hour) * 8..(k * hours + hour + h) * 8]
                        .copy_from_slice(&source[k * h * 8..(k + 1) * h * 8]);
                }
                hour += h;
            }
            let mut file = std::fs::OpenOptions::new().write(true).open(&path)?;
            file.seek(SeekFrom::Start((first * hours * 8) as u64))?;
            file.write_all(&data)?;
            Ok(())
        })?;
    Ok(
        json!({"name":"runoff_replay", "file":"runoff_replay.f64", "dtype":"<f8",
        "shape":[catches,hours], "bytes":bytes, "sha256":sha256_file(&path)?,
        "units":"m per hour", "meaning":"CFE Qout at every replay hour; catchment-major, chronological within each catchment"}),
    )
}

#[cfg(test)]
mod history_tests {
    use super::*;
    #[test]
    fn continental_history_blocks_preserve_legacy_bytes_and_axes() {
        let dir = std::env::temp_dir().join(format!("hydro-history-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let rows: Vec<Vec<f64>> = (0..2051)
            .map(|k| {
                (0..5)
                    .map(|h| {
                        if k == 0 && h == 0 {
                            -0.0
                        } else {
                            (k * 5 + h) as f64 / 7.0
                        }
                    })
                    .collect()
            })
            .collect();
        let expected: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let borrowed: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let entry = write_history(&dir, &borrowed, 5).unwrap();
        assert_eq!(
            std::fs::read(dir.join("runoff_replay.f64")).unwrap(),
            expected
        );
        assert_eq!(entry["sha256"], sha256_hex(&expected));
        assert_eq!(entry["shape"], json!([2051, 5]));
        assert!(write_history(&dir, &borrowed, 4).is_err());
        let mut slabs = Vec::new();
        for (i, (first, count)) in [(0, 2), (2, 3)].into_iter().enumerate() {
            let part = dir.join(format!("slab-{i}"));
            std::fs::create_dir(&part).unwrap();
            let slice: Vec<&[f64]> = rows.iter().map(|r| &r[first..first + count]).collect();
            write_history(&part, &slice, count).unwrap();
            slabs.push((part.join("runoff_replay.f64"), count));
        }
        let combined = dir.join("combined");
        std::fs::create_dir(&combined).unwrap();
        let entry = combine_history(&combined, &slabs, rows.len()).unwrap();
        assert_eq!(
            std::fs::read(combined.join("runoff_replay.f64")).unwrap(),
            expected
        );
        assert_eq!(entry["sha256"], sha256_hex(&expected));
        std::fs::remove_dir_all(dir).unwrap();
    }
}

pub fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

pub struct Array {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub units: String,
    pub meaning: String,
}

/// Write arrays as <name>.f64 (little-endian) in parallel and return their
/// manifest entries. The directory must be new: evidence is never overwritten.
pub fn write_arrays(dir: &Path, arrays: &[Array]) -> Result<Vec<Value>> {
    let entries: Vec<Result<Value>> = arrays
        .par_iter()
        .map(|a| {
            let n: usize = a.shape.iter().product();
            if n != a.data.len() {
                return Err(format!("{}: shape {:?} != {} values", a.name, a.shape, a.data.len()).into());
            }
            let mut b = Vec::with_capacity(8 * n);
            for v in &a.data {
                b.extend_from_slice(&v.to_le_bytes());
            }
            let file = format!("{}.f64", a.name);
            std::fs::write(dir.join(&file), &b)?;
            Ok(json!({"name": a.name, "file": file, "dtype": "<f8", "shape": a.shape, "bytes": b.len(),
                      "sha256": sha256_hex(&b), "units": a.units, "meaning": a.meaning}))
        })
        .collect();
    entries.into_iter().collect()
}

pub fn new_dir(path: &str) -> Result<&Path> {
    let p = Path::new(path);
    std::fs::create_dir(p).map_err(|e| format!("{path}: output directory must be new ({e})"))?;
    Ok(p)
}

pub fn write_json(path: &Path, v: &Value) -> Result<String> {
    let s = serde_json::to_string_pretty(v)? + "\n";
    std::fs::write(path, &s)?;
    Ok(sha256_hex(s.as_bytes()))
}

/// Emit the same sorted, pretty JSON object as serde_json::Map without a
/// continental tree of allocated Values. Row encoding runs on all cores;
/// the final writer only joins the ordered encoded chunks.
pub fn write_keyed_json<F>(path: &Path, ids: &[String], value: F) -> Result<String>
where F: Fn(usize) -> Value + Sync {
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.par_sort_unstable_by(|&a, &b| ids[a].cmp(&ids[b]));
    if order.windows(2).any(|w| ids[w[0]] == ids[w[1]]) {
        return Err("state packet has duplicate catchment IDs".into());
    }
    let chunks = order.par_chunks(4096).map(|part| -> Result<Vec<u8>> {
        let mut chunk = Vec::new();
        for (i, &k) in part.iter().enumerate() {
            let mut row = serde_json::Map::new();
            row.insert(ids[k].clone(), value(k));
            let encoded = serde_json::to_vec_pretty(&Value::Object(row))?;
            if i > 0 { chunk.extend_from_slice(b",\n"); }
            chunk.extend_from_slice(&encoded[2..encoded.len()-2]);
        }
        Ok(chunk)
    }).collect::<Result<Vec<_>>>()?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut hash = Sha256::new();
    let mut emit = |bytes: &[u8]| -> Result<()> {
        file.write_all(bytes)?;
        hash.update(bytes);
        Ok(())
    };
    if chunks.is_empty() { emit(b"{}\n")?; } else {
        emit(b"{\n")?;
        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 { emit(b",\n")?; }
            emit(chunk)?;
        }
        emit(b"\n}\n")?;
    }
    file.flush()?;
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod keyed_json_tests {
    use super::*;
    #[test]
    fn continental_keyed_state_has_identical_sorted_pretty_bytes() {
        let path = std::env::temp_dir().join(format!("hydro-keyed-{}.json", std::process::id()));
        for n in [0, 1, 8193] {
            let ids: Vec<String> = (0..n).rev().map(|k| format!("cat-\"{k}-?")).collect();
            let value = |k| json!({"soil": [-0.0, k as f64 / 7.0], "nested": {"snow": null}});
            let expected: serde_json::Map<String, Value> = ids.iter().enumerate().map(|(k,id)| (id.clone(),value(k))).collect();
            let expected = serde_json::to_string_pretty(&Value::Object(expected)).unwrap()+"\n";
            let sha = write_keyed_json(&path, &ids, value).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), expected.as_bytes());
            assert_eq!(sha, sha256_hex(expected.as_bytes()));
        }
        assert!(write_keyed_json(&path, &["same".into(), "same".into()], |_| Value::Null).is_err());
        std::fs::remove_file(path).unwrap();
    }
}

/// Pretty JSON array with independent row encoding and ordered byte joins.
pub fn write_json_rows<F>(path: &Path, count: usize, value: F) -> Result<String>
where F: Fn(usize) -> Value + Sync {
    let chunks = (0..count.div_ceil(4096)).into_par_iter().map(|block| -> Result<Vec<u8>> {
        let mut chunk = Vec::new();
        for k in block*4096..((block+1)*4096).min(count) {
            let encoded = serde_json::to_vec_pretty(&json!([value(k)]))?;
            if k > block*4096 { chunk.extend_from_slice(b",\n"); }
            chunk.extend_from_slice(&encoded[2..encoded.len()-2]);
        }
        Ok(chunk)
    }).collect::<Result<Vec<_>>>()?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut hash = Sha256::new();
    let mut emit = |bytes: &[u8]| -> Result<()> { file.write_all(bytes)?; hash.update(bytes); Ok(()) };
    if chunks.is_empty() { emit(b"[]\n")?; } else {
        emit(b"[\n")?;
        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 { emit(b",\n")?; }
            emit(chunk)?;
        }
        emit(b"\n]\n")?;
    }
    file.flush()?;
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod json_rows_tests {
    use super::*;
    #[test]
    fn continental_parameter_rows_have_identical_pretty_bytes() {
        let path = std::env::temp_dir().join(format!("hydro-rows-{}.json", std::process::id()));
        for n in [0, 1, 8193] {
            let value = |k| json!({"id": format!("cat-{k}-?"), "ini": "snow=0\nsoil=-0.0\n", "nested": [null,k]});
            let expected: Vec<Value> = (0..n).map(value).collect();
            let expected = serde_json::to_string_pretty(&expected).unwrap()+"\n";
            let sha = write_json_rows(&path,n,value).unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), expected.as_bytes());
            assert_eq!(sha,sha256_hex(expected.as_bytes()));
        }
        std::fs::remove_file(path).unwrap();
    }
}

/// CFE parameters per catchment, in divide order: the fabric's own
/// model_attributes when every divide has a row, else one parameter row file
/// (schema arwen.hydro-world.cfe-params-row.v1) for all catchments, which the
/// receipt names. Both go through hydro_world::derive::cfe_ini, the packet the
/// engine's nextgen-cfe-prepare reads, then the same INI parser as the port.
pub fn params_for(fabric: &hydro_world::contract::Fabric, ids: &[String], row_file: Option<&str>) -> Result<(Vec<Params>, Vec<String>, Value)> {
    use hydro_world::derive::{cfe_ini, CfeInitial};
    let init = CfeInitial { soil_storage: 0.5, gw_storage: 0.5 };
    if let Some(m) = fabric.model_attributes.as_ref().filter(|m| m.len() == ids.len()) {
        let by_id: std::collections::BTreeMap<&str, &hydro_world::contract::ModelAttributes> =
            m.iter().filter_map(|r| r.divide_id.as_deref().map(|d| (d, r))).collect();
        // One INI per catchment, built and parsed on all cores, in id order.

        let pairs = ids
            .par_iter()
            .map(|id| -> Result<(Params, String)> {
                let r = by_id.get(id.as_str()).ok_or_else(|| format!("{id}: no model_attributes row"))?;
                let ini = cfe_ini(r, init)?;
                Ok((cfe::parse_ini(&ini)?, ini))
            })
            .collect::<Result<Vec<_>>>()?;
        let (ps, inis): (Vec<Params>, Vec<String>) = pairs.into_iter().unzip();
        return Ok((ps, inis, json!({"source": "fabric model_attributes", "gw_storage_cold_start": init.gw_storage})));
    }
    let path = row_file.ok_or("fabric has no model_attributes; pass a CFE parameter row file (params)")?;
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if v["schema"] != "arwen.hydro-world.cfe-params-row.v1" {
        return Err(format!("{path}: not an arwen.hydro-world.cfe-params-row.v1 file").into());
    }
    let m: hydro_world::contract::ModelAttributes = serde_json::from_value(v["model_attributes"].clone())?;
    let gw0 = v["gw_storage_cold_start"].as_f64().ok_or("params row lacks gw_storage_cold_start")?;
    let ini = cfe_ini(&m, CfeInitial { soil_storage: 0.5, gw_storage: gw0 })?;
    let mut p = cfe::parse_ini(&ini)?;
    p.gw_storage = gw0;
    let bytes = std::fs::read(path)?;
    Ok((vec![p; ids.len()], vec![ini; ids.len()],
        json!({"source": "parameter row file", "file": path, "sha256": sha256_hex(&bytes),
               "provenance": v["provenance"], "gw_storage_cold_start": gw0})))
}
