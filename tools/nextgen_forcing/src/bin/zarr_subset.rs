// SPDX-License-Identifier: Apache-2.0
//! Setup-only, bounded local Zarr2 weather transport into the shared Rust NetCDF writer.
use arwen_nextgen_forcing::{sha256, Result};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    metadata: PathBuf,
    metadata_sha256: String,
    coordinate_manifest: PathBuf,
    coordinate_manifest_sha256: String,
    chunks_root: PathBuf,
    output_dir: PathBuf,
    // Ordered source axes; half-open source indices, no geographic inference.
    axes: [Axis; 3],
    fields: Vec<Field>,
    chunk_sha256: BTreeMap<String, String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Axis {
    name: String,
    start: usize,
    end: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Field {
    source: String,
    units: String,
}
struct Decoded {
    values: Vec<f64>,
    provenance: Value,
    keys: BTreeSet<String>,
}
fn fail<T>(s: impl Into<String>) -> Result<T> {
    Err(s.into().into())
}
fn safe(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}
fn sealed(path: &Path, hash: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if sha256(&bytes) != hash {
        return fail(format!("{}: SHA256 mismatch", path.display()));
    }
    Ok(bytes)
}
fn extents(v: &Value) -> Result<[usize; 3]> {
    let a = v.as_array().ok_or("missing source shape/chunks")?;
    if a.len() != 3 {
        return fail("weather arrays must have rank three");
    }
    let mut n = [0; 3];
    for i in 0..3 {
        n[i] = a[i]
            .as_u64()
            .filter(|&n| n > 0 && n <= 1_000_000)
            .ok_or("invalid source extent")? as usize;
    }
    Ok(n)
}
fn volume(shape: &[usize; 3]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or("extent overflow".into())
}
fn decode(r: &Request, metadata: &Value, f: &Field) -> Result<Decoded> {
    let array = &metadata[format!("{}/.zarray", f.source)];
    let attrs = &metadata[format!("{}/.zattrs", f.source)];
    if array["zarr_format"] != 2
        || array["dtype"] != "<i2"
        || array["order"] != "C"
        || array["compressor"]["id"] != "zstd"
        || !array["filters"].is_null()
        || ![Value::Null, json!(".")].contains(&array["dimension_separator"])
        || attrs["_ARRAY_DIMENSIONS"] != json!(r.axes.iter().map(|a| &a.name).collect::<Vec<_>>())
    {
        return fail(format!(
            "{}: requires declared rank-three C-order <i2 Zarr2/Zstd, dot keys, no filters",
            f.source
        ));
    }
    if attrs["units"].as_str() != Some(f.units.as_str()) {
        return fail(format!("{}: source units differ from request", f.source));
    }
    let shape = extents(&array["shape"])?;
    let chunk = extents(&array["chunks"])?;
    let raw_len = volume(&chunk)?
        .checked_mul(2)
        .filter(|&n| n <= 64 * 1024 * 1024)
        .ok_or("decoded chunk exceeds 64MiB bound")?;
    for i in 0..3 {
        if r.axes[i].end > shape[i] {
            return fail("selection exceeds source shape");
        }
    }
    let scale = attrs
        .get("scale_factor")
        .map(|x| x.as_f64().ok_or("invalid CF scale"))
        .transpose()?
        .unwrap_or(1.0);
    let offset = attrs
        .get("add_offset")
        .map(|x| x.as_f64().ok_or("invalid CF offset"))
        .transpose()?
        .unwrap_or(0.0);
    if !scale.is_finite() || scale == 0.0 || !offset.is_finite() {
        return fail("invalid CF packing transform");
    }
    let mut missing = Vec::new();
    for value in [
        &attrs["_FillValue"],
        &attrs["missing_value"],
        &array["fill_value"],
    ] {
        if !value.is_null() {
            let v = value
                .as_f64()
                .filter(|v| {
                    v.is_finite()
                        && v.fract() == 0.0
                        && *v >= i16::MIN as f64
                        && *v <= i16::MAX as f64
                })
                .ok_or("invalid int16 missing/fill value")?;
            missing.push(v as i16);
        }
    }
    let selected = std::array::from_fn(|i| r.axes[i].end - r.axes[i].start);
    let mut values = vec![f64::NAN; volume(&selected)?];
    let mut inputs = Vec::new();
    let mut keys = BTreeSet::new();
    for t in r.axes[0].start / chunk[0]..=(r.axes[0].end - 1) / chunk[0] {
        for y in r.axes[1].start / chunk[1]..=(r.axes[1].end - 1) / chunk[1] {
            for x in r.axes[2].start / chunk[2]..=(r.axes[2].end - 1) / chunk[2] {
                let key = format!("{}/{t}.{y}.{x}", f.source);
                let expected = r
                    .chunk_sha256
                    .get(&key)
                    .ok_or_else(|| format!("missing chunk seal {key}"))?;
                let path = r.chunks_root.join(&key);
                if fs::metadata(&path)?.len() > raw_len as u64 + 65536 {
                    return fail(format!("{key}: oversized compressed chunk"));
                }
                let compressed = sealed(&path, expected)?;
                let mut raw = Vec::new();
                zstd::stream::read::Decoder::new(compressed.as_slice())?
                    .take(raw_len as u64 + 1)
                    .read_to_end(&mut raw)?;
                if raw.len() != raw_len {
                    return fail(format!("{key}: decoded extent differs"));
                }
                let base = [t * chunk[0], y * chunk[1], x * chunk[2]];
                let lo: [usize; 3] = std::array::from_fn(|i| base[i].max(r.axes[i].start));
                let hi: [usize; 3] =
                    std::array::from_fn(|i| (base[i] + chunk[i]).min(r.axes[i].end));
                for a in lo[0]..hi[0] {
                    for b in lo[1]..hi[1] {
                        for c in lo[2]..hi[2] {
                            let j = ((a - base[0]) * chunk[1] * chunk[2]
                                + (b - base[1]) * chunk[2]
                                + c
                                - base[2])
                                * 2;
                            let word = i16::from_le_bytes([raw[j], raw[j + 1]]);
                            if missing.contains(&word) {
                                return fail(format!("{key}: missing selected source value"));
                            }
                            // Separate rounded multiply/add matches CF unpacking in the reader.
                            let v = word as f64 * scale + offset;
                            if !v.is_finite() {
                                return fail("nonfinite unpacked weather");
                            }
                            let k = ((a - r.axes[0].start) * selected[1] + b - r.axes[1].start)
                                * selected[2]
                                + c
                                - r.axes[2].start;
                            values[k] = v;
                        }
                    }
                }
                inputs.push(json!({"key":key,"sha256":expected,"compressed_bytes":compressed.len(),"decoded_bytes":raw.len()}));
                keys.insert(key);
            }
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return fail("unwritten weather subset");
    }
    Ok(Decoded {
        values,
        keys,
        provenance: json!({"source":f.source,"units":f.units,
        "source_array":array,"source_attributes":attrs,"cf_unpacking":"float64(raw)*scale_factor+add_offset once; missing rejected before unpacking",
        "chunks":inputs}),
    })
}
fn prepare(r: Request) -> Result<Value> {
    let started = Instant::now();
    if r.schema != "nextgen.forcing.zarr-subset-request/v1"
        || r.output_dir.exists()
        || r.fields.is_empty()
        || r.fields.len() > 16
    {
        return fail("requires schema v1, 1..16 fields and fresh output directory");
    }
    let names: BTreeSet<_> = r.axes.iter().map(|a| &a.name).collect();
    if names.len() != 3 || r.axes.iter().any(|a| !safe(&a.name) || a.start >= a.end) {
        return fail("requires three distinct named nonempty source-index ranges");
    }
    let vars: BTreeSet<_> = r.fields.iter().map(|f| &f.source).collect();
    if vars.len() != r.fields.len()
        || r.fields
            .iter()
            .any(|f| !safe(&f.source) || names.contains(&f.source) || f.units.is_empty())
    {
        return fail("invalid/duplicate field name or missing units");
    }
    let selected = std::array::from_fn(|i| r.axes[i].end - r.axes[i].start);
    if volume(&selected)?
        .checked_mul(r.fields.len() * 8)
        .filter(|&n| n <= 128 * 1024 * 1024)
        .is_none()
    {
        return fail("weather output exceeds 128MiB bound; split preparation window");
    }
    let bytes = sealed(&r.metadata, &r.metadata_sha256)?;
    let consolidated: Value = serde_json::from_slice(&bytes)?;
    let metadata = &consolidated["metadata"];
    if consolidated["zarr_consolidated_format"] != 1 {
        return fail("requires consolidated Zarr2 metadata");
    }
    let coordinates: Value = serde_json::from_slice(&sealed(
        &r.coordinate_manifest,
        &r.coordinate_manifest_sha256,
    )?)?;
    if coordinates["schema"] != "nextgen.forcing.zarr-coordinates/v1"
        || coordinates["metadata_sha256"] != r.metadata_sha256
    {
        return fail("coordinate manifest belongs to different metadata/schema");
    }
    let mut axes = Vec::new();
    for axis in &r.axes {
        let a = coordinates["axes"]
            .as_array()
            .ok_or("missing coordinate inventory")?
            .iter()
            .find(|a| a["name"] == axis.name)
            .ok_or("coordinate axis missing")?;
        let info = &a["coordinates"];
        let source = &metadata[format!("{}/.zarray", axis.name)];
        let attrs = &metadata[format!("{}/.zattrs", axis.name)];
        if a["source_array"] != *source
            || a["source_attributes"] != *attrs
            || info["dtype"] != "<f8"
            || info["shape"] != source["shape"]
        {
            return fail("coordinate metadata/shape differs from source");
        }
        let filename = info["path"]
            .as_str()
            .filter(|s| Path::new(s).file_name().and_then(|s| s.to_str()) == Some(*s))
            .ok_or("unsafe coordinate path")?;
        let raw = sealed(
            &r.coordinate_manifest
                .parent()
                .unwrap_or(Path::new("."))
                .join(filename),
            info["sha256"].as_str().ok_or("missing coordinate seal")?,
        )?;
        let n = source["shape"][0]
            .as_u64()
            .filter(|&n| n <= 1_000_000)
            .ok_or("invalid coordinate extent")? as usize;
        if raw.len() != n * 8 || axis.end > n {
            return fail("coordinate extent mismatch");
        }
        let values: Vec<f64> = raw[axis.start * 8..axis.end * 8]
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return fail("nonfinite coordinate");
        }
        // Preserve real coordinate attributes. Bounds are NOT generated or promoted.
        axes.push((values, attrs.clone()));
    }
    let loaded: Vec<Decoded> = r
        .fields
        .par_iter()
        .map(|f| decode(&r, metadata, f))
        .collect::<Result<_>>()?;
    let used: BTreeSet<_> = loaded.iter().flat_map(|d| d.keys.iter().cloned()).collect();
    if used != r.chunk_sha256.keys().cloned().collect() {
        return fail("unused chunk seals; request must name exactly selected chunks");
    }
    for f in &r.fields {
        let shape = extents(&metadata[format!("{}/.zarray", f.source)]["shape"])?;
        for i in 0..3 {
            if shape[i] as u64
                != metadata[format!("{}/.zarray", r.axes[i].name)]["shape"][0]
                    .as_u64()
                    .unwrap_or(0)
            {
                return fail("weather and coordinate source shape differ");
            }
        }
    }
    let decode_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut schema = Schema::new(NcFormat::Offset64);
    schema.put_global_attr(
        "source_store",
        AttrValue::Text(
            coordinates["source_store"]
                .as_str()
                .ok_or("missing source store")?
                .into(),
        ),
    )?;
    schema.put_global_attr(
        "source_metadata_sha256",
        AttrValue::Text(r.metadata_sha256.clone()),
    )?;
    schema.put_global_attr(
        "transport",
        AttrValue::Text("sealed local Zarr2 subset; no spatial resampling or time shift".into()),
    )?;
    let dims: Vec<_> = r
        .axes
        .iter()
        .map(|a| schema.def_dim(&a.name, a.end - a.start, false))
        .collect::<std::result::Result<_, _>>()?;
    let mut axis_ids = Vec::new();
    for (i, axis) in r.axes.iter().enumerate() {
        let id = schema.def_var(&axis.name, NcType::Double, &[dims[i]])?;
        for key in ["units", "calendar", "long_name"] {
            if let Some(v) = axes[i].1[key].as_str() {
                schema.put_var_attr(id, key, AttrValue::Text(v.into()))?;
            }
        }
        axis_ids.push(id);
    }
    let mut field_ids = Vec::new();
    for f in &r.fields {
        let id = schema.def_var(&f.source, NcType::Double, &dims)?;
        let attrs = &metadata[format!("{}/.zattrs", f.source)];
        for key in ["units", "crs", "long_name", "level"] {
            if let Some(v) = attrs[key].as_str() {
                schema.put_var_attr(id, key, AttrValue::Text(v.into()))?;
            }
        }
        field_ids.push(id);
    }
    fs::create_dir(&r.output_dir)?; // Exclusive creation: never overwrite an accepted packet.
    let output = r.output_dir.join("weather.nc");
    let mut writer = NcWriter::create(&output, schema)?;
    for i in 0..3 {
        writer.write_var(axis_ids[i], VarData::F64(&axes[i].0))?;
    }
    for (i, d) in loaded.iter().enumerate() {
        writer.write_var(field_ids[i], VarData::F64(&d.values))?;
    }
    writer.finish()?;
    let receipt = json!({"schema":"nextgen.forcing.zarr-subset/v1","source_store":coordinates["source_store"],
        "metadata_sha256":r.metadata_sha256,"coordinate_manifest_sha256":r.coordinate_manifest_sha256,
        "source_ranges":r.axes.iter().map(|a|json!({"axis":a.name,"start":a.start,"end_exclusive":a.end})).collect::<Vec<_>>(),
        "shape":selected,"fields":loaded.iter().map(|d|&d.provenance).collect::<Vec<_>>(),
        "output":{"path":"weather.nc","sha256":sha256(&fs::read(&output)?),"bytes":fs::metadata(output)?.len()},
        "decode_ms":decode_ms,"total_ms":started.elapsed().as_secs_f64()*1000.0,"rayon_threads":rayon::current_num_threads(),
        "scope":"source-order CF unpacking/NetCDF transport only; not geographic weights, native sampling, model replay or flood skill"});
    fs::write(
        r.output_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(receipt)
}
fn main() {
    let result = (|| -> Result<()> {
        let path = std::env::args_os()
            .nth(1)
            .ok_or("usage: forcing-zarr-subset REQUEST.json")?;
        let r = serde_json::from_slice(&fs::read(path)?)?;
        println!("{}", serde_json::to_string_pretty(&prepare(r)?)?);
        Ok(())
    })();
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(label: &str) -> (PathBuf, Value) {
        let root = std::env::temp_dir().join(format!(
            "forcing-zarr-subset-{}-{label}",
            std::process::id()
        ));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        fs::create_dir(&root).unwrap();
        let mut metadata = serde_json::Map::new();
        let mut axes = Vec::new();
        for (name, units, values) in [
            (
                "clock",
                "seconds since 2000-01-01",
                vec![0.0_f64, 3600.0, 7200.0],
            ),
            ("row", "degrees_north", vec![40.0, 39.0, 38.0]),
            ("col", "degrees_east", vec![-120.0, -119.0, -118.0]),
        ] {
            let array = json!({"shape":[3],"chunks":[2],"dtype":"<f8"});
            let attrs = json!({"_ARRAY_DIMENSIONS":[name],"units":units});
            metadata.insert(format!("{name}/.zarray"), array.clone());
            metadata.insert(format!("{name}/.zattrs"), attrs.clone());
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            fs::write(root.join(format!("{name}.f64")), &raw).unwrap();
            axes.push(json!({"name":name,"source_array":array,"source_attributes":attrs,
                "coordinates":{"path":format!("{name}.f64"),"shape":[3],"dtype":"<f8","sha256":sha256(&raw)}}));
        }
        metadata.insert(
            "arbitrary_weather_731/.zarray".into(),
            json!({"zarr_format":2,"shape":[3,3,3],"chunks":[2,2,2],
            "dtype":"<i2","order":"C","compressor":{"id":"zstd"},"fill_value":null,"filters":null}),
        );
        metadata.insert(
            "arbitrary_weather_731/.zattrs".into(),
            json!({"_ARRAY_DIMENSIONS":["clock","row","col"],
            "units":"K","scale_factor":0.25,"add_offset":270.0,"_FillValue":-32767}),
        );
        fs::create_dir(root.join("arbitrary_weather_731")).unwrap();
        let mut seals = BTreeMap::new();
        for t in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let mut raw = Vec::new();
                    for a in t * 2..t * 2 + 2 {
                        for b in y * 2..y * 2 + 2 {
                            for c in x * 2..x * 2 + 2 {
                                let word: i16 =
                                    if a >= 3 || b >= 3 || c >= 3 || a == 0 || b == 0 || c == 0 {
                                        -32767
                                    } else {
                                        (100 * a + 10 * b + c) as i16
                                    };
                                raw.extend(word.to_le_bytes());
                            }
                        }
                    }
                    let compressed = zstd::stream::encode_all(raw.as_slice(), 3).unwrap();
                    let key = format!("arbitrary_weather_731/{t}.{y}.{x}");
                    fs::write(root.join(&key), &compressed).unwrap();
                    seals.insert(key, sha256(&compressed));
                }
            }
        }
        let bytes =
            serde_json::to_vec(&json!({"zarr_consolidated_format":1,"metadata":metadata})).unwrap();
        fs::write(root.join("metadata.json"), &bytes).unwrap();
        let mhash = sha256(&bytes);
        let coords = serde_json::to_vec(&json!({"schema":"nextgen.forcing.zarr-coordinates/v1",
            "source_store":"synthetic://transport-only","metadata_sha256":mhash,"axes":axes}))
        .unwrap();
        fs::write(root.join("coordinates.json"), &coords).unwrap();
        let r = json!({"schema":"nextgen.forcing.zarr-subset-request/v1","metadata":root.join("metadata.json"),
            "metadata_sha256":mhash,"coordinate_manifest":root.join("coordinates.json"),"coordinate_manifest_sha256":sha256(&coords),
            "chunks_root":root,"output_dir":root.join("output"),"axes":[{"name":"clock","start":1,"end":3},
            {"name":"row","start":1,"end":3},{"name":"col","start":1,"end":3}],
            "fields":[{"source":"arbitrary_weather_731","units":"K"}],"chunk_sha256":seals});
        (root, r)
    }
    #[test]
    fn padded_multi_chunk_subset_preserves_source_order_and_unpacks_once() {
        let (root, r) = fixture("transport");
        prepare(serde_json::from_value(r.clone()).unwrap()).unwrap();
        let file = netcdf_reader::NcFile::open(root.join("output/weather.nc")).unwrap();
        let data = file
            .read_variable_unpacked_masked("arbitrary_weather_731")
            .unwrap();
        assert_eq!(
            data.iter().copied().collect::<Vec<_>>(),
            [297.75, 298.0, 300.25, 300.5, 322.75, 323.0, 325.25, 325.5]
        );
        assert_eq!(
            file.read_variable_unpacked_masked("row")
                .unwrap()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            [39.0, 38.0]
        );
        assert!(prepare(serde_json::from_value(r).unwrap())
            .unwrap_err()
            .to_string()
            .contains("fresh output"));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn bad_selected_word_hash_units_and_unused_seals_fail_before_publication() {
        let (root, r) = fixture("invalid");
        let mut changed = r.clone();
        changed["chunk_sha256"]["arbitrary_weather_731/0.0.0"] = json!("bad");
        assert!(prepare(serde_json::from_value(changed).unwrap())
            .unwrap_err()
            .to_string()
            .contains("SHA256 mismatch"));
        let mut changed = r.clone();
        changed["fields"][0]["units"] = json!("wrong");
        assert!(prepare(serde_json::from_value(changed).unwrap())
            .unwrap_err()
            .to_string()
            .contains("units differ"));
        let mut changed = r.clone();
        changed["chunk_sha256"]["arbitrary_weather_731/9.9.9"] = json!("unused");
        assert!(prepare(serde_json::from_value(changed).unwrap())
            .unwrap_err()
            .to_string()
            .contains("unused chunk"));
        let raw: Vec<u8> = [-32767_i16; 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let compressed = zstd::stream::encode_all(raw.as_slice(), 3).unwrap();
        fs::write(root.join("arbitrary_weather_731/0.0.0"), &compressed).unwrap();
        let mut changed = r;
        changed["chunk_sha256"]["arbitrary_weather_731/0.0.0"] = json!(sha256(&compressed));
        assert!(prepare(serde_json::from_value(changed).unwrap())
            .unwrap_err()
            .to_string()
            .contains("missing selected"));
        assert!(!root.join("output").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
