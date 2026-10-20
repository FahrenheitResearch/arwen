// SPDX-License-Identifier: Apache-2.0
// Name matrix rows for the shared comparator. No numerical comparison here.
use nextgen_replay::{file_sha256, sealed_bytes, Fallible, Seal};
use serde_json::{json, Value};
use std::{fs, io::Write, path::Path, time::Instant};
fn run() -> Fallible<()> {
    let started = Instant::now();
    let args: Vec<_> = std::env::args().collect();
    if !matches!(args.len(), 4 | 5) {
        return Err(
            "usage: split_fields MANIFEST COMPARE_REQUEST NEW_OUTPUT [SCALAR_FIELD,...]".into(),
        );
    }
    let selected: Option<Vec<_>> = args.get(4).map(|s| s.split(',').collect());
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let mut request: Value =
        serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let output = Path::new(&args[3]);
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let output = output.canonicalize().map_err(|e| e.to_string())?;
    let mut fields = vec![];
    for field in request["fields"].as_array().ok_or("missing fields")? {
        let name = field["name"].as_str().ok_or("missing name")?;
        let (key, width) = match name {
            "fields" => ("field_names", 8),
            "options" => ("option_names", 4),
            _ => {
                if selected.is_none() {
                    fields.push(field.clone());
                }
                continue;
            }
        };
        let seal: Seal =
            serde_json::from_value(field["expected"].clone()).map_err(|e| e.to_string())?;
        let expected = sealed_bytes(&seal)?;
        let actual = fs::read(field["actual"].as_str().ok_or("missing actual")?)
            .map_err(|e| e.to_string())?;
        let n = manifest["catchments"]
            .as_array()
            .ok_or("missing catchments")?
            .len();
        let names = manifest[key].as_array().ok_or("missing field names")?;
        if expected.len() != names.len() * n * width || actual.len() != expected.len() {
            return Err("packed matrix size differs".into());
        }
        for (row, native) in names.iter().enumerate() {
            let native = native.as_str().ok_or("invalid name")?;
            if selected
                .as_ref()
                .is_some_and(|names| !names.contains(&native))
            {
                continue;
            }
            let groups = if selected.is_some() { n } else { 1 };
            let count = if selected.is_some() { 1 } else { n };
            for group in 0..groups {
                let mut entry = field.clone();
                entry["name"] = json!(if selected.is_some() {
                    format!("{native}/model-{group}")
                } else {
                    native.into()
                });
                entry["shape"] = json!([count]);
                for (label, bytes) in [("expected", &expected), ("actual", &actual)] {
                    let path = output.join(format!("{name}-{row}-{group}-{label}.bin"));
                    fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&path)
                        .map_err(|e| e.to_string())?
                        .write_all(
                            &bytes[(row * n + group * count) * width
                                ..(row * n + (group + 1) * count) * width],
                        )
                        .map_err(|e| e.to_string())?;
                    entry[label] = if label == "expected" {
                        json!({"path":path,"sha256":file_sha256(&path)?})
                    } else {
                        json!(path)
                    };
                }
                fields.push(entry);
            }
        }
    }
    if selected.is_some() {
        request["case"] = json!(format!(
            "{}-selected-word-diagnostic",
            request["case"].as_str().ok_or("missing case")?
        ));
    }
    request["fields"] = json!(fields);
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("compare.json"))
        .map_err(|e| e.to_string())?
        .write_all(&serde_json::to_vec_pretty(&request).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"split_wall_ms":started.elapsed().as_secs_f64()*1000.,"scope":"raw row extraction; shared comparator owns all verdicts"})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(2)
    }
}
