//! Exact typed-word comparison for the six real-command aerosol init decks.
use netcrust::{DataType, File};
use serde_json::{json, Value};
use std::path::Path;

fn words(file: &File, name: &str, kind: &DataType) -> Result<Vec<u64>, String> {
    macro_rules! read {
        ($ty:ty, $convert:expr) => {
            file.read_array::<$ty>(name).map(|array| array.iter().copied().map($convert).collect())
                .map_err(|error| error.to_string())
        };
    }
    match kind {
        DataType::F32 => read!(f32, |v: f32| u64::from(v.to_bits())),
        DataType::F64 => read!(f64, |v: f64| v.to_bits()),
        DataType::I32 => read!(i32, |v: i32| u64::from(v as u32)),
        DataType::I16 => read!(i16, |v: i16| u64::from(v as u16)),
        DataType::I8 => read!(i8, |v: i8| u64::from(v as u8)),
        DataType::U8 => read!(u8, |v: u8| u64::from(v)),
        DataType::Char => {
            let variable = file.variable(name).unwrap();
            let width = variable.dimensions().last().map_or(1, |d| d.len());
            let strings = file.read_strings(name).map_err(|e| e.to_string())?;
            let mut result = Vec::new();
            for string in strings {
                if !string.is_ascii() || string.len() > width {
                    return Err("char field cannot be reconstructed without loss".into());
                }
                result.extend(string.bytes().map(u64::from));
                result.resize(result.len() + width - string.len(), 0);
            }
            Ok(result)
        }
        other => Err(format!("unhandled native field type {other:?}")),
    }
}

#[test]
fn compare_real_init_decks() {
    let Some(manifest) = std::env::var_os("MPAS_AEROSOL_DECK_MANIFEST") else { return; };
    let cases: Value = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    let mut rows = Vec::new();
    let mut all_identical = true;
    for case in cases.as_array().unwrap() {
        if !Path::new(case["candidate"].as_str().unwrap()).is_file() {
            rows.push(json!({"case": case["case"], "status": "candidate not emitted", "bitwise": false}));
            all_identical = false;
            continue;
        }
        let native = File::open(case["native"].as_str().unwrap()).unwrap();
        let candidate = File::open(case["candidate"].as_str().unwrap()).unwrap();
        let nv = native.variables().unwrap();
        let cv = candidate.variables().unwrap();
        let mut fields = Vec::new();
        let mut same = true;
        for field in &nv {
            let name = field.name();
            let matching = cv.iter().find(|v| v.name() == name);
            let shapes = |v: &netcrust::Variable| v.dimensions().iter()
                .map(|d| (d.name().to_string(), d.len())).collect::<Vec<_>>();
            if matching.is_none_or(|v| v.dtype() != field.dtype() || shapes(v) != shapes(field)) {
                fields.push(json!({"field": name, "status": "missing or schema differs"}));
                same = false;
                continue;
            }
            let (a, b) = (words(&native, name, field.dtype()), words(&candidate, name, field.dtype()));
            match (a, b) {
                (Ok(a), Ok(b)) if a.len() == b.len() => {
                    let different = a.iter().zip(&b).filter(|(a,b)| a != b).count();
                    let first = a.iter().zip(&b).position(|(a,b)| a != b);
                    same &= different == 0;
                    fields.push(json!({"field": name, "status": "compared", "words": a.len(),
                        "different_words": different, "first_difference": first,
                        "native_word": first.map(|i| format!("{:x}", a[i])),
                        "candidate_word": first.map(|i| format!("{:x}", b[i]))}));
                }
                (a, b) => {
                    same = false;
                    fields.push(json!({"field": name, "status": "decode or length failure",
                        "native_error": a.err(), "candidate_error": b.err()}));
                }
            }
        }
        let extra: Vec<_> = cv.iter().filter(|v| !nv.iter().any(|n| n.name() == v.name()))
            .map(|v| v.name().to_string()).collect();
        same &= extra.is_empty();
        all_identical &= same;
        rows.push(json!({"case": case["case"], "native_mpi_ranks": case["native_mpi_ranks"],
            "candidate_process_count": 1, "bitwise": same, "fields": fields, "extra_fields": extra}));
    }
    std::fs::write(std::env::var_os("MPAS_AEROSOL_DECK_COMPARISON").unwrap(),
        serde_json::to_vec_pretty(&json!({"all_fields_bitwise": all_identical, "cases": rows})).unwrap()).unwrap();
    assert!(all_identical, "init front-door comparison failed; see the complete word receipt");
}
