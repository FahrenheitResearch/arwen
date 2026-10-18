//! Stored-word transport for native column replay and undefined-read witnesses.
//! No CF decoding, floating promotion, or JSON floating numbers are used.
use netcrust::{DataType, File};
use serde_json::{json, Map, Value};
use std::{env, fs, path::Path};

#[derive(serde::Deserialize)]
struct Request {
    path: String,
    fields: Vec<String>,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: rw_mpas_words INPUT.nc OUTPUT.json FIELD... | --request REQUEST.json OUTPUT.json".into());
    }
    let (requests, output) = if args[0] == "--request" {
        (serde_json::from_slice::<Vec<Request>>(&fs::read(&args[1])?)?, &args[2])
    } else {
        (vec![Request { path: args[0].clone(), fields: args[2..].to_vec() }], &args[1])
    };
    let mut fields = Map::<String, Value>::new();
    let mut sources = Vec::new();
    for request in requests {
      let file = File::open(Path::new(&request.path))?;
      sources.push(request.path);
      for name in &request.fields {
        if fields.contains_key(name) {
            return Err(format!("duplicate requested field {name}: ambiguous source").into());
        }
        let variable = file.variable(name).ok_or_else(|| format!("missing field {name}"))?;
        let record = match variable.dtype() {
            DataType::F32 => {
                let array = file.read_array::<f32>(name)?;
                json!({"shape": array.shape(), "type": "f32",
                    "words": array.iter().map(|x| x.to_bits()).collect::<Vec<_>>()})
            }
            DataType::F64 => {
                // MPAS RKIND pool reads convert stored mesh doubles to REAL(4).
                let array = file.read_array::<f64>(name)?;
                json!({"shape": array.shape(), "type": "f32",
                    "words": array.iter().map(|x| (*x as f32).to_bits()).collect::<Vec<_>>()})
            }
            DataType::I32 => {
                let array = file.read_array::<i32>(name)?;
                json!({"shape": array.shape(), "type": "i32",
                    "words": array.iter().copied().collect::<Vec<_>>()})
            }
            dtype => return Err(format!("{name}: unsupported stored word type {dtype:?}").into()),
        };
        fields.insert(name.clone(), record);
      }
    }
    fs::write(output, serde_json::to_vec(&json!({"sources": sources, "fields": fields}))?)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rw_mpas_words: {error}");
        std::process::exit(1);
    }
}
