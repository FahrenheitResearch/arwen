use rw_nextgen_output::{Result, capture_csv, write_plan};
use std::path::Path;
fn run() -> Result<serde_json::Value> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("remap-ids") if args.len()==4 => {
            rw_nextgen_output::series::remap_ids(Path::new(&args[1]),
                &serde_json::from_slice(&std::fs::read(&args[2])?)?,Path::new(&args[3]))
        }
        Some("pack-history") if args.len() == 4 => {
            let request: serde_json::Value=serde_json::from_slice(&std::fs::read(&args[1])?)?;
            rw_nextgen_output::series::pack_history(serde_json::from_value(request["plan"].clone())?,
                &serde_json::from_slice(&std::fs::read(&args[2])?)?,Path::new(&args[3]))
        }
        Some("compare-series") if args.len() == 6 => {
            let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[1])?)?;
            let candidate: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[3])?)?;
            rw_nextgen_output::series::compare(&serde_json::from_value(expected["plan"].clone())?,
                &std::fs::read(&args[2])?,&serde_json::from_value(candidate["plan"].clone())?,
                &std::fs::read(&args[4])?,&args[5].split(',').collect::<Vec<_>>())
        }
        Some("compare-parquet") if args.len() == 3 => {
            rw_nextgen_output::parquet::compare(Path::new(&args[1]),Path::new(&args[2]))
        }
        Some("capture-troute-netcdf") if args.len() == 3 => {
            rw_nextgen_output::series::capture_native(Path::new(&args[1]),Path::new(&args[2]))
        }
        Some("compare-netcdf") if args.len() == 3 => {
            let (left,bytes) = rw_nextgen_output::netcdf::read(Path::new(&args[1]))?;
            let (right,other) = rw_nextgen_output::netcdf::read(Path::new(&args[2]))?;
            if left != right || bytes != other {
                return Err("ordered typed headers or variable bytes differ".into());
            }
            Ok(serde_json::json!({"operation":"compare-netcdf","header":"ordered names/types/shapes/typed attributes identical",
                "variable_bytes":bytes.len(),"different_variable_bytes":0,"variables":left.variables.len(),
                "whole_file_identical":std::fs::read(&args[1])?==std::fs::read(&args[2])?}))
        }
        Some("capture-native-nexus") if args.len() == 3 => {
            rw_nextgen_output::capture_native_nexus(Path::new(&args[1]), Path::new(&args[2]))
        }
        Some("replay-netcdf") if args.len() == 3 => {
            let started = std::time::Instant::now();
            let (schema, payload) = rw_nextgen_output::netcdf::read(Path::new(&args[1]))?;
            let write = rw_nextgen_output::netcdf::write(Path::new(&args[2]), &schema, &payload)?;
            let (after, bytes) = rw_nextgen_output::netcdf::read(Path::new(&args[2]))?;
            if after != schema || bytes != payload {
                return Err("ordered typed headers or variable bytes differ".into());
            }
            Ok(
                serde_json::json!({"operation":"replay-netcdf","header":"ordered names/types/shapes/typed attributes identical", "variables":schema.variables.len(),"variable_bytes":payload.len(),"different_variable_bytes":0,"whole_file_identical":std::fs::read(&args[1])?==std::fs::read(&args[2])?,"write":write,"total_ms":started.elapsed().as_secs_f64()*1000.}),
            )
        }
        Some("request") if args.len() == 4 => {
            let request = serde_json::from_slice(&std::fs::read(&args[1])?)?;
            let (receipt, payload) =
                rw_nextgen_output::capi::run(request, &std::fs::read(&args[2])?)?;
            std::fs::write(&args[3], payload)?;
            Ok(receipt)
        }
        Some("nexus") if args.len() == 2 => write_plan(Path::new(&args[1])),
        Some("capture-nexus-csv") if args.len() == 3 => {
            capture_csv(Path::new(&args[1]), Path::new(&args[2]))?;
            Ok(
                serde_json::json!({"scope":"upstream rounded CSV formatting fixture; not native module replay"}),
            )
        }
        _ => Err(
            "usage: rw_nextgen_output nexus PLAN.json | capture-native-nexus INPUT.raw NEW_DIRECTORY | capture-nexus-csv INPUT.csv NEW_DIRECTORY | replay-netcdf INPUT.nc NEW_OUTPUT.nc | request REQUEST.json INPUT.raw OUTPUT.raw"
                .into(),
        ),
    }
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(value) => {
            println!("{}", serde_json::to_string_pretty(&value).unwrap());
            if value.get("matches").and_then(serde_json::Value::as_bool) == Some(false) {
                std::process::ExitCode::FAILURE
            } else {
                std::process::ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("rw_nextgen_output: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
