//! The hail observation and verification door. No forecast physics lives here.
use rw_obs::{err, hail};
use serde_json::json;
use std::{path::Path, process::ExitCode};
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let first = args.first().map(String::as_str).unwrap_or("--help");
    let usage = "rw_hail spc --day YYYYMMDD --out REPORT.json [--file SPC.csv]\nrw_hail verify --request REQUEST.json --out RECEIPT.json\nrw_hail --list-products | --abi | --version";
    let allowed: &[&str] = match first {
        "spc" => &["--day", "--out", "--file"],
        "verify" => &["--request", "--out"],
        _ => &[],
    };
    let mut seen = std::collections::BTreeSet::new();
    if args.len() > 1 {
        for pair in args[1..].chunks(2) {
            if pair.len() != 2
                || !allowed.contains(&pair[0].as_str())
                || !seen.insert(pair[0].as_str())
                || pair[1].starts_with("--")
            {
                return Err(err("unknown, repeated, or incomplete hail option"));
            }
        }
    }
    let option = |name: &str| -> Option<&str> {
        args.windows(2)
            .find(|w| w[0] == name)
            .map(|w| w[1].as_str())
    };
    let required = |name: &str| -> Result<&str, Box<dyn std::error::Error>> {
        option(name).ok_or_else(|| err(format!("{name} is required")))
    };
    let value = match first {
        "--abi" => {
            println!("{}", hail::ABI);
            return Ok(());
        }
        "--version" => {
            println!(
                "rw_hail {} {}",
                env!("CARGO_PKG_VERSION"),
                GPUWM_BRIDGE_SOURCE_REV_STAMP
            );
            return Ok(());
        }
        "--help" | "-h" => {
            println!("{usage}");
            return Ok(());
        }
        "--list-products" => {
            json!({"schema":"gpuwm-obs.hail-products.v1","products":["SPC hail reports","MRMS MESH_Max_30min"],"size_bin_lower_bounds_mm":hail::BINS_MM})
        }
        "spc" => hail::spc(
            required("--day")?,
            option("--file").map(Path::new),
            Path::new(required("--out")?),
        )?,
        "verify" => {
            let request: hail::Request =
                serde_json::from_slice(&std::fs::read(required("--request")?)?)?;
            let record = hail::verify(&request)?;
            std::fs::write(required("--out")?, serde_json::to_vec_pretty(&record)?)?;
            record
        }
        _ => return Err(err(format!("unknown subcommand {first:?}\n\n{usage}"))),
    };
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rw_hail: {e}");
            ExitCode::from(2)
        }
    }
}
