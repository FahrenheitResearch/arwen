//! Precipitation observation decoding, exact interval accumulation and receipts.
use rw_obs::{err, precipitation as p, precipitation_hdf as h};
use std::error::Error;
use std::path::PathBuf;

const ABI: &str="gpuwm-obs.precipitation.v1\tgpuwm-obs.obs-grid.v1\tprecipitation_accumulation\tmm\taccumulation_start\tvalid_time\tgeometry_sha256";
const USAGE: &str="rw_precip decode --file FILE --product NAME --geometry GEO.obspack --out FIELD.obspack [--bbox W,S,E,N] [--valid-time TIME]\nrw_precip decode-hdf --file FILE --product NAME --geometry GEO.obspack --out FIELD.obspack [--bbox W,S,E,N] [--valid-time TIME]\nrw_precip inspect-hdf --file FILE --product NAME\nrw_precip accumulate --input FIELD.obspack [--input ...] --start TIME --end TIME --out TOTAL.obspack\nrw_precip verify --file FIELD.obspack\nrw_precip --abi\n";
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn run() -> Result<(), Box<dyn Error>> {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        print!("{USAGE}");
        return Ok(());
    };
    if command == "--abi" {
        println!("{ABI}");
        return Ok(());
    }
    if command == "--help" {
        print!("{USAGE}");
        return Ok(());
    }
    let mut options = std::collections::BTreeMap::new();
    let mut inputs = Vec::new();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| err(format!("{flag} requires a value")))?;
        if flag == "--input" {
            inputs.push(PathBuf::from(value));
            continue;
        }
        if ![
            "--file",
            "--product",
            "--geometry",
            "--out",
            "--bbox",
            "--start",
            "--end",
            "--valid-time",
        ]
        .contains(&flag.as_str())
        {
            return Err(err(format!("unknown flag {flag}")));
        }
        if options.insert(flag.clone(), value).is_some() {
            return Err(err(format!("repeated flag {flag}")));
        }
    }
    let required = |flag: &str| {
        options
            .get(flag)
            .ok_or_else(|| err(format!("{flag} is required")))
    };
    if command == "inspect-hdf" {
        println!(
            "{}",
            serde_json::to_string_pretty(&h::inspect(
                &PathBuf::from(required("--file")?),
                &h::product(required("--product")?)?
            )?)?
        );
        return Ok(());
    }
    let result = match command.as_str() {
        "decode" => p::decode(
            &PathBuf::from(required("--file")?),
            &p::product(required("--product")?)?,
            options.get("--bbox").map(|raw| p::bbox(raw)).transpose()?,
            options
                .get("--valid-time")
                .map(|raw| p::instant(raw))
                .transpose()?,
            &PathBuf::from(required("--geometry")?),
            &PathBuf::from(required("--out")?),
        )?,
        "decode-hdf" => h::decode(
            &PathBuf::from(required("--file")?),
            &h::product(required("--product")?)?,
            options.get("--bbox").map(|raw| p::bbox(raw)).transpose()?,
            options
                .get("--valid-time")
                .map(|raw| p::instant(raw))
                .transpose()?,
            &PathBuf::from(required("--geometry")?),
            &PathBuf::from(required("--out")?),
        )?,
        "accumulate" => p::accumulate(
            &inputs,
            p::instant(required("--start")?)?,
            p::instant(required("--end")?)?,
            &PathBuf::from(required("--out")?),
        )?,
        "verify" => p::verify(&PathBuf::from(required("--file")?))?,
        _ => return Err(err(format!("unknown command {command}\n{USAGE}"))),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"schema":"gpuwm-obs.precipitation.v1","status":"PASS","command":command,"field":result})
        )?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rw_precip: {error}");
        std::process::exit(2);
    }
}
