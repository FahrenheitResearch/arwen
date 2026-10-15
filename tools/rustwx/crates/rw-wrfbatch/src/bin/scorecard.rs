//! Campaign orchestration plus native aggregation. Field processing stays in rw_verify.
use rw_wrfbatch::{
    scorecard as s,
    verification_io::{read_json, source_identity},
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
use crate::cli_contracts::SCORECARD as ABI;
const USAGE: &str = "rw_scorecard --manifest MANIFEST.json --out DIR [--verify-binary PATH]\nReads native requests, native receipts or digest-bound recorded paired CSV.\nWrites scorecard.csv, scorecard.json, scorecard.md and native Rust comparison sheets.\nExit 3 means a required comparison is missing; exit 1 means invalid input or a failed native step.";

fn resolve_request_paths(request: &mut Value, root: &Path) {
    let make_path = |v: &mut Value| {
        if let Some(p) = v.as_str() {
            *v = json!(s::resolve(root, Path::new(p)));
        }
    };
    for key in ["stations_path", "station_table_path"] {
        if let Some(v) = request.get_mut(key) {
            make_path(v);
        }
    }
    if let Some(arms) = request["arms"].as_array_mut() {
        for arm in arms {
            for key in ["path", "grid_path", "previous_path", "points_path"] {
                if let Some(v) = arm.get_mut(key) {
                    make_path(v);
                }
            }
        }
    }
    if let Some(rows) = request.get_mut("radar").and_then(Value::as_array_mut) {
        for row in rows {
            for key in ["path", "grid_path"] {
                if let Some(v) = row.get_mut(key) {
                    make_path(v);
                }
            }
        }
    }
}

fn run() -> Result<bool, String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|s| s == "--abi") {
        println!("{ABI}");
        return Ok(true);
    }
    if args.is_empty() || args.iter().any(|s| s == "--help" || s == "-h") {
        println!("{USAGE}");
        return Ok(true);
    }
    let mut manifest_path = None;
    let mut out = None;
    let mut verify = std::env::current_exe()
        .map_err(|e| e.to_string())?
        .with_file_name(if cfg!(windows) {
            "rw_verify.exe"
        } else {
            "rw_verify"
        });
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        i += 1;
        let value = args.get(i).ok_or_else(|| format!("{flag} needs a value"))?;
        i += 1;
        match flag.as_str() {
            "--manifest" => manifest_path = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--verify-binary" => verify = PathBuf::from(value),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    let path = manifest_path
        .ok_or("--manifest is required")?
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let mut out = out.ok_or("--out is required")?;
    if !out.is_absolute() {
        out = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(out);
    }
    let manifest: s::Manifest =
        serde_json::from_value(read_json(&path)?).map_err(|e| e.to_string())?;
    manifest.validate()?;
    if manifest.cases.iter().any(|c| !c.requests.is_empty()) {
        let probe = Command::new(&verify)
            .arg("--abi")
            .output()
            .map_err(|e| e.to_string())?;
        if !probe.status.success()
            || !String::from_utf8_lossy(&probe.stdout).contains("case-scorecard-inputs-v1")
        {
            return Err("native verifier lacks case-scorecard-inputs-v1; the old reader cannot honor station quantity metadata or supply continuous grid errors. Rebuild rw_verify from the same source as rw_scorecard".into());
        }
    }
    let root = path.parent().unwrap();
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut scores = Vec::new();
    let mut identities = Vec::new();
    for case in &manifest.cases {
        if case.pairs.is_some() {
            let (rows, identity) = s::csv_scores(case, &manifest, root)?;
            scores.extend(rows);
            identities.push(json!({"case":case.id,"mode":"recorded-pairs","input":identity}));
            continue;
        }
        let mut receipts = case
            .receipts
            .iter()
            .map(|p| s::resolve(root, p))
            .collect::<Vec<_>>();
        for (index, p) in case.requests.iter().enumerate() {
            let request_path = s::resolve(root, p)
                .canonicalize()
                .map_err(|e| e.to_string())?;
            let mut request = read_json(&request_path)?;
            resolve_request_paths(&mut request, request_path.parent().unwrap());
            let labels = request["arms"]
                .as_array()
                .ok_or("request lacks arms")?
                .iter()
                .filter_map(|r| r["label"].as_str())
                .collect::<Vec<_>>();
            if !labels.contains(&manifest.candidate.as_str())
                || !labels.contains(&manifest.reference.as_str())
                || labels.len() != 2
            {
                return Err(format!(
                    "{} must contain exactly the manifest's two arms to prevent a third arm changing common support",
                    request_path.display()
                ));
            }
            let folder = out
                .join(&case.id)
                .join("native")
                .join(format!("{index:04}"));
            std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
            request["out_root"] = json!(out.join(&case.id));
            let staged = folder.join("request.json");
            std::fs::write(
                &staged,
                serde_json::to_vec_pretty(&request).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let receipt = folder.join("receipt.json");
            let image = folder.join("scorecard.png");
            let command = vec![
                "--request".to_string(),
                staged.display().to_string(),
                "--json".into(),
                receipt.display().to_string(),
                "--image".into(),
                image.display().to_string(),
            ];
            std::fs::write(
                folder.join("command.json"),
                serde_json::to_vec_pretty(
                    &json!({"binary":source_identity(&verify)?,"argv":command}),
                )
                .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let result = Command::new(&verify)
                .args(&command)
                .env("CUDA_VISIBLE_DEVICES", "")
                .output()
                .map_err(|e| format!("launch native verifier: {e}"))?;
            std::fs::write(folder.join("stderr.log"), &result.stderr).map_err(|e| e.to_string())?;
            if !result.status.success() {
                return Err(format!(
                    "case {} native request {index} failed: {}; log {}",
                    case.id,
                    String::from_utf8_lossy(&result.stderr),
                    folder.join("stderr.log").display()
                ));
            }
            let record = read_json(&receipt)?;
            if record["request_sha256"] != source_identity(&staged)?["sha256"] {
                return Err("native receipt does not bind the submitted request".into());
            }
            receipts.push(receipt);
        }
        let (rows, inputs) = s::native_scores(case, &manifest, &receipts)?;
        scores.extend(rows);
        identities.push(json!({"case":case.id,"mode":"native-verification","inputs":inputs}));
    }
    let document = s::aggregate(&manifest, &scores, identities, source_identity(&path)?);
    s::write_outputs(&out, &document)?;
    println!(
        "{}",
        json!({"schema":s::RECEIPT,"out":out,"coverage_status":document["coverage_status"],"missing_required":document["missing_required"]})
    );
    Ok(document["missing_required"].as_array().unwrap().is_empty())
}
pub(crate) fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(3),
        Err(e) => {
            eprintln!("rw_scorecard: {e}");
            ExitCode::FAILURE
        }
    }
}
