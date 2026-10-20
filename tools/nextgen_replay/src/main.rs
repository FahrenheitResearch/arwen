// SPDX-License-Identifier: Apache-2.0
use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    process::ExitCode,
};

fn execute() -> Result<u8, String> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: nextgen-replay REQUEST.json NEW_VERDICT.json".into());
    }
    let bytes = fs::read(&args[1]).map_err(|e| e.to_string())?;
    let verdict = nextgen_replay::compare(&bytes)?;
    let output = Path::new(&args[2]);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    file.write_all(&serde_json::to_vec_pretty(&verdict).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    println!(
        "{}: {} / {} different words; {}",
        verdict.unit, verdict.different_words, verdict.compared_words, verdict.parallelism
    );
    Ok(if verdict.bitwise_match { 0 } else { 1 })
}
fn main() -> ExitCode {
    match execute() {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("nextgen-replay: {e}");
            ExitCode::from(2)
        }
    }
}
