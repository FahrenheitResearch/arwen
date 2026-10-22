//! Whole-run restart proof; compare every daily field and carried cell in Rust.
use crate::{PRODUCTS, Result, State, hash, publish};
use serde::Deserialize;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    reference_state: PathBuf,
    chained_state: PathBuf,
    reference_output: PathBuf,
    chained_outputs: Vec<PathBuf>,
    receipt: PathBuf,
}
pub fn run(path: &Path) -> Result<()> {
    let start = Instant::now();
    let r: Request = serde_json::from_slice(&fs::read(path)?)?;
    let a: State = serde_json::from_slice(&fs::read(&r.reference_state)?)?;
    let b: State = serde_json::from_slice(&fs::read(&r.chained_state)?)?;
    let sa = serde_json::to_vec(&a.cells)?;
    let sb = serde_json::to_vec(&b.cells)?;
    if sa != sb
        || a.last_hour != b.last_hour
        || a.completed_days != b.completed_days
        || a.latitude != b.latitude
        || a.longitude != b.longitude
        || a.nfdrs != b.nfdrs
        || a.annual_rain_mm != b.annual_rain_mm
    {
        return Err("chained carried state differs from monolithic state".into());
    }
    let full = netcrust::open(&r.reference_output)?;
    let mut fields = vec!["time"];
    fields.extend(PRODUCTS);
    let mut checked = 0usize;
    for variable in fields {
        let all = full.read_f64(variable)?;
        let mut offset = 0;
        for p in &r.chained_outputs {
            let n = netcrust::open(p)?;
            let values = n.read_f64(variable)?;
            let end = offset + values.len();
            if end > all.len()
                || all[offset..end]
                    .iter()
                    .zip(&values)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(
                    format!("chained output differs in {variable}: {}", p.display()).into(),
                );
            }
            offset = end;
        }
        if offset != all.len() {
            return Err(format!("chained output length differs in {variable}").into());
        }
        checked += all.len();
    }
    publish(
        &r.receipt,
        &serde_json::to_vec_pretty(
            &json!({"schema":"arwen.fire-chain-comparison.v1","reference_state":r.reference_state,"reference_state_sha256":hash(&r.reference_state)?,"chained_state":r.chained_state,"chained_state_sha256":hash(&r.chained_state)?,"reference_output":r.reference_output,"reference_output_sha256":hash(&r.reference_output)?,"chained_outputs":r.chained_outputs,"exact_carried_state":true,"bitwise_equal_values":checked,"completed_days":a.completed_days,"grid":[a.latitude.len(),a.longitude.len()],"wall_seconds":start.elapsed().as_secs_f64()}),
        )?,
    )?;
    Ok(())
}
