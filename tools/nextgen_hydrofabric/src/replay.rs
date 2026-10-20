use crate::{bundle::digest, Result};
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};
/// Exact native-parser replay: reference bytes must originate in ngen itself.
pub fn compare(manifest: &Path, reference: &Path) -> Result<Value> {
    let timer = Instant::now();
    let m: Value = serde_json::from_slice(&fs::read(manifest)?)?;
    let actual = manifest
        .parent()
        .ok_or("manifest directory missing")?
        .join("parsed_network.tsv");
    let a = fs::read(&actual)?;
    let b = fs::read(reference)?;
    let different = a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
    Ok(
        json!({"mode":"strict_explicit_native_ID_edge_byte_replay","actual":actual,"reference":reference,"actual_sha256":digest(&actual)?,"reference_sha256":digest(reference)?,"actual_bytes":a.len(),"reference_bytes":b.len(),"differing_bytes":different,"actual_records":a.iter().filter(|&&b|b==b'\n').count().saturating_sub(1),"native_upstream_reduction_order_proven":false,"native_upstream_roster_supplied":m["upstream_order"]=="explicit_native_roster","pass":different==0,"wall_ms":timer.elapsed().as_secs_f64()*1000.}),
    )
}
