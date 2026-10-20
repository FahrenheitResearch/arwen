//! Executable driver handoff: real manifest/payloads, no inferred initialization.
use nextgen_hydrofabric::{bundle, manifest::PreparedManifest, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{ffi::CString, path::Path, time::Instant};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or("read_manifest MANIFEST.json")?;
    let start = Instant::now();
    let prepared = PreparedManifest::open(Path::new(&path))?;
    let open_ms = start.elapsed().as_secs_f64() * 1000.;
    let network: Value = serde_json::from_slice(&prepared.read_auxiliary("network.json")?)?;
    if network["node_ids"]
        .as_array()
        .ok_or("network node_ids absent")?
        .len()
        != prepared.document.node_count
        || network["catchment_ids"]
            .as_array()
            .ok_or("network catchment_ids absent")?
            .len()
            != prepared.document.catchment_count
    {
        return Err("network roster count differs from manifest".into());
    }
    let read = Instant::now();
    let totals = prepared
        .document
        .files
        .par_iter()
        .map(|(name, field)| -> Result<(usize, usize)> {
            // This Vec stands in for driver-owned pinned transport. No cast,
            // numeric conversion, state initialization or GPU binding occurs here.
            let mut target = vec![0u8; field.bytes];
            prepared.read_field_into(name, &mut target)?;
            // Verify exact compatibility with the pre-existing raw Rust C ABI.
            let raw_path = CString::new(
                Path::new(&path)
                    .parent()
                    .ok_or("manifest directory absent")?
                    .join(&field.path)
                    .to_str()
                    .ok_or("non-UTF8 fixture path")?,
            )?;
            let hash = CString::new(field.sha256.as_str())?;
            let mut legacy = vec![0u8; field.bytes];
            let status = unsafe {
                bundle::arwen_hydrofabric_read_into(
                    raw_path.as_ptr(),
                    hash.as_ptr(),
                    legacy.as_mut_ptr(),
                    legacy.len(),
                )
            };
            if status != 0 {
                return Err(format!("legacy transport failed: {name} status {status}").into());
            }
            let different = target.iter().zip(&legacy).filter(|(a, b)| a != b).count();
            Ok((field.bytes, different))
        })
        .collect::<Result<Vec<_>>>()?;
    let different: usize = totals.iter().map(|x| x.1).sum();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "source":nextgen_hydrofabric::SOURCE_REVISION, "schema":prepared.document.schema,
            "scope":"actual prepared transport versus existing raw Rust reader, not original GIS/initialization/coupled proof",
            "node_count":prepared.document.node_count,"catchment_count":prepared.document.catchment_count,
            "fields":totals.len(),"payload_bytes":totals.iter().map(|x|x.0).sum::<usize>(),
            "different_bytes":different,"auxiliary_network_sha_verified":true,
            "open_ms":open_ms,"dual_read_hash_compare_ms":read.elapsed().as_secs_f64()*1000.,
            "total_ms":start.elapsed().as_secs_f64()*1000.,"rayon_workers":rayon::current_num_threads(),"pass":different==0
        }))?
    );
    if different != 0 {
        return Err("prepared transport bytes differ".into());
    }
    Ok(())
}
