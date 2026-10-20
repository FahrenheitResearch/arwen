//! Read-only audit of the existing prepared graph contract, not a mesh adapter.
use nextgen_hydrofabric::{bundle::digest, manifest::PreparedManifest, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    path::Path,
    time::Instant,
};

fn ranks(order: &[i32], offsets: &[i32], n: usize) -> Result<Vec<usize>> {
    if offsets.first() != Some(&0)
        || offsets.last() != Some(&(n as i32))
        || offsets.windows(2).any(|w| w[0] >= w[1])
        || order.len() != n
    {
        return Err("bad level offsets/roster length".into());
    }
    let mut rank = vec![usize::MAX; n];
    for (level, w) in offsets.windows(2).enumerate() {
        let row = &order[w[0] as usize..w[1] as usize];
        if row.windows(2).any(|w| w[0] >= w[1]) {
            return Err("level members must ascend in node-axis indices".into());
        }
        for &node in row {
            let slot = rank
                .get_mut(usize::try_from(node)?)
                .ok_or("node outside axis")?;
            if *slot != usize::MAX {
                return Err("duplicate level member".into());
            }
            *slot = level;
        }
    }
    if rank.contains(&usize::MAX) {
        return Err("level roster omits a node".into());
    }
    Ok(rank)
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or("network_contract MANIFEST.json")?;
    let start = Instant::now();
    let m = PreparedManifest::open(Path::new(&path))?;
    let n = m.document.node_count;
    if n == 0 || n > i32::MAX as usize {
        return Err("invalid node axis".into());
    }
    let roster: Value = serde_json::from_slice(&m.read_auxiliary("network.json")?)?;
    let names = [
        "node_ids",
        "downstream",
        "upstream_offsets",
        "upstream_indices",
        "ordered_nodes",
        "level_offsets",
        "reverse_ordered_nodes",
        "reverse_level_offsets",
    ];
    let raw: BTreeMap<_, _> = names
        .par_iter()
        .map(|&name| -> Result<_> {
            let f = m.document.files.get(name).ok_or("missing graph field")?;
            if f.dtype != if name == "node_ids" { "<i8" } else { "<i4" } || f.shape.len() != 1 {
                return Err(
                    "graph fields must retain native integer widths and vector shape".into(),
                );
            }
            let mut b = vec![0; f.bytes];
            m.read_field_into(name, &mut b)?;
            Ok((name, b))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .collect();
    let read_hash_ms = start.elapsed().as_secs_f64() * 1000.;
    let decode_start = Instant::now();
    let ids: Vec<_> = raw["node_ids"]
        .par_chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let integer: BTreeMap<_, Vec<i32>> = names[1..]
        .par_iter()
        .map(|&name| {
            (
                name,
                raw[name]
                    .par_chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            )
        })
        .collect();
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.;
    let validation_start = Instant::now();
    let string_ids = roster["node_ids"]
        .as_array()
        .ok_or("missing string IDs")?
        .iter()
        .map(|v| v.as_str().ok_or("non-string roster ID"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if ids.len() != n
        || ids.iter().collect::<HashSet<_>>().len() != n
        || string_ids.len() != n
        || string_ids.iter().collect::<HashSet<_>>().len() != n
        || serde_json::to_value(&ids)? != roster["numeric_node_ids"]
    {
        return Err("raw IDs and sealed roster disagree or are not unique".into());
    }
    for (&name, values) in &integer {
        if serde_json::to_value(values)? != roster[name] {
            return Err(format!("raw {name} differs from sealed graph JSON").into());
        }
    }
    let ds = &integer["downstream"];
    let up = &integer["upstream_indices"];
    let uo = &integer["upstream_offsets"];
    if ds.len() != n
        || uo.len() != n + 1
        || uo[0] != 0
        || uo[n] != i32::try_from(up.len())?
        || uo.windows(2).any(|w| w[0] > w[1])
    {
        return Err("bad graph CSR shape".into());
    }
    if ds.iter().any(|&d| d < -1 || d >= n as i32) || up.iter().any(|&u| u < 0 || u >= n as i32) {
        return Err("graph index outside axis".into());
    }
    let forward = ranks(&integer["ordered_nodes"], &integer["level_offsets"], n)?;
    let reverse = ranks(
        &integer["reverse_ordered_nodes"],
        &integer["reverse_level_offsets"],
        n,
    )?;
    if integer["level_offsets"].len() - 1 != m.document.level_count {
        return Err("forward level count differs from manifest".into());
    }
    let valid = (0..n).into_par_iter().all(|node| {
        let incoming = &up[uo[node] as usize..uo[node + 1] as usize];
        let unique = incoming.iter().collect::<HashSet<_>>().len() == incoming.len();
        let expected_forward = incoming
            .iter()
            .map(|&u| forward[u as usize] + 1)
            .max()
            .unwrap_or(0);
        let expected_reverse = if ds[node] < 0 {
            0
        } else {
            reverse[ds[node] as usize] + 1
        };
        unique
            && incoming.iter().all(|&u| ds[u as usize] == node as i32)
            && forward[node] == expected_forward
            && reverse[node] == expected_reverse
    });
    if !valid || ds.iter().filter(|&&d| d >= 0).count() != up.len() {
        return Err("graph edge or forward/reverse dependency violation".into());
    }
    let validation_ms = validation_start.elapsed().as_secs_f64() * 1000.;
    // Field intentionally lacks Serialize in the transport API; preserve the
    // original descriptors as metadata without changing that production API.
    let manifest_json: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let descriptors: BTreeMap<_, _> = names
        .iter()
        .map(|&name| (name, manifest_json["files"][name].clone()))
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "source":nextgen_hydrofabric::SOURCE_REVISION,"manifest":path,
            "manifest_sha256":digest(Path::new(&path))?,"network_sha256":m.document.auxiliary["network.json"].sha256,
            "node_count":n,"catchment_count":m.document.catchment_count,"edges":up.len(),
            "forward_levels":integer["level_offsets"].len()-1,"reverse_levels":integer["reverse_level_offsets"].len()-1,
            "exterior_nodes":ds.iter().filter(|&&d|d<0).count(),"node_order":m.document.node_order,
            "upstream_order":m.document.upstream_order,"files":descriptors,
            "checked_bytes":raw.values().map(Vec::len).sum::<usize>(),"pass":true,
            "checks":"raw field hashes and graph JSON equality; unique stable IDs; CSR reciprocal edges; both complete canonical topological level sequences",
            "not_provided":"diffusive ncomp/frnw/mainstem roster/solver-reach-to-flowpath mapping/f64 F-order mesh or adaptive state",
            "read_hash_ms":read_hash_ms,"decode_ms":decode_ms,"validation_ms":validation_ms,
            "rayon_workers":rayon::current_num_threads(),"total_ms":start.elapsed().as_secs_f64()*1000.
        }))?
    );
    Ok(())
}
