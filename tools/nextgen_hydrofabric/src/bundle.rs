use crate::{
    network::{self, Config},
    weights, Result,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    ffi::{c_char, CStr},
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
pub fn digest(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
struct Array {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
macro_rules! array {
    ($v:expr,$n:literal,$t:literal) => {
        Array {
            name: $n,
            dtype: $t,
            shape: vec![$v.len()],
            bytes: $v.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    };
}
pub fn prepare(config: &Path) -> Result<Value> {
    let timer = Instant::now();
    let c: Config = serde_json::from_slice(&fs::read(config)?)?;
    if c.data_license.trim().is_empty() {
        return Err("data_license must record actual dataset terms, separate from software".into());
    }
    if let Some(parent) = &c.parent_grid_contract {
        if parent["schema"] != "arwen.nextgen.parent-grid.v1"
            || parent["parent_identity"]["schema"] != "arwen.nextgen.parent-geometry.v1"
            || c.grid.is_none()
            || parent["grid"] != serde_json::to_value(c.grid.as_ref().unwrap())?
        {
            return Err("parent_grid_contract must bind the executed-parent identity to the exact configured grid".into());
        }
    }
    // The input digest (one pass over the GeoPackage) runs beside the parallel
    // table reads instead of after them.
    let (prepared, input_hash) = rayon::join(|| network::prepare(&c), || digest(Path::new(&c.geopackage)));
    let (prepared, input_hash) = (prepared?, input_hash?);
    let n = &prepared.network;
    let weight_timer = Instant::now();
    let w = c
        .grid
        .as_ref()
        .map(|g| weights::build(&prepared, g))
        .transpose()?;
    let weights_ms = weight_timer.elapsed().as_secs_f64() * 1000.;
    let raster_timer = Instant::now();
    let channel_raster = c
        .grid
        .as_ref()
        .map(|g| weights::rasterize(&prepared, g))
        .transpose()?;
    let raster_ms = raster_timer.elapsed().as_secs_f64() * 1000.;
    let mut identity = Sha256::new();
    identity.update(fs::read(config)?);
    identity.update(input_hash.as_bytes());
    let key = format!("{:x}", identity.finalize());
    let out = PathBuf::from(&c.output).join(&key[..20]);
    fs::create_dir_all(&out)?;
    // Request-owned directories keep repeated or changed ordinary requests safe.
    // A repeat recomputes the same owned arrays so measurements remain actual work.
    if out.join("manifest.json").exists() {
        let old: Value = serde_json::from_slice(&fs::read(out.join("manifest.json"))?)?;
        if old["request_sha256"].as_str() != Some(&key) {
            return Err("prepared directory belongs to a different request".into());
        }
    }
    let mut arrays = vec![
        array!(n.downstream, "downstream", "<i4"),
        array!(n.upstream_offsets, "upstream_offsets", "<i4"),
        array!(n.upstream_indices, "upstream_indices", "<i4"),
        array!(n.ordered_nodes, "ordered_nodes", "<i4"),
        array!(n.level_offsets, "level_offsets", "<i4"),
        array!(n.reverse_ordered_nodes, "reverse_ordered_nodes", "<i4"),
        array!(n.reverse_level_offsets, "reverse_level_offsets", "<i4"),
        array!(n.catchment_to_node, "catchment_to_node", "<i4"),
    ];
    if let Some(ids) = &n.numeric_node_ids {
        arrays.push(array!(ids, "node_ids", "<i8"))
    }
    if let Some(p) = &n.routing_parameters {
        arrays.push(Array {
            name: "routing_parameters",
            dtype: "<f4",
            shape: vec![p.len(), 8],
            bytes: p.iter().flatten().flat_map(|v| v.to_le_bytes()).collect(),
        })
    }
    if let Some(w) = &w {
        // Native Noah-OWP exchange ABI uses signed indices and f32 reductions;
        // prepare aliases in Rust once, keeping forcing's f64 geometry weights.
        let offsets = w
            .offsets
            .iter()
            .copied()
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let indices = w
            .indices
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let weights = w.weights.iter().map(|&x| x as f32).collect::<Vec<_>>();
        arrays.extend([
            array!(offsets, "exchange_offsets", "<i8"),
            array!(indices, "exchange_indices", "<i4"),
            array!(weights, "exchange_weights", "<f4"),
            array!(w.offsets, "weights_offsets", "<u8"),
            array!(w.indices, "weights_indices", "<u4"),
            array!(w.weights, "weights", "<f8"),
            array!(w.overlap_m2, "overlap_m2", "<f8"),
            array!(w.catchment_area_m2, "catchment_area_m2", "<f8"),
            array!(w.coverage_fraction, "coverage_fraction", "<f8"),
        ]);
    }
    if let Some(raster) = &channel_raster {
        arrays.push(Array {
            name: "channel_node",
            dtype: "<i4",
            shape: vec![
                c.grid.as_ref().unwrap().ny as usize,
                c.grid.as_ref().unwrap().nx as usize,
            ],
            bytes: raster
                .channel_node
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        });
        arrays.push(Array {
            name: "stream_order",
            dtype: "<i4",
            shape: vec![
                c.grid.as_ref().unwrap().ny as usize,
                c.grid.as_ref().unwrap().nx as usize,
            ],
            bytes: raster
                .stream_order
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        });
    }
    let write_timer = Instant::now();
    let files=arrays.par_iter().map(|a|->Result<(String,Value)>{let filename=format!("{}.bin",a.name);fs::write(out.join(&filename),&a.bytes)?;Ok((a.name.into(),json!({"path":filename,"dtype":a.dtype,"shape":a.shape,"bytes":a.bytes.len(),"sha256":format!("{:x}",Sha256::digest(&a.bytes))}))) }).collect::<Result<Vec<_>>>()?.into_iter().collect::<serde_json::Map<_,_>>();
    let mut gauges = std::collections::BTreeSet::new();
    if let Some(layer) = prepared.layers.get("network") {
        for row in &layer.rows {
            if let Some(station) = network::field(row, "hl_uri")
                .and_then(|s| s.strip_prefix("Gages-").map(str::to_owned))
            {
                let feature = network::id(row, "network")?;
                if let Some(node) = n.node_ids.iter().position(|s| s == &feature) {
                    gauges.insert((
                        station,
                        feature,
                        node,
                        network::field(row, "toid").unwrap_or_default(),
                    ));
                }
            }
        }
    }
    write_json(
        &out.join("gauges.json"),
        &json!({"source":"native network.hl_uri Gages station linked to network.id; duplicates removed, ambiguous mappings retained", "rows":gauges.into_iter().map(|(station,feature,node,nexus)|json!({"station_id":station,"feature_id":feature,"node_index":node,"nexus_id":nexus})).collect::<Vec<_>>()}),
    )?;
    write_json(&out.join("network.json"), n)?;
    write_json(
        &out.join("attributes.json"),
        &network::properties(&prepared.layers),
    )?;
    if let Some(w) = &w {
        write_json(
            &out.join("weights.json"),
            &json!({"grid":w.grid,"method":w.method,"catchment_ids":n.catchment_ids,"minimum_coverage":w.coverage_fraction.iter().copied().fold(f64::INFINITY,f64::min),"maximum_coverage":w.coverage_fraction.iter().copied().fold(f64::NEG_INFINITY,f64::max),"maximum_weight_sum_error":w.offsets.windows(2).map(|r|(w.weights[r[0] as usize..r[1] as usize].iter().sum::<f64>()-1.).abs()).fold(0.,f64::max),"edge_count":w.indices.len()}),
        )?;
    }
    let mut parsed = String::from("layer\tid\ttoid\n");
    for role in ["divides", "flowpaths", "nexus"] {
        for row in &prepared.layers[role].rows {
            let id = network::id(row, role)?;
            let to = network::field(row, "toid").unwrap_or_default();
            if id.contains(['\t', '\n']) || to.contains(['\t', '\n']) {
                return Err("ID cannot contain TSV separators".into());
            }
            parsed.push_str(&format!("{role}\t{id}\t{to}\n"));
        }
    }
    fs::write(out.join("parsed_network.tsv"), parsed)?;
    let mut manifest = json!({"schema":"arwen.nextgen.hydrofabric.v1","output":out,"request_sha256":key,"ngen_parser_pin":"be11a180565a78ff069d855b0b5dbfc23845e7d8","input":{"path":c.geopackage,"sha256":input_hash,"data_license":c.data_license},"node_count":n.node_ids.len(),"catchment_count":n.catchment_ids.len(),"level_count":n.level_offsets.len()-1,"node_order":"flowpaths_SQLite_scan; string_IDs_preserved","upstream_order":n.upstream_order,"grid":c.grid,"files":files,"layers":prepared.layers.iter().map(|(role,l)|json!({"role":role,"table":l.name,"row_count":l.rows.len(),"srs_id":l.srs_id,"geometry_column":l.geom_column})).collect::<Vec<_>>(),"timings_ms":{"sqlite_read":prepared.read_ms,"topology":prepared.topology_ms,"area_weights":weights_ms,"channel_raster":raster_ms,"write_hash":write_timer.elapsed().as_secs_f64()*1000.,"total":timer.elapsed().as_secs_f64()*1000.},"parallelism":{"rayon_threads":rayon::current_num_threads(),"table_reads":"independent read-only connections in parallel","geometry":"parallel catchments; deterministic serial cell order","topology":"startup-only dependency traversal","gpu":"none; no per-step host work"},"numerics":"fast default; no numerical routing in this tool; f64 geometry/f32 MC parameters"});
    if let Some(parent) = &c.parent_grid_contract {
        manifest["parent_grid_contract"] = parent.clone();
    }
    for name in [
        "network.json",
        "attributes.json",
        "parsed_network.tsv",
        "gauges.json",
    ] {
        manifest["auxiliary"][name] = json!({"sha256":digest(&out.join(name))?});
    }
    write_json(&out.join("manifest.json"), &manifest)?;
    Ok(manifest)
}
/// Startup transport only. Rust validates the complete payload before copying.
/// # Safety
/// path and expected_sha256 point to valid NUL-terminated strings; destination
/// is writable for exactly length bytes and does not alias either string.
#[no_mangle]
pub unsafe extern "C" fn arwen_hydrofabric_read_into(
    path: *const c_char,
    expected_sha256: *const c_char,
    destination: *mut u8,
    length: usize,
) -> i32 {
    if path.is_null() || expected_sha256.is_null() || (destination.is_null() && length != 0) {
        return 1;
    }
    let run = std::panic::catch_unwind(|| -> Result<()> {
        let path = Path::new(CStr::from_ptr(path).to_str()?);
        let expected = CStr::from_ptr(expected_sha256).to_str()?;
        let bytes = fs::read(path)?;
        if bytes.len() != length {
            return Err("payload length mismatch".into());
        }
        if format!("{:x}", Sha256::digest(&bytes)) != expected {
            return Err("payload digest mismatch".into());
        }
        if length != 0 {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, length)
        }
        Ok(())
    });
    match run {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 2,
        Err(_) => 3,
    }
}
