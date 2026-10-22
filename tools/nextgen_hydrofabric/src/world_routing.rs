//! Native-free MC + level-pool routing preparation for a HYDRO-WORLD fabric.
//!
//! `prepare-routing-replay` (routing_capture.rs) transports a native t-route
//! capture; a fabric built by rw_hydro_world has no native capture. This module
//! derives the same routing manifest contract from the prepared fabric alone:
//! the waterbody collapse of collapse.rs (t-route 12a8eae preprocess_waterbodies
//! and replace_waterbodies_connections, Apache-2.0), dependency levels, MC
//! parameters, an explicit initial channel state, the reservoir fragment of
//! `nextgen-reservoir-prepare --fabric` and its level selections.
//!
//! `reservoir-initial-from-spinup` writes that tool's initial packet from a
//! spun-up channel state: release equals the steady inflow at the lake outlet
//! and the height solves the level-pool release equation for it (the same
//! equation as gpuwm/core/kernels/nextgen_reservoirs.cuh, evaluated in f64).
//!
//! Upstream incoming lists are in ascending node-ID order: there is no native
//! order to preserve, and the order is fixed by the IDs alone.
use crate::{bundle::digest, collapse::collapse_graph, network::levels, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

fn suffix(v: &Value) -> Result<i64> {
    v.as_str()
        .and_then(|s| s.rsplit('-').next())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "invalid fabric numeric ID".into())
}
fn number(v: &Value) -> Option<i64> {
    v.as_i64()
        .or_else(|| v.as_f64().filter(|x| x.is_finite() && x.fract() == 0.).map(|x| x as i64))
        .or_else(|| v.as_str().and_then(|x| x.parse().ok()))
}
fn checked(path: &Path, sha: &str) -> Result<Vec<u8>> {
    let b = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&b)) != sha {
        return Err(format!("{}: SHA-256 differs from its manifest", path.display()).into());
    }
    Ok(b)
}
fn fabric_attributes(manifest_path: &Path) -> Result<(Value, Value, PathBuf)> {
    let fabric: Value = serde_json::from_slice(&fs::read(manifest_path)?)?;
    let root = manifest_path.parent().ok_or("fabric parent")?.to_path_buf();
    let a = checked(
        &root.join("attributes.json"),
        fabric["auxiliary"]["attributes.json"]["sha256"].as_str().ok_or("attributes SHA")?,
    )?;
    Ok((fabric, serde_json::from_slice(&a)?, root))
}

/// Spun-up channel state: routing_state (n,3) f64 [Q m3/s, v m/s, h m] in the
/// order of routing_ids.json (flowpath IDs). New producers seal the ordered
/// ID bytes in the same manifest; legacy v1 packets lack that association.
struct ChannelState {
    by_id: HashMap<i64, [f64; 3]>,
    manifest_sha256: String,
}
fn channel_state(dir: &Path, fabric: &Value, time_origin: &str) -> Result<ChannelState> {
    let mpath = dir.join("manifest.json");
    let m: Value = serde_json::from_slice(&fs::read(&mpath)?)?;
    channel_source(&m, fabric, time_origin)?;
    let e = m["arrays"]
        .as_array()
        .and_then(|a| a.iter().find(|e| e["name"] == "routing_state"))
        .ok_or("spin-up manifest has no routing_state")?;
    let b = checked(&dir.join(e["file"].as_str().ok_or("routing_state file")?), e["sha256"].as_str().unwrap_or(""))?;
    let ids = channel_ids(&fs::read(dir.join("routing_ids.json"))?, m.get("routing_ids"))?;
    let v = channel_values(&b, e, ids.len())?;
    let mut by_id = HashMap::with_capacity(ids.len());
    for (k, id) in ids.iter().enumerate() {
        let key = suffix(&json!(id))?;
        let row = [v[3 * k], v[3 * k + 1], v[3 * k + 2]];
        if row.iter().any(|x| !x.is_finite()) || by_id.insert(key, row).is_some() {
            return Err("routing_state row nonfinite or duplicated".into());
        }
    }
    Ok(ChannelState { by_id, manifest_sha256: digest(&mpath)? })
}

// The spin-up producer hashes the source GeoPackage (not this prepared
// manifest) and stamps the state at `end`, after its last forcing record.
// Ignoring either field lets a stale/other-fabric packet be relabelled with
// the caller's current identity. Source paths remain informational/movable.
fn channel_source(m: &Value, fabric: &Value, time_origin: &str) -> Result<()> {
    let producer = m["schema"] == "arwen.hydro-world.init-spinup.v1";
    if producer || m.get("end").is_some() {
        if time_origin.trim().is_empty() || m["end"].as_str() != Some(time_origin) {
            return Err("spin-up end differs from requested routing time_origin".into());
        }
    }
    if producer || m.get("fabric_sha256").is_some() {
        let expected = fabric["input"]["sha256"].as_str().filter(|s|
            s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()));
        if expected.is_none() || m["fabric_sha256"].as_str() != expected {
            return Err("spin-up fabric_sha256 differs from routing fabric input.sha256".into());
        }
    }
    // Legacy descriptor-absent capture fixtures remain unsealed; they do
    // not acquire a source/epoch proof by passing the payload layout check.
    Ok(())
}

// A state payload hash alone cannot detect reordered IDs. Validate a producer's
// axis seal before associating any values. A malformed seal must not fall back
// to legacy decoding; existing packets without the new descriptor still load.
fn channel_ids(bytes: &[u8], seal: Option<&Value>) -> Result<Vec<String>> {
    if let Some(field) = seal {
        if field["file"] != "routing_ids.json" {
            return Err("routing_ids seal must name routing_ids.json".into());
        }
        if field["sha256"].as_str() != Some(format!("{:x}", Sha256::digest(bytes)).as_str()) {
            return Err("routing_ids.json: SHA-256 differs from its spin-up manifest seal".into());
        }
    }
    let ids: Vec<String> = serde_json::from_slice(bytes)?;
    if let Some(field) = seal {
        if field["count"].as_u64() != Some(ids.len() as u64) {
            return Err("routing_ids seal count differs from its ordered JSON axis".into());
        }
    }
    Ok(ids)
}

// Check the transport layout before decoding: chunks_exact alone silently
// discards a partial word, and equal value counts do not establish the axes.
fn channel_values(bytes: &[u8], field: &Value, rows: usize) -> Result<Vec<f64>> {
    let expected = rows.checked_mul(3).and_then(|n| n.checked_mul(8)).ok_or("routing_state extent overflows usize")?;
    if field["dtype"] != "<f8" || field["shape"] != json!([rows, 3]) {
        return Err("routing_state must be (routing_ids.json length, 3) <f8 [Q, velocity, depth]".into());
    }
    if bytes.len() != expected || field["bytes"].as_u64() != Some(expected as u64) {
        return Err("routing_state byte extent differs from its declared layout or routing_ids.json".into());
    }
    Ok(bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
}

// Fragment hashes establish byte identity, not the declared numeric layout.
// Check before chunks_exact can discard a partial word or a dtype fallback
// can reinterpret integer/big-endian bytes as little-endian release values.
fn fragment_vector(bytes: &[u8], field: &Value, rows: usize, dtype: &str, width: usize, name: &str) -> Result<()> {
    let expected = rows.checked_mul(width).ok_or_else(|| format!("{name} extent overflows usize"))?;
    if field["dtype"] != dtype || field["shape"] != json!([rows]) {
        return Err(format!("{name} must be ({rows},) {dtype} on the retained reservoir axis").into());
    }
    // The producer's reservoir_ids descriptor has no bytes member; when one
    // is supplied it must agree with the same exact payload extent.
    if bytes.len() != expected || field.get("bytes").is_some_and(|v| v.as_u64() != Some(expected as u64)) {
        return Err(format!("{name} byte extent differs from its declared reservoir layout").into());
    }
    Ok(())
}

fn reservoir_releases(bytes: &[u8], field: &Value, rows: usize) -> Result<Vec<f64>> {
    let (dtype, width) = match field["dtype"].as_str() {
        Some("<f8") => ("<f8", 8),
        Some("<f4") => ("<f4", 4),
        _ => return Err("reservoir release dtype must be little-endian <f8 or <f4".into()),
    };
    fragment_vector(bytes, field, rows, dtype, width, "reservoir release")?;
    Ok(bytes.chunks_exact(width)
        .map(|x| if width == 8 { f64::from_le_bytes(x.try_into().unwrap()) } else { f32::from_le_bytes(x.try_into().unwrap()) as f64 })
        .collect())
}

// Producer paths are informational and may move with the packet. Bind the
// declared source bytes instead: equal lake IDs do not identify the geometry.
// Legacy capture fragments lack these descriptors and retain their old scope.
fn fragment_source(fragment: &Value, fabric_sha: &str, attributes_sha: &str) -> Result<()> {
    if let Some(source) = fragment.get("source_manifest") {
        if source["sha256"].as_str() != Some(fabric_sha) {
            return Err("reservoir fragment source_manifest SHA differs from the routing fabric".into());
        }
    }
    if let Some(attributes) = fragment.get("attributes_sha256") {
        if attributes.as_str() != Some(attributes_sha) {
            return Err("reservoir fragment attributes SHA differs from the routing fabric geometry".into());
        }
    }
    Ok(())
}

// Validate the raw MC table before row indexing. A correct payload hash does
// not establish its layout: a truncated row used to panic inside Rayon, while
// trailing words and duplicate IDs could silently select the wrong row.
fn raw_routing_table(id_bytes: &[u8], ids: &Value, param_bytes: &[u8], params: &Value) -> Result<(Vec<i64>, Vec<f32>)> {
    let shape = ids["shape"].as_array().filter(|s| s.len() == 1).ok_or("prepared node_ids must be (raw nodes,) <i8")?;
    let rows = usize::try_from(shape[0].as_u64().ok_or("prepared node_ids row count must be an integer")?)?;
    if ids["dtype"] != "<i8" || params["dtype"] != "<f4" || params["shape"] != json!([rows, 8]) {
        return Err("prepared node_ids must be (raw nodes,) <i8 and routing_parameters (raw nodes, 8) <f4".into());
    }
    for (bytes, field, width, name) in [(id_bytes, ids, 8usize, "node_ids"), (param_bytes, params, 32usize, "routing_parameters")] {
        let expected = rows.checked_mul(width).ok_or_else(|| format!("prepared {name} extent overflows usize"))?;
        if bytes.len() != expected || field.get("bytes").is_some_and(|v| v.as_u64() != Some(expected as u64)) {
            return Err(format!("prepared {name} byte extent differs from its raw MC table layout").into());
        }
    }
    let raw_ids: Vec<i64> = id_bytes.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
    let mut seen = std::collections::HashSet::with_capacity(rows);
    if raw_ids.iter().any(|id| !seen.insert(*id)) {
        return Err("prepared node_ids contains duplicate raw MC row IDs".into());
    }
    let raw_params = param_bytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
    Ok((raw_ids, raw_params))
}

// A finite spin-up f64 can overflow when narrowed to the routing f32 ABI.
// Reject nonfinite output before creating a packet; retain ordinary rounding.
fn routing_f32(value: f64, node: i64, field: &str) -> Result<f32> {
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(format!("routing node {node}: {field} is not finite in the <f4 routing state").into());
    }
    Ok(narrowed)
}

/// Level-pool steady release for a stage (f64 form of ngen_reservoir_release
/// with initial_h = stage_h).
fn release(h: f64, p: &LakeParams) -> f64 {
    let mut depth = h - p.weir_e;
    if depth > p.max_e - p.weir_e {
        depth = p.max_e - p.weir_e;
    }
    let outlet = if h > p.orifice_e { p.orifice_c * p.orifice_a * (2.0 * 9.81 * (h - p.orifice_e)).sqrt() } else { 0.0 };
    let spill = if depth > 0.0 { p.weir_c * p.weir_l * depth.powf(1.5) } else { 0.0 };
    if h > p.max_e {
        return outlet + spill + p.weir_c * p.weir_l * p.dam_l * (h - p.max_e).powf(1.5);
    }
    outlet + spill
}
struct LakeParams {
    weir_e: f64,
    max_e: f64,
    weir_c: f64,
    weir_l: f64,
    dam_l: f64,
    orifice_e: f64,
    orifice_c: f64,
    orifice_a: f64,
}
fn lake_params(row: &Value) -> Result<LakeParams> {
    let f = |k: &str| row[k].as_f64().filter(|x| x.is_finite()).ok_or_else(|| -> crate::Error { format!("lake {k} missing").into() });
    Ok(LakeParams {
        weir_e: f("WeirE")?,
        max_e: f("LkMxE")?,
        weir_c: f("WeirC")?,
        weir_l: f("WeirL")?,
        dam_l: f("Dam_Length")?,
        orifice_e: f("OrificeE")?,
        orifice_c: f("OrificeC")?,
        orifice_a: f("OrificeA")?,
    })
}
/// Stage at which the steady release equals q (bisection; release is
/// non-decreasing in stage above the orifice).
fn stage_for(q: f64, p: &LakeParams) -> Result<f64> {
    if q <= 0.0 {
        return Ok(p.orifice_e);
    }
    let (mut lo, mut hi) = (p.orifice_e, p.max_e.max(p.orifice_e) + 1.0);
    loop {
        let upper_release = release(hi, p);
        if !hi.is_finite() || !upper_release.is_finite() {
            return Err("cannot bracket reservoir release at a finite stage; check outlet coefficients and geometry".into());
        }
        if upper_release >= q {
            break;
        }
        let next = p.orifice_e + 2.0 * (hi - p.orifice_e);
        if next <= hi {
            return Err("reservoir stage bracket cannot increase; check outlet coefficients and geometry".into());
        }
        hi = next;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if release(mid, p) < q { lo = mid } else { hi = mid }
    }
    Ok(hi)
}

#[derive(Deserialize)]
struct InitialConfig {
    fabric_manifest: PathBuf,
    initial_state_dir: PathBuf,
    time_origin: String,
    vertical_datum: String,
    output: PathBuf,
}
pub fn reservoir_initial(config_path: &Path) -> Result<Value> {
    let start = Instant::now();
    let c: InitialConfig = serde_json::from_slice(&fs::read(config_path)?)?;
    let (fabric, a, _) = fabric_attributes(&c.fabric_manifest)?;
    let g = collapse_graph(&a)?;
    let st = channel_state(&c.initial_state_dir, &fabric, &c.time_origin)?;
    let mut rows = Vec::new();
    for row in a["lakes"].as_array().ok_or("missing lakes")? {
        let hl = number(&row["hl_link"]).ok_or("lake hl_link")?;
        let node = *g.aliases.get(&hl).ok_or_else(|| format!("lake {hl} has incomplete level-pool columns"))?;
        if !g.lakes.contains(&node) {
            return Err(format!("lake {hl} was dropped by the collapse (multiple outlets); its reservoir row cannot be routed").into());
        }
        let outlet = *g.crosswalk.get(&node).ok_or_else(|| format!("lake {hl} has no unique outlet member"))?;
        let q = st.by_id.get(&outlet).ok_or_else(|| format!("lake {hl}: no spin-up channel state for outlet {outlet}"))?[0].max(0.0);
        let p = lake_params(row)?;
        let h = stage_for(q, &p).map_err(|e| format!("lake {hl}: {e}"))?;
        rows.push(json!({"id": hl, "height_m": h, "release_m3_s": release(h, &p), "steady_inflow_m3_s": q, "outlet_flowpath": outlet}));
    }
    rows.sort_by_key(|r| r["id"].as_i64().unwrap());
    let packet = json!({"schema": "arwen.nextgen.reservoir-initial.v1", "time_origin": c.time_origin,
        "vertical_datum": c.vertical_datum,
        "source": {"method": "replay-spinup", "sha256": st.manifest_sha256, "fabric_manifest_sha256": digest(&c.fabric_manifest)?,
                   "time_origin": c.time_origin, "vertical_datum": c.vertical_datum},
        "method": "release = spin-up steady channel discharge at the lake's outlet member flowpath; height solves the level-pool release equation (nextgen_reservoirs.cuh, f64) for that release by bisection",
        "rows": rows});
    fs::write(&c.output, serde_json::to_vec_pretty(&packet)?)?;
    Ok(json!({"output": c.output, "lakes": packet["rows"].as_array().unwrap().len(), "elapsed_ms": start.elapsed().as_secs_f64() * 1000.,
              "parallelism": "1 thread (startup only, one bisection per lake)"}))
}

#[derive(Deserialize)]
struct Config {
    fabric_manifest: PathBuf,
    output: PathBuf,
    time_origin: String,
    data_license: String,
    dt_seconds: f64,
    initial_state_dir: PathBuf,
    /// nextgen-reservoir-prepare --fabric fragment manifest; required when the
    /// fabric has lakes.
    reservoir_manifest: Option<PathBuf>,
}
struct Array {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

// A world fabric's raster indexes raw flowpaths. Routing indexes the collapsed
// graph, including lake nodes, so transport the raster through that same map.
// The inverse is the actual channel-cell incidence, not a drainage-area policy.
fn channel_cells(bytes: &[u8], field: &Value, grid: &Value, remap: &[i32], n: usize)
    -> Result<(Vec<i32>, Vec<i64>, Vec<i32>, usize)>
{
    let nx = usize::try_from(grid["nx"].as_u64().ok_or("channel grid nx missing")?)?;
    let ny = usize::try_from(grid["ny"].as_u64().ok_or("channel grid ny missing")?)?;
    let cells = nx.checked_mul(ny).ok_or("channel grid size overflow")?;
    if cells == 0 || cells > i32::MAX as usize || n > i32::MAX as usize
        || field["dtype"] != "<i4" || field["shape"] != json!([ny, nx])
        || bytes.len() != cells * 4 || field["bytes"].as_u64() != Some(bytes.len() as u64)
    {
        return Err("channel_node must be an exact int32 [ny,nx] raster on its grid".into());
    }
    let mut changed = 0;
    let mapped = bytes.chunks_exact(4).map(|b| -> Result<i32> {
        let raw = i32::from_le_bytes(b.try_into().unwrap());
        if raw == -1 { return Ok(-1); }
        if raw < -1 { return Err("channel_node has an invalid nonchannel sentinel".into()); }
        let node = *remap.get(raw as usize).ok_or("channel raster node outside fabric")?;
        if node < 0 || node as usize >= n {
            return Err("channel raster remap outside collapsed node axis".into());
        }
        changed += usize::from(node != raw);
        Ok(node)
    }).collect::<Result<Vec<_>>>()?;
    let mut pairs = mapped.par_iter().enumerate()
        .filter_map(|(cell, &node)| (node >= 0).then_some((node, cell as i32)))
        .collect::<Vec<_>>();
    pairs.par_sort_unstable();
    let offsets = (0..=n).into_par_iter()
        .map(|node| pairs.partition_point(|&(id, _)| id < node as i32) as i64)
        .collect::<Vec<_>>();
    let indices = pairs.par_iter().map(|&(_, cell)| cell).collect::<Vec<_>>();
    let mut reconstructed = vec![-1; cells];
    for node in 0..n {
        let row = &indices[offsets[node] as usize..offsets[node + 1] as usize];
        if row.windows(2).any(|p| p[0] >= p[1]) {
            return Err("inverse channel cells are not strictly ascending".into());
        }
        for &cell in row {
            if reconstructed[cell as usize] != -1 {
                return Err("duplicate inverse channel cell".into());
            }
            reconstructed[cell as usize] = node as i32;
        }
    }
    if reconstructed != mapped {
        return Err("collapsed channel-cell CSR does not reconstruct actual raster".into());
    }
    Ok((mapped, offsets, indices, changed))
}
macro_rules! payload {
    ($v:expr, $name:expr, $dtype:literal) => {
        Array { name: $name.into(), dtype: $dtype, shape: vec![$v.len()], bytes: $v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    };
}

pub fn prepare(config_path: &Path) -> Result<Value> {
    let start = Instant::now();
    let config_bytes = fs::read(config_path)?;
    let c: Config = serde_json::from_slice(&config_bytes)?;
    if c.data_license.trim().is_empty() || c.time_origin.trim().is_empty() || !(c.dt_seconds > 0.0) {
        return Err("explicit data_license, time_origin and positive dt_seconds required".into());
    }
    let (fabric, a, froot) = fabric_attributes(&c.fabric_manifest)?;
    let g = collapse_graph(&a)?;
    if !g.ambiguous_outlets.is_empty() {
        return Err(format!("{} lakes have ambiguous outlets; an explicit crosswalk is required", g.ambiguous_outlets.len()).into());
    }
    let mut ids: Vec<i64> = g.collapsed.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    let index: HashMap<i64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    if index.len() != ids.len() || ids.len() > i32::MAX as usize {
        return Err("collapsed roster has duplicate IDs or exceeds int32".into());
    }
    let n = ids.len();
    let mut downstream = vec![-1i32; n];
    let mut upstream: Vec<Vec<i32>> = vec![Vec::new(); n];
    for &(id, ds) in &g.collapsed {
        let i = index[&id];
        if let Some(d) = ds {
            let j = *index.get(&d).ok_or("downstream outside collapsed graph")?;
            downstream[i] = j as i32;
            upstream[j].push(i as i32);
        }
    }
    for up in &mut upstream {
        up.sort_unstable(); // indices are ID-sorted, so this is ascending node ID
    }
    let mut offsets = vec![0i32];
    let mut indices = Vec::<i32>::new();
    for up in &upstream {
        indices.extend(up);
        offsets.push(i32::try_from(indices.len())?);
    }
    let (ordered, level_offsets) = levels(&upstream, &downstream, false)?;
    let (reverse_ordered, reverse_offsets) = levels(&upstream, &downstream, true)?;
    let graph_ms = start.elapsed().as_secs_f64() * 1000.;
    let pack = Instant::now();
    // Reservoirs: fragment order is hl_link ascending.
    let (lake_ids, fragment, fragment_root): (Vec<i64>, Value, PathBuf) = if g.lakes.is_empty() {
        (Vec::new(), Value::Null, PathBuf::new())
    } else {
        let path = c.reservoir_manifest.as_ref().ok_or("fabric has lakes: reservoir_manifest required")?;
        let f: Value = serde_json::from_slice(&fs::read(path)?)?;
        let root = path.parent().ok_or("fragment parent")?.to_path_buf();
        if f["reservoirs"]["time_origin"] != c.time_origin.as_str() || f["reservoirs"]["validated_csr"] != true {
            return Err("reservoir fragment time origin or CSR validation differs".into());
        }
        fragment_source(&f, &digest(&c.fabric_manifest)?,
                        fabric["auxiliary"]["attributes.json"]["sha256"].as_str().ok_or("attributes SHA")?)?;
        let field = &f["reservoir_ids"];
        let b = checked(&root.join(field["path"].as_str().ok_or("reservoir_ids path")?), field["sha256"].as_str().unwrap_or(""))?;
        fragment_vector(&b, field, g.lakes.len(), "<i8", 8, "reservoir_ids")?;
        let hl: Vec<i64> = b.chunks_exact(8).map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect();
        (hl, f, root)
    };
    let mut reservoir_nodes = Vec::with_capacity(lake_ids.len());
    let mut lake_row = HashMap::new();
    for (row, hl) in lake_ids.iter().enumerate() {
        let node = *g.aliases.get(hl).ok_or_else(|| format!("reservoir {hl} is not a fabric lake with complete columns"))?;
        if !g.lakes.contains(&node) {
            return Err(format!("reservoir {hl} was dropped by the collapse").into());
        }
        reservoir_nodes.push(i32::try_from(index[&node])?);
        if lake_row.insert(node, row).is_some() {
            return Err(format!("reservoir {hl} duplicates a retained lake in the fragment axis").into());
        }
    }
    if lake_row.len() != g.lakes.len() {
        return Err(format!("reservoir fragment has {} rows for {} retained lakes", lake_row.len(), g.lakes.len()).into());
    }
    let mut mc = Vec::new();
    let mut mc_offsets = vec![0i32];
    let mut selections = Vec::new();
    let mut arrays = Vec::<Array>::new();
    for (level, range) in level_offsets.windows(2).enumerate() {
        let mut rows = Vec::<i32>::new();
        for &node in &ordered[range[0] as usize..range[1] as usize] {
            if let Some(&row) = lake_row.get(&ids[node as usize]) {
                rows.push(i32::try_from(row)?);
            } else {
                mc.push(node);
            }
        }
        mc_offsets.push(i32::try_from(mc.len())?);
        if !rows.is_empty() {
            let name = format!("reservoir_level_{level}_rows");
            arrays.push(payload!(rows, name.clone(), "<i4"));
            selections.push(json!({"level": level, "rows": name}));
        }
    }
    // MC parameters from the prepared raw-node table.
    let (raw_ids, raw_params) = {
        let id_field = &fabric["files"]["node_ids"];
        let param_field = &fabric["files"]["routing_parameters"];
        let id_bytes = checked(&froot.join(id_field["path"].as_str().ok_or("node_ids path")?), id_field["sha256"].as_str().unwrap_or(""))?;
        let param_bytes = checked(&froot.join(param_field["path"].as_str().ok_or("routing_parameters path")?), param_field["sha256"].as_str().unwrap_or(""))?;
        raw_routing_table(&id_bytes, id_field, &param_bytes, param_field)?
    };
    let raw_index: HashMap<i64, usize> = raw_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let parameters = ids
        .par_iter()
        .map(|id| -> Result<[f32; 8]> {
            if lake_row.contains_key(id) {
                return Ok([0.; 8]);
            }
            let r = *raw_index.get(id).ok_or("MC node missing from prepared parameters")?;
            let p: [f32; 8] = std::array::from_fn(|j| raw_params[8 * r + j]);
            if p.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite MC parameter".into());
            }
            Ok(p)
        })
        .collect::<Result<Vec<_>>>()?;
    // Initial state: MC nodes [Q, 0, h] from the spin-up (t-route starts velocity
    // at zero), reservoir nodes [release, 0, 0] from the reservoir packet.
    let st = channel_state(&c.initial_state_dir, &fabric, &c.time_origin)?;
    let mut state = vec![[0f32; 3]; n];
    for (i, id) in ids.iter().enumerate() {
        if !lake_row.contains_key(id) {
            let s = st.by_id.get(id).ok_or_else(|| format!("no spin-up channel state for flowpath {id}"))?;
            state[i] = [routing_f32(s[0], *id, "discharge")?, 0., routing_f32(s[2], *id, "depth")?];
        }
    }
    if !lake_ids.is_empty() {
        // The fragment carries initial releases in its f64 ABI payload.
        let key = fragment["reservoirs"]["arrays"]["release"].as_str().ok_or("reservoir fragment binds no release array")?;
        let rel = (key, &fragment["files"][key]);
        let b = checked(&fragment_root.join(rel.1["path"].as_str().ok_or("release path")?), rel.1["sha256"].as_str().unwrap_or(""))?;
        let vals = reservoir_releases(&b, rel.1, lake_ids.len())?;
        for (row, &node) in reservoir_nodes.iter().enumerate() {
            state[node as usize] = [routing_f32(vals[row], ids[node as usize], "reservoir release")?, 0., 0.];
        }
    }
    // Catchments drain to their flowpath, or to the lake that absorbed it.
    let network_bytes = checked(&froot.join("network.json"), fabric["auxiliary"]["network.json"]["sha256"].as_str().unwrap_or(""))?;
    let mut network: Value = serde_json::from_slice(&network_bytes)?;
    let old_ids: Vec<i64> = network["numeric_node_ids"]
        .as_array()
        .ok_or("fabric requires numeric node IDs")?
        .iter()
        .map(|v| v.as_i64().ok_or_else(|| "bad numeric node ID".into()))
        .collect::<Result<_>>()?;
    let remap = old_ids
        .iter()
        .map(|id| -> Result<i32> {
            let target = g.membership.get(id).copied().unwrap_or(*id);
            Ok(i32::try_from(*index.get(&target).ok_or("fabric node not in collapsed graph")?)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let catchment_to_node = network["catchment_to_node"]
        .as_array()
        .ok_or("missing catchment mapping")?
        .iter()
        .map(|v| -> Result<i32> { Ok(*remap.get(v.as_u64().ok_or("bad catchment node")? as usize).ok_or("catchment node outside fabric")?) })
        .collect::<Result<Vec<_>>>()?;
    let output_ids: Vec<i64> = ids.iter().map(|id| if lake_row.contains_key(id) { g.crosswalk[id] } else { *id }).collect();
    // Catchment areas (km2, the contract's planar equal-area areasqkm) on the
    // network's catchment axis, for the runoff depth -> discharge transfer.
    let divide_area: HashMap<&str, f64> = a["divides"]
        .as_array()
        .ok_or("missing divides")?
        .iter()
        .filter_map(|d| Some((d["divide_id"].as_str()?, d["areasqkm"].as_f64()?)))
        .collect();
    let catchment_area = network["catchment_ids"]
        .as_array()
        .ok_or("network lacks catchment_ids")?
        .iter()
        .map(|v| -> Result<f64> {
            let id = v.as_str().ok_or("bad catchment id")?;
            divide_area.get(id).copied().filter(|x| x.is_finite() && *x > 0.0).ok_or_else(|| format!("{id}: no positive areasqkm").into())
        })
        .collect::<Result<Vec<_>>>()?;
    let lateral = vec![0f32; n];
    let mut identity = Sha256::new();
    identity.update(&config_bytes);
    identity.update(digest(&c.fabric_manifest)?.as_bytes());
    if let Some(p) = &c.reservoir_manifest {
        identity.update(digest(p)?.as_bytes());
    }
    identity.update(st.manifest_sha256.as_bytes());
    let key = format!("{:x}", identity.finalize());
    let out = c.output.join(&key[..20]);
    fs::create_dir_all(&out)?;
    let mut files = serde_json::Map::new();
    let mut cell_mapping = None;
    for (name, field) in fabric["files"].as_object().ok_or("missing fabric files")? {
        if name.starts_with("weights") || name.starts_with("exchange")
            || ["overlap_m2", "catchment_area_m2", "coverage_fraction", "stream_order"].contains(&name.as_str())
        {
            let bytes = checked(&froot.join(field["path"].as_str().ok_or("payload path")?), field["sha256"].as_str().unwrap_or(""))?;
            let filename = format!("{name}.bin");
            fs::write(out.join(&filename), bytes)?;
            let mut f = field.clone();
            f["path"] = json!(filename);
            files.insert(name.clone(), f);
        } else if name == "channel_node" {
            let inverse_start = Instant::now();
            let bytes = checked(&froot.join(field["path"].as_str().ok_or("channel raster path")?),
                                field["sha256"].as_str().unwrap_or(""))?;
            let (mapped, cell_offsets, cell_indices, changed) =
                channel_cells(&bytes, field, &fabric["grid"], &remap, n)?;
            let mut mapping = json!({
                "offsets":"routing_cell_offsets", "indices":"routing_cell_indices",
                "node_axis":"node_ids", "source":"channel_node", "cells":cell_indices.len(),
                "source_channel_node_sha256":field["sha256"],
                "grid_sha256":format!("{:x}", Sha256::digest(serde_json::to_vec(&fabric["grid"])?)),
                "cell_order":"ascending flattened x-fast cell index within each current collapsed node",
                "scope":"actual rasterized channel-cell incidence only; nonchannel cells excluded; reservoir rows included where member flowlines rasterize; no drainage or lateral policy inferred",
                "reservoir_lateral_policy":"unbound; producer and receiver must explicitly agree",
                "remapped_cells":changed, "roundtrip_cells":mapped.len(), "different_cells":0,
                "prepare_ms":inverse_start.elapsed().as_secs_f64()*1000.
            });
            if let Some(parent) = fabric.get("parent_grid_contract") {
                mapping["parent_grid_contract_sha256"] = json!(format!("{:x}",
                    Sha256::digest(serde_json::to_vec(parent)?)));
            }
            cell_mapping = Some(mapping);
            arrays.push(payload!(cell_offsets, "routing_cell_offsets", "<i8"));
            arrays.push(payload!(cell_indices, "routing_cell_indices", "<i4"));
            let mut a = payload!(mapped, "channel_node", "<i4");
            a.shape = serde_json::from_value(field["shape"].clone())?;
            arrays.push(a);
        }
    }
    arrays.extend([
        payload!(output_ids, "output_reach_ids", "<i8"),
        payload!(ids, "node_ids", "<i8"),
        payload!(downstream, "downstream", "<i4"),
        payload!(offsets, "upstream_offsets", "<i4"),
        payload!(indices, "upstream_indices", "<i4"),
        payload!(ordered, "ordered_nodes", "<i4"),
        payload!(level_offsets, "level_offsets", "<i4"),
        payload!(reverse_ordered, "reverse_ordered_nodes", "<i4"),
        payload!(reverse_offsets, "reverse_level_offsets", "<i4"),
        payload!(mc, "mc_ordered_nodes", "<i4"),
        payload!(reservoir_nodes, "reservoir_node_indices", "<i4"),
        payload!(catchment_to_node, "catchment_to_node", "<i4"),
        payload!(remap, "raw_node_to_routing_node", "<i4"),
        payload!(lake_ids, "reservoir_ids", "<i8"),
        payload!(lateral, "routing_lateral", "<f4"),
        payload!(catchment_area, "catchment_areasqkm", "<f8"),
    ]);
    arrays.push(Array { name: "routing_parameters".into(), dtype: "<f4", shape: vec![n, 8],
        bytes: parameters.iter().flatten().flat_map(|x| x.to_le_bytes()).collect() });
    arrays.push(Array { name: "routing_state".into(), dtype: "<f4", shape: vec![n, 3],
        bytes: state.iter().flatten().flat_map(|x| x.to_le_bytes()).collect() });
    if !lake_ids.is_empty() {
        for (name, field) in fragment["files"].as_object().ok_or("missing reservoir fragment files")? {
            if files.contains_key(name) || arrays.iter().any(|a| &a.name == name) {
                return Err("duplicate reservoir payload key".into());
            }
            let bytes = checked(&fragment_root.join(field["path"].as_str().ok_or("reservoir payload path")?), field["sha256"].as_str().unwrap_or(""))?;
            let filename = format!("{name}.bin");
            fs::write(out.join(&filename), bytes)?;
            let mut f = field.clone();
            f["path"] = json!(filename);
            files.insert(name.clone(), f);
        }
    }
    let pack_ms = pack.elapsed().as_secs_f64() * 1000.;
    let write = Instant::now();
    let generated = arrays
        .par_iter()
        .map(|a| -> Result<(String, Value)> {
            let name = format!("{}.bin", a.name);
            fs::write(out.join(&name), &a.bytes)?;
            Ok((a.name.clone(), json!({"path": name, "dtype": a.dtype, "shape": a.shape, "bytes": a.bytes.len(),
                "sha256": format!("{:x}", Sha256::digest(&a.bytes))})))
        })
        .collect::<Result<Vec<_>>>()?;
    files.extend(generated);
    let node_strings: Vec<String> = ids
        .iter()
        .map(|id| if lake_row.contains_key(id) { format!("lake-{id}") } else { format!("wb-{id}") })
        .collect();
    network["node_ids"] = json!(node_strings);
    network["numeric_node_ids"] = json!(ids);
    network["downstream"] = json!(downstream);
    network["upstream_offsets"] = json!(offsets);
    network["upstream_indices"] = json!(indices);
    network["ordered_nodes"] = json!(ordered);
    network["level_offsets"] = json!(level_offsets);
    network["reverse_ordered_nodes"] = json!(reverse_ordered);
    network["reverse_level_offsets"] = json!(reverse_offsets);
    network["catchment_to_node"] = json!(catchment_to_node);
    network["routing_parameters"] = Value::Null;
    network["downstream_ids"] = json!(downstream.iter().map(|&i| if i < 0 { None } else { Some(node_strings[i as usize].clone()) }).collect::<Vec<_>>());
    network["upstream_order"] = json!("ascending collapsed node ID");
    fs::write(out.join("network.json"), serde_json::to_vec_pretty(&network)?)?;
    let mut manifest = fabric.clone();
    manifest["output"] = json!(out);
    manifest["request_sha256"] = json!(key);
    manifest["files"] = json!(files);
    manifest["node_count"] = json!(n);
    manifest["level_count"] = json!(level_offsets.len() - 1);
    manifest["node_order"] = json!("collapsed IDs sorted i64; catchments retain divide order");
    manifest["upstream_order"] = network["upstream_order"].clone();
    manifest["auxiliary"] = json!({"network.json": {"sha256": digest(&out.join("network.json"))?},
                                   "attributes.json": fabric["auxiliary"]["attributes.json"].clone()});
    fs::copy(froot.join("attributes.json"), out.join("attributes.json"))?;
    manifest["routing"] = json!({"mc_level_offsets": mc_offsets, "reservoir_node_indices": "reservoir_node_indices",
        "reservoir_levels": selections, "validated_reservoir_mapping": true,
        "parameter_columns": ["dx", "bw", "tw", "twcc", "n", "ncc", "cs", "s0"],
        "native_incoming_order_verified": false, "upstream_csr_edges": indices.len(), "mc_count": mc.len(),
        "reservoir_count": lake_ids.len(), "initial_state_array": "routing_state", "lateral_array": "routing_lateral",
        "output_reach_ids": "output_reach_ids",
        "native_output_id_semantics": "lake node -> its outlet member flowpath; internal node_ids remain collapsed routing IDs",
        "dt_seconds": c.dt_seconds, "assume_short_ts": false});
    let spinup: Value = serde_json::from_slice(&fs::read(c.initial_state_dir.join("manifest.json"))?)?;
    if let Some(replay) = spinup.get("routing_replay") {
        if replay["end"] != c.time_origin || replay["hours"].as_u64().unwrap_or(0) == 0
            || replay["interval_seconds"] != 3600 || 3600.0 % c.dt_seconds != 0.0 {
            return Err("routing replay must end at forecast origin and contain whole routing periods".into());
        }
        if spinup["catchment_ids"] != network["catchment_ids"] {
            return Err("routing replay catchment axis differs from this exact network".into());
        }
        let field = spinup["arrays"].as_array().ok_or("spinup arrays")?.iter()
            .find(|a| a["name"] == replay["array"]).ok_or("runoff replay array missing")?;
        if field["dtype"] != "<f8" || field["units"] != "m per hour"
            || field["shape"] != json!([catchment_to_node.len(), replay["hours"].as_u64().unwrap()]) {
            return Err("routing replay needs catchment-major f64 hourly runoff depths".into());
        }
        let data = checked(&c.initial_state_dir.join(field["file"].as_str().ok_or("replay file")?),
                           field["sha256"].as_str().ok_or("replay hash")?)?;
        if data.len() as u64 != field["bytes"].as_u64().unwrap_or(0)
            || data.chunks_exact(8).any(|b| { let x = f64::from_le_bytes(b.try_into().unwrap()); !x.is_finite() || x < 0.0 }) {
            return Err("invalid runoff replay depth or extent".into());
        }
        // The verified spin-up payload, hard-linked (a second copy was 509 MB per
        // forecast start on the 10 x 6 degree Oklahoma WoFS box); a copy only
        // where the filesystem cannot link.
        let src = c.initial_state_dir.join(field["file"].as_str().ok_or("replay file")?);
        let dst = out.join("runoff-replay.f64");
        if fs::hard_link(&src, &dst).is_err() {
            fs::write(&dst, &data)?;
        }
        let mut declaration = replay.clone();
        declaration["payload"] = json!({"path": "runoff-replay.f64", "sha256": field["sha256"],
            "bytes": data.len(), "shape": field["shape"], "dtype": "<f8"});
        declaration["spinup_manifest_sha256"] = json!(digest(&c.initial_state_dir.join("manifest.json"))?);
        manifest["routing"]["initial_replay"] = declaration;
    }
    if let Some(mut mapping) = cell_mapping {
        mapping["channel_node_sha256"] = manifest["files"]["channel_node"]["sha256"].clone();
        manifest["routing"]["cell_mapping"] = mapping;
    }
    if !lake_ids.is_empty() {
        manifest["reservoirs"] = fragment["reservoirs"].clone();
    }
    manifest["world_provenance"] = json!({"scope": "native-free collapse of a HYDRO-WORLD fabric; no native capture exists for it",
        "collapse_source_pin": "12a8eae0cdfed437143c590659fa7077605a5e70", "fabric_manifest_sha256": digest(&c.fabric_manifest)?,
        "reservoir_manifest_sha256": c.reservoir_manifest.as_ref().map(|p| digest(p)).transpose()?,
        "initial_channel_state": {"dir": c.initial_state_dir, "manifest_sha256": st.manifest_sha256, "velocity": "zero, as t-route initial state"},
        "dropped_lakes": g.dropped, "data_license": c.data_license, "time_origin": c.time_origin});
    manifest["timings_ms"] = json!({"collapse_levels": graph_ms, "pack": pack_ms, "write_hash": write.elapsed().as_secs_f64() * 1000.,
        "total": start.elapsed().as_secs_f64() * 1000.});
    manifest["parallelism"] = json!({"rayon_threads": rayon::current_num_threads(), "graph": "1 thread startup dependency traversal",
        "parameter_pack": "parallel node rows", "payload_write_hash": "parallel arrays", "gpu": "resident payloads; no per-step preprocessing"});
    fs::write(out.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(json!({"manifest": out.join("manifest.json"), "nodes": n, "mc": mc.len(), "reservoirs": lake_ids.len(),
              "levels": level_offsets.len() - 1, "reservoir_levels": selections.len(), "timings_ms": manifest["timings_ms"]}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn channel_source_binds_spinup_end_and_source_geopackage() {
        let origin = "2023-01-01T00:00:00Z";
        let sha = "a".repeat(64);
        let fabric = json!({"input": {"path": "/removed/source.gpkg", "sha256": sha}});
        let m = json!({"schema": "arwen.hydro-world.init-spinup.v1", "end": origin, "fabric_sha256": sha});
        channel_source(&m, &fabric, origin).unwrap();
        for (key, bad) in [("end", json!("2022-12-31T23:00:00Z")), ("end", Value::Null),
            ("end", json!(0)), ("fabric_sha256", json!("b".repeat(64))),
            ("fabric_sha256", Value::Null), ("fabric_sha256", json!([]))] {
            let mut wrong = m.clone(); wrong[key] = bad;
            assert!(channel_source(&wrong, &fabric, origin).is_err(), "{key}");
        }
        for key in ["end", "fabric_sha256"] {
            let mut missing = m.clone(); missing.as_object_mut().unwrap().remove(key);
            assert!(channel_source(&missing, &fabric, origin).is_err(), "{key}");
        }
        assert!(channel_source(&m, &json!({}), origin).is_err());
        assert!(channel_source(&m, &json!({"input": {"sha256": "a"}}), origin).is_err());
        assert!(channel_source(&json!({"end": Value::Null}), &fabric, origin).is_err());
        assert!(channel_source(&json!({"fabric_sha256": Value::Null}), &fabric, origin).is_err());
        channel_source(&json!({}), &json!({}), origin).unwrap(); // unsealed legacy capture
    }
    #[test]
    fn fragment_source_binds_bytes_and_allows_relocation() {
        let fabric = "a".repeat(64);
        let attributes = "b".repeat(64);
        let good = json!({"source_manifest": {"path": "/old/location", "sha256": fabric}, "attributes_sha256": attributes});
        assert!(fragment_source(&good, &fabric, &attributes).is_ok());
        assert!(fragment_source(&json!({}), &fabric, &attributes).is_ok());
        for bad in [json!({"source_manifest": null}), json!({"source_manifest": {}}),
                    json!({"source_manifest": {"sha256": "c".repeat(64)}}),
                    json!({"attributes_sha256": null}), json!({"attributes_sha256": "c".repeat(64)})] {
            assert!(fragment_source(&bad, &fabric, &attributes).is_err());
        }
    }
    #[test]
    fn raw_mc_layout_checked_before_parallel_row_lookup() {
        let ids: Vec<u8> = [1i64, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let values = [0f32, -0., 1. / 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let i = json!({"shape": [2], "dtype": "<i8", "bytes": 16});
        let p = json!({"shape": [2, 8], "dtype": "<f4", "bytes": 64});
        let (got_ids, got_values) = raw_routing_table(&ids, &i, &bytes, &p).unwrap();
        assert_eq!(got_ids, [1, 2]);
        assert_eq!(got_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), values.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        for size in [0, 31, 63] { assert!(raw_routing_table(&ids, &i, &bytes[..size], &p).is_err()); }
        assert!(raw_routing_table(&ids[..15], &i, &bytes, &p).is_err());
        for (key, bad) in [("dtype", json!("<f8")), ("shape", json!([2, 1])), ("bytes", Value::Null)] {
            let mut field = i.clone(); field[key] = bad;
            assert!(raw_routing_table(&ids, &field, &bytes, &p).is_err());
        }
        let duplicate: Vec<u8> = [1i64, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(raw_routing_table(&duplicate, &i, &bytes, &p).is_err());
        assert!(raw_routing_table(&ids, &i, &[bytes.as_slice(), &[0]].concat(), &p).is_err());
        let mut huge = i.clone(); huge["shape"] = json!([u64::MAX]);
        let mut huge_p = p.clone(); huge_p["shape"] = json!([u64::MAX, 8]);
        assert!(raw_routing_table(&[], &huge, &[], &huge_p).is_err());
    }
    #[test]
    fn reservoir_fragment_layout_rejects_truncation_and_reinterpretation() {
        let ids = 42i64.to_le_bytes();
        let id_field = json!({"dtype": "<i8", "shape": [1]}); // actual producer ABI, no bytes member
        fragment_vector(&ids, &id_field, 1, "<i8", 8, "reservoir_ids").unwrap();
        for (key, bad) in [("dtype", json!("<f8")), ("shape", json!([1, 1])), ("bytes", json!(7)), ("bytes", Value::Null)] {
            let mut field = id_field.clone();
            field[key] = bad;
            assert!(fragment_vector(&ids, &field, 1, "<i8", 8, "reservoir_ids").is_err(), "{key}");
        }
        assert!(fragment_vector(&ids[..7], &id_field, 1, "<i8", 8, "reservoir_ids").is_err());
        assert!(fragment_vector(&[ids.as_slice(), &[0]].concat(), &id_field, 1, "<i8", 8, "reservoir_ids").is_err());
        for dtype in ["<f8", "<f4"] {
            let values = [3.0f64, -0.0, 1.0 / 3.0];
            let bytes: Vec<u8> = if dtype == "<f8" { values.iter().flat_map(|v| v.to_le_bytes()).collect() }
                else { values.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect() };
            let field = json!({"dtype": dtype, "shape": [3], "bytes": bytes.len()});
            let actual = reservoir_releases(&bytes, &field, 3).unwrap();
            let expected: Vec<f64> = values.iter().map(|v| if dtype == "<f8" { *v } else { *v as f32 as f64 }).collect();
            assert_eq!(actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            let mut tail = bytes.clone();
            tail.push(0);
            let mut tail_field = field.clone();
            tail_field["bytes"] = json!(tail.len());
            assert!(reservoir_releases(&tail, &tail_field, 3).is_err());
            assert!(reservoir_releases(&bytes[..bytes.len() - 1], &field, 3).is_err());
            for (key, bad) in [("shape", json!([1, 3])), ("bytes", json!(1)), ("dtype", json!("<i4")), ("dtype", json!(">f4")), ("dtype", Value::Null)] {
                let mut malformed = field.clone();
                malformed[key] = bad;
                assert!(reservoir_releases(&bytes, &malformed, 3).is_err(), "{key}");
            }
        }
    }
    #[test]
    fn channel_axis_seal_binds_order_and_never_falls_back_when_present() {
        let bytes = br#"["wb-1","wb-2"]"#;
        let seal = json!({"file": "routing_ids.json", "count": 2, "sha256": format!("{:x}", Sha256::digest(bytes))});
        let ids = channel_ids(bytes, Some(&seal)).unwrap();
        assert_eq!(ids, vec!["wb-1", "wb-2"]);
        assert_eq!(ids, channel_ids(bytes, None).unwrap()); // legacy packet
        assert!(channel_ids(br#"["wb-2","wb-1"]"#, Some(&seal)).unwrap_err().to_string().contains("SHA-256"));
        for (key, bad) in [("file", json!("other.json")), ("count", json!(3)), ("count", Value::Null), ("sha256", Value::Null)] {
            let mut malformed = seal.clone();
            malformed[key] = bad;
            assert!(channel_ids(bytes, Some(&malformed)).is_err(), "{key}");
        }
        assert!(channel_ids(bytes, Some(&Value::Null)).is_err());
        assert!(channel_ids(bytes, Some(&json!({}))).is_err());
        assert!(channel_ids(br#"[1,2]"#, None).is_err());
    }
    #[test]
    fn routing_narrowing_keeps_finite_words_and_rejects_overflow() {
        for value in [0.0f64, -0.0, 1.0 / 3.0, f32::MAX as f64, -(f32::MAX as f64), f32::MIN_POSITIVE as f64, f64::MIN_POSITIVE] {
            assert_eq!(routing_f32(value, 42, "discharge").unwrap().to_bits(), (value as f32).to_bits());
        }
        for value in [f64::MAX, -f64::MAX, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = routing_f32(value, 42, "reservoir release").unwrap_err().to_string();
            assert!(err.contains("42") && err.contains("reservoir release") && err.contains("<f4"));
        }
    }
    #[test]
    fn channel_layout_preserves_valid_words_and_rejects_malformed_transport() {
        let values = [1.0f64, -0.0, 1.5];
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let field = json!({"dtype": "<f8", "shape": [1, 3], "bytes": 24});
        let decoded = channel_values(&bytes, &field, 1).unwrap();
        assert_eq!(decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        let mut trailing = bytes.clone();
        trailing.push(0x7f);
        let mut self_consistent_extra_byte = field.clone();
        self_consistent_extra_byte["bytes"] = json!(25);
        assert!(channel_values(&trailing, &self_consistent_extra_byte, 1).is_err());
        assert!(channel_values(&bytes[..23], &field, 1).is_err());
        for (key, bad) in [("shape", json!([3, 1])), ("dtype", json!("<f4")), ("bytes", json!(23))] {
            let mut malformed = field.clone();
            malformed[key] = bad;
            assert!(channel_values(&bytes, &malformed, 1).is_err(), "{key}");
        }
        assert!(channel_values(&bytes, &field, 2).is_err());
    }
    fn lake() -> LakeParams {
        LakeParams { weir_e: 2.5, max_e: 2.77, weir_c: 0.4, weir_l: 10.0, dam_l: 10.0, orifice_e: 0.97, orifice_c: 0.1, orifice_a: 1.0 }
    }
    #[test]
    fn stage_inverts_release_across_all_three_regimes() {
        let p = lake();
        for q in [0.0, 0.3, 1.0, 5.0, 300.0] {
            let h = stage_for(q, &p).unwrap();
            assert!(h >= p.orifice_e);
            assert!((release(h, &p) - q).abs() <= 1e-9 * q.max(1.0), "q={q} h={h} r={}", release(h, &p));
        }
    }
    #[test]
    fn unreachable_release_refuses_instead_of_writing_nonfinite_water() {
        let mut p = lake();
        p.orifice_c = 0.0;
        p.weir_c = 0.0;
        assert!(stage_for(1.0, &p).unwrap_err().to_string().contains("finite stage"));
    }
}
