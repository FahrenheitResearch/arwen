//! Explicit referee-input transport; not an independent GPKG collapse replay.
//! NOAA-OWP/t-route 12a8eae (Apache-2.0) capture schema. Reservoir numerical
//! initialization is imported from the reservoir owner's sealed Rust fragment.
use crate::{bundle::digest, network::levels, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
const PIN: &str = "12a8eae0cdfed437143c590659fa7077605a5e70";
#[derive(Deserialize)]
struct Config {
    capture_manifest: PathBuf,
    fabric_manifest: PathBuf,
    reservoir_manifest: PathBuf,
    output: PathBuf,
    time_origin: String,
    data_license: String,
}
struct Input {
    root: PathBuf,
    manifest: Value,
}
impl Input {
    fn bytes(&self, key: &str) -> Result<Vec<u8>> {
        let f = &self.manifest["arrays"][key];
        let rel = f["path"]
            .as_str()
            .ok_or_else(|| format!("missing capture column {key}"))?;
        let path = self.root.join(rel).canonicalize()?;
        if !path.starts_with(self.root.canonicalize()?) {
            return Err("capture path escapes root".into());
        }
        if digest(&path)? != f["sha256"].as_str().ok_or("missing capture hash")? {
            return Err(format!("capture SHA mismatch: {key}").into());
        }
        let b = fs::read(path)?;
        let width = match f["dtype"].as_str().ok_or("missing capture dtype")? {
            "<i8" | "<f8" => 8,
            "<f4" => 4,
            _ => return Err("unsupported capture dtype".into()),
        };
        if f["shape"].as_array().ok_or("missing shape")?.len() != 1
            || f["strides"] != json!([width])
            || b.len() != f["shape"][0].as_u64().ok_or("bad shape")? as usize * width
        {
            return Err(format!("non-contiguous capture payload {key}").into());
        }
        Ok(b)
    }
    fn ids(&self, key: &str) -> Result<Vec<i64>> {
        if self.manifest["arrays"][key]["dtype"] != "<i8" {
            return Err("ID column must be int64".into());
        }
        Ok(self
            .bytes(key)?
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn numbers(&self, key: &str) -> Result<Vec<f32>> {
        let bytes = self.bytes(key)?;
        match self.manifest["arrays"][key]["dtype"].as_str() {
            Some("<f8") => Ok(bytes
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect()),
            Some("<f4") => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()),
            _ => Err("numerical column must be f32/f64".into()),
        }
    }
    fn graph(&self, key: &str) -> Result<Value> {
        let f = &self.manifest[key];
        let path = self
            .root
            .join(f["path"].as_str().ok_or("missing graph path")?);
        if digest(&path)? != f["sha256"].as_str().ok_or("missing graph SHA")? {
            return Err("graph SHA mismatch".into());
        }
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }
}
struct Array {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
macro_rules! payload {
    ($v:expr,$name:expr,$dtype:literal) => {
        Array {
            name: $name.into(),
            dtype: $dtype,
            shape: vec![$v.len()],
            bytes: $v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    };
}
fn id_map(ids: &[i64]) -> Result<HashMap<i64, usize>> {
    let m = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect::<HashMap<_, _>>();
    if m.len() != ids.len() {
        return Err("duplicate row IDs".into());
    }
    Ok(m)
}
fn list(value: &Value) -> Result<Vec<i64>> {
    value
        .as_array()
        .ok_or("expected native edge list")?
        .iter()
        .map(|v| v.as_i64().ok_or_else(|| "non-i64 node ID".into()))
        .collect()
}
fn checked_source(path: &Path, field: &Value) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&bytes))
        != field["sha256"].as_str().ok_or("missing fabric SHA")?
    {
        return Err("fabric SHA mismatch".into());
    }
    Ok(bytes)
}
pub fn prepare(config_path: &Path) -> Result<Value> {
    let start = Instant::now();
    let config_bytes = fs::read(config_path)?;
    let c: Config = serde_json::from_slice(&config_bytes)?;
    if c.data_license.trim().is_empty() || c.time_origin.trim().is_empty() {
        return Err("explicit data_license/time_origin required".into());
    }
    let input = Input {
        root: c.capture_manifest.parent().ok_or("capture parent")?.into(),
        manifest: serde_json::from_slice(&fs::read(&c.capture_manifest)?)?,
    };
    if input.manifest["schema"] != "troute-native-network-observation-v1" {
        return Err("unsupported capture schema".into());
    }
    // This changes within-reach upstream discharge selection, so omitting it
    // silently changes the trajectory even with identical graph/array bytes.
    let assume_short_ts = input.manifest["assume_short_ts"]
        .as_bool()
        .ok_or("capture must include the actual compute assume_short_ts boolean")?;
    let graph = input.graph("network")?;
    let compute = input.graph("compute-network")?;
    for key in [
        "connections",
        "reverse_network",
        "waterbody_connections",
        "reaches_by_tailwater",
        "independent_networks",
    ] {
        if graph[key].is_null() || graph[key] != compute[key] {
            return Err(format!("native graph changed before compute: {key}").into());
        }
    }
    // ID-sorted common storage matches native binary_find. Incoming lists are
    // NEVER sorted: their exact order controls floating-point upstream sums.
    let connections = graph["connections"]
        .as_object()
        .ok_or("missing connections")?;
    let mut ids = connections
        .keys()
        .map(|k| k.parse::<i64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ids.sort_unstable();
    let mut independent_ids = BTreeSet::new();
    for net in graph["independent_networks"]
        .as_object()
        .ok_or("missing independent networks")?
        .values()
    {
        for (id, incoming) in net.as_object().ok_or("bad independent network")? {
            if !independent_ids.insert(id.parse::<i64>()?)
                || *incoming != graph["reverse_network"][id]
            {
                return Err(
                    "actual native compute incoming order differs from reverse network".into(),
                );
            }
        }
    }
    if independent_ids != ids.iter().copied().collect::<BTreeSet<_>>() {
        return Err("actual native compute roster differs from collapsed graph".into());
    }
    let index = id_map(&ids)?;
    let n = ids.len();
    if n > i32::MAX as usize {
        return Err("routing roster exceeds int32".into());
    }
    let mut upstream = Vec::with_capacity(n);
    let mut downstream = Vec::with_capacity(n);
    let mut offsets = vec![0i32];
    let mut indices = Vec::<i32>::new();
    for id in &ids {
        let k = id.to_string();
        let ds = list(&graph["connections"][&k])?;
        if ds.len() > 1 {
            return Err("MC graph has divergent downstream edges".into());
        }
        downstream.push(if let Some(id) = ds.first() {
            i32::try_from(*index.get(id).ok_or("downstream outside graph")?)?
        } else {
            -1
        });
        let up = list(&graph["reverse_network"][&k])?
            .iter()
            .map(|id| {
                index
                    .get(id)
                    .copied()
                    .ok_or_else(|| "upstream outside graph".into())
                    .and_then(|i| Ok(i32::try_from(i)?))
            })
            .collect::<Result<Vec<_>>>()?;
        if up.iter().collect::<BTreeSet<_>>().len() != up.len() {
            return Err("duplicate incoming edge".into());
        }
        indices.extend(&up);
        offsets.push(i32::try_from(indices.len())?);
        upstream.push(up);
    }
    // Check both representations, including absence of unrepresented edges.
    let mut edge_count = 0;
    for (i, up) in upstream.iter().enumerate() {
        for &u in up {
            if downstream[u as usize] != i as i32 {
                return Err("native forward/reverse edge disagreement".into());
            }
            edge_count += 1;
        }
    }
    if edge_count != downstream.iter().filter(|&&d| d >= 0).count() {
        return Err("native forward edge missing from reverse graph".into());
    }
    let (ordered, level_offsets) = levels(&upstream, &downstream, false)?;
    let (reverse_ordered, reverse_offsets) = levels(&upstream, &downstream, true)?;
    let lake_ids = input.ids("waterbody-parameters-row-index")?;
    let lake_index = id_map(&lake_ids)?;
    let r = lake_ids.len();
    if input.ids("waterbody-types-row-index")? != lake_ids {
        return Err("waterbody type row order differs".into());
    }
    let types = input.ids("waterbody-types-column-reservoir_type")?;
    if types.iter().any(|&t| t != 1) {
        return Err(
            "capture contains DA reservoirs: explicit DA controls/series capture required".into(),
        );
    }
    let reservoir_nodes = lake_ids
        .iter()
        .map(|id| {
            Ok(i32::try_from(
                *index
                    .get(id)
                    .ok_or("reservoir missing from collapsed graph")?,
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    let native_output_ids = ids
        .iter()
        .map(|id| -> Result<i64> {
            if lake_index.contains_key(id) {
                graph["link_lake_crosswalk"][id.to_string()]
                    .as_i64()
                    .ok_or_else(|| "reservoir output crosswalk missing".into())
            } else {
                Ok(*id)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    id_map(&native_output_ids)?;
    if lake_index.len()
        != graph["link_lake_crosswalk"]
            .as_object()
            .ok_or("missing lake crosswalk")?
            .len()
    {
        return Err("reservoir/crosswalk roster disagreement".into());
    }
    for id in &lake_ids {
        if graph["link_lake_crosswalk"][id.to_string()]
            .as_i64()
            .is_none()
        {
            return Err("reservoir crosswalk missing".into());
        }
    }
    let mut mc = Vec::new();
    let mut mc_offsets = vec![0i32];
    let mut selections = Vec::new();
    let mut arrays = Vec::<Array>::new();
    for (level, range) in level_offsets.windows(2).enumerate() {
        let mut rows = Vec::<i32>::new();
        for &node in &ordered[range[0] as usize..range[1] as usize] {
            if let Some(&row) = lake_index.get(&ids[node as usize]) {
                rows.push(i32::try_from(row)?);
            } else {
                mc.push(node);
            }
        }
        mc_offsets.push(i32::try_from(mc.len())?);
        if !rows.is_empty() {
            let name = format!("reservoir_level_{level}_rows");
            arrays.push(payload!(rows, name.clone(), "<i4"));
            selections.push(json!({"level":level,"rows":name}));
        }
    }
    let graph_ms = start.elapsed().as_secs_f64() * 1000.;
    let pack = Instant::now();
    let param_ids = input.ids("network-parameters-row-index")?;
    let param_index = id_map(&param_ids)?;
    let parameter_columns = ["dx", "bw", "tw", "twcc", "n", "ncc", "cs", "s0"];
    let columns = parameter_columns
        .iter()
        .map(|s| input.numbers(&format!("network-parameters-column-{s}")))
        .collect::<Result<Vec<_>>>()?;
    if columns.iter().any(|v| v.len() != param_ids.len()) {
        return Err("parameter column length differs".into());
    }
    let parameters = ids
        .par_iter()
        .map(|id| -> Result<[f32; 8]> {
            if lake_index.contains_key(id) {
                return Ok([0.; 8]);
            }
            let row = *param_index.get(id).ok_or("MC parameter row missing")?;
            let a = std::array::from_fn(|j| columns[j][row]);
            if a.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite MC parameter".into());
            }
            Ok(a)
        })
        .collect::<Result<Vec<_>>>()?;
    let initial_ids = input.ids("initial-row-index")?;
    let initial_index = id_map(&initial_ids)?;
    let q0 = input.numbers("initial-column-qu0")?;
    let h0 = input.numbers("initial-column-h0")?;
    if q0.len() != initial_ids.len() || h0.len() != initial_ids.len() {
        return Err("initial column length differs".into());
    }
    let lateral_ids = input.ids("lateral-row-index")?;
    let lateral_index = id_map(&lateral_ids)?;
    let lateral_keys = input.manifest["arrays"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|s| s.starts_with("lateral-column-"))
        .cloned()
        .collect::<Vec<_>>();
    if lateral_keys.len() != 1 {
        return Err("replay preparer requires one native lateral interval; resident schedule owner handles multiple intervals".into());
    }
    let lat = input.numbers(&lateral_keys[0])?;
    if lat.len() != lateral_ids.len() {
        return Err("lateral column length differs".into());
    }
    let mut state = vec![[0f32; 3]; n];
    let mut lateral = vec![0f32; n];
    for (node, id) in ids.iter().enumerate() {
        if !lake_index.contains_key(id) {
            let j = *initial_index.get(id).ok_or("initial MC row missing")?;
            state[node] = [q0[j], 0., h0[j]];
            lateral[node] = lat[*lateral_index.get(id).ok_or("lateral MC row missing")?];
        }
    }
    let release0 = input.numbers("waterbody-parameters-column-qd0")?;
    if release0.len() != r {
        return Err("reservoir initial release rows differ".into());
    }
    for row in 0..r {
        state[reservoir_nodes[row] as usize] = [release0[row], 0., 0.];
    }
    let reservoir_fragment: Value = serde_json::from_slice(&fs::read(&c.reservoir_manifest)?)?;
    let reservoir_root = c
        .reservoir_manifest
        .parent()
        .ok_or("reservoir fragment parent")?;
    if reservoir_fragment["source_manifest"]["sha256"] != digest(&c.capture_manifest)?
        || reservoir_fragment["reservoirs"]["source_pin"] != PIN
        || reservoir_fragment["reservoirs"]["time_origin"] != c.time_origin
        || reservoir_fragment["reservoirs"]["validated_csr"] != true
    {
        return Err("reservoir fragment source/time/CSR identity differs".into());
    }
    let fragment_id_field = &reservoir_fragment["reservoir_ids"];
    let fragment_ids = checked_source(
        &reservoir_root.join(
            fragment_id_field["path"]
                .as_str()
                .ok_or("reservoir IDs path")?,
        ),
        fragment_id_field,
    )?;
    if fragment_ids != input.bytes("waterbody-parameters-row-index")? {
        return Err("reservoir fragment order differs from native graph".into());
    }
    let fabric: Value = serde_json::from_slice(&fs::read(&c.fabric_manifest)?)?;
    let fabric_root = c.fabric_manifest.parent().ok_or("fabric parent")?;
    let old_network_bytes = checked_source(
        &fabric_root.join("network.json"),
        &fabric["auxiliary"]["network.json"],
    )?;
    let mut network: Value = serde_json::from_slice(&old_network_bytes)?;
    let old_ids = network["numeric_node_ids"]
        .as_array()
        .ok_or("fabric requires unambiguous numeric IDs")?
        .iter()
        .map(|v| v.as_i64().ok_or_else(|| "bad numeric fabric ID".into()))
        .collect::<Result<Vec<_>>>()?;
    let remap = old_ids
        .iter()
        .map(|id| -> Result<i32> {
            let target = graph["waterbody_connections"][id.to_string()]
                .as_i64()
                .unwrap_or(*id);
            Ok(i32::try_from(
                *index
                    .get(&target)
                    .ok_or("fabric node not in captured collapsed graph")?,
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    let catchment_to_node = network["catchment_to_node"]
        .as_array()
        .ok_or("missing catchment mapping")?
        .iter()
        .map(|v| -> Result<i32> {
            let i = v.as_u64().ok_or("bad catchment node")? as usize;
            Ok(*remap.get(i).ok_or("catchment node outside fabric")?)
        })
        .collect::<Result<Vec<_>>>()?;
    let hash_capture = digest(&c.capture_manifest)?;
    let mut identity = Sha256::new();
    identity.update(&config_bytes);
    identity.update(hash_capture.as_bytes());
    identity.update(digest(&c.fabric_manifest)?.as_bytes());
    identity.update(digest(&c.reservoir_manifest)?.as_bytes());
    let key = format!("{:x}", identity.finalize());
    let out = c.output.join(&key[..20]);
    fs::create_dir_all(&out)?;
    let mut files = serde_json::Map::new();
    let mut cell_mapping = None;
    // Preserve independent catchment/grid data, remapping only node-indexed raster.
    for (name, field) in fabric["files"].as_object().ok_or("missing fabric files")? {
        if name.starts_with("weights")
            || name.starts_with("exchange")
            || [
                "overlap_m2",
                "catchment_area_m2",
                "coverage_fraction",
                "stream_order",
            ]
            .contains(&name.as_str())
        {
            let bytes = checked_source(
                &fabric_root.join(field["path"].as_str().ok_or("fabric payload path")?),
                field,
            )?;
            let filename = format!("{name}.bin");
            fs::write(out.join(&filename), bytes)?;
            let mut f = field.clone();
            f["path"] = json!(filename);
            files.insert(name.clone(), f);
        } else if name == "channel_node" {
            let b = checked_source(
                &fabric_root.join(field["path"].as_str().ok_or("channel raster path")?),
                field,
            )?;
            let mapped = b
                .chunks_exact(4)
                .map(|b| -> Result<i32> {
                    let i = i32::from_le_bytes(b.try_into().unwrap());
                    if i < 0 {
                        Ok(i)
                    } else {
                        Ok(*remap
                            .get(i as usize)
                            .ok_or("channel raster node outside fabric")?)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let inverse_start = Instant::now();
            if mapped.len() > i32::MAX as usize
                || mapped.len()
                    != fabric["grid"]["nx"].as_u64().ok_or("grid nx missing")? as usize
                        * fabric["grid"]["ny"].as_u64().ok_or("grid ny missing")? as usize
            {
                return Err("channel raster/grid shape differs or exceeds int32 cell index".into());
            }
            // Group actual channel pixels by the CURRENT collapsed node axis.
            // Lexicographic ordering fixes ascending flattened cells per node.
            let mut pairs = mapped
                .par_iter()
                .enumerate()
                .filter_map(|(cell, &node)| (node >= 0).then_some((node, cell as i32)))
                .collect::<Vec<_>>();
            pairs.par_sort_unstable();
            let cell_offsets = (0..=n)
                .into_par_iter()
                .map(|node| pairs.partition_point(|&(id, _)| id < node as i32) as i64)
                .collect::<Vec<_>>();
            let cell_indices = pairs.par_iter().map(|&(_, cell)| cell).collect::<Vec<_>>();
            // Exhaustive actual-raster roundtrip catches stale raw-node axes,
            // duplicate/missing pixels and cell-order changes at this boundary.
            let mut reconstructed = vec![-1; mapped.len()];
            for node in 0..n {
                let row = &cell_indices[cell_offsets[node] as usize..cell_offsets[node + 1] as usize];
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
            cell_mapping = Some(json!({
                "offsets":"routing_cell_offsets", "indices":"routing_cell_indices",
                "node_axis":"node_ids", "source":"channel_node", "cells":cell_indices.len(),
                "grid_sha256":format!("{:x}", Sha256::digest(serde_json::to_vec(&fabric["grid"])?)),
                "cell_order":"ascending flattened x-fast cell index within each current collapsed node",
                "scope":"actual rasterized channel-cell incidence only; nonchannel cells excluded; reservoir rows included where member flowlines rasterize; no drainage or lateral policy inferred",
                "reservoir_lateral_policy":"unbound; producer and receiver must explicitly agree",
                "roundtrip_cells":mapped.len(), "different_cells":0,
                "prepare_ms":inverse_start.elapsed().as_secs_f64()*1000.
            }));
            arrays.push(payload!(cell_offsets, "routing_cell_offsets", "<i8"));
            arrays.push(payload!(cell_indices, "routing_cell_indices", "<i4"));
            let mut a = payload!(mapped, "channel_node", "<i4");
            a.shape = serde_json::from_value(field["shape"].clone())?;
            arrays.push(a);
        }
    }
    arrays.extend([
        payload!(native_output_ids, "output_reach_ids", "<i8"),
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
    ]);
    arrays.push(Array {
        name: "routing_parameters".into(),
        dtype: "<f4",
        shape: vec![n, 8],
        bytes: parameters
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
    });
    arrays.push(Array {
        name: "routing_state".into(),
        dtype: "<f4",
        shape: vec![n, 3],
        bytes: state
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
    });
    for (name, field) in reservoir_fragment["files"]
        .as_object()
        .ok_or("missing reservoir fragment files")?
    {
        if files.contains_key(name) || arrays.iter().any(|a| &a.name == name) {
            return Err("duplicate reservoir payload key".into());
        }
        let bytes = checked_source(
            &reservoir_root.join(field["path"].as_str().ok_or("reservoir payload path")?),
            field,
        )?;
        if bytes.len() != field["bytes"].as_u64().ok_or("reservoir bytes")? as usize {
            return Err("reservoir length mismatch".into());
        }
        let filename = format!("{name}.bin");
        fs::write(out.join(&filename), bytes)?;
        let mut f = field.clone();
        f["path"] = json!(filename);
        files.insert(name.clone(), f);
    }
    let pack_ms = pack.elapsed().as_secs_f64() * 1000.;
    let write = Instant::now();
    let generated=arrays.par_iter().map(|a|->Result<(String,Value)>{let name=format!("{}.bin",a.name);fs::write(out.join(&name),&a.bytes)?;Ok((a.name.clone(),json!({"path":name,"dtype":a.dtype,"shape":a.shape,"bytes":a.bytes.len(),"sha256":format!("{:x}",Sha256::digest(&a.bytes))})))}).collect::<Result<Vec<_>>>()?;
    files.extend(generated);
    network["node_ids"] = json!(ids.iter().map(ToString::to_string).collect::<Vec<_>>());
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
    network["downstream_ids"] = json!(downstream
        .iter()
        .map(|&i| if i < 0 {
            None
        } else {
            Some(ids[i as usize].to_string())
        })
        .collect::<Vec<_>>());
    network["upstream_order"] = json!("native captured reverse_network incoming-list order");
    fs::write(
        out.join("network.json"),
        serde_json::to_vec_pretty(&network)?,
    )?;
    let mut manifest = fabric.clone();
    manifest["output"] = json!(out);
    manifest["request_sha256"] = json!(key);
    manifest["files"] = json!(files);
    manifest["node_count"] = json!(n);
    manifest["level_count"] = json!(level_offsets.len() - 1);
    manifest["node_order"] =
        json!("native collapsed IDs sorted i64; catchments retain original divide order");
    manifest["upstream_order"] = network["upstream_order"].clone();
    manifest["input"] =
        json!({"path":c.capture_manifest,"sha256":hash_capture,"data_license":c.data_license});
    manifest["auxiliary"] = json!({"network.json":{"sha256":digest(&out.join("network.json"))?}});
    manifest["routing"] = json!({"mc_level_offsets":mc_offsets,"reservoir_node_indices":"reservoir_node_indices","reservoir_levels":selections,"validated_reservoir_mapping":true,"parameter_columns":parameter_columns,"native_incoming_order_verified":true,"upstream_csr_edges":indices.len(),"mc_count":mc.len(),"reservoir_count":r,"initial_state_array":"routing_state","lateral_array":"routing_lateral","output_reach_ids":"output_reach_ids","native_output_id_semantics":"native postprocessing lake ID -> captured outlet flowpath; internal node_ids remain collapsed routing IDs","dt_seconds":input.manifest["dt"],"steps":input.manifest["steps"],"lateral_interval_steps":input.manifest["qts_subdivisions"],"lateral_source_column":lateral_keys[0]});
    manifest["reservoirs"] = reservoir_fragment["reservoirs"].clone();
    manifest["routing"]["assume_short_ts"] = json!(assume_short_ts);
    if let Some(mut mapping) = cell_mapping {
        mapping["node_ids_sha256"] = manifest["files"]["node_ids"]["sha256"].clone();
        mapping["channel_node_sha256"] = manifest["files"]["channel_node"]["sha256"].clone();
        manifest["routing"]["cell_mapping"] = mapping;
    }
    manifest["replay_provenance"] = json!({"scope":"captured native graph/input transport, not independent GPKG collapse proof","source_pin":PIN,"capture_manifest_sha256":hash_capture,"fabric_manifest_sha256":digest(&c.fabric_manifest)?,"native_network_and_compute_graph_equal":true,"incoming_edge_order_preserved":true,"all_payload_hashes_checked":true,"reservoir_manifest_sha256":digest(&c.reservoir_manifest)?,"reservoir_initialization_owner":"reservoirs Rust packer","data_scope":"private development/referee only; source-specific fabric data terms unresolved; not a shipped demo"});
    manifest["numerics"]=json!("fast default; explicit strict replay selected by GPUWM_NUMERICS=strict; Rust f64->f32 native call conversions at setup");
    manifest["timings_ms"] = json!({"capture_graph_validate_levels":graph_ms,"typed_pack_and_grid_transport":pack_ms,"write_hash":write.elapsed().as_secs_f64()*1000.,"total":start.elapsed().as_secs_f64()*1000.});
    manifest["parallelism"] = json!({"rayon_threads":rayon::current_num_threads(),"parameter_pack":"parallel native ID rows","payload_write_hash":"parallel arrays","graph":"startup dependency traversal; incoming edge order retained","gpu":"resident payloads; no per-step preprocessing"});
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}
