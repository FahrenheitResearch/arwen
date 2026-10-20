//! Read-only native MC input/order audit; not a second preparer or initializer.
use nextgen_hydrofabric::{bundle::digest, manifest::PreparedManifest, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};

fn check(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn i64s(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn source_bytes(root: &Path, descriptor: &Value) -> Result<Vec<u8>> {
    let root = root.canonicalize()?;
    let path = root
        .join(descriptor["path"].as_str().ok_or("source path missing")?)
        .canonicalize()?;
    check(path.starts_with(&root), "source payload escapes capture")?;
    check(
        digest(&path)? == descriptor["sha256"].as_str().ok_or("source SHA missing")?,
        "source SHA mismatch",
    )?;
    Ok(fs::read(path)?)
}
fn column(root: &Path, capture: &Value, name: &str) -> Result<Vec<u8>> {
    let d = &capture["arrays"][name];
    let b = source_bytes(root, d)?;
    let width = match d["dtype"].as_str() {
        Some("<i8" | "<f8") => 8,
        Some("<f4") => 4,
        _ => return Err("unsupported native capture column dtype".into()),
    };
    let shape = d["shape"].as_array().ok_or("source shape missing")?;
    check(
        shape.len() == 1
            && shape[0].as_u64() == Some((b.len() / width) as u64)
            && b.len() % width == 0
            && d["strides"] == json!([width]),
        "bad native column shape/strides",
    )?;
    Ok(b)
}
fn floats(capture: &Value, raw: &BTreeMap<String, Vec<u8>>, name: &str) -> Result<Vec<f32>> {
    match capture["arrays"][name]["dtype"].as_str() {
        Some("<f4") => Ok(words(&raw[name]).into_iter().map(f32::from_bits).collect()),
        Some("<f8") => Ok(raw[name]
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect()),
        _ => Err("native numerical column is not f32/f64".into()),
    }
}
fn id_index(
    capture: &Value,
    raw: &BTreeMap<String, Vec<u8>>,
    name: &str,
) -> Result<BTreeMap<i64, usize>> {
    check(
        capture["arrays"][name]["dtype"] == "<i8",
        "native IDs must remain int64",
    )?;
    let ids = i64s(&raw[name]);
    let map: BTreeMap<_, _> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    check(map.len() == ids.len(), "duplicate native column IDs")?;
    Ok(map)
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or("mc_contract PREPARED_MANIFEST.json")?;
    let start = Instant::now();
    let m = PreparedManifest::open(Path::new(&path))?;
    let n = m.document.node_count;
    let routing = &m.document.metadata["routing"];
    let capture_path = Path::new(&m.document.input.path);
    check(
        digest(capture_path)? == m.document.input.sha256,
        "capture manifest authority mismatch",
    )?;
    let capture: Value = serde_json::from_slice(&fs::read(capture_path)?)?;
    check(
        capture["schema"] == "troute-native-network-observation-v1",
        "not native MC capture",
    )?;
    check(
        capture["assume_short_ts"] == routing["assume_short_ts"],
        "native upstream time mode changed",
    )?;
    let root = capture_path.parent().ok_or("capture parent missing")?;
    let graph: Value = serde_json::from_slice(&source_bytes(root, &capture["network"])?)?;
    let compute: Value = serde_json::from_slice(&source_bytes(root, &capture["compute-network"])?)?;
    for key in [
        "connections",
        "reverse_network",
        "waterbody_connections",
        "reaches_by_tailwater",
        "independent_networks",
    ] {
        check(
            !graph[key].is_null() && graph[key] == compute[key],
            "captured graph changed before compute",
        )?;
    }
    let mut fields: BTreeMap<String, (&str, Vec<usize>)> = [
        ("node_ids", "<i8", vec![n]),
        ("upstream_offsets", "<i4", vec![n + 1]),
        (
            "upstream_indices",
            "<i4",
            vec![routing["upstream_csr_edges"]
                .as_u64()
                .ok_or("edge count missing")? as usize],
        ),
        ("ordered_nodes", "<i4", vec![n]),
        ("level_offsets", "<i4", vec![m.document.level_count + 1]),
        (
            "mc_ordered_nodes",
            "<i4",
            vec![routing["mc_count"].as_u64().ok_or("MC count missing")? as usize],
        ),
        (
            "reservoir_node_indices",
            "<i4",
            vec![routing["reservoir_count"]
                .as_u64()
                .ok_or("lake count missing")? as usize],
        ),
        (
            "reservoir_ids",
            "<i8",
            vec![routing["reservoir_count"].as_u64().unwrap() as usize],
        ),
        ("routing_parameters", "<f4", vec![n, 8]),
        ("routing_state", "<f4", vec![n, 3]),
        ("routing_lateral", "<f4", vec![n]),
    ]
    .into_iter()
    .map(|(k, d, s)| (k.into(), (d, s)))
    .collect();
    let selections = routing["reservoir_levels"]
        .as_array()
        .ok_or("reservoir level selections missing")?;
    for selection in selections {
        let name = selection["rows"]
            .as_str()
            .ok_or("reservoir selection name missing")?;
        let d = m.document.files.get(name).ok_or("selection field absent")?;
        check(
            d.shape.len() == 1 && !fields.contains_key(name),
            "bad or duplicate selection field",
        )?;
        fields.insert(name.into(), ("<i4", d.shape.clone()));
    }
    let raw: BTreeMap<String, Vec<u8>> = fields
        .par_iter()
        .map(|(name, (dtype, shape))| -> Result<_> {
            let d = m.document.files.get(name).ok_or("MC field missing")?;
            check(
                d.dtype == *dtype && d.shape == *shape,
                "MC dtype/shape mismatch",
            )?;
            let mut b = vec![0; d.bytes];
            m.read_field_into(name, &mut b)?;
            Ok((name.clone(), b))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .collect();
    let parameters = ["dx", "bw", "tw", "twcc", "n", "ncc", "cs", "s0"];
    check(
        routing["parameter_columns"] == json!(parameters),
        "parameter column order changed",
    )?;
    let lateral_name = routing["lateral_source_column"]
        .as_str()
        .ok_or("native lateral source missing")?;
    let mut columns = parameters
        .iter()
        .map(|p| format!("network-parameters-column-{p}"))
        .collect::<Vec<_>>();
    columns.extend(
        [
            "network-parameters-row-index",
            "initial-row-index",
            "initial-column-qu0",
            "initial-column-h0",
            "lateral-row-index",
            lateral_name,
            "waterbody-parameters-row-index",
            "waterbody-parameters-column-qd0",
        ]
        .into_iter()
        .map(String::from),
    );
    let native: BTreeMap<String, Vec<u8>> = columns
        .par_iter()
        .map(|name| Ok((name.clone(), column(root, &capture, name)?)))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .collect();
    let load_hash_ms = start.elapsed().as_secs_f64() * 1000.;
    let validation = Instant::now();
    let ids = i64s(&raw["node_ids"]);
    check(
        ids.windows(2).all(|w| w[0] < w[1]),
        "native common-node axis is not unique/sorted",
    )?;
    let offsets = i32s(&raw["upstream_offsets"]);
    let incoming = i32s(&raw["upstream_indices"]);
    check(
        offsets.first() == Some(&0)
            && offsets.last() == Some(&(incoming.len() as i32))
            && offsets.windows(2).all(|w| w[0] >= 0 && w[0] <= w[1]),
        "bad incoming CSR offsets",
    )?;
    let mut compute_incoming = BTreeMap::new();
    for network in compute["independent_networks"]
        .as_object()
        .ok_or("compute networks missing")?
        .values()
    {
        for (id, up) in network.as_object().ok_or("bad compute network")? {
            check(
                compute_incoming.insert(id.parse::<i64>()?, up).is_none(),
                "duplicate compute node",
            )?;
        }
    }
    check(
        compute_incoming.keys().copied().collect::<Vec<_>>() == ids,
        "native compute and prepared IDs differ",
    )?;
    let order_mismatches: usize = (0..n)
        .into_par_iter()
        .map(|node| -> Result<usize> {
            let expected = compute_incoming[&ids[node]]
                .as_array()
                .ok_or("native incoming list missing")?;
            let actual = &incoming[offsets[node] as usize..offsets[node + 1] as usize];
            check(
                expected.len() == actual.len(),
                "native incoming list length differs",
            )?;
            let mut different = 0;
            for (&index, id) in actual.iter().zip(expected) {
                let actual_id = ids
                    .get(usize::try_from(index)?)
                    .ok_or("incoming node outside axis")?;
                different += usize::from(Some(*actual_id) != id.as_i64());
            }
            Ok(different)
        })
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum();
    check(
        order_mismatches == 0,
        "native upstream reduction order changed",
    )?;
    let lake_index = id_index(&capture, &native, "waterbody-parameters-row-index")?;
    let lake_ids = i64s(&native["waterbody-parameters-row-index"]);
    check(
        i64s(&raw["reservoir_ids"]) == lake_ids,
        "native reservoir row order changed",
    )?;
    let lake_nodes = i32s(&raw["reservoir_node_indices"]);
    for (row, &node) in lake_nodes.iter().enumerate() {
        check(
            ids.get(usize::try_from(node)?) == Some(&lake_ids[row]),
            "reservoir row-to-common-node map differs",
        )?;
    }
    let order = i32s(&raw["ordered_nodes"]);
    let levels = i32s(&raw["level_offsets"]);
    check(
        levels.first() == Some(&0)
            && levels.last() == Some(&(n as i32))
            && levels.windows(2).all(|w| w[0] >= 0 && w[0] < w[1]),
        "bad full level ranges",
    )?;
    let mc = i32s(&raw["mc_ordered_nodes"]);
    let mc_levels: Vec<usize> = serde_json::from_value(routing["mc_level_offsets"].clone())?;
    check(
        mc_levels.len() == levels.len()
            && mc_levels.first() == Some(&0)
            && mc_levels.last() == Some(&mc.len())
            && mc_levels.windows(2).all(|w| w[0] <= w[1]),
        "bad MC level ranges",
    )?;
    let mut seen_nodes = vec![false; n];
    let mut seen_rows = vec![false; lake_ids.len()];
    for (level, range) in levels.windows(2).enumerate() {
        let mut expected_mc = Vec::new();
        let mut expected_rows = Vec::new();
        for &node in &order[range[0] as usize..range[1] as usize] {
            let node_index = usize::try_from(node)?;
            let seen = seen_nodes
                .get_mut(node_index)
                .ok_or("full level node outside axis")?;
            check(!*seen, "duplicate full level node")?;
            *seen = true;
            if let Some(&row) = lake_index.get(&ids[node_index]) {
                expected_rows.push(row as i32);
            } else {
                expected_mc.push(node);
            }
        }
        check(
            mc_levels[level + 1] <= mc.len()
                && mc[mc_levels[level]..mc_levels[level + 1]] == expected_mc,
            "MC roster/ranges include reservoir or shift level",
        )?;
        let selected = selections
            .iter()
            .filter(|s| s["level"].as_u64() == Some(level as u64))
            .collect::<Vec<_>>();
        check(
            selected.len() == usize::from(!expected_rows.is_empty()),
            "missing/duplicate/extraneous reservoir hook",
        )?;
        if let Some(s) = selected.first() {
            let rows = i32s(&raw[s["rows"].as_str().unwrap()]);
            check(
                rows == expected_rows,
                "reservoir selection rows/order differ",
            )?;
            for row in rows {
                check(!seen_rows[row as usize], "reservoir selected twice")?;
                seen_rows[row as usize] = true;
            }
        }
    }
    check(
        seen_nodes.iter().all(|&b| b)
            && seen_rows.iter().all(|&b| b)
            && selections.iter().all(|s| {
                s["level"]
                    .as_u64()
                    .is_some_and(|l| l < m.document.level_count as u64)
            }),
        "incomplete node/hook coverage",
    )?;
    check(
        routing["validated_reservoir_mapping"] == true,
        "producer did not seal reservoir mapping",
    )?;
    let param_index = id_index(&capture, &native, "network-parameters-row-index")?;
    let initial_index = id_index(&capture, &native, "initial-row-index")?;
    let lateral_index = id_index(&capture, &native, "lateral-row-index")?;
    let parameter_values = parameters
        .iter()
        .map(|p| floats(&capture, &native, &format!("network-parameters-column-{p}")))
        .collect::<Result<Vec<_>>>()?;
    let q0 = floats(&capture, &native, "initial-column-qu0")?;
    let h0 = floats(&capture, &native, "initial-column-h0")?;
    let lat = floats(&capture, &native, lateral_name)?;
    let qd0 = floats(&capture, &native, "waterbody-parameters-column-qd0")?;
    let actual_params = words(&raw["routing_parameters"]);
    let actual_state = words(&raw["routing_state"]);
    let actual_lateral = words(&raw["routing_lateral"]);
    let different_words: usize = ids
        .par_iter()
        .enumerate()
        .map(|(node, id)| -> Result<usize> {
            let (expected_p, expected_s, expected_l) = if let Some(&row) = lake_index.get(id) {
                (
                    [0.; 8],
                    [
                        *qd0.get(row).ok_or("reservoir release row missing")?,
                        0.,
                        0.,
                    ],
                    0.,
                )
            } else {
                let p = *param_index.get(id).ok_or("native MC parameter ID absent")?;
                let i = *initial_index.get(id).ok_or("native MC initial ID absent")?;
                let l = *lateral_index.get(id).ok_or("native MC lateral ID absent")?;
                let mut values = [0.; 8];
                for j in 0..8 {
                    values[j] = *parameter_values[j]
                        .get(p)
                        .ok_or("native parameter column length differs")?;
                }
                (
                    values,
                    [
                        *q0.get(i).ok_or("native q row missing")?,
                        0.,
                        *h0.get(i).ok_or("native depth row missing")?,
                    ],
                    *lat.get(l).ok_or("native lateral row missing")?,
                )
            };
            Ok(expected_p
                .iter()
                .enumerate()
                .filter(|(j, v)| actual_params[node * 8 + j] != v.to_bits())
                .count()
                + expected_s
                    .iter()
                    .enumerate()
                    .filter(|(j, v)| actual_state[node * 3 + j] != v.to_bits())
                    .count()
                + usize::from(actual_lateral[node] != expected_l.to_bits()))
        })
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum();
    check(
        different_words == 0,
        "native MC setup parameter/state/lateral words differ",
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "source":nextgen_hydrofabric::SOURCE_REVISION,"manifest":path,"manifest_sha256":digest(Path::new(&path))?,
            "capture_manifest":capture_path,"capture_manifest_sha256":m.document.input.sha256,
            "pass":true,"node_count":n,"mc_count":mc.len(),"reservoir_count":lake_ids.len(),
            "upstream_edges":incoming.len(),"different_ordered_upstream_ids":order_mismatches,
            "levels":m.document.level_count,"reservoir_hooks":selections.len(),"empty_mc_levels":mc_levels.windows(2).filter(|w|w[0]==w[1]).count(),
            "parameter_state_lateral_words":n*12,"different_words":different_words,
            "prepared_fields":raw.len(),"prepared_bytes":raw.values().map(Vec::len).sum::<usize>(),
            "source_columns":native.len(),"source_column_bytes":native.values().map(Vec::len).sum::<usize>(),
            "load_hash_ms":load_hash_ms,"validation_ms":validation.elapsed().as_secs_f64()*1000.,
            "total_ms":start.elapsed().as_secs_f64()*1000.,"rayon_workers":rayon::current_num_threads(),
            "scope":"native captured setup/order/hook replay; not independent GPKG collapse, numerical MC replay, user initialization, coupled run or GPU speed"
        }))?
    );
    Ok(())
}
