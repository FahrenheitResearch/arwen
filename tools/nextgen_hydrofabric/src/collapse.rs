//! Changed from NOAA-OWP/t-route 12a8eae (Apache-2.0), HYFeaturesNetwork.py
//! preprocess_waterbodies/bandaid and nhd_network.py replace_waterbodies_connections.
//! Rust reconstructs from independently parsed GPKG tables, preserving insertion
//! order explicitly. Source-specific ID corrections are table rows, never code.
use crate::{bundle::digest, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::Path,
    time::Instant,
};
fn number(v: &Value) -> Option<i64> {
    v.as_i64()
        .or_else(|| {
            v.as_f64()
                .filter(|x| x.is_finite() && x.fract() == 0.)
                .map(|x| x as i64)
        })
        .or_else(|| v.as_str().and_then(|x| x.parse().ok()))
}
fn suffix(v: &Value) -> Result<i64> {
    v.as_str()
        .and_then(|s| s.rsplit('-').next())
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "invalid native numeric ID".into())
}
fn payload(root: &Path, field: &Value) -> Result<Vec<u8>> {
    let path = root.join(field["path"].as_str().ok_or("payload path")?);
    let b = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&b)) != field["sha256"].as_str().ok_or("payload SHA")? {
        return Err("payload hash mismatch".into());
    }
    Ok(b)
}
pub fn replay(fabric_path: &Path, capture_path: &Path, output: &Path) -> Result<Value> {
    let timer = Instant::now();
    let fabric: Value = serde_json::from_slice(&fs::read(fabric_path)?)?;
    let root = fabric_path.parent().ok_or("fabric parent")?;
    let attrs = root.join("attributes.json");
    if digest(&attrs)?
        != fabric["auxiliary"]["attributes.json"]["sha256"]
            .as_str()
            .ok_or("attributes SHA")?
    {
        return Err("attribute SHA mismatch".into());
    }
    let a: Value = serde_json::from_slice(&fs::read(&attrs)?)?;
    let Collapsed {
        raw,
        lakes,
        membership,
        dropped,
        collapsed,
        crosswalk,
        ambiguous_outlets,
        ..
    } = collapse_graph(&a)?;
    replay_tail(
        timer, fabric_path, &fabric, root, capture_path, output, raw, lakes, membership, dropped,
        collapsed, crosswalk, ambiguous_outlets,
    )
}

/// The waterbody collapse of a prepared fabric's sealed attributes.json, as
/// t-route preprocess_waterbodies/replace_waterbodies_connections does it.
pub struct Collapsed {
    /// Raw flowpath graph: id -> downstream id (None at a terminal or outside).
    pub raw: BTreeMap<i64, Option<i64>>,
    /// Lake hl_link -> collapsed node id (hl_link, or hl_link+999000000000
    /// when it collides with a flowpath id).
    pub aliases: BTreeMap<i64, i64>,
    /// Retained lake node ids.
    pub lakes: BTreeSet<i64>,
    /// Member flowpath -> lake node id.
    pub membership: BTreeMap<i64, i64>,
    pub dropped: Vec<i64>,
    /// Collapsed (node, downstream) in raw insertion order.
    pub collapsed: Vec<(i64, Option<i64>)>,
    /// Lake node -> its unique outlet member flowpath.
    pub crosswalk: BTreeMap<i64, i64>,
    pub ambiguous_outlets: BTreeMap<i64, Vec<i64>>,
}

pub fn collapse_graph(a: &Value) -> Result<Collapsed> {
    let flowpaths = a["flowpaths"].as_array().ok_or("missing flowpaths")?;
    let mut raw = BTreeMap::<i64, Option<i64>>::new();
    for row in flowpaths {
        // A terminal nexus (tnx-<k>) receives no flowpath. Its number is not a
        // flowpath ID; HYDRO-WORLD fabrics number it after the outlet flowpath
        // itself, which read as an ID would make a self-loop.
        let terminal = row["toid"].as_str().is_some_and(|s| s.starts_with("tnx-"));
        let ds = if terminal { None } else { Some(suffix(&row["toid"])?) };
        if raw
            .insert(suffix(&row["id"])?, ds)
            .is_some()
        {
            return Err("duplicate flowpath ID".into());
        }
    }
    let raw_ids = raw.keys().copied().collect::<BTreeSet<_>>();
    for ds in raw.values_mut() {
        if !ds.is_some_and(|id| raw_ids.contains(&id)) {
            *ds = None;
        }
    }
    let mut lakes = BTreeSet::new();
    let mut aliases = BTreeMap::new();
    for row in a["lakes"].as_array().ok_or("missing lakes")? {
        if [
            "hl_link", "ifd", "LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE", "WeirC",
            "WeirE", "WeirL", "id",
        ]
        .iter()
        .any(|key| row[*key].is_null())
        {
            continue;
        }
        let id = number(&row["hl_link"]).ok_or("invalid lake ID")?;
        let mapped = if raw.contains_key(&id) {
            id.checked_add(999000000000)
                .ok_or("synthetic lake ID overflow")?
        } else {
            id
        };
        aliases.insert(id, mapped);
        lakes.insert(mapped);
    }
    let mut membership = BTreeMap::<i64, i64>::new();
    for row in a["flowpath_attributes"]
        .as_array()
        .ok_or("missing flowpath attributes")?
    {
        let id = suffix(&row["id"])?;
        let values = match &row["rl_NHDWaterbodyComID"] {
            Value::Null => Vec::new(),
            Value::String(s) => s
                .split(',')
                .map(|s| s.trim().parse::<i64>())
                .collect::<std::result::Result<Vec<_>, _>>()?,
            v => vec![number(v).ok_or("bad waterbody membership")?],
        };
        let valid = values
            .iter()
            .filter_map(|id| aliases.get(id).copied())
            .collect::<BTreeSet<_>>();
        if valid.len() > 1 {
            return Err(
                "flowpath belongs to multiple retained lakes; explicit membership table required"
                    .into(),
            );
        }
        if let Some(&lake) = valid.first() {
            if !raw.contains_key(&id) {
                return Err("waterbody member outside flowpaths".into());
            }
            membership.insert(id, lake);
        }
    }
    // Native temporary-data correction: exclude any lake with multiple distinct
    // outgoing destinations after substituting provisional membership on both ends.
    let mut outgoing = BTreeMap::<i64, BTreeSet<Option<i64>>>::new();
    for (&id, &ds) in &raw {
        let from = membership.get(&id).copied().unwrap_or(id);
        let to = ds.map(|id| membership.get(&id).copied().unwrap_or(id));
        if to != Some(from) {
            outgoing.entry(from).or_default().insert(to);
        }
    }
    let dropped = outgoing
        .iter()
        .filter(|(_, dest)| dest.len() > 1)
        .map(|(&id, _)| id)
        .collect::<Vec<_>>();
    for id in &dropped {
        lakes.remove(id);
    }
    membership.retain(|_, lake| lakes.contains(lake));
    let mut collapsed = Vec::<(i64, Option<i64>)>::new();
    let mut inserted = BTreeSet::new();
    let mut crosswalk = BTreeMap::<i64, i64>::new();
    let mut ambiguous_outlets = BTreeMap::<i64, Vec<i64>>::new();
    for (&id, &ds) in &raw {
        if let Some(&lake) = membership.get(&id) {
            if !inserted.insert(lake) {
                continue;
            }
            let members = membership
                .iter()
                .filter(|(_, l)| **l == lake)
                .map(|(&id, _)| id)
                .collect::<BTreeSet<_>>();
            let shore = members
                .iter()
                .filter_map(|id| raw[id])
                .filter(|id| !members.contains(id))
                .collect::<BTreeSet<_>>();
            if shore.len() != 1 {
                return Err("retained lake has no unique native outlet".into());
            }
            let downstream = *shore.first().unwrap();
            let outlets = members
                .iter()
                .copied()
                .filter(|id| raw[id] == Some(downstream))
                .collect::<Vec<_>>();
            if outlets.len() != 1 {
                ambiguous_outlets.insert(lake, outlets);
            } else {
                crosswalk.insert(lake, outlets[0]);
            }
            // HYFeaturesNetwork's override also collapses a directly connected
            // downstream lake; the older nhd_network helper omits this case.
            collapsed.push((
                lake,
                Some(membership.get(&downstream).copied().unwrap_or(downstream)),
            ));
        } else {
            inserted.insert(id);
            collapsed.push((id, ds.map(|id| membership.get(&id).copied().unwrap_or(id))));
        }
    }
    Ok(Collapsed {
        raw,
        aliases,
        lakes,
        membership,
        dropped,
        collapsed,
        crosswalk,
        ambiguous_outlets,
    })
}

#[allow(clippy::too_many_arguments)]
fn replay_tail(
    timer: Instant,
    fabric_path: &Path,
    fabric: &Value,
    root: &Path,
    capture_path: &Path,
    output: &Path,
    raw: BTreeMap<i64, Option<i64>>,
    lakes: BTreeSet<i64>,
    membership: BTreeMap<i64, i64>,
    dropped: Vec<i64>,
    collapsed: Vec<(i64, Option<i64>)>,
    crosswalk: BTreeMap<i64, i64>,
    ambiguous_outlets: BTreeMap<i64, Vec<i64>>,
) -> Result<Value> {
    let mut reverse = BTreeMap::<i64, Vec<i64>>::new();
    let mut connections = BTreeMap::<i64, Vec<i64>>::new();
    for &(id, ds) in &collapsed {
        reverse.entry(id).or_default();
        connections.insert(id, ds.into_iter().collect());
        if let Some(ds) = ds {
            reverse.entry(ds).or_default().push(id);
        }
    }
    let actual = json!({"connections":connections,"reverse_network":reverse,"waterbody_connections":membership});
    let capture: Value = serde_json::from_slice(&fs::read(capture_path)?)?;
    let caproot = capture_path.parent().ok_or("capture parent")?;
    let expected_graph: Value = serde_json::from_slice(&payload(caproot, &capture["network"])?)?;
    let mut expected = serde_json::Map::new();
    for key in ["connections", "reverse_network", "waterbody_connections"] {
        expected.insert(key.into(), expected_graph[key].clone());
    }
    let mut row_differences = Vec::new();
    for key in ["connections", "reverse_network", "waterbody_connections"] {
        let left = actual[key].as_object().unwrap();
        let right = expected[key].as_object().unwrap();
        for id in left.keys().chain(right.keys()).collect::<BTreeSet<_>>() {
            if left.get(id) != right.get(id) {
                row_differences.push(
                    json!({"component":key,"id":id,"actual":left.get(id),"native":right.get(id)}),
                );
            }
        }
    }
    let actual_bytes = serde_json::to_vec(&actual)?;
    let expected_bytes = serde_json::to_vec(&expected)?;
    let differences = actual_bytes
        .iter()
        .zip(&expected_bytes)
        .filter(|(a, b)| a != b)
        .count()
        + actual_bytes.len().abs_diff(expected_bytes.len());
    fs::create_dir_all(output)?;
    fs::write(output.join("derived-network.json"), &actual_bytes)?;
    fs::write(
        output.join("native-network-canonical.json"),
        &expected_bytes,
    )?;
    // Independent geometry-reader parameters are also checked against the actual
    // native f64->f32 columns, with captured rows aligned by immutable numeric ID.
    let param_ids = payload(caproot, &capture["arrays"]["network-parameters-row-index"])?;
    let param_ids = param_ids
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let param_index = param_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect::<HashMap<_, _>>();
    let native_cols = ["dx", "bw", "tw", "twcc", "n", "ncc", "cs", "s0"]
        .iter()
        .map(|col| -> Result<Vec<u32>> {
            let field = &capture["arrays"][format!("network-parameters-column-{col}")];
            if field["dtype"] != "<f8" {
                return Err("native parameter dtype changed".into());
            }
            Ok(payload(caproot, field)?
                .chunks_exact(8)
                .map(|b| (f64::from_le_bytes(b.try_into().unwrap()) as f32).to_bits())
                .collect())
        })
        .collect::<Result<Vec<_>>>()?;
    let numeric = payload(root, &fabric["files"]["node_ids"])?;
    let routing = payload(root, &fabric["files"]["routing_parameters"])?;
    let mut param_differences = 0;
    let mut param_words = 0;
    for (i, b) in numeric.chunks_exact(8).enumerate() {
        let id = i64::from_le_bytes(b.try_into().unwrap());
        if membership.contains_key(&id) {
            continue;
        }
        let row = *param_index.get(&id).ok_or("native parameter missing ID")?;
        for col in 0..8 {
            let from = 4 * (i * 8 + col);
            let word = u32::from_le_bytes(
                routing
                    .get(from..from + 4)
                    .ok_or("Rust parameter shape mismatch")?
                    .try_into()
                    .unwrap(),
            );
            param_differences += usize::from(word != native_cols[col][row]);
            param_words += 1;
        }
    }
    let verdict = json!({"schema":"arwen.nextgen.collapse-replay.v1","strict":true,"scope":"independent GPKG flowpath/membership/lake collapse, ordered incoming CSR source and MC parameters; not whole routed trajectory","source_pin":"12a8eae0cdfed437143c590659fa7077605a5e70","fabric_manifest_sha256":digest(fabric_path)?,"capture_manifest_sha256":digest(capture_path)?,"raw_nodes":raw.len(),"collapsed_nodes":connections.len(),"reservoirs":lakes.len(),"unique_outlet_crosswalks":crosswalk,"ambiguous_outlet_candidates":ambiguous_outlets,"crosswalk_set_choice_bitwise_proven":false,"membership_rows":membership.len(),"dropped_lakes":dropped,"canonical_graph_bytes":actual_bytes.len(),"different_bytes":differences,"different_graph_rows":row_differences.len(),"first_graph_differences":row_differences.iter().take(10).collect::<Vec<_>>(),"actual_sha256":format!("{:x}",Sha256::digest(&actual_bytes)),"native_sha256":format!("{:x}",Sha256::digest(&expected_bytes)),"mc_parameter_words":param_words,"different_parameter_words":param_differences,"pass":differences==0&&param_differences==0,"elapsed_ms":timer.elapsed().as_secs_f64()*1000.,"parallelism":"startup topology dependency scan; source GPKG tables and parameters prepared on all-core Rayon reader; no CPU step"});
    fs::write(
        output.join("verdict.json"),
        serde_json::to_vec_pretty(&verdict)?,
    )?;
    Ok(verdict)
}
