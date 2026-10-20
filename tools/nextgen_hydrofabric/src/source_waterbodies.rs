//! Source identifier association only. Polygon feature types and values remain
//! verbatim: a polygonal water-feature reference does not establish a reservoir.
use crate::{
    dbf::Table,
    source_geometry::{bytes, Source},
    source_network::id,
    Result,
};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::Digest;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    network: Source,
    flowpaths: Source,
    flowpath_id_column: String,
    reference_column: String,
    // Exact source text, including an empty string if the provider defines it.
    no_reference_values: Vec<String>,
    unresolved_reference_values: Vec<String>,
    waterbodies: Source,
    // Same column mapping; every table remains independently SHA-pinned.
    #[serde(default)]
    additional_waterbodies: Vec<Source>,
    waterbody_id_column: String,
    attribute_columns: Vec<String>,
    data_license: String,
    provenance: Value,
    output: PathBuf,
}

fn positive(text: &str) -> Result<i64> {
    let n = id(text)?;
    if n <= 0 {
        return Err("source association requires positive IDs".into());
    }
    Ok(n)
}

fn unique_rows(rows: Vec<Vec<String>>) -> Result<BTreeMap<i64, Vec<String>>> {
    let decoded = rows
        .into_par_iter()
        .map(|r| Ok((positive(&r[0])?, r[1..].to_vec())))
        .collect::<Result<Vec<_>>>()?;
    let count = decoded.len();
    let map = decoded.into_iter().collect::<BTreeMap<_, _>>();
    if map.len() != count {
        return Err("duplicate source association ID".into());
    }
    Ok(map)
}

fn merge_targets(
    tables: Vec<BTreeMap<i64, Vec<String>>>,
) -> Result<BTreeMap<i64, (usize, Vec<String>)>> {
    let mut targets = BTreeMap::new();
    for (index, rows) in tables.into_iter().enumerate() {
        for (id, attributes) in rows {
            if let Some((previous, _)) = targets.insert(id, (index, attributes)) {
                return Err(format!(
                    "ambiguous polygon ID {id} occurs in source tables {previous} and {index}"
                )
                .into());
            }
        }
    }
    Ok(targets)
}

fn columns<'a>(id_column: &'a str, values: &'a [String]) -> Result<Vec<&'a str>> {
    let mut names = vec![id_column];
    names.extend(values.iter().map(String::as_str));
    if names.iter().any(|n| n.trim().is_empty())
        || names.iter().collect::<BTreeSet<_>>().len() != names.len()
    {
        return Err("association columns must be named and distinct".into());
    }
    Ok(names)
}

pub fn prepare(path: &Path) -> Result<Value> {
    let start = Instant::now();
    let c: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    if c.data_license.trim().is_empty() {
        return Err("name the source data licence".into());
    }
    let absent = c
        .no_reference_values
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let unresolved = c
        .unresolved_reference_values
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if unresolved.len() != c.unresolved_reference_values.len() || !absent.is_disjoint(&unresolved) {
        return Err("reference status values must be distinct and disjoint".into());
    }
    if absent.len() != c.no_reference_values.len() {
        return Err("duplicate no-reference value".into());
    }
    // A positive source reference cannot be relabelled as absent by configuration.
    if absent
        .iter()
        .chain(unresolved.iter())
        .any(|v| id(v).is_ok_and(|n| n > 0))
    {
        return Err("positive source IDs cannot be no-reference values".into());
    }
    let network: Value = serde_json::from_slice(&bytes(&c.network)?)?;
    if network["schema"] != "arwen.source-network-subset.v1" {
        return Err("expected sealed source-network subset".into());
    }
    let roster: Vec<i64> = serde_json::from_value(network["node_ids"].clone())?;
    let nodes = roster.iter().copied().collect::<BTreeSet<_>>();
    if nodes.is_empty() || nodes.len() != roster.len() || nodes.iter().any(|&n| n <= 0) {
        return Err("network roster must contain distinct positive IDs".into());
    }
    let references = vec![c.reference_column.clone()];
    let flow_columns = columns(&c.flowpath_id_column, &references)?;
    let water_columns = columns(&c.waterbody_id_column, &c.attribute_columns)?;
    let mut sources = vec![&c.flowpaths, &c.waterbodies];
    sources.extend(c.additional_waterbodies.iter());
    let tables = sources
        .par_iter()
        .map(|source| {
            let table = Table::open(&source.path)?;
            if table.sha256 != source.sha256 {
                return Err("source association table digest mismatch".into());
            }
            Ok(table)
        })
        .collect::<Result<Vec<_>>>()?;
    let (flow, water) = rayon::join(
        || unique_rows(tables[0].rows(&flow_columns)?),
        || {
            let decoded = tables[1..]
                .par_iter()
                .map(|table| unique_rows(table.rows(&water_columns)?))
                .collect::<Result<Vec<_>>>()?;
            merge_targets(decoded)
        },
    );
    let (flow, water) = (flow?, water?);
    let read_decode_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut links = Vec::new();
    let mut no_reference = Vec::new();
    let mut unresolved_nodes = Vec::new();
    let mut groups: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let mut source_references = BTreeMap::new();
    for &n in &nodes {
        let row = flow
            .get(&n)
            .ok_or_else(|| format!("selected flowpath {n} absent from source table"))?;
        let raw = &row[0];
        source_references.insert(n, raw);
        if unresolved.contains(raw) {
            unresolved_nodes.push(n);
            continue;
        }
        if absent.contains(raw) {
            no_reference.push(n);
            continue;
        }
        let target = positive(raw)?;
        links.push([n, target]);
        groups.entry(target).or_default().push(n);
    }
    let missing = groups
        .keys()
        .filter(|n| !water.contains_key(n))
        .copied()
        .collect::<Vec<_>>();
    let joined = groups
        .iter()
        .filter_map(|(n, members)| {
            water.get(n).map(|(source_index, attributes)| {
                let mut row = json!({"waterbody_id":n,"node_ids":members,"attributes":attributes});
                // Preserve the single-table v1 byte contract for existing callers.
                if !c.additional_waterbodies.is_empty() {
                    row["source_index"] = json!(source_index);
                }
                row
            })
        })
        .collect::<Vec<_>>();
    let mut payload = json!({"schema":"arwen.source-waterbody-crosswalk.v1",
        "scope":"source association and verbatim attributes only; no lake-only classification, outlet, hydraulic or storage inference",
        "node_ids":nodes,"flowpath_waterbody_ids":links,"source_reference_values":source_references,
        "no_reference_node_ids":no_reference,"missing_waterbody_ids":missing,
        "unresolved_reference_node_ids":unresolved_nodes,
        "complete_reference_join":missing.is_empty() && unresolved_nodes.is_empty(),"waterbodies":joined,
        "mapping":{"flowpath_id_column":c.flowpath_id_column,"reference_column":c.reference_column,
            "no_reference_values":absent,"unresolved_reference_values":unresolved,"waterbody_id_column":c.waterbody_id_column,"attribute_columns":c.attribute_columns},
        "sources":{"network":{"path":c.network.path,"sha256":c.network.sha256},
            "flowpaths":{"path":c.flowpaths.path,"sha256":c.flowpaths.sha256},
            "waterbodies":{"path":c.waterbodies.path,"sha256":c.waterbodies.sha256}},
        "data_license":c.data_license,"provenance":c.provenance});
    if !c.additional_waterbodies.is_empty() {
        payload["sources"]["additional_waterbodies"] = json!(c
            .additional_waterbodies
            .iter()
            .map(|s| json!({"path":s.path,"sha256":s.sha256}))
            .collect::<Vec<_>>());
    }
    let raw = serde_json::to_vec(&payload)?;
    let join_ms = start.elapsed().as_secs_f64() * 1000.0 - read_decode_ms;
    // All schema/source/join validation precedes creating any output.
    std::fs::create_dir(&c.output)?;
    std::fs::write(c.output.join("waterbodies.json"), &raw)?;
    use sha2::{Digest, Sha256};
    Ok(
        json!({"source":crate::SOURCE_REVISION,"output":c.output,"node_count":nodes.len(),
        "referenced_node_count":links.len(),"no_reference_node_count":no_reference.len(),
        "joined_waterbody_count":joined.len(),"missing_waterbody_ids":missing,
        "unresolved_reference_node_ids":unresolved_nodes,
        "complete_reference_join":missing.is_empty() && unresolved_nodes.is_empty(),"bytes":raw.len(),"sha256":format!("{:x}",Sha256::digest(&raw)),
        "read_decode_ms":read_decode_ms,"join_ms":join_ms,"total_ms":start.elapsed().as_secs_f64()*1000.0,
        "rayon_workers":rayon::current_num_threads(),"parallelism":"independent source table reads and fixed-width row decoding on Rayon; deterministic identifier association at preparation only"}),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundaryConfig {
    network: Source,
    crosswalk: Source,
    output: PathBuf,
}

fn distinct_ids(value: &Value) -> Result<BTreeSet<i64>> {
    let ids: Vec<i64> = serde_json::from_value(value.clone())?;
    let set = ids.iter().copied().collect::<BTreeSet<_>>();
    if set.len() != ids.len() || set.iter().any(|&n| n <= 0) {
        return Err("water-feature topology requires distinct positive IDs".into());
    }
    Ok(set)
}

fn boundary_rows(network: &Value, crosswalk: &Value, network_sha: &str) -> Result<Vec<Value>> {
    if network["schema"] != "arwen.source-network-subset.v1"
        || crosswalk["schema"] != "arwen.source-waterbody-crosswalk.v1"
        || crosswalk["sources"]["network"]["sha256"].as_str() != Some(network_sha)
    {
        return Err("water-feature crosswalk must bind this exact source network".into());
    }
    let nodes = distinct_ids(&network["node_ids"])?;
    if nodes.is_empty() || distinct_ids(&crosswalk["node_ids"])? != nodes {
        return Err("crosswalk and source network rosters differ or are empty".into());
    }
    let links: Vec<[i64; 2]> = serde_json::from_value(crosswalk["flowpath_waterbody_ids"].clone())?;
    let mut referenced = BTreeSet::new();
    let mut groups = BTreeMap::<i64, BTreeSet<i64>>::new();
    for [node, waterbody] in links {
        if !nodes.contains(&node) || waterbody <= 0 || !referenced.insert(node) {
            return Err("invalid or repeated crosswalk flowpath association".into());
        }
        groups.entry(waterbody).or_default().insert(node);
    }
    let absent = distinct_ids(&crosswalk["no_reference_node_ids"])?;
    let unresolved = distinct_ids(&crosswalk["unresolved_reference_node_ids"])?;
    if !referenced.is_disjoint(&absent)
        || !referenced.is_disjoint(&unresolved)
        || !absent.is_disjoint(&unresolved)
        || referenced
            .union(&absent)
            .copied()
            .collect::<BTreeSet<_>>()
            .union(&unresolved)
            .copied()
            .collect::<BTreeSet<_>>()
            != nodes
    {
        return Err("crosswalk reference status partition does not cover the network".into());
    }
    let mut features = BTreeMap::new();
    for feature in crosswalk["waterbodies"]
        .as_array()
        .ok_or("missing crosswalk features")?
    {
        let waterbody = feature["waterbody_id"]
            .as_i64()
            .ok_or("invalid feature ID")?;
        if groups.get(&waterbody) != Some(&distinct_ids(&feature["node_ids"])?)
            || features.insert(waterbody, feature).is_some()
        {
            return Err("crosswalk feature membership differs from its associations".into());
        }
    }
    let missing = groups
        .keys()
        .filter(|id| !features.contains_key(id))
        .copied()
        .collect::<BTreeSet<_>>();
    if distinct_ids(&crosswalk["missing_waterbody_ids"])? != missing
        || crosswalk["complete_reference_join"].as_bool()
            != Some(missing.is_empty() && unresolved.is_empty())
    {
        return Err("crosswalk completeness differs from its source gaps".into());
    }
    let connections: Vec<[i64; 2]> = serde_json::from_value(network["connections"].clone())?;
    let boundary: Vec<[i64; 2]> = serde_json::from_value(network["boundary_connections"].clone())?;
    let edges = connections.iter().copied().collect::<BTreeSet<_>>();
    let cuts = boundary.iter().copied().collect::<BTreeSet<_>>();
    if edges.len() != connections.len()
        || cuts.len() != boundary.len()
        || !edges.is_disjoint(&cuts)
        || edges
            .iter()
            .any(|[a, b]| !nodes.contains(a) || !nodes.contains(b))
        || cuts.iter().any(|[a, _]| !nodes.contains(a))
    {
        return Err("source network edge endpoints/categories are inconsistent".into());
    }
    let mut outgoing = BTreeMap::<i64, Vec<[i64; 2]>>::new();
    let mut incoming = BTreeMap::<i64, Vec<[i64; 2]>>::new();
    let mut domain_edges = BTreeMap::<i64, Vec<[i64; 2]>>::new();
    for &[a, b] in &edges {
        outgoing.entry(a).or_default().push([a, b]);
        incoming.entry(b).or_default().push([a, b]);
    }
    for &[a, b] in &cuts {
        domain_edges.entry(a).or_default().push([a, b]);
    }
    let has_successor = edges
        .iter()
        .chain(&cuts)
        .map(|e| e[0])
        .collect::<BTreeSet<_>>();
    Ok(groups
        .par_iter()
        .map(|(id, members)| {
            let (internal, exits): (Vec<_>, Vec<_>) = members
                .iter()
                .flat_map(|n| outgoing.get(n).into_iter().flatten().copied())
                .partition(|[_, b]| members.contains(b));
            let mut inlets = members
                .iter()
                .flat_map(|n| incoming.get(n).into_iter().flatten().copied())
                .filter(|[a, _]| !members.contains(a))
                .collect::<Vec<_>>();
            inlets.sort_unstable();
            let domain = members
                .iter()
                .flat_map(|n| domain_edges.get(n).into_iter().flatten().copied())
                .collect::<Vec<_>>();
            let no_successor = members
                .difference(&has_successor)
                .copied()
                .collect::<Vec<_>>();
            json!({"waterbody_id":id,"node_ids":members,"source_feature":features.get(id),
            "internal_connections":internal,"inlet_connections":inlets,"exit_connections":exits,
            "domain_boundary_connections":domain,"no_recorded_successor_node_ids":no_successor})
        })
        .collect())
}

/// Preparation evidence only: graph exits are not physical hydraulic outlets.
pub fn boundaries(path: &Path) -> Result<Value> {
    let start = Instant::now();
    let c: BoundaryConfig = serde_json::from_slice(&std::fs::read(path)?)?;
    let (network, crosswalk) = rayon::join(|| bytes(&c.network), || bytes(&c.crosswalk));
    let network: Value = serde_json::from_slice(&network?)?;
    let crosswalk: Value = serde_json::from_slice(&crosswalk?)?;
    let read_ms = start.elapsed().as_secs_f64() * 1000.;
    let rows = boundary_rows(&network, &crosswalk, &c.network.sha256)?;
    let payload = json!({"schema":"arwen.source-waterbody-boundaries.v1",
        "scope":"Graph evidence within the selected source network only. Inlets exclude unknown external incoming edges. Domain cuts, source-unknown references, divergences and absent recorded successors remain explicit. No physical outlet selection, reservoir classification, storage or hydraulic inference.",
        "sources":{"network":{"path":c.network.path,"sha256":c.network.sha256},
            "crosswalk":{"path":c.crosswalk.path,"sha256":c.crosswalk.sha256}},
        "data_license":crosswalk["data_license"],"node_ids":network["node_ids"],
        "no_reference_node_ids":crosswalk["no_reference_node_ids"],
        "unresolved_reference_node_ids":crosswalk["unresolved_reference_node_ids"],
        "missing_waterbody_ids":crosswalk["missing_waterbody_ids"],
        "complete_reference_join":crosswalk["complete_reference_join"],"waterbodies":rows});
    let raw = serde_json::to_vec(&payload)?;
    let classify_ms = start.elapsed().as_secs_f64() * 1000. - read_ms;
    std::fs::create_dir(&c.output)?;
    std::fs::write(c.output.join("waterbody-boundaries.json"), &raw)?;
    Ok(
        json!({"source":crate::SOURCE_REVISION,"waterbody_count":rows.len(),"bytes":raw.len(),
        "sha256":format!("{:x}", sha2::Sha256::digest(&raw)),"read_decode_ms":read_ms,"classify_ms":classify_ms,
        "total_ms":start.elapsed().as_secs_f64()*1000.,"rayon_workers":rayon::current_num_threads(),
        "parallelism":"independent sealed reads and per-water-feature graph classification on Rayon; deterministic sorted edges/IDs; CPU preparation only"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn boundary_fixture() -> (Value, Value) {
        let network = json!({"schema":"arwen.source-network-subset.v1","node_ids":[1,2,3,4,5,6,7],
            "connections":[[1,2],[2,3],[2,4],[3,5],[6,3]],
            // Explicit cut to another selected ID must remain a cut.
            "boundary_connections":[[4,3],[5,0]]});
        let crosswalk = json!({"schema":"arwen.source-waterbody-crosswalk.v1",
            "sources":{"network":{"sha256":"pin"}},"node_ids":[1,2,3,4,5,6,7],
            "flowpath_waterbody_ids":[[2,20],[3,20],[5,50],[7,70]],
            "no_reference_node_ids":[1,4],"unresolved_reference_node_ids":[6],
            "missing_waterbody_ids":[70],"complete_reference_join":false,
            "waterbodies":[{"waterbody_id":20,"node_ids":[2,3],"attributes":["StreamRiver"]},
                {"waterbody_id":50,"node_ids":[5],"attributes":["LakePond"]}]});
        (network, crosswalk)
    }
    #[test]
    fn boundary_evidence_preserves_divergences_cuts_and_missing_features() {
        let (network, crosswalk) = boundary_fixture();
        let rows = boundary_rows(&network, &crosswalk, "pin").unwrap();
        assert_eq!(rows[0]["internal_connections"], json!([[2, 3]]));
        assert_eq!(rows[0]["inlet_connections"], json!([[1, 2], [6, 3]]));
        assert_eq!(rows[0]["exit_connections"], json!([[2, 4], [3, 5]]));
        assert_eq!(
            rows[0]["source_feature"]["attributes"],
            json!(["StreamRiver"])
        );
        assert_eq!(rows[1]["domain_boundary_connections"], json!([[5, 0]]));
        assert_eq!(rows[1]["no_recorded_successor_node_ids"], json!([]));
        assert!(rows[2]["source_feature"].is_null());
        assert_eq!(rows[2]["no_recorded_successor_node_ids"], json!([7]));
        let mut crosswalk = crosswalk;
        crosswalk["flowpath_waterbody_ids"] = json!([[2, 20], [3, 20], [4, 20], [5, 50], [7, 70]]);
        crosswalk["no_reference_node_ids"] = json!([1]);
        crosswalk["waterbodies"][0]["node_ids"] = json!([2, 3, 4]);
        let rows = boundary_rows(&network, &crosswalk, "pin").unwrap();
        assert_eq!(rows[0]["internal_connections"], json!([[2, 3], [2, 4]]));
        assert_eq!(rows[0]["domain_boundary_connections"], json!([[4, 3]]));
    }
    #[test]
    fn boundary_evidence_refuses_misbound_or_inconsistent_inputs() {
        let (network, crosswalk) = boundary_fixture();
        assert!(boundary_rows(&network, &crosswalk, "different").is_err());
        for (key, value) in [
            ("node_ids", json!([1, 2, 3, 4, 5, 6, 8])),
            ("flowpath_waterbody_ids", json!([[2, 20], [2, 20]])),
            ("no_reference_node_ids", json!([1, 2, 4])),
            ("unresolved_reference_node_ids", json!([])),
            ("missing_waterbody_ids", json!([])),
            ("complete_reference_join", json!(true)),
            (
                "waterbodies",
                json!([{"waterbody_id":20,"node_ids":[2],"attributes":[]}]),
            ),
        ] {
            let mut bad = crosswalk.clone();
            bad[key] = value;
            assert!(boundary_rows(&network, &bad, "pin").is_err(), "{key}");
        }
        for (key, value) in [
            ("connections", json!([[1, 2], [1, 2]])),
            ("connections", json!([[1, 8]])),
            ("boundary_connections", json!([[8, 9]])),
            ("boundary_connections", json!([[1, 2]])),
        ] {
            let mut bad = network.clone();
            bad[key] = value;
            assert!(boundary_rows(&bad, &crosswalk, "pin").is_err(), "{key}");
        }
    }
    #[test]
    fn exact_identifiers_and_collisions() {
        assert_eq!(positive("9.007199254740993E15").unwrap(), 9007199254740993);
        assert!(positive("-9999").is_err());
        assert!(positive("1.5").is_err());
        assert!(unique_rows(vec![vec!["12".into()], vec!["1.2e1".into()]]).is_err());
    }
    #[test]
    fn mapping_cannot_overwrite_identifiers() {
        assert!(columns("id", &["id".into()]).is_err());
        assert!(columns("", &[]).is_err());
        assert!(columns("id", &["kind".into(), "kind".into()]).is_err());
    }
    #[test]
    fn polygon_union_keeps_source_and_rejects_any_ambiguous_id() {
        let primary = unique_rows(vec![vec!["12".into(), "LakePond".into()]]).unwrap();
        let secondary = unique_rows(vec![vec!["13".into(), "StreamRiver".into()]]).unwrap();
        let joined = merge_targets(vec![primary.clone(), secondary]).unwrap();
        assert_eq!(joined[&12], (0, vec!["LakePond".into()]));
        assert_eq!(joined[&13], (1, vec!["StreamRiver".into()]));
        let same = unique_rows(vec![vec!["1.2e1".into(), "LakePond".into()]]).unwrap();
        assert!(merge_targets(vec![primary.clone(), same]).is_err());
        assert!(merge_targets(vec![primary.clone(), primary]).is_err());
        assert!(merge_targets(vec![BTreeMap::new()]).unwrap().is_empty());
    }
}
