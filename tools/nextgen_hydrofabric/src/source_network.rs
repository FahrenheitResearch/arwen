//! Source-table network subset. Column mapping is configuration data, not a
//! source-specific adapter. This exports topology and verbatim attributes only;
//! it never manufactures routing or reservoir parameters.
use crate::{dbf::Table, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Deserialize)]
struct Source {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
struct Config {
    edges: Source,
    from_column: String,
    to_column: String,
    attributes: Source,
    id_column: String,
    attribute_columns: Vec<String>,
    outlet: Option<i64>,
    outlets: Option<Vec<i64>>,
    selection: Option<Source>,
    #[serde(default)]
    check_reaches: Vec<i64>,
    output: PathBuf,
    provenance: serde_json::Value,
}
impl Config {
    fn selected_outlets(&self) -> Result<BTreeSet<i64>> {
        if self.selection.is_some() {
            return Err("selection cannot be combined with outlet cuts".into());
        }
        let values = match (&self.outlet, &self.outlets) {
            (Some(outlet), None) => vec![*outlet],
            (None, Some(outlets)) => outlets.clone(),
            _ => return Err("configure exactly one of outlet or outlets".into()),
        };
        let unique = values.iter().copied().collect::<BTreeSet<_>>();
        if unique.is_empty() || unique.len() != values.len() || unique.iter().any(|&n| n <= 0) {
            return Err("outlets must contain distinct positive source IDs".into());
        }
        Ok(unique)
    }
}

impl Config {
    fn selected_inputs(&self) -> Result<(BTreeSet<i64>, BTreeSet<i64>)> {
        let Some(source) = &self.selection else {
            let cuts = self.selected_outlets()?;
            return Ok((cuts.clone(), cuts));
        };
        if self.outlet.is_some() || self.outlets.is_some() {
            return Err("configure selection OR outlet/outlets, not both".into());
        }
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(&source.path)?;
        if format!("{:x}", Sha256::digest(&bytes)) != source.sha256 {
            return Err("domain selection digest mismatch".into());
        }
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        if value["schema"] != "arwen.source-domain-selection.v1" {
            return Err("expected sealed source-domain selection schema".into());
        }
        let ids: Vec<i64> = serde_json::from_value(value["node_ids"].clone())?;
        let unique = ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.is_empty() || unique.len() != ids.len() || unique.iter().any(|&n| n <= 0) {
            return Err("domain selection must have distinct positive source IDs".into());
        }
        // A spatial seed is not a downstream cut. Preserve every source arc
        // whose endpoints are in the selected upstream closure.
        Ok((unique, BTreeSet::new()))
    }
}

fn upstream_nodes(edges: &[(i64, i64)], outlets: &BTreeSet<i64>) -> BTreeSet<i64> {
    let mut upstream: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for &(from, to) in edges {
        if from > 0 && to > 0 {
            upstream.entry(to).or_default().push(from);
        }
    }
    let mut selected = BTreeSet::new();
    let mut pending = outlets.iter().copied().collect::<Vec<_>>();
    while let Some(node) = pending.pop() {
        if selected.insert(node) {
            if let Some(inputs) = upstream.get(&node) {
                pending.extend(inputs);
            }
        }
    }
    selected
}
fn table(source: &Source) -> Result<Table> {
    let table = Table::open(&source.path)?;
    if table.sha256 != source.sha256 {
        return Err(format!("source table digest mismatch: {}", source.path.display()).into());
    }
    Ok(table)
}
// dBASE stores some integral IDs using scientific decimal notation. Parse the
// decimal/exponent exactly in i128; do not route identifiers through floats.
pub(crate) fn id(text: &str) -> Result<i64> {
    let mut exponent_split = text.split(['e', 'E']);
    let mantissa = exponent_split.next().ok_or("empty ID")?;
    let exponent: i32 = exponent_split.next().unwrap_or("0").parse()?;
    if exponent_split.next().is_some() {
        return Err("invalid ID exponent".into());
    }
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.strip_prefix(['+', '-']).unwrap_or(mantissa);
    let mut decimal = mantissa.split('.');
    let whole = decimal.next().ok_or("empty ID")?;
    let fraction = decimal.next().unwrap_or("");
    if decimal.next().is_some()
        || whole.is_empty()
        || !whole
            .bytes()
            .chain(fraction.bytes())
            .all(|c| c.is_ascii_digit())
    {
        return Err("invalid decimal ID".into());
    }
    let mut n: i128 = format!("{whole}{fraction}").parse()?;
    let scale = exponent
        .checked_sub(i32::try_from(fraction.len())?)
        .ok_or("ID scale overflow")?;
    let factor = 10i128
        .checked_pow(scale.unsigned_abs())
        .ok_or("ID exponent overflow")?;
    if scale < 0 {
        if n % factor != 0 {
            return Err("nonintegral source ID".into());
        }
        n /= factor;
    } else {
        n = n.checked_mul(factor).ok_or("ID overflow")?;
    }
    if negative {
        n = -n;
    }
    Ok(i64::try_from(n)?)
}
/// Compare with the independent GDAL SQLite numeric referee CSV.
pub fn replay(network: &Path, native: &Path) -> Result<serde_json::Value> {
    let start = Instant::now();
    let candidate: serde_json::Value = serde_json::from_slice(&std::fs::read(network)?)?;
    let nodes: BTreeSet<i64> = serde_json::from_value(candidate["node_ids"].clone())?;
    let edges: BTreeSet<(i64, i64)> = serde_json::from_value(candidate["connections"].clone())?;
    let text = std::fs::read_to_string(native)?;
    let mut rows = text.lines();
    if rows.next() != Some("kind,from_id,to_id") {
        return Err("expected GDAL numeric referee header kind,from_id,to_id".into());
    }
    let mut expected_nodes = BTreeSet::new();
    let mut expected_edges = BTreeSet::new();
    for row in rows {
        let c = row.split(',').collect::<Vec<_>>();
        if c.len() != 3 {
            return Err("invalid numeric referee record".into());
        }
        let a = id(c[1])?;
        let b = id(c[2])?;
        match c[0] {
            "1" if b == 0 => {
                if !expected_nodes.insert(a) {
                    return Err("duplicate referee node".into());
                }
            }
            "2" => {
                if !expected_edges.insert((a, b)) {
                    return Err("duplicate referee edge".into());
                }
            }
            _ => return Err("invalid referee record kind".into()),
        }
    }
    let canonical = |ns: &BTreeSet<i64>, es: &BTreeSet<(i64, i64)>| {
        ns.iter()
            .map(|n| format!("1,{n},0\n"))
            .chain(es.iter().map(|(a, b)| format!("2,{a},{b}\n")))
            .collect::<String>()
    };
    let actual = canonical(&nodes, &edges);
    let expected = canonical(&expected_nodes, &expected_edges);
    let different_bytes = actual
        .bytes()
        .zip(expected.bytes())
        .filter(|(a, b)| a != b)
        .count()
        + actual.len().abs_diff(expected.len());
    use sha2::{Digest, Sha256};
    Ok(
        json!({"pass":different_bytes==0,"different_canonical_bytes":different_bytes,"compared_canonical_bytes":expected.len(),
        "actual_sha256":format!("{:x}",Sha256::digest(actual.as_bytes())),"expected_sha256":format!("{:x}",Sha256::digest(expected.as_bytes())),
        "node_symmetric_difference":nodes.symmetric_difference(&expected_nodes).count(),"edge_symmetric_difference":edges.symmetric_difference(&expected_edges).count(),
        "node_count":nodes.len(),"edge_count":edges.len(),"network":network,"native_capture":native,"strict":true,
        "wall_ms":start.elapsed().as_secs_f64()*1000.0,"parallelism":"canonical preparation-only comparison; source table decoding uses Rayon"}),
    )
}
pub fn prepare(path: &Path) -> Result<serde_json::Value> {
    let start = Instant::now();
    let c: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let (seeds, outlets) = c.selected_inputs()?;
    let tables = [&c.edges, &c.attributes]
        .par_iter()
        .map(|s| table(s))
        .collect::<Result<Vec<_>>>()?;
    let edges = tables[0].rows(&[&c.from_column, &c.to_column])?;
    let edges = edges
        .par_iter()
        .map(|r| Ok((id(&r[0])?, id(&r[1])?)))
        .collect::<Result<Vec<_>>>()?;
    let mut cols = vec![c.id_column.as_str()];
    cols.extend(c.attribute_columns.iter().map(String::as_str));
    let attributes = tables[1].rows(&cols)?;
    let attributes = attributes
        .par_iter()
        .map(|r| Ok((id(&r[0])?, r[1..].to_vec())))
        .collect::<Result<Vec<_>>>()?;
    let source_attributes: BTreeMap<_, _> = attributes.iter().cloned().collect();
    if source_attributes.len() != attributes.len() {
        return Err("duplicate source attribute ID".into());
    }
    let read_ms = start.elapsed().as_secs_f64() * 1000.0;
    for seed in &seeds {
        if !source_attributes.contains_key(seed) {
            return Err(format!("selected reach {seed} absent from source attribute IDs").into());
        }
    }
    let selected = upstream_nodes(&edges, &seeds);
    let missing_attributes = selected
        .iter()
        .filter(|n| !source_attributes.contains_key(n))
        .copied()
        .collect::<Vec<_>>();
    let mut connections: BTreeSet<_> = edges
        .iter()
        .copied()
        .filter(|(a, b)| selected.contains(a) && selected.contains(b) && !outlets.contains(a))
        .collect();
    // Explicit original boundary arcs remain evidence, including diversions.
    let boundary = edges
        .iter()
        .copied()
        .filter(|(a, b)| selected.contains(a) && (!selected.contains(b) || outlets.contains(a)))
        .collect::<BTreeSet<_>>();
    let mut indegree = selected
        .iter()
        .map(|&n| (n, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut downstream: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for &(a, b) in &connections {
        *indegree.get_mut(&b).ok_or("missing selected node")? += 1;
        downstream.entry(a).or_default().push(b);
    }
    let mut queue = indegree
        .iter()
        .filter(|(_, n)| **n == 0)
        .map(|(&n, _)| n)
        .collect::<VecDeque<_>>();
    let mut levels = BTreeMap::new();
    while let Some(a) = queue.pop_front() {
        let level = *levels.entry(a).or_insert(0usize);
        if let Some(ds) = downstream.get(&a) {
            for &b in ds {
                let target = levels.entry(b).or_insert(0usize);
                *target = (*target).max(level + 1);
                let degree = indegree.get_mut(&b).ok_or("missing indegree")?;
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(b);
                }
            }
        }
    }
    let acyclic = indegree.values().all(|n| *n == 0);
    let checks = c
        .check_reaches
        .iter()
        .map(|&n| json!({"reach":n,"in_upstream_subset":selected.contains(&n)}))
        .collect::<Vec<_>>();
    let subset_attributes = selected
        .iter()
        .filter_map(|n| source_attributes.get(n).map(|v| (n, v)))
        .collect::<BTreeMap<_, _>>();
    let topology_ms = start.elapsed().as_secs_f64() * 1000.0 - read_ms;
    std::fs::create_dir(&c.output)?;
    let mut payload = json!({"schema":"arwen.source-network-subset.v1","scope":"topology and verbatim source attributes only; no routing parameters, catchment geometry or reservoir policy",
        "outlet":c.outlet,"node_ids":selected,"connections":std::mem::take(&mut connections),"boundary_connections":boundary,
        "attribute_columns":c.attribute_columns,"attributes":subset_attributes,"missing_attribute_ids":missing_attributes,
        "acyclic":acyclic,"topological_levels":if acyclic {Some(&levels)}else{None},"reach_checks":checks,"provenance":c.provenance,
        "source_tables":[{"path":c.edges.path,"sha256":c.edges.sha256},{"path":c.attributes.path,"sha256":c.attributes.sha256}]});
    if c.outlets.is_some() {
        payload.as_object_mut().unwrap().remove("outlet");
        payload["outlets"] = json!(outlets);
    }
    if let Some(source) = &c.selection {
        payload.as_object_mut().unwrap().remove("outlet");
        payload["selection"] = json!({"path":source.path,"sha256":source.sha256,
            "seed_node_ids":seeds,"policy":"upstream closure; no artificial outlet cuts"});
    }
    let bytes = serde_json::to_vec(&payload)?;
    std::fs::write(c.output.join("network.json"), &bytes)?;
    use sha2::{Digest, Sha256};
    Ok(
        json!({"source":crate::SOURCE_REVISION,"output":c.output,"node_count":payload["node_ids"].as_array().unwrap().len(),
        "edge_count":payload["connections"].as_array().unwrap().len(),"acyclic":acyclic,"reach_checks":checks,
        "missing_attribute_ids":missing_attributes,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes)),
        "read_decode_ms":read_ms,"subset_topology_ms":topology_ms,"total_ms":start.elapsed().as_secs_f64()*1000.0,
        "rayon_workers":rayon::current_num_threads(),"parallelism":"independent table reads and fixed-width row decoding on Rayon; dependency walk at preparation only"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(selection: serde_json::Value) -> Config {
        let mut c = json!({"edges":{"path":"unused","sha256":"unused"},
            "attributes":{"path":"unused","sha256":"unused"},
            "from_column":"from","to_column":"to","id_column":"id",
            "attribute_columns":[],"output":"unused","provenance":{}});
        c.as_object_mut()
            .unwrap()
            .extend(selection.as_object().unwrap().clone());
        serde_json::from_value(c).unwrap()
    }

    #[test]
    fn outlet_selection_rejects_ambiguous_empty_duplicate_and_sentinel_ids() {
        for invalid in [
            json!({}),
            json!({"outlets":[]}),
            json!({"outlets":[1,1]}),
            json!({"outlets":[0]}),
            json!({"outlet":-1}),
            json!({"outlet":1,"outlets":[2]}),
        ] {
            assert!(config(invalid).selected_outlets().is_err());
        }
        assert_eq!(
            config(json!({"outlet":3})).selected_outlets().unwrap(),
            config(json!({"outlets":[3]})).selected_outlets().unwrap()
        );
    }

    #[test]
    fn multiple_basins_share_upstream_nodes_without_copying_or_losing_diversions() {
        let edges = [
            (1, 2),
            (2, 3),
            (1, 4),
            (4, 5),
            (3, 0),
            (5, 6),
            (0, 3),
            (-1, 5),
            (7, 8),
        ];
        let outlets = BTreeSet::from([3, 5]);
        let nodes = upstream_nodes(&edges, &outlets);
        assert_eq!(nodes, BTreeSet::from([1, 2, 3, 4, 5]));
        assert_eq!(
            nodes,
            upstream_nodes(&edges, &BTreeSet::from([3]))
                .union(&upstream_nodes(&edges, &BTreeSet::from([5])))
                .copied()
                .collect()
        );
        let boundary = edges
            .iter()
            .copied()
            .filter(|(a, b)| nodes.contains(a) && (!nodes.contains(b) || outlets.contains(a)))
            .collect::<BTreeSet<_>>();
        assert_eq!(boundary, BTreeSet::from([(3, 0), (5, 6)]));
    }

    #[test]
    fn cyclic_upstream_walk_terminates_and_keeps_each_original_id() {
        assert_eq!(
            upstream_nodes(&[(1, 2), (2, 1), (2, 3), (4, 5)], &BTreeSet::from([3, 5])),
            BTreeSet::from([1, 2, 3, 4, 5])
        );
    }
}
