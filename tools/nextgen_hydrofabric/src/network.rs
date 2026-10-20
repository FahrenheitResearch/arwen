use crate::{
    sqlite::{ident, Db, Row},
    Result,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    time::Instant,
};

#[derive(Deserialize)]
pub struct Config {
    pub geopackage: String,
    pub output: String,
    pub data_license: String,
    #[serde(default)]
    pub tables: BTreeMap<String, String>,
    #[serde(default)]
    pub gauge: Option<String>,
    #[serde(default)]
    pub huc: Option<String>,
    #[serde(default)]
    pub huc_column: Option<String>,
    #[serde(default)]
    pub grid: Option<crate::weights::Grid>,
    // Optional executed-parent contract, including the driver's input identity.
    #[serde(default)]
    pub parent_grid_contract: Option<Value>,
    // A native-captured node roster controls upstream reduction order explicitly.
    #[serde(default)]
    pub upstream_order: Option<Vec<String>>,
}
pub struct Layer {
    pub name: String,
    pub rows: Vec<Row>,
    pub geom_column: Option<String>,
    pub srs_id: i64,
    pub crs: Option<String>,
}
#[derive(Serialize)]
pub struct Network {
    pub node_ids: Vec<String>,
    pub catchment_ids: Vec<String>,
    pub numeric_node_ids: Option<Vec<i64>>,
    pub downstream: Vec<i32>,
    pub downstream_ids: Vec<String>,
    pub upstream_offsets: Vec<i32>,
    pub upstream_indices: Vec<i32>,
    pub ordered_nodes: Vec<i32>,
    pub level_offsets: Vec<i32>,
    pub reverse_ordered_nodes: Vec<i32>,
    pub reverse_level_offsets: Vec<i32>,
    pub catchment_to_node: Vec<i32>,
    pub routing_parameters: Option<Vec<[f32; 8]>>,
    pub upstream_order: String,
}
pub struct Prepared {
    pub layers: BTreeMap<String, Layer>,
    pub network: Network,
    pub read_ms: f64,
    pub topology_ms: f64,
}
fn text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}
pub fn field(row: &Row, col: &str) -> Option<String> {
    row.properties.get(col).and_then(text)
}
pub fn id(row: &Row, role: &str) -> Result<String> {
    field(
        row,
        if role == "divides" && row.properties.contains_key("divide_id") {
            "divide_id"
        } else {
            "id"
        },
    )
    .ok_or_else(|| format!("missing {role} ID").into())
}
pub fn prepare(c: &Config) -> Result<Prepared> {
    let start = Instant::now();
    let db = Db::open(&c.geopackage)?;
    let available = db
        .query(
            "SELECT table_name FROM gpkg_contents ORDER BY table_name",
            None,
        )?
        .into_iter()
        .filter_map(|r| field(&r, "table_name"))
        .collect::<Vec<_>>();
    let geoms = db.query(
        "SELECT table_name,column_name,srs_id FROM gpkg_geometry_columns",
        None,
    )?;
    let spatial_refs = db.query(
        "SELECT srs_id,organization,organization_coordsys_id,definition FROM gpkg_spatial_ref_sys",
        None,
    )?;
    let roles = [
        "divides",
        "flowpaths",
        "nexus",
        "network",
        "flowpath_attributes",
        "model_attributes",
        "hydrolocations",
        "lakes",
    ];
    let choices = roles
        .iter()
        .filter_map(|role| {
            let table = c.tables.get(*role).cloned().or_else(|| {
                available
                    .iter()
                    .find(|n| n.replace('-', "_") == *role)
                    .cloned()
            });
            table.map(|t| ((*role).to_string(), t))
        })
        .collect::<Vec<_>>();
    for role in ["flowpaths", "divides", "nexus"] {
        if !choices.iter().any(|(r, _)| r == role) {
            return Err(
                format!("GeoPackage lacks {role}; supply its table name in config.tables").into(),
            );
        }
    }
    let layers = choices
        .par_iter()
        .map(|(role, name)| -> Result<(String, Layer)> {
            let db = Db::open(&c.geopackage)?;
            let g = geoms
                .iter()
                .find(|r| field(r, "table_name").as_deref() == Some(name));
            let geom = g.and_then(|r| field(r, "column_name"));
            let srs = g
                .and_then(|r| r.properties.get("srs_id"))
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let rows = db.query(&format!("SELECT * FROM {}", ident(name)), geom.as_deref())?;
            Ok((
                role.clone(),
                Layer {
                    name: name.clone(),
                    rows,
                    geom_column: geom,
                    srs_id: srs,
                    crs: spatial_refs
                        .iter()
                        .find(|r| r.properties["srs_id"].as_i64() == Some(srs))
                        .and_then(|r| {
                            let org = field(r, "organization")?;
                            if org.eq_ignore_ascii_case("EPSG") {
                                Some(format!("EPSG:{}", r.properties["organization_coordsys_id"]))
                            } else {
                                field(r, "definition").filter(|s| s != "undefined")
                            }
                        }),
                },
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let read_ms = start.elapsed().as_secs_f64() * 1000.;
    let timer = Instant::now();
    let flow = &layers["flowpaths"].rows;
    let nexus = &layers["nexus"].rows;
    if flow.len() > i32::MAX as usize {
        return Err("network exceeds int32 GPU index range".into());
    }
    let all_ids = flow
        .iter()
        .map(|r| id(r, "flowpaths"))
        .collect::<Result<Vec<_>>>()?;
    let all_index = all_ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect::<HashMap<_, _>>();
    if all_index.len() != all_ids.len() {
        return Err("duplicate flowpath IDs would alias GPU state".into());
    }
    let nex_edges = nexus
        .iter()
        .map(|r| Ok((id(r, "nexus")?, field(r, "toid").unwrap_or_default())))
        .collect::<Result<HashMap<_, _>>>()?;
    let resolve = |target: String| -> Result<String> {
        let mut cur = target;
        let mut seen = HashSet::new();
        while !all_index.contains_key(&cur) {
            match nex_edges.get(&cur) {
                Some(next) => {
                    if !seen.insert(cur.clone()) {
                        return Err(format!("nexus cycle at {cur}").into());
                    }
                    cur = next.clone()
                }
                None => break,
            }
        }
        Ok(cur)
    };
    let destinations = flow
        .par_iter()
        .map(|r| resolve(field(r, "toid").unwrap_or_default()))
        .collect::<Result<Vec<_>>>()?;
    let downstream = destinations
        .iter()
        .map(|s| all_index.get(s).copied())
        .collect::<Vec<_>>();
    let mut upstream = vec![vec![]; flow.len()];
    for (i, d) in downstream.iter().enumerate() {
        if let Some(d) = d {
            upstream[*d].push(i)
        }
    }
    let mut keep = vec![true; flow.len()];
    if let Some(gauge) = &c.gauge {
        if c.huc.is_some() {
            return Err("select gauge or HUC, one at a time".into());
        }
        let mut targets = HashSet::new();
        // Hydrolocations/network identify the gauge nexus. rl_gages can mark
        // multiple adjacent reaches and must not override this location.
        for authoritative in [true, false] {
            for (role, layer) in &layers {
                if matches!(role.as_str(), "hydrolocations" | "network") != authoritative {
                    continue;
                }
                for row in &layer.rows {
                    let hit = ["hl_link", "hl_uri", "rl_gages", "gages"]
                        .iter()
                        .any(|key| {
                            field(row, key)
                                .is_some_and(|s| s == *gauge || s == format!("Gages-{gauge}"))
                        });
                    if hit {
                        if let Ok(s) = id(row, role) {
                            targets.insert(s);
                        }
                        if role == "network" {
                            if let Some(s) = field(row, "toid") {
                                targets.insert(s);
                            }
                        }
                    }
                }
            }
            if !targets.is_empty() {
                break;
            }
        }
        let seeds = all_ids
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                targets.contains(*s)
                    || field(&flow[*i], "toid").is_some_and(|d| targets.contains(&d))
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if seeds.is_empty() {
            return Err(format!("gauge {gauge} has no exact network/hydrolocation match").into());
        }
        keep.fill(false);
        let mut stack = seeds;
        while let Some(i) = stack.pop() {
            if !keep[i] {
                keep[i] = true;
                stack.extend(&upstream[i]);
            }
        }
    }
    if let Some(huc) = &c.huc {
        let col = c
            .huc_column
            .as_deref()
            .ok_or("huc_column required: preserve the actual hydrofabric HUC field")?;
        let mut selected = HashSet::new();
        for (role, layer) in &layers {
            for row in &layer.rows {
                if field(row, col).is_some_and(|s| s.starts_with(huc)) {
                    if let Ok(s) = id(row, role) {
                        selected.insert(s);
                    }
                    for k in ["id", "divide_id"] {
                        if let Some(s) = field(row, k) {
                            selected.insert(s);
                        }
                    }
                }
            }
        }
        for (i, r) in flow.iter().enumerate() {
            keep[i] = selected.contains(&all_ids[i])
                || field(r, "divide_id").is_some_and(|d| selected.contains(&d));
        }
        if !keep.iter().any(|&v| v) {
            return Err(format!("HUC {huc} has no {col} match").into());
        }
    }
    let selected = (0..flow.len()).filter(|&i| keep[i]).collect::<Vec<_>>();
    let old_to_new = selected
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new as i32))
        .collect::<HashMap<_, _>>();
    let ids = selected
        .iter()
        .map(|&i| all_ids[i].clone())
        .collect::<Vec<_>>();
    let ds = selected
        .iter()
        .map(|&i| {
            downstream[i]
                .and_then(|d| old_to_new.get(&d).copied())
                .unwrap_or(-1)
        })
        .collect::<Vec<_>>();
    let rank = if let Some(roster) = &c.upstream_order {
        let r = roster
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect::<HashMap<_, _>>();
        if r.len() != roster.len() || ids.iter().any(|s| !r.contains_key(s)) {
            return Err(
                "native upstream roster must contain every selected ID exactly once".into(),
            );
        }
        Some(r)
    } else {
        None
    };
    let mut up = vec![vec![]; ids.len()];
    for (i, &d) in ds.iter().enumerate() {
        if d >= 0 {
            up[d as usize].push(i as i32)
        }
    }
    if let Some(rank) = rank {
        for u in &mut up {
            u.sort_by_key(|&i| rank[&ids[i as usize]]);
        }
    }
    let (ordered, offsets) = levels(&up, &ds, false)?;
    let (reverse, reverse_offsets) = levels(&up, &ds, true)?;
    let mut up_offsets = vec![0];
    let mut up_indices = Vec::new();
    for u in up {
        up_indices.extend(u);
        up_offsets.push(i32::try_from(up_indices.len())?);
    }
    let mut catch_ids = Vec::new();
    let mut catch_nodes = Vec::new();
    let flow_catch = selected
        .iter()
        .enumerate()
        .filter_map(|(i, &old)| field(&flow[old], "divide_id").map(|s| (s, i as i32)))
        .collect::<HashMap<_, _>>();
    for r in &layers["divides"].rows {
        let s = id(r, "divides")?;
        if let Some(&i) = flow_catch.get(&s) {
            catch_ids.push(s);
            catch_nodes.push(i)
        } else if let Some(i) = field(r, "id").and_then(|s| ids.iter().position(|t| t == &s)) {
            catch_ids.push(s);
            catch_nodes.push(i as i32)
        }
    }
    if catch_ids.iter().collect::<HashSet<_>>().len() != catch_ids.len() {
        return Err("duplicate catchment IDs".into());
    }
    let numeric = ids
        .iter()
        .map(|s| s.rsplit('-').next().unwrap_or(s).parse::<i64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()
        .filter(|v| v.iter().collect::<HashSet<_>>().len() == v.len());
    let parameters = if let Some(attrs) = layers.get("flowpath_attributes") {
        let map = attrs
            .rows
            .iter()
            .map(|r| Ok((id(r, "flowpath_attributes")?, &r.properties)))
            .collect::<Result<HashMap<_, _>>>()?;
        // Order is the production MC owner contract; preserve f64 attributes separately.
        let columns = [
            "length_m",
            "BtmWdth",
            "TopWdth",
            "TopWdthCC",
            "n",
            "nCC",
            "ChSlp",
            "So",
        ];
        Some(
            ids.par_iter()
                .map(|s| -> Result<[f32; 8]> {
                    let p = map
                        .get(s)
                        .ok_or_else(|| format!("no flowpath attributes for {s}"))?;
                    let mut a = [0.; 8];
                    for (i, k) in columns.iter().enumerate() {
                        let x = p
                            .get(*k)
                            .and_then(Value::as_f64)
                            .ok_or_else(|| format!("missing numeric {s}.{k}"))?;
                        if !x.is_finite() {
                            return Err(format!("nonfinite {s}.{k}").into());
                        }
                        a[i] = x as f32;
                    }
                    Ok(a)
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let network = Network {
        node_ids: ids,
        catchment_ids: catch_ids,
        numeric_node_ids: numeric,
        downstream: ds,
        downstream_ids: selected.iter().map(|&i| destinations[i].clone()).collect(),
        upstream_offsets: up_offsets,
        upstream_indices: up_indices,
        ordered_nodes: ordered,
        level_offsets: offsets,
        reverse_ordered_nodes: reverse,
        reverse_level_offsets: reverse_offsets,
        catchment_to_node: catch_nodes,
        routing_parameters: parameters,
        upstream_order: if c.upstream_order.is_some() {
            "explicit_native_roster"
        } else {
            "flowpaths_sqlite_scan_order_UNVERIFIED_NATIVE_REDUCTION"
        }
        .into(),
    };
    Ok(Prepared {
        layers,
        network,
        read_ms,
        topology_ms: timer.elapsed().as_secs_f64() * 1000.,
    })
}
pub(crate) fn levels(up: &[Vec<i32>], ds: &[i32], reverse: bool) -> Result<(Vec<i32>, Vec<i32>)> {
    let mut degree = (0..ds.len())
        .map(|i| {
            if reverse {
                usize::from(ds[i] >= 0)
            } else {
                up[i].len()
            }
        })
        .collect::<Vec<_>>();
    let mut current = (0..degree.len())
        .filter(|&i| degree[i] == 0)
        .map(|i| i as i32)
        .collect::<Vec<_>>();
    let mut ordered = Vec::with_capacity(ds.len());
    let mut offsets = vec![0];
    while !current.is_empty() {
        let mut next = Vec::new();
        for &i in &current {
            if reverse {
                for &j in &up[i as usize] {
                    degree[j as usize] -= 1;
                    if degree[j as usize] == 0 {
                        next.push(j)
                    }
                }
            } else {
                let j = ds[i as usize];
                if j >= 0 {
                    degree[j as usize] -= 1;
                    if degree[j as usize] == 0 {
                        next.push(j)
                    }
                }
            }
        }
        ordered.extend(current);
        offsets.push(i32::try_from(ordered.len())?);
        next.sort_unstable();
        current = next;
    }
    if ordered.len() != ds.len() {
        return Err("flowpath cycle: routing levels cannot be launched safely".into());
    }
    Ok((ordered, offsets))
}
pub fn properties(layers: &BTreeMap<String, Layer>) -> BTreeMap<&str, Vec<&Map<String, Value>>> {
    layers
        .iter()
        .map(|(role, l)| {
            (
                role.as_str(),
                l.rows.iter().map(|r| &r.properties).collect(),
            )
        })
        .collect()
}
