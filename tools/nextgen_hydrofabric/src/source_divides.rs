//! Evidence for pairing original-source routing reaches with local divides.
//! A zero source area explains a missing polygon; it never creates a polygon,
//! proves domain coverage, or authorizes dropping the routing reach.
use crate::{
    dbf::Table,
    source_geometry::{bytes, parts, u32le, Shape, Source},
    source_network::id,
    Result,
};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    time::Instant,
};

#[derive(Deserialize)]
struct Geometry {
    shp: Source,
    shx: Source,
    dbf: Source,
    id_column: String,
}
#[derive(Deserialize)]
struct Attributes {
    dbf: Source,
    id_column: String,
    /// Source local incremental area, never total upstream drainage area.
    local_area_column: String,
    /// Units are named, not converted or inferred.
    local_area_unit: String,
    evidence_columns: Vec<String>,
}
#[derive(Deserialize)]
struct Config {
    network: Source,
    polygons: Geometry,
    attributes: Attributes,
    data_license: String,
    provenance: Value,
}
fn local_area(raw: &str) -> Result<f64> {
    let value: f64 = raw.parse()?;
    if !value.is_finite() || value < 0. {
        return Err("source local area must be finite and nonnegative".into());
    }
    Ok(value)
}
fn sealed_table(source: &Source) -> Result<Table> {
    let table = Table::open(&source.path)?;
    if table.sha256 != source.sha256 {
        return Err("source DBF digest mismatch".into());
    }
    Ok(table)
}
pub fn audit(path: &Path) -> Result<Value> {
    let start = Instant::now();
    let c: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    if c.data_license.trim().is_empty() || c.attributes.local_area_unit.trim().is_empty() {
        return Err("source licence and local area unit must be named".into());
    }
    let network: Value = serde_json::from_slice(&bytes(&c.network)?)?;
    if network["schema"] != "arwen.source-network-subset.v1" {
        return Err("expected original-source network subset schema".into());
    }
    let ids: Vec<i64> = serde_json::from_value(network["node_ids"].clone())?;
    let selected = ids.iter().copied().collect::<BTreeSet<_>>();
    if selected.is_empty() || selected.len() != ids.len() || selected.iter().any(|n| *n <= 0) {
        return Err("network requires distinct positive source IDs".into());
    }
    let (geometry, attributes) = rayon::join(
        || -> Result<_> {
            let shape = Shape::open(bytes(&c.polygons.shp)?, bytes(&c.polygons.shx)?)?;
            if u32le(&shape.shp, 32)? != 5 {
                return Err("divide coverage requires Polygon source geometry".into());
            }
            let dbf = sealed_table(&c.polygons.dbf)?;
            if dbf.count != shape.records.len() {
                return Err("DBF/SHP physical record counts differ".into());
            }
            let rows = dbf.indexed_rows(&[&c.polygons.id_column])?;
            let found = rows
                .par_iter()
                .enumerate()
                .map(|(physical_record, row)| -> Result<_> {
                    let Some(row) = row else { return Ok(None) };
                    let source_id = id(&row[0])?;
                    if !selected.contains(&source_id) {
                        return Ok(None);
                    }
                    let raw = shape.record(physical_record)?;
                    let nonnull = u32le(raw, 0)? != 0;
                    if nonnull {
                        parts(raw, 5)?;
                    }
                    Ok(Some((source_id, physical_record, nonnull)))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let mut map = BTreeMap::new();
            for (n, physical, nonnull) in found {
                if map.insert(n, (physical, nonnull)).is_some() {
                    return Err(format!("duplicate selected polygon source ID {n}").into());
                }
            }
            Ok((map, dbf.count))
        },
        || -> Result<_> {
            let dbf = sealed_table(&c.attributes.dbf)?;
            let mut columns = vec![
                c.attributes.id_column.as_str(),
                c.attributes.local_area_column.as_str(),
            ];
            columns.extend(c.attributes.evidence_columns.iter().map(String::as_str));
            let mut map = BTreeMap::new();
            for (physical, row) in dbf.indexed_rows(&columns)?.into_iter().enumerate() {
                let Some(row) = row else { continue };
                let n = id(&row[0])?;
                if !selected.contains(&n) {
                    continue;
                }
                let area = local_area(&row[1])?;
                if map.insert(n, (physical, area, row)).is_some() {
                    return Err(format!("duplicate selected attribute source ID {n}").into());
                }
            }
            Ok((map, dbf.count))
        },
    );
    let (geometry, geometry_count) = geometry?;
    let (attributes, attribute_count) = attributes?;
    let mut rows = Vec::new();
    for n in selected {
        let (physical, area, raw) = attributes
            .get(&n)
            .ok_or_else(|| format!("source attributes absent for selected ID {n}"))?;
        let polygon = geometry.get(&n);
        let present = polygon.is_some_and(|(_, nonnull)| *nonnull);
        let classification = match (present, *area == 0.) {
            (true, false) => "polygon_positive_local_area",
            (true, true) => "polygon_zero_local_area",
            (false, true) => "absent_polygon_zero_local_area",
            (false, false) => "absent_polygon_positive_local_area",
        };
        rows.push(json!({"source_id":n,"classification":classification,
            "polygon_physical_record":polygon.map(|p|p.0),"polygon_nonnull":present,
            "attribute_physical_record":physical,"local_area_raw":raw[1],"local_area_f64_bits":area.to_bits(),
            "evidence_raw":raw[2..]}));
    }
    let mut counts = BTreeMap::<String, usize>::new();
    for row in &rows {
        *counts
            .entry(row["classification"].as_str().unwrap().to_owned())
            .or_default() += 1;
    }
    Ok(
        json!({"schema":"arwen.source-divide-coverage.v1","source":crate::SOURCE_REVISION,
        "network_sha256":c.network.sha256,"source_hashes":{"shp":c.polygons.shp.sha256,"shx":c.polygons.shx.sha256,"polygon_dbf":c.polygons.dbf.sha256,"attributes_dbf":c.attributes.dbf.sha256},
        "local_area_column":c.attributes.local_area_column,"local_area_unit":c.attributes.local_area_unit,"evidence_columns":c.attributes.evidence_columns,
        "data_license":c.data_license,"provenance":c.provenance,"source_polygon_record_count":geometry_count,"source_attribute_record_count":attribute_count,
        "selected_node_count":rows.len(),"counts":counts,"rows":rows,"rayon_workers":rayon::current_num_threads(),
        "parallelism":"independent polygon/attribute sources and physical DBF record decoding on Rayon; final sorted ID join is serial; CPU prep only",
        "scope":"Source subset presence and verbatim local-area evidence only. Zero local area is not zero upstream flow. Every selected routing ID is retained. No fabricated divides, area repair, total-drainage substitution, domain/tile coverage, geometric union, physical parameters/state or runtime-fabric admission.",
        "wall_ms":start.elapsed().as_secs_f64()*1000.}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_area_does_not_repair_missing_or_invalid_values() {
        for raw in ["", "NaN", "inf", "-0.01", "missing"] {
            assert!(local_area(raw).is_err(), "{raw}");
        }
        assert_eq!(local_area("0.00000000000e+000").unwrap(), 0.);
        assert_eq!(local_area("1.77404580000e+003").unwrap(), 1774.0458);
    }
}
