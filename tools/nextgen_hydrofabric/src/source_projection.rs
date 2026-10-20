//! Pinned original-source XY projection for the shared fabric builder.
//! This intermediate is deliberately not a GeoPackage or runtime fabric.
use crate::{
    bundle::digest,
    dbf::Table,
    source_geometry::*,
    source_network::id,
    weights::{validate_grid_crs, Projection},
    Result,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
struct Config {
    shp: Source,
    shx: Source,
    dbf: Source,
    prj: Source,
    network: Source,
    waterbody_source_index: Option<usize>,
    id_column: String,
    target_crs: String,
    output: PathBuf,
    data_license: String,
    provenance: Value,
}
#[derive(Serialize, Deserialize)]
struct Record {
    source_id: i64,
    physical_record: usize,
    parts_xy_f64_bits: Vec<Vec<[u64; 2]>>,
    // SHP specification: clockwise exterior, counter-clockwise interior.
    // Ring grouping/containment is still owned by final fabric assembly.
    signed_ring_area_m2: Option<Vec<f64>>,
    source_winding_area_m2: Option<f64>,
    signed_ring_area_m2_bits: Option<Vec<u64>>,
    source_winding_area_m2_bits: Option<u64>,
}
pub(crate) fn signed_area(r: &[[f64; 2]]) -> Result<f64> {
    if r.len() < 4 || r.first() != r.last() {
        return Err("projected source polygon ring is not closed".into());
    }
    let o = r[0];
    let a = r
        .windows(2)
        .map(|p| (p[0][0] - o[0]) * (p[1][1] - o[1]) - (p[1][0] - o[0]) * (p[0][1] - o[1]))
        .sum::<f64>()
        * 0.5;
    if !a.is_finite() || a == 0. {
        return Err("projected source ring has zero/nonfinite signed area".into());
    }
    Ok(a)
}
pub fn prepare(path: &Path) -> Result<Value> {
    let t = Instant::now();
    let c: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    if c.data_license.trim().is_empty() {
        return Err("actual source data_license required".into());
    }
    validate_grid_crs(&c.target_crs)?;
    let mut loaded = [&c.shp, &c.shx, &c.prj, &c.network]
        .par_iter()
        .map(|s| bytes(s))
        .collect::<Result<Vec<_>>>()?
        .into_iter();
    let shape = Shape::open(loaded.next().unwrap(), loaded.next().unwrap())?;
    let shape_type = u32le(&shape.shp, 32)?;
    let prj = loaded.next().unwrap();
    let source_crs = std::str::from_utf8(&prj)?;
    // Validate once even if the roster has no source records.
    Projection::new(source_crs, &c.target_crs)?;
    let net: Value = serde_json::from_slice(&loaded.next().unwrap())?;
    let ids = requested_ids(
        &net,
        c.waterbody_source_index,
        &c.dbf.sha256,
        &c.id_column,
        shape_type,
    )?;
    let selected = ids.iter().copied().collect::<BTreeSet<_>>();
    if selected.is_empty() || selected.len() != ids.len() || selected.iter().any(|n| *n <= 0) {
        return Err("network requires distinct positive source IDs".into());
    }
    let table = Table::open(&c.dbf.path)?;
    if table.sha256 != c.dbf.sha256 || table.count != shape.records.len() {
        return Err("DBF digest or SHP physical record count mismatch".into());
    }
    let rows = table.indexed_rows(&[&c.id_column])?;
    let read_ms = t.elapsed().as_secs_f64() * 1000.;
    let project = Instant::now();
    // Contexts never cross worker boundaries; no global PROJ state or network.
    let context_count = std::sync::atomic::AtomicUsize::new(0);
    let chunk_min = rows.len().div_ceil(rayon::current_num_threads()).max(1);
    let mut records = rows
        .par_iter()
        .enumerate()
        .with_min_len(chunk_min)
        .map_init(
            || None::<Projection>,
            |projection, (i, row)| -> Result<Option<Record>> {
                let Some(row) = row else { return Ok(None) };
                let n = id(&row[0])?;
                if !selected.contains(&n) {
                    return Ok(None);
                }
                let raw = shape.record(i)?;
                if u32le(raw, 0)? == 0 {
                    return Ok(None);
                }
                let source_parts = parts(raw, shape_type)?;
                if projection.is_none() {
                    *projection = Some(Projection::new(source_crs, &c.target_crs)?);
                    context_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let xy = source_parts
                    .iter()
                    .map(|p| {
                        p.points
                            .iter()
                            .map(|v| {
                                projection
                                    .as_ref()
                                    .unwrap()
                                    .transform([f64::from_bits(v[0]), f64::from_bits(v[1])])
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                let areas = if [5, 15].contains(&shape_type) {
                    Some(
                        xy.iter()
                            .map(|r| signed_area(r))
                            .collect::<Result<Vec<_>>>()?,
                    )
                } else {
                    None
                };
                let total = areas.as_ref().map(|a| -a.iter().sum::<f64>());
                if total.is_some_and(|a| !a.is_finite() || a <= 0.) {
                    return Err(format!(
                        "source polygon {n} has invalid SHP exterior/interior winding area"
                    )
                    .into());
                }
                Ok(Some(Record {
                    source_id: n,
                    physical_record: i,
                    parts_xy_f64_bits: xy
                        .into_iter()
                        .map(|r| {
                            r.into_iter()
                                .map(|p| [p[0].to_bits(), p[1].to_bits()])
                                .collect()
                        })
                        .collect(),
                    signed_ring_area_m2_bits: areas
                        .as_ref()
                        .map(|a| a.iter().map(|x| x.to_bits()).collect()),
                    source_winding_area_m2_bits: total.map(f64::to_bits),
                    signed_ring_area_m2: areas,
                    source_winding_area_m2: total,
                }))
            },
        )
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    records.par_sort_unstable_by_key(|r| (r.source_id, r.physical_record));
    if records.is_empty() {
        return Err("no active non-NULL source geometry matches requested network".into());
    }
    let matched = records.iter().map(|r| r.source_id).collect::<BTreeSet<_>>();
    let missing = selected.difference(&matched).copied().collect::<Vec<_>>();
    let projection_ms = project.elapsed().as_secs_f64() * 1000.;
    let mut result = json!({"schema":"arwen.source-projected-geometry.v1","source_shape_type":shape_type,
        "source_crs_wkt":source_crs,"target_crs":c.target_crs,"network_sha256":c.network.sha256,
        "sources":{"shp":c.shp.sha256,"shx":c.shx.sha256,"dbf":c.dbf.sha256,"prj":c.prj.sha256},
        "data_license":c.data_license,"provenance":c.provenance,"requested_node_count":selected.len(),
        "matched_node_ids":matched,"missing_node_ids":missing,"records":records,
        "semantics":{"coordinates":"XY metres as exact unsigned-u64 IEEE754 bits (parts_xy_f64_bits), decoded by f64::from_bits; diagnostic numeric areas have authoritative _bits companions; visualization axis order; source physical records/parts/vertex direction retained; no simplification/densification; source Z/M explicitly excluded",
        "area":"planar m2; negative sum of signed rings under SHP clockwise-exterior convention; only target EPSG:5070 is the adopted CONUS equal-area contract",
        "scope":"intermediate geometry only; no ring containment/validity, geodesic length, vertical datum, native CFE area, full source coverage, physical parameters/stores, GPKG or runtime admission"}});
    if let Some(index) = c.waterbody_source_index {
        result["selection"] = json!({"schema":net["schema"],"source_index":index,
            "id_scope":"matched/missing node IDs above are source polygon IDs, not flowpath IDs",
            "scope":"referenced polygon XY projection/planar area only; no lake classification, outlet, storage or hydraulic inference"});
    }
    let payload = serde_json::to_vec(&result)?;
    std::fs::create_dir(&c.output)?;
    std::fs::write(c.output.join("projected.json"), &payload)?;
    Ok(
        json!({"manifest":c.output.join("projected.json"),"sha256":format!("{:x}",Sha256::digest(&payload)),
        "bytes":payload.len(),"record_count":records.len(),"point_count":records.iter().flat_map(|r|&r.parts_xy_f64_bits).map(Vec::len).sum::<usize>(),
        "missing_node_ids":missing,"source":crate::SOURCE_REVISION,"rayon_workers":rayon::current_num_threads(),"projection_worker_contexts":context_count.load(std::sync::atomic::Ordering::Relaxed),
        "parallelism":"independent source reads/physical record validation/DBF join and per-worker PROJ contexts; startup only; network disabled",
        "timings_ms":{"read_hash_join":read_ms,"project":projection_ms,"total":t.elapsed().as_secs_f64()*1000.}}),
    )
}

pub fn replay(manifest: &Path, native: &Path) -> Result<Value> {
    let t = Instant::now();
    let m: Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    if m["schema"] != "arwen.source-projected-geometry.v1" {
        return Err("expected projected source geometry schema".into());
    }
    let polygon = matches!(m["source_shape_type"].as_u64(), Some(5 | 15));
    let records: Vec<Record> = serde_json::from_value(m["records"].clone())?;
    let mut actual = BTreeMap::<i64, Vec<Part>>::new();
    for r in records {
        actual
            .entry(r.source_id)
            .or_default()
            .extend(r.parts_xy_f64_bits.into_iter().map(|p| Part {
                axes: 0,
                points: p.into_iter().map(|xy| [xy[0], xy[1], 0, 0]).collect(),
            }));
    }
    let text = std::fs::read_to_string(native)?;
    let mut lines = text.lines();
    if lines.next() != Some("source_id,wkb") {
        return Err("expected GDAL source_id,wkb header".into());
    }
    let mut expected = BTreeMap::<i64, Vec<Part>>::new();
    for line in lines {
        let (a, b) = line
            .split_once(',')
            .ok_or("invalid native projection row")?;
        let raw = unhex(b.trim_matches('"'))?;
        let ps = if polygon {
            let mut gp = b"GP\0\x01\0\0\0\0".to_vec();
            gp.extend(raw);
            crate::weights::polygons(&gp)?
                .into_iter()
                .flatten()
                .map(|p| Part {
                    axes: 0,
                    points: p
                        .into_iter()
                        .map(|xy| [xy[0].to_bits(), xy[1].to_bits(), 0, 0])
                        .collect(),
                })
                .collect()
        } else {
            native_lines(&raw)?
        };
        expected
            .entry(id(a.trim_matches('"'))?)
            .or_default()
            .extend(ps);
    }
    let pack = |map: BTreeMap<i64, Vec<Part>>| {
        if polygon {
            canonical(
                map.into_iter()
                    .map(|(id, ps)| {
                        (
                            id,
                            ps.into_iter()
                                .map(|p| p.points.into_iter().map(|v| [v[0], v[1]]).collect())
                                .collect(),
                        )
                    })
                    .collect(),
            )
        } else {
            canonical_lines(map)
        }
    };
    let (a, b) = rayon::join(|| pack(actual), || pack(expected));
    let a = a?;
    let b = b?;
    let different = a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
    Ok(
        json!({"strict":true,"pass":different==0,"different_canonical_bytes":different,
        "compared_canonical_bytes":b.len(),"actual_sha256":format!("{:x}",Sha256::digest(&a)),
        "expected_sha256":format!("{:x}",Sha256::digest(&b)),"native_capture_sha256":digest(native)?,
        "scope":"projected XY word/sequence equality only; polygon ring rotation/winding/grouping normalized; directed line order retained; no area/containment/physical claim",
        "wall_ms":t.elapsed().as_secs_f64()*1000.,"source":crate::SOURCE_REVISION}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projected_words_survive_default_json_decoder() {
        let xy = [-2286794.999999999_f64, 2147594.9999999953_f64];
        let words = xy.map(f64::to_bits);
        let encoded = serde_json::to_vec(&words).unwrap();
        let decoded: [u64; 2] = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, words);
        assert_eq!(decoded.map(f64::from_bits), xy);
    }
    #[test]
    fn translated_signed_area_and_hole() {
        let shell = [
            [1e9, 1e9],
            [1e9, 1e9 + 4.],
            [1e9 + 4., 1e9 + 4.],
            [1e9 + 4., 1e9],
            [1e9, 1e9],
        ];
        let hole = [
            [1e9 + 1., 1e9 + 1.],
            [1e9 + 3., 1e9 + 1.],
            [1e9 + 3., 1e9 + 3.],
            [1e9 + 1., 1e9 + 3.],
            [1e9 + 1., 1e9 + 1.],
        ];
        assert_eq!(signed_area(&shell).unwrap(), -16.);
        assert_eq!(signed_area(&hole).unwrap(), 4.);
        assert_eq!(
            -signed_area(&shell).unwrap() - signed_area(&hole).unwrap(),
            12.
        );
        assert!(signed_area(&[[0., 0.]; 4]).is_err());
    }
}
