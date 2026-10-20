//! Associate valid, SHP-wound projected rings without changing source XY words.
//! This is a shared-builder intermediate, not a geometry validity checker.
use crate::{
    bundle::digest,
    source_geometry::{bytes, canonical_ring, unhex, Source},
    source_projection::signed_area,
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

type Ring = Vec<[u64; 2]>;
type Polygon = Vec<Ring>;
#[derive(Deserialize)]
struct Config {
    projected: Source,
    output: PathBuf,
}
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Group {
    exterior_ring_index: usize,
    interior_ring_indices: Vec<usize>,
    planar_area_m2_bits: u64,
}
#[derive(Deserialize)]
struct Record {
    source_id: i64,
    parts_xy_f64_bits: Vec<Ring>,
    signed_ring_area_m2_bits: Vec<u64>,
    polygons: Option<Vec<Group>>,
}
// Half-open ray crossing, with exact f64 boundary detection and no tolerance.
// This is deliberately not an exact-predicate or self-intersection validator.
fn location(p: [f64; 2], ring: &[[f64; 2]]) -> i8 {
    let mut inside = false;
    for e in ring.windows(2) {
        let (a, b) = (e[0], e[1]);
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        if cross == 0.
            && p[0] >= a[0].min(b[0])
            && p[0] <= a[0].max(b[0])
            && p[1] >= a[1].min(b[1])
            && p[1] <= a[1].max(b[1])
        {
            return 0;
        }
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if x > p[0] {
                inside = !inside;
            }
        }
    }
    if inside {
        1
    } else {
        -1
    }
}
fn group(parts: &[Ring], area_bits: &[u64]) -> Result<Vec<Group>> {
    if parts.is_empty() || parts.len() != area_bits.len() {
        return Err("polygon requires nonempty rings and matching area words".into());
    }
    let xy = parts
        .iter()
        .map(|r| r.iter().map(|p| p.map(f64::from_bits)).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let areas = xy
        .iter()
        .map(|r| {
            if r.iter().flatten().any(|v| !v.is_finite()) {
                return Err("nonfinite polygon XY".into());
            }
            signed_area(r)
        })
        .collect::<Result<Vec<_>>>()?;
    if areas
        .iter()
        .map(|a| a.to_bits())
        .ne(area_bits.iter().copied())
    {
        return Err("polygon area words disagree with projected rings".into());
    }
    let mut groups = areas
        .iter()
        .enumerate()
        .filter(|(_, a)| **a < 0.)
        .map(|(i, _)| Group {
            exterior_ring_index: i,
            interior_ring_indices: Vec::new(),
            planar_area_m2_bits: 0,
        })
        .collect::<Vec<_>>();
    if groups.is_empty() {
        return Err("SHP polygon has no clockwise exterior".into());
    }
    for (h, area) in areas.iter().enumerate().filter(|(_, a)| **a > 0.) {
        let mut candidates = Vec::new();
        for (g, shell) in groups.iter().enumerate() {
            let i = shell.exterior_ring_index;
            if -areas[i] <= *area {
                continue;
            }
            // All non-boundary hole vertices must be inside the shell. This
            // detects orphan/mixed-vertex holes; crossing edges still require
            // the shared geometry validator before runtime admission.
            let mut interior = false;
            let mut exterior = false;
            for p in xy[h].iter().take(xy[h].len() - 1) {
                match location(*p, &xy[i]) {
                    1 => interior = true,
                    -1 => exterior = true,
                    _ => {}
                }
            }
            if interior && !exterior {
                candidates.push((g, -areas[i]));
            }
        }
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        let Some(&(g, size)) = candidates.first() else {
            return Err(format!("SHP interior ring {h} has no enclosing exterior").into());
        };
        if candidates.get(1).is_some_and(|p| p.1 == size) {
            return Err("ambiguous equal-area enclosing exteriors".into());
        }
        groups[g].interior_ring_indices.push(h);
    }
    for g in &mut groups {
        let area = -areas[g.exterior_ring_index]
            - g.interior_ring_indices
                .iter()
                .map(|i| areas[*i])
                .sum::<f64>();
        if !area.is_finite() || area <= 0. {
            return Err("polygon group has nonpositive/nonfinite net area".into());
        }
        g.planar_area_m2_bits = area.to_bits();
    }
    Ok(groups)
}
pub fn prepare(path: &Path) -> Result<Value> {
    let t = Instant::now();
    let c: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut m: Value = serde_json::from_slice(&bytes(&c.projected)?)?;
    if m["schema"] != "arwen.source-projected-geometry.v1"
        || !matches!(m["source_shape_type"].as_u64(), Some(5 | 15))
    {
        return Err("expected pinned projected SHP Polygon/PolygonZ XY intermediate".into());
    }
    crate::weights::validate_grid_crs(m["target_crs"].as_str().ok_or("missing target CRS")?)?;
    if m["data_license"]
        .as_str()
        .is_none_or(|s| s.trim().is_empty())
    {
        return Err("source data licence required".into());
    }
    let records: Vec<Record> = serde_json::from_value(m["records"].clone())?;
    let matched: Vec<i64> = serde_json::from_value(m["matched_node_ids"].clone())?;
    let missing: Vec<i64> = serde_json::from_value(m["missing_node_ids"].clone())?;
    let ids = records.iter().map(|r| r.source_id).collect::<BTreeSet<_>>();
    let roster = matched
        .iter()
        .chain(&missing)
        .copied()
        .collect::<BTreeSet<_>>();
    if records.is_empty()
        || ids.len() != records.len()
        || ids.iter().any(|n| *n <= 0)
        || ids.iter().copied().collect::<Vec<_>>() != matched
        || roster.len() != matched.len() + missing.len()
        || roster.iter().any(|n| *n <= 0)
        || m["requested_node_count"].as_u64() != Some(roster.len() as u64)
    {
        return Err("inconsistent distinct positive source ID roster".into());
    }
    let groups = records
        .par_iter()
        .map(|r| group(&r.parts_xy_f64_bits, &r.signed_ring_area_m2_bits))
        .collect::<Result<Vec<_>>>()?;
    let polygons = groups.iter().map(Vec::len).sum::<usize>();
    let holes = groups
        .iter()
        .flatten()
        .map(|g| g.interior_ring_indices.len())
        .sum::<usize>();
    for (r, g) in m["records"].as_array_mut().unwrap().iter_mut().zip(groups) {
        r["polygons"] = serde_json::to_value(g)?;
    }
    m["schema"] = json!("arwen.source-polygon-groups.v1");
    m["projected_source_sha256"] = json!(c.projected.sha256);
    m["semantics"]["grouping"]=json!("SHP clockwise exterior / counter-clockwise interior; hole assigned to smallest exterior containing every non-boundary hole vertex, at least one interior; exact source ring indices and XY bits retained; f64 predicates, no tolerance; assumes simple, noncrossing, correctly wound source rings");
    m["semantics"]["scope"]=json!("grouped original-source projected intermediate only; not polygon validity, inter-record overlap, geodesic length, native CFE area, full source coverage, parameters/stores, GPKG or runtime admission; shared writer/checker still required");
    let payload = serde_json::to_vec(&m)?;
    std::fs::create_dir(&c.output)?;
    std::fs::write(c.output.join("polygons.json"), &payload)?;
    Ok(
        json!({"manifest":c.output.join("polygons.json"),"sha256":format!("{:x}",Sha256::digest(&payload)),"bytes":payload.len(),
        "record_count":records.len(),"polygon_count":polygons,"interior_ring_count":holes,"missing_node_ids":missing,
        "rayon_workers":rayon::current_num_threads(),"source":crate::SOURCE_REVISION,"wall_ms":t.elapsed().as_secs_f64()*1000.,
        "parallelism":"Rayon across source records; grouping within each record serial; CPU prep only"}),
    )
}
// Preserve exterior/hole associations and multiplicities. Normalize only ring
// start/direction, hole order and polygon order for the native comparison.
fn canonical(map: BTreeMap<i64, Vec<Polygon>>) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for (id, ps) in map {
        let mut polygons = Vec::new();
        for mut p in ps {
            if p.is_empty() {
                return Err("empty referee polygon".into());
            }
            let shell = canonical_ring(p.remove(0))?;
            let mut holes = p
                .into_iter()
                .map(canonical_ring)
                .collect::<Result<Vec<_>>>()?;
            holes.sort();
            let mut all = vec![shell];
            all.extend(holes);
            polygons.push(all);
        }
        polygons.sort();
        out.extend(id.to_le_bytes());
        out.extend((polygons.len() as u64).to_le_bytes());
        for p in polygons {
            out.extend((p.len() as u64).to_le_bytes());
            for ring in p {
                out.extend((ring.len() as u64).to_le_bytes());
                for xy in ring {
                    for b in xy {
                        out.extend(b.to_le_bytes());
                    }
                }
            }
        }
    }
    Ok(out)
}
fn grouped(records: Vec<Record>) -> Result<BTreeMap<i64, Vec<Polygon>>> {
    let mut map = BTreeMap::new();
    for r in records {
        let mut used = BTreeSet::new();
        let mut ps = Vec::new();
        for g in r.polygons.ok_or("missing polygon grouping")? {
            let mut p = Vec::new();
            for i in std::iter::once(g.exterior_ring_index).chain(g.interior_ring_indices) {
                if !used.insert(i) {
                    return Err("polygon grouping repeats source ring".into());
                }
                p.push(
                    r.parts_xy_f64_bits
                        .get(i)
                        .ok_or("polygon ring index outside source")?
                        .clone(),
                );
            }
            ps.push(p);
        }
        if used.len() != r.parts_xy_f64_bits.len()
            || ps.is_empty()
            || map.insert(r.source_id, ps).is_some()
        {
            return Err("polygon grouping drops rings or duplicates IDs".into());
        }
    }
    Ok(map)
}
pub fn replay(manifest: &Path, native: &Path) -> Result<Value> {
    let t = Instant::now();
    let m: Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    if m["schema"] != "arwen.source-polygon-groups.v1" {
        return Err("expected grouped polygon schema".into());
    }
    let actual = grouped(serde_json::from_value(m["records"].clone())?)?;
    let text = std::fs::read_to_string(native)?;
    let mut lines = text.lines();
    if lines.next() != Some("source_id,wkb") {
        return Err("expected GDAL source_id,wkb header".into());
    }
    let mut expected = BTreeMap::new();
    for line in lines {
        let (a, b) = line.split_once(',').ok_or("invalid native polygon row")?;
        let id = crate::source_network::id(a.trim_matches('"'))?;
        let mut gp = b"GP\0\x01\0\0\0\0".to_vec();
        gp.extend(unhex(b.trim_matches('"'))?);
        let ps = crate::weights::polygons(&gp)?
            .into_iter()
            .map(|p| {
                p.into_iter()
                    .map(|r| r.into_iter().map(|xy| xy.map(f64::to_bits)).collect())
                    .collect()
            })
            .collect();
        if expected.insert(id, ps).is_some() {
            return Err("duplicate native polygon ID".into());
        }
    }
    let ids = actual.len();
    let native_ids = expected.len();
    let (a, b) = rayon::join(|| canonical(actual), || canonical(expected));
    let (a, b) = (a?, b?);
    let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
    Ok(
        json!({"strict":true,"pass":diff==0,"different_canonical_bytes":diff,"compared_canonical_bytes":b.len(),"actual_records":ids,"native_records":native_ids,
        "actual_sha256":format!("{:x}",Sha256::digest(&a)),"expected_sha256":format!("{:x}",Sha256::digest(&b)),"native_capture_sha256":digest(native)?,"grouped_manifest_sha256":digest(manifest)?,
        "scope":"XY words and exterior/hole polygon associations; ring rotation/direction and hole/polygon order normalized; not geometry validity or native physical area",
        "source":crate::SOURCE_REVISION,"wall_ms":t.elapsed().as_secs_f64()*1000.}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    fn square(a: f64, b: f64, reverse: bool) -> Ring {
        let mut r = vec![[a, a], [a, b], [b, b], [b, a], [a, a]];
        if reverse {
            r.reverse();
        }
        r.into_iter().map(|p| p.map(f64::to_bits)).collect()
    }
    fn areas(r: &[Ring]) -> Vec<u64> {
        r.iter()
            .map(|r| {
                signed_area(&r.iter().map(|p| p.map(f64::from_bits)).collect::<Vec<_>>())
                    .unwrap()
                    .to_bits()
            })
            .collect()
    }
    #[test]
    fn shells_holes_islands_and_smallest_container() {
        let r = vec![
            square(0., 10., false),
            square(1., 9., true),
            square(2., 8., false),
            square(3., 7., true),
            square(20., 30., false),
        ];
        let g = group(&r, &areas(&r)).unwrap();
        assert_eq!(
            g.iter()
                .map(|g| (g.exterior_ring_index, g.interior_ring_indices.clone()))
                .collect::<Vec<_>>(),
            vec![(0, vec![1]), (2, vec![3]), (4, vec![])]
        );
        assert_eq!(
            g.iter()
                .map(|g| f64::from_bits(g.planar_area_m2_bits))
                .sum::<f64>(),
            156.
        );
    }
    #[test]
    fn rejects_orphan_zero_open_nan_and_wrong_area() {
        let r = vec![square(0., 10., false), square(20., 21., true)];
        assert!(group(&r, &areas(&r)).is_err());
        let r = vec![square(0., 10., true)];
        assert!(group(&r, &areas(&r)).is_err());
        assert!(group(&[vec![[0, 0]; 4]], &[0]).is_err());
        let mut r = vec![square(0., 10., false)];
        let a = areas(&r);
        r[0].pop();
        assert!(group(&r, &a).is_err());
        r[0][0][0] = f64::NAN.to_bits();
        assert!(group(&r, &a).is_err());
        let r = vec![square(0., 10., false)];
        assert!(group(&r, &[1]).is_err());
    }
    #[test]
    fn native_comparator_preserves_groups_and_normalizes_only_order() {
        let (s, h, i) = (
            square(0., 10., false),
            square(1., 9., true),
            square(2., 8., false),
        );
        let pack = |p| canonical(BTreeMap::from([(7, p)])).unwrap();
        let expected = pack(vec![vec![s.clone(), h.clone()], vec![i.clone()]]);
        let mut rotated = s.clone();
        rotated.pop();
        rotated.rotate_left(2);
        rotated.push(rotated[0]);
        rotated.reverse();
        assert_eq!(
            expected,
            pack(vec![vec![i.clone()], vec![rotated, h.clone()]])
        );
        assert_ne!(expected, pack(vec![vec![s], vec![i, h]]));
    }
}
