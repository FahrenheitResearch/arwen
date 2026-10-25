//! Original-source flowpath lengths for the shared fabric builder.
//! Source CRS -> WGS84 with existing PROJ; ellipsoidal XY distances through
//! PROJ's public GeographicLib C API. No GDAL runtime or planar-length shortcut.
use crate::{
    dbf::Table,
    source_geometry::{bytes, parts, u32le, Shape, Source},
    source_network::id,
    weights::Projection,
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

// Public geod_geodesic layout, PROJ 9.7.1 src/geodesic.h (GeographicLib 2.2).
// Copyright Charles Karney (2012-2022), MIT/X11; existing PROJ MIT dependency.
// No geodesic implementation is copied: geod_init/inverse execute in libproj.
#[repr(C)]
struct Geodesic {
    a: f64,
    f: f64,
    f1: f64,
    e2: f64,
    ep2: f64,
    n: f64,
    b: f64,
    c2: f64,
    etol2: f64,
    a3x: [f64; 6],
    c3x: [f64; 15],
    c4x: [f64; 21],
}
// Pinned bundled PROJ/SQLite; no host library lookup.
extern crate hydro_proj;
extern "C" {
    fn geod_init(g: *mut Geodesic, a: f64, f: f64);
    fn geod_inverse(
        g: *const Geodesic,
        lat1: f64,
        lon1: f64,
        lat2: f64,
        lon2: f64,
        distance: *mut f64,
        azi1: *mut f64,
        azi2: *mut f64,
    );
}


impl Geodesic {
    fn wgs84() -> Self {
        // The C initializer fills the public struct before it is read.
        let mut g = std::mem::MaybeUninit::uninit();
        unsafe {
            geod_init(g.as_mut_ptr(), 6378137., 1. / 298.257223563);
            g.assume_init()
        }
    }
    fn length(&self, xy: &[[f64; 2]]) -> Result<f64> {
        if xy.len() < 2
            || xy.iter().any(|p| {
                !p[0].is_finite() || !p[1].is_finite() || p[0].abs() > 180. || p[1].abs() > 90.
            })
        {
            return Err(
                "flowpath needs at least two finite WGS84 longitude/latitude vertices".into(),
            );
        }
        let mut length = 0.;
        for e in xy.windows(2) {
            let mut s = 0.;
            unsafe {
                geod_inverse(
                    self,
                    e[0][1],
                    e[0][0],
                    e[1][1],
                    e[1][0],
                    &mut s,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
            }
            if !s.is_finite() || s < 0. {
                return Err("invalid ellipsoidal segment length".into());
            }
            length += s;
        }
        if !length.is_finite() {
            return Err("flowpath length overflow".into());
        }
        Ok(length)
    }
}
#[derive(Deserialize)]
struct Config {
    shp: Source,
    shx: Source,
    dbf: Source,
    prj: Source,
    network: Source,
    id_column: String,
    output: PathBuf,
    data_license: String,
    provenance: Value,
}
#[derive(Serialize, Deserialize)]
struct Record {
    source_id: i64,
    physical_record: usize,
    point_count: usize,
    part_point_counts: Vec<usize>,
    part_lengths_m_bits: Vec<u64>,
    length_m_bits: u64,
}
fn validate_record(r: &Record) -> Result<()> {
    if r.source_id <= 0
        || r.part_point_counts.is_empty()
        || r.part_point_counts.len() != r.part_lengths_m_bits.len()
        || r.part_point_counts.iter().any(|n| *n < 2)
        || r.part_point_counts
            .iter()
            .try_fold(0usize, |a, n| a.checked_add(*n))
            != Some(r.point_count)
    {
        return Err("invalid flowpath source ID/part/point roster".into());
    }
    let mut sum = 0.;
    for b in &r.part_lengths_m_bits {
        let v = f64::from_bits(*b);
        if !v.is_finite() || v < 0. {
            return Err("invalid geodesic part length".into());
        }
        sum += v;
    }
    if sum <= 0. || !sum.is_finite() || sum.to_bits() != r.length_m_bits {
        return Err("nonpositive flowpath or total length words disagree with parts".into());
    }
    Ok(())
}
pub fn prepare(path: &Path) -> Result<Value> {
    let t = Instant::now();
    let c: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    if c.data_license.trim().is_empty() {
        return Err("actual source data_license required".into());
    }
    let mut loaded = [&c.shp, &c.shx, &c.prj, &c.network]
        .par_iter()
        .map(|s| bytes(s))
        .collect::<Result<Vec<_>>>()?
        .into_iter();
    let shape = Shape::open(loaded.next().unwrap(), loaded.next().unwrap())?;
    let shape_type = u32le(&shape.shp, 32)?;
    if ![3, 13].contains(&shape_type) {
        return Err("geodesic flowpaths require SHP PolyLine/PolyLineZ".into());
    }
    let prj = loaded.next().unwrap();
    let source_crs = std::str::from_utf8(&prj)?;
    Projection::new(source_crs, "EPSG:4326")?;
    let net: Value = serde_json::from_slice(&loaded.next().unwrap())?;
    if net["schema"] != "arwen.source-network-subset.v1" {
        return Err("expected source network subset schema".into());
    }
    let ids: Vec<i64> = serde_json::from_value(net["node_ids"].clone())?;
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
    let calc = Instant::now();
    let g = Geodesic::wgs84();
    let mut records = rows
        .par_iter()
        .enumerate()
        .with_min_len(rows.len().div_ceil(rayon::current_num_threads()).max(1))
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
                let ps = parts(raw, shape_type)?;
                if projection.is_none() {
                    *projection = Some(Projection::new(source_crs, "EPSG:4326")?);
                }
                let mut counts = Vec::new();
                let mut lengths = Vec::new();
                let mut total = 0.;
                for p in ps {
                    let xy = p
                        .points
                        .iter()
                        .map(|v| {
                            projection
                                .as_ref()
                                .unwrap()
                                .transform([f64::from_bits(v[0]), f64::from_bits(v[1])])
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let length = g.length(&xy)?;
                    counts.push(xy.len());
                    lengths.push(length.to_bits());
                    total += length;
                }
                let r = Record {
                    source_id: n,
                    physical_record: i,
                    point_count: counts.iter().sum(),
                    part_point_counts: counts,
                    part_lengths_m_bits: lengths,
                    length_m_bits: total.to_bits(),
                };
                validate_record(&r)?;
                Ok(Some(r))
            },
        )
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    records.par_sort_unstable_by_key(|r| r.source_id);
    if records.is_empty() || records.windows(2).any(|w| w[0].source_id == w[1].source_id) {
        return Err(
            "flowpaths require nonempty, unique matched source IDs (no double-counted lengths)"
                .into(),
        );
    }
    let matched = records.iter().map(|r| r.source_id).collect::<BTreeSet<_>>();
    let missing = selected.difference(&matched).copied().collect::<Vec<_>>();
    let geodesic_ms = calc.elapsed().as_secs_f64() * 1000.;
    let result = json!({"schema":"arwen.source-geodesic-lengths.v1","source_shape_type":shape_type,
        "source_crs_wkt":source_crs,"length_crs":"EPSG:4326","ellipsoid":{"a_m":6378137.,"inverse_flattening":298.257223563},
        "network_sha256":c.network.sha256,"sources":{"shp":c.shp.sha256,"shx":c.shx.sha256,"dbf":c.dbf.sha256,"prj":c.prj.sha256},
        "data_license":c.data_license,"provenance":c.provenance,"requested_node_count":selected.len(),
        "matched_node_ids":matched,"missing_node_ids":missing,"records":records,
        "semantics":{"length":"metres as exact u64 f64 bits; WGS84 shortest ellipsoidal distance per consecutive XY segment; sequential sum within each original source part, then sequential part sum; never join disconnected parts; Z/M excluded",
        "projection":"normalized visualization axis order, longitude/latitude degrees; installed PROJ operation with network disabled; runtime library/proj.db pins required in receipt; no assertion of survey-grade NAD83/WGS84 datum accuracy",
        "scope":"source intermediate, not collapsed-network length/native routing parameter or fabric/provider admission; no planar/spherical substitute, densification or source-coverage repair"}});
    let payload = serde_json::to_vec(&result)?;
    std::fs::create_dir(&c.output)?;
    std::fs::write(c.output.join("lengths.json"), &payload)?;
    Ok(
        json!({"manifest":c.output.join("lengths.json"),"sha256":format!("{:x}",Sha256::digest(&payload)),
        "bytes":payload.len(),"record_count":records.len(),"point_count":records.iter().map(|r|r.point_count).sum::<usize>(),
        "part_count":records.iter().map(|r|r.part_point_counts.len()).sum::<usize>(),"missing_node_ids":missing,
        "source":crate::SOURCE_REVISION,"rayon_workers":rayon::current_num_threads(),
        "timings_ms":{"read_hash_join":read_ms,"geodesic":geodesic_ms,"total":t.elapsed().as_secs_f64()*1000.}}),
    )
}
// Referee CSV is generated by native GDAL from the ORIGINAL source, not this output.
// Decimal integers carry exact f64 bits, with original source part/vertex counts.
pub fn replay(path: &Path, native: &Path) -> Result<Value> {
    let t = Instant::now();
    let m: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    if m["schema"] != "arwen.source-geodesic-lengths.v1" || m["length_crs"] != "EPSG:4326" {
        return Err("expected WGS84 source geodesic lengths schema".into());
    }
    let records: Vec<Record> = serde_json::from_value(m["records"].clone())?;
    let mut actual = BTreeMap::new();
    for r in records {
        validate_record(&r)?;
        if actual.insert(r.source_id, r).is_some() {
            return Err("duplicate actual source ID".into());
        }
    }
    let text = std::fs::read_to_string(native)?;
    let mut rows = text.lines();
    if rows.next() != Some("source_id,length_m_bits,part_lengths_m_bits,part_point_counts") {
        return Err("expected native geodesic CSV header".into());
    }
    let mut expected = BTreeMap::new();
    for line in rows {
        let cols = line.split(',').collect::<Vec<_>>();
        if cols.len() != 4 {
            return Err("invalid native geodesic CSV row".into());
        }
        let n = id(cols[0])?;
        let lens = cols[2]
            .split(';')
            .map(str::parse::<u64>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let counts = cols[3]
            .split(';')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let r = Record {
            source_id: n,
            physical_record: 0,
            point_count: counts.iter().sum(),
            part_point_counts: counts,
            part_lengths_m_bits: lens,
            length_m_bits: cols[1].parse()?,
        };
        validate_record(&r)?;
        if expected.insert(n, r).is_some() {
            return Err("duplicate native source ID".into());
        }
    }
    let pack = |map: &BTreeMap<i64, Record>| {
        let mut b = Vec::new();
        for (n, r) in map {
            b.extend(n.to_le_bytes());
            b.extend(r.length_m_bits.to_le_bytes());
            b.extend((r.part_point_counts.len() as u64).to_le_bytes());
            for (p, l) in r.part_point_counts.iter().zip(&r.part_lengths_m_bits) {
                b.extend((*p as u64).to_le_bytes());
                b.extend(l.to_le_bytes());
            }
        }
        b
    };
    let a = pack(&actual);
    let b = pack(&expected);
    let different = a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
    let ids =
        actual.keys().copied().collect::<Vec<_>>() == expected.keys().copied().collect::<Vec<_>>();
    Ok(
        json!({"pass":ids && !actual.is_empty() && different==0,"actual_records":actual.len(),"native_records":expected.len(),
        "source_ids_match":ids,"canonical_bytes":a.len(),"different_canonical_bytes":different,
        "actual_sha256":format!("{:x}",Sha256::digest(&a)),"native_sha256":format!("{:x}",Sha256::digest(&b)),
        "scope":"strict source ID/part/vertex roster and WGS84 length words; independent GDAL reader/transform/length calls share the pinned PROJ geodesic math library, not an independent math algorithm",
        "wall_ms":t.elapsed().as_secs_f64()*1000.}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ellipsoidal_and_disconnected_parts() {
        let g = Geodesic::wgs84();
        let a = g.length(&[[0., 0.], [1., 0.]]).unwrap();
        assert_eq!(a.to_bits(), 111319.49079327357f64.to_bits());
        assert_eq!(
            g.length(&[[1., 0.], [0., 0.]]).unwrap().to_bits(),
            a.to_bits()
        );
        assert_eq!(g.length(&[[0., 0.], [0., 0.]]).unwrap(), 0.);
        assert!(g.length(&[[179., 0.], [-179., 0.]]).unwrap() < 3. * a);
        assert!(g.length(&[[0., 0.], [179.999, 0.001]]).unwrap().is_finite());
        let r = Record {
            source_id: 1,
            physical_record: 0,
            point_count: 4,
            part_point_counts: vec![2, 2],
            part_lengths_m_bits: vec![a.to_bits(), a.to_bits()],
            length_m_bits: (2. * a).to_bits(),
        };
        validate_record(&r).unwrap();
    }
    #[test]
    fn invalid_geometry_and_totals() {
        let g = Geodesic::wgs84();
        for p in [
            vec![],
            vec![[0., 0.]],
            vec![[0., 0.], [181., 0.]],
            vec![[0., 0.], [0., 91.]],
            vec![[0., 0.], [f64::NAN, 0.]],
        ] {
            assert!(g.length(&p).is_err());
        }
        let mut r = Record {
            source_id: 1,
            physical_record: 0,
            point_count: 2,
            part_point_counts: vec![2],
            part_lengths_m_bits: vec![0],
            length_m_bits: 0,
        };
        assert!(validate_record(&r).is_err());
        r.part_lengths_m_bits[0] = 1f64.to_bits();
        r.length_m_bits = 2f64.to_bits();
        assert!(validate_record(&r).is_err());
    }
}
