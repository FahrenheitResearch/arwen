//! Original-source polygon/flowline subsets. SHP/SHX follows the published ESRI
//! Shapefile Technical Description (1998); no ESRI/GDAL program source used.
//! Copy source coordinates/attributes verbatim; never invent a projected grid.
use crate::{bundle::digest, dbf::Table, source_network::id, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
pub(crate) struct Source {
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
}
#[derive(Deserialize)]
struct Config {
    shp: Source,
    shx: Source,
    dbf: Source,
    prj: Source,
    network: Source,
    /// Select polygon IDs from this source in a sealed waterbody crosswalk.
    /// Absent keeps the original flowpath-network selection contract.
    waterbody_source_index: Option<usize>,
    id_column: String,
    #[serde(default = "default_stem")]
    stem: String,
    output: PathBuf,
    data_license: String,
    provenance: Value,
}
fn default_stem() -> String {
    "Catchment".into()
}
pub(crate) fn requested_ids(
    roster: &Value,
    source_index: Option<usize>,
    dbf_sha: &str,
    id_column: &str,
    shape_type: usize,
) -> Result<Vec<i64>> {
    let Some(index) = source_index else {
        if roster["schema"] != "arwen.source-network-subset.v1" {
            return Err("expected original-source network subset schema".into());
        }
        return Ok(serde_json::from_value(roster["node_ids"].clone())?);
    };
    if roster["schema"] != "arwen.source-waterbody-crosswalk.v1" || ![5, 15].contains(&shape_type) {
        return Err("waterbody selection requires a source crosswalk and Polygon geometry".into());
    }
    let extras = match roster["sources"].get("additional_waterbodies") {
        None => &[][..],
        Some(v) => v
            .as_array()
            .ok_or("invalid crosswalk source list")?
            .as_slice(),
    };
    let source = if index == 0 {
        &roster["sources"]["waterbodies"]
    } else {
        extras
            .get(index - 1)
            .ok_or("waterbody source index absent from crosswalk")?
    };
    if source["sha256"].as_str() != Some(dbf_sha)
        || roster["mapping"]["waterbody_id_column"].as_str() != Some(id_column)
    {
        return Err("geometry DBF digest/ID column differs from crosswalk source".into());
    }
    let rows = roster["waterbodies"]
        .as_array()
        .ok_or("crosswalk waterbody rows absent")?;
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for row in rows {
        let id = row["waterbody_id"]
            .as_i64()
            .filter(|n| *n > 0)
            .ok_or("crosswalk waterbody ID must be a positive integer")?;
        if !seen.insert(id) {
            return Err("duplicate crosswalk waterbody ID".into());
        }
        let row_index = match row.get("source_index") {
            None if extras.is_empty() => 0,
            Some(v) => usize::try_from(v.as_u64().ok_or("invalid waterbody row source index")?)?,
            _ => return Err("multi-source crosswalk row lacks source index".into()),
        };
        if row_index > extras.len() {
            return Err("waterbody row source index absent from crosswalk".into());
        }
        if row_index == index {
            ids.push(id);
        }
    }
    Ok(ids)
}
pub(crate) fn bytes(s: &Source) -> Result<Vec<u8>> {
    let b = std::fs::read(&s.path)?;
    if format!("{:x}", Sha256::digest(&b)) != s.sha256 {
        return Err(format!("source digest mismatch: {}", s.path.display()).into());
    }
    Ok(b)
}
fn u32be(b: &[u8], p: usize) -> Result<usize> {
    Ok(u32::from_be_bytes(b.get(p..p + 4).ok_or("short SHP integer")?.try_into()?) as usize)
}
pub(crate) fn u32le(b: &[u8], p: usize) -> Result<usize> {
    Ok(u32::from_le_bytes(b.get(p..p + 4).ok_or("short SHP integer")?.try_into()?) as usize)
}
fn f64le(b: &[u8], p: usize) -> Result<f64> {
    let v = f64::from_le_bytes(b.get(p..p + 8).ok_or("short SHP coordinate")?.try_into()?);
    if !v.is_finite() {
        return Err("nonfinite SHP coordinate".into());
    }
    Ok(v)
}
fn header(b: &[u8]) -> Result<()> {
    if b.len() < 100
        || u32be(b, 0)? != 9994
        || b[4..24].iter().any(|&v| v != 0)
        || u32be(b, 24)?.checked_mul(2) != Some(b.len())
        || u32le(b, 28)? != 1000
        || ![3, 5, 13, 15].contains(&u32le(b, 32)?)
    {
        return Err(
            "expected length-consistent Polygon/PolygonZ/PolyLine/PolyLineZ SHP/SHX header".into(),
        );
    }
    Ok(())
}
pub(crate) struct Shape {
    pub(crate) shp: Vec<u8>,
    pub(crate) records: Vec<(usize, usize)>,
}
impl Shape {
    pub(crate) fn open(shp: Vec<u8>, shx: Vec<u8>) -> Result<Self> {
        header(&shp)?;
        header(&shx)?;
        if (shx.len() - 100) % 8 != 0 || shp[28..100] != shx[28..100] {
            return Err("inconsistent SHP/SHX headers".into());
        }
        let n = (shx.len() - 100) / 8;
        let records = (0..n)
            .into_par_iter()
            .map(|i| {
                let offset = u32be(&shx, 100 + 8 * i)?
                    .checked_mul(2)
                    .ok_or("SHX offset overflow")?;
                let length = u32be(&shx, 104 + 8 * i)?
                    .checked_mul(2)
                    .ok_or("SHX length overflow")?;
                let end = offset
                    .checked_add(8)
                    .and_then(|v| v.checked_add(length))
                    .ok_or("SHP record overflow")?;
                if length < 4
                    || end > shp.len()
                    || u32be(&shp, offset)? != i + 1
                    || u32be(&shp, offset + 4)?.checked_mul(2) != Some(length)
                {
                    return Err("SHX index and SHP physical record disagree".into());
                }
                Ok((offset, length))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut end = 100;
        for &(offset, length) in &records {
            if offset != end {
                return Err("SHP indexed records have a gap/overlap".into());
            }
            end = offset + 8 + length;
        }
        if end != shp.len() {
            return Err("SHP unindexed trailing bytes".into());
        }
        Ok(Self { shp, records })
    }
    pub(crate) fn record(&self, i: usize) -> Result<&[u8]> {
        let &(p, n) = self.records.get(i).ok_or("SHP record absent")?;
        Ok(&self.shp[p + 8..p + 8 + n])
    }
}
type Ring = Vec<[u64; 2]>;
// axes follows ISO WKB dimensional modifiers: 0=XY, 1000=XYZ, 3000=XYZM.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Part {
    pub(crate) axes: u32,
    pub(crate) points: Vec<[u64; 4]>,
}
pub(crate) fn parts(b: &[u8], shape_type: usize) -> Result<Vec<Part>> {
    if u32le(b, 0)? == 0 {
        return Err("selected source geometry is NULL".into());
    }
    if u32le(b, 0)? != shape_type || b.len() < 44 {
        return Err("SHP record type differs from header".into());
    }
    let nr = u32le(b, 36)?;
    let np = u32le(b, 40)?;
    let points = 44usize
        .checked_add(nr.checked_mul(4).ok_or("SHP parts overflow")?)
        .ok_or("SHP parts overflow")?;
    let xy_end = points
        .checked_add(np.checked_mul(16).ok_or("SHP points overflow")?)
        .ok_or("SHP points overflow")?;
    let extra = 16usize
        .checked_add(np.checked_mul(8).ok_or("SHP ordinates overflow")?)
        .ok_or("SHP ordinates overflow")?;
    let (axes, z_start, m_start) = if [13, 15].contains(&shape_type) {
        let z_end = xy_end.checked_add(extra).ok_or("SHP Z overflow")?;
        if b.len() == z_end {
            (1000, Some(xy_end + 16), None)
        } else if Some(b.len()) == z_end.checked_add(extra) {
            (3000, Some(xy_end + 16), Some(z_end + 16))
        } else {
            return Err("inconsistent SHP Z/optional-M arrays".into());
        }
    } else {
        if b.len() != xy_end {
            return Err("inconsistent SHP XY length".into());
        }
        (0, None, None)
    };
    if nr == 0 || np == 0 {
        return Err("inconsistent SHP parts/points".into());
    }
    let mut out = Vec::new();
    let mut last = 0;
    for i in 0..nr {
        let first = u32le(b, 44 + 4 * i)?;
        let end = if i + 1 == nr {
            np
        } else {
            u32le(b, 48 + 4 * i)?
        };
        let min_points = if [5, 15].contains(&shape_type) { 4 } else { 2 };
        if first != last || end > np || end < first + min_points {
            return Err("invalid SHP part extent".into());
        }
        let mut r = Vec::with_capacity(end - first);
        for p in first..end {
            let mut v = [
                f64le(b, points + 16 * p)?.to_bits(),
                f64le(b, points + 16 * p + 8)?.to_bits(),
                0,
                0,
            ];
            if let Some(z) = z_start {
                v[2] = f64le(b, z + 8 * p)?.to_bits();
            }
            if let Some(m) = m_start {
                v[3] = f64le(b, m + 8 * p)?.to_bits();
            }
            r.push(v);
        }
        if [5, 15].contains(&shape_type) && r.first() != r.last() {
            return Err("unclosed SHP polygon ring".into());
        }
        out.push(Part { axes, points: r });
        last = end;
    }
    Ok(out)
}
fn bbox(parts: &[Part]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in parts.iter().flat_map(|p| &p.points) {
        let x = f64::from_bits(p[0]);
        let y = f64::from_bits(p[1]);
        b[0] = b[0].min(x);
        b[1] = b[1].min(y);
        b[2] = b[2].max(x);
        b[3] = b[3].max(y);
    }
    b
}
fn write_new(path: &Path, b: &[u8]) -> Result<()> {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(b)?;
    Ok(())
}
fn header_for(original: &[u8], length: usize, bounds: [f64; 4]) -> Result<Vec<u8>> {
    let mut h = original[..100].to_vec();
    h[24..28].copy_from_slice(&u32::try_from(length / 2)?.to_be_bytes());
    for (i, v) in bounds.iter().enumerate() {
        h[36 + 8 * i..44 + 8 * i].copy_from_slice(&v.to_le_bytes());
    }
    Ok(h)
}
pub fn prepare(path: &Path) -> Result<Value> {
    let t = Instant::now();
    let c: Config = serde_json::from_slice(&std::fs::read(path)?)?;
    if c.data_license.trim().is_empty() {
        return Err("actual source data_license required".into());
    }
    if c.stem.is_empty()
        || !c
            .stem
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err("output stem must contain only ASCII letters/digits/underscore".into());
    }
    let sources = [&c.shp, &c.shx, &c.prj, &c.network];
    let mut loaded = sources
        .par_iter()
        .map(|s| bytes(s))
        .collect::<Result<Vec<_>>>()?
        .into_iter();
    let shape = Shape::open(loaded.next().unwrap(), loaded.next().unwrap())?;
    let shape_type = u32le(&shape.shp, 32)?;
    let prj = loaded.next().unwrap();
    std::str::from_utf8(&prj)?;
    let net: Value = serde_json::from_slice(&loaded.next().unwrap())?;
    let requested = requested_ids(
        &net,
        c.waterbody_source_index,
        &c.dbf.sha256,
        &c.id_column,
        shape_type,
    )?;
    let selected = requested.iter().copied().collect::<BTreeSet<_>>();
    if selected.is_empty() || selected.len() != requested.len() {
        return Err("empty/duplicate requested node IDs".into());
    }
    let table = Table::open(&c.dbf.path)?;
    if table.sha256 != c.dbf.sha256 || table.count != shape.records.len() {
        return Err("DBF digest or SHP physical record count mismatch".into());
    }
    let rows = table.indexed_rows(&[&c.id_column])?;
    let selected_rows = rows
        .par_iter()
        .enumerate()
        .map(|(i, row)| -> Result<Option<(i64, usize)>> {
            let Some(row) = row else { return Ok(None) };
            let n = id(&row[0])?;
            Ok(selected.contains(&n).then_some((n, i)))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let ids = selected_rows
        .iter()
        .map(|&(n, _)| n)
        .collect::<BTreeSet<_>>();
    // Preserve multiple source records per ID rather than silently overwriting.
    let missing = selected.difference(&ids).copied().collect::<Vec<_>>();
    let read_ms = t.elapsed().as_secs_f64() * 1000.;
    let decode = Instant::now();
    let mut pieces = selected_rows
        .par_iter()
        .map(|&(n, i)| {
            let raw = shape.record(i)?;
            let r = parts(raw, shape_type)?;
            Ok((
                n,
                i,
                bbox(&r),
                r.iter().map(|p| p.points.len()).sum::<usize>(),
                r.len(),
                raw,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    pieces.par_sort_unstable_by_key(|&(n, i, _, _, _, _)| (n, i));
    if pieces.is_empty() {
        return Err("no source geometry matches requested network".into());
    }
    let bounds = pieces.iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |a, p| {
            [
                a[0].min(p.2[0]),
                a[1].min(p.2[1]),
                a[2].max(p.2[2]),
                a[3].max(p.2[3]),
            ]
        },
    );
    let shp_len = 100 + pieces.iter().map(|p| 8 + p.5.len()).sum::<usize>();
    let mut shp = header_for(&shape.shp, shp_len, bounds)?;
    let mut shx = header_for(&shape.shp, 100 + 8 * pieces.len(), bounds)?;
    // Recompute selected Z/M extrema too. Raw record arrays remain verbatim;
    // finite ESRI M no-data sentinels are retained as source values.
    if [13, 15].contains(&shape_type) {
        let ranges = pieces
            .par_iter()
            .map(|p| -> Result<[f64; 4]> {
                let mut r = [
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                ];
                for part in parts(p.5, shape_type)? {
                    for v in part.points {
                        let z = f64::from_bits(v[2]);
                        r[0] = r[0].min(z);
                        r[1] = r[1].max(z);
                        if part.axes == 3000 {
                            let m = f64::from_bits(v[3]);
                            r[2] = r[2].min(m);
                            r[3] = r[3].max(m);
                        }
                    }
                }
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(
                [
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                ],
                |a, b| {
                    [
                        a[0].min(b[0]),
                        a[1].max(b[1]),
                        a[2].min(b[2]),
                        a[3].max(b[3]),
                    ]
                },
            );
        for (i, value) in ranges.into_iter().enumerate() {
            let value = if value.is_finite() { value } else { 0. };
            shp[68 + 8 * i..76 + 8 * i].copy_from_slice(&value.to_le_bytes());
            shx[68 + 8 * i..76 + 8 * i].copy_from_slice(&value.to_le_bytes());
        }
    }
    for (j, p) in pieces.iter().enumerate() {
        shx.extend_from_slice(&u32::try_from(shp.len() / 2)?.to_be_bytes());
        shx.extend_from_slice(&u32::try_from(p.5.len() / 2)?.to_be_bytes());
        shp.extend_from_slice(&u32::try_from(j + 1)?.to_be_bytes());
        shp.extend_from_slice(&u32::try_from(p.5.len() / 2)?.to_be_bytes());
        shp.extend_from_slice(p.5);
    }
    let dbf = table.subset_bytes(&pieces.iter().map(|p| p.1).collect::<Vec<_>>())?;
    let geometry_ms = decode.elapsed().as_secs_f64() * 1000.;
    std::fs::create_dir(&c.output)?;
    let files = [
        (format!("{}.shp", c.stem), &shp),
        (format!("{}.shx", c.stem), &shx),
        (format!("{}.dbf", c.stem), &dbf),
        (format!("{}.prj", c.stem), &prj),
    ];
    let mut payloads = BTreeMap::new();
    for (name, b) in files {
        write_new(&c.output.join(&name), b)?;
        payloads.insert(
            name,
            json!({"sha256":format!("{:x}",Sha256::digest(b)),"bytes":b.len()}),
        );
    }
    let mut manifest = json!({"schema":if [5,15].contains(&shape_type) {"arwen.source-polygon-subset.v1"} else {"arwen.source-polyline-subset.v1"},
        "shape_type":shape_type,"stem":c.stem,"source":crate::SOURCE_REVISION,"network_sha256":c.network.sha256,
        "id_column":c.id_column,"requested_node_count":selected.len(),"matched_node_ids":ids,"missing_node_ids":missing,
        "record_count":pieces.len(),"point_count":pieces.iter().map(|p|p.3).sum::<usize>(),"part_count":pieces.iter().map(|p|p.4).sum::<usize>(),
        "ring_count":if [5,15].contains(&shape_type) {Some(pieces.iter().map(|p|p.4).sum::<usize>())} else {None},
        "bbox_source_crs":bounds,"crs_wkt":std::str::from_utf8(&prj)?,"payloads":payloads,
        "record_join":"SHP/SHX physical record number to DBF physical record; deleted DBF rows keep their positions",
        "coordinates":"verbatim IEEE754 source ordinates, part order and direction; no transform, simplification or area/grid assumption; Z/M presence follows each source record",
        "data_license":c.data_license,"provenance":c.provenance,
        "sources":{"shp":{"sha256":c.shp.sha256},"shx":{"sha256":c.shx.sha256},"dbf":{"sha256":c.dbf.sha256},"prj":{"sha256":c.prj.sha256}},
        "timing_ms":{"read_hash_index_join":read_ms,"selected_geometry_pack":geometry_ms,"total":t.elapsed().as_secs_f64()*1000.},
        "rayon_workers":rayon::current_num_threads(),"parallelism":"independent source files; physical record validation; DBF decode/join; selected geometry and sorting on Rayon; startup only"});
    if let Some(index) = c.waterbody_source_index {
        manifest["selection"] = json!({"schema":net["schema"],"source_index":index,
            "id_scope":"matched/missing node IDs above are source polygon IDs, not flowpath IDs",
            "scope":"verbatim referenced polygons only; no lake classification, outlet, storage or hydraulic inference"});
    }
    write_new(
        &c.output.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(json!({"manifest":c.output.join("manifest.json"),"result":manifest}))
}
// Normalize ring start/orientation/group order only. Preserve coordinate bits,
// edges and multiplicities. Containment and projected areas are outside scope.
pub(crate) fn canonical_ring(mut r: Ring) -> Result<Ring> {
    if r.len() < 4 || r.first() != r.last() {
        return Err("invalid referee polygon ring".into());
    }
    r.pop();
    let min = *r.iter().min().ok_or("empty ring")?;
    let n = r.len();
    let mut best: Option<Ring> = None;
    for i in 0..n {
        if r[i] != min {
            continue;
        }
        for reverse in [false, true] {
            let v = (0..n)
                .map(|j| {
                    r[if reverse {
                        (i + n - j) % n
                    } else {
                        (i + j) % n
                    }]
                })
                .collect::<Ring>();
            if best.as_ref().is_none_or(|b| v < *b) {
                best = Some(v);
            }
        }
    }
    Ok(best.unwrap())
}
pub(crate) fn canonical(map: BTreeMap<i64, Vec<Ring>>) -> Result<Vec<u8>> {
    let mut records = map
        .into_par_iter()
        .map(|(id, rings)| -> Result<(i64, Vec<u8>)> {
            let mut out = Vec::new();
            let mut rs = rings
                .into_iter()
                .map(canonical_ring)
                .collect::<Result<Vec<_>>>()?;
            rs.sort();
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(rs.len() as u64).to_le_bytes());
            for r in rs {
                out.extend_from_slice(&(r.len() as u64).to_le_bytes());
                for p in r {
                    for b in p {
                        out.extend_from_slice(&b.to_le_bytes());
                    }
                }
            }
            Ok((id, out))
        })
        .collect::<Result<Vec<_>>>()?;
    records.par_sort_unstable_by_key(|p| p.0);
    Ok(records.into_iter().flat_map(|p| p.1).collect())
}
pub(crate) fn unhex(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err("odd native WKB hex length".into());
    }
    s.as_bytes()
        .chunks_exact(2)
        .map(|c| Ok(u8::from_str_radix(std::str::from_utf8(c)?, 16)?))
        .collect()
}
pub(crate) fn native_lines(b: &[u8]) -> Result<Vec<Part>> {
    struct Wkb<'a> {
        b: &'a [u8],
        p: usize,
    }
    impl Wkb<'_> {
        fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
            let out = self
                .b
                .get(self.p..self.p + N)
                .ok_or("short native line WKB")?
                .try_into()?;
            self.p += N;
            Ok(out)
        }
        fn u32(&mut self, le: bool) -> Result<u32> {
            let v = self.take()?;
            Ok(if le {
                u32::from_le_bytes(v)
            } else {
                u32::from_be_bytes(v)
            })
        }
        fn read(&mut self, out: &mut Vec<Part>, depth: usize) -> Result<()> {
            if depth > 32 {
                return Err("native line WKB nesting exceeds32".into());
            }
            let le = match self.take::<1>()?[0] {
                0 => false,
                1 => true,
                _ => return Err("invalid native line WKB byte order".into()),
            };
            let code = self.u32(le)?;
            let axes = code / 1000 * 1000;
            let dims = match axes {
                0 => 2,
                1000 | 2000 => 3,
                3000 => 4,
                _ => return Err("expected ISO XY/Z/M/ZM WKB".into()),
            };
            match code % 1000 {
                2 => {
                    let n = self.u32(le)? as usize;
                    if n < 2 || n > (self.b.len() - self.p) / (dims * 8) {
                        return Err("invalid native line point count".into());
                    }
                    let mut points = Vec::with_capacity(n);
                    for _ in 0..n {
                        let mut point = [0; 4];
                        let ordinals = match axes {
                            0 => &[0, 1][..],
                            1000 => &[0, 1, 2][..],
                            2000 => &[0, 1, 3][..],
                            _ => &[0, 1, 2, 3][..],
                        };
                        for &o in ordinals {
                            let raw = self.take()?;
                            let bits = if le {
                                u64::from_le_bytes(raw)
                            } else {
                                u64::from_be_bytes(raw)
                            };
                            if !f64::from_bits(bits).is_finite() {
                                return Err("nonfinite native line ordinate".into());
                            }
                            point[o] = bits;
                        }
                        points.push(point);
                    }
                    out.push(Part { axes, points });
                }
                5 | 7 => {
                    let n = self.u32(le)? as usize;
                    if n > (self.b.len() - self.p) / 5 {
                        return Err("invalid native WKB collection count".into());
                    }
                    for _ in 0..n {
                        self.read(out, depth + 1)?;
                    }
                }
                _ => return Err("expected LineString/MultiLineString native WKB".into()),
            }
            Ok(())
        }
    }
    let mut w = Wkb { b, p: 0 };
    let mut out = Vec::new();
    w.read(&mut out, 0)?;
    if w.p != b.len() {
        return Err("trailing native line WKB bytes".into());
    }
    Ok(out)
}
pub(crate) fn canonical_lines(map: BTreeMap<i64, Vec<Part>>) -> Result<Vec<u8>> {
    let mut records = map
        .into_par_iter()
        .map(|(id, mut parts)| {
            // Ignore collection/part grouping order only. Preserve every directed
            // coordinate sequence, dimensional modifier and point multiplicity.
            parts.sort();
            let mut out = Vec::new();
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(parts.len() as u64).to_le_bytes());
            for part in parts {
                out.extend_from_slice(&part.axes.to_le_bytes());
                out.extend_from_slice(&(part.points.len() as u64).to_le_bytes());
                let ordinals = match part.axes {
                    0 => &[0, 1][..],
                    1000 => &[0, 1, 2][..],
                    2000 => &[0, 1, 3][..],
                    _ => &[0, 1, 2, 3][..],
                };
                for point in part.points {
                    for &o in ordinals {
                        out.extend_from_slice(&point[o].to_le_bytes());
                    }
                }
            }
            (id, out)
        })
        .collect::<Vec<_>>();
    records.par_sort_unstable_by_key(|p| p.0);
    Ok(records.into_iter().flat_map(|p| p.1).collect())
}
pub fn replay(manifest: &Path, native: &Path) -> Result<Value> {
    let t = Instant::now();
    let m: Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let dir = manifest.parent().ok_or("manifest directory absent")?;
    let polygon = match m["schema"].as_str() {
        Some("arwen.source-polygon-subset.v1") => true,
        Some("arwen.source-polyline-subset.v1") => false,
        _ => return Err("expected source polygon/polyline subset schema".into()),
    };
    let stem = m["stem"].as_str().unwrap_or("Catchment");
    if stem.is_empty() || !stem.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        return Err("invalid prepared geometry stem".into());
    }
    for (name, p) in m["payloads"].as_object().ok_or("payloads absent")? {
        if !["shp", "shx", "dbf", "prj"]
            .iter()
            .any(|ext| name == &format!("{stem}.{ext}"))
        {
            return Err("unexpected geometry payload name".into());
        }
        if digest(&dir.join(name))? != p["sha256"].as_str().ok_or("payload SHA absent")? {
            return Err("prepared payload digest mismatch".into());
        }
    }
    for ext in ["shp", "shx", "dbf", "prj"] {
        if !m["payloads"][format!("{stem}.{ext}")]["sha256"].is_string() {
            return Err("required geometry payload seal absent".into());
        }
    }
    let shape = Shape::open(
        std::fs::read(dir.join(format!("{stem}.shp")))?,
        std::fs::read(dir.join(format!("{stem}.shx")))?,
    )?;
    let shape_type = u32le(&shape.shp, 32)?;
    if ([5, 15].contains(&shape_type)) != polygon {
        return Err("geometry schema/type mismatch".into());
    }
    let table = Table::open(&dir.join(format!("{stem}.dbf")))?;
    if table.count != shape.records.len() {
        return Err("subset DBF/SHP record count mismatch".into());
    }
    let rows = table.indexed_rows(&[m["id_column"].as_str().ok_or("id column absent")?])?;
    let mut actual = BTreeMap::<i64, Vec<Part>>::new();
    for (i, row) in rows.iter().enumerate() {
        if let Some(row) = row {
            actual
                .entry(id(&row[0])?)
                .or_default()
                .extend(parts(shape.record(i)?, shape_type)?);
        }
    }
    let text = std::fs::read_to_string(native)?;
    let mut lines = text.lines();
    if lines.next() != Some("source_id,wkb") {
        return Err("expected GDAL source_id,wkb header".into());
    }
    let parsed = lines
        .collect::<Vec<_>>()
        .par_iter()
        .map(|line| -> Result<(i64, Vec<Part>)> {
            let (a, b) = line.split_once(',').ok_or("invalid native WKB row")?;
            let raw = unhex(b.trim_matches('"'))?;
            if !polygon {
                return Ok((id(a.trim_matches('"'))?, native_lines(&raw)?));
            }
            let mut gp = b"GP\0\x01\0\0\0\0".to_vec();
            gp.extend(raw);
            let rs = crate::weights::polygons(&gp)?
                .into_iter()
                .flatten()
                .map(|ring| Part {
                    axes: 0,
                    points: ring
                        .into_iter()
                        .map(|p| [p[0].to_bits(), p[1].to_bits(), 0, 0])
                        .collect(),
                })
                .collect();
            Ok((id(a.trim_matches('"'))?, rs))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut expected = BTreeMap::<i64, Vec<Part>>::new();
    for (n, r) in parsed {
        expected.entry(n).or_default().extend(r);
    }
    let actual_ids = actual.keys().copied().collect::<BTreeSet<_>>();
    let expected_ids = expected.keys().copied().collect::<BTreeSet<_>>();
    let pack = |map: BTreeMap<i64, Vec<Part>>| {
        if polygon {
            canonical(
                map.into_iter()
                    .map(|(id, parts)| {
                        (
                            id,
                            parts
                                .into_iter()
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
        json!({"strict":true,"pass":different==0,"different_canonical_bytes":different,"compared_canonical_bytes":b.len(),
        "actual_sha256":format!("{:x}",Sha256::digest(&a)),"expected_sha256":format!("{:x}",Sha256::digest(&b)),
        "id_symmetric_difference":actual_ids.symmetric_difference(&expected_ids).count(),"node_count":actual_ids.len(),
        "scope":if polygon {"all selected polygon XY bits and closed-ring edge sequences; rotation/orientation/grouping normalized, no containment/area/weights/physics claim"}
            else {"all selected line XYZ/M bits, dimensional modifiers and directed point sequences; collection grouping/order normalized only; no Z elevation/datum, M chainage, hydraulic parameter or parent-grid claim"},
        "source":crate::SOURCE_REVISION,"manifest":manifest,"native_capture":native,"native_capture_sha256":digest(native)?,
        "wall_ms":t.elapsed().as_secs_f64()*1000.,"rayon_workers":rayon::current_num_threads(),
        "parallelism":"native WKB rows and geometry canonicalization on Rayon; independent candidate/referee canonical streams"}),
    )
}

// Domain selection uses XY straight segments in the pinned source CRS. The
// domain is a simple closed ring, with boundary contact included. No bbox-only
// admission or geodesic-edge approximation is hidden. Caller-CRS requests
// explicitly sample their edges before native PROJ transformation.
type Point = [f64; 2];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionDomain {
    crs_sha256: Option<String>,
    crs: Option<String>,
    edge_segments: Option<usize>,
    ring: Vec<Point>,
}
impl SelectionDomain {
    fn resolve(&self, prj_hash: &str, prj: &str) -> Result<(Vec<Point>, Option<Value>)> {
        match (&self.crs_sha256, &self.crs, self.edge_segments) {
            (Some(hash), None, None) if hash == prj_hash => {
                domain_ring(&self.ring)?;
                Ok((self.ring.clone(), None))
            }
            (None, Some(crs), Some(segments)) => {
                // Bound the quadratic simple-ring validation and preparation
                // allocation, not the source dataset or selected reach count.
                let edges = self.ring.len().saturating_sub(1);
                if !(1..=256).contains(&segments)
                    || edges == 0
                    || edges.checked_mul(segments).is_none_or(|n| n > 4096)
                {
                    return Err("projected domain needs edge_segments 1..256 and at most 4096 sampled edges (bounds native ring validation/allocation)".into());
                }
                domain_ring(&self.ring)?;
                if crs.trim().is_empty() {
                    return Err("domain input CRS is empty".into());
                }
                let projection = crate::weights::Projection::new(crs, prj)?;
                let mut ring = Vec::with_capacity(edges * segments + 1);
                for e in self.ring.windows(2) {
                    for j in 0..segments {
                        let point = if j == 0 {
                            e[0]
                        } else {
                            let f = j as f64 / segments as f64;
                            [e[0][0] + f * (e[1][0] - e[0][0]),
                             e[0][1] + f * (e[1][1] - e[0][1])]
                        };
                        ring.push(projection.transform(point)?);
                    }
                }
                ring.push(ring[0]);
                domain_ring(&ring)?;
                let words = |points: &[Point]| points.iter().map(|p| p.map(f64::to_bits)).collect::<Vec<_>>();
                let metadata = json!({"input_crs":crs,"input_ring_f64_bits":words(&self.ring),
                    "edge_segments":segments,"source_ring_f64_bits":words(&ring),
                    "method":"uniform straight-edge sampling in input XY, then PROJ visualization axis order with network disabled; sampled source-CRS segments define the selection boundary; no geodesic edges, longitude wrapping or error-bound guarantee"});
                Ok((ring, Some(metadata)))
            }
            _ => Err("domain needs either matching source crs_sha256 only, or input crs plus explicit edge_segments; do not mix CRS forms".into()),
        }
    }
}
fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}
fn on_segment(a: Point, b: Point, p: Point) -> bool {
    cross(a, b, p) == 0. && (0..2).all(|k| p[k] >= a[k].min(b[k]) && p[k] <= a[k].max(b[k]))
}
fn segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
    let sides = [
        cross(a, b, c),
        cross(a, b, d),
        cross(c, d, a),
        cross(c, d, b),
    ];
    ((sides[0] > 0. && sides[1] < 0. || sides[0] < 0. && sides[1] > 0.)
        && (sides[2] > 0. && sides[3] < 0. || sides[2] < 0. && sides[3] > 0.))
        || on_segment(a, b, c)
        || on_segment(a, b, d)
        || on_segment(c, d, a)
        || on_segment(c, d, b)
}
fn domain_ring(ring: &[Point]) -> Result<()> {
    if ring.len() < 4
        || ring.first() != ring.last()
        || ring.iter().flatten().any(|v| !v.is_finite())
    {
        return Err("domain requires a finite closed simple source-CRS ring".into());
    }
    let edges = ring.windows(2).collect::<Vec<_>>();
    let origin = ring[0];
    let area = edges.iter().map(|e| cross(origin, e[0], e[1])).sum::<f64>();
    if !area.is_finite() || area == 0. || edges.iter().any(|e| e[0] == e[1]) {
        return Err(
            "domain area arithmetic overflowed, or has zero area or a zero-length edge".into(),
        );
    }
    for i in 0..edges.len() {
        for j in i + 1..edges.len() {
            let a = edges[i];
            let b = edges[j];
            if j == i + 1 || i == 0 && j == edges.len() - 1 {
                // Adjacent edges may share their endpoint, not fold back.
                let (other, common, last) = if j == i + 1 {
                    (a[0], a[1], b[1])
                } else {
                    (a[1], a[0], b[0])
                };
                if on_segment(common, other, last) || on_segment(common, last, other) {
                    return Err("domain has overlapping adjacent edges".into());
                }
            } else if segments_intersect(a[0], a[1], b[0], b[1]) {
                return Err("domain ring self-intersects".into());
            }
        }
    }
    Ok(())
}
fn contains(ring: &[Point], point: Point) -> bool {
    let mut inside = false;
    for e in ring.windows(2) {
        if on_segment(e[0], e[1], point) {
            return true;
        }
        let [a, b] = [e[0], e[1]];
        if (a[1] > point[1]) != (b[1] > point[1])
            && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
    }
    inside
}
fn intersects_domain(parts: &[Part], ring: &[Point]) -> bool {
    parts.iter().any(|p| {
        let xy = p
            .points
            .iter()
            .map(|v| [f64::from_bits(v[0]), f64::from_bits(v[1])])
            .collect::<Vec<_>>();
        xy.iter().any(|&v| contains(ring, v))
            || xy.windows(2).any(|s| {
                ring.windows(2)
                    .any(|e| segments_intersect(s[0], s[1], e[0], e[1]))
            })
    })
}
/// Sealed spatial seeds for source_network's upstream-closure selection.
pub fn select_domain(path: &Path) -> Result<Value> {
    #[derive(Deserialize)]
    struct Request {
        shp: Source,
        shx: Source,
        dbf: Source,
        prj: Source,
        id_column: String,
        domain: SelectionDomain,
        output: PathBuf,
        data_license: String,
        provenance: Value,
    }
    let start = Instant::now();
    let c: Request = serde_json::from_slice(&std::fs::read(path)?)?;
    if c.data_license.trim().is_empty() {
        return Err("data_license must be named".into());
    }
    let sources = [&c.shp, &c.shx, &c.prj];
    let mut loaded = sources
        .par_iter()
        .map(|s| bytes(s))
        .collect::<Result<Vec<_>>>()?
        .into_iter();
    let shp = loaded.next().ok_or("missing SHP source")?;
    let shx = loaded.next().ok_or("missing SHX source")?;
    let prj = String::from_utf8(loaded.next().ok_or("missing PRJ source")?)?;
    if prj.trim().is_empty() {
        return Err("source PRJ is empty".into());
    }
    let projection_start = Instant::now();
    let (ring, projection_metadata) = c.domain.resolve(&c.prj.sha256, &prj)?;
    let projection_ms = projection_start.elapsed().as_secs_f64() * 1000.;
    let shape = Shape::open(shp, shx)?;
    let shape_type = u32le(&shape.shp, 32)?;
    if ![3, 13].contains(&shape_type) {
        return Err("domain seeds require source flowpath polylines".into());
    }
    let table = Table::open(&c.dbf.path)?;
    if table.sha256 != c.dbf.sha256 || table.count != shape.records.len() {
        return Err("DBF digest or SHP physical record count mismatch".into());
    }
    let rows = table.indexed_rows(&[&c.id_column])?;
    let read_ms = start.elapsed().as_secs_f64() * 1000.;
    let hits = rows
        .par_iter()
        .enumerate()
        .map(|(i, row)| -> Result<Option<(i64, usize)>> {
            let Some(row) = row else { return Ok(None) };
            let raw = shape.record(i)?;
            if u32le(raw, 0)? == 0 {
                return Ok(None);
            };
            let p = parts(raw, shape_type)?;
            if !intersects_domain(&p, &ring) {
                return Ok(None);
            };
            let n = id(&row[0])?;
            if n <= 0 {
                return Err("intersecting source reach has nonpositive ID".into());
            }
            Ok(Some((n, i)))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let ids = hits.iter().map(|h| h.0).collect::<BTreeSet<_>>();
    if ids.is_empty() {
        return Err(
            "no source flowpath intersects domain; source coverage is not established".into(),
        );
    }
    let null_records = rows
        .iter()
        .enumerate()
        .map(|(i, row)| -> Result<usize> {
            Ok(usize::from(
                row.is_some() && u32le(shape.record(i)?, 0)? == 0,
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>();
    let selection_ms = start.elapsed().as_secs_f64() * 1000. - read_ms;
    let mut result = json!({"schema":"arwen.source-domain-selection.v1","node_ids":ids,
        "matched_physical_records":hits.iter().map(|h|h.1).collect::<Vec<_>>(),
        "source_record_count":rows.len(),"deleted_dbf_records":rows.iter().filter(|r|r.is_none()).count(),"null_active_geometries":null_records,
        "domain":{"crs_sha256":c.prj.sha256,"ring":ring},"crs_wkt":prj,
        "sources":{"shp":c.shp.sha256,"shx":c.shx.sha256,"dbf":c.dbf.sha256,"prj":c.prj.sha256},
        "data_license":c.data_license,"provenance":c.provenance,
        "policy":"XY straight-segment intersection in source CRS, boundary inclusive; active non-NULL flowpaths only; upstream closure is a separate source-network stage",
        "coverage":"intersections only; no tile/domain coverage, catchment completeness, projection, hydraulic parameters or runtime-fabric admission"});
    if let Some(metadata) = projection_metadata {
        result["domain_projection"] = metadata;
        result["coverage"] = json!("sampled projected boundary intersections only; no projection error bound, tile/domain coverage, catchment completeness, hydraulic parameters or runtime-fabric admission");
    }
    let payload = serde_json::to_vec(&result)?;
    std::fs::create_dir(&c.output)?;
    write_new(&c.output.join("selection.json"), &payload)?;
    let mut receipt = json!({"source":crate::SOURCE_REVISION,"output":c.output.join("selection.json"),
        "node_count":ids.len(),"matched_record_count":hits.len(),"null_active_geometries":null_records,
        "sha256":format!("{:x}",Sha256::digest(&payload)),"bytes":payload.len(),
        "read_hash_join_ms":read_ms,"intersection_ms":selection_ms,"total_ms":start.elapsed().as_secs_f64()*1000.,
        "rayon_workers":rayon::current_num_threads(),"parallelism":"file reads, source record validation, DBF decode and independent polyline intersections on Rayon; preparation only"});
    if result.get("domain_projection").is_some() {
        receipt["domain_projection_ms"] = json!(projection_ms);
    }
    Ok(receipt)
}

#[cfg(test)]
mod domain_tests {
    use super::*;
    #[test]
    fn polygonz_preserves_z_optional_m_and_rejects_broken_rings_arrays() {
        let mut raw = vec![0u8; 48];
        raw[..4].copy_from_slice(&15u32.to_le_bytes());
        raw[36..40].copy_from_slice(&1u32.to_le_bytes());
        raw[40..44].copy_from_slice(&4u32.to_le_bytes());
        for xy in [[0f64, 0.], [1., 0.], [1., 1.], [0., 0.]] {
            for v in xy {
                raw.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [2f64, 4., 2., 3., 4., 2.] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let xyz = parts(&raw, 15).unwrap();
        assert_eq!(xyz[0].axes, 1000);
        assert_eq!(xyz[0].points[2][2], 4f64.to_bits());
        for v in [-1e39f64, -1e39, -1e39, -1e39, -1e39, -1e39] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let xyzm = parts(&raw, 15).unwrap();
        assert_eq!(xyzm[0].axes, 3000);
        assert!(xyzm[0].points.iter().all(|p| p[3] == (-1e39f64).to_bits()));
        let mut bad = raw.clone();
        bad.pop();
        assert!(parts(&bad, 15).is_err());
        let mut bad = raw.clone();
        bad[96..104].copy_from_slice(&9f64.to_le_bytes());
        assert!(parts(&bad, 15).is_err());
        let mut bad = raw;
        bad[128..136].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(parts(&bad, 15).is_err());
    }
    fn water_roster() -> Value {
        json!({"schema":"arwen.source-waterbody-crosswalk.v1",
            "sources":{"waterbodies":{"sha256":"primary"},"additional_waterbodies":[{"sha256":"area"}]},
            "mapping":{"waterbody_id_column":"COMID"},
            "waterbodies":[{"waterbody_id":11,"source_index":0},{"waterbody_id":22,"source_index":1}]})
    }
    #[test]
    fn water_geometry_selects_bound_source_not_flowpath_roster() {
        let r = water_roster();
        assert_eq!(
            requested_ids(&r, Some(0), "primary", "COMID", 5).unwrap(),
            vec![11]
        );
        assert_eq!(
            requested_ids(&r, Some(1), "area", "COMID", 5).unwrap(),
            vec![22]
        );
        let mut single = r;
        single["sources"]
            .as_object_mut()
            .unwrap()
            .remove("additional_waterbodies");
        single["waterbodies"] = json!([{"waterbody_id":11}]);
        assert_eq!(
            requested_ids(&single, Some(0), "primary", "COMID", 5).unwrap(),
            vec![11]
        );
        let network = json!({"schema":"arwen.source-network-subset.v1","node_ids":[7,9]});
        assert_eq!(
            requested_ids(&network, None, "unused", "unused", 3).unwrap(),
            vec![7, 9]
        );
    }
    #[test]
    fn water_geometry_refuses_wrong_binding_and_ambiguous_ids() {
        let r = water_roster();
        for (index, sha, column, kind) in [
            (0, "area", "COMID", 5),
            (1, "area", "OTHER", 5),
            (2, "area", "COMID", 5),
            (0, "primary", "COMID", 3),
        ] {
            assert!(requested_ids(&r, Some(index), sha, column, kind).is_err());
        }
        assert!(requested_ids(&r, None, "primary", "COMID", 5).is_err());
        for mode in 0..9 {
            let mut r = water_roster();
            match mode {
                0 => r["waterbodies"][1]["waterbody_id"] = json!(11),
                1 => r["waterbodies"][1]["waterbody_id"] = json!(0),
                2 => r["waterbodies"][1]["waterbody_id"] = json!(1.5),
                3 => {
                    r["waterbodies"][1]
                        .as_object_mut()
                        .unwrap()
                        .remove("source_index");
                }
                4 => r["waterbodies"][1]["source_index"] = json!(-1),
                5 => r["waterbodies"][1]["source_index"] = json!(2),
                6 => r["sources"]["additional_waterbodies"] = Value::Null,
                7 => r["schema"] = json!("other"),
                _ => r["waterbodies"][1]["source_index"] = json!(0.5),
            }
            assert!(
                requested_ids(&r, Some(0), "primary", "COMID", 5).is_err(),
                "mode {mode}"
            );
        }
    }
    fn projected_domain() -> SelectionDomain {
        SelectionDomain {
            crs_sha256: None,
            crs: Some("EPSG:3857".into()),
            edge_segments: Some(2),
            ring: vec![[0., 0.], [1000., 0.], [1000., 1000.], [0., 1000.], [0., 0.]],
        }
    }
    #[test]
    fn legacy_domain_does_not_transform_or_add_provenance() {
        let mut d = projected_domain();
        d.crs_sha256 = Some("pinned".into());
        d.crs = None;
        d.edge_segments = None;
        let (ring, metadata) = d
            .resolve("pinned", "not a CRS; legacy needs no PROJ")
            .unwrap();
        assert_eq!(ring, d.ring);
        assert!(metadata.is_none());
        assert!(d.resolve("different", "EPSG:4326").is_err());
    }
    #[test]
    fn projected_domain_has_explicit_samples_axis_order_and_exact_word_provenance() {
        let d = projected_domain();
        let (ring, metadata) = d.resolve("source", "EPSG:4326").unwrap();
        let metadata = metadata.unwrap();
        assert_eq!(ring.len(), 9);
        assert_eq!(ring.first(), ring.last());
        assert_eq!(ring[0], [0., 0.]);
        // Web Mercator x=1000 is about .009 degrees longitude, not latitude.
        assert!((ring[2][0] - 0.008983152841195215).abs() < 1e-14);
        assert_eq!(ring[2][1], 0.);
        assert!(ring[4][1] > 0.00898 && ring[4][1] < 0.009);
        let words: Vec<[u64; 2]> =
            serde_json::from_value(metadata["source_ring_f64_bits"].clone()).unwrap();
        assert_eq!(
            words,
            ring.iter().map(|p| p.map(f64::to_bits)).collect::<Vec<_>>()
        );
        assert_eq!(metadata["edge_segments"], 2);
    }
    #[test]
    fn domain_rejects_mixed_unbounded_invalid_and_outside_projection_requests() {
        for segments in [0, 257, usize::MAX] {
            let mut d = projected_domain();
            d.edge_segments = Some(segments);
            assert!(d.resolve("source", "EPSG:4326").is_err());
        }
        let mut d = projected_domain();
        d.ring.resize(18, [0., 0.]);
        d.edge_segments = Some(256);
        assert!(d
            .resolve("source", "EPSG:4326")
            .unwrap_err()
            .to_string()
            .contains("4096"));
        for mode in 0..5 {
            let mut d = projected_domain();
            match mode {
                0 => d.crs_sha256 = Some("source".into()),
                1 => d.edge_segments = None,
                2 => d.crs = Some("".into()),
                3 => d.crs = Some("no-such-crs".into()),
                _ => {
                    d.crs = Some("EPSG:4326".into());
                    d.ring = vec![[0., 95.], [1., 95.], [1., 96.], [0., 96.], [0., 95.]];
                }
            }
            assert!(d.resolve("source", "EPSG:3857").is_err(), "mode {mode}");
        }
    }
    fn line(points: &[Point]) -> Vec<Part> {
        vec![Part {
            axes: 0,
            points: points
                .iter()
                .map(|p| [p[0].to_bits(), p[1].to_bits(), 0, 0])
                .collect(),
        }]
    }
    #[test]
    fn segment_intersection_includes_crossing_and_boundary_not_just_bbox() {
        let r = [[0., 0.], [2., 0.], [2., 2.], [0., 2.], [0., 0.]];
        domain_ring(&r).unwrap();
        assert!(intersects_domain(&line(&[[-1., 1.], [3., 1.]]), &r));
        assert!(intersects_domain(&line(&[[-1., 0.], [0., 0.]]), &r));
        assert!(intersects_domain(&line(&[[0., 0.], [2., 0.]]), &r));
        assert!(intersects_domain(&line(&[[0.5, 0.5], [1.5, 1.5]]), &r));
        assert!(!intersects_domain(
            &line(&[[-1., 1.], [-1., 3.], [1., 3.]]),
            &r
        ));
    }
    #[test]
    fn concave_ring_orientation_and_invalid_edges() {
        let translated = [
            [1e13, 1e13],
            [1e13 + 10., 1e13],
            [1e13 + 10., 1e13 + 10.],
            [1e13, 1e13 + 10.],
            [1e13, 1e13],
        ];
        domain_ring(&translated).unwrap();
        assert!(intersects_domain(
            &line(&[[1e13 - 10., 1e13 + 5.], [1e13 + 20., 1e13 + 5.]]),
            &translated
        ));
        assert!(
            domain_ring(&[[0., 0.], [1e308, 0.], [1e308, 1e308], [0., 1e308], [0., 0.]]).is_err()
        );
        let mut r = vec![
            [0., 0.],
            [3., 0.],
            [3., 1.],
            [1., 1.],
            [1., 3.],
            [0., 3.],
            [0., 0.],
        ];
        domain_ring(&r).unwrap();
        assert!(!intersects_domain(&line(&[[2., 2.], [3., 3.]]), &r));
        r.reverse();
        domain_ring(&r).unwrap();
        assert!(intersects_domain(&line(&[[0.5, 2.], [2., 0.5]]), &r));
        for invalid in [
            vec![[0., 0.], [2., 2.], [0., 2.], [2., 0.], [0., 0.]],
            vec![[0., 0.], [2., 0.], [1., 0.], [1., 1.], [0., 0.]],
            vec![[0., 0.], [1., 0.], [0., 0.]],
            vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        ] {
            assert!(domain_ring(&invalid).is_err());
        }
    }
}
