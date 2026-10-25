//! Catchments laid onto the analysis grid as a label raster (0 = none,
//! k + 1 = catchment k). Three sources, chosen by the run config:
//! - `gpkg`: the `divides` of a HYDRO-WORLD contract fabric (EPSG:5070 or the
//!   ARWEN LAEA row), read with `hydro_world::gpkg::read`;
//! - `shapefile`: a polygon shapefile in geographic coordinates (NHDPlus V2.1
//!   Catchment, NAD83), with the id column named in the config;
//! - `dem_tessellation`: a stand-in for domains with no fabric yet: stream-link
//!   sub-basins from the DEM, Priority-Flood per 1 degree tile (all tiles in
//!   parallel). It exists so a whole box can be parameterised before its fabric
//!   lands; it is not a fabric and claims no contract ids.
//! A polygon cell belongs to a catchment when the cell centre is inside it
//! (even-odd rule over all rings). A centre inside two polygons (overlapping or
//! shared-edge divides) belongs to the polygon with the LOWEST source index, so
//! the label raster is the same on every run whatever the thread schedule. A
//! catchment too small to hold a cell centre takes the cell under its centroid
//! for its statistics.
use crate::grid::Grid;
use crate::proj::Crs;
use crate::terrain::fkey;
use crate::Result;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct Catchments {
    pub ids: Vec<String>,
    pub labels: Vec<u32>,
    /// polygon centroid (lon, lat) when the source has polygons
    pub centroid: Vec<Option<(f64, f64)>>,
    /// area the source declares (km2), when it does
    pub declared_km2: Vec<Option<f64>>,
    pub kind: String,
    pub notes: Vec<String>,
}

type Ring = Vec<(f64, f64)>;

fn ring_centroid(rings: &[Ring]) -> Option<(f64, f64)> {
    // planar centroid on lon * cos(mean lat), lat (holes subtract by orientation-free even-odd use of |area| per ring sign)
    let mut lat_m = 0.0;
    let mut np = 0usize;
    for r in rings {
        for p in r {
            lat_m += p.1;
            np += 1;
        }
    }
    if np == 0 {
        return None;
    }
    lat_m /= np as f64;
    let k = lat_m.to_radians().cos();
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for (ri, r) in rings.iter().enumerate() {
        let (mut ar, mut x, mut y) = (0.0, 0.0, 0.0);
        for w in 0..r.len() {
            let (x0, y0) = (r[w].0 * k, r[w].1);
            let (x1, y1) = (r[(w + 1) % r.len()].0 * k, r[(w + 1) % r.len()].1);
            let cr = x0 * y1 - x1 * y0;
            ar += cr;
            x += (x0 + x1) * cr;
            y += (y0 + y1) * cr;
        }
        // signed sums: holes wind opposite to their exterior ring (shapefile and
        // GeoPackage both guarantee it), so they subtract
        let _ = ri;
        a += ar;
        cx += x;
        cy += y;
    }
    if a.abs() < 1e-300 {
        let (sx, sy) = rings.iter().flatten().fold((0.0, 0.0), |s, p| (s.0 + p.0, s.1 + p.1));
        return Some((sx / np as f64, sy / np as f64));
    }
    Some((cx / (3.0 * a) / k, cy / (3.0 * a)))
}

/// cells whose centre lies inside the rings (even-odd)
fn rasterize(g: &Grid, rings: &[Ring]) -> Vec<u32> {
    let mut out = vec![];
    let (mut ymin, mut ymax) = (f64::MAX, f64::MIN);
    for r in rings {
        for p in r {
            ymin = ymin.min(p.1);
            ymax = ymax.max(p.1);
        }
    }
    if ymin > ymax {
        return out;
    }
    let cpd = g.cells_per_degree as f64;
    let j0 = (((g.north - ymax) * cpd - 0.5).ceil().max(0.0)) as usize;
    let j1 = (((g.north - ymin) * cpd - 0.5).floor()).min(g.ny as f64 - 1.0);
    if j1 < 0.0 {
        return out;
    }
    let mut xs: Vec<f64> = vec![];
    for j in j0..=(j1 as usize) {
        let y = g.lat(j);
        xs.clear();
        for r in rings {
            let n = r.len();
            for k in 0..n {
                let (x0, y0) = r[k];
                let (x1, y1) = r[(k + 1) % n];
                if (y0 <= y && y < y1) || (y1 <= y && y < y0) {
                    xs.push(x0 + (y - y0) * (x1 - x0) / (y1 - y0));
                }
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for p in xs.chunks_exact(2) {
            let ia = ((p[0] - g.west) * cpd - 0.5).ceil().max(0.0) as usize;
            let ib = ((p[1] - g.west) * cpd - 0.5).ceil().min(g.nx as f64).max(0.0) as usize;
            for i in ia..ib {
                out.push((j * g.nx + i) as u32);
            }
        }
    }
    out
}

fn finish(g: &Grid, ids: Vec<String>, polys: Vec<Vec<Ring>>, declared: Vec<Option<f64>>, kind: &str) -> Catchments {
    // Tie rule: a cell covered by two polygons takes the lowest polygon index.
    // fetch_min is order-independent, so parallel rasterisation gives the same
    // raster on every run (a plain store left the cell to the last writer and the
    // Rhine derive differed run to run in 1,437 of 18,917 catchments).
    let labels: Vec<AtomicU32> = (0..g.n()).map(|_| AtomicU32::new(u32::MAX)).collect();
    let cents: Vec<Option<(f64, f64)>> = polys
        .par_iter()
        .enumerate()
        .map(|(k, rings)| {
            for c in rasterize(g, rings) {
                labels[c as usize].fetch_min(k as u32 + 1, Ordering::Relaxed);
            }
            ring_centroid(rings)
        })
        .collect();
    let labels: Vec<u32> = labels
        .into_iter()
        .map(|a| match a.into_inner() {
            u32::MAX => 0,
            l => l,
        })
        .collect();
    Catchments { ids, labels, centroid: cents, declared_km2: declared, kind: kind.into(), notes: vec![] }
}

pub fn from_shapefile(g: &Grid, path: &str, id_field: &str, id_format: &str, area_field: Option<&str>) -> Result<Catchments> {
    let mut reader = shapefile::Reader::from_path(path).map_err(|e| format!("{path}: {e}"))?;
    let mut ids = vec![];
    let mut polys = vec![];
    let mut declared = vec![];
    for r in reader.iter_shapes_and_records_as::<shapefile::Polygon, shapefile::dbase::Record>() {
        let (poly, rec) = r.map_err(|e| format!("{path}: {e}"))?;
        let num = |f: &str| -> Option<f64> {
            match rec.get(f)? {
                shapefile::dbase::FieldValue::Numeric(v) => *v,
                shapefile::dbase::FieldValue::Integer(v) => Some(*v as f64),
                shapefile::dbase::FieldValue::Double(v) => Some(*v),
                shapefile::dbase::FieldValue::Float(v) => v.map(|x| x as f64),
                shapefile::dbase::FieldValue::Character(Some(s)) => s.trim().parse().ok(),
                _ => None,
            }
        };
        let id = num(id_field).ok_or_else(|| format!("{path}: record without numeric {id_field}"))?;
        ids.push(id_format.replace("{}", &format!("{}", id as i64)));
        declared.push(area_field.and_then(num));
        let rings: Vec<Ring> = poly.rings().iter().map(|r| r.points().iter().map(|p| (p.x, p.y)).collect()).collect();
        polys.push(rings);
    }
    Ok(finish(g, ids, polys, declared, "shapefile"))
}

/// GeoPackage binary geometry -> rings in the fabric CRS
fn gpkg_rings(b: &[u8]) -> Result<Vec<Ring>> {
    if b.len() < 8 || &b[0..2] != b"GP" {
        return Err("not GeoPackage binary geometry".into());
    }
    let flags = b[3];
    let env = match (flags >> 1) & 7 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        e => return Err(format!("bad envelope code {e}").into()),
    };
    let w = &b[8 + env..];
    let mut p = 0usize;
    let mut rings = vec![];
    fn u32at(w: &[u8], p: usize, le: bool) -> u32 {
        let a = [w[p], w[p + 1], w[p + 2], w[p + 3]];
        if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }
    }
    fn f64at(w: &[u8], p: usize, le: bool) -> f64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&w[p..p + 8]);
        if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }
    }
    fn polygon(w: &[u8], p: &mut usize, rings: &mut Vec<Ring>) -> Result<()> {
        let le = w[*p] == 1;
        let t = u32at(w, *p + 1, le);
        *p += 5;
        match t {
            3 => {
                let nr = u32at(w, *p, le) as usize;
                *p += 4;
                for _ in 0..nr {
                    let np = u32at(w, *p, le) as usize;
                    *p += 4;
                    let mut r = Vec::with_capacity(np);
                    for _ in 0..np {
                        r.push((f64at(w, *p, le), f64at(w, *p + 8, le)));
                        *p += 16;
                    }
                    rings.push(r);
                }
                Ok(())
            }
            6 => {
                let ng = u32at(w, *p, le) as usize;
                *p += 4;
                for _ in 0..ng {
                    polygon(w, p, rings)?;
                }
                Ok(())
            }
            t => Err(format!("WKB type {t} is not a 2D polygon").into()),
        }
    }
    polygon(w, &mut p, &mut rings)?;
    Ok(rings)
}

pub fn from_gpkg(g: &Grid, path: &str) -> Result<(Catchments, hydro_world::contract::Fabric)> {
    let (fab, _rep) = hydro_world::gpkg::read(path)?;
    let crs = Crs::from_srs(&fab.srs.organization, fab.srs.organization_coordsys_id, &fab.srs.definition)?;
    let rows = fab.divides.par_iter().map(|d| -> Result<_> {
        let id = d.divide_id.clone().ok_or_else(|| format!("{path}: divide without divide_id"))?;
        let geom = d.geom.as_ref().ok_or_else(|| format!("{path}: {id} has no geometry"))?;
        let rings = gpkg_rings(&geom.0)?;
        let rings: Vec<Ring> = rings.into_iter().map(|r| r.into_iter().map(|(x, y)| crs.inverse(x, y)).collect()).collect();
        Ok((id,rings,d.areasqkm))
    }).collect::<Result<Vec<_>>>()?;
    let (mut ids,mut polys,mut declared)=(Vec::with_capacity(rows.len()),Vec::with_capacity(rows.len()),Vec::with_capacity(rows.len()));
    for (id,rings,area) in rows {ids.push(id);polys.push(rings);declared.push(area);}
    Ok((finish(g, ids, polys, declared, "gpkg"), fab))
}

/// Stand-in sub-basins from the DEM (see module docs). `outside` marks cells
/// that belong to no catchment (open sea, no DEM).
pub fn dem_tessellation(g: &Grid, dem: &[f32], outside: &[bool], tile: usize, stream_area_km2: f64, id_namespace: u64) -> Catchments {
    let (tx, ty) = (g.nx.div_ceil(tile), g.ny.div_ceil(tile));
    let thr = stream_area_km2 * 1e6;
    // per tile: local labels (0 none, k+1)
    let per_tile: Vec<(usize, Vec<u32>, u32, Vec<(u32, usize)>)> = (0..tx * ty)
        .into_par_iter()
        .map(|t| {
            let (bi, bj) = (t % tx, t / tx);
            let (i0, j0) = (bi * tile, bj * tile);
            let (w, h) = (tile.min(g.nx - i0), tile.min(g.ny - j0));
            let gidx = |li: usize, lj: usize| (j0 + lj) * g.nx + i0 + li;
            let n = w * h;
            let inside = |k: usize| !outside[gidx(k % w, k / w)];
            let mut parent = vec![u32::MAX; n];
            let mut visited = vec![false; n];
            let mut order: Vec<u32> = Vec::with_capacity(n);
            let mut heap: BinaryHeap<Reverse<(u32, u32, u32)>> = BinaryHeap::new();
            let mut seq = 0u32;
            for k in 0..n {
                if !inside(k) {
                    continue;
                }
                let (li, lj) = (k % w, k / w);
                let edge = li == 0 || lj == 0 || li == w - 1 || lj == h - 1;
                let coast = !edge
                    && [(-1isize, -1isize), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                        .iter()
                        .any(|&(a, b)| !inside(((lj as isize + a) as usize) * w + (li as isize + b) as usize));
                if edge || coast {
                    visited[k] = true;
                    heap.push(Reverse((fkey(dem[gidx(li, lj)]), seq, k as u32)));
                    seq += 1;
                }
            }
            while let Some(Reverse((key, _, k))) = heap.pop() {
                let k = k as usize;
                order.push(k as u32);
                let fill = f32::from_bits(if key & 0x8000_0000 != 0 { key & 0x7fff_ffff } else { !key });
                let (li, lj) = ((k % w) as isize, (k / w) as isize);
                for (a, b) in [(-1isize, -1isize), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                    let (y, x) = (lj + a, li + b);
                    if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
                        continue;
                    }
                    let m = y as usize * w + x as usize;
                    if visited[m] || !inside(m) {
                        continue;
                    }
                    visited[m] = true;
                    parent[m] = k as u32;
                    let e = dem[gidx(x as usize, y as usize)];
                    let f = if e.is_finite() && (!fill.is_finite() || e > fill) { e } else { fill };
                    heap.push(Reverse((fkey(f), seq, m as u32)));
                    seq += 1;
                }
            }
            let mut acc: Vec<f64> = (0..n).map(|k| g.cell_area_m2(j0 + k / w)).collect();
            for &k in order.iter().rev() {
                let p = parent[k as usize];
                if p != u32::MAX {
                    acc[p as usize] += acc[k as usize];
                }
            }
            let stream: Vec<bool> = acc.iter().map(|&a| a >= thr).collect();
            let mut nchild = vec![0u8; n];
            for &k in &order {
                let k = k as usize;
                let p = parent[k];
                if stream[k] && p != u32::MAX && stream[p as usize] {
                    nchild[p as usize] = nchild[p as usize].saturating_add(1);
                }
            }
            // non-stream cells that leave the tile at an interior tile edge get a
            // label of their own that is joined, after all tiles, to the catchment
            // their flow continues into across the edge (the lowest neighbour in the
            // adjacent tile). Only cells draining to the sea or the domain edge
            // without reaching a stream share one coastal remnant per tile.
            let mut lab = vec![0u32; n];
            let mut next = 1u32;
            let mut remnant = 0u32;
            let mut joins: Vec<(u32, usize)> = vec![];
            for &k in &order {
                let k = k as usize;
                let p = parent[k];
                lab[k] = if p == u32::MAX {
                    if stream[k] {
                        next += 1;
                        next - 1
                    } else {
                        let (li, lj) = ((k % w) as isize, (k / w) as isize);
                        let mut best: Option<(u32, usize)> = None;
                        for (a, b) in [(-1isize, -1isize), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                            let (y, x) = (lj + a, li + b);
                            if y >= 0 && x >= 0 && y < h as isize && x < w as isize {
                                continue; // same tile
                            }
                            let (gy, gx) = (j0 as isize + y, i0 as isize + x);
                            if gy < 0 || gx < 0 || gy >= g.ny as isize || gx >= g.nx as isize {
                                continue; // domain edge
                            }
                            let gc = gy as usize * g.nx + gx as usize;
                            if outside[gc] {
                                continue;
                            }
                            let key = fkey(dem[gc]);
                            if best.map(|b| key < b.0).unwrap_or(true) {
                                best = Some((key, gc));
                            }
                        }
                        // join only downhill in the total order (elevation, cell index),
                        // so join chains strictly descend and can never loop
                        match best {
                            Some((_, gc)) => {
                                next += 1;
                                joins.push((next - 1, gc));
                                next - 1
                            }
                            None => {
                                if remnant == 0 {
                                    remnant = next;
                                    next += 1;
                                }
                                remnant
                            }
                        }
                    }
                } else if stream[k] && nchild[p as usize] >= 2 {
                    next += 1;
                    next - 1
                } else {
                    lab[p as usize]
                };
            }
            (t, lab, next - 1, joins)
        })
        .collect();
    let mut offset = vec![0u32; tx * ty];
    let mut total = 0u32;
    for (t, _, n, _) in &per_tile {
        offset[*t] = total;
        total += n;
    }
    let mut labels = vec![0u32; g.n()];
    // tiles own disjoint cells, so rows of a tile row can be filled in parallel
    labels.par_chunks_mut(g.nx).enumerate().for_each(|(j, row)| {
        let bj = j / tile;
        for bi in 0..tx {
            let t = bj * tx + bi;
            let (_, lab, _, _) = &per_tile[t];
            let i0 = bi * tile;
            let w = tile.min(g.nx - i0);
            let lj = j - bj * tile;
            for li in 0..w {
                let l = lab[lj * w + li];
                if l != 0 {
                    row[i0 + li] = offset[t] + l;
                }
            }
        }
    });
    // join edge groups to what their flow enters across the tile edge. Each
    // group has one outgoing join, so every joined set has at most one sink: a
    // stream or coastal catchment (the set takes it), or a loop of groups along
    // a flat edge (the loop becomes one catchment). Union-find keeps the sink.
    let mut is_group = vec![false; total as usize + 1];
    for (t, _, _, joins) in &per_tile {
        for &(l, _) in joins {
            is_group[(offset[*t] + l) as usize] = true;
        }
    }
    let mut parent: Vec<u32> = (0..=total).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for (t, _, _, joins) in &per_tile {
        for &(l, gc) in joins {
            let a = find(&mut parent, offset[*t] + l);
            let b = find(&mut parent, labels[gc]);
            if a != b {
                // keep a non-group root (stream or coastal catchment) as the set's id
                if is_group[a as usize] {
                    parent[a as usize] = b;
                } else {
                    parent[b as usize] = a;
                }
            }
        }
    }
    let mut resolved: Vec<u32> = {
        let mut p = parent.clone();
        (0..=total).map(|l| find(&mut p, l)).collect()
    };
    // a loop of groups along a flat tile edge has no sink of its own: it spills
    // into the lowest neighbouring catchment that has one (scan of the cells
    // beside tile edges, Rayon over edge lines; repeated for loops that only
    // touch other loops)
    for _ in 0..8 {
        let mut lines: Vec<(bool, usize)> = vec![];
        for bj in 1..ty {
            lines.push((true, bj * tile - 1));
            lines.push((true, bj * tile));
        }
        for bi in 1..tx {
            lines.push((false, bi * tile - 1));
            lines.push((false, bi * tile));
        }
        let cand: Vec<(u32, u32, u32)> = lines
            .par_iter()
            .flat_map_iter(|&(is_row, k)| {
                let mut v = vec![];
                let len = if is_row { g.nx } else { g.ny };
                for q in 0..len {
                    let (j, i) = if is_row { (k, q) } else { (q, k) };
                    let c = j * g.nx + i;
                    let la = labels[c];
                    if la == 0 {
                        continue;
                    }
                    let ra = resolved[la as usize];
                    if !is_group[ra as usize] {
                        continue;
                    }
                    for (a, b) in [(-1isize, -1isize), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                        let (y, x) = (j as isize + a, i as isize + b);
                        if y < 0 || x < 0 || y >= g.ny as isize || x >= g.nx as isize {
                            continue;
                        }
                        let n = y as usize * g.nx + x as usize;
                        let lb = labels[n];
                        if lb == 0 {
                            continue;
                        }
                        let rb = resolved[lb as usize];
                        if rb != ra && !is_group[rb as usize] {
                            v.push((ra, fkey(dem[n].max(dem[c])), rb));
                        }
                    }
                }
                v.into_iter()
            })
            .collect();
        if cand.is_empty() {
            break;
        }
        let mut spill: std::collections::HashMap<u32, (u32, u32)> = Default::default();
        for (ra, key, rb) in cand {
            let e = spill.entry(ra).or_insert((key, rb));
            if (key, rb) < *e {
                *e = (key, rb);
            }
        }
        resolved.par_iter_mut().for_each(|r| {
            if let Some(&(_, rb)) = spill.get(r) {
                *r = rb;
            }
        });
    }
    let mut compact = vec![0u32; total as usize + 1];
    let mut m = 0u32;
    let mut used = vec![false; total as usize + 1];
    for &r in &resolved[1..] {
        used[r as usize] = true;
    }
    for l in 1..=total as usize {
        if used[l] {
            m += 1;
            compact[l] = m;
        }
    }
    labels.par_iter_mut().for_each(|l| {
        if *l != 0 {
            *l = compact[resolved[*l as usize] as usize];
        }
    });
    let njoin: usize = per_tile.iter().map(|t| t.3.len()).sum();
    let joined = (1..=total as usize).filter(|&l| is_group[l] && !is_group[resolved[l] as usize]).count();
    let notes_join = format!("{njoin} tile-edge groups, {joined} joined to a stream or coastal catchment, {} labels before joins, {m} after", total);
    let total = m;
    let base = id_namespace * 10_000_000_000;
    let ids = (0..total).map(|k| format!("cat-{}", base + k as u64 + 1)).collect();
    Catchments {
        ids,
        labels,
        centroid: vec![None; total as usize],
        declared_km2: vec![None; total as usize],
        kind: "dem_tessellation".into(),
        notes: vec![format!(
            "stand-in sub-basins: stream links at {stream_area_km2} km2 contributing area, Priority-Flood per {tile}x{tile}-cell tile, id namespace {id_namespace} (test only, not a contract namespace); cells leaving a tile before reaching a stream join the catchment their flow enters across the tile edge; cells draining to the sea or the domain edge without reaching a stream form one coastal remnant per tile"
        ), notes_join],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f64, x1: f64) -> Vec<Ring> {
        vec![vec![(x0, 0.0), (x1, 0.0), (x1, 4.0), (x0, 4.0)]]
    }

    #[test]
    fn overlapping_polygons_take_the_lowest_index_every_run() {
        // 4 x 4 one-degree cells; polygon A covers centres in columns 0-2, B columns 1-3,
        // so columns 1 and 2 are covered by both
        let g = Grid { west: 0.0, north: 4.0, cells_per_degree: 1, nx: 4, ny: 4 };
        let a = rect(0.0, 3.0);
        let b = rect(1.0, 4.0);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for _ in 0..200 {
            // A first: A is label 1 and wins the overlap
            let ab = pool.install(|| finish(&g, vec!["A".into(), "B".into()], vec![a.clone(), b.clone()], vec![None, None], "test"));
            // B first: B is label 1 and wins the overlap
            let ba = pool.install(|| finish(&g, vec!["B".into(), "A".into()], vec![b.clone(), a.clone()], vec![None, None], "test"));
            for j in 0..4 {
                assert_eq!(&ab.labels[j * 4..j * 4 + 4], &[1, 1, 1, 2], "row {j} A first");
                assert_eq!(&ba.labels[j * 4..j * 4 + 4], &[2, 1, 1, 1], "row {j} B first");
            }
        }
        // a cell no polygon covers stays 0
        let one = finish(&g, vec!["A".into()], vec![rect(0.0, 1.0)], vec![None], "test");
        assert_eq!(&one.labels[0..4], &[1, 0, 0, 0]);
    }
}
