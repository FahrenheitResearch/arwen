//! `vector-basins` method: basin polygons with downstream ids (HydroBASINS
//! level 12) as divides, and in each basin the main stem of the river lines
//! (HydroRIVERS, reaches tagged with their level-12 basin) as its flowpath.
//! Opt-in cross-check route only: the source row marks these inputs
//! `research-only`, so a private cross-check needs the named licence opt-in; it cannot ship.
use crate::build::{load_lakes, threads, Hydro, LakeIn, ReachIn, Receipt};
use crate::grid::BBox;
use crate::source::{BuildSpec, Input, SourceRow};
use crate::{geo, shp, Result};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

fn column(inp: &Input, meaning: &str) -> Result<String> {
    inp.columns.iter().find(|(_, v)| v.split(' ').next() == Some(meaning)).map(|(k, _)| k.clone()).ok_or_else(|| format!("{} maps no column to '{meaning}'", inp.dataset).into())
}

/// Shapefile header bounding box (xmin, ymin, xmax, ymax).
fn shp_bbox(p: &Path) -> Result<[f64; 4]> {
    let mut f = std::fs::File::open(p)?;
    let mut h = [0u8; 100];
    std::io::Read::read_exact(&mut f, &mut h)?;
    let d = |o: usize| f64::from_le_bytes(h[o..o + 8].try_into().unwrap());
    Ok([d(36), d(44), d(52), d(60)])
}

fn region_files(root: &Path, inp: &Input, b: &BBox) -> Result<Vec<std::path::PathBuf>> {
    let pat = inp.local.as_deref().ok_or_else(|| format!("{} has no local path", inp.dataset))?;
    let regions: Vec<String> = if pat.contains("{region}") { inp.regions.clone() } else { vec![String::new()] };
    let mut out = Vec::new();
    for r in regions {
        let p = root.join(pat.replace("{region}", &r));
        if !p.exists() {
            continue;
        }
        let bb = shp_bbox(&p)?;
        if bb[0] <= b.east && bb[2] >= b.west && bb[1] <= b.north && bb[3] >= b.south {
            out.push(p);
        }
    }
    if out.is_empty() {
        return Err(format!("no local {} file covers the box (pattern {pat})", inp.dataset).into());
    }
    Ok(out)
}

/// Sutherland-Hodgman clip of one ring against the box (lon/lat).
fn clip_ring(r: &[(f64, f64)], b: &BBox) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = r.to_vec();
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    let edges: [(Box<dyn Fn((f64, f64)) -> bool>, Box<dyn Fn((f64, f64), (f64, f64)) -> (f64, f64)>); 4] = [
        (Box::new(|p: (f64, f64)| p.0 >= b.west), Box::new(|a: (f64, f64), c: (f64, f64)| (b.west, a.1 + (c.1 - a.1) * (b.west - a.0) / (c.0 - a.0)))),
        (Box::new(|p: (f64, f64)| p.0 <= b.east), Box::new(|a: (f64, f64), c: (f64, f64)| (b.east, a.1 + (c.1 - a.1) * (b.east - a.0) / (c.0 - a.0)))),
        (Box::new(|p: (f64, f64)| p.1 >= b.south), Box::new(|a: (f64, f64), c: (f64, f64)| (a.0 + (c.0 - a.0) * (b.south - a.1) / (c.1 - a.1), b.south))),
        (Box::new(|p: (f64, f64)| p.1 <= b.north), Box::new(|a: (f64, f64), c: (f64, f64)| (a.0 + (c.0 - a.0) * (b.north - a.1) / (c.1 - a.1), b.north))),
    ];
    for (inside, cut) in edges.iter() {
        if pts.is_empty() {
            break;
        }
        let mut out = Vec::with_capacity(pts.len() + 4);
        for i in 0..pts.len() {
            let (cur, prev) = (pts[i], pts[(i + pts.len() - 1) % pts.len()]);
            match (inside(cur), inside(prev)) {
                (true, true) => out.push(cur),
                (true, false) => {
                    out.push(cut(prev, cur));
                    out.push(cur);
                }
                (false, true) => out.push(cut(prev, cur)),
                (false, false) => {}
            }
        }
        pts = out;
    }
    pts
}

pub fn vector_hydro(row: &SourceRow, spec: &BuildSpec, b: &BBox, root: &Path, coarsen: u32, rc: &mut Receipt) -> Result<Hydro> {
    let cat = row.inputs.get("catchments").ok_or("row has no catchments input")?;
    let riv = row.inputs.get("river_network").ok_or("row has no river_network input")?;
    let min_inside = spec.basin_min_inside_fraction.unwrap_or(0.5);
    let proj = geo::Laea::new((b.south + b.north) / 2.0, (b.west + b.east) / 2.0);
    // Basins.
    struct Basin {
        id: i64,
        down: i64,
        rings: Vec<Vec<(f64, f64)>>,
    }
    let basins: Vec<Basin> = rc.time("1 read basin polygons", &threads(), || -> Result<Vec<Basin>> {
        let mut all = Vec::new();
        for p in region_files(root, cat, b)? {
            let dbf = shp::Dbf::open(&p.with_extension("dbf"))?;
            let id = dbf.i64s(&column(cat, "id")?)?;
            let down = dbf.i64s(&column(cat, "down")?)?;
            let shapes = shp::read_shapes(&p)?;
            all.extend(shapes.into_par_iter().enumerate().filter_map(|(i, s)| match s {
                shp::Shape::Parts(rings) => {
                    let hit = rings.iter().flatten().any(|&(x, y)| b.contains(x, y)) || rings.iter().any(|r| {
                        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                        for &(x, y) in r {
                            x0 = x0.min(x);
                            y0 = y0.min(y);
                            x1 = x1.max(x);
                            y1 = y1.max(y);
                        }
                        x0 <= b.east && x1 >= b.west && y0 <= b.north && y1 >= b.south
                    });
                    hit.then(|| Basin { id: id[i], down: down[i], rings })
                }
                _ => None,
            }).collect::<Vec<_>>());
        }
        Ok(all)
    })?;
    // Clip to the box and keep basins mostly inside.
    struct Kept {
        id: i64,
        down: i64,
        exterior: Vec<(f64, f64)>,
        holes: Vec<Vec<(f64, f64)>>,
        area_km2: f64,
    }
    let pa = |r: &[(f64, f64)]| geo::ring_area(&r.iter().map(|&(x, y)| proj.forward(x, y)).collect::<Vec<_>>());
    let mut kept: Vec<Kept> = rc.time("1 clip basins to box", &threads(), || {
        basins
            .par_iter()
            .filter_map(|bs| {
                let full: f64 = bs.rings.iter().map(|r| pa(r)).sum::<f64>().abs();
                let clipped: Vec<Vec<(f64, f64)>> = bs.rings.iter().map(|r| clip_ring(r, b)).filter(|r| r.len() >= 3).collect();
                // ESRI rings: exteriors clockwise (negative), holes counter-clockwise.
                let signed: Vec<(f64, &Vec<(f64, f64)>)> = clipped.iter().map(|r| (pa(r), r)).collect();
                let inside: f64 = -signed.iter().map(|s| s.0).sum::<f64>();
                if full <= 0.0 || inside / full < min_inside {
                    return None;
                }
                let (_, ext) = signed.iter().filter(|s| s.0 < 0.0).min_by(|a, b| a.0.total_cmp(&b.0))?;
                let holes = signed.iter().filter(|s| s.0 > 0.0).map(|s| s.1.clone()).collect();
                Some(Kept { id: bs.id, down: bs.down, exterior: (*ext).clone(), holes, area_km2: inside / 1e6 })
            })
            .collect()
    });
    kept.sort_by_key(|k| k.id);
    rc.count("basins_touching_box", basins.len());
    rc.count("basins_kept", kept.len());
    let kept_ids: HashSet<i64> = kept.iter().map(|k| k.id).collect();
    // Rivers.
    struct River {
        id: i64,
        down: i64,
        upland: f64,
        basin: i64,
        line: Vec<(f64, f64)>,
    }
    let rivers: Vec<River> = rc.time("2 read river lines", &threads(), || -> Result<Vec<River>> {
        let mut all = Vec::new();
        for p in region_files(root, riv, b)? {
            let dbf = shp::Dbf::open(&p.with_extension("dbf"))?;
            let id = dbf.i64s(&column(riv, "id")?)?;
            let down = dbf.i64s(&column(riv, "down")?)?;
            let up = dbf.f64s(&column(riv, "upstream_km2")?)?;
            let bas = dbf.i64s(&column(riv, "basin")?)?;
            let want: Vec<usize> = (0..dbf.n).filter(|&i| kept_ids.contains(&bas[i])).collect();
            let shapes = shp::read_shapes(&p)?;
            all.extend(want.into_par_iter().filter_map(|i| match &shapes[i] {
                shp::Shape::Parts(parts) => Some(River { id: id[i], down: down[i], upland: up[i], basin: bas[i], line: parts.concat() }),
                _ => None,
            }).collect::<Vec<_>>());
        }
        Ok(all)
    })?;
    rc.count("river_reaches_in_kept_basins", rivers.len());
    let mut by_basin: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, r) in rivers.iter().enumerate() {
        by_basin.entry(r.basin).or_default().push(i);
    }
    // DEM for slopes.
    let res = spec.grid_arcsec.ok_or("vector-basins needs grid_arcsec for the DEM")?;
    let (g, z, _) = crate::build::dem_on_grid(row, b, root, res, coarsen, None, rc)?;
    let zat = |p: (f64, f64)| g.rowcol(p.0, p.1).map(|(r, c)| z[r * g.nx + c] as f64);
    // Main stem per basin.
    let with_river: Vec<usize> = (0..kept.len()).filter(|&k| by_basin.contains_key(&kept[k].id)).collect();
    let index: HashMap<i64, usize> = with_river.iter().enumerate().map(|(j, &k)| (kept[k].id, j)).collect();
    rc.count("basins_without_river_dropped", kept.len() - with_river.len());
    rc.count("basins_without_river_km2", with_river.iter().fold(kept.iter().map(|k| k.area_km2).sum::<f64>(), |a, &k| a - kept[k].area_km2));
    let mut lake_reach_vertices: Vec<(f64, f64, f64, usize)> = Vec::new();
    let reaches: Vec<ReachIn> = rc.time("3 basin main stems, slopes", &threads(), || {
        with_river
            .par_iter()
            .map(|&k| {
                let bs = &kept[k];
                let members = &by_basin[&bs.id];
                let in_basin: HashSet<i64> = members.iter().map(|&i| rivers[i].id).collect();
                let outlet = *members.iter().max_by(|&&a, &&b| rivers[a].upland.total_cmp(&rivers[b].upland).then(rivers[b].id.cmp(&rivers[a].id))).unwrap();
                let mut chain = vec![outlet];
                loop {
                    let cur = rivers[*chain.last().unwrap()].id;
                    let up = members.iter().filter(|&&i| rivers[i].down == cur && in_basin.contains(&rivers[i].id)).max_by(|&&a, &&b| rivers[a].upland.total_cmp(&rivers[b].upland).then(rivers[b].id.cmp(&rivers[a].id)));
                    match up {
                        Some(&u) => chain.push(u),
                        None => break,
                    }
                }
                chain.reverse();
                let mut line: Vec<(f64, f64)> = Vec::new();
                for &i in &chain {
                    for &p in &rivers[i].line {
                        if line.last() != Some(&p) {
                            line.push(p);
                        }
                    }
                }
                if line.len() < 2 {
                    line.push(line[0]);
                }
                let len = geo::polyline_geodesic_m(&line);
                let slope = match (zat(line[0]), zat(*line.last().unwrap())) {
                    (Some(a), Some(c)) if len > 0.0 => (a - c) / len,
                    _ => 0.0,
                };
                ReachIn {
                    down: if kept_ids.contains(&bs.down) { index.get(&bs.down).copied() } else { None },
                    line,
                    divide: Some((bs.exterior.clone(), bs.holes.clone())),
                    local_area_km2: bs.area_km2,
                    slope,
                    hf_id: bs.id as f64,
                    terrain: None,
                }
            })
            .collect()
    });
    for (j, &k) in with_river.iter().enumerate() {
        for &i in &by_basin[&kept[k].id] {
            for &(x, y) in &rivers[i].line {
                lake_reach_vertices.push((x, y, rivers[i].upland, j));
            }
        }
    }
    // Lakes: nearest river vertex within the radius (km) whose upstream area matches.
    let lakes = load_lakes(row, spec, b, root)?;
    rc.count("lakes_in_box", lakes.len());
    let cell = 0.05;
    let mut hash: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (v, p) in lake_reach_vertices.iter().enumerate() {
        hash.entry(((p.0 / cell).floor() as i64, (p.1 / cell).floor() as i64)).or_default().push(v);
    }
    let r_km = spec.lake_snap_radius;
    let mut by_reach: HashMap<usize, usize> = HashMap::new();
    let mut unsnapped = 0;
    for (li, l) in lakes.iter().enumerate() {
        let (cx, cy) = ((l.lon / cell).floor() as i64, (l.lat / cell).floor() as i64);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for &v in hash.get(&(cx + dx, cy + dy)).into_iter().flatten() {
                    let p = lake_reach_vertices[v];
                    let d = geo::geodesic_m(l.lon, l.lat, p.0, p.1) / 1000.0;
                    if d > r_km {
                        continue;
                    }
                    if l.wshd_km2 > 0.0 {
                        let ratio = p.2 / l.wshd_km2;
                        if ratio > spec.lake_snap_area_ratio || ratio < 1.0 / spec.lake_snap_area_ratio {
                            continue;
                        }
                    }
                    if best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, p.3));
                    }
                }
            }
        }
        match best {
            Some((_, j)) => {
                let e = by_reach.entry(j).or_insert(li);
                if lakes[li].area_km2 > lakes[*e].area_km2 {
                    *e = li;
                }
            }
            None => unsnapped += 1,
        }
    }
    rc.count("lakes_unsnapped", unsnapped);
    let mut out_lakes = Vec::new();
    let mut terminal = 0;
    let mut ks: Vec<_> = by_reach.into_iter().collect();
    ks.sort();
    for (j, li) in ks {
        if reaches[j].down.is_none() {
            terminal += 1;
            continue;
        }
        let l = &lakes[li];
        out_lakes.push(LakeIn { hylak: l.hylak, area_km2: l.area_km2, mean_depth_m: l.depth_m, weir_e_m: zat((l.lon, l.lat)).unwrap_or(0.0), outlet: (l.lon, l.lat), reach: j });
    }
    rc.count("lakes_at_domain_outlet_dropped", terminal);
    rc.count("lakes_snapped_to_inner_reaches", out_lakes.len());
    rc.count("reaches", reaches.len());
    Ok(Hydro { reaches, lakes: out_lakes, dams: Vec::new() })
}
