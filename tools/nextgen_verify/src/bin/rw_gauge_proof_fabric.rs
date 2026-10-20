// SPDX-License-Identifier: Apache-2.0
//! Gauge-matching proof fixture: a coarse D8 reach network from Copernicus DEM
//! GLO-90 tiles, written as a HYDRO-WORLD contract GeoPackage (flowpaths,
//! nexus, network only). It exists so station-to-reach matching can be proved
//! on a non-US basin before the hydro-world fabric builder lands. It is NOT a
//! product fabric: no catchment polygons, no channel or model attributes.
//!
//! Method: each output cell takes the MINIMUM of the GLO-90 pixels whose
//! centres fall in it (keeps valley floors at coarse resolution); depressions
//! and flats are resolved by Priority-Flood (Barnes, Lehman and Mulla 2014,
//! Computers and Geosciences 62) whose processing order gives every cell a
//! downstream neighbour; accumulation runs in reverse processing order with
//! spherical cell areas; channel cells are those draining at least the
//! threshold area; reaches run between junctions.
use hydro_world::contract::{Contents, Fabric, Flowpath, MetadataRow, NetworkRow, Nexus, Srs, CONTRACT};
use nextgen_verify::geo::{gpkg_line_bytes, gpkg_point_bytes, haversine_m, Projection};
use rayon::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::time::Instant;

type R<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Total-ordered f32 key for the priority queue.
#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct Key(f32, u64);
impl Eq for Key {}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
    }
}

fn tile_name(lat: i32, lon: i32) -> String {
    let ns = if lat >= 0 { 'N' } else { 'S' };
    let ew = if lon >= 0 { 'E' } else { 'W' };
    format!("Copernicus_DSM_COG_30_{ns}{:02}_00_{ew}{:03}_00_DEM.tif", lat.abs(), lon.abs())
}

/// Minimum elevation of the tile's pixels per output cell (cpd x cpd block). None when the tile is absent (sea).
fn tile_block(dir: &str, lat: i32, lon: i32, cpd: usize) -> R<Option<(Vec<f32>, String)>> {
    use tiff::decoder::{Decoder, DecodingResult};
    use tiff::tags::Tag;
    let path = format!("{dir}/{}", tile_name(lat, lon));
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let sha = format!("{:x}", Sha256::digest(&bytes));
    let mut d = Decoder::new(std::io::Cursor::new(&bytes))?;
    let (w, h) = d.dimensions()?;
    let scale = d.get_tag_f64_vec(Tag::ModelPixelScaleTag)?;
    let tie = d.get_tag_f64_vec(Tag::ModelTiepointTag)?;
    let data = match d.read_image()? {
        DecodingResult::F32(v) => v,
        _ => return Err(format!("{path}: expected float32 samples").into()),
    };
    let (sx, sy, x0, y0) = (scale[0], scale[1], tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1]);
    let mut block = vec![f32::INFINITY; cpd * cpd];
    for r in 0..h as usize {
        let plat = y0 - (r as f64 + 0.5) * sy;
        let br = (((lat as f64 + 1.0) - plat) * cpd as f64).floor();
        if !(0.0..cpd as f64).contains(&br) {
            continue;
        }
        for c in 0..w as usize {
            let plon = x0 + (c as f64 + 0.5) * sx;
            let bc = ((plon - lon as f64) * cpd as f64).floor();
            if !(0.0..cpd as f64).contains(&bc) {
                continue;
            }
            let v = data[r * w as usize + c];
            if v.is_finite() && v > -1000.0 {
                let k = br as usize * cpd + bc as usize;
                block[k] = block[k].min(v);
            }
        }
    }
    Ok(Some((block, sha)))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 8 {
        return Err("rw_gauge_proof_fabric TILE_DIR MINLON MINLAT MAXLON MAXLAT CELLS_PER_DEGREE CHANNEL_KM2 OUT_GPKG".into());
    }
    let (minlon, minlat, maxlon, maxlat): (i32, i32, i32, i32) = (a[1].parse()?, a[2].parse()?, a[3].parse()?, a[4].parse()?);
    let cpd: usize = a[5].parse()?;
    let channel_km2: f64 = a[6].parse()?;
    let out = &a[7];
    let (nc, nr) = ((maxlon - minlon) as usize * cpd, (maxlat - minlat) as usize * cpd);
    let mut stages = Vec::new();

    let t = Instant::now();
    let tiles: Vec<(i32, i32)> = (minlat..maxlat).flat_map(|la| (minlon..maxlon).map(move |lo| (la, lo))).collect();
    let blocks: R<Vec<_>> = tiles.par_iter().map(|&(la, lo)| tile_block(&a[0], la, lo, cpd).map(|b| (la, lo, b))).collect();
    let blocks = blocks?;
    let mut z = vec![f32::NAN; nr * nc];
    let mut used = BTreeMap::new();
    for (la, lo, b) in &blocks {
        let Some((block, sha)) = b else { continue };
        used.insert(tile_name(*la, *lo), sha.clone());
        let r0 = (maxlat - la - 1) as usize * cpd;
        let c0 = (lo - minlon) as usize * cpd;
        for br in 0..cpd {
            for bc in 0..cpd {
                let v = block[br * cpd + bc];
                z[(r0 + br) * nc + c0 + bc] = if v.is_finite() { v } else { f32::NAN };
            }
        }
    }
    stages.push(json!({"stage":"decode_tiles_min_aggregate","wall_ms":t.elapsed().as_secs_f64()*1e3,"parallelism":format!("Rayon {} workers across {} tiles",rayon::current_num_threads(),tiles.len())}));

    // Priority-Flood with FIFO tie order: downstream = the cell that reached this one.
    let t = Instant::now();
    let n = nr * nc;
    let mut ds = vec![u32::MAX; n];
    let mut closed = vec![false; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut heap = BinaryHeap::new();
    let mut tick = 0u64;
    let nb = |i: usize| -> [Option<usize>; 8] {
        let (r, c) = ((i / nc) as i64, (i % nc) as i64);
        let mut o = [None; 8];
        for (k, (dr, dc)) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)].iter().enumerate() {
            let (rr, cc) = (r + dr, c + dc);
            if rr >= 0 && cc >= 0 && rr < nr as i64 && cc < nc as i64 {
                o[k] = Some(rr as usize * nc + cc as usize);
            }
        }
        o
    };
    for i in 0..n {
        if z[i].is_nan() {
            closed[i] = true;
            continue;
        }
        let edge = nb(i).iter().any(|x| x.is_none_or(|j| z[j].is_nan()));
        if edge {
            closed[i] = true;
            heap.push(Reverse((Key(z[i], tick), i as u32)));
            tick += 1;
        }
    }
    while let Some(Reverse((Key(e, _), c))) = heap.pop() {
        order.push(c);
        for j in nb(c as usize).into_iter().flatten() {
            if closed[j] {
                continue;
            }
            closed[j] = true;
            ds[j] = c;
            heap.push(Reverse((Key(z[j].max(e), tick), j as u32)));
            tick += 1;
        }
    }
    stages.push(json!({"stage":"priority_flood","wall_ms":t.elapsed().as_secs_f64()*1e3,"parallelism":"1 thread (priority queue; fixture builder, startup only)","cells":n}));

    let t = Instant::now();
    let r_earth = 6_371.0072_f64; // authalic radius, km
    let dl = (1.0 / cpd as f64).to_radians();
    let row_area: Vec<f64> = (0..nr)
        .map(|r| {
            let top = (maxlat as f64 - r as f64 / cpd as f64).to_radians();
            let bot = (maxlat as f64 - (r + 1) as f64 / cpd as f64).to_radians();
            r_earth * r_earth * dl * (top.sin() - bot.sin())
        })
        .collect();
    let mut acc = vec![0.0f64; n];
    for &c in order.iter().rev() {
        let c = c as usize;
        acc[c] += row_area[c / nc];
        if ds[c] != u32::MAX {
            acc[ds[c] as usize] += acc[c];
        }
    }
    stages.push(json!({"stage":"accumulate","wall_ms":t.elapsed().as_secs_f64()*1e3,"parallelism":"1 thread (reverse topological pass)"}));

    // Reaches between junctions.
    let t = Instant::now();
    let channel: Vec<bool> = acc.iter().map(|&x| x >= channel_km2).collect();
    let mut donors = vec![0u8; n];
    for i in 0..n {
        if channel[i] && ds[i] != u32::MAX && channel[ds[i] as usize] {
            donors[ds[i] as usize] += 1;
        }
    }
    let start: Vec<bool> = (0..n).map(|i| channel[i] && donors[i] != 1).collect();
    let starts: Vec<usize> = (0..n).filter(|&i| start[i]).collect();
    let ns_ = 9_i64; // gauge-proof fixture id namespace (contract section 5); never a product namespace
    let id_of = |i: usize| ns_ * 10_000_000_000 + i as i64 + 1;
    let center = |i: usize| ((minlon as f64) + ((i % nc) as f64 + 0.5) / cpd as f64, (maxlat as f64) - ((i / nc) as f64 + 0.5) / cpd as f64);
    let (lat0, lon0) = ((minlat + maxlat) as f64 / 2.0, (minlon + maxlon) as f64 / 2.0);
    let proj = Projection::Laea { a: 6_378_137.0, e: { let f = 1.0 / 298.257_223_563_f64; (2.0 * f - f * f).sqrt() }, lat0, lon0, fe: 0.0, fn_: 0.0 };
    struct Rch {
        start: usize,
        cells: Vec<usize>,
        next: Option<usize>,
    }
    let reaches: Vec<Rch> = starts
        .par_iter()
        .map(|&s| {
            let mut cells = vec![s];
            let mut cur = s;
            let next = loop {
                let d = ds[cur];
                if d == u32::MAX {
                    break None;
                }
                let d = d as usize;
                if start[d] {
                    break Some(d);
                }
                cells.push(d);
                cur = d;
            };
            Rch { start: s, cells, next }
        })
        .collect();
    // Upstream reaches into each junction start cell, for local area and Strahler order.
    let mut inflow: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, r) in reaches.iter().enumerate() {
        if let Some(j) = r.next {
            inflow.entry(j).or_default().push(k);
        }
    }
    let by_start: BTreeMap<usize, usize> = reaches.iter().enumerate().map(|(k, r)| (r.start, k)).collect();
    let tot: Vec<f64> = reaches.iter().map(|r| acc[*r.cells.last().unwrap()]).collect();
    let mut topo: Vec<usize> = (0..reaches.len()).collect();
    topo.sort_by(|&x, &y| tot[x].total_cmp(&tot[y]).then(reaches[x].start.cmp(&reaches[y].start)));
    let mut strahler = vec![1.0f64; reaches.len()];
    for &k in &topo {
        if let Some(up) = inflow.get(&reaches[k].start) {
            let m = up.iter().map(|&u| strahler[u]).fold(0.0, f64::max);
            let count = up.iter().filter(|&&u| strahler[u] == m).count();
            strahler[k] = if count >= 2 { m + 1.0 } else { m.max(1.0) };
        }
    }
    let rank: BTreeMap<usize, i64> = topo.iter().enumerate().map(|(i, &k)| (k, i as i64 + 1)).collect();
    let srs_id = 100_001_i64;
    let mut flowpaths = Vec::new();
    let mut network = Vec::new();
    let mut nexus = Vec::new();
    for (k, r) in reaches.iter().enumerate() {
        let n_id = id_of(r.start);
        let mut pts: Vec<(f64, f64)> = r.cells.iter().map(|&c| center(c)).collect();
        if let Some(j) = r.next {
            pts.push(center(j));
        }
        let length_km: f64 = pts.windows(2).map(|w| haversine_m(w[0].0, w[0].1, w[1].0, w[1].1)).sum::<f64>() / 1000.0;
        let line: Vec<[f64; 2]> = pts.iter().map(|&(lo, la)| proj.forward(lo, la)).collect();
        let upstream: f64 = inflow.get(&r.start).map_or(0.0, |v| v.iter().map(|&u| tot[u]).sum());
        let local = tot[k] - upstream;
        let toid = match r.next {
            Some(j) => format!("nex-{}", id_of(reaches[by_start[&j]].start)),
            None => format!("tnx-{n_id}"),
        };
        if r.next.is_none() {
            let (lo, la) = center(*r.cells.last().unwrap());
            nexus.push(Nexus { fid: 0, geom: Some(hydro_world::contract::Geometry(gpkg_point_bytes(srs_id as i32, proj.forward(lo, la)))), id: Some(toid.clone()), toid: Some("wb-0".into()), hl_id: None, hl_uri: None, kind: Some("terminal".into()) });
        }
        flowpaths.push(Flowpath {
            fid: 0,
            geom: Some(hydro_world::contract::Geometry(gpkg_line_bytes(srs_id as i32, &line))),
            id: Some(format!("wb-{n_id}")),
            toid: Some(toid.clone()),
            mainstem: None,
            order: Some(strahler[k]),
            hydroseq: Some(rank[&k]),
            lengthkm: Some(length_km),
            areasqkm: Some(local),
            tot_drainage_areasqkm: Some(tot[k]),
            has_divide: Some(false),
            divide_id: None,
        });
        network.push(NetworkRow { fid: 0, id: Some(format!("wb-{n_id}")), toid: Some(toid), divide_id: None, ds_id: None, mainstem: None, hl_id: None, hydroseq: Some(rank[&k]), hl_uri: None, hf_source: Some("copernicus-glo90-d8-gauge-proof".into()), hf_id: Some(r.start as f64), lengthkm: Some(length_km), areasqkm: Some(local), tot_drainage_areasqkm: Some(tot[k]), kind: Some("flowpath".into()), vpu: None });
    }
    for (&j, _) in &inflow {
        let (lo, la) = center(j);
        nexus.push(Nexus { fid: 0, geom: Some(hydro_world::contract::Geometry(gpkg_point_bytes(srs_id as i32, proj.forward(lo, la)))), id: Some(format!("nex-{}", id_of(j))), toid: Some(format!("wb-{}", id_of(j))), hl_id: None, hl_uri: None, kind: Some("nexus".into()) });
    }
    stages.push(json!({"stage":"reaches_topology","wall_ms":t.elapsed().as_secs_f64()*1e3,"parallelism":format!("Rayon {} workers (reach walks); ordering 1 thread",rayon::current_num_threads())}));

    let t = Instant::now();
    let wkt = Projection::laea_wkt2(&format!("ARWEN LAEA {lat0} {lon0}"), lat0, lon0);
    let meta = |k: &str, v: String| MetadataRow { fid: 0, key: Some(k.into()), value: Some(v) };
    let tiles_sha = format!("{:x}", Sha256::digest(serde_json::to_vec(&used)?));
    let mut f = Fabric {
        contract: CONTRACT.into(),
        srs: Srs { srs_name: format!("ARWEN LAEA {lat0} {lon0}"), srs_id, organization: "ARWEN".into(), organization_coordsys_id: srs_id, definition: wkt, description: Some("Lambert azimuthal equal area on WGS 84, centred on the proof domain".into()) },
        contents: BTreeMap::<String, Contents>::new(),
        divides: vec![],
        flowpaths,
        nexus,
        network,
        flowpath_attributes: vec![],
        hydrolocations: vec![],
        lakes: vec![],
        model_attributes: None,
        fabric_metadata: Some(vec![
            meta("contract", CONTRACT.into()),
            meta("builder", "rw_gauge_proof_fabric (gauge-matching fixture, not a product fabric)".into()),
            meta("source_row", "copernicus-glo90-d8-gauge-proof".into()),
            meta("data_terms", "Copernicus DEM GLO-90: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved (licence verdict: docs/public/HYDRO-WORLD-LICENCES.md)".into()),
            meta("id_namespace", ns_.to_string()),
            meta("horizontal_crs", "laea-domain-centroid".into()),
            meta("domain", format!("{minlon},{minlat},{maxlon},{maxlat}; {cpd} cells per degree; channel >= {channel_km2} km2")),
            meta("dem_tiles_sha256", tiles_sha.clone()),
        ]),
    };
    hydro_world::derive::canonicalize(&mut f);
    if let Some(m) = f.fabric_metadata.as_mut() {
        for (i, r) in m.iter_mut().enumerate() {
            r.fid = i as i64 + 1;
        }
    }
    let _ = std::fs::remove_file(out);
    hydro_world::gpkg::write(&f, out)?;
    stages.push(json!({"stage":"write_gpkg","wall_ms":t.elapsed().as_secs_f64()*1e3,"parallelism":"1 thread (SQLite writer)"}));
    let receipt = json!({
        "schema":"nextgen-gauge-proof-fabric-v1",
        "scope":"gauge-matching fixture; not a product fabric",
        "grid":{"cells":n,"rows":nr,"cols":nc,"cells_per_degree":cpd,"aggregation":"minimum of GLO-90 pixels per cell"},
        "channel_threshold_km2":channel_km2,
        "flowpaths":f.flowpaths.len(),"nexus":f.nexus.len(),
        "tiles_used":used,"tiles_missing":tiles.len()-used.len(),"tiles_sha256_of_list":tiles_sha,
        "max_drainage_area_km2": acc.iter().cloned().fold(0.0, f64::max),
        "stages":stages,
        "out":out,
    });
    std::fs::write(format!("{out}.receipt.json"), serde_json::to_vec_pretty(&receipt)?)?;
    println!("{} flowpaths, {} nexus -> {out}", f.flowpaths.len(), f.nexus.len());
    Ok(())
}
