//! Open global hydrography from a DEM, all cores:
//! 1. sea mask (cells at the dataset's sea value connected to the domain edge
//!    or to missing tiles);
//! 2. depression filling by tiled parallel Priority-Flood (Barnes 2016,
//!    "Parallel Priority-Flood depression filling for trillion cell digital
//!    elevation models", Computers & Geosciences 96): every tile is flooded
//!    independently from its perimeter with watershed labels, the label
//!    spill graph is solved once, and each cell is raised to
//!    max(tile fill, label spill level);
//! 3. D8 steepest descent on the filled surface;
//! 4. flats drained towards lower and away from higher terrain (Barnes,
//!    Lehman and Mulla 2014, Computers & Geosciences 62), one flat per task;
//! 5. flow accumulation of exact ellipsoidal cell areas (integer units, so
//!    the parallel sum is deterministic) by level-synchronous Kahn order;
//! 6. stream cells, reaches split at confluences and at lake outlets, and the
//!    catchment of every reach by pointer jumping down the D8 tree.
use crate::grid::Grid;
use crate::Result;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering as AO};

/// D8 codes 0..7: E, NE, N, NW, W, SW, S, SE (rows run south).
pub const DR: [isize; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
pub const DC: [isize; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
/// Drains out of the domain (box edge) or into the sea.
pub const OUT: u8 = 8;
/// Sea cell: not land, never part of a catchment.
pub const SEA: u8 = 9;
/// Not yet drained (only inside flat resolution).
pub const UNDRAINED: u8 = 255;
/// Area unit: 0.01 m2.
pub const AREA_UNITS_PER_M2: f64 = 100.0;

#[derive(Clone, Copy)]
pub struct Shared<T>(*mut T);
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}
impl<T> Shared<T> {
    pub fn new(v: &mut [T]) -> Self {
        Shared(v.as_mut_ptr())
    }
    /// # Safety
    /// Callers write each index from at most one task.
    pub unsafe fn set(&self, i: usize, x: T) {
        unsafe { *self.0.add(i) = x }
    }
}

#[inline]
pub fn neighbour(g: &Grid, i: usize, k: usize) -> Option<usize> {
    let (r, c) = ((i / g.nx) as isize, (i % g.nx) as isize);
    let (nr, nc) = (r + DR[k], c + DC[k]);
    (nr >= 0 && nc >= 0 && (nr as usize) < g.ny && (nc as usize) < g.nx).then(|| nr as usize * g.nx + nc as usize)
}

#[inline]
pub fn on_border(g: &Grid, i: usize) -> bool {
    let (r, c) = (i / g.nx, i % g.nx);
    r == 0 || c == 0 || r + 1 == g.ny || c + 1 == g.nx
}

#[inline]
pub fn receiver(g: &Grid, dir: &[u8], i: usize) -> Option<usize> {
    let d = dir[i];
    if d < 8 { neighbour(g, i, d as usize) } else { None }
}

/// Sea: cells equal to `sea_value` (or without a tile) 4-connected to a
/// box-edge sea cell or to a missing tile.
pub fn sea_mask(g: &Grid, z: &[f32], missing: &[bool], sea_value: Option<f32>) -> Vec<bool> {
    let is_sea_value = |i: usize| missing[i] || sea_value.is_some_and(|v| z[i] == v);
    let mut sea = vec![false; g.len()];
    let mut q: VecDeque<usize> = (0..g.len()).into_par_iter().filter(|&i| missing[i] || (on_border(g, i) && is_sea_value(i))).collect::<Vec<_>>().into();
    for &i in &q {
        sea[i] = true;
    }
    while let Some(i) = q.pop_front() {
        for k in [0, 2, 4, 6] {
            if let Some(n) = neighbour(g, i, k) {
                if !sea[n] && is_sea_value(n) {
                    sea[n] = true;
                    q.push_back(n);
                }
            }
        }
    }
    sea
}

#[derive(Clone, Copy, PartialEq)]
struct Item(f32, u32);
impl Eq for Item {}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        // Min-heap on elevation, then index (deterministic ties).
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[derive(Debug, Default, serde::Serialize)]
pub struct FillStats {
    pub tiles: usize,
    pub tile_size: usize,
    pub labels: usize,
    pub graph_edges: usize,
    pub raised_cells: usize,
}

const NOLAB: u32 = u32::MAX;
const OUTSIDE: u32 = 0;

/// Tiled parallel Priority-Flood. Returns the filled surface (sea cells keep
/// their value). Box-edge and sea cells are the outlets.
pub fn fill(g: &Grid, z: &[f32], sea: &[bool], ts: usize) -> Result<(Vec<f32>, FillStats)> {
    let (tny, tnx) = (g.ny.div_ceil(ts), g.nx.div_ceil(ts));
    let ntiles = tny * tnx;
    let per_max = 4 * ts;
    let mut f = z.to_vec();
    let mut lab = vec![NOLAB; g.len()];
    let fs = Shared::new(&mut f);
    let ls = Shared::new(&mut lab);
    let tile_edges: Vec<Vec<(u32, u32, f32)>> = (0..ntiles)
        .into_par_iter()
        .map(|t| {
            let (r0, c0) = ((t / tnx) * ts, (t % tnx) * ts);
            let (th, tw) = (ts.min(g.ny - r0), ts.min(g.nx - c0));
            let n = th * tw;
            let mut lf = vec![0f32; n];
            let mut ll = vec![NOLAB; n];
            let mut heap = BinaryHeap::new();
            let mut pit = VecDeque::new();
            let base = 1 + (t * per_max) as u32;
            let mut k = 0u32;
            for lr in 0..th {
                for lc in 0..tw {
                    let li = lr * tw + lc;
                    let gi = (r0 + lr) * g.nx + c0 + lc;
                    lf[li] = z[gi];
                    let perim = lr == 0 || lc == 0 || lr + 1 == th || lc + 1 == tw;
                    if sea[gi] || on_border(g, gi) {
                        ll[li] = OUTSIDE;
                        heap.push(Item(z[gi], li as u32));
                    } else if perim {
                        ll[li] = base + k;
                        k += 1;
                        heap.push(Item(z[gi], li as u32));
                    }
                }
            }
            let mut edges: HashMap<(u32, u32), f32> = HashMap::new();
            loop {
                let c = match pit.pop_front() {
                    Some(c) => c,
                    None => match heap.pop() {
                        Some(Item(_, c)) => c as usize,
                        None => break,
                    },
                };
                let (lr, lc) = ((c / tw) as isize, (c % tw) as isize);
                let level = lf[c];
                let lcab = ll[c];
                for kk in 0..8 {
                    let (nr, nc) = (lr + DR[kk], lc + DC[kk]);
                    if nr < 0 || nc < 0 || nr as usize >= th || nc as usize >= tw {
                        continue;
                    }
                    let nn = nr as usize * tw + nc as usize;
                    if ll[nn] == NOLAB {
                        ll[nn] = lcab;
                        if lf[nn] <= level {
                            lf[nn] = level;
                            pit.push_back(nn);
                        } else {
                            heap.push(Item(lf[nn], nn as u32));
                        }
                    } else if ll[nn] != lcab {
                        let key = (lcab.min(ll[nn]), lcab.max(ll[nn]));
                        let w = level.max(lf[nn]);
                        edges.entry(key).and_modify(|x| *x = x.min(w)).or_insert(w);
                    }
                }
            }
            for lr in 0..th {
                for lc in 0..tw {
                    let li = lr * tw + lc;
                    let gi = (r0 + lr) * g.nx + c0 + lc;
                    unsafe {
                        fs.set(gi, lf[li]);
                        ls.set(gi, ll[li]);
                    }
                }
            }
            edges.into_iter().map(|((a, b), w)| (a, b, w)).collect()
        })
        .collect();
    // Seam edges between perimeter cells of different tiles.
    let tile_of = |i: usize| ((i / g.nx) / ts) * tnx + (i % g.nx) / ts;
    let seam: Vec<(u32, u32, f32)> = (0..g.ny)
        .into_par_iter()
        .flat_map_iter(|r| {
            let mut v = Vec::new();
            let row_seam = r % ts == 0 || r % ts == ts - 1;
            for c in 0..g.nx {
                if !(row_seam || c % ts == 0 || c % ts == ts - 1) {
                    continue;
                }
                let i = r * g.nx + c;
                for kk in 0..8 {
                    if let Some(n) = neighbour(g, i, kk) {
                        if n > i && tile_of(n) != tile_of(i) && lab[i] != lab[n] {
                            v.push((lab[i].min(lab[n]), lab[i].max(lab[n]), f[i].max(f[n])));
                        }
                    }
                }
            }
            v
        })
        .collect();
    let mut all: Vec<(u32, u32, f32)> = tile_edges.into_iter().flatten().chain(seam).collect();
    all.par_sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(a.2.total_cmp(&b.2)));
    all.dedup_by(|b, a| a.0 == b.0 && a.1 == b.1);
    let nlab = 1 + ntiles * per_max;
    let mut deg = vec![0u32; nlab + 1];
    for e in &all {
        deg[e.0 as usize + 1] += 1;
        deg[e.1 as usize + 1] += 1;
    }
    for i in 0..nlab {
        deg[i + 1] += deg[i];
    }
    let mut adj = vec![(0u32, 0f32); deg[nlab] as usize];
    let mut fillp = deg.clone();
    for e in &all {
        adj[fillp[e.0 as usize] as usize] = (e.1, e.2);
        fillp[e.0 as usize] += 1;
        adj[fillp[e.1 as usize] as usize] = (e.0, e.2);
        fillp[e.1 as usize] += 1;
    }
    let mut wl = vec![f32::INFINITY; nlab];
    wl[OUTSIDE as usize] = f32::NEG_INFINITY;
    let mut heap = BinaryHeap::new();
    heap.push(Item(f32::NEG_INFINITY, OUTSIDE));
    let mut done = vec![false; nlab];
    while let Some(Item(w, l)) = heap.pop() {
        let l = l as usize;
        if done[l] {
            continue;
        }
        done[l] = true;
        for &(m, ew) in &adj[deg[l] as usize..deg[l + 1] as usize] {
            let cand = w.max(ew);
            if cand < wl[m as usize] {
                wl[m as usize] = cand;
                heap.push(Item(cand, m));
            }
        }
    }
    let unreached = (0..g.len()).into_par_iter().filter(|&i| lab[i] != OUTSIDE && wl[lab[i] as usize] == f32::INFINITY).count();
    if unreached > 0 {
        return Err(format!("fill: {unreached} cells in labels with no spill path to an outlet").into());
    }
    f.par_iter_mut().zip(lab.par_iter()).for_each(|(v, &l)| {
        if l != OUTSIDE {
            *v = v.max(wl[l as usize]);
        }
    });
    let raised = f.par_iter().zip(z.par_iter()).filter(|(a, b)| a != b).count();
    let used = done.iter().filter(|x| **x).count();
    Ok((f, FillStats { tiles: ntiles, tile_size: ts, labels: used, graph_edges: all.len(), raised_cells: raised }))
}

fn dists(g: &Grid, row: usize) -> [f64; 8] {
    let (dx, dy) = g.spacing_m(row);
    let dg = (dx * dx + dy * dy).sqrt();
    [dx, dg, dy, dg, dx, dg, dy, dg]
}

#[derive(Debug, Default, serde::Serialize)]
pub struct FlatStats {
    pub undrained_cells: usize,
    pub flats: usize,
    pub largest_flat_cells: usize,
}

/// D8 on the filled surface plus flat resolution. Sea cells get SEA; cells
/// that drain into the sea or off the box edge get OUT.
pub fn directions(g: &Grid, f: &[f32], sea: &[bool]) -> Result<(Vec<u8>, FlatStats)> {
    let mut dir = vec![UNDRAINED; g.len()];
    dir.par_chunks_mut(g.nx).enumerate().for_each(|(r, row)| {
        let d = dists(g, r);
        for (c, out) in row.iter_mut().enumerate() {
            let i = r * g.nx + c;
            if sea[i] {
                *out = SEA;
                continue;
            }
            let mut best = None;
            let mut bs = 0.0;
            let mut touches_sea = false;
            for k in 0..8 {
                if let Some(n) = neighbour(g, i, k) {
                    touches_sea |= sea[n];
                    let drop = (f[i] - f[n]) as f64;
                    if drop > 0.0 {
                        let s = drop / d[k];
                        if s > bs {
                            bs = s;
                            best = Some(k);
                        }
                    }
                }
            }
            *out = match best {
                Some(k) if !sea[neighbour(g, i, k).unwrap()] => k as u8,
                Some(_) => OUT,
                None if on_border(g, i) || touches_sea => OUT,
                None => UNDRAINED,
            };
        }
    });
    // Flats: connected (8-neighbour, equal filled elevation) sets of undrained
    // cells. Labelling is one linear pass (index arrays, no hashing); the
    // per-flat gradients and directions then run one flat per task.
    let und: Vec<u32> = (0..g.len() as u32).into_par_iter().filter(|&i| dir[i as usize] == UNDRAINED).collect();
    let mut flat_id = vec![u32::MAX; g.len()];
    let mut pos = vec![0u32; g.len()];
    let mut flats: Vec<Vec<u32>> = Vec::new();
    for &s in &und {
        if flat_id[s as usize] != u32::MAX {
            continue;
        }
        let id = flats.len() as u32;
        let mut members = vec![s];
        flat_id[s as usize] = id;
        let mut h = 0;
        while h < members.len() {
            let c = members[h] as usize;
            h += 1;
            for k in 0..8 {
                if let Some(n) = neighbour(g, c, k) {
                    if dir[n] == UNDRAINED && f[n] == f[c] && flat_id[n] == u32::MAX {
                        flat_id[n] = id;
                        pos[n] = members.len() as u32;
                        members.push(n as u32);
                    }
                }
            }
        }
        flats.push(members);
    }
    let stats = FlatStats { undrained_cells: und.len(), flats: flats.len(), largest_flat_cells: flats.iter().map(|m| m.len()).max().unwrap_or(0) };
    let d0 = &dir;
    let resolved: Vec<Result<Vec<(u32, u8)>>> = flats
        .par_iter()
        .enumerate()
        .map(|(id, members)| {
            let id = id as u32;
            let inflat = |nb: usize| flat_id[nb] == id;
            let elev = f[members[0] as usize];
            let n = members.len();
            let mut dl = vec![u32::MAX; n];
            let mut q = VecDeque::new();
            for (j, &c) in members.iter().enumerate() {
                let c = c as usize;
                if (0..8).any(|k| neighbour(g, c, k).is_some_and(|nb| f[nb] == elev && d0[nb] != UNDRAINED)) {
                    dl[j] = 1;
                    q.push_back(j);
                }
            }
            while let Some(j) = q.pop_front() {
                let c = members[j] as usize;
                for k in 0..8 {
                    if let Some(nb) = neighbour(g, c, k) {
                        if inflat(nb) {
                            let m = pos[nb] as usize;
                            if dl[m] == u32::MAX {
                                dl[m] = dl[j] + 1;
                                q.push_back(m);
                            }
                        }
                    }
                }
            }
            let mut dh = vec![u32::MAX; n];
            for (j, &c) in members.iter().enumerate() {
                let c = c as usize;
                if (0..8).any(|k| neighbour(g, c, k).is_some_and(|nb| f[nb] > elev)) {
                    dh[j] = 1;
                    q.push_back(j);
                }
            }
            while let Some(j) = q.pop_front() {
                let c = members[j] as usize;
                for k in 0..8 {
                    if let Some(nb) = neighbour(g, c, k) {
                        if inflat(nb) {
                            let m = pos[nb] as usize;
                            if dh[m] == u32::MAX {
                                dh[m] = dh[j] + 1;
                                q.push_back(m);
                            }
                        }
                    }
                }
            }
            let hmax = dh.iter().filter(|&&x| x != u32::MAX).max().copied().unwrap_or(0);
            let mask = |j: usize| -> u64 {
                let away = if dh[j] == u32::MAX { 0 } else { (hmax - dh[j]) as u64 };
                2 * dl[j] as u64 + away
            };
            let mut out = Vec::with_capacity(n);
            for (j, &c) in members.iter().enumerate() {
                let c = c as usize;
                if dl[j] == u32::MAX {
                    return Err(format!("flat {id} ({n} cells at {elev}) has no outlet at its own elevation").into());
                }
                let mj = mask(j);
                let mut best: Option<(u64, usize)> = None;
                for k in 0..8 {
                    let Some(nb) = neighbour(g, c, k) else { continue };
                    if f[nb] != elev {
                        continue;
                    }
                    let cand = if inflat(nb) {
                        let mm = mask(pos[nb] as usize);
                        (mm < mj).then_some(mm)
                    } else if d0[nb] != UNDRAINED {
                        Some(0)
                    } else {
                        None
                    };
                    if let Some(v) = cand {
                        if best.is_none_or(|(b, _)| v < b) {
                            best = Some((v, k));
                        }
                    }
                }
                let (_, k) = best.ok_or_else(|| format!("flat {id}: cell {c} has no lower mask neighbour"))?;
                let nb = neighbour(g, c, k).unwrap();
                out.push((c as u32, if d0[nb] == SEA { OUT } else { k as u8 }));
            }
            Ok(out)
        })
        .collect();
    for r in resolved {
        for (c, code) in r? {
            dir[c as usize] = code;
        }
    }
    Ok((dir, stats))
}

#[derive(Debug, Default, serde::Serialize)]
pub struct AccumStats {
    pub land_cells: usize,
    pub levels: usize,
    pub parallel_levels: usize,
}

/// Upstream area (0.01 m2 units) of every land cell. Errors on a cycle.
pub fn accumulate(g: &Grid, dir: &[u8]) -> Result<(Vec<u64>, AccumStats)> {
    let area: Vec<u64> = (0..g.ny).map(|r| (g.cell_area_m2(r) * AREA_UNITS_PER_M2).round() as u64).collect();
    let acc: Vec<AtomicU64> = (0..g.len()).into_par_iter().map(|i| AtomicU64::new(if dir[i] == SEA { 0 } else { area[i / g.nx] })).collect();
    let indeg: Vec<AtomicU8> = (0..g.len())
        .into_par_iter()
        .map(|i| {
            let mut n = 0u8;
            for k in 0..8 {
                if let Some(nb) = neighbour(g, i, k) {
                    if dir[nb] as usize == (k + 4) % 8 {
                        n += 1;
                    }
                }
            }
            AtomicU8::new(n)
        })
        .collect();
    let land = (0..g.len()).into_par_iter().filter(|&i| dir[i] != SEA).count();
    let mut frontier: Vec<u32> = (0..g.len() as u32).into_par_iter().filter(|&i| dir[i as usize] != SEA && indeg[i as usize].load(AO::Relaxed) == 0).collect();
    let mut processed = 0usize;
    let (mut levels, mut par_levels) = (0, 0);
    let step = |c: u32| -> Option<u32> {
        let c = c as usize;
        let r = receiver(g, dir, c)?;
        acc[r].fetch_add(acc[c].load(AO::Relaxed), AO::Relaxed);
        (indeg[r].fetch_sub(1, AO::AcqRel) == 1).then_some(r as u32)
    };
    while !frontier.is_empty() {
        processed += frontier.len();
        levels += 1;
        frontier = if frontier.len() > 4096 {
            par_levels += 1;
            frontier.par_iter().filter_map(|&c| step(c)).collect()
        } else {
            frontier.iter().filter_map(|&c| step(c)).collect()
        };
    }
    if processed != land {
        return Err(format!("flow directions contain a cycle: {} of {land} land cells never drained", land - processed).into());
    }
    Ok((acc.into_iter().map(|a| a.into_inner()).collect(), AccumStats { land_cells: land, levels, parallel_levels: par_levels }))
}

pub struct Reaches {
    /// Cells of each reach, upstream to downstream.
    pub cells: Vec<Vec<u32>>,
    /// Downstream reach index, None for terminal reaches.
    pub down: Vec<Option<u32>>,
    /// Reach index + 1 for stream cells, 0 elsewhere.
    pub cell_reach: Vec<u32>,
}

/// Stream cells (upstream area >= threshold) cut into reaches at confluences
/// and right after every cell in `break_after` (lake outlets). Reach order is
/// the raster order of the head cells, so ids are deterministic.
pub fn reaches(g: &Grid, dir: &[u8], acc: &[u64], thr: u64, break_after: &[bool]) -> Reaches {
    let stream = |i: usize| dir[i] != SEA && acc[i] >= thr;
    let ups = |i: usize| -> (u32, Option<usize>) {
        let mut n = 0;
        let mut one = None;
        for k in 0..8 {
            if let Some(nb) = neighbour(g, i, k) {
                if dir[nb] as usize == (k + 4) % 8 && stream(nb) {
                    n += 1;
                    one = Some(nb);
                }
            }
        }
        (n, one)
    };
    let is_head = |i: usize| -> bool {
        if !stream(i) {
            return false;
        }
        match ups(i) {
            (1, Some(u)) => break_after[u],
            _ => true,
        }
    };
    let heads: Vec<u32> = (0..g.len() as u32).into_par_iter().filter(|&i| is_head(i as usize)).collect();
    let walks: Vec<(Vec<u32>, Option<u32>)> = heads
        .par_iter()
        .map(|&h| {
            let mut cells = vec![h];
            let mut cur = h as usize;
            loop {
                if break_after[cur] {
                    break;
                }
                match receiver(g, dir, cur) {
                    Some(d) if !is_head(d) => {
                        cells.push(d as u32);
                        cur = d;
                    }
                    _ => break,
                }
            }
            let down = receiver(g, dir, cur).map(|d| d as u32);
            (cells, down)
        })
        .collect();
    let mut cell_reach = vec![0u32; g.len()];
    let s = Shared::new(&mut cell_reach);
    walks.par_iter().enumerate().for_each(|(k, (cells, _))| {
        for &c in cells {
            unsafe { s.set(c as usize, k as u32 + 1) }
        }
    });
    let down = walks.iter().map(|(_, d)| d.and_then(|d| (cell_reach[d as usize] > 0).then(|| cell_reach[d as usize] - 1))).collect();
    Reaches { cells: walks.into_iter().map(|(c, _)| c).collect(), down, cell_reach }
}

/// Catchment label (reach index + 1, 0 = drains out without meeting a
/// stream) of every cell by pointer jumping down the D8 tree.
pub fn catchments(g: &Grid, dir: &[u8], cell_reach: &[u32]) -> (Vec<u32>, usize) {
    let mut t: Vec<u32> = (0..g.len())
        .into_par_iter()
        .map(|i| if cell_reach[i] > 0 || dir[i] == SEA { i as u32 } else { receiver(g, dir, i).unwrap_or(i) as u32 })
        .collect();
    let mut rounds = 0;
    loop {
        rounds += 1;
        let next: Vec<u32> = t.par_iter().map(|&p| t[p as usize]).collect();
        let changed = next.par_iter().zip(t.par_iter()).any(|(a, b)| a != b);
        t = next;
        if !changed {
            break;
        }
    }
    (t.par_iter().map(|&p| cell_reach[p as usize]).collect(), rounds)
}

#[derive(Debug, Default, serde::Serialize)]
pub struct CollapseStats {
    pub reaches_before: usize,
    pub merged_into_downstream: usize,
    pub reaches_after: usize,
    pub shorter_than_min_kept_terminal_or_lake: usize,
}

/// Refactor step: a reach shorter than `min_len_m` with a downstream reach is
/// merged into it (its cells are prepended, its upstream reaches then join
/// at the merged reach's new head), in upstream-to-downstream order so chains
/// of short links collapse together. Reaches that end at a lake outlet
/// (`keep_end`) and terminal reaches are never merged.
pub fn collapse_short(g: &Grid, dir: &[u8], r: Reaches, min_len_m: f64, keep_end: &[bool]) -> (Reaches, CollapseStats) {
    let n = r.cells.len();
    let center = |i: usize| (g.lon(i % g.nx), g.lat(i / g.nx));
    let mut len: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|k| {
            let c = &r.cells[k];
            let mut pts: Vec<(f64, f64)> = c.iter().map(|&x| center(x as usize)).collect();
            if r.down[k].is_some() {
                if let Some(d) = receiver(g, dir, *c.last().unwrap() as usize) {
                    pts.push(center(d));
                }
            }
            crate::geo::polyline_geodesic_m(&pts)
        })
        .collect();
    let mut indeg = vec![0usize; n];
    for d in r.down.iter().flatten() {
        indeg[*d as usize] += 1;
    }
    let mut q: VecDeque<usize> = (0..n).filter(|&k| indeg[k] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(k) = q.pop_front() {
        order.push(k);
        if let Some(d) = r.down[k] {
            indeg[d as usize] -= 1;
            if indeg[d as usize] == 0 {
                q.push_back(d as usize);
            }
        }
    }
    let mut cells = r.cells;
    let mut alive = vec![true; n];
    let mut merged = 0;
    for &k in &order {
        let Some(d) = r.down[k] else { continue };
        let d = d as usize;
        if len[k] >= min_len_m || keep_end[*cells[k].last().unwrap() as usize] {
            continue;
        }
        let mut c = std::mem::take(&mut cells[k]);
        c.extend_from_slice(&cells[d]);
        cells[d] = c;
        len[d] += len[k];
        alive[k] = false;
        merged += 1;
    }
    let short_kept = (0..n).filter(|&k| alive[k] && len[k] < min_len_m).count();
    // Re-index survivors in raster order of their (new) head cells.
    let mut kept: Vec<usize> = (0..n).filter(|&k| alive[k]).collect();
    kept.sort_by_key(|&k| cells[k][0]);
    let mut cell_reach = vec![0u32; g.len()];
    let s = Shared::new(&mut cell_reach);
    kept.par_iter().enumerate().for_each(|(j, &k)| {
        for &c in &cells[k] {
            unsafe { s.set(c as usize, j as u32 + 1) }
        }
    });
    // Downstream of a survivor: the reach owning the cell its last cell drains to.
    let down: Vec<Option<u32>> = kept
        .iter()
        .map(|&k| {
            r.down[k]?;
            let last = *cells[k].last().unwrap() as usize;
            let d = receiver(g, dir, last)?;
            (cell_reach[d] > 0).then(|| cell_reach[d] - 1)
        })
        .collect();
    let stats = CollapseStats { reaches_before: n, merged_into_downstream: merged, reaches_after: kept.len(), shorter_than_min_kept_terminal_or_lake: short_kept };
    let new_cells = kept.into_iter().map(|k| std::mem::take(&mut cells[k])).collect();
    (Reaches { cells: new_cells, down, cell_reach }, stats)
}
