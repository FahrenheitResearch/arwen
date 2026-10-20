//! Label raster -> one polygon per label along exact cell boundaries.
//! Boundary edges are collected in parallel over rows, sorted by label, and
//! each label's rings are traced in its own task. Rings run with the label on
//! their left (counter-clockwise exteriors, clockwise holes, north up). At a
//! vertex where two same-label cells touch only diagonally the trace turns
//! right, so a label is traced 8-connected: a D8 catchment is always
//! 8-connected (every cell reaches its reach through D8 steps), so it is one
//! exterior ring that may touch itself at such a corner, and every cell of the
//! catchment is inside its polygon. A label with several 8-connected parts
//! (not produced by D8 labelling) keeps its largest part; the other parts'
//! cells are reported as slivers.
use crate::grid::Grid;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LabelPolygon {
    /// Exterior ring in grid corner coordinates (ci, cj), not closed.
    pub exterior: Vec<(u32, u32)>,
    pub holes: Vec<Vec<(u32, u32)>>,
    /// Cells in parts other than the kept one.
    pub dropped_cells: u64,
    pub parts: usize,
}

// Directions in corner space, counter-clockwise: E, N, W, S. Row index grows south.
const STEP: [(i64, i64); 4] = [(1, 0), (0, -1), (-1, 0), (0, 1)];

/// Doubled signed area in corner space with north up (positive = CCW).
fn area2(r: &[(u32, u32)]) -> i64 {
    let n = r.len();
    let mut s = 0i64;
    for i in 0..n {
        let (x1, y1) = (r[i].0 as i64, -(r[i].1 as i64));
        let (x2, y2) = (r[(i + 1) % n].0 as i64, -(r[(i + 1) % n].1 as i64));
        s += x1 * y2 - x2 * y1;
    }
    s
}

fn inside(pt: (f64, f64), ring: &[(u32, u32)]) -> bool {
    let mut c = false;
    let n = ring.len();
    for i in 0..n {
        let (xi, yi) = (ring[i].0 as f64, ring[i].1 as f64);
        let (xj, yj) = (ring[(i + n - 1) % n].0 as f64, ring[(i + n - 1) % n].1 as f64);
        if (yi > pt.1) != (yj > pt.1) && pt.0 < (xj - xi) * (pt.1 - yi) / (yj - yi) + xi {
            c = !c;
        }
    }
    c
}

/// Polygons for labels 1..=nlabels (index = label - 1); None for an empty label.
pub fn polygonize(g: &Grid, labels: &[u32], nlabels: usize) -> Vec<Option<LabelPolygon>> {
    let nx = g.nx;
    let ny = g.ny;
    let cw = (nx + 1) as u64;
    // (label, start corner, direction)
    let mut edges: Vec<(u32, u64, u8)> = (0..ny)
        .into_par_iter()
        .flat_map_iter(|r| {
            let mut v = Vec::new();
            for c in 0..nx {
                let l = labels[r * nx + c];
                if l == 0 {
                    continue;
                }
                let at = |rr: isize, cc: isize| -> u32 {
                    if rr < 0 || cc < 0 || rr as usize >= ny || cc as usize >= nx { 0 } else { labels[rr as usize * nx + cc as usize] }
                };
                let (ri, ci) = (r as isize, c as isize);
                let corner = |x: usize, y: usize| y as u64 * cw + x as u64;
                // east side: SE -> NE (north)
                if at(ri, ci + 1) != l {
                    v.push((l, corner(c + 1, r + 1), 1u8));
                }
                // north side: NE -> NW (west)
                if at(ri - 1, ci) != l {
                    v.push((l, corner(c + 1, r), 2u8));
                }
                // west side: NW -> SW (south)
                if at(ri, ci - 1) != l {
                    v.push((l, corner(c, r), 3u8));
                }
                // south side: SW -> SE (east)
                if at(ri + 1, ci) != l {
                    v.push((l, corner(c, r + 1), 0u8));
                }
            }
            v
        })
        .collect();
    edges.par_sort_unstable();
    let mut bounds = vec![(0usize, 0usize); nlabels];
    let mut i = 0;
    while i < edges.len() {
        let l = edges[i].0;
        let mut j = i;
        while j < edges.len() && edges[j].0 == l {
            j += 1;
        }
        if (l as usize) <= nlabels {
            bounds[l as usize - 1] = (i, j);
        }
        i = j;
    }
    // Cell counts per label for sliver accounting.
    let counts: Vec<u64> = {
        let parts: Vec<HashMap<u32, u64>> = labels
            .par_chunks(nx * 64)
            .map(|ch| {
                let mut m = HashMap::new();
                for &l in ch {
                    if l > 0 {
                        *m.entry(l).or_insert(0) += 1;
                    }
                }
                m
            })
            .collect();
        let mut v = vec![0u64; nlabels];
        for m in parts {
            for (l, n) in m {
                if (l as usize) <= nlabels {
                    v[l as usize - 1] += n;
                }
            }
        }
        v
    };
    bounds
        .par_iter()
        .enumerate()
        .map(|(k, &(a, b))| {
            if a == b {
                return None;
            }
            let es = &edges[a..b];
            let mut out: HashMap<u64, [u32; 4]> = HashMap::with_capacity(es.len());
            for (idx, e) in es.iter().enumerate() {
                out.entry(e.1).or_insert([u32::MAX; 4])[e.2 as usize] = idx as u32;
            }
            let mut used = vec![false; es.len()];
            let mut rings: Vec<Vec<(u32, u32)>> = Vec::new();
            for s in 0..es.len() {
                if used[s] {
                    continue;
                }
                let mut ring = Vec::new();
                let mut cur = s;
                let mut prev_dir = 255u8;
                loop {
                    used[cur] = true;
                    let (_, v, d) = es[cur];
                    if d != prev_dir {
                        ring.push(((v % cw) as u32, (v / cw) as u32));
                    }
                    prev_dir = d;
                    let (dx, dy) = STEP[d as usize];
                    let x = (v % cw) as i64 + dx;
                    let y = (v / cw) as i64 + dy;
                    let nv = y as u64 * cw + x as u64;
                    let slots = out.get(&nv).copied().unwrap_or([u32::MAX; 4]);
                    // Prefer a right turn, then straight, then left (8-connected label).
                    let order = [(d + 3) % 4, d, (d + 1) % 4];
                    let next = order.iter().map(|&nd| slots[nd as usize]).find(|&e| e != u32::MAX && !used[e as usize]);
                    match next {
                        Some(e) => cur = e as usize,
                        None => break,
                    }
                }
                // Drop a closing vertex that continues the first edge's direction.
                if ring.len() > 2 {
                    let first_dir = es[s].2;
                    if prev_dir == first_dir {
                        ring.remove(0);
                    }
                }
                rings.push(ring);
            }
            let mut ext: Vec<(Vec<(u32, u32)>, i64)> = Vec::new();
            let mut holes: Vec<Vec<(u32, u32)>> = Vec::new();
            for r in rings {
                let a2 = area2(&r);
                if a2 > 0 { ext.push((r, a2)) } else if a2 < 0 { holes.push(r) }
            }
            if ext.is_empty() {
                return None;
            }
            let parts = ext.len();
            ext.sort_by(|x, y| y.1.cmp(&x.1));
            let (keep, keep_a2) = ext.swap_remove(0);
            let kept_holes: Vec<Vec<(u32, u32)>> = holes
                .into_iter()
                .filter(|h| {
                    // Centre of the label cell on the left of the hole's first
                    // edge: inside the kept exterior iff the hole belongs to it.
                    let (dx, dy) = ((h[1].0 as f64 - h[0].0 as f64).signum(), (h[1].1 as f64 - h[0].1 as f64).signum());
                    let p = (h[0].0 as f64 + 0.5 * dx + 0.5 * dy, h[0].1 as f64 + 0.5 * dy - 0.5 * dx);
                    inside(p, &keep)
                })
                .collect();
            let hole_cells: i64 = kept_holes.iter().map(|h| -area2(h)).sum::<i64>() / 2;
            let kept_cells = (keep_a2 / 2 - hole_cells).max(0) as u64;
            Some(LabelPolygon { exterior: keep, holes: kept_holes, dropped_cells: counts[k].saturating_sub(kept_cells), parts })
        })
        .collect()
}
