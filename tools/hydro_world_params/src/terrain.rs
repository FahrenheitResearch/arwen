//! Terrain attributes and the GIUH.
//!
//! - Slope and aspect: Horn (1981), "Hill shading and the reflectance map",
//!   Proc. IEEE 69(1):14-47, third-order finite difference on the 3x3 window,
//!   metric spacing per row on the lon/lat grid.
//! - Flow directions inside a catchment: Priority-Flood (Barnes, Lehman and
//!   Mulla, 2014, Computers & Geosciences 62:117-127) seeded at the catchment
//!   outlet (its lowest boundary cell), which fills depressions and gives every
//!   cell a downslope path to the outlet.
//! - GIUH: time-area method (Clark, 1945, Trans. ASCE 110:1419-1446; as a
//!   spatially distributed travel-time histogram in Maidment et al., 1996,
//!   Hydrol. Process. 10:831-844). Cell velocity V = k S^0.5 with k by land
//!   cover from USDA-NRCS (2010), National Engineering Handbook Part 630,
//!   ch. 15, Table 15-3 (also TR-55 Fig. 3-1); channel cells (contributing area
//!   at or above the channel-initiation area) use the "unpaved area; small
//!   upland gullies" row. The ordinates are the area fractions per CFE time step.
use crate::grid::Grid;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Horn slope (degrees) and downslope aspect (degrees clockwise from north;
/// NaN where flat or no data). Rayon over rows.
pub fn slope_aspect(g: &Grid, dem: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (nx, ny) = (g.nx, g.ny);
    let mut slope = vec![f32::NAN; g.n()];
    let mut aspect = vec![f32::NAN; g.n()];
    let dy = g.dy_m();
    slope.par_chunks_mut(nx).zip(aspect.par_chunks_mut(nx)).enumerate().for_each(|(j, (srow, arow))| {
        let dx = g.dx_m(j);
        let at = |jj: isize, ii: isize, c: f32| -> f32 {
            let jj = jj.clamp(0, ny as isize - 1) as usize;
            let ii = ii.clamp(0, nx as isize - 1) as usize;
            let v = dem[jj * nx + ii];
            if v.is_finite() { v } else { c }
        };
        for i in 0..nx {
            let e = dem[j * nx + i];
            if !e.is_finite() {
                continue;
            }
            let (ji, ii) = (j as isize, i as isize);
            let a = at(ji - 1, ii - 1, e) as f64;
            let b = at(ji - 1, ii, e) as f64;
            let c = at(ji - 1, ii + 1, e) as f64;
            let d = at(ji, ii - 1, e) as f64;
            let f = at(ji, ii + 1, e) as f64;
            let gg = at(ji + 1, ii - 1, e) as f64;
            let h = at(ji + 1, ii, e) as f64;
            let k = at(ji + 1, ii + 1, e) as f64;
            let dzdx = ((c + 2.0 * f + k) - (a + 2.0 * d + gg)) / (8.0 * dx); // east positive
            let dzdy = ((a + 2.0 * b + c) - (gg + 2.0 * h + k)) / (8.0 * dy); // north positive
            let s = (dzdx * dzdx + dzdy * dzdy).sqrt();
            srow[i] = s.atan().to_degrees() as f32;
            if s > 1e-6 {
                let az = (-dzdx).atan2(-dzdy).to_degrees();
                arow[i] = (if az < 0.0 { az + 360.0 } else { az }) as f32;
            }
        }
    });
    (slope, aspect)
}

/// tan(slope) of the DEM block-averaged to about `coarse_m` (the 1 km terrain
/// slope the NWM/NextGen `slope_1km` attribute is built on), painted back onto
/// the analysis cells. Rayon over block rows.
pub fn coarse_tan_slope(g: &Grid, dem: &[f32], coarse_m: f64) -> Vec<f32> {
    let fy = ((coarse_m / g.dy_m()).round() as usize).max(1);
    let lat_mid = g.lat(g.ny / 2);
    let fx = ((coarse_m / (g.dy_m() * lat_mid.to_radians().cos())).round() as usize).max(1);
    let (bx, by) = (g.nx.div_ceil(fx), g.ny.div_ceil(fy));
    let mut coarse = vec![f64::NAN; bx * by];
    coarse.par_chunks_mut(bx).enumerate().for_each(|(cj, row)| {
        for (ci, v) in row.iter_mut().enumerate() {
            let (mut s, mut n) = (0.0, 0usize);
            for j in cj * fy..((cj + 1) * fy).min(g.ny) {
                for i in ci * fx..((ci + 1) * fx).min(g.nx) {
                    let e = dem[j * g.nx + i];
                    if e.is_finite() {
                        s += e as f64;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                *v = s / n as f64;
            }
        }
    });
    let dyc = fy as f64 * g.dy_m();
    let mut tanc = vec![f32::NAN; bx * by];
    tanc.par_chunks_mut(bx).enumerate().for_each(|(cj, row)| {
        let j = (cj * fy + fy / 2).min(g.ny - 1);
        let dxc = fx as f64 * g.dx_m(j);
        for (ci, v) in row.iter_mut().enumerate() {
            let c0 = coarse[cj * bx + ci];
            if !c0.is_finite() {
                continue;
            }
            let get = |a: isize, b: isize| -> f64 {
                let a = a.clamp(0, by as isize - 1) as usize;
                let b = b.clamp(0, bx as isize - 1) as usize;
                let x = coarse[a * bx + b];
                if x.is_finite() { x } else { c0 }
            };
            let (a, b) = (cj as isize, ci as isize);
            let dzdx = (get(a, b + 1) - get(a, b - 1)) / (2.0 * dxc);
            let dzdy = (get(a - 1, b) - get(a + 1, b)) / (2.0 * dyc);
            *v = (dzdx * dzdx + dzdy * dzdy).sqrt() as f32;
        }
    });
    let mut out = vec![f32::NAN; g.n()];
    out.par_chunks_mut(g.nx).enumerate().for_each(|(j, row)| {
        let cj = j / fy;
        for (i, v) in row.iter_mut().enumerate() {
            *v = tanc[cj * bx + i / fx];
        }
    });
    out
}

/// sortable key of an f32 (total order, NaN last)
#[inline]
pub fn fkey(v: f32) -> u32 {
    if v.is_nan() {
        return u32::MAX;
    }
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

pub struct GiuhOut {
    pub ordinates: Vec<f64>,
    pub t_mean_s: f64,
    pub t_max_s: f64,
    pub outlet: usize,
    /// area-weighted mean flow distance (m) from hillslope cells to the first channel cell
    /// (or to the outlet where the path meets no channel) along the Priority-Flood paths
    pub hillslope_length_m: f64,
    /// area-weighted mean sin(slope) of the hillslope (non-channel) cells
    pub hillslope_sin_slope: f64,
    /// summed flow-path length (m) of the channel cells (contributing area >= channel initiation)
    pub channel_length_m: f64,
    /// share of the catchment area in channel cells
    pub channel_area_fraction: f64,
    /// elevation range (m) over the catchment cells with data
    pub relief_m: f64,
}

pub struct GiuhInputs<'a> {
    pub g: &'a Grid,
    pub dem: &'a [f32],
    /// tan(local slope) per cell
    pub tan_slope: &'a [f32],
    /// hillslope velocity coefficient k (m/s) per cell
    pub k_hill: &'a [f32],
    pub k_channel_m_s: f64,
    pub channel_area_m2: f64,
    pub slope_floor: f64,
    pub dt_s: f64,
    pub max_ordinates: usize,
}

const NB: [(isize, isize); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

/// Priority-Flood inside one catchment from its outlet, then the time-area
/// histogram. `cells` are global analysis-cell indices of the catchment.
pub fn giuh(inp: &GiuhInputs, cells: &[u32]) -> GiuhOut {
    let g = inp.g;
    let nx = g.nx;
    let (mut i0, mut i1, mut j0, mut j1) = (usize::MAX, 0, usize::MAX, 0);
    for &c in cells {
        let (j, i) = (c as usize / nx, c as usize % nx);
        i0 = i0.min(i);
        i1 = i1.max(i);
        j0 = j0.min(j);
        j1 = j1.max(j);
    }
    let (w, h) = (i1 - i0 + 1, j1 - j0 + 1);
    let mut pos = vec![u32::MAX; w * h];
    for (k, &c) in cells.iter().enumerate() {
        let (j, i) = (c as usize / nx, c as usize % nx);
        pos[(j - j0) * w + (i - i0)] = k as u32;
    }
    let n = cells.len();
    let lj = |k: usize| cells[k] as usize / nx - j0;
    let li = |k: usize| cells[k] as usize % nx - i0;
    let nbr = |k: usize, d: (isize, isize)| -> Option<usize> {
        let (a, b) = (lj(k) as isize + d.0, li(k) as isize + d.1);
        if a < 0 || b < 0 || a >= h as isize || b >= w as isize {
            return None;
        }
        let p = pos[a as usize * w + b as usize];
        if p == u32::MAX { None } else { Some(p as usize) }
    };
    let boundary: Vec<bool> = (0..n).map(|k| NB.iter().any(|&d| nbr(k, d).is_none())).collect();
    let elev = |k: usize| inp.dem[cells[k] as usize];
    let mut parent = vec![u32::MAX; n];
    let mut visited = vec![false; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut heap: BinaryHeap<Reverse<(u32, u32, u32)>> = BinaryHeap::new();
    let mut seq = 0u32;
    let mut first_outlet = usize::MAX;
    // every connected part has boundary cells; seeds are taken lowest first
    let mut seeds: Vec<(u32, u32)> = (0..n).filter(|&k| boundary[k]).map(|k| (fkey(elev(k)), k as u32)).collect();
    seeds.sort_unstable();
    let mut sp = 0usize;
    while order.len() < n {
        // seed: lowest unvisited boundary cell (the outlet of this part)
        while sp < seeds.len() && visited[seeds[sp].1 as usize] {
            sp += 1;
        }
        let best = if sp < seeds.len() { seeds[sp].1 as usize } else { (0..n).find(|&k| !visited[k]).unwrap() };
        if first_outlet == usize::MAX {
            first_outlet = best;
        }
        visited[best] = true;
        heap.push(Reverse((fkey(elev(best)), seq, best as u32)));
        seq += 1;
        while let Some(Reverse((key, _, k))) = heap.pop() {
            let k = k as usize;
            order.push(k as u32);
            let fill = f32::from_bits(if key & 0x8000_0000 != 0 { key & 0x7fff_ffff } else { !key });
            for d in NB {
                if let Some(m) = nbr(k, d) {
                    if !visited[m] {
                        visited[m] = true;
                        parent[m] = k as u32;
                        let e = elev(m);
                        let f = if e.is_finite() && (!fill.is_finite() || e > fill) { e } else { fill };
                        heap.push(Reverse((fkey(f), seq, m as u32)));
                        seq += 1;
                    }
                }
            }
        }
    }
    // contributing area (reverse visit order)
    let area = |k: usize| g.cell_area_m2(cells[k] as usize / nx);
    let mut acc: Vec<f64> = (0..n).map(area).collect();
    for &k in order.iter().rev() {
        let p = parent[k as usize];
        if p != u32::MAX {
            acc[p as usize] += acc[k as usize];
        }
    }
    // travel time to the outlet (visit order: parents first)
    let mut t = vec![0f64; n];
    let dy = g.dy_m();
    let vel = |k: usize| -> f64 {
        let c = cells[k] as usize;
        let s = (inp.tan_slope[c] as f64).max(inp.slope_floor);
        let s = if s.is_finite() { s } else { inp.slope_floor };
        let kk = if acc[k] >= inp.channel_area_m2 { inp.k_channel_m_s } else { inp.k_hill[c] as f64 };
        (kk * s.sqrt()).max(1e-3)
    };
    for &k in &order {
        let k = k as usize;
        let p = parent[k];
        let v = vel(k);
        if p == u32::MAX {
            t[k] = 0.5 * dy / v;
        } else {
            let p = p as usize;
            let dj = lj(k) as isize - lj(p) as isize;
            let di = li(k) as isize - li(p) as isize;
            let dx = 0.5 * (g.dx_m(cells[k] as usize / nx) + g.dx_m(cells[p] as usize / nx));
            let l = ((dj as f64 * dy).powi(2) + (di as f64 * dx).powi(2)).sqrt();
            t[k] = t[p] + l / v;
        }
    }
    // hillslope geometry on the same paths (visit order: parents first). Hillslope
    // length is the flow distance to the first channel cell (Horton 1945 overland
    // flow length, measured on the paths rather than taken as 1/(2 Dd)).
    let mut dch = vec![0f64; n];
    let (mut hl_a, mut hl_sum, mut hs_sum, mut ch_len, mut ch_a) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut zmin, mut zmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for &k in &order {
        let k = k as usize;
        let a = area(k);
        let e = elev(k);
        if e.is_finite() {
            zmin = zmin.min(e as f64);
            zmax = zmax.max(e as f64);
        }
        let p = parent[k];
        let l = if p == u32::MAX {
            0.5 * dy
        } else {
            let p = p as usize;
            let dj = lj(k) as isize - lj(p) as isize;
            let di = li(k) as isize - li(p) as isize;
            let dx = 0.5 * (g.dx_m(cells[k] as usize / nx) + g.dx_m(cells[p] as usize / nx));
            ((dj as f64 * dy).powi(2) + (di as f64 * dx).powi(2)).sqrt()
        };
        if acc[k] >= inp.channel_area_m2 {
            dch[k] = 0.0;
            ch_len += l;
            ch_a += a;
        } else {
            dch[k] = l + if p == u32::MAX { 0.0 } else { dch[p as usize] };
            let ts = inp.tan_slope[cells[k] as usize] as f64;
            let ts = if ts.is_finite() { ts } else { 0.0 };
            hl_a += a;
            hl_sum += a * dch[k];
            hs_sum += a * (ts.atan()).sin();
        }
    }
    let mut hist = vec![0f64; inp.max_ordinates];
    let (mut tot, mut tsum, mut tmax) = (0.0, 0.0, 0f64);
    for k in 0..n {
        let a = area(k);
        let b = ((t[k] / inp.dt_s).floor() as usize).min(inp.max_ordinates - 1);
        hist[b] += a;
        tot += a;
        tsum += a * t[k];
        tmax = tmax.max(t[k]);
    }
    while hist.len() > 1 && *hist.last().unwrap() == 0.0 {
        hist.pop();
    }
    for x in hist.iter_mut() {
        *x /= tot;
    }
    // make the ordinates sum to 1 to the last bit the CFE reader can see
    let s: f64 = hist.iter().sum();
    let last = hist.len() - 1;
    hist[last] += 1.0 - s;
    if hist[last] < 0.0 {
        hist[last] = 0.0;
    }
    GiuhOut {
        ordinates: hist,
        t_mean_s: tsum / tot,
        t_max_s: tmax,
        outlet: cells[first_outlet] as usize,
        hillslope_length_m: if hl_a > 0.0 { hl_sum / hl_a } else { 0.5 * dy },
        hillslope_sin_slope: if hl_a > 0.0 { hs_sum / hl_a } else { 0.0 },
        channel_length_m: ch_len,
        channel_area_fraction: if tot > 0.0 { ch_a / tot } else { 0.0 },
        relief_m: if zmax >= zmin { zmax - zmin } else { 0.0 },
    }
}
