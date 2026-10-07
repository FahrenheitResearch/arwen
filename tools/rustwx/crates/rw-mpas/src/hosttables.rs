//! Cold-path hex mesh tables: deformation weights, advection coefficients,
//! the v8.4.1 vertical-velocity damping profile and the v8.4.1 host-mesh
//! connectivity laws.
//!
//! Each function is an operation-for-operation transcription of the hexcore
//! Python authority it replaces (named on each function).  Every floating
//! operation is one IEEE operation in the storage dtype, in the authority's
//! order; Rust never contracts to FMA.  Transcendentals follow the hexcore
//! ``rkind_libm`` convention: evaluate in binary64 through the platform libm,
//! round once to the storage dtype.  Scalar ``x ** 2`` in the advection
//! source order is the platform ``powf``/``pow`` call numpy's scalar power
//! makes; the exponent is passed through ``black_box`` so the compiler cannot
//! rewrite it as a multiply.
//!
//! A table that would raise in Python returns an error here, and the Python
//! caller then reruns its own authority to raise the exact refusal.  Only a
//! successful table is ever consumed from this module.

use crate::hostprep::{filled, product, read_indices, read_values, write_indices, write_values, Real};
use rayon::prelude::*;
use std::io::{Read, Write};
use std::ops::{Div, Neg};

pub const TABLES_MARKER: &str = "rw_mpas_hostprep --tables hex-hostprep-tables-v1";

pub(crate) trait Tab: Real + Div<Output = Self> + Neg<Output = Self> {
    fn wide(self) -> f64;
    fn narrow(value: f64) -> Self;
    fn sqrt_ieee(self) -> Self;
    fn is_finite_value(self) -> bool;
    fn is_nan_value(self) -> bool;
    fn pow_libm(self, exponent: Self) -> Self;
    fn sin_rk(self) -> Self {
        Self::narrow(self.wide().sin())
    }
    fn cos_rk(self) -> Self {
        Self::narrow(self.wide().cos())
    }
    fn asin_rk(self) -> Self {
        Self::narrow(self.wide().asin())
    }
    fn atan2_rk(self, x: Self) -> Self {
        Self::narrow(self.wide().atan2(x.wide()))
    }
}
impl Tab for f32 {
    fn wide(self) -> f64 {
        self as f64
    }
    fn narrow(value: f64) -> Self {
        value as f32
    }
    fn sqrt_ieee(self) -> Self {
        self.sqrt()
    }
    fn is_finite_value(self) -> bool {
        self.is_finite()
    }
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
    fn pow_libm(self, exponent: Self) -> Self {
        self.powf(std::hint::black_box(exponent))
    }
}
impl Tab for f64 {
    fn wide(self) -> f64 {
        self
    }
    fn narrow(value: f64) -> Self {
        value
    }
    fn sqrt_ieee(self) -> Self {
        self.sqrt()
    }
    fn is_finite_value(self) -> bool {
        self.is_finite()
    }
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
    fn pow_libm(self, exponent: Self) -> Self {
        self.powf(std::hint::black_box(exponent))
    }
}

/// ``np.maximum``: the first argument wins ties and NaN propagates.
fn np_max<T: Tab>(a: T, b: T) -> T {
    if a >= b || a.is_nan_value() {
        a
    } else {
        b
    }
}
/// ``np.minimum``: the first argument wins ties and NaN propagates.
fn np_min<T: Tab>(a: T, b: T) -> T {
    if a <= b || a.is_nan_value() {
        a
    } else {
        b
    }
}

/// ``mixing_v841._sphere_arc_length``.
fn arc_length<T: Tab>(a: [T; 3], b: [T; 3], one: T) -> T {
    let c = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let r = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt_ieee();
    let chord = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt_ieee();
    let two = one + one;
    r * two * (chord / (two * r)).asin_rk()
}

/// ``mixing_v841._sphere_angle``.
fn sphere_angle<T: Tab>(a: [T; 3], b: [T; 3], c: [T; 3], one: T) -> T {
    let zero = one - one;
    let half = one / (one + one);
    let two = one + one;
    let side_a = arc_length(b, c, one);
    let side_b = arc_length(a, c, one);
    let side_c = arc_length(a, b, one);
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let d_x = (ab[1] * ac[2]) - (ab[2] * ac[1]);
    let d_y = -((ab[0] * ac[2]) - (ab[2] * ac[0]));
    let d_z = (ab[0] * ac[1]) - (ab[1] * ac[0]);
    let s = half * (side_a + side_b + side_c);
    let ratio = ((s - side_b).sin_rk() * (s - side_c).sin_rk()) / (side_b.sin_rk() * side_c.sin_rk());
    let sin_angle = np_min(one, np_max(zero, ratio)).sqrt_ieee();
    let magnitude = two * np_max(np_min(sin_angle, one), -one).asin_rk();
    if (d_x * a[0] + d_y * a[1] + d_z * a[2]) >= zero {
        magnitude
    } else {
        -magnitude
    }
}

/// ``mixing_v841.initialize_deformation_weights_v841`` (spherical branch).
/// Constants arrive from Python: [radius, pii].
pub(crate) fn deformation_weights<T: Tab>(
    input: &mut impl Read,
    output: &mut impl Write,
    nc: usize,
    ne: usize,
    me: usize,
    counts: &[i64],
    eoc: &[i64],
) -> Result<(), String> {
    let nv = read_indices(input, 1)?[0];
    let nv = usize::try_from(nv).map_err(|_| "invalid vertex count")?;
    let coc = read_indices(input, product(&[nc, me])?)?;
    let voc = read_indices(input, product(&[nc, me])?)?;
    let coe = read_indices(input, product(&[ne, 2])?)?;
    let consts = read_values::<T>(input, 2)?;
    let (radius, pii) = (consts[0], consts[1]);
    let xc = read_values::<T>(input, nc)?;
    let yc = read_values::<T>(input, nc)?;
    let zc = read_values::<T>(input, nc)?;
    let xv = read_values::<T>(input, nv)?;
    let yv = read_values::<T>(input, nv)?;
    let zv = read_values::<T>(input, nv)?;
    let one = T::one();
    let zero = T::zero();
    let quarter = T::narrow(0.25);
    let size = product(&[nc, me])?;
    let mut c2 = filled(size, T::zero())?;
    let mut s2 = filled(size, T::zero())?;
    let mut cs = filled(size, T::zero())?;
    let results: Vec<Result<(), String>> = c2
        .par_chunks_mut(me)
        .zip(s2.par_chunks_mut(me))
        .zip(cs.par_chunks_mut(me))
        .enumerate()
        .map(|(cell, ((c2r, s2r), csr))| {
            let count = counts[cell];
            if count < 3 || count as usize > me {
                return Err(format!("cell {cell} has invalid nEdgesOnCell {count}"));
            }
            let count = count as usize;
            let row = cell * me;
            if coc[row..row + count].iter().any(|&n| n < 0 || n as u64 >= nc as u64) {
                return Ok(());
            }
            let verts = &voc[row..row + count];
            if verts.iter().any(|&v| v < 0 || v as u64 >= nv as u64) {
                return Err(format!("verticesOnCell reaches outside the mesh at cell {cell}"));
            }
            let slot_edges = &eoc[row..row + count];
            if slot_edges.iter().any(|&e| e < 0 || e as u64 >= ne as u64) {
                return Err(format!("edgesOnCell reaches outside the mesh at cell {cell}"));
            }
            let c = [xc[cell] / radius, yc[cell] / radius, zc[cell] / radius];
            let v: Vec<[T; 3]> = verts
                .iter()
                .map(|&k| {
                    let k = k as usize;
                    [xv[k] / radius, yv[k] / radius, zv[k] / radius]
                })
                .collect();
            let theta_abs = if c[2] == one {
                pii / (one + one)
            } else {
                pii / (one + one) - sphere_angle(c, v[0], [zero, zero, one], one)
            };
            let mut thetav = vec![zero; count];
            let mut dl_sphere = vec![zero; count];
            for j in 0..count {
                let jp1 = (j + 1) % count;
                thetav[j] = sphere_angle(c, v[j], v[jp1], one);
                dl_sphere[j] = radius * arc_length(c, v[j], one);
            }
            let mut thetat = vec![zero; count];
            thetat[0] = theta_abs;
            for j in 1..count {
                thetat[j] = thetat[j - 1] + thetav[j - 1];
            }
            let xp: Vec<T> = (0..count).map(|j| thetat[j].cos_rk() * dl_sphere[j]).collect();
            let yp: Vec<T> = (0..count).map(|j| thetat[j].sin_rk() * dl_sphere[j]).collect();
            let mut area_cell = zero;
            let mut theta_edge = vec![zero; count];
            for j in 0..count {
                let jp1 = (j + 1) % count;
                let dx = xp[jp1] - xp[j];
                let dy = yp[jp1] - yp[j];
                area_cell = area_cell + quarter * (xp[j] + xp[jp1]) * (yp[jp1] - yp[j])
                    - quarter * (yp[j] + yp[jp1]) * (xp[jp1] - xp[j]);
                theta_edge[j] = dy.atan2_rk(dx) - pii / (one + one);
            }
            for j in 0..count {
                let jp1 = (j + 1) % count;
                let dx = xp[jp1] - xp[j];
                let dy = yp[jp1] - yp[j];
                let dl = (dx * dx + dy * dy).sqrt_ieee();
                let sin_t = theta_edge[j].sin_rk();
                let cos_t = theta_edge[j].cos_rk();
                let sint2 = sin_t * sin_t;
                let cost2 = cos_t * cos_t;
                let sint_cost = sin_t * cos_t;
                let mut a = dl * cost2 / area_cell;
                let mut b = dl * sint2 / area_cell;
                let mut d = dl * sint_cost / area_cell;
                if coe[slot_edges[j] as usize * 2] != cell as i64 {
                    a = -a;
                    b = -b;
                    d = -d;
                }
                c2r[j] = a;
                s2r[j] = b;
                csr[j] = d;
            }
            Ok(())
        })
        .collect();
    if let Some(error) = results.into_iter().find_map(Result::err) {
        return Err(error);
    }
    if c2.iter().chain(s2.iter()).chain(cs.iter()).any(|v| !v.is_finite_value()) {
        return Err("deformation weights are not finite".into());
    }
    write_values(output, &c2)?;
    write_values(output, &s2)?;
    write_values(output, &cs)?;
    Ok(())
}

/// ``transport.build_advection_coefficients`` stencil compression.
/// Metadata arrives as [width, source_order_v841, garbage(-1 or nCells)].
pub(crate) fn advection_coefficients<T: Tab>(
    input: &mut impl Read,
    output: &mut impl Write,
    nc: usize,
    ne: usize,
    me: usize,
    counts: &[i64],
) -> Result<(), String> {
    let meta = read_indices(input, 3)?;
    let width = usize::try_from(meta[0]).map_err(|_| "invalid stencil width")?;
    let source_order = meta[1] != 0;
    let garbage = meta[2];
    if width == 0 {
        return Err("stencil width must be positive".into());
    }
    let coc = read_indices(input, product(&[nc, me])?)?;
    let coe = read_indices(input, product(&[ne, 2])?)?;
    let ids = read_indices(input, nc)?;
    let deriv = read_values::<T>(input, product(&[ne, 2, width])?)?;
    let dc = read_values::<T>(input, ne)?;
    let dv = read_values::<T>(input, ne)?;
    let half = T::narrow(0.5);
    let twelve = T::narrow(12.0);
    let two = T::narrow(2.0);
    let n_cells = nc as i64;
    let cell_count = |cell: i64| -> Result<usize, String> {
        if cell == garbage {
            Ok(0)
        } else if cell >= 0 && cell < n_cells {
            Ok(counts[cell as usize] as usize)
        } else {
            Err(format!("advection stencil reaches cell {cell} outside the mesh"))
        }
    };
    let cell_id = |cell: i64| -> i64 {
        if cell == garbage {
            n_cells + 1
        } else {
            ids[cell as usize]
        }
    };
    let size = product(&[ne, width])?;
    let mut adv = filled(size, T::zero())?;
    let mut adv3 = filled(size, T::zero())?;
    let mut cells = filled(size, -1i64)?;
    let mut n_adv = filled(ne, 0i64)?;
    let results: Vec<Result<(), String>> = adv
        .par_chunks_mut(width)
        .zip(adv3.par_chunks_mut(width))
        .zip(cells.par_chunks_mut(width))
        .zip(n_adv.par_iter_mut())
        .enumerate()
        .map(|(edge, (((a, a3), cl), n))| {
            let cell1 = coe[edge * 2];
            let cell2 = coe[edge * 2 + 1];
            let count1 = cell_count(cell1)?;
            let count2 = cell_count(cell2)?;
            let mut unique: Vec<i64> = Vec::with_capacity(2 + count1 + count2);
            let mut push = |value: i64| {
                if !unique.contains(&value) {
                    unique.push(value);
                }
            };
            push(cell1);
            push(cell2);
            for slot in 0..count1 {
                push(coc[cell1 as usize * me + slot]);
            }
            for slot in 0..count2 {
                push(coc[cell2 as usize * me + slot]);
            }
            for &cell in &unique {
                if cell != garbage && (cell < 0 || cell >= n_cells) {
                    return Err(format!("advection stencil reaches cell {cell} outside the mesh"));
                }
            }
            unique.sort_by_key(|&cell| cell_id(cell));
            if unique.len() > width {
                return Err(format!(
                    "edge {edge} needs {} advection cells but deriv_two has width {width}",
                    unique.len()
                ));
            }
            if count1 + 1 > width || count2 + 1 > width {
                return Err("deriv_two is narrower than the cell stencil".into());
            }
            *n = unique.len() as i64;
            cl[..unique.len()].copy_from_slice(&unique);
            let position = |cell: i64| unique.iter().position(|&c| c == cell).unwrap();
            let base = edge * 2 * width;
            let target = position(cell1);
            a[target] = a[target] + deriv[base];
            a3[target] = a3[target] + deriv[base];
            for slot in 0..count1 {
                let target = position(coc[cell1 as usize * me + slot]);
                let value = deriv[base + slot + 1];
                a[target] = a[target] + value;
                a3[target] = a3[target] + value;
            }
            let target = position(cell2);
            a[target] = a[target] + deriv[base + width];
            a3[target] = a3[target] - deriv[base + width];
            for slot in 0..count2 {
                let target = position(coc[cell2 as usize * me + slot]);
                let value = deriv[base + width + slot + 1];
                a[target] = a[target] + value;
                a3[target] = a3[target] - value;
            }
            let len = unique.len();
            if source_order {
                let dc_squared = dc[edge].pow_libm(two);
                for slot in 0..len {
                    a[slot] = -dc_squared * a[slot] / twelve;
                    a3[slot] = -dc_squared * a3[slot] / twelve;
                }
            } else {
                let scale = -(dc[edge] * dc[edge]) / twelve;
                for slot in 0..len {
                    a[slot] = scale * a[slot];
                    a3[slot] = scale * a3[slot];
                }
            }
            let p1 = position(cell1);
            a[p1] = a[p1] + half;
            let p2 = position(cell2);
            a[p2] = a[p2] + half;
            for slot in 0..len {
                a[slot] = dv[edge] * a[slot];
                a3[slot] = dv[edge] * a3[slot];
            }
            Ok(())
        })
        .collect();
    if let Some(error) = results.into_iter().find_map(Result::err) {
        return Err(error);
    }
    write_values(output, &adv)?;
    write_values(output, &adv3)?;
    write_indices(output, &n_adv)?;
    write_indices(output, &cells)?;
    Ok(())
}

/// ``damping_v841.build_v841_vertical_velocity_damping`` loop.  Python keeps
/// the vectorized input validation; constants arrive as [xnutr, start, pi].
pub(crate) fn vertical_velocity_damping<T: Tab>(
    input: &mut impl Read,
    output: &mut impl Write,
    nc: usize,
    nl: usize,
) -> Result<(), String> {
    let consts = read_values::<T>(input, 3)?;
    let (xnutr, start, pi) = (consts[0], consts[1], consts[2]);
    let heights = read_values::<T>(input, product(&[nl + 1, nc])?)?;
    let half = T::narrow(0.5);
    let mut dss = filled(product(&[nl, nc])?, T::zero())?;
    if xnutr == T::zero() {
        write_values(output, &dss)?;
        return Ok(());
    }
    let results: Vec<Result<(), String>> = dss
        .par_chunks_mut(nc)
        .enumerate()
        .map(|(level, row)| {
            for cell in 0..nc {
                let top = heights[nl * nc + cell];
                let height = half * (heights[level * nc + cell] + heights[(level + 1) * nc + cell]);
                if !height.is_finite_value() {
                    return Err(format!("midpoint overflow at level {level}, cell {cell}"));
                }
                if height > start {
                    if top == start {
                        return Err("the active v8.4.1 damping phase would divide by zero".into());
                    }
                    let phase = half * pi * (height - start) / (top - start);
                    let sin_phase = phase.sin_rk();
                    let value = xnutr * (sin_phase * sin_phase);
                    if !phase.is_finite_value() || !value.is_finite_value() {
                        return Err(format!("non-finite damping phase at level {level}, cell {cell}"));
                    }
                    row[cell] = value;
                }
            }
            Ok(())
        })
        .collect();
    if let Some(error) = results.into_iter().find_map(Result::err) {
        return Err(error);
    }
    write_values(output, &dss)?;
    Ok(())
}

/// The per-row connectivity laws of ``cuda_driver._validate_v841_host_mesh``.
/// Writes one byte: 1 when every law holds.  Python reruns its own loops to
/// raise the exact refusal when this writes 0.  Negative indices follow
/// Python's wraparound so a regional sentinel is read exactly as the
/// authority reads it.
pub(crate) fn host_mesh_laws(
    input: &mut impl Read,
    output: &mut impl Write,
    nc: usize,
    ne: usize,
    me: usize,
    counts: &[i64],
    eoc: &[i64],
) -> Result<(), String> {
    let meta = read_indices(input, 2)?;
    let nv = usize::try_from(meta[0]).map_err(|_| "invalid vertex count")?;
    let me2 = usize::try_from(meta[1]).map_err(|_| "invalid maxEdges2")?;
    let coc = read_indices(input, product(&[nc, me])?)?;
    let voc = read_indices(input, product(&[nc, me])?)?;
    let coe = read_indices(input, product(&[ne, 2])?)?;
    let edge_counts = read_indices(input, ne)?;
    let eoe = read_indices(input, product(&[ne, me2])?)?;
    let voe = read_indices(input, product(&[ne, 2])?)?;
    let eov = read_indices(input, product(&[nv, 3])?)?;
    let cov = read_indices(input, product(&[nv, 3])?)?;
    let mut masks = [vec![0u8; nc], vec![0u8; ne], vec![0u8; nv]];
    for mask in masks.iter_mut() {
        input
            .read_exact(mask)
            .map_err(|error| format!("truncated mask input: {error}"))?;
    }
    let [cell_ok, edge_ok, vertex_ok] = masks;
    let wrap = |value: i64, extent: usize| -> Option<usize> {
        let n = extent as i64;
        if value >= 0 && value < n {
            Some(value as usize)
        } else if value < 0 && value >= -n {
            Some((value + n) as usize)
        } else {
            None
        }
    };
    let unique = |row: &[i64]| -> bool {
        row.iter().enumerate().all(|(i, v)| !row[..i].contains(v))
    };
    let cells_pass = (0..nc).into_par_iter().all(|cell| {
        if cell_ok[cell] == 0 {
            return true;
        }
        let count = counts[cell] as usize;
        let row = cell * me;
        let edges = &eoc[row..row + count];
        let cells = &coc[row..row + count];
        let vertices = &voc[row..row + count];
        if edges.iter().any(|&e| e < 0 || e as u64 >= ne as u64)
            || cells.iter().any(|&c| c < 0 || c as u64 >= nc as u64)
            || vertices.iter().any(|&v| v < 0 || v as u64 >= nv as u64)
            || !unique(edges)
            || !unique(cells)
            || !unique(vertices)
        {
            return false;
        }
        (0..count).all(|slot| {
            let edge = edges[slot] as usize;
            let e0 = coe[edge * 2];
            let e1 = coe[edge * 2 + 1];
            let me_cell = cell as i64;
            if e0 != me_cell && e1 != me_cell {
                return false;
            }
            let opposite = if e0 == me_cell { e1 } else { e0 };
            cells[slot] == opposite
        })
    });
    if !cells_pass {
        output.write_all(&[0]).map_err(|error| error.to_string())?;
        return Ok(());
    }
    let edges_pass = (0..ne).into_par_iter().all(|edge| {
        if edge_ok[edge] == 0 {
            return true;
        }
        let count = edge_counts[edge] as usize;
        if count > me2 {
            return false;
        }
        let neighbors = &eoe[edge * me2..edge * me2 + count];
        if neighbors.iter().any(|&n| n < 0 || n as u64 >= ne as u64) {
            return false;
        }
        if neighbors.contains(&(edge as i64)) || !unique(neighbors) {
            return false;
        }
        let endpoints = [coe[edge * 2], coe[edge * 2 + 1]];
        for &neighbor in neighbors {
            let n = neighbor as usize;
            let other = [coe[n * 2], coe[n * 2 + 1]];
            if !endpoints.iter().any(|value| other.contains(value)) {
                return false;
            }
        }
        let mut expected: Vec<i64> = Vec::with_capacity(2 * me);
        let mut seen_endpoint: Vec<i64> = Vec::with_capacity(2);
        for &cell in &endpoints {
            if seen_endpoint.contains(&cell) {
                continue;
            }
            seen_endpoint.push(cell);
            let Some(index) = wrap(cell, nc) else {
                return false;
            };
            let cell_count = counts[index] as usize;
            for &value in &eoc[index * me..index * me + cell_count] {
                if value != edge as i64 && !expected.contains(&value) {
                    expected.push(value);
                }
            }
        }
        neighbors.len() == expected.len() && neighbors.iter().all(|value| expected.contains(value))
    });
    if !edges_pass {
        output.write_all(&[0]).map_err(|error| error.to_string())?;
        return Ok(());
    }
    let vertices_pass = (0..nv).into_par_iter().all(|vertex| {
        if vertex_ok[vertex] == 0 {
            return true;
        }
        let edges = &eov[vertex * 3..vertex * 3 + 3];
        let cells = &cov[vertex * 3..vertex * 3 + 3];
        if !unique(edges) || !unique(cells) {
            return false;
        }
        edges.iter().all(|&edge| {
            if edge < 0 || edge as u64 >= ne as u64 {
                return false;
            }
            let edge = edge as usize;
            voe[edge * 2] == vertex as i64 || voe[edge * 2 + 1] == vertex as i64
        })
    });
    let reciprocal = vertices_pass
        && (0..ne).into_par_iter().all(|edge| {
            if edge_ok[edge] == 0 {
                return true;
            }
            (0..2).all(|side| {
                let vertex = voe[edge * 2 + side];
                if vertex < 0 || vertex as u64 >= nv as u64 {
                    return false;
                }
                let vertex = vertex as usize;
                vertex_ok[vertex] == 0 || eov[vertex * 3..vertex * 3 + 3].contains(&(edge as i64))
            })
        });
    let kites = reciprocal
        && (0..nc).into_par_iter().all(|cell| {
            if cell_ok[cell] == 0 {
                return true;
            }
            (0..counts[cell] as usize).all(|slot| {
                let vertex = voc[cell * me + slot];
                if vertex < 0 || vertex as u64 >= nv as u64 {
                    return false;
                }
                let vertex = vertex as usize;
                cov[vertex * 3..vertex * 3 + 3].contains(&(cell as i64))
            })
        });
    output
        .write_all(&[u8::from(kites)])
        .map_err(|error| error.to_string())?;
    Ok(())
}

/// Elementwise platform-libm ``pow`` in the storage dtype: the call a numpy
/// scalar ``x ** y`` makes (``npy_powf``/``npy_pow`` are the platform
/// ``powf``/``pow``).  Serves ``driver._frozen_vertical_damping``, whose
/// ``sin(phase) ** 2`` and ``meshDensity ** 0.25`` are scalar powers.
pub(crate) fn pow_elementwise<T: Tab>(
    input: &mut impl Read,
    output: &mut impl Write,
    count: usize,
) -> Result<(), String> {
    let exponent = read_values::<T>(input, 1)?[0];
    let mut values = read_values::<T>(input, count)?;
    values.par_iter_mut().for_each(|value| *value = value.pow_libm(exponent));
    write_values(output, &values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numpy_extrema_keep_the_first_argument_on_ties_and_nan() {
        assert!(np_max(f32::NAN, 1.0).is_nan());
        assert!(np_max(1.0f32, f32::NAN).is_nan());
        assert!(np_min(f64::NAN, 1.0).is_nan());
        assert!(np_min(1.0f64, f64::NAN).is_nan());
        assert_eq!(np_min(2.0f32, 3.0), 2.0);
        assert_eq!(np_max(2.0f32, 3.0), 3.0);
    }
    #[test]
    fn transcendentals_round_once_from_binary64() {
        let x = 1.234_567_9f32;
        assert_eq!(x.sin_rk().to_bits(), ((x as f64).sin() as f32).to_bits());
        assert_eq!(x.atan2_rk(0.5).to_bits(), ((x as f64).atan2(0.5) as f32).to_bits());
    }
    #[test]
    fn pow_mode_is_elementwise_libm() {
        let mut input = Vec::new();
        input.extend_from_slice(&2.0f32.to_le_bytes());
        for value in [3.0f32, 0.1, -2.5] {
            input.extend_from_slice(&value.to_le_bytes());
        }
        let mut output = Vec::new();
        pow_elementwise::<f32>(&mut &input[..], &mut output, 3).unwrap();
        let got: Vec<f32> = output.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let want: Vec<f32> = [3.0f32, 0.1, -2.5].iter().map(|v| v.powf(std::hint::black_box(2.0))).collect();
        assert_eq!(got, want);
    }
}
