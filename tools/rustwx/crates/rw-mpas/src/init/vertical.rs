//! Case-7 terrain, coordinate surfaces and omega metrics, in native RKIND order.
//! MPAS v8.4.2 src/core_init_atmosphere/mpas_init_atm_cases.F:3018-3443.
//! Each protocol call owns the complete mesh (or a caller-owned working halo).
//! No contraction, parallel reduction or algebraic rearrangement is used.
//! Loops run in parallel only over independent outputs (one cell, edge or
//! level each), and every output keeps its serial term order, so the words
//! are those of the one-thread loops. On the 30 km global mesh the serial
//! surface-acceptance passes held one core for 11 min (HEX-GLOBAL-1-2-CARD,
//! 2026-10-16, "Found, not fixed" item 2).

use crate::geometry::Real;
use rayon::prelude::*;
use serde::Deserialize;
use std::io::{Read, Write};

pub const ABI_MARKER: &str = "rw_mpas_geometry --protocol hex-vertical-v1";
/// The same request with its arrays framed as raw little-endian words instead
/// of JSON numbers. At 30 km the v1 edge request is 1.65 GB of JSON text that
/// hex encoded and this side parsed on one thread each (HEX-GLOBAL-1-2-CARD,
/// 2026-10-16, item 2 follow-up). Same values, same outputs.
pub const ABI_MARKER_V2: &str = "rw_mpas_geometry --protocol hex-vertical-v2";
const V2_MAGIC: &[u8; 8] = b"HEXVRT2\0";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub mode: String,
    pub width: usize,
    pub terrain: Vec<f64>,
    pub counts: Vec<usize>,
    pub edges_on_cell: Vec<Vec<i64>>,
    pub cells_on_cell: Vec<Vec<i64>>,
    pub cells_on_edge: Vec<[i64; 2]>,
    pub dc: Vec<f64>,
    pub dv: Vec<f64>,
    pub area: Vec<f64>,
    pub levels: usize,
    pub top: f64,
    pub scheme: String,
    pub specified: Option<Vec<f64>>,
    pub projection: String,
    pub terrain_passes: usize,
    pub smooth: bool,
    pub surface_passes: usize,
    pub minimum_fraction: f64,
    pub hybrid: bool,
    pub transition: f64,
    pub damping: f64,
    pub damping_start: f64,
    pub regional_halo: bool,
    pub order: i32,
    /// Edge-major, side-major, coefficient-fastest.
    pub deriv: Vec<Vec<Vec<f64>>>,
    /// Level-major, cell-fastest; only the edge mode reads it.
    pub zgrid: Vec<Vec<f64>>,
    #[serde(default)]
    pub boundary: Option<Boundary>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub met_path: String,
    pub lat: Vec<f32>, pub lon: Vec<f32>, pub mask: Vec<i32>,
}

fn fail(message: &str) -> Result<(), String> { Err(message.to_string()) }

/// Validate array topology before any indexing; invalid neighbors otherwise
/// address unrelated cells or panic. MPAS mesh ABI: mpas_init_atm_cases.F:3032-3067,3415-3464.
fn validate(r: &Request) -> Result<(), String> {
    let nc = r.terrain.len();
    let ne = r.dc.len();
    if nc == 0 || ne == 0 || r.counts.len() != nc || r.edges_on_cell.len() != nc
        || r.cells_on_cell.len() != nc || r.dv.len() != ne || r.area.len() != nc
        || r.cells_on_edge.len() != ne {
        return fail("vertical topology/metric dimensions are inconsistent");
    }
    if r.terrain.iter().any(|v| !v.is_finite())
        || r.dc.iter().chain(&r.area).any(|&v| !v.is_finite() || v <= 0.0)
        || r.dv.iter().any(|v| !v.is_finite()) {
        return fail("vertical terrain, edge lengths or areas are non-finite or non-positive");
    }
    for c in 0..nc {
        let n = r.counts[c];
        if n == 0 || r.edges_on_cell[c].len() < n || r.cells_on_cell[c].len() < n {
            return fail("nEdgesOnCell exceeds connectivity rows or is zero");
        }
        for j in 0..n {
            if r.edges_on_cell[c][j] < 0 || r.edges_on_cell[c][j] >= ne as i64 || r.cells_on_cell[c][j] < -1
                || r.cells_on_cell[c][j] >= nc as i64 {
                return fail("vertical connectivity contains an out-of-range cell or edge");
            }
        }
    }
    for (edge, &[a,b]) in r.cells_on_edge.iter().enumerate() {
        if a < -1 || b < -1 || a >= nc as i64 || b >= nc as i64 || (a == -1 && b == -1) {
            return Err(if a == -1 && b == -1 { format!("edge {edge} has no interior cell") } else { format!("edge {edge} references an out-of-range cell") });
        }
        if (a == -1 || b == -1) && !r.regional_halo {
            return fail("regional_boundary_metrics requires a caller-owned working halo");
        }
    }
    Ok(())
}

/// Ordered unscaled Laplacian and two-pass smoother, including garbage-cell
/// substitution and the exact-zero guard. mpas_init_atm_cases.F:3025-3084.
fn laplacian<T: Real>(plan: &Stencil<T>, field: &[T], skip_zero: bool) -> Vec<T> {
    (0..field.len()).into_par_iter().map(|c| plan.cell(field, c, skip_zero)).collect()
}

/// The Laplacian's neighbor table, flattened once per request (CSR, native
/// j order). `weight` is the same rounded dv/dc quotient the per-term
/// expression forms, and a garbage neighbor points at the cell itself, which
/// is the value native substitutes; the sums are word-identical. The nested
/// connectivity rows otherwise stream about 170 MB per pass, over 3,000
/// passes at 30 km.
struct Stencil<T> { start: Vec<usize>, source: Vec<u32>, weight: Vec<T> }

impl<T: Real> Stencil<T> {
    fn new(r: &Request) -> Self {
        let nc = r.counts.len();
        let mut start = Vec::with_capacity(nc+1);
        let mut source = Vec::new();
        let mut weight = Vec::new();
        start.push(0);
        for c in 0..nc {
            for j in 0..r.counts[c] {
                let e = r.edges_on_cell[c][j] as usize;
                let neighbor = r.cells_on_cell[c][j];
                source.push(if neighbor < 0 { c as u32 } else { neighbor as u32 });
                weight.push(T::number(r.dv[e])/T::number(r.dc[e]));
            }
            start.push(source.len());
        }
        Stencil { start, source, weight }
    }

    /// One cell's sum: it reads only `field`, in native j order.
    #[inline]
    fn cell(&self, field: &[T], c: usize, skip_zero: bool) -> T {
        let mut result = T::zero();
        if skip_zero && field[c] == T::zero() { return result; }
        for i in self.start[c]..self.start[c+1] {
            result = result + self.weight[i]*(field[self.source[i] as usize]-field[c]);
        }
        result
    }
}

/// Fourth-order terrain smoothing. MPAS mpas_init_atm_cases.F:3025-3084.
fn smooth_terrain<T: Real>(r: &Request, plan: &Stencil<T>) -> Vec<T> {
    let mut terrain: Vec<T> = r.terrain.iter().map(|&v| T::number(v)).collect();
    for _ in 0..r.terrain_passes {
        let first = laplacian(plan, &terrain, true);
        let hs: Vec<T> = terrain.iter().zip(first).map(|(&v,l)| v+T::number(0.216)*l).collect();
        let second = laplacian(plan, &hs, true);
        terrain = hs.iter().zip(second).map(|(&v,l)| v-T::number(0.216)*l).collect();
    }
    terrain
}

fn write<T: Real>(out: &mut impl Write, values: &[T]) -> Result<(),String> {
    // Blocks are encoded in parallel and written in order, so the bytes are
    // those of the one-value-at-a-time loop; a bounded batch keeps the extra
    // memory to a few tens of MB.
    for batch in values.chunks(1 << 22) {
        let blocks: Vec<Result<Vec<u8>,String>> = batch.par_chunks(1 << 16).map(|chunk| {
            let mut bytes = Vec::with_capacity(chunk.len()*T::WIDTH);
            for &v in chunk { v.encode(&mut bytes)?; }
            Ok(bytes)
        }).collect();
        for block in blocks { out.write_all(&block?).map_err(|e|e.to_string())?; }
    }
    Ok(())
}

/// All three interface branches and both projections; hybrid/basic coordinate,
/// iterative surface acceptance and metric closure. mpas_init_atm_cases.F:3097-3397.
fn grid<T: Real>(r: &Request, out: &mut impl Write) -> Result<(),String> {
    let n = r.levels;
    let nc = r.terrain.len();
    if n < 3 { return fail("lower-boundary extrapolation requires at least three levels"); }
    if !r.minimum_fraction.is_finite() || r.minimum_fraction <= 0.0 || r.minimum_fraction >= 1.0 {
        return fail("minimum layer fraction must lie strictly between zero and one");
    }
    let one = T::one();
    let half = T::half();
    let pi = T::number(std::f64::consts::PI);
    let mut top = T::number(r.top);
    let plan = Stencil::<T>::new(r);
    let terrain = smooth_terrain::<T>(r, &plan);
    let mut zw = vec![T::zero(); n+1];
    if r.scheme == "specified" {
        let specified = r.specified.as_ref().ok_or("specified grid needs interface heights")?;
        if specified.len() != n+1 { return fail("specified heights must contain nVertLevels+1 values"); }
        for k in 0..=n { zw[k] = T::number(specified[k]); }
        top = zw[n];
    } else {
        if !top.finite() || top <= T::zero() { return fail("model top must be finite and positive"); }
        let dz = top/T::count(n);
        let (als,alt,zetal) = if n >= 55 { (0.075,1.70,0.75) } else { (0.075,1.23,0.31) };
        let (als,alt,zetal) = (T::number(als),T::number(alt),T::number(zetal));
        for k in 0..=n {
            let eta = T::count(k)/T::count(n);
            zw[k] = match r.scheme.as_str() {
                "legacy" => eta.pow(T::number(1.5))*top,
                "tc" if eta < zetal => {
                    let x = T::count(k)*dz/(top*zetal);
                    // GNU integer powers: x**2 = x*x; x**3 = (x*x)*x.
                    (als*T::count(k)/T::count(n)
                        +(T::number(3.0)*(one-alt)+T::number(2.0)*(alt-als)*zetal)*(x*x)
                        -(T::number(2.0)*(one-alt)+(alt-als)*zetal)*((x*x)*x))*top
                }
                "tc" => {
                    let zl = one-alt*(one-zetal);
                    (zl+alt*(eta-zetal))*top
                }
                _ => return fail("vertical scheme must be tc, legacy or specified"),
            };
        }
    }
    if zw.iter().any(|v| !v.finite()) || zw.windows(2).any(|p| p[1] <= p[0]) {
        return fail("vertical interfaces must be finite and strictly increasing");
    }
    if zw[0] != T::zero() { return fail("first vertical interface must be zero"); }
    let zh = if r.hybrid { T::number(r.transition) } else { top };
    if !zh.finite() || zh <= T::zero() { return fail("hybrid transition must be finite and positive"); }
    let mut kz = n; // Fortran kz=nz => zero-based n (even if no height level).
    let mut ah = vec![T::zero(); n+1];
    for k in 0..=n {
        ah[k] = if r.hybrid {
            if zw[k] < zh {
                let a = (half*pi*zw[k]/zh).cos();
                let a2 = a*a;
                // GNU powi(6) multiplies the cube by itself.
                let a3 = a2*a;
                a3*a3
            } else { kz = kz.min(k); T::zero() }
        } else { one-zw[k]/top };
    }
    let mut dzw = vec![T::zero(); n];
    let mut rdzw = dzw.clone();
    let mut zu = dzw.clone();
    let mut dzu = dzw.clone();
    let mut rdzu = dzw.clone();
    let mut rdzwp = dzw.clone();
    let mut rdzwm = dzw.clone();
    let mut fzp = dzw.clone();
    let mut fzm = dzw.clone();
    for k in 0..n {
        dzw[k] = zw[k+1]-zw[k];
        rdzw[k] = one/dzw[k];
        zu[k] = half*(zw[k]+zw[k+1]);
    }
    for k in 1..n {
        dzu[k] = half*(dzw[k]+dzw[k-1]);
        rdzu[k] = one/dzu[k];
        rdzwp[k] = dzw[k-1]/(dzw[k]*(dzw[k]+dzw[k-1]));
        rdzwm[k] = dzw[k]/(dzw[k-1]*(dzw[k]+dzw[k-1]));
        let a = half*dzw[k]/dzu[k];
        let b = half*dzw[k-1]/dzu[k];
        match r.projection.as_str() {
            "linear_interpolation" => { fzp[k] = a; fzm[k] = b; }
            "layer_integral" => { fzm[k] = a; fzp[k] = b; }
            _ => return fail("projection must be linear_interpolation or layer_integral"),
        }
    }
    let cof1 = (T::number(2.0)*dzu[1]+dzu[2])/(dzu[1]+dzu[2])*dzw[0]/dzu[1];
    let cof2 = dzu[1]/(dzu[1]+dzu[2])*dzw[0]/dzu[2];
    let cf = [fzp[1]+cof1, (fzm[1]-cof1)-cof2, cof2];
    let mut hx: Vec<Vec<T>> = vec![terrain; n+1];
    if r.smooth {
        let mut sm0 = vec![T::zero(); nc];
        for c in 0..nc {
            let mut sum = T::zero();
            for j in 0..r.counts[c] { sum = sum+T::number(r.dc[r.edges_on_cell[c][j] as usize]); }
            sum = sum/T::count(r.counts[c]);
            let mut factor = T::number(3000.0)/sum;
            if factor > one { factor = one; }
            factor = T::number(0.125)*factor;
            if factor < T::number(0.01) { factor = T::number(0.01); }
            sm0[c] = factor;
        }
        for k in 1..kz {
            hx[k] = hx[k-1].clone();
            let a = T::number(3.0)*zw[k]/zh;
            let mut factor = a*a;
            if factor > one { factor = one; }
            let (lower, upper) = hx.split_at_mut(k);
            let (below, row) = (&lower[k-1], &mut upper[0]);
            let mut next = row.clone();
            for _ in 0..r.surface_passes+k+1 {
                // Native computes the whole pass's Laplacian before accepting
                // any cell, so a pass reads only the previous row: write the
                // accepted row into `next` and swap, one parallel sweep a pass.
                let current: &[T] = row;
                next.par_iter_mut().enumerate().for_each(|(c,h)| {
                    let l = plan.cell(current, c, false);
                    let sm = sm0[c]*factor;
                    let hs = current[c]+sm*l;
                    let dz = (zw[k]+ah[k]*hs)-(zw[k-1]+ah[k-1]*below[c]);
                    *h = if dz > T::number(r.minimum_fraction)*(zw[k]-zw[k-1]) { hs } else { current[c] };
                });
                std::mem::swap(row, &mut next);
            }
        }
        for row in &mut hx[kz..] { row.fill(T::zero()); }
    }
    let mut zg = vec![vec![T::zero();nc];n+1];
    let mut zz = vec![vec![T::zero();nc];n];
    zg.par_iter_mut().zip(hx.par_iter()).enumerate().for_each(|(k,(g,h))| {
        for c in 0..nc { g[c] = zw[k]+ah[k]*h[c]; }
    });
    zz.par_iter_mut().enumerate().try_for_each(|(k,z)| {
        for c in 0..nc {
            let dz = zg[k+1][c]-zg[k][c];
            if !dz.finite() || dz <= T::zero() { return fail("non-positive physical layer thickness"); }
            z[c] = (zw[k+1]-zw[k])/dz;
        }
        Ok(())
    })?;
    let mut zxu = vec![vec![T::zero();r.dc.len()];n];
    zxu.par_iter_mut().enumerate().for_each(|(k,row)| {
        for e in 0..r.dc.len() {
            let [a,b] = r.cells_on_edge[e];
            let c1 = if a < 0 { b as usize } else { a as usize };
            let c2 = if b < 0 { c1 } else { b as usize };
            row[e] = half*(((zg[k][c2]-zg[k][c1])+zg[k+1][c2])-zg[k+1][c1])/T::number(r.dc[e]);
        }
    });
    let mut dss = vec![vec![T::zero();nc];n];
    let zd = T::number(r.damping_start);
    let amp = T::number(r.damping);
    if !zd.finite() || !amp.finite() || (amp != T::zero() && zd >= top) {
        return fail("nonzero damping requires a finite start below model top");
    }
    dss.par_iter_mut().enumerate().for_each(|(k,row)| { for c in 0..nc {
        if zg[k][c] > zd+T::number(0.1) {
            let a = (half*pi*(zg[k][c]-zd)/(top-zd)).sin();
            row[c] = T::zero()+amp*(a*a);
        }
    } });
    for array in [&zw,&dzw,&rdzw,&zu,&dzu,&rdzu,&rdzwp,&rdzwm,&fzp,&fzm,&ah] { write(out,array)?; }
    for matrix in [&hx,&zg,&zz,&zxu,&dss] { for row in matrix { write(out,row)?; } }
    write(out,&cf)?;
    out.write_all(&(kz as u64).to_le_bytes()).map_err(|e|e.to_string())
}

/// Omega edge metrics. Arithmetic deliberately matches multiply then divide,
/// rather than pre-dividing geometric factors. mpas_init_atm_cases.F:3387-3443.
fn edge<T: Real>(r: &Request, out: &mut impl Write) -> Result<(),String> {
    let n = r.levels;
    let nc = r.terrain.len();
    let ne = r.dc.len();
    if ![2,3,4].contains(&r.order) { return fail("theta advection order must be 2, 3 or 4"); }
    if r.zgrid.len() != n+1 || r.zgrid.iter().any(|row| row.len() != nc) {
        return fail("edge metric zgrid dimensions are inconsistent");
    }
    // maxEdges once: re-scanning every cell's count per edge side cost
    // nEdges*2*nCells compares, 7.5 min on one core at 30 km.
    let max_edges = r.counts.iter().copied().max().unwrap_or(0);
    if r.order != 2 && (r.deriv.len() != ne || r.deriv.par_iter().any(|sides|
        sides.len() != 2 || sides.iter().any(|row| row.len() <= max_edges))) {
        return fail("deriv_two has insufficient edge/side/neighbor coefficients");
    }
    let zg: Vec<Vec<T>> = r.zgrid.iter().map(|row| row.iter().map(|&v| T::number(v)).collect()).collect();
    let mut zb = vec![T::zero();(n+1)*2*ne];
    let mut zb3 = zb.clone();
    // One level per task: level k owns zb[(k*2+side)*ne+e] for both sides,
    // and level n stays the +0 the serial loop never wrote.
    zb.par_chunks_mut(2*ne).zip(zb3.par_chunks_mut(2*ne)).enumerate().take(n)
        .for_each(|(k,(zb,zb3))| for e in 0..ne {
        let [a,b] = r.cells_on_edge[e];
        let c1 = if a < 0 { b as usize } else { a as usize };
        let c2 = if b < 0 { c1 } else { b as usize };
        {
            // MPAS-A v8.4.2 mpas_init_atm_cases.F:3407-3438 does not set z_edge3 in
            // order2. The pinned strict single-precision executable leaves
            // four blank-character bytes (0x20202020) here: repeated 1/2-rank
            // runs on both hosts and the legacy/unsmoothed branch agree.
            // Reproduce those measured words, including both rounded
            // geometry operations, rather than repairing the native defect.
            let mut z3 = if r.order == 2 {
                T::number(f32::from_bits(0x20202020) as f64)
            } else {T::zero()};
            let z = if r.order == 2 { (zg[k][c1]+zg[k][c2])/T::number(2.0) } else {
                let mut d = [T::zero();2];
                for (side,c) in [c1,c2].into_iter().enumerate() {
                    d[side] = T::number(r.deriv[e][side][0])*zg[k][c];
                    for j in 0..r.counts[c] {
                        let neighbor = r.cells_on_cell[c][j];
                        if neighbor >= 0 {
                            d[side] = d[side]+T::number(r.deriv[e][side][j+1])*zg[k][neighbor as usize];
                        }
                    }
                }
                let dc = T::number(r.dc[e]);
                if r.order == 3 { z3 = -(dc*dc)*(d[0]-d[1])/T::number(12.0); }
                T::half()*(zg[k][c1]+zg[k][c2])-(dc*dc)*(d[0]+d[1])/T::number(12.0)
            };
            for (side,c) in [c1,c2].into_iter().enumerate() {
                let index = side*ne+e;
                zb[index] = (z-zg[k][c])*T::number(r.dv[e])/T::number(r.area[c]);
                zb3[index] = z3*T::number(r.dv[e])/T::number(r.area[c]);
            }
        }
    });
    write(out,&zb)?;
    write(out,&zb3)
}

/// Versioned JSON topology/parameter input and raw IEEE output. The binary
/// protocol preserves every field word, including unused +0 slots.
pub fn run(mut reader: impl Read, writer: impl Write) -> Result<(),String> {
    let _abi = std::hint::black_box(ABI_MARKER);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    let r: Request = serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
    process_request(r, writer)
}

/// v2 header: every v1 scalar plus one descriptor per array, in stream order.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderV2 {
    mode: String, width: usize, levels: usize, top: f64, scheme: String,
    specified: Option<Vec<f64>>, projection: String, terrain_passes: usize, smooth: bool,
    surface_passes: usize, minimum_fraction: f64, hybrid: bool, transition: f64,
    damping: f64, damping_start: f64, regional_halo: bool, order: i32,
    #[serde(default)]
    boundary: Option<Boundary>,
    arrays: Vec<ArrayV2>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArrayV2 { name: String, dtype: String, shape: Vec<usize> }

const V2_ARRAYS: [&str; 10] = ["terrain","counts","edges_on_cell","cells_on_cell","cells_on_edge",
    "dc","dv","area","deriv","zgrid"];

fn v2_floats(bytes: &[u8], dtype: &str, name: &str) -> Result<Vec<f64>,String> {
    // A JSON number was the f64 nearest the sender's value; an f4 word widens
    // to exactly the double Python's float() of it printed.
    match dtype {
        "<f8" => Ok(bytes.par_chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect()),
        "<f4" => Ok(bytes.par_chunks_exact(4).map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap()))).collect()),
        _ => Err(format!("vertical v2 array {name} must be <f4 or <f8, got {dtype}")),
    }
}

fn v2_ints(bytes: &[u8], dtype: &str, name: &str) -> Result<Vec<i64>,String> {
    match dtype {
        "<i8" => Ok(bytes.par_chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()),
        "<i4" => Ok(bytes.par_chunks_exact(4).map(|b| i64::from(i32::from_le_bytes(b.try_into().unwrap()))).collect()),
        _ => Err(format!("vertical v2 array {name} must be <i4 or <i8, got {dtype}")),
    }
}

fn v2_rows<T: Copy + Send + Sync>(flat: Vec<T>, shape: &[usize], name: &str) -> Result<Vec<Vec<T>>,String> {
    match shape {
        [0] => Ok(Vec::new()),
        [n, 0] => Ok(vec![Vec::new(); *n]),
        [_, m] => Ok(flat.par_chunks_exact(*m).map(<[T]>::to_vec).collect()),
        _ => Err(format!("vertical v2 array {name} must be two-dimensional")),
    }
}

fn v2_vector<T>(flat: Vec<T>, shape: &[usize], name: &str) -> Result<Vec<T>,String> {
    if shape.len() != 1 { return Err(format!("vertical v2 array {name} must be one-dimensional")); }
    Ok(flat)
}

fn read_v2(reader: &mut impl Read) -> Result<Request,String> {
    let io = |e: std::io::Error| format!("vertical v2 stream: {e}");
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(io)?;
    if &magic != V2_MAGIC { return Err("expected the HEXVRT2 vertical framing".into()); }
    let mut length = [0u8; 8];
    reader.read_exact(&mut length).map_err(io)?;
    let mut header = vec![0u8; u64::from_le_bytes(length) as usize];
    reader.read_exact(&mut header).map_err(io)?;
    let h: HeaderV2 = serde_json::from_slice(&header).map_err(|e|e.to_string())?;
    let mut names: Vec<&str> = h.arrays.iter().map(|a| a.name.as_str()).collect();
    names.sort_unstable();
    let mut wanted = V2_ARRAYS.to_vec();
    wanted.sort_unstable();
    if names != wanted { return Err(format!("vertical v2 arrays must be exactly {V2_ARRAYS:?}")); }
    let (mut terrain, mut counts, mut edges_on_cell, mut cells_on_cell, mut cells_on_edge) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut dc, mut dv, mut area, mut deriv, mut zgrid) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for a in &h.arrays {
        let size = match a.dtype.as_str() { "<f8" | "<i8" => 8, "<f4" | "<i4" => 4,
            other => return Err(format!("vertical v2 array {} has unsupported dtype {other}", a.name)) };
        let count = a.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| format!("vertical v2 array {} is too large", a.name))?;
        let mut bytes = vec![0u8; count*size];
        reader.read_exact(&mut bytes).map_err(io)?;
        let (name, shape, dtype) = (a.name.as_str(), a.shape.as_slice(), a.dtype.as_str());
        match name {
            "terrain" => terrain = v2_vector(v2_floats(&bytes, dtype, name)?, shape, name)?,
            "dc" => dc = v2_vector(v2_floats(&bytes, dtype, name)?, shape, name)?,
            "dv" => dv = v2_vector(v2_floats(&bytes, dtype, name)?, shape, name)?,
            "area" => area = v2_vector(v2_floats(&bytes, dtype, name)?, shape, name)?,
            "counts" => {
                let flat = v2_vector(v2_ints(&bytes, dtype, name)?, shape, name)?;
                if flat.iter().any(|&n| n < 0) { return Err("vertical v2 counts must be non-negative".into()); }
                counts = flat.into_iter().map(|n| n as usize).collect();
            }
            "edges_on_cell" => edges_on_cell = v2_rows(v2_ints(&bytes, dtype, name)?, shape, name)?,
            "cells_on_cell" => cells_on_cell = v2_rows(v2_ints(&bytes, dtype, name)?, shape, name)?,
            "cells_on_edge" => {
                if shape.len() != 2 || shape[1] != 2 { return Err("vertical v2 cells_on_edge must be (nEdges, 2)".into()); }
                cells_on_edge = v2_ints(&bytes, dtype, name)?.par_chunks_exact(2).map(|p| [p[0], p[1]]).collect();
            }
            "zgrid" => zgrid = v2_rows(v2_floats(&bytes, dtype, name)?, shape, name)?,
            "deriv" => deriv = match shape {
                [0] => Vec::new(),
                [n, 2, 0] => vec![vec![Vec::new(), Vec::new()]; *n],
                [_, 2, k] => v2_floats(&bytes, dtype, name)?.par_chunks_exact(2*k)
                    .map(|e| e.chunks_exact(*k).map(<[f64]>::to_vec).collect()).collect(),
                _ => return Err("vertical v2 deriv must be (nEdges, 2, nCoefficients)".into()),
            },
            _ => unreachable!(),
        }
    }
    let mut tail = [0u8; 1];
    if reader.read(&mut tail).map_err(io)? != 0 { return Err("vertical v2 stream has bytes after its arrays".into()); }
    Ok(Request {
        mode: h.mode, width: h.width, terrain, counts, edges_on_cell, cells_on_cell, cells_on_edge,
        dc, dv, area, levels: h.levels, top: h.top, scheme: h.scheme, specified: h.specified,
        projection: h.projection, terrain_passes: h.terrain_passes, smooth: h.smooth,
        surface_passes: h.surface_passes, minimum_fraction: h.minimum_fraction, hybrid: h.hybrid,
        transition: h.transition, damping: h.damping, damping_start: h.damping_start,
        regional_halo: h.regional_halo, order: h.order, deriv, zgrid, boundary: h.boundary,
    })
}

/// hex-vertical-v2: the v1 request with raw array framing; identical output.
pub fn run_v2(mut reader: impl Read, writer: impl Write) -> Result<(),String> {
    let _abi = std::hint::black_box(ABI_MARKER_V2);
    let r = read_v2(&mut reader)?;
    process_request(r, writer)
}

fn process_request(mut r: Request, mut writer: impl Write) -> Result<(),String> {
    if let Some(b) = &r.boundary {
        if r.width != 4 || !["grid","terrain"].contains(&r.mode.as_str()) {
            return fail("boundary blending uses native single precision before terrain/grid construction");
        }
        let field = super::metfile::read_first_named_field(std::path::Path::new(&b.met_path),"SOILHGT")
            .map_err(|e|e.to_string())?.ok_or("SOILHGT field not found for boundary blending")?;
        let mut terrain: Vec<f32> = r.terrain.iter().map(|&v|v as f32).collect();
        super::boundary::blend_bdy_terrain(&field,&b.lat,&b.lon,&b.mask,&mut terrain)?;
        r.terrain=terrain.into_iter().map(f64::from).collect();
    }
    validate(&r)?;
    fn process<T: Real>(r: &Request, out: &mut impl Write) -> Result<(),String> {
        match r.mode.as_str() {
            "terrain" => write(out,&smooth_terrain::<T>(r,&Stencil::new(r))),
            "grid" => grid::<T>(r,out),
            "edge" => edge::<T>(r,out),
            _ => fail("vertical mode must be terrain, grid or edge"),
        }
    }
    match r.width {
        4 => process::<f32>(&r,&mut writer),
        8 => process::<f64>(&r,&mut writer),
        _ => fail("vertical width must be float32 or float64"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unused_connectivity_padding_accepts_native_negative_sentinels() {
        let request=serde_json::json!({"mode":"terrain","width":4,"terrain":[0.,1.],"counts":[1,1],
            "edges_on_cell":[[0,-1],[0,-1]],"cells_on_cell":[[1,-1],[0,-1]],"cells_on_edge":[[0,1]],
            "dc":[1.],"dv":[1.],"area":[1.,1.],"levels":3,"top":30000.,"scheme":"tc","specified":null,
            "projection":"linear_interpolation","terrain_passes":0,"smooth":false,"surface_passes":0,
            "minimum_fraction":0.3,"hybrid":false,"transition":30000.,"damping":0.,"damping_start":22000.,
            "regional_halo":false,"order":3,"deriv":[],"zgrid":[]});
        let mut out=Vec::new();run(serde_json::to_vec(&request).unwrap().as_slice(),&mut out).unwrap();
        assert_eq!(out,[0f32.to_le_bytes(),1f32.to_le_bytes()].concat());
        let mut framed=Vec::new();
        let header=serde_json::json!({"mode":"terrain","width":4,"levels":3,"top":30000.,"scheme":"tc",
            "specified":null,"projection":"linear_interpolation","terrain_passes":0,"smooth":false,
            "surface_passes":0,"minimum_fraction":0.3,"hybrid":false,"transition":30000.,"damping":0.,
            "damping_start":22000.,"regional_halo":false,"order":3,"arrays":[
            {"name":"terrain","dtype":"<f4","shape":[2]},{"name":"counts","dtype":"<i4","shape":[2]},
            {"name":"edges_on_cell","dtype":"<i8","shape":[2,2]},{"name":"cells_on_cell","dtype":"<i4","shape":[2,2]},
            {"name":"cells_on_edge","dtype":"<i8","shape":[1,2]},{"name":"dc","dtype":"<f8","shape":[1]},
            {"name":"dv","dtype":"<f4","shape":[1]},{"name":"area","dtype":"<f8","shape":[2]},
            {"name":"deriv","dtype":"<f8","shape":[0]},{"name":"zgrid","dtype":"<f8","shape":[0]}]});
        let header=serde_json::to_vec(&header).unwrap();
        framed.extend_from_slice(V2_MAGIC);framed.extend_from_slice(&(header.len() as u64).to_le_bytes());
        framed.extend_from_slice(&header);
        for v in [0f32,1.] {framed.extend_from_slice(&v.to_le_bytes());}
        for v in [1i32,1] {framed.extend_from_slice(&v.to_le_bytes());}
        for v in [0i64,-1,0,-1] {framed.extend_from_slice(&v.to_le_bytes());}
        for v in [1i32,-1,0,-1] {framed.extend_from_slice(&v.to_le_bytes());}
        for v in [0i64,1] {framed.extend_from_slice(&v.to_le_bytes());}
        framed.extend_from_slice(&1f64.to_le_bytes());framed.extend_from_slice(&1f32.to_le_bytes());
        for v in [1f64,1.] {framed.extend_from_slice(&v.to_le_bytes());}
        let mut out2=Vec::new();run_v2(framed.as_slice(),&mut out2).unwrap();
        assert_eq!(out2,out);
        let mut invalid=request;invalid["edges_on_cell"][0][0]=serde_json::json!(-1);
        assert!(run(serde_json::to_vec(&invalid).unwrap().as_slice(),Vec::new()).unwrap_err().contains("out-of-range"));
    }
}
