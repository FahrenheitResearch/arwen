//! Group B of the clean-room post: pressure levels, sea-level pressure and
//! column water (specification section 5).
//!
//! The CPU functions in this file are the float64 reference.  The CUDA
//! kernels (kernels/group_b.cu, driven by `gpu_b`) perform the same
//! operations in the same order; `compute` with `Device::Gpu` and
//! `Device::Cpu` return identical bits.
//!
//! Methods, per field:
//! - preparation: none of its own.  Pressure (mass and interface), interface
//!   height, T, Tv, q and the destaggered winds are Group A's shared column
//!   state (specification 3.4, [`crate::state`]), read as binary64.
//! - temperature, q, u, v above the lowest mass level: linear in ln p
//!   between the bracketing mass levels [R12].
//! - height above ground: hypsometric from the nearest (in ln p) bracketing
//!   interface with the layer's virtual temperature [R2], [R12].
//! - between the lowest mass level and the ground: temperature linear in
//!   ln p toward the ground value T* = T_1 (1 + alpha (p_g / p_1 - 1));
//!   humidity (as RH) and wind held at the lowest level.
//! - below ground: NCAR/TN-396 [R12] temperature (eq. 16) and height
//!   (eq. 15, in virtual temperature); RH and wind held (decision D7).
//! - 1000 hPa height below ground: membrane reduction [R14] (decision D8).
//! - dewpoint and RH from q, T and p w.r.t. liquid water [R6], [R8];
//!   dewpoint never above temperature.
//! - mslp: the NMC reduction [R15]: ground Tv from the lowest level at the
//!   standard lapse rate, sea-level T0 = Ts + gamma zs with the published
//!   warm treatment (T0 = 290.66 K when Ts <= 290.66 < T0, else
//!   290.66 - 0.005 (Ts - 290.66)^2 when both are warmer), then
//!   p_msl = p_s exp(g zs / (Rd (Ts + T0) / 2)).
//! - maps_mslp [R17]: 700 hPa temperature (as above), extrapolated to the
//!   ground at the standard lapse rate, p_msl = p_s (1 + gamma zs / Ts)^(g / (Rd gamma)),
//!   then `consts::maps_smoothing_passes(DX)` 1-2-1 passes [R18].
//! - pwat [R2]: (1/g) sum q (p_int,k - p_int,k+1), surface to top, vapour only.

pub mod consts;
pub mod frame;
pub mod membrane;
pub mod smooth;
pub mod thermo;

#[cfg(any(windows, target_os = "linux"))]
pub mod gpu;

use rayon::prelude::*;

use crate::plev::consts::*;
use crate::plev::frame::Frame;
use crate::state::{InterfaceState, MassState};
use crate::plev::thermo::*;
use crate::PostDevice as Device;

/// Pressure-level fields in output order.
pub const PLEV_FIELDS: [&str; 6] = ["height", "temperature", "dewpoint", "rh", "u", "v"];
/// GRIB2 (discipline.category.parameter) and units of [`PLEV_FIELDS`].
pub const PLEV_GRIB: [(&str, &str); 6] = [
    ("0.3.5", "gpm"),
    ("0.0.0", "K"),
    ("0.0.6", "K"),
    ("0.1.1", "%"),
    ("0.2.2", "m s-1"),
    ("0.2.3", "m s-1"),
];

/// Indices of the 2D planes after the pressure-level block of `out32`.
pub const P2_PWAT: usize = 0;
pub const P2_MSLP: usize = 1;
pub const P2_T700: usize = 2;
pub const P2_MAPS: usize = 3;
pub const P2_MEMBRANE_SLP: usize = 4;
pub const N2D: usize = 5;
/// Planes after the membrane block of `out64`.
pub const Q_PGROUND: usize = 0;
pub const Q_TVGROUND: usize = 1;
pub const Q_MAPS_RAW: usize = 2;
pub const N64_EXTRA: usize = 3;

/// The largest level count the CUDA column kernel holds per thread.
pub const GPU_NZ_MAX: usize = 160;

#[derive(Debug, Clone)]
pub struct Request {
    /// Output pressure levels (Pa), any order.
    pub levels_pa: Vec<f64>,
    pub device: Device,
    /// CUDA ordinal for `Device::Gpu`/`Auto`.
    pub ordinal: usize,
}

#[derive(Debug, Clone)]
pub struct Output {
    pub nx: usize,
    pub ny: usize,
    pub levels_pa: Vec<f64>,
    /// `[level][field][cell]`, fields in [`PLEV_FIELDS`] order, f32, NaN missing.
    pub plev: Vec<f32>,
    pub pwat: Vec<f32>,
    pub mslp: Vec<f32>,
    pub maps_mslp: Vec<f32>,
    /// Internal: the membrane-reduced sea-level pressure [R14].
    pub membrane_slp: Vec<f32>,
    /// Internal: the 700 hPa temperature MAPS starts from.
    pub t700: Vec<f32>,
    pub maps_passes: usize,
    pub membrane_active_levels: usize,
    pub device: String,
    pub vertical_source: String,
    pub pressure_source: String,
    /// Wall-clock seconds per phase.
    pub timings: Vec<(&'static str, f64)>,
}

impl Output {
    pub fn plev_plane(&self, level: usize, field: usize) -> &[f32] {
        let n = self.nx * self.ny;
        let o = (level * 6 + field) * n;
        &self.plev[o..o + n]
    }
}

/// Raw output pointers shared by the rayon workers; each column writes only
/// its own cells.
#[derive(Clone, Copy)]
pub(crate) struct Sink {
    o32: *mut f32,
    o64: *mut f64,
    ncell: usize,
}
unsafe impl Send for Sink {}
unsafe impl Sync for Sink {}
impl Sink {
    #[inline]
    fn put32(&self, plane: usize, cell: usize, v: f64) {
        let x = if v.is_nan() { f32::NAN } else { v as f32 };
        unsafe { *self.o32.add(plane * self.ncell + cell) = x }
    }
    #[inline]
    fn put64(&self, plane: usize, cell: usize, v: f64) {
        unsafe { *self.o64.add(plane * self.ncell + cell) = v }
    }
}

/// Per-thread column scratch.
pub(crate) struct Scratch {
    pi: Vec<f64>,
    lnpi: Vec<f64>,
    zi: Vec<f64>,
    pm: Vec<f64>,
    lnp: Vec<f64>,
    tk: Vec<f64>,
    tv: Vec<f64>,
    q: Vec<f64>,
    uu: Vec<f64>,
    vv: Vec<f64>,
}

impl Scratch {
    fn new(nz: usize) -> Self {
        let z = |n| vec![0.0f64; n];
        Self {
            pi: z(nz + 1),
            lnpi: z(nz + 1),
            zi: z(nz + 1),
            pm: z(nz),
            lnp: z(nz),
            tk: z(nz),
            tv: z(nz),
            q: z(nz),
            uu: z(nz),
            vv: z(nz),
        }
    }
}

#[inline]
fn clamp0(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Linear-in-ln-p interpolation of `a` at `lnp_t` between mass levels k,
/// k + 1 found by scanning up from the ground (pressure falls with k).
#[inline]
fn interp_mass(s: &Scratch, nz: usize, p_t: f64, lnp_t: f64, a: &[f64]) -> f64 {
    let mut k = 0;
    while k + 1 < nz {
        if p_t <= s.pm[k] && p_t >= s.pm[k + 1] {
            let w = (lnp_t - s.lnp[k]) / (s.lnp[k + 1] - s.lnp[k]);
            return a[k] + w * (a[k + 1] - a[k]);
        }
        k += 1;
    }
    f64::NAN
}

/// Temperature (K) at `p_t` for any target at or below the top mass level.
#[inline]
fn temperature_at(s: &Scratch, nz: usize, p_t: f64, lnp_t: f64, t_star: f64, pg: f64, zs: f64) -> f64 {
    if p_t < s.pm[nz - 1] {
        f64::NAN
    } else if p_t <= s.pm[0] {
        interp_mass(s, nz, p_t, lnp_t, &s.tk)
    } else if p_t <= pg {
        let w = (lnp_t - s.lnp[0]) / (s.lnpi[0] - s.lnp[0]);
        s.tk[0] + w * (t_star - s.tk[0])
    } else {
        tn396_temperature(t_star, pg, zs, p_t)
    }
}

/// Virtual temperature (K) at `p_t`, the membrane's boundary values above
/// ground and its first guess below.
#[inline]
fn virtual_temperature_at(s: &Scratch, nz: usize, p_t: f64, lnp_t: f64, tv_star: f64, pg: f64, zs: f64) -> f64 {
    if p_t < s.pm[nz - 1] {
        f64::NAN
    } else if p_t <= s.pm[0] {
        interp_mass(s, nz, p_t, lnp_t, &s.tv)
    } else if p_t <= pg {
        let w = (lnp_t - s.lnp[0]) / (s.lnpi[0] - s.lnp[0]);
        s.tv[0] + w * (tv_star - s.tv[0])
    } else {
        tn396_temperature(tv_star, pg, zs, p_t)
    }
}

/// The column function: everything Group B reads off one column.
#[allow(clippy::too_many_arguments)]
pub(crate) fn column(
    f: &Frame,
    levels: &[f64],
    mem_levels: &[f64],
    cell: usize,
    s: &mut Scratch,
    out: Sink,
) {
    let (nx, ny, nz) = (f.nx, f.ny, f.nz);
    let n2 = nx * ny;
    let nl = levels.len();
    let nm = mem_levels.len();
    let base2 = 6 * nl;

    // Required inputs must be finite; a missing cell makes only its own
    // outputs missing.
    let sp = f.mass(MassState::P);
    let stk = f.mass(MassState::Tk);
    let stv = f.mass(MassState::Tv);
    let sq = f.mass(MassState::Q);
    let su = f.mass(MassState::U);
    let sv = f.mass(MassState::V);
    let pint = f.iface(InterfaceState::PInt);
    let zint = f.iface(InterfaceState::ZInt);
    let mut ok = f.psfc[cell].is_finite() && f.hgt[cell].is_finite();
    for k in 0..nz {
        let c = k * n2 + cell;
        ok = ok
            && sp[c].is_finite()
            && stk[c].is_finite()
            && stv[c].is_finite()
            && sq[c].is_finite()
            && su[c].is_finite()
            && sv[c].is_finite();
    }
    for k in 0..=nz {
        ok = ok && pint[k * n2 + cell].is_finite() && zint[k * n2 + cell].is_finite();
    }
    if !ok {
        for p in 0..base2 + N2D {
            if p != base2 + P2_MAPS && p != base2 + P2_MEMBRANE_SLP {
                out.put32(p, cell, f64::NAN);
            }
        }
        for p in 0..nm + N64_EXTRA {
            out.put64(p, cell, f64::NAN);
        }
        return;
    }

    // The column from the shared state, in binary64.
    for k in 0..=nz {
        s.pi[k] = pint[k * n2 + cell] as f64;
        s.lnpi[k] = ln(s.pi[k]);
        s.zi[k] = zint[k * n2 + cell] as f64;
    }
    for k in 0..nz {
        let c = k * n2 + cell;
        s.pm[k] = sp[c] as f64;
        s.tk[k] = stk[c] as f64;
        s.tv[k] = stv[c] as f64;
        s.q[k] = sq[c] as f64;
        s.lnp[k] = ln(s.pm[k]);
        s.uu[k] = su[c] as f64;
        s.vv[k] = sv[c] as f64;
    }

    let pg = s.pi[0];
    let zs = f.hgt[cell] as f64;
    let ratio = pg / s.pm[0] - 1.0;
    let t_star = s.tk[0] * (1.0 + ALPHA * ratio);
    let tv_star = s.tv[0] * (1.0 + ALPHA * ratio);
    let e_bot = vapour_pressure(s.q[0], s.pm[0]);
    let mut rh_bot = e_bot / esat_water(s.tk[0]);
    if rh_bot > 1.0 {
        rh_bot = 1.0;
    }

    // Pressure levels.
    for (l, &pt) in levels.iter().enumerate() {
        let lnpt = ln(pt);
        let t;
        let e;
        let u;
        let v;
        if pt < s.pm[nz - 1] {
            t = f64::NAN;
            e = f64::NAN;
            u = f64::NAN;
            v = f64::NAN;
        } else if pt <= s.pm[0] {
            t = interp_mass(s, nz, pt, lnpt, &s.tk);
            let q = interp_mass(s, nz, pt, lnpt, &s.q);
            e = vapour_pressure(q, pt);
            u = interp_mass(s, nz, pt, lnpt, &s.uu);
            v = interp_mass(s, nz, pt, lnpt, &s.vv);
        } else {
            t = temperature_at(s, nz, pt, lnpt, t_star, pg, zs);
            e = rh_bot * esat_water(t);
            u = s.uu[0];
            v = s.vv[0];
        }
        let (td, rh) = if t.is_nan() {
            (f64::NAN, f64::NAN)
        } else {
            let td0 = dewpoint(e);
            (if td0 > t { t } else { td0 }, rh_percent(e, t))
        };
        // Height.
        let z = if pt < s.pi[nz] {
            f64::NAN
        } else if pt <= pg {
            let mut kk = 0;
            while kk + 1 < nz + 1 && !(pt <= s.pi[kk] && pt >= s.pi[kk + 1]) {
                kk += 1;
            }
            let down = s.lnpi[kk] - lnpt;
            let up = lnpt - s.lnpi[kk + 1];
            let a = RD * s.tv[kk] / G;
            if down <= up { s.zi[kk] + a * down } else { s.zi[kk + 1] - a * up }
        } else {
            tn396_height(tv_star, pg, zs, pt)
        };
        let o = l * 6;
        out.put32(o, cell, z);
        out.put32(o + 1, cell, t);
        out.put32(o + 2, cell, td);
        out.put32(o + 3, cell, rh);
        out.put32(o + 4, cell, u);
        out.put32(o + 5, cell, v);
    }

    // Precipitable water [R2].
    let mut pw = 0.0;
    for k in 0..nz {
        pw = pw + s.q[k] * (s.pi[k] - s.pi[k + 1]);
    }
    out.put32(base2 + P2_PWAT, cell, pw / G);

    // NMC sea-level pressure [R15].
    let ps = f.psfc[cell] as f64;
    let tsv = s.tv[0] * pow(ps / s.pm[0], ALPHA);
    let mut t0 = tsv + GAMMA * zs;
    if t0 > NMC_T_WARM {
        if tsv <= NMC_T_WARM {
            t0 = NMC_T_WARM;
        } else {
            let d = tsv - NMC_T_WARM;
            t0 = NMC_T_WARM - NMC_WARM_COEF * d * d;
        }
    }
    let tm = 0.5 * (tsv + t0);
    out.put32(base2 + P2_MSLP, cell, ps * exp(G * zs / (RD * tm)));

    // MAPS sea-level pressure [R17], before smoothing.
    let t700 = temperature_at(s, nz, MAPS_P_REF, ln(MAPS_P_REF), t_star, pg, zs);
    out.put32(base2 + P2_T700, cell, t700);
    let tsm = t700 * pow(ps / MAPS_P_REF, ALPHA);
    let maps = ps * pow(1.0 + GAMMA * zs / tsm, G / (RD * GAMMA));
    out.put64(nm + Q_MAPS_RAW, cell, maps);

    // Membrane inputs [R14].
    for (m, &pt) in mem_levels.iter().enumerate() {
        out.put64(m, cell, virtual_temperature_at(s, nz, pt, ln(pt), tv_star, pg, zs));
    }
    out.put64(nm + Q_PGROUND, cell, pg);
    out.put64(nm + Q_TVGROUND, cell, tv_star);
}

/// Column pass on the CPU: fills `out32` (pressure levels and the 2D
/// planes except MAPS and membrane SLP) and `out64` (membrane inputs,
/// ground pressure and Tv, raw MAPS).
pub(crate) fn column_pass_cpu(
    f: &Frame,
    levels: &[f64],
    mem_levels: &[f64],
    out32: &mut [f32],
    out64: &mut [f64],
) {
    let ncell = f.ncell();
    let sink = Sink { o32: out32.as_mut_ptr(), o64: out64.as_mut_ptr(), ncell };
    (0..ncell).into_par_iter().for_each_init(
        || Scratch::new(f.nz),
        |s, cell| column(f, levels, mem_levels, cell, s, sink),
    );
}

/// The membrane column after the solve: heights of underground membrane
/// levels integrated down from the ground with the layer-mean virtual
/// temperature, the membrane sea-level pressure, and the 1000 hPa height
/// where 1000 hPa is underground.  Returns (membrane slp, z at 1000 hPa
/// or NaN when 1000 hPa is above ground).
#[inline]
pub fn membrane_column(mem_levels: &[f64], tv_of: impl Fn(usize) -> f64, pg: f64, tvg: f64, zs: f64) -> (f64, f64) {
    let mut p_prev = pg;
    let mut z_prev = zs;
    let mut tv_prev = tvg;
    let mut slp = f64::NAN;
    let mut z1000 = f64::NAN;
    if zs <= 0.0 {
        slp = pg * exp(G * zs / (RD * tvg));
    }
    for (m, &pt) in mem_levels.iter().enumerate() {
        if pt <= pg {
            continue;
        }
        let tvm = tv_of(m);
        let tbar = 0.5 * (tv_prev + tvm);
        let z = z_prev - RD * tbar * ln(pt / p_prev) / G;
        if slp.is_nan() && z <= 0.0 {
            slp = p_prev * exp(G * z_prev / (RD * tbar));
        }
        if pt == 100000.0 {
            z1000 = z;
        }
        p_prev = pt;
        z_prev = z;
        tv_prev = tvm;
    }
    if slp.is_nan() {
        slp = p_prev * exp(G * z_prev / (RD * tv_prev));
    }
    (slp, z1000)
}

/// Compute every Group B field for one frame.  `gpu` is the opened kernel
/// set for `Device::Gpu` and `Device::Auto` (Auto falls back to the CPU
/// when it is `None` or the device path fails).
#[cfg(any(windows, target_os = "linux"))]
pub fn compute(f: &Frame, req: &Request, gpu: Option<&crate::gpu::GpuPost>) -> Result<Output, String> {
    f.validate()?;
    if req.levels_pa.iter().any(|&p| !(p > 0.0)) {
        return Err("pressure levels must be positive".into());
    }
    match (req.device, gpu) {
        (Device::Cpu, _) => compute_cpu(f, &req.levels_pa),
        (Device::Gpu, Some(g)) => crate::plev::gpu::compute(f, &req.levels_pa, g),
        (Device::Gpu, None) => Err("Device::Gpu needs an opened GPU".into()),
        (Device::Auto, Some(g)) => match crate::plev::gpu::compute(f, &req.levels_pa, g) {
            Ok(o) => Ok(o),
            Err(_) => compute_cpu(f, &req.levels_pa),
        },
        (Device::Auto, None) => compute_cpu(f, &req.levels_pa),
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn compute(f: &Frame, req: &Request) -> Result<Output, String> {
    f.validate()?;
    match req.device {
        Device::Gpu => Err("the CUDA post path is built for Windows and Linux only".into()),
        _ => compute_cpu(f, &req.levels_pa),
    }
}

pub(crate) fn pressure_source(f: &Frame) -> String {
    f.pressure_source.to_owned()
}

/// CPU reference path.
pub fn compute_cpu(f: &Frame, levels: &[f64]) -> Result<Output, String> {
    let ncell = f.ncell();
    let nl = levels.len();
    let mem_levels = membrane_levels();
    let nm = mem_levels.len();
    let base2 = 6 * nl;
    let mut out32 = vec![f32::NAN; (base2 + N2D) * ncell];
    let mut out64 = vec![f64::NAN; (nm + N64_EXTRA) * ncell];
    let mut timings: Vec<(&'static str, f64)> = Vec::new();
    let mut clock = std::time::Instant::now();
    column_pass_cpu(f, levels, &mem_levels, &mut out32, &mut out64);
    timings.push(("column pass", clock.elapsed().as_secs_f64()));
    clock = std::time::Instant::now();

    // Membrane solve on every level with an underground cell.
    let pg = &out64[(nm + Q_PGROUND) * ncell..(nm + Q_PGROUND + 1) * ncell];
    let active = membrane::active_levels(&mem_levels, pg);
    let mut solved: Vec<Vec<f64>> = vec![Vec::new(); nm];
    for &m in &active {
        let tv = &out64[m * ncell..(m + 1) * ncell];
        let (vals, mask) = membrane::level_inputs(tv, pg, mem_levels[m]);
        solved[m] = membrane::solve_cpu(f.nx, f.ny, vals, mask);
    }
    let i1000 = levels.iter().position(|&p| p == 100000.0);
    {
        let tvg = &out64[(nm + Q_TVGROUND) * ncell..(nm + Q_TVGROUND + 1) * ncell];
        let sink = Sink { o32: out32.as_mut_ptr(), o64: std::ptr::null_mut(), ncell };
        let solved = &solved;
        let mem_levels = &mem_levels;
        (0..ncell).into_par_iter().for_each(|cell| {
            let pgc = pg[cell];
            if pgc.is_nan() {
                sink.put32(base2 + P2_MEMBRANE_SLP, cell, f64::NAN);
                return;
            }
            let (slp, z1000) = membrane_column(
                mem_levels,
                |m| solved[m][cell],
                pgc,
                tvg[cell],
                f.hgt[cell] as f64,
            );
            sink.put32(base2 + P2_MEMBRANE_SLP, cell, slp);
            if let Some(l) = i1000 {
                if !z1000.is_nan() {
                    sink.put32(l * 6, cell, z1000);
                }
            }
        });
    }

    timings.push(("membrane", clock.elapsed().as_secs_f64()));
    clock = std::time::Instant::now();
    // MAPS smoothing.
    let passes = maps_smoothing_passes(f.dx);
    let raw = out64[(nm + Q_MAPS_RAW) * ncell..(nm + Q_MAPS_RAW + 1) * ncell].to_vec();
    let smoothed = smooth::passes_cpu(f.nx, f.ny, raw, passes);
    for (cell, v) in smoothed.iter().enumerate() {
        out32[(base2 + P2_MAPS) * ncell + cell] = if v.is_nan() { f32::NAN } else { *v as f32 };
    }

    timings.push(("maps smoothing", clock.elapsed().as_secs_f64()));
    let mut o = split_output(f, levels, out32, passes, active.len(), "cpu".into());
    o.timings = timings;
    Ok(o)
}

pub(crate) fn split_output(
    f: &Frame,
    levels: &[f64],
    out32: Vec<f32>,
    passes: usize,
    active: usize,
    device: String,
) -> Output {
    let ncell = f.ncell();
    let base2 = 6 * levels.len();
    // The pressure-level block keeps its allocation; only the small 2D
    // tail is copied.
    let mut out32 = out32;
    let tail = out32.split_off(base2 * ncell);
    let plane = |p: usize| tail[p * ncell..(p + 1) * ncell].to_vec();
    Output {
        nx: f.nx,
        ny: f.ny,
        levels_pa: levels.to_vec(),
        pwat: plane(P2_PWAT),
        mslp: plane(P2_MSLP),
        maps_mslp: plane(P2_MAPS),
        membrane_slp: plane(P2_MEMBRANE_SLP),
        t700: plane(P2_T700),
        plev: out32,
        maps_passes: passes,
        membrane_active_levels: active,
        device,
        vertical_source: f.vertical_source.to_owned(),
        pressure_source: pressure_source(f),
        timings: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic isothermal, dry, calm column set on a small grid with a
    /// mountain in the middle: every quantity has a closed form.
    pub(crate) struct Synthetic {
        pub nx: usize,
        pub ny: usize,
        pub nz: usize,
        pub t: Vec<f32>,
        pub p: Vec<f32>,
        pub pb: Vec<f32>,
        pub qv: Vec<f32>,
        pub ph: Vec<f32>,
        pub phb: Vec<f32>,
        pub u: Vec<f32>,
        pub v: Vec<f32>,
        pub mu: Vec<f32>,
        pub mub: Vec<f32>,
        pub psfc: Vec<f32>,
        pub hgt: Vec<f32>,
        pub znw: Vec<f64>,
        pub znu: Vec<f64>,
    }

    /// Isothermal T0, terrain-following eta, pressure-consistent geopotential.
    pub(crate) fn synthetic(nx: usize, ny: usize, nz: usize, t0: f64, qv: f64) -> Synthetic {
        let ptop = 5000.0;
        let znw: Vec<f64> = (0..=nz).map(|k| 1.0 - k as f64 / nz as f64).collect();
        let znu: Vec<f64> = (0..nz).map(|k| 0.5 * (znw[k] + znw[k + 1])).collect();
        let n2 = nx * ny;
        let mut s = Synthetic {
            nx,
            ny,
            nz,
            t: vec![0.0; n2 * nz],
            p: vec![0.0; n2 * nz],
            pb: vec![0.0; n2 * nz],
            qv: vec![qv as f32; n2 * nz],
            ph: vec![0.0; n2 * (nz + 1)],
            phb: vec![0.0; n2 * (nz + 1)],
            u: vec![5.0; (nx + 1) * ny * nz],
            v: vec![-3.0; nx * (ny + 1) * nz],
            mu: vec![0.0; n2],
            mub: vec![0.0; n2],
            psfc: vec![0.0; n2],
            hgt: vec![0.0; n2],
            znw: znw.clone(),
            znu: znu.clone(),
        };
        let r = qv;
        let tv = t0 * (1.0 + r / EPS) / (1.0 + r);
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                let dx = i as f64 - (nx as f64 - 1.0) / 2.0;
                let dy = j as f64 - (ny as f64 - 1.0) / 2.0;
                let zs = 1500.0 * (-(dx * dx + dy * dy) / 8.0).exp();
                // Moist ground pressure from an isothermal atmosphere over sea-level 101325.
                let pg = 101325.0 * (-G * zs / (RD * tv)).exp();
                let mu_d = (pg - ptop) / (1.0 + r);
                s.mub[c] = mu_d as f32;
                s.hgt[c] = zs as f32;
                s.psfc[c] = pg as f32;
                let mut pi = vec![0.0; nz + 1];
                pi[nz] = ptop;
                for k in (0..nz).rev() {
                    pi[k] = pi[k + 1] + (s.znw[k] - s.znw[k + 1]) * mu_d * (1.0 + r);
                }
                for k in 0..=nz {
                    let z = zs + RD * tv / G * (pg / pi[k]).ln();
                    s.phb[k * n2 + c] = (G * z) as f32;
                }
                for k in 0..nz {
                    let pm = pi[k + 1] + (s.znu[k] - s.znw[k + 1]) * mu_d * (1.0 + r);
                    s.pb[k * n2 + c] = pm as f32;
                    let theta = t0 * (P0 / pm).powf(KAPPA);
                    s.t[k * n2 + c] = (theta - 300.0) as f32;
                }
            }
        }
        s
    }

    /// Group A's shared state of a synthetic frame (terrain-following eta:
    /// C3F = ZNW, C4F = 0).
    pub(crate) fn state_of(s: &Synthetic) -> crate::state::ColumnState {
        use crate::state::{MassInput, Shape, StateInputs};
        let shape = Shape { nx: s.nx, ny: s.ny, nz: s.nz };
        let v = shape.mass_volume();
        let mut mass = vec![0.0f32; MassInput::COUNT * v];
        let mut present = 0u32;
        for (slot, src) in [(MassInput::T, &s.t), (MassInput::P, &s.p), (MassInput::PB, &s.pb), (MassInput::QVapor, &s.qv)] {
            mass[slot as usize * v..(slot as usize + 1) * v].copy_from_slice(src);
            present |= 1 << slot as u32;
        }
        let inp = StateInputs {
            shape,
            mass,
            present,
            ph: s.ph.clone(),
            phb: s.phb.clone(),
            u_stag: s.u.clone(),
            v_stag: s.v.clone(),
            mu: s.mu.clone(),
            mub: s.mub.clone(),
            c3f: s.znw.iter().map(|&x| x as f32).collect(),
            c4f: vec![0.0; s.nz + 1],
            p_top: 5000.0,
        };
        crate::state::build_cpu(&inp).unwrap()
    }

    pub(crate) fn frame<'a>(s: &'a Synthetic, st: &'a crate::state::ColumnState) -> Frame<'a> {
        Frame::new(st, &s.psfc, &s.hgt, 3000.0, "synthetic", "synthetic terrain-following eta")
    }

    #[test]
    fn isothermal_column_has_closed_forms() {
        let t0 = 270.0;
        let s = synthetic(9, 7, 40, t0, 0.0);
        let st = state_of(&s);
        let f = frame(&s, &st);
        let levels: Vec<f64> = vec![100000.0, 85000.0, 70000.0, 50000.0, 20000.0];
        let o = compute_cpu(&f, &levels).unwrap();
        let n = f.ncell();
        for c in 0..n {
            // pwat of a dry column is zero.
            assert_eq!(o.pwat[c], 0.0);
            for (l, &p) in levels.iter().enumerate() {
                let z_exact = RD * t0 / G * (101325.0f64 / p).ln();
                let pg = f.psfc[c] as f64;
                let z = o.plev_plane(l, 0)[c] as f64;
                let t = o.plev_plane(l, 1)[c] as f64;
                if p <= pg - 2000.0 {
                    assert!((z - z_exact).abs() < 0.05, "z {z} vs {z_exact} at {p} cell {c}");
                    assert!((t - t0).abs() < 1e-3, "t {t} at {p}");
                    assert!((o.plev_plane(l, 4)[c] - 5.0).abs() < 1e-5);
                    assert!((o.plev_plane(l, 5)[c] + 3.0).abs() < 1e-5);
                }
                assert!(o.plev_plane(l, 3)[c] >= 0.0 && o.plev_plane(l, 3)[c] <= 100.0);
            }
            // The isothermal NMC reduction recovers 101325 at low ground
            // within the lapse-rate difference.
            let zs = f.hgt[c] as f64;
            let slp = o.mslp[c] as f64;
            assert!(slp > 100800.0 && slp < 101900.0, "slp {slp} zs {zs}");
            assert!((o.membrane_slp[c] as f64 - 101325.0).abs() < 60.0, "membrane {} zs {zs}", o.membrane_slp[c]);
        }
    }

    /// The kernels carry the constants as literals; pin them to consts.rs.
    #[test]
    fn kernel_constants_match_the_reference() {
        let src = include_str!("../../kernels/group_b.cu");
        let def = |name: &str| -> String {
            let key = format!("#define {name} ");
            let line = src.lines().find(|l| l.starts_with(&key)).unwrap_or_else(|| panic!("{name}"));
            line[key.len()..].trim().to_string()
        };
        for (name, v) in [
            ("G", G),
            ("RD", RD),
            ("RV", RV),
            ("P0", P0),
            ("THETA_BASE", THETA_BASE),
            ("GAMMA", GAMMA),
            ("T_MELT", T_MELT),
            ("MAGNUS_A", MAGNUS_A),
            ("MAGNUS_B", MAGNUS_B),
            ("MAGNUS_C", MAGNUS_C),
            ("Q_FLOOR", Q_FLOOR),
            ("TN396_Z_LOW", TN396_Z_LOW),
            ("TN396_Z_HIGH", TN396_Z_HIGH),
            ("TN396_T_PLATEAU", TN396_T_PLATEAU),
            ("TN396_T_WARM", TN396_T_WARM),
            ("TN396_T_COLD", TN396_T_COLD),
            ("NMC_T_WARM", NMC_T_WARM),
            ("NMC_WARM_COEF", NMC_WARM_COEF),
            ("MAPS_P_REF", MAPS_P_REF),
        ] {
            assert_eq!(def(name).parse::<f64>().unwrap(), v, "{name}");
        }
        assert_eq!(def("CP"), "(7.0 * RD / 2.0)");
        assert_eq!(def("EPS"), "(RD / RV)");
        assert_eq!(def("KAPPA"), "(RD / CP)");
        assert_eq!(def("ALPHA"), "(GAMMA * RD / G)");
        assert_eq!(def("NZ_MAX").parse::<usize>().unwrap(), GPU_NZ_MAX);
        assert_eq!(def("N2D").parse::<usize>().unwrap(), N2D);
        assert_eq!(def("N64_EXTRA").parse::<usize>().unwrap(), N64_EXTRA);
    }

    /// CPU and GPU agree bit for bit (needs an NVIDIA driver and card:
    /// `cargo test -p rw-post -- --ignored`).
    #[test]
    #[ignore]
    fn gpu_matches_cpu_bitwise() {
        let mut s = synthetic(97, 83, 45, 262.0, 0.004);
        // Texture so every branch runs: warm and cold columns, winds,
        // moisture, a below-sea-level patch.
        for (c, x) in s.t.iter_mut().enumerate() {
            *x += ((c * 7919 % 1000) as f32) * 0.03;
        }
        for (c, x) in s.qv.iter_mut().enumerate() {
            *x *= 0.2 + ((c * 104729 % 1000) as f32) * 0.0016;
        }
        for (c, x) in s.u.iter_mut().enumerate() {
            *x += ((c * 31 % 97) as f32) * 0.1;
        }
        s.hgt[0] = -20.0;
        let st = state_of(&s);
        let f = frame(&s, &st);
        let levels: Vec<f64> = (0..37).map(|i| 10000.0 + 2500.0 * i as f64).collect();
        let cpu = compute_cpu(&f, &levels).unwrap();
        let g = crate::gpu::GpuPost::open(0).expect("an NVIDIA GPU");
        let gpu = crate::plev::gpu::compute(&f, &levels, &g).expect("GPU path");
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
        assert!(same(&cpu.plev, &gpu.plev), "pressure levels differ");
        assert!(same(&cpu.pwat, &gpu.pwat));
        assert!(same(&cpu.mslp, &gpu.mslp));
        assert!(same(&cpu.maps_mslp, &gpu.maps_mslp));
        assert!(same(&cpu.membrane_slp, &gpu.membrane_slp));
        assert!(cpu.membrane_active_levels > 0);
    }

    /// Pressure-level heights on an analytic moist isothermal atmosphere laid
    /// out on the hybrid coordinate WOOF histories use (HYBRID_OPT = 2,
    /// ETAC = 0.2, stretched ZNW, no stored C3F/C4F or P_HYD), over a
    /// 1500 m mountain.  The height of pressure p is Rd Tv0 / g ln(p_sl / p)
    /// exactly, and every pair of output levels must obey the hypsometric
    /// equation.
    ///
    /// Breakage this gate names (B1 of the 2.8.7 audit): a grade against the
    /// pre-holdout (2.8.5-method) exporter's output asked these heights to
    /// move by up to 15 m.  Run as a black box on this same kind of history
    /// (isothermal, hybrid eta, no P_HYD), that exporter placed 850, 500 and
    /// 100 hPa at -7.5, +6.5 and +10.3 m from the analytic answer, with a
    /// hypsometric break near 625 hPa, while this path is exact.  Any change
    /// that pulls the heights toward that output fails here.
    #[test]
    fn hybrid_moist_isothermal_heights_are_exact() {
        use crate::hybrid::InterfaceCoefficients;
        use crate::state::{MassInput, Shape, StateInputs};
        let (nx, ny, nz) = (9usize, 7usize, 49usize);
        let n2 = nx * ny;
        let ptop = 5000.0f32;
        let psl = 101325.0f64;
        let r = 0.008f64;
        let tv0 = 262.0f64;
        let fac = (1.0 + r / EPS) / (1.0 + r);
        let t0 = tv0 / fac;
        let h = RD * tv0 / G;
        // Stretched eta: thin layers near the ground, as in WOOF histories.
        let znw: Vec<f32> = (0..=nz)
            .map(|k| {
                let s = k as f64 / nz as f64;
                (1.0 - (0.35 * s + 0.65 * s * s)) as f32
            })
            .collect();
        let coeff = InterfaceCoefficients::resolve(nz, None, &znw, 2, Some(0.2), ptop).unwrap();
        let c3f: Vec<f64> = coeff.c3f.iter().map(|&x| x as f64).collect();
        let c4f: Vec<f64> = coeff.c4f.iter().map(|&x| x as f64).collect();
        let v = nz * n2;
        let mut mass = vec![0.0f32; MassInput::COUNT * v];
        let mut ph = vec![0.0f32; (nz + 1) * n2];
        let mut phb = vec![0.0f32; (nz + 1) * n2];
        let mut mub = vec![0.0f32; n2];
        let mut psfc = vec![0.0f32; n2];
        let mut hgt = vec![0.0f32; n2];
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                let dx = i as f64 - (nx as f64 - 1.0) / 2.0;
                let dy = j as f64 - (ny as f64 - 1.0) / 2.0;
                let zs = 1500.0 * (-(dx * dx + dy * dy) / 6.0).exp();
                let pg = psl * (-zs / h).exp();
                let mud = (pg - ptop as f64) / (1.0 + r);
                mub[c] = mud as f32;
                psfc[c] = pg as f32;
                hgt[c] = zs as f32;
                let pint: Vec<f64> = (0..=nz)
                    .map(|k| {
                        let pd = c3f[k] * mud + c4f[k] + ptop as f64;
                        ptop as f64 + (pd - ptop as f64) * (1.0 + r)
                    })
                    .collect();
                for k in 0..=nz {
                    let z = if k == 0 { zs } else { h * (psl / pint[k]).ln() };
                    phb[k * n2 + c] = (G * z) as f32;
                    ph[k * n2 + c] = 0.0;
                }
                for k in 0..nz {
                    let pm = 0.5 * (pint[k] + pint[k + 1]);
                    let at = k * n2 + c;
                    mass[MassInput::T as usize * v + at] = (t0 * (P0 / pm).powf(KAPPA) - 300.0) as f32;
                    mass[MassInput::PB as usize * v + at] = pm as f32;
                    mass[MassInput::QVapor as usize * v + at] = r as f32;
                }
            }
        }
        let present = (1 << MassInput::T as u32) | (1 << MassInput::P as u32) | (1 << MassInput::PB as u32) | (1 << MassInput::QVapor as u32);
        let inp = StateInputs {
            shape: Shape { nx, ny, nz },
            mass,
            present,
            ph,
            phb,
            u_stag: vec![4.0; (nx + 1) * ny * nz],
            v_stag: vec![-2.0; nx * (ny + 1) * nz],
            mu: vec![0.0; n2],
            mub,
            c3f: coeff.c3f.clone(),
            c4f: coeff.c4f.clone(),
            p_top: ptop,
        };
        let st = crate::state::build_cpu(&inp).unwrap();
        let f = Frame::new(&st, &psfc, &hgt, 3000.0, "synthetic", "synthetic hybrid eta");
        let levels: Vec<f64> = (0..37).map(|i| 10000.0 + 2500.0 * i as f64).collect();
        let o = compute_cpu(&f, &levels).unwrap();
        let mut worst = 0.0f64;
        let mut worst_thick = 0.0f64;
        for c in 0..n2 {
            let pg = psfc[c] as f64;
            for (l, &p) in levels.iter().enumerate() {
                if p >= pg {
                    continue;
                }
                let z = o.plev_plane(l, 0)[c] as f64;
                let err = (z - h * (psl / p).ln()).abs();
                worst = worst.max(err);
                if l + 1 < levels.len() && levels[l + 1] < pg {
                    let z2 = o.plev_plane(l + 1, 0)[c] as f64;
                    let thick = (z - z2) - h * (levels[l + 1] / p).ln();
                    worst_thick = worst_thick.max(thick.abs());
                }
            }
        }
        assert!(worst < 0.05, "height off the analytic answer by {worst} m");
        assert!(worst_thick < 0.05, "hypsometric thickness off by {worst_thick} m");
    }

    #[test]
    fn moist_pwat_matches_the_column_mass() {
        let r = 0.01;
        let s = synthetic(5, 4, 30, 285.0, r);
        let st = state_of(&s);
        let f = frame(&s, &st);
        let o = compute_cpu(&f, &[50000.0]).unwrap();
        for c in 0..f.ncell() {
            let pg = f.psfc[c] as f64;
            let expect = (r / (1.0 + r)) * (pg - 5000.0) / G;
            assert!((o.pwat[c] as f64 - expect).abs() < 1e-3 * expect, "{} {}", o.pwat[c], expect);
        }
    }
}
