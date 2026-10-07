//
// WOOF post-processor, group E: the binary64 CPU reference.
//
// Clean-room implementation written only from the WOOF clean-room post
// specification (section 8, "Group E") and the public sources it cites
// (listed in kernels/group_e.cu and in the crate documentation).  This is the
// CPU path AND the float64 reference the CUDA kernel is tested against: every
// expression here has a twin in kernels/group_e.cu with the same operation
// order, so the two agree bit for bit.

use crate::math::{cbrt, exp, ln};

pub const G: f64 = 9.81;
pub const RD: f64 = 287.0;
pub const CP: f64 = 3.5 * RD;
pub const RV: f64 = 461.6;
pub const P0: f64 = 100_000.0;
pub const EPS: f64 = RD / RV;
pub const KAPPA: f64 = RD / CP;
pub const LV: f64 = 2.501e6;
pub const KARMAN: f64 = 0.4;
/// Critical bulk Richardson number (Troen and Mahrt 1986; Seidel et al. 2012).
pub const RI_CRIT: f64 = 0.25;
/// Weight of the friction-velocity term (Vogelezang and Holtslag 1996).
pub const B_UST: f64 = 100.0;
/// IFS gust constant (ECMWF IFS Part IV, from Panofsky et al. 1977).
pub const C_UGN: f64 = 7.71;
pub const DEN_FLOOR: f64 = 1.0e-10;
/// WMO 1957 tropopause: lapse rate at or below 2 K/km ...
pub const TROP_LAPSE: f64 = 0.002;
/// ... and the mean lapse rate to every level within 2 km above stays at or below it.
pub const TROP_DEPTH: f64 = 2000.0;
/// RULED (RULINGS.md item 8, 2026-10-05: WMO lapse-rate tropopause searched
/// from 500 hPa to 50 hPa).  Pressure window searched for the tropopause, so
/// a low-level inversion is never taken for it.  The group's original window
/// was 550 to 75 hPa; changing it is a change of these two values (and their
/// twins W_TROP_P_MAX/MIN in kernels/group_e.cu).
pub const TROP_P_MAX: f64 = 50_000.0;
pub const TROP_P_MIN: f64 = 5_000.0;

pub const T_FREEZE: f64 = 273.15;
pub const T_MINUS10: f64 = 263.15;
pub const T_MINUS20: f64 = 253.15;

/// Number of output planes, in this order.
pub const N_OUT: usize = 10;
pub const OUT_NAMES: [&str; N_OUT] = [
    "pbl_height",
    "gust",
    "u80",
    "v80",
    "freezing_height",
    "freezing_pressure",
    "highest_freezing_height",
    "highest_freezing_pressure",
    "minus10_height",
    "minus20_height",
];

#[inline]
fn out(x: f64) -> f32 {
    if x.is_nan() { f32::NAN } else { x as f32 }
}

/// Profile of one column as the field computation sees it (state values,
/// bottom-up, already f32-rounded).
#[derive(Clone, Debug, Default)]
pub struct Column {
    pub theta_v: Vec<f32>,
    pub tk: Vec<f32>,
    pub p: Vec<f32>,
    pub z: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    /// Turbulent kinetic energy per unit mass (m2 s-2) when the history has it.
    pub tke: Option<Vec<f32>>,
}

/// The 2D carriers of one cell.  `None` = carrier absent from the history.
#[derive(Clone, Copy, Debug, Default)]
pub struct Surface {
    pub hgt: f32,
    pub psfc: f32,
    pub u10: f32,
    pub v10: f32,
    pub ust: Option<f32>,
    pub t2: Option<f32>,
    pub th2: Option<f32>,
    pub q2: Option<f32>,
    pub hfx: Option<f32>,
    pub qfx: Option<f32>,
    pub lh: Option<f32>,
    pub sinalpha: Option<f32>,
    pub cosalpha: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GustMethod {
    /// Surface-layer similarity gust (ECMWF IFS Part IV; Panofsky et al. 1977).
    /// The method for histories without TKE.  RULINGS 7 (no standard exists)
    /// was settled by the ASOS gust score: the exporter's default `auto`
    /// runs `MixedLayerTke` wherever the history carries TKE and this one
    /// elsewhere (`--gust similarity` forces it).
    #[default]
    Similarity,
    /// Mixing-layer momentum transfer (Brasseur 2001); needs TKE.
    MixedLayerTke,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub earth_winds: bool,
    pub gust: GustMethod,
}

/// Bulk-Richardson boundary-layer height, metres above ground (spec 8).
#[must_use]
pub fn pbl_height(c: &Column, zsfc: f64, ust: Option<f64>) -> f64 {
    let nz = c.z.len();
    let thv_s = f64::from(c.theta_v[0]);
    let u_s = f64::from(c.u[0]);
    let v_s = f64::from(c.v[0]);
    let z_s = f64::from(c.z[0]);
    let mut ok = !(thv_s.is_nan() || u_s.is_nan() || v_s.is_nan() || z_s.is_nan() || zsfc.is_nan());
    let us2 = match ust {
        Some(u_star) => {
            if u_star.is_nan() {
                ok = false;
            }
            B_UST * u_star * u_star
        }
        None => {
            ok = false;
            0.0
        }
    };
    if !ok {
        return f64::NAN;
    }
    let mut ri_prev = 0.0;
    let mut z_prev = z_s;
    for k in 1..nz {
        let thv = f64::from(c.theta_v[k]);
        let zk = f64::from(c.z[k]);
        let du = f64::from(c.u[k]) - u_s;
        let dv = f64::from(c.v[k]) - v_s;
        if thv.is_nan() || zk.is_nan() || du.is_nan() || dv.is_nan() {
            break;
        }
        let mut den = du * du + dv * dv + us2;
        if den < DEN_FLOOR {
            den = DEN_FLOOR;
        }
        let ri = (G / thv_s) * (thv - thv_s) * (zk - z_s) / den;
        if ri >= RI_CRIT {
            let f = (RI_CRIT - ri_prev) / (ri - ri_prev);
            let h = z_prev + f * (zk - z_prev);
            return h - zsfc;
        }
        ri_prev = ri;
        z_prev = zk;
    }
    f64::NAN
}

/// Wind at 80 m above ground, grid-relative (spec 8, D25).
#[must_use]
pub fn wind80(c: &Column, zsfc: f64, u10: f64, v10: f64) -> (f64, f64) {
    let nz = c.z.len();
    let zt = zsfc + 80.0;
    let z0 = f64::from(c.z[0]);
    if zt.is_nan() || z0.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if z0 > zt {
        let zb = zsfc + 10.0;
        if !u10.is_nan() && !v10.is_nan() && z0 > zb {
            let f = (zt - zb) / (z0 - zb);
            return (u10 + f * (f64::from(c.u[0]) - u10), v10 + f * (f64::from(c.v[0]) - v10));
        }
        return (f64::NAN, f64::NAN);
    }
    for k in 0..nz.saturating_sub(1) {
        let za = f64::from(c.z[k]);
        let zb = f64::from(c.z[k + 1]);
        if za.is_nan() || zb.is_nan() {
            break;
        }
        if zb >= zt {
            let f = (zt - za) / (zb - za);
            let ua = f64::from(c.u[k]);
            let va = f64::from(c.v[k]);
            return (ua + f * (f64::from(c.u[k + 1]) - ua), va + f * (f64::from(c.v[k + 1]) - va));
        }
    }
    (f64::NAN, f64::NAN)
}

/// Index of the WMO lapse-rate tropopause among the mass levels, or the top
/// level when none is found in the search window.
#[must_use]
pub fn tropopause_level(c: &Column) -> usize {
    let nz = c.z.len();
    for k in 0..nz.saturating_sub(1) {
        let pk = f64::from(c.p[k]);
        if !(pk <= TROP_P_MAX && pk >= TROP_P_MIN) {
            continue;
        }
        let zk = f64::from(c.z[k]);
        let tkk = f64::from(c.tk[k]);
        let gam = (tkk - f64::from(c.tk[k + 1])) / (f64::from(c.z[k + 1]) - zk);
        if !(gam <= TROP_LAPSE) {
            continue;
        }
        let mut ok = true;
        for j in k + 1..nz {
            let dz = f64::from(c.z[j]) - zk;
            if dz > TROP_DEPTH {
                break;
            }
            let avg = (tkk - f64::from(c.tk[j])) / dz;
            if !(avg <= TROP_LAPSE) {
                ok = false;
                break;
            }
        }
        if ok {
            return k;
        }
    }
    nz - 1
}

/// Where the profile crosses `tc` going from warm (a, below) to cold (b,
/// above): height linear in z, pressure linear in ln p.
#[must_use]
pub fn cross(a: (f64, f64, f64), b: (f64, f64, f64), tc: f64) -> (f64, f64) {
    let (za, ta, pa) = a;
    let (zb, tb, pb) = b;
    let f = (ta - tc) / (ta - tb);
    let z = za + f * (zb - za);
    let la = ln(pa);
    let lb = ln(pb);
    (z, exp(la + f * (lb - la)))
}

/// Shelter temperature, shelter potential temperature and mixing ratio.
fn shelter(s: &Surface) -> (f64, f64, f64) {
    let ps = f64::from(s.psfc);
    let mut r_sh = s.q2.map_or(f64::NAN, f64::from);
    if !r_sh.is_nan() && r_sh < 0.0 {
        r_sh = 0.0;
    }
    let ex = exp(KAPPA * ln(ps / P0));
    let mut t_sh = s.t2.map_or(f64::NAN, f64::from);
    let mut th_sh = s.th2.map_or(f64::NAN, f64::from);
    if t_sh.is_nan() && !th_sh.is_nan() {
        t_sh = th_sh * ex;
    }
    if th_sh.is_nan() && !t_sh.is_nan() {
        th_sh = t_sh / ex;
    }
    (t_sh, th_sh, r_sh)
}

/// All ten group E outputs for one column, in `OUT_NAMES` order.
#[must_use]
pub fn fields(c: &Column, s: &Surface, opt: Options) -> [f32; N_OUT] {
    let nz = c.z.len();
    let zsfc = f64::from(s.hgt);
    let ps = f64::from(s.psfc);

    let pbl = pbl_height(c, zsfc, s.ust.map(f64::from));

    let (mut u80, mut v80) = wind80(c, zsfc, f64::from(s.u10), f64::from(s.v10));
    if opt.earth_winds {
        if let (Some(sa), Some(ca)) = (s.sinalpha, s.cosalpha) {
            let sa = f64::from(sa);
            let ca = f64::from(ca);
            let ue = u80 * ca - v80 * sa;
            let ve = v80 * ca + u80 * sa;
            u80 = ue;
            v80 = ve;
        }
    }

    // shelter point
    let (t_sh, th_sh, r_sh) = shelter(s);
    let has_sh = !t_sh.is_nan() && !ps.is_nan() && !zsfc.is_nan();
    let z_sh = zsfc + 2.0;
    let mut p_sh = f64::NAN;
    if has_sh {
        let tv = if r_sh.is_nan() { t_sh } else { t_sh * (1.0 + r_sh / EPS) / (1.0 + r_sh) };
        p_sh = ps * exp(-(G * 2.0) / (RD * tv));
    }
    let off = usize::from(has_sh);
    let npt = nz + off;
    let pt = |n: usize| -> (f64, f64, f64) {
        if has_sh && n == 0 {
            (z_sh, t_sh, p_sh)
        } else {
            let k = n - off;
            (f64::from(c.z[k]), f64::from(c.tk[k]), f64::from(c.p[k]))
        }
    };

    let ktrop = tropopause_level(c);
    let ntop = ktrop + off;

    let mut fz = f64::NAN;
    let mut fp = f64::NAN;
    let mut hl = [(f64::NAN, f64::NAN); 3];
    let mut col_ok = !zsfc.is_nan() && !ps.is_nan();
    if col_ok {
        for n in 0..npt {
            let (zz, tt, pp) = pt(n);
            if zz.is_nan() || tt.is_nan() || pp.is_nan() {
                col_ok = false;
                break;
            }
        }
    }
    if col_ok {
        let tground = pt(0).1;
        if tground <= T_FREEZE {
            fz = zsfc;
            fp = ps;
        } else {
            for n in 0..npt - 1 {
                let a = pt(n);
                let b = pt(n + 1);
                if a.1 > T_FREEZE && b.1 <= T_FREEZE {
                    (fz, fp) = cross(a, b, T_FREEZE);
                    break;
                }
            }
        }
        for (which, tc) in [T_FREEZE, T_MINUS10, T_MINUS20].into_iter().enumerate() {
            let mut found = None;
            for n in (0..ntop).rev() {
                let a = pt(n);
                let b = pt(n + 1);
                if a.1 > tc && b.1 <= tc {
                    found = Some(cross(a, b, tc));
                    break;
                }
            }
            hl[which] = match found {
                Some(v) => v,
                None if tground <= tc => (zsfc, ps),
                None => (f64::NAN, f64::NAN),
            };
        }
    }

    let gust = gust(c, s, opt.gust, pbl, t_sh, th_sh, r_sh);

    [
        out(pbl),
        out(gust),
        out(u80),
        out(v80),
        out(fz),
        out(fp),
        out(hl[0].0),
        out(hl[0].1),
        out(hl[1].0),
        out(hl[2].0),
    ]
}

#[allow(clippy::too_many_arguments)]
fn gust(c: &Column, s: &Surface, method: GustMethod, pbl: f64, t_sh: f64, th_sh: f64, r_sh: f64) -> f64 {
    let nz = c.z.len();
    let zsfc = f64::from(s.hgt);
    let ps = f64::from(s.psfc);
    let a = f64::from(s.u10);
    let b = f64::from(s.v10);
    let spd = (a * a + b * b).sqrt();
    match method {
        GustMethod::MixedLayerTke => {
            let Some(tke) = c.tke.as_ref() else {
                return f64::NAN;
            };
            if spd.is_nan() || pbl.is_nan() || th_sh.is_nan() {
                return f64::NAN;
            }
            let thv_g = if r_sh.is_nan() { th_sh } else { th_sh * (1.0 + r_sh / EPS) / (1.0 + r_sh) };
            let mut gust = spd;
            let e0 = f64::from(tke[0]);
            for kp in 0..nz {
                let zp = f64::from(c.z[kp]) - zsfc;
                if zp > pbl {
                    break;
                }
                let thp = f64::from(c.theta_v[kp]);
                let mut e_int = 0.0;
                let mut b_int = 0.0;
                let mut z_lo = 0.0;
                let mut e_lo = e0;
                let mut f_lo = G * (thp - thv_g) / thv_g;
                for k in 0..=kp {
                    let z_hi = f64::from(c.z[k]) - zsfc;
                    let e_hi = f64::from(tke[k]);
                    let thk = f64::from(c.theta_v[k]);
                    let f_hi = G * (thp - thk) / thk;
                    let dz = z_hi - z_lo;
                    e_int = e_int + 0.5 * (e_lo + e_hi) * dz;
                    b_int = b_int + 0.5 * (f_lo + f_hi) * dz;
                    z_lo = z_hi;
                    e_lo = e_hi;
                    f_lo = f_hi;
                }
                if zp > 0.0 && e_int / zp >= b_int {
                    let uk = f64::from(c.u[kp]);
                    let vk = f64::from(c.v[kp]);
                    let sk = (uk * uk + vk * vk).sqrt();
                    if sk > gust {
                        gust = sk;
                    }
                }
            }
            gust
        }
        GustMethod::Similarity => {
            let (Some(ust), Some(hfx)) = (s.ust, s.hfx) else {
                return f64::NAN;
            };
            let (wq_src, is_qfx) = match (s.qfx, s.lh) {
                (Some(q), _) => (f64::from(q), true),
                (None, Some(l)) => (f64::from(l), false),
                (None, None) => return f64::NAN,
            };
            if spd.is_nan() || t_sh.is_nan() || th_sh.is_nan() || r_sh.is_nan() || ps.is_nan() {
                return f64::NAN;
            }
            let mut u_star = f64::from(ust);
            let h = f64::from(hfx);
            if u_star.is_nan() || h.is_nan() || wq_src.is_nan() {
                return f64::NAN;
            }
            if u_star < 0.0 {
                u_star = 0.0;
            }
            let vf = (1.0 + r_sh / EPS) / (1.0 + r_sh);
            let tv = t_sh * vf;
            let thv = th_sh * vf;
            let rho = ps / (RD * tv);
            let wth = h / (rho * CP);
            let wq = if is_qfx { wq_src / rho } else { wq_src / (rho * LV) };
            let qs = r_sh / (1.0 + r_sh);
            let c61 = 1.0 / EPS - 1.0;
            let wthv = wth * (1.0 + c61 * qs) + c61 * th_sh * wq;
            let mut sig = u_star;
            if wthv > 0.0 && !pbl.is_nan() && pbl > 0.0 {
                sig = cbrt(u_star * u_star * u_star + pbl * KARMAN * G * wthv / (24.0 * thv));
            }
            spd + C_UGN * sig
        }
    }
}
