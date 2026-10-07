//! Group D column computation: the CPU twin of `kernels/group_d.cu`,
//! statement for statement (binary64 from binary32 carriers, one rounding
//! per output, transcendentals from [`crate::math`] only), so the CPU and
//! GPU planes are the same bits.

use crate::math;
use crate::state::{MassInput, MassState};

// Flag bits, shared with kernels/group_d.cu (D_*).
pub const DF_REFL: i32 = 1;
pub const DF_QR: i32 = 2;
pub const DF_QS: i32 = 4;
pub const DF_QG: i32 = 8;
pub const DF_QC: i32 = 16;
pub const DF_QI: i32 = 32;
pub const DF_CF_MODEL: i32 = 64;
pub const DF_CF_DIAG: i32 = 128;
pub const DF_CF_BINARY: i32 = 256;
pub const DF_AER: i32 = 512;
pub const DF_HGT: i32 = 1024;
pub const DF_INTERP_DBZ: i32 = 2048;
pub const DF_OLR: i32 = 4096;

/// Planes of the column kernel (`woof_post_d_fields_v1`).
pub const N_COLUMN: usize = 9;
/// All group D planes: the column planes plus `simulated_ir`
/// (`woof_post_d_ir_v1`).
pub const N_OUT: usize = 10;
pub const O_COMP: usize = 0;
pub const O_R1KM: usize = 1;
pub const O_LOW: usize = 2;
pub const O_MID: usize = 3;
pub const O_HIGH: usize = 4;
pub const O_BASE: usize = 5;
pub const O_TOP: usize = 6;
pub const O_CEIL: usize = 7;
pub const O_VIS: usize = 8;
pub const O_IR: usize = 9;

const RD: f64 = 287.0;
const EPS: f64 = 287.0 / 461.6;
const PI: f64 = 3.141592653589793;
const LN10: f64 = 2.302585092994046;
// Reflectivity (spec 7.1, D17).
pub const DBZ_FLOOR: f64 = -20.0;
const Z_TINY: f64 = 1.0e-10;
pub const REFL_AGL: f64 = 1000.0;
const ICE_RATIO: f64 = 0.224;
const RHO_W: f64 = 1000.0;
const N0R: f64 = 8.0e6;
const RHOR: f64 = 1000.0;
const N0S: f64 = 2.0e7;
const RHOS: f64 = 100.0;
const N0G: f64 = 4.0e6;
const RHOG: f64 = 400.0;
// Cloud layers (D18) and geometry (D19, D20).
pub const P_LOW_TOP: f64 = 68000.0;
pub const P_MID_TOP: f64 = 44000.0;
pub const CF_MIN: f64 = 0.01;
pub const COND_MIN: f64 = 1.0e-6;
pub const CEIL_CF: f64 = 0.5;
// Diagnosed cloud fraction, Xu and Randall (1996).
const XR_P: f64 = 0.25;
const XR_ALPHA0: f64 = 100.0;
const XR_GAMMA: f64 = 0.49;
// Visibility (D21).
pub const VIS_CAP: f64 = 20000.0;
const EXT_C: (f64, f64) = (144.7, 0.88);
const EXT_R: (f64, f64) = (2.24, 0.75);
const EXT_I: (f64, f64) = (327.8, 1.0);
const EXT_S: (f64, f64) = (10.36, 0.7776);
const EXT_G: (f64, f64) = (2.24, 0.75);
const HAZE_A: f64 = 40.10;
const HAZE_B: f64 = 5.19e-10;
const HAZE_C: f64 = 5.44;
const HAZE_RH_MIN: f64 = 30.0;
const MAG: (f64, f64, f64) = (610.94, 17.625, 243.04);
const MAG_ICE: (f64, f64, f64) = (611.21, 22.587, 273.86);
// Simulated infrared brightness temperature (b2-ir lane, see `ir_column`).
/// Stefan-Boltzmann constant, W m-2 K-4 (CODATA 2018; exact in the 2019 SI).
pub const SIGMA_SB: f64 = 5.670374419e-8;
/// Ohring, Gruber and Ellingson (1984) flux-to-window relation
/// T_f = T_b (a + b T_b), constants as given by Yang and Slingo (2001).
pub const OGE_A: f64 = 1.228;
pub const OGE_B: f64 = -1.106e-3;
/// Infrared absorption per gram of condensate, m2 g-1: cloud water, and
/// cloud ice plus snow (WOOF's renderer values, docs/2.7.6 record).
/// PROVISIONAL: no peer-reviewed source traced; chosen and kept by the GOES
/// band-13 score of the b2-ir lane.  These two lines are the switch.
pub const IR_K_LIQUID: f64 = 0.145;
pub const IR_K_ICE: f64 = 0.272;
/// Optical depth of the emission level (Eddington-Barbier).
pub const IR_TAU_EMIT: f64 = 1.0;
const G_IR: f64 = 9.81;

/// One column's view of the inputs (the device-side `DCol`).
pub struct Col<'a> {
    pub st: &'a [f32],
    pub in3: &'a [f32],
    pub refl: Option<&'a [f32]>,
    pub cf: Option<&'a [f32]>,
    pub vol: usize,
    pub n: usize,
    pub i: usize,
    pub flags: i32,
}

#[inline]
fn nan() -> f64 {
    math::NAN64
}
#[inline]
fn isnan(x: f64) -> bool {
    x.is_nan()
}
#[inline]
fn dmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}
#[inline]
fn dmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}
#[inline]
fn clamp01(w: f64) -> f64 {
    if w < 0.0 {
        0.0
    } else if w > 1.0 {
        1.0
    } else {
        w
    }
}
#[inline]
fn log10(x: f64) -> f64 {
    math::ln(x) / LN10
}

impl Col<'_> {
    #[inline]
    fn st(&self, slot: MassState, k: usize) -> f64 {
        f64::from(self.st[slot as usize * self.vol + k * self.n + self.i])
    }
    #[inline]
    fn inp(&self, slot: MassInput, bit: i32, k: usize) -> f64 {
        if self.flags & bit == 0 {
            return 0.0;
        }
        f64::from(self.in3[slot as usize * self.vol + k * self.n + self.i])
    }
}

#[inline]
fn rho_air(p: f64, tk: f64, r: f64) -> f64 {
    let tv = tk * (1.0 + r / EPS) / (1.0 + r);
    p / (RD * tv)
}

#[inline]
fn z_species(rho_q: f64, n0: f64, rho_x: f64) -> f64 {
    if !(rho_q > 0.0) {
        return 0.0;
    }
    let lam = math::pow(PI * rho_x * n0 / rho_q, 0.25);
    720.0 * n0 * math::pow(lam, -7.0) * 1.0e18
}

/// dBZ at level k: REFL_10CM, else the RIP-note volume rounded once to
/// binary32 (the precision a stored volume has).
pub fn dbz(c: &Col<'_>, k: usize) -> f64 {
    if c.flags & DF_REFL != 0 {
        return f64::from(c.refl.expect("REFL flag without carrier")[k * c.n + c.i]);
    }
    let p = c.st(MassState::PFull, k);
    let tk = c.st(MassState::Tk, k);
    let r = c.st(MassState::R, k);
    let qr = c.inp(MassInput::QRain, DF_QR, k);
    let qs = c.inp(MassInput::QSnow, DF_QS, k);
    let qg = c.inp(MassInput::QGraupel, DF_QG, k);
    if isnan(p) || isnan(tk) || isnan(r) || isnan(qr) || isnan(qs) || isnan(qg) {
        return nan();
    }
    let rho = rho_air(p, tk, r);
    let fs = ICE_RATIO * ((RHOS / RHO_W) * (RHOS / RHO_W));
    let fg = ICE_RATIO * ((RHOG / RHO_W) * (RHOG / RHO_W));
    let mut z = z_species(rho * qr, N0R, RHOR);
    z = z + fs * z_species(rho * qs, N0S, RHOS);
    z = z + fg * z_species(rho * qg, N0G, RHOG);
    z = dmax(z, Z_TINY);
    f64::from((10.0 * log10(z)) as f32)
}

#[inline]
fn condensate(c: &Col<'_>, k: usize) -> f64 {
    let mut cond = c.inp(MassInput::QCloud, DF_QC, k) + c.inp(MassInput::QIce, DF_QI, k);
    if c.flags & DF_QS != 0 {
        cond = cond + c.inp(MassInput::QSnow, DF_QS, k);
    }
    cond
}

/// Saturation vapour pressure (Pa): water at or above 0 C, ice below [R6].
#[inline]
fn esat_phase(tk: f64) -> f64 {
    let tc = tk - 273.15;
    if tc >= 0.0 {
        return MAG.0 * math::exp(MAG.1 * tc / (tc + MAG.2));
    }
    MAG_ICE.0 * math::exp(MAG_ICE.1 * tc / (tc + MAG_ICE.2))
}

/// Cloud fraction (0..1) at level k.
pub fn cloud_fraction(c: &Col<'_>, k: usize) -> f64 {
    if c.flags & DF_CF_MODEL != 0 {
        return f64::from(c.cf.expect("CLDFRA flag without carrier")[k * c.n + c.i]);
    }
    let qc = c.inp(MassInput::QCloud, DF_QC, k);
    let qi = c.inp(MassInput::QIce, DF_QI, k);
    if c.flags & DF_CF_BINARY != 0 {
        let cond = condensate(c, k);
        if isnan(cond) {
            return nan();
        }
        return if cond > COND_MIN { 1.0 } else { 0.0 };
    }
    let p = c.st(MassState::PFull, k);
    let tk = c.st(MassState::Tk, k);
    let r = c.st(MassState::R, k);
    if isnan(p) || isnan(tk) || isnan(r) || isnan(qc) || isnan(qi) {
        return nan();
    }
    let l = dmax(qc, 0.0) + dmax(qi, 0.0);
    let es = esat_phase(tk);
    if !(p > es) {
        return if l > 0.0 { 1.0 } else { 0.0 };
    }
    let rs = EPS * es / (p - es);
    let rh = r / rs;
    if rh >= 1.0 {
        return 1.0;
    }
    if !(rh > 0.0) || !(l > 0.0) {
        return 0.0;
    }
    let arg = XR_ALPHA0 * l / math::pow((1.0 - rh) * rs, XR_GAMMA);
    let f = math::pow(rh, XR_P) * (1.0 - math::exp(-arg));
    clamp01(f)
}

#[inline]
fn cross(cf: f64, cond: f64, fthr: f64) -> f64 {
    dmin(cf / fthr, cond / COND_MIN)
}

/// The nine column-kernel values of one column (binary64; NaN = missing).
pub fn column(c: &Col<'_>, nz: usize, aer: Option<&[f32]>, hgt: Option<&[f32]>, contrast: f64) -> [f64; N_COLUMN] {
    let mut o = [nan(); N_COLUMN];
    let has_refl = c.flags & (DF_REFL | DF_QR) != 0;
    let has_cf = c.flags & (DF_CF_MODEL | DF_CF_DIAG) != 0;

    if has_refl {
        let mut m = -1.0e300;
        let mut bad = false;
        for k in 0..nz {
            let d = dbz(c, k);
            if isnan(d) {
                bad = true;
            } else if d > m {
                m = d;
            }
        }
        if !bad {
            o[O_COMP] = dmax(m, DBZ_FLOOR);
        }
        if c.flags & DF_HGT != 0 {
            let h = f64::from(hgt.expect("HGT flag without plane")[c.i]);
            let mut bad1 = isnan(h);
            let mut kf: Option<usize> = None;
            for k in 0..nz {
                let z = c.st(MassState::ZMass, k);
                if isnan(z) || isnan(dbz(c, k)) {
                    bad1 = true;
                }
                if kf.is_none() && (z - h) >= REFL_AGL {
                    kf = Some(k);
                }
            }
            if let (false, Some(kf)) = (bad1, kf) {
                let lo = if kf > 0 { kf - 1 } else { 0 };
                let z_hi = c.st(MassState::ZMass, kf) - h;
                let z_lo = c.st(MassState::ZMass, lo) - h;
                let w = if kf > 0 { (REFL_AGL - z_lo) / (z_hi - z_lo) } else { 1.0 };
                let d_hi = dbz(c, kf);
                let d_lo = dbz(c, lo);
                let d = if c.flags & DF_INTERP_DBZ != 0 {
                    d_lo + w * (d_hi - d_lo)
                } else {
                    let zh = math::pow(10.0, d_hi / 10.0);
                    let zl = math::pow(10.0, d_lo / 10.0);
                    let zz = zl + w * (zh - zl);
                    10.0 * log10(dmax(zz, Z_TINY))
                };
                o[O_R1KM] = dmax(d, DBZ_FLOOR);
            }
        }
    }

    if has_cf {
        let (mut l, mut m, mut h) = (0.0, 0.0, 0.0);
        let mut bad = false;
        let (mut kb, mut kt, mut kc): (Option<usize>, Option<usize>, Option<usize>) = (None, None, None);
        for k in 0..nz {
            let p = c.st(MassState::PFull, k);
            let cf = cloud_fraction(c, k);
            let cond = condensate(c, k);
            let z = c.st(MassState::ZMass, k);
            if isnan(p) || isnan(cf) || isnan(cond) || isnan(z) {
                bad = true;
                continue;
            }
            if p > P_LOW_TOP {
                l = dmax(l, cf);
            } else if p > P_MID_TOP {
                m = dmax(m, cf);
            } else {
                h = dmax(h, cf);
            }
            let cloudy = (cf > CF_MIN) && (cond > COND_MIN);
            let ceil_cloudy = (cf >= CEIL_CF) && (cond > COND_MIN);
            if cloudy {
                if kb.is_none() {
                    kb = Some(k);
                }
                kt = Some(k);
            }
            if ceil_cloudy && kc.is_none() {
                kc = Some(k);
            }
        }
        if !bad {
            o[O_LOW] = dmin(dmax(100.0 * l, 0.0), 100.0);
            o[O_MID] = dmin(dmax(100.0 * m, 0.0), 100.0);
            o[O_HIGH] = dmin(dmax(100.0 * h, 0.0), 100.0);
            for pass in 0..2 {
                let k = if pass == 0 { kb } else { kc };
                let fthr = if pass == 0 { CF_MIN } else { CEIL_CF };
                let dst = if pass == 0 { O_BASE } else { O_CEIL };
                let Some(k) = k else { continue };
                let z_k = c.st(MassState::ZMass, k);
                if k == 0 {
                    o[dst] = z_k;
                    continue;
                }
                let z_lo = c.st(MassState::ZMass, k - 1);
                let s_k = cross(cloud_fraction(c, k), condensate(c, k), fthr);
                let s_lo = cross(cloud_fraction(c, k - 1), condensate(c, k - 1), fthr);
                let den = s_k - s_lo;
                let mut w = if den > 0.0 { (1.0 - s_lo) / den } else { 1.0 };
                w = clamp01(w);
                o[dst] = z_lo + w * (z_k - z_lo);
            }
            if let Some(kt) = kt {
                let z_k = c.st(MassState::ZMass, kt);
                if kt == nz - 1 {
                    o[O_TOP] = z_k;
                } else {
                    let z_hi = c.st(MassState::ZMass, kt + 1);
                    let s_k = cross(cloud_fraction(c, kt), condensate(c, kt), CF_MIN);
                    let s_hi = cross(cloud_fraction(c, kt + 1), condensate(c, kt + 1), CF_MIN);
                    let den = s_k - s_hi;
                    let mut w = if den > 0.0 { (s_k - 1.0) / den } else { 0.0 };
                    w = clamp01(w);
                    o[O_TOP] = z_k + w * (z_hi - z_k);
                }
            }
        }
    }

    // Visibility at the lowest mass level.
    {
        let p = c.st(MassState::PFull, 0);
        let tk = c.st(MassState::Tk, 0);
        let r = c.st(MassState::R, 0);
        let qc = c.inp(MassInput::QCloud, DF_QC, 0);
        let qr = c.inp(MassInput::QRain, DF_QR, 0);
        let qi = c.inp(MassInput::QIce, DF_QI, 0);
        let qs = c.inp(MassInput::QSnow, DF_QS, 0);
        let qg = c.inp(MassInput::QGraupel, DF_QG, 0);
        let a = if c.flags & DF_AER != 0 { f64::from(aer.expect("AER flag without plane")[c.i]) } else { 0.0 };
        let bad = isnan(p)
            || isnan(tk)
            || isnan(r)
            || isnan(qc)
            || isnan(qr)
            || isnan(qi)
            || isnan(qs)
            || isnan(qg)
            || isnan(a);
        if !bad {
            let rho = rho_air(p, tk, r);
            let mut beta = EXT_C.0 * math::pow(1000.0 * rho * dmax(qc, 0.0), EXT_C.1);
            beta = beta + EXT_R.0 * math::pow(1000.0 * rho * dmax(qr, 0.0), EXT_R.1);
            beta = beta + EXT_I.0 * math::pow(1000.0 * rho * dmax(qi, 0.0), EXT_I.1);
            beta = beta + EXT_S.0 * math::pow(1000.0 * rho * dmax(qs, 0.0), EXT_S.1);
            beta = beta + EXT_G.0 * math::pow(1000.0 * rho * dmax(qg, 0.0), EXT_G.1);
            let q = r / (1.0 + r);
            let e = q * p / (EPS + (1.0 - EPS) * q);
            let tc = tk - 273.15;
            let es = MAG.0 * math::exp(MAG.1 * tc / (tc + MAG.2));
            let rh = dmin(dmax(100.0 * e / es, 0.0), 100.0);
            let rh_h = dmax(rh, HAZE_RH_MIN);
            let vis_haze_km = HAZE_A - HAZE_B * math::pow(rh_h, HAZE_C);
            beta = beta + contrast / vis_haze_km;
            beta = beta + a;
            o[O_VIS] = dmin(1000.0 * contrast / beta, VIS_CAP);
        }
    }
    o
}


/// Window brightness temperature from the model's top-of-atmosphere
/// outgoing longwave flux (W m-2): flux-equivalent temperature
/// T_f = (OLR / sigma)^(1/4), then the root of T_f = T_b (a + b T_b)
/// (Ohring, Gruber and Ellingson, J. Climate Appl. Meteor. 23 (1984) 416;
/// a = 1.228, b = -1.106e-3 K-1 from Yang and Slingo, Mon. Wea. Rev. 129
/// (2001) 784).  NaN for a missing, non-positive or out-of-range flux.
#[inline]
pub fn olr_brightness_temperature(olr: f64) -> f64 {
    if !(olr > 0.0) {
        return nan();
    }
    let tf = math::sqrt(math::sqrt(olr / SIGMA_SB));
    let disc = OGE_A * OGE_A + 4.0 * OGE_B * tf;
    if !(disc >= 0.0) {
        return nan();
    }
    (math::sqrt(disc) - OGE_A) / (2.0 * OGE_B)
}

/// Simulated infrared brightness temperature of one column (K, binary64).
///
/// Where the column is opaque in the infrared window, the emission level:
/// absorption optical depth is summed down from the model top, layer
/// k contributing (K_LIQUID QCLOUD + K_ICE (QICE + QSNOW)) x 1000 x
/// (p_int[k] - p_int[k+1]) / g, and the brightness temperature is the air
/// temperature where the sum reaches 1 (Eddington-Barbier: the emergent
/// intensity is the source function at unit optical depth), linear in
/// optical depth between the mass level above and the crossing level.
/// Where the column never reaches unit optical depth (clear or thin cloud),
/// the model's own OLR through [`olr_brightness_temperature`].  The opaque
/// branch needs QCLOUD and QICE (`DF_QC | DF_QI`); without them every
/// column takes the OLR branch.  NaN when a needed input is missing.
pub fn ir_column(c: &Col<'_>, iface: &[f32], olr: Option<&[f32]>, nz: usize) -> f64 {
    if c.flags & DF_OLR == 0 {
        return nan();
    }
    if c.flags & DF_QC != 0 && c.flags & DF_QI != 0 {
        let ivol = (nz + 1) * c.n;
        let pint = |k: usize| f64::from(iface[crate::state::InterfaceState::PInt as usize * ivol + k * c.n + c.i]);
        let mut cum = 0.0;
        let mut k = nz;
        while k > 0 {
            k -= 1;
            let qc = c.inp(MassInput::QCloud, DF_QC, k);
            let qi = c.inp(MassInput::QIce, DF_QI, k);
            let qs = c.inp(MassInput::QSnow, DF_QS, k);
            let dp = pint(k) - pint(k + 1);
            let tk = c.st(MassState::Tk, k);
            if isnan(qc) || isnan(qi) || isnan(qs) || isnan(dp) || isnan(tk) {
                return nan();
            }
            let cond = IR_K_LIQUID * dmax(qc, 0.0) + IR_K_ICE * (dmax(qi, 0.0) + dmax(qs, 0.0));
            let dtau = cond * 1000.0 * dmax(dp, 0.0) / G_IR;
            if cum + dtau >= IR_TAU_EMIT {
                let w = (IR_TAU_EMIT - cum) / dtau;
                let t_above = if k + 1 < nz { c.st(MassState::Tk, k + 1) } else { tk };
                if isnan(t_above) {
                    return nan();
                }
                return t_above + w * (tk - t_above);
            }
            cum = cum + dtau;
        }
    }
    let o = f64::from(olr.expect("OLR flag without plane")[c.i]);
    if isnan(o) {
        return nan();
    }
    olr_brightness_temperature(o)
}
