//
// Wind profile diagnostics and severe indices (CPU reference, f64).
//
// Sources (SPEC section 6.3):
// - Right-moving supercell motion, internal-dynamics method: Wea.
//   Forecasting 15 (2000) 61-79.  V_mean = 0-6 km non-pressure-weighted
//   mean wind; S = (5.5-6 km mean) - (0-0.5 km mean); C = V_mean +
//   D (S x k)/|S| with D = 7.5 m/s; |S| = 0 gives C = V_mean.  Right mover
//   everywhere (D14).
// - Storm-relative helicity: 16th Conf. on Severe Local Storms (1990)
//   588-592, SRH = -int k . ((V - C) x dV/dz) dz, as the standard discrete
//   sum over the profile points from 10 m to the layer top (interpolated).
//   The sum is exact for a hodograph that is piecewise linear in height.
// - Bulk shear: vector difference between the layer top and 10 m (endpoints,
//   SPC Mesoanalysis help, D14), and its magnitude.
// - STP, fixed layer: Wea. Forecasting 18 (2003) 1243-1261 as revised in
//   Wea. Forecasting 27 (2012) 1136-1154 and documented by SPC:
//   (sbCAPE/1500) x LCL term x (SRH01/150) x shear term x CIN term with
//   LCL term 1 below 1000 m, 0 above 2000 m, (2000 - LCL)/1000 between;
//   shear term 0 below 12.5 m/s, 1.5 above 30 m/s, BWD/20 between;
//   CIN term 1 above -50 J/kg, 0 below -200 J/kg, (200 + CIN)/150 between.
// - EHI = sbCAPE x SRH01 / 160000 (SHARP workstation user guide, 1991).
//
// Profile: heights above ground are z_mass - zsfc.  The 10 m wind is the
// lowest point (at 10 m, held constant down to the ground for layer means);
// mass levels at or below 10 m are skipped; between points the wind is
// linear in height.

/// One column's wind profile in f64.
use crate::math as m;

#[derive(Clone, Debug, Default)]
pub struct WindColumn {
    pub z_agl: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub u10: f64,
    pub v10: f64,
}

const Z10: f64 = 10.0;
const BUNKERS_D: f64 = 7.5;

/// RULED (RULINGS.md item 3, 2026-10-05: 0-6 km shear is the bulk vector
/// difference from the 10 m wind to the 6 km AGL wind, SPC standard).  false
/// (the ruled default): the vector difference between the 6 km and 10 m
/// winds (SPEC D14, SPC endpoints).  true: the 5.5-6 km layer mean minus the
/// 0-0.5 km layer mean (the [R24] shear vector).  Twin:
/// `SHEAR06_LAYER_MEANS` in kernels/severe_c.cu; change both together.
pub const SHEAR06_LAYER_MEANS: bool = false;

/// RULED (RULINGS.md item 4, 2026-10-05: STP floored at zero, SPC practice;
/// negative STP has no meaning).  true (the ruled default): floored at 0.
/// false: the fixed-layer STP product as published, negative where SRH is
/// negative.  Twin: `STP_FLOOR_ZERO` in
/// kernels/severe_c.cu; change both together.
pub const STP_FLOOR_ZERO: bool = true;

// RULED (RULINGS.md item 2, 2026-10-05: Bunkers (2000) internal-dynamics
// method, 0-6 km non-pressure-weighted mean wind, 7.5 m/s deviation, SPC
// standard): the 0-6 km mean wind and the shear vector use height-weighted
// layer means ([R24], SPEC 6.3).  The level-arithmetic alternative named in
// the audit has no public description, so it is not implemented.

impl WindColumn {
    /// Profile points: (0, w10), (10, w10), then mass levels above 10 m.
    fn points(&self) -> Vec<(f64, f64, f64)> {
        let mut out = vec![(0.0, self.u10, self.v10), (Z10, self.u10, self.v10)];
        for k in 0..self.z_agl.len() {
            if self.z_agl[k] > Z10 {
                out.push((self.z_agl[k], self.u[k], self.v[k]));
            }
        }
        out
    }

    /// Wind at height h AGL (h >= 10 m), linear in z; None above the profile.
    pub fn at(&self, h: f64) -> Option<(f64, f64)> {
        let pts = self.points();
        if h <= Z10 {
            return Some((self.u10, self.v10));
        }
        for w in pts.windows(2) {
            let (za, ua, va) = w[0];
            let (zb, ub, vb) = w[1];
            if h >= za && h <= zb {
                let t = (h - za) / (zb - za);
                return Some((ua + (ub - ua) * t, va + (vb - va) * t));
            }
        }
        None
    }

    /// Height-weighted mean wind over [h1, h2] AGL.
    pub fn layer_mean(&self, h1: f64, h2: f64) -> Option<(f64, f64)> {
        let pts = self.points();
        if pts.last().map(|p| p.0).unwrap_or(0.0) < h2 {
            return None;
        }
        let (mut su, mut sv) = (0.0, 0.0);
        for w in pts.windows(2) {
            let (za, ua, va) = w[0];
            let (zb, ub, vb) = w[1];
            let lo = za.max(h1);
            let hi = zb.min(h2);
            if hi > lo {
                let tl = (lo - za) / (zb - za);
                let th = (hi - za) / (zb - za);
                let (ul, vl) = (ua + (ub - ua) * tl, va + (vb - va) * tl);
                let (uh, vh) = (ua + (ub - ua) * th, va + (vb - va) * th);
                su += 0.5 * (ul + uh) * (hi - lo);
                sv += 0.5 * (vl + vh) * (hi - lo);
            }
        }
        Some((su / (h2 - h1), sv / (h2 - h1)))
    }

    /// Right-mover storm motion.
    pub fn storm_motion(&self) -> Option<(f64, f64)> {
        let (um, vm) = self.layer_mean(0.0, 6000.0)?;
        let (ul, vl) = self.layer_mean(0.0, 500.0)?;
        let (uh, vh) = self.layer_mean(5500.0, 6000.0)?;
        let (su, sv) = (uh - ul, vh - vl);
        let mag = m::sqrt(su * su + sv * sv);
        if mag > 0.0 {
            Some((um + BUNKERS_D * sv / mag, vm - BUNKERS_D * su / mag))
        } else {
            Some((um, vm))
        }
    }

    /// Storm-relative helicity from 10 m to h AGL for storm motion (cu, cv).
    pub fn srh(&self, h: f64, cu: f64, cv: f64) -> Option<f64> {
        let pts = self.points();
        let (mut pu, mut pv) = (self.u10, self.v10);
        let mut pz = Z10;
        let mut sum = 0.0;
        for &(z, u, v) in pts.iter().skip(2) {
            let (nu, nv, done) = if z < h {
                (u, v, false)
            } else {
                let t = (h - pz) / (z - pz);
                (pu + (u - pu) * t, pv + (v - pv) * t, true)
            };
            sum += (nu - cu) * (pv - cv) - (pu - cu) * (nv - cv);
            if done {
                return Some(sum);
            }
            pu = nu;
            pv = nv;
            pz = z;
        }
        None
    }
}

/// Wind results for one column: [storm_u, storm_v, srh01, srh03, shear_u01,
/// shear_v01, shear_u06, shear_v06, bulk01, bulk06].
pub fn winds(col: &WindColumn) -> [f64; 10] {
    let mut out = [f64::NAN; 10];
    if let Some((cu, cv)) = col.storm_motion() {
        out[0] = cu;
        out[1] = cv;
        out[2] = col.srh(1000.0, cu, cv).unwrap_or(f64::NAN);
        out[3] = col.srh(3000.0, cu, cv).unwrap_or(f64::NAN);
    }
    if let Some((u1, v1)) = col.at(1000.0) {
        out[4] = u1 - col.u10;
        out[5] = v1 - col.v10;
        out[8] = m::sqrt(out[4] * out[4] + out[5] * out[5]);
    }
    let s06 = if SHEAR06_LAYER_MEANS {
        match (col.layer_mean(5500.0, 6000.0), col.layer_mean(0.0, 500.0)) {
            (Some((uh, vh)), Some((ul, vl))) => Some((uh - ul, vh - vl)),
            _ => None,
        }
    } else {
        col.at(6000.0).map(|(u6, v6)| (u6 - col.u10, v6 - col.v10))
    };
    if let Some((du, dv)) = s06 {
        out[6] = du;
        out[7] = dv;
        out[9] = m::sqrt(out[6] * out[6] + out[7] * out[7]);
    }
    out
}

/// Fixed-layer significant tornado parameter.
pub fn stp(sbcape: f64, sbcin: f64, sb_lcl: f64, srh01: f64, bwd06: f64) -> f64 {
    if !(sbcape.is_finite() && sbcin.is_finite() && sb_lcl.is_finite() && srh01.is_finite() && bwd06.is_finite()) {
        return f64::NAN;
    }
    let lcl_term = if sb_lcl < 1000.0 {
        1.0
    } else if sb_lcl > 2000.0 {
        0.0
    } else {
        (2000.0 - sb_lcl) / 1000.0
    };
    let shear_term = if bwd06 < 12.5 {
        0.0
    } else if bwd06 > 30.0 {
        1.5
    } else {
        bwd06 / 20.0
    };
    let cin_term = if sbcin > -50.0 {
        1.0
    } else if sbcin < -200.0 {
        0.0
    } else {
        (200.0 + sbcin) / 150.0
    };
    let stp = (sbcape / 1500.0) * lcl_term * (srh01 / 150.0) * shear_term * cin_term;
    if STP_FLOOR_ZERO && stp < 0.0 { 0.0 } else { stp }
}

/// Energy-helicity index, 0-1 km.
pub fn ehi(sbcape: f64, srh01: f64) -> f64 {
    sbcape * srh01 / 160_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight(shear: f64) -> WindColumn {
        // u = shear * z, v = 0; 10 m wind consistent with the profile.
        let z: Vec<f64> = (1..=40).map(|k| 250.0 * k as f64).collect();
        WindColumn {
            u: z.iter().map(|z| shear * z).collect(),
            v: vec![0.0; z.len()],
            z_agl: z,
            u10: shear * 10.0,
            v10: 0.0,
        }
    }

    #[test]
    fn straight_hodograph_has_no_helicity_about_its_own_line() {
        let c = straight(0.005);
        let (cu, cv) = c.storm_motion().unwrap();
        // Westerly shear: the right mover sits 7.5 m/s south of the mean.
        assert!(cv < -7.4 && cv > -7.6, "{cv}");
        assert!(cu > 0.0);
        // SRH relative to a point off the line is nonzero; relative to a
        // point on it, zero.
        assert!(c.srh(3000.0, cu, 0.0).unwrap().abs() < 1e-9);
        assert!(c.srh(3000.0, cu, cv).unwrap() > 0.0);
    }

    #[test]
    fn stp_bounds() {
        assert_eq!(stp(1500.0, 0.0, 500.0, 150.0, 20.0), 1.0);
        assert_eq!(stp(1500.0, -250.0, 500.0, 150.0, 20.0), 0.0);
        assert_eq!(stp(1500.0, 0.0, 2500.0, 150.0, 20.0), 0.0);
        assert_eq!(stp(1500.0, 0.0, 500.0, 150.0, 10.0), 0.0);
        assert_eq!(stp(1500.0, 0.0, 500.0, 150.0, 40.0), 1.5);
    }
}
