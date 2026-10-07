//
// Parcel engine and the CAPE/CIN/LCL fields (CPU reference, f64).
//
// Sources (SPEC sections 6.1 and 6.2):
// - Dry ascent conserving theta and mixing ratio to the LCL; LCL temperature
//   Bolton, Mon. Wea. Rev. 108 (1980) eq. 21; p_lcl = p0 (T_lcl/T0)^(cp/Rd).
// - Moist ascent on the pseudoadiabat of constant theta-e (Bolton eq. 43),
//   temperature by Newton iteration (Mon. Wea. Rev. 136 (2008) 2764-2785).
// - Buoyancy from virtual temperature of parcel and environment, Wea.
//   Forecasting 9 (1994) 625-629; the parcel carries its saturation mixing
//   ratio above the LCL and no condensate (pseudoadiabatic).
// - CAPE and CIN as in the AMS Glossary of Meteorology:
//   CAPE = Rd sum (Tv,p - Tv,e) d ln p over the positively buoyant parts from
//   the LFC to the EL, CIN the same sum over the negatively buoyant parts from
//   the parcel origin to the LFC (negative).  No LFC: CAPE 0, CIN 0 (D12).
// - Parcels: surface-based from 2 m T and q at shelter pressure (SPC, D10);
//   mixed-layer mean theta and mixing ratio of the lowest 90 hPa lifted from
//   the ground (Wea. Forecasting 17 (2002) 885-890, D11); most-unstable model
//   level of maximum theta-e in the lowest 300 hPa (SPC); best of six 30 hPa
//   layer-mean parcels in the lowest 180 hPa (D13).
//
// Discretisation (WOOF's own, stated here so it can be reviewed): buoyancy is
// taken as piecewise linear in ln p between profile points (the parcel origin,
// the LCL when it falls between levels, then every model level above the
// origin), and each layer's positive and negative parts are integrated
// exactly, splitting a layer at its zero crossing.  The LFC is the first point
// at or above the LCL where the parcel is positively buoyant, extended down
// through the positively buoyant run that contains it; CAPE then sums every
// positive part from there to the model top (which is everything up to the
// highest EL), CIN every negative part below it.

use crate::math as m;

use super::thermo::*;

/// One column in f64, mass levels bottom to top, interfaces ground to top.
#[derive(Clone, Debug, Default)]
pub struct Column {
    pub p: Vec<f64>,
    pub tk: Vec<f64>,
    pub r: Vec<f64>,
    pub p_int: Vec<f64>,
    pub z_int: Vec<f64>,
    pub zsfc: f64,
    pub psfc: f64,
    pub t2: f64,
    pub q2: f64,
    pub p2: f64,
}

impl Column {
    #[inline]
    fn nz(&self) -> usize {
        self.p.len()
    }

    #[inline]
    fn tv_env(&self, k: usize) -> f64 {
        virtual_temperature(self.tk[k], self.r[k].max(0.0))
    }

    /// Environmental virtual temperature at pressure `px`, linear in ln p
    /// between mass levels; between the shelter level and the lowest mass
    /// level it uses the shelter values; below the shelter level it holds them.
    fn tv_env_at(&self, px: f64) -> f64 {
        let tv2 = virtual_temperature(self.t2, self.q2.max(0.0));
        if px >= self.p2 {
            return tv2;
        }
        let nz = self.nz();
        let (mut pa, mut ta) = (self.p2, tv2);
        for k in 0..nz {
            let pb = self.p[k];
            let tb = self.tv_env(k);
            if px >= pb {
                let w = m::ln(pa / px) / m::ln(pa / pb);
                return ta + (tb - ta) * w;
            }
            pa = pb;
            ta = tb;
        }
        f64::NAN
    }

    /// Pressure-weighted mean theta and mixing ratio over the pressure layer
    /// [p_top, p_bot], each mass level standing for its own model layer
    /// between interfaces.
    fn layer_mean(&self, p_top: f64, p_bot: f64) -> (f64, f64) {
        let (mut w, mut sth, mut sr) = (0.0, 0.0, 0.0);
        for k in 0..self.nz() {
            let lo = self.p_int[k].min(p_bot);
            let hi = self.p_int[k + 1].max(p_top);
            let overlap = lo - hi;
            if overlap > 0.0 {
                w += overlap;
                sth += overlap * theta(self.tk[k], self.p[k]);
                sr += overlap * self.r[k].max(0.0);
            }
        }
        if w > 0.0 { (sth / w, sr / w) } else { (f64::NAN, f64::NAN) }
    }
}

/// Integrated positive and negative parts of a layer whose buoyancy is
/// linear in x = ln p from (xa, ba) at the bottom to (xb, bb) at the top.
#[inline]
fn layer_parts(xa: f64, ba: f64, xb: f64, bb: f64) -> (f64, f64) {
    let dx = xa - xb;
    if ba > 0.0 && bb > 0.0 {
        (0.5 * (ba + bb) * dx, 0.0)
    } else if ba <= 0.0 && bb <= 0.0 {
        (0.0, 0.5 * (ba + bb) * dx)
    } else if ba > 0.0 {
        let t = ba / (ba - bb);
        (0.5 * ba * t * dx, 0.5 * bb * (1.0 - t) * dx)
    } else {
        let t = ba / (ba - bb);
        (0.5 * bb * (1.0 - t) * dx, 0.5 * ba * t * dx)
    }
}

/// CAPE/CIN bookkeeping across layers (see the module header).
#[derive(Default)]
struct Accumulator {
    lfc: bool,
    neg_below: f64,
    run_pos: f64,
    cape: f64,
    cin: f64,
}

impl Accumulator {
    #[inline]
    fn add(&mut self, xa: f64, ba: f64, xb: f64, bb: f64, above_lcl: bool) {
        let (pos, neg) = layer_parts(xa, ba, xb, bb);
        if self.lfc {
            self.cape += pos;
            return;
        }
        let pos_a = ba > 0.0;
        let pos_b = bb > 0.0;
        if pos_a && pos_b {
            if above_lcl {
                self.lfc = true;
                self.cin = self.neg_below;
                self.cape = self.run_pos + pos;
            } else {
                self.run_pos += pos;
            }
        } else if !pos_a && !pos_b {
            self.neg_below += neg;
            self.run_pos = 0.0;
        } else if pos_a {
            if above_lcl {
                self.lfc = true;
                self.cin = self.neg_below;
                self.cape = self.run_pos + pos;
            } else {
                self.run_pos = 0.0;
                self.neg_below += neg;
            }
        } else if above_lcl {
            self.lfc = true;
            self.cin = self.neg_below + neg;
            self.cape = pos;
        } else {
            self.neg_below += neg;
            self.run_pos = pos;
        }
    }

    fn finish(&self) -> (f64, f64) {
        if self.lfc { (RD * self.cape, RD * self.cin) } else { (0.0, 0.0) }
    }
}

/// LCL temperature and pressure of a parcel; a supersaturated start is
/// saturated where it stands.
#[inline]
pub fn lcl(p_o: f64, t_o: f64, r_o: f64) -> (f64, f64) {
    let t_l = lcl_temperature(t_o, r_o, p_o);
    if t_l >= t_o { (t_o, p_o) } else { (t_l, p_o * m::pow(t_l / t_o, 1.0 / KAPPA)) }
}

/// Lift a parcel from (p_o, t_o, r_o) through the column; `tv_env_o` is the
/// environment's virtual temperature at the origin.  Returns (CAPE, CIN).
pub fn lift(col: &Column, p_o: f64, t_o: f64, r_o: f64, tv_env_o: f64) -> (f64, f64) {
    // A supersaturated start is saturated where it stands (see
    // thermo::theta_e_parcel): its vapour is capped at saturation.
    let r_o = r_o.max(R_FLOOR).min(rsat(t_o, p_o));
    let (t_l, p_lcl) = lcl(p_o, t_o, r_o);
    let th_o = theta(t_o, p_o);
    let ln_te = m::ln(theta_e(t_o, p_o, r_o, t_l));
    let mut acc = Accumulator::default();
    let mut p_prev = p_o;
    let mut x_prev = m::ln(p_o);
    let mut b_prev = virtual_temperature(t_o, r_o) - tv_env_o;
    let mut tve_prev = tv_env_o;
    let mut t_moist = t_l;
    for k in 0..col.nz() {
        let pk = col.p[k];
        if pk >= p_prev {
            continue;
        }
        let xk = m::ln(pk);
        let tve_k = col.tv_env(k);
        if p_lcl < p_prev && p_lcl > pk {
            let xl = m::ln(p_lcl);
            let tve_l = tve_prev + (tve_k - tve_prev) * (x_prev - xl) / (x_prev - xk);
            let b_l = virtual_temperature(t_l, r_o) - tve_l;
            acc.add(x_prev, b_prev, xl, b_l, p_prev <= p_lcl);
            p_prev = p_lcl;
            x_prev = xl;
            b_prev = b_l;
        }
        let tvp = if pk >= p_lcl {
            virtual_temperature(temperature_from_theta(th_o, pk), r_o)
        } else {
            t_moist = pseudoadiabat_temperature(ln_te, pk, t_moist);
            virtual_temperature(t_moist, rsat(t_moist, pk))
        };
        let b_k = tvp - tve_k;
        acc.add(x_prev, b_prev, xk, b_k, p_prev <= p_lcl);
        p_prev = pk;
        x_prev = xk;
        b_prev = b_k;
        tve_prev = tve_k;
    }
    acc.finish()
}

/// Height above ground of a parcel's LCL, interpolating interface height
/// linearly in ln p.  0 when the LCL is at or below the start; missing when
/// it lies above the model top, and for a parcel with no vapour (a dry
/// parcel never condenses, so it has no LCL).
pub fn lcl_height_agl(col: &Column, p_o: f64, t_o: f64, r_o: f64) -> f64 {
    if !(p_o.is_finite() && t_o.is_finite() && r_o > 0.0) {
        return f64::NAN;
    }
    let (_, p_lcl) = lcl(p_o, t_o, r_o);
    if p_lcl >= p_o || p_lcl >= col.p_int[0] {
        return 0.0;
    }
    let nz = col.nz();
    if p_lcl < col.p_int[nz] {
        return f64::NAN;
    }
    for k in 0..nz {
        let (pa, pb) = (col.p_int[k], col.p_int[k + 1]);
        if p_lcl <= pa && p_lcl > pb {
            let w = m::ln(pa / p_lcl) / m::ln(pa / pb);
            let z = col.z_int[k] + (col.z_int[k + 1] - col.z_int[k]) * w;
            return (z - col.zsfc).max(0.0);
        }
    }
    // p_lcl equals the top interface pressure exactly.
    (col.z_int[nz] - col.zsfc).max(0.0)
}

/// Parcel results for one column: [sbcape, sbcin, mlcape, mlcin, mucape,
/// mucin, cape_best180, cin_best180, lcl_height, sb_lcl_height].
pub fn parcels(col: &Column) -> [f64; 10] {
    let mut out = [f64::NAN; 10];
    // Surface-based: shelter T and q at shelter pressure; the parcel is the
    // environment at its origin.
    let r2 = col.q2.max(R_FLOOR);
    let tv2 = virtual_temperature(col.t2, r2);
    let (c, n) = lift(col, col.p2, col.t2, r2, tv2);
    out[0] = c;
    out[1] = n;
    out[9] = lcl_height_agl(col, col.p2, col.t2, col.q2);

    let pg = col.p_int[0];
    // Mixed layer: lowest 90 hPa, lifted from the ground.
    let (th_ml, r_ml) = col.layer_mean(pg - 9000.0, pg);
    let t_ml = temperature_from_theta(th_ml, col.psfc);
    let (c, n) = lift(col, col.psfc, t_ml, r_ml, col.tv_env_at(col.psfc));
    out[2] = c;
    out[3] = n;

    // Most unstable: model level of maximum theta-e in the lowest 300 hPa.
    let mut best_k = usize::MAX;
    let mut best_te = f64::NEG_INFINITY;
    for k in 0..col.nz() {
        if col.p[k] < col.psfc - 30_000.0 {
            break;
        }
        let te = theta_e_parcel(col.tk[k], col.p[k], col.r[k].max(R_FLOOR));
        if te > best_te {
            best_te = te;
            best_k = k;
        }
    }
    if best_k != usize::MAX {
        let (c, n) = lift(col, col.p[best_k], col.tk[best_k], col.r[best_k], col.tv_env(best_k));
        out[4] = c;
        out[5] = n;
    }

    // Best of six 30 hPa layers in the lowest 180 hPa; layer 0 also gives
    // the lcl_height parcel.
    let mut best = (f64::NEG_INFINITY, 0.0, 0.0, 0.0);
    for i in 0..6 {
        let p_bot = pg - 3000.0 * i as f64;
        let (th, r) = col.layer_mean(p_bot - 3000.0, p_bot);
        let pm = p_bot - 1500.0;
        let t = temperature_from_theta(th, pm);
        let te = theta_e_parcel(t, pm, r.max(R_FLOOR));
        if i == 0 {
            out[8] = lcl_height_agl(col, col.psfc, temperature_from_theta(th, col.psfc), r);
        }
        if te > best.0 {
            best = (te, pm, t, r);
        }
    }
    if best.0.is_finite() {
        let (c, n) = lift(col, best.1, best.2, best.3, col.tv_env_at(best.1));
        out[6] = c;
        out[7] = n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_parts_split_at_zero() {
        let (p, n) = layer_parts(1.0, 2.0, 0.0, -2.0);
        assert!((p - 0.5).abs() < 1e-15 && (n + 0.5).abs() < 1e-15);
        let (p, n) = layer_parts(1.0, -1.0, 0.0, 3.0);
        assert!((p - 1.125).abs() < 1e-15 && (n + 0.125).abs() < 1e-15);
    }
}
