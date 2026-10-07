//! Binary64 reference implementations, written from the specification's
//! formulas independently of the binary32 device code.
//!
//! These use the platform's binary64 functions and straightforward
//! loops.  They are the yardstick the binary32 CPU and GPU paths are tested
//! against (tolerances, not bits); they are not used to produce output.

/// Physical constants in binary64 (specification 3.2).
pub mod c {
    pub const G: f64 = 9.81;
    pub const RD: f64 = 287.0;
    pub const RV: f64 = 461.6;
    pub const CP: f64 = 3.5 * RD;
    pub const P0: f64 = 1.0e5;
    pub const EPS: f64 = RD / RV;
    pub const LV: f64 = 2.501e6;
    pub const T0C: f64 = 273.15;
}

/// Saturation vapour pressure over water, AERK [R6].
pub fn es_water(tk: f64) -> f64 {
    let tc = tk - c::T0C;
    610.94 * (17.625 * tc / (tc + 243.04)).exp()
}

/// Saturation vapour pressure over ice, AERKi [R6].
pub fn es_ice(tk: f64) -> f64 {
    let tc = tk - c::T0C;
    611.21 * (22.587 * tc / (tc + 273.86)).exp()
}

/// e from specific humidity: e = q p / (eps + (1 - eps) q).
pub fn vapour_pressure(q: f64, p: f64) -> f64 {
    q * p / (c::EPS + (1.0 - c::EPS) * q)
}

/// Dewpoint (K): inverse AERK.
pub fn dewpoint(e: f64) -> f64 {
    if e <= 0.0 {
        return f64::NAN;
    }
    let l = (e / 610.94).ln();
    243.04 * l / (17.625 - l) + c::T0C
}

/// RH (%) over water, clipped.
pub fn rh(e: f64, tk: f64) -> f64 {
    (100.0 * e / es_water(tk)).clamp(0.0, 100.0)
}

/// Virtual temperature.
pub fn virtual_temperature(tk: f64, r: f64) -> f64 {
    tk * (1.0 + r / c::EPS) / (1.0 + r)
}

/// [R9] eq. 43 with T_L from eq. 21.
pub fn theta_e(tk: f64, p: f64, r: f64) -> f64 {
    let q = r / (1.0 + r);
    let e_hpa = vapour_pressure(q, p) / 100.0;
    if e_hpa <= 0.0 {
        return f64::NAN;
    }
    let tl = 2840.0 / (3.5 * tk.ln() - e_hpa.ln() - 4.805) + 55.0;
    let rg = r * 1000.0;
    tk * (1000.0 / (p / 100.0)).powf(0.2854 * (1.0 - 0.28e-3 * rg))
        * ((3.376 / tl - 0.00254) * rg * (1.0 + 0.81e-3 * rg)).exp()
}

/// The [R4] solar zenith cosine at a Julian date, in binary64 throughout.
pub fn cosz(julian_date: f64, ut_hours: f64, lat_deg: f64, lon_deg: f64) -> f64 {
    let rad = std::f64::consts::PI / 180.0;
    let n = julian_date - 2451545.0;
    let l = (280.460 + 0.9856474 * n).rem_euclid(360.0);
    let g = ((357.528 + 0.9856003 * n).rem_euclid(360.0)) * rad;
    let lam = (l + 1.915 * g.sin() + 0.020 * (2.0 * g).sin()) * rad;
    let eps = (23.439 - 0.0000004 * n) * rad;
    let ra = (eps.cos() * lam.sin()).atan2(lam.cos());
    let dec = (eps.sin() * lam.sin()).asin();
    let gmst = (6.697375 + 0.0657098242 * n + ut_hours).rem_euclid(24.0);
    let ha = (15.0 * gmst + lon_deg) * rad - ra;
    let lat = lat_deg * rad;
    dec.sin() * lat.sin() + dec.cos() * lat.cos() * ha.cos()
}

/// Column state of one column in binary64, from the same carriers.
pub struct ColumnRef {
    pub theta: Vec<f64>,
    pub q: Vec<f64>,
    pub p: Vec<f64>,
    pub p_full: Vec<f64>,
    pub tk: Vec<f64>,
    pub tv: Vec<f64>,
    pub z_mass: Vec<f64>,
    pub p_int: Vec<f64>,
    pub z_int: Vec<f64>,
}

/// Inputs of one column, bottom-up.
pub struct ColumnIn<'a> {
    pub t_pert: &'a [f64],
    pub p_pert: &'a [f64],
    pub p_base: &'a [f64],
    pub qv: &'a [f64],
    /// Every other water species present (each bottom-up).
    pub condensate: &'a [Vec<f64>],
    pub p_hyd: Option<&'a [f64]>,
    pub ph: &'a [f64],
    pub phb: &'a [f64],
    pub mu_d: f64,
    pub c3f: &'a [f64],
    pub c4f: &'a [f64],
    pub p_top: f64,
}

/// Hydrostatic column in binary64 ([R1] 2.1 to 2.3): the dry layer weights
/// from C3F/C4F, moistened by (1 + total water), summed from the top.
pub fn column(inp: &ColumnIn) -> ColumnRef {
    let nz = inp.t_pert.len();
    let pd: Vec<f64> = (0..=nz).map(|k| inp.c3f[k] * inp.mu_d + inp.c4f[k] + inp.p_top).collect();
    let mut p_int = vec![0.0; nz + 1];
    let mut p = vec![0.0; nz];
    p_int[nz] = inp.p_top;
    for k in (0..nz).rev() {
        let mut qt = inp.qv[k].max(0.0);
        for s in inp.condensate {
            qt += s[k].max(0.0);
        }
        let w = (pd[k] - pd[k + 1]) * (1.0 + qt);
        p[k] = p_int[k + 1] + 0.5 * w;
        p_int[k] = p_int[k + 1] + w;
    }
    if let Some(ph) = inp.p_hyd {
        p.copy_from_slice(ph);
    }
    let z_int: Vec<f64> = (0..=nz).map(|k| (inp.ph[k] + inp.phb[k]) / c::G).collect();
    let mut out = ColumnRef {
        theta: vec![0.0; nz],
        q: vec![0.0; nz],
        p,
        p_full: vec![0.0; nz],
        tk: vec![0.0; nz],
        tv: vec![0.0; nz],
        z_mass: vec![0.0; nz],
        p_int,
        z_int: z_int.clone(),
    };
    for k in 0..nz {
        let theta = inp.t_pert[k] + 300.0;
        let r = inp.qv[k].max(0.0);
        let pf = inp.p_pert[k] + inp.p_base[k];
        let tk = theta * (pf / c::P0).powf(c::RD / c::CP);
        out.theta[k] = theta;
        out.q[k] = r / (1.0 + r);
        out.p_full[k] = pf;
        out.tk[k] = tk;
        out.tv[k] = virtual_temperature(tk, r);
        out.z_mass[k] = 0.5 * (z_int[k] + z_int[k + 1]);
    }
    out
}

/// Shelter fields in binary64: (p2, t2, q2, td2, rh2).
pub fn shelter(psfc: f64, tv_lowest: f64, t2: Option<f64>, th2: f64, q2_mixing: f64) -> (f64, f64, f64, f64, f64) {
    let p2 = psfc * (-c::G * 2.0 / (c::RD * tv_lowest)).exp();
    let t2 = t2.unwrap_or(th2 * (p2 / c::P0).powf(c::RD / c::CP));
    let r = q2_mixing.max(0.0);
    let q2 = r / (1.0 + r);
    let e = vapour_pressure(q2, p2);
    let td = dewpoint(e).min(t2);
    (p2, t2, q2, td, rh(e, t2))
}
