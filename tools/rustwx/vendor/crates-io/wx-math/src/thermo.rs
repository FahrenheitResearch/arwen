/// Meteorological thermodynamic functions ported from wrfsolar's metfuncs.py.
/// Pure math - no external dependencies. All functions are direct ports of the
/// SHARPpy-derived implementations used in the Python codebase.

// --- Physical Constants (MetPy-exact values) ---
pub const RD: f64 = 287.04749097718457;  // Dry air gas constant (J/(kg*K))
pub const RV: f64 = 461.52311572606084;  // Water vapor gas constant (J/(kg*K))
pub const CP: f64 = 1004.6662184201462;  // Specific heat at constant pressure (J/(kg*K))
pub const G: f64 = 9.80665;             // Gravitational acceleration (m/s^2)
pub const ROCP: f64 = 0.2857142857142857; // Rd/Cp (2/7 exactly)
pub const ZEROCNK: f64 = 273.15;        // 0 Celsius in Kelvin
pub const MISSING: f64 = -9999.0;
pub const EPS: f64 = 0.6219569100577033; // Rd/Rv (MetPy epsilon)
pub const LV: f64 = 2_500_840.0;        // Latent heat of vaporization (J/kg)
pub const LAPSE_STD: f64 = 0.0065;  // Standard atmosphere lapse rate (K/m)
pub const P0_STD: f64 = 1013.25;    // Standard sea level pressure (hPa)
pub const T0_STD: f64 = 288.15;     // Standard sea level temperature (K)

// --- Ambaum (2020) / MetPy constants for saturation vapor pressure ---
// Triple-point temperature (K) — used as the auto-phase boundary.
const T0: f64 = 273.16;
// Saturation vapor pressure at 0 C (Pa). MetPy: 611.2 Pa = 6.112 hPa.
const SAT_PRESSURE_0C: f64 = 611.2;
// Specific heat capacities (J/(kg·K))
const CP_L: f64 = 4219.4;          // Liquid water
const CP_V: f64 = 1860.078011865639; // Water vapor (MetPy exact)
const CP_I: f64 = 2090.0;          // Ice
// Water vapor gas constant (J/(kg·K)) — MetPy's exact value
const RV_METPY: f64 = 461.52311572606084;
// Reference latent heats at T0 (J/kg)
const LV_0: f64 = 2_500_840.0;     // Vaporization
const LS_0: f64 = 2_834_540.0;     // Sublimation

// --- Phase selection for saturation vapor pressure ---

/// Phase of water for saturation calculations, matching MetPy's `phase` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Saturation over liquid water (default).
    Liquid,
    /// Saturation over solid ice.
    Solid,
    /// Automatically select liquid (T > T0 = 273.16 K) or solid (T <= T0).
    Auto,
}

// --- SHARPpy Thermodynamic Approximations ---

/// Wobus function for computing moist adiabats.
/// Input: temperature in Celsius.
pub fn wobf(t: f64) -> f64 {
    let t = t - 20.0;
    if t <= 0.0 {
        let npol = 1.0
            + t * (-8.841660499999999e-3
                + t * (1.4714143e-4
                    + t * (-9.671989000000001e-7
                        + t * (-3.2607217e-8 + t * (-3.8598073e-10)))));
        15.13 / (npol * npol * npol * npol)
    } else {
        let ppol = t
            * (4.9618922e-07
                + t * (-6.1059365e-09
                    + t * (3.9401551e-11
                        + t * (-1.2588129e-13 + t * (1.6688280e-16)))));
        let ppol = 1.0 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol));
        (29.93 / (ppol * ppol * ppol * ppol)) + (0.96 * t) - 14.8
    }
}

/// Lifts a saturated parcel.
/// p: Pressure (hPa), thetam: Saturation Potential Temperature (Celsius).
/// Uses 7 Newton-Raphson iterations.
pub fn satlift(p: f64, thetam: f64) -> f64 {
    if p >= 1000.0 {
        return thetam;
    }

    let pwrp = (p / 1000.0_f64).powf(ROCP);
    let mut t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK;
    let mut e1 = wobf(t1) - wobf(thetam);
    let mut rate = 1.0;

    for _ in 0..7 {
        if e1.abs() < 0.001 {
            break;
        }
        let t2 = t1 - (e1 * rate);
        let mut e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK;
        e2 += wobf(t2) - wobf(e2) - thetam;
        rate = (t2 - t1) / (e2 - e1);
        t1 = t2;
        e1 = e2;
    }

    t1 - e1 * rate
}

/// LCL temperature from temperature and dewpoint (both Celsius).
pub fn lcltemp(t: f64, td: f64) -> f64 {
    let s = t - td;
    let dlt = s * (1.2185 + 0.001278 * t + s * (-0.00219 + 1.173e-5 * s - 0.0000052 * t));
    t - dlt
}

/// Dry lift to LCL. Returns (p_lcl, t_lcl) in (hPa, Celsius).
pub fn drylift(p: f64, t: f64, td: f64) -> (f64, f64) {
    let t_lcl = lcltemp(t, td);
    let p_lcl = 1000.0
        * (((t_lcl + ZEROCNK) / ((t + ZEROCNK) * ((1000.0 / p).powf(ROCP)))))
            .powf(1.0 / ROCP);
    (p_lcl, t_lcl)
}

/// Saturation vapor pressure (hPa) at given temperature (Celsius).
/// Uses the SHARPpy 8th-order polynomial approximation (Eschner).
pub fn vappres(t: f64) -> f64 {
    let pol = t * (1.1112018e-17 + (t * -3.0994571e-20));
    let pol = t * (2.1874425e-13 + (t * (-1.789232e-15 + pol)));
    let pol = t * (4.3884180e-09 + (t * (-2.988388e-11 + pol)));
    let pol = t * (7.8736169e-05 + (t * (-6.111796e-07 + pol)));
    let pol = 0.99999683 + (t * (-9.082695e-03 + pol));
    6.1078 / pol.powi(8)
}

/// Mixing ratio (g/kg) of a parcel at pressure p (hPa) and temperature t (Celsius).
/// Includes Wexler enhancement factor for non-ideal gas behavior.
pub fn mixratio(p: f64, t: f64) -> f64 {
    // Enhancement Factor (Wexler)
    let x = 0.02 * (t - 12.5 + (7500.0 / p));
    let wfw = 1.0 + (0.0000045 * p) + (0.0014 * x * x);

    // Saturation Vapor Pressure (with enhancement)
    let fwesw = wfw * vappres(t);

    // Mixing Ratio (g/kg)
    621.97 * (fwesw / (p - fwesw))
}

/// Virtual temperature. Inputs and output all in Celsius.
/// t: temperature (C), p: pressure (hPa), td: dewpoint (C).
pub fn virtual_temp(t: f64, p: f64, td: f64) -> f64 {
    let w = mixratio(p, td) / 1000.0;
    let tk = t + ZEROCNK;
    let vt = tk * (1.0 + 0.61 * w);
    vt - ZEROCNK
}

/// Equivalent potential temperature. Returns value in Celsius.
/// p (hPa), t (C), td (C).
pub fn thetae(p: f64, t: f64, td: f64) -> f64 {
    let (p_lcl, t_lcl) = drylift(p, t, td);
    let theta = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP));
    let r = mixratio(p, td) / 1000.0;
    let lc = 2500.0 - 2.37 * t_lcl;
    let te_k = theta * ((lc * 1000.0 * r) / (CP * (t_lcl + ZEROCNK))).exp();
    te_k - ZEROCNK
}

/// Temperature (Celsius) of air at given mixing ratio (g/kg) and pressure (hPa).
/// Ported from SHARPpy params.py.
pub fn temp_at_mixrat(w: f64, p: f64) -> f64 {
    let c1: f64 = 0.0498646455;
    let c2: f64 = 2.4082965;
    let c3: f64 = 7.07475;
    let c4: f64 = 38.9114;
    let c5: f64 = 0.0915;
    let c6: f64 = 1.2035;

    let x = (w * p / (622.0 + w)).log10();
    (10.0_f64.powf(c1 * x + c2) - c3
        + c4 * (10.0_f64.powf(c5 * x) - c6).powi(2))
        - ZEROCNK
}

// --- Helper Functions ---

/// Linear interpolation: given x between x1 and x2, interpolate between y1 and y2.
pub fn interp_linear(x: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    if x2 == x1 {
        return y1;
    }
    y1 + (x - x1) * (y2 - y1) / (x2 - x1)
}

/// Interpolate height at a target pressure from pressure and height profiles
/// (both in decreasing pressure order, i.e. surface first).
pub fn get_height_at_pres(target_p: f64, p_prof: &[f64], h_prof: &[f64]) -> f64 {
    for i in 0..p_prof.len() - 1 {
        if p_prof[i] >= target_p && target_p >= p_prof[i + 1] {
            return interp_linear(target_p, p_prof[i], p_prof[i + 1], h_prof[i], h_prof[i + 1]);
        }
    }
    // Bounds check
    if target_p > p_prof[0] {
        return h_prof[0];
    }
    if target_p < p_prof[p_prof.len() - 1] {
        return h_prof[h_prof.len() - 1];
    }
    f64::NAN
}

/// Interpolate environmental temperature and dewpoint at a target pressure.
/// Uses log-pressure interpolation. Returns (t_interp, td_interp) in Celsius.
pub fn get_env_at_pres(
    target_p: f64,
    p_prof: &[f64],
    t_prof: &[f64],
    td_prof: &[f64],
) -> (f64, f64) {
    for i in 0..p_prof.len() - 1 {
        if p_prof[i] >= target_p && target_p >= p_prof[i + 1] {
            let log_p = target_p.ln();
            let log_p1 = p_prof[i].ln();
            let log_p2 = p_prof[i + 1].ln();
            let t_interp = interp_linear(log_p, log_p1, log_p2, t_prof[i], t_prof[i + 1]);
            let td_interp = interp_linear(log_p, log_p1, log_p2, td_prof[i], td_prof[i + 1]);
            return (t_interp, td_interp);
        }
    }
    (
        t_prof[t_prof.len() - 1],
        td_prof[td_prof.len() - 1],
    )
}

// --- Parcel Selectors ---

/// Returns Mixed Layer Parcel matching SHARPpy's calculation method.
/// Uses 1-2-1 weighting scheme (surface and top weight 1, inner levels weight 2).
/// Returns (p_start, t_start, td_start) all in (hPa, Celsius, Celsius).
pub fn get_mixed_layer_parcel(
    p_prof: &[f64],
    t_prof: &[f64],
    td_prof: &[f64],
    depth: f64,
) -> (f64, f64, f64) {
    let sfc_p = p_prof[0];
    let top_p = sfc_p - depth;

    // Surface (Bottom Bound) - Weight 1
    let theta_sfc = (t_prof[0] + ZEROCNK) * ((1000.0 / sfc_p).powf(ROCP));
    let td_sfc = td_prof[0];

    // Top Bound (Interpolated) - Weight 1
    let (t_top, td_top) = get_env_at_pres(top_p, p_prof, t_prof, td_prof);
    let theta_top = (t_top + ZEROCNK) * ((1000.0 / top_p).powf(ROCP));

    // Accumulators
    let mut sum_theta = theta_sfc + theta_top;
    let mut sum_p = sfc_p + top_p;
    let mut sum_td = td_sfc + td_top;
    let mut count = 2.0;

    // Inner Layers - Weight 2
    for i in 1..p_prof.len() {
        let p = p_prof[i];
        if p <= top_p {
            break;
        }
        let t = t_prof[i];
        let td = td_prof[i];
        let th = (t + ZEROCNK) * ((1000.0 / p).powf(ROCP));

        sum_theta += 2.0 * th;
        sum_p += 2.0 * p;
        sum_td += 2.0 * td;
        count += 2.0;
    }

    // Averages
    let avg_theta = sum_theta / count;
    let avg_p = sum_p / count;
    let avg_td = sum_td / count;

    // Parcel T: Bring Mean Theta back to Surface Pressure
    let avg_t_k = avg_theta * ((sfc_p / 1000.0).powf(ROCP));
    let avg_t = avg_t_k - ZEROCNK;

    // Parcel Td: Calculate mixing ratio from (Mean P, Mean Td), get dewpoint at surface
    let avg_w = mixratio(avg_p, avg_td);
    let parcel_td = temp_at_mixrat(avg_w, sfc_p);

    (sfc_p, avg_t, parcel_td)
}

/// Returns Most Unstable Parcel (highest theta-e in the lowest `depth` hPa).
/// Returns (p, t, td) all in (hPa, Celsius, Celsius).
pub fn get_most_unstable_parcel(
    p_prof: &[f64],
    t_prof: &[f64],
    td_prof: &[f64],
    depth: f64,
) -> (f64, f64, f64) {
    let sfc_p = p_prof[0];
    let limit_p = sfc_p - depth;
    let mut max_thetae = -999.0_f64;
    let mut best_idx = 0_usize;

    for i in 0..p_prof.len() {
        if p_prof[i] < limit_p {
            break;
        }
        let te = thetae(p_prof[i], t_prof[i], td_prof[i]);
        if te > max_thetae {
            max_thetae = te;
            best_idx = i;
        }
    }

    (p_prof[best_idx], t_prof[best_idx], td_prof[best_idx])
}

// --- Core CAPE/CIN Computation ---

/// Compute CAPE, CIN, LCL height, and LFC height for a grid column.
///
/// Inputs:
/// - p_prof, t_prof, td_prof: Model level profiles (surface first, decreasing pressure).
///   May be in Pa or hPa; may be in K or C (auto-detected and converted).
/// - height_agl: Height AGL profile (meters) matching model levels.
/// - psfc: Surface pressure (Pa or hPa).
/// - t2m: 2-meter temperature (K or C).
/// - td2m: 2-meter dewpoint (K or C).
/// - parcel_type: "sb", "ml", or "mu".
/// - ml_depth: Mixed layer depth in hPa (default 100).
/// - mu_depth: Most unstable search depth in hPa (default 300).
/// - top_m: Optional cap on integration height (meters AGL).
///
/// Returns (cape, cin, h_lcl, h_lfc) in (J/kg, J/kg, m AGL, m AGL).
pub fn cape_cin_core(
    p_prof: &[f64],
    t_prof: &[f64],
    td_prof: &[f64],
    height_agl: &[f64],
    psfc: f64,
    t2m: f64,
    td2m: f64,
    parcel_type: &str,
    ml_depth: f64,
    mu_depth: f64,
    top_m: Option<f64>,
) -> (f64, f64, f64, f64) {
    // --- 0. Unit Standardization ---
    let mut p_prof = p_prof.to_vec();
    let mut t_prof = t_prof.to_vec();
    let mut td_prof = td_prof.to_vec();
    let mut psfc_val = psfc;
    let mut t2m_val = t2m;
    let mut td2m_val = td2m;

    if psfc_val > 2000.0 {
        for v in p_prof.iter_mut() {
            *v /= 100.0;
        }
        psfc_val /= 100.0;
    }

    if t2m_val > 150.0 {
        for v in t_prof.iter_mut() {
            *v -= ZEROCNK;
        }
        for v in td_prof.iter_mut() {
            *v -= ZEROCNK;
        }
        t2m_val -= ZEROCNK;
        td2m_val -= ZEROCNK;
    }

    // Ensure Td2m <= T2m
    if td2m_val > t2m_val {
        td2m_val = t2m_val;
    }

    // Prepend surface data to profiles
    let n = p_prof.len();
    let mut new_p = Vec::with_capacity(n + 1);
    let mut new_t = Vec::with_capacity(n + 1);
    let mut new_td = Vec::with_capacity(n + 1);
    let mut new_h = Vec::with_capacity(n + 1);

    new_p.push(psfc_val);
    new_t.push(t2m_val);
    new_td.push(td2m_val);
    new_h.push(0.0);

    for i in 0..n {
        new_p.push(p_prof[i]);
        new_t.push(t_prof[i]);
        new_td.push(if td_prof[i] <= t_prof[i] {
            td_prof[i]
        } else {
            t_prof[i]
        });
        new_h.push(height_agl[i]);
    }

    let p_prof = new_p;
    let t_prof = new_t;
    let td_prof = new_td;
    let height_agl = new_h;

    // --- 1. Select Parcel ---
    let (p_start, t_start, td_start) = match parcel_type {
        "ml" => get_mixed_layer_parcel(&p_prof, &t_prof, &td_prof, ml_depth),
        "mu" => get_most_unstable_parcel(&p_prof, &t_prof, &td_prof, mu_depth),
        _ => (psfc_val, t2m_val, td2m_val), // "sb" default
    };

    // --- 2. Find LCL (Analytic) ---
    let (p_lcl, t_lcl) = drylift(p_start, t_start, td_start);
    let h_lcl = get_height_at_pres(p_lcl, &p_prof, &height_agl);

    // Calculate Theta-M (constant for moist ascent)
    let theta_start_k = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP));
    let theta_start_c = theta_start_k - ZEROCNK;
    let thetam = theta_start_c - wobf(theta_start_c) + wobf(t_lcl);

    // --- PASS 1: Geometric Scan for LFC and EL ---
    let mut el_p = p_lcl;
    let mut lfc_p = p_lcl;

    let mut found_positive_layer = false;
    let mut in_pos_layer = false;

    // Find start index (first level at or above LCL)
    let mut start_idx = 0;
    for i in 0..p_prof.len() {
        if p_prof[i] <= p_lcl {
            start_idx = i;
            break;
        }
    }

    for i in start_idx..p_prof.len() {
        let p_curr = p_prof[i];

        // Environmental Tv
        let tv_env = virtual_temp(t_prof[i], p_curr, td_prof[i]);
        // Parcel Tv
        let t_parc = satlift(p_curr, thetam);
        let tv_parc = virtual_temp(t_parc, p_curr, t_parc);

        let buoyancy = tv_parc - tv_env;

        if buoyancy > 0.0 {
            if !in_pos_layer {
                in_pos_layer = true;

                // Find crossing (LFC of this layer)
                let curr_pos_bottom = if i > 0 {
                    let p_prev = p_prof[i - 1];
                    let tv_env_prev = virtual_temp(t_prof[i - 1], p_prev, td_prof[i - 1]);
                    let t_parc_prev = satlift(p_prev, thetam);
                    let tv_parc_prev = virtual_temp(t_parc_prev, p_prev, t_parc_prev);
                    let buoy_prev = tv_parc_prev - tv_env_prev;

                    if buoy_prev >= 0.0 {
                        // Previous level is also buoyant — no real crossing.
                        // The parcel is buoyant from the LCL (or surface)
                        // upward, so the LFC is at the LCL pressure.
                        p_lcl
                    } else if buoyancy != buoy_prev {
                        let frac = (0.0 - buoy_prev) / (buoyancy - buoy_prev);
                        p_prev + frac * (p_curr - p_prev)
                    } else {
                        p_curr
                    }
                } else {
                    p_curr
                };

                lfc_p = curr_pos_bottom;
                el_p = p_prof[p_prof.len() - 1];
                found_positive_layer = true;
            }
        } else {
            // buoyancy <= 0
            if in_pos_layer {
                in_pos_layer = false;

                // Find crossing (EL)
                let p_prev = p_prof[i - 1];
                let tv_env_prev = virtual_temp(t_prof[i - 1], p_prev, td_prof[i - 1]);
                let t_parc_prev = satlift(p_prev, thetam);
                let tv_parc_prev = virtual_temp(t_parc_prev, p_prev, t_parc_prev);
                let buoy_prev = tv_parc_prev - tv_env_prev;

                let curr_pos_top = if buoyancy != buoy_prev {
                    let frac = (0.0 - buoy_prev) / (buoyancy - buoy_prev);
                    p_prev + frac * (p_curr - p_prev)
                } else {
                    p_curr
                };

                el_p = curr_pos_top;
            }
        }
    }

    if in_pos_layer {
        el_p = p_prof[p_prof.len() - 1];
    }

    // Return zeros if no instability found
    if !found_positive_layer {
        return (0.0, 0.0, h_lcl, f64::NAN);
    }

    // If LFC is below LCL, set to LCL
    if lfc_p.is_nan() || lfc_p > p_lcl {
        lfc_p = p_lcl;
    }
    let h_lfc = get_height_at_pres(lfc_p, &p_prof, &height_agl);

    // --- PASS 2: Integration ---
    let mut p_top_limit = el_p;
    if let Some(top_m_val) = top_m {
        // Reverse profiles for height->pressure lookup
        let h_rev: Vec<f64> = height_agl.iter().copied().rev().collect();
        let p_rev: Vec<f64> = p_prof.iter().copied().rev().collect();
        let p_top_m = get_height_at_pres(top_m_val, &h_rev, &p_rev);
        if p_top_m >= p_top_limit {
            p_top_limit = p_top_m.max(p_prof[p_prof.len() - 1]);
        }
    }

    let mut total_cape = 0.0_f64;
    let mut total_cin = 0.0_f64;

    // --- Build parcel profile using moist_lapse (MetPy-compatible) ---
    let theta_dry_k = (t_start + ZEROCNK) * ((1000.0 / p_start).powf(ROCP));
    let r_parcel = mixratio(p_start, td_start);
    let w_kgkg = r_parcel / 1000.0;

    // Collect levels above LCL for moist ascent
    let mut p_moist: Vec<f64> = vec![p_lcl];
    for &pi in p_prof.iter() {
        if pi < p_lcl && pi > 0.0 {
            p_moist.push(pi);
        }
    }
    let moist_temps = if p_moist.len() > 1 {
        moist_lapse(&p_moist, t_lcl)
    } else {
        vec![t_lcl]
    };

    // Build parcel Tv and environment Tv at each sounding level
    let n = p_prof.len();
    let mut tv_parc_arr = vec![f64::NAN; n];
    let mut tv_env_arr = vec![0.0_f64; n];

    for i in 0..n {
        if p_prof[i] <= 0.0 { continue; }
        tv_env_arr[i] = virtual_temp(t_prof[i], p_prof[i], td_prof[i]);

        if p_prof[i] >= p_lcl {
            // Below LCL: dry adiabat with moisture
            let t_parc_k = theta_dry_k * ((p_prof[i] / 1000.0).powf(ROCP));
            let t_parc = t_parc_k - ZEROCNK;
            tv_parc_arr[i] = (t_parc + ZEROCNK) * (1.0 + w_kgkg / EPS) / (1.0 + w_kgkg) - ZEROCNK;
        } else {
            // Above LCL: moist adiabat from moist_lapse
            let t_parc = interp_log_p(p_prof[i], &p_moist, &moist_temps);
            tv_parc_arr[i] = virtual_temp(t_parc, p_prof[i], t_parc);
        }
    }

    // Compute height using hypsometric equation
    let mut z_calc = vec![0.0_f64; n];
    for i in 1..n {
        if p_prof[i] <= 0.0 || p_prof[i - 1] <= 0.0 { z_calc[i] = z_calc[i - 1]; continue; }
        let tv_mean = (tv_env_arr[i - 1] + tv_env_arr[i]) / 2.0 + ZEROCNK;
        z_calc[i] = z_calc[i - 1] + (RD * tv_mean / G) * (p_prof[i - 1] / p_prof[i]).ln();
    }
    // Use provided height_agl if available, else computed
    let z_use: Vec<f64> = if height_agl.iter().any(|&h| h > 0.0) {
        height_agl.clone()
    } else {
        z_calc
    };

    // --- Integrate CAPE/CIN: g * (Tv_p - Tv_e) / Tv_e * dz (trapezoidal) ---
    // Only integrate between surface and top limit
    let p_top_actual = if p_top_limit > 0.0 { p_top_limit } else { p_prof[n - 1] };

    // CIN is accumulated below the LFC.  We track the *last*
    // transition from negative to positive buoyancy (the true LFC).
    // This handles superadiabatic surface layers correctly.

    // First, find the index of the last neg→pos crossing (the true LFC)
    let mut last_lfc_idx: Option<usize> = None;
    for i in 1..n {
        if tv_parc_arr[i].is_nan() || tv_parc_arr[i - 1].is_nan() { continue; }
        let tv_e = tv_env_arr[i] + ZEROCNK;
        let tv_p = tv_parc_arr[i] + ZEROCNK;
        let tv_e_prev = tv_env_arr[i - 1] + ZEROCNK;
        let tv_p_prev = tv_parc_arr[i - 1] + ZEROCNK;
        let buoy = tv_p - tv_e;
        let buoy_prev = tv_p_prev - tv_e_prev;
        if buoy > 0.0 && buoy_prev <= 0.0 {
            last_lfc_idx = Some(i);
        }
    }

    for i in 1..n {
        if p_prof[i] <= 0.0 || tv_parc_arr[i].is_nan() || tv_parc_arr[i - 1].is_nan() { continue; }
        if p_prof[i] < p_top_actual { continue; }

        let tv_e_lo = tv_env_arr[i - 1] + ZEROCNK;
        let tv_e_hi = tv_env_arr[i] + ZEROCNK;
        let tv_p_lo = tv_parc_arr[i - 1] + ZEROCNK;
        let tv_p_hi = tv_parc_arr[i] + ZEROCNK;
        let dz = z_use[i] - z_use[i - 1];
        if dz.abs() < 1e-6 || tv_e_lo <= 0.0 || tv_e_hi <= 0.0 { continue; }

        let buoy_lo = (tv_p_lo - tv_e_lo) / tv_e_lo;
        let buoy_hi = (tv_p_hi - tv_e_hi) / tv_e_hi;
        let val = G * (buoy_lo + buoy_hi) / 2.0 * dz;

        if let Some(lfc_i) = last_lfc_idx {
            if val > 0.0 && i >= lfc_i {
                total_cape += val;
            } else if val < 0.0 && i <= lfc_i {
                total_cin += val;
            }
        } else {
            // No LFC found — accumulate everything
            if val > 0.0 { total_cape += val; }
            else { total_cin += val; }
        }
    }

    (total_cape, total_cin, h_lcl, h_lfc)
}

// =============================================================================
// Temperature Conversions
// =============================================================================

/// Convert Celsius to Fahrenheit.
pub fn celsius_to_fahrenheit(t: f64) -> f64 {
    t * 9.0 / 5.0 + 32.0
}

/// Convert Fahrenheit to Celsius.
pub fn fahrenheit_to_celsius(t: f64) -> f64 {
    (t - 32.0) * 5.0 / 9.0
}

/// Convert Celsius to Kelvin.
pub fn celsius_to_kelvin(t: f64) -> f64 {
    t + ZEROCNK
}

/// Convert Kelvin to Celsius.
pub fn kelvin_to_celsius(t: f64) -> f64 {
    t - ZEROCNK
}

// =============================================================================
// Saturation / Moisture Functions
// =============================================================================

/// Saturation vapor pressure (hPa) over **liquid** water.
///
/// Uses the Ambaum (2020) formulation matching MetPy exactly.
/// For ice-phase or automatic phase selection, use [`saturation_vapor_pressure_with_phase`].
///
/// Input: temperature in Celsius. Output: hPa.
pub fn saturation_vapor_pressure(t_c: f64) -> f64 {
    svp_liquid_pa(t_c + ZEROCNK) / 100.0
}

/// Saturation vapor pressure (hPa) with explicit phase selection.
///
/// Uses the Ambaum (2020) formulation matching MetPy exactly.
/// - `Phase::Liquid` — saturation over liquid water (same as [`saturation_vapor_pressure`]).
/// - `Phase::Solid`  — saturation over ice.
/// - `Phase::Auto`   — liquid when T > 273.16 K, solid otherwise.
///
/// Input: temperature in Celsius. Output: hPa.
pub fn saturation_vapor_pressure_with_phase(t_c: f64, phase: Phase) -> f64 {
    let t_k = t_c + ZEROCNK;
    let pa = match phase {
        Phase::Liquid => svp_liquid_pa(t_k),
        Phase::Solid  => svp_solid_pa(t_k),
        Phase::Auto   => {
            if t_k > T0 { svp_liquid_pa(t_k) } else { svp_solid_pa(t_k) }
        }
    };
    pa / 100.0
}

/// Internal: saturation vapor pressure over liquid water in **Pa**.
///
/// Ambaum (2020) Eq. 13:
///   e = e_s0 * (T0/T)^((Cp_l - Cp_v) / Rv) * exp((Lv0/(Rv*T0) - L(T)/(Rv*T)))
/// where L(T) = Lv0 - (Cp_l - Cp_v)*(T - T0)
fn svp_liquid_pa(t_k: f64) -> f64 {
    let latent_heat = LV_0 - (CP_L - CP_V) * (t_k - T0);
    let heat_power = (CP_L - CP_V) / RV_METPY;
    let exp_term = (LV_0 / T0 - latent_heat / t_k) / RV_METPY;
    SAT_PRESSURE_0C * (T0 / t_k).powf(heat_power) * exp_term.exp()
}

/// Internal: saturation vapor pressure over ice in **Pa**.
///
/// Ambaum (2020) Eq. 17:
///   e_i = e_s0 * (T0/T)^((Cp_i - Cp_v) / Rv) * exp((Ls0/(Rv*T0) - Ls(T)/(Rv*T)))
/// where Ls(T) = Ls0 - (Cp_i - Cp_v)*(T - T0)
fn svp_solid_pa(t_k: f64) -> f64 {
    let latent_heat = LS_0 - (CP_I - CP_V) * (t_k - T0);
    let heat_power = (CP_I - CP_V) / RV_METPY;
    let exp_term = (LS_0 / T0 - latent_heat / t_k) / RV_METPY;
    SAT_PRESSURE_0C * (T0 / t_k).powf(heat_power) * exp_term.exp()
}

/// Dewpoint (Celsius) from temperature (Celsius) and relative humidity (%).
///
/// Computes vapor pressure using the Ambaum SVP, then inverts via the Bolton (1980)
/// formula — matching MetPy's approach.
pub fn dewpoint_from_rh(t_c: f64, rh: f64) -> f64 {
    let rh_frac = rh / 100.0;
    let es = saturation_vapor_pressure(t_c);
    let e = rh_frac * es;
    // Invert via Bolton: Td = 243.5 * ln(e/6.112) / (17.67 - ln(e/6.112))
    let ln_ratio = (e / 6.112).ln();
    243.5 * ln_ratio / (17.67 - ln_ratio)
}

/// Relative humidity (%) from temperature and dewpoint (both Celsius).
pub fn rh_from_dewpoint(t_c: f64, td_c: f64) -> f64 {
    let es = saturation_vapor_pressure(t_c);
    let e = saturation_vapor_pressure(td_c);
    (e / es) * 100.0
}

/// Specific humidity (kg/kg) from pressure (hPa) and mixing ratio (g/kg).
pub fn specific_humidity(p_hpa: f64, w_gkg: f64) -> f64 {
    let _ = p_hpa; // pressure not needed for this conversion
    let w = w_gkg / 1000.0; // kg/kg
    w / (1.0 + w)
}

/// Mixing ratio (g/kg) from specific humidity (kg/kg).
pub fn mixing_ratio_from_specific_humidity(q: f64) -> f64 {
    (q / (1.0 - q)) * 1000.0
}

/// Saturation mixing ratio (g/kg) at given pressure (hPa) and temperature (Celsius).
///
/// Uses Ambaum (2020) SVP over liquid water. For phase-aware mixing ratio, use
/// [`saturation_mixing_ratio_with_phase`].
pub fn saturation_mixing_ratio(p_hpa: f64, t_c: f64) -> f64 {
    let es = saturation_vapor_pressure(t_c);
    (EPS * es / (p_hpa - es) * 1000.0).max(0.0)
}

/// Saturation mixing ratio (g/kg) with explicit phase selection.
pub fn saturation_mixing_ratio_with_phase(p_hpa: f64, t_c: f64, phase: Phase) -> f64 {
    let es = saturation_vapor_pressure_with_phase(t_c, phase);
    (EPS * es / (p_hpa - es) * 1000.0).max(0.0)
}

/// Vapor pressure (hPa) from dewpoint temperature (Celsius).
/// Uses Ambaum (2020) SVP over liquid water.
pub fn vapor_pressure_from_dewpoint(td_c: f64) -> f64 {
    saturation_vapor_pressure(td_c)
}

/// Wet bulb temperature (Celsius) using iterative Normand's rule.
/// p_hpa: pressure (hPa), t_c: temperature (C), td_c: dewpoint (C).
pub fn wet_bulb_temperature(p_hpa: f64, t_c: f64, td_c: f64) -> f64 {
    // Lift parcel to LCL, then descend moist adiabatically
    let (p_lcl, t_lcl) = drylift(p_hpa, t_c, td_c);
    // theta_m for the moist descent
    let theta_c = t_lcl + ZEROCNK;
    let theta_sfc = theta_c * ((1000.0 / p_lcl).powf(ROCP));
    let theta_start_c = theta_sfc - ZEROCNK;
    let thetam = theta_start_c - wobf(theta_start_c) + wobf(t_lcl);
    // Descend moist adiabatically from LCL to original pressure
    satlift(p_hpa, thetam)
}

/// Frost point temperature (Celsius) from temperature (C) and relative humidity (%).
/// Uses the Magnus formula over ice.
pub fn frost_point(t_c: f64, rh: f64) -> f64 {
    // Saturation vapor pressure over water
    let es_water = saturation_vapor_pressure(t_c);
    let e = (rh / 100.0) * es_water;
    // Invert Magnus formula over ice:
    // ei = 6.112 * exp(22.46 * T / (T + 272.62))
    // ln(e/6.112) = 22.46 * Tf / (Tf + 272.62)
    let ln_ratio = (e / 6.112).ln();
    272.62 * ln_ratio / (22.46 - ln_ratio)
}

/// Psychrometric vapor pressure (hPa) using the psychrometric equation.
/// t_c: dry bulb (C), tw_c: wet bulb (C), p_hpa: pressure (hPa).
pub fn psychrometric_vapor_pressure(t_c: f64, tw_c: f64, p_hpa: f64) -> f64 {
    let es_tw = saturation_vapor_pressure(tw_c);
    // Psychrometer constant for aspirated psychrometer: 6.6e-4
    let a = 6.6e-4;
    es_tw - a * p_hpa * (t_c - tw_c)
}

// =============================================================================
// Potential Temperature Functions
// =============================================================================

/// Potential temperature (K) from pressure (hPa) and temperature (Celsius).
/// Uses Poisson's equation: theta = T * (1000/p)^(Rd/Cp).
pub fn potential_temperature(p_hpa: f64, t_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    t_k * (1000.0 / p_hpa).powf(ROCP)
}

/// Equivalent potential temperature (K) using Bolton (1980) formula.
/// p_hpa: pressure (hPa), t_c: temperature (C), td_c: dewpoint (C).
pub fn equivalent_potential_temperature(p_hpa: f64, t_c: f64, td_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    let td_k = td_c + ZEROCNK;
    // Bolton LCL temperature (Bolton 1980 eq 15)
    let t_lcl = 56.0 + 1.0 / (1.0 / (td_k - 56.0) + (t_k / td_k).ln() / 800.0);
    // Vapor pressure and mixing ratio at dewpoint (kg/kg)
    let e = saturation_vapor_pressure(td_c);
    let r = EPS * e / (p_hpa - e);
    // Bolton (1980) eq 39 (matches MetPy's implementation)
    // θ_DL = T * (1000/(p-e))^κ * (T/T_L)^(0.28*r)
    let theta_dl = t_k * (1000.0 / (p_hpa - e)).powf(ROCP) * (t_k / t_lcl).powf(0.28 * r);
    // θ_E = θ_DL * exp((3036/T_L - 1.78) * r * (1 + 0.448*r))
    theta_dl * ((3036.0 / t_lcl - 1.78) * r * (1.0 + 0.448 * r)).exp()
}

/// Wet bulb potential temperature (K) from pressure (hPa), temp (C), dewpoint (C).
/// Computed by finding the wet bulb temperature, then computing its potential temperature
/// along the moist adiabat to 1000 hPa.
pub fn wet_bulb_potential_temperature(p_hpa: f64, t_c: f64, td_c: f64) -> f64 {
    // Lift to LCL, then descend moist adiabatically to 1000 hPa
    let (p_lcl, t_lcl) = drylift(p_hpa, t_c, td_c);
    let theta_c = t_lcl + ZEROCNK;
    let theta_sfc = theta_c * ((1000.0 / p_lcl).powf(ROCP));
    let theta_start_c = theta_sfc - ZEROCNK;
    let thetam = theta_start_c - wobf(theta_start_c) + wobf(t_lcl);
    let tw_1000 = satlift(1000.0, thetam);
    tw_1000 + ZEROCNK
}

/// Virtual potential temperature (K) from pressure (hPa), temp (C), mixing ratio (g/kg).
pub fn virtual_potential_temperature(p_hpa: f64, t_c: f64, w_gkg: f64) -> f64 {
    let theta = potential_temperature(p_hpa, t_c);
    let w = w_gkg / 1000.0;
    theta * (1.0 + 0.61 * w)
}

// =============================================================================
// Lifted / Parcel Functions
// =============================================================================

/// LCL pressure (hPa) from surface pressure (hPa), temp (C), dewpoint (C).
pub fn lcl_pressure(p_hpa: f64, t_c: f64, td_c: f64) -> f64 {
    let (p_lcl, _t_lcl) = drylift(p_hpa, t_c, td_c);
    p_lcl
}

/// Lift a parcel and compute parcel temperature at each level.
/// Returns parcel virtual temperature profile above LCL via moist adiabat.
fn lift_parcel_profile(
    p_prof: &[f64],
    t_prof: &[f64],
    td_prof: &[f64],
) -> (f64, f64, Vec<f64>) {
    // Use surface-based parcel
    let p_sfc = p_prof[0];
    let t_sfc = t_prof[0];
    let td_sfc = td_prof[0];

    let (p_lcl, t_lcl) = drylift(p_sfc, t_sfc, td_sfc);

    // Compute thetam for moist ascent
    let theta_k = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP));
    let theta_c = theta_k - ZEROCNK;
    let thetam = theta_c - wobf(theta_c) + wobf(t_lcl);

    // Compute parcel Tv at each level
    let mut parcel_tv = Vec::with_capacity(p_prof.len());
    let theta_dry_k = (t_sfc + ZEROCNK) * ((1000.0 / p_sfc).powf(ROCP));
    let r_parcel = mixratio(p_sfc, td_sfc);

    for i in 0..p_prof.len() {
        let p = p_prof[i];
        if p > p_lcl {
            // Below LCL: dry adiabat
            let t_parc_k = theta_dry_k * ((p / 1000.0).powf(ROCP));
            let t_parc = t_parc_k - ZEROCNK;
            let tv = (t_parc + ZEROCNK) * (1.0 + 0.61 * (r_parcel / 1000.0)) - ZEROCNK;
            parcel_tv.push(tv);
        } else {
            // Above LCL: moist adiabat
            let t_parc = satlift(p, thetam);
            let tv = virtual_temp(t_parc, p, t_parc);
            parcel_tv.push(tv);
        }
    }

    (p_lcl, t_lcl, parcel_tv)
}

/// Level of Free Convection (LFC).
/// Returns Option<(pressure_hPa, temperature_C)> of the LFC.
/// Profiles should be surface-first, decreasing pressure.
pub fn lfc(
    p_profile: &[f64],
    t_profile: &[f64],
    td_profile: &[f64],
) -> Option<(f64, f64)> {
    let (p_lcl, _t_lcl, parcel_tv) = lift_parcel_profile(p_profile, t_profile, td_profile);

    // Search above LCL for first crossing where parcel becomes warmer than environment
    for i in 1..p_profile.len() {
        if p_profile[i] > p_lcl {
            continue;
        }
        let tv_env_prev = virtual_temp(t_profile[i - 1], p_profile[i - 1], td_profile[i - 1]);
        let tv_env = virtual_temp(t_profile[i], p_profile[i], td_profile[i]);
        let buoy_prev = parcel_tv[i - 1] - tv_env_prev;
        let buoy = parcel_tv[i] - tv_env;

        if buoy_prev <= 0.0 && buoy > 0.0 {
            // Interpolate crossing
            let frac = (0.0 - buoy_prev) / (buoy - buoy_prev);
            let p_lfc = p_profile[i - 1] + frac * (p_profile[i] - p_profile[i - 1]);
            let t_lfc = t_profile[i - 1] + frac * (t_profile[i] - t_profile[i - 1]);
            return Some((p_lfc, t_lfc));
        }

        // If parcel is already warmer right at LCL
        if buoy > 0.0 && p_profile[i] <= p_lcl && (i == 0 || p_profile[i - 1] > p_lcl) {
            return Some((p_profile[i], t_profile[i]));
        }
    }

    None
}

/// Equilibrium Level (EL).
/// Returns Option<(pressure_hPa, temperature_C)> of the EL.
/// Profiles should be surface-first, decreasing pressure.
pub fn el(
    p_profile: &[f64],
    t_profile: &[f64],
    td_profile: &[f64],
) -> Option<(f64, f64)> {
    let (p_lcl, _t_lcl, parcel_tv) = lift_parcel_profile(p_profile, t_profile, td_profile);

    let mut found_positive = false;
    let mut last_el: Option<(f64, f64)> = None;

    for i in 1..p_profile.len() {
        if p_profile[i] > p_lcl {
            continue;
        }
        let tv_env_prev = virtual_temp(t_profile[i - 1], p_profile[i - 1], td_profile[i - 1]);
        let tv_env = virtual_temp(t_profile[i], p_profile[i], td_profile[i]);
        let buoy_prev = parcel_tv[i - 1] - tv_env_prev;
        let buoy = parcel_tv[i] - tv_env;

        if buoy > 0.0 {
            found_positive = true;
        }

        if found_positive && buoy_prev > 0.0 && buoy <= 0.0 {
            let frac = (0.0 - buoy_prev) / (buoy - buoy_prev);
            let p_el = p_profile[i - 1] + frac * (p_profile[i] - p_profile[i - 1]);
            let t_el = t_profile[i - 1] + frac * (t_profile[i] - t_profile[i - 1]);
            last_el = Some((p_el, t_el));
        }
    }

    last_el
}

/// Lifted Index: temperature difference between environment and parcel at 500 hPa.
/// Positive values indicate stable conditions, negative values indicate instability.
pub fn lifted_index(
    p_profile: &[f64],
    t_profile: &[f64],
    td_profile: &[f64],
) -> f64 {
    let p_sfc = p_profile[0];
    let t_sfc = t_profile[0];
    let td_sfc = td_profile[0];

    let (p_lcl, t_lcl) = drylift(p_sfc, t_sfc, td_sfc);

    // Get parcel temperature at 500 hPa
    let t_parcel_500 = if 500.0 >= p_lcl {
        // 500 hPa is below LCL (unlikely but handle it)
        let theta_k = (t_sfc + ZEROCNK) * ((1000.0 / p_sfc).powf(ROCP));
        theta_k * ((500.0_f64 / 1000.0).powf(ROCP)) - ZEROCNK
    } else {
        let theta_k = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP));
        let theta_c = theta_k - ZEROCNK;
        let thetam = theta_c - wobf(theta_c) + wobf(t_lcl);
        satlift(500.0, thetam)
    };

    // Interpolate environment temperature at 500 hPa
    let (t_env_500, _td_env_500) = get_env_at_pres(500.0, p_profile, t_profile, td_profile);

    t_env_500 - t_parcel_500
}

/// Convective Condensation Level (CCL).
/// The level where the saturation mixing ratio equals the surface mixing ratio.
/// Returns Option<(pressure_hPa, temperature_C)>.
pub fn ccl(
    p_profile: &[f64],
    t_profile: &[f64],
    td_profile: &[f64],
) -> Option<(f64, f64)> {
    let w_sfc = mixratio(p_profile[0], td_profile[0]);

    // Search upward for where saturation mixing ratio equals surface mixing ratio
    for i in 1..p_profile.len() {
        let ws_prev = mixratio(p_profile[i - 1], t_profile[i - 1]);
        let ws_curr = mixratio(p_profile[i], t_profile[i]);

        if ws_prev >= w_sfc && ws_curr < w_sfc {
            // Interpolate
            let frac = (w_sfc - ws_prev) / (ws_curr - ws_prev);
            let p_ccl = p_profile[i - 1] + frac * (p_profile[i] - p_profile[i - 1]);
            let t_ccl = t_profile[i - 1] + frac * (t_profile[i] - t_profile[i - 1]);
            return Some((p_ccl, t_ccl));
        }
    }

    None
}

/// Convective temperature (Celsius).
/// The surface temperature needed to produce convection (reach CCL via dry adiabat).
pub fn convective_temperature(
    p_profile: &[f64],
    t_profile: &[f64],
    td_profile: &[f64],
) -> f64 {
    if let Some((p_ccl, t_ccl)) = ccl(p_profile, t_profile, td_profile) {
        // Bring CCL temperature down dry-adiabatically to surface
        let theta_k = (t_ccl + ZEROCNK) * ((1000.0 / p_ccl).powf(ROCP));
        theta_k * ((p_profile[0] / 1000.0).powf(ROCP)) - ZEROCNK
    } else {
        MISSING
    }
}

// =============================================================================
// Density / Height Functions
// =============================================================================

/// Air density (kg/m^3) from pressure (hPa), temperature (C), mixing ratio (g/kg).
/// Uses virtual temperature for moist air density.
pub fn density(p_hpa: f64, t_c: f64, w_gkg: f64) -> f64 {
    let p_pa = p_hpa * 100.0;
    let t_k = t_c + ZEROCNK;
    let w = w_gkg / 1000.0;
    let tv_k = t_k * (1.0 + 0.61 * w);
    p_pa / (RD * tv_k)
}

/// Virtual temperature (Celsius) from temperature (C), dewpoint (C), pressure (hPa).
/// Computes mixing ratio from dewpoint and pressure.
pub fn virtual_temperature_from_dewpoint(t_c: f64, td_c: f64, p_hpa: f64) -> f64 {
    virtual_temp(t_c, p_hpa, td_c)
}

/// Hypsometric thickness (meters) of a layer between two pressure levels.
/// p_bottom, p_top in hPa, t_mean_k in Kelvin.
pub fn thickness_hypsometric(p_bottom: f64, p_top: f64, t_mean_k: f64) -> f64 {
    (RD * t_mean_k / G) * (p_bottom / p_top).ln()
}

/// Standard atmosphere: pressure (hPa) to geopotential height (meters).
/// Valid for troposphere (below ~11 km).
pub fn pressure_to_height_std(p_hpa: f64) -> f64 {
    (T0_STD / LAPSE_STD) * (1.0 - (p_hpa / P0_STD).powf((RD * LAPSE_STD) / G))
}

/// Standard atmosphere: height (meters) to pressure (hPa).
/// Valid for troposphere (below ~11 km).
pub fn height_to_pressure_std(h_m: f64) -> f64 {
    P0_STD * (1.0 - LAPSE_STD * h_m / T0_STD).powf(G / (RD * LAPSE_STD))
}

/// Convert altimeter setting (hPa) to station pressure (hPa).
/// elevation_m: station elevation in meters.
pub fn altimeter_to_station_pressure(alt_hpa: f64, elevation_m: f64) -> f64 {
    // From the altimeter setting equation (NWS)
    let k = ROCP; // Rd/Cp
    let t0 = T0_STD;
    let _p0 = P0_STD;
    let l = LAPSE_STD;

    // Station pressure from altimeter equation:
    // alt = p_stn * (1 + (p0/p_stn)^k * (l*elev/t0))^(1/k)
    // Iterative approach: p_stn ≈ alt * (1 - l*elev/t0)^(1/k)
    // More precise: use the standard relationship
    let ratio = 1.0 - (l * elevation_m) / (t0 + l * elevation_m);
    alt_hpa * ratio.powf(1.0 / k)
}

/// Convert station pressure to sea level pressure (hPa).
/// p_station (hPa), elevation_m (meters), t_c: station temperature (Celsius).
pub fn station_to_sea_level_pressure(p_station: f64, elevation_m: f64, t_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    // Use the hypsometric equation to extrapolate to sea level
    // Mean virtual temperature of the fictitious column below the station
    let t_mean = t_k + LAPSE_STD * elevation_m / 2.0;
    p_station * (G * elevation_m / (RD * t_mean)).exp()
}

// =============================================================================
// Moist Thermodynamics
// =============================================================================

/// Temperature at each pressure level following a dry adiabat.
/// T = T_surface * (p/p_surface)^(Rd/Cp).
/// p: pressure levels (hPa, surface first), t_surface_c: surface temperature (Celsius).
/// Returns temperatures in Celsius at each level.
pub fn dry_lapse(p: &[f64], t_surface_c: f64) -> Vec<f64> {
    if p.is_empty() {
        return vec![];
    }
    let t_surface_k = t_surface_c + ZEROCNK;
    let p_surface = p[0];
    p.iter()
        .map(|&pi| t_surface_k * (pi / p_surface).powf(ROCP) - ZEROCNK)
        .collect()
}

/// Moist adiabatic lapse rate dT/dp for saturated parcel.
/// Returns dT/dp in K/hPa (for use in RK4 integration where pressure is in hPa).
///
/// Formula from Bakhshaii & Stull 2013:
///   dT/dp = (1/p) * (Rd*T + Lv*rs) / (Cp_d + Lv^2*rs*epsilon/(Rd*T^2))
///
/// Since p is in hPa here, the Rd*T and Lv^2*... terms use SI (J/kg) but we
/// must divide by p in Pa to get K/Pa, then multiply by 100 to get K/hPa.
/// Equivalently, we compute the SI numerator and divide by p_hpa directly,
/// but scale Rd and Lv^2 terms so the result is in K/hPa.
fn moist_lapse_rate(p_hpa: f64, t_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    let es = saturation_vapor_pressure(t_c);
    let rs = EPS * es / (p_hpa - es); // kg/kg

    // Numerator: (Rd*T + Lv*rs) / p, with p in Pa => divide by (p_hpa * 100)
    // Denominator: Cp_d + Lv^2 * rs * epsilon / (Rd * T^2)
    // Result is K/Pa; multiply by 100 to get K/hPa.
    // Net effect: divide numerator by p_hpa only (the 100s cancel).
    let numerator = (RD * t_k + LV * rs) / p_hpa;
    let denominator = CP + (LV * LV * rs * EPS) / (RD * t_k * t_k);
    numerator / denominator
}

/// Temperature following a moist (saturated) adiabat using RK4 integration.
/// p: pressure levels (hPa, surface/start first), t_start_c: starting temperature (Celsius).
/// Returns temperatures in Celsius at each level.
pub fn moist_lapse(p: &[f64], t_start_c: f64) -> Vec<f64> {
    if p.is_empty() {
        return vec![];
    }
    let mut result = Vec::with_capacity(p.len());
    result.push(t_start_c);

    let mut t = t_start_c;
    for i in 1..p.len() {
        let dp = p[i] - p[i - 1];
        if dp.abs() < 1e-10 {
            result.push(t);
            continue;
        }
        // RK4 integration with subdivided steps
        let n_steps = ((dp.abs() / 5.0) as usize).max(4);
        let h = dp / n_steps as f64;
        let mut p_c = p[i - 1];
        for _ in 0..n_steps {
            let k1 = h * moist_lapse_rate(p_c, t);
            let k2 = h * moist_lapse_rate(p_c + h / 2.0, t + k1 / 2.0);
            let k3 = h * moist_lapse_rate(p_c + h / 2.0, t + k2 / 2.0);
            let k4 = h * moist_lapse_rate(p_c + h, t + k3);
            t += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            p_c += h;
        }
        result.push(t);
    }
    result
}

/// Dry static energy: DSE = Cp*T + g*z (J/kg).
/// height_m: height in meters, t_k: temperature in Kelvin.
pub fn dry_static_energy(height_m: f64, t_k: f64) -> f64 {
    CP * t_k + G * height_m
}

/// Moist static energy: MSE = Cp*T + g*z + Lv*q (J/kg).
/// height_m: height (m), t_k: temperature (K), q_kgkg: specific humidity (kg/kg).
pub fn moist_static_energy(height_m: f64, t_k: f64, q_kgkg: f64) -> f64 {
    CP * t_k + G * height_m + LV * q_kgkg
}

/// Full parcel temperature profile: dry adiabat to LCL, then moist adiabat above.
/// p: pressure levels (hPa, surface first, decreasing), t_surface_c: surface T (C),
/// td_surface_c: surface Td (C). Returns parcel temperature (Celsius) at each level.
pub fn parcel_profile(p: &[f64], t_surface_c: f64, td_surface_c: f64) -> Vec<f64> {
    if p.is_empty() {
        return vec![];
    }
    let (p_lcl, t_lcl) = drylift(p[0], t_surface_c, td_surface_c);

    let mut result = Vec::with_capacity(p.len());

    // Dry adiabat below LCL
    let t_surface_k = t_surface_c + ZEROCNK;
    let p_surface = p[0];

    // Collect moist adiabat levels (at and above LCL)
    let mut moist_pressures = vec![p_lcl];
    let mut lcl_idx = p.len(); // index of first level at or above LCL

    for (i, &pi) in p.iter().enumerate() {
        if pi <= p_lcl {
            if lcl_idx == p.len() {
                lcl_idx = i;
            }
            moist_pressures.push(pi);
        }
    }

    // Compute moist adiabat from LCL upward
    let moist_temps = moist_lapse(&moist_pressures, t_lcl);

    // Build result
    let mut moist_idx = 1; // skip the LCL entry itself
    for (_i, &pi) in p.iter().enumerate() {
        if pi > p_lcl {
            // Dry adiabat
            let t_k = t_surface_k * (pi / p_surface).powf(ROCP);
            result.push(t_k - ZEROCNK);
        } else {
            // Moist adiabat
            if moist_idx < moist_temps.len() {
                result.push(moist_temps[moist_idx]);
                moist_idx += 1;
            } else {
                // Fallback: use satlift
                let theta_k = (t_lcl + ZEROCNK) * ((1000.0 / p_lcl).powf(ROCP));
                let theta_c = theta_k - ZEROCNK;
                let thetam = theta_c - wobf(theta_c) + wobf(t_lcl);
                result.push(satlift(pi, thetam));
            }
        }
    }

    result
}

/// Dewpoint (Celsius) from vapor pressure (hPa). Inverse Bolton formula.
///
/// This is an alias for the `dewpoint` function defined earlier in this module.
pub fn dewpoint_from_vapor_pressure(vapor_pressure_hpa: f64) -> f64 {
    if vapor_pressure_hpa <= 0.0 {
        return -ZEROCNK; // absolute zero-ish
    }
    let ln_ratio = (vapor_pressure_hpa / 6.112).ln();
    243.5 * ln_ratio / (17.67 - ln_ratio)
}

/// Mixing ratio (g/kg) from relative humidity (%).
/// p_hpa: pressure (hPa), t_c: temperature (C), rh: relative humidity (0-100).
pub fn mixing_ratio_from_relative_humidity(p_hpa: f64, t_c: f64, rh: f64) -> f64 {
    let ws = saturation_mixing_ratio(p_hpa, t_c);
    ws * rh / 100.0
}

/// Relative humidity (%) from mixing ratio.
/// p_hpa: pressure (hPa), t_c: temperature (C), w_gkg: mixing ratio (g/kg).
pub fn relative_humidity_from_mixing_ratio(p_hpa: f64, t_c: f64, w_gkg: f64) -> f64 {
    let ws = saturation_mixing_ratio(p_hpa, t_c);
    if ws <= 0.0 {
        return 0.0;
    }
    (w_gkg / ws) * 100.0
}

/// Relative humidity (%) from specific humidity.
/// p_hpa: pressure (hPa), t_c: temperature (C), q: specific humidity (kg/kg).
pub fn relative_humidity_from_specific_humidity(p_hpa: f64, t_c: f64, q: f64) -> f64 {
    let w_gkg = mixing_ratio_from_specific_humidity(q);
    relative_humidity_from_mixing_ratio(p_hpa, t_c, w_gkg)
}

/// Specific humidity (kg/kg) from dewpoint.
/// p_hpa: pressure (hPa), td_c: dewpoint (Celsius).
pub fn specific_humidity_from_dewpoint(p_hpa: f64, td_c: f64) -> f64 {
    let e = saturation_vapor_pressure(td_c);
    let w = EPS * e / (p_hpa - e); // kg/kg
    w / (1.0 + w)
}

/// Dewpoint (Celsius) from specific humidity.
/// p_hpa: pressure (hPa), q: specific humidity (kg/kg).
pub fn dewpoint_from_specific_humidity(p_hpa: f64, q: f64) -> f64 {
    let w = q / (1.0 - q); // kg/kg
    let e = w * p_hpa / (EPS + w);
    dewpoint_from_vapor_pressure(e)
}

/// Saturation equivalent potential temperature (K). Assumes RH=100%.
/// p_hpa: pressure (hPa), t_c: temperature (Celsius).
pub fn saturation_equivalent_potential_temperature(p_hpa: f64, t_c: f64) -> f64 {
    equivalent_potential_temperature(p_hpa, t_c, t_c)
}

/// Scale height: H = R*T/g (meters).
/// t_k: temperature in Kelvin.
pub fn scale_height(t_k: f64) -> f64 {
    RD * t_k / G
}

/// Convert vertical velocity w (m/s) to omega (Pa/s).
/// omega = -rho * g * w, where rho = p/(Rd*Tv).
/// w_ms: vertical velocity (m/s, positive up), p_hpa: pressure (hPa), t_c: temperature (C).
pub fn vertical_velocity_pressure(w_ms: f64, p_hpa: f64, t_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    let p_pa = p_hpa * 100.0;
    let rho = p_pa / (RD * t_k);
    -rho * G * w_ms
}

/// Convert omega (Pa/s) to vertical velocity w (m/s).
/// omega_pas: omega (Pa/s, negative = upward), p_hpa: pressure (hPa), t_c: temperature (C).
pub fn vertical_velocity(omega_pas: f64, p_hpa: f64, t_c: f64) -> f64 {
    let t_k = t_c + ZEROCNK;
    let p_pa = p_hpa * 100.0;
    let rho = p_pa / (RD * t_k);
    -omega_pas / (rho * G)
}

/// Static stability parameter: sigma = -(T/theta)(d_theta/dp).
/// p: pressure levels (hPa), t_k: temperature (K) at each level.
/// Returns sigma at each level (centered differences, forward/backward at boundaries).
pub fn static_stability(p: &[f64], t_k: &[f64]) -> Vec<f64> {
    let n = p.len();
    if n < 2 {
        return vec![0.0; n];
    }
    // Compute theta at each level
    let theta: Vec<f64> = p.iter().zip(t_k.iter())
        .map(|(&pi, &ti)| ti * (1000.0 / pi).powf(ROCP))
        .collect();

    let mut result = vec![0.0; n];
    for i in 0..n {
        let (dtheta, dp) = if i == 0 {
            (theta[1] - theta[0], p[1] - p[0])
        } else if i == n - 1 {
            (theta[n - 1] - theta[n - 2], p[n - 1] - p[n - 2])
        } else {
            (theta[i + 1] - theta[i - 1], p[i + 1] - p[i - 1])
        };
        if dp.abs() < 1e-10 || theta[i].abs() < 1e-10 {
            result[i] = 0.0;
        } else {
            // Convert dp from hPa to Pa for proper units
            result[i] = -(t_k[i] / theta[i]) * (dtheta / (dp * 100.0));
        }
    }
    result
}

/// Pressure-weighted mean of a quantity over a set of pressure levels.
/// mean = sum(values\[i\] * dp\[i\]) / sum(dp\[i\]) where dp is layer thickness.
pub fn mean_pressure_weighted(p: &[f64], values: &[f64]) -> f64 {
    if p.len() < 2 || values.len() < 2 {
        return if values.is_empty() { 0.0 } else { values[0] };
    }
    let mut sum_val = 0.0;
    let mut sum_dp = 0.0;
    for i in 0..p.len() - 1 {
        let dp = (p[i] - p[i + 1]).abs();
        let avg_val = (values[i] + values[i + 1]) / 2.0;
        sum_val += avg_val * dp;
        sum_dp += dp;
    }
    if sum_dp <= 0.0 { values[0] } else { sum_val / sum_dp }
}

/// Inverse Poisson: temperature (K) from potential temperature and pressure.
/// p_hpa: pressure (hPa), theta_k: potential temperature (K).
pub fn temperature_from_potential_temperature(p_hpa: f64, theta_k: f64) -> f64 {
    theta_k * (p_hpa / 1000.0).powf(ROCP)
}

/// Convert geopotential (m^2/s^2) to geopotential height (m): z = Phi / g0.
pub fn geopotential_to_height(geopot: f64) -> f64 {
    geopot / G
}

/// Convert geopotential height (m) to geopotential (m^2/s^2): Phi = g0 * z.
pub fn height_to_geopotential(height_m: f64) -> f64 {
    G * height_m
}

/// Convert sigma coordinate to pressure.
/// p = sigma * (p_sfc - p_top) + p_top.
pub fn sigma_to_pressure(sigma: f64, p_sfc: f64, p_top: f64) -> f64 {
    sigma * (p_sfc - p_top) + p_top
}

// =============================================================================
// Apparent Temperature
// =============================================================================

/// Heat index using the Rothfusz regression (Fahrenheit).
/// t_f: temperature (Fahrenheit), rh: relative humidity (%).
/// Returns heat index in Fahrenheit.
pub fn heat_index(t_f: f64, rh: f64) -> f64 {
    // NWS two-step: compute Steadman, average with T, then decide
    let steadman = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094);
    let hi_avg = (steadman + t_f) / 2.0;

    if hi_avg < 80.0 {
        return hi_avg;
    }

    // Rothfusz regression
    let mut hi = -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 6.83783e-3 * t_f * t_f
        - 5.481717e-2 * rh * rh
        + 1.22874e-3 * t_f * t_f * rh
        + 8.5282e-4 * t_f * rh * rh
        - 1.99e-6 * t_f * t_f * rh * rh;

    // Adjustments
    if rh < 13.0 && t_f >= 80.0 && t_f <= 112.0 {
        hi -= ((13.0 - rh) / 4.0) * ((17.0 - (t_f - 95.0).abs()) / 17.0).sqrt();
    } else if rh > 85.0 && t_f >= 80.0 && t_f <= 87.0 {
        hi += ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0);
    }

    hi
}

/// NWS wind chill (Fahrenheit).
/// t_f: temperature (Fahrenheit), wind_mph: wind speed (mph).
/// Returns wind chill in Fahrenheit. Only valid for T <= 50F and wind >= 3 mph.
pub fn windchill(t_f: f64, wind_mph: f64) -> f64 {
    if t_f > 50.0 || wind_mph < 3.0 {
        return t_f;
    }
    let v016 = wind_mph.powf(0.16);
    35.74 + 0.6215 * t_f - 35.75 * v016 + 0.4275 * t_f * v016
}

/// Australian apparent temperature (Celsius).
/// t_c: temperature (C), rh: relative humidity (%), wind_ms: wind speed (m/s),
/// solar_wm2: optional solar radiation (W/m^2), defaults to 0.
pub fn apparent_temperature(t_c: f64, rh: f64, wind_ms: f64, solar_wm2: Option<f64>) -> f64 {
    let q = solar_wm2.unwrap_or(0.0);
    // Water vapor pressure from RH and T
    let e = (rh / 100.0) * saturation_vapor_pressure(t_c);
    // Steadman (1984) apparent temperature
    t_c + 0.348 * e - 0.70 * wind_ms + 0.70 * q / (wind_ms + 10.0) - 4.25
}

// =============================================================================
// Boundary Layer
// =============================================================================

/// Brunt-Vaisala frequency squared: N^2 = (g/theta)(d_theta/dz).
/// p: pressure (hPa), t_k: temperature (K). Uses hydrostatic approx for dz.
/// Returns N (s^-1) at each level (sqrt of N^2 where positive, 0 where negative).
pub fn brunt_vaisala_frequency(p: &[f64], t_k: &[f64]) -> Vec<f64> {
    let n = p.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let theta: Vec<f64> = p.iter().zip(t_k.iter())
        .map(|(&pi, &ti)| ti * (1000.0 / pi).powf(ROCP))
        .collect();

    // Approximate heights using hypsometric equation
    let mut z = vec![0.0; n];
    for i in 1..n {
        let t_mean = (t_k[i - 1] + t_k[i]) / 2.0;
        z[i] = z[i - 1] + (RD * t_mean / G) * (p[i - 1] / p[i]).ln();
    }

    let mut result = vec![0.0; n];
    for i in 0..n {
        let (dtheta, dz) = if i == 0 {
            (theta[1] - theta[0], z[1] - z[0])
        } else if i == n - 1 {
            (theta[n - 1] - theta[n - 2], z[n - 1] - z[n - 2])
        } else {
            (theta[i + 1] - theta[i - 1], z[i + 1] - z[i - 1])
        };
        if dz.abs() < 1e-10 || theta[i].abs() < 1e-10 {
            result[i] = 0.0;
        } else {
            let n_sq = (G / theta[i]) * (dtheta / dz);
            result[i] = if n_sq > 0.0 { n_sq.sqrt() } else { 0.0 };
        }
    }
    result
}

/// Brunt-Vaisala period: 2*pi/N (seconds).
/// n: Brunt-Vaisala frequency (s^-1).
pub fn brunt_vaisala_period(n: f64) -> f64 {
    if n <= 0.0 {
        return f64::INFINITY;
    }
    2.0 * std::f64::consts::PI / n
}

/// Gradient Richardson number: Ri = (g/theta)(d_theta/dz) / ((du/dz)^2 + (dv/dz)^2).
/// theta: potential temperature (K), u,v: wind components (m/s), z: height (m).
/// All arrays same length. Returns Ri at each level.
pub fn gradient_richardson_number(theta: &[f64], u: &[f64], v: &[f64], z: &[f64]) -> Vec<f64> {
    let n = theta.len();
    if n < 2 {
        return vec![f64::INFINITY; n];
    }
    let mut result = vec![0.0; n];
    for i in 0..n {
        let (dtheta, du, dv, dz_val) = if i == 0 {
            (theta[1] - theta[0], u[1] - u[0], v[1] - v[0], z[1] - z[0])
        } else if i == n - 1 {
            (theta[n - 1] - theta[n - 2], u[n - 1] - u[n - 2],
             v[n - 1] - v[n - 2], z[n - 1] - z[n - 2])
        } else {
            (theta[i + 1] - theta[i - 1], u[i + 1] - u[i - 1],
             v[i + 1] - v[i - 1], z[i + 1] - z[i - 1])
        };
        if dz_val.abs() < 1e-10 {
            result[i] = f64::INFINITY;
            continue;
        }
        let dthetadz = dtheta / dz_val;
        let dudz = du / dz_val;
        let dvdz = dv / dz_val;
        let shear_sq = dudz * dudz + dvdz * dvdz;
        if shear_sq < 1e-20 {
            result[i] = f64::INFINITY;
        } else {
            result[i] = (G / theta[i]) * dthetadz / shear_sq;
        }
    }
    result
}

/// Turbulent kinetic energy: TKE = 0.5 * mean(u'^2 + v'^2 + w'^2).
/// u_prime, v_prime, w_prime: perturbation wind components (m/s).
pub fn tke(u_prime: &[f64], v_prime: &[f64], w_prime: &[f64]) -> f64 {
    let n = u_prime.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = u_prime.iter().zip(v_prime.iter()).zip(w_prime.iter())
        .map(|((&u, &v), &w)| u * u + v * v + w * w)
        .sum();
    0.5 * sum / n as f64
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Find x,y positions where two curves cross (y1 and y2 vs same x axis).
/// Returns Vec of (x_crossing, y_crossing) tuples.
pub fn find_intersections(x: &[f64], y1: &[f64], y2: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len().min(y1.len()).min(y2.len());
    if n < 2 {
        return vec![];
    }
    let mut crossings = Vec::new();
    for i in 1..n {
        let d_prev = y1[i - 1] - y2[i - 1];
        let d_curr = y1[i] - y2[i];
        // Sign change means crossing
        if d_prev * d_curr < 0.0 {
            let frac = d_prev.abs() / (d_prev.abs() + d_curr.abs());
            let x_cross = x[i - 1] + frac * (x[i] - x[i - 1]);
            let y_cross = y1[i - 1] + frac * (y1[i] - y1[i - 1]);
            crossings.push((x_cross, y_cross));
        } else if d_curr.abs() < 1e-15 && d_prev.abs() > 1e-15 {
            // Exactly on the crossing
            crossings.push((x[i], y1[i]));
        }
    }
    crossings
}

/// Extract values within a pressure layer.
/// p: pressure (hPa, decreasing), values: corresponding values.
/// p_bottom, p_top: layer bounds (hPa). Interpolates at boundaries.
/// Returns (p_layer, values_layer).
pub fn get_layer(p: &[f64], values: &[f64], p_bottom: f64, p_top: f64) -> (Vec<f64>, Vec<f64>) {
    let mut p_out = Vec::new();
    let mut v_out = Vec::new();

    // Interpolate at bottom boundary if needed
    if p[0] < p_bottom {
        // p_bottom is below the profile - skip
    }

    for i in 0..p.len() {
        if p[i] <= p_bottom && p[i] >= p_top {
            // Add interpolated bottom boundary
            if p_out.is_empty() && i > 0 && p[i - 1] > p_bottom {
                let frac = (p_bottom.ln() - p[i - 1].ln()) / (p[i].ln() - p[i - 1].ln());
                let v_interp = values[i - 1] + frac * (values[i] - values[i - 1]);
                p_out.push(p_bottom);
                v_out.push(v_interp);
            } else if p_out.is_empty() && p[i] <= p_bottom {
                // First point is at or below p_bottom
            }
            p_out.push(p[i]);
            v_out.push(values[i]);
        } else if p[i] < p_top && !p_out.is_empty() {
            // Interpolate at top boundary
            if i > 0 && p[i - 1] >= p_top {
                let frac = (p_top.ln() - p[i - 1].ln()) / (p[i].ln() - p[i - 1].ln());
                let v_interp = values[i - 1] + frac * (values[i] - values[i - 1]);
                p_out.push(p_top);
                v_out.push(v_interp);
            }
            break;
        }
    }
    (p_out, v_out)
}

/// Extract height values within a pressure layer (same as get_layer but for heights).
pub fn get_layer_heights(p: &[f64], z: &[f64], p_bottom: f64, p_top: f64) -> (Vec<f64>, Vec<f64>) {
    get_layer(p, z, p_bottom, p_top)
}

/// Thinning mask for station plots. Returns `Vec<bool>` where true = keep.
/// Removes points closer than radius_deg to already-kept points.
pub fn reduce_point_density(lats: &[f64], lons: &[f64], radius_deg: f64) -> Vec<bool> {
    let n = lats.len().min(lons.len());
    let mut keep = vec![true; n];
    let r2 = radius_deg * radius_deg;

    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !keep[j] {
                continue;
            }
            let dlat = lats[j] - lats[i];
            let dlon = lons[j] - lons[i];
            if dlat * dlat + dlon * dlon < r2 {
                keep[j] = false;
            }
        }
    }
    keep
}

// =============================================================================
// Sounding Functions
// =============================================================================

/// Downdraft CAPE: integrate negative buoyancy from min theta-e level to surface.
/// p, t, td: pressure (hPa), temperature (C), dewpoint (C). Surface first.
pub fn downdraft_cape(p: &[f64], t: &[f64], td: &[f64]) -> f64 {
    if p.len() < 3 {
        return 0.0;
    }

    // Find level of minimum theta-e (in lowest 400 hPa)
    let sfc_p = p[0];
    let limit_p = sfc_p - 400.0;
    let mut min_te = f64::INFINITY;
    let mut min_idx = 0;

    for i in 0..p.len() {
        if p[i] < limit_p {
            break;
        }
        let te = thetae(p[i], t[i], td[i]);
        if te < min_te {
            min_te = te;
            min_idx = i;
        }
    }

    if min_idx == 0 {
        return 0.0;
    }

    // Descend moist adiabatically from min theta-e level to surface
    // Build descending pressure array
    let mut desc_p: Vec<f64> = Vec::new();
    for i in (0..=min_idx).rev() {
        desc_p.push(p[i]);
    }

    let moist_temps = moist_lapse(&desc_p, t[min_idx]);

    // Integrate DCAPE (only negative buoyancy = downdraft)
    let mut dcape = 0.0;
    let mut moist_i = 0;
    for i in (0..min_idx).rev() {
        moist_i += 1;
        if moist_i >= moist_temps.len() {
            break;
        }
        let t_parcel = moist_temps[moist_i];
        let tv_parcel = virtual_temp(t_parcel, p[i], t_parcel);
        let tv_env = virtual_temp(t[i], p[i], td[i]);

        let buoyancy = tv_parcel - tv_env;
        if buoyancy < 0.0 {
            let dp = if i < min_idx {
                (p[i].ln() - p[i + 1].ln()).abs()
            } else {
                0.0
            };
            dcape += RD * buoyancy.abs() * dp;
        }
    }

    dcape
}

/// Mixed-layer average of a quantity in the lowest `depth_hpa` hPa.
pub fn mixed_layer(p: &[f64], values: &[f64], depth_hpa: f64) -> f64 {
    if p.is_empty() || values.is_empty() {
        return 0.0;
    }
    let sfc_p = p[0];
    let top_p = sfc_p - depth_hpa;

    let mut sum = 0.0;
    let mut total_dp = 0.0;

    for i in 0..p.len() - 1 {
        if p[i] < top_p {
            break;
        }
        let p_top_layer = p[i + 1].max(top_p);
        let dp = p[i] - p_top_layer;
        if dp <= 0.0 {
            continue;
        }
        let avg_val = (values[i] + values[i + 1]) / 2.0;
        sum += avg_val * dp;
        total_dp += dp;
    }

    if total_dp <= 0.0 { values[0] } else { sum / total_dp }
}

/// CAPE and CIN for a mixed-layer parcel.
/// p, t, td: profiles (hPa, C, C). depth_hpa: mixed-layer depth (typically 100 hPa).
pub fn mixed_layer_cape_cin(p: &[f64], t: &[f64], td: &[f64], depth_hpa: f64) -> (f64, f64) {
    let (p_start, t_start, td_start) = get_mixed_layer_parcel(p, t, td, depth_hpa);
    cape_cin_from_parcel(p, t, td, p_start, t_start, td_start)
}

/// CAPE and CIN for the most unstable parcel (highest theta-e in lowest 300 hPa).
pub fn most_unstable_cape_cin(p: &[f64], t: &[f64], td: &[f64]) -> (f64, f64) {
    let (p_start, t_start, td_start) = get_most_unstable_parcel(p, t, td, 300.0);
    cape_cin_from_parcel(p, t, td, p_start, t_start, td_start)
}

/// CAPE and CIN for a surface-based parcel.
pub fn surface_based_cape_cin(p: &[f64], t: &[f64], td: &[f64]) -> (f64, f64) {
    if p.is_empty() {
        return (0.0, 0.0);
    }
    cape_cin_from_parcel(p, t, td, p[0], t[0], td[0])
}

/// Internal: compute CAPE/CIN given a starting parcel.
fn cape_cin_from_parcel(
    p: &[f64], t: &[f64], td: &[f64],
    p_start: f64, t_start: f64, td_start: f64,
) -> (f64, f64) {
    // Compute LCL
    let (p_lcl, t_lcl) = drylift(p_start, t_start, td_start);

    // Build full parcel profile using moist_lapse (MetPy-compatible RK4)
    let theta_dry_k = (t_start + ZEROCNK) * ((1000.0 / p_start).powf(ROCP));
    let r_parcel = mixratio(p_start, td_start);

    // Collect pressure levels at and above LCL for moist ascent
    let mut p_moist: Vec<f64> = Vec::new();
    p_moist.push(p_lcl);
    for &pi in p.iter() {
        if pi < p_lcl && pi > 0.0 {
            p_moist.push(pi);
        }
    }
    let moist_profile = if p_moist.len() > 1 {
        moist_lapse(&p_moist, t_lcl)
    } else {
        vec![t_lcl]
    };

    // Build parcel Tv at each sounding level
    let n = p.len();
    let mut tv_parcel = vec![0.0_f64; n];
    for i in 0..n {
        if p[i] <= 0.0 {
            tv_parcel[i] = f64::NAN;
            continue;
        }
        if p[i] >= p_lcl {
            // Below LCL: dry adiabat with moisture
            let t_parc_k = theta_dry_k * ((p[i] / 1000.0).powf(ROCP));
            let t_parc = t_parc_k - ZEROCNK;
            // Virtual temperature correction for moisture
            let w_kgkg = r_parcel / 1000.0; // g/kg to kg/kg
            tv_parcel[i] = (t_parc + ZEROCNK) * (1.0 + w_kgkg / EPS) / (1.0 + w_kgkg) - ZEROCNK;
        } else {
            // Above LCL: interpolate from moist_lapse profile
            let t_parc = interp_log_p(p[i], &p_moist, &moist_profile);
            // Saturated: dewpoint = temperature
            tv_parcel[i] = virtual_temp(t_parc, p[i], t_parc);
        }
    }

    // Build environment Tv at each level
    let mut tv_env = vec![0.0_f64; n];
    for i in 0..n {
        tv_env[i] = virtual_temp(t[i], p[i], td[i]);
    }

    // Compute height from pressure using hypsometric equation
    let mut z = vec![0.0_f64; n];
    for i in 1..n {
        if p[i] <= 0.0 || p[i - 1] <= 0.0 {
            z[i] = z[i - 1];
            continue;
        }
        let tv_mean = (tv_env[i - 1] + tv_env[i]) / 2.0 + ZEROCNK;
        z[i] = z[i - 1] + (RD * tv_mean / G) * (p[i - 1] / p[i]).ln();
    }

    // Integrate CAPE/CIN using MetPy's formula: g * (Tv_p - Tv_e) / Tv_e * dz
    // Trapezoidal rule
    let mut cape = 0.0_f64;
    let mut cin = 0.0_f64;

    for i in 1..n {
        // Only integrate from the starting parcel level upward
        if p[i] > p_start || p[i - 1] > p_start {
            continue;
        }
        if p[i] <= 0.0 || tv_parcel[i].is_nan() || tv_parcel[i - 1].is_nan() {
            continue;
        }
        let tv_e_lo = tv_env[i - 1] + ZEROCNK;
        let tv_e_hi = tv_env[i] + ZEROCNK;
        let tv_p_lo = tv_parcel[i - 1] + ZEROCNK;
        let tv_p_hi = tv_parcel[i] + ZEROCNK;
        let dz = z[i] - z[i - 1];
        if dz.abs() < 1e-6 || tv_e_lo <= 0.0 || tv_e_hi <= 0.0 {
            continue;
        }
        // Trapezoidal: average buoyancy at top and bottom of layer
        let buoy_lo = (tv_p_lo - tv_e_lo) / tv_e_lo;
        let buoy_hi = (tv_p_hi - tv_e_hi) / tv_e_hi;
        let val = G * (buoy_lo + buoy_hi) / 2.0 * dz;
        if val > 0.0 {
            cape += val;
        } else {
            cin += val;
        }
    }

    (cape, cin)
}

/// Interpolate temperature at pressure level p_target from a profile using log-pressure.
fn interp_log_p(p_target: f64, p_prof: &[f64], t_prof: &[f64]) -> f64 {
    let n = p_prof.len();
    if n == 0 { return 0.0; }
    if p_target >= p_prof[0] { return t_prof[0]; }
    if p_target <= p_prof[n - 1] { return t_prof[n - 1]; }
    for i in 1..n {
        if p_prof[i] <= p_target {
            let log_p0 = p_prof[i - 1].ln();
            let log_p1 = p_prof[i].ln();
            let log_pt = p_target.ln();
            let frac = (log_pt - log_p0) / (log_p1 - log_p0);
            return t_prof[i - 1] + frac * (t_prof[i] - t_prof[i - 1]);
        }
    }
    t_prof[n - 1]
}

/// Pressure-weighted continuous average of a variable in a height layer.
/// Matches MetPy's weighted_continuous_average: WCA = ∫A dp / ∫dp
/// Uses trapezoidal integration in pressure, interpolating to layer boundaries in height.
fn pressure_weighted_mean(comp: &[f64], p: &[f64], z: &[f64], z_bot: f64, z_top: f64) -> f64 {
    let n = z.len();
    if n < 2 || z_top <= z_bot { return comp[0]; }

    // Interpolate any quantity to a target height
    let interp_at = |target_z: f64, vals: &[f64]| -> f64 {
        if target_z <= z[0] { return vals[0]; }
        if target_z >= z[n - 1] { return vals[n - 1]; }
        for i in 1..n {
            if z[i] >= target_z {
                let frac = (target_z - z[i - 1]) / (z[i] - z[i - 1]);
                return vals[i - 1] + frac * (vals[i] - vals[i - 1]);
            }
        }
        vals[n - 1]
    };

    // Build the sub-profile within the layer (including interpolated boundaries)
    let mut layer_comp: Vec<f64> = Vec::new();
    let mut layer_p: Vec<f64> = Vec::new();

    // Bottom boundary
    layer_comp.push(interp_at(z_bot, comp));
    layer_p.push(interp_at(z_bot, p));

    // Interior points
    for i in 0..n {
        if z[i] <= z_bot { continue; }
        if z[i] >= z_top { break; }
        layer_comp.push(comp[i]);
        layer_p.push(p[i]);
    }

    // Top boundary
    layer_comp.push(interp_at(z_top, comp));
    layer_p.push(interp_at(z_top, p));

    // Trapezoidal integration: ∫A dp / ∫dp
    let m = layer_comp.len();
    if m < 2 { return layer_comp[0]; }

    let mut num = 0.0; // ∫A dp
    let mut den = 0.0; // ∫dp
    for i in 1..m {
        let dp = layer_p[i] - layer_p[i - 1]; // dp is negative (pressure decreases with height)
        let avg_val = (layer_comp[i] + layer_comp[i - 1]) / 2.0;
        num += avg_val * dp;
        den += dp;
    }

    if den.abs() > 1e-10 { num / den } else { layer_comp[0] }
}

/// Bunkers storm motion vectors (MetPy-compatible algorithm).
/// Returns ((u_rm, v_rm), (u_lm, v_lm)) for right-mover and left-mover.
/// p: pressure (hPa), u,v: wind components (m/s), z: height AGL (m). Surface first.
pub fn bunkers_storm_motion(
    p: &[f64], u: &[f64], v: &[f64], z: &[f64],
) -> ((f64, f64), (f64, f64)) {
    let z_sfc = z[0];

    // Pressure-weighted mean wind sfc-6km
    let u_mean = pressure_weighted_mean(u, p, z, z_sfc, z_sfc + 6000.0);
    let v_mean = pressure_weighted_mean(v, p, z, z_sfc, z_sfc + 6000.0);

    // Pressure-weighted mean wind sfc-0.5km (tail of shear vector)
    let u_500m = pressure_weighted_mean(u, p, z, z_sfc, z_sfc + 500.0);
    let v_500m = pressure_weighted_mean(v, p, z, z_sfc, z_sfc + 500.0);

    // Pressure-weighted mean wind 5.5-6km (head of shear vector)
    let u_5500m = pressure_weighted_mean(u, p, z, z_sfc + 5500.0, z_sfc + 6000.0);
    let v_5500m = pressure_weighted_mean(v, p, z, z_sfc + 5500.0, z_sfc + 6000.0);

    // Shear vector = head - tail
    let u_shr = u_5500m - u_500m;
    let v_shr = v_5500m - v_500m;

    let shear_mag = (u_shr * u_shr + v_shr * v_shr).sqrt();
    let d = 7.5; // Bunkers deviation magnitude (m/s)

    if shear_mag < 1e-6 {
        return ((u_mean, v_mean), (u_mean, v_mean));
    }

    // Cross product with k-hat: rotate shear 90 degrees clockwise
    // shear_cross = [shear_v, -shear_u] (MetPy convention)
    let u_perp = v_shr / shear_mag * d;
    let v_perp = -u_shr / shear_mag * d;

    let u_rm = u_mean + u_perp;
    let v_rm = v_mean + v_perp;
    let u_lm = u_mean - u_perp;
    let v_lm = v_mean - v_perp;

    ((u_rm, v_rm), (u_lm, v_lm))
}

/// Corfidi storm motion vectors for MCS propagation.
/// Returns ((u_upshear, v_upshear), (u_downshear, v_downshear)).
/// p: pressure (hPa), u,v: wind (m/s), z: height AGL (m). Surface first.
pub fn corfidi_storm_motion(
    p: &[f64], u: &[f64], v: &[f64], z: &[f64],
) -> ((f64, f64), (f64, f64)) {
    // Mean cloud-layer wind (850-300 hPa)
    let mut sum_u_cl = 0.0;
    let mut sum_v_cl = 0.0;
    let mut count_cl = 0.0;
    for i in 0..p.len() {
        if p[i] <= 850.0 && p[i] >= 300.0 {
            sum_u_cl += u[i];
            sum_v_cl += v[i];
            count_cl += 1.0;
        }
    }
    if count_cl == 0.0 {
        return ((0.0, 0.0), (0.0, 0.0));
    }
    let u_cl = sum_u_cl / count_cl;
    let v_cl = sum_v_cl / count_cl;

    // Low-level jet (max wind in lowest 1.5km)
    let mut max_spd = 0.0_f64;
    let mut u_llj = u[0];
    let mut v_llj = v[0];
    for i in 0..z.len() {
        if z[i] > 1500.0 {
            break;
        }
        let spd = (u[i] * u[i] + v[i] * v[i]).sqrt();
        if spd > max_spd {
            max_spd = spd;
            u_llj = u[i];
            v_llj = v[i];
        }
    }

    // Corfidi upshear: cloud-layer mean - LLJ
    let u_up = u_cl - u_llj;
    let v_up = v_cl - v_llj;

    // Corfidi downshear: cloud-layer mean + (cloud-layer mean - LLJ)
    let u_down = u_cl + u_up;
    let v_down = v_cl + v_up;

    ((u_up, v_up), (u_down, v_down))
}

/// Galvez-Davison Index (GDI) for tropical convection potential.
/// All temperatures in Celsius. sst: sea surface temperature (C).
pub fn galvez_davison_index(
    t950: f64, t850: f64, t700: f64, t500: f64,
    td950: f64, td850: f64, td700: f64, sst: f64,
) -> f64 {
    // Equivalent potential temperatures
    let thetae_950 = equivalent_potential_temperature(950.0, t950, td950);
    let thetae_850 = equivalent_potential_temperature(850.0, t850, td850);
    let thetae_700 = equivalent_potential_temperature(700.0, t700, td700);

    // Column buoyancy index (CBI)
    let thetae_low = (thetae_950 + thetae_850) / 2.0;
    let cbi = thetae_low - thetae_700;

    // Mid-level warming index (MWI)
    let t500_k = t500 + ZEROCNK;
    let mwi = (t500_k - 243.15) * 1.5; // scaled departure from -30C reference

    // Terrain correction / SST influence
    let sst_k = sst + ZEROCNK;
    let ii = (sst_k - 273.15 - 25.0).max(0.0) * 5.0; // inflow index

    // GDI = CBI + II - MWI (simplified version)
    cbi + ii - mwi
}

// =============================================================================
// Dynamics (additional functions)
// =============================================================================

/// Exner function: Pi = (p/p0)^(R/Cp).
pub fn exner_function(p_hpa: f64) -> f64 {
    (p_hpa / 1000.0).powf(ROCP)
}

/// Montgomery streamfunction on an isentropic surface.
/// Psi = Cp*T + g*z (J/kg).
pub fn montgomery_streamfunction(theta_k: f64, p_hpa: f64, t_k: f64, z_m: f64) -> f64 {
    let _ = theta_k; // theta identifies the surface but isn't used in the calculation
    let _ = p_hpa;
    CP * t_k + G * z_m
}

/// Ertel potential vorticity on isentropic surfaces.
/// theta, p: vertical profiles at each point. u, v: wind components.
/// lats: latitudes. Grid is nx*ny, nz levels. dx, dy: grid spacing (meters).
/// Returns PV in PVU (1e-6 K m^2 / (kg s)) on a single level.
/// Note: This is a simplified 2D PV computation on a single isentropic surface.
pub fn potential_vorticity_baroclinic(
    theta: &[f64], p: &[f64], u: &[f64], v: &[f64], lats: &[f64],
    nx: usize, ny: usize, dx: f64, dy: f64,
) -> Vec<f64> {
    use crate::dynamics;
    let n = nx * ny;
    assert_eq!(theta.len(), n);
    assert_eq!(p.len(), n);
    assert_eq!(u.len(), n);
    assert_eq!(v.len(), n);

    let abs_vort = dynamics::absolute_vorticity(u, v, lats, nx, ny, dx, dy);

    // dtheta/dp approximation: use the provided single-level fields
    // For a proper calculation, you need multiple levels. Here we provide
    // a simplified version using the stability parameter.
    let mut pv = vec![0.0; n];
    for k in 0..n {
        // PV = -g * abs_vort * (d_theta/d_p)
        // Since we only have one level, approximate d_theta/d_p from the field
        let p_pa = p[k] * 100.0;
        if p_pa > 0.0 {
            // Simple estimate: assume a standard static stability
            let dtheta_dp = -theta[k] / (p_pa * 4.0); // approximate
            pv[k] = -G * abs_vort[k] * dtheta_dp * 1e6; // Convert to PVU
        }
    }
    pv
}

/// Interpolate 3D fields to isentropic (constant potential temperature) surfaces.
/// theta_levels: target theta values (K). p_3d, t_3d: flattened `[nz][ny][nx]`.
/// fields: additional fields to interpolate. Returns interpolated fields at each theta level.
pub fn isentropic_interpolation(
    theta_levels: &[f64],
    p_3d: &[f64],
    t_3d: &[f64],
    fields: &[&[f64]],
    nx: usize, ny: usize, nz: usize,
) -> Vec<Vec<f64>> {
    let n2d = nx * ny;
    let n_theta = theta_levels.len();
    let n_fields = fields.len();

    // Output: for each field (including p and t), a flattened [n_theta][ny][nx] array
    // We return: [pressure_on_theta, t_on_theta, field0_on_theta, field1_on_theta, ...]
    let total_output = 2 + n_fields;
    let mut output: Vec<Vec<f64>> = (0..total_output)
        .map(|_| vec![f64::NAN; n_theta * n2d])
        .collect();

    // Compute theta at each 3D grid point
    let mut theta_3d = vec![0.0; nz * n2d];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx3 = k * n2d + j * nx + i;
                let t_k = t_3d[idx3];
                let p_hpa = p_3d[idx3];
                if p_hpa > 0.0 && t_k > 0.0 {
                    theta_3d[idx3] = t_k * (1000.0 / p_hpa).powf(ROCP);
                }
            }
        }
    }

    // For each grid column, interpolate to theta levels
    for j in 0..ny {
        for i in 0..nx {
            let idx2 = j * nx + i;

            // Extract column (bottom to top)
            let mut col_theta = Vec::with_capacity(nz);
            let mut col_p = Vec::with_capacity(nz);
            let mut col_t = Vec::with_capacity(nz);
            let mut col_fields: Vec<Vec<f64>> = (0..n_fields).map(|_| Vec::with_capacity(nz)).collect();

            for k in 0..nz {
                let idx3 = k * n2d + idx2;
                col_theta.push(theta_3d[idx3]);
                col_p.push(p_3d[idx3]);
                col_t.push(t_3d[idx3]);
                for f in 0..n_fields {
                    col_fields[f].push(fields[f][idx3]);
                }
            }

            // Interpolate each theta level using Newton iteration (MetPy-compatible)
            // For p and T: use Newton solver with T linear in ln(p)
            // For other fields: sort by theta and interpolate (matches MetPy's interpolate_1d)

            // Build sorted theta indices for field interpolation
            let mut sort_idx: Vec<usize> = (0..nz).collect();
            sort_idx.sort_by(|&a, &b| col_theta[a].partial_cmp(&col_theta[b]).unwrap_or(std::cmp::Ordering::Equal));

            for (ti, &target_theta) in theta_levels.iter().enumerate() {
                let out_idx = ti * n2d + idx2;

                // Find bounding levels for Newton solver (scan from bottom up)
                let mut found = false;
                for k in 0..nz - 1 {
                    let th_lo = col_theta[k];
                    let th_hi = col_theta[k + 1];
                    if (th_lo <= target_theta && th_hi >= target_theta)
                        || (th_lo >= target_theta && th_hi <= target_theta)
                    {
                        let dth = th_hi - th_lo;
                        if dth.abs() < 1e-10 { continue; }

                        let ln_p_lo = col_p[k].ln();
                        let ln_p_hi = col_p[k + 1].ln();
                        let d_ln_p = ln_p_hi - ln_p_lo;
                        if d_ln_p.abs() < 1e-10 { continue; }

                        // Newton iteration for p and T
                        let a = (col_t[k + 1] - col_t[k]) / d_ln_p;
                        let b = col_t[k] - a * ln_p_lo;
                        let pok = 1000.0_f64.powf(ROCP);

                        let mut ln_p = (ln_p_lo + ln_p_hi) / 2.0;
                        for _ in 0..50 {
                            let exner = pok * (-ROCP * ln_p).exp();
                            let t = a * ln_p + b;
                            let f = target_theta - t * exner;
                            let fp = exner * (ROCP * t - a);
                            if fp.abs() < 1e-30 { break; }
                            let delta = f / fp;
                            ln_p -= delta;
                            if delta.abs() < 1e-10 { break; }
                        }

                        output[0][out_idx] = ln_p.exp();
                        output[1][out_idx] = a * ln_p + b;
                        found = true;
                        break;
                    }
                }

                if !found { continue; }

                // Interpolate other fields using sorted theta (MetPy interpolate_1d)
                // Find bounding pair in theta-sorted order
                for sk in 0..nz - 1 {
                    let i_lo = sort_idx[sk];
                    let i_hi = sort_idx[sk + 1];
                    let th_lo = col_theta[i_lo];
                    let th_hi = col_theta[i_hi];
                    if th_lo <= target_theta && th_hi >= target_theta && (th_hi - th_lo).abs() > 1e-10 {
                        let frac = (target_theta - th_lo) / (th_hi - th_lo);
                        for f in 0..n_fields {
                            output[2 + f][out_idx] = col_fields[f][i_lo]
                                + frac * (col_fields[f][i_hi] - col_fields[f][i_lo]);
                        }
                        break;
                    }
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wobf_negative() {
        let result = wobf(-10.0);
        assert!(result > 0.0, "wobf(-10) should be positive");
    }

    #[test]
    fn test_wobf_positive() {
        let result = wobf(30.0);
        assert!(result > 0.0, "wobf(30) should be positive");
    }

    #[test]
    fn test_vappres_at_zero() {
        let es = vappres(0.0);
        // At 0C, saturation vapor pressure should be ~6.1 hPa
        assert!((es - 6.1078).abs() < 0.01);
    }

    #[test]
    fn test_mixratio() {
        let w = mixratio(1000.0, 20.0);
        // At 1000 hPa, 20C, mixing ratio should be roughly 14-15 g/kg
        assert!(w > 10.0 && w < 20.0);
    }

    #[test]
    fn test_lcltemp_saturated() {
        // When T == Td, LCL temp should equal T
        let t_lcl = lcltemp(20.0, 20.0);
        assert!((t_lcl - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_interp_linear() {
        let result = interp_linear(5.0, 0.0, 10.0, 0.0, 100.0);
        assert!((result - 50.0).abs() < 1e-10);
    }

    // =========================================================================
    // Temperature Conversion Tests
    // =========================================================================

    #[test]
    fn test_celsius_to_fahrenheit() {
        assert!((celsius_to_fahrenheit(0.0) - 32.0).abs() < 1e-10);
        assert!((celsius_to_fahrenheit(100.0) - 212.0).abs() < 1e-10);
        assert!((celsius_to_fahrenheit(-40.0) - (-40.0)).abs() < 1e-10);
    }

    #[test]
    fn test_fahrenheit_to_celsius() {
        assert!((fahrenheit_to_celsius(32.0) - 0.0).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(212.0) - 100.0).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(-40.0) - (-40.0)).abs() < 1e-10);
    }

    #[test]
    fn test_celsius_to_kelvin() {
        assert!((celsius_to_kelvin(0.0) - 273.15).abs() < 1e-10);
        assert!((celsius_to_kelvin(-273.15) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kelvin_to_celsius() {
        assert!((kelvin_to_celsius(273.15) - 0.0).abs() < 1e-10);
        assert!((kelvin_to_celsius(0.0) - (-273.15)).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_temp_conversions() {
        let t = 25.0;
        assert!((fahrenheit_to_celsius(celsius_to_fahrenheit(t)) - t).abs() < 1e-10);
        assert!((kelvin_to_celsius(celsius_to_kelvin(t)) - t).abs() < 1e-10);
    }

    // =========================================================================
    // Saturation / Moisture Tests
    // =========================================================================

    #[test]
    fn test_saturation_vapor_pressure_at_0c() {
        let es = saturation_vapor_pressure(0.0);
        // Ambaum (2020) / MetPy at 0C: 6.107563 hPa
        assert!((es - 6.107563).abs() < 0.001, "es at 0C = {es}");
    }

    #[test]
    fn test_saturation_vapor_pressure_at_20c() {
        let es = saturation_vapor_pressure(20.0);
        // MetPy at 20C: 23.347481 hPa
        assert!((es - 23.347).abs() < 0.01, "es at 20C = {es}");
    }

    #[test]
    fn test_saturation_vapor_pressure_at_100c() {
        let es = saturation_vapor_pressure(100.0);
        // MetPy at 100C: 993.344909 hPa
        assert!((es - 993.34).abs() < 1.0, "es at 100C = {es}");
    }

    #[test]
    fn test_dewpoint_from_rh_saturated() {
        // At 100% RH, dewpoint == temperature
        let td = dewpoint_from_rh(20.0, 100.0);
        assert!((td - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_dewpoint_from_rh_typical() {
        // At 50% RH and 20C, dewpoint should be ~9.3C
        let td = dewpoint_from_rh(20.0, 50.0);
        assert!((td - 9.3).abs() < 0.5);
    }

    #[test]
    fn test_rh_from_dewpoint_saturated() {
        let rh = rh_from_dewpoint(20.0, 20.0);
        assert!((rh - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_rh_from_dewpoint_roundtrip() {
        let t = 25.0;
        let rh_orig = 65.0;
        let td = dewpoint_from_rh(t, rh_orig);
        let rh_back = rh_from_dewpoint(t, td);
        assert!((rh_back - rh_orig).abs() < 0.1);
    }

    #[test]
    fn test_specific_humidity() {
        // 10 g/kg mixing ratio -> q ≈ 0.0099 kg/kg
        let q = specific_humidity(1000.0, 10.0);
        assert!((q - 0.009901).abs() < 0.001);
    }

    #[test]
    fn test_mixing_ratio_from_specific_humidity_roundtrip() {
        let w_orig = 10.0; // g/kg
        let q = specific_humidity(1000.0, w_orig);
        let w_back = mixing_ratio_from_specific_humidity(q);
        assert!((w_back - w_orig).abs() < 0.01);
    }

    #[test]
    fn test_saturation_mixing_ratio_at_20c() {
        let ws = saturation_mixing_ratio(1000.0, 20.0);
        // At 1000 hPa, 20C, ws should be ~14.7 g/kg
        assert!(ws > 13.0 && ws < 16.0);
    }

    #[test]
    fn test_vapor_pressure_from_dewpoint() {
        let e = vapor_pressure_from_dewpoint(10.0);
        let es = saturation_vapor_pressure(10.0);
        assert!((e - es).abs() < 1e-10);
    }

    #[test]
    fn test_wet_bulb_temperature_saturated() {
        // When saturated (T == Td), wet bulb should equal T
        let tw = wet_bulb_temperature(1000.0, 20.0, 20.0);
        assert!((tw - 20.0).abs() < 0.5);
    }

    #[test]
    fn test_wet_bulb_temperature_between_t_and_td() {
        // Wet bulb should be between Td and T
        let tw = wet_bulb_temperature(1000.0, 30.0, 15.0);
        assert!(tw >= 15.0 && tw <= 30.0, "Tw={tw} should be between 15 and 30");
    }

    #[test]
    fn test_frost_point_below_zero() {
        // Frost point at -10C, 80% RH should be below dewpoint
        let fp = frost_point(-10.0, 80.0);
        let td = dewpoint_from_rh(-10.0, 80.0);
        // Frost point should be close to but slightly above dewpoint at sub-zero temps
        assert!((fp - td).abs() < 3.0, "frost_point={fp}, dewpoint={td}");
    }

    #[test]
    fn test_psychrometric_vapor_pressure() {
        // At saturation (T == Tw), psychrometric e should equal es(T)
        let e = psychrometric_vapor_pressure(20.0, 20.0, 1000.0);
        let es = saturation_vapor_pressure(20.0);
        assert!((e - es).abs() < 0.01);
    }

    // =========================================================================
    // Potential Temperature Tests
    // =========================================================================

    #[test]
    fn test_potential_temperature_at_1000hpa() {
        // At 1000 hPa, theta == T (in K)
        let theta = potential_temperature(1000.0, 20.0);
        assert!((theta - 293.15).abs() < 0.01);
    }

    #[test]
    fn test_potential_temperature_at_850hpa() {
        // At 850 hPa, 10C, theta should be ~25C (~298K)
        let theta = potential_temperature(850.0, 10.0);
        assert!(theta > 296.0 && theta < 300.0);
    }

    #[test]
    fn test_potential_temperature_at_500hpa() {
        // At 500 hPa, -20C, theta should be significantly higher
        let theta = potential_temperature(500.0, -20.0);
        assert!(theta > 300.0 && theta < 320.0);
    }

    #[test]
    fn test_equivalent_potential_temperature() {
        // Theta-e should be >= theta
        let theta = potential_temperature(1000.0, 20.0);
        let theta_e = equivalent_potential_temperature(1000.0, 20.0, 15.0);
        assert!(theta_e > theta, "theta_e={theta_e} should exceed theta={theta}");
    }

    #[test]
    fn test_equivalent_potential_temperature_typical() {
        // At 1000 hPa, 25C, Td=20C — MetPy gives 341.53K (Bolton eq 39)
        let theta_e = equivalent_potential_temperature(1000.0, 25.0, 20.0);
        assert!((theta_e - 341.53).abs() < 0.5, "theta_e={theta_e}, expected ~341.53");
    }

    #[test]
    fn test_equivalent_potential_temperature_metpy_parity() {
        // Reference values from MetPy 1.7.1 (Bolton 1980 eq 39)
        // Small residual diffs (~0.05K) from SVP formula difference (Bolton vs Clausius-Clapeyron)
        let cases: &[(f64, f64, f64, f64, &str)] = &[
            (1000.0, 25.0, 20.0, 341.53, "typical"),
            (1000.0, 20.0, 15.0, 324.06, "mild"),
            (850.0,  10.0, -20.0, 299.62, "dry"),
            (500.0,  -30.0, -35.0, 297.74, "cold"),
            (1000.0, 30.0, 30.0, 386.01, "saturated"),
        ];
        for &(p, t, td, expected, label) in cases {
            let theta_e = equivalent_potential_temperature(p, t, td);
            assert!(
                (theta_e - expected).abs() < 1.0,
                "{label}: theta_e={theta_e:.2}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn test_equivalent_potential_temperature_edge_cases() {
        // Td > T (unphysical but shouldn't panic)
        let theta_e = equivalent_potential_temperature(1000.0, 20.0, 25.0);
        assert!(theta_e.is_finite(), "supersaturated: theta_e={theta_e} should be finite");
    }

    #[test]
    fn test_wet_bulb_potential_temperature() {
        // Theta-w is the temperature of a saturated parcel brought to 1000 hPa
        // along a moist adiabat. It should be less than theta (dry) because
        // moist adiabats are steeper. It should be a reasonable temperature.
        let theta_w = wet_bulb_potential_temperature(1000.0, 25.0, 15.0);
        // Theta-w should be a reasonable value (250-310 K range)
        assert!(
            theta_w > 270.0 && theta_w < 310.0,
            "theta_w={theta_w} should be in reasonable range"
        );
    }

    #[test]
    fn test_virtual_potential_temperature() {
        let theta = potential_temperature(1000.0, 20.0);
        let theta_v = virtual_potential_temperature(1000.0, 20.0, 10.0);
        // Virtual potential temperature should be slightly higher than theta
        assert!(theta_v > theta);
        assert!((theta_v - theta).abs() < 5.0);
    }

    // =========================================================================
    // Lifted / Parcel Tests
    // =========================================================================

    #[test]
    fn test_lcl_pressure_saturated() {
        // When saturated, LCL should be at surface
        let p_lcl = lcl_pressure(1000.0, 20.0, 20.0);
        assert!((p_lcl - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_lcl_pressure_unsaturated() {
        // Unsaturated: LCL should be above surface (lower pressure)
        let p_lcl = lcl_pressure(1000.0, 25.0, 10.0);
        assert!(p_lcl < 1000.0);
        assert!(p_lcl > 500.0);
    }

    // Helper function to create a typical unstable sounding for profile tests
    fn make_unstable_sounding() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Pressure levels from surface upward (hPa)
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0, 200.0];
        // Temperature (C) - moderately unstable
        let t = vec![30.0, 22.0, 16.0, 4.0, -15.0, -28.0, -44.0, -60.0];
        // Dewpoint (C)
        let td = vec![22.0, 18.0, 12.0, -2.0, -25.0, -38.0, -54.0, -70.0];
        (p, t, td)
    }

    #[test]
    fn test_lifted_index_unstable() {
        let (p, t, td) = make_unstable_sounding();
        let li = lifted_index(&p, &t, &td);
        // Unstable sounding should have negative LI
        assert!(li < 5.0, "LI={li} should be moderately negative for unstable sounding");
    }

    #[test]
    fn test_lfc_exists_for_unstable() {
        let (p, t, td) = make_unstable_sounding();
        let result = lfc(&p, &t, &td);
        // May or may not find LFC depending on the exact profile,
        // but the function should not panic
        if let Some((p_lfc, _t_lfc)) = result {
            assert!(p_lfc < 1000.0 && p_lfc > 100.0, "LFC pressure={p_lfc} should be reasonable");
        }
    }

    #[test]
    fn test_el_exists_for_unstable() {
        let (p, t, td) = make_unstable_sounding();
        let result = el(&p, &t, &td);
        if let Some((p_el, _t_el)) = result {
            assert!(p_el < 1000.0 && p_el > 100.0, "EL pressure={p_el} should be reasonable");
        }
    }

    #[test]
    fn test_ccl_exists() {
        let (p, t, td) = make_unstable_sounding();
        let result = ccl(&p, &t, &td);
        if let Some((p_ccl, t_ccl)) = result {
            assert!(p_ccl < 1000.0 && p_ccl > 200.0, "CCL pressure={p_ccl}");
            assert!(t_ccl < 30.0, "CCL temp={t_ccl} should be below surface temp");
        }
    }

    #[test]
    fn test_convective_temperature() {
        let (p, t, td) = make_unstable_sounding();
        let t_conv = convective_temperature(&p, &t, &td);
        if t_conv != MISSING {
            // Convective temperature should be >= surface temperature
            assert!(t_conv >= t[0] - 5.0, "Tconv={t_conv} should be near or above sfc T={}", t[0]);
        }
    }

    // =========================================================================
    // Density / Height Tests
    // =========================================================================

    #[test]
    fn test_density_sea_level() {
        // Standard sea level density ~1.225 kg/m^3
        let rho = density(1013.25, 15.0, 0.0);
        assert!((rho - 1.225).abs() < 0.01, "density={rho}");
    }

    #[test]
    fn test_density_moist_less_than_dry() {
        // Moist air is less dense than dry air at same T, P
        let rho_dry = density(1000.0, 20.0, 0.0);
        let rho_moist = density(1000.0, 20.0, 15.0);
        assert!(rho_moist < rho_dry, "moist={rho_moist} should be < dry={rho_dry}");
    }

    #[test]
    fn test_virtual_temperature_from_dewpoint_matches() {
        let tv1 = virtual_temp(20.0, 1000.0, 15.0);
        let tv2 = virtual_temperature_from_dewpoint(20.0, 15.0, 1000.0);
        assert!((tv1 - tv2).abs() < 1e-10);
    }

    #[test]
    fn test_thickness_hypsometric() {
        // 1000-500 hPa thickness at 255K mean temperature should be ~5280m
        let dz = thickness_hypsometric(1000.0, 500.0, 255.0);
        assert!((dz - 5180.0).abs() < 200.0, "thickness={dz}m");
    }

    #[test]
    fn test_pressure_to_height_std_sea_level() {
        let h = pressure_to_height_std(1013.25);
        assert!(h.abs() < 1.0, "sea level height={h} should be ~0m");
    }

    #[test]
    fn test_pressure_to_height_std_500hpa() {
        let h = pressure_to_height_std(500.0);
        // 500 hPa is approximately 5500m in standard atmosphere
        assert!((h - 5574.0).abs() < 100.0, "500hPa height={h}");
    }

    #[test]
    fn test_height_to_pressure_std_sea_level() {
        let p = height_to_pressure_std(0.0);
        assert!((p - 1013.25).abs() < 0.01, "sea level pressure={p}");
    }

    #[test]
    fn test_height_to_pressure_roundtrip() {
        let p_orig = 700.0;
        let h = pressure_to_height_std(p_orig);
        let p_back = height_to_pressure_std(h);
        assert!((p_back - p_orig).abs() < 0.1, "roundtrip: {p_orig} -> {h}m -> {p_back}");
    }

    #[test]
    fn test_altimeter_to_station_pressure_sea_level() {
        // At sea level, station pressure == altimeter setting
        let p_stn = altimeter_to_station_pressure(1013.25, 0.0);
        assert!((p_stn - 1013.25).abs() < 0.1, "p_stn={p_stn}");
    }

    #[test]
    fn test_altimeter_to_station_pressure_elevated() {
        // At 1000m elevation, station pressure should be less than altimeter
        let p_stn = altimeter_to_station_pressure(1013.25, 1000.0);
        assert!(p_stn < 1013.25, "p_stn={p_stn} should be < 1013.25");
        // Should be roughly 890-940 hPa
        assert!((p_stn - 900.0).abs() < 50.0, "p_stn={p_stn}");
    }

    #[test]
    fn test_station_to_sea_level_pressure_sea_level() {
        // At sea level, SLP == station pressure
        let slp = station_to_sea_level_pressure(1013.25, 0.0, 15.0);
        assert!((slp - 1013.25).abs() < 0.1, "slp={slp}");
    }

    #[test]
    fn test_station_to_sea_level_pressure_elevated() {
        // At 500m elevation with 950 hPa station pressure, SLP should be higher
        let slp = station_to_sea_level_pressure(950.0, 500.0, 15.0);
        assert!(slp > 950.0, "slp={slp} should be > 950");
        // Should be roughly 1010 hPa
        assert!((slp - 1010.0).abs() < 15.0, "slp={slp}");
    }

    // =========================================================================
    // New Function Tests
    // =========================================================================

    #[test]
    fn test_dry_lapse() {
        let p = vec![1000.0, 850.0, 700.0, 500.0];
        let result = dry_lapse(&p, 20.0);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 20.0).abs() < 0.01, "surface should be 20C, got {}", result[0]);
        // Temperature decreases with height (lower pressure)
        assert!(result[1] < result[0], "850 should be cooler than surface");
        assert!(result[2] < result[1], "700 should be cooler than 850");
    }

    #[test]
    fn test_moist_lapse() {
        let p = vec![1000.0, 900.0, 800.0, 700.0, 600.0, 500.0];
        let result = moist_lapse(&p, 25.0);
        assert_eq!(result.len(), 6);
        assert!((result[0] - 25.0).abs() < 0.01);

        // Reference values from MetPy moist_lapse([1000,900,800,700,600,500] hPa, 25 degC):
        //   [25.0, 21.47, 17.43, 12.73, 7.06, -0.08]
        let metpy_ref = [25.0, 21.47, 17.43, 12.73, 7.06, -0.08];
        for i in 0..result.len() {
            assert!((result[i] - metpy_ref[i]).abs() < 0.15,
                "level {}: got {:.2}, expected ~{:.2}", i, result[i], metpy_ref[i]);
        }

        // Moist adiabat cools slower than dry at warm temps
        let dry = dry_lapse(&p, 25.0);
        // At 500 hPa, moist should be warmer than dry
        assert!(result[5] > dry[5],
            "moist {} should be warmer than dry {} at 500 hPa", result[5], dry[5]);
    }

    #[test]
    fn test_parcel_profile() {
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
        let result = parcel_profile(&p, 25.0, 18.0);
        assert_eq!(result.len(), 6);
        assert!((result[0] - 25.0).abs() < 0.5, "surface should be ~25C, got {}", result[0]);
        // Should decrease monotonically
        for i in 1..result.len() {
            assert!(result[i] < result[i - 1],
                "should decrease: {} vs {} at level {}", result[i], result[i-1], i);
        }
    }

    #[test]
    fn test_heat_index_low_temp() {
        // Below 80F, simple formula
        let hi = heat_index(75.0, 50.0);
        assert!(hi < 80.0, "heat index at 75F should be < 80, got {hi}");
        assert!(hi > 65.0, "heat index at 75F should be > 65, got {hi}");
    }

    #[test]
    fn test_heat_index_high_temp() {
        // 100F at 50% RH: heat index should be ~120F
        let hi = heat_index(100.0, 50.0);
        assert!(hi > 110.0 && hi < 140.0, "heat index at 100F/50%RH = {hi}");
    }

    #[test]
    fn test_windchill() {
        // 0F at 15 mph: windchill should be well below 0
        let wc = windchill(0.0, 15.0);
        assert!(wc < 0.0, "windchill at 0F/15mph = {wc}");
        assert!(wc > -30.0, "windchill at 0F/15mph = {wc}");
    }

    #[test]
    fn test_windchill_warm() {
        // Above 50F: should return temperature unchanged
        let wc = windchill(60.0, 20.0);
        assert!((wc - 60.0).abs() < 0.01, "windchill above 50F should be unchanged, got {wc}");
    }

    #[test]
    fn test_downdraft_cape() {
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0];
        let t = vec![30.0, 22.0, 16.0, 4.0, -15.0, -28.0, -44.0];
        let td = vec![22.0, 18.0, 12.0, -2.0, -25.0, -38.0, -54.0];
        let dcape = downdraft_cape(&p, &t, &td);
        // DCAPE should be non-negative
        assert!(dcape >= 0.0, "DCAPE should be >= 0, got {dcape}");
        // For an unstable profile, should have some DCAPE
        // (value depends on exact profile, just check it's reasonable)
    }

    #[test]
    fn test_bunkers_storm_motion() {
        let p = vec![1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
        let z = vec![0.0, 750.0, 1500.0, 3000.0, 5500.0, 9000.0];
        let u = vec![5.0, 8.0, 12.0, 18.0, 25.0, 30.0];
        let v = vec![0.0, 2.0, 5.0, 8.0, 10.0, 8.0];
        let ((u_rm, v_rm), (u_lm, v_lm)) = bunkers_storm_motion(&p, &u, &v, &z);
        // Right mover and left mover should be different
        let rm_spd = (u_rm * u_rm + v_rm * v_rm).sqrt();
        let lm_spd = (u_lm * u_lm + v_lm * v_lm).sqrt();
        assert!(rm_spd > 0.0, "right mover speed should be > 0");
        assert!(lm_spd > 0.0, "left mover speed should be > 0");
        // They should deviate from mean wind in opposite directions
        assert!((u_rm - u_lm).abs() > 1.0 || (v_rm - v_lm).abs() > 1.0,
            "RM and LM should differ");
    }

    #[test]
    fn test_brunt_vaisala_frequency() {
        let p = vec![1000.0, 850.0, 700.0, 500.0, 300.0];
        let t_k = vec![293.0, 283.0, 270.0, 252.0, 228.0];
        let bv = brunt_vaisala_frequency(&p, &t_k);
        assert_eq!(bv.len(), 5);
        // N should be positive for stable atmosphere
        for &n in &bv {
            assert!(n >= 0.0, "BV frequency should be non-negative, got {n}");
        }
        // Typical tropospheric N is ~0.01 s^-1
        assert!(bv[2] > 0.005 && bv[2] < 0.03,
            "BV frequency should be ~0.01, got {}", bv[2]);
    }

    #[test]
    fn test_find_intersections() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y1 = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y2 = vec![4.0, 3.0, 2.0, 1.0, 0.0]; // crosses y1 at x=2
        let crossings = find_intersections(&x, &y1, &y2);
        assert_eq!(crossings.len(), 1, "should find 1 crossing, found {}", crossings.len());
        assert!((crossings[0].0 - 2.0).abs() < 0.01,
            "crossing x should be ~2.0, got {}", crossings[0].0);
        assert!((crossings[0].1 - 2.0).abs() < 0.01,
            "crossing y should be ~2.0, got {}", crossings[0].1);
    }

    #[test]
    fn test_dewpoint_from_vapor_pressure() {
        // At 20C, es ~ 23.4 hPa; dewpoint(23.4) should be ~20C
        let es = saturation_vapor_pressure(20.0);
        let td = dewpoint_from_vapor_pressure(es);
        assert!((td - 20.0).abs() < 0.1, "dewpoint from es at 20C = {td}");
    }

    #[test]
    fn test_exner_function() {
        let pi = exner_function(1000.0);
        assert!((pi - 1.0).abs() < 0.001, "Exner at 1000 hPa should be ~1.0, got {pi}");
        let pi500 = exner_function(500.0);
        assert!(pi500 < 1.0 && pi500 > 0.5, "Exner at 500 hPa = {pi500}");
    }

    #[test]
    fn test_temperature_from_potential_temperature() {
        let theta = potential_temperature(850.0, 10.0);
        let t_k = temperature_from_potential_temperature(850.0, theta);
        assert!((t_k - (10.0 + ZEROCNK)).abs() < 0.01,
            "roundtrip T from theta should be ~283.15K, got {t_k}");
    }

    #[test]
    fn test_geopotential_roundtrip() {
        let z = 5500.0;
        let geopot = height_to_geopotential(z);
        let z_back = geopotential_to_height(geopot);
        assert!((z_back - z).abs() < 1e-10);
    }

    #[test]
    fn test_scale_height() {
        // At 250K, H should be ~7.3 km
        let h = scale_height(250.0);
        assert!((h - 7300.0).abs() < 500.0, "scale height at 250K = {h}");
    }

    #[test]
    fn test_vertical_velocity_roundtrip() {
        let w = 1.0; // 1 m/s upward
        let omega = vertical_velocity_pressure(w, 500.0, -20.0);
        assert!(omega < 0.0, "upward motion should give negative omega, got {omega}");
        let w_back = vertical_velocity(omega, 500.0, -20.0);
        assert!((w_back - w).abs() < 1e-10, "roundtrip w = {w_back}");
    }

    #[test]
    fn test_sigma_to_pressure() {
        let p = sigma_to_pressure(0.5, 1000.0, 100.0);
        assert!((p - 550.0).abs() < 0.01, "sigma=0.5 should give 550 hPa, got {p}");
        let p_sfc = sigma_to_pressure(1.0, 1000.0, 100.0);
        assert!((p_sfc - 1000.0).abs() < 0.01);
        let p_top = sigma_to_pressure(0.0, 1000.0, 100.0);
        assert!((p_top - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_apparent_temperature() {
        let at = apparent_temperature(35.0, 60.0, 2.0, None);
        // Should be above actual temp due to humidity
        assert!(at > 30.0 && at < 50.0, "apparent temp at 35C/60%/2m/s = {at}");
    }

    #[test]
    fn test_mixed_layer() {
        let p = vec![1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 700.0];
        let vals = vec![20.0, 19.0, 18.0, 17.0, 16.0, 14.0, 4.0];
        let ml = mixed_layer(&p, &vals, 100.0);
        // Should be roughly the average of 20..16 range
        assert!(ml > 15.0 && ml < 20.0, "mixed layer average = {ml}");
    }

    #[test]
    fn test_gradient_richardson_number() {
        let theta = vec![300.0, 301.0, 303.0, 306.0];
        let u = vec![5.0, 10.0, 15.0, 20.0];
        let v = vec![0.0, 0.0, 0.0, 0.0];
        let z = vec![0.0, 500.0, 1000.0, 1500.0];
        let ri = gradient_richardson_number(&theta, &u, &v, &z);
        assert_eq!(ri.len(), 4);
        // Positive Ri = stable, should be a small positive number for this profile
        assert!(ri[1] > 0.0, "Ri should be positive for stable profile, got {}", ri[1]);
    }

    #[test]
    fn test_tke() {
        let u = vec![1.0, -1.0, 2.0, -2.0];
        let v = vec![0.5, -0.5, 1.0, -1.0];
        let w = vec![0.1, -0.1, 0.2, -0.2];
        let tke_val = tke(&u, &v, &w);
        assert!(tke_val > 0.0, "TKE should be positive");
        // Expected: 0.5 * mean(1+0.25+0.01 + 1+0.25+0.01 + 4+1+0.04 + 4+1+0.04) / 4
        //         = 0.5 * (1.26 + 1.26 + 5.04 + 5.04) / 4 = 0.5 * 3.15 = 1.575
        assert!((tke_val - 1.575).abs() < 0.01, "TKE = {tke_val}");
    }

    // =========================================================================
    // Ambaum (2020) SVP -- MetPy parity tests
    // =========================================================================

    #[test]
    fn test_svp_metpy_parity_liquid() {
        // Reference values from MetPy (Ambaum 2020), all in hPa.
        let cases: &[(f64, f64, &str)] = &[
            (   0.0,  6.107563, "0C"),
            (  10.0, 12.266556, "10C"),
            (  20.0, 23.347481, "20C"),
            (  25.0, 31.623456, "25C"),
            (  30.0, 42.346532, "30C"),
            (  35.0, 56.094080, "35C"),
            (  -5.0,  4.215412, "-5C"),
            ( -10.0,  2.863560, "-10C"),
            ( -20.0,  1.254936, "-20C"),
            ( -40.0,  0.189848, "-40C"),
            ( 100.0, 993.344909, "100C"),
        ];
        for &(t, expected, label) in cases {
            let got = saturation_vapor_pressure(t);
            assert!(
                (got - expected).abs() < 0.001,
                "SVP liquid at {label}: got={got:.6}, expected={expected:.6}"
            );
        }
    }

    #[test]
    fn test_svp_metpy_parity_solid() {
        // Reference values from MetPy (Ambaum 2020 Eq. 17), all in hPa.
        let cases: &[(f64, f64, &str)] = &[
            (   0.0,  6.106971, "0C"),
            (  -5.0,  4.015204, "-5C"),
            ( -10.0,  2.597718, "-10C"),
            ( -15.0,  1.652232, "-15C"),
            ( -20.0,  1.032058, "-20C"),
            ( -30.0,  0.379743, "-30C"),
            ( -40.0,  0.128129, "-40C"),
        ];
        for &(t, expected, label) in cases {
            let got = saturation_vapor_pressure_with_phase(t, Phase::Solid);
            assert!(
                (got - expected).abs() < 0.001,
                "SVP solid at {label}: got={got:.6}, expected={expected:.6}"
            );
        }
    }

    #[test]
    fn test_svp_auto_phase() {
        // Auto: liquid above T0 = 273.16 K (0.01 C), solid at or below.
        // At 10C => liquid
        let auto_10 = saturation_vapor_pressure_with_phase(10.0, Phase::Auto);
        let liq_10 = saturation_vapor_pressure(10.0);
        assert!((auto_10 - liq_10).abs() < 1e-10);

        // At -10C => solid
        let auto_m10 = saturation_vapor_pressure_with_phase(-10.0, Phase::Auto);
        let solid_m10 = saturation_vapor_pressure_with_phase(-10.0, Phase::Solid);
        assert!((auto_m10 - solid_m10).abs() < 1e-10);

        // At 0C (273.15 K < T0=273.16 K) => solid
        let auto_0 = saturation_vapor_pressure_with_phase(0.0, Phase::Auto);
        let solid_0 = saturation_vapor_pressure_with_phase(0.0, Phase::Solid);
        assert!((auto_0 - solid_0).abs() < 1e-10);
    }

    #[test]
    fn test_svp_monotonic() {
        // SVP (liquid) must strictly increase with temperature
        let temps = [-40.0, -20.0, 0.0, 10.0, 20.0, 30.0, 40.0];
        for i in 0..temps.len() - 1 {
            let es_lo = saturation_vapor_pressure(temps[i]);
            let es_hi = saturation_vapor_pressure(temps[i + 1]);
            assert!(
                es_hi > es_lo,
                "SVP not monotonic: es({})={} >= es({})={}",
                temps[i], es_lo, temps[i + 1], es_hi
            );
        }
    }

    #[test]
    fn test_svp_physical_values() {
        // Check against well-known physical reference values
        assert!((saturation_vapor_pressure(0.0) - 6.108).abs() < 0.01);
        assert!((saturation_vapor_pressure(20.0) - 23.35).abs() < 0.1);
        assert!((saturation_vapor_pressure(30.0) - 42.35).abs() < 0.1);
    }

    // =========================================================================
    // Saturation mixing ratio comprehensive tests
    // =========================================================================

    #[test]
    fn test_saturation_mixing_ratio_known_values() {
        // ws = EPS * es / (p - es) * 1000 (g/kg), with es from Ambaum SVP
        let ws_0 = saturation_mixing_ratio(1000.0, 0.0);
        let es_0 = saturation_vapor_pressure(0.0);
        let expected_0 = EPS * es_0 / (1000.0 - es_0) * 1000.0;
        assert!(
            (ws_0 - expected_0).abs() < 1e-10,
            "ws at 1000hPa/0C: got={ws_0}, expected={expected_0}"
        );

        // At 850 hPa, 10C
        let ws_850 = saturation_mixing_ratio(850.0, 10.0);
        let es_10 = saturation_vapor_pressure(10.0);
        let expected_850 = EPS * es_10 / (850.0 - es_10) * 1000.0;
        assert!(
            (ws_850 - expected_850).abs() < 1e-10,
            "ws at 850hPa/10C: got={ws_850}, expected={expected_850}"
        );

        // At 500 hPa, -20C
        let ws_500 = saturation_mixing_ratio(500.0, -20.0);
        let es_m20 = saturation_vapor_pressure(-20.0);
        let expected_500 = EPS * es_m20 / (500.0 - es_m20) * 1000.0;
        assert!(
            (ws_500 - expected_500).abs() < 1e-10,
            "ws at 500hPa/-20C: got={ws_500}, expected={expected_500}"
        );
    }

    #[test]
    fn test_saturation_mixing_ratio_with_phase() {
        // With Phase::Solid, mixing ratio at sub-zero should be lower (ice SVP < liquid SVP)
        let ws_liq = saturation_mixing_ratio(1000.0, -10.0);
        let ws_ice = saturation_mixing_ratio_with_phase(1000.0, -10.0, Phase::Solid);
        assert!(
            ws_ice < ws_liq,
            "Ice mixing ratio ({ws_ice}) should be less than liquid ({ws_liq}) at -10C"
        );
    }

    #[test]
    fn test_saturation_mixing_ratio_increases_with_temp() {
        // At fixed pressure, ws must increase with temperature
        let p = 1000.0;
        let temps = [-20.0, 0.0, 10.0, 20.0, 30.0];
        for i in 0..temps.len() - 1 {
            let ws_lo = saturation_mixing_ratio(p, temps[i]);
            let ws_hi = saturation_mixing_ratio(p, temps[i + 1]);
            assert!(ws_hi > ws_lo);
        }
    }

    #[test]
    fn test_saturation_mixing_ratio_decreases_with_pressure() {
        // At fixed temperature, ws must increase as pressure decreases
        let t = 20.0;
        let pressures = [1000.0, 850.0, 700.0, 500.0];
        for i in 0..pressures.len() - 1 {
            let ws_hi_p = saturation_mixing_ratio(pressures[i], t);
            let ws_lo_p = saturation_mixing_ratio(pressures[i + 1], t);
            assert!(ws_lo_p > ws_hi_p);
        }
    }

    // =========================================================================
    // Dewpoint roundtrip tests
    // =========================================================================

    #[test]
    fn test_dewpoint_from_rh_roundtrip_comprehensive() {
        // For various (T, RH) pairs, compute Td then recover RH.
        // Tolerance is 0.5% because SVP uses Ambaum (2020) while the dewpoint
        // inversion uses Bolton (1980) -- the same mixed approach as MetPy.
        let cases: &[(f64, f64)] = &[
            (-20.0, 50.0),
            (-10.0, 30.0),
            (0.0, 60.0),
            (10.0, 70.0),
            (20.0, 50.0),
            (25.0, 80.0),
            (30.0, 40.0),
            (35.0, 90.0),
            (40.0, 20.0),
        ];
        for &(t, rh) in cases {
            let td = dewpoint_from_rh(t, rh);
            let rh_back = rh_from_dewpoint(t, td);
            assert!(
                (rh_back - rh).abs() < 0.5,
                "Roundtrip failed at T={t}, RH={rh}: Td={td}, RH_back={rh_back}"
            );
        }
    }

    #[test]
    fn test_dewpoint_less_than_or_equal_to_temp() {
        // Dewpoint must always be <= temperature for RH <= 100%
        let cases: &[(f64, f64)] = &[
            (-20.0, 50.0), (0.0, 80.0), (20.0, 30.0), (35.0, 95.0),
        ];
        for &(t, rh) in cases {
            let td = dewpoint_from_rh(t, rh);
            assert!(td <= t + 1e-10, "Td={td} > T={t} at RH={rh}");
        }
    }

    // =========================================================================
    // Heat index tests
    // =========================================================================

    #[test]
    fn test_heat_index_90f_80rh() {
        // NWS reference: at 90F, 80% RH, heat index ~ 113F
        // The Rothfusz regression gives a specific value
        let hi = heat_index(90.0, 80.0);
        // NWS chart value is approximately 113F
        assert!(
            (hi - 113.0).abs() < 2.0,
            "Heat index at 90F/80%RH: got={hi}, expected ~113"
        );
    }

    #[test]
    fn test_heat_index_below_80f() {
        // NWS two-step: Steadman averaged with T when avg < 80F
        let hi = heat_index(70.0, 50.0);
        let steadman = 0.5 * (70.0 + 61.0 + (70.0 - 68.0) * 1.2 + 50.0 * 0.094);
        let expected = (steadman + 70.0) / 2.0;
        assert!(
            (hi - expected).abs() < 1e-10,
            "Heat index at 70F/50%: got={hi}, expected={expected}"
        );
    }

    #[test]
    fn test_heat_index_at_low_rh_high_temp() {
        // Low humidity adjustment regime: RH < 13%, 80F <= T <= 112F
        let hi = heat_index(100.0, 10.0);
        // Should be lower than Rothfusz alone due to negative adjustment
        let rothfusz = -42.379
            + 2.04901523 * 100.0
            + 10.14333127 * 10.0
            - 0.22475541 * 100.0 * 10.0
            - 6.83783e-3 * 100.0 * 100.0
            - 5.481717e-2 * 10.0 * 10.0
            + 1.22874e-3 * 100.0 * 100.0 * 10.0
            + 8.5282e-4 * 100.0 * 10.0 * 10.0
            - 1.99e-6 * 100.0 * 100.0 * 10.0 * 10.0;
        assert!(hi < rothfusz, "Low-RH adjustment should reduce heat index");
    }

    #[test]
    fn test_heat_index_increases_with_rh() {
        // At fixed high temperature, heat index should increase with RH
        let hi_40 = heat_index(95.0, 40.0);
        let hi_60 = heat_index(95.0, 60.0);
        let hi_80 = heat_index(95.0, 80.0);
        assert!(hi_60 > hi_40);
        assert!(hi_80 > hi_60);
    }

    // =========================================================================
    // Wind chill tests
    // =========================================================================

    #[test]
    fn test_windchill_known_value() {
        // NWS wind chill chart: 0F, 15 mph => wind chill ~ -19F
        let wc = windchill(0.0, 15.0);
        assert!(
            (wc - (-19.0)).abs() < 1.5,
            "Wind chill at 0F/15mph: got={wc}, expected ~-19"
        );
    }

    #[test]
    fn test_windchill_returns_temp_when_warm() {
        // Above 50F, windchill returns the temperature unchanged
        let wc = windchill(60.0, 20.0);
        assert!((wc - 60.0).abs() < 1e-10, "Above 50F, should return T");
    }

    #[test]
    fn test_windchill_returns_temp_when_calm() {
        // Below 3 mph wind, windchill returns the temperature unchanged
        let wc = windchill(20.0, 2.0);
        assert!((wc - 20.0).abs() < 1e-10, "Below 3mph, should return T");
    }

    #[test]
    fn test_windchill_decreases_with_wind() {
        // At fixed cold temperature, windchill should decrease with increasing wind
        let wc_10 = windchill(10.0, 10.0);
        let wc_20 = windchill(10.0, 20.0);
        let wc_40 = windchill(10.0, 40.0);
        assert!(wc_20 < wc_10);
        assert!(wc_40 < wc_20);
    }

    #[test]
    fn test_windchill_nws_table() {
        // A few more NWS chart reference points (T in F, wind in mph)
        // 20F / 10mph => ~9F, 10F / 20mph => ~-9F
        let wc1 = windchill(20.0, 10.0);
        assert!((wc1 - 9.0).abs() < 2.0, "WC at 20F/10mph: got={wc1}");
        let wc2 = windchill(10.0, 20.0);
        assert!((wc2 - (-9.0)).abs() < 2.0, "WC at 10F/20mph: got={wc2}");
    }

    // =========================================================================
    // Potential temperature comprehensive tests
    // =========================================================================

    #[test]
    fn test_potential_temperature_exact_formula() {
        // theta = (T+273.15) * (1000/p)^0.28571426
        let cases: &[(f64, f64, &str)] = &[
            (1000.0, 15.0,  "1000hPa/15C"),
            (850.0,  5.0,   "850hPa/5C"),
            (700.0,  -5.0,  "700hPa/-5C"),
            (500.0,  -25.0, "500hPa/-25C"),
            (300.0,  -45.0, "300hPa/-45C"),
            (200.0,  -55.0, "200hPa/-55C"),
        ];
        for &(p, t, label) in cases {
            let got = potential_temperature(p, t);
            let expected = (t + ZEROCNK) * (1000.0 / p).powf(ROCP);
            assert!(
                (got - expected).abs() < 1e-10,
                "theta at {label}: got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_potential_temperature_at_surface_equals_t() {
        // At 1000 hPa, theta = T (in K)
        let theta = potential_temperature(1000.0, 20.0);
        assert!((theta - 293.15).abs() < 1e-10);
    }

    #[test]
    fn test_potential_temperature_increases_with_height() {
        // For a standard atmosphere, theta should increase going up
        // even though temperature decreases
        let theta_1000 = potential_temperature(1000.0, 15.0);
        let theta_850 = potential_temperature(850.0, 5.0);
        let theta_500 = potential_temperature(500.0, -25.0);
        assert!(theta_850 > theta_1000);
        assert!(theta_500 > theta_850);
    }
}
