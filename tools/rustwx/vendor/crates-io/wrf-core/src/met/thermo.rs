/// Meteorological thermodynamic functions ported from wrfsolar's metfuncs.py.
/// Pure math - no external dependencies. All functions are direct ports of the
/// SHARPpy-derived implementations used in the Python codebase.
///
/// Vendored from wx-math crate for self-contained builds.

// --- Physical Constants ---
pub const RD: f64 = 287.058; // Dry air gas constant (J/(kg*K))
pub const RV: f64 = 461.5; // Water vapor gas constant (J/(kg*K))
pub const CP: f64 = 1005.7; // Specific heat at constant pressure (J/(kg*K))
pub const G: f64 = 9.80665; // Gravitational acceleration (m/s^2)
pub const ROCP: f64 = 0.28571426; // Rd/Cp
pub const ZEROCNK: f64 = 273.15; // 0 Celsius in Kelvin
pub const MISSING: f64 = -9999.0;
pub const EPS: f64 = 0.62197; // Rd/Rv = Mw/Md (ratio of molecular weights)
pub const LV: f64 = 2.501e6; // Latent heat of vaporization (J/kg)
pub const LAPSE_STD: f64 = 0.0065; // Standard atmosphere lapse rate (K/m)
pub const P0_STD: f64 = 1013.25; // Standard sea level pressure (hPa)
pub const T0_STD: f64 = 288.15; // Standard sea level temperature (K)

// --- SHARPpy Thermodynamic Approximations ---

/// Wobus function for computing moist adiabats.
/// Input: temperature in Celsius.
pub fn wobf(t: f64) -> f64 {
    let t = t - 20.0;
    if t <= 0.0 {
        let npol = 1.0
            + t * (-8.841660499999999e-3
                + t * (1.4714143e-4
                    + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))));
        15.13 / (npol * npol * npol * npol)
    } else {
        let ppol = t
            * (4.9618922e-07
                + t * (-6.1059365e-09
                    + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))));
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
    let p_lcl =
        1000.0 * ((t_lcl + ZEROCNK) / ((t + ZEROCNK) * ((1000.0 / p).powf(ROCP)))).powf(1.0 / ROCP);
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
    (10.0_f64.powf(c1 * x + c2) - c3 + c4 * (10.0_f64.powf(c5 * x) - c6).powi(2)) - ZEROCNK
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
    (t_prof[t_prof.len() - 1], td_prof[td_prof.len() - 1])
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
    let pressure_in_pa = psfc > 2000.0;
    let temperature_in_k = t2m > 150.0;
    let psfc_val = if pressure_in_pa { psfc / 100.0 } else { psfc };
    let t2m_val = if temperature_in_k { t2m - ZEROCNK } else { t2m };
    let mut td2m_val = if temperature_in_k {
        td2m - ZEROCNK
    } else {
        td2m
    };

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
        let pressure = if pressure_in_pa {
            p_prof[i] / 100.0
        } else {
            p_prof[i]
        };
        let temperature = if temperature_in_k {
            t_prof[i] - ZEROCNK
        } else {
            t_prof[i]
        };
        let dewpoint = if temperature_in_k {
            td_prof[i] - ZEROCNK
        } else {
            td_prof[i]
        };
        new_p.push(pressure);
        new_t.push(temperature);
        new_td.push(if dewpoint <= temperature {
            dewpoint
        } else {
            temperature
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

                    if buoyancy != buoy_prev {
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
        // Find the pressure at the target height AGL
        // Interpolate: given height_agl profile and p_prof, find p at top_m_val
        let mut p_at_top = p_prof[p_prof.len() - 1]; // default: model top
        for i in 0..height_agl.len() - 1 {
            if height_agl[i] <= top_m_val && top_m_val <= height_agl[i + 1] {
                let frac = (top_m_val - height_agl[i]) / (height_agl[i + 1] - height_agl[i]);
                p_at_top = p_prof[i] + frac * (p_prof[i + 1] - p_prof[i]);
                break;
            }
        }
        // For 3CAPE: stop at 3km, which is a HIGHER pressure (closer to surface)
        // than the EL. Use the larger pressure value (lower altitude) as the cap.
        if p_at_top > p_top_limit || p_top_limit <= 0.0 {
            p_top_limit = p_at_top;
        }
    }

    let mut total_cape = 0.0_f64;
    let mut total_cin = 0.0_f64;

    // --- Integrate CIN from Surface (p_start) to LCL (dry adiabat) ---
    let mut curr_dry_p = p_start;
    let mut dry_idx = start_idx;
    let dry_theta_start_k = (t_start + ZEROCNK) * ((1000.0 / p_start).powf(ROCP));
    let dry_parcel_mixratio = mixratio(p_start, td_start);

    while curr_dry_p > p_lcl {
        // Find next model level
        let mut next_p = -1.0_f64;
        let mut temp_idx = dry_idx;
        while temp_idx < p_prof.len() {
            if p_prof[temp_idx] < curr_dry_p - 0.01 {
                next_p = p_prof[temp_idx];
                dry_idx = temp_idx;
                break;
            }
            temp_idx += 1;
        }

        let target_dry_p = if next_p == -1.0 || next_p < p_lcl {
            p_lcl
        } else {
            next_p
        };

        // Standard sub-stepping for the dry layer
        let p1 = curr_dry_p;
        let p2 = target_dry_p;
        let p_mid = (p1 + p2) / 2.0;

        // Environment at p_mid
        let (t_env, td_env) = get_env_at_pres(p_mid, &p_prof, &t_prof, &td_prof);
        let tv_env = virtual_temp(t_env, p_mid, td_env);

        // Parcel temperature via dry adiabat
        let t_parc_k = dry_theta_start_k * ((p_mid / 1000.0).powf(ROCP));
        let t_parc = t_parc_k - ZEROCNK;

        // Parcel mixing ratio is constant (from starting dewpoint)
        // Virtual Temp of Parcel with known W
        let tv_parc = (t_parc + ZEROCNK) * (1.0 + 0.61 * (dry_parcel_mixratio / 1000.0)) - ZEROCNK;

        let val = RD * (tv_parc - tv_env) * (p1 / p2).ln();

        // In the dry layer, only accumulate CIN
        if val < 0.0 {
            total_cin += val;
        }

        curr_dry_p = target_dry_p;
    }

    // --- Integrate from LCL to EL (moist adiabat) ---
    let mut curr_p = p_lcl;
    let mut idx = 0;
    while idx < p_prof.len() && p_prof[idx] > p_lcl {
        idx += 1;
    }

    while curr_p > p_top_limit {
        // Find next model level
        let mut next_model_p = -1.0_f64;
        let mut temp_idx = idx;
        while temp_idx < p_prof.len() {
            if p_prof[temp_idx] < curr_p - 0.01 {
                next_model_p = p_prof[temp_idx];
                idx = temp_idx;
                break;
            }
            temp_idx += 1;
        }

        let target_p = if next_model_p == -1.0 || next_model_p < p_top_limit {
            p_top_limit
        } else {
            next_model_p
        };

        let dp_total = curr_p - target_p;
        let n_steps = if dp_total > 10.0 {
            (dp_total / 10.0) as usize + 1
        } else {
            1
        };
        let step_size = dp_total / n_steps as f64;

        for k in 0..n_steps {
            let p1 = curr_p - k as f64 * step_size;
            let p2 = curr_p - (k + 1) as f64 * step_size;
            let p_mid = (p1 + p2) / 2.0;

            let (t_env, td_env) = get_env_at_pres(p_mid, &p_prof, &t_prof, &td_prof);
            let tv_env = virtual_temp(t_env, p_mid, td_env);

            let t_parc = satlift(p_mid, thetam);
            let tv_parc = virtual_temp(t_parc, p_mid, t_parc);

            let val = RD * (tv_parc - tv_env) * (p1 / p2).ln();

            if val > 0.0 {
                total_cape += val;
            } else {
                total_cin += val;
            }
        }

        curr_p = target_p;
    }

    (total_cape, total_cin, h_lcl, h_lfc)
}

// =============================================================================
// Saturation / Moisture Functions
// =============================================================================

/// Saturation vapor pressure (hPa) using Bolton (1980) formula.
/// Input: temperature in Celsius.
pub fn saturation_vapor_pressure(t_c: f64) -> f64 {
    6.112 * ((17.67 * t_c) / (t_c + 243.5)).exp()
}

/// Dewpoint (Celsius) from temperature (Celsius) and relative humidity (%).
/// Uses the Magnus formula inverted.
pub fn dewpoint_from_rh(t_c: f64, rh: f64) -> f64 {
    let rh_frac = rh / 100.0;
    let es = saturation_vapor_pressure(t_c);
    let e = rh_frac * es;
    // Invert Bolton: Td = 243.5 * ln(e/6.112) / (17.67 - ln(e/6.112))
    let ln_ratio = (e / 6.112).ln();
    243.5 * ln_ratio / (17.67 - ln_ratio)
}

// =============================================================================
// Potential Temperature Functions
// =============================================================================

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
    // theta_DL = T * (1000/(p-e))^kappa * (T/T_L)^(0.28*r)
    let theta_dl = t_k * (1000.0 / (p_hpa - e)).powf(ROCP) * (t_k / t_lcl).powf(0.28 * r);
    // theta_E = theta_DL * exp((3036/T_L - 1.78) * r * (1 + 0.448*r))
    theta_dl * ((3036.0 / t_lcl - 1.78) * r * (1.0 + 0.448 * r)).exp()
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

// =============================================================================
// Lifted / Parcel Functions
// =============================================================================

/// Lift a parcel and compute parcel temperature at each level.
/// Returns parcel virtual temperature profile above LCL via moist adiabat.
fn lift_parcel_profile(p_prof: &[f64], t_prof: &[f64], td_prof: &[f64]) -> (f64, f64, Vec<f64>) {
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

/// Equilibrium Level (EL).
/// Returns Option<(pressure_hPa, temperature_C)> of the EL.
/// Profiles should be surface-first, decreasing pressure.
pub fn el(p_profile: &[f64], t_profile: &[f64], td_profile: &[f64]) -> Option<(f64, f64)> {
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
