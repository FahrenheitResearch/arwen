//! CFE (Conceptual Functional Equivalent) on the CPU, for the prep-time spin-up.
//!
//! A line-for-line Rust port of NOAA-OWP/cfe f9182dfb3d81c407c66c5b782114173893c30cf2
//! (LICENSE: "Apache License", "Version 2.0, January 2004"): src/cfe.c `cfe`,
//! `Schaake_partitioning_scheme`, `et_from_rainfall`, `et_from_soil`;
//! src/conceptual_reservoir.c; src/giuh.c; src/nash_cascade.c
//! `nash_cascade_subsurface`; src/bmi_cfe.c `read_init_config_cfe` (the keys
//! used here), `Initialize`, `init_soil_reservoir` and `Update`. The order of
//! every floating-point operation follows the C source so the trajectory can
//! be compared word for word with the native build (`rw_hydro_init cfe-trace`).
//! Ported scope: Schaake partitioning, GIUH surface runoff, no root-zone AET,
//! no soil freeze-thaw coupling. Other options are refused, never approximated.
//! The run-time module stays the engine's CUDA CFE; this runs only before the
//! forecast, one catchment per rayon task.
use crate::Result;

pub const STANDARD_ATMOSPHERIC_PRESSURE_PASCALS: f64 = 101325.0;
pub const WATER_SPECIFIC_WEIGHT: f64 = 9810.0;
pub const TIME_STEP_SIZE_S: f64 = 3600.0;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params {
    pub depth: f64,
    pub bb: f64,
    pub satdk: f64,
    pub satpsi: f64,
    pub slop: f64,
    pub smcmax: f64,
    pub wltsmc: f64,
    pub expon: f64,
    pub expon_secondary: f64,
    pub refkdt: f64,
    pub alpha_fc: f64,
    pub max_gw_storage: f64,
    pub cgw: f64,
    pub gw_expon: f64,
    pub gw_storage: f64,
    pub soil_storage: f64,
    pub k_nash_subsurface: f64,
    pub n_nash_subsurface: usize,
    pub nash_storage_subsurface: Vec<f64>,
    pub k_lf: f64,
    pub giuh: Vec<f64>,
}

fn strtod(v: &str) -> f64 {
    // Leading numeric prefix, as C strtod reads it (0.0 when none).
    let v = v.trim_start();
    let mut end = 0;
    let b = v.as_bytes();
    let mut seen_e = false;
    for (i, c) in b.iter().enumerate() {
        let ok = c.is_ascii_digit()
            || *c == b'.'
            || ((*c == b'-' || *c == b'+') && (i == 0 || matches!(b[i - 1], b'e' | b'E')))
            || ((*c == b'e' || *c == b'E') && !seen_e && i > 0);
        if !ok {
            break;
        }
        seen_e |= *c == b'e' || *c == b'E';
        end = i + 1;
    }
    let mut s = &v[..end];
    while !s.is_empty() && s.parse::<f64>().is_err() {
        s = &s[..s.len() - 1];
    }
    s.parse::<f64>().unwrap_or(0.0)
}

/// Parse a CFE INI (the keys `hydro_world::derive::cfe_ini` and the ngen
/// examples write). Values are read with strtod semantics, units in [] dropped.
pub fn parse_ini(text: &str) -> Result<Params> {
    let mut p = Params { refkdt: 3.0, n_nash_subsurface: 2, expon: 1.0, expon_secondary: 1.0, ..Default::default() };
    let mut seen = std::collections::BTreeSet::new();
    let mut giuh = None;
    let mut nash = None;
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        let v = v.split('[').next().unwrap_or("");
        let k = k.trim();
        let x = || strtod(v);
        match k {
            "forcing_file" => {
                if v.trim() != "BMI" {
                    return Err("CFE spin-up takes forcing from the init source, so forcing_file must be BMI".into());
                }
            }
            "soil_params.refkdt" | "refkdt" => p.refkdt = x(),
            "soil_params.D" | "soil_params.depth" => p.depth = x(),
            "soil_params.bb" | "soil_params.b" => p.bb = x(),
            "soil_params.satdk" => p.satdk = x(),
            "soil_params.satpsi" => p.satpsi = x(),
            "soil_params.slope" | "soil_params.slop" => p.slop = x(),
            "soil_params.smcmax" | "soil_params.maxsmc" => p.smcmax = x(),
            "soil_params.wltsmc" => p.wltsmc = x(),
            "soil_params.expon" => p.expon = x(),
            "soil_params.expon_secondary" => p.expon_secondary = x(),
            "max_gw_storage" => p.max_gw_storage = x(),
            "Cgw" => p.cgw = x(),
            "expon" => p.gw_expon = x(),
            "gw_storage" => p.gw_storage = x(),
            "soil_params.alpha_fc" | "alpha_fc" => p.alpha_fc = x(),
            "soil_storage" => p.soil_storage = x(),
            "N_nash_subsurface" => p.n_nash_subsurface = v.trim().parse::<usize>()?,
            "K_nash_subsurface" | "K_nash" => p.k_nash_subsurface = x(),
            "K_lf" => p.k_lf = x(),
            "nash_storage_subsurface" => nash = Some(v.trim().to_string()),
            "giuh_ordinates" => giuh = Some(v.trim().to_string()),
            "surface_partitioning_scheme" | "surface_water_partitioning_scheme" => {
                if v.trim() != "Schaake" {
                    return Err(format!("CFE spin-up port covers Schaake partitioning; {} is not ported", v.trim()).into());
                }
            }
            "surface_runoff_scheme" => {
                if v.trim() != "GIUH" {
                    return Err(format!("CFE spin-up port covers GIUH surface runoff; {} is not ported", v.trim()).into());
                }
            }
            "is_aet_rootzone" | "is_sft_coupled" => {
                if strtod(v) != 0.0 || v.trim().eq_ignore_ascii_case("true") {
                    return Err(format!("CFE spin-up port does not cover {k}").into());
                }
            }
            _ => {}
        }
        seen.insert(k.to_string());
    }
    for need in ["soil_params.satdk", "soil_params.satpsi", "soil_params.smcmax", "soil_params.wltsmc", "max_gw_storage", "Cgw", "expon", "gw_storage", "soil_storage", "K_lf"] {
        if !seen.contains(need) {
            return Err(format!("CFE config lacks {need}").into());
        }
    }
    if !(seen.contains("K_nash") || seen.contains("K_nash_subsurface")) {
        return Err("CFE config lacks K_nash".into());
    }
    let g = giuh.ok_or("CFE config lacks giuh_ordinates")?;
    p.giuh = g.split(',').map(strtod).collect();
    p.nash_storage_subsurface = match nash {
        Some(s) => s.split(',').map(strtod).collect(),
        None => vec![0.0; p.n_nash_subsurface],
    };
    if p.nash_storage_subsurface.len() != p.n_nash_subsurface {
        p.n_nash_subsurface = p.nash_storage_subsurface.len();
    }
    if p.n_nash_subsurface > 3 {
        return Err("N_nash_subsurface exceeds CFE's MAX_NUM_NASH_CASCADE (3)".into());
    }
    Ok(p)
}

/// CFE's own field capacity as volumetric water: the soil reservoir's primary
/// threshold (init_soil_reservoir) divided by the soil depth.
pub fn field_capacity_m3m3(p: &Params) -> f64 {
    Cfe::new(p).soil.storage_threshold_primary_m / p.depth
}

#[derive(Clone, Debug, Default)]
pub struct Reservoir {
    pub is_exponential: bool,
    pub storage_max_m: f64,
    pub storage_m: f64,
    pub coeff_primary: f64,
    pub exponent_primary: f64,
    pub storage_threshold_primary_m: f64,
    pub coeff_secondary: f64,
    pub exponent_secondary: f64,
    pub storage_threshold_secondary_m: f64,
}

fn flux_calc(r: &Reservoir) -> (f64, f64) {
    if r.is_exponential {
        let exp_term = (r.exponent_primary * r.storage_m / r.storage_max_m).exp();
        return (r.coeff_primary * (exp_term - 1.0), 0.0);
    }
    let mut primary = 0.0;
    let above = r.storage_m - r.storage_threshold_primary_m;
    if above > 0.0 {
        primary = r.coeff_primary * (above / (r.storage_max_m - r.storage_threshold_primary_m)).powf(r.exponent_primary);
        if primary > above {
            primary = above;
        }
    }
    let mut secondary = 0.0;
    let above = r.storage_m - r.storage_threshold_secondary_m;
    if above > 0.0 {
        secondary = r.coeff_secondary * (above / (r.storage_max_m - r.storage_threshold_secondary_m)).powf(r.exponent_secondary);
        if secondary > above - primary {
            secondary = above - primary;
        }
    }
    (primary, secondary)
}

#[inline]
fn cmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// Complete per-catchment CFE state (the C struct members the step reads or writes).
#[derive(Clone, Debug)]
pub struct Cfe {
    pub soil: Reservoir,
    pub gw: Reservoir,
    pub wilting_point_m: f64,
    pub smcmax: f64,
    pub depth: f64,
    pub schaake_constant: f64,
    pub timestep_h: f64,
    pub giuh: Vec<f64>,
    pub queue: Vec<f64>,
    pub nash: Vec<f64>,
    pub k_nash: f64,
    // carried flux pointees (the C code reads some of them on the next call)
    pub soil_deficit_m: f64,
    pub infiltration_excess_m: f64,
    pub infiltration_depth_m: f64,
    pub flux_perc_m: f64,
    pub flux_lat_m: f64,
    pub gw_deficit_m: f64,
    pub flux_from_deep_gw_m: f64,
    pub direct_runoff_m: f64,
    pub nash_lateral_m: f64,
    pub qout_m: f64,
    pub actual_et_m: f64,
    pub perc_overflow_events: u64,
}

impl Cfe {
    pub fn new(p: &Params) -> Self {
        let timestep_h = TIME_STEP_SIZE_S / 3600.0;
        let gw = Reservoir {
            is_exponential: true,
            storage_max_m: p.max_gw_storage,
            storage_m: p.gw_storage * p.max_gw_storage,
            coeff_primary: p.cgw,
            exponent_primary: p.gw_expon,
            storage_threshold_primary_m: 0.0,
            coeff_secondary: 0.0,
            exponent_secondary: 1.0,
            storage_threshold_secondary_m: 0.0,
        };
        // init_soil_reservoir
        let omega = (p.alpha_fc * STANDARD_ATMOSPHERIC_PRESSURE_PASCALS / WATER_SPECIFIC_WEIGHT) - 0.5;
        let lower_lim = omega.powf(1.0 - 1.0 / p.bb) / (1.0 - 1.0 / p.bb);
        let upper_lim = (omega + p.depth).powf(1.0 - 1.0 / p.bb) / (1.0 - 1.0 / p.bb);
        let thr = p.smcmax * (1.0 / p.satpsi).powf(-1.0 / p.bb) * (upper_lim - lower_lim);
        let soil_storage = if p.soil_storage < 0.0 { 0.0 } else { p.soil_storage };
        let soil = Reservoir {
            is_exponential: false,
            storage_max_m: p.smcmax * p.depth,
            storage_m: soil_storage * p.smcmax * p.depth,
            coeff_primary: p.satdk * p.slop * TIME_STEP_SIZE_S,
            exponent_primary: p.expon,
            storage_threshold_primary_m: thr,
            coeff_secondary: p.k_lf,
            exponent_secondary: p.expon_secondary,
            storage_threshold_secondary_m: thr,
        };
        Cfe {
            soil,
            gw,
            wilting_point_m: p.wltsmc * p.depth,
            smcmax: p.smcmax,
            depth: p.depth,
            schaake_constant: p.refkdt * p.satdk / 0.000002,
            timestep_h,
            giuh: p.giuh.clone(),
            queue: vec![0.0; p.giuh.len() + 1],
            nash: p.nash_storage_subsurface.clone(),
            k_nash: p.k_nash_subsurface,
            soil_deficit_m: 0.0,
            infiltration_excess_m: 0.0,
            infiltration_depth_m: 0.0,
            flux_perc_m: 0.0,
            flux_lat_m: 0.0,
            gw_deficit_m: 0.0,
            flux_from_deep_gw_m: 0.0,
            direct_runoff_m: 0.0,
            nash_lateral_m: 0.0,
            qout_m: 0.0,
            actual_et_m: 0.0,
            perc_overflow_events: 0,
        }
    }

    /// Water held by the catchment (soil + groundwater + GIUH queue + Nash), m.
    pub fn storage(&self) -> f64 {
        let mut v = self.soil.storage_m + self.gw.storage_m;
        for q in &self.queue[..self.giuh.len()] {
            v += q;
        }
        for s in &self.nash {
            v += s;
        }
        v
    }

    /// One BMI Update: precipitation in mm/h (CFE's BMI unit), PET in m/s.
    pub fn update(&mut self, precip_mm_h: f64, pet_m_per_s: f64) {
        let mut rain = precip_mm_h / 1000.0;
        rain *= 1.0; // time_step_fraction
        self.step(rain, pet_m_per_s);
    }

    fn step(&mut self, rain_in: f64, pet_m_per_s: f64) {
        let mut rain = rain_in;
        let storage_temp_m = self.soil.storage_m;
        let pet_ts = pet_m_per_s * TIME_STEP_SIZE_S;
        let mut reduced = pet_m_per_s * TIME_STEP_SIZE_S;
        let mut et_rain = 0.0;
        if rain > 0.0 {
            // et_from_rainfall
            if rain > pet_ts {
                et_rain = pet_ts;
                rain -= et_rain;
            } else {
                et_rain = rain;
                rain = 0.0;
            }
            reduced = pet_ts - et_rain;
        }
        let et_retention = 0.0; // GIUH: no surface Nash storage
        let mut et_soil = 0.0;
        if self.soil.storage_m > self.wilting_point_m {
            // et_from_soil, non-root-zone branch
            if reduced > 0.0 {
                if self.soil.storage_m >= self.soil.storage_threshold_primary_m {
                    et_soil = cmin(reduced, self.soil.storage_m);
                } else if self.soil.storage_m > self.wilting_point_m && self.soil.storage_m < self.soil.storage_threshold_primary_m {
                    let budyko = (self.soil.storage_m - self.wilting_point_m) / (self.soil.storage_threshold_primary_m - self.wilting_point_m);
                    et_soil = cmin(budyko * reduced, self.soil.storage_m);
                }
                self.soil.storage_m -= et_soil;
                reduced -= et_soil;
            }
        }
        let _ = reduced;
        self.actual_et_m = et_rain + et_retention + et_soil;
        let mut deficit = self.smcmax * self.depth - self.soil.storage_m;
        let mut excess;
        let mut depth;
        if 0.0 < rain {
            // Schaake_partitioning_scheme (ice fraction 0, factor 1)
            if 0.0 < rain {
                if 0.0 > deficit {
                    excess = rain;
                    depth = 0.0;
                } else {
                    let timestep_d = self.timestep_h / 24.0;
                    let term = 1.0 - (-self.schaake_constant * timestep_d).exp();
                    let ic = deficit * term;
                    let px = rain;
                    depth = px * (ic / (px + ic));
                    if 0.0 < rain - depth {
                        excess = rain - depth;
                    } else {
                        excess = 0.0;
                    }
                    depth = rain - excess;
                }
            } else {
                excess = 0.0;
                depth = 0.0;
            }
            let factor = 1.0;
            depth *= factor;
            excess = rain - depth;
        } else {
            excess = 0.0;
            depth = 0.0;
        }
        if deficit < depth {
            excess += depth - deficit;
            depth = deficit;
            self.soil.storage_m = self.soil.storage_max_m;
            deficit = 0.0;
        }
        if self.flux_perc_m > deficit {
            let diff = self.flux_perc_m - deficit;
            depth = deficit;
            excess += diff;
            deficit = 0.0;
            self.perc_overflow_events += 1;
        }
        self.soil.storage_m += depth;
        let (perc, lat) = flux_calc(&self.soil);
        let mut flux_perc = perc;
        let flux_lat = lat;
        let gw_deficit = self.gw.storage_max_m - self.gw.storage_m;
        if flux_perc > gw_deficit {
            flux_perc = gw_deficit;
        }
        self.gw.storage_m += flux_perc;
        self.soil.storage_m -= flux_perc;
        self.soil.storage_m -= flux_lat;
        let _ = storage_temp_m;
        let (primary, _secondary) = flux_calc(&self.gw);
        let mut from_gw = primary;
        if from_gw > self.gw.storage_m {
            from_gw = self.gw.storage_m;
        }
        self.gw.storage_m -= from_gw;
        // giuh_convolution_integral
        let n = self.giuh.len();
        self.queue[n] = 0.0;
        for i in 0..n {
            self.queue[i] += self.giuh[i] * excess;
        }
        let direct = self.queue[0];
        for i in 1..=n {
            self.queue[i - 1] = self.queue[i];
        }
        // nash_cascade_subsurface
        let mut q = [0.0f64; 3];
        let nn = self.nash.len();
        for i in 0..nn {
            q[i] = self.k_nash * self.nash[i];
            self.nash[i] -= q[i];
            if i == 0 {
                self.nash[i] += flux_lat;
            } else {
                self.nash[i] += q[i - 1];
            }
        }
        let nash_out = q[nn - 1];
        self.soil_deficit_m = deficit;
        self.infiltration_excess_m = excess;
        self.infiltration_depth_m = depth;
        self.flux_perc_m = flux_perc;
        self.flux_lat_m = flux_lat;
        self.gw_deficit_m = gw_deficit;
        self.flux_from_deep_gw_m = from_gw;
        self.direct_runoff_m = direct;
        self.nash_lateral_m = nash_out;
        self.qout_m = direct + nash_out + from_gw;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strtod_prefixes() {
        assert_eq!(strtod("4.698e-06"), 4.698e-06);
        assert_eq!(strtod(" 0.35"), 0.35);
        assert_eq!(strtod("1.0992179"), 1.0992179);
        assert_eq!(strtod("abc"), 0.0);
        assert_eq!(strtod("2.0e"), 2.0);
    }
    #[test]
    fn dry_column_conserves_water() {
        let ini = "forcing_file=BMI\nsurface_partitioning_scheme=Schaake\nsoil_params.depth=2.0[m]\nsoil_params.b=6.0[]\nsoil_params.satdk=4.698e-06[m s-1]\nsoil_params.satpsi=0.141000003[m]\nsoil_params.slop=0.028126024[m/m]\nsoil_params.smcmax=0.444680541[m/m]\nsoil_params.wltsmc=0.046999998[m/m]\nrefkdt=1.0992179\nmax_gw_storage=0.012863189697[m]\nCgw=1.8e-05[m h-1]\nexpon=7.0[]\ngw_storage=0.35[m/m]\nalpha_fc=0.33\nsoil_storage=0.35[m/m]\nK_nash=0.03[]\nK_lf=0.01[]\ngiuh_ordinates=0.63,0.37\n";
        let p = parse_ini(ini).unwrap();
        let mut c = Cfe::new(&p);
        let s0 = c.storage();
        let (mut inp, mut out) = (0.0, 0.0);
        for h in 0..2000 {
            let rain = if h % 50 < 5 { 3.0 } else { 0.0 };
            c.update(rain, 2.0e-8);
            inp += rain / 1000.0;
            out += c.actual_et_m + c.qout_m;
        }
        let resid = inp - out - (c.storage() - s0);
        assert!(resid.abs() < 1e-12, "residual {resid}");
    }
}
