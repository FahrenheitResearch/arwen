//! WRF 4.6.1 `compute_eta` / `levels`, independent of forcing source.
//!
//! The stretched option uses WRF's REAL arithmetic; the original option
//! preserves its mixed REAL/REAL(KIND=8) assignments and iterations. A
//! separately compiled, unmodified Fortran extraction supplies the oracle.
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Clone, Copy, Debug)]
pub struct EtaOptions {
    pub option: i32,
    pub p_top: f32,
    pub max_dz: f32,
    pub dzbot: f32,
    pub stretch_s: f32,
    pub stretch_u: f32,
    pub base_temp: f32,
}

const RD: f32 = 287.0;
const CP: f32 = 7.0 * RD / 2.0;
const G: f32 = 9.81;
const P0: f32 = 100000.0;
const T0: f32 = 300.0;

pub fn generate(e_vert: usize, p: EtaOptions) -> Result<Vec<f32>, &'static str> {
    if e_vert < 3 {
        return Err("e_vert must include at least two mass layers for automatic eta generation");
    }
    if ![p.p_top, p.max_dz, p.dzbot, p.stretch_s, p.stretch_u, p.base_temp]
        .iter().all(|x| x.is_finite()) {
        return Err("automatic eta controls must be finite FP32 values");
    }
    if !(0.0 < p.p_top && p.p_top < P0) {
        return Err("automatic eta generation requires 0 < p_top_requested < 100000 Pa");
    }
    if p.max_dz <= 0.0 {
        return Err("max_dz must be positive for automatic eta generation");
    }
    let eta = match p.option {
        1 => original(e_vert, p)?,
        2 => stretched(e_vert - 1, p)?,
        _ => return Err("auto_levels_opt must be 1 or 2"),
    };
    if !eta.iter().all(|x| x.is_finite()) || eta[0] != 1.0 || eta[e_vert - 1] != 0.0
        || !eta.windows(2).all(|w| w[0] > w[1]) {
        return Err("automatic eta calculation produced nonfinite or nondecreasing levels; adjust e_vert, p_top_requested or generator controls");
    }
    Ok(eta)
}

fn stretched(nlev: usize, p: EtaOptions) -> Result<Vec<f32>, &'static str> {
    if p.dzbot <= 0.0 || p.stretch_s <= 0.0 || p.stretch_u <= 0.0 {
        return Err("dzbot, dzstretch_s and dzstretch_u must be positive for auto_levels_opt=2");
    }
    let mut eta = vec![0.0; nlev + 1];
    let mut zup = vec![0.0; nlev + 1]; // 1-based WRF indexing
    let tt = 290.0;
    let ztop = RD * tt / G * (P0 / p.p_top).ln();
    let mut dz = p.dzbot;
    zup[1] = dz;
    eta[0] = 1.0;
    // Evaluate the elementary exponential in FP64 and round to REAL once.
    // Windows expf differs from the WRF/Linux expf by an ULP at some inputs;
    // this wider elementary evaluation preserves all surrounding FP32
    // operations while giving the same full-level grid on both platforms.
    let at_height = |z: f32| (P0 * ((-G * z / RD / tt) as f64).exp() as f32 - p.p_top) / (P0 - p.p_top);
    eta[1] = at_height(zup[1]);
    let mut isave = 1;
    for i in 1..nlev {
        let a = p.stretch_u + (p.stretch_s - p.stretch_u)
            * ((p.max_dz * 0.5 - dz) / (p.max_dz * 0.5)).max(0.0);
        dz = a * dz;
        let dztest = (ztop - zup[isave]) / (nlev - isave) as f32;
        if dztest < dz { break; }
        isave = i + 1;
        zup[i + 1] = zup[i] + dz;
        eta[i + 1] = at_height(zup[i + 1]);
        if i == nlev - 1 {
            return Err("not enough eta levels to reach p_top; increase e_vert, dzbot or stretching, or adjust p_top_requested");
        }
    }
    dz = (ztop - zup[isave]) / (nlev - isave) as f32;
    if dz > 1.5 * p.max_dz {
        return Err("WRF automatic upper-layer thickness exceeds 1.5 * max_dz; increase e_vert or max_dz, or adjust generator controls");
    }
    for i in isave..nlev {
        zup[i + 1] = zup[i] + dz;
        eta[i + 1] = at_height(zup[i + 1]);
    }
    eta[nlev] = 0.0;
    Ok(eta)
}

fn original(n: usize, p: EtaOptions) -> Result<Vec<f32>, &'static str> {
    // This algorithm fixes eight PBL levels and inserts three transition
    // levels through index 12; fewer levels cannot address those stencils.
    if n < 12 { return Err("auto_levels_opt=1 requires e_vert >= 12 for its fixed PBL and transition levels"); }
    const PRAC: [f64; 59] = [
        1.0, 0.993, 0.983, 0.970, 0.954, 0.934, 0.909, 0.880, 0.850, 0.800, 0.750,
        0.700, 0.650, 0.600, 0.550, 0.500, 0.450, 0.400, 0.350, 0.300, 0.250, 0.200,
        0.150, 0.100, 0.080, 0.060, 0.040, 0.020, 0.015, 0.010, 0.009, 0.008, 0.007,
        0.006, 0.005, 0.004, 0.0035, 0.003, 0.0028, 0.0026, 0.0024, 0.0022, 0.002,
        0.0018, 0.0016, 0.0014, 0.0012, 0.001, 0.0009, 0.0008, 0.0007, 0.0006,
        0.0005, 0.0004, 0.0003, 0.0002, 0.0001, 0.00005, 0.0,
    ];
    let mut alb = vec![0.0_f64; n.max(59) + 1];
    let mut phb = vec![0.0_f64; n.max(59) + 1];
    let mut eta = vec![0.0_f32; n + 1];
    let mut znw = vec![0.0_f32; n + 1];
    let top = p.p_top as f64;
    let mub = P0 as f64 - top;
    // Registry defaults of the shared real initializer: base_pres=100000,
    // base_lapse=50, iso_temp=200, base_pres_strat=0 (strat branch inert).
    let alpha = |pb: f64| {
        let temp = 200.0_f64.max(p.base_temp as f64 + 50.0 * (pb / P0 as f64).ln());
        let ti = temp * (P0 as f64 / pb).powf((RD / CP) as f64) - T0 as f64;
        (RD / P0) as f64 * (ti + T0 as f64)
            * (pb / P0 as f64).powf((-(CP - RD) / CP) as f64)
    };
    for k in 1..59 {
        let pb = (PRAC[k - 1] + PRAC[k]) * 0.5 * mub + top;
        alb[k] = alpha(pb);
        phb[k + 1] = phb[k] - (PRAC[k] - PRAC[k - 1]) * mub * alb[k];
    }
    let mut dz = (phb[59] / G as f64 - phb[8] / G as f64) / (n - 8) as f32 as f64;
    if dz >= p.max_dz as f64 {
        return Err("WRF automatic layer thickness exceeds max_dz; increase e_vert or max_dz, or adjust p_top_requested");
    }
    for k in 1..=8 { eta[k] = PRAC[k - 1] as f32; }
    for k in 8..=n - 3 {
        let practical = PRAC.iter().copied().find(|z| *z < eta[k] as f64)
            .ok_or("auto_levels_opt=1 exhausted its reference column before reaching p_top")?;
        let pb = 0.5 * (eta[k] as f64 + practical) * mub + top;
        alb[k] = alpha(pb);
        eta[k + 1] = (eta[k] as f64 - dz * G as f64 / (mub * alb[k])) as f32;
        // The sum and multiplication are REAL in the Fortran expression.
        let pb = (0.5 * (eta[k] + eta[k + 1])) as f64 * mub + top;
        alb[k] = alpha(pb);
        eta[k + 1] = (eta[k] as f64 - dz * G as f64 / (mub * alb[k])) as f32;
        phb[k + 1] = phb[k] - (eta[k + 1] - eta[k]) as f64 * mub * alb[k];
    }
    let alb_max = alb[n - 3];
    znw[1..=n - 3].copy_from_slice(&eta[1..=n - 3]);
    for _outer in 0..5 {
        for _inner in 0..10 {
            for k in 8..=n - 4 {
                let pb = ((znw[k] + znw[k + 1]) * 0.5) as f64 * mub + top;
                alb[k] = alpha(pb);
                znw[k + 1] = (znw[k] as f64 - dz * G as f64 / (mub * alb[k])) as f32;
            }
            alb[n - 3] = alb_max;
            znw[n - 2] = 0.0;
        }
        for k in 1..=n - 3 {
            alb[k] = alpha(((znw[k] + znw[k + 1]) * 0.5) as f64 * mub + top);
        }
        phb[1] = 0.0;
        for k in 2..=n - 2 {
            phb[k] = phb[k - 1] - (znw[k] - znw[k - 1]) as f64 * mub * alb[k - 1];
        }
        dz = (phb[n - 2] / G as f64 - phb[8] / G as f64) / (n - 10) as f32 as f64;
    }
    if dz > p.max_dz as f64 {
        return Err("WRF converged layer thickness exceeds max_dz; increase e_vert or max_dz, or adjust p_top_requested");
    }
    for k in (9..=n - 2).rev() { znw[k + 2] = znw[k]; }
    znw[9] = 0.75 * znw[8] + 0.25 * znw[12];
    znw[10] = 0.50 * znw[8] + 0.50 * znw[12];
    znw[11] = 0.25 * znw[8] + 0.75 * znw[12];
    Ok(znw[1..].to_vec())
}

/// Additive ABI-v1 entry. On failure, output is untouched and `message`
/// receives the named generator failure (UTF-8, NUL terminated).
///
/// # Safety
/// `out` must address `e_vert` writable f32s. `message` must address
/// `message_capacity` writable bytes; it may be null only at capacity zero.
#[no_mangle]
pub unsafe extern "C" fn gpuwm_wrf_eta_f32(
    out: *mut f32, e_vert: usize, option: i32, p_top: f32, max_dz: f32,
    dzbot: f32, stretch_s: f32, stretch_u: f32, base_temp: f32,
    message: *mut u8, message_capacity: usize,
) -> i32 {
    if out.is_null() || (message_capacity > 0 && message.is_null()) { return crate::ERR_NULL; }
    if e_vert > isize::MAX as usize / std::mem::size_of::<f32>() { return crate::ERR_DIMENSION; }
    let result = catch_unwind(AssertUnwindSafe(|| generate(e_vert,
        EtaOptions { option, p_top, max_dz, dzbot, stretch_s, stretch_u, base_temp })));
    match result {
        Ok(Ok(values)) => {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, e_vert);
            if message_capacity > 0 { *message = 0; }
            crate::OK
        }
        Ok(Err(reason)) => {
            if message_capacity > 0 {
                let count = reason.len().min(message_capacity - 1);
                std::ptr::copy_nonoverlapping(reason.as_ptr(), message, count);
                *message.add(count) = 0;
            }
            crate::ERR_DIMENSION
        }
        Err(_) => crate::ERR_PANIC,
    }
}
