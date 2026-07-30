//! ECAPE-family diagnostics backed by ecape-rs.
//!
//! Uses earth-rotated winds plus a synthetic surface layer from PSFC/T2/Q2/U10/V10
//! so parcel selection and storm motion see the same near-surface context as the
//! rest of wrf-rust's severe-weather diagnostics.

use ecape_rs::{
    calc_ecape_parcel, CapeType, ParcelOptions, StormMotionType as EcapeStormMotionType,
};
use rayon::prelude::*;

use crate::compute::{ComputeOpts, StormMotion};
use crate::error::{WrfError, WrfResult};
use crate::file::WrfFile;

#[derive(Debug, Clone, Copy, Default)]
struct EcapeSummary {
    ecape: f64,
    ncape: f64,
    cape: f64,
    cin: f64,
    lfc: f64,
    el: f64,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedEcapeOpts {
    parcel: CapeType,
    storm_motion: EcapeStormMotionType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EcapeFailure {
    InsufficientLevels,
    Solver,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EcapeFailureReport {
    insufficient_columns: usize,
    solver_failures: usize,
}

impl EcapeFailureReport {
    fn record(&mut self, failure: Option<EcapeFailure>) {
        match failure {
            Some(EcapeFailure::InsufficientLevels) => self.insufficient_columns += 1,
            Some(EcapeFailure::Solver) => self.solver_failures += 1,
            None => {}
        }
    }

    fn total(self) -> usize {
        self.insufficient_columns + self.solver_failures
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct EcapeColumnResult {
    summary: EcapeSummary,
    failure: Option<EcapeFailure>,
}

const ECAPE_STACK_FIELDS: usize = 6;

fn dewpoint_k_from_q(q_kgkg: f64, p_pa: f64, temp_k: f64) -> f64 {
    let td_c = crate::met::composite::dewpoint_from_q(q_kgkg, p_pa / 100.0);
    (td_c + 273.15).min(temp_k)
}

fn validate_ecape_opts(opts: &ComputeOpts) -> WrfResult<()> {
    if opts.parcel_pressure.is_some()
        || opts.parcel_temperature.is_some()
        || opts.parcel_dewpoint.is_some()
    {
        return Err(WrfError::InvalidParam(
            "ecape variables do not support custom parcel thermodynamics; use parcel_type=\"sb\", \"ml\", or \"mu\"".into(),
        ));
    }

    if opts.storm_motion_method.is_some() && opts.storm_motion.is_none() {
        return Err(WrfError::InvalidParam(
            "ecape variables do not use storm_motion_method; use storm_motion for an explicit vector or storm_motion_type=\"bunkers_rm\"|\"bunkers_lm\"|\"mean_wind\"".into(),
        ));
    }

    if opts.top_m.is_some()
        || opts.bottom_m.is_some()
        || opts.depth_m.is_some()
        || opts.bottom_p.is_some()
        || opts.top_p.is_some()
    {
        return Err(WrfError::InvalidParam(
            "ecape variables do not support CAPE/SRH layer-bound options like top_m, bottom_m, depth_m, bottom_p, or top_p".into(),
        ));
    }

    Ok(())
}

fn resolve_ecape_opts(opts: &ComputeOpts) -> WrfResult<ResolvedEcapeOpts> {
    let parcel =
        CapeType::parse_normalized(opts.parcel_type.as_deref().unwrap_or("sb")).map_err(|err| {
            WrfError::InvalidParam(format!(
                "unsupported ECAPE parcel_type '{}'; expected 'sb', 'ml', or 'mu'",
                err.value()
            ))
        })?;
    if matches!(parcel, CapeType::UserDefined) {
        return Err(WrfError::InvalidParam(
            "ecape variables do not support custom parcel thermodynamics; use parcel_type=\"sb\", \"ml\", or \"mu\"".into(),
        ));
    }

    let storm_motion =
        EcapeStormMotionType::parse_normalized(opts.storm_motion_type.as_deref().unwrap_or("right_moving"))
            .map_err(|err| {
                WrfError::InvalidParam(format!(
                    "unsupported ECAPE storm_motion_type '{}'; expected 'bunkers_rm', 'bunkers_lm', or 'mean_wind'",
                    err.value()
                ))
            })?;

    Ok(ResolvedEcapeOpts {
        parcel,
        storm_motion,
    })
}

fn ecape_parcel_cache_slug(parcel: CapeType) -> &'static str {
    match parcel {
        CapeType::SurfaceBased => "sb",
        CapeType::MixedLayer => "ml",
        CapeType::MostUnstable => "mu",
        CapeType::UserDefined => "user_defined",
    }
}

fn ecape_storm_motion_cache_slug(storm_motion: EcapeStormMotionType) -> &'static str {
    match storm_motion {
        EcapeStormMotionType::RightMoving => "bunkers_rm",
        EcapeStormMotionType::LeftMoving => "bunkers_lm",
        EcapeStormMotionType::MeanWind => "mean_wind",
        EcapeStormMotionType::UserDefined => "user_defined",
    }
}

fn build_parcel_options(
    opts: &ComputeOpts,
    ij: usize,
    parcel_type: CapeType,
    storm_motion_type: EcapeStormMotionType,
) -> ParcelOptions {
    let mut parcel_opts = ParcelOptions {
        cape_type: parcel_type,
        storm_motion_type,
        entrainment_rate: opts.entrainment_rate,
        pseudoadiabatic: opts.pseudoadiabatic,
        ..ParcelOptions::default()
    };

    if let Some(storm_motion) = opts.storm_motion.as_ref() {
        let (u, v) = storm_motion.at(ij);
        parcel_opts.storm_motion_type = EcapeStormMotionType::UserDefined;
        parcel_opts.storm_motion_u_ms = Some(u);
        parcel_opts.storm_motion_v_ms = Some(v);
    }

    parcel_opts
}

fn push_level(
    pressure_pa: &mut Vec<f64>,
    height_m: &mut Vec<f64>,
    temp_k: &mut Vec<f64>,
    dewpoint_k: &mut Vec<f64>,
    u_ms: &mut Vec<f64>,
    v_ms: &mut Vec<f64>,
    p: f64,
    z: f64,
    t: f64,
    td: f64,
    u: f64,
    v: f64,
) {
    if !p.is_finite()
        || !z.is_finite()
        || !t.is_finite()
        || !td.is_finite()
        || !u.is_finite()
        || !v.is_finite()
    {
        return;
    }

    if let (Some(&last_p), Some(&last_z)) = (pressure_pa.last(), height_m.last()) {
        if p >= last_p || z <= last_z {
            return;
        }
    }

    pressure_pa.push(p);
    height_m.push(z);
    temp_k.push(t);
    dewpoint_k.push(td.min(t));
    u_ms.push(u);
    v_ms.push(v);
}

fn build_surface_augmented_ecape_column(
    pressure_3d: &[f64],
    temperature_k_3d: &[f64],
    qv_3d: &[f64],
    height_agl_3d: &[f64],
    u_earth_3d: &[f64],
    v_earth_3d: &[f64],
    psfc_pa: f64,
    t2_k: f64,
    q2_kgkg: f64,
    u10_ms: f64,
    v10_ms: f64,
    nz: usize,
    nxy: usize,
    ij: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut pressure_pa = Vec::with_capacity(nz + 1);
    let mut height_m = Vec::with_capacity(nz + 1);
    let mut temp_k = Vec::with_capacity(nz + 1);
    let mut dewpoint_k = Vec::with_capacity(nz + 1);
    let mut u_ms = Vec::with_capacity(nz + 1);
    let mut v_ms = Vec::with_capacity(nz + 1);

    push_level(
        &mut pressure_pa,
        &mut height_m,
        &mut temp_k,
        &mut dewpoint_k,
        &mut u_ms,
        &mut v_ms,
        psfc_pa,
        0.0,
        t2_k,
        dewpoint_k_from_q(q2_kgkg, psfc_pa, t2_k),
        u10_ms,
        v10_ms,
    );

    for k in 0..nz {
        let idx = k * nxy + ij;
        push_level(
            &mut pressure_pa,
            &mut height_m,
            &mut temp_k,
            &mut dewpoint_k,
            &mut u_ms,
            &mut v_ms,
            pressure_3d[idx],
            height_agl_3d[idx],
            temperature_k_3d[idx],
            dewpoint_k_from_q(qv_3d[idx], pressure_3d[idx], temperature_k_3d[idx]),
            u_earth_3d[idx],
            v_earth_3d[idx],
        );
    }

    (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms)
}

fn ecape_cache_key(opts: &ComputeOpts, resolved: ResolvedEcapeOpts) -> Option<String> {
    let parcel_type = ecape_parcel_cache_slug(resolved.parcel);
    let storm_motion_type = if opts.storm_motion.is_some() {
        "user_defined"
    } else {
        ecape_storm_motion_cache_slug(resolved.storm_motion)
    };
    let entrainment = opts
        .entrainment_rate
        .map(|v| format!("{:016x}", v.to_bits()))
        .unwrap_or_else(|| "auto".to_string());
    let pseudoadiabatic = match opts.pseudoadiabatic {
        Some(true) => "1",
        Some(false) => "0",
        None => "default",
    };
    let lake_interp = opts
        .lake_interp
        .map(|v| format!("{:016x}", v.to_bits()))
        .unwrap_or_else(|| "none".to_string());
    let storm_motion = match opts.storm_motion.as_ref() {
        None => "default".to_string(),
        Some(StormMotion::Uniform { u, v }) => {
            format!("uniform_{:016x}_{:016x}", u.to_bits(), v.to_bits())
        }
        Some(StormMotion::Grid { .. }) => return None,
    };
    let strict = if opts.ecape_strict.unwrap_or(false) {
        "strict"
    } else {
        "silent"
    };

    Some(format!(
        "ecape_stack_{parcel_type}_{storm_motion_type}_{entrainment}_{pseudoadiabatic}_{lake_interp}_{storm_motion}_{strict}"
    ))
}

fn pack_ecape_stack(
    ecape: &[f64],
    ncape: &[f64],
    cape: &[f64],
    cin: &[f64],
    lfc: &[f64],
    el: &[f64],
) -> Vec<f64> {
    let nxy = ecape.len();
    let mut stacked = Vec::with_capacity(ECAPE_STACK_FIELDS * nxy);
    stacked.extend_from_slice(ecape);
    stacked.extend_from_slice(ncape);
    stacked.extend_from_slice(cape);
    stacked.extend_from_slice(cin);
    stacked.extend_from_slice(lfc);
    stacked.extend_from_slice(el);
    stacked
}

fn unpack_ecape_stack(
    stacked: &[f64],
    nxy: usize,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if stacked.len() != ECAPE_STACK_FIELDS * nxy {
        return None;
    }

    Some((
        stacked[0..nxy].to_vec(),
        stacked[nxy..2 * nxy].to_vec(),
        stacked[2 * nxy..3 * nxy].to_vec(),
        stacked[3 * nxy..4 * nxy].to_vec(),
        stacked[4 * nxy..5 * nxy].to_vec(),
        stacked[5 * nxy..6 * nxy].to_vec(),
    ))
}

fn summarize_ecape_failures(results: &[EcapeColumnResult]) -> EcapeFailureReport {
    let mut report = EcapeFailureReport::default();
    for result in results {
        report.record(result.failure);
    }
    report
}

fn check_ecape_failures(report: EcapeFailureReport, opts: &ComputeOpts) -> WrfResult<()> {
    if opts.ecape_strict.unwrap_or(false) && report.total() > 0 {
        return Err(WrfError::Compute(format!(
            "ECAPE strict mode found {} failed columns ({} insufficient-column, {} solver-error); default mode silently zero-fills these columns",
            report.total(),
            report.insufficient_columns,
            report.solver_failures
        )));
    }

    Ok(())
}

fn compute_ecape_fields(
    f: &WrfFile,
    t: usize,
    opts: &ComputeOpts,
) -> WrfResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_ecape_opts(opts)?;
    let resolved = resolve_ecape_opts(opts)?;

    let nxy = f.nx * f.ny;
    if let Some(key) = ecape_cache_key(opts, resolved) {
        if let Some(stacked) = f.cached_field(&key) {
            if let Some(fields) = unpack_ecape_stack(&stacked, nxy) {
                return Ok(fields);
            }
        }
    }

    let pressure = f.full_pressure(t)?;
    let temperature = f.temperature(t)?;
    let qvapor = f.qvapor(t)?;
    let height_agl = f.height_agl(t)?;
    let psfc = f.psfc(t)?;
    let t2 = f.t2_for_opts(t, opts)?;
    let q2 = f.q2_for_opts(t, opts)?;
    let u_grid = f.u_destag(t)?;
    let v_grid = f.v_destag(t)?;
    let u10_grid = f.u10(t)?;
    let v10_grid = f.v10(t)?;
    let sina = f.sinalpha(t)?;
    let cosa = f.cosalpha(t)?;

    let nx = f.nx;
    let ny = f.ny;
    let nz = f.nz;
    let nxy = nx * ny;

    let mut u_earth = vec![0.0f64; u_grid.len()];
    let mut v_earth = vec![0.0f64; v_grid.len()];
    for idx in 0..u_grid.len() {
        let ij = idx % nxy;
        u_earth[idx] = u_grid[idx] * cosa[ij] - v_grid[idx] * sina[ij];
        v_earth[idx] = u_grid[idx] * sina[ij] + v_grid[idx] * cosa[ij];
    }

    let mut u10_earth = vec![0.0f64; nxy];
    let mut v10_earth = vec![0.0f64; nxy];
    for ij in 0..nxy {
        u10_earth[ij] = u10_grid[ij] * cosa[ij] - v10_grid[ij] * sina[ij];
        v10_earth[ij] = u10_grid[ij] * sina[ij] + v10_grid[ij] * cosa[ij];
    }

    let parcel_type = resolved.parcel;
    let storm_motion_type = resolved.storm_motion;

    let results: Vec<EcapeColumnResult> = (0..nxy)
        .into_par_iter()
        .map(|ij| {
            let (pressure_pa, height_m, temp_k, dewpoint_k, u_ms, v_ms) =
                build_surface_augmented_ecape_column(
                    &pressure,
                    &temperature,
                    &qvapor,
                    &height_agl,
                    &u_earth,
                    &v_earth,
                    psfc[ij],
                    t2[ij],
                    q2[ij],
                    u10_earth[ij],
                    v10_earth[ij],
                    nz,
                    nxy,
                    ij,
                );

            if pressure_pa.len() < 2 {
                return EcapeColumnResult {
                    summary: EcapeSummary::default(),
                    failure: Some(EcapeFailure::InsufficientLevels),
                };
            }

            let parcel_opts = build_parcel_options(opts, ij, parcel_type, storm_motion_type);
            match calc_ecape_parcel(
                &height_m,
                &pressure_pa,
                &temp_k,
                &dewpoint_k,
                &u_ms,
                &v_ms,
                &parcel_opts,
            ) {
                Ok(result) => EcapeColumnResult {
                    summary: EcapeSummary {
                        ecape: result.ecape_jkg,
                        ncape: result.ncape_jkg,
                        cape: result.cape_jkg,
                        cin: result.cin_jkg,
                        lfc: result.lfc_m.unwrap_or(0.0),
                        el: result.el_m.unwrap_or(0.0),
                    },
                    failure: None,
                },
                Err(_) => EcapeColumnResult {
                    summary: EcapeSummary::default(),
                    failure: Some(EcapeFailure::Solver),
                },
            }
        })
        .collect();

    check_ecape_failures(summarize_ecape_failures(&results), opts)?;

    let mut ecape = Vec::with_capacity(nxy);
    let mut ncape = Vec::with_capacity(nxy);
    let mut cape = Vec::with_capacity(nxy);
    let mut cin = Vec::with_capacity(nxy);
    let mut lfc = Vec::with_capacity(nxy);
    let mut el = Vec::with_capacity(nxy);

    for result in results {
        ecape.push(result.summary.ecape);
        ncape.push(result.summary.ncape);
        cape.push(result.summary.cape);
        cin.push(result.summary.cin);
        lfc.push(result.summary.lfc);
        el.push(result.summary.el);
    }

    if let Some(key) = ecape_cache_key(opts, resolved) {
        let stacked = pack_ecape_stack(&ecape, &ncape, &cape, &cin, &lfc, &el);
        f.store_cached_field(key, stacked);
    }

    Ok((ecape, ncape, cape, cin, lfc, el))
}

pub fn compute_ecape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (ecape, _, _, _, _, _) = compute_ecape_fields(f, t, opts)?;
    Ok(ecape)
}

pub fn compute_ncape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, ncape, _, _, _, _) = compute_ecape_fields(f, t, opts)?;
    Ok(ncape)
}

pub fn compute_ecape_cape(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, cape, _, _, _) = compute_ecape_fields(f, t, opts)?;
    Ok(cape)
}

pub fn compute_ecape_cin(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, cin, _, _) = compute_ecape_fields(f, t, opts)?;
    Ok(cin)
}

pub fn compute_ecape_lfc(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, _, lfc, _) = compute_ecape_fields(f, t, opts)?;
    Ok(lfc)
}

pub fn compute_ecape_el(f: &WrfFile, t: usize, opts: &ComputeOpts) -> WrfResult<Vec<f64>> {
    let (_, _, _, _, _, el) = compute_ecape_fields(f, t, opts)?;
    Ok(el)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::StormMotion;

    fn base_opts() -> ComputeOpts {
        ComputeOpts::default()
    }

    fn q_from_dewpoint_k(td_k: f64, p_pa: f64) -> f64 {
        let td_c = td_k - 273.15;
        let e_hpa = 6.112 * ((17.67 * td_c) / (td_c + 243.5)).exp();
        let p_hpa = p_pa / 100.0;
        0.622 * e_hpa / (p_hpa - e_hpa)
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1e-6_f64.max(expected.abs() * 1e-10);
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    #[test]
    fn rejects_custom_parcel_thermodynamics() {
        let mut opts = base_opts();
        opts.parcel_pressure = Some(900.0);
        opts.parcel_temperature = Some(20.0);
        opts.parcel_dewpoint = Some(15.0);

        let err = validate_ecape_opts(&opts).unwrap_err();
        assert!(matches!(err, WrfError::InvalidParam(_)));
    }

    #[test]
    fn resolves_supported_parcel_types() {
        let mut opts = base_opts();
        assert!(matches!(
            resolve_ecape_opts(&opts).unwrap().parcel,
            CapeType::SurfaceBased
        ));

        opts.parcel_type = Some("ml".into());
        assert!(matches!(
            resolve_ecape_opts(&opts).unwrap().parcel,
            CapeType::MixedLayer
        ));

        opts.parcel_type = Some("mu".into());
        assert!(matches!(
            resolve_ecape_opts(&opts).unwrap().parcel,
            CapeType::MostUnstable
        ));
    }

    #[test]
    fn resolves_supported_storm_motion_types() {
        let mut opts = base_opts();
        assert!(matches!(
            resolve_ecape_opts(&opts).unwrap().storm_motion,
            EcapeStormMotionType::RightMoving
        ));

        opts.storm_motion_type = Some("left moving".into());
        assert!(matches!(
            resolve_ecape_opts(&opts).unwrap().storm_motion,
            EcapeStormMotionType::LeftMoving
        ));

        opts.storm_motion_type = Some("mean-wind".into());
        assert!(matches!(
            resolve_ecape_opts(&opts).unwrap().storm_motion,
            EcapeStormMotionType::MeanWind
        ));
    }

    #[test]
    fn explicit_storm_motion_becomes_user_defined() {
        let mut opts = base_opts();
        opts.storm_motion = Some(StormMotion::Uniform { u: 12.0, v: 8.0 });

        let parcel_opts = build_parcel_options(
            &opts,
            0,
            CapeType::SurfaceBased,
            EcapeStormMotionType::RightMoving,
        );

        assert!(matches!(
            parcel_opts.storm_motion_type,
            EcapeStormMotionType::UserDefined
        ));
        assert_eq!(parcel_opts.storm_motion_u_ms, Some(12.0));
        assert_eq!(parcel_opts.storm_motion_v_ms, Some(8.0));
    }

    #[test]
    fn grid_storm_motion_skips_ecape_cache() {
        let mut opts = base_opts();
        opts.storm_motion = Some(StormMotion::Grid {
            u: vec![1.0, 2.0],
            v: vec![3.0, 4.0],
        });

        let resolved = resolve_ecape_opts(&opts).unwrap();
        assert!(ecape_cache_key(&opts, resolved).is_none());
    }

    #[test]
    fn ecape_cache_key_uses_resolver_canonical_names() {
        let mut alias_opts = base_opts();
        alias_opts.parcel_type = Some("surface".into());
        alias_opts.storm_motion_type = Some("right-moving".into());

        let mut canonical_opts = base_opts();
        canonical_opts.parcel_type = Some("sb".into());
        canonical_opts.storm_motion_type = Some("bunkers_rm".into());

        let alias_resolved = resolve_ecape_opts(&alias_opts).unwrap();
        let canonical_resolved = resolve_ecape_opts(&canonical_opts).unwrap();

        assert_eq!(alias_resolved.parcel, canonical_resolved.parcel);
        assert_eq!(alias_resolved.storm_motion, canonical_resolved.storm_motion);
        assert_eq!(
            ecape_cache_key(&alias_opts, alias_resolved),
            ecape_cache_key(&canonical_opts, canonical_resolved)
        );
    }

    #[test]
    fn ecape_strict_mode_has_separate_cache_key() {
        let silent_opts = base_opts();
        let mut strict_opts = base_opts();
        strict_opts.ecape_strict = Some(true);

        assert_ne!(
            ecape_cache_key(&silent_opts, resolve_ecape_opts(&silent_opts).unwrap()),
            ecape_cache_key(&strict_opts, resolve_ecape_opts(&strict_opts).unwrap())
        );
    }

    #[test]
    fn explicit_storm_motion_key_ignores_unused_motion_type() {
        let mut right_opts = base_opts();
        right_opts.storm_motion = Some(StormMotion::Uniform { u: 10.0, v: 5.0 });
        right_opts.storm_motion_type = Some("bunkers_rm".into());

        let mut left_opts = right_opts.clone();
        left_opts.storm_motion_type = Some("bunkers_lm".into());

        assert_eq!(
            ecape_cache_key(&right_opts, resolve_ecape_opts(&right_opts).unwrap()),
            ecape_cache_key(&left_opts, resolve_ecape_opts(&left_opts).unwrap())
        );
    }

    #[test]
    fn ecape_failure_report_counts_reasons() {
        let results = vec![
            EcapeColumnResult {
                summary: EcapeSummary::default(),
                failure: None,
            },
            EcapeColumnResult {
                summary: EcapeSummary::default(),
                failure: Some(EcapeFailure::InsufficientLevels),
            },
            EcapeColumnResult {
                summary: EcapeSummary::default(),
                failure: Some(EcapeFailure::Solver),
            },
        ];

        let report = summarize_ecape_failures(&results);
        assert_eq!(report.insufficient_columns, 1);
        assert_eq!(report.solver_failures, 1);
        assert_eq!(report.total(), 2);
    }

    #[test]
    fn ecape_failure_report_only_errors_in_strict_mode() {
        let report = EcapeFailureReport {
            insufficient_columns: 1,
            solver_failures: 2,
        };
        assert!(check_ecape_failures(report, &base_opts()).is_ok());

        let mut strict_opts = base_opts();
        strict_opts.ecape_strict = Some(true);
        let err = check_ecape_failures(report, &strict_opts).unwrap_err();
        assert!(matches!(err, WrfError::Compute(_)));
    }

    #[test]
    fn ecape_stack_round_trip_preserves_field_order() {
        let stacked = pack_ecape_stack(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
            &[11.0, 12.0],
        );

        let unpacked = unpack_ecape_stack(&stacked, 2).unwrap();
        assert_eq!(unpacked.0, vec![1.0, 2.0]);
        assert_eq!(unpacked.1, vec![3.0, 4.0]);
        assert_eq!(unpacked.2, vec![5.0, 6.0]);
        assert_eq!(unpacked.3, vec![7.0, 8.0]);
        assert_eq!(unpacked.4, vec![9.0, 10.0]);
        assert_eq!(unpacked.5, vec![11.0, 12.0]);
    }

    #[test]
    fn shared_parity_fixture_matches_ecape_rs_expected_values() {
        let height_m = [
            0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0,
            6000.0, 7500.0, 9000.0, 10500.0, 12000.0, 14000.0, 16000.0,
        ];
        let pressure_pa = [
            100000.0,
            96923.32344763441,
            93941.30628134758,
            91051.03613800342,
            88249.69025845954,
            82902.91181804004,
            77880.07830714049,
            73161.56289466418,
            68728.92787909723,
            60653.06597126334,
            53526.142851899036,
            47236.65527410147,
            39160.5626676799,
            32465.24673583497,
            26914.634872918385,
            22313.016014842982,
            17377.394345044515,
            13533.52832366127,
        ];
        let temperature_k = [
            302.0, 300.2, 298.4, 296.6, 294.8, 291.2, 287.6, 284.0, 280.4, 273.2, 266.0, 258.8,
            248.0, 237.2, 226.4, 215.6, 215.6, 215.6,
        ];
        let dewpoint_k = [
            296.0, 295.625, 295.25, 294.875, 294.3, 290.7, 287.1, 283.5, 279.9, 272.7, 265.5,
            258.3, 247.5, 236.7, 225.9, 215.1, 215.1, 215.1,
        ];
        let u_wind_ms = [
            4.0, 4.625, 5.25, 5.875, 6.5, 7.75, 9.0, 10.25, 11.5, 14.0, 16.5, 19.0, 22.75, 26.5,
            30.25, 34.0, 39.0, 44.0,
        ];
        let v_wind_ms = [
            1.0, 1.375, 1.75, 2.125, 2.5, 3.25, 4.0, 4.75, 5.5, 7.0, 8.5, 10.0, 12.25, 14.5, 16.75,
            19.0, 22.0, 25.0,
        ];
        let qvapor: Vec<f64> = pressure_pa
            .iter()
            .zip(dewpoint_k)
            .map(|(&p, td)| q_from_dewpoint_k(td, p))
            .collect();

        let (pressure, height, temp, dewpoint, u, v) = build_surface_augmented_ecape_column(
            &pressure_pa[1..],
            &temperature_k[1..],
            &qvapor[1..],
            &height_m[1..],
            &u_wind_ms[1..],
            &v_wind_ms[1..],
            pressure_pa[0],
            temperature_k[0],
            qvapor[0],
            u_wind_ms[0],
            v_wind_ms[0],
            pressure_pa.len() - 1,
            1,
            0,
        );

        let cases = [
            (
                CapeType::SurfaceBased,
                (
                    2011.5445493759416,
                    0.0,
                    2846.0409852115004,
                    -44.991140025487326,
                    1360.0,
                    12220.0,
                ),
            ),
            (
                CapeType::MixedLayer,
                (
                    2115.38982529213,
                    0.0,
                    3040.829940651471,
                    -8.677832569217891,
                    1180.0,
                    12240.0,
                ),
            ),
            (
                CapeType::MostUnstable,
                (
                    2097.6810414544825,
                    0.0,
                    3010.1256185714574,
                    -0.23088078138348503,
                    1100.0,
                    12200.0,
                ),
            ),
        ];

        for (cape_type, expected) in cases {
            let options = ParcelOptions {
                cape_type,
                storm_motion_type: EcapeStormMotionType::UserDefined,
                storm_motion_u_ms: Some(12.0),
                storm_motion_v_ms: Some(6.0),
                ..ParcelOptions::default()
            };

            let result =
                calc_ecape_parcel(&height, &pressure, &temp, &dewpoint, &u, &v, &options).unwrap();
            assert_close(result.ecape_jkg, expected.0);
            assert_close(result.ncape_jkg, expected.1);
            assert_close(result.cape_jkg, expected.2);
            assert_close(result.cin_jkg, expected.3);
            assert_close(result.lfc_m.unwrap_or(0.0), expected.4);
            assert_close(result.el_m.unwrap_or(0.0), expected.5);
        }
    }
}
