use rayon::prelude::*;
use rustwx_core::GridShape;

use crate::ecape::{
    EcapeVolumeInputs, SurfaceInputs, VolumeShape, validate_len, validate_len_or_absent,
};
use crate::error::CalcError;

const ZEROCNK: f64 = 273.15;

#[derive(Debug, Clone, Copy)]
pub struct WindGridInputs<'a> {
    pub shape: VolumeShape,
    pub u_3d_ms: &'a [f64],
    pub v_3d_ms: &'a [f64],
    pub height_agl_3d_m: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct FixedStpInputs<'a> {
    pub grid: GridShape,
    pub sbcape_jkg: &'a [f64],
    pub lcl_m: &'a [f64],
    pub srh_1km_m2s2: &'a [f64],
    pub shear_6km_ms: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct EffectiveStpInputs<'a> {
    pub grid: GridShape,
    pub mlcape_jkg: &'a [f64],
    pub mlcin_jkg: &'a [f64],
    pub ml_lcl_m: &'a [f64],
    pub effective_srh_m2s2: &'a [f64],
    pub effective_bulk_wind_difference_ms: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct EffectiveScpInputs<'a> {
    pub grid: GridShape,
    pub mucape_jkg: &'a [f64],
    pub effective_srh_m2s2: &'a [f64],
    pub effective_bulk_wind_difference_ms: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct EffectiveSevereInputs<'a> {
    pub grid: GridShape,
    pub mlcape_jkg: &'a [f64],
    pub mlcin_jkg: &'a [f64],
    pub ml_lcl_m: &'a [f64],
    pub mucape_jkg: &'a [f64],
    pub effective_srh_m2s2: &'a [f64],
    pub effective_bulk_wind_difference_ms: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct TornadicBetaInputs<'a> {
    pub grid: GridShape,
    pub srh_1km_m2s2: &'a [f64],
    pub mlcape_jkg: &'a [f64],
    pub mlcape_03km_jkg: &'a [f64],
    pub shear_6km_ms: &'a [f64],
    pub ml_lcl_m: &'a [f64],
    pub mlcin_jkg: &'a [f64],
    pub sbcin_jkg: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct VtpModInputs<'a> {
    pub grid: GridShape,
    pub mlcape_jkg: &'a [f64],
    pub effective_srh_m2s2: &'a [f64],
    pub effective_bulk_wind_difference_ms: &'a [f64],
    pub ml_lcl_m: &'a [f64],
    pub mlcin_jkg: &'a [f64],
    pub mlcape_03km_jkg: &'a [f64],
    pub lapse_rate_700_500_cpkm: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct ScpEhiInputs<'a> {
    pub grid: GridShape,
    pub scp_cape_jkg: &'a [f64],
    pub scp_srh_m2s2: &'a [f64],
    pub scp_bulk_wind_difference_ms: &'a [f64],
    pub ehi_cape_jkg: &'a [f64],
    pub ehi_srh_m2s2: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct ShipInputs<'a> {
    pub grid: GridShape,
    pub mucape_jkg: &'a [f64],
    pub shear_6km_ms: &'a [f64],
    pub temperature_500c: &'a [f64],
    pub lapse_rate_700_500_cpkm: &'a [f64],
    pub mixing_ratio_500_gkg: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct BulkRichardsonInputs<'a> {
    pub grid: GridShape,
    pub cape_jkg: &'a [f64],
    pub brn_shear_ms: &'a [f64],
}

#[derive(Debug, Clone, PartialEq)]
pub struct CapeCinOutputs {
    pub cape_jkg: Vec<f64>,
    pub cin_jkg: Vec<f64>,
    pub lcl_m: Vec<f64>,
    pub lfc_m: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveSevereOutputs {
    pub stp_effective: Vec<f64>,
    pub scp_effective: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TornadicBetaOutputs {
    pub tehi: Vec<f64>,
    pub tts: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveLayerDiagnosticsBundle {
    pub effective_srh_m2s2: Vec<f64>,
    pub effective_bulk_wind_difference_ms: Vec<f64>,
    pub effective_inflow_bottom_m: Vec<f64>,
    pub effective_inflow_top_m: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScpEhiOutputs {
    pub scp: Vec<f64>,
    pub ehi: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SupportedSevereFields {
    pub sbcape_jkg: Vec<f64>,
    pub mlcin_jkg: Vec<f64>,
    pub mucape_jkg: Vec<f64>,
    pub srh_01km_m2s2: Vec<f64>,
    pub srh_03km_m2s2: Vec<f64>,
    pub shear_06km_ms: Vec<f64>,
    pub stp_fixed: Vec<f64>,
    pub scp_mu_03km_06km_proxy: Vec<f64>,
    pub ehi_sb_01km_proxy: Vec<f64>,
    pub tehi: Vec<f64>,
    pub tts: Vec<f64>,
    pub vtp_mod: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindDiagnosticsBundle {
    pub srh_01km_m2s2: Vec<f64>,
    pub srh_03km_m2s2: Vec<f64>,
    pub shear_06km_ms: Vec<f64>,
}

pub fn compute_cape_cin(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    parcel_type: &str,
    top_m: Option<f64>,
) -> Result<CapeCinOutputs, CalcError> {
    validate_cape_cin_inputs(grid, volume, surface)?;
    let (cape, cin, lcl, lfc) = if pressure_is_levels(volume) {
        compute_cape_cin_with_pressure_levels(grid, volume, surface, parcel_type, top_m)
    } else {
        metrust::calc::severe::grid::compute_cape_cin(
            volume.pressure_pa,
            volume.temperature_c,
            volume.qvapor_kgkg,
            volume.height_agl_m,
            surface.psfc_pa,
            surface.t2_k,
            surface.q2_kgkg,
            grid.nx,
            grid.ny,
            volume.nz,
            parcel_type,
            top_m,
        )
    };
    Ok(CapeCinOutputs {
        cape_jkg: cape,
        cin_jkg: cin,
        lcl_m: lcl,
        lfc_m: lfc,
    })
}

/// Surface-based, mixed-layer, and most-unstable CAPE/CIN bundles from one
/// shared pass over the columns: each column's profile is extracted and
/// converted once, then lifted with the same `cape_cin_core` kernel the
/// per-parcel entry points dispatch to, once per parcel type. Outputs are
/// bit-identical to three separate `compute_{sb,ml,mu}cape_cin` calls; this
/// exists because the store-ingest derived lane needs all three and the
/// shared column prep + cache locality save a full pass of overhead.
#[derive(Debug, Clone, PartialEq)]
pub struct CapeCinTriplet {
    pub sb: CapeCinOutputs,
    pub ml: CapeCinOutputs,
    pub mu: CapeCinOutputs,
}

pub fn compute_cape_cin_triplet(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    top_m: Option<f64>,
) -> Result<CapeCinTriplet, CalcError> {
    validate_cape_cin_inputs(grid, volume, surface)?;
    if !pressure_is_levels(volume) {
        // Full 3D pressure: no shared per-column prep to reuse here; defer
        // to the per-parcel grid kernels (identical results, one pass each).
        return Ok(CapeCinTriplet {
            sb: compute_cape_cin(grid, volume, surface, "sb", top_m)?,
            ml: compute_cape_cin(grid, volume, surface, "ml", top_m)?,
            mu: compute_cape_cin(grid, volume, surface, "mu", top_m)?,
        });
    }

    let n2d = grid.len();
    let (pressure_levels_hpa, pressure_needs_reverse) = pressure_levels_prep(volume);
    let results = (0..n2d)
        .into_par_iter()
        .map(|ij| {
            let column = PreparedLevelsColumn::build(
                volume,
                surface,
                &pressure_levels_hpa,
                pressure_needs_reverse,
                n2d,
                ij,
            );
            (
                column.cape_cin("sb", top_m),
                column.cape_cin("ml", top_m),
                column.cape_cin("mu", top_m),
            )
        })
        .collect::<Vec<_>>();

    let sb_tuples: Vec<_> = results.par_iter().map(|values| values.0).collect();
    let ml_tuples: Vec<_> = results.par_iter().map(|values| values.1).collect();
    let mu_tuples: Vec<_> = results.par_iter().map(|values| values.2).collect();
    Ok(CapeCinTriplet {
        sb: cape_cin_outputs_from_tuples(&sb_tuples),
        ml: cape_cin_outputs_from_tuples(&ml_tuples),
        mu: cape_cin_outputs_from_tuples(&mu_tuples),
    })
}

pub fn compute_srh(wind: WindGridInputs<'_>, top_m: f64) -> Result<Vec<f64>, CalcError> {
    validate_wind_inputs(wind)?;
    Ok(metrust::calc::severe::grid::compute_srh(
        wind.u_3d_ms,
        wind.v_3d_ms,
        wind.height_agl_3d_m,
        wind.shape.grid.nx,
        wind.shape.grid.ny,
        wind.shape.nz,
        top_m,
    ))
}

pub fn compute_srh_hemispheric(
    wind: WindGridInputs<'_>,
    lat_deg: &[f64],
    top_m: f64,
) -> Result<Vec<f64>, CalcError> {
    validate_wind_inputs(wind)?;
    validate_len("lat_deg", lat_deg.len(), wind.shape.grid.len())?;
    Ok(metrust::calc::severe::grid::compute_srh_hemispheric(
        wind.u_3d_ms,
        wind.v_3d_ms,
        wind.height_agl_3d_m,
        lat_deg,
        wind.shape.grid.nx,
        wind.shape.grid.ny,
        wind.shape.nz,
        top_m,
    ))
}

pub fn compute_shear(
    wind: WindGridInputs<'_>,
    bottom_m: f64,
    top_m: f64,
) -> Result<Vec<f64>, CalcError> {
    validate_wind_inputs(wind)?;
    Ok(metrust::calc::severe::grid::compute_shear(
        wind.u_3d_ms,
        wind.v_3d_ms,
        wind.height_agl_3d_m,
        wind.shape.grid.nx,
        wind.shape.grid.ny,
        wind.shape.nz,
        bottom_m,
        top_m,
    ))
}

pub fn compute_wind_diagnostics_bundle(
    wind: WindGridInputs<'_>,
) -> Result<WindDiagnosticsBundle, CalcError> {
    validate_wind_inputs(wind)?;
    let diagnostics = metrust::calc::severe::grid::compute_wind_diagnostics_bundle(
        wind.u_3d_ms,
        wind.v_3d_ms,
        wind.height_agl_3d_m,
        wind.shape.grid.nx,
        wind.shape.grid.ny,
        wind.shape.nz,
    );
    Ok(WindDiagnosticsBundle {
        srh_01km_m2s2: diagnostics.srh_01km_m2s2,
        srh_03km_m2s2: diagnostics.srh_03km_m2s2,
        shear_06km_ms: diagnostics.shear_06km_ms,
    })
}

pub fn compute_effective_layer_diagnostics(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    lat_deg: Option<&[f64]>,
) -> Result<EffectiveLayerDiagnosticsBundle, CalcError> {
    validate_severe_inputs(grid, volume, surface)?;
    if let Some(lat_deg) = lat_deg {
        validate_len("lat_deg", lat_deg.len(), grid.len())?;
    }

    let n2d = grid.len();
    let pressure_levels_hpa = if pressure_is_levels(volume) {
        Some(
            volume
                .pressure_pa
                .iter()
                .map(|pressure_pa| *pressure_pa / 100.0)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let results = (0..n2d)
        .into_par_iter()
        .map(|ij| {
            let column = build_effective_layer_column(
                volume,
                surface,
                n2d,
                ij,
                pressure_levels_hpa.as_deref(),
            );
            effective_layer_for_column(&column, lat_deg.map(|lat| lat[ij]))
        })
        .collect::<Vec<_>>();

    let mut effective_srh_m2s2 = Vec::with_capacity(n2d);
    let mut effective_bulk_wind_difference_ms = Vec::with_capacity(n2d);
    let mut effective_inflow_bottom_m = Vec::with_capacity(n2d);
    let mut effective_inflow_top_m = Vec::with_capacity(n2d);
    for result in results {
        effective_srh_m2s2.push(result.effective_srh_m2s2);
        effective_bulk_wind_difference_ms.push(result.effective_bulk_wind_difference_ms);
        effective_inflow_bottom_m.push(result.effective_inflow_bottom_m);
        effective_inflow_top_m.push(result.effective_inflow_top_m);
    }

    Ok(EffectiveLayerDiagnosticsBundle {
        effective_srh_m2s2,
        effective_bulk_wind_difference_ms,
        effective_inflow_bottom_m,
        effective_inflow_top_m,
    })
}

/// Compute fixed-layer STP from precomputed surface-based CAPE, LCL, 0-1 km SRH,
/// and 0-6 km bulk shear grids.
///
/// This follows the operational Thompson-style gates used in the local
/// `wrf-rust-plots` implementation: LCL is capped at 1.0 for values at or below
/// 1000 m, shear is zeroed below 12.5 m/s, and the shear term is capped at 1.5
/// once 0-6 km shear reaches 30 m/s.
pub fn compute_stp_fixed(inputs: FixedStpInputs<'_>) -> Result<Vec<f64>, CalcError> {
    validate_fixed_stp_inputs(inputs)?;
    Ok(inputs
        .sbcape_jkg
        .iter()
        .zip(inputs.lcl_m.iter())
        .zip(inputs.srh_1km_m2s2.iter())
        .zip(inputs.shear_6km_ms.iter())
        .map(|(((cape, lcl), srh), shear)| fixed_stp_value(*cape, *lcl, *srh, *shear))
        .collect())
}

/// Compatibility wrapper for fixed-layer STP.
pub fn compute_stp(
    grid: GridShape,
    sbcape_jkg: &[f64],
    lcl_m: &[f64],
    srh_1km_m2s2: &[f64],
    shear_6km_ms: &[f64],
) -> Result<Vec<f64>, CalcError> {
    compute_stp_fixed(FixedStpInputs {
        grid,
        sbcape_jkg,
        lcl_m,
        srh_1km_m2s2,
        shear_6km_ms,
    })
}

/// Compute effective-layer STP from precomputed mixed-layer parcel and
/// effective-layer kinematic ingredient grids.
///
/// This function intentionally does not derive the effective inflow layer. Callers
/// must provide mixed-layer CAPE/CIN/LCL together with effective SRH and
/// effective bulk wind difference from a profile-aware workflow.
pub fn compute_stp_effective(inputs: EffectiveStpInputs<'_>) -> Result<Vec<f64>, CalcError> {
    validate_effective_stp_inputs(inputs)?;
    Ok(inputs
        .mlcape_jkg
        .iter()
        .zip(inputs.mlcin_jkg.iter())
        .zip(inputs.ml_lcl_m.iter())
        .zip(inputs.effective_srh_m2s2.iter())
        .zip(inputs.effective_bulk_wind_difference_ms.iter())
        .map(|((((cape, cin), lcl), srh), ebwd)| {
            effective_stp_value(*cape, *cin, *lcl, *srh, *ebwd)
        })
        .collect())
}

/// Compute effective-layer STP and SCP together from shared effective-layer
/// kinematic inputs.
///
/// This is intended for callers that already cache effective SRH and effective
/// bulk wind difference upstream and want both high-value effective composites
/// in a single validation and loop pass. Effective inflow-layer derivation and
/// parcel extraction remain upstream/profile-aware responsibilities.
pub fn compute_effective_severe(
    inputs: EffectiveSevereInputs<'_>,
) -> Result<EffectiveSevereOutputs, CalcError> {
    validate_effective_severe_inputs(inputs)?;

    let n = inputs.grid.len();
    let mut stp_effective = Vec::with_capacity(n);
    let mut scp_effective = Vec::with_capacity(n);

    for idx in 0..n {
        let effective_srh = inputs.effective_srh_m2s2[idx];
        let effective_bulk_wind_difference = inputs.effective_bulk_wind_difference_ms[idx];
        stp_effective.push(effective_stp_value(
            inputs.mlcape_jkg[idx],
            inputs.mlcin_jkg[idx],
            inputs.ml_lcl_m[idx],
            effective_srh,
            effective_bulk_wind_difference,
        ));
        scp_effective.push(scp_effective_value(
            inputs.mucape_jkg[idx],
            effective_srh,
            effective_bulk_wind_difference,
        ));
    }

    Ok(EffectiveSevereOutputs {
        stp_effective,
        scp_effective,
    })
}

pub fn compute_ehi(grid: GridShape, cape_jkg: &[f64], srh: &[f64]) -> Result<Vec<f64>, CalcError> {
    validate_grid_fields(grid, &[("cape_jkg", cape_jkg), ("srh", srh)])?;
    Ok(cape_jkg
        .iter()
        .zip(srh.iter())
        .map(|(cape, srh)| ehi_value(*cape, *srh))
        .collect())
}

/// Compute effective-layer SCP from precomputed most-unstable CAPE, effective
/// SRH, and effective bulk wind difference grids.
///
/// This mirrors the local `wrf-rust-plots` gridded SCP behavior. The effective
/// bulk wind difference term is zero below 10 m/s and capped at 1.0 once EBWD
/// reaches 20 m/s.
pub fn compute_scp_effective(inputs: EffectiveScpInputs<'_>) -> Result<Vec<f64>, CalcError> {
    validate_effective_scp_inputs(inputs)?;
    Ok(inputs
        .mucape_jkg
        .iter()
        .zip(inputs.effective_srh_m2s2.iter())
        .zip(inputs.effective_bulk_wind_difference_ms.iter())
        .map(|((cape, srh), ebwd)| scp_effective_value(*cape, *srh, *ebwd))
        .collect())
}

/// Compatibility wrapper for effective-layer SCP ingredients.
pub fn compute_scp(
    grid: GridShape,
    mucape_jkg: &[f64],
    effective_srh_m2s2: &[f64],
    effective_bulk_wind_difference_ms: &[f64],
) -> Result<Vec<f64>, CalcError> {
    compute_scp_effective(EffectiveScpInputs {
        grid,
        mucape_jkg,
        effective_srh_m2s2,
        effective_bulk_wind_difference_ms,
    })
}

/// Compute SPC beta tornado composites from precomputed ingredient grids.
///
/// `TEHI` is Tornadic 0-1 km EHI and `TTS` is Tornadic Tilting and Stretching;
/// this is not the Total Totals index. The inputs are model-agnostic, so HRRR,
/// RAP, GFS, WRF, or any other source can use this once the required parcel and
/// kinematic ingredients have been derived upstream.
pub fn compute_tornadic_beta(
    inputs: TornadicBetaInputs<'_>,
) -> Result<TornadicBetaOutputs, CalcError> {
    validate_tornadic_beta_inputs(inputs)?;

    let n = inputs.grid.len();
    let pairs = (0..n)
        .into_par_iter()
        .map(|idx| {
            (
                tehi_value(
                    inputs.srh_1km_m2s2[idx],
                    inputs.mlcape_jkg[idx],
                    inputs.mlcape_03km_jkg[idx],
                    inputs.shear_6km_ms[idx],
                    inputs.ml_lcl_m[idx],
                    inputs.mlcin_jkg[idx],
                    inputs.sbcin_jkg[idx],
                ),
                tts_value(
                    inputs.srh_1km_m2s2[idx],
                    inputs.mlcape_03km_jkg[idx],
                    inputs.mlcape_jkg[idx],
                    inputs.shear_6km_ms[idx],
                    inputs.ml_lcl_m[idx],
                    inputs.mlcin_jkg[idx],
                    inputs.sbcin_jkg[idx],
                ),
            )
        })
        .collect::<Vec<_>>();

    let mut tehi = Vec::with_capacity(n);
    let mut tts = Vec::with_capacity(n);
    for (tehi_value, tts_value) in pairs {
        tehi.push(tehi_value);
        tts.push(tts_value);
    }

    Ok(TornadicBetaOutputs { tehi, tts })
}

/// Compute modified Violent Tornado Parameter from precomputed ingredient
/// grids.
///
/// This mirrors the local `wrf-rust` `vtp_mod` math but is not tied to WRF
/// files. Effective SRH/BWD, MLCAPE/CIN/LCL, 0-3 km MLCAPE, and 700-500 hPa
/// lapse rate should all come from the same model/sample definition.
pub fn compute_vtp_mod(inputs: VtpModInputs<'_>) -> Result<Vec<f64>, CalcError> {
    validate_vtp_mod_inputs(inputs)?;
    Ok((0..inputs.grid.len())
        .into_par_iter()
        .map(|idx| {
            vtp_mod_value(
                inputs.mlcape_jkg[idx],
                inputs.effective_srh_m2s2[idx],
                inputs.effective_bulk_wind_difference_ms[idx],
                inputs.ml_lcl_m[idx],
                inputs.mlcin_jkg[idx],
                inputs.mlcape_03km_jkg[idx],
                inputs.lapse_rate_700_500_cpkm[idx],
            )
        })
        .collect())
}

/// Compute SCP and EHI together from precomputed grids.
///
/// This helper is intentionally agnostic about parcel type and SRH depth. It is
/// useful for proof and render flows that already cache CAPE, SRH, and bulk-wind
/// grids once and want paired SCP/EHI outputs without repeated validation or
/// call-site wiring.
pub fn compute_scp_ehi(inputs: ScpEhiInputs<'_>) -> Result<ScpEhiOutputs, CalcError> {
    validate_scp_ehi_inputs(inputs)?;

    let n = inputs.grid.len();
    let mut scp = Vec::with_capacity(n);
    let mut ehi = Vec::with_capacity(n);

    for idx in 0..n {
        scp.push(scp_effective_value(
            inputs.scp_cape_jkg[idx],
            inputs.scp_srh_m2s2[idx],
            inputs.scp_bulk_wind_difference_ms[idx],
        ));
        ehi.push(ehi_value(
            inputs.ehi_cape_jkg[idx],
            inputs.ehi_srh_m2s2[idx],
        ));
    }

    Ok(ScpEhiOutputs { scp, ehi })
}

/// Compute the current local `wrf-rust` SHIP-style hail proxy from
/// precomputed most-unstable parcel, 500 hPa, and 700-500 hPa ingredient
/// grids.
///
/// This mirrors the local `wrf-rust` component math, including the SPC-style
/// reduction when MUCAPE is below 1300 J/kg. It intentionally does not derive
/// the 500 hPa temperature/mixing ratio or the 700-500 hPa lapse rate from
/// profiles; callers must provide those upstream. This should not be treated
/// as a canonical SHARPpy-style SHIP implementation yet.
pub fn compute_ship(inputs: ShipInputs<'_>) -> Result<Vec<f64>, CalcError> {
    validate_ship_inputs(inputs)?;
    Ok((0..inputs.grid.len())
        .map(|idx| {
            ship_value(
                inputs.mucape_jkg[idx],
                inputs.shear_6km_ms[idx],
                inputs.temperature_500c[idx],
                inputs.lapse_rate_700_500_cpkm[idx],
                inputs.mixing_ratio_500_gkg[idx],
            )
        })
        .collect())
}

/// Compute Bulk Richardson Number Index (BRI) from CAPE and BRN-shear grids.
///
/// The `brn_shear_ms` input must be the BRN-shear magnitude used by the local
/// `wrf-rust` product: the vector difference between the 0-500 m mean wind and
/// the 0-6 km mean wind. This is not interchangeable with plain 0-6 km bulk
/// shear. Degenerate denominators are zero-filled to match local gridded
/// behavior.
pub fn compute_bri(inputs: BulkRichardsonInputs<'_>) -> Result<Vec<f64>, CalcError> {
    validate_bulk_richardson_inputs(inputs)?;
    Ok((0..inputs.grid.len())
        .map(|idx| bri_value(inputs.cape_jkg[idx], inputs.brn_shear_ms[idx]))
        .collect())
}

/// Compute the currently supported gridded severe bundle.
///
/// This bundle is intentionally conservative:
/// - `stp_fixed` uses the fixed-layer Thompson-style formula with `sbCAPE`,
///   `sbLCL`, `0-1 km SRH`, and `0-6 km bulk shear`
/// - `scp_mu_03km_06km_proxy` uses `muCAPE` with `0-3 km SRH` and `0-6 km bulk
///   shear` through the existing SCP wrapper, but is still a fixed-depth proxy,
///   not an effective-layer SCP
/// - `ehi_sb_01km_proxy` uses `sbCAPE` with `0-1 km SRH`
/// - `vtp_mod` uses the effective inflow layer derived from each grid column,
///   Bunkers storm motion, effective-layer SRH/BWD, 0-3 km MLCAPE, and the
///   700-500 hPa lapse rate.
///
/// The fixed-depth SCP/EHI products keep their explicit proxy names so they are
/// not confused with the effective-layer VTP mod calculation.
pub fn compute_supported_severe_fields(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
) -> Result<SupportedSevereFields, CalcError> {
    compute_supported_severe_fields_impl(grid, volume, surface, None)
}

pub fn compute_supported_severe_fields_hemispheric(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    lat_deg: &[f64],
) -> Result<SupportedSevereFields, CalcError> {
    validate_len("lat_deg", lat_deg.len(), grid.len())?;
    compute_supported_severe_fields_impl(grid, volume, surface, Some(lat_deg))
}

fn compute_supported_severe_fields_impl(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    lat_deg: Option<&[f64]>,
) -> Result<SupportedSevereFields, CalcError> {
    validate_severe_inputs(grid, volume, surface)?;

    let sb = compute_cape_cin(grid, volume, surface, "sb", None)?;
    let ml = compute_cape_cin(grid, volume, surface, "ml", None)?;
    let ml_03km = compute_cape_cin(grid, volume, surface, "ml", Some(3000.0))?;
    let mu = compute_cape_cin(grid, volume, surface, "mu", None)?;

    let wind = WindGridInputs {
        shape: VolumeShape::new(grid, volume.nz)?,
        u_3d_ms: volume.u_ms,
        v_3d_ms: volume.v_ms,
        height_agl_3d_m: volume.height_agl_m,
    };
    let wind_diagnostics = if let Some(lat_deg) = lat_deg {
        WindDiagnosticsBundle {
            srh_01km_m2s2: compute_srh_hemispheric(wind, lat_deg, 1000.0)?,
            srh_03km_m2s2: compute_srh_hemispheric(wind, lat_deg, 3000.0)?,
            shear_06km_ms: compute_shear(wind, 0.0, 6000.0)?,
        }
    } else {
        compute_wind_diagnostics_bundle(wind)?
    };
    let stp_fixed = compute_stp_fixed(FixedStpInputs {
        grid,
        sbcape_jkg: &sb.cape_jkg,
        lcl_m: &sb.lcl_m,
        srh_1km_m2s2: &wind_diagnostics.srh_01km_m2s2,
        shear_6km_ms: &wind_diagnostics.shear_06km_ms,
    })?;
    let scp_ehi = compute_scp_ehi(ScpEhiInputs {
        grid,
        scp_cape_jkg: &mu.cape_jkg,
        scp_srh_m2s2: &wind_diagnostics.srh_03km_m2s2,
        scp_bulk_wind_difference_ms: &wind_diagnostics.shear_06km_ms,
        ehi_cape_jkg: &sb.cape_jkg,
        ehi_srh_m2s2: &wind_diagnostics.srh_01km_m2s2,
    })?;
    let beta = compute_tornadic_beta(TornadicBetaInputs {
        grid,
        srh_1km_m2s2: &wind_diagnostics.srh_01km_m2s2,
        mlcape_jkg: &ml.cape_jkg,
        mlcape_03km_jkg: &ml_03km.cape_jkg,
        shear_6km_ms: &wind_diagnostics.shear_06km_ms,
        ml_lcl_m: &ml.lcl_m,
        mlcin_jkg: &ml.cin_jkg,
        sbcin_jkg: &sb.cin_jkg,
    })?;
    let effective_layer = compute_effective_layer_diagnostics(grid, volume, surface, lat_deg)?;
    let lapse_rate_700_500 = lapse_rate_700_500_for_supported(grid, volume)?;
    let vtp_mod = compute_vtp_mod(VtpModInputs {
        grid,
        mlcape_jkg: &ml.cape_jkg,
        effective_srh_m2s2: &effective_layer.effective_srh_m2s2,
        effective_bulk_wind_difference_ms: &effective_layer.effective_bulk_wind_difference_ms,
        ml_lcl_m: &ml.lcl_m,
        mlcin_jkg: &ml.cin_jkg,
        mlcape_03km_jkg: &ml_03km.cape_jkg,
        lapse_rate_700_500_cpkm: &lapse_rate_700_500,
    })?;

    Ok(SupportedSevereFields {
        sbcape_jkg: sb.cape_jkg,
        mlcin_jkg: ml.cin_jkg,
        mucape_jkg: mu.cape_jkg,
        srh_01km_m2s2: wind_diagnostics.srh_01km_m2s2,
        srh_03km_m2s2: wind_diagnostics.srh_03km_m2s2,
        shear_06km_ms: wind_diagnostics.shear_06km_ms,
        stp_fixed,
        scp_mu_03km_06km_proxy: scp_ehi.scp,
        ehi_sb_01km_proxy: scp_ehi.ehi,
        tehi: beta.tehi,
        tts: beta.tts,
        vtp_mod,
    })
}

pub use metrust::calc::severe::critical_angle;
pub use metrust::calc::severe::significant_tornado_parameter;
pub use metrust::calc::severe::supercell_composite_parameter;

fn validate_severe_inputs(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
) -> Result<(), CalcError> {
    let n2d = grid.len();
    let n3d = n2d * volume.nz;
    if pressure_is_levels(volume) {
        validate_len("pressure_levels_pa", volume.pressure_pa.len(), volume.nz)?;
    } else {
        validate_len("pressure_pa", volume.pressure_pa.len(), n3d)?;
    }
    validate_len("temperature_c", volume.temperature_c.len(), n3d)?;
    validate_len("qvapor_kgkg", volume.qvapor_kgkg.len(), n3d)?;
    validate_len("height_agl_m", volume.height_agl_m.len(), n3d)?;
    validate_len("u_ms", volume.u_ms.len(), n3d)?;
    validate_len("v_ms", volume.v_ms.len(), n3d)?;
    validate_len("psfc_pa", surface.psfc_pa.len(), n2d)?;
    validate_len("t2_k", surface.t2_k.len(), n2d)?;
    validate_len("q2_kgkg", surface.q2_kgkg.len(), n2d)?;
    validate_len("u10_ms", surface.u10_ms.len(), n2d)?;
    validate_len("v10_ms", surface.v10_ms.len(), n2d)?;
    Ok(())
}

/// [`validate_severe_inputs`] for the pure CAPE/CIN entry points
/// (`compute_cape_cin`, `compute_cape_cin_triplet`), which never read the
/// wind volumes: winds may be absent (empty) so the store-ingest derived
/// lane can free them before the long parcel pass. Wrong-length non-empty
/// winds are still rejected.
fn validate_cape_cin_inputs(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
) -> Result<(), CalcError> {
    let n2d = grid.len();
    let n3d = n2d * volume.nz;
    if pressure_is_levels(volume) {
        validate_len("pressure_levels_pa", volume.pressure_pa.len(), volume.nz)?;
    } else {
        validate_len("pressure_pa", volume.pressure_pa.len(), n3d)?;
    }
    validate_len("temperature_c", volume.temperature_c.len(), n3d)?;
    validate_len("qvapor_kgkg", volume.qvapor_kgkg.len(), n3d)?;
    validate_len("height_agl_m", volume.height_agl_m.len(), n3d)?;
    validate_len_or_absent("u_ms", volume.u_ms.len(), n3d)?;
    validate_len_or_absent("v_ms", volume.v_ms.len(), n3d)?;
    validate_len("psfc_pa", surface.psfc_pa.len(), n2d)?;
    validate_len("t2_k", surface.t2_k.len(), n2d)?;
    validate_len("q2_kgkg", surface.q2_kgkg.len(), n2d)?;
    validate_len("u10_ms", surface.u10_ms.len(), n2d)?;
    validate_len("v10_ms", surface.v10_ms.len(), n2d)?;
    Ok(())
}

fn pressure_is_levels(volume: EcapeVolumeInputs<'_>) -> bool {
    volume.pressure_pa.len() == volume.nz
}

fn extract_column(data: &[f64], nz: usize, n2d: usize, ij: usize) -> Vec<f64> {
    (0..nz).map(|k| data[k * n2d + ij]).collect()
}

fn dewpoint_from_q(q_kgkg: f64, pressure_hpa: f64) -> f64 {
    let q = q_kgkg.max(1.0e-10);
    let e_hpa = (q * pressure_hpa / (0.622 + q)).max(1.0e-10);
    let ln_e = (e_hpa / 6.112).ln();
    (243.5 * ln_e) / (17.67 - ln_e)
}

/// One grid column's profile, prepared exactly as the per-parcel CAPE path
/// always has: pressure levels to hPa, columns extracted, dewpoint from Q,
/// everything reversed to surface-first when the levels arrive top-first.
struct PreparedLevelsColumn {
    pressure_hpa: Vec<f64>,
    temperature_c: Vec<f64>,
    dewpoint_c: Vec<f64>,
    height_agl_m: Vec<f64>,
    psfc_hpa: f64,
    t2m_c: f64,
    td2m_c: f64,
}

impl PreparedLevelsColumn {
    fn build(
        volume: EcapeVolumeInputs<'_>,
        surface: SurfaceInputs<'_>,
        pressure_levels_hpa: &[f64],
        pressure_needs_reverse: bool,
        n2d: usize,
        ij: usize,
    ) -> Self {
        let mut pressure_hpa = pressure_levels_hpa.to_vec();
        let mut temperature_c = extract_column(volume.temperature_c, volume.nz, n2d, ij);
        let mut height_agl_m = extract_column(volume.height_agl_m, volume.nz, n2d, ij);
        let qvapor_kgkg = extract_column(volume.qvapor_kgkg, volume.nz, n2d, ij);
        let mut dewpoint_c = pressure_hpa
            .iter()
            .enumerate()
            .map(|(level, &pressure_hpa)| dewpoint_from_q(qvapor_kgkg[level], pressure_hpa))
            .collect::<Vec<_>>();

        if pressure_needs_reverse {
            pressure_hpa.reverse();
            temperature_c.reverse();
            dewpoint_c.reverse();
            height_agl_m.reverse();
        }

        let psfc_hpa = surface.psfc_pa[ij] / 100.0;
        PreparedLevelsColumn {
            pressure_hpa,
            temperature_c,
            dewpoint_c,
            height_agl_m,
            psfc_hpa,
            t2m_c: surface.t2_k[ij] - ZEROCNK,
            td2m_c: dewpoint_from_q(surface.q2_kgkg[ij], psfc_hpa),
        }
    }

    fn cape_cin(&self, parcel_type: &str, top_m: Option<f64>) -> (f64, f64, f64, f64) {
        metrust::calc::thermo::cape_cin_core(
            &self.pressure_hpa,
            &self.temperature_c,
            &self.dewpoint_c,
            &self.height_agl_m,
            self.psfc_hpa,
            self.t2m_c,
            self.td2m_c,
            parcel_type,
            100.0,
            300.0,
            top_m,
        )
    }
}

fn pressure_levels_prep(volume: EcapeVolumeInputs<'_>) -> (Vec<f64>, bool) {
    let pressure_levels_hpa = volume
        .pressure_pa
        .iter()
        .map(|value| *value / 100.0)
        .collect::<Vec<_>>();
    let pressure_needs_reverse = pressure_levels_hpa.len() > 1
        && pressure_levels_hpa[0] < pressure_levels_hpa[pressure_levels_hpa.len() - 1];
    (pressure_levels_hpa, pressure_needs_reverse)
}

/// Unzip per-column `(cape, cin, lcl, lfc)` tuples into the output bundle,
/// each plane in parallel.
fn cape_cin_outputs_from_tuples(results: &[(f64, f64, f64, f64)]) -> CapeCinOutputs {
    CapeCinOutputs {
        cape_jkg: results.par_iter().map(|values| values.0).collect(),
        cin_jkg: results.par_iter().map(|values| values.1).collect(),
        lcl_m: results.par_iter().map(|values| values.2).collect(),
        lfc_m: results.par_iter().map(|values| values.3).collect(),
    }
}

fn compute_cape_cin_with_pressure_levels(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    parcel_type: &str,
    top_m: Option<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n2d = grid.len();
    let (pressure_levels_hpa, pressure_needs_reverse) = pressure_levels_prep(volume);
    let results = (0..n2d)
        .into_par_iter()
        .map(|ij| {
            PreparedLevelsColumn::build(
                volume,
                surface,
                &pressure_levels_hpa,
                pressure_needs_reverse,
                n2d,
                ij,
            )
            .cape_cin(parcel_type, top_m)
        })
        .collect::<Vec<_>>();

    let outputs = cape_cin_outputs_from_tuples(&results);
    (
        outputs.cape_jkg,
        outputs.cin_jkg,
        outputs.lcl_m,
        outputs.lfc_m,
    )
}

#[derive(Debug, Clone)]
struct EffectiveLayerColumn {
    pressure_hpa: Vec<f64>,
    temperature_c: Vec<f64>,
    dewpoint_c: Vec<f64>,
    height_agl_m: Vec<f64>,
    u_ms: Vec<f64>,
    v_ms: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct EffectiveLayerColumnResult {
    effective_srh_m2s2: f64,
    effective_bulk_wind_difference_ms: f64,
    effective_inflow_bottom_m: f64,
    effective_inflow_top_m: f64,
}

fn build_effective_layer_column(
    volume: EcapeVolumeInputs<'_>,
    surface: SurfaceInputs<'_>,
    n2d: usize,
    ij: usize,
    pressure_levels_hpa: Option<&[f64]>,
) -> EffectiveLayerColumn {
    let mut rows = Vec::with_capacity(volume.nz + 1);
    let psfc_hpa = surface.psfc_pa[ij] / 100.0;
    rows.push((
        psfc_hpa,
        surface.t2_k[ij] - ZEROCNK,
        dewpoint_from_q(surface.q2_kgkg[ij], psfc_hpa),
        0.0,
        surface.u10_ms[ij],
        surface.v10_ms[ij],
    ));

    for k in 0..volume.nz {
        let idx = k * n2d + ij;
        let pressure_hpa = match pressure_levels_hpa {
            Some(levels) => levels[k],
            None => volume.pressure_pa[idx] / 100.0,
        };
        rows.push((
            pressure_hpa,
            volume.temperature_c[idx],
            dewpoint_from_q(volume.qvapor_kgkg[idx], pressure_hpa),
            volume.height_agl_m[idx],
            volume.u_ms[idx],
            volume.v_ms[idx],
        ));
    }

    rows.retain(|(p, t, td, h, u, v)| {
        p.is_finite()
            && *p > 0.0
            && t.is_finite()
            && td.is_finite()
            && h.is_finite()
            && *h >= 0.0
            && u.is_finite()
            && v.is_finite()
    });
    rows.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    rows.dedup_by(|a, b| (a.3 - b.3).abs() < 1.0 && (a.0 - b.0).abs() < 0.1);

    let mut pressure_hpa = Vec::with_capacity(rows.len());
    let mut temperature_c = Vec::with_capacity(rows.len());
    let mut dewpoint_c = Vec::with_capacity(rows.len());
    let mut height_agl_m = Vec::with_capacity(rows.len());
    let mut u_ms = Vec::with_capacity(rows.len());
    let mut v_ms = Vec::with_capacity(rows.len());
    for (p, t, td, h, u, v) in rows {
        pressure_hpa.push(p);
        temperature_c.push(t);
        dewpoint_c.push(td.min(t));
        height_agl_m.push(h);
        u_ms.push(u);
        v_ms.push(v);
    }

    EffectiveLayerColumn {
        pressure_hpa,
        temperature_c,
        dewpoint_c,
        height_agl_m,
        u_ms,
        v_ms,
    }
}

fn effective_layer_for_column(
    column: &EffectiveLayerColumn,
    lat_deg: Option<f64>,
) -> EffectiveLayerColumnResult {
    const ECAPE_THRESHOLD: f64 = 100.0;
    const ECIN_THRESHOLD: f64 = -250.0;

    let missing = EffectiveLayerColumnResult {
        effective_srh_m2s2: f64::NAN,
        effective_bulk_wind_difference_ms: f64::NAN,
        effective_inflow_bottom_m: f64::NAN,
        effective_inflow_top_m: f64::NAN,
    };
    let n = column.pressure_hpa.len();
    if n < 3 {
        return missing;
    }

    let mut bottom_idx = None;
    let mut top_idx = None;
    for idx in 0..n {
        let Some((cape, cin)) = parcel_cape_cin_from_level(column, idx) else {
            continue;
        };
        let effective = cape >= ECAPE_THRESHOLD && cin > ECIN_THRESHOLD;
        if effective {
            if bottom_idx.is_none() {
                bottom_idx = Some(idx);
            }
            top_idx = Some(idx);
        } else if bottom_idx.is_some() {
            break;
        }
    }

    let (Some(bottom_idx), Some(top_idx)) = (bottom_idx, top_idx) else {
        return missing;
    };
    if top_idx <= bottom_idx {
        return missing;
    }

    let bottom_m = column.height_agl_m[bottom_idx];
    let top_m = column.height_agl_m[top_idx];
    if !bottom_m.is_finite() || !top_m.is_finite() || top_m <= bottom_m {
        return missing;
    }

    let pressure_hpa = &column.pressure_hpa;
    let heights = &column.height_agl_m;
    let u = &column.u_ms;
    let v = &column.v_ms;
    let (rm, lm, _) = metrust::calc::wind::bunkers_storm_motion(pressure_hpa, u, v, heights);
    let southern = lat_deg.is_some_and(|lat| lat.is_finite() && lat < 0.0);
    let storm_motion = if southern { lm } else { rm };
    let mut srh = srh_between_heights(
        heights,
        u,
        v,
        bottom_m,
        top_m,
        storm_motion.0,
        storm_motion.1,
    );
    if southern {
        srh = -srh;
    }

    let Some((u_bot, v_bot)) = interp_wind_at_height(heights, u, v, bottom_m) else {
        return missing;
    };
    let Some((u_top, v_top)) = interp_wind_at_height(heights, u, v, top_m) else {
        return missing;
    };
    let bwd = ((u_top - u_bot).powi(2) + (v_top - v_bot).powi(2)).sqrt();

    EffectiveLayerColumnResult {
        effective_srh_m2s2: srh,
        effective_bulk_wind_difference_ms: bwd,
        effective_inflow_bottom_m: bottom_m,
        effective_inflow_top_m: top_m,
    }
}

fn parcel_cape_cin_from_level(
    column: &EffectiveLayerColumn,
    start_idx: usize,
) -> Option<(f64, f64)> {
    if start_idx + 2 >= column.pressure_hpa.len() {
        return None;
    }

    let p_start = column.pressure_hpa[start_idx];
    let t_start = column.temperature_c[start_idx];
    let td_start = column.dewpoint_c[start_idx].min(t_start);
    let h_start = column.height_agl_m[start_idx];
    if !p_start.is_finite() || !t_start.is_finite() || !td_start.is_finite() || !h_start.is_finite()
    {
        return None;
    }

    let mut pressure_hpa = Vec::new();
    let mut temperature_c = Vec::new();
    let mut dewpoint_c = Vec::new();
    let mut height_agl_m = Vec::new();
    for idx in (start_idx + 1)..column.pressure_hpa.len() {
        let height = column.height_agl_m[idx] - h_start;
        if height <= 0.0 {
            continue;
        }
        pressure_hpa.push(column.pressure_hpa[idx]);
        temperature_c.push(column.temperature_c[idx]);
        dewpoint_c.push(column.dewpoint_c[idx].min(column.temperature_c[idx]));
        height_agl_m.push(height);
    }
    if pressure_hpa.len() < 2 {
        return None;
    }

    let (cape, cin, _, _) = metrust::calc::thermo::cape_cin_core(
        &pressure_hpa,
        &temperature_c,
        &dewpoint_c,
        &height_agl_m,
        p_start,
        t_start,
        td_start,
        "sb",
        100.0,
        300.0,
        None,
    );
    Some((cape, cin))
}

fn interp_wind_at_height(
    heights: &[f64],
    u: &[f64],
    v: &[f64],
    target_m: f64,
) -> Option<(f64, f64)> {
    Some((
        interp_scalar_at_height(heights, u, target_m)?,
        interp_scalar_at_height(heights, v, target_m)?,
    ))
}

fn interp_scalar_at_height(heights: &[f64], values: &[f64], target_m: f64) -> Option<f64> {
    if heights.len() != values.len() || heights.is_empty() || !target_m.is_finite() {
        return None;
    }
    if target_m < heights[0] || target_m > heights[heights.len() - 1] {
        return None;
    }
    if (target_m - heights[0]).abs() < 1.0e-6 {
        return Some(values[0]);
    }
    for idx in 1..heights.len() {
        if target_m <= heights[idx] {
            let dz = heights[idx] - heights[idx - 1];
            if dz.abs() < 1.0e-9 {
                return Some(values[idx]);
            }
            let weight = (target_m - heights[idx - 1]) / dz;
            return Some(values[idx - 1] + weight * (values[idx] - values[idx - 1]));
        }
    }
    Some(values[values.len() - 1])
}

fn srh_between_heights(
    heights: &[f64],
    u: &[f64],
    v: &[f64],
    bottom_m: f64,
    top_m: f64,
    storm_u: f64,
    storm_v: f64,
) -> f64 {
    let Some((mut prev_u, mut prev_v)) = interp_wind_at_height(heights, u, v, bottom_m) else {
        return f64::NAN;
    };
    let mut srh = 0.0;
    let mut prev_h = bottom_m;

    for idx in 0..heights.len() {
        let h = heights[idx];
        if h <= bottom_m {
            continue;
        }
        if h >= top_m {
            break;
        }
        if h <= prev_h {
            continue;
        }
        let next_u = u[idx];
        let next_v = v[idx];
        srh += srh_segment(prev_u, prev_v, next_u, next_v, storm_u, storm_v);
        prev_u = next_u;
        prev_v = next_v;
        prev_h = h;
    }

    if top_m > prev_h {
        if let Some((top_u, top_v)) = interp_wind_at_height(heights, u, v, top_m) {
            srh += srh_segment(prev_u, prev_v, top_u, top_v, storm_u, storm_v);
        }
    }

    srh
}

fn srh_segment(u0: f64, v0: f64, u1: f64, v1: f64, storm_u: f64, storm_v: f64) -> f64 {
    let sru0 = u0 - storm_u;
    let srv0 = v0 - storm_v;
    let sru1 = u1 - storm_u;
    let srv1 = v1 - storm_v;
    sru1 * srv0 - sru0 * srv1
}

fn lapse_rate_700_500_for_supported(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
) -> Result<Vec<f64>, CalcError> {
    validate_severe_volume_inputs(grid, volume)?;
    let n2d = grid.len();
    let pressure_levels_hpa = if pressure_is_levels(volume) {
        Some(
            volume
                .pressure_pa
                .iter()
                .map(|pressure_pa| *pressure_pa / 100.0)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    Ok((0..n2d)
        .into_par_iter()
        .map(|ij| lapse_rate_700_500_column(volume, n2d, ij, pressure_levels_hpa.as_deref()))
        .collect())
}

fn validate_severe_volume_inputs(
    grid: GridShape,
    volume: EcapeVolumeInputs<'_>,
) -> Result<(), CalcError> {
    let n3d = grid.len() * volume.nz;
    if pressure_is_levels(volume) {
        validate_len("pressure_levels_pa", volume.pressure_pa.len(), volume.nz)?;
    } else {
        validate_len("pressure_pa", volume.pressure_pa.len(), n3d)?;
    }
    validate_len("temperature_c", volume.temperature_c.len(), n3d)?;
    validate_len("qvapor_kgkg", volume.qvapor_kgkg.len(), n3d)?;
    validate_len("height_agl_m", volume.height_agl_m.len(), n3d)?;
    validate_len("u_ms", volume.u_ms.len(), n3d)?;
    validate_len("v_ms", volume.v_ms.len(), n3d)?;
    Ok(())
}

fn lapse_rate_700_500_column(
    volume: EcapeVolumeInputs<'_>,
    n2d: usize,
    ij: usize,
    pressure_levels_hpa: Option<&[f64]>,
) -> f64 {
    let mut pressure_hpa = Vec::with_capacity(volume.nz);
    let mut virtual_temperature_c = Vec::with_capacity(volume.nz);
    let mut height_agl_m = Vec::with_capacity(volume.nz);
    for k in 0..volume.nz {
        let idx = k * n2d + ij;
        let pressure = match pressure_levels_hpa {
            Some(levels) => levels[k],
            None => volume.pressure_pa[idx] / 100.0,
        };
        let temperature = volume.temperature_c[idx];
        let dewpoint = dewpoint_from_q(volume.qvapor_kgkg[idx], pressure).min(temperature);
        let tv = metrust::calc::thermo::virtual_temperature_from_dewpoint(
            temperature,
            dewpoint,
            pressure,
        );
        pressure_hpa.push(pressure);
        virtual_temperature_c.push(tv);
        height_agl_m.push(volume.height_agl_m[idx]);
    }

    if pressure_hpa.len() > 1 && pressure_hpa[0] < pressure_hpa[pressure_hpa.len() - 1] {
        pressure_hpa.reverse();
        virtual_temperature_c.reverse();
        height_agl_m.reverse();
    }

    let Some(tv700) = interp_at_pressure(&pressure_hpa, &virtual_temperature_c, 700.0) else {
        return f64::NAN;
    };
    let Some(tv500) = interp_at_pressure(&pressure_hpa, &virtual_temperature_c, 500.0) else {
        return f64::NAN;
    };
    let Some(z700) = interp_at_pressure(&pressure_hpa, &height_agl_m, 700.0) else {
        return f64::NAN;
    };
    let Some(z500) = interp_at_pressure(&pressure_hpa, &height_agl_m, 500.0) else {
        return f64::NAN;
    };
    let dz_km = (z500 - z700) / 1000.0;
    if dz_km > 0.0 {
        (tv700 - tv500) / dz_km
    } else {
        f64::NAN
    }
}

fn interp_at_pressure(pressures_hpa: &[f64], values: &[f64], target_hpa: f64) -> Option<f64> {
    if pressures_hpa.len() != values.len() || pressures_hpa.is_empty() {
        return None;
    }
    for idx in 1..pressures_hpa.len() {
        let p0 = pressures_hpa[idx - 1];
        let p1 = pressures_hpa[idx];
        if (p0 >= target_hpa && target_hpa >= p1) || (p1 >= target_hpa && target_hpa >= p0) {
            let denom = p1 - p0;
            if denom.abs() < 1.0e-9 {
                return Some(values[idx]);
            }
            let weight = (target_hpa - p0) / denom;
            return Some(values[idx - 1] + weight * (values[idx] - values[idx - 1]));
        }
    }
    None
}

fn fixed_stp_value(sbcape_jkg: f64, lcl_m: f64, srh_1km_m2s2: f64, shear_6km_ms: f64) -> f64 {
    let cape_term = (sbcape_jkg / 1500.0).max(0.0);
    let lcl_term = if lcl_m >= 2000.0 {
        0.0
    } else if lcl_m <= 1000.0 {
        1.0
    } else {
        (2000.0 - lcl_m) / 1000.0
    };
    let srh_term = (srh_1km_m2s2 / 150.0).max(0.0);
    let shear_term = if shear_6km_ms < 12.5 {
        0.0
    } else if shear_6km_ms >= 30.0 {
        1.5
    } else {
        shear_6km_ms / 20.0
    };

    cape_term * lcl_term * srh_term * shear_term
}

fn effective_stp_value(
    mlcape_jkg: f64,
    mlcin_jkg: f64,
    ml_lcl_m: f64,
    effective_srh_m2s2: f64,
    effective_bulk_wind_difference_ms: f64,
) -> f64 {
    let cape_term = (mlcape_jkg / 1500.0).max(0.0);
    let lcl_term = if ml_lcl_m >= 2000.0 {
        0.0
    } else if ml_lcl_m <= 1000.0 {
        1.0
    } else {
        (2000.0 - ml_lcl_m) / 1000.0
    };
    let srh_term = (effective_srh_m2s2 / 150.0).max(0.0);
    let shear_term = if effective_bulk_wind_difference_ms < 12.5 {
        0.0
    } else if effective_bulk_wind_difference_ms >= 30.0 {
        1.5
    } else {
        effective_bulk_wind_difference_ms / 20.0
    };
    let cin_term = ((200.0 + mlcin_jkg) / 150.0).clamp(0.0, 1.0);

    cape_term * lcl_term * srh_term * shear_term * cin_term
}

fn fixed_layer_tornado_shear_term(shear_6km_ms: f64) -> f64 {
    if shear_6km_ms < 12.5 {
        0.0
    } else if shear_6km_ms > 30.0 {
        1.5
    } else {
        shear_6km_ms / 20.0
    }
}

fn tornadic_low_level_limit_exceeded(ml_lcl_m: f64, mlcin_jkg: f64, sbcin_jkg: f64) -> bool {
    ml_lcl_m > 1700.0 || mlcin_jkg < -100.0 || sbcin_jkg < -200.0
}

fn tehi_value(
    srh_1km_m2s2: f64,
    mlcape_jkg: f64,
    mlcape_03km_jkg: f64,
    shear_6km_ms: f64,
    ml_lcl_m: f64,
    mlcin_jkg: f64,
    sbcin_jkg: f64,
) -> f64 {
    let mut mlcape_03km_term = if mlcape_03km_jkg > 300.0 {
        1.5
    } else {
        mlcape_03km_jkg / 200.0
    };
    if mlcape_jkg > 1500.0 {
        mlcape_03km_term = mlcape_03km_term.max(1.0);
    }

    let value = ((srh_1km_m2s2 * mlcape_jkg) / 160000.0)
        * mlcape_03km_term
        * fixed_layer_tornado_shear_term(shear_6km_ms);

    if tornadic_low_level_limit_exceeded(ml_lcl_m, mlcin_jkg, sbcin_jkg) || value < 0.0 {
        0.0
    } else {
        value
    }
}

fn tts_value(
    srh_1km_m2s2: f64,
    mlcape_03km_jkg: f64,
    mlcape_jkg: f64,
    shear_6km_ms: f64,
    ml_lcl_m: f64,
    mlcin_jkg: f64,
    sbcin_jkg: f64,
) -> f64 {
    let mlcape_03km_capped = mlcape_03km_jkg.min(150.0);
    let mlcape_term = if mlcape_jkg < 2000.0 {
        1.0
    } else if mlcape_jkg > 3000.0 {
        1.5
    } else {
        mlcape_jkg / 2000.0
    };

    let value = ((srh_1km_m2s2 * mlcape_03km_capped) / 6500.0)
        * mlcape_term
        * fixed_layer_tornado_shear_term(shear_6km_ms);

    if tornadic_low_level_limit_exceeded(ml_lcl_m, mlcin_jkg, sbcin_jkg) || value < 0.0 {
        0.0
    } else {
        value
    }
}

fn vtp_mod_value(
    mlcape_jkg: f64,
    effective_srh_m2s2: f64,
    effective_bulk_wind_difference_ms: f64,
    ml_lcl_m: f64,
    mlcin_jkg: f64,
    mlcape_03km_jkg: f64,
    lapse_rate_700_500_cpkm: f64,
) -> f64 {
    let ebwd_term = if effective_bulk_wind_difference_ms <= 20.0 {
        0.0
    } else if effective_bulk_wind_difference_ms >= 45.0 {
        1.5
    } else {
        effective_bulk_wind_difference_ms / 30.0
    };
    let mllcl_term = if ml_lcl_m >= 1750.0 {
        0.0
    } else if ml_lcl_m <= 750.0 {
        1.0
    } else {
        (1750.0 - ml_lcl_m) / 750.0
    };
    let mlcin_term = if mlcin_jkg <= -200.0 {
        0.0
    } else if mlcin_jkg >= -50.0 {
        1.0
    } else {
        (mlcin_jkg + 200.0) / 150.0
    };
    let mlcape_03km_term = if mlcape_03km_jkg >= 100.0 {
        2.0
    } else {
        mlcape_03km_jkg / 50.0
    };
    let lr_term = if lapse_rate_700_500_cpkm <= 4.5 {
        0.0
    } else if lapse_rate_700_500_cpkm >= 8.5 {
        2.0
    } else {
        (lapse_rate_700_500_cpkm - 4.5) / 2.0
    };

    let p1 = (mlcape_jkg / 1700.0) * (effective_srh_m2s2 / 250.0) * ebwd_term * mllcl_term;
    let p2 = mlcin_term * mlcape_03km_term * lr_term;
    p1 * p2
}

fn ehi_value(cape_jkg: f64, srh_m2s2: f64) -> f64 {
    (cape_jkg * srh_m2s2) / 160000.0
}

fn scp_effective_value(
    mucape_jkg: f64,
    effective_srh_m2s2: f64,
    effective_bulk_wind_difference_ms: f64,
) -> f64 {
    let cape_term = (mucape_jkg / 1000.0).max(0.0);
    let srh_term = (effective_srh_m2s2 / 50.0).max(0.0);
    let shear_term = if effective_bulk_wind_difference_ms > 20.0 {
        1.0
    } else if effective_bulk_wind_difference_ms < 10.0 {
        0.0
    } else {
        effective_bulk_wind_difference_ms / 20.0
    };

    cape_term * srh_term * shear_term
}

fn ship_value(
    mucape_jkg: f64,
    shear_6km_ms: f64,
    temperature_500c: f64,
    lapse_rate_700_500_cpkm: f64,
    mixing_ratio_500_gkg: f64,
) -> f64 {
    let mucape = mucape_jkg.max(0.0);
    let shear = shear_6km_ms.max(0.0);
    let temperature_500_term = (-temperature_500c).max(0.0);
    let lapse_rate = lapse_rate_700_500_cpkm.max(0.0);
    let mixing_ratio = mixing_ratio_500_gkg.max(0.0);

    let ship = (mucape * mixing_ratio * lapse_rate * temperature_500_term * shear) / 42_000_000.0;

    if mucape < 1300.0 {
        ship * (mucape / 1300.0)
    } else {
        ship
    }
}

fn bri_value(cape_jkg: f64, brn_shear_ms: f64) -> f64 {
    let denom = 0.5 * brn_shear_ms * brn_shear_ms;
    if denom > 0.1 {
        cape_jkg.max(0.0) / denom
    } else {
        0.0
    }
}

fn validate_wind_inputs(wind: WindGridInputs<'_>) -> Result<(), CalcError> {
    let n3d = wind.shape.len3d();
    validate_len("u_3d_ms", wind.u_3d_ms.len(), n3d)?;
    validate_len("v_3d_ms", wind.v_3d_ms.len(), n3d)?;
    validate_len("height_agl_3d_m", wind.height_agl_3d_m.len(), n3d)?;
    Ok(())
}

fn validate_grid_fields(
    grid: GridShape,
    fields: &[(&'static str, &[f64])],
) -> Result<(), CalcError> {
    let n = grid.len();
    for (field, values) in fields {
        validate_len(field, values.len(), n)?;
    }
    Ok(())
}

fn validate_fixed_stp_inputs(inputs: FixedStpInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("sbcape_jkg", inputs.sbcape_jkg),
            ("lcl_m", inputs.lcl_m),
            ("srh_1km_m2s2", inputs.srh_1km_m2s2),
            ("shear_6km_ms", inputs.shear_6km_ms),
        ],
    )
}

fn validate_effective_stp_inputs(inputs: EffectiveStpInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("mlcape_jkg", inputs.mlcape_jkg),
            ("mlcin_jkg", inputs.mlcin_jkg),
            ("ml_lcl_m", inputs.ml_lcl_m),
            ("effective_srh_m2s2", inputs.effective_srh_m2s2),
            (
                "effective_bulk_wind_difference_ms",
                inputs.effective_bulk_wind_difference_ms,
            ),
        ],
    )
}

fn validate_effective_scp_inputs(inputs: EffectiveScpInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("mucape_jkg", inputs.mucape_jkg),
            ("effective_srh_m2s2", inputs.effective_srh_m2s2),
            (
                "effective_bulk_wind_difference_ms",
                inputs.effective_bulk_wind_difference_ms,
            ),
        ],
    )
}

fn validate_effective_severe_inputs(inputs: EffectiveSevereInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("mlcape_jkg", inputs.mlcape_jkg),
            ("mlcin_jkg", inputs.mlcin_jkg),
            ("ml_lcl_m", inputs.ml_lcl_m),
            ("mucape_jkg", inputs.mucape_jkg),
            ("effective_srh_m2s2", inputs.effective_srh_m2s2),
            (
                "effective_bulk_wind_difference_ms",
                inputs.effective_bulk_wind_difference_ms,
            ),
        ],
    )
}

fn validate_tornadic_beta_inputs(inputs: TornadicBetaInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("srh_1km_m2s2", inputs.srh_1km_m2s2),
            ("mlcape_jkg", inputs.mlcape_jkg),
            ("mlcape_03km_jkg", inputs.mlcape_03km_jkg),
            ("shear_6km_ms", inputs.shear_6km_ms),
            ("ml_lcl_m", inputs.ml_lcl_m),
            ("mlcin_jkg", inputs.mlcin_jkg),
            ("sbcin_jkg", inputs.sbcin_jkg),
        ],
    )
}

fn validate_vtp_mod_inputs(inputs: VtpModInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("mlcape_jkg", inputs.mlcape_jkg),
            ("effective_srh_m2s2", inputs.effective_srh_m2s2),
            (
                "effective_bulk_wind_difference_ms",
                inputs.effective_bulk_wind_difference_ms,
            ),
            ("ml_lcl_m", inputs.ml_lcl_m),
            ("mlcin_jkg", inputs.mlcin_jkg),
            ("mlcape_03km_jkg", inputs.mlcape_03km_jkg),
            ("lapse_rate_700_500_cpkm", inputs.lapse_rate_700_500_cpkm),
        ],
    )
}

fn validate_scp_ehi_inputs(inputs: ScpEhiInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("scp_cape_jkg", inputs.scp_cape_jkg),
            ("scp_srh_m2s2", inputs.scp_srh_m2s2),
            (
                "scp_bulk_wind_difference_ms",
                inputs.scp_bulk_wind_difference_ms,
            ),
            ("ehi_cape_jkg", inputs.ehi_cape_jkg),
            ("ehi_srh_m2s2", inputs.ehi_srh_m2s2),
        ],
    )
}

fn validate_ship_inputs(inputs: ShipInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("mucape_jkg", inputs.mucape_jkg),
            ("shear_6km_ms", inputs.shear_6km_ms),
            ("temperature_500c", inputs.temperature_500c),
            ("lapse_rate_700_500_cpkm", inputs.lapse_rate_700_500_cpkm),
            ("mixing_ratio_500_gkg", inputs.mixing_ratio_500_gkg),
        ],
    )
}

fn validate_bulk_richardson_inputs(inputs: BulkRichardsonInputs<'_>) -> Result<(), CalcError> {
    validate_grid_fields(
        inputs.grid,
        &[
            ("cape_jkg", inputs.cape_jkg),
            ("brn_shear_ms", inputs.brn_shear_ms),
        ],
    )
}

#[cfg(test)]
mod tests;
