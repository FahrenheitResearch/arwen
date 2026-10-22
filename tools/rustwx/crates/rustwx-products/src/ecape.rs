use crate::gridded::{PressureFields, SharedTiming, SurfaceFields};
use crate::heavy::{HeavyComputeTiming, HeavyRenderedArtifact};
use crate::publication::PublishedFetchIdentity;
use crate::shared_context::{DomainSpec, WeatherPanelField};
use rustwx_calc::{
    EcapeTripletOptions, EcapeVolumeInputs, EffectiveStpInputs, ScpEhiInputs, SurfaceInputs,
    WindGridInputs, compute_ecape_triplet_with_failure_mask_from_parts, compute_ehi,
    compute_mlcape_cin, compute_scp_ehi, compute_stp_effective, compute_wind_diagnostics_bundle,
};
use rustwx_core::{ModelId, SourceId};
use rustwx_render::WeatherProduct;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

const ECAPE_CAPE_RATIO_MIN_DENOMINATOR_JKG: f64 = 100.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcapeBatchRequest {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_override_utc: Option<u8>,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub out_dir: PathBuf,
    pub cache_root: PathBuf,
    pub use_cache: bool,
    pub surface_product_override: Option<String>,
    pub pressure_product_override: Option<String>,
    pub allow_large_heavy_domain: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcapeBatchReport {
    pub model: ModelId,
    pub date_yyyymmdd: String,
    pub cycle_utc: u8,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub domain: DomainSpec,
    pub outputs: Vec<HeavyRenderedArtifact>,
    pub input_fetches: Vec<PublishedFetchIdentity>,
    pub shared_timing: SharedTiming,
    pub heavy_timing: HeavyComputeTiming,
    pub project_ms: u128,
    pub compute_ms: u128,
    pub render_ms: u128,
    pub total_ms: u128,
    pub failure_count: usize,
}

/// Per-kernel wall times inside [`compute_ecape_map_fields_with_prepared_volume`],
/// for callers (the store-ingest lane) that need an accurate breakdown of
/// where the heavy stage's time goes. Pure observation: the timed variant
/// runs the exact same kernels in the exact same order.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EcapeMapFieldsTiming {
    /// The SB/ML/MU entraining-parcel ECAPE triplet (the dominant cost).
    pub ecape_triplet_ms: u128,
    /// SRH 0-1/0-3 km + 0-6 km bulk shear for the experimental composites.
    pub wind_diagnostics_ms: u128,
    /// The classic (non-entraining) ML parcel pass feeding the STP's LCL.
    pub ml_classic_ms: u128,
    /// Elementwise composites + ratios (SCP/EHI/STP, derived/native ratios).
    pub composites_ms: u128,
}

pub fn compute_ecape_map_fields_with_prepared_volume(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    prepared: &crate::gridded::PreparedHeavyVolume,
) -> Result<(Vec<WeatherPanelField>, usize), Box<dyn std::error::Error>> {
    let (fields, failure_count, _) =
        compute_ecape_map_fields_with_prepared_volume_timed(surface, pressure, prepared)?;
    Ok((fields, failure_count))
}

pub fn compute_ecape_map_fields_with_prepared_volume_timed(
    surface: &SurfaceFields,
    pressure: &PressureFields,
    prepared: &crate::gridded::PreparedHeavyVolume,
) -> Result<(Vec<WeatherPanelField>, usize, EcapeMapFieldsTiming), Box<dyn std::error::Error>> {
    let mut timing = EcapeMapFieldsTiming::default();
    let triplet_start = Instant::now();
    let triplet = compute_ecape_triplet_with_failure_mask_from_parts(
        prepared.grid,
        EcapeVolumeInputs {
            pressure_pa: prepared
                .pressure_3d_pa
                .as_deref()
                .unwrap_or(&prepared.pressure_levels_pa),
            temperature_c: &pressure.temperature_c_3d,
            qvapor_kgkg: &pressure.qvapor_kgkg_3d,
            height_agl_m: &prepared.height_agl_3d,
            u_ms: &pressure.u_ms_3d,
            v_ms: &pressure.v_ms_3d,
            nz: prepared.shape.nz,
        },
        SurfaceInputs {
            psfc_pa: &surface.psfc_pa,
            t2_k: &surface.t2_k,
            q2_kgkg: &surface.q2_kgkg,
            u10_ms: &surface.u10_ms,
            v10_ms: &surface.v10_ms,
        },
        EcapeTripletOptions::new("right_moving"),
    )?;
    timing.ecape_triplet_ms = triplet_start.elapsed().as_millis();
    let wind_start = Instant::now();
    let wind = WindGridInputs {
        shape: prepared.shape,
        u_3d_ms: &pressure.u_ms_3d,
        v_3d_ms: &pressure.v_ms_3d,
        height_agl_3d_m: &prepared.height_agl_3d,
    };
    let wind_diagnostics = compute_wind_diagnostics_bundle(wind)?;
    timing.wind_diagnostics_ms = wind_start.elapsed().as_millis();
    let composites_start = Instant::now();
    let experimental = compute_scp_ehi(ScpEhiInputs {
        grid: prepared.grid,
        scp_cape_jkg: &triplet.mu.fields.ecape_jkg,
        scp_srh_m2s2: &wind_diagnostics.srh_03km_m2s2,
        scp_bulk_wind_difference_ms: &wind_diagnostics.shear_06km_ms,
        ehi_cape_jkg: &triplet.sb.fields.ecape_jkg,
        ehi_srh_m2s2: &wind_diagnostics.srh_01km_m2s2,
    })?;
    let ecape_ehi_03km = compute_ehi(
        prepared.grid,
        &triplet.sb.fields.ecape_jkg,
        &wind_diagnostics.srh_03km_m2s2,
    )?;
    timing.composites_ms += composites_start.elapsed().as_millis();
    let ml_classic_start = Instant::now();
    let ml_classic = compute_mlcape_cin(
        prepared.grid,
        EcapeVolumeInputs {
            pressure_pa: prepared
                .pressure_3d_pa
                .as_deref()
                .unwrap_or(&prepared.pressure_levels_pa),
            temperature_c: &pressure.temperature_c_3d,
            qvapor_kgkg: &pressure.qvapor_kgkg_3d,
            height_agl_m: &prepared.height_agl_3d,
            u_ms: &pressure.u_ms_3d,
            v_ms: &pressure.v_ms_3d,
            nz: prepared.shape.nz,
        },
        SurfaceInputs {
            psfc_pa: &surface.psfc_pa,
            t2_k: &surface.t2_k,
            q2_kgkg: &surface.q2_kgkg,
            u10_ms: &surface.u10_ms,
            v10_ms: &surface.v10_ms,
        },
        None,
    )?;
    timing.ml_classic_ms = ml_classic_start.elapsed().as_millis();
    let tail_start = Instant::now();
    let ecape_stp = compute_stp_effective(EffectiveStpInputs {
        grid: prepared.grid,
        mlcape_jkg: &triplet.ml.fields.ecape_jkg,
        mlcin_jkg: &triplet.ml.fields.cin_jkg,
        ml_lcl_m: &ml_classic.lcl_m,
        effective_srh_m2s2: &wind_diagnostics.srh_01km_m2s2,
        effective_bulk_wind_difference_ms: &wind_diagnostics.shear_06km_ms,
    })?;
    let failure_count = triplet.total_failure_count();

    let sb_derived_ratio =
        ecape_cape_ratio(&triplet.sb.fields.ecape_jkg, &triplet.sb.fields.cape_jkg);
    let ml_derived_ratio =
        ecape_cape_ratio(&triplet.ml.fields.ecape_jkg, &triplet.ml.fields.cape_jkg);
    let mu_derived_ratio =
        ecape_cape_ratio(&triplet.mu.fields.ecape_jkg, &triplet.mu.fields.cape_jkg);

    let mut fields = vec![
        WeatherPanelField::new(WeatherProduct::Sbecape, "J/kg", triplet.sb.fields.ecape_jkg),
        WeatherPanelField::new(WeatherProduct::Mlecape, "J/kg", triplet.ml.fields.ecape_jkg),
        WeatherPanelField::new(WeatherProduct::Muecape, "J/kg", triplet.mu.fields.ecape_jkg),
        WeatherPanelField::new(
            WeatherProduct::SbEcapeDerivedCapeRatio,
            "ratio",
            sb_derived_ratio,
        ),
        WeatherPanelField::new(
            WeatherProduct::MlEcapeDerivedCapeRatio,
            "ratio",
            ml_derived_ratio,
        ),
        WeatherPanelField::new(
            WeatherProduct::MuEcapeDerivedCapeRatio,
            "ratio",
            mu_derived_ratio,
        ),
        WeatherPanelField::new(WeatherProduct::Sbncape, "J/kg", triplet.sb.fields.ncape_jkg),
        WeatherPanelField::new(WeatherProduct::Sbecin, "J/kg", triplet.sb.fields.cin_jkg),
        WeatherPanelField::new(WeatherProduct::Mlecin, "J/kg", triplet.ml.fields.cin_jkg),
        WeatherPanelField::new(
            WeatherProduct::EcapeScpExperimental,
            "dimensionless",
            experimental.scp,
        ),
        WeatherPanelField::new(
            WeatherProduct::EcapeEhi01kmExperimental,
            "dimensionless",
            experimental.ehi,
        ),
        WeatherPanelField::new(
            WeatherProduct::EcapeEhi03kmExperimental,
            "dimensionless",
            ecape_ehi_03km,
        ),
        WeatherPanelField::new(
            WeatherProduct::EcapeStpExperimental,
            "dimensionless",
            ecape_stp,
        ),
    ];
    if let Some(native_sbcape_jkg) = surface.native_sbcape_jkg.as_ref() {
        fields.push(WeatherPanelField::new(
            WeatherProduct::SbEcapeNativeCapeRatio,
            "ratio",
            ecape_cape_ratio(&fields[0].values, native_sbcape_jkg),
        ));
    }
    if let Some(native_mlcape_jkg) = surface.native_mlcape_jkg.as_ref() {
        fields.push(WeatherPanelField::new(
            WeatherProduct::MlEcapeNativeCapeRatio,
            "ratio",
            ecape_cape_ratio(&fields[1].values, native_mlcape_jkg),
        ));
    }
    if let Some(native_mucape_jkg) = surface.native_mucape_jkg.as_ref() {
        fields.push(WeatherPanelField::new(
            WeatherProduct::MuEcapeNativeCapeRatio,
            "ratio",
            ecape_cape_ratio(&fields[2].values, native_mucape_jkg),
        ));
    }
    timing.composites_ms += tail_start.elapsed().as_millis();
    Ok((fields, failure_count, timing))
}

fn ecape_cape_ratio(ecape_jkg: &[f64], cape_jkg: &[f64]) -> Vec<f64> {
    ecape_jkg
        .iter()
        .zip(cape_jkg.iter())
        .map(|(&ecape, &cape)| {
            if ecape.is_finite() && cape.is_finite() && cape >= ECAPE_CAPE_RATIO_MIN_DENOMINATOR_JKG
            {
                ecape / cape
            } else {
                f64::NAN
            }
        })
        .collect()
}
