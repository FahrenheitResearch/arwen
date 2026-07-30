use rayon::prelude::*;
use rustwx_calc::{
    CalcError, CapeCinOutputs, EcapeVolumeInputs, FixedStpInputs, GridShape as CalcGridShape,
    SurfaceInputs, SurfaceThermoOutputs, TemperatureAdvectionInputs, VolumeShape, WindGridInputs,
    compute_cape_cin_triplet, compute_dcape, compute_ehi_01km, compute_ehi_03km,
    compute_lapse_rate_0_3km, compute_lapse_rate_700_500, compute_lifted_index, compute_mlcape_cin,
    compute_mucape_cin, compute_sbcape_cin, compute_shear_01km, compute_shear_06km,
    compute_srh_01km_hemispheric, compute_srh_03km_hemispheric, compute_stp_fixed,
    compute_surface_thermo,
};

use crate::gridded::{
    PressureFields as GenericPressureFields, SurfaceFields as GenericSurfaceFields,
};

use super::KNOTS_PER_MS;
use super::recipes::{DerivedRecipe, DerivedRequirements, derived_compute_recipes_need_pressure};

pub(super) trait SurfaceFieldSet {
    fn lat(&self) -> &[f64];
    fn lon(&self) -> &[f64];
    fn nx(&self) -> usize;
    fn ny(&self) -> usize;
    fn projection(&self) -> Option<&rustwx_core::GridProjection>;
    fn orog_m(&self) -> &[f64];
    fn psfc_pa(&self) -> &[f64];
    fn t2_k(&self) -> &[f64];
    fn q2_kgkg(&self) -> &[f64];
    fn u10_ms(&self) -> &[f64];
    fn v10_ms(&self) -> &[f64];
}

pub(super) trait PressureFieldSet {
    fn pressure_levels_hpa(&self) -> &[f64];
    fn pressure_3d_pa(&self) -> Option<&[f64]>;
    fn temperature_c_3d(&self) -> &[f64];
    fn qvapor_kgkg_3d(&self) -> &[f64];
    fn u_ms_3d(&self) -> &[f64];
    fn v_ms_3d(&self) -> &[f64];
    fn gh_m_3d(&self) -> &[f64];
}

impl SurfaceFieldSet for GenericSurfaceFields {
    fn lat(&self) -> &[f64] {
        &self.lat
    }

    fn lon(&self) -> &[f64] {
        &self.lon
    }

    fn nx(&self) -> usize {
        self.nx
    }

    fn ny(&self) -> usize {
        self.ny
    }

    fn projection(&self) -> Option<&rustwx_core::GridProjection> {
        self.projection.as_ref()
    }

    fn orog_m(&self) -> &[f64] {
        &self.orog_m
    }

    fn psfc_pa(&self) -> &[f64] {
        &self.psfc_pa
    }

    fn t2_k(&self) -> &[f64] {
        &self.t2_k
    }

    fn q2_kgkg(&self) -> &[f64] {
        &self.q2_kgkg
    }

    fn u10_ms(&self) -> &[f64] {
        &self.u10_ms
    }

    fn v10_ms(&self) -> &[f64] {
        &self.v10_ms
    }
}

impl PressureFieldSet for GenericPressureFields {
    fn pressure_levels_hpa(&self) -> &[f64] {
        &self.pressure_levels_hpa
    }

    fn pressure_3d_pa(&self) -> Option<&[f64]> {
        self.pressure_3d_pa.as_deref()
    }

    fn temperature_c_3d(&self) -> &[f64] {
        &self.temperature_c_3d
    }

    fn qvapor_kgkg_3d(&self) -> &[f64] {
        &self.qvapor_kgkg_3d
    }

    fn u_ms_3d(&self) -> &[f64] {
        &self.u_ms_3d
    }

    fn v_ms_3d(&self) -> &[f64] {
        &self.v_ms_3d
    }

    fn gh_m_3d(&self) -> &[f64] {
        &self.gh_m_3d
    }
}

struct EmptyPressureFields;

impl PressureFieldSet for EmptyPressureFields {
    fn pressure_levels_hpa(&self) -> &[f64] {
        &[]
    }

    fn pressure_3d_pa(&self) -> Option<&[f64]> {
        None
    }

    fn temperature_c_3d(&self) -> &[f64] {
        &[]
    }

    fn qvapor_kgkg_3d(&self) -> &[f64] {
        &[]
    }

    fn u_ms_3d(&self) -> &[f64] {
        &[]
    }

    fn v_ms_3d(&self) -> &[f64] {
        &[]
    }

    fn gh_m_3d(&self) -> &[f64] {
        &[]
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct DerivedComputedFields {
    pub(super) sbcape_jkg: Option<Vec<f64>>,
    pub(super) sbcin_jkg: Option<Vec<f64>>,
    pub(super) sblcl_m: Option<Vec<f64>>,
    pub(super) mlcape_jkg: Option<Vec<f64>>,
    pub(super) mlcin_jkg: Option<Vec<f64>>,
    pub(super) mucape_jkg: Option<Vec<f64>>,
    pub(super) mucin_jkg: Option<Vec<f64>>,
    pub(super) dcape_jkg: Option<Vec<f64>>,
    pub(super) theta_e_2m_k: Option<Vec<f64>>,
    pub(super) vpd_2m_hpa: Option<Vec<f64>>,
    pub(super) dewpoint_depression_2m_c: Option<Vec<f64>>,
    pub(super) wetbulb_2m_c: Option<Vec<f64>>,
    pub(super) fire_weather_composite: Option<Vec<f64>>,
    pub(super) apparent_temperature_2m_c: Option<Vec<f64>>,
    pub(super) heat_index_2m_c: Option<Vec<f64>>,
    pub(super) wind_chill_2m_c: Option<Vec<f64>>,
    pub(super) surface_u10_ms: Option<Vec<f64>>,
    pub(super) surface_v10_ms: Option<Vec<f64>>,
    pub(super) lifted_index_c: Option<Vec<f64>>,
    pub(super) lapse_rate_700_500_cpkm: Option<Vec<f64>>,
    pub(super) lapse_rate_0_3km_cpkm: Option<Vec<f64>>,
    pub(super) shear_01km_kt: Option<Vec<f64>>,
    pub(super) shear_06km_kt: Option<Vec<f64>>,
    pub(super) srh_01km_m2s2: Option<Vec<f64>>,
    pub(super) srh_03km_m2s2: Option<Vec<f64>>,
    pub(super) ehi_01km: Option<Vec<f64>>,
    pub(super) ehi_03km: Option<Vec<f64>>,
    pub(super) stp_fixed: Option<Vec<f64>>,
    pub(super) scp_mu_03km_06km_proxy: Option<Vec<f64>>,
    pub(super) temperature_advection_700mb_cph: Option<Vec<f64>>,
    pub(super) temperature_advection_850mb_cph: Option<Vec<f64>>,
}

/// What the parcel (CAPE) side of the kernel fork produced.
type ParcelOutputs = (
    Option<CapeCinOutputs>,
    Option<CapeCinOutputs>,
    Option<CapeCinOutputs>,
);

/// What the independent (non-parcel) side of the kernel fork produced, in
/// the same units the sequential lane produced them.
#[derive(Default)]
struct IndependentOutputs {
    dcape_jkg: Option<Vec<f64>>,
    surface_thermo: Option<SurfaceThermoOutputs>,
    lifted_index_c: Option<Vec<f64>>,
    lapse_rate_700_500_cpkm: Option<Vec<f64>>,
    lapse_rate_0_3km_cpkm: Option<Vec<f64>>,
    shear_01km_ms: Option<Vec<f64>>,
    shear_06km_ms: Option<Vec<f64>>,
    srh_01km_m2s2: Option<Vec<f64>>,
    srh_03km_m2s2: Option<Vec<f64>>,
    temperature_advection_700mb_cph: Option<Vec<f64>>,
    temperature_advection_850mb_cph: Option<Vec<f64>>,
}

/// Errors crossing the `rayon::join` boundary must be `Send + Sync`.
type TaskError = Box<dyn std::error::Error + Send + Sync>;

pub(super) fn compute_derived_fields_generic<S, P>(
    surface: &S,
    pressure: &P,
    recipes: &[DerivedRecipe],
) -> Result<DerivedComputedFields, Box<dyn std::error::Error>>
where
    S: SurfaceFieldSet + Sync,
    P: PressureFieldSet + Sync,
{
    compute_derived_fields_generic_with_height_agl(surface, pressure, recipes, None)
}

/// [`compute_derived_fields_generic`] with an optional precomputed
/// height-AGL volume. `Some` skips the internal height-AGL assembly and
/// must hold exactly the volume [`compute_height_agl_3d_generic`] would
/// produce (the store-ingest lane builds it once, by an in-place transform
/// of the gh volume with the identical per-element arithmetic, and shares
/// it across the derived and heavy stages). `None` is the historical path.
fn missing_dependency(name: &str) -> std::io::Error {
    std::io::Error::other(format!(
        "derived compute missing required dependency: {name}"
    ))
}

// Concrete error type so `?` converts into both the outer
// `Box<dyn Error>` and the join sides' `TaskError`.
fn require_option_ref<'a, T>(option: &'a Option<T>, name: &str) -> Result<&'a T, std::io::Error> {
    option.as_ref().ok_or_else(|| missing_dependency(name))
}

fn require_option_copy<T: Copy>(option: Option<T>, name: &str) -> Result<T, std::io::Error> {
    option.ok_or_else(|| missing_dependency(name))
}

fn require_height_agl<'a>(
    option: Option<&'a [f64]>,
    name: &str,
) -> Result<&'a [f64], std::io::Error> {
    option.ok_or_else(|| missing_dependency(name))
}

pub(super) fn compute_derived_fields_generic_with_height_agl<S, P>(
    surface: &S,
    pressure: &P,
    recipes: &[DerivedRecipe],
    height_agl_override: Option<&[f64]>,
) -> Result<DerivedComputedFields, Box<dyn std::error::Error>>
where
    S: SurfaceFieldSet + Sync,
    P: PressureFieldSet + Sync,
{
    let requirements = DerivedRequirements::from_recipes(recipes);
    let grid = CalcGridShape::new(surface.nx(), surface.ny())?;

    let surface_inputs = SurfaceInputs {
        psfc_pa: surface.psfc_pa(),
        t2_k: surface.t2_k(),
        q2_kgkg: surface.q2_kgkg(),
        u10_ms: surface.u10_ms(),
        v10_ms: surface.v10_ms(),
    };

    let shape = if requirements.needs_height_agl() {
        Some(VolumeShape::new(
            grid,
            pressure.pressure_levels_hpa().len(),
        )?)
    } else {
        None
    };
    let pressure_pa = if requirements.needs_volume() {
        Some(
            pressure
                .pressure_3d_pa()
                .map(|values| values.to_vec())
                .unwrap_or_else(|| {
                    pressure
                        .pressure_levels_hpa()
                        .iter()
                        .map(|level_hpa| level_hpa * 100.0)
                        .collect()
                }),
        )
    } else {
        None
    };
    let height_agl_owned = if requirements.needs_height_agl() && height_agl_override.is_none() {
        Some(compute_height_agl_3d_generic(
            surface,
            pressure,
            grid,
            require_option_copy(shape, "volume shape for height_agl")?,
        ))
    } else {
        None
    };
    let height_agl_3d: Option<&[f64]> = height_agl_override.or(height_agl_owned.as_deref());

    let make_volume = || -> Result<EcapeVolumeInputs<'_>, std::io::Error> {
        Ok(EcapeVolumeInputs {
            pressure_pa: require_option_ref(
                &pressure_pa,
                "pressure volume for derived thermodynamics",
            )?,
            temperature_c: pressure.temperature_c_3d(),
            qvapor_kgkg: pressure.qvapor_kgkg_3d(),
            height_agl_m: require_height_agl(
                height_agl_3d,
                "height_agl for derived thermodynamics",
            )?,
            u_ms: pressure.u_ms_3d(),
            v_ms: pressure.v_ms_3d(),
            nz: require_option_copy(shape, "volume shape for derived thermodynamics")?.nz,
        })
    };
    let make_wind = || -> Result<WindGridInputs<'_>, std::io::Error> {
        Ok(WindGridInputs {
            shape: require_option_copy(shape, "volume shape for wind diagnostics")?,
            u_3d_ms: pressure.u_ms_3d(),
            v_3d_ms: pressure.v_ms_3d(),
            height_agl_3d_m: require_height_agl(height_agl_3d, "height_agl for wind diagnostics")?,
        })
    };

    // Fork the kernels into two concurrent groups: the heavy parcel (CAPE)
    // pass on one side, every kernel that does not depend on it on the
    // other. Each kernel is rayon-parallel inside, and both sides share the
    // one global pool, so the light kernels fill scheduling gaps instead of
    // serializing behind the parcel physics. Outputs are identical to the
    // former sequential order — only the wall-clock interleaving changes.
    let (parcels, independent) = rayon::join(
        || -> Result<ParcelOutputs, TaskError> {
            if requirements.sb && requirements.ml && requirements.mu {
                // All three parcel types wanted (the store-ingest lane):
                // one shared pass over the columns, bit-identical outputs.
                let triplet = compute_cape_cin_triplet(grid, make_volume()?, surface_inputs, None)?;
                return Ok((Some(triplet.sb), Some(triplet.ml), Some(triplet.mu)));
            }
            let sb = if requirements.sb {
                Some(compute_sbcape_cin(
                    grid,
                    make_volume()?,
                    surface_inputs,
                    None,
                )?)
            } else {
                None
            };
            let ml = if requirements.ml {
                Some(compute_mlcape_cin(
                    grid,
                    make_volume()?,
                    surface_inputs,
                    None,
                )?)
            } else {
                None
            };
            let mu = if requirements.mu {
                Some(compute_mucape_cin(
                    grid,
                    make_volume()?,
                    surface_inputs,
                    None,
                )?)
            } else {
                None
            };
            Ok((sb, ml, mu))
        },
        || -> Result<IndependentOutputs, TaskError> {
            let mut outputs = IndependentOutputs::default();
            if requirements.dcape {
                outputs.dcape_jkg = Some(compute_dcape(grid, make_volume()?, surface_inputs)?);
            }
            if requirements.surface_thermo {
                outputs.surface_thermo = Some(compute_surface_thermo(grid, surface_inputs)?);
            }
            if requirements.lifted_index {
                outputs.lifted_index_c =
                    Some(compute_lifted_index(grid, make_volume()?, surface_inputs)?);
            }
            if requirements.lapse_rate_700_500 {
                outputs.lapse_rate_700_500_cpkm =
                    Some(compute_lapse_rate_700_500(grid, make_volume()?)?);
            }
            if requirements.lapse_rate_0_3km {
                outputs.lapse_rate_0_3km_cpkm = Some(compute_lapse_rate_0_3km(
                    grid,
                    make_volume()?,
                    surface_inputs,
                )?);
            }
            if requirements.shear_01km {
                outputs.shear_01km_ms = Some(compute_shear_01km(make_wind()?)?);
            }
            if requirements.shear_06km {
                outputs.shear_06km_ms = Some(compute_shear_06km(make_wind()?)?);
            }
            if requirements.srh_01km {
                outputs.srh_01km_m2s2 =
                    Some(compute_srh_01km_hemispheric(make_wind()?, surface.lat())?);
            }
            if requirements.srh_03km {
                outputs.srh_03km_m2s2 =
                    Some(compute_srh_03km_hemispheric(make_wind()?, surface.lat())?);
            }
            if requirements.needs_grid_spacing() {
                let (dx_m, dy_m) = estimate_grid_spacing_m(surface)?;
                if requirements.temperature_advection_700mb {
                    outputs.temperature_advection_700mb_cph =
                        Some(compute_temperature_advection_cph(
                            pressure,
                            pressure.u_ms_3d(),
                            pressure.v_ms_3d(),
                            grid,
                            700.0,
                            dx_m,
                            dy_m,
                        )?);
                }
                if requirements.temperature_advection_850mb {
                    outputs.temperature_advection_850mb_cph =
                        Some(compute_temperature_advection_cph(
                            pressure,
                            pressure.u_ms_3d(),
                            pressure.v_ms_3d(),
                            grid,
                            850.0,
                            dx_m,
                            dy_m,
                        )?);
                }
            }
            Ok(outputs)
        },
    );
    let (sb, ml, mu) = parcels.map_err(|err| err as Box<dyn std::error::Error>)?;
    let independent = independent.map_err(|err| err as Box<dyn std::error::Error>)?;
    assemble_derived_outputs(
        surface,
        recipes,
        &requirements,
        grid,
        (sb, ml, mu),
        independent,
    )
}

/// Fold the kernel outputs into [`DerivedComputedFields`] — the composites
/// (which borrow parcel + wind locals) first, then the moves and unit
/// conversions. Shared verbatim by the generic single-join path and the
/// store lane's phased path so the two cannot drift.
fn assemble_derived_outputs<S>(
    surface: &S,
    recipes: &[DerivedRecipe],
    requirements: &DerivedRequirements,
    grid: CalcGridShape,
    parcels: ParcelOutputs,
    independent: IndependentOutputs,
) -> Result<DerivedComputedFields, Box<dyn std::error::Error>>
where
    S: SurfaceFieldSet,
{
    let (sb, ml, mu) = parcels;
    let mut computed = DerivedComputedFields::default();

    computed.dcape_jkg = independent.dcape_jkg;

    if let Some(surface_thermo) = independent.surface_thermo {
        if recipes.contains(&DerivedRecipe::ThetaE2m10mWinds) {
            computed.theta_e_2m_k = Some(surface_thermo.theta_e_2m_k);
            computed.surface_u10_ms = Some(surface.u10_ms().to_vec());
            computed.surface_v10_ms = Some(surface.v10_ms().to_vec());
        }
        if recipes.contains(&DerivedRecipe::Vpd2m) {
            computed.vpd_2m_hpa = Some(surface_thermo.vpd_2m_hpa);
        }
        if recipes.contains(&DerivedRecipe::DewpointDepression2m) {
            computed.dewpoint_depression_2m_c = Some(surface_thermo.dewpoint_depression_2m_c);
        }
        if recipes.contains(&DerivedRecipe::Wetbulb2m) {
            computed.wetbulb_2m_c = Some(surface_thermo.wetbulb_2m_c);
        }
        if recipes.contains(&DerivedRecipe::FireWeatherComposite) {
            computed.fire_weather_composite = Some(surface_thermo.fire_weather_composite);
        }
        if recipes.contains(&DerivedRecipe::ApparentTemperature2m) {
            computed.apparent_temperature_2m_c = Some(surface_thermo.apparent_temperature_2m_c);
        }
        if recipes.contains(&DerivedRecipe::HeatIndex2m) {
            computed.heat_index_2m_c = Some(surface_thermo.heat_index_2m_c);
        }
        if recipes.contains(&DerivedRecipe::WindChill2m) {
            computed.wind_chill_2m_c = Some(surface_thermo.wind_chill_2m_c);
        }
    }
    computed.lifted_index_c = independent.lifted_index_c;
    computed.lapse_rate_700_500_cpkm = independent.lapse_rate_700_500_cpkm;
    computed.lapse_rate_0_3km_cpkm = independent.lapse_rate_0_3km_cpkm;
    computed.temperature_advection_700mb_cph = independent.temperature_advection_700mb_cph;
    computed.temperature_advection_850mb_cph = independent.temperature_advection_850mb_cph;
    let shear_01km_ms = independent.shear_01km_ms;
    let shear_06km_ms = independent.shear_06km_ms;
    let srh_01km_m2s2 = independent.srh_01km_m2s2;
    let srh_03km_m2s2 = independent.srh_03km_m2s2;

    // Composites first — they borrow the parcel and wind locals. The
    // locals then MOVE into `computed` below; the historical order cloned
    // ~150 MB of grids it was about to move anyway. Same kernels, same
    // call order, same values.
    if requirements.ehi_01km {
        let sb = require_option_ref(&sb, "surface-based CAPE/CIN outputs for EHI 0-1 km")?;
        let srh_01km = require_option_ref(&srh_01km_m2s2, "0-1 km SRH for EHI 0-1 km")?;
        computed.ehi_01km = Some(compute_ehi_01km(grid, &sb.cape_jkg, srh_01km)?);
    }
    if requirements.ehi_03km {
        let sb = require_option_ref(&sb, "surface-based CAPE/CIN outputs for EHI 0-3 km")?;
        let srh_03km = require_option_ref(&srh_03km_m2s2, "0-3 km SRH for EHI 0-3 km")?;
        computed.ehi_03km = Some(compute_ehi_03km(grid, &sb.cape_jkg, srh_03km)?);
    }
    if requirements.stp_fixed {
        let sb = require_option_ref(&sb, "surface-based CAPE/CIN outputs for STP fixed")?;
        let srh_01km = require_option_ref(&srh_01km_m2s2, "0-1 km SRH for STP fixed")?;
        let shear_06km = require_option_ref(&shear_06km_ms, "0-6 km shear for STP fixed")?;
        computed.stp_fixed = Some(compute_stp_fixed(FixedStpInputs {
            grid,
            sbcape_jkg: &sb.cape_jkg,
            lcl_m: &sb.lcl_m,
            srh_1km_m2s2: srh_01km,
            shear_6km_ms: shear_06km,
        })?);
    }
    if requirements.scp_mu_03km_06km_proxy {
        let mu = require_option_ref(&mu, "most-unstable CAPE/CIN outputs for SCP proxy")?;
        let srh_03km = require_option_ref(&srh_03km_m2s2, "0-3 km SRH for SCP proxy")?;
        let shear_06km = require_option_ref(&shear_06km_ms, "0-6 km shear for SCP proxy")?;
        computed.scp_mu_03km_06km_proxy = Some(rustwx_calc::compute_scp(
            grid,
            &mu.cape_jkg,
            srh_03km,
            shear_06km,
        )?);
    }

    // Move the parcel and wind outputs into `computed` (values unchanged;
    // the kt conversions apply the same per-element scale as before).
    if let Some(sb) = sb {
        computed.sbcape_jkg = Some(sb.cape_jkg);
        computed.sbcin_jkg = Some(sb.cin_jkg);
        computed.sblcl_m = Some(sb.lcl_m);
    }
    if let Some(ml) = ml {
        computed.mlcape_jkg = Some(ml.cape_jkg);
        computed.mlcin_jkg = Some(ml.cin_jkg);
    }
    if let Some(mu) = mu {
        computed.mucape_jkg = Some(mu.cape_jkg);
        computed.mucin_jkg = Some(mu.cin_jkg);
    }
    if let Some(values) = shear_01km_ms {
        computed.shear_01km_kt = Some(
            values
                .into_iter()
                .map(|value| value * KNOTS_PER_MS)
                .collect(),
        );
    }
    if let Some(values) = shear_06km_ms {
        computed.shear_06km_kt = Some(
            values
                .into_iter()
                .map(|value| value * KNOTS_PER_MS)
                .collect(),
        );
    }
    computed.srh_01km_m2s2 = srh_01km_m2s2;
    computed.srh_03km_m2s2 = srh_03km_m2s2;

    Ok(computed)
}

/// The wind volumes taken OUT of a `PressureFields` for the store lane's
/// phased compute (~565 MB each at HRRR size).
pub(super) struct TakenWindVolumes {
    pub u_ms_3d: Vec<f64>,
    pub v_ms_3d: Vec<f64>,
}

/// Store-lane variant of [`compute_derived_fields_generic_with_height_agl`]
/// with an early wind release: the caller takes the u/v volumes out of
/// `pressure` and hands them in OWNED; every wind-consuming kernel (the
/// 0-1/0-6 km shears, the 0-1/0-3 km SRHs, and the 700/850 mb temperature
/// advections) runs FIRST, then the wind volumes either leave RAM
/// (`keep_winds = false` — the no-heavy ingest, where nothing downstream
/// reads them; ~1.13 GB off the long parcel window) or come back to the
/// caller (`keep_winds = true` — the heavy stage still needs them). The
/// remaining kernels then run under the exact `rayon::join` fork of the
/// generic path, with empty wind slices in `EcapeVolumeInputs` — the
/// kernels dispatched there are purely thermodynamic and validate winds
/// as present-or-absent (see rustwx-calc), so outputs are bit-identical;
/// `derived::store` pins that equivalence with a synthetic-hour test.
pub(super) fn compute_store_derived_fields_phased<S, P>(
    surface: &S,
    pressure: &P,
    winds: TakenWindVolumes,
    recipes: &[DerivedRecipe],
    height_agl_override: Option<&[f64]>,
    keep_winds: bool,
) -> Result<(DerivedComputedFields, Option<TakenWindVolumes>), Box<dyn std::error::Error>>
where
    S: SurfaceFieldSet + Sync,
    P: PressureFieldSet + Sync,
{
    let requirements = DerivedRequirements::from_recipes(recipes);
    let grid = CalcGridShape::new(surface.nx(), surface.ny())?;

    let surface_inputs = SurfaceInputs {
        psfc_pa: surface.psfc_pa(),
        t2_k: surface.t2_k(),
        q2_kgkg: surface.q2_kgkg(),
        u10_ms: surface.u10_ms(),
        v10_ms: surface.v10_ms(),
    };

    let shape = if requirements.needs_height_agl() {
        Some(VolumeShape::new(
            grid,
            pressure.pressure_levels_hpa().len(),
        )?)
    } else {
        None
    };
    let pressure_pa = if requirements.needs_volume() {
        Some(
            pressure
                .pressure_3d_pa()
                .map(|values| values.to_vec())
                .unwrap_or_else(|| {
                    pressure
                        .pressure_levels_hpa()
                        .iter()
                        .map(|level_hpa| level_hpa * 100.0)
                        .collect()
                }),
        )
    } else {
        None
    };
    let height_agl_owned = if requirements.needs_height_agl() && height_agl_override.is_none() {
        Some(compute_height_agl_3d_generic(
            surface,
            pressure,
            grid,
            require_option_copy(shape, "volume shape for height_agl")?,
        ))
    } else {
        None
    };
    let height_agl_3d: Option<&[f64]> = height_agl_override.or(height_agl_owned.as_deref());

    // --- the wind-consuming kernels, in the historical in-closure order
    // (shear 0-1, shear 0-6, SRH 0-1, SRH 0-3, temperature advections);
    // each is rayon-parallel inside and independent of the others, so
    // running them ahead of the join changes scheduling, never values ---
    let make_wind = || -> Result<WindGridInputs<'_>, std::io::Error> {
        Ok(WindGridInputs {
            shape: require_option_copy(shape, "volume shape for wind diagnostics")?,
            u_3d_ms: &winds.u_ms_3d,
            v_3d_ms: &winds.v_ms_3d,
            height_agl_3d_m: require_height_agl(height_agl_3d, "height_agl for wind diagnostics")?,
        })
    };
    let mut wind_outputs = IndependentOutputs::default();
    if requirements.shear_01km {
        wind_outputs.shear_01km_ms = Some(compute_shear_01km(make_wind()?)?);
    }
    if requirements.shear_06km {
        wind_outputs.shear_06km_ms = Some(compute_shear_06km(make_wind()?)?);
    }
    if requirements.srh_01km {
        wind_outputs.srh_01km_m2s2 =
            Some(compute_srh_01km_hemispheric(make_wind()?, surface.lat())?);
    }
    if requirements.srh_03km {
        wind_outputs.srh_03km_m2s2 =
            Some(compute_srh_03km_hemispheric(make_wind()?, surface.lat())?);
    }
    if requirements.needs_grid_spacing() {
        let (dx_m, dy_m) = estimate_grid_spacing_m(surface)?;
        if requirements.temperature_advection_700mb {
            wind_outputs.temperature_advection_700mb_cph = Some(
                compute_temperature_advection_cph(
                    pressure,
                    &winds.u_ms_3d,
                    &winds.v_ms_3d,
                    grid,
                    700.0,
                    dx_m,
                    dy_m,
                )
                .map_err(|err| err as Box<dyn std::error::Error>)?,
            );
        }
        if requirements.temperature_advection_850mb {
            wind_outputs.temperature_advection_850mb_cph = Some(
                compute_temperature_advection_cph(
                    pressure,
                    &winds.u_ms_3d,
                    &winds.v_ms_3d,
                    grid,
                    850.0,
                    dx_m,
                    dy_m,
                )
                .map_err(|err| err as Box<dyn std::error::Error>)?,
            );
        }
    }

    // --- last wind read is behind us: free (or hand back) ~1.13 GB ---
    let kept_winds = if keep_winds {
        Some(winds)
    } else {
        drop(winds);
        None
    };

    // The thermodynamic kernels below never read the wind members; both
    // modes pass them absent so the two cannot diverge.
    let make_volume = || -> Result<EcapeVolumeInputs<'_>, std::io::Error> {
        Ok(EcapeVolumeInputs {
            pressure_pa: require_option_ref(
                &pressure_pa,
                "pressure volume for derived thermodynamics",
            )?,
            temperature_c: pressure.temperature_c_3d(),
            qvapor_kgkg: pressure.qvapor_kgkg_3d(),
            height_agl_m: require_height_agl(
                height_agl_3d,
                "height_agl for derived thermodynamics",
            )?,
            u_ms: &[],
            v_ms: &[],
            nz: require_option_copy(shape, "volume shape for derived thermodynamics")?.nz,
        })
    };

    // Same fork as the generic path; the wind kernels merely moved out of
    // the independent closure (they already ran above).
    let (parcels, independent) = rayon::join(
        || -> Result<ParcelOutputs, TaskError> {
            if requirements.sb && requirements.ml && requirements.mu {
                let triplet = compute_cape_cin_triplet(grid, make_volume()?, surface_inputs, None)?;
                return Ok((Some(triplet.sb), Some(triplet.ml), Some(triplet.mu)));
            }
            let sb = if requirements.sb {
                Some(compute_sbcape_cin(
                    grid,
                    make_volume()?,
                    surface_inputs,
                    None,
                )?)
            } else {
                None
            };
            let ml = if requirements.ml {
                Some(compute_mlcape_cin(
                    grid,
                    make_volume()?,
                    surface_inputs,
                    None,
                )?)
            } else {
                None
            };
            let mu = if requirements.mu {
                Some(compute_mucape_cin(
                    grid,
                    make_volume()?,
                    surface_inputs,
                    None,
                )?)
            } else {
                None
            };
            Ok((sb, ml, mu))
        },
        || -> Result<IndependentOutputs, TaskError> {
            let mut outputs = IndependentOutputs::default();
            if requirements.dcape {
                outputs.dcape_jkg = Some(compute_dcape(grid, make_volume()?, surface_inputs)?);
            }
            if requirements.surface_thermo {
                outputs.surface_thermo = Some(compute_surface_thermo(grid, surface_inputs)?);
            }
            if requirements.lifted_index {
                outputs.lifted_index_c =
                    Some(compute_lifted_index(grid, make_volume()?, surface_inputs)?);
            }
            if requirements.lapse_rate_700_500 {
                outputs.lapse_rate_700_500_cpkm =
                    Some(compute_lapse_rate_700_500(grid, make_volume()?)?);
            }
            if requirements.lapse_rate_0_3km {
                outputs.lapse_rate_0_3km_cpkm = Some(compute_lapse_rate_0_3km(
                    grid,
                    make_volume()?,
                    surface_inputs,
                )?);
            }
            Ok(outputs)
        },
    );
    let (sb, ml, mu) = parcels.map_err(|err| err as Box<dyn std::error::Error>)?;
    let mut independent = independent.map_err(|err| err as Box<dyn std::error::Error>)?;
    independent.shear_01km_ms = wind_outputs.shear_01km_ms;
    independent.shear_06km_ms = wind_outputs.shear_06km_ms;
    independent.srh_01km_m2s2 = wind_outputs.srh_01km_m2s2;
    independent.srh_03km_m2s2 = wind_outputs.srh_03km_m2s2;
    independent.temperature_advection_700mb_cph = wind_outputs.temperature_advection_700mb_cph;
    independent.temperature_advection_850mb_cph = wind_outputs.temperature_advection_850mb_cph;

    let computed = assemble_derived_outputs(
        surface,
        recipes,
        &requirements,
        grid,
        (sb, ml, mu),
        independent,
    )?;
    Ok((computed, kept_winds))
}

/// Temperature advection (C/hour) at one pressure level, exactly as the
/// sequential lane computed it inline: slice (or log-p interpolate) the
/// level from the volume, run the kinematics kernel, scale C/s to C/hour.
/// The wind volumes ride as explicit slices so the store lane's phased
/// path can pass its taken-out copies; the generic path passes
/// `pressure.u_ms_3d()` / `pressure.v_ms_3d()` (same data, same math).
fn compute_temperature_advection_cph<P>(
    pressure: &P,
    u_ms_3d: &[f64],
    v_ms_3d: &[f64],
    grid: CalcGridShape,
    level_hpa: f64,
    dx_m: f64,
    dy_m: f64,
) -> Result<Vec<f64>, TaskError>
where
    P: PressureFieldSet,
{
    let missing = |what: &str| -> TaskError {
        format!("missing {level_hpa:.0} mb {what} slice in HRRR pressure bundle").into()
    };
    let temperature = pressure_level_slice_or_interp(
        pressure,
        pressure.temperature_c_3d(),
        level_hpa,
        grid.len(),
    )
    .ok_or_else(|| missing("temperature"))?;
    let u_ms = pressure_level_slice_or_interp(pressure, u_ms_3d, level_hpa, grid.len())
        .ok_or_else(|| missing("u-wind"))?;
    let v_ms = pressure_level_slice_or_interp(pressure, v_ms_3d, level_hpa, grid.len())
        .ok_or_else(|| missing("v-wind"))?;
    Ok(
        rustwx_calc::compute_temperature_advection(TemperatureAdvectionInputs {
            grid,
            temperature_2d: &temperature,
            u_2d_ms: &u_ms,
            v_2d_ms: &v_ms,
            dx_m,
            dy_m,
        })?
        .into_iter()
        .map(|value| value * 3600.0)
        .collect(),
    )
}

pub(super) fn compute_surface_only_derived_fields<S>(
    surface: &S,
    recipes: &[DerivedRecipe],
) -> Result<DerivedComputedFields, Box<dyn std::error::Error>>
where
    S: SurfaceFieldSet + Sync,
{
    if derived_compute_recipes_need_pressure(recipes) {
        return Err("surface-only derived compute received a pressure-dependent recipe".into());
    }
    compute_derived_fields_generic(surface, &EmptyPressureFields, recipes)
}

fn level_slice<'a>(
    values_3d: &'a [f64],
    levels_hpa: &[f64],
    target_hpa: f64,
    nxy: usize,
) -> Option<&'a [f64]> {
    let level_idx = levels_hpa
        .iter()
        .position(|level| (level - target_hpa).abs() < 0.25)?;
    let start = level_idx * nxy;
    let end = start + nxy;
    values_3d.get(start..end)
}

pub(super) fn pressure_level_slice_or_interp<P>(
    pressure: &P,
    values_3d: &[f64],
    target_hpa: f64,
    nxy: usize,
) -> Option<Vec<f64>>
where
    P: PressureFieldSet,
{
    if let Some(slice) = level_slice(values_3d, pressure.pressure_levels_hpa(), target_hpa, nxy) {
        return Some(slice.to_vec());
    }

    let pressure_3d_pa = pressure.pressure_3d_pa()?;
    let nz = pressure.pressure_levels_hpa().len();
    if nz == 0 || values_3d.len() != pressure_3d_pa.len() || values_3d.len() != nxy * nz {
        return None;
    }

    let log_target = target_hpa.ln();
    Some(
        (0..nxy)
            .into_par_iter()
            .map(|ij| {
                for k in 0..nz.saturating_sub(1) {
                    let idx0 = k * nxy + ij;
                    let idx1 = (k + 1) * nxy + ij;
                    let p0 = pressure_3d_pa[idx0] / 100.0;
                    let p1 = pressure_3d_pa[idx1] / 100.0;
                    if !p0.is_finite() || !p1.is_finite() || p0 <= 0.0 || p1 <= 0.0 {
                        continue;
                    }
                    if (p0 >= target_hpa && p1 <= target_hpa)
                        || (p0 <= target_hpa && p1 >= target_hpa)
                    {
                        let v0 = values_3d[idx0];
                        let v1 = values_3d[idx1];
                        let log0 = p0.ln();
                        let log1 = p1.ln();
                        let denom = log1 - log0;
                        if denom.abs() < 1.0e-12 {
                            return 0.5 * (v0 + v1);
                        }
                        let frac = (log_target - log0) / denom;
                        return v0 + frac * (v1 - v0);
                    }
                }
                f64::NAN
            })
            .collect(),
    )
}

fn compute_height_agl_3d_generic<S, P>(
    surface: &S,
    pressure: &P,
    grid: CalcGridShape,
    shape: VolumeShape,
) -> Vec<f64>
where
    S: SurfaceFieldSet,
    P: PressureFieldSet,
{
    let n2d = grid.len();
    let orog_m = surface.orog_m();
    let mut height_agl_3d = vec![0.0f64; pressure.gh_m_3d().len()];
    height_agl_3d
        .par_chunks_mut(n2d)
        .zip(pressure.gh_m_3d().par_chunks(n2d))
        .for_each(|(out, gh)| {
            for (ij, (dst, &value)) in out.iter_mut().zip(gh.iter()).enumerate() {
                *dst = (value - orog_m[ij]).max(0.0);
            }
        });

    // The monotonic clamp recurs on the level below, so sweep levels in
    // order but apply each level's clamp in parallel across the grid; the
    // per-point math is unchanged from the serial sweep.
    for k in 1..shape.nz {
        let (below, level) = height_agl_3d.split_at_mut(k * n2d);
        let prev = &below[(k - 1) * n2d..];
        level[..n2d]
            .par_iter_mut()
            .zip(prev.par_iter())
            .for_each(|(value, &prev_value)| {
                let min_height = prev_value + 1.0;
                if *value < min_height {
                    *value = min_height;
                }
            });
    }

    height_agl_3d
}

fn estimate_grid_spacing_m<S>(surface: &S) -> Result<(f64, f64), CalcError>
where
    S: SurfaceFieldSet,
{
    if surface.nx() < 2 || surface.ny() < 2 {
        return Err(CalcError::LengthMismatch {
            field: "grid_spacing",
            expected: 4,
            actual: surface.nx() * surface.ny(),
        });
    }

    let mut dx_sum = 0.0;
    let mut dx_count = 0usize;
    for y in 0..surface.ny() {
        let row_offset = y * surface.nx();
        for x in 0..(surface.nx() - 1) {
            let left = row_offset + x;
            let right = left + 1;
            let distance = haversine_m(
                surface.lat()[left],
                surface.lon()[left],
                surface.lat()[right],
                surface.lon()[right],
            );
            if distance.is_finite() && distance > 0.0 {
                dx_sum += distance;
                dx_count += 1;
            }
        }
    }

    let mut dy_sum = 0.0;
    let mut dy_count = 0usize;
    for y in 0..(surface.ny() - 1) {
        let row_offset = y * surface.nx();
        let next_row_offset = (y + 1) * surface.nx();
        for x in 0..surface.nx() {
            let top = row_offset + x;
            let bottom = next_row_offset + x;
            let distance = haversine_m(
                surface.lat()[top],
                surface.lon()[top],
                surface.lat()[bottom],
                surface.lon()[bottom],
            );
            if distance.is_finite() && distance > 0.0 {
                dy_sum += distance;
                dy_count += 1;
            }
        }
    }

    if dx_count == 0 || dy_count == 0 {
        return Err(CalcError::LengthMismatch {
            field: "grid_spacing",
            expected: 2,
            actual: 0,
        });
    }

    Ok((dx_sum / dx_count as f64, dy_sum / dy_count as f64))
}

pub(super) fn haversine_m(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> f64 {
    let lat1 = lat1_deg.to_radians();
    let lon1 = lon1_deg.to_radians();
    let lat2 = lat2_deg.to_radians();
    let lon2 = lon2_deg.to_radians();
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat * 0.5).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon * 0.5).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    6_371_000.0 * c
}
