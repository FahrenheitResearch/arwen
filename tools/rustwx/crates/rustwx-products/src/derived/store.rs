//! Store-ingest derived lane: compute every supported derived recipe grid
//! (non-heavy and heavy/ECAPE) from already-assembled surface + pressure
//! bundles, with no fetching, projection, or rendering involved.
//!
//! This is the same compute path the render/query lanes use. Non-heavy:
//! [`compute_derived_fields_generic`] does the shared input prep (pressure
//! volume, height-AGL assembly, grid spacing) once for all recipes, and
//! [`derived_query_field_from_computed`] maps each recipe to its grid and
//! units. Heavy: [`prepare_heavy_volume`] +
//! [`compute_ecape_map_fields_with_prepared_volume`] — the exact prep +
//! kernel dispatch the ECAPE/heavy render lane runs — produce the panel
//! fields, which are mapped to recipe slugs by their artifact slugs (the
//! two namespaces are identical for heavy recipes). Grids stored at ingest
//! are therefore bit-identical to what the corresponding render lane would
//! compute from the same inputs.

use std::time::Instant;

use rayon::prelude::*;
use rustwx_calc::{GridShape as CalcGridShape, VolumeShape};

use crate::ecape::{EcapeMapFieldsTiming, compute_ecape_map_fields_with_prepared_volume_timed};
use crate::gridded::{
    PreparedHeavyVolume, PressureFields, SurfaceFields, compute_height_agl_3d,
    prepare_heavy_volume_timed,
};

use super::compute::{
    TakenWindVolumes, compute_derived_fields_generic, compute_store_derived_fields_phased,
};
use super::inventory::supported_derived_recipe_inventory;
use super::query::{derived_query_field_from_computed, derived_query_field_take_from_computed};
use super::recipes::DerivedRecipe;

/// One derived grid computed for store ingest: the recipe slug (also the
/// store variable name), display units, and the full-grid row-major values.
#[derive(Debug, Clone)]
pub struct StoreDerivedGrid {
    pub slug: &'static str,
    pub units: String,
    pub values: Vec<f64>,
}

/// Slugs of every supported non-heavy derived recipe, in inventory order.
/// These are the grids [`compute_store_derived_grids`] realizes; heavy
/// (ECAPE-class) recipes stay out of the ingest path.
pub fn store_derived_recipe_slugs() -> Vec<&'static str> {
    supported_derived_recipe_inventory()
        .iter()
        .filter(|entry| !entry.heavy)
        .map(|entry| entry.slug)
        .collect()
}

/// Compute all non-heavy derived recipe grids in one shared pass over the
/// prepared surface + pressure inputs. Fails as a whole if the shared prep
/// or any recipe fails: ingest inputs carry the full field set, so a
/// missing dependency is a wiring bug, not an expected degradation.
pub fn compute_store_derived_grids(
    surface: &SurfaceFields,
    pressure: &PressureFields,
) -> Result<Vec<StoreDerivedGrid>, Box<dyn std::error::Error>> {
    let recipes = store_derived_recipe_slugs()
        .into_iter()
        .map(|slug| DerivedRecipe::parse(slug).map_err(std::io::Error::other))
        .collect::<Result<Vec<_>, _>>()?;
    let mut computed = compute_derived_fields_generic(surface, pressure, &recipes)?;
    let mut grids = Vec::with_capacity(recipes.len());
    for recipe in recipes {
        let query =
            derived_query_field_from_computed(surface.nx, surface.ny, recipe, &mut computed)?;
        grids.push(StoreDerivedGrid {
            slug: recipe.slug(),
            units: query.units,
            values: query.values,
        });
    }
    Ok(grids)
}

/// One derived grid in its stored form: the recipe slug, display units,
/// and f32 values — the exact `as f32` cast the render lane applies when
/// it builds its raster input, applied per recipe so the full f64 grid set
/// is never duplicated.
#[derive(Debug, Clone)]
pub struct StoreDerivedGridF32 {
    pub slug: &'static str,
    pub units: String,
    pub values: Vec<f32>,
}

/// Shared inputs for the store-ingest compute stages: the decoded thermo
/// pair plus ONE height-AGL volume shared by the derived and heavy lanes.
///
/// Memory shape: both lanes assemble bit-identical height-AGL volumes from
/// `gh_m_3d` (same `(gh - orog).max(0)` element op, same +1 m monotonic
/// sweep — see [`prepare_store_compute_inputs`]), and neither lane reads
/// `gh_m_3d` for anything else. Building the volume once by transforming
/// gh IN PLACE removes both the per-lane 580 MB assembly and gh's own
/// residency across the compute window. The proxy-orography case (no orog
/// message in the surface file) keeps the historical per-lane assembly:
/// there the two lanes derive different orography proxies, so no shared
/// volume exists.
pub struct StoreComputeInputs {
    surface: SurfaceFields,
    pressure: PressureFields,
    height_agl_3d: Option<Vec<f64>>,
    prepare_height_agl_ms: u128,
}

impl StoreComputeInputs {
    pub fn surface(&self) -> &SurfaceFields {
        &self.surface
    }

    pub fn pressure(&self) -> &PressureFields {
        &self.pressure
    }
}

/// Build [`StoreComputeInputs`] from the decoded thermo pair: move
/// `gh_m_3d` out of the pressure fields and transform it in place into the
/// height-AGL volume both compute lanes consume.
///
/// Bit-identity: the per-element arithmetic is exactly the assembly both
/// lanes run today — `(gh - orog_m[ij]).max(0.0)` per cell, then for each
/// level above the first, `value = max(value, below + 1.0)` expressed as
/// the same compare-and-set. Each output element depends only on its own
/// column inputs, so the parallel sweep cannot reorder any float op.
pub fn prepare_store_compute_inputs(
    surface: SurfaceFields,
    mut pressure: PressureFields,
) -> StoreComputeInputs {
    if surface.orog_is_proxy {
        return StoreComputeInputs {
            surface,
            pressure,
            height_agl_3d: None,
            prepare_height_agl_ms: 0,
        };
    }
    let started = Instant::now();
    let n2d = surface.nx * surface.ny;
    let mut height_agl_3d = std::mem::take(&mut pressure.gh_m_3d);
    let orog_m = surface.orog_m.as_slice();
    height_agl_3d.par_chunks_mut(n2d).for_each(|level| {
        for (ij, value) in level.iter_mut().enumerate() {
            *value = (*value - orog_m[ij]).max(0.0);
        }
    });
    let nz = pressure.pressure_levels_hpa.len();
    for k in 1..nz {
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
    StoreComputeInputs {
        surface,
        pressure,
        height_agl_3d: Some(height_agl_3d),
        prepare_height_agl_ms: started.elapsed().as_millis(),
    }
}

/// [`compute_store_derived_grids`] through the shared inputs, emitting f32
/// grids recipe by recipe (take semantics — each computed f64 grid is
/// freed as soon as its f32 twin exists). Identical kernels, identical
/// recipe order, identical stored values — pinned bit-exactly against the
/// generic path by `phased_store_derived_compute_matches_generic_path`.
///
/// Memory shape: the compute runs PHASED (see
/// [`compute_store_derived_fields_phased`]) — every wind-consuming kernel
/// first, after which the u/v f64 volumes (~1.13 GB at HRRR size) leave
/// RAM for the rest of the derived window when `keep_winds` is false (the
/// no-heavy ingest). With `keep_winds` true the volumes return to
/// `inputs` untouched for the heavy ECAPE stage, which does read them.
pub fn compute_store_derived_grids_f32(
    inputs: &mut StoreComputeInputs,
    keep_winds: bool,
) -> Result<Vec<StoreDerivedGridF32>, Box<dyn std::error::Error>> {
    let recipes = store_derived_recipe_slugs()
        .into_iter()
        .map(|slug| DerivedRecipe::parse(slug).map_err(std::io::Error::other))
        .collect::<Result<Vec<_>, _>>()?;
    let winds = TakenWindVolumes {
        u_ms_3d: std::mem::take(&mut inputs.pressure.u_ms_3d),
        v_ms_3d: std::mem::take(&mut inputs.pressure.v_ms_3d),
    };
    let (mut computed, kept_winds) = compute_store_derived_fields_phased(
        &inputs.surface,
        &inputs.pressure,
        winds,
        &recipes,
        inputs.height_agl_3d.as_deref(),
        keep_winds,
    )?;
    if let Some(kept) = kept_winds {
        inputs.pressure.u_ms_3d = kept.u_ms_3d;
        inputs.pressure.v_ms_3d = kept.v_ms_3d;
    }
    let mut grids = Vec::with_capacity(recipes.len());
    for recipe in recipes {
        let query = derived_query_field_take_from_computed(
            inputs.surface.nx,
            inputs.surface.ny,
            recipe,
            &mut computed,
        )?;
        grids.push(StoreDerivedGridF32 {
            slug: recipe.slug(),
            units: query.units,
            values: query.values.par_iter().map(|&value| value as f32).collect(),
        });
    }
    Ok(grids)
}

/// One heavy recipe skipped at store ingest, with the documented reason
/// (the blocker pattern: record why, never invent a substitute).
#[derive(Debug, Clone)]
pub struct StoreHeavySkip {
    pub slug: &'static str,
    pub reason: String,
}

/// Where the heavy stage's wall time went: the shared height-AGL prep plus
/// the per-kernel breakdown from the ECAPE lane. Observation only — the
/// kernels and their order are identical to the untimed render path.
#[derive(Debug, Clone, Copy, Default)]
pub struct StoreHeavyTiming {
    pub prepare_height_agl_ms: u128,
    pub kernels: EcapeMapFieldsTiming,
}

/// Output of [`compute_store_heavy_grids`]: realized heavy recipe grids in
/// inventory order, recipes skipped with reasons, the ECAPE triplet's
/// per-column failure count (cells whose parcel ascent failed — those carry
/// NaN in the grids, same as the render lane), and the stage timing.
#[derive(Debug, Clone)]
pub struct StoreHeavyGrids {
    pub grids: Vec<StoreDerivedGrid>,
    pub skipped: Vec<StoreHeavySkip>,
    pub ecape_failure_count: usize,
    pub timing: StoreHeavyTiming,
}

/// Slugs of every heavy (ECAPE-class) derived recipe, in inventory order.
/// These are the grids [`compute_store_heavy_grids`] realizes.
pub fn store_heavy_recipe_slugs() -> Vec<&'static str> {
    supported_derived_recipe_inventory()
        .iter()
        .filter(|entry| entry.heavy)
        .map(|entry| entry.slug)
        .collect()
}

/// The heavy recipes that depend on the model's *native* CAPE planes
/// (decoded from the surface GRIB's CAPE messages). These are the only
/// heavy recipes allowed to skip: the surface file may simply not carry a
/// matching CAPE message, the same optionality the render lane has.
const NATIVE_RATIO_SLUGS: [(&str, &str); 3] = [
    ("sb_ecape_native_cape_ratio", "surface-based (level type 1)"),
    ("ml_ecape_native_cape_ratio", "0-90 mb mixed layer"),
    ("mu_ecape_native_cape_ratio", "0-255 mb most-unstable layer"),
];

/// Compute every heavy (ECAPE-class) derived recipe grid through the
/// existing heavy lane: `prepare_heavy_volume` (height-AGL assembly, no
/// pressure broadcast) feeding `compute_ecape_map_fields_with_prepared_volume`
/// (ECAPE triplet, wind diagnostics, ML LCL, and the experimental
/// composites). The native-CAPE ratio recipes realize only when the
/// corresponding `surface.native_*cape_jkg` plane is present; their absence
/// is recorded as a skip, not an error. Any other missing recipe is a
/// wiring bug and fails the whole call.
pub fn compute_store_heavy_grids(
    surface: &SurfaceFields,
    pressure: &PressureFields,
) -> Result<StoreHeavyGrids, Box<dyn std::error::Error>> {
    let (prepared, prep_timing) = prepare_heavy_volume_timed(surface, pressure, false)?;
    let (fields, ecape_failure_count, kernel_timing) =
        compute_ecape_map_fields_with_prepared_volume_timed(surface, pressure, &prepared)?;
    drop(prepared);
    let timing = StoreHeavyTiming {
        prepare_height_agl_ms: prep_timing.prepare_height_agl_ms,
        kernels: kernel_timing,
    };

    let mut remaining: Vec<crate::shared_context::WeatherPanelField> = fields;
    let mut grids = Vec::new();
    let mut skipped = Vec::new();
    for slug in store_heavy_recipe_slugs() {
        match remaining
            .iter()
            .position(|field| field.artifact_slug() == slug)
        {
            Some(index) => {
                let field = remaining.swap_remove(index);
                grids.push(StoreDerivedGrid {
                    slug,
                    units: field.units,
                    values: field.values,
                });
            }
            None => match NATIVE_RATIO_SLUGS.iter().find(|(have, _)| *have == slug) {
                Some((_, layer)) => skipped.push(StoreHeavySkip {
                    slug,
                    reason: format!(
                        "surface inputs carry no native {layer} CAPE plane \
                         (no matching CAPE message in the surface GRIB)"
                    ),
                }),
                None => {
                    return Err(format!(
                        "heavy recipe '{slug}' missing from the ECAPE field set; \
                         this is a wiring bug, not a degradation"
                    )
                    .into());
                }
            },
        }
    }
    Ok(StoreHeavyGrids {
        grids,
        skipped,
        ecape_failure_count,
        timing,
    })
}

/// Output of [`compute_store_heavy_grids_f32`] — the f32 sibling of
/// [`StoreHeavyGrids`], same realized order / skip semantics / timing.
#[derive(Debug, Clone)]
pub struct StoreHeavyGridsF32 {
    pub grids: Vec<StoreDerivedGridF32>,
    pub skipped: Vec<StoreHeavySkip>,
    pub ecape_failure_count: usize,
    pub timing: StoreHeavyTiming,
}

/// [`compute_store_heavy_grids`] through the shared inputs, consuming them
/// (the heavy stage is the store lane's last compute). The shared
/// height-AGL volume moves into the prepared volume instead of being
/// reassembled; `gh_m_3d` (never read by the kernels) is freed before the
/// long kernel window; the thermo volumes and surface planes are freed
/// before the recipe mapping; and each f64 output grid is freed as soon as
/// its f32 twin exists. Identical kernels, identical values.
pub fn compute_store_heavy_grids_f32(
    mut inputs: StoreComputeInputs,
) -> Result<StoreHeavyGridsF32, Box<dyn std::error::Error>> {
    let grid = CalcGridShape::new(inputs.surface.nx, inputs.surface.ny)?;
    let shape = VolumeShape::new(grid, inputs.pressure.pressure_levels_hpa.len())?;
    let (height_agl_3d, prepare_height_agl_ms) = match inputs.height_agl_3d.take() {
        Some(shared) => (shared, inputs.prepare_height_agl_ms),
        None => {
            // Proxy-orography fallback: the historical per-lane assembly
            // (the derived lane derives a different proxy, so no shared
            // volume exists in this case).
            let started = Instant::now();
            let volume = compute_height_agl_3d(&inputs.surface, &inputs.pressure, grid, shape);
            (volume, started.elapsed().as_millis())
        }
    };
    // The heavy kernels consume height-AGL, never gh; in the shared path
    // gh is already empty (moved), in the fallback path free it now.
    inputs.pressure.gh_m_3d = Vec::new();
    let pressure_levels_pa = inputs
        .pressure
        .pressure_levels_hpa
        .iter()
        .map(|level_hpa| level_hpa * 100.0)
        .collect::<Vec<_>>();
    let prepared = PreparedHeavyVolume {
        grid,
        shape,
        pressure_levels_pa,
        pressure_3d_pa: None,
        height_agl_3d,
    };
    let (fields, ecape_failure_count, kernel_timing) =
        compute_ecape_map_fields_with_prepared_volume_timed(
            &inputs.surface,
            &inputs.pressure,
            &prepared,
        )?;
    drop(prepared);
    drop(inputs);
    let timing = StoreHeavyTiming {
        prepare_height_agl_ms,
        kernels: kernel_timing,
    };

    let mut remaining: Vec<crate::shared_context::WeatherPanelField> = fields;
    let mut grids = Vec::new();
    let mut skipped = Vec::new();
    for slug in store_heavy_recipe_slugs() {
        match remaining
            .iter()
            .position(|field| field.artifact_slug() == slug)
        {
            Some(index) => {
                let field = remaining.swap_remove(index);
                grids.push(StoreDerivedGridF32 {
                    slug,
                    units: field.units,
                    values: field.values.par_iter().map(|&value| value as f32).collect(),
                });
            }
            None => match NATIVE_RATIO_SLUGS.iter().find(|(have, _)| *have == slug) {
                Some((_, layer)) => skipped.push(StoreHeavySkip {
                    slug,
                    reason: format!(
                        "surface inputs carry no native {layer} CAPE plane \
                         (no matching CAPE message in the surface GRIB)"
                    ),
                }),
                None => {
                    return Err(format!(
                        "heavy recipe '{slug}' missing from the ECAPE field set; \
                         this is a wiring bug, not a degradation"
                    )
                    .into());
                }
            },
        }
    }
    Ok(StoreHeavyGridsF32 {
        grids,
        skipped,
        ecape_failure_count,
        timing,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gridded::mixing_ratio_from_dewpoint_k;

    const NX: usize = 3;
    const NY: usize = 2;
    const NXY: usize = NX * NY;
    /// Descending pressure (ground up) — the order the decode lane aligns
    /// to. 850/700/500 are present so the advection and lapse recipes
    /// realize.
    const LEVELS: [u16; 5] = [1000, 925, 850, 700, 500];

    /// A warm, moist, sheared synthetic hour (mirrors the rw-ingest wiring
    /// test fixture): enough instability that sbcape is finite and nonzero
    /// somewhere, so the bit-compares are not all-NaN-trivial.
    fn synthetic_pair() -> (SurfaceFields, PressureFields) {
        let mut lat = Vec::with_capacity(NXY);
        let mut lon = Vec::with_capacity(NXY);
        for y in 0..NY {
            for x in 0..NX {
                lat.push(35.0 + 0.01 * y as f64);
                lon.push(-97.0 + 0.01 * x as f64);
            }
        }
        let psfc_pa: Vec<f64> = (0..NXY).map(|ij| 97_500.0 + 50.0 * ij as f64).collect();
        let td2_k: Vec<f64> = (0..NXY).map(|ij| 295.0 + 0.2 * ij as f64).collect();
        let q2_kgkg: Vec<f64> = psfc_pa
            .iter()
            .zip(td2_k.iter())
            .map(|(&psfc, &td_k)| mixing_ratio_from_dewpoint_k(psfc / 100.0, td_k))
            .collect();
        let volume = |base: f64, dk: f64, dij: f64| -> Vec<f64> {
            (0..LEVELS.len())
                .flat_map(|k| (0..NXY).map(move |ij| base + dk * k as f64 + dij * ij as f64))
                .collect()
        };
        let qvapor_kgkg_3d: Vec<f64> = LEVELS
            .iter()
            .enumerate()
            .flat_map(|(k, &level)| {
                (0..NXY).map(move |ij| {
                    let td_k = 294.0 - 8.5 * k as f64 + 0.15 * ij as f64;
                    mixing_ratio_from_dewpoint_k(f64::from(level), td_k)
                })
            })
            .collect();
        (
            SurfaceFields {
                lat,
                lon,
                nx: NX,
                ny: NY,
                projection: None,
                psfc_pa,
                orog_m: (0..NXY).map(|ij| 300.0 + 5.0 * ij as f64).collect(),
                orog_is_proxy: false,
                t2_k: (0..NXY).map(|ij| 302.0 + 0.3 * ij as f64).collect(),
                q2_kgkg,
                u10_ms: (0..NXY).map(|ij| 2.0 + 0.5 * ij as f64).collect(),
                v10_ms: (0..NXY).map(|ij| 5.0 - 0.25 * ij as f64).collect(),
                native_sbcape_jkg: None,
                native_mlcape_jkg: None,
                native_mucape_jkg: None,
                native_pblh_m: None,
            },
            PressureFields {
                pressure_levels_hpa: LEVELS.iter().map(|&level| f64::from(level)).collect(),
                pressure_3d_pa: None,
                temperature_c_3d: volume(301.0 - 273.15, -7.5, 0.2),
                qvapor_kgkg_3d,
                u_ms_3d: volume(3.0, 4.0, 0.3),
                v_ms_3d: volume(6.0, 2.5, -0.2),
                // Heights well above the orography and rising far faster
                // than the +1 m monotonic clamp.
                gh_m_3d: volume(400.0, 1400.0, 2.0),
                omega_pa_s_3d: None,
                absolute_vorticity_s_3d: None,
                cloud_liquid_kgkg_3d: None,
                cloud_ice_kgkg_3d: None,
                rain_kgkg_3d: None,
                snow_kgkg_3d: None,
                graupel_kgkg_3d: None,
            },
        )
    }

    /// The phased store compute (wind kernels first, winds freed or handed
    /// back before the parcel pass) must be bit-identical to the generic
    /// single-join path, in BOTH wind modes, across all 29 recipes — this
    /// is the determinism contract behind the early wind free.
    #[test]
    fn phased_store_derived_compute_matches_generic_path() {
        let (surface, pressure) = synthetic_pair();
        let legacy = compute_store_derived_grids(&surface, &pressure)
            .expect("generic-path store compute succeeds");
        assert_eq!(legacy.len(), 29, "all 29 non-heavy recipes realize");

        let mut keep = prepare_store_compute_inputs(surface.clone(), pressure.clone());
        let kept_grids =
            compute_store_derived_grids_f32(&mut keep, true).expect("phased compute (keep winds)");
        assert!(
            !keep.pressure().u_ms_3d.is_empty() && !keep.pressure().v_ms_3d.is_empty(),
            "keep_winds must hand the wind volumes back for the heavy stage"
        );

        let mut free = prepare_store_compute_inputs(surface, pressure);
        let freed_grids =
            compute_store_derived_grids_f32(&mut free, false).expect("phased compute (free winds)");
        assert!(
            free.pressure().u_ms_3d.is_empty() && free.pressure().v_ms_3d.is_empty(),
            "freeing must leave the wind volumes empty"
        );

        assert_eq!(legacy.len(), kept_grids.len());
        assert_eq!(legacy.len(), freed_grids.len());
        let mut finite = 0usize;
        for ((reference, kept), freed) in legacy.iter().zip(&kept_grids).zip(&freed_grids) {
            assert_eq!(reference.slug, kept.slug);
            assert_eq!(reference.slug, freed.slug);
            assert_eq!(reference.units, kept.units, "{}: units", reference.slug);
            assert_eq!(reference.units, freed.units, "{}: units", reference.slug);
            assert_eq!(reference.values.len(), kept.values.len());
            assert_eq!(reference.values.len(), freed.values.len());
            for (i, &expected_f64) in reference.values.iter().enumerate() {
                let expected = expected_f64 as f32;
                assert_eq!(
                    expected.to_bits(),
                    kept.values[i].to_bits(),
                    "{}[{i}]: keep-winds mismatch",
                    reference.slug
                );
                assert_eq!(
                    expected.to_bits(),
                    freed.values[i].to_bits(),
                    "{}[{i}]: free-winds mismatch",
                    reference.slug
                );
                if expected.is_finite() {
                    finite += 1;
                }
            }
        }
        assert!(
            finite > 0,
            "synthetic hour must produce finite values or the compare is NaN-trivial"
        );
    }

    /// The heavy store inventory is the 16 ECAPE-class recipes, in
    /// inventory order, and every slug maps 1:1 onto a heavy-lane panel
    /// artifact slug (the mapping `compute_store_heavy_grids` relies on).
    #[test]
    fn heavy_store_slugs_pin_the_sixteen_ecape_recipes() {
        assert_eq!(
            store_heavy_recipe_slugs(),
            vec![
                "sbecape",
                "mlecape",
                "muecape",
                "sb_ecape_derived_cape_ratio",
                "ml_ecape_derived_cape_ratio",
                "mu_ecape_derived_cape_ratio",
                "sb_ecape_native_cape_ratio",
                "ml_ecape_native_cape_ratio",
                "mu_ecape_native_cape_ratio",
                "sbncape",
                "sbecin",
                "mlecin",
                "ecape_scp",
                "ecape_ehi_0_1km",
                "ecape_ehi_0_3km",
                "ecape_stp",
            ]
        );
    }

    /// Heavy + non-heavy store slugs partition the supported inventory:
    /// nothing is double-stored and nothing is dropped.
    #[test]
    fn heavy_and_derived_store_slugs_partition_the_inventory() {
        let mut combined = store_derived_recipe_slugs();
        combined.extend(store_heavy_recipe_slugs());
        combined.sort_unstable();
        let mut inventory: Vec<&str> = supported_derived_recipe_inventory()
            .iter()
            .map(|entry| entry.slug)
            .collect();
        inventory.sort_unstable();
        assert_eq!(combined, inventory);
    }
}
