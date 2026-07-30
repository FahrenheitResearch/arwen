//! Ingest-time derived + heavy precompute: decode the surface + pressure
//! thermodynamic pair from the already-fetched family GRIB bytes through
//! the SAME products decode lane the derived/heavy render paths use
//! (`rustwx_products::gridded::decode_store_thermo_pair_owned` — same
//! message matching, same moisture preference with specific humidity
//! first, same f64 precision), then run all 29 non-heavy derived recipes
//! and all 16 heavy ECAPE-class recipes through the products store lanes
//! (`compute_store_derived_grids_f32` / `compute_store_heavy_grids_f32` —
//! the exact prep + kernel dispatch the render lanes run, emitting the
//! same `as f32` cast grid per recipe), handing back f32 grids ready to
//! store as ordinary 2D variables named by recipe slug.
//!
//! No science lives here, and no input lives here either: an earlier
//! version assembled the compute inputs from the f32 extraction planes,
//! which rounded every input through f32 and — worse — sourced 2 m
//! moisture from dewpoint while the render lane's decode prefers the 2 m
//! specific-humidity message the HRRR sfc file actually carries. Stored
//! grids were therefore NOT the grids the render lane computes. Decoding
//! the pair through the render lane's own decoder makes the stored grids
//! bit-identical to a render-lane compute over the same files (the only
//! rounding is the single f64 -> f32 cast in the store lanes, which is
//! exactly the cast the render lane applies when it builds its `Field2D`).
//! Native CAPE planes (the heavy native-ratio denominators) ride on the
//! decoded `SurfaceFields` the same way they do in the render lane.
//!
//! Memory shape: [`decode_products_inputs`] also builds the ONE height-AGL
//! volume both compute stages consume (an in-place transform of the gh
//! volume — see `prepare_store_compute_inputs`), and the heavy stage takes
//! the inputs BY VALUE so the f64 thermo volumes are freed inside the
//! products lane as soon as the kernels are done with them.

use rustwx_products::derived::{
    StoreComputeInputs, StoreHeavyTiming, compute_store_derived_grids_f32,
    compute_store_heavy_grids_f32, prepare_store_compute_inputs,
};
use rustwx_products::gridded::decode_store_thermo_pair_owned;

/// One derived grid ready to store: variable name (the recipe slug), display
/// units, and full-grid row-major values.
pub struct DerivedGrid2D {
    pub name: &'static str,
    pub units: String,
    pub values: Vec<f32>,
}

/// The decoded products-side compute inputs for one hour: the f64
/// `SurfaceFields`/`PressureFields` pair plus the shared height-AGL volume
/// every store compute stage (derived and heavy) consumes. Decoded once
/// per hour by [`decode_products_inputs`].
pub struct ProductsComputeInputs {
    inner: StoreComputeInputs,
}

impl ProductsComputeInputs {
    /// Wrap an already-decoded thermo pair (the test seam; production goes
    /// through [`decode_products_inputs`]).
    pub fn new(
        surface: rustwx_products::gridded::SurfaceFields,
        pressure: rustwx_products::gridded::PressureFields,
    ) -> Self {
        Self {
            inner: prepare_store_compute_inputs(surface, pressure),
        }
    }

    pub fn surface(&self) -> &rustwx_products::gridded::SurfaceFields {
        self.inner.surface()
    }
}

/// Output of the heavy (ECAPE-class) compute stage: realized grids,
/// recipes skipped with the products lane's documented reason, the ECAPE
/// triplet's per-column failure count, and the products lane's per-kernel
/// timing breakdown for honest stage reporting.
pub struct HeavyGrids2D {
    pub grids: Vec<DerivedGrid2D>,
    pub skipped: Vec<(&'static str, String)>,
    pub ecape_failure_count: usize,
    pub timing: StoreHeavyTiming,
}

/// Decode the compute inputs from the fetched sfc + prs bytes via the
/// render lane's own thermo decoder. The optional pressure volumes are
/// skipped (no store-computed recipe consumes them); everything else —
/// including the native CAPE planes the heavy native-ratio recipes
/// divide by — is exactly what the render lane sees.
///
/// Takes the raw buffers by value so each is freed at its true last use
/// inside the decode (surface bytes after the surface decode, pressure
/// bytes once the parser owns its message copies) instead of riding
/// resident through both compute stages.
pub fn decode_products_inputs(
    surface_bytes: Vec<u8>,
    pressure_bytes: Vec<u8>,
) -> Result<ProductsComputeInputs, Box<dyn std::error::Error>> {
    decode_products_inputs_cropped(surface_bytes, pressure_bytes, None)
}

/// [`decode_products_inputs`] with an optional crop: when `crop` is set, the
/// decoded surface+pressure pair is sliced to that index block via the
/// products lane's own `crop_heavy_domain_with` (the same crop the extracted
/// field planes used), so the derived/heavy grids land on the identical cropped
/// grid as the stored 2D fields — provably in lock-step. The crop's
/// `(x_end, y_end)` must lie within the decoded surface shape (it does: it was
/// computed from the same GRIB grid).
pub fn decode_products_inputs_cropped(
    surface_bytes: Vec<u8>,
    pressure_bytes: Vec<u8>,
    crop: Option<rustwx_products::gridded::GridCrop>,
) -> Result<ProductsComputeInputs, Box<dyn std::error::Error>> {
    let (surface, pressure) = decode_store_thermo_pair_owned(surface_bytes, pressure_bytes)?;
    let (surface, pressure) = match crop {
        Some(crop) => {
            let cropped =
                rustwx_products::gridded::crop_heavy_domain_with(&surface, &pressure, crop)?;
            (cropped.surface, cropped.pressure)
        }
        None => (surface, pressure),
    };
    Ok(ProductsComputeInputs::new(surface, pressure))
}

/// Run the shared non-heavy compute pass on the decoded inputs. The store
/// lane emits each grid as f32 — the same cast the render lane applies
/// building its `Field2D`, so the stored f32 grid equals the render lane's
/// raster input bit for bit. The expensive recipe kernels are
/// rayon-parallel inside rustwx-calc.
///
/// `keep_winds` controls the u/v f64 wind volumes (~1.13 GB at HRRR size)
/// after the derived lane's wind-consuming kernels finish: `true` keeps
/// them resident on `inputs` for the heavy ECAPE stage (which reads them);
/// `false` frees them mid-stage — the no-heavy ingest, where nothing
/// downstream reads winds. Either way the computed grids are bit-identical
/// (the products lane pins that with a synthetic-hour test).
pub fn compute_derived_2d_from_inputs(
    inputs: &mut ProductsComputeInputs,
    keep_winds: bool,
) -> Result<Vec<DerivedGrid2D>, Box<dyn std::error::Error>> {
    let grids = compute_store_derived_grids_f32(&mut inputs.inner, keep_winds)?;
    Ok(grids
        .into_iter()
        .map(|grid| DerivedGrid2D {
            name: grid.slug,
            units: grid.units,
            values: grid.values,
        })
        .collect())
}

/// Run the heavy ECAPE-class compute pass (the heavy render lane's exact
/// prep + kernels via `compute_store_heavy_grids_f32`) on the decoded
/// inputs, consuming them — heavy is the last compute stage, and the
/// products lane frees the f64 thermo volumes as soon as the kernels are
/// done. The native-CAPE ratio recipes realize only when the decode found
/// the matching native plane; otherwise they come back in `skipped` with
/// the products lane's documented reason.
pub fn compute_heavy_2d_from_inputs(
    inputs: ProductsComputeInputs,
) -> Result<HeavyGrids2D, Box<dyn std::error::Error>> {
    let heavy = compute_store_heavy_grids_f32(inputs.inner)?;
    Ok(HeavyGrids2D {
        grids: heavy
            .grids
            .into_iter()
            .map(|grid| DerivedGrid2D {
                name: grid.slug,
                units: grid.units,
                values: grid.values,
            })
            .collect(),
        skipped: heavy
            .skipped
            .into_iter()
            .map(|skip| (skip.slug, skip.reason))
            .collect(),
        ecape_failure_count: heavy.ecape_failure_count,
        timing: heavy.timing,
    })
}
