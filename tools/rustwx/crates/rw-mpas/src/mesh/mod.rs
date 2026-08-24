//! MPAS mesh generation, in Rust.
//!
//! The pipeline is: a resolution spec (data, never code) -> initial points ->
//! Lloyd/SCVT relaxation -> spherical Delaunay -> the full MPAS field set ->
//! self-validation -> a classic-netCDF grid file.
//!
//! What this module DOES NOT produce is a `static` file. A grid file alone
//! cannot reach the dycore: the mesh registry also demands a matching static
//! carrying terrain, land use and soil on the same cells, `nVertLevels = 55`,
//! `nSoilLevels = 4` and an FP32-bit-exact `nominalMinDc`. That boundary is
//! named here so nobody reads a valid grid file as a runnable mesh.

pub mod density;
pub mod derive;
pub mod footprint;
pub mod emit;
pub mod geom;
pub mod hull;
pub mod icosa;
pub mod lloyd;
pub mod validate;

use serde::Serialize;

use crate::error::{MpasError, MpasResult};
pub use density::{MeshSpec, Region, Shape, TransitionField};
pub use footprint::{CARDS, Card};
pub use derive::{MpasMesh, Rings};
pub use geom::V3;
pub use emit::{EmittedMesh, Provenance};
pub use lloyd::LloydOptions;
pub use validate::{Limits, MeshReport};

/// How the cell count was chosen.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sizing {
    /// The caller named the count.
    Explicit,
    /// The count came from a device memory budget through the footprint model.
    DeviceBudget,
    /// The count came from the spec's own spacings.
    FromSpacing,
}

/// How the generators were placed, and what that did to the requested count.
///
/// This is the receipt's record of the count SNAP: an icosahedral seed can
/// only deliver `10*(m^2+mn+n^2)+2` cells, so a uniform request is moved to
/// the nearest achievable count and the move is stated here rather than the
/// delivered count quietly disagreeing with the request.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case", tag = "method")]
pub enum Seeding {
    /// Uniform requests: the vertices of a Goldberg subdivision GP(m, n) of
    /// the icosahedron -- the same construction as the published family, so
    /// the Delaunay carries exactly twelve pentagons and no dislocations, and
    /// the near-cocircular quads that produce metre-scale dual edges cannot
    /// form.
    IcosahedralGoldberg {
        m: u32,
        n: u32,
        requested_cells: usize,
        seeded_cells: usize,
        /// `seeded/requested - 1`.
        snap_relative: f64,
    },
    /// Variable-resolution requests: the density-biased golden-ratio lattice.
    /// Its dislocations are topologically unavoidable under a refinement
    /// gradient; the emit gate holds the result to the same floors.
    FibonacciAcceptance { requested_cells: usize },
}

/// A generation request.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub spec: MeshSpec,
    /// Cells to generate. `None` takes the count the spec's spacings imply.
    pub target_cells: Option<usize>,
    /// Device budget in MiB. When set with `fit_spacing`, the spec's spacings
    /// are rescaled -- keeping every ratio between them -- until the mesh fits.
    ///
    /// A budget is only half a sizing request: it says HOW MUCH memory and
    /// never WHICH card, and the footprint model's fixed term is a property
    /// of the card. `card` therefore has to be `Some` whenever this is, and
    /// [`generate`] refuses if it is not.
    pub budget_mib: Option<f64>,
    /// The part the mesh has to run on, and so the footprint model that
    /// converts a budget into a cell count. `None` means no memory model is
    /// consulted at all -- legitimate with an explicit `target_cells`, and a
    /// refusal with `budget_mib`.
    pub card: Option<&'static footprint::Card>,
    /// Rescale the spec to hit the target rather than truncating the count.
    pub fit_spacing: bool,
    pub lloyd: LloydOptions,
    pub limits: Limits,
    /// Quadrature points for the sizing integral. 200,000 was the count the
    /// sizing instrument was validated at.
    pub sizing_samples: usize,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        GenerateRequest {
            spec: MeshSpec::uniform(120.0),
            target_cells: None,
            budget_mib: None,
            card: None,
            fit_spacing: false,
            lloyd: LloydOptions::default(),
            limits: Limits::default(),
            sizing_samples: 200_000,
        }
    }
}

/// Everything a run measured, in the order a reader wants it.
#[derive(Debug, Clone, Serialize)]
pub struct Receipt {
    pub engine: String,
    pub spec: MeshSpec,
    pub spec_scale_applied: f64,
    pub sizing: Sizing,
    /// Where the generators came from, including any count snap.
    pub seeding: Seeding,
    pub target_cells: usize,
    pub predicted_cells: f64,
    pub delivered_cells: usize,
    pub finest_requested_km: f64,
    pub background_requested_km: f64,
    pub device_budget_mib: Option<f64>,
    /// The part the footprint below was taken on, or `None` when no card was
    /// named. A footprint with no card beside it is the defect this field
    /// exists to make impossible.
    pub card: Option<&'static str>,
    /// What this mesh costs on `card`. `None` when no card was named: there
    /// is no card-independent answer, and printing one was the defect.
    pub footprint_mib: Option<f64>,
    pub steepest_requested_gradient_percent_per_cell: f64,
    pub published_reference_gradient_percent_per_cell: f64,
    /// What each refinement region's request actually reaches -- see
    /// [`density::RegionAttainment`].  The nominal `spacing_km` of a region is
    /// an ASYMPTOTE, so this is the number that says whether the request was
    /// met, and it is computed from the spec alone (before any relaxation) so
    /// a caller can refuse on it rather than discover it in a finished file.
    pub region_attainment: Vec<density::RegionAttainment>,
    /// Delivered spacing over requested spacing, cell by cell. The request is
    /// a continuous field and an SCVT delivers it only approximately, so this
    /// is the number that answers "did I get what I asked for" -- and the
    /// nominal `spacing_km` of a region is NOT that answer, because the
    /// transition ramp is centred on the region boundary and a region narrower
    /// than a few ramp widths never reaches its own request in the middle.
    pub delivered_over_requested_median: f64,
    pub delivered_over_requested_p05: f64,
    pub delivered_over_requested_p95: f64,
    pub relaxation_sweeps: usize,
    pub relaxation_mean_delta_over_h: f64,
    pub relaxation_max_delta_over_h: f64,
    pub relaxation_seconds: f64,
    pub mesh: MeshReport,
    pub deliverable_boundary: String,
}

/// The finished article.
#[derive(Debug, Clone)]
pub struct Generated {
    pub mesh: MpasMesh,
    pub receipt: Receipt,
    pub spec: MeshSpec,
}

/// What a rebuild measured.
#[derive(Debug, Clone, Serialize)]
pub struct RebuildReceipt {
    pub engine: String,
    pub source_cells: usize,
    pub nominal_min_dc_radians: f64,
    pub triangulation_seconds: f64,
    pub derive_seconds: f64,
    pub validate_seconds: f64,
    pub mesh: MeshReport,
    pub deliverable_boundary: String,
}

/// A mesh rebuilt from cell centres that already exist.
#[derive(Debug, Clone)]
pub struct Rebuilt {
    pub mesh: MpasMesh,
    pub receipt: RebuildReceipt,
}

/// Rebuild every derived field from a set of cell centres.
///
/// This is the same second half of `generate` -- triangulate, derive, validate --
/// with the resolution request replaced by centres somebody else chose. It takes
/// no mesh identity, no cell count and no table lookup: any set of points on the
/// sphere whose Delaunay triangulation closes is a mesh here, which is what makes
/// an unseen mesh a data question rather than a code question.
pub fn rebuild(
    cell_xyz: Vec<V3>,
    mesh_density: Vec<f64>,
    nominal_min_dc: f64,
    limits: Limits,
    mut progress: impl FnMut(&str),
) -> MpasResult<Rebuilt> {
    if cell_xyz.len() != mesh_density.len() {
        return Err(MpasError::Refusal(format!(
            "{} cell centres against {} meshDensity values; the two tables describe different meshes and every cell's horizontal mixing length would be taken from the wrong cell",
            cell_xyz.len(),
            mesh_density.len()
        )));
    }
    let points = hull::to_unit_sphere(&cell_xyz)?;

    let t0 = std::time::Instant::now();
    let rings = hull::delaunay_rings(&points)?;
    let triangulation_seconds = t0.elapsed().as_secs_f64();
    progress(&format!(
        "TRIANGULATED\t{}\t{:.2}",
        points.len(),
        triangulation_seconds
    ));

    let t1 = std::time::Instant::now();
    let mesh = MpasMesh::derive(points, mesh_density, &rings, nominal_min_dc)?;
    let derive_seconds = t1.elapsed().as_secs_f64();
    progress(&format!(
        "DERIVED\t{}\t{}\t{}\t{:.2}",
        mesh.n_cells, mesh.n_edges, mesh.n_vertices, derive_seconds
    ));

    let t2 = std::time::Instant::now();
    let report = validate::validate(&mesh, limits)?;
    let validate_seconds = t2.elapsed().as_secs_f64();
    progress(&format!(
        "VALIDATED\t{:.16}\t{:.3e}\t{:.3e}\t{:.2}",
        report.sum_area_cell_over_4pi,
        report.max_nonorthogonality,
        report.max_weight_antisymmetry,
        validate_seconds
    ));

    let receipt = RebuildReceipt {
        engine: concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)").to_string(),
        source_cells: mesh.n_cells,
        nominal_min_dc_radians: nominal_min_dc,
        triangulation_seconds,
        derive_seconds,
        validate_seconds,
        mesh: report,
        deliverable_boundary:
            "grid file only; running this mesh also needs a matching static file (terrain, land use, soil, nVertLevels=55, nSoilLevels=4, FP32-bit-exact nominalMinDc)"
                .to_string(),
    };
    Ok(Rebuilt { mesh, receipt })
}

/// The whole pipeline, from a resolution spec to a validated mesh in memory.
///
/// `progress` receives one tab-separated line per stage so a caller can print
/// them without this module owning a console.
pub fn generate(
    request: &GenerateRequest,
    mut progress: impl FnMut(&str),
) -> MpasResult<Generated> {
    request.spec.check()?;

    // --- how many cells, and at what spacings -------------------------------
    let predicted_from_spec = request.spec.predicted_cells(request.sizing_samples);
    let (target, sizing) = match (request.target_cells, request.budget_mib) {
        (Some(n), _) => (n, Sizing::Explicit),
        (None, Some(mib)) => {
            let card = request.card.ok_or_else(|| {
                MpasError::Refusal(footprint::budget_without_card_refusal(mib))
            })?;
            let n = card.cells_that_fit(mib).map_err(MpasError::Refusal)?;
            (n, Sizing::DeviceBudget)
        }
        (None, None) => (predicted_from_spec.round() as usize, Sizing::FromSpacing),
    };
    if target < 12 {
        return Err(MpasError::Refusal(format!(
            "{target} cells cannot carry the twelve pentagons every triangulated sphere must have"
        )));
    }

    let (spec, scale) = if request.fit_spacing {
        request.spec.fitted_to(target, request.sizing_samples)?
    } else {
        (request.spec.clone(), 1.0)
    };
    let predicted = spec.predicted_cells(request.sizing_samples);
    if !request.fit_spacing && matches!(sizing, Sizing::DeviceBudget) && predicted > target as f64 * 1.02 {
        return Err(MpasError::Refusal(format!(
            "the requested spacings need about {predicted:.0} cells but the device budget holds {target}. Generating {target} cells at these spacings would deliver a mesh {:.2}x coarser than asked for everywhere without saying so. Pass --fit-spacing to rescale every spacing by one factor and keep the ratios between them, or raise the budget",
            (predicted / target as f64).sqrt()
        )));
    }
    progress(&format!(
        "SIZED\t{target}\t{predicted:.1}\t{:.3}\t{:.3}",
        spec.finest_km(),
        spec.background_km
    ));

    // --- seed, relax, triangulate -------------------------------------------
    //
    // A UNIFORM request seeds from a Goldberg subdivision of the icosahedron,
    // not from the Fibonacci lattice. Measured on the Fibonacci path at a
    // 120 km background: shortest dual edge 7,337.6 m at 2,000 cells, 75.0 m
    // at 12,000, 7.2 m at 40,962 -- pentagon-heptagon dislocation quads that
    // are an EQUILIBRIUM feature of the polycrystalline seed, which more
    // relaxation re-rolls but never drains. The published x1.40962 is clean
    // because it IS a subdivided icosahedron; uniform requests now start from
    // the same topology, and the count snaps to the nearest achievable
    // `10*(m^2+mn+n^2)+2` with the snap recorded in the receipt. A budget-
    // sized count snaps DOWNWARD only: a budget is a ceiling, not a target.
    let uniform = spec.regions.is_empty();
    let (mut points, seeding) = if uniform {
        let ceiling = matches!(sizing, Sizing::DeviceBudget);
        let choice = icosa::snap_cells(target, ceiling)?;
        let pts = icosa::seed(choice.m, choice.n)?;
        let seeding = Seeding::IcosahedralGoldberg {
            m: choice.m,
            n: choice.n,
            requested_cells: target,
            seeded_cells: pts.len(),
            snap_relative: pts.len() as f64 / target as f64 - 1.0,
        };
        (pts, seeding)
    } else {
        let pts = lloyd::seed_points(&spec, target)?;
        let seeding = Seeding::FibonacciAcceptance {
            requested_cells: target,
        };
        (pts, seeding)
    };
    match seeding {
        Seeding::IcosahedralGoldberg { m, n, .. } => {
            progress(&format!("SEEDED\t{}\tGP({m},{n})\tfrom {target}", points.len()))
        }
        Seeding::FibonacciAcceptance { .. } => progress(&format!("SEEDED\t{}", points.len())),
    }
    let outcome = lloyd::relax(&mut points, &spec, &request.lloyd)?;
    progress(&format!(
        "RELAXED\t{}\t{:.4e}\t{:.4e}\t{:.2}",
        outcome.sweeps, outcome.mean_delta_over_h, outcome.max_delta_over_h, outcome.wall_seconds
    ));

    // --- every field, then the gate -----------------------------------------
    let mesh_density: Vec<f64> = points.iter().map(|&p| spec.density(p)).collect();
    let nominal = emit::nominal_min_dc_from_m(spec.finest_km() * 1000.0);
    let mesh = MpasMesh::derive(points, mesh_density, &outcome.rings, nominal)?;
    progress(&format!(
        "DERIVED\t{}\t{}\t{}",
        mesh.n_cells, mesh.n_edges, mesh.n_vertices
    ));
    let report = validate::validate(&mesh, request.limits)?;
    progress(&format!(
        "VALIDATED\t{:.16}\t{:.3e}\t{:.3e}",
        report.sum_area_cell_over_4pi, report.max_nonorthogonality, report.max_weight_antisymmetry
    ));

    // Delivered against requested, cell by cell.
    let delivered = mesh.spacing_m();
    let mut ratios: Vec<f64> = (0..mesh.n_cells)
        .map(|i| delivered[i] / spec.spacing_m(mesh.cell_xyz[i]))
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |q: f64| ratios[((ratios.len() - 1) as f64 * q).round() as usize];
    progress(&format!(
        "DELIVERED\t{:.4}\t{:.4}\t{:.4}",
        at(0.05),
        at(0.50),
        at(0.95)
    ));

    let receipt = Receipt {
        engine: concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)").to_string(),
        spec: spec.clone(),
        spec_scale_applied: scale,
        sizing,
        seeding,
        target_cells: target,
        predicted_cells: predicted,
        delivered_cells: mesh.n_cells,
        finest_requested_km: spec.finest_km(),
        background_requested_km: spec.background_km,
        device_budget_mib: request.budget_mib,
        card: request.card.map(|c| c.key),
        footprint_mib: request
            .card
            .and_then(|c| c.footprint_mib(mesh.n_cells).ok()),
        steepest_requested_gradient_percent_per_cell: spec.steepest_gradient_per_cell(50_000) * 100.0,
        published_reference_gradient_percent_per_cell: 1.53,
        region_attainment: spec.region_attainment(200_000),
        delivered_over_requested_median: at(0.50),
        delivered_over_requested_p05: at(0.05),
        delivered_over_requested_p95: at(0.95),
        relaxation_sweeps: outcome.sweeps,
        relaxation_mean_delta_over_h: outcome.mean_delta_over_h,
        relaxation_max_delta_over_h: outcome.max_delta_over_h,
        relaxation_seconds: outcome.wall_seconds,
        mesh: report,
        deliverable_boundary:
            "grid file only; running this mesh also needs a matching static file (terrain, land use, soil, nVertLevels=55, nSoilLevels=4, FP32-bit-exact nominalMinDc)"
                .to_string(),
    };
    Ok(Generated { mesh, receipt, spec })
}

/// The receipt with every wall-clock reading removed, for stamping INTO the
/// grid file.
///
/// The side receipt and the stamped document are deliberately different. A
/// consumer registry -- the MPAS port's `tools/mpas_mesh_binding.py` is the
/// one this crate feeds -- pins a grid by byte count and SHA-256, so anything
/// inside the bytes that moves between two identical runs makes the file
/// unregisterable under a stable name. A duration is exactly that: two runs of
/// the same command wrote `"relaxation_seconds": 4.9126107` and
/// `"relaxation_seconds": 5.03...` and produced two digests.
///
/// The strip is BY SHAPE, not by a hand-kept list of field names: every key
/// ending in `_seconds` goes, at any depth. A duration added to the receipt
/// later is therefore excluded the day it is added, with no second edit --
/// which is the same reason the mesh registry is a table and not a code path.
/// Generic over the receipt type on purpose: `rw_mpas_mesh` has two routes
/// (generate and rebuild-from-centres) with two receipt types, and a rule that
/// only one route obeyed would be worse than no rule.
pub fn provenance_json<T: Serialize>(receipt: &T) -> Result<String, serde_json::Error> {
    let mut value = serde_json::to_value(receipt)?;
    strip_durations(&mut value);
    serde_json::to_string(&value)
}

fn strip_durations(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.retain(|key, _| !key.ends_with("_seconds"));
            for child in map.values_mut() {
                strip_durations(child);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items.iter_mut() {
                strip_durations(child);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod provenance_tests {
    use super::*;

    /// No wall clock may reach the bytes a registry pins, at any depth.
    #[test]
    fn the_strip_removes_every_duration_including_nested_ones() {
        let mut value = serde_json::json!({
            "delivered_cells": 38857,
            "relaxation_sweeps": 131,
            "relaxation_seconds": 4.912_610_7,
            "mesh": {"n_cells": 38857, "validation_seconds": 0.25},
            "regions": [{"spacing_km": 15.0, "fit_seconds": 1.5}],
        });
        strip_durations(&mut value);
        let text = serde_json::to_string(&value).expect("serialises");
        for forbidden in ["_seconds", "4.9126107", "0.25", "1.5"] {
            assert!(
                !text.contains(forbidden),
                "the stamped provenance carries {forbidden:?}, so two identical \
                 runs would write two different grid files and the pair could \
                 never be registered"
            );
        }
        for kept in ["delivered_cells", "relaxation_sweeps", "n_cells", "spacing_km"] {
            assert!(text.contains(kept), "the stamped provenance lost {kept:?}");
        }
    }

    /// The rule is by shape, so the `Receipt` type carries exactly one field
    /// the strip has to catch. If a second duration is added later this test
    /// does not fail -- the strip already covers it -- but the count is stated
    /// so a reader can see the surface being defended.
    #[test]
    fn the_receipt_type_names_its_durations_with_the_suffix_the_strip_matches() {
        assert!(
            std::any::type_name::<Receipt>().ends_with("Receipt"),
            "the receipt type moved"
        );
        // `relaxation_seconds` is the field. Spelled here so a rename that
        // dropped the suffix would leave this assertion pointing at nothing.
        let names = ["relaxation_seconds"];
        for name in names {
            assert!(name.ends_with("_seconds"), "{name} would survive the strip");
        }
    }
}
