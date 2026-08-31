//! Hierarchical Goldberg refinement: graded meshes with no Fibonacci lattice
//! at any level.
//!
//! THE MECHANISM. Level 0 is the proven uniform Goldberg path (12 pentagons,
//! min dv/dc 0.39 class). Each level halves the spacing by density-driven
//! Delaunay edge-midpoint insertion: full midpoint insertion on GP(m, n) IS
//! GP(2m, 2n), so the interior of a refined patch is the same measured-clean
//! crystal, and irregularity is confined BY CONSTRUCTION to the nested
//! transition annuli where the level field varies. Each level is annealed by
//! the existing weighted Lloyd under the level-clamped field, then the
//! residual near-cocircular quads in the annuli are drained by
//! count-changing surgery ([`crate::mesh::surgery`]) -- the degree of
//! freedom relaxation does not have (campaign #304: `--sweeps` 200/600/2000
//! all reproduced the same 75.04 m collapsed edge; the recorded
//! `v15.150.38857` failure is 61 edges under the 0.02 admission floor from a
//! density-biased Fibonacci seed).
//!
//! The target class is the PUBLISHED-VARIABLE band (x4.163842: 44 pentagons
//! + 32 heptagons, min dv/dc 0.03365), not defect-free, which topology
//! forbids on a graded sphere and no gate requires.
//!
//! Determinism: no RNG anywhere -- the seed is the icosahedral lattice,
//! insertion order is canonical edge order, surgery orders by
//! (`f64::total_cmp`, canonical edge id). A grid registry pins files by
//! SHA-256; regeneration must reproduce the bytes.

use serde::Serialize;

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::mesh::density::{Coverage, DensityField, LevelClamp, MeshSpec};
use crate::mesh::derive::Rings;
use crate::mesh::geom::{EARTH_RADIUS_M, V3, add, arc, unit};
use crate::mesh::hull::delaunay_rings;
use crate::mesh::lloyd::{LloydOptions, LloydOutcome, relax};
use crate::mesh::surgery::{self, SurgeryOptions, SurgerySummary};

/// The splitting threshold: a Delaunay edge splits when its arc exceeds
/// `beta` times the level field's spacing at its midpoint. At the parent
/// equilibrium an interior edge reads 2.0 there and a settled edge 1.0;
/// any value between separates them and the doubling stays exact.
///
/// CALIBRATED, not taste, and stamped into every receipt. The design's
/// starting 4/3 was MEASURED to over-deliver: halves land in
/// (2/3, 1) x h_bar, the annulus comes out systematically overdense, and
/// the level delivery gate read p05 = 0.877 / median = 0.964 on the
/// calibration case. sqrt(2) is the log-symmetric threshold -- halves in
/// (0.707, 1) x h_bar against unsplit edges in (1, 1.414) x h_bar, so the
/// local log-mean spacing is the field's own -- and the same case then
/// reads its delivery inside the G7 bounds.
pub const DEFAULT_BETA: f64 = std::f64::consts::SQRT_2;

/// Insertion batches per level before the level refuses. Full midpoint
/// insertion needs exactly one; the follow-up batches only chase ramp
/// edges, and a level still splitting after four is measuring a field
/// mismatch, not making progress.
const MAX_INSERT_BATCHES: usize = 4;

/// Surgery locality radius in cells; a level band narrower than TWICE this
/// cannot contain its own repairs and is refused up front.
const SURGERY_LOCALITY_CELLS: f64 = 3.0;

/// What one level measured, for the receipt.
#[derive(Debug, Clone, Serialize)]
pub struct LevelReport {
    pub level: usize,
    pub level_spacing_km: f64,
    pub inserted: usize,
    pub insert_batches: usize,
    pub relaxation_sweeps: usize,
    pub relaxation_mean_delta_over_h: f64,
    pub min_dv_over_dc_after_anneal: f64,
    pub surgery: SurgerySummary,
    /// Delivered-over-requested spacing against the LEVEL field.
    pub delivered_median: f64,
    pub delivered_p05: f64,
    pub delivered_p95: f64,
}

/// The level ladder in metres: `h_l = h_bg * 2^-l`, last level clamped to
/// the spec's finest spacing. Pure data, derived from the spec; no region
/// enumeration anywhere.
pub fn ladder(spec: &MeshSpec) -> Vec<f64> {
    let h_bg = spec.background_km * 1000.0;
    let h_min = spec.finest_km() * 1000.0;
    if h_min >= h_bg {
        return vec![h_bg];
    }
    let levels = (h_bg / h_min).log2().ceil() as usize;
    let mut out = Vec::with_capacity(levels + 1);
    for l in 0..=levels {
        let h = h_bg * 0.5f64.powi(l as i32);
        out.push(h.max(h_min));
    }
    out
}

/// Delivered spacing per cell in metres, from the Voronoi cell areas of the
/// current triangulation: `h = sqrt(2A / sqrt(3))`, the regular-hexagon
/// across-flats inversion, the same convention `MpasMesh::spacing_m` uses.
fn delivered_spacing_m(points: &[V3], rings: &Rings) -> Vec<f64> {
    use crate::mesh::geom::{circumcenter, tri_area};
    (0..points.len())
        .map(|i| {
            let ring = rings.ring(i);
            let deg = ring.len();
            let mut verts: Vec<V3> = Vec::with_capacity(deg);
            for k in 0..deg {
                let prev = ring[(k + deg - 1) % deg] as usize;
                let cur = ring[k] as usize;
                match circumcenter(points[i], points[prev], points[cur]) {
                    Some(v) => verts.push(v),
                    None => return 0.0,
                }
            }
            let mut area = 0.0;
            for k in 0..deg {
                area += tri_area(points[i], verts[k], verts[(k + 1) % deg]);
            }
            (2.0 * area.abs() / 3f64.sqrt()).sqrt() * EARTH_RADIUS_M
        })
        .collect()
}

/// One level's insertion pass: split every Delaunay edge longer than
/// `beta * h_bar_l` at its spacing-true point, batch by batch, until no
/// edge qualifies; then a COUNT-EXACT top-up. Returns (inserted count,
/// batches used).
///
/// The top-up exists because a threshold cannot serve sub-threshold density:
/// a tanh spec's far tail asks for a few percent more cells than any edge
/// individually justifies splitting, and the anneal then spreads that
/// deficit as a uniform spacing bias -- MEASURED at +1.16% on the delivery
/// median (2.3% count-light) on the calibration case, which is past the 1%
/// the design allows before the count contract engages. The top-up happens
/// at INSERTION time, before the anneal, where an added point is just
/// another seed -- never on a converged, gated mesh.
fn insert_level(
    points: &mut Vec<V3>,
    field: &LevelClamp<'_>,
    beta: f64,
    sizing_samples: usize,
) -> MpasResult<(usize, usize)> {
    // COUNT-TARGETED splitting. The threshold alone cannot count: in a wide
    // gentle annulus every parent edge sits in the split band and thresholded
    // doubling over-delivers (MEASURED: +9.8% on a 30-in-120 case whose ramp
    // covered ~30% of the sphere; the anneal then has to transport the
    // surplus across the band, which it cannot finish). The level field's own
    // sizing integral says how many cells the level is owed, so insertion
    // splits WORST-FIRST -- longest edge over its local spacing first, the
    // innermost band before the outer -- and stops at the predicted count.
    // Delivered count is then exact by construction wherever enough edges
    // qualify, and the shortfall lands in the outer band where the field is
    // coarsening anyway.
    let n_target = crate::mesh::density::predicted_cells_of(field, sizing_samples).round() as i64;
    let mut inserted_total = 0usize;
    let mut batches_used = 0usize;
    for batch in 0..=MAX_INSERT_BATCHES {
        let need = n_target - points.len() as i64;
        if need <= 0 {
            batches_used = batch;
            break;
        }
        let rings = delaunay_rings(points)?;
        // Canonical edge order: (i, j) ascending with i < j; ranked
        // worst-first by ratio with the canonical id as the tie-break.
        // PARALLEL, and the same list. Every edge is independent and the sort
        // below is a TOTAL order -- `(ratio, i, j)` with `(i, j)` unique per
        // edge -- so the order the edges are collected in cannot reach the
        // answer. The per-cell lists are collected by cell index and
        // concatenated, so even the pre-sort vector is identical to the
        // serial one; the field evaluation at each midpoint is the cost, and
        // it is 8.8% of a maintained-arm graded run's wall.
        let qualifying: Vec<(f64, u32, u32)> = (0..points.len())
            .into_par_iter()
            .map(|i| {
                let mut local: Vec<(f64, u32, u32)> = Vec::new();
                for &j in rings.ring(i) {
                    let j = j as usize;
                    if j <= i {
                        continue;
                    }
                    let Some(mid) = unit(add(points[i], points[j])) else {
                        continue;
                    };
                    let h_bar = field.spacing_m(mid) / EARTH_RADIUS_M;
                    let a = arc(points[i], points[j]);
                    let ratio = a / h_bar;
                    if ratio > beta {
                        local.push((ratio, i as u32, j as u32));
                    }
                }
                local
            })
            .collect::<Vec<_>>()
            .concat();
        let mut qualifying = qualifying;
        if qualifying.is_empty() {
            batches_used = batch;
            break;
        }
        if batch == MAX_INSERT_BATCHES {
            return Err(MpasError::Refusal(format!(
                "level insertion at h_l = {:.1} km did not terminate: after {MAX_INSERT_BATCHES} batches {} edges still exceed beta = {beta:.4} times the level spacing while the level still needs {need} cells. The splitting criterion and the level field disagree about equilibrium, which means beta is mis-calibrated for this field's gradient rather than the mesh needing more splitting; inserting further would runaway-refine the ramp instead of doubling the crystal",
                field.level_spacing_m / 1000.0,
                qualifying.len()
            )));
        }
        qualifying.sort_by(|x, y| {
            y.0.total_cmp(&x.0)
                .then_with(|| (x.1, x.2).cmp(&(y.1, y.2)))
        });
        qualifying.truncate(need as usize);
        let new_points: Vec<V3> = qualifying
            .iter()
            .map(|&(_, i, j)| {
                surgery::spacing_true_point(field, points[i as usize], points[j as usize])
            })
            .collect();
        inserted_total += new_points.len();
        points.extend(new_points);
    }

    // ---- count-exact top-up ----------------------------------------------
    // Nothing to do on a level that inserted nothing: the field never
    // crossed this level's spacing and the count belongs to level 0's snap.
    if inserted_total == 0 {
        return Ok((0, batches_used));
    }
    let predicted = n_target as f64;
    let deficit = predicted.round() as i64 - points.len() as i64;
    if deficit.unsigned_abs() as usize > points.len() / 20 {
        return Err(MpasError::Refusal(format!(
            "the level at h_l = {:.1} km delivered {} points against a predicted {predicted:.0} -- a {deficit} gap, over 5% of the mesh. A top-up that large is not serving a sub-threshold tail, it is papering over a field/criterion disagreement, and the anneal would have to transport the correction across the whole sphere",
            field.level_spacing_m / 1000.0,
            points.len()
        )));
    }
    if deficit > 0 {
        // Insert at the spacing-true points of the LONGEST sub-threshold
        // edges, worst-first in canonical order, no two on a shared cell:
        // exactly where the un-served tail lives, and deterministic.
        let rings = delaunay_rings(points)?;
        // Parallel for the same reason and with the same guarantee as the
        // batch scan above: per-cell lists concatenated in cell order, then a
        // total-order sort.
        let ranked: Vec<(f64, u32, u32)> = (0..points.len())
            .into_par_iter()
            .map(|i| {
                let mut local: Vec<(f64, u32, u32)> = Vec::new();
                for &j in rings.ring(i) {
                    let j = j as usize;
                    if j <= i {
                        continue;
                    }
                    let Some(mid) = unit(add(points[i], points[j])) else {
                        continue;
                    };
                    // Only edges where the level field genuinely VARIES: the
                    // deficit lives in the annulus and the tanh tail, and a
                    // top-up point dropped into the uniform crystal (where
                    // the clamp holds the field at exactly h_l) would break
                    // the GP-doubled interior for a deficit that is not local
                    // to it.
                    let h_bar_m = field.spacing_m(mid);
                    if h_bar_m <= field.level_spacing_m {
                        continue;
                    }
                    let h_bar = h_bar_m / EARTH_RADIUS_M;
                    let ratio = arc(points[i], points[j]) / h_bar;
                    if ratio <= beta {
                        local.push((ratio, i as u32, j as u32));
                    }
                }
                local
            })
            .collect::<Vec<_>>()
            .concat();
        let mut ranked = ranked;
        ranked.sort_by(|x, y| {
            y.0.total_cmp(&x.0)
                .then_with(|| (x.1, x.2).cmp(&(y.1, y.2)))
        });
        let mut used = vec![false; points.len()];
        let mut added = 0i64;
        let mut top_up: Vec<V3> = Vec::new();
        for &(_, i, j) in &ranked {
            if added >= deficit {
                break;
            }
            if used[i as usize] || used[j as usize] {
                continue;
            }
            used[i as usize] = true;
            used[j as usize] = true;
            top_up.push(surgery::spacing_true_point(
                field,
                points[i as usize],
                points[j as usize],
            ));
            added += 1;
        }
        inserted_total += top_up.len();
        points.extend(top_up);
    }
    // A surplus is left to the anneal: sqrt(2) splitting runs count-light by
    // construction, and deleting seeds here would un-double crystal interior.
    Ok((inserted_total, batches_used))
}

/// Generate a graded mesh by hierarchical refinement.
///
/// Returns the relaxed points, their final rings, the FINAL level's
/// relaxation outcome (its per-sweep min dv/dc trajectory is the one the
/// receipt stamps), and the per-level reports. The caller (the `generate`
/// fork in `mod.rs`) derives, validates and stamps the receipt exactly as
/// it does for the uniform arm.
pub fn generate_graded(
    spec: &MeshSpec,
    sizing_samples: usize,
    lloyd: &LloydOptions,
    surgery_opts: &SurgeryOptions,
    beta: f64,
    mut progress: impl FnMut(&str),
) -> MpasResult<(
    Vec<V3>,
    Rings,
    LloydOutcome,
    crate::mesh::icosa::GoldbergChoice,
    Vec<LevelReport>,
)> {
    spec.check()?;
    if spec.regions.is_empty() {
        return Err(MpasError::Refusal(
            "the hierarchical ladder is the graded arm; a uniform request has no bands to refine and takes the icosahedral arm directly".to_string(),
        ));
    }

    // ---- pre-run arithmetic gates (the ones a spent run cannot un-spend) --
    //
    // G0 -- EVERY REQUEST MUST SIT ON THE LADDER. Refinement here is midpoint
    // insertion, which changes a spacing by exactly two, so a refined core can
    // only ever sit at `background / 2^k`. A request off that ladder is not
    // approximated, it is MISSED, and the level delivery gate cannot see the
    // miss because it is a median over every cell while the miss is confined
    // to the core. MEASURED 2026-08-29: background 480 km with a 51.2 km
    // region delivered 59.49 km, and with a 93.75 km region delivered
    // 120.02 km; both passed every gate and wrote a grid file naming the
    // resolution they did not have. `mesh::ladder_snap` moves a request onto
    // the ladder before this function is reached, so this refusal is what
    // makes that a contract rather than a courtesy.
    if let Some((i, requested, levels, delivered)) =
        crate::mesh::ladder_snap::first_off_ladder(spec)
    {
        return Err(MpasError::Refusal(format!(
            "region {i} asks for {requested:.4} km against a {:.4} km background, a ratio of              {:.4} that is not a power of two. This ladder refines by MIDPOINT INSERTION, which              halves a spacing exactly, so {levels} levels reach {delivered:.4} km and there is no              rung at {requested:.4}: the core would be built at {delivered:.4} km and the file              would name {requested:.4}. The level delivery gate cannot catch it -- it is a median              over every cell and the miss is confined to the core, measured at 1.0070 against a              1.0212 bound while the core ran 16.2% coarse. Snap the request with              mesh::ladder_snap::snap_to_ladder (which delivers {delivered:.4} km, finer than              asked), or set the background to {:.4} km, which puts {requested:.4} km on a rung              exactly",
            spec.background_km,
            spec.background_km / requested,
            requested * 2f64.powi(levels as i32)
        )));
    }
    let steps = ladder(spec);
    let reading = spec.steepest_gradient_reading(50_000);
    // AN UNMEASURED GRADIENT REFUSES. THE BREAKAGE THIS PREVENTS: the reading
    // is a max over a probe set, so a set that never touched a region reports
    // 0.0, which the band arithmetic below maps to f64::INFINITY -- the most
    // permissive verdict in this file. MEASURED on the shipped sampler: a
    // 3 km cap at 0.15 km spacing on a 76.8 km background read exactly
    // 0.0000 %/cell and the ladder built it, against a true 2200 %/cell and a
    // band of 0.22 cells. A catastrophic measurement produced the friendliest
    // answer. These two refusals separate "the field is flat" from "nobody
    // looked", which used to be spelled the same.
    if let Coverage::Partial(loci) = &reading.coverage {
        return Err(MpasError::Refusal(format!(
            "the steepest per-cell spacing change could not be MEASURED for {}: covering the transition shell of a region that narrow, against a boundary that long, needs more probe points than this gate is allowed to spend. A gradient the gate cannot see is one it cannot refuse, and an unmeasured ramp is not a gentle one. Widen the transition or shrink the region",
            loci.join(", ")
        )));
    }
    if !reading.saw_the_refinement() {
        return Err(MpasError::Refusal(format!(
            "the spec asks for {:.4} km spacing somewhere, but every one of the {} probe points read the {:.4} km background, so the transition between them was never visited and its steepness was not measured. A build seeded from this field would put a resolution jump inside a single surgery neighbourhood with no gate having looked at it",
            reading.declared_finest_m / 1000.0,
            reading.probe_points,
            reading.background_m / 1000.0
        )));
    }
    let gradient = reading.per_cell;
    // Cells across a 2x band at this per-cell gradient: n = ln 2 / ln(1+g).
    // The INFINITY branch is sound now and was not before: it is reachable
    // only under complete coverage, where a zero reading means a genuinely
    // uniform field.
    let band_cells = if gradient > 0.0 {
        (2f64).ln() / (1.0 + gradient).ln()
    } else {
        f64::INFINITY
    };
    if band_cells < 2.0 * SURGERY_LOCALITY_CELLS {
        return Err(MpasError::Refusal(format!(
            "the requested gradient ({:.2}% per cell) makes each level's transition band about {band_cells:.1} cells wide -- narrower than twice the {SURGERY_LOCALITY_CELLS:.0}-cell surgery locality radius, so a repair at the band's centre would reach across the whole band and repairs could not be contained where the localization claim confines them. Widen the transition (the receipt's region_attainment carries widest_transition_km as the printed remedy); the reachable spec space shrinks at extreme gradients and that is the correct outcome",
            gradient * 100.0
        )));
    }

    progress(&format!(
        "LADDER\t{}\t{}",
        steps.len() - 1,
        steps
            .iter()
            .map(|h| format!("{:.1}", h / 1000.0))
            .collect::<Vec<_>>()
            .join(",")
    ));

    // ---- level 0: the shipped uniform arm, byte for byte ------------------
    let uniform0 = MeshSpec::uniform(steps[0] / 1000.0);
    let n0 = uniform0.predicted_cells(sizing_samples).round() as usize;
    let choice = crate::mesh::icosa::snap_cells(n0.max(12), false)?;
    let mut points = crate::mesh::icosa::seed(choice.m, choice.n)?;
    progress(&format!(
        "SEEDED\t{}\tGP({},{})\tlevel 0 of {}",
        points.len(),
        choice.m,
        choice.n,
        steps.len() - 1
    ));
    // The seed snap's own quantization: GP counts are discrete, and spacing
    // scales as count^-1/2, so half the snap magnitude is the median
    // delivery offset the seed itself imposes. The level delivery gate
    // widens its median bound by exactly this much -- data, not taste; at
    // production scale the snap is ~0.1% and the widening vanishes.
    let snap_median_offset = 0.5 * (choice.cells as f64 / n0.max(12) as f64 - 1.0).abs();
    // The resolution field is asked about 10^9 times over a ladder this size;
    // prepared ONCE here, and every level clamp is a view over this one.
    let prepared = spec.prepared();
    let clamp0 = prepared.clamped(steps[0]);
    let mut outcome = relax(&mut points, &clamp0, lloyd)?;
    progress(&format!(
        "ANNEALED\t0\t{}\t{:.4e}\t{:.4}",
        outcome.sweeps, outcome.mean_delta_over_h, outcome.min_dv_over_dc
    ));

    let mut reports: Vec<LevelReport> = Vec::new();

    // ---- levels 1..L ------------------------------------------------------
    // A level anneal starts from an inserted annulus whose density surplus
    // has to migrate outward at one cell width per sweep, so it legitimately
    // needs more sweeps than a uniform relaxation from a biased seed: three
    // full budgets, the upper end of the design's own per-level estimate.
    //
    // The monitor's collapse REFUSAL is disarmed for these anneals -- and
    // only these -- because their consumer is the surgery step right below:
    // a dislocation annealing toward its near-cocircular equilibrium is the
    // exact state count-changing surgery exists to drain, so refusing the
    // anneal for producing surgery's input would make the ladder
    // unbuildable. The trajectory is still recorded every sweep and stamped
    // into the receipt, and the sub-floor tail cannot ship: the level gate
    // holds the post-surgery mesh to 0.03 and the emit and admission gates
    // to 0.02. Standalone and uniform relaxations, which have NO surgery
    // downstream, keep the armed monitor.
    let level_lloyd = LloydOptions {
        max_sweeps: lloyd.max_sweeps.saturating_mul(3),
        monitor_floor: 0.0,
        ..*lloyd
    };

    for (l, &h_l) in steps.iter().enumerate().skip(1) {
        let clamp = prepared.clamped(h_l);
        let (inserted, batches) = crate::mesh::profile::timed(
            &crate::mesh::profile::INSERT,
            points.len() as u64,
            || insert_level(&mut points, &clamp, beta, sizing_samples),
        )?;
        progress(&format!("INSERTED\t{l}\t{inserted}\t{batches}"));
        if inserted == 0 {
            // A level with no band (spec never crosses h_l) is a no-op level.
            continue;
        }

        outcome = relax(&mut points, &clamp, &level_lloyd)?;
        progress(&format!(
            "ANNEALED\t{l}\t{}\t{:.4e}\t{:.4}",
            outcome.sweeps, outcome.mean_delta_over_h, outcome.min_dv_over_dc
        ));

        // G4 -- LOCATION FALSIFICATION, refuse-don't-repair, BEFORE surgery.
        // Every quad reading under 0.10 must sit where the ladder predicts
        // irregularity: somewhere the spec is genuinely graded (inside some
        // level band, dilated), never out in the uniform crystal. Repairing
        // an off-band offender would mask a systemic ladder defect, so it is
        // a refusal that names its coordinates and reading.
        let rings_check = outcome.rings.clone();
        let mut off_band: Option<(V3, f64)> = None;
        let h_bg_m = spec.background_km * 1000.0;
        surgery::for_each_quad(&points, &rings_check, |r| {
            if r.q < 0.10 && off_band.is_none() {
                let mid = unit(add(points[r.i as usize], points[r.j as usize]))
                    .unwrap_or(points[r.i as usize]);
                let h_here = prepared.spacing_m(mid);
                let inside_bands = h_here < 0.99 * h_bg_m && h_here > h_l * 0.95;
                if !inside_bands {
                    off_band = Some((mid, r.q));
                }
            }
        });
        if let Some((m, q)) = off_band {
            let (lat, lon) = crate::mesh::geom::lat_lon(m);
            return Err(MpasError::Refusal(format!(
                "level {l} carries a dislocation OUTSIDE the predicted transition annuli: dv/dc = {q:.3e} at lat/lon ({:.3}, {:.3}) deg, where the spec's spacing is {:.1} km against a level band of ({:.1}, {:.1}) km. Localization is the ladder's one load-bearing claim -- irregularity is confined to the annuli by construction -- so an off-band defect means the ladder itself is wrong and repairing it would mask that. Refused, never repaired",
                lat.to_degrees(),
                lon.to_degrees(),
                prepared.spacing_m(m) / 1000.0,
                h_l / 1000.0,
                steps[l - 1] / 1000.0
            )));
        }

        // Surgery: drain the annuli's near-cocircular tail. Drift bounded at
        // 1% of this level's insertions.
        let drift_budget = (inserted / 100).max(1);
        let (rings_after, ledger) = crate::mesh::profile::timed(
            &crate::mesh::profile::SURGERY,
            points.len() as u64,
            || surgery::drain(&mut points, &clamp, surgery_opts, drift_budget),
        )?;
        progress(&format!(
            "SURGERY\t{l}\t{}\t{}\t{:.4}",
            ledger.rounds,
            ledger.ops.len(),
            ledger.min_q_after
        ));

        // Level gate: delivered percentiles against the LEVEL field, over
        // the HEXAGONAL cells. The area estimator reads a correctly spaced
        // pentagon at sqrt(5/6) = 0.913 of its true spacing -- pure shape,
        // not delivery -- and a rung-scale annulus carries a few percent of
        // defect cells, which parks p05 at the artifact level (MEASURED:
        // 0.8901 on a 12k-class rung whose crystal was in contract). The
        // gate grades what the level DELIVERED; the final receipt's
        // all-cells percentiles remain the campaign's G7 reading, where the
        // canonical defect fraction (~10^-3) cannot move a percentile.
        let delivered = delivered_spacing_m(&points, &rings_after);
        let mut ratios: Vec<f64> = (0..points.len())
            .filter(|&i| rings_after.degree(i) == 6)
            .map(|i| delivered[i] / DensityField::spacing_m(&clamp, points[i]))
            .collect();
        ratios.sort_by(|a, b| a.total_cmp(b));
        let at = |q: f64| ratios[((ratios.len() - 1) as f64 * q).round() as usize];
        let (p05, p50, p95) = (at(0.05), at(0.50), at(0.95));
        // 2% + the seed snap: the LEVEL gate is a tripwire against gross
        // delivery lies (the measured defect class read 0.964 median / 0.877
        // p05 under the 4/3 beta), while healthy levels across the measured
        // fixture set spread 0.99-1.016 -- the composite of the area
        // estimator's convention and the SCVT's own graded packing overhead.
        // The 1% contract stands where the design states it: G7, on the
        // canonical mesh's final all-cells receipt.
        let median_bound = 0.02 + snap_median_offset;
        // BOTH TAILS ARE REPORTED, NOT GATED, and that is a measurement,
        // not a relaxation. Every level run this generator has produced:
        //
        //   level run                      p05      median   p95
        //   DEFECT (4/3-beta over-split)   0.8773   0.9643   1.0138
        //   480->240 rung L1               0.9314   1.0158   1.0427
        //   240->120 rung L1               0.9025   1.0105   1.0592
        //   120->30  rung L1               0.8902   1.0060   1.1121
        //   120->30  rung L2               0.8665   1.0026   1.1465
        //   60->15   canonical L1          0.8717   1.0130   1.0885
        //   60->15   canonical L2          0.8628   1.0080   1.1235
        //   60->15   second-coords L2      0.8484   1.0088   1.1347
        //   66->16.5 fit L2                0.8620   1.0069   1.1390
        //   80->20   run20 L2              0.8609   1.0062   1.1544
        //
        // The one REAL delivery defect carries the LOWEST p95 of the ten
        // and a p05 above most of them: both tails are anti-correlated with
        // the fault they were supposed to catch, because an over-splitting
        // level delivers a tighter distribution around a wrong centre. A
        // tail clause therefore cannot separate health from defect -- it
        // can only refuse a healthy mesh by roll, which it did twice (p05
        // 0.8484 by 0.0016, then p95 0.1544 by 0.0044). The MEDIAN
        // separates cleanly: 0.9643 for the defect against 1.0026-1.0158
        // for every healthy roll, a 1.3%-wide healthy band inside a 2%
        // bound. G7's all-cells percentiles stand where the design states
        // them -- reported on the canonical receipt, where they inform
        // rather than gate.
        if !((p50 - 1.0).abs() <= median_bound) {
            return Err(MpasError::Refusal(format!(
                "level {l} (h_l = {:.1} km) missed its delivery contract: delivered/requested spacing median = {p50:.4} against a bound of 1 +/- {median_bound:.4} (2% plus the seed snap's own {snap_median_offset:.4} quantization). The tails are reported, not gated -- p05 = {p05:.4}, p95 = {p95:.4} -- because the measured delivery defect carried the lowest p95 and a mid-range p05 of every level run, so a tail clause separates nothing and refuses healthy meshes by roll. A level that under- or over-delivers its own field hands the next level a lattice whose edges are the wrong length for the splitting criterion, and the final mesh would carry the miss as a silent resolution lie -- the spacing on paper would not be the spacing in the window it was bought for",
                h_l / 1000.0
            )));
        }
        // The surgery ledger already guarantees min q >= repair hysteresis on
        // touched sites; the SHIP floor is asserted here for the whole level.
        if ledger.min_q_after < surgery_opts.ship_floor {
            return Err(MpasError::Refusal(format!(
                "level {l} finished surgery at min dv/dc = {:.4e}, under the {:.2} ship floor (1.5x the untouchable 0.02 admission floor). The TRiSK tangential weights divide by dvEdge, so this level would hand the next one a {:.0}x amplification to bury deeper in the ladder",
                ledger.min_q_after,
                surgery_opts.ship_floor,
                1.0 / ledger.min_q_after.max(1e-300)
            )));
        }

        reports.push(LevelReport {
            level: l,
            level_spacing_km: h_l / 1000.0,
            inserted,
            insert_batches: batches,
            relaxation_sweeps: outcome.sweeps,
            relaxation_mean_delta_over_h: outcome.mean_delta_over_h,
            min_dv_over_dc_after_anneal: outcome.min_dv_over_dc,
            surgery: SurgerySummary::from(&ledger),
            delivered_median: p50,
            delivered_p05: p05,
            delivered_p95: p95,
        });
    }

    // The final rings: re-derived once so the returned pair is exactly the
    // Delaunay of the returned points (surgery already guarantees this, but
    // deriving it here makes the contract independent of the loop's tail).
    let rings = delaunay_rings(&points)?;
    // The outcome the receipt stamps must describe the RETURNED mesh, not
    // the pre-surgery anneal state: its rings are replaced with the final
    // Delaunay and the post-surgery monitor reading is appended as the
    // trajectory's last sample.
    let (q_final, edge_final) = surgery::min_dv_over_dc(&points, &rings);
    outcome.rings = rings.clone();
    outcome.min_dv_over_dc = q_final;
    outcome.min_dv_over_dc_edge = edge_final;
    outcome.min_dv_over_dc_trajectory.push(q_final);
    Ok((points, rings, outcome, choice, reports))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::density::{Region, Shape, TransitionField};

    fn small_graded_spec() -> MeshSpec {
        // 240 -> 480 km: one doubling, a single cap band, small enough for a
        // test-speed sphere (~2,500 cells at level 1).
        MeshSpec {
            background_km: 480.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 4000.0,
                },
                spacing_km: 240.0,
                transition: TransitionField::Km(3000.0),
            }],
            name: None,
        }
    }

    #[test]
    fn the_ladder_is_pure_data_from_the_spec() {
        let spec15to60 = MeshSpec {
            background_km: 60.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [37.0, -97.0],
                    radius_km: 500.0,
                },
                spacing_km: 15.0,
                transition: TransitionField::Km(3000.0),
            }],
            name: None,
        };
        assert_eq!(ladder(&spec15to60), vec![60_000.0, 30_000.0, 15_000.0]);
        let spec15to120 = MeshSpec {
            background_km: 120.0,
            regions: spec15to60.regions.clone(),
            name: None,
        };
        assert_eq!(
            ladder(&spec15to120),
            vec![120_000.0, 60_000.0, 30_000.0, 15_000.0]
        );
        // An 18 -> 120 request is not a power of two: the last level clamps.
        let spec18 = MeshSpec {
            background_km: 120.0,
            regions: vec![Region {
                shape: spec15to60.regions[0].shape.clone(),
                spacing_km: 18.0,
                transition: TransitionField::Km(3000.0),
            }],
            name: None,
        };
        let l = ladder(&spec18);
        assert_eq!(l.len(), 4);
        assert_eq!(*l.last().unwrap(), 18_000.0);
        assert_eq!(ladder(&MeshSpec::uniform(120.0)), vec![120_000.0]);
    }

    #[test]
    fn full_midpoint_insertion_on_a_goldberg_patch_is_exact_gp_doubling() {
        // A UNIFORM interior under a level whose spacing is half the current
        // one: every edge splits at its exact midpoint, and the point set is
        // exactly GP(2m, 2n)'s -- same crystal, no new grain boundaries.
        let m = 4u32;
        let n = 0u32;
        let mut pts = crate::mesh::icosa::seed(m, n).unwrap();
        // GP(4,0) delivered spacing on the unit sphere in "metres" of the
        // earth-scaled field: mean arc * R. Halving it is the level request.
        let rings = delaunay_rings(&pts).unwrap();
        let mut mean_arc = 0.0;
        let mut count = 0usize;
        for i in 0..pts.len() {
            for &j in rings.ring(i) {
                if (j as usize) > i {
                    mean_arc += arc(pts[i], pts[j as usize]);
                    count += 1;
                }
            }
        }
        mean_arc /= count as f64;
        let uniform_half = MeshSpec::uniform(mean_arc * EARTH_RADIUS_M / 2.0 / 1000.0);
        let prepared_half = uniform_half.prepared();
        let clamp = prepared_half.clamped(mean_arc * EARTH_RADIUS_M / 2.0);
        let (inserted, batches) = insert_level(&mut pts, &clamp, DEFAULT_BETA, 50_000).unwrap();
        assert_eq!(
            pts.len(),
            crate::mesh::icosa::goldberg_cells(2 * m, 2 * n),
            "midpoint insertion did not deliver GP({},{})'s exact count",
            2 * m,
            2 * n
        );
        assert!(batches <= 2, "doubling took {batches} batches");
        assert_eq!(inserted, pts.len() - crate::mesh::icosa::goldberg_cells(m, n));

        // The doubled set carries GP(2m,2n)'s TOPOLOGY: exactly twelve
        // pentagons, no heptagons -- no dislocation for a near-cocircular
        // quad to form around. (The identity is combinatorial, not
        // coordinate-exact: projection norms differ between the two
        // constructions and the anneal owns the geometry.)
        let rings2 = delaunay_rings(&pts).unwrap();
        let mut hist = std::collections::BTreeMap::<usize, usize>::new();
        for i in 0..pts.len() {
            *hist.entry(rings2.ring(i).len()).or_default() += 1;
        }
        assert_eq!(hist.get(&5).copied().unwrap_or(0), 12, "{hist:?}");
        assert_eq!(
            hist.get(&6).copied().unwrap_or(0),
            pts.len() - 12,
            "a dislocation exists in the doubled crystal: {hist:?}"
        );

        // And every doubled point sits within a small fraction of the child
        // spacing of a true GP(2m,2n) lattice point: the same crystal, with
        // only the projection-metric offset the anneal polishes away.
        let reference = crate::mesh::icosa::seed(2 * m, 2 * n).unwrap();
        let child_h = (4.0 * std::f64::consts::PI / pts.len() as f64).sqrt();
        for r in &reference {
            let nearest = pts
                .iter()
                .map(|p| crate::mesh::geom::chord(*p, *r))
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest < 0.2 * child_h,
                "a GP(8,0) lattice point is {nearest:.3e} from the doubled set against spacing {child_h:.3e}"
            );
        }

        // Determinism: the same insertion twice is bit-identical.
        let mut again = crate::mesh::icosa::seed(m, n).unwrap();
        insert_level(&mut again, &clamp, DEFAULT_BETA, 50_000).unwrap();
        assert_eq!(pts.len(), again.len());
        for k in 0..pts.len() {
            assert_eq!(pts[k], again[k], "insertion is not deterministic at point {k}");
        }
    }

    #[test]
    fn spacing_true_placement_lands_within_tolerance_of_the_fixed_point() {
        // A graded field at the warn-line gradient class: the two-iteration
        // placement must land within 3e-4 of the local spacing from the
        // converged fixed point.
        let spec = small_graded_spec();
        let prepared = spec.prepared();
        let clamp = prepared.clamped(240_000.0);
        let p = crate::mesh::geom::from_lat_lon(0.55, -1.62);
        let q = crate::mesh::geom::from_lat_lon(0.62, -1.55);
        let two = surgery::spacing_true_point(&clamp, p, q);
        // Converge the same fixed point to numerical rest.
        let slerp = |t: f64| -> V3 {
            let omega = arc(p, q);
            let so = omega.sin();
            let wp = ((1.0 - t) * omega).sin() / so;
            let wq = (t * omega).sin() / so;
            unit(add(
                crate::mesh::geom::scale(p, wp),
                crate::mesh::geom::scale(q, wq),
            ))
            .unwrap()
        };
        let mut t = 0.5;
        for _ in 0..64 {
            let hl = DensityField::spacing_m(&clamp, slerp(0.5 * t));
            let hr = DensityField::spacing_m(&clamp, slerp(t + 0.5 * (1.0 - t)));
            t = hl / (hl + hr);
        }
        let converged = slerp(t);
        let h_local = DensityField::spacing_m(&clamp, converged) / EARTH_RADIUS_M;
        let err = arc(two, converged);
        assert!(
            err < 3e-4 * h_local,
            "two-iteration placement is {:.3e} rad from the fixed point against 3e-4 h = {:.3e}",
            err,
            3e-4 * h_local
        );
    }

    #[test]
    fn a_band_narrower_than_the_surgery_locality_is_refused_by_name() {
        // A brutal ramp: 30 km over a 35 km transition -- around 40% per
        // cell, so a 2x band spans ~2 cells against the 6 the surgery
        // locality needs.
        let spec = MeshSpec {
            background_km: 480.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 2000.0,
                },
                spacing_km: 240.0,
                transition: TransitionField::Km(35.0),
            }],
            name: None,
        };
        let err = generate_graded(
            &spec,
            50_000,
            &LloydOptions::default(),
            &SurgeryOptions::default(),
            DEFAULT_BETA,
            |_| {},
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("surgery locality radius"), "{err}");
        assert!(err.contains("widest_transition_km"), "{err}");
    }

    /// THE GATE'S BLIND SPOT, closed.
    ///
    /// This request is the same shape as the one above and 200 times smaller:
    /// a 10 km cap refined to 200 m over a 3.253 km ramp, on a 51.2 km
    /// background. Its true per-cell gradient is far steeper than the one
    /// above -- a band of a fifth of a cell against the surgery locality's
    /// six -- and until the gradient was measured where the regions are, this
    /// ladder BUILT: the 50,000 lattice points the gate sampled sit 101 km
    /// apart, none of them landed inside the ramp, and the gate was handed the
    /// flat background instead. It is the geometry the auto-spawned sub-km
    /// nests are made of, which is why a refusal here is the point of the
    /// change and not a side effect of it.
    #[test]
    fn a_band_narrower_than_the_lattice_that_measures_it_is_refused_too() {
        let spec = MeshSpec {
            background_km: 51.2,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [-60.0, 0.0],
                    radius_km: 10.0,
                },
                spacing_km: 0.2,
                transition: TransitionField::Km(3.253),
            }],
            name: None,
        };
        let err = generate_graded(
            &spec,
            50_000,
            &LloydOptions::default(),
            &SurgeryOptions::default(),
            DEFAULT_BETA,
            |_| {},
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("surgery locality radius"), "{err}");
    }

    #[test]
    fn a_uniform_request_does_not_take_the_graded_arm() {
        let err = generate_graded(
            &MeshSpec::uniform(240.0),
            50_000,
            &LloydOptions::default(),
            &SurgeryOptions::default(),
            DEFAULT_BETA,
            |_| {},
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("uniform request"), "{err}");
    }

    #[test]
    fn the_graded_ladder_generates_a_small_two_level_mesh_end_to_end() {
        // The smallest honest end-to-end: one doubling with a real band.
        let spec = small_graded_spec();
        let (points, rings, outcome, choice, reports) = generate_graded(
            &spec,
            50_000,
            &LloydOptions::default(),
            &SurgeryOptions::default(),
            DEFAULT_BETA,
            |_| {},
        )
        .unwrap_or_else(|e| panic!("graded generation refused: {e}"));
        eprintln!(
            "graded 240->480: GP({},{}) level 0, {} cells final, {} level reports, final min dv/dc {:.4}",
            choice.m,
            choice.n,
            points.len(),
            reports.len(),
            outcome.min_dv_over_dc
        );
        assert_eq!(rings.n_cells(), points.len());
        // No cell below the Goldberg coordination floor. This is the ladder's
        // half of the 2026-08-26 fix: `v16.66.195630` came off this same code
        // path with one 4-coordinated cell at index 195615 and killed its
        // forecast (gpuwm-hex tree/evidence/graded-blowup-20260826/).
        assert!(
            crate::mesh::surgery::cells_below_coordination(&rings).is_empty(),
            "the ladder emitted cells below coordination {}: {:?}",
            crate::mesh::surgery::MIN_COORDINATION,
            crate::mesh::surgery::cells_below_coordination(&rings)
        );
        assert!(!outcome.min_dv_over_dc_trajectory.is_empty());
        assert_eq!(reports.len(), 1, "one refinement level expected");
        assert!(reports[0].inserted > 0);
        assert!(
            reports[0].surgery.min_q_after >= 0.03,
            "surgery left the level at min dv/dc {:.4e}",
            reports[0].surgery.min_q_after
        );
        // The whole mesh derives and validates under the RATIO floor: this
        // is the emit gate's own geometry checks on the graded arm. The
        // absolute length floor is not exercised at this coarse scale
        // (~240 km spacing sits far above it).
        let density: Vec<f64> = points.iter().map(|&p| spec.density(p)).collect();
        let nominal = crate::mesh::emit::nominal_min_dc_from_m(spec.finest_km() * 1000.0);
        let mesh =
            crate::mesh::derive::MpasMesh::derive(points.clone(), density, &rings, nominal)
                .expect("derive");
        let report = crate::mesh::validate::validate(&mesh, crate::mesh::validate::Limits::default())
            .unwrap_or_else(|e| panic!("the graded mesh failed the emit gate: {e}"));
        assert!(report.min_dv_over_dc >= 0.03);
        // Census: the published-variable class, not the polycrystalline one.
        let defects: usize = report
            .coordination_histogram
            .iter()
            .filter(|(d, _)| *d != 6)
            .map(|(_, c)| *c)
            .sum();
        eprintln!(
            "  coordination {:?} -> {defects} non-hexagonal cells of {}",
            report.coordination_histogram, report.n_cells
        );
        // This fixture's ramp (~3.7% per cell) is over twice the 1.53%
        // published warn line, so its geometrically required dislocation
        // array is denser than the canonical class; the canonical census
        // (the ~10^2 published-variable class at <= 1.53% per cell) is the
        // campaign's G3 measurement, not this unit test's. The bound here
        // separates "dense annulus array" (a few percent) from the
        // polycrystalline class (v15's 6,906 of 38,857 = 17.8%).
        assert!(
            (defects as f64) < 0.05 * report.n_cells as f64,
            "{defects} defect cells on {} is the polycrystalline class",
            report.n_cells
        );
    }
}
