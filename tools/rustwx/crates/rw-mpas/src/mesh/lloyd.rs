//! Initial point placement and the centroidal Voronoi relaxation.
//!
//! The relaxation is Lloyd's algorithm with the CONSTRAINED SPHERICAL centroid
//! of Du, Gunzburger & Ju: `c_i = normalize(integral over V_i of rho(x) x dA)`,
//! the R^3 mass centroid projected radially back onto the sphere. It is what
//! makes the mesh orthogonal, and orthogonality is what makes the TRiSK
//! operators second order.
//!
//! Non-convergence is detected rather than shipped. Three detectors, all counts
//! or monotone scalars and never fits, refuse with the number that made them
//! refuse; the emit gate in [`crate::mesh::validate`] is the fourth.

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::mesh::density::{DensityField, MeshSpec};
use crate::mesh::derive::Rings;
use crate::mesh::geom::{V3, add, arc, circumcenter, scale, tri_area, unit};
use crate::mesh::hull::{
    TriangulationMode, delaunay_rings, delaunay_triangulation, repair_or_rebuild,
};

/// Counter-based PRNG so a mesh is reproducible on every machine without a
/// crate dependency or global state.
struct SplitMix(u64);

impl SplitMix {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Place `n_target` generators so their number density already follows the
/// spec, before any relaxation.
///
/// Cell AREA scales as `rho^(-1/2)` (since `h ~ rho^(-1/4)`), so the number
/// density of generators must go as `sqrt(rho)`. Points are drawn from a
/// golden-ratio lattice and kept with probability proportional to `sqrt(rho)`,
/// with the acceptance level solved so the count comes out EXACTLY `n_target`.
///
/// This matters for more than tidiness. Lloyd moves a generator about one cell
/// width per sweep, so making a uniform start migrate across the published
/// mesh's 81-cell transition band would cost a hundred sweeps of pure transport
/// before any relaxation began. Biasing the seed does that transport
/// analytically and leaves Lloyd the local packing it is actually good at.
pub fn seed_points(spec: &MeshSpec, n_target: usize) -> MpasResult<Vec<V3>> {
    if n_target < 4 {
        return Err(MpasError::Refusal(format!(
            "{n_target} generators cannot tessellate a sphere; four is the minimum that encloses a volume"
        )));
    }
    // The field is asked millions of times below; prepared once.
    let spec = spec.prepared();
    // Mean acceptance weight, probed on a coarse lattice, sets the overdraw.
    let probe = 20_000.min(n_target * 4).max(4_000);
    let mean_w: f64 = crate::mesh::density::fibonacci_lattice(probe)
        .map(|p| spec.density(p).powf(0.25).powi(2))
        .sum::<f64>()
        / probe as f64;
    if !(mean_w > 0.0) {
        return Err(MpasError::Refusal(
            "the resolution spec has no positive density anywhere; there is nothing to place generators in proportion to".to_string(),
        ));
    }

    let mut lattice = ((n_target as f64 / mean_w) * 1.3).ceil() as usize;
    for _attempt in 0..8 {
        let pts: Vec<V3> = crate::mesh::density::fibonacci_lattice(lattice).collect();
        // sqrt(rho), normalised to a maximum of one
        let w: Vec<f64> = pts.par_iter().map(|&p| spec.density(p).sqrt()).collect();
        let wmax = w.iter().cloned().fold(0.0f64, f64::max);
        if !(wmax > 0.0) {
            return Err(MpasError::Refusal(
                "the resolution spec evaluates to zero density at every lattice point".to_string(),
            ));
        }
        let mut rng = SplitMix(0xA5EE_D000_0000_0000 ^ lattice as u64);
        let u: Vec<f64> = (0..lattice).map(|_| rng.unit()).collect();

        // The accepted count is monotone in the acceptance level, so bisect it
        // to land on exactly n_target rather than on a random draw near it.
        let count = |s: f64| -> usize {
            (0..lattice).filter(|&k| u[k] * wmax < w[k] * s).count()
        };
        if count(1.0) < n_target {
            lattice = (lattice as f64 * 1.6).ceil() as usize;
            continue;
        }
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if count(mid) < n_target { lo = mid } else { hi = mid }
        }
        let mut kept: Vec<usize> = (0..lattice).filter(|&k| u[k] * wmax < w[k] * hi).collect();
        // Bisection lands on the first level that reaches the target; drop the
        // marginal extras from the end so the count is exact.
        kept.truncate(n_target.max(kept.len().min(n_target)));
        while kept.len() > n_target {
            kept.pop();
        }
        if kept.len() == n_target {
            return Ok(kept.into_iter().map(|k| pts[k]).collect());
        }
        lattice = (lattice as f64 * 1.6).ceil() as usize;
    }
    Err(MpasError::Refusal(format!(
        "could not place {n_target} generators in proportion to this spec's density after eight attempts; the density field is too concentrated for a golden-ratio lattice to sample, which means the requested refinement ratio is extreme rather than the placement being wrong"
    )))
}

/// How hard to relax, and when to give up.
#[derive(Debug, Clone, Copy)]
pub struct LloydOptions {
    /// Stop when the MEAN of `delta / h` falls below this. `delta` is the
    /// distance from a generator to its own density-weighted Voronoi centroid
    /// and `h` is the local spacing, so this is a dimensionless measure of how
    /// far the mesh is from centroidal.
    ///
    /// The contract is on the mean and not on the max BY MEASUREMENT. Over 400
    /// sweeps of a uniform 1,200-cell relaxation the mean falls monotonically
    /// from 8.6e-2 to 2.2e-4 while the max wanders between 3e-3 and 1.5e-2 with
    /// no trend: the max is set by a handful of cells next to the twelve
    /// pentagons and it steps every time a Delaunay edge flips. Contracting on
    /// a quantity that does not contract would make the stop condition a
    /// coin toss. The max is measured and reported on every run.
    pub tolerance: f64,
    /// Hard sweep budget. Reaching it without meeting `tolerance` is a refusal,
    /// not a quiet emit.
    pub max_sweeps: usize,
    /// Over-relaxation factor: the generator moves `omega` times the way to its
    /// centroid. 1.0 is plain Lloyd.
    pub omega: f64,
    /// Sweeps of history the stall detector looks back over.
    pub stall_window: usize,
    /// Per-sweep contraction above which progress counts as stalled.
    pub stall_contraction: f64,
    /// Sweeps of history the oscillation detector looks back over.
    pub oscillation_window: usize,
    /// Increases within that window that count as a limit cycle.
    pub oscillation_increases: usize,
    /// The per-sweep `min dv/dc` monitor's floor. DEFAULT-ON: below it the
    /// mesh is inside the near-cocircular class the admission gate refuses
    /// (floor 0.02, published x4 reads 0.0336), and campaign #304 measured
    /// that further relaxation RE-ROLLS that tail rather than draining it,
    /// so a converged-but-degraded state must stop at its best iterate or
    /// refuse -- never quietly ship the last roll of the dice.
    pub monitor_floor: f64,
    /// Sweeps the monitor must read below the floor, consecutively, after
    /// arming, before it stops the relaxation.
    pub monitor_consecutive: usize,
    /// The mean residual at which the monitor arms. Before the mesh is
    /// near-converged the floor is legitimately in flux.
    pub monitor_arm_mean: f64,
    /// Which arm keeps the triangulation between sweeps. Defaults to
    /// [`TriangulationMode::Rebuild`], the class-A arm; see that type for why
    /// the fast arm cannot be the default.
    pub triangulation: TriangulationMode,
}

impl Default for LloydOptions {
    fn default() -> Self {
        LloydOptions {
            // Measured: a healthy relaxation contracts the mean by about 1.4%
            // per sweep and crosses 1e-3 near sweep 100 at these sizes.
            tolerance: 1e-3,
            max_sweeps: 300,
            omega: 1.4,
            stall_window: 20,
            stall_contraction: 0.999,
            oscillation_window: 10,
            oscillation_increases: 5,
            monitor_floor: 0.03,
            monitor_consecutive: 5,
            monitor_arm_mean: 3e-3,
            triangulation: TriangulationMode::Rebuild,
        }
    }
}

/// What a relaxation run did, whether or not it converged.
#[derive(Debug, Clone)]
pub struct LloydOutcome {
    pub sweeps: usize,
    pub max_delta_over_h: f64,
    pub mean_delta_over_h: f64,
    pub history: Vec<f64>,
    /// Per-sweep `min dv/dc`, sampled EVERY sweep by the default-on monitor.
    /// Stamped into every production receipt so a reader can see whether the
    /// relaxation held the admission gate's own metric or wandered.
    pub min_dv_over_dc_trajectory: Vec<f64>,
    /// The final sweep's reading and its edge.
    pub min_dv_over_dc: f64,
    pub min_dv_over_dc_edge: (u32, u32),
    pub wall_seconds: f64,
    pub rings: Rings,
}

/// One Lawson pass over the PREVIOUS triangulation at the MOVED positions:
/// how long it takes, and how many ring steps it finds non-Delaunay.
///
/// This is a measurement, not a mechanism. The relaxation rebuilds the whole
/// spherical Delaunay from scratch after every sweep; whether that is
/// necessary work or redundant work turns on how much the triangulation
/// actually moves, and this is the cost of finding out the cheap way. Nothing
/// here feeds back into the mesh.
///
/// The test is the hull's own: `(i, ring[k-1], ring[k])` is a Delaunay facet
/// wound the way `hull::Face` winds one, and `ring[k+1]` lying OUTSIDE its
/// plane -- which on a set of cospherical points is the empty-circumcircle
/// condition -- means the facet is no longer locally Delaunay.
fn lawson_pass_probe(points: &[V3], rings: &Rings) {
    let t = std::time::Instant::now();
    let mut tests = 0u64;
    let mut bad = 0u64;
    for i in 0..points.len() {
        let ring = rings.ring(i);
        let deg = ring.len();
        if deg < 3 {
            continue;
        }
        for k in 0..deg {
            let a = points[ring[(k + deg - 1) % deg] as usize];
            let b = points[ring[k] as usize];
            let c = points[ring[(k + 1) % deg] as usize];
            tests += 1;
            if crate::mesh::geom::orient3d_sign(points[i], a, b, c) > 0.0 {
                bad += 1;
            }
        }
    }
    crate::mesh::profile::FLIP_PROBE.add(t.elapsed().as_nanos() as u64, tests);
    crate::mesh::profile::FLIP_VIOLATIONS.add(0, bad);
}

/// Cells whose neighbour SET changed between two triangulations of the same
/// point count, and cells whose set is the same but whose ring START moved.
///
/// The second number matters as much as the first: the emitted grid depends
/// on the ring's starting slot, so a triangulation that is topologically the
/// same but rotated is a different file.
fn ring_churn(before: &Rings, after: &Rings) -> (u64, u64) {
    let n = before.n_cells().min(after.n_cells());
    let mut changed = 0u64;
    let mut rotated = 0u64;
    let mut a: Vec<u32> = Vec::with_capacity(12);
    let mut b: Vec<u32> = Vec::with_capacity(12);
    for i in 0..n {
        let ra = before.ring(i);
        let rb = after.ring(i);
        if ra == rb {
            continue;
        }
        a.clear();
        a.extend_from_slice(ra);
        a.sort_unstable();
        b.clear();
        b.extend_from_slice(rb);
        b.sort_unstable();
        if a == b {
            rotated += 1;
        } else {
            changed += 1;
        }
    }
    (changed, rotated)
}

/// Relax `points` toward the density-weighted centroidal tessellation of the
/// field -- the user's [`MeshSpec`] for a uniform or final relaxation, a
/// [`crate::mesh::density::LevelClamp`] for one rung of the graded ladder.
pub fn relax<F: DensityField + Sync>(
    points: &mut Vec<V3>,
    spec: &F,
    opts: &LloydOptions,
) -> MpasResult<LloydOutcome> {
    if !(opts.tolerance > 0.0 && opts.tolerance.is_finite()) {
        return Err(MpasError::Refusal(format!(
            "a relaxation tolerance of {} cannot be reached; the tolerance is the quality contract the mesh is held to and it has to be a positive number",
            opts.tolerance
        )));
    }
    if opts.max_sweeps == 0 {
        return Err(MpasError::Refusal(
            "a sweep budget of zero produces the seed placement unrelaxed; that is not a centroidal tessellation and its edges are not orthogonal to their duals".to_string(),
        ));
    }
    let started = std::time::Instant::now();
    let mut history: Vec<f64> = Vec::with_capacity(opts.max_sweeps);
    let mut worst_history: Vec<f64> = Vec::with_capacity(opts.max_sweeps);
    let mut monitor_traj: Vec<f64> = Vec::with_capacity(opts.max_sweeps);
    // CLASS A vs CLASS B, decided once, here. On the maintained arm the
    // FIRST rings are still taken from a full build compacted in facet order,
    // so sweep 1 is bit-identical on both arms; the two only part company at
    // the first repair.
    let mut tri = if opts.triangulation.is_maintained() {
        Some(delaunay_triangulation(points)?)
    } else {
        None
    };
    let mut rings = match &tri {
        Some(t) => t.rings()?,
        None => delaunay_rings(points)?,
    };

    let mut armed = false;
    let mut consecutive_degrading = 0usize;
    let mut previous_mq = f64::INFINITY;

    for sweep in 1..=opts.max_sweeps {
        let step: Vec<(V3, f64)> = crate::mesh::profile::timed(
            &crate::mesh::profile::LLOYD_STEP,
            points.len() as u64,
            || {
                (0..points.len())
                    .into_par_iter()
                    .map(|i| cell_step(points, &rings, spec, i))
                    .collect()
            },
        );
        let worst = step.iter().map(|(_, r)| *r).fold(0.0f64, f64::max);
        let mean = step.iter().map(|(_, r)| *r).sum::<f64>() / step.len() as f64;
        history.push(mean);
        worst_history.push(worst);

        for (i, (c, _)) in step.iter().enumerate() {
            let moved = if opts.omega == 1.0 {
                *c
            } else {
                let d = crate::mesh::geom::sub(*c, points[i]);
                unit(add(points[i], scale(d, opts.omega))).unwrap_or(*c)
            };
            points[i] = moved;
        }
        if crate::mesh::profile::on() {
            lawson_pass_probe(points, &rings);
        }
        let previous_rings = if crate::mesh::profile::on() {
            Some(rings.clone())
        } else {
            None
        };
        rings = match &mut tri {
            Some(t) => {
                repair_or_rebuild(points, t)?;
                t.rings()?
            }
            None => delaunay_rings(points)?,
        };
        if let Some(prev) = previous_rings {
            let (changed, rotated) = ring_churn(&prev, &rings);
            crate::mesh::profile::RING_CHURN.add(rotated, changed);
        }

        // The DEFAULT-ON per-sweep monitor: the admission gate's own metric,
        // O(E) from the rings just rebuilt, sampled EVERY sweep.
        let (mq, medge) = crate::mesh::profile::timed(
            &crate::mesh::profile::LLOYD_MONITOR,
            points.len() as u64,
            || crate::mesh::surgery::min_dv_over_dc(points, &rings),
        );
        monitor_traj.push(mq);

        if mean < opts.tolerance {
            // Converged. The monitor reading rides out in the outcome -- a
            // sub-floor reading here is the near-cocircular tail that only
            // count-changing surgery drains, and the CALLER's gate (the
            // ladder's level gate, or the emit gate's ratio floor) owns that
            // verdict; the relaxation's own refusal below is for a tail that
            // is actively COLLAPSING while convergence has not arrived.
            return Ok(LloydOutcome {
                sweeps: sweep,
                max_delta_over_h: worst,
                mean_delta_over_h: mean,
                history,
                min_dv_over_dc_trajectory: monitor_traj,
                min_dv_over_dc: mq,
                min_dv_over_dc_edge: medge,
                wall_seconds: started.elapsed().as_secs_f64(),
                rings,
            });
        }

        if mean < opts.monitor_arm_mean {
            armed = true;
        }
        // R1's own wording: below the floor AND TRENDING DOWN, consecutively,
        // after near-convergence. A statically low, wandering tail is the
        // state surgery exists to drain; a collapsing one is the re-roll
        // failure mode and burning more sweeps into it is refused.
        if armed && mq < opts.monitor_floor && mq < previous_mq {
            consecutive_degrading += 1;
        } else {
            consecutive_degrading = 0;
        }
        previous_mq = mq;
        if armed && consecutive_degrading >= opts.monitor_consecutive {
            // This is a DETECTOR WITH A REFUSAL, not a waiver: no floor
            // moves, and the refusal carries both numbers.
            return Err(MpasError::Refusal(format!(
                "the relaxation is collapsing into the near-cocircular class: min dv/dc read {mq:.3e} at edge ({}, {}) -- below the {:.2} monitor floor and strictly falling for {} consecutive sweeps -- after the mean residual reached {mean:.4e} (armed under {:.1e}) without meeting the {:.1e} tolerance. The TRiSK tangential weights divide by dvEdge, so continuing would relax toward a mesh whose worst edge amplifies the tangential wind {:.0}x; the admission gate refuses below 0.02, and campaign #304 measured that more relaxation re-rolls this tail (--sweeps 200/600/2000 all reproduced the same 75.04 m shortest edge) rather than draining it -- draining it needs count-changing surgery, not more sweeps",
                medge.0,
                medge.1,
                opts.monitor_floor,
                opts.monitor_consecutive,
                opts.monitor_arm_mean,
                opts.tolerance,
                1.0 / mq.max(1e-300)
            )));
        }

        // D1 STALL: the residual has stopped contracting -- or contracts too
        // slowly to reach the contract inside the remaining budget. The
        // second clause is an EXTRAPOLATION, not taste: a graded level's
        // post-insertion redistribution legitimately contracts slower than a
        // uniform relaxation (density surplus migrates one cell width per
        // sweep across the annulus), and refusing a run whose own arithmetic
        // says it will finish was this detector's measured false positive.
        if history.len() > opts.stall_window {
            let recent = &history[history.len() - opts.stall_window..];
            let ratios: Vec<f64> = recent
                .windows(2)
                .map(|w| if w[0] > 0.0 { w[1] / w[0] } else { 1.0 })
                .collect();
            let logmean: f64 =
                ratios.iter().map(|r| r.max(1e-30).ln()).sum::<f64>() / ratios.len() as f64;
            let contraction = logmean.exp();
            if contraction > opts.stall_contraction {
                let remaining = opts.max_sweeps - sweep;
                let projected = if contraction < 1.0 {
                    (opts.tolerance / mean).ln() / contraction.ln()
                } else {
                    f64::INFINITY
                };
                if !(projected <= remaining as f64) {
                    return Err(MpasError::Refusal(format!(
                        "the relaxation stalled at mean(delta/h) = {mean:.4e} (worst cell {worst:.4e}) against a target of {:.1e}: over the last {} sweeps the residual contracted by a factor of {:.6} per sweep, which needs {} more sweeps against the {remaining} the budget has left. The mesh is not a centroidal tessellation, so its edge points do not bisect their dual arcs and the mimetic operators lose their second-order cancellation between neighbours",
                        opts.tolerance,
                        opts.stall_window,
                        contraction,
                        if projected.is_finite() {
                            format!("about {:.0}", projected)
                        } else {
                            "infinitely many".to_string()
                        }
                    )));
                }
            }
        }

        // D2 OSCILLATION: a limit cycle, which on a variable mesh is nearly
        // always the density field asking for too steep a ramp.
        if history.len() > opts.oscillation_window {
            let recent = &history[history.len() - opts.oscillation_window..];
            let ups = recent.windows(2).filter(|w| w[1] > w[0] * 1.03).count();
            if ups >= opts.oscillation_increases {
                return Err(MpasError::Refusal(format!(
                    "the relaxation is in a limit cycle: mean(delta/h) rose in {ups} of the last {} sweeps and sits at {mean:.4e} (worst cell {worst:.4e}) against a target of {:.1e}. The steepest per-cell spacing change this field asks for is {:.2}%, against 1.53% in the published variable-resolution mesh; generators cannot settle across a ramp that steep, and a mesh emitted from a cycling relaxation is not centroidal anywhere near it",
                    opts.oscillation_window,
                    opts.tolerance,
                    // The same instrument the build gate reads, so the
                    // number a user is handed when a relaxation limit-cycles
                    // is the number that judged the request. It used to be
                    // taken at 20,000 samples -- points 160 km apart, coarser
                    // still than the gate's own 101 km -- and on a narrow-ramp
                    // spec it under-reported by the same mechanism, telling a
                    // user their ramp was five times the published one when it
                    // was fifty-five times. Not a gate: the branch above is
                    // the oscillation count, and this is the magnitude of the
                    // remedy the message points at.
                    crate::mesh::density::steepest_gradient_reading_of(spec, 20_000).per_cell
                        * 100.0
                )));
            }
        }
    }

    // D3 BUDGET.
    let mean = *history.last().unwrap_or(&f64::NAN);
    let worst = *worst_history.last().unwrap_or(&f64::NAN);
    Err(MpasError::Refusal(format!(
        "the relaxation used its whole budget of {} sweeps and reached mean(delta/h) = {mean:.4e} (worst cell {worst:.4e}), short of the {:.1e} it was asked for. Emitting here would ship a mesh whose generators sit on average {:.3}% of a cell width away from their own centroids; the edge points then miss the midpoints of their dual arcs by that much, which is what turns the second-order mimetic operators into first-order ones. Raise --sweeps, or raise --tolerance and accept the stated quality",
        opts.max_sweeps,
        opts.tolerance,
        mean * 100.0
    )))
}

/// One cell's density-weighted centroid and its normalised displacement.
fn cell_step<F: DensityField>(points: &[V3], rings: &Rings, spec: &F, i: usize) -> (V3, f64) {
    let ring = rings.ring(i);
    let deg = ring.len();
    let centre = points[i];
    let mut verts: Vec<V3> = Vec::with_capacity(deg);
    for j in 0..deg {
        let prev = ring[(j + deg - 1) % deg] as usize;
        let cur = ring[j] as usize;
        match circumcenter(centre, points[prev], points[cur]) {
            Some(v) => verts.push(v),
            None => return (centre, 0.0),
        }
    }
    // THE RADIAL MIDPOINTS ARE EACH ASKED FOR TWICE, and the field is the
    // most expensive thing in this function. Sub-triangle `j` evaluates the
    // density at the midpoint of `(centre, verts[j])` and sub-triangle `j-1`
    // evaluates it at the midpoint of `(verts[j], centre)` -- the same point
    // to the last bit, because componentwise `f64` addition is commutative
    // and `unit` is a function. Three evaluations per sub-triangle is
    // therefore 3*deg calls for 2*deg distinct points: a third of the field
    // evaluations in the hot loop of the whole generator were duplicates.
    //
    // MEASURED SHARE: the centroid sweep is 54.1% of a maintained-arm graded
    // run's wall, and the field -- a 21-vertex polygon signed distance under
    // a `tanh` ramp -- is nearly all of it.
    //
    // This is a MEMO, not a rewrite: the same points, the same values, the
    // same accumulation order into `moment`. Nothing about the arithmetic
    // moves, which is what lets the class-A arm keep it.
    let radial: Vec<Option<(V3, f64)>> = (0..deg)
        .map(|j| unit(add(centre, verts[j])).map(|m| (m, spec.density(m))))
        .collect();
    let mut moment: V3 = [0.0; 3];
    let mut area = 0.0f64;
    for j in 0..deg {
        let a = verts[j];
        let b = verts[(j + 1) % deg];
        let tri = tri_area(centre, a, b);
        area += tri;
        // Three-point edge-midpoint rule on each sub-triangle. A one-point rule
        // carries an O(h^2 grad^2 rho) error -- about 1e-3 h at the published
        // mesh's measured 3.1%-per-cell density variation, which is ten times
        // the residual this relaxation is trying to reach.
        let w3 = tri / 3.0;
        if let Some((m, d)) = radial[j] {
            moment = add(moment, scale(m, w3 * d));
        }
        if let Some(m) = unit(add(a, b)) {
            moment = add(moment, scale(m, w3 * spec.density(m)));
        }
        if let Some((m, d)) = radial[(j + 1) % deg] {
            moment = add(moment, scale(m, w3 * d));
        }
    }
    let centroid = match unit(moment) {
        Some(c) => c,
        None => return (centre, 0.0),
    };
    let h = (2.0 * area.abs() / 3f64.sqrt()).sqrt();
    let delta = arc(centre, centroid);
    let ratio = if h > 0.0 { delta / h } else { 0.0 };
    (centroid, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::density::{MeshSpec, Region, Shape, TransitionField};
    use crate::mesh::geom::from_lat_lon;

    #[test]
    fn the_seed_places_exactly_the_requested_count() {
        for &n in &[500usize, 2_000, 7_777] {
            let pts = seed_points(&MeshSpec::uniform(120.0), n).unwrap();
            assert_eq!(pts.len(), n, "uniform seed count");
        }
    }

    #[test]
    fn the_seed_is_already_denser_where_the_spec_is_finer() {
        // The seed's job is to do the long-range transport analytically. Count
        // generators inside the refined cap against the count a uniform seed
        // would put there -- a count, not a fit.
        let spec = MeshSpec {
            background_km: 120.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 1500.0,
                },
                spacing_km: 30.0,
                transition: TransitionField::Km(600.0),
            }],
            name: None,
        };
        let n = 6_000;
        let centre = from_lat_lon(39f64.to_radians(), (-98f64).to_radians());
        let radius = 1500_000.0 / crate::mesh::geom::EARTH_RADIUS_M;
        let inside = |pts: &[V3]| pts.iter().filter(|&&p| arc(centre, p) < radius).count();

        let biased = seed_points(&spec, n).unwrap();
        let uniform = seed_points(&MeshSpec::uniform(120.0), n).unwrap();
        let (b, u) = (inside(&biased), inside(&uniform));
        // The cap covers a fixed fraction of the sphere; a 4x finer spacing is
        // 16x the cell density there, so the biased seed must put many times
        // more generators in it.
        assert!(
            b > 4 * u,
            "the biased seed put {b} generators in the cap against {u} for a uniform seed"
        );
    }

    #[test]
    fn a_uniform_density_relaxes_to_a_quasi_uniform_mesh() {
        // 1,200 cells on the whole sphere is about 700 km across flats; the
        // spec's spacing is quoted to match so the numbers in the receipt below
        // are the ones the request asked for.
        //
        // Seeded the way the shipped uniform path seeds: from the icosahedral
        // subdivision (1,200 snaps to GP(11,0) = 1,212). A Fibonacci seed at
        // this size relaxes to a mesh with a pentagon-heptagon dislocation
        // whose dual edge measured 6,019.8 m -- in the sub-0.02 dv/dc class
        // the emit gate below refuses, as the port's own DualEdgePolicy
        // would -- and that is a property of the seed, not of the relaxation
        // this test exercises.
        let spec = MeshSpec::uniform(700.0);
        let choice = crate::mesh::icosa::snap_cells(1_200, false).unwrap();
        let n = choice.cells;
        let mut pts = crate::mesh::icosa::seed(choice.m, choice.n).unwrap();
        let opts = LloydOptions {
            tolerance: 2e-3,
            max_sweeps: 300,
            ..Default::default()
        };
        let out = relax(&mut pts, &spec, &opts).unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "uniform {n}: converged in {} sweeps to max(delta/h) {:.3e}, mean {:.3e}, {:.2} s",
            out.sweeps, out.max_delta_over_h, out.mean_delta_over_h, out.wall_seconds
        );
        assert_eq!(pts.len(), n);
        // The mesh must be a proper closed sphere and quasi-uniform: the
        // published "uniform" mesh itself spans 1.23x in across-flats spacing,
        // so that is the yardstick, not 1.0.
        let mesh = crate::mesh::derive::MpasMesh::derive(
            pts.clone(),
            vec![1.0; n],
            &out.rings,
            700_000.0 / crate::mesh::geom::EARTH_RADIUS_M,
        )
        .unwrap();
        // The DEFAULT emit gate, not a loosened one. Orthogonality is a
        // property of the Voronoi construction and not of convergence -- the
        // dual edge lies on the perpendicular bisector of its two generators by
        // definition -- so a relaxed mesh has to meet the same 1e-10 the
        // published NCAR meshes meet.
        let report = crate::mesh::validate::validate(&mesh, crate::mesh::validate::Limits::default())
        .unwrap_or_else(|e| panic!("the relaxed uniform mesh failed the emit gate: {e}"));
        eprintln!(
            "  spacing {:.1} .. {:.1} km ratio {:.4}, worst adjacent {:.4}, nonorthogonality {:.2e}, coordination {:?}",
            report.min_spacing_m / 1000.0,
            report.max_spacing_m / 1000.0,
            report.spacing_ratio,
            report.max_adjacent_spacing_ratio,
            report.max_nonorthogonality,
            report.coordination_histogram
        );
        assert_eq!(report.coordination_defect, 12);
        assert!(
            report.spacing_ratio < 1.5,
            "a uniform spec relaxed to a spacing ratio of {:.3}, worse than the 1.23 the published uniform mesh carries",
            report.spacing_ratio
        );
    }

    #[test]
    fn the_delivered_cell_count_matches_the_prediction_for_a_uniform_spec() {
        // The sizing instrument, checked end to end: ask for a spacing, take
        // the count the model predicts, generate, and measure what came out.
        let spec = MeshSpec::uniform(600.0);
        let predicted = spec.predicted_cells(100_000);
        let n = predicted.round() as usize;
        let mut pts = seed_points(&spec, n).unwrap();
        let out = relax(
            &mut pts,
            &spec,
            &LloydOptions {
                tolerance: 3e-3,
                max_sweeps: 300,
                ..Default::default()
            },
        )
        .unwrap();
        let mesh = crate::mesh::derive::MpasMesh::derive(
            pts.clone(),
            vec![1.0; n],
            &out.rings,
            600_000.0 / crate::mesh::geom::EARTH_RADIUS_M,
        )
        .unwrap();
        let spacing = mesh.spacing_m();
        let mean: f64 = spacing.iter().sum::<f64>() / spacing.len() as f64;
        eprintln!(
            "600 km spec -> predicted {predicted:.1} cells, generated {n}, delivered mean spacing {:.1} km in {} sweeps",
            mean / 1000.0,
            out.sweeps
        );
        assert!(
            (mean / 600_000.0 - 1.0).abs() < 0.03,
            "asked for 600 km, delivered a mean of {:.1} km",
            mean / 1000.0
        );
    }

    #[test]
    fn a_budget_that_cannot_reach_the_contract_is_refused_by_name() {
        let spec = MeshSpec::uniform(300.0);
        let mut pts = seed_points(&spec, 400).unwrap();
        let err = relax(
            &mut pts,
            &spec,
            &LloydOptions {
                tolerance: 1e-12,
                max_sweeps: 3,
                stall_window: 1_000,
                oscillation_window: 1_000,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("used its whole budget"), "{err}");
        assert!(err.contains("second-order mimetic"), "the refusal does not name the breakage: {err}");
    }

    #[test]
    fn an_unreachable_tolerance_stalls_and_says_so() {
        // A tolerance far below the residual floor of a weighted CVT: the
        // relaxation converges to its floor and then stops contracting.
        let spec = MeshSpec::uniform(400.0);
        let mut pts = seed_points(&spec, 300).unwrap();
        let err = relax(
            &mut pts,
            &spec,
            &LloydOptions {
                tolerance: 1e-16,
                max_sweeps: 4_000,
                stall_window: 25,
                stall_contraction: 0.999,
                oscillation_window: 100_000,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("stalled") || err.contains("used its whole budget"),
            "{err}"
        );
    }

    #[test]
    #[ignore]
    fn measure_the_convergence_curve() {
        for (label, spec, n) in [
            ("uniform 1200", MeshSpec::uniform(600.0), 1200usize),
            (
                "cap 4x 1200",
                MeshSpec {
                    background_km: 600.0,
                    regions: vec![Region {
                        shape: Shape::Cap {
                            center_deg: [39.0, -98.0],
                            radius_km: 2500.0,
                        },
                        spacing_km: 150.0,
                        transition: TransitionField::Km(2000.0),
                    }],
                    name: None,
                },
                1200usize,
            ),
        ] {
            for &omega in &[1.0f64, 1.4, 1.8] {
                let mut pts = seed_points(&spec, n).unwrap();
                let opts = LloydOptions {
                    tolerance: 1e-12,
                    max_sweeps: 400,
                    omega,
                    stall_window: 1_000_000,
                    oscillation_window: 1_000_000,
                    ..Default::default()
                };
                let err = relax(&mut pts, &spec, &opts).unwrap_err();
                let _ = err;
                // rerun capturing history via a direct loop
                let mut pts = seed_points(&spec, n).unwrap();
                let mut rings = delaunay_rings(&pts).unwrap();
                let mut marks: Vec<String> = Vec::new();
                for s in 1..=400usize {
                    let step: Vec<(V3, f64)> = (0..pts.len())
                        .map(|i| cell_step(&pts, &rings, &spec, i))
                        .collect();
                    let worst = step.iter().map(|(_, r)| *r).fold(0.0f64, f64::max);
                    let mean = step.iter().map(|(_, r)| *r).sum::<f64>() / step.len() as f64;
                    for (i, (c, _)) in step.iter().enumerate() {
                        let d = crate::mesh::geom::sub(*c, pts[i]);
                        pts[i] = unit(add(pts[i], scale(d, omega))).unwrap_or(*c);
                    }
                    if s % 50 == 0 || s <= 3 {
                        marks.push(format!("{s}:{worst:.2e}/{mean:.2e}"));
                    }
                    rings = delaunay_rings(&pts).unwrap();
                }
                eprintln!("{label} omega={omega}: {}", marks.join("  "));
            }
        }
    }

    #[test]
    fn a_degenerate_request_is_refused_before_any_work() {
        let spec = MeshSpec::uniform(120.0);
        let err = seed_points(&spec, 3).unwrap_err().to_string();
        assert!(err.contains("cannot tessellate a sphere"), "{err}");
        let mut pts = seed_points(&spec, 100).unwrap();
        let err = relax(&mut pts, &spec, &LloydOptions { max_sweeps: 0, ..Default::default() })
            .unwrap_err()
            .to_string();
        assert!(err.contains("seed placement unrelaxed"), "{err}");
    }
}
