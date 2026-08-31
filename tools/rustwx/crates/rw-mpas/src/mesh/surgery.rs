//! Count-changing defect surgery: the degree of freedom Lloyd relaxation
//! does not have.
//!
//! MEASURED MOTIVE. A graded (density-biased Fibonacci) SCVT relaxes its
//! pentagon-heptagon dislocation quads to NEAR-COCIRCULAR at equilibrium: two
//! Voronoi vertices nearly coincide while their cells stay a full spacing
//! apart, so `dvEdge/dcEdge` collapses (registered `v15.150.38857`: 61 edges
//! under the 0.02 admission floor, worst 1.685e-4, `dvEdge` 6.514 m at a
//! 38,657 m `dcEdge` -- a 5,935x TRiSK tangential amplification, and the
//! first non-finite of the recorded run). Campaign #304 measured that more
//! relaxation RE-ROLLS this tail rather than draining it: `--sweeps` 200,
//! 600 and 2000 all reproduce the same 75.04 m shortest edge at 12,000
//! cells. A dislocation is a surplus (or missing) half-row of generators;
//! draining it needs the point COUNT to change locally, which is what the
//! two operators here do:
//!
//! * S1 (overdense, local fill ratio < 1): DELETE the most crowded
//!   generator of the offending quad -- removing a surplus half-row
//!   terminus merges the rows.
//! * S2 (underdense): INSERT a generator at the spacing-true point of the
//!   quad's long diagonal -- completing the missing half-row.
//!
//! Every operation is a point-set edit followed by EXACT re-triangulation
//! (`hull::delaunay_rings`), so `vertexDegree = 3`, Euler, mutual rings and
//! CCW winding hold by construction. There is no dual-mesh surgery here and
//! no chain operator (S3 was deliberately not built: the least-constrained
//! operator, carried unexercised, is where an oscillating loop would hide).
//!
//! WHAT THAT SENTENCE DOES NOT COVER, AND WHAT IT COST (2026-08-26). Those
//! identities are closure identities, and they hold for a mesh that is not a
//! Goldberg mesh. A single point inserted into a locally hexagonal Delaunay
//! lands in the cavity of the two triangles it splits and nothing else, so S2
//! is BORN with four neighbours -- MEASURED at 18 of 18 and 13 of 13
//! insertions on the two graded meshes regenerated for this note -- and the
//! two opposite quad cells go 6 -> 7 with it. That triple sums to zero, so
//! `sum(6 - nEdgesOnCell)` still reads 12 and `validate` used to pass it. The
//! local polish is what anneals a newborn quadrilateral into a legal cell,
//! and it is not obliged to succeed; worse, `local_polish` PINS every cell
//! outside its seeds, and the seeds were the current batch's sites only, so a
//! cell damaged in one round was pinned for every round after it. Registered
//! mesh `v16.66.195630` shipped one survivor -- cell 195615 -- and it killed
//! the forecast (see [`MIN_COORDINATION`]). Coordination is therefore half of
//! this loop's exit test now, the polish is seeded at the damage, and a
//! defect that survives both is a refusal.
//!
//! HALTING IS PROVED BY COUNTING, not hoped for: a hysteresis gap (flag
//! below [`SurgeryOptions::flag_floor`], cured only at or above
//! [`SurgeryOptions::repair_floor`]) kills threshold flip-flop; every site
//! carries a bounded op budget and then ONE cavity resample and is then
//! STUBBORN, which is a refusal; rounds are capped; net point drift is
//! bounded. The coordination clause halts the same way: a cell below
//! [`MIN_COORDINATION`] gets [`COORD_REANNEAL_ROUNDS`] of re-anneal and then
//! its generator is deleted, all inside the same capped round count and the
//! same drift budget. Finite sites times bounded ops halts. Success is a
//! separate, measured question -- the caller's gates.
//!
//! Everything here is deterministic: no RNG anywhere, every ordering is
//! canonical (`f64::total_cmp` plus stable ids), so a repaired mesh is
//! byte-reproducible across machines -- the mesh registry pins grids by
//! SHA-256 and an unreproducible repair would make every registered graded
//! mesh permanently red on regeneration.

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::mesh::density::DensityField;
use crate::mesh::derive::Rings;
use crate::mesh::geom::{EARTH_RADIUS_M, V3, add, arc, circumcenter, cross, scale, tri_area, unit};
use crate::mesh::hull::{
    TriangulationMode, delaunay_rings, delaunay_triangulation, repair_or_rebuild,
};

use serde::Serialize;

/// The fewest edges a cell of this family may have.
///
/// A Goldberg polyhedron carries pentagons, hexagons and (on a graded sphere)
/// heptagons. A QUADRILATERAL is not a legal cell in the family at all, and
/// no closure identity notices one: a quad plus the two heptagons the same
/// operation creates sums to zero, so `sum(6 - nEdgesOnCell)` still reads 12
/// and every Euler, kite and reciprocity check in `validate` passes.
///
/// THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26). Mesh
/// `v16.66.195630` shipped with exactly one 4-coordinated cell -- 195615, at
/// 33.74N 117.65W, fourteen indices from the end of the array, put there by
/// S2 below. It is the worst-shaped cell in that mesh (perimeter/sqrt(area)
/// 4.028 against a hexagonal median of 3.725) and the stiffest by the
/// discrete-Laplacian row sum (1.17x a regular hexagon, rank 0 of 195,630).
/// In every forecast arm run on that mesh the model's potential-temperature
/// maximum AND its vertical-velocity maximum both sit on that one cell at the
/// top model level: theta there stands ~197 K above its initial value at
/// 1,800 s at dt = 100 s, ~198 K at 75 s and ~181 K at 20 s -- timestep
/// converged, so it is not a Courant instability -- and the run ends in a
/// vertical-velocity runaway at the same cell (281 m/s at 9,900 s on the 20 s
/// arm). A finer timestep buys time, not survival: 38 minutes of model time at
/// 100 s, 2 h 45 m at 20 s. A 224,210-cell sibling from the same generator at
/// the same minimum spacing carries the identical defect at cell 224206,
/// three indices from the end. Evidence: gpuwm-hex
/// `tree/evidence/graded-blowup-20260826/`.
pub const MIN_COORDINATION: usize = 5;

/// Every cell whose Delaunay ring is smaller than [`MIN_COORDINATION`], in
/// ascending index order. Empty is the healthy answer.
pub fn cells_below_coordination(rings: &Rings) -> Vec<usize> {
    (0..rings.n_cells())
        .filter(|&i| rings.ring(i).len() < MIN_COORDINATION)
        .collect()
}

/// One Delaunay edge's dislocation reading: the admission gate's own metric
/// plus the cocircularity depth `mesh304_probe` measures, so every number
/// here is commensurable with the recorded failure evidence.
#[derive(Debug, Clone, Copy)]
pub struct QuadReading {
    /// The Delaunay edge, canonical (`i < j`).
    pub i: u32,
    pub j: u32,
    /// The quad's opposite generators: `(i, a, j)` and `(i, j, b)` are the
    /// two Delaunay triangles sharing the edge.
    pub a: u32,
    pub b: u32,
    /// `dvEdge / dcEdge` -- arc between the two circumcentres over the arc
    /// between the two generators. The TRiSK tangential weights divide by
    /// `dvEdge`, so `1/q` is the amplification this edge would ship.
    pub q: f64,
    /// Cocircularity depth in metres: how far `b` sits off the circumcircle
    /// through `(i, a, j)`. Near zero IS the defect (four cocircular sites).
    pub dev_m: f64,
}

/// Visit every Delaunay edge once (canonical `i < j`) with its quad reading.
///
/// Skips an edge only when a circumcentre is degenerate, which
/// `delaunay_rings` has already refused to produce for a valid point set.
pub fn for_each_quad(points: &[V3], rings: &Rings, mut visit: impl FnMut(QuadReading)) {
    for i in 0..points.len() {
        let deg = rings.degree(i);
        for k in 0..deg {
            if let Some(r) = quad_reading(points, rings, i, k) {
                visit(r);
            }
        }
    }
}

/// The reading for cell `i`'s ring slot `k`, or `None` where the serial
/// [`for_each_quad`] skips.
///
/// ONE COPY OF THIS ARITHMETIC, because there are now two readers -- the
/// serial visitor above and the parallel monitor below -- and two copies of
/// a quad formula that drifted apart would put the per-sweep monitor and the
/// surgery detector on different definitions of the same metric while both
/// printed the same name.
#[inline]
fn quad_reading(points: &[V3], rings: &Rings, i: usize, k: usize) -> Option<QuadReading> {
    let ring = rings.ring(i);
    let deg = ring.len();
    let j = ring[k] as usize;
    if j <= i {
        return None;
    }
    let a = ring[(k + deg - 1) % deg] as usize;
    let b = ring[(k + 1) % deg] as usize;
    let (Some(u), Some(v)) = (
        circumcenter(points[i], points[a], points[j]),
        circumcenter(points[i], points[j], points[b]),
    ) else {
        return None;
    };
    let dc = arc(points[i], points[j]);
    if !(dc > 0.0) {
        return None;
    }
    let dv = arc(u, v);
    let dev = (arc(u, points[b]) - arc(u, points[i])).abs() * EARTH_RADIUS_M;
    Some(QuadReading {
        i: i as u32,
        j: j as u32,
        a: a as u32,
        b: b as u32,
        q: dv / dc,
        dev_m: dev,
    })
}

/// The worst `dvEdge/dcEdge` on the triangulation, with its edge.
///
/// This is the per-sweep monitor the relaxation samples: O(E) from rings and
/// circumcentres, the same metric `dual_edge_admission` refuses on, so a
/// trajectory printed from here can be read directly against the 0.02
/// admission floor and the recorded failure numbers.
pub fn min_dv_over_dc(points: &[V3], rings: &Rings) -> (f64, (u32, u32)) {
    // PARALLEL, AND THE SAME ANSWER TO THE BIT. This runs once per Lloyd
    // sweep over every edge -- 24.6% of a maintained-arm relaxation's wall
    // and up to 5.6% of a rebuild-arm one -- and every edge's reading is
    // independent, so the only thing that needed care is WHICH edge is
    // returned when two read the same worst value.
    //
    // The serial loop it replaces kept the FIRST edge, in `(cell, ring slot)`
    // order, that was strictly better than everything before it. The
    // reduction below is written to that rule explicitly -- lexicographic
    // minimum of `(q, cell, slot)`, which is a total order, so it is
    // associative and rayon's join order cannot reach it. `min_by` on `q`
    // alone would not be enough: surgery keys its sites off the returned
    // EDGE, and a different winner on a tie is a different repair and a
    // different mesh.
    let best = (0..points.len())
        .into_par_iter()
        .map(|i| {
            let deg = rings.degree(i);
            let mut b: Option<(f64, u32, u32, u32)> = None;
            for k in 0..deg {
                let Some(r) = quad_reading(points, rings, i, k) else {
                    continue;
                };
                // Strictly better, exactly as the serial loop had it, so a
                // NaN reading never becomes a winner.
                if b.map_or(true, |(bq, _, _, _)| r.q < bq) {
                    b = Some((r.q, i as u32, k as u32, r.j));
                }
            }
            b
        })
        .reduce(
            || None,
            |a, b| match (a, b) {
                (None, x) => x,
                (x, None) => x,
                (Some(x), Some(y)) => {
                    if y.0 < x.0 || (y.0 == x.0 && (y.1, y.2) < (x.1, x.2)) {
                        Some(y)
                    } else {
                        Some(x)
                    }
                }
            },
        );
    match best {
        Some((q, i, _, j)) => (q, (i, j)),
        None => (f64::INFINITY, (0, 0)),
    }
}

/// Voronoi cell area of cell `i`, in steradians (unit sphere).
fn cell_area(points: &[V3], rings: &Rings, i: usize) -> f64 {
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
    area.abs()
}

/// The spacing-true point on the arc `p -> q`: the split point whose two
/// sub-arcs are proportional to the local requested spacing, so grading
/// conformity is PLACED rather than relaxed into existence.
///
/// Solved by two fixed-point iterations of
/// `t <- h(left mid) / (h(left mid) + h(right mid))`, which contracts at
/// `|grad h| / 2` per iteration -- at the 1.53%-per-cell warn-line gradient
/// that is a contraction rate of at most 0.008, so two iterations land
/// within `3e-4 h` of the fixed point. On a uniform field both evaluations
/// are equal and `t` is EXACTLY 0.5, and the exact `unit(p + q)` midpoint is
/// returned -- which is what keeps interior refinement exact GP-doubling.
pub fn spacing_true_point(field: &impl DensityField, p: V3, q: V3) -> V3 {
    let slerp = |t: f64| -> V3 {
        let omega = arc(p, q);
        if omega <= 0.0 {
            return p;
        }
        let so = omega.sin();
        if so == 0.0 {
            return p;
        }
        let wp = ((1.0 - t) * omega).sin() / so;
        let wq = (t * omega).sin() / so;
        unit(add(scale(p, wp), scale(q, wq))).unwrap_or(p)
    };
    let mut t = 0.5f64;
    for _ in 0..2 {
        let hl = field.spacing_m(slerp(0.5 * t));
        let hr = field.spacing_m(slerp(t + 0.5 * (1.0 - t)));
        let denom = hl + hr;
        if !(denom > 0.0) {
            break;
        }
        t = hl / denom;
    }
    if t == 0.5 {
        // Exact midpoint, bit-for-bit the GP-doubling construction.
        unit(add(p, q)).unwrap_or(p)
    } else {
        slerp(t)
    }
}

/// How hard surgery may work, and the hysteresis that makes it halt.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SurgeryOptions {
    /// A quad is FLAGGED for repair below this `dv/dc`.
    pub flag_floor: f64,
    /// A repaired site counts CURED only at or above this. The gap between
    /// this and `flag_floor` is what kills threshold flip-flop.
    pub repair_floor: f64,
    /// The caller's ship floor; drain succeeds only if the finished mesh
    /// clears it. 1.5x the untouchable 0.02 admission floor.
    pub ship_floor: f64,
    /// Rounds per drain call. Measured expectation is under 10.
    pub max_rounds: usize,
    /// Non-overlapping quads repaired per round.
    pub batch_cap: usize,
    /// Graph radius of the post-batch local polish.
    pub local_radius: usize,
    /// Sweeps of local polish after each batch, at omega = 1.0.
    pub polish_sweeps: usize,
    /// Ops a site may receive before its one cavity resample: two chosen by
    /// the fill ratio, then one with the op type FORCED to swap.
    pub site_op_cap: usize,
    /// Which arm keeps the triangulation across the polish sweeps. Defaults
    /// to [`TriangulationMode::Rebuild`]; see that type.
    pub triangulation: TriangulationMode,
}

/// Rounds of re-anneal a cell below [`MIN_COORDINATION`] gets before the
/// deletion operator takes over. Three, because that is one more than the
/// number of rounds the `v16.66.195630` level-1 drain had left after it
/// created its two 4-coordinated cells (rounds 1 and 2, both of which pinned
/// them): a bound that only reproduces the observed failure would prove
/// nothing.
pub const COORD_REANNEAL_ROUNDS: usize = 3;

impl Default for SurgeryOptions {
    fn default() -> Self {
        SurgeryOptions {
            flag_floor: 0.04,
            repair_floor: 0.05,
            ship_floor: 0.03,
            max_rounds: 40,
            batch_cap: 64,
            local_radius: 3,
            polish_sweeps: 8,
            site_op_cap: 3,
            triangulation: TriangulationMode::Rebuild,
        }
    }
}

/// What kind of edit one operation made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    DeleteCrowded,
    InsertDiagonal,
    CavityResample,
    /// S1 aimed at a cell S2 left below [`MIN_COORDINATION`]: deleting that
    /// generator is the exact inverse of the insertion that created it, so
    /// the neighbourhood returns to the topology it had before the operation
    /// rather than to some new one. Spent only after the re-anneal below has
    /// been given its rounds and the cell is still not a legal cell.
    DeleteCoordinationDefect,
}

/// One operation, recorded for the ledger.
#[derive(Debug, Clone, Serialize)]
pub struct OpRecord {
    pub round: usize,
    pub kind: OpKind,
    /// Quad quality that triggered it.
    pub q_before: f64,
    /// Local fill ratio that chose the operator.
    pub fill_ratio: f64,
    /// Where, as a unit vector (sites survive re-indexing; indices do not).
    pub site: V3,
}

/// A site surgery could not cure within its op budget. Reported in the
/// refusal with its coordinates and reading, never silently shipped.
#[derive(Debug, Clone, Serialize)]
pub struct StubbornSite {
    pub site: V3,
    pub q: f64,
    pub ops_spent: usize,
}

/// The full account of a drain call: every count a halting argument needs.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SurgeryLedger {
    pub rounds: usize,
    pub flagged_initial: usize,
    pub inserted: usize,
    pub deleted: usize,
    pub cavity_resamples: usize,
    /// Operated sites whose neighbourhood reads at or above the repair
    /// floor at exit -- CURED under the hysteresis bar.
    pub sites_cured: usize,
    /// Operated sites parked between the flag and repair floors at exit:
    /// shippable (above the flag floor, a fortiori above the ship floor's
    /// gate at the caller), but never claimed cured.
    pub sites_parked_in_hysteresis_band: usize,
    pub ops: Vec<OpRecord>,
    pub stubborn: Vec<StubbornSite>,
    /// `min dv/dc` before the first round and after the last.
    pub min_q_before: f64,
    pub min_q_after: f64,
    /// Rounds spent RE-ANNEALING cells the operators left below
    /// [`MIN_COORDINATION`]. Zero on a drain whose every operation landed on
    /// a legal topology first time, which is what makes this whole clause a
    /// bit-exact no-op on such a drain.
    pub coordination_reanneal_rounds: usize,
    /// Generators deleted because the re-anneal did not make them legal
    /// cells. Counted in `deleted` as well; broken out so the receipt can
    /// say which deletions were dislocation drainage and which were this.
    pub coordination_deletions: usize,
    /// The smallest cell coordination in the mesh at exit. Below
    /// [`MIN_COORDINATION`] is unreachable -- `drain` refuses instead.
    pub min_coordination_after: usize,
}

/// Site bookkeeping across rounds. Sites are POSITIONS, not indices: an
/// index changes when the point count changes, a position does not.
struct SiteTracker {
    sites: Vec<(V3, usize, Option<OpKind>, bool)>, // (pos, ops, last kind, cavity spent)
}

impl SiteTracker {
    fn new() -> Self {
        SiteTracker { sites: Vec::new() }
    }
    /// Find the tracked site within `radius_rad` of `p`, or create one.
    fn find_or_create(&mut self, p: V3, radius_rad: f64) -> usize {
        let mut best: Option<(usize, f64)> = None;
        for (k, (s, _, _, _)) in self.sites.iter().enumerate() {
            let d = arc(*s, p);
            if d < radius_rad && best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((k, d));
            }
        }
        match best {
            Some((k, _)) => {
                // The site's position follows the defect.
                self.sites[k].0 = p;
                k
            }
            None => {
                self.sites.push((p, 0, None, false));
                self.sites.len() - 1
            }
        }
    }
}

/// Index of the generator nearest `to`, by great-circle arc.
///
/// The TIE RULE IS THE CONTRACT: the lowest index among equal distances, which
/// is what a `d < best` linear scan from index 0 returns. Surgery keys sites,
/// polish seeds and cavity centres off this index, so a different winner on a
/// tie is a different repair and a different mesh. The reduction below is
/// therefore written to that rule explicitly rather than left to whatever
/// order rayon happens to join in -- `min_by` alone would not be enough.
fn nearest_generator(points: &[V3], to: V3) -> usize {
    crate::mesh::profile::timed(
        &crate::mesh::profile::NEAREST,
        points.len() as u64,
        || nearest_generator_inner(points, to),
    )
}

fn nearest_generator_inner(points: &[V3], to: V3) -> usize {
    points
        .par_iter()
        .enumerate()
        .map(|(i, &p)| (arc(to, p), i))
        .reduce(
            || (f64::INFINITY, usize::MAX),
            |a, b| {
                if b.0 < a.0 || (b.0 == a.0 && b.1 < a.1) {
                    b
                } else {
                    a
                }
            },
        )
        .1
        .min(points.len().saturating_sub(1))
}

/// Local Lloyd polish: only cells within `radius` ring hops of the given
/// positions move; everything else is pinned. Plain Lloyd (omega 1.0),
/// exactly re-triangulated after every sweep.
fn local_polish<F: DensityField + Sync>(
    points: &mut Vec<V3>,
    field: &F,
    seeds: &[V3],
    radius: usize,
    sweeps: usize,
    mode: TriangulationMode,
) -> MpasResult<Rings> {
    let _polish_timer = crate::mesh::profile::Span::new(
        &crate::mesh::profile::SURGERY_POLISH,
        points.len() as u64,
    );
    // The point set has just been EDITED by the operators above -- inserted,
    // deleted, swap-removed -- so the entry triangulation is a build on both
    // arms. What the maintained arm saves is the `sweeps` rebuilds after it,
    // and the default `polish_sweeps` is 8: nine builds become one.
    let mut tri = if mode.is_maintained() {
        Some(delaunay_triangulation(points)?)
    } else {
        None
    };
    let mut rings = match &tri {
        Some(t) => t.rings()?,
        None => delaunay_rings(points)?,
    };
    for _ in 0..sweeps {
        // Nearest cell to each seed position, then BFS out `radius` hops.
        let mut active = vec![false; points.len()];
        for &s in seeds {
            let nearest = nearest_generator(points, s);
            let mut frontier = vec![nearest];
            active[nearest] = true;
            for _ in 0..radius {
                let mut next = Vec::new();
                for &c in &frontier {
                    for &nb in rings.ring(c) {
                        if !active[nb as usize] {
                            active[nb as usize] = true;
                            next.push(nb as usize);
                        }
                    }
                }
                frontier = next;
            }
        }
        let moves: Vec<(usize, V3)> = (0..points.len())
            .filter(|&i| active[i])
            .map(|i| (i, polish_step(points, &rings, field, i)))
            .collect();
        for (i, c) in moves {
            points[i] = c;
        }
        rings = match &mut tri {
            Some(t) => {
                repair_or_rebuild(points, t)?;
                t.rings()?
            }
            None => delaunay_rings(points)?,
        };
    }
    Ok(rings)
}

/// One plain-Lloyd step for one cell (omega 1.0): the density-weighted
/// constrained centroid, same quadrature as `lloyd::cell_step`.
fn polish_step<F: DensityField>(points: &[V3], rings: &Rings, field: &F, i: usize) -> V3 {
    let ring = rings.ring(i);
    let deg = ring.len();
    let centre = points[i];
    let mut verts: Vec<V3> = Vec::with_capacity(deg);
    for j in 0..deg {
        let prev = ring[(j + deg - 1) % deg] as usize;
        let cur = ring[j] as usize;
        match circumcenter(centre, points[prev], points[cur]) {
            Some(v) => verts.push(v),
            None => return centre,
        }
    }
    // Same memo, same reason, as `lloyd::cell_step`: the radial midpoints are
    // each asked for by two sub-triangles and are bit-identical between them,
    // so a third of the field evaluations here were duplicates. Same points,
    // same values, same accumulation order.
    let radial: Vec<Option<(V3, f64)>> = (0..deg)
        .map(|j| unit(add(centre, verts[j])).map(|m| (m, field.density(m))))
        .collect();
    let mut moment: V3 = [0.0; 3];
    for j in 0..deg {
        let a = verts[j];
        let b = verts[(j + 1) % deg];
        let tri = tri_area(centre, a, b);
        let w3 = tri / 3.0;
        if let Some((m, d)) = radial[j] {
            moment = add(moment, scale(m, w3 * d));
        }
        if let Some(m) = unit(add(a, b)) {
            moment = add(moment, scale(m, w3 * field.density(m)));
        }
        if let Some((m, d)) = radial[(j + 1) % deg] {
            moment = add(moment, scale(m, w3 * d));
        }
    }
    unit(moment).unwrap_or(centre)
}

/// Deterministic cavity resample: delete the quad's four generators plus
/// their 1-ring, then re-place EXACTLY the removed count on a hexagonal
/// tangent-plane lattice around the cavity centre at the local requested
/// spacing, in canonical (radius, angle) order. No RNG anywhere -- the
/// registry's SHA-256 determinism gate must survive this operator.
fn cavity_points(field: &impl DensityField, centre: V3, count: usize) -> Vec<V3> {
    let h = field.spacing_m(centre) / EARTH_RADIUS_M;
    // Tangent frame; the pole singularity has measure zero for real cavities
    // but is still handled by falling back to an arbitrary fixed frame.
    let (e1, e2) = {
        let trial = cross(centre, [0.0, 0.0, 1.0]);
        match unit(trial) {
            Some(e1) => {
                let e2 = unit(cross(centre, e1)).expect("orthogonal frame");
                (e1, e2)
            }
            None => {
                let e1 = unit(cross(centre, [1.0, 0.0, 0.0])).expect("frame");
                let e2 = unit(cross(centre, e1)).expect("frame");
                (e1, e2)
            }
        }
    };
    // Hexagonal lattice sites in the tangent plane, canonically ordered.
    let rings_needed = (count as f64 / 3.0).sqrt().ceil() as i64 + 2;
    let mut sites: Vec<(f64, f64, V3)> = Vec::new();
    for u in -rings_needed..=rings_needed {
        for v in -rings_needed..=rings_needed {
            let x = (u as f64 + 0.5 * v as f64) * h;
            let y = v as f64 * (3f64.sqrt() / 2.0) * h;
            let r = (x * x + y * y).sqrt();
            let ang = y.atan2(x);
            let p = unit(add(centre, add(scale(e1, x), scale(e2, y)))).unwrap_or(centre);
            sites.push((r, ang, p));
        }
    }
    sites.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    sites.truncate(count);
    sites.into_iter().map(|(_, _, p)| p).collect()
}

/// Drain the near-cocircular tail of `points` under `field`.
///
/// On success the returned rings are the exact Delaunay of the repaired
/// points, every flagged quad is cured past the hysteresis floor, and the
/// ledger carries the whole account. `drift_budget_points` bounds
/// `|inserted - deleted|`; the caller states it (1% of a level's insertions
/// for the ladder, 1% of the mesh for a standalone repair).
pub fn drain<F: DensityField + Sync>(
    points: &mut Vec<V3>,
    field: &F,
    opts: &SurgeryOptions,
    drift_budget_points: usize,
) -> MpasResult<(Rings, SurgeryLedger)> {
    let mut ledger = SurgeryLedger::default();
    let mut tracker = SiteTracker::new();
    // Consecutive rounds that have carried a cell below MIN_COORDINATION.
    // The re-anneal gets COORD_REANNEAL_ROUNDS of them before the deletion
    // operator takes over, and `max_rounds` bounds the whole thing, so the
    // clause halts by counting exactly like the rest of this loop.
    let mut coord_stall = 0usize;
    let mut rings = delaunay_rings(points)?;
    let (q0, _) = min_dv_over_dc(points, &rings);
    ledger.min_q_before = q0;
    {
        let mut n = 0usize;
        for_each_quad(points, &rings, |r| {
            if r.q < opts.flag_floor {
                n += 1;
            }
        });
        ledger.flagged_initial = n;
    }

    for round in 0..opts.max_rounds {
        // ---- detect, worst first, canonical order ------------------------
        let mut flagged: Vec<QuadReading> = Vec::new();
        crate::mesh::profile::timed(
            &crate::mesh::profile::QUAD_SCAN,
            points.len() as u64,
            || {
                for_each_quad(points, &rings, |r| {
                    if r.q < opts.flag_floor {
                        flagged.push(r);
                    }
                });
            },
        );
        // THE OTHER HALF OF "REPAIRED", AND THE HALF THIS LOOP USED TO MISS.
        // Every operator here is a point-set edit followed by exact
        // re-triangulation, and a single insertion into a locally hexagonal
        // Delaunay generically lands in the cavity of just the two triangles
        // it splits -- so S2 is BORN with four neighbours, and its two
        // opposite quad cells go 6 -> 7. The polish below is what anneals
        // that into a legal cell, and it is not obliged to succeed. Until
        // 2026-08-26 nothing in this loop ever read a coordination number:
        // the cure test was `dv/dc` alone, cell 195615 of `v16.66.195630`
        // read 0.847 on its own worst dual edge, and a quadrilateral cell
        // shipped inside a mesh this function reported CURED. Coordination is
        // now half of the exit test, so the loop cannot leave one behind.
        let sub5 = cells_below_coordination(&rings);
        let sub5_pos: Vec<V3> = sub5.iter().map(|&i| points[i]).collect();
        if flagged.is_empty() && sub5.is_empty() {
            // No quad under the flag floor: the drain is done, and the ship
            // floor (0.03 < 0.04) is met a fortiori. The HYSTERESIS lives in
            // the site tracker, not here: an operated site that later dips
            // below the flag floor again spends what remains of its op
            // budget rather than starting fresh, so a flip-flopping site
            // runs out of ops, takes its one cavity resample, and then
            // refuses -- it can never oscillate the round counter forever.
            // The repair floor is the CURED bar the ledger reports against:
            // an operated site parked between the two floors is shippable
            // but recorded as parked, never claimed cured.
            ledger.rounds = round;
            let (qf, _) = min_dv_over_dc(points, &rings);
            ledger.min_q_after = qf;
            ledger.min_coordination_after = (0..points.len())
                .map(|i| rings.ring(i).len())
                .min()
                .unwrap_or(0);
            for (s, ops, _, _) in &tracker.sites {
                if *ops == 0 {
                    continue;
                }
                if site_quality(points, &rings, *s) >= opts.repair_floor {
                    ledger.sites_cured += 1;
                } else {
                    ledger.sites_parked_in_hysteresis_band += 1;
                }
            }
            return Ok((rings, ledger));
        }
        flagged.sort_by(|x, y| {
            x.q.total_cmp(&y.q)
                .then_with(|| (x.i, x.j).cmp(&(y.i, y.j)))
        });

        // ---- batch selection: worst-first, non-overlapping ---------------
        let mut used = vec![false; points.len()];
        let mut batch: Vec<QuadReading> = Vec::new();
        for r in &flagged {
            let cells = [r.i as usize, r.j as usize, r.a as usize, r.b as usize];
            if cells.iter().any(|&c| used[c]) {
                continue;
            }
            for &c in &cells {
                used[c] = true;
            }
            batch.push(*r);
            if batch.len() >= opts.batch_cap {
                break;
            }
        }
        if batch.is_empty() && !flagged.is_empty() {
            batch.push(flagged[0]);
        }

        // ---- operate -----------------------------------------------------
        // Deletions are gathered as indices and applied together (descending)
        // so earlier removals cannot shift later ones; insertions are new
        // points appended after. Batch quads are disjoint, so the two sets
        // cannot collide inside one round.
        let mut delete: Vec<usize> = Vec::new();
        let mut insert: Vec<V3> = Vec::new();
        let mut cavity_jobs: Vec<(V3, Vec<usize>)> = Vec::new();
        let mut polish_seeds: Vec<V3> = Vec::new();

        for r in &batch {
            let quad = [r.i as usize, r.j as usize, r.a as usize, r.b as usize];
            let site_pos = unit(add(
                add(points[quad[0]], points[quad[1]]),
                add(points[quad[2]], points[quad[3]]),
            ))
            .unwrap_or(points[quad[0]]);
            let h_here = field.spacing_m(site_pos) / EARTH_RADIUS_M;
            let sk = tracker.find_or_create(site_pos, 0.75 * h_here);
            polish_seeds.push(site_pos);

            let (ops_spent, last_kind, cavity_spent) = {
                let s = &tracker.sites[sk];
                (s.1, s.2, s.3)
            };

            if ops_spent >= opts.site_op_cap {
                if !cavity_spent {
                    // ONE deterministic cavity resample: quad + 1-ring.
                    let mut removal: Vec<usize> = quad.to_vec();
                    for &c in &quad {
                        for &nb in rings.ring(c) {
                            removal.push(nb as usize);
                        }
                    }
                    removal.sort_unstable();
                    removal.dedup();
                    cavity_jobs.push((site_pos, removal));
                    tracker.sites[sk].3 = true;
                    tracker.sites[sk].1 += 1;
                    ledger.cavity_resamples += 1;
                    ledger.ops.push(OpRecord {
                        round,
                        kind: OpKind::CavityResample,
                        q_before: r.q,
                        fill_ratio: f64::NAN,
                        site: site_pos,
                    });
                } else {
                    // Budget exhausted: stubborn. Recorded here; refused after
                    // the loop so one pass reports EVERY stubborn site.
                    if !ledger
                        .stubborn
                        .iter()
                        .any(|s| arc(s.site, site_pos) < 0.75 * h_here)
                    {
                        ledger.stubborn.push(StubbornSite {
                            site: site_pos,
                            q: r.q,
                            ops_spent,
                        });
                    }
                }
                continue;
            }

            // Local fill ratio phi = mean(A_c / A_tgt) over the quad's cells.
            let mut phi = 0.0;
            let mut crowd: Vec<(f64, usize)> = Vec::new();
            for &c in &quad {
                let a_c = cell_area(points, &rings, c);
                let h_c = field.spacing_m(points[c]) / EARTH_RADIUS_M;
                let a_tgt = (3f64.sqrt() / 2.0) * h_c * h_c;
                let ratio = if a_tgt > 0.0 { a_c / a_tgt } else { f64::INFINITY };
                phi += ratio / 4.0;
                crowd.push((ratio, c));
            }
            let phi_kind = if phi < 1.0 {
                OpKind::DeleteCrowded
            } else {
                OpKind::InsertDiagonal
            };
            // Third op on a site: the op type is FORCED to swap. A site that
            // two same-kind ops did not cure is not asking for a third.
            let kind = if ops_spent == opts.site_op_cap - 1 && last_kind == Some(phi_kind) {
                match phi_kind {
                    OpKind::DeleteCrowded => OpKind::InsertDiagonal,
                    _ => OpKind::DeleteCrowded,
                }
            } else {
                phi_kind
            };

            match kind {
                OpKind::DeleteCrowded => {
                    // A dislocation is a surplus half-row terminus; deleting
                    // the most crowded generator merges the rows.
                    crowd.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
                    delete.push(crowd[0].1);
                }
                OpKind::InsertDiagonal => {
                    // The long diagonal is (a, b): the collapsed dual edge's
                    // own diagonal (i, j) is the SHORT one by construction.
                    insert.push(spacing_true_point(
                        field,
                        points[r.a as usize],
                        points[r.b as usize],
                    ));
                }
                OpKind::CavityResample | OpKind::DeleteCoordinationDefect => {
                    unreachable!("chosen only above / by the coordination clause")
                }
            }
            tracker.sites[sk].1 += 1;
            tracker.sites[sk].2 = Some(kind);
            ledger.ops.push(OpRecord {
                round,
                kind,
                q_before: r.q,
                fill_ratio: phi,
                site: site_pos,
            });
        }

        if !ledger.stubborn.is_empty() {
            let worst = ledger
                .stubborn
                .iter()
                .min_by(|x, y| x.q.total_cmp(&y.q))
                .expect("non-empty");
            return Err(MpasError::Refusal(format!(
                "surgery leaves {} stubborn site(s): the worst, at unit vector [{:.6}, {:.6}, {:.6}], still reads dv/dc = {:.3e} after {} operations and its one cavity resample. The TRiSK tangential weights divide by dvEdge, so shipping this site would amplify the wind term across its edge by dc/dv = {:.0}x; the admission gate refuses below 0.02 and this drain's own floor is {}. A stubborn site is a refusal, not a waiver",
                ledger.stubborn.len(),
                worst.site[0],
                worst.site[1],
                worst.site[2],
                worst.q,
                worst.ops_spent,
                1.0 / worst.q.max(1e-300),
                opts.ship_floor
            )));
        }

        // ---- coordination repair -----------------------------------------
        // A cell below MIN_COORDINATION is an operation that has not landed.
        // Two things are done about it, in this order, and BOTH are edits to
        // the same point set the operators above edit -- neither is a filter
        // on the output:
        //
        //  1. RE-ANNEAL. `local_polish` pins every cell outside `radius` hops
        //     of its seeds, and until 2026-08-26 the seeds were the CURRENT
        //     batch's sites only. A cell damaged in round 0 was therefore
        //     pinned for every later round: the two 4-coordinated cells the
        //     `v16.66.195630` level-1 drain created were never moved again by
        //     the three rounds that followed them. Seeding the polish at the
        //     damage is the repair the loop already owned and never aimed.
        //  2. DELETE. A cell the re-anneal will not make legal is a generator
        //     the lattice has no room for, and deleting it is the exact
        //     inverse of the insertion that placed it -- the neighbourhood
        //     returns to a topology that demonstrably existed, rather than to
        //     a new one nobody has measured.
        //
        // Both are bounded; a defect that survives both leaves the loop by
        // the refusal below, which is the correct outcome for a mesh whose
        // cells are not cells of this family.
        let mut coord_seeds: Vec<V3> = Vec::new();
        if !sub5.is_empty() {
            ledger.coordination_reanneal_rounds += 1;
            coord_stall += 1;
            if coord_stall > COORD_REANNEAL_ROUNDS {
                for (&i, &p) in sub5.iter().zip(sub5_pos.iter()) {
                    delete.push(i);
                    ledger.coordination_deletions += 1;
                    ledger.ops.push(OpRecord {
                        round,
                        kind: OpKind::DeleteCoordinationDefect,
                        q_before: f64::NAN,
                        fill_ratio: f64::NAN,
                        site: p,
                    });
                }
                coord_stall = 0;
            }
            coord_seeds.extend_from_slice(&sub5_pos);
        } else {
            coord_stall = 0;
        }
        polish_seeds.extend_from_slice(&coord_seeds);

        // Apply removals together (cavity removals may include quad cells),
        // then append the replacements. A cavity re-places EXACTLY its own
        // removed count, so only S1/S2 contribute to drift.
        let mut removed_set: Vec<usize> = delete.clone();
        let mut cavity_new: Vec<V3> = Vec::new();
        for (centre, removal) in &cavity_jobs {
            let fresh: Vec<usize> = removal
                .iter()
                .copied()
                .filter(|c| !removed_set.contains(c))
                .collect();
            let n_new = fresh.len();
            removed_set.extend(fresh);
            cavity_new.extend(cavity_points(field, *centre, n_new));
        }
        removed_set.sort_unstable();
        removed_set.dedup();
        for &idx in removed_set.iter().rev() {
            points.swap_remove(idx);
        }
        // swap_remove reorders; determinism is preserved because the same
        // input produces the same removal set and the same swaps. Canonical
        // numbering is re-derived from positions at emit time.
        points.extend(cavity_new.iter().copied());
        points.extend(insert.iter().copied());

        // Ledger drift accounting: S1 deletions and S2 insertions only.
        ledger.deleted = ledger
            .ops
            .iter()
            .filter(|o| {
                matches!(
                    o.kind,
                    OpKind::DeleteCrowded | OpKind::DeleteCoordinationDefect
                )
            })
            .count();
        ledger.inserted = ledger
            .ops
            .iter()
            .filter(|o| o.kind == OpKind::InsertDiagonal)
            .count();
        let drift = ledger.inserted.abs_diff(ledger.deleted);
        // The budget floors at TWICE the initially flagged count: a
        // dislocation is a pentagon-heptagon PAIR, and draining it
        // legitimately costs up to two net count changes (S1 deletes a
        // surplus half-row terminus; the exposed partner can need one
        // more -- that is the mechanism, not a rewrite). MEASURED
        // deadlocks that set this floor: a 151-insertion level needed
        // net -7 against a 1%-of-insertions budget of 1, and a 9-flagged
        // level needed net -10. The 1% clause still governs at scale;
        // every count is in the ledger the receipt stamps.
        let budget = drift_budget_points
            .max(2 * ledger.flagged_initial)
            .max(1);
        if drift > budget {
            return Err(MpasError::Refusal(format!(
                "surgery drift: |{} inserted - {} deleted| = {drift} exceeds the {budget}-point budget (1% of the caller's scale, floored at twice the {} initially flagged quads). A repair that rewrites the point count wholesale is not repairing dislocations, it is regenerating the mesh through the back door; the sizing contract (requested vs delivered cells in the receipt) would no longer describe what was delivered",
                ledger.inserted, ledger.deleted, ledger.flagged_initial
            )));
        }

        // ---- exact re-triangulation + bounded local polish ---------------
        rings = local_polish(
            points,
            field,
            &polish_seeds,
            opts.local_radius,
            opts.polish_sweeps,
            opts.triangulation,
        )?;
    }

    let (qf, we) = min_dv_over_dc(points, &rings);
    let left = cells_below_coordination(&rings);
    if !left.is_empty() {
        let c = left[0];
        let (lat, lon) = crate::mesh::geom::lat_lon(points[c]);
        return Err(MpasError::Refusal(format!(
            "surgery used its whole budget of {} rounds and {} cell(s) are still below {MIN_COORDINATION} edges -- the first is cell {c} with {} edges at lat/lon ({:.3}, {:.3}) deg. A Goldberg mesh has pentagons, hexagons and heptagons; a quadrilateral is not a cell of this family, and no closure identity catches one because the same operation makes two heptagons and sum(6 - nEdgesOnCell) still reads 12. MEASURED COST OF SHIPPING ONE (2026-08-26, gpuwm-hex evidence/graded-blowup-20260826/): the single 4-coordinated cell in v16.66.195630 was the worst-shaped and stiffest cell in that mesh, carried a ~197 K standing potential-temperature error at the model top that a 5x smaller timestep did not remove, and ended the forecast in a vertical-velocity runaway at that cell -- 38 minutes of model time at dt 100 s, 2 h 45 m at dt 20 s. {} re-anneal round(s) and {} coordination deletion(s) were spent trying. Refused, never shipped",
            opts.max_rounds,
            left.len(),
            rings.ring(c).len(),
            lat.to_degrees(),
            lon.to_degrees(),
            ledger.coordination_reanneal_rounds,
            ledger.coordination_deletions
        )));
    }
    Err(MpasError::Refusal(format!(
        "surgery used its whole budget of {} rounds and the mesh still reads min dv/dc = {qf:.3e} at edge ({}, {}) against a ship floor of {}. {} operations were spent ({} deletions, {} insertions, {} cavity resamples). The repair loop is not converging on this configuration; shipping it would hand the TRiSK operators a {:.0}x tangential amplification, and an oscillating repair is the D2 limit-cycle class relocated into surgery -- it is refused, not waited out",
        opts.max_rounds,
        we.0,
        we.1,
        opts.ship_floor,
        ledger.ops.len(),
        ledger.deleted,
        ledger.inserted,
        ledger.cavity_resamples,
        1.0 / qf.max(1e-300)
    )))
}

/// Worst quad quality within one spacing of a position -- how a tracked
/// site is re-measured after its indices have changed.
fn site_quality(points: &[V3], rings: &Rings, site: V3) -> f64 {
    // Nearest generator, then the worst quad touching its 2-ring.
    let nearest = nearest_generator(points, site);
    let mut zone = vec![nearest];
    for &nb in rings.ring(nearest) {
        zone.push(nb as usize);
        for &nb2 in rings.ring(nb as usize) {
            zone.push(nb2 as usize);
        }
    }
    zone.sort_unstable();
    zone.dedup();
    let mut worst = f64::INFINITY;
    for_each_quad(points, rings, |r| {
        if zone.binary_search(&(r.i as usize)).is_ok() || zone.binary_search(&(r.j as usize)).is_ok()
        {
            worst = worst.min(r.q);
        }
    });
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE BREAKAGE: `min_dv_over_dc` is the per-sweep monitor AND the metric
    /// surgery keys its repair sites off. It used to be a serial scan that
    /// kept the FIRST edge, in `(cell, ring slot)` order, strictly better than
    /// everything before it. It is now a parallel reduction, and a reduction
    /// written as `min_by` on the quality alone would return a different EDGE
    /// whenever two read the same worst value -- a different repair, a
    /// different mesh, and a mesh that changes between runs on the same
    /// machine because rayon's join order is not fixed.
    ///
    /// This holds the parallel answer against the serial rule it replaced,
    /// spelled out here rather than referenced, so a future rewrite of either
    /// side has something to fail against.
    #[test]
    fn the_parallel_monitor_returns_the_edge_the_serial_scan_returned() {
        for n in [500usize, 2_000, 6_000] {
            let pts: Vec<V3> = {
                let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
                (0..n)
                    .map(|k| {
                        let z = 1.0 - (2 * k + 1) as f64 / n as f64;
                        let r = (1.0 - z * z).max(0.0).sqrt();
                        let t = ga * k as f64;
                        [r * t.cos(), r * t.sin(), z]
                    })
                    .collect()
            };
            let rings = delaunay_rings(&pts).expect("delaunay");
            // The rule the serial loop had, written out.
            let mut worst = f64::INFINITY;
            let mut edge = (0u32, 0u32);
            for_each_quad(&pts, &rings, |r| {
                if r.q < worst {
                    worst = r.q;
                    edge = (r.i, r.j);
                }
            });
            let (q, e) = min_dv_over_dc(&pts, &rings);
            assert_eq!(
                q.to_bits(),
                worst.to_bits(),
                "{n} generators: the parallel monitor read {q:e} where the serial scan read {worst:e}"
            );
            assert_eq!(
                e, edge,
                "{n} generators: the parallel monitor named edge {e:?} where the serial scan named {edge:?}; surgery keys its sites off this edge"
            );
        }
    }

    /// The same reduction, forced onto a genuine TIE. A uniform icosahedral
    /// mesh has whole orbits of exactly equal readings, so this is the case
    /// where a `min_by` written without the index rule picks a different
    /// winner on every thread count.
    #[test]
    fn a_tied_worst_reading_goes_to_the_lowest_cell_and_slot() {
        let pts = crate::mesh::icosa::seed(8, 0).expect("icosahedral seed");
        let rings = delaunay_rings(&pts).expect("delaunay");
        let mut worst = f64::INFINITY;
        let mut edge = (0u32, 0u32);
        let mut ties = 0usize;
        for_each_quad(&pts, &rings, |r| {
            if r.q < worst {
                worst = r.q;
                edge = (r.i, r.j);
                ties = 1;
            } else if r.q == worst {
                ties += 1;
            }
        });
        assert!(
            ties > 1,
            "this fixture was chosen for its ties and produced {ties}; it is not testing what it claims to"
        );
        let (q, e) = min_dv_over_dc(&pts, &rings);
        assert_eq!(q.to_bits(), worst.to_bits());
        assert_eq!(
            e, edge,
            "{ties} edges read the same worst value and the parallel reduction picked {e:?} instead of the lowest-indexed {edge:?}"
        );
    }
    use crate::mesh::density::MeshSpec;
    use crate::mesh::lloyd::{LloydOptions, relax};

    /// A relaxed uniform Goldberg mesh with ONE quad driven near-cocircular
    /// by hand: generator `b` of a quad is moved onto the circumcircle of
    /// its opposite triangle, which is the defect's own geometry (two
    /// Voronoi vertices coincide when four sites share a circle). Returns
    /// (points, spec, the collapsed reading).
    fn mesh_with_forced_dislocation() -> (Vec<V3>, MeshSpec, QuadReading) {
        let mut pts = crate::mesh::icosa::seed(8, 0).unwrap();
        let n = pts.len();
        let spec = MeshSpec::uniform((4.0 * std::f64::consts::PI / n as f64).sqrt() * EARTH_RADIUS_M / 1000.0);
        relax(&mut pts, &spec, &LloydOptions::default()).expect("healthy relax");

        // Pick a quad away from the pentagons and drive b onto the (i,a,j)
        // circumcircle: the dual edge collapses as b approaches it.
        let rings = delaunay_rings(&pts).unwrap();
        let mut chosen: Option<QuadReading> = None;
        for_each_quad(&pts, &rings, |r| {
            if chosen.is_none()
                && [r.i, r.j, r.a, r.b]
                    .iter()
                    .all(|&c| rings.ring(c as usize).len() == 6)
                && r.i > 100
            {
                chosen = Some(r);
            }
        });
        let r = chosen.expect("an interior hexagonal quad exists");
        let u = circumcenter(pts[r.i as usize], pts[r.a as usize], pts[r.j as usize]).unwrap();
        let radius = arc(u, pts[r.i as usize]);
        // Slide b along the great circle from u through b, to radius + a
        // hair -- almost exactly cocircular, dual edge nearly zero.
        let b = pts[r.b as usize];
        let t = crate::mesh::geom::tangent_at(u, crate::mesh::geom::sub(b, u));
        let dir = unit(t).unwrap();
        let target = radius * 1.0005;
        let moved = unit(add(
            scale(u, target.cos()),
            scale(dir, target.sin()),
        ))
        .unwrap();
        pts[r.b as usize] = moved;

        let rings2 = delaunay_rings(&pts).unwrap();
        let mut worst: Option<QuadReading> = None;
        for_each_quad(&pts, &rings2, |q| {
            if worst.as_ref().map(|w| q.q < w.q).unwrap_or(true) {
                worst = Some(q);
            }
        });
        (pts, spec, worst.unwrap())
    }

    #[test]
    fn the_detector_reads_a_forced_cocircular_quad_in_both_directions() {
        // Healthy direction: a relaxed Goldberg mesh reads 0.39-class.
        let mut pts = crate::mesh::icosa::seed(6, 0).unwrap();
        let n = pts.len();
        let spec = MeshSpec::uniform(
            (4.0 * std::f64::consts::PI / n as f64).sqrt() * EARTH_RADIUS_M / 1000.0,
        );
        relax(&mut pts, &spec, &LloydOptions::default()).unwrap();
        let rings = delaunay_rings(&pts).unwrap();
        let (q_healthy, _) = min_dv_over_dc(&pts, &rings);
        assert!(
            q_healthy > 0.3,
            "a relaxed Goldberg mesh reads {q_healthy:.3}, not the published class"
        );

        // Refusing direction: the forced quad reads under the 0.04 flag.
        let (_, _, worst) = mesh_with_forced_dislocation();
        assert!(
            worst.q < 0.04,
            "the forced cocircular quad reads {:.3e}, not under the flag floor",
            worst.q
        );
        // And its cocircularity depth is metre-scale -- the probe's own
        // signature of the defect (dev collapses as the quad degenerates).
        assert!(
            worst.dev_m < 10_000.0,
            "cocircularity depth {:.1} m does not read as near-cocircular",
            worst.dev_m
        );
    }

    #[test]
    fn drain_cures_the_forced_dislocation_within_its_op_budget() {
        let (mut pts, spec, worst) = mesh_with_forced_dislocation();
        let n_before = pts.len();
        let (rings, ledger) = drain(
            &mut pts,
            &spec,
            &SurgeryOptions::default(),
            (n_before / 100).max(1),
        )
        .unwrap_or_else(|e| panic!("drain refused a single forced dislocation: {e}"));
        assert!(ledger.min_q_before <= worst.q * 1.5);
        assert!(
            ledger.min_q_after >= SurgeryOptions::default().ship_floor,
            "drained to {:.3e}, under the ship floor",
            ledger.min_q_after
        );
        assert!(ledger.rounds <= 10, "took {} rounds", ledger.rounds);
        assert!(ledger.stubborn.is_empty());
        // The drift stayed inside the 1% budget and the mesh stayed closed.
        assert!(pts.len().abs_diff(n_before) <= (n_before / 100).max(1));
        assert_eq!(rings.n_cells(), pts.len());
        let (q_final, _) = min_dv_over_dc(&pts, &rings);
        assert!(q_final >= 0.03);
    }

    #[test]
    fn drain_is_deterministic_to_the_bit() {
        let (pts0, spec, _) = mesh_with_forced_dislocation();
        let mut a = pts0.clone();
        let mut b = pts0;
        let opts = SurgeryOptions::default();
        let budget = (a.len() / 100).max(1);
        drain(&mut a, &spec, &opts, budget).unwrap();
        drain(&mut b, &spec, &opts, budget).unwrap();
        assert_eq!(a.len(), b.len());
        for k in 0..a.len() {
            assert_eq!(a[k], b[k], "surgery is not deterministic at point {k}");
        }
    }

    #[test]
    fn a_clean_mesh_drains_as_a_no_op_in_zero_rounds() {
        let mut pts = crate::mesh::icosa::seed(6, 0).unwrap();
        let n = pts.len();
        let spec = MeshSpec::uniform(
            (4.0 * std::f64::consts::PI / n as f64).sqrt() * EARTH_RADIUS_M / 1000.0,
        );
        relax(&mut pts, &spec, &LloydOptions::default()).unwrap();
        let before = pts.clone();
        let (_, ledger) = drain(&mut pts, &spec, &SurgeryOptions::default(), 1).unwrap();
        assert_eq!(ledger.rounds, 0);
        assert!(ledger.ops.is_empty());
        assert_eq!(before, pts, "a no-op drain moved a point");
    }

    /// The placement S2 makes, in isolation: a generator dropped on the
    /// common circumcentre of a near-cocircular quad. That point is inside
    /// the circumcircles of exactly the quad's two triangles and nothing
    /// else, so its Delaunay ring is exactly the four quad cells and its
    /// Voronoi cell is a QUADRILATERAL -- the geometry measured in
    /// `v16.66.195630` cell 195615 and in the 224,210-cell sibling's 224206.
    /// Returns (points, spec, the index of the 4-coordinated generator).
    fn mesh_with_an_s2_quadrilateral() -> (Vec<V3>, MeshSpec, usize) {
        let mut pts = crate::mesh::icosa::seed(8, 0).unwrap();
        let n = pts.len();
        let spec = MeshSpec::uniform(
            (4.0 * std::f64::consts::PI / n as f64).sqrt() * EARTH_RADIUS_M / 1000.0,
        );
        relax(&mut pts, &spec, &LloydOptions::default()).expect("healthy relax");
        let rings = delaunay_rings(&pts).unwrap();
        let mut quads: Vec<QuadReading> = Vec::new();
        for_each_quad(&pts, &rings, |r| quads.push(r));
        quads.sort_by(|x, y| (x.i, x.j).cmp(&(y.i, y.j)));
        for q in quads.iter().take(64) {
            let Some(u) = circumcenter(pts[q.i as usize], pts[q.a as usize], pts[q.j as usize])
            else {
                continue;
            };
            let mut trial = pts.clone();
            trial.push(u);
            let k = trial.len() - 1;
            let rings2 = delaunay_rings(&trial).unwrap();
            if rings2.ring(k).len() < MIN_COORDINATION {
                return (trial, spec, k);
            }
        }
        panic!("no circumcentre placement in the first 64 canonical quads produced a sub-5 cell");
    }

    #[test]
    fn an_s2_placement_makes_a_quadrilateral_that_no_closure_identity_catches() {
        let (pts, _spec, k) = mesh_with_an_s2_quadrilateral();
        let rings = delaunay_rings(&pts).unwrap();
        assert_eq!(
            rings.ring(k).len(),
            4,
            "the circumcentre placement did not land a four-neighbour generator"
        );
        assert_eq!(cells_below_coordination(&rings), vec![k]);
        // And this is exactly why it shipped: the SAME operation makes two
        // heptagons, so the total coordination defect still reads 12 and
        // every Euler-class check in `validate` still passes.
        let defect: i64 = (0..pts.len())
            .map(|i| 6 - rings.ring(i).len() as i64)
            .sum();
        assert_eq!(
            defect, 12,
            "the sphere's coordination defect is 12 with the quadrilateral present -- that is the point"
        );
    }

    #[test]
    fn drain_never_returns_a_cell_below_the_coordination_floor() {
        // THE DEFECT, AS THE GENERATOR MAKES IT. Before the coordination
        // clause this drain returned at round 0 with the quadrilateral
        // untouched: `flagged` reads only dv/dc, cell 195615's own worst dual
        // edge read 0.847, and the ledger said CURED.
        let (mut pts, spec, k) = mesh_with_an_s2_quadrilateral();
        let n_before = pts.len();
        assert_eq!(delaunay_rings(&pts).unwrap().ring(k).len(), 4);
        let (rings, ledger) = drain(
            &mut pts,
            &spec,
            &SurgeryOptions::default(),
            (n_before / 100).max(1),
        )
        .unwrap_or_else(|e| panic!("drain refused a single quadrilateral it can repair: {e}"));
        assert!(
            cells_below_coordination(&rings).is_empty(),
            "drain returned a mesh with cells below coordination {MIN_COORDINATION}: {:?}",
            cells_below_coordination(&rings)
        );
        assert!(ledger.min_coordination_after >= MIN_COORDINATION);
        assert!(
            ledger.coordination_reanneal_rounds > 0,
            "the coordination clause never engaged on a mesh that needed it"
        );
        assert_eq!(rings.n_cells(), pts.len());
    }

    #[test]
    fn the_coordination_clause_is_a_no_op_on_a_drain_that_never_needs_it() {
        // FIXED MEANS DEFAULT, and default must not move healthy bytes: a
        // drain whose operations all land on legal topology must produce the
        // same points it always did. `v20.80.151649` is the registered mesh
        // this protects.
        let (mut pts, spec, worst) = mesh_with_forced_dislocation();
        let budget = (pts.len() / 100).max(1);
        let (_, ledger) = drain(&mut pts, &spec, &SurgeryOptions::default(), budget).unwrap();
        assert!(ledger.min_q_before <= worst.q * 1.5);
        assert_eq!(
            ledger.coordination_reanneal_rounds, 0,
            "this fixture's drain never carries a sub-5 cell, so the clause must not fire"
        );
        assert_eq!(ledger.coordination_deletions, 0);
        assert!(ledger.min_coordination_after >= MIN_COORDINATION);
    }

    #[test]
    fn the_spacing_true_point_is_the_exact_midpoint_on_a_uniform_field() {
        let spec = MeshSpec::uniform(120.0);
        let p = crate::mesh::geom::from_lat_lon(0.3, 1.0);
        let q = crate::mesh::geom::from_lat_lon(0.33, 1.04);
        let m = spacing_true_point(&spec, p, q);
        let exact = unit(add(p, q)).unwrap();
        assert_eq!(m, exact, "uniform-field split is not the bit-exact midpoint");
    }
}

/// A compact summary for the receipt.
#[derive(Debug, Clone, Serialize)]
pub struct SurgerySummary {
    pub rounds: usize,
    pub flagged_initial: usize,
    pub sites_cured: usize,
    pub sites_parked_in_hysteresis_band: usize,
    pub inserted: usize,
    pub deleted: usize,
    pub cavity_resamples: usize,
    pub stubborn: usize,
    pub min_q_before: f64,
    pub min_q_after: f64,
    pub coordination_reanneal_rounds: usize,
    pub coordination_deletions: usize,
    pub min_coordination_after: usize,
}

impl From<&SurgeryLedger> for SurgerySummary {
    fn from(l: &SurgeryLedger) -> Self {
        SurgerySummary {
            rounds: l.rounds,
            flagged_initial: l.flagged_initial,
            sites_cured: l.sites_cured,
            sites_parked_in_hysteresis_band: l.sites_parked_in_hysteresis_band,
            inserted: l.inserted,
            deleted: l.deleted,
            cavity_resamples: l.cavity_resamples,
            stubborn: l.stubborn.len(),
            min_q_before: l.min_q_before,
            min_q_after: l.min_q_after,
            coordination_reanneal_rounds: l.coordination_reanneal_rounds,
            coordination_deletions: l.coordination_deletions,
            min_coordination_after: l.min_coordination_after,
        }
    }
}

