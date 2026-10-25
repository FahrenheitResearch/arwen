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

/// The quad reading below which G4 asks where a dislocation is. Unchanged by
/// the 2026-08-31 re-aiming: what moved was WHERE such a reading is allowed
/// to sit, never how bad a reading has to be to be asked about.
const Q_DISLOCATION: f64 = 0.10;

/// Fewest interior plateau cells the plateau top-up reads its delivered area
/// ratio from. Below this the "interior" is a few rings of boundary cells,
/// whose Voronoi areas reach into the unrefined ramp beside them, so the
/// ratio would measure the boundary and not the plateau's crystal; such a
/// plateau holds too few cells to move any median either.
const PLATEAU_MIN_INTERIOR: usize = 64;

/// What one level's plateau top-up measured and did, for the receipt.
///
/// THE BREAKAGE IT FIXES (box J, 2026-10-24, hex 0.4.4 / gpuwm 2.9.6): every
/// refined plateau was an exact doubled copy of the level before, so it
/// inherited that level's packing shortfall, and the count-exact top-up was
/// barred from the plateau, so the cells the plateau was short went to the
/// ramps instead. A 1 km core of 150 km on the 64 km ladder delivered
/// 1.029 km in its core while the all-hexagon median read the background and
/// passed; the same core at 300 km, two thirds of the mesh, was refused at
/// 1.0209 against 1.0201.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PlateauTopUp {
    /// Points inside the plateau (level field held at exactly `h_l`) right
    /// after its doubling.
    pub cells: usize,
    /// Of those, the ones whose whole ring is also in the plateau: the cells
    /// the area ratio is read from.
    pub interior_cells: usize,
    /// Mean interior Voronoi area over the area one `h_l` hexagon owns,
    /// `(sqrt 3 / 2) h_l^2` -- the same convention as the sizing integral.
    /// 1.0 is a plateau holding exactly its count.
    pub delivered_area_ratio: f64,
    /// Cells the plateau is owed: `cells * (delivered_area_ratio - 1)`,
    /// rounded; zero or negative means no top-up (a surplus is left to the
    /// anneal, as the global top-up leaves one).
    pub owed: i64,
    /// Top-up points actually placed inside the plateau.
    pub inserted: usize,
}

/// What one level measured, for the receipt.
#[derive(Debug, Clone, Serialize)]
pub struct LevelReport {
    pub level: usize,
    pub level_spacing_km: f64,
    pub inserted: usize,
    pub insert_batches: usize,
    /// The coarsest field spacing the ladder had inserted at once this level
    /// finished -- its refinement front, and the yardstick G4 classifies
    /// dislocations against. Stamped so the front is a receipt number rather
    /// than an in-flight one nobody can check afterwards.
    pub insert_front_spacing_km: f64,
    pub relaxation_sweeps: usize,
    pub relaxation_mean_delta_over_h: f64,
    pub min_dv_over_dc_after_anneal: f64,
    pub surgery: SurgerySummary,
    /// Delivered-over-requested spacing against the LEVEL field.
    pub delivered_median: f64,
    pub delivered_p05: f64,
    pub delivered_p95: f64,
    /// The plateau top-up's measurement and insertion.
    pub plateau: PlateauTopUp,
    /// Delivered-over-requested median over the plateau's hexagons alone,
    /// after surgery. The all-hexagon median above reads whichever part of
    /// the mesh is the majority; this reads the core the level was bought for.
    pub delivered_plateau_median: Option<f64>,
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
///
/// On every core: each cell is independent and the result is collected by
/// cell index, so it is the serial vector bit for bit (THE BREAKAGE: a
/// whole-mesh pass on one thread at every level gate).
fn delivered_spacing_m(points: &[V3], rings: &Rings) -> Vec<f64> {
    (0..points.len())
        .into_par_iter()
        .map(|i| (2.0 * voronoi_area_sr(points, rings, i) / 3f64.sqrt()).sqrt() * EARTH_RADIUS_M)
        .collect()
}

/// Voronoi area of cell `i` on the unit sphere (steradians), from the
/// circumcentres of its Delaunay fan; 0.0 for a degenerate fan.
fn voronoi_area_sr(points: &[V3], rings: &Rings, i: usize) -> f64 {
    use crate::mesh::geom::{circumcenter, tri_area};
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

/// A fixed 64-bit mix of a canonical edge id: a deterministic, spatially
/// uncorrelated ranking for the plateau top-up (SplitMix64's finaliser; no
/// generator state, so the rank of an edge is a function of the edge alone).
fn edge_rank(i: u32, j: u32) -> u64 {
    let mut z = ((i as u64) << 32 | j as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// How close to `h_l` the level field must read for a point to be on the
/// level's PLATEAU: a tenth of a percent, twenty times inside the 2% level
/// delivery bound.
///
/// THE BREAKAGE A ROUNDING TOLERANCE HAD: a region core is a tanh plateau,
/// and the tanh saturates to exactly 1.0 only 19 ramp widths inside the
/// boundary. A 150 km core under an 18 km ramp is 8.3 widths deep at its
/// centre, so its field reads h_l * (1 + 6e-8) there and never reaches h_l
/// within 1e-9 anywhere: the whole core counted as "graded". MEASURED on the
/// 300 km core scaled by 4 (box L): a 1e-9 test found 46,543 plateau cells
/// of the 296,000 in the core.
const PLATEAU_TOLERANCE: f64 = 1e-3;

/// Is `p` on this level's plateau -- the level field at `h_l` to within
/// [`PLATEAU_TOLERANCE`]? The same predicate keeps the global top-up off the
/// plateau, so the plateau's cells come from its own top-up and the ramps'
/// from theirs, never one from the other.
fn on_plateau(field: &LevelClamp<'_>, p: V3) -> bool {
    field.spacing_m(p) <= field.level_spacing_m * (1.0 + PLATEAU_TOLERANCE)
}

/// WHAT THE PLATEAU IS OWED, read on the PARENT triangulation before the
/// level splits anything.
///
/// A doubled plateau holds four times the cells its parent held in the same
/// ground (each cell keeps its point and gains three edge midpoints), so it
/// carries the parent's shortfall forward, and each level's anneal adds its
/// own: MEASURED, the level-0 Goldberg crystal's local spacing under a core
/// was off by a few percent (area ratio 1.045 under 35N 97W, 0.968 under
/// 48N 10E), and a 1 km core came out 2.9% coarse after six levels. The count
/// it is owed is read from the cells themselves: the mean Voronoi area of the
/// plateau's INTERIOR parent cells (whole ring on the plateau, so no area
/// reaches into the ramp beside it), shared among the children the doubling
/// makes of each parent, over the area one `h_l` hexagon owns. That ratio, less one, times the plateau's doubled
/// count is what it is short.
///
/// Read BEFORE the split so the level still splits every edge it qualifies
/// in one pass of one triangulation -- the consistent one-to-four doubling --
/// and only the count the ramps may take changes (the caller reserves it).
/// MEASURED: splitting the plateau as its own pass first and reading it after
/// moved the plateau's rim edges between two triangulations, and the defect
/// census rose from 1.52% to 2.30% of cells on the 80 -> 5 km CONUS globe,
/// which owed nothing.
fn plateau_owed(points: &[V3], rings: &Rings, field: &LevelClamp<'_>) -> PlateauTopUp {
    let n = points.len();
    let in_plateau: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| on_plateau(field, points[i]))
        .collect();
    let parents = in_plateau.iter().filter(|&&b| b).count();
    // The doubled plateau: its parents plus one child per edge whose midpoint
    // is on it (V + E, which is GP(2m,2n)'s count on a whole sphere).
    let midpoints: usize = (0..n)
        .into_par_iter()
        .map(|i| {
            rings
                .ring(i)
                .iter()
                .filter(|&&j| {
                    (j as usize) > i
                        && unit(add(points[i], points[j as usize])).is_some_and(|m| on_plateau(field, m))
                })
                .count()
        })
        .sum();
    let cells = parents + midpoints;
    let h_rad = field.level_spacing_m / EARTH_RADIUS_M;
    let target_area_sr = 3f64.sqrt() / 2.0 * h_rad * h_rad;
    // Per-cell areas on every core, summed in cell order so the count is the
    // same to the bit on any thread count.
    let areas: Vec<Option<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !in_plateau[i] || !rings.ring(i).iter().all(|&j| in_plateau[j as usize]) {
                return None;
            }
            Some(voronoi_area_sr(points, rings, i))
        })
        .collect();
    let mut interior_cells = 0usize;
    let mut sum = 0.0f64;
    for a in areas.into_iter().flatten() {
        interior_cells += 1;
        sum += a;
    }
    let mut out = PlateauTopUp {
        cells,
        interior_cells,
        delivered_area_ratio: 0.0,
        owed: 0,
        inserted: 0,
    };
    if interior_cells < PLATEAU_MIN_INTERIOR || parents == 0 {
        return out;
    }
    // A parent's area shared among the children the doubling makes of it
    // (cells / parents, four in a crystal), over one h_l hexagon's area.
    let child_area_sr = sum / interior_cells as f64 * parents as f64 / cells as f64;
    out.delivered_area_ratio = child_area_sr / target_area_sr;
    out.owed = (out.cells as f64 * (out.delivered_area_ratio - 1.0)).round() as i64;
    out
}

/// TOP THE DOUBLED PLATEAU UP by `owed` points.
///
/// The points go on plateau edges ranked by a fixed hash of the edge id, and
/// each placed point first keeps the cells next to the last one free, so the
/// insertions spread over the plateau like blue noise instead of clumping:
/// the anneal and surgery then absorb each as a separate small defect. This
/// does put insertions into the doubled crystal, which the old global top-up
/// refused to do; the crystal it kept was the wrong spacing, and G4 already
/// explains plateau irregularity by the front (`h_l` is always inside it).
fn plateau_insert(
    points: &mut Vec<V3>,
    field: &LevelClamp<'_>,
    owed: usize,
    front_spacing_m: &mut f64,
) -> MpasResult<usize> {
    if owed == 0 {
        return Ok(0);
    }
    let rings = delaunay_rings(points)?;
    let n = points.len();
    let in_plateau: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| on_plateau(field, points[i]))
        .collect();
    // Candidate edges: both ends on the plateau, ranked by the edge hash.
    let mut candidates: Vec<(u64, u32, u32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut local: Vec<(u64, u32, u32)> = Vec::new();
            if !in_plateau[i] {
                return local;
            }
            for &j in rings.ring(i) {
                let j = j as usize;
                if j <= i || !in_plateau[j] {
                    continue;
                }
                local.push((edge_rank(i as u32, j as u32), i as u32, j as u32));
            }
            local
        })
        .collect::<Vec<_>>()
        .concat();
    // A total order: the hash, then the unique canonical id.
    candidates.par_sort_unstable();
    let mut chosen: Vec<(u32, u32)> = Vec::with_capacity(owed);
    // Pass 1 keeps a one-ring gap around every placed point; pass 2, only if
    // the plateau is owed more than that spacing allows, only forbids a
    // shared cell (the global top-up's own rule).
    let mut blocked = vec![false; n];
    let mut used = vec![false; n];
    for pass in 0..2 {
        for &(_, i, j) in &candidates {
            if chosen.len() >= owed {
                break;
            }
            let (iu, ju) = (i as usize, j as usize);
            if used[iu] || used[ju] || (pass == 0 && (blocked[iu] || blocked[ju])) {
                continue;
            }
            used[iu] = true;
            used[ju] = true;
            for &c in rings.ring(iu).iter().chain(rings.ring(ju)) {
                blocked[c as usize] = true;
            }
            blocked[iu] = true;
            blocked[ju] = true;
            chosen.push((i, j));
        }
    }
    let new_points: Vec<V3> = chosen
        .par_iter()
        .map(|&(i, j)| surgery::spacing_true_point(field, points[i as usize], points[j as usize]))
        .collect();
    // Plateau points sit at h_l, inside every front the ladder has; the
    // front is still advanced through the same reading for one rule.
    if !chosen.is_empty() {
        *front_spacing_m = front_spacing_m.max(field.level_spacing_m);
    }
    let added = new_points.len();
    points.extend(new_points);
    Ok(added)
}

/// What one level's insertion pass did.
///
/// `front_spacing_m` is the essential addition: the COARSEST level-field
/// spacing at any midpoint this level actually inserted at. It is the
/// ladder's own measurement of how far outward its refinement reached, and
/// G4 below classifies dislocations against it instead of against a
/// hard-coded fraction of the background. Nothing else can answer that
/// question: the top-up's reach is set by the level's cell DEFICIT against
/// a ratio ranking, which is a property of the run, not of the spec.
struct InsertOutcome {
    inserted: usize,
    batches: usize,
    front_spacing_m: f64,
    plateau: PlateauTopUp,
}

/// One level's insertion pass: split every Delaunay edge longer than
/// `beta * h_bar_l` at its spacing-true point, batch by batch, until no
/// edge qualifies; then a COUNT-EXACT top-up.
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
) -> MpasResult<InsertOutcome> {
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
    // The outermost field value this level inserted at; 0.0 until it does.
    let mut front_spacing_m = 0.0f64;

    // ---- the plateau's own count -----------------------------------------
    // Measured on the parent triangulation in batch 0 (`plateau_owed`); what
    // it is owed is RESERVED out of the level count the splitting may take,
    // and placed inside the doubled plateau once splitting is done, so the
    // level stays count-exact and the ramps stop receiving the plateau's
    // cells.
    let mut plateau = PlateauTopUp::default();
    let mut reserve = 0i64;
    for batch in 0..=MAX_INSERT_BATCHES {
        let need = n_target - points.len() as i64 - reserve;
        if need <= 0 {
            batches_used = batch;
            break;
        }
        let rings = delaunay_rings(points)?;
        if batch == 0 {
            plateau = plateau_owed(points, &rings, field);
            if plateau.owed > 0 {
                reserve = plateau.owed.min(need - 1);
            }
        }
        let need = n_target - points.len() as i64 - reserve;
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
        // The accepted set's outermost field value, measured on the same
        // midpoints the splitting criterion read. One evaluation per ACCEPTED
        // edge against the ~3N the scan above already spent, so the ladder
        // learns where its own front is for a rounding error of its cost.
        for &(_, i, j) in &qualifying {
            if let Some(mid) = unit(add(points[i as usize], points[j as usize])) {
                front_spacing_m = front_spacing_m.max(field.spacing_m(mid));
            }
        }
        let new_points: Vec<V3> = qualifying
            .par_iter()
            .map(|&(_, i, j)| {
                surgery::spacing_true_point(field, points[i as usize], points[j as usize])
            })
            .collect();
        inserted_total += new_points.len();
        points.extend(new_points);
    }

    // ---- the plateau's reserved points, inside the doubled plateau -------
    if reserve > 0 && inserted_total > 0 {
        plateau.inserted = plateau_insert(points, field, reserve as usize, &mut front_spacing_m)?;
        inserted_total += plateau.inserted;
    }

    // ---- count-exact top-up ----------------------------------------------
    // Nothing to do on a level that inserted nothing: the field never
    // crossed this level's spacing and the count belongs to level 0's snap.
    if inserted_total == 0 {
        return Ok(InsertOutcome {
            inserted: 0,
            batches: batches_used,
            front_spacing_m: 0.0,
            plateau,
        });
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
                    // plateau (where the clamp holds the field at exactly
                    // h_l) was already topped up to its OWN count by
                    // `plateau_top_up`, so what is left here is the annulus
                    // and the tanh tail, and a point dropped on the plateau
                    // now would be one it is not owed.
                    let h_bar_m = field.spacing_m(mid);
                    // AT the level means within the plateau tolerance of it
                    // (never an exact `<=`, which made the count hinge on the
                    // last bit of a km -> m round trip): the plateau's own
                    // top-up already served every edge that reads it.
                    if on_plateau(field, mid) {
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
            if let Some(mid) = unit(add(points[i as usize], points[j as usize])) {
                front_spacing_m = front_spacing_m.max(field.spacing_m(mid));
            }
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
    Ok(InsertOutcome {
        inserted: inserted_total,
        batches: batches_used,
        front_spacing_m,
        plateau,
    })
}

/// G4's classification, as a pure function of three measured numbers so it
/// can be tested against readings from real runs rather than only through a
/// whole build.
///
/// TWO MECHANISMS MAKE IRREGULARITY IN THIS LADDER, and a dislocation is
/// explained if EITHER of them operates where it sits. They are separate
/// answers and neither one covers the other's cases; a single test was the
/// defect.
///
/// 1. GRADING. A hexagonal crystal cannot tile a spatially varying density:
///    wherever the field has a slope, the packing owes geometrically
///    necessary dislocations. `local_variation_m` is the field's spread over
///    the quad's own surgery-locality neighbourhood, so ANY variation there
///    -- however gentle -- means the relaxation was solving a graded problem
///    at this site and a dislocation is its expected output.
///
/// 2. INSERTION. `front_m` is the coarsest field value the ladder actually
///    inserted at. Inside a PLATEAU (a region's flat core, or a mid-ladder
///    rung) the field has no slope at all, so mechanism 1 says nothing --
///    but the level split every edge there, and the dislocations are its
///    own. Everything the ladder refined is finer than its front, so
///    `h_here <= front` is exactly "insertion reached this spacing".
///
/// What is left over is the case G4 exists for: the field is bit-for-bit
/// FLAT across the whole neighbourhood -- so the relaxation there ran the
/// identical arithmetic it runs on a uniform sphere, grading cannot have
/// produced anything -- and the spacing is coarser than anything the ladder
/// inserted at, so insertion cannot have either. A near-cocircular quad
/// there has no mechanism behind it.
fn on_band(h_here_m: f64, front_m: f64, local_variation_m: f64) -> bool {
    local_variation_m > 0.0 || h_here_m <= front_m
}

/// A dislocation G4 could not explain, with every number its refusal quotes.
struct OffBand {
    mid: V3,
    q: f64,
    h_here_m: f64,
    variation_m: f64,
    ball_cells: usize,
}

/// The field's spread over the ball of `hops` ring-steps around cells
/// `seed_a` and `seed_b`: `max - min` of the spacing every generator in it
/// reads, and the count of cells that were read.
///
/// The ball is the SURGERY LOCALITY, in the same cells the rest of this file
/// measures locality in, because that is how far a repair reaches and how
/// far a defect can have been carried from the graded field that made it.
fn local_field_variation<F: DensityField>(
    points: &[V3],
    rings: &Rings,
    field: &F,
    seed_a: usize,
    seed_b: usize,
    hops: usize,
) -> (f64, usize) {
    let mut seen = vec![seed_a, seed_b];
    let mut frontier = seen.clone();
    for _ in 0..hops {
        let mut next = Vec::new();
        for &c in &frontier {
            for &n in rings.ring(c) {
                let n = n as usize;
                if !seen.contains(&n) {
                    seen.push(n);
                    next.push(n);
                }
            }
        }
        frontier = next;
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &c in &seen {
        let h = field.spacing_m(points[c]);
        lo = lo.min(h);
        hi = hi.max(h);
    }
    (hi - lo, seen.len())
}

/// The first quad reading under `Q_DISLOCATION` that neither mechanism in
/// [`on_band`] accounts for, in canonical edge order.
fn first_off_band_dislocation<F: DensityField>(
    points: &[V3],
    rings: &Rings,
    field: &F,
    front_m: f64,
) -> Option<OffBand> {
    let mut found: Option<OffBand> = None;
    // The sub-threshold quads, found in parallel and in the same canonical
    // order the serial visitor used (THE BREAKAGE: a whole-mesh scan on one
    // thread per ladder level); the few that come back are judged in order.
    for r in surgery::quads_below(points, rings, Q_DISLOCATION) {
        if found.is_some() {
            break;
        }
        let (a, b) = (points[r.i as usize], points[r.j as usize]);
        let mid = unit(add(a, b)).unwrap_or(a);
        let h_here_m = field.spacing_m(mid);
        let (variation_m, ball_cells) = local_field_variation(
            points,
            rings,
            field,
            r.i as usize,
            r.j as usize,
            SURGERY_LOCALITY_CELLS as usize,
        );
        if !on_band(h_here_m, front_m, variation_m) {
            found = Some(OffBand {
                mid,
                q: r.q,
                h_here_m,
                variation_m,
                ball_cells,
            });
        }
    }
    found
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
    // Indexed: a polygon rung is answered from its bucket index, same bits.
    let prepared = spec.prepared().indexed();
    let clamp0 = prepared.clamped(steps[0]);
    let mut outcome = relax(&mut points, &clamp0, lloyd)?;
    progress(&format!(
        "ANNEALED\t0\t{}\t{:.4e}\t{:.4e}",
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

    // The ladder's own insertion front, in field units: the coarsest spacing
    // any level has inserted at so far. It only ever grows, because a level
    // that reached further out does not un-reach it at the next one.
    let mut front_m = 0.0f64;

    for (l, &h_l) in steps.iter().enumerate().skip(1) {
        let clamp = prepared.clamped(h_l);
        let InsertOutcome {
            inserted,
            batches,
            front_spacing_m,
            plateau,
        } = crate::mesh::profile::timed(
            &crate::mesh::profile::INSERT,
            points.len() as u64,
            || insert_level(&mut points, &clamp, beta, sizing_samples),
        )?;
        progress(&format!("INSERTED\t{l}\t{inserted}\t{batches}"));
        progress(&format!(
            "PLATEAU\t{l}\t{}\t{}\t{:.4}\t{}\t{}",
            plateau.cells,
            plateau.interior_cells,
            plateau.delivered_area_ratio,
            plateau.owed,
            plateau.inserted
        ));
        if inserted == 0 {
            // A level with no band (spec never crosses h_l) is a no-op level.
            continue;
        }
        front_m = front_m.max(front_spacing_m);
        progress(&format!("FRONT\t{l}\t{:.4}", front_m / 1000.0));

        outcome = relax(&mut points, &clamp, &level_lloyd)?;
        progress(&format!(
            "ANNEALED\t{l}\t{}\t{:.4e}\t{:.4e}",
            outcome.sweeps, outcome.mean_delta_over_h, outcome.min_dv_over_dc
        ));

        // G4 -- LOCATION FALSIFICATION, refuse-don't-repair, BEFORE surgery.
        // Every quad reading under Q_DISLOCATION must sit where the ladder
        // has actually refined; a dislocation out in crystal the ladder never
        // touched means the ladder itself is wrong, and repairing it would
        // mask that. Refusal names its coordinates and reading.
        //
        // RE-AIMED 2026-08-31, against a reproduction that made the old
        // predicate's misalignment measurable. It read
        //
        //     h_here < 0.99 * h_bg && h_here > h_l * 0.95
        //
        // and both halves were wrong by CONSTRUCTION rather than by tuning.
        //
        // The upper half asked whether the spec's spacing here is at least
        // 1% below the background, as a stand-in for "the field is graded
        // here". A tanh approaches the background ASYMPTOTICALLY, so its
        // crossing of 0.99 * h_bg sits INSIDE genuinely graded field, at a
        // radius that is a property of the outermost ramp's width and of
        // nothing else. The relaxation past that crossing is still solving a
        // graded problem -- the field there is a different number from cell
        // to cell -- while the check called it uniform crystal and refused
        // the dislocations grading owes.
        //
        // MEASURED on a 200 m Hong Kong reproduction (work/cpas-bench):
        // refusals at 2,996 km and 2,004 km from the refinement centre, in
        // two runs whose 0.99 crossings sat 1,000 km apart at 2,992 km and
        // 1,997 km. Each dislocation sat 4 to 8 km outside a cutoff that
        // moved a thousand kilometres WITH it -- a boundary tracking the
        // ramp it was cutting, not an incident. The field at the first site
        // reads 50.6963 km against a 51.2 km background: 8.3 m of 51,200
        // above the cutoff, and still sloping. A 1% stand-in cannot be
        // right, because no fraction of the background is where a tanh stops
        // being graded.
        //
        // AND THE LADDER'S OWN FRONT NOW SAYS WHERE ITS WORK REACHED, which
        // is the census the reproduction could not run. On that spec the
        // front holds at a 37.1548 km field value through levels 1 to 6 and
        // then JUMPS to 50.9427 km at level 7 -- past the 50.688 km cutoff,
        // past the 50.6963 km site that was refused, and 0.26 km short of
        // the background itself. The deep level's count-exact top-up ranks
        // sub-threshold edges by delivered-over-requested, and the largest
        // such ratios left on the sphere are in the tanh tail, where the
        // mesh still sits at background and the field asks for a hair less.
        // So at the level that refused, BOTH mechanisms reach the site: the
        // field is still sloping there, and the ladder inserted out past it.
        // The retired check was refusing work the ladder had just done.
        //
        // The lower half asked that the site not be finer than this level's
        // own band. At level l the core is clamped to h_l while the mesh
        // there is still at steps[l-1], a ratio of exactly 2 against
        // beta = sqrt(2), so the core splits at EVERY level by construction.
        // Irregularity finer than the level's band is the level's own work,
        // and refusing it was refusing the ladder for functioning.
        //
        // What replaces both is [`on_band`]: the two mechanisms that
        // actually make irregularity here, each asked directly. GRADING --
        // the field's spread over the quad's own surgery-locality
        // neighbourhood, so any slope at all means the relaxation was
        // solving a graded problem at this site. INSERTION -- `front_m`, the
        // coarsest field value the ladder actually inserted at, which is
        // what covers the flat PLATEAUS (a region's core, a mid-ladder rung)
        // where grading says nothing but every edge was split.
        //
        // THE BREAKAGE THIS STILL CATCHES, and it is the same one: a
        // near-cocircular quad out in undisturbed crystal. There the field
        // is bit-for-bit identical across the whole neighbourhood -- the
        // relaxation ran the arithmetic it runs on a uniform sphere, so
        // grading produced nothing -- and the spacing is coarser than
        // anything the ladder inserted at, so insertion produced nothing
        // either. The published quasi-uniform x1.40962 reads 0.394 in such
        // crystal; a reading under 0.10 there has no mechanism behind it,
        // and localization -- the ladder's one essential claim -- would
        // be false. Refuse, never repair: repairing it would mask a
        // systemic ladder defect.
        let rings_check = outcome.rings.clone();
        let h_bg_m = spec.background_km * 1000.0;
        // A front that has reached the background itself means the top-up
        // spilled out of the graded field and into the uniform crystal, which
        // breaks the GP-doubled interior insertion is built to preserve AND
        // leaves G4 with no crystal to falsify against. That is a defect in
        // the insertion, not in the mesh, and it is refused here rather than
        // silently disarming the check below.
        if front_m >= h_bg_m {
            return Err(MpasError::Refusal(format!(
                "level {l} inserted out at {:.4} km, the {:.4} km background spacing itself: the count-exact top-up has spilled past the graded field into the uniform crystal. Insertion there breaks the GP-doubled interior it is built to preserve, and it leaves the location check with no undisturbed crystal to falsify a dislocation against -- every reading would classify as explained. Refused: a check that cannot fail is not a check",
                front_m / 1000.0,
                h_bg_m / 1000.0
            )));
        }
        if let Some(off) = first_off_band_dislocation(&points, &rings_check, &prepared, front_m) {
            let (lat, lon) = crate::mesh::geom::lat_lon(off.mid);
            return Err(MpasError::Refusal(format!(
                "level {l} carries a dislocation in undisturbed crystal: dv/dc = {:.3e} at lat/lon ({:.3}, {:.3}) deg, where the spec asks for {:.4} km. NEITHER mechanism that makes irregularity here operates at that site. Grading did not: the field's spread over the {} cells within {SURGERY_LOCALITY_CELLS:.0} of it is {:.6} km, so the relaxation ran the same arithmetic there that it runs on a uniform sphere. Insertion did not: the ladder's front reached {:.4} km, finer than the {:.4} km asked for here, so no level ever split an edge at this spacing. Localization is the ladder's one essential claim -- irregularity is confined to where a mechanism put it -- so a dislocation with no mechanism means the ladder itself is wrong, and repairing it would mask that. Refused, never repaired",
                off.q,
                lat.to_degrees(),
                lon.to_degrees(),
                off.h_here_m / 1000.0,
                off.ball_cells,
                off.variation_m / 1000.0,
                front_m / 1000.0,
                off.h_here_m / 1000.0
            )));
        }

        // Surgery: drain the annuli's near-cocircular tail. Drift bounded at
        // 1% of this level's insertions.
        let drift_budget = (inserted / 100).max(1);
        let (rings_after, ledger) = crate::mesh::profile::timed(
            &crate::mesh::profile::SURGERY,
            points.len() as u64,
            || surgery::drain_with_retry(&mut points, &clamp, surgery_opts, drift_budget),
        )?;
        progress(&format!(
            "SURGERY\t{l}\t{}\t{}\t{:.4e}",
            ledger.rounds,
            ledger.ops.len(),
            ledger.min_q_after
        ));
        if ledger.retried_at_ship_floor.is_some() {
            progress(&format!(
                "SURGERY\t{l}\tdrained again at the {} ship floor after a stubborn site",
                surgery_opts.ship_floor
            ));
        }

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
        let hex_cells: Vec<usize> = (0..points.len())
            .filter(|&i| rings_after.degree(i) == 6)
            .collect();
        let mut ratios: Vec<f64> = hex_cells
            .par_iter()
            .map(|&i| delivered[i] / DensityField::spacing_m(&clamp, points[i]))
            .collect();
        let mut plateau_ratios: Vec<f64> = hex_cells
            .par_iter()
            .filter(|&&i| on_plateau(&clamp, points[i]))
            .map(|&i| delivered[i] / clamp.level_spacing_m)
            .collect();
        ratios.par_sort_unstable_by(|a, b| a.total_cmp(b));
        plateau_ratios.par_sort_unstable_by(|a, b| a.total_cmp(b));
        let delivered_plateau_median = (!plateau_ratios.is_empty())
            .then(|| plateau_ratios[(plateau_ratios.len() - 1) / 2]);
        progress(&format!(
            "PLATEAU_DELIVERED\t{l}\t{}\t{}",
            plateau_ratios.len(),
            delivered_plateau_median.map_or("-".to_string(), |m| format!("{m:.4}"))
        ));
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
            insert_front_spacing_km: front_m / 1000.0,
            relaxation_sweeps: outcome.sweeps,
            relaxation_mean_delta_over_h: outcome.mean_delta_over_h,
            min_dv_over_dc_after_anneal: outcome.min_dv_over_dc,
            surgery: SurgerySummary::from(&ledger),
            delivered_median: p50,
            delivered_p05: p05,
            delivered_p95: p95,
            plateau,
            delivered_plateau_median,
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

    /// The spacing whose hexagons, `(sqrt 3 / 2) h^2` each, tile the sphere
    /// with exactly `cells` cells -- the sizing integral's own convention.
    fn area_equivalent_spacing_m(cells: usize) -> f64 {
        let sphere = 4.0 * std::f64::consts::PI * EARTH_RADIUS_M * EARTH_RADIUS_M;
        (sphere / (cells as f64 * 3f64.sqrt() / 2.0)).sqrt()
    }

    /// THE DEFECT: a doubled plateau keeps its parent's count, so a plateau
    /// that is short stays short, level after level (box J, 2026-10-24: a
    /// 1 km core delivered at 1.029 km; at 300 km the level gate refused it).
    /// A plateau owed cells must get them, inside itself, and come out at its
    /// own count.
    #[test]
    fn a_short_plateau_is_topped_up_to_its_own_count() {
        let (m, n) = (4u32, 0u32);
        let doubled = crate::mesh::icosa::goldberg_cells(2 * m, 2 * n);
        // 3% finer than GP(2m,2n) holds: the doubled crystal is ~6% short.
        let h = 0.97 * area_equivalent_spacing_m(doubled);
        let spec = MeshSpec::uniform(h / 1000.0);
        let prepared = spec.prepared();
        let clamp = prepared.clamped(h);
        let mut pts = crate::mesh::icosa::seed(m, n).unwrap();
        let out = insert_level(&mut pts, &clamp, DEFAULT_BETA, 50_000).unwrap();
        let p = out.plateau;
        assert_eq!(p.cells, doubled, "the plateau pass did not double the crystal: {p:?}");
        assert!(p.owed > 0, "a 6%-short plateau read no deficit: {p:?}");
        assert_eq!(p.inserted as i64, p.owed, "the plateau was not given what it is owed: {p:?}");
        let target = crate::mesh::density::predicted_cells_of(&clamp, 50_000);
        let off = pts.len() as f64 / target - 1.0;
        assert!(
            off.abs() < 0.01,
            "{} cells against the {target:.0} the level owes ({:+.2}%) -- the plateau is still off its own count",
            pts.len(),
            100.0 * off
        );
        // Deterministic: the same level twice is bit-identical.
        let mut again = crate::mesh::icosa::seed(m, n).unwrap();
        insert_level(&mut again, &clamp, DEFAULT_BETA, 50_000).unwrap();
        assert_eq!(pts, again);
    }

    #[test]
    fn full_midpoint_insertion_on_a_goldberg_patch_is_exact_gp_doubling() {
        // A UNIFORM interior under a level whose spacing is half the current
        // one: every edge splits at its exact midpoint, and the point set is
        // exactly GP(2m, 2n)'s -- same crystal, no new grain boundaries.
        let m = 4u32;
        let n = 0u32;
        let mut pts = crate::mesh::icosa::seed(m, n).unwrap();
        // The level request: the spacing whose hexagons tile the sphere with
        // exactly GP(2m,2n)'s count, so the level is owed exactly the doubled
        // count and the doubled plateau holds it (its cells' areas sum to the
        // sphere): the top-up owes nothing and the doubling is the whole
        // insertion. (A plateau that is OWED cells is the test above.)
        let half_m = area_equivalent_spacing_m(crate::mesh::icosa::goldberg_cells(2 * m, 2 * n));
        let uniform_half = MeshSpec::uniform(half_m / 1000.0);
        let prepared_half = uniform_half.prepared();
        let clamp = prepared_half.clamped(half_m);
        let InsertOutcome {
            inserted,
            batches,
            plateau,
            ..
        } = insert_level(&mut pts, &clamp, DEFAULT_BETA, 50_000).unwrap();
        assert!(plateau.owed <= 0, "the plateau was owed {} cells: {plateau:?}", plateau.owed);
        assert_eq!(plateau.inserted, 0);
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

    /// The outermost rung of the 200 m Hong Kong reproduction
    /// (`work/cpas-bench/cpas-hk200m.json`), as its own spec: a 25.6 km rung
    /// capped at 2,030 km under a 419.8 km ramp on a 51.2 km background.
    /// Nothing inside 2,000 km matters to the tail this reproduces, so the
    /// inner seven rungs are left off.
    fn cpas_outer_rung() -> MeshSpec {
        cpas_rung(2030.0, 25.6, 419.8)
    }

    /// Either reproduction's outermost rung. Run 1 kept the inferred 25.6 km
    /// rung (cap 2,030 km, ramp 419.8 km) and refused at 2,996 km; run 2
    /// removed it, leaving 12.8 km at cap 1,400 km under a 209.9 km ramp,
    /// and refused at 2,004 km.
    fn cpas_rung(radius_km: f64, spacing_km: f64, transition_km: f64) -> MeshSpec {
        MeshSpec {
            background_km: 51.2,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [90.0, 0.0],
                    radius_km,
                },
                spacing_km,
                transition: TransitionField::Km(transition_km),
            }],
            name: None,
        }
    }

    /// A point `r_km` from the north pole along the prime meridian.
    fn out_from_pole(r_km: f64) -> V3 {
        let theta = r_km * 1000.0 / EARTH_RADIUS_M;
        [theta.sin(), 0.0, theta.cos()]
    }

    /// THE DEFECT THIS FILE'S G4 CARRIED, measured on the geometry that
    /// exposed it, and the re-aimed check's verdict on the same three sites.
    ///
    /// `work/cpas-bench` refused two 200 m Hong Kong builds at 2,996 km and
    /// 2,004 km from the refinement centre, each 4 to 8 km outside a
    /// `0.99 * background` crossing that itself moved 1,000 km between the
    /// two runs when the outermost rung changed. This test re-derives the
    /// field readings from the spec rather than quoting that write-up, and
    /// pins all three verdicts.
    #[test]
    fn the_retired_location_check_called_a_live_tanh_tail_uniform_crystal() {
        // BOTH reproductions, each at the radius it actually refused at.
        for (radius_km, rung_km, ramp_km, r_refused) in [
            (2030.0, 25.6, 419.8, 2996.0),
            (1400.0, 12.8, 209.9, 2004.0),
        ] {
            let spec = cpas_rung(radius_km, rung_km, ramp_km);
            let prepared = spec.prepared();
            let h_bg = spec.background_km * 1000.0;
            let retired_cutoff = 0.99 * h_bg;

            // The refusal site: still sloping, but above 99% of the background.
            let h_here = prepared.spacing_m(out_from_pole(r_refused));
            assert!(
                h_here > retired_cutoff,
                "the retired predicate would not have refused {h_here:.1} m against {retired_cutoff:.1} m at {r_refused} km -- the reproduction does not reproduce"
            );
            assert!(
                h_here < h_bg,
                "{h_here:.4} m is not below the {h_bg:.1} m background, so this site is not in the tail at all"
            );
            // ... and the field is genuinely varying there: one background
            // cell either side reads a different number, so the relaxation
            // was solving a graded problem at this site.
            let step = prepared.spacing_m(out_from_pole(r_refused + 51.2))
                - prepared.spacing_m(out_from_pole(r_refused - 51.2));
            assert!(
                step > 0.0,
                "the tail is flat at {r_refused} km, which would make this a genuine dislocation site"
            );
            // GRADING ALONE MUST CARRY THESE SITES. The reproduction's own
            // front holds at a 37.1548 km field value through levels 1 to 6
            // before the deep level's top-up pushes it out to 50.9427 km, so
            // the levels-1-to-6 value is used here deliberately: it puts both
            // sites OUTSIDE the front and leaves grading as the only
            // mechanism available, which is the half the retired check could
            // not credit.
            let front_before_the_deep_level = 37_154.8;
            assert!(
                h_here > front_before_the_deep_level,
                "this site sits inside the front, so it does not isolate grading"
            );
            assert!(
                on_band(h_here, front_before_the_deep_level, step),
                "the re-aimed check still refuses the reproduction's site at {r_refused} km"
            );
            assert!(
                !on_band(h_here, front_before_the_deep_level, 0.0),
                "with no variation and no front reaching it, this site must still be refused"
            );
        }

        let spec = cpas_outer_rung();
        let prepared = spec.prepared();
        let h_bg = spec.background_km * 1000.0;

        // THE CHECK STILL HAS TEETH ON THE SAME SPEC. Beyond
        // SHELL_SATURATION_WIDTHS = 19.07 ramp widths the tanh saturates and
        // the field is BIT IDENTICAL to the background -- density.rs's own
        // definition of outside. 2030 + 19.07 * 419.8 = 10,036 km.
        let r_flat = 12_000.0;
        let h_flat = prepared.spacing_m(out_from_pole(r_flat));
        assert_eq!(
            h_flat, h_bg,
            "the field at {r_flat} km is not bit-identical to the background, so this site is still inside the ramp"
        );
        let flat_step = prepared.spacing_m(out_from_pole(r_flat + 51.2))
            - prepared.spacing_m(out_from_pole(r_flat - 51.2));
        assert_eq!(flat_step, 0.0, "the far field is not flat to the last bit");
        // Every front the ladder can have is finer than the background, so
        // neither mechanism reaches here and the dislocation is refused.
        assert!(
            !on_band(h_flat, h_bg - 1.0, flat_step),
            "a dislocation in saturated crystal is no longer refused -- the re-aiming loosened the check"
        );
    }

    /// A PLATEAU is not uniform crystal, and the insertion front is what
    /// says so: inside a region's flat core the field has no slope at all,
    /// but every level split every edge there.
    #[test]
    fn a_flat_plateau_inside_the_refinement_is_explained_by_the_front() {
        // A cap wide enough to hold one: the tanh saturates 19.07 widths
        // inside the boundary, so 3,000 km of cap under a 100 km ramp leaves
        // a flat core out to 1,093 km.
        let spec = MeshSpec {
            background_km: 51.2,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [90.0, 0.0],
                    radius_km: 3000.0,
                },
                spacing_km: 25.6,
                transition: TransitionField::Km(100.0),
            }],
            name: None,
        };
        let prepared = spec.prepared();
        // Deep inside the cap: the field reads the rung's own spacing and is
        // flat to the last bit.
        let h_core = prepared.spacing_m(out_from_pole(10.0));
        let core_step =
            prepared.spacing_m(out_from_pole(20.0)) - prepared.spacing_m(out_from_pole(10.0));
        assert_eq!(core_step, 0.0, "the core is not a plateau");
        assert_eq!(h_core, 25_600.0, "the plateau does not read the rung");
        assert!(h_core < spec.background_km * 1000.0);
        // The retired check refused exactly this: `h_here > h_l * 0.95` reads
        // the core as "finer than this level's band" at every level above the
        // last one. The front explains it instead.
        assert!(
            on_band(h_core, 45_000.0, core_step),
            "a plateau inside the ladder's own front is not explained"
        );
        assert!(
            !on_band(h_core, h_core * 0.5, core_step),
            "a plateau the ladder never refined down to must still be refused"
        );
    }

    /// A relaxed uniform crystal, and a spec whose refinement is a small cap
    /// at the north pole under a ramp that SATURATES: beyond
    /// `300 + 19.07 * 200 = 4,114 km` the field is bit-identical to the
    /// background, so the southern hemisphere is undisturbed crystal by
    /// density.rs's own definition of the word.
    fn crystal_and_spec() -> (Vec<V3>, MeshSpec) {
        let spec = MeshSpec {
            background_km: 120.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [90.0, 0.0],
                    radius_km: 300.0,
                },
                spacing_km: 60.0,
                transition: TransitionField::Km(200.0),
            }],
            name: None,
        };
        let mut points = crate::mesh::icosa::seed(16, 0).unwrap();
        let uniform = MeshSpec::uniform(120.0);
        let prepared = uniform.prepared();
        let clamp = prepared.clamped(120_000.0);
        relax(&mut points, &clamp, &LloydOptions::default()).unwrap();
        (points, spec)
    }

    /// Push cell `b` of the quad nearest `target` towards the circumcircle
    /// through `(i, a, j)` until that edge's `dv/dc` lands under
    /// `Q_DISLOCATION` -- four near-cocircular sites, which IS the defect
    /// surgery exists to drain. Returns the site and the reading it reached.
    fn inject_cocircular_quad(points: &mut [V3], target: V3) -> (V3, f64) {
        use crate::mesh::geom::circumcenter;
        let rings = delaunay_rings(points).unwrap();
        let mut best: Option<(f64, u32, u32, u32, u32)> = None;
        surgery::for_each_quad(points, &rings, |r| {
            let mid = unit(add(points[r.i as usize], points[r.j as usize])).unwrap();
            let d = arc(mid, target);
            if best.is_none_or(|(bd, ..)| d < bd) {
                best = Some((d, r.i, r.j, r.a, r.b));
            }
        });
        let (_, i, j, a, b) = best.expect("a quad near the target");
        let (pi, pj, pa) = (
            points[i as usize],
            points[j as usize],
            points[a as usize],
        );
        let origin = points[b as usize];
        let centre = circumcenter(pi, pa, pj).expect("circumcircle");
        let radius = arc(centre, pi);
        let out = arc(centre, origin);
        let slerp = |t: f64| -> V3 {
            let omega = arc(centre, origin);
            let so = omega.sin();
            unit(add(
                crate::mesh::geom::scale(centre, ((1.0 - t) * omega).sin() / so),
                crate::mesh::geom::scale(origin, (t * omega).sin() / so),
            ))
            .unwrap()
        };
        // t = 1 leaves b where it was; t = radius/out puts it exactly on the
        // circumcircle, where the quad is perfectly cocircular. Bisect
        // between them for a reading just inside the flag.
        let (mut lo, mut hi) = (radius / out, 1.0);
        let mut reached = f64::NAN;
        for _ in 0..40 {
            let t = 0.5 * (lo + hi);
            points[b as usize] = slerp(t);
            let r2 = delaunay_rings(points).unwrap();
            let mut q = f64::NAN;
            surgery::for_each_quad(points, &r2, |r| {
                if r.i == i && r.j == j {
                    q = r.q;
                }
            });
            if !q.is_finite() {
                // The edge flipped away: back off towards the original.
                lo = t;
                continue;
            }
            reached = q;
            if q < 0.04 {
                lo = t;
            } else if q > 0.08 {
                hi = t;
            } else {
                break;
            }
        }
        let mid = unit(add(points[i as usize], points[j as usize])).unwrap();
        (mid, reached)
    }

    /// THE BREAKAGE G4 EXISTS FOR, still caught after the re-aiming: a
    /// near-cocircular quad in crystal no mechanism disturbed.
    ///
    /// The same injected defect, at two places on one sphere, gets opposite
    /// verdicts -- which is what makes this a re-aiming and not a loosening.
    #[test]
    fn an_injected_dislocation_is_refused_in_flat_crystal_and_allowed_in_the_ramp() {
        // The ladder's finest rung as the front: fine enough that insertion
        // explains nothing at either site, so GRADING is the only mechanism
        // in play and the two verdicts differ on it alone.
        let front_m = 60_000.0;

        // ---- the clean crystal answers nothing ---------------------------
        let (points, spec) = crystal_and_spec();
        let prepared = spec.prepared();
        let rings = delaunay_rings(&points).unwrap();
        assert!(
            first_off_band_dislocation(&points, &rings, &prepared, front_m).is_none(),
            "the relaxed crystal already carries an unexplained dislocation"
        );

        // ---- inject one in the saturated far field: REFUSED --------------
        let (mut flat_points, _) = crystal_and_spec();
        let (site, q) = inject_cocircular_quad(&mut flat_points, [0.0, 0.0, -1.0]);
        assert!(
            q < Q_DISLOCATION,
            "the injection did not reach the flag: dv/dc = {q:.4e}"
        );
        let flat_rings = delaunay_rings(&flat_points).unwrap();
        let caught = first_off_band_dislocation(&flat_points, &flat_rings, &prepared, front_m)
            .expect("a dislocation in undisturbed crystal was NOT refused");
        assert_eq!(
            caught.variation_m, 0.0,
            "the site G4 refused is not bit-flat, so it was not the injected one"
        );
        assert!(
            arc(caught.mid, site) < 4.0 * 120_000.0 / EARTH_RADIUS_M,
            "G4 refused somewhere other than the injection site"
        );
        assert_eq!(caught.h_here_m, spec.background_km * 1000.0);

        // ---- the same injection inside the live ramp: ALLOWED ------------
        // 400 km from the pole is 100 km outside the 300 km cap, one ramp
        // half-width into a transition that is 200 km wide -- the steepest
        // part of the field this spec has.
        let (mut ramp_points, _) = crystal_and_spec();
        let ramp_target = out_from_pole(400.0);
        let (_, q_ramp) = inject_cocircular_quad(&mut ramp_points, ramp_target);
        assert!(
            q_ramp < Q_DISLOCATION,
            "the ramp injection did not reach the flag: dv/dc = {q_ramp:.4e}"
        );
        let ramp_rings = delaunay_rings(&ramp_points).unwrap();
        assert!(
            first_off_band_dislocation(&ramp_points, &ramp_rings, &prepared, front_m).is_none(),
            "an identical dislocation inside a live ramp is refused -- grading is not being credited"
        );
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
        // The smallest accurate end-to-end: one doubling with a real band.
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
            "graded 240->480: GP({},{}) level 0, {} cells final, {} level reports, final min dv/dc {:.4e}, front {:.4} km vs 0.99*bg {:.4} km",
            choice.m,
            choice.n,
            points.len(),
            reports.len(),
            outcome.min_dv_over_dc,
            reports[0].insert_front_spacing_km,
            0.99 * spec.background_km,
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
