//! Spherical Delaunay triangulation, as a three-dimensional convex hull.
//!
//! For points ON a sphere the convex hull IS the Delaunay triangulation: a
//! triangle has an empty spherical circumcircle exactly when every other point
//! lies on one side of its plane, which is the hull's own visibility test.
//!
//! THE DEGENERACY THAT KILLS A NAIVE IMPLEMENTATION, stated here because it is
//! the single most likely way to build a silently-broken generator: all the
//! points are COSPHERICAL, so the classical lifted-paraboloid `insphere(a,b,c,d)`
//! predicate returns exactly 0 for every quadruple. Reaching for a
//! lifted-Delaunay library hands back a zero-information answer on every test.
//! The correct in-circle test here is `orient3d` on a consistently oriented
//! facet -- see [`crate::mesh::geom::orient3d`], which is evaluated in
//! double-double arithmetic because the naive f64 determinant has only four
//! decades of margin at 3 km spacing, and 3 km is inside the range this door
//! will be asked for.
//!
//! The algorithm is randomized-incremental with conflict lists (Clarkson-Shor),
//! expected O(n log n). The insertion order comes from a counter-based PRNG
//! seeded from the point count, so a mesh is reproducible between runs and
//! between machines.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::mesh::derive::Rings;
use crate::mesh::geom::{V3, add, orient3d, orient3d_sign, orient3d_sos, scale, unit};

/// One hull facet. `v` winds counter-clockwise seen from OUTSIDE the sphere;
/// `adj[i]` is the facet across the directed edge `(v[i], v[(i+1)%3])`.
#[derive(Debug, Clone)]
struct Face {
    v: [u32; 3],
    adj: [u32; 3],
    alive: bool,
    conflicts: Vec<u32>,
}

const NONE: u32 = u32::MAX;

/// SplitMix64. A dependency-free, counter-based generator: no global state, the
/// same seed gives the same permutation on every machine, and the offline
/// vendor tree does not have to grow a crate for it.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Build the neighbour rings of the spherical Voronoi tessellation of `points`.
///
/// The returned rings wind counter-clockwise seen from outside the sphere,
/// which is the winding every sign convention in the TRiSK operators is read
/// off. The starting slot of each ring is not meaningful and is not something
/// MPAS fixes.
pub fn delaunay_rings(points: &[V3]) -> MpasResult<Rings> {
    crate::mesh::profile::timed(
        &crate::mesh::profile::HULL,
        points.len() as u64,
        || delaunay_rings_inner(points),
    )
}

fn delaunay_rings_inner(points: &[V3]) -> MpasResult<Rings> {
    let faces = build_hull(points)?;
    let live: Vec<[u32; 3]> = faces.iter().filter(|f| f.alive).map(|f| f.v).collect();
    crate::mesh::profile::timed(
        &crate::mesh::profile::HULL_RINGS,
        points.len() as u64,
        || rings_from_faces(points.len(), &live),
    )
}

/// The randomized-incremental construction itself: every facet ever created,
/// live and retired, with `alive` marking the finished hull.
fn build_hull(points: &[V3]) -> MpasResult<Vec<Face>> {
    let n = points.len();
    if n < 4 {
        return Err(MpasError::Refusal(format!(
            "{n} generators cannot triangulate a sphere; four is the minimum that encloses a volume, and below it there is no Delaunay triangle to take a Voronoi vertex from"
        )));
    }
    if let Some((i, j)) = crate::mesh::profile::timed(
        &crate::mesh::profile::HULL_DUP,
        n as u64,
        || exact_duplicate(points),
    ) {
        return Err(MpasError::Refusal(format!(
            "generators {i} and {j} are the same point to the last bit; two coincident generators have no Voronoi boundary between them, so the hull would build a zero-area facet and dcEdge would be 0 -- which the tangential weight formula divides by"
        )));
    }

    let mut faces = crate::mesh::profile::timed(
        &crate::mesh::profile::HULL_TET,
        n as u64,
        || initial_tetrahedron(points),
    )?;
    // Every point not in the seed tetrahedron starts conflicting with some
    // facet: it lies on the sphere and the seed hull is strictly inside it.
    let seed: Vec<u32> = faces[0].v.iter().chain(faces[1].v.iter()).copied().collect();
    let mut in_seed = vec![false; n];
    for &s in &seed {
        in_seed[s as usize] = true;
    }
    let mut order: Vec<u32> = (0..n as u32).filter(|&i| !in_seed[i as usize]).collect();
    let mut rng = SplitMix(0x5EED_0000_0000_0000 ^ n as u64);
    for i in (1..order.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }

    // Free counters: plain locals, flushed once at the end of the build.
    let mut c_flood = 0u64;
    let mut c_orphan = 0u64;
    let mut c_seed = 0u64;
    let mut c_orphans = 0u64;
    let mut c_horizon = 0u64;

    let mut point_face = vec![NONE; n];
    for &p in &order {
        let mut placed = false;
        for (fi, f) in faces.iter_mut().enumerate() {
            c_seed += 1;
            if visible(points, f, p) {
                f.conflicts.push(p);
                point_face[p as usize] = fi as u32;
                placed = true;
                break;
            }
        }
        if !placed {
            return Err(MpasError::Refusal(format!(
                "generator {p} lies inside the hull of the four seed generators; it is not on the sphere the other points are on, and a mesh built from a mixture of radii is not a spherical Voronoi tessellation"
            )));
        }
    }

    // Scratch reused across insertions so the sweep does not allocate per point.
    let mut stamp = vec![0u32; 8];
    let mut generation = 0u32;
    let mut stack: Vec<u32> = Vec::new();
    let mut visible_faces: Vec<u32> = Vec::new();
    let mut horizon: Vec<(u32, u32, u32, usize)> = Vec::new();
    // THE HORIZON'S POSITIONS, GATHERED ONCE PER INSERTION. Every facet the
    // fan below creates is `[u_k, w_k, p]`, so the orphan re-home -- 83.8% of
    // the hull's visibility tests, measured -- re-gathers the same handful of
    // `V3` out of a point array that is 40 MB at the 900 m parent's size. Six
    // horizon edges is a mean of 288 bytes: it stays in L1 for the whole
    // orphan sweep, where the point array does not.
    let mut horizon_xyz: Vec<(V3, V3)> = Vec::new();
    let mut orphans: Vec<u32> = Vec::new();
    // A DOZEN ENTRIES, NOT A HASH TABLE. The fan built around one inserted
    // generator has as many facets as the horizon has edges -- MEASURED, a
    // mean of 6.0 on the graded meshes this generator makes -- and this map
    // was a default-hasher `HashMap` cleared and refilled for every one of
    // the 24.2 million insertions a 54,247-cell mesh performs: 290 million
    // SipHash inserts and 290 million SipHash lookups, 3.2% of the run's
    // cycles, to index twelve rows. A linear scan of a reused vector answers
    // the same question, and it scans BACKWARDS so that if a directed edge
    // ever did repeat -- the non-simple horizon the refusal below names --
    // the last write would win exactly as the map's did.
    let mut edge_ring: Vec<((u32, u32), (u32, usize))> = Vec::new();

    let _insert_timer = crate::mesh::profile::Span::new(&crate::mesh::profile::HULL_INSERT, n as u64);
    for &p in &order {
        // The inserted generator's own position: fixed for this whole
        // insertion, and one of the four operands of every visibility test
        // it performs.
        let pp = points[p as usize];
        let start = point_face[p as usize];
        if start == NONE || !faces[start as usize].alive {
            return Err(MpasError::Refusal(format!(
                "generator {p} lost its conflicting facet during construction; the hull's conflict graph is inconsistent and the triangulation it produced cannot be trusted"
            )));
        }
        generation += 1;
        if stamp.len() < faces.len() {
            stamp.resize(faces.len() * 2, 0);
        }

        // --- the visible region, which is connected on a convex hull
        visible_faces.clear();
        stack.clear();
        stack.push(start);
        stamp[start as usize] = generation;
        while let Some(fi) = stack.pop() {
            visible_faces.push(fi);
            let adj = faces[fi as usize].adj;
            for &a in &adj {
                if a != NONE && stamp[a as usize] != generation {
                    c_flood += 1;
                    let fv = faces[a as usize].v;
                    if visible_at(
                        points[fv[0] as usize],
                        points[fv[1] as usize],
                        points[fv[2] as usize],
                        pp,
                        fv[0],
                        fv[1],
                        fv[2],
                        p,
                    ) {
                        stamp[a as usize] = generation;
                        stack.push(a);
                    }
                }
            }
        }

        // --- the horizon: edges of visible facets whose other side is not visible
        horizon.clear();
        for &fi in &visible_faces {
            // The two `[u32; 3]` arrays, copied out to release the borrow --
            // cloning the whole `Face` here also cloned its conflict list,
            // which early in the construction is tens of thousands of indices.
            let (fv, fadj) = {
                let f = &faces[fi as usize];
                (f.v, f.adj)
            };
            for s in 0..3 {
                let a = fadj[s];
                if a == NONE || stamp[a as usize] != generation {
                    let back = (0..3)
                        .find(|&t| faces[a as usize].adj[t] == fi)
                        .ok_or_else(|| {
                            MpasError::Refusal(
                                "the hull's facet adjacency is not mutual; the triangulation is not a closed surface".to_string(),
                            )
                        })?;
                    horizon.push((fv[s], fv[(s + 1) % 3], a, back));
                }
            }
        }
        if horizon.len() < 3 {
            return Err(MpasError::Refusal(format!(
                "inserting generator {p} left a horizon of {} edges; a point outside a convex hull always sees a closed ring of at least three, so the orientation predicate has returned an inconsistent answer for a near-degenerate quadruple",
                horizon.len()
            )));
        }

        // --- retire the visible facets, keeping their unprocessed points
        c_horizon += horizon.len() as u64;
        orphans.clear();
        for &fi in &visible_faces {
            faces[fi as usize].alive = false;
            orphans.append(&mut faces[fi as usize].conflicts);
        }

        // --- one new facet per horizon edge. The edge is taken in the
        //     direction the RETIRED facet carried it, which is what leaves the
        //     rebuilt surface consistently wound.
        let first_new = faces.len() as u32;
        horizon_xyz.clear();
        for &(u, w, _, _) in &horizon {
            horizon_xyz.push((points[u as usize], points[w as usize]));
        }
        for &(u, w, outside, back) in &horizon {
            let idx = faces.len() as u32;
            faces.push(Face {
                v: [u, w, p],
                adj: [outside, NONE, NONE],
                alive: true,
                conflicts: Vec::new(),
            });
            faces[outside as usize].adj[back] = idx;
        }
        edge_ring.clear();
        for idx in first_new..faces.len() as u32 {
            let v = faces[idx as usize].v;
            edge_ring.push(((v[1], v[2]), (idx, 1)));
            edge_ring.push(((v[2], v[0]), (idx, 2)));
        }
        for idx in first_new..faces.len() as u32 {
            let v = faces[idx as usize].v;
            for (slot, (a, b)) in [(1usize, (v[1], v[2])), (2usize, (v[2], v[0]))] {
                if let Some(&(_, (other, _))) =
                    edge_ring.iter().rev().find(|(key, _)| *key == (b, a))
                {
                    faces[idx as usize].adj[slot] = other;
                } else {
                    return Err(MpasError::Refusal(format!(
                        "the facets created around generator {p} do not close into a fan; the horizon was not a single ring, which means the orientation predicate answered inconsistently on a near-degenerate configuration"
                    )));
                }
            }
        }

        // --- redistribute the orphaned points onto the new facets
        c_orphans += orphans.len() as u64;
        for q in orphans.drain(..) {
            if q == p {
                continue;
            }
            // The orphan's own position, gathered ONCE rather than once per
            // facet tested: the mean orphan is tested against 2.62 facets.
            let qq = points[q as usize];
            let mut home = NONE;
            for (k, &(u, w, _, _)) in horizon.iter().enumerate() {
                c_orphan += 1;
                let (uu, ww) = horizon_xyz[k];
                // Facet `first_new + k` is `[u, w, p]` by construction above,
                // so these are the same four operands, in the same order,
                // that `visible(points, &faces[first_new + k], q)` formed.
                if visible_at(uu, ww, pp, qq, u, w, p, q) {
                    home = first_new + k as u32;
                    break;
                }
            }
            if home == NONE {
                return Err(MpasError::Refusal(format!(
                    "generator {q} sees no facet after generator {p} was inserted; on a sphere every uninserted point is outside the partial hull, so this is a hull that has stopped being convex"
                )));
            }
            faces[home as usize].conflicts.push(q);
            point_face[q as usize] = home;
        }
    }

    if crate::mesh::profile::on() {
        crate::mesh::profile::VIS_FLOOD.add(0, c_flood);
        crate::mesh::profile::VIS_ORPHAN.add(0, c_orphan);
        crate::mesh::profile::VIS_SEED.add(0, c_seed);
        crate::mesh::profile::FACES.add(0, faces.len() as u64);
        crate::mesh::profile::ORPHANS.add(0, c_orphans);
        let _ = c_horizon;
    }
    let alive = faces.iter().filter(|f| f.alive).count();
    let expected = 2 * n - 4;
    if alive != expected {
        return Err(MpasError::Refusal(format!(
            "the hull closed with {alive} facets; a Delaunay triangulation of {n} points on a sphere has exactly 2n-4 = {expected}. A different count means at least one generator was swallowed or duplicated, and the mesh would carry a cell with no dual triangle"
        )));
    }
    drop(_insert_timer);
    Ok(faces)
}

/// Two generators identical to the last bit. Sorted bit patterns, so this is
/// `O(n log n)` and exact -- a hash on floats would miss `-0.0` against `0.0`.
fn exact_duplicate(points: &[V3]) -> Option<(usize, usize)> {
    let mut keyed: Vec<([u64; 3], usize)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                [
                    (p[0] + 0.0).to_bits(),
                    (p[1] + 0.0).to_bits(),
                    (p[2] + 0.0).to_bits(),
                ],
                i,
            )
        })
        .collect();
    keyed.sort_unstable();
    for w in keyed.windows(2) {
        if w[0].0 == w[1].0 {
            return Some((w[0].1.min(w[1].1), w[0].1.max(w[1].1)));
        }
    }
    None
}

/// Is `p` outside the plane of this facet?
///
/// Ties -- an exactly coplanar quadruple -- are broken by index so the answer
/// is at least deterministic and antisymmetric. Genuine exact ties need exactly
/// symmetric input, which the golden-ratio spiral this crate seeds from does not
/// produce; if one ever does slip through and the tie-break is not coherent, the
/// hull fails to close and refuses by name rather than emitting a mesh.
#[inline]
fn visible(points: &[V3], f: &Face, p: u32) -> bool {
    visible_at(
        points[f.v[0] as usize],
        points[f.v[1] as usize],
        points[f.v[2] as usize],
        points[p as usize],
        f.v[0],
        f.v[1],
        f.v[2],
        p,
    )
}

/// [`visible`] with the four generator POSITIONS already in hand.
///
/// THE BREAKAGE THIS EXISTS FOR, and it is a cache one, not an arithmetic one.
/// `visible` gathers four `V3` out of the whole point array. At 1.69 M cells
/// that array is 40 MB, the profile lane measured per-build cost growing as
/// `N^1.48` against an expected `N log N`, and named the point array leaving
/// L2 above ~90 k cells as the reason. The orphan re-home is 83.8% of every
/// visibility test the hull performs (1.459 of 1.741 billion on a 54 k mesh)
/// and it tests ONE query point against the fan of new facets around ONE
/// inserted generator -- so three of its four gathers are loop-invariant and
/// the fourth ranges over at most a dozen horizon vertices. Handing the
/// positions in lets the caller hoist them into a scratch that fits in L1.
///
/// The arithmetic is byte-for-byte the arithmetic `visible` performed: the
/// same four operands in the same order into the same predicate. This is a
/// LOAD change, not a numerics change, and the registered-digest proof is
/// what says so.
#[inline]
fn visible_at(a: V3, b: V3, c: V3, d: V3, ia: u32, ib: u32, ic: u32, p: u32) -> bool {
    // SIGN, not value: `orient3d_sign` answers from a plain f64 determinant
    // whenever that determinant is provably past its own rounding error, and
    // falls back to the double-double `orient3d` otherwise. Every branch below
    // is a comparison against zero, so the two agree here by construction --
    // `geom::the_filtered_sign_never_disagrees_with_the_exact_predicate` is
    // the gate. Nothing else in this file may take the filtered value.
    let det = orient3d_sign(a, b, c, d);
    if det != 0.0 {
        det > 0.0
    } else {
        orient3d_sos(a, b, c, d, ia, ib, ic, p) > 0.0
    }
}

/// Four generators that enclose a volume, chosen to be as far from degenerate
/// as the input allows: the farthest pair, then the point farthest off that
/// line, then the point farthest off that plane.
fn initial_tetrahedron(points: &[V3]) -> MpasResult<Vec<Face>> {
    let n = points.len();
    let a = 0usize;
    // ONE EVALUATION PER CANDIDATE. `max_by` calls its comparator n-1 times
    // and each call evaluated BOTH sides' key, so every candidate's distance
    // was computed twice -- and for `d` that key is the double-double
    // `orient3d`, so a 54,247-cell mesh paid 24.2 million exact predicates it
    // did not need. The fold below keeps `max_by`'s tie rule exactly: on an
    // equal key the LATER index wins, which is what `Ordering::Equal` gives
    // `max_by`, so the seed tetrahedron -- and with it every facet index and
    // every emitted ring rotation -- is the same one.
    //
    // GENERIC, not `&dyn Fn`: a trait object here would put an indirect call
    // in front of every one of the n evaluations, which for the first two
    // passes costs more than the arithmetic it is calling.
    fn pick<F: Fn(usize) -> f64>(n: usize, key: F) -> usize {
        let mut best = 0usize;
        let mut best_key = key(0);
        for i in 1..n {
            let k = key(i);
            if k >= best_key {
                best = i;
                best_key = k;
            }
        }
        best
    }
    let b = pick(n, |i| crate::mesh::geom::chord(points[a], points[i]));
    let ab = crate::mesh::geom::sub(points[b], points[a]);
    let c = pick(n, |i| {
        crate::mesh::geom::norm(crate::mesh::geom::cross(
            ab,
            crate::mesh::geom::sub(points[i], points[a]),
        ))
    });
    let d = pick(n, |i| orient3d(points[a], points[b], points[c], points[i]).abs());
    if orient3d(points[a], points[b], points[c], points[d]) == 0.0 {
        return Err(MpasError::Refusal(
            "every generator lies on one plane; a flat set of points has no spherical Voronoi tessellation and no closed-sphere mesh can be built from it".to_string(),
        ));
    }

    // Orient every facet so its winding is counter-clockwise seen from outside,
    // judged against the tetrahedron's own centroid.
    let centroid = scale(
        add(add(points[a], points[b]), add(points[c], points[d])),
        0.25,
    );
    let quad = [a as u32, b as u32, c as u32, d as u32];
    let mut faces: Vec<Face> = Vec::with_capacity(4);
    for skip in 0..4 {
        let mut v: [u32; 3] = [0; 3];
        let mut k = 0;
        for t in 0..4 {
            if t != skip {
                v[k] = quad[t];
                k += 1;
            }
        }
        if orient3d(
            points[v[0] as usize],
            points[v[1] as usize],
            points[v[2] as usize],
            centroid,
        ) > 0.0
        {
            v.swap(1, 2);
        }
        faces.push(Face {
            v,
            adj: [NONE; 3],
            alive: true,
            conflicts: Vec::new(),
        });
    }
    let mut edge_map: HashMap<(u32, u32), (u32, usize)> = HashMap::new();
    for (fi, f) in faces.iter().enumerate() {
        for s in 0..3 {
            edge_map.insert((f.v[s], f.v[(s + 1) % 3]), (fi as u32, s));
        }
    }
    for fi in 0..4 {
        for s in 0..3 {
            let (x, y) = (faces[fi].v[s], faces[fi].v[(s + 1) % 3]);
            let (other, _) = *edge_map.get(&(y, x)).ok_or_else(|| {
                MpasError::Refusal(
                    "the seed tetrahedron's facets are not consistently wound; the hull would grow from a surface that is already inside out".to_string(),
                )
            })?;
            faces[fi].adj[s] = other;
        }
    }
    Ok(faces)
}

/// Turn the finished facet set into the counter-clockwise neighbour rings.
///
/// Around a cell, each facet `(a, b, c)` contributes one step of the ring: at
/// `a` the step is `b -> c`, at `b` it is `c -> a`, at `c` it is `a -> b`.
/// Following those steps from any start walks the neighbours counter-clockwise
/// seen from outside, which is the winding MPAS reads its sign conventions off.
fn rings_from_faces(n: usize, faces: &[[u32; 3]]) -> MpasResult<Rings> {
    let mut degree = vec![0u32; n];
    for f in faces {
        for &c in f {
            degree[c as usize] += 1;
        }
    }
    let mut offsets = vec![0u32; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + degree[i];
    }
    let total = offsets[n] as usize;
    let mut cursor = offsets.clone();
    let mut step_from = vec![0u32; total];
    let mut step_to = vec![0u32; total];
    for f in faces {
        for s in 0..3 {
            let owner = f[s] as usize;
            let slot = cursor[owner] as usize;
            cursor[owner] += 1;
            step_from[slot] = f[(s + 1) % 3];
            step_to[slot] = f[(s + 2) % 3];
        }
    }

    let mut values = vec![0u32; total];
    // The degree floor is checked first and serially: it is a property of
    // the facet count alone, it costs one subtraction per cell, and the walk
    // below would index an empty slice without it.
    for i in 0..n {
        let deg = (offsets[i + 1] - offsets[i]) as usize;
        if deg < 3 {
            return Err(MpasError::Refusal(format!(
                "generator {i} is on only {deg} Delaunay triangles; its Voronoi cell is not a polygon, so it has no area to divide a flux by"
            )));
        }
    }
    // ONE CELL'S RING IS INDEPENDENT OF EVERY OTHER'S: the walk below reads
    // only `step_from`/`step_to` inside this cell's own slice and writes only
    // this cell's own slice, so the loop is a `par_iter` over disjoint
    // mutable slices. It is not a numerics change -- no arithmetic happens
    // here at all, only index chasing -- so the rings come out identical
    // slot for slot, which is what lets a class-A regeneration keep this.
    //
    // THE REFUSAL STAYS DETERMINISTIC. rayon would report whichever thread
    // failed first, and a refusal message that names a different cell on
    // every run is a refusal nobody can reproduce. Each cell records WHY it
    // failed into its own slot; the lowest-numbered failure is then re-read
    // in order, and it is the one refused on -- exactly the cell the serial
    // loop would have stopped at.
    const RING_OK: u8 = 0;
    const RING_OPEN: u8 = 1;
    const RING_EARLY: u8 = 2;
    const RING_REPEAT: u8 = 3;
    let mut fault = vec![RING_OK; n];
    {
        let mut slices: Vec<&mut [u32]> = Vec::with_capacity(n);
        let mut rest = &mut values[..];
        for i in 0..n {
            let deg = (offsets[i + 1] - offsets[i]) as usize;
            let (head, tail) = rest.split_at_mut(deg);
            slices.push(head);
            rest = tail;
        }
        slices
            .par_iter_mut()
            .zip(fault.par_iter_mut())
            .enumerate()
            .for_each(|(i, (out, flag))| {
                let lo = offsets[i] as usize;
                let hi = offsets[i + 1] as usize;
                let deg = hi - lo;
                let mut current = step_from[lo];
                for k in 0..deg {
                    out[k] = current;
                    match (lo..hi).find(|&t| step_from[t] == current) {
                        Some(t) => current = step_to[t],
                        None => {
                            *flag = RING_OPEN;
                            return;
                        }
                    }
                }
                if current != out[0] {
                    *flag = RING_EARLY;
                    return;
                }
                let mut seen: Vec<u32> = out.to_vec();
                seen.sort_unstable();
                seen.dedup();
                if seen.len() != deg {
                    *flag = RING_REPEAT;
                }
            });
    }
    if let Some(i) = fault.iter().position(|&f| f != RING_OK) {
        let deg = (offsets[i + 1] - offsets[i]) as usize;
        return Err(MpasError::Refusal(match fault[i] {
            RING_OPEN => format!(
                "the Delaunay triangles around generator {i} do not close into a single ring; its Voronoi cell would be an open figure and its area meaningless"
            ),
            RING_EARLY => format!(
                "the ring around generator {i} closes early, after fewer than its {deg} triangles; the triangles around it form more than one loop, which is a non-manifold vertex"
            ),
            _ => format!(
                "generator {i} lists a neighbour twice in its ring; two Delaunay triangles share two edges, which is a folded triangulation"
            ),
        }));
    }
    Ok(Rings { offsets, values })
}


// ======================================================================
// THE MAINTAINED TRIANGULATION -- CLASS B ONLY
// ======================================================================
//
// WHAT THIS IS FOR, in one sentence: the relaxation rebuilds the whole
// spherical Delaunay from scratch after every sweep to discover that almost
// nothing moved, and this is the machinery that discovers the same thing by
// repairing what it already has.
//
// THE MEASUREMENT THAT JUSTIFIES IT (profile lane, 2026-08-29, node-4): a
// rebuild costs 88.1 ms on a 54,247-cell graded mesh and 234.9 ms at 159,898,
// against 0.93 ms and 2.13 ms for one Lawson pass over the previous
// triangulation at the moved positions -- 95x and 110x -- and the pass finds
// 13.0 and 63.2 non-Delaunay edges out of 286,000 and 610,912. On the
// published UNIFORM family it finds exactly ZERO, every sweep, for 195 and 122
// consecutive sweeps, while 196 and 123 full rebuilds were performed.
//
// WHY IT IS CLASS B AND CAN NEVER BE ANYTHING ELSE. A rebuilt triangulation
// returns the same neighbour SET but a DIFFERENT ring rotation for 26.7-26.9%
// of cells on the uniform family and 4-16% on graded meshes, every sweep,
// because the facet array is laid out by the randomized insertion order. The
// centroid quadrature sums its sub-triangles in ring order and floating-point
// addition is not associative, so a rotated ring moves the generator's
// centroid in its last bits; the emitted grid file also records the rotation
// directly. A maintained triangulation KEEPS the rotation it had, which is
// exactly why it is cheap and exactly why it cannot reproduce a registered
// digest. This module states above that MPAS does not fix the starting slot;
// making it canonical would collapse the two classes and is a ruling nobody
// on this lane is entitled to make.
//
// WHAT IT DOES GUARANTEE. On termination every edge of the maintained surface
// is locally Delaunay, the facet count is still 2n-4, and `rings_from_faces`
// has accepted every ring as a closed, repeat-free manifold loop. A closed
// polyhedral surface whose vertices are cospherical and whose every edge is
// locally convex IS the convex hull of those vertices, and on cospherical
// points the convex hull IS the Delaunay triangulation -- so the maintained
// arm returns the SAME TRIANGULATION a rebuild would return, differing only in
// the ring rotation. That is the class-B quality claim, and the distributions
// in the receipt are what test it.

/// WHICH ARM keeps the triangulation between sweeps, stated in the request
/// rather than inferred.
///
/// THE BREAKAGE THIS ENUM PREVENTS, and it is the whole reason it is a request
/// field and not a heuristic: a maintained triangulation produces a DIFFERENT
/// GRID FILE from the same generators (see [`Triangulation`]). If the
/// generator chose the arm for itself, `rw_mpas_mesh --cells 40962` would
/// silently stop reproducing the published x1 mesh's SHA-256, every registry
/// pin that names it would lapse, and nothing in the run would say why. So
/// the fast arm is asked for by name, it stamps its name into the receipt,
/// and the default is the arm every registered digest was minted on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriangulationMode {
    /// CLASS A. Rebuild the spherical Delaunay from scratch after every
    /// sweep. This is what every registered mesh was generated with, and the
    /// only arm that reproduces one byte for byte.
    Rebuild,
    /// CLASS B. Keep the facet list and repair it by Lawson flips. Same
    /// triangulation, different ring rotation, different bytes. For meshes
    /// that have never existed.
    Maintained,
}

impl TriangulationMode {
    /// The CLI spelling, and the receipt's.
    pub fn parse(s: &str) -> Result<TriangulationMode, String> {
        match s {
            "rebuild" => Ok(TriangulationMode::Rebuild),
            "incremental" | "maintained" => Ok(TriangulationMode::Maintained),
            other => Err(format!(
                "--triangulation {other} is not a mode; it is `rebuild` (class A: rebuilt every sweep, the arm every registered mesh digest was minted on and the only one that reproduces one) or `incremental` (class B: {})",
                Triangulation::CLASS_NOTE
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TriangulationMode::Rebuild => "rebuild",
            TriangulationMode::Maintained => "incremental",
        }
    }

    pub fn is_maintained(self) -> bool {
        matches!(self, TriangulationMode::Maintained)
    }
}

impl Default for TriangulationMode {
    /// REBUILD. A default that changed the bytes of every mesh anybody
    /// regenerated would break every registered digest without being asked.
    fn default() -> Self {
        TriangulationMode::Rebuild
    }
}

/// One facet of a maintained triangulation: the same winding and adjacency
/// convention as [`Face`], without the construction-time conflict list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tri {
    /// Counter-clockwise seen from OUTSIDE the sphere.
    pub v: [u32; 3],
    /// `adj[i]` is the facet across the directed edge `(v[i], v[(i+1)%3])`.
    pub adj: [u32; 3],
}

/// A spherical Delaunay triangulation kept between sweeps.
///
/// CLASS B. Nothing that has to reproduce a registered digest may take rings
/// from here; see the module note above and [`Triangulation::CLASS_NOTE`].
#[derive(Debug, Clone)]
pub struct Triangulation {
    tris: Vec<Tri>,
    n_points: usize,
}

impl Triangulation {
    /// The sentence a caller prints, a receipt stamps and a refusal quotes, so
    /// the class is never implicit at any surface.
    pub const CLASS_NOTE: &'static str =
        "maintained triangulation (class B): the facet list is repaired by Lawson flips instead of rebuilt, which preserves each cell's ring ROTATION between sweeps where a rebuild re-rolls it for about 27% of cells. The triangulation is the same one; the emitted bytes are not. A mesh with a registered SHA-256 must be regenerated on the rebuild path";

    pub fn n_facets(&self) -> usize {
        self.tris.len()
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// The neighbour rings of the current facet list.
    pub fn rings(&self) -> MpasResult<Rings> {
        let v: Vec<[u32; 3]> = self.tris.iter().map(|t| t.v).collect();
        crate::mesh::profile::timed(
            &crate::mesh::profile::TRI_RINGS,
            self.n_points as u64,
            || rings_from_faces(self.n_points, &v),
        )
    }
}

/// Build the triangulation from scratch, keeping the facets.
///
/// The rings this yields are the rings [`delaunay_rings`] yields for the same
/// points: the live facets are compacted in their original array order and
/// `rings_from_faces` reads nothing else.
pub fn delaunay_triangulation(points: &[V3]) -> MpasResult<Triangulation> {
    let n = points.len();
    let faces = crate::mesh::profile::timed(&crate::mesh::profile::HULL, n as u64, || {
        build_hull(points)
    })?;
    // Compact, remapping adjacency into the compacted array. Retired facets
    // are never referenced by a live one on a closed hull; a reference that
    // survives here is a broken conflict graph and is refused rather than
    // carried into a maintained structure that would then flip against it.
    let mut remap = vec![NONE; faces.len()];
    let mut tris: Vec<Tri> = Vec::with_capacity(2 * n - 4);
    for (i, f) in faces.iter().enumerate() {
        if f.alive {
            remap[i] = tris.len() as u32;
            tris.push(Tri { v: f.v, adj: f.adj });
        }
    }
    for t in tris.iter_mut() {
        for s in 0..3 {
            let a = t.adj[s];
            if a == NONE || remap[a as usize] == NONE {
                return Err(MpasError::Refusal(
                    "a finished hull facet is adjacent to a retired one; the surface the maintained triangulation would flip on is not closed, and a flip across an open edge would fold the mesh".to_string(),
                ));
            }
            t.adj[s] = remap[a as usize];
        }
    }
    Ok(Triangulation { tris, n_points: n })
}

/// The slot of `f` whose directed edge faces `g`.
#[inline]
fn slot_facing(tris: &[Tri], f: u32, g: u32) -> Option<usize> {
    (0..3).find(|&s| tris[f as usize].adj[s] == g)
}

/// Restore the Delaunay property after the generators have MOVED, by Lawson
/// edge flips on the triangulation they had before they moved.
///
/// CLASS B; see [`Triangulation`]. Returns the number of flips performed.
///
/// TERMINATION, because a flip loop that does not terminate is a hang and not
/// a refusal. Every flip replaces a locally reflex edge of a closed polyhedral
/// surface whose vertices are cospherical, which strictly increases the volume
/// the surface encloses, so no configuration can repeat. The budget below is
/// therefore not the argument -- it is the tripwire for the case the argument
/// does not cover, an input whose motion has folded the surface, and it hands
/// the caller a REFUSAL naming the budget rather than spinning.
pub fn lawson_repair(points: &[V3], tri: &mut Triangulation) -> MpasResult<usize> {
    let _span =
        crate::mesh::profile::Span::new(&crate::mesh::profile::LAWSON, tri.n_points as u64);
    if points.len() != tri.n_points {
        return Err(MpasError::Refusal(format!(
            "a maintained triangulation of {} generators cannot be repaired against {} positions; the point set changed size, which is an insertion or a deletion and needs a rebuild, not a flip",
            tri.n_points,
            points.len()
        )));
    }
    let tris = &mut tri.tris;
    // PHASE 1, PARALLEL: which edges stopped being locally Delaunay.
    //
    // The common answer is NONE. Measured on the published uniform family,
    // 195 and 122 consecutive sweeps each moved every generator and flipped
    // exactly zero edges -- so the sweep-to-sweep cost of the maintained arm
    // is dominated by PROVING nothing changed, and proving it is one
    // independent predicate per edge. That is a `par_iter`, and `collect`
    // keeps the edges in index order so the seeds handed to the serial phase
    // are the same on every machine and every thread count.
    //
    // ONE DIRECTION PER UNDIRECTED EDGE (`f < g`). The two directions ask the
    // same question: the four-point orientation is an alternating function
    // and going from facet `f`'s operand order to facet `g`'s is two
    // transpositions, so the exact signs are equal by construction. Testing
    // both was half the work for none of the information.
    let n_slots = tris.len() * 3;
    let seeds: Vec<u32> = (0..n_slots as u32)
        .into_par_iter()
        .filter(|&e| {
            let f = (e / 3) as usize;
            let s = (e % 3) as usize;
            let g = tris[f].adj[s];
            if (g as usize) < f {
                return false;
            }
            let Some(t) = slot_facing(tris, g, f as u32) else {
                // A broken adjacency is the serial phase's refusal to make,
                // with the message that names it; here it is only a seed.
                return true;
            };
            let fv = tris[f].v;
            let y = tris[g as usize].v[(t + 2) % 3];
            visible_at(
                points[fv[0] as usize],
                points[fv[1] as usize],
                points[fv[2] as usize],
                points[y as usize],
                fv[0],
                fv[1],
                fv[2],
                y,
            )
        })
        .collect();
    let mut tests = n_slots as u64 / 2;
    if seeds.is_empty() {
        crate::mesh::profile::LAWSON.add(0, tests);
        crate::mesh::profile::LAWSON_FLIPS.add(0, 0);
        return Ok(0);
    }
    // PHASE 2, SERIAL: a flip rewrites two facets and re-aims two more, so it
    // cannot be done concurrently with its own neighbours. It is seeded with
    // the violations phase 1 found and propagates from there.
    let mut stack: Vec<(u32, u8)> = Vec::with_capacity(seeds.len() * 4 + 16);
    for &e in seeds.iter().rev() {
        stack.push(((e / 3), (e % 3) as u8));
    }
    let mut flips = 0usize;
    // 3 per facet is already generous: the measured pass finds 13 to 63
    // violations in 286,000 to 610,912 ring steps.
    let budget = 3 * tris.len() + 64;
    while let Some((f, s)) = stack.pop() {
        let s = s as usize;
        let fv = tris[f as usize].v;
        let g = tris[f as usize].adj[s];
        let u = fv[s];
        let w = fv[(s + 1) % 3];
        let x = fv[(s + 2) % 3];
        let t = match slot_facing(tris, g, f) {
            Some(t) => t,
            None => {
                return Err(MpasError::Refusal(
                    "the maintained triangulation's facet adjacency stopped being mutual; a flip has left the surface non-manifold and the rings taken from it would not close".to_string(),
                ));
            }
        };
        let gv = tris[g as usize].v;
        let y = gv[(t + 2) % 3];
        tests += 1;
        // The edge (u, w) is locally Delaunay exactly when the opposite
        // generator y lies INSIDE the plane of facet f -- the hull's own
        // visibility test, the same predicate, the same tie-break.
        if !visible_at(
            points[fv[0] as usize],
            points[fv[1] as usize],
            points[fv[2] as usize],
            points[y as usize],
            fv[0],
            fv[1],
            fv[2],
            y,
        ) {
            continue;
        }
        if flips >= budget {
            return Err(MpasError::Refusal(format!(
                "the Lawson repair spent its whole budget of {budget} flips on {} facets without reaching a locally Delaunay surface. Each flip on a cospherical, closed surface strictly increases the volume it encloses, so a budget this size is only reachable if the generators moved far enough to fold the surface; repairing that is a rebuild, not a flip",
                tris.len()
            )));
        }
        // The quad is w -> x -> u -> y counter-clockwise; the diagonal moves
        // from (u, w) to (x, y).
        let n_wx = tris[f as usize].adj[(s + 1) % 3];
        let n_xu = tris[f as usize].adj[(s + 2) % 3];
        let n_uy = tris[g as usize].adj[(t + 1) % 3];
        let n_yw = tris[g as usize].adj[(t + 2) % 3];
        tris[f as usize] = Tri {
            v: [x, u, y],
            adj: [n_xu, n_uy, g],
        };
        tris[g as usize] = Tri {
            v: [y, w, x],
            adj: [n_yw, n_wx, f],
        };
        // The two outer facets that changed hands.
        match slot_facing(tris, n_uy, g) {
            Some(k) => tris[n_uy as usize].adj[k] = f,
            None => {
                return Err(MpasError::Refusal(
                    "a flip found no back-pointer to re-aim across the (u, y) edge; the maintained triangulation's adjacency was already broken before the flip".to_string(),
                ));
            }
        }
        match slot_facing(tris, n_wx, f) {
            Some(k) => tris[n_wx as usize].adj[k] = g,
            None => {
                return Err(MpasError::Refusal(
                    "a flip found no back-pointer to re-aim across the (w, x) edge; the maintained triangulation's adjacency was already broken before the flip".to_string(),
                ));
            }
        }
        flips += 1;
        // The four edges of the quad can now be non-Delaunay; the diagonal
        // just created cannot be, and is not re-pushed.
        stack.push((f, 0));
        stack.push((f, 1));
        stack.push((g, 0));
        stack.push((g, 1));
    }
    crate::mesh::profile::LAWSON.add(0, tests);
    crate::mesh::profile::LAWSON_FLIPS.add(0, flips as u64);
    Ok(flips)
}

/// Repair `tri` in place, or rebuild it from scratch BY NAME if the repair
/// refuses.
///
/// A REFUSAL HERE IS NOT A FAILURE OF THE RUN. The maintained arm is an
/// optimisation of a computation the rebuild path performs exactly; when the
/// flip loop cannot get there, the rebuild is the answer it was approximating
/// and taking it costs time, not correctness. The fallback is COUNTED so it
/// cannot be silent: `hull.lawson_fallback_to_rebuild` in the profile report
/// carries it, and a non-zero count is a reportable event rather than a
/// tuning knob.
pub fn repair_or_rebuild(points: &[V3], tri: &mut Triangulation) -> MpasResult<usize> {
    match lawson_repair(points, tri) {
        Ok(flips) => Ok(flips),
        Err(_) => {
            crate::mesh::profile::LAWSON_FALLBACK.add(0, 1);
            *tri = delaunay_triangulation(points)?;
            Ok(0)
        }
    }
}

/// Normalise a batch of points onto the unit sphere, refusing anything without
/// a direction. Every entry point that accepts caller-supplied coordinates
/// should go through here rather than trusting the caller's normalisation.
pub fn to_unit_sphere(points: &[V3]) -> MpasResult<Vec<V3>> {
    points
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            unit(p).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "generator {i} is {p:?}, which has no direction; a point at the origin has no place on the sphere and the hull's orientation predicate would be meaningless for every facet touching it"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::geom::{arc, from_lat_lon};

    // --- the maintained arm, class B ------------------------------------

    /// Move every generator by a fraction of a cell width, the way one Lloyd
    /// sweep does.
    fn jitter(points: &[V3], amount: f64, seed: u64) -> Vec<V3> {
        let mut rng = SplitMix(seed);
        points
            .iter()
            .map(|&p| {
                let d = [
                    (rng.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5,
                    (rng.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5,
                    (rng.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5,
                ];
                unit(add(p, scale(d, amount))).expect("jittered point keeps a direction")
            })
            .collect()
    }

    /// THE BREAKAGE: `delaunay_triangulation` compacts the live facets and
    /// remaps their adjacency, and `Triangulation::rings` then reads the
    /// compacted array. If the compaction reordered the facets, the ring
    /// ROTATIONS would move and the maintained arm would start diverging from
    /// the rebuild arm on its very FIRST sweep -- before a single flip, with
    /// nothing in the run to say so. The two must agree exactly on entry.
    #[test]
    fn the_maintained_arm_starts_from_exactly_the_rings_a_rebuild_gives() {
        for n in [200usize, 1_000, 4_000] {
            let pts = spiral(n);
            let direct = delaunay_rings(&pts).expect("rebuild");
            let tri = delaunay_triangulation(&pts).expect("triangulation");
            let kept = tri.rings().expect("rings from facets");
            assert_eq!(
                direct.offsets, kept.offsets,
                "{n} generators: the compacted facet list gives different ring OFFSETS"
            );
            assert_eq!(
                direct.values, kept.values,
                "{n} generators: the compacted facet list gives a different ring order, so the maintained arm would diverge from the rebuild arm before any flip"
            );
        }
    }

    /// THE CLASS-B QUALITY CLAIM, and the only one worth making: after the
    /// generators move, repairing the old triangulation by Lawson flips gives
    /// the SAME triangulation a full rebuild gives -- the same neighbour SET
    /// for every cell. If that failed, the maintained arm would be relaxing a
    /// mesh whose Voronoi cells are not the Voronoi cells of its generators,
    /// and every dv/dc reading taken off it would be measuring the wrong
    /// polygon.
    #[test]
    fn a_repaired_triangulation_is_the_same_one_a_rebuild_gives() {
        for (n, amount) in [(200usize, 0.02f64), (1_000, 0.01), (4_000, 0.004)] {
            let pts = spiral(n);
            let mut tri = delaunay_triangulation(&pts).expect("triangulation");
            let moved = jitter(&pts, amount, 0xA11CE ^ n as u64);
            let flips = lawson_repair(&moved, &mut tri).expect("repair");
            assert!(
                flips > 0,
                "{n} generators moved by {amount} and the repair found nothing to flip; the case is not exercising what it claims to"
            );
            let repaired = tri.rings().expect("rings");
            let rebuilt = delaunay_rings(&moved).expect("rebuild");
            assert_eq!(repaired.n_cells(), rebuilt.n_cells());
            for i in 0..rebuilt.n_cells() {
                let mut a = repaired.ring(i).to_vec();
                let mut b = rebuilt.ring(i).to_vec();
                a.sort_unstable();
                b.sort_unstable();
                assert_eq!(
                    a, b,
                    "cell {i} of {n}: the repaired triangulation gives neighbours {a:?} where a rebuild gives {b:?}; the maintained arm is not the Delaunay of its own generators"
                );
            }
        }
    }

    /// The other half of the same claim, stated the way the class boundary is
    /// stated: the two arms agree on the SET and disagree on the ROTATION,
    /// which is exactly why one can reproduce a registered digest and the
    /// other cannot. A test that found zero rotations would mean the class
    /// split had quietly become unnecessary, and that is a finding, not a
    /// pass.
    #[test]
    fn the_repaired_rings_differ_from_a_rebuild_only_by_rotation() {
        let n = 4_000usize;
        let pts = spiral(n);
        let mut tri = delaunay_triangulation(&pts).expect("triangulation");
        let moved = jitter(&pts, 0.004, 0xB0B);
        lawson_repair(&moved, &mut tri).expect("repair");
        let repaired = tri.rings().expect("rings");
        let rebuilt = delaunay_rings(&moved).expect("rebuild");
        let mut rotated = 0usize;
        for i in 0..n {
            if repaired.ring(i) != rebuilt.ring(i) {
                rotated += 1;
            }
        }
        assert!(
            rotated > 0,
            "not one ring rotated between the maintained and the rebuilt triangulation of {n} moved generators. If that is so the two classes have merged and `TriangulationMode` is dead weight -- check it before deleting the split"
        );
    }

    /// THE BREAKAGE: a flip rewrites two facets and has to re-aim the
    /// back-pointers of the two OUTER facets that changed hands. Miss one and
    /// the surface stops being mutually adjacent; the next flip across that
    /// edge folds the mesh, and `rings_from_faces` reports a non-manifold
    /// vertex several sweeps later with nothing pointing at the cause.
    #[test]
    fn every_facet_stays_mutually_adjacent_across_a_repair() {
        let n = 2_000usize;
        let pts = spiral(n);
        let mut tri = delaunay_triangulation(&pts).expect("triangulation");
        let mut moved = pts.clone();
        for round in 0..6 {
            moved = jitter(&moved, 0.006, 0xC0FFEE + round);
            lawson_repair(&moved, &mut tri).expect("repair");
            assert_eq!(
                tri.n_facets(),
                2 * n - 4,
                "round {round}: a flip changed the facet count, which no flip can do"
            );
            for f in 0..tri.tris.len() {
                for s in 0..3 {
                    let g = tri.tris[f].adj[s] as usize;
                    let back = (0..3).find(|&t| tri.tris[g].adj[t] == f as u32);
                    assert!(
                        back.is_some(),
                        "round {round}: facet {f} slot {s} points at {g}, which does not point back"
                    );
                    let t = back.expect("mutual");
                    let (u, w) = (tri.tris[f].v[s], tri.tris[f].v[(s + 1) % 3]);
                    assert_eq!(
                        (tri.tris[g].v[t], tri.tris[g].v[(t + 1) % 3]),
                        (w, u),
                        "round {round}: facet {f}'s edge ({u}, {w}) faces {g}, whose matching edge is not ({w}, {u}); the surface is folded"
                    );
                }
            }
        }
    }

    /// THE BREAKAGE THIS NAMES: a maintained triangulation cannot absorb an
    /// insertion or a deletion, because a flip only moves a diagonal. Surgery
    /// edits the point set between polish calls; if the repair accepted a
    /// different-sized point array it would index off the end of the facet
    /// list or, worse, flip against stale vertices and emit a mesh whose
    /// topology belongs to the point set from before the edit.
    #[test]
    fn a_repair_against_a_resized_point_set_is_refused_by_name() {
        let pts = spiral(500);
        let mut tri = delaunay_triangulation(&pts).expect("triangulation");
        let mut shorter = pts.clone();
        shorter.pop();
        let err = lawson_repair(&shorter, &mut tri).expect_err("must refuse");
        let text = err.to_string();
        assert!(
            text.contains("insertion or a deletion") && text.contains("rebuild"),
            "the refusal has to say the point set changed size and that a rebuild is what absorbs it; it said: {text}"
        );
    }

    /// The class switch is a REQUEST field with a rebuild default, and the
    /// default is what every registered mesh digest was minted on. A default
    /// that flipped to the fast arm would lapse every one of those pins with
    /// nothing in the run saying why.
    #[test]
    fn the_default_triangulation_mode_is_the_class_a_arm() {
        assert_eq!(TriangulationMode::default(), TriangulationMode::Rebuild);
        assert!(!TriangulationMode::default().is_maintained());
        assert_eq!(TriangulationMode::Rebuild.as_str(), "rebuild");
        assert_eq!(TriangulationMode::Maintained.as_str(), "incremental");
        assert_eq!(
            TriangulationMode::parse("rebuild"),
            Ok(TriangulationMode::Rebuild)
        );
        assert_eq!(
            TriangulationMode::parse("incremental"),
            Ok(TriangulationMode::Maintained)
        );
        let err = TriangulationMode::parse("fast").expect_err("must refuse");
        assert!(
            err.contains("registered") && err.contains("rebuild") && err.contains("incremental"),
            "an unknown mode must be refused by naming both modes and what the default protects; it said: {err}"
        );
    }

    /// `repair_or_rebuild` is the door the relaxation uses, and it must never
    /// return a triangulation that is not the Delaunay of the points it was
    /// handed -- including on the path where the repair refuses and the
    /// rebuild takes over.
    #[test]
    fn the_fallback_door_returns_the_delaunay_either_way() {
        let n = 1_500usize;
        let pts = spiral(n);
        let mut tri = delaunay_triangulation(&pts).expect("triangulation");
        let moved = jitter(&pts, 0.008, 0xD00D);
        repair_or_rebuild(&moved, &mut tri).expect("door");
        let got = tri.rings().expect("rings");
        let want = delaunay_rings(&moved).expect("rebuild");
        for i in 0..n {
            let mut a = got.ring(i).to_vec();
            let mut b = want.ring(i).to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "cell {i}");
        }
    }

    /// Points on a golden-ratio spiral: the seed this crate's generator uses,
    /// and a case with no exact symmetry.
    fn spiral(n: usize) -> Vec<V3> {
        let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        (0..n)
            .map(|k| {
                let z = 1.0 - (2 * k + 1) as f64 / n as f64;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let t = ga * k as f64;
                [r * t.cos(), r * t.sin(), z]
            })
            .collect()
    }

    fn check_closed(n: usize, rings: &Rings) {
        let mut edges = std::collections::HashSet::new();
        let mut deg_sum = 0usize;
        for i in 0..n {
            let ring = rings.ring(i);
            deg_sum += ring.len();
            for &j in ring {
                assert!(rings.ring(j as usize).contains(&(i as u32)), "{i} -> {j} is not mutual");
                edges.insert((i.min(j as usize), i.max(j as usize)));
            }
        }
        let n_edges = edges.len();
        let n_vertices = 2 * n - 4;
        assert_eq!(deg_sum, 2 * n_edges, "sum(degree) != 2 * edges");
        assert_eq!(n_edges, 3 * n - 6, "edge count is not 3n-6");
        assert_eq!(
            n as i64 - n_edges as i64 + n_vertices as i64,
            2,
            "Euler characteristic"
        );
        let defect: i64 = (0..n).map(|i| 6 - rings.degree(i) as i64).sum();
        assert_eq!(defect, 12, "total coordination defect");
    }

    #[test]
    fn a_tetrahedron_of_generators_triangulates() {
        let pts = to_unit_sphere(&[
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ])
        .unwrap();
        let rings = delaunay_rings(&pts).unwrap();
        check_closed(4, &rings);
        for i in 0..4 {
            assert_eq!(rings.degree(i), 3);
        }
    }

    #[test]
    fn spiral_points_close_the_sphere_at_several_sizes() {
        for &n in &[8usize, 37, 200, 1013, 4096] {
            let pts = to_unit_sphere(&spiral(n)).unwrap();
            let rings = delaunay_rings(&pts).unwrap();
            check_closed(n, &rings);
        }
    }

    #[test]
    fn the_rings_wind_counter_clockwise_seen_from_outside() {
        let n = 500;
        let pts = to_unit_sphere(&spiral(n)).unwrap();
        let rings = delaunay_rings(&pts).unwrap();
        // For a Voronoi ring the neighbours are in angular order around the
        // cell; measure it directly with the azimuth in the cell's own tangent
        // frame, which must increase monotonically once around.
        for i in 0..n {
            let c = pts[i];
            let (east, north) = crate::mesh::geom::east_north(c).unwrap();
            let ring = rings.ring(i);
            let angles: Vec<f64> = ring
                .iter()
                .map(|&j| {
                    let t = crate::mesh::geom::tangent_at(c, crate::mesh::geom::sub(pts[j as usize], c));
                    crate::mesh::geom::dot(t, north).atan2(crate::mesh::geom::dot(t, east))
                })
                .collect();
            let mut turns = 0.0f64;
            for k in 0..angles.len() {
                let mut d = angles[(k + 1) % angles.len()] - angles[k];
                while d > std::f64::consts::PI {
                    d -= std::f64::consts::TAU;
                }
                while d < -std::f64::consts::PI {
                    d += std::f64::consts::TAU;
                }
                turns += d;
            }
            assert!(
                (turns - std::f64::consts::TAU).abs() < 1e-6,
                "cell {i}: the ring turns {turns} rad, not +2pi -- it is wound clockwise or is not in angular order"
            );
        }
    }

    #[test]
    fn every_triangle_has_an_empty_circumcircle() {
        // The defining Delaunay property, checked by brute force at a size
        // where brute force is affordable. This is the direction that proves
        // the hull is a triangulation and not merely a closed surface.
        let n = 300;
        let pts = to_unit_sphere(&spiral(n)).unwrap();
        let rings = delaunay_rings(&pts).unwrap();
        let mut triangles: Vec<[usize; 3]> = Vec::new();
        for i in 0..n {
            let ring = rings.ring(i);
            let deg = ring.len();
            for k in 0..deg {
                let mut t = [i, ring[k] as usize, ring[(k + 1) % deg] as usize];
                t.sort_unstable();
                triangles.push(t);
            }
        }
        triangles.sort_unstable();
        triangles.dedup();
        assert_eq!(
            triangles.len(),
            2 * n - 4,
            "the ring set does not carry exactly 2n-4 distinct triangles"
        );
        for t in &triangles {
            let (a, b, c) = (t[0], t[1], t[2]);
            let cc = crate::mesh::geom::circumcenter(pts[a], pts[b], pts[c]).unwrap();
            let radius = arc(cc, pts[a]);
            for d in 0..n {
                if d == a || d == b || d == c {
                    continue;
                }
                assert!(
                    arc(cc, pts[d]) >= radius * (1.0 - 1e-12),
                    "point {d} is inside the circumcircle of triangle ({a},{b},{c})"
                );
            }
        }
    }

    #[test]
    fn coincident_generators_are_refused_by_name() {
        let mut pts = to_unit_sphere(&spiral(50)).unwrap();
        pts[17] = pts[4];
        let err = delaunay_rings(&pts).unwrap_err().to_string();
        assert!(err.contains("coincident generators"), "{err}");
    }

    #[test]
    fn too_few_generators_are_refused_by_name() {
        let pts = to_unit_sphere(&spiral(3)).unwrap();
        let err = delaunay_rings(&pts).unwrap_err().to_string();
        assert!(err.contains("cannot triangulate a sphere"), "{err}");
    }

    #[test]
    fn a_point_at_the_origin_is_refused_by_name() {
        let err = to_unit_sphere(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            .unwrap_err()
            .to_string();
        assert!(err.contains("has no direction"), "{err}");
    }

    #[test]
    fn insertion_order_does_not_change_the_answer() {
        // The conflict-graph shuffle is seeded from the point count, so the
        // same input gives the same mesh. Reversing the input must give the
        // same triangulation as a SET, or the orientation predicate is not
        // deciding degeneracies consistently.
        let n = 700;
        let pts = to_unit_sphere(&spiral(n)).unwrap();
        let a = delaunay_rings(&pts).unwrap();
        let mut rev: Vec<V3> = pts.clone();
        rev.reverse();
        let b = delaunay_rings(&rev).unwrap();
        let key = |r: &Rings, n: usize, flip: bool| -> Vec<(u32, u32)> {
            let mut out = Vec::new();
            for i in 0..n {
                for &j in r.ring(i) {
                    let (x, y) = if flip {
                        ((n - 1 - i) as u32, n as u32 - 1 - j)
                    } else {
                        (i as u32, j)
                    };
                    out.push((x.min(y), x.max(y)));
                }
            }
            out.sort_unstable();
            out.dedup();
            out
        };
        assert_eq!(key(&a, n, false), key(&b, n, true), "the edge set depends on input order");
    }

    #[test]
    fn a_lopsided_cluster_still_closes() {
        // Half the points packed into a small cap, half spread over the rest:
        // the near-degenerate case a variable-resolution mesh actually creates,
        // where circumradii span two orders of magnitude.
        let mut pts: Vec<V3> = Vec::new();
        let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        for k in 0..400 {
            let t = ga * k as f64;
            let r = 0.02 * ((k as f64 + 0.5) / 400.0).sqrt();
            pts.push(from_lat_lon(0.6 + r * t.cos(), -1.7 + r * t.sin()));
        }
        for k in 0..400 {
            let z = 1.0 - (2 * k + 1) as f64 / 400.0;
            let rr = (1.0 - z * z).max(0.0).sqrt();
            let t = ga * (k + 7) as f64;
            pts.push([rr * t.cos(), rr * t.sin(), z]);
        }
        let pts = to_unit_sphere(&pts).unwrap();
        let n = pts.len();
        let rings = delaunay_rings(&pts).unwrap();
        check_closed(n, &rings);
    }

    /// THE BREAKAGE: a different winner on a tie is a different seed
    /// tetrahedron, and a different seed tetrahedron reorders every facet,
    /// which rotates every emitted neighbour ring -- a different grid file
    /// from the same generators, which no registry could pin under a stable
    /// name. `initial_tetrahedron` used `max_by`, which evaluates BOTH sides'
    /// key on every comparison and returns the LAST of several equal maxima;
    /// the single-evaluation fold that replaced it has to answer identically,
    /// ties included. Ties are the case that matters: the icosahedral seed is
    /// exactly symmetric and produces them by construction.
    #[test]
    fn the_single_evaluation_pick_answers_exactly_what_max_by_answered() {
        // A key with deliberate plateaus, so ties are the common case rather
        // than the rare one.
        let keys: Vec<f64> = (0..1000).map(|i| ((i % 7) as f64) * 0.5).collect();
        let by_max_by = (0..keys.len())
            .max_by(|&i, &j| keys[i].partial_cmp(&keys[j]).unwrap())
            .unwrap();
        let mut best = 0usize;
        let mut best_key = keys[0];
        for i in 1..keys.len() {
            if keys[i] >= best_key {
                best = i;
                best_key = keys[i];
            }
        }
        assert_eq!(
            best, by_max_by,
            "the fold picked candidate {best} where max_by picked {by_max_by};              on a tie the two disagree, so the seed tetrahedron moves and every              emitted ring rotates"
        );
    }

    /// THE BREAKAGE: the horizon's edge index was a `HashMap`, whose `insert`
    /// overwrites, so a repeated directed edge resolved to the LAST facet
    /// written. The linear scan that replaced it must resolve the same way or
    /// a degenerate horizon would close its fan against a different facet --
    /// silently, since both spellings return `Some`.
    #[test]
    fn the_horizon_scan_resolves_a_repeated_edge_the_way_the_map_did() {
        let mut map: HashMap<(u32, u32), (u32, usize)> = HashMap::new();
        let mut ring: Vec<((u32, u32), (u32, usize))> = Vec::new();
        for entry in [
            ((1u32, 2u32), (10u32, 1usize)),
            ((3, 4), (11, 2)),
            ((1, 2), (12, 2)),
        ] {
            map.insert(entry.0, entry.1);
            ring.push(entry);
        }
        let from_map = *map.get(&(1, 2)).unwrap();
        let from_scan = ring.iter().rev().find(|(k, _)| *k == (1, 2)).unwrap().1;
        assert_eq!(
            from_map, from_scan,
            "the backwards scan disagreed with the map it replaced on a              repeated directed edge"
        );
    }
}
