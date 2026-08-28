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
    let n = points.len();
    if n < 4 {
        return Err(MpasError::Refusal(format!(
            "{n} generators cannot triangulate a sphere; four is the minimum that encloses a volume, and below it there is no Delaunay triangle to take a Voronoi vertex from"
        )));
    }
    if let Some((i, j)) = exact_duplicate(points) {
        return Err(MpasError::Refusal(format!(
            "generators {i} and {j} are the same point to the last bit; two coincident generators have no Voronoi boundary between them, so the hull would build a zero-area facet and dcEdge would be 0 -- which the tangential weight formula divides by"
        )));
    }

    let mut faces = initial_tetrahedron(points)?;
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

    let mut point_face = vec![NONE; n];
    for &p in &order {
        let mut placed = false;
        for (fi, f) in faces.iter_mut().enumerate() {
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
    let mut orphans: Vec<u32> = Vec::new();
    let mut edge_map: HashMap<(u32, u32), (u32, usize)> = HashMap::new();

    for &p in &order {
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
                if a != NONE && stamp[a as usize] != generation && visible(points, &faces[a as usize], p)
                {
                    stamp[a as usize] = generation;
                    stack.push(a);
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
        orphans.clear();
        for &fi in &visible_faces {
            faces[fi as usize].alive = false;
            orphans.append(&mut faces[fi as usize].conflicts);
        }

        // --- one new facet per horizon edge. The edge is taken in the
        //     direction the RETIRED facet carried it, which is what leaves the
        //     rebuilt surface consistently wound.
        let first_new = faces.len() as u32;
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
        edge_map.clear();
        for idx in first_new..faces.len() as u32 {
            let v = faces[idx as usize].v;
            edge_map.insert((v[1], v[2]), (idx, 1));
            edge_map.insert((v[2], v[0]), (idx, 2));
        }
        for idx in first_new..faces.len() as u32 {
            let v = faces[idx as usize].v;
            for (slot, (a, b)) in [(1usize, (v[1], v[2])), (2usize, (v[2], v[0]))] {
                if let Some(&(other, _)) = edge_map.get(&(b, a)) {
                    faces[idx as usize].adj[slot] = other;
                } else {
                    return Err(MpasError::Refusal(format!(
                        "the facets created around generator {p} do not close into a fan; the horizon was not a single ring, which means the orientation predicate answered inconsistently on a near-degenerate configuration"
                    )));
                }
            }
        }

        // --- redistribute the orphaned points onto the new facets
        for q in orphans.drain(..) {
            if q == p {
                continue;
            }
            let mut home = NONE;
            for idx in first_new..faces.len() as u32 {
                if visible(points, &faces[idx as usize], q) {
                    home = idx;
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

    let live: Vec<&Face> = faces.iter().filter(|f| f.alive).collect();
    let expected = 2 * n - 4;
    if live.len() != expected {
        return Err(MpasError::Refusal(format!(
            "the hull closed with {} facets; a Delaunay triangulation of {n} points on a sphere has exactly 2n-4 = {expected}. A different count means at least one generator was swallowed or duplicated, and the mesh would carry a cell with no dual triangle",
            live.len()
        )));
    }
    rings_from_faces(n, &live)
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
    let a = points[f.v[0] as usize];
    let b = points[f.v[1] as usize];
    let c = points[f.v[2] as usize];
    let d = points[p as usize];
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
        orient3d_sos(a, b, c, d, f.v[0], f.v[1], f.v[2], p) > 0.0
    }
}

/// Four generators that enclose a volume, chosen to be as far from degenerate
/// as the input allows: the farthest pair, then the point farthest off that
/// line, then the point farthest off that plane.
fn initial_tetrahedron(points: &[V3]) -> MpasResult<Vec<Face>> {
    let n = points.len();
    let a = 0usize;
    let b = (0..n)
        .max_by(|&i, &j| {
            crate::mesh::geom::chord(points[a], points[i])
                .partial_cmp(&crate::mesh::geom::chord(points[a], points[j]))
                .unwrap()
        })
        .unwrap();
    let ab = crate::mesh::geom::sub(points[b], points[a]);
    let c = (0..n)
        .max_by(|&i, &j| {
            let di = crate::mesh::geom::norm(crate::mesh::geom::cross(
                ab,
                crate::mesh::geom::sub(points[i], points[a]),
            ));
            let dj = crate::mesh::geom::norm(crate::mesh::geom::cross(
                ab,
                crate::mesh::geom::sub(points[j], points[a]),
            ));
            di.partial_cmp(&dj).unwrap()
        })
        .unwrap();
    let d = (0..n)
        .max_by(|&i, &j| {
            orient3d(points[a], points[b], points[c], points[i])
                .abs()
                .partial_cmp(&orient3d(points[a], points[b], points[c], points[j]).abs())
                .unwrap()
        })
        .unwrap();
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
fn rings_from_faces(n: usize, faces: &[&Face]) -> MpasResult<Rings> {
    let mut degree = vec![0u32; n];
    for f in faces {
        for &c in &f.v {
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
            let owner = f.v[s] as usize;
            let slot = cursor[owner] as usize;
            cursor[owner] += 1;
            step_from[slot] = f.v[(s + 1) % 3];
            step_to[slot] = f.v[(s + 2) % 3];
        }
    }

    let mut values = vec![0u32; total];
    for i in 0..n {
        let lo = offsets[i] as usize;
        let hi = offsets[i + 1] as usize;
        let deg = hi - lo;
        if deg < 3 {
            return Err(MpasError::Refusal(format!(
                "generator {i} is on only {deg} Delaunay triangles; its Voronoi cell is not a polygon, so it has no area to divide a flux by"
            )));
        }
        let mut current = step_from[lo];
        for k in 0..deg {
            values[lo + k] = current;
            let next = (lo..hi)
                .find(|&t| step_from[t] == current)
                .map(|t| step_to[t])
                .ok_or_else(|| {
                    MpasError::Refusal(format!(
                        "the Delaunay triangles around generator {i} do not close into a single ring; its Voronoi cell would be an open figure and its area meaningless"
                    ))
                })?;
            current = next;
        }
        if current != values[lo] {
            return Err(MpasError::Refusal(format!(
                "the ring around generator {i} closes early, after fewer than its {deg} triangles; the triangles around it form more than one loop, which is a non-manifold vertex"
            )));
        }
        // no repeats
        let mut seen = values[lo..hi].to_vec();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != deg {
            return Err(MpasError::Refusal(format!(
                "generator {i} lists a neighbour twice in its ring; two Delaunay triangles share two edges, which is a folded triangulation"
            )));
        }
    }
    Ok(Rings { offsets, values })
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
}
