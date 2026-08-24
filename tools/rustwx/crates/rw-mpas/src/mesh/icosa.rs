//! Icosahedral (Goldberg) seeding for uniform meshes.
//!
//! WHY THIS EXISTS. A Fibonacci-seeded uniform SCVT is polycrystalline: its
//! Delaunay carries pentagon-heptagon dislocation pairs beyond the twelve
//! pentagons topology requires, and a dislocation quad is near-cocircular, so
//! its dual edge can be arbitrarily short AT EQUILIBRIUM. Measured on the
//! shipped pipeline at a 120 km background: shortest dual edge 7,337.6 m at
//! 2,000 cells, 75.0 m at 12,000, 7.2 m at 40,962 -- and `--sweeps` 200, 600
//! and 2000 all reproduce the same short edges, because relaxation re-rolls
//! the tail rather than draining it. The published x1.40962 has no dual edge
//! under 45 km at the same cell count because it IS a subdivided icosahedron:
//! exactly 12 pentagons, no heptagons, nothing for a near-cocircular quad to
//! form around.
//!
//! So uniform meshes seed the same way the published family was built: the
//! vertices of a Goldberg subdivision GP(m, n) of the icosahedron, projected
//! to the sphere. Lloyd then polishes GEOMETRY on a topology that has no
//! dislocations to begin with. The defect class does not exist here, rather
//! than being caught afterwards.
//!
//! THE COUNT IS SNAPPED. GP(m, n) delivers exactly `10*(m^2 + mn + n^2) + 2`
//! cells, so an arbitrary request is moved to the nearest achievable count
//! (127,051 -> 127,032 via GP(102,19), -0.015%) and the move is recorded in
//! the receipt. Variable-resolution meshes keep the density-biased Fibonacci
//! seed: their refinement gradient is incompatible with a global subdivision,
//! and their dislocations are topologically unavoidable.

use crate::error::{MpasError, MpasResult};
use crate::mesh::geom::{V3, add, chord, scale, unit};

/// A cell count the icosahedral subdivision can actually deliver, and the
/// subdivision that delivers it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldbergChoice {
    /// Subdivision frequencies: GP(m, n), with `n <= m` canonical (GP(n, m) is
    /// the mirror image at the same count).
    pub m: u32,
    pub n: u32,
    /// `10*(m^2 + mn + n^2) + 2`.
    pub cells: usize,
}

/// The triangulation number `T = m^2 + mn + n^2`.
fn triangulation_number(m: u32, n: u32) -> u64 {
    let (m, n) = (m as u64, n as u64);
    m * m + m * n + n * n
}

/// Cells of GP(m, n): `10T + 2`.
pub fn goldberg_cells(m: u32, n: u32) -> usize {
    (10 * triangulation_number(m, n) + 2) as usize
}

/// The nearest achievable Goldberg count to `requested`.
///
/// With `ceiling` set, only counts `<= requested` are considered -- a count
/// that came from a device memory budget is a ceiling, not a preference, and
/// snapping upward would size a mesh past the card that asked for it.
pub fn snap_cells(requested: usize, ceiling: bool) -> MpasResult<GoldbergChoice> {
    if requested < 12 {
        return Err(MpasError::Refusal(format!(
            "{requested} cells cannot carry the twelve pentagons every triangulated sphere must have; GP(1,0) -- the dodecahedron itself -- is the smallest closed mesh at 12"
        )));
    }
    let t_target = (requested.saturating_sub(2)) as f64 / 10.0;
    // The next Class-I frequency up bounds every candidate worth looking at.
    let m_max = (t_target.sqrt().ceil() as u32).max(1) + 2;
    let mut best: Option<GoldbergChoice> = None;
    for m in 1..=m_max {
        for n in 0..=m {
            let cells = goldberg_cells(m, n);
            if ceiling && cells > requested {
                continue;
            }
            let gap = cells.abs_diff(requested);
            let better = match best {
                None => true,
                Some(b) => {
                    let bgap = b.cells.abs_diff(requested);
                    gap < bgap || (gap == bgap && (n, m) < (b.n, b.m))
                }
            };
            if better {
                best = Some(GoldbergChoice { m, n, cells });
            }
        }
    }
    best.ok_or_else(|| {
        MpasError::Refusal(format!(
            "no Goldberg subdivision fits under {requested} cells; the smallest, GP(1,0), already needs 12"
        ))
    })
}

/// The 12 vertices of the icosahedron, on the unit sphere.
fn icosahedron_vertices() -> Vec<V3> {
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    let mut verts = Vec::with_capacity(12);
    for &s in &[1.0, -1.0] {
        for &t in &[phi, -phi] {
            verts.push([0.0, s, t]);
            verts.push([s, t, 0.0]);
            verts.push([t, 0.0, s]);
        }
    }
    verts
        .into_iter()
        .map(|v| unit(v).expect("icosahedron vertices are not at the origin"))
        .collect()
}

/// The 20 faces, found from the vertices' own adjacency rather than typed in:
/// a hand-kept index table is exactly the kind of silent wrongness a seed
/// cannot afford, and the triple count is checked instead.
///
/// Every face is returned wound COUNTER-CLOCKWISE seen from outside, and that
/// is load-bearing for a CHIRAL breakdown (n != 0, n != m): GP(m, n) mapped
/// onto a clockwise face is its mirror GP(n, m), and a sphere tiled with both
/// chiralities meets itself along twin seams where the two lattices misalign
/// by a fraction of a spacing -- near-coincident generators, which is the
/// defect class this seeding exists to make impossible. MEASURED before the
/// orientation was enforced: 44 cross-seam generator pairs at ~4% of a
/// spacing on GP(13,2).
fn icosahedron_faces(verts: &[V3]) -> MpasResult<Vec<[usize; 3]>> {
    let mut min_chord = f64::INFINITY;
    for i in 0..verts.len() {
        for j in i + 1..verts.len() {
            min_chord = min_chord.min(chord(verts[i], verts[j]));
        }
    }
    let adjacent = |i: usize, j: usize| chord(verts[i], verts[j]) < min_chord * 1.5;
    let mut faces = Vec::with_capacity(20);
    for i in 0..verts.len() {
        for j in i + 1..verts.len() {
            if !adjacent(i, j) {
                continue;
            }
            for k in j + 1..verts.len() {
                if adjacent(i, k) && adjacent(j, k) {
                    // Outward-CCW: the face normal points away from the origin.
                    let n = crate::mesh::geom::cross(
                        crate::mesh::geom::sub(verts[j], verts[i]),
                        crate::mesh::geom::sub(verts[k], verts[i]),
                    );
                    if crate::mesh::geom::dot(n, verts[i]) > 0.0 {
                        faces.push([i, j, k]);
                    } else {
                        faces.push([i, k, j]);
                    }
                }
            }
        }
    }
    if faces.len() != 20 {
        return Err(MpasError::Refusal(format!(
            "the icosahedron adjacency produced {} faces, not 20; a seed built on it would not be a subdivision of the icosahedron and its Delaunay would not carry exactly twelve pentagons",
            faces.len()
        )));
    }
    Ok(faces)
}

/// The `10T + 2` generators of GP(m, n), projected to the unit sphere.
///
/// Each face carries the integer triangular lattice with corners `(0,0)`,
/// `(m,n)` and `(-n, m+n)` -- the standard Goldberg breakdown -- mapped
/// barycentrically onto the face and projected radially. Lattice membership is
/// decided in integer arithmetic, so no point is gained or lost to rounding;
/// points shared between faces are merged geometrically, and the final count
/// is CHECKED against `10T + 2` rather than trusted.
pub fn seed(m: u32, n: u32) -> MpasResult<Vec<V3>> {
    if m == 0 {
        return Err(MpasError::Refusal(
            "GP(0, n) subdivides nothing; the frequency m has to be at least 1".to_string(),
        ));
    }
    let t = triangulation_number(m, n);
    let expected = goldberg_cells(m, n);
    let verts = icosahedron_vertices();
    let faces = icosahedron_faces(&verts)?;

    // Corners of the breakdown triangle in lattice coordinates.
    let (mi, ni) = (m as i64, n as i64);
    let corner_b = (mi, ni);
    let corner_c = (-ni, mi + ni);
    let det = |a: (i64, i64), b: (i64, i64)| a.0 * b.1 - a.1 * b.0;
    debug_assert_eq!(det(corner_b, corner_c), t as i64);

    // Merge tolerance: the same point computed from two faces agrees to
    // machine precision (corners are bitwise equal; gcd-shared edge points
    // agree to ~1e-16), while the closest genuinely distinct pair -- two
    // near-edge lattice rows folded toward each other across the dihedral --
    // sits at ~0.9/sqrt(T) of a chord. 1e-9 is seven decades above the
    // duplicates and, out to T ~ 1e17, decades below the distinct pairs.
    let tol = 1e-9;

    let mut points: Vec<V3> = Vec::with_capacity(expected);
    let mut grid: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    let cell = tol;
    let key_of = |p: V3| -> (i64, i64, i64) {
        (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        )
    };

    for face in &faces {
        let (v0, v1, v2) = (verts[face[0]], verts[face[1]], verts[face[2]]);
        for i in -ni..=mi {
            for j in 0..=(mi + ni) {
                let p = (i, j);
                let num_b = det(p, corner_c);
                let num_c = det(corner_b, p);
                let num_a = t as i64 - num_b - num_c;
                if num_a < 0 || num_b < 0 || num_c < 0 {
                    continue;
                }
                let inv_t = 1.0 / t as f64;
                let q = add(
                    add(
                        scale(v0, num_a as f64 * inv_t),
                        scale(v1, num_b as f64 * inv_t),
                    ),
                    scale(v2, num_c as f64 * inv_t),
                );
                let Some(u) = unit(q) else { continue };
                // Already seen from another face?
                let k = key_of(u);
                let mut dup = false;
                'probe: for dx in -1..=1i64 {
                    for dy in -1..=1i64 {
                        for dz in -1..=1i64 {
                            if let Some(ids) = grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                                for &id in ids {
                                    if chord(points[id], u) < tol {
                                        dup = true;
                                        break 'probe;
                                    }
                                }
                            }
                        }
                    }
                }
                if !dup {
                    grid.entry(k).or_default().push(points.len());
                    points.push(u);
                }
            }
        }
    }

    if points.len() != expected {
        return Err(MpasError::Refusal(format!(
            "GP({m},{n}) produced {} distinct generators instead of the {expected} the subdivision defines; the seed is not the icosahedral lattice, so its Delaunay would not carry exactly twelve pentagons and the dislocation-free guarantee this seeding exists for would be a lie",
            points.len()
        )));
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::geom::arc;

    #[test]
    fn the_count_formula_reproduces_the_published_family() {
        // GP(64,0) is the published x1.40962; GP(1,0) is the dodecahedron.
        assert_eq!(goldberg_cells(64, 0), 40_962);
        assert_eq!(goldberg_cells(1, 0), 12);
        assert_eq!(goldberg_cells(128, 0), 163_842);
    }

    #[test]
    fn the_snap_lands_on_the_nearest_achievable_count() {
        // The three sizes the defect was measured at, and the count that
        // motivated the snap. Every expectation is checkable by hand from
        // 10*(m^2+mn+n^2)+2.
        for &(requested, m, n, cells) in &[
            (2_000usize, 13u32, 2u32, 1_992usize),
            (12_000, 20, 20, 12_002),
            (40_962, 64, 0, 40_962),
            (127_051, 102, 19, 127_032),
            (12, 1, 0, 12),
        ] {
            let c = snap_cells(requested, false).unwrap();
            assert_eq!(
                (c.m, c.n, c.cells),
                (m, n, cells),
                "snap({requested}) chose GP({},{}) = {}",
                c.m,
                c.n,
                c.cells
            );
        }
    }

    #[test]
    fn a_budget_snap_never_exceeds_the_request() {
        // 12,000 snaps NEAREST to 12,002 -- but a device budget is a ceiling,
        // and two cells over a card's capacity is over the card's capacity.
        let c = snap_cells(12_000, true).unwrap();
        assert!(c.cells <= 12_000, "ceiling snap chose {}", c.cells);
        assert_eq!(c.cells, 11_972, "the largest GP count under 12,000 is GP(27,12)");
        let free = snap_cells(12_000, false).unwrap();
        assert_eq!(free.cells, 12_002);
    }

    #[test]
    fn a_request_below_the_dodecahedron_is_refused_by_name() {
        let err = snap_cells(11, false).unwrap_err().to_string();
        assert!(err.contains("twelve pentagons"), "{err}");
    }

    #[test]
    fn the_seed_delivers_exactly_the_subdivision_count() {
        // One from each class: I (n=0), II (n=m), III (chiral), and the
        // gcd>1 chiral case whose shared boundary points exercise the merge.
        for &(m, n) in &[(1u32, 0u32), (4, 0), (3, 3), (13, 2), (6, 2)] {
            let pts = seed(m, n).unwrap();
            assert_eq!(
                pts.len(),
                goldberg_cells(m, n),
                "GP({m},{n}) delivered {} generators",
                pts.len()
            );
            for p in &pts {
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                assert!((r - 1.0).abs() < 1e-14, "a generator left the sphere: r = {r}");
            }
        }
    }

    #[test]
    fn the_seed_is_deterministic() {
        let a = seed(13, 2).unwrap();
        let b = seed(13, 2).unwrap();
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i], b[i], "generator {i} moved between two identical calls");
        }
    }

    #[test]
    fn the_seeds_delaunay_carries_twelve_pentagons_and_no_heptagons() {
        // The whole point of the seeding: the topology has no dislocations.
        // GP(13,2) is the snap for the 2,000-cell request the defect was
        // measured at.
        let pts = seed(13, 2).unwrap();
        let rings = crate::mesh::hull::delaunay_rings(&pts).unwrap();
        let mut hist = std::collections::BTreeMap::<usize, usize>::new();
        for i in 0..pts.len() {
            *hist.entry(rings.ring(i).len()).or_default() += 1;
        }
        assert_eq!(
            hist.get(&5).copied().unwrap_or(0),
            12,
            "coordination histogram {hist:?}"
        );
        assert_eq!(
            hist.get(&6).copied().unwrap_or(0),
            pts.len() - 12,
            "coordination histogram {hist:?}"
        );
        assert!(
            hist.keys().all(|&d| d == 5 || d == 6),
            "a dislocation exists in the raw seed: {hist:?}"
        );

        // And no near-coincident pair anywhere near the defect scale: the
        // sparsest healthy spacing is ~ sqrt(4pi/N).
        let h = (4.0 * std::f64::consts::PI / pts.len() as f64).sqrt();
        let mut min_arc = f64::INFINITY;
        for i in 0..pts.len() {
            for &j in rings.ring(i) {
                min_arc = min_arc.min(arc(pts[i], pts[j as usize]));
            }
        }
        assert!(
            min_arc > 0.5 * h,
            "nearest generators sit {min_arc:.3e} rad apart against a nominal {h:.3e}"
        );
    }
}
