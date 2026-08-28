//! Sampling an unstructured spherical source onto arbitrary target points.
//!
//! The horizontal half of a parent-native boundary transfer.  Nothing in this
//! module knows what model produced the source or what the target is for: it
//! is given cell centres, the dual triangulation that connects them, edge
//! locations with their normal angles, and it answers two questions.
//!
//! * **Cell fields** — [`SphereSampler::cell_weights`] returns the three
//!   source cells and barycentric weights whose combination is the target
//!   point.  Linear-exact: a field that is linear in the embedding coordinates
//!   is reproduced exactly, and a target sitting on a source cell centre
//!   reproduces that cell's value.
//! * **Edge-normal fields** — [`SphereSampler::edge_operator`] returns a fixed
//!   row of coefficients over nearby source edges such that
//!   `u_target = sum(coef_j * u_source_j)`.  The two-component earth-relative
//!   wind is recovered from the source's normal components by a locally
//!   weighted least-squares fit and then projected onto the target's own
//!   normal.
//!
//! Both operators are geometry-only: they are computed once per target point
//! and reused for every vertical level and every valid time, which is what
//! makes a long boundary series cheap.
//!
//! ## The coincidence snap
//! A target point that lands on a source point is a *degenerate* transfer, and
//! a transfer operator that cannot reproduce identity on a degenerate transfer
//! is wrong.  Solving the 3x3 barycentric system at a coincident point already
//! returns weights within a few units in the last place of `(1, 0, 0)`, which
//! survives the cast to `f32` in almost every case — but "almost every" is not
//! a property worth resting a proof on.  When the target is within
//! [`SNAP_RADIANS`] of a source point the operator collapses to that point,
//! exactly.  This is a property of the operator, not of any particular source:
//! coincident nodes snap whether the source is a forecast of our own or
//! anyone else's.
//!
//! [`SphereSampler::without_snap`] turns it off, so the operator's accuracy
//! without the snap can be measured rather than assumed.
//!
//! ## The skirt, and why containment is not the convex hull
//! A source's *domain* is the union of its cells.  Its dual triangulation
//! covers less than that: the triangulation stops at the outermost cell
//! centres, while the cells themselves reach half a spacing further out.  On a
//! culled regional mesh that gap is not academic — the mesh's own outermost
//! edge midpoints sit in it, so a mesh is not even contained in its own
//! triangulation.  Refusing there would refuse the degenerate transfer, which
//! is the one case that must work.
//!
//! So a target is placed in one of three ways, and the caller is told which:
//!
//! * inside a dual triangle — barycentric weights, linear-exact;
//! * outside the triangulation but within its nearest source cell's own reach
//!   — that cell's value, unweighted, which is what the source itself holds
//!   over that area.  [`CellOperator::skirt`] marks it and the receipt counts
//!   it, because a skirt point is first-order where the interior is second;
//! * further out than that — genuinely outside the source, and refused.
//!
//! A cell's reach is the circumradius of a regular hexagon with that cell's
//! largest neighbour spacing.  For the hexagons that make up almost all of a
//! spherical centroidal Voronoi mesh that is the distance to its own corners,
//! exactly; for the twelve pentagons and for a distorted cell it is generous
//! by a fraction of one cell, which is the right direction to err when the
//! alternative is refusing a boundary that is really there.

#![allow(clippy::needless_range_loop)]

use crate::error::{MpasError, MpasResult};
use crate::weights::KdTree;

/// A target within this angular distance of a source point *is* that source
/// point.  1e-9 rad is about 6 mm on Earth: far below any mesh spacing, far
/// above the rounding of a latitude/longitude round trip through `f32`.
pub const SNAP_RADIANS: f64 = 1.0e-9;

/// Barycentric weights are accepted with this much slack, so a target exactly
/// on a triangle edge belongs to one of the two triangles sharing it rather
/// than to neither.
const INSIDE_SLACK: f64 = -1.0e-9;

/// Unit vector from latitude and longitude in radians.
pub fn unit_vector(lat: f64, lon: f64) -> [f64; 3] {
    let (clat, slat) = (lat.cos(), lat.sin());
    [clat * lon.cos(), clat * lon.sin(), slat]
}

/// The horizontal operator for one target cell: three source cells and the
/// weights that sum to one.
#[derive(Debug, Clone, Copy)]
pub struct CellOperator {
    pub cells: [usize; 3],
    pub weights: [f64; 3],
    /// True when the target coincided with a source cell centre.
    pub snapped: bool,
    /// True when the target fell in the skirt beyond the dual triangulation
    /// and took its nearest source cell's value whole.
    pub skirt: bool,
}

impl CellOperator {
    /// Apply to one source field.
    #[inline]
    pub fn apply(&self, values: &[f32]) -> f32 {
        (self.weights[0] * values[self.cells[0]] as f64
            + self.weights[1] * values[self.cells[1]] as f64
            + self.weights[2] * values[self.cells[2]] as f64) as f32
    }

    /// Apply to one level of a source column store laid out `[cell][level]`.
    #[inline]
    pub fn apply_column(&self, columns: &[Vec<f32>], k: usize) -> f32 {
        (self.weights[0] * columns[self.cells[0]][k] as f64
            + self.weights[1] * columns[self.cells[1]][k] as f64
            + self.weights[2] * columns[self.cells[2]][k] as f64) as f32
    }
}

/// The edge-normal operator for one target edge: a fixed linear functional
/// over source edges.
#[derive(Debug, Clone)]
pub struct EdgeOperator {
    pub edges: Vec<usize>,
    pub coefs: Vec<f64>,
    /// True when the target edge coincided with a source edge, in position
    /// and in normal direction; then `edges` holds that one edge and `coefs`
    /// holds `+1` or `-1`.
    pub snapped: bool,
    /// Condition number of the 2x2 normal-equation matrix, so a caller can
    /// report how well posed the local fit was.  `1.0` for a snap.
    pub condition: f64,
}

impl EdgeOperator {
    #[inline]
    pub fn apply_column(&self, columns: &[Vec<f32>], k: usize) -> f32 {
        let mut acc = 0.0f64;
        for (i, &e) in self.edges.iter().enumerate() {
            acc += self.coefs[i] * columns[e][k] as f64;
        }
        acc as f32
    }
}

/// The source side of a horizontal transfer.
#[derive(Debug)]
pub struct SphereSampler {
    cell_xyz: Vec<[f64; 3]>,
    /// Zero-based cell triples from the source's dual triangulation.
    triangles: Vec<[usize; 3]>,
    /// Triangle indices incident to each source cell.
    triangles_of_cell: Vec<Vec<u32>>,
    /// Exact nearest-neighbour index over the cell centres.
    cell_tree: KdTree,
    edge_xyz: Vec<[f64; 3]>,
    edge_angle: Vec<f64>,
    /// Source edges gathered per source cell: the cell's own edges plus its
    /// neighbours', zero-based, deduplicated.
    edge_patch: Vec<Vec<usize>>,
    edge_tree: KdTree,
    /// How far each source cell reaches beyond its own centre: the
    /// circumradius of a regular hexagon at that cell's largest neighbour
    /// spacing, as a chord length.
    reach: Vec<f64>,
    snap: bool,
}

/// The circumradius of a regular hexagon whose neighbouring centres are one
/// spacing apart, as a multiple of that spacing.  A hexagon's corner is
/// `1/sqrt(3)` of the centre-to-centre distance away from its centre.
const HEX_CIRCUMRADIUS_PER_SPACING: f64 = 0.577_350_269_189_625_8;

impl SphereSampler {
    /// Build from the source mesh.
    ///
    /// `cells_on_vertex` is the source's `cellsOnVertex`, one-based and
    /// `vertex_degree` wide, exactly as the file stores it.  Triangles naming
    /// a stored zero are dropped: on a culled mesh those are the dual cells
    /// hanging off the outer boundary, and a triangle with a missing corner
    /// has no interior to interpolate over.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        cell_lat: &[f64],
        cell_lon: &[f64],
        cells_on_vertex: &[i64],
        vertex_degree: usize,
        edge_lat: &[f64],
        edge_lon: &[f64],
        edge_angle: &[f64],
        edges_on_cell: &[Vec<usize>],
        n_edges_on_cell: &[usize],
        cells_on_cell: &[Vec<usize>],
    ) -> MpasResult<SphereSampler> {
        let n_cells = cell_lat.len();
        if n_cells < 3 {
            return Err(MpasError::Refusal(format!(
                "the source mesh carries {n_cells} cell(s); a barycentric transfer needs at \
                 least the three corners of one triangle"
            )));
        }
        if vertex_degree != 3 {
            return Err(MpasError::Refusal(format!(
                "the source mesh has vertexDegree {vertex_degree}; this transfer walks the dual \
                 triangulation, and a dual cell with {vertex_degree} corners is not a triangle.  \
                 Splitting it here would invent a diagonal the mesh never had"
            )));
        }
        let cell_xyz: Vec<[f64; 3]> = (0..n_cells)
            .map(|c| unit_vector(cell_lat[c], cell_lon[c]))
            .collect();

        let n_vertices = cells_on_vertex.len() / vertex_degree;
        let mut triangles = Vec::with_capacity(n_vertices);
        for v in 0..n_vertices {
            let a = cells_on_vertex[v * 3];
            let b = cells_on_vertex[v * 3 + 1];
            let c = cells_on_vertex[v * 3 + 2];
            if a < 1 || b < 1 || c < 1 {
                continue;
            }
            let (a, b, c) = (a as usize, b as usize, c as usize);
            if a > n_cells || b > n_cells || c > n_cells {
                return Err(MpasError::Refusal(format!(
                    "cellsOnVertex names cell {} at vertex {}, outside 1..{n_cells}; the source \
                     mesh and its dual disagree about how many cells exist, and every weight \
                     computed from that dual would address the wrong column",
                    a.max(b).max(c),
                    v + 1
                )));
            }
            triangles.push([a - 1, b - 1, c - 1]);
        }
        if triangles.is_empty() {
            return Err(MpasError::Refusal(
                "the source mesh's dual triangulation is empty once triangles with a stored-zero \
                 corner are dropped; there is no interior to interpolate over"
                    .to_string(),
            ));
        }

        let mut triangles_of_cell = vec![Vec::new(); n_cells];
        for (t, tri) in triangles.iter().enumerate() {
            for &c in tri {
                triangles_of_cell[c].push(t as u32);
            }
        }

        let edge_xyz: Vec<[f64; 3]> = (0..edge_lat.len())
            .map(|e| unit_vector(edge_lat[e], edge_lon[e]))
            .collect();
        let cell_tree = KdTree::build(cell_xyz.clone());
        let edge_tree = KdTree::build(edge_xyz.clone());

        // Per source cell, the edge patch its neighbourhood offers to a
        // least-squares fit: its own edges plus each neighbour's.
        let n_edges = edge_lat.len();
        let mut edge_patch = vec![Vec::new(); n_cells];
        for c in 0..n_cells {
            let mut list: Vec<usize> = Vec::with_capacity(24);
            let push_cell = |cc: usize, list: &mut Vec<usize>| {
                for i in 0..n_edges_on_cell[cc] {
                    let e = edges_on_cell[cc][i];
                    if e >= 1 && e <= n_edges {
                        list.push(e - 1);
                    }
                }
            };
            push_cell(c, &mut list);
            for i in 0..n_edges_on_cell[c] {
                let nb = cells_on_cell[c][i];
                if nb >= 1 && nb <= n_cells {
                    push_cell(nb - 1, &mut list);
                }
            }
            list.sort_unstable();
            list.dedup();
            edge_patch[c] = list;
        }

        // How far each cell reaches past its own centre.  A cell with no
        // readable neighbour reaches nowhere, so a target can only be placed
        // on it by landing on it.
        let mut reach = vec![0.0f64; n_cells];
        for c in 0..n_cells {
            let mut widest = 0.0f64;
            for i in 0..n_edges_on_cell[c] {
                let nb = cells_on_cell[c][i];
                if nb >= 1 && nb <= n_cells {
                    widest = widest.max(chord(&cell_xyz[c], &cell_xyz[nb - 1]));
                }
            }
            reach[c] = widest * HEX_CIRCUMRADIUS_PER_SPACING;
        }

        Ok(SphereSampler {
            cell_xyz,
            triangles,
            triangles_of_cell,
            cell_tree,
            edge_xyz,
            edge_angle: edge_angle.to_vec(),
            edge_patch,
            edge_tree,
            reach,
            snap: true,
        })
    }

    /// The `want` nearest source cells to a unit vector, nearest first.
    fn near_cells(&self, t: &[f64; 3], want: usize) -> Vec<usize> {
        self.cell_tree
            .nearest_k(*t, want)
            .into_iter()
            .map(|(i, _)| i as usize)
            .collect()
    }

    /// The `want` nearest source edges to a unit vector, nearest first.
    fn near_edges(&self, t: &[f64; 3], want: usize) -> Vec<usize> {
        self.edge_tree
            .nearest_k(*t, want)
            .into_iter()
            .map(|(i, _)| i as usize)
            .collect()
    }

    /// Turn the coincidence snap off, so the operator's accuracy at a
    /// coincident point can be measured instead of asserted.
    pub fn without_snap(mut self) -> Self {
        self.snap = false;
        self
    }

    pub fn n_cells(&self) -> usize {
        self.cell_xyz.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edge_xyz.len()
    }

    /// The barycentric operator at one target point.
    ///
    /// `Err` carries how far the point missed by: the chord distance to its
    /// nearest source cell and that cell's own reach, so a caller's refusal
    /// can say by how much rather than only that it happened.
    pub fn cell_weights(&self, lat: f64, lon: f64) -> Result<CellOperator, Miss> {
        let t = unit_vector(lat, lon);
        let near = self.near_cells(&t, 12);
        let Some(&nearest) = near.first() else {
            return Err(Miss {
                distance: f64::INFINITY,
                reach: 0.0,
            });
        };
        if self.snap {
            for &c in &near {
                if chord(&t, &self.cell_xyz[c]) < SNAP_RADIANS {
                    return Ok(CellOperator {
                        cells: [c, c, c],
                        weights: [1.0, 0.0, 0.0],
                        snapped: true,
                        skirt: false,
                    });
                }
            }
        }
        for &c in &near {
            for &tri_index in &self.triangles_of_cell[c] {
                let tri = self.triangles[tri_index as usize];
                if let Some(w) = barycentric(
                    &t,
                    &self.cell_xyz[tri[0]],
                    &self.cell_xyz[tri[1]],
                    &self.cell_xyz[tri[2]],
                ) {
                    return Ok(CellOperator {
                        cells: tri,
                        weights: w,
                        snapped: false,
                        skirt: false,
                    });
                }
            }
        }
        // The skirt: past the triangulation, but still inside the source's
        // outermost cells, which do hold a value over that area.
        let distance = chord(&t, &self.cell_xyz[nearest]);
        if distance <= self.reach[nearest] {
            return Ok(CellOperator {
                cells: [nearest, nearest, nearest],
                weights: [1.0, 0.0, 0.0],
                snapped: false,
                skirt: true,
            });
        }
        Err(Miss {
            distance,
            reach: self.reach[nearest],
        })
    }

    /// The edge-normal operator at one target edge, or `None` when no source
    /// cell is near enough to offer a patch.
    ///
    /// `angle` is the target's `angleEdge`: the normal in earth-relative
    /// (eastward, northward) components is `(cos angle, sin angle)`, which is
    /// the convention `init_atm_case_lbc` itself uses when it rotates a
    /// first-guess wind onto an edge.
    pub fn edge_operator(&self, lat: f64, lon: f64, angle: f64) -> Option<EdgeOperator> {
        let t = unit_vector(lat, lon);
        let n_target = [angle.cos(), angle.sin()];

        if self.snap {
            for &e in &self.near_edges(&t, 8) {
                if chord(&t, &self.edge_xyz[e]) < SNAP_RADIANS {
                    let dot = n_target[0] * self.edge_angle[e].cos()
                        + n_target[1] * self.edge_angle[e].sin();
                    if dot.abs() > 1.0 - 1.0e-9 {
                        return Some(EdgeOperator {
                            edges: vec![e],
                            coefs: vec![dot.signum()],
                            snapped: true,
                            condition: 1.0,
                        });
                    }
                }
            }
        }

        // The patch of the nearest source cell.
        let near = self.near_cells(&t, 4);
        let host = *near.first()?;
        let patch = &self.edge_patch[host];
        if patch.len() < 2 {
            return None;
        }

        // Weighted least squares for the earth-relative vector V from the
        // source's normal components, then projected onto the target normal.
        // Weight falls off with distance; the floor keeps a coincident source
        // edge from dividing by zero when the snap is off.
        let dists: Vec<f64> = patch.iter().map(|&e| chord(&t, &self.edge_xyz[e])).collect();
        let scale = dists.iter().copied().fold(0.0f64, f64::max).max(1.0e-12);
        let floor = 0.02 * scale;
        let mut m = [[0.0f64; 2]; 2];
        let mut rows: Vec<([f64; 2], f64)> = Vec::with_capacity(patch.len());
        for (i, &e) in patch.iter().enumerate() {
            let w = 1.0 / (dists[i] * dists[i] + floor * floor);
            let n = [self.edge_angle[e].cos(), self.edge_angle[e].sin()];
            m[0][0] += w * n[0] * n[0];
            m[0][1] += w * n[0] * n[1];
            m[1][0] += w * n[1] * n[0];
            m[1][1] += w * n[1] * n[1];
            rows.push((n, w));
        }
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let trace = m[0][0] + m[1][1];
        if det <= 0.0 || trace <= 0.0 {
            return None;
        }
        // Eigenvalues of the symmetric 2x2 give the condition number directly.
        let disc = ((m[0][0] - m[1][1]).powi(2) + 4.0 * m[0][1] * m[1][0]).max(0.0).sqrt();
        let (hi, lo) = (0.5 * (trace + disc), 0.5 * (trace - disc));
        let condition = if lo > 0.0 { hi / lo } else { f64::INFINITY };
        let inv = [
            [m[1][1] / det, -m[0][1] / det],
            [-m[1][0] / det, m[0][0] / det],
        ];
        // r = n_target^T * inv, so coef_j = w_j * (r . n_j).
        let r = [
            n_target[0] * inv[0][0] + n_target[1] * inv[1][0],
            n_target[0] * inv[0][1] + n_target[1] * inv[1][1],
        ];
        let coefs: Vec<f64> = rows
            .iter()
            .map(|(n, w)| w * (r[0] * n[0] + r[1] * n[1]))
            .collect();

        Some(EdgeOperator {
            edges: patch.clone(),
            coefs,
            snapped: false,
            condition,
        })
    }
}

/// How far a target missed the source by, in chord length on the unit sphere
/// and in the nearest source cell's own reach.  Multiply by the sphere's
/// radius for metres.
#[derive(Debug, Clone, Copy)]
pub struct Miss {
    pub distance: f64,
    pub reach: f64,
}

/// Chord length between two unit vectors — monotone in angular distance and
/// free of the `acos` cancellation near zero, which is the only regime the
/// snap test cares about.
#[inline]
fn chord(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Solve `[p0 p1 p2] w = t` by Cramer's rule and accept when every weight is
/// non-negative: the target lies in the spherical triangle.  Weights are
/// normalised to sum to one, which is what makes the operator reproduce a
/// constant field exactly.
fn barycentric(t: &[f64; 3], p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]) -> Option<[f64; 3]> {
    let det = det3(p0, p1, p2);
    if det.abs() < 1.0e-18 {
        return None;
    }
    let w0 = det3(t, p1, p2) / det;
    let w1 = det3(p0, t, p2) / det;
    let w2 = det3(p0, p1, t) / det;
    if w0 < INSIDE_SLACK || w1 < INSIDE_SLACK || w2 < INSIDE_SLACK {
        return None;
    }
    let sum = w0 + w1 + w2;
    if sum.abs() < 1.0e-18 {
        return None;
    }
    Some([w0 / sum, w1 / sum, w2 / sum])
}

#[inline]
fn det3(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four cells arranged so their dual has two triangles, plus the edges
    /// that would separate them.  Small enough to reason about by hand.
    fn tiny() -> SphereSampler {
        let d = 0.02f64; // ~127 km at the equator
        let cell_lat = vec![0.0, 0.0, d, d];
        let cell_lon = vec![0.0, d, 0.0, d];
        // Two dual triangles: (1,2,3) and (2,4,3), one-based.
        let cov = vec![1, 2, 3, 2, 4, 3];
        // Four edges, midway between the pairs they separate.
        let edge_lat = vec![0.0, 0.5 * d, 0.5 * d, d];
        let edge_lon = vec![0.5 * d, 0.0, d, 0.5 * d];
        // Normal angles: 0 = eastward, pi/2 = northward.
        let half = std::f64::consts::FRAC_PI_2;
        let edge_angle = vec![0.0, half, half, 0.0];
        let eoc = vec![
            vec![1, 2, 0, 0],
            vec![1, 3, 0, 0],
            vec![2, 4, 0, 0],
            vec![3, 4, 0, 0],
        ];
        let neoc = vec![2, 2, 2, 2];
        let coc = vec![
            vec![2, 3, 0, 0],
            vec![1, 4, 0, 0],
            vec![1, 4, 0, 0],
            vec![2, 3, 0, 0],
        ];
        SphereSampler::build(
            &cell_lat, &cell_lon, &cov, 3, &edge_lat, &edge_lon, &edge_angle, &eoc, &neoc, &coc,
        )
        .unwrap()
    }

    #[test]
    fn a_target_on_a_source_cell_centre_reproduces_that_cell() {
        let s = tiny();
        let d = 0.02f64;
        let op = s.cell_weights(d, 0.0).expect("inside");
        assert!(op.snapped, "a coincident target must snap");
        assert!(!op.skirt);
        let values = [10.0f32, 20.0, 30.0, 40.0];
        assert_eq!(op.apply(&values), 30.0);
    }

    #[test]
    fn the_snap_can_be_switched_off_and_the_answer_is_still_that_cell() {
        // Without the snap the barycentric solve must still land on the
        // corner: this is what makes the snap a convenience rather than the
        // thing holding the identity proof up.
        let s = tiny().without_snap();
        let d = 0.02f64;
        let op = s.cell_weights(d, 0.0).expect("inside");
        assert!(!op.snapped);
        let values = [10.0f32, 20.0, 30.0, 40.0];
        let got = op.apply(&values);
        assert!((got - 30.0).abs() < 1.0e-4, "{got}");
    }

    #[test]
    fn weights_sum_to_one_so_a_constant_field_survives() {
        let s = tiny();
        let d = 0.02f64;
        let op = s.cell_weights(0.3 * d, 0.4 * d).expect("inside");
        let sum: f64 = op.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-12, "{sum}");
        let values = [7.5f32; 4];
        assert_eq!(op.apply(&values), 7.5);
    }

    #[test]
    fn a_target_just_past_the_triangulation_lands_in_the_skirt() {
        // Below the bottom row of cell centres: outside every dual triangle,
        // but well inside the bottom row's own cells.
        let s = tiny();
        let d = 0.02f64;
        let op = s.cell_weights(-0.2 * d, 0.1 * d).expect("inside the skirt");
        assert!(op.skirt, "a point past the triangulation is a skirt point");
        assert_eq!(op.cells, [0, 0, 0]);
        let values = [10.0f32, 20.0, 30.0, 40.0];
        assert_eq!(op.apply(&values), 10.0);
    }

    #[test]
    fn a_target_beyond_the_outermost_cells_is_refused_with_the_distance() {
        let s = tiny();
        let miss = s.cell_weights(1.0, 1.0).unwrap_err();
        assert!(miss.distance > miss.reach, "{miss:?}");
        assert!(miss.reach > 0.0, "{miss:?}");
    }

    #[test]
    fn the_skirt_stops_where_the_source_does() {
        // One cell spacing straight out from the corner is past every cell.
        let s = tiny();
        let d = 0.02f64;
        assert!(s.cell_weights(-1.5 * d, -1.5 * d).is_err());
    }

    #[test]
    fn a_coincident_edge_transfers_its_own_normal_component_exactly() {
        let s = tiny();
        let d = 0.02f64;
        let op = s
            .edge_operator(0.0, 0.5 * d, 0.0)
            .expect("a patch exists");
        assert!(op.snapped);
        assert_eq!(op.edges, vec![0]);
        assert_eq!(op.coefs, vec![1.0]);
    }

    #[test]
    fn a_reversed_coincident_edge_flips_sign() {
        let s = tiny();
        let d = 0.02f64;
        let op = s
            .edge_operator(0.0, 0.5 * d, std::f64::consts::PI)
            .expect("a patch exists");
        assert!(op.snapped);
        assert_eq!(op.coefs, vec![-1.0]);
    }

    #[test]
    fn a_uniform_wind_is_reconstructed_and_reprojected() {
        // Give every source edge the normal component of one uniform vector,
        // then ask for a target edge at a fresh angle.  A least-squares fit
        // that is worth anything returns the same vector's component.
        let s = tiny().without_snap();
        let d = 0.02f64;
        let wind = [7.0f64, -3.0f64]; // eastward, northward
        let columns: Vec<Vec<f32>> = (0..s.n_edges())
            .map(|e| {
                let a = s.edge_angle[e];
                vec![(wind[0] * a.cos() + wind[1] * a.sin()) as f32]
            })
            .collect();
        for angle in [0.0f64, 0.7, 2.4, -1.1] {
            let op = s.edge_operator(0.5 * d, 0.5 * d, angle).expect("a patch");
            let want = wind[0] * angle.cos() + wind[1] * angle.sin();
            let got = op.apply_column(&columns, 0) as f64;
            assert!(
                (got - want).abs() < 1.0e-4,
                "angle {angle}: got {got}, want {want}, condition {}",
                op.condition
            );
        }
    }

    #[test]
    fn a_dual_with_a_stored_zero_corner_drops_that_triangle_rather_than_guessing() {
        let d = 0.02f64;
        let cell_lat = vec![0.0, 0.0, d, d];
        let cell_lon = vec![0.0, d, 0.0, d];
        let cov = vec![1, 2, 3, 2, 0, 3];
        let edge_lat = vec![0.0];
        let edge_lon = vec![0.5 * d];
        let edge_angle = vec![0.0];
        let eoc = vec![vec![1], vec![1], vec![0], vec![0]];
        let neoc = vec![1, 1, 0, 0];
        let coc = vec![vec![2], vec![1], vec![0], vec![0]];
        let s = SphereSampler::build(
            &cell_lat, &cell_lon, &cov, 3, &edge_lat, &edge_lon, &edge_angle, &eoc, &neoc, &coc,
        )
        .unwrap();
        assert_eq!(s.triangles.len(), 1);
    }

    #[test]
    fn a_dual_that_is_not_triangular_is_refused_by_name() {
        let err = SphereSampler::build(
            &[0.0, 0.0, 0.1],
            &[0.0, 0.1, 0.0],
            &[1, 2, 3, 1],
            4,
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("vertexDegree 4"), "{err}");
        assert!(err.contains("invent"), "{err}");
    }
}
