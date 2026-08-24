//! Every MPAS grid-file field, derived from cell centres and their neighbour
//! rings alone.
//!
//! The input is the primal topology in CSR form -- for each cell, the ring of
//! neighbouring cells wound counter-clockwise seen from OUTSIDE the sphere --
//! plus the cell centres as unit vectors. Measured over every ring slot of both
//! published meshes with zero violations, that ring determines `edgesOnCell`,
//! `verticesOnCell`, `cellsOnEdge`, `verticesOnEdge`, `cellsOnVertex`,
//! `edgesOnVertex`, `nEdgesOnEdge`, `edgesOnEdge` and every metric exactly.
//!
//! Edges and vertices are numbered CANONICALLY: an edge is its unordered cell
//! pair `(a, b)` with `a < b`, a vertex its sorted cell triple, both sorted
//! lexicographically. A generator invents its own numbering, so grading one
//! against a published file's numbering would mark a correct generator wrong;
//! the canonical numbering is a pure function of the cell centres and removes
//! that entirely.

use crate::error::{MpasError, MpasResult};
use crate::mesh::geom::{
    self, V3, add, arc, circumcenter, cross, dot, east_north, polygon_area, sub, tangent_at,
    tri_area, unit,
};

/// The neighbour rings, CSR. `values[offsets[i]..offsets[i+1]]` is cell `i`'s
/// ring of neighbours in counter-clockwise order seen from outside the sphere.
#[derive(Debug, Clone, Default)]
pub struct Rings {
    pub offsets: Vec<u32>,
    pub values: Vec<u32>,
}

impl Rings {
    pub fn n_cells(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    #[inline]
    pub fn ring(&self, i: usize) -> &[u32] {
        &self.values[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }
    #[inline]
    pub fn degree(&self, i: usize) -> usize {
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }
}

/// A complete MPAS mesh in this crate's own 0-based numbering.
///
/// Padding slots follow the rules measured on BOTH published meshes rather than
/// a tidier invention, so a reader written against upstream MPAS sees exactly
/// what it sees there: `cellsOnCell` repeats its last valid entry,
/// `edgesOnCell` holds the maximum of its valid entries, `verticesOnCell` and
/// `edgesOnEdge` hold the 1-based zero (stored here as -1), and
/// `weightsOnEdge` padding is exactly 0.0.
#[derive(Debug, Clone)]
pub struct MpasMesh {
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    pub max_edges: usize,
    pub max_edges2: usize,
    pub vertex_degree: usize,

    pub cell_xyz: Vec<V3>,
    pub edge_xyz: Vec<V3>,
    pub vertex_xyz: Vec<V3>,

    pub mesh_density: Vec<f64>,
    pub nominal_min_dc: f64,

    pub n_edges_on_cell: Vec<i32>,
    /// `n_cells * max_edges`, 0-based, padding = repeat of last valid.
    pub cells_on_cell: Vec<i32>,
    /// `n_cells * max_edges`, 0-based, padding = max of valid.
    pub edges_on_cell: Vec<i32>,
    /// `n_cells * max_edges`, 0-based, padding = -1.
    pub vertices_on_cell: Vec<i32>,
    /// `n_edges * 2`, 0-based. Order is an orientation lock with
    /// `vertices_on_edge`; see [`MpasMesh::derive`].
    pub cells_on_edge: Vec<i32>,
    /// `n_edges * 2`, 0-based.
    pub vertices_on_edge: Vec<i32>,
    /// `n_vertices * vertex_degree`, 0-based, ascending (the canonical triple).
    pub cells_on_vertex: Vec<i32>,
    /// `n_vertices * vertex_degree`, 0-based.
    pub edges_on_vertex: Vec<i32>,
    pub n_edges_on_edge: Vec<i32>,
    /// `n_edges * max_edges2`, 0-based, padding = -1.
    pub edges_on_edge: Vec<i32>,
    /// `n_edges * max_edges2`, padding exactly 0.0.
    pub weights_on_edge: Vec<f64>,

    pub dc_edge: Vec<f64>,
    pub dv_edge: Vec<f64>,
    pub angle_edge: Vec<f64>,
    pub area_cell: Vec<f64>,
    pub area_triangle: Vec<f64>,
    /// `n_vertices * vertex_degree`; slot `i` pairs with `cells_on_vertex[v][i]`.
    pub kite_areas_on_vertex: Vec<f64>,
}

/// The published convention for `maxEdges`, carried on both released meshes.
/// A mesh whose highest coordination number exceeds it grows the dimension
/// rather than truncating a ring.
const MAX_EDGES_FLOOR: usize = 10;

impl MpasMesh {
    /// Derive every field from cell centres, densities and neighbour rings.
    ///
    /// `nominal_min_dc` is a GENERATOR STAMP in unit-sphere radians, not a
    /// measured minimum: the published variable mesh stamps 25.000 km while its
    /// finest achieved `dcEdge` is 17.893 km. It is carried, not checked.
    pub fn derive(
        cell_xyz: Vec<V3>,
        mesh_density: Vec<f64>,
        rings: &Rings,
        nominal_min_dc: f64,
    ) -> MpasResult<MpasMesh> {
        let n_cells = cell_xyz.len();
        if n_cells < 4 {
            return Err(MpasError::Refusal(format!(
                "a closed-sphere mesh needs at least 4 cells to have any Delaunay triangle at all; {n_cells} were given"
            )));
        }
        if rings.n_cells() != n_cells {
            return Err(MpasError::Refusal(format!(
                "{} neighbour rings for {n_cells} cell centres; the ring table and the centre table describe different meshes and every derived field would be indexed off the wrong one",
                rings.n_cells()
            )));
        }
        if mesh_density.len() != n_cells {
            return Err(MpasError::Refusal(format!(
                "{} meshDensity values for {n_cells} cells; MPAS scales its horizontal mixing by meshDensity, so a short table would apply one cell's diffusion length to another",
                mesh_density.len()
            )));
        }

        let n_edges_on_cell: Vec<i32> = (0..n_cells).map(|i| rings.degree(i) as i32).collect();
        let min_deg = n_edges_on_cell.iter().copied().min().unwrap_or(0);
        if min_deg < 3 {
            let which = n_edges_on_cell.iter().position(|&d| d == min_deg).unwrap_or(0);
            return Err(MpasError::Refusal(format!(
                "cell {which} has {min_deg} neighbours; a Voronoi cell with fewer than 3 has no polygon to take an area of, so areaCell and every mass budget built on it would be zero"
            )));
        }
        let max_deg = n_edges_on_cell.iter().copied().max().unwrap_or(0) as usize;
        let max_edges = max_deg.max(MAX_EDGES_FLOOR);
        let max_edges2 = 2 * max_edges;
        let vertex_degree = 3usize;

        // ---------------------------------------------------- canonical numbering
        let mut edge_keys: Vec<[u32; 2]> = Vec::with_capacity(rings.values.len());
        let mut vertex_keys: Vec<[u32; 3]> = Vec::with_capacity(rings.values.len());
        for i in 0..n_cells {
            let ring = rings.ring(i);
            let deg = ring.len();
            for j in 0..deg {
                let cur = ring[j];
                if cur as usize >= n_cells {
                    return Err(MpasError::Refusal(format!(
                        "cell {i} ring slot {j} names cell {cur}, which is outside the {n_cells}-cell mesh; every connectivity array built from this ring would index past the end of the field arrays"
                    )));
                }
                let a = (i as u32).min(cur);
                let b = (i as u32).max(cur);
                if a == b {
                    return Err(MpasError::Refusal(format!(
                        "cell {i} lists itself as its own neighbour at ring slot {j}; that edge has zero length, so dcEdge would be 0 and the tangential weight formula divides by it"
                    )));
                }
                // The ring table has to be mutual before anything is built on
                // it. Without this the canonical enumeration produces an edge
                // whose first cell does not list its second, and every lookup
                // afterwards has to guess.
                if !rings.ring(cur as usize).contains(&(i as u32)) {
                    return Err(MpasError::Refusal(format!(
                        "cell {i} lists {cur} as a neighbour but {cur} does not list {i}: the neighbour relation is not mutual, so the edge between them would carry a flux out of one cell that never arrives in the other"
                    )));
                }
                edge_keys.push([a, b]);
                let prev = ring[(j + deg - 1) % deg];
                let mut t = [i as u32, prev, cur];
                t.sort_unstable();
                vertex_keys.push(t);
            }
        }
        edge_keys.sort_unstable();
        edge_keys.dedup();
        vertex_keys.sort_unstable();
        vertex_keys.dedup();
        let n_edges = edge_keys.len();
        let n_vertices = vertex_keys.len();

        let edge_of = |a: u32, b: u32| -> Option<usize> {
            let key = [a.min(b), a.max(b)];
            edge_keys.binary_search(&key).ok()
        };
        let vertex_of = |mut t: [u32; 3]| -> Option<usize> {
            t.sort_unstable();
            vertex_keys.binary_search(&t).ok()
        };

        // ------------------------------------------------------ cell connectivity
        let mut cells_on_cell = vec![0i32; n_cells * max_edges];
        let mut edges_on_cell = vec![0i32; n_cells * max_edges];
        let mut vertices_on_cell = vec![-1i32; n_cells * max_edges];
        for i in 0..n_cells {
            let ring = rings.ring(i);
            let deg = ring.len();
            let base = i * max_edges;
            let mut max_edge_here = -1i32;
            for j in 0..deg {
                let cur = ring[j];
                let e = edge_of(i as u32, cur).expect("every ring pair was enumerated");
                let prev = ring[(j + deg - 1) % deg];
                let v = vertex_of([i as u32, prev, cur]).expect("every ring triple was enumerated");
                cells_on_cell[base + j] = cur as i32;
                edges_on_cell[base + j] = e as i32;
                vertices_on_cell[base + j] = v as i32;
                max_edge_here = max_edge_here.max(e as i32);
            }
            // Padding, reproducing the published rules exactly.
            let last_neighbour = cells_on_cell[base + deg - 1];
            for j in deg..max_edges {
                cells_on_cell[base + j] = last_neighbour;
                edges_on_cell[base + j] = max_edge_here;
                vertices_on_cell[base + j] = -1;
            }
        }

        // ------------------------------------------------------ edge connectivity
        //
        // ORIENTATION LOCK. `cellsOnEdge` order and `verticesOnEdge` order are
        // coupled: with n the tangent-projected normal c0 -> c1 and t the
        // tangent-projected v0 -> v1, (n x t) . r_edge must be positive on every
        // edge. Flip either one alone and the mesh still validates clean while
        // every tangential wind reconstruction runs backwards. This code fixes
        // c0 < c1 and then DERIVES the vertex order from the ring winding, so
        // the lock holds by construction rather than by a later check -- and the
        // check is still run, in `mesh::validate`, because nothing downstream
        // of this crate looks at it.
        let mut cells_on_edge = vec![0i32; n_edges * 2];
        let mut vertices_on_edge = vec![0i32; n_edges * 2];
        for (e, key) in edge_keys.iter().enumerate() {
            let c0 = key[0] as usize;
            let c1 = key[1];
            let ring = rings.ring(c0);
            let deg = ring.len();
            let j = ring
                .iter()
                .position(|&x| x == c1)
                .expect("the canonical edge came from this ring");
            // Walking counter-clockwise around c0, the vertex formed with the
            // PREVIOUS neighbour comes before the one formed with the NEXT, and
            // the tangential direction v0 -> v1 is the normal rotated +90
            // degrees, which is what the lock demands.
            let v0 = vertices_on_cell[c0 * max_edges + j];
            let v1 = vertices_on_cell[c0 * max_edges + (j + 1) % deg];
            cells_on_edge[e * 2] = c0 as i32;
            cells_on_edge[e * 2 + 1] = c1 as i32;
            vertices_on_edge[e * 2] = v0;
            vertices_on_edge[e * 2 + 1] = v1;
        }

        // ---------------------------------------------------- vertex connectivity
        let mut cells_on_vertex = vec![0i32; n_vertices * vertex_degree];
        let mut edges_on_vertex = vec![0i32; n_vertices * vertex_degree];
        for (v, t) in vertex_keys.iter().enumerate() {
            for s in 0..3 {
                cells_on_vertex[v * 3 + s] = t[s] as i32;
            }
            // The three edges of the triple, in the order (ab, ac, bc). MPAS
            // derives its sign convention at run time from verticesOnEdge, so
            // this ordering is free; it is fixed here only so a mesh is
            // reproducible byte-for-byte between runs.
            let pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])];
            for (s, (a, b)) in pairs.iter().enumerate() {
                edges_on_vertex[v * 3 + s] = match edge_of(*a, *b) {
                    Some(e) => e as i32,
                    None => {
                        return Err(MpasError::Refusal(format!(
                            "vertex {v} joins cells {a}, {b} and {} but {a} and {b} share no edge; the dual mesh is not the dual of this primal mesh and every kite area would be taken over an open figure",
                            t[2]
                        )));
                    }
                };
            }
        }

        // ------------------------------------------------------------- geometry
        let mut vertex_xyz = vec![[0.0f64; 3]; n_vertices];
        for (v, t) in vertex_keys.iter().enumerate() {
            let a = cell_xyz[t[0] as usize];
            let b = cell_xyz[t[1] as usize];
            let c = cell_xyz[t[2] as usize];
            vertex_xyz[v] = circumcenter(a, b, c).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "cells {}, {} and {} are collinear on a great circle, so their Voronoi vertex is undefined; the dual edge through it would have no position and dvEdge no length",
                    t[0], t[1], t[2]
                ))
            })?;
        }

        let mut edge_xyz = vec![[0.0f64; 3]; n_edges];
        let mut dc_edge = vec![0.0f64; n_edges];
        let mut dv_edge = vec![0.0f64; n_edges];
        let mut angle_edge = vec![0.0f64; n_edges];
        for e in 0..n_edges {
            let c0 = cells_on_edge[e * 2] as usize;
            let c1 = cells_on_edge[e * 2 + 1] as usize;
            let p0 = cell_xyz[c0];
            let p1 = cell_xyz[c1];
            // MEASURED: xyzEdge is the great-circle midpoint of the two CELL
            // CENTRES, not of the two vertices -- decided by 10 to 11 orders of
            // magnitude on both published meshes.
            let m = unit(add(p0, p1)).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "cells {c0} and {c1} are antipodal, so the edge between them has no midpoint; latEdge and lonEdge would be arbitrary and the wind rotation angle built on them meaningless"
                ))
            })?;
            edge_xyz[e] = m;
            dc_edge[e] = arc(p0, p1);
            let v0 = vertices_on_edge[e * 2] as usize;
            let v1 = vertices_on_edge[e * 2 + 1] as usize;
            dv_edge[e] = arc(vertex_xyz[v0], vertex_xyz[v1]);
            angle_edge[e] = edge_angle(m, p0, p1).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "edge {e} sits exactly on a pole, where the local east and north directions are undefined; angleEdge is the init-time wind rotation angle, so a wrong value there rotates the initial wind by an arbitrary amount at that edge"
                ))
            })?;
        }

        // ------------------------------------------------ areas and kite partition
        // MEASURED on both published files: `areaCell` is the SUM OF ITS OWN
        // KITES and `areaTriangle` is the sum of the vertex's three kites, both
        // bit-exact (max relative gap 0.000e+00). Taking the kites as the
        // primitive and summing them makes the partition exact by construction
        // rather than exact to within whatever two independent triangulations of
        // the same cell happen to agree on -- which was measured at 7.9e-11, far
        // too coarse for a mass budget that has to close over a whole forecast.
        // The independent spherical-polygon area is still computed, and
        // `mesh::validate` refuses if the two decompositions disagree.
        let mut area_cell = vec![0.0f64; n_cells];
        let mut kite_areas_on_vertex = vec![0.0f64; n_vertices * vertex_degree];
        let mut kite_buf: Vec<f64> = Vec::with_capacity(max_edges);
        for i in 0..n_cells {
            let deg = rings.degree(i);
            let base = i * max_edges;
            // Kite of cell i at vertex slot j: the quadrilateral
            // (vertex, edge_before, cell centre, edge_after), as two spherical
            // triangles. Slot i of kiteAreasOnVertex pairs with
            // cellsOnVertex[v][i], and cellsOnVertex is the ascending triple.
            let centre = cell_xyz[i];
            kite_buf.clear();
            for j in 0..deg {
                let v = vertices_on_cell[base + j] as usize;
                let e_before = edges_on_cell[base + (j + deg - 1) % deg] as usize;
                let e_after = edges_on_cell[base + j] as usize;
                let pv = vertex_xyz[v];
                // The two halves share the (vertex, cell centre) diagonal and
                // are consistently wound, so the SUM is signed and its
                // magnitude is the kite area even when the quad is not convex.
                let kite = (tri_area(pv, edge_xyz[e_before], centre)
                    + tri_area(pv, centre, edge_xyz[e_after]))
                .abs();
                let slot = (0..3)
                    .find(|&s| cells_on_vertex[v * 3 + s] == i as i32)
                    .expect("the vertex triple contains this cell by construction");
                kite_areas_on_vertex[v * 3 + slot] = kite;
                kite_buf.push(kite);
            }
            area_cell[i] = nsum(&kite_buf);
        }
        let mut area_triangle = vec![0.0f64; n_vertices];
        for v in 0..n_vertices {
            // MEASURED: areaTriangle is the SUM OF ITS OWN THREE KITES, not the
            // spherical excess of the three cell centres. On the published
            // variable mesh those two differ by up to 3.53e-03 relative, and it
            // is not an obtuse-triangle effect.
            area_triangle[v] = kite_areas_on_vertex[v * 3]
                + kite_areas_on_vertex[v * 3 + 1]
                + kite_areas_on_vertex[v * 3 + 2];
        }

        // -------------------------------------------- tangential stencil + weights
        let mut n_edges_on_edge = vec![0i32; n_edges];
        let mut edges_on_edge = vec![-1i32; n_edges * max_edges2];
        let mut weights_on_edge = vec![0.0f64; n_edges * max_edges2];
        for e in 0..n_edges {
            let c0 = cells_on_edge[e * 2] as usize;
            let c1 = cells_on_edge[e * 2 + 1] as usize;
            let n0 = rings.degree(c0);
            let n1 = rings.degree(c1);
            n_edges_on_edge[e] = (n0 + n1) as i32 - 2;
            let mut slot = 0usize;
            for &c in &[c0, c1] {
                let n = rings.degree(c);
                let base = c * max_edges;
                let k = (0..n)
                    .find(|&t| edges_on_cell[base + t] == e as i32)
                    .expect("the edge is on both of its cells");
                let sign_c = |x: usize| -> f64 {
                    if cells_on_edge[x * 2] == c as i32 {
                        1.0
                    } else {
                        -1.0
                    }
                };
                let sign_e = sign_c(e);
                // Running-kite closed form for the TRiSK/Thuburn tangential
                // reconstruction, solved to 1 ULP against both published
                // meshes. The kite is taken at VERTEX INDEX i, not i-1: the
                // off-by-one gives a 6.0e-02 error, which is large enough to
                // ruin a forecast and small enough to look like a mesh
                // difference. The l/d factors are folded into the stored
                // weight, so w = R * dvEdge[e'] / dcEdge[e].
                let mut run = 0.0f64;
                for i in 1..n {
                    let er = edges_on_cell[base + (k + i) % n] as usize;
                    let vr = vertices_on_cell[base + (k + i) % n] as usize;
                    let kslot = (0..3)
                        .find(|&s| cells_on_vertex[vr * 3 + s] == c as i32)
                        .expect("the ring vertex belongs to this cell");
                    run += kite_areas_on_vertex[vr * 3 + kslot];
                    let w =
                        -sign_e * sign_c(er) * (run / area_cell[c] - 0.5) * dv_edge[er] / dc_edge[e];
                    edges_on_edge[e * max_edges2 + slot] = er as i32;
                    weights_on_edge[e * max_edges2 + slot] = w;
                    slot += 1;
                }
            }
            debug_assert_eq!(slot, n_edges_on_edge[e] as usize);
        }

        Ok(MpasMesh {
            n_cells,
            n_edges,
            n_vertices,
            max_edges,
            max_edges2,
            vertex_degree,
            cell_xyz,
            edge_xyz,
            vertex_xyz,
            mesh_density,
            nominal_min_dc,
            n_edges_on_cell,
            cells_on_cell,
            edges_on_cell,
            vertices_on_cell,
            cells_on_edge,
            vertices_on_edge,
            cells_on_vertex,
            edges_on_vertex,
            n_edges_on_edge,
            edges_on_edge,
            weights_on_edge,
            dc_edge,
            dv_edge,
            angle_edge,
            area_cell,
            area_triangle,
            kite_areas_on_vertex,
        })
    }

    /// The spherical-polygon area of every cell, taken independently of the
    /// kite decomposition that produced `area_cell`. Two triangulations of the
    /// same region: if they disagree, the cell's vertex ring is not the
    /// boundary of the region its kites tile, and the mesh is not a Voronoi
    /// tessellation of its own generators.
    pub fn polygon_areas(&self) -> Vec<f64> {
        let mut out = vec![0.0f64; self.n_cells];
        let mut ring: Vec<V3> = Vec::with_capacity(self.max_edges);
        for i in 0..self.n_cells {
            let deg = self.n_edges_on_cell[i] as usize;
            ring.clear();
            for j in 0..deg {
                ring.push(self.vertex_xyz[self.vertices_on_cell[i * self.max_edges + j] as usize]);
            }
            out[i] = polygon_area(&ring);
        }
        out
    }

    /// Hexagon across-flats spacing of every cell, in metres.
    ///
    /// `sqrt(2 * areaCell / sqrt(3)) * R` -- NOT `dcEdge` and NOT
    /// `sqrt(areaCell)`. This is the metric this program's published resolution
    /// figures are quoted in; a receipt that used a different one would silently
    /// disagree with every number already on the record.
    pub fn spacing_m(&self) -> Vec<f64> {
        self.area_cell
            .iter()
            .map(|&a| (2.0 * a / 3f64.sqrt()).sqrt() * geom::EARTH_RADIUS_M)
            .collect()
    }
}

/// `angleEdge`: the angle of the edge normal, measured counter-clockwise from
/// local east at the edge point.
///
/// `atan2(n.north, n.east)`, with `n` the tangent-projected direction from
/// `cellsOnEdge[e][0]` to `cellsOnEdge[e][1]` at the edge location. Three
/// competing forms were ruled out by measurement -- all three sat about pi/2
/// away on average. MPAS uses this angle to rotate the initial wind into the
/// edge frame, and the port's own validator only checks `|angleEdge| <= pi`, so
/// a wrong value here is invisible downstream and rotates every initial wind.
fn edge_angle(at: V3, from: V3, to: V3) -> Option<f64> {
    let (east, north) = east_north(at)?;
    let n = tangent_at(at, sub(to, from));
    let n = unit(n)?;
    Some(dot(n, north).atan2(dot(n, east)))
}

/// Orientation of one edge: `(n x t) . r_edge`, which the mesh requires to be
/// positive on every edge. Exposed so the validator and the tests measure the
/// same quantity as the constructor.
pub fn edge_orientation(mesh: &MpasMesh, e: usize) -> f64 {
    let c0 = mesh.cells_on_edge[e * 2] as usize;
    let c1 = mesh.cells_on_edge[e * 2 + 1] as usize;
    let v0 = mesh.vertices_on_edge[e * 2] as usize;
    let v1 = mesh.vertices_on_edge[e * 2 + 1] as usize;
    let r = mesh.edge_xyz[e];
    let n = tangent_at(r, sub(mesh.cell_xyz[c1], mesh.cell_xyz[c0]));
    let t = tangent_at(r, sub(mesh.vertex_xyz[v1], mesh.vertex_xyz[v0]));
    let (n, t) = match (unit(n), unit(t)) {
        (Some(n), Some(t)) => (n, t),
        _ => return 0.0,
    };
    dot(cross(n, t), r)
}

/// Signed cosine of the angle between an edge's primal and dual great circles.
/// Zero on a perfectly orthogonal (SCVT) mesh; the published meshes read
/// 5.34e-13 and 5.07e-13.
pub fn edge_nonorthogonality(mesh: &MpasMesh, e: usize) -> f64 {
    let c0 = mesh.cells_on_edge[e * 2] as usize;
    let c1 = mesh.cells_on_edge[e * 2 + 1] as usize;
    let v0 = mesh.vertices_on_edge[e * 2] as usize;
    let v1 = mesh.vertices_on_edge[e * 2 + 1] as usize;
    let r = mesh.edge_xyz[e];
    let n = tangent_at(r, sub(mesh.cell_xyz[c1], mesh.cell_xyz[c0]));
    let t = tangent_at(r, sub(mesh.vertex_xyz[v1], mesh.vertex_xyz[v0]));
    match (unit(n), unit(t)) {
        (Some(n), Some(t)) => dot(n, t),
        _ => 1.0,
    }
}

/// Neumaier compensated sum. Plain accumulation over 160,000 cell areas loses
/// enough low bits to blur the `4*pi` sphere-closure check the emit gate rests
/// on, so every global sum in this module goes through here.
pub fn nsum(values: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for &v in values {
        let t = sum + v;
        if sum.abs() >= v.abs() {
            comp += (sum - t) + v;
        } else {
            comp += (v - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// Chord distance, re-exported so callers measuring point agreement do not
/// reach for an angle and hit `acos`'s 2.1e-8 rad floor.
pub use geom::chord as point_gap;
