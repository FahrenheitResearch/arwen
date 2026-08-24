//! The gate a generated mesh has to pass before a single byte is written.
//!
//! Every check here names the concrete breakage it prevents. Four of them exist
//! because nothing downstream looks at them: the port's own validator was run
//! against the published mesh and against deliberately corrupted copies of it,
//! and it silently PASSES a swapped `cellsOnEdge`, a swapped `verticesOnEdge`,
//! a wrong `angleEdge` value and a wrong `weightsOnEdge` value. Those four
//! produce a mesh that validates clean everywhere else and integrates wrong, so
//! they are checked here or they are checked nowhere.

use std::collections::VecDeque;

use serde::Serialize;

use crate::error::{MpasError, MpasResult};
use crate::mesh::derive::{MpasMesh, edge_nonorthogonality, edge_orientation, nsum};
use crate::mesh::geom::{V3, cross, dot, sub};

/// Thresholds, every one anchored to a number measured on the two published
/// MPAS meshes rather than chosen by taste. `x1` and `x4` below name those
/// meshes' readings.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Limits {
    /// `|sum(areaCell)/4pi - 1|`. Published: 0.0 on both, exactly.
    pub sphere_closure: f64,
    /// `|kite row sum / areaTriangle - 1|`. Published: 0.0 on both, exactly.
    pub kite_partition: f64,
    /// `|polygon fan area / kite sum area - 1|`, the two independent
    /// triangulations of the same Voronoi cell.
    pub area_decomposition: f64,
    /// `max |cos(primal, dual)|`, the SCVT orthogonality defect.
    /// Published: 5.07e-13 (x1), 5.34e-13 (x4).
    pub orthogonality: f64,
    /// `max |R[e,e'] + R[e',e]|` for the TRiSK weights, absolute, against a
    /// weight scale bounded by 1/2. Published median: 1.67e-16.
    pub weight_antisymmetry: f64,
    /// Shortest dual edge, in METRES at earth radius. Below this the mesh
    /// cannot survive the FP32 static that has to accompany it: an FP32
    /// vertex at earth-radius magnitude is quantised at ~0.5 m ULP, the port
    /// recomputes `dvEdge` from those stored vertices and compares at
    /// `rtol = 2e-5` with `atol = 0.0`, and the measured survival boundary is
    /// ~7.5 km -- so the pair is refused whole at load, after the static was
    /// built. Published x1.40962: 45,016.7 m. Fibonacci-seeded generated
    /// meshes measured 7,337.6 m at 2,000 cells, 75.0 m at 12,000, 7.2 m at
    /// 40,962, all of which this floor turns from a silent PASS into a
    /// refusal with the numbers.
    pub min_dv_edge_m: f64,
    /// Smallest `dvEdge / dcEdge`. A near-zero ratio is the signature of a
    /// near-cocircular pentagon-heptagon dislocation quad: two Voronoi
    /// vertices nearly coincide while their cells stay a full spacing apart.
    /// The TRiSK tangential weights divide by `dvEdge`, so the wind term
    /// across such an edge is amplified by `dc/dv`. Published: 0.3945 on
    /// x1.40962, 0.0336 on x4.163842; the measured defect class on
    /// Fibonacci-seeded uniform meshes sits at 0.0147 and below.
    pub min_dv_over_dc: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            // Loose enough for a compensated sum over 200,000 areas of 1e-5
            // each, tight enough that a single dropped or double-counted cell
            // (which moves the sum by ~1e-5) is a hard failure.
            sphere_closure: 1e-12,
            kite_partition: 1e-13,
            // Two independent triangulations of the same cell, both through the
            // half-tangent form. Measured on the published meshes at 6.6e-12
            // worst; a decade of headroom above that.
            area_decomposition: 1e-10,
            // Three decades above the published readings, so a mesh emitted
            // here has to be as orthogonal as a converged NCAR SCVT.
            orthogonality: 1e-10,
            weight_antisymmetry: 1e-9,
            // The FP32 survival boundary, measured through the port's own
            // load check (rtol 2e-5, atol 0.0 against ~0.5 m ULP vertices).
            min_dv_edge_m: 7_500.0,
            // Between the published family's floor (0.0336 on x4.163842) and
            // the measured dislocation class (0.0147 and below), anchored to
            // both readings rather than chosen by taste.
            min_dv_over_dc: 0.02,
        }
    }
}

/// The numbers a validation run measured, whether or not it passed. These go
/// into the receipt so a mesh carries its own evidence.
#[derive(Debug, Clone, Serialize)]
pub struct MeshReport {
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    pub euler_characteristic: i64,
    pub coordination_defect: i64,
    pub coordination_histogram: Vec<(i32, usize)>,
    pub sum_area_cell_over_4pi: f64,
    pub sum_area_triangle_over_4pi: f64,
    pub sum_kite_areas_over_4pi: f64,
    pub max_kite_partition_rel: f64,
    pub max_area_decomposition_rel: f64,
    pub max_nonorthogonality: f64,
    pub min_edge_orientation: f64,
    pub max_weight_antisymmetry: f64,
    pub weight_pairs_checked: usize,
    pub nonzero_weight_padding_slots: usize,
    pub min_dc_edge_m: f64,
    pub max_dc_edge_m: f64,
    pub min_dv_edge_m: f64,
    pub min_dv_over_dc: f64,
    pub max_dv_over_dc: f64,
    pub median_dv_over_dc: f64,
    pub min_spacing_m: f64,
    pub max_spacing_m: f64,
    pub spacing_ratio: f64,
    pub max_adjacent_spacing_ratio: f64,
    pub limits: Limits,
}

/// Validate a derived mesh. Returns the measured report on success; on failure
/// the refusal carries the numbers that made it say no.
pub fn validate(mesh: &MpasMesh, limits: Limits) -> MpasResult<MeshReport> {
    let (nc, ne, nv) = (mesh.n_cells, mesh.n_edges, mesh.n_vertices);

    // ------------------------------------------------------------- topology
    let euler = nc as i64 - ne as i64 + nv as i64;
    if euler != 2 {
        return Err(MpasError::Refusal(format!(
            "nCells - nEdges + nVertices = {euler}, not 2: this is not a closed sphere. The dycore validates the Euler characteristic at start-up and every global mass and vorticity budget assumes a surface with no boundary and no handle; {nc} cells, {ne} edges, {nv} vertices"
        )));
    }
    if nv != 2 * nc - 4 || ne != 3 * nc - 6 {
        return Err(MpasError::Refusal(format!(
            "a closed triangulated sphere on {nc} cells has exactly {} vertices and {} edges; this mesh has {nv} and {ne}, so at least one Voronoi cell is not a simple polygon and its area integral is taken over the wrong figure",
            2 * nc - 4,
            3 * nc - 6
        )));
    }
    let deg_sum: i64 = mesh.n_edges_on_cell.iter().map(|&d| d as i64).sum();
    if deg_sum != 2 * ne as i64 {
        return Err(MpasError::Refusal(format!(
            "sum(nEdgesOnCell) = {deg_sum}, but every edge is on exactly two cells so it must be 2*nEdges = {}. An edge counted once is an edge whose flux is added to one cell and never subtracted from the other, which leaks mass every step",
            2 * ne
        )));
    }
    let defect: i64 = mesh.n_edges_on_cell.iter().map(|&d| 6 - d as i64).sum();
    if defect != 12 {
        return Err(MpasError::Refusal(format!(
            "sum(6 - nEdgesOnCell) = {defect}, not 12: the total coordination defect of any triangulated sphere is 12 exactly. A different value means the neighbour rings do not describe a sphere at all"
        )));
    }
    if mesh.vertex_degree * nv != 2 * ne {
        return Err(MpasError::Refusal(format!(
            "vertexDegree*nVertices = {} but 2*nEdges = {}; the dual mesh is not a triangulation and areaTriangle would be taken over figures that are not triangles",
            mesh.vertex_degree * nv,
            2 * ne
        )));
    }

    // --------------------------------------------------- reciprocity of the ring
    for i in 0..nc {
        let deg = mesh.n_edges_on_cell[i] as usize;
        for j in 0..deg {
            let nb = mesh.cells_on_cell[i * mesh.max_edges + j];
            if nb < 0 || nb as usize >= nc {
                return Err(MpasError::Refusal(format!(
                    "cell {i} slot {j} names neighbour {nb}, outside 0..{nc}; a flux would be gathered from unallocated memory"
                )));
            }
            let nb = nb as usize;
            let back = (0..mesh.n_edges_on_cell[nb] as usize)
                .any(|k| mesh.cells_on_cell[nb * mesh.max_edges + k] == i as i32);
            if !back {
                return Err(MpasError::Refusal(format!(
                    "cell {i} lists {nb} as a neighbour but {nb} does not list {i}: the neighbour relation is not mutual, so the edge between them carries a flux out of one cell that never arrives in the other"
                )));
            }
            let e = mesh.edges_on_cell[i * mesh.max_edges + j];
            if e < 0 || e as usize >= ne {
                return Err(MpasError::Refusal(format!(
                    "cell {i} slot {j} names edge {e}, outside 0..{ne}"
                )));
            }
            let e = e as usize;
            let (c0, c1) = (mesh.cells_on_edge[e * 2], mesh.cells_on_edge[e * 2 + 1]);
            let pair_ok = (c0 == i as i32 && c1 == nb as i32) || (c1 == i as i32 && c0 == nb as i32);
            if !pair_ok {
                return Err(MpasError::Refusal(format!(
                    "edgesOnCell[{i}][{j}] = {e} separates cells {c0} and {c1}, not {i} and {nb}: the edge-to-cell map and the cell-to-cell ring disagree, so a normal wind would be applied across the wrong pair of cells"
                )));
            }
        }
    }

    // ------------------------------------------------------------- ring winding
    for i in 0..nc {
        let deg = mesh.n_edges_on_cell[i] as usize;
        let base = i * mesh.max_edges;
        let mut acc: V3 = [0.0, 0.0, 0.0];
        for j in 0..deg {
            let a = mesh.vertex_xyz[mesh.vertices_on_cell[base + j] as usize];
            let b = mesh.vertex_xyz[mesh.vertices_on_cell[base + (j + 1) % deg] as usize];
            let c = cross(a, b);
            acc = [acc[0] + c[0], acc[1] + c[1], acc[2] + c[2]];
        }
        let signed = dot(acc, mesh.cell_xyz[i]);
        if !(signed > 0.0) {
            return Err(MpasError::Refusal(format!(
                "cell {i}'s vertex ring winds clockwise seen from outside (signed area {signed:.3e}). Every sign convention in the TRiSK operators is read off that winding, so a reversed cell reverses its vorticity and its tangential flux while its area still comes out positive"
            )));
        }
    }

    // ------------------------------------------------- the four silent surfaces
    //
    // 1. Edge orientation. `cellsOnEdge` order and `verticesOnEdge` order are a
    //    coupled pair; the port's validator accepts either one swapped.
    let mut min_orient = f64::INFINITY;
    let mut worst_orient_edge = 0usize;
    for e in 0..ne {
        let o = edge_orientation(mesh, e);
        if o < min_orient {
            min_orient = o;
            worst_orient_edge = e;
        }
    }
    if !(min_orient > 0.0) {
        return Err(MpasError::Refusal(format!(
            "edge {worst_orient_edge} has (n x t).r = {min_orient:.3e}, not positive: its cellsOnEdge order and verticesOnEdge order are not the locked right-handed pair. A flipped edge validates clean everywhere else and reverses the tangential wind reconstruction at that edge for the whole run"
        )));
    }

    // 2. Orthogonality. The mimetic operators are second-order only while the
    //    primal and dual arcs cross at a right angle.
    let mut max_nonorth = 0.0f64;
    let mut worst_nonorth_edge = 0usize;
    for e in 0..ne {
        let c = edge_nonorthogonality(mesh, e).abs();
        if c > max_nonorth {
            max_nonorth = c;
            worst_nonorth_edge = e;
        }
    }
    if max_nonorth > limits.orthogonality {
        return Err(MpasError::Refusal(format!(
            "edge {worst_nonorth_edge}: the primal and dual great circles cross at {:.6} degrees off perpendicular (|cos| = {max_nonorth:.3e}, limit {:.3e}). The published meshes read 5.1e-13 and 5.3e-13. A non-orthogonal mesh is not a centroidal Voronoi tessellation; the divergence and gradient operators lose their second-order cancellation between neighbours and the leftover first-order term appears as grid-scale noise wherever the mesh is worst",
            (max_nonorth.asin()).to_degrees(),
            limits.orthogonality
        )));
    }

    // 3. Weight values, via the Thuburn antisymmetry that makes the scheme
    //    energy-conserving. Nothing downstream checks a weight VALUE at all.
    let mut max_anti = 0.0f64;
    let mut pairs = 0usize;
    let mut worst_pair = (0usize, 0usize);
    for e in 0..ne {
        let n = mesh.n_edges_on_edge[e] as usize;
        for k in 0..n {
            let ep = mesh.edges_on_edge[e * mesh.max_edges2 + k];
            if ep < 0 {
                return Err(MpasError::Refusal(format!(
                    "edge {e} stencil slot {k} is empty inside its declared length {n}; MPAS sums the whole declared stencil, so an empty slot is a term silently dropped from the tangential wind"
                )));
            }
            let ep = ep as usize;
            let r_ee = mesh.weights_on_edge[e * mesh.max_edges2 + k] * mesh.dc_edge[e]
                / mesh.dv_edge[ep];
            // find e in e''s stencil
            let np = mesh.n_edges_on_edge[ep] as usize;
            let back = (0..np).find(|&t| mesh.edges_on_edge[ep * mesh.max_edges2 + t] == e as i32);
            let Some(t) = back else {
                return Err(MpasError::Refusal(format!(
                    "edge {e} reconstructs from edge {ep}, but {ep} does not reconstruct from {e}: the tangential stencil is not symmetric, so the Thuburn energy identity has no pair to cancel against and the scheme injects energy every step"
                )));
            };
            let r_ep = mesh.weights_on_edge[ep * mesh.max_edges2 + t] * mesh.dc_edge[ep]
                / mesh.dv_edge[e];
            let residual = (r_ee + r_ep).abs();
            if residual > max_anti {
                max_anti = residual;
                worst_pair = (e, ep);
            }
            pairs += 1;
        }
    }
    if max_anti > limits.weight_antisymmetry {
        return Err(MpasError::Refusal(format!(
            "the TRiSK weights are not antisymmetric: R[{},{}] + R[{},{}] = {max_anti:.3e}, limit {:.3e} over {pairs} ordered pairs. The published meshes read 1.7e-16 at the median. Antisymmetry is what makes the tangential reconstruction energy-conserving; without it the scheme adds or removes kinetic energy at every edge, and nothing downstream of this crate ever looks at a weight value",
            worst_pair.0, worst_pair.1, worst_pair.1, worst_pair.0, limits.weight_antisymmetry
        )));
    }

    // 4. Weight padding. MPAS sums the whole row, so a nonzero padding slot is a
    //    wind term added out of nowhere.
    let mut bad_padding = 0usize;
    for e in 0..ne {
        let n = mesh.n_edges_on_edge[e] as usize;
        for k in n..mesh.max_edges2 {
            if mesh.weights_on_edge[e * mesh.max_edges2 + k] != 0.0 {
                bad_padding += 1;
            }
        }
    }
    if bad_padding > 0 {
        return Err(MpasError::Refusal(format!(
            "{bad_padding} weightsOnEdge padding slots are not exactly 0.0; MPAS sums the full maxEdges2 row, so each one adds an unowned term to a tangential wind. The published meshes have 0"
        )));
    }

    // ----------------------------------------------------------- sphere closure
    let s_cell = nsum(&mesh.area_cell);
    let s_tri = nsum(&mesh.area_triangle);
    let s_kite = nsum(&mesh.kite_areas_on_vertex);
    let four_pi = 4.0 * std::f64::consts::PI;
    for (label, sum) in [
        ("areaCell", s_cell),
        ("areaTriangle", s_tri),
        ("kiteAreasOnVertex", s_kite),
    ] {
        let rel = (sum / four_pi - 1.0).abs();
        if !(rel <= limits.sphere_closure) {
            return Err(MpasError::Refusal(format!(
                "sum({label}) / 4*pi = {:.16}, off by {rel:.3e} (limit {:.3e}). The cells do not tile the sphere: some area is counted twice or not at all, and every global mass and energy budget carries that error for the whole run",
                sum / four_pi,
                limits.sphere_closure
            )));
        }
    }

    // ------------------------------------------------------- the kite partition
    let mut max_kite_rel = 0.0f64;
    for v in 0..nv {
        let row = &mesh.kite_areas_on_vertex[v * 3..v * 3 + 3];
        let s = row[0] + row[1] + row[2];
        let rel = if mesh.area_triangle[v] > 0.0 {
            (s / mesh.area_triangle[v] - 1.0).abs()
        } else {
            1.0
        };
        max_kite_rel = max_kite_rel.max(rel);
    }
    let mut per_cell = vec![0.0f64; nc];
    for v in 0..nv {
        for s in 0..3 {
            let c = mesh.cells_on_vertex[v * 3 + s] as usize;
            per_cell[c] += mesh.kite_areas_on_vertex[v * 3 + s];
        }
    }
    for c in 0..nc {
        let rel = (per_cell[c] / mesh.area_cell[c] - 1.0).abs();
        max_kite_rel = max_kite_rel.max(rel);
    }
    if max_kite_rel > limits.kite_partition {
        return Err(MpasError::Refusal(format!(
            "the kite areas do not partition the mesh: worst relative gap {max_kite_rel:.3e}, limit {:.3e}. kiteAreasOnVertex is the area weight MPAS uses to move vorticity between the dual and primal grids, so a partition that does not close moves circulation that does not exist",
            limits.kite_partition
        )));
    }

    // The independent second opinion on every cell's area: the spherical
    // polygon bounded by its vertex ring, against the kites that tile it.
    let polygon = mesh.polygon_areas();
    let mut max_decomp = 0.0f64;
    let mut worst_decomp_cell = 0usize;
    for c in 0..nc {
        let r = (polygon[c] / mesh.area_cell[c] - 1.0).abs();
        if r > max_decomp {
            max_decomp = r;
            worst_decomp_cell = c;
        }
    }
    if max_decomp > limits.area_decomposition {
        return Err(MpasError::Refusal(format!(
            "cell {worst_decomp_cell}: the spherical polygon bounded by its vertex ring and the kites that tile it disagree by {max_decomp:.3e} (limit {:.3e}). Those are two triangulations of the same region, so a disagreement means the vertex ring is not the boundary of the cell -- the mesh is not the Voronoi tessellation of its own generators, and areaCell is not the area any flux is divided by",
            limits.area_decomposition
        )));
    }

    // ---------------------------------------------------------- positive metrics
    for (label, values) in [
        ("dcEdge", &mesh.dc_edge),
        ("dvEdge", &mesh.dv_edge),
        ("areaCell", &mesh.area_cell),
        ("areaTriangle", &mesh.area_triangle),
        ("kiteAreasOnVertex", &mesh.kite_areas_on_vertex),
    ] {
        if let Some(i) = values.iter().position(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(MpasError::Refusal(format!(
                "{label}[{i}] = {}; every length and area on the mesh must be finite and strictly positive. dcEdge divides the tangential weight formula and sets the global time step through max(u+c)/min(dcEdge), so a zero there stops the model and a negative one runs it backwards",
                values[i]
            )));
        }
    }
    if let Some(i) = mesh.angle_edge.iter().position(|a| !a.is_finite()) {
        return Err(MpasError::Refusal(format!(
            "angleEdge[{i}] is not finite; it is the init-time wind rotation angle, so every initial wind at that edge would be rotated by a non-number"
        )));
    }
    if let Some(i) = mesh
        .angle_edge
        .iter()
        .position(|a| a.abs() > std::f64::consts::PI + 1e-12)
    {
        return Err(MpasError::Refusal(format!(
            "angleEdge[{i}] = {} rad, outside [-pi, pi]; the wind rotation would wrap",
            mesh.angle_edge[i]
        )));
    }

    // -------------------------------------------------------------- connectivity
    let mut seen = vec![false; nc];
    let mut queue = VecDeque::new();
    queue.push_back(0usize);
    seen[0] = true;
    let mut reached = 1usize;
    while let Some(i) = queue.pop_front() {
        let deg = mesh.n_edges_on_cell[i] as usize;
        for j in 0..deg {
            let nb = mesh.cells_on_cell[i * mesh.max_edges + j] as usize;
            if !seen[nb] {
                seen[nb] = true;
                reached += 1;
                queue.push_back(nb);
            }
        }
    }
    if reached != nc {
        return Err(MpasError::Refusal(format!(
            "the cell graph splits: {reached} of {nc} cells are reachable from cell 0. A disconnected component cannot exchange mass with the rest of the mesh and the halo exchange would deadlock or silently drop it"
        )));
    }

    // ------------------------------------------------------------ shape metrics
    let mut hist: std::collections::BTreeMap<i32, usize> = Default::default();
    for &d in &mesh.n_edges_on_cell {
        *hist.entry(d).or_default() += 1;
    }
    let r = crate::mesh::geom::EARTH_RADIUS_M;
    let dc_m: Vec<f64> = mesh.dc_edge.iter().map(|&d| d * r).collect();

    // ------------------------------------------- the FP32 survival floor
    //
    // Near-coincident cell corners are an EQUILIBRIUM feature of a
    // polycrystalline (Fibonacci-seeded) SCVT -- measured at 7,337.6 m
    // shortest dual edge at 2,000 cells, 75.0 m at 12,000, 7.2 m at 40,962,
    // and identical under --sweeps 200/600/2000, so more relaxation is not a
    // remedy. Until this gate they PASSED here and failed two stages later,
    // after the static build, at port load.
    let (mut min_dv_m, mut worst_dv_edge) = (f64::INFINITY, 0usize);
    let (mut min_ratio, mut worst_ratio_edge) = (f64::INFINITY, 0usize);
    for e in 0..ne {
        let dv_m = mesh.dv_edge[e] * r;
        if dv_m < min_dv_m {
            min_dv_m = dv_m;
            worst_dv_edge = e;
        }
        let ratio = mesh.dv_edge[e] / mesh.dc_edge[e];
        if ratio < min_ratio {
            min_ratio = ratio;
            worst_ratio_edge = e;
        }
    }
    if min_dv_m < limits.min_dv_edge_m {
        return Err(MpasError::Refusal(format!(
            "edge {worst_dv_edge}: the two dual vertices sit {min_dv_m:.1} m apart (dcEdge {:.0} m, dv/dc {:.2e}), under the {:.0} m FP32 survival floor. This mesh can never be run: the static file that has to accompany it stores vertices as f32 at earth-radius magnitude (~0.5 m ULP), the port recomputes dvEdge from those stored vertices and compares at rtol 2e-5 with atol 0.0, and below ~7.5 km the quantisation alone exceeds the tolerance -- the port refuses the whole grid/static pair at load with 'dvEdge disagrees with spherical vertex arc length', two stages after this gate ran. The published x1.40962 measures 45,016.7 m at its shortest. On a uniform request this does not occur (the icosahedral seed has no dislocations); on a variable-resolution mesh it is a pentagon-heptagon dislocation, and more relaxation re-rolls it rather than draining it (--sweeps 200, 600 and 2000 all measured the same 75.04 m shortest edge). What works, measured: fewer cells, or a different refinement layout",
            dc_m[worst_dv_edge],
            mesh.dv_edge[worst_dv_edge] / mesh.dc_edge[worst_dv_edge],
            limits.min_dv_edge_m
        )));
    }
    if min_ratio < limits.min_dv_over_dc {
        return Err(MpasError::Refusal(format!(
            "edge {worst_ratio_edge}: dvEdge/dcEdge = {min_ratio:.3e} ({:.1} m over {:.0} m), under the {:.0e} floor. Two Voronoi vertices nearly coincide while their cells stay a full spacing apart -- the signature of a near-cocircular pentagon-heptagon dislocation quad. The TRiSK tangential weights divide by dvEdge, so the wind term reconstructed across this edge is amplified by dc/dv = {:.0}; the published family never carries this (x1.40962 reads 0.3945 at its floor, x4.163842 reads 0.0336), and the measured defect class on Fibonacci-seeded uniform meshes sits at 0.0147 and below",
            mesh.dv_edge[worst_ratio_edge] * r,
            dc_m[worst_ratio_edge],
            limits.min_dv_over_dc,
            mesh.dc_edge[worst_ratio_edge] / mesh.dv_edge[worst_ratio_edge]
        )));
    }

    let mut ratios: Vec<f64> = (0..ne).map(|e| mesh.dv_edge[e] / mesh.dc_edge[e]).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let spacing = mesh.spacing_m();
    let mut max_adj = 1.0f64;
    for i in 0..nc {
        let deg = mesh.n_edges_on_cell[i] as usize;
        for j in 0..deg {
            let nb = mesh.cells_on_cell[i * mesh.max_edges + j] as usize;
            let a = spacing[i];
            let b = spacing[nb];
            let ratio = if a > b { a / b } else { b / a };
            max_adj = max_adj.max(ratio);
        }
    }
    let min_sp = spacing.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_sp = spacing.iter().cloned().fold(0.0f64, f64::max);

    Ok(MeshReport {
        n_cells: nc,
        n_edges: ne,
        n_vertices: nv,
        euler_characteristic: euler,
        coordination_defect: defect,
        coordination_histogram: hist.into_iter().collect(),
        sum_area_cell_over_4pi: s_cell / four_pi,
        sum_area_triangle_over_4pi: s_tri / four_pi,
        sum_kite_areas_over_4pi: s_kite / four_pi,
        max_kite_partition_rel: max_kite_rel,
        max_area_decomposition_rel: max_decomp,
        max_nonorthogonality: max_nonorth,
        min_edge_orientation: min_orient,
        max_weight_antisymmetry: max_anti,
        weight_pairs_checked: pairs,
        nonzero_weight_padding_slots: bad_padding,
        min_dc_edge_m: dc_m.iter().cloned().fold(f64::INFINITY, f64::min),
        max_dc_edge_m: dc_m.iter().cloned().fold(0.0f64, f64::max),
        min_dv_edge_m: min_dv_m,
        min_dv_over_dc: ratios[0],
        max_dv_over_dc: ratios[ne - 1],
        median_dv_over_dc: ratios[ne / 2],
        min_spacing_m: min_sp,
        max_spacing_m: max_sp,
        spacing_ratio: max_sp / min_sp,
        max_adjacent_spacing_ratio: max_adj,
        limits,
    })
}

/// Difference vector between two unit vectors, exposed for callers that want to
/// measure a displacement without pulling in the whole geometry module.
pub fn displacement(a: V3, b: V3) -> V3 {
    sub(a, b)
}

#[cfg(test)]
mod fp32_floor_tests {
    use super::*;
    use crate::mesh::density::MeshSpec;
    use crate::mesh::derive::MpasMesh;
    use crate::mesh::emit::nominal_min_dc_from_m;
    use crate::mesh::geom::EARTH_RADIUS_M;
    use crate::mesh::hull::delaunay_rings;
    use crate::mesh::lloyd::{LloydOptions, relax, seed_points};

    fn derive_relaxed(points: Vec<crate::mesh::geom::V3>, spec: &MeshSpec) -> MpasMesh {
        let mut pts = points;
        let out = relax(&mut pts, spec, &LloydOptions::default())
            .unwrap_or_else(|e| panic!("relaxation refused: {e}"));
        let density: Vec<f64> = pts.iter().map(|&p| spec.density(p)).collect();
        let nominal = nominal_min_dc_from_m(spec.finest_km() * 1000.0);
        MpasMesh::derive(pts, density, &out.rings, nominal).expect("derive")
    }

    /// The refusing direction, on the real pipeline: the 2,000-cell
    /// Fibonacci-seeded uniform request whose converged mesh measured a
    /// 7,337.6 m shortest dual edge -- a PASS before this gate existed, a
    /// refusal that names the port-load breakage after it.
    #[test]
    fn a_fibonacci_seeded_symptomatic_mesh_is_refused_with_the_numbers() {
        let spec = MeshSpec::uniform(120.0);
        let points = seed_points(&spec, 2_000).expect("seed");
        let mesh = derive_relaxed(points, &spec);

        let min_dv_m = mesh
            .dv_edge
            .iter()
            .map(|&d| d * EARTH_RADIUS_M)
            .fold(f64::INFINITY, f64::min);
        let min_ratio = (0..mesh.n_edges)
            .map(|e| mesh.dv_edge[e] / mesh.dc_edge[e])
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "fibonacci 2,000: min dvEdge {min_dv_m:.1} m, min dv/dc {min_ratio:.4e}"
        );

        let err = validate(&mesh, Limits::default())
            .expect_err(
                "the converged Fibonacci mesh carries a near-degenerate corner \
                 and validate PASSED it; the port would refuse the grid/static \
                 pair two stages later",
            )
            .to_string();
        assert!(
            err.contains("FP32 survival floor") || err.contains("dislocation"),
            "the refusal does not name the breakage: {err}"
        );
        assert!(
            err.contains("dvEdge disagrees with spherical vertex arc length")
                || err.contains("dc/dv"),
            "the refusal does not carry the consumer's own words or the amplification: {err}"
        );
    }

    /// The admitting direction: the SAME request routed through the
    /// icosahedral seed (its snap is GP(13,2) = 1,992 cells) relaxes to a
    /// mesh in the published mesh's class and the gate passes it.
    #[test]
    fn a_goldberg_seeded_uniform_mesh_admits() {
        let spec = MeshSpec::uniform(120.0);
        let choice = crate::mesh::icosa::snap_cells(2_000, false).expect("snap");
        let points = crate::mesh::icosa::seed(choice.m, choice.n).expect("seed");
        let mesh = derive_relaxed(points, &spec);

        let report = validate(&mesh, Limits::default())
            .unwrap_or_else(|e| panic!("the emit gate refused an icosahedrally seeded uniform mesh: {e}"));
        eprintln!(
            "goldberg GP({},{}) {} cells: min dvEdge {:.1} m, min dv/dc {:.4}, coordination {:?}",
            choice.m,
            choice.n,
            report.n_cells,
            report.min_dv_edge_m,
            report.min_dv_over_dc,
            report.coordination_histogram
        );
        // The published mesh's class: order 0.3+ at the floor, twelve
        // pentagons, no heptagons, nothing anywhere near the FP32 boundary.
        assert!(
            report.min_dv_over_dc > 0.3,
            "min dv/dc {:.4} is not in the published mesh's class",
            report.min_dv_over_dc
        );
        assert_eq!(
            report.coordination_histogram,
            vec![(5, 12), (6, report.n_cells - 12)],
            "the relaxed topology is not the subdivided icosahedron"
        );
        // Well clear of the floor, not merely over it: no dual edge below a
        // quarter of the nominal spacing.
        assert!(
            report.min_dv_edge_m > 0.25 * report.min_dc_edge_m,
            "min dvEdge {:.1} m against min dcEdge {:.1} m",
            report.min_dv_edge_m,
            report.min_dc_edge_m
        );
    }

    /// A mesh that is exactly at the boundary refuses below and admits above:
    /// the floor itself is data (`Limits`), so a consumer with an FP64 static
    /// CAN name a lower one -- but the default is the FP32 truth.
    #[test]
    fn the_floor_is_the_default_and_a_caller_can_name_a_different_consumer() {
        let spec = MeshSpec::uniform(120.0);
        let points = seed_points(&spec, 2_000).expect("seed");
        let mesh = derive_relaxed(points, &spec);
        assert!(validate(&mesh, Limits::default()).is_err());
        let relaxed_limits = Limits {
            min_dv_edge_m: 0.0,
            min_dv_over_dc: 0.0,
            ..Limits::default()
        };
        let report = validate(&mesh, relaxed_limits)
            .expect("with the floors named away the geometric checks still pass");
        eprintln!(
            "fibonacci 2,000 under named-away floors: min dvEdge {:.1} m, min dv/dc {:.4e}",
            report.min_dv_edge_m, report.min_dv_over_dc
        );
        assert!(report.min_dv_edge_m < 7_500.0);
    }
}
