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
    /// `|polygon fan area - kite sum area|` for one Voronoi cell, an ABSOLUTE
    /// area on the unit sphere, in units of `f64::EPSILON`.
    ///
    /// ABSOLUTE, and in epsilons, because that is what the quantity turns out
    /// to be. It was a RELATIVE tolerance until 2026-08-26 and had been
    /// re-anchored once already (1e-10 -> 1e-9) when a graded mesh sat on it;
    /// the five-level swath ladder then read 1.023e-9 and sat on it again.
    /// What the measurement found is that neither move was about mesh quality:
    ///
    /// * The two decompositions agree ANALYTICALLY. Every ring vertex is a
    ///   circumcentre of three generators and so is equidistant from the two
    ///   that own each edge; every edge point is the midpoint of those same
    ///   two. Vertex, edge point and next vertex therefore lie on ONE great
    ///   circle -- the bisector -- so inserting the edge point into the
    ///   boundary cannot change the area. In exact arithmetic the gap is zero
    ///   for any consistently wound ring.
    /// * The generators are stored as `f64` unit vectors and are unit only to
    ///   about one ulp. `circumcenter` takes the plane through three of them,
    ///   and a radial error `eps` on a generator tilts that plane by `eps/h`
    ///   over a lever arm `h`: the Voronoi vertex is displaced TANGENTIALLY by
    ///   about `eps/h`. The sliver that opens between the ring and the edge
    ///   point is then `dvEdge/2 * eps/h`, and with `dvEdge` proportional to
    ///   `h` the two cancel: the ABSOLUTE area gap is about `eps/2` per edge
    ///   regardless of how big the cell is.
    /// * MEASURED, on the real generator, over 123,423 cells of one graded
    ///   mesh spanning 4.5 km to 79.5 km: the median absolute gap is
    ///   0.459 to 0.495 eps in every one of twelve spacing bins, and the worst
    ///   is 2.50 eps, while the relative reading falls 208x across the same
    ///   cells, as `h^-1.94`. Recomputed at 60 decimal digits on the same
    ///   stored points, both sums are within 2e-20 of their exact values and
    ///   the exact gap is the f64 gap to four figures; rebuild the ring from
    ///   the cell centres at 60 digits and the gap is 1e-55. The disagreement
    ///   is in the POINTS, not in the area arithmetic.
    ///
    /// So the old relative form measured `eps / area(smallest cell)`. It was a
    /// reading of how fine the mesh was and of nothing else, which is why it
    /// had to be re-anchored every time the generator reached finer -- and why
    /// a cap and a polygon at the same depth read 1.007e-9 and 1.023e-9, 1.6%
    /// apart. Comparing the gap to the noise it is made of removes the free
    /// parameter: see `Limits::default` for the anchor and the margin.
    pub area_decomposition_ulps: f64,
    /// `max |cos(primal, dual)|`, the SCVT orthogonality defect.
    /// Published: 5.07e-13 (x1), 5.34e-13 (x4).
    pub orthogonality: f64,
    /// `max |R[e,e'] + R[e',e]|` for the TRiSK weights, absolute, against a
    /// weight scale bounded by 1/2. Published median: 1.67e-16.
    pub weight_antisymmetry: f64,
    /// Shortest dual edge, in METRES at earth radius. DERIVED, never chosen:
    /// [`DV_EDGE_FLOOR_QUANTA`] times the coordinate quantum of
    /// [`Limits::storage`]. See [`Limits::for_storage`] for the measurement it
    /// rests on and for what it reads in each representation.
    ///
    /// GUARDS: the ORTHOGONALITY OF THE STORED POINT SET. Rounding a vertex to
    /// the storage quantum tilts the dual edge off the primal edge it is
    /// supposed to be perpendicular to, by `1.935 * q / dvEdge` at the worst
    /// edge -- MEASURED on both published statics' own bytes. Every operator
    /// the dycore builds from those points (the edge normal, the tangent
    /// plane, the reconstruction coefficients the init stage derives) assumes
    /// that perpendicularity, and the defect grows without bound as the dual
    /// edge shrinks toward the quantum.
    ///
    /// WHAT IT DOES NOT GUARD, corrected 2026-08-29: the sentence this comment
    /// carried until then -- "the TRiSK weights would divide by it" -- names a
    /// breakage that does not occur. The stored `weightsOnEdge` is
    /// `R * dvEdge[e'] / dcEdge[e]`, with dvEdge a NUMERATOR, and the port
    /// reads that array rather than rebuilding it. The only `1/dvEdge` in the
    /// shipped port is the momentum-mixing Laplacian, limited to `4/dcEdge` in
    /// all three arms exactly as MPAS-A limits it, and the v841 PV tangential
    /// gradient, whose danger is a small `dv/dc` and which
    /// [`Limits::min_dv_over_dc`] already gates. Measured, on real cells from
    /// 115 m to 1,600 m of dual edge, the TRiSK coefficient error is FLAT at
    /// 4.1e-5 to 4.8e-5: it tracks cell spacing, not dvEdge. Evidence:
    /// gpuwm `evidence/dvedge-floor-20260829/`.
    pub min_dv_edge_m: f64,
    /// The representation the mesh this run emits will be STORED in. It is
    /// what [`Limits::min_dv_edge_m`] is derived from, and it is recorded here
    /// so a receipt carries the assumption its floor was written against
    /// rather than a bare number whose units a reader has to guess.
    pub storage: crate::staticfile::coordframe::CoordinateRepresentation,
    /// Smallest `dvEdge / dcEdge`. A near-zero ratio is the signature of a
    /// near-cocircular pentagon-heptagon dislocation quad: two Voronoi
    /// vertices nearly coincide while their cells stay a full spacing apart.
    /// The TRiSK tangential weights divide by `dvEdge`, so the wind term
    /// across such an edge is amplified by `dc/dv`. Published: 0.3945 on
    /// x1.40962, 0.0336 on x4.163842; the measured defect class on
    /// Fibonacci-seeded uniform meshes sits at 0.0147 and below.
    ///
    /// This is also the PORT'S OWN admission gate: `DualEdgePolicy` refuses
    /// `dvEdge/dcEdge < 0.02` with `DualEdgeAdmissionError` before any CUDA
    /// allocation, so the two floors must not drift apart.  It is not the
    /// only dual-edge gate: `min_dv_edge_m` sits beside it as a bound on the
    /// STORED length.  What that bound no longer rests on is the port's
    /// retired `rtol 2e-5 / atol 0.0` FP32 storage comparison -- the live
    /// storage gate is a 1.73 m atol that generated pairs clear by
    /// construction, and the old 7,500 m anchor wrongly refused the
    /// published x4.163842 (dv/dc 0.0336, dual edge 1,170.0 m) which the
    /// live port ADMITS.  RE-ANCHORED to 200 m rather than retired, by
    /// ruling 2026-08-25: stale-guard audit finding 3 measured the retired
    /// bound's irrelevance, not the noise fraction's.
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
            // The noise floor of the pair, not a tolerance on the mesh. The
            // two decompositions are analytically identical (see the field
            // doc); what separates them in f64 is that a generator is unit
            // only to an ulp, which tilts each circumcentre's plane by
            // eps/h and opens a dvEdge/2 * eps/h sliver at every edge -- an
            // ABSOLUTE area gap of about eps/2 per edge, the same for a
            // 4 km cell and a 79 km one.
            //
            // MEASURED, real generator, ladder depths 1 to 5 and finest
            // spacings 32.0 km down to 4.5 km, 123k-138k cells each. The
            // spacings come in PAIRS that straddle a ladder-level boundary,
            // so depth moves across a pair while the mesh barely does:
            //
            //   depth  finest      median      worst   retired relative
            //     1   32.017 km   0.461 eps   2.349    1.778e-11   \ pair
            //     2   28.153 km   0.468 eps   2.193    1.955e-11   /
            //     2   17.778 km   0.467 eps   2.383    6.735e-11   \ pair
            //     3   17.286 km   0.464 eps   2.469    6.980e-11   /
            //     3    9.011 km   0.464 eps   2.518    2.540e-10   \ pair
            //     4    8.912 km   0.462 eps   2.289    2.803e-10   /
            //     4    4.534 km   0.462 eps   2.383    9.771e-10   \ pair
            //     5    4.513 km   0.463 eps   2.488    9.761e-10   /
            //     5    4.517 km   0.462 eps   2.502    1.023e-9    (swath s01)
            //
            // A whole extra ladder level moves the reading 0.1%-10%; going from
            // 32 km to 4.5 km moves it 55x. The absolute gap is flat to 1.5% on
            // the median and 15% on the worst across all nine. The published
            // x1.40962 and x4.163842 sit in the same band (1.22 and 1.41 eps).
            //
            // 32 eps is 12.7x the worst measured and 6.4x the mechanism's own
            // ceiling (half of maxEdges, at dvEdge = dcEdge, is 5 eps). The
            // breakage it catches -- a vertex ring that is not the boundary
            // of the region the kites tile -- is measured in
            // `area_decomposition_tests`, six decades above this floor.
            area_decomposition_ulps: 32.0,
            // Three decades above the published readings, so a mesh emitted
            // here has to be as orthogonal as a converged NCAR SCVT.
            orthogonality: 1e-10,
            weight_antisymmetry: 1e-9,
            // DERIVED from the storage representation; see
            // `Limits::for_storage`. At the binary32 default this is 200.0 m
            // to the bit -- 400 quanta of 0.5 m -- so the 2026-08-25 ruling's
            // number is unchanged for every file that has a native
            // counterpart.
            min_dv_edge_m: DV_EDGE_FLOOR_QUANTA
                * crate::staticfile::coordframe::CoordinateRepresentation::default()
                    .quantum_m(crate::mesh::geom::EARTH_RADIUS_M),
            storage: crate::staticfile::coordframe::CoordinateRepresentation::default(),
            // Between the published family's floor (0.0336 on x4.163842) and
            // the measured dislocation class (0.0147 and below), anchored to
            // both readings rather than chosen by taste -- and equal to the
            // port's own DualEdgePolicy admission floor, so what this gate
            // emits is what the port admits.
            min_dv_over_dc: 0.02,
        }
    }
}

/// The dual-edge floor, in COORDINATE QUANTA of the representation the mesh
/// will be stored in.
///
/// WHY A MULTIPLE OF THE QUANTUM AND NOT A LENGTH. The breakage the floor
/// prevents is the orthogonality defect that storage rounding puts into the
/// point set: `abs(cos(primal, dual)) = 1.935 * q / dvEdge` at the worst edge,
/// MEASURED 2026-08-29 off the published `x1.40962.static.nc` and
/// `x4.163842.static.nc` themselves, over 614,400 edges, with the constant
/// flat across six dual-edge bins from 1 km to 131 km and across a 16x range
/// of quantum (`evidence/dvedge-floor-20260829/`). The defect is a function of
/// `q/dvEdge` and of nothing else, so a floor stated in metres is a floor
/// stated in the wrong units: it has to be re-anchored by hand every time the
/// storage changes, which is how the 7,500 m anchor came to refuse the
/// published x4.163842 that the port runs.
///
/// WHERE 400 COMES FROM, and it is not a new ruling. The floor in force since
/// 2026-08-25 is 200.0 m at binary32 Earth-centred storage, whose quantum is
/// 0.5 m exactly: 400 quanta. Held constant, that is a WORST-EDGE
/// stored-point orthogonality budget of `1.935/400 = 4.84e-3` (0.28 degrees
/// off perpendicular). This constant re-expresses the standing ruling in the
/// units its mechanism has; it does not relax it, and at binary32 it computes
/// 200.0 m to the bit.
///
/// WHAT IT READS, PER REPRESENTATION, at `sphere_radius = 6 371 229 m`:
///
/// * `binary32_earth_centred`, q = 0.5 m: **200.0 m** -- unchanged, and this
///   is what every file with a native MPAS-A counterpart is judged by.
/// * `binary64_earth_centred`, q = 9.313e-10 m: **3.725e-7 m**. At that
///   storage the quantum stops being what a fine mesh runs into at all, and
///   what remains between the generator and a sub-kilometre mesh is
///   [`Limits::min_dv_over_dc`] -- which is about mesh SHAPE, is equal to the
///   port's own admission floor, and moves for no representation.
///
/// LIMIT, stated because the number depends on it: whether a 4.84e-3
/// orthogonality defect degrades a FORECAST is NOT MEASURED. 4.84e-3 is what
/// the standing ruling encodes, carried forward unchanged, not a value
/// derived from forecast skill. Tightening it is a ruling, not an edit; for
/// reference a 1e-3 budget would put the binary32 floor at 968 m, i.e. today's
/// 200 m is nearly five times LOOSER than that, and the published
/// x4.163842.static.nc's own worst reading is 1.247e-4.
pub const DV_EDGE_FLOOR_QUANTA: f64 = 400.0;

impl Limits {
    /// The limits for a mesh that will be STORED in `storage`.
    ///
    /// Only [`Limits::min_dv_edge_m`] depends on it, and it depends on nothing
    /// else. Every other limit here is about the mesh rather than about the
    /// file, and none of them moves.
    pub fn for_storage(
        storage: crate::staticfile::coordframe::CoordinateRepresentation,
    ) -> Self {
        Limits {
            min_dv_edge_m: DV_EDGE_FLOOR_QUANTA
                * storage.quantum_m(crate::mesh::geom::EARTH_RADIUS_M),
            storage,
            ..Limits::default()
        }
    }

    /// The largest area, on the unit sphere, that the two decompositions of one
    /// cell may enclose differently. An ABSOLUTE area, because the disagreement
    /// is an absolute constant; see [`Limits::area_decomposition_ulps`].
    pub fn area_decomposition_floor(&self) -> f64 {
        self.area_decomposition_ulps * f64::EPSILON
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
    /// The worst RELATIVE disagreement between the two decompositions. Kept
    /// because every archived receipt carries it, but it is not what the gate
    /// reads any more: it is the absolute gap below divided by a cell area, so
    /// it tracks the finest cell in the mesh and not the mesh's quality.
    pub max_area_decomposition_rel: f64,
    /// The worst ABSOLUTE disagreement between the two decompositions, as an
    /// area on the unit sphere. This is what the gate reads.
    pub max_area_decomposition_abs: f64,
    /// The same reading in machine epsilons -- the units it is naturally in.
    /// Healthy meshes measure 2.3 to 2.6 across every depth and spacing tried;
    /// the floor is `Limits::area_decomposition_ulps`.
    pub max_area_decomposition_ulps: f64,
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
    //
    // Compared ABSOLUTELY, against the arithmetic's own noise. The two are
    // analytically the same number -- vertex, edge point and next vertex share
    // the bisector great circle by construction -- so the whole of a healthy
    // reading is floating point, and floating point puts a CONSTANT area there
    // (eps/2 per edge), not a constant fraction of the cell. Dividing that
    // constant by the cell area, as this check did until 2026-08-26, produces a
    // number that measures the finest cell in the mesh; it had to be moved
    // every time the generator reached finer, and it had been moved once
    // already. See `Limits::area_decomposition_ulps`.
    let polygon = mesh.polygon_areas();
    let noise_floor = limits.area_decomposition_ulps * f64::EPSILON;
    let mut max_gap = 0.0f64;
    let mut worst_gap_cell = 0usize;
    let mut max_decomp = 0.0f64;
    for c in 0..nc {
        let gap = (polygon[c] - mesh.area_cell[c]).abs();
        if gap > max_gap {
            max_gap = gap;
            worst_gap_cell = c;
        }
        max_decomp = max_decomp.max(gap / mesh.area_cell[c]);
    }
    if max_gap > noise_floor {
        return Err(MpasError::Refusal(format!(
            "cell {worst_gap_cell}: the spherical polygon bounded by its vertex ring and the kites that tile it enclose areas {max_gap:.3e} apart on the unit sphere ({:.1} machine epsilon; the floor is {:.1}, {noise_floor:.3e}), which is {:.3e} of that cell's own area. Those are two triangulations of the SAME region and they agree exactly in exact arithmetic, so a gap above the arithmetic's own noise means the vertex ring is not the boundary of the region the kites tile: verticesOnCell, edgesOnCell and the kite quads no longer describe one polygon, and areaCell is not the area any flux is divided by",
            max_gap / f64::EPSILON,
            limits.area_decomposition_ulps,
            max_gap / mesh.area_cell[worst_gap_cell]
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

    // ------------------------------------------------ the coordination floor
    //
    // THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26; gpuwm-hex
    // tree/evidence/graded-blowup-20260826/, node-2 RTX 5090). Registered
    // mesh `v16.66.195630` was emitted through this very function carrying one
    // 4-coordinated cell -- 195615, at 33.74N 117.65W, put there by
    // `surgery`'s S2 insertion -- and every check above passed it, because the
    // same operation makes two heptagons and `sum(6 - nEdgesOnCell)` still
    // reads 12 exactly. That mesh bound, allocated, integrated 22 steps and
    // died at step 23 of 36: the model's theta maximum and its |w| maximum sat
    // on that one cell at the top model level in every arm, ~197 K above its
    // initial value at 1,800 s, and the anomaly was TIMESTEP CONVERGED across
    // a 5x range (197.4 K at dt 100 s, 197.7 K at 75 s, 181.3 K at 20 s) --
    // so a finer timestep bought 2 h 45 m of model time instead of 38 minutes
    // and never bought a forecast. A 224,210-cell sibling from the same
    // generator carries the same defect at cell 224206.
    //
    // The floor is ONE-SIDED by measurement, and deliberately so: the
    // 8-coordinated cell in that same mesh (168727) is nowhere near the top
    // forty by growth in any arm, and `v20.80.151649` completes six forecast
    // hours with 1,017 heptagons. Refusing high coordination would be a gate
    // with no breakage behind it.
    //
    // The producer-side fix is `surgery::MIN_COORDINATION` and the
    // coordination clause in `surgery::drain`, which is what stops such a
    // mesh being MADE. This is the backstop that stops one being WRITTEN, so
    // a regression in the operators cannot ship silently the way this one did.
    if let Some(bad) = mesh
        .n_edges_on_cell
        .iter()
        .position(|&d| (d as usize) < crate::mesh::surgery::MIN_COORDINATION)
    {
        let below = mesh
            .n_edges_on_cell
            .iter()
            .filter(|&&d| (d as usize) < crate::mesh::surgery::MIN_COORDINATION)
            .count();
        let (lat, lon) = crate::mesh::geom::lat_lon(mesh.cell_xyz[bad]);
        return Err(MpasError::Refusal(format!(
            "cell {bad} has {} edges, under the {}-edge floor, and {below} cell(s) in this mesh do. A Goldberg polyhedron carries pentagons, hexagons and heptagons; a quadrilateral is not a cell of this family, and nothing above this line catches one -- the operation that makes a quadrilateral makes two heptagons with it, so the Euler characteristic, the degree sum and the total coordination defect of 12 all still hold exactly. MEASURED COST OF EMITTING ONE (2026-08-26): the single 4-coordinated cell in v16.66.195630 was that mesh's worst-shaped cell (perimeter/sqrt(area) 4.028 against a hexagonal median of 3.725) and its stiffest by the discrete-Laplacian row sum (1.17x a regular hexagon, rank 0 of 195,630); it grew a ~197 K standing potential-temperature error at the model top that a five-fold smaller timestep did not remove, and the forecast ended in a vertical-velocity runaway at that same cell. The offender here is at lat/lon ({:.3}, {:.3}) deg. Coordination histogram: {:?}. This is a generator defect, not a spec defect: regenerate with an engine whose surgery carries the coordination clause",
            mesh.n_edges_on_cell[bad],
            crate::mesh::surgery::MIN_COORDINATION,
            lat.to_degrees(),
            lon.to_degrees(),
            hist
        )));
    }

    let r = crate::mesh::geom::EARTH_RADIUS_M;
    let dc_m: Vec<f64> = mesh.dc_edge.iter().map(|&d| d * r).collect();

    // ------------------------- the two dual-edge gates: length, then ratio
    //
    // GUARD CITATION (ruling 2026-08-25, evidence gpuwm-hex
    // tree/evidence/graded-goldberg-20260825/): the port's load check
    // tolerates an ABSOLUTE ~1.732 m between a stored dvEdge and the arc of
    // its stored f32 vertices, so a dual length under a couple hundred
    // metres is materially quantisation noise by the time the pair loads.
    // The old 7,500 m anchor guarded the RETIRED rtol 2e-5 / atol 0.0 check
    // and was measured refusing meshes the port loads and runs -- the
    // published x4.163842 at its own 1,170.0 m included -- so the LENGTH
    // floor is re-anchored to the noise fraction, not deleted with the
    // premise that first sized it (ruling 2026-08-25; stale-guard audit
    // 2026-08-25 finding 3 measured the retired bound's irrelevance, and a
    // 654,432-cell pair with 12,732 edges past it, worst absolute 0.634 m,
    // that the live port accepts whole).
    //
    // The RATIO gate beside it is the port's live admission contract:
    // DualEdgePolicy refuses dvEdge/dcEdge < 0.02 with
    // DualEdgeAdmissionError before any CUDA allocation. Near-coincident
    // cell corners are an EQUILIBRIUM feature of a polycrystalline
    // (Fibonacci-seeded) SCVT, and more relaxation re-rolls the dislocation
    // rather than draining it (measured under --sweeps 200/600/2000, and
    // re-measured 2026-08-25: three refinement layouts, three
    // dislocations). Until this gate they PASSED here and failed two stages
    // later, at port load.
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
            "edge {worst_dv_edge}: the two dual vertices sit {min_dv_m:.1} m apart (dcEdge {:.0} m, dv/dc {:.2e}), under the {:.4e} m stored-point floor for {} storage ({:.0} coordinate quanta of {:.4e} m). The static that accompanies this grid stores its vertices at that quantum, and rounding them there tilts this dual edge off the primal edge it is meant to be perpendicular to by 1.935*q/dvEdge = {:.3e} at the worst edge -- MEASURED on the published statics' own bytes over 614,400 edges. Every operator the dycore builds from those points (edge normals, cell tangent planes, the reconstruction coefficients the init stage derives) assumes that perpendicularity, and the defect grows without bound as the dual edge approaches the quantum. For scale: published x1.40962 measures 45,016.7 m at its shortest and x4.163842 runs at 1,170.0 m, both carrying under 1.4e-4. What moves this floor is the STORAGE, not the gate: a mesh with no native MPAS-A counterpart may be stored binary64, where the same budget reads 3.7e-7 m. Evidence: evidence/dvedge-floor-20260829/",
            dc_m[worst_dv_edge],
            mesh.dv_edge[worst_dv_edge] / mesh.dc_edge[worst_dv_edge],
            limits.min_dv_edge_m,
            limits.storage.tag(),
            DV_EDGE_FLOOR_QUANTA,
            limits.storage.quantum_m(r),
            crate::staticfile::coordframe::ORTHOGONALITY_WORST_CONSTANT
                * limits.storage.quantum_m(r)
                / min_dv_m
        )));
    }
    if min_ratio < limits.min_dv_over_dc {
        return Err(MpasError::Refusal(format!(
            "edge {worst_ratio_edge}: dvEdge/dcEdge = {min_ratio:.3e} ({:.1} m over {:.0} m), under the {:.0e} floor. Two Voronoi vertices nearly coincide while their cells stay a full spacing apart -- the signature of a near-cocircular pentagon-heptagon dislocation quad. The TRiSK tangential weights divide by dvEdge, so the wind term reconstructed across this edge is amplified by dc/dv = {:.0}, and the MPAS port refuses the pair itself (DualEdgeAdmissionError, the same 0.02 floor) before any CUDA allocation; the published family never carries this (x1.40962 reads 0.3945 at its floor, x4.163842 reads 0.0336), and the measured defect class on Fibonacci-seeded uniform meshes sits at 0.0147 and below. What works, measured: a different refinement layout, or fewer cells in the refined region -- more relaxation re-rolls the dislocation rather than draining it",
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
        max_area_decomposition_abs: max_gap,
        max_area_decomposition_ulps: max_gap / f64::EPSILON,
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

#[cfg(test)]mod fp32_floor_tests {
    use super::*;
    use crate::mesh::density::MeshSpec;
    use crate::mesh::derive::MpasMesh;
    use crate::mesh::emit::nominal_min_dc_from_m;
    use crate::mesh::geom::EARTH_RADIUS_M;
    use crate::mesh::lloyd::{LloydOptions, relax, seed_points};

    fn derive_relaxed(points: Vec<crate::mesh::geom::V3>, spec: &MeshSpec) -> MpasMesh {
        let mut pts = points;
        // The monitor's collapse refusal is DISARMED here on purpose: these
        // tests deliberately construct the symptomatic Fibonacci mesh the
        // production relaxation now refuses mid-flight, because the gate
        // under test is `validate`, two stages later. The floor is options
        // data the same way `Limits` is; no production door zeroes it.
        let opts = LloydOptions {
            monitor_floor: 0.0,
            ..LloydOptions::default()
        };
        let out = relax(&mut pts, spec, &opts)
            .unwrap_or_else(|e| panic!("relaxation refused: {e}"));
        let density: Vec<f64> = pts.iter().map(|&p| spec.density(p)).collect();
        let nominal = nominal_min_dc_from_m(spec.finest_km() * 1000.0);
        MpasMesh::derive(pts, density, &out.rings, nominal).expect("derive")
    }

    /// The refusing direction, on the real pipeline, AFTER the 2026-08-25
    /// floor ruling: the 2,000-cell Fibonacci mesh (measured 7,337.6 m dual
    /// edge over a 721,237 m dcEdge -- ratio 1.017e-2, a 98x TRiSK
    /// amplification) is still refused, and the refusal now names the LIVE
    /// breakage -- the dislocation ratio, which the port itself refuses with
    /// DualEdgeAdmissionError -- rather than the retired FP32 storage
    /// comparison. (Guard citation: the floor ruling, evidence gpuwm-hex
    /// tree/evidence/graded-goldberg-20260825/; stale-guard audit
    /// 2026-08-25, finding 3.)
    #[test]
    fn a_fibonacci_dislocation_is_refused_by_the_ratio_floor_with_the_numbers() {
        let spec = MeshSpec::uniform(120.0);
        let points = seed_points(&spec, 2_000).expect("seed");
        let mesh = derive_relaxed(points, &spec);
        let err = validate(&mesh, Limits::default())
            .expect_err("the dislocation class must refuse under the ruled floors")
            .to_string();
        assert!(
            err.contains("dvEdge/dcEdge") && err.contains("dislocation"),
            "the refusal does not name the surviving breakage: {err}"
        );
        assert!(
            err.contains("dc/dv") && err.contains("DualEdgeAdmissionError"),
            "the refusal does not carry the amplification and the port's own gate: {err}"
        );
        assert!(
            !err.contains("storage-length floor"),
            "the refusal cites the length floor instead of the ratio floor: {err}"
        );
        assert!(
            !err.contains("FP32 survival floor") && !err.contains("rtol 2e-5"),
            "the refusal quotes the retired storage premise: {err}"
        );
    }

    /// THE RULING DID NOT MOVE. Re-expressing the floor as a multiple of the
    /// coordinate quantum has to reproduce the standing 200.0 m at the
    /// published representation EXACTLY -- not nearly -- or every file with a
    /// native MPAS-A counterpart is being judged by a different number than
    /// the one that was ruled.
    #[test]
    fn the_binary32_floor_is_the_ruled_200_m_to_the_bit() {
        use crate::staticfile::coordframe::CoordinateRepresentation as CR;
        let d = Limits::default();
        assert_eq!(
            d.min_dv_edge_m, 200.0,
            "the 2026-08-25 ruling is 200.0 m at binary32 and the derivation must land on it"
        );
        assert_eq!(d.storage, CR::Binary32EarthCentred);
        assert_eq!(
            Limits::for_storage(CR::Binary32EarthCentred).min_dv_edge_m,
            200.0
        );
        // 400 quanta, and the quantum is the file's, not a table's.
        assert_eq!(
            DV_EDGE_FLOOR_QUANTA * CR::Binary32EarthCentred.quantum_m(EARTH_RADIUS_M),
            200.0
        );
    }

    /// The same BUDGET at binary64 storage: 400 quanta of 9.313e-10 m. The
    /// floor stops being what a sub-kilometre mesh runs into, and the shape
    /// gate (`min_dv_over_dc`, the port's own admission floor) is untouched.
    #[test]
    fn binary64_storage_earns_the_same_budget_at_its_own_quantum() {
        use crate::staticfile::coordframe::CoordinateRepresentation as CR;
        let l = Limits::for_storage(CR::Binary64EarthCentred);
        let expect = 400.0 * 9.313225746154785e-10;
        assert!(
            (l.min_dv_edge_m - expect).abs() < 1e-18,
            "binary64 floor {:e}, expected {expect:e}",
            l.min_dv_edge_m
        );
        assert!(l.min_dv_edge_m < 4.0e-7 && l.min_dv_edge_m > 3.0e-7);
        // The SHAPE gate does not move for a representation.
        assert_eq!(l.min_dv_over_dc, Limits::default().min_dv_over_dc);
        assert_eq!(l.orthogonality, Limits::default().orthogonality);
        assert_eq!(l.area_decomposition_ulps, Limits::default().area_decomposition_ulps);
        assert_eq!(l.weight_antisymmetry, Limits::default().weight_antisymmetry);
        assert_eq!(l.kite_partition, Limits::default().kite_partition);
        assert_eq!(l.sphere_closure, Limits::default().sphere_closure);
    }

    /// The refusal names the storage it was written against, the quantum, and
    /// the orthogonality defect -- not the retired claim that the TRiSK
    /// weights divide by dvEdge.
    #[test]
    fn the_length_refusal_names_the_storage_and_the_defect_it_prevents() {
        let spec = MeshSpec::uniform(120.0);
        let points = seed_points(&spec, 2_000).expect("seed");
        let mesh = derive_relaxed(points, &spec);
        // A floor high enough that this mesh's 7.3 km dual edge trips the
        // LENGTH clause, with the ratio clause named away so the length
        // refusal is the one that speaks.
        let tall = Limits {
            min_dv_edge_m: 50_000.0,
            min_dv_over_dc: 0.0,
            ..Limits::default()
        };
        let err = validate(&mesh, tall).expect_err("50 km floor must refuse").to_string();
        assert!(err.contains("stored-point floor"), "{err}");
        assert!(err.contains("binary32_earth_centred"), "{err}");
        assert!(err.contains("coordinate quanta"), "{err}");
        assert!(err.contains("perpendicular"), "{err}");
        assert!(
            !err.contains("TRiSK tangential weights divide by it"),
            "the refusal repeats the breakage measured not to occur: {err}"
        );
        assert!(err.contains("binary64"), "the remedy is not named: {err}");
    }

    /// The ruled length floor contributes no refusal to the symptomatic
    /// mesh: its 7,337.6 m shortest dual edge clears 200 m with decades to
    /// spare, so the whole verdict is the ratio's -- while the published
    /// x4.163842 (1,170 m, gated in the goldens suite) and the canonical
    /// graded mesh (925.7 m) are the integration-scale length-floor admits.
    /// (Guard citation: the 2026-08-25 ruling, evidence gpuwm-hex
    /// tree/evidence/graded-goldberg-20260825/.)
    #[test]
    fn the_ruled_length_floor_leaves_the_verdict_to_the_ratio() {
        let spec = MeshSpec::uniform(120.0);
        let points = seed_points(&spec, 2_000).expect("seed");
        let mesh = derive_relaxed(points, &spec);
        let length_only = Limits {
            min_dv_over_dc: 0.0,
            ..Limits::default()
        };
        let report = validate(&mesh, length_only).expect(
            "with only the ratio floor named away, the 7.3 km edge clears the 200 m length floor",
        );
        eprintln!(
            "fibonacci 2,000 under the ruled length floor: min dvEdge {:.1} m, min dv/dc {:.4e}",
            report.min_dv_edge_m, report.min_dv_over_dc
        );
        assert!(report.min_dv_edge_m > 200.0 && report.min_dv_edge_m < 7_500.0);
        assert!(report.min_dv_over_dc < 0.02);
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
        // pentagons, no heptagons, nothing anywhere near the admission floor.
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

    /// The WRITER'S BACKSTOP, refusing direction, on a mesh built exactly the
    /// way the generator built `v16.66.195630`: a generator dropped on a
    /// quad's common circumcentre, which is what `surgery`'s S2 insertion
    /// does, and which lands a four-neighbour cell.
    #[test]
    fn a_four_coordinated_cell_is_refused_by_the_writer_with_its_measured_cost() {
        let (mesh, quad_cell) = mesh_with_a_quadrilateral_cell();
        assert_eq!(mesh.n_edges_on_cell[quad_cell], 4);
        // Every closure identity above the floor still holds -- that is why
        // this shipped once. Named away, the rest of `validate` passes it.
        let defect: i64 = mesh.n_edges_on_cell.iter().map(|&d| 6 - d as i64).sum();
        assert_eq!(defect, 12);

        let err = validate(&mesh, Limits::default())
            .expect_err("a quadrilateral cell must not be emitted")
            .to_string();
        assert!(
            err.contains(&format!("cell {quad_cell} has 4 edges")),
            "the refusal does not name the offending cell: {err}"
        );
        assert!(
            err.contains("Goldberg") && err.contains("quadrilateral is not a cell of this family"),
            "the refusal does not name the family invariant: {err}"
        );
        assert!(
            err.contains("197 K") && err.contains("2026-08-26"),
            "the refusal does not carry the measured breakage and its date: {err}"
        );
        assert!(
            err.contains("v16.66.195630"),
            "the refusal does not name the mesh that measured the cost: {err}"
        );
    }

    /// The admitting direction, and the one-sidedness: the floor is a FLOOR.
    /// High coordination is admitted, because the 8-coordinated cell in
    /// `v16.66.195630` was measured harmless and `v20.80.151649` completes six
    /// forecast hours with 1,017 heptagons.
    #[test]
    fn the_coordination_floor_is_one_sided_and_admits_high_coordination() {
        let spec = MeshSpec::uniform(120.0);
        let choice = crate::mesh::icosa::snap_cells(2_000, false).expect("snap");
        let points = crate::mesh::icosa::seed(choice.m, choice.n).expect("seed");
        let mesh = derive_relaxed(points, &spec);
        let report = validate(&mesh, Limits::default()).expect("the healthy control admits");
        assert!(
            report
                .coordination_histogram
                .iter()
                .all(|&(d, _)| d >= crate::mesh::surgery::MIN_COORDINATION as i32),
            "{:?}",
            report.coordination_histogram
        );
        // And a mesh carrying HIGH coordination is admitted on its own
        // merits: the graded ladder's own two-level fixture reads heptagons
        // and passes.
        let graded = crate::mesh::density::MeshSpec {
            background_km: 480.0,
            regions: vec![crate::mesh::density::Region {
                shape: crate::mesh::density::Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 4000.0,
                },
                spacing_km: 240.0,
                transition: crate::mesh::density::TransitionField::Km(3000.0),
            }],
            name: None,
        };
        let (points, rings, _, _, _) = crate::mesh::hierarchy::generate_graded(
            &graded,
            50_000,
            &LloydOptions::default(),
            &crate::mesh::surgery::SurgeryOptions::default(),
            crate::mesh::hierarchy::DEFAULT_BETA,
            |_| {},
        )
        .expect("the graded ladder builds its own fixture");
        let density: Vec<f64> = points.iter().map(|&p| graded.density(p)).collect();
        let nominal = nominal_min_dc_from_m(graded.finest_km() * 1000.0);
        let gmesh = MpasMesh::derive(points, density, &rings, nominal).expect("derive");
        let greport = validate(&gmesh, Limits::default())
            .unwrap_or_else(|e| panic!("the graded fixture is refused: {e}"));
        assert!(
            greport
                .coordination_histogram
                .iter()
                .any(|&(d, c)| d == 7 && c > 0),
            "the graded fixture carries no heptagons, so it proves nothing about the floor's one-sidedness: {:?}",
            greport.coordination_histogram
        );
    }

    /// A relaxed uniform mesh with one generator dropped on a quad's common
    /// circumcentre: the placement `surgery`'s S2 makes. Returns the derived
    /// mesh and the index of the four-neighbour cell.
    fn mesh_with_a_quadrilateral_cell() -> (MpasMesh, usize) {
        use crate::mesh::geom::circumcenter;
        let spec = MeshSpec::uniform(120.0);
        let choice = crate::mesh::icosa::snap_cells(2_000, false).expect("snap");
        let mut pts = crate::mesh::icosa::seed(choice.m, choice.n).expect("seed");
        relax(&mut pts, &spec, &LloydOptions::default()).expect("relax");
        let rings = crate::mesh::hull::delaunay_rings(&pts).expect("delaunay");
        let mut quads: Vec<crate::mesh::surgery::QuadReading> = Vec::new();
        crate::mesh::surgery::for_each_quad(&pts, &rings, |r| quads.push(r));
        quads.sort_by(|x, y| (x.i, x.j).cmp(&(y.i, y.j)));
        for q in quads.iter().take(64) {
            let Some(u) = circumcenter(pts[q.i as usize], pts[q.a as usize], pts[q.j as usize])
            else {
                continue;
            };
            let mut trial = pts.clone();
            trial.push(u);
            let k = trial.len() - 1;
            let r2 = crate::mesh::hull::delaunay_rings(&trial).expect("delaunay");
            if r2.ring(k).len() == 4 {
                let density: Vec<f64> = trial.iter().map(|&p| spec.density(p)).collect();
                let nominal = nominal_min_dc_from_m(spec.finest_km() * 1000.0);
                let mesh = MpasMesh::derive(trial, density, &r2, nominal).expect("derive");
                return (mesh, k);
            }
        }
        panic!("no circumcentre placement in the first 64 canonical quads landed a quadrilateral");
    }

    /// The floors are data (`Limits`), so a consumer without the port's
    /// DualEdgePolicy gate and with an FP64 static CAN name lower ones: the
    /// same mesh the default refuses passes every geometric check under
    /// named-away floors. The default stays the port's admission truth.
    #[test]
    fn the_floors_are_the_default_and_a_caller_can_name_a_different_consumer() {
        let spec = MeshSpec::uniform(120.0);
        let points = seed_points(&spec, 2_000).expect("seed");
        let mesh = derive_relaxed(points, &spec);
        assert!(
            validate(&mesh, Limits::default()).is_err(),
            "the dislocation class must refuse under the default floors"
        );
        let relaxed_limits = Limits {
            min_dv_over_dc: 0.0,
            ..Limits::default()
        };
        let report = validate(&mesh, relaxed_limits)
            .expect("with the floor named away the geometric checks still pass");
        eprintln!(
            "fibonacci 2,000 under a named-away floor: min dvEdge {:.1} m, min dv/dc {:.4e}",
            report.min_dv_edge_m, report.min_dv_over_dc
        );
        assert!(report.min_dv_over_dc < 0.02);
    }
}

/// What the area-decomposition gate reads, and what it does not.
///
/// The gate was a RELATIVE tolerance that had been re-anchored once (1e-10 ->
/// 1e-9) and was about to need it a second time: the swath layer's five-level
/// ladder built to completion and read 1.023e-9. These tests hold the
/// measurement that replaced the constant with a derived one, in both
/// directions -- the healthy band and the breakage the gate exists for.
///
/// (Guard citation: the 2026-08-26 re-derivation; evidence
/// `evidence/area-tolerance-20260826/` in this repo.)
#[cfg(test)]
mod area_decomposition_tests {
    use super::*;
    use crate::mesh::density::{MeshSpec, Region, Shape, TransitionField};
    use crate::mesh::derive::MpasMesh;
    use crate::mesh::emit::nominal_min_dc_from_m;
    use crate::mesh::lloyd::LloydOptions;

    /// The swath spec `s01`, built by the real `rw_mpas_mesh` binary at
    /// 74f5725b8 with this gate lifted: 123,423 cells, five-level ladder,
    /// 75 km background, 4 km finest, 4.517 km delivered.
    ///
    /// The worst cell is 110971. Its two decompositions enclose areas
    /// 4.673e-16 sr apart -- and that cell is a plain degree-6 hexagon whose
    /// tightest dvEdge/dcEdge is 0.5392, not a sliver.
    const S01_WORST_GAP_SR: f64 = 4.673e-16;
    const S01_WORST_AREA_SR: f64 = 4.5678e-7;
    /// What that same 4.673e-16 reads on the coarsest cell of the SAME mesh
    /// (78.378 km across, 1.219e-4 sr): the retired form's verdict was a
    /// reading of the cell, not of the mesh.
    const S01_COARSE_AREA_SR: f64 = 1.219e-4;
    /// The scalar the retired relative form carried.
    const RETIRED_RELATIVE_LIMIT: f64 = 1e-9;

    /// RED FIRST, on the real artifact's numbers: the ladder the generator now
    /// runs to completion was refused by the retired relative form, and the
    /// derived floor admits it.
    #[test]
    fn the_five_level_swath_ladder_is_inside_the_arithmetics_own_noise() {
        let retired_reading = S01_WORST_GAP_SR / S01_WORST_AREA_SR;
        assert!(
            retired_reading > RETIRED_RELATIVE_LIMIT,
            "the recorded refusal does not follow from the recorded numbers: \
             {retired_reading:.4e} against {RETIRED_RELATIVE_LIMIT:.1e}"
        );
        assert!(
            (retired_reading - 1.0231e-9).abs() < 5e-13,
            "the reproduced reading {retired_reading:.5e} is not the 1.0231e-9 the binary printed"
        );
        let floor = Limits::default().area_decomposition_floor();
        assert!(
            S01_WORST_GAP_SR <= floor,
            "the swath ladder's worst cell ({:.2} eps) is outside the derived floor ({:.2} eps)",
            S01_WORST_GAP_SR / f64::EPSILON,
            Limits::default().area_decomposition_ulps
        );
        // The same absolute gap on the coarsest cell of the same mesh reads
        // 267x smaller. One mesh, one arithmetic, two verdicts.
        let on_a_coarse_cell = S01_WORST_GAP_SR / S01_COARSE_AREA_SR;
        assert!(
            on_a_coarse_cell < RETIRED_RELATIVE_LIMIT / 100.0,
            "the scale dependence that made the retired form need re-anchoring is not \
             reproduced: {on_a_coarse_cell:.3e}"
        );
    }

    /// The floor is a MARGIN over a measurement, and the margin is stated. If
    /// either number moves, this fails and the comment above `Limits::default`
    /// has to be re-measured rather than quietly re-rolled.
    #[test]
    fn the_floor_is_the_measured_worst_with_the_margin_it_claims() {
        // The worst absolute gap measured over five real graded meshes,
        // ladder depths 1 to 5, 123k-138k cells each (2.349, 2.383, 2.518,
        // 2.383 and 2.502 machine epsilon).
        const MEASURED_WORST_ULPS: f64 = 2.518;
        // The mechanism's own ceiling: maxEdges edges, each opening a
        // dvEdge/2 * eps/h sliver, at dvEdge = dcEdge.
        const MECHANISM_CEILING_ULPS: f64 = 5.0;
        let floor = Limits::default().area_decomposition_ulps;
        assert!(
            floor >= 8.0 * MEASURED_WORST_ULPS,
            "the floor {floor} is under 8x the measured worst {MEASURED_WORST_ULPS}"
        );
        assert!(
            floor >= 4.0 * MECHANISM_CEILING_ULPS,
            "the floor {floor} is under 4x the mechanism's ceiling {MECHANISM_CEILING_ULPS}"
        );
        assert!(
            floor < 1e6,
            "the floor {floor} has stopped being a noise floor"
        );
    }

    /// A graded fixture whose cells span a factor of several in spacing, so
    /// one mesh carries both ends of the argument.
    fn graded_fixture() -> MpasMesh {
        let spec = MeshSpec {
            background_km: 480.0,
            regions: vec![Region {
                shape: Shape::Cap {
                    center_deg: [39.0, -98.0],
                    radius_km: 4000.0,
                },
                spacing_km: 120.0,
                transition: TransitionField::Km(3000.0),
            }],
            name: None,
        };
        let (points, rings, _, _, _) = crate::mesh::hierarchy::generate_graded(
            &spec,
            50_000,
            &LloydOptions::default(),
            &crate::mesh::surgery::SurgeryOptions::default(),
            crate::mesh::hierarchy::DEFAULT_BETA,
            |_| {},
        )
        .expect("the graded ladder builds its own fixture");
        let density: Vec<f64> = points.iter().map(|&p| spec.density(p)).collect();
        let nominal = nominal_min_dc_from_m(spec.finest_km() * 1000.0);
        MpasMesh::derive(points, density, &rings, nominal).expect("derive")
    }

    /// The finding, on a mesh this test builds itself: the ABSOLUTE gap is the
    /// same size on the smallest cells and the biggest, while the RELATIVE
    /// reading -- what the gate used to compare -- swings with the cell area.
    /// A scalar relative tolerance therefore cannot be anchored; it is a
    /// reading of the finest cell in the mesh.
    #[test]
    fn the_gap_is_a_constant_area_and_the_relative_reading_is_not() {
        let mesh = graded_fixture();
        let polygon = mesh.polygon_areas();
        let mut fine_gap = 0.0f64;
        let mut coarse_gap = 0.0f64;
        let mut fine_rel = 0.0f64;
        let mut coarse_rel = 0.0f64;
        let (mut a_min, mut a_max) = (f64::INFINITY, 0.0f64);
        for c in 0..mesh.n_cells {
            a_min = a_min.min(mesh.area_cell[c]);
            a_max = a_max.max(mesh.area_cell[c]);
        }
        // Split at the geometric mean so both halves are populated whatever
        // the spec is.
        let split = (a_min * a_max).sqrt();
        for c in 0..mesh.n_cells {
            let gap = (polygon[c] - mesh.area_cell[c]).abs();
            let rel = gap / mesh.area_cell[c];
            if mesh.area_cell[c] < split {
                fine_gap = fine_gap.max(gap);
                fine_rel = fine_rel.max(rel);
            } else {
                coarse_gap = coarse_gap.max(gap);
                coarse_rel = coarse_rel.max(rel);
            }
        }
        eprintln!(
            "graded {} cells, area span {:.2}x: fine gap {:.3} eps rel {:.3e} | coarse gap {:.3} eps rel {:.3e}",
            mesh.n_cells,
            a_max / a_min,
            fine_gap / f64::EPSILON,
            fine_rel,
            coarse_gap / f64::EPSILON,
            coarse_rel
        );
        assert!(
            a_max / a_min > 8.0,
            "the fixture does not span enough cell sizes to say anything: {:.2}x",
            a_max / a_min
        );
        // The absolute gap is the same animal at both ends -- within a factor
        // of three, on a max over two samples of a noise distribution.
        let gap_ratio = (fine_gap / coarse_gap).max(coarse_gap / fine_gap);
        assert!(
            gap_ratio < 3.0,
            "the absolute gap is not scale free: {gap_ratio:.2}x between the fine and coarse halves"
        );
        // The relative reading is not: it tracks the area ratio.
        assert!(
            fine_rel / coarse_rel > 4.0,
            "the relative reading did not swing with cell size ({:.2}x), so this fixture \
             cannot show why a scalar relative tolerance has to be re-anchored",
            fine_rel / coarse_rel
        );
        validate(&mesh, Limits::default()).expect("the graded fixture is refused");
    }

    /// The breakage the gate exists to prevent, measured rather than quoted: a
    /// cell whose vertex ring is no longer the boundary of the region its kites
    /// tile. Two ring entries are transposed, which is what a `verticesOnCell`
    /// ordering defect produces -- the polygon fan then runs over a
    /// figure-of-eight while the kites still tile the cell.
    ///
    /// This is the number that justifies the floor: the separation between the
    /// worst healthy reading and the mildest broken one.
    #[test]
    fn a_ring_that_is_not_the_cells_boundary_is_refused_with_decades_to_spare() {
        let mut mesh = graded_fixture();
        let healthy = validate(&mesh, Limits::default())
            .expect("the fixture must be healthy before it is broken");

        // EVERY adjacent transposition on EVERY cell, so the number quoted is
        // the MILDEST breakage the fixture can produce and not a lucky one.
        // The signature of this corruption is a fraction of the cell it
        // happens on, so it is measured as a fraction and the separation is
        // taken against the mesh's own smallest cell -- the worst case for an
        // absolute floor.
        let clean_polygon = mesh.polygon_areas();
        let mut mildest = f64::INFINITY;
        let mut mildest_at = (0usize, 0usize);
        let mut worst_fraction = 0.0f64;
        let mut ring: Vec<crate::mesh::geom::V3> = Vec::with_capacity(mesh.max_edges);
        for c in 0..mesh.n_cells {
            let deg = mesh.n_edges_on_cell[c] as usize;
            let base = c * mesh.max_edges;
            for j in 0..deg {
                ring.clear();
                for k in 0..deg {
                    ring.push(mesh.vertex_xyz[mesh.vertices_on_cell[base + k] as usize]);
                }
                ring.swap(j, (j + 1) % deg);
                let broken = crate::mesh::geom::polygon_area(&ring);
                let fraction = ((broken - clean_polygon[c]) / mesh.area_cell[c]).abs();
                if fraction < mildest {
                    mildest = fraction;
                    mildest_at = (c, j);
                }
                worst_fraction = worst_fraction.max(fraction);
            }
        }
        let a_min = mesh
            .area_cell
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let floor = Limits::default().area_decomposition_floor();
        let mildest_abs = mildest * a_min;
        eprintln!(
            "healthy worst {:.3e} sr ({:.2} eps); ring transpositions over {} cells read \
             {:.3e} to {:.3e} of the cell's own area (mildest at cell {} slot {}); on this \
             mesh's smallest cell ({:.3e} sr) the mildest is {:.3e} sr against a floor of \
             {:.3e} sr -- {:.1} decades",
            healthy.max_area_decomposition_abs,
            healthy.max_area_decomposition_ulps,
            mesh.n_cells,
            mildest,
            worst_fraction,
            mildest_at.0,
            mildest_at.1,
            a_min,
            mildest_abs,
            floor,
            (mildest_abs / floor).log10()
        );
        assert!(
            mildest_abs / floor > 1e4,
            "the MILDEST ring transposition is only {:.3e}x the noise floor on this mesh's \
             smallest cell; the floor is not separating anything",
            mildest_abs / floor
        );

        // And the gate refuses one, naming what it found.
        let (victim, slot) = mildest_at;
        let base = victim * mesh.max_edges;
        let deg = mesh.n_edges_on_cell[victim] as usize;
        mesh.vertices_on_cell
            .swap(base + slot, base + (slot + 1) % deg);
        let err = validate(&mesh, Limits::default())
            .expect_err("a ring that is not the cell's boundary must be refused")
            .to_string();
        assert!(
            err.contains("vertex ring is not the boundary") && err.contains("machine epsilon"),
            "the refusal does not name the breakage and the units it measured it in: {err}"
        );
    }

    /// What this gate CANNOT see, stated so nobody reads more into it than it
    /// checks. Every ring vertex is a circumcentre of three generators and is
    /// therefore equidistant from the two that own each of its edges; the edge
    /// point is the midpoint of those same two. All three sit on one great
    /// circle whichever diagonal a near-cocircular quad was triangulated on,
    /// so the two decompositions agree for a NON-Delaunay triangulation just as
    /// they do for a Delaunay one.
    ///
    /// The retired refusal text claimed this check proved "the mesh is the
    /// Voronoi tessellation of its own generators". It does not, and the text
    /// no longer says so.
    #[test]
    fn inserting_the_edge_point_never_moves_the_area_whichever_diagonal_was_taken() {
        use crate::mesh::geom::{V3, circumcenter, from_lat_lon, polygon_area, tri_area, unit};
        // Four generators on one small circle -- both diagonals are Delaunay --
        // plus an owner outside the quad.
        let quad: Vec<V3> = (0..4)
            .map(|k| {
                let t = std::f64::consts::TAU * k as f64 / 4.0;
                let r: f64 = 0.02;
                from_lat_lon(r * t.sin(), r * t.cos())
            })
            .collect();
        let owner = from_lat_lon(0.0, 0.05);
        for diagonal in [(0usize, 2usize), (1usize, 3usize)] {
            let (p, q) = (quad[diagonal.0], quad[diagonal.1]);
            let v0 = circumcenter(p, q, owner).expect("circumcentre");
            let v1 = circumcenter(q, p, owner).expect("circumcentre");
            let m = unit([p[0] + q[0], p[1] + q[1], p[2] + q[2]]).expect("midpoint");
            let without = polygon_area(&[v0, v1, owner]);
            let with = tri_area(v0, m, owner) + tri_area(m, v1, owner);
            let gap = (without - with).abs();
            eprintln!("diagonal {diagonal:?}: inserting the edge point moves the area by {gap:.3e}");
            assert!(
                gap < 64.0 * f64::EPSILON,
                "inserting the edge point moved the area by {gap:.3e}, so the premise this gate \
                 rests on -- vertex, edge point and vertex on one great circle -- does not hold \
                 and the refusal text is wrong"
            );
        }
    }
}
