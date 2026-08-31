//! Writing an MPAS grid file.
//!
//! The schema is the published one, read off both released meshes rather than
//! from a specification: 8 dimensions, 38 variables, 7 global attributes, CDF-2
//! (`64-bit offset`). Indices go out 1-BASED, which is how MPAS reads them, and
//! the padding slots reproduce the published fill rules exactly.
//!
//! Provenance attributes are added beyond the published seven. A reader that
//! does not know them ignores them; a reader that does can tell what request
//! produced the mesh without a side file.
//!
//! WHAT THIS FILE IS NOT. A grid file alone cannot reach the dycore. The mesh
//! registry also requires a matching STATIC file carrying terrain, land use and
//! soil on the same cells, `nVertLevels = 55`, `nSoilLevels = 4`, and a
//! `nominalMinDc` that is FP32-bit-exact against the declared nominal spacing.
//! Producing that static is a separate job against a terrain archive and is not
//! part of this binary.

use std::path::{Path, PathBuf};

use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::{MpasError, MpasResult};
use crate::mesh::derive::MpasMesh;
use crate::mesh::geom::{EARTH_RADIUS_M, lat_lon};

/// The mesh-spec version string the published files carry.
pub const MESH_SPEC: &str = "1.1";

/// `densityFunctionCode` is a single byte on BOTH published files, and that
/// byte is a newline. It is a vestigial placeholder; the real resolution spec
/// goes into the provenance attributes.
const DENSITY_FUNCTION_CODE: u8 = b'\n';

/// What an emit produced.
#[derive(Debug, Clone, Serialize)]
pub struct EmittedMesh {
    pub path: PathBuf,
    pub bytes: u64,
    pub file_id: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    pub max_edges: usize,
    pub sha256: String,
}

/// Extra attributes to stamp beside the seven the published files carry.
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    /// The resolution spec, as the JSON the caller was given.
    pub spec_json: String,
    /// A one-line human summary of the request.
    pub request: String,
    /// The generator's own receipt, as JSON.
    pub receipt_json: String,
    /// The coordinate representation a STATIC built from this grid must
    /// declare and store.
    ///
    /// `None` writes no attribute at all, and a reader that finds none must
    /// take `binary32_earth_centred` -- what every published file carries and
    /// what the dycore byte-identity anchor is a property of. It is `Some`
    /// only for a mesh this generator created, which by construction has no
    /// native MPAS-A counterpart and therefore nothing to be byte-identical
    /// to. The choice travels with the MESH rather than with a command line so
    /// a static cannot be built at a representation its grid never asked for.
    pub static_coordinates: Option<crate::staticfile::coordframe::CoordinateRepresentation>,
}

/// Attribute a grid carries to say what representation its static must use.
pub const STATIC_COORDINATES_ATTR: &str = "rw_static_coordinate_representation";

/// Write `mesh` as a classic-netCDF MPAS grid file.
///
/// Lands atomically: the file is built under a sibling temporary name and
/// renamed into place only after the writer has confirmed every declared slab
/// was written and flushed. A half-written grid file that a reader opens and
/// half-believes is worse than no file.
pub fn write_grid(
    mesh: &MpasMesh,
    destination: &Path,
    provenance: &Provenance,
    clobber: bool,
) -> MpasResult<EmittedMesh> {
    if destination.exists() && !clobber {
        return Err(MpasError::Refusal(format!(
            "{} exists; pass --clobber to replace it. Overwriting a mesh silently would leave every init file, static file and run receipt that cites it pointing at different bytes under the same name",
            destination.display()
        )));
    }
    if let Some(parent) = destination.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let (nc, ne, nv) = (mesh.n_cells, mesh.n_edges, mesh.n_vertices);
    let (me, me2, vd) = (mesh.max_edges, mesh.max_edges2, mesh.vertex_degree);

    let dims = vec![
        NcDim::fixed("nCells", nc),
        NcDim::fixed("nEdges", ne),
        NcDim::fixed("nVertices", nv),
        NcDim::fixed("maxEdges", me),
        NcDim::fixed("maxEdges2", me2),
        NcDim::fixed("TWO", 2),
        NcDim::fixed("vertexDegree", vd),
        NcDim::fixed("codeLen", 1),
    ];
    let dimid = |name: &str| -> usize {
        dims.iter()
            .position(|d| d.name == name)
            .expect("dimension declared above")
    };
    let (d_cells, d_edges, d_verts) = (dimid("nCells"), dimid("nEdges"), dimid("nVertices"));
    let (d_me, d_me2, d_two, d_vd, d_code) = (
        dimid("maxEdges"),
        dimid("maxEdges2"),
        dimid("TWO"),
        dimid("vertexDegree"),
        dimid("codeLen"),
    );

    let file_id = mint_file_id(mesh);
    let mut gattrs = vec![
        NcAttr::text("mesh_spec", MESH_SPEC),
        NcAttr::text("on_a_sphere", "YES"),
        NcAttr::doubles("sphere_radius", vec![1.0]),
        NcAttr::text("is_periodic", "NO"),
        NcAttr::doubles("x_period", vec![0.0]),
        NcAttr::doubles("y_period", vec![0.0]),
        NcAttr::text("file_id", file_id.clone()),
    ];
    gattrs.extend([
        NcAttr::text(
            "rw_mesh_engine",
            concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
        ),
        NcAttr::text("rw_mesh_request", provenance.request.clone()),
        NcAttr::text("rw_mesh_spec_json", provenance.spec_json.clone()),
        NcAttr::text("rw_mesh_receipt_json", provenance.receipt_json.clone()),
        NcAttr::text(
            "rw_mesh_units",
            "sphere_radius is 1.0: areaCell, areaTriangle, kiteAreasOnVertex, dcEdge, dvEdge and nominalMinDc are UNIT-SPHERE quantities despite their units attributes. Multiply lengths by 6371229.0 m and areas by its square.",
        ),
        NcAttr::text(
            "rw_mesh_boundary",
            "This is a grid file. Running it needs a matching static file carrying terrain, land use and soil on these cells, nVertLevels=55, nSoilLevels=4 and an FP32-bit-exact nominalMinDc; this binary does not produce one.",
        ),
    ]);
    // Written only when the caller says so; absent means binary32, which is
    // what every published grid is and what keeps the byte-identity anchor.
    if let Some(rep) = provenance.static_coordinates {
        gattrs.push(NcAttr::text(STATIC_COORDINATES_ATTR, rep.tag()));
        gattrs.push(NcAttr::text(
            "rw_static_coordinate_note",
            "A static built from this grid must store xCell/yCell/zCell and the lat/lon pair              beside them in rw_static_coordinate_representation and must declare it as              rw_coordinate_representation. This mesh was GENERATED and has no native MPAS-A              counterpart, so no byte-identity anchor binds its storage precision. A grid              without this attribute is binary32_earth_centred and stays so.",
        ));
    }

    // --- 38 variables, in the order the published files carry them ----------
    let dbl = |name: &str, dims: Vec<usize>, units: &str, long: &str| -> NcVarDef {
        NcVarDef::new(name, NcType::Double, dims).with_attrs(vec![
            NcAttr::text("units", units),
            NcAttr::text("long_name", long),
        ])
    };
    let int = |name: &str, dims: Vec<usize>, long: &str| -> NcVarDef {
        NcVarDef::new(name, NcType::Int, dims)
            .with_attrs(vec![NcAttr::text("units", "-"), NcAttr::text("long_name", long)])
    };

    let vars = vec![
        dbl("latCell", vec![d_cells], "rad", "Latitude of cell centres"),
        dbl("lonCell", vec![d_cells], "rad", "Longitude of cell centres"),
        dbl("xCell", vec![d_cells], "m", "Cartesian x of cell centres"),
        dbl("yCell", vec![d_cells], "m", "Cartesian y of cell centres"),
        dbl("zCell", vec![d_cells], "m", "Cartesian z of cell centres"),
        int("indexToCellID", vec![d_cells], "Global cell identifier"),
        dbl("latEdge", vec![d_edges], "rad", "Latitude of edge points"),
        dbl("lonEdge", vec![d_edges], "rad", "Longitude of edge points"),
        dbl("xEdge", vec![d_edges], "m", "Cartesian x of edge points"),
        dbl("yEdge", vec![d_edges], "m", "Cartesian y of edge points"),
        dbl("zEdge", vec![d_edges], "m", "Cartesian z of edge points"),
        int("indexToEdgeID", vec![d_edges], "Global edge identifier"),
        dbl("latVertex", vec![d_verts], "rad", "Latitude of dual vertices"),
        dbl("lonVertex", vec![d_verts], "rad", "Longitude of dual vertices"),
        dbl("xVertex", vec![d_verts], "m", "Cartesian x of dual vertices"),
        dbl("yVertex", vec![d_verts], "m", "Cartesian y of dual vertices"),
        dbl("zVertex", vec![d_verts], "m", "Cartesian z of dual vertices"),
        int("indexToVertexID", vec![d_verts], "Global vertex identifier"),
        int("nEdgesOnCell", vec![d_cells], "Edges bounding each cell"),
        int("nEdgesOnEdge", vec![d_edges], "Edges in each tangential stencil"),
        int("cellsOnCell", vec![d_cells, d_me], "Neighbouring cells, counter-clockwise"),
        int("edgesOnCell", vec![d_cells, d_me], "Edges of each cell, counter-clockwise"),
        int("verticesOnCell", vec![d_cells, d_me], "Vertices of each cell, counter-clockwise"),
        int("edgesOnEdge", vec![d_edges, d_me2], "Tangential reconstruction stencil"),
        int("cellsOnEdge", vec![d_edges, d_two], "The two cells an edge separates"),
        int("verticesOnEdge", vec![d_edges, d_two], "The two endpoints of the dual edge"),
        int("cellsOnVertex", vec![d_verts, d_vd], "The cells meeting at each vertex"),
        int("edgesOnVertex", vec![d_verts, d_vd], "The edges meeting at each vertex"),
        dbl("areaCell", vec![d_cells], "m^2", "Voronoi cell area"),
        dbl("areaTriangle", vec![d_verts], "m^2", "Dual triangle area"),
        dbl("dcEdge", vec![d_edges], "m", "Distance between the two cell centres"),
        dbl("dvEdge", vec![d_edges], "-", "Distance between the two dual vertices"),
        dbl("angleEdge", vec![d_edges], "rad", "Angle of the edge normal from local east"),
        dbl("weightsOnEdge", vec![d_edges, d_me2], "-", "Tangential reconstruction weights"),
        dbl("kiteAreasOnVertex", vec![d_verts, d_vd], "m^2", "Area of each kite at a vertex"),
        dbl("meshDensity", vec![d_cells], "unitless", "Density function that generated this mesh"),
        dbl("nominalMinDc", vec![], "m", "Nominal minimum dcEdge where meshDensity is 1"),
        NcVarDef::new("densityFunctionCode", NcType::Char, vec![d_code]).with_attrs(vec![
            NcAttr::text("long_name", "Vestigial placeholder; see rw_mesh_spec_json"),
        ]),
    ];

    let temporary = destination.with_file_name(format!(
        ".{}.tmp",
        destination
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("mesh")
    ));
    if temporary.exists() {
        std::fs::remove_file(&temporary)?;
    }

    let result = (|| -> MpasResult<()> {
        let mut w = NcClassicWriter::create(&temporary, NcFormat::Offset64, dims.clone(), gattrs, vars, 0)?;

        let (lat_c, lon_c) = split_lat_lon(&mesh.cell_xyz);
        let (lat_e, lon_e) = split_lat_lon(&mesh.edge_xyz);
        let (lat_v, lon_v) = split_lat_lon(&mesh.vertex_xyz);
        w.put("latCell", NcData::Doubles(&lat_c))?;
        w.put("lonCell", NcData::Doubles(&lon_c))?;
        for (name, k) in [("xCell", 0usize), ("yCell", 1), ("zCell", 2)] {
            let v: Vec<f64> = mesh.cell_xyz.iter().map(|p| p[k]).collect();
            w.put(name, NcData::Doubles(&v))?;
        }
        w.put("indexToCellID", NcData::Ints(&one_based_ids(nc)))?;
        w.put("latEdge", NcData::Doubles(&lat_e))?;
        w.put("lonEdge", NcData::Doubles(&lon_e))?;
        for (name, k) in [("xEdge", 0usize), ("yEdge", 1), ("zEdge", 2)] {
            let v: Vec<f64> = mesh.edge_xyz.iter().map(|p| p[k]).collect();
            w.put(name, NcData::Doubles(&v))?;
        }
        w.put("indexToEdgeID", NcData::Ints(&one_based_ids(ne)))?;
        w.put("latVertex", NcData::Doubles(&lat_v))?;
        w.put("lonVertex", NcData::Doubles(&lon_v))?;
        for (name, k) in [("xVertex", 0usize), ("yVertex", 1), ("zVertex", 2)] {
            let v: Vec<f64> = mesh.vertex_xyz.iter().map(|p| p[k]).collect();
            w.put(name, NcData::Doubles(&v))?;
        }
        w.put("indexToVertexID", NcData::Ints(&one_based_ids(nv)))?;

        w.put("nEdgesOnCell", NcData::Ints(&mesh.n_edges_on_cell))?;
        w.put("nEdgesOnEdge", NcData::Ints(&mesh.n_edges_on_edge))?;
        w.put("cellsOnCell", NcData::Ints(&to_one_based(&mesh.cells_on_cell)))?;
        w.put("edgesOnCell", NcData::Ints(&to_one_based(&mesh.edges_on_cell)))?;
        w.put("verticesOnCell", NcData::Ints(&to_one_based(&mesh.vertices_on_cell)))?;
        w.put("edgesOnEdge", NcData::Ints(&to_one_based(&mesh.edges_on_edge)))?;
        w.put("cellsOnEdge", NcData::Ints(&to_one_based(&mesh.cells_on_edge)))?;
        w.put("verticesOnEdge", NcData::Ints(&to_one_based(&mesh.vertices_on_edge)))?;
        w.put("cellsOnVertex", NcData::Ints(&to_one_based(&mesh.cells_on_vertex)))?;
        w.put("edgesOnVertex", NcData::Ints(&to_one_based(&mesh.edges_on_vertex)))?;

        w.put("areaCell", NcData::Doubles(&mesh.area_cell))?;
        w.put("areaTriangle", NcData::Doubles(&mesh.area_triangle))?;
        w.put("dcEdge", NcData::Doubles(&mesh.dc_edge))?;
        w.put("dvEdge", NcData::Doubles(&mesh.dv_edge))?;
        w.put("angleEdge", NcData::Doubles(&mesh.angle_edge))?;
        w.put("weightsOnEdge", NcData::Doubles(&mesh.weights_on_edge))?;
        w.put("kiteAreasOnVertex", NcData::Doubles(&mesh.kite_areas_on_vertex))?;
        w.put("meshDensity", NcData::Doubles(&mesh.mesh_density))?;
        w.put("nominalMinDc", NcData::Doubles(&[mesh.nominal_min_dc]))?;
        w.put("densityFunctionCode", NcData::Chars(&[DENSITY_FUNCTION_CODE]))?;
        w.finish()?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(e);
    }
    std::fs::rename(&temporary, destination)?;

    Ok(EmittedMesh {
        path: destination.to_path_buf(),
        bytes: std::fs::metadata(destination)?.len(),
        file_id,
        n_cells: nc,
        n_edges: ne,
        n_vertices: nv,
        max_edges: me,
        sha256: crate::sha256_file(destination)?,
    })
}

/// The closed-form size of a grid file with this shape, in bytes.
///
/// Derived from the schema and verified to the byte against both published
/// meshes with `maxEdges = 10`: `1368 * nCells + 3316` reads 56,039,332 for
/// 40,962 cells and 224,139,172 for 163,842. Only the header differs when
/// provenance attributes are added, so this is a floor for a file this crate
/// writes, and the caller is told the actual size after the write.
pub fn published_schema_bytes(n_cells: usize) -> u64 {
    1368 * n_cells as u64 + 3316
}

fn one_based_ids(n: usize) -> Vec<i32> {
    (1..=n as i32).collect()
}

fn to_one_based(v: &[i32]) -> Vec<i32> {
    v.iter().map(|&x| x + 1).collect()
}

fn split_lat_lon(points: &[crate::mesh::geom::V3]) -> (Vec<f64>, Vec<f64>) {
    let mut lat = Vec::with_capacity(points.len());
    let mut lon = Vec::with_capacity(points.len());
    for &p in points {
        let (la, lo) = lat_lon(p);
        lat.push(la);
        lon.push(lo);
    }
    (lat, lon)
}

/// Eight hex characters, the shape the published files' `file_id` carries,
/// derived from the mesh's own contents so the same mesh always stamps the same
/// identifier.
fn mint_file_id(mesh: &MpasMesh) -> String {
    let mut h = Sha256::new();
    h.update(mesh.n_cells.to_le_bytes());
    h.update(mesh.n_edges.to_le_bytes());
    h.update(mesh.n_vertices.to_le_bytes());
    for p in &mesh.cell_xyz {
        for k in 0..3 {
            h.update(p[k].to_le_bytes());
        }
    }
    for &d in &mesh.mesh_density {
        h.update(d.to_le_bytes());
    }
    h.update(mesh.nominal_min_dc.to_le_bytes());
    crate::weights::hex(&h.finalize())[..8].to_string()
}

/// Spacing in metres from a unit-sphere `nominalMinDc`, and back. The stamp is
/// a LABEL: the published variable mesh stamps 25.000 km while its finest
/// achieved `dcEdge` is 17.893 km, so nothing should gate on it.
pub fn nominal_min_dc_from_m(spacing_m: f64) -> f64 {
    spacing_m / EARTH_RADIUS_M
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_schema_size_reproduces_both_real_files() {
        assert_eq!(published_schema_bytes(40_962), 56_039_332);
        assert_eq!(published_schema_bytes(163_842), 224_139_172);
    }

    #[test]
    fn the_file_id_is_eight_hex_characters_and_follows_the_contents() {
        use crate::mesh::density::MeshSpec;
        use crate::mesh::hull::{delaunay_rings, to_unit_sphere};
        let spec = MeshSpec::uniform(2000.0);
        let pts = to_unit_sphere(&crate::mesh::lloyd::seed_points(&spec, 60).unwrap()).unwrap();
        let rings = delaunay_rings(&pts).unwrap();
        let mesh = MpasMesh::derive(pts.clone(), vec![1.0; 60], &rings, 1e-3).unwrap();
        let a = mint_file_id(&mesh);
        assert_eq!(a.len(), 8, "file_id is {a}");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()), "file_id is {a}");
        let mut moved = mesh.clone();
        moved.mesh_density[0] = 0.5;
        assert_ne!(a, mint_file_id(&moved), "file_id does not follow the contents");
    }
}
