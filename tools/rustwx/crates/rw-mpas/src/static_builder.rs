//! Bounded MPAS static-field builder.
//!
//! Large geography data are processed tile-by-tile and plane-by-plane.
//! The only persistent geography-derived arrays are destination-sized
//! accumulators and the final fields.  No full global WPS source mosaic and
//! no source-pixel x destination-cell table is constructed.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::staticfile::coordframe::CoordinateRepresentation;
use crate::static_geog::{GeogDataset, SourceGeometry, SourceKind, TileRef};
use crate::static_memory::{
    checked_len, checked_vec, checked_vec_filled, checked_vec_with, plan_tile_parallelism,
    BufferLedger, MemoryReceipt, TileParallelPlan,
};
use crate::static_operators::{build_operator_fields, OperatorFields, OperatorMesh, FIFTEEN};
use rw_store::netcdf_classic::{
    NcAttr, NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef,
};

pub const MPAS_EARTH_RADIUS_M: f64 = 6_371_229.0;

/// The outermost regional boundary ring, MPAS's `nBdyLayers = 7` (five
/// relaxation plus two specified layers). Only cells, edges and vertices at
/// this mask value may carry absent-neighbour sentinels, and only pixels
/// landing on these cells take the `mpas_in_cell` containment test.
pub const REGIONAL_OUTERMOST_MASK: i32 = 7;

/// The in-memory marker for "that neighbour was culled away". Deliberately
/// `usize::MAX` so an unguarded index panics instead of reading cell 0.
pub const ABSENT_NEIGHBOR: usize = usize::MAX;
pub const MPAS_OMEGA: f64 = 7.29212e-5;

/// The schema tag the unified writer stamps into every file it writes.
pub const UNIFIED_SCHEMA_TAG: &str = "rw-mpas-static/v3-unified";

/// Model-side unit conversions, applied on top of each archive's own
/// `scale_factor`.
///
/// A dataset's `index` declares what its numbers mean and this crate's reader
/// honours that declaration. The MODEL separately expects a convention of its
/// own, and the two are not the same statement. Grading a built static against
/// a published one caught both places where they disagree, and neither would
/// have raised anything -- both are finite, both are positive, and both pass
/// every range check a validator is likely to hold:
///
/// * `greenfrac` -- the archive declares `units="fraction"` and, after its own
///   0.01 scale, delivers 0..0.97. The published static carries 0..92.07. The
///   model wants PERCENT, so honouring the archive alone hands the surface
///   scheme a vegetation fraction one hundredth of the real one, everywhere.
/// * `snoalb` -- the archive declares `units="percent"` and, after its own
///   0.01 scale, delivers 0..84. The published static carries 0..0.84. The
///   model wants a FRACTION, so honouring the archive alone hands the
///   radiation scheme a snow albedo of 84, which is not an albedo at all.
///
/// They are constants rather than branches so the next field's conversion is a
/// value and not a code path.
pub const GREENFRAC_MODEL_SCALE: f32 = 100.0;
/// See [`GREENFRAC_MODEL_SCALE`].
pub const SNOALB_MODEL_SCALE: f32 = 0.01;

#[derive(Debug, Clone)]
pub struct StaticGeogPaths {
    pub terrain: PathBuf,
    pub landuse: PathBuf,
    pub soilcat: PathBuf,
    pub greenfrac: PathBuf,
    pub albedo: PathBuf,
    pub snow_albedo: PathBuf,
    pub soil_temperature: PathBuf,
    pub soilcomp: PathBuf,
    pub soilcl1: PathBuf,
    pub soilcl2: PathBuf,
    pub soilcl3: PathBuf,
    pub soilcl4: PathBuf,
}

#[derive(Debug, Clone)]
pub struct StaticBuildConfig {
    pub grid_path: PathBuf,
    pub out_path: PathBuf,
    pub geog: StaticGeogPaths,
    pub supersample: usize,
    pub supersample_landuse: usize,
    pub supersample_30s: usize,
    pub host_memory_limit_bytes: Option<u64>,
    /// Tiles to process concurrently.  `None` takes the host CPU count,
    /// which the host-memory budget then bounds.  A build is deterministic
    /// at every setting: see [`resolve_tile_workers`].
    pub tile_workers: Option<usize>,
    pub receipt_path: Option<PathBuf>,
    pub provenance: String,
    /// The `xtime` stamp the file carries.
    pub valid_time: String,
    /// The nominal spacing to DECLARE, in metres. `None` takes the grid's own.
    ///
    /// A static's `nominalMinDc` is compared BIT FOR BIT, as FP32, against the
    /// nominal spacing a mesh binding declares: it is the scalar the
    /// gravity-wave drag length scale is rebound to, so a value that is merely
    /// close is a wrong answer wearing a right one.
    pub nominal_dx_m: Option<f64>,
    /// Refuse rather than replace an existing `out_path`.
    ///
    /// Overwriting a static silently would leave every init file and run
    /// receipt that pinned its sha256 pointing at different bytes under the
    /// same name.
    pub clobber: bool,
    /// How the fifteen coordinate arrays are stored, and what the file
    /// DECLARES it stored them as.
    ///
    /// [`CoordinateRepresentation::Binary32EarthCentred`] is the default and
    /// is what preserves the dycore byte-identity anchor: it is what native
    /// MPAS-A writes and what both published statics carry.
    /// [`CoordinateRepresentation::Binary64EarthCentred`] is legal only for a
    /// mesh with NO native counterpart, where there is nothing to be
    /// byte-identical to; `rw_mpas_static` takes it from the grid file's own
    /// declaration rather than from a flag, so the choice travels with the
    /// mesh instead of with the command line.
    pub coordinates: CoordinateRepresentation,
}

/// The `xtime` a static carries when a caller did not say.
pub const DEFAULT_VALID_TIME: &str = "2000-01-01_00:00:00";

/// Tolerance the consumer applies to the grid/static cross-check: the
/// unit-sphere angle times the earth radius must land within 0.1 % of the
/// declared physical value.
pub const NOMINAL_DX_CROSS_CHECK_RELATIVE: f64 = 1e-3;

/// Which WPS_GEOG directory serves which static field.
///
/// A TABLE, and the whole extension mechanism: a new source is a row here, not
/// a branch and not a per-dataset module. The names are the archive's own
/// directory names, which is what `gpuwm fetch-geog` stages and what the
/// door's completeness check looks for, so the three cannot drift apart.
pub const GEOG_DATASET_TABLE: [(&str, &str); 12] = [
    ("terrain", "topo_gmted2010_30s"),
    ("landuse", "modis_landuse_20class_30s_with_lakes"),
    ("soilcat", "soiltype_top_30s"),
    ("greenfrac", "greenfrac_fpar_modis"),
    ("albedo", "albedo_modis"),
    ("snow_albedo", "maxsnowalb_modis"),
    ("soil_temperature", "soiltemp_1deg"),
    ("soilcomp", "soilcomp"),
    ("soilcl1", "texture_layer1"),
    ("soilcl2", "texture_layer2"),
    ("soilcl3", "texture_layer3"),
    ("soilcl4", "texture_layer4"),
];

impl StaticGeogPaths {
    /// Resolve every geography directory from one archive root.
    pub fn under(root: &Path) -> Self {
        let at = |name: &str| -> PathBuf {
            root.join(
                GEOG_DATASET_TABLE
                    .iter()
                    .find(|(slot, _)| *slot == name)
                    .expect("every slot is a row of GEOG_DATASET_TABLE")
                    .1,
            )
        };
        Self {
            terrain: at("terrain"),
            landuse: at("landuse"),
            soilcat: at("soilcat"),
            greenfrac: at("greenfrac"),
            albedo: at("albedo"),
            snow_albedo: at("snow_albedo"),
            soil_temperature: at("soil_temperature"),
            soilcomp: at("soilcomp"),
            soilcl1: at("soilcl1"),
            soilcl2: at("soilcl2"),
            soilcl3: at("soilcl3"),
            soilcl4: at("soilcl4"),
        }
    }

    /// Every slot in declaration order, for a completeness check.
    pub fn slots(&self) -> [(&'static str, &Path); 12] {
        [
            ("terrain", &self.terrain),
            ("landuse", &self.landuse),
            ("soilcat", &self.soilcat),
            ("greenfrac", &self.greenfrac),
            ("albedo", &self.albedo),
            ("snow_albedo", &self.snow_albedo),
            ("soil_temperature", &self.soil_temperature),
            ("soilcomp", &self.soilcomp),
            ("soilcl1", &self.soilcl1),
            ("soilcl2", &self.soilcl2),
            ("soilcl3", &self.soilcl3),
            ("soilcl4", &self.soilcl4),
        ]
    }

    /// The datasets that are not there, by directory name.
    ///
    /// Checked before the mesh is read so a missing archive is a five-second
    /// refusal instead of a wasted geography pass.
    pub fn missing(&self) -> Vec<String> {
        self.slots()
            .iter()
            .filter(|(_, path)| !path.join("index").is_file())
            .map(|(_, path)| {
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string())
            })
            .collect()
    }
}

/// The requested worker count before the memory budget bounds it.
///
/// The output is byte-identical at every worker count -- every accumulator in
/// this module is an integer, and integer addition does not care what order
/// it happens in -- so this knob buys throughput and never changes an answer.
/// The determinism gate in this module's tests is what holds that true.
pub fn resolve_tile_workers(explicit: Option<usize>) -> usize {
    if let Some(n) = explicit {
        return n.max(1);
    }
    if let Ok(raw) = std::env::var("RW_MPAS_TILE_WORKERS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                return v;
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[derive(Debug, serde::Serialize)]
pub struct DatasetReceipt {
    pub field: String,
    pub path: String,
    pub tiles: usize,
    pub source_plane_bytes: u64,
    pub supersample: usize,
    pub destination_bytes: u64,
    pub elapsed_seconds: f64,
}

/// What goes INTO the static: identity, never duration and never a PATH.
///
/// Both exclusions exist for the same reason and are the same rule. A consumer
/// registry pins this file by byte count and sha256, so anything in it that
/// varies with WHEN or WHERE the build ran makes two identical builds into two
/// unregisterable files. A duration varies with the run; an absolute path
/// varies with the machine, which is worse, because a registry pin is
/// cross-machine by nature. `grid_sha256` names the grid better than a path
/// does: it survives the file being moved and it catches the file being
/// replaced. The paths themselves stay on [`StaticBuildReceipt`], which goes
/// to the `--receipt` sidecar.
///
/// The band row count is deliberately ABSENT: it is a function of the host's
/// memory budget, and the same mesh built on two boxes must be the same bytes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StaticProvenance {
    pub engine: String,
    pub schema: String,
    pub grid_sha256: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    pub sphere_radius: f64,
    pub nominal_dx_m: f64,
    pub nominal_dx_f32_bits: u32,
    pub gwd_water_category: i64,
    pub gwd_water_category_source: String,
    pub gwd_variance_smoothed_cells: usize,
    pub mminlu: String,
}

#[derive(Debug, serde::Serialize)]
pub struct StaticBuildReceipt {
    pub schema: String,
    pub grid_path: String,
    pub output_path: String,
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    pub max_edges: usize,
    pub grid_scaled_from_unit_sphere: bool,
    pub output_bytes: u64,
    pub datasets: Vec<DatasetReceipt>,
    pub parallel: TileParallelPlan,
    pub tile_map_cache: TileMapCacheReceipt,
    pub memory: MemoryReceipt,
    pub generated_fields: Vec<String>,
    pub operator_provenance: BTreeMap<String, String>,
    pub provenance: String,
    /// The document stamped into the file itself.
    pub stamped_provenance: StaticProvenance,
    pub gwd_seconds: f64,
    pub gwd_band_rows: i64,
    pub gwd_bands: usize,
    pub gwd_band_bytes: u64,
    pub gwd_variance_smoothed_cells: usize,
    /// Present when the grid carries a regional boundary zone: the counts
    /// that say exactly where this static CANNOT match a whole-sphere build,
    /// because the data needed is outside the region.
    pub regional: Option<RegionalStaticNotes>,
    /// What FP32 storage did to this file's own edge lengths, and whether the
    /// consumer's recomputation will accept them.
    pub fp32_metric_agreement: crate::staticfile::fp32metrics::Fp32MetricAgreement,
    pub sha256: String,
    pub variables: Vec<String>,
}

/// The measured regional divergences of one static build, stated so a
/// comparison against a culled whole-sphere static reads them as expected
/// rather than as defects.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RegionalStaticNotes {
    /// Cells at each bdyMaskCell value 0..7.
    pub ring_cell_counts: [usize; 8],
    /// Cells whose deriv_two side is 0 because their neighbour ring reaches a
    /// culled cell (native `atm_initialize_advection_rk` regional behaviour).
    pub deriv_two_zero_stencil_cells: usize,
    /// Outermost-ring cells left out of the GWD isolated-point smooth.
    pub gwd_smooth_skipped_boundary_cells: usize,
}

#[derive(Debug)]
struct Mesh {
    n_cells: usize,
    n_edges: usize,
    n_vertices: usize,
    max_edges: usize,
    max_edges2: usize,
    vertex_degree: usize,
    scaled_from_unit_sphere: bool,
    sphere_radius: f64,

    lat_cell: Vec<f64>,
    lon_cell: Vec<f64>,
    lat_edge: Vec<f64>,
    lon_edge: Vec<f64>,
    lat_vertex: Vec<f64>,
    lon_vertex: Vec<f64>,
    x_cell: Vec<f64>,
    y_cell: Vec<f64>,
    z_cell: Vec<f64>,
    x_edge: Vec<f64>,
    y_edge: Vec<f64>,
    z_edge: Vec<f64>,
    x_vertex: Vec<f64>,
    y_vertex: Vec<f64>,
    z_vertex: Vec<f64>,
    dc_edge: Vec<f64>,
    dv_edge: Vec<f64>,
    area_cell: Vec<f64>,
    area_triangle: Vec<f64>,
    angle_edge: Vec<f64>,
    mesh_density: Vec<f64>,
    nominal_min_dc: f64,

    n_edges_on_cell: Vec<usize>,
    cells_on_cell: Vec<usize>,
    edges_on_cell: Vec<usize>,
    vertices_on_cell: Vec<usize>,
    cells_on_edge: Vec<[usize; 2]>,
    vertices_on_edge: Vec<[usize; 2]>,

    // ---- topology as the grid file stores it, 1-BASED and VERBATIM --------
    //
    // The arrays above are the zero-based, padding-normalised views the
    // geography and operator passes index with. These are the SAME arrays as
    // the grid file wrote them, and they are what the static carries.
    //
    // The distinction is not tidiness. A consumer cross-examines grid and
    // static for bit-identity of every topology array and refuses the PAIR
    // when they differ, so a re-derived padding slot -- writing 0 where the
    // grid wrote a repeat of the last neighbour, say -- makes a static that
    // is arithmetically right and unusable. Copy, do not reconstruct.
    raw_cells_on_cell: Vec<i32>,
    raw_edges_on_cell: Vec<i32>,
    raw_vertices_on_cell: Vec<i32>,
    raw_cells_on_edge: Vec<i32>,
    raw_vertices_on_edge: Vec<i32>,
    raw_edges_on_edge: Vec<i32>,
    raw_cells_on_vertex: Vec<i32>,
    raw_edges_on_vertex: Vec<i32>,

    n_edges_on_edge: Vec<i32>,
    weights_on_edge: Vec<f64>,
    kite_areas_on_vertex: Vec<f64>,

    index_to_cell_id: Vec<i32>,
    index_to_edge_id: Vec<i32>,
    index_to_vertex_id: Vec<i32>,

    bdy_mask_cell: Vec<i32>,
    bdy_mask_edge: Vec<i32>,
    bdy_mask_vertex: Vec<i32>,
}

fn dim(file: &netcrust::File, name: &str) -> MpasResult<usize> {
    file.dimension(name)
        .map(|d| d.len())
        .ok_or_else(|| MpasError::Refusal(format!("grid has no {name} dimension")))
}

fn f64v(file: &netcrust::File, name: &str) -> MpasResult<Vec<f64>> {
    Ok(file
        .read_array_f64(name)
        .map_err(|e| MpasError::Refusal(format!("grid has no readable {name}: {e}")))?
        .into_values())
}

fn i32v(file: &netcrust::File, name: &str) -> MpasResult<Vec<i32>> {
    Ok(f64v(file, name)?
        .into_iter()
        .map(|v| v.round() as i32)
        .collect())
}

fn optional_i32v(file: &netcrust::File, name: &str, len: usize, fill: i32) -> MpasResult<Vec<i32>> {
    if file.variable(name).is_some() {
        let got = i32v(file, name)?;
        if got.len() != len {
            return Err(MpasError::Refusal(format!(
                "grid {name} has {} values, expected {len}",
                got.len()
            )));
        }
        Ok(got)
    } else {
        Ok(vec![fill; len])
    }
}

/// An integer grid array read verbatim, with its shape held to `want`.
fn shaped_i32(file: &netcrust::File, name: &str, want: usize) -> MpasResult<Vec<i32>> {
    let got = i32v(file, name)?;
    if got.len() != want {
        return Err(MpasError::Refusal(format!(
            "grid {name} has {} values, expected {want}; writing it into a static \
             would shift every later element by the difference",
            got.len()
        )));
    }
    Ok(got)
}

/// A floating grid array read verbatim, with its shape held to `want`.
fn shaped_f64(file: &netcrust::File, name: &str, want: usize) -> MpasResult<Vec<f64>> {
    let got = f64v(file, name)?;
    if got.len() != want {
        return Err(MpasError::Refusal(format!(
            "grid {name} has {} values, expected {want}; writing it into a static \
             would shift every later element by the difference",
            got.len()
        )));
    }
    Ok(got)
}

/// A global-ID array, taken from the grid when it has one and generated as
/// the 1-based identity when it does not.
///
/// Generating is safe and copying is safer. A serial mesh numbers its cells
/// 1..nCells and every published grid stores exactly that, so the identity is
/// the right answer; but a decomposed or renumbered grid carries a permutation
/// there, and re-deriving it would hand a consumer a static whose global IDs
/// disagree with the grid it is paired with.
fn identity_ids(file: &netcrust::File, name: &str, len: usize) -> MpasResult<Vec<i32>> {
    if file.variable(name).is_some() {
        return shaped_i32(file, name, len);
    }
    Ok((1..=len as i32).collect())
}

fn zero_based(raw: Vec<i32>, upper: usize, name: &str, allow_zero_padding: bool) -> MpasResult<Vec<usize>> {
    raw.into_iter()
        .enumerate()
        .map(|(k, v)| {
            if allow_zero_padding && v == 0 {
                return Ok(usize::MAX);
            }
            if v < 1 || v as usize > upper {
                return Err(MpasError::Refusal(format!(
                    "grid {name}[{k}]={v} is outside 1..={upper}"
                )));
            }
            Ok(v as usize - 1)
        })
        .collect()
}

impl Mesh {
    fn read(path: &Path) -> MpasResult<Self> {
        let f = netcrust::File::open(path)?;
        let n_cells = dim(&f, "nCells")?;
        let n_edges = dim(&f, "nEdges")?;
        let n_vertices = dim(&f, "nVertices")?;
        let max_edges = dim(&f, "maxEdges")?;
        let max_edges2 = dim(&f, "maxEdges2")?;
        let vertex_degree = dim(&f, "vertexDegree")?;

        let mut x_cell = f64v(&f, "xCell")?;
        let mut y_cell = f64v(&f, "yCell")?;
        let mut z_cell = f64v(&f, "zCell")?;
        let mut x_edge = f64v(&f, "xEdge")?;
        let mut y_edge = f64v(&f, "yEdge")?;
        let mut z_edge = f64v(&f, "zEdge")?;
        let mut x_vertex = f64v(&f, "xVertex")?;
        let mut y_vertex = f64v(&f, "yVertex")?;
        let mut z_vertex = f64v(&f, "zVertex")?;
        let mut dc_edge = f64v(&f, "dcEdge")?;
        let mut dv_edge = f64v(&f, "dvEdge")?;
        let mut area_cell = f64v(&f, "areaCell")?;
        let mut area_triangle = f64v(&f, "areaTriangle")?;
        let mut kite_areas_on_vertex = f64v(&f, "kiteAreasOnVertex")?;
        let mut nominal = f64v(&f, "nominalMinDc")?
            .first()
            .copied()
            .ok_or_else(|| MpasError::Refusal("grid nominalMinDc is empty".to_string()))?;

        let sample = x_cell
            .iter()
            .zip(&y_cell)
            .zip(&z_cell)
            .take(64)
            .map(|((&x, &y), &z)| (x * x + y * y + z * z).sqrt())
            .sum::<f64>()
            / x_cell.len().min(64).max(1) as f64;
        let scaled = sample < 10.0;
        let r = MPAS_EARTH_RADIUS_M;
        if scaled {
            for v in x_cell.iter_mut().chain(y_cell.iter_mut()).chain(z_cell.iter_mut())
                .chain(x_edge.iter_mut()).chain(y_edge.iter_mut()).chain(z_edge.iter_mut())
                .chain(x_vertex.iter_mut()).chain(y_vertex.iter_mut()).chain(z_vertex.iter_mut())
            {
                *v *= r;
            }
            for v in dc_edge.iter_mut().chain(dv_edge.iter_mut()) {
                *v *= r;
            }
            for v in area_cell
                .iter_mut()
                .chain(area_triangle.iter_mut())
                .chain(kite_areas_on_vertex.iter_mut())
            {
                *v *= r * r;
            }
            nominal *= r;
        }

        let neoc_i32 = i32v(&f, "nEdgesOnCell")?;
        if neoc_i32.len() != n_cells {
            return Err(MpasError::Refusal("nEdgesOnCell shape mismatch".to_string()));
        }
        let n_edges_on_cell: Vec<usize> = neoc_i32.into_iter().map(|v| v as usize).collect();
        let cells_on_cell_raw = i32v(&f, "cellsOnCell")?;
        let edges_on_cell_raw = i32v(&f, "edgesOnCell")?;
        let vertices_on_cell_raw = i32v(&f, "verticesOnCell")?;
        for (name, len) in [
            ("cellsOnCell", cells_on_cell_raw.len()),
            ("edgesOnCell", edges_on_cell_raw.len()),
            ("verticesOnCell", vertices_on_cell_raw.len()),
        ] {
            if len != n_cells * max_edges {
                return Err(MpasError::Refusal(format!(
                    "{name} has {len} values, expected {}",
                    n_cells * max_edges
                )));
            }
        }

        // The boundary masks come BEFORE the topology normalisation: on a
        // regional (culled) mesh the sentinel rules below are keyed off them.
        let bdy_mask_cell = optional_i32v(&f, "bdyMaskCell", n_cells, 0)?;
        let bdy_mask_edge = optional_i32v(&f, "bdyMaskEdge", n_edges, 0)?;
        let bdy_mask_vertex = optional_i32v(&f, "bdyMaskVertex", n_vertices, 0)?;

        // Padding slots in MPAS topology are often zero.  Replace them with a
        // harmless zero-based 0 only after active slots are validated.
        //
        // `absent_ok`: on a regional mesh, a VALID slot of 0 is the culler's
        // "that neighbour was culled" sentinel and is legitimate exactly on
        // outermost-ring cells (bdyMaskCell == 7). Those slots become
        // ABSENT_NEIGHBOR so every stencil that reads them has to decide, and
        // can never silently read cell 0.
        let normalize_padded = |raw: Vec<i32>,
                                upper: usize,
                                name: &str,
                                absent_ok: Option<&[i32]>|
         -> MpasResult<Vec<usize>> {
            let mut out = vec![0usize; raw.len()];
            for c in 0..n_cells {
                let used = n_edges_on_cell[c];
                if used > max_edges {
                    return Err(MpasError::Refusal(format!(
                        "cell {c} nEdgesOnCell={used} exceeds maxEdges={max_edges}"
                    )));
                }
                for s in 0..max_edges {
                    let k = c * max_edges + s;
                    if s < used {
                        let v = raw[k];
                        if v == 0 {
                            if let Some(mask) = absent_ok {
                                if mask[c] == REGIONAL_OUTERMOST_MASK {
                                    out[k] = ABSENT_NEIGHBOR;
                                    continue;
                                }
                                return Err(MpasError::Refusal(format!(
                                    "grid {name}[cell={c},slot={s}]=0 in a VALID slot of a \
                                     cell with bdyMaskCell={}; only outermost regional cells \
                                     (mask {REGIONAL_OUTERMOST_MASK}) may carry absent \
                                     neighbours -- anywhere else the mesh is torn and every \
                                     stencil over it would read a cell that does not exist",
                                    mask[c]
                                )));
                            }
                        }
                        if v < 1 || v as usize > upper {
                            return Err(MpasError::Refusal(format!(
                                "grid {name}[cell={c},slot={s}]={v} is outside 1..={upper}"
                            )));
                        }
                        out[k] = v as usize - 1;
                    }
                }
            }
            Ok(out)
        };

        let raw_cells_on_cell = cells_on_cell_raw.clone();
        let raw_edges_on_cell = edges_on_cell_raw.clone();
        let raw_vertices_on_cell = vertices_on_cell_raw.clone();
        let cells_on_cell =
            normalize_padded(cells_on_cell_raw, n_cells, "cellsOnCell", Some(&bdy_mask_cell))?;
        let edges_on_cell = normalize_padded(edges_on_cell_raw, n_edges, "edgesOnCell", None)?;
        let vertices_on_cell =
            normalize_padded(vertices_on_cell_raw, n_vertices, "verticesOnCell", None)?;

        let coe_raw = i32v(&f, "cellsOnEdge")?;
        let voe_raw = i32v(&f, "verticesOnEdge")?;
        if coe_raw.len() != 2 * n_edges || voe_raw.len() != 2 * n_edges {
            return Err(MpasError::Refusal("edge-pair topology shape mismatch".to_string()));
        }
        let raw_cells_on_edge = coe_raw.clone();
        let raw_vertices_on_edge = voe_raw.clone();
        // A rim edge of a regional mesh has ONE cell; the culler stores 0 in
        // the other slot, and only on outermost (mask 7) edges.
        let mut coe = vec![0usize; 2 * n_edges];
        for e in 0..n_edges {
            for s in 0..2 {
                let v = raw_cells_on_edge[2 * e + s];
                if v == 0 {
                    if bdy_mask_edge[e] == REGIONAL_OUTERMOST_MASK {
                        coe[2 * e + s] = ABSENT_NEIGHBOR;
                        continue;
                    }
                    return Err(MpasError::Refusal(format!(
                        "grid cellsOnEdge[edge={e},slot={s}]=0 on an edge with \
                         bdyMaskEdge={}; only outermost regional edges (mask \
                         {REGIONAL_OUTERMOST_MASK}) are one-sided -- anywhere else \
                         the mesh is torn",
                        bdy_mask_edge[e]
                    )));
                }
                if v < 1 || v as usize > n_cells {
                    return Err(MpasError::Refusal(format!(
                        "grid cellsOnEdge[{}]={v} is outside 1..={n_cells}",
                        2 * e + s
                    )));
                }
                coe[2 * e + s] = v as usize - 1;
            }
        }
        let voe = zero_based(voe_raw, n_vertices, "verticesOnEdge", false)?;
        let cells_on_edge = (0..n_edges).map(|e| [coe[2 * e], coe[2 * e + 1]]).collect();
        let vertices_on_edge = (0..n_edges).map(|e| [voe[2 * e], voe[2 * e + 1]]).collect();

        // The mesh DUAL, read verbatim. A static that omits it leaves the
        // dycore with no vertex triangles: `cellsOnVertex` and
        // `kiteAreasOnVertex` are what the vorticity and the potential-
        // vorticity flux are built on, and `weightsOnEdge` / `edgesOnEdge`
        // are the tangential reconstruction stencil itself.
        let raw_edges_on_edge = shaped_i32(&f, "edgesOnEdge", n_edges * max_edges2)?;
        let raw_cells_on_vertex = shaped_i32(&f, "cellsOnVertex", n_vertices * vertex_degree)?;
        let raw_edges_on_vertex = shaped_i32(&f, "edgesOnVertex", n_vertices * vertex_degree)?;
        let n_edges_on_edge = shaped_i32(&f, "nEdgesOnEdge", n_edges)?;
        let weights_on_edge = shaped_f64(&f, "weightsOnEdge", n_edges * max_edges2)?;
        if kite_areas_on_vertex.len() != n_vertices * vertex_degree {
            return Err(MpasError::Refusal(format!(
                "grid kiteAreasOnVertex has {} values, expected {}",
                kite_areas_on_vertex.len(),
                n_vertices * vertex_degree
            )));
        }

        let mesh = Self {
            n_cells,
            n_edges,
            n_vertices,
            max_edges,
            max_edges2,
            vertex_degree,
            scaled_from_unit_sphere: scaled,
            sphere_radius: r,
            lat_cell: f64v(&f, "latCell")?,
            lon_cell: f64v(&f, "lonCell")?,
            lat_edge: f64v(&f, "latEdge")?,
            lon_edge: f64v(&f, "lonEdge")?,
            lat_vertex: f64v(&f, "latVertex")?,
            lon_vertex: f64v(&f, "lonVertex")?,
            x_cell,
            y_cell,
            z_cell,
            x_edge,
            y_edge,
            z_edge,
            x_vertex,
            y_vertex,
            z_vertex,
            dc_edge,
            dv_edge,
            area_cell,
            area_triangle,
            angle_edge: f64v(&f, "angleEdge")?,
            mesh_density: f64v(&f, "meshDensity")?,
            nominal_min_dc: nominal,
            n_edges_on_cell,
            cells_on_cell,
            edges_on_cell,
            vertices_on_cell,
            cells_on_edge,
            vertices_on_edge,
            raw_cells_on_cell,
            raw_edges_on_cell,
            raw_vertices_on_cell,
            raw_cells_on_edge,
            raw_vertices_on_edge,
            raw_edges_on_edge,
            raw_cells_on_vertex,
            raw_edges_on_vertex,
            n_edges_on_edge,
            weights_on_edge,
            kite_areas_on_vertex,
            index_to_cell_id: identity_ids(&f, "indexToCellID", n_cells)?,
            index_to_edge_id: identity_ids(&f, "indexToEdgeID", n_edges)?,
            index_to_vertex_id: identity_ids(&f, "indexToVertexID", n_vertices)?,
            bdy_mask_cell,
            bdy_mask_edge,
            bdy_mask_vertex,
        };
        mesh.validate()?;
        Ok(mesh)
    }

    fn validate(&self) -> MpasResult<()> {
        for (name, len, want) in [
            ("latCell", self.lat_cell.len(), self.n_cells),
            ("lonCell", self.lon_cell.len(), self.n_cells),
            ("latEdge", self.lat_edge.len(), self.n_edges),
            ("lonEdge", self.lon_edge.len(), self.n_edges),
            ("latVertex", self.lat_vertex.len(), self.n_vertices),
            ("lonVertex", self.lon_vertex.len(), self.n_vertices),
        ] {
            if len != want {
                return Err(MpasError::Refusal(format!(
                    "grid {name} has {len} values, expected {want}"
                )));
            }
        }
        // Regional (limited-area) admission. The mpas_in_cell containment the
        // old blanket bdyMaskCell==7 refusal named as missing is ported now
        // (see `point_in_cell` and its use in `tile_map`), so a regional mesh
        // is admitted -- against the sentinel geometry the culler declares,
        // with every violation named.
        for (name, mask) in [
            ("bdyMaskCell", &self.bdy_mask_cell),
            ("bdyMaskEdge", &self.bdy_mask_edge),
            ("bdyMaskVertex", &self.bdy_mask_vertex),
        ] {
            if let Some((i, &v)) = mask
                .iter()
                .enumerate()
                .find(|&(_, &v)| !(0..=REGIONAL_OUTERMOST_MASK).contains(&v))
            {
                return Err(MpasError::Refusal(format!(
                    "{name}[{i}]={v} is outside 0..={REGIONAL_OUTERMOST_MASK}; the \
                     boundary-zone stages would index a relaxation ring that does \
                     not exist"
                )));
            }
        }
        let regional = self.bdy_mask_cell.iter().any(|&v| v != 0);
        if regional {
            // edgesOnCell, verticesOnCell and verticesOnEdge never carry a
            // zero valid slot on ANY mesh -- every edge and vertex of a kept
            // cell is kept by the cull -- and the reader above already
            // refused them by name. What is left to check here are the raw
            // dual rows the reader carries verbatim.
            //
            // The sentinel-bearing dual rows: zeros are legitimate only on
            // outermost (mask 7) elements. Anywhere else the boundary zone
            // would start inside the relaxation rings the dycore nudges.
            for e in 0..self.n_edges {
                let used = (self.n_edges_on_edge[e].max(0) as usize).min(self.max_edges2);
                let row = &self.raw_edges_on_edge[e * self.max_edges2..e * self.max_edges2 + used];
                if row.iter().any(|&v| v == 0) && self.bdy_mask_edge[e] != REGIONAL_OUTERMOST_MASK {
                    return Err(MpasError::Refusal(format!(
                        "regional grid edgesOnEdge row {e} carries a 0 inside its \
                         declared nEdgesOnEdge={used} but bdyMaskEdge={}; absent \
                         reconstruction stencils belong to mask-{REGIONAL_OUTERMOST_MASK} \
                         edges only",
                        self.bdy_mask_edge[e]
                    )));
                }
            }
            for vx in 0..self.n_vertices {
                let base = vx * self.vertex_degree;
                let cz = self.raw_cells_on_vertex[base..base + self.vertex_degree]
                    .iter()
                    .any(|&v| v == 0);
                let ez = self.raw_edges_on_vertex[base..base + self.vertex_degree]
                    .iter()
                    .any(|&v| v == 0);
                if (cz || ez) && self.bdy_mask_vertex[vx] != REGIONAL_OUTERMOST_MASK {
                    return Err(MpasError::Refusal(format!(
                        "regional grid vertex {vx} has absent adjacent cells/edges but \
                         bdyMaskVertex={}; one-sided vertices belong to \
                         mask-{REGIONAL_OUTERMOST_MASK} only",
                        self.bdy_mask_vertex[vx]
                    )));
                }
            }
        }
        Ok(())
    }

    /// True when this mesh carries a regional boundary zone.
    fn is_regional(&self) -> bool {
        self.bdy_mask_cell.iter().any(|&v| v != 0)
    }

    fn operator_view(&self) -> OperatorMesh<'_> {
        OperatorMesh {
            n_cells: self.n_cells,
            n_edges: self.n_edges,
            max_edges: self.max_edges,
            sphere_radius: self.sphere_radius,
            n_edges_on_cell: &self.n_edges_on_cell,
            cells_on_cell: &self.cells_on_cell,
            edges_on_cell: &self.edges_on_cell,
            cells_on_edge: &self.cells_on_edge,
            vertices_on_cell: &self.vertices_on_cell,
            vertices_on_edge: &self.vertices_on_edge,
            x_cell: &self.x_cell,
            y_cell: &self.y_cell,
            z_cell: &self.z_cell,
            x_vertex: &self.x_vertex,
            y_vertex: &self.y_vertex,
            z_vertex: &self.z_vertex,
            angle_edge: &self.angle_edge,
            dc_edge: &self.dc_edge,
        }
    }

    fn max_search_distance2(&self) -> f64 {
        let mut max_r = 0.0f64;
        for c in 0..self.n_cells {
            for s in 0..self.n_edges_on_cell[c] {
                let v = self.vertices_on_cell[c * self.max_edges + s];
                let dot = (
                    self.x_cell[c] * self.x_vertex[v]
                        + self.y_cell[c] * self.y_vertex[v]
                        + self.z_cell[c] * self.z_vertex[v]
                ) / (self.sphere_radius * self.sphere_radius);
                let arc = self.sphere_radius * dot.clamp(-1.0, 1.0).acos();
                max_r = max_r.max(arc);
            }
        }
        let diameter = 2.0 * max_r;
        2.0 * diameter * diameter
    }
}

/// What a grid file declares, through the same read-and-validate path the
/// static builder itself uses.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GridProbe {
    pub n_cells: usize,
    pub n_edges: usize,
    pub n_vertices: usize,
    /// True when the grid carries a regional boundary zone.
    pub regional: bool,
    /// Cells at each bdyMaskCell value 0..7. All in slot 0 for a global mesh.
    pub ring_cell_counts: [usize; 8],
    /// cellsOnCell valid slots pointing at culled cells.
    pub absent_neighbor_slots: usize,
    /// Edges with one culled side.
    pub one_sided_edges: usize,
}

/// Admit or refuse a grid through the static builder's own reader, WITHOUT
/// touching any geography. This is the probe a door runs before hours of
/// tile streaming: a regional grid whose sentinel geometry is torn is refused
/// here with the breakage named, instead of after the archive was read.
pub fn probe_grid(path: &Path) -> MpasResult<GridProbe> {
    let mesh = Mesh::read(path)?;
    let mut ring_cell_counts = [0usize; 8];
    for &m in &mesh.bdy_mask_cell {
        ring_cell_counts[m as usize] += 1;
    }
    let mut absent_neighbor_slots = 0usize;
    for c in 0..mesh.n_cells {
        for s in 0..mesh.n_edges_on_cell[c] {
            if mesh.cells_on_cell[c * mesh.max_edges + s] == ABSENT_NEIGHBOR {
                absent_neighbor_slots += 1;
            }
        }
    }
    let one_sided_edges = (0..mesh.n_edges)
        .filter(|&e| mesh.cells_on_edge[e].contains(&ABSENT_NEIGHBOR))
        .count();
    Ok(GridProbe {
        n_cells: mesh.n_cells,
        n_edges: mesh.n_edges,
        n_vertices: mesh.n_vertices,
        regional: mesh.is_regional(),
        ring_cell_counts,
        absent_neighbor_slots,
        one_sided_edges,
    })
}

// ---------------------------------------------------------------------------
// 3-D kd tree; one compact permutation and split-axis array.
// ---------------------------------------------------------------------------

struct KdTree {
    points: Vec<[f64; 3]>,
    order: Vec<u32>,
    axis: Vec<u8>,
}

fn kd_build(points: &[[f64; 3]], order: &mut [u32], axis: &mut [u8], lo: usize, hi: usize) {
    if hi <= lo {
        return;
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &idx in &order[lo..hi] {
        for k in 0..3 {
            min[k] = min[k].min(points[idx as usize][k]);
            max[k] = max[k].max(points[idx as usize][k]);
        }
    }
    let mut ax = 0usize;
    if max[1] - min[1] > max[ax] - min[ax] {
        ax = 1;
    }
    if max[2] - min[2] > max[ax] - min[ax] {
        ax = 2;
    }
    let mid = lo + (hi - lo) / 2;
    order[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
        points[a as usize][ax].total_cmp(&points[b as usize][ax])
    });
    axis[mid] = ax as u8;
    kd_build(points, order, axis, lo, mid);
    kd_build(points, order, axis, mid + 1, hi);
}

impl KdTree {
    fn new(mesh: &Mesh) -> Self {
        let points: Vec<[f64; 3]> = (0..mesh.n_cells)
            .map(|c| [mesh.x_cell[c], mesh.y_cell[c], mesh.z_cell[c]])
            .collect();
        let mut order: Vec<u32> = (0..mesh.n_cells as u32).collect();
        let mut axis = vec![0u8; order.len()];
        let n = order.len();
        kd_build(&points, &mut order, &mut axis, 0, n);
        Self { points, order, axis }
    }

    fn nearest(&self, target: [f64; 3]) -> (usize, f64) {
        let mut best = (usize::MAX, f64::INFINITY);
        self.search(0, self.order.len(), target, &mut best);
        best
    }

    fn search(&self, lo: usize, hi: usize, t: [f64; 3], best: &mut (usize, f64)) {
        if hi <= lo {
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let idx = self.order[mid] as usize;
        let p = self.points[idx];
        let d = (p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2) + (p[2] - t[2]).powi(2);
        if d < best.1 {
            *best = (idx, d);
        }
        let a = self.axis[mid] as usize;
        let delta = t[a] - p[a];
        let (nlo, nhi, flo, fhi) = if delta < 0.0 {
            (lo, mid, mid + 1, hi)
        } else {
            (mid + 1, hi, lo, mid)
        };
        self.search(nlo, nhi, t, best);
        if delta * delta < best.1 {
            self.search(flo, fhi, t, best);
        }
    }
}

fn xyz(radius: f64, lat: f64, lon: f64) -> [f64; 3] {
    let c = lat.cos();
    [radius * c * lon.cos(), radius * c * lon.sin(), radius * lat.sin()]
}

/// MPAS's `mpas_arc_length` verbatim: `r * 2 * asin(|b-a| / (2r))` with
/// `r = |a|`. Kept literal -- no clamp -- because the containment decisions
/// below are graded against the native routine's answers.
#[inline]
fn native_arc_length(a: [f64; 3], b: [f64; 3]) -> f64 {
    let r = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let c = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
    r * 2.0 * (c / (2.0 * r)).asin()
}

/// MPAS's `mpas_in_cell` (mpas_geometry_utils.F, v8.4.1), the containment
/// test the old blanket regional refusal named as unported.
///
/// A point is inside a Voronoi cell when it is no closer to the MIRROR of the
/// cell's generating point -- reflected across the great circle through each
/// pair of adjacent cell vertices -- than to the generating point itself. On
/// a regional mesh this is what keeps a source pixel whose true owner was
/// culled from being accumulated into the outermost ring cell the kd search
/// falls back to.
fn point_in_cell(mesh: &Mesh, cell: usize, p: [f64; 3]) -> bool {
    let cc = [mesh.x_cell[cell], mesh.y_cell[cell], mesh.z_cell[cell]];
    let radius = (cc[0] * cc[0] + cc[1] * cc[1] + cc[2] * cc[2]).sqrt();
    let inv = 1.0 / radius;
    let cc_unit = [cc[0] * inv, cc[1] * inv, cc[2] * inv];
    let in_dist = native_arc_length(p, cc);
    let ne = mesh.n_edges_on_cell[cell];
    for i in 0..ne {
        let v1 = mesh.vertices_on_cell[cell * mesh.max_edges + i];
        let v2 = mesh.vertices_on_cell[cell * mesh.max_edges + (i + 1) % ne];
        let a = [
            mesh.x_vertex[v1] * inv,
            mesh.y_vertex[v1] * inv,
            mesh.z_vertex[v1] * inv,
        ];
        let b = [
            mesh.x_vertex[v2] * inv,
            mesh.y_vertex[v2] * inv,
            mesh.z_vertex[v2] * inv,
        ];
        // `mpas_mirror_point`: reflect the generating point across the great
        // circle through (a, b) -- rotate it by twice the angle at `a`
        // between the arcs a->point and a->b, about the axis through `a`.
        let alpha = crate::static_operators::sphere_angle(a, cc_unit, b);
        let m = crate::mesh::cull::rotate_about_vector(cc_unit, a, 2.0 * alpha);
        let mirror = [m[0] * radius, m[1] * radius, m[2] * radius];
        if native_arc_length(p, mirror) < in_dist {
            return false;
        }
    }
    true
}

/// Destination cell for every supersampled subpixel of one tile interior.
/// Map layout is `[interior_pixel * factor^2 + subpixel]`.
/// The supersample factor squares into the per-tile destination map, so it is
/// the one knob that can turn a bounded tile into an unbounded allocation.
fn validate_supersample(factor: usize) -> MpasResult<()> {
    if factor == 0 || factor > 16 {
        return Err(MpasError::Refusal(format!(
            "supersample factor {factor} is outside supported 1..=16"
        )));
    }
    Ok(())
}

/// A conservative reject sphere around the whole cell cloud.
///
/// On a GLOBAL mesh every pixel is near some cell and the kd search prunes
/// well. On a REGIONAL mesh most of the planet's pixels are far from EVERY
/// cell, and a far query defeats kd pruning -- the best distance and every
/// split-plane distance are the same magnitude, so the search visits most of
/// the tree, per pixel, for ~96% of the globe. Measured on the first
/// regional build: 13 CPU-hours into the first dataset with no end in sight.
///
/// A pixel farther than `centroid_dist` from the cloud centroid cannot be
/// within the search radius of any cell (triangle inequality in chord
/// space), so it is the same `-1` the kd search would have produced, decided
/// in nanoseconds. Strictly conservative: never rejects a pixel the search
/// would have kept.
struct FarReject {
    centroid: [f64; 3],
    /// `(R_cloud + sqrt(max_distance2))^2`, in chord metres squared.
    reject_d2: f64,
}

impl FarReject {
    fn new(mesh: &Mesh, max_distance2: f64) -> Self {
        let n = mesh.n_cells.max(1) as f64;
        let mut c = [0.0f64; 3];
        for i in 0..mesh.n_cells {
            c[0] += mesh.x_cell[i];
            c[1] += mesh.y_cell[i];
            c[2] += mesh.z_cell[i];
        }
        c = [c[0] / n, c[1] / n, c[2] / n];
        let mut r_cloud2 = 0.0f64;
        for i in 0..mesh.n_cells {
            let d2 = (mesh.x_cell[i] - c[0]).powi(2)
                + (mesh.y_cell[i] - c[1]).powi(2)
                + (mesh.z_cell[i] - c[2]).powi(2);
            r_cloud2 = r_cloud2.max(d2);
        }
        let reach = r_cloud2.sqrt() + max_distance2.max(0.0).sqrt();
        FarReject {
            centroid: c,
            reject_d2: reach * reach,
        }
    }

    #[inline]
    fn certainly_out(&self, p: [f64; 3]) -> bool {
        let d2 = (p[0] - self.centroid[0]).powi(2)
            + (p[1] - self.centroid[1]).powi(2)
            + (p[2] - self.centroid[2]).powi(2);
        d2 > self.reject_d2
    }
}

fn tile_map(
    mesh: &Mesh,
    tree: &KdTree,
    max_distance2: f64,
    ds: &GeogDataset,
    tile: &TileRef,
    factor: usize,
) -> MpasResult<Vec<i32>> {
    validate_supersample(factor)?;
    let far = FarReject::new(mesh, max_distance2);
    let pixels = (ds.index.tile_x as usize)
        .checked_mul(ds.index.tile_y as usize)
        .and_then(|v| v.checked_mul(factor * factor))
        .ok_or_else(|| MpasError::Refusal("tile-map element count overflow".to_string()))?;
    let mut out = Vec::new();
    out.try_reserve_exact(pixels).map_err(|e| {
        MpasError::Refusal(format!(
            "host memory allocation failed for tile destination map: {pixels} i32 values: {e}"
        ))
    })?;
    for j in 0..ds.index.tile_y as usize {
        for i in 0..ds.index.tile_x as usize {
            let x_center = tile.xs as f64 + i as f64;
            let y_center = tile.ys as f64 + j as f64;
            for sj in 0..factor {
                for si in 0..factor {
                    let x = x_center - 0.5 + (si as f64 + 0.5) / factor as f64;
                    let y = y_center - 0.5 + (sj as f64 + 0.5) / factor as f64;
                    let (lat, lon) = ds.source_xy_to_latlon(x, y);
                    let p = xyz(mesh.sphere_radius, lat, lon);
                    if far.certainly_out(p) {
                        out.push(-1);
                        continue;
                    }
                    let (cell, d2) = tree.nearest(p);
                    // The native regional rule (init_atm_static, nBdyLayers=7):
                    // a pixel is accepted unconditionally at mask < 7, and on
                    // an outermost-ring cell only when it is geometrically
                    // INSIDE that cell -- otherwise its true owner was culled
                    // and the pixel belongs to nobody in this mesh.
                    let accepted = d2 <= max_distance2
                        && (mesh.bdy_mask_cell[cell] != REGIONAL_OUTERMOST_MASK
                            || point_in_cell(mesh, cell, p));
                    out.push(if accepted { cell as i32 } else { -1 });
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Reusable tile maps, and the bounded pool that fills them.
// ---------------------------------------------------------------------------

/// A destination map identified by what actually determines it.
///
/// Not by dataset name: the 30-arcsec products share tile geometry, so nine
/// datasets would otherwise pay for the same kd search nine times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TileMapKey {
    geometry: SourceGeometry,
    xs: i64,
    ys: i64,
    factor: usize,
}

/// A cached map, run-length encoded when that is smaller.
///
/// A destination cell of a global mesh spans many source pixels, so a scan
/// line crosses few cells and the raw map is enormously redundant.  Storing
/// the runs is what lets every tile of every shared-geometry dataset stay
/// cached inside a budget the admission gate has actually granted.  When a
/// mesh is fine enough that runs stop paying -- source pixels coarser than
/// destination cells -- the raw form is kept instead, so the encoding can
/// never cost more than not having it.
#[derive(Debug)]
enum CachedTileMap {
    Runs(Vec<(i32, u32)>),
    Raw(Vec<i32>),
}

impl CachedTileMap {
    fn encode(map: &[i32]) -> MpasResult<Self> {
        let mut runs: Vec<(i32, u32)> = Vec::new();
        let raw_cap = map.len() / 2 + 1; // where runs stop being smaller
        for &c in map {
            match runs.last_mut() {
                Some((cell, len)) if *cell == c && *len < u32::MAX => *len += 1,
                _ => {
                    if runs.len() >= raw_cap {
                        return Ok(CachedTileMap::Raw(map.to_vec()));
                    }
                    runs.push((c, 1));
                }
            }
        }
        Ok(CachedTileMap::Runs(runs))
    }

    fn bytes(&self) -> u64 {
        match self {
            CachedTileMap::Runs(r) => (r.len() * std::mem::size_of::<(i32, u32)>()) as u64,
            CachedTileMap::Raw(v) => (v.len() * 4) as u64,
        }
    }

    fn decode(&self, len: usize) -> MpasResult<Vec<i32>> {
        match self {
            CachedTileMap::Raw(v) => Ok(v.clone()),
            CachedTileMap::Runs(runs) => {
                let mut out =
                    checked_vec::<i32>(len, "geog-tile-map", "cached tile destination map")?;
                let mut at = 0usize;
                for &(cell, run) in runs {
                    let end = at + run as usize;
                    out[at..end].fill(cell);
                    at = end;
                }
                debug_assert_eq!(at, len);
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TileMapCacheReceipt {
    pub budget_bytes: u64,
    pub stored_bytes: u64,
    pub stored_maps: usize,
    pub hits: u64,
    pub misses: u64,
    /// Maps recomputed because storing them would have passed the budget.
    pub declined_for_budget: u64,
}

#[derive(Default)]
struct TileMapCacheInner {
    maps: HashMap<TileMapKey, CachedTileMap>,
    bytes: u64,
    hits: u64,
    misses: u64,
    declined: u64,
}

/// Tile maps kept between datasets, inside a granted byte budget.
///
/// A miss is only ever a cost in time: the map is a pure function of the
/// mesh, the kd tree and the geometry key, so a recomputed map and a cached
/// one are the same bytes.  That is why running the budget out changes how
/// long a build takes and never what it produces.
struct TileMapCache {
    budget_bytes: u64,
    inner: Mutex<TileMapCacheInner>,
}

impl TileMapCache {
    fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            inner: Mutex::new(TileMapCacheInner::default()),
        }
    }

    fn lock(&self) -> MpasResult<std::sync::MutexGuard<'_, TileMapCacheInner>> {
        self.inner.lock().map_err(|_| {
            MpasError::Refusal(
                "the tile-map cache lock was poisoned by a panicking tile worker; \
                 the build cannot continue without knowing which tiles were mapped"
                    .to_string(),
            )
        })
    }

    fn get(&self, key: &TileMapKey, len: usize) -> MpasResult<Option<Vec<i32>>> {
        let mut inner = self.lock()?;
        match inner.maps.get(key) {
            Some(entry) => {
                let decoded = entry.decode(len)?;
                inner.hits += 1;
                Ok(Some(decoded))
            }
            None => {
                inner.misses += 1;
                Ok(None)
            }
        }
    }

    fn insert(&self, key: TileMapKey, map: &[i32]) -> MpasResult<()> {
        if self.budget_bytes == 0 {
            return Ok(());
        }
        let encoded = CachedTileMap::encode(map)?;
        let bytes = encoded.bytes();
        let mut inner = self.lock()?;
        if inner.maps.contains_key(&key) {
            return Ok(());
        }
        if inner.bytes + bytes > self.budget_bytes {
            inner.declined += 1;
            return Ok(());
        }
        inner.bytes += bytes;
        inner.maps.insert(key, encoded);
        Ok(())
    }

    fn receipt(&self) -> MpasResult<TileMapCacheReceipt> {
        let inner = self.lock()?;
        Ok(TileMapCacheReceipt {
            budget_bytes: self.budget_bytes,
            stored_bytes: inner.bytes,
            stored_maps: inner.maps.len(),
            hits: inner.hits,
            misses: inner.misses,
            declined_for_budget: inner.declined,
        })
    }
}

/// The mesh-side state every geography pass needs, plus the pool and cache.
struct GeogContext<'a> {
    mesh: &'a Mesh,
    tree: &'a KdTree,
    max_d2: f64,
    workers: usize,
    pool: Option<&'a rayon::ThreadPool>,
    cache: &'a TileMapCache,
}

impl GeogContext<'_> {
    /// The destination map for one tile, from the cache when it is there.
    fn tile_map(&self, ds: &GeogDataset, tile: &TileRef, factor: usize) -> MpasResult<Vec<i32>> {
        validate_supersample(factor)?;
        let len = (ds.index.tile_x as usize)
            .checked_mul(ds.index.tile_y as usize)
            .and_then(|v| v.checked_mul(factor * factor))
            .ok_or_else(|| MpasError::Refusal("tile-map element count overflow".to_string()))?;
        let key = TileMapKey {
            geometry: ds.geometry(),
            xs: tile.xs,
            ys: tile.ys,
            factor,
        };
        if let Some(hit) = self.cache.get(&key, len)? {
            return Ok(hit);
        }
        let map = tile_map(self.mesh, self.tree, self.max_d2, ds, tile, factor)?;
        self.cache.insert(key, &map)?;
        Ok(map)
    }

    /// Run `body` over every tile, on the admitted number of workers.
    ///
    /// Failures are collected and the lowest-indexed one is returned, so the
    /// refusal a user reads names the same tile on every run rather than
    /// whichever worker happened to finish first.
    fn for_each_tile<F>(&self, tiles: &[TileRef], body: F) -> MpasResult<()>
    where
        F: Fn(&TileRef) -> MpasResult<()> + Sync + Send,
    {
        let pool = match self.pool {
            Some(pool) if self.workers > 1 && tiles.len() > 1 => pool,
            _ => {
                for tile in tiles {
                    body(tile)?;
                }
                return Ok(());
            }
        };
        let results: Vec<MpasResult<()>> =
            pool.install(|| tiles.par_iter().map(&body).collect());
        for result in results {
            result?;
        }
        Ok(())
    }
}

/// Add a run of equal-cell contributions with one atomic operation.
///
/// Consecutive source pixels overwhelmingly land in the same destination
/// cell, so folding the run before touching the shared accumulator is what
/// keeps the parallel loop from being a queue at a handful of cache lines.
/// The fold is integer addition, which is what makes it free of consequences.
#[inline]
fn flush_i64(target: &[AtomicI64], index: usize, acc: i64) {
    if acc != 0 {
        target[index].fetch_add(acc, Ordering::Relaxed);
    }
}

#[inline]
fn flush_u64(target: &[AtomicU64], index: usize, acc: u64) {
    if acc != 0 {
        target[index].fetch_add(acc, Ordering::Relaxed);
    }
}

#[inline]
fn plane_raw_at(ds: &GeogDataset, plane: &[i64], i: usize, j: usize) -> i64 {
    let b = ds.index.tile_bdr as usize;
    let nx = ds.index.full_tile_nx();
    plane[(j + b) * nx + (i + b)]
}

fn categorical(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
) -> MpasResult<(Vec<i32>, i64, i64, GeogDataset)> {
    let mesh = ctx.mesh;
    let ds = GeogDataset::open(path)?;
    if ds.index.kind != SourceKind::Categorical || ds.index.nz() != 1 {
        return Err(MpasError::Refusal(format!(
            "{} must be a one-plane categorical WPS_GEOG dataset",
            path.display()
        )));
    }
    let cmin = ds.index.category_min.ok_or_else(|| {
        MpasError::Refusal(format!("{} lacks category_min", path.display()))
    })?;
    let cmax = ds.index.category_max.ok_or_else(|| {
        MpasError::Refusal(format!("{} lacks category_max", path.display()))
    })?;
    if cmax < cmin || cmax - cmin > 255 {
        return Err(MpasError::Refusal(format!(
            "{} has unsupported category range {cmin}..{cmax}",
            path.display()
        )));
    }
    let ncat = (cmax - cmin + 1) as usize;
    let counts = checked_vec_with(
        checked_len(mesh.n_cells, ncat, "categorical", "category counts")?,
        "categorical",
        "category counts",
        || AtomicU32::new(0),
    )?;

    ctx.for_each_tile(&ds.tiles, |tile| {
        let map = ctx.tile_map(&ds, tile, factor)?;
        let plane = ds.read_plane(tile, 0)?;
        let sub = factor * factor;
        // One atomic per (cell, category) run rather than per subpixel.
        let mut run_slot = usize::MAX;
        let mut run_len = 0u32;
        for j in 0..ds.index.tile_y as usize {
            for i in 0..ds.index.tile_x as usize {
                let raw = plane_raw_at(&ds, &plane, i, j);
                if raw < cmin || raw > cmax {
                    continue;
                }
                let cat = (raw - cmin) as usize;
                let p = j * ds.index.tile_x as usize + i;
                for q in 0..sub {
                    let c = map[p * sub + q];
                    if c < 0 {
                        continue;
                    }
                    let slot = c as usize * ncat + cat;
                    if slot != run_slot {
                        if run_len != 0 {
                            counts[run_slot].fetch_add(run_len, Ordering::Relaxed);
                        }
                        run_slot = slot;
                        run_len = 0;
                    }
                    run_len += 1;
                }
            }
        }
        if run_len != 0 {
            counts[run_slot].fetch_add(run_len, Ordering::Relaxed);
        }
        Ok(())
    })?;

    let mut out =
        checked_vec_filled::<i32>(mesh.n_cells, cmin as i32, "categorical", "category field")?;
    for c in 0..mesh.n_cells {
        let row = &counts[c * ncat..(c + 1) * ncat];
        let row: Vec<u32> = row.iter().map(|v| v.load(Ordering::Relaxed)).collect();
        let mut best = 0usize;
        for k in 1..ncat {
            if row[k] > row[best] {
                best = k;
            }
        }
        if row.iter().all(|&v| v == 0) {
            return Err(MpasError::Refusal(format!(
                "{} mapped no valid category pixels to cell {c}; refusing silent category fill",
                path.display()
            )));
        }
        out[c] = (cmin + best as i64) as i32;
    }
    Ok((out, cmin, cmax, ds))
}

/// Carry a value to every cell that has none, across the mesh's own
/// neighbour graph.
///
/// THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-27, node-2).  A swath the
/// placement layer put on a winter storm at 66.4 S 162.5 E generated a
/// 128,019-cell mesh in 435 s and then lost its static outright:
/// `albedo_modis mapped no valid pixels to cell 111276`.  The archive is not
/// broken and the mesh is not broken.  MODIS surface albedo is a LAND-ONLY
/// product, and the Antarctic sea-ice margin is the one band on earth where
/// the land-use archive calls a cell ice -- so the model's own landmask says
/// land -- while every albedo pixel inside it is fill.  Measured in the
/// archive: from 55 S to 65 S the albedo planes are 100 % fill over the
/// Southern Ocean at every longitude sampled, and the continent's own pixels
/// resume around 67.5 S.  On a 96 km parent, and on the 4.6 km grid this
/// project placed at 60.1 S, no cell fell entirely inside that band; at
/// 66.4 S one did, and one cell cost the whole grid.
///
/// `needed` are the cells that must end with a value.  `conduit` are cells
/// that may CARRY one without keeping it -- the mask-excluded cells, which is
/// what lets an ice margin separated from the continent by open water still
/// be reached.  Returns the conduits that were used, so the caller can put
/// them back to the value they had.
///
/// The walk stops as soon as every `needed` cell holds a value: conducting
/// across the rest of an ocean after the last one is served is work with no
/// output.  A round settles every cell it can BEFORE any of them becomes a
/// source, so the spread is a breadth-first ring and what a cell receives does
/// not depend on the order cells are visited.
///
/// `Err` carries the cells that could not be reached at all, which is the case
/// the refusal was really written for: a field with no valid pixel anywhere a
/// value could come from.
#[allow(clippy::too_many_arguments)]
fn spread_to_unsampled(
    out: &mut [f32],
    filled: &mut [bool],
    nz: usize,
    needed: &[usize],
    conduit: Vec<usize>,
    n_cells: usize,
    max_edges: usize,
    n_edges_on_cell: &[usize],
    cells_on_cell: &[usize],
) -> Result<Vec<usize>, Vec<usize>> {
    let mut wants = needed.len();
    if wants == 0 {
        return Ok(Vec::new());
    }
    let is_conduit = {
        let mut flags = vec![false; n_cells];
        for &c in &conduit {
            flags[c] = true;
        }
        flags
    };
    let mut pending: Vec<usize> = needed.iter().copied().chain(conduit).collect();
    let mut used: Vec<usize> = Vec::new();
    let mut accumulator = vec![0.0f64; nz];
    let mut settled: Vec<usize> = Vec::new();
    while wants > 0 {
        let mut still: Vec<usize> = Vec::with_capacity(pending.len());
        settled.clear();
        for &c in &pending {
            accumulator.iter_mut().for_each(|v| *v = 0.0);
            let mut n = 0usize;
            for s in 0..n_edges_on_cell[c] {
                let k = cells_on_cell[c * max_edges + s];
                if k == ABSENT_NEIGHBOR || k >= n_cells || !filled[k] {
                    continue;
                }
                for z in 0..nz {
                    accumulator[z] += out[k * nz + z] as f64;
                }
                n += 1;
            }
            if n == 0 {
                still.push(c);
                continue;
            }
            for z in 0..nz {
                out[c * nz + z] = (accumulator[z] / n as f64) as f32;
            }
            settled.push(c);
        }
        if settled.is_empty() {
            return Err(still);
        }
        for &c in &settled {
            filled[c] = true;
            if is_conduit[c] {
                used.push(c);
            } else {
                wants -= 1;
            }
        }
        if still.is_empty() {
            break;
        }
        pending = still;
    }
    Ok(used)
}

fn continuous_mean(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
    landmask: Option<&[i32]>,
    missing_zero: bool,
) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let mesh = ctx.mesh;
    let ds = GeogDataset::open(path)?;
    if ds.index.kind != SourceKind::Continuous {
        return Err(MpasError::Refusal(format!(
            "{} must be continuous", path.display()
        )));
    }
    let nz = ds.index.nz();
    let slots = checked_len(mesh.n_cells, nz, "continuous", "plane accumulator")?;
    let sums = checked_vec_with(slots, "continuous", "plane accumulator", || AtomicI64::new(0))?;
    let count =
        checked_vec_with(mesh.n_cells, "continuous", "sample counts", || AtomicU64::new(0))?;

    ctx.for_each_tile(&ds.tiles, |tile| {
        let map = ctx.tile_map(&ds, tile, factor)?;
        let sub = factor * factor;
        // Count destination mappings once; all planes share the same footprint.
        let mut run_cell = -1i32;
        let mut run_len = 0u64;
        for &c in &map {
            if c != run_cell {
                if run_cell >= 0 {
                    flush_u64(&count, run_cell as usize, run_len);
                }
                run_cell = c;
                run_len = 0;
            }
            if c >= 0 && landmask.map_or(true, |m| m[c as usize] != 0) {
                run_len += 1;
            }
        }
        if run_cell >= 0 {
            flush_u64(&count, run_cell as usize, run_len);
        }
        for z in 0..nz {
            let plane = ds.read_plane(tile, z)?;
            let mut run_slot = usize::MAX;
            let mut acc = 0i64;
            for j in 0..ds.index.tile_y as usize {
                for i in 0..ds.index.tile_x as usize {
                    let raw = plane_raw_at(&ds, &plane, i, j);
                    let missing = ds.index.missing_value == Some(raw);
                    if missing && !missing_zero {
                        continue;
                    }
                    let raw = if missing { 0 } else { raw };
                    let p = j * ds.index.tile_x as usize + i;
                    for q in 0..sub {
                        let c = map[p * sub + q];
                        if c < 0 || landmask.map_or(false, |m| m[c as usize] == 0) {
                            continue;
                        }
                        let slot = c as usize * nz + z;
                        if slot != run_slot {
                            if run_slot != usize::MAX {
                                flush_i64(&sums, run_slot, acc);
                            }
                            run_slot = slot;
                            acc = 0;
                        }
                        acc += raw;
                    }
                }
            }
            if run_slot != usize::MAX {
                flush_i64(&sums, run_slot, acc);
            }
            drop(plane);
        }
        Ok(())
    })?;
    let sums: Vec<i64> = sums.iter().map(|v| v.load(Ordering::Relaxed)).collect();
    let count: Vec<u64> = count.iter().map(|v| v.load(Ordering::Relaxed)).collect();
    let mut out = checked_vec::<f32>(slots, "continuous", "destination field")?;
    // A cell is FILLED when it holds a value: either its own average, or one
    // carried to it across the mesh below.
    let mut filled = checked_vec::<bool>(mesh.n_cells, "continuous", "fill state")?;
    let mut needed: Vec<usize> = Vec::new();
    let mut conduit: Vec<usize> = Vec::new();
    for c in 0..mesh.n_cells {
        if count[c] == 0 {
            // A cell the mask excludes was never sampled on purpose and keeps
            // the zero it was allocated with, exactly as before.
            if landmask.map_or(false, |m| m[c] == 0) {
                conduit.push(c);
            } else {
                needed.push(c);
            }
            continue;
        }
        filled[c] = true;
        for z in 0..nz {
            out[c * nz + z] =
                (sums[c * nz + z] as f64 / count[c] as f64 * ds.index.scale_factor) as f32;
        }
    }

    // THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-27, node-2).  A swath the
    // placement layer put on a winter storm at 66.4 S 162.5 E generated a
    // 128,019-cell mesh in 435 s and then lost its static outright:
    // `albedo_modis mapped no valid pixels to cell 111276`.  The archive is
    // not broken and the mesh is not broken.  MODIS surface albedo is a
    // LAND-ONLY product, and the Antarctic sea-ice margin is the one band on
    // earth where the land-use archive calls a cell ice -- so the model's own
    // landmask says land -- while every albedo pixel inside it is fill.
    // Measured in the archive: at 55 S through 65 S the albedo planes are
    // 100 % fill over the Southern Ocean at every longitude sampled, and the
    // continent's own pixels resume around 67.5 S.  On a 96 km parent, and on
    // the 4.6 km grid this project placed at 60.1 S, no single cell fell
    // entirely inside that band; at 66.4 S one did, and one cell cost the
    // whole grid.
    //
    // A cell with no sample of its own now takes the mean of the neighbours
    // that have one, spreading outward over the mesh's own cellsOnCell until
    // every cell that needs a value has one.  Excluded cells conduct a value
    // without keeping it, so an ice margin separated from the continent by
    // open water is still reached.  The refusal is not removed -- it is moved
    // to the case it was actually written for, a field with no valid pixel
    // ANYWHERE that a value could come from, and it now says how many cells
    // and where one of them is.
    //
    // Nothing that builds today changes: this path is only entered where the
    // build previously returned an error and produced no static at all.
    if !needed.is_empty() {
        let carried = spread_to_unsampled(
            &mut out,
            &mut filled,
            nz,
            &needed,
            conduit,
            mesh.n_cells,
            mesh.max_edges,
            &mesh.n_edges_on_cell,
            &mesh.cells_on_cell,
        )
        .map_err(|stranded| {
            let example = stranded
                .iter()
                .copied()
                .find(|&c| needed.contains(&c))
                .unwrap_or(stranded[0]);
            MpasError::Refusal(format!(
                "{} mapped no valid pixels to {} cell(s), and no neighbour of theirs has                  a value to carry -- cell {example} at {:.4} N {:.4} E is one. Their whole                  connected region of the mesh is outside the archive's coverage, so a value                  there would be invented rather than carried",
                path.display(),
                stranded.len(),
                mesh.lat_cell[example].to_degrees(),
                mesh.lon_cell[example].to_degrees()
            ))
        })?;
        // A conduit keeps nothing: it goes back to the zero it was allocated
        // with, so a mask-excluded cell reads exactly as it did before.
        for c in carried {
            for z in 0..nz {
                out[c * nz + z] = 0.0;
            }
        }
    }
    Ok((out, ds))
}

fn soilcomp(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
    landmask: &[i32],
) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let mesh = ctx.mesh;
    let ds = GeogDataset::open(path)?;
    let nz = ds.index.nz();
    if nz == 0 {
        return Err(MpasError::Refusal(format!("{} has no soil components", path.display())));
    }
    let slots = checked_len(mesh.n_cells, nz, "soilcomp", "component accumulator")?;
    let sums = checked_vec_with(slots, "soilcomp", "component accumulator", || AtomicI64::new(0))?;
    let count =
        checked_vec_with(mesh.n_cells, "soilcomp", "sample counts", || AtomicU64::new(0))?;

    ctx.for_each_tile(&ds.tiles, |tile| {
        let map = ctx.tile_map(&ds, tile, factor)?;
        let mut planes = Vec::with_capacity(nz);
        for z in 0..nz {
            planes.push(ds.read_plane(tile, z)?);
        }
        let sub = factor * factor;
        // A whole soil column is folded per destination-cell run, so one run
        // costs nz+1 atomics instead of (nz+1) per subpixel.
        let mut run_cell = usize::MAX;
        let mut acc = vec![0i64; nz];
        let mut acc_count = 0u64;
        for j in 0..ds.index.tile_y as usize {
            for i in 0..ds.index.tile_x as usize {
                let first = plane_raw_at(&ds, &planes[0], i, j);
                if ds.index.missing_value == Some(first) {
                    continue;
                }
                let p = j * ds.index.tile_x as usize + i;
                for q in 0..sub {
                    let c = map[p * sub + q];
                    if c < 0 || landmask[c as usize] == 0 {
                        continue;
                    }
                    let c = c as usize;
                    if c != run_cell {
                        if run_cell != usize::MAX {
                            for (z, &value) in acc.iter().enumerate() {
                                flush_i64(&sums, run_cell * nz + z, value);
                            }
                            flush_u64(&count, run_cell, acc_count);
                        }
                        run_cell = c;
                        acc.iter_mut().for_each(|v| *v = 0);
                        acc_count = 0;
                    }
                    for (slot, plane) in acc.iter_mut().zip(&planes) {
                        *slot += plane_raw_at(&ds, plane, i, j);
                    }
                    acc_count += 1;
                }
            }
        }
        if run_cell != usize::MAX {
            for (z, &value) in acc.iter().enumerate() {
                flush_i64(&sums, run_cell * nz + z, value);
            }
            flush_u64(&count, run_cell, acc_count);
        }
        Ok(())
    })?;

    let sums: Vec<i64> = sums.iter().map(|v| v.load(Ordering::Relaxed)).collect();
    let count: Vec<u64> = count.iter().map(|v| v.load(Ordering::Relaxed)).collect();
    let mut out = checked_vec::<f32>(slots, "soilcomp", "destination field")?;
    for c in 0..mesh.n_cells {
        if landmask[c] == 0 || count[c] == 0 {
            continue;
        }
        for z in 0..nz {
            out[c * nz + z] =
                (sums[c * nz + z] as f64 / count[c] as f64 * ds.index.scale_factor) as f32;
        }
    }
    Ok((out, ds))
}

fn full_plane(ds: &GeogDataset, z: usize, max_bytes: u64) -> MpasResult<Vec<f64>> {
    let bytes = (ds.nx_global as u64)
        .checked_mul(ds.ny_global as u64)
        .and_then(|v| v.checked_mul(8))
        .ok_or_else(|| MpasError::Refusal("full-plane byte size overflow".to_string()))?;
    if bytes > max_bytes {
        return Err(MpasError::Refusal(format!(
            "{} needs {bytes} bytes for direct interpolation, past bounded full-plane ceiling {max_bytes}; \
             this path is reserved for small climatology grids such as soiltemp_1deg",
            ds.path.display()
        )));
    }
    let mut out = checked_vec_filled::<f64>(
        checked_len(
            ds.nx_global as usize,
            ds.ny_global as usize,
            "full-plane",
            "climatology plane",
        )?,
        f64::NAN,
        "full-plane",
        "climatology plane",
    )?;
    for tile in &ds.tiles {
        let p = ds.read_plane(tile, z)?;
        for j in 0..ds.index.tile_y as usize {
            for i in 0..ds.index.tile_x as usize {
                let x = tile.xs - 1 + i as i64;
                let y = tile.ys - 1 + j as i64;
                if x < 0 || y < 0 || x >= ds.nx_global || y >= ds.ny_global {
                    continue;
                }
                let raw = plane_raw_at(ds, &p, i, j);
                if let Some(v) = ds.raw_to_value(raw) {
                    out[y as usize * ds.nx_global as usize + x as usize] = v;
                }
            }
        }
    }
    Ok(out)
}

fn grid_at(ds: &GeogDataset, a: &[f64], j: i64, i: i64) -> f64 {
    let mut ii = i;
    if ds.wraps_x {
        ii = ii.rem_euclid(ds.nx_global);
    }
    if ii < 0 || ii >= ds.nx_global || j < 0 || j >= ds.ny_global {
        return f64::NAN;
    }
    a[j as usize * ds.nx_global as usize + ii as usize]
}

fn sample_small_regular(ds: &GeogDataset, a: &[f64], x1: f64, y1: f64) -> f64 {
    // Source coordinates are 1-based; convert to zero-based continuous.
    let x = x1 - 1.0;
    let y = y1 - 1.0;
    let i0 = x.floor() as i64;
    let i1 = x.ceil() as i64;
    let j0 = y.floor() as i64;
    let j1 = y.ceil() as i64;
    let wx = x - i0 as f64;
    let wy = y - j0 as f64;
    let v = [
        grid_at(ds, a, j0, i0),
        grid_at(ds, a, j0, i1),
        grid_at(ds, a, j1, i0),
        grid_at(ds, a, j1, i1),
    ];
    if v.iter().all(|v| v.is_finite()) {
        return (1.0 - wy) * ((1.0 - wx) * v[0] + wx * v[1])
            + wy * ((1.0 - wx) * v[2] + wx * v[3]);
    }

    let mut sum = 0.0;
    let mut n = 0;
    for &vv in &v {
        if vv.is_finite() {
            sum += vv;
            n += 1;
        }
    }
    if n > 0 {
        return sum / n as f64;
    }

    // WPS-style expanding search, deterministic ring order.
    let ci = x.round() as i64;
    let cj = y.round() as i64;
    for r in 1..=1200i64 {
        let mut best: Option<(f64, f64)> = None;
        for dj in -r..=r {
            for di in -r..=r {
                if di.abs() != r && dj.abs() != r {
                    continue;
                }
                let vv = grid_at(ds, a, cj + dj, ci + di);
                if vv.is_finite() {
                    let d2 = (di as f64).powi(2) + (dj as f64).powi(2);
                    if best.map_or(true, |b| d2 < b.0) {
                        best = Some((d2, vv));
                    }
                }
            }
        }
        if let Some((_, vv)) = best {
            return vv;
        }
    }
    f64::NAN
}

fn soil_temperature(mesh: &Mesh, path: &Path, landmask: &[i32]) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let ds = GeogDataset::open(path)?;
    if ds.index.nz() != 1 {
        return Err(MpasError::Refusal(format!(
            "{} soil temperature must have one plane",
            path.display()
        )));
    }
    let plane = full_plane(&ds, 0, 64 * 1024 * 1024)?;
    let mut out = checked_vec::<f32>(mesh.n_cells, "soiltemp", "destination field")?;
    for c in 0..mesh.n_cells {
        if landmask[c] == 0 {
            continue;
        }
        let (x, mut y) = ds.latlon_rad_to_source_xy(mesh.lat_cell[c], mesh.lon_cell[c]);
        y = y.clamp(1.0, ds.ny_global as f64);
        let v = sample_small_regular(&ds, &plane, x, y);
        if !v.is_finite() {
            return Err(MpasError::Refusal(format!(
                "{} could not interpolate soil temperature for cell {c}",
                path.display()
            )));
        }
        out[c] = v as f32;
    }
    Ok((out, ds))
}

fn snow_albedo(
    ctx: &GeogContext<'_>,
    path: &Path,
    factor: usize,
    landmask: &[i32],
) -> MpasResult<(Vec<f32>, GeogDataset)> {
    let (mut v, ds) = continuous_mean(ctx, path, factor, Some(landmask), false)?;
    if ds.index.nz() != 1 {
        return Err(MpasError::Refusal(format!(
            "{} snow albedo must have one plane",
            path.display()
        )));
    }
    // Archive percent -> model fraction. See [`SNOALB_MODEL_SCALE`].
    for x in &mut v {
        *x *= SNOALB_MODEL_SCALE;
    }
    Ok((v, ds))
}

/// Bytes the tile-map cache would like, capped so the ask stays sane.
///
/// The cache pays for itself on the 30-arcsec products, which share tile
/// geometry: without a cap the ask on a global mesh is the raw size of every
/// unique map at once, tens of gigabytes, most of which the run-length
/// encoding never needs.  The cap keeps the *predicted peak* -- and therefore
/// the admission gate -- from being dominated by a term the cache will not
/// actually use, while still leaving room for every map a global build needs.
///
/// MEASURED 2026-08-26 (stale-guard audit 2026-08-25, tile-cache
/// adjudication), full-geography global builds, receipts beside the packs:
///
/// * 12,002 cells: cache stored 93.5 MiB encoded (1,962 maps, 0 declined);
///   predicted peak 8.42 GiB, OS peak working set 2.73 GB.
/// * 654,432 cells: cache stored 608 MiB encoded (1,962 maps, 0 declined);
///   predicted peak 8.94 GiB, OS peak working set 4.48 GB.
///
/// So the cap is LIVE as the cache's budget -- encoded use grows roughly
/// with the square root of cell count (~2.4 GiB extrapolated at 10 M
/// cells), and 4 GiB holds headroom for every global build in reach while
/// `declined_for_budget` in the receipt makes an overflow visible.  What
/// the measurements also show, recorded as a named follow-up rather than
/// fixed by guess: charging the FULL cap into `predicted_peak_bytes`
/// over-prices a global build by ~2-3x against the measured peak, because
/// the raw-sum ask saturates the cap while the encoded reality is 2-14% of
/// it.  A measured-scaling prediction term needs a third point in the
/// multi-million-cell class before it can be fit.
const TILE_MAP_CACHE_CAP_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// One dataset's scratch and its share of the reusable-map ask.
struct DatasetFootprint {
    /// What one worker holds while it is inside this dataset's tile loop.
    tile_bytes: u64,
    /// Raw bytes of this dataset's tile maps, before dedup and encoding.
    map_bytes: u64,
    geometry: SourceGeometry,
    factor: usize,
}

fn dataset_footprint(path: &Path, factor: usize, planes_held: usize) -> MpasResult<DatasetFootprint> {
    validate_supersample(factor)?;
    let ds = GeogDataset::open(path)?;
    let sub = (factor * factor) as u64;
    // encoded bytes + the decoded i64 plane(s) the pass holds at once
    let encoded = ds.plane_bytes();
    let decoded = ds.index.plane_words() as u64 * 8 * planes_held.max(1) as u64;
    let one_map = ds.index.tile_x as u64 * ds.index.tile_y as u64 * sub * 4;
    Ok(DatasetFootprint {
        // The map itself, plus the transient its cache encoding costs while
        // it is being built.
        tile_bytes: encoded + decoded + 2 * one_map,
        map_bytes: one_map * ds.tiles.len() as u64,
        geometry: ds.geometry(),
        factor,
    })
}

/// The resident, per-worker and cache terms the admission gate plans against.
fn build_footprint(mesh: &Mesh, cfg: &StaticBuildConfig) -> MpasResult<(u64, u64, u64, u64)> {
    let resident = ((mesh.n_cells * (12 + 8 + 4) * 8)
        + (mesh.n_cells * mesh.max_edges * 4 * 4)
        + (mesh.n_edges * 2 * FIFTEEN * 4)
        + (mesh.n_edges * 32)
        + (mesh.n_vertices * 32)) as u64;

    let g = &cfg.geog;
    // `planes_held`: every pass streams one plane at a time except soilcomp,
    // which needs the whole soil column of a tile resident together.
    let soilcomp_planes = GeogDataset::open(&g.soilcomp)?.index.nz();
    let specs: [(&PathBuf, usize, usize); 11] = [
        (&g.terrain, cfg.supersample, 1),
        (&g.landuse, cfg.supersample_landuse, 1),
        (&g.soilcat, cfg.supersample, 1),
        (&g.greenfrac, cfg.supersample_30s, 1),
        (&g.albedo, cfg.supersample_30s, 1),
        (&g.snow_albedo, cfg.supersample_30s, 1),
        (&g.soilcomp, cfg.supersample_30s, soilcomp_planes),
        (&g.soilcl1, cfg.supersample_30s, 1),
        (&g.soilcl2, cfg.supersample_30s, 1),
        (&g.soilcl3, cfg.supersample_30s, 1),
        (&g.soilcl4, cfg.supersample_30s, 1),
    ];

    let mut per_worker = 0u64;
    let mut unique: BTreeMap<(u64, usize), u64> = BTreeMap::new();
    for (path, factor, planes) in specs {
        let f = dataset_footprint(path, factor, planes)?;
        per_worker = per_worker.max(f.tile_bytes);
        // Datasets sharing a geometry key share their maps, so their raw map
        // bytes are counted once, not once per dataset.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&f.geometry, &mut hasher);
        let geometry_hash = std::hash::Hasher::finish(&hasher);
        let slot = unique.entry((geometry_hash, f.factor)).or_insert(0);
        *slot = (*slot).max(f.map_bytes);
    }
    let cache_request = unique
        .values()
        .fold(0u64, |a, b| a.saturating_add(*b))
        .min(TILE_MAP_CACHE_CAP_BYTES);

    // NetCDF writer temporarily encodes the largest slab.  `deriv_two` is
    // normally the largest field generated here.
    let writer_slab = (mesh.n_edges * 2 * FIFTEEN * 4) as u64;
    Ok((resident, per_worker, writer_slab, cache_request))
}

// Output helpers ------------------------------------------------------------


fn f32_from_f64(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}



/// Every variable this writer declares, in declaration order.
///
/// Spelled out so the schema gate can read it without building a file. The
/// `declared_variables_match_the_writer` test below holds it to what
/// [`write_static`] actually emits, so it cannot drift into a comfortable
/// fiction; `staticfile::schema::gate_bounded_writer_matches_pin` holds it to
/// the schema the mesh registry pins.
///
/// THE ORDER IS THE PINNED ORDER, then the extras. The first 69 names are
/// [`crate::staticfile::schema::PINNED_STATIC_VARIABLES`] in exactly the
/// sequence the published statics carry them, so a reader that walks the file
/// linearly meets the same fields in the same places it always has. The 13
/// that follow are the ones this builder adds: the four deformation/gradient
/// operator tables, the Noah-MP soil-composition group, and the land-use
/// aliases and category scalars a surface scheme reads by name.
pub fn declared_variables() -> &'static [&'static str] {
    &[
        // -- the pinned 69, in published order --------------------------
        "xtime",
        "latCell",
        "lonCell",
        "xCell",
        "yCell",
        "zCell",
        "indexToCellID",
        "latEdge",
        "lonEdge",
        "xEdge",
        "yEdge",
        "zEdge",
        "indexToEdgeID",
        "latVertex",
        "lonVertex",
        "xVertex",
        "yVertex",
        "zVertex",
        "indexToVertexID",
        "cellsOnEdge",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "edgesOnCell",
        "edgesOnEdge",
        "weightsOnEdge",
        "dvEdge",
        "dcEdge",
        "angleEdge",
        "areaCell",
        "areaTriangle",
        "cellsOnCell",
        "verticesOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "kiteAreasOnVertex",
        "meshDensity",
        "nominalMinDc",
        "bdyMaskCell",
        "bdyMaskEdge",
        "bdyMaskVertex",
        "edgeNormalVectors",
        "localVerticalUnitVectors",
        "cellTangentPlane",
        "coeffs_reconstruct",
        "deriv_two",
        "fEdge",
        "fVertex",
        "ter",
        "landmask",
        "mminlu",
        "ivgtyp",
        "isltyp",
        "snoalb",
        "soiltemp",
        "greenfrac",
        "shdmin",
        "shdmax",
        "albedo12m",
        "var2d",
        "con",
        "oa1",
        "oa2",
        "oa3",
        "oa4",
        "ol1",
        "ol2",
        "ol3",
        "ol4",
        // -- the 13 this builder adds -----------------------------------
        "cell_gradient_coef_x",
        "cell_gradient_coef_y",
        "defc_a",
        "defc_b",
        "lu_index",
        "soilcat_top",
        "soilcomp",
        "soilcl1",
        "soilcl2",
        "soilcl3",
        "soilcl4",
        "isice_lu",
        "iswater_lu",
    ]
}

/// The geography group, as the streaming passes produced it.
///
/// Grouped rather than passed as nineteen arguments because the writer's job
/// is to lay this out, not to remember which of nineteen slices was which.
struct GeographyFields {
    ter: Vec<f32>,
    landmask: Vec<i32>,
    ivgtyp: Vec<i32>,
    isltyp: Vec<i32>,
    /// `[cell][month]`, in PERCENT. See [`GREENFRAC_MODEL_SCALE`].
    greenfrac: Vec<f32>,
    /// `[cell][month]`, in percent, as the archive delivers it.
    albedo12m: Vec<f32>,
    /// A FRACTION, not a percentage. See [`SNOALB_MODEL_SCALE`].
    snoalb: Vec<f32>,
    soiltemp: Vec<f32>,
    soilcomp: Vec<f32>,
    soilcl: [Vec<i32>; 4],
    isice_lu: i32,
    iswater_lu: i32,
    mminlu: String,
}

/// Length of the `StrLen` dimension the published statics carry.
const STR_LEN: usize = 64;
/// Months in the seasonal fields.
const N_MONTHS: usize = 12;

/// Write the unified static.
///
/// The layout is the PUBLISHED one: the same dimensions, the same variable
/// order and the same attributes the mesh registry's pinned pair carries,
/// followed by the thirteen fields this builder adds. Nothing here is a
/// convenient re-spelling -- a consumer reads the file by name and refuses on
/// a shape it did not expect, and the registry pins it by byte count.
fn write_static(
    cfg: &StaticBuildConfig,
    mesh: &Mesh,
    operators: &OperatorFields,
    geo: &GeographyFields,
    gwd: &crate::static_gwd::GwdFields,
    nominal_dx_m: f64,
    provenance_json: &str,
) -> MpasResult<()> {
    if geo.greenfrac.len() != mesh.n_cells * N_MONTHS
        || geo.albedo12m.len() != mesh.n_cells * N_MONTHS
    {
        return Err(MpasError::Refusal(
            "monthly static fields must have exactly 12 planes".to_string(),
        ));
    }
    let nsoil = if mesh.n_cells == 0 { 0 } else { geo.soilcomp.len() / mesh.n_cells };
    if nsoil == 0 || geo.soilcomp.len() != mesh.n_cells * nsoil {
        return Err(MpasError::Refusal("soilcomp shape is invalid".to_string()));
    }
    for (name, got, want) in [
        ("var2d", gwd.var2d.len(), mesh.n_cells),
        ("con", gwd.con.len(), mesh.n_cells),
    ] {
        if got != want {
            return Err(MpasError::Refusal(format!(
                "the drag band's {name} has {got} values, expected {want}"
            )));
        }
    }

    let (nc, ne, nv) = (mesh.n_cells, mesh.n_edges, mesh.n_vertices);
    let (me, me2, vd) = (mesh.max_edges, mesh.max_edges2, mesh.vertex_degree);

    let dims = vec![
        NcDim::fixed("StrLen", STR_LEN),
        NcDim::record("Time"),
        NcDim::fixed("nCells", nc),
        NcDim::fixed("nEdges", ne),
        NcDim::fixed("nVertices", nv),
        NcDim::fixed("TWO", 2),
        NcDim::fixed("maxEdges", me),
        NcDim::fixed("maxEdges2", me2),
        NcDim::fixed("vertexDegree", vd),
        NcDim::fixed("R3", 3),
        NcDim::fixed("nMonths", N_MONTHS),
        NcDim::fixed("FIFTEEN", FIFTEEN),
        NcDim::fixed("nSoilComps", nsoil),
    ];
    let d = |name: &str| -> usize {
        dims.iter()
            .position(|x| x.name == name)
            .expect("dimension declared above")
    };
    let (d_str, d_time, d_c, d_e, d_v) =
        (d("StrLen"), d("Time"), d("nCells"), d("nEdges"), d("nVertices"));
    let (d_two, d_me, d_me2, d_vd, d_r3, d_mon) = (
        d("TWO"),
        d("maxEdges"),
        d("maxEdges2"),
        d("vertexDegree"),
        d("R3"),
        d("nMonths"),
    );
    let (d_fifteen, d_soil) = (d("FIFTEEN"), d("nSoilComps"));

    let flt = |name: &str, dd: Vec<usize>, units: &str, long: &str| -> NcVarDef {
        NcVarDef::new(name, NcType::Float, dd).with_attrs(vec![
            NcAttr::text("units", units),
            NcAttr::text("long_name", long),
        ])
    };
    // The FIFTEEN coordinate arrays -- the three lat/lon/xyz families -- are
    // the only ones whose type follows `cfg.coordinates`.  They move together
    // because the port cross-checks xyz against the lat/lon pair beside it and
    // tightens its comparison the moment it sees binary64 metrics; a binary64
    // `xCell` beside a binary32 `latCell` fails the load.  Every METRIC
    // (dcEdge, dvEdge, areaCell, areaTriangle, kiteAreasOnVertex,
    // weightsOnEdge) stays binary32 whatever the coordinates do: a stored
    // metric is a relative quantity that binary32 carries to 6e-8 at any
    // length, and leaving them alone keeps the port's `metric_rtol` on its
    // 2.0e-5 binary32 branch -- the branch every published mesh is judged on.
    let coord = |name: &str, dd: Vec<usize>, units: &str, long: &str| -> NcVarDef {
        NcVarDef::new(name, cfg.coordinates.nc_type(), dd).with_attrs(vec![
            NcAttr::text("units", units),
            NcAttr::text("long_name", long),
        ])
    };
    let int = |name: &str, dd: Vec<usize>, long: &str| -> NcVarDef {
        NcVarDef::new(name, NcType::Int, dd)
            .with_attrs(vec![NcAttr::text("units", "-"), NcAttr::text("long_name", long)])
    };

    let mut vars = vec![
        NcVarDef::new("xtime", NcType::Char, vec![d_time, d_str]).with_attrs(vec![
            NcAttr::text("units", "YYYY-MM-DD_hh:mm:ss"),
            NcAttr::text("long_name", "Model valid time"),
        ]),
        coord("latCell", vec![d_c], "rad", "Latitude of cells"),
        coord("lonCell", vec![d_c], "rad", "Longitude of cells"),
        coord("xCell", vec![d_c], "m", "Cartesian x-coordinate of cells"),
        coord("yCell", vec![d_c], "m", "Cartesian y-coordinate of cells"),
        coord("zCell", vec![d_c], "m", "Cartesian z-coordinate of cells"),
        int("indexToCellID", vec![d_c], "Mapping from local array index to global cell ID"),
        coord("latEdge", vec![d_e], "rad", "Latitude of edges"),
        coord("lonEdge", vec![d_e], "rad", "Longitude of edges"),
        coord("xEdge", vec![d_e], "m", "Cartesian x-coordinate of edges"),
        coord("yEdge", vec![d_e], "m", "Cartesian y-coordinate of edges"),
        coord("zEdge", vec![d_e], "m", "Cartesian z-coordinate of edges"),
        int("indexToEdgeID", vec![d_e], "Mapping from local array index to global edge ID"),
        coord("latVertex", vec![d_v], "rad", "Latitude of vertices"),
        coord("lonVertex", vec![d_v], "rad", "Longitude of vertices"),
        coord("xVertex", vec![d_v], "m", "Cartesian x-coordinate of vertices"),
        coord("yVertex", vec![d_v], "m", "Cartesian y-coordinate of vertices"),
        coord("zVertex", vec![d_v], "m", "Cartesian z-coordinate of vertices"),
        int("indexToVertexID", vec![d_v], "Mapping from local array index to global vertex ID"),
        int("cellsOnEdge", vec![d_e, d_two], "IDs of cells divided by an edge"),
        int("nEdgesOnCell", vec![d_c], "Number of edges forming the boundary of a cell"),
        int("nEdgesOnEdge", vec![d_e], "Number of edges involved in reconstruction"),
        int("edgesOnCell", vec![d_c, d_me], "IDs of edges forming the boundary of a cell"),
        int("edgesOnEdge", vec![d_e, d_me2], "IDs of edges used in reconstruction"),
        flt("weightsOnEdge", vec![d_e, d_me2], "-", "Weights used in reconstruction"),
        flt("dvEdge", vec![d_e], "m", "Spherical distance between vertices"),
        flt("dcEdge", vec![d_e], "m", "Spherical distance between cells"),
        flt("angleEdge", vec![d_e], "rad", "Angle between local north and the edge normal"),
        flt("areaCell", vec![d_c], "m^2", "Spherical area of a Voronoi cell"),
        flt("areaTriangle", vec![d_v], "m^2", "Spherical area of a Delaunay triangle"),
        int("cellsOnCell", vec![d_c, d_me], "IDs of cells neighbouring a cell"),
        int("verticesOnCell", vec![d_c, d_me], "IDs of vertices bounding a cell"),
        int("verticesOnEdge", vec![d_e, d_two], "IDs of the vertices bounding an edge"),
        int("edgesOnVertex", vec![d_v, d_vd], "IDs of the edges meeting at a vertex"),
        int("cellsOnVertex", vec![d_v, d_vd], "IDs of the cells meeting at a vertex"),
        flt("kiteAreasOnVertex", vec![d_v, d_vd], "m^2", "Intersection area of a cell and a triangle"),
        flt("meshDensity", vec![d_c], "-", "Value of the density function used to generate the mesh"),
        NcVarDef::new("nominalMinDc", NcType::Float, vec![]).with_attrs(vec![
            NcAttr::text("units", "m"),
            NcAttr::text("long_name", "Nominal minimum dcEdge value where meshDensity == 1.0"),
        ]),
        int("bdyMaskCell", vec![d_c], "Limited-area boundary relaxation zone of a cell"),
        int("bdyMaskEdge", vec![d_e], "Limited-area boundary relaxation zone of an edge"),
        int("bdyMaskVertex", vec![d_v], "Limited-area boundary relaxation zone of a vertex"),
        flt("edgeNormalVectors", vec![d_e, d_r3], "-", "Cartesian components of the edge normal"),
        flt("localVerticalUnitVectors", vec![d_c, d_r3], "-", "Cartesian components of the local vertical"),
        flt("cellTangentPlane", vec![d_c, d_two, d_r3], "-", "Cartesian basis of a cell's tangent plane"),
        flt(
            "coeffs_reconstruct",
            vec![d_c, d_me, d_r3],
            "-",
            "Coefficients for reconstructing a cell-centre vector from its \
             edge normal components (bitwise +0 placeholder; a consumer \
             overlays the exact values from the init file)",
        ),
        flt(
            "deriv_two",
            vec![d_e, d_two, d_fifteen],
            "unitless",
            "weights for cell-centered second derivative, normal to edge, for \
             transport scheme",
        ),
        flt("fEdge", vec![d_e], "s^-1", "Coriolis parameter at an edge"),
        flt("fVertex", vec![d_v], "s^-1", "Coriolis parameter at a vertex"),
        flt("ter", vec![d_c], "m", "Terrain height"),
        int("landmask", vec![d_c], "Land mask, 1 for land and 0 for water"),
        NcVarDef::new("mminlu", NcType::Char, vec![d_str])
            .with_attrs(vec![NcAttr::text("long_name", "Land use dataset")]),
        int("ivgtyp", vec![d_c], "Dominant land use category"),
        int("isltyp", vec![d_c], "Dominant soil category"),
        flt("snoalb", vec![d_c], "-", "Annual maximum snow albedo"),
        flt("soiltemp", vec![d_c], "K", "Deep-soil temperature"),
        flt("greenfrac", vec![d_c, d_mon], "-", "Monthly climatological green-ness fraction"),
        flt("shdmin", vec![d_c], "-", "Minimum climatological green-ness fraction"),
        flt("shdmax", vec![d_c], "-", "Maximum climatological green-ness fraction"),
        flt("albedo12m", vec![d_c, d_mon], "-", "Monthly climatological background albedo"),
    ];
    for (name, long) in [
        ("var2d", "Standard deviation of sub-grid orography"),
        ("con", "Orographic convexity"),
        ("oa1", "Orographic asymmetry, westward"),
        ("oa2", "Orographic asymmetry, southward"),
        ("oa3", "Orographic asymmetry, south-westward"),
        ("oa4", "Orographic asymmetry, north-westward"),
        ("ol1", "Orographic effective length, westward"),
        ("ol2", "Orographic effective length, southward"),
        ("ol3", "Orographic effective length, south-westward"),
        ("ol4", "Orographic effective length, north-westward"),
    ] {
        vars.push(flt(name, vec![d_c], "-", long));
    }
    // The thirteen this builder adds beyond the published manifest.
    vars.extend([
        flt("cell_gradient_coef_x", vec![d_c, d_me], "m^-1", "Zonal cell-gradient weights"),
        flt("cell_gradient_coef_y", vec![d_c, d_me], "m^-1", "Meridional cell-gradient weights"),
        flt("defc_a", vec![d_c, d_me], "m^-1", "Deformation weights, symmetric part"),
        flt("defc_b", vec![d_c, d_me], "m^-1", "Deformation weights, antisymmetric part"),
        int("lu_index", vec![d_c], "Dominant land use category, WRF spelling of ivgtyp"),
        int("soilcat_top", vec![d_c], "Dominant top-layer soil category, WRF spelling of isltyp"),
        flt("soilcomp", vec![d_c, d_soil], "-", "Soil composition by depth"),
        flt("soilcl1", vec![d_c], "-", "Soil texture class, layer 1"),
        flt("soilcl2", vec![d_c], "-", "Soil texture class, layer 2"),
        flt("soilcl3", vec![d_c], "-", "Soil texture class, layer 3"),
        flt("soilcl4", vec![d_c], "-", "Soil texture class, layer 4"),
        NcVarDef::new("isice_lu", NcType::Int, vec![])
            .with_attrs(vec![NcAttr::text("units", "-"), NcAttr::text("long_name", "Ice land-use category")]),
        NcVarDef::new("iswater_lu", NcType::Int, vec![])
            .with_attrs(vec![NcAttr::text("units", "-"), NcAttr::text("long_name", "Water land-use category")]),
    ]);

    // The schema gate reads `declared_variables()`, not this vector. If the
    // two ever part, the gate would be grading a manifest nothing writes --
    // green while the file changed underneath it. Catch that here.
    let names: Vec<String> = vars.iter().map(|v| v.name.clone()).collect();
    {
        let actual: Vec<&str> = vars.iter().map(|v| v.name.as_str()).collect();
        if actual != declared_variables() {
            return Err(MpasError::Refusal(format!(
                "this writer emits {} variables but declared_variables() lists {}; \
                 the schema gate reads the second, so it would be grading a stale \
                 manifest until the two agree",
                actual.len(),
                declared_variables().len()
            )));
        }
    }

    // The manifest IS the registered schema. The mesh registry pins grid and
    // static together by byte count and SHA-256, so a variable added or
    // dropped here makes every previously registered pair unmatchable -- and
    // silently, since the run still exits 0 and the file still opens. Refuse
    // before writing rather than after registering.
    if let Some(said) = crate::staticfile::schema::divergence_from_pin(&names) {
        return Err(MpasError::Refusal(format!(
            "the static about to be written does not match the registered \
             schema: {said}. A pair built this way is refused at registration \
             by byte count before anything reads it."
        )));
    }

    let mut gattrs = vec![
        NcAttr::text("mesh_spec", crate::mesh::emit::MESH_SPEC),
        NcAttr::text("on_a_sphere", "YES"),
        NcAttr::floats("sphere_radius", vec![mesh.sphere_radius as f32]),
        NcAttr::text("is_periodic", "NO"),
        NcAttr::floats("x_period", vec![0.0]),
        NcAttr::floats("y_period", vec![0.0]),
        NcAttr::text("Conventions", "MPAS"),
        NcAttr::text("source", "MPAS"),
        NcAttr::text(
            "rw_static_engine",
            concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)"),
        ),
        NcAttr::text("rw_mpas_static_schema", UNIFIED_SCHEMA_TAG),
        // The provenance carries no wall clock and no filesystem path: a
        // registry pins this file by byte count and sha256, and either one
        // makes two identical builds unregisterable under the same name.
        NcAttr::text("rw_static_provenance_json", provenance_json.to_string()),
    ];
    // The coordinate frame is DECLARED, never inferred.  See
    // `staticfile::coordframe` for what an inferred one breaks.
    gattrs.extend(cfg.coordinates.declaration_attributes(mesh.sphere_radius));

    let mut w = NcClassicWriter::create(
        &cfg.out_path,
        NcFormat::Offset64,
        dims,
        gattrs,
        vars,
        1,
    )?;

    let mut xtime = [b' '; STR_LEN];
    for (i, b) in cfg.valid_time.bytes().take(STR_LEN).enumerate() {
        xtime[i] = b;
    }
    w.put_record("xtime", 0, NcData::Chars(&xtime))?;

    macro_rules! put_f {
        ($name:literal, $v:expr) => {{
            let tmp = f32_from_f64($v);
            w.put($name, NcData::Floats(&tmp))?;
        }};
    }
    // A coordinate array goes out in the DECLARED representation, and nothing
    // else does.  At binary32 this is byte-for-byte the `put_f!` path (the
    // same `f32_from_f64` on the same values); at binary64 the f64 the mesh
    // was derived in reaches the file unrounded.
    macro_rules! put_coord {
        ($name:literal, $v:expr) => {{
            if cfg.coordinates.is_binary64() {
                w.put($name, NcData::Doubles($v))?;
            } else {
                let tmp = f32_from_f64($v);
                w.put($name, NcData::Floats(&tmp))?;
            }
        }};
    }
    put_coord!("latCell", &mesh.lat_cell);
    put_coord!("lonCell", &mesh.lon_cell);
    put_coord!("xCell", &mesh.x_cell);
    put_coord!("yCell", &mesh.y_cell);
    put_coord!("zCell", &mesh.z_cell);
    w.put("indexToCellID", NcData::Ints(&mesh.index_to_cell_id))?;
    put_coord!("latEdge", &mesh.lat_edge);
    put_coord!("lonEdge", &mesh.lon_edge);
    put_coord!("xEdge", &mesh.x_edge);
    put_coord!("yEdge", &mesh.y_edge);
    put_coord!("zEdge", &mesh.z_edge);
    w.put("indexToEdgeID", NcData::Ints(&mesh.index_to_edge_id))?;
    put_coord!("latVertex", &mesh.lat_vertex);
    put_coord!("lonVertex", &mesh.lon_vertex);
    put_coord!("xVertex", &mesh.x_vertex);
    put_coord!("yVertex", &mesh.y_vertex);
    put_coord!("zVertex", &mesh.z_vertex);
    w.put("indexToVertexID", NcData::Ints(&mesh.index_to_vertex_id))?;

    // Topology goes out exactly as the grid stored it; see the `raw_*` note
    // on [`Mesh`].
    w.put("cellsOnEdge", NcData::Ints(&mesh.raw_cells_on_edge))?;
    let neoc: Vec<i32> = mesh.n_edges_on_cell.iter().map(|&v| v as i32).collect();
    w.put("nEdgesOnCell", NcData::Ints(&neoc))?;
    w.put("nEdgesOnEdge", NcData::Ints(&mesh.n_edges_on_edge))?;
    w.put("edgesOnCell", NcData::Ints(&mesh.raw_edges_on_cell))?;
    w.put("edgesOnEdge", NcData::Ints(&mesh.raw_edges_on_edge))?;
    put_f!("weightsOnEdge", &mesh.weights_on_edge);
    put_f!("dvEdge", &mesh.dv_edge);
    put_f!("dcEdge", &mesh.dc_edge);
    put_f!("angleEdge", &mesh.angle_edge);
    put_f!("areaCell", &mesh.area_cell);
    put_f!("areaTriangle", &mesh.area_triangle);
    w.put("cellsOnCell", NcData::Ints(&mesh.raw_cells_on_cell))?;
    w.put("verticesOnCell", NcData::Ints(&mesh.raw_vertices_on_cell))?;
    w.put("verticesOnEdge", NcData::Ints(&mesh.raw_vertices_on_edge))?;
    w.put("edgesOnVertex", NcData::Ints(&mesh.raw_edges_on_vertex))?;
    w.put("cellsOnVertex", NcData::Ints(&mesh.raw_cells_on_vertex))?;
    put_f!("kiteAreasOnVertex", &mesh.kite_areas_on_vertex);
    put_f!("meshDensity", &mesh.mesh_density);
    w.put("nominalMinDc", NcData::Floats(&[nominal_dx_m as f32]))?;

    w.put("bdyMaskCell", NcData::Ints(&mesh.bdy_mask_cell))?;
    w.put("bdyMaskEdge", NcData::Ints(&mesh.bdy_mask_edge))?;
    w.put("bdyMaskVertex", NcData::Ints(&mesh.bdy_mask_vertex))?;

    // The four local-frame arrays are PLACEHOLDERS in a static, not values.
    //
    // Measured on the published `x4.163842.static.nc`: `edgeNormalVectors`,
    // `localVerticalUnitVectors`, `cellTangentPlane` and `coeffs_reconstruct`
    // are ALL exactly zero, every element. The real frames belong to the
    // vertical grid the init stage builds, and a consumer overlays them from
    // the init file. Its pin is the digest of an exact +0 array at this
    // shape, so a derived frame here -- right as geometry -- is refused with
    // `static edgeNormalVectors placeholder bytes changed` before any device
    // memory is allocated.
    //
    // The SLOT is not optional either. Leaving `coeffs_reconstruct` out
    // stops the port at `AttributeError: mesh has no 'coeffs_reconstruct'`.
    w.put("edgeNormalVectors", NcData::Floats(&vec![0.0f32; ne * 3]))?;
    w.put("localVerticalUnitVectors", NcData::Floats(&vec![0.0f32; nc * 3]))?;
    w.put("cellTangentPlane", NcData::Floats(&vec![0.0f32; nc * 2 * 3]))?;
    w.put("coeffs_reconstruct", NcData::Floats(&vec![0.0f32; nc * me * 3]))?;

    w.put("deriv_two", NcData::Floats(&operators.deriv_two))?;

    let fedge: Vec<f32> = mesh.lat_edge.iter().map(|&lat| (2.0 * MPAS_OMEGA * lat.sin()) as f32).collect();
    let fvertex: Vec<f32> = mesh.lat_vertex.iter().map(|&lat| (2.0 * MPAS_OMEGA * lat.sin()) as f32).collect();
    w.put("fEdge", NcData::Floats(&fedge))?;
    w.put("fVertex", NcData::Floats(&fvertex))?;

    w.put("ter", NcData::Floats(&geo.ter))?;
    w.put("landmask", NcData::Ints(&geo.landmask))?;
    let mut mminlu = [b' '; STR_LEN];
    let label = geo.mminlu.as_bytes();
    if label.len() > STR_LEN {
        return Err(MpasError::Refusal(format!(
            "mminlu label is {} bytes, exceeds StrLen={STR_LEN}",
            label.len()
        )));
    }
    mminlu[..label.len()].copy_from_slice(label);
    w.put("mminlu", NcData::Chars(&mminlu))?;
    w.put("ivgtyp", NcData::Ints(&geo.ivgtyp))?;
    w.put("isltyp", NcData::Ints(&geo.isltyp))?;
    w.put("snoalb", NcData::Floats(&geo.snoalb))?;
    w.put("soiltemp", NcData::Floats(&geo.soiltemp))?;
    w.put("greenfrac", NcData::Floats(&geo.greenfrac))?;

    let mut shdmin = vec![0.0f32; nc];
    let mut shdmax = vec![0.0f32; nc];
    for c in 0..nc {
        let row = &geo.greenfrac[c * N_MONTHS..(c + 1) * N_MONTHS];
        shdmin[c] = row.iter().copied().fold(f32::INFINITY, f32::min);
        shdmax[c] = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }
    w.put("shdmin", NcData::Floats(&shdmin))?;
    w.put("shdmax", NcData::Floats(&shdmax))?;
    w.put("albedo12m", NcData::Floats(&geo.albedo12m))?;

    w.put("var2d", NcData::Floats(&f32_from_f64(&gwd.var2d)))?;
    w.put("con", NcData::Floats(&f32_from_f64(&gwd.con)))?;
    for k in 0..4 {
        w.put(&format!("oa{}", k + 1), NcData::Floats(&f32_from_f64(&gwd.oa[k])))?;
    }
    for k in 0..4 {
        w.put(&format!("ol{}", k + 1), NcData::Floats(&f32_from_f64(&gwd.ol[k])))?;
    }

    w.put("cell_gradient_coef_x", NcData::Floats(&operators.cell_gradient_coef_x))?;
    w.put("cell_gradient_coef_y", NcData::Floats(&operators.cell_gradient_coef_y))?;
    w.put("defc_a", NcData::Floats(&operators.defc_a))?;
    w.put("defc_b", NcData::Floats(&operators.defc_b))?;
    w.put("lu_index", NcData::Ints(&geo.ivgtyp))?;
    w.put("soilcat_top", NcData::Ints(&geo.isltyp))?;
    w.put("soilcomp", NcData::Floats(&geo.soilcomp))?;
    for (name, src) in [
        ("soilcl1", &geo.soilcl[0]),
        ("soilcl2", &geo.soilcl[1]),
        ("soilcl3", &geo.soilcl[2]),
        ("soilcl4", &geo.soilcl[3]),
    ] {
        let tmp: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        w.put(name, NcData::Floats(&tmp))?;
    }
    w.put("isice_lu", NcData::Ints(&[geo.isice_lu]))?;
    w.put("iswater_lu", NcData::Ints(&[geo.iswater_lu]))?;
    w.finish()?;
    Ok(())
}

/// How many source rows the sub-grid orography pass may hold at once when the
/// budget allows it. Below this the pass still works; above it the extra rows
/// buy nothing, because a band is only useful up to the tallest box in it.
const GWD_PREFERRED_BAND_ROWS: i64 = 768;

/// Each cell's mean `dcEdge`, in metres. The box the drag statistics are cut
/// from is sized from this and nothing else.
fn mean_edge_length_m(mesh: &Mesh) -> Vec<f64> {
    let mut out = vec![0.0f64; mesh.n_cells];
    for c in 0..mesh.n_cells {
        let k = mesh.n_edges_on_cell[c];
        if k == 0 {
            continue;
        }
        let mut sum = 0.0;
        for s in 0..k {
            let e = mesh.edges_on_cell[c * mesh.max_edges + s];
            if e < mesh.n_edges {
                sum += mesh.dc_edge[e];
            }
        }
        out[c] = sum / k as f64;
    }
    out
}

/// Build a static, silently. See [`build_static_reporting`].
pub fn build_static(cfg: &StaticBuildConfig) -> MpasResult<StaticBuildReceipt> {
    build_static_reporting(cfg, &|_| {})
}

/// Build a static, streaming one tab-separated progress line per stage.
///
/// The grammar is the door's: `GRID`, `NOMINALDX`, `VECTORS`, `OPERATORS`,
/// `GEOGGRID`, `GWDBAND`. A wrapper parses these, so a token that stops being
/// printed is a wrapper that stops reporting rather than a wrapper that fails.
pub fn build_static_reporting(
    cfg: &StaticBuildConfig,
    progress: &dyn Fn(&str),
) -> MpasResult<StaticBuildReceipt> {
    if cfg.out_path.exists() && !cfg.clobber {
        return Err(MpasError::Refusal(format!(
            "{} exists; pass --clobber to replace it. Overwriting a static \
             silently would leave every init file and run receipt that pinned \
             its sha256 pointing at different bytes under the same name",
            cfg.out_path.display()
        )));
    }
    let missing = cfg.geog.missing();
    if !missing.is_empty() {
        return Err(MpasError::Refusal(format!(
            "the geography archive is missing {} dataset(s) a static needs: {}. \
             Terrain, land use, soil class and composition, deep-soil \
             temperature, green-ness and albedo all come from these \
             directories, and a static written without one would hand the model \
             a field of zeros that the run would carry as real geography.\n\
             what does work:\n  \
             gpuwm fetch-geog --datasets all, which stages every one of them\n  \
             --geog DIR pointing at a complete archive",
            missing.len(),
            missing.join(", ")
        )));
    }
    let mesh = Mesh::read(&cfg.grid_path)?;
    progress(&format!(
        "GRID\t{}\t{}\t{}\t{}",
        mesh.n_cells, mesh.n_edges, mesh.n_vertices, mesh.sphere_radius
    ));

    // The one value that is DECLARED rather than measured. A consumer compares
    // its bit pattern against the mesh binding's nominal dx and refuses on a
    // near-miss, because that scalar is what the drag length scale is rebound
    // to.
    let implied = mesh.nominal_min_dc;
    let nominal_dx_m = match cfg.nominal_dx_m {
        None => implied,
        Some(metres) => {
            if !(metres.is_finite() && metres > 0.0) {
                return Err(MpasError::Refusal(format!(
                    "nominal dx {metres} is not a positive length; the drag \
                     length scale rebound to it would be meaningless"
                )));
            }
            if (implied - metres).abs() > NOMINAL_DX_CROSS_CHECK_RELATIVE * metres {
                return Err(MpasError::Refusal(format!(
                    "the grid's nominalMinDc implies {implied:.3} m, but the \
                     static is being told to declare {metres:.3} m. They are \
                     more than {:.3} % apart, so this grid and this static are \
                     not the same mesh and pairing them would give every cell a \
                     horizontal mixing length taken from a different resolution",
                    NOMINAL_DX_CROSS_CHECK_RELATIVE * 100.0
                )));
            }
            metres
        }
    };
    progress(&format!(
        "NOMINALDX\t{:.6}\t{:.9}\t{:#010x}",
        nominal_dx_m,
        nominal_dx_m as f32,
        (nominal_dx_m as f32).to_bits()
    ));
    progress(
        "VECTORS\tfEdge fVertex edgeNormalVectors localVerticalUnitVectors cellTangentPlane",
    );

    let (resident, per_worker, writer_slab, cache_request) = build_footprint(&mesh, cfg)?;

    // The drag band's own resident cost, sized from the geography it will
    // actually read. It does not overlap the tile workers -- the pool is gone
    // by then -- so it competes with the writer slab for the same one-shot
    // slot rather than adding to it.
    let dc_m = mean_edge_length_m(&mesh);
    let gwd_topo = GeogDataset::open(&cfg.geog.terrain)?;
    let gwd_min_rows = crate::static_gwd::minimum_band_rows(&gwd_topo, &mesh.lat_cell, &dc_m)?;
    let gwd_rows = gwd_min_rows.max(GWD_PREFERRED_BAND_ROWS);
    let one_shot = |rows: i64| writer_slab.max(crate::static_gwd::band_bytes(&gwd_topo, rows));

    let mut gwd_band_rows = gwd_rows;
    let plan = match plan_tile_parallelism(
        "preflight",
        resident,
        per_worker,
        one_shot(gwd_rows),
        1024 * 1024 * 1024,
        resolve_tile_workers(cfg.tile_workers),
        cache_request,
        cfg.host_memory_limit_bytes,
    ) {
        Ok(plan) => plan,
        Err(wide) if gwd_min_rows < gwd_rows => {
            // Fall back to the narrowest band that still serves every cell.
            // The ANSWER is identical either way -- a band always covers the
            // whole of every box it serves -- so this trades throughput for
            // headroom and never a number in the file.
            gwd_band_rows = gwd_min_rows;
            plan_tile_parallelism(
                "preflight",
                resident,
                per_worker,
                one_shot(gwd_min_rows),
                1024 * 1024 * 1024,
                resolve_tile_workers(cfg.tile_workers),
                cache_request,
                cfg.host_memory_limit_bytes,
            )
            .map_err(|narrow| {
                MpasError::Refusal(format!(
                    "{narrow}\n\nA narrower sub-grid orography band was tried \
                     first and also refused; the wide-band refusal was: {wide}"
                ))
            })?
        }
        Err(e) => return Err(e),
    };
    drop(gwd_topo);
    let predicted = plan.predicted_peak_bytes;
    let limit = plan.limit_bytes;
    let mut memory = MemoryReceipt::new(limit, Some(predicted));
    memory.event(
        "preflight",
        None,
        "mesh loaded and host-memory admission passed",
        None,
        Some(predicted),
        None,
        vec![
            BufferLedger {
                name: "mesh geometry/topology".to_string(),
                bytes: resident,
                lifetime: "whole build".to_string(),
            },
            BufferLedger {
                name: format!("tile scratch x {} concurrent workers", plan.workers),
                bytes: (plan.workers as u64).saturating_mul(per_worker),
                lifetime: "one geography pass".to_string(),
            },
            BufferLedger {
                name: "reusable tile-map cache".to_string(),
                bytes: plan.tile_map_cache_bytes,
                lifetime: "whole build".to_string(),
            },
        ],
    );

    let tree = KdTree::new(&mesh);
    let max_d2 = mesh.max_search_distance2();
    let cache = TileMapCache::new(plan.tile_map_cache_bytes);
    let pool = if plan.workers > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(plan.workers)
                .thread_name(|i| format!("rw-mpas-tile-{i}"))
                .build()
                .map_err(|e| {
                    MpasError::Refusal(format!(
                        "cannot start {} admitted tile workers: {e}. The build would \
                         otherwise run on whatever pool happened to exist, which is not \
                         the worker count host-memory admission sized. Remedy: set \
                         --tile-workers 1 to build serially.",
                        plan.workers
                    ))
                })?,
        )
    } else {
        None
    };
    let ctx = GeogContext {
        mesh: &mesh,
        tree: &tree,
        max_d2,
        workers: plan.workers,
        pool: pool.as_ref(),
        cache: &cache,
    };

    memory.event("operators", None, "before operator construction", None, None, None, vec![]);
    let operators = build_operator_fields(&mesh.operator_view())?;
    progress(&format!(
        "OPERATORS\tderiv_two\t{}\t{}",
        operators.deriv_two.len(),
        FIFTEEN
    ));
    memory.event(
        "operators",
        None,
        "after operator construction",
        None,
        Some(
            (operators.cell_gradient_coef_x.len()
                + operators.cell_gradient_coef_y.len()
                + operators.defc_a.len()
                + operators.defc_b.len()
                + operators.deriv_two.len()) as u64
                * 4,
        ),
        None,
        vec![],
    );

    let mut datasets = Vec::new();
    let mut time_field = |field: &str, path: &Path, factor: usize, f: &mut dyn FnMut() -> MpasResult<(u64, u64)>| -> MpasResult<()> {
        let started = Instant::now();
        memory.event(field, Some(path), "before dataset", None, None, None, vec![]);
        let (plane_bytes, dest_bytes) = f()?;
        memory.event(
            field,
            Some(path),
            "after dataset; source tile/plane workspace dropped",
            Some(plane_bytes),
            Some(dest_bytes),
            None,
            vec![],
        );
        let ds = GeogDataset::open(path)?;
        let elapsed = started.elapsed().as_secs_f64();
        progress(&format!(
            "GEOGGRID\t{field}\t{}\t{}\t{}\t{elapsed:.2}",
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            ds.tiles.len(),
            factor,
        ));
        datasets.push(DatasetReceipt {
            field: field.to_string(),
            path: path.display().to_string(),
            tiles: ds.tiles.len(),
            source_plane_bytes: ds.plane_bytes(),
            supersample: factor,
            destination_bytes: dest_bytes,
            elapsed_seconds: elapsed,
        });
        Ok(())
    };

    let mut ter: Vec<f32> = Vec::new();
    time_field("ter", &cfg.geog.terrain, cfg.supersample, &mut || {
        let (v, ds) = continuous_mean(&ctx, &cfg.geog.terrain, cfg.supersample, None, false)?;
        if ds.index.nz() != 1 {
            return Err(MpasError::Refusal("terrain source must have one plane".to_string()));
        }
        ter = v;
        Ok((ds.plane_bytes(), (ter.len() * 4) as u64))
    })?;

    let mut ivgtyp = Vec::new();
    let mut landuse_ds_opt = None;
    time_field("landuse", &cfg.geog.landuse, cfg.supersample_landuse, &mut || {
        let (v, _, _, ds) = categorical(&ctx, &cfg.geog.landuse, cfg.supersample_landuse)?;
        ivgtyp = v;
        let pb = ds.plane_bytes();
        landuse_ds_opt = Some(ds);
        Ok((pb, (ivgtyp.len() * 4) as u64))
    })?;
    let landuse_ds = landuse_ds_opt.take().expect("landuse populated");
    let iswater_lu = landuse_ds.index.iswater.unwrap_or(17) as i32;
    let isice_lu = landuse_ds.index.isice.unwrap_or(15) as i32;
    let landmask: Vec<i32> = ivgtyp.iter().map(|&v| if v == iswater_lu { 0 } else { 1 }).collect();

    let mut isltyp = Vec::new();
    time_field("soilcat_top", &cfg.geog.soilcat, cfg.supersample, &mut || {
        let (v, _, _, ds) = categorical(&ctx, &cfg.geog.soilcat, cfg.supersample)?;
        isltyp = v;
        Ok((ds.plane_bytes(), (isltyp.len() * 4) as u64))
    })?;

    let mut greenfrac = Vec::new();
    time_field("greenfrac", &cfg.geog.greenfrac, cfg.supersample_30s, &mut || {
        let (v, ds) = continuous_mean(&ctx, &cfg.geog.greenfrac, cfg.supersample_30s, Some(&landmask), true)?;
        if ds.index.nz() != 12 {
            return Err(MpasError::Refusal(format!(
                "greenfrac must carry 12 months, got {}",
                ds.index.nz()
            )));
        }
        // Archive fraction -> model percent. See [`GREENFRAC_MODEL_SCALE`].
        greenfrac = v;
        for x in &mut greenfrac {
            *x *= GREENFRAC_MODEL_SCALE;
        }
        Ok((ds.plane_bytes(), (greenfrac.len() * 4) as u64))
    })?;

    let mut albedo12m = Vec::new();
    time_field("albedo12m", &cfg.geog.albedo, cfg.supersample_30s, &mut || {
        let (v, ds) = continuous_mean(&ctx, &cfg.geog.albedo, cfg.supersample_30s, Some(&landmask), true)?;
        if ds.index.nz() != 12 {
            return Err(MpasError::Refusal(format!(
                "albedo must carry 12 months, got {}",
                ds.index.nz()
            )));
        }
        albedo12m = v;
        Ok((ds.plane_bytes(), (albedo12m.len() * 4) as u64))
    })?;

    let mut snoalb = Vec::new();
    time_field("snoalb", &cfg.geog.snow_albedo, cfg.supersample_30s, &mut || {
        let (v, ds) = snow_albedo(&ctx, &cfg.geog.snow_albedo, cfg.supersample_30s, &landmask)?;
        snoalb = v;
        Ok((ds.plane_bytes(), (snoalb.len() * 4) as u64))
    })?;

    let mut soiltemp = Vec::new();
    time_field("soiltemp", &cfg.geog.soil_temperature, 1, &mut || {
        let (v, ds) = soil_temperature(&mesh, &cfg.geog.soil_temperature, &landmask)?;
        soiltemp = v;
        Ok((ds.plane_bytes(), (soiltemp.len() * 4) as u64))
    })?;

    let mut soilcomp_v = Vec::new();
    time_field("soilcomp", &cfg.geog.soilcomp, cfg.supersample_30s, &mut || {
        let (v, ds) = soilcomp(&ctx, &cfg.geog.soilcomp, cfg.supersample_30s, &landmask)?;
        soilcomp_v = v;
        Ok((ds.plane_bytes(), (soilcomp_v.len() * 4) as u64))
    })?;

    let mut soilcls: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::new());
    let paths = [&cfg.geog.soilcl1, &cfg.geog.soilcl2, &cfg.geog.soilcl3, &cfg.geog.soilcl4];
    for idx in 0..4 {
        let field = format!("soilcl{}", idx + 1);
        time_field(&field, paths[idx], cfg.supersample_30s, &mut || {
            let (v, _, _, ds) = categorical(&ctx, paths[idx], cfg.supersample_30s)?;
            soilcls[idx] = v;
            Ok((ds.plane_bytes(), (soilcls[idx].len() * 4) as u64))
        })?;
    }

    let tile_map_cache = cache.receipt()?;
    // Release the workers before the drag band and the writer allocate: the
    // peak the admission gate was shown does not include them at once.
    drop(ctx);
    drop(pool);

    // The sub-grid orography band. It runs after the tile pool is gone and
    // holds two full-width source bands; that is the term the admission gate
    // was shown as this phase's one-shot.
    let gwd_bytes = crate::static_gwd::band_bytes(
        &GeogDataset::open(&cfg.geog.terrain)?,
        gwd_band_rows,
    );
    memory.event(
        "gwd",
        Some(&cfg.geog.terrain),
        "before the sub-grid orography band",
        None,
        None,
        Some(gwd_bytes),
        vec![BufferLedger {
            name: format!("terrain + land-use bands, {gwd_band_rows} rows"),
            bytes: gwd_bytes,
            lifetime: "one orography band".to_string(),
        }],
    );
    let t_gwd = Instant::now();
    let gwd = {
        let topo = GeogDataset::open(&cfg.geog.terrain)?;
        let lu = GeogDataset::open(&cfg.geog.landuse)?;
        crate::static_gwd::compute(
            &topo,
            &lu,
            &mesh.lat_cell,
            &mesh.lon_cell,
            &dc_m,
            &mesh.n_edges_on_cell,
            &mesh.cells_on_cell,
            mesh.max_edges,
            gwd_band_rows,
            progress,
        )?
    };
    let gwd_seconds = t_gwd.elapsed().as_secs_f64();
    memory.event(
        "gwd",
        Some(&cfg.geog.terrain),
        "after the sub-grid orography band; source bands dropped",
        Some(gwd_bytes),
        Some((mesh.n_cells * 10 * 8) as u64),
        None,
        vec![],
    );

    // What FP32 storage does to this file's own edge lengths, measured on the
    // values about to be written rather than predicted from the resolution,
    // and judged against the port's LIVE load contract (the 1.73 m storage
    // atol and the 0.02 dvEdge/dcEdge admission floor -- the retired
    // rtol 2e-5 / atol 0.0 comparison died with the port's 2026-08-23
    // contract change; stale-guard audit 2026-08-25, finding 3).  This
    // reading decides whether the mesh being built can reach a forecast.
    let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    // Coordinates go through the DECLARED representation, metrics always
    // through f32: that is exactly what the writer below does, so this
    // reading is taken on the bytes the file will carry rather than on a
    // model of them.
    let coordv = |v: &[f64]| -> Vec<f64> {
        if cfg.coordinates.is_binary64() {
            v.to_vec()
        } else {
            v.iter().map(|&x| (x as f32) as f64).collect()
        }
    };
    let fp32_metric_agreement = crate::staticfile::fp32metrics::measure(
        &f32v(&mesh.dv_edge),
        &coordv(&mesh.x_vertex),
        &coordv(&mesh.y_vertex),
        &coordv(&mesh.z_vertex),
        &mesh
            .raw_vertices_on_edge
            .iter()
            .map(|&v| v as i64)
            .collect::<Vec<i64>>(),
        &f32v(&mesh.dc_edge),
        mesh.sphere_radius,
        cfg.coordinates,
    );
    progress(&format!(
        "FP32METRICS\t{:.3e}\t{:.3}\t{}\t{:.3e}\t{}\t{}",
        fp32_metric_agreement.max_dv_edge_absolute_m,
        fp32_metric_agreement.min_dv_edge_m,
        fp32_metric_agreement.edges_past_port_storage_tolerance,
        fp32_metric_agreement.min_dv_over_dc,
        fp32_metric_agreement.edges_below_admission_floor,
        fp32_metric_agreement.port_accepts()
    ));

    let provenance = StaticProvenance {
        engine: concat!("rw-mpas ", env!("CARGO_PKG_VERSION"), " (rust)").to_string(),
        schema: UNIFIED_SCHEMA_TAG.to_string(),
        grid_sha256: crate::sha256_file(&cfg.grid_path)?,
        n_cells: mesh.n_cells,
        n_edges: mesh.n_edges,
        n_vertices: mesh.n_vertices,
        sphere_radius: mesh.sphere_radius,
        nominal_dx_m,
        nominal_dx_f32_bits: (nominal_dx_m as f32).to_bits(),
        gwd_water_category: gwd.water_category,
        gwd_water_category_source: gwd.water_category_source.clone(),
        gwd_variance_smoothed_cells: gwd.smoothed_cells,
        mminlu: landuse_ds.index.mminlu.clone(),
    };
    let provenance_json = serde_json::to_string(&provenance)
        .map_err(|e| MpasError::Refusal(format!("cannot serialise the provenance: {e}")))?;

    let geo = GeographyFields {
        ter,
        landmask,
        ivgtyp,
        isltyp,
        greenfrac,
        albedo12m,
        snoalb,
        soiltemp,
        soilcomp: soilcomp_v,
        soilcl: soilcls,
        isice_lu,
        iswater_lu,
        mminlu: landuse_ds.index.mminlu.clone(),
    };

    memory.event("write", None, "before static write", None, None, None, vec![]);
    write_static(cfg, &mesh, &operators, &geo, &gwd, nominal_dx_m, &provenance_json)?;
    memory.event("write", None, "static fsync complete", None, None, None, vec![]);

    let output_bytes = std::fs::metadata(&cfg.out_path)?.len();
    let generated_fields = vec![
        "cell_gradient_coef_x", "cell_gradient_coef_y", "defc_a", "defc_b",
        "deriv_two", "fEdge", "fVertex", "ter", "landmask", "lu_index", "ivgtyp",
        "soilcat_top", "isltyp", "greenfrac", "albedo12m", "shdmin", "shdmax",
        "snoalb", "soiltemp", "soilcomp", "soilcl1", "soilcl2", "soilcl3", "soilcl4",
        "var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    let operator_provenance = BTreeMap::from([
        (
            "cell_gradient_coef_x/y".to_string(),
            "MPAS-A v8.4.1 core_init_atmosphere/mpas_atm_advection.F::atm_initialize_deformation_weights".to_string(),
        ),
        (
            "deriv_two".to_string(),
            "MPAS-A v8.4.1 core_init_atmosphere/mpas_atm_advection.F::atm_initialize_advection_rk".to_string(),
        ),
        (
            "defc_a/b".to_string(),
            "MPAS v8.4.1 core_sw/mpas_sw_advection.F::sw_initialize_deformation_weights; consumed by gpuwm-hex Smagorinsky strain operator".to_string(),
        ),
    ]);

    let receipt = StaticBuildReceipt {
        schema: "rw-mpas-static.build/v3-unified".to_string(),
        grid_path: cfg.grid_path.display().to_string(),
        output_path: cfg.out_path.display().to_string(),
        n_cells: mesh.n_cells,
        n_edges: mesh.n_edges,
        n_vertices: mesh.n_vertices,
        max_edges: mesh.max_edges,
        grid_scaled_from_unit_sphere: mesh.scaled_from_unit_sphere,
        output_bytes,
        datasets,
        parallel: plan,
        tile_map_cache,
        memory,
        generated_fields,
        operator_provenance,
        provenance: cfg.provenance.clone(),
        stamped_provenance: provenance,
        gwd_seconds,
        gwd_band_rows,
        gwd_bands: gwd.bands,
        gwd_band_bytes: gwd.band_bytes,
        gwd_variance_smoothed_cells: gwd.smoothed_cells,
        regional: if mesh.is_regional() {
            let mut ring_cell_counts = [0usize; 8];
            for &m in &mesh.bdy_mask_cell {
                ring_cell_counts[m as usize] += 1;
            }
            Some(RegionalStaticNotes {
                ring_cell_counts,
                deriv_two_zero_stencil_cells: operators.deriv_two_zero_stencil_cells,
                gwd_smooth_skipped_boundary_cells: gwd.smooth_skipped_boundary_cells,
            })
        } else {
            None
        },
        fp32_metric_agreement,
        sha256: crate::sha256_file(&cfg.out_path)?,
        variables: declared_variables().iter().map(|s| s.to_string()).collect(),
    };

    if let Some(path) = &cfg.receipt_path {
        let json = serde_json::to_vec_pretty(&receipt).map_err(|e| {
            MpasError::Refusal(format!("cannot serialize static receipt: {e}"))
        })?;
        std::fs::write(path, json)?;
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_geog::fixture::{dataset, plane_1234_be, MINIMAL_INDEX, TILE_NAME};
    use crate::sha256_file;

    // -- carrying a value to cells the archive has no pixel for -------------

    /// A ring of `n` cells, each neighbouring the two beside it. Enough
    /// topology to exercise a breadth-first spread and nothing more.
    fn ring(n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut n_edges = vec![2usize; n];
        let mut cells = vec![ABSENT_NEIGHBOR; n * 2];
        for c in 0..n {
            cells[c * 2] = (c + n - 1) % n;
            cells[c * 2 + 1] = (c + 1) % n;
        }
        n_edges[0] = 2;
        (n_edges, cells)
    }

    #[test]
    fn a_cell_with_no_pixels_takes_the_mean_of_the_neighbours_that_have_some() {
        let n = 4;
        let (n_edges, cells) = ring(n);
        // Cell 1 has nothing; its neighbours 0 and 2 hold 10 and 20.
        let mut out = vec![10.0f32, 0.0, 20.0, 30.0];
        let mut filled = vec![true, false, true, true];
        let used = spread_to_unsampled(
            &mut out, &mut filled, 1, &[1], Vec::new(), n, 2, &n_edges, &cells,
        )
        .expect("one unsampled cell between two sampled ones is reachable");
        assert!(used.is_empty());
        assert_eq!(out[1], 15.0);
        assert!(filled[1]);
        // Nothing else moved.
        assert_eq!((out[0], out[2], out[3]), (10.0, 20.0, 30.0));
    }

    #[test]
    fn every_level_of_a_column_is_carried() {
        let n = 3;
        let (n_edges, cells) = ring(n);
        let nz = 2;
        let mut out = vec![1.0f32, 3.0, 0.0, 0.0, 5.0, 7.0];
        let mut filled = vec![true, false, true];
        spread_to_unsampled(
            &mut out, &mut filled, nz, &[1], Vec::new(), n, 2, &n_edges, &cells,
        )
        .expect("reachable");
        assert_eq!((out[2], out[3]), (3.0, 5.0));
    }

    #[test]
    fn a_conduit_carries_a_value_without_keeping_it() {
        // 0 has data; 1 is mask-excluded (a water cell); 2 needs a value and
        // touches only 1. Without conduction 2 is unreachable.
        let n = 4;
        let (n_edges, cells) = ring(n);
        let mut out = vec![8.0f32, 0.0, 0.0, 0.0];
        let mut filled = vec![true, false, false, false];
        let used = spread_to_unsampled(
            &mut out, &mut filled, 1, &[2], vec![1, 3], n, 2, &n_edges, &cells,
        )
        .expect("the conduit makes cell 2 reachable");
        assert_eq!(out[2], 8.0, "carried across the excluded cell unchanged");
        assert!(used.contains(&1) || used.contains(&3),
                "the conduit that carried it is reported so the caller can reset it");
    }

    #[test]
    fn a_region_with_no_value_anywhere_is_refused_with_the_cells_named() {
        // Two disjoint pairs: 0-1 hold data, 2-3 are isolated and need one.
        let n_cells = 4;
        let n_edges = vec![1usize, 1, 1, 1];
        let cells = vec![1usize, 0, 3, 2];
        let mut out = vec![4.0f32, 6.0, 0.0, 0.0];
        let mut filled = vec![true, true, false, false];
        let stranded = spread_to_unsampled(
            &mut out, &mut filled, 1, &[2, 3], Vec::new(), n_cells, 1, &n_edges, &cells,
        )
        .expect_err("an unreachable region must refuse, not invent a value");
        assert_eq!(stranded.len(), 2);
        assert!(stranded.contains(&2) && stranded.contains(&3));
    }

    #[test]
    fn the_spread_is_a_ring_so_the_visit_order_does_not_change_the_answer() {
        // 0 holds 100; 1 and 2 need values and neighbour each other as well
        // as 0. If 1 were settled and then USED as a source inside the same
        // round, 2 would get (100+100)/2 through one path and 100 through the
        // other depending on order. It must be 100 either way.
        let n_cells = 4;
        let (n_edges, cells) = ring(n_cells);
        let mut out = vec![100.0f32, 0.0, 0.0, 0.0];
        let mut filled = vec![true, false, false, false];
        spread_to_unsampled(
            &mut out, &mut filled, 1, &[1, 2, 3], Vec::new(), n_cells, 2, &n_edges, &cells,
        )
        .expect("reachable");
        assert_eq!(out[1], 100.0);
        assert_eq!(out[3], 100.0);
        assert_eq!(out[2], 100.0, "a uniform source stays uniform");
    }

    #[test]
    fn nothing_needed_means_nothing_is_touched() {
        let n_cells = 3;
        let (n_edges, cells) = ring(n_cells);
        let mut out = vec![1.0f32, 2.0, 3.0];
        let mut filled = vec![true, true, true];
        let used = spread_to_unsampled(
            &mut out, &mut filled, 1, &[], vec![0, 1, 2], n_cells, 2, &n_edges, &cells,
        )
        .expect("no work");
        assert!(used.is_empty());
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    // -- the regional load path, graded on bytes this module writes ----------

    /// One tiny grid file with configurable boundary masks and sentinel
    /// placement, through the same classic writer the crate ships. Geometry
    /// is unit-sphere garbage on purpose: what is under test is the sentinel
    /// ADMISSION, which reads topology and masks only.
    fn tiny_grid_nc(
        label: &str,
        bdy_cell: Option<[i32; 4]>,
        bdy_edge: [i32; 4],
        bdy_vertex: [i32; 4],
        cells_on_cell: [[i32; 3]; 4],
        cells_on_edge: [[i32; 2]; 4],
        vertices_on_cell: [[i32; 3]; 4],
        edges_on_edge_rows: [[i32; 6]; 4],
    ) -> PathBuf {
        use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcVarDef, NcType};
        let path = std::env::temp_dir().join(format!(
            "rw-mpas-regional-admission-{}-{label}.nc",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let dims = vec![
            NcDim::fixed("nCells", 4),
            NcDim::fixed("nEdges", 4),
            NcDim::fixed("nVertices", 4),
            NcDim::fixed("maxEdges", 3),
            NcDim::fixed("maxEdges2", 6),
            NcDim::fixed("TWO", 2),
            NcDim::fixed("vertexDegree", 3),
        ];
        let (c, e, v, me, me2, two, vd) = (0usize, 1, 2, 3, 4, 5, 6);
        let mut vars = vec![
            NcVarDef::new("latCell", NcType::Double, vec![c]),
            NcVarDef::new("lonCell", NcType::Double, vec![c]),
            NcVarDef::new("latEdge", NcType::Double, vec![e]),
            NcVarDef::new("lonEdge", NcType::Double, vec![e]),
            NcVarDef::new("latVertex", NcType::Double, vec![v]),
            NcVarDef::new("lonVertex", NcType::Double, vec![v]),
            NcVarDef::new("xCell", NcType::Double, vec![c]),
            NcVarDef::new("yCell", NcType::Double, vec![c]),
            NcVarDef::new("zCell", NcType::Double, vec![c]),
            NcVarDef::new("xEdge", NcType::Double, vec![e]),
            NcVarDef::new("yEdge", NcType::Double, vec![e]),
            NcVarDef::new("zEdge", NcType::Double, vec![e]),
            NcVarDef::new("xVertex", NcType::Double, vec![v]),
            NcVarDef::new("yVertex", NcType::Double, vec![v]),
            NcVarDef::new("zVertex", NcType::Double, vec![v]),
            NcVarDef::new("dcEdge", NcType::Double, vec![e]),
            NcVarDef::new("dvEdge", NcType::Double, vec![e]),
            NcVarDef::new("areaCell", NcType::Double, vec![c]),
            NcVarDef::new("areaTriangle", NcType::Double, vec![v]),
            NcVarDef::new("kiteAreasOnVertex", NcType::Double, vec![v, vd]),
            NcVarDef::new("nominalMinDc", NcType::Double, vec![]),
            NcVarDef::new("angleEdge", NcType::Double, vec![e]),
            NcVarDef::new("meshDensity", NcType::Double, vec![c]),
            NcVarDef::new("weightsOnEdge", NcType::Double, vec![e, me2]),
            NcVarDef::new("nEdgesOnCell", NcType::Int, vec![c]),
            NcVarDef::new("nEdgesOnEdge", NcType::Int, vec![e]),
            NcVarDef::new("cellsOnCell", NcType::Int, vec![c, me]),
            NcVarDef::new("edgesOnCell", NcType::Int, vec![c, me]),
            NcVarDef::new("verticesOnCell", NcType::Int, vec![c, me]),
            NcVarDef::new("cellsOnEdge", NcType::Int, vec![e, two]),
            NcVarDef::new("verticesOnEdge", NcType::Int, vec![e, two]),
            NcVarDef::new("edgesOnEdge", NcType::Int, vec![e, me2]),
            NcVarDef::new("cellsOnVertex", NcType::Int, vec![v, vd]),
            NcVarDef::new("edgesOnVertex", NcType::Int, vec![v, vd]),
        ];
        if bdy_cell.is_some() {
            vars.push(NcVarDef::new("bdyMaskCell", NcType::Int, vec![c]));
            vars.push(NcVarDef::new("bdyMaskEdge", NcType::Int, vec![e]));
            vars.push(NcVarDef::new("bdyMaskVertex", NcType::Int, vec![v]));
        }
        let mut w =
            NcClassicWriter::create(&path, NcFormat::Offset64, dims, Vec::new(), vars, 0).unwrap();
        // Unit-sphere-ish geometry: four distinct points, nothing degenerate.
        let ang = [0.1f64, 0.7, 1.3, 1.9];
        let sc: Vec<f64> = ang.iter().map(|a| a.cos()).collect();
        let ss: Vec<f64> = ang.iter().map(|a| a.sin()).collect();
        let zz = vec![0.0f64; 4];
        for name in ["latCell", "latEdge", "latVertex"] {
            w.put(name, NcData::Doubles(&zz)).unwrap();
        }
        for name in ["lonCell", "lonEdge", "lonVertex"] {
            w.put(name, NcData::Doubles(&ang)).unwrap();
        }
        for name in ["xCell", "xEdge", "xVertex"] {
            w.put(name, NcData::Doubles(&sc)).unwrap();
        }
        for name in ["yCell", "yEdge", "yVertex"] {
            w.put(name, NcData::Doubles(&ss)).unwrap();
        }
        for name in ["zCell", "zEdge", "zVertex"] {
            w.put(name, NcData::Doubles(&zz)).unwrap();
        }
        for name in ["dcEdge", "dvEdge"] {
            w.put(name, NcData::Doubles(&[0.1; 4])).unwrap();
        }
        for name in ["areaCell", "areaTriangle", "meshDensity"] {
            w.put(name, NcData::Doubles(&[1.0; 4])).unwrap();
        }
        w.put("kiteAreasOnVertex", NcData::Doubles(&[1.0; 12])).unwrap();
        w.put("nominalMinDc", NcData::Doubles(&[0.1])).unwrap();
        w.put("angleEdge", NcData::Doubles(&[0.0; 4])).unwrap();
        w.put("weightsOnEdge", NcData::Doubles(&[0.0; 24])).unwrap();
        w.put("nEdgesOnCell", NcData::Ints(&[3; 4])).unwrap();
        w.put("nEdgesOnEdge", NcData::Ints(&[2; 4])).unwrap();
        let flat3 = |rows: [[i32; 3]; 4]| rows.concat();
        let flat2 = |rows: [[i32; 2]; 4]| rows.concat();
        w.put("cellsOnCell", NcData::Ints(&flat3(cells_on_cell))).unwrap();
        w.put("edgesOnCell", NcData::Ints(&flat3([[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]])))
            .unwrap();
        w.put("verticesOnCell", NcData::Ints(&flat3(vertices_on_cell))).unwrap();
        w.put("cellsOnEdge", NcData::Ints(&flat2(cells_on_edge))).unwrap();
        w.put("verticesOnEdge", NcData::Ints(&flat2([[1, 2], [2, 3], [3, 4], [4, 1]])))
            .unwrap();
        w.put("edgesOnEdge", NcData::Ints(&edges_on_edge_rows.concat())).unwrap();
        w.put("cellsOnVertex", NcData::Ints(&flat3([[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]])))
            .unwrap();
        w.put("edgesOnVertex", NcData::Ints(&flat3([[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]])))
            .unwrap();
        if let Some(bc) = bdy_cell {
            w.put("bdyMaskCell", NcData::Ints(&bc)).unwrap();
            w.put("bdyMaskEdge", NcData::Ints(&bdy_edge)).unwrap();
            w.put("bdyMaskVertex", NcData::Ints(&bdy_vertex)).unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// The clean regional layout every corruption below is one delta from:
    /// cells 2 and 3 are outermost (mask 7) and carry one absent neighbour
    /// each; edges 2 and 3 are one-sided; vertices 2 and 3 are one-sided.
    fn clean_regional() -> PathBuf {
        tiny_grid_nc(
            "clean",
            Some([5, 6, 7, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        )
    }

    /// A culler-shaped regional grid loads: sentinels at mask-7 only.
    #[test]
    fn a_regional_grid_with_mask7_sentinels_is_admitted() {
        let path = clean_regional();
        let probe = probe_grid(&path).expect("the clean regional layout admits");
        assert!(probe.regional);
        assert_eq!(probe.ring_cell_counts, [0, 0, 0, 0, 0, 1, 1, 2]);
        assert_eq!(probe.absent_neighbor_slots, 2);
        assert_eq!(probe.one_sided_edges, 2);
        let _ = std::fs::remove_file(&path);
    }

    /// An absent neighbour on a NON-outermost cell is a torn mesh, refused
    /// with the rule named.
    #[test]
    fn a_sentinel_inside_the_relaxation_rings_is_refused() {
        let path = tiny_grid_nc(
            "torn-cell",
            Some([5, 6, 3, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        );
        let err = probe_grid(&path).unwrap_err().to_string();
        assert!(err.contains("only outermost regional cells"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    /// A one-sided edge below mask 7 is refused with the rule named.
    #[test]
    fn a_one_sided_edge_inside_the_rings_is_refused() {
        let path = tiny_grid_nc(
            "torn-edge",
            Some([5, 6, 7, 7]),
            [5, 6, 2, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        );
        let err = probe_grid(&path).unwrap_err().to_string();
        assert!(err.contains("one-sided"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    /// A mask value past the seven rings is refused with its location.
    #[test]
    fn a_mask_outside_the_seven_rings_is_refused() {
        let path = tiny_grid_nc(
            "bad-mask",
            Some([5, 9, 7, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 1], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        );
        let err = probe_grid(&path).unwrap_err().to_string();
        assert!(err.contains("outside 0..=7"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    /// A zero in verticesOnCell's VALID slots can never be a boundary: the
    /// cull keeps every vertex of a kept cell.
    #[test]
    fn a_missing_own_vertex_is_refused_on_a_regional_grid() {
        let path = tiny_grid_nc(
            "torn-vertex",
            Some([5, 6, 7, 7]),
            [5, 6, 7, 7],
            [0, 0, 7, 7],
            [[2, 3, 4], [1, 3, 4], [1, 2, 0], [1, 2, 0]],
            [[1, 2], [2, 3], [3, 0], [4, 0]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 1, 2]],
            [
                [1, 2, 0, 0, 0, 0],
                [2, 3, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0],
            ],
        );
        let err = probe_grid(&path).unwrap_err().to_string();
        assert!(
            err.contains("verticesOnCell") && err.contains("outside 1..="),
            "{err}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The far-pixel reject sphere is strictly conservative: every point it
    /// rejects is a point the kd search would have rejected on distance.
    #[test]
    fn the_far_reject_never_drops_a_pixel_the_search_would_keep() {
        let mut mesh = synthetic_mesh(64);
        // Confine the cells to a small window, the shape the guard exists
        // for: a regional cloud on one side of the sphere. (The band the
        // synthetic mesh spreads over is longitude-symmetric, which puts the
        // centroid at the origin and correctly leaves the guard inert -- and
        // this test measuring nothing.)
        for c in 0..mesh.n_cells {
            let lat = 0.3 * (c as f64 / mesh.n_cells as f64);
            let lon = 0.3 * ((c * 7 % 64) as f64 / 64.0);
            let p = xyz(mesh.sphere_radius, lat, lon);
            mesh.x_cell[c] = p[0];
            mesh.y_cell[c] = p[1];
            mesh.z_cell[c] = p[2];
        }
        let tree = KdTree::new(&mesh);
        // A 200 km search radius: what a coarse mesh's own reach looks like,
        // and small beside the sphere so far points exist.
        let max_d2 = (200_000.0f64).powi(2);
        let far = FarReject::new(&mesh, max_d2);
        let mut rejected = 0usize;
        for i in 0..2000 {
            let lat = -1.5 + 3.0 * (i as f64 * 0.618_033_988_749_895).fract();
            let lon = std::f64::consts::TAU * (i as f64 * 0.414_213_562_373_1).fract();
            let p = xyz(mesh.sphere_radius, lat, lon);
            if far.certainly_out(p) {
                rejected += 1;
                let (_, d2) = tree.nearest(p);
                assert!(
                    d2 > max_d2,
                    "far reject dropped a pixel at ({lat:.3},{lon:.3}) that the \
                     search keeps: d2={d2} <= {max_d2}"
                );
            }
        }
        // The band mesh leaves the poles far away, so the reject sphere must
        // actually fire or this test is measuring nothing.
        assert!(rejected > 0, "no probe point was ever far; the guard is untested");
    }

    /// The mpas_in_cell port, on real spherical geometry: a triangle cell
    /// around the north pole. The edge through two vertices at colatitude
    /// 0.1 rad with 120 degrees of longitude between them passes closest to
    /// the pole at colatitude atan(tan(0.1)*cos(60 deg)) ~ 0.0501 rad.
    #[test]
    fn point_in_cell_matches_the_voronoi_edges_of_a_polar_triangle() {
        let r = MPAS_EARTH_RADIUS_M;
        let mut mesh = synthetic_mesh(4);
        let colat = 0.1f64;
        mesh.x_cell[0] = 0.0;
        mesh.y_cell[0] = 0.0;
        mesh.z_cell[0] = r;
        mesh.n_edges_on_cell[0] = 3;
        for (k, lon_deg) in [0.0f64, 120.0, 240.0].iter().enumerate() {
            let lon = lon_deg.to_radians();
            mesh.x_vertex[k] = r * colat.sin() * lon.cos();
            mesh.y_vertex[k] = r * colat.sin() * lon.sin();
            mesh.z_vertex[k] = r * colat.cos();
            mesh.vertices_on_cell[k] = k; // slots 0..3 of cell 0
        }
        let at = |colat: f64, lon_deg: f64| -> [f64; 3] {
            let lon = lon_deg.to_radians();
            [
                r * colat.sin() * lon.cos(),
                r * colat.sin() * lon.sin(),
                r * colat.cos(),
            ]
        };
        assert!(point_in_cell(&mesh, 0, at(0.0, 0.0)), "the generating point itself");
        assert!(point_in_cell(&mesh, 0, at(0.03, 77.0)), "deep inside");
        assert!(point_in_cell(&mesh, 0, at(0.049, 60.0)), "just inside the far edge");
        assert!(!point_in_cell(&mesh, 0, at(0.08, 60.0)), "just past the far edge");
        assert!(!point_in_cell(&mesh, 0, at(0.3, 200.0)), "far outside");
    }

    // -- a synthetic case, built from bytes this module owns -----------------

    const NX_TILES: i64 = 6;
    const NY_TILES: i64 = 3;
    const TILE: i64 = 30;

    /// A multi-tile WPS_GEOG tree covering the whole globe at 2 degrees.
    ///
    /// Every value varies with its own global coordinate, so a destination
    /// cell accumulates a different number from every tile that reaches it.
    /// That is what makes the order tiles are visited in observable at all.
    fn geog_fixture(label: &str, kind: &str, nz: usize, wordsize: usize) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rw-mpas-static-{}-{}",
            std::process::id(),
            label
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("fixture directory");
        let mut index = format!(
            "type={kind}\nprojection=regular_ll\ndx=2.0\ndy=2.0\n\
             known_x=1.0\nknown_y=1.0\nknown_lat=-89.0\nknown_lon=-179.0\n\
             wordsize={wordsize}\ntile_x={TILE}\ntile_y={TILE}\ntile_z={nz}\n\
             tile_bdr=0\nendian=big\nsigned=no\nscale_factor=1.0\n"
        );
        if kind == "categorical" {
            index.push_str("category_min=1\ncategory_max=8\niswater=3\nmminlu=FIXTURE\n");
        }
        std::fs::write(dir.join("index"), &index).expect("fixture index");
        for ty in 0..NY_TILES {
            for tx in 0..NX_TILES {
                let xs = tx * TILE + 1;
                let ys = ty * TILE + 1;
                let mut bytes = Vec::with_capacity(nz * (TILE * TILE) as usize * wordsize);
                for z in 0..nz as i64 {
                    for j in 0..TILE {
                        for i in 0..TILE {
                            let x = xs + i;
                            let y = ys + j;
                            let v = if kind == "categorical" {
                                (x * 7 + y * 13).rem_euclid(8) + 1
                            } else {
                                (x * 31 + y * 17 + z * 101).rem_euclid(4096)
                            };
                            match wordsize {
                                1 => bytes.push(v as u8),
                                _ => bytes.extend_from_slice(&(v as u16).to_be_bytes()),
                            }
                        }
                    }
                }
                let name = format!("{xs:05}-{:05}.{ys:05}-{:05}", xs + TILE - 1, ys + TILE - 1);
                std::fs::write(dir.join(name), &bytes).expect("fixture tile");
            }
        }
        dir
    }

    /// A mesh whose cells cover a latitude band, not the whole sphere.
    ///
    /// Deliberately a band: the tile mapper refuses a source pixel further
    /// than the search radius from any cell, so a global geography source
    /// over a band mesh produces destination maps with refused entries in
    /// them.  A mesh covering the sphere would leave that branch of the
    /// accumulate loop untested by the determinism gate.
    ///
    /// Only the geometry the geography passes read has to be physical: the
    /// topology is filled with in-range indices so the writer can serialize
    /// it, because what is under test is which pixels reach which cell.
    fn synthetic_mesh(n_cells: usize) -> Mesh {
        let r = MPAS_EARTH_RADIUS_M;
        let max_edges = 3usize;
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let mut lat_cell = Vec::with_capacity(n_cells);
        let mut lon_cell = Vec::with_capacity(n_cells);
        for c in 0..n_cells {
            let z = 0.5 - (c as f64 + 0.5) / n_cells as f64;
            lat_cell.push(z.asin());
            lon_cell.push((golden * c as f64).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI);
        }
        let xyz_of = |lat: &[f64], lon: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let mut x = Vec::with_capacity(lat.len());
            let mut y = Vec::with_capacity(lat.len());
            let mut z = Vec::with_capacity(lat.len());
            for (la, lo) in lat.iter().zip(lon) {
                let p = xyz(r, *la, *lo);
                x.push(p[0]);
                y.push(p[1]);
                z.push(p[2]);
            }
            (x, y, z)
        };
        // Vertices sit a fixed arc from their cell, which sets the search
        // radius the tile mapper refuses beyond.  Small enough that a polar
        // source pixel is outside every cell's reach.
        let lat_vertex: Vec<f64> = lat_cell
            .iter()
            .map(|v| (v + 0.2).clamp(-1.5, 1.5))
            .collect();
        let lon_vertex = lon_cell.clone();
        let (x_cell, y_cell, z_cell) = xyz_of(&lat_cell, &lon_cell);
        let (x_vertex, y_vertex, z_vertex) = xyz_of(&lat_vertex, &lon_vertex);
        let n_edges = n_cells;
        let n_vertices = n_cells;
        let max_edges2 = 2 * max_edges;
        let vertex_degree = 3usize;
        let cells_on_cell: Vec<usize> = (0..n_cells * max_edges).map(|k| k % n_cells).collect();
        let edges_on_cell: Vec<usize> = (0..n_cells * max_edges).map(|k| k % n_edges).collect();
        let vertices_on_cell: Vec<usize> =
            (0..n_cells * max_edges).map(|k| k / max_edges).collect();
        let one_based = |v: &[usize]| -> Vec<i32> { v.iter().map(|&k| k as i32 + 1).collect() };
        Mesh {
            n_cells,
            n_edges,
            n_vertices,
            max_edges,
            max_edges2,
            vertex_degree,
            raw_cells_on_cell: one_based(&cells_on_cell),
            raw_edges_on_cell: one_based(&edges_on_cell),
            raw_vertices_on_cell: one_based(&vertices_on_cell),
            raw_cells_on_edge: (0..n_edges)
                .flat_map(|e| [(e % n_cells) as i32 + 1, ((e + 1) % n_cells) as i32 + 1])
                .collect(),
            raw_vertices_on_edge: (0..n_edges)
                .flat_map(|e| [(e % n_vertices) as i32 + 1, ((e + 1) % n_vertices) as i32 + 1])
                .collect(),
            raw_edges_on_edge: (0..n_edges * max_edges2)
                .map(|k| (k % n_edges) as i32 + 1)
                .collect(),
            raw_cells_on_vertex: (0..n_vertices * vertex_degree)
                .map(|k| (k % n_cells) as i32 + 1)
                .collect(),
            raw_edges_on_vertex: (0..n_vertices * vertex_degree)
                .map(|k| (k % n_edges) as i32 + 1)
                .collect(),
            n_edges_on_edge: vec![max_edges2 as i32; n_edges],
            weights_on_edge: vec![0.0; n_edges * max_edges2],
            kite_areas_on_vertex: vec![1.0; n_vertices * vertex_degree],
            index_to_cell_id: (1..=n_cells as i32).collect(),
            index_to_edge_id: (1..=n_edges as i32).collect(),
            index_to_vertex_id: (1..=n_vertices as i32).collect(),
            bdy_mask_edge: vec![0; n_edges],
            bdy_mask_vertex: vec![0; n_vertices],
            scaled_from_unit_sphere: false,
            sphere_radius: r,
            lat_edge: lat_cell.clone(),
            lon_edge: lon_cell.clone(),
            x_edge: x_cell.clone(),
            y_edge: y_cell.clone(),
            z_edge: z_cell.clone(),
            lat_vertex,
            lon_vertex,
            x_vertex,
            y_vertex,
            z_vertex,
            dc_edge: vec![1.0; n_edges],
            dv_edge: vec![1.0; n_edges],
            area_cell: vec![1.0; n_cells],
            area_triangle: vec![1.0; n_vertices],
            angle_edge: vec![0.0; n_edges],
            mesh_density: vec![1.0; n_cells],
            nominal_min_dc: 1.0,
            n_edges_on_cell: vec![max_edges; n_cells],
            cells_on_cell,
            edges_on_cell,
            vertices_on_cell,
            cells_on_edge: (0..n_edges).map(|e| [e % n_cells, (e + 1) % n_cells]).collect(),
            vertices_on_edge: (0..n_edges)
                .map(|e| [e % n_vertices, (e + 1) % n_vertices])
                .collect(),
            bdy_mask_cell: vec![0; n_cells],
            lat_cell,
            lon_cell,
            x_cell,
            y_cell,
            z_cell,
        }
    }

    /// A drag band of the right shape and no content, for the writer tests.
    ///
    /// The sub-grid orography has its own case in `static_gwd`; what the
    /// writer tests measure is byte-identity of the file it lays out, and a
    /// constant band keeps that measurement about the writer.
    fn flat_gwd(mesh: &Mesh) -> crate::static_gwd::GwdFields {
        crate::static_gwd::GwdFields {
            var2d: vec![0.0; mesh.n_cells],
            con: vec![0.0; mesh.n_cells],
            oa: std::array::from_fn(|_| vec![0.0; mesh.n_cells]),
            ol: std::array::from_fn(|_| vec![0.0; mesh.n_cells]),
            hlanduse: vec![1; mesh.n_cells],
            smoothed_cells: 0,
            smooth_skipped_boundary_cells: 0,
            water_category: 3,
            water_category_source: "fixture".to_string(),
            band_rows: 0,
            bands: 0,
            band_bytes: 0,
        }
    }

    fn fixture_geography(
        ter: Vec<f32>,
        landmask: Vec<i32>,
        ivgtyp: Vec<i32>,
        isltyp: Vec<i32>,
        greenfrac: Vec<f32>,
        albedo12m: Vec<f32>,
        snoalb: Vec<f32>,
        soiltemp: Vec<f32>,
        soilcomp: Vec<f32>,
        soilcl: [Vec<i32>; 4],
        iswater_lu: i32,
    ) -> GeographyFields {
        GeographyFields {
            ter,
            landmask,
            ivgtyp,
            isltyp,
            greenfrac,
            albedo12m,
            snoalb,
            soiltemp,
            soilcomp,
            soilcl,
            isice_lu: 15,
            iswater_lu,
            mminlu: "FIXTURE".to_string(),
        }
    }

    fn zero_operators(mesh: &Mesh) -> OperatorFields {
        let per_cell = mesh.n_cells * mesh.max_edges;
        OperatorFields {
            cell_gradient_coef_x: vec![0.0; per_cell],
            cell_gradient_coef_y: vec![0.0; per_cell],
            defc_a: vec![0.0; per_cell],
            defc_b: vec![0.0; per_cell],
            deriv_two: vec![0.0; mesh.n_edges * 2 * FIFTEEN],
            deriv_two_zero_stencil_cells: 0,
        }
    }

    struct SyntheticCase {
        cont1: PathBuf,
        cont12: PathBuf,
        cat: PathBuf,
        comp: PathBuf,
    }

    impl SyntheticCase {
        fn build(tag: &str) -> Self {
            Self {
                cont1: geog_fixture(&format!("{tag}-cont1"), "continuous", 1, 2),
                cont12: geog_fixture(&format!("{tag}-cont12"), "continuous", 12, 2),
                cat: geog_fixture(&format!("{tag}-cat"), "categorical", 1, 1),
                comp: geog_fixture(&format!("{tag}-comp"), "continuous", 4, 2),
            }
        }

        fn config(&self, out: PathBuf, workers: usize) -> StaticBuildConfig {
            StaticBuildConfig {
                grid_path: PathBuf::from("<synthetic>"),
                out_path: out,
                geog: StaticGeogPaths {
                    terrain: self.cont1.clone(),
                    landuse: self.cat.clone(),
                    soilcat: self.cat.clone(),
                    greenfrac: self.cont12.clone(),
                    albedo: self.cont12.clone(),
                    snow_albedo: self.cont1.clone(),
                    soil_temperature: self.cont1.clone(),
                    soilcomp: self.comp.clone(),
                    soilcl1: self.cat.clone(),
                    soilcl2: self.cat.clone(),
                    soilcl3: self.cat.clone(),
                    soilcl4: self.cat.clone(),
                },
                supersample: 2,
                supersample_landuse: 1,
                supersample_30s: 1,
                host_memory_limit_bytes: None,
                tile_workers: Some(workers),
                receipt_path: None,
                provenance: "synthetic determinism case".to_string(),
                valid_time: DEFAULT_VALID_TIME.to_string(),
                nominal_dx_m: None,
                clobber: true,
                coordinates: CoordinateRepresentation::default(),
            }
        }
    }

    /// Run every geography pass and write the static, at a chosen worker
    /// count and cache budget.  Returns the output path and the cache tally.
    fn run_case(
        case: &SyntheticCase,
        mesh: &Mesh,
        workers: usize,
        cache_bytes: u64,
        tag: &str,
    ) -> (PathBuf, TileMapCacheReceipt) {
        let out = std::env::temp_dir().join(format!(
            "rw-mpas-static-{}-{tag}.nc",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&out);
        let cfg = case.config(out.clone(), workers);
        let tree = KdTree::new(mesh);
        let max_d2 = mesh.max_search_distance2();
        let cache = TileMapCache::new(cache_bytes);
        let pool = if workers > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .expect("pool"),
            )
        } else {
            None
        };
        let ctx = GeogContext {
            mesh,
            tree: &tree,
            max_d2,
            workers,
            pool: pool.as_ref(),
            cache: &cache,
        };

        let (ter, _) = continuous_mean(&ctx, &cfg.geog.terrain, cfg.supersample, None, false)
            .expect("terrain");
        let (ivgtyp, _, _, lu) =
            categorical(&ctx, &cfg.geog.landuse, cfg.supersample_landuse).expect("landuse");
        let iswater = lu.index.iswater.unwrap_or(17) as i32;
        let landmask: Vec<i32> = ivgtyp
            .iter()
            .map(|&v| if v == iswater { 0 } else { 1 })
            .collect();
        let (isltyp, _, _, _) =
            categorical(&ctx, &cfg.geog.soilcat, cfg.supersample).expect("soilcat");
        let (greenfrac, _) = continuous_mean(
            &ctx,
            &cfg.geog.greenfrac,
            cfg.supersample_30s,
            Some(&landmask),
            true,
        )
        .expect("greenfrac");
        let (albedo, _) = continuous_mean(
            &ctx,
            &cfg.geog.albedo,
            cfg.supersample_30s,
            Some(&landmask),
            true,
        )
        .expect("albedo");
        let (snoalb, _) = snow_albedo(
            &ctx,
            &cfg.geog.snow_albedo,
            cfg.supersample_30s,
            &landmask,
        )
        .expect("snoalb");
        let (soiltemp, _) =
            soil_temperature(mesh, &cfg.geog.soil_temperature, &landmask).expect("soiltemp");
        let (comp, _) = soilcomp(&ctx, &cfg.geog.soilcomp, cfg.supersample_30s, &landmask)
            .expect("soilcomp");
        let mut cls: Vec<Vec<i32>> = Vec::new();
        for path in [
            &cfg.geog.soilcl1,
            &cfg.geog.soilcl2,
            &cfg.geog.soilcl3,
            &cfg.geog.soilcl4,
        ] {
            let (v, _, _, _) = categorical(&ctx, path, cfg.supersample_30s).expect("soilcl");
            cls.push(v);
        }
        let receipt = cache.receipt().expect("cache receipt");

        let geo = fixture_geography(
            ter,
            landmask,
            ivgtyp,
            isltyp,
            greenfrac,
            albedo,
            snoalb,
            soiltemp,
            comp,
            [cls[0].clone(), cls[1].clone(), cls[2].clone(), cls[3].clone()],
            iswater,
        );
        write_static(
            &cfg,
            mesh,
            &zero_operators(mesh),
            &geo,
            &flat_gwd(mesh),
            1.0,
            "{\"fixture\":true}",
        )
        .expect("static writes");
        (out, receipt)
    }

    #[test]
    fn a_parallel_build_is_byte_identical_to_a_serial_one() {
        // THE GATE. Splitting the tile loop across workers changes the order
        // contributions reach a destination cell.  If any accumulator were
        // floating point, that reordering would move low bits and two runs of
        // the same build would disagree; a static that differs run to run
        // then breaks every downstream identity check that compares one.
        // Every accumulator here is an integer for exactly that reason, and
        // this is what holds it true.
        let case = SyntheticCase::build("gate");
        let mesh = synthetic_mesh(120);
        let (serial, _) = run_case(&case, &mesh, 1, 0, "serial");
        let (parallel, _) = run_case(&case, &mesh, 8, 0, "parallel");
        let a = sha256_file(&serial).expect("serial digest");
        let b = sha256_file(&parallel).expect("parallel digest");
        assert_eq!(
            a, b,
            "serial {} and 8-worker {} statics differ",
            serial.display(),
            parallel.display()
        );
        assert!(
            std::fs::metadata(&serial).expect("serial static").len() > 4096,
            "the comparison must be of a real static, not an empty file"
        );
    }

    #[test]
    fn the_gate_case_really_does_map_some_pixels_to_nothing() {
        // The gate is only a gate over the branches it reaches.  A band mesh
        // under a global source puts refused entries in the middle of a tile
        // map, which is the case the run-folding accumulator has to carry
        // without merging across the gap.
        let case = SyntheticCase::build("refused");
        let mesh = synthetic_mesh(120);
        let ds = GeogDataset::open(&case.cont1).expect("fixture opens");
        let tree = KdTree::new(&mesh);
        let cache = TileMapCache::new(0);
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2: mesh.max_search_distance2(),
            workers: 1,
            pool: None,
            cache: &cache,
        };
        let mut refused = 0usize;
        let mut mapped = 0usize;
        for tile in &ds.tiles {
            for c in ctx.tile_map(&ds, tile, 1).expect("tile map") {
                if c < 0 {
                    refused += 1;
                } else {
                    mapped += 1;
                }
            }
        }
        assert!(refused > 0, "no pixel was out of reach: {refused}/{mapped}");
        assert!(mapped > 0, "no pixel reached a cell at all");
    }

    #[test]
    fn the_byte_comparison_catches_a_single_moved_value() {
        // A gate never shown to fail is not evidence.  Move one destination
        // cell by one ulp and the digests must part.
        let case = SyntheticCase::build("perturb");
        let mesh = synthetic_mesh(120);
        let (baseline, _) = run_case(&case, &mesh, 1, 0, "perturb-base");
        let baseline_digest = sha256_file(&baseline).expect("baseline digest");

        let out = std::env::temp_dir()
            .join(format!("rw-mpas-static-{}-perturbed.nc", std::process::id()));
        let _ = std::fs::remove_file(&out);
        let cfg = case.config(out.clone(), 1);
        let tree = KdTree::new(&mesh);
        let cache = TileMapCache::new(0);
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2: mesh.max_search_distance2(),
            workers: 1,
            pool: None,
            cache: &cache,
        };
        let (mut ter, _) =
            continuous_mean(&ctx, &cfg.geog.terrain, cfg.supersample, None, false).expect("ter");
        ter[7] = f32::from_bits(ter[7].to_bits() ^ 1);
        let (ivgtyp, _, _, lu) =
            categorical(&ctx, &cfg.geog.landuse, cfg.supersample_landuse).expect("landuse");
        let iswater = lu.index.iswater.unwrap_or(17) as i32;
        let landmask: Vec<i32> = ivgtyp
            .iter()
            .map(|&v| if v == iswater { 0 } else { 1 })
            .collect();
        let (isltyp, _, _, _) =
            categorical(&ctx, &cfg.geog.soilcat, cfg.supersample).expect("soilcat");
        let (greenfrac, _) = continuous_mean(
            &ctx,
            &cfg.geog.greenfrac,
            cfg.supersample_30s,
            Some(&landmask),
            true,
        )
        .expect("greenfrac");
        let (snoalb, _) =
            snow_albedo(&ctx, &cfg.geog.snow_albedo, cfg.supersample_30s, &landmask)
                .expect("snoalb");
        let (soiltemp, _) =
            soil_temperature(&mesh, &cfg.geog.soil_temperature, &landmask).expect("soiltemp");
        let (comp, _) = soilcomp(&ctx, &cfg.geog.soilcomp, cfg.supersample_30s, &landmask)
            .expect("soilcomp");
        let (cl, _, _, _) =
            categorical(&ctx, &cfg.geog.soilcl1, cfg.supersample_30s).expect("soilcl");
        let geo = fixture_geography(
            ter,
            landmask,
            ivgtyp,
            isltyp,
            greenfrac.clone(),
            greenfrac,
            snoalb,
            soiltemp,
            comp,
            [cl.clone(), cl.clone(), cl.clone(), cl],
            iswater,
        );
        write_static(
            &cfg,
            &mesh,
            &zero_operators(&mesh),
            &geo,
            &flat_gwd(&mesh),
            1.0,
            "{\"fixture\":true}",
        )
        .expect("perturbed static writes");
        assert_ne!(
            baseline_digest,
            sha256_file(&out).expect("perturbed digest"),
            "a one-ulp move slipped past the digest, so the gate proves nothing"
        );
    }

    #[test]
    fn the_reordering_would_be_visible_if_the_accumulator_were_floating_point() {
        // Why integers, stated as a measurement rather than an assurance.
        // These are contributions of the size the terrain pass accumulates.
        // Summed forwards and backwards as f32 they disagree; summed as the
        // i64 the pass actually uses they cannot.
        let contributions: Vec<i64> = (0..20_000)
            .map(|k: i64| (k * 31 + 17).rem_euclid(4096) + 1_000_000)
            .collect();
        let forward = contributions.iter().fold(0.0f32, |a, &b| a + b as f32);
        let backward = contributions.iter().rev().fold(0.0f32, |a, &b| a + b as f32);
        assert_ne!(
            forward.to_bits(),
            backward.to_bits(),
            "the fixture cannot demonstrate order sensitivity at all"
        );
        let f_int: i64 = contributions.iter().sum();
        let b_int: i64 = contributions.iter().rev().sum();
        assert_eq!(f_int, b_int);
    }

    #[test]
    fn the_reusable_tile_map_changes_the_cost_and_not_the_bytes() {
        // Datasets that share tile geometry reuse one map.  A cached map and
        // a recomputed one are the same bytes -- the map is a pure function
        // of the mesh and the geometry -- so the cache may only ever change
        // how long a build takes.
        let case = SyntheticCase::build("cache");
        let mesh = synthetic_mesh(120);
        let (uncached, cold) = run_case(&case, &mesh, 4, 0, "uncached");
        let (cached, warm) = run_case(&case, &mesh, 4, 64 * 1024 * 1024, "cached");
        assert_eq!(cold.hits, 0, "a zero budget cannot serve a hit");
        assert!(
            warm.hits > 0,
            "the shared-geometry datasets never reused a map: {warm:?}"
        );
        assert!(warm.stored_maps > 0 && warm.stored_bytes > 0, "{warm:?}");
        assert_eq!(
            sha256_file(&uncached).expect("uncached digest"),
            sha256_file(&cached).expect("cached digest")
        );
    }

    #[test]
    fn a_budget_too_small_to_hold_a_map_still_builds_the_same_static() {
        // Running the budget out must degrade to recomputation, not to a
        // partial map or a refusal.
        let case = SyntheticCase::build("tight");
        let mesh = synthetic_mesh(120);
        let (roomy, _) = run_case(&case, &mesh, 4, 64 * 1024 * 1024, "roomy");
        let (tight, tally) = run_case(&case, &mesh, 4, 512, "tight");
        assert!(tally.declined_for_budget > 0, "{tally:?}");
        assert_eq!(
            sha256_file(&roomy).expect("roomy digest"),
            sha256_file(&tight).expect("tight digest")
        );
    }

    #[test]
    fn a_run_length_encoded_map_decodes_to_what_it_encoded() {
        let map: Vec<i32> = vec![-1, -1, 4, 4, 4, 4, 7, 7, 7, 2];
        let encoded = CachedTileMap::encode(&map).expect("encodes");
        assert!(matches!(encoded, CachedTileMap::Runs(_)));
        assert_eq!(encoded.decode(map.len()).expect("decodes"), map);
    }

    #[test]
    fn a_map_that_run_length_encoding_would_grow_is_kept_raw() {
        // A mesh finer than the source has no runs to fold.  Storing the
        // runs anyway would cost twice the raw map, so it is not done.
        let map: Vec<i32> = (0..1000).collect();
        let encoded = CachedTileMap::encode(&map).expect("encodes");
        assert!(matches!(encoded, CachedTileMap::Raw(_)), "{encoded:?}");
        assert_eq!(encoded.bytes(), 4000);
        assert_eq!(encoded.decode(map.len()).expect("decodes"), map);
    }

    #[test]
    fn the_worker_request_comes_from_the_flag_then_the_environment() {
        assert_eq!(resolve_tile_workers(Some(3)), 3);
        assert_eq!(resolve_tile_workers(Some(0)), 1, "zero cannot stall a build");
        assert!(resolve_tile_workers(None) >= 1);
    }

    // -- real WPS_GEOG measurement, on demand ------------------------------
    //
    // Ignored by default: these need a real mesh and a real WPS_GEOG tree,
    // which no CI machine is promised to have.  Run one arm at a time so an
    // external peak-working-set measurement belongs to that arm:
    //
    //   RW_MPAS_REAL_GRID=... RW_MPAS_REAL_TERRAIN=... \
    //   RW_MPAS_REAL_WORKERS=1 RW_MPAS_REAL_OUT=ter-serial.bin \
    //   cargo test -p rw-mpas --release -- --ignored --nocapture real_inputs

    fn real_path(key: &str) -> PathBuf {
        PathBuf::from(std::env::var(key).unwrap_or_else(|_| {
            panic!(
                "{key} must name a real input for this measurement; it is \
                 ignored by default because no machine is promised to have one"
            )
        }))
    }

    #[test]
    #[ignore = "needs a real mesh and a real WPS_GEOG tree"]
    fn real_inputs_terrain_pass() {
        let grid = real_path("RW_MPAS_REAL_GRID");
        let terrain = real_path("RW_MPAS_REAL_TERRAIN");
        let workers = std::env::var("RW_MPAS_REAL_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| resolve_tile_workers(None));

        let read_started = Instant::now();
        let mesh = Mesh::read(&grid).expect("mesh reads");
        let tree = KdTree::new(&mesh);
        let max_d2 = mesh.max_search_distance2();
        let mesh_seconds = read_started.elapsed().as_secs_f64();

        let cache = TileMapCache::new(0);
        let pool = if workers > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .expect("pool"),
            )
        } else {
            None
        };
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2,
            workers,
            pool: pool.as_ref(),
            cache: &cache,
        };
        let started = Instant::now();
        let (ter, ds) =
            continuous_mean(&ctx, &terrain, 1, None, false).expect("terrain pass");
        let seconds = started.elapsed().as_secs_f64();

        let out = PathBuf::from(
            std::env::var("RW_MPAS_REAL_OUT")
                .unwrap_or_else(|_| format!("ter-{workers}-workers.bin")),
        );
        let mut bytes = Vec::with_capacity(ter.len() * 4);
        for v in &ter {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&out, &bytes).expect("terrain bytes write");
        println!(
            "REAL-TERRAIN workers={workers} cells={} tiles={} mesh_seconds={mesh_seconds:.2} \
             pass_seconds={seconds:.2} out={} sha256={}",
            mesh.n_cells,
            ds.tiles.len(),
            out.display(),
            sha256_file(&out).expect("digest")
        );
    }

    #[test]
    #[ignore = "needs a real mesh and a real WPS_GEOG tree"]
    fn real_inputs_shared_geometry_reuses_maps() {
        // Two 30-arcsec products on the same grid.  The second pass pays for
        // reading and accumulating, and not for the kd search again.
        let grid = real_path("RW_MPAS_REAL_GRID");
        let first = real_path("RW_MPAS_REAL_CAT_A");
        let second = real_path("RW_MPAS_REAL_CAT_B");
        let workers = std::env::var("RW_MPAS_REAL_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| resolve_tile_workers(None));
        let budget: u64 = std::env::var("RW_MPAS_REAL_CACHE_BYTES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(4 * 1024 * 1024 * 1024);

        let mesh = Mesh::read(&grid).expect("mesh reads");
        let tree = KdTree::new(&mesh);
        let max_d2 = mesh.max_search_distance2();
        let cache = TileMapCache::new(budget);
        let pool = if workers > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .expect("pool"),
            )
        } else {
            None
        };
        let ctx = GeogContext {
            mesh: &mesh,
            tree: &tree,
            max_d2,
            workers,
            pool: pool.as_ref(),
            cache: &cache,
        };
        let t0 = Instant::now();
        let (a, _, _, _) = categorical(&ctx, &first, 1).expect("first categorical");
        let cold = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let (b, _, _, _) = categorical(&ctx, &second, 1).expect("second categorical");
        let warm = t1.elapsed().as_secs_f64();
        let tally = cache.receipt().expect("cache receipt");
        println!(
            "REAL-SHARED workers={workers} cold_seconds={cold:.2} warm_seconds={warm:.2} \
             hits={} misses={} stored_maps={} stored_bytes={} declined={} \
             a_len={} b_len={}",
            tally.hits,
            tally.misses,
            tally.stored_maps,
            tally.stored_bytes,
            tally.declined_for_budget,
            a.len(),
            b.len()
        );
        assert!(tally.hits > 0, "the second product remapped the same grid");
    }

    fn tiny_dataset(label: &str) -> GeogDataset {
        GeogDataset::open(&dataset(label, MINIMAL_INDEX, &[(TILE_NAME, plane_1234_be())]))
            .expect("fixture dataset opens")
    }

    #[test]
    fn the_full_plane_ceiling_refuses_a_source_past_it() {
        // The direct-interpolation path is reserved for coarse climatologies.
        // A source that would need more than the ceiling has to say so, and
        // has to say what the path is for, rather than quietly allocating.
        let ds = tiny_dataset("ceiling");
        let needed = ds.nx_global as u64 * ds.ny_global as u64 * 8;
        let err = full_plane(&ds, 0, needed - 1).expect_err("one byte short refuses");
        let text = err.to_string();
        assert!(text.contains("past bounded full-plane ceiling"), "{text}");
        assert!(text.contains(&format!("{needed}")), "{text}");
        assert!(text.contains("small climatology grids"), "{text}");
    }

    #[test]
    fn a_source_inside_the_ceiling_assembles_at_its_global_offsets() {
        let ds = tiny_dataset("assemble");
        let needed = ds.nx_global as u64 * ds.ny_global as u64 * 8;
        let plane = full_plane(&ds, 0, needed).expect("exactly the ceiling is admitted");
        assert_eq!(plane.len(), ds.nx_global as usize * ds.ny_global as usize);
        let nx = ds.nx_global as usize;
        // The 2x2 tile starts at source (1,1), which is zero-based (0,0).
        assert_eq!(plane[0], 1.0);
        assert_eq!(plane[1], 2.0);
        assert_eq!(plane[nx], 3.0);
        assert_eq!(plane[nx + 1], 4.0);
        // Everything the tile inventory does not cover stays absent, not zero.
        assert!(plane[2].is_nan());
        assert!(plane[nx * 5].is_nan());
    }

    #[test]
    fn the_ceiling_is_checked_before_any_plane_is_read() {
        // A truncated tile would refuse during read; the ceiling refusal has
        // to come first, or the refusal a user sees names the wrong cause.
        let ds = GeogDataset::open(&dataset(
            "ceiling-first",
            MINIMAL_INDEX,
            &[(TILE_NAME, vec![0, 1])],
        ))
        .expect("fixture dataset opens");
        let err = full_plane(&ds, 0, 8).expect_err("the ceiling refuses");
        assert!(
            err.to_string().contains("past bounded full-plane ceiling"),
            "the truncation was reported instead of the ceiling: {err}"
        );
    }

    #[test]
    fn a_supersample_factor_outside_the_supported_range_is_refused() {
        // The factor squares into the per-tile destination map, so it is the
        // one knob that turns a bounded tile into an unbounded one.
        for bad in [0usize, 17, 1024] {
            let err = validate_supersample(bad).expect_err("outside 1..=16");
            let text = err.to_string();
            assert!(text.contains(&format!("supersample factor {bad}")), "{text}");
            assert!(text.contains("1..=16"), "{text}");
        }
        for good in [1usize, 4, 16] {
            validate_supersample(good).expect("inside 1..=16");
        }
    }
}
